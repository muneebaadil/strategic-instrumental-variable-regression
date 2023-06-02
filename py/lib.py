
# %%
import subprocess
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
# for notebook.

def get_git_revision_hash() -> str:
  return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
# %%
def get_args(cmd):
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--n-cores', default=1, type=int)

  # dataset
  parser.add_argument('--num-repeat', default=10, type=int)
  parser.add_argument('--num-applicants', default=10000, type=int)
  parser.add_argument('--admit-all', action='store_true', help='admit all students, as in Harris et. al')
  parser.add_argument('--applicants-per-round', default=1, type=int, help='used for identical thetas')
  parser.add_argument('--fixed-effort-conversion', action='store_true')
  parser.add_argument('--scaled-duplicates', default=None, choices=['random', 'sequence', None], type=str)
  parser.add_argument('--clip', action='store_true')
  parser.add_argument('--b-bias', type=float, default=1.25)
  parser.add_argument('--save-dataset', action='store_true')

  # multienv
  parser.add_argument('--num-envs', default=1, type=int)
  parser.add_argument('--pref',default='uniform',choices=['uniform', 'geometric'], type=str)
  parser.add_argument('--prob', default=0.5, type=float)
  parser.add_argument('--no-protocol', action='store_true')

  # algorithm
  parser.add_argument('--methods', choices=('ols', '2sls', 'ours'), nargs='+', default='ours')

  # experiment
  parser.add_argument('--test-run', action='store_true')
  parser.add_argument('--experiment-root', type=str, default='experiments')
  parser.add_argument('--experiment-name', type=str)

  # temporary
  parser.add_argument('--generate', default=1, choices=[1,2], type=int)
  parser.add_argument('--stream', action='store_true')
  parser.add_argument('--hack', action='store_true')

  if cmd is None:
    args = parser.parse_args()
  else:
    args = parser.parse_args(cmd.split(' '))
  return args


eqs_data = {}
for i in (1, 0):
  for k in ('y', 'b', 'theta', 'o', 'theta_hat', 'b_mean', 'o_mean'):
    eqs_data[f'grp{i}_{k}'] = []
  # %%
def sample_effort_conversion(EW, n_samples, adv_idx, fixed_effort_conversion):
  assert adv_idx.max() < n_samples
  EWi = np.zeros(shape=(n_samples,2,2))

  for i in range(n_samples):
    EWi[i] = EW.copy()
    if not fixed_effort_conversion:
      noise_mean = [0.5, 0, 0, 0.1]
      noise_cov = [[0.25, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.01]]

      noise = np.random.multivariate_normal(noise_mean, noise_cov).reshape((2,2))
      if i in adv_idx:
        EWi[i] += noise
      else:
        EWi[i] -= noise
  return EWi

def generate_bt(n_samples, sigma_sat, sigma_gpa, args):
  assert n_samples % 2 == 0, f"{n_samples} not divisible by 2"
  half = int(n_samples / 2)
  b = np.zeros([n_samples,2])

  # indices for shuffling
  idx = np.arange(n_samples)
  np.random.shuffle(idx)
  disadv_idx = idx[:half]
  adv_idx = idx[half:]

  mean_sat_disadv = 800
  mean_sat_adv = 1000

  mean_gpa_disadv =  1.8
  mean_gpa_adv = mean_gpa_disadv * args.b_bias

  # disadvantaged students
  b[disadv_idx,0] = np.random.normal(mean_sat_disadv,sigma_sat,b[disadv_idx][:,0].shape) #SAT
  b[disadv_idx,1] = np.random.normal(mean_gpa_disadv,sigma_gpa,b[disadv_idx][:,1].shape) #GPA

  # advantaged students
  b[adv_idx,0] = np.random.normal(mean_sat_adv,sigma_sat,b[adv_idx][:,0].shape) #SAT
  b[adv_idx,1] = np.random.normal(mean_gpa_adv,sigma_gpa,b[adv_idx][:,1].shape) #GPA

  if args.clip:
    b[:,0] = np.clip(b[:,0],400,1600) # clip to 400 to 1600
    b[:,1] = np.clip(b[:,1],0,4) # clip to 0 to 4.0

  # confounding error term g (error on true college GPA)
  # args.o_bias =1
  # g = np.zeros(shape=(n_samples,args.num_envs))
  # g[adv_idx] = np.random.normal((args.o_bias / 2.), scale=0.2, size=(half,args.num_envs) )
  # g[disadv_idx] = np.random.normal(-(args.o_bias / 2.), scale=0.2, size=(half, args.num_envs))
  g = np.ones(args.num_applicants)*0.5 # legacy students shifted up
  g[disadv_idx]=-0.5 # first-gen students shifted down
  g += np.random.normal(1,0.2,size=args.num_applicants) # non-zero-mean
  g = g[:, np.newaxis]

  return b, g, adv_idx, disadv_idx

def compute_xt(EWi, b, theta, pref_vect, args):
  assert EWi.shape[0] == b.shape[0]
  assert b.shape[0] == theta.shape[1]

  n_applicants = b.shape[0]

  x = np.zeros([n_applicants,b.shape[1]])
  for i in range(n_applicants):
    thetas_applicant = theta[:, i, :]
    assert thetas_applicant.shape == (args.num_envs, 2)
    thetai = thetas_applicant.T.dot(pref_vect)
    x[i] = b[i] + np.matmul(EWi[i].dot(EWi[i].T),thetai) # optimal solution

  if args.clip:
    x[:,0] = np.clip(x[:,0],400,1600) # clip to 400 to 1600
    x[:,1] = np.clip(x[:,1],0,4) # clip to 0 to 4.0
  return x

def get_selection(y_hat):
  assert y_hat.ndim == 1
  n_applicants = y_hat.shape[0]
  w = np.zeros_like(y_hat)
  for i, _y_hat in enumerate(y_hat):
    y_hat_peers = y_hat[np.arange(n_applicants) != i] # scores of everyone but the current applicant
    prob = np.mean(y_hat_peers <= _y_hat)
    w[i] = np.random.binomial(n=1, p=prob)
  return w

def generate_thetas(args):
  thetas = [generate_theta(i, args) for i in range(args.num_envs)]
  thetas = np.stack(thetas)
  return thetas


def to_duplicate(sd, np, i):
  if sd is None:
    return False
  else:
    if np==False:
      return True
    else:
      if i==0:
        return True
      else:
        return False

def generate_theta(i, args):
  if args.admit_all: # harris et. al settings.
    theta = np.random.multivariate_normal([1,1],[[10, 0], [0, 2]],args.num_applicants)
    return theta
  else: # selection. in our settings, require theta to be repeating across a batch of students.
    assert args.num_applicants % args.applicants_per_round == 0
    n_rounds = int(args.num_applicants / args.applicants_per_round)

    td = to_duplicate(args.scaled_duplicates, args.no_protocol, i)
    if not td: # random vectors for every round
      theta = np.random.multivariate_normal([1,1],[[10, 0], [0, 2]],n_rounds)

    else: # making sure there exists a scaled duplicate of each theta per round
      theta = np.random.multivariate_normal([1,1],[[10, 0], [0, 2]],int(n_rounds / 2))

      # scaled duplicate of each theta.
      scale = np.random.uniform(low=0, high=2, size=(int(n_rounds/2),))
      scale = np.diag(v=scale)
      theta_scaled = scale.dot(theta)

      if args.scaled_duplicates=='random':
        theta = np.concatenate((theta, theta_scaled))
        np.random.shuffle(theta)
      elif args.scaled_duplicates=='sequence':
        theta_temp = np.zeros((n_rounds,2))
        theta_temp[0::2] = theta
        theta_temp[1::2] = theta_scaled
        theta = theta_temp

    # theta repeating over the rounds.
    theta = np.repeat(theta, repeats=args.applicants_per_round, axis=0)
    assert theta.shape[0] == args.num_applicants
    return theta

def generate_data(num_applicants, admit_all, applicants_per_round, fixed_effort_conversion, args, _theta_star=None):
  theta_star = np.zeros(shape=(args.num_envs, 2))
  if _theta_star is None:
    theta_star[:, 1] = np.random.normal(loc=0.5, scale=0.2, size=(args.num_envs,))
  else:
    theta_star[:, 1] = _theta_star # np.random.normal(loc=_theta_star, scale=0.2, size=(args.num_envs, ))

  sigma_sat = 200
  sigma_gpa = 0.5

  if args.pref == 'uniform':
    pref_vect = np.ones(shape=(args.num_envs,))
    pref_vect = pref_vect / np.sum(pref_vect)
  elif args.pref == 'geometric':
    pref_vect = ((1 - args.prob) ** np.arange(args.num_envs)) * args.prob
    pref_vect = pref_vect / np.sum(pref_vect)
  b, g, adv_idx, disadv_idx = generate_bt(num_applicants, sigma_sat, sigma_gpa, args)

  # assessment rule
  theta = generate_thetas(args)
  assert num_applicants % applicants_per_round == 0
  n_rounds = int(num_applicants / applicants_per_round)

  # effort conversion matrices
  EW = np.array([[10.0,0],[0,1.0]])
  EWi = sample_effort_conversion(EW, num_applicants, adv_idx, fixed_effort_conversion)

  # observable features x
  x = compute_xt(EWi, b, theta, pref_vect, args)

  # true outcomes (college gpa)
  # y = np.clip() # clipped outcomes
  assert x[np.newaxis].shape == (1, args.num_applicants, 2)
  assert theta.shape == (args.num_envs, args.num_applicants, 2)
  assert g.shape == (args.num_applicants, args.num_envs)
  assert theta_star.shape == (args.num_envs, 2)

  y = (x[np.newaxis] * theta_star[:, np.newaxis]).sum(axis=-1) + g.T
  # y = np.matmul(x,theta_star) + g
  if args.clip:
    y = np.clip(y, 0, 4)

  # our setup addition
  # computing admission results.
  y_hat_logits = (x * theta)
  y_hat_logits = y_hat_logits - y_hat_logits.mean(axis=1, keepdims=True)
  y_hat_logits = y_hat_logits / y_hat_logits.std(axis=1, keepdims=True)
  y_hat = y_hat_logits.sum(axis=-1)

  def _get_selection(_y_hat, admit_all, n_rounds):
    if not admit_all:
      _w = np.zeros_like(_y_hat)
      # comparing people coming in the same rounds.
      for r in range(n_rounds):
        _y_hat_r = _y_hat[r * applicants_per_round: (r+1) * applicants_per_round]

        w_r = get_selection(_y_hat_r)
        _w[r*applicants_per_round: (r+1)*applicants_per_round] = w_r
    else:
      _w = np.ones_like(y_hat)
    return _w

  w = np.zeros((args.num_envs, num_applicants))
  for env_idx in range(args.num_envs):
    w[env_idx ] = _get_selection(y_hat[env_idx], admit_all, n_rounds)

  z = np.zeros((args.num_applicants, ))
  for idx in range(args.num_applicants):
    w_idx = w[:, idx]
    if w_idx.sum() == 0: # applicant is not accepted anywhere
      z[idx] = 0
    else:
      pvals = w_idx * pref_vect / (w_idx * pref_vect).sum()
      temp = np.random.multinomial(n=1, pvals=pvals, size=1)
      _idx = temp.flatten().nonzero()[0]
      z[idx] = _idx+1 # offset to avoid conflict with "no uni" decision
  return b,x,y,EW,theta, w, z, y_hat, adv_idx, disadv_idx, g.T, theta_star, pref_vect

# %%
def ols(x,y):
  model = LinearRegression(fit_intercept=True)
  model.fit(x, y)
  theta_hat_ols_ = model.coef_
  return theta_hat_ols_

def tsls(x,y,theta): # runs until round T
  # my implementation
  # regress x onto theta: estimate omega
  model = LinearRegression()
  model.fit(theta, x)
  omega_hat_ = model.coef_.T

  # regress y onto theta: estimate lambda
  model = LinearRegression()
  model.fit(theta, y)
  lambda_hat_ = model.coef_

  # estimate theta^*
  theta_hat_tsls_ = np.matmul(
    np.linalg.inv(omega_hat_), lambda_hat_
  )
  return theta_hat_tsls_
def our2(x, y, theta, w):
  model = LinearRegression()
  model.fit(theta, x)
  omega_hat_ = model.coef_.T # EET estimate.

  # recover scaled duplicate
  theta_admit = theta[w==1]
  x_admit = x[w==1]
  theta_admit_, counts = np.unique(theta_admit, axis=0, return_counts=True)
  # to use thetas having the most number of applicants
  idx = np.argsort(counts); idx = idx[::-1]
  theta_admit_ = theta_admit_[idx]

  # construct linear system of eqs.
  assert theta_admit.shape[1] > 1 # algo not applicable for only one feature.
  n_eqs_required = theta_admit.shape[1]

  curr_n_eqs = 0
  A, b = [], []

  i = 0
  while (i < theta_admit_.shape[0]):
    j = i+1
    while (j < theta_admit_.shape[0]):
      pair = theta_admit_[[i,j]]

      if np.linalg.matrix_rank(pair) == 1: # if scalar multiple, and exact duplicates are ruled out.
        idx_grp1 = np.all(theta_admit == pair[1], axis=-1)
        idx_grp0 = np.all(theta_admit == pair[0], axis=-1)
        est1 = y[idx_grp1].mean()
        est0 = y[idx_grp0].mean()

        A.append(
          (x_admit[idx_grp1].mean(axis=0) - x_admit[idx_grp0].mean(axis=0))[np.newaxis]
        )
        b.append(np.array([est1-est0]))

        curr_n_eqs += 1
      j += 1
    i += 1

  if curr_n_eqs < n_eqs_required:
    out = np.empty(shape=(n_eqs_required,))
    out[:] = np.nan
    return out

  A = np.concatenate(A, axis=0)
  b = np.concatenate(b)

  m = LinearRegression()
  m.fit(A, b)
  theta_star_est = m.coef_
  return theta_star_est
#%%
def run_multi_exp(seed, args, env_idx=None):
  np.random.seed(seed)
  b, x, y, EW, theta, w, z, y_hat, adv_idx, disadv_idx, o, theta_star, pref_vect  = generate_data(
    args.num_applicants, args.admit_all, args.applicants_per_round, args.fixed_effort_conversion, args, _theta_star=0.5
  )

  err_list = {}
  envs_itr = [env_idx] if env_idx is not None else range(args.num_envs)
  for env_idx in envs_itr:
    dictenv = run_single_env(args, x, y, theta, z, theta_star, env_idx)
    for k, v in dictenv.items():
      err_list[f'{k}_env{env_idx}'] = v

  return err_list

def run_single_env(args, x, y, theta, z, theta_star, env_idx):
  y_env = y[env_idx].flatten()
  theta_env = theta[env_idx]
  z_env = z==env_idx+1

  upp_limits = [x for x in range(args.applicants_per_round*2, args.num_applicants+1, args.applicants_per_round)]

  err_list = {m: [None] * len(upp_limits) for m in args.methods}
  for i, t in tqdm(enumerate(upp_limits)):
    x_round = x[:t]
    y_env_round = y_env[:t]
    z_env_round = z_env[:t]
    theta_env_round = theta_env[:t]

    # filtering out rejected students
    y_env_round_selected = y_env_round[z_env_round]
    x_round_selected = x_round[z_env_round]
    theta_env_round_selected = theta_env_round[z_env_round]

    for m in args.methods:
      if m == 'ours':
        est = our2(x_round, y_env_round_selected, theta_env_round, z_env_round)
      elif m == '2sls':
        try:
          est = tsls(x_round_selected, y_env_round_selected, theta_env_round_selected)
        except np.linalg.LinAlgError:
          est = np.array([np.nan, np.nan])
      elif m == 'ols':
        est = ols(x_round_selected, y_env_round_selected)

      assert theta_star[env_idx].shape == est.shape, f"{theta[0].shape}, {est.shape}"
      err_list[m][i] = np.linalg.norm(theta_star[env_idx] - est )
  return err_list

# convert to dataframe.
def runs2df(runs):
  """
  Args:
  runs: List of dictionaries

  Returns:
  Converts to list of dataframes and concatenate those together.
  """
  dfs = []
  for r in runs:
    df = pd.DataFrame(r)
    dfs.append(df)
  df = pd.concat(dfs)
  df.reset_index(inplace=True); df.rename({'index': 'iterations'}, axis=1, inplace=True)
  return df