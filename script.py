
# %%
import subprocess
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import argparse
# for notebook. 

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
# %%
def get_args(cmd):
  parser = argparse.ArgumentParser()

  # dataset
  parser.add_argument('--num-repeat', default=10, type=int)
  parser.add_argument('--num-applicants', default=10000, type=int)
  parser.add_argument('--admit-all', action='store_true', help='admit all students, as in Harris et. al')
  parser.add_argument('--applicants-per-round', default=1, type=int, help='used for identical thetas')
  parser.add_argument('--fixed-effort-conversion', action='store_true')
  parser.add_argument('--scaled-duplicates', default=None, choices=['sequence', None], type=str)
  parser.add_argument('--num-cooperative-envs', default=None, type=int)
  parser.add_argument('--clip', action='store_true')
  parser.add_argument('--normalize', action='store_true')
  parser.add_argument('--b-bias', type=float, default=1.25)
  parser.add_argument('--theta-star-std', type=float, default=0)
  parser.add_argument('--theta-per-env', action='store_true')
  parser.add_argument('--save-dataset', action='store_true')
  parser.add_argument('--rank-type', type=str, default='prediction', choices=('prediction', 'uniform'))
  parser.add_argument('--utility-dataset-std', type=float, default=2)

  # multienv
  parser.add_argument('--num-envs', default=1, type=int)
  parser.add_argument('--pref',default='uniform',choices=['uniform', 'geometric'], type=str)
  parser.add_argument('--prob', default=0.5, type=float)
  parser.add_argument('--no-protocol', action='store_true')
  parser.add_argument('--envs-accept-rates', nargs='+', default=[1.00], type=float)
  parser.add_argument('--pref-vect', nargs='+', default=[1.00], type=float)

  # algorithm
  parser.add_argument('--methods', choices=('ols', '2sls', 'ours'), nargs='+', default='ours')

  # misc
  parser.add_argument('--offline-eval', action='store_true', help='evaluate '
                      'the algorithms only once for the entire dataset together.')
  
  if cmd is None:
    args = parser.parse_args()
  else:
    args = parser.parse_args(cmd.split(' '))
  
  if len(args.envs_accept_rates) == 1:
    args.envs_accept_rates = [args.envs_accept_rates[0]] * args.num_envs
  if len(args.pref_vect) == 1:
    args.pref_vect = [args.pref_vect[0]] * args.num_envs
  args.pref_vect = [p/sum(args.pref_vect) for p in args.pref_vect]
  args.pref_vect = np.array(args.pref_vect)

  if args.num_cooperative_envs is None:
    args.num_cooperative_envs = args.num_envs # by default, everyone is cooperative

  assert len(args.envs_accept_rates) == args.num_envs
  assert len(args.pref_vect) == args.num_envs
  return args


eqs_data = {}
for i in (1, 0):
  for k in ('y', 'b', 'theta', 'o', 'theta_hat', 'b_mean', 'o_mean'):
    eqs_data[f'grp{i}_{k}'] = [] 
# %%
def normalize(arrs, new_min, new_max):
  new_range = new_max - new_min
  curr_min = np.concatenate(arrs).min()
  curr_max = np.concatenate(arrs).max()

  curr_range = curr_max - curr_min

  out = []
  for arr in arrs:
    out.append(
      (((arr - curr_min) / curr_range) * new_range) + new_min
    )
  return out, new_range / curr_range

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
  g = np.ones((args.num_applicants, args.num_envs))*0.5 # legacy students shifted up
  g[disadv_idx, :]=-0.5 # first-gen students shifted down
  g += np.random.normal(1,0.2,size=(args.num_applicants, args.num_envs)) # non-zero-mean
  return b, g, adv_idx, disadv_idx

def compute_xt(EWi, b, theta, pref_vect, args):
  assert EWi.shape[0] == b.shape[0]
  assert b.shape[0] == theta.shape[1]

  assert EWi.shape == (args.num_applicants, 2, 2)
  assert theta.shape == (args.num_envs, args.num_applicants, 2)
  assert pref_vect.shape == (args.num_envs,)

  eet = np.matmul(EWi, np.transpose(EWi, axes=(0, 2, 1))) # (n_appl, 2, 2)
  thetai = np.matmul(np.transpose(theta, axes=(1, -1, 0)), pref_vect) # (n_appl, 2 )
  impr = np.matmul(eet, thetai[:, :, np.newaxis])[:, :, 0] # (n_appl, 2)
  
  x = b + impr
  
  if args.clip:
    x[:,0] = np.clip(x[:,0],400,1600) # clip to 400 to 1600
    x[:,1] = np.clip(x[:,1],0,4) # clip to 0 to 4.0
  return x

def get_selection(y_hat, accept_rate, rank_type):
  if rank_type == 'prediction':
    sort_idx = np.argsort(y_hat )
  
    thres = int((1-accept_rate) * sort_idx.size) 
    rejected_idx = sort_idx[:thres]
    accepted_idx = sort_idx[thres: ]
  
    w_test = np.zeros_like(y_hat)
    w_test[accepted_idx] = True
    w_test[rejected_idx] = False
  
  elif rank_type == 'uniform':
     w_test = np.zeros_like(y_hat)
     idx = np.arange(y_hat.size )
     np.random.shuffle(idx)
     idx = idx[int((1-accept_rate) * y_hat.size) :]
     w_test = np.zeros_like(y_hat )
     w_test[idx ] = True 
  return w_test 

def generate_thetas(args):
  deploy_sd_every = 2
  thetas = []
  for i in range(args.num_envs):
    if i < args.num_cooperative_envs: # cooperative env.
      thetas.append(generate_theta(i, args, 1))
    else: # non-cooperative env.
      thetas.append(generate_theta(i, args, deploy_sd_every))
      deploy_sd_every += 1 
      
  thetas = np.stack(thetas)
  return thetas


def distribute(n_rounds, theta, theta_scaled, deploy_sd_every):
    theta_temp = np.zeros((n_rounds,2 ))
    j = 0
    for i in range(deploy_sd_every, int(n_rounds/2), deploy_sd_every):
        theta_temp [j:j+deploy_sd_every ] = theta[i-deploy_sd_every: i]
        theta_temp[j+deploy_sd_every: j + deploy_sd_every + deploy_sd_every ] = theta_scaled[i - deploy_sd_every: i]
     
        j = j + 2*deploy_sd_every

    # residual from the first vector
    theta_temp[j: j + int(n_rounds/2) - i] = theta[i:] 
    j = j + int(n_rounds/2) - i

    # residual from the second vector 
    theta_temp[j: j + int(n_rounds/2) - i] = theta_scaled [i:]
    return theta_temp 

def generate_theta(i, args, deploy_sd_every ):
  if args.admit_all: # harris et. al settings. 
    theta = np.random.multivariate_normal([1,1+i*args.theta_per_env],[[10, 0], [0, 1]],args.num_applicants)
    return theta 
  else: # selection. in our settings, require theta to be repeating across a batch of students.
    assert args.num_applicants % args.applicants_per_round == 0
    n_rounds = int(args.num_applicants / args.applicants_per_round)

    if args.scaled_duplicates is None: # random vectors for every round
      theta = np.random.multivariate_normal([1,1+i*args.theta_per_env],[[10, 0], [0, 1]],n_rounds)

    elif args.scaled_duplicates == 'sequence': # making sure there exists a scaled duplicate of each theta per round
      theta = np.random.multivariate_normal([1,1+i*args.theta_per_env],[[10, 0], [0, 1]],int(n_rounds / 2))
      assert theta.shape == (int(n_rounds/2), 2)

      # scaled duplicate of each theta. 
      scale = np.random.uniform(low=0, high=2, size=(int(n_rounds/2),))
      scale = np.diag(v=scale)
      theta_scaled = scale.dot(theta)
  
      theta = distribute(n_rounds, theta, theta_scaled, deploy_sd_every)

    # theta repeating over the rounds.
    theta = np.repeat(theta, repeats=args.applicants_per_round, axis=0)
    assert theta.shape[0] == args.num_applicants
    return theta

def generate_data(num_applicants, admit_all, applicants_per_round, fixed_effort_conversion, args):
  theta_star = np.zeros(shape=(args.num_envs, 2))
  theta_star[:, 1] = np.random.normal(loc=0.5, scale=args.theta_star_std, size=(args.num_envs,))

  sigma_sat = 200
  sigma_gpa = 0.5

  b, g, adv_idx, disadv_idx = generate_bt(num_applicants, sigma_sat, sigma_gpa, args)

  # assessment rule 
  theta = generate_thetas(args)
  assert num_applicants % applicants_per_round == 0
  n_rounds = int(num_applicants / applicants_per_round)
  
  # effort conversion matrices
  EW = np.array([[10.0,0],[0,1.0]])
  EWi = sample_effort_conversion(EW, num_applicants, adv_idx, fixed_effort_conversion)

  # observable features x
  x = compute_xt(EWi, b, theta, args.pref_vect, args)

  EET = EW.dot(EW.T)
  # normalize
  if args.normalize:
    (b[:, 0], x[:, 0]), scale1 = normalize((b[:, 0], x[:, 0]), new_min=400, new_max=1600)
    (b[:, 1], x[:, 1]), scale2= normalize((b[:, 1], x[:, 1]), new_min=0, new_max=4)
    
    # scale EET accordingly.
    EET = np.matmul(np.array([[scale1, 0], [0, scale2]]), EET)


  # true outcomes (college gpa)
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
  x_norm = (x - x.mean(axis=0, keepdims=True) ) / x.std(axis=0, keepdims=True)
  x_norm = x_norm[np.newaxis]
  y_hat_logits = (x_norm * theta)
  # y_hat_logits = y_hat_logits - y_hat_logits.mean(axis=1, keepdims=True)
  # y_hat_logits = y_hat_logits / y_hat_logits.std(axis=1, keepdims=True)
  y_hat = y_hat_logits.sum(axis=-1)

  def _get_selection(_y_hat, admit_all, n_rounds, accept_rate, rank_type):
    if not admit_all:
      _w = np.zeros_like(_y_hat)
      # comparing applicants coming in the same rounds. 
      for r in range(n_rounds):
        _y_hat_r = _y_hat[r * applicants_per_round: (r+1) * applicants_per_round]
  
        w_r = get_selection(_y_hat_r, accept_rate, rank_type)
        _w[r*applicants_per_round: (r+1)*applicants_per_round] = w_r
    else:
      _w = np.ones_like(y_hat)
    return _w

  w = np.zeros((args.num_envs, num_applicants))
  for env_idx in range(args.num_envs):
    w[env_idx ] = _get_selection(
      y_hat[env_idx], admit_all, n_rounds, args.envs_accept_rates[env_idx], args.rank_type
      )

  z = np.zeros((args.num_applicants, ))
  for idx in range(args.num_applicants):
    w_idx = w[:, idx]
    if w_idx.sum() == 0: # applicant is not accepted anywhere
      z[idx] = 0
    else:
      pvals = w_idx * args.pref_vect / (w_idx * args.pref_vect).sum()
      temp = np.random.multinomial(n=1, pvals=pvals, size=1)
      _idx = temp.flatten().nonzero()[0]
      z[idx] = _idx+1 # offset to avoid conflict with "no uni" decision
  return b,x,y,EET,theta, w, z, y_hat, adv_idx, disadv_idx, g.T, theta_star, args.pref_vect

# %%
def ols(x,y):
  model = LinearRegression(fit_intercept=True)
  model.fit(x, y)
  theta_hat_ols_ = model.coef_
  return theta_hat_ols_
  
def tsls(x,y,theta, pref_vect): # runs until round T
  # my implementation
  # regress x onto theta: estimate omega
  model = LinearRegression()

  _pref_vect = pref_vect.reshape((pref_vect.shape[0], 1, 1))
  theta = (theta * _pref_vect).sum(axis=0)
  assert theta.shape == (y.shape[0], 2)

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

def our(x, y, theta, w):
  model = LinearRegression()
  model.fit(theta, x)

  # recover scaled duplicate 
  theta_admit = theta[w==1]
  x_admit = x[w==1]
  theta_admit_unique, counts = np.unique(theta_admit, axis=0, return_counts=True)

  # to use thetas having the most number of applicants
  idx = np.argsort(counts); idx = idx[::-1]
  theta_admit_unique = theta_admit_unique [idx]

  # construct linear system of eqs.
  assert theta_admit.shape[1] > 1 # algo not applicable for only one feature.
  n_eqs_required = theta_admit.shape[1]

  curr_n_eqs = 0
  A, b = [], []

  i = 0
  while (i < theta_admit_unique.shape[0]):
    j = i+1
    found_pair = False
    while (j < theta_admit_unique.shape[0]) and (found_pair==False):
      pair = theta_admit_unique[[i,j]]

      if np.linalg.matrix_rank(pair) == 1: # if scalar multiple, and exact duplicates are ruled out.
        found_pair =True
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
def run_multi_env(seed, args, env_idx=None):
    np.random.seed(seed)
    _, x, y, EW, theta, w, z, _, _, _, _, theta_star, pref_vect  = generate_data(
    args.num_applicants, args.admit_all, args.applicants_per_round, args.fixed_effort_conversion, args
    )
    
    assert args.num_envs == 1
    y_utility, A, B = generate_data_utility(args, EW, theta, z, theta_star)

    err_list, est_list = {}, {}
    envs_itr = [env_idx] if env_idx is not None else range(args.num_envs)
    for env_idx in envs_itr:
        dictenv_err, dictenv_est = run_single_env(args, x, y, theta, z, theta_star, env_idx, pref_vect, EW, y_utility, A, B)
        for k, v in dictenv_err.items():
            err_list[f'{k}_env{env_idx}'] = v
        for k, v in dictenv_est.items():
            est_list[f'{k}_env{env_idx}'] = v
    
    return err_list, w, z, est_list

def run_single_env(args, x, y, theta, z, theta_star, env_idx, pref_vect, EW, y_utility, A, B):
    # extracting relevant variables for the environment i.
    y_env = y[env_idx].flatten() 
    theta_env = theta[env_idx]
    z_env = z==env_idx+1
        
    upp_limits = [x for x in range(args.applicants_per_round*2, args.num_applicants+1, args.applicants_per_round)]
    if args.offline_eval:
      upp_limits = [args.num_applicants]
        
    err_list, est_list = estimate_causal_params(args, x, theta, theta_star, env_idx, pref_vect, y_env, theta_env, z_env, upp_limits)    
    # TODO: 
    return err_list, est_list

def estimate_causal_params(args, x, theta, theta_star, env_idx, pref_vect, y_env, theta_env, z_env, upp_limits):
    err_list = {m: [None] * len(upp_limits) for m in args.methods}
    est_list = {m: [None] * len(upp_limits) for m in args.methods}
    for i, t in tqdm(enumerate(upp_limits)):
        x_round = x[:t]
        y_env_round = y_env[:t]
        z_env_round = z_env[:t]
        theta_env_round = theta_env[:t]

        # filtering out rejected students
        y_env_round_selected = y_env_round[z_env_round]
        x_round_selected = x_round[z_env_round]
        theta_round_selected = theta[:, :t][:, z_env_round]
        assert theta_round_selected.shape == (args.num_envs, z_env_round.sum(), 2)


        for m in args.methods:
            if m == 'ours':
                est = our(x_round, y_env_round_selected, theta_env_round, z_env_round)
            elif m == '2sls':
                try:
                    est = tsls(x_round_selected, y_env_round_selected, theta_round_selected, pref_vect)
                except np.linalg.LinAlgError:
                    est = np.array([np.nan, np.nan])
            elif m == 'ols':
                est = ols(x_round_selected, y_env_round_selected)
            
            assert theta_star[env_idx].shape == est.shape, f"{theta[0].shape}, {est.shape}"
            err_list[m][i] = np.linalg.norm(theta_star[env_idx] - est )
            est_list[m][i] = est
    return err_list, est_list

def generate_data_utility(args: argparse.Namespace, EET: np.ndarray, theta: np.ndarray, z: np.ndarray, theta_star: np.ndarray):
    """
    Generate data according for utility.
    
    Args:
    EET: np.array, shape = (2,2). 
    theta: np.array, shape = (num_envs, num_applicants, 2)
    z: np.array, shape = (num_applicants, )
    theta_star: np.array, shape = (num_envs, 2)
    
    Returns:
    y, np.array, shape = (n_rounds, applicants_per_round). y_supportive_protocol[i] contains the output of
      accepted applicants, np.nan is left elsewhere.
    A: np.array, shape = (2, )
    B: float
    """
    assert args.num_envs == 1, f"so far only implemented for a single environment"
    env_idx = 0
    theta_unique_env = theta[env_idx, 0::args.applicants_per_round] # (n_rounds, 2) 
    assert theta_unique_env.ndim == 2
    n_rounds = theta_unique_env.shape[env_idx]

    A = np.array([1, 2])
    B = 1.99 
    y_utility = np.ones(shape=(n_rounds, args.applicants_per_round)) * np.nan
    for i, th in enumerate(theta_unique_env):
        # number of accepted students at uni, same as in cooperative protocol.
        n_samples = (z[i*args.applicants_per_round: (i*args.applicants_per_round) + args.applicants_per_round] == env_idx+1).sum ()     
        base = A.dot(th) + B + np.random.normal(loc=0, scale=args.utility_dataset_std, size=(n_samples,))
        impr = th.dot((EET).dot(theta_star[env_idx]))
    
        y_utility[i, :n_samples] = base + impr
    return y_utility, A, B

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