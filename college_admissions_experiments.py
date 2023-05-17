# %%
import sys
import json
import os
import subprocess
from time import time 
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from types import SimpleNamespace
# for notebook. 
args = SimpleNamespace(num_applicants=5000, num_repeat=1, test_run=True, admit_all=False)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# %%
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

# multienv
parser.add_argument('--num-envs', default=1, type=int)
parser.add_argument('--pref',default='uniform',choices=['uniform'], type=str)

# algorithm
parser.add_argument('--sample-weights', action='store_true')

# experiment
parser.add_argument('--test-run', action='store_true')
parser.add_argument('--experiment-root', type=str, default='experiments')
parser.add_argument('--experiment-name', type=str)

# temporary
parser.add_argument('--generate', default=1, choices=[1,2], type=int)
parser.add_argument('--stream', action='store_true')
args = parser.parse_args()


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

def generate_bt(n_samples, mean_sat, mean_gpa, sigma_sat, sigma_gpa):
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
  args.o_bias =1 
  g = np.zeros(shape=(n_samples,args.num_envs))
  g[adv_idx] = np.random.normal((args.o_bias / 2.), scale=0.2, size=(half,args.num_envs) )
  g[disadv_idx] = np.random.normal(-(args.o_bias / 2.), scale=0.2, size=(half, args.num_envs))
  return b, g, adv_idx, disadv_idx

def compute_xt(EWi, b, theta, pref_vect):
  assert EWi.shape[0] == b.shape[0]
  assert b.shape[0] == theta.shape[1]

  n_applicants = b.shape[0]

  x = np.zeros([n_applicants,b.shape[1]])
  # TODO: vectorize this?
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
  thetas = [generate_theta(args) for _ in range(args.num_envs)]
  thetas = np.stack(thetas)
  return thetas

def generate_theta(args):
  if args.admit_all: # harris et. al settings. 
    theta = np.random.multivariate_normal([1,1],[[1, 0], [0, 1]],args.num_applicants)
    return theta 
  else: # selection. in our settings, require theta to be repeating across a batch of students.
    assert args.num_applicants % args.applicants_per_round == 0
    n_rounds = int(args.num_applicants / args.applicants_per_round)

    if args.scaled_duplicates is None: # random vectors for every round
      theta = np.random.multivariate_normal([1,1],[[1, 0], [0, 1]],n_rounds)
    else: # making sure there exists a scaled duplicate of each theta per round
      theta = np.random.multivariate_normal([1,1],[[1, 0], [0, 1]],int(n_rounds / 2))
  
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

def generate_data(num_applicants, admit_all, applicants_per_round, fixed_effort_conversion):
  theta_star = np.zeros(shape=(args.num_envs, 2))
  theta_star[:, 1] = np.random.normal(loc=0.5, scale=0.2, size=(args.num_envs,))

  mean_sat = 900
  mean_gpa = 2
  sigma_sat = 200
  sigma_gpa = 0.5

  if args.pref == 'uniform':
      pref_vect = np.ones(shape=(args.num_envs,))
      pref_vect = pref_vect / np.sum(pref_vect)
  b, g, adv_idx, disadv_idx = generate_bt(num_applicants, mean_sat, mean_gpa, sigma_sat, sigma_gpa)

  # assessment rule 
  theta = generate_thetas(args)
  assert num_applicants % applicants_per_round == 0
  n_rounds = int(num_applicants / applicants_per_round)

  # effort conversion matrices
  EW = np.array([[10.0,0],[0,1.0]])
  EWi = sample_effort_conversion(EW, num_applicants, adv_idx, fixed_effort_conversion)

  # observable features x
  x = compute_xt(EWi, b, theta, pref_vect)

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
  x_ = x.copy()
  x_[:,0] = (x_[:,0] - 400) / 300
  y_hat = (x_ * theta).sum(axis=-1)


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
  return b,x,y,EW,theta, z, y_hat, adv_idx, disadv_idx, g.T, theta_star, pref_vect

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
def test_params(num_applicants, x, y, z, theta, applicants_per_round, theta_star, pref_vect):
  assert x.shape == (args.num_applicants,2)
  assert y.shape == (args.num_envs, args.num_applicants,)
  assert theta.shape == (args.num_envs, args.num_applicants, 2)
  assert z.shape == (args.num_applicants,)
  assert theta_star.shape == (args.num_envs, 2 )

  # test_params_env(num_applicants, x, y, )
  estimates_list_env, error_list_env = [], [] 

  for env_idx in range(args.num_envs):
    # TODO: make it multiprocessing?
    z_env = z==(env_idx+1)
    theta_star_env= theta_star[env_idx]

    [estimates_list, error_list] = test_params_env(
      num_applicants, x, y[env_idx],  z_env, theta, applicants_per_round, theta_star_env, env_idx, pref_vect
    )
    estimates_list_env.append(estimates_list)
    error_list_env.append(error_list)
  
  estimates_list_env, error_list_env = np.stack(estimates_list_env), np.stack(error_list_env)
  return estimates_list_env, error_list_env # \in (num_envs, num_rounds, 3, [2])

def test_params_env(num_applicants, x, y, z, theta, applicants_per_round, theta_star, env_idx, pref_vect):
  assert y.shape == (args.num_applicants, )
  assert z.shape == (args.num_applicants, ) 
  assert theta_star.shape == (2,)

  # save estimates and errors for every even round 
  if args.stream:
    upp_limits = [x for x in range(applicants_per_round*2, num_applicants+1, applicants_per_round)]
  else:
    upp_limits = [num_applicants]
  estimates_list = np.zeros([len(upp_limits),3,2])
  error_list = np.zeros([len(upp_limits),3])

  i=0
  for t in tqdm(upp_limits, leave=False):
    x_round = x[:t]
    y_round = y[:t]
    theta_round = theta[env_idx, :t]
    z_round = z[:t]

    # filtering out rejected students
    x_round_admitted = x_round[z_round==1]
    y_round_admitted = y_round[z_round==1]
    theta_round_admitted = theta_round[z_round==1]

    # estimates
    ols_estimate = ols(x_round_admitted, y_round_admitted) # ols w/ intercept estimate
    try:
      tsls_estimate = tsls(x_round_admitted, y_round_admitted, theta_round_admitted) # 2sls w/ intercept estimate
    except np.linalg.LinAlgError:
      tsls_estimate = np.array([np.nan, np.nan])
    our_estimate = our2(x_round, y_round_admitted, theta_round, z_round)
    # our_estimate = np.array([np.nan, np.nan])
    estimates_list[i,:] += [ols_estimate,tsls_estimate, our_estimate]

    # errors
    ols_error = np.linalg.norm(theta_star-ols_estimate)
    tsls_error = np.linalg.norm(theta_star-tsls_estimate)
    our_error = np.linalg.norm(theta_star-our_estimate)
    error_list[i] = [ols_error,tsls_error, our_error ]
    i+=1
  return [estimates_list, error_list]



def plot_data(data, condition, prefix):
  """
  data \in (n_envs, n_applicants)
  condition \in (n_envs, n_applicants)
  """
  assert data.shape == (args.num_envs, args.num_applicants)
  assert condition.shape == (args.num_applicants,)
  
  n_envs, _ = data.shape
  for env_idx in range(n_envs):
    _,ax = plt.subplots()
    data_  = data[env_idx]
    condition_ = condition
    ax.hist(data_, bins='auto', color='green', label='all',  histtype='step')
    ax.hist(data_[condition_==0], bins='auto', color='red', label='rejected',  histtype='step')
    ax.hist(data_[condition_==1], bins='auto', color='blue', label='accepted',  histtype='step')
    ax.legend()
    savename = os.path.join(dirname, f'{prefix}_env{env_idx}.png')
    plt.savefig(savename)
    plt.close()

def plot_features(x, z, adv_idx, disadv_idx, fname, fname2):
  ## plot first-gen & legacy shift unobservable features (z) to observable (x) 
  fig, ((ax1, ax2)) = plt.subplots(nrows=2, sharex=True)
  ax1.hist(x[disadv_idx,1],bins='auto', label='after manipulation', color='darkorange')
  ax1.axvline(x=np.mean(x[disadv_idx,1]), color='red', linestyle='--', label='mean after manipulation') # before mean
  ax1.set_title("observable high school GPA (x1)")
  ax1.set(ylabel='Number of applicants')
  ax1.hist(z[disadv_idx,1], bins='auto', label='before manipulation', color='yellow', alpha=0.75)
  ax1.axvline(x=np.mean(z[disadv_idx,1]), color='blue', linestyle='--', label='mean before manipulation') # before manipulation
  ax1.set_title("Disadvantaged HS GPA before & after manipulation", fontsize=14)
  # ax1.set_xlim(0,4)
  ax1.set_xlabel('High school GPA (4.0 scale)',fontsize=14)
  ax1.set_ylabel('Number of applicants',fontsize=14)
  ax1.tick_params(axis="x", labelsize=14)
  ax1.tick_params(axis="y", labelsize=14)
  ax1.legend()

  ax2.hist(x[adv_idx,1],bins='auto', label='after manipulation', color='green')
  ax2.hist(z[adv_idx,1], bins='auto', label='before manipulation', color='lightgreen', alpha=0.75)
  ax2.axvline(x=np.mean(z[adv_idx,1]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax2.axvline(x=np.mean(x[adv_idx,1]), color='red', linestyle='--', label='mean after manipulation') # after mean

  ax2.set_title("Advantaged HS GPA before & after manipulation", fontsize=13)
  # ax2.set_xlim(0,4)
  ax2.set_xlabel('High school GPA (4.0 scale)',fontsize=14)
  ax2.set_ylabel('Number of applicants',fontsize=14)
  ax2.tick_params(axis="x", labelsize=14)
  ax2.tick_params(axis="y", labelsize=14)
  ax2.legend()
  fig.tight_layout()
  fname = os.path.join(dirname, f'{fname}')
  plt.savefig(fname, dpi=500)
  plt.close()

  fig,((ax1, ax2)) = plt.subplots(nrows=2, sharex=True)
  ### 2) first-gen SAT
  ax1.hist(x[disadv_idx,0], bins='auto', label='after manipulation', color='orange')
  ax1.axvline(x=np.mean(x[disadv_idx,0]), color='red', linestyle='--', label='mean after manipulation') # after mean
  ax1.set(xlabel='GPA (4.0 scale)', ylabel='Number of applicants')
  ax1.hist(z[disadv_idx,0], bins='auto', label='before manipulation', color='yellow', alpha=0.75)
  ax1.axvline(x=np.mean(z[disadv_idx,0]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax1.set_title("Disadvantaged SAT before & after manipulation", fontsize=14)
  ax1.set_xlabel('SAT score (400 to 1600 points)',fontsize=14)
  ax1.set_ylabel('Number of applicants',fontsize=14)
  ax1.tick_params(axis="x", labelsize=14)
  ax1.tick_params(axis="y", labelsize=14)
  ax1.legend(loc='upper left', fontsize=14)
  ax1.legend()
  ### 4) non-first-gen SAT
  ax2.hist(x[adv_idx,0], bins='auto', label='after manipulation', color='green')
  ax2.hist(z[adv_idx,0], bins='auto', label='before manipulation', color='lightgreen', alpha=0.75)
  ax2.axvline(x=np.mean(z[adv_idx,0]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax2.axvline(x=np.mean(x[adv_idx,0]), color='red', linestyle='--', label='mean after manipulation') # before mean
  ax2.set_title("Advantaged SAT before & after manipulation", fontsize=13)
  ax2.set_xlabel('SAT score (400 to 1600 points)',fontsize=14)
  ax2.set_ylabel('Number of applicants',fontsize=14)
  ax2.tick_params(axis="x", labelsize=14)
  ax2.tick_params(axis="y", labelsize=14)

  #ax4.legend(bbox_to_anchor=(-3, 0), loc='upper left', fontsize=14,ncol=4)
  ax2.legend()
  fig.tight_layout()
  fname2 = os.path.join(dirname, f'{fname2}')
  plt.savefig(fname2, dpi=500)
  plt.close()

def plot_error_estimate(error_list_mean):
  fig,ax=plt.subplots()
  # TODO: check this. 
  ticks = list(range(int(args.num_applicants/5), args.num_applicants+1, int(args.num_applicants/5)))
  ticks.insert(0,1)

  # plot error of OLS vs 2SLS with error bar
  
  assert error_list_mean.ndim == 3
  x = np.arange(error_list_mean.shape[1])
  plt.errorbar(x, np.mean(error_list_mean,axis=0)[:,0], yerr=np.std(error_list_mean,axis=0)[:,0], 
              color='darkorange', ecolor='wheat', label='OLS',elinewidth=10)
  plt.errorbar(x, np.mean(error_list_mean,axis=0)[:,1], yerr=np.std(error_list_mean,axis=0)[:,1], 
              color='darkblue', ecolor='lightblue', label='2SLS',elinewidth=10)
  plt.errorbar(x, np.mean(error_list_mean,axis=0)[:,2], yerr=np.std(error_list_mean,axis=0)[:,2], 
              color='green', ecolor='lightgreen', label='Our',elinewidth=10)
  plt.ylim(0,.25)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  # plt.yscale('log')

  plt.xlabel('iterations', fontsize=14)
  plt.ylabel(r'$|| \hat{\theta} - \theta^* ||$', fontsize=14)

  upp_limits = range(args.applicants_per_round*2, args.num_applicants+1, 2)
  _upp_limits = range(len(upp_limits))
  plt.plot(_upp_limits, 1/(np.sqrt(_upp_limits) + 1e-9), color='red',linestyle='dashed', linewidth=2, label='1/sqrt(T)')

  plt.legend(fontsize=14)
  #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
  #plt.title("Estimation error over rounds for OLS vs 2SLS")

  fname = os.path.join(dirname, 'error_estimation.png')
  plt.savefig(fname, dpi=500, bbox_inches='tight')
  plt.close()
  # plt.show()

def plot_outcome(y, adv_idx, disadv_idx, prefix):
  """
  y \in (n_envs, n_applicants)
  """
  assert y.shape == (args.num_envs, args.num_applicants)
  num_envs, num_applicants = y.shape
  for env_idx in range(num_envs):
    y_ = y[env_idx]
    fig,ax=plt.subplots()
    plt.hist(y_,bins='auto',label='combined')
    plt.axvline(x=np.mean(y_),color='blue',linestyle='--', linewidth = 2, label='combined mean')

    # disadvantaged
    plt.hist(y_[disadv_idx],bins='auto',label='disadvantaged', alpha=.85)
    plt.axvline(x=np.mean(y_[disadv_idx]),color='orange',linestyle='--', linewidth = 2, label='disadvantaged mean')

    # advantaged
    plt.hist(y_[adv_idx],bins='auto',label='advantaged', alpha=0.7)
    plt.axvline(x=np.mean(y_[adv_idx]), linestyle='--', color = 'green', linewidth = 2, label='advantaged mean')

    # plt.xlim(0,4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('College GPA (4.0 scale)', fontsize=14)
    plt.ylabel('Number of applicants', fontsize=14)

    plt.legend(bbox_to_anchor=(0, 1.3), loc='upper left', fontsize=12, ncol=2)

    savename = os.path.join(dirname, f'{prefix}_env{env_idx}.png')
    plt.savefig(savename, dpi=500, bbox_inches='tight')
    plt.close()

def run_experiment(args, i):
  np.random.seed(i)
  b,x,y,EW, theta, z, y_hat, adv_idx, disadv_idx, o, theta_star, pref_vect = generate_data(
    num_applicants=args.num_applicants, admit_all=args.admit_all, applicants_per_round=args.applicants_per_round,
    fixed_effort_conversion=args.fixed_effort_conversion
    )
  plot_data(y, z, f'outcome_select_d{i}')
  plot_data(y_hat, z, f'outcome_pred_select_d{i}')
  plot_features(x, b, adv_idx, disadv_idx, f'features_d{i}_x1.png', f'features_d{i}_x2.png')
  plot_outcome(y, adv_idx, disadv_idx, f'outcome_d{i}')

  [estimates_list, error_list] = test_params(args.num_applicants, x, y, z, theta, args.applicants_per_round, theta_star, pref_vect)
  return estimates_list[np.newaxis], error_list[np.newaxis]
  

# %% 
import pickle as pkl
import time
experiment_name = time.strftime('%Y%m%d-%H%H%S') if args.experiment_name is None else args.experiment_name

if not args.test_run:
  dirname = os.path.join(args.experiment_root, f'{experiment_name}')
  if os.path.exists(dirname):
    dirname = f"{dirname}_"
else:
  dirname = os.path.join(args.experiment_root, f'test-run')
  if os.path.exists(dirname):
    import shutil
    shutil.rmtree(dirname)
  
os.makedirs(dirname)
git_hash = get_git_revision_hash()
with open(os.path.join(dirname, 'git_hash.txt'), 'w+') as f:
  f.write(git_hash)
  f.write('\n')
  f.write(' '.join(sys.argv))
  
epochs = args.num_repeat
half = args.num_applicants

estimates_list_mean = np.zeros((epochs,half,2,2))
error_list_mean = np.zeros((epochs,half,2))
estimates_list_mean = []
error_list_mean = []

estimates_list_mean, error_list_mean = [], [] 

if args.n_cores == 1:  # sequential
  for i in tqdm(range(epochs)):
    # run experiment
    estimates_list, error_list = run_experiment(args, i)

    estimates_list_mean.append(estimates_list)
    error_list_mean.append(error_list)
elif args.n_cores > 1:
  import multiprocessing as mp  
  args_list = [(args, i) for i in range(epochs)]
  with mp.Pool(processes=args.n_cores) as p:
    out = p.starmap(
      run_experiment, args_list
    )

  for _out in out:
    if not (_out is None): # if no LA error.
      estimates_list_mean.append(_out[0])
      error_list_mean.append(_out[1])

  # estimates_list_mean = [_out[0] for _out in out]
  # error_list_mean = [_out[1] for _out in out]
else:
  print('n cores invalid.')
  
estimates_list_mean = np.concatenate(estimates_list_mean,axis=0)
error_list_mean = np.concatenate(error_list_mean,axis=0)


filename = os.path.join(dirname, "data")
# filename = 'college_admission_'+timestr
with open(filename, 'wb') as f:
    save = {
      'estimates_list_mean': estimates_list_mean, 
      'error_list_mean': error_list_mean,
      # 'y': y, 'x': x, 'z': z,  'EW':EW, 'theta':theta, 'theta_star': theta_star, 'w': w, 'y_hat':y_hat
    }
    pkl.dump(save, f)

# plot_error_estimate(error_list_mean)

# save group data.
filename = os.path.join(dirname, "eqs_data.json")
with open(filename, "w") as f:
  json.dump(eqs_data, f)