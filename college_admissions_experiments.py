# %%
import sys
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

#%%
from types import SimpleNamespace
# for notebook. 
args = SimpleNamespace(num_applicants=5000, num_repeat=1, test_run=True, admit_all=False)

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

# %%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num-applicants', default=10000, type=int)
parser.add_argument('--num-repeat', default=10, type=int)
parser.add_argument('--test-run', action='store_true')
parser.add_argument('--admit-all', action='store_true', help='admit all students, as in Harris et. al')
parser.add_argument('--applicants-per-round', default=1, type=int, help='used for identical thetas')
parser.add_argument('--fixed-effort-conversion', action='store_true')

# experiment
parser.add_argument('--experiment-root', type=str, default='experiments')
parser.add_argument('--experiment-name', type=str)

# temporary
parser.add_argument('--generate', default=1, choices=[1,2], type=int)
args = parser.parse_args()

theta_star = np.array([0,0.5])
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

def generate_data2(n_seen_applicants, admit_all, applicants_per_round, fixed_effort_conversion):

  mean_sat, mean_gpa = 900, 2
  sigma_sat, sigma_gpa = 200, 0.5
  EW = np.array([[10.0,0],[0,1.0]])
  total_applicants, accepted_applicants = 0, 0
  b, x, y, theta, w, y_hat, adv_idx, disadv_idx = \
    [], [], [], [], [], [], [], []
  while accepted_applicants < n_seen_applicants:

    # generate applicants b_t
    br, gr, adv_idxr, disadv_idxr = generate_bt(
      applicants_per_round, mean_sat, mean_gpa, sigma_sat, sigma_gpa
      )
    
    # effort conversion
    EWir = sample_effort_conversion(EW, applicants_per_round, adv_idxr, fixed_effort_conversion)

    # theta
    thetar = np.random.multivariate_normal([1,1],[[1, 0], [0, 1]], 1)
    thetar = np.repeat(thetar, repeats=applicants_per_round, axis=0)
    
    # x_t
    xr = compute_xt(EWir, br, thetar)
    yr = np.clip(np.matmul(xr,theta_star) + gr,0,4) # clipped outcomes

    assert xr.shape == thetar.shape
    xr_ = xr.copy()
    xr_[:,0] = (xr_[:,0] - 400) / 300
    y_hatr = (xr_ * thetar).sum(axis=-1)

    # selection
    if not admit_all:
      wr = get_selection(y_hatr)
    else:
      wr = np.ones((applicants_per_round,))

    # update data for iteration
    b.append(br); x.append(xr); y.append(yr); theta.append(thetar)
    w.append(wr); y_hat.append(y_hatr)
    adv_idx.append(adv_idxr + total_applicants) # offset 
    disadv_idx.append(disadv_idxr + total_applicants) # offset

    total_applicants += applicants_per_round
    accepted_applicants += np.sum(wr)

  b, x, y, theta, w, y_hat, adv_idx, disadv_idx = map(
      lambda x: np.concatenate(x, axis=0),
      (b, x, y, theta, w, y_hat, adv_idx, disadv_idx)
      )
  return b, x, y, EW, theta, theta_star, w, y_hat, adv_idx, disadv_idx

    


def generate_bt(n_samples, mean_sat, mean_gpa, sigma_sat, sigma_gpa):
  assert n_samples % 2 == 0, f"{n_samples} not divisible by 2"
  half = int(n_samples / 2)
  b = np.zeros([n_samples,2])
  
  # indices for shuffling
  idx = np.arange(n_samples)
  np.random.shuffle(idx)
  disadv_idx = idx[:half]
  adv_idx = idx[half:]

  # disadvantaged students
  b[disadv_idx,0] = np.random.normal(mean_sat-100,sigma_sat,b[disadv_idx][:,0].shape) #SAT
  b[disadv_idx,1] = np.random.normal(mean_gpa-.2,sigma_gpa,b[disadv_idx][:,1].shape) #GPA

  # advantaged students
  b[adv_idx,0] = np.random.normal(mean_sat+100,sigma_sat,b[adv_idx][:,0].shape) #SAT
  b[adv_idx,1] = np.random.normal(mean_gpa+.2,sigma_gpa,b[adv_idx][:,1].shape) #GPA

  b[:,0] = np.clip(b[:,0],400,1600) # clip to 400 to 1600
  b[:,1] = np.clip(b[:,1],0,4) # clip to 0 to 4.0

  # confounding error term g (error on true college GPA)
  g = np.ones(n_samples)*0.5 # legacy students shifted up
  g[disadv_idx]=-0.5 # first-gen students shifted down
  g += np.random.normal(1,0.2,size=n_samples) # non-zero-mean

  return b, g, adv_idx, disadv_idx

def compute_xt(EWi, b, theta):
  assert EWi.shape[0] == b.shape[0]
  assert b.shape[0] == theta.shape[0]

  n_applicants = b.shape[0]

  x = np.zeros([n_applicants,b.shape[1]])
  # TODO: vectorize this?
  for i in range(n_applicants):
    x[i] = b[i] + np.matmul(EWi[i].dot(EWi[i].T),theta[i]) # optimal solution

  x[:,0] = np.clip(x[:,0],400,1600) # clip to 400 to 1600
  x[:,1] = np.clip(x[:,1],0,4) # clip to 0 to 4.0
  return x

def generate_data(num_applicants, admit_all, applicants_per_round, fixed_effort_conversion):
  half = int(num_applicants/2) 
  m = theta_star.size

  mean_sat = 900
  mean_gpa = 2
  sigma_sat = 200
  sigma_gpa = 0.5

  b, g, adv_idx, disadv_idx = generate_bt(num_applicants, mean_sat, mean_gpa, sigma_sat, sigma_gpa)

  # assessment rule 
  assert num_applicants % applicants_per_round == 0
  n_rounds = int(num_applicants / applicants_per_round)
  theta = np.random.multivariate_normal([1,1],[[1, 0], [0, 1]],n_rounds)

  # theta repeating over the rounds.
  theta = np.repeat(theta, repeats=applicants_per_round, axis=0)
  assert theta.shape[0] == num_applicants

  # effort conversion matrices
  EW = np.array([[10.0,0],[0,1.0]])
  EWi = sample_effort_conversion(EW, num_applicants, adv_idx, fixed_effort_conversion)

  # observable features x
  x = compute_xt(EWi, b, theta)

  # true outcomes (college gpa)
  y = np.clip(np.matmul(x,theta_star) + g,0,4) # clipped outcomes
  
  # our setup addition 
  # computing admission results.
  assert x.shape == theta.shape
  x_ = x.copy()
  x_[:,0] = (x_[:,0] - 400) / 300
  y_hat = (x_ * theta).sum(axis=-1)
  if not admit_all:
    w = np.zeros_like(y_hat)
    i = 0
    # comparing people coming in the same rounds. 
    for r in range(n_rounds):
      y_hat_r = y_hat[r * applicants_per_round: (r+1) * applicants_per_round]
  
      for _y_hat_r in y_hat_r:
        prob = np.mean(y_hat_r <= _y_hat_r)
        w[i] = np.random.binomial(n=1, p=prob)
        i += 1
  else:
    w = np.ones_like(y_hat)

  return b,x,y,EW,theta,theta_star, w, y_hat, adv_idx, disadv_idx

# %%
def get_selection(y_hat):
  assert y_hat.ndim == 1
  n_applicants = y_hat.shape[0]
  w = np.zeros_like(y_hat)
  for i, _y_hat in enumerate(y_hat):
    y_hat_peers = y_hat[np.arange(n_applicants) != i] # scores of everyone but the current applicant
    prob = np.mean(y_hat_peers <= _y_hat)
    w[i] = np.random.binomial(n=1, p=prob)
  return w

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

def test_params2(num_applicants, x, y, w, theta, theta_star, applicants_per_round):
  [x_, y_, theta_, w_] = [x.copy(),y.copy(),theta.copy(), w.copy()]

  # admitted datapoints
  x_, y_, theta_ = x_[w_==1], y_[w_==1], theta_[w_ == 1]

  x_ = x_[:num_applicants]
  y_ = y_[:num_applicants]
  theta_ = theta_[:num_applicants]

  assert x_.shape[0] == num_applicants
  assert y_.shape[0] == num_applicants
  assert theta_.shape[0] == num_applicants
  
  upp_limits = range(applicants_per_round*2,num_applicants+1,2)
  estimates_list = np.zeros([len(upp_limits),2,2])
  error_list = np.zeros([len(upp_limits),2])

  i=0
  for t in tqdm(upp_limits, leave=False):
    # filtering out rejected students
    x_round = x_[:t]
    y_round = y_[:t]
    theta_round = theta_[:t]
    # w_round = w_[:t]

    # x_round = x_round[w_round==1]
    # y_round = y_round[w_round==1]
    # theta_round = theta_round[w_round==1] # TOASK:limit access to theta as well? 

    # estimates
    ols_estimate = ols(x_round, y_round) # ols w/ intercept estimate
    tsls_estimate = tsls(x_round, y_round, theta_round) # 2sls w/ intercept estimate
    estimates_list[i,:] += [ols_estimate,tsls_estimate]

    # errors
    ols_error = np.linalg.norm(theta_star-ols_estimate)
    tsls_error = np.linalg.norm(theta_star-tsls_estimate)
    error_list[i] = [ols_error,tsls_error]
    
    i+=1

  return [estimates_list, error_list]
#%%
# def _filterAndPartition(x, y_admit, w, theta):
#   # D_partial is same as Dfull, right?
# def _inferCausalParams(EET_hat, x, y_admit, w, theta):
#   Psi = _filterAndPartition(x, y_admit, w, theta)  # TOASK: dimensions of Psi?
#   pass
# def our(x, y_admit, w, theta): 
#   assert y_admit.size == w.sum()
#   # step 1. estimate EE^T.
#   model = LinearRegression()
#   model.fit(theta, x)
#   EET_hat = model.coef_.T

#   # step 2. infer causal params.
#   theta_star_hat = _inferCausalParams(EET_hat, x, y_admit, w, theta)
  
# def test_params(n_applicants, x, y, w, theta, theta_star, applicants_per_round):
#   # [x_, y_, theta, w] = [x.copy(), y.copy(), theta.copy(), w.copy()]

#   upp_limits = range(applicants_per_round*2, n_applicants+1, 2)
#   estimates_list = np.zeros(shape=(len(upp_limits), 2, 2))
#   error_list = np.zeros(shape=(len(upp_limits), 2))

#   i = 0
#   for t in tqdm(upp_limits):
#     xr, yr, thetar, wr = x[:t], y[:t], theta[:t], w[:t]

#     yr_admit = yr[wr==1]
#     xr_admit = xr[wr==1]
#     thetar_admit = thetar[wr==1]

#     ols_estimate = ols(xr_admit, yr_admit)
#     tsls_estimate = tsls(xr_admit, yr_admit, thetar_admit)
    
#     estimates_list[i,:] += [ols_estimate, tsls_estimate]

#     ols_error = np.linalg.norm(theta_star-ols_estimate)
#     tsls_error = np.linalg.norm(theta_star-tsls_estimate)
#     error_list[i] = [ols_error, tsls_error]
#     i += 1
  
#   return estimates_list, error_list
#%%
def plot_data(data, condition, name='dataset.pdf'):
  fig,ax=plt.subplots()
  ax.hist(data, bins='auto', color='green', label='all',  histtype='step')
  ax.hist(data[condition==0], bins='auto', color='red', label='rejected',  histtype='step')
  ax.hist(data[condition==1], bins='auto', color='blue', label='accepted',  histtype='step')
  ax.legend()
  plt.savefig(os.path.join(dirname, name))
  plt.close()

# %% 
import pickle as pkl
import time
experiment_name = time.strftime('%Y%m%d-%H%H%S') if args.experiment_name is None else args.experiment_name

if not args.test_run:
  dirname = os.path.join(args.experiment_root, f'{experiment_name}')
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
  
# %%
# fix T variable.  
T = args.num_applicants
epochs = args.num_repeat
half = int(T/2)

estimates_list_mean = np.zeros((epochs,half,2,2))
error_list_mean = np.zeros((epochs,half,2))
estimates_list_mean = []
error_list_mean = []

#%%
for i in tqdm(range(epochs)):
  np.random.seed(i)
  if args.generate == 1:
    z,x,y,EW, theta, theta_star, w, y_hat, adv_idx, disadv_idx = generate_data(
      num_applicants=T, admit_all=args.admit_all, applicants_per_round=args.applicants_per_round,
      fixed_effort_conversion=args.fixed_effort_conversion
      )
  elif args.generate == 2:
    z,x,y,EW, theta, theta_star, w, y_hat, adv_idx, disadv_idx = generate_data2(
      n_seen_applicants=T, admit_all=args.admit_all, applicants_per_round=args.applicants_per_round,
      fixed_effort_conversion=args.fixed_effort_conversion
    )
  
  # plot data.
  plot_data(y, w, 'dataset_y.pdf')
  plot_data(y_hat, w, 'dataset_y_hat.pdf')
  try:
    [estimates_list, error_list] = test_params2(T, x, y, w, theta, theta_star, args.applicants_per_round)
    estimates_list_mean.append(estimates_list[np.newaxis])
    error_list_mean.append(error_list[np.newaxis])
  except np.linalg.LinAlgError:
    pass # record nothing in case the algorithm fails.  

# %%
estimates_list_mean = np.concatenate(estimates_list_mean,axis=0)
error_list_mean = np.concatenate(error_list_mean,axis=0)

T = x.shape[0]

filename = os.path.join(dirname, "data")
# filename = 'college_admission_'+timestr
with open(filename, 'wb') as f:
    save = {
      'estimates_list_mean': estimates_list_mean, 
      'error_list_mean': error_list_mean,
      'y': y, 'x': x, 'z': z,  'EW':EW, 'theta':theta, 'theta_star': theta_star, 'w': w, 'y_hat':y_hat
    }
    pkl.dump(save, f)

# %%
def plot_features():
  ## plot first-gen & legacy shift unobservable features (z) to observable (x) 
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(12,10)) #constrained_layout=False

  ### first-gen HS GPA
  ax1.hist(x[disadv_idx,1],bins='auto', label='after manipulation', color='darkorange')

  ax1.axvline(x=np.mean(x[disadv_idx,1]), color='red', linestyle='--', label='mean after manipulation') # before mean

  ax1.set_title("observable high school GPA (x1)")
  ax1.set(ylabel='Number of applicants')

  ax1.hist(z[disadv_idx,1], bins='auto', label='before manipulation', color='yellow', alpha=0.75)
  ax1.axvline(x=np.mean(z[disadv_idx,1]), color='blue', linestyle='--', label='mean before manipulation') # before manipulation

  ax1.set_title("Disadvantaged HS GPA before & after manipulation", fontsize=14)
  ax1.set_xlim(0,4)
  ax1.set_xlabel('High school GPA (4.0 scale)',fontsize=14)
  ax1.set_ylabel('Number of applicants',fontsize=14)
  ax1.tick_params(axis="x", labelsize=14)
  ax1.tick_params(axis="y", labelsize=14)

  ax1.legend()

  ### 2) first-gen SAT
  ax2.hist(x[disadv_idx,0], bins='auto', label='after manipulation', color='orange')
  ax2.axvline(x=np.mean(x[disadv_idx,0]), color='red', linestyle='--', label='mean after manipulation') # after mean

  ax2.set(xlabel='GPA (4.0 scale)', ylabel='Number of applicants')
  ax2.hist(z[disadv_idx,0], bins='auto', label='before manipulation', color='yellow', alpha=0.75)
  ax2.axvline(x=np.mean(z[disadv_idx,0]), color='blue', linestyle='--', label='mean before manipulation') # before mean

  ax2.set_title("Disadvantaged SAT before & after manipulation", fontsize=14)
  ax2.set_xlim(400,1600)
  ax2.set_xlabel('SAT score (400 to 1600 points)',fontsize=14)
  ax2.set_ylabel('Number of applicants',fontsize=14)
  ax2.tick_params(axis="x", labelsize=14)
  ax2.tick_params(axis="y", labelsize=14)

  #ax2.legend(loc='upper left', fontsize=14)
  ax2.legend()

  ### 3) non-first-gen HS GPA
  ax3.hist(x[adv_idx,1],bins='auto', label='after manipulation', color='green')
  ax3.hist(z[adv_idx,1], bins='auto', label='before manipulation', color='lightgreen', alpha=0.75)
  ax3.axvline(x=np.mean(z[adv_idx,1]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax3.axvline(x=np.mean(x[adv_idx,1]), color='red', linestyle='--', label='mean after manipulation') # after mean

  ax3.set_title("Advantaged HS GPA before & after manipulation", fontsize=13)
  ax3.set_xlim(0,4)
  ax3.set_xlabel('High school GPA (4.0 scale)',fontsize=14)
  ax3.set_ylabel('Number of applicants',fontsize=14)
  ax3.tick_params(axis="x", labelsize=14)
  ax3.tick_params(axis="y", labelsize=14)
  ax3.legend()

  ### 4) non-first-gen SAT
  ax4.hist(x[adv_idx,0], bins='auto', label='after manipulation', color='green')
  ax4.hist(z[adv_idx,0], bins='auto', label='before manipulation', color='lightgreen', alpha=0.75)
  ax4.axvline(x=np.mean(z[adv_idx,0]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax4.axvline(x=np.mean(x[adv_idx,0]), color='red', linestyle='--', label='mean after manipulation') # before mean

  ax4.set_title("Advantaged SAT before & after manipulation", fontsize=13)
  ax4.set_xlim(400,1600)
  ax4.set_xlabel('SAT score (400 to 1600 points)',fontsize=14)
  ax4.set_ylabel('Number of applicants',fontsize=14)
  ax4.tick_params(axis="x", labelsize=14)
  ax4.tick_params(axis="y", labelsize=14)

  #ax4.legend(bbox_to_anchor=(-3, 0), loc='upper left', fontsize=14,ncol=4)
  ax4.legend()

  ## legend
  pre_fg = Patch(color='yellow', label='disadvantaged unobserved (unmanipulated)', alpha=0.75)
  post_fg = Patch(color='darkorange', label='disadvantaged observed (manipulated)')

  pre_ls = Patch(color='lightgreen', label='legacy unobserved (unmanipulated)', alpha=0.75)
  post_ls = Patch(color='green', label='legacy observed (manipulated)')

  before_mean = Line2D([0], [0], color='blue', linestyle='--', lw=2, label='mean before manipulation')
  after_mean = Line2D([0], [0], color='red', linestyle='--', lw=2, label='mean after manipulation')

  fig.tight_layout()

  fname = os.path.join(dirname, 'fg-ls_shifted_features.pdf')
  plt.savefig(fname, dpi=500)
  plt.close()

# plt.show()

# %%
# vars for pyplot
def plot_error_estimate():
  fig,ax=plt.subplots()
  ticks = list(range(int(T/5), T+1, int(T/5)))
  ticks.insert(0,1)

  # plot error of OLS vs 2SLS with error bar
  
  assert error_list_mean.ndim == 3
  x = np.arange(error_list_mean.shape[1])
  plt.errorbar(x, np.mean(error_list_mean,axis=0)[:,0], yerr=np.std(error_list_mean,axis=0)[:,0], 
              color='darkorange', ecolor='wheat', label='OLS',elinewidth=10)
  plt.errorbar(x, np.mean(error_list_mean,axis=0)[:,1], yerr=np.std(error_list_mean,axis=0)[:,1], 
              color='darkblue', ecolor='lightblue', label='2SLS',elinewidth=10)
  plt.ylim(0,.25)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  # plt.yscale('log')
  # plt.xlim(50,T-11)

  plt.xlabel('Number of applicants (rounds)', fontsize=14)
  plt.ylabel(r'$|| \hat{\theta} - \theta^* ||$', fontsize=14)

  # plt.plot(range(1,T+1), 1/np.sqrt(range(1,T+1)), color='red',linestyle='dashed', linewidth=2, label='1/sqrt(T)')

  plt.legend(fontsize=14)
  #plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
  #plt.title("Estimation error over rounds for OLS vs 2SLS")

  fname = os.path.join(dirname, 'error_estimation.png')
  plt.savefig(fname, dpi=500, bbox_inches='tight')
  plt.close()
  # plt.show()


# %%
def plot_outcome():
  fig,ax=plt.subplots()
  plt.hist(y,bins='auto',label='combined')
  plt.axvline(x=np.mean(y),color='blue',linestyle='--', linewidth = 2, label='combined mean')

  # disadvantaged
  plt.hist(y[disadv_idx],bins='auto',label='disadvantaged', alpha=.85)
  plt.axvline(x=np.mean(y[disadv_idx]),color='orange',linestyle='--', linewidth = 2, label='disadvantaged mean')

  # advantaged
  plt.hist(y[adv_idx],bins='auto',label='advantaged', alpha=0.7)
  plt.axvline(x=np.mean(y[adv_idx]), linestyle='--', color = 'green', linewidth = 2, label='advantaged mean')

  plt.xlim(0,4)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlabel('College GPA (4.0 scale)', fontsize=14)
  plt.ylabel('Number of applicants', fontsize=14)

  #plt.title("True college GPA (y) for disadvantaged vs. advantaged students")

  plt.legend(bbox_to_anchor=(0, 1.3), loc='upper left', fontsize=12, ncol=2)

  fname = os.path.join(dirname, 'all_outcome.png')
  plt.savefig(fname, dpi=500, bbox_inches='tight')
  plt.close()
  # plt.show()

plot_features()
plot_error_estimate()
plot_outcome()
