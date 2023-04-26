# %%
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
parser.add_argument('--num-applicants', default=5000, type=int)
parser.add_argument('--num-repeat', default=10, type=int)
parser.add_argument('--test-run', action='store_true')
parser.add_argument('--admit-all', action='store_true', help='admit all students, as in Harris et. al')
parser.add_argument('--experiment-name', type=str)
args = parser.parse_args()


 # %%
# ### First, estimate theta*, true causal effect of (SAT, HS GPA) on college GPA
# ## based on real data from 1000 students
# df = pd.read_csv("clean_gpa.csv")

# x_real = df[['new_sat','new_gpa']].to_numpy()
# y_real = df['college_gpa'].to_numpy()

# ## find true causal effects theta*
# # ordinary least squares (ols)
# x_tilde = np.hstack((x_real,np.ones((len(x_real),1)))) # add 1 for intercept

# m = x_real.shape[1]
# x_sum = np.zeros([m+1,m+1])
# xy_sum = np.zeros(m+1)

# for i in range(len(x_real)):
#   x_sum += np.outer(x_tilde[i],x_tilde[i])
#   xy_sum += x_tilde[i]*y_real[i]

# theta_star = np.matmul(np.linalg.inv(x_sum),xy_sum)[:-1]

# set theta* to nice values for synthetic data
# 1st entry in theta* is for SAT score, 2nd is for HSGPA
theta_star = np.array([0,0.5])
# %%
def generate_data(num_applicants, admit_all):
  half = int(num_applicants/2) 
  m = theta_star.size

  mean_sat = 900
  mean_gpa = 2
  sigma_sat = 200
  sigma_gpa = 0.5

  # initial features (z)
  z = np.zeros([num_applicants,m])

  # disadvantaged students
  z[0:half,0] = np.random.normal(mean_sat-100,sigma_sat,z[0:half,0].shape) #SAT
  z[0:half,1] = np.random.normal(mean_gpa-.2,sigma_gpa,z[0:half,1].shape) #GPA

  # advantaged students
  z[half:,0] = np.random.normal(mean_sat+100,sigma_sat,z[0:half,0].shape) #SAT
  z[half:,1] = z[half:,1] + np.random.normal(mean_gpa+.2,sigma_gpa,z[half:,1].shape) #GPA

  z[:,0] = np.clip(z[:,0],400,1600) # clip to 400 to 1600
  z[:,1] = np.clip(z[:,1],0,4) # clip to 0 to 4.0

  # confounding error term g (error on true college GPA)
  g = np.ones(num_applicants)*0.5 # legacy students shifted up
  g[0:half]=-0.5 # first-gen students shifted down
  g += np.random.normal(1,0.2,size=num_applicants) # non-zero-mean

  # assessment rule 
  theta = np.random.multivariate_normal([1,1],[[1, 0], [0, 1]],num_applicants)

  # effort conversion matrices
  EW = np.matrix([[10.0,0],[0,1.0]])

  # observable features x
  x = np.zeros([num_applicants,z.shape[1]])
  for i in range(num_applicants):
    x[i] = z[i] + np.matmul(EW.dot(EW.T),theta[i]) # optimal solution

  x[:,0] = np.clip(x[:,0],400,1600) # clip to 400 to 1600
  x[:,1] = np.clip(x[:,1],0,4) # clip to 0 to 4.0

  # true outcomes (college gpa)
  y = np.clip(np.matmul(x,theta_star) + g,0,4) # clipped outcomes
  
  # our setup addition 
  # computing admission results.
  assert x.shape == theta.shape
  x_ = x.copy()
  x_[:,0] = (x[:,0] - 400 ) / 300
  y_hat = (x_ * theta).sum(axis=-1)
  if not admit_all:
    w = np.zeros_like(y_hat)
    for idx, _y_hat in enumerate(y_hat):
      prob = np.mean(y_hat <= _y_hat)
      w[idx] = np.random.binomial(n=1, p=prob)

  else:
    w = np.ones_like(y_hat)

  return z,x,y,EW,theta,theta_star, w, y_hat

# %%
def test_params(num_applicants, x, y, w, theta, theta_star):
  # inputs:  num_applicants = number of applicants (time horizon T), 
  #          EW = expected effort conversion matrix E[W],
  #          theta_star = true causal effects theta* (set to [0,0.5] by default)
  #
  # output:  synthetic data for num_applicants rounds, including:
  #           z (unobserved, unmanipulated features), 
  #           x (observed, manipulated features), y (outcome), 
  #           theta (decision rule), and WWT (effort conversion matrix)
  #           
  #           estimate_list: OLS & 2SLS estimates @ rounds 10 to num_applicants
  #           error_list: L2-norm of OLS & 2SLS estimates minus true theta*
  #
  # outline: 1) create synthetic unobserved data (z_t, W_tW_t^T, g_t), 
  #             add confounding by splitting data into two types split 50/50,
  #             (1st half disadvantaged, 2nd half advantaged), 
  #             make z & WW^T worse for disadvantaged, better for advantaged
  #             & set mean g lesser for disadvantaged, high for advantaged
  #          2) set decision rule theta_t & solve for x_t and y_t based on model
  #          3) OLS estimate by regressing x onto y (w/ intercept estimate)
  #          4) 2SLS estimate by regressing x onto theta, then theta onto y (w/ intercept estimate)


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
    
    # assert np.allclose(omega_hat_, omega_hat, rtol=0, atol=1e-5), f"ours: {omega_hat_}; theirs: {omega_hat}"
    # assert np.allclose(lambda_hat_, lambda_hat, rtol=0, atol=1e-5), f"ours: {lambda_hat_}; theirs: {lambda_hat}"
    # assert np.allclose(theta_hat_tsls_, theta_hat_tsls, rtol=0, atol=1e-5), f"ours: {theta_hat_tsls_}; theirs: {theta_hat_tsls}"
    return theta_hat_tsls_

  # shuffle the samples so types show up randomly
  [x_shuffle,y_shuffle,theta_shuffle, w_shuffle] = [x.copy(),y.copy(),theta.copy(), w.copy()]
  shuffle_iter = list(range(len(x)))
  np.random.shuffle(shuffle_iter)

  j = 0
  for k in shuffle_iter:
    x_shuffle[j] = x[k]
    y_shuffle[j] = y[k]
    theta_shuffle[j] = theta[k]
    w_shuffle[j] = w[k]
    j+=1

  i=0
  # save estimates and errors for every even round 
  estimates_list = np.zeros([int((num_applicants/2)),2,2])
  error_list = np.zeros([int((num_applicants)/2),2])

  for t in tqdm(range(10,num_applicants,2), leave=False):
    # filtering out rejected students
    x_round = x_shuffle[:t]
    y_round = y_shuffle[:t]
    theta_round = theta_shuffle[:t]
    w_round = w_shuffle[:t]

    x_round = x_round[w_round==1]
    y_round = y_round[w_round==1]
    theta_round = theta_round[w_round==1] # TOASK:limit access to theta as well? 

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
def plot_data(data, condition, name='dataset.png'):
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
  dirname = os.path.join('experiments', f'{experiment_name}')
else:
  dirname = os.path.join('experiments',f'test-run')
  if os.path.exists(dirname):
    import shutil
    shutil.rmtree(dirname)
  
os.makedirs(dirname)
git_hash = get_git_revision_hash()
with open(os.path.join(dirname, 'git_hash.txt'), 'w+') as f:
  f.write(git_hash)
  
# %%
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
  z,x,y,EW, theta, theta_star, w, y_hat = generate_data(num_applicants=T, admit_all=args.admit_all)
  
  # plot data.
  plot_data(y, w, 'dataset_y.png')
  plot_data(y_hat, w, 'dataset_y_hat.png')

  try:
    [estimates_list, error_list] = test_params(T, x, y, w, theta, theta_star)
    estimates_list_mean.append(estimates_list[np.newaxis])
    error_list_mean.append(error_list[np.newaxis])
  except np.linalg.LinAlgError:
    pass # record nothing in case the algorithm fails.  

# %%
estimates_list_mean = np.concatenate(estimates_list_mean,axis=0)
error_list_mean = np.concatenate(error_list_mean,axis=0)

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
# # save to file
# y.tofile("saved_y_list")
# x.tofile("saved_x_list")
# z.tofile("saved_z_list")
# np.array(WWT).tofile("saved_WWT_list")
# # EWWT.tofile("saved_EWWT")
# theta_star.tofile("saved_theta_star")
# estimates_list_mean.tofile("saved_estimates_list_mean")
# error_list_mean.tofile("saved_error_list_mean")
# %%
def plot_features():
  ## plot first-gen & legacy shift unobservable features (z) to observable (x) 
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(12,10)) #constrained_layout=False

  ### first-gen HS GPA
  ax1.hist(x[0:half,1],bins='auto', label='after manipulation', color='darkorange')

  ax1.axvline(x=np.mean(x[0:half,1]), color='red', linestyle='--', label='mean after manipulation') # before mean
  ax1.axvline(x=np.mean(x[0:half,1]), linestyle='-', color = 'red', linewidth = 4) # after mean
  ax1.axvline(x=np.mean(x[0:half,1]), linestyle='--', color = 'white', linewidth = 4) # after mean

  ax1.set_title("observable high school GPA (x1)")
  ax1.set(ylabel='Number of applicants')

  ax1.hist(z[0:half,1], bins='auto', label='before manipulation', color='yellow', alpha=0.75)
  ax1.axvline(x=np.mean(z[0:half,1]), color='blue', linestyle='--', label='mean before manipulation') # before manipulation
  ax1.axvline(x=np.mean(z[0:half,1]), linestyle='-', color = 'blue', linewidth = 4) # before mean
  ax1.axvline(x=np.mean(z[0:half,1]), linestyle='--', color = 'white', linewidth = 4) # before mean

  ax1.set_title("Disadvantaged HS GPA before & after manipulation", fontsize=14)
  ax1.set_xlim(0,4)
  ax1.set_xlabel('High school GPA (4.0 scale)',fontsize=14)
  ax1.set_ylabel('Number of applicants',fontsize=14)
  ax1.tick_params(axis="x", labelsize=14)
  ax1.tick_params(axis="y", labelsize=14)

  ax1.legend()

  ### 2) first-gen SAT
  ax2.hist(x[0:half,0], bins='auto', label='after manipulation', color='orange')
  ax2.axvline(x=np.mean(x[0:half,0]),color='blue',)
  ax2.axvline(x=np.mean(x[0:half,0]), color='red', linestyle='--', label='mean after manipulation') # after mean
  ax2.axvline(x=np.mean(x[0:half,0]), linestyle='-', color = 'red', linewidth = 4) # after mean
  ax2.axvline(x=np.mean(x[0:half,0]), linestyle='--', color = 'white', linewidth = 4) # after mean

  ax2.set(xlabel='GPA (4.0 scale)', ylabel='Number of applicants')

  ax2.hist(z[0:half,0], bins='auto', label='before manipulation', color='yellow', alpha=0.75)

  ax2.axvline(x=np.mean(z[0:half,0]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax2.axvline(x=np.mean(z[0:half,0]), linestyle='-', color = 'blue', linewidth = 4) # before mean
  ax2.axvline(x=np.mean(z[0:half,0]), linestyle='--', color = 'white', linewidth = 4) # before mean

  ax2.set_title("Disadvantaged SAT before & after manipulation", fontsize=14)
  ax2.set_xlim(400,1600)
  ax2.set_xlabel('SAT score (400 to 1600 points)',fontsize=14)
  ax2.set_ylabel('Number of applicants',fontsize=14)
  ax2.tick_params(axis="x", labelsize=14)
  ax2.tick_params(axis="y", labelsize=14)

  #ax2.legend(loc='upper left', fontsize=14)
  ax2.legend()

  ### 3) non-first-gen HS GPA
  ax3.hist(x[half:,1],bins='auto', label='after manipulation', color='green')

  ax3.hist(z[half:,1], bins='auto', label='before manipulation', color='lightgreen', alpha=0.75)

  ax3.axvline(x=np.mean(z[half:,1]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax3.axvline(x=np.mean(z[half:,1]), linestyle='-', color = 'blue', linewidth = 4) # before mean
  ax3.axvline(x=np.mean(z[half:,1]), linestyle='--', color = 'white', linewidth = 4) # before mean

  ax3.axvline(x=np.mean(x[half:,1]), color='red', linestyle='--', label='mean after manipulation') # after mean
  ax3.axvline(x=np.mean(x[half:,1]), linestyle='-', color = 'red', linewidth = 4) # after mean
  ax3.axvline(x=np.mean(x[half:,1]), linestyle='--', color = 'white', linewidth = 4) # after mean

  ax3.set_title("Advantaged HS GPA before & after manipulation", fontsize=13)
  ax3.set_xlim(0,4)
  ax3.set_xlabel('High school GPA (4.0 scale)',fontsize=14)
  ax3.set_ylabel('Number of applicants',fontsize=14)
  ax3.tick_params(axis="x", labelsize=14)
  ax3.tick_params(axis="y", labelsize=14)

  ax3.legend()


  ### 4) non-first-gen SAT
  ax4.hist(x[half:,0], bins='auto', label='after manipulation', color='green')

  ax4.hist(z[half:,0], bins='auto', label='before manipulation', color='lightgreen', alpha=0.75)

  ax4.axvline(x=np.mean(z[half:,0]), color='blue', linestyle='--', label='mean before manipulation') # before mean
  ax4.axvline(x=np.mean(z[half:,0]), linestyle='-', color = 'blue', linewidth = 4) # before mean
  ax4.axvline(x=np.mean(z[half:,0]), linestyle='--', color = 'white', linewidth = 4) # before mean

  ax4.axvline(x=np.mean(x[half:,0]), color='red', linestyle='--', label='mean after manipulation') # before mean
  ax4.axvline(x=np.mean(x[half:,0]), linestyle='-', color = 'red', linewidth = 4) # before mean
  ax4.axvline(x=np.mean(x[half:,0]), linestyle='--', color = 'white', linewidth = 4) # before mean

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

  fname = os.path.join(dirname, 'fg-ls_shifted_features.png')
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
  plt.errorbar(list(range(2,T+1,2)), np.mean(error_list_mean,axis=0)[:,0], yerr=np.std(error_list_mean,axis=0)[:,0], 
              color='darkorange', ecolor='wheat', label='OLS',elinewidth=10)
  plt.errorbar(list(range(2,T+1,2)), np.mean(error_list_mean,axis=0)[:,1], yerr=np.std(error_list_mean,axis=0)[:,1], 
              color='darkblue', ecolor='lightblue', label='2SLS',elinewidth=10)
  plt.ylim(0,.25)
  plt.xticks(fontsize=14)
  plt.yticks(fontsize=14)
  plt.xlim(50,T-11)

  plt.xlabel('Number of applicants (rounds)', fontsize=14)
  plt.ylabel(r'$|| \hat{\theta} - \theta^* ||$', fontsize=14)

  plt.plot(range(1,T+1), 1/np.sqrt(range(1,T+1)), color='red',linestyle='dashed', linewidth=2, label='1/sqrt(T)')

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
  plt.axvline(x=np.mean(y), linestyle='-', color = 'blue', linewidth = 4)
  plt.axvline(x=np.mean(y), linestyle='--', color = 'white', linewidth = 4)

  # disadvantaged
  plt.hist(y[0:half],bins='auto',label='disadvantaged', alpha=.85)
  plt.axvline(x=np.mean(y[0:half]),color='orange',linestyle='--', linewidth = 2, label='disadvantaged mean')
  plt.axvline(x=np.mean(y[0:half]), linestyle='-', color = 'orange', linewidth = 4)
  plt.axvline(x=np.mean(y[0:half]), linestyle='--', color = 'white', linewidth = 4)

  # advantaged
  plt.hist(y[half:],bins='auto',label='advantaged', alpha=0.7)
  plt.axvline(x=np.mean(y[half:]), linestyle='--', color = 'green', linewidth = 2, label='advantaged mean')
  plt.axvline(x=np.mean(y[half:]),color='green',linestyle='-', linewidth = 4)
  plt.axvline(x=np.mean(y[half:]), linestyle='--', color = 'white', linewidth = 4)

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
