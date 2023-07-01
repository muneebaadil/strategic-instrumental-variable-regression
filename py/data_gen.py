# %%
import argparse
from typing import Tuple, List

import numpy as np

from py.agents_gen import gen_base_agents, gen_covariates, gen_outcomes, \
  compute_percentile_admissions, compute_random_admissions, realise_enrollments
from py.decisions import ThetaGenerator, Simulator
from py.utils import normalize


def sample_effort_conversion(EW, n_samples, adv_idx, fixed_effort_conversion):
  assert adv_idx.max() < n_samples
  EWi = np.zeros(shape=(n_samples, 2, 2))

  for i in range(n_samples):
    EWi[i] = EW.copy()
    if not fixed_effort_conversion:
      noise_mean = [0.5, 0, 0, 0.1]
      noise_cov = [[0.25, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.01]]

      noise = np.random.multivariate_normal(noise_mean, noise_cov).reshape((2, 2))
      if i in adv_idx:
        EWi[i] += noise
      else:
        EWi[i] -= noise
  return EWi


def distribute(n_rounds, theta, theta_scaled, deploy_sd_every):
  theta_temp = np.zeros((n_rounds, 2))
  j = 0
  for i in range(deploy_sd_every, int(n_rounds / 2), deploy_sd_every):
    theta_temp[j:j + deploy_sd_every] = theta[i - deploy_sd_every: i]
    theta_temp[j + deploy_sd_every: j + deploy_sd_every + deploy_sd_every] = theta_scaled[
                                                                             i - deploy_sd_every: i]

    j = j + 2 * deploy_sd_every

  # residual from the first vector
  theta_temp[j: j + int(n_rounds / 2) - i] = theta[i:]
  j = j + int(n_rounds / 2) - i

  # residual from the second vector
  theta_temp[j: j + int(n_rounds / 2) - i] = theta_scaled[i:]
  return theta_temp


def generate_theta(i: int, args: argparse.Namespace, deploy_sd_every: int) -> np.ndarray:
  assert args.num_applicants % args.applicants_per_round == 0
  n_rounds = int(args.num_applicants / args.applicants_per_round)

  if args.scaled_duplicates is None:  # random vectors for every round
    theta = np.random.multivariate_normal([1, 1 + i], [[10, 0], [0, 1]], n_rounds)

  elif args.scaled_duplicates == 'sequence':  # making sure there exists a scaled duplicate of each theta per round
    theta = np.random.multivariate_normal([1, 1 + i], [[10, 0], [0, 1]], int(n_rounds / 2))
    assert theta.shape == (int(n_rounds / 2), 2)

    # scaled duplicate of each theta.
    scale = np.random.uniform(low=0, high=2, size=(int(n_rounds / 2),))
    scale = np.diag(v=scale)
    theta_scaled = scale.dot(theta)

    theta = distribute(n_rounds, theta, theta_scaled, deploy_sd_every)

  # theta repeating over the rounds.
  theta = np.repeat(theta, repeats=args.applicants_per_round, axis=0)
  assert theta.shape[0] == args.num_applicants
  return theta


def generate_bt(n_samples: int, sigma_sat: float, sigma_gpa: float, args: argparse.Namespace) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  assert n_samples % 2 == 0, f"{n_samples} not divisible by 2"

  half = int(n_samples / 2)
  b = np.zeros([n_samples, 2])

  # indices for shuffling
  idx = np.arange(n_samples)
  np.random.shuffle(idx)
  disadv_idx = idx[:half]
  adv_idx = idx[half:]

  mean_sat_disadv = 800
  mean_sat_adv = 1000

  mean_gpa_disadv = 1.8
  mean_gpa_adv = mean_gpa_disadv * 1.25

  # disadvantaged students
  b[disadv_idx, 0] = np.random.normal(mean_sat_disadv, sigma_sat, b[disadv_idx][:, 0].shape)  # SAT
  b[disadv_idx, 1] = np.random.normal(mean_gpa_disadv, sigma_gpa, b[disadv_idx][:, 1].shape)  # GPA

  # advantaged students
  b[adv_idx, 0] = np.random.normal(mean_sat_adv, sigma_sat, b[adv_idx][:, 0].shape)  # SAT
  b[adv_idx, 1] = np.random.normal(mean_gpa_adv, sigma_gpa, b[adv_idx][:, 1].shape)  # GPA

  if args.clip:
    b[:, 0] = np.clip(b[:, 0], 400, 1600)  # clip to 400 to 1600
    b[:, 1] = np.clip(b[:, 1], 0, 4)  # clip to 0 to 4.0

  # confounding error term g (error on true college GPA)
  g = np.ones((args.num_applicants, args.num_envs)) * 0.5  # legacy students shifted up
  g[disadv_idx, :] = -0.5  # first-gen students shifted down
  g += np.random.normal(1, 0.2, size=(args.num_applicants, args.num_envs))  # non-zero-mean
  return b, g, adv_idx, disadv_idx


def compute_xt(EWi: np.ndarray, b: np.ndarray, theta: np.ndarray, pref_vect: np.ndarray,
               args: argparse.Namespace) -> np.ndarray:
  assert EWi.shape[0] == b.shape[0]
  assert b.shape[0] == theta.shape[1]

  assert EWi.shape == (args.num_applicants, 2, 2)
  assert theta.shape == (args.num_envs, args.num_applicants, 2)
  assert pref_vect.shape == (args.num_envs,)

  eet = np.matmul(EWi, np.transpose(EWi, axes=(0, 2, 1)))  # (n_appl, 2, 2)
  thetai = np.matmul(np.transpose(theta, axes=(1, -1, 0)), pref_vect)  # (n_appl, 2 )
  impr = np.matmul(eet, thetai[:, :, np.newaxis])[:, :, 0]  # (n_appl, 2)

  x = b + impr

  if args.clip:
    x[:, 0] = np.clip(x[:, 0], 400, 1600)  # clip to 400 to 1600
    x[:, 1] = np.clip(x[:, 1], 0, 4)  # clip to 0 to 4.0
  return x


def _get_selection(_y_hat, n_rounds, accept_rate, rank_type, applicants_per_round):
  _w = np.zeros_like(_y_hat)
  # comparing applicants coming in the same rounds.
  for r in range(n_rounds):
    _y_hat_r = _y_hat[r * applicants_per_round: (r + 1) * applicants_per_round]

    w_r = get_selection(_y_hat_r, accept_rate, rank_type)
    _w[r * applicants_per_round: (r + 1) * applicants_per_round] = w_r
  return _w


def get_selection(y_hat, accept_rate, rank_type):
  if rank_type == 'prediction':
    sort_idx = np.argsort(y_hat)

    thres = int((1 - accept_rate) * sort_idx.size)
    rejected_idx = sort_idx[:thres]
    accepted_idx = sort_idx[thres:]

    w_test = np.zeros_like(y_hat)
    w_test[accepted_idx] = True
    w_test[rejected_idx] = False

  elif rank_type == 'uniform':
    w_test = np.zeros_like(y_hat)
    idx = np.arange(y_hat.size)
    np.random.shuffle(idx)
    idx = idx[int((1 - accept_rate) * y_hat.size):]
    w_test = np.zeros_like(y_hat)
    w_test[idx] = True
  return w_test


def generate_thetas(args: argparse.Namespace) -> np.ndarray:
  deploy_sd_every = 2
  thetas = []
  for i in range(args.num_envs):
    if i < args.num_cooperative_envs:  # cooperative env.
      thetas.append(generate_theta(i, args, 1))
    else:  # non-cooperative env.
      thetas.append(generate_theta(i, args, deploy_sd_every))
      deploy_sd_every += 1

  thetas = np.stack(thetas)
  return thetas


# for notebook.
def generate_data(num_applicants: int, applicants_per_round: int, fixed_effort_conversion: bool,
                  args: argparse.Namespace, _theta=None, ) \
-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
             np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  assert num_applicants % applicants_per_round == 0
  n_rounds = int(num_applicants / applicants_per_round)

  theta_star = np.zeros(shape=(args.num_envs, 2))
  theta_star[:, 1] = np.random.normal(loc=0.5, scale=args.theta_star_std, size=(args.num_envs,))

  sigma_sat = 200
  sigma_gpa = 0.5

  b, g, adv_idx, disadv_idx = generate_bt(num_applicants, sigma_sat, sigma_gpa, args)

  # assessment rule
  if _theta is None: # distribute randomly.
    theta = generate_thetas(args)
  else: # set as given
    assert _theta.shape == (args.num_envs, 2)
    theta = np.copy(_theta)
    theta = theta[:, np.newaxis]
    theta = np.repeat(theta, repeats=num_applicants, axis=1)

  # effort conversion matrices
  EW = np.array([[10.0, 0], [0, 1.0]])
  EWi = sample_effort_conversion(EW, num_applicants, adv_idx, fixed_effort_conversion)

  # observable features x
  x = compute_xt(EWi, b, theta, args.pref_vect, args)

  EET = EW.dot(EW.T)
  # normalize
  if args.normalize:
    (b[:, 0], x[:, 0]), scale1 = normalize((b[:, 0], x[:, 0]), new_min=400, new_max=1600)
    (b[:, 1], x[:, 1]), scale2 = normalize((b[:, 1], x[:, 1]), new_min=0, new_max=4)

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
  x_norm = (x - x.mean(axis=0, keepdims=True)) / x.std(axis=0, keepdims=True)
  x_norm = x_norm[np.newaxis]
  y_hat_logits = (x_norm * theta)
  # y_hat_logits = y_hat_logits - y_hat_logits.mean(axis=1, keepdims=True)
  # y_hat_logits = y_hat_logits / y_hat_logits.std(axis=1, keepdims=True)
  y_hat = y_hat_logits.sum(axis=-1)

  w = np.zeros((args.num_envs, num_applicants))
  for env_idx in range(args.num_envs):
    w[env_idx] = _get_selection(
      y_hat[env_idx], n_rounds, args.envs_accept_rates[env_idx], args.rank_type,
      args.applicants_per_round
    )

  z = np.zeros((args.num_applicants,))
  for idx in range(args.num_applicants):
    w_idx = w[:, idx]
    if w_idx.sum() == 0:  # applicant is not accepted anywhere
      z[idx] = 0
    else:
      pvals = w_idx * args.pref_vect / (w_idx * args.pref_vect).sum()
      temp = np.random.multinomial(n=1, pvals=pvals, size=1)
      _idx = temp.flatten().nonzero()[0]
      z[idx] = _idx + 1  # offset to avoid conflict with "no uni" decision
  return b, x, y, EET, theta, w, z, y_hat, adv_idx, disadv_idx, g.T, theta_star, args.pref_vect


def generate_data_v2(num_applicants: int, applicants_per_round: int, fixed_effort_conversion: bool,
                     args: argparse.Namespace, _theta: np.ndarray = None) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
             np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  theta_star = np.zeros(shape=(args.num_envs, 2))
  theta_star[:, 1] = np.random.normal(loc=0.5, scale=args.theta_star_std, size=(args.num_envs,))

  assert num_applicants % applicants_per_round == 0
  n_rounds = int(num_applicants / applicants_per_round)

  # assessment rule
  if _theta is None:  # distribute randomly.
    thegen = ThetaGenerator(length=n_rounds, num_principals=args.num_envs)
    if args.scaled_duplicates is None:
      theta = thegen.generate_randomly()
    elif args.scaled_duplicates == 'sequence':
      theta = thegen.generate_general_coop_case(num_cooperative_principals=args.num_cooperative_envs)
    else:
      raise ValueError(args.scaled_duplicates)
  else:  # set as given
    raise NotImplementedError("Use Simulator instead!")

  sim = Simulator(
    num_agents=applicants_per_round, has_same_effort=fixed_effort_conversion,
    does_clip=args.clip, does_normalise=args.normalize,
    ranking_type=args.rank_type
  )
  sim.deploy(thetas_tr=theta,gammas=args.pref_vect, admission_rates=args.envs_accept_rates)
  u, b_tr, theta, x_tr, eet_mean = sim.u, sim.b_tr, sim.thetas_tr, sim.x_tr, sim.eet_mean

  # true outcomes (college gpa)
  sim.enroll(theta_stars_tr=theta_star)
  o, y = sim.o, sim.y

  # for backwards compatibility
  theta = theta.transpose((1,0,2))
  y = y.T

  assert x_tr[np.newaxis].shape == (1, args.num_applicants, 2)
  assert theta.shape == (args.num_envs, args.num_applicants, 2)
  assert o.shape == (args.num_applicants, args.num_envs)
  assert theta_star.shape == (args.num_envs, 2)

  # our setup addition
  # computing admission results.
  y_hat = sim.y_hat.T
  w, z = sim.w_tr.T, sim.z

  # for backwards compatibility
  adv_idx = np.where(u is True)
  disadv_idx = np.where(u is False)

  return b_tr, x_tr, y, eet_mean, theta, w, z, y_hat, adv_idx, disadv_idx, o.T, theta_star, args.pref_vect
