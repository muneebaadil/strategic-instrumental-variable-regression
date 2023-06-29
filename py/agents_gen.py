"""
To generate data for agents.
Might be coupled with environments/decision makers:
- In generating noise.
"""

from typing import Tuple

import numpy as np
from sklearn.preprocessing import normalize


def gen_base_agents(length: int, has_same_effort: bool) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Args:
    length (int): number of agents to be generated.
    has_same_effort (bool): 'True' if everyone has the same effort conversion matrix.
  Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a (T,) vector of binary confounders,
        a (T,2) matrix of 'T' base covariates,
        a (T,2,2) collection of 'T' conversion matrices.
  """
  T = length

  # binary confounder, where value '1' stands for advantaged agents
  u = np.random.choice([0, 1], size=T)

  b_tr = (
      (1 - u).reshape(-1, 1) * np.random.multivariate_normal([800, 1.8],
                                                             [[200 ** 2, 0], [0, 0.5 ** 2]],
                                                             size=T)
      +
      u.reshape(-1, 1) * np.random.multivariate_normal([1000, 2.25],
                                                       [[200 ** 2, 0], [0, 0.5 ** 2]],
                                                       size=T)
  )

  top_noise = np.zeros(T)
  bottom_noise = np.zeros(T)
  if not has_same_effort:
    multipliers = (u - 0.5) * 2
    top_noise = multipliers * np.random.normal(loc=0.5, scale=0.5, size=T)
    bottom_noise = multipliers * np.random.normal(loc=0.1, scale=0.1, size=T)

  e = np.tile([[10.0, 0], [0, 1.0]], reps=(T, 1, 1))
  e[:, 0, 0] += top_noise
  e[:, 1, 1] += bottom_noise

  return u, b_tr, e


def gen_covariates(b_tr: np.ndarray, e: np.ndarray, avg_theta_tr: np.ndarray) -> np.ndarray:
  """
  Args:
    b_tr (np.ndarray): a (T,m) matrix of 'T' base covariates
    e (np.ndarray): a (T,m,m) collection of 'T' effort matrices.
    avg_theta_tr (np.ndarray): a (T,m) matrix of 'T' avg_thetas.

  Returns:
    np.ndarray: a (T,m) matrix of 'T' (improved) covariates
  """

  # check the dimensions
  T, m = b_tr.shape
  assert e.shape == (T, m, m)
  assert avg_theta_tr.shape == (T, m)

  # G is a collection of eeT
  G = e @ e.transpose((0, 2, 1))

  # a vertical stack of individual row vectors,
  # where each row can be expressed as: x_tr = b_tr + avg_theta_tr(eeT)
  x_tr = b_tr + np.einsum('ij,ikj->ij', avg_theta_tr, G)

  return x_tr


def gen_outcomes(u: np.ndarray, x_tr: np.ndarray, theta_stars_tr: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
  """
  Counterfactually/Potentially enroll agents into environments and get their outcomes.

  Args:
    u (np.ndarray): a (T,) vector of binary confounders.
    x_tr (np.ndarray): a (T,m) matrix of 'T' covariates.
    theta_stars_tr (np.ndarray): a (n,m) matrix of 'n' theta_stars.

  Returns:
    Tuple[np.ndarray, np.ndarray]: an (T,n) matrix of noise and an (T,n) matrix of outcomes.
  """

  # check the dimensions
  T = u.shape[0]
  n, m = theta_stars_tr.shape
  assert x_tr.shape == (T, m)

  o = (
      (1 - u) * np.random.normal(loc=0.5, scale=0.2, size=T)
      + u * np.random.normal(loc=1.5, scale=0.2, size=T)
  )
  o = np.tile(o.reshape(-1, 1), n)  # all envs have the same noise-generating mechanism
  y = x_tr @ theta_stars_tr.T + o

  return o, y


def compute_percentile_admissions(y_hat: np.ndarray, p: float) -> np.ndarray:
  """
  Args:
    y_hat (np.ndarray): a (T,) vector containing predicted outcomes.
    p (float): the acceptance rate.

  Returns:
    np.ndarray: a (T,) vector containing admission statuses.
  """
  cdf = y_hat.argsort() / len(y_hat)
  admissions: np.ndarray = (cdf >= (1 - p))
  return admissions


def compute_random_admissions(length: int, p: float) -> np.ndarray:
  """
  Args:
    length (int): the number of candidate agents.
    p (float): the acceptance rate.

  Returns:
    np.ndarray: a (T,) vector containing admission statuses.
  """
  return np.random.choice([True, False], size=length, p=[p, 1 - p])


def realise_enrollments(w_tr: np.ndarray, gammas_tr: np.ndarray) -> np.ndarray:
  """
  Args:
    w_tr (np.ndarray): a (T,n) matrix of admissions.
    gammas_tr (np.ndarray): a (T,n) matrix of preferences.

  Returns:

  """

  # check the dimensions
  T, n = w_tr.shape
  assert T, n == gammas_tr.shape[0]

  probs_tr = normalize(w_tr * gammas_tr, axis=1, norm='l1')  # auto handles the zero case.

  z = np.zeros(T)
  for i in range(T):
    p = probs_tr[i]
    if p.sum() > 0:  # only for agents with at least 1 admission offer.
      z = np.random.choice(n, p=p) + 1  # offset to avoid conflict with "no uni" decision

  return z
