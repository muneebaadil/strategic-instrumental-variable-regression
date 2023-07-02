"""
To generate data for agents.
Specifically, the CSL setting with 2-dimensional covariates and with linear mechanisms.

Might be coupled with environments/decision makers:
- In generating noise.
- In 'not' generating gammas alongside base agents.
"""

import copy
from typing import Tuple, Iterable

import numpy as np
from sklearn.preprocessing import normalize as skl_normalize

from py.utils import normalize


def clip_covariates(x_tr: np.ndarray) -> np.ndarray:
  """
  This works for both base covariates (i.e., b) and improved covariates (i.e., X).

  Args:
    x_tr (np.ndarray): a (T,m) matrix of covariates.

  Returns:
    np.ndarray: a (T,m) matrix of clipped covariates.
  """
  x_tr = copy.deepcopy(x_tr)
  x_tr[:, 0] = np.clip(x_tr[:, 0], 400, 1600)  # clip to 400 to 1600
  x_tr[:, 1] = np.clip(x_tr[:, 1], 0, 4)  # clip to 0 to 4.0
  return x_tr


def clip_outcomes(y: np.ndarray):
  return np.clip(y, 0, 4)  # GPA


def normalise_agents(b_tr: np.ndarray, x_tr: np.ndarray, eet_mean: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """
  Args:
    b_tr (np.ndarray): a (T,2) matrix of base covariates.
    x_tr (np.ndarray): a (T,2) matrix of covariates.
    eet_mean (np.ndarray): a (2,2) matrix of the 'mean' of effort.
  Returns:
    Tuple[np.ndarray,np.ndarray,np.ndarray]: two (T,2) matrices and one (2,2) matrix.
  """

  # start here
  b_tr = copy.deepcopy(b_tr)
  x_tr = copy.deepcopy(x_tr)
  (b_tr[:, 0], x_tr[:, 0]), scale1 = normalize([b_tr[:, 0], x_tr[:, 0]], new_min=400, new_max=1600)
  (b_tr[:, 1], x_tr[:, 1]), scale2 = normalize([b_tr[:, 1], x_tr[:, 1]], new_min=0, new_max=4)

  # scale EET accordingly.
  eet_mean = np.matmul(np.array([[scale1, 0], [0, scale2]]), eet_mean)

  return b_tr, x_tr, eet_mean


def evaluate_agents(x_tr: np.ndarray, thetas_tr: np.ndarray) -> np.ndarray:
  """
  Args:
    x_tr (np.ndarray): a (T,m) matrix of covariates.
    thetas_tr (np.ndarray): a (T,n,m) tensor of thetas.
  Returns:
    np.ndarray: a (T,n) matrix of predictions.
  """
  # check the dimensions
  (T, n, m) = thetas_tr.shape
  assert (T, m) == x_tr.shape

  # (T,m) & (T,n,m) -> (T,n)
  y_hat = np.einsum('ij,ikj->ik', x_tr, thetas_tr)

  # verify the dimensions
  assert (T, n) == y_hat.shape

  return y_hat


def compute_percentile_admissions(y_hat: np.ndarray, p: float) -> np.ndarray:
  """
  Args:
    y_hat (np.ndarray): a (T,) vector containing predicted outcomes.
    p (float): the acceptance rate.

  Returns:
    np.ndarray: a (T,) vector containing Boolean-typed admission statuses.
  """
  # check the dimensions
  assert len(y_hat.shape) == 1

  cdf = y_hat.argsort() / len(y_hat)
  admissions: np.ndarray = (cdf >= (1 - p))
  return admissions


def compute_random_admissions(y_hat: np.ndarray, p: float) -> np.ndarray:
  """
  Args:
    y_hat (np.ndarray): a (T,) vector containing predicted outcomes.
    p (float): the acceptance rate.

  Returns:
    np.ndarray: a (T,) vector containing Boolean-typed admission statuses.
  """
  # check the dimensions
  assert len(y_hat.shape) == 1

  return np.random.choice([True, False], size=len(y_hat), p=[p, 1 - p])


class AgentsGenericModel:
  """
  To generate any bunch of agents following a predefined setup which can be customised.
  See the variable 'DEFAULT_AGENTS_MODEL' below this class for a usage example.
  """

  def __init__(self,
               group_0_base_mean: Iterable, group_0_base_cov: Iterable,
               group_1_base_mean: Iterable, group_1_base_cov: Iterable,
               group_0_outcome_mean_shift: float, group_0_outcome_std: float,
               group_1_outcome_mean_shift: float, group_1_outcome_std: float):

    # np.random.multivariate_normal
    self.b_dis_0 = {'mean': group_0_base_mean, 'cov': group_0_base_cov}
    self.b_dis_1 = {'mean': group_1_base_mean, 'cov': group_1_base_cov}

    # np.random.normal
    self.o_dis_0 = {'loc': group_0_outcome_mean_shift, 'scale': group_0_outcome_std}
    self.o_dis_1 = {'loc': group_1_outcome_mean_shift, 'scale': group_1_outcome_std}

    return

  def gen_base_agents(self, length: int, has_same_effort: bool) \
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
    # init params
    b_dis_0 = self.b_dis_0
    b_dis_1 = self.b_dis_1
    T = length

    # binary confounder, where value '1' stands for advantaged agents
    u = np.random.choice([0, 1], size=T)

    b_tr = (
        (1 - u).reshape(-1, 1) * np.random.multivariate_normal(**b_dis_0, size=T)
        +
        u.reshape(-1, 1) * np.random.multivariate_normal(**b_dis_1, size=T)
    )

    top_noise = np.zeros(T)
    bottom_noise = np.zeros(T)
    if not has_same_effort:
      multipliers = (u - 0.5) * 2
      top_noise = multipliers * np.random.normal(loc=0.5, scale=0.5, size=T)
      bottom_noise = multipliers * np.random.normal(loc=0.1, scale=0.1, size=T)

    e_stack = np.tile([[10.0, 0], [0, 1.0]], reps=(T, 1, 1))
    e_stack[:, 0, 0] += top_noise
    e_stack[:, 1, 1] += bottom_noise

    return u, b_tr, e_stack

  @staticmethod
  def gen_covariates(b_tr: np.ndarray, e_stack: np.ndarray, avg_theta_tr: np.ndarray) -> np.ndarray:
    """
    This is static because the strategic behaviour is the same, conditioned on
    the same agents' base profile (i.e., b and e).

    Args:
      b_tr (np.ndarray): a (T,m) matrix of 'T' base covariates
      e_stack (np.ndarray): a (T,m,m) collection of 'T' effort matrices.
      avg_theta_tr (np.ndarray): a (T,m) matrix of 'T' avg_thetas.

    Returns:
      np.ndarray: a (T,m) matrix of 'T' (improved) covariates
    """

    # check the dimensions
    T, m = b_tr.shape
    assert e_stack.shape == (T, m, m)
    assert avg_theta_tr.shape == (T, m)

    # G is a collection of eeT
    G = e_stack @ e_stack.transpose((0, 2, 1))

    # (T,m) & (T,m,m) -> (T,m)
    # a vertical stack of individual row vectors,
    # where each row can be expressed as: x_tr = b_tr + avg_theta_tr(eeT)
    x_tr: np.ndarray = b_tr + np.einsum('ij,ikj->ij', avg_theta_tr, G)

    # verify the dimensions
    assert (T, m) == x_tr.shape

    return x_tr

  def gen_outcomes(self, u: np.ndarray, x_tr: np.ndarray, theta_stars_tr: np.ndarray) \
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
    # init params
    o_dis_0 = self.o_dis_0
    o_dis_1 = self.o_dis_1

    # check the dimensions
    T = u.shape[0]
    n, m = theta_stars_tr.shape
    assert x_tr.shape == (T, m)

    o = (
        (1 - u) * np.random.normal(**o_dis_0, size=T)
        + u * np.random.normal(**o_dis_1, size=T)
    )
    o = np.tile(o.reshape(-1, 1), n)  # all envs have the same noise-generating mechanism
    y = x_tr @ theta_stars_tr.T + o

    return o, y

  @staticmethod
  def realise_enrollments(w_tr: np.ndarray, gammas_tr: np.ndarray) -> np.ndarray:
    """
    This is static because no customisation is implemented for now.

    Args:
      w_tr (np.ndarray): a (T,n) matrix of admissions.
      gammas_tr (np.ndarray): a (T,n) matrix of preferences.
    Returns:
      np.ndarray: a (T,) vector of enrollments.
    """

    # check the dimensions
    T, n = w_tr.shape
    assert (T, n) == gammas_tr.shape

    probs_tr = skl_normalize(w_tr * gammas_tr, axis=1, norm='l1')  # auto handles the zero case.

    z = np.zeros(T)
    for i in range(T):
      p = probs_tr[i]
      if p.sum() > 0:  # only for agents with at least 1 admission offer.
        z[i] = np.random.choice(n, p=p) + 1  # offset to avoid conflict with "no uni" decision

    return z


DEFAULT_AGENTS_MODEL = AgentsGenericModel(
  group_0_base_mean=[800, 1.8], group_0_base_cov=[[200 ** 2, 0], [0, 0.5 ** 2]],
  group_1_base_mean=[1000, 2.25], group_1_base_cov=[[200 ** 2, 0], [0, 0.5 ** 2]],
  group_0_outcome_mean_shift=0.5, group_0_outcome_std=0.2,
  group_1_outcome_mean_shift=1.5, group_1_outcome_std=0.2
)
