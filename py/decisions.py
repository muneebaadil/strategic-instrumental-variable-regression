from typing import Iterable

import numpy as np

from py.agents_gen import clip_covariates, normalise_agents, DEFAULT_AGENTS_MODEL
from py.agents_gen import clip_outcomes, evaluate_agents, compute_percentile_admissions, \
  compute_random_admissions, AgentsGenericModel


class ThetaGenerator:
  def __init__(self, length: int, num_principals: int):
    self._T = length
    self._n = num_principals
    return

  def generate_randomly(self) -> np.ndarray:
    """
    Returns:
      np.ndarray: a (T,n,m) tensor of thetas.
    """
    T = self._T
    n = self._n

    thetas_tr = [np.array([])] * n  # empty list
    for i in range(n):
      thetas_tr[i] = np.random.multivariate_normal([1, 1 + i], [[10, 0], [0, 1]], size=T)

    return np.stack(thetas_tr).transpose((1, 0, 2))  # (T,n,m)

  def generate_scaled_duplicates(self, deploy_sd_every: int, mean_shift: int = 0) -> np.ndarray:
    """
    Args:
      deploy_sd_every (int):
      mean_shift (int):
    Returns:
      np.ndarray: a (T,n,m) tensor of thetas
    """

    def _distribute(n_rounds, theta, theta_scaled, deploy_sd_every):
      theta_temp = np.zeros((n_rounds, 2))
      j = 0
      for i in range(deploy_sd_every, int(n_rounds / 2), deploy_sd_every):
        theta_temp[j:j + deploy_sd_every] = theta[i - deploy_sd_every: i]
        theta_temp[j + deploy_sd_every: j + deploy_sd_every + deploy_sd_every] = (
          theta_scaled[i - deploy_sd_every: i]
        )
        j = j + 2 * deploy_sd_every

      # residual from the first vector
      theta_temp[j: j + int(n_rounds / 2) - i] = theta[i:]
      j = j + int(n_rounds / 2) - i

      # residual from the second vector
      theta_temp[j: j + int(n_rounds / 2) - i] = theta_scaled[i:]
      return theta_temp

    # start here
    T = self._T
    n = self._n

    thetas_tr = [np.array([])] * n  # empty list
    for i in range(n):
      theta = np.random.multivariate_normal([1, 1 + i + mean_shift], [[10, 0], [0, 1]], int(T / 2))
      assert theta.shape == (int(T / 2), 2)

      # scaled duplicate of each theta.
      scale = np.random.uniform(low=0, high=2, size=(int(T / 2),))
      scale = np.diag(v=scale)
      theta_scaled = scale.dot(theta)

      theta = _distribute(T, theta, theta_scaled, deploy_sd_every=deploy_sd_every)
      thetas_tr[i] = theta

    return np.stack(thetas_tr).transpose((1, 0, 2))  # (T,n,m)

  def generate_general_coop_case(self, num_cooperative_principals: int) -> np.ndarray:
    """
    When only a subset of principals follows the cooperative protocol.
    Args:
      num_cooperative_principals (int):
    Returns:
      np.ndarray
    """
    T = self._T
    n = self._n

    # check input
    num_free_principals = n - num_cooperative_principals
    assert num_free_principals >= 0

    # generate thetas of coop decision makers.
    thetas_tr = (
      ThetaGenerator(length=T, num_principals=num_cooperative_principals)
        .generate_scaled_duplicates(deploy_sd_every=1)
    ).transpose((1, 0, 2))  # (n,T,m)

    # generate thetas of non-coop decision makers.
    if num_free_principals >= 1:
      non_coop_thetas_tr = [None] * num_free_principals
      for i in range(num_free_principals):
        out = (ThetaGenerator(length=T, num_principals=1)
            .generate_scaled_duplicates(deploy_sd_every=(2 + i),
                                        mean_shift=num_cooperative_principals).transpose((1, 0, 2))  # (n,T,m)
        )
        non_coop_thetas_tr[i] = out

      thetas_tr = np.concatenate((thetas_tr, *non_coop_thetas_tr), axis=0)

    return thetas_tr.transpose((1, 0, 2))  # (T,n,m)

  @staticmethod
  def intra_round_repeat(thetas_tr: np.ndarray, repeats: int) -> np.ndarray:
    """
    When each round has 's' agents arriving.
    Args:
      thetas_tr (np.ndarray): a (T,n,m) tensor of thetas.
      repeats (int): e.g., repeat 's' times within each round.
    Returns:
      np.ndarray: a (Txs,n,m) tensor of thetas.
    """
    return np.repeat(thetas_tr, repeats=repeats, axis=0)


class Simulator:
  def __init__(self,
               num_agents: int,
               has_same_effort: bool, does_clip: bool, does_normalise: bool, ranking_type: str,
               agents_model: AgentsGenericModel = DEFAULT_AGENTS_MODEL):

    self._num_agents = num_agents  # per round
    self._has_same_effort = has_same_effort
    self._does_clip = does_clip
    self._does_normalise = does_normalise

    self._agents_model = agents_model

    if ranking_type == 'prediction':
      self._selection_func = compute_percentile_admissions
    elif ranking_type == 'uniform':
      self._selection_func = compute_random_admissions
    else:
      raise ValueError(ranking_type)

    self.u = None
    self.b_tr = None
    self.thetas_tr = None
    self.x_tr = None
    self.eet_mean = None
    self.y_hat = None
    self.w_tr = None
    self.z = None

    self.o = None
    self.y = None

    return

  def deploy(self, thetas_tr: np.ndarray, gammas: Iterable, admission_rates: Iterable) -> None:
    """
    Args:
      thetas_tr (np.ndarray): a (T,n,m) matrix of 'non-repeated' thetas.
      gammas (Iterable): a (n,) vector of gammas.
      admission_rates (Iterable): a (n,) vector of admission rates.
    Returns:
      None
    """
    gammas = np.array(gammas)
    admission_rates = np.array(admission_rates)

    # check the dimensions
    T, n, m = thetas_tr.shape
    assert n == gammas.shape[0]
    assert n == admission_rates.shape[0]

    # init params
    s = self._num_agents
    has_same_effort = self._has_same_effort
    does_clip = self._does_clip
    does_normalise = self._does_normalise
    selection_func = self._selection_func
    am = self._agents_model

    # agents are spawned
    u, b_tr, e_stack = am.gen_base_agents(length=(T * s), has_same_effort=has_same_effort)
    b_tr = clip_covariates(b_tr) if does_clip else b_tr

    # compute average thetas
    # (n,) and (n,T,m) -> (T,m)
    avg_theta_tr = (gammas.reshape((n, 1, 1)) * thetas_tr.transpose((1, 0, 2))).sum(axis=0)

    # release thetas & average thetas
    thetas_tr = ThetaGenerator.intra_round_repeat(thetas_tr=thetas_tr, repeats=s)  # (Txs,n,m)
    avg_theta_tr = ThetaGenerator.intra_round_repeat(thetas_tr=avg_theta_tr, repeats=s)  # (Txs,m)

    # agents take strategic actions
    x_tr = am.gen_covariates(b_tr=b_tr, e_stack=e_stack, avg_theta_tr=avg_theta_tr)
    x_tr = clip_covariates(x_tr) if does_clip else x_tr

    # normalise agents' data
    e_mean = e_stack.mean(axis=0)
    eet_mean = e_mean.dot(e_mean.T)
    if does_normalise:
      b_tr, x_tr, eet_mean = normalise_agents(b_tr=b_tr, x_tr=x_tr, eet_mean=eet_mean)

    # predict agents' performance
    x_norm = (x_tr - x_tr.mean(axis=0, keepdims=True)) / x_tr.std(axis=0, keepdims=True)
    y_hat = evaluate_agents(x_tr=x_norm, thetas_tr=thetas_tr)  # (Txs,n)

    # send them admission offers
    w = np.zeros(shape=(n, T * s))
    for i in range(n):
      for j in range(T * s):
        idx1 = j * s  # first index in this batch of agents
        idx2 = (j + 1) * s  # first index of the next batch

        # admissions are computed within each batch (i.e., each round).
        w[i, idx1:idx2] = selection_func(y_hat=y_hat[idx1:idx2, i], p=admission_rates[i])

    # agents comply to at most one principal
    z = am.realise_enrollments(w_tr=w.T, gammas_tr=np.tile(gammas, reps=(T * s, 1)))

    # assignment
    self.u = u  
    self.b_tr = b_tr
    self.thetas_tr = thetas_tr  
    self.x_tr = x_tr
    self.eet_mean = eet_mean
    self.y_hat = y_hat
    self.w_tr = w.T
    self.z = z

    return

  def enroll(self, theta_stars_tr: Iterable) -> None:
    """
    Counterfactually/Potentially enroll agents into environments and get their outcomes.
    Args:
      theta_stars_tr (Iterable): a (n,m) matrix of 'n' theta_stars.
    Returns:
      None
    """
    theta_stars_tr = np.array(theta_stars_tr)

    # init params
    u = self.u
    x_tr = self.x_tr
    does_clip = self._does_clip
    am = self._agents_model

    # start here
    o, y = am.gen_outcomes(u=u, x_tr=x_tr, theta_stars_tr=theta_stars_tr)
    y = clip_outcomes(y) if does_clip else y

    # assignment
    self.o = o
    self.y = y

    return
