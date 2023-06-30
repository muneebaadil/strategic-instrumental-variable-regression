import numpy as np

from py.agents_gen import gen_base_agents, clip_covariates, gen_covariates, normalise_agents


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

    # TODO: is 'mean_shift' necessary?

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

  def generate_general_coop_case(self, num_cooperative_principals: int):
    """
    When only a subset of principals follows the cooperative protocol.

    Args:
      num_cooperative_principals (int):

    Returns:

    """
    T = self._T
    n = self._n

    # check input
    num_free_principals = n - num_cooperative_principals
    assert num_free_principals >= 1

    coop_thetas_tr = (
      ThetaGenerator(length=T, num_principals=num_cooperative_principals)
        .generate_scaled_duplicates(deploy_sd_every=1)
    ).transpose((1, 0, 2))  # (n,T,m)

    non_coop_thetas_tr = [] * num_free_principals
    for i in range(num_free_principals):
      non_coop_thetas_tr[i] = (
        ThetaGenerator(length=T, num_principals=1)
          .generate_scaled_duplicates(deploy_sd_every=(2 + i),  # TODO: is shifting this necessary?
                                      mean_shift=num_cooperative_principals)
      ).transpose((1, 0, 2))  # (n,T,m)

    # (T,n,m)
    thetas_tr = np.concatenate((coop_thetas_tr, *non_coop_thetas_tr), axis=0).transpose((1, 0, 2))

    return thetas_tr

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
  def __init__(self, num_agents: int, has_same_effort: bool, does_clip: bool, does_normalise: bool):
    self._num_agents = num_agents  # per round
    self._has_same_effort = has_same_effort
    self._does_clip = does_clip
    self._does_normalise = does_normalise

    self.u = None
    self.b_tr = None
    self.thetas_tr = None
    self.x_tr = None
    self.eet_mean = None
    return

  def deploy(self, thetas_tr: np.ndarray, gammas: np.ndarray):
    """
    Args:
      thetas_tr (np.ndarray): a (T,n,m) matrix of thetas.
      gammas (np.ndarray): a (n,) vector of gammas.

    Returns:

    """
    # check the dimensions
    T, n, m = thetas_tr.shape
    assert n == gammas.shape[0]

    # init params
    s = self._num_agents
    has_same_effort = self._has_same_effort
    does_clip = self._does_clip
    does_normalise = self._does_normalise

    # agents are spawned
    u, b_tr, e = gen_base_agents(length=(T * s), has_same_effort=has_same_effort)
    b_tr = clip_covariates(b_tr) if does_clip else b_tr

    # compute average thetas
    # (n,) and (n,T,m) -> (T,m)
    avg_theta_tr = (gammas.reshape((n, 1, 1)) * thetas_tr.transpose((1, 0, 2))).sum(axis=0)

    # release thetas & average thetas
    thetas_tr = ThetaGenerator.intra_round_repeat(thetas_tr=thetas_tr, repeats=s)  # (Txs,n,m)
    avg_theta_tr = ThetaGenerator.intra_round_repeat(thetas_tr=avg_theta_tr, repeats=s)  # (Txs,m)

    # agents take strategic actions
    x_tr = gen_covariates(b_tr=b_tr, e=e, avg_theta_tr=avg_theta_tr)
    x_tr = clip_covariates(x_tr) if does_clip else x_tr

    # normalise agents' data
    e_mean = e.mean(axis=0)
    eet_mean = e_mean.dot(e_mean.T)
    if does_normalise:
      b_tr, x_tr, eet_mean = normalise_agents(b_tr=b_tr, x_tr=x_tr, eet_mean=eet_mean)

    # TODO: add admissions and enrollments

    # assignment
    self.u = u  # TODO: to be removed
    self.b_tr = b_tr
    self.thetas_tr = thetas_tr  # TODO: to be removed
    self.x_tr = x_tr
    self.eet_mean = eet_mean

    return

  def enroll(self, theta_stars_tr: np.ndarray):
    return
