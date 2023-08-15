import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize as sk_normalize

from py.agents_gen import AgentsGenericModel
from py.decisions import ThetaGenerator, Simulator

OUT_DIR = "./out"


def run_utility_exp():
  # params
  T = 20
  s = 1000
  gammas = [0.45, 0.55]
  admission_rates = [0.6, 0.6]
  theta_stars_tr = [[0.0001, 0.5], [0.4, 0.6]]

  # customise my agents' model
  # am = DEFAULT_AGENTS_MODEL
  am = AgentsGenericModel(
    group_0_base_mean=[800, 0.8], group_0_base_cov=[[200 ** 2, 0], [0, 0.5 ** 2]],
    group_1_base_mean=[1000, 2.25], group_1_base_cov=[[200 ** 2, 0], [0, 0.5 ** 2]],
    group_0_outcome_mean_shift=200.5, group_0_outcome_std=0.2,
    group_1_outcome_mean_shift=1.5, group_1_outcome_std=0.2
  )

  # generate thetas for regression
  my_thetas_tr = ThetaGenerator(length=T, num_principals=1).generate_randomly()  # (T,1,2)
  your_thetas_tr = np.tile([1, 1], reps=(T, 1, 1))  # (T,1,2)
  delpoyed_thetas_tr = np.concatenate([my_thetas_tr, your_thetas_tr], axis=1)  # (T,2,2)

  # deploy
  sim = Simulator(num_agents=s, has_same_effort=True, does_clip=False,
                  does_normalise=False, ranking_type='prediction', agents_model=am)
  sim.deploy(thetas_tr=delpoyed_thetas_tr, gammas=gammas, admission_rates=admission_rates)
  sim.enroll(theta_stars_tr=theta_stars_tr)

  # compute outputs, E[y|z]
  y = sim.y[:, 0]  # extract the outcomes in env 1
  z = sim.z
  outputs = [
    y[i * s:(i + 1) * s][z[i * s:(i + 1) * s] == 1].mean()
    for i in range(T)
  ]

  # do OLS
  reg = LinearRegression()
  reg.fit(X=my_thetas_tr.reshape(T, 2), y=outputs)

  # [theta_AO, normalised_theta_star]
  candidates = sk_normalize([reg.coef_, theta_stars_tr[0]], norm='l2', axis=1)
  print(candidates.round(1))

  return


if __name__ == "__main__":
  run_utility_exp()
