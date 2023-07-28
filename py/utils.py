from sklearn.linear_model import LinearRegression
from typing import List, Tuple

import numpy as np


def normalize(arrs: list, new_min: float, new_max: float) -> Tuple[List[float], float]:
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

def recover_thetas(num_applicants, applicants_per_round, y, theta, z, env_idx, est_list, test_theta):
    """having inferred the theta_star and theta_ols, this function returns either theta_star, theta_ols, 
    or theta_ols. Note that all returned vectors are normalized. 

    Args:
        num_applicants (int):
        applicants_per_round (int):
        y (np.ndarray):
        theta (np.ndarray):
        z (np.ndarray):
        env_idx (int):
        est_list (dict):
        test_theta (np.ndarray):

    Returns:
        theta_star, theta_ao, or theta_ols (np.ndarray)
    """
    assert test_theta in ('theta_star_hat', 'theta_ols_hat', 'theta_ao_hat')

    if test_theta == 'theta_star_hat':
        theta_star_est = est_list[f'ours_env{env_idx}'][-1]
        theta_star_est_norm = theta_star_est / np.linalg.norm(theta_star_est)
        return theta_star_est_norm

    elif test_theta == 'theta_ols_hat':
        theta_ols = est_list[f'ols_env{env_idx}'][-1]
        theta_ols /= np.linalg.norm(theta_ols)
        return theta_ols

    elif test_theta == 'theta_ao_hat':
        # recovering theta_ao
        theta_ao_target, theta_ao_input = [] , [] 
        n_rounds = num_applicants / applicants_per_round

        for t in range(int(n_rounds)):
            lower =t*applicants_per_round  
            upper = lower + applicants_per_round
  
            idx = z[lower:upper] == env_idx+1
            theta_ao_target.append(y[env_idx, lower:upper][idx].mean())
            theta_ao_input.append(
            theta[env_idx, lower]
        )

        theta_ao_input, theta_ao_target = np.array(theta_ao_input), np.array(theta_ao_target)
        m = LinearRegression()
        m.fit(theta_ao_input, theta_ao_target)
        theta_ao_est = m.coef_ 
        theta_ao_est /= np.linalg.norm(theta_ao_est)
        return theta_ao_est 