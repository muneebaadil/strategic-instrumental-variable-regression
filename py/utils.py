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
