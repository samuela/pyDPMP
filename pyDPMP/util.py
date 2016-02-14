import numpy as np
import random

from functools import wraps

def axisify(arr, target_axis, total_axes):
  """Reshape a vector to lie on target_axis in an array with total_axes
  dimensions."""
  shape = np.ones(total_axes)
  shape[target_axis] = len(arr)
  return np.reshape(arr, shape)

def merge_dicts(*dict_args):
  """Given any number of dicts, shallow copy and merge into a new dict,
  precedence goes to key value pairs in latter dicts."""
  result = {}
  for dictionary in dict_args:
      result.update(dictionary)
  return result

def set_seed(seed=0):
  np.random.seed(seed)
  random.seed(seed)

def seeded(f, seed=0):
  @wraps(f)
  def f_seeded(*args, **kwargs):
    set_seed(seed)
    return f(*args, **kwargs)
  return f_seeded
