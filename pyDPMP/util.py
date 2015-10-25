import numpy as np
import random

from functools import wraps

def set_seed(seed=0):
  np.random.seed(seed)
  random.seed(seed)

def seeded(f, seed=0):
  @wraps(f)
  def f_seeded(*args, **kwargs):
    set_seed(seed)
    return f(*args, **kwargs)
  return f_seeded
