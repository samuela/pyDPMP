import numpy as np
import random

from functools import wraps
from nose.tools import make_decorator

SEED = 0

# @make_decorator
def seeded(f):
  @wraps(f)
  # @make_decorator
  def f_seeded(*args, **kwargs):
    np.random.seed(SEED)
    random.seed(SEED)
    return f(*args, **kwargs)
  return f_seeded
