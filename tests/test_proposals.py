import numpy as np

from pyDPMP.mrf import MRF
from pyDPMP.proposals import random_walk_proposal
from pyDPMP.util import seeded

@seeded
def test_random_walk_proposal():
  d = 10
  mrf = MRF([0], [], None, None)
  prop = random_walk_proposal(np.eye(d))
  x = {0: [np.zeros(d)]}
  nAdd = {0: 1000}
  x_prop = prop(mrf, nAdd, x)

  # Check that they have empirical mean close to zero
  assert np.max(sum(x_prop[0]) / nAdd[0]) < 0.1
