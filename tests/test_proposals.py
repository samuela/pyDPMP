import numpy as np

from pyDPMP.mrf import MRF
from pyDPMP.proposals import random_walk_proposal, mixture_proposal
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

@seeded
def test_mixture_proposal1():
  def prop0(mrf, nAdd, x):
    return {'v': [0]}
  def prop1(mrf, nAdd, x):
    return {'v': [1]}
  prop = mixture_proposal([prop0, prop1])
  points = [prop(None, None, None)['v'][0] for _ in range(1000)]
  assert np.abs(np.sum(points) / 1000.0 - 0.5) < 0.1

@seeded
def test_mixture_proposal2():
  def prop0(mrf, nAdd, x):
    return {'v': [0]}
  def prop1(mrf, nAdd, x):
    return {'v': [1]}
  prop = mixture_proposal([prop0, prop1], weights=[0, 1])
  points = [prop(None, None, None)['v'][0] for _ in range(1000)]
  assert all([p == 1 for p in points])

@seeded
def test_mixture_proposal3():
  def prop0(mrf, nAdd, x):
    return {'v': [0]}
  def prop1(mrf, nAdd, x):
    return {'v': [1]}
  prop = mixture_proposal([prop0, prop1], weights=[1, 0])
  points = [prop(None, None, None)['v'][0] for _ in range(1000)]
  assert all([p == 0 for p in points])
