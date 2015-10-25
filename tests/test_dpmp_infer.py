import numpy as np

from pyDPMP.messagepassing import MaxSumBP
from pyDPMP.mrf import MRF
from pyDPMP.particleselection import SelectDiverse
from pyDPMP.proposals import random_walk_proposal_1d
from pyDPMP.util import seeded
from pyDPMP import DPMP_infer

@seeded
def test_dpmp_infer():
  """Test DPMP when the true MAP is in x0. Final MAP should the true MAP."""
  mrf = MRF([0, 1], [(0, 1)],
            lambda _1, x: -(x ** 2),
            lambda _1, _2, x, y: -((x - y) ** 2))
  x0 = {0: [0.0], 1: [0.0]}
  nParticles = 5

  def proposal(x, mrf, nParticlesAdd):
    return {v: list(100 * np.random.randn(nParticlesAdd[v])) for v in mrf.nodes}

  xMAP, x, stats = DPMP_infer(mrf, x0, nParticles, proposal, \
      SelectDiverse(), MaxSumBP(mrf), max_iters=50)

  assert xMAP == {0: 0.0, 1: 0.0}
  assert stats['converged'] == True

@seeded
def test_dpmp_infer_rw_prop_1d():
  """Test DPMP when the true MAP is in x0. Final MAP should the true MAP."""
  mrf = MRF([0, 1], [(0, 1)],
            lambda _1, x: -(x ** 2),
            lambda _1, _2, x, y: -((x - y) ** 2))
  x0 = {0: [0.0], 1: [0.0]}
  nParticles = 5

  prop = random_walk_proposal_1d(10)

  xMAP, x, stats = DPMP_infer(mrf, x0, nParticles, prop, SelectDiverse(), \
      MaxSumBP(mrf), max_iters=50)

  assert xMAP == {0: 0.0, 1: 0.0}
  assert stats['converged'] == True

@seeded
def test_dpmp_infer_callback():
  """Test that DPMP callback is called."""
  mrf = MRF([0, 1], [(0, 1)],
            lambda _1, x: -(x ** 2),
            lambda _1, _2, x, y: -((x - y) ** 2))
  x0 = {0: [0.0], 1: [0.0]}
  nParticles = 5

  def proposal(x, mrf, nParticlesAdd):
    return {v: list(100 * np.random.randn(nParticlesAdd[v])) for v in mrf.nodes}

  called = [False]
  def callback(info):
    called[0] = True
    return info['iter']

  xMAP, x, stats = DPMP_infer(mrf, x0, nParticles, proposal, \
      SelectDiverse(), MaxSumBP(mrf), max_iters=50, callback=callback)

  assert called[0] == True
  assert stats['converged'] == True
  assert stats['callback_results'][-1] == stats['last_iter']
