import numpy as np

from dpmp.mrf import MRF
from dpmp.particleselection import SelectDiverse
from dpmp.messagepassing import MaxSumBP
from dpmp import DPMP_infer

def test_dpmp_infer():
  np.random.seed(0)

  mrf = MRF([0, 1], [(0, 1)],
            lambda _1, x: -(x ** 2),
            lambda _1, _2, x, y: -((x - y) ** 2))
  # x0 = {0: [-1, 1], 1: [-1, 1]}
  x0 = {0: [0.0], 1: [0.0]}
  nParticles = 5

  def proposal(x, mrf, nParticlesAdd):
    return {v: list(100 * np.random.randn(nParticlesAdd[v])) for v in mrf.nodes}

  xMAP, x, stats = DPMP_infer(mrf, x0, nParticles, proposal, \
      SelectDiverse(), MaxSumBP(mrf), max_iters=50)

  assert xMAP == {0: 0.0, 1: 0.0}
