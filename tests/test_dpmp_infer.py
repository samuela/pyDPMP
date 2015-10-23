import numpy as np

from pyDPMP.mrf import MRF
from pyDPMP.particleselection import SelectDiverse
from pyDPMP.messagepassing import MaxSumBP
from pyDPMP import DPMP_infer

from test_util import seeded

@seeded
def test_dpmp_infer():
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
