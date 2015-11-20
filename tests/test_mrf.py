import numpy as np

from pyDPMP.mrf import MRF, calc_potentials, neighbors

def test_neighbors():
  V = 10
  mrf = MRF(range(V), [(0, v) for v in range(1, V)], None, None)
  assert neighbors(mrf, 0) == list(range(1, 10))
  assert neighbors(mrf, 1) == [0]

def test_calc_potentials_isingish():
  node_pot_f = lambda s, x_s: np.log(x_s + 1)
  edge_pot_f = lambda s, t, x_s, x_t: np.log(x_s * x_t + 1)
  mrf = MRF([0, 1], [(0, 1)], node_pot_f, edge_pot_f)

  x = [[0, 1], [0, 1]]

  node_pot, edge_pot = calc_potentials(mrf, x)

  np.testing.assert_almost_equal(node_pot[0], np.array([0, np.log(2)]))
  np.testing.assert_almost_equal(node_pot[1], np.array([0, np.log(2)]))
  np.testing.assert_almost_equal(edge_pot[(0, 1)], np.array([[0, 0], [0, np.log(2)]]))
