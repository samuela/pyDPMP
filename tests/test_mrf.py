import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

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

def random_tree_mrf(V):
  weights = np.triu(np.random.rand(V, V), 1)
  tree = minimum_spanning_tree(weights).toarray() > 0.0

  nodes = range(V)
  edges = zip(*np.where(tree))

  node_pot = {v: np.log(np.random.rand(2)) for v in nodes}
  edge_pot = {e: np.log(np.random.rand(2, 2)) for e in edges}

  mrf = MRF(nodes,
            edges,
            lambda s, x_s: node_pot[s][x_s],
            lambda s, t, x_s, x_t: edge_pot[(s, t)][x_s, x_t])

  return mrf
