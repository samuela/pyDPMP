import numpy as np

from pyDPMP.mrf import MRF, calc_potentials
from pyDPMP.messagepassing import MaxSumBP
from test_mrf import random_tree_mrf
from test_util import seeded

def test_maxsum_basic():
  node_pot_f = lambda s, x_s: np.log(x_s + 1)
  edge_pot_f = lambda s, t, x_s, x_t: np.log(x_s * x_t + 1)
  mrf = MRF([0, 1], [(0, 1)], node_pot_f, edge_pot_f)

  x = [[0, 0.5, 1], [0, 1]]

  node_pot, edge_pot = calc_potentials(x, mrf)

  maxsum = MaxSumBP(mrf, 100, 0.001, 1.0, [(0, 1), (1, 0)])
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = maxsum.decode_MAP_states(node_pot, edge_pot, node_bel)

def test_maxsum_basic_fwd_bwd():
  node_pot_f = lambda s, x_s: np.log(x_s + 1)
  edge_pot_f = lambda s, t, x_s, x_t: np.log(x_s * x_t + 1)
  mrf = MRF([0, 1], [(0, 1)], node_pot_f, edge_pot_f)

  x = [[0, 0.5, 1], [0, 1]]

  node_pot, edge_pot = calc_potentials(x, mrf)

  maxsum = MaxSumBP(mrf, 100, 0.001, 1.0)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = maxsum.decode_MAP_states(node_pot, edge_pot, node_bel)

  assert stats['converged'] == True
  assert stats['last_iter'] == 1
  assert stats['error'][-1] == 0.0

  assert map_state == {0: 2, 1: 1}
  assert n_ties == 0

def check_maxsum_tree(mrf):
  x = [[0, 1]] * len(mrf.nodes)

  node_pot, edge_pot = calc_potentials(x, mrf)

  maxsum = MaxSumBP(mrf, 100, 0.001, 1.0)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = maxsum.decode_MAP_states(node_pot, edge_pot, node_bel)

  # print stats
  assert stats['converged'] == True
  assert stats['error'][-1] < 1e-10

# For some reason decorating a test generator doesn't work. See
# https://github.com/nose-devs/nose/issues/958.
# @seeded
def test_maxsum_trees():
  np.random.seed(0)

  for _ in range(100):
    mrf = random_tree_mrf(20)
    yield check_maxsum_tree, mrf

# TODO: test n_ties, test correctness
