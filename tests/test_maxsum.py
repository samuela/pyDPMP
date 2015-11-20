import numpy as np

from pyDPMP.messagepassing import MaxSumMP, fwd_bwd_sched, decode_MAP_states
from pyDPMP.mrf import MRF, calc_potentials, log_prob_states
from pyDPMP.util import set_seed
from .test_util import random_tree_mrf, mrf_brute_MAP

def test_fwd_bwd_sched():
  """Test the forward/backward schedule."""
  nodes = ['a', 'b', 'c', 'd']
  mrf = MRF(nodes, None, None, None)
  sched = fwd_bwd_sched(mrf)
  exp = [('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'c'), ('c', 'b'), ('b', 'a')]
  assert sched == exp

def test_maxsum_basic():
  """Test that maxsum converges on a simple graph with 2 nodes."""
  node_pot_f = lambda s, x_s: np.log(x_s + 1)
  edge_pot_f = lambda s, t, x_s, x_t: np.log(x_s * x_t + 1)
  mrf = MRF([0, 1], [(0, 1)], node_pot_f, edge_pot_f)

  x = [[0, 0.5, 1], [0, 1]]

  node_pot, edge_pot = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf, 100, 0.001, 1.0, [(0, 1), (1, 0)])
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel, _ = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)

  assert n_ties == 0
  assert stats['converged'] == True
  assert map_state == {0: 2, 1: 1}

def test_maxsum_basic_fwd_bwd():
  """Test the fwd/bwd schedule on a simple 2-node chain."""
  node_pot_f = lambda s, x_s: np.log(x_s + 1)
  edge_pot_f = lambda s, t, x_s, x_t: np.log(x_s * x_t + 1)
  mrf = MRF([0, 1], [(0, 1)], node_pot_f, edge_pot_f)

  x = [[0, 0.5, 1], [0, 1]]

  node_pot, edge_pot = calc_potentials(mrf, x)

  sched = fwd_bwd_sched(mrf)
  maxsum = MaxSumMP(mrf, 100, 0.001, 1.0, sched=sched)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel, _ = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)

  assert stats['converged'] == True
  assert stats['last_iter'] == 1
  assert stats['error'][-1] == 0.0

  assert map_state == {0: 2, 1: 1}
  assert n_ties == 0

def test_maxsum_basic_auto_sched():
  """Test the auto schedule on a simple 2-node chain."""
  node_pot_f = lambda s, x_s: np.log(x_s + 1)
  edge_pot_f = lambda s, t, x_s, x_t: np.log(x_s * x_t + 1)
  mrf = MRF([0, 1], [(0, 1)], node_pot_f, edge_pot_f)

  x = [[0, 0.5, 1], [0, 1]]

  node_pot, edge_pot = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf, 100, 0.001, 1.0)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel, _ = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)

  assert stats['converged'] == True
  assert stats['last_iter'] == 1
  assert stats['error'][-1] == 0.0

  assert map_state == {0: 2, 1: 1}
  assert n_ties == 0

def test_maxsum_2d_gaussian():
  mu = np.array([0.0, 0.0])
  cov = np.array([[1, 0], [1, 3]])
  cov_i = np.linalg.inv(cov)

  nodes = [0, 1]
  edges = [(0, 1)]
  node_pot_f = lambda s, x_s: -0.5 * cov_i[s, s] * ((x_s - mu[s]) ** 2)
  edge_pot_f = lambda s, t, x_s, x_t: \
      -0.5 * (x_s - mu[s]) * (x_t - mu[t]) * (cov_i[s, t] + cov_i[t, s])

  mrf = MRF(nodes, edges, node_pot_f, edge_pot_f)

  x = {0: [-1.0, 0.0, 1.0], 1: [-1, 0.0, 1.0, 2.0]}
  node_pot, edge_pot = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf)
  msgs, _ = maxsum.messages(node_pot, edge_pot)
  node_bel, _ = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  assert n_ties == 0
  assert xMAP == {0: 0.0, 1: 0.0}

def test_maxsum_2d_gaussian_alt():
  mu = np.array([0.0, 0.0])
  cov = np.array([[1, 0], [1, 3]])
  cov_i = np.linalg.inv(cov)

  nodes = [0, 1]
  edges = [(0, 1)]
  node_pot_f = lambda s, x_s: 0
  def edge_pot_f(s, t, x_s, x_t):
    x = [x_s - mu[s], x_t - mu[t]] if s == 0 else [x_t - mu[t], x_s - mu[s]]
    return -0.5 * np.dot(x, np.dot(cov_i, x))

  mrf = MRF(nodes, edges, node_pot_f, edge_pot_f)

  x = {0: [-2.0, -1.0, 0.0, 1.0, 2.0], 1: [-2.0, -1.0, 0.0, 1.0, 2.0]}
  node_pot, edge_pot = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf)
  msgs, _ = maxsum.messages(node_pot, edge_pot)
  node_bel, _ = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  assert n_ties == 0
  assert xMAP == {0: 0.0, 1: 0.0}

def check_maxsum_tree(mrf):
  x = {v: [0, 1] for v in mrf.nodes}

  node_pot, edge_pot = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf, 100, 0.001, 1.0)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel, _ = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, _ = decode_MAP_states(mrf, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  brute_MAP = mrf_brute_MAP(mrf, node_pot, edge_pot)

  assert stats['converged'] == True
  assert stats['error'][-1] < 1e-10
  assert xMAP == brute_MAP

# For some reason decorating a test generator doesn't work. See
# https://github.com/nose-devs/nose/issues/958.
# @seeded
def test_maxsum_trees():
  """Test that MaxSumMP converges on random tree-structured graphs."""
  set_seed(0)

  for _ in range(100):
    mrf = random_tree_mrf(10)
    yield check_maxsum_tree, mrf

def test_n_ties1():
  mrf = MRF([0, 1], [(0, 1)], None, None)
  node_bel = {0: np.zeros(2), 1: np.zeros(2)}

  _, n_ties = decode_MAP_states(mrf, node_bel)
  assert n_ties == 2

def test_n_ties2():
  mrf = MRF([0, 1], [(0, 1)], None, None)
  node_bel = {0: np.zeros(10), 1: 1e-5 * np.random.rand(10)}

  _, n_ties = decode_MAP_states(mrf, node_bel)
  assert n_ties == 18

def test_n_ties3():
  mrf = MRF([0, 1], [(0, 1)], None, None)
  node_bel = {0: np.zeros(10), 1: 5 * np.random.rand(10)}

  _, n_ties = decode_MAP_states(mrf, node_bel, epsilon=5)
  assert n_ties == 18

def test_logP():
  """Test that the final logP for a tree is the true MAP logP."""
  set_seed(0)

  mrf = random_tree_mrf(10)
  x = {v: [0, 1] for v in mrf.nodes}

  node_pot, edge_pot = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf, 100, 0.001, stepsize=0.75)
  _, stats = maxsum.messages(node_pot, edge_pot, calc_logP=True)

  brute_MAP = mrf_brute_MAP(mrf, node_pot, edge_pot)

  np.testing.assert_almost_equal(
      stats['logP'][-1],
      log_prob_states(mrf, node_pot, edge_pot, brute_MAP))

# TODO: test edge beliefs
