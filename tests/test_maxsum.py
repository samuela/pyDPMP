import numpy as np
import itertools

from pyDPMP.mrf import MRF, calc_potentials, log_prob_states
from pyDPMP.messagepassing import MaxSumBP
from pyDPMP.util import seeded
from .test_mrf import random_tree_mrf

def test_maxsum_basic():
  """Test that maxsum converges on a simple graph with 2 nodes."""
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
  """Test the fwd/bwd schedule on a simple 2-node chain."""
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

def test_maxsum_2d_gaussian():
  mu = np.array([0.0, 0.0])
  cov = np.array([[1, 0], [1, 3]])
  cov_i = np.linalg.inv(cov)

  nodes = [0, 1]
  edges = [(0, 1)]
  node_pot_f = lambda s, x_s: -0.5 * cov_i[s, s] * ((x_s - mu[s]) ** 2)
  edge_pot_f = lambda s, t, x_s, x_t: -0.5 * (x_s - mu[s]) * (x_t - mu[t]) * (cov_i[s,t] + cov_i[t,s])

  mrf = MRF(nodes, edges, node_pot_f, edge_pot_f)

  x = {0: [-1.0, 0.0, 1.0], 1: [-1, 0.0, 1.0, 2.0]}
  node_pot, edge_pot = calc_potentials(x, mrf)

  maxsum = MaxSumBP(mrf)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = maxsum.decode_MAP_states(node_pot, edge_pot, node_bel)
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
  node_pot, edge_pot = calc_potentials(x, mrf)

  maxsum = MaxSumBP(mrf)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = maxsum.decode_MAP_states(node_pot, edge_pot, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  assert n_ties == 0
  assert xMAP == {0: 0.0, 1: 0.0}

def mrf_brute_MAP(mrf, node_pot, edge_pot):
  """Compute the MAP by brute force. Assumes that the nodes are 0, 1, 2, ..."""
  ranges = [range(len(node_pot[v])) for v in mrf.nodes]
  best_map = max(itertools.product(*ranges),
                 key=lambda s: log_prob_states(mrf, s, node_pot, edge_pot))

  # We generally use dicts for the MAP states, so convert for consistency
  return {v: best_map[v] for v in mrf.nodes}

def check_maxsum_tree(mrf):
  x = {v: [0, 1] for v in mrf.nodes}

  node_pot, edge_pot = calc_potentials(x, mrf)

  maxsum = MaxSumBP(mrf, 100, 0.001, 1.0)
  msgs, stats = maxsum.messages(node_pot, edge_pot)
  node_bel = maxsum.log_beliefs(node_pot, edge_pot, msgs)
  map_state, n_ties = maxsum.decode_MAP_states(node_pot, edge_pot, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  brute_MAP = mrf_brute_MAP(mrf, node_pot, edge_pot)

  assert stats['converged'] == True
  assert stats['error'][-1] < 1e-10
  assert xMAP == brute_MAP

# For some reason decorating a test generator doesn't work. See
# https://github.com/nose-devs/nose/issues/958.
# @seeded
def test_maxsum_trees():
  """Test that MaxSumBP converges on random tree-structured graphs."""
  np.random.seed(0)

  for _ in range(100):
    mrf = random_tree_mrf(10)
    yield check_maxsum_tree, mrf

# TODO: test n_ties
