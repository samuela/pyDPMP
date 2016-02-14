import numpy as np

from pyDPMP.messagepassing import tree_sched, MaxSumMP, decode_MAP_states
from pyDPMP.mrf import Factor, MRF, calc_potentials, log_prob_states
from pyDPMP.util import set_seed
from .test_util import random_tree_mrf, random_tree_mrf2, mrf_brute_MAP


def test_tree_sched1():
  mrf = MRF(['v0'], {'f0': Factor(['v0'], None)})
  assert tree_sched(mrf, 'v0') == [('v0', 'f0'), ('f0', 'v0')]
  assert tree_sched(mrf, 'f0') == [('f0', 'v0'), ('v0', 'f0')]

def test_tree_sched2():
  mrf = MRF(['v0', 'v1'], {'f0': Factor(['v0', 'v1'], None)})
  assert tree_sched(mrf, 'v0') == [('v0', 'f0'), ('f0', 'v1'),
                                   ('v1', 'f0'), ('f0', 'v0')]
  assert tree_sched(mrf, 'v1') == [('v1', 'f0'), ('f0', 'v0'),
                                   ('v0', 'f0'), ('f0', 'v1')]
  assert tree_sched(mrf, 'f0') == [('f0', 'v0'), ('f0', 'v1'),
                                   ('v1', 'f0'), ('v0', 'f0')]

def test_maxsum_basic():
  """Test that maxsum converges on a simple graph with 2 nodes."""
  mrf = MRF([0, 1], {
    'un0': Factor([0], lambda x_s: np.log(x_s + 1)),
    'un1': Factor([1], lambda x_s: np.log(x_s + 1)),
    'pw01': Factor([0, 1], lambda x_s, x_t: np.log(x_s * x_t + 1))
  })

  x = [[0, 0.5, 1], [0, 1]]

  pots = calc_potentials(mrf, x)
  nStates = {0: 3, 1: 2}

  maxsum = MaxSumMP(mrf)
  msgs, stats = maxsum.messages(pots, nStates)
  node_bel, _ = maxsum.log_beliefs(pots, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)

  assert n_ties == 0
  assert stats['converged'] == True
  assert map_state == {0: 2, 1: 1}

def test_maxsum_basic_w_tree_sched():
  """Test that the tree schedule on a simple 2-node chain converges
  immediately."""
  mrf = MRF([0, 1], {
    'un0': Factor([0], lambda x_s: np.log(x_s + 1)),
    'un1': Factor([1], lambda x_s: np.log(x_s + 1)),
    'pw01': Factor([0, 1], lambda x_s, x_t: np.log(x_s * x_t + 1))
  })

  x = [[0, 0.5, 1], [0, 1]]
  nStates = {0: 3, 1: 2}

  pots = calc_potentials(mrf, x)

  for root in [0, 1, 'un0', 'un1', 'pw01']:
    sched = tree_sched(mrf, root)
    maxsum = MaxSumMP(mrf, 100, 0.001, 1.0, sched=sched)
    msgs, stats = maxsum.messages(pots, nStates)
    node_bel, _ = maxsum.log_beliefs(pots, msgs)
    map_state, n_ties = decode_MAP_states(mrf, node_bel)

    assert stats['converged'] == True
    assert stats['last_iter'] <= 2
    assert stats['error'][-1] == 0.0

    assert map_state == {0: 2, 1: 1}
    assert n_ties == 0

def test_maxsum_2d_gaussian():
  mu = np.array([0.0, 0.0])
  cov = np.array([[1, 0], [1, 3]])
  cov_i = np.linalg.inv(cov)

  mrf = MRF([0, 1], {
    'u0': Factor([0], lambda x0: -0.5 * cov_i[0, 0] * ((x0 - mu[0]) ** 2)),
    'u1': Factor([1], lambda x1: -0.5 * cov_i[1, 1] * ((x1 - mu[1]) ** 2)),
    'pw': Factor([0, 1], lambda x0, x1: \
      -0.5 * (x0 - mu[0]) * (x1 - mu[1]) * (cov_i[0, 1] + cov_i[1, 0]))
  })

  x = {0: [-1.0, 0.0, 1.0], 1: [-1, 0.0, 1.0, 2.0]}
  pots = calc_potentials(mrf, x)
  nStates = {v: len(x[v]) for v in mrf.nodes}

  maxsum = MaxSumMP(mrf)
  msgs, _ = maxsum.messages(pots, nStates)
  node_bel, _ = maxsum.log_beliefs(pots, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  assert n_ties == 0
  assert xMAP == {0: 0.0, 1: 0.0}

def test_maxsum_2d_gaussian_alt():
  mu = np.array([0.0, 0.0])
  cov = np.array([[1, 0], [1, 3]])
  cov_i = np.linalg.inv(cov)

  def pot(x0, x1):
    x = [x0 - mu[0], x1 - mu[1]]
    return -0.5 * np.dot(x, np.dot(cov_i, x))

  mrf = MRF([0, 1], {'f': Factor([0, 1], pot)})

  x = {0: [-2.0, -1.0, 0.0, 1.0, 2.0], 1: [-2.0, -1.0, 0.0, 1.0, 2.0]}
  pots = calc_potentials(mrf, x)
  nStates = {v: len(x[v]) for v in mrf.nodes}

  maxsum = MaxSumMP(mrf)
  msgs, _ = maxsum.messages(pots, nStates)
  node_bel, _ = maxsum.log_beliefs(pots, msgs)
  map_state, n_ties = decode_MAP_states(mrf, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  assert n_ties == 0
  assert xMAP == {0: 0.0, 1: 0.0}

def check_maxsum_tree(mrf):
  x = {v: [0, 1] for v in mrf.nodes}
  nStates = {v: len(x[v]) for v in mrf.nodes}

  pots = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf, 100, 0.001, 1.0)
  msgs, stats = maxsum.messages(pots, nStates)
  node_bel, _ = maxsum.log_beliefs(pots, msgs)
  map_state, _ = decode_MAP_states(mrf, node_bel)
  xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

  brute_MAP = mrf_brute_MAP(mrf, pots, nStates)

  assert stats['converged'] == True
  assert stats['error'][-1] < 1e-10
  assert xMAP == brute_MAP

def check_maxsum_tree_w_tree_sched(mrf):
  x = {v: [0, 1] for v in mrf.nodes}
  nStates = {v: len(x[v]) for v in mrf.nodes}

  pots = calc_potentials(mrf, x)
  brute_MAP = mrf_brute_MAP(mrf, pots, nStates)

  for root in list(mrf.nodes) + list(mrf.factors.keys()):
    sched = tree_sched(mrf, root)
    maxsum = MaxSumMP(mrf, 100, 0.001, 1.0, sched=sched)
    msgs, stats = maxsum.messages(pots, nStates)
    node_bel, _ = maxsum.log_beliefs(pots, msgs)
    map_state, _ = decode_MAP_states(mrf, node_bel)
    xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}

    assert stats['converged'] == True
    assert stats['error'][-1] < 1e-10
    assert xMAP == brute_MAP
    assert stats['last_iter'] <= 2

# For some reason decorating a test generator doesn't work. See
# https://github.com/nose-devs/nose/issues/958.
# @seeded
def test_maxsum_trees():
  """Test that MaxSumMP converges on random tree-structured graphs."""
  set_seed(0)

  for _ in range(100):
    mrf = random_tree_mrf(10)
    yield check_maxsum_tree, mrf

def test_maxsum_trees2():
  """Test that MaxSumMP converges on random tree-structured graphs generated
  with higher-order factors."""
  set_seed(0)

  for _ in range(100):
    mrf = random_tree_mrf2(20)
    yield check_maxsum_tree, mrf

def test_maxsum_trees2_w_tree_sched():
  """Test that MaxSumMP converges on random tree-structured graphs generated
  with higher-order factors. Tests that message passing converges
  immediately."""
  set_seed(0)

  for _ in range(100):
    mrf = random_tree_mrf2(20)
    yield check_maxsum_tree_w_tree_sched, mrf

def test_n_ties1():
  mrf = MRF([0, 1], [])
  node_bel = {0: np.zeros(2), 1: np.zeros(2)}

  _, n_ties = decode_MAP_states(mrf, node_bel)
  assert n_ties == 2

def test_n_ties2():
  mrf = MRF([0, 1], [])
  node_bel = {0: np.zeros(10), 1: 1e-5 * np.random.rand(10)}

  _, n_ties = decode_MAP_states(mrf, node_bel)
  assert n_ties == 18

def test_n_ties3():
  mrf = MRF([0, 1], [])
  node_bel = {0: np.zeros(10), 1: 5 * np.random.rand(10)}

  _, n_ties = decode_MAP_states(mrf, node_bel, epsilon=5)
  assert n_ties == 18

def test_logP():
  """Test that the final logP for a tree is the true MAP logP."""
  set_seed(0)

  mrf = random_tree_mrf(10)
  x = {v: [0, 1] for v in mrf.nodes}
  nStates = {v: len(x[v]) for v in mrf.nodes}

  pots = calc_potentials(mrf, x)

  maxsum = MaxSumMP(mrf, 100, 0.001, stepsize=0.75)
  _, stats = maxsum.messages(pots, nStates, calc_logP=True)

  brute_MAP = mrf_brute_MAP(mrf, pots, nStates)

  np.testing.assert_almost_equal(
      stats['logP'][-1],
      log_prob_states(mrf, pots, brute_MAP))

# TODO: test factor beliefs
