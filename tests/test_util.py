import itertools

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from pyDPMP.mrf import Factor, MRF, log_prob_states
from pyDPMP.util import merge_dicts

def random_tree_mrf(V):
  """Constructs a random tree on V nodes and adds random edge and unary factors
  to it. The resulting MRF will be defined on binary variables."""
  weights = np.triu(np.random.rand(V, V), 1)
  tree = minimum_spanning_tree(weights).toarray() > 0.0

  nodes = range(V)
  edges = zip(*np.where(tree))

  node_pot = {v: np.log(np.random.rand(2)) for v in nodes}
  edge_pot = {e: np.log(np.random.rand(2, 2)) for e in edges}

  node_factors = {'u{}'.format(v): Factor([v], lambda x_v: node_pot[v][x_v])
                  for v in nodes}
  edge_factors = {'e{}_{}'.format(s, t): Factor([s, t], lambda x_s, x_t: \
                                                edge_pot[(s, t)][x_s, x_t])
                  for (s, t) in edges}

  return MRF(nodes, merge_dicts(node_factors, edge_factors))

def random_mrf(V, p):
  nodes = range(V)
  edges = []
  for s in range(V):
    for t in range(s + 1, V):
      if np.random.rand() < p:
        edges.append((s, t))

  node_pot = {v: np.log(np.random.rand(2)) for v in nodes}
  edge_pot = {e: np.log(np.random.rand(2, 2)) for e in edges}

  mrf = MRF(nodes,
            edges,
            lambda s, x_s: node_pot[s][x_s],
            lambda s, t, x_s, x_t: edge_pot[(s, t)][x_s, x_t])

  return mrf

def mrf_brute_MAP(mrf, pots, nStates):
  """Compute the MAP by brute force. Assumes that the nodes are 0, 1, 2, ..."""
  ranges = [range(nStates[v]) for v in mrf.nodes]
  best_map = max(itertools.product(*ranges),
                 key=lambda s: log_prob_states(mrf, pots, s))

  # We generally use dicts for the MAP states, so convert for consistency
  return {v: best_map[v] for v in mrf.nodes}
