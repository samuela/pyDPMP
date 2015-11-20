import itertools

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from pyDPMP.mrf import MRF, log_prob_states

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

def mrf_brute_MAP(mrf, node_pot, edge_pot):
  """Compute the MAP by brute force. Assumes that the nodes are 0, 1, 2, ..."""
  ranges = [range(len(node_pot[v])) for v in mrf.nodes]
  best_map = max(itertools.product(*ranges),
                 key=lambda s: log_prob_states(mrf, node_pot, edge_pot, s))

  # We generally use dicts for the MAP states, so convert for consistency
  return {v: best_map[v] for v in mrf.nodes}
