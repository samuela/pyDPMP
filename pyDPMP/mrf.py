import numpy as np
from collections import namedtuple


Factor = namedtuple('Factor', ['nodes', 'potential'])

class MRF(object):
  def __init__(self, nodes, factors):
    """Pairwise Markov Random Field.

    Parameters
    ----------
    nodes : list of vertex IDs, not necessarily integers
    factors : dict (id -> Factor)
    """
    self.nodes = nodes
    self.factors = factors

# def neighbors(mrf, v):
#   """Get the neighbors of a given vertex.
#
#   Parameters
#   ----------
#   v : node
#
#   Returns
#   -------
#   List of nodes that are connected to v.
#   """
#   return [t for t in mrf.nodes if ((v, t) in mrf.edges)
#                                or ((t, v) in mrf.edges)]

def calc_potentials(mrf, x):
  """Calculate all unary and pairwise log potentials.

  Parameters
  ----------
  mrf : MRF
  x : dict (v -> list of particles)
      The particle set to evaluate.

  Returns
  -------
  pots : dict (id -> np.array)
      Log factor potentials.
  """
  pots = {}

  def _axisify(arr, target_axis, total_axes):
    shape = np.ones(total_axes)
    shape[target_axis] = len(arr)
    return np.reshape(arr, shape)

  for fid, f in mrf.factors.items():
    vf = np.vectorize(f.potential, otypes=[np.float])
    total_axes = len(f.nodes)
    reshaped = [_axisify(x[v], i, total_axes)
                for (i, v) in enumerate(f.nodes)]
    pots[fid] = vf(*reshaped)

  return pots

def log_prob_states(mrf, pots, states):
  """Evaluate the log probability of a particular state sequence.

  Parameters
  ----------
  mrf : MRF
  pots : dict (factor -> array of potentials)
      The log potentials for each factor.
  states : dict (v -> int)
      A representation of the state of every node.

  Returns
  -------
  The log probability of the given state assignment.
  """
  logprob = 0
  for fid, f in mrf.factors.items():
    ixs = tuple([states[v] for v in f.nodes])
    logprob += pots[fid][ixs]
  return logprob
