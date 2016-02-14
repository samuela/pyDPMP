import numpy as np
from collections import namedtuple
from .util import axisify

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

def neighboring_factors(mrf, v):
  """Get the neighboring factors of a given vertex.

  Parameters
  ----------
  v : node

  Returns
  -------
  List of ids of factors that are connected to v.
  """
  return [fid for (fid, f) in mrf.factors.items() if v in f.nodes]

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

  for fid, f in mrf.factors.items():
    vf = np.vectorize(f.potential, otypes=[np.float])
    total_axes = len(f.nodes)
    reshaped = [axisify(x[v], i, total_axes) for (i, v) in enumerate(f.nodes)]
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
  logprob = 0.0
  for fid, f in mrf.factors.items():
    ixs = tuple([states[v] for v in f.nodes])
    logprob += pots[fid][ixs]
  return logprob
