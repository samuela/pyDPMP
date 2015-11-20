import numpy as np

class MRF(object):
  def __init__(self, nodes, edges, node_pot, edge_pot):
    """Pairwise Markov Random Field.

    Parameters
    ----------
    nodes : list of vertex IDs, not necessarily integers
    edges : list of ordered pairs (s, t)
        Should only contain (s, t) or (t, s) but not both.
    node_pot : function (s, x_s -> R)
        Function for evaluating log unary potentials.
    edge_pot : function (s, t, x_s, x_t -> R)
        Function for evaluating log pairwise potentials.
    """
    self.nodes = nodes
    self.edges = edges

    # Single lambdas for each type of potential, since throwing around lambdas
    # everywhere isn't efficient.
    self.node_pot = node_pot
    self.edge_pot = edge_pot

def neighbors(mrf, v):
  """Get the neighbors of a given vertex.

  Parameters
  ----------
  v : node

  Returns
  -------
  List of nodes that are connected to v.
  """
  return [t for t in mrf.nodes if ((v, t) in mrf.edges)
                               or ((t, v) in mrf.edges)]

def calc_potentials(mrf, x):
  """Calculate all unary and pairwise log potentials.

  Parameters
  ----------
  mrf : MRF
  x : dict (v -> list of particles)
      The particle set to evaluate.

  Returns
  -------
  node_pot : dict (s -> [pot])
      Log unary potentials.
  edge_pot : dict ((s,t) -> [[pot]])
      Matrix of log pairwise potentials.
  """
  node_pot = {}
  edge_pot = {}

  for s in mrf.nodes:
    node_pot[s] = np.array([mrf.node_pot(s, x_s) for x_s in x[s]])

  for (s, t) in mrf.edges:
    pot_st = np.zeros((len(x[s]), len(x[t])))
    for (n_s, x_s) in enumerate(x[s]):
      for (n_t, x_t) in enumerate(x[t]):
        pot_st[n_s, n_t] = mrf.edge_pot(s, t, x_s, x_t)
    edge_pot[(s, t)] = pot_st

  return node_pot, edge_pot

def log_prob_states(mrf, node_pot, edge_pot, states):
  """Evaluate the log probability of a particular state sequence.

  Parameters
  ----------
  mrf : MRF
  node_pot : dict (v -> array of unary potentials)
      The log unary potentials for each particle at each node.
  edge_pot : dict ((s, t) -> Ns x Nt matrix of pairwise potentials)
      The log pairwise potentials for every pair of particles across each edge.
  states : dict (v -> int)
      A representation of the state of every node.

  Returns
  -------
  The log probability of the given state assignment.
  """
  return sum(node_pot[v][states[v]] for v in mrf.nodes) \
       + sum(edge_pot[(s, t)][states[s], states[t]] for (s, t) in mrf.edges)
