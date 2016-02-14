import numpy as np

from .mrf import log_prob_states, neighboring_factors
from .util import axisify

def tree_sched(mrf, root):
  """Constructs a forward-backward schedule on tree-structured graphs."""

  visited = set([root])
  frontier = set([root])
  fwd_sched = []
  while len(visited) < len(mrf.nodes) + len(mrf.factors.items()):
    new_wave = []
    for fn in frontier:
      nbrs = None
      if fn in mrf.nodes:
        # fn is a node
        nbrs = set(neighboring_factors(mrf, fn)) - visited
      else:
        # fn is a factor
        nbrs = set(mrf.factors[fn].nodes) - visited
      fwd_sched.extend([(fn, n) for n in nbrs])
      new_wave.extend(nbrs)

    visited |= set(new_wave)
    frontier |= set(new_wave)

  bwd_sched = list(reversed([(t, s) for (s, t) in fwd_sched]))
  return fwd_sched + bwd_sched

def fwd_bwd_sched(mrf):
  """Constructs a forward-backward message update schedule.

  It's assumed that the factors are in a chain in sorted order.

  Parameters
  ----------
  mrf : MRF

  Returns
  -------
  Message update schedule.
  """
  factors = sorted(mrf.factors.items(), key=lambda kv: kv[0])
  return [s
          for (fid, f) in factors
          for v in f.nodes
          for s in [(v, fid), (fid, v)]]

def full_sched(mrf):
  """Returns a complete message update schedule that runs in no particular
  order.

  Parameters
  ----------
  mrf : MRF

  Returns
  -------
  A schedule which covers every edge in both directions.
  """
  return ([(fid, v) for (fid, f) in mrf.factors.items() for v in f.nodes]
        + [(v, fid) for (fid, f) in mrf.factors.items() for v in f.nodes])

def decode_MAP_states(mrf, node_bel, epsilon=1e-5):
  """Decodes the MAP states given the node beliefs and calculates the number of
  ties.

  Parameters
  ----------
  mrf : MRF
  node_bel : dict (edges -> array of beliefs)
  epsilon : float (default: 1e-5)
      The tolerance between two log beliefs deemed to be negligible. In other
      words, we say that two states at a particular vertex are tied if the
      difference between their log beliefs is less than `epsilon`. `n_ties`
      counts the total number of states which are tied with the MAP state across
      all vertices.

  Returns
  -------
  map_state : dict (nodes -> int)
      The MAP state. Each `map_state[v]` specifies the index of the MAP particle
      at vertex v.
  n_ties : int
      The total number of MAP ties across all of the vertices.
  """
  map_state = {v: np.argmax(node_bel[v]) for v in mrf.nodes}
  n_ties = sum([np.sum(np.abs(node_bel[v]
                              - node_bel[v][map_state[v]]) < epsilon) - 1
                for v in mrf.nodes])
  return map_state, n_ties

class MessagePassingScheme(object):
  def __init__(self, mrf):
    self.mrf = mrf

  def messages(self, pots):
    pass

  def log_beliefs(self, pots, msg):
    pass

class MaxSumMP(MessagePassingScheme):
  def __init__(self,
               mrf,
               max_iters=100,
               conv_tol=1e-5,
               stepsize=1.0,
               sched=None):
    """Max-product belief propogation (max sum over log potentials).

    Parameters
    ----------
    mrf : MRF
    max_iters : int (default: 100)
    conv_tol : float (default: 1e-5)
        The message passing algorithm will abort once the maximum difference
        between messages across iterations is less than `conv_tol`.
    stepsize : float (default: 1.0)
        A factor which controls the "strength" of updates to messages. Use 1.0
        for tree-structured graphs.
    sched : list of edges or None
        The schedule by which messages should be sent. Defaults to
        `full_sched(mrf)`.
    """
    super(self.__class__, self).__init__(mrf)
    self.max_iters = max_iters
    self.conv_tol = conv_tol
    self.stepsize = stepsize
    self.sched = full_sched(mrf) if sched == None else sched

  def vf_message(self, pots, msgs, v, fid):
    return sum([msgs[(fid0, v)]
                for fid0 in neighboring_factors(self.mrf, v)
                if fid0 != fid])

  def fv_message(self, pots, msgs, fid, v):
    factor = self.mrf.factors[fid]
    total_axes = len(factor.nodes)

    assert v in factor.nodes
    assert total_axes == len(pots[fid].shape)

    # Calculate the message foundation matrix. It will have shape
    # (nStates[v_1] x ... x nStates[v_n]) for v_1, ..., v_n in factor.nodes.
    msg_foundation = sum([axisify(msgs[(v0, fid)], i, total_axes)
                          for (i, v0) in enumerate(factor.nodes)
                          if v0 != v],
                         pots[fid])

    # The axis corresponding to v.
    ax_v = factor.nodes.index(v)

    # Roll the axis corresponding to v to be the first. Resulting shape will be
    # (nStates[v_i] x nStates[v_1] x ... x nStates[v_n]).
    rolled = np.rollaxis(msg_foundation, ax_v)

    # Flatten out to have shape (nStates[v_i], <whatever>).
    flat = rolled.reshape((rolled.shape[0], -1))

    return np.max(flat, axis=1)

  def messages(self, pots, nStates, calc_logP=False):
    # Init old messages to 0 everywhere
    old_msgs = {}
    for (fid, f) in self.mrf.factors.items():
      for v in f.nodes:
        old_msgs[(fid, v)] = np.zeros(nStates[v])
        old_msgs[(v, fid)] = np.zeros(nStates[v])

    # Init messages to be uniform
    msgs = {}
    for (fid, f) in self.mrf.factors.items():
      for v in f.nodes:
        msgs[(fid, v)] = np.log(1.0 / nStates[v]) * np.ones(nStates[v])
        msgs[(v, fid)] = np.log(1.0 / nStates[v]) * np.ones(nStates[v])

    stats = {'last_iter': None, 'error': [], 'converged': False}
    if calc_logP:
      stats['logP'] = []

    for it in range(self.max_iters):
      # Set the damping factor
      damp = 1.0 if (it == 0) else self.stepsize

      for (s, t) in self.sched:
        new_msg = None
        if s in self.mrf.nodes:
          # Node to factor
          v, fid = (s, t)
          new_msg = self.vf_message(pots, msgs, v, fid)
        else:
          # Factor to node
          fid, v = (s, t)
          new_msg = self.fv_message(pots, msgs, fid, v)

        # Normalize for numerical stability
        new_msg -= np.max(new_msg)

        # Damp and set messages
        new_msg_damped = damp * new_msg + (1 - damp) * old_msgs[(s, t)]
        old_msgs[(s, t)] = np.copy(msgs[(s, t)])
        msgs[(s, t)] = new_msg_damped

      # Compute log probability and bound
      if calc_logP:
        node_bel, _ = self.log_beliefs(pots, msgs)
        map_states, _ = decode_MAP_states(self.mrf, node_bel)
        logP = log_prob_states(self.mrf, pots, map_states)
        stats['logP'].append(logP)

      # Check convergence
      edge_error = lambda s, t: \
          np.max(np.abs(np.exp(msgs[(s, t)]) - np.exp(old_msgs[(s, t)])))
      msg_diff = max([max(edge_error(fid, v), edge_error(v, fid))
                      for (fid, f) in self.mrf.factors.items()
                      for v in f.nodes])
      stats['error'].append(msg_diff)

      if it > 0 and msg_diff < self.conv_tol:
        # We have converged!
        stats['converged'] = True
        stats['last_iter'] = it
        break

    return msgs, stats

  def log_beliefs(self, pots, msgs):
    node_bel = {v: sum([msgs[(fid, v)]
                        for fid in neighboring_factors(self.mrf, v)])
                for v in self.mrf.nodes}

    factor_bel = {fid: sum([axisify(msgs[(v0, fid)], i, len(f.nodes))
                            for (i, v0) in enumerate(f.nodes)],
                           pots[fid])
                  for (fid, f) in self.mrf.factors.items()}

    return node_bel, factor_bel

# class TreeReweightedMP(MessagePassingScheme):
#   def __init__(self,
#                mrf,
#                rho,
#                max_iters=100,
#                conv_tol=1e-5,
#                stepsize=1.0,
#                sched=None):
#     """Tree-Reweighted max-product message passing.
#
#     Parameters
#     ----------
#     mrf : MRF
#     rho : dict (edges -> prob.)
#         The edge appearance probabilities. rho is expected to be complete and
#         symmetric.
#     max_iters : int (default: 100)
#     conv_tol : float (default: 1e-5)
#         The message passing algorithm will abort once the maximum difference
#         between messages across iterations is less than `conv_tol`.
#     stepsize : float (default: 1.0)
#         A factor which controls the "strength" of updates to messages. Use 1.0
#         for tree-structured graphs.
#     sched : list of edges or None
#         The schedule by which messages should be sent. Defaults to
#         `full_sched(mrf)`.
#     """
#     super(self.__class__, self).__init__(mrf)
#     self.rho = rho
#     self.max_iters = max_iters
#     self.conv_tol = conv_tol
#     self.stepsize = stepsize
#     self.sched = full_sched(mrf) if sched == None else sched
#
#   def message_foundation(self, node_pot, edge_pot, msg, nStates, s, t):
#     # Get the unary potential at s (nStates[s])
#     pot_s = node_pot[s]
#
#     # Get edge potential between s and t (nStates[s] x nStates[t])
#     pot_st = edge_pot[(s, t)] if (s, t) in edge_pot else edge_pot[(t, s)].T
#     pot_st_rw = (1.0 / self.rho[(s, t)]) * pot_st
#
#     # Compute incoming messages to s except for t (nStates[s])
#     in_msg_a = sum([self.rho[(v, s)] * msg[(v, s)]
#                     for v in neighbors(self.mrf, s) if v != t],
#                    np.zeros(nStates[s]))
#     incoming_msg = in_msg_a - (1.0 - self.rho[(s, t)]) * msg[(t, s)]
#
#     # Return the total message matrix (nStates[s] x nStates[t])
#     return pot_s.reshape((nStates[s], 1)) \
#          + pot_st_rw \
#          + incoming_msg.reshape((nStates[s], 1))
#
#   def messages(self, node_pot, edge_pot, calc_logP=False):
#     nStates = {v: len(node_pot[v]) for v in self.mrf.nodes}
#
#     # Init old messages to 0 everywhere
#     msg_old = {}
#     for (s, t) in self.mrf.edges:
#       msg_old[(s, t)] = np.zeros(nStates[t])
#       msg_old[(t, s)] = np.zeros(nStates[s])
#
#     # Init messages to be uniform
#     msg = {}
#     for (s, t) in self.mrf.edges:
#       msg[(s, t)] = np.log(1.0 / nStates[t]) * np.ones(nStates[t])
#       msg[(t, s)] = np.log(1.0 / nStates[s]) * np.ones(nStates[s])
#
#     stats = {'last_iter': None, 'error': [], 'converged': False}
#     if calc_logP:
#       stats['logP'] = []
#       stats['logPbound'] = []
#
#     for it in range(self.max_iters):
#       # Set the damping factor
#       damp = 1.0 if (it == 0) else self.stepsize
#
#       for (s, t) in self.sched:
#         # Calculate message s => t
#         msg_foundation = self.message_foundation(node_pot, edge_pot, msg, \
#             nStates, s, t)
#         new_msg = np.max(msg_foundation, axis=0)
#
#         # Normalize for numerical stability
#         new_msg -= np.max(new_msg)
#
#         # Damp and set messages
#         new_msg_damped = damp * new_msg + (1 - damp) * msg_old[(s, t)]
#         msg_old[(s, t)] = np.copy(msg[(s, t)])
#         msg[(s, t)] = new_msg_damped
#
#       # Compute log probability and bound
#       if calc_logP:
#         node_bel, edge_bel = self.log_beliefs(node_pot, edge_pot, msg,
#                                               normalize=False)
#         node_bel_max = {v: np.max(node_bel[v]) for v in self.mrf.nodes}
#         logPbound_un = sum([node_bel_max[v] for v in self.mrf.nodes])
#
#         # Note that if (s, t) \in mrf.edges => (s, t) \in edge_bel
#         logPbound_pw = sum([self.rho[(s, t)] * (np.max(edge_bel[(s, t)])
#                                                 - node_bel_max[s]
#                                                 - node_bel_max[t])
#                             for (s, t) in self.mrf.edges])
#
#         logPbound = logPbound_un + logPbound_pw
#         stats['logPbound'].append(logPbound)
#
#         map_states, _ = decode_MAP_states(self.mrf, node_bel)
#         logP = log_prob_states(self.mrf, node_pot, edge_pot, map_states)
#         stats['logP'].append(logP)
#
#       # Check convergence
#       edge_error = lambda s, t: \
#           np.max(np.abs(np.exp(msg[(s, t)]) - np.exp(msg_old[(s, t)])))
#       # Add zero on the end to ensure that the list is non-empty and will
#       # default to zero.
#       msg_diff = max([max(edge_error(s, t), edge_error(t, s))
#                       for (s, t) in self.mrf.edges] + [0.0])
#       stats['error'].append(msg_diff)
#
#       if it > 0 and msg_diff < self.conv_tol:
#         # We have converged!
#         stats['converged'] = True
#         stats['last_iter'] = it
#         break
#
#     return msg, stats
#
#   def log_beliefs(self, node_pot, edge_pot, msg, normalize=True):
#     node_bel = {}
#     for v in self.mrf.nodes:
#       # Note that we start the sum with the node potentials.
#       node_bel[v] = sum([self.rho[(t, v)] * msg[(t, v)]
#                          for t in neighbors(self.mrf, v)],
#                         node_pot[v])
#
#     edge_bel = {}
#     for (s, t) in self.mrf.edges:
#       # Caclulate the sum of all the messages to s, except t. And vice versa.
#       s_bel = node_bel[s] - msg[(t, s)]
#       t_bel = node_bel[t] - msg[(s, t)]
#
#       unary_bel = np.reshape(s_bel, (-1, 1)) + np.reshape(t_bel, (1, -1))
#       pot_st = edge_pot[(s, t)] if (s, t) in edge_pot else edge_pot[(t, s)].T
#       edge_bel[(s, t)] = unary_bel + (1.0 / self.rho[(s, t)]) * pot_st
#
#     if normalize:
#       node_bel_norm = {v: node_bel[v] - np.max(node_bel[v])
#                        for v in self.mrf.nodes}
#       edge_bel_norm = {e: edge_bel[e] - np.max(edge_bel[e])
#                        for e in self.mrf.edges}
#       return node_bel_norm, edge_bel_norm
#     else:
#       return node_bel, edge_bel
