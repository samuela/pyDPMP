import numpy as np

from .mrf import neighbors

def fwd_bwd_sched(mrf):
  """Constructs a forward-backward message update schedule.

  It's assumed that the nodes are in a chain in the order they appear in
  `mrf.nodes`.

  Parameters
  ----------
  mrf : MRF

  Returns
  -------
  Message update schedule.
  """
  return list(zip(mrf.nodes, mrf.nodes[1:])) \
       + list(zip(reversed(mrf.nodes), reversed(mrf.nodes[:-1])))

def full_sched(mrf):
  """Returns a message update schedule that updates each edge in no particular
  order.

  Parameters
  ----------
  mrf : MRF

  Returns
  -------
  A schedule which covers every edge in both directions.
  """
  return [(s, t) for (s, t) in mrf.edges] + [(t, s) for (s, t) in mrf.edges]

class MessagePassingScheme(object):
  def __init__(self, mrf):
    self.mrf = mrf

  def messages(self, node_pot, edge_pot):
    pass

  def log_beliefs(self, node_pot, edge_pot, msg):
    pass

class MaxSumBP(MessagePassingScheme):
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

  def message_foundation(self, node_pot, edge_pot, msg, nStates, s, t):
    # Get the unary potential at s (nStates[s])
    pot_s = node_pot[s]

    # Get edge potential between s and t (nStates[s] x nStates[t])
    pot_st = edge_pot[(s, t)] if (s, t) in edge_pot else edge_pot[(t, s)].T

    # Compute incoming messages to s except for t (nStates[s])
    incoming_msg = sum([msg[(v, s)] for v in neighbors(self.mrf, s) if v != t],
                       np.zeros(nStates[s]))

    # Return the total message matrix (nStates[s] x nStates[t])
    return pot_s.reshape((nStates[s], 1)) \
         + pot_st \
         + incoming_msg.reshape((nStates[s], 1))

  def messages(self, node_pot, edge_pot):
    nStates = {v: len(node_pot[v]) for v in self.mrf.nodes}

    # Init old messages to 0 everywhere
    msg_old = {}
    for (s, t) in self.mrf.edges:
      msg_old[(s, t)] = np.zeros(nStates[t])
      msg_old[(t, s)] = np.zeros(nStates[s])

    # Init messages to be uniform
    msg = {}
    for (s, t) in self.mrf.edges:
      msg[(s, t)] = np.log(1.0 / nStates[t]) * np.ones(nStates[t])
      msg[(t, s)] = np.log(1.0 / nStates[s]) * np.ones(nStates[s])

    stats = {'last_iter': None, 'error': [], 'converged': False}

    for it in range(self.max_iters):
      # Set the damping factor
      damp = 1.0 if (it == 0) else self.stepsize

      for (s, t) in self.sched:
        # Calculate message s => t
        msg_foundation = self.message_foundation(node_pot, edge_pot, msg, \
            nStates, s, t)
        new_msg = np.max(msg_foundation, axis=0)
        # new_msg_argmax = np.argmax(msg_mat, axis=0)

        # Normalize for numerical stability
        new_msg -= np.max(new_msg)

        # Damp and set messages
        new_msg_damped = damp * new_msg + (1 - damp) * msg_old[(s, t)]
        msg_old[(s, t)] = np.copy(msg[(s, t)])
        msg[(s, t)] = new_msg_damped

      # Check convergence
      edge_error = lambda s, t: \
          np.max(np.abs(np.exp(msg[(s, t)]) - np.exp(msg_old[(s, t)])))
      # Add zero on the end to ensure that the list is non-empty and will
      # default to zero.
      msg_diff = max([max(edge_error(s, t), edge_error(t, s))
                      for (s, t) in self.mrf.edges] + [0.0])
      stats['error'].append(msg_diff)

      if it > 0 and msg_diff < self.conv_tol:
        # We have converged!
        stats['converged'] = True
        stats['last_iter'] = it
        break

    return msg, stats

  def log_beliefs(self, node_pot, edge_pot, msg):
    node_bel = {}
    for v in self.mrf.nodes:
      # Note that we start the sum with the node potentials.
      pre_msg = sum([msg[(t, v)] for t in neighbors(self.mrf, v)], node_pot[v])
      node_bel[v] = pre_msg - np.max(pre_msg)

    return node_bel

  def decode_MAP_states(self, node_pot, edge_pot, node_bel):
    map_state = {v: np.argmax(node_bel[v]) for v in self.mrf.nodes}
    n_ties = sum([np.sum(np.abs(node_bel[v]
                    - node_bel[v][map_state[v]]) < self.conv_tol) - 1
                  for v in self.mrf.nodes])
    return map_state, n_ties
