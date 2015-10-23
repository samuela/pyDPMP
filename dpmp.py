import numpy as np

class MRF(object):
  def __init__(self, nodes, edges, node_pot, edge_pot):
    self.nodes = nodes            # List of vertice IDs, 0, 1, 2, ..., V - 1
    self.edges = edges            # List of pairs describing edge relationships

    # Single lambdas for each type of potential, since throwing around lambdas
    # everywhere isn't efficient.
    self.node_pot = node_pot       # (s, x_s) -> R. The unary log potentials
    self.edge_pot = edge_pot       # (s, t, x_s, x_t) -> R. The pairwise log potentials

  def nbrs(self, v):
    return [t for t in self.nodes if ((v, t) in self.edges)
                                  or ((t, v) in self.edges)]

def _calc_potentials(x, mrf):
  """Calculate all unary and pairwise log potentials.

  Parameters
  ----------
  x : TODO.

  mrf : The MRF.

  Returns
  -------
  node_pot : dict (s -> [pot]) of log unary potentials
  edge_pot : dict ((s,t) -> [[pot]]) matrix of log pairwise potentials
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

def log_prob_states(mrf, states, node_pot, edge_pot):
  return sum(node_pot[v][states[v]] for v in mrf.nodes) \
       + sum(edge_pot[(s, t)][states[s], states[t]] for (s, t) in mrf.edges)

def fwd_bwd_sched(mrf):
  return zip(mrf.nodes, mrf.nodes[1:]) \
       + zip(reversed(mrf.nodes), reversed(mrf.nodes[:-1]))

def full_sched(mrf):
  return [(s, t) for (s, t) in mrf.edges] + [(t, s) for (s, t) in mrf.edges]

class MessagePassingScheme(object):
  def __init__(self, mrf):
    self.mrf = mrf

  def messages(self, node_pot, edge_pot):
    pass

  def log_beliefs(self, node_pot, edge_pot):
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
    stepsize : float (default: 1.0)
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
    pot_st = edge_pot[(s, t)] if (s <= t) else edge_pot[(t, s)].T

    # Compute incoming messages to s except for t (nStates[s])
    incoming_msg = sum([msg[(v, s)] for v in self.mrf.nbrs(s) if v != t],
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

    for it in xrange(self.max_iters):
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
      msg_diff = max([max(edge_error(s, t), edge_error(t, s))
                      for (s, t) in self.mrf.edges])
      stats['error'].append(msg_diff)

      if it > 0 and msg_diff < self.conv_tol:
        # We have converged!
        stats['converged'] = True
        stats['last_iter'] = it
        break

    return msg, stats

  def log_beliefs(self, node_pot, edge_pot, msg):
    nStates = {v: len(node_pot[v]) for v in self.mrf.nodes}

    node_bel = {}
    for v in self.mrf.nodes:
      pre_msg = sum([msg[(t, v)] for t in self.mrf.nbrs(v)],
                    np.zeros(nStates[v]))
      node_bel[v] = pre_msg - np.max(pre_msg)

    return node_bel

  def decode_MAP_states(self, node_pot, edge_pot, node_bel):
    map_state = {v: np.argmax(node_bel[v]) for v in self.mrf.nodes}
    n_ties = sum([np.sum(np.abs(node_bel[v]
                    - node_bel[v][map_state[v]]) < self.conv_tol) - 1
                  for v in self.mrf.nodes])
    return map_state, n_ties

class ParticleSelectionScheme(object):
  def select(self, mrf, map_state, msg_passing, msg, x_aug, nSelect, node_pot, \
      edge_pot, temp):
    pass

class SelectDiverse(ParticleSelectionScheme):
  def select(self, mrf, map_state, msg_passing, msg, x_aug, nSelect, node_pot, \
      edge_pot, temp):
    """Run particle selection from the ICML 2014 paper (dpmpmax)."""
    nStates = {v: len(node_pot[v]) for v in mrf.nodes}

    I_accept = {v: [] for v in mrf.nodes}
    for t in mrf.nodes:
      # Build the "stacked" message foundations matrix (S x T)
      M_t = np.vstack([msg_passing.message_foundation(node_pot, edge_pot, msg, \
                           nStates, t, s).T
                       for s in mrf.nbrs(t)])

      logMstar = np.max(M_t, axis=1)                        # (S,)
      logZ = 1.0 / temp * np.max(M_t)                       # scalar
      Mstar = np.exp(1.0 / temp * logMstar - logZ)          # (S,)
      Psi = np.exp(1.0 / temp * M_t - logZ)                 # (S, T)

      # Select first particle
      b = map_state[t]
      Mhat = Psi[:,b]                                       # (S,)
      delta = Mstar - Mhat                                  # (S,)
      I_accept[t].append(b)

      while len(I_accept[t]) < nSelect[t] and np.max(delta) > 0:
        # Get unused particles
        b_used = set(I_accept[t])
        b_unused = [bb for bb in range(nStates[t]) if bb not in b_used]

        # Select next particle
        a_star = np.argmax(delta)
        min_a = delta[a_star]
        idx_max = np.argmax(Psi[a_star, b_unused])
        b = b_unused[idx_max]
        I_accept[t].append(b)

        # Update message approximation
        Mhat = np.maximum(Mhat, Psi[:,b])
        delta = Mstar - Mhat
    return I_accept

class SelectLazyGreedy(ParticleSelectionScheme):
  pass

def DPMP_infer(mrf,
               x0,
               nParticles,
               proposal,
               particle_selection,
               msg_passing,
               max_iters=100,
               conv_tol=1e-5,
               nAugmented=None,
               callback=None,
               temp=1.0,
               verbose=False):
  """Run D-PMP inference.

  Parameters
  ----------
  mrf : MRF
  x0 : dict (v -> list of particles)
      The initial particle set.
  nParticles : int or dict (v -> int)
      The number of particles to keep after selection each iteration.
  proposal : function (x, mrf, nAdd -> list of particles)
  particle_selection : ParticleSelectionScheme
  msg_passing : MessagePassingScheme
  max_iters : int (default: 100)
  conv_tol : float (default: 1e-5)
      The log-probability convergence tolerance. If the log-probability of the
      MAP decoding does not improve by more than conv_tol, then we assume
      convergence and stop.
  nAugmented : int or dict (v -> int) or None (default: None)
      The number of particles to be selected from at every iteration. At each
      node, nAugmented determines the number of extra particles that will be
      proposed. If None, then it defaults to twice nParticles.
  callback : function or None (default: None)
      The function to be called at the end of each iteration.
  temp : float (default: 1.0)
      Tricky. TODO.
  verbose : boolean (default: False)

  Returns
  -------
  TODO
  """

  # Handle default arguments
  if isinstance(nParticles, int):
    nParticles = {v: nParticles for v in mrf.nodes}
  nAugmented = {v: 2 * nParticles[v] for v in mrf.nodes}

  # Start with initial particle set
  x = x0

  stats = {'logP': [], 'converged': False, 'last_iter': None}

  for i in xrange(max_iters):
    if verbose: print 'Iter', i

    # Sample new particles
    x_aug = None
    if i > 0:
      # Propose new particles
      if verbose: print '    ... Proposing new particles'
      nParticlesAdd = {v: nAugmented[v] - len(x[v]) for v in mrf.nodes}
      x_new = proposal(x, mrf, nParticlesAdd)

      # Construct augmented particle set
      # x_aug = [old_ps + new_ps for (old_ps, new_ps) in zip(x, x_new)]
      x_aug = {v: x[v] + list(x_new[v]) for v in mrf.nodes}
    else:
      x_aug = x

    # Calculate potentials
    if verbose: print '    ... Calculating potentials and MAP'
    node_pot, edge_pot = _calc_potentials(x_aug, mrf)

    # Calculate messages, log beliefs, and MAP states
    msgs, msg_passing_stats = msg_passing.messages(node_pot, edge_pot)
    node_bel_aug = msg_passing.log_beliefs(node_pot, edge_pot, msgs)
    map_states, n_ties = msg_passing.decode_MAP_states(node_pot, edge_pot, \
        node_bel_aug)
    logP_map = log_prob_states(mrf, map_states, node_pot, edge_pot)
    stats['logP'].append(logP_map)

    # Particle selection
    if verbose: print '    ... Selecting particles'
    accept_idx = particle_selection.select(mrf, map_states, msg_passing, msgs, \
        x_aug, nParticles, node_pot, edge_pot, temp)
    x = {v: [x_aug[v][i] for i in accept_idx[v]] for v in mrf.nodes}

    # Callback
    if callback != None:
      pass

    # If the difference in logP between this iteration and the previous is less
    # than conv_tol, then we have converged
    if i > 1 and np.abs(stats['logP'][-1] - stats['logP'][-2]) < conv_tol:
      stats['converged'] = True
      stats['last_iter'] = i
      break

  # Run final message passing
  node_pot, edge_pot = _calc_potentials(x, mrf)

  msgs, msg_passing_stats = msg_passing.messages(node_pot, edge_pot)
  node_bel_aug = msg_passing.log_beliefs(node_pot, edge_pot, msgs)
  map_states, n_ties = msg_passing.decode_MAP_states(node_pot, edge_pot, \
      node_bel_aug)
  xMAP = {v: x[v][map_states[v]] for v in mrf.nodes}

  return xMAP, x, stats

def test_dpmp_infer():
  np.random.seed(0)

  mrf = MRF([0, 1], [(0, 1)],
            lambda _1, x: -(x ** 2),
            lambda _1, _2, x, y: -((x - y) ** 2))
  # x0 = {0: [-1, 1], 1: [-1, 1]}
  x0 = {0: [0.0], 1: [0.0]}
  nParticles = 5

  def proposal(x, mrf, nParticlesAdd):
    return {v: list(100 * np.random.randn(nParticlesAdd[v])) for v in mrf.nodes}

  xMAP, x, stats = DPMP_infer(mrf, x0, nParticles, proposal, \
      SelectDiverse(), MaxSumBP(mrf), max_iters=50)

  assert xMAP == {0: 0.0, 1: 0.0}

if __name__ == '__main__':
  test_dpmp_infer()
