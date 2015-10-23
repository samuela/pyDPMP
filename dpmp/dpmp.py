from mrf import calc_potentials, log_prob_states

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
    node_pot, edge_pot = calc_potentials(x_aug, mrf)

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
    if i > 1 and abs(stats['logP'][-1] - stats['logP'][-2]) < conv_tol:
      stats['converged'] = True
      stats['last_iter'] = i
      break

  # Run final message passing
  node_pot, edge_pot = calc_potentials(x, mrf)

  msgs, msg_passing_stats = msg_passing.messages(node_pot, edge_pot)
  node_bel_aug = msg_passing.log_beliefs(node_pot, edge_pot, msgs)
  map_states, n_ties = msg_passing.decode_MAP_states(node_pot, edge_pot, \
      node_bel_aug)
  xMAP = {v: x[v][map_states[v]] for v in mrf.nodes}

  return xMAP, x, stats
