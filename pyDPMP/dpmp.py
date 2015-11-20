from .messagepassing import decode_MAP_states
from .mrf import calc_potentials, log_prob_states

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
  proposal : function (mrf, nAdd, x -> list of particles)
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
  xMAP : dict (v -> state)
      The MAP decoding found.
  x : dict (v -> list of particles)
      The final particle set.
  stats : dict
      Contains
        - 'logP', a list of the log probabilities of the MAP configurations at
          each iteration.
        - 'n_ties', the number of message passing ties at each iteration.
        - 'msg_passing_stats', the message passing stats at each iteration.
        - 'callback_results', the result of the callback function, if it exists,
          at each iteration.
        - 'converged', a boolean specifying whether DPMP converged.
        - 'last_iter', the number of iterations until convergence, if DPMP
          converged.
  """

  # Handle default arguments
  if isinstance(nParticles, int):
    nParticles = {v: nParticles for v in mrf.nodes}
  if nAugmented == None:
    nAugmented = {v: 2 * nParticles[v] for v in mrf.nodes}
  elif isinstance(nAugmented, int):
    nAugmented = {v: nAugmented for v in mrf.nodes}

  # Start with initial particle set
  x = x0

  stats = {
    'logP': [],
    'n_ties': [],
    'msg_passing_stats': [],
    'callback_results': [],
    'converged': False,
    'last_iter': None
  }

  for i in range(max_iters):
    if verbose: print('Iter', i)

    # Propose new particles
    if verbose: print('    ... Proposing new particles')
    nParticlesAdd = {v: nAugmented[v] - len(x[v]) for v in mrf.nodes}
    x_prop = proposal(mrf, nParticlesAdd, x)

    # Construct augmented particle set
    x_aug = {v: x[v] + list(x_prop[v]) for v in mrf.nodes}

    # Calculate potentials on the augmented particle set
    if verbose: print('    ... Calculating potentials and MAP')
    if verbose: print('        ... potentials')
    node_pot, edge_pot = calc_potentials(mrf, x_aug)

    # Calculate messages, log beliefs, and MAP states
    if verbose: print('        ... message passing')
    msgs, msg_passing_stats = msg_passing.messages(node_pot, edge_pot)
    node_bel_aug, _ = msg_passing.log_beliefs(node_pot, edge_pot, msgs)
    map_states, n_ties = decode_MAP_states(mrf, node_bel_aug)
    logP_map = log_prob_states(mrf, node_pot, edge_pot, map_states)
    xMAP = {v: x_aug[v][map_states[v]] for v in mrf.nodes}

    stats['logP'].append(logP_map)
    stats['n_ties'].append(n_ties)
    stats['msg_passing_stats'].append(msg_passing_stats)

    # Particle selection
    if verbose: print('    ... Selecting particles')
    accept_idx = particle_selection.select(mrf, map_states, msg_passing, msgs, \
        x_aug, nParticles, node_pot, edge_pot, temp)
    x_sel = {v: [x_aug[v][i] for i in accept_idx[v]] for v in mrf.nodes}

    # Set particles to be the selected particle set
    x = x_sel

    # Callback
    if callback != None:
      cb_res = callback({
        'mrf': mrf,
        'x0': x0,
        'nParticles': nParticles,
        'proposal': proposal,
        'particle_selection': particle_selection,
        'msg_passing': msg_passing,
        'max_iters': max_iters,
        'conv_tol': conv_tol,
        'nAugmented': nAugmented,
        'temp': temp,
        'verbose': verbose,

        'iter': i,
        'x': x,
        'x_prop': x_prop,
        'x_aug': x_aug,
        'x_sel': x_sel,
        'xMAP': xMAP,

        'stats': stats
      })
      stats['callback_results'].append(cb_res)

    # If the difference in logP between this iteration and the previous is less
    # than conv_tol, then we have converged
    if i > 0 and abs(stats['logP'][-1] - stats['logP'][-2]) < conv_tol:
      stats['converged'] = True
      stats['last_iter'] = i
      break

  # Run final message passing
  node_pot, edge_pot = calc_potentials(mrf, x)

  msgs, msg_passing_stats = msg_passing.messages(node_pot, edge_pot)
  node_bel_aug, _ = msg_passing.log_beliefs(node_pot, edge_pot, msgs)
  map_states, n_ties = decode_MAP_states(mrf, node_bel_aug)
  xMAP = {v: x[v][map_states[v]] for v in mrf.nodes}

  return xMAP, x, stats
