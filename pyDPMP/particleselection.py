import numpy as np

from .mrf import neighboring_factors

def message_foundation(mrf, msg_passing, msgs, pots, t):
  mats = []
  for fid in neighboring_factors(mrf, t):
    # Messages from all other factors (T,)
    incoming_msg = msg_passing.vf_message(pots, msgs, t, fid)

    factor = mrf.factors[fid]
    pot = pots[fid]

    # The axis of the pot matrix corresponding to t
    ax_t = factor.nodes.index(t)

    # Roll the ax_t axis to be the first
    rolled_pot = np.rollaxis(pot, ax_t)

    # Flatten out all other axes (T x <whatever>)
    flat_pot = rolled_pot.reshape((rolled_pot.shape[0], -1))

    # Calculate final message foundation matrix and add to mats
    mats.append(flat_pot.T + incoming_msg.reshape((1, len(incoming_msg))))

  return np.vstack(mats)

class ParticleSelectionScheme(object):
  def select(self, mrf, nStates, map_state, msg_passing, msgs, nSelect, pots, temp):
    pass

class SelectDiverse(ParticleSelectionScheme):
  def select(self, mrf, nStates, map_state, msg_passing, msgs, nSelect, pots, temp):
    """Run particle selection from the ICML 2014 paper (dpmpmax)."""
    I_accept = {v: [] for v in mrf.nodes}
    for t in mrf.nodes:
      # Build the "stacked" message foundations matrix (S x T)
      M_t = message_foundation(mrf, msg_passing, msgs, pots, t)

      logMstar = np.max(M_t, axis=1)                        # (S,)
      logZ = 1.0 / temp * np.max(M_t)                       # scalar
      Mstar = np.exp(1.0 / temp * logMstar - logZ)          # (S,)
      Psi = np.exp(1.0 / temp * M_t - logZ)                 # (S, T)

      # Select first particle
      b = map_state[t]
      Mhat = Psi[:, b]                                      # (S,)
      delta = Mstar - Mhat                                  # (S,)
      I_accept[t].append(b)

      while len(I_accept[t]) < nSelect[t] and np.max(delta) > 0:
        # Get unused particles
        b_used = set(I_accept[t])
        b_unused = [bb for bb in range(nStates[t]) if bb not in b_used]

        # Select next particle
        a_star = np.argmax(delta)
        idx_max = np.argmax(Psi[a_star, b_unused])
        b = b_unused[idx_max]
        I_accept[t].append(b)

        # Update message approximation
        Mhat = np.maximum(Mhat, Psi[:, b])
        delta = Mstar - Mhat
    return I_accept

class SelectLazyGreedy(ParticleSelectionScheme):
  def select(self, mrf, nStates, map_state, msg_passing, msgs, nSelect, pots, temp):
    """Run particle selection from the ICML 2015 paper (dpmpsum/LazyGreedy)."""
    I_accept = {v: [] for v in mrf.nodes}
    for v in mrf.nodes:
      # Build the "stacked" message foundations matrix (S x T)
      M_t = message_foundation(mrf, msg_passing, msgs, pots, v)

      # logMstar = np.max(M_t, axis=1)                        # (S,)
      logZ = 1.0 / temp * np.max(M_t)                       # scalar
      # Mstar = np.exp(1.0 / temp * logMstar - logZ)          # (S,)
      Psi = np.exp(1.0 / temp * M_t - logZ)                 # (S, T)

      # Select first particle
      delta = np.sum(Psi, axis=0)
      b = map_state[v]
      I_accept[v].append(b)
      Mhat = Psi[:, b]

      deltaMax = np.inf
      while (len(I_accept[v]) < nSelect[v]) and (deltaMax > 0):
        # Get unused particles
        b_used = set(I_accept[v])
        b_unused = [bb for bb in range(nStates[v]) if bb not in b_used]

        # Pick next particle
        stale = []
        b_max = None
        for _ in range(len(b_unused) + 1):
          # Pick highest margin
          idx_max = np.argmax(delta[b_unused])
          deltaMax = delta[b_unused][idx_max]
          b_max = b_unused[idx_max]
          if b_max in stale:
            break

          # Recompute margin
          delta[b_max] = np.sum(np.maximum(Mhat, Psi[:, b_max])) - np.sum(Mhat)
          stale.append(b_max)

        # Add particle
        if deltaMax > 0:
          I_accept[v].append(b_max)
          Mhat = np.maximum(Mhat, Psi[:, b_max])

    return I_accept
