import numpy as np

from .mrf import neighbors

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
                       for s in neighbors(mrf, t)])

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
        idx_max = np.argmax(Psi[a_star, b_unused])
        b = b_unused[idx_max]
        I_accept[t].append(b)

        # Update message approximation
        Mhat = np.maximum(Mhat, Psi[:,b])
        delta = Mstar - Mhat
    return I_accept

class SelectLazyGreedy(ParticleSelectionScheme):
  pass
