import numpy as np
from scipy.stats import norm
import time
import random

from pyDPMP.messagepassing import MaxSumMP, fwd_bwd_sched
from pyDPMP.mrf import MRF
from pyDPMP.particleselection import SelectDiverse
# from pyDPMP.proposals import random_walk_proposal_1d
from pyDPMP.util import set_seed
from pyDPMP import DPMP_infer

set_seed(0)

def norm_logpdf(x, mu, sigma):
  return -0.5 * ((x - mu) ** 2) / (sigma ** 2)

sig_u = np.sqrt(10)
sig_v = np.sqrt(1)

T = 50

def trans_x(x_prev, t):
  return x_prev / 2 \
                + 25 * x_prev / (1 + x_prev ** 2) \
                + 8 * np.cos(1.2 * t)

def obs_y(x_t):
  return (x_t ** 2) / 20.0

x_true = np.zeros(T)
y = np.zeros(T)

x_true[0] = sig_u * np.random.randn()
y[0] = obs_y(x_true[0]) + sig_v * np.random.randn()
for t in range(1, T):
  x_true[t] = trans_x(x_true[t-1], t) + sig_u * np.random.randn()
  y[t] = obs_y(x_true[t]) + sig_v * np.random.randn()


nodes = range(T)
edges = [(t, t + 1) for t in range(T - 1)]

node_pot = lambda s, x_s: norm_logpdf(y[s], obs_y(x_s), sig_v)
edge_pot = lambda s, t, x_s, x_t: norm_logpdf(x_t, trans_x(x_s, t), sig_u)

mrf = MRF(nodes, edges, node_pot, edge_pot)

def proposal(x, mrf, nAdd):
  x_prop = {}

    # t = 0
  x_prop[0] = [sig_u * np.random.randn() for _ in range(nAdd[0])]

  for t in range(1, T):
      # Pick random particles from x[t - 1] and propogate them according to
      # the transition dynamics.
      x_prop[t] = [trans_x(random.choice(x[t - 1]), t) + sig_u * np.random.randn()
                   for _ in range(nAdd[t])]

  return x_prop

x0 = {t: [0.0] for t in range(T)}
nParticles = 5

set_seed(0)

def callback(info):
  x = info['x']
  xMAP = info['xMAP']

  plt.clf()
  plt.plot(range(T), x_true)
  plt.plot(range(T), [xMAP[t] for t in range(T)])
#     particles = [(t, x_t) for t in range(T) for x_t in x[t]]
#     plt.scatter(*zip(*particles), s=10, c='r', marker='x')
  plt.title('Iter. %d' % info['iter'])

  time.sleep(1.0)

maxsum = MaxSumMP(mrf, sched=fwd_bwd_sched(mrf))

xMAP, xParticles, stats = DPMP_infer(mrf,
                                     x0,
                                     nParticles,
                                     proposal,
                                     SelectDiverse(),
                                     maxsum,
                                     conv_tol=None,
                                     max_iters=100,
#                                      callback=callback,
                                     verbose=True)
