import numpy as np
import random

def random_walk_proposal(cov):
  """Random walk proposal with covariance `cov`.

  Parameters
  ----------
  cov : d x d matrix

  Returns
  -------
  proposal : function
      A proposal function which randomly selects points and adds Gaussian random
      noise to them with covariance `cov`.
  """
  cov = np.array(cov)
  d = cov.shape[0]
  mu = np.zeros(d)

  def proposal(mrf, nAdd, x):
    x_prop = {}
    for v in mrf.nodes:
      x_prop[v] = [np.random.multivariate_normal(mu, cov) + random.choice(x[v])
                   for _ in range(nAdd[v])]
    return x_prop

  return proposal

def random_walk_proposal_1d(sig):
  """Random walk proposal for 1-dimensional models.

  Parameters
  ----------
  sig : float
      The standard deviation of the random walk.

  Returns
  -------
  proposal : function
      A random walk proposal function.
  """
  def proposal(mrf, nAdd, x):
    x_prop = {}
    for v in mrf.nodes:
      x_prop[v] = [sig * np.random.randn() + random.choice(x[v])
                   for _ in range(nAdd[v])]
    return x_prop

  return proposal

def mixture_proposal(props, weights=None):
  """Mixture of multiple proposals.

  Parameters
  ----------
  props : list of proposal functions
  weights : None or a list of probabilities which sum to 1 (default: None)
      If None, then proposals will be selected uniformly.

  Returns
  -------
  proposal : function
      A proposal function which will select a new proposal at random on each
      invocation. In other words, at each iteration of D-PMP one of the given
      proposals will be chosen at random according to the weights and applied.
  """
  ws = (1.0 / len(props)) * np.ones(len(props)) \
       if weights == None else weights
  def proposal(mrf, nAdd, x):
    return np.random.choice(props, p=ws)(mrf, nAdd, x)
  return proposal
