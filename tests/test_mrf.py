import numpy as np

from pyDPMP.mrf import MRF, Factor, calc_potentials, log_prob_states

def test_calc_potentials1():
  mrf = MRF([1], {'f1': Factor([1], lambda x: x - 1)})
  pots = calc_potentials(mrf, {1: np.arange(5)})
  np.testing.assert_almost_equal(pots['f1'], np.arange(5) - 1)

def test_calc_potentials2():
  mrf = MRF([1, 2, 3], {'f1': Factor([1, 2, 3], lambda x, y, z: x + y + z)})
  pots = calc_potentials(mrf, {1: [0, 1], 2: [0, 1], 3: [0, 1]})
  np.testing.assert_almost_equal(
    pots['f1'],
    np.array([[[0, 1],
               [1, 2]],

              [[1, 2],
               [2, 3]]]))

def test_calc_potentials_isingish():
  mrf = MRF([0, 1], {
    'un0': Factor([0], lambda x_s: np.log(x_s + 1)),
    'un1': Factor([1], lambda x_s: np.log(x_s + 1)),
    'pw01': Factor([0, 1], lambda x_s, x_t: np.log(x_s * x_t + 1))
  })

  x = [[0, 1], [0, 1]]

  pots = calc_potentials(mrf, x)

  np.testing.assert_almost_equal(pots['un0'], np.array([0, np.log(2)]))
  np.testing.assert_almost_equal(pots['un1'], np.array([0, np.log(2)]))
  np.testing.assert_almost_equal(
    pots['pw01'],
    np.array([[0, 0], [0, np.log(2)]]))

def test_log_prob_states1():
  mrf = MRF([1], {'f1': Factor([1], lambda x: x - 1)})
  pots = calc_potentials(mrf, {1: np.arange(5)})
  logprob = log_prob_states(mrf, pots, {1: 0})
  np.testing.assert_almost_equal(logprob, -1)
