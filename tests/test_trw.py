# import numpy as np
#
# from pyDPMP.messagepassing import TreeReweightedMP, decode_MAP_states
# from pyDPMP.mrf import calc_potentials
# from pyDPMP.util import set_seed
#
# from .test_util import random_mrf, random_tree_mrf, mrf_brute_MAP
#
# def check_trw_tree(mrf):
#   x = {v: [0, 1] for v in mrf.nodes}
#
#   node_pot, edge_pot = calc_potentials(mrf, x)
#
#   rho = {(s, t): 1 for s in mrf.nodes for t in mrf.nodes}
#
#   trw = TreeReweightedMP(mrf, rho)
#   msgs, stats = trw.messages(node_pot, edge_pot)
#   node_bel, _ = trw.log_beliefs(node_pot, edge_pot, msgs)
#   map_state, _ = decode_MAP_states(mrf, node_bel)
#   xMAP = {v: x[v][map_state[v]] for v in mrf.nodes}
#
#   brute_MAP = mrf_brute_MAP(mrf, node_pot, edge_pot)
#
#   assert stats['converged'] == True
#   assert stats['error'][-1] < 1e-10
#   assert xMAP == brute_MAP
#
# # For some reason decorating a test generator doesn't work. See
# # https://github.com/nose-devs/nose/issues/958.
# # @seeded
# def test_trw_trees():
#   """Test that TreeReweightedMP converges on random tree-structured graphs."""
#   set_seed(0)
#
#   for _ in range(100):
#     mrf = random_tree_mrf(10)
#     yield check_trw_tree, mrf
#
# def check_trw_bound_general(mrf, rho):
#   x = {v: [0, 1] for v in mrf.nodes}
#   node_pot, edge_pot = calc_potentials(mrf, x)
#
#   # Pick random edge appearances
#   trw = TreeReweightedMP(mrf, rho)
#   _, stats = trw.messages(node_pot, edge_pot, calc_logP=True)
#
#   assert all([logP <= bound + 1e-10
#               for (logP, bound) in zip(stats['logP'], stats['logPbound'])])
#
# def test_trw_general_half_rho():
#   """Test that TreeReweightedMP works on random (general) graphs and the bound
#   works properly. rho = 0.5"""
#   set_seed(0)
#
#   for _ in range(100):
#     mrf = random_mrf(5, 0.5)
#     rho = {(s, t): 0.5 for s in mrf.nodes for t in mrf.nodes}
#     yield check_trw_bound_general, mrf, rho
#
# def test_trw_general_random_rho():
#   """Test that TreeReweightedMP works on random (general) graphs and the bound
#   works properly. rho is random"""
#   set_seed(0)
#
#   for _ in range(100):
#     V = 5
#     mrf = random_mrf(V, 0.5)
#     rho_mat = np.random.rand(V, V)
#     rho = {(s, t): rho_mat[min(s, t), max(s, t)]
#            for s in mrf.nodes for t in mrf.nodes}
#     yield check_trw_bound_general, mrf, rho
