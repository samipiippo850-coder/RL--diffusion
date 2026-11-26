import unittest
import numpy as np
from agent_measure.policy import gibbs_policy, replicator_field, diagnostics
from agent_measure.mdp import solve_kl_mdp
from agent_measure.residuals import final_rank_reward, residual_summary


class PolicyTests(unittest.TestCase):
    def test_gibbs_mask_and_temperature(self):
        q, mask, ref = [0, 1, 1e9], [1, 1, 0], [0.5, 0.5, 0]
        np.testing.assert_allclose(gibbs_policy(q, mask, ref, 0), ref)
        p = gibbs_policy(q, mask, ref, 2)
        np.testing.assert_allclose(p, [1/(1+np.exp(2)), 1/(1+np.exp(-2)), 0])
        self.assertAlmostEqual(p.sum(), 1)
        np.testing.assert_allclose(gibbs_policy([1000, 1001, -1], mask, ref, 2), p)

    def test_support_and_false_fixed_point(self):
        pi = [1., 0.]
        np.testing.assert_array_equal(gibbs_policy([0, 5], [1, 1], pi, 1), pi)
        np.testing.assert_array_equal(replicator_field([0, 5], [1, 1], pi), [0, 0])
        self.assertEqual(diagnostics([0, 5], [1, 1], pi, pi, 1)['best_legal_advantage'], 5)
        self.assertAlmostEqual(replicator_field([0, 1], [1, 1], [0.5, 0.5]).sum(), 0)

    def test_invalid_inputs(self):
        for values, mask, ref, beta in [([1], [0], [1], 1), ([1], [1], [1], -1),
                                        ([1, 2], [1, 0], [0.5, 0.5], 1), ([float('nan')], [1], [1], 1)]:
            with self.assertRaises(ValueError): gibbs_policy(values, mask, ref, beta)

    def test_known_one_state_fixed_point(self):
        p = np.ones((1, 2, 1))
        r = np.array([[0., 1.]])
        mask = np.ones((1, 2), dtype=bool)
        ref = np.array([[0.5, 0.5]])
        result = solve_kl_mdp(p, r, mask, ref, gamma=0.8, beta=2)
        expected_value = np.log((1+np.exp(2))/2)/2/(1-0.8)
        self.assertAlmostEqual(result['value'][0], expected_value, places=8)
        self.assertLess(result['bellman_residual'], 1e-9)
        self.assertLess(diagnostics(result['q'][0], mask[0], result['policy'][0], ref[0], 2)['gibbs_residual_l1'], 1e-12)

    def test_rank_and_residuals(self):
        self.assertEqual(final_rank_reward(1), 1)
        self.assertEqual(final_rank_reward(4), 4/7)
        self.assertEqual(final_rank_reward(8), 0)
        self.assertIsNone(final_rank_reward(None))
        with self.assertRaises(ValueError): final_rank_reward(0)
        summary = residual_summary([0, 1, 1, 0], [0.5]*4, ['a','a','b','b'])
        self.assertEqual(summary['groups']['a']['mean'], 0)
        self.assertEqual(summary['groups']['b']['n'], 2)


if __name__ == '__main__':
    unittest.main()
