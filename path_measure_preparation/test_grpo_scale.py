import unittest
import numpy as np
from tft_grpo_scale import fit_reward,pair_accuracy,summary

class RewardDiagnosticsTests(unittest.TestCase):
    def test_fit_and_heldout_pair_order(self):
        groups={i:{'a':(1,np.array([1.,0.])), 'b':(8,np.array([-1.,0.]))} for i in range(20)}
        w=fit_reward(groups)
        self.assertEqual(pair_accuracy(groups,w)['correct'],20)
        hold={21:{'a':(1,np.array([2.,0.])), 'b':(8,np.array([-2.,0.]))}}
        self.assertEqual(pair_accuracy(hold,w)['correct'],1)

    def test_summary_only_counts_completed_games(self):
        result=summary([{'reason':'natural_end','rank':1,'learned_reward':2},
                        {'reason':'natural_end','rank':8,'learned_reward':0},
                        {'reason':'step_limit'}])
        self.assertEqual(result['attempted'],3)
        self.assertEqual(result['valid'],2)
        self.assertEqual(result['mean_rank'],4.5)
        self.assertAlmostEqual(result['reward_rank_utility_correlation'],1.)

if __name__=='__main__':unittest.main()
