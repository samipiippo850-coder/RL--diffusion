import unittest
import numpy as np
from tft_dagger_experiment import clean,summary

class DaggerDataTests(unittest.TestCase):
    def test_clean_checks_expert_and_executed_action(self):
        m=np.zeros(55*38,dtype=bool);m[52*38]=True;m[53*38]=True
        meta={'reason':'natural_end','rank':4,'internal_warning_count':0,'max_pending_items_per_player':0}
        row=(np.ones(3,dtype=np.float32),m,52*38,53*38)
        self.assertIsNone(clean(meta,[row],True))
        self.assertEqual(clean(meta,[(row[0],m,row[2],0)],True),'invalid_executed_action')
        self.assertEqual(clean(meta,[(row[0],m,0,row[3])],True),'invalid_mask_or_action')

    def test_summary_retains_incomplete_attempts(self):
        x=summary([{'reason':'natural_end','rank':2,'target_action_counts':{'pass':3}},
                   {'reason':'time_limit'}])
        self.assertEqual(x['attempted'],2);self.assertEqual(x['valid'],1)
        self.assertEqual(x['mean_rank'],2);self.assertEqual(x['action_fraction']['pass'],1)

if __name__=='__main__':unittest.main()
