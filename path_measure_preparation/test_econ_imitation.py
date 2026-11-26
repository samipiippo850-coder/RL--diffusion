import unittest
import numpy as np
from tft_econ_imitation import inspect_episode,normalizer

class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.m=np.zeros(55*38,dtype=bool);self.m[52*38]=True
        self.meta={'reason':'natural_end','rank':4,'internal_warning_count':0,'max_pending_items_per_player':0}
        self.row=(np.array([1.,2.],dtype=np.float32),self.m,52*38)

    def test_clean_pass_is_preserved(self):
        self.assertIsNone(inspect_episode(self.meta,[self.row,self.row]))

    def test_reject_bad_game_and_bad_action(self):
        self.assertEqual(inspect_episode({**self.meta,'reason':'time_limit'},[self.row]),'not_natural_end')
        self.assertEqual(inspect_episode({**self.meta,'internal_warning_count':1},[self.row]),'simulator_warning')
        self.assertEqual(inspect_episode(self.meta,[(self.row[0],self.m,0)]),'invalid_mask_or_action')
        self.assertEqual(inspect_episode(self.meta,[(np.array([np.nan]),self.m,52*38)]),'nonfinite_observation')

    def test_normalizer_uses_training_samples(self):
        ep=(np.array([[1.,3.],[3.,3.]],dtype=np.float32),None,None,None)
        mean,std=normalizer([ep])
        np.testing.assert_allclose(mean,[2.,3.]);np.testing.assert_allclose(std,[1.,1.])

if __name__=='__main__':unittest.main()
