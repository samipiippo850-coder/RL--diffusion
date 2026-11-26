import unittest
from types import SimpleNamespace
import numpy as np
from tft_segment_pilot import choice_for

class SegmentSelectionTests(unittest.TestCase):
    def test_selects_only_when_both_actions_are_legal(self):
        mask=np.zeros(55*38,dtype=bool);mask[53*38]=True;mask[54*38]=True
        player=SimpleNamespace(health=30,round=8)
        self.assertEqual(choice_for('level_refresh',mask,player),{'level':53*38,'refresh':54*38})
        self.assertIsNone(choice_for('low_health',mask,player))
        mask[52*38]=True
        self.assertEqual(choice_for('low_health',mask,player),{'refresh':54*38,'pass':52*38})
        self.assertIsNone(choice_for('low_health',mask,SimpleNamespace(health=41,round=8)))
        self.assertIsNone(choice_for('level_refresh',mask,SimpleNamespace(health=20,round=5)))

if __name__=='__main__':unittest.main()
