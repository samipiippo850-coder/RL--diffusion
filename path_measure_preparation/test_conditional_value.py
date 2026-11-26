import unittest
import numpy as np
from agent_measure.conditional_value import crossfit, predict_all_families


class ConditionalValueTests(unittest.TestCase):
    def test_game_held_out_and_finite(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(80, 82))
        game = np.repeat(np.arange(4), 20)
        actions = np.tile(np.arange(20), 4)
        y = np.repeat([0.2, 0.4, 0.6, 0.8], 20)
        out = crossfit(x, actions, y, game)
        self.assertEqual(len(out['per_game']), 4)
        self.assertTrue(np.isfinite(out['residual']).all())
        self.assertTrue(np.isfinite(predict_all_families(x[0], out['coef'])).all())
        # Every game gets a prediction from the other games, not its own labels.
        for g in range(4):
            self.assertEqual(out['per_game'][str(g)]['n'], 20)
        with self.assertRaises(ValueError):
            crossfit(x[:20], actions[:20], y[:20], game[:20])


if __name__ == '__main__':
    unittest.main()
