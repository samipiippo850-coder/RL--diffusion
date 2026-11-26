import math
import unittest
from gibbs import diagnostics, gibbs_weights


class GibbsTests(unittest.TestCase):
    def test_uniform_and_shift_invariance(self):
        self.assertEqual(gibbs_weights([1, 2, 3], 0), [1 / 3] * 3)
        a = gibbs_weights([1, 2, 3], 2)
        b = gibbs_weights([101, 102, 103], 2)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y)

    def test_concentration_and_stability(self):
        d = diagnostics([0, 1000, 2000], 1000)
        self.assertTrue(math.isfinite(d['ess']))
        self.assertAlmostEqual(d['ess'], 1)
        self.assertAlmostEqual(d['weighted_cost'], 0)
        self.assertAlmostEqual(d['kl_to_uniform'], math.log(3))

    def test_invalid(self):
        for costs, beta in [([], 1), ([float('nan')], 1), ([1], -1), ([1], float('inf'))]:
            with self.assertRaises(ValueError):
                gibbs_weights(costs, beta)


if __name__ == '__main__':
    unittest.main()
