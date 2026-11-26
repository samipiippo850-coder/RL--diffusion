import unittest
import numpy as np
import torch
from agent_measure.discrete_diffusion import CategoricalDenoiser, guided_logits, reverse_posterior


class DiscreteDiffusionTests(unittest.TestCase):
    def test_reverse_posterior_matches_brute_force(self):
        p0 = torch.tensor([[0.2, 0.3, 0.5, 0.0]], dtype=torch.float64)
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        xt = torch.tensor([1])
        alpha, abar_prev = 0.7, 0.4
        actual = reverse_posterior(p0, xt, mask, alpha, abar_prev).numpy()[0]
        expected = np.zeros(4)
        for x0 in range(3):
            denom = alpha*abar_prev*(xt.item() == x0) + (1-alpha*abar_prev)/3
            for prev in range(3):
                likelihood = alpha*(prev == xt.item()) + (1-alpha)/3
                prior = abar_prev*(prev == x0) + (1-abar_prev)/3
                expected[prev] += p0[0, x0].item() * likelihood * prior / denom
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(actual[3], 0)

    def test_guidance_replicator_and_mask(self):
        logits = torch.tensor([[0., 0., 99.]], dtype=torch.float64)
        q = torch.tensor([[1., -1., 999.]], dtype=torch.float64)
        mask = torch.tensor([[True, True, False]])
        p = torch.softmax(guided_logits(logits, mask, q, 0), -1)
        delta = 1e-5
        p2 = torch.softmax(guided_logits(logits, mask, q, delta), -1)
        field = p * (q - (p*q).sum(-1, keepdim=True))
        np.testing.assert_allclose(((p2-p)/delta).numpy(), field.numpy(), atol=1e-5)
        self.assertEqual(p2[0, 2].item(), 0)

    def test_sample_and_loss_respect_legal_mask(self):
        torch.manual_seed(9)
        model = CategoricalDenoiser(3, 4)
        obs = torch.randn(20, 3)
        mask = torch.tensor([[True, False, True, False]]*20)
        sample = model.sample(obs, mask)
        self.assertTrue(torch.all(mask.gather(1, sample[:, None])))
        loss = model.loss(obs, sample, mask)
        self.assertTrue(torch.isfinite(loss).all())


if __name__ == '__main__':
    unittest.main()
