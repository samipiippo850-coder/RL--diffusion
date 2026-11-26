import unittest
import torch
from agent_measure.hierarchical_policy import HierarchicalPolicy,HierarchicalDenoiser,action_families

class HierarchicalPolicyTests(unittest.TestCase):
    def test_masked_distribution_and_family_normalization(self):
        torch.manual_seed(2)
        model=HierarchicalPolicy(4)
        mask=torch.zeros(3,55*38,dtype=torch.bool)
        mask[0,0]=1;mask[0,52*38]=1
        mask[1,53*38]=1
        mask[2,37*38+3]=1;mask[2,47*38]=1;mask[2,54*38]=1
        logp=model(torch.randn(3,4),mask)
        probs=logp.exp()
        self.assertTrue(torch.allclose(probs.sum(1),torch.ones(3),atol=1e-6))
        self.assertEqual(float(probs[~mask].sum()),0.)
        self.assertTrue((action_families()[mask[0]]==torch.tensor([0,4])).all())
        (-logp[0,52*38]).backward()
        self.assertTrue(torch.isfinite(model.family.weight.grad).all())

    def test_diffusion_samples_only_legal(self):
        model=HierarchicalDenoiser(4)
        mask=torch.zeros(2,55*38,dtype=torch.bool);mask[0,52*38]=1;mask[1,47*38]=1
        x=torch.randn(2,4)
        a=model.sample(x,mask)
        self.assertTrue(mask.gather(1,a[:,None]).all())
        loss=model.loss(x,a,mask).mean();loss.backward()
        self.assertTrue(torch.isfinite(loss))

if __name__=='__main__': unittest.main()
