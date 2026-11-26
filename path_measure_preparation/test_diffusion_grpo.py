import unittest
import torch
from agent_measure.discrete_diffusion import CategoricalDenoiser, reverse_posterior
from agent_measure.diffusion_grpo import path_log_prob, relative_advantages, clipped_loss

class DiffusionGRPOTests(unittest.TestCase):
    def test_joint_path_logprob_matches_sampling_transitions_and_gradients(self):
        torch.manual_seed(4)
        model=CategoricalDenoiser(3,5,steps=4)
        x=torch.randn(2,3); mask=torch.tensor([[1,1,0,1,0],[1,1,1,0,0]],dtype=torch.bool)
        xt=torch.tensor([0,1]); chain=[xt]; expected=torch.zeros(2)
        for t in range(4,0,-1):
            p0=model(x,xt,torch.full((2,),t,dtype=torch.long),mask).softmax(-1)
            p=reverse_posterior(p0,xt,mask,model.alpha,model.alpha**(t-1))
            nxt=torch.multinomial(p,1).squeeze(1)
            expected=expected+p.gather(1,nxt[:,None]).squeeze(1).log()
            xt=nxt; chain.append(xt)
        actual=path_log_prob(model,x,mask,torch.stack(chain,dim=1))
        self.assertTrue(torch.allclose(actual,expected))
        actual.sum().backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None),0)

    def test_relative_advantages_ties_and_clip(self):
        self.assertTrue(torch.all(relative_advantages(torch.tensor([1.,1.,1.,1.]))==0))
        a=relative_advantages(torch.tensor([0.,1.,2.,3.]))
        self.assertAlmostEqual(float(a.mean()),0.,places=6)
        old=torch.zeros(4); new=torch.zeros(4,requires_grad=True)
        clipped_loss(new,old,a).backward()
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertGreater(float(new.grad.abs().sum()),0)

if __name__=='__main__': unittest.main()
