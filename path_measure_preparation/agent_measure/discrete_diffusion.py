"""Masked categorical diffusion for one decision, not a trajectory bridge."""
import torch
from torch import nn
from torch.nn import functional as F


def reverse_posterior(p0, xt, mask, alpha, abar_prev):
    """Mixture of q(x[t-1]|x[t],x[0]) under predicted x[0] probabilities.

    Forward transition retains the previous action with probability alpha and
    otherwise draws uniformly over the *current decision's fixed* legal set.
    """
    legal = mask.bool()
    if not torch.all(legal.any(dim=-1)):
        raise ValueError('each mask must contain a legal action')
    if not torch.all(legal.gather(1, xt[:, None])):
        raise ValueError('xt must be legal')
    p0 = p0.masked_fill(~legal, 0)
    p0 = p0 / p0.sum(-1, keepdim=True)
    count = legal.sum(-1, keepdim=True).to(p0.dtype)
    uniform = legal.to(p0.dtype) / count
    one_xt = F.one_hot(xt, num_classes=p0.shape[-1]).to(p0.dtype)
    abar_t = alpha * abar_prev
    denom = abar_t * one_xt + (1-abar_t) * uniform
    ratio = torch.where(legal, p0 / denom.clamp_min(1e-30), torch.zeros_like(p0))
    mixture = abar_prev * ratio + (1-abar_prev) * uniform * ratio.sum(-1, keepdim=True)
    likelihood = alpha * one_xt + (1-alpha) * uniform
    posterior = mixture * likelihood
    return posterior / posterior.sum(-1, keepdim=True)


def guided_logits(logits, mask, q_values, eta=0., uncertainty=None):
    """Exponential policy tilt: first derivative at eta=0 is replicator field."""
    if uncertainty is not None:
        q_values = q_values / (1.0 + uncertainty.clamp_min(0))
    return (logits + eta * q_values).masked_fill(~mask.bool(), -1e9)


class CategoricalDenoiser(nn.Module):
    def __init__(self, obs_dim, n_actions, steps=4, alpha=0.1, hidden=128):
        super().__init__()
        self.n_actions, self.steps, self.alpha = n_actions, steps, alpha
        self.action_embedding = nn.Embedding(n_actions, 16)
        self.time_embedding = nn.Embedding(steps+1, 8)
        self.net = nn.Sequential(nn.Linear(obs_dim+24, hidden), nn.SiLU(), nn.Linear(hidden, n_actions))

    def forward(self, obs, xt, t, mask):
        z = torch.cat((obs, self.action_embedding(xt), self.time_embedding(t)), dim=-1)
        return self.net(z).masked_fill(~mask.bool(), -1e9)

    def loss(self, obs, action, mask, t=None):
        batch = len(action)
        if t is None:
            t = torch.randint(1, self.steps+1, (batch,), device=obs.device)
        uniform = mask.float() / mask.sum(-1, keepdim=True)
        random_action = torch.multinomial(uniform, 1).squeeze(1)
        keep = torch.rand(batch, device=obs.device) < (self.alpha ** t.float())
        xt = torch.where(keep, action, random_action)
        return F.cross_entropy(self(obs, xt, t, mask), action, reduction='none')

    @torch.no_grad()
    def sample(self, obs, mask, q_values=None, eta=0., uncertainty=None):
        mask = mask.bool()
        uniform = mask.float() / mask.sum(-1, keepdim=True)
        xt = torch.multinomial(uniform, 1).squeeze(1)
        for t in range(self.steps, 0, -1):
            ts = torch.full((len(obs),), t, dtype=torch.long, device=obs.device)
            logits = self(obs, xt, ts, mask)
            if q_values is not None:
                logits = guided_logits(logits, mask, q_values, eta, uncertainty)
            p0 = torch.softmax(logits, dim=-1)
            posterior = reverse_posterior(p0, xt, mask, self.alpha, self.alpha**(t-1))
            xt = torch.multinomial(posterior, 1).squeeze(1)
        return xt
