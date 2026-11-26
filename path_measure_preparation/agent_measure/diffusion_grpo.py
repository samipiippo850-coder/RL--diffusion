"""Group-relative policy gradients on sampled categorical denoising paths.

The clipped ratio is for the *joint latent denoising path* of each game action,
not an exact marginal action likelihood. This is a DDPO-style extension, not a
claim to reproduce the original language-model GRPO objective exactly.
"""
import torch
from agent_measure.discrete_diffusion import reverse_posterior


def path_log_prob(model, obs, mask, chains):
    """Log p_theta(x[K-1:0] | x[K], obs); initial uniform is parameter-free."""
    if chains.shape != (len(obs), model.steps + 1):
        raise ValueError('unexpected chain shape')
    total = torch.zeros(len(obs), device=obs.device)
    for j, t in enumerate(range(model.steps, 0, -1)):
        xt = chains[:, j].long()
        next_xt = chains[:, j+1].long()
        ts = torch.full((len(obs),), t, dtype=torch.long, device=obs.device)
        p0 = torch.softmax(model(obs, xt, ts, mask), dim=-1)
        probs = reverse_posterior(p0, xt, mask, model.alpha, model.alpha**(t-1))
        total = total + torch.log(probs.gather(1, next_xt[:, None]).squeeze(1).clamp_min(1e-30))
    return total


def relative_advantages(returns, eps=1e-6):
    """One reward per same-start rollout; constant-return groups give zero."""
    if len(returns) < 2:
        raise ValueError('need at least two rollouts')
    return (returns - returns.mean()) / (returns.std(unbiased=False) + eps)


def clipped_loss(new_logp, old_logp, advantage, clip=0.2):
    log_ratio = (new_logp - old_logp).clamp(-20, 20)
    ratio = log_ratio.exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage).mean()
