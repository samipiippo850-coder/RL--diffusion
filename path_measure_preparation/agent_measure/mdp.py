"""Exact small discounted MDP control, for a fixed opponent/environment only."""
import numpy as np
from .policy import gibbs_policy


def solve_kl_mdp(transition, reward, legal, reference, gamma=0.9, beta=2.0,
                 tolerance=1e-10, max_iterations=10000):
    """Solve discounted KL-regularized Bellman equations with soft value iteration.

    P[s,a,s'], reward[s,a], legal[s,a], reference[s,a]. Requires beta>0,
    gamma<1 and positive reference on every legal action. No convergence claim
    for learned nonlinear values or multi-agent self-play follows from this.
    """
    p = np.asarray(transition, dtype=float)
    r = np.asarray(reward, dtype=float)
    m = np.asarray(legal, dtype=bool)
    ref = np.asarray(reference, dtype=float)
    if r.ndim != 2 or m.shape != r.shape or ref.shape != r.shape or p.shape != (*r.shape, r.shape[0]):
        raise ValueError('incompatible MDP shapes')
    if not (0 <= gamma < 1 and np.isfinite(beta) and beta > 0 and tolerance > 0 and max_iterations >= 1):
        raise ValueError('require 0<=gamma<1, beta>0, tolerance>0 and max_iterations>=1')
    if not np.isfinite(p).all() or not np.isfinite(r).all() or np.any(p < 0) or np.any(~m.any(axis=1)):
        raise ValueError('invalid transitions, rewards or legal mask')
    if not np.allclose(p[m].sum(axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError('legal transition rows must sum to one')
    if not np.isfinite(ref).all() or np.any(ref[m] <= 0) or np.any(ref[~m] != 0) or not np.allclose(ref.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError('reference must be normalized and positive on legal support only')
    v = np.zeros(r.shape[0])
    for iteration in range(1, max_iterations + 1):
        q = r + gamma * np.einsum('sak,k->sa', p, v)
        next_v = np.empty_like(v)
        for s in range(len(v)):
            qs = q[s, m[s]]
            center = qs.max()
            next_v[s] = center + np.log(np.sum(ref[s, m[s]] * np.exp(beta * (qs-center)))) / beta
        if np.max(np.abs(next_v-v)) <= tolerance:
            v = next_v
            break
        v = next_v
    else:
        raise RuntimeError('value iteration did not converge within max_iterations')
    q = r + gamma * np.einsum('sak,k->sa', p, v)
    policy = np.stack([gibbs_policy(q[s], m[s], ref[s], beta) for s in range(len(v))])
    bellman = np.empty_like(v)
    for s in range(len(v)):
        active = m[s]
        ps = policy[s, active]
        positive = ps > 0
        kl = np.sum(ps[positive] * np.log(ps[positive] / ref[s, active][positive]))
        bellman[s] = np.dot(ps, q[s, active]) - kl / beta
    return {'value': v, 'q': q, 'policy': policy, 'iterations': iteration,
            'bellman_residual': float(np.max(np.abs(bellman-v)))}
