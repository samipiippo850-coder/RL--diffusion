"""Policy updates on a finite legal-action simplex."""
import numpy as np


def _inputs(values, mask, reference):
    q = np.asarray(values, dtype=np.float64)
    legal = np.asarray(mask, dtype=bool)
    ref = np.asarray(reference, dtype=np.float64)
    if q.ndim != 1 or legal.shape != q.shape or ref.shape != q.shape or not legal.any():
        raise ValueError('values, mask and reference must be equal-length vectors with a legal action')
    if not np.isfinite(q[legal]).all() or not np.isfinite(ref).all() or np.any(ref < 0):
        raise ValueError('legal values and all reference probabilities must be finite; probabilities nonnegative')
    if np.any(ref[~legal] != 0) or not np.isclose(ref.sum(), 1.0, atol=1e-10, rtol=0):
        raise ValueError('reference must sum to one and assign zero probability to illegal actions')
    return q, legal, ref


def gibbs_policy(values, mask, reference, beta):
    """Argmax <p,Q> - KL(p||reference)/beta for beta>0; beta=0 returns reference.

    Zero reference support is preserved; illegal action values are ignored.
    """
    q, legal, ref = _inputs(values, mask, reference)
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    out = np.zeros_like(ref)
    if beta == 0:
        return ref.copy()
    supported = legal & (ref > 0)
    centered = q[supported] - np.max(q[supported])
    with np.errstate(over='ignore'):
        logw = np.log(ref[supported]) + beta * centered
    logw -= np.max(logw)
    w = np.exp(logw)
    out[supported] = w / w.sum()
    return out


def replicator_field(values, mask, policy):
    """Tangent vector pi*(Q-E_pi[Q]); a zero field need not be optimal."""
    q, legal, pi = _inputs(values, mask, policy)
    field = np.zeros_like(pi)
    mean = np.dot(pi[legal], q[legal])
    field[legal] = pi[legal] * (q[legal] - mean)
    return field


def diagnostics(values, mask, policy, reference, beta):
    q, legal, pi = _inputs(values, mask, policy)
    target = gibbs_policy(q, legal, reference, beta)
    field = replicator_field(q, legal, pi)
    expected = float(np.dot(pi[legal], q[legal]))
    return {'gibbs_residual_l1': float(np.abs(target-pi).sum()),
            'field_l1': float(np.abs(field).sum()),
            'best_legal_advantage': float(np.max(q[legal]) - expected),
            'expected_value': expected}
