"""Estimated conditional residual diagnostics, not a proof of a martingale."""
import numpy as np


def residual_summary(outcomes, predicted_conditional_means, groups):
    y = np.asarray(outcomes, dtype=float)
    pred = np.asarray(predicted_conditional_means, dtype=float)
    group = np.asarray(groups)
    if y.ndim != 1 or not len(y) or pred.shape != y.shape or group.shape != y.shape:
        raise ValueError('all inputs must be equal-length, nonempty vectors')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('outcomes and predictions must be finite')
    residual = y-pred
    by_group = {}
    for label in np.unique(group):
        sample = residual[group == label]
        by_group[str(label)] = {'n': len(sample), 'mean': float(sample.mean()),
                                'rmse': float(np.sqrt(np.mean(sample**2)))}
    return {'n': len(y), 'mean': float(residual.mean()),
            'rmse': float(np.sqrt(np.mean(residual**2))), 'groups': by_group}


def final_rank_reward(rank, n_players=8):
    """Terminal rank reward; unknown/truncated rank must be represented by None."""
    if isinstance(n_players, bool) or not isinstance(n_players, (int, np.integer)) or n_players < 2:
        raise ValueError('n_players must be an integer >= 2')
    if rank is None:
        return None
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= n_players:
        raise ValueError('rank must be an integer in [1,n_players], or None')
    return (n_players-rank)/(n_players-1)
