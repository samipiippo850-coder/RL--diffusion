"""Leave-one-game-out conditional return estimates and residual diagnostics."""
import numpy as np

N_FAMILIES = 55


def design(observations, actions):
    x = np.asarray(observations, dtype=np.float64)
    a = np.asarray(actions, dtype=np.int64)
    if x.ndim != 2 or x.shape[1] < 82 or a.shape != (len(x),) or np.any((a < 0) | (a >= 2090)):
        raise ValueError('invalid state or action shapes')
    # The frozen ObservationToken schema puts 76 scalars before 6 embedding scalars.
    scalars = x[:, -82:-6]
    family = a // 38
    return np.concatenate((np.ones((len(x), 1)), scalars, np.eye(N_FAMILIES)[family]), axis=1)


def _fit(X, y, game_id, ridge):
    # Equal total weight per game, avoiding long games dominating the fit.
    _, inverse, counts = np.unique(game_id, return_inverse=True, return_counts=True)
    weight = 1 / counts[inverse]
    weight = weight / weight.sum()
    penalty = np.eye(X.shape[1]) * ridge
    penalty[0, 0] = 0
    return np.linalg.solve(X.T @ (weight[:, None]*X) + penalty, X.T @ (weight*y))


def crossfit(observations, actions, returns, game_id, ridge=0.05):
    X = design(observations, actions)
    y = np.asarray(returns, dtype=np.float64)
    game = np.asarray(game_id)
    if y.shape != (len(X),) or game.shape != (len(X),) or len(np.unique(game)) < 3:
        raise ValueError('at least 3 distinct games and aligned returns are required')
    prediction = np.empty(len(X), dtype=np.float64)
    per_game = {}
    for g in np.unique(game):
        test = game == g
        coefficient = _fit(X[~test], y[~test], game[~test], ridge)
        prediction[test] = np.clip(X[test] @ coefficient, 0, 1)
        residual = y[test]-prediction[test]
        per_game[str(g)] = {'n': int(test.sum()), 'mean_residual': float(residual.mean()),
                            'mse': float(np.mean(residual**2))}
    residual = y-prediction
    global_mse = float(np.mean([v['mse'] for v in per_game.values()]))
    family = np.asarray(actions, dtype=np.int64)//38
    mse = np.full(N_FAMILIES, global_mse, dtype=np.float64)
    counts = np.bincount(family, minlength=N_FAMILIES)
    for f in range(N_FAMILIES):
        # Multiple observations in a game are correlated; require 3 games too.
        sel = family == f
        if counts[f] >= 30 and len(np.unique(game[sel])) >= 3:
            mse[f] = np.mean(residual[sel]**2)
    return {'coef': _fit(X, y, game, ridge), 'family_rmse': np.sqrt(mse),
            'family_counts': counts, 'global_oof_mse': global_mse,
            'per_game': per_game, 'residual': residual, 'prediction': prediction}


def predict_all_families(observation, coef):
    obs = np.asarray(observation, dtype=np.float64).reshape(1, -1)
    actions = np.arange(N_FAMILIES, dtype=np.int64)*38
    X = design(np.repeat(obs, N_FAMILIES, axis=0), actions)
    return np.clip(X @ coef, 0, 1)
