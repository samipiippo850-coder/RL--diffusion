"""Reproducible two-state example: python -m agent_measure.demo."""
import json
import numpy as np
from .mdp import solve_kl_mdp
from .policy import diagnostics
from .residuals import residual_summary


def main():
    # A continuing toy system with known transitions, not a game simulator.
    transition = np.array([[[0.8, 0.2], [0.3, 0.7]], [[0.6, 0.4], [0.1, 0.9]]])
    reward = np.array([[0.0, 0.4], [0.2, 0.8]])
    mask = np.ones((2, 2), dtype=bool)
    ref = np.full((2, 2), 0.5)
    result = solve_kl_mdp(transition, reward, mask, ref, gamma=0.8, beta=2)
    checks = [diagnostics(result['q'][s], mask[s], result['policy'][s], ref[s], 2)
              for s in range(2)]
    # Known conditional mean in this synthetic experiment, not a learned estimate.
    rng = np.random.default_rng(42)
    group = np.repeat([0, 1], 5000)
    probability = np.where(group == 0, 0.2, 0.7)
    y = rng.binomial(1, probability)
    summary = residual_summary(y, probability, group)
    output = {'kind': 'synthetic_known_mdp', 'policy': result['policy'].tolist(),
              'value': result['value'].tolist(), 'iterations': result['iterations'],
              'bellman_residual': result['bellman_residual'], 'policy_diagnostics': checks,
              'conditional_residuals': summary}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
