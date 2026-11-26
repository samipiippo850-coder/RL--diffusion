"""Bounded interface check; not an evaluation of playing strength."""
import numpy as np
from Simulator import TFTConfig, parallel_env

env = parallel_env(TFTConfig(num_players=8))
obs, infos = env.reset(seed=42)
print('agents', len(obs), 'observation_keys', list(next(iter(obs.values())).keys()))
for step in range(10):
    actions = {}
    for agent in env.agents:
        mask = np.asarray(obs[agent]['action_mask']).reshape(-1)
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError('No legal action for ' + agent)
        actions[agent] = int(legal[0])
    obs, rewards, terms, truncs, infos = env.step(actions)
print('steps', step + 1, 'active_agents', len(env.agents))
env.close()
