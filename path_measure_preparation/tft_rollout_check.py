"""Bounded natural-termination probe; no inferred rankings for truncations."""
import time
import numpy as np
from Simulator import TFTConfig, parallel_env
from tft_pilot_train import teacher

env = parallel_env(TFTConfig(num_players=8))
obs, info = env.reset(seed=7001)
rng = np.random.default_rng(7001)
start = time.monotonic()
for step in range(1200):
    if not env.agents:
        print('NATURAL_NO_AGENTS', step, flush=True)
        break
    actions = {a: teacher(np.asarray(obs[a]['action_mask'], dtype=bool), rng)
               for a in env.agents if a in obs and obs[a] is not None}
    if not actions:
        print('NO_ACTIONS', step, flush=True)
        break
    obs, rewards, terms, truncs, info = env.step(actions)
    if any(terms.values()):
        print('terminal_details', [(a, rewards.get(a), getattr(info.get(a, {}).get('player'), 'health', None),
                                    getattr(info.get(a, {}).get('player'), 'reward', None))
                                   for a, done in terms.items() if done], flush=True)
    if step % 25 == 0 or any(terms.values()):
        print('step', step+1, 'agents', len(env.agents), 'terminations', sum(terms.values()),
              'truncations', sum(truncs.values()), 'rounds', sorted(set(
                  x.get('game_round') for x in info.values() if x.get('game_round') is not None)),
              'elapsed', round(time.monotonic()-start, 2), flush=True)
    if time.monotonic()-start > 120:
        print('TIME_LIMIT', step+1, flush=True)
        break
else:
    print('STEP_LIMIT', flush=True)
env.close()
