"""Small REINFORCE pilot against a frozen scripted opponent, not diffusion.

Only complete games with a unique verified upstream ranking are accepted.
Upstream assigns simultaneous deaths by internal iteration order, not ties.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from Simulator import TFTConfig, parallel_env
from tft_pilot_train import Policy, features, teacher

BASE = Path('/root/autodl-tmp/path_measure_preparation')
OUTPUT = BASE / 'tft_rl_pilot_20261121'
OPPONENT_VERSION = 'scripted_teacher_v1'


def rollout(model, normalizer, seed, collect=False):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed + 300000)
    opponent_rng = np.random.default_rng(seed + 100000)
    mean, std = normalizer
    env = parallel_env(TFTConfig(num_players=8))
    obs, infos = env.reset(seed=seed)
    alive = set(env.agents)
    ranks = {}
    records = []
    start = time.monotonic()
    reason = 'step_limit'
    for step in range(900):
        if time.monotonic() - start > 120:
            reason = 'time_limit'
            break
        if not env.agents:
            reason = 'natural_end'
            break
        actions = {}
        pending = None
        for agent in env.agents:
            if agent not in obs or obs[agent] is None:
                continue
            mask = np.asarray(obs[agent]['action_mask'], dtype=bool).reshape(-1)
            if not mask.any():
                raise ValueError('empty legal mask')
            if agent == 'player_0':
                x = np.clip((features(obs[agent])-mean)/std, -10, 10).astype(np.float32)
                with torch.no_grad():
                    logits = model(torch.from_numpy(x).unsqueeze(0),
                                   torch.from_numpy(mask.copy()).unsqueeze(0))
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
                pending = (x, mask.copy(), action)
            else:
                action = teacher(mask, opponent_rng)
            if not mask[action]:
                raise AssertionError('illegal action')
            actions[agent] = action
        if not actions:
            reason = 'no_actions'
            break
        next_obs, rewards, terms, truncs, infos = env.step(actions)
        if pending is not None and collect:
            records.append(pending)
        ended = [a for a, done in terms.items() if done and a in alive]
        if ended:
            health = {a: getattr(infos.get(a, {}).get('player'), 'health', None) for a in ended}
            if any(v is None for v in health.values()):
                reason = 'unknown_terminal_health'
                break
            dead = [a for a in ended if health[a] <= 0]
            survived = [a for a in ended if health[a] > 0]
            if len(survived) > 1 or (survived and len(ended) != len(alive)):
                reason = 'ambiguous_survivors'
                break
            # Upstream emits 40-5*(rank-1) plus accumulated player.reward.
            # Preserve upstream's ordering for simultaneous eliminations.
            for a in ended:
                player = infos[a]['player']
                component = float(rewards[a]) - float(player.reward)
                rank = 1 + (40-component)/5
                rounded = round(rank)
                if abs(rank-rounded) > 1e-6 or not 1 <= rounded <= 8:
                    reason = 'invalid_rank_component'
                    break
                ranks[a] = int(rounded)
            if reason == 'invalid_rank_component':
                break
            alive.difference_update(ended)
        obs = next_obs
    else:
        step = 899
    env.close()
    if reason == 'natural_end' and (len(ranks) != 8 or alive or set(ranks.values()) != set(range(1, 9))):
        reason = 'incomplete_ranking'
    return {'seed': seed, 'reason': reason, 'steps': step+1, 'ranks': ranks,
            'elapsed_seconds': round(time.monotonic()-start, 2)}, records


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(123)
    checkpoint = torch.load(BASE/'tft_pilot_20261121/policy.pt', map_location='cpu', weights_only=False)
    normalizer = (checkpoint['mean'], checkpoint['std'])
    model = Policy(len(normalizer[0]))
    model.load_state_dict(checkpoint['state_dict'])
    val_seeds = [1001, 1002, 1003]
    train_seeds = [201, 202, 203, 204]
    before = []
    for seed in val_seeds:
        meta, _ = rollout(model, normalizer, seed)
        before.append(meta)
        print('before', meta, flush=True)
    episodes = []
    for seed in train_seeds:
        episode = rollout(model, normalizer, seed, collect=True)
        episodes.append(episode)
        print('train', episode[0], flush=True)
    usable = [(meta, rows) for meta, rows in episodes
              if meta['reason'] == 'natural_end' and 'player_0' in meta['ranks'] and rows]
    if len(usable) < 2:
        raise RuntimeError('fewer than two verified complete training games: ' + str([m for m, _ in episodes]))
    returns = np.array([(8 - m['ranks']['player_0'])/7 for m, _ in usable], dtype=np.float32)
    # Leave-one-episode-out baseline avoids using the same episode in its own baseline.
    advantages = (returns - (returns.sum()-returns)/(len(returns)-1)).astype(np.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    losses = []
    for (_, rows), advantage in zip(usable, advantages):
        xs, masks, actions = zip(*rows)
        x = torch.as_tensor(np.stack(xs), device=device)
        m = torch.as_tensor(np.stack(masks), device=device)
        y = torch.as_tensor(actions, device=device)
        logp = nn.functional.log_softmax(model(x, m), dim=1)
        # Sum log probabilities within each game: the score-function estimator.
        losses.append(-float(advantage) * logp.gather(1, y[:, None]).sum())
    loss = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0).cpu())
    optimizer.step()
    model.cpu().eval()
    after = []
    for seed in val_seeds:
        meta, _ = rollout(model, normalizer, seed)
        after.append(meta)
        print('after', meta, flush=True)
    torch.save({'state_dict': model.state_dict(), 'mean': normalizer[0], 'std': normalizer[1],
                'training_seeds': train_seeds, 'opponent': OPPONENT_VERSION}, OUTPUT/'policy_rl.pt')
    data = {'algorithm': 'one_batch_REINFORCE_fixed_scripted_opponents',
            'device_for_update': str(device), 'training_seeds': train_seeds,
            'validation_seeds': val_seeds, 'train_games': [m for m, _ in episodes],
            'train_returns': returns.tolist(), 'advantages': advantages.tolist(),
            'loss': float(loss.detach().cpu()), 'gradient_norm_before_clip': grad_norm,
            'before': before, 'after': after, 'opponent_version': OPPONENT_VERSION,
            'note': 'Tiny pilot; upstream assigns simultaneous deaths by iteration order; not diffusion or evidence of improvement.'}
    (OUTPUT/'metrics.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps(data, indent=2), flush=True)


if __name__ == '__main__':
    main()
