"""Bounded masked behavior-cloning pilot on real TFT observations.

This is not RL, diffusion, or a ranking evaluation. Episodes are deliberately
truncated and are used only for supervised interface validation.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from Simulator import TFTConfig, parallel_env

FIELDS = ('board', 'traits', 'bench', 'items', 'shop', 'scalars', 'emb_scalars')
ACTION_COUNT = 55 * 38


def features(obs):
    # Only fields returned by the simulator's observation interface, no info.player.
    x = obs['observations']
    result = np.concatenate([np.asarray(x[k], dtype=np.float32).reshape(-1) for k in FIELDS])
    if not np.isfinite(result).all():
        raise ValueError('non-finite observation')
    return result


def teacher(mask, rng):
    """Simple behavior policy, not a validated strong strategy."""
    legal = np.flatnonzero(mask)
    if not len(legal):
        raise ValueError('no legal action')
    row, col = legal // 38, legal % 38
    # Play units from bench, buy, then move items. Exploration includes pass.
    choices = [legal[(row >= 28) & (row <= 36) & (col < 28)],
               legal[(row >= 47) & (row <= 51)],
               legal[(row >= 37) & (row <= 46) & (col < 28)]]
    if rng.random() < 0.2:
        return int(rng.choice(legal))
    for candidates in choices:
        if len(candidates):
            return int(rng.choice(candidates))
    passes = legal[row == 52]
    return int(rng.choice(passes if len(passes) else legal))


def collect(seeds, max_steps, max_seconds):
    rows, metadata = [], []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed + 100000)
        env = parallel_env(TFTConfig(num_players=8))
        start = time.monotonic()
        obs, infos = env.reset(seed=seed)
        count = 0
        reason = 'step_limit'
        for step in range(max_steps):
            if time.monotonic() - start > max_seconds:
                reason = 'time_limit'
                break
            actions = {}
            pending = []
            for agent in env.agents:
                if agent not in obs or obs[agent] is None:
                    continue
                mask = np.asarray(obs[agent]['action_mask'], dtype=np.bool_).reshape(-1).copy()
                if len(mask) != ACTION_COUNT:
                    raise ValueError('unexpected action mask shape')
                action = teacher(mask, rng)
                if not mask[action]:
                    raise AssertionError('teacher selected illegal action')
                actions[agent] = action
                pending.append((features(obs[agent]).copy(), mask, action))
            if not actions:
                reason = 'no_agents'
                break
            next_obs, rewards, terminated, truncated, infos = env.step(actions)
            rows.extend(pending)  # store only after successful environment step
            count += 1
            obs = next_obs
            if not env.agents:
                reason = 'no_agents'
                break
        env.close()
        metadata.append({'seed': seed, 'steps': count, 'stop_reason': reason,
                         'elapsed_seconds': round(time.monotonic()-start, 3)})
        print('collected', metadata[-1], flush=True)
    if not rows:
        raise RuntimeError('no data collected')
    x, masks, actions = zip(*rows)
    return np.stack(x), np.stack(masks), np.asarray(actions, dtype=np.int64), metadata


class Policy(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, ACTION_COUNT))

    def forward(self, x, mask):
        return self.net(x).masked_fill(~mask, -1e9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='tft_pilot_output')
    ap.add_argument('--max-steps', type=int, default=24)
    ap.add_argument('--max-seconds', type=float, default=60)
    ap.add_argument('--updates', type=int, default=300)
    args = ap.parse_args()
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    torch.set_num_threads(4)
    train_x, train_m, train_y, train_meta = collect([11, 22, 33, 44], args.max_steps, args.max_seconds)
    val_x, val_m, val_y, val_meta = collect([101, 102], args.max_steps, args.max_seconds)
    # Normalize with train-only moments. Clip extreme standardized values.
    mean = train_x.mean(0)
    std = train_x.std(0)
    std[std < 1e-5] = 1
    tx = np.clip((train_x-mean)/std, -10, 10)
    vx = np.clip((val_x-mean)/std, -10, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Policy(tx.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    x = torch.as_tensor(tx, device=device)
    masks = torch.as_tensor(train_m, device=device)
    y = torch.as_tensor(train_y, device=device)
    for step in range(args.updates):
        index = torch.randint(len(y), (min(128, len(y)),), device=device)
        loss = nn.functional.cross_entropy(model(x[index], masks[index]), y[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        val_logits = model(torch.as_tensor(vx, device=device), torch.as_tensor(val_m, device=device))
        prediction = val_logits.argmax(1).cpu().numpy()
        accuracy = float(np.mean(prediction == val_y))
        illegal = float(np.mean(~val_m[np.arange(len(val_y)), prediction]))
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'mean': mean, 'std': std,
                'fields': FIELDS, 'action_count': ACTION_COUNT}, path/'policy.pt')
    results = {'type': 'bounded_supervised_pilot_not_rl_or_diffusion', 'device': str(device),
               'torch_version': torch.__version__, 'train_seed': 123,
               'train_environment_seeds': [11,22,33,44], 'validation_environment_seeds': [101,102],
               'train_samples': len(train_y), 'validation_samples': len(val_y),
               'updates': args.updates, 'final_minibatch_loss': float(loss.detach().cpu()),
               'validation_action_accuracy': accuracy, 'validation_illegal_rate': illegal,
               'train_episodes': train_meta, 'validation_episodes': val_meta}
    (path/'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
