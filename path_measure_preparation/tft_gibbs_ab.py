"""Small predeclared Gibbs-weighted trajectory imitation ablation.

The reference and Gibbs models share initial weights, 8 behavior-policy games,
sampled minibatches, architecture, optimizer, and update budget. This is not
diffusion, PPO, or a powered significance study.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from gibbs import gibbs_weights, diagnostics
from tft_pilot_train import Policy
from tft_rl_pilot import rollout, BASE, OPPONENT_VERSION

OUT = BASE / 'tft_gibbs_ab_20261121'
TRAIN_SEEDS = list(range(301, 309))
EVAL_SEEDS = list(range(1101, 1107))
BETA = 2.0
UPDATES = 100


def build_model(checkpoint):
    model = Policy(len(checkpoint['mean']))
    model.load_state_dict(checkpoint['state_dict'])
    return model


def fit(checkpoint, episodes, weights, device, sampler_seed=2718):
    torch.manual_seed(sampler_seed)
    model = build_model(checkpoint).to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-5)
    rng = np.random.default_rng(sampler_seed)
    final_loss = None
    for _ in range(UPDATES):
        # Uniform over complete games, then uniform over decisions within game.
        # Each episode's weight is N*p_i; no weighted replay is used.
        episode_ids = rng.integers(0, len(episodes), size=4)
        xs, masks, actions, sample_weights = [], [], [], []
        for i in episode_ids:
            rows = episodes[int(i)][1]
            for t in rng.integers(0, len(rows), size=32):
                x, mask, action = rows[int(t)]
                xs.append(x); masks.append(mask); actions.append(action)
                sample_weights.append(len(episodes) * weights[int(i)])
        x = torch.as_tensor(np.stack(xs), device=device)
        mask = torch.as_tensor(np.stack(masks), device=device)
        y = torch.as_tensor(actions, device=device)
        w = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)
        loss = (nn.functional.cross_entropy(model(x, mask), y, reduction='none') * w).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        final_loss = float(loss.detach().cpu())
    return model.cpu().eval(), final_loss


def evaluate(model, normalizer, label):
    output = []
    for seed in EVAL_SEEDS:
        meta, _ = rollout(model, normalizer, seed)
        output.append(meta)
        print(label, meta, flush=True)
    valid = [x['ranks']['player_0'] for x in output
             if x['reason'] == 'natural_end' and 'player_0' in x['ranks']]
    return {'games': output, 'valid_games': len(valid), 'attempted_games': len(output),
            'mean_rank': float(np.mean(valid)) if valid else None,
            'top4_rate_midrank': float(np.mean(np.asarray(valid) <= 4)) if valid else None,
            'first_rate_midrank': float(np.mean(np.asarray(valid) == 1)) if valid else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(BASE/'tft_pilot_20261121/policy.pt', map_location='cpu', weights_only=False)
    reference = build_model(checkpoint).eval()
    norm = checkpoint['mean'], checkpoint['std']
    episodes, all_meta = [], []
    for seed in TRAIN_SEEDS:
        meta, rows = rollout(reference, norm, seed, collect=True)
        all_meta.append(meta)
        print('collect', meta, flush=True)
        if meta['reason'] == 'natural_end' and 'player_0' in meta['ranks'] and rows:
            episodes.append((meta, rows))
    if len(episodes) < 6:
        raise RuntimeError('fewer than six naturally completed training games')
    costs = [(m['ranks']['player_0']-1)/7 for m, _ in episodes]
    gibbs = gibbs_weights(costs, BETA)
    uniform = [1/len(episodes)]*len(episodes)
    weight_stats = diagnostics(costs, BETA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline, a_loss = fit(checkpoint, episodes, uniform, device)
    treatment, b_loss = fit(checkpoint, episodes, gibbs, device)
    for label, model in [('uniform', baseline), ('gibbs', treatment)]:
        torch.save({'state_dict': model.state_dict(), 'mean': norm[0], 'std': norm[1]}, OUT/f'{label}.pt')
    print('weight_diagnostics', weight_stats, flush=True)
    a = evaluate(baseline, norm, 'uniform')
    b = evaluate(treatment, norm, 'gibbs')
    result = {'kind': 'small_offline_trajectory_weight_ab_not_diffusion', 'device': str(device),
              'train_seeds_attempted': TRAIN_SEEDS, 'train_metadata': all_meta,
              'train_games_used': len(episodes), 'eval_seeds': EVAL_SEEDS, 'beta': BETA,
              'cost': '(midrank-1)/7', 'weight_diagnostics': weight_stats,
              'normalized_weights': gibbs, 'updates_each': UPDATES, 'last_minibatch_loss': {'uniform': a_loss, 'gibbs': b_loss},
              'uniform': a, 'gibbs': b, 'opponent': OPPONENT_VERSION,
              'tie_rule': 'same-step deaths midpoint; first/top4 use midpoint threshold; unverified provisional',
              'notes': ['same seed does not guarantee same future random events after actions diverge',
                        'too few games and one training seed; no significance or stable gain claim',
                        'models do not include diffusion, martingale or vector-field components']}
    (OUT/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
