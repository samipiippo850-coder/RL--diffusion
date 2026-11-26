"""Exploratory multi-seed Gibbs ablation using validated upstream rank events."""
import json
from pathlib import Path
import numpy as np
import torch
from gibbs import gibbs_weights, diagnostics
from tft_gibbs_ab import build_model, fit
from tft_rl_pilot import BASE, OPPONENT_VERSION, rollout

OUT = BASE / 'tft_gibbs_multiseed_20261121'
DATA_SEEDS = list(range(401, 411))
EVAL_SEEDS = list(range(1201, 1205))
TRAIN_SEEDS = [17, 23, 37]
BETAS = [0.0, 0.5, 2.0]


def persist(data):
    target = OUT/'metrics.json'
    tmp = OUT/'metrics.tmp'
    tmp.write_text(json.dumps(data, indent=2), encoding='utf8')
    tmp.replace(target)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    checkpoint = torch.load(BASE/'tft_pilot_20261121/policy.pt', map_location='cpu', weights_only=False)
    behavior = build_model(checkpoint).eval()
    normalizer = checkpoint['mean'], checkpoint['std']
    data = {'kind': 'exploratory_multiseed_gibbs_imitation_not_diffusion',
            'rank_rule': 'upstream terminal reward minus accumulated player.reward; sequential tie ordering',
            'train_data_seeds': DATA_SEEDS, 'fine_tuning_seeds': TRAIN_SEEDS,
            'eval_seeds': EVAL_SEEDS, 'beta_values': BETAS, 'updates_each': 100,
            'shared_initial_model': 'tft_pilot_20261121/policy.pt', 'opponent': OPPONENT_VERSION,
            'notes': ['fine-tuning seeds share one behavior dataset and one initial checkpoint',
                      'same seed does not guarantee identical random events after policy divergence',
                      'small exploratory sample; no claims of statistical significance',
                      'ranking depends on upstream internal ordering for simultaneous deaths'],
            'collection': [], 'models': [], 'status': 'collecting'}
    persist(data)
    episodes = []
    for seed in DATA_SEEDS:
        meta, rows = rollout(behavior, normalizer, seed, collect=True)
        data['collection'].append(meta)
        if meta['reason'] == 'natural_end' and 'player_0' in meta['ranks'] and rows:
            episodes.append((meta, rows))
        print('collect', meta, flush=True)
        persist(data)
    if len(episodes) < 8:
        raise RuntimeError('fewer than eight verified complete games')
    costs = [(m['ranks']['player_0'] - 1)/7 for m, _ in episodes]
    data['train_games_used'] = len(episodes)
    data['train_costs'] = costs
    data['weight_diagnostics'] = {str(beta): diagnostics(costs, beta) for beta in BETAS}
    # Save pre-action observations, masks and actions per game for reproducibility.
    arrays = {}
    for i, (meta, rows) in enumerate(episodes):
        xs, masks, actions = zip(*rows)
        arrays[f'x_{i}'] = np.stack(xs).astype(np.float32)
        arrays[f'mask_{i}'] = np.packbits(np.stack(masks), axis=1)
        arrays[f'action_{i}'] = np.asarray(actions, dtype=np.int16)
        arrays[f'seed_{i}'] = np.asarray(meta['seed'], dtype=np.int64)
        arrays[f'rank_{i}'] = np.asarray(meta['ranks']['player_0'], dtype=np.int8)
    np.savez_compressed(OUT/'train_trajectories.npz', **arrays)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data['device'] = str(device)
    data['status'] = 'training_and_evaluation'
    persist(data)
    for seed in TRAIN_SEEDS:
        for beta in BETAS:
            model, loss = fit(checkpoint, episodes, gibbs_weights(costs, beta), device, sampler_seed=seed)
            label = f'seed{seed}_beta{beta:g}'
            torch.save({'state_dict': model.state_dict(), 'mean': normalizer[0], 'std': normalizer[1],
                        'fine_tuning_seed': seed, 'beta': beta}, OUT/f'{label}.pt')
            result = {'seed': seed, 'beta': beta, 'loss_last_minibatch': loss, 'eval_games': []}
            data['models'].append(result)
            persist(data)
            print('model', label, 'loss', loss, flush=True)
            for eval_seed in EVAL_SEEDS:
                meta, _ = rollout(model, normalizer, eval_seed)
                result['eval_games'].append(meta)
                print('evaluate', label, meta, flush=True)
                persist(data)
            valid = [x['ranks']['player_0'] for x in result['eval_games']
                     if x['reason'] == 'natural_end' and 'player_0' in x['ranks']]
            result.update({'valid_games': len(valid), 'attempted_games': len(EVAL_SEEDS),
                           'mean_rank': float(np.mean(valid)) if valid else None,
                           'top4_rate': float(np.mean(np.asarray(valid) <= 4)) if valid else None,
                           'first_rate': float(np.mean(np.asarray(valid) == 1)) if valid else None})
            persist(data)
    data['status'] = 'completed'
    persist(data)
    print('COMPLETED', flush=True)


if __name__ == '__main__':
    main()
