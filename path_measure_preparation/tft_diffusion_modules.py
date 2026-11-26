"""Exploratory single-action categorical diffusion and guidance ablation.

Uses only training games from the previous experiment. This is not a trajectory
diffuser, Schrödinger bridge, or a proof of a martingale or policy improvement.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from agent_measure.conditional_value import crossfit
from agent_measure.discrete_diffusion import CategoricalDenoiser
from gibbs import gibbs_weights, diagnostics
from tft_pilot_train import ACTION_COUNT, Policy
from tft_rl_pilot import BASE, OPPONENT_VERSION, rollout

OUT = BASE/'tft_diffusion_modules_20261121'
SOURCE = BASE/'tft_gibbs_multiseed_20261121/train_trajectories.npz'
SEEDS = [1301, 1302, 1303]
TRAIN_SEED = 47
UPDATES = 300


def load_data():
    file = np.load(SOURCE, allow_pickle=False)
    n = len([x for x in file.files if x.startswith('rank_')])
    episodes = []
    for i in range(n):
        x = file[f'x_{i}'].astype(np.float32)
        mask = np.unpackbits(file[f'mask_{i}'], axis=1, count=ACTION_COUNT).astype(bool)
        actions = file[f'action_{i}'].astype(np.int64)
        rank = int(file[f'rank_{i}'])
        if x.shape[0] != len(actions) or mask.shape != (len(actions), ACTION_COUNT):
            raise ValueError('trajectory shapes disagree')
        if not np.all(mask[np.arange(len(actions)), actions]):
            raise ValueError('recorded illegal action')
        episodes.append((x, mask, actions, rank))
    return episodes


def estimate_value(episodes):
    rng = np.random.default_rng(49)
    xs, actions, returns, ids = [], [], [], []
    for i, (x, mask, a, rank) in enumerate(episodes):
        ix = rng.choice(len(a), min(len(a), 128), replace=False)
        xs.extend(x[ix]); actions.extend(a[ix]); returns.extend([(8-rank)/7]*len(ix)); ids.extend([i]*len(ix))
    result = crossfit(np.asarray(xs), np.asarray(actions), np.asarray(returns), np.asarray(ids))
    return result


def train_model(episodes, beta, device, train_seed=TRAIN_SEED):
    torch.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    model = CategoricalDenoiser(episodes[0][0].shape[1], ACTION_COUNT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    costs = [(rank-1)/7 for _, _, _, rank in episodes]
    probabilities = gibbs_weights(costs, beta)
    last_loss = None
    for step in range(UPDATES):
        ix = rng.integers(0, len(episodes), size=128)
        obs, masks, acts, w = [], [], [], []
        for game in ix:
            x, mask, a, _ = episodes[int(game)]
            j = int(rng.integers(len(a)))
            obs.append(x[j]); masks.append(mask[j]); acts.append(a[j]); w.append(len(episodes)*probabilities[int(game)])
        obs = torch.as_tensor(np.stack(obs), device=device)
        masks = torch.as_tensor(np.stack(masks), device=device)
        acts = torch.as_tensor(acts, device=device)
        weight = torch.as_tensor(w, dtype=torch.float32, device=device)
        loss = (model.loss(obs, acts, masks)*weight).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    return model.cpu().eval(), last_loss, diagnostics(costs, beta)


class DiffusionPolicy(nn.Module):
    """Compatibility adapter for rollout's single-step policy interface."""
    def __init__(self, model, coef=None, sigma=None, eta=0.):
        super().__init__()
        self.model, self.coef, self.sigma, self.eta = model, coef, sigma, eta

    def forward(self, obs, mask):
        q = uncertainty = None
        if self.coef is not None:
            x = obs.detach().cpu().numpy()
            # coef: intercept, 76 scalars, 55 action-family intercepts.
            q_family = np.clip(self.coef[0] + x[:, -82:-6] @ self.coef[1:77, None] + self.coef[77:][None, :], 0, 1)
            q = torch.as_tensor(np.repeat(q_family, 38, axis=1), dtype=obs.dtype, device=obs.device)
            if self.sigma is not None:
                uncertainty = torch.as_tensor(np.repeat(self.sigma, 38), dtype=obs.dtype, device=obs.device)[None, :]
        action = self.model.sample(obs, mask, q_values=q, eta=self.eta, uncertainty=uncertainty)
        # rollout samples a Categorical; put all mass on the diffusion sample.
        logits = torch.full((len(obs), ACTION_COUNT), -1e9, dtype=obs.dtype, device=obs.device)
        return logits.scatter(1, action[:, None], 0.)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    data = load_data()
    cond = estimate_value(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(BASE/'tft_pilot_20261121/policy.pt', map_location='cpu', weights_only=False)
    normalizer = checkpoint['mean'], checkpoint['std']
    bc_state = torch.load(BASE/'tft_gibbs_multiseed_20261121/seed17_beta0.pt', map_location='cpu', weights_only=False)
    bc = Policy(len(normalizer[0])); bc.load_state_dict(bc_state['state_dict']); bc.eval()
    model, loss, diag = train_model(data, 0., device)
    gibbs_model, gibbs_loss, gibbs_diag = train_model(data, 0.5, device)
    torch.save({'model': model.state_dict(), 'obs_dim': len(normalizer[0]), 'steps': 4, 'alpha': .1}, OUT/'diffusion.pt')
    torch.save({'model': gibbs_model.state_dict(), 'obs_dim': len(normalizer[0]), 'steps': 4, 'alpha': .1}, OUT/'diffusion_gibbs.pt')
    np.savez(OUT/'conditional_value.npz', coef=cond['coef'], family_rmse=cond['family_rmse'], family_counts=cond['family_counts'])
    results = {'status':'evaluating', 'device':str(device), 'data_games':len(data),
        'source':str(SOURCE), 'train_seed':TRAIN_SEED, 'updates_each':UPDATES, 'eval_seeds':SEEDS,
        'model':'4-step masked categorical single-action diffusion, alpha=0.1',
        'initial_noise_note':'alpha_bar(T)=1e-4; uniform prior approximates, rather than exactly equals, forward marginal',
        'diffusion_loss_last':loss, 'gibbs_diffusion_loss_last':gibbs_loss,
        'weights': {'uniform':diag, 'gibbs_0.5':gibbs_diag},
        'crossfit_residuals': {'global_game_balanced_mse':cond['global_oof_mse'], 'per_game':cond['per_game'],
                               'sample_count':len(cond['residual']), 'family_counts':cond['family_counts'].tolist()},
        'opponent':OPPONENT_VERSION, 'groups':{},
        'limitations':['estimated residuals are not proven martingale differences',
          'RMSE by action family is heuristic, not conditional variance',
          'vector field is an inference-time policy tilt, not an SDE drift or SB',
          'Q fit from ten training games only; no hidden info intentionally used; simulator observation visibility not fully audited',
          'single training seed and three evaluation seeds; no statistical validity claim',
          'same environment seed does not ensure identical randomness across differing actions',
          'upstream resolves simultaneous deaths by iteration order; seat bias not audited']}
    groups = [('bc', bc), ('diffusion', DiffusionPolicy(model)),
              ('diffusion_gibbs', DiffusionPolicy(gibbs_model)),
              ('diffusion_vector', DiffusionPolicy(model, cond['coef'], eta=1.)),
              ('diffusion_vector_residual', DiffusionPolicy(model, cond['coef'], cond['family_rmse'], eta=1.))]
    def save():
        tmp = OUT/'metrics.tmp'; tmp.write_text(json.dumps(results, indent=2), encoding='utf8'); tmp.replace(OUT/'metrics.json')
    save()
    for label, policy in groups:
        records = []
        results['groups'][label] = {'games':records}
        save()
        for seed in SEEDS:
            meta, _ = rollout(policy, normalizer, seed)
            records.append(meta)
            print('evaluate', label, meta, flush=True)
            save()
        valid = [m['ranks']['player_0'] for m in records if m['reason']=='natural_end' and 'player_0' in m['ranks']]
        results['groups'][label].update({'valid_games':len(valid), 'attempted_games':len(SEEDS),
            'mean_rank':float(np.mean(valid)) if valid else None,
            'top4_rate':float(np.mean(np.asarray(valid)<=4)) if valid else None,
            'first_rate':float(np.mean(np.asarray(valid)==1)) if valid else None})
        save()
    results['status'] = 'completed'
    save()
    print('COMPLETED', flush=True)


if __name__ == '__main__': main()
