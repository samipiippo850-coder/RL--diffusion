"""Isolated seeded game worker for stronger-opponent experiments."""
import random
import time
import contextlib
import io
from pathlib import Path
import numpy as np
import torch
from Simulator import TFTConfig, parallel_env
from agent_measure.discrete_diffusion import CategoricalDenoiser, reverse_posterior
from agent_measure.hierarchical_policy import HierarchicalPolicy, HierarchicalDenoiser
from tft_diffusion_modules import DiffusionPolicy
from tft_opponents import econ_policy, rank_events
from tft_pilot_train import ACTION_COUNT, features, teacher

BASE = Path('/root/autodl-tmp/path_measure_preparation')


def game(task):
    """Each task is one process-local game; returns pre-action samples if asked."""
    seed = int(task['seed'])
    seat = int(task.get('seat', seed % 8))
    target = f'player_{seat}'
    kind = task['kind']
    collect = bool(task.get('collect', False))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(int(task.get('sample_seed', seed+400000)))
    torch.set_num_threads(1)
    ckpt = torch.load(BASE/'tft_pilot_20261121/policy.pt', map_location='cpu', weights_only=False)
    mean, std = ckpt['mean'], ckpt['std']
    policy = None
    hierarchical = None
    if kind in ('hier_bc', 'hier_diffusion'):
        state = torch.load(Path(task['model_path']), map_location='cpu', weights_only=False)
        mean, std = state['mean'], state['std']
        hierarchical = (HierarchicalPolicy(len(mean)) if kind == 'hier_bc' else HierarchicalDenoiser(len(mean)))
        hierarchical.load_state_dict(state['model']); hierarchical.eval()
    if kind in ('diffusion', 'diffusion_gibbs', 'diffusion_vector', 'diffusion_vector_residual', 'diffusion_grpo'):
        model_path = Path(task['model_path'])
        state = torch.load(model_path, map_location='cpu', weights_only=False)
        model = CategoricalDenoiser(len(mean), ACTION_COUNT)
        model.load_state_dict(state['model']); model.eval()
        value = None
        if 'vector' in kind:
            with np.load(task['value_path'], allow_pickle=False) as f:
                value = (f['coef'].copy(), f['family_rmse'].copy())
        policy = DiffusionPolicy(model, None if value is None else value[0],
                                 value[1] if kind == 'diffusion_vector_residual' else None,
                                 1.0 if value is not None else 0.)
        policy.eval()
    env = parallel_env(TFTConfig(num_players=8))
    obs, infos = env.reset(seed=seed)
    rngs = {a: np.random.default_rng(seed*100 + i + 8192) for i, a in enumerate(env.possible_agents)}
    expert_rng = np.random.default_rng(seed*100 + seat + 271828)
    alive = set(env.agents); ranks = {}; records = []; warning_count = 0; warning_examples = []
    max_pending_items = 0
    target_action_counts = {}
    dagger_count = dagger_agree = 0
    start = time.monotonic(); reason = 'step_limit'; steps = 0
    try:
        for step in range(900):
            if time.monotonic()-start > 180:
                reason = 'time_limit'; break
            if not env.agents:
                reason = 'natural_end'; break
            actions = {}; pending = None
            for a in env.agents:
                if a not in obs or obs[a] is None:
                    continue
                mask = np.asarray(obs[a]['action_mask'], dtype=bool).reshape(-1).copy()
                if len(mask) != ACTION_COUNT or not mask.any():
                    raise ValueError('invalid legal mask')
                if a != target:
                    action = econ_policy(mask, infos.get(a, {}), rngs[a], obs[a])
                elif hierarchical is not None:
                    x = np.clip((features(obs[a])-mean)/std, -10, 10).astype(np.float32)
                    with torch.no_grad():
                        xx=torch.from_numpy(x).unsqueeze(0); mm=torch.from_numpy(mask).unsqueeze(0)
                        if kind == 'hier_diffusion':
                            action=int(hierarchical.sample(xx,mm).item())
                        else:
                            action=int(torch.distributions.Categorical(logits=hierarchical(xx,mm)).sample().item())
                elif kind == 'diffusion_grpo':
                    x = np.clip((features(obs[a])-mean)/std, -10, 10).astype(np.float32)
                    with torch.no_grad():
                        xx = torch.from_numpy(x).unsqueeze(0)
                        mm = torch.from_numpy(mask).unsqueeze(0)
                        xt = torch.multinomial(mm.float() / mm.sum(-1, keepdim=True), 1).squeeze(1)
                        chain = [int(xt.item())]
                        for t in range(model.steps, 0, -1):
                            ts = torch.full((1,), t, dtype=torch.long)
                            p0 = torch.softmax(model(xx, xt, ts, mm), dim=-1)
                            posterior = reverse_posterior(p0, xt, mm, model.alpha, model.alpha**(t-1))
                            xt = torch.multinomial(posterior, 1).squeeze(1)
                            chain.append(int(xt.item()))
                        action = int(xt.item())
                elif policy is not None:
                    x = np.clip((features(obs[a])-mean)/std, -10, 10).astype(np.float32)
                    with torch.no_grad():
                        logits = policy(torch.from_numpy(x).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0))
                        action = int(torch.distributions.Categorical(logits=logits).sample().item())
                elif kind == 'econ':
                    action = econ_policy(mask, infos.get(a, {}), rngs[a], obs[a])
                elif kind == 'teacher':
                    action = teacher(mask, rngs[a])
                else:
                    raise ValueError('unknown target policy ' + kind)
                if not mask[action]: raise AssertionError('illegal action submitted')
                actions[a] = action
                if a == target:
                    row, col = divmod(action, 38)
                    family = ('sell' if col == 37 and row <= 36 else
                              'move' if row <= 36 else 'item' if row <= 46 else
                              'buy' if row <= 51 else 'pass' if row == 52 else
                              'level' if row == 53 else 'refresh')
                    target_action_counts[family] = target_action_counts.get(family, 0) + 1
                if a == target and collect:
                    x = features(obs[a]).astype(np.float32) if task.get('raw_collect') else np.clip((features(obs[a])-mean)/std, -10, 10).astype(np.float32)
                    if task.get('dagger_collect'):
                        label = econ_policy(mask, infos.get(a, {}), expert_rng, obs[a])
                        if not mask[label]: raise AssertionError('illegal expert label')
                        dagger_count += 1; dagger_agree += int(label == action)
                        pending = (x.copy(), mask, label, action)
                    else:
                        pending = (x.copy(), mask, action, np.asarray(chain, dtype=np.int16)) if kind == 'diffusion_grpo' else (x.copy(), mask, action)
            if not actions:
                reason = 'no_actions'; break
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                new_obs, rewards, terms, truncs, infos = env.step(actions)
            max_pending_items = max(max_pending_items, *(len(getattr(info.get('player'), 'pending_items', []))
                                    for info in infos.values()), 0)
            for line in stream.getvalue().splitlines():
                if any(word in line.lower() for word in ('failed', 'could not', 'did not', 'should never')):
                    warning_count += 1
                    if len(warning_examples) < 4:
                        warning_examples.append(line[:180])
            steps += 1
            if pending is not None:
                records.append(pending)
            ended = [a for a, done in terms.items() if done and a in alive]
            if ended:
                result = rank_events(ended, alive, infos, rewards)
                if result is None:
                    reason = 'ambiguous_terminal'; break
                ranks.update(result); alive.difference_update(ended)
            obs = new_obs
        if reason == 'natural_end' and (alive or set(ranks.values()) != set(range(1,9))):
            reason = 'invalid_ranking'
    finally:
        env.close()
    meta = {'seed':seed, 'seat':seat, 'kind':kind, 'reason':reason, 'steps':steps,
            'rank':ranks.get(target), 'ranks':ranks, 'internal_warning_count':warning_count,
            'internal_warning_examples':warning_examples, 'max_pending_items_per_player':max_pending_items,
            'target_action_counts':target_action_counts,
            'dagger_count':dagger_count, 'dagger_exact_agree':dagger_agree,
            'elapsed_seconds':round(time.monotonic()-start, 2)}
    return meta, records
