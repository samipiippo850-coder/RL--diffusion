"""Auditable legal-mask heuristic opponents and a seat-balanced strength probe."""
import json
import random
import time
from pathlib import Path
import numpy as np
from Simulator import TFTConfig, parallel_env
from tft_pilot_train import teacher


def econ_policy(mask, info, rng, observation):
    """Simple economy/level/board heuristic. Strength must be measured."""
    legal = np.flatnonzero(mask)
    if not len(legal):
        raise ValueError('no legal action')
    row, col = legal // 38, legal % 38
    player = info.get('player')
    gold = getattr(player, 'gold', 0)
    level = getattr(player, 'level', 1)
    game_round = info.get('game_round', 1)
    count = info.get('actions_taken', 0) or 0
    units = getattr(player, 'num_units_in_play', 0)
    cap = getattr(player, 'max_units', level)
    board = np.asarray(observation['observations']['board'])[0]
    empty_board = np.all(board == 0, axis=-1)
    # Properties read are this player's own gold, level and current round only.
    # Each decision is checked against the live mask.
    is_board_target = col < 28
    empty_target = np.zeros(len(legal), dtype=bool)
    empty_target[is_board_target] = empty_board[col[is_board_target]]
    deploy = legal[(row >= 28) & (row <= 36) & is_board_target & empty_target] if units < cap else []
    candidates = [deploy]
    for possible in candidates:
        if len(possible):
            return int(rng.choice(possible))
    # Equip ordinary items on the board; avoid consuming special utility items
    # blindly. The simulator's live mask determines which placements are legal.
    item_actions = legal[(row >= 37) & (row <= 46) & (col < 28)]
    item_bench = getattr(player, 'item_bench', [])
    ordinary = [a for a in item_actions if int(a // 38 - 37) < len(item_bench)
                and item_bench[int(a // 38 - 37)] not in
                ('champion_duplicator', 'magnetic_remover', 'reforger',
                 'kayn_shadowassassin', 'kayn_rhast')]
    if ordinary:
        return int(rng.choice(ordinary))
    level_actions = legal[row == 53]
    if len(level_actions) and gold >= 8 and level < min(8, 3 + game_round//4):
        return int(rng.choice(level_actions))
    buy = legal[(row >= 47) & (row <= 51)]
    bench = np.asarray(observation['observations']['bench'])[0]
    has_empty_bench = bool(np.any(np.all(bench == 0, axis=-1)))
    if len(buy) and has_empty_bench and gold >= (1 if game_round < 5 else 3):
        return int(rng.choice(buy))
    refresh = legal[row == 54]
    if len(refresh) and gold >= 8 and count < 10:
        return int(rng.choice(refresh))
    passes = legal[row == 52]
    return int(rng.choice(passes if len(passes) else legal))


def rank_events(ended, alive, infos, rewards):
    health = {a: getattr(infos.get(a, {}).get('player'), 'health', None) for a in ended}
    if any(v is None for v in health.values()):
        return None
    survivors = [a for a in ended if health[a] > 0]
    if len(survivors) > 1 or (survivors and len(ended) != len(alive)):
        return None
    ranks = {}
    for a in ended:
        raw = 1 + (40 - (float(rewards[a])-float(infos[a]['player'].reward)))/5
        rank = round(raw)
        if abs(raw-rank) > 1e-6 or not 1 <= rank <= 8:
            return None
        ranks[a] = int(rank)
    return ranks


def strength_lobby(seed, offset):
    random.seed(seed); np.random.seed(seed)
    env = parallel_env(TFTConfig(num_players=8))
    obs, infos = env.reset(seed=seed)
    rngs = {a: np.random.default_rng(seed*100 + i) for i, a in enumerate(env.possible_agents)}
    # Four seats per policy; offset alternates which seat parity receives each.
    kind = {f'player_{i}': ('econ' if (i+offset)%2 == 0 else 'teacher') for i in range(8)}
    alive, ranks = set(env.agents), {}
    events = {k: {} for k in ('econ', 'teacher')}
    start = time.monotonic()
    reason = 'step_limit'
    for step in range(900):
        if time.monotonic()-start > 120:
            reason = 'time_limit'; break
        if not env.agents:
            reason = 'natural_end'; break
        actions = {}
        for a in env.agents:
            mask = np.asarray(obs[a]['action_mask'], dtype=bool).reshape(-1)
            act = (econ_policy(mask, infos.get(a, {}), rngs[a], obs[a]) if kind[a]=='econ'
                   else teacher(mask, rngs[a]))
            if not mask[act]: raise AssertionError('illegal action')
            actions[a] = act
            row = str(act//38)
            events[kind[a]][row] = events[kind[a]].get(row, 0) + 1
        obs, rewards, terms, truncs, infos = env.step(actions)
        ended = [a for a, done in terms.items() if done and a in alive]
        if ended:
            result = rank_events(ended, alive, infos, rewards)
            if result is None:
                reason = 'ambiguous_terminal'; break
            ranks.update(result); alive.difference_update(ended)
    env.close()
    if reason == 'natural_end' and (alive or set(ranks.values()) != set(range(1,9))):
        reason = 'invalid_ranking'
    return {'seed':seed, 'seat_offset':offset, 'reason':reason, 'ranks':ranks, 'kinds':kind,
            'action_rows':events, 'elapsed_seconds':round(time.monotonic()-start, 2)}


def main():
    out = Path('/root/autodl-tmp/path_measure_preparation/tft_opponent_probe_20261121')
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i, seed in enumerate(range(2101, 2113)):
        game = strength_lobby(seed, i%2)
        records.append(game)
        print('game', game, flush=True)
        (out/'metrics.json').write_text(json.dumps({'games':records}, indent=2), encoding='utf8')
    print('COMPLETED', flush=True)


if __name__ == '__main__': main()
