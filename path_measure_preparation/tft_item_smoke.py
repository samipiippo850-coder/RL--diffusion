"""Eight isolated full games to assess simulator warnings after item changes."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tft_intense_game import BASE, game

out = BASE / 'tft_item_smoke_20261122' / 'metrics.json'

def main():
    records = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(game, {'seed':seed,'seat':seed%8,'kind':'econ'}):seed
                   for seed in range(7101,7109)}
        for future in as_completed(futures):
            try:
                meta, _ = future.result()
            except Exception as exc:
                meta = {'seed':futures[future], 'reason':'worker_error', 'error':repr(exc)}
            records.append(meta)
            out.write_text(json.dumps({'games':records},indent=2), encoding='utf8')
            print(meta, flush=True)
    print('COMPLETED',flush=True)

if __name__ == '__main__': main()
