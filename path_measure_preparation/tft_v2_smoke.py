"""Full-game smoke check in the second simulator repair copy."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tft_intense_game import BASE, game

out = BASE / 'tft_v2_smoke_20261122' / 'metrics.json'

def main():
    jobs = []
    for seed in (11004, 12014, 14001, 14002, 14003, 14004, 14005, 14006):
        for kind in ('econ', 'diffusion'):
            jobs.append({'seed':seed, 'seat':seed % 8, 'kind':kind,
                         'model_path':str(BASE/'tft_item_suite_20261122/diffusion_seed147_beta0.pt')})
    records=[]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as pool:
        futures = {pool.submit(game, job):job for job in jobs}
        for future in as_completed(futures):
            try:
                meta, _ = future.result()
            except Exception as exc:
                meta = {**futures[future], 'reason':'worker_error', 'error':repr(exc)}
            records.append(meta)
            out.write_text(json.dumps({'games':records},indent=2),encoding='utf8')
            print(meta,flush=True)
    print('COMPLETED',flush=True)

if __name__ == '__main__': main()
