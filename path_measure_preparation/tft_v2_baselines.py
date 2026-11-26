"""Scripted reference policies in exactly the same V2 evaluation seed sets."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tft_intense_game import BASE, game

OUT = BASE/'tft_v2_baselines_20261122'/'metrics.json'

def main():
    jobs = [{'seed':seed,'seat':seed%8,'kind':kind,'split':split}
            for split,start in [('validation',17001),('test',18001)]
            for kind in ('teacher','econ') for seed in range(start,start+16)]
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(game,job):job for job in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:
                meta,_=future.result()
            except Exception as exc:
                meta={**job,'reason':'worker_error','error':repr(exc)}
            records.append({'split':job['split'],**meta})
            OUT.write_text(json.dumps({'status':'running','games':records},indent=2),encoding='utf8')
            print(records[-1],flush=True)
    OUT.write_text(json.dumps({'status':'completed','games':records},indent=2),encoding='utf8')
    print('COMPLETED',flush=True)

if __name__ == '__main__': main()
