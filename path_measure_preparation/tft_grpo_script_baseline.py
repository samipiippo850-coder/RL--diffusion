"""Economic-script reference on the same held-out GRPO test seeds."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tft_intense_game import BASE, game

out=BASE/'tft_grpo_script_baseline_20261122'/'metrics.json'

def main():
    jobs=[{'seed':s,'seat':s%8,'kind':'econ'} for s in range(22001,22017)]
    records=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(game,j):j for j in jobs}
        for f in as_completed(futures):
            try: meta,_=f.result()
            except Exception as exc: meta={**futures[f],'reason':'worker_error','error':repr(exc)}
            records.append(meta)
            out.write_text(json.dumps({'status':'running','games':records},indent=2),encoding='utf8')
            print(meta,flush=True)
    out.write_text(json.dumps({'status':'completed','games':records},indent=2),encoding='utf8')
    print('COMPLETED',flush=True)

if __name__=='__main__': main()
