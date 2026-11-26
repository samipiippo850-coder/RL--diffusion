"""Count attempted legal action families on matched full-game seeds."""
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,as_completed
from tft_intense_game import BASE,game

OUT=BASE/'tft_action_diagnostic_20261124'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={'frozen':BASE/'tft_v2_suite_20261122/diffusion_seed247_beta0.pt',
           'preference_grpo':BASE/'tft_grpo_scale_20261124/preference_grpo_init247.pt'}
    jobs=[]
    for s in range(34001,34009):
        jobs.append({'seed':s,'seat':s%8,'kind':'econ','arm':'econ'})
        jobs.extend({'seed':s,'seat':s%8,'kind':'diffusion_grpo','arm':arm,'model_path':str(path)}
                    for arm,path in paths.items())
    records=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(game,j):j for j in jobs}
        for f in as_completed(futures):
            j=futures[f]
            try: meta,_=f.result()
            except Exception as exc:meta={**j,'reason':'worker_error','error':repr(exc)}
            records.append({'arm':j['arm'],**meta})
            tmp=OUT/'metrics.tmp';tmp.write_text(json.dumps({'status':'running','games':records},indent=2),encoding='utf8');tmp.replace(OUT/'metrics.json')
            print(records[-1],flush=True)
    (OUT/'metrics.json').write_text(json.dumps({'status':'completed','games':records},indent=2),encoding='utf8')
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
