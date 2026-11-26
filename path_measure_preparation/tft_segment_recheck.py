"""Fresh-randomness recheck for two preselected exploratory snapshots.

State fingerprints must match prior result before interventions are run.
"""
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from tft_segment_pilot import BASE,seek

OUT=BASE/'tft_segment_recheck_20261125'
SOURCE=BASE/'tft_segment_pilot_20261124/metrics.json'

def fingerprint(state):
    obs=np.asarray(state['observation'],dtype=np.float32)
    mask=np.zeros(55*38,dtype=np.uint8);mask[np.asarray(state['legal_actions'],dtype=np.int64)]=1
    return hashlib.sha256(obs.tobytes()+mask.tobytes()).hexdigest()

def save(result):
    tmp=OUT/'metrics.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/'metrics.json')

def main():
    OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    prior=json.loads(SOURCE.read_text(encoding='utf8'))
    states={x['seed']:x for x in prior['tasks'] if x.get('case')=='level_refresh' and x.get('reason')=='completed'}
    # Independent continuation seeds start at index 4; first pilot used 0..3.
    jobs=[{'seed':seed,'case':'level_refresh','replicate_start':4+8*i,'replicates':8,
           'expected_fingerprint':fingerprint(states[seed])}
          for seed,count in ((62106,4),(62103,2)) for i in range(count)]
    result={'status':'running','jobs':[],'preselected':[62106,62103],
            'notes':['postselected illustrative states; no population-level estimate',
                     'continuation seeds 4 onward disjoint from pilot seeds 0..3',
                     'identical observed features and legal mask do not prove equality of hidden state',
                     'one forced action then all players follow the same scripted policy',
                     'paired seeds do not synchronize future draws after actions diverge']};save(result)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(seek,j):j for j in jobs}
        for f in as_completed(futures):
            job=futures[f]
            try:out=f.result()
            except Exception as exc:out={'seed':job['seed'],'reason':'worker_error','error':repr(exc)}
            result['jobs'].append({'replicate_start':job['replicate_start'],**out});save(result)
            print('job',{'seed':job['seed'],'replicate_start':job['replicate_start'],'reason':out['reason'],'error':out.get('error')},flush=True)
    result['status']='completed';result['elapsed_seconds']=round(time.monotonic()-started,2);save(result)
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
