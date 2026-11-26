"""Larger fixed-opponent exploratory categorical diffusion ablation.

All game seeds, seat rotation and model groups are predeclared. Single-action
diffusion only; no claim of trajectory diffusion or proven martingale property.
"""
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from agent_measure.conditional_value import crossfit, design
from gibbs import diagnostics
from tft_diffusion_modules import train_model
from tft_intense_game import BASE, game

OUT = BASE/'tft_intense_suite_20261122'
PREVIOUS = BASE/'tft_diffusion_modules_20261121'
DATA_SEEDS = list(range(3001, 3041))
CALIBRATION_SEEDS = list(range(4001, 4009))
VALIDATION_SEEDS = list(range(5001, 5009))
TEST_SEEDS = list(range(6001, 6009))
TRAIN_SEEDS = [47, 83]
GROUPS = ['diffusion', 'diffusion_gibbs', 'diffusion_vector', 'diffusion_vector_residual']
WORKERS = 4


def save(result):
    tmp = OUT/'metrics.tmp'
    tmp.write_text(json.dumps(result, indent=2), encoding='utf8')
    tmp.replace(OUT/'metrics.json')


def task(seed, kind, collect=False, model_path=None, value_path=None):
    result = {'seed':seed, 'seat':seed % 8, 'kind':kind, 'collect':collect}
    if model_path is not None: result['model_path'] = str(model_path)
    if value_path is not None: result['value_path'] = str(value_path)
    return result


def collect_tasks(seeds):
    kinds = ['teacher', 'econ', 'diffusion', 'diffusion_gibbs']
    return [task(seed, kinds[i%4], True,
                 PREVIOUS/('diffusion_gibbs.pt' if kinds[i%4]=='diffusion_gibbs' else 'diffusion.pt'))
            for i, seed in enumerate(seeds)]


def batch(pool, tasks, label, on_result):
    futures = {pool.submit(game, item): item for item in tasks}
    for future in as_completed(futures):
        original = futures[future]
        try:
            meta, rows = future.result()
        except Exception as exc:
            meta = {'seed':original['seed'], 'seat':original['seat'], 'kind':original['kind'],
                    'reason':'worker_error', 'error':str(exc), 'rank':None, 'steps':0}
            rows = []
        on_result(meta, rows)
        print(label, meta, 'samples', len(rows), flush=True)


def valid(meta, rows):
    return meta['reason'] == 'natural_end' and meta.get('rank') in range(1,9) and bool(rows)


def save_data(episodes, filename):
    payload = {}
    for i,(meta,rows) in enumerate(episodes):
        x,m,a = zip(*rows)
        payload[f'x_{i}'] = np.stack(x).astype(np.float32)
        payload[f'mask_{i}'] = np.packbits(np.stack(m), axis=1)
        payload[f'action_{i}'] = np.asarray(a, dtype=np.int16)
        payload[f'seed_{i}'] = np.asarray(meta['seed'], dtype=np.int64)
        payload[f'rank_{i}'] = np.asarray(meta['rank'], dtype=np.int8)
    np.savez_compressed(OUT/filename, **payload)


def value_fit(episodes):
    rng = np.random.default_rng(49)
    xs=[]; aa=[]; yy=[]; groups=[]
    for i,(meta,rows) in enumerate(episodes):
        ix = rng.choice(len(rows), min(len(rows),128), replace=False)
        for j in ix:
            x,_,a = rows[int(j)]
            xs.append(x); aa.append(a); yy.append((8-meta['rank'])/7); groups.append(i)
    return crossfit(np.asarray(xs), np.asarray(aa), np.asarray(yy), np.asarray(groups))


def calibration(episodes, coefficient):
    items=[]
    for meta,rows in episodes:
        x,_,a = zip(*rows)
        pred = np.clip(design(np.stack(x), np.asarray(a)) @ coefficient,0,1)
        residual = (8-meta['rank'])/7 - pred
        items.append({'seed':meta['seed'],'seat':meta['seat'], 'n':len(rows),
                      'mean_residual':float(residual.mean()), 'mse':float(np.mean(residual**2))})
    return {'valid_games':len(items), 'game_balanced_mse':float(np.mean([x['mse'] for x in items])) if items else None,
            'per_game':items}


def group_metrics(games):
    complete=[g['rank'] for g in games if g['reason']=='natural_end' and g.get('rank') in range(1,9)]
    return {'attempted':len(games), 'valid':len(complete),
            'mean_rank':float(np.mean(complete)) if complete else None,
            'top4_rate':float(np.mean(np.asarray(complete)<=4)) if complete else None,
            'first_rate':float(np.mean(np.asarray(complete)==1)) if complete else None,
            'warning_games':sum(g.get('internal_warning_count',0)>0 for g in games),
            'warning_lines':sum(g.get('internal_warning_count',0) for g in games)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started=time.monotonic()
    result={'status':'collecting', 'data_seeds':DATA_SEEDS, 'calibration_seeds':CALIBRATION_SEEDS,
            'validation_seeds':VALIDATION_SEEDS, 'test_seeds':TEST_SEEDS, 'training_seeds':TRAIN_SEEDS,
            'groups':GROUPS, 'workers':WORKERS, 'rank_rule':'upstream sequential tie ordering',
            'opponent':'seven econ_policy fixed across groups', 'train_games':[], 'calibration_games':[],
            'evaluation':[], 'notes':['all models share same collected data; 2 training seeds only',
              'seed and seat matched, but different actions can change random-number consumption',
              'stderr and simulator behavior not comprehensively audited; warning count is only text heuristic',
              'single-action diffusion, vector tilt and residual scaling do not prove martingale or fixed-point properties',
              'test seeds held out of data and calibration; multiple groups reported, no selection correction']}
    save(result)
    train=[]; calibr=[]
    ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        def on_train(meta, rows):
            result['train_games'].append(meta)
            if valid(meta, rows): train.append((meta, rows))
            save(result)
        batch(pool, collect_tasks(DATA_SEEDS), 'train_data', on_train)
        def on_calib(meta, rows):
            result['calibration_games'].append(meta)
            if valid(meta, rows): calibr.append((meta, rows))
            save(result)
        batch(pool, collect_tasks(CALIBRATION_SEEDS), 'calibration_data', on_calib)
        train.sort(key=lambda x:x[0]['seed']); calibr.sort(key=lambda x:x[0]['seed'])
        if len(train)<30 or len(calibr)<6:
            raise RuntimeError(f'insufficient valid games: train={len(train)} calibration={len(calibr)}')
        save_data(train, 'train_trajectories.npz'); save_data(calibr, 'calibration_trajectories.npz')
        cond=value_fit(train)
        np.savez(OUT/'conditional_value.npz', coef=cond['coef'], family_rmse=cond['family_rmse'], family_counts=cond['family_counts'])
        result['crossfit']={'global_game_balanced_mse':cond['global_oof_mse'], 'per_game':cond['per_game'],
                            'sample_count':len(cond['residual'])}
        result['heldout_calibration']=calibration(calibr, cond['coef'])
        episodes=[]
        for meta,rows in train:
            xx,mm,aa=zip(*rows)
            episodes.append((np.stack(xx),np.stack(mm),np.asarray(aa),meta['rank']))
        costs=[(m['rank']-1)/7 for m,_ in train]
        result['weight_diagnostics']={str(beta):diagnostics(costs,beta) for beta in (0.,0.5)}
        result['device']=str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        result['status']='training_and_evaluation'; save(result)
        for training_seed in TRAIN_SEEDS:
            paths={}
            for beta in (0.,0.5):
                model,loss,_=train_model(episodes,beta,torch.device(result['device']),train_seed=training_seed)
                path=OUT/f'diffusion_seed{training_seed}_beta{beta:g}.pt'
                torch.save({'model':model.state_dict(),'obs_dim':episodes[0][0].shape[1], 'steps':4, 'alpha':.1},path)
                paths[beta]=path
                print('trained',training_seed,beta,'loss',loss,flush=True)
            for split,seeds in [('validation',VALIDATION_SEEDS),('test',TEST_SEEDS)]:
                jobs=[]
                for kind in GROUPS:
                    model_path=paths[.5 if kind=='diffusion_gibbs' else 0.]
                    jobs.extend(task(seed,kind,False,model_path,OUT/'conditional_value.npz') for seed in seeds)
                def on_eval(meta,rows):
                    result['evaluation'].append({'split':split,'training_seed':training_seed,**meta})
                    save(result)
                batch(pool,jobs,f'eval_{training_seed}_{split}',on_eval)
                for kind in GROUPS:
                    records=[r for r in result['evaluation'] if r['split']==split and r['training_seed']==training_seed and r['kind']==kind]
                    print('summary',training_seed,split,kind,group_metrics(records),flush=True)
        result['summaries']={}
        for split in ('validation','test'):
            result['summaries'][split]={}
            for kind in GROUPS:
                result['summaries'][split][kind]={str(seed):group_metrics([r for r in result['evaluation']
                    if r['split']==split and r['training_seed']==seed and r['kind']==kind]) for seed in TRAIN_SEEDS}
    result['elapsed_seconds']=round(time.monotonic()-started,2)
    result['status']='completed'; save(result)
    print('COMPLETED',flush=True)


if __name__=='__main__': main()
