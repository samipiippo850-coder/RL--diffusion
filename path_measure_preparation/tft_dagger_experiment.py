"""Small DAgger study with an equal-sized fresh teacher-data control.

Expert labels are requested on the learner's actual pre-action state; the
learner's action, not the label, is executed. Demonstrator uses its own
simulator-info fields, so exact information equivalence is not yet established.
"""
import json
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from tft_intense_game import BASE,game
from tft_econ_imitation import (inspect_episode,normalizer,train,heldout_metrics,
                                load_episode,FAMILY_NAMES,action_families)
from agent_measure.hierarchical_policy import HierarchicalDenoiser

OUT=BASE/'tft_dagger_20261124'
ORIGINAL=BASE/'tft_econ_imitation_20261124'
SPLITS={'dagger_train':list(range(49001,49033)),
        'dagger_validation':list(range(50001,50009)),
        'fresh_teacher':list(range(51001,51033))}
EVAL=list(range(52001,52025))

def save(result):
    tmp=OUT/'metrics.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/'metrics.json')

def clean(meta,rows,dagger):
    core=[r[:3] for r in rows]
    reason=inspect_episode(meta,core)
    if reason:return reason
    if dagger:
        for r in rows:
            if len(r)!=4 or not 0<=int(r[3])<55*38 or not r[1][int(r[3])]:return 'invalid_executed_action'
    return None

def write(split,meta,rows,dagger):
    x=np.stack([r[0] for r in rows]).astype(np.float32)
    masks=np.stack([r[1] for r in rows]);actions=np.asarray([r[2] for r in rows],dtype=np.int16)
    family=action_families().numpy()[actions].astype(np.int8)
    behavior=np.asarray([r[3] if dagger else r[2] for r in rows],dtype=np.int16)
    path=OUT/'data'/split/f"seed_{meta['seed']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,obs=x,mask=np.packbits(masks,axis=1),action=actions,family=family,behavior=behavior)
    return {'seed':meta['seed'],'count':len(rows),'family_counts':dict(Counter(FAMILY_NAMES[i] for i in family)),
            'exact_agreement':float((actions==behavior).mean()) if dagger else None}

def read(split,seed):
    with np.load(OUT/'data'/split/f'seed_{seed}.npz',allow_pickle=False) as f:
        return f['obs'].copy(),np.unpackbits(f['mask'],axis=1,count=55*38).astype(bool),f['action'].astype(np.int64),f['family'].astype(np.int64)

def summary(games):
    valid=[g for g in games if g.get('reason')=='natural_end' and g.get('rank') in range(1,9)]
    if not valid:return {'attempted':len(games),'valid':0}
    ranks=np.array([g['rank'] for g in valid]);counts=Counter()
    for g in valid:counts.update(g.get('target_action_counts',{}))
    n=sum(counts.values())
    return {'attempted':len(games),'valid':len(valid),'mean_rank':float(ranks.mean()),
            'top4_rate':float((ranks<=4).mean()),'first_rate':float((ranks==1).mean()),
            'action_counts':dict(counts),'action_fraction':{k:v/n for k,v in counts.items()} if n else {},
            'warning_games':sum(g.get('internal_warning_count',0)>0 for g in games)}

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);start=time.monotonic()
    prev=json.loads((ORIGINAL/'metrics.json').read_text(encoding='utf8'))
    baseline=ORIGINAL/'hier_diffusion.pt'
    initial=[load_episode('train',r['seed']) for r in sorted(prev['accepted']['train'],key=lambda x:x['seed'])]
    result={'status':'collecting','splits':SPLITS,'eval_seeds':EVAL,'games':[],
            'accepted':{k:[] for k in SPLITS},'rejected':[],
            'notes':['teacher is economy script, not human or proven expert',
                     'on-policy labels use scripted simulator info; learner observes encoded fields',
                     'training is natural frequency; no assumed useless move or pass is deleted',
                     'same game seed does not force identical future after different actions',
                     'fresh-teacher control matches number of added games when all pass cleaning',
                     'single initialization and small number of games; not proof of improvement']}
    save(result)
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        jobs=[]
        for split,seeds in SPLITS.items():
            dagger=split!='fresh_teacher'
            for s in seeds:
                j={'seed':s,'seat':s%8,'kind':'hier_diffusion' if dagger else 'econ','collect':True,
                   'raw_collect':True,'dagger_collect':dagger,'split':split}
                if dagger:j['model_path']=str(baseline)
                jobs.append(j)
        futures={pool.submit(game,j):j for j in jobs}
        for f in as_completed(futures):
            j=futures[f]
            try:meta,rows=f.result()
            except Exception as exc:meta,rows={**j,'reason':'worker_error','error':repr(exc)},[]
            reason=clean(meta,rows,j['dagger_collect'])
            result['games'].append({'split':j['split'],**meta,'rejection':reason})
            if reason:result['rejected'].append({'seed':j['seed'],'split':j['split'],'reason':reason})
            else:result['accepted'][j['split']].append(write(j['split'],meta,rows,j['dagger_collect']))
            save(result);print('data',j['split'],meta,'rejection',reason,flush=True)
        for split,minimum in [('dagger_train',24),('fresh_teacher',24),('dagger_validation',6)]:
            if len(result['accepted'][split])<minimum:raise RuntimeError('insufficient clean '+split)
        # Equalize added-game budgets deterministically if cleaning rejects games.
        n=min(len(result['accepted']['dagger_train']),len(result['accepted']['fresh_teacher']))
        additions={k:[read(k,x['seed']) for x in sorted(result['accepted'][k],key=lambda z:z['seed'])[:n]]
                   for k in ('dagger_train','fresh_teacher')}
        val=[read('dagger_validation',x['seed']) for x in sorted(result['accepted']['dagger_validation'],key=lambda z:z['seed'])]
        result['data_summary']={'base_games':len(initial),'added_games_each':n,
          'accepted_games':{k:len(v) for k,v in result['accepted'].items()},
          'samples_by_split':{k:sum(x['count'] for x in v) for k,v in result['accepted'].items()},
          'dagger_labels':dict(sum((Counter(x['family_counts']) for x in result['accepted']['dagger_train']),Counter()))}
        result['status']='training';save(result)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_ck=torch.load(baseline,map_location='cpu',weights_only=False)
        base_model=HierarchicalDenoiser(len(base_ck['mean']));base_model.load_state_dict(base_ck['model'])
        result['on_policy_validation']={'original':heldout_metrics(base_model,'hier_diffusion',val,base_ck['mean'],base_ck['std'],device,91)}
        for arm,split in [('dagger','dagger_train'),('fresh_teacher','fresh_teacher')]:
            episodes=initial+additions[split];mean,std=normalizer(episodes)
            model,losses=train('hier_diffusion',episodes,mean,std,device,71)
            result['on_policy_validation'][arm]=heldout_metrics(model,'hier_diffusion',val,mean,std,device,91)
            torch.save({'model':model.cpu().state_dict(),'mean':mean,'std':std,'kind':'hier_diffusion'},OUT/(arm+'.pt'))
            result.setdefault('training',{})[arm]={'losses':losses,'updates':1200};save(result)
        result['status']='evaluation';result['evaluation']=[];save(result)
        models={'original':baseline,'dagger':OUT/'dagger.pt','fresh_teacher':OUT/'fresh_teacher.pt'}
        jobs=[{'seed':s,'seat':s%8,'kind':'hier_diffusion','model_path':str(path),'arm':arm}
              for arm,path in models.items() for s in EVAL]
        jobs += [{'seed':s,'seat':s%8,'kind':'econ','arm':'econ'} for s in EVAL]
        futures={pool.submit(game,j):j for j in jobs}
        for f in as_completed(futures):
            j=futures[f]
            try:meta,_=f.result()
            except Exception as exc:meta={**j,'reason':'worker_error','error':repr(exc)}
            result['evaluation'].append({'arm':j['arm'],**meta});save(result);print('eval',j['arm'],meta,flush=True)
    result['summaries']={arm:summary([g for g in result['evaluation'] if g['arm']==arm])
                         for arm in ('original','dagger','fresh_teacher','econ')}
    result['status']='completed';result['elapsed_seconds']=round(time.monotonic()-start,2);save(result)
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
