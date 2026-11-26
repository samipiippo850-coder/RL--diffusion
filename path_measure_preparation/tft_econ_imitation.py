"""Clean full-game economy demonstrations and train factorized policies.

Natural action frequencies are retained. Splits are by complete game seed.
All outputs are exploratory; the scripted demonstrator uses simulator info.
"""
import json
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import torch
from agent_measure.hierarchical_policy import (HierarchicalPolicy,HierarchicalDenoiser,
                                               FAMILY_NAMES,action_families)
from tft_intense_game import BASE,game

OUT=BASE/'tft_econ_imitation_20261124'
SPLITS={'train':list(range(41001,41065)), 'validation':list(range(42001,42017)),
        'test':list(range(43001,43017))}
EVAL_SEEDS=list(range(44001,44017))
UPDATES=1200

def save(result):
    tmp=OUT/'metrics.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8');tmp.replace(OUT/'metrics.json')

def inspect_episode(meta,rows):
    if meta.get('reason')!='natural_end': return 'not_natural_end'
    if meta.get('rank') not in range(1,9): return 'invalid_rank'
    if meta.get('internal_warning_count',0): return 'simulator_warning'
    if meta.get('max_pending_items_per_player',0): return 'unobserved_item_overflow'
    if not rows:return 'empty_trajectory'
    for row in rows:
        if len(row)!=3:return 'invalid_row'
        x,mask,a=row
        if x.ndim!=1 or not np.isfinite(x).all():return 'nonfinite_observation'
        if mask.shape!=(55*38,) or not mask.any() or not 0<=int(a)<len(mask) or not mask[int(a)]:return 'invalid_mask_or_action'
    return None

def write_episode(split,meta,rows):
    x,m,a=zip(*rows);a=np.asarray(a,dtype=np.int16)
    fam=action_families().numpy()[a].astype(np.int8)
    path=OUT/'data'/split/f"seed_{meta['seed']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,obs=np.stack(x).astype(np.float32),
                        mask=np.packbits(np.stack(m),axis=1),action=a,family=fam)
    return {'seed':meta['seed'],'count':len(a),'family_counts':dict(Counter(FAMILY_NAMES[i] for i in fam))}

def load_episode(split,seed):
    with np.load(OUT/'data'/split/f'seed_{seed}.npz',allow_pickle=False) as f:
        return (f['obs'].copy(),np.unpackbits(f['mask'],axis=1,count=55*38).astype(bool),
                f['action'].astype(np.int64),f['family'].astype(np.int64))

def normalizer(episodes):
    sums=np.zeros(episodes[0][0].shape[1],dtype=np.float64);squares=sums.copy();n=0
    for x,_,_,_ in episodes:
        sums+=x.sum(0,dtype=np.float64);squares+=np.square(x.astype(np.float64)).sum(0);n+=len(x)
    mean=sums/n;std=np.sqrt(np.maximum(squares/n-mean*mean,0))
    std[std<1e-5]=1
    return mean.astype(np.float32),std.astype(np.float32)

def heldout_metrics(model,kind,episodes,mean,std,device,seed):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    xs=[];ms=[];ys=[];fs=[]
    for x,m,y,f in episodes:
        ix=rng.choice(len(y),min(64,len(y)),replace=False)
        xs.append(x[ix]);ms.append(m[ix]);ys.append(y[ix]);fs.append(f[ix])
    x=np.clip((np.concatenate(xs)-mean)/std,-10,10).astype(np.float32)
    masks=np.concatenate(ms); labels=np.concatenate(ys);families=np.concatenate(fs)
    predictions=[];model.to(device).eval()
    with torch.no_grad():
        for start in range(0,len(x),128):
            xx=torch.from_numpy(x[start:start+128]).to(device);mm=torch.from_numpy(masks[start:start+128]).to(device)
            # Both metrics use sampled actions, matching the deployed policies.
            pred=(model.sample(xx,mm) if kind=='hier_diffusion' else torch.distributions.Categorical(logits=model(xx,mm)).sample())
            predictions.extend(pred.cpu().tolist())
    pred=np.asarray(predictions); pred_family=action_families().numpy()[pred]
    per_family={}
    for i,name in enumerate(FAMILY_NAMES):
        ix=families==i
        if ix.any():per_family[name]={'count':int(ix.sum()),'family_recall':float((pred_family[ix]==i).mean()),
                                      'exact_accuracy':float((pred[ix]==labels[ix]).mean())}
    return {'samples':len(labels),'exact_accuracy':float((pred==labels).mean()),
            'family_accuracy':float((pred_family==families).mean()),'per_family':per_family,
            'predicted_counts':dict(Counter(FAMILY_NAMES[i] for i in pred_family)),
            'illegal_count':int((~masks[np.arange(len(pred)),pred]).sum())}

def train(kind,episodes,mean,std,device,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=(HierarchicalPolicy(len(mean)) if kind=='hier_bc' else HierarchicalDenoiser(len(mean))).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=3e-4);losses=[]
    for step in range(UPDATES):
        games=rng.integers(0,len(episodes),size=128)
        batch=[(episodes[int(g)],None) for g in games]
        ix=[int(rng.integers(len(e[0]))) for e,_ in batch]
        x=np.stack([e[0][j] for (e,_),j in zip(batch,ix)])
        m=np.stack([e[1][j] for (e,_),j in zip(batch,ix)])
        y=np.asarray([e[2][j] for (e,_),j in zip(batch,ix)])
        xx=torch.as_tensor(np.clip((x-mean)/std,-10,10),device=device)
        mm=torch.as_tensor(m,device=device);yy=torch.as_tensor(y,device=device)
        loss=(-model(xx,mm).gather(1,yy[:,None]).mean() if kind=='hier_bc' else model.loss(xx,yy,mm).mean())
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if (step+1)%200==0:losses.append({'step':step+1,'loss':float(loss.detach().cpu())})
    return model,losses

def main():
    OUT.mkdir(parents=True,exist_ok=True);start=time.monotonic();torch.set_num_threads(3)
    result={'status':'collecting','splits':SPLITS,'eval_seeds':EVAL_SEEDS,'games':[],
            'accepted':{'train':[],'validation':[],'test':[]},'rejected':[],
            'notes':['natural action frequencies retained; no automatic removal of waiting or moves',
                     'script uses own simulator player information; information equivalence not fully audited',
                     'warnings measured through captured text only',
                     'model uses only observation; same seed can diverge after different actions',
                     'factorized categorical diffusion is a single-action generator, not trajectory diffusion']}
    save(result)
    ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=4,mp_context=ctx) as pool:
        jobs=[{'seed':s,'seat':s%8,'kind':'econ','collect':True,'raw_collect':True,'split':split}
              for split,seeds in SPLITS.items() for s in seeds]
        futures={pool.submit(game,j):j for j in jobs}
        for future in as_completed(futures):
            j=futures[future]
            try:meta,rows=future.result()
            except Exception as exc:meta,rows={**j,'reason':'worker_error','error':repr(exc)},[]
            reason=inspect_episode(meta,rows)
            result['games'].append({'split':j['split'],**meta,'rejection':reason})
            if reason:result['rejected'].append({'seed':j['seed'],'split':j['split'],'reason':reason})
            else:result['accepted'][j['split']].append(write_episode(j['split'],meta,rows))
            save(result);print('data',j['split'],meta,'rejection',reason,flush=True)
        for split,minimum in [('train',48),('validation',12),('test',12)]:
            if len(result['accepted'][split])<minimum:raise RuntimeError('insufficient clean '+split+' games')
        episodes={split:[load_episode(split,r['seed']) for r in sorted(items,key=lambda x:x['seed'])]
                  for split,items in result['accepted'].items()}
        mean,std=normalizer(episodes['train'])
        np.savez(OUT/'normalizer.npz',mean=mean,std=std)
        result['data_summary']={split:{'games':len(items),'samples':sum(x['count'] for x in items),
            'family_counts':dict(sum((Counter(x['family_counts']) for x in items),Counter()))} for split,items in result['accepted'].items()}
        result['status']='training';save(result)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for kind in ('hier_bc','hier_diffusion'):
            model,losses=train(kind,episodes['train'],mean,std,device,71)
            val=heldout_metrics(model,kind,episodes['validation'],mean,std,device,88)
            test=heldout_metrics(model,kind,episodes['test'],mean,std,device,89)
            torch.save({'model':model.cpu().state_dict(),'mean':mean,'std':std,'kind':kind},OUT/(kind+'.pt'))
            result.setdefault('training',{})[kind]={'updates':UPDATES,'losses':losses,'validation':val,'test':test}
            save(result);print('trained',kind,'val',val,'test',test,flush=True)
        result['status']='evaluation';result['evaluation']=[];save(result)
        jobs=[{'seed':s,'seat':s%8,'kind':kind,'model_path':str(OUT/(kind+'.pt'))}
              for kind in ('hier_bc','hier_diffusion') for s in EVAL_SEEDS]
        jobs += [{'seed':s,'seat':s%8,'kind':'econ'} for s in EVAL_SEEDS]
        futures={pool.submit(game,j):j for j in jobs}
        for future in as_completed(futures):
            j=futures[future]
            try:meta,_=future.result()
            except Exception as exc:meta={**j,'reason':'worker_error','error':repr(exc)}
            result['evaluation'].append(meta);save(result);print('eval',meta,flush=True)
    result['status']='completed';result['elapsed_seconds']=round(time.monotonic()-start,2);save(result)
    print('COMPLETED',flush=True)

if __name__=='__main__':main()
