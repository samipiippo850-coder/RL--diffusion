"""Small online diffusion GRPO vs preference-learned-reward GRPO pilot.

Preference labels come from simulated final ranks, not human preferences; this
is preference-based reward learning, not demonstrated recovery of true reward.
"""
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from agent_measure.discrete_diffusion import CategoricalDenoiser
from agent_measure.diffusion_grpo import path_log_prob, relative_advantages, clipped_loss
from tft_intense_game import BASE, game

OUT = BASE/'tft_grpo_pilot_20261122'
INIT = BASE/'tft_v2_suite_20261122/diffusion_seed247_beta0.pt'
GROUP_SIZE = 4
ROUNDS = 2
GROUPS_PER_ROUND = 4

def valid(meta, rows):
    return meta['reason'] == 'natural_end' and meta.get('rank') in range(1,9) and bool(rows)

def episode_features(rows):
    # Public simulator observation encoding plus action-family frequencies;
    # does not include the terminal rank or infos/player state.
    x = np.stack([r[0][-82:-6] for r in rows])
    families = np.bincount([int(r[2])//38 for r in rows], minlength=55).astype(np.float32)
    return np.concatenate((x.mean(0), families/max(1,len(rows)))).astype(np.float32)

def run_batch(pool, jobs, sink, label):
    futures = {pool.submit(game,job):job for job in jobs}
    for f in as_completed(futures):
        job=futures[f]
        try: meta,rows=f.result()
        except Exception as exc: meta,rows={**job,'reason':'worker_error','error':repr(exc)},[]
        sink(job,meta,rows)
        print(label,meta,'samples',len(rows),flush=True)

def fit_preference(train):
    pairs=[]
    for seed,episodes in sorted(train.items()):
        if set(episodes) != {'econ','teacher'}: continue
        strong, weak = episodes['econ'], episodes['teacher']
        if strong[0]['rank'] == weak[0]['rank']: continue
        winner, loser = (strong,weak) if strong[0]['rank'] < weak[0]['rank'] else (weak,strong)
        pairs.append(episode_features(winner[1])-episode_features(loser[1]))
    if len(pairs)<8: raise RuntimeError('not enough non-tied preference comparisons')
    diffs = np.stack(pairs)
    scale = np.maximum(diffs.std(axis=0), 0.1).astype(np.float32)
    features = torch.from_numpy(diffs/scale)
    w = torch.zeros(features.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w],lr=0.02)
    for _ in range(300):
        loss = F.softplus(-(features@w)).mean() + .05*(w*w).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return (w.detach().numpy()/scale).astype(np.float32), len(pairs)

def preference_accuracy(pairs, weights):
    correct=total=0
    for _, episodes in pairs.items():
        if set(episodes) != {'econ','teacher'}: continue
        a,b=episodes['econ'],episodes['teacher']
        if a[0]['rank']==b[0]['rank']: continue
        predicted=float((episode_features(a[1])-episode_features(b[1])) @ weights)
        if predicted==0: continue
        correct += (predicted>0)==(a[0]['rank']<b[0]['rank']); total+=1
    return {'correct':int(correct),'count':total,'accuracy':correct/total if total else None}

def update(model, optimizer, episodes, reward_fn, device, seed):
    rng=np.random.default_rng(seed)
    selected=[]; group_stats=[]
    groups={}
    for meta,rows in episodes: groups.setdefault(meta['seed'],[]).append((meta,rows))
    for group_seed, group in sorted(groups.items()):
        # A partly failed group is excluded, never silently reweighted.
        if len(group)!=GROUP_SIZE: continue
        returns=torch.tensor([reward_fn(meta,rows) for meta,rows in group],dtype=torch.float32)
        advantages=relative_advantages(returns).numpy()
        group_stats.append({'seed':group_seed,'rewards':returns.tolist(),'advantages':advantages.tolist()})
        for (meta,rows),adv in zip(group,advantages):
            ix=rng.choice(len(rows),min(32,len(rows)),replace=False)
            selected.extend((rows[int(i)],float(adv)) for i in ix)
    if not selected: raise RuntimeError('no complete groups')
    x=torch.as_tensor(np.stack([r[0][0] for r in selected]),device=device)
    mask=torch.as_tensor(np.stack([r[0][1] for r in selected]),device=device)
    chains=torch.as_tensor(np.stack([r[0][3] for r in selected]),device=device)
    advantage=torch.as_tensor([r[1] for r in selected],dtype=torch.float32,device=device)
    model.to(device).train()
    with torch.no_grad(): old=path_log_prob(model,x,mask,chains).detach()
    losses=[]
    for _ in range(2):
        new=path_log_prob(model,x,mask,chains)
        loss=clipped_loss(new,old,advantage)
        optimizer.zero_grad(); loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0))
        optimizer.step(); losses.append(float(loss.detach().cpu()))
    with torch.no_grad():
        new=path_log_prob(model,x,mask,chains)
        delta=new-old
        mean_kl=float((delta.exp()-1-delta).mean().cpu())
    model.cpu().eval()
    return {'groups':group_stats,'n_transitions':len(selected),'losses':losses,
            'gradient_norm_before_clip':grad,'approx_reverse_kl':mean_kl}

def save(result):
    tmp=OUT/'metrics.tmp'; tmp.write_text(json.dumps(result,indent=2),encoding='utf8'); tmp.replace(OUT/'metrics.json')

def main():
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(3); start=time.monotonic()
    result={'status':'demonstrations','demo_games':[],'updates':[],'evaluation':[],
            'notes':['preference labels from simulated ranks, not human experts or recovered true reward',
                     'economic script is a proxy demonstrator',
                     'clipped likelihood ratio uses the sampled joint denoising path, not marginal action probability',
                     'identical initial seeds do not synchronize later random streams after different actions',
                     'group normalization uses per-game terminal scores; no learned critic',
                     'small pilot, shared initialization; held-out pair accuracy does not establish reward identifiability']}
    save(result)
    demos={}; holdout={}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        jobs=[{'seed':seed,'seat':seed%8,'kind':kind,'collect':True}
              for seed in range(20001,20025) for kind in ('econ','teacher')]
        def demo_sink(job,meta,rows):
            result['demo_games'].append(meta)
            if valid(meta,rows): (demos if job['seed']<20017 else holdout).setdefault(job['seed'],{})[job['kind']]=(meta,rows)
            save(result)
        run_batch(pool,jobs,demo_sink,'demo')
        weights,n=fit_preference(demos)
        np.savez(OUT/'preference_reward.npz',weights=weights)
        result['preference_reward']={'train_pairs':n,'train':preference_accuracy(demos,weights),
                                     'heldout':preference_accuracy(holdout,weights)}
        result['status']='training'; save(result)
        original=torch.load(INIT,map_location='cpu',weights_only=False)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for arm in ('rank_grpo','preference_grpo'):
            model=CategoricalDenoiser(original['obs_dim'],55*38)
            model.load_state_dict(original['model']); model.eval()
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
            for round_no in range(ROUNDS):
                current=OUT/f'{arm}_round{round_no}_behavior.pt'
                torch.save({'model':model.state_dict()},current)
                # Same game seed per group, independent policy-sampling seed.
                group_seeds=range(21001+round_no*GROUPS_PER_ROUND,21001+(round_no+1)*GROUPS_PER_ROUND)
                jobs=[{'seed':s,'seat':s%8,'kind':'diffusion_grpo','collect':True,
                       'model_path':str(current),'sample_seed':s*1000+j+700000}
                      for s in group_seeds for j in range(GROUP_SIZE)]
                episodes=[]; metas=[]
                def train_sink(job,meta,rows):
                    meta['sample_seed']=job['sample_seed']; metas.append(meta)
                    if valid(meta,rows): episodes.append((meta,rows))
                run_batch(pool,jobs,train_sink,f'{arm}_round{round_no}')
                reward_fn=(lambda meta,rows:(8-meta['rank'])/7) if arm=='rank_grpo' else (lambda meta,rows:float(episode_features(rows)@weights))
                stat=update(model,optimizer,episodes,reward_fn,device,550+round_no)
                result['updates'].append({'arm':arm,'round':round_no,'games':metas,**stat}); save(result)
                torch.save({'model':model.state_dict()},OUT/f'{arm}.pt')
        result['status']='evaluation'; save(result)
        paths={'frozen_diffusion':INIT,'rank_grpo':OUT/'rank_grpo.pt','preference_grpo':OUT/'preference_grpo.pt'}
        jobs=[{'seed':s,'seat':s%8,'kind':'diffusion_grpo','collect':False,'model_path':str(path),'arm':arm}
              for arm,path in paths.items() for s in range(22001,22017)]
        def eval_sink(job,meta,rows):
            result['evaluation'].append({'arm':job['arm'],**meta}); save(result)
        run_batch(pool,jobs,eval_sink,'eval')
    result['status']='completed';result['elapsed_seconds']=round(time.monotonic()-start,2);save(result)
    print('COMPLETED',flush=True)

if __name__=='__main__': main()
