"""Larger diffusion GRPO and rank-supervised preference-reward comparison.

The reward model learns pairwise game-ranking preferences from four existing
behaviours, not human preference and not an identifiable inverse-RL reward.
"""
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
import numpy as np
import torch
import torch.nn.functional as F
from agent_measure.discrete_diffusion import CategoricalDenoiser
import tft_grpo_experiment as pilot
from tft_intense_game import BASE, game

OUT=BASE/'tft_grpo_scale_20261124'
MODEL_DIR=BASE/'tft_v2_suite_20261122'
INITS=(247,283)
TRAIN_DEMO=range(30001,30025)
HOLDOUT_DEMO=range(30025,30037)
TEST_SEEDS=range(32001,32025)
ROUNDS=3
GROUPS_PER_ROUND=4
GROUP_SIZE=4
DEMO_KINDS=('teacher','econ','diffusion','pref_pilot')

def save(result):
    tmp=OUT/'metrics.tmp'
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    tmp.replace(OUT/'metrics.json')

def run_batch(pool,jobs,sink,label):
    futures={pool.submit(game,job):job for job in jobs}
    for future in as_completed(futures):
        job=futures[future]
        try: meta,rows=future.result()
        except Exception as exc: meta,rows={**job,'reason':'worker_error','error':repr(exc)},[]
        sink(job,meta,rows)
        print(label,meta,'samples',len(rows),flush=True)

def pair_differences(groups):
    diffs=[]
    for seed, records in sorted(groups.items()):
        for a,b in combinations(sorted(records),2):
            r1,x1=records[a];r2,x2=records[b]
            if r1==r2: continue
            diffs.append((seed, x1-x2 if r1<r2 else x2-x1))
    return diffs

def fit_reward(groups):
    pairs=pair_differences(groups)
    if len(groups)<16 or len(pairs)<20: raise RuntimeError('insufficient demonstration comparisons')
    dx=np.stack([x for _,x in pairs]).astype(np.float32)
    scale=np.maximum(dx.std(axis=0),.1).astype(np.float32)
    x=torch.from_numpy(dx/scale)
    w=torch.zeros(x.shape[1],requires_grad=True)
    opt=torch.optim.Adam([w],lr=.02)
    for _ in range(300):
        loss=F.softplus(-(x@w)).mean()+.05*(w*w).mean()
        opt.zero_grad();loss.backward();opt.step()
    return (w.detach().numpy()/scale).astype(np.float32)

def pair_accuracy(groups,w):
    pairs=pair_differences(groups)
    correct=sum(float(dx@w)>0 for _,dx in pairs)
    return {'groups':len(groups),'pairs':len(pairs),'correct':int(correct),
            'accuracy':correct/len(pairs) if pairs else None,
            'note':'pairs within a seed are dependent; not independent trials'}

def summary(games):
    valid=[g for g in games if g['reason']=='natural_end' and g.get('rank') in range(1,9)]
    if not valid: return {'attempted':len(games),'valid':0}
    ranks=np.array([g['rank'] for g in valid],dtype=float)
    scores=np.array([g['learned_reward'] for g in valid],dtype=float)
    corr=None
    if len(valid)>1 and scores.std()>1e-12 and ranks.std()>1e-12:
        corr=float(np.corrcoef(scores,(8-ranks)/7)[0,1])
    return {'attempted':len(games),'valid':len(valid),'mean_rank':float(ranks.mean()),
            'top4_rate':float((ranks<=4).mean()),'first_rate':float((ranks==1).mean()),
            'mean_learned_reward':float(scores.mean()),'reward_rank_utility_correlation':corr,
            'warning_games':sum(g.get('internal_warning_count',0)>0 for g in games)}

def main():
    OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(3); start=time.monotonic()
    result={'status':'demonstrations','demo_games':[],'updates':[],'evaluation':[],
      'plan':{'initializations':list(INITS),'demo_train_seeds':list(TRAIN_DEMO),
      'demo_holdout_seeds':list(HOLDOUT_DEMO),'test_seeds':list(TEST_SEEDS),
      'rounds':ROUNDS,'groups_per_round':GROUPS_PER_ROUND,'group_size':GROUP_SIZE},
      'notes':['preference labels use simulated final ranks; no human experts or recovered true reward',
      'demonstrator pool is heterogeneous, not verified expert or top competitive AI',
      'model probability ratio refers to sampled joint denoising path, not marginal action probability',
      'initial models share training dataset; initializations are not independent datasets',
      'same game seed does not force identical random future after different actions',
      'reward scores are held-out fixed reward model outputs; temporal association is not proof of reward hacking',
      'upstream orders simultaneous eliminations by iteration; output rank may be seat biased']}
    save(result)
    demos={}; hold={}
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
        jobs=[]
        for seed in list(TRAIN_DEMO)+list(HOLDOUT_DEMO):
            for label in DEMO_KINDS:
                job={'seed':seed,'seat':seed%8,'kind':label,'collect':True,'label':label}
                if label=='diffusion': job['model_path']=str(MODEL_DIR/'diffusion_seed247_beta0.pt')
                elif label=='pref_pilot':
                    job['kind']='diffusion_grpo';job['model_path']=str(BASE/'tft_grpo_pilot_20261122/preference_grpo.pt')
                jobs.append(job)
        def demo_sink(job,meta,rows):
            result['demo_games'].append({'demonstrator':job['label'],**meta})
            if pilot.valid(meta,rows):
                target=demos if job['seed'] in TRAIN_DEMO else hold
                target.setdefault(job['seed'],{})[job['label']]=(meta['rank'],pilot.episode_features(rows))
            save(result)
        run_batch(pool,jobs,demo_sink,'demo')
        weights=fit_reward(demos)
        np.savez(OUT/'preference_reward.npz',weights=weights)
        result['preference_reward']={'train':pair_accuracy(demos,weights),'heldout':pair_accuracy(hold,weights)}
        result['status']='training';save(result)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for init in INITS:
            initial=torch.load(MODEL_DIR/f'diffusion_seed{init}_beta0.pt',map_location='cpu',weights_only=False)
            for arm in ('rank_grpo','preference_grpo'):
                model=CategoricalDenoiser(initial['obs_dim'],55*38)
                model.load_state_dict(initial['model']);model.eval()
                optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
                for n in range(ROUNDS):
                    checkpoint=OUT/f'{arm}_init{init}_behavior{n}.pt'
                    torch.save({'model':model.state_dict()},checkpoint)
                    seeds=range(31001+n*GROUPS_PER_ROUND,31001+(n+1)*GROUPS_PER_ROUND)
                    jobs=[{'seed':s,'seat':s%8,'kind':'diffusion_grpo','collect':True,
                           'model_path':str(checkpoint),'sample_seed':s*1000+j+900000}
                          for s in seeds for j in range(GROUP_SIZE)]
                    episodes=[];metas=[]
                    def train_sink(job,meta,rows):
                        meta['sample_seed']=job['sample_seed'];metas.append(meta)
                        if pilot.valid(meta,rows): episodes.append((meta,rows))
                    run_batch(pool,jobs,train_sink,f'{init}_{arm}_{n}')
                    episodes.sort(key=lambda z:(z[0]['seed'],z[0]['sample_seed']))
                    if len(episodes)!=len(jobs): raise RuntimeError('incomplete training batch; no partial update')
                    for meta,rows in episodes: meta['learned_reward']=float(pilot.episode_features(rows)@weights)
                    reward=(lambda m,r:(8-m['rank'])/7) if arm=='rank_grpo' else (lambda m,r:m['learned_reward'])
                    stat=pilot.update(model,optimizer,episodes,reward,device,550+n)
                    result['updates'].append({'initialization':init,'arm':arm,'round':n,
                        'training_game_summary':summary(metas),'games':metas,**stat})
                    torch.save({'model':model.state_dict()},OUT/f'{arm}_init{init}.pt');save(result)
        result['status']='evaluation';save(result)
        jobs=[]
        for init in INITS:
            paths={'frozen':MODEL_DIR/f'diffusion_seed{init}_beta0.pt',
                   'rank_grpo':OUT/f'rank_grpo_init{init}.pt',
                   'preference_grpo':OUT/f'preference_grpo_init{init}.pt'}
            for arm,path in paths.items():
                jobs.extend({'seed':s,'seat':s%8,'kind':'diffusion_grpo','collect':True,
                             'model_path':str(path),'arm':arm,'initialization':init} for s in TEST_SEEDS)
        jobs.extend({'seed':s,'seat':s%8,'kind':'econ','collect':True,'arm':'econ','initialization':None}
                    for s in TEST_SEEDS)
        def eval_sink(job,meta,rows):
            if pilot.valid(meta,rows): meta['learned_reward']=float(pilot.episode_features(rows)@weights)
            result['evaluation'].append({'arm':job['arm'],'initialization':job['initialization'],**meta});save(result)
        run_batch(pool,jobs,eval_sink,'eval')
    result['summaries']={f'{init}_{arm}':summary([g for g in result['evaluation'] if g['initialization']==init and g['arm']==arm])
        for init in INITS for arm in ('frozen','rank_grpo','preference_grpo')}
    result['summaries']['econ']=summary([g for g in result['evaluation'] if g['arm']=='econ'])
    result['status']='completed';result['elapsed_seconds']=round(time.monotonic()-start,2);save(result)
    print('COMPLETED',flush=True)

if __name__=='__main__': main()
