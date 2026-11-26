"""Aggregate exploratory branch outcomes by state, not by raw rollouts."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

PATH=Path('path_measure_preparation/training_results_20261124/segment_pilot.json')

def summarize(p):
    groups={};details=[];all_branches=[]
    for state in p['tasks']:
        if state['reason']!='completed':continue
        names=list(state['choices'])
        if len(names)!=2:continue
        a,b=names
        byseed={name:{x['continuation_seed']:x for x in state['outcomes'][name]} for name in names}
        pairs=[]
        for seed in sorted(set(byseed[a]) & set(byseed[b])):
            x,y=byseed[a][seed],byseed[b][seed]
            if all(z['reason']=='natural_end' and z.get('rank') in range(1,9) and not z.get('warnings') for z in (x,y)):
                pairs.append((x['rank'],y['rank']))
        all_branches.extend(x for name in names for x in state['outcomes'][name])
        detail={'case':state['case'],'seed':state['seed'],'round':state['round'],'health':state['health'],
                'gold':state['gold'],'level':state['level'],'actions':[a,b],'valid_pairs':len(pairs),
                'source_warnings':state.get('source_warnings',0),'proposals':state['proposals'],
                'choices':state['choices']}
        if pairs:
            arr=np.asarray(pairs,dtype=float)
            detail.update({'mean_rank':{a:float(arr[:,0].mean()),b:float(arr[:,1].mean())},
                           'rank_delta_a_minus_b':float((arr[:,0]-arr[:,1]).mean()),
                           'top4':{a:float((arr[:,0]<=4).mean()),b:float((arr[:,1]<=4).mean())}})
            if not detail['source_warnings']:groups.setdefault(state['case'],[]).append(detail)
        details.append(detail)
    summary={}
    for case,states in groups.items():
        summary[case]={'states':len(states),'action_order':states[0]['actions'],
                       'mean_state_paired_rank_delta':float(np.mean([x['rank_delta_a_minus_b'] for x in states])),
                       'states_a_better':sum(x['rank_delta_a_minus_b']<0 for x in states),
                       'states_b_better':sum(x['rank_delta_a_minus_b']>0 for x in states),
                       'states_tied':sum(x['rank_delta_a_minus_b']==0 for x in states)}
    return {'status':p['status'],'task_counts':dict(Counter(f"{x['case']}:{x['reason']}" for x in p['tasks'])),
            'branch_count':len(all_branches),'branch_reasons':dict(Counter(x['reason'] for x in all_branches)),
            'branch_warning_count':sum(bool(x.get('warnings')) for x in all_branches),
            'source_warning_states':sum(bool(x.get('source_warnings')) for x in details),
            'summary':summary,'states':sorted(details,key=lambda x:(x['case'],x['seed']))}

if __name__=='__main__':
    result=summarize(json.loads(PATH.read_text(encoding='utf8')))
    print(json.dumps(result,indent=2,ensure_ascii=False))
