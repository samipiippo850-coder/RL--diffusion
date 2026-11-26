"""Descriptive matched-seed comparison; hindsight oracle is not a policy."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

data=json.loads(Path('path_measure_preparation/training_results_20261124/dagger.json').read_text(encoding='utf8'))
arms=('original','dagger','fresh_teacher','econ')
groups={arm:{g['seed']:g for g in data['evaluation'] if g['arm']==arm and g['reason']=='natural_end'} for arm in arms}
seeds=sorted(set.intersection(*(set(groups[a]) for a in arms)))
print('paired_seed_count',len(seeds))
for a,b in [('dagger','original'),('dagger','fresh_teacher')]:
    d=np.array([groups[a][s]['rank']-groups[b][s]['rank'] for s in seeds])
    print(a,'vs',b,{'better':int((d<0).sum()),'tied':int((d==0).sum()),'worse':int((d>0).sum()),
                     'mean_difference':float(d.mean())})
models=arms[:3]
for a in models:
    wins=sum(all(groups[a][s]['rank']<groups[b][s]['rank'] for b in models if b!=a) for s in seeds)
    best_or_tie=sum(groups[a][s]['rank']==min(groups[b][s]['rank'] for b in models) for s in seeds)
    print('descriptive',a,'sole_best',wins,'best_or_tied',best_or_tie)
print('hindsight_best_mean_rank',float(np.mean([min(groups[a][s]['rank'] for a in models) for s in seeds])))
print('warning_games',Counter(g['arm'] for g in data['evaluation'] if g.get('internal_warning_count',0)))
