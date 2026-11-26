"""Paired-seed exploratory comparisons; bootstrap unit is a full game seed."""
import json
from pathlib import Path
import numpy as np

p=json.loads(Path('path_measure_preparation/training_results_20261124/dagger.json').read_text(encoding='utf8'))
groups={kind:{g['seed']:g for g in p['evaluation'] if g['arm']==kind}
        for kind in ('dagger','original','fresh_teacher','econ')}
rng=np.random.default_rng(9)
for a,b in (('dagger','original'),('dagger','fresh_teacher')):
    seeds=sorted(groups[a].keys() & groups[b].keys())
    delta=np.array([groups[a][s]['rank']-groups[b][s]['rank'] for s in seeds])
    boot=delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
    clean=[s for s in seeds if not groups[a][s]['internal_warning_count'] and not groups[b][s]['internal_warning_count']]
    print(a,b,'n',len(seeds),'mean_delta',float(delta.mean()),'ci95',np.quantile(boot,[.025,.975]).tolist(),
          'warning_free_n',len(clean),'warning_free_delta',float(np.mean([groups[a][s]['rank']-groups[b][s]['rank'] for s in clean])))
