import collections
import json
from pathlib import Path

p=json.loads(Path('path_measure_preparation/training_results_20261124/econ_imitation.json').read_text(encoding='utf8'))
print('STATUS',p['status'],'data',p['data_summary'],'rejected',len(p['rejected']))
for kind,run in p['training'].items():
    print('IMITATION',kind,json.dumps(run['test'],ensure_ascii=False))
for kind in ('econ','hier_bc','hier_diffusion'):
    games=[g for g in p['evaluation'] if g['kind']==kind]
    valid=[g for g in games if g['reason']=='natural_end' and g.get('rank') in range(1,9)]
    c=collections.Counter()
    for g in valid:c.update(g['target_action_counts'])
    n=sum(c.values())
    print('EVAL',kind,'n',len(valid),'rank',sum(g['rank'] for g in valid)/len(valid),
          'top4',sum(g['rank']<=4 for g in valid)/len(valid),'first',sum(g['rank']==1 for g in valid)/len(valid),
          'actions_pct',{k:round(v/n*100,2) for k,v in c.items()},
          'warning_games',sum(g.get('internal_warning_count',0)>0 for g in games),
          'warnings',[(g['seed'],g.get('internal_warning_examples')) for g in games if g.get('internal_warning_count',0)])
