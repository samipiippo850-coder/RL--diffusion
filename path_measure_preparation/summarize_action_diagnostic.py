import collections
import json
from pathlib import Path

p=json.loads(Path('path_measure_preparation/training_results_20261124/action_diagnostic.json').read_text(encoding='utf8'))
print('status',p['status'])
for arm in ('econ','frozen','preference_grpo'):
    games=[g for g in p['games'] if g['arm']==arm]
    counts=collections.Counter()
    for g in games: counts.update(g.get('target_action_counts',{}))
    total=sum(counts.values())
    print(arm,'games',len(games),'reasons',dict(collections.Counter(g['reason'] for g in games)),
          'mean_rank',sum(g['rank'] for g in games)/len(games),'actions',total,
          'counts',dict(counts),'pct',{k:round(v/total*100,1) for k,v in counts.items()},
          'warnings',sum(g.get('internal_warning_count',0)>0 for g in games))
