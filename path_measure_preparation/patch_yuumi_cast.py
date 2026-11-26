"""Stop Yuumi's cast if pre-cast ionic-spark damage kills the caster."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / 'Simulator/battle/ability.py'
source = path.read_text(encoding='utf8')
anchor = '''def yuumi(champion):
    default_ability_calls(champion)
    champion.add_que('change_stat', -1, None, 'ability_active', True)'''
replacement = '''def yuumi(champion):
    default_ability_calls(champion)
    # Ionic Spark can kill Yuumi during default_ability_calls. Dead casters
    # must not enqueue the remainder of the heal/buff ability.
    if champion.health <= 0:
        return
    champion.add_que('change_stat', -1, None, 'ability_active', True)'''
if replacement in source:
    raise SystemExit('patch already applied')
assert source.count(anchor) == 1
source = source.replace(anchor, replacement, 1)
compile(source, str(path), 'exec')
path.write_text(source, encoding='utf8')
print('patched', path)
