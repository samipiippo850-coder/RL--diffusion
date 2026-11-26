"""Apply small, fail-closed item-return fixes to the pinned simulator copy."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) / 'Simulator/game/player.py'
source = path.read_text(encoding='utf8')
if 'def _can_return_items_safely(' in source:
    raise SystemExit('patch already applied')

anchor = '    def return_item_from_bench(self, x) -> bool:'
helper = '''    def _can_return_items_safely(self, champ):
        """Preflight before mutating a champion, catalog or item bench."""
        if champ is None:
            return False
        items = champ.items
        needed = 1 if items and items[0] == 'thieves_gloves' else len(items)
        return not self.item_bench_full(needed)

'''
assert source.count(anchor) == 1
source = source.replace(anchor, helper + anchor, 1)
anchor = '        if self.bench[x]:\n            # skip if there are no items'
replacement = '''        if self.bench[x]:
            if not self._can_return_items_safely(self.bench[x]):
                return False
            # skip if there are no items'''
assert source.count(anchor) == 1
source = source.replace(anchor, replacement, 1)
source = source.replace('                    self.thieves_gloves_loc.remove([x, -1])', "                    if [x, -1] in self.thieves_gloves_loc:\n                        self.thieves_gloves_loc.remove([x, -1])", 1)
source = source.replace('self.item_bench_cvacancy()', 'self.item_bench_vacancy()')
anchor = '        if a_champion:\n            # skip if there are no items'
replacement = '''        if a_champion:
            if not self._can_return_items_safely(a_champion):
                return False
            # skip if there are no items'''
assert source.count(anchor) == 1
source = source.replace(anchor, replacement, 1)
anchor = '        if not (self.remove_triple_catalog(s_champion, golden=golden) and self.return_item(s_champion) and not\n                s_champion.target_dummy):'
replacement = '        if not (not s_champion.target_dummy and self._can_return_items_safely(s_champion) and\n                self.remove_triple_catalog(s_champion, golden=golden) and self.return_item(s_champion)):'
assert source.count(anchor) == 1
source = source.replace(anchor, replacement, 1)
anchor = '            if not (self.remove_triple_catalog(self.bench[location], golden=golden) and\n                    self.return_item_from_bench(location)):'
replacement = '            if not (self._can_return_items_safely(self.bench[location]) and\n                    self.remove_triple_catalog(self.bench[location], golden=golden) and\n                    self.return_item_from_bench(location)):'
assert source.count(anchor) == 1
source = source.replace(anchor, replacement, 1)
compile(source, str(path), 'exec')
path.write_text(source, encoding='utf8')
print('patched', path)
