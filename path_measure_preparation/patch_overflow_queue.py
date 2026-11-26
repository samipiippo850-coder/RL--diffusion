"""Keep overflow items in a FIFO until a visible item-bench slot is available.

This is an explicit experimental approximation: pending items are not encoded in
the original fixed-size observation space. Apply only to a simulator copy.
"""
from pathlib import Path
import sys

root = Path(sys.argv[1])
player_path = root / 'Simulator/game/player.py'
step_path = root / 'Simulator/game/step_function.py'
manager_path = root / 'Simulator/game/player_manager.py'
player = player_path.read_text(encoding='utf8')
step = step_path.read_text(encoding='utf8')
manager = manager_path.read_text(encoding='utf8')
if 'def flush_pending_items(' in player:
    raise SystemExit('patch already applied')

anchor = '        self.item_bench = [None for _ in range(config.ITEM_BENCH_SIZE)]'
assert player.count(anchor) == 1
player = player.replace(anchor, anchor + '\n        self.pending_items = []', 1)
anchor = '''        if self.item_bench_full(1):
            if DEBUG:
                print("Failed to add item to item bench")
            return False

        bench_loc = self.item_bench_vacancy()
        self.item_bench[bench_loc] = item
'''
replacement = '''        # Overflow is retained in FIFO order until a bench slot becomes free.
        self.flush_pending_items()
        if self.item_bench_full(1):
            self.pending_items.append(item)
        else:
            self.item_bench[self.item_bench_vacancy()] = item
        return True

    def flush_pending_items(self):
        """Move pending items into free visible slots; return the number moved."""
        moved = 0
        while self.pending_items and not self.item_bench_full(1):
            self.item_bench[self.item_bench_vacancy()] = self.pending_items.pop(0)
            moved += 1
        return moved
'''
assert player.count(anchor) == 1
player = player.replace(anchor, replacement, 1)
anchor = '        self.item_bench = [None for _ in range(10)]\n        self.gold = 0'
assert player.count(anchor) == 1
player = player.replace(anchor, '        self.item_bench = [None for _ in range(10)]\n        self.pending_items = []\n        self.gold = 0', 1)
anchor = '        player.actions_remaining -= 1\n'
assert step.count(anchor) == 1
step = step.replace(anchor, '        restored_items = player.flush_pending_items()\n' + anchor, 1)
# The action-driven observation/mask update runs after the flush. A slot can
# normally become free only via an item action; refresh the corresponding
# vectors and mask explicitly for all action types to cover other edge cases.
anchor = '        self.player_manager.action_handlers[player_id].update_action_mask(action)\n'
assert step.count(anchor) == 1
step = step.replace(anchor, anchor + '''        if restored_items and action_type != 6:
            observation = self.player_manager.observation_states[player_id]
            handler = self.player_manager.action_handlers[player_id]
            observation.item_bench_vector = observation.create_item_bench_vector(player)
            handler.item_mask = handler.create_item_action_mask(player)
''', 1)
anchor = '                self.observation_states[player].update_game_round()'
assert manager.count(anchor) == 1
manager = manager.replace(anchor, '                self.player_states[player].flush_pending_items()\n' + anchor, 1)
for path, source in ((player_path, player), (step_path, step), (manager_path, manager)):
    compile(source, str(path), 'exec')
for path, source in ((player_path, player), (step_path, step), (manager_path, manager)):
    path.write_text(source, encoding='utf8')
print('overflow queue patch applied:', root)
