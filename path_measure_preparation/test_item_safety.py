"""Regression tests for item preservation during failed/successful returns."""
import unittest
from types import SimpleNamespace
from Simulator.game.player import Player
from Simulator.game.step_function import Step_Function


class ItemSafety(unittest.TestCase):
    def make_player(self, items, spaces):
        p = Player.__new__(Player)
        p.item_bench = ['spatula'] * (10-spaces) + [None] * spaces
        p.pending_items = []
        c = SimpleNamespace(items=list(items), x=0, y=0, origin=[])
        p.bench = [c] + [None]*8
        p.thieves_gloves_loc = []
        p.print = lambda *args: None
        return p, c

    def test_full_bench_leaves_items_untouched(self):
        p, c = self.make_player(['chain_vest'], 0)
        self.assertFalse(p.return_item_from_bench(0))
        self.assertEqual(c.items, ['chain_vest'])
        self.assertFalse(p.return_item(c))
        self.assertEqual(c.items, ['chain_vest'])

    def test_successful_return_preserves_item(self):
        p, c = self.make_player(['chain_vest'], 1)
        self.assertTrue(p.return_item_from_bench(0))
        self.assertEqual(c.items, [])
        self.assertEqual(p.item_bench[-1], 'chain_vest')

    def test_gloves_do_not_mutate_when_full(self):
        p, c = self.make_player(['thieves_gloves', 'chain_vest'], 0)
        p.thieves_gloves_loc = [[0, -1]]
        self.assertFalse(p.return_item_from_bench(0))
        self.assertEqual(c.items, ['thieves_gloves', 'chain_vest'])
        self.assertEqual(p.thieves_gloves_loc, [[0, -1]])

    def test_sell_preflight_preserves_catalog(self):
        p, c = self.make_player(['chain_vest'], 0)
        c.name, c.stars, c.target_dummy = 'test', 1, False
        p.triple_catalog = [{'name':'test', 'level':1, 'num':1}]
        p.reward, p.mistake_reward = 0, 0
        self.assertFalse(p.sell_from_bench(0))
        self.assertEqual(p.triple_catalog[0]['num'], 1)
        self.assertEqual(c.items, ['chain_vest'])

    def test_overflow_preserves_fifo_and_restores_slots(self):
        p, _ = self.make_player([], 0)
        self.assertTrue(p.add_to_item_bench('chain_vest'))
        self.assertTrue(p.add_to_item_bench('recurve_bow'))
        self.assertEqual(p.pending_items, ['chain_vest', 'recurve_bow'])
        p.item_bench[3] = None
        self.assertEqual(p.flush_pending_items(), 1)
        self.assertEqual(p.item_bench[3], 'chain_vest')
        self.assertEqual(p.pending_items, ['recurve_bow'])
        p.item_bench[2] = None
        self.assertEqual(p.flush_pending_items(), 1)
        self.assertEqual(p.item_bench[2], 'recurve_bow')
        self.assertEqual(p.pending_items, [])

    def test_new_item_does_not_jump_pending_queue(self):
        p, _ = self.make_player([], 1)
        p.pending_items = ['chain_vest']
        self.assertTrue(p.add_to_item_bench('recurve_bow'))
        self.assertEqual(p.item_bench[-1], 'chain_vest')
        self.assertEqual(p.pending_items, ['recurve_bow'])

    def test_item_action_flushes_before_observation_and_mask_update(self):
        p, _ = self.make_player([], 0)
        p.pending_items = ['chain_vest']
        p.actions_remaining = 1
        p.move_item_action = lambda item_slot, target: p.item_bench.__setitem__(item_slot, None)
        seen = []
        observation = SimpleNamespace(update_observation=lambda action: seen.append(('obs', p.item_bench[0])))
        handler = SimpleNamespace(update_action_mask=lambda action: seen.append(('mask', p.item_bench[0])))
        manager = SimpleNamespace(player_states={'p':p}, observation_states={'p':observation},
                                  action_handlers={'p':handler})
        Step_Function(manager).perform_action('p', [6, 0, 0])
        self.assertEqual(seen, [('obs', 'chain_vest'), ('mask', 'chain_vest')])
        self.assertEqual(p.pending_items, [])


if __name__ == '__main__':
    unittest.main()
