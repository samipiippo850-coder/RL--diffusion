"""Regression checks for a pre-cast death and a genuine missing-team warning."""
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from Simulator.battle import ability


class YuumiCast(unittest.TestCase):
    def test_dead_from_precall_damage_does_not_enqueue_ability(self):
        caster = SimpleNamespace(health=10, add_que=lambda *args: self.fail('dead caster queued ability'))
        with patch.object(ability, 'default_ability_calls', side_effect=lambda c: setattr(c, 'health', 0)):
            ability.yuumi(caster)

    def test_alive_empty_team_remains_visible_as_warning(self):
        caster = SimpleNamespace(health=10, name='yuumi', add_que=lambda *args: None,
                                 own_team=lambda: [])
        stream = io.StringIO()
        with patch.object(ability, 'default_ability_calls'), contextlib.redirect_stdout(stream):
            ability.yuumi(caster)
        self.assertIn('some dead yummi allies', stream.getvalue())


if __name__ == '__main__':
    unittest.main()
