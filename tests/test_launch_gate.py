import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import launch_gate as gate


class LaunchGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = 'new-20261019-test-only'
        self.settings = {'format': 'paper-experiment-control-v1', 'phase': 'approved_new',
                         'experiment_id': self.identity, 'approved_at': '2026-10-19T17:06:27+09:00',
                         'frozen_source_sha256': {'test_only': 'hash'}}
        (self.root / 'frozen_source_sha256.json').write_text(json.dumps(self.settings['frozen_source_sha256']))
        self.save()

    def save(self):
        (self.root / 'control.json').write_text(json.dumps(self.settings))

    def record(self, number, event='schedule', fail=False):
        def observe():
            if fail:
                raise RuntimeError('Network blocked by policy')
            return {'synthetic': True}
        with patch.object(gate.probe.time, 'time', return_value=1000000 + number * 900):
            gate.probe.run(self.root / 'validation/history.json', event, str(number), observe)

    def test_push_and_one_schedule_cannot_start_two_distinct_schedules_can(self):
        self.record(0, 'push')
        self.record(1)
        self.assertFalse(gate.decide(self.root, 'workflow_run')['run'])
        self.record(2)
        result = gate.decide(self.root, 'workflow_run')
        self.assertEqual(result['operation'], 'start')
        self.assertEqual(result['confirmation'], self.identity)
        self.assertFalse((self.root / 'experiments').exists())

    def test_approval_and_manual_confirmation_are_required(self):
        self.settings['phase'] = 'awaiting_new_experiment_approval'
        self.save()
        self.assertFalse(gate.decide(self.root, 'schedule')['run'])
        self.settings['phase'] = 'approved_new'
        self.save()
        self.record(1)
        self.record(2)
        self.assertFalse(gate.decide(self.root, 'workflow_dispatch')['run'])
        self.assertFalse(gate.decide(self.root, 'workflow_dispatch', 'start', 'wrong')['run'])
        self.assertTrue(gate.decide(self.root, 'workflow_dispatch', 'start', self.identity)['run'])

    def test_running_state_uses_tick_and_never_starts_again(self):
        self.settings['phase'] = 'running'
        self.save()
        self.assertEqual(gate.decide(self.root, 'schedule')['operation'], 'tick')
        self.assertFalse(gate.decide(self.root, 'workflow_run')['run'])
        self.assertFalse(gate.decide(self.root, 'push')['run'])
        with self.assertRaisesRegex(ValueError, 'Already started'):
            gate.decide(self.root, 'workflow_dispatch', 'start', self.identity)

    def test_denial_partial_state_and_changed_source_cannot_launch(self):
        self.record(1, fail=True)
        with self.assertRaisesRegex(ValueError, 'Validation failed'):
            gate.decide(self.root, 'schedule')
        (self.root / 'validation/history.json').unlink()
        directory = self.root / 'experiments' / self.identity
        directory.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'existing experiment'):
            gate.decide(self.root, 'schedule')
        self.settings['frozen_source_sha256'] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, 'Source differs'):
            gate.decide(self.root, 'schedule')


if __name__ == '__main__':
    unittest.main(verbosity=2)
