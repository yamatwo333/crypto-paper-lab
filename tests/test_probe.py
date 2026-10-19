import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[1] / 'scripts/probe.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ProbeTests(unittest.TestCase):
    def test_persistence_and_two_distinct_scheduled_slots(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'history.json'
            for i, event in enumerate(['push', 'schedule', 'schedule']):
                with patch.object(p.time, 'time', return_value=1000000 + i * 900):
                    result = p.run(path, event, str(i), lambda: {'synthetic': True})
                self.assertEqual(result['previous_records_loaded'], i)
            self.assertEqual(result['reason'], 'two_scheduled_checks_complete')
            result = p.run(path, 'schedule', 'next', lambda: self.fail('Must not fetch after completion'))
            self.assertTrue(result['no_request'])
            self.assertEqual(len(p.load(path)), 3)

    def test_denial_prevents_further_fetches(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'history.json'
            def denied():
                raise RuntimeError('Network access was blocked by policy')
            result = p.run(path, 'push', 'a', denied)
            self.assertEqual(result['reason'], 'explicit_denial')
            self.assertTrue(p.run(path, 'schedule', 'b', lambda: self.fail('No retry'))['no_request'])

    def test_integrity_error_prevents_network(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'history.json'
            p.run(path, 'push', '1', lambda: {})
            rows = json.loads(path.read_text())
            rows[0]['payload']['success'] = False
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                p.run(path, 'schedule', '2', lambda: self.fail('Do not fetch with corrupt records'))

    def test_duplicate_job_and_attempt_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'history.json'
            p.run(path, 'push', '1', lambda: {})
            self.assertTrue(p.run(path, 'push', '1', lambda: self.fail('Duplicate'))['duplicate_run'])
            for i in range(2, 6):
                p.run(path, 'push', str(i), lambda: {})
            self.assertEqual(p.stop_reason(p.load(path)), 'attempt_limit_reached')


if __name__ == '__main__':
    unittest.main(verbosity=2)

