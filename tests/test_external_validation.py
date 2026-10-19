import copy
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import probe
import validation_gate as gate
import launch_gate


class ExternalValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.start = 1792409520  # synthetic timestamps; no live fetches
        self.settings = {
            'format': 'paper-experiment-control-v1', 'phase': 'approved_new',
            'experiment_id': 'new-20261019-test-only',
            'approved_at': '2026-10-19T17:06:27+09:00',
            'frozen_source_sha256': {'synthetic': 'hash'},
        }
        (self.root / 'control.json').write_text(json.dumps(self.settings))
        (self.root / 'frozen_source_sha256.json').write_text(json.dumps({'synthetic': 'hash'}))
        self.history = self.root / 'validation/history.json'
        for i in range(5):
            with patch.object(probe.time, 'time', return_value=self.start + (i - 3) * 900 + 10):
                probe.run(self.history, 'workflow_dispatch', str(i + 1),
                          lambda: {p: {'synthetic_test_only': True} for p in probe.PAIRS})
        self.rows = probe.load(self.history)
        self.data = {
            'format': 'external-scheduled-storage-proof-v1',
            'provider': 'cron-job.org', 'job_id': 8676656,
            'repository': 'yamatwo333/crypto-paper-lab',
            'workflow': '.github/workflows/probe.yml',
            'experiment_id': self.settings['experiment_id'],
            'verified_at': self.iso(self.start + 960), 'checks': [],
        }
        for i, row in enumerate(self.rows[3:]):
            scheduled = self.start + i * 900
            self.data['checks'].append({
                'scheduled_at': self.iso(scheduled), 'executed_at': self.iso(scheduled + 5),
                'github_completed_at': self.iso(scheduled + 20),
                'github_run_id': row['payload']['run_id'], 'github_job_id': 100 + i,
                'cron_history_row': f'8676656-synthetic-{i}',
                'history_hash': row['hash'], 'http_status': 200,
                'github_event': 'workflow_dispatch', 'github_conclusion': 'success',
            })

    @staticmethod
    def iso(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    def save(self, data=None):
        data = self.data if data is None else data
        (self.root / 'validation/external-receipts.json').write_text(
            json.dumps({'payload': data, 'sha256': probe.digest(data)}))

    def test_dispatch_success_alone_cannot_unlock_start(self):
        self.assertEqual(probe.stop_reason(self.rows), 'attempt_limit_reached')
        self.assertFalse(gate.ready(self.root))
        self.assertFalse(launch_gate.decide(self.root, 'push')['run'])

    def test_reviewed_external_evidence_unlocks_without_relabeling_or_rewriting(self):
        before = self.history.read_bytes()
        self.save()
        self.assertTrue(gate.ready(self.root))
        self.assertEqual(launch_gate.decide(self.root, 'push')['operation'], 'start')
        self.assertFalse(launch_gate.decide(self.root, 'workflow_dispatch')['run'])
        self.assertEqual(self.history.read_bytes(), before)
        self.assertFalse((self.root / 'experiments').exists())

    def test_wrong_run_hash_workflow_experiment_or_status_is_rejected(self):
        for field, value in [('github_run_id', 'unknown'), ('history_hash', 'wrong'),
                             ('github_conclusion', 'failure'), ('http_status', 403),
                             ('github_event', 'schedule')]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data['checks'][1][field] = value
                self.save(data)
                with self.assertRaises(ValueError):
                    gate.ready(self.root)
        for field, value in [('workflow', '.github/workflows/scheduler-diagnostic.yml'),
                             ('experiment_id', 'new-20261019-other'), ('job_id', 123)]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data[field] = value
                self.save(data)
                with self.assertRaises(ValueError):
                    gate.ready(self.root)

    def test_single_duplicate_and_nonconsecutive_checks_are_rejected(self):
        variants = [[self.data['checks'][0]], [self.data['checks'][0]] * 2]
        for checks in variants:
            data = copy.deepcopy(self.data)
            data['checks'] = checks
            self.save(data)
            with self.assertRaises(ValueError):
                gate.ready(self.root)
        data = copy.deepcopy(self.data)
        data['checks'][0]['scheduled_at'] = self.iso(self.start - 60)
        self.save(data)
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            gate.ready(self.root)

    def test_preapproval_timezone_and_future_completion_are_rejected(self):
        for field, value in [('executed_at', '2026-10-19T20:00:00'),
                             ('scheduled_at', '2026-10-18T20:00:00+09:00'),
                             ('github_completed_at', self.iso(self.start + 9999))]:
            with self.subTest(field=field):
                data = copy.deepcopy(self.data)
                data['checks'][0][field] = value
                self.save(data)
                with self.assertRaises(ValueError):
                    gate.ready(self.root)

    def test_corrupt_receipt_or_probe_history_fails_closed(self):
        self.save()
        path = self.root / 'validation/external-receipts.json'
        receipt = json.loads(path.read_text())
        receipt['payload']['job_id'] += 1
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            gate.ready(self.root)
        self.save()
        rows = json.loads(self.history.read_text())
        rows[-1]['payload']['success'] = False
        self.history.write_text(json.dumps(rows))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            gate.ready(self.root)

    def test_saved_failure_blocks_even_with_receipts(self):
        rows = json.loads(self.history.read_text())
        rows[-1]['payload']['success'] = False
        rows[-1]['hash'] = probe.digest(rows[-1]['payload'])
        self.history.write_text(json.dumps(rows))
        self.save()
        with self.assertRaisesRegex(ValueError, 'Validation failed'):
            gate.ready(self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
