import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'comparison'))
import cloud_runner as cloud
import runner
import test_comparison as fixture


class CloudRunnerTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(cloud.os.environ, {'GITHUB_RUN_ID': ''})
        environment.start()
        self.addCleanup(environment.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.identity = 'new-20261019-test-only'
        self.directory = self.root / 'experiments' / self.identity
        self.hashes = json.loads((ROOT / 'frozen_source_sha256.json').read_text())
        for name in list(self.hashes) + ['frozen_source_sha256.json']:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        self.settings = {
            'format': 'paper-experiment-control-v1', 'phase': 'approved_new',
            'experiment_id': self.identity, 'approved_at': datetime.fromtimestamp(fixture.T / 1000, timezone.utc).isoformat(),
            'frozen_source_sha256': self.hashes,
        }
        cloud.save(self.root / 'control.json', self.settings)
        for k in range(2):
            with patch.object(cloud.probe.time, 'time', return_value=1000000 + k * 900):
                cloud.probe.run(self.root / 'validation/history.json', 'schedule', str(k), lambda: {'synthetic_test_only': True})

    def no_fetch(self, cache):
        self.fail('Must not fetch')

    def begin(self):
        return cloud.start(self.root, self.identity, lambda cache: fixture.observation(fixture.T), lambda: fixture.T + 2000)

    def step(self, k, **kwargs):
        obs = fixture.observation(fixture.T + k * fixture.I, **kwargs)
        return cloud.tick(self.root, lambda cache: obs, lambda: obs['observed_ms'] + 1000)

    def test_no_start_without_approval_and_scheduled_checks(self):
        self.settings['phase'] = 'awaiting_new_experiment_approval'
        cloud.save(self.root / 'control.json', self.settings)
        with self.assertRaisesRegex(ValueError, 'not authorized'):
            cloud.start(self.root, self.identity, self.no_fetch)
        self.settings['phase'] = 'approved_new'
        cloud.save(self.root / 'control.json', self.settings)
        (self.root / 'validation/history.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Two distinct'):
            cloud.start(self.root, self.identity, self.no_fetch)
        self.assertFalse(self.directory.exists())

    def test_start_is_new_simultaneous_and_never_repeated(self):
        result = self.begin()
        for a in result['summary']['accounts'].values():
            self.assertEqual(a['equity_yen'], 10000)
            self.assertEqual(a['orders'], 0)
        self.assertFalse(result['summary']['previous_experiment_continued'])
        self.assertEqual(cloud.verify(self.root)['records'], 1)
        with self.assertRaisesRegex(ValueError, 'not authorized'):
            self.begin()
        # Even an accidentally re-approved control cannot overwrite an existing directory.
        cloud.save(self.root / 'control.json', self.settings)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.begin()

    def test_disposable_runtimes_match_continuous_engine_and_skip_duplicates(self):
        self.begin()
        hashes, rules = cloud.sources(self.root)
        expected = runner.initial(fixture.T + 2000, hashes)
        for k in range(9):
            obs = fixture.observation(fixture.T + k * fixture.I, entry=k < 5, exit=k >= 5, price=1000 if k < 5 else 1100)
            expected, _, _ = runner.apply_tick(expected, obs, rules)
            result = self.step(k, entry=k < 5, exit=k >= 5, price=1000 if k < 5 else 1100)
            self.assertEqual(result['summary']['accounts'], runner.summary(expected)['accounts'])
        self.assertGreater(result['summary']['accounts']['multi_base']['round_trips'], 0)
        before = (self.directory / 'ledger/checkpoint.json').read_bytes()
        same = cloud.tick(self.root, self.no_fetch, lambda: fixture.T + 8 * fixture.I + 2000)
        self.assertTrue(same['duplicate_slot'])
        self.assertEqual(before, (self.directory / 'ledger/checkpoint.json').read_bytes())
        self.assertTrue(cloud.verify(self.root)['ok'])
        self.assertFalse(list(self.root.rglob('*.sqlite3')))

    def test_lost_or_corrupt_archive_never_resets_or_fetches(self):
        self.begin()
        path = self.directory / 'ledger/checkpoint.json'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            cloud.tick(self.root, self.no_fetch)
        self.assertFalse(path.exists())
        path.write_bytes(original)
        state = json.loads(original)
        state['state_payload'] += 'corruption'
        cloud.save(path, state)
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            cloud.tick(self.root, self.no_fetch)

    def test_timeout_notification_is_once_then_recovery_and_denial_halts(self):
        self.begin()
        self.step(0)
        def timeout(cache):
            raise TimeoutError('timeout')
        a = cloud.tick(self.root, timeout, lambda: fixture.T + fixture.I + 2000)
        b = cloud.tick(self.root, timeout, lambda: fixture.T + 2 * fixture.I + 2000)
        self.assertEqual(len(a['notices']), 1)
        self.assertEqual(b['notices'], [])
        self.assertFalse(b['stop'])
        restored = self.step(3)
        self.assertIn('data_recovered', [n['kind'] for n in restored['notices']])
        self.assertEqual(restored['summary']['missing_slots'], 2)
        self.assertEqual(restored['summary']['accounts']['eth_base']['orders'], 0)
        def denied(cache):
            raise RuntimeError('Network access was blocked by policy')
        result = cloud.tick(self.root, denied, lambda: fixture.T + 4 * fixture.I + 2000)
        self.assertTrue(result['stop'])
        self.assertTrue(cloud.verify(self.root)['ok'])
        with self.assertRaisesRegex(ValueError, 'halted'):
            cloud.tick(self.root, self.no_fetch)

    def test_wrong_identity_stale_preflight_and_source_change_fail_before_start(self):
        with self.assertRaisesRegex(ValueError, 'does not match'):
            cloud.start(self.root, 'different-experiment', self.no_fetch)
        with self.assertRaisesRegex(ValueError, 'no longer fresh'):
            cloud.start(self.root, self.identity, lambda c: fixture.observation(fixture.T), lambda: fixture.T + fixture.I)
        path = self.root / 'comparison/core.py'
        path.write_text(path.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'source hash mismatch'):
            cloud.start(self.root, self.identity, self.no_fetch)
        self.assertFalse(self.directory.exists())

    def test_same_job_rerun_cannot_fetch_or_append_again(self):
        with patch.dict(cloud.os.environ, {'GITHUB_RUN_ID': 'first'}):
            self.begin()
        with patch.dict(cloud.os.environ, {'GITHUB_RUN_ID': 'second'}):
            self.step(0)
            before = (self.directory / 'ledger/checkpoint.json').read_bytes()
            again = cloud.tick(self.root, self.no_fetch, lambda: fixture.T + fixture.I)
            self.assertTrue(again['duplicate_run'])
            self.assertEqual(before, (self.directory / 'ledger/checkpoint.json').read_bytes())
    def test_failure_to_save_never_reports_a_complete_bundle(self):
        self.begin()
        checkpoint = (self.directory / 'ledger/checkpoint.json').read_bytes()
        with patch.object(cloud.ledger_store, 'export_ledger', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.step(0)
        self.assertEqual(checkpoint, (self.directory / 'ledger/checkpoint.json').read_bytes())
        self.assertTrue(cloud.verify(self.root)['ok'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
