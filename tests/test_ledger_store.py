import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'work/paper-trading'))
sys.path.insert(0, str(ROOT / 'comparison'))
import ledger_store as store
import paper
import runner
import test_paper as eth_fixture
import test_comparison as multi_fixture


def rows(path):
    db = sqlite3.connect(Path(path).as_uri() + '?mode=ro', uri=True)
    try:
        return (db.execute('SELECT * FROM state').fetchall(), db.execute('SELECT * FROM journal ORDER BY seq').fetchall())
    finally:
        db.close()


class LedgerStoreTests(unittest.TestCase):
    def test_both_engines_roundtrip_after_buy_and_sell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind in ['eth', 'comparison']:
                path = root / (kind + '.sqlite3')
                if kind == 'eth':
                    db = paper.connect(path)
                    state = paper.initial_state(eth_fixture.START)
                    paper.commit(db, state, [{'type': 'initialized', 'at': eth_fixture.START}])
                    for i in range(9):
                        state, events, _ = eth_fixture.step(state, i, entry=i < 5, leave=i >= 5, mid=100000 if i < 5 else 110000)
                        paper.commit(db, state, events)
                    self.assertGreater(state['accounts']['candidate_base']['orders'], 0)
                    expected = paper.verify(db)
                else:
                    db = runner.connect(path, True)
                    state = runner.initialize(db, multi_fixture.T, {})
                    for i in range(9):
                        obs = multi_fixture.observation(multi_fixture.T + i * multi_fixture.I, entry=i < 5, exit=i >= 5, price=1000 if i < 5 else 1100)
                        state, events, _ = runner.apply_tick(state, obs, multi_fixture.RULES)
                        runner.commit(db, state, {'type': 'tick', 'observation': obs, 'events': events})
                    self.assertGreater(state['multi']['base']['orders'], 0)
                    expected = runner.verify(db, multi_fixture.RULES, {})
                db.close()
                original_rows = rows(path)
                archive = root / kind
                exported = store.export_ledger(path, archive)
                restored = root / (kind + '-restored.sqlite3')
                self.assertEqual(exported, store.restore_ledger(archive, restored))
                self.assertEqual(rows(restored), original_rows)
                self.assertEqual(rows(path), original_rows)
                with sqlite3.connect(restored) as restored_db:
                    actual = paper.verify(restored_db) if kind == 'eth' else runner.verify(restored_db, multi_fixture.RULES, {})
                    self.assertEqual(expected, actual)
                with self.assertRaisesRegex(ValueError, 'replace'):
                    store.restore_ledger(archive, restored)
                self.assertEqual(exported, store.export_ledger(path, archive))

    def test_missing_corrupt_or_extra_record_cannot_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.sqlite3'
            db = paper.connect(source)
            paper.commit(db, paper.initial_state(eth_fixture.START), [{'type': 'initialized'}])
            db.close()
            archive = root / 'archive'
            store.export_ledger(source, archive)
            record = store.record_path(archive, 1)
            original = record.read_bytes()
            record.unlink()
            with self.assertRaises(ValueError):
                store.restore_ledger(archive, root / 'missing.sqlite3')
            record.write_bytes(b'corrupt gzip')
            with self.assertRaises(Exception):
                store.restore_ledger(archive, root / 'corrupt.sqlite3')
            record.write_bytes(original)
            store.record_path(archive, 2).write_bytes(original)
            with self.assertRaises(ValueError):
                store.restore_ledger(archive, root / 'extra.sqlite3')
            self.assertFalse((root / 'extra.sqlite3').exists())

    def test_refuse_stale_ledger_and_unjournalled_state_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.sqlite3'
            db = paper.connect(source)
            state = paper.initial_state(eth_fixture.START)
            paper.commit(db, state, [{'type': 'initialized'}])
            archive = root / 'archive'
            store.export_ledger(source, archive)
            checkpoint = (archive / 'checkpoint.json').read_bytes()
            state['accounts']['candidate_base']['cash'] += 10
            with db:
                db.execute('UPDATE state SET payload=?', (paper.encode(state),))
            with self.assertRaisesRegex(ValueError, 'without journal'):
                store.export_ledger(source, archive)
            self.assertEqual((archive / 'checkpoint.json').read_bytes(), checkpoint)
            db.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
