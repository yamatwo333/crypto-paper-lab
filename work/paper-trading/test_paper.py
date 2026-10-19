import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper as p

START = 1792378800000 // p.INTERVAL * p.INTERVAL


def signal(slot, entry=True, leave=False):
    return {'slot': slot, 'p': 100000., 'mom30': .05, 'd_atr': 3000., 'h4_atr': 1000.,
            'h1_atr': 500., 'entry': entry, 'exit': leave, 'gate': True, 'daily_on': entry,
            'boundary_band': .015}


def book(slot, mid=100000., offset=30000):
    return {'observed_ms': slot + offset, 'timestamp': slot + offset - 100,
            'ask': mid + 1, 'bid': mid - 1, 'mid': mid, 'ask_qty': 100., 'bid_qty': 100.,
            'spread_fraction': 2 / mid, 'raw_depth': {}, 'raw_circuit': {}}


def step(state, i, entry=True, leave=False, mid=100000.):
    slot = START + i * p.INTERVAL
    s = signal(slot, entry, leave)
    s['p'] = mid
    return p.advance(state, s, book(slot, mid))


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.state = p.initial_state(START)

    def test_four_observations_then_future_quote(self):
        s = self.state
        for i in range(4):
            s, events, _ = step(s, i)
            self.assertEqual(s['accounts']['candidate_base']['orders'], 0)
        self.assertIsNotNone(s['accounts']['candidate_base']['pending'])
        s, events, _ = step(s, 4, mid=105000.)
        fills = [e for e in events if e['type'] == 'paper_fill' and e['account'] == 'candidate_base']
        self.assertEqual(len(fills), 1)
        self.assertGreater(fills[0]['price'], 105000)
        self.assertEqual(fills[0]['at'], START + 4 * p.INTERVAL + 30000)
        self.assertLessEqual(fills[0]['qty'] * fills[0]['price'] + fills[0]['fee'], 2500)

    def test_duplicate_does_not_mutate(self):
        s, _, _ = step(self.state, 0)
        before = copy.deepcopy(s)
        s, events, result = step(s, 0)
        self.assertEqual(s, before)
        self.assertFalse(events)
        self.assertTrue(result['duplicate_slot'])

    def test_gap_resets_confirmation_and_cancels_entry(self):
        s = self.state
        for i in range(4):
            s, _, _ = step(s, i)
        s, events, _ = step(s, 6)
        self.assertEqual(s['accounts']['candidate_base']['orders'], 0)
        self.assertEqual(s['accounts']['candidate_base']['confirmation'], 1)
        self.assertEqual(s['missing_slots'], 2)
        self.assertTrue(any(e['type'] == 'cancel' and e['reason'] == 'monitoring_gap' for e in events))

    def test_delayed_exit_uses_current_price(self):
        s = self.state
        for i in range(5):
            s, _, _ = step(s, i)
        s, _, _ = step(s, 5, entry=False, leave=True)
        s, events, _ = step(s, 10, entry=False, mid=80000.)
        fills = [e for e in events if e['type'] == 'paper_fill' and e['account'] == 'candidate_base']
        self.assertEqual(fills[0]['side'], 'sell')
        self.assertLess(fills[0]['price'], 80000)
        self.assertEqual(s['accounts']['candidate_base']['qty'], 0)
        self.assertEqual(s['accounts']['candidate_base']['round_trips'], 1)
        self.assertLess(s['accounts']['candidate_base']['cash'], 10000)

    def test_compound_sizing_and_min_lot(self):
        a = p.account()
        a['cash'] = 12000.
        p.queue(a, 'buy', 'test', START, START + 30000, 1000)
        events = []
        p.execute_pending('candidate_base', a, book(START + p.INTERVAL), START + p.INTERVAL, events)
        spent = 12000 - a['cash']
        self.assertTrue(2990 < spent <= 3000)
        b = p.account()
        b['cash'] = 1.
        p.queue(b, 'buy', 'test', START, START + 30000, 1000)
        p.execute_pending('candidate_base', b, book(START + p.INTERVAL), START + p.INTERVAL, [])
        self.assertEqual(b['orders'], 0)
        self.assertEqual(b['cash'], 1)

    def test_wide_spread_defers_entry_not_exit(self):
        a = p.account()
        p.queue(a, 'buy', 'test', START, START + 30000, 1000)
        q = book(START + p.INTERVAL)
        q['spread_fraction'] = .01
        p.execute_pending('candidate_base', a, q, START + p.INTERVAL, [])
        self.assertEqual(a['orders'], 0)
        self.assertIsNotNone(a['pending'])
        q['spread_fraction'] = .00002
        p.execute_pending('candidate_base', a, q, START + p.INTERVAL, [])
        p.queue(a, 'sell', 'test', START + p.INTERVAL, q['observed_ms'], 0)
        q = book(START + 2 * p.INTERVAL)
        q['spread_fraction'] = .01
        p.execute_pending('candidate_base', a, q, START + 2 * p.INTERVAL, [])
        self.assertEqual(a['qty'], 0)

    def test_capacity_and_stale_book(self):
        a = p.account()
        p.queue(a, 'buy', 'test', START, START + 30000, 1000)
        q = book(START + p.INTERVAL)
        q['ask_qty'] = .0001
        p.execute_pending('candidate_base', a, q, START + p.INTERVAL, [])
        self.assertEqual(a['orders'], 0)
        responses = [({'data': {'mode': 'NONE'}}, b''),
                     ({'data': {'asks': [['101', '1']], 'bids': [['100', '1']], 'timestamp': START - 100000}}, b'')]
        with patch.object(p, 'get_public', side_effect=responses), patch.object(p, 'now_ms', return_value=START):
            with self.assertRaisesRegex(RuntimeError, 'Stale'):
                p.collect_book()

    def test_journal_restart_replay_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'test.sqlite3'
            db = p.connect(path)
            s = self.state
            p.commit(db, s, [{'type': 'initialized', 'at': START}])
            for i in range(7):
                s, events, _ = step(s, i, entry=i < 5, leave=i == 5)
                p.commit(db, s, events)
            db.close()
            db = p.connect(path)
            self.assertEqual(p.load_state(db), s)
            self.assertTrue(p.verify(db)['verified'])
            db.execute("UPDATE journal SET payload='{}' WHERE seq=2")
            db.commit()
            with self.assertRaises(AssertionError):
                p.verify(db)
            db.close()

    def test_daily_expired_entry_can_rearm_and_no_backfill(self):
        s, _, _ = step(self.state, 0)
        s, events, _ = step(s, 3)
        a = s['accounts']['daily_base']
        self.assertEqual(a['orders'], 0)
        self.assertIsNotNone(a['pending'])
        self.assertEqual(a['pending']['decision_slot'], START + 3 * p.INTERVAL)

    def test_committed_state_survives_report_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            db = p.connect(Path(temp) / 'test.sqlite3')
            p.commit(db, self.state, [{'type': 'initialized'}])
            with patch.object(p, 'now_ms', return_value=START), patch.object(p, 'check_frozen'), \
                 patch.object(p, 'collect_signal', return_value=(signal(START), [])), \
                 patch.object(p, 'collect_book', return_value=book(START)), \
                 patch.object(p, 'render', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    p.tick(db, {})
            self.assertEqual(p.load_state(db)['successful_slots'], 1)
            self.assertTrue(p.verify(db)['verified'])
            db.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
