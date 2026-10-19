"""Forward-only ETH paper accounts. Public GET requests only; no real order API.

One invocation observes one current 15-minute slot. Missing slots are never replayed.
The SQLite transaction commits account state and a chained journal together.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
OUTPUT = PROJECT / 'outputs'
INTERVAL = 900_000
DAY = 86_400_000
BASE = 'https://public.bitbank.cc/eth_jpy/'
ACCOUNT_INFO = {
    'candidate_base': ('ETH 上昇追随・通常コスト', 'candidate', .0015, .0005),
    'candidate_stress': ('ETH 上昇追随・高コスト', 'candidate', .0015, .0035),
    'daily_base': ('ETH 日次ルール・通常コスト', 'daily', .0015, .0005),
    'daily_stress': ('ETH 日次ルール・高コスト', 'daily', .0015, .0035),
}


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat()


def jst(ms):
    return datetime.fromtimestamp(ms / 1000, timezone(timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S JST')


def now_ms():
    return int(time.time() * 1000)


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def frozen_module():
    spec = importlib.util.spec_from_file_location('paper_frozen_engine', ROOT / 'frozen/engine.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_public(endpoint):
    # The only network path in this program. No credentials, private API, POST, or order route.
    if not (endpoint in ['depth', 'circuit_break_info'] or endpoint.startswith('candlestick/15min/')):
        raise ValueError('Unsupported public endpoint')
    url = BASE + endpoint
    request = urllib.request.Request(url, headers={'User-Agent': 'PaperResearch/1.0'}, method='GET')
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read()
    obj = json.loads(raw)
    if obj.get('success') != 1:
        raise RuntimeError('Public data response failed: ' + str(obj))
    return obj, raw


def account():
    return {'cash': 10000., 'qty': 0., 'fees': 0., 'basis': 0., 'realized_pnl': 0.,
            'entry_ms': None, 'entry_price': None, 'highwater': None,
            'stop_distance': None, 'trail_distance': None, 'last_exit_slot': None,
            'confirmation': 0, 'pending': None, 'orders': 0, 'round_trips': 0,
            'daily_day': None, 'daily_target': None, 'daily_week': None,
            'peak_liquidation': 10000., 'max_drawdown': 0., 'liquidation_value': 10000.,
            'marked_value': 10000., 'net_return': 0.}


def initial_state(start):
    return {'started_ms': start, 'last_slot': None, 'last_observed_ms': None,
            'successful_slots': 0, 'missing_slots': 0, 'failures': 0, 'health': 'starting',
            'last_error': None, 'last_signal': None, 'last_book': None,
            'reported_milestones': [], 'accounts': {name: account() for name in ACCOUNT_INFO}}


def connect(path):
    db = sqlite3.connect(path, timeout=30)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS journal (seq INTEGER PRIMARY KEY, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, record_hash TEXT NOT NULL)')
    return db


def load_state(db):
    row = db.execute('SELECT payload FROM state WHERE id=1').fetchone()
    if row is None:
        raise RuntimeError('Account not initialized; do not silently create or reset it.')
    return json.loads(row[0])


def commit(db, state, records):
    with db:
        row = db.execute('SELECT record_hash FROM journal ORDER BY seq DESC LIMIT 1').fetchone()
        previous = row[0] if row else ''
        for record in records:
            payload = encode(record)
            digest = sha((previous + '\n' + payload).encode())
            db.execute('INSERT INTO journal(payload,previous_hash,record_hash) VALUES (?,?,?)', (payload, previous, digest))
            previous = digest
        db.execute('INSERT OR REPLACE INTO state(id,payload) VALUES (1,?)', (encode(state),))


def check_frozen(config):
    for filename, digest in config['code_sha256'].items():
        if sha((ROOT / filename).read_bytes()) != digest:
            raise RuntimeError('Frozen source changed: ' + filename + '. A versioned review is required.')


def collect_signal(start):
    slot = start // INTERVAL * INTERVAL
    first = (slot // DAY - 65) * DAY
    today = slot // DAY * DAY
    cache = ROOT / 'data'
    cache.mkdir(exist_ok=True)
    source_cache = ROOT.parent / 'selective-research/data/eth_jpy'
    snapshots = []
    n = (slot - first) // INTERVAL + 1
    candles = np.full((n, 6), np.nan)
    candles[:, 5] = first + np.arange(n) * INTERVAL
    for day in range(first, today + DAY, DAY):
        name = datetime.fromtimestamp(day / 1000, timezone.utc).strftime('%Y%m%d') + '.json'
        path = cache / name
        if day < today - DAY and not path.exists() and (source_cache / name).exists():
            shutil.copyfile(source_cache / name, path)
        if not path.exists() or day >= today - DAY:
            obj, raw = get_public('candlestick/15min/' + name[:8])
            atomic_write(path, raw.decode())
        else:
            raw = path.read_bytes()
            obj = json.loads(raw)
        snapshots.append({'date': name[:8], 'sha256': sha(raw)})
        if obj.get('success') != 1:
            raise RuntimeError('Invalid candle archive')
        for group in obj['data']['candlestick']:
            if group['type'] != '15min':
                raise RuntimeError('Unexpected candle type')
            for row in group['ohlcv']:
                t = int(row[5])
                # Never read any current, unfinished 15-minute bar into the strategy.
                if first <= t < slot:
                    i = (t - first) // INTERVAL
                    if t != first + i * INTERVAL:
                        raise RuntimeError('Unaligned candle')
                    value = np.asarray(row, float)
                    if np.isfinite(candles[i, 0]) and not np.array_equal(candles[i], value):
                        raise RuntimeError('Conflicting candles')
                    candles[i] = value
    b = candles[:-1]
    if not np.all(np.isfinite(b)) or np.any(b[:, :4] <= 0) or np.any(b[:, 4] < 0):
        raise RuntimeError('Missing or invalid completed candles; no interpolation')
    if np.any(b[:, 2] > np.minimum(b[:, 0], b[:, 3])) or np.any(b[:, 1] < np.maximum(b[:, 0], b[:, 3])):
        raise RuntimeError('Invalid OHLC range')
    engine = frozen_module()
    features, _ = engine.features(candles, with_models=False)
    plan = engine.make_plan(features, 'buffer_momentum_S')
    i = len(candles) - 1
    if not features['valid'][i]:
        raise RuntimeError('Incomplete indicator history')
    values = {k: float(features[k][i]) for k in ['p', 'mom30', 'd_atr', 'h4_atr', 'h1_atr', 'd_ma5', 'd_ma20', 'h4_ma5', 'h4_ma20']}
    values.update({'slot': slot, 'closed_bar_start': slot - INTERVAL, 'entry': bool(plan['entry'][i]),
                   'exit': bool(plan['exit'][i]), 'gate': bool(plan['gate'][i]),
                   'daily_on': bool(features['daily_on'][i]), 'boundary_band': max(.01, .5 * values['d_atr'] / values['p']),
                   'history_fingerprint': sha(encode(snapshots).encode())})
    return values, snapshots


def collect_book():
    circuit, _ = get_public('circuit_break_info')
    book, _ = get_public('depth')
    observed = now_ms()
    d = book['data']
    asks = [(float(p), float(q)) for p, q in d['asks']]
    bids = [(float(p), float(q)) for p, q in d['bids']]
    if not asks or not bids or not all(math.isfinite(x) and x > 0 for row in asks + bids for x in row):
        raise RuntimeError('Invalid order book')
    ask = min(p for p, q in asks)
    bid = max(p for p, q in bids)
    age = observed - int(d['timestamp'])
    if not (-5000 <= age <= 90000 and 0 < bid < ask):
        raise RuntimeError('Stale or crossed order book')
    if circuit['data']['mode'] != 'NONE':
        raise RuntimeError('Exchange is not in continuous trading')
    return {'observed_ms': observed, 'timestamp': int(d['timestamp']), 'bid': bid, 'ask': ask,
            'bid_qty': sum(q for p, q in bids if p == bid), 'ask_qty': sum(q for p, q in asks if p == ask),
            'mid': (bid + ask) / 2, 'spread_fraction': (ask - bid) / ((ask + bid) / 2),
            'raw_depth': d, 'raw_circuit': circuit['data']}


def lot(value):
    return max(0., math.floor((value + 1e-12) / .0001) * .0001)


def queue(account_state, side, reason, slot, observed, atr, target=None):
    account_state['pending'] = {'side': side, 'reason': reason, 'decision_slot': slot,
                                'decision_ms': observed, 'due_slot': slot + INTERVAL,
                                'atr': atr, 'target': target}
    account_state['confirmation'] = 0


def execute_pending(name, a, book, slot, events):
    pending = a['pending']
    if not pending or slot < pending['due_slot']:
        return False
    observed = book['observed_ms']
    _, kind, fee, slip = ACCOUNT_INFO[name]
    if observed > pending['due_slot'] + INTERVAL:
        # An exit remains an exit; missed buys/positive rebalance intents expire.
        if pending['side'] == 'buy' or (pending['side'] == 'target' and pending['target'] > 0):
            events.append({'type': 'cancel', 'account': name, 'at': observed, 'reason': 'entry_expired', 'intent': pending})
            a['pending'] = None
            if kind == 'daily':
                a.update(daily_target=None, daily_day=None)
            return False
    buy = math.ceil(book['ask'] * (1 + slip))
    sell = math.floor(book['bid'] * (1 - slip))
    side = pending['side']
    if side == 'buy':
        qty = lot(.25 * a['cash'] / (buy * (1 + fee)))
    elif side == 'sell':
        qty = a['qty']
    else:
        p = book['mid']
        target = pending['target']
        before = a['cash'] + a['qty'] * p
        gap = target * before - a['qty'] * p
        if gap > 1e-8:
            desired = lot(a['qty'] + gap / (p * (1 + target * (buy * (1 + fee) / p - 1))))
            qty = min(lot(desired - a['qty']), lot(a['cash'] / (buy * (1 + fee))))
            side = 'buy'
        else:
            desired = lot(a['qty'] + gap / (p * (1 - target * (1 - sell * (1 - fee) / p))))
            qty = min(a['qty'], max(0., a['qty'] - desired))
            side = 'sell'
    if qty < 1e-10:
        a['pending'] = None
        events.append({'type': 'cancel', 'account': name, 'at': observed, 'reason': 'no_executable_quantity'})
        return False
    if side == 'buy' and book['spread_fraction'] > .005:
        events.append({'type': 'defer', 'account': name, 'at': observed, 'reason': 'wide_spread'})
        return False
    capacity = book['ask_qty' if side == 'buy' else 'bid_qty']
    if qty > capacity + 1e-12:
        events.append({'type': 'defer', 'account': name, 'at': observed, 'reason': 'insufficient_top_of_book'})
        return False
    price = buy if side == 'buy' else sell
    cost = qty * price * fee
    was_flat = a['qty'] < 1e-10
    if side == 'buy':
        a['cash'] -= qty * price + cost
        a['qty'] += qty
        a['basis'] += qty * price + cost
        if was_flat:
            a.update(entry_ms=observed, entry_price=price, highwater=price,
                     stop_distance=max(.02 * price, 3 * pending['atr']),
                     trail_distance=max(.02 * price, 4 * pending['atr']))
    else:
        basis_sold = a['basis'] * qty / a['qty']
        a['cash'] += qty * price - cost
        a['realized_pnl'] += qty * price - cost - basis_sold
        a['basis'] -= basis_sold
        a['qty'] -= qty
        if a['qty'] < 1e-10:
            a.update(qty=0., basis=0., entry_ms=None, entry_price=None, highwater=None,
                     stop_distance=None, trail_distance=None, last_exit_slot=slot)
            a['round_trips'] += 1
    a['fees'] += cost
    a['orders'] += 1
    a['pending'] = None
    a['confirmation'] = 0
    if a['cash'] < -1e-7 or a['qty'] < -1e-10:
        raise AssertionError('Negative paper balance')
    events.append({'type': 'paper_fill', 'account': name, 'at': observed, 'slot': slot,
                   'side': side, 'qty': qty, 'price': price, 'fee': cost,
                   'quote_timestamp': book['timestamp'], 'reason': pending['reason'],
                   'decision_ms': pending['decision_ms'], 'due_slot': pending['due_slot'],
                   'actual_delay_seconds': (observed - pending['decision_ms']) / 1000,
                   'cash_after': a['cash'], 'qty_after': a['qty'], 'real_order': False})
    return side == 'buy' and was_flat


def advance(state, signal, book):
    state = copy.deepcopy(state)
    slot, observed = signal['slot'], book['observed_ms']
    if observed // INTERVAL * INTERVAL != slot:
        raise RuntimeError('A candle boundary passed during collection; retry in the current slot')
    if state['last_slot'] is not None and slot <= state['last_slot']:
        return state, [], {'duplicate_slot': True, 'notices': []}
    events = []
    notices = []
    gap = (max(0, (slot - state['started_ms'] // INTERVAL * INTERVAL) // INTERVAL)
           if state['last_slot'] is None else max(0, (slot - state['last_slot']) // INTERVAL - 1))
    if gap:
        state['missing_slots'] += gap
        events.append({'type': 'observation_gap', 'at': observed, 'missed_slots': gap,
                       'after_slot': state['last_slot'], 'before_slot': slot, 'backfilled_trades': False})
        notices.append({'kind': 'gap', 'missed_slots': gap})
        for name, a in state['accounts'].items():
            a['confirmation'] = 0
            pending = a['pending']
            if pending and (pending['side'] == 'buy' or (pending['side'] == 'target' and pending['target'] > 0)):
                events.append({'type': 'cancel', 'account': name, 'at': observed, 'reason': 'monitoring_gap', 'intent': pending})
                a['pending'] = None
                if ACCOUNT_INFO[name][1] == 'daily':
                    a.update(daily_target=None, daily_day=None)
    if state['health'] == 'error':
        notices.append({'kind': 'data_recovered'})
    actions = {}
    for name, a in state['accounts'].items():
        _, kind, fee, slip = ACCOUNT_INFO[name]
        filled_entry = execute_pending(name, a, book, slot, events)
        reason = 'pending_execution' if a['pending'] else 'wait'
        if kind == 'candidate' and a['pending'] is None and not filled_entry:
            if a['qty'] > 0:
                a['highwater'] = max(a['highwater'], signal['p'])
                exit_reason = None
                if signal['p'] <= a['entry_price'] - a['stop_distance']:
                    exit_reason = 'initial_stop'
                elif signal['p'] <= a['highwater'] - a['trail_distance']:
                    exit_reason = 'trailing_exit'
                elif signal['exit']:
                    exit_reason = 'momentum_exit'
                elif observed - a['entry_ms'] >= 60 * DAY:
                    exit_reason = 'time_exit'
                if exit_reason:
                    queue(a, 'sell', exit_reason, slot, observed, 0.)
                    reason = exit_reason + '_queued'
                else:
                    reason = 'hold'
            else:
                cooldown = a['last_exit_slot'] is not None and slot - a['last_exit_slot'] < DAY
                if signal['entry'] and signal['gate'] and not cooldown:
                    a['confirmation'] += 1
                    reason = 'confirming_' + str(a['confirmation']) + '_of_4'
                    if a['confirmation'] >= 4:
                        queue(a, 'buy', 'entry_conditions', slot, observed, signal['h4_atr'])
                        reason = 'entry_queued'
                else:
                    a['confirmation'] = 0
                    reason = 'cooldown' if cooldown else ('entry_filter_failed' if signal['entry'] else 'trend_condition_failed')
        elif kind == 'daily' and a['pending'] is None:
            day = slot // DAY
            # Evaluate the current completed daily signal once per UTC day; startup is evaluated once.
            if day != a['daily_day']:
                date = datetime.fromtimestamp(slot / 1000, timezone.utc)
                week = date.strftime('%G-%V')
                target = .25 if signal['daily_on'] else 0.
                weekly = date.weekday() == 0 and week != a['daily_week']
                if a['daily_target'] != target or weekly:
                    queue(a, 'target', 'daily_signal_or_weekly_rebalance', slot, observed, signal['h4_atr'], target)
                    reason = 'daily_target_queued'
                else:
                    reason = 'daily_hold' if a['qty'] else 'daily_wait'
                a.update(daily_target=target, daily_day=day, daily_week=week if weekly else a['daily_week'])
            else:
                reason = 'daily_hold' if a['qty'] else 'daily_wait'
        if filled_entry:
            reason = 'entry_filled'
        a['marked_value'] = a['cash'] + a['qty'] * book['mid']
        a['liquidation_value'] = a['cash'] + a['qty'] * math.floor(book['bid'] * (1 - slip)) * (1 - fee)
        a['net_return'] = a['liquidation_value'] / 10000 - 1
        a['peak_liquidation'] = max(a['peak_liquidation'], a['liquidation_value'])
        a['max_drawdown'] = max(a['max_drawdown'], 1 - a['liquidation_value'] / a['peak_liquidation'])
        actions[name] = reason
    for event in events:
        if event['type'] == 'paper_fill':
            notices.append({'kind': 'paper_fill', 'account': event['account'], 'side': event['side'], 'qty': event['qty'], 'price': event['price']})
    state.update(last_slot=slot, last_observed_ms=observed, health='ok', last_error=None,
                 last_signal=signal, last_book={k: v for k, v in book.items() if not k.startswith('raw_')})
    state['successful_slots'] += 1
    primary = state['accounts']['candidate_base']
    for key, condition in [('drawdown_10pct', primary['max_drawdown'] >= .10),
                           ('drawdown_20pct', primary['max_drawdown'] >= .20),
                           ('doubled', primary['liquidation_value'] >= 20000),
                           ('90day_review', observed - state['started_ms'] >= 90 * DAY),
                           ('180day_review', observed - state['started_ms'] >= 180 * DAY and primary['round_trips'] >= 20)]:
        if condition and key not in state['reported_milestones']:
            state['reported_milestones'].append(key)
            notices.append({'kind': key})
    events.append({'type': 'observation', 'at': observed, 'slot': slot, 'signal': signal,
                   'book': book, 'actions': actions,
                   'account_metrics': {n: {k: a[k] for k in ['cash', 'qty', 'fees', 'realized_pnl', 'liquidation_value', 'max_drawdown', 'orders', 'round_trips']} for n, a in state['accounts'].items()}})
    return state, events, {'duplicate_slot': False, 'actions': actions, 'notices': notices}


def render(state, config):
    last = state['last_observed_ms']
    rows = ['ETH 仮想運用｜現在の成績', '', '実資金・実注文：0円／なし',
            '開始：' + jst(state['started_ms']), '最終確認：' + (jst(last) if last else '未取得'),
            '状態：' + ('正常' if state['health'] == 'ok' else state['health']),
            '初期資金：各比較口座10,000円。追加入金・出金なし。購入時は資産の25%。',
            '通常・高コストと日次基準は独立した比較用の仮想口座で、合計4万円を投資する意味ではありません。', '',
            '口座名 | 今売却した場合の推定残高 | 損益 | 最大下落 | 注文回数 | 往復回数']
    for name, a in state['accounts'].items():
        rows.append(f"{ACCOUNT_INFO[name][0]} | {a['liquidation_value']:,.2f}円 | {a['net_return']:+.2%} | {a['max_drawdown']:.2%} | {a['orders']} | {a['round_trips']}")
    rows.extend(['現金待機 | 10,000.00円 | +0.00% | 0.00% | 0 | 0', '',
                 f"正常確認：{state['successful_slots']}回　未確認の15分区間：{state['missing_slots']}　取得失敗：{state['failures']}回",
                 '残高は、現在の買い気配で売却し、価格ずれと手数料も引いた推定値です。',
                 '最大下落は観測できた時点での値です。監視停止中の下落を含む完全な最大下落ではありません。',
                 '通常：片道手数料0.15%＋不利な価格ずれ0.05%。高コスト：0.15%＋0.35%。売買価格差も別に反映。',
                 '判断は確定した15分足のみ。予約は次の15分区間以降に取得した公開気配で仮想約定します。',
                 '取り逃した区間の売買は作り直しません。古い買い予約は取り消し、売り予約は再開時の価格で処理します。',
                 '短期の成績でルールを変更しません。90日で見直し、180日・20往復以上を評価の最低目安とします。',
                 '税・電気代・既存サービス利用料は損益に含みません。実運用の利益や約定を保証しません。',
                 'このMacがオンラインでアプリが動いている間の定期実行です。既存のChatGPT利用枠を使います。',
                 'スケジュールの設定情報：work/paper-trading/schedule.json（設定後に保存）'])
    if state['last_error']:
        rows.extend(['', '最新の取得エラー：' + state['last_error']])
    if state['last_signal']:
        sig = state['last_signal']
        rows.extend(['', '直近の判断材料：', f"30日騰落率 {sig['mom30']:+.2%}／購入の境界 {sig['boundary_band']:.2%}",
                     f"上昇条件：{sig['entry']}／値幅・急変条件：{sig['gate']}"])
    for name, a in state['accounts'].items():
        if a['pending']:
            rows.append(f"予約中：{ACCOUNT_INFO[name][0]}／{a['pending']['side']}／最早 {jst(a['pending']['due_slot'])}")
    atomic_write(OUTPUT / '仮想運用_現在の成績.txt', '\n'.join(rows) + '\n')
    atomic_write(OUTPUT / '仮想運用_残高と状態.json', json.dumps({'config': config, 'state': state}, ensure_ascii=False, indent=2, allow_nan=False))


def initialize(db):
    if (ROOT / 'config.json').exists() or db.execute('SELECT 1 FROM state').fetchone():
        raise RuntimeError('Already initialized. Never reset the live paper balances.')
    started = now_ms()
    code_hashes = {p: sha((ROOT / p).read_bytes()) for p in ['paper.py', 'frozen/engine.py', 'frozen/protocol.json']}
    config = {'version': 'eth-forward-paper-v1', 'pair': 'eth_jpy', 'started_ms': started,
              'candidate': 'buffer_momentum_S', 'initial_yen': 10000, 'allocation_at_entry': .25,
              'interval_minutes': 15, 'real_orders_enabled': False, 'paid_model_api': False,
              'code_sha256': code_hashes, 'rule_freeze_days': 90, 'minimum_evaluation_days': 180,
              'minimum_round_trips': 20, 'history_role': 'indicator_warmup_only_no_backfilled_trades',
              'forward_execution': 'Next observed 15m slot at actual public ask/bid plus adverse slippage and fee; no historical fills.',
              'research_origin': 'Candidate was selected after examining past ETH results; this forward run begins now.',
              'accounts': ACCOUNT_INFO}
    state = initial_state(started)
    commit(db, state, [{'type': 'initialized', 'at': started, 'config': config}])
    atomic_write(ROOT / 'config.json', json.dumps(config, ensure_ascii=False, indent=2))
    render(state, config)
    return {'initialized': True, 'started_jst': jst(started), 'balance': 10000}


def tick(db, config):
    state = load_state(db)
    start = now_ms()
    if state['last_slot'] == start // INTERVAL * INTERVAL:
        render(state, config)
        return {'duplicate_slot': True, 'notices': [], 'last_observed_jst': jst(state['last_observed_ms'])}
    try:
        check_frozen(config)
        signal, data_hashes = collect_signal(start)
        book = collect_book()
        new_state, events, result = advance(state, signal, book)
        events[-1]['history_files'] = data_hashes
    except Exception as exc:
        changed = state['health'] != 'error' or state['last_error'] != str(exc)
        state.update(health='error', last_error=str(exc))
        state['failures'] += 1
        commit(db, state, [{'type': 'error', 'at': now_ms(), 'message': str(exc)}])
        render(state, config)
        return {'health': 'error', 'error': str(exc), 'notices': [{'kind': 'data_error', 'message': str(exc)}] if changed else []}
    # Never roll back committed balances to a stale in-memory state if report writing fails.
    commit(db, new_state, events)
    render(new_state, config)
    result.update(health='ok', observed_jst=jst(book['observed_ms']),
                  primary_balance_yen=new_state['accounts']['candidate_base']['liquidation_value'],
                  missing_slots=new_state['missing_slots'])
    return result


def verify(db):
    previous = ''
    ledger = {n: {'cash': 10000., 'qty': 0.} for n in ACCOUNT_INFO}
    count = 0
    for _, payload, prev, digest in db.execute('SELECT seq,payload,previous_hash,record_hash FROM journal ORDER BY seq'):
        assert prev == previous and digest == sha((prev + '\n' + payload).encode()), 'Journal hash mismatch'
        previous = digest
        count += 1
        event = json.loads(payload)
        if event['type'] == 'paper_fill':
            a = ledger[event['account']]
            direction = 1 if event['side'] == 'buy' else -1
            a['cash'] -= direction * event['qty'] * event['price'] + event['fee']
            a['qty'] += direction * event['qty']
            assert event['at'] >= event['due_slot'] > event['decision_ms'] - INTERVAL
            assert math.isclose(a['cash'], event['cash_after'], abs_tol=1e-6)
            assert math.isclose(a['qty'], event['qty_after'], abs_tol=1e-9)
    state = load_state(db)
    for n, a in ledger.items():
        assert math.isclose(a['cash'], state['accounts'][n]['cash'], abs_tol=1e-6)
        assert math.isclose(a['qty'], state['accounts'][n]['qty'], abs_tol=1e-9)
    return {'verified': True, 'journal_records': count, 'real_orders': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'tick', 'status', 'verify'])
    args = parser.parse_args()
    with (ROOT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        db = connect(ROOT / 'paper.sqlite3')
        try:
            if args.command == 'init':
                result = initialize(db)
            else:
                config = json.loads((ROOT / 'config.json').read_text())
                if args.command == 'tick':
                    result = tick(db, config)
                elif args.command == 'verify':
                    result = verify(db)
                else:
                    state = load_state(db)
                    render(state, config)
                    result = {'health': state['health'], 'last_observed_ms': state['last_observed_ms'], 'accounts': state['accounts'], 'successful_slots': state['successful_slots'], 'missing_slots': state['missing_slots'], 'failures': state['failures']}
            print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        finally:
            db.close()


if __name__ == '__main__':
    main()
