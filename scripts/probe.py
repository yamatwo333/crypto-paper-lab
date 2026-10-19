"""Bounded public-data/storage integration test. Never opens a trading ledger.

At most five attempts, and at most two successful scheduled observations.
An explicit denial stops the test. No retry, alternate host, or proxy is used.
"""
import argparse
import hashlib
import json
import math
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ('btc_jpy', 'eth_jpy', 'xrp_jpy', 'sol_jpy')
INTERVAL = 900000
MAX_ATTEMPTS = 5
PATH = ROOT / 'validation' / 'history.json'


def encode(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def load(path):
    if not path.exists():
        return []
    rows = json.loads(path.read_text())
    previous = ''
    for index, row in enumerate(rows):
        payload = row['payload']
        if payload['seq'] != index + 1 or payload['previous_hash'] != previous or row['hash'] != digest(payload):
            raise ValueError('Validation history integrity mismatch; preserve records')
        previous = row['hash']
    return rows


def stop_reason(rows):
    if any(r['payload'].get('policy_denial') for r in rows):
        return 'explicit_denial'
    slots = {r['payload']['slot'] for r in rows
             if r['payload']['success'] and r['payload']['event'] == 'schedule'}
    if len(slots) >= 2:
        return 'two_scheduled_checks_complete'
    if len(rows) >= MAX_ATTEMPTS:
        return 'attempt_limit_reached'
    return ''


def get(pair, endpoint):
    today = datetime.now(timezone.utc).strftime('%Y%m%d')
    if pair not in PAIRS or endpoint not in ('depth', 'circuit_break_info', 'candlestick/15min/' + today):
        raise ValueError('Endpoint is not in the public-data test allowlist')
    url = 'https://public.bitbank.cc/' + pair + '/' + endpoint
    request = urllib.request.Request(url, headers={'User-Agent': 'PaperComparison/1.0'}, method='GET')
    # Do not follow redirects to a different host or route.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(request, timeout=20) as response:
        status = response.status
        raw = response.read(5_000_001)
    if len(raw) > 5_000_000:
        raise ValueError('Public response exceeds size limit')
    obj = json.loads(raw)
    if status != 200 or obj.get('success') != 1:
        raise ValueError('Public API did not confirm success: ' + pair + '/' + endpoint)
    return obj, {'endpoint': endpoint, 'http': status, 'api_success': 1,
                 'sha256': hashlib.sha256(raw).hexdigest()}


def observe():
    result = {}
    for pair in PAIRS:
        circuit, c_meta = get(pair, 'circuit_break_info')
        depth, d_meta = get(pair, 'depth')
        stamp = int(time.time() * 1000)
        data = depth['data']
        asks = [(float(p), float(q)) for p, q in data['asks']]
        bids = [(float(p), float(q)) for p, q in data['bids']]
        if not asks or not bids or not all(math.isfinite(x) and x > 0 for row in asks + bids for x in row):
            raise ValueError('Invalid order book for ' + pair)
        ask, bid = min(p for p, _ in asks), max(p for p, _ in bids)
        if not 0 < bid < ask or not -5000 <= stamp - int(data['timestamp']) <= 90000:
            raise ValueError('Stale or crossed order book for ' + pair)
        if circuit['data']['mode'] != 'NONE':
            raise ValueError('Continuous trading unavailable for ' + pair)
        today = datetime.now(timezone.utc).strftime('%Y%m%d')
        candles, f_meta = get(pair, 'candlestick/15min/' + today)
        rows = []
        for group in candles['data']['candlestick']:
            if group['type'] != '15min':
                raise ValueError('Unexpected candle interval')
            rows.extend(group['ohlcv'])
        if not rows:
            raise ValueError('Current-day candles empty; no alternate request attempted')
        for row in rows:
            numbers = [float(x) for x in row]
            if len(numbers) != 6 or not all(math.isfinite(x) for x in numbers):
                raise ValueError('Invalid candle data')
            o, h, low, close, volume, at = numbers
            if min(o, h, low, close) <= 0 or volume < 0 or not low <= min(o, close) <= max(o, close) <= h or at % INTERVAL:
                raise ValueError('Invalid candle values')
        result[pair] = {'responses': [c_meta, d_meta, f_meta], 'bid': bid, 'ask': ask,
                        'book_timestamp': int(data['timestamp']), 'observed_ms': stamp,
                        'candle_count': len(rows), 'last_candle_ms': max(int(r[5]) for r in rows)}
    return result


def is_denial(exc):
    if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
        return True
    message = str(exc).lower()
    return any(s in message for s in ('blocked by policy', 'denied by policy', 'network access denied'))


def output_decision(reason):
    target = os.environ.get('GITHUB_OUTPUT')
    if target:
        with open(target, 'a') as f:
            f.write('stop=' + ('true' if reason else 'false') + '\n')
            f.write('reason=' + reason + '\n')


def run(path=PATH, event=None, run_id=None, observer=observe):
    rows = load(path)
    reason = stop_reason(rows)
    if reason:
        output_decision(reason)
        return {'stop': True, 'reason': reason, 'no_request': True}
    event = event or os.environ.get('GITHUB_EVENT_NAME', 'local')
    run_id = run_id or os.environ.get('GITHUB_RUN_ID', 'local')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    if any(r['payload']['run_id'] == run_id and run_id != 'local' for r in rows):
        output_decision('')
        return {'stop': False, 'duplicate_run': True, 'no_request': True}
    started = int(time.time() * 1000)
    payload = {'seq': len(rows) + 1, 'previous_hash': rows[-1]['hash'] if rows else '',
               'started_ms': started, 'slot': started // INTERVAL * INTERVAL,
               'event': event, 'run_id': run_id, 'run_attempt': attempt,
               'previous_records_loaded': len(rows), 'success': False, 'policy_denial': False,
               'kind': 'public_data_storage_test_only', 'paper_trades': False}
    try:
        payload['markets'] = observer()
        payload['success'] = True
    except Exception as exc:
        payload['error'] = type(exc).__name__ + ': ' + str(exc)
        payload['policy_denial'] = is_denial(exc)
    payload['finished_ms'] = int(time.time() * 1000)
    rows.append({'payload': payload, 'hash': digest(payload)})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)
    if load(path) != rows:
        raise ValueError('Saved validation data differs')
    reason = stop_reason(rows)
    output_decision(reason)
    result = {'success': payload['success'], 'records': len(rows), 'previous_records_loaded': len(rows) - 1,
              'stop': bool(reason), 'reason': reason, 'paper_trades': False, 'ledger_imported': False}
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as f:
            f.write('## 相場取得・保存の検証（仮想売買はまだ停止中）\n\n')
            f.write('```json\n' + json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'verify'])
    args = parser.parse_args()
    if args.command == 'verify':
        print(encode({'records_verified': len(load(PATH)), 'paper_trades': False}))
    else:
        result = run()
        print(encode(result))
        raise SystemExit(0 if result.get('no_request') or result.get('success') else 1)

