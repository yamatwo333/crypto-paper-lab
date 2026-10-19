"""One read-only end-to-end price-window check; never opens a trading ledger."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'comparison'))
import runner
import check_sources
import probe

PATH = ROOT / 'validation/window.json'


def verify():
    report = json.loads(PATH.read_text())
    payload = report['payload']
    if report['sha256'] != runner.sha(runner.enc(payload).encode()):
        raise ValueError('Preflight record hash mismatch')
    if payload['kind'] != 'readonly_65_day_window' or payload['paper_trades'] is not False:
        raise ValueError('Unexpected preflight record')
    return payload


def run():
    check_sources.verify_sources()
    if PATH.exists():
        payload = verify()
        print(json.dumps({'already_checked': True, 'ok': payload['ok'], 'no_request': True}))
        return 0 if payload['ok'] else 1
    payload = {'kind': 'readonly_65_day_window', 'paper_trades': False, 'started_ms': runner.now(),
               'run_id': os.environ.get('GITHUB_RUN_ID'), 'code_sha256': runner.check_files()}
    try:
        with tempfile.TemporaryDirectory() as directory:
            obs = runner.collect(Path(directory) / 'cache')
            payload.update(ok=True, observation=obs, policy_denial=False)
    except Exception as exc:
        payload.update(ok=False, error=type(exc).__name__ + ': ' + str(exc), policy_denial=probe.is_denial(exc))
    payload['finished_ms'] = runner.now()
    report = {'payload': payload, 'sha256': runner.sha(runner.enc(payload).encode())}
    runner.write(PATH, json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if verify() != payload:
        raise ValueError('Saved preflight differs')
    print(json.dumps({k: v for k, v in payload.items() if k != 'observation'}, ensure_ascii=False))
    return 0 if payload['ok'] else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['verify']:
        print(json.dumps({'verified': True, 'ok': verify()['ok'], 'paper_trades': False}))
    else:
        raise SystemExit(run())
