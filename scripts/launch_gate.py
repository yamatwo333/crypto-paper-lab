"""Read-only launch gate. It cannot fetch prices, initialize, or alter a ledger."""
import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import probe

ROOT = Path(__file__).resolve().parents[1]


def decide(root, event, operation='tick', confirmation=''):
    root = Path(root)
    settings = json.loads((root / 'control.json').read_text())
    if settings.get('format') != 'paper-experiment-control-v1':
        raise ValueError('Invalid control format')
    phase = settings.get('phase')
    if phase == 'awaiting_new_experiment_approval':
        return {'run': False, 'reason': '新しい実験の開始確認待ち。相場取得・帳簿作成なし。'}
    if phase not in ('approved_new', 'running'):
        raise ValueError('Unknown or stopped phase; do not restart automatically')
    identity = settings.get('experiment_id')
    if not isinstance(identity, str) or not re.fullmatch(r'new-[0-9]{8}-[a-z0-9-]{1,40}', identity):
        raise ValueError('Invalid approved experiment ID')
    if datetime.fromisoformat(settings['approved_at']).tzinfo is None:
        raise ValueError('Approval timestamp must include timezone')
    if settings.get('frozen_source_sha256') != json.loads((root / 'frozen_source_sha256.json').read_text()):
        raise ValueError('Source differs from the approved source manifest')
    if phase == 'running':
        if operation == 'start' and event == 'workflow_dispatch':
            raise ValueError('Already started; never initialize again')
        if event in ('push', 'workflow_run'):
            return {'run': False, 'reason': '開始済み。コード更新や検証の終了で売買を追加実行しません。'}
        return {'run': True, 'operation': 'tick', 'confirmation': '', 'reason': '保存済みの帳簿から継続。'}
    if (root / 'experiments' / identity).exists():
        raise ValueError('Partial or existing experiment found; never overwrite or initialize again')
    window_path = root / 'validation/window.json'
    if window_path.exists():
        window = json.loads(window_path.read_text())
        if window['sha256'] != probe.digest(window['payload']) or not window['payload'].get('ok'):
            raise ValueError('Full price-window validation failed or is corrupt; review before starting')
    history = probe.load(root / 'validation/history.json')
    if any(not item['payload']['success'] for item in history):
        raise ValueError('Validation failed; preserve evidence and review before starting')
    if probe.stop_reason(history) != 'two_scheduled_checks_complete':
        count = len({x['payload']['slot'] for x in history if x['payload']['success'] and x['payload']['event'] == 'schedule'})
        return {'run': False, 'reason': f'予約検証の成功 {count}/2 回。まだ相場取得・帳簿作成・売買は行いません。'}
    if event == 'workflow_dispatch' and (operation != 'start' or confirmation != identity):
        return {'run': False, 'reason': '手動開始には承認済み実験IDの指定が必要です。'}
    if event not in ('push', 'workflow_run', 'schedule', 'workflow_dispatch'):
        raise ValueError('Unsupported launch event')
    return {'run': True, 'operation': 'start', 'confirmation': identity,
            'reason': '新規開始への同意と、予約検証2回の成功を確認。'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operation', default='tick', choices=['tick', 'start'])
    parser.add_argument('--confirmation', default='')
    args = parser.parse_args()
    result = decide(ROOT, os.environ.get('GITHUB_EVENT_NAME', 'manual'), args.operation, args.confirmation)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write('run=' + str(result['run']).lower() + '\n')
            stream.write('operation=' + result.get('operation', '') + '\n')
            stream.write('confirmation=' + result.get('confirmation', '') + '\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(result['reason'] + '\n\n実注文は行いません。\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
