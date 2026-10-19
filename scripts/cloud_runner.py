"""Persistent forward-test adapter; the frozen strategy files are unchanged.

Start is a separate, explicit, one-use operation. Tick cannot initialize state.
Git is the authority between jobs, not the disposable runner filesystem.
"""
import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'comparison'))
import runner
import ledger_store
import probe


def save(path, value):
    ledger_store.atomic_write(path, ledger_store.encode(value) + b'\n')


def control(root, phase):
    data = json.loads((root / 'control.json').read_text())
    if data.get('format') != 'paper-experiment-control-v1' or data.get('phase') != phase:
        raise ValueError('Experiment is not authorized for this operation; no automatic initialization')
    identity = data.get('experiment_id')
    if not isinstance(identity, str) or not re.fullmatch(r'new-[0-9]{8}-[a-z0-9-]{1,40}', identity):
        raise ValueError('Missing or invalid separate experiment identity')
    approved = datetime.fromisoformat(data['approved_at'])
    if approved.tzinfo is None:
        raise ValueError('Approval time must have a timezone')
    if data.get('frozen_source_sha256') != json.loads((root / 'frozen_source_sha256.json').read_text()):
        raise ValueError('Approved source hashes differ')
    return data, root / 'experiments' / identity


def sources(root):
    expected = json.loads((root / 'frozen_source_sha256.json').read_text())
    for name, digest in expected.items():
        if ledger_store.sha((root / name).read_bytes()) != digest:
            raise ValueError('Frozen source hash mismatch: ' + name)
    return runner.check_files(), json.loads((runner.ROOT / 'rules.json').read_text())


def check_identity(directory, state, settings, hashes):
    identity = json.loads((directory / 'manifest.json').read_text())
    expected = {
        'format': 'new-paper-comparison-v1',
        'experiment_id': settings['experiment_id'],
        'approved_at': settings['approved_at'],
        'started_ms': state['started_ms'],
        'initial_yen_per_method': 10000,
        'previous_experiment_continued': False,
        'frozen_source_sha256': settings['frozen_source_sha256'],
    }
    if identity != expected or state['code_sha256'] != hashes:
        raise ValueError('Experiment identity, approval or source changed')


def public_status(settings, state):
    result = runner.summary(state)
    result.update(experiment_id=settings['experiment_id'], previous_experiment_continued=False)
    return result


def status_page(directory, result):
    summary = result['summary']
    if result.get('stop'):
        label = '停止。通信拒否などの確認が必要です。'
    elif result['operation'] == 'start_new_experiment':
        label = '初期帳簿を作成。別の実行からの初回観測を確認する段階です。'
    elif not result['ok']:
        label = '相場取得に失敗。この回は売買せず、次の予約で確認します。'
    else:
        label = '最終記録時点で観測成功。仮想比較を継続する設定です。'
    lines = ['# 仮想1万円の比較：現在の記録', '', label, '',
             '実資金・実注文なし。各方式の仮想1万円を比較する実験です。', '',
             '- 新規実験ID：`' + summary['experiment_id'] + '`',
             '- 開始：' + summary['started_jst'],
             '- 最終相場確認：' + str(summary['last_observed_jst'] or 'まだ確認していません'),
             '- 未確認の15分区間：' + str(summary['missing_slots']),
             '- 相場取得の失敗回数：' + str(summary['failed_runs']), '',
             '| 方式 | 売却費用控除後の資産 | 純損益 | 最大下落 | 往復売買数 |',
             '| --- | ---: | ---: | ---: | ---: |']
    names = {'eth_base': 'ETH・通常コスト', 'eth_stress': 'ETH・高コスト',
             'multi_base': '複数銘柄・通常コスト', 'multi_stress': '複数銘柄・高コスト'}
    for key, name in names.items():
        a = summary['accounts'][key]
        lines.append(f"| {name} | {a['equity_yen']:,.2f}円 | {a['pnl_yen']:+,.2f}円 | {a['mdd']:.2%} | {a['round_trips']} |")
    lines += ['', '複数銘柄はBTC・ETH・XRP・SOLを共通の1万円で観測します。銘柄ごとに資金を追加しません。',
              '旧クラウドの成績とは合算しません。最大下落は観測時点のみ、未確認区間の注文は後付けしません。', '',
              f"[最新の実行記録](experiments/{summary['experiment_id']}/latest_run.json) · "
              '[GitHubの実行一覧](https://github.com/yamatwo333/crypto-paper-lab/actions)', '',
              'このページは帳簿と同じコミットに保存される時点の記録です。上の最終確認時刻と、実行一覧の失敗・停止も確認してください。',
              '15分ごとの予約には遅延・欠落があり得ます。チャットへの自動通知はまだ接続していません。']
    runner.write(directory.parents[1] / 'STATUS.md', '\n'.join(lines) + '\n')


def save_result(directory, settings, db, result):
    state = runner.load(db)
    runner.export(state, directory)
    result['experiment_id'] = settings['experiment_id']
    result['summary'] = public_status(settings, state)
    result['real_orders'] = False
    result['journal'] = ledger_store.export_ledger(Path(db.execute('PRAGMA database_list').fetchone()[2]), directory / 'ledger')
    save(directory / 'status.json', result['summary'])
    save(directory / 'latest_run.json', result)
    status_page(directory, result)
    return result


def verify(root):
    root = Path(root)
    settings, directory = control(root, 'running')
    hashes, rules = sources(root)
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / 'restored.sqlite3'
        ledger_store.restore_ledger(directory / 'ledger', target)
        db = runner.connect(target)
        try:
            report = runner.verify(db, rules, hashes)
            state = runner.load(db)
            check_identity(directory, state, settings, hashes)
            if json.loads((directory / 'status.json').read_text()) != public_status(settings, state):
                raise ValueError('Published status differs from the ledger')
            latest = json.loads((directory / 'latest_run.json').read_text())
            snapshot = ledger_store.verify_archive(directory / 'ledger')
            if latest['experiment_id'] != settings['experiment_id'] or latest['summary'] != public_status(settings, state):
                raise ValueError('Last run belongs to a different state or experiment')
            if any(latest['journal'][k] != snapshot[v] for k, v in [('records', 'journal_records'), ('last_hash', 'last_hash'), ('state_sha256', 'state_sha256')]):
                raise ValueError('Last run checkpoint differs')
            return report
        finally:
            db.close()


def start(root, expected_id, collector=None, clock=None):
    """A new cohort only: collect first, initialize once, make no trade decision."""
    root = Path(root)
    settings, directory = control(root, 'approved_new')
    if expected_id != settings['experiment_id']:
        raise ValueError('Explicit start confirmation does not match the approved experiment')
    hashes, rules = sources(root)
    if directory.exists():
        raise ValueError('Experiment directory already exists; preserve it, never reinitialize')
    if probe.stop_reason(probe.load(root / 'validation/history.json')) != 'two_scheduled_checks_complete':
        raise ValueError('Two distinct successful scheduled storage checks are required before starting')
    collector, clock = collector or runner.collect, clock or runner.now
    with tempfile.TemporaryDirectory() as temp:
        temporary = Path(temp)
        obs = collector(temporary / 'cache')
        at = clock()
        if at < int(datetime.fromisoformat(settings['approved_at']).timestamp() * 1000):
            raise ValueError('Experiment cannot start before the recorded approval')
        if not 0 <= at - obs['observed_ms'] < 90000 or at // runner.I != obs['slot'] // runner.I:
            raise ValueError('Preflight is no longer fresh; no new ledger was created')
        # Validate synchronized input without applying it to an account.
        if set(obs['signals']) != set(runner.core.PAIRS) or set(obs['books']) != set(runner.core.PAIRS):
            raise ValueError('Incomplete preflight')
        db = runner.connect(temporary / 'new.sqlite3', True)
        try:
            state = runner.initialize(db, at, hashes)
            runner.verify(db, rules, hashes)
            directory.mkdir(parents=True, exist_ok=False)
            save(directory / 'manifest.json', {
                'format': 'new-paper-comparison-v1', 'experiment_id': settings['experiment_id'],
                'approved_at': settings['approved_at'], 'started_ms': at,
                'initial_yen_per_method': 10000, 'previous_experiment_continued': False,
                'frozen_source_sha256': settings['frozen_source_sha256'],
            })
            # Only public historical prices are copied; there is no old balance or order input.
            if (temporary / 'cache').exists():
                import shutil
                shutil.copytree(temporary / 'cache', directory / 'cache')
            result = save_result(directory, settings, db, {
                'ok': True, 'operation': 'start_new_experiment', 'at': at,
                'run_id': os.environ.get('GITHUB_RUN_ID'), 'stop': False,
                'notices': [{'kind': 'new_experiment_started', 'paper_trades': False}],
            })
        finally:
            db.close()
    settings['phase'] = 'running'
    save(root / 'control.json', settings)
    verify(root)
    return result


def denied(exc):
    message = str(exc).lower()
    return probe.is_denial(exc) or getattr(exc, 'code', None) in (401, 403) or ('policy' in message and ('block' in message or 'denied' in message))


def tick(root, collector=None, clock=None):
    root = Path(root)
    settings, directory = control(root, 'running')
    hashes, rules = sources(root)
    # A damaged/missing authority fails before any network call or state creation.
    verify(root)
    previous = json.loads((directory / 'latest_run.json').read_text())
    if previous.get('stop'):
        raise ValueError('Experiment halted; explicit review is required, no automatic retry')
    job = os.environ.get('GITHUB_RUN_ID')
    if job and previous.get('run_id') == job:
        return {'ok': True, 'duplicate_run': True, 'notices': [], 'stop': False, 'real_orders': False}
    collector, clock = collector or runner.collect, clock or runner.now
    with tempfile.TemporaryDirectory() as temp:
        target = Path(temp) / 'runtime.sqlite3'
        ledger_store.restore_ledger(directory / 'ledger', target)
        db = runner.connect(target)
        try:
            old = runner.load(db)
            slot = clock() // runner.I * runner.I
            if old['last_slot'] is not None and slot <= old['last_slot']:
                return {'ok': True, 'duplicate_slot': True, 'notices': [], 'stop': False, 'real_orders': False}
            try:
                obs = collector(directory / 'cache')
            except Exception as exc:
                message = type(exc).__name__ + ': ' + str(exc)
                state = runner.error_state(old, message)
                runner.commit(db, state, {'type': 'error', 'at': clock(), 'message': message})
                result = {'ok': False, 'stop': denied(exc), 'notices': ([{'kind': 'data_error', 'message': message}]
                          if old['health'] != 'error' or denied(exc) else [])}
            else:
                state, events, notices = runner.apply_tick(old, obs, rules)
                runner.commit(db, state, {'type': 'tick', 'observation': obs, 'events': events})
                result = {'ok': True, 'stop': False, 'notices': notices}
            runner.verify(db, rules, hashes)
            result.update(operation='tick', at=clock(), run_id=os.environ.get('GITHUB_RUN_ID'))
            result = save_result(directory, settings, db, result)
        finally:
            db.close()
    verify(root)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['start', 'tick', 'verify'])
    parser.add_argument('--confirm-new-experiment', default='')
    args = parser.parse_args()
    if args.operation == 'verify':
        print(json.dumps(verify(ROOT), ensure_ascii=False))
        return
    result = start(ROOT, args.confirm_new_experiment) if args.operation == 'start' else tick(ROOT)
    # Only a verified complete bundle may be committed by the workflow.
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write('prepared=true\nstop=' + str(result.get('stop', False)).lower() + '\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write('仮想比較の実行。実注文なし。Gitへの保存完了までは確定記録ではありません。\n\n')
            stream.write('```json\n' + json.dumps(result, ensure_ascii=False, indent=2) + '\n```\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
