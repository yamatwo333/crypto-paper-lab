"""Check native schedule evidence or reviewed external scheduler receipts.

External receipts are an operator-reviewed attestation, not a cryptographic
claim by cron-job.org. They must be committed only after checking its saved
HTTP response against the successful GitHub job and immutable probe history.
The hash detects corruption; workflow_dispatch is never relabeled schedule.
"""
import json
from datetime import datetime
from pathlib import Path

import probe


def milliseconds(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError('External evidence timestamp has no timezone')
    return int(stamp.timestamp() * 1000)


def ready(root, rows=None):
    root = Path(root)
    rows = probe.load(root / 'validation/history.json') if rows is None else rows
    if any(not r['payload']['success'] or r['payload'].get('policy_denial') for r in rows):
        raise ValueError('Validation failed; preserve evidence and review before starting')
    if probe.stop_reason(rows) == 'two_scheduled_checks_complete':
        return True
    path = root / 'validation/external-receipts.json'
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    data = receipt['payload']
    if receipt['sha256'] != probe.digest(data):
        raise ValueError('External evidence integrity mismatch')
    settings = json.loads((root / 'control.json').read_text())
    expected = {
        'format': 'external-scheduled-storage-proof-v1',
        'provider': 'cron-job.org', 'job_id': 8676656,
        'repository': 'yamatwo333/crypto-paper-lab',
        'workflow': '.github/workflows/probe.yml',
        'experiment_id': settings['experiment_id'],
    }
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('External evidence targets a different experiment or workflow')
    verified = milliseconds(data['verified_at'])
    approved = milliseconds(settings['approved_at'])
    checks = data['checks']
    if len(checks) != 2:
        raise ValueError('Two external scheduled storage receipts are required')
    by_run = {r['payload']['run_id']: r for r in rows}
    if len(by_run) != len(rows):
        raise ValueError('Duplicate validation run')
    seen_runs, seen_slots, seen_jobs, seen_cron = set(), set(), set(), set()
    planned, sequences = [], []
    for check in checks:
        run_id = check['github_run_id']
        row = by_run.get(run_id)
        if not row or row['hash'] != check['history_hash']:
            raise ValueError('External receipt does not match a saved validation run')
        p = row['payload']
        if (p['event'] != 'workflow_dispatch' or p['run_attempt'] != '1'
                or check['github_event'] != 'workflow_dispatch'
                or check['github_conclusion'] != 'success'
                or check['http_status'] != 200
                or p['paper_trades'] is not False
                or p['kind'] != 'public_data_storage_test_only'
                or p['previous_records_loaded'] != p['seq'] - 1
                or set(p.get('markets', {})) != set(probe.PAIRS)):
            raise ValueError('External scheduled storage check is incomplete')
        job = check['github_job_id']
        cron = check['cron_history_row']
        if (not isinstance(job, int) or job <= 0 or not isinstance(cron, str)
                or not cron.startswith('8676656-')
                or job in seen_jobs or cron in seen_cron
                or run_id in seen_runs or p['slot'] in seen_slots):
            raise ValueError('External checks must use distinct jobs, runs and slots')
        scheduled = milliseconds(check['scheduled_at'])
        executed = milliseconds(check['executed_at'])
        completed = milliseconds(check['github_completed_at'])
        if not (approved <= scheduled <= executed <= p['started_ms']
                <= p['finished_ms'] <= completed <= verified):
            raise ValueError('External evidence timestamps are out of order')
        if completed - scheduled > probe.INTERVAL or executed - scheduled > 120000:
            raise ValueError('External check was delayed outside its validation interval')
        seen_runs.add(run_id)
        seen_slots.add(p['slot'])
        seen_jobs.add(job)
        seen_cron.add(cron)
        planned.append(scheduled)
        sequences.append(p['seq'])
    if planned[1] - planned[0] != probe.INTERVAL or sequences[1] != sequences[0] + 1:
        raise ValueError('External checks must be consecutive 15-minute checks with restored history')
    return True
