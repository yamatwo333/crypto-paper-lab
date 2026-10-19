"""Lossless, append-only journal transport for the existing SQLite formats.

No network, initialization, price collection, or trading. SQLite remains the
runtime format. Small immutable gzip records let Git preserve the complete
journal without committing an ever-growing database binary on every run.
"""
import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def record_path(directory, seq):
    return directory / 'journal' / f'{(seq - 1) // 1000:06d}' / f'{seq:012d}.json.gz'


def checked_record(seq, payload, previous, digest, expected_seq, expected_previous):
    if seq != expected_seq or previous != expected_previous:
        raise ValueError('Missing, reordered, or conflicting journal records')
    if digest != sha((previous + '\n' + payload).encode()):
        raise ValueError('Journal hash mismatch')
    json.loads(payload)
    return digest


def export_ledger(source, directory):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if not source.is_file():
        raise ValueError('Missing original ledger; never create or reset it')
    checkpoint = directory / 'checkpoint.json'
    old = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    previous = ''
    count = 0
    prior_hash_seen = None
    db = sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite integrity check failed')
        state_row = db.execute('SELECT payload FROM state WHERE id=1').fetchone()
        if not state_row:
            raise ValueError('Missing state; never initialize')
        state = state_row[0]
        json.loads(state)
        rows = db.execute('SELECT seq,payload,previous_hash,record_hash FROM journal ORDER BY seq')
        for seq, payload, prev, digest in rows:
            previous = checked_record(seq, payload, prev, digest, count + 1, previous)
            count += 1
            if old and count == old['journal_records']:
                prior_hash_seen = digest
            raw = encode([seq, payload, prev, digest])
            path = record_path(directory, seq)
            if path.exists():
                if gzip.decompress(path.read_bytes()) != raw:
                    raise ValueError('Refusing to overwrite an existing journal record')
            else:
                atomic_write(path, gzip.compress(raw, mtime=0))
    finally:
        db.close()
    if count == 0:
        raise ValueError('Empty journal; never initialize')
    if old:
        if count < old['journal_records'] or prior_hash_seen != old['last_hash']:
            raise ValueError('Refusing ledger rollback or a different journal chain')
        if count == old['journal_records'] and sha(state.encode()) != old['state_sha256']:
            raise ValueError('State changed without journal advancement')
    actual = sorted((directory / 'journal').glob('*/*.json.gz'))
    if len(actual) != count:
        raise ValueError('Archive contains records outside this checkpoint; preserve and investigate')
    snapshot = {'format': 'paper-ledger-git-v1', 'journal_records': count, 'last_hash': previous,
                'state_payload': state, 'state_sha256': sha(state.encode())}
    atomic_write(checkpoint, encode(snapshot) + b'\n')
    verify_archive(directory)
    return {'records': count, 'last_hash': previous, 'state_sha256': snapshot['state_sha256']}


def archive_rows(directory, snapshot):
    previous = ''
    for seq in range(1, snapshot['journal_records'] + 1):
        row = json.loads(gzip.decompress(record_path(directory, seq).read_bytes()))
        previous = checked_record(*row, seq, previous)
        yield row
    if previous != snapshot['last_hash']:
        raise ValueError('Checkpoint and journal differ')


def verify_archive(directory):
    directory = Path(directory)
    snapshot = json.loads((directory / 'checkpoint.json').read_text())
    if snapshot['format'] != 'paper-ledger-git-v1' or not isinstance(snapshot['journal_records'], int) or snapshot['journal_records'] < 1:
        raise ValueError('Invalid checkpoint')
    if sha(snapshot['state_payload'].encode()) != snapshot['state_sha256']:
        raise ValueError('State snapshot hash mismatch')
    json.loads(snapshot['state_payload'])
    if len(list((directory / 'journal').glob('*/*.json.gz'))) != snapshot['journal_records']:
        raise ValueError('Missing or extra archived records')
    for _ in archive_rows(directory, snapshot):
        pass
    return snapshot


def restore_ledger(directory, target):
    directory, target = Path(directory).resolve(), Path(target).resolve()
    if target.exists() or Path(str(target) + '-wal').exists() or Path(str(target) + '-shm').exists():
        raise ValueError('Refusing to replace an existing runtime ledger')
    snapshot = verify_archive(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=target.parent, suffix='.sqlite3')
    os.close(fd)
    temporary = Path(temp_name)
    db = sqlite3.connect(temporary)
    try:
        with db:
            db.execute('CREATE TABLE state(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL)')
            db.execute('CREATE TABLE journal(seq INTEGER PRIMARY KEY,payload TEXT NOT NULL,previous_hash TEXT NOT NULL,record_hash TEXT NOT NULL)')
            db.executemany('INSERT INTO journal VALUES(?,?,?,?)', archive_rows(directory, snapshot))
            db.execute('INSERT INTO state VALUES(1,?)', (snapshot['state_payload'],))
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Restored SQLite is inconsistent')
        db.close()
        # Exclusive link avoids overwriting a ledger created concurrently.
        os.link(temporary, target)
    finally:
        db.close()
        temporary.unlink(missing_ok=True)
    return {'records': snapshot['journal_records'], 'last_hash': snapshot['last_hash'],
            'state_sha256': snapshot['state_sha256']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('export', 'restore'):
        command = commands.add_parser(name)
        command.add_argument('source', type=Path)
        command.add_argument('destination', type=Path)
    verify = commands.add_parser('verify')
    verify.add_argument('source', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        result = export_ledger(args.source, args.destination)
    elif args.command == 'restore':
        result = restore_ledger(args.source, args.destination)
    else:
        result = verify_archive(args.source)
        result.pop('state_payload')
    print(json.dumps(result, ensure_ascii=False, indent=2))

