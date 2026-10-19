"""Check the exact strategy sources before testing or importing any ledger."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_sources():
    hashes = json.loads((ROOT / 'frozen_source_sha256.json').read_text())
    for name, expected in hashes.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen strategy source changed: ' + name)
    return len(hashes)


if __name__ == '__main__':
    print(json.dumps({'frozen_sources_verified': verify_sources(), 'real_orders': False}))

