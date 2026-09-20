"""Reviewed post-snapshot revisions; all other hashes remain unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROWS = {r['path']: r for r in json.loads((ROOT/'docs/loop_linearno_audit/ll9r/frozen-revisions-v1.json').read_text())['rows']}


def expected_hash(path, original):
    if path not in ROWS:
        return original
    row = ROWS[path]
    if original == row['new_sha256']:
        return row['new_sha256']
    if original != row['old_sha256']:
        raise AssertionError('unrecognized historical document revision: ' + path)
    return row['new_sha256']
