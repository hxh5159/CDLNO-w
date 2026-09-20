"""Exact reviewed LL9R test edits, compared back to the immutable LL1 inventory."""
import json
from pathlib import Path


def project_test_source(relative, raw):
    path=Path(__file__).resolve().parents[1]/'docs/loop_linearno_audit/ll9r/test-source-repairs.json'
    changes=json.loads(path.read_text())
    for row in reversed(changes.get(relative, [])):
        after=row['after'].encode();before=row['before'].encode()
        if raw.count(after)!=1:
            raise AssertionError('unrecognized LL9R test change: '+relative)
        raw=raw.replace(after,before)
    return raw
