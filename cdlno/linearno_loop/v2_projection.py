"""Exact compatibility view for the reviewed LF5/LF6 version dispatch edits.

Only the five listed files at their exact approved SHA-256 are projected to the
pre-v2 base revision. Any mutation fails closed instead of being hidden from
the v1 pure/history provenance fingerprints.
"""

import hashlib
from pathlib import Path
import subprocess

BASE_REVISION = "80ebe42d5755fc58ac6b41e2f6a0512d601ac8a8"
EXPECTED = {
    # LAA6 extends this router with an explicit V3 branch.  The projection
    # still returns the immutable LF5 source bytes, so v1/v2 fingerprints do
    # not drift while the V3 hash records the live dispatch separately.
    "cdlno/linearno_loop/standard_entry.py": "923f70aa3fcf80e1b4dbb35308b52324c66bf380287cbf781c7ef59804d56e2e",
    "cdlno/linearno_loop/industrial_entry.py": "eca13a787add62e06637f6fbb9d8036ff7958041ab5fe30fd7d53c7c2d5ffe32",
    "cdlno/linearno_loop/industrial_state.py": "692019b3bd6e3932df22e45e16d113d2025654bae7edb1b316d33cd378e03acf",
    "cdlno/linearno_loop/air_entry.py": "9ebf6b7e20c0cbe5269926d57b854c742816b7bb6dc233ea0a69e1496351b096",
    "cdlno/linearno_loop/car_entry.py": "90fa8be7a781dcdafe18c0d5107f355894a82ef9f6d4a29433839aaf3d623141",
    "cdlno/linearno_loop/ll7_projection.py": "9dcf6c19ee1e1a32435f7d6194ba164394d2f8156a1cb56ce8b09fc57381b740",
}


def project(relative, text):
    if relative == 'cdlno/linearno/standard_entry.py':
        # The sole pure-family edit dispatches an explicitly selected V4 only.
        # Preserve its reviewed V3 bridge verbatim in all legacy fingerprints.
        before = ('        from linearno_loop.versioning import is_v3\n'
                  '        if is_v3(args._linearno_loop_config):\n')
        after = ('        from linearno_loop.versioning import is_v3, is_v4\n'
                 '        if is_v3(args._linearno_loop_config) or is_v4(args._linearno_loop_config):\n')
        if text.count(after) == 1:
            return text.replace(after, before)
        if text.count(before) != 1:
            raise ValueError('unrecognized V4 pure constructor bridge')
        return text
    expected = EXPECTED.get(relative)
    if expected is None:
        return text
    actual = hashlib.sha256(text.encode()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    base = subprocess.check_output(
        ["git", "show", f"{BASE_REVISION}:{relative}"], cwd=root
    ).decode()
    if actual == expected:
        return base
    # Compatibility projections compose from newest to oldest. Accept only the
    # exact already-projected base bytes; every other mutation still fails.
    if text == base:
        return text
    raise ValueError(f"unrecognized v2 dispatch source mutation: {relative}")
