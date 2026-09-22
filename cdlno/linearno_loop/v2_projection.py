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
    "cdlno/linearno_loop/standard_entry.py": "b35c2ea89eff484a39dd1adceaee594cba9245614a18686fdceb6e4744cb70d3",
    "cdlno/linearno_loop/industrial_entry.py": "9d7d0403105b509ac801e429a95b46b0c906fb872ed3f11bafb8cc8a9fe56823",
    "cdlno/linearno_loop/industrial_state.py": "692019b3bd6e3932df22e45e16d113d2025654bae7edb1b316d33cd378e03acf",
    "cdlno/linearno_loop/air_entry.py": "9571d3603938b6fa3d28126f307e0de9fec6646c078eff5085363118813389a4",
    "cdlno/linearno_loop/car_entry.py": "8f80b1d96f23173428da8e2344144be3263b31534b5f6ebf97277e635081245e",
    "cdlno/linearno_loop/ll7_projection.py": "9dcf6c19ee1e1a32435f7d6194ba164394d2f8156a1cb56ce8b09fc57381b740",
}


def project(relative, text):
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
