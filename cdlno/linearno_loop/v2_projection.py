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
    "cdlno/linearno_loop/standard_entry.py": "629ce1b8db77b714c89067a541482596c3719eb0ee89aa1e9d8d4d6d843426a1",
    "cdlno/linearno_loop/industrial_entry.py": "5243e59ac0fe09aa9b9f35c07be8eb0a351fc032646cbea66d5e6e2a4e999352",
    "cdlno/linearno_loop/industrial_state.py": "692019b3bd6e3932df22e45e16d113d2025654bae7edb1b316d33cd378e03acf",
    "cdlno/linearno_loop/air_entry.py": "a3b12497d0a19a8f745e183dff75361a285ee9618a17786e129571253ddeba83",
    "cdlno/linearno_loop/car_entry.py": "de0c772c19b10aa51fc17d443ac12e5a65905316c73b2affde6a8f7b941e1828",
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
