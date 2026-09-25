"""Fail-closed compatibility view for V5 shared dispatch additions.

Only exact reviewed V5 source bytes are projected to the immutable pre-V5
HEAD.  This keeps pure LinearNO and loop v1-v4 provenance stable while any
unreviewed mutation still fails instead of being hidden.
"""
import hashlib
from pathlib import Path
import subprocess


BASE_REVISION = "c721ed161f0b94e5293d7e43b7b55ef20ba48167"
LEGACY_BASE_REVISION = "80ebe42d5755fc58ac6b41e2f6a0512d601ac8a8"
EXPECTED = {
    "cdlno/linearno/standard_entry.py": "9ca6d801dac936b86c01b779c15dd1f3f14be284f92b3695a2c15ec78834d1b7",
    "cdlno/linearno_loop/air_entry.py": "43dca4a1c30a3b3228b0bd36ee81de1559069fe711b5d5114a24a2fbe41c5d94",
    "cdlno/linearno_loop/car_entry.py": "a0b6566d85447382d58762c3ee001d67140613e44fbe3bee1f24e40e917cc60b",
    "cdlno/linearno_loop/industrial_entry.py": "351d5787398009ce05ea02bdb555ac7d7f5a7064479d0ced806ccfcb119a6fec",
    "cdlno/linearno_loop/standard_entry.py": "3a180ea32b44f8dab74a9a30c7a04d375e413526109d38da512585dd7d303181",
    "cdlno/linearno_loop/versioning.py": "c273e9466cd47a02cefc29181486f60163a0da576dcf7546dba6040a0e61ae69",
    "linearno_loop/versioning.py": "d0d4f7643c274ffcc873f5a5c172701b218c50d7df3db5b5109fbb02d24761f8",
    "tran_evaluate/linearno_loop/recording.py": "a7e886fde3b9d6a2f01230b09ef0544ce8a36e40d1c67141b9d8496815481bb9",
    "tools/linearno_loop_accounting.py": "5a8055a12587d47be0148eef205b6e9066313a80e827e1358f55021654690885",
}
PREVIOUS_PROJECTION_HASHES = {
    "cdlno/linearno/standard_entry.py": {
        "25ab61cf275a9fdddb9533476c7a069b5130de2eaabbb26609ee0dc7c4280361"},
}


def project(relative, text):
    expected = EXPECTED.get(relative)
    if expected is None:
        return text
    root = Path(__file__).resolve().parents[2]
    base = subprocess.check_output(
        ["git", "show", f"{BASE_REVISION}:{relative}"], cwd=root).decode()
    accepted = [base]
    try:
        accepted.append(subprocess.check_output(
            ["git", "show", f"{LEGACY_BASE_REVISION}:{relative}"],
            cwd=root, stderr=subprocess.DEVNULL).decode())
    except subprocess.CalledProcessError:
        pass
    if text in accepted:
        return text
    actual = hashlib.sha256(text.encode()).hexdigest()
    if actual in PREVIOUS_PROJECTION_HASHES.get(relative, set()):
        return text
    if actual == expected:
        return base
    raise ValueError("unrecognized V5 shared source mutation: " + relative)
