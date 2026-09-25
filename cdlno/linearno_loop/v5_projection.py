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
    "cdlno/linearno_loop/air_entry.py": "2180dbe9e7458b9689a8df72f3ee44eb74e2d9c0de6f1dbed8ef25d59a69e117",
    "cdlno/linearno_loop/car_entry.py": "6cdcf9db726ad05af006a74c4b11b2f2db1ac18512ffadf04ae18860d26abee1",
    "cdlno/linearno_loop/industrial_entry.py": "c9b8ece64df44913bf7c02bb2009a4ecf13bd162f386e57939fcccb6bbbc1ddc",
    "cdlno/linearno_loop/standard_entry.py": "c9f62c08d450a957179bd6971e2435810dfb4e23d853b8e93c2509cf139a8d0a",
    "cdlno/linearno_loop/versioning.py": "c273e9466cd47a02cefc29181486f60163a0da576dcf7546dba6040a0e61ae69",
    "linearno_loop/versioning.py": "1e186a2b1e57234b3f662cb1a208cf8948859e241668fbfb261efb192a72b8f2",
    "tran_evaluate/linearno_loop/recording.py": "a7e886fde3b9d6a2f01230b09ef0544ce8a36e40d1c67141b9d8496815481bb9",
    "tools/linearno_loop_accounting.py": "94f07a367e0b0aaa3fb83cd286211c1db73f8cf2960bfbdd19119aafe61ff47b",
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
