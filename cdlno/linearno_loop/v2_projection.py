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
    # This path uses the dedicated exact bridge projection below.  Listing it
    # here also lets older LL9R freeze checks route the file through this
    # fail-closed projector instead of hashing the live V5 branch directly.
    "cdlno/linearno/standard_entry.py": "9ca6d801dac936b86c01b779c15dd1f3f14be284f92b3695a2c15ec78834d1b7",
    # LAA6 extends this router with an explicit V3 branch.  The projection
    # still returns the immutable LF5 source bytes, so v1/v2 fingerprints do
    # not drift while the V3 hash records the live dispatch separately.
    "cdlno/linearno_loop/standard_entry.py": "c9f62c08d450a957179bd6971e2435810dfb4e23d853b8e93c2509cf139a8d0a",
    "cdlno/linearno_loop/industrial_entry.py": "c9b8ece64df44913bf7c02bb2009a4ecf13bd162f386e57939fcccb6bbbc1ddc",
    "cdlno/linearno_loop/industrial_state.py": "692019b3bd6e3932df22e45e16d113d2025654bae7edb1b316d33cd378e03acf",
    "cdlno/linearno_loop/air_entry.py": "2180dbe9e7458b9689a8df72f3ee44eb74e2d9c0de6f1dbed8ef25d59a69e117",
    "cdlno/linearno_loop/car_entry.py": "6cdcf9db726ad05af006a74c4b11b2f2db1ac18512ffadf04ae18860d26abee1",
    "cdlno/linearno_loop/ll7_projection.py": "9dcf6c19ee1e1a32435f7d6194ba164394d2f8156a1cb56ce8b09fc57381b740",
}

# Exact bytes produced by the preceding V5 projection.  Keeping these
# separate from EXPECTED lets historical V2 tests audit the current reviewed
# files while this projector still validates every intermediate layer.
PROJECTED_EXPECTED = {
    "cdlno/linearno_loop/standard_entry.py": "923f70aa3fcf80e1b4dbb35308b52324c66bf380287cbf781c7ef59804d56e2e",
    "cdlno/linearno_loop/industrial_entry.py": "eca13a787add62e06637f6fbb9d8036ff7958041ab5fe30fd7d53c7c2d5ffe32",
    "cdlno/linearno_loop/industrial_state.py": "692019b3bd6e3932df22e45e16d113d2025654bae7edb1b316d33cd378e03acf",
    "cdlno/linearno_loop/air_entry.py": "9ebf6b7e20c0cbe5269926d57b854c742816b7bb6dc233ea0a69e1496351b096",
    "cdlno/linearno_loop/car_entry.py": "90fa8be7a781dcdafe18c0d5107f355894a82ef9f6d4a29433839aaf3d623141",
    "cdlno/linearno_loop/ll7_projection.py": "9dcf6c19ee1e1a32435f7d6194ba164394d2f8156a1cb56ce8b09fc57381b740",
}


def project(relative, text):
    from .v5_projection import project as v5_source
    try:
        text = v5_source(relative, text)
    except ValueError as error:
        raise ValueError(
            f"unrecognized v2 dispatch source mutation: {relative}"
        ) from error
    if relative == 'cdlno/linearno/standard_entry.py':
        # V3 first added this constructor bridge and V4/V5 only extended its
        # explicit version guard.  Pure/v1/v2 archives predate the complete
        # bridge, so their compatibility view must remove the exact reviewed
        # block instead of retaining its V3 spelling.
        root = Path(__file__).resolve().parents[2]
        legacy = subprocess.check_output(
            ["git", "show", f"{BASE_REVISION}:{relative}"], cwd=root
        ).decode()
        if text == legacy:
            return text
        v4_guard = ('        from linearno_loop.versioning import is_v3, is_v4\n'
                    '        if is_v3(args._linearno_loop_config) or is_v4(args._linearno_loop_config):\n')
        v3_guard = ('        from linearno_loop.versioning import is_v3\n'
                    '        if is_v3(args._linearno_loop_config):\n')
        if text.count(v4_guard) == 1:
            text = text.replace(v4_guard, v3_guard)
        bridge = (
            "    # The unchanged six exp files import this helper directly.  V3 keeps its\n"
            "    # own constructor contract and exposes a compatibility bridge only when\n"
            "    # that explicit family is selected; v1/v2 continue through the exact\n"
            "    # historical branch below.\n"
            "    if hasattr(args, '_linearno_loop_config'):\n"
            "        from linearno_loop.versioning import is_v3\n"
            "        if is_v3(args._linearno_loop_config):\n"
            "            from cdlno.linearno_loop.standard_entry import model_kwargs as v3_kwargs\n"
            "            return v3_kwargs(args, **grid)\n"
        )
        if text.count(bridge) != 1:
            raise ValueError('unrecognized V3/V4 pure constructor bridge')
        projected = text.replace(bridge, '')
        if projected != legacy:
            raise ValueError('V3/V4 pure constructor projection did not reach legacy source')
        return projected
    expected = PROJECTED_EXPECTED.get(relative)
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
