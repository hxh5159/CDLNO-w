"""V2 source fingerprint layered on the accepted v1 provenance pins."""

import ast
import hashlib
from pathlib import Path

from linearno_loop.contracts import digest


def provenance(config, *, task=None):
    if task is None:
        from cdlno.linearno_loop.provenance import provenance as base
        result = base()
    else:
        from cdlno.linearno_loop.industrial_state import provenance as base
        result = base(task)
    root = Path(__file__).resolve().parents[3]
    paths = sorted({*(root / "linearno_loop" / "v2").glob("*.py"),
                    *(root / "cdlno" / "linearno_loop" / "v2").glob("*.py")})
    source = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    syntax = {str(path.relative_to(root)): ast.dump(ast.parse(path.read_text())) for path in paths}
    result.update(config_schema_version=2, metadata_schema_version=2,
                  code_version="loop-linearno-ffn-LF7-v2",
                  source_sha256=digest(dict(v1=result["source_sha256"], v2=source)),
                  normalized_patch_sha256=digest(dict(v1=result["normalized_patch_sha256"], v2=syntax)))
    return result

