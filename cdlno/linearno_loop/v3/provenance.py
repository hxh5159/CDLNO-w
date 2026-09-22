"""V3 source fingerprint layered on the accepted V1 provenance pins."""
import ast
import hashlib
from pathlib import Path

from linearno_loop.contracts import digest


def provenance(config, *, task=None):
    root = Path(__file__).resolve().parents[3]
    # The legacy provenance projector intentionally rejects any routing-source
    # mutation.  LAA6 is precisely such an additive dispatch, so V3 records a
    # fresh source fingerprint while retaining the frozen reference pins.
    from linearno_loop.schema import REFERENCE_PINS
    from cdlno.experiment import Experiment
    environment = Experiment._environment()
    try:
        commit = __import__('subprocess').check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, stderr=__import__('subprocess').DEVNULL,
            text=True).strip()
        dirty = bool(__import__('subprocess').check_output(
            ['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=root,
            stderr=__import__('subprocess').DEVNULL))
    except (OSError, __import__('subprocess').CalledProcessError):
        commit, dirty = ('0' * 40), True
    paths = sorted({*(root / "linearno_loop" / "v3").glob("*.py"),
                    *(root / "cdlno" / "linearno_loop" / "v3").glob("*.py")})
    source = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    syntax = {str(path.relative_to(root)): ast.dump(ast.parse(path.read_text()))
              for path in paths}
    result = dict(target_sha=commit, base_commit=commit, dirty=dirty,
                  **REFERENCE_PINS, paper_version="2511.06294v3",
                  code_version="loop-linearno-latent-adapter-LAA6-v3",
                  command=list(__import__('sys').argv), environment=environment)
    result.update(config_schema_version=3, metadata_schema_version=3,
                  residual_scaling_version="2606.18524v1",
                  code_version="loop-linearno-latent-adapter-LAA6-v3",
                  source_sha256=digest(dict(v3=source)),
                  normalized_patch_sha256=digest(dict(v3=syntax)))
    return result
