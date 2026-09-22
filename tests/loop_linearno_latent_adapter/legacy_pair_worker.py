"""Fresh-process V1/V2 pair replay used by LAA5 compatibility tests."""
import json
from pathlib import Path
import sys


def main():
    version, directory, stem = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if version == 1:
        from cdlno.linearno_loop import checkpoint
    elif version == 2:
        from cdlno.linearno_loop.v2 import checkpoint
    else:
        raise ValueError("legacy version must be 1 or 2")
    metadata, weights = checkpoint.read_pair(directory / "checkpoints" / (stem + ".json"))
    print(json.dumps(dict(version=version, config_hash=metadata["config_hash"],
                          keys=list(weights), format=checkpoint.FORMAT)))


if __name__ == "__main__": main()
