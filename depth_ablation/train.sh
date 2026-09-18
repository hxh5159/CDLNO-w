#!/usr/bin/env bash
# Train LinearNO at one depth on one dataset.
# Usage: bash depth_ablation/train.sh TASK DEPTH --seed N [--gpu N] [entry options]
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$here/_depth.sh" "$1" "$2" train "${@:3}"
