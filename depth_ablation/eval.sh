#!/usr/bin/env bash
# Evaluate an already-trained LinearNO run at one depth.
# Usage: bash depth_ablation/eval.sh TASK DEPTH --seed N [--gpu N] [entry options]
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$here/_depth.sh" "$1" "$2" eval "${@:3}"
