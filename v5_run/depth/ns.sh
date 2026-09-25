#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/../_common.sh" ns 1 "$@" \
  --topology custom --prefix-blocks 2 --core-blocks 6 \
  --loop-repeats 2 --suffix-blocks 2
