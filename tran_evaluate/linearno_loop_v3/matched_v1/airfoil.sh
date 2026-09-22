#!/usr/bin/env bash
set -euo pipefail
v3_task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$v3_task_dir/../_common.sh" matched_v1 airfoil "$@"
