#!/usr/bin/env bash
set -euo pipefail
v3_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
v3_repo_root="$(cd -- "$v3_launcher_dir/../.." && pwd)"
if [[ -f "$v3_repo_root/path.sh" ]]; then
    source "$v3_repo_root/path.sh"
fi
export CDLNO_REPO_ROOT="$v3_repo_root"
export CDLNO_PYTHON="${CDLNO_PYTHON:-${PYTHON_BIN:-python}}"
exec "$CDLNO_PYTHON" -B "$v3_launcher_dir/launcher.py" "$@"
