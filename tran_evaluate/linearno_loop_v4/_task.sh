#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task="${1:?task required}"
shift
exec "${CDLNO_PYTHON:-python}" -B "$here/launch.py" "$task" "$@"
