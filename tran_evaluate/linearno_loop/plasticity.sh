#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$launcher/../_common.sh"
exec "$CDLNO_PYTHON" -B "$launcher/launch.py" plasticity "$@"
