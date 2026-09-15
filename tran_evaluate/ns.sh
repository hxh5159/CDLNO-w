#!/usr/bin/env bash
set -euo pipefail
cdlno_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$cdlno_launcher_dir/_standard.sh" ns "$@"
