#!/usr/bin/env bash
# Offline reports only. Paths follow this checkout, independent of caller cwd.
set -euo pipefail
show_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${CDLNO_PYTHON:-python}" -B "$show_dir/_report.py" Car-Design-ShapeNetCar "$@"
