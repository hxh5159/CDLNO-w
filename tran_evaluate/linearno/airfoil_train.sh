#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$launcher/_static.sh" airfoil train "$@"
