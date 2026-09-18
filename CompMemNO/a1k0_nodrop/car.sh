#!/usr/bin/env bash
set -euo pipefail
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_train_eval.sh" car "$@"
