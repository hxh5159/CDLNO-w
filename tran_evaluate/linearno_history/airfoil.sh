#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:?train/resume/eval required}"; shift
case "$action" in
  train) exec bash "$launcher/../linearno/airfoil_train.sh" --linearno-fair-run 1 "$@" ;;
  resume) exec bash "$launcher/../linearno/airfoil_train.sh" --resume "$@" ;;
  eval) exec bash "$launcher/../linearno/airfoil_eval.sh" "$@" ;;
  *) printf 'Expected train/resume/eval\n' >&2; exit 2 ;;
esac
