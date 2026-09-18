#!/usr/bin/env bash
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
action="${1:?train/resume/eval required}"; shift
case "$action" in
  train) exec bash "$launcher/../linearno/pipe_train.sh" --linearno-fair-run 1 "$@" ;;
  resume) exec bash "$launcher/../linearno/pipe_train.sh" --resume "$@" ;;
  eval) exec bash "$launcher/../linearno/pipe_eval.sh" "$@" ;;
  *) printf 'Expected train/resume/eval\n' >&2; exit 2 ;;
esac
