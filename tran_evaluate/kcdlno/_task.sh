#!/usr/bin/env bash
# Thin per-dataset alias for the verified tran_evaluate/kcdno launchers.
set -euo pipefail

launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
task="${1:?dataset is required}"
shift

usage() {
    cat >&2 <<EOF
Usage: $task.sh train_eval [all|off|lrsa_matched] [options]
       $task.sh train [options]
       $task.sh eval --kcdno-run-dir RUN_DIR [options]
Append --dry-run to preview without Python/data access.
Defaults: model kcdno, profile kcdno_v1, history all.
See tran_evaluate/kcdlno/README.md for dataset paths and evaluation limits.
EOF
}

(($#)) || { usage; exit 0; }
# Handle help here: original experiment entries may read data at top level.
for arg in "$@"; do
    case "$arg" in -h|--help) usage; exit 0 ;; esac
done

case "$1" in
    train|eval)
        action="$1"
        shift
        exec bash "$launcher/../kcdno/$task.sh" "$action" "$@"
        ;;
    train_eval)
        shift
        ;;
    --*)
        # Options alone imply train_eval. Preserve the first option verbatim,
        # especially --dry-run, which must never be consumed here.
        ;;
    *)
        usage
        exit 2
        ;;
esac

variant=all
if (($#)) && [[ "$1" == all || "$1" == off || "$1" == lrsa_matched ]]; then
    variant="$1"
    shift
fi
exec bash "$launcher/../kcdno/train_eval.sh" "$task" "$variant" "$@"
