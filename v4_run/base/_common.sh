#!/usr/bin/env bash
set -euo pipefail

task="${1:?dataset task is required}"
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
delegate="$repo_root/v4_run/full/_common.sh"
[[ -f "$delegate" ]] || { echo "v4 full wrapper not found: $delegate" >&2; exit 2; }

# This directory is reserved for the static-temperature ablation.  Reject an
# accidental dynamic mode instead of silently writing a misleading base run.
action="train_eval"
if [[ $# -gt 0 && "$1" != -* ]]; then
  action="$1"
  shift
fi
arguments=()
while (($#)); do
  case "$1" in
    --temperature-mode)
      [[ $# -ge 2 && "$2" == base ]] || { echo "v4/base requires --temperature-mode base" >&2; exit 2; }
      shift 2 ;;
    --temperature-mode=*)
      [[ "${1#*=}" == base ]] || { echo "v4/base requires --temperature-mode base" >&2; exit 2; }
      shift ;;
    *) arguments+=("$1"); shift ;;
  esac
done
# An explicit mode is also required for resume/eval: the metadata-first
# parser must reject a dynamic-temperature run selected through this script.
exec bash "$delegate" "$task" "$action" --temperature-mode base "${arguments[@]}"
