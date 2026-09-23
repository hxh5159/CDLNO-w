#!/usr/bin/env bash
set -euo pipefail

# Download and validate the complete ShapeNet-Car dataset used by this checkout.
# No existing dataset directory is removed, moved, or overwritten by this script.
# ModelScope currently publishes four raw-only directories in addition to the
# 889 cached samples. A separate symlink view selects the exact raw/cache
# intersection expected by the repository loader while preserving the download.

readonly DATASET_ID="OneScience/ShapeNetCar"
readonly REMOTE_DATA_ROOT="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data"
readonly DEFAULT_PROJECT_ROOT="/inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/transolver/looplin-v3"

usage() {
    cat <<'USAGE'
Usage: bash download/download_car.sh [--verify-only]

Downloads ModelScope dataset OneScience/ShapeNetCar to:
  /inspire/hdd/project/urbanlowaltitude/yuanmeilu-253114050257/houwenzhe-drivaer/data/ShapeNetCar_ModelScope

Options:
  --verify-only  Do not download; validate the existing downloaded dataset.
  -h, --help     Show this help.

Optional environment overrides:
  CDLNO_DATA_ROOT              Parent data directory.
  CDLNO_REPO_ROOT              looplin-v3 checkout directory.
  CAR_MODELSCOPE_DOWNLOAD_DIR  Complete ModelScope download directory.
  CAR_DOWNLOAD_PYTHON          Python used to create an isolated ModelScope CLI venv.
  CAR_DOWNLOAD_VENV            Isolated ModelScope CLI venv path.
  CAR_DOWNLOAD_MIN_FREE_GIB    Required free space before a new download (default: 30).
USAGE
}

verify_only=0
case "${1:-}" in
    "") ;;
    --verify-only) verify_only=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
if (($# > 1)); then
    usage >&2
    printf 'Only one optional argument is accepted.\n' >&2
    exit 2
fi

data_root="${CDLNO_DATA_ROOT:-$REMOTE_DATA_ROOT}"
project_root="${CDLNO_REPO_ROOT:-$DEFAULT_PROJECT_ROOT}"
download_dir="${CAR_MODELSCOPE_DOWNLOAD_DIR:-$data_root/ShapeNetCar_ModelScope}"
environment_file="$download_dir/use_car_data.sh"
validation_python="${CDLNO_PYTHON:-python}"

mkdir -p "$data_root" "$download_dir"
data_root="$(cd -- "$data_root" && pwd -P)"
download_dir="$(cd -- "$download_dir" && pwd -P)"

find_dataset_root() {
    local candidate
    for candidate in \
        "$download_dir/data/mlcfd_data" \
        "$download_dir/mlcfd_data"
    do
        if [[ -d "$candidate/training_data" && -d "$candidate/preprocessed_data" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

if ((verify_only == 0)); then
    # A fresh complete download is about 18.3 GB. Keep headroom for temporary
    # files and filesystem accounting. Set CAR_DOWNLOAD_MIN_FREE_GIB=0 to
    # disable this guard when resuming a mostly complete download.
    if ! find_dataset_root >/dev/null 2>&1; then
        min_free_gib="${CAR_DOWNLOAD_MIN_FREE_GIB:-30}"
        [[ "$min_free_gib" =~ ^[0-9]+$ ]] || {
            printf 'CAR_DOWNLOAD_MIN_FREE_GIB must be a nonnegative integer.\n' >&2
            exit 2
        }
        free_kib="$(df -Pk "$data_root" | awk 'NR == 2 {print $4}')"
        required_kib=$((min_free_gib * 1024 * 1024))
        if ((free_kib < required_kib)); then
            printf 'Insufficient free space under %s: need at least %s GiB, have about %s GiB.\n' \
                "$data_root" "$min_free_gib" "$((free_kib / 1024 / 1024))" >&2
            printf 'For a partial resumed download, inspect it first and explicitly set CAR_DOWNLOAD_MIN_FREE_GIB.\n' >&2
            exit 1
        fi
    fi

    if command -v modelscope >/dev/null 2>&1; then
        modelscope_bin="$(command -v modelscope)"
    else
        download_python="${CAR_DOWNLOAD_PYTHON:-python3}"
        command -v "$download_python" >/dev/null 2>&1 || {
            printf 'Python for the isolated ModelScope download environment was not found: %s\n' "$download_python" >&2
            exit 1
        }
        download_venv="${CAR_DOWNLOAD_VENV:-$HOME/.venvs/modelscope-download}"
        if [[ ! -x "$download_venv/bin/python" ]]; then
            printf 'Creating isolated ModelScope download environment: %s\n' "$download_venv"
            "$download_python" -m venv "$download_venv"
        fi
        if [[ ! -x "$download_venv/bin/modelscope" ]]; then
            printf 'Installing ModelScope CLI in the isolated download environment.\n'
            "$download_venv/bin/python" -m pip install --upgrade pip
            "$download_venv/bin/python" -m pip install modelscope
        fi
        modelscope_bin="$download_venv/bin/modelscope"
    fi

    printf 'Dataset: %s\n' "$DATASET_ID"
    printf 'Download directory: %s\n' "$download_dir"
    "$modelscope_bin" download --dataset "$DATASET_ID" --local_dir "$download_dir"
fi

if ! dataset_root="$(find_dataset_root)"; then
    printf 'Downloaded dataset layout was not found. Expected one of:\n' >&2
    printf '  %s\n' "$download_dir/data/mlcfd_data" >&2
    printf '  %s\n' "$download_dir/mlcfd_data" >&2
    printf 'The download may be incomplete; rerun this script to resume it.\n' >&2
    exit 1
fi

source_raw_root="$dataset_root/training_data"
cache_root="$dataset_root/preprocessed_data"
raw_root="$dataset_root/training_data_cdlno_889"

command -v "$validation_python" >/dev/null 2>&1 || {
    printf 'Validation Python was not found: %s\n' "$validation_python" >&2
    exit 1
}

printf 'Building the non-destructive 889-sample raw-data compatibility view.\n'
"$validation_python" - "$source_raw_root" "$cache_root" "$raw_root" <<'PY'
from pathlib import Path
import os
import sys

source_raw = Path(sys.argv[1]).resolve()
cache = Path(sys.argv[2]).resolve()
view = Path(sys.argv[3]).absolute()
expected_counts = (100, 99, 97, 100, 100, 96, 100, 98, 99)
view.mkdir(parents=True, exist_ok=True)

raw_only_total = 0
for fold, expected_count in enumerate(expected_counts):
    source_fold = source_raw / f"param{fold}"
    cache_fold = cache / f"param{fold}"
    view_fold = view / f"param{fold}"
    if not source_fold.is_dir():
        raise SystemExit(f"Missing downloaded raw fold: {source_fold}")
    if not cache_fold.is_dir():
        raise SystemExit(f"Missing downloaded cache fold: {cache_fold}")

    source_samples = {path.name: path for path in source_fold.iterdir() if path.is_dir()}
    cache_samples = {path.name: path for path in cache_fold.iterdir() if path.is_dir()}
    if len(cache_samples) != expected_count:
        raise SystemExit(
            f"param{fold}: expected {expected_count} cached samples, "
            f"found {len(cache_samples)}"
        )
    missing_raw = sorted(set(cache_samples) - set(source_samples))
    if missing_raw:
        raise SystemExit(f"param{fold}: cache samples missing raw VTK directories: {missing_raw[:5]}")

    raw_only = sorted(set(source_samples) - set(cache_samples))
    raw_only_total += len(raw_only)
    if raw_only:
        print(f"param{fold}: ignoring raw-only directories without cache: {raw_only}")

    view_fold.mkdir(parents=True, exist_ok=True)
    existing = {path.name: path for path in view_fold.iterdir()}
    unexpected = sorted(set(existing) - set(cache_samples))
    if unexpected:
        raise SystemExit(
            f"Compatibility view contains unexpected entries under {view_fold}: {unexpected[:5]}"
        )

    # Sorted creation gives fresh downloads a deterministic directory insertion
    # order. The native loader still records its observed os.listdir order in
    # checkpoint metadata and enforces that order on resume/eval.
    for name in sorted(cache_samples):
        source = source_samples[name].resolve()
        destination = view_fold / name
        if os.path.lexists(destination):
            if not destination.is_symlink() or destination.resolve() != source:
                raise SystemExit(
                    f"Refusing to replace nonmatching compatibility entry: {destination}"
                )
        else:
            destination.symlink_to(source, target_is_directory=True)

print(f"Compatibility view ready: 889 samples; ignored {raw_only_total} raw-only directories")
PY

printf 'Validating raw data and preprocessed cache. This checks all 889 samples.\n'
"$validation_python" - "$raw_root" "$cache_root" <<'PY'
from pathlib import Path
import sys

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "NumPy is required for cache validation. Activate the existing "
        "training environment or set CDLNO_PYTHON to its Python executable."
    ) from error

raw = Path(sys.argv[1]).resolve()
cache = Path(sys.argv[2]).resolve()
expected_counts = (100, 99, 97, 100, 100, 96, 100, 98, 99)
cache_files = ("x", "y", "pos", "surf", "edge_index")
total = 0

for fold, expected_count in enumerate(expected_counts):
    raw_fold = raw / f"param{fold}"
    cache_fold = cache / f"param{fold}"
    if not raw_fold.is_dir():
        raise SystemExit(f"Missing raw fold: {raw_fold}")
    if not cache_fold.is_dir():
        raise SystemExit(f"Missing cache fold: {cache_fold}")

    raw_samples = {path.name: path for path in raw_fold.iterdir() if path.is_dir()}
    cache_samples = {path.name: path for path in cache_fold.iterdir() if path.is_dir()}
    if len(raw_samples) != expected_count:
        raise SystemExit(
            f"param{fold}: expected {expected_count} raw sample directories, "
            f"found {len(raw_samples)}"
        )
    if set(raw_samples) != set(cache_samples):
        missing_cache = sorted(set(raw_samples) - set(cache_samples))[:5]
        extra_cache = sorted(set(cache_samples) - set(raw_samples))[:5]
        raise SystemExit(
            f"param{fold}: raw/cache sample names differ; "
            f"missing cache examples={missing_cache}, extra cache examples={extra_cache}"
        )

    for name, sample in raw_samples.items():
        for filename in ("quadpress_smpl.vtk", "hexvelo_smpl.vtk"):
            path = sample / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"Missing or empty raw file: {path}")

        cached = cache_samples[name]
        arrays = {}
        for key in cache_files:
            path = cached / f"{key}.npy"
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f"Missing or empty cache file: {path}")
            try:
                arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as error:
                raise SystemExit(f"Cannot read cache array {path}: {error}") from error

        x, y, pos, surf, edge_index = (arrays[key] for key in cache_files)
        n = x.shape[0] if x.ndim == 2 else -1
        relative = sample.relative_to(raw)
        expected_shapes = {
            "x": (n, 7),
            "y": (n, 4),
            "pos": (n, 3),
            "surf": (n,),
        }
        for key, expected_shape in expected_shapes.items():
            if arrays[key].shape != expected_shape:
                raise SystemExit(
                    f"{relative}/{key}.npy: expected {expected_shape}, "
                    f"found {arrays[key].shape}"
                )
        if n <= 0:
            raise SystemExit(f"Empty point set: {relative}")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise SystemExit(
                f"{relative}/edge_index.npy: expected [2,E], found {edge_index.shape}"
            )
        if not all(np.issubdtype(arrays[key].dtype, np.number) for key in cache_files):
            raise SystemExit(f"Non-numeric cache dtype under {relative}")

    total += len(raw_samples)
    print(f"param{fold}: {len(raw_samples)} samples OK")

if total != 889:
    raise SystemExit(f"Expected 889 samples, found {total}")
print("ShapeNet-Car validation passed: 889/889 raw and cached samples")
PY

{
    printf '#!/usr/bin/env bash\n'
    printf '# Generated by download/download_car.sh; source before Car train/resume/eval.\n'
    printf 'export CDLNO_CAR_RAW_ROOT=%q\n' "$raw_root"
    printf 'export CDLNO_CAR_CACHE_ROOT=%q\n' "$cache_root"
} > "$environment_file"
chmod 0644 "$environment_file"

printf '\nShapeNet-Car is ready.\n'
printf 'Downloaded raw: %s\n' "$source_raw_root"
printf '889 raw view:   %s\n' "$raw_root"
printf 'Preprocessed:   %s\n' "$cache_root"
printf 'Environment:    %s\n' "$environment_file"
printf '\nRun the current efficient-v1 D12/SR/adapter training and evaluation with:\n'
printf '  cd %q\n' "$project_root"
printf '  source %q\n' "$environment_file"
printf '  bash succeed_run/lr_12_sr_low/car_train_eval.sh --seed 0 --gpu 0\n'
printf '\nThe launcher creates a new unique run and evaluates only after training succeeds.\n'
