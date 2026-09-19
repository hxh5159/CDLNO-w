#!/usr/bin/env python3
"""Export existing PURE LinearNO kernel diagnostics, without importing a model.

Run this file directly: python /path/to/LinearNO-monitor/monitor/export_linearno_results.py
Only the Python standard library is required. Source artifacts are read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import tarfile
from datetime import datetime, timezone
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
PURE_CLASSES = {
    "model.LinearNO.Model",
    "cdlno.linearno.airfrans.AirfRANSLinearNO",
    "cdlno.linearno.shapenet.ShapeNetLinearNO",
}
PURE_KEYS = {"LinearNO", "LinearNO_Structured_Mesh_2D", "LinearNO_Irregular_Mesh"}
FEATURES = ("linearno_latent_attnres", "linearno_history_k_conditioning")
CONTROL_FILES = ("runtime_config.json", "monitor_config.json", "run_summary.json",
                 "monitor_error.json", "bootstrap_error.txt")
RUN_FILES = ("config.json", "linearno_run_manifest.json", "train_history.jsonl",
             "train_results.json", "eval_results.json")
META_FIELDS = ("family", "schema_version", "model_spec", "profile_spec", "objective_spec",
               "evaluation_spec", "provenance_spec", "config_hash", "metadata_hash")


def read_object(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def options(tokens):
    """Collect explicit long options, normalizing underscore aliases."""
    result = {}
    for index, token in enumerate(tokens):
        if token.startswith("--"):
            flag, sep, value = token.partition("=")
            if not sep:
                value = tokens[index + 1] if index + 1 < len(tokens) else None
            flag = flag.replace("_", "-")
            if flag == "--linearno-run-dir":
                flag = "--experiment-dir"
            result[flag] = value
    return result


def metadata_identity(doc):
    """Research/other-model evidence always vetoes a baseline launcher name."""
    evidence, errors = [], []

    def walk(value):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in ("family", "linearno_family"):
                if item == "linearno":
                    evidence.append(f"{key}=linearno")
                elif item is not None:
                    errors.append(f"{key}={item!r}")
            if key == "model" and isinstance(item, str):
                if item in PURE_KEYS or item == "linearno":
                    evidence.append(f"model={item}")
                else:
                    errors.append(f"model={item!r}")
            if key == "class_path":
                if item in PURE_CLASSES:
                    evidence.append(f"class_path={item}")
                else:
                    errors.append(f"non-baseline class_path={item!r}")
            if key == "innovation_spec" or (key == "architecture_extension" and item):
                errors.append(f"research metadata: {key}")
            if key == "feature_signature" and item != "A0K0":
                errors.append(f"feature_signature={item!r}")
            if key in FEATURES and item is not False and not (type(item) is int and item == 0):
                errors.append(f"{key}={item!r}")
            # Traverse configuration objects only; lists contain e.g. argv/hash inventories.
            if isinstance(item, dict):
                walk(item)
    walk(doc)
    return evidence, errors


def locate_run(flags, repo):
    value = flags.get("--experiment-dir") or flags.get("--linearno-run-dir")
    if not value:
        return None, "No explicit experiment directory in recorded command."
    path = Path(value).expanduser()
    direct = path if path.is_absolute() else repo / path
    if direct.is_dir():
        return direct.resolve(), "Recorded experiment directory."
    # Exact relocation of an old checkout's output subtree, never latest-run guessing.
    if "output" in path.parts:
        offset = path.parts.index("output")
        relocated = repo.joinpath(*path.parts[offset:])
        if relocated.is_dir():
            return relocated.resolve(), f"Relocated exact output suffix from {value}."
    return None, f"Recorded experiment directory unavailable: {value}"


def inspect_monitor(directory, repo):
    row = dict(directory=str(directory), included=False, warnings=[], evidence=[])
    configs = []
    try:
        for name in ("runtime_config.json", "monitor_config.json"):
            if (directory / name).is_file():
                configs.append(read_object(directory / name))
        if ((directory / "diagnostics.jsonl").exists() or any(
                "max_snapshots" in cfg or "max_points" in cfg for cfg in configs)):
            row["reason"] = "History/A-K diagnostic format; excluded."
            return row
        if not configs or any("capture_every_validations" not in cfg for cfg in configs):
            row["reason"] = "Not the pure monitor.run format, or incomplete configuration."
            return row
        commands = [cfg.get("command") for cfg in configs]
        if any(not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd)
               for cmd in commands) or any(cmd != commands[0] for cmd in commands):
            row["reason"] = "Missing/conflicting recorded commands."
            return row
        command = row["command"] = commands[0]
        flags = options(command)
        for key in FEATURES:
            flag = "--" + key.replace("_", "-")
            if flag in flags and flags[flag] != "0":
                row["reason"] = f"Research/invalid flag {flag}={flags[flag]!r}; excluded."
                return row
        if any("a1k0_nodrop" in token.lower() for token in command):
            row["reason"] = "A1K0 launcher; excluded."
            return row
        for flag in ("--model", "--cfd-model"):
            if flag in flags:
                if flags[flag] not in PURE_KEYS:
                    row["reason"] = f"Non-baseline model option {flag}={flags[flag]!r}."
                    return row
                row["evidence"].append(f"command {flag}={flags[flag]}")
        # This exact launcher directory is the pure family entry, not linearno_history.
        if any("tran_evaluate/linearno/" in token for token in command):
            row["evidence"].append("pure LinearNO launcher")
        run, note = locate_run(flags, repo)
        row["run_directory"] = str(run) if run else None
        row["run_resolution"] = note
        row["run_metadata"] = {}
        for name in ("architecture.json", "config.json", "linearno_run_manifest.json"):
            if run is not None and (run / name).is_file():
                doc = read_object(run / name)
                evidence, errors = metadata_identity(doc)
                if errors:
                    row["reason"] = f"{name}: " + "; ".join(errors)
                    return row
                row["evidence"].extend(evidence)
                row["run_metadata"][name] = doc
        if not row["evidence"]:
            row["reason"] = "Model identity unverified; no pure model metadata/command evidence."
            return row
        if "architecture.json" not in row["run_metadata"]:
            row["warnings"].append("No architecture.json; identity relies on other recorded evidence.")
        if run is None:
            row["warnings"].append(note)
        row.update(included=True, reason="Pure LinearNO; no contradictory research metadata/flags.")
    except (OSError, ValueError, TypeError) as error:
        row["reason"] = f"Unreadable/inconsistent provenance: {error}"
    return row


def discover(root, repo):
    rows = []
    for parent, dirs, filenames in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in {
            "checkpoints", "weights", ".git", "__pycache__", "snapshots", "monitor_exports"})
        if {"runtime_config.json", "monitor_config.json", "diagnostics.jsonl"} & set(filenames):
            rows.append(inspect_monitor(Path(parent), repo))
    return rows


def csv_matrix(path):
    """Mean over captured samples and heads, matching rho_mean in the monitor."""
    totals, counts = {}, {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            pair = (int(row["layer_i"]), int(row["layer_j"]))
            value = float(row["rho"])
            if min(pair) < 1 or not math.isfinite(value):
                raise ValueError("invalid layer index or nonfinite rho")
            totals[pair] = totals.get(pair, 0.0) + value
            counts[pair] = counts.get(pair, 0) + 1
    if not totals:
        raise ValueError("empty similarity.csv")
    depth = max(max(pair) for pair in totals)
    if len(totals) != depth * depth or len(set(counts.values())) != 1:
        raise ValueError("incomplete layer-pair matrix")
    return [[totals[i, j] / counts[i, j] for j in range(1, depth + 1)]
            for i in range(1, depth + 1)]


def summarize_snapshot(snapshot):
    text = [f"Snapshot: {snapshot.name}"]
    try:
        meta = read_object(snapshot / "metadata.json")
        text.append("metadata: " + json.dumps(meta, ensure_ascii=False, sort_keys=True))
        matrix = csv_matrix(snapshot / "similarity.csv")
        text.append("rho_mean (mean over captured samples and heads):")
        text.extend("  " + " ".join(f"{v:.8f}" for v in row) for row in matrix)
        off = [v for i, row in enumerate(matrix) for j, v in enumerate(row) if i != j]
        if off:
            text.append(f"off-diagonal: mean={statistics.fmean(off):.8f}, "
                        f"min={min(off):.8f}, max={max(off):.8f}")
    except (OSError, ValueError, KeyError) as error:
        text.append(f"SUMMARY UNAVAILABLE: {error}; available raw files are retained.")
    return text


def export(root, output, *, repo=REPO, plots="all", list_only=False):
    rows = discover(root, repo)
    included = [row for row in rows if row["included"]]
    for row in rows:
        print(f'{"INCLUDE" if row["included"] else "SKIP"}: {row["directory"]}\n  {row["reason"]}')
    if not included:
        raise ValueError("未找到可确认属于纯 LinearNO 的监测目录；不会打包研究模型。"
                         "请检查 --root 和上面的 SKIP 原因。")
    if list_only:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / f"pure_linearno_{stamp}_{uuid4().hex[:8]}"
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / "pure_linearno_monitor.tar.gz"
    manifest = dict(format="pure-linearno-monitor-export-v1", source_root=str(root),
                    created_at_utc=stamp, plots=plots, monitors=[], files=[])
    summary = ["PURE LinearNO propagation-kernel monitoring", f"Source: {root}",
               f"Included: {len(included)}; skipped: {len(rows)-len(included)}",
               "P_l=Q_l K_l^T; validation_index counts eval forwards, NOT epochs.",
               "All numerical snapshots retained. Summary shows first/last saved snapshot,",
               "which does NOT prove coverage of the final epoch/checkpoint."]

    with tarfile.open(archive, "x:gz") as bundle:
        def add_bytes(name, data, source=None):
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), int(datetime.now(timezone.utc).timestamp())
            bundle.addfile(info, io.BytesIO(data))
            manifest["files"].append(dict(path=name, source=source, size=len(data),
                sha256=hashlib.sha256(data).hexdigest()))

        def add_file(path, name):
            if path.is_file() and not path.is_symlink():
                add_bytes(name, path.read_bytes(), str(path))

        for row in rows:
            docs = row.pop("run_metadata", {})
            manifest["monitors"].append(row)
            if not row["included"]:
                continue
            directory = Path(row["directory"])
            relative = directory.relative_to(root)
            prefix = "monitors/" + (str(relative) if relative.parts else directory.name)
            snapshots = sorted(p for p in (directory / "snapshots").glob("validation_*")
                               if p.is_dir() and not p.is_symlink())
            row["snapshots"] = len(snapshots)
            row["numerical_snapshots"] = sum(any((p / n).is_file() for n in
                ("similarity_values.npz", "similarity.csv")) for p in snapshots)
            summary.extend(["", f"===== {relative} =====", "command: " + json.dumps(
                row["command"], ensure_ascii=False), f"Snapshots: {row['snapshots']}; "
                f"with numerical data: {row['numerical_snapshots']}"])
            summary.extend("WARNING: " + warning for warning in row["warnings"])
            for name in CONTROL_FILES:
                add_file(directory / name, f"{prefix}/{name}")
            for name in ("run_summary.json", "monitor_error.json", "bootstrap_error.txt"):
                if (directory / name).is_file():
                    summary.append(name + ": " + (directory / name).read_text())
            for snapshot in snapshots:
                for name in ("similarity_values.npz", "similarity.csv", "metadata.json"):
                    add_file(snapshot / name, f"{prefix}/snapshots/{snapshot.name}/{name}")
                if plots == "all" or (plots == "endpoints" and snapshot in (snapshots[0], snapshots[-1])):
                    for ext in ("png", "pdf", "svg"):
                        name = "similarity_heatmap." + ext
                        add_file(snapshot / name, f"{prefix}/snapshots/{snapshot.name}/{name}")
            for snapshot in dict.fromkeys(snapshots[:1] + snapshots[-1:]):
                summary.extend(summarize_snapshot(snapshot))
            if not row["numerical_snapshots"]:
                summary.append("WARNING: no captured numerical results; this is a diagnostic-only record.")
            if row["run_directory"]:
                run = Path(row["run_directory"])
                for name in RUN_FILES:
                    add_file(run / name, f"{prefix}/run_records/{name}")
                for path in sorted((run / "evaluations").glob("*/results.json")):
                    add_file(path, f"{prefix}/run_records/{path.relative_to(run)}")
                if "architecture.json" in docs:
                    # Do not copy optimizer/RNG/normalizer arrays embedded in the full sidecar.
                    subset = {k: docs["architecture.json"][k] for k in META_FIELDS
                              if k in docs["architecture.json"]}
                    data = dict(note="Selected fields from architecture.json, not a checkpoint",
                                source=str(run / "architecture.json"), metadata=subset)
                    add_bytes(f"{prefix}/run_records/architecture_summary.json",
                              json.dumps(data, ensure_ascii=False, indent=2).encode())
        skipped = [row for row in rows if not row["included"]]
        if skipped:
            summary.append("\nSkipped directories:")
            summary.extend(f'{row["directory"]}: {row["reason"]}' for row in skipped)
        text = "\n".join(summary) + "\n"
        add_bytes("summary.txt", text.encode("utf-8"))
        inventory = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        info = tarfile.TarInfo("manifest.json")
        payload = inventory.encode("utf-8")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))
    (destination / "summary.txt").write_text(text, encoding="utf-8")
    (destination / "manifest.json").write_text(inventory, encoding="utf-8")
    print(f"\n完成：{len(included)} 个纯 LinearNO 监测目录。")
    print(f"压缩包：{archive}\n可粘贴摘要：{destination / 'summary.txt'}")
    print(f"大小：{archive.stat().st_size / 1024**2:.2f} MiB；无权重/数据集文件。")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "output", help="监测结果搜索根目录")
    parser.add_argument("--output-dir", type=Path, default=REPO / "monitor_exports")
    parser.add_argument("--plots", choices=("all", "endpoints", "none"), default="all")
    parser.add_argument("--list-only", action="store_true", help="仅列出收集/跳过项，不创建文件")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"结果目录不存在：{args.root}")
    try:
        export(args.root.resolve(), args.output_dir.resolve(), plots=args.plots,
               list_only=args.list_only)
    except (OSError, ValueError) as error:
        parser.exit(2, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
