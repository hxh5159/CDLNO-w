"""Summarize current read-only evidence; never update old fixtures or tests."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def method_id(label):
    return re.search(r"\(([\w.]+)\)", label).group(1)


def regressions():
    rows = read(OUT / "regressions/regression-results.json")
    prior = read(ROOT / "docs/loop_linearno_ffn_audit/lf7/regression-results.json")
    prior_failed = {t for row in prior["rows"] for t in row["failed_methods"]}
    prior_skipped = {t for row in prior["rows"] for t in row["skipped_methods"]}
    failed = set()
    skipped = set()
    cases = []
    for row in rows:
        for category in ("errors", "failures"):
            for label, traceback in row.get(category, []):
                method = method_id(label)
                failed.add(method)
                cases.append(dict(module=row["module"], method=method, label=label,
                                  category=category, traceback=traceback,
                                  already_failed_in_LF7=method in prior_failed))
        skipped.update(method_id(label) for label, why in row.get("skipped", []))
    total = sum(row.get("tests", 0) for row in rows)
    result = dict(
        modules=len(rows), tests=total, passed=total-len(failed)-len(skipped),
        failed_methods=len(failed), skipped_methods=len(skipped),
        failure_subcases=len(cases), failed_method_ids=sorted(failed),
        new_failed_methods=sorted(failed-prior_failed),
        no_longer_failed_methods=sorted(prior_failed-failed),
        new_skipped_methods=sorted(skipped-prior_skipped),
        no_longer_skipped_methods=sorted(prior_skipped-skipped),
        incomplete=[row["module"] for row in rows if "tests" not in row],
        per_module_seconds={row["module"]:row["seconds"] for row in rows},
        summed_module_seconds=sum(row["seconds"] or 0 for row in rows),
        elapsed_note="Module wall durations measured by original runner; three workers overlap. Do not treat their sum as overall wall time.",
        skipped_reasons=dict(collections.Counter(why for row in rows for _,why in row.get("skipped", []))),
        failed_method_groups={row["module"]:sorted({method_id(t) for t,_ in row.get("errors", [])+row.get("failures", [])}) for row in rows if row["exit_code"]},
        historical_reference="docs/loop_linearno_ffn_audit/lf7/regression-results.json",
    )
    result["baseline_status"] = "PASS_WITH_INHERITED_FAILURES" if not result["incomplete"] and not result["new_failed_methods"] and not result["new_skipped_methods"] else "PARTIAL"
    write("regression-summary.json", result)
    write("regression-failure-details.json", cases)
    return result


def checks():
    rows=[]
    start=time.monotonic()
    # The manifest is the complete frozen inventory, not an inferred package list.
    manifest=read(OUT/"start-manifest.json")
    python_files=[ROOT/name for name in manifest["files"] if name.endswith(".py")]
    python_files.extend(OUT.glob("*.py"))
    errors=[]
    for path in python_files:
        try: compile(path.read_bytes(),str(path),"exec")
        except Exception as error: errors.append(dict(path=str(path.relative_to(ROOT)),error=str(error)))
    rows.append(dict(command="compile(file bytes, path, exec), no bytecode writes",files=len(python_files),errors=errors,seconds=time.monotonic()-start))
    shells=[ROOT/"path.sh",*sorted((ROOT/"tran_evaluate/linearno_loop").glob("*.sh"))]
    for path in shells:
        start=time.monotonic()
        result=subprocess.run(["bash","-n",str(path)],cwd=ROOT,capture_output=True,text=True)
        rows.append(dict(command=["bash","-n",str(path.relative_to(ROOT))],exit_code=result.returncode,stderr=result.stderr,seconds=time.monotonic()-start))
    start=time.monotonic()
    result=subprocess.run(["git","diff","--check"],cwd=ROOT,capture_output=True,text=True)
    rows.append(dict(command=["git","diff","--check"],exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,seconds=time.monotonic()-start))
    write("static-checks.json",rows)
    return rows


def freeze():
    start=read(OUT/"start-manifest.json")
    changed=[];missing=[]
    for name, before in start["files"].items():
        path=ROOT/name
        if not path.exists():missing.append(name);continue
        after=hashlib.sha256(path.read_bytes()).hexdigest()
        if before["sha256"]!=after: changed.append(dict(path=name,before=before["sha256"],after=after,classification=before["classification"]))
    git=lambda *args:subprocess.check_output(["git",*args],cwd=ROOT,text=True)
    current=set()
    for args in (("ls-files","-z"),("ls-files","--others","--exclude-standard","-z"),("ls-files","--others","--ignored","--exclude-standard","-z")):
        current.update(filter(None,git(*args).split("\0")))
    new=sorted(current-set(start["files"]))
    generated=read(OUT/"generated-output-inventory.json")
    known_generated={row['path']:row for row in generated}
    for name,row in known_generated.items():
        assert name not in start['files']
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==row['sha256']
    allowed=lambda name:name.startswith("docs/loop_linearno_latent_adapter_audit/laa0/") or name in {
        "docs/LOOP_LINEARNO_LATENT_ADAPTER_REFERENCE_AUDIT.md",
        "docs/LOOP_LINEARNO_LATENT_ADAPTER_IMPLEMENTATION_STATUS.md"} or name in known_generated
    result=dict(start_count=len(start["files"]),compared=len(start["files"])-len(missing),
                changed=changed,missing=missing,new_files=new,
                synthetic_test_sidecars=generated,
                unexpected_new_files=[name for name in new if not allowed(name)],
                head=git("rev-parse","HEAD").strip(),branch=git("branch","--show-current").strip(),
                status=git("status","--short"),diff=git("diff","--binary"),
                staged=git("diff","--cached","--binary"),
                initial_git_diff_preserved=git("diff","--binary")==start["git"]["diff"],
                initial_staging_preserved=git("diff","--cached","--binary")==start["git"]["staged"])
    result["status_result"]="PASS" if not changed and not missing and not result["unexpected_new_files"] and result["initial_git_diff_preserved"] and result["initial_staging_preserved"] else "PARTIAL"
    write("end-freeze.json",result)
    return result


if __name__=="__main__":
    regression=regressions();static=checks();frozen=freeze()
    print(json.dumps(dict(regression=regression["baseline_status"],passed=regression["passed"],
        failed=regression["failed_methods"],skipped=regression["skipped_methods"],
        new_failed=regression["new_failed_methods"],static_checks=len(static),
        static_errors=[row for row in static if row.get("errors") or row.get("exit_code",0)],
        freeze=frozen["status_result"],files_compared=frozen["compared"]),indent=2))
