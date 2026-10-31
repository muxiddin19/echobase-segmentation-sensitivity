"""Append one attempt to the run journal (and the GPU ledger when gpu_h > 0). Never rewrites rows.

    python tools/log_run.py --run-id e1/baseline/s1/train --status ok --gpu-h 4.1 --metric val_acc --value 0.812
    python tools/log_run.py --run-id e0/timing --experiment e0 --status ok --gpu-h 2.0 --note "timing test"
"""
import argparse
import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
JOURNAL = REPORTS / "run_journal.tsv"
LEDGER = REPORTS / "gpu_ledger.csv"
COLUMNS = ["run_id", "date", "experiment", "arm", "seed", "git_sha", "config", "status",
           "gpu_h", "metric", "value", "note"]
STATUSES = {"launched", "ok", "failed", "killed"}


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() or "nogit"


def read_journal() -> list[dict]:
    if not JOURNAL.exists():
        return []
    with open(JOURNAL, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append(row: dict) -> None:
    if row["status"] not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    parts = row["run_id"].split("/")
    row.setdefault("experiment", parts[0])
    row.setdefault("arm", parts[1] if len(parts) > 1 else "")
    row.setdefault("seed", parts[2].lstrip("s") if len(parts) > 2 else "")
    row.setdefault("date", date.today().isoformat())
    row.setdefault("git_sha", git_sha())
    clean = {c: str(row.get(c, "") or "").replace("\t", " ").replace("\n", " ") for c in COLUMNS}
    REPORTS.mkdir(parents=True, exist_ok=True)
    new = not JOURNAL.exists()
    with open(JOURNAL, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS, delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(clean)
    gpu_h = float(row.get("gpu_h") or 0)
    if gpu_h > 0:
        new = not LEDGER.exists()
        with open(LEDGER, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["date", "experiment", "run_id", "gpu_h"])
            w.writerow([clean["date"], clean["experiment"], clean["run_id"], f"{gpu_h:.3f}"])


def gpu_h_used() -> float:
    if not LEDGER.exists():
        return 0.0
    with open(LEDGER, newline="", encoding="utf-8") as f:
        return sum(float(r["gpu_h"] or 0) for r in csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--status", required=True, choices=sorted(STATUSES))
    ap.add_argument("--experiment")
    ap.add_argument("--arm")
    ap.add_argument("--seed")
    ap.add_argument("--config", default="")
    ap.add_argument("--gpu-h", default="0")
    ap.add_argument("--metric", default="")
    ap.add_argument("--value", default="")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    row = {"run_id": a.run_id, "status": a.status, "config": a.config, "gpu_h": a.gpu_h,
           "metric": a.metric, "value": a.value, "note": a.note}
    for k in ("experiment", "arm", "seed"):
        if getattr(a, k) is not None:
            row[k] = getattr(a, k)
    append(row)
    print(f"logged {a.run_id} {a.status}; GPU-h used {gpu_h_used():.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
