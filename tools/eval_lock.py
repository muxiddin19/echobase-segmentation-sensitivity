"""Freeze evaluation assets by sha256 so later edits are detectable (and blocked by the guard hook).

Run --write yourself (a human, outside Claude) right after the freeze gate (G1); commit the lock file.
Claude and the Stop hook only ever run --check.

    python tools/eval_lock.py --write   # human only: hash every LOCKED file that exists
    python tools/eval_lock.py --check   # exit 2 on any mismatch or missing file; exit 0 if no lock yet
    python tools/eval_lock.py --list    # show which files are (or would be) locked
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_FILE = ROOT / ".claude" / "eval.lock.json"

# Frozen at the freeze gate (G1): everything that decides a score. ADAPT this list per project: the metric code,
# the evaluation loop, the statistics, the fixed data split / eval set, the noise or perturbation grid, the
# preregistration, the job matrix and the loop's own instructions. A pattern matching no file blocks --write.
LOCKED = [
    "pilots/segmap.py",            # Pilot A: change mapping (split/merge/n:m/relabel)
    "pilots/metrics.py",           # Pilot A: strong baseline + wrong-reuse / over-recompute
    "pilots/guarantee_eval.py",    # Pilot B: labels used + violation rate
    "pilots/stats.py",             # meeting-level bootstrap CIs
    "pilots/verdict.py",           # PASS / FAIL / INCONCLUSIVE rules
    "pilots/run_pilotA.py",        # Pilot A evaluation loop
    "pilots/run_pilotB_labels.py", # Pilot B units, shared transcript, predicate scores
    "pilots/extract.py",           # the LLM column (prompt, model scoring) shared by both pilots
    "pilots/build_versions.py",    # how version 0 / version 1 are built (conditions)
    "pilots/ami.py",               # annotation parsing = the ground truth
    "tests/*.py",
    "configs/ami_meetings.txt",    # dev / eval split
    "configs/run_matrix.toml",
    "program.md",
    "program_pilotA.md",
    "program_pilotB.md",
    # EchoBase confirmatory phase (G1, PREREGISTRATION.md §10)
    "pilots/run_diar.py",          # W1 old side on AMI eval (pyannote 3.1, known speaker count)
    "echobase/llm.py",             # prompts, answer sets, models, deterministic scoring
    "echobase/lineage.py",         # versions over the same reference words
    "echobase/workloads.py",       # W2 placebo
    "echobase/wiki.py",            # W4 sample, split, chunkers
    "echobase/drift.py",           # queries, the 0.02 guard, D
    "echobase/power_sim.py",       # median CI (BCa, percentile fallback) used by confirm.py
    "echobase/confirm.py",         # the confirmatory rules and fixed sequence
    "echobase/gate.py",            # the EchoBase G1 marker every eval/test script requires
    "echobase/score_versions.py",
    "echobase/score_wiki.py",
    "echobase/e0_determinism.py",  # eval spot check
    "echobase/rules.py",           # E5 (exploratory, settings fixed)
    "echobase/certify.py",
    "echobase/replay.py",
    "echobase/tests/*.py",
    "configs/frozen_lists.json",
    "PREREGISTRATION.md",
    "program_echobase.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def matched_files() -> list[Path]:
    files = set()
    for pattern in LOCKED:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def write() -> int:
    files = matched_files()
    if not files:
        print("No locked files exist yet; nothing written.", file=sys.stderr)
        return 1
    empty = [p for p in LOCKED if not any(q.is_file() for q in ROOT.glob(p))]
    if empty:                                    # a pattern with no file would silently leave that asset unlocked
        print("Refusing to write: these locked patterns match no file: " + ", ".join(empty), file=sys.stderr)
        return 2
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": {rel(p): sha256(p) for p in files},
    }
    LOCK_FILE.parent.mkdir(exist_ok=True)
    LOCK_FILE.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print(f"Locked {len(files)} files -> {rel(LOCK_FILE)}. Commit it.")
    return 0


def check(quiet: bool) -> int:
    if not LOCK_FILE.exists():
        if not quiet:
            print("No eval lock yet (expected before G1).")
        return 0
    lock = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    bad = []
    for path, digest in lock["files"].items():
        p = ROOT / path
        if not p.exists():
            bad.append(f"missing  {path}")
        elif sha256(p) != digest:
            bad.append(f"changed  {path}")
    if bad:
        print("EVAL LOCK VIOLATION (locked " + lock["created_utc"] + "):\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    if not quiet:
        print(f"Eval lock OK ({len(lock['files'])} files).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--list", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.write:
        return write()
    if args.check:
        return check(args.quiet)
    locked = set(json.loads(LOCK_FILE.read_text(encoding="utf-8"))["files"]) if LOCK_FILE.exists() else set()
    for p in matched_files():
        print(("[locked] " if rel(p) in locked else "[open]   ") + rel(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
