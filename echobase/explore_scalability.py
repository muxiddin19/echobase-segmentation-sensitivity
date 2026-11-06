"""D11 (a), post-hoc exploratory: how the maintenance replay scales with the number of changed cells. CPU only.

    OMP_NUM_THREADS=1 python -m echobase.explore_scalability --out reports/explore_scalability.json

Workload W1 (old = w1old, new = manual) on the 136 scale meetings (configs/ami_scale_done.txt) with the primary 1.5B
model's labels from the D7 scale cache (outputs/echobase/v2-scale-1.5B). Cell records, policies and audit are the
locked code (e5_maintenance.cells, replay.run_policy, certify.audit), unchanged; this script only times them.
- stage 1 (lineage, change map, rule derivation, anchoring) on nested prefixes of the seed-0 meeting order;
- stage 2 (one policy run, eps 0.05, delta 0.05) on the same prefixes and on the 136-meeting cell set replicated
  x2, x4, x8 (copies get new meeting ids: tests the algorithm's scaling, not new data), median of 5 seeds.
LLM time of each policy's calls uses the measured 1.5B throughput on TITAN Xp (reports/throughput.json, manual).
Rules fixed in D11 (reports/deviations.md) before the run.
"""
import argparse
import json
import platform
try:
    import resource  # Linux/macOS
except ImportError:  # Windows
    resource = None
import time
from pathlib import Path

import numpy as np

from . import e5_maintenance as e5
from .gate import require_g1
from .replay import run_policy

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("hash", "anchor_cert", "echobase", "echobase_conf")
EPS, DELTA, REPEATS = 0.05, 0.05, 5
PREFIXES = (8, 16, 32, 64, 136)
REPLICAS = (2, 4, 8)


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def peak_rss_mb():
    """Peak resident memory in MB (Linux ru_maxrss is in KB); on Windows the peak working set, or None."""
    if resource is not None:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    try:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20
    except Exception:
        return None


def stage2(cs):
    out = {}
    for pol in POLICIES:
        ts, calls, wrong = [], [], []
        for s in range(REPEATS):
            t = time.perf_counter()
            c, w, n, _ = run_policy(pol, cs, np.random.default_rng([s, 7]), EPS, DELTA)
            ts.append(time.perf_counter() - t)
            calls.append(c)
            wrong.append(w)
        out[pol] = {"time_s": float(np.median(ts)), "time_s_all": ts, "calls": float(np.median(calls)),
                    "call_share": float(np.median(calls)) / len(cs), "cell_error": float(np.median(wrong)) / len(cs)}
    return out


def slope(xs, ys):
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default="configs/ami_scale_done.txt")
    ap.add_argument("--cache", default="v2-scale-1.5B")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    require_g1("post-hoc scalability analysis")
    e5.V2 = ROOT / "outputs" / "echobase" / a.cache
    ms = [l.strip() for l in Path(a.meetings).read_text().splitlines() if l.strip()]
    order = list(ms)
    np.random.default_rng(0).shuffle(order)
    rate = {(r["model"], r["version"]): r["calls_per_s"]
            for r in json.loads((ROOT / "reports" / "throughput.json").read_text())}[("Qwen2.5-1.5B", "manual")]
    res = {"workload": "w1", "cache": a.cache, "meetings_total": len(order), "eps": EPS, "delta": DELTA,
           "repeats": REPEATS, "llm_calls_per_s": rate, "cpu": cpu_model(), "python": platform.python_version(),
           "sizes": []}
    full = None
    for k in PREFIXES:
        sub = order[:k]
        e5.meetings = lambda split, sub=sub: sub
        t = time.perf_counter()
        cs = e5.cells("w1")
        t1 = time.perf_counter() - t
        row = {"kind": "prefix", "meetings": k, "cells": len(cs), "stage1_s": t1, "policies": stage2(cs),
               "peak_rss_mb": peak_rss_mb()}
        res["sizes"].append(row)
        print(k, len(cs), f"stage1 {t1:.1f}s", {p: round(v["time_s"], 3) for p, v in row["policies"].items()}, flush=True)
        if k == len(order):
            full = cs
    for r in REPLICAS:
        cs = [dict(c, meeting=f"{c['meeting']}#{i}") for i in range(r) for c in full]
        row = {"kind": "replicated", "meetings": r * len(order), "replicas": r, "cells": len(cs),
               "policies": stage2(cs), "peak_rss_mb": peak_rss_mb()}
        res["sizes"].append(row)
        print("x", r, len(cs), {p: round(v["time_s"], 3) for p, v in row["policies"].items()}, flush=True)
    cells = [s["cells"] for s in res["sizes"]]
    res["slope_stage2"] = {p: slope(cells, [s["policies"][p]["time_s"] for s in res["sizes"]]) for p in POLICIES}
    pre = [s for s in res["sizes"] if s["kind"] == "prefix"]
    res["slope_stage1"] = slope([s["cells"] for s in pre], [s["stage1_s"] for s in pre])
    last = pre[-1]
    res["overhead_at_136"] = {p: {"replay_s": last["stage1_s"] + last["policies"][p]["time_s"],
                                  "llm_s": last["policies"][p]["calls"] / rate,
                                  "fraction": (last["stage1_s"] + last["policies"][p]["time_s"])
                                  / (last["policies"][p]["calls"] / rate)} for p in POLICIES}
    res["full_recompute_llm_s_at_136"] = last["cells"] / rate
    print("slopes", res["slope_stage1"], res["slope_stage2"])
    print("overhead", res["overhead_at_136"])
    Path(a.out).write_text(json.dumps(res, indent=1))
