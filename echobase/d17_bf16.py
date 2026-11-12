"""D17-B (post-hoc; reports/deviations.md): segmentation drift against rerun noise under bf16, batch-varying inference.

    python -m echobase.d17_bf16 --out reports/d17_bf16.json

Caches: outputs/echobase/v2-bf16-b3072 and v2-bf16-b1024 (primary scorer in bf16, two batch budgets, gpulab2).
Per unit, in percentage points: rerun = |s(base, 3072) - s(base, 1024)|, seg = |s(changed, 3072) - s(base, 3072)|,
where base/changed = manual/placebo (W2), w1old/manual (W1, base = w1old), fixed/para (W4). Rule: segmentation drift
exceeds rerun noise if the median of seg - rerun has a BCa interval above 0. The fp32 change (v2 cache) is reported
for comparison, and label flip rates between the budgets.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from pilots.ami import meetings

from .gate import require_g1
from .power_sim import median_ci
from .wiki import articles

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "outputs" / "echobase"
A, B, F = E / "v2-bf16-b3072", E / "v2-bf16-b1024", E / "v2"


def stat(x):
    x = np.asarray(x, float)
    lo, hi, how = median_ci(x, np.random.default_rng(0), 10_000)
    return {"n": len(x), "median": float(np.median(x)), "ci": [float(lo), float(hi)], "ci_method": how}


def load(f):
    d = json.loads(f.read_text())
    lens = d["lens"] if "lens" in d else [e - s for s, e in d["spans"]]
    return d["values"], lens


def sh(vals, lens):
    return 100 * sum(n for v, n in zip(vals, lens) if v == "yes") / sum(lens)


def main(out):
    require_g1("post-hoc bf16 rerun noise (D17-B)")
    res = {"design": "reports/deviations.md D17-B", "workloads": {}, "flip_rate": {}}
    specs = [("W2", c, "manual", "placebo") for c in ("action_item", "question")] + \
            [("W1", c, "w1old", "manual") for c in ("action_item", "question")]
    for wl, col, base, chg in specs:
        rerun, seg, fp32 = [], [], []
        for m in meetings("eval"):
            fa = lambda root, v: root / v / col / f"{m}.json"
            if not all(fa(r, v).exists() for r in (A, B, F) for v in (base, chg)):
                continue
            sb_a, sb_b = sh(*load(fa(A, base))), sh(*load(fa(B, base)))
            sc_a = sh(*load(fa(A, chg)))
            rerun.append(abs(sb_a - sb_b)); seg.append(abs(sc_a - sb_a))
            fp32.append(abs(sh(*load(fa(F, chg))) - sh(*load(fa(F, base)))))
        d = np.array(seg) - np.array(rerun)
        r = {"rerun": stat(rerun), "seg_bf16": stat(seg), "seg_fp32": stat(fp32), "seg_minus_rerun": stat(d)}
        r["seg_exceeds_rerun"] = r["seg_minus_rerun"]["ci"][0] > 0
        res["workloads"][f"{wl}/{col}"] = r
    rerun, seg, fp32 = [], [], []
    for a in articles("test"):
        p = lambda root, ch: root / "wiki" / ch / "doc_date" / f"{a['id']}.json"
        if not all(p(r, ch).exists() for r in (A, B, F) for ch in ("fixed", "para")):
            continue
        fa_, fb_ = sh(*load(p(A, "fixed"))), sh(*load(p(B, "fixed")))
        rerun.append(abs(fa_ - fb_)); seg.append(abs(sh(*load(p(A, "para"))) - fa_))
        fp32.append(abs(sh(*load(p(F, "para"))) - sh(*load(p(F, "fixed")))))
    d = np.array(seg) - np.array(rerun)
    r = {"rerun": stat(rerun), "seg_bf16": stat(seg), "seg_fp32": stat(fp32), "seg_minus_rerun": stat(d)}
    r["seg_exceeds_rerun"] = r["seg_minus_rerun"]["ci"][0] > 0
    res["workloads"]["W4/doc_date"] = r
    for sub in ("manual", "placebo", "w1old", "wiki/fixed", "wiki/para"):
        n = k = 0
        for fa in (A / sub).rglob("*.json"):
            fb = B / fa.relative_to(A)
            if fb.exists():
                x, y = json.loads(fa.read_text())["values"], json.loads(fb.read_text())["values"]
                n += len(x); k += sum(i != j for i, j in zip(x, y))
        res["flip_rate"][sub] = {"labels": n, "flipped": k, "rate": k / n if n else None}
    Path(out).write_text(json.dumps(res, indent=1))
    for k, r in res["workloads"].items():
        print(f"{k:22s} rerun {r['rerun']['median']:5.2f} seg bf16 {r['seg_bf16']['median']:5.2f} "
              f"fp32 {r['seg_fp32']['median']:5.2f}  seg-rerun {r['seg_minus_rerun']['median']:5.2f} "
              f"[{r['seg_minus_rerun']['ci'][0]:.2f}, {r['seg_minus_rerun']['ci'][1]:.2f}] n={r['rerun']['n']} "
              f"-> {r['seg_exceeds_rerun']}")
    print(json.dumps(res["flip_rate"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
