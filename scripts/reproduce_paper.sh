#!/usr/bin/env bash
# Level-1 reproduction (ARTIFACT.md): every statistic, table and figure of the paper from the released caches, CPU only.
#   bash scripts/reproduce_paper.sh [outdir]
# Results go to <outdir> (default: reproduced/) so the released reports are not overwritten; the script then diffs
# the reproduced JSON against the released files.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-reproduced}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES= PYTHONPATH=.
PY=${PYTHON:-python}

$PY tools/eval_lock.py --check
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
from echobase import confirm
out = Path(sys.argv[1])
res = confirm.run()                       # the preregistered analysis, same code path as `python -m echobase.confirm`
(out / "confirmatory.json").write_text(json.dumps(res, indent=1))
(out / "confirmatory.md").write_text(confirm.report(res), encoding="utf-8")
print(confirm.report(res))
EOF
$PY -m echobase.explore_eval --out "$OUT/explore_eval.json"
$PY -m echobase.explore_eval --model Qwen/Qwen2.5-0.5B-Instruct --out "$OUT/explore_eval_model2.json"
$PY -m echobase.explore_flips --out "$OUT/explore_flips.json"
$PY -m echobase.e5_maintenance --workload w1 --out "$OUT/e5_w1.json"
$PY -m echobase.e5_maintenance --workload w3 --out "$OUT/e5_w3.json"

# post-hoc analyses (deviations D5, D6 in reports/deviations.md), from the cached labels of every scorer
for M in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct Phi-3.5-mini-instruct; do
  ORG=Qwen; [ "$M" = Phi-3.5-mini-instruct ] && ORG=microsoft
  $PY -m echobase.explore_signed --model "$ORG/$M" --out "$OUT/explore_signed_$M.json"
  $PY -m echobase.explore_gold --model "$ORG/$M" --out "$OUT/explore_gold_$M.json"
done
$PY -m echobase.explore_eval --model Qwen/Qwen2.5-3B-Instruct --out "$OUT/explore_eval_model3.json"
$PY -m echobase.explore_eval --model Qwen/Qwen2.5-7B-Instruct --out "$OUT/explore_eval_7b.json"
$PY -m echobase.explore_eval --model microsoft/Phi-3.5-mini-instruct --out "$OUT/explore_eval_phi.json"
$PY -m echobase.explore_eval --cache v2-ctx --out "$OUT/explore_eval_ctx.json"
$PY -m echobase.explore_signed --cache v2-ctx --out "$OUT/explore_signed_ctx.json"
$PY -m echobase.explore_gold --cache v2-ctx --out "$OUT/explore_gold_ctx.json"
$PY -m echobase.explore_calib --out "$OUT/explore_calib.json"
$PY -m echobase.explore_ctxcost --out "$OUT/explore_ctxcost.json"
$PY -m echobase.explore_formal --out "$OUT/explore_formal.json"
$PY -m echobase.explore_scale --meetings configs/ami_scale_done.txt --out "$OUT/explore_scale.json"
# deviation D7
$PY -m echobase.explore_scale --meetings configs/ami_scale_done.txt --model Qwen/Qwen2.5-1.5B-Instruct   --cache v2-scale-1.5B --out "$OUT/explore_scale_1.5b.json"
$PY -m echobase.explore_confident --out "$OUT/explore_confident.json"
$PY -m echobase.explore_pooled --workload w1 --out "$OUT/explore_pooled_w1.json"
$PY -m echobase.explore_pooled --workload w3 --out "$OUT/explore_pooled_w3.json"
$PY -m echobase.explore_chunksize --out "$OUT/explore_chunksize.json"
$PY -m echobase.explore_crosshost --a v2 --b v2-gp231-check --out "$OUT/post_crosshost_1.5b.json"
$PY -m echobase.explore_cost --out "$OUT/explore_cost.json"  # review 4: calls and prompt tokens per version (tokenizer only)

# Compare with the released reports. Floats are compared to a relative 1e-9: the topic-distance queries sum over a
# Python set, whose order (string hashing) varies between runs, so their last digit can differ. NaN equals NaN.
$PY - "$OUT" <<'EOF'
import json, math, sys
from pathlib import Path

def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, (int, float)):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b

out, ok = Path(sys.argv[1]), True
for f in sorted(out.glob("*.json")):
    a, b = json.loads(f.read_text()), json.loads((Path("reports") / f.name).read_text())
    for d in (a, b):
        if isinstance(d, dict):
            d.pop("wall_clock_s", None)
    s = same(a, b)
    ok &= s
    print(f"{'same' if s else 'DIFFERS'}  {f.name}")
sys.exit(0 if ok else 1)
EOF

# tables, figures and the numbers in the text (from the released reports, which the step above showed identical)
$PY paper/figs/workloads.py && $PY paper/figs/example.py
$PY paper/figs/drift.py && $PY paper/figs/drift_all.py && $PY paper/figs/per_unit.py && $PY paper/figs/maintenance.py
$PY paper/figs/gold.py && $PY paper/figs/scale.py && $PY paper/figs/models.py && $PY paper/figs/cost.py
$PY paper/gen_numbers.py
(cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
  && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null) \
  && echo "paper/main.pdf built" || echo "LaTeX not available: skipped the PDF"
