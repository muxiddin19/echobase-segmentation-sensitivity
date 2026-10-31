"""Fig. 2: median drift D with 95% CI per workload (primary queries), preregistered thresholds marked.

    python paper/figs/drift.py      # reads reports/confirmatory.json, reports/explore_eval.json -> paper/figs/drift.pdf
Confirmatory rows in ink, exploratory rows in gray; each row's threshold T is a short vertical tick.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, GRAY, MUTED = "#1a1a1a", "#8c8c8c", "#b3b3b3"

conf = {r["id"]: r for r in json.loads((ROOT / "reports" / "confirmatory.json").read_text())}
ex = json.loads((ROOT / "reports" / "explore_eval.json").read_text())

rows = [("H1  W1 diarizer → manual (Q1)", conf["H1"]["median"], conf["H1"]["ci_low"], conf["H1"]["ci_high"], conf["H1"]["T"], True),
        ("H2  W2 placebo (Q1)", conf["H2"]["median"], conf["H2"]["ci_low"], conf["H2"]["ci_high"], conf["H2"]["T"], True),
        ("H4  W4 chunker (Qd1)", conf["H4"]["median"], conf["H4"]["ci_low"], conf["H4"]["ci_high"], conf["H4"]["T"], True)]
for name, label in (("W3", "W3 speaker count (Q1)"),):  # E6 has its own figure (granularity.py)
    r = ex["ami"].get(name, {}).get("Q1")
    if r and "median" in r:
        rows.append((label, r["median"], r["ci"][0], r["ci"][1], None, False))

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.4, 0.32 * len(rows) + 0.55))
for i, (label, med, lo, hi, T, confirm) in enumerate(rows):
    y = len(rows) - 1 - i
    c = INK if confirm else GRAY
    ax.plot([lo, hi], [y, y], color=c, lw=1.4, solid_capstyle="round")
    ax.plot(med, y, "o", color=c, ms=4.5, mec="white", mew=0.8, zorder=3)
    ax.text(hi + 0.012, y, f"{med:.3f}", va="center", fontsize=7, color=INK)
    if T is not None:
        ax.plot([T, T], [y - 0.28, y + 0.28], color=MUTED, lw=1.0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows][::-1])
for t, r in zip(ax.get_yticklabels(), rows[::-1]):
    t.set_color(INK if r[5] else GRAY)
ax.set_xlim(0, max(r[3] for r in rows) + 0.12)
ax.set_xlabel("drift D (median, 95% CI)")
ax.grid(axis="x", color="#e6e6e6", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.3)
out = Path(__file__).with_suffix(".pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
print(out)
