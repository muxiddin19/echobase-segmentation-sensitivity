"""Fig. 4: per-meeting drift of Q1 under W1 (diarizer -> manual) and W2 (placebo), held-out meetings.

    python paper/figs/per_unit.py    # reads reports/confirmatory.json -> paper/figs/per_unit.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, GRAY, LINE = "#1a1a1a", "#8c8c8c", "#d0d0d0"
conf = {r["id"]: r for r in json.loads((ROOT / "reports" / "confirmatory.json").read_text())}
w1, w2 = conf["H1"]["per_unit"], conf["H2"]["per_unit"]
ms = sorted(w1, key=lambda m: w1[m])

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.4, 2.9))
for y, m in enumerate(ms):
    ax.plot([w1[m], w2[m]], [y, y], color=LINE, lw=1.2, zorder=1)
    ax.plot(w2[m], y, "s", color=GRAY, ms=4, mec="white", mew=0.6, zorder=2)
    ax.plot(w1[m], y, "o", color=INK, ms=4.5, mec="white", mew=0.6, zorder=3)
ax.axvline(conf["H1"]["T"], color=LINE, lw=0.8, ls="--", zorder=0)
ax.text(conf["H1"]["T"], len(ms) - 0.45, " H1 threshold", fontsize=6.5, color=GRAY, va="bottom")
ax.set_yticks(range(len(ms)))
ax.set_yticklabels(ms)
ax.set_xlim(0, max(max(w1.values()), max(w2.values())) * 1.08)
ax.set_xlabel("drift D of the action-item share")
ax.plot([], [], "o", color=INK, ms=4.5, label="W1 diarizer → manual")
ax.plot([], [], "s", color=GRAY, ms=4, label="W2 placebo")
ax.legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.3)
ax.set_ylim(-0.7, len(ms) - 0.3)
ax.grid(axis="x", color="#ececec", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name("per_unit.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(Path(__file__).with_name("per_unit.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
