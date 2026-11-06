"""Fig. (post-hoc, D11): wall time of the maintenance replay against the number of changed cells (W1, 1.5B labels,
136 scale meetings; beyond them the cell set replicated x2-x8, hollow markers), single CPU thread, next to the LLM
time of the calls the certified policy still makes.

    python paper/figs/scalability.py     # reads reports/explore_scalability.json -> paper/figs/scalability.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, MID, LIGHT = "#1a1a1a", "#6b6b6b", "#a6a6a6"
d = json.loads((ROOT / "reports" / "explore_scalability.json").read_text())
rate = d["llm_calls_per_s"]
sizes = d["sizes"]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.4, 2.1))


def series(xs, ys, real, color, marker, label, ls="-"):
    ax.plot(xs, ys, color=color, lw=1.2, ls=ls, zorder=2)
    for x, y, r in zip(xs, ys, real):
        ax.plot(x, y, marker, color=color, ms=4.2, mec=color if not r else "white", mew=0.8,
                mfc=color if r else "white", zorder=3)
    ax.plot([], [], marker, color=color, ls=ls, lw=1.2, ms=4.2, mec="white", mew=0.8, label=label)


cells = [s["cells"] for s in sizes]
real = [s["kind"] == "prefix" for s in sizes]
series(cells, [s["policies"]["echobase"]["calls"] / rate for s in sizes], real, LIGHT, "D",
       "LLM time of calls left", ls="--")
pre = [s for s in sizes if s["kind"] == "prefix"]
series([s["cells"] for s in pre], [s["stage1_s"] for s in pre], [True] * len(pre), MID, "s", "row mapping")
series(cells, [s["policies"]["echobase"]["time_s"] for s in sizes], real, INK, "o", "rules/anchor + audit")
series(cells, [s["policies"]["hash"]["time_s"] for s in sizes], real, INK, "^", "content hash", ls=":")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("changed cells")
ax.set_ylabel("seconds")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=6.5, handletextpad=0.3,
          labelspacing=0.25, columnspacing=0.8, borderaxespad=0.1)
ax.grid(color="#e6e6e6", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name("scalability.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(Path(__file__).with_name("scalability.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
