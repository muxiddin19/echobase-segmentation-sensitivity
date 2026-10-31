"""Fig. 5 (post-hoc, D5): Q1 drift of W1 and W2 on nested random subsets of the AMI meetings outside dev/eval
(second model), median and 95% interval, against the number of meetings.

    python paper/figs/scale.py     # reads reports/explore_scale.json -> paper/figs/scale.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INK, GRAY = "#1a1a1a", "#8c8c8c"
sc = json.loads((ROOT / "reports" / "explore_scale.json").read_text())

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for w, color, marker, dx, label in (("W1", INK, "o", 0.96, "W1 diarizer → manual"), ("W2", GRAY, "s", 1.04, "W2 placebo")):
    nested = sc["pairs"][w]["Q1"]["nested"]
    ks = sorted(nested, key=int)
    xs = [int(k) * dx for k in ks]
    meds = [nested[k]["median"] for k in ks]
    ax.plot(xs, meds, color=color, lw=1.2, zorder=2)
    for x, k in zip(xs, ks):
        ax.plot([x, x], nested[k]["ci"], color=color, lw=1.4, solid_capstyle="round")
    ax.plot(xs, meds, marker, color=color, ms=4.5, mec="white", mew=0.7, zorder=3, label=label)
ax.set_xscale("log", base=2)
ks = sorted(sc["pairs"]["W1"]["Q1"]["nested"], key=int)
ax.set_xticks([int(k) for k in ks])
ax.set_xticklabels(ks)
ax.set_ylim(0, None)
ax.set_xlabel("meetings in the sample (nested random subsets)")
ax.set_ylabel("drift D of Q1")
ax.legend(loc="upper right", frameon=False, fontsize=7, handletextpad=0.3)
ax.grid(axis="y", color="#e6e6e6", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(Path(__file__).with_name("scale.pdf"), bbox_inches="tight", pad_inches=0.02)
fig.savefig(Path(__file__).with_name("scale.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
