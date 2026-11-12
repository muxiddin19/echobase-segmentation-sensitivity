"""D18 (post-hoc): how many reference words each segmentation version contains, relative to the manual version.

    python -m echobase.d18_wordsets --out reports/d18_wordsets.json      # CPU
Canonical units depend only on the set of words a version contains, so this count bounds what can still differ.
"""
import argparse
import json
from pathlib import Path

from pilots.ami import meetings, reference_words

from .d18_canonical import VERSIONS
from .explore_gold import version

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {v: {"manual_words": 0, "only_manual": 0, "only_version": 0} for v in VERSIONS}
    for m in meetings("eval"):
        reference_words(m)
        man = set(version("manual", m).word_seg)
        for v in VERSIONS:
            s = set(version(v, m).word_seg)
            r = res[v]
            r["manual_words"] += len(man); r["only_manual"] += len(man - s); r["only_version"] += len(s - man)
    Path(a.out).write_text(json.dumps({"design": "reports/deviations.md D18", "versions": res}, indent=1))
    print(json.dumps(res))
