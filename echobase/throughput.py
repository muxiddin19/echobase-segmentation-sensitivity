"""Scoring throughput from the job journals (systemd user units on gpulab2) -> reports/throughput.json.

    python -m echobase.throughput --units e1-gpu0 e1-gpu1 e4-gpu0 e6-gpu1 --out reports/throughput.json
Rate per (unit, version) = calls between the first and the last per-meeting progress line / elapsed seconds (the first
meeting, which includes model loading, is excluded). Wikipedia test scoring logs no per-article lines: its rate is all
its calls over the time from the shell `date` line that precedes it to its "done test" line (model loading included).
"""
import argparse
import datetime as dt
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

DATE = re.compile(r"^(\S+) \S+ \S+ \w{3} \w{3} +\d+ [\d:]+ \w+ \d{4}$")
WIKI = re.compile(r"^(\S+) \S+ \S+ done test: calls (\d+)")
LINE = re.compile(r"^(\S+) \S+ \S+ (\w+) (\S+): (\d+) segments \(calls so far (\d+)\)")
MODEL = {"e1-gpu0": "Qwen2.5-1.5B", "e1-gpu1": "Qwen2.5-1.5B", "e4-gpu0": "Qwen2.5-0.5B", "e6-gpu1": "Qwen2.5-1.5B"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for u in a.units:
        log = subprocess.run(["journalctl", "--user", "-u", u, "-o", "short-iso", "--no-pager"], capture_output=True,
                             text=True).stdout
        seq = defaultdict(list)
        last_date = None
        for line in log.splitlines():
            if DATE.match(line):
                last_date = dt.datetime.strptime(line.split()[0], "%Y-%m-%dT%H:%M:%S%z")
            w = WIKI.match(line)
            if w and last_date:
                sec = (dt.datetime.strptime(w.group(1), "%Y-%m-%dT%H:%M:%S%z") - last_date).total_seconds()
                res.append({"unit": u, "model": MODEL.get(u, "?"), "version": "wiki-test", "articles": 200,
                            "calls": int(w.group(2)), "calls_per_s": int(w.group(2)) / sec})
            m = LINE.match(line)
            if m:
                seq[m.group(2)].append((dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S%z"), int(m.group(5)), int(m.group(4))))
        for v, xs in seq.items():
            xs.sort()
            sec = (xs[-1][0] - xs[0][0]).total_seconds()
            res.append({"unit": u, "model": MODEL.get(u, "?"), "version": v, "meetings": len(xs), "calls": xs[-1][1],
                        "segments": sum(n for _, _, n in xs), "calls_per_s": (xs[-1][1] - xs[0][1]) / sec})
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r in res:
        print(r)
