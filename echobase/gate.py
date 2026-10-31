"""The G1 gate for EchoBase eval/test work: a dated marker line that only the human adds to the STATUS.md header.

    - **EchoBase G1:** 2026-11-03
A bare "G1" is not enough: the pilot phase already wrote "G1 freeze done 2026-10-27" into the header (found
2026-10-28, before any eval run; no eval or test output existed).
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(r"^- \*\*EchoBase G1:\*\* \d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)


def passed(status_text):
    return bool(MARKER.search(status_text.split("## Log")[0]))


def require_g1(what):
    if not passed((ROOT / "STATUS.md").read_text(encoding="utf-8")):
        raise SystemExit(f"{what} is not allowed before the EchoBase G1 gate: the human adds "
                         f"'- **EchoBase G1:** YYYY-MM-DD' to the STATUS.md header after writing the eval lock")
