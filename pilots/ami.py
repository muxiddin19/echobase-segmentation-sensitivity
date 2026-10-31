"""AMI manual annotations (NXT 1.6.2): meeting lists, gold segments, words and dialogue acts."""
import re
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

from .segmap import Segment

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "ami" / "annotations"
NITE = "{http://nite.sourceforge.net/}"
HREF = re.compile(r"^(?P<file>[^#]+)#id\((?P<a>[^)]+)\)(?:\.\.id\((?P<b>[^)]+)\))?$")


def meetings(split):
    section, out = None, {}
    for line in (ROOT / "configs" / "ami_meetings.txt").read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif line and not line.startswith("#") and section:
            out.setdefault(section, []).extend(line.split())
    return out[split]


def audio_ready(meeting):
    """Mix-Headset wav exists and no running process (e.g. wget) still has its name on the command line."""
    name = f"{meeting}.Mix-Headset.wav"
    if not (ROOT / "data" / "ami" / "audio" / name).exists():
        return False
    for cmd in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if name.encode() in cmd.read_bytes():
                return False
        except OSError:
            pass
    return True


def speakers(meeting):
    return sorted(p.name.split(".")[1] for p in (ANN / "words").glob(f"{meeting}.*.words.xml"))


@lru_cache(maxsize=None)
def _word_file(name):
    """All elements of a words file in document order, plus an id -> position index."""
    elems = []
    for el in ET.parse(ANN / "words" / name).getroot():
        st, en = el.get("starttime"), el.get("endtime")
        elems.append({
            "id": el.get(f"{NITE}id"),
            "word": el.tag == "w" and el.get("punc") != "true",
            "text": (el.text or "").strip(),
            "start": float(st) if st else None,
            "end": float(en) if en else None,
        })
    return elems, {e["id"]: i for i, e in enumerate(elems)}


def _resolve(href):
    m = HREF.match(href)
    elems, index = _word_file(m["file"])
    a = index[m["a"]]
    b = index[m["b"]] if m["b"] else a
    return elems[a:b + 1]


def _span(elems):
    starts = [e["start"] for e in elems if e["start"] is not None]
    ends = [e["end"] for e in elems if e["end"] is not None]
    text = " ".join(e["text"] for e in elems if e["word"])
    return (min(starts) if starts else None, max(ends) if ends else None, text)


def reference_words(meeting):
    """All timed reference words (no punctuation, no non-word events), with their speaker."""
    out = []
    for spk in speakers(meeting):
        elems, _ = _word_file(f"{meeting}.{spk}.words.xml")
        out += [{"speaker": spk, "start": e["start"], "end": e["end"], "word": e["text"]}
                for e in elems if e["word"] and e["start"] is not None and e["end"] is not None]
    return sorted(out, key=lambda w: w["start"])


def gold_segments(meeting):
    """Manual speaker segments with their reference words (the corrected 'after' state)."""
    segs = []
    for spk in speakers(meeting):
        path = ANN / "segments" / f"{meeting}.{spk}.segments.xml"
        if not path.exists():
            continue
        for seg in ET.parse(path).getroot():
            elems = [e for ch in seg if ch.tag == f"{NITE}child" for e in _resolve(ch.get("href"))]
            _, _, text = _span(elems)
            segs.append(Segment(seg.get(f"{NITE}id"), spk, float(seg.get("transcriber_start")),
                                float(seg.get("transcriber_end")), text))
    return sorted(segs, key=lambda s: (s.start, s.speaker))


@lru_cache(maxsize=None)
def da_types():
    root = ET.parse(ANN / "ontologies" / "da-types.xml").getroot()
    return {el.get(f"{NITE}id"): el.get("name") for el in root.iter("da-type")}


def dialogue_acts(meeting):
    """Manual dialogue acts: speaker, time span, reference text and DA type name (human ground truth)."""
    types, acts = da_types(), []
    for spk in speakers(meeting):
        path = ANN / "dialogueActs" / f"{meeting}.{spk}.dialog-act.xml"
        if not path.exists():
            continue
        for da in ET.parse(path).getroot():
            elems, label = [], None
            for ch in da:
                if ch.tag == f"{NITE}child":
                    elems += _resolve(ch.get("href"))
                elif ch.tag == f"{NITE}pointer" and ch.get("role") == "da-aspect":
                    label = types.get(HREF.match(ch.get("href"))["a"])
            start, end, text = _span(elems)
            if start is not None and end is not None:
                acts.append({"id": da.get(f"{NITE}id"), "speaker": spk, "start": start, "end": end,
                             "type": label, "text": text})
    return sorted(acts, key=lambda d: (d["start"], d["speaker"]))
