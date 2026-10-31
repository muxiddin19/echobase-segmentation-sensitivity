"""EchoBase delta rules: derive a new cell from old cells and provenance, without an LLM call.

Returns the derived value, or None when no rule applies (the cell must be recomputed or certified).
Rules per decomposability class (exact under exact decomposability; their real error is measured, not assumed):
  any    : identical text to a contributing old segment -> reuse (content hash)
  binary : (OR) every word comes from old 'no' segments          -> 'no'   (a sub-span of 'no' is 'no')
           a contained old segment is 'yes'                       -> 'yes'  (a super-span of 'yes' is 'yes')
  count  : (SUM) the new segment is exactly a union of old ones   -> sum (capped at 9)
           every word comes from old segments with count 0        -> 0
  label  : (holistic) no algebraic rule
"""


def derive(kind, new_text, prov, old_segs, old_vals):
    contrib = prov["contrib"]
    for i in contrib:
        if old_segs[i].text == new_text:
            return old_vals[i], "hash"
    if not contrib or not prov["covered"]:
        return None, None
    vals = [old_vals[i] for i in contrib]
    if kind == "binary":
        if all(v == "no" for v in vals):
            return "no", "or-sub"
        if any(old_vals[i] == "yes" for i in prov["contained"]):
            return "yes", "or-super"
    elif kind == "count":
        if prov["exact_union"]:
            return min(9, sum(vals)), "sum-union"
        if all(v == 0 for v in vals):
            return 0, "sum-zero"
    return None, None
