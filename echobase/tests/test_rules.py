from pilots.segmap import Segment
from echobase.lineage import Version, provenance
from echobase.rules import derive


def V(texts, word_seg):
    return Version([Segment(str(i), "A", i, i + 1, t) for i, t in enumerate(texts)], word_seg)


def test_provenance_merge_and_split():
    # words 0-3; old: [0,1] [2,3]; new: [0,1,2,3] (merge)
    old, new = V(["a b", "c d"], {0: 0, 1: 0, 2: 1, 3: 1}), V(["a b c d"], {0: 0, 1: 0, 2: 0, 3: 0})
    p = provenance(old, new)[0]
    assert p["contrib"] == [0, 1] and p["contained"] == [0, 1] and p["exact_union"]
    # split: old [0..3] -> new [0,1] [2,3]
    old, new = V(["a b c d"], {0: 0, 1: 0, 2: 0, 3: 0}), V(["a b", "c d"], {0: 0, 1: 0, 2: 1, 3: 1})
    p = provenance(old, new)
    assert p[0]["contrib"] == [0] and p[0]["contained"] == [] and not p[0]["exact_union"]


def test_binary_rules():
    old = [Segment("0", "A", 0, 1, "x"), Segment("1", "A", 1, 2, "y")]
    merge = {"contrib": [0, 1], "contained": [0, 1], "covered": True, "exact_union": True}
    assert derive("binary", "x y", merge, old, ["no", "no"]) == ("no", "or-sub")
    assert derive("binary", "x y", merge, old, ["no", "yes"]) == ("yes", "or-super")
    split_of_yes = {"contrib": [0], "contained": [], "covered": True, "exact_union": False}
    assert derive("binary", "x1", split_of_yes, old, ["yes", "no"]) == (None, None)
    uncovered = dict(merge, covered=False)
    assert derive("binary", "x y z", uncovered, old, ["no", "no"]) == (None, None)


def test_count_and_label_rules():
    old = [Segment("0", "A", 0, 1, "x"), Segment("1", "A", 1, 2, "y")]
    merge = {"contrib": [0, 1], "contained": [0, 1], "covered": True, "exact_union": True}
    assert derive("count", "x y", merge, old, [2, 8]) == (9, "sum-union")
    split = {"contrib": [0], "contained": [], "covered": True, "exact_union": False}
    assert derive("count", "x1", split, old, [0, 3]) == (0, "sum-zero")
    assert derive("label", "x y", merge, old, ["cost", "cost"]) == (None, None)
    assert derive("label", "x", split, old, ["cost", "design"]) == ("cost", "hash")


def test_hypergeom_upper_bound_hand_values():
    from scipy.stats import hypergeom
    from echobase.certify import d_upper
    # full census: the bound equals the observed count
    assert d_upper(50, 50, 3, 0.05) == 3
    # zero errors in 30 of 100: the bound D* is the largest D with P(X=0) >= alpha
    D = d_upper(100, 30, 0, 0.05)
    assert hypergeom.cdf(0, 100, D, 30) >= 0.05 > hypergeom.cdf(0, 100, D + 1, 30)


def test_audit_never_certifies_a_bad_stratum_often():
    import numpy as np
    from echobase.certify import audit
    truth = [0] * 1000
    prov = [1] * 100 + [0] * 900          # 10% wrong, eps = 5%: certifying would be a violation
    certified = sum(audit(prov, truth, np.random.default_rng(s), 0.05, 0.05)[2] for s in range(200))
    assert certified / 200 <= 0.05


def test_hypergeom_lower_bound_hand_values():
    from scipy.stats import hypergeom
    from echobase.certify import d_lower
    assert d_lower(50, 50, 3, 0.05) == 3
    D = d_lower(100, 30, 5, 0.05)
    assert hypergeom.sf(4, 100, D, 30) >= 0.05 > hypergeom.sf(4, 100, D - 1, 30)
