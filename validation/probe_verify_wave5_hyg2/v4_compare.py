"""V4 driver -- archive-to-archive key comparison with an EXPECTED-TO-DIFFER
declaration, reported in both directions.

    python v4_compare.py <base.json> <branch.json> <out.json>

"expected to differ but identical" counts as a failure, so the change is
proved to have LANDED as well as to have been contained.
"""
from __future__ import annotations

import json
import re
import sys

LEAF_OWNED = ("CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
              "_load_numba", "_get_aspheric_sag_accum_numba",
              "_ensure_numexpr_loaded", "_collect_semi_diameters",
              "_warn_if_aperture_exceeds_grid")

#: The ONLY keys this verification expects to move: the four write/delete
#: outcomes of each of the eight leaf-owned names.
EXPECTED = {f"A-{fam}-{n}" for n in LEAF_OWNED
            for fam in ("write", "rw", "del", "rd")}


def main():
    base = json.load(open(sys.argv[1], encoding="utf-8"))
    branch = json.load(open(sys.argv[2], encoding="utf-8"))
    keys = sorted(set(base) | set(branch))
    ident, exp_diff, exp_same, unexp, only_b, only_br = [], [], [], [], [], []
    for k in keys:
        if k not in branch:
            only_b.append(k)
            continue
        if k not in base:
            only_br.append(k)
            continue
        same = base[k] == branch[k]
        if k in EXPECTED:
            (exp_same if same else exp_diff).append(k)
        else:
            (ident if same else unexp).append(k)
    out = {
        "base_file": sys.argv[1], "branch_file": sys.argv[2],
        "keys": len(keys),
        "identical": len(ident),
        "expected_differ_and_did": len(exp_diff),
        "expected_differ_but_identical": len(exp_same),
        "unexpected_differ": len(unexp),
        "only_base": len(only_b), "only_branch": len(only_br),
        "expected_declared": len(EXPECTED),
        "unexpected_differ_keys": unexp,
        "expected_differ_but_identical_keys": exp_same,
        "only_base_keys": only_b, "only_branch_keys": only_br,
        "VERDICT": ("PASS" if not unexp and not exp_same and not only_b
                    and not only_br else "FAIL"),
    }
    json.dump(out, open(sys.argv[3], "w", encoding="utf-8"),
              indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items()
                      if not k.endswith("_keys") and not k.endswith("_file")},
                     indent=1))
    for k in unexp:
        print("  UNEXPECTED DIFFER:", k)
    for k in exp_same:
        print("  EXPECTED-BUT-IDENTICAL:", k)


if __name__ == "__main__":
    main()
