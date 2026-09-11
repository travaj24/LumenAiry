"""TASK E -- the cross-arm comparator.

Loads every ``ve_run_*.json`` and answers three separate questions, which the
build's own report conflates:

1. Does an answer move ACROSS ARMS within one build?  (kernel / threads /
   platform).  That is a defect on either build.
2. Does an answer move ACROSS BUILDS on a fixed arm?  Expected only where the
   branch decision changed; anything else is a finding.
3. What is the flipped-mode census, per class, per build, per arm?
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def load():
    runs = {}
    for p in sorted(HERE.glob("ve_run_*.json")):
        tag = p.stem[len("ve_run_"):]
        if tag.startswith("smoke"):
            continue
        runs[tag] = json.loads(p.read_text(encoding="cp1252"))
    return runs


def arms(runs):
    print("=" * 100)
    print("ARMS (kernel READ BACK from threadpoolctl, never inferred)")
    print("=" * 100)
    print("%-28s %-5s %-4s %-12s %-12s %-7s %-8s" % (
        "tag", "plat", "thr", "requested", "loaded", "numpy", "secs"))
    for tag, d in runs.items():
        a = d["arm"]
        print("%-28s %-5s %-4s %-12s %-12s %-7s %-8s" % (
            tag, a["platform"], a["blas_threads"], a["requested_coretype"],
            a["loaded_kernel"], a["numpy"], d.get("secs")))


def hash_matrix(runs, section, keyfields):
    """{fixture: {tag: (h1, h2, ...)}}"""
    out = collections.defaultdict(dict)
    for tag, d in runs.items():
        for row in d.get(section, []):
            out[row["name"]][tag] = tuple(row.get(k) for k in keyfields)
    return out


def report_section(runs, section, keyfields, label):
    M = hash_matrix(runs, section, keyfields)
    if not M:
        return
    print()
    print("=" * 100)
    print("SECTION %s -- hash fields %s" % (label, keyfields))
    print("=" * 100)
    n_arm_move = n_build_move = n_stable = 0
    for name in sorted(M):
        per = M[name]
        pre = {t: v for t, v in per.items() if t.startswith("pre_")}
        post = {t: v for t, v in per.items() if t.startswith("post_")}
        upre, upost = set(pre.values()), set(post.values())
        moved_arm = (len(upre) > 1, len(upost) > 1)
        moved_build = bool(upre and upost and upre != upost)
        if any(moved_arm):
            n_arm_move += 1
            print("  ARM-DEPENDENT  %-40s pre:%d distinct  post:%d distinct"
                  % (name, len(upre), len(upost)))
            for lbl, grp in (("pre", pre), ("post", post)):
                if len(set(grp.values())) > 1:
                    byh = collections.defaultdict(list)
                    for t, v in grp.items():
                        byh[v].append(t)
                    for v, tags in byh.items():
                        print("      %-4s %s  <- %s"
                              % (lbl, str(v)[:60], ",".join(sorted(tags))))
        elif moved_build:
            n_build_move += 1
            print("  BUILD-MOVED    %-40s pre=%s post=%s"
                  % (name, str(sorted(upre)[0])[:34],
                     str(sorted(upost)[0])[:34]))
        else:
            n_stable += 1
    print("  --- %s: %d stable, %d moved with the BUILD, %d moved with the ARM"
          % (label, n_stable, n_build_move, n_arm_move))


def census(runs):
    print()
    print("=" * 100)
    print("FLIPPED-MODE CENSUS (per build; the union over arms is shown when "
          "an arm disagrees)")
    print("=" * 100)
    per = collections.defaultdict(lambda: collections.defaultdict(set))
    for tag, d in runs.items():
        b = "pre" if tag.startswith("pre_") else "post"
        for r in d.get("strip", []):
            per[r["name"]][b].add((r["n_prop"], r["n_evan"], r["n_lossy"],
                                   r["flip_prop"], r["flip_evan"],
                                   r["flip_lossy"], r["conj_prop"],
                                   r["conj_evan"], r["conj_lossy"],
                                   round(r["worst_dky"], 6)))
    hdr = ("%-40s | %-28s | %-28s" %
           ("fixture", "PRE  negP/negE/negL cnjP wdky", "POST negP/negE/negL cnjP wdky"))
    print(hdr)
    print("-" * len(hdr))
    for name in sorted(per):
        row = []
        for b in ("pre", "post"):
            vals = per[name][b]
            if len(vals) == 1:
                v = next(iter(vals))
                row.append("%3d/%3d/%3d %3d %10.4g"
                           % (v[3], v[4], v[5], v[6], v[9]))
            else:
                row.append("ARM-VARIES:%d" % len(vals))
        print("%-40s | %-28s | %-28s" % (name, row[0], row[1]))


def margins(runs):
    print()
    print("=" * 100)
    print("BAND TWO-SIDED MARGINS  (|Im z| / max(1, max|z|); band = 1e-9 * "
          "max(1, max|z|))")
    print("=" * 100)
    print("%-40s %-5s %-10s %-11s %-11s %-9s %-9s"
          % ("fixture", "build", "band", "oncut_max", "lossy_min",
             "dec_below", "dec_above"))
    seen = set()
    for tag, d in runs.items():
        b = "pre" if tag.startswith("pre_") else "post"
        if not tag.endswith("HASWELL_t1") or "win" not in tag:
            continue
        for r in d.get("strip", []):
            k = (r["name"], b)
            if k in seen:
                continue
            seen.add(k)
            band = r["band"]
            oc = r["oncut_pop_max"]
            lm = r["lossy_pop_min"]
            import math
            db = ("%.2f" % math.log10(band / oc)) if oc else "inf"
            da = ("%.2f" % math.log10(lm / band)) if lm else "n/a"
            print("%-40s %-5s %-10.3e %-11.4g %-11s %-9s %-9s"
                  % (r["name"], b, band, oc if oc is not None else -1,
                     ("%.4g" % lm) if lm is not None else "n/a", db, da))


def paired(runs, section, keyfields, label):
    """PRE vs POST on the SAME arm -- the only pre/post comparison that is not
    confounded by the LAPACK bits moving with the kernel."""
    M = hash_matrix(runs, section, keyfields)
    if not M:
        return
    print()
    print("=" * 100)
    print("PAIRED (same arm) PRE vs POST -- %s, fields %s" % (label, keyfields))
    print("=" * 100)
    moved = collections.defaultdict(list)
    for name in sorted(M):
        per = M[name]
        for t, v in per.items():
            if not t.startswith("post_"):
                continue
            pt = "pre_" + t[len("post_"):]
            if pt not in per:
                continue
            if per[pt] != v:
                moved[name].append(t)
    n = len(M)
    print("  %d/%d fixtures MOVED pre->post on at least one arm" % (len(moved), n))
    for name in sorted(moved):
        arms_ = moved[name]
        allarms = sum(1 for t in M[name] if t.startswith("post_"))
        print("     %-42s moved on %d/%d arms" % (name, len(arms_), allarms))
    unmoved = [k for k in sorted(M) if k not in moved]
    print("  UNMOVED on every arm (%d): %s" % (len(unmoved), ", ".join(unmoved)))


def main():
    runs = load()
    print("loaded %d runs" % len(runs))
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "arms"):
        arms(runs)
    if what in ("all", "paired"):
        paired(runs, "strip", ("hash_ky",), "strip ANSWER ky")
        paired(runs, "strip", ("hash_lam",), "strip PROVENANCE lam")
        paired(runs, "mode_match", ("hash_r", "hash_t"), "mode_match")
        paired(runs, "diffraction_fd", ("hash_r", "hash_t", "hash_RT"),
               "diffraction_fd")
        paired(runs, "vector", ("hash_ky", "hash_W"), "vector")
        paired(runs, "split_forward", ("hash_idx", "err"), "split_forward")
        paired(runs, "layer_modes", ("hash_q", "n"), "layer_modes")
        paired(runs, "diffraction_eme", ("hash_r", "hash_t"),
               "diffraction_eme")
    if what in ("all", "decision"):
        report_section(runs, "strip", ("hash_code",), "strip DECISION code")
        report_section(runs, "strip", ("n_oncut",), "strip n_oncut")
        report_section(runs, "strip", ("n_neg", "n_conj", "n_same"),
                       "strip decision COUNTS")
        report_section(runs, "strip", ("n_prop", "n_evan", "n_lossy"),
                       "strip CLASS counts")
        report_section(runs, "layer_modes", ("n",), "layer_modes COUNT")
    if what in ("all", "hash"):
        report_section(runs, "strip",
                       ("hash_lam", "hash_ky", "hash_ky_unsorted"), "strip")
        report_section(runs, "mode_match", ("hash_r", "hash_t"), "mode_match")
        report_section(runs, "diffraction_fd", ("hash_r", "hash_t", "hash_RT"),
                       "diffraction_fd")
        report_section(runs, "vector", ("hash_ky", "hash_W"), "vector")
        report_section(runs, "split_forward", ("hash_idx", "err"),
                       "split_forward")
        report_section(runs, "layer_modes", ("hash_q", "n"), "layer_modes")
        report_section(runs, "diffraction_eme", ("hash_r", "hash_t"),
                       "diffraction_eme")
    if what in ("all", "census"):
        census(runs)
    if what in ("all", "margins"):
        margins(runs)


if __name__ == "__main__":
    main()
