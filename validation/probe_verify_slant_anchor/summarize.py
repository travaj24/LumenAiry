"""Re-read the probe JSONs and print the verification's tables.

``python summarize.py identity``  -- task 1, the surface census
``python summarize.py o2``        -- task 4, the T22 bar, two-sided
``python summarize.py v1``        -- task 3, the JAX dispatch (raw dump)
``python summarize.py v2``        -- task 2, the frame anchor (raw dump)
``python summarize.py dur``       -- task 5, the durability rows (raw dump)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
from fixtures import SHEARED, TRANSMISSION_SURFACES  # noqa: E402

NL = "\n"
SLANTED_2D = ("j2d_slant_ob25", "j2d_slant_conical", "hyb_slant_ob25",
              "pure_slant_ob25")


def identity():
    js = _lib.jsons("t1_identity")
    if not js:
        print("no t1_identity results")
        return
    for b in sorted({k.split(".")[1] for k in js}):
        have = {k.split(".")[0]: v for k, v in js.items()
                if k.split(".")[1] == b}
        post = have.get("post")
        if post is None:
            continue
        for other in ("pre", "rev"):
            ref = have.get(other)
            if ref is None:
                continue
            print(NL + f"=== {b.upper()}  post vs {other} "
                       f"({ref['stamp']['tree']}) ===")
            ident = moved = unexpected = 0
            movers = {}
            for grp in ("oned", "twod"):
                for fx, row in post[grp].items():
                    ref_row = ref[grp].get(fx, {})
                    for surf, val in row.items():
                        if surf.startswith("_"):
                            continue
                        rv = ref_row.get(surf)
                        if val == rv:
                            ident += 1
                            continue
                        moved += 1
                        movers.setdefault(fx, []).append(surf)
                        ok = ((fx in SHEARED or fx in SLANTED_2D)
                              and surf in TRANSMISSION_SURFACES)
                        if not ok:
                            unexpected += 1
            print(f"identical {ident}   moved {moved}   "
                  f"UNEXPECTED {unexpected}")
            for fx in sorted(movers):
                tag = "sheared" if fx in SHEARED else "NOT-SHEARED"
                print(f"  {fx:38s} [{tag}] {', '.join(movers[fx])}")


def o2():
    """The O2 two-sided table: BROKEN (measured on the guard-free tree) vs
    REFUSED (measured on the tree that holds the guard)."""
    js = _lib.jsons("t4_o2_census")
    for b in sorted({k.split(".")[1] for k in js}):
        post, rev = js.get(f"post.{b}"), js.get(f"rev.{b}")
        if not post or not rev:
            print(NL + f"=== O2 {b.upper()}: missing arm "
                       f"(post={bool(post)} rev={bool(rev)}) ===")
            continue
        P, V = post["census"], rev["census"]
        broken, refused, healthy, unreached = set(), set(), set(), set()
        for k, v in V.items():
            rt = v.get("sumRT")
            if not v.get("n_interfaces"):
                unreached.add(k)
            elif rt is None or rt > 1.10:
                broken.add(k)
            else:
                healthy.add(k)
        for k, v in P.items():
            if v.get("outcome") == "_ConditioningError":
                refused.add(k)
        print(NL + f"=== O2  {b.upper()} ===")
        print(f"fixtures {len(P)}  interfaces "
              f"{sum(v.get('n_interfaces', 0) for v in P.values())}")
        print(f"BROKEN (rev: sum R+T > 1.10 or raised) {len(broken)}   "
              f"HEALTHY {len(healthy)}   no-interface {len(unreached)}")
        print(f"REFUSED (post) {len(refused)}")
        print(f"refused == broken : {refused == broken}")
        if refused - broken:
            print("  FALSE REFUSALS:", sorted(refused - broken))
        if broken - refused:
            print("  LET THROUGH:", sorted(broken - refused))

        def _rng(names, src, key="rcond_min"):
            xs = [src[n][key] for n in names if src[n].get(key) is not None]
            return (min(xs), max(xs)) if xs else (None, None)

        bl, bh = _rng(broken, P)
        hl, hh = _rng(healthy, P)
        rl, rh = _rng(broken, P, "resid_max")
        if bl is not None:
            print(f"  refused rcond   {bl:.3e} .. {bh:.3e}")
        if hl is not None:
            print(f"  healthy rcond   {hl:.3e} .. {hh:.3e}")
        if rl is not None:
            print(f"  refused resid   {rl:.3e} .. {rh:.3e}")
        if bh and hl:
            print(f"  gap {math.log10(hl / bh):.2f} decades; 1e-10 sits "
                  f"{math.log10(1e-10 / bh):.2f} above the worst refused, "
                  f"{math.log10(hl / 1e-10):.2f} below the worst healthy")
        ident = moved = 0
        movers = []
        for k, v in P.items():
            a, c = v.get("sha"), V.get(k, {}).get("sha")
            if a and c:
                if a == c:
                    ident += 1
                else:
                    moved += 1
                    movers.append(k)
        print(f"  bit-identity over the solves BOTH trees produced: "
              f"{ident} identical, {moved} moved {movers[:6]}")
        na = post.get("no_census", {})
        same = sum(1 for k, v in P.items()
                   if v.get("sha") and v["sha"] == na.get(k, {}).get("sha"))
        nsha = sum(1 for v in P.values() if v.get("sha"))
        print(f"  census-armed vs census-off, same bytes: {same}/{nsha}")
        rows = sorted((P[n]["rcond_min"], n) for n in healthy
                      if P[n].get("rcond_min") is not None)
        print("  five tightest HEALTHY rows:")
        for r, n in rows[:5]:
            print(f"    {r:.4e}  {n}")
        rows = sorted((P[n]["rcond_min"], n) for n in broken
                      if P[n].get("rcond_min") is not None)
        print("  five loosest REFUSED rows:")
        for r, n in rows[-5:]:
            print(f"    {r:.4e}  {n}")


def _walk(d, pre=""):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            _walk(v, pre + k + ".")
        else:
            print(f"  {pre + k:62s} {v}")


def dump(name):
    for key, js in sorted(_lib.jsons(name).items()):
        print(NL + f"=== {name}  {key} ===")
        for k, v in js.items():
            if k == "stamp":
                continue
            if isinstance(v, dict):
                print(f"-- {k}")
                _walk(v, "  ")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "identity"
    if what == "identity":
        identity()
    elif what == "o2":
        o2()
    else:
        dump({"v1": "t3_v1_jax", "v2": "t2_v2_sign",
              "dur": "t5_durability", "dur2": "t5b_durability_rest",
              "census": "t4_o2_census"}.get(what, what))
