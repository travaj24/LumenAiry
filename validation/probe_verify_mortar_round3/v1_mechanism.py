"""V1 -- claim (a): the generalized mortar operand is rank-deficient BY
CONSTRUCTION when either side is a PROMOTED in-plane region.

Measures, on fixtures built in ``_vfix.py`` (independent of the fix's):
the full SVD, the near-null right singular vector's split between the two
block columns, the residual of ``np.linalg.solve``'s answer, and the
right-hand side's component along the near-null LEFT direction
(``||u_min^H B|| / ||B||``, the "range membership"), which predicts the
cross-build relative spread ``eps ||B|| / ||u_min^H B||``.

Run: ``python v1_mechanism.py <tag>``  (tag = win | wsl)
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _path                                        # noqa: E402,F401,I001

import json                                                    # noqa: E402
import time                                                    # noqa: E402

import _vfix as F                                              # noqa: E402
from _capture import GEN_SITE, capture, rcond_of, svd_facts    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

CASES = [
    # (kind, [M...], n_orders)
    ("mix_spacer", [4, 5, 6, 7], 2),
    ("mix_pattern", [4, 5, 6], 2),
    ("mix_magnetic", [4, 5], 2),
    ("mix_slant_inplane", [4, 5, 6], 2),
    ("adv_strong_oop_next_to_inplane", [4, 5], 2),
    ("adv_tiny_slant_inplane", [4, 5], 2),
    ("ctrl_oop_both", [4, 5, 6], 2),
    ("ctrl_slant_both", [4, 5], 2),
    ("adv_near_inplane_oop", [4, 5], 2),
    ("both_promoted", [4, 5, 6], 2),
    ("ctrl_inplane_both", [4, 5], 2),
    ("mix_spacer", [4, 5], 1),
    ("mix_pattern", [4], 3),
]


def main(tag):
    rows = []
    for kind, Ms, no in CASES:
        for M in Ms:
            t0 = time.time()
            try:
                with capture(record_all=True) as recs:
                    st = F.build(kind, M, n_orders=no)
                    import warnings
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter("always")
                        o, R, T = st.solve(jones=False)
                    warn_txt = [str(w.message)[:70] for w in ws]
                r00 = F.R00(o, R)
                closure = abs(float(R.sum(axis=1)[1] + T.sum(axis=1)[1]) - 1.0)
            except Exception as e:                      # noqa: BLE001
                rows.append({"kind": kind, "M": M, "n_orders": no,
                             "error": f"{type(e).__name__}: {e}"[:400]})
                continue
            dt = time.time() - t0
            gen = [r for r in recs if r["site"] == GEN_SITE]
            inp = [r for r in recs if r["site"] != GEN_SITE]
            for i, r in enumerate(gen):
                f = svd_facts(r["A"], r["B"], r.get("ma"))
                f.update(kind=kind, M=M, n_orders=no, ifc=i,
                         prom_a=r.get("prom_a"), prom_b=r.get("prom_b"),
                         screen=r["screen"], rcond=rcond_of(r["A"]),
                         R00=r00, closure=closure, seconds=round(dt, 2),
                         ma=r.get("ma"), mb=r.get("mb"),
                         n_gen_sites=len(gen), n_inplane_sites=len(inp),
                         warnings=warn_txt)
                rows.append(f)
            if not gen:
                rows.append({"kind": kind, "M": M, "n_orders": no,
                             "note": "NO generalized mortar site reached",
                             "n_inplane_sites": len(inp), "R00": r00,
                             "closure": closure, "seconds": round(dt, 2),
                             "inplane_sites": sorted({r["site"] for r in inp}),
                             "warnings": warn_txt})
            print(f"{kind:32s} M={M} no={no} gen={len(gen)} inp={len(inp)} "
                  f"{dt:6.1f}s", flush=True)
            for r in gen:
                pass
    out = {"env": F.env(), "tag": tag, "rows": rows}
    (HERE / f"v1_mechanism_{tag}.json").write_text(
        json.dumps(out, indent=1), encoding="cp1252")
    print(f"wrote v1_mechanism_{tag}.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "win")
