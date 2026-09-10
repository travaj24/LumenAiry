"""V1 -- BIT-IDENTITY of every returned answer across the round-3 change,
and a classified census of every verdict that FLIPS (task 1).

Run twice: once on the tree that carries round 3 and once on the round-3
branch point ``f2371e0`` (round 1 + round 2 + round 2's verification).  The
two JSONs are then compared by ``v1_compare.py``.

Per fixture the probe records

* what the LIBRARY did -- refused or returned, the full message, and every
  warning text;
* a SHA-256 of the returned ``R``, ``T`` and Jones buffers' raw bytes, so
  "bit-identical" is a byte comparison and not a tolerance;
* the arbiter's verdict and its whole evidence dict, taken by calling
  ``_sliver_arbiter`` directly on the unguarded solve;
* the two INDEPENDENT continuity references a flip is classified against:
  the exact ``delta -> 0`` solve of the same device, and the solve on the
  PRESCRIBED ``min_feature`` grid.

The fixture set spans the paths the change must not touch -- ordinary
vertical stacks with no manufactured cell, anisotropic in-plane and
out-of-plane directors, a stack whose walls are shared exactly, the
per-layer grid route, stacks carrying an OWNED liner -- and the paths it
may touch: the sliver families at wall steps above and below the trigger,
on five devices.

    python v1_bitid.py out.json
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import v_fixtures as F  # noqa: E402
from v_fixtures import ps  # noqa: E402


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


# --------------------------------------------------------------- fixtures --
def fixtures():
    """``(name, builder, delta, ref_builder_or_None)`` for every solve.

    ``ref_builder`` is the exact ``delta -> 0`` device: present wherever a
    continuity classification is meaningful."""
    out = []

    # ---- (A) ORDINARY stacks with NO manufactured cell ---------------------
    out.append(("A1_plain_2layer", lambda: F.vplain(), 0.0, None))
    out.append(("A2_plain_3layer", lambda: F.vplain(nl=3), 0.0, None))
    out.append(("A3_plain_spacer", lambda: F.vplain(spacer=0.4e-6), 0.0, None))
    out.append(("A4_plain_grazing",
                lambda: F.vplain(theta=1.31, nsup=2.05,
                                 nsub=complex(2.90, 1.10)), 0.0, None))
    out.append(("A5_plain_deg14", lambda: F.vplain(degree=14), 0.0, None))
    out.append(("A6_tensor_inplane", lambda: F.vtensor(0.0), 0.0, None))
    out.append(("A7_tensor_oop", lambda: F.voop(0.0), 0.0, None))
    out.append(("A8_tensor_oop_lossy",
                lambda: F.voop(0.0, nsub=complex(1.46, 0.02)), 0.0, None))
    out.append(("A9_gmr_exact", lambda: F.vgmr(0.0), 0.0, None))
    out.append(("A10_fp_exact", lambda: F.vfp(0.0), 0.0, None))
    out.append(("A11_wood_exact", lambda: F.vwood(0.0), 0.0, None))
    out.append(("A12_box_exact",
                lambda: F.box_stack(0.0, period=1.35e-6, wl=1.064e-6,
                                    nsup=2.05, nsub=complex(2.90, 1.10),
                                    theta=1.18, degree=8, e_hi=12.25, nl=2),
                0.0, None))

    # ---- (B) the PER-LAYER grid route (a union is never formed) ------------
    def _perlayer(delta, nl=6):
        from lumenairy.elements.pmm import PMMStack
        st = PMMStack(1.35e-6, n_superstrate=1.0, n_substrate=1.52, degree=8,
                      far_field_orders=21, min_feature=1.35e-6 * F.NO_SNAP,
                      layer_grids="per-layer")
        for k in range(nl):
            dd = delta * k / max(nl - 1, 1)
            st.add_layer(0.12e-6, segments=[(0.3120 - dd, 2.56),
                                            (0.6790 + dd - (0.3120 - dd),
                                             12.25),
                                            (1.0 - (0.6790 + dd), 2.56)])
        st.set_source(1.064e-6, theta=0.28)
        return st

    for d in (0.0, 1e-3, 1e-5, 3e-6):
        out.append((f"B_perlayer_d{d:g}", (lambda d=d: _perlayer(d)), d,
                    (lambda: _perlayer(0.0))))

    # ---- (C) an OWNED liner (the R3-B / D-1 geometry) ----------------------
    for liner in (1e-4, 1e-6):
        for d in (0.0, 1e-3, 3e-5, 3e-6):
            out.append((f"C_liner{liner:g}_d{d:g}",
                        (lambda d=d, li=liner: F.vliner(d, liner=li)), d,
                        (lambda li=liner: F.vliner(0.0, liner=li))))

    # ---- (D) the SLIVER families, above and below the trigger --------------
    fam = {
        "D_box": (lambda d: F.box_stack(
            d, period=1.35e-6, wl=1.064e-6, nsup=2.05,
            nsub=complex(2.90, 1.10), theta=1.18, degree=8, e_hi=12.25,
            nl=2)),
        "D_gmr6": (lambda d: F.vgmr(d, degree=6)),
        "D_gmr8": (lambda d: F.vgmr(d, degree=8)),
        "D_fp": (lambda d: F.vfp(d)),
        "D_wood": (lambda d: F.vwood(d)),
        "D_tensor": (lambda d: F.vtensor(d)),
        "D_oop": (lambda d: F.voop(d)),
    }
    for nm, fn in fam.items():
        for d in (3e-3, 1e-4, 1e-5, 5e-6, 3e-6, 1e-6):
            out.append((f"{nm}_d{d:g}", (lambda d=d, fn=fn: fn(d)), d,
                        (lambda fn=fn: fn(0.0))))
    return out


# ------------------------------------------------------------------ run ----
def one(name, build, delta, ref_build):
    st = build()
    cur = F.unguarded(st)
    rec = dict(name=name, delta=delta, worst=cur["worst"],
               least=cur["least"],
               raw_R=_sha(cur["R"]), raw_T=_sha(cur["T"]),
               raw_J=_sha(cur["J"]))
    pre = F.prescribed(st)
    rec["screen"] = pre is not None
    snap = None
    if pre is not None:
        rec.update(w_wide=pre["w_wide"], w_narrow=pre["w_narrow"],
                   own=pre["own"], n_hit=pre["n_hit"], mf_fix=pre["mf"])
        snap = F.snapped(st, pre["mf"])
        su = max(snap["worst"] - 1.0, 0.0)
        mv = F.move_shared(cur, snap)
        rec.update(su_snap=su, drop=F.drop(cur["worst"], su), move=mv,
                   move_w=(None if mv is None else mv / pre["w_wide"]))
    # the arbiter, called directly on the unguarded solve
    got = ps._sliver_arbiter(build(), cur["worst"], cur["R"], cur["T"], None)
    if got is None:
        rec["arb"] = None
    else:
        v, ev = got
        rec["arb"] = v
        rec["arb_ev"] = (None if ev is None else
                         {k: F.jsonable(ev[k]) for k in sorted(ev)
                          if k != "hit"})
        if ev is not None:
            rec["arb_hit"] = F.jsonable(list(ev["hit"]))
    # what the library actually did
    refused, msg, out, warns = F.guarded(build())
    rec["refused"] = refused
    rec["msg"] = msg
    rec["warns"] = warns
    if out is not None:
        rec.update(ret_R=_sha(out["R"]), ret_T=_sha(out["T"]),
                   ret_J=_sha(out["J"]),
                   ret_bitid=(_sha(out["R"]) == rec["raw_R"]
                              and _sha(out["T"]) == rec["raw_T"]
                              and _sha(out["J"]) == rec["raw_J"]))
    # the two independent continuity references
    if ref_build is not None and delta > 0.0:
        ref = F.unguarded(ref_build())
        e = F.move_shared(cur, ref, pol=1)
        rec.update(err=e, err_d=e / delta, kind=F.classify(e, delta))
        if snap is not None:
            es = F.move_shared(snap, ref, pol=1)
            rec.update(err_snap=es, err_snap_d=es / delta,
                       kind_snap=F.classify(es, delta))
    return rec


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else "v1_bitid.json"
    t0 = time.perf_counter()
    rows = [one(*f) for f in fixtures()]
    doc = dict(build=F.build_info(), wall=time.perf_counter() - t0,
               n=len(rows), rows=rows)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(F.jsonable(doc), fh, indent=1)
    ref = sum(1 for r in rows if r["refused"])
    print(f"{len(rows)} fixtures, {ref} refused, "
          f"{sum(1 for r in rows if r.get('ret_bitid') is False)} not "
          f"bit-identical to the unguarded solve, "
          f"{doc['wall']:.1f} s -> {dest}")


if __name__ == "__main__":
    main()
