"""V2 -- the mechanism, re-derived on fixtures built here.

CLAIM UNDER TEST (audit secs. 3, 4, 6): what makes the interface mode-match
``a + b`` singular is a LAYER mode that numerically equals a REGION mode; an
equal permittivity alone is necessary but NOT sufficient.  Pre-fix the layer's
propagating roots are decided by the eigensolver's backward error, so some come
back INCOMING; post-fix none do, ``cond(a+b)`` drops by ~11 decades on the
coincident fixtures and the lossless closure returns to the arithmetic floor.

Two families are measured, both built here rather than copied:

  COINCIDENT-MODE (a layer mode sits on a region mode)
    c1  anisotropic uniaxial block, twist 0.4, bg 2.25, n_sub 1.5  (normal)
    c2  anisotropic, no = 1.7 so bg 2.89 = n_sub^2, twist 0.9      (oblique)
    c3  anisotropic, bg 1.96 = n_SUPERSTRATE^2 -- the coincidence on the
        INCIDENCE side, which the fix's own probes never exercised
    c4  anisotropic, conical theta 0.2 phi 0.7, bg 2.25 = n_sub^2
    c5  SCALAR 2-D, weakly modulated eps = 2.25 (1e-6 contrast), n_sub 1.5:
        the layer is a perturbation of the substrate, so EVERY layer mode is
        a region mode
    c6  1-D binary grating with a 2%-duty ridge in an n_groove = n_sub = 1.5
        host: again a near-homogeneous layer at the region's index
    c7  TM twin of c6

  COINCIDENT-PERMITTIVITY, NON-COINCIDENT MODE
    n1  1-D binary grating n_groove = n_sub = 1.5, ridge 2.1, duty 0.5 (TE)
    n2  the same, TM
    n3  SCALAR 2-D eps = 4 block in an eps = 2.25 background, n_sub 1.5
    n4  anisotropic with the block filling 90% of the cell, bg 2.25, n_sub 1.5
    n5  1-D grating whose RIDGE index equals the substrate's, duty 0.5

Usage: python v2_mechanism.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402


def _jones2d(cell, n_sub, n_sup=1.0, theta=0.0, phi=0.0, n_orders=5,
             symmetry="auto"):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    return rcwa_jones_2d(V._P, V._P, cell, n_sub, n_sup, V._DEPTH, V._WL,
                         theta=theta, phi=phi, n_orders_x=n_orders,
                         n_orders_y=n_orders, symmetry=symmetry)


def _eff2d(cell, n_sub, n_sup=1.0, theta=0.0, n_orders=5, pol="te"):
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    return rcwa_efficiency_2d(V._P, V._P, cell, n_sub, n_sup, V._DEPTH, V._WL,
                              theta=theta, polarization=pol,
                              n_orders_x=n_orders, n_orders_y=n_orders)


def _eff1d(n_ridge, n_groove, n_sub, duty=0.5, pol="te", n_orders=15,
           theta=0.0, n_sup=1.0):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    return rcwa_efficiency_1d(V._P, n_ridge, n_groove, n_sub, n_sup, V._DEPTH,
                              duty, V._WL, theta=theta, polarization=pol,
                              n_orders=n_orders)


# ------------------------------------------------------------------- fixtures
def fixtures():
    f = {}
    f["c1_aniso_sub_normal"] = dict(
        kind="jones2d", coincident=True,
        call=lambda: _jones2d(V.uniaxial_cell(twist=0.4, bg=2.25), 1.5))
    f["c2_aniso_ne_sub_oblique"] = dict(
        kind="jones2d", coincident=True,
        call=lambda: _jones2d(V.uniaxial_cell(twist=0.9, no=1.7, ne=1.9,
                                              bg=2.89), 1.7, theta=0.25))
    f["c3_aniso_SUPERstrate"] = dict(
        kind="jones2d", coincident=True,
        call=lambda: _jones2d(V.uniaxial_cell(twist=0.55, no=1.4, ne=1.65,
                                              bg=1.96), 1.6, n_sup=1.4))
    f["c4_aniso_conical"] = dict(
        kind="jones2d", coincident=True,
        call=lambda: _jones2d(V.uniaxial_cell(twist=0.7, bg=2.25), 1.5,
                              theta=0.2, phi=0.7, n_orders=4))
    f["c5_scalar2d_weak"] = dict(
        kind="eff2d", coincident=True,
        call=lambda: _eff2d(V.scalar_cell(bg=2.25, blk=2.25 + 1e-6), 1.5))
    f["c6_oned_thin_ridge_te"] = dict(
        kind="eff1d", coincident=True,
        call=lambda: _eff1d(2.1, 1.5, 1.5, duty=0.02, pol="te"))
    f["c7_oned_thin_ridge_tm"] = dict(
        kind="eff1d", coincident=True,
        call=lambda: _eff1d(2.1, 1.5, 1.5, duty=0.02, pol="tm"))

    f["n1_oned_groove_eq_sub_te"] = dict(
        kind="eff1d", coincident=False,
        call=lambda: _eff1d(2.1, 1.5, 1.5, duty=0.5, pol="te"))
    f["n2_oned_groove_eq_sub_tm"] = dict(
        kind="eff1d", coincident=False,
        call=lambda: _eff1d(2.1, 1.5, 1.5, duty=0.5, pol="tm"))
    f["n3_scalar2d_block4"] = dict(
        kind="eff2d", coincident=False,
        call=lambda: _eff2d(V.scalar_cell(bg=2.25, blk=4.0), 1.5))
    f["n4_aniso_block90"] = dict(
        kind="jones2d", coincident=False,
        call=lambda: _jones2d(V.uniaxial_cell(twist=0.7, bg=2.25, half=0.45),
                              1.5))
    f["n5_oned_ridge_eq_sub"] = dict(
        kind="eff1d", coincident=False,
        call=lambda: _eff1d(1.5, 1.15, 1.5, duty=0.5, pol="te"))
    return f


# ----------------------------------------------------------------- measurement
def measure(spec):
    """One fixture: interface conditioning, on-cut mode census, closure."""
    out = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        eig = V.EigSpy()
        ifc = V.InterfaceSpy()
        with eig, ifc:
            res = spec["call"]()
        out["warnings"] = sorted({str(w.category.__name__) for w in caught})
    # -- closure oracle
    if spec["kind"] == "jones2d":
        out["closure"] = V.closure_defect_jones(res)
    else:
        out["closure"] = V.closure_defect_eff(res)
    # -- interface conditioning
    rows = [r for r in ifc.rows if "cond" in r]
    out["n_interfaces"] = len(rows)
    if rows:
        worst = max(rows, key=lambda r: r["cond"])
        out["cond_max"] = worst["cond"]
        out["cond_all"] = [r["cond"] for r in rows]
        out["ipr_at_worst"] = worst["ipr"]
        out["n_tiny_sv_at_worst"] = worst["n_tiny"]
        out["n_tiny_sv_total"] = sum(r["n_tiny"] for r in rows)
        out["top_weight_at_worst"] = worst["top_weight"]
    # -- on-cut mode census, from the eigenvalues the solve itself produced
    from lumenairy.elements.rcwa import _core as rc
    band = 1e-8
    total = on_cut = incoming = 0
    survives = 0                      # incoming AFTER the library's _sqrt_decay
    worst_ratio_incoming = None
    lam_examples = []
    for lam2 in eig.seen:
        r = V.principal_root(lam2)
        ratio, _scale = V.band_ratio(r)
        oc = ratio <= band
        bad = oc & (r.imag < 0)
        lam_lib = np.asarray(rc._sqrt_decay(lam2))
        ratio_lib, _s2 = V.band_ratio(lam_lib)
        survives += int(np.sum((ratio_lib <= band) & (lam_lib.imag < 0)))
        total += r.size
        on_cut += int(oc.sum())
        incoming += int(bad.sum())
        if bad.any() and len(lam_examples) < 6:
            for k in np.where(bad)[0][:3]:
                lam_examples.append(dict(lam2_re=float(lam2[k].real),
                                         lam2_im=float(lam2[k].imag),
                                         lam_re=float(r[k].real),
                                         lam_im=float(r[k].imag),
                                         ratio=float(ratio[k])))
        if bad.any():
            m = float(np.max(ratio[bad]))
            worst_ratio_incoming = m if worst_ratio_incoming is None else max(
                worst_ratio_incoming, m)
    out["n_eig_arrays"] = len(eig.seen)
    out["modes_total"] = total
    out["modes_on_cut"] = on_cut
    out["modes_on_cut_incoming_principal"] = incoming
    out["modes_on_cut_incoming_after_sqrt_decay"] = survives
    out["worst_ratio_of_an_incoming_on_cut_mode"] = worst_ratio_incoming
    out["incoming_examples"] = lam_examples
    return out


def main():
    V.require_local_tree()
    out = sys.argv[1]
    rows = {}
    for name, spec in fixtures().items():
        try:
            rows[name] = measure(spec)
            rows[name]["coincident_mode_expected"] = spec["coincident"]
        except Exception as exc:
            rows[name] = dict(error=repr(exc))
        r = rows[name]
        print("%-26s coinc=%-5s cond=%-10.3e tinySV=%-3s onCut=%-4s "
              "incoming(lib/princ)=%-8s closure=%+.3e" % (
                  name, spec["coincident"], r.get("cond_max", float("nan")),
                  r.get("n_tiny_sv_total", "-"), r.get("modes_on_cut", "-"),
                  "%s/%s" % (r.get("modes_on_cut_incoming_after_sqrt_decay",
                                          "-"),
                             r.get("modes_on_cut_incoming_principal", "-")),
                  r.get("closure", float("nan"))))
    V.dump(out, dict(fixtures=rows))


if __name__ == "__main__":
    main()
