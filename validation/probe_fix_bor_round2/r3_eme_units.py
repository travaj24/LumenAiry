"""ROUND 2, D13 -- ``elements/eme/_branch.cut_band``'s floor and the unit system.

THE DEFECT.  ``cut_band`` floors the spectrum scale at a LITERAL 1.0, justified
by "``ky`` here is DIMENSIONLESS".  It is not: ``eme_2d.strip_x_modes``
assembles ``d2/dx2 + eps k0^2`` on a spacing ``Lx / Nx``, so ``lam`` carries
1/length^2 and ``ky`` carries 1/length; ``k0`` is a free argument carrying
units, not a normalisation.  The floor therefore engages whenever
``max|ky| < 1`` in the caller's units -- the ordinary case for a sub-micron
cell written in nanometres -- and the BRANCH DECISION moves with the unit
system.  The pre-5.45.1 exact-zero pin had no scale and was unit-invariant.

WHAT IS MEASURED.  The SAME physical device stated in three unit systems
(um, nm, m), through all three EME sites that read the band:

  * ``eme_2d._ky_forward``                  the scalar strip driver
  * ``eme_diffraction.mode_match``          the diffraction driver
  * ``eme_2d_vector._strip_split_forward``  the vector forward-set census

The unit-free observable is ``ky * scale`` (``ky`` scales as 1/length), and the
ORIENTATION CENSUS -- how many roots come back kept / negated / conjugated --
must be identical to the bit up to that scaling.

Run:  python validation/probe_fix_bor_round2/r3_eme_units.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, dump  # noqa: E402

# ``scale`` is the length of one micrometre in the caller's units: 1.0 = um,
# 1e3 = nm (a micrometre written in nanometres is 1000 units long), 1e-6 = m.
SCALES = {"um": 1.0, "nm": 1e3, "m": 1e-6}


def _strip(scale, im_eps, Nx=96, wl_um=1.55):
    """One 1 um cell at lambda = 1550 nm, stated in the unit system ``scale``."""
    from lumenairy.elements.eme import eme_2d as e2
    Lx = 1.0 * scale
    k0 = 2.0 * np.pi / (wl_um * scale)
    eps = np.full(Nx, 1.0 + 0j)
    eps[Nx // 3:2 * Nx // 3] = 12.0 + im_eps
    lam = np.asarray(e2.strip_x_modes(eps, Lx, Nx, k0)[0], complex)
    raw = np.sqrt(lam + 0j)                        # the UN-oriented root
    fwd = np.asarray(e2._ky_forward(lam, 0.0))
    return dict(lam=lam, raw=raw, fwd=fwd, k0=k0, Lx=Lx)


def _census(raw, fwd, atol):
    """How each root was decided: kept, negated, or conjugated.  Evaluated
    PER MODE against that mode's own un-oriented root, so it does not depend
    on the eigensolver's ordering."""
    kept = np.abs(fwd - raw) <= atol
    neg = (~kept) & (np.abs(fwd + raw) <= atol)
    conj = (~kept) & (~neg) & (np.abs(fwd - np.conj(raw)) <= atol)
    return dict(kept=int(kept.sum()), negated=int(neg.sum()),
                conjugated=int(conj.sum()),
                other=int((~kept & ~neg & ~conj).sum()))


def _sorted(z):
    """Canonical order.  The eigensolver's own ordering is NOT stable between
    two solves (the verification's first-pass numbers were wrong for exactly
    this reason), so every cross-unit-system difference below is taken on a
    SORTED spectrum: an elementwise difference of the raw arrays measures a
    permutation, not a moved root."""
    z = np.asarray(z)
    return z[np.lexsort((np.imag(z), np.real(z)))]


def scalar_leg(im_eps, label, Nx=96):
    rows, out = {}, {}
    for name, s in SCALES.items():
        d = _strip(s, im_eps, Nx=Nx)
        fwd_u = d["fwd"] * s                       # unit-free: ky ~ 1/length
        raw_u = d["raw"] * s
        atol = 1e-9 * max(float(np.max(np.abs(raw_u))), 1.0)
        # the band the LIBRARY computes, read back rather than re-derived
        from lumenairy.elements.eme._branch import cut_band
        try:                                   # ROUND 2 signature
            lib_band = float(cut_band(d["raw"], k0=d["k0"], xp=np))
        except TypeError:                      # pre-ROUND-2 (literal 1.0)
            lib_band = float(cut_band(d["raw"], xp=np))
        rows[name] = dict(
            max_abs_ky=float(np.max(np.abs(d["fwd"]))),
            raw_band=lib_band,
            band_unit_free=lib_band * s,
            census=_census(raw_u, fwd_u, atol),
            fwd=_sorted(fwd_u))
    base = rows["um"]["fwd"]
    for name in SCALES:
        f = rows[name]["fwd"]
        dd = np.abs(f - base)
        tol = 1e-6 * max(float(np.max(np.abs(base))), 1.0)
        out[name] = dict(
            max_abs_ky=rows[name]["max_abs_ky"],
            band_in_own_units=rows[name]["raw_band"],
            band_unit_free=rows[name]["band_unit_free"],
            census=rows[name]["census"],
            census_same_as_um=bool(rows[name]["census"]
                                   == rows["um"]["census"]),
            n_diff=int((dd > tol).sum()),
            worst_dky=float(np.max(dd)) if dd.size else 0.0)
        print("  %-6s %-3s max|ky|=%12.6g band=%10.3e band*s=%10.3e "
              "census=%s same=%s ndiff=%d worst|dky*s|=%.6g"
              % (label, name, out[name]["max_abs_ky"],
                 out[name]["band_in_own_units"], out[name]["band_unit_free"],
                 out[name]["census"], out[name]["census_same_as_um"],
                 out[name]["n_diff"], out[name]["worst_dky"]))
    return out


def match_leg(im_eps, label):
    """``eme_diffraction.mode_match`` -- the same slab in three unit systems."""
    from lumenairy.elements.eme import eme_diffraction as ed
    out, base = {}, None
    orders = [(0, 0), (1, 0), (0, 1), (-1, 0)]
    for name, s in SCALES.items():
        Lx = Ly = 1.0 * s
        Nx = Ny = 16
        k0 = 2.0 * np.pi / (1.55 * s)
        depth = 0.4 * s
        rng = np.random.default_rng(20260912)
        Psi = (rng.standard_normal((Nx * Ny, 6))
               + 1j * rng.standard_normal((Nx * Ny, 6)))
        qz2_um = np.array([30.0, 12.0, 3.0, -8.0, -40.0, 5.0], complex) + im_eps
        qz2 = qz2_um / s ** 2
        res = ed.mode_match(qz2, Psi, orders, kx0=0.0, ky0=0.0, k0=k0,
                            eps_sup=1.0, eps_sub=1.0, depth=depth,
                            Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, inc_order=(0, 0))
        j = res["orders"].index((0, 0))
        rec = dict(T00=float(np.real(res["T"][j])),
                   R00=float(np.real(res["R"][j])),
                   energy=float(np.real(res["energy"])))
        if base is None:
            base = rec
        rec["dT00"] = abs(rec["T00"] - base["T00"])
        out[name] = rec
        print("  %-6s %-3s T00=%.12f R00=%.12f dT00=%.3e"
              % (label, name, rec["T00"], rec["R00"], rec["dT00"]))
    return out


def vector_leg(label):
    """``eme_2d_vector._strip_split_forward`` -- the forward SET (an index set,
    so unit-free by construction) in three unit systems."""
    from lumenairy.elements.eme import eme_2d_vector as ev
    out, base = {}, None
    for name, s in SCALES.items():
        Lx, Nx = 1.0 * s, 32
        k0 = 2.0 * np.pi / (1.55 * s)
        eps = np.full(Nx, 1.0 + 0j)
        eps[Nx // 3:2 * Nx // 3] = 12.0 - 1e-6j
        res = ev.strip_vector_modes(eps, Lx, Nx, k0)
        kys = np.asarray(res[0])
        fwd = ev._strip_split_forward(kys)
        out[name] = dict(n_forward=int(fwd.size),
                         fwd=sorted(int(i) for i in fwd))
        if base is None:
            base = out[name]["fwd"]
        out[name]["same_as_um"] = bool(out[name]["fwd"] == base)
        print("  %-6s %-3s n_forward=%d same_as_um=%s"
              % (label, name, out[name]["n_forward"], out[name]["same_as_um"]))
    return out


def main():
    rec = banner("r3_eme_units")
    res = {}
    print("-- scalar strip, _ky_forward --")
    for im, lab in ((-1e-6j, "gain6"), (-1e-5j, "gain5"), (-1e-7j, "gain7"),
                    (0.0, "real"), (+1e-6j, "loss6")):
        res["scalar_" + lab] = scalar_leg(im, lab)
    print("-- diffraction driver, mode_match --")
    for im, lab in ((-1e-6j, "gain6"), (0.0, "real")):
        try:
            res["match_" + lab] = match_leg(im, lab)
        except Exception as exc:                   # pragma: no cover - probe
            print("  match_%s FAILED: %r" % (lab, exc))
            res["match_" + lab] = dict(error=repr(exc))
    print("-- vector strip, _strip_split_forward --")
    try:
        res["vector"] = vector_leg("vec")
    except Exception as exc:                       # pragma: no cover - probe
        print("  vector FAILED: %r" % (exc,))
        res["vector"] = dict(error=repr(exc))
    n_bad = sum(v.get("n_diff", 0) for k, val in res.items()
                if k.startswith("scalar_") for v in val.values())
    worst_dt = max([v.get("dT00", 0.0) for k, val in res.items()
                    if k.startswith("match_") and "error" not in val
                    for v in val.values()] or [float("nan")])
    print("\n--- summary ---")
    print("  scalar roots differing between unit systems: %d" % (n_bad,))
    print("  worst mode_match dT00 between unit systems : %.6g" % (worst_dt,))
    dump("r3_eme_units", res, rec)


if __name__ == "__main__":
    main()
