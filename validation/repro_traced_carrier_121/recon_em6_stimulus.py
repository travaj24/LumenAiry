# Did niche C8 silence E-M6's energy self-check by removing a MANUFACTURED
# lobe, or by cutting a GENUINE fold caustic?
#
# THE QUESTION.  ``test_niche_audit_w3_elements.py::TestEM6RayDensityEnergy
# SelfCheck::test_fires_when_a_fold_caustic_manufactures_energy`` builds a
# strong biconcave on a grid that barely covers its aperture and asserts the
# v5.30 ray-density ENERGY self-check fires.  Pre-C8 the measured ratio was
# 1.100 against a band upper bound of 1.050; on the settled tree it reads
# 1.0193 and the check is silent.  Two mutually exclusive readings:
#
#   (A) PATTERN-MATCH with the C7 fires-test.  The stimulus was manufactured
#       by the Newton inverse extrapolating OUTSIDE the traced exit support,
#       exactly the defect class C8 removes at source.  Then the test's
#       stimulus is gone, not its subject, and the C8 S9.3 reconciliation
#       applies (fires-arm with the bound OFF, plus new assertions that the
#       bound silences it).
#
#   (B) REAL REGRESSION.  The stimulus is a genuine FOLD of the EXACT ray map
#       (det J -> 0 or a sign change INSIDE the traced support), whose
#       ``1/sqrt(|det J|)`` blow-up C8 cannot legitimately touch -- the bound
#       only ever zeroes amplitude OUTSIDE the convex hull of the alive
#       stop-passing exit landings.  If C8 silenced THAT, it cut real light.
#
# WHAT SEPARATES THEM, measured, not argued:
#
#   1. THE PARTITION.  Split the power C8 removes against the call's OWN exact
#      ray bundle into (a) outside the hull of every alive ray, (b) between the
#      stop-passing hull and the all-rays hull, (c) INSIDE the stop-passing
#      hull.  (A) predicts (c) == 0 and (a) ~ everything.  (B) predicts a
#      large (c): a fold caustic lives where rays ARE.
#
#   2. THE EXACT MAP'S OWN det J.  Finite-difference the EXACT traced
#      entrance->exit landing lattice (no fit, no Newton, no upsample) over the
#      stop-passing rays and look for a sign change or a collapse toward zero.
#      A fold caustic in the exact map is a sign change; extrapolation of a
#      FITTED map is not.
#
#   3. WHERE THE PRE-C8 EXCESS SAT.  With the bound off, how much of the
#      returned power lies outside the all-rays hull, and at what amplitude?
#
# usage:  python recon_em6_stimulus.py
import hashlib
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

import lumenairy as la                                     # noqa: E402,I001
import lumenairy.elements._lens_traced as LT               # noqa: E402
import lumenairy.raytrace as _RT                           # noqa: E402

_WL = 1.31e-6
_N, _AP = 256, 3.0e-3
_DX = 1.01 * _AP / _N
_W0 = 1.4e-3

_PRESC = {'name': 'biconcave', 'aperture_diameter': _AP,
          'thicknesses': [3e-3],
          'surfaces': [
              {'radius': -3e-3, 'glass_before': 'air',
               'glass_after': 'N-BK7', 'conic': 0.0,
               'aspheric_coeffs': None},
              {'radius': 3e-3, 'glass_before': 'N-BK7',
               'glass_after': 'air', 'conic': 0.0,
               'aspheric_coeffs': None}]}


def _gauss():
    x = (np.arange(_N) - _N / 2) * _DX
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128), X, Y


def _run(bound, **over):
    """One E-M6 element call with the C8 bound forced, capturing the EXACT ray
    bundle its own tracer produced (read AFTER the call, because the library
    mutates ``image_rays`` in place post-trace)."""
    old = LT.REMAP_INVERSE_SUPPORT_BOUND
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    otrace = _RT.trace
    grab = {}

    def _tr(rays, surfaces, wavelength, **kw):
        res = otrace(rays, surfaces, wavelength, **kw)
        if rays.x.size > grab.get('n', 0):
            grab.update(n=int(rays.x.size), h_x=np.array(rays.x, copy=True),
                        result=res)
        return res

    _RT.trace = _tr
    E0, X, Y = _gauss()
    kw = dict(prescription=_PRESC, wavelength=_WL, dx=_DX,
              amplitude_model='ray_density', ray_subsample=8, n_workers=1,
              parallel_amp=False, on_undersample='silent',
              on_aperture_beam='silent')
    kw.update(over)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            E = np.asarray(la.apply_real_lens_traced(E0, **kw))
    finally:
        _RT.trace = otrace
        LT.REMAP_INVERSE_SUPPORT_BOUND = old
    return E, [str(r.message) for r in rec], grab, E0, X, Y


def _hulls(grab):
    """(A, b) half-plane sets for the hull of every ALIVE ray and for the hull
    of the STOP-PASSING alive rays -- the bound's own support."""
    from scipy.spatial import ConvexHull
    fin = grab['result'].image_rays
    n = int(round(np.sqrt(fin.x.size)))
    XE = np.asarray(fin.x, dtype=float).reshape(n, n)
    YE = np.asarray(fin.y, dtype=float).reshape(n, n)
    ok = (np.asarray(fin.alive, bool).reshape(n, n)
          & np.isfinite(XE) & np.isfinite(YE))
    xs = np.asarray(grab['h_x'], dtype=float).reshape(n, n)[:, 0]
    ok_ap = ok & ((xs[:, None] ** 2 + xs[None, :] ** 2) <= (0.5 * _AP) ** 2)
    out = []
    for m in (ok, ok_ap):
        eq = ConvexHull(np.column_stack([XE[m], YE[m]])).equations
        out.append((np.ascontiguousarray(eq[:, :2].T),
                    np.ascontiguousarray(eq[:, 2])))
    return out[0], out[1], XE, YE, ok, ok_ap, xs


def _signed(hull, X, Y):
    A, b = hull
    return ((np.column_stack([X.ravel(), Y.ravel()]) @ A + b)
            .max(axis=1)).reshape(X.shape)


def _detj_exact(XE, YE, xs, ok_ap):
    """det J of the EXACT traced map, by central differences on the launch
    lattice.  No fit, no Newton, no upsample -- this is the geometry itself."""
    h = float(np.diff(xs).mean())
    dxdx = (XE[2:, 1:-1] - XE[:-2, 1:-1]) / (2.0 * h)
    dxdy = (XE[1:-1, 2:] - XE[1:-1, :-2]) / (2.0 * h)
    dydx = (YE[2:, 1:-1] - YE[:-2, 1:-1]) / (2.0 * h)
    dydy = (YE[1:-1, 2:] - YE[1:-1, :-2]) / (2.0 * h)
    det = dxdx * dydy - dxdy * dydx
    m = (ok_ap[2:, 1:-1] & ok_ap[:-2, 1:-1] & ok_ap[1:-1, 2:]
         & ok_ap[1:-1, :-2] & ok_ap[1:-1, 1:-1] & np.isfinite(det))
    return det, m


def main():
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    print("E-M6 stimulus adjudication: manufactured extrapolation (A) or a "
          "genuine fold (B)?")
    print(f"  fixture N={_N} dx={_DX*1e6:.4f} um aperture={_AP*1e3:.3f} mm "
          f"w0={_W0*1e3:.3f} mm  (grid span {_N*_DX*1e3:.4f} mm)")
    print()

    E_off, w_off, grab, E0, X, Y = _run(False)
    E_on, w_on, _g2, _E0, _X, _Y = _run(True)
    disc = (X ** 2 + Y ** 2) <= (_AP / 2) ** 2
    p_den = float((np.abs(E0[disc]) ** 2).sum())
    r_off = float((np.abs(E_off) ** 2).sum()) / p_den
    r_on = float((np.abs(E_on) ** 2).sum()) / p_den

    def _cls(ws):
        e = [m for m in ws if 'energy self-check' in m]
        f = [m for m in ws if 'fold' in m.lower() or 'caustic' in m.lower()]
        h = [m for m in ws if 'HALO self-check' in m]
        return e, f, h

    for lbl, r, ws in (('C8 OFF (= pre-C8 library)', r_off, w_off),
                       ('C8 ON  (= settled tree)  ', r_on, w_on)):
        e, f, h = _cls(ws)
        print(f"  {lbl}  ratio {r:.5f}   energy-check fires: "
              f"{'YES' if e else 'no ':>3}   fold-caustic warning: "
              f"{'YES' if f else 'no ':>3}   halo-check fires: "
              f"{'YES' if h else 'no ':>3}")
    print(f"  band: gain tol {LT._RD_ENERGY_GAIN_TOL:.3f} "
          f"-> upper bound {1.0 + LT._RD_ENERGY_GAIN_TOL:.3f}")
    print()

    H_all, H_ap, XE, YE, ok, ok_ap, xs = _hulls(grab)
    s_all = _signed(H_all, X, Y)
    s_ap = _signed(H_ap, X, Y)
    out_all, out_ap = s_all > 0.0, s_ap > 0.0
    print(f"  exact ray bundle: {int(ok.sum())} alive of {ok.size}, "
          f"{int(ok_ap.sum())} of them stop-passing")
    print(f"  exit |r| max: alive {float(np.hypot(XE[ok], YE[ok]).max())*1e3:.4f}"
          f" mm, stop-passing "
          f"{float(np.hypot(XE[ok_ap], YE[ok_ap]).max())*1e3:.4f} mm; "
          f"grid reach {float(np.hypot(X, Y).max())*1e3:.4f} mm")
    print(f"  exit pixels outside H_all {int(out_all.sum())}, outside H_ap "
          f"{int(out_ap.sum())}, of {X.size}")
    print()

    # ---- 1. the partition ------------------------------------------------
    d = np.abs(E_off) ** 2 - np.abs(E_on) ** 2
    tot, a = float(d.sum()), float(d[out_all].sum())
    b = float(d[out_ap & ~out_all].sum())
    c = float(d[~out_ap].sum())
    print("  1. PARTITION of the power C8 removes, over the aperture-"
          "transmitted input power")
    print(f"     total dP/P_ap            {tot/p_den:12.5e}")
    print(f"     (a) outside H_all        {a/p_den:12.5e}   "
          f"({100.0*a/tot if tot else 0.0:6.2f} % of the removal)")
    print(f"     (b) H_ap..H_all          {b/p_den:12.5e}   "
          f"({100.0*b/tot if tot else 0.0:6.2f} %)")
    print(f"     (c) INSIDE H_ap          {c/p_den:12.5e}   "
          f"({100.0*c/tot if tot else 0.0:6.2f} %)   <- must be ~0 for (A)")
    print()

    # ---- 2. det J of the EXACT map --------------------------------------
    det, m = _detj_exact(XE, YE, xs, ok_ap)
    sd = np.sign(det)
    flip = (((sd[:, 1:] * sd[:, :-1] < 0) & (m[:, 1:] & m[:, :-1])).sum()
            + ((sd[1:, :] * sd[:-1, :] < 0) & (m[1:, :] & m[:-1, :])).sum())
    ad = np.abs(det[m])
    print("  2. det J of the EXACT traced map (central differences on the "
          "launch lattice,")
    print("     stop-passing rays only -- no fit, no Newton, no upsample)")
    print(f"     samples {int(m.sum())}   sign changes between adjacent "
          f"cells: {int(flip)}")
    print(f"     det J range [{float(det[m].min()):+.6e}, "
          f"{float(det[m].max()):+.6e}]  all one sign: "
          f"{bool((det[m] > 0).all() or (det[m] < 0).all())}")
    print(f"     |det J| min {float(ad.min()):.6e}  median "
          f"{float(np.median(ad)):.6e}  min/median "
          f"{float(ad.min()/np.median(ad)):.4f}")
    print(f"     library caustic floor rel {LT._RAY_DENSITY_CAUSTIC_FLOOR_REL:g}"
          f", max/min bound {LT._RAY_DENSITY_CAUSTIC_MAXMIN:g}; "
          f"exact max/min {float(ad.max()/ad.min()):.4f}")
    print()

    # ---- 3. where the pre-C8 excess sat ---------------------------------
    for lbl, F in (('C8 OFF', E_off), ('C8 ON ', E_on)):
        aE = np.abs(F)
        pk = float(aE.max())
        p_out = float((aE[out_all] ** 2).sum()) / p_den
        amax = float(aE[out_all].max()) / pk if out_all.any() and pk else 0.0
        print(f"  3. {lbl}: power beyond H_all {p_out:.5e} of P_ap, "
              f"max |E| there {amax:.4e} of peak, total ratio "
              f"{float((aE**2).sum())/p_den:.5f}")
    print()

    # ---- 4. control: does the lobe survive without the ray-density gain? --
    Es_off, _w, _g, _E, _X, _Y = _run(False, amplitude_model='screen')
    aE = np.abs(Es_off)
    pk = float(aE.max())
    print(f"  4. control, amplitude_model='screen' (no 1/sqrt|det J| gain), "
          f"C8 OFF:")
    print(f"     power beyond H_all "
          f"{float((aE[out_all]**2).sum())/p_den:.5e} of P_ap, max |E| there "
          f"{(float(aE[out_all].max())/pk if out_all.any() and pk else 0.0):.4e}"
          f" of peak")
    print()

    # ---- 5. the ABSOLUTE oracle: geometric transport conserves energy -----
    # An eikonal element moves the power the ALIVE, stop-passing rays carry
    # and nothing else.  Summing |E_in|^2 over the launch lattice at its own
    # cell area is a pure raytrace + input-field statement: it shares no code
    # with the fit, the Newton inverse, the upsample or the amplitude model.
    fin = grab['result'].image_rays
    nL = int(round(np.sqrt(fin.x.size)))
    hx = np.asarray(grab['h_x'], dtype=float).reshape(nL, nL)[:, 0]
    h = float(np.diff(hx).mean())
    XI, YI = hx[:, None] * np.ones(nL)[None, :], np.ones(nL)[:, None] * hx[None, :]
    ain = np.abs(np.interp(0, [0], [0]))  # placeholder, replaced below
    from scipy.ndimage import map_coordinates as _mc
    ain = _mc(np.abs(E0).astype(float),
              np.vstack([(YI.ravel() / _DX + _N / 2.0),
                         (XI.ravel() / _DX + _N / 2.0)]),
              order=1, mode='constant', cval=0.0).reshape(nL, nL)
    p_geom = float((ain[ok_ap] ** 2).sum()) * h * h
    p_disc = p_den * _DX * _DX
    print("  5. ABSOLUTE geometric-transport oracle (raytrace + input field "
          "only)")
    print(f"     alive stop-passing entrance |r| max "
          f"{float(np.hypot(XI, YI)[ok_ap].max())*1e3:.4f} mm "
          f"(aperture radius {_AP*0.5e3:.3f} mm)")
    print(f"     P transported by the alive rays / P over the test's disc = "
          f"{p_geom/p_disc:.5f}   <- the TRUE ratio ceiling")
    print(f"     |ratio - truth|:  C8 OFF {abs(r_off - p_geom/p_disc):.5f}   "
          f"C8 ON {abs(r_on - p_geom/p_disc):.5f}")
    print()

    # ---- 6. what is left INSIDE the support -----------------------------
    for lbl, F in (('C8 OFF', E_off), ('C8 ON ', E_on)):
        print(f"  6. {lbl}: power INSIDE H_all "
              f"{float((np.abs(F)[~out_all]**2).sum())/p_den:.5f} of P_ap "
              f"(the fold's own share; C8 cannot touch it)")
    print()
    # ---- 7. raw warnings, and the radial reach of each field -------------
    print("  7. RAW warnings")
    for lbl, ws in (('C8 OFF', w_off), ('C8 ON ', w_on)):
        if not ws:
            print(f"     {lbl}: (none)")
        for m in ws:
            print(f"     {lbl}: {m[:160]}")
    print()
    R = np.hypot(X, Y)
    s_hull = float(np.hypot(XE[ok_ap], YE[ok_ap]).max())
    d0 = float(np.sqrt(2.0) * 8 * _DX)
    for lbl, F in (('C8 OFF', E_off), ('C8 ON ', E_on)):
        nz = np.abs(F) > 0.0
        print(f"     {lbl}: |E|>0 out to r = {float(R[nz].max())*1e3:.4f} mm; "
              f"support r_max {s_hull*1e3:.4f} mm, plateau d0 "
              f"{d0*1e3:.4f} mm")
    print()
    print("VERDICT INPUTS: (A) needs (c) ~ 0, the pre-C8 excess sitting "
          "outside H_all, and the")
    print("                fold diagnostic still firing.  (B) needs a det J "
          "collapse INSIDE H_ap")
    print("                that C8 removed, i.e. a large (c).")


if __name__ == '__main__':
    sys.exit(main())
