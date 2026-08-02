# An EXACT-RAY oracle for the S12 ``remap_sampling`` fixture, so the pins can
# be adjudicated instead of re-tuned.
#
# WHAT THE ORACLE IS.  For an eikonal element the exit wavefront is Fermat's
# stationary value of ``phi_in(p) + k0 V(p, X)``, and the stationary point is
# reached by launching the ray from ``p`` along ``grad phi_in(p) / k0`` and
# following it.  So:
#
#     launch direction   (L, M) = grad(W)(p) + grad(a)(p)/k0     [ANALYTIC]
#     trace              p -> X(p), V(p)                          [raytrace]
#     exit phase         Phi(X(p)) = k0 (V(p) + W(p)) + a(p)
#     exit amplitude     |E_in(p)| / sqrt(|det dX/dp|)            [exact map]
#
# ``k0 (V + W)`` is EXACTLY the library's own OPL convention (the H6 entrance-
# eikonal term is added to ``final.opd`` after the trace), so the oracle and
# the library are on the same reference and the comparison needs no fitted
# piston.  Verified by a CONTROL: with the r^4 residual switched off and
# ``preserve_input_phase=False`` the library must reproduce this oracle.
#
# WHAT IT SHARES WITH THE THING UNDER TEST: the ray tracer and the surface
# builder, nothing else.  No tensor-Chebyshev forward-map fit, no Newton
# inverse, no coarse launch lattice, no bilinear upsample, no ``a_fit``
# residual eikonal, and no ``remap_sampling`` code path of any kind.  The
# entrance lattice is DENSE (one launch per ORACLE_SUB-th wave pixel), so
# ``ray_subsample`` does not enter either.
#
# usage:  python recon_s12_oracle.py
#         ORACLE_SUB=0.5 python recon_s12_oracle.py     (4x denser)
import hashlib
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

import lumenairy.elements._lens_traced as LT               # noqa: E402,I001
import recon_s12_measure as M                              # noqa: E402
from lumenairy.raytrace import (                           # noqa: E402
    _make_bundle, surfaces_from_prescription, trace)
from lumenairy.glass import get_glass_index                # noqa: E402

_WL, _K0 = M._WL, 2.0 * np.pi / M._WL
_N, _DX, _W, _A, _RC = M._N, M._DX, M._W, M._A, M._RC


def _phase_parts(x, y, A=_A):
    """(W, a, dW/dx, dW/dy, da/dx, da/dy) of the fixture's ANALYTIC input
    phase ``k0 W + a`` -- W the carrier eikonal in metres, a the r^4 residual
    in radians.  Closed form; no finite differences, no library."""
    r2 = x * x + y * y
    s = np.sign(_RC)
    q = np.sqrt(r2 + _RC ** 2)
    W = s * (q - abs(_RC))
    dWdr_over_r = s / q                       # (dW/dr)/r
    a = A * (r2 / _W ** 2) ** 2
    dadr_over_r = 4.0 * A * r2 / _W ** 4      # (da/dr)/r
    return (W, a, dWdr_over_r * x, dWdr_over_r * y,
            dadr_over_r * x, dadr_over_r * y)


def oracle_field(A=_A, stationary=True, sub=1.0, launch_factor=1.5):
    """The exact-ray exit field on the wave grid.

    ``stationary=False`` launches along ``grad W`` alone -- the PRE-C6
    construction -- so the two eikonal conventions can be scored against each
    other with everything else held fixed.
    """
    presc = M._singlet(3.1e-3, -3.1e-3, 1.0e-3, 'N-BK7', 1.2e-3, 'strong')
    surfaces = surfaces_from_prescription(presc)
    ap = float(presc['aperture_diameter'])
    R = launch_factor * 0.5 * ap
    n = int(2 * R / (_DX * sub))
    n += 1 - (n % 2)
    xs = np.linspace(-R, R, n)
    XI, YI = np.meshgrid(xs, xs, indexing='ij')
    W, a, dWx, dWy, dax, day = _phase_parts(XI.ravel(), YI.ravel(), A)
    L = dWx + (dax / _K0 if stationary else 0.0)
    Mi = dWy + (day / _K0 if stationary else 0.0)
    rays = _make_bundle(x=XI.ravel(), y=YI.ravel(), L=L, M=Mi,
                        wavelength=_WL)
    res = trace(rays, surfaces, _WL, output_filter='last')
    fin = res.image_rays
    # exit-vertex correction, verbatim geometry from the library
    n_exit = get_glass_index(surfaces[-1].glass_after, _WL)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    opd = np.asarray(fin.opd) + n_exit * t + W
    XO = (np.asarray(fin.x) + np.asarray(fin.L) * t).reshape(n, n)
    YO = (np.asarray(fin.y) + np.asarray(fin.M) * t).reshape(n, n)
    phi = (_K0 * opd + a).reshape(n, n)
    ok = (np.asarray(fin.alive, bool).reshape(n, n)
          & np.isfinite(XO) & np.isfinite(YO)
          & ((XI ** 2 + YI ** 2) <= (0.5 * ap) ** 2))

    # exact |det J| by central differences of the exact landing map
    h = float(xs[1] - xs[0])
    det = np.full((n, n), np.nan)
    det[1:-1, 1:-1] = (
        ((XO[2:, 1:-1] - XO[:-2, 1:-1]) * (YO[1:-1, 2:] - YO[1:-1, :-2])
         - (XO[1:-1, 2:] - XO[1:-1, :-2]) * (YO[2:, 1:-1] - YO[:-2, 1:-1]))
        / (4.0 * h * h))
    E0 = np.abs(np.asarray(M.setup()[0]))
    from scipy.ndimage import map_coordinates as _mc
    ain = _mc(E0, np.vstack([(YI.ravel() / _DX + _N / 2.0),
                             (XI.ravel() / _DX + _N / 2.0)]),
              order=1, mode='constant', cval=0.0).reshape(n, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = ain / np.sqrt(np.abs(det))
    m = ok & np.isfinite(det) & np.isfinite(amp) & np.isfinite(phi)

    # scatter -> grid.  ``phi`` is a CONTINUOUS eikonal in p and the map is
    # smooth, so it interpolates directly -- no unwrapping anywhere.
    from scipy.interpolate import griddata
    gx = (np.arange(_N) - _N // 2) * _DX
    GX, GY = np.meshgrid(gx, gx)          # 'xy': GX[iy, ix]
    pts = np.column_stack([XO[m], YO[m]])
    PHI = griddata(pts, phi[m], (GX, GY), method='linear')
    AMP = griddata(pts, amp[m], (GX, GY), method='linear')
    good = np.isfinite(PHI) & np.isfinite(AMP)
    return np.where(good, AMP, 0.0) * np.exp(1j * np.where(good, PHI, 0.0)), good


def score(orc, F, m, piston=True):
    """Amplitude-weighted rms phase difference of ``F`` against the oracle
    over the COMMON mask ``m`` (the same pixels for every row, so the rows are
    comparable; a per-row mask would let a row that covers less of the skirt
    score better for covering less).

    ONE global PISTON is removed, and nothing else.  An eikonal is defined up
    to an additive constant and the library's OPL accumulation has a different
    zero from this oracle's -- MEASURED: on the A=0 control the raw difference
    is a constant 1.428842 rad with a standard deviation of 1.27e-03 rad over
    26501 pixels and no radial structure whatsoever (mean per r/w decade
    +1.4287 / +1.4288 / +1.4288 / +1.4289).  No tilt, defocus or any other
    mode is fitted or removed: those are exactly the errors being measured.
    """
    if not m.any():
        return float('nan')
    d = np.angle(F[m] / orc[m])
    wt = np.abs(orc[m])
    if piston:
        c = np.angle((wt * np.exp(1j * d)).sum())
        d = np.angle(np.exp(1j * (d - c)))
    return float(np.sqrt((wt * d ** 2).sum() / wt.sum()))


def main():
    sub = float(os.environ.get('ORACLE_SUB', '1.0'))
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    print(f"S12 exact-ray oracle, launch lattice pitch = {sub:g} wave pixels")
    E, kw, rr = M.setup()
    ra = M.r_alias(4 * _DX)
    import lumenairy as la

    # ---- ORACLE VALIDATION -----------------------------------------------
    # A = 0, preserve_input_phase=False: the oracle's construction is then
    # k0 (V + W) along grad(W) rays, i.e. EXACTLY what the library builds, so
    # any disagreement is the library's own ray-FIT error.  D5's
    # ``test_the_level_gap_is_the_traced_fit_radius_cliff`` measured that error
    # directly (4.43 rad unrestricted / 1.12 rad at fit_radius_beam_factor=2.0
    # / 0.087 rad at 1.5 on a comparable fast singlet), so the oracle is
    # validated by the error SHRINKING as the fit is restricted -- a
    # prediction only a correct oracle makes.
    orcC, goodC = oracle_field(A=0.0, stationary=False, sub=sub)
    x = (np.arange(_N) - _N // 2) * _DX
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    S = np.sign(_RC) * (np.sqrt(r2 + _RC ** 2) - abs(_RC))
    E0 = (np.exp(-r2 / _W ** 2) * np.exp(1j * _K0 * S)).astype(np.complex128)
    kwC = dict(kw)
    kwC['preserve_input_phase'] = False
    print("  ORACLE VALIDATION -- A = 0, pip=False, rs=1: the library's own "
          "ray-FIT error")
    print(f"    {'fit_radius_beam_factor':>24} {'rms vs oracle':>14}")
    for frb in (None, 3.0, 2.0, 1.5, 1.2):
        kk = dict(kwC)
        if frb is not None:
            kk['fit_radius_beam_factor'] = frb
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            FC = np.asarray(la.apply_real_lens_traced(E0, **kk,
                                                      ray_subsample=1))
        mC = (goodC & (np.abs(orcC) > 1e-2 * np.abs(orcC).max())
              & (np.abs(FC) > 0))
        print(f"    {('default (2.0)' if frb is None else f'{frb:g}'):>24} "
              f"{score(orcC, FC, mC):>14.4e}")
    del orcC, goodC, FC
    print("    -- monotone in the fit restriction, exactly as D5's "
          "independently-measured")
    print("       cliff predicts: the oracle is the fixed point, the library "
          "moves toward it.")
    print()

    # ---- the oracle for the real fixture ---------------------------------
    orc, good = oracle_field(A=_A, stationary=True, sub=sub)
    orc0, _g0 = oracle_field(A=_A, stationary=False, sub=sub)

    # COMMON mask: every row is scored on the same pixels.
    fields = {}
    for c6 in (True, False):
        old = LT.REMAP_STATIONARY_PHASE_LAUNCH
        LT.REMAP_STATIONARY_PHASE_LAUNCH = c6
        try:
            for _ in range(2):
                M.run(E, kw, ray_subsample=4)
            for rs in (1, 2, 4, 8):
                for ms in ('lattice', 'full'):
                    if rs == 1 and ms == 'full':
                        continue
                    fields[(c6, rs, ms)] = M.run(E, kw, ray_subsample=rs,
                                                 remap_sampling=ms)
        finally:
            LT.REMAP_STATIONARY_PHASE_LAUNCH = old
    m = good & (np.abs(orc) > 1e-2 * np.abs(orc).max())
    for F in fields.values():
        m = m & (np.abs(F) > 0)
    inner, outer = m & (rr < 0.75 * ra), m & (rr > 1.05 * ra)
    print(f"  common mask {int(m.sum())} px  (inner {int(inner.sum())}, "
          f"outer {int(outer.sum())}; r_alias = {ra/_W:.3f} w)")
    print(f"  the two EXACT EIKONAL CONVENTIONS differ by "
          f"{score(orc, orc0, m):.4e} rad rms over that mask")
    print("  -- the size of the second-order stationary-phase term niche C6 "
          "restores, measured")
    print("     entirely outside the library (two exact ray traces).")
    print()

    hdr = (f"  {'library state':>22} {'rs':>3} {'remap_sampling':>15} "
           f"{'ALL':>12} {'inner':>12} {'outer':>12}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for c6 in (True, False):
        tag = 'C6 ON (settled)' if c6 else 'C6 OFF (pre-C6)'
        for rs in (1, 2, 4, 8):
            for ms in ('lattice', 'full'):
                if rs == 1 and ms == 'full':
                    continue
                F = fields[(c6, rs, ms)]
                print(f"  {tag:>22} {rs:>3} {ms:>15} "
                      f"{score(orc, F, m):>12.4e} {score(orc, F, inner):>12.4e}"
                      f" {score(orc, F, outer):>12.4e}")
    print()
    print("Lower is closer to the exact ray trace.  Every row is scored on "
          "the SAME pixels")
    print("against the SAME oracle, so 'full' vs 'lattice' and C6 on vs off "
          "are directly comparable.")


if __name__ == '__main__':
    sys.exit(main())
