# ``test_remap_carries_injected_residual`` (tests/unit/
# test_niche_upsample_lattice_fix.py) pins that ``preserve_input_phase='remap'``
# differs from ``preserve_input_phase=False`` by a PHASE-ONLY factor -- same
# |E|, different phase.  Niche C6 breaks that by design: the remap launch is
# now ``grad(W + a_fit)`` rather than ``grad(W)``, so the ray TUBE changes and
# ``ray_density``'s ``1/sqrt(|det J|)`` follows it.
#
# THE QUESTION THIS RUNNER SETTLES.  Is the amplitude that now moves a DEFECT
# (the mode has stopped being a pure phase operator, as the pin says it must
# be) or the CORRECTION (the ray tube of the congruence the input field
# actually defines)?  Those make opposite predictions about an exact ray trace,
# and nothing else has to be assumed.
#
# THE ORACLE.  Two exact constructions of the same fixture, both pure raytrace
# + closed-form input phase -- no fit, no Newton inverse, no coarse lattice, no
# upsample, no ``preserve_input_phase`` code path:
#
#   ORACLE-W    rays along grad(W) alone           = what pip=False builds
#   ORACLE-Wa   rays along grad(W + a/k0)          = Fermat's stationary point
#                                                    of the TOTAL entrance
#                                                    eikonal, i.e. the truth
#
# In both, exit amplitude = |E_in(p)| / sqrt(|det dX/dp|) with det J from the
# EXACT landing map, and exit phase = k0 (V(p) + W(p)) [+ a(p) where carried].
# If C6 is right, ``remap``'s |E| must track ORACLE-Wa and pip=False's must
# track ORACLE-W -- and the two oracles must differ by the amount the test now
# rejects.
#
# usage:  python recon_remap_residual_oracle.py
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
from lumenairy.glass import get_glass_index                # noqa: E402
from lumenairy.raytrace import (                           # noqa: E402
    _make_bundle, surfaces_from_prescription, trace)

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_N, _DX, _W, _RIN, _AP = 512, 10e-6, 0.6e-3, 50e-3, 2.4e-3


def _singlet():
    surfaces = [
        {'radius': 9.0e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': -9.0e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': 'lat', 'aperture_diameter': _AP,
            'surfaces': surfaces, 'thicknesses': [1.5e-3]}


def _parts(x, y):
    """W (m), a (rad) and their radial gradients for the test's own fixture:
    a sphere of radius ``_RIN`` carrying ``0.5 (r^2/w^2)^2 exp(-r^2/2w^2)``."""
    r2 = x * x + y * y
    q = np.sqrt(r2 + _RIN ** 2)
    W = np.sign(_RIN) * (q - abs(_RIN))
    dW_r = np.sign(_RIN) / q                       # (dW/dr)/r
    g = np.exp(-r2 / (2.0 * _W * _W))
    a = 0.5 * (r2 / _W ** 2) ** 2 * g
    # d/dr [0.5 (r^2/w^2)^2 exp(-r^2/2w^2)] / r
    da_r = 0.5 * g * (4.0 * r2 / _W ** 4 - r2 * r2 / _W ** 6)
    return W, a, dW_r * x, dW_r * y, da_r * x, da_r * y


def _fields():
    x = (np.arange(_N) - _N // 2) * _DX
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    S = np.sign(_RIN) * (np.sqrt(r2 + _RIN * _RIN) - abs(_RIN))
    E_in = (np.exp(-r2 / _W ** 2) * np.exp(1j * _K0 * S)).astype(np.complex128)
    g = np.exp(-r2 / (2.0 * _W * _W))
    resid = 0.5 * (r2 / _W ** 2) ** 2 * g
    return E_in, (E_in * np.exp(1j * resid)).astype(np.complex128), resid


def oracle(stationary, carry_residual, sub=1.0, launch_factor=1.5):
    surfaces = surfaces_from_prescription(_singlet())
    R = launch_factor * 0.5 * _AP
    n = int(2 * R / (_DX * sub))
    n += 1 - (n % 2)
    xs = np.linspace(-R, R, n)
    XI, YI = np.meshgrid(xs, xs, indexing='ij')
    W, a, dWx, dWy, dax, day = _parts(XI.ravel(), YI.ravel())
    L = dWx + (dax / _K0 if stationary else 0.0)
    Mi = dWy + (day / _K0 if stationary else 0.0)
    res = trace(_make_bundle(x=XI.ravel(), y=YI.ravel(), L=L, M=Mi,
                             wavelength=_WL), surfaces, _WL,
                output_filter='last')
    fin = res.image_rays
    n_exit = get_glass_index(surfaces[-1].glass_after, _WL)
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    opd = np.asarray(fin.opd) + n_exit * t + W
    XO = (np.asarray(fin.x) + np.asarray(fin.L) * t).reshape(n, n)
    YO = (np.asarray(fin.y) + np.asarray(fin.M) * t).reshape(n, n)
    phi = (_K0 * opd + (a if carry_residual else 0.0)).reshape(n, n)
    ok = (np.asarray(fin.alive, bool).reshape(n, n)
          & np.isfinite(XO) & np.isfinite(YO)
          & ((XI ** 2 + YI ** 2) <= (0.5 * _AP) ** 2))
    h = float(xs[1] - xs[0])
    det = np.full((n, n), np.nan)
    det[1:-1, 1:-1] = (
        ((XO[2:, 1:-1] - XO[:-2, 1:-1]) * (YO[1:-1, 2:] - YO[1:-1, :-2])
         - (XO[1:-1, 2:] - XO[1:-1, :-2]) * (YO[2:, 1:-1] - YO[:-2, 1:-1]))
        / (4.0 * h * h))
    E_in = np.abs(_fields()[0])
    from scipy.ndimage import map_coordinates as _mc
    ain = _mc(E_in, np.vstack([(YI.ravel() / _DX + _N / 2.0),
                               (XI.ravel() / _DX + _N / 2.0)]),
              order=1, mode='constant', cval=0.0).reshape(n, n)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = ain / np.sqrt(np.abs(det))
    m = ok & np.isfinite(det) & np.isfinite(amp) & np.isfinite(phi)
    from scipy.interpolate import griddata
    gx = (np.arange(_N) - _N // 2) * _DX
    GX, GY = np.meshgrid(gx, gx)
    pts = np.column_stack([XO[m], YO[m]])
    PHI = griddata(pts, phi[m], (GX, GY), method='linear')
    AMP = griddata(pts, amp[m], (GX, GY), method='linear')
    good = np.isfinite(PHI) & np.isfinite(AMP)
    return (np.where(good, AMP, 0.0) * np.exp(1j * np.where(good, PHI, 0.0)),
            good)


def _call(E, pip, launch=None):
    old = LT.REMAP_STATIONARY_PHASE_LAUNCH
    if launch is not None:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                E, prescription=_singlet(), wavelength=_WL, dx=_DX,
                carrier=_RIN, ray_subsample=8, n_workers=1,
                on_undersample='silent', on_noncollimated='silent',
                amplitude_model='ray_density', preserve_input_phase=pip))
    finally:
        LT.REMAP_STATIONARY_PHASE_LAUNCH = old


def amp_err(orc, F, m):
    """rms relative amplitude error, |E|-weighted, over the common mask."""
    a, b = np.abs(orc[m]), np.abs(F[m])
    return float(np.sqrt((a * (b - a) ** 2).sum() / (a * a * a).sum()))


def phase_err(orc, F, m):
    d = np.angle(F[m] / orc[m])
    wt = np.abs(orc[m])
    c = np.angle((wt * np.exp(1j * d)).sum())
    d = np.angle(np.exp(1j * (d - c)))
    return float(np.sqrt((wt * d ** 2).sum() / wt.sum()))


def main():
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    print("Adjudicating test_remap_carries_injected_residual against two "
          "EXACT ray constructions.")
    E_in, E_res, resid = _fields()

    oW, gW = oracle(stationary=False, carry_residual=True)
    oWa, gWa = oracle(stationary=True, carry_residual=True)

    F_false = _call(E_res, False)
    F_remap = _call(E_res, 'remap')
    F_remap0 = _call(E_res, 'remap', launch=False)      # pre-C6 remap

    m = gW & gWa
    for F in (F_false, F_remap, F_remap0):
        m = m & (np.abs(F) > 0)
    m = m & (np.abs(oWa) > 5e-2 * np.abs(oWa).max())
    print(f"  common mask {int(m.sum())} px of {_N*_N}")
    print()
    print("  1. Do the two EXACT constructions differ in AMPLITUDE at all?")
    print(f"     ORACLE-W vs ORACLE-Wa: rms rel |E| difference "
          f"{amp_err(oWa, oW, m):.4e}, phase {phase_err(oWa, oW, m):.4e} rad")
    print("     -- if this is ~0 the pin's 'phase-only' premise is exact "
          "physics and C6 broke it;")
    print("        if it is not, the amplitude MUST move and the pin was "
          "wrong about the mechanism.")
    print()
    def l2(orc, F):
        c = np.vdot(orc[m], F[m])
        c = c / abs(c) if abs(c) > 0 else 1.0     # piston only
        return float(np.linalg.norm(F[m] / c - orc[m])
                     / np.linalg.norm(orc[m]))

    hdr = (f"  {'library call':>28} {'|E| err vs ORACLE-Wa':>21} "
           f"{'vs ORACLE-W':>13} {'phase vs Wa':>13} {'relL2 vs Wa':>13}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for lbl, F in (('pip=False', F_false),
                   ("pip='remap' (C6 on, shipped)", F_remap),
                   ("pip='remap' (C6 off, pre-C6)", F_remap0)):
        print(f"  {lbl:>28} {amp_err(oWa, F, m):>21.4e} "
              f"{amp_err(oW, F, m):>13.4e} {phase_err(oWa, F, m):>13.4e} "
              f"{l2(oWa, F):>13.4e}")
    print()

    # 2. the test's own statistics, in both library states
    print("  2. The test's own statistics")
    for lbl, F in (("C6 on (shipped)", F_remap), ("C6 off (pre-C6)", F_remap0)):
        a_f, a_r = np.abs(F_false), np.abs(F)
        tolv = 1e-12 + 1e-6 * a_f.max()
        worst = float(np.abs(a_f - a_r).max())
        mm = a_f > 0.05 * a_f.max()
        dphi = np.angle(F[mm] * np.conj(F_false[mm]))
        inj = resid[mm]
        print(f"     {lbl}: allclose(|E|) tol {tolv:.3e}, worst |d|E|| "
              f"{worst:.4e} -> {'PASS' if worst <= tolv else 'FAIL'}")
        print(f"                     std(dphi)/std(inj) = "
              f"{np.std(dphi)/np.std(inj):.4f}   (pin: 0.3 .. 3.0)")
    print()

    # 3. energy
    p_in = float((np.abs(E_res) ** 2).sum())
    print("  3. Power, over the input power on the grid")
    for lbl, F in (('ORACLE-Wa', oWa), ('ORACLE-W', oW), ('pip=False', F_false),
                   ("remap C6 on", F_remap), ("remap C6 off", F_remap0)):
        print(f"     {lbl:>14}: {float((np.abs(F)**2).sum())/p_in:.6f}")


if __name__ == '__main__':
    sys.exit(main())
