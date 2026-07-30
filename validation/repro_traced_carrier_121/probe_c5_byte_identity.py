# Niche C5 acceptance on the REAL design 121.
#
#  (a) BYTE IDENTITY of the untilted (order (0,0)) post-DOE chain with the C5
#      flag on and off -- np.array_equal on the returned field, over several
#      configurations (grid, ray_subsample, final_leg, focus_readout).  The
#      shipped single-beam acceptance must not re-baseline.
#  (b) the FAIL-BEFORE switch: a TILTED order reproduces the pre-C5 field bit
#      for bit with the flag off.
#  (c) TAPER sensitivity: scale the band-limit radius of the C5 term by 0.5x /
#      2x / infinity (no taper at all) and show the tilted field does not move
#      -- i.e. the mixed-convention skirt is inert, measured not assumed.
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C  # noqa: E402

import lumenairy.elements._lens_traced as _LT  # noqa: E402
import lumenairy.propagators.carrier as _CA  # noqa: E402

LAM = C.LAM
_REAL_XF = _CA._tilt_exactness_phase


def run(order, flag, rn=1024, rs=4, leg='paraxial', fr=None, ngroups=None):
    _LT.TILTED_CARRIER_EXACT_EIKONAL = flag
    try:
        pre, post, _g, period = C.geometry()
        env, R, dx, _P = C.chain_a(n=rn, rs=rs)
        L, M = order[0] * LAM / period, order[1] * LAM / period
        kw = {}
        if fr is not None:
            kw['focus_readout'] = fr
        res = C.la.propagate_traced_carrier_chain(
            env, post[:ngroups], LAM, dx,
            r_in=C.la.TiltedCarrier(R, L, M), ray_subsample=rs, n_workers=8,
            final_distance=0.0, final_leg=leg, on_decentred_fit='ignore',
            on_na_proximity='ignore', on_tilt_exact_grid='ignore',
            on_rs_fine_clamp='ignore', **kw)
        return np.asarray(res.field)
    finally:
        _LT.TILTED_CARRIER_EXACT_EIKONAL = True


def _tapered(scale):
    """``_tilt_exactness_phase`` with its band-limit radius multiplied by
    ``scale`` and NOTHING else changed (the coordinate grid, and therefore the
    term itself, is untouched -- scaling ``dx`` would have moved both)."""
    def f(shape, dx, dy, wavelength, R, L, M, sign, centre=(0.0, 0.0)):
        if not _LT.TILTED_CARRIER_EXACT_EIKONAL:
            return None
        L = float(L)
        M = float(M)
        if (L == 0.0 and M == 0.0) or not np.isfinite(R) or R == 0.0:
            return None
        ny, nx = int(shape[-2]), int(shape[-1])
        x = (np.arange(nx, dtype=np.float64) - nx / 2) * dx - float(centre[0])
        y = (np.arange(ny, dtype=np.float64) - ny / 2) * dy - float(centre[1])
        X, Y = x[None, :], y[:, None]
        sgn = 1.0 if R > 0 else -1.0
        s = L * L + M * M
        n_par = np.sqrt(1.0 - s)
        uu = X + R * L / n_par
        vv = Y + R * M / n_par
        r2 = X * X + Y * Y
        D = (sgn * (np.sqrt(uu * uu + vv * vv + R * R) - abs(R) / n_par)
             - sgn * (np.sqrt(r2 + R * R) - abs(R)) - (L * X + M * Y))
        a = 1.5 * np.sqrt(s) / (R * R)
        b = s / abs(R)
        c = wavelength / (2.0 * min(float(dx), float(dy)))
        r_safe = (np.sqrt(b * b + 4.0 * a * c) - b) / (2.0 * a) * scale
        t = np.clip((np.sqrt(r2) - 0.75 * r_safe) / (0.25 * r_safe), 0.0, 1.0)
        k = 2.0 * np.pi / wavelength
        return np.exp(sign * 1j * k * D * np.cos(0.5 * np.pi * t) ** 2)
    return f


def main():
    print("=== (a) UNTILTED (order 0,0): C5 ON vs OFF, np.array_equal ===")
    cases = [
        ('RN=1024 rs=4 paraxial', dict(rn=1024, rs=4)),
        ('RN=1024 rs=2 paraxial', dict(rn=1024, rs=2)),
        ('RN=2048 rs=4 paraxial', dict(rn=2048, rs=4)),
        ('RN=1024 rs=4, 3 groups', dict(rn=1024, rs=4, ngroups=3)),
        ('RN=1024 rs=4, 5 groups', dict(rn=1024, rs=4, ngroups=5)),
        ('RN=1024 rs=4 focus_readout paraxial',
         dict(rn=1024, rs=4, fr=dict(dx_out=0.4e-6, N_out=64))),
        ('RN=1024 rs=4 final_leg=exact',
         dict(rn=1024, rs=4, leg='exact',
              fr=dict(dx_out=0.4e-6, N_out=64, n_fine_cap=8192))),
    ]
    ok = True
    for lbl, kw in cases:
        a = run((0, 0), True, **kw)
        b = run((0, 0), False, **kw)
        eq = bool(np.array_equal(a, b)) and a.dtype == b.dtype
        ok &= eq
        print(f"  {lbl:42s} shape {a.shape} {a.dtype}  array_equal={eq}  "
              f"max|dE| {float(np.abs(a - b).max()):.3e}")

    print("=== (b) TILTED (-4,-2): the fail-before switch is EXACT ===")
    a = run((-4, -2), False)
    b = run((-4, -2), False)
    c = run((-4, -2), True)
    print(f"  flag OFF, two runs      array_equal={np.array_equal(a, b)}  "
          f"(determinism control)")
    print(f"  flag OFF vs ON          array_equal={np.array_equal(a, c)}  "
          f"max|dE| {float(np.abs(a - c).max()):.3e}  (C5 really acts)")
    ok &= bool(np.array_equal(a, b)) and not np.array_equal(a, c)

    print("=== (c) TAPER sensitivity on the tilted order (-4,-2) ===")
    ref = run((-4, -2), True)
    for scale, lbl in ((0.5, 'r_safe x 0.5'), (1.5, 'r_safe x 1.5'),
                       (1e6, 'r_safe -> huge (NO taper at all)')):
        try:
            _CA._tilt_exactness_phase = _tapered(scale)
            f = run((-4, -2), True)
        finally:
            _CA._tilt_exactness_phase = _REAL_XF
        num = float(np.abs(f - ref).max())
        den = float(np.abs(ref).max())
        _p = float(np.sum(np.abs(ref) ** 2))
        _d = float(np.sum(np.abs(f - ref) ** 2))
        print(f"  {lbl:30s} max|dE|/max|E| = {num / den:.3e}   "
              f"||dE||^2/||E||^2 = {_d / _p:.3e}")
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
