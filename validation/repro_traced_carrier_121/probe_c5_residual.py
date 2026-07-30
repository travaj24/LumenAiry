# Niche C5: what does the last group's entrance residual look like under the
# two carrier conventions, ON THE SAME (C5-corrected) FIELD?
#
# The element's `preserve_input_phase='remap'` model error is known to track
# ``grad a``, the input residual's own transverse slope (DIAG_LAST_GROUP_
# DECENTRE_2026_07_30 S3): remap launches along grad(W) alone and evaluates
# the residual at THAT ray's foot.  So which carrier gives the SMALLER
# ``grad a`` decides which carrier the element prefers -- a different question
# from which carrier the CHAIN must transport against.
#
# No unwrap, no FFT derivative: the slope is a WRAPPED nearest-neighbour
# central difference of the residual phasor, amplitude-weighted, and the
# per-pixel step is printed so the reading can be checked against pi.
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C  # noqa: E402

import lumenairy.elements as _EL  # noqa: E402
import lumenairy.elements._lens_traced as _LT  # noqa: E402

LAM = C.LAM
K0 = 2 * np.pi / LAM
_REAL = _EL.apply_real_lens_traced


def parts(spec, X, Y, exact):
    f = _LT.TILTED_CARRIER_EXACT_EIKONAL
    _LT.TILTED_CARRIER_EXACT_EIKONAL = exact
    try:
        return _LT._tilted_carrier_parts(spec, X, Y)
    finally:
        _LT.TILTED_CARRIER_EXACT_EIKONAL = f


def main():
    m, n = (int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    grab = {}

    def patched(E_in, *, prescription, wavelength, dx, **kw):
        grab.setdefault(prescription.get('name'),
                        (np.array(E_in), kw.get('carrier'), float(dx)))
        return _REAL(E_in, prescription=prescription, wavelength=wavelength,
                     dx=dx, **kw)

    _pre, post, _g, period = C.geometry()
    env, R, dx, _P = C.chain_a(n=1024)
    L, M = m * LAM / period, n * LAM / period
    _EL.apply_real_lens_traced = patched
    try:
        C.la.propagate_traced_carrier_chain(
            env, post, LAM, dx, r_in=C.la.TiltedCarrier(R, L, M),
            ray_subsample=4, n_workers=8, final_distance=0.0,
            final_leg='paraxial', on_decentred_fit='ignore')
    finally:
        _EL.apply_real_lens_traced = _REAL

    for name in [g['prescription'].get('name') for g in post]:
        E, car, dxg = grab[name]
        if car is None or not getattr(car, 'is_tilted', False):
            continue
        N = E.shape[-1]
        t = (np.arange(N, dtype=np.float64) - N / 2) * dxg
        X, Y = np.meshgrid(t, t)
        a_amp = np.abs(E)
        wq = a_amp ** 2
        wq = wq / max(wq.sum(), 1e-300)
        row = [f"{name:12s} R {car.R * 1e3:9.4f} mm  |n| "
               f"{np.hypot(car.L, car.M):.6f}"]
        for lbl, ex in (('exact     ', True), ('sphere+ramp', False)):
            W, _, _ = parts(car, X, Y, ex)
            res = E * np.exp(-1j * K0 * W)
            u = res / np.maximum(np.abs(res), 1e-300)
            # wrapped central differences of the unit phasor -> slope, rad
            gx = np.zeros_like(X)
            gy = np.zeros_like(X)
            gx[:, 1:-1] = np.angle(u[:, 2:] * np.conj(u[:, :-2])) \
                / (2 * dxg * K0)
            gy[1:-1, :] = np.angle(u[2:, :] * np.conj(u[:-2, :])) \
                / (2 * dxg * K0)
            st = np.abs(np.angle(u[:, 1:] * np.conj(u[:, :-1])))
            g2 = gx * gx + gy * gy
            rms = float(np.sqrt(max(np.sum(wq * g2), 0.0)))
            mn = (float(np.sum(wq * gx)), float(np.sum(wq * gy)))
            # slope with the mean (a pure pointing offset) removed
            rms0 = float(np.sqrt(max(np.sum(wq * ((gx - mn[0]) ** 2
                                                  + (gy - mn[1]) ** 2)), 0.0)))
            aw = np.minimum(a_amp[:, 1:], a_amp[:, :-1]) / a_amp.max()
            row.append(f"    grad a [{lbl}] rms {rms * 1e3:7.4f} mrad, "
                       f"mean-removed {rms0 * 1e3:7.4f} mrad, "
                       f"mean ({mn[0] * 1e3:+.4f},{mn[1] * 1e3:+.4f}) mrad"
                       f"   nn-step p99.9 (amp-wt) "
                       f"{np.percentile(st * aw, 99.9):.4f} rad")
        print("\n".join(row))


if __name__ == '__main__':
    sys.exit(main())
