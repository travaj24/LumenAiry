# PROBE: is design 121's last COARSE LEG lost to the sphere-plus-ramp carrier
# eikonal?  (Candidate A of the 2026-07-30 tilted-leg brief.)
#
# THEORY.  The chain represents a tilted congruence as
#
#     W_chain(du) = S(|du|; R) + n.du ,   S(rho;R) = sgn(R)(sqrt(rho^2+R^2)-|R|)
#
# (an on-axis sphere about the CHIEF RAY plus a linear ramp).  The EXACT
# eikonal of the congruence -- a point source displaced so that the chief ray
# carries direction cosines n = (L, M) -- is
#
#     W_true(du; R) = sgn(R) ( sqrt(|du|^2 + 2 R n.du + R^2) - |R| )
#
# which is the SAME on-axis form re-centred on the source projection
# du + R n and with the AXIAL radius R*N, N = sqrt(1-|n|^2).  The two differ by
#
#     dW(du; R) = W_true - S - n.du  ~  -(n.du)|du|^2/(2R^2) - (n.du)^2/(2R)
#
# i.e. COMA (linear in field angle) plus ASTIGMATISM (quadratic).
#
# Neither dW is an error on its own -- the chain divides it out at the leg
# entrance and multiplies it back at the exit.  The error is that the two do
# not MATCH: the envelope is transported by the Sziklas-Siegman dilation
# du -> m du (m = R1/R0), so what the chain re-imposes at the exit is
# dW(du/m; R0) while the truth wants dW(du; R1) -- and to leading order
# dW(du/m; R0) = (1/m) dW(du; R1).  The residual
#
#     C(du) = exp( i k0 [ dW(du; R1) - dW(du/m; R0) ] )
#
# is a spurious phase screen the chain writes onto the field at every tilted
# leg, with NO free parameter: R0, R1, m and n are all the chain's own.
#
# A SECOND, smaller term: the carrier's own transport law advances the radius
# by the AXIAL gap z, but a tilted congruence's radius is measured ALONG the
# chief ray, so it should advance by z/N.  MODE=full also corrects that.
#
# METHOD.  ``apply_real_lens_traced`` is monkeypatched for ONE group and the
# incoming field is multiplied by C(du) (or C(du)^SCALE).  No library change.
# SCALE=0 must reproduce the shipped number to every digit.  Scoring is
# ``hybrid_localize_121.main()`` at NMIN=NMAX=6, unchanged, so the EE3 is
# directly comparable with the ablation table (shipped 66.24 at (-4,-2)).
#
# Env: ORD, MODE=eik|full|off, SCALE, GRP (default: every tilted group; set to
# an int for one group), plus every hybrid_localize_121 knob.
import io
import os
import re
import sys
import warnings
from contextlib import redirect_stdout

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C  # noqa: E402

import lumenairy.elements as _EL  # noqa: E402
import lumenairy.elements._lens_traced as _LT  # noqa: E402

_REAL = _EL.apply_real_lens_traced
_PARTS = _LT._tilted_carrier_parts


def _exact_parts(spec, X, Y):
    """``_tilted_carrier_parts`` with the EXACT displaced-point-source
    eikonal instead of the shipped sphere-plus-linear-ramp."""
    s = float(spec.R)
    L, M = float(spec.L), float(spec.M)
    if s == 0.0 or not np.isfinite(s) or (L == 0.0 and M == 0.0):
        return _PARTS(spec, X, Y)
    u = X - float(spec.x0)
    v = Y - float(spec.y0)
    sgn = 1.0 if s > 0.0 else -1.0
    uu = u + s * L
    vv = v + s * M
    rho = np.sqrt(uu * uu + vv * vv + s * s * (1.0 - L * L - M * M))
    W = sgn * (rho - abs(s))
    return W, sgn * uu / rho, sgn * vv / rho


def w_true(du2, ndu, R):
    """Exact displaced-point-source eikonal, metres."""
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(du2 + 2.0 * R * ndu + R * R) - abs(R))


def w_chain(du2, ndu, R):
    """Shipped sphere-plus-ramp eikonal, metres."""
    sgn = 1.0 if R > 0 else -1.0
    return sgn * (np.sqrt(du2 + R * R) - abs(R)) + ndu


def make_patch(names, gaps, mode, scale, log):

    def patched(E_in, *, prescription, wavelength, dx, **kw):
        nm = prescription.get('name')
        car = kw.get('carrier')
        if (nm not in names or car is None
                or not getattr(car, 'is_tilted', False)
                or not np.isfinite(getattr(car, 'R', np.inf))
                or (car.L == 0.0 and car.M == 0.0)):
            return _REAL(E_in, prescription=prescription,
                         wavelength=wavelength, dx=dx, **kw)
        z = float(gaps[nm])
        R1 = float(car.R)
        L, M = float(car.L), float(car.M)
        R0 = R1 - z
        if z == 0.0 or R0 == 0.0 or R1 == 0.0:
            return _REAL(E_in, prescription=prescription,
                         wavelength=wavelength, dx=dx, **kw)
        m = R1 / R0
        s = L * L + M * M
        N = np.sqrt(1.0 - s)
        R1p = R0 + z / N                      # radius ALONG the chief ray
        k0 = 2.0 * np.pi / float(wavelength)
        n_ = np.shape(E_in)[-1]
        ny = np.shape(E_in)[-2]
        u = (np.arange(n_, dtype=np.float64) - n_ / 2) * dx - float(car.x0)
        v = (np.arange(ny, dtype=np.float64) - ny / 2) * dx - float(car.y0)
        U = u[None, :]
        V = v[:, None]
        du2 = U * U + V * V
        ndu = L * U + M * V
        # entrance-plane pull-back du/m
        du2b = du2 / (m * m)
        ndub = ndu / m
        dW0 = w_true(du2b, ndub, R0) - w_chain(du2b, ndub, R0)
        R_ex = R1p if mode == 'full' else R1
        dW1 = w_true(du2, ndu, R_ex) - w_chain(du2, ndu, R1)
        corr = (dW1 - dW0) * float(scale)
        ph = k0 * corr
        # sampling adequacy of the SCREEN, over the illuminated support only
        a = np.abs(np.asarray(E_in))
        wgt = a / max(a.max(), 1e-300)
        st = np.maximum(
            np.abs(np.diff(ph, axis=0)) * np.minimum(wgt[1:], wgt[:-1]),
            0.0)
        st2 = np.abs(np.diff(ph, axis=1)) * np.minimum(wgt[:, 1:], wgt[:, :-1])
        allst = np.concatenate([st.ravel(), st2.ravel()])
        # power-weighted rms / peak of the screen over the beam
        w2 = (a ** 2)
        w2 = w2 / max(w2.sum(), 1e-300)
        pm = float(np.sum(w2 * ph))
        rms = float(np.sqrt(max(np.sum(w2 * (ph - pm) ** 2), 0.0)))
        big = a >= np.exp(-2.0) * a.max()
        log.append(
            f"[{nm}] z={z * 1e3:.4f} mm R0={R0 * 1e3:.4f} R1={R1 * 1e3:.4f} "
            f"R1'={R1p * 1e3:.4f} mm m={m:.6f} |n|={np.sqrt(s):.5f} "
            f"mode={mode} scale={scale}")
        log.append(
            f"[{nm}] screen: rms {rms / (2 * np.pi):.5f} waves, "
            f"|max| over the exp(-2) core {np.abs(ph[big]).max() / (2 * np.pi):.5f} "
            f"waves, per-pixel step (amp-weighted) p50 "
            f"{np.percentile(allst, 50):.2e} p99.9 "
            f"{np.percentile(allst, 99.9):.3f} max {allst.max():.3f} rad "
            f"(pi = 3.1416)")
        E2 = np.asarray(E_in) * np.exp(1j * ph).astype(np.complex128)
        return _REAL(E2.astype(np.asarray(E_in).dtype, copy=False),
                     prescription=prescription, wavelength=wavelength, dx=dx,
                     **kw)

    return patched


def main():
    mode = os.environ.get('MODE', 'eik')
    scale = float(os.environ.get('SCALE', '1.0'))
    os.environ.setdefault('NMIN', '6')
    os.environ.setdefault('NMAX', '6')
    os.environ.setdefault('NOUT', '61')
    os.environ.setdefault('DXO', '0.4')
    os.environ.setdefault('NL', '121')
    _pre, post, _g, _p = C.geometry()
    grp = os.environ.get('GRP', '')
    idx = [int(v) for v in grp.split(',')] if grp else list(range(len(post)))
    names = {post[i]['prescription'].get('name') for i in idx}
    gaps = {post[i]['prescription'].get('name'):
            float(post[i].get('gap_before', 0.0)) for i in range(len(post))}
    log = []
    if mode != 'off':
        _EL.apply_real_lens_traced = make_patch(names, gaps, mode, scale, log)
    eikel = os.environ.get('EIKEL', '0') == '1'
    if eikel:
        _LT._tilted_carrier_parts = _exact_parts
        log.append("[EIKEL] element eikonal = EXACT displaced point source")
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            import hybrid_localize_121 as H
            H.main()
    finally:
        _EL.apply_real_lens_traced = _REAL
        _LT._tilted_carrier_parts = _PARTS
    txt = buf.getvalue()
    print(txt)
    for line in log:
        print(line)
    ee = re.findall(r"EE3\s+([0-9.]+)\s+EE6\s+([0-9.]+)\s+EE12\s+([0-9.]+)",
                    txt)
    fw = re.findall(r"FWHM\s+([0-9.]+)\s+um", txt)
    print(f"RESULT  ORD={os.environ.get('ORD', '-4,-2')}  MODE={mode}  "
          f"SCALE={scale}  GRP={sorted(idx)}  "
          + "  ".join(f"EE3={a} EE6={b} EE12={c} FWHM={f}"
                      for (a, b, c), f in zip(ee, fw)))


if __name__ == '__main__':
    sys.exit(main())
