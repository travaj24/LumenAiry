"""VERIFY-WP-C3 -- arbitrate defect D6 with an oracle that does no propagation
at all.

D6 is a leg that resolves a FLAT output reference, where the default runs an
unrepresentable chirp-Z.  Grid refinement already says which arm converges,
but "converges" is not "correct".  This probe settles it with the EXACT
free-space second-moment law, which needs no transform of the answer and no
independent propagator:

    <r^2>(z) = <r^2>(0) + 2 z <r.theta>(0) + z^2 <theta^2>(0)

This is exact for any paraxial field in free space (it is the ABCD transform
of the beam matrix, and it is why M^2 is an invariant).  All three moments are
read off the CHAIN'S OWN EXIT FIELD, which both transports share bit for bit
(the exit plane is before the final leg), so the prediction is common to both
arms and arbitrates between them.

Conventions: this library is ``exp(-i omega t)`` / ``exp(+i k z)``, so the
local ray angle is ``theta = (1/k) grad(phase)`` and the mixed moment is
``<r.theta> = (1/k) Im<E* (x d/dx + y d/dy) E> / <|E|^2>``.  The returned
``field`` is a carrier-referenced ENVELOPE, and the carrier is a pure phase,
so the PHYSICAL field is ``env * exp(i k r^2 / 2R)`` -- reconstructed here
before the moments are taken, because the carrier carries real ray angle.
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)

LAM = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


def moments(E, dx, lam=LAM):
    """``(<r^2>, <r.theta>, <theta^2>)`` of a paraxial field on a square grid."""
    n = E.shape[0]
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    I = np.abs(E) ** 2
    tot = I.sum()
    r2 = float((( xx ** 2 + yy ** 2) * I).sum() / tot)
    # angular second moment from the spectrum (Parseval-consistent)
    S = np.fft.fft2(E)
    Ps = np.abs(S) ** 2
    f = np.fft.fftfreq(n, d=dx)
    fx, fy = np.meshgrid(f, f, indexing='ij')
    th2 = float((((lam * fx) ** 2 + (lam * fy) ** 2) * Ps).sum() / Ps.sum())
    # mixed moment by spectral derivative
    k = 2.0 * np.pi / lam
    dEdx = np.fft.ifft2(S * (2j * np.pi * fx))
    dEdy = np.fft.ifft2(S * (2j * np.pi * fy))
    mix = float(np.imag(np.sum(np.conj(E) * (xx * dEdx + yy * dEdy))) / tot / k)
    return r2, mix, th2


def r2m_of(field, dx):
    n = field.shape[0]
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    I = np.abs(field) ** 2
    return float(np.sqrt((((xx ** 2 + yy ** 2)) * I).sum() / I.sum()))


def main():
    from tests.unit.test_audit2609_b4_collins_transport import _singlet
    p = _singlet(120e-3, -120e-3, 6e-3, 'N-BK7', 25.4e-3, 'p')
    groups = [{'prescription': p, 'gap_before': 20e-3}]
    W, Z = 10.24e-3, 10e-3
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'final_distance': Z, 'rows': []}
    for n in (256, 512, 1024):
        dx = W / n
        row = {'N': n, 'dx_um': dx * 1e6}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex = C.propagate_traced_carrier_chain(
                gauss(n, dx, 2e-3), groups, LAM, dx, r_in=np.inf,
                ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                final_leg='paraxial', final_distance=0.0,
                transport='sziklas')
        dxe = float(ex.dx[0]) if isinstance(ex.dx, tuple) else float(ex.dx)
        Rx, _ry, _ = C._parse_carrier(ex.R, 'p')
        xe = (np.arange(n) - n // 2) * dxe
        X, Y = np.meshgrid(xe, xe, indexing='ij')
        k = 2.0 * np.pi / LAM
        phys = np.asarray(ex.field) * (
            np.exp(1j * k * (X ** 2 + Y ** 2) / (2.0 * Rx))
            if np.isfinite(Rx) and Rx != 0 else 1.0)
        r2, mix, th2 = moments(phys, dxe)
        pred = r2 + 2.0 * Z * mix + Z * Z * th2
        row.update(exit_dx_um=dxe * 1e6, exit_R_mm=Rx * 1e3,
                   r2_exit_um=np.sqrt(r2) * 1e6, mix=mix,
                   theta_rms_mrad=np.sqrt(th2) * 1e3,
                   ORACLE_r2m_um=float(np.sqrt(pred)) * 1e6)
        for tr in ('sziklas', 'collins'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                r = C.propagate_traced_carrier_chain(
                    gauss(n, dx, 2e-3), groups, LAM, dx, r_in=np.inf,
                    ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                    final_leg='paraxial', final_distance=Z, transport=tr)
            d = float(r.dx[0]) if isinstance(r.dx, tuple) else float(r.dx)
            row[f'{tr}_r2m_um'] = r2m_of(np.asarray(r.field), d) * 1e6
            row[f'{tr}_ratio_to_oracle'] = (
                row[f'{tr}_r2m_um'] / row['ORACLE_r2m_um'])
        out['rows'].append(row)
        print(f"N={n:5d}  ORACLE r2m = {row['ORACLE_r2m_um']:10.4f} um  |  "
              f"sziklas {row['sziklas_r2m_um']:10.4f} "
              f"({row['sziklas_ratio_to_oracle']:.4f}x)  |  "
              f"collins {row['collins_r2m_um']:10.4f} "
              f"({row['collins_ratio_to_oracle']:.4f}x)")
    tag = os.environ.get('VC3_TAG', 'x')
    fp = os.path.join(os.environ['VC3_OUT'], f'moment_oracle_{tag}.json')
    with open(fp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', fp)


if __name__ == '__main__':
    main()
