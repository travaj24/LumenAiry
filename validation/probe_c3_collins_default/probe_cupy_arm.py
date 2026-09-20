"""WP-C3 -- the CuPy arm of the Collins chain, measured on whatever this box
actually has.

Run as a CHILD process bound to ONE tree::

    python probe_cupy_arm.py <tree> <out.json>

WHAT IS BEING MEASURED, AND WHY IT IS A MEASUREMENT AND NOT A SKIP.  WP-B4
sec. 5 item 8 names the backends as a precondition for a default every backend
reaches.  Hygiene-2 H2-2 threaded ``(xp, is_jax, bld)`` through the whole
Collins chain and proved the JAX half; its own "decisions owed" item 2 records
that the CuPy half was exercised STRUCTURALLY and not on hardware, because
``cupy.fft`` on this box raises ``ImportError ... cufft``.

This probe asserts nothing.  It RECORDS, as facts of the running box:

* whether CuPy imports, how many devices it sees, whether ELEMENTWISE device
  kernels work, and whether ``cupy.fft`` works;
* for every field-independent helper on the Collins path that does NOT need a
  transform, the device build against the host build, relatively -- these run
  here, so the CuPy half of ``bld`` is covered by arithmetic and not by
  inspection;
* what the PUBLIC leg does with a device array.  On a box whose cuFFT is
  broken the interesting content of that reading is WHICH failure it is: an
  ``ImportError`` naming cufft means the array reached the device transform,
  while a ``TypeError: Implicit conversion to a NumPy array is not allowed``
  would mean the chain demoted it to the host somewhere first.  The second is
  the defect H2-2's V-D3 fixed and the one this campaign must keep fixed;
* on a box with a working cuFFT, the whole leg on the device against the
  NumPy leg, with the bar measured from the two backends' own FFTs.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import clib  # noqa: E402

import numpy as np  # noqa: E402

clib.anchor(_TREE)

import lumenairy.propagators.carrier as CA          # noqa: E402

WL = 633e-9
N = 64
DX = 8e-6


def _gauss(n=N, dx=DX, w=60e-6):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def _premise():
    """The running box's CuPy state, read rather than assumed."""
    out = {'cupy_importable': False, 'n_devices': None,
           'elementwise_ok': False, 'elementwise_error': None,
           'fft_ok': False, 'fft_error': None, 'version': None}
    try:
        import cupy as cp
    except Exception as exc:                          # noqa: BLE001
        out['import_error'] = f'{type(exc).__name__}: {exc}'
        return out, None
    out['cupy_importable'] = True
    out['version'] = cp.__version__
    try:
        out['n_devices'] = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:                          # noqa: BLE001
        out['n_devices'] = f'{type(exc).__name__}: {exc}'
    try:
        a = cp.ones((4, 4), dtype=cp.complex128)
        cp.asnumpy(a * 2.0 + 1.0)
        out['elementwise_ok'] = True
    except Exception as exc:                          # noqa: BLE001
        out['elementwise_error'] = f'{type(exc).__name__}: {exc}'
        return out, cp
    try:
        cp.asnumpy(cp.fft.fft2(cp.ones((4, 4), dtype=cp.complex128)))
        out['fft_ok'] = True
    except Exception as exc:                          # noqa: BLE001
        out['fft_error'] = f'{type(exc).__name__}: {exc}'
    return out, cp


def _device_helpers(cp):
    """Every Collins helper that builds a FIELD-INDEPENDENT grid, device
    against host.  None of these needs a transform, so they run here."""
    rows = []

    def rel(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        n = float(np.linalg.norm(b))
        return float(np.linalg.norm(a - b) / n) if n else float('nan')

    for n, dx, R in ((64, 8e-6, -0.05), (256, 4e-6, -0.02),
                     (512, 3.5e-6, -0.04)):
        d = cp.asnumpy(CA._collins_axis_chirp(n, dx, WL, R, bld=cp))
        h = CA._collins_axis_chirp(n, dx, WL, R, bld=np)
        rows.append({'helper': '_collins_axis_chirp',
                     'n': n, 'dx': dx, 'R': R, 'rel': rel(d, h),
                     'max_abs': float(np.abs(d - h).max()),
                     'ulps_of_one': float(np.abs(d - h).max()
                                          / np.finfo(np.float64).eps)})
    arg = np.linspace(-3.0e4, 3.0e4, 512 * 512).reshape(512, 512)
    Hd = cp.asnumpy(CA._tf_phase_to_H(cp.asarray(arg), np.complex128,
                                      cp, False, cp))
    Hh = CA._tf_phase_to_H(arg, np.complex128, np, False, np)
    rows.append({'helper': '_tf_phase_to_H', 'n': 512, 'rel': rel(Hd, Hh),
                 'max_abs': float(np.abs(Hd - Hh).max())})
    qx = 2.0 * np.pi * np.fft.fftfreq(128, d=4e-6)
    k = 2.0 * np.pi / WL
    for tilt in ((0.0, 0.0), (0.02, -0.01)):
        pd = cp.asnumpy(CA._exact_dispersion_phase(
            cp.asarray(qx), cp.asarray(qx), k, tilt, cp, 'probe'))
        ph = CA._exact_dispersion_phase(qx, qx, k, tilt, np, 'probe')
        rows.append({'helper': '_exact_dispersion_phase', 'tilt': list(tilt),
                     'rel': rel(pd, ph),
                     'max_abs': float(np.abs(pd - ph).max())})
    E = _gauss()
    Ed = cp.asarray(E)
    rows.append({'helper': '_collins_space_support',
                 'device': list(CA._collins_space_support(Ed, DX, DX, 1e-6)),
                 'host': list(CA._collins_space_support(E, DX, DX, 1e-6))})
    xp, is_jax, bld = CA._backend_of(Ed)
    rows.append({'helper': '_backend_of',
                 'xp': getattr(xp, '__name__', str(xp)),
                 'is_jax': bool(is_jax),
                 'bld': getattr(bld, '__name__', str(bld)),
                 'bld_is_xp': bld is xp})
    return rows


def _public_leg(cp, fft_ok):
    """What the PUBLIC leg does with a device array."""
    E = _gauss()
    Ed = cp.asarray(E)
    rec = {}
    try:
        out = CA.propagate_carrier_referenced(
            Ed, -0.05, 5e-3, WL, DX, transport='collins',
            gap_kernel='fresnel', on_collins_sampling='ignore')
        rec['outcome'] = 'ran'
        rec['out_namespace'] = type(out.env).__module__.split('.')[0]
        rec['stayed_on_device'] = (type(out.env).__module__.split('.')[0]
                                   == 'cupy')
        host = CA.propagate_carrier_referenced(
            E, -0.05, 5e-3, WL, DX, transport='collins',
            gap_kernel='fresnel', on_collins_sampling='ignore')
        a = np.asarray(cp.asnumpy(out.env))
        b = np.asarray(host.env)
        rec['rel_vs_numpy'] = float(np.linalg.norm(a - b)
                                    / np.linalg.norm(b))
        # the bar, measured on THIS box from the only thing entitled to
        # differ: one forward transform through each backend's own FFT,
        # times the chain depth the transport applies.
        from lumenairy.propagators.fft_infra import _fft2
        fa = np.asarray(_fft2(np.ascontiguousarray(E, dtype=np.complex128)))
        fb = np.asarray(cp.asnumpy(cp.fft.fft2(cp.asarray(
            E, dtype=cp.complex128))))
        spread = float(np.linalg.norm(fa - fb) / np.linalg.norm(fa))
        rec['single_fft_spread'] = spread
        rec['bar'] = max(6.0 * spread, 32.0 * float(np.finfo(np.float64).eps))
    except Exception as exc:                          # noqa: BLE001
        rec['outcome'] = 'raised'
        rec['error_type'] = type(exc).__name__
        rec['error'] = str(exc)[:400]
        # THE decision this arm exists to record on a broken-cuFFT box:
        # a cuFFT ImportError means the array REACHED the device transform;
        # an implicit-conversion TypeError would mean the chain demoted it.
        rec['reached_device_transform'] = (
            type(exc).__name__ == 'ImportError' and 'cufft' in str(exc).lower())
        rec['demoted_to_host'] = (
            type(exc).__name__ == 'TypeError'
            and 'implicit conversion' in str(exc).lower())
    rec['fft_premise_ok'] = bool(fft_ok)
    return rec


def main():
    out_path = sys.argv[2]
    prem, cp = _premise()
    rec = {'build': clib.build_tag(), 'tree': _TREE,
           'carrier_file': CA.__file__, 'premise': prem}
    if cp is not None and prem['elementwise_ok']:
        rec['device_helpers'] = _device_helpers(cp)
        rec['public_leg'] = _public_leg(cp, prem['fft_ok'])
    else:
        rec['device_helpers'] = None
        rec['public_leg'] = None
    clib.write_json(rec, out_path)
    print(f"[probe_cupy_arm] build={rec['build']} "
          f"cupy={prem['cupy_importable']} devices={prem['n_devices']} "
          f"elementwise={prem['elementwise_ok']} fft={prem['fft_ok']}")
    if rec['public_leg']:
        print(f"  public leg: {rec['public_leg']['outcome']} "
              f"{rec['public_leg'].get('error_type', '')} "
              f"reached_device={rec['public_leg'].get('reached_device_transform')}")


if __name__ == '__main__':
    main()
