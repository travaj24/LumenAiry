"""Which reconstruct route does the P4 path take, and is that DECISION near a
boundary that resource pressure could flip?

``_fft_reconstruct_applicable`` swallows ``Exception`` (MemoryError included)
and falls back to the windowed sum, which is a DIFFERENT summation order.  If
the P4 bundle were near that decision boundary, memory pressure really could
change the bits -- so measure where the decision sits.
"""
import numpy as np
import common as c
from lumenairy.propagators import gbd as G

calls = []
_orig_app = G._fft_reconstruct_applicable
_orig_impl = G._fft_applicable_impl
_orig_win = G._reconstruct_windowed
_orig_fft = G._reconstruct_fft


def spy_app(beamlets, Nx, Ny, dx, dy, centre):
    try:
        impl = _orig_impl(beamlets, Nx, Ny, dx, dy, centre)
        err = None
    except Exception as e:                                    # noqa: BLE001
        impl, err = False, f'{type(e).__name__}: {e}'
    dr = np.asarray(beamlets.directions) if beamlets.directions is not None else None
    Q = np.asarray(beamlets.Q)
    calls.append(dict(
        n=int(np.asarray(beamlets.positions).shape[0]),
        impl=bool(impl), impl_error=err,
        dir_ptp=(None if dr is None
                 else [float(np.ptp(dr[:, 0])), float(np.ptp(dr[:, 1]))]),
        Q_ptp=float(np.ptp(np.real(Q))) if Q.ndim == 1 else None,
        Q_ndim=int(Q.ndim)))
    return _orig_app(beamlets, Nx, Ny, dx, dy, centre)


def spy_win(*a, **k):
    calls[-1]['route'] = 'windowed'
    return _orig_win(*a, **k)


def spy_fft(*a, **k):
    calls[-1]['route'] = 'fft'
    return _orig_fft(*a, **k)


G._fft_reconstruct_applicable = spy_app
G._reconstruct_windowed = spy_win
G._reconstruct_fft = spy_fft

E = c.conv_input()
d = {}
c.gbd(E, c.m5_biconcave(), reexpand='auto', diagnostics=d)
for i, r in enumerate(calls):
    print(f"reconstruct[{i}]: n={r['n']} route={r.get('route')} "
          f"fft_applicable={r['impl']} err={r['impl_error']} "
          f"dir_ptp={r['dir_ptp']} (boundary 1e-12) Q_ndim={r['Q_ndim']} "
          f"Q_ptp={r['Q_ptp']}")
