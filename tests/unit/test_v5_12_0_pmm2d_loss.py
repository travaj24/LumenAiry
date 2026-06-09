"""Post-reorg audit fix: the 2-D PMM solvers must handle LOSS (Im(eps)>0 public).

The HYBRID `pmm_efficiency_2d` ran its modal solve in the internal exp(+iwt)
convention but did NOT conjugate the public eps into it, so a passive lossy
material was read as GAIN -- R+T>1, negative absorptance, T00>1 (lossless masked
it; conj of real eps is identity).  Fixed by the conjugation bridge.  The
STAGGERED `pmm_efficiency_2d_staggered` is built in the PUBLIC exp(-iwt)
convention throughout (its docstring: Im(eps)>0 for loss), so it was ALWAYS
passive under loss -- these tests pin that it stays so (a speculative conjugation
"fix" to it was a regression and was reverted).

See docs/audits/RCWA_PMM_POST_REORG_AUDIT_2026_06_09.md.
"""
import warnings

import numpy as np

from lumenairy.elements.pmm import pmm_efficiency_2d, pmm_efficiency_2d_staggered
from lumenairy.elements.rcwa import rcwa_efficiency_2d

warnings.simplefilter("ignore")

_PX = _PY = 0.6e-6
_DEPTH = 0.3e-6
_WL = 0.633e-6
_FILL = 0.5
_X0, _X1 = (1 - _FILL) / 2 * _PX, (1 + _FILL) / 2 * _PX
_Y0, _Y1 = (1 - _FILL) / 2 * _PY, (1 + _FILL) / 2 * _PY


def _rcwa_oracle_A(eps_pillar, n):
    S = 256
    xs = (np.arange(S) + 0.5) / S * _PX
    ys = (np.arange(S) + 0.5) / S * _PY
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cell = np.where((X >= _X0) & (X <= _X1) & (Y >= _Y0) & (Y <= _Y1),
                    eps_pillar, 1.0).astype(complex)
    o, R, T = rcwa_efficiency_2d(_PX, _PY, cell, 1.0, 1.0, _DEPTH, _WL,
                                 theta=1e-4, phi=0.0, polarization="te",
                                 n_orders_x=n, n_orders_y=n, formulation="li")
    return 1.0 - float(R.sum()) - float(T.sum())


def test_hybrid_2d_lossy_is_passive_and_matches_rcwa():
    """Hybrid 2-D PMM under loss: R+T <= 1 (passive, not gain) and absorptance
    matches the RCWA-2D oracle to the ~few-1e-3 hybrid-vs-FMM tolerance."""
    for loss in (0.5, 1.5):
        epp = 2.25 + loss * 1j
        o, R, T = pmm_efficiency_2d(
            _PX, _PY, epp, 1.0, (_X0, _X1), (_Y0, _Y1), 1.0, 1.0, _DEPTH, _WL,
            degree=11, polarization="te", theta=0.0, n_orders=9)
        rt = float(np.asarray(R).sum() + np.asarray(T).sum())
        assert rt <= 1.0 + 1e-6, f"loss={loss}: R+T={rt} > 1 (loss read as gain)"
        A_pmm = 1.0 - rt
        A_oracle = _rcwa_oracle_A(epp, 9)
        assert abs(A_pmm - A_oracle) < 5e-3, (
            f"loss={loss}: A_pmm={A_pmm:.4f} vs A_rcwa={A_oracle:.4f}")


def test_hybrid_2d_lossless_unchanged():
    """The conjugation bridge is identity on real eps -- the validated dielectric
    path stays exactly energy-conserving."""
    o, R, T = pmm_efficiency_2d(
        _PX, _PY, 2.25 + 0j, 1.0, (_X0, _X1), (_Y0, _Y1), 1.0, 1.0, _DEPTH, _WL,
        degree=11, polarization="te", theta=0.0, n_orders=9)
    rt = float(np.asarray(R).sum() + np.asarray(T).sum())
    assert abs(rt - 1.0) < 1e-9


def test_staggered_2d_lossy_is_passive():
    """Staggered 2-D PMM (public exp(-iwt) convention) is passive under loss:
    R+T <= 1 (NOT >1).  Minimal config -- the staggered solve is expensive; this
    pins passivity/convention, not per-order accuracy."""
    Nx = Ny = 4
    mask = np.zeros((Nx, Ny), bool)
    mask[1:3, 1:3] = True
    cell = np.where(mask, 2.25 + 0.8j, 1.0).astype(complex)
    o, R, T = pmm_efficiency_2d_staggered(
        _PX, _PY, cell, 1.0, 1.0, _DEPTH, _WL, degree=3, polarization="te",
        theta=0.0, n_orders=2)
    rt = float(np.asarray(R).sum() + np.asarray(T).sum())
    assert rt <= 1.0 + 1e-6, f"staggered R+T={rt} > 1 (loss read as gain)"
