"""PMMStack(layer_grids='per-layer') -- audit R-6 (2026-07-28).

Per-layer element grids (own walls + the two neighbours' walls, the
interface-conforming enrichment) coupled by an exact L2 mortar
(:func:`_interface_smatrix_mortar`).  Pins measured on the implementation run:

* identical-grid stacks and 2-layer stacks reproduce the shared grid
  BIT-EXACTLY (a 2-layer neighbour union IS the full union);
* the mortar on identical grids reduces to :func:`_interface_smatrix` to
  solver round-off (the algebraic identity behind the fast path);
* lossless energy closure of a synthetic colliding-wall staircase converges
  spectrally (measured 1.10e-4 at degree 6 -> 1.17e-6 at degree 10 -- the
  non-conforming mortar residual, NOT round-off: tolerances carry headroom);
* every unsupported combination raises loudly.
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm._core import (
    _C,
    _build_sem_tensor_segments,
    _interface_smatrix,
    _interface_smatrix_mortar,
    _sem_cross_mass,
    _sem_mass_exact,
    _sem_modes_tensor,
    _tensor3_dict,
)

WL = 700e-9
P = 1.0e-6


def _stack(layers, grids, degree=6, ffo=7, **kw):
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  far_field_orders=ffo, layer_grids=grids, **kw)
    for t, segs in layers:
        st.add_layer(t, segments=segs)
    return st


def _solve(st, th=8.0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(WL, theta=np.deg2rad(th)).solve()
    return np.asarray(R), np.asarray(T), np.asarray(J)


LAY_COMMON = [(200e-9, [(0.35, 4.0 + 0j), (0.65, 1.0 + 0j)]),
              (150e-9, [(0.35, 2.25 + 0j), (0.65, 1.0 + 0j)]),
              (120e-9, [(0.35, 6.25 + 0.1j), (0.65, 1.0 + 0j)])]
LAY_TWO = [(220e-9, [(0.30, 4.0 + 0j), (0.70, 1.0 + 0j)]),
           (180e-9, [(0.50, 2.25 + 0j), (0.50, 1.0 + 0j)])]


def test_identical_wall_stack_is_bit_exact_vs_shared():
    Rs, Ts, Js = _solve(_stack(LAY_COMMON, "shared"))
    Rp, Tp, Jp = _solve(_stack(LAY_COMMON, "per-layer"))
    assert float(np.max(np.abs(Js - Jp))) == 0.0
    assert float(np.max(np.abs(Rs - Rp))) == 0.0
    assert float(np.max(np.abs(Ts - Tp))) == 0.0


def test_two_layer_stack_is_bit_exact_vs_shared():
    # each layer's neighbour union IS the full 2-layer union -> conforming
    # interfaces -> the per-layer path must reproduce shared bit-for-bit.
    for deg in (6, 10):
        Rs, Ts, Js = _solve(_stack(LAY_TWO, "shared", degree=deg))
        Rp, Tp, Jp = _solve(_stack(LAY_TWO, "per-layer", degree=deg))
        assert float(np.max(np.abs(Js - Jp))) == 0.0
        assert float(np.max(np.abs(Rs - Rp))) == 0.0


def test_mortar_reduces_to_plain_interface_on_identical_grids():
    w = np.array([0.4, 0.6])
    t1 = [_tensor3_dict(np.eye(3) * (4.0 + 0j)),
          _tensor3_dict(np.eye(3) * (1.0 + 0j))]
    t2 = [_tensor3_dict(np.eye(3) * (2.25 + 0j)),
          _tensor3_dict(np.eye(3) * (1.0 + 0j))]
    k0 = 2.0 * np.pi / WL
    m1 = _build_sem_tensor_segments(P, w, t1, 5, 1, True)
    m2 = _build_sem_tensor_segments(P, w, t2, 5, 1, True)
    W1, V1, _l, _q = _sem_modes_tensor(m1, k0, 0.1 * k0, True)
    W2, V2, _l, _q = _sem_modes_tensor(m2, k0, 0.1 * k0, True)
    ref = _interface_smatrix(W1, V1, W2, V2)
    M1 = _sem_mass_exact(m1)
    M2 = _sem_mass_exact(m2)
    C = _sem_cross_mass(m1, m2)
    got = _interface_smatrix_mortar(W1, V1, W2, V2, M1, M2, C)
    for a, b in zip(ref, got):
        assert float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) < 1e-9


def test_energy_closure_synthetic_staircase():
    # six lossless slices whose walls shift 4 nm per slice: the colliding-wall
    # staircase geometry with no metals.  Measured mortar residual: 1.10e-4 at
    # degree 6, 1.17e-6 at degree 10 (spectral decay) -- pinned with headroom.
    lay = [(60e-9, [(0.5 - 0.35 / 2 - 0.002 * i, 1.0 + 0j),
                    (0.35 + 0.004 * i, 4.0 + 0j),
                    (0.5 - 0.35 / 2 - 0.002 * i, 1.0 + 0j)])
           for i in range(6)]
    R6, T6, _ = _solve(_stack(lay, "per-layer", degree=6, ffo=11))
    R10, T10, _ = _solve(_stack(lay, "per-layer", degree=10, ffo=11))
    e6 = float(np.max(np.abs(R6.sum(axis=1) + T6.sum(axis=1) - 1.0)))
    e10 = float(np.max(np.abs(R10.sum(axis=1) + T10.sum(axis=1) - 1.0)))
    assert e6 < 5e-4
    assert e10 < 2e-5
    assert e10 < e6            # the residual must DECAY with degree


def test_unsupported_combinations_raise():
    st = _stack(LAY_TWO, "per-layer")
    st.set_source(WL, theta=0.1)
    with pytest.raises(NotImplementedError, match="per-layer"):
        st.solve(stabilize="slices")
    with pytest.raises(NotImplementedError, match="per-layer"):
        st.prepare()
    sl = PMMStack(P, degree=6, far_field_orders=7, layer_grids="per-layer")
    sl.add_layer(100e-9, segments=[(0.4, 4.0 + 0j), (0.6, 1.0 + 0j)],
                 slant_angle=0.2)
    with pytest.raises(NotImplementedError, match="per-layer"):
        sl.set_source(WL, theta=0.1).solve()
    with pytest.raises(ValueError, match="layer_grids"):
        PMMStack(P, layer_grids="bananas")

def test_conical_per_layer_matches_shared_bit_exact():
    # conical (phi != 0) v2: a 2-layer patterned stack's neighbour window IS
    # the full union, so per-layer conical must reproduce the shared-grid
    # nodal conical cascade bit-for-bit (same grids, same eigs, plain
    # interfaces).
    for phi in (0.3, 1.2):
        Rs = Ts = Js = Rp = Tp = Jp = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, Rs, Ts, Js = (_stack(LAY_TWO, "shared")
                             .set_source(WL, theta=0.15, phi=phi).solve())
            o, Rp, Tp, Jp = (_stack(LAY_TWO, "per-layer")
                             .set_source(WL, theta=0.15, phi=phi).solve())
        assert float(np.max(np.abs(np.asarray(Js) - np.asarray(Jp)))) == 0.0
        assert float(np.max(np.abs(np.asarray(Rs) - np.asarray(Rp)))) == 0.0


def test_retain_internal_per_layer_fields_and_absorption():
    # identical-wall stack: per-layer grids == shared grid, so fields and
    # absorption must match the shared path to solver round-off; a lossy
    # different-wall stack must close energy: sum(A) ~= 1 - R - T (the
    # method's own invariant, flux-vs-farfield).
    zprobe = np.array([50e-9, 250e-9, 400e-9])
    ss = _stack(LAY_COMMON, "shared")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, Rs, Ts, _ = ss.set_source(WL, theta=np.deg2rad(8.0)).solve(
            retain_internal=True)
        Fs = ss.internal_field(zprobe, nx=32)
        As = ss.layer_absorption()
    sp = _stack(LAY_COMMON, "per-layer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, Rp, Tp, _ = sp.set_source(WL, theta=np.deg2rad(8.0)).solve(
            retain_internal=True)
        Fp = sp.internal_field(zprobe, nx=32)
        Ap = sp.layer_absorption()
    for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        assert float(np.max(np.abs(np.asarray(Fs[c])
                                   - np.asarray(Fp[c])))) < 1e-8
    assert float(np.max(np.abs(np.asarray(As) - np.asarray(Ap)))) < 1e-10
    # per-layer internal_field REQUIRES nx (no single shared nodal axis)
    with pytest.raises(ValueError, match="nx"):
        sp.internal_field(100e-9)
    # lossy DIFFERENT-wall stack: absorption closure within the mortar band
    lay = [(200e-9, [(0.30, 4.0 + 0j), (0.70, 1.0 + 0j)]),
           (150e-9, [(0.45, 6.25 + 0.30j), (0.55, 1.0 + 0j)]),
           (120e-9, [(0.60, 2.25 + 0j), (0.40, 1.0 + 0j)])]
    sq = _stack(lay, "per-layer", degree=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, _ = sq.set_source(WL, theta=np.deg2rad(8.0)).solve(
            retain_internal=True)
        A = sq.layer_absorption()
    R = np.asarray(R)
    T = np.asarray(T)
    A = np.asarray(A)
    for col in range(2):
        gap = 1.0 - R.sum(axis=1)[col] - T.sum(axis=1)[col]
        assert abs(float(A[:, col].sum()) - gap) < 5e-3


def test_solve_vs_wavelength_per_layer():
    # (a) the per-layer sweep must be bit-identical to per-wavelength
    # per-layer solve() on the propagating orders; (b) on a conforming
    # (2-layer) stack it must also match the SHARED sweep bit-for-bit.
    wls = [640e-9, 700e-9, 760e-9]
    sp = _stack(LAY_TWO, "per-layer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_sw, R_sw, T_sw, J_sw = sp.solve_vs_wavelength(
            wls, angle=np.deg2rad(8.0), jones=True, max_workers=1)
        for iw, w in enumerate(wls):
            o1, R1, T1, J1 = sp.set_source(
                w, theta=np.deg2rad(8.0)).solve()
            m = np.isin(o1, o_sw)
            msw = np.isin(o_sw, o1)
            # ULP-level: the sweep pins worker BLAS threads
            # (_blas_threads_quiet) while solve() uses the ambient pool
            assert float(np.max(np.abs(np.asarray(R1)[:, m]
                                       - R_sw[iw][:, msw]))) < 1e-14
            assert float(np.max(np.abs(np.asarray(J1) - J_sw[iw]))) < 1e-12
        ss = _stack(LAY_TWO, "shared")
        o_sh, R_sh, T_sh, J_sh = ss.solve_vs_wavelength(
            wls, angle=np.deg2rad(8.0), jones=True, max_workers=1)
    m2 = np.isin(o_sh, o_sw)
    m2b = np.isin(o_sw, o_sh)
    assert float(np.max(np.abs(R_sh[:, :, m2] - R_sw[:, :, m2b]))) < 1e-14
    assert float(np.max(np.abs(J_sh - J_sw))) < 1e-12
