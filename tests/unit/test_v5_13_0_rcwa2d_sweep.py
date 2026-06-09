"""v5.13.0 -- 2-D RCWA wavelength-sweep reuse (``PreparedRCWA2D`` /
``prepare_rcwa_2d`` / ``rcwa_efficiency_2d_vs_wavelength``).

The prepared object hoists the GEOMETRY-ONLY permittivity factorization (the
Laurent ``[[eps]]``, the Li ``[[1/eps]]`` z-rule, and the fff_nv normal-vector
tensor incl. the ``O(N^3)`` ``inv([[1/eps]])``) + order set + incident vector,
so a wavelength sweep recomputes only the per-wavelength eig + S-matrix.  These
tests pin that ``prepared.solve(wl)`` reproduces ``rcwa_efficiency_2d(...)`` at
the same wavelength EXACTLY (same NumPy ops -> byte-identical here; the contract
is ~1e-13) across formulations / angle / loss / symmetry, that the sweep wrapper
conserves energy and matches the naive per-wavelength loop, and that the JAX
path is rejected (use ``jax.vmap`` instead).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.rcwa import (
    PreparedRCWA2D,
    prepare_rcwa_2d,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_vs_wavelength,
)

_PX = _PY = 0.5e-6
_DEPTH = 0.22e-6
_WLS = (0.5e-6, 0.55e-6, 0.62e-6, 0.7e-6)


def _cell(loss=0.0, S=32):
    xs = (np.arange(S) + 0.5) / S * _PX
    ys = (np.arange(S) + 0.5) / S * _PY
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pil = (X > 0.15 * _PX) & (X < 0.6 * _PX) & (Y > 0.15 * _PY) & (Y < 0.6 * _PY)
    return np.where(pil, 6.25 + loss * 1j, 1.0).astype(complex)


_CFGS = [
    ("laurent normal te", dict(formulation="laurent", theta=0.0, polarization="te"), 0.0),
    ("li oblique tm", dict(formulation="li", theta=0.25, phi=0.3, polarization="tm"), 0.0),
    ("laurent lossy te", dict(formulation="laurent", theta=0.1, polarization="te"), 0.4),
    ("fff_nv normal te", dict(formulation="fff_nv", theta=0.0, polarization="te"), 0.0),
    ("symmetry normal te",
     dict(formulation="laurent", theta=0.0, polarization="te", symmetry=True), 0.0),
]


@pytest.mark.parametrize("name,kw,loss", _CFGS, ids=[c[0] for c in _CFGS])
def test_prepared_solve_matches_single_call(name, kw, loss):
    """prepared.solve(wl) reproduces rcwa_efficiency_2d(...) at each wavelength
    to ~1e-13 (the geometry hoist must not perturb the result)."""
    cell = _cell(loss)
    prep = prepare_rcwa_2d(_PX, _PY, cell, 1.5, 1.0, _DEPTH,
                           n_orders_x=5, n_orders_y=5, **kw)
    assert isinstance(prep, PreparedRCWA2D)
    for w in _WLS:
        o1, R1, T1, *_ = rcwa_efficiency_2d(_PX, _PY, cell, 1.5, 1.0, _DEPTH, w,
                                            n_orders_x=5, n_orders_y=5, **kw)
        res = prep.solve(w)
        assert np.allclose(np.asarray(res[1]), np.asarray(R1), rtol=0, atol=1e-12)
        assert np.allclose(np.asarray(res[2]), np.asarray(T1), rtol=0, atol=1e-12)
        assert np.array_equal(np.asarray(res[0]), np.asarray(o1))


def test_sweep_wrapper_matches_loop_and_conserves():
    """rcwa_efficiency_2d_vs_wavelength == the naive per-wavelength loop, with a
    fixed (wavelength-independent) order set and energy conservation."""
    cell = _cell(0.0)
    orders, R, T = rcwa_efficiency_2d_vs_wavelength(
        _PX, _PY, cell, 1.5, 1.0, _DEPTH, _WLS, n_orders_x=5, n_orders_y=5)
    assert R.shape == (len(_WLS), len(orders))
    assert T.shape == (len(_WLS), len(orders))
    for i, w in enumerate(_WLS):
        o1, R1, T1, *_ = rcwa_efficiency_2d(_PX, _PY, cell, 1.5, 1.0, _DEPTH, w,
                                            n_orders_x=5, n_orders_y=5)
        assert np.allclose(R[i], np.asarray(R1), rtol=0, atol=1e-12)
        assert np.allclose(T[i], np.asarray(T1), rtol=0, atol=1e-12)
        assert abs(float(R[i].sum() + T[i].sum()) - 1.0) < 1e-9   # lossless


def test_sweep_scalar_wavelength_and_validation():
    cell = _cell(0.0)
    # scalar wavelength -> one-row result
    _o, R, T = rcwa_efficiency_2d_vs_wavelength(
        _PX, _PY, cell, 1.5, 1.0, _DEPTH, [0.6e-6], n_orders_x=4, n_orders_y=4)
    assert R.shape[0] == 1 and T.shape[0] == 1
    with pytest.raises(ValueError):
        rcwa_efficiency_2d_vs_wavelength(_PX, _PY, cell, 1.5, 1.0, _DEPTH, [],
                                         n_orders_x=4, n_orders_y=4)
    with pytest.raises(ValueError):
        rcwa_efficiency_2d_vs_wavelength(_PX, _PY, cell, 1.5, 1.0, _DEPTH,
                                         [-0.6e-6], n_orders_x=4, n_orders_y=4)


def test_prepared_rejects_jax():
    """The imperative prepared cache is NumPy/CuPy only -- a JAX sweep must use
    jax.vmap; prepare_rcwa_2d should raise rather than silently mis-trace."""
    jnp = pytest.importorskip("jax.numpy")
    import jax
    jax.config.update("jax_enable_x64", True)
    cell = jnp.asarray(_cell(0.0))
    with pytest.raises(NotImplementedError):
        prepare_rcwa_2d(_PX, _PY, cell, 1.5, 1.0, _DEPTH,
                        n_orders_x=4, n_orders_y=4)
