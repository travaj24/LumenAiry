"""v5.13.0 -- hybrid 2-D PMM wavelength-sweep reuse (``PreparedPMM2D`` /
``prepare_pmm_2d`` / ``pmm_efficiency_2d_vs_wavelength``).

The hybrid PMM-2D's expensive parts -- the nodal mass inverse + the
``[[1/eps]]^-1`` nodal inversion and the ``O(N^3)`` Fourier-projection
pseudo-inverse -- are wavelength-INDEPENDENT.  The prepared object hoists their
projected forms so a sweep recomputes only the per-wavelength operator rescale
(``GxF = Gx0F/k0 + kx0*IprojF``), the small projected eig, and the S-matrix
cascade.  These tests pin that ``prepared.solve(wl)`` reproduces
``pmm_efficiency_2d(...)`` to ~1e-13 (byte-identical on the uniform-layer path)
across li/laurent / loss / TE-TM / oblique, and that the sweep wrapper conserves
energy and matches the naive per-wavelength loop.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PreparedPMM2D,
    pmm_efficiency_2d,
    pmm_efficiency_2d_vs_wavelength,
    prepare_pmm_2d,
)

_PX = _PY = 0.5e-6
_DEPTH = 0.2e-6
_XB = (0.15 * _PX, 0.6 * _PX)
_YB = (0.15 * _PY, 0.6 * _PY)
_WLS = (0.5e-6, 0.55e-6, 0.62e-6, 0.7e-6)

_CFGS = [
    ("li struct te", dict(eps_pillar=6.25, eps_host=1.0, formulation="li",
                          theta=0.0, polarization="te")),
    ("laurent struct te", dict(eps_pillar=6.25, eps_host=1.0, formulation="laurent",
                               theta=0.0, polarization="te")),
    ("li lossy te", dict(eps_pillar=6.25 + 0.5j, eps_host=1.0, formulation="li",
                         theta=0.0, polarization="te")),
    ("li struct tm", dict(eps_pillar=6.25, eps_host=1.0, formulation="li",
                          theta=0.0, polarization="tm")),
    ("li oblique te", dict(eps_pillar=6.25, eps_host=1.0, formulation="li",
                           theta=0.08, polarization="te")),
    ("uniform layer", dict(eps_pillar=2.25, eps_host=2.25, formulation="li",
                           theta=0.0, polarization="te")),
]


@pytest.mark.parametrize("name,kw", _CFGS, ids=[c[0] for c in _CFGS])
def test_prepared_solve_matches_single_call(name, kw):
    prep = prepare_pmm_2d(_PX, _PY, kw["eps_pillar"], kw["eps_host"], _XB, _YB,
                          1.5, 1.0, _DEPTH, degree=11, n_orders=5,
                          formulation=kw["formulation"], theta=kw["theta"],
                          polarization=kw["polarization"])
    assert isinstance(prep, PreparedPMM2D)
    for w in _WLS:
        o1, R1, T1 = pmm_efficiency_2d(
            _PX, _PY, kw["eps_pillar"], kw["eps_host"], _XB, _YB, 1.5, 1.0,
            _DEPTH, w, degree=11, n_orders=5, formulation=kw["formulation"],
            theta=kw["theta"], polarization=kw["polarization"])[:3]
        res = prep.solve(w)
        assert np.allclose(np.asarray(res[1]), np.asarray(R1), rtol=0, atol=1e-12)
        assert np.allclose(np.asarray(res[2]), np.asarray(T1), rtol=0, atol=1e-12)
        assert np.array_equal(np.asarray(res[0]), np.asarray(o1))


def test_sweep_wrapper_matches_loop_and_conserves():
    orders, R, T = pmm_efficiency_2d_vs_wavelength(
        _PX, _PY, 6.25, 1.0, _XB, _YB, 1.5, 1.0, _DEPTH, _WLS,
        degree=11, n_orders=5)
    assert R.shape == (len(_WLS), len(orders))
    for i, w in enumerate(_WLS):
        o1, R1, T1 = pmm_efficiency_2d(_PX, _PY, 6.25, 1.0, _XB, _YB, 1.5, 1.0,
                                       _DEPTH, w, degree=11, n_orders=5)[:3]
        assert np.allclose(R[i], np.asarray(R1), rtol=0, atol=1e-12)
        assert np.allclose(T[i], np.asarray(T1), rtol=0, atol=1e-12)
        # Energy is set by the solver's Fourier floor (the SWEEP just reuses the
        # same operators -- proven exact above); at this n_orders=5 the hybrid
        # floor is ~1e-2, so this is only a loose physicality sanity, NOT a
        # sweep-correctness check (that is the np.allclose vs the single call).
        assert abs(float(R[i].sum() + T[i].sum()) - 1.0) < 3e-2


def test_sweep_validation():
    with pytest.raises(ValueError):
        pmm_efficiency_2d_vs_wavelength(_PX, _PY, 6.25, 1.0, _XB, _YB, 1.5, 1.0,
                                        _DEPTH, [], degree=11, n_orders=5)
    with pytest.raises(ValueError):
        pmm_efficiency_2d_vs_wavelength(_PX, _PY, 6.25, 1.0, _XB, _YB, 1.5, 1.0,
                                        _DEPTH, [-0.6e-6], degree=11, n_orders=5)
