"""v5.13.0 -- PMMStack.add_tapered_grating: trapezoidal / slanted-sidewall 1-D
grating as a z-staircase of thin VERTICAL PMM layers (lateral-exact, no Fourier
floor in x), the spectral-element counterpart of RCWAStack.add_tapered_grating.

Pins: the vertical limit (equal duties) reproduces a single vertical layer
EXACTLY; a true taper conserves energy and converges monotonically in n_slices;
the new PMMStack energy tripwire warns on the many-interface cascade instability
(it cannot return non-physical gain silently); input validation.
"""
import os

# A BEST-EFFORT request only: OpenBLAS reads these at LIBRARY LOAD time, so the
# setdefault below is inert whenever anything (conftest, another test module)
# has already imported numpy -- which is the normal case under pytest.  Nothing
# here may assume a single-threaded BLAS; ``test_sweep_matches_perwavelength``
# controls its own threading explicitly instead (see its docstring).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm.stack import _blas_limit, _blas_threads_quiet, _warn_stack_energy

_P = 0.8e-6
_WL = 0.55e-6
_TH = 0.3e-6
_ER, _EG = 6.0 + 0j, 1.0 + 0j


def _tapered(duty_b, duty_t, n_slices, degree=12):
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=degree)
    st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                           duty_bottom=duty_b, duty_top=duty_t, n_slices=n_slices)
    return st.set_source(_WL, angle=0.0).solve()


def test_vertical_limit_matches_single_layer():
    """duty_top == duty_bottom is a vertical binary grating -> the staircase of
    identical slices must reproduce a single vertical layer EXACTLY."""
    o_t, R_t, T_t, _j = _tapered(0.5, 0.5, n_slices=5, degree=12)
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(_TH, segments=[(0.25, _EG), (0.5, _ER), (0.25, _EG)])
    o_v, R_v, T_v, _ = st.set_source(_WL, angle=0.0).solve()
    assert np.array_equal(o_t, o_v)
    assert np.max(np.abs(R_t - R_v)) < 1e-11
    assert np.max(np.abs(T_t - T_v)) < 1e-11


def test_taper_converges_and_conserves():
    """A true trapezoid (0.6 -> 0.3) conserves energy and the zeroth-order
    transmission converges monotonically with n_slices (no plateau)."""
    t0s = []
    for ns in (2, 4, 8, 14):
        o, R, T, _ = _tapered(0.6, 0.3, n_slices=ns)
        for pol in (0, 1):                       # both incident polarizations
            assert abs(float(R[pol].sum() + T[pol].sum()) - 1.0) < 1e-4
        t0s.append(float(T[1][o == 0][0]))       # TE (Ey) zeroth order
    d = np.diff(t0s)
    assert np.all(d > 0)                          # monotone increasing
    assert d[-1] < d[0]                           # and converging (shrinking steps)


def test_energy_tripwire_warns_on_gain():
    """The guard warns when a solve returns non-physical R+T > 1 (the cascade
    instability), and stays silent on a conserving result."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # conserving -> no warning
        _warn_stack_energy(np.array([[0.3, 0.1]]), np.array([[0.4, 0.2]]))
    with pytest.warns(UserWarning, match="energy not conserved"):
        _warn_stack_energy(np.array([[0.9, 0.5]]), np.array([[0.8, 0.2]]))


def _sweep_wls():
    return np.linspace(0.5e-6, 0.7e-6, 5)


def _sweep_stack(tapered):
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    if tapered:
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=0.6, duty_top=0.35, n_slices=5)
    else:
        st.add_layer(0.15e-6, segments=[(0.25, _EG), (0.5, _ER), (0.25, _EG)])
        st.add_layer(0.15e-6, eps=2.25)
    return st


def _reference_solves(tapered, wls, cap):
    """Per-wavelength ``solve()`` results run in the SAME BLAS regime that
    ``solve_vs_wavelength(blas_per_worker=cap)`` imposes on itself.

    Deliberately the library's OWN limiter pair rather than threadpoolctl
    directly, so the regime MATCHES BY CONSTRUCTION on every mount: where the
    cap can be applied both sides get it, and where ``threadpoolctl`` is absent
    ``_blas_limit()`` degrades to a null context for both sides alike.
    ``cap=None`` is the environment's untouched threading."""
    with _blas_threads_quiet(cap), _blas_limit():
        return [_sweep_stack(tapered).set_source(float(w), angle=0.0).solve()[:3]
                for w in wls]


def _max_gap(orders, R, T, refs):
    """max |sweep - per-wavelength| over the COMMON propagating orders."""
    gap = 0.0
    for iw, (o1, R1, T1) in enumerate(refs):
        for m in o1:
            if m in orders:
                js = int(np.where(orders == m)[0][0])
                j1 = int(np.where(o1 == m)[0][0])
                gap = max(gap, float(np.max(np.abs(R[iw, :, js] - R1[:, j1]))),
                          float(np.max(np.abs(T[iw, :, js] - T1[:, j1]))))
    return gap


def test_sweep_matches_perwavelength():
    """PMMStack.solve_vs_wavelength (reusing the geometry-only SEM assembly)
    reproduces per-wavelength solve() on the propagating orders, for a tapered
    staircase and a plain vertical stack.

    BIT-EXACT, and asserted in BOTH BLAS threading regimes -- which is the whole
    point of this test's shape (fifth-name referral 2026-08-05,
    ``docs/audits/PMM_FIFTHNAME_TAPERED_SWEEP_2026_08_05.md``).  The sweep caps
    the BLAS pool to ``blas_per_worker`` (default 1) around its dispatch while a
    bare ``solve()`` runs at the environment's pool, so on a many-core box with
    ``threadpoolctl`` installed the two take DIFFERENT LAPACK reduction orders:
    the patterned layers' eigenVECTORS move ~1e-16 relative and the Redheffer
    cascade amplifies that by ~1e8, to ~2e-9 absolute in the efficiencies.  The
    predecessor of this test compared the capped sweep against an UNCAPPED
    solve() at ``atol=1e-10`` and so failed at N BLAS threads and passed at 1 --
    an apples-to-oranges comparison, not a library defect.

    The repair is to compare LIKE WITH LIKE, which then needs no tolerance at
    all: in each regime the two paths agree to the LAST BIT.  Nothing here is a
    numeric bar calibrated on one configuration."""
    wls = _sweep_wls()

    for tapered in (True, False):
        # Regime A -- the SHIPPED DEFAULT (sweep pins BLAS to 1).
        o_cap, R_cap, T_cap = _sweep_stack(tapered).solve_vs_wavelength(
            wls, angle=0.0)
        ref_cap = _reference_solves(tapered, wls, 1)
        # Regime B -- cap disabled, so the sweep inherits the environment's
        # threading exactly as a bare solve() does.  Serial: with no cap, N
        # workers would oversubscribe the BLAS pool.
        o_amb, R_amb, T_amb = _sweep_stack(tapered).solve_vs_wavelength(
            wls, angle=0.0, max_workers=1, blas_per_worker=None)
        ref_amb = _reference_solves(tapered, wls, None)

        assert R_cap.shape == (len(wls), 2, len(o_cap))
        assert np.array_equal(o_amb, o_cap)

        for tag, orders, R, T, refs in (
                ("blas_per_worker=1 (default)", o_cap, R_cap, T_cap, ref_cap),
                ("blas_per_worker=None", o_amb, R_amb, T_amb, ref_amb)):
            gap = _max_gap(orders, R, T, refs)
            assert gap == 0.0, (
                f"tapered={tapered}, {tag}: the assemble-once sweep and "
                f"per-wavelength solve() ran the same arithmetic in the same "
                f"BLAS regime but differ by {gap:.3e} -- that is an "
                f"ALGORITHMIC divergence, not round-off.")

        # And the DEFAULT sweep's residual disagreement with an UNCAPPED
        # solve() is attributable ENTIRELY to the BLAS reduction order: it is
        # no larger than the disagreement solve() has with ITSELF across the
        # two regimes.  Self-calibrating -- zero on a mount where the cap is
        # inert, and it fails the moment the sweep contributes error of its own.
        blas_only = max(
            (max(float(np.max(np.abs(a[1] - b[1]))),
                 float(np.max(np.abs(a[2] - b[2]))))
             for a, b in zip(ref_cap, ref_amb)), default=0.0)
        sweep_gap = _max_gap(o_cap, R_cap, T_cap, ref_amb)
        assert sweep_gap <= blas_only, (
            f"tapered={tapered}: capped sweep vs uncapped solve() differs by "
            f"{sweep_gap:.3e}, MORE than solve()'s own cross-regime spread "
            f"{blas_only:.3e} -- the excess is the sweep's own error.")


def test_sweep_rejects_slanted_and_validates():
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=8)
    st.add_layer(_TH, segments=[(0.5, _ER), (0.5, _EG)], slant_angle=0.1)
    with pytest.raises(NotImplementedError):
        st.solve_vs_wavelength([_WL], angle=0.0)
    st2 = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=8)
    st2.add_layer(_TH, eps=2.25)
    with pytest.raises(ValueError):
        st2.solve_vs_wavelength([], angle=0.0)
    with pytest.raises(ValueError):
        st2.solve_vs_wavelength([-1e-6], angle=0.0)


def test_validation():
    st = PMMStack(_P, n_substrate=1.5, n_superstrate=1.0, degree=8)
    with pytest.raises(ValueError):
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=1.4, n_slices=4)     # duty > 1
    with pytest.raises(ValueError):
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=0.5, n_slices=0)     # n_slices < 1
    with pytest.raises(ValueError):
        st.add_tapered_grating(_TH, eps_ridge=_ER, eps_groove=_EG,
                               duty_bottom=0.5, n_slices=4, rule="bogus")
