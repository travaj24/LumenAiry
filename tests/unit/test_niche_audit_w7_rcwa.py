"""Wave-7 niche-audit pins -- ``lumenairy/elements/rcwa/**`` interiors.

Territory: the recorded NON-COVERAGE of the original rcwa audit (ASR
``asr_eta > 0``, ``truncation='circular'``, the slanted/tapered generators,
internal-field reconstruction, dispersive sweeps, the JAX twin) plus a
sibling-class sweep of the W6 findings (BOR-B1 half-cell anchors, EME-W6-1
solver routing, EME-W6-3 discarded Im, Berreman-F-1 flux gauge,
Asymptotic-A11/A13 cache identity, BOR-B2 dimensional tolerances, Berreman
mode sorting).

Every FOUND finding below carries the measured pre-fix number in its
docstring; every CLEAN section is an explicit lock with a MEASURED tolerance
(eigensolve-level agreement is ~1e-11 relative cross-platform, so no exact /
hash pins on solver output).  Solves are slow -- every case here is small.

FOUND + FIXED
  W7-A  oned.py  binary-grating duty QUANTIZATION (BOR-B1 class, and the
        root cause of the ASR non-convergence)
  W7-B  _core.py / stack.py  cache values handed out by identity, writable
        (Asymptotic-A13 / PMM-M9 class)
  W7-C  twod.py  PreparedRCWA2D.solve returned its own ``orders`` array
  W7-D  stack.py ``_cell_grid_index`` floor-vs-nearest half-cell anchor
        (BOR-B1 class)
  W7-E  oned.py  ``jones_retardance_diattenuation`` fast axis on an
        elliptical eigenpolarization (EME-W6-3 class: a dropped Im)
  W7-F  _core.py lossy incidence medium violated R+T=1 with no diagnostic
  W7-G  _core.py ``rcwa_extrapolate`` on a complex sequence
  W7-H  twod.py  ``_nv_curved_wall_fraction`` blind to modulus-degenerate
        materials

REFUTED / VERIFIED CLEAN (locks, all measured)
  absorbing layer + absorbing substrate + metal vs an independent TMM
  (incl. absorptance) ; internal-field reconstruction (interface continuity,
  analytic slab, public-gauge identity) ; dispersive sweep == per-wavelength
  solve, serial == threaded ; even-parity solver routing ; forward/backward
  mode classification at the evanescent / degenerate edges ; circular
  truncation ; JAX twin forward parity + AD-vs-FD + x64 ; unit-scale
  invariance (BOR-B2).
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements import rcwa
from lumenairy.elements.rcwa import _core, oned, twod
from lumenairy.elements.rcwa import stack as _stack

_C = np.complex128


# ===========================================================================
# shared oracles
# ===========================================================================

def _tmm(n_list, d_list, wl, theta0=0.0, pol="s"):
    """From-scratch transfer-matrix (R, T, A) for an isotropic stack.

    PUBLIC convention ``n = n' + i n''`` with ``n'' >= 0`` for loss,
    ``exp(-i w t)``.  Self-validated against closed-form Fresnel in
    :func:`test_clean_tmm_oracle_self_check`."""
    n = np.asarray(n_list, dtype=_C)
    k0 = 2 * np.pi / wl
    kt = np.real(n[0]) * np.sin(theta0)      # RCWA's real-kx0 convention
    kz = np.sqrt(n ** 2 - kt ** 2 + 0j)
    kz = np.where(kz.imag < 0, -kz, kz)

    def rt(j):
        a, b = kz[j], kz[j + 1]
        na, nb = n[j], n[j + 1]
        if pol == "s":
            return (a - b) / (a + b), 2 * a / (a + b)
        den = nb ** 2 * a + na ** 2 * b
        return (nb ** 2 * a - na ** 2 * b) / den, 2 * na * nb * a / den

    M = np.eye(2, dtype=_C)
    for j in range(len(n) - 1):
        r, t = rt(j)
        M = M @ (np.array([[1, r], [r, 1]], dtype=_C) / t)
        if j + 1 < len(n) - 1:
            ph = np.exp(1j * kz[j + 1] * k0 * d_list[j])
            M = M @ np.array([[1 / ph, 0], [0, ph]], dtype=_C)
    r_amp, t_amp = M[1, 0] / M[0, 0], 1.0 / M[0, 0]
    R = abs(r_amp) ** 2
    if pol == "s":
        T = abs(t_amp) ** 2 * (kz[-1].real / kz[0].real)
    else:
        T = abs(t_amp) ** 2 * (np.real(n[-1] * np.conj(kz[-1]) / np.conj(n[-1]))
                               / np.real(n[0] * np.conj(kz[0]) / np.conj(n[0])))
    return R, T, 1.0 - R - T


def _analytic_binary_coeffs(er, eg, d, n_coeffs):
    """Independent (loop-free-formula-free) reference for the exact centred
    Fourier series of a ridge on ``[0, d)``: ``c_k = (er-eg)(1-e^{-2 pi i k d})
    /(2 pi i k)``, ``c_0 = er d + eg (1-d)`` -- derived straight from the
    integral, NOT from the shipped sinc form."""
    k = np.arange(-(n_coeffs - 1), n_coeffs)
    out = np.empty(k.shape, dtype=_C)
    nz = k != 0
    out[~nz] = er * d + eg * (1.0 - d)
    kk = k[nz]
    out[nz] = (er - eg) * (1.0 - np.exp(-2j * np.pi * kk * d)) / (2j * np.pi * kk)
    return out


def _r0(o, R):
    return float(np.asarray(R)[int(np.where(np.asarray(o) == 0)[0][0])])


_B1D = dict(period=1.0e-6, depth=0.4e-6, wavelength=0.633e-6,
            n_substrate=1.5, n_superstrate=1.0)


def _run1d(duty, M, pol="te", eta=0.0, nr=2.0, ng=1.0, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = rcwa.rcwa_efficiency_1d(
            n_ridge=nr, n_groove=ng, duty_cycle=duty, n_orders=M,
            polarization=pol, asr_eta=eta, **_B1D, **kw)
    return _r0(o, R), float(np.asarray(R).sum() + np.asarray(T).sum() - 1.0)


# ===========================================================================
# W7-A -- the 1-D binary profile is realised EXACTLY (BOR-B1 class)
# ===========================================================================

def test_w7a_binary_convolutions_are_the_exact_analytic_series():
    """FOUND+FIXED.  ``_binary_grating_convolutions`` sampled the profile on an
    ``n_samples``-point midpoint grid and FFT'd it.  The DFT of that sampling
    is provably the analytic series of a grating with duty ``J/n_samples``,
    ``J = #{j : (j+0.5)/n_samples < duty}`` -- the requested duty ROUNDED to
    the grid.  Measured pre-fix at ``duty=0.4`` / the 4096 default:
    ``max|c_fft - c_exact| = 2.9e-4`` on a scale of 2.2, and the realised duty
    was 0.399902344 (error -9.766e-05).  Post-fix the coefficients match an
    independently derived integral to machine precision."""
    er, eg = _C(2.0) ** 2, _C(1.0) ** 2
    for duty in (0.5, 0.4, 0.37, 0.123456):
        for M in (3, 11, 40):
            EPS, EPS_II = oned._binary_grating_convolutions(2.0, 1.0, duty, M)
            ref = _analytic_binary_coeffs(er, eg, duty, 2 * M + 1)
            want = _core._toeplitz_1d(ref, M)
            assert np.max(np.abs(EPS - want)) < 1e-14 * max(1.0, abs(er))
            ref_i = _analytic_binary_coeffs(1 / er, 1 / eg, duty, 2 * M + 1)
            want_i = np.linalg.inv(_core._toeplitz_1d(ref_i, M))
            assert np.max(np.abs(EPS_II - want_i)) < 1e-12 * np.max(np.abs(want_i))


def test_w7a_realised_duty_is_the_requested_duty():
    """The DC coefficient IS the realised duty: ``c_0 = eps_g + (eps_r-eps_g) d``.
    Pre-fix ``d`` was ``J/4096``; at ``duty=0.4`` that is 0.399902344, a
    -9.77e-05 geometry error invisible to every convergence and energy check
    (closure stayed at 1e-14).  This pin reads the duty straight back out."""
    er, eg = _C(2.5) ** 2, _C(1.0) ** 2
    for duty in (0.4, 0.37, 0.123456, 0.5):
        EPS, _ = oned._binary_grating_convolutions(2.5, 1.0, duty, 5)
        realised = float(np.real((EPS[5, 5] - eg) / (er - eg)))
        assert abs(realised - duty) < 5e-15, (duty, realised)


def test_w7a_n_samples_is_inert():
    """``n_samples`` is accepted for back-compat and ignored: the exact
    coefficients need no grid, so the old ``4*n_orders`` Nyquist-aliasing
    hazard is structurally gone."""
    a = oned._binary_grating_convolutions(1.5, 1.0, 0.37, 60)
    b = oned._binary_grating_convolutions(1.5, 1.0, 0.37, 60, n_samples=64)
    c = oned._binary_grating_convolutions(1.5, 1.0, 0.37, 60, n_samples=1 << 20)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[0], c[0])
    assert np.array_equal(a[1], b[1]) and np.array_equal(a[1], c[1])


def test_w7a_duty_error_moves_r0_by_the_predicted_amount():
    """Prediction-vs-measurement for the fix's SIZE.  ``dR0/d(duty) = 1.0013``
    around duty 0.4 (measured by a 5-point fit on grid-distinct duties), the
    pre-fix duty error was -9.766e-05, so the pre-fix ``R_0`` had to be low by
    9.78e-05 -- and it was (measured 9.779e-05).  Here we re-derive the slope
    from the FIXED solver and check the old value sits where the model puts
    it."""
    duties = np.array([0.4 - 8 / 4096, 0.4 - 4 / 4096, 0.4,
                       0.4 + 4 / 4096, 0.4 + 8 / 4096])
    vals = np.array([_run1d(d, 60)[0] for d in duties])
    slope = float(np.polyfit(duties, vals, 1)[0])
    assert 0.9 < slope < 1.1, slope
    x = (np.arange(4096) + 0.5) / 4096
    old_duty = float(np.count_nonzero(x < 0.4)) / 4096
    predicted_old_r0 = vals[2] + slope * (old_duty - 0.4)
    assert abs(predicted_old_r0 - 0.09243361823817842) < 2e-6


def test_w7a_duty_half_is_essentially_unmoved():
    """Blast-radius lock: ``duty_cycle=0.5`` is exact on every power-of-two
    grid, so the pre-fix path already had the right GEOMETRY there and only
    the O((k/n_samples)^2) quadrature factor changes.  Measured shift 5.0e-08
    (TE) / 3.0e-09 (TM) -- the value nearly every existing regression pins."""
    assert abs(_run1d(0.5, 11, "te")[0] - 0.105190627893) < 5e-7
    assert abs(_run1d(0.5, 40, "te")[0] - 0.105114779652) < 5e-7
    assert abs(_run1d(0.5, 11, "tm")[0] - 0.019697008489) < 5e-8


@pytest.mark.parametrize("duty", [0.4, 0.5])
def test_w7a_asr_now_converges_to_the_uniform_answer(duty):
    """FOUND+FIXED (the recorded non-coverage: ``asr_eta > 0`` numerics).

    ASR is a COORDINATE CHANGE, so its converged answer must be the uniform
    solver's.  Pre-fix it never got there: the uniform path quantised the duty
    on its 4096-point grid and the ASR path on its own 16384-point grid, so at
    ``duty=0.4`` the two sat 1.05e-04..1.15e-04 apart at EVERY order
    (measured M=9,15,25,40 at eta=0.3: 9.76e-05, 1.05e-04, 1.13e-04,
    1.15e-04 -- flat, i.e. not convergence at all), and the gap was
    insensitive to ``asr_samples``.  The predicted gap from the two grids'
    duties (1.0013 x 1.222e-04 = 1.222e-04) matched.  Post-fix the sequence
    decreases monotonically."""
    ref = _run1d(duty, 100)[0]
    for eta in (0.3, 0.6):
        errs = [abs(_run1d(duty, M, eta=eta)[0] - ref) for M in (15, 25, 40)]
        assert errs[-1] < 3e-6, (duty, eta, errs)
        assert errs[0] > errs[-1], (duty, eta, errs)


def test_w7a_asr_beats_the_uniform_solver_at_equal_order():
    """ASR's whole purpose.  Pre-fix its floor made it WORSE than uniform at
    high order (1.1e-04 vs 3.7e-06 at M=40, duty=0.4)."""
    ref = _run1d(0.4, 100)[0]
    for M in (15, 25, 40):
        uni = abs(_run1d(0.4, M)[0] - ref)
        asr = abs(_run1d(0.4, M, eta=0.3)[0] - ref)
        assert asr < uni, (M, uni, asr)


def test_w7a_asr_energy_closure_is_preserved():
    """The fix must not cost the uniform path its exact closure, nor change
    the ASR path's own.

    Honest bar: ``asr_eta > 0`` does NOT close to machine precision even for a
    provably lossless grating -- the truncated ``u <-> x`` Rayleigh bridge
    ``G`` is not unitary at finite order, so the residual is a truncation
    error that shrinks with ``n_orders``.  Measured (duty 0.4, TE, M =
    9/15/21/31), pre-fix -> post-fix, essentially unmoved:
    eta 0.3  1.71e-06 -> 1.76e-06 ... 8.95e-09 -> 3.57e-09;
    eta 0.9  2.70e-05 -> 2.71e-05 ... 5.17e-08 -> 2.75e-08."""
    for M in (9, 21):
        assert abs(_run1d(0.4, M, eta=0.0)[1]) < 1e-12      # uniform: exact
    for eta, bar9 in ((0.3, 5e-6), (0.6, 2e-5), (0.9, 6e-5)):
        c9 = abs(_run1d(0.4, 9, eta=eta)[1])
        c31 = abs(_run1d(0.4, 31, eta=eta)[1])
        assert c9 < bar9, (eta, c9)
        assert c31 < 0.05 * c9, (eta, c9, c31)              # converges


# ===========================================================================
# W7-B / W7-C -- cache identity + writeability (Asymptotic-A13 / PMM-M9)
# ===========================================================================

def _mk_stack2d(nord=2):
    st = rcwa.RCWAStack(period=1.0e-6, period_y=1.0e-6, n_substrate=1.5,
                        n_orders=nord, n_orders_y=nord)
    cell = np.ones((16, 16), dtype=_C)
    cell[4:12, 4:12] = 4.0
    st.add_layer(0.3e-6, eps_cell=cell)
    st.set_source(0.633e-6, theta=0.0)
    return st


def test_w7b_homogeneous_cache_values_are_frozen():
    """FOUND+FIXED.  ``_cached_homogeneous_eigenmodes`` returned its stored
    ``(W, V, kz)`` BY IDENTITY and writable, and ``kz`` escapes to the public
    API through ``RCWAResult.per_order_amplitudes()['kz']``.  Measured
    pre-fix: ``d['kz'] is cache[key][2]`` was True, an in-place ``*= 2``
    changed the cached array by 1.4851 and the NEXT solve returned the doubled
    ``kz`` (max delta 1.4851 vs pristine)."""
    _core._clear_rcwa_caches()
    _mk_stack2d().solve()
    assert _core._HOMOG_CACHE, "probe did not populate the cache"
    for key, val in _core._HOMOG_CACHE.items():
        for a in val:
            assert isinstance(a, np.ndarray) and not a.flags.writeable, key


def test_w7b_public_kz_is_a_writable_copy_and_cannot_poison_the_cache():
    """The public contract (a writable ndarray) is preserved by copying, while
    the cache itself stays frozen."""
    _core._clear_rcwa_caches()
    r1 = _mk_stack2d().solve()
    d = r1.per_order_amplitudes("reflection")
    kz = d["kz"]
    assert kz.flags.writeable
    pristine = np.array(kz, copy=True)
    kz *= 2.0                                   # would have poisoned the cache
    kz2 = _mk_stack2d().solve().per_order_amplitudes("reflection")["kz"]
    assert np.array_equal(kz2, pristine)


def test_w7b_cache_cold_equals_warm_bit_for_bit():
    """Determinism lock (unchanged by the freeze)."""
    _core._clear_rcwa_caches()
    cold = np.asarray(_mk_stack2d().solve().efficiencies()[1])
    warm = np.asarray(_mk_stack2d().solve().efficiencies()[1])
    _core._clear_rcwa_caches()
    recold = np.asarray(_mk_stack2d().solve().efficiencies()[1])
    assert np.array_equal(cold, warm) and np.array_equal(cold, recold)


def test_w7b_repeated_layer_modes_are_shared_but_frozen():
    """FOUND+FIXED (latent).  The per-solve eig dedup appends the SAME tuple
    for every layer with an identical key, so with ``retain_internal=True`` a
    3-period stack had ``info['W'][0] is info['W'][2]`` (measured True, as for
    ``EPS``).  A write through one layer's view silently rewrote the others;
    now it raises."""
    st = rcwa.RCWAStack(period=1.0e-6, n_substrate=1.5, n_orders=4)
    c1 = np.ones((32, 1), dtype=_C)
    c1[:16] = 4.0
    for _ in range(3):
        st.add_layer(0.2e-6, eps_cell=c1)
        st.add_layer(0.1e-6, eps=2.25)
    st.set_source(0.633e-6, theta=0.0)
    info = st.solve(retain_internal=True, symmetry=False)._modal["internal"]
    assert info["W"][0] is info["W"][2]          # dedup still saves the eig
    for name in ("W", "V", "lam", "EPS"):
        assert not np.asarray(info[name][0]).flags.writeable, name
    with pytest.raises(ValueError):
        info["W"][0][0, 0] = 0.0


def test_w7c_prepared_2d_hands_out_a_private_orders_array():
    """FOUND+FIXED.  ``PreparedRCWA2D.solve`` returned ``self.orders`` by
    identity, so every ``Efficiency2D`` of a sweep shared ONE array with the
    prepared object.  Measured pre-fix: ``a[0] is prep.orders`` True,
    ``b[0] is a[0]`` True; ``a[0][:] = 0`` left ``prep.orders`` all-zero and a
    LATER ``solve()`` reported zeroed orders."""
    cell = np.ones((16, 16), dtype=_C)
    cell[4:12, 4:12] = 4.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep = rcwa.prepare_rcwa_2d(
            period_x=1e-6, period_y=1e-6, eps_cell=cell, n_substrate=1.5,
            n_superstrate=1.0, depth=0.3e-6, n_orders_x=2, n_orders_y=2)
        a = prep.solve(0.60e-6)
        b = prep.solve(0.65e-6)
        assert a[0] is not prep.orders and b[0] is not a[0]
        want = np.array(prep.orders, copy=True)
        a[0][:] = 0
        assert np.array_equal(prep.orders, want)
        assert np.array_equal(b[0], want)
        assert np.array_equal(prep.solve(0.70e-6)[0], want)


# ===========================================================================
# W7-D -- half-cell anchor in the field <-> cell registration (BOR-B1 class)
# ===========================================================================

def test_w7d_cell_grid_index_is_nearest_not_floor():
    """FOUND+FIXED.  ``_cell_grid_index``'s docstring says "Nearest-cell-pixel"
    but it used ``floor``.  ``_eps_convolution_2d``'s ``fft2/(Sx Sy)`` places
    ``eps_cell[j, i]`` at the NODE ``j Px/Sx``, so ``round`` is the
    co-registered choice; ``floor`` is a systematic ``P/(2 Sx)`` bias.
    Measured pre-fix at ``Sx=4``, one period of samples:
    ``floor -> [2 2 3 3 0 0 1 1]`` vs ``nearest -> [2 2 3 0 0 0 1 2]``."""
    px = 0.6e-6
    xg = (np.arange(8) - 4) * (px / 8)
    ix, _iy = _stack.RCWAResult._cell_grid_index(xg, np.zeros(1), px, px, 4, 1)
    want = np.round((xg % px) / px * 4).astype(int) % 4
    assert np.array_equal(ix, want)
    # exact-node samples must land on their own pixel under either rule
    xn = np.arange(4) * (px / 4)
    ixn, _ = _stack.RCWAResult._cell_grid_index(xn, np.zeros(1), px, px, 4, 1)
    assert np.array_equal(ixn, np.arange(4))


def test_w7d_layer_absorption_split_uses_the_registered_map():
    """The bias did NOT vanish with the field grid (it is fixed by the CELL
    resolution): measured on a 64-pixel half-metal cell over a lossy spacer,
    the metal layer's converged share was 0.16548922 (floor) vs 0.16541527
    (nearest) -- 7.4e-05 absolute, still 7.4e-05 at ``nx=4096``.  The total is
    normalised, so only the SPLIT moved and energy could not see it."""
    st = rcwa.RCWAStack(period=0.6e-6, n_substrate=1.5, n_orders=6)
    cw = np.ones((64, 1), dtype=_C)
    cw[:32] = complex(0.15, 3.5) ** 2
    st.add_layer(0.05e-6, eps_cell=cw)
    st.add_layer(0.10e-6, eps=complex(1.5, 0.05) ** 2)
    st.set_source(0.633e-6, theta=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve(retain_internal=True, symmetry=False)
    a = res.layer_absorption(nx=1024, nz_per_layer=16)
    assert abs(float(a[0].sum()) - float(res.absorptance()[0])) < 1e-12
    assert abs(float(a[0][0]) - 0.16541074) < 3e-5
    assert not abs(float(a[0][0]) - 0.16548682) < 1e-6     # the floor answer


# ===========================================================================
# W7-E -- fast axis of an elliptical eigenpolarization (a dropped Im)
# ===========================================================================

def test_w7e_fast_axis_is_the_polarization_ellipse_azimuth():
    """FOUND+FIXED.  ``fast_axis`` was ``arctan2(Re v_y, Re v_x)``, valid only
    for a LINEAR eigenpolarization.  Measured on a Hermitian ``J`` whose
    max-transmittance eigenvector is ``(0.6, 0.8i)`` -- major axis clearly y,
    ``|E_y| = 0.8 > |E_x| = 0.6`` -- the old form returned 3.141593 rad
    (the x axis): a 90-degree error, and outside any half-open pi range."""
    sla = pytest.importorskip("scipy.linalg")
    v0 = np.array([0.6, 0.8j], dtype=_C)
    v1 = np.array([0.8, -0.6j], dtype=_C)
    v1 = v1 / np.linalg.norm(v1)
    J = sla.sqrtm(1.0 * np.outer(v0, np.conj(v0))
                  + 0.09 * np.outer(v1, np.conj(v1)))
    _ret, _dia, fa = rcwa.jones_retardance_diattenuation(J)
    assert abs(fa - np.pi / 2) < 1e-9
    assert -np.pi / 2 - 1e-12 < fa <= np.pi / 2 + 1e-12


@pytest.mark.parametrize("ang", [0.0, 0.3, -0.7, 1.2])
def test_w7e_fast_axis_reduces_to_the_linear_answer(ang):
    """Back-compatibility: for a LINEAR eigenpolarization the ellipse azimuth
    IS the old arctan2 value modulo pi (the case every existing pin uses)."""
    c, s = np.cos(ang), np.sin(ang)
    v0 = np.array([c, s], dtype=_C)
    v1 = np.array([-s, c], dtype=_C)
    J = (1.0 * np.outer(v0, np.conj(v0)) + 0.25 * np.outer(v1, np.conj(v1)))
    _r, _d, fa = rcwa.jones_retardance_diattenuation(J)
    assert abs(np.tan(fa) - np.tan(ang)) < 1e-9


def test_w7e_diagonal_jones_fast_axis_is_an_axis():
    """A TE/TM-aligned grating (diagonal J) must report 0 or +-pi/2."""
    for a, b in ((0.9, 0.4), (0.4, 0.9)):
        _r, _d, fa = rcwa.jones_retardance_diattenuation(
            np.diag([a, b]).astype(_C))
        assert min(abs(fa), abs(abs(fa) - np.pi / 2)) < 1e-12


# ===========================================================================
# W7-F -- a lossy incidence medium is announced, not silent
# ===========================================================================

def test_w7f_lossy_superstrate_warns_with_the_measured_size():
    """FOUND+FIXED.  A lossy incidence medium is out of ``_forward_flux_kz``'s
    scope by construction (the incident/reflected cross-term carries net flux)
    and ``_check_energy``'s lossless clause is disarmed for it, so the
    violation was completely silent.  Measured on a lossless 200 nm n=2 slab
    over n_sub=1.5: ``Im(n_sup)=0.01 -> R+T = 1.001896`` (normal) /
    ``1.002479`` (theta=0.4); ``Im(n_sup)=0.1 -> 1.022957 / 1.030396``."""
    def solve(nsup, th):
        st = rcwa.RCWAStack(period=0.3e-6, n_superstrate=nsup,
                            n_substrate=1.5, n_orders=2)
        st.add_layer(0.2e-6, eps=4.0)
        st.set_source(0.633e-6, theta=th)
        return st.solve(symmetry=False)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = solve(complex(1.5, 0.1), 0.0)
    assert any("LOSSY incidence medium" in str(x.message) for x in w)
    o, R, T = res.efficiencies()
    tot = float(np.asarray(R)[1].sum() + np.asarray(T)[1].sum())
    assert abs(tot - 1.022957) < 1e-4          # the documented, now-announced size

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solve(1.0, 0.4)
    assert not [x for x in w if "LOSSY incidence" in str(x.message)]


def test_w7f_shared_guard_stays_silent_for_non_rcwa_callers():
    """The guard is shared with PMM / Berreman, whose own audits pinned
    silent-on-loss (test_niche_audit_m2_m3_m9 M3).  The diagnostic is opt-in."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _core._require_propagating_incidence("probe", _C(1.0 - 0.02j), 0.0)
    assert not w
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _core._require_propagating_incidence("probe", _C(1.0 - 0.02j), 0.0,
                                             warn_lossy=True)
    assert len(w) == 1 and "LOSSY incidence medium" in str(w[0].message)


# ===========================================================================
# W7-G / W7-H -- small named-failure fixes
# ===========================================================================

def test_w7g_extrapolate_rejects_a_complex_sequence_by_name():
    """FOUND+FIXED.  ``np.asarray(values, dtype=float)`` on a complex sequence
    died with a bare ``TypeError: float() argument must be a string or a real
    number, not 'complex'`` -- no mention of the function or of what to do."""
    seq = [0.30 + 0.40j, 0.34 + 0.30j, 0.355 + 0.26j]
    with pytest.raises(ValueError, match="must be REAL"):
        rcwa.rcwa_extrapolate(seq, n_orders=[4, 6, 8])
    with pytest.raises(ValueError, match="must be REAL"):
        rcwa.rcwa_extrapolate(np.asarray(seq), method="shanks")
    # the real path is untouched
    got = rcwa.rcwa_extrapolate([0.30, 0.34, 0.355], n_orders=[4, 6, 8])
    assert np.isfinite(got)


def test_w7h_curved_wall_fraction_sees_modulus_degenerate_materials():
    """FOUND+FIXED.  The fff_nv CURVATURE discriminator built its indicator
    from ``|eps|``: a disk of ``2.5 e^{i}`` in a background of ``2.5`` read
    ``frac = 0.0000`` against ``0.1852`` for the identical geometry with
    distinct moduli.  (No silent wrong answer was reachable -- such a cell is
    metallic and cornered, so the guard's OTHER arm still raised -- but the
    curvature verdict and its message were wrong.)  The documented
    discriminator values are unchanged."""
    S = 48
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    disk = ((xx - .5) ** 2 + (yy - .5) ** 2) < .25 ** 2
    plain = np.where(disk, _C(6.0), _C(1.0))
    degen = np.where(disk, _C(2.5) * np.exp(1j), _C(2.5))
    f_plain = twod._nv_curved_wall_fraction(plain)
    assert abs(twod._nv_curved_wall_fraction(degen) - f_plain) < 1e-12
    # documented values (docstring: ~0.0-0.03 axis-aligned, ~0.17+ curved)
    assert twod._nv_curved_wall_fraction(
        np.where(xx < .5, _C(6.0), _C(1.0))) == 0.0
    square = np.where((np.abs(xx - .5) < .25) & (np.abs(yy - .5) < .25),
                      _C(6.0), _C(1.0))
    assert twod._nv_curved_wall_fraction(square) < twod._NV_CURVED_FRAC_MAX
    assert f_plain > 0.17


# ===========================================================================
# VERIFIED-CLEAN LOCKS
# ===========================================================================

def test_clean_tmm_oracle_self_check():
    """The oracle used below, validated against closed-form Fresnel."""
    for th in (0.0, 0.35, 0.8):
        ct0 = np.cos(th)
        ct1 = np.sqrt(1 - (np.sin(th) / 1.5) ** 2)
        rs = (ct0 - 1.5 * ct1) / (ct0 + 1.5 * ct1)
        rp = (1.5 * ct0 - ct1) / (1.5 * ct0 + ct1)
        Rs, Ts, _ = _tmm([1.0, 1.5], [], 633e-9, th, "s")
        Rp, Tp, _ = _tmm([1.0, 1.5], [], 633e-9, th, "p")
        assert abs(Rs - rs ** 2) < 1e-15 and abs(Rp - rp ** 2) < 1e-15
        assert abs(Rs + Ts - 1) < 1e-15 and abs(Rp + Tp - 1) < 1e-15


@pytest.mark.parametrize("theta", [0.0, 0.35, 0.9])
@pytest.mark.parametrize("name,lays,nsub", [
    ("lossless slab", [(1.5, 0.2e-6)], 1.5),
    ("absorbing layer", [(complex(1.5, 0.3), 0.2e-6)], 1.5),
    ("metal layer", [(complex(0.15, 3.5), 0.03e-6)], 1.5),
    ("absorbing substrate", [(1.5, 0.2e-6)], complex(1.5, 0.3)),
    ("both absorbing", [(complex(2.0, 0.1), 0.25e-6)], complex(1.5, 0.5)),
])
def test_clean_absorbing_media_match_an_independent_tmm(name, lays, nsub, theta):
    """EME-W6-3 sibling REFUTED for the layer / substrate physics: no Im is
    dropped anywhere on the absorbing path.  Worst |RCWA - TMM| over the full
    18-config matrix (R, T AND absorptance, both polarizations, theta up to
    0.9 rad) measured 9.99e-16."""
    st = rcwa.RCWAStack(period=0.3e-6, n_superstrate=1.0, n_substrate=nsub,
                        n_orders=2)
    for nn, dd in lays:
        st.add_layer(dd, eps=complex(nn) ** 2)
    st.set_source(0.633e-6, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve(symmetry=False)
    o, R, T = res.efficiencies()
    i0 = int(np.where(np.asarray(o) == 0)[0][0])
    R, T, A = np.asarray(R)[:, i0], np.asarray(T)[:, i0], np.asarray(
        res.absorptance())
    nl = [1.0] + [x[0] for x in lays] + [nsub]
    dl = [x[1] for x in lays]
    Rs, Ts, As = _tmm(nl, dl, 0.633e-6, theta, "s")
    Rp, Tp, Ap = _tmm(nl, dl, 0.633e-6, theta, "p")
    # RCWAStack rows: 0 = incident E_x (p at phi=0), 1 = incident E_y (s)
    worst = max(abs(R[1] - Rs), abs(T[1] - Ts), abs(A[1] - As),
                abs(R[0] - Rp), abs(T[0] - Tp), abs(A[0] - Ap))
    assert worst < 1e-13, (name, theta, worst)


def test_clean_internal_field_reconstruction():
    """Recorded non-coverage, now locked (also the Berreman-F-1 gauge check at
    an INTERNAL boundary).  Measured: tangential E/H continuous across an
    internal interface to 4.9e-07 (truncation-limited), ``eps Ez`` continuous
    to 0.0, a uniform slab matches the analytic 3-medium field to 2.8e-17,
    and the surface value equals the PUBLIC ``1 + r`` exactly."""
    st = rcwa.RCWAStack(period=0.4e-6, n_superstrate=1.0, n_substrate=1.5,
                        n_orders=3)
    e1, e2 = complex(2.0) ** 2, complex(1.6, 0.2) ** 2
    st.add_layer(0.20e-6, eps=e1)
    st.add_layer(0.15e-6, eps=e2)
    st.set_source(0.633e-6, theta=0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve(retain_internal=True, symmetry=False)
    f = res.internal_field([0.19999999e-6, 0.20000001e-6], nx=8,
                           incident=(0.0, 1.0))
    for c in ("Ex", "Ey", "Hx", "Hy"):
        assert np.max(np.abs(f[c][0] - f[c][1])) < 5e-6, c
    assert np.max(np.abs(e1 * f["Ez"][0] - e2 * f["Ez"][1])) < 5e-6

    st2 = rcwa.RCWAStack(period=0.4e-6, n_superstrate=1.0, n_substrate=1.5,
                         n_orders=2)
    st2.add_layer(0.3e-6, eps=4.0)
    st2.set_source(0.633e-6, theta=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = st2.solve(retain_internal=True, symmetry=False)
    zz = np.linspace(0, 0.3e-6, 7)
    g = r2.internal_field(zz, nx=4, incident=(0.0, 1.0))
    k0 = 2 * np.pi / 0.633e-6
    r01, t01, r1s = (1 - 2) / 3, 2 / 3, (2 - 1.5) / 3.5
    ph = np.exp(1j * 2 * k0 * 0.3e-6)
    A = t01 / (1 + r01 * r1s * ph ** 2)
    Ean = (A * np.exp(1j * 2 * k0 * zz)
           + A * r1s * ph ** 2 * np.exp(-1j * 2 * k0 * zz))
    En = g["Ey"][:, 0]
    assert abs(abs(En[0] / Ean[0]) - 1.0) < 1e-12
    assert np.max(np.abs(En - (En[0] / Ean[0]) * Ean)) < 1e-14
    # PUBLIC gauge at the top surface: E_y(0) == 1 + r_yy
    assert abs(En[0] - (1.0 + r2.jones_reflection()[1, 1])) < 1e-14


def test_clean_dispersive_sweep_equals_per_wavelength_solve():
    """Recorded non-coverage (``solve_vs_wavelength`` numerics), now locked:
    serial == threaded == per-wavelength ``solve()``, BIT-for-bit."""
    st = rcwa.RCWAStack(period=0.5e-6, n_substrate=1.5, n_orders=3)
    c1 = np.ones((32, 1), dtype=_C)
    c1[:16] = 4.0
    st.add_layer(0.2e-6, eps_cell=c1)
    st.set_source(0.6e-6, theta=0.2)
    wls = np.array([0.55e-6, 0.60e-6, 0.65e-6, 0.70e-6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_s, R_s, T_s, J_s = st.solve_vs_wavelength(wls, max_workers=1)
        _o, R_p, T_p, J_p = st.solve_vs_wavelength(wls, max_workers=4)
        per = []
        for w in wls:
            st.set_source(w, theta=0.2)
            per.append(np.asarray(st.solve().efficiencies()[1]))
    assert np.array_equal(R_s, R_p) and np.array_equal(T_s, T_p)
    assert np.array_equal(J_s, J_p)
    assert np.array_equal(R_s, np.stack(per))


@pytest.mark.parametrize("tag", ["disk", "metal disk", "lossy cross",
                                 "off-centre", "wedge"])
def test_clean_even_parity_routing_matches_the_full_solve(tag):
    """EME-W6-1 sibling REFUTED.  ``symmetry='auto'`` (the DEFAULT) routes a
    normal-incidence centro-symmetric stack through an ``(N+1)``-d even-sector
    cascade.  Measured worst deviation from the full ``2N`` solve over five
    cells (incl. a metal disk and a lossy cross): 9.2e-12; the two
    non-symmetric cells fall back BIT-IDENTICALLY (0.0)."""
    S = 48
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    cells = {
        "disk": np.where(((xx - .5) ** 2 + (yy - .5) ** 2) < .2 ** 2,
                         _C(6.0), _C(1.0)),
        "metal disk": np.where(((xx - .5) ** 2 + (yy - .5) ** 2) < .2 ** 2,
                               _C(0.15 + 3.5j) ** 2, _C(1.0)),
        "lossy cross": np.where((np.abs(xx - .5) < .12)
                                | (np.abs(yy - .5) < .12),
                                _C(2.5 + 0.4j), _C(1.0)),
        "off-centre": np.where(((xx - .3) ** 2 + (yy - .62) ** 2) < .18 ** 2,
                               _C(6.0), _C(1.0)),
        "wedge": np.where(xx + 0.6 * yy < 0.7, _C(5.0), _C(1.0)),
    }
    outs = []
    for sym in (True, False):
        st = rcwa.RCWAStack(period=0.9e-6, period_y=0.9e-6, n_substrate=1.5,
                            n_orders=3, n_orders_y=3)
        st.add_layer(0.25e-6, eps_cell=cells[tag])
        st.set_source(0.633e-6, theta=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = st.solve(symmetry=sym)
        o, R, T = r.efficiencies()
        outs.append((np.asarray(R), np.asarray(T),
                     np.asarray(r.jones_reflection())))
    for a, b in zip(outs[0], outs[1]):
        assert np.max(np.abs(a - b)) < 5e-11, tag
    if tag in ("off-centre", "wedge"):
        for a, b in zip(outs[0], outs[1]):
            assert np.array_equal(a, b), tag


def test_clean_forward_mode_classification_at_the_edges():
    """Berreman mode-sorting sibling REFUTED: the CLASSIFICATION itself (not
    just twin-consistency) is right at the flux-null / exactly-degenerate
    edges -- exactly 2N forward modes and no GROWING propagator in the forward
    set."""
    rng = np.random.default_rng(7)
    N = 6
    V = (rng.standard_normal((4 * N, 4 * N))
         + 1j * rng.standard_normal((4 * N, 4 * N))) * 1e-14
    for gam in (np.concatenate([np.linspace(.2, 5, 2 * N),
                                -np.linspace(.2, 5, 2 * N)]).astype(_C),
                np.zeros(4 * N, dtype=_C)):
        idx = _core._select_forward_flux(gam, V, N)
        assert len(idx) == 2 * N
        assert float(np.min(np.real(gam)[idx])) >= -1e-12


def test_clean_lossy_out_of_plane_cascade_is_bounded_and_accounted():
    """The same classification inside a real generalized cascade: a lossy
    OUT-OF-PLANE tensor pillar at conical incidence stays finite and closes
    R + T + A = 1 (measured 1.0 +- 2e-9 at theta = 0, 0.4 and 1.2 rad)."""
    Sx = 24
    xx, yy = np.meshgrid((np.arange(Sx) + .5) / Sx, (np.arange(Sx) + .5) / Sx,
                         indexing="ij")
    base = np.diag([_C(4.0 + 0.2j), _C(3.0 + 0.1j), _C(3.5 + 0.15j)])
    base[0, 2] = base[2, 0] = 0.8
    base[1, 2] = base[2, 1] = -0.4
    cell = np.tile(np.eye(3, dtype=_C) * 2.0, (Sx, Sx, 1, 1))
    cell[((xx - .5) ** 2 + (yy - .5) ** 2) < .25 ** 2] = base
    for th, ph in ((0.0, 0.0), (0.4, 0.7), (1.2, 0.3)):
        st = rcwa.RCWAStack(period=0.7e-6, period_y=0.7e-6, n_substrate=1.5,
                            n_orders=2, n_orders_y=2)
        st.add_layer(0.3e-6, eps_tensor_cell=cell)
        st.set_source(0.633e-6, theta=th, phi=ph)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = st.solve(symmetry=False).efficiencies()
        R, T = np.asarray(R), np.asarray(T)
        assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
        assert np.all(R >= -1e-12) and np.all(T >= -1e-12)
        assert np.all(R.sum(1) + T.sum(1) <= 1.0 + 1e-9)


def test_clean_circular_truncation_agrees_with_rectangular():
    """Recorded non-coverage (``truncation='circular'``), now locked: the two
    order sets converge to the SAME answer (measured |dR0| = 2.9e-05 at M=5
    and 3.1e-05 at M=6, while the rectangular sequence itself still drifts
    3e-05 between those orders) at 1/3 fewer harmonics, with clean closure.

    CLOSURE BOUND SPLIT PER ARM 2026-08-01 (release verification for
    v5.32.0).  The single ``abs(defect) < 1e-11`` gate applied to BOTH arms
    was knife-edge on the RECTANGULAR control -- the 121-order (242x242)
    ``zgeev`` + S-matrix cascade, whose closure residual is LAPACK round-off
    and moves with the BLAS reduction order.  Measured on this box, the
    identical computation over BLAS thread counts 1 / 2 / 3 / 4 / 6 / 8 /
    12 / 16 / 24::

        threads |  1        2        3        4        6
        rect    | +1.13e-11 -5.34e-13 -3.35e-14 +3.55e-14 +3.55e-14
        circ    | -3.15e-14 -2.42e-14 -1.04e-14 -1.31e-14 -2.78e-15
        threads |  8        12       16       24
        rect    | -1.38e-12 -8.99e-15 -7.70e-14 +1.53e-14
        circ    | +2.09e-14 +3.55e-14 -7.35e-14 +1.84e-14

    i.e. the rectangular arm reaches 1.13e-11 -- ABOVE the old gate -- on
    the single-threaded path, which is why this pin passed or failed as a
    function of pytest collection order (whichever earlier test last moved
    the effective threading).  That is consistent with this file's own
    header: "eigensolve-level agreement is ~1e-11 relative cross-platform,
    so no exact / hash pins on solver output".

    Nothing is loosened where this test does its work.  The CIRCULAR arm --
    the recorded non-coverage being locked -- keeps the original 1e-11 gate
    and clears it by 136x at every thread count; only the rectangular
    CONTROL arm moves to 1e-10 (8.8x headroom over its worst measured
    value).  The physics claim is untouched and is not round-off-sensitive
    at all: |dR0| reads 2.9247e-05 to five significant figures at every one
    of the nine thread counts above.
    """
    S = 48
    xx, yy = np.meshgrid((np.arange(S) + .5) / S, (np.arange(S) + .5) / S,
                         indexing="ij")
    cell = np.where(((xx - .5) ** 2 + (yy - .5) ** 2) < .22 ** 2,
                    _C(6.0), _C(1.0))
    out = {}
    for trunc in ("rectangular", "circular"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, *_ = rcwa.rcwa_efficiency_2d(
                period_x=0.9e-6, period_y=0.9e-6, eps_cell=cell,
                n_substrate=1.5, n_superstrate=1.0, depth=0.25e-6,
                wavelength=0.633e-6, n_orders_x=5, n_orders_y=5,
                truncation=trunc)
        i0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
        out[trunc] = (len(o), float(R[i0]),
                      float(R.sum() + T.sum() - 1.0))
    assert out["circular"][0] < out["rectangular"][0]
    assert abs(out["circular"][1] - out["rectangular"][1]) < 2e-4
    # Per-arm closure (see the docstring for the measured thread-count table).
    assert abs(out["circular"][2]) < 1e-11, out["circular"]
    assert abs(out["rectangular"][2]) < 1e-10, out["rectangular"]


@pytest.mark.parametrize("scale", [1e9, 1e-3])
def test_clean_unit_scale_invariance(scale):
    """BOR-B2 sibling REFUTED with measurement: every length-valued threshold
    in the territory is relative or a ratio.  The SAME problem expressed in
    nanometres / millimetres instead of metres reproduces to <= 7.5e-16
    through the 1-D, 2-D, stack and layer_absorption entry points."""
    def run(s):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, _T = rcwa.rcwa_efficiency_1d(
                period=0.9e-6 * s, n_ridge=2.5, n_groove=1.0, n_substrate=1.5,
                n_superstrate=1.0, depth=0.22e-6 * s, duty_cycle=0.4,
                wavelength=0.633e-6 * s, angle=0.17, n_orders=8,
                polarization="tm")
            cell = np.ones((32, 32), dtype=_C)
            cell[8:24, 8:24] = 6.0
            _o2, R2, _T2, *_ = rcwa.rcwa_efficiency_2d(
                period_x=0.9e-6 * s, period_y=0.9e-6 * s, eps_cell=cell,
                n_substrate=1.5, n_superstrate=1.0, depth=0.25e-6 * s,
                wavelength=0.633e-6 * s, n_orders_x=3, n_orders_y=3, theta=0.2)
            st = rcwa.RCWAStack(period=0.9e-6 * s, n_substrate=1.5, n_orders=6)
            c1 = np.ones((64, 1), dtype=_C)
            c1[:26] = 6.25
            st.add_layer(0.22e-6 * s, eps_cell=c1)
            st.add_layer(0.10e-6 * s, eps=complex(1.5, 0.05) ** 2)
            st.set_source(0.633e-6 * s, theta=0.17)
            res = st.solve(retain_internal=True, symmetry=False)
        return (np.asarray(R), np.asarray(R2),
                np.asarray(res.efficiencies()[1]),
                res.layer_absorption(nx=64, nz_per_layer=8))

    # 1e-12, not 5e-15: the eigensolve's unit-scale residue drifts to
    # 1.27e-14 on CI Linux at scale=1e9 (measured, 8a07baf run) vs 7.5e-16
    # on the authoring box.  1e-12 still discriminates the B2 defect class
    # by ~10 orders (PMM's unit-dependent floor measured 2.7e-2 absolute).
    for a, b in zip(run(1.0), run(scale)):
        assert np.max(np.abs(a - b)) < 1e-12


# ===========================================================================
# JAX twin (recorded non-coverage: parity matrix, gradients, x64)
# ===========================================================================

jax = pytest.importorskip("jax")


@pytest.fixture(scope="module")
def _x64():
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("form", ["laurent", "li"])
def test_clean_jax_twin_forward_parity(_x64, pol, form):
    """Recorded non-coverage.  Measured worst |JAX - NumPy| over a 48-config
    matrix (2 pol x 3 formulations x 4 material sets x 2 incidences):
    8.34e-15; the 2-D entry 3.3e-14 and RCWAStack 8.2e-16."""
    import jax.numpy as jnp
    base = dict(period=0.8e-6, depth=0.35e-6, wavelength=0.633e-6,
                duty_cycle=0.5)
    for nr, ng, nsub in ((2.0, 1.0, 1.5),
                         (complex(0.15, 3.5), 1.0, 1.5),
                         (2.0, 1.0, complex(1.5, 0.3))):
        for ang in (0.0, 0.35):
            kw = dict(polarization=pol, angle=ang, n_orders=6,
                      formulation=form, n_superstrate=1.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _o, Rn, Tn = rcwa.rcwa_efficiency_1d(
                    n_ridge=nr, n_groove=ng, n_substrate=nsub, **kw, **base)
                _oj, Rj, Tj = rcwa.rcwa_efficiency_1d(
                    n_ridge=jnp.asarray(nr), n_groove=jnp.asarray(ng),
                    n_substrate=jnp.asarray(nsub), **kw, **base)
            assert np.max(np.abs(np.asarray(Rj) - Rn)) < 5e-14
            assert np.max(np.abs(np.asarray(Tj) - Tn)) < 5e-14


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_clean_jax_gradients_match_finite_differences(_x64, pol):
    """AD vs central FD on the three documented differentiable arguments.
    Measured relative agreement 1.2e-09 (n_ridge, angle) to 5.1e-06 (depth,
    TM -- FD truncation on a derivative of 2.3e+05)."""
    import jax.numpy as jnp

    def make(which):
        def f(v):
            kw = dict(period=0.8e-6, n_ridge=jnp.asarray(2.0),
                      n_groove=jnp.asarray(1.0), n_substrate=jnp.asarray(1.5),
                      n_superstrate=jnp.asarray(1.0),
                      depth=jnp.asarray(0.35e-6), duty_cycle=0.5,
                      wavelength=jnp.asarray(0.633e-6), polarization=pol,
                      n_orders=5)
            kw[which] = v
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, _T = rcwa.rcwa_efficiency_1d(**kw)
            return R[o.shape[0] // 2]
        return f

    for which, x0, h, tol in (("depth", 0.35e-6, 1e-10, 1e-5),
                              ("n_ridge", 2.0, 1e-6, 1e-7),
                              ("angle", 0.3, 1e-6, 1e-6)):
        f = make(which)
        g = float(jax.grad(f)(jnp.asarray(x0)))
        fd = float((f(jnp.asarray(x0 + h)) - f(jnp.asarray(x0 - h))) / (2 * h))
        assert abs(g - fd) <= tol * max(abs(fd), 1e-12), (which, pol, g, fd)


def test_clean_jax_x64_contract_raises():
    """Every JAX entry point must refuse single precision (12 sites were
    checked by the original audit; this re-locks the 1-D entry)."""
    import jax.numpy as jnp
    old = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            rcwa.rcwa_efficiency_1d(
                period=0.8e-6, n_ridge=jnp.asarray(2.0),
                n_groove=jnp.asarray(1.0), n_substrate=jnp.asarray(1.5),
                n_superstrate=jnp.asarray(1.0), depth=jnp.asarray(0.35e-6),
                duty_cycle=0.5, wavelength=jnp.asarray(0.633e-6), n_orders=4)
    finally:
        jax.config.update("jax_enable_x64", old)
