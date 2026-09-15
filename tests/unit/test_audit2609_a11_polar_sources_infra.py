"""WP-A11 regression pins for AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 Z1-Z4
(partition POLAR-SOURCES-INFRA + the ORCHESTRATOR coatings row).

Every bar below is either an exact/bit-identity claim or carries its
derivation, its measured envelope and the pre-fix reading it has to separate
from.  Where a pre-fix reading is needed, it is produced IN THIS PROCESS --
either by an escape hatch that reproduces the old code path exactly
(``_schell_phase_realizations(pad_sigma=0.0)``) or by a local re-statement of
the pre-fix expression -- so the "fails before / passes after" claim is
measured on the running build rather than quoted (TESTING_STANDARDS S2 / rule
3).

Findings covered
----------------
Z1  ``coating_reflectance`` (+ JAX twin) routed 'te'/'S'/'tm'/junk down the
    p branch.
Z2  Gaussian-Schell sources realised the grid-PERIODISED coherence kernel.
Z3  ``estimate_lens_memory(lens_model='real')`` under-predicted;
    ``create_gaussian_beam`` dense meshgrid + out-of-place normalise;
    algebra ``FreeSpace`` warning spam + pitch-vs-ABCD contradiction;
    ``stokes_parameters`` / ``degree_of_polarization`` temporaries.
Z4  ``deprecated_alias`` stacklevel; ``algebra.from_prescription`` module
    shadowing the function; ``_check_2d_scalar_field`` accepting object dtype
    / ``np.matrix``; cache-registry collisions and swallowed clearer
    failures; ``_plane_wave_carrier`` odd-N centring; ``JonesField.propagate*``
    in-place contract; ``user_library``'s unbounded ``Pow``.
"""
from __future__ import annotations

import gc
import tracemalloc
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import _cache_registry as _CR
from lumenairy._validation import _check_2d_scalar_field
from lumenairy.elements.coatings import (
    _normalize_coating_pol,
    coating_reflectance,
)
from lumenairy.elements.polarization import (
    JonesField,
    _plane_wave_carrier,
    degree_of_polarization,
    stokes_parameters,
)
from lumenairy.elements.rcwa._core import _normalize_pol
from lumenairy.sources.core import (
    _resolve_complex_dtype,
    _schell_phase_realizations,
)

WL = 550e-9


def _first(x):
    """First scalar of a possibly-vectorised coating_reflectance return."""
    return float(np.atleast_1d(x)[0])


# ===========================================================================
# Z1 -- coatings polarization routing
# ===========================================================================

def _airy_R(n1, n2, n3, d, wl, th1, pol):
    """Analytic single-layer Airy reflectance, standard Fresnel convention.

    Independent of the library: a three-line closed form, not the
    characteristic-matrix product under test.  Complex ``cos(theta)`` on the
    decaying branch ``Im(n cos) >= 0`` so it is valid past critical too.
    """
    s1 = n1 * np.sin(th1)
    c1 = np.cos(th1)
    c2 = np.sqrt(1 - (s1 / n2) ** 2 + 0j)
    c2 = c2 if (n2 * c2).imag >= 0 else -c2
    c3 = np.sqrt(1 - (s1 / n3) ** 2 + 0j)
    c3 = c3 if (n3 * c3).imag >= 0 else -c3
    if pol == 's':
        r12 = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2)
        r23 = (n2 * c2 - n3 * c3) / (n2 * c2 + n3 * c3)
    else:
        r12 = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2)
        r23 = (n3 * c2 - n2 * c3) / (n3 * c2 + n2 * c3)
    beta = 2 * np.pi / wl * n2 * d * c2
    r = (r12 + r23 * np.exp(2j * beta)) / (1 + r12 * r23 * np.exp(2j * beta))
    return float(abs(r) ** 2)


# The audit's own fixture (POLAR-SOURCES-INFRA p2_coatings.py section D):
# 50 nm MgF2 on n = 1.52 at 60 deg, where R_s = 0.15427715 and
# R_p = 0.00322899 -- a 48x split, so a mis-routed spelling is unmissable.
_Z1_LAYERS = [(1.38, 50e-9)]
_Z1_ANGLE = np.radians(60.0)           # near Brewster for n_sub = 1.52
_Z1_NSUB = 1.52
_Z1_D = 50e-9


@pytest.mark.parametrize("spelling,branch", [
    ('s', 's'), ('S', 's'), ('te', 's'), ('TE', 's'), ('Te', 's'),
    ('p', 'p'), ('P', 'p'), ('tm', 'p'), ('TM', 'p'), ('tM', 'p'),
])
def test_z1_coating_reflectance_every_alias_reaches_its_own_branch(
        spelling, branch):
    """CONVENTIONS Section 7 names this file as accepting te/tm/s/p
    case-insensitively.  Pre-fix the code tested ``pol == 's'`` and sent
    everything else -- 'te', 'S', 'tm', junk -- down the p branch.

    Oracle: the analytic Airy formula above, evaluated for the branch the
    spelling is supposed to select.  Bar 1e-12: the two implementations are
    the same float64 arithmetic by different routes, measured |dR| <= 2.3e-16
    over this table (four decades below the bar); the quantity the bar has to
    resolve is the s-vs-p SPLIT at this fixture, R_s = 0.1543 vs
    R_p = 0.0032, i.e. 0.151 -- eleven decades above the bar.
    """
    R = _first(coating_reflectance(
        _Z1_LAYERS, WL, angle=_Z1_ANGLE, n_substrate=_Z1_NSUB,
        polarization=spelling)[0])
    want = _airy_R(1.0, 1.38, _Z1_NSUB, _Z1_D, WL, _Z1_ANGLE, branch)
    other = _airy_R(1.0, 1.38, _Z1_NSUB, _Z1_D, WL, _Z1_ANGLE,
                    'p' if branch == 's' else 's')
    assert abs(R - want) < 1e-12, (
        f"polarization={spelling!r} returned R={R!r}; the {branch} oracle is "
        f"{want!r} and the OTHER branch is {other!r}.")
    # Two-sided: the reading must also be far from the wrong branch, so a
    # coincidence cannot pass.
    assert abs(R - other) > 0.1


@pytest.mark.parametrize("junk", ['banana', '', 'ss', 'S ', 'TEM', 0, None,
                                  'average', 'unpolarized'])
def test_z1_coating_reflectance_rejects_junk(junk):
    """A spelling outside {s, te, p, tm, avg} must raise, not silently run
    the p branch.  CONVENTIONS Section 2: the message starts with the
    function name."""
    with pytest.raises(ValueError, match=r"^coating_reflectance: polarization"):
        coating_reflectance(_Z1_LAYERS, WL, angle=_Z1_ANGLE,
                            n_substrate=_Z1_NSUB, polarization=junk)


def test_z1_avg_is_the_mean_of_the_two_branches():
    """'avg' must still be the unpolarised average (unchanged behaviour)."""
    R_s = _first(coating_reflectance(_Z1_LAYERS, WL, angle=_Z1_ANGLE,
                                     n_substrate=_Z1_NSUB,
                                     polarization='s')[0])
    R_p = _first(coating_reflectance(_Z1_LAYERS, WL, angle=_Z1_ANGLE,
                                     n_substrate=_Z1_NSUB,
                                     polarization='p')[0])
    R_a = _first(coating_reflectance(_Z1_LAYERS, WL, angle=_Z1_ANGLE,
                                     n_substrate=_Z1_NSUB,
                                     polarization='AVG')[0])
    assert abs(R_a - 0.5 * (R_s + R_p)) < 1e-15


def test_z1_coatings_normalizer_accepts_exactly_the_rcwa_set():
    """Single source of truth: the coatings helper must accept precisely the
    set ``rcwa._core._normalize_pol`` accepts, plus 'avg'.  If the two ever
    drift, CONVENTIONS Section 7's "accepted everywhere" claim is false again.
    """
    probes = ['s', 'S', 'te', 'TE', 'p', 'P', 'tm', 'TM', 'avg', 'AVG',
              'banana', '', 'ss', 'sp', 'TEM', '0']
    for probe in probes:
        try:
            _normalize_pol('probe', probe)
            rcwa_ok = True
        except ValueError:
            rcwa_ok = False
        try:
            got = _normalize_coating_pol('probe', probe)
            coat_ok = True
        except ValueError:
            got = None
            coat_ok = False
        is_avg = probe.lower() == 'avg'
        assert coat_ok == (rcwa_ok or is_avg), (
            f"{probe!r}: rcwa accepts={rcwa_ok}, coatings accepts={coat_ok}")
        if coat_ok and not is_avg:
            assert got == {'te': 's', 'tm': 'p'}[_normalize_pol('probe', probe)]


def test_z1_te_reaches_the_s_coefficient_through_propagate_through_system():
    """The public wave path: a ``{'type': 'coating', 'polarization': 'te'}``
    element used to apply the TM coefficient to a TE beam.  Measured pre-fix
    |t| = {'s': 0.94840576, 'p': 0.9969635, 'te': 0.9969635}; post-fix 'te'
    reads 0.94840576.  The two candidates differ by 0.049 in amplitude
    (10 % in power), 13 decades above the 1e-14 agreement bar.
    """
    from lumenairy.propagators.system import propagate_through_system
    E = np.ones((8, 8), complex)
    amp = {}
    for pol in ('s', 'p', 'te', 'tm'):
        out = propagate_through_system(
            E, [{'type': 'coating', 'layers': _Z1_LAYERS, 'wavelength': WL,
                 'angle': _Z1_ANGLE, 'n_substrate': _Z1_NSUB,
                 'polarization': pol, 'port': 'transmission'}],
            dx=1e-6, wavelength=WL)
        arr = np.atleast_2d(out[0] if isinstance(out, tuple) else out)
        amp[pol] = float(abs(arr[0, 0]))
    assert abs(amp['te'] - amp['s']) < 1e-14
    assert abs(amp['tm'] - amp['p']) < 1e-14
    assert abs(amp['s'] - amp['p']) > 0.01     # the fixture really splits


# --- JAX twin ---------------------------------------------------------------

try:                                            # pragma: no cover - env probe
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except Exception:                               # pragma: no cover
    jax = jnp = None
    _HAS_JAX = False

# SCOPED skip (house pattern, cf. test_v5_5_3_coatings_jax.py): jax is an
# OPTIONAL extra (CONVENTIONS Section 10), not a resource precondition -- the
# NumPy arm of every claim above runs unconditionally.
_requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="could not import 'jax'")

if _HAS_JAX:                                    # pragma: no cover - env setup
    jax.config.update('jax_enable_x64', True)


@_requires_jax
@pytest.mark.parametrize("spelling,twin", [
    ('te', 's'), ('TE', 's'), ('S', 's'), ('tm', 'p'), ('TM', 'p'),
    ('P', 'p'),
])
def test_z1_jax_twin_routes_the_same_aliases(spelling, twin):
    """``coating_reflectance_jax`` carried the identical defect, so a
    gradient-based AR/HR design on ``polarization='te'`` optimised the TM
    stack.  Bar 1e-13: the NumPy/JAX agreement measured over this table is
    <= 1.2e-16 (three decades under), while the s-vs-p split is 0.151."""
    from lumenairy.elements.coatings import coating_reflectance_jax
    Rj = float(coating_reflectance_jax(
        _Z1_LAYERS, WL, angle=_Z1_ANGLE, n_substrate=_Z1_NSUB,
        polarization=spelling))
    Rn = _first(coating_reflectance(
        _Z1_LAYERS, WL, angle=_Z1_ANGLE, n_substrate=_Z1_NSUB,
        polarization=twin)[0])
    assert abs(Rj - Rn) < 1e-13


@_requires_jax
def test_z1_jax_twin_gradient_follows_the_named_polarization():
    """d R/d(thickness) under 'te' must equal the 's' gradient exactly (same
    computation) and differ from the 'p' one -- the property an inverse
    design actually depends on."""
    from lumenairy.elements.coatings import coating_reflectance_jax

    def g(pol):
        return float(jax.grad(lambda t: coating_reflectance_jax(
            [(1.38, t)], WL, angle=_Z1_ANGLE, n_substrate=_Z1_NSUB,
            polarization=pol))(100e-9))
    assert g('te') == g('s')
    assert g('tm') == g('p')
    assert abs(g('te') - g('tm')) > 1.0       # measured split ~1.0e6 1/m


@_requires_jax
def test_z1_jax_twin_rejects_junk():
    from lumenairy.elements.coatings import coating_reflectance_jax
    with pytest.raises(ValueError,
                       match=r"^coating_reflectance_jax: polarization"):
        coating_reflectance_jax(_Z1_LAYERS, WL, angle=_Z1_ANGLE,
                                n_substrate=_Z1_NSUB, polarization='banana')


# ===========================================================================
# Z2 -- Gaussian-Schell coherence kernel must not be the periodised one
# ===========================================================================

_Z2_N, _Z2_DX, _Z2_SIGMA, _Z2_K = 48, 1e-6, 6e-6, 1200      # sigma_g = L/8


def _mu_profile(pad_sigma, seed, N=_Z2_N, dx=_Z2_DX, sigma=_Z2_SIGMA,
                K=_Z2_K):
    """Empirical ``<phi(x+m dx) conj(phi(x))>`` vs separation index ``m``.

    Pair-averaged over every LINEAR (non-wrapping) column pair, every row and
    every realisation -- a zero-padded FFT correlation, so no wrapped pair is
    ever counted and the estimator itself introduces no periodicity.
    """
    kw = {} if pad_sigma is None else {'pad_sigma': pad_sigma}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        phi = _schell_phase_realizations(
            Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma,
            n_realizations=K, rng=np.random.default_rng(seed), **kw)
    F = np.fft.fft(phi, n=2 * N, axis=2)
    corr = np.fft.ifft(F * np.conj(F), axis=2)[:, :, :N]
    acc = corr.sum(axis=(0, 1)).real
    cnt = K * N * (N - np.arange(N))
    mu = acc / cnt
    return mu / mu[0]


def test_z2_realised_kernel_is_the_gaussian_not_the_periodised_one():
    """The FFT filter is a CIRCULAR convolution, so on the bare grid the
    realised two-point kernel is ``sum_m exp(-|d + mL|^2 / 2 sigma^2)``
    (Poisson summation), not the documented Gaussian.

    Two hypotheses, evaluated on the same ensemble:
      * documented Gaussian at the largest separation the grid can form,
        d = (N-1) dx = 47 um with sigma_g = 6 um  ->  mu = 4.7e-14;
      * grid-periodised Gaussian (L = 48 um, so the nearest image sits at
        1 um)                                     ->  mu = 0.9862.

    BAR 0.15 on |mu_emp - Gaussian|.  Measured envelope (12 seeds x 1200
    realisations, 2026-09-12): post-fix |mu(edge)| max 0.0220 (std 0.0102),
    max|mu - Gaussian| over all separations max 0.0229; pre-fix (the same
    generator with ``pad_sigma=0.0``, which reproduces the old code path
    bit-for-bit) mu(edge) mean +0.9886, std 0.0080.  The bar therefore sits
    6.8x above the post-fix envelope and 6.5x below the pre-fix signal.  The
    residual is pure Monte-Carlo sampling error, which shrinks as 1/sqrt(K)
    and carries no build dependence.
    """
    d = np.arange(_Z2_N) * _Z2_DX
    gauss = np.exp(-d ** 2 / (2 * _Z2_SIGMA ** 2))
    L = _Z2_N * _Z2_DX
    images = np.arange(-3, 4)
    wrapped = np.exp(
        -(d[:, None] + images * L) ** 2 / (2 * _Z2_SIGMA ** 2)).sum(axis=1)
    wrapped = wrapped / wrapped[0]
    assert gauss[-1] < 1e-12 and wrapped[-1] > 0.9    # the two really differ

    mu_fixed = _mu_profile(None, seed=20260912)
    assert np.abs(mu_fixed - gauss).max() < 0.15, (
        f"realised kernel departs from the documented Gaussian by "
        f"{np.abs(mu_fixed - gauss).max():.4f}")
    assert abs(mu_fixed[-1]) < 0.15, (
        f"spurious edge-to-edge coherence {mu_fixed[-1]:.4f}; the "
        f"Gaussian-Schell model says {gauss[-1]:.3e}")

    # Fail-before, measured in this process: the legacy path is the wrapped
    # kernel, and it is separated from the fixed one by ~0.98.
    mu_legacy = _mu_profile(0.0, seed=20260912)
    assert abs(mu_legacy[-1] - wrapped[-1]) < 0.15
    assert mu_legacy[-1] - mu_fixed[-1] > 0.5


def test_z2_unit_mean_intensity_is_preserved_by_the_pad():
    """The v5.4.6 P3-10 deterministic normalisation must survive the crop:
    the padded grid is statistically homogeneous, so ``sum_k |H|^2 / (Ny_p
    Nx_p)`` is the right constant for the cropped window too.

    Bar 0.05 on |E[<|phi|^2>] - 1|: with K = 600 realisations of a 48x48 grid
    the estimator's own standard error is ~1/sqrt(K * (L/sigma)^2) ~ 0.005
    (measured readings 0.9967-0.9999 across the audit's three sigma values),
    an order of magnitude inside the bar; a mis-scaled normalisation would
    move it by the ratio of padded to unpadded filter power (here 1.7x).
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        phi = _schell_phase_realizations(
            Ny=48, Nx=48, dx=_Z2_DX, dy=_Z2_DX, coherence_length=_Z2_SIGMA,
            n_realizations=600, rng=np.random.default_rng(4))
    assert abs(float(np.mean(np.abs(phi) ** 2)) - 1.0) < 0.05


def test_z2_pad_sigma_zero_reproduces_the_legacy_path_bit_for_bit():
    """The escape hatch has to be exact, or "pre-fix" comparisons (and any
    archived ensemble) are not reproducible.  Reference: a local restatement
    of the pre-v5.46 body."""
    N, dx, sigma, K, seed = 24, 2e-6, 5e-6, 5, 99
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    spec = np.exp(-(kx[None, :] ** 2 + ky[:, None] ** 2) * sigma ** 2 / 4.0)
    norm = np.sqrt(float(np.sum(np.abs(spec) ** 2) / (N * N)))
    rng = np.random.default_rng(seed)
    ref = np.empty((K, N, N), complex)
    for k in range(K):
        w = ((rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N)))
             / np.sqrt(2.0))
        ref[k] = np.fft.ifft2(np.fft.fft2(w) * spec) / norm
    got = _schell_phase_realizations(
        Ny=N, Nx=N, dx=dx, dy=dx, coherence_length=sigma, n_realizations=K,
        rng=np.random.default_rng(seed), pad_sigma=0.0)
    assert np.array_equal(got, ref)


def test_z2_warns_when_the_grid_holds_fewer_than_six_coherence_lengths():
    """``sigma_g > L/6`` is the regime the old docstring steered users into
    ("sigma_g >> w0 approaches the coherent limit" on a grid with L ~ 4 w0).
    The kernel is right there now, but the ensemble is aperture-dominated and
    the pad is expensive -- say so."""
    with pytest.warns(UserWarning, match=r"fewer than 6 coherence lengths"):
        _schell_phase_realizations(
            Ny=32, Nx=32, dx=1e-6, dy=1e-6, coherence_length=12e-6,
            n_realizations=2, rng=np.random.default_rng(0))
    # ... and stays silent comfortably inside the regime.
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        _schell_phase_realizations(
            Ny=32, Nx=32, dx=1e-6, dy=1e-6, coherence_length=2e-6,
            n_realizations=2, rng=np.random.default_rng(0))


def test_z2_public_factories_inherit_the_fix():
    """``create_gaussian_schell_source`` / ``create_schell_model_source`` both
    route through the generator, so both must show the un-periodised kernel.
    Property: the MCF between opposite edges of the illuminated region, which
    the wrap made ~100 % coherent."""
    N, dx, w0, sigma, K = 32, 2e-6, 24e-6, 8e-6, 800
    for maker in ('gsm', 'schell'):
        if maker == 'gsm':
            ens, _, _, _ = la.create_gaussian_schell_source(
                N=N, dx=dx, wavelength=WL, w0=w0, sigma_g=sigma,
                n_realizations=K, rng=7)
        else:
            x = (np.arange(N) - N / 2) * dx
            prof = np.exp(-2 * (x[None, :] ** 2 + x[:, None] ** 2) / w0 ** 2)
            ens, _, _, _ = la.create_schell_model_source(
                N=N, dx=dx, wavelength=WL, intensity_profile=prof,
                coherence_length=sigma, n_realizations=K, rng=7)
        row = ens[:, N // 2, :]
        j_lo, j_hi = N // 2 - N // 3, N // 2 + N // 3     # separation 2N/3 dx
        num = np.mean(row[:, j_hi] * np.conj(row[:, j_lo]))
        den = np.sqrt(np.mean(np.abs(row[:, j_hi]) ** 2)
                      * np.mean(np.abs(row[:, j_lo]) ** 2))
        mu = abs(num / den)
        # separation = 2N/3 dx = 42.7 um = 5.3 sigma -> |mu| = 8.6e-7;
        # the periodised kernel put an image at L - 42.7 = 21.3 um = 2.7
        # sigma -> |mu| = 0.029, and closer still for the pairs the MCF
        # eigen-decomposition uses.  Bar 0.15: measured post-fix |mu| <= 0.05
        # at K = 800 (the estimator's own 1/sqrt(K) floor is 0.035).
        assert mu < 0.15, f"{maker}: |mu| at 5.3 sigma is {mu:.4f}"


# ===========================================================================
# Z3 -- memory / warning-spam items
# ===========================================================================

def _old_gaussian_beam(N, dx, w0, dtype=None, normalize='peak'):
    """The pre-v5.46 body of ``create_gaussian_beam`` (dense meshgrid,
    out-of-place normalise), restated locally as the bit-identity oracle."""
    sigma = w0 / np.sqrt(2.0)
    x = (np.arange(N) - N / 2) * dx
    y = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, y)
    E = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    E = E.astype(_resolve_complex_dtype(dtype))
    if normalize == 'peak':
        mx = float(np.abs(E).max())
        if mx > 0:
            E = E / mx
    elif normalize == 'power':
        norm = np.sqrt(np.sum(np.abs(E) ** 2) * dx * dx)
        if float(norm) > 0:
            E = E / norm
    return E, x, y


@pytest.mark.parametrize("N", [17, 64])
@pytest.mark.parametrize("normalize", ['peak', 'power', 'none'])
@pytest.mark.parametrize("dtype", [None, np.complex64])
def test_z3_gaussian_beam_broadcast_is_bit_identical(N, normalize, dtype):
    """Dropping the dense meshgrid and normalising in place must not move a
    single bit (the arithmetic is elementwise and the sign fold
    ``(-S)/c == -(S/c)`` is exact in IEEE)."""
    E, x, y = la.create_gaussian_beam(N, 2e-6, 633e-9, w0=20e-6,
                                      normalize=normalize, dtype=dtype)
    Er, xr, yr = _old_gaussian_beam(N, 2e-6, 20e-6, dtype=dtype,
                                    normalize=normalize)
    assert E.dtype == Er.dtype
    assert np.array_equal(E, Er)
    assert np.array_equal(x, xr) and np.array_equal(y, yr)


@pytest.mark.parametrize("dtype,bar", [(np.complex128, 2.0),
                                       (np.complex64, 3.0)])
def test_z3_gaussian_beam_peak_over_output(dtype, bar):
    """``create_gaussian_beam`` was the last factory still building a dense
    meshgrid: peak/output 3.00x (c128) and 5.00x (c64), ~2.1 GB of avoidable
    transient at N = 8192.

    tracemalloc peaks for straight-line NumPy code are exact allocation
    counts, not timings -- deterministic, with no cross-build spread.
    Measured at N = 1024 on 2026-09-12: 1.50x (c128) / 2.00x (c64) after,
    3.00x / 5.00x before (the pre-fix arm is measured in this same test, so
    the separation is not quoted from elsewhere).  Bars 2.0 / 3.0 sit between
    the two readings with >= 25 % margin on each side.
    """
    N = 1024
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    E, _, _ = la.create_gaussian_beam(N, 1e-6, 633e-9, w0=200e-6, dtype=dtype)
    _, peak_new = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    out_bytes = E.nbytes
    del E
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    E_old, _, _ = _old_gaussian_beam(N, 1e-6, 200e-6, dtype=dtype)
    _, peak_old = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del E_old
    gc.collect()
    assert peak_new / out_bytes < bar, (
        f"peak/output {peak_new / out_bytes:.2f}x (pre-fix arm measured "
        f"{peak_old / out_bytes:.2f}x)")
    assert peak_old / out_bytes > bar        # the bar really separates them


#: Grid the warm-up call runs on, in pixels.  DERIVED, not chosen: at 64 it is
#: below the size at which ``apply_real_lens`` takes its deferred-import branch,
#: so ~11.2 MB of one-time module imports (9.55 MB of bytecode at
#: ``<frozen importlib._bootstrap_external>``, plus dask / jinja2 / pathlib /
#: inspect objects) landed INSIDE the measured region on Windows, where those
#: modules are not already resident -- 0.45 MB of the same line on Linux, which
#: is why the bar below held there and read 0.81 here.  At 256 the warm-up takes
#: the same branch and the measured peak is BYTE-IDENTICAL on the two arms.
#: The branch threshold is BRACKETED, not guessed: warming at 128 still leaves
#: the Windows reading contaminated (peak 60.8 MB, ratio 0.81), warming at 256
#: does not (46.5 MB, ratio 1.064, the same byte as Linux), so it lies in
#: (128, 256] and 256 is the first power of two above it.
_Z3_LENS_WARM_N = 256


def test_z3_estimate_lens_memory_real_bounds_apply_real_lens():
    """``lens_model='real'`` is the DOCUMENTED model for ``apply_real_lens``;
    it under-predicted the measured peak by 1.6x (default parallel_amp=True)
    to 2.8x (parallel_amp=False), i.e. a pre-flight budget under-reserved --
    the exact failure ``check_sim_memory`` exists to prevent.

    Bar: the estimate must BOUND the measurement (>= 1.0, the fail-safe
    direction) and not by more than 1.6x (an estimate that over-reserves by
    more than that stops being useful).  Neither bar is moved here; what is
    fixed is the MEASUREMENT, which was picking up allocations that are not
    ``apply_real_lens``'s working set at all (see ``_Z3_LENS_WARM_N``).

    MEASURED after, in FRESH processes, one (N, dtype) each, on Windows
    py3.14 / numpy 2.4.4 AND WSL py3.12 / numpy 2.4.6 -- every number below is
    byte-identical on the two arms, which is the point::

        N     dtype       estimate   first call   est/peak   retained
        512   complex128    49.5 MB     46.5 MB     1.064     6.09 grids
        1024  complex128   198.0 MB    159.8 MB     1.239     6.02 grids
        512   complex64     33.8 MB     31.5 MB     1.072     6.01 grids
        1024  complex64    135.1 MB    102.1 MB     1.323     6.04 grids

    so the gate's own row (512, complex128) sits 6.4 % above the fail-safe
    bar and 33 % below the upper one, and the worst row of the four is still
    21 % inside the upper bar.

    The RETAINED column is asserted too, as the premise that the reading is
    clean: the call keeps the N-sized FFT and ASM caches it built
    (``propagators/fft_infra.py`` 4 complex grids, ``propagators/asm.py`` 1 + 1)
    and nothing else, 6.01 .. 6.09 grids on every row and both arms.  With the
    old 64-pixel warm-up the same reading is 8.38 grids on Windows under
    pytest and 9.49 standalone -- the deferred imports -- so this guard
    catches exactly the contamination that red this test, two-sided: 15 %
    above the worst clean reading (6.09) and 16 % below the lowest
    contaminated one (8.38).
    """
    N, dt = 512, np.complex128
    wl, dx = 633e-9, 30e-3 / N
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                         aperture=25e-3)
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3, dtype=dt)
    E = np.ascontiguousarray(E)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        la.apply_real_lens(E[:_Z3_LENS_WARM_N, :_Z3_LENS_WARM_N].copy(),
                           prescription=rx, wavelength=wl, dx=dx)
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del out
    gc.collect()
    held = retained / float(N * N * 16)
    assert 5.5 < held < 7.0, (
        f"the measured call retained {held:.2f} full complex grids, not the "
        f"6.0 of N-sized FFT/ASM cache it builds -- something else is being "
        f"allocated inside the measured region (with a 64-pixel warm-up this "
        f"reads 9.49 on Windows: deferred module imports), so the peak below "
        f"is not apply_real_lens's working set")
    est = la.estimate_lens_memory(N, dt, lens_model='real')
    ratio = est / peak
    assert 1.0 <= ratio <= 1.6, (
        f"estimate_lens_memory(lens_model='real') = {est / 1e6:.1f} MB vs a "
        f"measured apply_real_lens peak of {peak / 1e6:.1f} MB "
        f"(ratio {ratio:.2f}).")


def test_z3_freespace_default_preserves_the_pitch_its_abcd_reports():
    """``FreeSpace(method='auto')`` routed far-field segments to SAS, which
    resamples: on this 4f chain the ABCD said magnification -1 (so, the input
    pitch) while the delivered pitch was 1.93x the input, and the library's
    own "no stable contract" UserWarning fired three times per evaluation at
    algebra/primitives.py -- advice the caller could not act on, because the
    argument it names is the algebra layer's own.
    """
    from lumenairy.algebra.primitives import FreeSpace, ThinLens
    wl, f, N, dx = 633e-9, 200e-3, 256, 8e-6
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=100e-6)
    E = np.roll(E, int(round(300e-6 / dx)), axis=1)
    chain = (FreeSpace(f) * ThinLens(f) * FreeSpace(2 * f)
             * ThinLens(f) * FreeSpace(f))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        Eo, dx_out, dy_out, _ = chain((E, dx, dx, wl))
        spam = [r for r in rec if issubclass(r.category, UserWarning)]
    assert spam == [], [str(r.message)[:120] for r in spam]
    M = np.asarray(chain.abcd)
    assert np.abs(M + np.eye(2)).max() == 0.0          # exact 4f inverter
    # |A| = 1 -> the ABCD says the field returns at the input pitch.  Exact
    # equality, not a tolerance: 'asm' does not touch the grid at all.
    assert dx_out == dx and dy_out == dx
    # ... and the physics is still the 4f inversion.
    xs = (np.arange(N) - N / 2) * dx
    I0, I1 = np.abs(E) ** 2, np.abs(Eo) ** 2
    c_in = (I0.sum(0) * xs).sum() / I0.sum()
    c_out = (I1.sum(0) * xs).sum() / I1.sum()
    assert abs(c_out + c_in) < 0.02 * abs(c_in)        # inverted, |M| = 1
    assert abs(I1.sum() / I0.sum() - 1.0) < 0.01


def test_z3_freespace_auto_still_available_and_no_longer_spams():
    """Naming ``method='auto'`` keeps the dispatcher's kernel choice (and its
    resampling, honestly reported) but must not emit the un-actionable
    return-contract warning any more."""
    from lumenairy.algebra.primitives import FreeSpace, ThinLens
    wl, f, N, dx = 633e-9, 200e-3, 256, 8e-6
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=100e-6)
    chain = (FreeSpace(f, method='auto') * ThinLens(f)
             * FreeSpace(2 * f, method='auto') * ThinLens(f)
             * FreeSpace(f, method='auto'))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, dx_out, _, _ = chain((E, dx, dx, wl))
        spam = [r for r in rec if issubclass(r.category, UserWarning)]
    assert spam == [], [str(r.message)[:120] for r in spam]
    assert dx_out != dx          # 'auto' really does resample here


def test_z3_freespace_anamorphic_dy_still_threaded():
    """The v5.30 reason for the legacy return contract was the anamorphic
    y-pitch: on the wrapped path a pitch-preserving kernel reports
    ``result.dy == result.dx``.  That branch must keep its own contract."""
    from lumenairy.algebra.primitives import FreeSpace
    E = np.ones((64, 64), complex)
    for method in ('asm', 'auto'):
        _, dx_out, dy_out, _ = FreeSpace(1e-3, method=method)(
            (E, 2e-6, 3e-6, 633e-9))
        assert (dx_out, dy_out) == (2e-6, 3e-6), method


def _old_stokes(field):
    Ex, Ey = field.Ex, field.Ey
    return {'S0': np.abs(Ex) ** 2 + np.abs(Ey) ** 2,
            'S1': np.abs(Ex) ** 2 - np.abs(Ey) ** 2,
            'S2': 2 * np.real(Ex * np.conj(Ey)),
            'S3': -2 * np.imag(Ex * np.conj(Ey))}


def _old_dop(field):
    S = _old_stokes(field)
    S0 = S['S0']
    finite = np.isfinite(S0)
    s0_max = float(S0[finite].max()) if finite.any() else 0.0
    eps = float(np.finfo(S0.dtype).eps
                if np.issubdtype(S0.dtype, np.floating) else np.finfo(float).eps)
    live = S0 > s0_max * eps * eps
    safe = np.where(live, S0, 1.0)
    with np.errstate(invalid='ignore'):
        dop = np.sqrt((S['S1'] / safe) ** 2 + (S['S2'] / safe) ** 2
                      + (S['S3'] / safe) ** 2)
    dop = np.clip(np.where(live, dop, 0.0), 0.0, 1.0)
    return np.where(np.isnan(S0) | np.isnan(dop), np.nan, dop)


def _pathological_field(N, dtype):
    rng = np.random.default_rng(7)
    Ex = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(dtype)
    Ey = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(dtype)
    Ex[0, 0] = Ey[0, 0] = 0.0            # dark pixel
    Ex[0, 1] = np.nan                    # NaN in
    Ex[1, 0] = np.inf                    # S0 = inf with S1 = inf - inf = NaN
    Ex[2, 0] = Ey[2, 0] = 1e-160         # underflow regime
    return JonesField(Ex, Ey, 1e-6, 1e-6)


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_z3_stokes_and_dop_are_bit_identical(dtype):
    """Computing |Ex|^2, |Ey|^2 and Ex conj(Ey) once (and accumulating the DOP
    in place) must not move a bit -- including on dark / NaN / inf /
    underflow pixels, where the guards live.  ``Ex`` must stay the LEFT
    operand of the complex product: numpy's vectorised complex multiply is
    NOT bitwise commutative here (measured 1.8e-15 on S3 at N = 64)."""
    for special in (False, True):
        f = _pathological_field(64, dtype)
        if not special:
            f = JonesField(np.nan_to_num(f.Ex, posinf=3.0),
                           np.nan_to_num(f.Ey, posinf=3.0), 1e-6, 1e-6)
        new, old = stokes_parameters(f), _old_stokes(f)
        for key in ('S0', 'S1', 'S2', 'S3'):
            assert new[key].dtype == old[key].dtype
            assert np.array_equal(new[key], old[key], equal_nan=True), key
        with np.errstate(invalid='ignore'):
            d_new, d_old = degree_of_polarization(f), _old_dop(f)
        assert d_new.dtype == d_old.dtype
        assert np.array_equal(d_new, d_old, equal_nan=True)


#: How far above a whole number of full grids a tracemalloc peak may sit.
#: DERIVED, two-sided: the reading is an EXACT allocation count in units of one
#: full grid plus tracemalloc's own bookkeeping, and that bookkeeping measured
#: 624 .. 2984 B over 5 repeats on each of two arms (Windows py3.14 /
#: numpy 2.4.4 and WSL py3.12 / numpy 2.4.6), i.e. at most 3.6e-04 grids at
#: N = 1024.  0.05 is two decades above that spread and 1.3 decades below the
#: 1.0 that separates one allocation count from the next -- so it can absorb
#: any bookkeeping and can never absorb an array.
_Z3_PEAK_SLACK = 0.05


def _z3_peak_units(fn, unit):
    """Peak transient of ``fn()`` in units of one full-grid REAL array."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    res = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del res
    gc.collect()
    return peak / unit


def _numpy_elides_binary_temporaries(field, unit):
    """MEASURED premise: does this build rewrite ``Ex * conj(Ey)`` into the
    ``conj`` temporary's own buffer instead of allocating a second array?

    NumPy's ``temp_elide.c`` does exactly that when an operand is an
    unreferenced temporary, but only where the optimisation is compiled in (it
    needs ``backtrace()``) and only when its stack walk can confirm the
    temporary came from the interpreter.  Both are BUILD properties: measured
    ACTIVE on WSL (py3.12, numpy 2.4.6) and on the py3.11 and py3.14 Linux
    runners of CI run 34914295323, INACTIVE on Windows (py3.14, numpy 2.4.4).
    So the peak of an expression written with a free temporary is a per-build
    quantity and has to be read on the running arm, never assumed from it.

    Returns ``(free, held)`` in full-grid REAL arrays.  A complex grid is two
    of those, so ``conj(Ey)`` plus the product is 4.00 unelided and 2.00
    elided.  ``held`` binds the temporary to a name, which lifts its reference
    count to 2 and puts elision out of reach on every build: it is 4.00
    everywhere, and is asserted, so a reading of "no elision here" can never
    come from an instrument that measured nothing at all.  ``free`` is 4.00
    where elision is unavailable and 2.00 where it is.  MEASURED: 4.000 /
    4.000 on Windows, 2.000 / 4.000 on WSL.
    """
    free = _z3_peak_units(lambda: field.Ex * np.conj(field.Ey), unit)

    def _held():
        c = np.conj(field.Ey)             # named -> refcount 2 -> not elidable
        return field.Ex * c

    held = _z3_peak_units(_held, unit)
    assert 3.9 < held < 4.1, (
        f"the elision instrument read {held:.3f} full grids for a conj plus a "
        f"complex product that must cost exactly 4.00 -- it is not measuring "
        f"what it claims, so no premise can be drawn from it")
    return free, held


def _z3_peaks(N=1024):
    """The four peak readings, in full-grid REAL arrays, with the field."""
    unit = N * N * 8
    f = _pathological_field(N, np.complex128)
    f = JonesField(np.nan_to_num(f.Ex, posinf=3.0),
                   np.nan_to_num(f.Ey, posinf=3.0), 1e-6, 1e-6)
    peaks = {tag: _z3_peak_units(lambda fn=fn: fn(f), unit)
             for tag, fn in (('stokes_new', stokes_parameters),
                             ('stokes_old', _old_stokes),
                             ('dop_new', degree_of_polarization),
                             ('dop_old', _old_dop))}
    return f, unit, peaks


def test_z3_stokes_and_dop_peak_arrays():
    """The UNCONDITIONAL half of the peak-array claim, in full-grid REAL
    arrays (N*N*8 B).

    6.00 is the derived FLOOR for a bit-identical Stokes implementation: the
    four outputs, plus the one complex cross term (2 real grids) the exact
    S2 / S3 need while S3 is being written.  Both shipped functions sit ON
    that floor, and the DOP costs no more than the ``stokes_parameters`` call
    inside it, which is the whole content of the in-place accumulation.  The
    pre-fix DOP does not: it holds 8.25.

    Every claim here holds on both arms of the two-arm ladder -- Windows
    py3.14 / numpy 2.4.4 (no temporary elision) and WSL py3.12 / numpy 2.4.6
    (elision active) -- and on every CI runner, because none of them can be
    reached by eliding a temporary.  MEASURED, 5 repeats per arm::

        stokes_new  6.000237 .. 6.000345 (Win)  6.000234 .. 6.000333 (WSL)
        dop_new     6.000237             (Win)  6.000234             (WSL)
        dop_old     8.250353 .. 8.250356 (Win)  8.250346 .. 8.250349 (WSL)

    The one reading that IS build-dependent -- the pre-fix Stokes form's
    seventh grid -- is premise-gated in
    :func:`test_z3_the_pre_fix_stokes_form_holds_a_seventh_grid`.
    """
    _f, _unit, peaks = _z3_peaks()
    # the four outputs + the complex cross term, and not one array more
    assert 6.0 <= peaks['stokes_new'] < 6.0 + _Z3_PEAK_SLACK, peaks
    assert 6.0 <= peaks['dop_new'] < 6.0 + _Z3_PEAK_SLACK, peaks
    # the DOP accumulates over the very arrays its own Stokes call returned
    assert peaks['dop_new'] <= peaks['stokes_new'] + _Z3_PEAK_SLACK, peaks
    # ... which the pre-fix DOP did not: 2.25 grids of avoidable transient
    assert peaks['dop_old'] > 6.5, peaks
    assert peaks['dop_old'] - peaks['dop_new'] > 2.0, peaks
    # and no arm may make the shipped Stokes form the more expensive one
    assert peaks['stokes_new'] <= peaks['stokes_old'] + _Z3_PEAK_SLACK, peaks


def test_z3_the_pre_fix_stokes_form_holds_a_seventh_grid():
    """PREMISE-GATED (TESTING_STANDARDS S3).  The pathology the shipped Stokes
    form removed -- the second full-grid COMPLEX temporary that
    ``Ex * conj(Ey)``, written twice, costs -- is observable only on a build
    where NumPy does not elide that temporary away.

    The premise is measured on the running arm by
    :func:`_numpy_elides_binary_temporaries`, never assumed from the platform.
    Where it holds, the pre-fix form peaks at 7.00 grids against the shipped
    form's 6.00 (Windows py3.14 / numpy 2.4.4: 7.000074 vs 6.000237).  Where
    NumPy elides, the pre-fix form is rewritten into the same 6.00 and there
    is no seventh grid to see: WSL py3.12 / numpy 2.4.6 and the py3.11 and
    py3.14 Linux runners of CI run 34914295323 all read 6.000234, which is
    what red this gate's previous ``6.5 < stokes_old`` form.  That arm asserts
    the collapse explicitly and then skips WITH the reading, so it can never
    pass silently.

    The DOP half of the claim needs no gate and is asserted unconditionally in
    the test above: 8.25 -> 6.00 on every arm measured.
    """
    f, unit, peaks = _z3_peaks()
    free, held = _numpy_elides_binary_temporaries(f, unit)
    if free < held - 1.0:                  # a whole complex grid saved
        assert 6.0 <= peaks['stokes_old'] < 6.0 + _Z3_PEAK_SLACK, (
            f"NumPy elided the pre-fix form's complex temporary, so it must "
            f"land on the same 6.00-grid floor as the shipped one, but it "
            f"read {peaks['stokes_old']:.6f}: {peaks}")
        pytest.skip(
            f"premise absent on this arm: NumPy's temporary elision is ACTIVE "
            f"(Ex*conj(Ey) peaks at {free:.3f} full grids free, {held:.3f} "
            f"with the temporary name-bound), so the pre-fix Stokes form "
            f"allocates no seventh grid -- measured stokes_old="
            f"{peaks['stokes_old']:.6f} against stokes_new="
            f"{peaks['stokes_new']:.6f}, both on the 6.00 floor")
    assert 7.0 <= peaks['stokes_old'] < 7.0 + _Z3_PEAK_SLACK, (
        f"elision is inactive here (free={free:.3f}, held={held:.3f}), so the "
        f"pre-fix form must hold its seventh grid: {peaks}")
    assert peaks['stokes_old'] - peaks['stokes_new'] > 0.95, peaks


# ===========================================================================
# Z4 -- infrastructure
# ===========================================================================

def test_z4_deprecated_alias_warning_names_the_caller():
    """``_shim`` emitted at stacklevel 3, which from inside ``_emit``
    resolves to ``_deprecation.py`` itself -- so ``-W
    error::DeprecationWarning`` tracebacks and
    ``filterwarnings(..., module='myapp')`` pointed into lumenairy.  The
    warning must name the line in THIS file."""
    from lumenairy import _deprecation as dep
    alias = dep.deprecated_alias(lambda: None, old_name='old_thing')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        alias()                                     # <- must be named
    got = [r for r in rec if issubclass(r.category, DeprecationWarning)]
    assert len(got) == 1
    assert got[0].filename == __file__, (
        f"warning attributed to {got[0].filename}:{got[0].lineno}, not the "
        f"caller ({__file__})")

    # The direct helpers carry the same arithmetic: a public function that
    # calls one must have the warning attributed to ITS caller, not to its
    # own body.
    def public_fn():
        dep.warn_deprecated_kwarg('old', 'new', function='public_fn',
                                  version_removed='6.0')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        public_fn()
    got = [r for r in rec if issubclass(r.category, DeprecationWarning)]
    assert len(got) == 1 and got[0].filename == __file__


def test_z4_live_alias_points_at_user_code():
    """The library's two live aliases go through the same shim."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            la.load_zmx_prescription("does-not-exist.zmx")
        except Exception:
            pass
    got = [r for r in rec if issubclass(r.category, DeprecationWarning)]
    assert got and got[0].filename == __file__


@pytest.mark.parametrize("obj,exc", [
    (np.zeros((4, 4), complex), None),
    (np.zeros((4, 4), float), None),          # amplitude mask -- still valid
    (np.zeros((4, 4), np.int64), None),
    (np.zeros((4, 4), bool), None),
    (np.zeros((8, 8), complex)[::2, ::2], None),
    (np.empty((4, 4), object), TypeError),
    (np.zeros((4, 4), dtype='U3'), TypeError),
])
def test_z4_check_2d_scalar_field_dtype_gate(obj, exc):
    """The guard's message has always said "expected 2-D complex" while it
    enforced only ``ndim == 2``, so an object-dtype array reached the kernels
    and ran NumPy's Python-object slow path."""
    if exc is None:
        _check_2d_scalar_field(obj, 'probe', input_kind='field')
    else:
        with pytest.raises(exc, match=r"^probe: "):
            _check_2d_scalar_field(obj, 'probe', input_kind='field')


def test_z4_check_2d_scalar_field_rejects_np_matrix():
    """``np.matrix``'s ``*`` is a matrix product, so every elementwise mask /
    phase-screen multiply downstream silently became a matmul.  The
    difference is not subtle: for this fixture the elementwise product and
    the matrix product differ by 3 in every entry."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')          # np.matrix is deprecated
        M = np.matrix(np.ones((4, 4), complex))
        with pytest.raises(TypeError, match=r"^probe: np.matrix"):
            _check_2d_scalar_field(M, 'probe', input_kind='field')
        # np.asarray(M) is the documented fix and must pass.
        _check_2d_scalar_field(np.asarray(M), 'probe', input_kind='field')
        mask = np.ones((4, 4), complex)
        assert not np.array_equal(np.asarray(M * mask),
                                  np.asarray(M) * mask)


def test_z4_cache_registry_warns_on_a_colliding_name():
    """A second, DIFFERENT clearer registered under a live name was dropped
    silently, leaving that cache permanently unclearable -- in the module
    whose purpose is to retire the "fix N, miss N+1" pattern."""
    calls = []
    name = 'probe_a11_dup'
    try:
        _CR.register_cache_clearer(name, lambda: calls.append('first'))
        with pytest.warns(RuntimeWarning, match=r"already registered to a "
                                                r"different clearer"):
            _CR.register_cache_clearer(name, lambda: calls.append('second'))
    finally:
        _CR._unregister_for_test(name)


def test_z4_cache_registry_is_still_reload_idempotent():
    """Re-registering the SAME call site (what ``importlib.reload`` produces)
    must stay silent -- warnings churn during interactive development."""
    name = 'probe_a11_reload'

    def make():
        def clearer():
            return None
        return clearer
    try:
        _CR.register_cache_clearer(name, make())
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _CR.register_cache_clearer(name, make())     # same file+line
    finally:
        _CR._unregister_for_test(name)


def test_z4_clear_all_reports_a_failing_clearer():
    """A clearer that raises was swallowed with no signal, so a caller who
    cleared caches to free RAM before a big allocation got no warning that a
    cache stayed full."""
    name = 'probe_a11_boom'

    def boom():
        raise ImportError("clearer broken")
    try:
        _CR.register_cache_clearer(name, boom)
        with pytest.warns(RuntimeWarning, match=r"still holding memory"):
            _CR.clear_all_registered_caches()
    finally:
        _CR._unregister_for_test(name)
    # ... and the walk stays best-effort: no exception escapes.
    _CR.clear_all_registered_caches()


@pytest.mark.parametrize("nx", [7, 8, 9, 16])
def test_z4_plane_wave_carrier_uses_the_package_grid_centring(nx):
    """``_plane_wave_carrier`` centred on the integer ``nx // 2`` while
    ``apply_jones_matrix``'s callable grid, every source factory and every
    element grid use ``N / 2`` -- identical for even N, half a pixel apart
    for odd N, so a JonesField from ``jones_field_from_orders`` on an odd
    grid sat dx/2 off every element applied to it afterwards.

    Oracle: the phase of a unit-kx carrier read against the package grid.
    Exact to 1e-12 (float64 trig round-off on these magnitudes is ~1e-16)."""
    dx = 1e-6
    kx_m = 0.37                       # normalised by k0; arbitrary, non-trivial
    c = _plane_wave_carrier(kx_m, 0.0, WL, nx, nx, dx, dx)
    k0 = 2.0 * np.pi / WL
    xg = (np.arange(nx) - nx / 2) * dx            # the package convention
    want = np.exp(1j * k0 * kx_m * xg)
    assert np.abs(c[0] - want).max() < 1e-12
    # Cross-check against the grid apply_jones_matrix hands to a callable.
    seen = {}

    def cb(X, Y):
        seen['x'] = np.asarray(X)[0] if np.asarray(X).ndim == 2 else np.asarray(X)
        J = np.zeros((2, 2, nx, nx), complex)
        J[0, 0] = J[1, 1] = 1.0
        return J
    from lumenairy.elements.polarization import apply_jones_matrix
    apply_jones_matrix(JonesField(np.ones((nx, nx), complex),
                                  np.zeros((nx, nx), complex), dx), cb)
    assert np.allclose(seen['x'], xg, rtol=0, atol=1e-18)


def test_z4_jones_propagate_methods_are_in_place_and_return_self():
    """Two docstrings said "Returns new grid spacings"; all of them return
    ``self`` and rebind ``dx``/``dy`` on the object.  Pin the real contract
    (not the prose -- TESTS-ARCH: tests that assert docstring strings)."""
    N, dx = 32, 2e-6
    Ex = np.ones((N, N), complex)
    Ey = np.zeros((N, N), complex)
    jf = JonesField(Ex.copy(), Ey.copy(), dx)
    out = jf.propagate(1e-4, WL)
    assert out is jf
    assert jf.dx == dx and jf.dy == dx          # ASM preserves the pitch
    assert not np.shares_memory(jf.Ex, Ex)      # rebound, caller's array intact
    assert np.array_equal(Ex, np.ones((N, N), complex))

    jf2 = JonesField(np.ones((N, N), complex), np.zeros((N, N), complex), dx)
    out2 = jf2.propagate_fresnel(0.05, WL)
    assert out2 is jf2
    assert jf2.dx != dx and jf2.dy != dx        # Fresnel is pitch-CHANGING
    assert abs(jf2.dx - WL * 0.05 / (N * dx)) < 1e-18


def test_z4_algebra_from_prescription_is_callable():
    """``from lumenairy.algebra import from_prescription`` bound the MODULE,
    so the call the submodule's own ``__all__`` and docstring advertise raised
    ``TypeError: 'module' object is not callable``."""
    import importlib
    from lumenairy.algebra import from_prescription as fp
    assert callable(fp)
    rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7',
                         aperture=25.4e-3)
    op = fp(rx, 633e-9)
    ref = la.Operator.from_prescription(rx, 633e-9)
    assert np.array_equal(np.asarray(op.abcd), np.asarray(ref.abcd))
    # The dotted module path must keep working for existing callers.
    mod = importlib.import_module('lumenairy.algebra.from_prescription')
    assert getattr(mod, 'from_prescription') is fp


@pytest.mark.parametrize("expr", [
    '2 ** (10 ** 9)', '1 << (10 ** 9)', '(10 ** 9) ** (10 ** 9)', '3 ** 5000',
])
def test_z4_user_library_bounds_integer_growth(expr):
    """The allowlist AST interpreter is a genuine sandbox, but CPython's
    ``int`` is arbitrary precision, so ``2 ** (10 ** 9)`` asked for a 125 MB
    integer from an untrusted user-library string.  The guard is a bit-length
    check, so it refuses before allocating (measured 0.02 ms per rejection)."""
    from lumenairy.user_library import _safe_eval_expression
    with pytest.raises(ValueError, match=r"^load_phase_mask: the integer"):
        _safe_eval_expression(expr, {'np': np})


@pytest.mark.parametrize("expr,want", [
    ('2 ** 10', 1024), ('10 ** 9', 10 ** 9), ('1 << 32', 1 << 32),
    ('2 ** -3', 0.125), ('(-1) ** 1000000000', 1),
])
def test_z4_user_library_pow_guard_passes_real_expressions(expr, want):
    """The guard must not touch anything a phase mask legitimately writes."""
    from lumenairy.user_library import _safe_eval_expression
    assert _safe_eval_expression(expr, {'np': np}) == want


def test_z4_user_library_pow_guard_ignores_array_operands():
    """Array exponents never reach the bignum path; the guard must be a
    no-op there."""
    from lumenairy.user_library import _safe_eval_expression
    X = np.linspace(-1.0, 1.0, 8)[None, :] * np.ones((8, 1))
    out = _safe_eval_expression('X ** 2', {'X': X, 'np': np})
    assert np.array_equal(out, X ** 2)


# ===========================================================================
# VERIFY-A11 -- pins added by the independent re-verification of this WP.
# Each covers a defect the WP's own arms did not reach; the "fails before"
# arm is measured in this process.
# ===========================================================================

def test_verify_a11_stokes_bit_identical_for_a_mixed_precision_field():
    """``JonesField`` does not harmonise its two components: it coerces a
    real / integer input to complex but leaves a ``complex64`` one alone, so
    ``JonesField(Ex_c128, Ey_c64, dx)`` is constructible.

    The lean ``stokes_parameters`` writes ``Ex * conj(Ey)`` into a buffer made
    from ``Ey``; unless that buffer carries the PROMOTED dtype, NumPy's default
    ``same_kind`` casting rounds the complex128 product back down into the
    complex64 buffer.  Measured before the ``astype(result_type(...))``:
    S2 / S3 came back **float32** with max |diff| 2.4e-7 / 3.4e-7 against the
    four-expression form (relative 2.1e-8 / 2.9e-8, i.e. exactly float32 eps),
    while S0 / S1 stayed float64.  Bar: bitwise equality and dtype equality --
    an exact claim with no tolerance to calibrate.
    """
    rng = np.random.default_rng(5)
    N = 17
    ex = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    ey = (rng.standard_normal((N, N))
          + 1j * rng.standard_normal((N, N))).astype(np.complex64)
    for Ex, Ey in ((ex, ey), (ex.astype(np.complex64), ey.astype(np.complex128))):
        jf = JonesField(Ex, Ey, 1e-6, 1e-6)
        # The premise: the field really is heterogeneous after construction.
        assert jf.Ex.dtype != jf.Ey.dtype
        new, old = stokes_parameters(jf), _old_stokes(jf)
        for key in ('S0', 'S1', 'S2', 'S3'):
            assert new[key].dtype == old[key].dtype, (key, new[key].dtype,
                                                      old[key].dtype)
            assert np.array_equal(new[key], old[key], equal_nan=True), key
        with np.errstate(invalid='ignore'):
            d_new, d_old = degree_of_polarization(jf), _old_dop(jf)
        assert d_new.dtype == d_old.dtype
        assert np.array_equal(d_new, d_old, equal_nan=True)


def test_verify_a11_cache_registry_separates_partials_of_different_functions():
    """``_clearer_identity``'s no-code-object fallback keyed on
    ``type(fn).__name__``, so EVERY ``functools.partial`` compared equal: a
    partial of one clearer and a partial of a different one collided
    SILENTLY -- the exact defect Z4 makes audible for plain functions.

    Two-sided: different wrapped functions must warn, the same wrapped
    function must not (reload-idempotence), and an instance of a callable
    class must stay silent against itself (nothing about an instance survives
    a reload, so a per-instance key would make every reload warn).
    """
    import functools

    def clear_a():
        return None

    def clear_b():
        return None

    name = 'verify_a11_partial_probe'

    def register_pair(first, second):
        _CR._unregister_for_test(name)
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                _CR.register_cache_clearer(name, first)
                _CR.register_cache_clearer(name, second)
                return [r for r in rec
                        if issubclass(r.category, RuntimeWarning)]
        finally:
            _CR._unregister_for_test(name)

    warned = register_pair(functools.partial(clear_a),
                           functools.partial(clear_b))
    assert len(warned) == 1, (
        "two partials wrapping DIFFERENT clearers collided silently; the "
        "second cache would never be cleared")
    assert 'already registered to a different clearer' in str(warned[0].message)
    assert register_pair(functools.partial(clear_a),
                         functools.partial(clear_a)) == []
    assert register_pair(functools.partial(functools.partial(clear_a)),
                         functools.partial(clear_a)) == []

    class Clearer:
        def __call__(self):
            return None

    assert register_pair(Clearer(), Clearer()) == []


@pytest.mark.parametrize("d1,d2,want_A", [(3.0, 1.5, 2.0),      # 1:2 imager
                                          (1.5, 3.0, 0.5)])     # 2:1 imager
def test_verify_a11_freespace_pitch_is_preserved_whatever_the_abcd_says(
        d1, d2, want_A):
    """The PITCH CONTRACT note on :attr:`Operator.abcd` is a claim about every
    chain, not just the 4f inverter the WP measured.  A pitch-PRESERVING
    propagator delivers ``dx_out == dx_in`` for ANY ``|A|`` -- a magnified
    image lands on more pixels of the same grid -- so ``|A|`` is the RAY
    magnification only.

    Exact equality, not a tolerance: ``'asm'`` does not touch the grid at all.
    The two arms bracket unity (``|A| = 2`` and ``|A| = 0.5``) so a docstring
    that re-asserts "``|A|`` is the grid magnification" cannot pass either.

    The fixture is sized so the beam stays INSIDE the window on both arms
    (measured power kept 1.0000 / 1.0000 at N = 512, dx = 8 um, f = 50 mm,
    w0 = 300 um), which keeps the two claims separable: this test pins the
    pitch, and the O-1 truncation warning below pins the clipping.  A
    narrower grid genuinely truncates these chains -- at N = 128 the same
    geometry keeps well under half its power -- and would conflate them.
    """
    from lumenairy.algebra.primitives import FreeSpace, ThinLens
    wl, f, N, dx = 633e-9, 50e-3, 512, 8e-6
    E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=300e-6)
    chain = FreeSpace(d1 * f) * ThinLens(f) * FreeSpace(d2 * f)
    A = float(np.asarray(chain.abcd)[0, 0])
    assert abs(abs(A) - want_A) < 1e-9        # the fixture really magnifies
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E_out, dx_out, dy_out, _ = chain((E, dx, dx, wl))
        spam = [r for r in rec if issubclass(r.category, UserWarning)]
    assert spam == [], [str(r.message)[:120] for r in spam]
    # The premise of "no warnings": this chain does not clip.
    assert abs(float((np.abs(E_out) ** 2).sum()
                     / (np.abs(E) ** 2).sum()) - 1.0) < 0.02
    assert dx_out == dx and dy_out == dx, (
        f"|A| = {abs(A)} but the delivered pitch moved: "
        f"dx_out/dx_in = {dx_out / dx}")


def test_verify_a11_schell_kernel_on_an_anisotropic_odd_grid():
    """The WP measured the kernel only on square, even, isotropic grids.  The
    pad is computed per axis (``ceil(pad_sigma*sigma_g/dx)`` vs ``/dy``) and
    the crop offset is ``(N_pad - N)//2``, so ``dy != dx`` with an ODD N is the
    arm that would catch a wrong axis or an off-by-one crop.

    Estimator: a plain sliced sample covariance -- no FFT, so it cannot
    inherit the generator's own circularity, and it counts only pairs that
    exist on the grid.  Oracle: the documented Gaussian, which at the largest
    separation this grid can form is 2.3e-21 on both axes.

    BAR 0.15 on |mu(edge)|.  Measured 2026-09-12 over 8 seeds x 1200
    realisations: post-fix max |mu| 0.0116 (x) / 0.0182 (y), std 0.008;
    pre-fix (``pad_sigma=0.0``, which reproduces the old code path
    bit-for-bit) mean +0.9688 (x) / +0.9359 (y), std <= 0.009.  The bar sits
    8.2x above the post-fix envelope and 6.2x below the pre-fix signal, and
    the residual is pure Monte-Carlo error (falls as 1/sqrt(K), no build
    dependence).
    """
    Ny, Nx, dx, dy, sigma, K = 27, 40, 1e-6, 1.5e-6, 4e-6, 1200

    def edge_mu(pad_sigma):
        kw = {} if pad_sigma is None else {'pad_sigma': pad_sigma}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            phi = _schell_phase_realizations(
                Ny=Ny, Nx=Nx, dx=dx, dy=dy, coherence_length=sigma,
                n_realizations=K, rng=np.random.default_rng(20260912), **kw)
        assert phi.shape == (K, Ny, Nx)
        den = float(np.mean(np.abs(phi) ** 2))
        mx = float((np.vdot(phi[:, :, :1], phi[:, :, Nx - 1:])
                    / (K * Ny)).real) / den
        my = float((np.vdot(phi[:, :1, :], phi[:, Ny - 1:, :])
                    / (K * Nx)).real) / den
        return mx, my, den

    gauss_x = float(np.exp(-((Nx - 1) * dx) ** 2 / (2 * sigma ** 2)))
    gauss_y = float(np.exp(-((Ny - 1) * dy) ** 2 / (2 * sigma ** 2)))
    assert gauss_x < 1e-12 and gauss_y < 1e-12        # the oracle says ~0

    mx, my, mean_I = edge_mu(None)
    assert abs(mx) < 0.15 and abs(my) < 0.15, (mx, my)
    # Unit mean intensity must survive the anisotropic crop too (the
    # normalisation constant is computed on the PADDED grid).
    assert abs(mean_I - 1.0) < 0.05

    # Fail-before, measured in this process on the same grid.
    lx, ly, _ = edge_mu(0.0)
    assert lx > 0.5 and ly > 0.5, (lx, ly)


# ---------------------------------------------------------------------------
# VERIFY-A11 O-1 / O-2 -- the orchestrator's rulings on the two open items.
# ---------------------------------------------------------------------------

_O1_WL, _O1_N, _O1_DX = 633e-9, 256, 8e-6
#: z_max = N*dx^2/lambda = 25.88 mm for this grid.
_O1_ZMAX = _O1_N * _O1_DX ** 2 / _O1_WL


def _o1_eval(op, E, dx=_O1_DX, dy=None):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = op((E, dx, dy if dy is not None else dx, _O1_WL))
        spam = [r for r in rec if issubclass(r.category, UserWarning)]
    return out, spam


def test_o1_freespace_reports_far_field_truncation_and_stays_quiet_otherwise():
    """``method='asm'`` keeps the window fixed, so a beam that has diverged
    past ``N*dx`` is clipped -- silently, before this warning.  Measured at
    N = 256, dx = 8 um, 633 nm: a 20 um waist over 500 mm keeps **0.0961** of
    its power (true ``w(z) = 5037 um`` vs a +-1024 um window) where a
    resampling kernel keeps 0.9998.

    Two-sided, and that is the whole point: ``z > z_max`` alone is NOT the
    predicate.  Every leg of the audit's own 4f fixture is past ``z_max``
    (f = 200 mm vs 25.9 mm) and none of them truncates -- warning on the gate
    would put the Z3 headline back at three UserWarnings per 4f evaluation,
    which is the spam this version removed.  So the benign arms below are as
    load-bearing as the truncating ones.

    Bar: warned / not warned -- a decision, not a reading.  The separating
    quantity is the power kept, measured 2026-09-12 as 0.0466-0.1267 on the
    truncating arms and 0.98239-1.00000 on the benign ones (the tightest
    benign case being the 4f chain's own 2f leg); the shipped tolerance of
    5 % sits 2.8x above the worst benign loss and 17x below the smallest real
    one.
    """
    from lumenairy.algebra.primitives import FreeSpace, ThinLens

    def beam(w0):
        E, _, _ = la.create_gaussian_beam(_O1_N, _O1_DX, _O1_WL, w0=w0)
        return E

    # --- truncating: must warn, with the Section 2 prefix and the remedy ---
    E20 = beam(20e-6)
    (Eo, _, _, _), spam = _o1_eval(FreeSpace(0.5), E20)
    kept = float((np.abs(Eo) ** 2).sum() / (np.abs(E20) ** 2).sum())
    assert kept < 0.2, kept                      # the fixture really clips
    assert len(spam) == 1, [str(r.message)[:120] for r in spam]
    msg = str(spam[0].message)
    assert msg.startswith('FreeSpace._apply: ')
    assert "method='auto'" in msg                # the named remedy
    assert 'z_max' in msg

    # --- benign, and PAST z_max in every case: must stay silent ---
    for w0, z in ((100e-6, 0.4),        # the 4f chain's own 2f leg
                  (200e-6, 0.5),
                  (400e-6, 0.026),      # just past z_max
                  (20e-6, 0.05)):
        assert z > _O1_ZMAX or w0 == 400e-6
        _, spam = _o1_eval(FreeSpace(z), beam(w0))
        assert spam == [], (w0, z, [str(r.message)[:120] for r in spam])

    # --- the Z3 fixture itself: still zero, still pitch-preserving ---
    f = 200e-3
    E100 = beam(100e-6)
    chain = (FreeSpace(f) * ThinLens(f) * FreeSpace(2 * f)
             * ThinLens(f) * FreeSpace(f))
    (Eo, dx_out, dy_out, _), spam = _o1_eval(chain, E100)
    assert spam == [], [str(r.message)[:120] for r in spam]
    assert dx_out == _O1_DX and dy_out == _O1_DX
    assert abs(float((np.abs(Eo) ** 2).sum()
                     / (np.abs(E100) ** 2).sum()) - 1.0) < 0.01


def test_o1_far_field_warning_is_once_per_instance_and_skips_the_remedy():
    """An optimiser re-applies one ``FreeSpace`` thousands of times; the
    condition is a property of ``(z, N, dx, lambda)``, so it is reported once
    per instance.  And the kernels the warning RECOMMENDS must never trigger
    it, or the advice would be circular."""
    from lumenairy.algebra.primitives import FreeSpace
    E, _, _ = la.create_gaussian_beam(_O1_N, _O1_DX, _O1_WL, w0=20e-6)

    one = FreeSpace(0.5)
    assert sum(len(_o1_eval(one, E)[1]) for _ in range(4)) == 1
    assert sum(len(_o1_eval(FreeSpace(0.5), E)[1]) for _ in range(3)) == 3

    for method in ('auto', 'sas', 'fresnel'):    # resampling -> no truncation
        _, spam = _o1_eval(FreeSpace(0.5, method=method), E)
        assert spam == [], (method, [str(r.message)[:120] for r in spam])
    # 'rs' is pitch-preserving like 'asm', so it must warn.
    assert len(_o1_eval(FreeSpace(0.5, method='rs'), E)[1]) == 1

    # Anamorphic (the branch that FORCES 'asm'): warns, keeps its own pitch,
    # and names the remedy that applies there instead of 'auto'.
    Ea, _, _ = la.create_gaussian_beam((128, 128), 2e-6, _O1_WL, w0=8e-6,
                                       dy=3e-6)
    (_, dx_out, dy_out, _), spam = _o1_eval(FreeSpace(20e-3), Ea, dx=2e-6,
                                            dy=3e-6)
    assert (dx_out, dy_out) == (2e-6, 3e-6)
    assert len(spam) == 1 and 'forces the pitch-preserving' in str(spam[0].message)


@pytest.mark.parametrize("bad", ['Traced', 'REAL', 'Real', 'banana', '',
                                 'real ', ' traced', None, 0, True])
def test_o2_estimate_lens_memory_rejects_an_unknown_lens_model(bad):
    """``_real = (lens_model != 'traced')`` meant every typo silently selected
    the ``'real'`` model and returned a DIFFERENT pre-flight budget (49.5 MB
    vs 47.1 MB at N = 512 / complex128) with no signal -- in the function
    ``check_sim_memory`` exists to make trustworthy.  Case-SENSITIVE, matching
    the lower-case string contract the module compares against throughout."""
    with pytest.raises(ValueError,
                       match=r"^estimate_lens_memory: lens_model must be one "
                             r"of \['real', 'traced'\]"):
        la.estimate_lens_memory(512, np.complex128, lens_model=bad)


def test_o2_estimate_lens_memory_keeps_both_valid_models_distinct():
    """Counter-pin: the guard must not collapse the vocabulary it protects.
    The two entry points have separate calibrations, so their budgets differ
    (measured 47.1 MB traced vs 49.5 MB real at N = 512 / complex128 -- a
    5.1 % split that a wrong token used to hand back silently)."""
    traced = la.estimate_lens_memory(512, np.complex128, lens_model='traced')
    real = la.estimate_lens_memory(512, np.complex128, lens_model='real')
    assert traced > 0 and real > 0 and traced != real
    # The default is 'traced' (unchanged).
    assert la.estimate_lens_memory(512, np.complex128) == traced
    # ... and the row-band branch reads the same closed vocabulary.
    assert la.estimate_lens_memory(2048, np.complex128, lens_model='real',
                                   sag_chunk_rows=128) > 0
    with pytest.raises(ValueError, match=r"^estimate_lens_memory: lens_model"):
        la.estimate_lens_memory(2048, np.complex128, lens_model='Real',
                                sag_chunk_rows=128)


if __name__ == '__main__':                      # pragma: no cover
    pytest.main([__file__, '-v'])
