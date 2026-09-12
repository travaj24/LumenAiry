"""WP-A4 regression pins for the Maslov / GBD / FGA lens propagators.

AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, findings S2-S7 (report section
2.5).  Every numeric bar carries its oracle, the measured value on both
sides of the fix, and the decades of headroom.

Oracles, none of them produced by the code under test:

* **S2** -- the closed-form value of a Gaussian (Fresnel) integral with a
  quadratic exponent, ``exp(i pi sigma / 4) / sqrt|det H|``, on synthetic
  charts built in this file.
* **S3** -- a physically NULL prescription edit (appending a
  zero-thickness FLAT dummy surface, which moves the trace's last surface
  to the vertex plane without changing any optics).  The canonical chart
  must be invariant under it.
* **S4** -- ``angular_spectrum_propagate``, exact for a band-limited field
  on this grid, through a two-flat-surface "lens" that is pure free space.
* **S5** -- the textbook ``q``-parameter Gaussian beam, written here.
* **S7** -- ``jax.jit`` traceability and the dtype contract.
"""
import math
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import raytrace as rt
from lumenairy.elements import lenses_maslov as LM
from lumenairy.elements.lenses import _multi_indices_total_degree
from lumenairy.propagators.asm import angular_spectrum_propagate

# ===========================================================================
# S2 -- local_quadrature: principal axes, a window, and the window divided out
# ===========================================================================

def _synthetic_chart(A, B, Cxy=0.0, c=1.0):
    """``OPD = 0.5 A u3^2 + 0.5 B u4^2 + Cxy u3 u4`` waves, ``s1 = c u3,
    c u4``, ``E_in == 1`` -- whose exact v2 integral is closed-form.

    (``u^2 = (T0 + T2)/2`` in the Chebyshev basis.)
    """
    mi = _multi_indices_total_degree(4, 2)
    idx = {k: j for j, k in enumerate(mi)}
    co = np.zeros(len(mi))
    co[idx[(0, 0, 0, 0)]] += 0.25 * A + 0.25 * B
    co[idx[(0, 0, 2, 0)]] += 0.25 * A
    co[idx[(0, 0, 0, 2)]] += 0.25 * B
    co[idx[(0, 0, 1, 1)]] += Cxy
    sx = np.zeros(len(mi))
    sx[idx[(0, 0, 1, 0)]] = c
    sy = np.zeros(len(mi))
    sy[idx[(0, 0, 0, 1)]] = c
    K = [np.array([k[i] for k in mi], np.int64) for i in range(4)]
    return co, sx, sy, K, mi


def _exact_quadratic(A, B, Cxy=0.0, c=1.0, lam=1.0e-6):
    """``INT exp(i pi (A u3^2 + B u4^2 + 2 Cxy u3 u4)) d^2u`` times the
    S4 Van Vleck weight of this chart.

    The chart has ``ds1/dv2 = c I`` (constant), so the Van Vleck density is
    ``sqrt(c^2) = c`` and the kernel prefactor is ``1/(i lambda)``.
    """
    H = np.array([[A, Cxy], [Cxy, B]], float)
    ev = np.linalg.eigvalsh(H)
    sig = int(np.sign(ev).sum())
    gauss = np.exp(1j * np.pi * sig / 4.0) / math.sqrt(abs(np.linalg.det(H)))
    return gauss * c


def _run_local_quadrature(A, B, Cxy=0.0, n=8, ws=3.0, lam=1.0e-6):
    co, sx, sy, K, mi = _synthetic_chart(A, B, Cxy)
    u0 = np.zeros((1, 1))
    ib = np.array([True])
    val = LM._integrate_local_quadrature(
        co, sx, sy, K[0], K[1], K[2], K[3], 2, 1, u0, u0, ib,
        1.0, 1.0, lambda a, b: np.ones_like(a, dtype=np.complex128),
        30, 1e-12, n, ws, lambda *a, **k: None, False)[0, 0]
    return val


@pytest.mark.parametrize('A,B,Cxy,pre_fix_rel', [
    (40.0, 40.0, 0.0, 1.19),
    (40.0, 4.0, 0.0, 1.19),
    (4.0, 40.0, 0.0, 9.12),     # the eigenvalue/axis swap
    (100.0, 10.0, 0.0, 1.19),
    (40.0, 40.0, 30.0, 2.48),   # cross term
    (60.0, 20.0, 25.0, 3.07),
])
def test_s2_local_quadrature_is_exact_on_a_quadratic_chart(A, B, Cxy,
                                                           pre_fix_rel):
    """``local_quadrature`` at its SHIPPED defaults, against the closed form.

    Three defects compounded (audit S2): the sampling box was scaled by the
    Hessian EIGENVALUES but laid out on the COORDINATE axes (so the two
    widths swapped whenever ``H44 > H33``, and ``H34`` was ignored
    entirely), the samples were a hard-truncated uniform Riemann sum of a
    chirp (leaving the Fresnel endpoint oscillation, an ``O(1/extent)``
    error that does NOT shrink with the sample count), and ``np.clip``
    folded out-of-box samples onto the box edge at full cell weight.

    MEASURED at the defaults (``local_n_samples=8``,
    ``local_window_sigma=3.0``): post-fix 1.15e-14 / 1.29e-15 / 5.87e-15 /
    1.45e-14 / 7.83e-15 / 2.67e-16 relative on the six charts; pre-fix
    1.19 / 1.19 / 9.12 / 1.19 / 2.48 / 3.07.

    Bar: 1e-9.  Five decades above the worst measured residual and 9
    decades below the smallest pre-fix error.
    """
    got = _run_local_quadrature(A, B, Cxy)
    exact = _exact_quadratic(A, B, Cxy)
    rel = abs(got - exact) / abs(exact)
    assert rel < 1e-9, (
        f'A={A} B={B} H34={Cxy}: local_quadrature {got!r} vs closed form '
        f'{exact!r}, rel {rel:.3e} (pre-fix {pre_fix_rel})')


def test_s2_local_quadrature_is_symmetric_under_swapping_the_two_axes():
    """The axis swap, isolated.  ``(A, B)`` and ``(B, A)`` are the same
    integral with the coordinates relabelled, so the two answers must be
    identical -- pre-fix they differed by a factor 7.7 in relative error
    (1.19 vs 9.12) because the narrow-curvature axis got the narrow window.
    """
    for A, B in ((40.0, 4.0), (200.0, 4.0), (1000.0, 10.0)):
        a = _run_local_quadrature(A, B)
        b = _run_local_quadrature(B, A)
        assert abs(a - b) <= 1e-12 * abs(a), (
            f'(A={A}, B={B}) gave {a!r} but (A={B}, B={A}) gave {b!r}')


def test_s2_local_quadrature_does_not_have_a_convergence_floor():
    """No 40 % floor: the error must fall (or stay at machine precision)
    as the sample count rises, at every window size the caller can pick
    inside the chart box.

    MEASURED post-fix (A = B = 40): 1e-14 ... 1e-15 at every
    ``(window_sigma, n)`` in ``{3, 4, 6, 10} x {8, 16, 32, 64, 128}``.
    Pre-fix the same table read 1.2e+00 / 4.7e-01 / 4.1e-01 / 4.0e-01 /
    4.0e-01 along the ``window_sigma = 3`` row -- flat from n = 32 on.
    """
    exact = _exact_quadratic(40.0, 40.0)
    for ws in (3.0, 4.0, 6.0, 10.0):
        for n in (8, 16, 32, 64, 128):
            rel = abs(_run_local_quadrature(40.0, 40.0, n=n, ws=ws)
                      - exact) / abs(exact)
            assert rel < 1e-9, (
                f'window_sigma={ws} n={n}: rel {rel:.3e} (pre-fix 4.0e-01 '
                f'at the n -> infinity floor)')


def test_s2_auto_no_longer_routes_to_local_quadrature():
    """``integration_method='auto'`` must resolve to ``'quadrature'`` or
    ``'stationary_phase'``, never to ``'local_quadrature'``.

    The pre-fix router sent any chart needing more than ``_N_V2_AUTO_MAX``
    uniform samples to ``local_quadrature`` -- measured relL2 2.58 against
    a converged uniform quadrature on an f = 6 mm singlet exit plane,
    versus 0.84 for ``stationary_phase`` (and both are warned about now).
    """
    import inspect
    src = inspect.getsource(LM.apply_real_lens_maslov)
    block = src[src.index("if integration_method == 'auto':"):]
    block = block[:block.index('if n_v2 is None:')]
    # Only the ASSIGNMENT matters; the surrounding comment names both
    # integrators deliberately (it records the measurement that chose).
    code = [ln for ln in block.splitlines()
            if not ln.lstrip().startswith('#')]
    assign = [ln for ln in code if 'integration_method = (' in ln]
    assert len(assign) == 1, f'auto assignment not found in:\n{block}'
    tail = '\n'.join(code[code.index(assign[0]):code.index(assign[0]) + 3])
    assert "'local_quadrature'" not in tail, (
        "the 'auto' router still resolves to local_quadrature:\n" + tail)
    assert "'stationary_phase'" in tail


def test_s2_v2_oscillation_bound_uses_the_total_variation():
    """``sum |c_k| * max(k3, k4)``, not ``sum |c_k|``.

    Oracle: for a single term ``c T_k(u3)`` the phase sweeps the interval
    ``k`` times, so the cycle count is ``|c| k`` -- written out here.
    MEASURED on the audit's real fitted charts the two differ by up to
    2.47x (f = 2 mm biconvex at order 8: 136.2 -> 337.1).
    """
    mi = _multi_indices_total_degree(4, 3)
    coef = np.zeros(len(mi))
    idx = {k: j for j, k in enumerate(mi)}
    coef[idx[(0, 0, 3, 0)]] = 2.0      # 2 waves on T_3(u3) -> 6 cycles
    coef[idx[(0, 0, 0, 1)]] = 5.0      # 5 waves on T_1(u4) -> 5 cycles
    coef[idx[(1, 0, 0, 0)]] = 99.0     # constant in v2 -> 0
    got = LM._v2_oscillation_bound(mi, coef)
    assert got == pytest.approx(2.0 * 3 + 5.0 * 1, rel=0, abs=1e-12), (
        f'total-variation bound {got} != 11.0; the pre-fix excursion sum '
        f'would have given 7.0')


# ===========================================================================
# S3 -- the canonical chart lives on the exit VERTEX plane
# ===========================================================================

_S3_LAM = 1.0e-6
_S3_AP = 0.30e-3


def _s3_prescriptions():
    """A plano-convex with a CURVED last surface, and the same optic with a
    zero-thickness FLAT dummy appended -- a physically null edit that puts
    ``trace``'s last surface ON the vertex plane."""
    import copy
    p1 = la.make_singlet(np.inf, -2.0e-3, 0.7e-3, 'N-BK7', aperture=_S3_AP)
    p2 = copy.deepcopy(p1)
    p2['surfaces'].append({'radius': np.inf, 'conic': 0.0,
                           'glass_before': 'air', 'glass_after': 'air'})
    p2['thicknesses'] = list(p2['thicknesses']) + [0.0]
    return p1, p2


def _s3_capture_chart(prescription, monkeypatch_target=None):
    """Run the driver and capture the (design matrix, RHS) it fits.

    ``B[:, 0]`` is the linear-detrended OPD in waves, ``B[:, 1:]`` the
    entrance coordinates -- i.e. the canonical chart.
    """
    N, dx = 64, 4.0e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    E0 = ((R2 <= (0.5 * _S3_AP) ** 2)
          * np.exp(-R2 / (0.22e-3) ** 2)).astype(complex)
    cap = {}
    orig = LM._solve_fit

    def spy(A, B):
        cap['A'], cap['B'] = A.copy(), B.copy()
        return orig(A, B)

    LM._solve_fit = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_maslov(
                E0, prescription=prescription, wavelength=_S3_LAM, dx=dx,
                normalize_output='none', collimated_input=True,
                integration_method='quadrature', n_v2=32, poly_order=4,
                ray_field_samples=10, ray_pupil_samples=10)
    finally:
        LM._solve_fit = orig
    return cap['A'], cap['B']


def test_s3_canonical_chart_is_invariant_under_a_null_prescription_edit():
    """Appending a zero-thickness FLAT dummy surface changes no optics, so
    the fitted canonical chart ``(s2, v2, OPD)`` must be identical.

    Oracle: the edit is physically null by construction -- refraction
    air->air at a plane is the identity and the propagation distance is 0.
    What it DOES change is where ``rt.trace`` leaves the rays: on the
    curved surface (``z = sag(rho)``) for the original, on the vertex plane
    for the edited one.  So this is a direct probe of the exit-vertex
    transfer, needing no third-party ray tracer.

    MEASURED: post-fix the two design matrices agree to 4.4e-15 and the two
    OPD right-hand sides to 2.3e-13 waves (float64 noise on a ~1055-wave
    OPL).  PRE-fix the OPD differed by **4.90 waves** at rho = 0.14 mm --
    exactly ``|sag| = 4.906 um`` at lambda = 1 um.

    Bar: 1e-9 waves.  Four decades above the measured 2.3e-13 and 9
    decades below the pre-fix 4.90.
    """
    p1, p2 = _s3_prescriptions()
    A1, B1 = _s3_capture_chart(p1)
    A2, B2 = _s3_capture_chart(p2)
    assert A1.shape == A2.shape, (
        f'premise: both prescriptions must produce the same ray set; got '
        f'{A1.shape} vs {A2.shape}')
    d_opd = float(np.max(np.abs(B1[:, 0] - B2[:, 0])))
    d_s1 = float(np.max(np.abs(B1[:, 1:] - B2[:, 1:])))
    d_A = float(np.max(np.abs(A1 - A2)))
    assert d_opd < 1e-9, (
        f'OPD chart differs by {d_opd:.3e} waves under a null prescription '
        f'edit (measured 2.3e-13 post-fix, 4.90 waves pre-fix)')
    assert d_s1 < 1e-12, f's1 chart differs by {d_s1:.3e} m'
    assert d_A < 1e-12, f'design matrix differs by {d_A:.3e}'


def test_s3_driver_transfers_the_exit_rays_to_the_vertex_plane():
    """White-box: the driver must call the shared exit-vertex operator.

    The ray-level content is what :meth:`TraceResult.at_exit_vertex`
    guarantees (pinned by WP-A1); what is pinned here is that
    ``apply_real_lens_maslov`` reads the chart THROUGH it rather than
    straight off ``image_rays``, which is the S3 defect.
    """
    import inspect
    src = inspect.getsource(LM.apply_real_lens_maslov)
    assert 'tr.at_exit_vertex()' in src, (
        'apply_real_lens_maslov must read its exit chart through '
        'TraceResult.at_exit_vertex (audit S3 / section 15.1)')
    assert 'tr.image_rays' not in src, (
        'apply_real_lens_maslov still reads image_rays directly')


# ===========================================================================
# S4 -- the Van Vleck density and the k/(2 pi i) prefactor
# ===========================================================================

def _free_space_gap(z, aperture):
    return {'name': 'gap', 'aperture_diameter': aperture,
            'surfaces': [
                {'radius': np.inf, 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'air'},
                {'radius': np.inf, 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'air'}],
            'thicknesses': [z]}


@pytest.mark.parametrize('lam,z', [(1.0e-6, 0.5e-3), (1.0e-6, 2.0e-3),
                                   (2.0e-6, 1.0e-3)])
def test_s4_free_space_maslov_is_absolutely_normalised(lam, z):
    """``apply_real_lens_maslov`` through a pure free-space "lens".

    Oracle: ``angular_spectrum_propagate``, exact for a band-limited field
    on this grid.  The chart is two flat air surfaces, where
    ``ds1/dv2 = -z I`` EXACTLY, so ``|det J| = z^2`` and the two candidate
    weights differ by a clean, z-dependent factor.

    MEASURED with ``normalize_output='none'``: post-fix
    ``|E_maslov / E_ASM| = 0.9980 ... 0.9997`` with
    ``arg = -0.0020 ... +0.0024`` rad.  PRE-fix the same ratio was
    ``i * lambda * z`` -- modulus 4.998e-10 / 1.996e-09 / 1.996e-09 on
    these three rows (``ratio/(lambda z) = 0.9996`` at every wavelength and
    distance) with ``arg = +pi/2``.

    Bar: 1 % on the modulus and 0.05 rad on the phase.  The residual is the
    Chebyshev-chart + quadrature accuracy (measured 0.2-0.3 %, and the
    pixel-to-pixel spread of the ratio is 1.4e-3 ... 2.3e-3); the pre-fix
    modulus is 9 decades below the bar and its phase 30x above it.
    """
    N, dx, w0 = 48, 4.0e-6, 30e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(complex)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Em = np.asarray(la.apply_real_lens_maslov(
            E0.copy(), prescription=_free_space_gap(z, N * dx * 0.9),
            wavelength=lam, dx=dx, normalize_output='none',
            integration_method='quadrature', n_v2=64, input_na=0.06,
            poly_order=4, ray_field_samples=12, ray_pupil_samples=12))
    ref = angular_spectrum_propagate(E0, z, lam, dx)
    m = np.abs(ref) > 0.2 * np.abs(ref).max()
    r = Em[m] / ref[m]
    mod = float(np.mean(np.abs(r)))
    arg = float(np.angle(np.mean(r)))
    assert abs(mod - 1.0) < 0.01, (
        f'lam={lam} z={z}: |E_maslov/E_ASM| = {mod:.6e} (pre-fix '
        f'lambda*z = {lam * z:.3e})')
    assert abs(arg) < 0.05, (
        f'lam={lam} z={z}: arg(E_maslov/E_ASM) = {arg:+.4f} rad (pre-fix '
        f'+pi/2 from the missing 1/i)')


# ===========================================================================
# S5 -- the GBD beamlet Gouy / Collins phase sign
# ===========================================================================

def test_s5_single_beamlet_reproduces_the_analytic_gaussian():
    """One beamlet through free space vs the textbook Gaussian beam.

    Oracle written here: ``E(r, z) = (w0/w) exp(-r^2/w^2)
    exp(i(k z + k r^2 z / (2 (z^2 + zR^2)) - arctan(z/zR)))`` under the
    library's ``exp(-i omega t)`` / ``exp(+i k z)`` convention
    (CONVENTIONS section 7).

    MEASURED: post-fix ``arg(E/A) = 0.000000`` and relative L2
    1.41e-13 / 5.37e-13 / 4.20e-12 at ``z = 0.5 / 2 / 10 zR`` with NO phase
    fit.  PRE-fix ``arg(E/A) = +2 arctan(z/zR)`` exactly (0.927295 /
    2.214297 / 2.942255 rad, matching ``2 psi`` to 1e-13) and relative L2
    0.894 / 1.789 / 1.990 -- the Gouy phase carried with the wrong SIGN.

    Bar: 1e-9 relative L2.  Three decades above the measured 4.2e-12 and 8
    decades below the pre-fix 0.894.
    """
    from lumenairy.propagators.gbd import (
        BeamletBundle,
        propagate_beamlets_freespace,
        reconstruct_field_from_beamlets,
    )
    lam, w0 = 1.0e-6, 30e-6
    k = 2 * math.pi / lam
    zR = math.pi * w0 ** 2 / lam
    N, dx = 128, 2.0e-6
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    for factor in (0.5, 2.0, 10.0):
        z = factor * zR
        b = BeamletBundle(
            positions=np.zeros((1, 3)),
            directions=np.array([[0.0, 0.0, 1.0]]),
            Q=np.array([-1j / zR], dtype=np.complex128),
            amplitude=np.array([1.0 + 0.0j]),
            waist0=np.array([w0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bz = propagate_beamlets_freespace(b, z, lam)
            E = np.asarray(reconstruct_field_from_beamlets(
                bz, Ny=N, Nx=N, dx=dx, dy=dx, wavelength=lam))
        w_z = w0 * math.sqrt(1.0 + (z / zR) ** 2)
        psi = math.atan(z / zR)
        A = ((w0 / w_z) * np.exp(-R2 / w_z ** 2)
             * np.exp(1j * (k * z + k * R2 * z / (2 * (z ** 2 + zR ** 2))
                            - psi)))
        rel = float(np.linalg.norm(E - A) / np.linalg.norm(A))
        assert rel < 1e-9, (
            f'z = {factor} zR: relative L2 {rel:.3e} with no phase fit '
            f'(pre-fix {0.894 if factor == 0.5 else 1.99:.3f}; the whole '
            f'discrepancy was exp(+2 i psi), psi = {psi:.6f})')


def test_s5_gouy_compensator_api_is_a_deprecated_no_op():
    """The three functions that existed to compensate the S5 sign error."""
    from lumenairy.propagators.gbd import asm_field_to_gbd, gbd_asm_gouy_phase, gbd_field_to_asm
    E = np.ones((4, 4), dtype=np.complex128)
    kw = dict(z=1e-3, wavelength=1e-6, dx=4e-6, waist_factor=1.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert gbd_asm_gouy_phase(kw['z'], kw['wavelength'], kw['dx'],
                                  kw['waist_factor']) == 0.0
        assert np.array_equal(gbd_field_to_asm(E, **kw), E)
        assert np.array_equal(asm_field_to_gbd(E, **kw), E)
    cats = [c.category for c in caught]
    assert sum(issubclass(c, DeprecationWarning) for c in cats) >= 3, (
        f'all three compensators must warn; got {cats}')


# ===========================================================================
# S6 -- the asymptotic saddle ignores the input field's phase
# ===========================================================================

def test_s6_asymptotic_methods_warn_on_a_non_collimated_input():
    """``stationary_phase`` / ``local_quadrature`` solve
    ``grad_v2 OPD = 0``, which the symplectic identity
    ``dOPD/dv2 = -n1 (v1 . ds1/dv2)`` makes the ``v1 = 0`` (collimated)
    launch ray at EVERY pixel -- so they are the wrong expansion for a
    diverging / converging / tilted input, which is exactly what the chart
    sizing (``na_proxy = na_lens + na_input``) is built to cover.

    Pre-fix there was no warning and no docstring caveat.
    """
    N, dx, lam = 48, 4.0e-6, 1.0e-6
    ap = 0.30e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R2 = X ** 2 + Y ** 2
    # A strongly diverging input: a 12 um waist has NA ~ 0.027 >> 1e-3.
    E0 = np.exp(-R2 / (12e-6) ** 2).astype(complex)
    presc = la.make_singlet(6.0e-3, -6.0e-3, 0.7e-3, 'N-BK7', aperture=ap)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        la.apply_real_lens_maslov(
            E0, prescription=presc, wavelength=lam, dx=dx,
            integration_method='stationary_phase', poly_order=4,
            ray_field_samples=10, ray_pupil_samples=10)
    msgs = [str(c.message) for c in caught]
    assert any('saddle of the OPD alone' in m for m in msgs), (
        f'no S6 warning for a diverging input; warnings were {msgs}')
    # ... and NOT for a declared-collimated one.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter('always')
        la.apply_real_lens_maslov(
            E0, prescription=presc, wavelength=lam, dx=dx,
            collimated_input=True,
            integration_method='stationary_phase', poly_order=4,
            ray_field_samples=10, ray_pupil_samples=10)
    assert not any('saddle of the OPD alone' in str(c.message)
                   for c in caught2), (
        'collimated_input=True must silence the S6 warning')


# ===========================================================================
# S7 -- _lens_jax: x64 policy and jit-ability
# ===========================================================================

_S7_LAM = 1.0e-6


def _s7_fixture():
    N, dx = 48, 6.0e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (1.0e-4) ** 2).astype(np.complex128)
    rx = la.make_singlet(6.0e-3, -6.0e-3, 1.0e-3, 'N-BK7', aperture=3.0e-4)
    return E0, dict(prescription=rx, wavelength=_S7_LAM, dx=dx,
                    ray_subsample=2, cheb_order=6)


@pytest.mark.parametrize('fn_name', ['apply_real_lens_traced_jax',
                                     'apply_real_lens_maslov_jax'])
def test_s7_jax_lens_entry_points_are_jittable(fn_name):
    """``jax.jit`` on the DEFAULT (static-geometry) path.

    Pre-fix both entry points took ``float(x_out_grid[...])`` to seed the
    Newton inversion, which raises ``ConcretizationTypeError`` under
    ``jit`` -- so the docstring's "vmap+JIT replaces the [NumPy] pool" was
    unreachable.  MEASURED post-fix: jit compiles and agrees with the eager
    call to 1.30e-12 absolute on a unit-peak field.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    import lumenairy.elements._lens_jax as LJ
    fn = getattr(LJ, fn_name)
    E0, kw = _s7_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        eager = np.asarray(fn(jnp.asarray(E0), **kw))
        jitted = np.asarray(jax.jit(lambda E: fn(E, **kw))(jnp.asarray(E0)))
    d = float(np.max(np.abs(jitted - eager)))
    assert d < 1e-9, (
        f'{fn_name}: jit vs eager max |dE| = {d:.3e} (measured 1.30e-12)')
    assert eager.dtype == np.complex128, (
        f'{fn_name} returned {eager.dtype} for a complex128 input under '
        f'x64 (pre-fix, with x64 OFF, it returned complex64 silently)')


def test_s7_jax_lens_refuses_float32():
    """With ``jax_enable_x64`` disabled the entry points must RAISE.

    Pre-fix a complex128 input came back **complex64**: the
    ``exp(i k0 OPL)`` screen (``k0*OPL ~ 1e4`` rad here) then carried
    ~1e-3 waves of rounding.  MEASURED pre-fix on this fixture: 1.71e-05
    waves rms / 8.17e-05 waves p-v between the float32 and float64 phase
    screens; the audit measured 1.5e-4 ... 1.8e-3 waves rms on a larger
    lens, against 2e-10 ... 3e-9 with x64.

    This test can only run in a fresh process with x64 OFF -- JAX cannot
    turn it back off once enabled -- so it checks the POLICY function
    directly, which is what every entry point calls first.
    """
    pytest.importorskip('jax')
    import jax

    import lumenairy.elements._lens_jax as LJ
    if bool(jax.config.read('jax_enable_x64')):
        # x64 already on (another test enabled it): drive the guard with a
        # stubbed config so the refusal path is still exercised.
        class _Cfg:
            @staticmethod
            def read(_name):
                return False
        real = jax.config
        jax.config = _Cfg()
        try:
            with pytest.raises(RuntimeError, match='jax_enable_x64'):
                LJ._require_jax_x64('apply_real_lens_traced_jax')
        finally:
            jax.config = real
    else:
        with pytest.raises(RuntimeError, match='jax_enable_x64'):
            LJ._require_jax_x64('apply_real_lens_traced_jax')


def test_s7_jax_uses_the_shared_exit_vertex_operator():
    """Both JAX entry points must route through
    ``exit_vertex_transfer_jax`` rather than the hand-written
    ``t = -z / max(|N|, 1e-30)`` copies, which gave a grazing ray
    ``t = -z/1e-30`` (~1e26 m of phantom OPL) where the NumPy copies gave
    ``t = 0`` -- a cross-backend divergence in the same primitive."""
    import inspect

    import lumenairy.elements._lens_jax as LJ
    for fn in (LJ.apply_real_lens_traced_jax, LJ.apply_real_lens_maslov_jax):
        src = inspect.getsource(fn)
        assert 'exit_vertex_transfer_jax(' in src, (
            f'{fn.__name__} does not call the shared exit-vertex operator')
        assert 'eps = 1e-30' not in src, (
            f'{fn.__name__} still carries a hand-written clamped transfer')


def test_s7_jax_traced_matches_the_numpy_exit_vertex_chart():
    """Cross-backend: the JAX traced phase screen must agree with a NumPy
    ray trace carried to the exit vertex by the shared operator.

    Oracle: ``rt.trace(...).at_exit_vertex()`` on the same launch lattice,
    which is the operator WP-A1 verified against the analytic vertex-plane
    OPL ``n_exit * (-sag / N)`` to 0.0e+00 m.
    """
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import exit_vertex_transfer_jax, make_jax_ray_state, trace_jax
    _E0, kw = _s7_fixture()
    rx = kw['prescription']
    pres_no_ap = {k: v for k, v in rx.items() if k != 'aperture_diameter'}
    xs = np.linspace(-1.4e-4, 1.4e-4, 9)
    XS, YS = np.meshgrid(xs, xs, indexing='ij')
    hx, hy = XS.ravel(), YS.ravel()
    state = make_jax_ray_state(
        x=jnp.asarray(hx), y=jnp.asarray(hy), z=jnp.zeros_like(hx),
        L=jnp.zeros_like(hx), M=jnp.zeros_like(hx), N=jnp.ones_like(hx))
    final = exit_vertex_transfer_jax(trace_jax(state, pres_no_ap, _S7_LAM),
                                     1.0)
    bundle = rt.RayBundle(
        x=hx.copy(), y=hy.copy(), z=np.zeros_like(hx),
        L=np.zeros_like(hx), M=np.zeros_like(hx), N=np.ones_like(hx),
        wavelength=_S7_LAM, alive=np.ones(hx.size, bool),
        opd=np.zeros_like(hx))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = rt.trace(bundle, rt.surfaces_from_prescription(pres_no_ap),
                       _S7_LAM).at_exit_vertex()
    alive = np.asarray(ref.alive)
    d_opd = float(np.max(np.abs(np.asarray(final.opd)[alive]
                                - ref.opd[alive])))
    d_x = float(np.max(np.abs(np.asarray(final.x)[alive] - ref.x[alive])))
    # 1e-12 m == 1e-6 waves at lambda = 1 um; the two backends differ only
    # in reduction order (measured < 1e-15 m here).
    assert d_opd < 1e-12, f'OPL differs by {d_opd:.3e} m across backends'
    assert d_x < 1e-12, f'x differs by {d_x:.3e} m across backends'


# ===========================================================================
# WP-A2 hand-off: stop_index validation and the anamorphic aperture check
# ===========================================================================

def _tiny_maslov_kwargs():
    return dict(wavelength=1.0e-6, dx=4.0e-6, collimated_input=True,
                integration_method='quadrature', n_v2=16, poly_order=3,
                ray_field_samples=8, ray_pupil_samples=8)


@pytest.mark.parametrize('stop_index,should_raise', [
    (None, False), (0, False), (-1, False),
    (7, True), (-9, True), (1.5, True), (True, True),
])
def test_a2_stop_index_is_validated_like_apply_real_lens(stop_index,
                                                         should_raise):
    """``apply_real_lens_maslov`` routes ``prescription['stop_index']``
    through ``_lens_real._normalise_stop_index`` (WP-A2), so an
    out-of-range or non-integer value RAISES with the Section 2 prefix
    instead of being ``int()``-ed into the "non-entrance stop" warning
    path (where a negative index read as a mid-train stop and a float
    raised a bare ``TypeError`` from ``int()``).  ``-1`` now means the
    LAST surface, as Python indexing does.
    """
    N = 32
    E = np.ones((N, N), dtype=np.complex128)
    rx = la.make_singlet(6e-3, -6e-3, 0.7e-3, 'N-BK7', aperture=1e-4)
    presc = dict(rx)
    if stop_index is not None:
        presc['stop_index'] = stop_index
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if should_raise:
            with pytest.raises(ValueError,
                               match=r'apply_real_lens_maslov: '
                                     r"prescription\['stop_index'\]"):
                la.apply_real_lens_maslov(E.copy(), prescription=presc,
                                          **_tiny_maslov_kwargs())
        else:
            la.apply_real_lens_maslov(E.copy(), prescription=presc,
                                      **_tiny_maslov_kwargs())


def test_a2_anamorphic_aperture_check_uses_the_smaller_semi_extent():
    """The pre-flight aperture-vs-grid check must see the y axis.

    ``_warn_if_aperture_exceeds_grid`` gained ``N_y`` / ``dy`` (WP-A2) and
    checks the SMALLER semi-extent, because that is the axis that truncates
    first.  Here the x semi-extent is 64*8um/2 = 256 um (big enough for the
    300 um aperture's 150 um semi-diameter) while the y semi-extent is
    64*1um/2 = 32 um (far too small), so the warning must fire.
    """
    N = 64
    dx, dy = 8.0e-6, 1.0e-6
    E = np.ones((N, N), dtype=np.complex128)
    rx = la.make_singlet(6e-3, -6e-3, 0.7e-3, 'N-BK7', aperture=3.0e-4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            la.apply_real_lens_maslov(
                E, prescription=rx, wavelength=1.0e-6, dx=dx, dy=dy,
                collimated_input=True, integration_method='quadrature',
                n_v2=16, poly_order=3, ray_field_samples=8,
                ray_pupil_samples=8)
        except Exception:
            pass    # the propagation itself may fail on this tiny grid
    msgs = [str(c.message) for c in caught]
    assert any('exceed the simulation grid' in m for m in msgs), (
        f'the anamorphic y extent must trigger the aperture warning; '
        f'warnings were {msgs}')


# ===========================================================================
# Fit conditioning: the normal-equations Cholesky is gated on cond(A^T A)
# ===========================================================================

def test_solve_fit_routes_an_ill_conditioned_gram_to_the_min_norm_svd():
    """``_solve_fit`` must be a DETERMINISTIC function of ``(A, RHS)``.

    The v5.21 normal-equations Cholesky is justified by "``A`` is a normalised
    tensor-Chebyshev Vandermonde, well-conditioned and ~1.5x oversampled".
    That does not hold on every chart, and the ``LinAlgError`` fallback ladder
    cannot see the failure: a numerically positive-semidefinite but RANK-
    DEFICIENT Gram factors happily and returns an arbitrary member of the
    solution set.

    Oracle: build a design matrix with an EXACTLY duplicated column, so the
    null space is known analytically (dimension 1) and the minimum-norm
    solution is the unique one that splits the duplicated weight evenly.  Then
    perturb the matrix by 1e-15 relative and require the coefficients to move
    by no more than that: a solver that picks an arbitrary null-space member
    fails, a min-norm SVD does not.
    """
    rng = np.random.default_rng(20260912)
    n, m = 400, 12
    A = rng.standard_normal((n, m))
    A[:, -1] = A[:, 0]                       # exact duplicate -> rank m-1
    coef_true = rng.standard_normal(m)
    RHS = (A @ coef_true).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        c1 = LM._solve_fit(A, RHS)
        c2 = LM._solve_fit(A * (1.0 + 1e-15), RHS)
    assert np.all(np.isfinite(c1)), 'ill-conditioned solve returned non-finite'
    # The fit itself must be right on the training data either way.
    res = float(np.sqrt(np.mean((A @ c1[:, 0] - RHS[:, 0]) ** 2)))
    assert res < 1e-10 * float(np.abs(RHS).max()), (
        f'the solve no longer fits its own data: residual {res:.3e}')
    # ... and REPRODUCIBLE under a float64-noise perturbation.  Measured on a
    # real f = 6 mm / 0.2 mm-aperture chart (rank 65 of 70, cond(A^T A) =
    # 6.18e+18), the pre-fix Cholesky returned coefficients 0.869 WAVES apart
    # for design matrices agreeing to 3.1e-15, which moved the propagated
    # field by relL2 0.45.
    move = float(np.max(np.abs(c1 - c2)) / max(np.max(np.abs(c1)), 1e-300))
    assert move < 1e-6, (
        f'coefficients moved {move:.3e} relative under a 1e-15 perturbation '
        f'of A -- the solve is picking an arbitrary null-space member')
    # Min-norm splits the duplicated column evenly; the Cholesky need not.
    assert abs(c1[0, 0] - c1[-1, 0]) < 1e-9 * max(abs(c1[0, 0]), 1e-300), (
        'the duplicated columns did not get the minimum-norm even split, so '
        'the ill-conditioned branch is not the SVD')


def test_solve_fit_warns_when_the_gram_is_singular_at_float64():
    """A genuinely rank-deficient design matrix must say so."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((200, 8))
    A[:, -1] = A[:, 0]                       # exactly singular Gram
    RHS = rng.standard_normal((200, 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        LM._solve_fit(A, RHS)
    msgs = [str(c.message) for c in caught]
    assert any('RANK-DEFICIENT' in m for m in msgs), (
        f'no rank-deficiency warning; warnings were {msgs}')


def test_solve_fit_keeps_the_cholesky_fast_path_when_well_conditioned():
    """A well-conditioned chart must NOT pay for the SVD, and must not warn."""
    rng = np.random.default_rng(11)
    A = rng.standard_normal((500, 10))       # cond(A^T A) ~ 1e2
    RHS = rng.standard_normal((500, 2))
    G = A.T @ A
    ev = np.linalg.eigvalsh(G)
    assert ev[-1] / ev[0] < LM._GRAM_COND_MAX, 'premise: well conditioned'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        coef = LM._solve_fit(A, RHS)
    assert not [c for c in caught if 'RANK-DEFICIENT' in str(c.message)]
    ref, *_ = np.linalg.lstsq(A, RHS, rcond=None)
    # Both solve the same full-rank system; they agree to the conditioning
    # floor (measured 1.1e-14 here).
    assert float(np.max(np.abs(coef - ref))) < 1e-9 * float(np.abs(ref).max())
