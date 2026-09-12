"""VERIFY-A3: independent re-verification of WP-A3 (audit 2026-09-11).

Written by the VERIFIER, not by the implementer, and deliberately NOT reusing
the library's own ray tracer where an oracle is needed: the exit statistics
below are checked against a self-contained Newton-intersection + vector-Snell
trace written from the equations (:func:`_oracle_trace`), so a defect shared by
``lumenairy.raytrace`` and the element under test cannot hide in both.

Covers:

* **T13 + VERIFY-A3 OI-1** -- ``na_exit`` is gated on the ENTRANCE aperture
  disc while the RETURNED FIELD is masked on the OUTPUT grid, and on a thick
  fast element those two sets differ by 1.7x.  The undersample guard is now
  decided on the larger of the two; ``na_exit`` itself is unmoved.
* **T2 follow-up** -- the multibranch energy tripwire's launched-power
  normaliser, on geometries the implementer's sweep did not use.
* **T12** -- the ``carrier=<ndarray>`` branch against the closed-form tilted
  congruence on a SECOND curvature/tilt pair, in both axes.
* **WP-A2 §5.3** -- what ``stop_index`` normalisation actually buys: the
  equivalence classes, bit for bit.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import _compute_carrier
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch,
)
from lumenairy.glass import get_glass_index

MODULE_GLASSES = {'_A3V': lambda wl: 1.5168}

_WL = 1.0e-6


# ===========================================================================
# the independent oracle: Newton intersection + vector Snell, no lumenairy
# ray tracing anywhere in it (only the glass DISPERSION, which is data)
# ===========================================================================
def _oracle_trace(prescription, wavelength, x0, y0):
    """Trace a collimated bundle to the last surface's VERTEX plane.

    Returns ``(x, y, L, M, alive)`` at ``z = z_vertex(last surface)``.  Conic
    sag ``z = c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2))``; the normal is
    ``(-dz/dx, -dz/dy, 1)`` normalised; refraction is the vector Snell form
    ``t = mu d + (mu cos_i - sqrt(1 - mu^2 (1 - cos_i^2))) n``.
    """
    surfs = prescription['surfaces']
    th = list(prescription.get('thicknesses', []) or [])
    zv, z_acc = [], 0.0
    for i in range(len(surfs)):
        zv.append(z_acc)
        if i < len(th):
            z_acc += float(th[i])

    x = np.asarray(x0, dtype=float).ravel().copy()
    y = np.asarray(y0, dtype=float).ravel().copy()
    z = np.zeros_like(x)
    L = np.zeros_like(x)
    M = np.zeros_like(x)
    Nz = np.ones_like(x)
    alive = np.ones(x.shape, dtype=bool)

    for i, s in enumerate(surfs):
        R = s.get('radius')
        c = 0.0 if (R is None or not np.isfinite(R) or R == 0) else 1.0 / float(R)
        k = float(s.get('conic') or 0.0)
        n1 = get_glass_index(s.get('glass_before', 'air'), wavelength)
        n2 = get_glass_index(s.get('glass_after', 'air'), wavelength)
        semi = (float(s['semi_diameter']) if s.get('semi_diameter')
                else np.inf)

        def sag(r2, _c=c, _k=k):
            if _c == 0.0:
                return np.zeros_like(r2), np.zeros_like(r2)
            q = 1.0 - (1.0 + _k) * _c * _c * r2
            q = np.where(q > 0.0, q, np.nan)
            sq = np.sqrt(q)
            zz = _c * r2 / (1.0 + sq)
            dzz = ((_c * (1.0 + sq)
                    - _c * r2 * (-0.5 * (1.0 + _k) * _c * _c / sq))
                   / (1.0 + sq) ** 2)
            return zz, dzz                      # dzz = d(sag)/d(r^2)

        t = (zv[i] - z) / np.where(np.abs(Nz) > 1e-30, Nz, 1e-30)
        for _ in range(80):
            xi, yi = x + t * L, y + t * M
            r2 = xi * xi + yi * yi
            sg, ds = sag(r2)
            f = (z + t * Nz) - zv[i] - sg
            df = Nz - (2.0 * xi * L + 2.0 * yi * M) * ds
            with np.errstate(invalid='ignore', divide='ignore'):
                step = np.where(np.isfinite(f / df), f / df, 0.0)
            t = t - step
            if np.nanmax(np.abs(step)) < 1e-17:
                break
        xi, yi = x + t * L, y + t * M
        zi = z + t * Nz
        r2 = xi * xi + yi * yi
        sg, ds = sag(r2)
        hit = np.isfinite(sg) & np.isfinite(t) & (np.sqrt(r2) <= semi)
        nx, ny = -2.0 * xi * ds, -2.0 * yi * ds
        nn = np.sqrt(nx * nx + ny * ny + 1.0)
        nx, ny, nz = nx / nn, ny / nn, 1.0 / nn
        cosi = -(L * nx + M * ny + Nz * nz)
        flip = cosi < 0
        nx, ny, nz = (np.where(flip, -nx, nx), np.where(flip, -ny, ny),
                      np.where(flip, -nz, nz))
        cosi = np.abs(cosi)
        mu = n1 / n2
        sin2t = mu * mu * (1.0 - cosi * cosi)
        fac = mu * cosi - np.sqrt(np.clip(1.0 - sin2t, 0.0, None))
        alive = alive & hit & (sin2t <= 1.0)
        x, y, z = (np.where(alive, xi, x), np.where(alive, yi, y),
                   np.where(alive, zi, z))
        L = np.where(alive, mu * L + fac * nx, L)
        M = np.where(alive, mu * M + fac * ny, M)
        Nz = np.where(alive, mu * Nz + fac * nz, Nz)

    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.where(np.abs(Nz) > 1e-30, (zv[-1] - z) / Nz, 0.0)
    return x + L * t, y + M * t, L, M, alive & (np.abs(Nz) > 1e-30)


def _singlet(r1, r2, thick, aperture):
    return {'aperture_diameter': aperture,
            'surfaces': [
                {'radius': r1, 'conic': 0.0, 'glass_before': 'air',
                 'glass_after': '_A3V'},
                {'radius': r2, 'conic': 0.0, 'glass_before': '_A3V',
                 'glass_after': 'air'}],
            'thicknesses': [thick]}


def _oracle_na(presc, dx, sub):
    """(entrance-disc NA, output-disc NA) on the element's OWN launch lattice.

    The element launches an ``n_launch`` square of half-width
    ``0.75 * aperture`` (``_lens_traced.py``: ``launch_radius = 0.5 *
    aperture * 1.50``, ``n_launch = max(8, int(2 lr / (dx sub)))``, forced
    odd), so the oracle uses the same lattice and the comparison is of
    PHYSICS, not of sampling.
    """
    ap = float(presc['aperture_diameter'])
    lr = 0.75 * ap
    n = max(8, int(2 * lr / (dx * sub)))
    if n % 2 == 0:
        n += 1
    xs = np.linspace(-lr, lr, n)
    Xi, Yi = np.meshgrid(xs, xs, indexing='ij')
    xo, yo, L, M, alive = _oracle_trace(presc, _WL, Xi, Yi)
    st = np.hypot(L, M)
    ok = alive & np.isfinite(st)
    a2 = (0.5 * ap) ** 2
    m_in = ok & ((Xi ** 2 + Yi ** 2).ravel() <= a2)
    m_out = ok & (xo * xo + yo * yo <= a2)
    return float(st[m_in].max()), float(st[m_out].max())


# ===========================================================================
# T13 / OI-1 -- the two discs
# ===========================================================================
@pytest.mark.parametrize('r1,r2,thick,ap,dx_fac', [
    (51.68e-3, -51.68e-3, 4e-3, 24e-3, 1.2),     # the WP's own f/5 fixture
    (20e-3, -20e-3, 12e-3, 18e-3, 1.3),          # THICK and fast: the gap
    (30e-3, 60e-3, 3e-3, 20e-3, 1.25),           # a meniscus
])
def test_exit_na_statistics_match_an_independent_snell_trace(
        r1, r2, thick, ap, dx_fac):
    """Both reported NA statistics must be what they say they are.

    ORACLE: :func:`_oracle_trace`, a Newton + vector-Snell tracer written from
    the equations, run on the element's own launch lattice so only the PHYSICS
    is compared.

    BAR: 1 % relative on each statistic.  The two implementations share only
    the glass index; their difference is the conic-intersection iteration
    (both converged to < 1e-17 m of step) plus float64 round-off, measured
    below 2e-3 relative on all three fixtures, while the quantity being
    separated -- entrance-disc vs output-disc -- differs by 4 % to 72 %.  So
    the bar has ~1 decade of gap below and 4x..70x above on the thick fixture.
    """
    presc = _singlet(r1, r2, thick, ap)
    N, sub = 512, 4
    dx = ap / N * dx_fac
    sink = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        apply_real_lens_traced(
            np.ones((N, N), dtype=np.complex128), prescription=presc,
            wavelength=_WL, dx=dx, ray_subsample=sub,
            min_coarse_samples_per_aperture=0, on_undersample='silent',
            _exit_na_out=sink)
    na_in, na_out = _oracle_na(presc, dx, sub)

    assert sink['na_exit'] == pytest.approx(na_in, rel=0.01), (
        f"na_exit {sink['na_exit']:.6f} is not the ENTRANCE-disc statistic "
        f'{na_in:.6f} the code says it is')
    assert sink['na_exit_entrance_disc'] == sink['na_exit']
    assert sink['na_exit_output_disc'] == pytest.approx(na_out, rel=0.01), (
        f"na_exit_output_disc {sink['na_exit_output_disc']:.6f} != oracle "
        f'{na_out:.6f}')
    # the guard is the conservative one, always
    assert sink['na_exit_guard'] == max(sink['na_exit'],
                                        sink['na_exit_output_disc'])
    assert sink['na_exit_guard'] >= sink['na_exit'] - 1e-15


def test_the_undersample_guard_is_not_understated_on_a_thick_fast_element():
    """FAIL-BEFORE for VERIFY-A3 OI-1.

    T13 intersected the significance mask with the ENTRANCE aperture disc,
    which removed the audit's 3.14x OVERstatement -- but the mask the returned
    field carries is applied on the OUTPUT grid, and on a thick fast element a
    ray entering outside ``aperture/2`` still lands inside it.  Measured with
    the oracle above on R = +-20 mm / t = 12 mm / ap = 18 mm at
    lambda = 1 um: entrance disc 0.4981, output disc 0.8580 -- so the guard's
    advice ``dx <= lambda/(2 NA)`` read 1.00 um where 0.58 um is needed, i.e.
    1.72x too coarse, in the UNSAFE direction for a Nyquist guard.

    The arm that would fail on the pre-fix code is the ``dx_need`` the WARNING
    quotes: pre-fix it was ``lambda/(2 * na_exit)``, which on this fixture is
    the entrance-disc number.  Both the ratio and the quoted micron figure are
    derived from the run's own measurements, not recorded.
    """
    presc = _singlet(20e-3, -20e-3, 12e-3, 18e-3)
    N, sub = 512, 4
    dx = 18e-3 / N * 1.3
    sink = {}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        apply_real_lens_traced(
            np.ones((N, N), dtype=np.complex128), prescription=presc,
            wavelength=_WL, dx=dx, ray_subsample=sub,
            min_coarse_samples_per_aperture=0, _exit_na_out=sink)
    na_in, na_out = _oracle_na(presc, dx, sub)

    # the fixture really does separate the two discs (else the test is vacuous)
    assert na_out / na_in > 1.5, (
        f'fixture no longer separates the discs: {na_out:.4f} / {na_in:.4f}')
    msgs = [str(w.message) for w in rec if 'NA_exit=' in str(w.message)]
    assert msgs, 'the undersample guard did not fire on an NA 0.86 exit beam'
    dx_need_guard = _WL / (2.0 * sink['na_exit_guard'])
    dx_need_prefix = _WL / (2.0 * sink['na_exit'])
    assert dx_need_guard < dx_need_prefix / 1.5, (
        'the guard is still priced on the entrance-disc NA: '
        f'{dx_need_guard*1e6:.2f} um vs {dx_need_prefix*1e6:.2f} um')
    assert f'{dx_need_guard*1e6:.2f} um' in msgs[0], (
        f'the message does not quote the conservative dx: {msgs[0]}')
    # ...and the value the traced-carrier chain reads is NOT moved by this
    # (its thresholds are calibrated against the entrance-disc statistic).
    assert sink['na_exit'] == pytest.approx(na_in, rel=0.01)


# ===========================================================================
# T2 follow-up -- the tripwire's launched-power normaliser
# ===========================================================================
def _d3_singlet():
    """The delta-audit's D3 air-focus singlet: a 6 mm aperture whose grid
    (48 x 25 um = 1.2 mm) holds 1/25 of its area."""
    return {'aperture_diameter': 6e-3, 'surfaces': [
        {'radius': 25e-3, 'conic': 0., 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 3e-3},
        {'radius': -25e-3, 'conic': 0., 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 3e-3}],
        'thicknesses': [3e-3, 40e-3]}


def _aperture_normaliser(E, dx, aperture):
    """The SHIPPED-BEFORE denominator, recomputed here so both are measured in
    the same run rather than quoted from the report."""
    N = E.shape[0]
    xg = (np.arange(N) - N / 2.0) * dx
    r2 = xg[None, :] ** 2 + xg[:, None] ** 2
    ar = 0.5 * float(aperture)
    return float(np.sum(np.abs(E[r2 <= ar * ar]) ** 2))


def test_the_two_normalisers_agree_when_the_grid_holds_the_aperture():
    """Where the launch sampler does not clamp and nothing leaves the grid,
    ``P_out/P_launched-onto-grid`` and ``P_out/P_aperture`` are the same
    physical ratio, so the new normaliser cannot be hiding a systematic shift.

    BAR: 1e-3 relative.  The two differ only by the launch lattice's
    quadrature of the same power (a bilinear resample of ``E_in`` onto an odd
    lattice of pitch ``sub*dx``), measured 6e-5 here; the pathological
    geometry below separates them by 130 %.  Three decades of gap below, three
    above.
    """
    presc = _d3_singlet()
    N, dx = 256, 25e-6                      # 6.4 mm grid vs a 6 mm aperture
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    for z in (38e-3, 45e-3):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F, D = apply_real_lens_traced_multibranch(
                E, prescription=presc, wavelength=0.633e-6, dx=dx,
                output_plane_distance=z, return_diagnostics=True)
        old = float(np.sum(np.abs(F) ** 2)) / _aperture_normaliser(
            E, dx, presc['aperture_diameter'])
        assert D['power_ratio'] == pytest.approx(old, rel=1e-3), (
            f'z={z*1e3:g} mm: new {D["power_ratio"]:.6f} vs old {old:.6f}')


def test_the_new_normaliser_separates_geometry_from_a_real_blow_up():
    """Two-sided, on the geometry the old denominator could not read.

    (a) aperture 25x the grid area, an ORDINARY through-focus plane: the old
        denominator counts only the aperture circle ON the E_in grid while the
        launch sampler clamps E_in at the grid edge and launches that
        amplitude over the whole rim annulus, so it reads ~2.3x with the
        energy conserved to 1 %.  The new one must stay inside [0.5, 2.0] and
        say nothing.
    (b) the same lens at a plane where the branch sum really does blow up:
        both read far above the bar, so the correction costs no detection.

    DECISION test: (a) asserts silence and a ratio in band; (b) scans toward
    the focus until the reconstructed power leaves the band and requires the
    ENERGY arm (not merely some warning) to have fired.
    """
    presc = _d3_singlet()
    N, dx = 48, 25e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    p_ap = _aperture_normaliser(E, dx, presc['aperture_diameter'])

    # (a) the geometry artefact
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        F, D = apply_real_lens_traced_multibranch(
            E, prescription=presc, wavelength=0.633e-6, dx=dx,
            output_plane_distance=38e-3, return_diagnostics=True)
    old = float(np.sum(np.abs(F) ** 2)) / p_ap
    assert old > 2.0, (
        f'the fixture no longer exercises the artefact (old ratio {old:.3f}); '
        f'the old 10x bar and the new 2x bar both need it above 2')
    assert 0.5 < D['power_ratio'] < 2.0, (
        f'the launched-power ratio left the band: {D["power_ratio"]:.4f}')
    assert not [w for w in rec if 'multibranch' in str(w.message)], (
        'the tripwire fired on a geometry artefact: '
        + '; '.join(str(w.message)[:80] for w in rec))

    # (b) a real blow-up on the SAME fixture, found by scanning FRACTIONS of
    # the BFL this prescription actually has (measured by the oracle, never a
    # constant): a paraxial ray's vertex-plane crossing, bfl = -x/L.
    xo, _yo, L0, _M0, _al = _oracle_trace(
        presc, 0.633e-6, np.array([5e-5]), np.array([0.0]))
    bfl = float(-xo[0] / L0[0])
    assert 20e-3 < bfl < 30e-3, f'BFL {bfl} m outside the expected range'
    fired = False
    for frac in (0.95, 0.97, 0.98, 0.99, 0.995):
        z = frac * bfl
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            try:
                _F, D = apply_real_lens_traced_multibranch(
                    E, prescription=presc, wavelength=0.633e-6, dx=dx,
                    output_plane_distance=z, return_diagnostics=True)
            except RuntimeError:
                continue                     # the total-collapse refusal
        if D['power_ratio'] > 2.0:
            assert [w for w in rec
                    if 'reconstructed grid power is' in str(w.message)], (
                f'P/P_launched = {D["power_ratio"]:.4g} at z = {z*1e3:.3f} mm '
                f'with no ENERGY warning')
            fired = True
            break
    assert fired, 'the scan never reached a blow-up; widen the ladder'


def test_the_gain_arm_is_bracketed_against_boundary_straddling_triangles():
    """FAIL-BEFORE for VERIFY-A3 OI-2.

    ``p_in`` counts a launch node only when its OWN mapped point lands on the
    grid; ``p_out`` counts every pixel a triangle covers, including triangles
    that STRADDLE the grid boundary with their nodes outside.  On a coarse
    launch lattice over a grid much smaller than the beam that mismatch alone
    produced ratios of 2.07..3.26 with ``n_branch = 1`` and ZERO degenerate
    triangles -- no coalescence anywhere, three spurious RuntimeWarnings, on
    the very geometry the launched-power normaliser was introduced to quieten
    (the D3 fixture: a 6 mm aperture on a 1.2 mm grid).

    The gain arm is now bracketed by a second denominator that counts the
    WHOLE launch power of every contributing triangle, so a gain must clear
    BOTH.  This test asserts, at each plane, that the node-based ratio really
    is above the bar (so the arm would have fired pre-fix), that the
    triangle-based one is not, and that no energy warning is emitted.

    No detection is lost: the same fixture's true blow-up measures 1.8e+05 on
    BOTH denominators (the other test above), i.e. five decades outside.
    """
    presc = _d3_singlet()
    N, dx = 48, 25e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    seen = 0
    for sub, z in ((8, 100e-3), (8, 110e-3), (8, 120e-3), (4, 200e-3)):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            _F, D = apply_real_lens_traced_multibranch(
                E, prescription=presc, wavelength=0.633e-6, dx=dx,
                output_plane_distance=z, ray_subsample=sub,
                return_diagnostics=True)
        if D['power_ratio'] <= 2.0:
            continue                      # this plane no longer straddles
        seen += 1
        assert int(D['n_branch'].max()) == 1, (
            'the fixture has developed real multi-branch coalescence at '
            f'z = {z*1e3:g} mm, so it no longer isolates the artefact')
        assert D['n_triangles_degenerate'] == 0
        assert D['power_ratio_triangles'] <= 2.0, (
            f'sub={sub} z={z*1e3:g} mm: the bracketing denominator also reads '
            f'a gain ({D["power_ratio_triangles"]:.4f}), so this is not the '
            f'straddle artefact')
        assert not [w for w in rec
                    if 'reconstructed grid power is' in str(w.message)], (
            f'sub={sub} z={z*1e3:g} mm: spurious energy warning at '
            f'node-ratio {D["power_ratio"]:.4f} / triangle-ratio '
            f'{D["power_ratio_triangles"]:.4f} with one branch and no '
            f'degenerate triangles')
    assert seen >= 3, (
        f'only {seen} of 4 planes still exercise the straddle artefact; '
        f'the fixture no longer demonstrates it')


# ===========================================================================
# S9 -- the uniform dark fill, measured on the FIELD rather than on Ai(x)
# ===========================================================================
def test_the_airy_annulus_does_not_move_the_returned_field():
    """S9 restricts the CFU dark-side fill to ``r_c < r < r_c + 20 l_airy``.

    The shipped regression test for that argues from ``Ai(x)`` and from the
    value of the constant; this one measures the thing the caller sees, by
    running the SAME call with the annulus widened past the grid (which is the
    pre-fix "every pixel outside r_c" behaviour) and differencing the fields.

    BAR: 1e-12 of the peak.  The Airy tail at 20 ``l_airy`` is
    ``Ai(20) ~ 4e-16`` in amplitude and the measured difference is 1.1e-26 of
    the peak -- fourteen decades below the bar -- while removing the fill
    ALTOGETHER (which is the failure this guards against) would leave the
    multibranch's exact zero on the whole dark side, i.e. an O(1) change over
    the ring.  Fixture: the ``caustic_fold_ref`` plano-convex at its own
    reference plane, where the completion is proved to have ENGAGED.
    """
    import lumenairy.elements._lens_traced_uniform as U
    presc = {'wavelength': 1.31e-6, 'aperture_diameter': 1.4e-3, 'surfaces': [
        {'radius': 2.7e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': '_A3V', 'semi_diameter': 0.75e-3},
        {'radius': float('inf'), 'conic': 0.0, 'glass_before': '_A3V',
         'glass_after': 'air', 'semi_diameter': 0.75e-3}],
        'thicknesses': [1.0e-3], 'stop_index': 0}
    N, dx = 256, 3e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / 0.55e-3 ** 2).astype(np.complex128)

    def _run():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return U.apply_real_lens_traced_uniform(
                E, prescription=presc, wavelength=1.31e-6, dx=dx,
                output_plane_distance=4.3704e-3, ray_subsample=2,
                return_diagnostics=True)

    orig = U._AIRY_TAIL_CELLS
    try:
        A, dA = _run()
        U._AIRY_TAIL_CELLS = 1e9          # the pre-fix "everything outside r_c"
        B, _dB = _run()
    finally:
        U._AIRY_TAIL_CELLS = orig
    assert not dA.get('fell_back', True), (
        f'the uniform completion did not engage ({dA.get("reason")!r}), so '
        f'this fixture cannot see the dark fill at all')
    peak = float(np.abs(B).max())
    dev = float(np.abs(A - B).max()) / peak
    assert dev < 1e-12, (
        f'restricting the dark fill to {orig} Airy lengths moved the field by '
        f'{dev:.3e} of its peak')
    assert 10.0 <= float(orig) <= 50.0, (
        f'_AIRY_TAIL_CELLS = {orig}: below ~15 the physical tail is clipped, '
        f'above ~50 the Airy argument cap takes over anyway')


# ===========================================================================
# T12 -- the ndarray carrier on a SECOND curvature/tilt pair, both axes
# ===========================================================================
@pytest.mark.parametrize('R,Lc,Mc', [
    (+50e-3, 0.120, 0.080),          # positive radius, a 2-D tilt
    (-12e-3, 0.300, -0.150),         # strong curvature, opposite y tilt
])
def test_ndarray_carrier_is_discretisation_limited_in_both_axes(R, Lc, Mc):
    """``TiltedCarrier`` is closed form (exact); the ndarray branch is
    ``np.gradient`` (central difference) + a bilinear resample, so its error is

        (dx^2/6)|g''|   +   (dx^2/8)|g''|   =   (7/24) dx^2 |g''|

    with ``g_x = sign(R) u_x / s``, ``u = (x,y) + R (L,M)/N``,
    ``s = |(u, R)|`` and ``d^2 g_x/dx^2 = 3 u_x (R^2 + u_y^2) / s^5``.

    The band is checked on a 4x refinement ladder, in BOTH axes, on fixtures
    the WP's own D1 pin does not use (it is a single R = -30 mm, M = 0 case).
    Measured ratio to the derived bar 0.90..1.12 over the ladder; a return to
    the pre-T12 nearest-neighbour lookup is LINEAR in dx and overshoots by
    ~4 decades at the first rung.
    """
    import lumenairy as la
    spec = la.TiltedCarrier(R, Lc, Mc)
    Nn = np.sqrt(1.0 - Lc * Lc - Mc * Mc)
    sg = np.sign(R)

    def exact(xq, yq):
        ux, uy = xq + R * Lc / Nn, yq + R * Mc / Nn
        s = np.sqrt(ux * ux + uy * uy + R * R)
        return sg * ux / s, sg * uy / s

    def bar(xq, yq, h):
        bx, by = [], []
        for ox in (-h, 0.0, h):
            for oy in (-h, 0.0, h):
                ux, uy = xq + ox + R * Lc / Nn, yq + oy + R * Mc / Nn
                s = np.sqrt(ux * ux + uy * uy + R * R)
                bx.append(np.abs(3.0 * ux * (R * R + uy * uy) / s ** 5))
                by.append(np.abs(3.0 * uy * (R * R + ux * ux) / s ** 5))
        f = (7.0 / 24.0) * h * h
        return f * np.max(bx, axis=0), f * np.max(by, axis=0)

    for k in (1, 2, 4):
        n, h = 96 * k, 20e-6 / k
        ax = (np.arange(n) - n / 2) * h
        Y, X = np.meshgrid(ax, ax, indexing='ij')
        W, grad_a, _ = _compute_carrier(spec, None, 1.31e-6, h, X, Y)
        _, grad_n, _ = _compute_carrier(np.asarray(W), None, 1.31e-6, h, X, Y)
        xq = np.array([0.37 * h, 4.5 * h, -2.3 * h])   # off-lattice on purpose
        yq = np.array([0.11 * h, -3.7 * h, 6.6 * h])
        ex, ey = exact(xq, yq)
        ax_, ay_ = grad_a(xq, yq)
        nx_, ny_ = grad_n(xq, yq)
        bx, by = bar(xq, yq, h)
        # closed form: exact at every rung, both axes
        np.testing.assert_allclose(ax_, ex, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(ay_, ey, rtol=1e-12, atol=1e-15)
        # sampled: inside the derived band, from both sides
        assert (np.abs(nx_ - ex) <= 1.5 * bx).all(), (
            f'dx={h:g}: x error {np.abs(nx_ - ex)} above the derived '
            f'O(dx^2) bar {bx}')
        assert (np.abs(ny_ - ey) <= 1.5 * by).all(), (
            f'dx={h:g}: y error {np.abs(ny_ - ey)} above {by}')
        assert (np.abs(nx_ - ex) >= 0.3 * bx).all(), (
            'the ndarray branch has become exact -- it is no longer the '
            'branch this test measures')


# ===========================================================================
# WP-A2 §5.3 -- the stop_index equivalence classes, bit for bit
# ===========================================================================
def test_stop_index_spellings_collapse_into_their_equivalence_classes():
    """Normalisation is worth something only if equivalent spellings give the
    SAME field and inequivalent ones do not.

    On a 2-surface prescription: ``None`` == ``0`` == ``-2`` (the entrance)
    and ``1`` == ``-1`` (the rear), bit for bit; the two classes must DIFFER
    (else the test would pass on a build that ignored the key altogether --
    which is what the pre-fix traced entry effectively did, since it only ever
    compared ``int(stop_index) != 0``).
    """
    presc = _singlet(50e-3, -50e-3, 3e-3, 2.0e-3)
    N, dx = 96, 25e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / (0.5e-3) ** 2).astype(np.complex128)

    def run(stop):
        p = dict(presc)
        if stop is not None:
            p['stop_index'] = stop
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return apply_real_lens_traced(
                E, prescription=p, wavelength=_WL, dx=dx, ray_subsample=4,
                on_undersample='silent')

    entrance = [run(s) for s in (None, 0, -2, np.int64(0))]
    rear = [run(s) for s in (1, -1, np.int32(1))]
    for f in entrance[1:]:
        assert np.array_equal(entrance[0], f)
    for f in rear[1:]:
        assert np.array_equal(rear[0], f)
    assert not np.array_equal(entrance[0], rear[0]), (
        'the two stop positions return the same field, so this prescription '
        'cannot distinguish them -- the test is vacuous')


# ===========================================================================
# VERIFY-A3 follow-up (coordinator rulings 1-6)
# ===========================================================================
def test_the_exit_na_guard_is_priced_on_the_in_medium_numerical_aperture():
    """OI-3.  The exit leg runs in the medium AFTER the last surface, whose
    in-medium wavelength is ``lambda/n_exit``, so the Nyquist requirement is
    ``dx <= lambda/(2 n_exit sin theta)``.  The guard used the bare direction
    cosine, i.e. it told an immersed design it had ``n_exit`` times more room
    than it has.

    TWO-SIDED and derived: the immersed arm requires the advised ``dx`` to
    shrink by EXACTLY ``n_exit`` (rtol 1e-9 -- it is one multiply, so the only
    error is float64 round-off, ~1e-16, against a 1.76x effect: fifteen
    decades of gap); the air arm requires it to be bit-identical to the bare
    statistic, which is the no-op the fix must preserve on every air-ending
    prescription in the tree.
    """
    for last, expect_n1 in (('air', True), ('N-SF11', False), ('N-BK7', False)):
        presc = {'aperture_diameter': 16e-3, 'surfaces': [
            {'radius': 30e-3, 'conic': 0.0, 'glass_before': 'air',
             'glass_after': '_A3V'},
            {'radius': -30e-3, 'conic': 0.0, 'glass_before': '_A3V',
             'glass_after': last}], 'thicknesses': [4e-3]}
        N, dx = 512, 16e-3 / 512 * 1.3
        sink = {}
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            apply_real_lens_traced(
                np.ones((N, N), dtype=np.complex128), prescription=presc,
                wavelength=_WL, dx=dx, ray_subsample=4,
                min_coarse_samples_per_aperture=0, _exit_na_out=sink)
        n_exit = sink['n_exit']
        bare = max(sink['na_exit_entrance_disc'], sink['na_exit_output_disc'])
        assert sink['na_exit_guard'] == pytest.approx(n_exit * bare, rel=1e-9)
        if expect_n1:
            assert n_exit == 1.0
            assert sink['na_exit_guard'] == bare       # bit-identical no-op
        else:
            assert n_exit > 1.4, n_exit
            assert sink['na_exit_guard'] / bare == pytest.approx(n_exit,
                                                                 rel=1e-9)
            msgs = [str(w.message) for w in rec if 'NA_exit=' in str(w.message)]
            assert msgs, 'the guard did not fire on an immersed rear'
            assert 'medium after the last surface' in msgs[0], msgs[0]
        # the bare statistics themselves are unmoved: they are what the
        # traced-carrier chain reads
        assert sink['na_exit'] == sink['na_exit_entrance_disc']


@pytest.mark.parametrize('dx,dy', [(20e-6, 60e-6), (25e-6, 10e-6)])
def test_the_ndarray_carrier_uses_its_own_pitch_on_each_axis(dx, dy):
    """OI-10.  ``_compute_carrier``'s ndarray branch differentiated with
    ``np.gradient(W, dx, dx)`` and indexed with ``/dx`` on both axes.
    ``apply_real_lens_traced`` refuses a non-square grid, but
    ``apply_real_lens`` does not and passes ``conjugate=<ndarray>`` straight
    through, so the y gradient was wrong by ``dy/dx``.

    ORACLE: the closed-form tilted congruence.  BAR: the same
    ``(7/24) dx^2 max|g''|`` discretisation band the square case obeys, on
    each axis with ITS OWN pitch.  Two-sided, and the arm that would fail
    before the fix is the y one: measured at dy = 3 dx the old call reads
    1.051e-01 against a 1.198e-07 bar (6 decades).
    """
    import lumenairy as la
    from lumenairy.elements._lens_traced import _compute_carrier as CC
    R, Lc, Mc = -30e-3, 0.046, 0.031
    spec = la.TiltedCarrier(R, Lc, Mc)
    Nn = np.sqrt(1.0 - Lc * Lc - Mc * Mc)
    sg = np.sign(R)
    Ny, Nx = 128, 128
    xax = (np.arange(Nx) - Nx / 2) * dx
    yax = (np.arange(Ny) - Ny / 2) * dy
    Xg, Yg = np.meshgrid(xax, yax)
    W, grad_a, _ = CC(spec, None, 1.31e-6, dx, Xg, Yg, dy=dy)
    _, grad_n, _ = CC(np.asarray(W), None, 1.31e-6, dx, Xg, Yg, dy=dy)
    _, grad_sq, _ = CC(np.asarray(W), None, 1.31e-6, dx, Xg, Yg)  # dy := dx

    xq = np.array([0.37 * dx, 3.5 * dx])
    yq = np.array([0.11 * dy, -2.7 * dy])
    ux, uy = xq + R * Lc / Nn, yq + R * Mc / Nn
    s = np.sqrt(ux * ux + uy * uy + R * R)
    ex, ey = sg * ux / s, sg * uy / s

    # DERIVED BAR.  ``g_x = sign(R) u_x / s`` with ``s = |(u, R)|`` gives
    # ``d2g_x/dx2 = -3 u_x (u_y^2 + R^2) / s^5`` and
    # ``d2g_x/dy2 = -u_x (s^2 - 3 u_y^2) / s^5`` (and the mirror pair for
    # ``g_y``).  The ndarray branch makes two discretisations: a central
    # difference along its OWN axis, ``(h^2/6)|d2g/dh2|``, and a BILINEAR
    # resample of that derivative field, which errs by ``(h^2/8)|d2g/dh2|``
    # in each direction separately.  So
    #     bar_x = (7/24) dx^2 |d2g_x/dx2| + (1/8) dy^2 |d2g_x/dy2|
    # and likewise for y.  The cross term is what makes this a dy test: at
    # dy = 3 dx it is the LARGER half.  Measured ratio to the bar 0.855..0.953
    # over dy/dx = 3, 0.4 and 1 (VERIFY-A3, 2026-09-12), so 1.5x leaves ~1.6x
    # of headroom above and the dy := dx arm below overshoots by 30x.
    def _second_derivs(ox, oy):
        ax_, ay_ = ux + ox, uy + oy
        ss = np.sqrt(ax_ * ax_ + ay_ * ay_ + R * R)
        return (np.abs(3.0 * ax_ * (ay_ * ay_ + R * R) / ss ** 5),
                np.abs(ax_ * (ss * ss - 3.0 * ay_ * ay_) / ss ** 5),
                np.abs(3.0 * ay_ * (ax_ * ax_ + R * R) / ss ** 5),
                np.abs(ay_ * (ss * ss - 3.0 * ax_ * ax_) / ss ** 5))

    _nb = [_second_derivs(ox, oy)
           for ox in (-dx, 0.0, dx) for oy in (-dy, 0.0, dy)]
    gxx = np.max([t[0] for t in _nb], axis=0)
    gxy = np.max([t[1] for t in _nb], axis=0)
    gyy = np.max([t[2] for t in _nb], axis=0)
    gyx = np.max([t[3] for t in _nb], axis=0)
    barx = ((7.0 / 24.0) * dx * dx * gxx + 0.125 * dy * dy * gxy).max()
    bary = ((7.0 / 24.0) * dy * dy * gyy + 0.125 * dx * dx * gyx).max()

    ax_, ay_ = grad_a(xq, yq)
    np.testing.assert_allclose(ax_, ex, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(ay_, ey, rtol=1e-12, atol=1e-15)
    nx_, ny_ = grad_n(xq, yq)
    assert np.abs(nx_ - ex).max() <= 1.5 * barx
    assert np.abs(ny_ - ey).max() <= 1.5 * bary, (
        f'y gradient {np.abs(ny_ - ey).max():.3e} above the derived bar '
        f'{bary:.3e} -- the branch is not using dy')
    # FAIL-BEFORE: the dy := dx call (what every caller got) is decades out
    sx_, sy_ = grad_sq(xq, yq)
    assert np.abs(sy_ - ey).max() > 30.0 * bary, (
        'the dy := dx call is no longer distinguishable, so this fixture '
        'cannot demonstrate the defect')


def test_the_ndarray_carrier_is_bit_identical_on_a_square_grid():
    """OI-10 must be a NO-OP where ``dy is None`` or ``dy == dx``: that is
    every path through ``apply_real_lens_traced`` and every square-grid
    ``apply_real_lens`` call, i.e. essentially the whole tree."""
    import lumenairy as la
    from lumenairy.elements._lens_traced import _compute_carrier as CC
    spec = la.TiltedCarrier(-30e-3, 0.046, 0.031)
    n, h = 96, 20e-6
    ax = (np.arange(n) - n / 2) * h
    Yg, Xg = np.meshgrid(ax, ax, indexing='ij')
    W, _, _ = CC(spec, None, 1.31e-6, h, Xg, Yg)
    q = np.array([0.37 * h, 4.5 * h, -2.3 * h])
    out = []
    for kw in ({}, {'dy': None}, {'dy': h}):
        _, g, w = CC(np.asarray(W), None, 1.31e-6, h, Xg, Yg, **kw)
        out.append((g(q, q), w(q, q)))
    for (g1, w1) in out[1:]:
        assert np.array_equal(g1[0], out[0][0][0])
        assert np.array_equal(g1[1], out[0][0][1])
        assert np.array_equal(w1, out[0][1])


def test_the_local_tilt_estimator_says_when_it_is_at_its_sampling_limit():
    """OI-5.  ``_sample_local_tilts`` reads a WRAPPED phase difference, so it
    cannot return a direction cosine above ``lambda/(2 dx)`` whatever the
    field does -- a steeper tilt folds in and returns as a plausible small
    number.  The audit's own repro prints the case and calls it "silently
    WRONG, no warning": a 0.8 launch tilt at dx = 4 um, lambda = 1.31 um,
    ``max_sin=0.5`` returns ``max|L| = 0.1450`` against a 0.16375 Nyquist,
    nowhere near the 0.5 clip.

    TWO-SIDED: it must fire on that case and on a genuine clip
    (``max_sin`` below the Nyquist), and stay SILENT on ordinary well-sampled
    tilts.  The bar is 0.8 of the Nyquist fold -- measured 0.885 for the
    aliased case against 0.183 (tilt 0.03) and 0.611 (tilt 0.10) for the
    silent ones, so there is 1.4x of gap above and 1.3x below.
    """
    from lumenairy.elements._lens_traced import _sample_local_tilts
    lam, dx, n = 1.31e-6, 4e-6, 256
    k0 = 2.0 * np.pi / lam
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    sin_nyq = lam / (2.0 * dx)

    def run(tilt, max_sin, sigma):
        E = (np.exp(1j * k0 * tilt * X)
             * np.exp(-(X ** 2 + Y ** 2) / (0.2e-3) ** 2)).astype(complex)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            L, _M = _sample_local_tilts(E, lam, dx, X, Y, max_sin=max_sin,
                                        smooth_sigma_px=sigma)
        return L, [str(w.message) for w in rec
                   if 'local-tilt estimator' in str(w.message)]

    # (a) the audit's aliased case: max_sin beyond the fold, reading at it
    L, msgs = run(0.8, 0.5, 0.0)
    assert msgs, 'the aliased 0.8 tilt still returns silently'
    assert f'{sin_nyq:.5f}' in msgs[0], msgs[0]
    assert float(np.abs(L).max()) < sin_nyq          # it DID fold
    # (b) a genuine clip
    _L2, msgs2 = run(0.8, 0.10, 0.0)
    assert msgs2 and 'REPLACED by it' in msgs2[0], msgs2
    # (c) silent where the estimator is within its means
    for tilt, sigma in ((0.03, 0.0), (0.10, 0.0), (0.03, 4.0), (0.0, 4.0)):
        _L3, msgs3 = run(tilt, 0.5, sigma)
        assert not msgs3, (tilt, sigma, msgs3)


def test_a_vignetting_semi_diameter_reaches_the_default_polynomial_answer():
    """OI-11.  The polynomial forward-map fit drops dead launch samples and
    is then SMOOTH across the hole, so a per-surface ``semi_diameter`` that
    vignettes the ray trace used to leave the returned field untouched:
    measured ``P/P_in = 0.9983`` with 21 821 non-zero pixels on an N-SF11
    singlet whose rear semi-diameter is 1.4 mm inside a 5 mm aperture --
    identical to the same call with no ``semi_diameter`` at all, while the
    spline path (which rejects its filled nodes) read 0.8657 / 6 637.

    Output pixels whose converged entrance solution lands on a dead launch
    node are now masked on BOTH fits.  Two-sided: the vignetted call must
    lose a decisive fraction of its power AND the un-vignetted call must be
    bit-identical (the mask is ``None`` there, so the arithmetic is
    untouched).
    """
    presc_v = {'aperture_diameter': 5e-3, 'surfaces': [
        {'radius': 70e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': '_A3V'},
        {'radius': -70e-3, 'conic': 0.0, 'glass_before': '_A3V',
         'glass_after': 'air', 'semi_diameter': 1.4e-3}],
        'thicknesses': [3e-3]}
    presc_0 = {'aperture_diameter': 5e-3, 'surfaces': [
        {'radius': 70e-3, 'conic': 0.0, 'glass_before': 'air',
         'glass_after': '_A3V'},
        {'radius': -70e-3, 'conic': 0.0, 'glass_before': '_A3V',
         'glass_after': 'air'}], 'thicknesses': [3e-3]}
    N, dx = 192, 30e-6
    ax = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    E = np.exp(-(X ** 2 + Y ** 2) / (1.4e-3) ** 2).astype(np.complex128)
    kw = dict(wavelength=1.064e-6, dx=dx, ray_subsample=4,
              on_aperture_beam='silent', on_noncollimated='off',
              min_coarse_samples_per_aperture=0)
    p_in = float((np.abs(E) ** 2).sum())

    def run(presc, fit):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = apply_real_lens_traced(E, prescription=presc,
                                         newton_fit=fit, **kw)
        return out, [str(w.message) for w in rec
                     if 'vignetted or lost' in str(w.message)]

    poly_v, msg_v = run(presc_v, 'polynomial')
    spline_v, _ = run(presc_v, 'spline')
    poly_0, msg_0 = run(presc_0, 'polynomial')

    r_v = float((np.abs(poly_v) ** 2).sum()) / p_in
    r_s = float((np.abs(spline_v) ** 2).sum()) / p_in
    r_0 = float((np.abs(poly_0) ** 2).sum()) / p_in
    # the un-vignetted call is untouched: the transmitted power is the
    # aperture-limited ~1.0 the screen model always returns
    assert r_0 == pytest.approx(0.9983, abs=2e-3), r_0
    assert not msg_0, msg_0
    # the vignetted call LOSES power, and lands with the spline rather than
    # with the un-vignetted run.  BAR: within 10 % of the spline's reading
    # (the residual is the spline's extra filled-node ring), and at least
    # 5 % below the un-vignetted one -- measured 0.8840 vs 0.8657 vs 0.9983.
    assert abs(r_v - r_s) / r_s < 0.10, (r_v, r_s)
    assert r_v < 0.95 * r_0, (r_v, r_0)
    assert msg_v and 'INSIDE the' in msg_v[0], msg_v
    # ...and the notice is about pixels, not about the launch square's
    # corners: a semi_diameter AT the aperture rim vignettes nothing that the
    # output mask keeps, so it must stay silent.
    presc_rim = {**presc_v, 'surfaces': [presc_v['surfaces'][0],
                                         {**presc_v['surfaces'][1],
                                          'semi_diameter': 2.5e-3}]}
    poly_rim, msg_rim = run(presc_rim, 'polynomial')
    assert not msg_rim, msg_rim
    # The FIELD is not bit-identical to the no-semi_diameter run -- it never
    # was, because a semi_diameter kills the launch square's corners and so
    # changes which samples the fit sees -- but nothing the output mask KEEPS
    # is lost: the transmitted power and the non-zero pixel count are
    # unchanged (measured 0.998312 vs 0.998312, 21 821 vs 21 821).
    r_rim = float((np.abs(poly_rim) ** 2).sum()) / p_in
    assert r_rim == pytest.approx(r_0, rel=1e-4), (r_rim, r_0)
    assert int((poly_rim != 0).sum()) == int((poly_0 != 0).sum())


def test_spectral_gap_cuts_only_counts_peaks_inside_the_occupied_band():
    """OI-4.  WP-A3 changed ``_spectral_gap_cuts`` (it did not merely correct
    the docstring): the flanking-peak test is now restricted to the
    0.995-power support, so spectral leakage OUTSIDE that band can no longer
    justify a cut.  That is a behaviour change on
    ``apply_real_lens_traced_segmented`` and it needs a pin.

    Synthetic marginal, so nothing depends on a build: one in-band lobe at
    f = +20 and an out-of-band lobe at f = -80, with the support declared as
    [-40, +40].  The pre-fix predicate (re-created here) cuts at the band
    edge on the strength of the out-of-band peak; the shipped one does not.
    """
    from lumenairy.elements._lens_traced import _spectral_gap_cuts

    def prefix(marg, freqs, lo, hi, vf, pf):
        pp = np.asarray(marg, float)
        pp = pp / max(pp.max(), 1e-300)
        cuts = []
        for i in np.where((freqs >= lo) & (freqs <= hi))[0]:
            if i <= 0 or i >= len(pp) - 1:
                continue
            if pp[i] <= pp[i - 1] and pp[i] < pp[i + 1] and pp[i] < vf:
                if pp[:i].max() > pf and pp[i + 1:].max() > pf:
                    cuts.append(float(freqs[i]))
        return cuts

    f = np.linspace(-100.0, 100.0, 201)
    asym = (np.exp(-((f - 20) / 6.0) ** 2)
            + 0.30 * np.exp(-((f + 80) / 4.0) ** 2))
    assert prefix(asym, f, -40.0, 40.0, 0.05, 0.2) == [-40.0]
    assert _spectral_gap_cuts(asym, f, -40.0, 40.0, 0.05, 0.2) == []
    # a REAL two-lobe gap is still found, by both
    sym = (np.exp(-((f - 20) / 6.0) ** 2) + np.exp(-((f + 20) / 6.0) ** 2)
           + 0.30 * np.exp(-((f - 80) / 4.0) ** 2)
           + 0.30 * np.exp(-((f + 80) / 4.0) ** 2))
    assert _spectral_gap_cuts(sym, f, -40.0, 40.0, 0.05, 0.2) == [0.0]
    assert prefix(sym, f, -40.0, 40.0, 0.05, 0.2) == [0.0]


def test_the_angular_segmentation_still_sums_back_to_the_input():
    """The contract OI-4's change must not touch: with
    ``min_segment_power=0`` the flat-top partition is exactly unity, so the
    segments sum to the input.  BAR 1e-13 relative; measured 5.55e-16 (three
    decades), at both ``min_segment_power`` settings."""
    from lumenairy.elements._lens_traced import _segment_field_by_angle
    n, dx, wl = 384, 2e-6, 1.31e-6
    ax = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    k0 = 2.0 * np.pi / wl
    E = (np.exp(-(X ** 2 + Y ** 2) / (0.12e-3) ** 2)
         * np.exp(1j * k0 * 0.025 * X)
         + np.exp(-(X ** 2 + Y ** 2) / (0.12e-3) ** 2)
         * np.exp(-1j * k0 * 0.025 * X)).astype(np.complex128)
    for msp in (0.0, 1e-3):
        segs = _segment_field_by_angle(E, dx, dx, 'auto', 'auto', msp,
                                       0.995, 0.05, 8)
        err = (float(np.max(np.abs(sum(segs) - E)))
               / float(np.abs(E).max()))
        assert err < 1e-13, (msp, len(segs), err)


@pytest.mark.parametrize('bad', [2, 5, -3, 0.0, 1.5, 'first'])
def test_stop_index_is_refused_up_front_with_this_functions_own_name(bad):
    """Out-of-range or non-integer must raise from ``apply_real_lens_traced``
    BEFORE the trace, naming this entry point -- not ``apply_real_lens`` from
    inside the amplitude leg's worker thread, and not a bare ``int()``
    message."""
    presc = _singlet(50e-3, -50e-3, 3e-3, 2.0e-3)
    presc['stop_index'] = bad
    with pytest.raises(ValueError) as exc:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens_traced(
                np.ones((64, 64), dtype=np.complex128), prescription=presc,
                wavelength=_WL, dx=25e-6, ray_subsample=4,
                on_undersample='silent')
    msg = str(exc.value)
    assert msg.startswith('apply_real_lens_traced:'), msg
    assert 'stop_index' in msg
