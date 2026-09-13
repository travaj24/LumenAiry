"""WP-B3b regression pins -- AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, K6.

WP-B3 shipped ``resample_field(method='chirpz')`` and left the two call
sites that resample a kernel's natural output grid back onto a fixed
working grid to a follow-up.  This file pins what that follow-up did:

* ``propagate_through_system``'s ``'fresnel'`` leg evaluates the Fresnel
  integral STRAIGHT ONTO the chain grid
  (:func:`~lumenairy.propagators.fresnel_propagate_mft`), so there is no
  resample to crop and no interpolator MTF to pay;
* the remaining three resample-backs -- the chain's ``'sas'`` leg and
  ``apply_real_lens``'s two in-glass gap legs -- choose between the
  band-limited chirp-Z interpolant (unit MTF) and the historical cubic
  spline by asking whether the requested output WINDOW fits inside one
  period of the chirp-Z reconstruction.  Outside one period the chirp-Z
  leg returns periodic replicas, not the zeros the spline pads with.

Oracles here are INDEPENDENT of the code under test:

* the Fresnel diffraction integral written out as an explicit double sum
  over the input samples (no FFT, no Bluestein, no library call) -- it
  referees the ``'fresnel'`` leg absolutely, not relatively;
* an exactly-tiling field, for which the replica count is an integer the
  test derives from the geometry rather than reads off a build;
* the library's own pre-gate behaviour, reconstructed in-process by
  forcing ``method='spline'``, for the byte-identity arm.

Every bar carries its oracle, that oracle's own floor, the values
measured on 2026-09-13 and the decades of gap on each side, per
``docs/TESTING_STANDARDS.md``.  Nothing here reads a clock and nothing
pins a count produced by nondeterministic machinery (S5).

Author: Andrew Traverso -- WP-B3b.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
import warnings

import numpy as np
import pytest

import lumenairy.elements._lens_real as _lens_real
import lumenairy.propagators.propagation as _propagation
import lumenairy.propagators.system as _system
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.propagators.mft import fresnel_propagate_mft, resample_field
from lumenairy.propagators.propagation import (
    fresnel_propagate,
    scalable_angular_spectrum_propagate,
)
from lumenairy.propagators.system import propagate_through_system

LAM = 633e-9


# ---------------------------------------------------------------------------
# fixtures and the independent oracle
# ---------------------------------------------------------------------------

def _gauss(nx, ny, dx, wfac=0.18):
    x = (np.arange(nx) - nx / 2.0) * dx
    y = (np.arange(ny) - ny / 2.0) * dx
    xx, yy = np.meshgrid(x, y)
    w0 = wfac * nx * dx
    return np.exp(-(xx ** 2 + yy ** 2) / w0 ** 2).astype(np.complex128)


def _tophat(n, dx, frac=0.42):
    x = (np.arange(n) - n / 2.0) * dx
    xx, yy = np.meshgrid(x, x)
    return (np.hypot(xx, yy) <= frac * n * dx).astype(np.complex128)


def _fresnel_double_sum(E_in, z, lam, d_in, d_out, n_out):
    """The Fresnel diffraction integral as an EXPLICIT double sum.

        E_out[ky, kx] = e^{ikz}/(i lam z)
            * sum_{my, nx} E_in[my, nx]
              * exp(i k/(2z) [(x_out[kx]-x_in[nx])^2
                              + (y_out[ky]-y_in[my])^2]) d_in^2

    No FFT, no Bluestein, no library import -- two dense matrix products
    because the quadratic kernel is separable.  This is the quantity the
    chain's ``'fresnel'`` leg claims to evaluate on the chain grid, so it
    is an ABSOLUTE reference: agreement does not depend on any other
    propagator being right.

    Its own error floor is the f64 accumulation of ``N_in`` terms per
    output sample: ``eps * sqrt(N_in) ~ 2.2e-16 * 8`` = 1.8e-15 relative
    at ``N_in = 64``, which is what the measured 3.5e-15 below reflects.
    """
    k = 2.0 * np.pi / lam
    ny_in, nx_in = E_in.shape
    x_in = (np.arange(nx_in) - nx_in / 2.0) * d_in
    y_in = (np.arange(ny_in) - ny_in / 2.0) * d_in
    x_out = (np.arange(n_out) - n_out / 2.0) * d_out
    y_out = (np.arange(n_out) - n_out / 2.0) * d_out
    kx = np.exp(1j * k / (2.0 * z) * (x_out[:, None] - x_in[None, :]) ** 2)
    ky = np.exp(1j * k / (2.0 * z) * (y_out[:, None] - y_in[None, :]) ** 2)
    out = ky @ E_in.astype(np.complex128) @ kx.T
    return out * (np.exp(1j * k * z) / (1j * lam * z)) * d_in * d_in


def _rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _power(E, dx):
    return float(np.sum(np.abs(np.asarray(E)) ** 2)) * dx * dx


class _ResampleSpy:
    """Record every ``resample_field`` call the code under test makes.

    Both call sites are patched: ``system.py`` binds the name at import,
    ``_lens_real`` imports it from ``propagation`` inside the function
    body.
    """

    def __init__(self, force_method=None):
        self.calls = []
        self._force = force_method
        self._real = resample_field

    def __call__(self, E_in, dx_in, dx_out, N_out=None, order=3, *,
                 method='spline'):
        n_in = int(np.asarray(E_in).shape[-1])
        n_out = int(N_out) if N_out is not None else int(
            round(n_in * dx_in / dx_out))
        self.calls.append(dict(
            method=method, N_in=n_in,
            shape=tuple(np.asarray(E_in).shape),
            dx_in=float(dx_in), dx_out=float(dx_out), N_out=n_out,
            pitch_ratio=float(dx_in) / float(dx_out),
            window=n_out * float(dx_out),
            period=min(np.asarray(E_in).shape[-2:]) * float(dx_in)))
        if self._force is not None:
            method = self._force
        return self._real(E_in, dx_in, dx_out, N_out, order, method=method)

    def install(self, monkeypatch):
        monkeypatch.setattr(_system, 'resample_field', self)
        monkeypatch.setattr(_propagation, 'resample_field', self)
        return self


# ---------------------------------------------------------------------------
# 1. the 'fresnel' leg now evaluates onto the chain grid
# ---------------------------------------------------------------------------

class TestK6FresnelLegEvaluatesOntoTheChainGrid:
    """The leg's claim is absolute -- "this IS the Fresnel integral on the
    chain grid" -- so the test is against the integral itself, written
    out as a double sum, and not against any other propagator."""

    # (label, nx, ny, dx, z).  The non-square rows matter: the resample
    # back onto a square grid had only ONE pitch to work with and used
    # the x ratio on both axes, so the y scale came out wrong by nx/ny.
    CASES = [
        ('square_24', 24, 24, 2e-6, 1e-3),
        ('square_32_z2mm', 32, 32, 2e-6, 2e-3),
        ('square_64', 64, 64, 2e-6, 1e-3),
        ('nonsquare_24x18', 24, 18, 2e-6, 1e-3),
        ('nonsquare_64x48', 64, 48, 2e-6, 1e-3),
    ]

    @pytest.mark.parametrize('label,nx,ny,dx,z',
                             CASES, ids=[c[0] for c in CASES])
    def test_the_leg_reproduces_the_fresnel_double_sum(self, label, nx, ny,
                                                       dx, z):
        """BAR 1e-11 relative L2 against the double-sum oracle.

        Derivation.  The oracle's own floor is the f64 accumulation of
        ``N_in`` terms per output sample (``eps*sqrt(N_in)`` = 1.8e-15 at
        N = 64); measured 5.3e-16 .. 3.6e-15 across these five fixtures
        on 2026-09-13, i.e. the bar sits ~3.5 decades ABOVE the floor.
        The smallest real signal it must separate from is the leg this
        replaced -- propagate-to-the-natural-grid-then-interpolate --
        which reads 1.0e-4 (square_64) to 4.3e-1 (nonsquare_24x18) on the
        same fixtures, i.e. ~7 decades ABOVE the bar.  Decades on both
        sides, as ``docs/TESTING_STANDARDS.md`` restatement 5 requires.
        """
        E = _gauss(nx, ny, dx)
        ref = _fresnel_double_sum(E, z, LAM, dx, dx, nx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out, _ = propagate_through_system(
                E, [{'type': 'propagate', 'z': z}], LAM, dx,
                method='fresnel')
        assert out.shape == (nx, nx), (
            f"{label}: the leg must deliver the chain's own sample count")
        assert _rel_l2(out, ref) < 1e-11, (
            f"{label}: the 'fresnel' leg is {_rel_l2(out, ref):.3e} from "
            f"the Fresnel double sum evaluated on the same grid")

    def test_the_resample_back_is_the_thing_the_bar_separates_from(self):
        """Fail-before, constructed through the public API.

        Rebuild the pre-gate leg -- ``fresnel_propagate`` onto its natural
        grid, then ``resample_field`` back -- and show it is FURTHER from
        the double-sum oracle than the shipped leg by at least two
        decades.  Stated as a RATIO derived at runtime, so it tracks the
        library instead of pinning one build's numbers.
        """
        n, dx, z = 64, 2e-6, 1e-3
        E = _gauss(n, n, dx)
        ref = _fresnel_double_sum(E, z, LAM, dx, dx, n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            shipped, _ = propagate_through_system(
                E, [{'type': 'propagate', 'z': z}], LAM, dx,
                method='fresnel')
            nat, dx_new, _ = fresnel_propagate(E, z, LAM, dx, dx)
            old, _ = resample_field(nat, dx_new, dx, N_out=n)
        err_new, err_old = _rel_l2(shipped, ref), _rel_l2(old, ref)
        assert err_old > 100.0 * err_new, (
            f"the resample-back reads {err_old:.3e} and the direct "
            f"evaluation {err_new:.3e} against the same oracle; the "
            f"direct evaluation must be at least two decades closer")

    def test_the_leg_keeps_the_chain_pitch_and_the_input_sample_count(self):
        """No element after a 'fresnel' step may find a different grid."""
        n, dx = 48, 2e-6
        E = _gauss(n, n, dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out, _ = propagate_through_system(
                E, [{'type': 'propagate', 'z': 1e-3},
                    {'type': 'lens', 'f': 5e-3},
                    {'type': 'propagate', 'z': 1e-3}], LAM, dx,
                method='fresnel')
        assert out.shape == E.shape

    def test_the_leg_performs_no_resample(self, monkeypatch):
        """Structural, two-sided: the 'fresnel' leg calls
        ``resample_field`` ZERO times where it used to call it once, and
        the 'sas' leg on the same geometry still calls it, so the spy
        cannot be passing because it is blind."""
        spy = _ResampleSpy().install(monkeypatch)
        E = _gauss(64, 64, 2e-6)
        elems = [{'type': 'propagate', 'z': 1e-3}]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            propagate_through_system(E, elems, LAM, 2e-6, method='fresnel')
        assert spy.calls == [], (
            f"the 'fresnel' leg still resamples: {spy.calls}")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            propagate_through_system(E, elems, LAM, 2e-6, method='sas')
        assert len(spy.calls) == 1, (
            "the 'sas' leg must still resample back -- otherwise this "
            "spy proves nothing about the 'fresnel' leg")

    def test_the_undersampled_chirp_guard_still_fires_from_this_leg(self):
        """K1's validity bound is a property of ``dx_in`` and ``z``, which
        the direct evaluation shares; losing the diagnostic would make a
        silently-aliased chain quieter than it was."""
        n, dx = 128, 2e-6
        z = 0.25 * n * dx ** 2 / LAM          # a quarter of the bound
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            propagate_through_system(_gauss(n, n, dx),
                                     [{'type': 'propagate', 'z': z}],
                                     LAM, dx, method='fresnel')
        texts = [str(x.message) for x in w]
        assert any('quadratic Fresnel chirp is UNDER-SAMPLED' in t
                   for t in texts), texts

    @pytest.mark.parametrize('z', [0.0, -1e-3])
    def test_a_non_positive_z_is_still_refused(self, z):
        with pytest.raises(ValueError, match=r"z must be > 0"):
            propagate_through_system(_gauss(32, 32, 2e-6),
                                     [{'type': 'propagate', 'z': z}],
                                     LAM, 2e-6, method='fresnel')

    def test_an_anamorphic_working_pitch_is_still_refused(self):
        with pytest.raises(ValueError,
                           match=r"'fresnel' step assumes a square"):
            propagate_through_system(_gauss(32, 32, 2e-6),
                                     [{'type': 'propagate', 'z': 1e-3}],
                                     LAM, 2e-6, dy=3e-6, method='fresnel')


# ---------------------------------------------------------------------------
# 2. the chirp-Z gate on the three surviving resample-backs
# ---------------------------------------------------------------------------

class TestK6TheChirpZGate:
    """The gate's rule is ``N_out*dx_out <= N_in*dx_in`` per axis -- the
    window against one period of the chirp-Z reconstruction."""

    # Chain fixtures that reach the 'sas' resample-back in both
    # directions.  SAS's natural pitch is lambda*z/(pad*N*dx), so a long
    # z coarsens and a short one refines.
    SAS_CASES = [
        ('sas_coarsens_64', 64, 1e-3),
        ('sas_coarsens_256', 256, 5e-3),
        ('sas_refines_512', 512, 5e-3),
        ('sas_refines_256', 256, 1e-3),
    ]

    @pytest.mark.parametrize('label,n,z',
                             SAS_CASES, ids=[c[0] for c in SAS_CASES])
    def test_the_system_sas_leg_gates_on_window_against_period(
            self, label, n, z, monkeypatch):
        spy = _ResampleSpy().install(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            propagate_through_system(_tophat(n, 2e-6),
                                     [{'type': 'propagate', 'z': z}],
                                     LAM, 2e-6, method='sas')
        assert len(spy.calls) == 1, spy.calls
        c = spy.calls[0]
        fits = c['window'] <= c['period'] * (1.0 + 1e-9)
        assert c['method'] == ('chirpz' if fits else 'spline'), c
        # and, since ``N_out == N_in`` at this call site, the short form
        # the WP-B3 report wrote must agree with the general one here
        assert fits == (c['pitch_ratio'] >= 1.0), c

    # The in-glass gap.  ``lam_medium = wavelength/n`` makes ``dx_new``
    # smaller by ``n`` than the same geometry in air, so a lens on a wide
    # grid lands in the spline's half and a thin plate on a fine grid in
    # the chirp-Z's.
    PLATE = {
        'surfaces': [{'radius': float('inf'), 'glass_before': 'AIR',
                      'glass_after': 'N-BK7'},
                     {'radius': float('inf'), 'glass_before': 'N-BK7',
                      'glass_after': 'AIR'}],
        'thicknesses': [1e-3],
        'aperture_diameter': 5e-5,
    }
    DOUBLET = {
        'surfaces': [
            dict(radius=33.3e-3, glass_before='AIR', glass_after='N-BAF10'),
            dict(radius=-22.28e-3, glass_before='N-BAF10',
                 glass_after='N-SF6HT'),
            dict(radius=-291.07e-3, glass_before='N-SF6HT',
                 glass_after='AIR'),
        ],
        'thicknesses': [9.0e-3, 2.5e-3],
        'aperture_diameter': 6.0e-3,
    }
    GAP_CASES = [
        ('plate_fine_grid_fresnel', 'PLATE', 2e-6, 'fresnel'),
        ('plate_fine_grid_sas', 'PLATE', 6e-6, 'sas'),
        ('doublet_wide_grid_fresnel', 'DOUBLET', 1.2 * 6e-3 / 64, 'fresnel'),
        ('doublet_wide_grid_sas', 'DOUBLET', 1.2 * 6e-3 / 64, 'sas'),
    ]

    @pytest.mark.parametrize('label,pres,dx,prop',
                             GAP_CASES, ids=[c[0] for c in GAP_CASES])
    def test_the_in_glass_gap_legs_gate_on_window_against_period(
            self, label, pres, dx, prop, monkeypatch):
        spy = _ResampleSpy().install(monkeypatch)
        E = np.ones((64, 64), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E, prescription=getattr(self, pres),
                            wavelength=LAM, dx=dx, dy=dx,
                            wave_propagator=prop)
        assert spy.calls, f"{label} did not reach the gap resample-back"
        for c in spy.calls:
            fits = c['window'] <= c['period'] * (1.0 + 1e-9)
            assert c['method'] == ('chirpz' if fits else 'spline'), c

    def test_the_covering_array_doublet_lands_in_the_spline_half(
            self, monkeypatch):
        """Two-sided companion to the row above, stated as the physics:
        on the WP-A15a covering-array grid (N = 64 over 1.2 aperture
        diameters) BOTH in-glass gaps converge, so neither may take the
        chirp-Z leg -- and the same prescription on a fine grid does."""
        spy = _ResampleSpy().install(monkeypatch)
        dx = 1.2 * 6.0e-3 / 64
        E = np.ones((64, 64), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            apply_real_lens(E, prescription=self.DOUBLET, wavelength=632.8e-9,
                            dx=dx, dy=dx, wave_propagator='fresnel')
        assert len(spy.calls) == 2, spy.calls
        assert [c['method'] for c in spy.calls] == ['spline', 'spline'], (
            spy.calls)
        assert all(c['pitch_ratio'] < 0.01 for c in spy.calls), spy.calls

    def test_an_ungated_chirpz_returns_the_tiling_power(self):
        """Fail-before for the gate, with no library state involved: a
        field that fills its grid tiles EXACTLY under the periodic
        reconstruction, so an m-period-wide window must return m**2 times
        the power.  ``m`` is derived from the requested geometry, not read
        off a build; the bar is a basin of 0.02 around the integer m**2,
        two decades below the 3.0 separating m=1 from m=2."""
        n, dx_in = 64, 1e-6
        rng = np.random.default_rng(0)
        E = (rng.standard_normal((n, n))
             + 1j * rng.standard_normal((n, n))).astype(np.complex128)
        # keep only frequencies the grid genuinely represents, so the
        # reconstruction is exact and the tiling is not a sampling artefact
        spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))
        keep = np.zeros((n, n), bool)
        keep[n // 4:3 * n // 4, n // 4:3 * n // 4] = True
        E = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spec * keep)))
        p_in = _power(E, dx_in)
        for m in (1, 2, 3):
            dx_out = dx_in / m
            n_out = n * m * m          # window = m * (n * dx_in)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                out, _ = resample_field(E, dx_in, dx_out, N_out=n_out,
                                        method='chirpz')
            ratio = _power(out, dx_out) / p_in
            assert abs(ratio - m * m) < 0.02, (m, ratio)
            warned = any('faithful zone' in str(x.message) for x in w)
            assert warned == (m > 1), (m, [str(x.message)[:80] for x in w])

    def test_the_gate_is_written_in_the_window_period_form(self):
        """Structural pin on the two modules' call sites.

        VERIFY-B3 measured the replica condition as
        ``N_out*dx_out > N_in*dx_in``; ``dx_new >= dx`` is only equivalent
        to it because ``N_out == N_in`` at these call sites.  Walk the AST
        of each owner function, find every ``resample_field`` call, and
        require that its ``method=`` selector compares two quantities that
        BOTH involve a sample count -- i.e. a window against a period, not
        a bare pitch test.  This is what stops a future change of
        ``N_out`` from silently turning the gate wrong.
        """
        for fn in (_system.propagate_through_system,
                   _lens_real._propagate_through_glass):
            tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
            # one level of local naming is allowed: a gate may read
            # ``window_out <= period_in * (1 + eps)`` with those two
            # assigned just above, so resolve simple ``name = expr``
            # bindings before looking for the sample counts.
            local = {}
            for node in ast.walk(tree):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name)):
                    local[node.targets[0].id] = ast.dump(node.value)
            found = 0
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'resample_field'):
                    continue
                sel = [kw.value for kw in node.keywords
                       if kw.arg == 'method']
                assert sel, (
                    f"{fn.__qualname__}: a resample_field call with no "
                    f"explicit method= selector")
                assert isinstance(sel[0], ast.IfExp), (
                    f"{fn.__qualname__}: method= is not a gate")
                src = ast.dump(sel[0].test)
                expanded = src + ''.join(
                    local.get(n.id, '') for n in ast.walk(sel[0].test)
                    if isinstance(n, ast.Name))
                assert expanded.count("attr='shape'") >= 2, (
                    f"{fn.__qualname__}: the gate reads {src}, which does "
                    f"not compare a window against a period (both sides "
                    f"must carry a sample count)")
                found += 1
            assert found >= 1, f"{fn.__qualname__}: no gated resample found"


# ---------------------------------------------------------------------------
# 3. byte identity where the gate selects the spline
# ---------------------------------------------------------------------------

class TestK6ByteIdentityWhereTheGateSelectsTheSpline:
    """The pre-gate library called ``resample_field`` with its default
    ``method='spline'`` unconditionally.  Forcing that default back in
    process reconstructs it exactly, so "byte-identical where the gate
    picks the spline" is a claim this suite can make on any build -- and
    the diverging arm proves the comparison is not vacuous."""

    def _run(self, fn, force):
        spy = _ResampleSpy(force_method=force)
        old_sys = _system.resample_field
        old_prop = _propagation.resample_field
        _system.resample_field = spy
        _propagation.resample_field = spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = fn()
        finally:
            _system.resample_field = old_sys
            _propagation.resample_field = old_prop
        return np.asarray(out[0] if isinstance(out, tuple) else out), spy

    CONVERGING = [
        ('sys_sas_512_z5mm',
         lambda: propagate_through_system(
             _tophat(512, 2e-6), [{'type': 'propagate', 'z': 5e-3}],
             LAM, 2e-6, method='sas')),
        ('sys_sas_256_z1mm',
         lambda: propagate_through_system(
             _tophat(256, 2e-6), [{'type': 'propagate', 'z': 1e-3}],
             LAM, 2e-6, method='sas')),
        ('gap_sas_bk7_plate',
         lambda: apply_real_lens(
             np.ones((64, 64), dtype=np.complex128),
             prescription=TestK6TheChirpZGate.PLATE, wavelength=LAM,
             dx=6e-6, dy=6e-6, wave_propagator='sas')),
        ('gap_fresnel_doublet',
         lambda: apply_real_lens(
             np.ones((64, 64), dtype=np.complex128),
             prescription=TestK6TheChirpZGate.DOUBLET, wavelength=632.8e-9,
             dx=1.2 * 6e-3 / 64, dy=1.2 * 6e-3 / 64,
             wave_propagator='fresnel')),
    ]

    @pytest.mark.parametrize('label,fn', CONVERGING,
                             ids=[c[0] for c in CONVERGING])
    def test_identical_bits_where_the_window_exceeds_the_period(self, label,
                                                                fn):
        gated, spy_a = self._run(fn, None)
        forced, spy_b = self._run(fn, 'spline')
        assert spy_a.calls, f"{label} reached no resample-back"
        assert all(c['method'] == 'spline' for c in spy_a.calls), spy_a.calls
        assert gated.dtype == forced.dtype and gated.shape == forced.shape
        assert gated.tobytes() == forced.tobytes(), (
            f"{label}: the gated call site must be bit-for-bit the "
            f"unconditional spline it replaced")
        assert spy_b.calls

    DIVERGING = [
        ('sys_sas_64_z1mm',
         lambda: propagate_through_system(
             np.ones((64, 64), dtype=np.complex128),
             [{'type': 'propagate', 'z': 1e-3}], LAM, 2e-6, method='sas')),
        ('gap_fresnel_bk7_plate',
         lambda: apply_real_lens(
             np.ones((64, 64), dtype=np.complex128),
             prescription=TestK6TheChirpZGate.PLATE, wavelength=LAM,
             dx=2e-6, wave_propagator='fresnel')),
    ]

    @pytest.mark.parametrize('label,fn', DIVERGING,
                             ids=[c[0] for c in DIVERGING])
    def test_the_comparison_is_not_vacuous_where_the_window_fits(self, label,
                                                                 fn):
        """The same harness on a fixture the gate sends to chirp-Z must
        show a DIFFERENCE -- otherwise the identity above would pass on a
        build where the selector had no effect at all."""
        gated, spy_a = self._run(fn, None)
        forced, _ = self._run(fn, 'spline')
        assert all(c['method'] == 'chirpz' for c in spy_a.calls), spy_a.calls
        assert gated.tobytes() != forced.tobytes(), (
            f"{label}: the chirp-Z leg produced the spline's bits")

    def test_the_default_resampler_leg_is_untouched(self):
        """``resample_field``'s own default is still the cubic
        ``map_coordinates`` interpolation this work package did not
        touch: driven directly with the documented coordinate map it is
        bit-identical."""
        from scipy.ndimage import map_coordinates
        rng = np.random.default_rng(7)
        E = (rng.standard_normal((33, 21))
             + 1j * rng.standard_normal((33, 21))).astype(np.complex128)
        dx_in, dx_out, n_out = 2e-6, 1.4e-6, 45
        out, dx_ret = resample_field(E, dx_in, dx_out, n_out)
        ny_in, nx_in = E.shape
        scale = dx_out / dx_in
        ix = (np.arange(n_out) - n_out / 2) * scale + nx_in / 2
        iy = (np.arange(n_out) - n_out / 2) * scale + ny_in / 2
        IX, IY = np.meshgrid(ix, iy)
        coords = np.array([IY.ravel(), IX.ravel()])
        ref = (map_coordinates(E.real, coords, order=3, mode='constant',
                               cval=0.0)
               + 1j * map_coordinates(E.imag, coords, order=3,
                                      mode='constant', cval=0.0)
               ).reshape(n_out, n_out)
        assert dx_ret == dx_out
        assert out.tobytes() == ref.tobytes()


# ---------------------------------------------------------------------------
# 4. the improvement the gate buys where it selects chirp-Z
# ---------------------------------------------------------------------------

class TestK6TheImprovementWhereTheGateSelectsChirpZ:

    def test_chirpz_tracks_the_direct_evaluation_better_than_the_spline(
            self):
        """On a fixture whose window fits inside one period, the unit-MTF
        leg must reproduce the direct Fresnel evaluation's window power
        more closely than the spline does.  Both distances are measured
        here; nothing is pinned.
        """
        n, dx, z = 512, 2e-6, 5e-3
        E = _tophat(n, dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nat, dx_new, _ = fresnel_propagate(E, z, LAM, dx, dx)
            direct = fresnel_propagate_mft(E, z, LAM, dx, dx, n,
                                           dy_in=dx, dy_out=dx)
            sp, _ = resample_field(nat, dx_new, dx, N_out=n,
                                   method='spline')
            cz, _ = resample_field(nat, dx_new, dx, N_out=n,
                                   method='chirpz')
        assert n * dx <= nat.shape[-1] * dx_new * (1 + 1e-9), (
            "fixture must be on the chirp-Z side of the gate")
        p_dir = _power(direct, dx)
        d_sp = abs(_power(sp, dx) - p_dir)
        d_cz = abs(_power(cz, dx) - p_dir)
        assert d_cz < 0.2 * d_sp, (d_cz, d_sp)

    def test_chirpz_is_the_better_interpolant_on_a_contained_field(self):
        """A field wholly inside the window is resampled EXACTLY by the
        band-limited interpolant, so its distance to the direct
        evaluation must be decades below the spline's.  Ratio derived at
        runtime; bar 100x against a measured 1.6e5 (2026-09-13)."""
        n, dx, z = 256, 2e-6, 5e-3
        E = _gauss(n, n, dx, wfac=0.06)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nat, dx_new, _ = fresnel_propagate(E, z, LAM, dx, dx)
            direct = fresnel_propagate_mft(E, z, LAM, dx, dx, n,
                                           dy_in=dx, dy_out=dx)
            sp, _ = resample_field(nat, dx_new, dx, N_out=n,
                                   method='spline')
            cz, _ = resample_field(nat, dx_new, dx, N_out=n,
                                   method='chirpz')
        assert _rel_l2(sp, direct) > 100.0 * _rel_l2(cz, direct), (
            _rel_l2(sp, direct), _rel_l2(cz, direct))


# ---------------------------------------------------------------------------
# 5. F6 -- what "unit MTF" is a property OF
# ---------------------------------------------------------------------------

class TestF6TheUnitMtfIsAPropertyOfTheWindow:
    """VERIFY-B3 F6: ``resample_field``'s unit MTF reads exactly 1 only
    when the output window is exactly one reconstruction period.  The
    extent-preserving default ``N_out = round(N_in*dx_in/dx_out)`` lands
    there only when that ratio comes out whole."""

    def _carrier(self, n, dx, wfac):
        x = (np.arange(n) - n / 2.0) * dx
        xx, yy = np.meshgrid(x, x)
        env = np.exp(-(xx ** 2 + yy ** 2) / (wfac * n * dx) ** 2)
        return (env * np.exp(2j * np.pi * 0.30 * xx / dx)).astype(
            np.complex128)

    def _out_of_band_share(self, E, dx_in, dx_out):
        """Share of the INPUT's own spectral power above the output
        Nyquist -- measured from the fixture, not assumed."""
        n = E.shape[-1]
        spec = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E)))
        f = (np.arange(n) - n / 2.0) / (n * dx_in)
        fx, fy = np.meshgrid(f, f)
        nyq = 1.0 / (2.0 * dx_out)
        out = (np.abs(fx) > nyq) | (np.abs(fy) > nyq)
        return float(np.sum(np.abs(spec[out]) ** 2)
                     / np.sum(np.abs(spec) ** 2))

    @pytest.mark.parametrize('scale', [0.5, 0.8, 1.0])
    def test_an_exact_period_window_returns_the_power_exactly(self, scale):
        """BAR 1e-12.  Oracle: Parseval on the input itself (no library
        call).  These three scales do not DOWN-sample, so nothing folds
        and the ratio is the interpolant's gain alone: measured
        1.1e-16 .. 6.7e-16 on 2026-09-13 for both a contained and a
        rim-filling envelope, i.e. the bar sits ~3.5 decades above the
        f64 floor and ~8 below the 1e-4-class departure a ROUNDED window
        shows on the same fixture (the row below)."""
        n, dx = 128, 1e-6
        for wfac in (0.18, 0.45):
            E = self._carrier(n, dx, wfac)
            dx_out = scale * dx
            n_out = n * dx / dx_out
            assert abs(n_out - round(n_out)) < 1e-12, scale
            assert self._out_of_band_share(E, dx, dx_out) == 0.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out, _ = resample_field(E, dx, dx_out, int(round(n_out)),
                                        method='chirpz')
            ratio = _power(out, dx_out) / _power(E, dx)
            assert abs(ratio - 1.0) < 1e-12, (scale, wfac, ratio)

    @pytest.mark.parametrize('scale', [2.0, 4.0])
    def test_a_downsample_at_an_exact_window_is_the_fold_not_the_gain(
            self, scale):
        """The unit-gain reading is a statement about the interpolant,
        not about what survives a coarser Nyquist.  At these scales the
        0.30 cyc/px carrier is ABOVE the new Nyquist, so essentially the
        whole input spectrum folds -- measured out-of-band share 0.9977
        to 1.0000 of the input's own power on 2026-09-13 -- and the
        ratio departs from 1 by the interference between the folded
        components: -3.3e-9 (contained envelope) to -4.6e-4 (rim-filling)
        there.  Both sides are measured on the fixture: the fold must be
        large, and the power it costs must be at least two decades below
        it, because a fold RELABELS frequencies rather than destroying
        them."""
        n, dx = 128, 1e-6
        for wfac in (0.18, 0.45):
            E = self._carrier(n, dx, wfac)
            dx_out = scale * dx
            n_out = int(round(n * dx / dx_out))
            oob = self._out_of_band_share(E, dx, dx_out)
            assert oob > 0.9, (scale, wfac, oob)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out, _ = resample_field(E, dx, dx_out, n_out,
                                        method='chirpz')
            ratio = _power(out, dx_out) / _power(E, dx)
            assert abs(ratio - 1.0) < 0.01 * oob, (scale, wfac, ratio, oob)

    def test_a_rounded_window_departs_by_the_field_in_the_sliver(self):
        """Two-sided: the SAME scale factor reads 1 to 1e-12 on a
        contained envelope and departs by more than 1e-4 on one with
        power at the rim, because the departure is the field in the
        half-pixel the rounding adds or drops -- not an MTF."""
        n, dx, scale = 128, 1e-6, 1.25
        assert abs(n / scale - round(n / scale)) > 1e-9, (
            "this scale must NOT divide the grid, or there is no sliver")
        ratios = {}
        for wfac in (0.18, 0.45):
            E = self._carrier(n, dx, wfac)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out, _ = resample_field(E, dx, scale * dx, method='chirpz')
            ratios[wfac] = _power(out, scale * dx) / _power(E, dx)
        assert abs(ratios[0.18] - 1.0) < 1e-4, ratios
        assert abs(ratios[0.45] - 1.0) > 1e-4, ratios

    def test_the_docstring_states_the_condition(self):
        doc = resample_field.__doc__
        assert 'reconstruction period' in doc
        assert 'N_out*dx_out == N_in*dx_in' in doc
