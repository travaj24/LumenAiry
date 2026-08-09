"""v5.1.0 Agent A regression tests -- library-wide default-config
knob resolver rollout.

Closes the long-deferred work item from
``docs/audits/AUDIT_V5_0_1_2026_05_20.md`` Part 7 v5.1 horizon item
1: ``set_default_wave_propagator`` / ``set_default_dy`` /
``set_default_real_dtype`` were API-only at v4.16.2 - v5.0.x; v5.1.0
wires the resolver at:

* ``apply_real_lens``                       (``_lens_real.py``)
* ``apply_real_lens_traced``                (``_lens_traced.py``)
* ``propagate_through_system``              (``system.py`` ``method=`` arg)
* ``_geometric_lens_phase``                 (``_lens_traced.py`` real-dtype)

The v4.16.3 sibling-gap pin
(``test_wave_propagator_and_dy_have_zero_consumers_lib_wide``) was
designed to fail loudly when the first consumer wired; v5.1.0 Agent
B retired it and replaced it with the inverse pin
(``test_wave_propagator_and_dy_now_have_consumers``).  The setters'
v4.16.3 one-shot ``UserWarning`` ("API-only; consumer wiring lands
in v5.0") is now retired.

The pins below are functional: each verifies the resolver actually
steers downstream behaviour, not just that the source string is
present.

Author: LumenAiry v5.1.0 Agent A.
"""
from __future__ import annotations

import contextlib
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers + fixtures
# ---------------------------------------------------------------------------


def _function_source(func):
    """Return ``func``'s source text, read FRESH from disk and located
    BY NAME rather than by the code object's line number.

    Do not replace this with ``inspect.getsource``.  ``getsource``
    slices the file at ``func.__code__.co_firstlineno``, which is baked
    in when the module is IMPORTED, but ``inspect.findsource`` calls
    ``linecache.checkcache`` first and therefore RE-READS the file
    whenever its ``(size, mtime)`` has moved.  Those two facts disagree
    the moment the module's file is edited after the process imported
    it -- routine on a dev box, where a sibling agent editing
    ``lumenairy/elements/_lens_traced.py`` (an ~8 kLOC file whose
    ``_geometric_lens_phase`` sits around line 2550) while a serial
    full-suite preflight is running shifts every line below the edit.
    ``getsource`` then hands back a DIFFERENT function's body, and a
    source-text pin fails with no library defect, no steering-global
    leak and no test-order dependence -- then passes again the instant
    the tree settles, which is the most expensive shape of false
    failure this suite produces.  That is the adjudicated cause of the
    2026-08-09 sighting of
    ``test_geometric_lens_phase_honours_real_dtype``; see
    ``docs/audits/FIX_STEERING_FAMILY_2026_08_09.md``.

    Finding the ``def`` by name and slicing to the next line at or
    below its indentation removes the line-number dependence entirely:
    the answer becomes a property of the file on disk, whatever that
    file currently is.  ``tests/unit/test_v5_1_0_agent_f_split.py``
    (``test_prescriptions_shell_is_thin``) reads from disk for the same
    class of reason.
    """
    import inspect
    path = inspect.getsourcefile(func)
    assert path, f"no source file for {func!r}"
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    name = func.__name__
    pat = re.compile(r'^(\s*)def\s+' + re.escape(name) + r'\s*\(')
    start = indent = None
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m is not None:
            assert start is None, (
                f"{name!r} is defined more than once in {path}; this "
                f"helper cannot tell which one the caller means.")
            start, indent = i, len(m.group(1))
    assert start is not None, (
        f"could not find 'def {name}(' in {path} -- the function was "
        f"renamed or moved to another module; update the pin.")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if not stripped:
            continue
        if len(lines[j]) - len(lines[j].lstrip()) <= indent:
            end = j
            break
    return '\n'.join(lines[start:end]) + '\n'


# ---------------------------------------------------------------------------
# Steering-global save/restore.
#
# These restore the ``fft_infra`` module attribute DIRECTLY instead of
# calling ``set_default_*(get_default_*())``.  ``fft_infra`` is the single
# source of truth -- ``propagation`` deletes its import-time bindings and
# forwards through ``__getattr__`` (see ``_LIVE_FORWARD_NAMES`` there) --
# so one ``setattr`` restores every view of the knob.
#
# The reason not to round-trip through the public setter is specific and
# measured: ``set_default_real_dtype`` / ``set_default_complex_dtype``
# normalise their argument through ``np.dtype(...)``, while the SHIPPED
# values are the numpy scalar TYPES ``np.float64`` / ``np.complex128``.
# So ``set_default_real_dtype(get_default_real_dtype())`` -- a no-op by
# intent -- rewrites ``DEFAULT_REAL_DTYPE`` from ``np.float64`` to
# ``np.dtype('float64')``.  The two compare equal and behave identically,
# so nothing breaks today, but the "restore" left the process in a state
# it did not find it in, which a state-diff harness reports as a leak and
# which any future ``is``-comparison would trip over.  Saving and
# restoring the attribute itself is exact.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _restored(attr):
    """Save and restore one ``fft_infra`` steering knob EXACTLY.

    Restores on the exception path too, so a failing assertion inside
    the ``with`` body still cannot hand the knob to the next test.
    """
    from lumenairy.propagators import fft_infra as _fft
    original = getattr(_fft, attr)
    try:
        yield original
    finally:
        setattr(_fft, attr, original)


@pytest.fixture
def restore_default_wave_propagator():
    """Save / restore ``DEFAULT_WAVE_PROPAGATOR`` so a test that
    changes the library-wide default does not leak into other tests."""
    with _restored('DEFAULT_WAVE_PROPAGATOR'):
        yield


@pytest.fixture
def restore_default_dy():
    with _restored('DEFAULT_DY'):
        yield


@pytest.fixture
def restore_default_real_dtype():
    with _restored('DEFAULT_REAL_DTYPE'):
        yield


def _minimal_prescription(aperture_diameter=5e-5):
    """A trivial flat-plate prescription suitable for driving
    ``apply_real_lens`` end-to-end without expensive ray-tracing
    or grid-vs-aperture warnings.  Refractive (BK7 in glass between
    the two flat surfaces) so the in-glass ASM step exercises both
    ``dx`` and ``dy``, but flat curvature means no per-surface sag
    phase (the resolver paths under test fire regardless).

    Default aperture (50 um) is small enough that a 32x32 grid at
    dx=2 um (64 um span) doesn't trigger the aperture-exceeds-grid
    warning.
    """
    return {
        'surfaces': [
            {'radius': float('inf'),
             'glass_before': 'AIR', 'glass_after': 'N-BK7'},
            {'radius': float('inf'),
             'glass_before': 'N-BK7', 'glass_after': 'AIR'},
        ],
        'thicknesses': [1e-3],  # 1 mm flat plate
        'aperture_diameter': aperture_diameter,
    }


def _trivial_no_refraction_prescription(aperture_diameter=5e-5):
    """A truly trivial single-glass prescription (AIR-to-AIR with
    no glass change anywhere).  Used by direct-call tests of
    ``_geometric_lens_phase`` where the inner sag loop short-circuits
    via ``abs(n2 - n1) < 1e-15`` (avoiding the latent
    ``_rt._surface_sag_xy`` lookup that's outside this agent's
    scope to fix)."""
    return {
        'surfaces': [
            {'radius': float('inf'),
             'glass_before': 'AIR', 'glass_after': 'AIR'},
            {'radius': float('inf'),
             'glass_before': 'AIR', 'glass_after': 'AIR'},
        ],
        'thicknesses': [1e-3],
        'aperture_diameter': aperture_diameter,
    }


# ============================================================================
# Wave-propagator knob -- functional consumer wiring at apply_real_lens.
# ============================================================================


class TestSetDefaultWavePropagatorSteersApplyRealLens:
    """v5.1.0: ``set_default_wave_propagator('fresnel')`` must steer
    the inner propagator dispatch when ``apply_real_lens`` is called
    without an explicit ``wave_propagator=`` kwarg.  Behaviourally
    pinned (not just source-string pinned)."""

    def test_default_resolves_to_asm_on_fresh_import(
            self, restore_default_wave_propagator):
        """Bit-for-bit back-compat: with the library default at its
        starting value ``'asm'``, an ``apply_real_lens`` call with
        no ``wave_propagator=`` kwarg matches an explicit
        ``wave_propagator='asm'`` call exactly."""
        import lumenairy as la

        # Reset to library default explicitly so this test is robust
        # against test-order side-effects.
        la.set_default_wave_propagator('asm')

        E_in = np.ones((64, 64), dtype=np.complex128)
        pres = _minimal_prescription()

        # No kwarg (resolves via default).
        E_default = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
        )
        # Explicit asm kwarg.
        E_explicit = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
            wave_propagator='asm',
        )
        np.testing.assert_array_equal(
            E_default, E_explicit,
            err_msg=("v5.1.0 back-compat regression: "
                     "apply_real_lens with no wave_propagator= kwarg "
                     "must match wave_propagator='asm' bit-for-bit "
                     "when the library default is 'asm'."),
        )

    def test_set_default_to_fresnel_changes_apply_real_lens_output(
            self, restore_default_wave_propagator):
        """After ``set_default_wave_propagator('fresnel')``, a
        kwargless ``apply_real_lens`` call must produce the same
        output as an explicit ``wave_propagator='fresnel'`` call,
        and that output must differ from the ASM result (since the
        two methods produce numerically different fields)."""
        import lumenairy as la

        E_in = np.ones((64, 64), dtype=np.complex128)
        pres = _minimal_prescription()

        # Baseline: ASM.
        la.set_default_wave_propagator('asm')
        E_asm = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
        )

        # Switch the library default to Fresnel.
        la.set_default_wave_propagator('fresnel')
        E_default_fresnel = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
        )
        E_explicit_fresnel = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
            wave_propagator='fresnel',
        )

        # Default-resolved must match explicit-Fresnel.
        np.testing.assert_array_equal(
            E_default_fresnel, E_explicit_fresnel,
            err_msg=("v5.1.0 functional pin: kwargless "
                     "apply_real_lens must use the library default "
                     "when no explicit wave_propagator= is passed."),
        )

        # And it must differ from ASM (otherwise the test is vacuous).
        assert not np.allclose(E_default_fresnel, E_asm), (
            "v5.1.0 functional pin: setting the library default "
            "to 'fresnel' did not change the apply_real_lens output "
            "vs ASM.  The setter is not steering the resolver.")

    def test_per_call_kwarg_overrides_library_default(
            self, restore_default_wave_propagator):
        """An explicit ``wave_propagator=<name>`` kwarg must override
        the library-wide default -- the resolver only fires when the
        kwarg is ``None`` (its new default)."""
        import lumenairy as la

        E_in = np.ones((64, 64), dtype=np.complex128)
        pres = _minimal_prescription()

        # Set library default to 'fresnel' but pass 'asm' per-call.
        la.set_default_wave_propagator('fresnel')
        E_with_asm = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
            wave_propagator='asm',
        )

        # Reset library default to 'asm' for a baseline ASM call.
        la.set_default_wave_propagator('asm')
        E_baseline_asm = la.apply_real_lens(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
        )

        # Per-call override matches baseline ASM regardless of the
        # library-wide default.
        np.testing.assert_array_equal(
            E_with_asm, E_baseline_asm,
            err_msg=("v5.1.0 per-call override regression: "
                     "wave_propagator='asm' must beat "
                     "set_default_wave_propagator('fresnel') -- the "
                     "resolver fires only when the kwarg is None."),
        )


class TestSetDefaultWavePropagatorSteersApplyRealLensTraced:
    """v5.1.0: same wiring contract for ``apply_real_lens_traced``."""

    def test_default_propagated_to_amplitude_leg(
            self, restore_default_wave_propagator):
        """``apply_real_lens_traced`` forwards ``wave_propagator`` to
        its inner amplitude-leg ``apply_real_lens`` call; the v5.1.0
        wiring must resolve ``wave_propagator=None`` BEFORE the
        forward, so the inner call also sees the library default."""
        import lumenairy as la

        E_in = np.ones((64, 64), dtype=np.complex128)
        pres = _minimal_prescription()

        # Drive the resolved kwarg into the inner amplitude leg.  We
        # don't need a numerical equality here; the source-level
        # pin in test_v4_16_3_agent_b.py
        # (test_wave_propagator_and_dy_now_have_consumers) covers
        # the structural side.  Here we functionally verify the
        # library default does NOT raise for a Fresnel choice.
        la.set_default_wave_propagator('fresnel')
        # Call with no explicit kwarg -- if the resolver works, the
        # amplitude leg uses Fresnel (a supported choice for the
        # inner apply_real_lens call).  Disable the
        # min_coarse_samples_per_aperture guard so the small test
        # grid is accepted.
        E_out = la.apply_real_lens_traced(
            E_in, prescription=pres,
            wavelength=633e-9, dx=2e-6,
            ray_subsample=4, parallel_amp=False,
            min_coarse_samples_per_aperture=0,
        )
        assert E_out.shape == E_in.shape
        # The traced result is finite at this grid.
        assert np.all(np.isfinite(E_out)), (
            "v5.1.0 functional pin: apply_real_lens_traced with the "
            "library wave_propagator default set to 'fresnel' "
            "produced non-finite output.")


# ============================================================================
# Wave-propagator knob -- consumer wiring at propagate_through_system.
# ============================================================================


class TestSetDefaultWavePropagatorSteersPropagateThroughSystem:
    """v5.1.0: ``propagate_through_system(... method=None)`` resolves
    via the library-wide default.  The legacy ``method='asm'`` is
    preserved bit-for-bit when the default is at its starting value."""

    def test_default_resolves_to_asm_on_fresh_import(
            self, restore_default_wave_propagator):
        import lumenairy as la

        la.set_default_wave_propagator('asm')

        E_in = np.ones((64, 64), dtype=np.complex128)
        elements = [{'type': 'propagate', 'z': 1e-3}]

        # No method= kwarg.
        E_default, _ = la.propagate_through_system(
            E_in, elements, wavelength=633e-9, dx=2e-6,
        )
        # Explicit method='asm'.
        E_explicit, _ = la.propagate_through_system(
            E_in, elements, wavelength=633e-9, dx=2e-6,
            method='asm',
        )
        np.testing.assert_array_equal(
            E_default, E_explicit,
            err_msg=("v5.1.0 back-compat regression: "
                     "propagate_through_system with no method= kwarg "
                     "must match method='asm' bit-for-bit when the "
                     "library default is 'asm'."),
        )

    def test_set_default_to_fresnel_changes_system_output(
            self, restore_default_wave_propagator):
        """``set_default_wave_propagator('fresnel')`` flips the
        free-space method for kwargless ``propagate_through_system``
        calls."""
        import lumenairy as la

        E_in = np.ones((64, 64), dtype=np.complex128)
        elements = [{'type': 'propagate', 'z': 1e-3}]

        la.set_default_wave_propagator('asm')
        E_asm, _ = la.propagate_through_system(
            E_in, elements, wavelength=633e-9, dx=2e-6,
        )
        la.set_default_wave_propagator('fresnel')
        E_default_fresnel, _ = la.propagate_through_system(
            E_in, elements, wavelength=633e-9, dx=2e-6,
        )

        assert not np.allclose(E_default_fresnel, E_asm), (
            "v5.1.0 functional pin: setting the library default to "
            "'fresnel' did not change propagate_through_system "
            "output vs ASM.  The setter is not steering method=.")

    def test_unsupported_default_raises_value_error(
            self, restore_default_wave_propagator):
        """``propagate_through_system`` natively supports only
        ``'asm' / 'sas' / 'fresnel'``.  If the library default is
        set to ``'rs'`` / ``'rayleigh_sommerfeld'`` (which the
        setter accepts), a kwargless call must raise ``ValueError``
        with a clear migration message rather than silently falling
        through to ASM."""
        import lumenairy as la

        la.set_default_wave_propagator('rs')

        E_in = np.ones((64, 64), dtype=np.complex128)
        elements = [{'type': 'propagate', 'z': 1e-3}]

        with pytest.raises(ValueError, match="propagate_through_system"):
            la.propagate_through_system(
                E_in, elements, wavelength=633e-9, dx=2e-6,
            )


# ============================================================================
# Dy knob -- consumer wiring at apply_real_lens / apply_real_lens_traced.
# ============================================================================


class TestSetDefaultDySteersApplyRealLens:
    """v5.1.0: ``set_default_dy(value)`` flips the default dy at
    ``apply_real_lens``; the chain is
    ``dy=None -> get_default_dy() -> dx``.  Bit-for-bit back-compat
    preserved when the default is ``None``."""

    def test_default_none_falls_back_to_dx(
            self, restore_default_dy):
        """Back-compat: ``set_default_dy(None)`` (the starting value)
        means ``dy=None`` at ``apply_real_lens`` resolves to ``dx``."""
        import lumenairy as la

        la.set_default_dy(None)

        E_in = np.ones((32, 32), dtype=np.complex128)
        pres = _minimal_prescription()
        dx = 2e-6

        # No dy= kwarg (resolves via default = None -> dx).
        E_default = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx,
        )
        # Explicit dy=dx (the legacy default).
        E_explicit = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx, dy=dx,
        )
        np.testing.assert_array_equal(
            E_default, E_explicit,
            err_msg=("v5.1.0 back-compat regression: "
                     "apply_real_lens with no dy= kwarg must "
                     "match dy=dx bit-for-bit when the library "
                     "default dy is None."),
        )

    def test_set_default_dy_to_value_steers_apply_real_lens(
            self, restore_default_dy):
        """Setting ``set_default_dy(<value>)`` to a value differing
        from ``dx`` should make a kwargless ``apply_real_lens`` call
        match an explicit ``dy=<value>`` call -- and differ from the
        ``dy=dx`` result (since anamorphic dy changes the y-axis
        FFT spacing in the in-glass propagation step).

        Uses a Gaussian input (spatial structure) so the anamorphic
        ASM step produces a numerically distinguishable result from
        the isotropic baseline.
        """
        import lumenairy as la

        N = 32
        dx = 2e-6
        # Gaussian input centred on the grid -- gives spatial
        # structure so anamorphic dy actually changes the in-glass
        # ASM output.
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        sigma = 8e-6
        E_in = np.exp(-(X**2 + Y**2) / (2 * sigma**2)).astype(np.complex128)

        pres = _minimal_prescription()
        dy_value = 4e-6  # 2x dx, anamorphic.

        # Baseline (dy=dx).
        la.set_default_dy(None)
        E_baseline = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx,
        )

        # Switch the default.
        la.set_default_dy(dy_value)
        E_default_dy = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx,
        )
        E_explicit_dy = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx, dy=dy_value,
        )

        np.testing.assert_array_equal(
            E_default_dy, E_explicit_dy,
            err_msg=("v5.1.0 functional pin: kwargless "
                     "apply_real_lens must use get_default_dy() "
                     "when no explicit dy= is passed."),
        )
        assert not np.allclose(E_default_dy, E_baseline), (
            "v5.1.0 functional pin: setting the library default dy "
            "to a non-dx value did not change the apply_real_lens "
            "output.  The setter is not steering the resolver.")

    def test_per_call_dy_overrides_library_default(
            self, restore_default_dy):
        """An explicit ``dy=<value>`` per-call kwarg overrides the
        library default."""
        import lumenairy as la

        E_in = np.ones((32, 32), dtype=np.complex128)
        pres = _minimal_prescription()
        dx = 2e-6

        # Set library default to a non-isotropic value, then pass
        # the isotropic-equivalent per-call -- the per-call kwarg wins.
        la.set_default_dy(4e-6)
        E_per_call_isotropic = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx, dy=dx,
        )

        la.set_default_dy(None)
        E_default_isotropic = la.apply_real_lens(
            E_in, prescription=pres, wavelength=633e-9, dx=dx,
        )
        np.testing.assert_array_equal(
            E_per_call_isotropic, E_default_isotropic,
            err_msg=("v5.1.0 per-call override regression: "
                     "dy=dx must beat set_default_dy(4e-6)."),
        )


class TestSetDefaultDyAtApplyRealLensTraced:
    """``apply_real_lens_traced`` requires square pixels (``dy ==
    dx``); the v5.1.0 wiring honours the library default but raises
    ``ValueError`` if the resolved ``dy`` is anamorphic."""

    def test_default_none_works_with_traced(self, restore_default_dy):
        import lumenairy as la

        la.set_default_dy(None)

        E_in = np.ones((32, 32), dtype=np.complex128)
        pres = _minimal_prescription()
        # Just verify the kwargless call doesn't raise the
        # anamorphic-pixel guard when the default is None.
        E_out = la.apply_real_lens_traced(
            E_in, prescription=pres, wavelength=633e-9, dx=2e-6,
            ray_subsample=4, parallel_amp=False,
            min_coarse_samples_per_aperture=0,
        )
        assert E_out.shape == E_in.shape

    def test_anamorphic_default_raises_traced(self, restore_default_dy):
        """If the library default dy differs from dx, the traced
        variant's square-pixel guard raises -- preserving the v4.x
        contract."""
        import lumenairy as la

        la.set_default_dy(5e-6)  # != dx below

        E_in = np.ones((32, 32), dtype=np.complex128)
        pres = _minimal_prescription()
        with pytest.raises(ValueError, match="square pixels"):
            la.apply_real_lens_traced(
                E_in, prescription=pres, wavelength=633e-9, dx=2e-6,
                ray_subsample=4, parallel_amp=False,
                min_coarse_samples_per_aperture=0,
            )


# ============================================================================
# Real-dtype knob -- v5.1.0 extension.
# ============================================================================


class TestSetDefaultRealDtypeSteersGeometricLensPhase:
    """v5.1.0 extension: the OPL accumulator in
    ``_geometric_lens_phase`` (called from inside
    ``apply_real_lens_traced``'s ``preserve_input_phase=True`` path)
    allocates a fresh real array with no complex parent.  v5.1.0
    wires it to ``get_default_real_dtype()`` so callers who set
    ``np.float32`` see the memory savings."""

    def test_geometric_lens_phase_honours_real_dtype(
            self, restore_default_real_dtype):
        """Source-level pin: the consumer wiring at
        ``_lens_traced.py`` must include a
        ``get_default_real_dtype`` call inside the
        ``_geometric_lens_phase`` body.

        Reads the source via :func:`_function_source` (fresh disk read,
        located by name) rather than ``inspect.getsource``.  See that
        helper's docstring: ``getsource`` slices at the import-time line
        number against a possibly-newer file, so it fails this pin
        whenever the tree is edited mid-run -- a false failure that
        looks exactly like a steering-global leak and is not one.
        """
        from lumenairy.elements import _lens_traced as _lt
        src = _function_source(_lt._geometric_lens_phase)
        # Guard the guard: if the extraction ever grabs the wrong block,
        # the marker assertion below must not be what tells us.
        assert src.lstrip().startswith('def _geometric_lens_phase('), (
            "_function_source did not return _geometric_lens_phase's own "
            "body; the pin below would be reading unrelated text.")
        assert 'get_default_real_dtype' in src, (
            "v5.1.0 wiring regression: _geometric_lens_phase no "
            "longer reads get_default_real_dtype.  Restore the "
            "wiring or update this pin (and the v5.1.0 release "
            "notes per-site rationale).")

    def test_geometric_lens_phase_returns_float32_when_default_is_float32(
            self, restore_default_real_dtype):
        """End-to-end behavioural pin: setting the default to
        ``np.float32`` flips the returned ``phase`` array's dtype.

        Uses the no-refraction prescription so the inner sag loop
        short-circuits before reaching ``_rt._surface_sag_xy`` (a
        pre-existing latent bug unrelated to this agent's scope --
        the v5.1.0 wiring under test is the ``np.zeros`` dtype line,
        which fires regardless of the per-surface loop body)."""
        import lumenairy as la
        from lumenairy.elements._lens_traced import _geometric_lens_phase

        pres = _trivial_no_refraction_prescription()
        la.set_default_real_dtype(np.float32)
        phase = _geometric_lens_phase(pres, wavelength=633e-9,
                                      dx=2e-6, N=32)
        assert phase.dtype == np.float32, (
            f"v5.1.0 functional pin: set_default_real_dtype(np.float32) "
            f"did not flip _geometric_lens_phase output dtype; got "
            f"{phase.dtype}.")

    def test_geometric_lens_phase_returns_float64_when_default_is_float64(
            self, restore_default_real_dtype):
        """Back-compat: with the default at its starting value
        (``np.float64``), the OPL accumulator dtype is unchanged."""
        import lumenairy as la
        from lumenairy.elements._lens_traced import _geometric_lens_phase

        pres = _trivial_no_refraction_prescription()
        la.set_default_real_dtype(np.float64)
        phase = _geometric_lens_phase(pres, wavelength=633e-9,
                                      dx=2e-6, N=32)
        assert phase.dtype == np.float64


# ============================================================================
# Retirement pin -- setters no longer emit the "API-only" UserWarning.
# ============================================================================


class TestSettersNoLongerEmitNoConsumerUserWarning:
    """v5.1.0 retired the v4.16.3 one-shot ``UserWarning`` from
    ``set_default_wave_propagator`` and ``set_default_dy``.  The
    setters must now be silent at the no-consumer-warning class."""

    def test_set_default_wave_propagator_is_silent(
            self, restore_default_wave_propagator):
        """Even with the latch reset, the setter must not emit the
        retired no-consumer UserWarning."""
        import lumenairy.propagators.propagation as _prop
        _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED = False
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                _prop.set_default_wave_propagator('fresnel')
            no_consumer = [w for w in ws
                           if issubclass(w.category, UserWarning)
                           and ('no library consumer' in str(w.message)
                                or 'API-only' in str(w.message))]
            assert no_consumer == [], (
                f"v5.1.0 retirement regression: "
                f"set_default_wave_propagator still emits the "
                f"retired no-consumer UserWarning: "
                f"{[str(w.message) for w in no_consumer]}")
        finally:
            _prop._DEFAULT_WAVE_PROPAGATOR_NO_CONSUMER_WARNED = True

    def test_set_default_dy_is_silent(self, restore_default_dy):
        import lumenairy.propagators.propagation as _prop
        _prop._DEFAULT_DY_NO_CONSUMER_WARNED = False
        try:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter("always")
                _prop.set_default_dy(5e-6)
            no_consumer = [w for w in ws
                           if issubclass(w.category, UserWarning)
                           and ('no library consumer' in str(w.message)
                                or 'API-only' in str(w.message))]
            assert no_consumer == [], (
                f"v5.1.0 retirement regression: set_default_dy still "
                f"emits the retired no-consumer UserWarning: "
                f"{[str(w.message) for w in no_consumer]}")
        finally:
            _prop._DEFAULT_DY_NO_CONSUMER_WARNED = True


# ============================================================================
# The save/restore contract itself -- this file steers PROCESS-GLOBAL knobs.
# ============================================================================


class TestSteeringKnobSaveRestoreIsExact:
    """Every test above steers a process-global knob, so the restore has
    to be exact or this file becomes a polluter of whatever runs next.

    The bar is IDENTITY, not equality (2026-08-09,
    ``docs/audits/FIX_STEERING_FAMILY_2026_08_09.md``).  The original
    fixtures restored by calling ``set_default_real_dtype(
    get_default_real_dtype())``, which looks like a no-op and is not:
    the setter normalises through ``np.dtype(...)``, so the shipped
    ``np.float64`` (a numpy scalar TYPE) came back as
    ``np.dtype('float64')`` (a dtype INSTANCE).  Equal, interchangeable
    in every ``np.zeros(..., dtype=...)`` call, and still a change this
    file made to the process and did not undo -- a per-test state-diff
    harness reported it as a leak from
    ``test_geometric_lens_phase_honours_real_dtype``.  Pin identity so
    the round-trip cannot quietly come back.
    """

    @pytest.mark.parametrize('attr,steered', [
        ('DEFAULT_WAVE_PROPAGATOR', 'fresnel'),
        ('DEFAULT_DY', 5e-6),
        ('DEFAULT_REAL_DTYPE', np.float32),
        ('DEFAULT_COMPLEX_DTYPE', np.complex64),
    ])
    def test_restored_puts_the_knob_back_identically(self, attr, steered):
        from lumenairy.propagators import fft_infra as _fft

        before = getattr(_fft, attr)
        with _restored(attr):
            setattr(_fft, attr, steered)
            assert getattr(_fft, attr) is steered, (
                f"{attr}: the steer itself did not take.")
        after = getattr(_fft, attr)
        assert after is before, (
            f"{attr}: _restored() did not put the knob back IDENTICALLY "
            f"({before!r} -> {after!r}).  Restore the module attribute "
            f"directly; do NOT round-trip through the public setter, "
            f"which renormalises dtype knobs.")

    def test_restored_puts_the_knob_back_when_the_body_raises(self):
        """A failing assertion inside a steering test must not leak."""
        from lumenairy.propagators import fft_infra as _fft

        before = _fft.DEFAULT_REAL_DTYPE
        with pytest.raises(RuntimeError):
            with _restored('DEFAULT_REAL_DTYPE'):
                _fft.DEFAULT_REAL_DTYPE = np.float32
                raise RuntimeError('simulated test failure')
        assert _fft.DEFAULT_REAL_DTYPE is before, (
            "_restored() leaked the real-dtype knob on the exception "
            "path; a red steering test would then red every later test "
            "that reads it.")

    def test_propagation_view_follows_the_fft_infra_attribute(self):
        """The restore targets ``fft_infra`` only.  That is complete
        because ``propagation`` deletes its import-time bindings and
        forwards through ``__getattr__`` -- pin that, so a future
        re-binding there cannot make the restores half-effective."""
        from lumenairy.propagators import fft_infra as _fft
        from lumenairy.propagators import propagation as _prop

        for attr in ('DEFAULT_WAVE_PROPAGATOR', 'DEFAULT_DY',
                     'DEFAULT_REAL_DTYPE', 'DEFAULT_COMPLEX_DTYPE'):
            assert getattr(_prop, attr) is getattr(_fft, attr), (
                f"propagation.{attr} no longer forwards to fft_infra; "
                f"the save/restore fixtures in this file would leave a "
                f"stale second copy behind.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
