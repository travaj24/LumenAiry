"""v4.15.3 Agent B tests -- algebra anamorphic SAS crash, Schell
stacklevel sweep, ``Source.gaussian_schell``/``schell_model`` sentinel
wiring, and ``_RETURN_KIND_UNSET`` subclass promotion.

This module pins the v4.15.3 closures for audit findings P1-NEW-F1-1,
P1-NEW-F1-2, P1-NEW-F1-4, and the P2 sentinel-subclass-promotion item
from ``docs/audits/AUDIT_V4_15_2_2026_05_18.md``.

v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path Schell
``DeprecationWarning`` is retired now that the v4.15.0 -> v4.15.1
return-shape change has had four subsequent releases of exposure.
``TestSchellDeprecationWarningStacklevel`` and the two
``TestSourceSchellClassmethodSentinel.test_*_default_emits_*`` methods
have been inverted from "warning fires" assertions to "warning does
NOT fire" assertions.  v5.30 (the W5 shim-removal wave) then removed the
apparatus ENTIRELY: the ``_warn_schell_return_kind_default`` helper,
``_RETURN_KIND_UNSET`` and ``_SchellReturnKindUnsetSentinel`` are no
longer importable, and the pins below assert that removal.  (This
paragraph used to say they were "preserved for back-compat"; that was
true only up to v4.16.1.)  The
``TestLibraryWideStacklevelSweep`` (5 Source.* legacy-positional
classmethods) and ``TestReturnKindUnsetSubclassSentinel`` test
classes are unchanged -- they test orthogonal back-compat surfaces.

Scope
-----

* **B.1 -- P1-NEW-F1-1:** ``FreeSpace._apply`` (and
  ``FourierTransform._apply`` by composition) must forced
  ``method='asm'`` when the input grid is anamorphic
  (``dy != dx``) and the user did not pick a specific method
  (``self.method == 'auto'``).  Pre-v4.15.3 the dispatcher's
  far-field regime routed to SAS in that case, but SAS does NOT
  accept the ``dy`` kwarg the v4.15.2 closure threads in -- the
  call raised
  ``TypeError: sas_propagate() got an unexpected keyword argument
  'dy'``.

* **B.2 -- P1-NEW-F1-2:** ``_warn_schell_return_kind_default``
  invoked ``warn_deprecated_signature(stacklevel=4)``, but the
  call chain has 5 internal frames (``warnings.warn`` -> ``_emit``
  -> ``warn_deprecated_signature`` -> ``_warn_schell_return_kind_default``
  -> factory body -> user).  v4.15.3 bumps to ``stacklevel=5``.
  Also sweeps the 5 ``Source.*`` classmethods that called
  ``warn_deprecated_signature(stacklevel=3)`` (off by one for the
  same reason at one level less of indirection).

* **B.3 -- P1-NEW-F1-4:** ``Source.gaussian_schell`` /
  ``Source.schell_model`` hardcoded ``return_kind: str =
  'ensemble'`` and bypassed the ``_RETURN_KIND_UNSET`` sentinel
  path used by the 3 top-level factories.  Pre-v4.15.0 callers
  using these classmethods got a 4-tuple silently with no
  ``DeprecationWarning``.  v4.15.3 routes both classmethods
  through the same sentinel logic.

* **B.4 -- P2-NEW:** ``_RETURN_KIND_UNSET`` was a bare
  ``_Sentinel`` instance, inconsistent with the
  ``_ZeroApertureMaskSentinel`` / ``_AngleUnsetSentinel`` /
  ``_NoDefaultSentinel`` subclass pattern.  v4.15.3 promotes it to
  a dedicated ``_SchellReturnKindUnsetSentinel(_Sentinel)``
  subclass; pickle round-trip via ``_SENTINEL_REGISTRY`` is
  preserved.

Library-wide ``_warn_*`` helper sweep (audit P1-NEW-F1-2 / Tier-1)
------------------------------------------------------------------

Grep result for ``def _warn_`` / ``def warn_deprecated_`` in
``lumenairy/``:

  * ``lumenairy/_deprecation.py``:
      - ``warn_deprecated_kwarg`` (default ``stacklevel=3``)
      - ``warn_deprecated_alias`` (default ``stacklevel=3``)
      - ``warn_deprecated_default`` (default ``stacklevel=3``)
      - ``warn_deprecated_signature`` (default ``stacklevel=3``)
      - ``warn_renamed_function`` (default ``stacklevel=3``)
  * ``lumenairy/propagators/system.py``:
      - ``_warn_legacy`` (nested in ``_resolve_aperture_params``;
        direct ``warnings.warn(stacklevel=3)``)
  * ``lumenairy/glass.py``:
      - ``_warn_missing_kappa_once`` (uses ``RuntimeWarning``,
        ``stacklevel=3``)
  * ``lumenairy/sources/core.py``:
      - ``_warn_schell_return_kind_default``
        (``stacklevel=4`` -> v4.15.3 bumped to ``stacklevel=5``,
        chain has 5 internal frames)
  * ``lumenairy/elements/lenses.py``:
      - ``_warn_if_aperture_exceeds_grid`` (direct
        ``warnings.warn(stacklevel=3)``; analysed -- stacklevel=3
        is correct because the helper calls ``warnings.warn``
        directly without going through ``_emit``).

Adjustments made in v4.15.3 (all in ``lumenairy/sources/core.py``):

  - L1721: ``_warn_schell_return_kind_default``: 4 -> 5.  Chain:
    user -> factory -> helper -> ``warn_deprecated_signature`` ->
    ``_emit`` -> ``warnings.warn``.
  - L1203: ``create_led_source`` legacy-positional branch: 3 -> 4.
    Chain: user -> factory body -> ``warn_deprecated_signature`` ->
    ``_emit`` -> ``warnings.warn``.
  - L2368: ``Source.gaussian`` legacy-positional branch: 3 -> 4.
    Same chain as ``create_led_source``.
  - L2451: ``Source.plane_wave``: 3 -> 4.
  - L2525: ``Source.point_source``: 3 -> 4.
  - L2596: ``Source.top_hat``: 3 -> 4.
  - L2682: ``Source.fiber_mode``: 3 -> 4.

Cases analysed but NOT adjusted (outside scope or
not-genuinely-off-by-one):

  - ``elements/lenses.py:_warn_if_aperture_exceeds_grid``
    (``stacklevel=3``): genuinely correct; the helper calls
    ``warnings.warn`` directly (no ``_emit`` wrapping), so chain
    has only 2 internal frames (helper + apply_real_lens body) +
    user code at stacklevel=3.
  - ``elements/_lens_traced.py`` and ``propagators/asymptotic.py``
    inline ``warnings.warn(stacklevel=3)``: out of v4.15.3 scope;
    these are direct inline warns inside named functions, not
    helper-wrapper indirection patterns.  Whether ``stacklevel=3``
    or ``stacklevel=2`` is correct depends on whether the named
    function is typically called via dispatch or directly by the
    user -- this is dispatcher-context-dependent and outside the
    Agent-B scope.
  - ``system.py:_warn_legacy`` (nested in
    ``_resolve_aperture_params``): may have an off-by-one but the
    chain depth depends on deeply-nested
    ``propagate_through_system_jax`` dispatch; conservative skip
    out of scope.

Author: Andrew Traverso -- v4.15.3 / Agent B
"""

from __future__ import annotations

import pickle
import re
import warnings
from typing import Any

import numpy as np
import pytest

import lumenairy as la
from lumenairy.algebra import (
    FourierTransform,
    FreeSpace,
    Magnify,
    ThinLens,
)
from lumenairy.sources.core import (
    PartialCoherenceMCF,
    Source,
    create_annular_incoherent_source,
    create_gaussian_schell_source,
    create_schell_model_source,
)

# v5.30 (W5 shim-removal wave): ``_RETURN_KIND_UNSET`` and
# ``_SchellReturnKindUnsetSentinel`` are no longer importable -- the whole
# return_kind sentinel apparatus was removed.  See
# ``TestReturnKindUnsetSubclassSentinel`` below for the supersession record.

# Shared parameter bundles for Schell-family tests.
_GAUSSIAN_SCHELL_KW = dict(
    N=16, dx=2e-6, wavelength=633e-9,
    w0=20e-6, sigma_g=8e-6,
    n_realizations=4, rng=0,
)

_SCHELL_MODEL_KW = dict(
    N=16, dx=2e-6, wavelength=633e-9,
    intensity_profile=np.exp(
        -((np.linspace(-1, 1, 16)[:, None]) ** 2
          + (np.linspace(-1, 1, 16)[None, :]) ** 2) / 0.25),
    coherence_length=8e-6,
    n_realizations=4, rng=0,
)


# ============================================================================
# B.1 -- P1-NEW-F1-1: FreeSpace + FourierTransform SAS-anamorphic crash
# ============================================================================


class TestFreeSpaceAnamorphicSAS:
    """v4.15.3 (audit P1-NEW-F1-1) closure: ``FreeSpace._apply`` must
    not crash with ``TypeError: sas_propagate() got an unexpected
    keyword argument 'dy'`` when called on an anamorphic grid
    (``dy != dx``) in the far-field regime that the v4.15.2 closure
    nominally pushed to SAS.

    Fix: when ``dy != dx`` and ``self.method == 'auto'``, force
    ``method='asm'`` because SAS is square-grid-only.  Keeps
    ``method='auto'`` as a hint rather than a contract.
    """

    @staticmethod
    def _make_anamorphic_field(N=32, dx=1e-6, dy=2e-6, wavelength=633e-9):
        """Build a small anamorphic Gaussian field for round-trip
        sanity.  Coordinates honour the (dx, dy) anisotropy."""
        ny, nx = N, N
        x = (np.arange(nx) - nx / 2) * dx
        y = (np.arange(ny) - ny / 2) * dy
        X, Y = np.meshgrid(x, y)
        # Use a Gaussian whose 1/e^2 radius is small enough to be
        # well-contained on the smaller-pitch axis.
        w0 = 0.2 * min(nx * dx, ny * dy)
        E = np.exp(-(X * X + Y * Y) / (w0 * w0)).astype(np.complex128)
        return E, dx, dy, wavelength

    def _q_far_field_distance(self, N, dx, wavelength, q_target=4.0):
        """Pick ``z`` so that the grid Fresnel ratio ``Q = lambda * z /
        (N * dx**2)`` is well above 1 (where the v4.15.2 dispatcher
        routes to SAS for square grids)."""
        return q_target * N * dx * dx / wavelength

    def test_freespace_anamorphic_auto_method_no_crash(self):
        """``FreeSpace(d, method='auto')._apply`` on an anamorphic grid
        in the far-field regime must succeed (not raise the SAS
        ``TypeError: sas_propagate() got an unexpected keyword
        argument 'dy'``).  Output must be finite and shape-preserving.
        """
        E, dx, dy, wavelength = self._make_anamorphic_field(
            N=32, dx=1e-6, dy=2e-6, wavelength=633e-9)
        d = self._q_far_field_distance(
            N=E.shape[0], dx=dx, wavelength=wavelength, q_target=4.0)
        op = FreeSpace(d, method='auto')
        # Pre-v4.15.3 this raised TypeError(... unexpected keyword
        # argument 'dy').  Post-v4.15.3 it must succeed.
        E_out, dx_out, dy_out = op._apply(
            E, dx=dx, dy=dy, wavelength=wavelength)
        assert E_out.shape == E.shape
        assert np.all(np.isfinite(np.abs(E_out)))
        # ASM is shape- AND pitch-preserving (no SAS-style rescale),
        # so dx_out == dx, dy_out == dy at the output plane.
        assert np.isclose(dx_out, dx, rtol=1e-12)
        assert np.isclose(dy_out, dy, rtol=1e-12)

    def test_freespace_anamorphic_forces_asm_when_dy_ne_dx(self,
                                                            monkeypatch):
        """Inspect the routed method by monkey-patching the dispatcher:
        when ``self.method='auto'`` and ``dy != dx``, the FreeSpace
        ``_apply`` must call ``propagate(..., method='asm')`` rather
        than ``method='auto'``.
        """
        captured = {}

        from lumenairy.propagators import dispatch as _dispatch_mod

        def _spy_propagate(E_in, *, z, wavelength, dx, method,
                            prescription=None, accuracy='balanced',
                            output_grid=None, output_dx=None,
                            return_result=False, **kwargs):
            captured['method'] = method
            captured['kwargs'] = dict(kwargs)
            captured['dx'] = dx
            captured['z'] = z
            # Return a trivial passthrough so the rest of the call
            # path doesn't blow up.
            return E_in

        monkeypatch.setattr(
            _dispatch_mod, 'propagate', _spy_propagate)

        E, dx, dy, wavelength = self._make_anamorphic_field()
        d = self._q_far_field_distance(
            N=E.shape[0], dx=dx, wavelength=wavelength, q_target=4.0)
        op = FreeSpace(d, method='auto')
        op._apply(E, dx=dx, dy=dy, wavelength=wavelength)
        assert captured['method'] == 'asm', (
            f"FreeSpace anamorphic auto must force method='asm', got "
            f"method={captured['method']!r}.")
        assert 'dy' in captured['kwargs'], (
            "FreeSpace anamorphic must forward dy in **kwargs; got "
            f"kwargs={captured['kwargs']!r}.")
        assert float(captured['kwargs']['dy']) == float(dy)

    def test_freespace_square_grid_auto_still_picks_sas_in_far_field(
        self, monkeypatch,
    ):
        """Regression pin: when ``dy == dx``, the v4.15.3 fix MUST NOT
        force ``method='asm'`` -- the dispatcher's regime-dependent
        SAS pick on square grids in the far-field regime is the
        v4.15.2 design we don't want to regress.

        Verify by inspecting that the dispatcher receives
        ``method='auto'`` (and then routes internally to SAS).
        """
        captured = {}

        from lumenairy.propagators import dispatch as _dispatch_mod

        def _spy_propagate(E_in, *, z, wavelength, dx, method,
                            prescription=None, accuracy='balanced',
                            output_grid=None, output_dx=None,
                            return_result=False, **kwargs):
            captured['method'] = method
            captured['kwargs'] = dict(kwargs)
            return E_in

        monkeypatch.setattr(
            _dispatch_mod, 'propagate', _spy_propagate)

        N = 32
        dx = 1e-6
        wavelength = 633e-9
        E = np.ones((N, N), dtype=np.complex128)
        d = self._q_far_field_distance(
            N=N, dx=dx, wavelength=wavelength, q_target=4.0)
        op = FreeSpace(d, method='auto')
        # Square grid: dy == dx.
        op._apply(E, dx=dx, dy=dx, wavelength=wavelength)
        assert captured['method'] == 'auto', (
            f"FreeSpace square-grid auto must keep method='auto' "
            f"(letting the dispatcher pick SAS in the far-field); "
            f"got method={captured['method']!r}.")
        # dx == dy case: dy kwarg is NOT forwarded (the v4.15.2
        # closure only adds it when dy != dx).
        assert 'dy' not in captured['kwargs'], (
            f"FreeSpace square-grid auto must NOT forward dy; got "
            f"kwargs={captured['kwargs']!r}.")

    def test_freespace_explicit_asm_anamorphic_no_crash(self):
        """Explicit ``method='asm'`` on an anamorphic grid was already
        fine in v4.15.2; pin it as a sanity check that the v4.15.3
        fix doesn't disturb the explicit-method path.
        """
        E, dx, dy, wavelength = self._make_anamorphic_field()
        d = self._q_far_field_distance(
            N=E.shape[0], dx=dx, wavelength=wavelength, q_target=4.0)
        op = FreeSpace(d, method='asm')
        E_out, dx_out, dy_out = op._apply(
            E, dx=dx, dy=dy, wavelength=wavelength)
        assert E_out.shape == E.shape
        assert np.all(np.isfinite(np.abs(E_out)))

    def test_fourier_transform_anamorphic_no_crash(self):
        """``FourierTransform._apply`` invokes two ``FreeSpace(f)``
        legs internally with ``method='auto'``.  v4.15.3: the fix in
        ``FreeSpace._apply`` propagates through, so an anamorphic
        input field no longer crashes the FT chain in the far-field
        regime where the v4.15.2 closure pushed both legs into SAS.
        """
        E, dx, dy, wavelength = self._make_anamorphic_field(
            N=32, dx=1e-6, dy=2e-6, wavelength=633e-9)
        # Pick f so each FreeSpace(f) leg sits in the far-field
        # regime: Q = lambda*f / (N*dx^2) >> 1.
        f = self._q_far_field_distance(
            N=E.shape[0], dx=dx, wavelength=wavelength, q_target=4.0)
        op = FourierTransform(f)
        E_out, dx_out, dy_out = op._apply(
            E, dx=dx, dy=dy, wavelength=wavelength)
        assert E_out.shape == E.shape
        assert np.all(np.isfinite(np.abs(E_out)))


# ============================================================================
# B.2 -- P1-NEW-F1-2: Schell DeprecationWarning stacklevel one short
#
# v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
# DeprecationWarning was retired now that the v4.15.0 -> v4.15.1
# return-shape change has had multiple releases of exposure.  The
# tests below were originally written to verify the warning's
# ``stacklevel`` lands at user code; v4.16.1 inverts them to assert
# the warning no longer fires on the default path.  The underlying
# ``_warn_schell_return_kind_default`` helper was kept importable at
# v4.16.1 but REMOVED at v5.30 -- see ``test_warn_helper_is_removed``
# below, which now asserts its absence.
# ============================================================================


class TestSchellDeprecationWarningStacklevel:
    """v4.16.1 contract: the Schell default-return-kind path must NOT
    emit the historic v4.15.0 -> v4.15.1 return-shape
    DeprecationWarning.  The pre-v4.16.1 contract (warning fires with
    ``stacklevel=5`` pointing at user code) was retired in line with
    audit item 6 once the new ensemble contract had multiple releases
    of exposure.  Pin the new contract so a re-introduction of the
    warning is loud.
    """

    def test_gaussian_schell_default_no_deprecation_warning(self):
        """Call ``create_gaussian_schell_source`` without
        ``return_kind``.  No ``return_kind`` DeprecationWarning may
        fire; the call returns the canonical 4-tuple ensemble form.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = create_gaussian_schell_source(**_GAUSSIAN_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_gaussian_schell_source' in str(w.message)
               and 'return_kind' in str(w.message)]
        assert dep == [], (
            f"v4.16.1: default-return_kind call must NOT emit "
            f"DeprecationWarning; got "
            f"{[str(w.message) for w in dep]}.")
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_schell_model_default_no_deprecation_warning(self):
        """Same contract for ``create_schell_model_source``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = create_schell_model_source(**_SCHELL_MODEL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_schell_model_source' in str(w.message)
               and 'return_kind' in str(w.message)]
        assert dep == []
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_warn_helper_is_removed(self):
        """SUPERSEDES ``test_warn_helper_still_callable_for_back_compat``
        (v4.16.1).

        v4.16.1 kept ``_warn_schell_return_kind_default`` importable after
        retiring its call sites, on the theory that an external custom
        Schell wrapper might invoke it.  v5.30 (W5) removes it: the
        ``test_niche_audit_w3_ui_deprecation`` reachability pin had
        MEASURED zero production call sites, and the helper's only
        remaining behaviour was to emit a banner for a transition that
        completed five releases earlier."""
        import lumenairy.sources.core as core
        assert not hasattr(core, '_warn_schell_return_kind_default')


# ============================================================================
# B.2 (continued) -- library-wide stacklevel sweep
# ============================================================================


class TestLibraryWideStacklevelSweep:
    """v4.15.3 (audit P1-NEW-F1-2 Tier-1): parametrised sweep over
    the ``Source.*`` legacy-positional classmethods that share the
    same warn-helper-indirection pattern.  Each classmethod's
    DeprecationWarning must point at user code, not at the
    classmethod body.

    Frame chain (each classmethod calls ``warn_deprecated_signature``
    directly):
      1. ``warnings.warn``
      2. ``_emit`` body
      3. ``warn_deprecated_signature`` body
      4. Classmethod body
      5. User code  <-- target

    Required ``stacklevel`` value = 4.  Pre-v4.15.3 these used
    ``stacklevel=3`` which landed at frame 3 -- the
    ``warn_deprecated_signature`` body.
    """

    # Parametrised list of legacy-positional invocations and the
    # function-name token expected to appear in the warning message.
    # Each is the documented v4.15-deprecated legacy positional call
    # form.  We craft a minimal valid positional invocation per
    # classmethod (the canonical kwarg-only signatures all expect
    # specific defaults the classmethod fills in).
    LEGACY_INVOCATIONS = [
        ('Source.gaussian',
         lambda: la.Source.gaussian(50e-6, 16, 2e-6, 633e-9)),
        ('Source.plane_wave',
         lambda: la.Source.plane_wave(16, 2e-6, 633e-9)),
        ('Source.point_source',
         lambda: la.Source.point_source(16, 2e-6, 633e-9)),
        ('Source.top_hat',
         lambda: la.Source.top_hat(50e-6, 16, 2e-6, 633e-9)),
        ('Source.fiber_mode',
         lambda: la.Source.fiber_mode(50e-6, 16, 2e-6, 633e-9)),
    ]

    @pytest.mark.parametrize('label,call', LEGACY_INVOCATIONS,
                              ids=[lbl for lbl, _ in LEGACY_INVOCATIONS])
    def test_classmethod_legacy_positional_raises_at_user_code(
            self, label, call):
        """SUPERSEDES ``test_classmethod_legacy_positional_stacklevel``
        (v4.15.3 P1-NEW-F1-2).

        The original pin fixed the warning's ``stacklevel`` so the
        DeprecationWarning was attributed to the user's call site rather
        than to ``sources/core.py``.  v5.30 (W5) removes the shim, so
        there is no warning to place -- but the equivalent property still
        matters and is still checkable: the ``TypeError`` traceback's
        LAST user frame must be this file, i.e. the rejection happens at
        the classmethod boundary and does not leak a deep library stack.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(TypeError) as info:
                call()
        assert label in str(info.value), str(info.value)
        assert not [w for w in caught
                    if issubclass(w.category, DeprecationWarning)], (
            label, [str(w.message) for w in caught])
        # Walk to the deepest frame: the rejection helper lives one frame
        # inside the classmethod, so the library depth is bounded at 2.
        tb, depth = info.tb, 0
        frames = []
        while tb is not None:
            frames.append(tb.tb_frame.f_code.co_filename)
            tb = tb.tb_next
            depth += 1
        norm = [f.replace('\\', '/').lower() for f in frames]
        assert __file__.replace('\\', '/').lower() in norm[0], norm
        lib = [f for f in norm if 'lumenairy/sources/core.py' in f]
        assert len(lib) <= 2, (
            f"{label}: the rejection must fire at the classmethod "
            f"boundary, not deep in the library; frames={norm}")


# ============================================================================
# B.3 -- P1-NEW-F1-4: Source.gaussian_schell/schell_model route through sentinel
# ============================================================================


class TestSourceSchellClassmethodSentinel:
    """v4.15.3 (audit P1-NEW-F1-4): the two ``Source.*`` Schell
    classmethods (``gaussian_schell``, ``schell_model``) route
    through the same ``_RETURN_KIND_UNSET`` sentinel logic as the 3
    top-level factories.

    v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
    DeprecationWarning emitted by the classmethods on the sentinel
    path is retired in line with the top-level factories.  The two
    tests below were originally written to assert the warning fires;
    v4.16.1 inverts them to pin the new silent contract.  The
    ensemble-tuple return shape pin is unchanged.
    """

    def test_source_gaussian_schell_default_no_deprecation_warning(
        self,
    ):
        """v4.16.1: ``Source.gaussian_schell(...)`` without
        ``return_kind`` must NOT emit the historic v4.15.2-era
        return-shape DeprecationWarning.  The call still returns the
        canonical 4-tuple ensemble form (NOT a Source-wrapped 3-D
        field)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.Source.gaussian_schell(**_GAUSSIAN_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'Source.gaussian_schell' in str(w.message)
               and 'return_kind' in str(w.message)]
        assert dep == [], (
            f"v4.16.1: default-return_kind classmethod call must NOT "
            f"emit DeprecationWarning; got "
            f"{[str(w.message) for w in dep]}.")
        # Behaviour pin: still get the 4-tuple ensemble return.
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_source_schell_model_default_no_deprecation_warning(self):
        """Same contract for ``Source.schell_model``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.Source.schell_model(**_SCHELL_MODEL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'Source.schell_model' in str(w.message)
               and 'return_kind' in str(w.message)]
        assert dep == []
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_source_gaussian_schell_explicit_ensemble_no_warning(self):
        """Explicit ``return_kind='ensemble'`` -- no DeprecationWarning
        should fire from the Source.gaussian_schell wrapper or from
        the underlying top-level factory."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.Source.gaussian_schell(
                return_kind='ensemble', **_GAUSSIAN_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and ('return_kind' in str(w.message)
                    or 'Source.gaussian_schell' in str(w.message))]
        assert dep == [], (
            f"Explicit return_kind='ensemble' must NOT emit a "
            f"DeprecationWarning; got "
            f"{[str(w.message) for w in dep]}.")

    def test_source_gaussian_schell_explicit_mcf_no_warning(self):
        """Explicit ``return_kind='mcf'`` -- no warning, return
        bare PartialCoherenceMCF."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = la.Source.gaussian_schell(
                return_kind='mcf', **_GAUSSIAN_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and ('return_kind' in str(w.message)
                    or 'Source.gaussian_schell' in str(w.message))]
        assert dep == []
        assert isinstance(result, PartialCoherenceMCF)

    def test_source_gaussian_schell_docstring_mentions_3d_e_invariant_break(
        self,
    ):
        """v4.15.3 (audit B.3 docstring requirement): the
        ``Source.gaussian_schell`` docstring must explicitly mention
        the 3-D ensemble shape (breaking the 2-D ``Source.E``
        invariant) and point at the future ``Source.realizations()``
        API.  This sets caller expectations honestly -- the audit
        labelled this a "soft inconsistency that is HONEST"."""
        doc = la.Source.gaussian_schell.__doc__ or ''
        assert 'invariant' in doc.lower(), (
            f"Source.gaussian_schell docstring must mention the "
            f"2-D Source.E invariant break; got docstring excerpt: "
            f"{doc[:200]!r}...")
        # Must call out the (n_realizations, Ny, Nx) shape so the
        # caller can see why iteration is required.
        assert ('n_realizations, Ny, Nx' in doc
                or '(n_realizations' in doc), (
            f"Source.gaussian_schell docstring must call out the "
            f"3-D ensemble shape (n_realizations, Ny, Nx); got "
            f"docstring excerpt: {doc[:400]!r}...")

    def test_source_schell_model_docstring_mentions_3d_e_invariant_break(
        self,
    ):
        """Same docstring requirement for ``Source.schell_model``."""
        doc = la.Source.schell_model.__doc__ or ''
        assert 'invariant' in doc.lower(), (
            f"Source.schell_model docstring must mention the 2-D "
            f"Source.E invariant break; got docstring excerpt: "
            f"{doc[:200]!r}...")


# ============================================================================
# B.4 -- P2-NEW: _RETURN_KIND_UNSET subclass promotion
# ============================================================================


class TestReturnKindUnsetSubclassSentinel:
    """SUPERSEDES the five v4.15.3 ``_RETURN_KIND_UNSET`` pins
    (``..._is_subclass_instance``, ``..._pickle_round_trip``,
    ``..._bool_false``, ``..._repr_matches_name``,
    ``..._registry_singleton``).

    v4.15.3 (audit P2-NEW) promoted ``_RETURN_KIND_UNSET`` from a bare
    ``_Sentinel`` instance to a dedicated
    ``_SchellReturnKindUnsetSentinel(_Sentinel)`` subclass, and those five
    pins fixed its identity / pickle / repr / registry contract.

    v5.30 (W5 shim-removal wave) removes the sentinel entirely: it existed
    only to detect "caller did not pass ``return_kind``" for the v4.15.0 ->
    v4.15.1 return-shape ``DeprecationWarning``, that warning was retired
    in v4.16.1, and the five remaining branches were no-ops.  There is no
    object left to have an identity contract.  The sentinel PATTERN those
    pins guarded is still exercised by the other five ``_Sentinel``
    subclasses via the auto-discovering walker in
    ``tests/unit/test_v5_2_walker_sentinel_reduce.py``, so the contract
    itself loses no coverage.
    """

    def test_sentinel_and_subclass_are_not_importable(self):
        import lumenairy.sources.core as core
        for name in ('_RETURN_KIND_UNSET',
                     '_SchellReturnKindUnsetSentinel'):
            assert not hasattr(core, name), name
        with pytest.raises(ImportError):
            from lumenairy.sources.core import (  # noqa: F401
                _RETURN_KIND_UNSET as _probe,
            )

    def test_registry_no_longer_holds_the_singleton(self):
        """The name-keyed ``_SENTINEL_REGISTRY`` entry goes with the
        class -- a stale entry would let an unpickle silently resurrect
        a sentinel whose meaning no longer exists.

        (The registry is populated at ``_Sentinel.__init__``, so with the
        subclass gone nothing registers the name.  A leftover entry would
        mean a copy of the class survived somewhere.)"""
        from lumenairy._deprecation import _SENTINEL_REGISTRY
        assert '_SCHELL_RETURN_KIND_UNSET' not in _SENTINEL_REGISTRY, (
            'a _SchellReturnKindUnsetSentinel is still being constructed '
            'somewhere')

    def test_stale_string_sentinel_gets_the_modern_error(self):
        """A caller who forwarded the removed sentinel now lands on
        ``_validate_return_kind``'s existing ``ValueError``, which already
        names the modern values -- which is why the removal needed no
        bespoke rejection branch."""
        with pytest.raises(ValueError, match="'ensemble' or 'mcf'"):
            create_gaussian_schell_source(
                return_kind=object(), **_GAUSSIAN_SCHELL_KW)
