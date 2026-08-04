"""v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion).

Pins:

* the bundled ``_polynomial_index`` evaluator's mathematical correctness
  + scalar/array (xp-dispatch) behaviour;
* the 24-glass stub manifest in ``_POLYNOMIAL_STUB_NAMES`` matches the
  catalogue tuples in ``GLASS_REGISTRY`` (auto-enumerated -- never
  hardcoded so a future ingestion PR doesn't have to update two
  places);
* the documented migration path: looking up a stubbed glass on a
  minimal install (no ``refractiveindex`` package) either returns a
  finite float (if coefficients were ingested between this test's
  write date and the user's checkout) OR raises ``NotImplementedError``
  pointing the user at the [glass] extras / v5.2.1 issue tracker.

v5.2 ROADMAP note: the n_d cross-check tolerance for any glass with
real ingested coefficients is 5e-5 (may relax for BK7-like polynomial
fits per the V3 v4.16.2 note).

STATE AS OF 2026-08-03 (the v5.2 text above described the ship state and
has been corrected here).  The ingestion COMPLETED: measured in this tree,
``POLYNOMIAL_COEFFICIENTS`` holds all 24 entries and
``_POLYNOMIAL_STUB_NAMES`` is EMPTY.  Two consequences worth knowing before
reading a green run as coverage:

* the cross-check arm guarded by ``if POLYNOMIAL_COEFFICIENTS`` now has
  full teeth -- it iterates all 24 glasses;
* the four ``TestPolynomialStubManifest`` tests below iterate or intersect
  ``_POLYNOMIAL_STUB_NAMES`` and therefore pass VACUOUSLY over an empty
  set.  They are deliberately retained as forward guards: they regain
  teeth the moment a glass is re-stubbed.  They are not evidence that the
  manifest is correct today, because there is no manifest today.
"""
from __future__ import annotations

import importlib.util as _importlib_util

import numpy as np
import pytest

from lumenairy import glass as _glass
from lumenairy.glass import (
    _POLYNOMIAL_STUB_NAMES,
    GLASS_REGISTRY,
    GLASS_VALIDITY,
    POLYNOMIAL_COEFFICIENTS,
    SELLMEIER_COEFFICIENTS,
    _polynomial_index,
    get_glass_index,
)

_JAX_AVAILABLE = _importlib_util.find_spec('jax') is not None


# ============================================================================
# _polynomial_index -- mathematical correctness
# ============================================================================

class TestPolynomialIndexMath:
    """Pin formula-3 evaluator math beyond the v4.16.2 smoke checks."""

    def test_six_coefficient_set_closed_form(self):
        """Closed-form check against ``[1.0, 0.5, -2, 0.3, -4]`` style
        coefficients (the layout described in the v5.2 task brief).

        Pick lam = 2.0 um so the powers are exact rationals::

            n^2 = 1.0 + 0.5 * 2^-2 + 0.3 * 2^-4
                = 1.0 + 0.125 + 0.01875
                = 1.14375
            n   = sqrt(1.14375) = 1.069462...
        """
        coeffs = (1.0, [(0.5, -2.0), (0.3, -4.0)])
        n = _polynomial_index(2.0e-6, coeffs)
        expected = float(np.sqrt(1.0 + 0.5 * 2.0**-2 + 0.3 * 2.0**-4))
        assert n == pytest.approx(expected, abs=1e-12)

    def test_dimensional_unit_check_micrometers(self):
        """Lambda enters in micrometers, NOT metres.  A polynomial of
        ``n^2 = lam^2`` at lam = 1 um should give n = 1.0, NOT 1e-12."""
        coeffs = (0.0, [(1.0, 2.0)])  # n^2 = lam_um^2
        n = _polynomial_index(1.0e-6, coeffs)
        assert n == pytest.approx(1.0, abs=1e-12)

    def test_glass_name_in_error_message(self):
        """When the polynomial extrapolates to n^2 <= 0, the error
        must name the offending glass for downstream debugging."""
        coeffs = (-1.0, [])
        with pytest.raises(ValueError, match="MY_GLASS"):
            _polynomial_index(587.6e-9, coeffs, glass_name='MY_GLASS')


# ============================================================================
# _polynomial_index -- xp-dispatch / vectorisation
# ============================================================================

class TestPolynomialIndexXpDispatch:
    """Pin the scalar/array contract: scalar in -> float out; array in
    -> ndarray out with the same shape.

    Mirrors the convention of the sibling ``_sellmeier_index`` which is
    scalar-only at v5.2.  The polynomial evaluator extends the contract
    one step further (vector lambda) because the SciPy-style array
    parametric sweep across a spectrum is more naturally vectorised on
    a polynomial than on the 3-pole Sellmeier (the latter has the
    resonance-coincidence check per-pole).
    """

    def test_scalar_input_returns_python_float(self):
        coeffs = (2.25, [])
        n = _polynomial_index(587.6e-9, coeffs)
        assert isinstance(n, float)
        assert n == pytest.approx(1.5, abs=1e-12)

    def test_numpy_array_input_returns_numpy_array(self):
        coeffs = (2.25, [])
        lam = np.array([400e-9, 500e-9, 600e-9, 700e-9])
        n = _polynomial_index(lam, coeffs)
        assert isinstance(n, np.ndarray)
        assert n.shape == lam.shape
        np.testing.assert_allclose(n, 1.5, atol=1e-12)

    def test_numpy_array_dispersive(self):
        """A non-constant polynomial must produce a non-constant n
        across the wavelength array."""
        # Same coefficients as test_six_coefficient_set_closed_form
        coeffs = (1.0, [(0.5, -2.0), (0.3, -4.0)])
        lam = np.array([1.0e-6, 1.5e-6, 2.0e-6])
        n = _polynomial_index(lam, coeffs)
        # Verify each element against the scalar evaluator.
        for i, lam_i in enumerate(lam):
            assert n[i] == pytest.approx(
                _polynomial_index(float(lam_i), coeffs), abs=1e-12)

    def test_list_input_lifted_via_numpy(self):
        """Python-list input should lift via ``np.asarray`` rather than
        crashing -- the typical user passes ``[400e-9, ..., 700e-9]``
        from a comprehension, not a pre-allocated ndarray."""
        coeffs = (2.25, [])
        n = _polynomial_index([400e-9, 500e-9, 600e-9], coeffs)
        assert isinstance(n, np.ndarray)
        np.testing.assert_allclose(n, 1.5, atol=1e-12)

    @pytest.mark.skipif(not _JAX_AVAILABLE,
                        reason="JAX not installed")
    def test_jax_array_input_accepted(self):
        """JAX arrays expose ``__array__`` so the ``np.asarray`` lift
        path accepts them transparently.  We don't promise JAX-typed
        output (the polynomial sum runs on numpy) -- the contract is
        only that the call doesn't crash and produces correct values
        for downstream numpy / scipy consumers."""
        import jax.numpy as jnp
        coeffs = (2.25, [])
        lam = jnp.array([400e-9, 500e-9, 600e-9])
        n = _polynomial_index(lam, coeffs)
        # Result is numpy ndarray; values still correct.
        np.testing.assert_allclose(np.asarray(n), 1.5, atol=1e-6)


# ============================================================================
# _POLYNOMIAL_STUB_NAMES -- manifest invariants
# ============================================================================

class TestPolynomialStubManifest:
    """The manifest of stubbed formula-3 catalogue entries.

    Critically, NONE of the assertions hardcode glass names: they all
    auto-enumerate from GLASS_REGISTRY and the stub set.  This keeps
    the test future-proof against v5.2.1+ ingestions (a real ingestion
    just removes a name from _POLYNOMIAL_STUB_NAMES and adds it to
    POLYNOMIAL_COEFFICIENTS; the assertions below shift their target
    automatically).
    """

    def test_manifest_state_at_v5_2_3_ship(self):
        """v5.2.0 shipped with 24 known formula-3 catalogue entries
        STUBBED.  v5.2.3 completed the ingestion: ``POLYNOMIAL_COEFFICIENTS``
        now has 24 entries and ``_POLYNOMIAL_STUB_NAMES`` is empty.
        The frozenset is kept (rather than deleted) as a forward
        parking spot for future formula-3 catalogue additions.

        Pin: either the manifest is empty (full ingestion completed)
        OR every entry in it has NO matching POLYNOMIAL_COEFFICIENTS
        entry (regression guard against accidental dual-state).
        """
        if len(_POLYNOMIAL_STUB_NAMES) == 0:
            # v5.2.3 ship state: all 24 ingested.
            assert len(POLYNOMIAL_COEFFICIENTS) >= 24, (
                f"v5.2.3 expected at least 24 ingested formula-3 "
                f"glasses; found {len(POLYNOMIAL_COEFFICIENTS)}.")
        else:
            # Future state: stubs may re-appear if new formula-3
            # catalogue entries are added before their coefficients
            # are ingested.  In that case, the stub-to-coefficient
            # disjointness must still hold.
            overlap = (set(_POLYNOMIAL_STUB_NAMES)
                       & set(POLYNOMIAL_COEFFICIENTS))
            assert not overlap, (
                f"_POLYNOMIAL_STUB_NAMES and POLYNOMIAL_COEFFICIENTS "
                f"overlap on {overlap!r} -- a glass cannot be both "
                f"stubbed and ingested.")

    def test_every_stub_resolves_in_glass_registry(self):
        """Every name in the stub manifest must point at a real
        catalogue tuple in GLASS_REGISTRY -- else the
        NotImplementedError dispatch arm is unreachable (the unknown-
        glass ValueError fires first)."""
        for name in _POLYNOMIAL_STUB_NAMES:
            assert name in GLASS_REGISTRY, (
                f"_POLYNOMIAL_STUB_NAMES[{name!r}] has no GLASS_REGISTRY "
                f"entry -- the migration NotImplementedError is "
                f"unreachable.")
            entry = GLASS_REGISTRY[name]
            assert isinstance(entry, tuple) and len(entry) == 3, (
                f"GLASS_REGISTRY[{name!r}] = {entry!r} expected to be a "
                f"(shelf, book, page) tuple (catalogue dispatch path); "
                f"a __sellmeier__ / __polynomial__ sentinel would short-"
                f"circuit before reaching the stub fallback and "
                f"defeat the manifest.")

    def test_manifest_disjoint_from_polynomial_coefficients(self):
        """An ingested glass must be removed from the stub manifest at
        the same commit -- the v5.2 consistency check enforces this,
        but pin it in tests as well so regressions caught by either
        path."""
        overlap = set(_POLYNOMIAL_STUB_NAMES) & set(POLYNOMIAL_COEFFICIENTS)
        assert overlap == set(), (
            f"_POLYNOMIAL_STUB_NAMES overlaps with POLYNOMIAL_COEFFICIENTS "
            f"at {sorted(overlap)} -- these are mutually exclusive states.")

    def test_manifest_disjoint_from_sellmeier_coefficients(self):
        """A formula-3 glass should not also have Sellmeier-2 bundled
        coefficients -- the catalogue specifies one dispersion form
        per glass, and a duplicated entry would route inconsistently
        depending on whether refractiveindex is installed."""
        overlap = set(_POLYNOMIAL_STUB_NAMES) & set(SELLMEIER_COEFFICIENTS)
        assert overlap == set(), (
            f"_POLYNOMIAL_STUB_NAMES overlaps with SELLMEIER_COEFFICIENTS "
            f"at {sorted(overlap)} -- a glass has one dispersion form "
            f"per the refractiveindex.info YAML.")

    def test_manifest_covers_every_formula3_catalogue_glass(self):
        """Cross-check: every catalogue tuple-style entry for the 3
        formula-3 catalogues (CDGM polynomial subset, Hikari, Sumita)
        must either be in POLYNOMIAL_COEFFICIENTS (ingested) or in
        _POLYNOMIAL_STUB_NAMES (manifest-tracked).  An entry in
        neither would silently fall through to the ImportError arm on
        a minimal install -- the v5.2 contract is that ALL formula-3
        glasses get the NotImplementedError migration message.

        Auto-discovered by catalogue book name; the only formula-3
        catalogue subset that isn't enumerated by book is the 4 CDGM
        polynomial entries (since the CDGM book also has 8 Sellmeier-2
        entries).  We discover the CDGM polynomial entries by
        elimination: CDGM tuple-style entries NOT also bundled in
        SELLMEIER_COEFFICIENTS.
        """
        formula3_books = {'HIKARI-optical', 'SUMITA-optical'}
        # Auto-discovered formula-3 names per book.
        discovered = set()
        for name, entry in GLASS_REGISTRY.items():
            if not (isinstance(entry, tuple) and len(entry) == 3):
                continue
            shelf, book, _page = entry
            if book in formula3_books:
                discovered.add(name)
            elif book == 'CDGM-optical' and name not in SELLMEIER_COEFFICIENTS:
                # CDGM polynomial subset = CDGM entries lacking
                # Sellmeier coefficient rows.
                discovered.add(name)
        covered = set(POLYNOMIAL_COEFFICIENTS) | set(_POLYNOMIAL_STUB_NAMES)
        missing = discovered - covered
        assert missing == set(), (
            f"formula-3 catalogue entries missing from both "
            f"POLYNOMIAL_COEFFICIENTS and _POLYNOMIAL_STUB_NAMES: "
            f"{sorted(missing)}.  Add to _POLYNOMIAL_STUB_NAMES (or "
            f"to POLYNOMIAL_COEFFICIENTS with ingested values) so the "
            f"NotImplementedError migration message dispatches.")

    def test_every_stub_has_validity_range(self):
        """Each stubbed glass should have a GLASS_VALIDITY entry --
        even without bundled coefficients the validity range is known
        from the catalogue YAML and is useful for the off-range
        warning when the user eventually does install [glass]."""
        for name in _POLYNOMIAL_STUB_NAMES:
            assert name in GLASS_VALIDITY, (
                f"_POLYNOMIAL_STUB_NAMES[{name!r}] has no GLASS_VALIDITY "
                f"entry -- when the user installs [glass] and looks up "
                f"this glass off-range, no warning will fire.")


# ============================================================================
# get_glass_index dispatch -- behaviour for stubbed and ingested glasses
# ============================================================================

class TestStubbedGlassDispatch:
    """The user-facing contract: looking up a stubbed glass on a
    minimal install raises NotImplementedError with the documented
    migration text.  When refractiveindex IS installed, the call
    transparently delegates to the live database (no NotImplementedError).
    """

    def test_minimal_install_stubbed_glass_raises_not_implemented(
            self, monkeypatch):
        """v5.2.0 ship-state pin: when ``_POLYNOMIAL_STUB_NAMES`` is
        non-empty (i.e. some catalogue entries haven't been ingested
        yet), looking up such a name on a minimal install raises
        ``NotImplementedError`` with the documented migration text.

        v5.2.3 ship-state: ``_POLYNOMIAL_STUB_NAMES`` is empty (all 24
        ingested), so this test SKIPS rather than asserting a contract
        that no longer has a triggering input.  The dispatch arm is
        still present in ``get_glass_index`` and would re-activate if
        a future catalogue addition re-populates the manifest.
        """
        if not _POLYNOMIAL_STUB_NAMES:
            pytest.skip(
                "_POLYNOMIAL_STUB_NAMES is empty (v5.2.3 finished the "
                "ingestion); the NotImplementedError dispatch arm has "
                "no triggering input until a future catalogue addition.")
        monkeypatch.setattr(_glass, '_REFRACTIVEINDEX_AVAILABLE', False)
        # Pick a representative -- the contract applies to every
        # currently-stubbed name.
        for name in sorted(_POLYNOMIAL_STUB_NAMES)[:3]:
            with pytest.raises(NotImplementedError, match="formula-3"):
                get_glass_index(name, 587.6e-9)

    def test_minimal_install_error_text_directs_to_extras_or_issue(
            self, monkeypatch):
        """The NotImplementedError text must mention BOTH remediation
        paths: install [glass] extras OR open a v5.2.1 issue.

        v5.2.3 ship-state: ``_POLYNOMIAL_STUB_NAMES`` is empty (all 24
        ingested), so this test SKIPS until a future catalogue
        addition re-introduces a stubbed name.
        """
        if not _POLYNOMIAL_STUB_NAMES:
            pytest.skip(
                "_POLYNOMIAL_STUB_NAMES is empty (v5.2.3 finished the "
                "ingestion); no stubbed glass to trigger the error.")
        monkeypatch.setattr(_glass, '_REFRACTIVEINDEX_AVAILABLE', False)
        sample = next(iter(_POLYNOMIAL_STUB_NAMES))
        with pytest.raises(NotImplementedError) as excinfo:
            get_glass_index(sample, 587.6e-9)
        msg = str(excinfo.value)
        assert 'lumenairy[glass]' in msg or 'refractiveindex' in msg
        assert 'v5.2.1' in msg

    def test_minimal_install_unknown_glass_still_raises_value_error(
            self, monkeypatch):
        """The stub fallback must NOT swallow the unknown-glass error
        for typos -- a non-registered glass name still raises
        ValueError with the suggestion list."""
        monkeypatch.setattr(_glass, '_REFRACTIVEINDEX_AVAILABLE', False)
        with pytest.raises(ValueError, match="not in registry"):
            get_glass_index('NOT_A_REAL_GLASS_XYZ', 587.6e-9)

    def test_minimal_install_sellmeier_glass_still_works(
            self, monkeypatch):
        """The stub fallback path must not break the bundled Sellmeier
        fallback for catalogue entries that DO have bundled
        coefficients (e.g. N-BK7)."""
        monkeypatch.setattr(_glass, '_REFRACTIVEINDEX_AVAILABLE', False)
        n = get_glass_index('N-BK7', 587.6e-9)
        assert n == pytest.approx(1.5168, abs=5e-5)

    def test_refractiveindex_available_stubbed_glass_resolves(self):
        """When the optional refractiveindex package IS installed (the
        v5.2 [glass] extras path), looking up a stubbed glass must
        succeed via the live database -- no NotImplementedError.

        Skipped if refractiveindex is not installed in this test env."""
        if not _glass._REFRACTIVEINDEX_AVAILABLE:
            pytest.skip("refractiveindex package not installed; "
                        "stub dispatch on minimal install covered "
                        "by the test_minimal_install_* tests above")
        # Validity range for H-ZK7 is [0.365 um, 1.014 um] per
        # GLASS_VALIDITY; 587.6 nm (d-line) is in-range.
        n = get_glass_index('H-ZK7', 587.6e-9)
        assert np.isfinite(n)
        # n_d for H-ZK7 per CDGM 2022-06 catalog is ~1.61272; allow
        # generous tolerance since this is the live-database path,
        # not the bundled cross-check.
        assert 1.5 < n < 1.8


# ============================================================================
# Ingestion cross-check (becomes load-bearing in v5.2.1+ as coefficients
# land).  At v5.2 ship POLYNOMIAL_COEFFICIENTS is empty so the loop
# vacuously passes; the test grows teeth automatically as entries land.
# ============================================================================

class TestPolynomialCoefficientsCrossCheck:
    """Auto-enumerated 5e-5 n_d cross-check for any ingested
    coefficients.  Per the v5.2 ROADMAP note ("may need relaxation for
    BK7-like polynomial fits per the V3 v4.16.2 note"), the tolerance
    starts at 5e-5 and any glass requiring a relaxation must annotate
    its row in POLYNOMIAL_COEFFICIENTS with the rationale.
    """

    def test_every_ingested_glass_matches_catalog_n_d(self):
        """For every glass with bundled formula-3 coefficients,
        cross-check the n_d (587.6 nm) value against the live
        refractiveindex.info lookup (if available) at 5e-5 tolerance.

        At v5.2 ship POLYNOMIAL_COEFFICIENTS is empty so this test
        vacuously passes; as v5.2.1+ ingestions land, the loop body
        runs and pins each ingestion to the catalogue source."""
        if not POLYNOMIAL_COEFFICIENTS:
            pytest.skip("no formula-3 coefficients ingested yet "
                        "(v5.2 ship state); cross-check vacuous")
        if not _glass._REFRACTIVEINDEX_AVAILABLE:
            pytest.skip("refractiveindex not installed; cannot "
                        "cross-check ingested formula-3 coefficients "
                        "against live catalogue values")
        lam_d = 587.6e-9
        for name, coeffs in POLYNOMIAL_COEFFICIENTS.items():
            # Skip if d-line is out of validity range (the cross-check
            # only makes physical sense in-range).
            lmin, lmax = GLASS_VALIDITY.get(name, (0.0, float('inf')))
            if not (lmin <= lam_d <= lmax):
                continue
            n_bundled = _polynomial_index(lam_d, coeffs, glass_name=name)
            # Force the live-database path.
            entry = GLASS_REGISTRY[name]
            if not (isinstance(entry, tuple) and len(entry) == 3):
                continue
            from refractiveindex import RefractiveIndexMaterial as _RIM
            shelf, book, page = entry
            mat = _RIM(shelf=shelf, book=book, page=page)
            n_live = float(mat.get_refractive_index(
                lam_d * 1e9, unit='nm'))
            assert n_bundled == pytest.approx(n_live, abs=5e-5), (
                f"formula-3 ingestion cross-check failed for "
                f"{name!r}: bundled n_d = {n_bundled:.6f} vs "
                f"live n_d = {n_live:.6f} (delta "
                f"{abs(n_bundled - n_live):.2e}, tol 5e-5)")
