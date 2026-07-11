"""v4.16.3 Agent D regression tests.

Scope: P3 cluster closures from
``docs/audits/AUDIT_V4_16_2_2026_05_20.md``.

* P3-NEW-F1-1 -- ``'__polynomial__'`` sentinel parity with
  ``'__sellmeier__'`` in :mod:`lumenairy.glass`.
* P3-NEW-V3-1 -- inline-comment dispatch-order reconcile in
  :mod:`lumenairy.glass`.
* P3-NEW-F2-LOW-1 -- ``DEFAULT_REAL_DTYPE`` / ``DEFAULT_WAVE_PROPAGATOR``
  / ``DEFAULT_DY`` re-exported at top level (sibling parity with
  ``DEFAULT_COMPLEX_DTYPE``).
* P3-NEW-F1-4 -- per-surface (not max) thickness reported in the
  high-NA hoist warning in :mod:`lumenairy.raytrace.jax_trace`.
* P3-NEW-F1-2 + P3-NEW-F1-3 -- multiprocess/fork-safety notes
  documented in :mod:`lumenairy.propagators.propagation`.
* P3-NEW-F2-LOW-2 -- "structurally retired" claim softened in
  CHANGELOG + ROADMAP.
* P3-NEW-F1-5 -- ``psutil`` promoted to Required in requirements.txt.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# audit closure: P3-NEW-F1-1 -- __polynomial__ sentinel parity
# ============================================================================

class TestPolynomialSentinelParity:
    """Sibling parity with the ``'__sellmeier__'`` sentinel.  Both
    flavours must:

    * be reachable from ``get_glass_index`` regardless of whether the
      optional ``refractiveindex`` package is installed,
    * raise a sentinel-specific :class:`ValueError` when the registry
      flag is set but the coefficient row is missing,
    * be cross-checked at module-load time by
      :func:`lumenairy.glass._check_glass_registry_consistency`
      (forward and reverse directions).
    """

    def test_polynomial_sentinel_dispatch(self):
        """audit closure: P3-NEW-F1-1

        Registering a glass with the ``'__polynomial__'`` sentinel and
        a coefficient row in :data:`POLYNOMIAL_COEFFICIENTS` must
        dispatch via :func:`_polynomial_index`, independent of
        ``_REFRACTIVEINDEX_AVAILABLE``.
        """
        from lumenairy.glass import (
            GLASS_REGISTRY,
            POLYNOMIAL_COEFFICIENTS,
            get_glass_index,
        )
        name = '__TEST_POLY_DISPATCH__'
        # Polynomial form: n^2 = 2.25 (constant) -> n = 1.5 at every
        # wavelength.
        POLYNOMIAL_COEFFICIENTS[name] = (2.25, [])
        GLASS_REGISTRY[name] = '__polynomial__'
        try:
            n = get_glass_index(name, 587.6e-9)
            assert abs(n - 1.5) < 1e-12, (
                f"__polynomial__ sentinel must dispatch to "
                f"_polynomial_index; got n={n!r}.")
        finally:
            GLASS_REGISTRY.pop(name, None)
            POLYNOMIAL_COEFFICIENTS.pop(name, None)

    def test_polynomial_sentinel_missing_row_raises(self):
        """audit closure: P3-NEW-F1-1

        Sibling to the ``'__sellmeier__'`` ValueError: a registry entry
        flagged ``'__polynomial__'`` without a
        :data:`POLYNOMIAL_COEFFICIENTS` row must raise
        :class:`ValueError`.  The module-load consistency check would
        normally catch this at import time, so the test bypasses the
        check by mutating the registry after import.
        """
        from lumenairy.glass import GLASS_REGISTRY, get_glass_index
        name = '__TEST_POLY_MISSING__'
        GLASS_REGISTRY[name] = '__polynomial__'
        try:
            with pytest.raises(ValueError, match="__polynomial__"):
                get_glass_index(name, 587.6e-9)
        finally:
            GLASS_REGISTRY.pop(name, None)

    def test_consistency_check_polynomial_forward(self):
        """audit closure: P3-NEW-F1-1

        Sibling to the ``'__sellmeier__'`` forward check: a registry
        entry flagged ``'__polynomial__'`` without a coefficient row
        must raise :class:`RuntimeError` from
        ``_check_glass_registry_consistency``.
        """
        from lumenairy.glass import (
            GLASS_REGISTRY,
            POLYNOMIAL_COEFFICIENTS,
            _check_glass_registry_consistency,
        )
        name = '__TEST_POLY_FORWARD__'
        GLASS_REGISTRY[name] = '__polynomial__'
        try:
            with pytest.raises(RuntimeError, match="__polynomial__"):
                _check_glass_registry_consistency()
        finally:
            GLASS_REGISTRY.pop(name, None)
            POLYNOMIAL_COEFFICIENTS.pop(name, None)

    def test_consistency_check_polynomial_reverse(self):
        """audit closure: P3-NEW-F1-1

        Sibling to the SELLMEIER_COEFFICIENTS reverse check: a
        :data:`POLYNOMIAL_COEFFICIENTS` row without a registry entry
        must raise :class:`RuntimeError`.
        """
        from lumenairy.glass import (
            GLASS_REGISTRY,
            POLYNOMIAL_COEFFICIENTS,
            _check_glass_registry_consistency,
        )
        name = '__TEST_POLY_REVERSE__'
        POLYNOMIAL_COEFFICIENTS[name] = (2.25, [])
        try:
            with pytest.raises(RuntimeError, match="POLYNOMIAL_COEFFICIENTS"):
                _check_glass_registry_consistency()
        finally:
            GLASS_REGISTRY.pop(name, None)
            POLYNOMIAL_COEFFICIENTS.pop(name, None)

    def test_polynomial_sentinel_complex_path(self):
        """audit closure: P3-NEW-F1-1

        :func:`get_glass_index_complex` must accept the
        ``'__polynomial__'`` sentinel and return ``n + 0j`` (no
        extinction data; same convention as the ``'__sellmeier__'``
        branch).
        """
        from lumenairy.glass import (
            GLASS_REGISTRY,
            POLYNOMIAL_COEFFICIENTS,
            get_glass_index_complex,
        )
        name = '__TEST_POLY_COMPLEX__'
        POLYNOMIAL_COEFFICIENTS[name] = (2.25, [])
        GLASS_REGISTRY[name] = '__polynomial__'
        try:
            n = get_glass_index_complex(name, 587.6e-9)
            assert isinstance(n, complex)
            assert abs(n.real - 1.5) < 1e-12
            assert n.imag == 0.0
        finally:
            GLASS_REGISTRY.pop(name, None)
            POLYNOMIAL_COEFFICIENTS.pop(name, None)


# ============================================================================
# audit closure: P3-NEW-F2-LOW-1 -- DEFAULT_* re-exports
# ============================================================================

class TestDefaultConfigRebexports:
    """The v4.16.2 default-config knob globals must be reachable as
    ``lumenairy.DEFAULT_*`` for sibling parity with
    ``lumenairy.DEFAULT_COMPLEX_DTYPE``."""

    def test_default_real_dtype_top_level(self):
        """audit closure: P3-NEW-F2-LOW-1"""
        import lumenairy as la
        assert hasattr(la, 'DEFAULT_REAL_DTYPE')
        assert la.DEFAULT_REAL_DTYPE == np.float64

    def test_default_wave_propagator_top_level(self):
        """audit closure: P3-NEW-F2-LOW-1"""
        import lumenairy as la
        assert hasattr(la, 'DEFAULT_WAVE_PROPAGATOR')
        assert la.DEFAULT_WAVE_PROPAGATOR == 'asm'

    def test_default_dy_top_level(self):
        """audit closure: P3-NEW-F2-LOW-1"""
        import lumenairy as la
        assert hasattr(la, 'DEFAULT_DY')
        # Library default is ``None`` ("match dx").
        assert la.DEFAULT_DY is None

    def test_default_constants_listed_in_all(self):
        """audit closure: P3-NEW-F2-LOW-1

        Sibling parity: ``DEFAULT_COMPLEX_DTYPE`` is in ``__all__``;
        the three v4.16.2 globals must be too.
        """
        import lumenairy as la
        for name in ('DEFAULT_REAL_DTYPE',
                     'DEFAULT_WAVE_PROPAGATOR',
                     'DEFAULT_DY'):
            assert name in la.__all__, (
                f"{name!r} missing from lumenairy.__all__; sibling "
                f"to DEFAULT_COMPLEX_DTYPE which is listed.")


# ============================================================================
# audit closure: P3-NEW-F1-2 + P3-NEW-F1-3 -- multiprocess/fork notes
# ============================================================================

def test_propagation_documents_multiprocess_fork_semantics():
    """audit closure: P3-NEW-F1-2

    The ``DEFAULT_*`` module-level globals are not pickle/fork-safe
    under ``multiprocessing.get_context('spawn')``; the file must
    document this honestly so users aren't surprised when their
    ``set_default_*`` calls silently fail to propagate to workers.

    v5.1.0 split (Agent C): the ``DEFAULT_*`` globals and their
    multiprocess/fork documentation moved from
    ``propagators/propagation.py`` (now a thin re-export shell) to
    ``propagators/fft_infra.py``.  The test reads both files so the
    contract is anchored to wherever the documentation actually lives.
    """
    candidates = [
        _REPO_ROOT / 'lumenairy' / 'propagators' / 'fft_infra.py',
        _REPO_ROOT / 'lumenairy' / 'propagators' / 'propagation.py',
    ]
    src = ''
    for p in candidates:
        if p.exists():
            src += p.read_text(encoding='utf-8')
    # Section header (cited verbatim in the audit recommendation).
    assert 'Multiprocess / fork notes' in src, (
        "Neither propagation.py nor fft_infra.py contains the "
        "`Multiprocess / fork notes` section near the DEFAULT_* "
        "globals (audit P3-NEW-F1-2 + P3-NEW-F1-3).")
    # Must mention spawn explicitly (Windows + macOS default; the
    # context where the limitation is most surprising).
    assert 'spawn' in src.lower(), (
        "Multiprocess section must mention `spawn` explicitly -- "
        "the context where the DEFAULT_* re-import drift is silent.")
    # Must mention the one-shot latch fork-safety corollary.
    assert ('latch' in src.lower()) and (
        'fork-safe' in src.lower() or 'fork safety' in src.lower()
        or 'fork-safety' in src.lower()), (
        "Multiprocess section must mention the one-shot latch "
        "fork-safety corollary (audit P3-NEW-F1-2).")


# ============================================================================
# audit closure: P3-NEW-F2-LOW-2 -- softened "structurally retired" claim
# ============================================================================

def test_changelog_structurally_retired_claim_softened():
    """audit closure: P3-NEW-F2-LOW-2

    The honest framing per the audit: ``structurally retired at
    known classes; new classes will continue to surface``.  CHANGELOG
    must include the qualifying clause.
    """
    text = (_REPO_ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    # The original assertion is preserved with a qualifying clause.
    # Two occurrences in the file (v4.16.0 + v4.16.2 mentions).
    matches = re.findall(
        r'structurally retired[^.\n]*'
        r'(?:currently-known|new classes will continue to surface)',
        text, re.IGNORECASE,
    )
    assert matches, (
        "CHANGELOG.md must soften 'structurally retired across all "
        "known classes' with a clause like 'currently-known classes; "
        "new classes will continue to surface' (audit P3-NEW-F2-LOW-2).")
    # And the bare overreach phrase ("retired across all known classes")
    # without the qualifying clause must NOT survive in the CHANGELOG.
    bare = re.findall(
        r'structurally retired across all known classes(?![\s\S]{0,150}'
        r'(?:currently-known|new classes))',
        text,
    )
    assert not bare, (
        f"CHANGELOG.md still contains bare 'structurally retired "
        f"across all known classes' without softening clause: {bare}")


def test_roadmap_structurally_retired_claim_softened():
    """audit closure: P3-NEW-F2-LOW-2

    Lock-step with CHANGELOG.  ROADMAP must carry the same
    qualifying clause.
    """
    text = (_REPO_ROOT / 'ROADMAP.md').read_text(encoding='utf-8')
    matches = re.findall(
        r'structurally\s+retired[^.\n]*'
        r'(?:currently-known|new classes will continue to surface)',
        text, re.IGNORECASE | re.DOTALL,
    )
    assert matches, (
        "ROADMAP.md must soften 'structurally retired across all "
        "known classes' with the qualifying clause (audit "
        "P3-NEW-F2-LOW-2; lock-step with CHANGELOG).")


# ============================================================================
# audit closure: P3-NEW-F1-5 -- psutil promoted to Required
# ============================================================================

def test_requirements_psutil_in_required_block():
    """audit closure: P3-NEW-F1-5

    psutil is a hard dep in pyproject.toml ``[project.dependencies]``;
    requirements.txt must list it in the Required block (not the
    Recommended block) for parity.
    """
    text = (_REPO_ROOT / 'requirements.txt').read_text(encoding='utf-8')
    # Split on the explicit `=== Required` / `=== Recommended` /
    # `=== Optional` headings.  Find the Required block.
    parts = re.split(r'#\s*===.*?===\s*\n', text)
    headings = re.findall(r'#\s*===\s*(.*?)\s*===', text)
    required_idx = None
    for i, h in enumerate(headings):
        if h.lower().startswith('required'):
            required_idx = i
            break
    assert required_idx is not None, (
        "requirements.txt missing a `=== Required ===` block.")
    # parts[required_idx + 1] is the body of the Required block.
    required_body = parts[required_idx + 1]
    assert 'psutil' in required_body, (
        "requirements.txt must list `psutil` in the Required block "
        "for parity with pyproject.toml [project.dependencies] "
        "(audit P3-NEW-F1-5).")


# ============================================================================
# audit closure: P3-NEW-V3-1 -- dispatch-order doc/code reconcile
# ============================================================================

def test_glass_dispatch_order_comment_matches_code():
    """audit closure: P3-NEW-V3-1

    The inline comment near the Sellmeier branch in glass.py must
    cite the actual code order (SELLMEIER then POLYNOMIAL), not the
    inverted phrasing that drifted into prior release notes.
    """
    src = (_REPO_ROOT / 'lumenairy' / 'glass.py').read_text(
        encoding='utf-8')
    assert 'SELLMEIER -> POLYNOMIAL' in src, (
        "glass.py must carry an inline comment matching the actual "
        "code order (SELLMEIER -> POLYNOMIAL); audit P3-NEW-V3-1.")
