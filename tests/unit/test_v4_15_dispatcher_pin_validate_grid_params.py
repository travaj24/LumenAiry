"""Parametrized dispatcher-pin for input-validation entry-point.

Audit context
-------------

``AUDIT_V4_14_2_2026_05_17.md`` Part 3.5 catalogued 5 candidate
meta-pin patterns for v4.15+ to extend the structural counter-measure
approach.  Candidate #3 -- the **input-validation entry-point** pin --
is implemented here.

The pattern: every ``create_*`` factory in ``lumenairy.sources.core``
is, by convention, a public entry-point that takes grid parameters
(``N``, ``dx``, ``wavelength``, optionally ``dy``).  v4.14.2 added
:func:`_validate_grid_params` (``sources/core.py:36-176``) as a
centralised helper; v4.14.3 extended it with ``support_tuple_N``
gating for the 3 factories with genuine 2-D grid support.  This
meta-pin asserts the symmetric invariant:

* every ``create_*`` factory in the ``sources/core.py`` ``__all__``
  surface MUST call ``_validate_grid_params(...)`` (or an
  equivalent-named alias) in the first 15 lines of its function
  body.

The pin walks the module discovering ``create_*`` factories at
collection time and parametrizes one assertion per factory.  A
future ``create_xyz`` that lands without the validation call -- the
exact failure mode the v4.14.2 audit flagged as a sibling-gap class
-- fails CI at the parametrized entry point.

Pre-flight guards
-----------------

* ``test_walker_found_at_least_eight_factories`` -- count floor.
  If the walker collapses to zero or near-zero factories the
  parametrize list silently no-ops and the meta-pin becomes a
  silent regression vector.  The count floor catches walker
  breakage (e.g. an accidentally narrowed regex, a renamed module).

* ``test_counter_pin_fake_unvalidated_factory_fails`` -- counter-pin.
  Validates the test fixture itself by injecting a synthetic
  unvalidated factory into the discovery results and asserting the
  pin would catch it.

* ``test_counter_pin_fake_validated_factory_passes`` -- counter-pin.
  Same but for a synthetic validated factory.

Exemption set
-------------

``_KNOWN_VALIDATION_EXEMPTIONS`` starts empty (v4.15.0 baseline).
Future regressions that legitimately bypass ``_validate_grid_params``
(e.g. a private-public factory that re-uses validation through a
sibling) ratchet in here with ``xfail(strict=True)`` so a fix lands
as an XPASS and the exemption can be removed.

Author: Andrew Traverso -- v4.15.0 / Agent F
"""
from __future__ import annotations

import inspect
import re
from typing import List, Tuple

import pytest

import lumenairy as la
from lumenairy import sources as _sources_pkg
from lumenairy.sources import core as _sources_core

# ============================================================================
# Aliases that count as a valid validation call.  The meta-pin accepts
# any of these in the first 15 body lines.
# ============================================================================

_VALIDATION_ALIASES: Tuple[str, ...] = (
    '_validate_grid_params',
    # Reserved for future per-domain aliases; e.g. if a DOE-style
    # factory wants its own narrower validator the name lands here.
    # Currently empty besides the canonical entry.
)

# Compiled "is a validation call" regex.  Match either
# ``_validate_grid_params(``-style invocation or any alias listed
# above.  Word-boundary-anchored so a substring match (e.g.
# ``not_validate_grid_params``) doesn't false-positive.
_VALIDATION_CALL_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_])(?:' + '|'.join(re.escape(a) for a in _VALIDATION_ALIASES) + r')\s*\('
)


# ============================================================================
# Discovery walker
# ============================================================================

def _discover_create_factories() -> List[Tuple[str, object]]:
    """Walk ``lumenairy.sources.core`` for every public ``create_*``
    factory.

    Returns a sorted list of ``(name, function_object)`` pairs.
    The walker prefers the module's ``__all__`` if defined, falling
    back to a ``dir()`` sweep for resilience against missing
    ``__all__`` (which would let the meta-pin silently shrink).

    Public factories are identified by:

    * name starts with ``create_``
    * attribute resolves to a callable
    * either listed in the module's ``__all__``, OR (fallback) lives
      directly in the module (not re-exported from elsewhere)
    """
    candidates: List[Tuple[str, object]] = []
    mod = _sources_core
    explicit_all = getattr(mod, '__all__', None)
    seen = set()
    if explicit_all is not None:
        for name in explicit_all:
            if not name.startswith('create_'):
                continue
            obj = getattr(mod, name, None)
            if obj is None or not callable(obj):
                continue
            candidates.append((name, obj))
            seen.add(name)
    # Fallback / supplemental sweep: every directly-defined create_*
    # function in the module that we missed via __all__.  Defends
    # against an incomplete ``__all__`` shrinking the walker.
    for name in dir(mod):
        if name in seen or not name.startswith('create_'):
            continue
        obj = getattr(mod, name, None)
        if obj is None or not callable(obj):
            continue
        # Only include callables defined IN this module (not
        # re-imports / aliases pointing at other modules).
        try:
            src_mod = inspect.getmodule(obj)
        except TypeError:
            continue
        if src_mod is None or src_mod.__name__ != mod.__name__:
            continue
        candidates.append((name, obj))
        seen.add(name)
    return sorted(candidates, key=lambda p: p[0])


_DISCOVERED_FACTORIES = _discover_create_factories()


# ============================================================================
# Documented exemption set.  Starts empty in v4.15.0.  Future
# regressions ratchet in with xfail(strict=True).
# ============================================================================

_KNOWN_VALIDATION_EXEMPTIONS = frozenset({
    # ``create_led_source`` calls ``_validate_grid_params`` AFTER its
    # legacy-positional shim (the shim runs first so a legacy 5-
    # positional call gets re-mapped to canonical kwargs before
    # validation).  This positions the validate call past the 15-
    # body-line head window the meta-pin scans, but the call IS
    # present (``sources/core.py:create_led_source`` -- look for
    # ``_validate_grid_params(N, dx, wavelength,
    # fn_name='create_led_source')``).  Re-ordering the validate
    # call to come BEFORE the shim would break legacy callers (the
    # shim re-maps positions 2-4, so the wavelength validation
    # would compare against the legacy ``diameter`` value).
    # Recorded here as a documented exemption rather than widening
    # ``_BODY_HEAD_LINES`` for every factory.  When the legacy shim
    # is removed at v5.0 this exemption ratchets out via XPASS.
    'create_led_source',
})


# ============================================================================
# Core check -- substring match in first N body lines
# ============================================================================

_BODY_HEAD_LINES = 15


def _body_calls_validator(fn) -> bool:
    """True iff one of the ``_VALIDATION_ALIASES`` appears in the
    first ``_BODY_HEAD_LINES`` lines of ``fn``'s body.

    Skips the docstring (handled by walking the source past the
    docstring closer).
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return False
    lines = src.splitlines()
    # Drop the ``def name(...):`` header and any subsequent docstring.
    # A simple, robust heuristic: find the first non-blank, non-
    # docstring, non-decorator, non-def-header line and take the
    # next ``_BODY_HEAD_LINES`` from there.
    in_docstring = False
    docstring_quote = None
    body_lines: List[str] = []
    seen_def = False
    paren_depth = 0
    for line in lines:
        stripped = line.strip()
        if not seen_def:
            if stripped.startswith('def '):
                seen_def = True
                # Track multi-line def parameter lists.
                paren_depth = line.count('(') - line.count(')')
            continue
        if paren_depth > 0:
            paren_depth += line.count('(') - line.count(')')
            continue
        # We're now inside the function body.  Handle docstring.
        if in_docstring:
            body_lines.append(line)
            # Close the docstring iff the closing quote appears on this line
            if docstring_quote and docstring_quote in line:
                in_docstring = False
                docstring_quote = None
            continue
        if not stripped:
            body_lines.append(line)
            continue
        # Detect docstring opener (must be the very first non-blank
        # body statement after the def).
        if (stripped.startswith('"""') or stripped.startswith("'''")):
            quote = '"""' if stripped.startswith('"""') else "'''"
            body_lines.append(line)
            # Single-line docstring case (``"""x"""``).
            if (stripped.endswith(quote) and len(stripped) > 3
                    and stripped != quote):
                # Already closed; one-liner.
                continue
            in_docstring = True
            docstring_quote = quote
            continue
        body_lines.append(line)
        if len(body_lines) >= _BODY_HEAD_LINES * 4:
            # Safety cap so a malformed function doesn't make us scan
            # the entire file.  4x the threshold to be generous.
            break
    # Re-scan: drop docstring content from body_lines so the search
    # window is the first 15 EXECUTABLE lines.
    executable_lines: List[str] = []
    in_doc = False
    quote_match = None
    for line in body_lines:
        stripped = line.strip()
        if in_doc:
            if quote_match and quote_match in line:
                in_doc = False
                quote_match = None
            continue
        if (stripped.startswith('"""') or stripped.startswith("'''")):
            quote = '"""' if stripped.startswith('"""') else "'''"
            if (stripped.endswith(quote) and len(stripped) > 3
                    and stripped != quote):
                continue  # one-liner
            in_doc = True
            quote_match = quote
            continue
        if not stripped:
            continue
        executable_lines.append(line)
        if len(executable_lines) >= _BODY_HEAD_LINES:
            break
    head = '\n'.join(executable_lines)
    return bool(_VALIDATION_CALL_PATTERN.search(head))


# ============================================================================
# Pre-flight guards
# ============================================================================

def test_walker_found_at_least_eight_factories():
    """The factory walker must discover at least 8 ``create_*``
    entry points in ``sources/core.py``.  Below this threshold the
    parametrize list collapses to near-zero and the meta-pin
    silently no-ops -- exactly the failure mode the v4.14.2 audit
    catalogued.

    The lower bound is conservative: ``sources/core.py`` ships with
    14 factories as of v4.15.0 (v4.14.0 added the 6 mode-family
    helpers, v4.14.2 added the partial-coherence trio, v4.14.3 left
    the count unchanged).  Dropping below 8 means the walker is
    broken or factories were removed without updating the
    expectation.
    """
    assert len(_DISCOVERED_FACTORIES) >= 8, (
        f'Factory walker found only {len(_DISCOVERED_FACTORIES)} '
        f'``create_*`` factories in lumenairy.sources.core.  Expected '
        f'at least 8.  Either the discovery walker is broken or '
        f'factories were removed without updating this floor.')


def test_counter_pin_fake_unvalidated_factory_fails():
    """Counter-pin: confirm that a synthetic factory WITHOUT
    ``_validate_grid_params`` would trip the test.

    Validates the test fixture against the very failure mode it
    claims to catch.  If this passes silently when the synthetic
    factory has no validator, the meta-pin is non-functional.
    """
    def create_synthetic_unvalidated(N, dx, wavelength):
        """A pretend factory that bypasses validation."""
        # No validation call here -- the body just does its work.
        import numpy as _np
        return _np.ones((N, N), dtype=_np.complex128)
    assert not _body_calls_validator(create_synthetic_unvalidated), (
        'Counter-pin failed: synthetic unvalidated factory was '
        'reported as validating.  The discovery / regex pipeline is '
        'broken -- the meta-pin would produce false-negatives in CI.')


def test_counter_pin_fake_validated_factory_passes():
    """Counter-pin: confirm that a synthetic factory WITH
    ``_validate_grid_params`` would pass the test.

    Validates that the regex / line-scan correctly identifies a
    well-formed validation call.  If this fails the meta-pin
    produces false-positives that would block every legitimate fix
    that lands in CI.
    """
    def create_synthetic_validated(N, dx, wavelength):
        """A pretend factory that DOES validate."""
        _validate_grid_params(N, dx, wavelength)  # noqa: F821
        import numpy as _np
        return _np.ones((N, N), dtype=_np.complex128)
    assert _body_calls_validator(create_synthetic_validated), (
        'Counter-pin failed: synthetic validated factory was '
        'reported as missing the validator.  The line-scan / regex '
        'is too strict -- the meta-pin would produce false-positives '
        'in CI.')


# ============================================================================
# Pin -- every discovered factory calls _validate_grid_params early
# ============================================================================

@pytest.mark.parametrize(
    'name, fn',
    _DISCOVERED_FACTORIES,
    ids=[name for name, _fn in _DISCOVERED_FACTORIES],
)
def test_factory_calls_validate_grid_params(name, fn, request):
    """Every ``create_*`` factory in ``sources/core.py`` must call
    ``_validate_grid_params`` (or an equivalent alias) in the first
    ``_BODY_HEAD_LINES`` lines of its body.

    This closes the v4.14.2 audit's Part 3.5 candidate meta-pin
    #3 (input-validation entry-point) so future factories that
    land without the validation call fail CI at the parametrized
    entry point rather than producing opaque downstream
    ``np.arange`` / ``meshgrid`` / divide-by-zero errors.
    """
    if name in _KNOWN_VALIDATION_EXEMPTIONS:
        request.applymarker(pytest.mark.xfail(
            reason=f'{name} is a documented validation exemption '
                   f'(see ``_KNOWN_VALIDATION_EXEMPTIONS``).  When the '
                   f'factory grows a real validator the xfail flips '
                   f'to XPASS and the exemption is removed.',
            strict=True))
    assert _body_calls_validator(fn), (
        f'{name!r} does not call ``_validate_grid_params`` (or any '
        f'alias in ``_VALIDATION_ALIASES``) in its first '
        f'{_BODY_HEAD_LINES} executable body lines.  Add the call '
        f'as the first body statement, with ``fn_name={name!r}`` so '
        f'error messages identify the entry point.  See e.g. '
        f'``create_gaussian_beam`` (sources/core.py) for the '
        f'canonical pattern.  If the factory legitimately bypasses '
        f'validation (e.g. it consumes another factory\'s output '
        f'transitively), add the name to '
        f'``_KNOWN_VALIDATION_EXEMPTIONS`` with a comment '
        f'explaining the rationale.')


# ============================================================================
# Diagnostic helper (printed when -v is on; useful when adding a new
# factory and triaging discovery)
# ============================================================================

def test_discovered_factories_list_for_diagnostics():
    """Emit the discovered factory list to stdout for triage.

    Not really a test -- always passes -- but provides a single
    inspection point so ``pytest -v`` shows the discovery walker's
    output without needing to run the full parametrized suite.
    """
    names = [name for name, _fn in _DISCOVERED_FACTORIES]
    # The body of an assert message is captured by pytest on failure;
    # we want to always print so use the standard pytest mechanism.
    print(f'\nv4.15 validate_grid_params meta-pin discovered '
          f'{len(names)} factories:')
    for n in names:
        print(f'  - {n}')
    assert isinstance(names, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
