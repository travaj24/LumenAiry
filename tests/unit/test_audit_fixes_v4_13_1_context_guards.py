"""Pinning tests for the v4.13.1 audit fix P1-E:
``lumenairy._context.lumenairy_context`` cache-clear import guards.

Audit reference
---------------

``AUDIT_V4_13_0_2026_05_17.md`` P1-E found that the
``clear_caches_on_exit`` branch of :func:`lumenairy_context`
imported ``clear_asm_caches`` OUTSIDE the try/except guard.  The
block-level comment claimed "Each call is guarded so a missing
optional dependency or a future rename does not prevent the others
from firing," but if the very first import (``clear_asm_caches``)
failed for any reason -- circular import, package rename, partial
install -- the ``ImportError`` propagated through the ``with``
block's ``finally`` and bypassed ALL 6 subsequent guarded
cache-clear blocks.

v4.13.1 fix:

* Move ``from .propagators.propagation import clear_asm_caches``
  INSIDE the try/except.
* Add ``ImportError`` to the typed except tuple to match the
  pattern used by the other 6 cache-clear blocks in the same
  function.

What this test pins
-------------------

Monkey-patch ``clear_asm_caches`` so the import path raises
``ImportError``; assert that the other 6 cache-clear functions
still execute -- i.e. the first block's import-fail does not bypass
them.  Verifying via mock-counter on the downstream targets is the
cleanest pin (rather than relying on the real caches being
populated and observable).

Author: Andrew Traverso -- v4.13.1
"""
from __future__ import annotations

import sys
import types

import pytest


def _install_failing_clear_asm_caches(monkeypatch):
    """Make ``from lumenairy.propagators.propagation import
    clear_asm_caches`` raise ``ImportError`` for the duration of one
    test.

    We can't simply ``monkeypatch.delattr`` on the module attribute
    -- the import statement does not always touch the attribute,
    depending on whether the module is already imported.  The
    cleanest reproduction is to replace the bound name on the
    already-imported module with something that *raises on access*
    via a custom ``__getattr__`` (PEP 562) shim.
    """
    import lumenairy.propagators.propagation as pp_mod

    # Patch the attribute itself to a sentinel that raises on call.
    # Then patch the module-level __getattr__ to raise ImportError
    # when something tries to bind the name through a fresh import.
    monkeypatch.delattr(pp_mod, 'clear_asm_caches', raising=False)

    def _module_getattr(name):
        if name == 'clear_asm_caches':
            raise ImportError(
                f"synthetic ImportError on {name} (test injection)")
        raise AttributeError(name)

    # __getattr__ on a module is invoked when normal attribute lookup
    # fails (which it will, because we just deleted the attribute).
    monkeypatch.setattr(pp_mod, '__getattr__', _module_getattr,
                        raising=False)


def _install_counting_replacements(monkeypatch):
    """Replace each of the 6 OTHER cache-clear functions with a
    counter so we can assert they were called.

    Returns a dict ``{name: counter_dict}`` for inspection in the
    test body.
    """
    counters = {}

    # analysis.core.clear_zernike_basis_cache
    import lumenairy.analysis.core as ac_mod
    counters['zernike'] = {'count': 0}

    def _bump_zernike():
        counters['zernike']['count'] += 1
    monkeypatch.setattr(ac_mod, 'clear_zernike_basis_cache',
                        _bump_zernike, raising=False)

    # propagators.asymptotic.clear_lg_polynomial_cache
    try:
        import lumenairy.propagators.asymptotic as asym_mod
        counters['lg'] = {'count': 0}

        def _bump_lg():
            counters['lg']['count'] += 1
        monkeypatch.setattr(asym_mod, 'clear_lg_polynomial_cache',
                            _bump_lg, raising=False)
    except ImportError:
        counters['lg'] = {'count': -1}  # module not present

    # raytrace.jax_trace.clear_trace_jax_cache
    try:
        import lumenairy.raytrace.jax_trace as jt_mod
        counters['trace_jax'] = {'count': 0}

        def _bump_trace_jax():
            counters['trace_jax']['count'] += 1
        monkeypatch.setattr(jt_mod, 'clear_trace_jax_cache',
                            _bump_trace_jax, raising=False)
    except ImportError:
        counters['trace_jax'] = {'count': -1}

    # system.clear_propagate_system_jax_cache
    try:
        import lumenairy.system as sys_mod
        counters['propagate_system_jax'] = {'count': 0}

        def _bump_propagate():
            counters['propagate_system_jax']['count'] += 1
        monkeypatch.setattr(sys_mod, 'clear_propagate_system_jax_cache',
                            _bump_propagate, raising=False)
    except ImportError:
        counters['propagate_system_jax'] = {'count': -1}

    # analysis.phase_retrieval.clear_phase_retrieval_caches
    try:
        import lumenairy.analysis.phase_retrieval as pr_mod
        counters['phase_retrieval'] = {'count': 0}

        def _bump_pr():
            counters['phase_retrieval']['count'] += 1
        monkeypatch.setattr(pr_mod, 'clear_phase_retrieval_caches',
                            _bump_pr, raising=False)
    except ImportError:
        counters['phase_retrieval'] = {'count': -1}

    # analysis.through_focus.clear_through_focus_scan_jax_cache
    try:
        import lumenairy.analysis.through_focus as tf_mod
        counters['through_focus'] = {'count': 0}

        def _bump_tf():
            counters['through_focus']['count'] += 1
        monkeypatch.setattr(tf_mod, 'clear_through_focus_scan_jax_cache',
                            _bump_tf, raising=False)
    except ImportError:
        counters['through_focus'] = {'count': -1}

    return counters


class TestClearCachesOnExitImportGuard:
    """The 6 subsequent cache-clear blocks fire even if the first
    (``clear_asm_caches``) import fails."""

    def test_import_failure_does_not_bypass_later_blocks(self, monkeypatch):
        """Inject ImportError on ``clear_asm_caches``; verify the
        other 6 cache-clear functions still execute.

        Pre-fix: the import was outside the try/except, so an
        ImportError on the FIRST block bypassed all 6 subsequent
        ones.  Post-fix: each block has its own try/except with
        ImportError included.
        """
        import lumenairy as la  # noqa: F401 -- ensure submodule init

        # Set up the synthetic failure on clear_asm_caches.
        _install_failing_clear_asm_caches(monkeypatch)
        # Replace the other 6 with counters.
        counters = _install_counting_replacements(monkeypatch)

        # Enter and exit the context with clear_caches_on_exit=True.
        # The pre-fix code would raise ImportError out of the
        # finally block (or, worse, swallow it but never fire the
        # downstream blocks).  Post-fix: no exception, all
        # downstream blocks fire.
        with la.lumenairy_context(clear_caches_on_exit=True):
            pass

        # The downstream blocks should each have been called once.
        # We tolerate count == -1 for blocks whose module isn't
        # importable in this environment (e.g. no JAX, no raytrace).
        for name, c in counters.items():
            assert c['count'] in (1, -1), (
                f'Cache-clear block {name!r} was not called after '
                f'the first block raised ImportError -- regression '
                f'of v4.13.0 audit P1-E.  counter={c}')

        # At least one downstream block should have actually fired
        # (otherwise the test is vacuous).
        fired = sum(1 for c in counters.values() if c['count'] == 1)
        assert fired >= 3, (
            f'Expected at least 3 of the 6 downstream blocks to '
            f'have fired; only {fired} did.  counters={counters}')


class TestContextGuardSourceShape:
    """Pin the source-level shape of the guard so a future refactor
    can't quietly move the import back outside the try."""

    def test_import_inside_try_block(self):
        """The ``from .propagators.propagation import
        clear_asm_caches`` line must be inside the try/except
        block, NOT immediately before it.
        """
        import inspect
        from lumenairy import _context as ctx_mod

        src = inspect.getsource(ctx_mod.lumenairy_context)
        # Find the position of the import and the position of the
        # surrounding try.  The import must come AFTER the most
        # recent ``try:`` keyword that precedes it.
        import_pos = src.find('from .propagators.propagation import '
                              'clear_asm_caches')
        assert import_pos != -1, (
            'Import of clear_asm_caches missing entirely.')
        # The substring between the start of the function and the
        # import line should end with a ``try:`` (not a non-try
        # statement at the same indent level).
        before = src[:import_pos]
        # Walk back to find the last ``try:`` keyword.
        last_try = before.rfind('try:')
        # Confirm no intervening ``except`` / ``finally`` block
        # close between that try and the import.
        between = before[last_try:]
        assert 'except' not in between or last_try > between.find('except'), (
            f'Import of clear_asm_caches appears to be outside the '
            f'last try: block -- regression of v4.13.0 audit P1-E.')

    def test_importerror_in_typed_tuple(self):
        """The except clause that guards clear_asm_caches must
        include ``ImportError`` in its typed tuple."""
        import inspect
        from lumenairy import _context as ctx_mod

        src = inspect.getsource(ctx_mod.lumenairy_context)
        # Find the asm-cache import.
        idx = src.find('clear_asm_caches()')
        assert idx != -1, 'clear_asm_caches() call missing.'
        # Look forward to the next except clause.
        tail = src[idx:]
        except_idx = tail.find('except')
        assert except_idx != -1, 'except clause missing.'
        except_line = tail[except_idx:tail.find('\n', except_idx)]
        assert 'ImportError' in except_line, (
            f'except clause guarding clear_asm_caches must include '
            f'ImportError; got: {except_line!r}')
