"""VERIFY-WP-C4 ROUND 2, item 13 -- the shipped mutation matrix re-run, plus
four mutants of this verification's own.

The five shipped mutations are applied through the shipped test module's own
helpers, so what is measured is the assertion the release carries.  The four
added here are the ones the shipped matrix does not cover:

* ``work_comparison_strict`` -- the SECOND condition compares ``>`` instead of
  ``>=``.  The constant is set to exactly the work ratio of the shape it was
  derived from (``256x64 -> 4x1`` reads exactly 16.00), so a strict comparison
  refuses the very shape the derivation names as the smallest measured safe.
  The ``_auto_selects_direct`` docstring states the ``<=`` explicitly.
* ``work_formula_max_instead_of_min`` -- the work count takes the MORE
  expensive association order, which ``_direct_matrix_2d`` never takes.  The
  screen then over-counts and captures thin shapes again.
* ``ratio_condition_dropped`` -- the FIRST condition is removed from the
  function while both constants keep their shipped values.
* ``mft_method_ignored_at_one_entry_point`` -- ``compute_psf`` accepts the
  keyword and drops it on the floor, which is the failure the census (a
  signature sweep) structurally cannot see.

    PYTHONPATH=<tree> python vc4b_mutations.py <tree>

Author:  Andrew Traverso
"""
from __future__ import annotations

import importlib.util
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(tree):
    la = L.anchor(tree)
    L.single_thread_ffts()
    repo = os.path.dirname(os.path.dirname(HERE))
    T = _load(os.path.join(repo, 'tests', 'unit',
                           'test_c4_mft_direct_default.py'), 'c4main')
    R = _load(os.path.join(repo, 'tests', 'unit',
                           'test_c4_round2_mft_method.py'), 'c4r2')
    from lumenairy.propagators import _bluestein as B

    # ---- the claim set: the shipped five, plus the round-2 file's ids -----
    def _round2_none_and_route():
        """The round-2 file's two spy ids, over every driven entry point."""
        import contextlib

        class _MP:
            def __init__(self):
                self._undo = []

            def setattr(self, obj, name, val):
                self._undo.append((obj, name, getattr(obj, name)))
                setattr(obj, name, val)

            def undo(self):
                for obj, name, val in reversed(self._undo):
                    setattr(obj, name, val)
                self._undo = []

        for entry in R._DRIVEN:
            mp = _MP()
            try:
                seen = R.spy.__wrapped__(mp)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    R._drive(entry, {})
                    silent = list(seen)
                    seen.clear()
                    R._drive(entry, {'mft_method': None})
                    with_none = list(seen)
                    seen.clear()
                    R._drive(entry, {'mft_method': 'bluestein'})
                    named = list(seen)
                assert silent and silent[0][1] == '<absent>', entry
                assert with_none == silent, entry
                assert named and named[0][1] == 'bluestein', entry
            finally:
                mp.undo()

    def _census():
        assert R._exported_entry_points_without_a_way_back() == []

    CLAIMS = [
        ('boundary', T._claim_the_boundary_comes_from_the_constant),
        ('work_screen', T._claim_the_work_screen_refuses_a_thin_input),
        ('dispatch_dense_side',
         lambda: T._claim_auto_dispatches_as_the_rule_says(
             T._DENSE_SIDE[:2], 'plain', False)),
        ('dispatch_chirp_side',
         lambda: T._claim_auto_dispatches_as_the_rule_says(
             T._CHIRP_SIDE[:2], 'plain', False)),
        ('way_back',
         lambda: T._claim_the_way_back_is_byte_identical(T._DENSE_SIDE[:1])),
        ('accuracy',
         lambda: T._claim_the_dense_side_is_the_more_accurate_side(
             [s for s in T._DENSE_SIDE
              if s[0] == s[1] and s[2] == s[3]][:1])),
        ('r2_census', _census),
        ('r2_none_stamps_nothing_and_route_arrives', _round2_none_and_route),
    ]

    def run_all():
        caught = []
        for name, call in CLAIMS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    call()
            except AssertionError:
                caught.append(name)
        return caught

    # ---- this verification's own mutants ---------------------------------
    # A source edit changes EVERY binding of the name, including the one the
    # shipped test module imported at import time.  Rebinding only
    # ``B._auto_selects_direct`` would model something a maintainer cannot do:
    # the predicate claims call the name they imported, so they would hold the
    # pre-mutation function and the mutant would read as uncatchable for a
    # reason that has nothing to do with the tests.  ``_rebind`` therefore
    # replaces the function at every module attribute that holds it.
    def _rebind(original, replacement):
        undo = []
        for mod in list(sys.modules.values()):
            name = getattr(mod, '__name__', '')
            if not (name.startswith('lumenairy') or name in ('c4main', 'c4r2')):
                continue
            if not hasattr(mod, '__dict__'):
                continue
            for attr in list(vars(mod)):
                try:
                    if getattr(mod, attr) is original:
                        setattr(mod, attr, replacement)
                        undo.append((mod, attr))
                except Exception:                            # noqa: BLE001
                    continue

        def restore():
            for mod, attr in undo:
                setattr(mod, attr, original)
        return restore

    def m_work_comparison_strict():
        """``flops > w*entries`` instead of ``>=``."""
        orig = B._auto_selects_direct

        def strict(ny, nx, my, mx):
            r = float(B._MFT_DIRECT_MAX_RATIO)
            if not (r > 0.0):
                return False
            ny, nx, my, mx = int(ny), int(nx), int(my), int(mx)
            if ny < 1 or nx < 1 or my < 1 or mx < 1:
                return False
            if r == float('inf'):
                return True
            if not (max(my / ny, mx / nx) <= r):
                return False
            entries = my * ny + mx * nx
            flops = min(my * ny * nx + my * nx * mx,
                        ny * nx * mx + my * ny * mx)
            return flops > float(
                B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries
        return _rebind(orig, strict)

    def m_work_formula_max():
        """The work count takes the association order the code never takes."""
        orig = B._auto_selects_direct

        def bymax(ny, nx, my, mx):
            r = float(B._MFT_DIRECT_MAX_RATIO)
            if not (r > 0.0):
                return False
            ny, nx, my, mx = int(ny), int(nx), int(my), int(mx)
            if ny < 1 or nx < 1 or my < 1 or mx < 1:
                return False
            if r == float('inf'):
                return True
            if not (max(my / ny, mx / nx) <= r):
                return False
            entries = my * ny + mx * nx
            flops = max(my * ny * nx + my * nx * mx,
                        ny * nx * mx + my * ny * mx)
            return flops >= float(
                B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries
        return _rebind(orig, bymax)

    def m_ratio_condition_dropped():
        """The FIRST condition removed, both constants left alone."""
        orig = B._auto_selects_direct

        def workonly(ny, nx, my, mx):
            r = float(B._MFT_DIRECT_MAX_RATIO)
            if not (r > 0.0):
                return False
            ny, nx, my, mx = int(ny), int(nx), int(my), int(mx)
            if ny < 1 or nx < 1 or my < 1 or mx < 1:
                return False
            if r == float('inf'):
                return True
            entries = my * ny + mx * nx
            flops = min(my * ny * nx + my * nx * mx,
                        ny * nx * mx + my * ny * mx)
            return flops >= float(
                B._MFT_DIRECT_MIN_WORK_PER_KERNEL_ENTRY) * entries
        return _rebind(orig, workonly)

    def m_mft_method_ignored_at_one_entry_point():
        """``compute_psf`` accepts ``mft_method=`` and drops it."""
        from lumenairy.analysis import psf_mtf_otf as P
        orig = P._compute_psf_mft

        def deaf(*args, **kwargs):
            kwargs.pop('mft_method', None)
            return orig(*args, **kwargs)
        P._compute_psf_mft = deaf
        return lambda: setattr(P, '_compute_psf_mft', orig)

    MUTANTS = dict(T._MUTATIONS)
    MUTANTS['work_comparison_strict'] = (m_work_comparison_strict, '?')
    MUTANTS['work_formula_max_instead_of_min'] = (m_work_formula_max, '?')
    MUTANTS['ratio_condition_dropped'] = (m_ratio_condition_dropped, '?')
    MUTANTS['mft_method_ignored_at_one_entry_point'] = (
        m_mft_method_ignored_at_one_entry_point, '?')

    out = {'build': L.build(), 'python': sys.version.split()[0],
           'lumenairy_version': la.__version__, 'rows': [],
           'control_caught_with_no_mutation': None}
    control = run_all()
    out['control_caught_with_no_mutation'] = control
    print("control (no mutation), claims refusing:", control, flush=True)
    for name in sorted(MUTANTS):
        apply_it, expected = MUTANTS[name]
        undo = apply_it()
        try:
            caught = run_all()
        finally:
            undo()
        row = {'mutation': name, 'shipped_expectation': expected,
               'claims_that_caught_it': caught,
               'caught_at_all': bool(caught),
               'expectation_holds': (expected in caught
                                     if expected != '?' else None)}
        out['rows'].append(row)
        print("  %-42s caught_by=%s  expected=%s  %s"
              % (name, caught or 'NOTHING', expected,
                 'OK' if (expected == '?' or expected in caught) else 'MISS'),
              flush=True)
    out['UNCAUGHT'] = [r['mutation'] for r in out['rows']
                       if not r['caught_at_all']]
    print("UNCAUGHT:", out['UNCAUGHT'])
    L.write(out, os.path.join(HERE, "vc4b_mutations_%s.json" % L.tag()))


if __name__ == '__main__':
    main(sys.argv[1])
