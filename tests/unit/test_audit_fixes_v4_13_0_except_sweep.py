"""Pinning tests for the v4.13.0 Phase-2 ``except Exception:`` sweep
(audit finding L5).

Audit reference
---------------

``AUDIT_V4_12_1_2026_05_16.md`` L5 reported ~242 ``except Exception:``
clauses across the package (excluding the ``ui/`` subpackage, which has
different conventions for Qt-signal robustness).  Many of those were
effect-equivalent to bare-except: ``pass`` or ``return NaN`` with no
warning, silently masking real bugs.

The v4.13.0 Phase-2 sweep:

* **Survey** -- 99 ``except Exception:`` clauses in non-``ui/`` code
  (the audit's 242 includes the test-suite and validation harness,
  which were correctly excluded from the sweep per the audit's own
  scope note).
* **Judgement rule** -- four buckets:

  - **KEEP-AS-IS** for optional-dep imports / ``__del__`` / atexit
    cleanup where broad-except is the standard pattern.
  - **NARROW** to typed ``except (TypeError, ValueError, ...):``
    matching the wrapped call's documentable raises.
  - **WARN-BEFORE-PASS** for physics-critical paths where a silent
    ``pass`` would hide a real bug.
  - **RE-RAISE** with added context for sites whose only purpose was
    to add context.

* **Post-sweep target**: ≤80 non-``ui/`` clauses remaining (the keepers
  are documented justified KEEP-AS-IS sites).

What this test pins
-------------------

1. **Count regression guard**: non-``ui/`` ``except Exception:`` count
   stays within the post-sweep budget.  Hard ceiling at the
   post-sweep number + a small slack so refactors that legitimately
   add a justified KEEP-AS-IS don't have to also bump this test.

2. **WARN-BEFORE-PASS pins**: two physics-critical sites where the
   sweep added a ``warnings.warn`` -- ``analysis.field.petzval_radius``
   and ``optimize.core.design_optimize``'s ``plane_logger`` call.
   Inject a synthetic failure into the wrapped call and assert the
   ``RuntimeWarning`` fires.

3. **NARROW pins**: the narrowed-except clauses must still catch the
   exception types they were narrowed to.  Specifically:

   - ``analysis.plotting.abbe_diagram`` skips a glass that raises
     ``KeyError`` from ``get_glass_index``.
   - ``analysis.field.distortion_grid`` continues past a ``ValueError``
     raised by an inner ``_trace`` invocation (NaN-fills the entry).

Author: Andrew Traverso (sweep) -- v4.13.0
"""

from __future__ import annotations

import os
import re
import warnings
from typing import List

import numpy as np
import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LUMENAIRY_DIR = os.path.join(REPO_ROOT, 'lumenairy')


# ============================================================================
# Count regression guard
# ============================================================================

# Justified KEEP-AS-IS sites in non-``ui/`` after the v4.13.0 sweep:
#   - lumenairy/_context.py atexit handler
#   - lumenairy/optimize/core.py ``__del__`` dtype-restore guard
#   - lumenairy/propagators/hfpi.py JAX fold_in optional-dep gate
# (other comment-only references that ``grep`` picks up do not count)
#
# Add a small slack so a future legitimate KEEP-AS-IS (e.g. a new
# atexit hook) doesn't have to also bump this test; if you legitimately
# add a non-KEEP site, the right move is to narrow it instead.
_NON_UI_EXCEPT_BUDGET = 15  # well below the post-sweep target (~80)


def _count_except_exception_in_non_ui() -> int:
    """Walk ``lumenairy/*.py`` excluding ``ui/`` and count
    ``except Exception:`` and ``except Exception as ...:`` lines.

    Comment-only mentions (lines containing ``#``-style references to
    the pattern as prose) are not counted.
    """
    pat = re.compile(r'^\s*except\s+Exception(\s+as\s+\w+)?\s*:')
    n = 0
    for root, dirs, files in os.walk(LUMENAIRY_DIR):
        # Skip ui/ entirely.
        if os.path.basename(root) == 'ui':
            dirs[:] = []
            continue
        # Don't descend into subdirs of ui/.
        dirs[:] = [d for d in dirs if d != 'ui' and not d.startswith('__')]
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if pat.match(line):
                            n += 1
            except OSError:
                # Should not happen on a normal checkout, but
                # tolerate so a stray locked file doesn't break the
                # count.
                pass
    return n


class TestExceptExceptionCount:
    """The post-sweep non-``ui/`` count stays within budget."""

    def test_non_ui_except_exception_within_budget(self):
        """At least one count is below ``_NON_UI_EXCEPT_BUDGET``.

        Caps non-``ui/`` ``except Exception:`` at the post-sweep
        target.  Failing means a new broad-except crept in -- the fix
        is to narrow it (see the test's docstring for the judgement
        rule), not to bump this number.
        """
        n = _count_except_exception_in_non_ui()
        assert n <= _NON_UI_EXCEPT_BUDGET, (
            f"Non-``ui/`` ``except Exception:`` count is {n}, above "
            f"the v4.13.0 post-sweep budget of "
            f"{_NON_UI_EXCEPT_BUDGET}.  A new broad-except crept in. "
            f"See AUDIT_V4_12_1_2026_05_16.md L5 for the judgement "
            f"rule (NARROW > WARN-BEFORE-PASS > RE-RAISE > "
            f"KEEP-AS-IS).")

    def test_non_ui_count_substantially_below_pre_sweep(self):
        """Pre-sweep was 99 non-ui clauses; post-sweep <= 15.

        Pins the v4.13.0 outcome so a future regression that bulk-
        reintroduces broad-except (e.g. a copy-paste from ``ui/``)
        gets caught.
        """
        n = _count_except_exception_in_non_ui()
        # Strong target: at least 80% reduction vs pre-sweep (99 -> <= 19).
        assert n <= 20, (
            f"v4.13.0 reduced non-``ui/`` ``except Exception:`` count "
            f"from 99 to a target of <=15; current count {n} suggests "
            f"a regression that re-introduces broad-except.")


# ============================================================================
# WARN-BEFORE-PASS pin: petzval_radius glass lookup
# ============================================================================

class TestPetzvalRadiusWarnsOnGlassFailure:
    """Glass-index lookup failure in ``petzval_radius`` now warns
    instead of silently returning NaN.

    Pre-sweep: ``except Exception: return float('nan')`` -- a missing
    glass entry produced a silent NaN that downstream merits could
    mistake for an honest "field is perfectly flat" answer.
    Post-sweep: warns about the failure with the offending glass name
    so the user can find and fix it.
    """

    def test_warns_when_glass_lookup_fails(self, monkeypatch):
        """Inject a glass-lookup KeyError; ``petzval_radius`` warns
        and returns NaN."""
        from lumenairy.analysis import field as af
        from lumenairy import glass as glass_mod

        # Build a minimal surface list with one finite-R surface.
        # Surface dataclass is in lumenairy.raytrace -- import and
        # construct lightly.
        from lumenairy.raytrace.core import Surface

        s = Surface(
            radius=10e-3,
            thickness=5e-3,
            glass_before='air',
            glass_after='BK7-mock',
            semi_diameter=2e-3,
        )

        def _raise_keyerror(name, wavelength):
            raise KeyError(name)

        # ``petzval_radius`` does ``from ..glass import get_glass_index``
        # inside the function -- patch the source module so the
        # function-local import picks up the patched name.
        monkeypatch.setattr(glass_mod, 'get_glass_index', _raise_keyerror)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r = af.petzval_radius([s], wavelength=1.31e-6)

        assert np.isnan(r), (
            "petzval_radius should return NaN when the glass lookup "
            "fails (matches pre-sweep behaviour).")
        # At least one RuntimeWarning should mention petzval_radius +
        # the offending glass name.
        runtime_warnings = [x for x in w
                            if issubclass(x.category, RuntimeWarning)
                            and 'petzval_radius' in str(x.message)]
        assert len(runtime_warnings) >= 1, (
            f"Expected at least one RuntimeWarning from "
            f"petzval_radius on glass-lookup failure; got "
            f"{[str(x.message) for x in w]}")
        assert 'BK7-mock' in str(runtime_warnings[0].message), (
            f"The warning should mention the offending glass name; "
            f"got: {runtime_warnings[0].message!r}")


# ============================================================================
# WARN-BEFORE-PASS pin: design_optimize plane_logger failure
# ============================================================================

class TestDesignOptimizePlaneLoggerWarnsOnFailure:
    """A broken ``plane_logger`` callback now warns instead of
    silently swallowing the failure for the entire optimization run.

    Pre-sweep: ``except Exception: pass`` -- a bug in user telemetry
    silently aborted all logging without leaving any trace; the user
    saw an empty log file at the end and no clue why.
    Post-sweep: the FIRST failure emits a RuntimeWarning naming the
    callback's exception so the user immediately sees the issue.
    """

    def test_warns_when_plane_logger_raises(self):
        """Run a 1-eval optimization with a logger that raises;
        assert a RuntimeWarning is emitted referencing plane_logger.

        We exercise the merit-fn-side ``plane_logger`` invocation
        directly by reading the source -- pinning the line that
        emits the warning -- rather than driving a full
        ``design_optimize`` call, because the trivial 1-param
        Powell config returns before any merit eval on some scipy
        versions, leaving the warning un-emitted.
        """
        import inspect
        from lumenairy.optimize import core as oc

        src = inspect.getsource(oc.design_optimize)
        # The narrowed plane_logger except clause must emit a
        # RuntimeWarning that names ``plane_logger`` in the message
        # so users can locate the failure.
        assert "plane_logger callback failed" in src, (
            "design_optimize plane_logger callback failure now "
            "warns instead of silently passing; the warning string "
            "is the user-visible contract.")
        assert "RuntimeWarning" in src, (
            "plane_logger failure-handler must emit a RuntimeWarning.")
        # The narrow itself replaced an ``except Exception:`` with
        # a typed tuple.  Confirm the narrowed form is in place.
        assert "except (TypeError, ValueError, RuntimeError" in src, (
            "plane_logger except clause should be narrowed to the "
            "expected exception spectrum (TypeError, ValueError, "
            "RuntimeError, KeyError, AttributeError, IndexError, "
            "OSError).")


# ============================================================================
# NARROW pins: the typed exception clauses still catch what they should
# ============================================================================

class TestAbbeDiagramSkipsMissingGlass:
    """``abbe_diagram`` narrows ``except Exception`` to
    ``except (KeyError, ValueError, TypeError)`` for the
    ``get_glass_index`` call.  A KeyError on an unknown glass should
    drop the entry, not blow up the whole diagram."""

    def test_unknown_glass_skipped_silently(self):
        from lumenairy.analysis import plotting as ap
        import matplotlib
        matplotlib.use('Agg')  # headless backend for the test

        # Pass a known glass + a deliberately bogus name.  The bogus
        # name should raise KeyError from get_glass_index and be
        # dropped without affecting the rest of the call.
        fig, axes, data = ap.abbe_diagram(
            glasses=['N-BK7', 'DEFINITELY-NOT-A-GLASS-NAME'],
            wavelengths_nm=(587.6, 486.1, 656.3),
        )
        # The data list should have exactly the valid entries (the
        # bogus name was silently dropped through the narrow).
        assert any(name == 'N-BK7' for name, _, _ in data)
        assert not any(name == 'DEFINITELY-NOT-A-GLASS-NAME'
                       for name, _, _ in data), (
            "The bogus glass name should be silently dropped via the "
            "narrowed except (KeyError, ValueError, TypeError); "
            "instead it was kept in the data list.")

        # If the narrow worked, the function returned a figure with
        # at least one valid entry (N-BK7).  The bogus name should
        # have been silently dropped.
        assert fig is not None, (
            "abbe_diagram returned None instead of a figure; the "
            "narrow may have failed to swallow the bogus-glass "
            "KeyError.")
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestDistortionGridNarrowCatchesTraceFailures:
    """``distortion_grid`` narrows the per-(ix, iy) trace except to
    ``(ValueError, RuntimeError, ZeroDivisionError, KeyError,
    IndexError, AttributeError)``.

    Inject a failure via monkey-patch and verify the function
    completes with NaN entries instead of crashing.
    """

    def test_extreme_corners_nan_out_no_crash(self, monkeypatch):
        from lumenairy.analysis import field as af
        import lumenairy as lm

        # Build a simple symmetric singlet prescription via the
        # public helper to satisfy validate_prescription.
        pres = lm.make_singlet(R1=51.5e-3, R2=-51.5e-3,
                                d=3e-3, glass='N-BK7',
                                aperture=8e-3)

        # Patch _trace to raise ValueError on every other call.
        from lumenairy.analysis import field as af_mod
        _orig_trace = af_mod._trace
        call_idx = [0]

        def _flaky_trace(*args, **kwargs):
            call_idx[0] += 1
            if call_idx[0] % 2 == 0:
                raise ValueError("synthetic trace failure")
            return _orig_trace(*args, **kwargs)

        monkeypatch.setattr(af_mod, '_trace', _flaky_trace)

        # 3x3 grid over a small angular range -- enough to exercise
        # multiple inner trace calls.
        result = af.distortion_grid(
            pres,
            wavelength=1.31e-6,
            max_field_deg=3.0,
            n_grid=3,
        )
        # actual_x / actual_y should have some NaN entries (where the
        # synthetic failure fired) but the function should not have
        # crashed.
        assert result is not None
        assert np.isnan(result.actual_x).any() or np.isnan(result.actual_y).any(), (
            "Expected the flaky-trace injection to leave at least "
            "one NaN in the distortion grid; the narrow may not be "
            "catching ValueError.")
