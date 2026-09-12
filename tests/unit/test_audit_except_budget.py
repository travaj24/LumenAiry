"""Non-``ui/`` broad-``except`` budget (extracted from ``test_audit_misc.py``).

EXTRACTED 2026-06-10 (v5.14.1 RCWA audit): ``test_audit_misc.py`` gained a
MODULE-LEVEL ``pytest.importorskip('jax')`` when the JAX audit pins were
appended, which silently skipped this budget guard (and every other
non-JAX pin below it) on CI -- jax is not installed there.  The count crept
13 -> 20 unobserved between v5.5.2 and v5.12.0.  This file has NO jax
dependency, so the guard runs everywhere again.

The judgement rule (AUDIT_V4_12_1_2026_05_16.md L5) is
NARROW > WARN-BEFORE-PASS > RE-RAISE > KEEP-AS-IS.  A broad ``except
Exception:`` is justified only where the exception type cannot be named
without importing an optional package (a JAX tracer's concretization error,
numba's compilation errors), where the guarded call is user-supplied code, or
in a teardown path that must never raise.

RE-CENSUSED 2026-09-12 (WP-A15a, audit 2026-09-11).  The scalar budget had
been a running total with a comment trail, which is why it read 48 while the
tree read 51 at the audit base and 55 mid-campaign: a scalar cannot say WHICH
file grew.  It is replaced by a PER-FILE census below -- every one of the 53
current sites was read and justified individually -- plus the scalar total as
a second bar.  Both are ``<=`` bars, so narrowing a clause never fails the
gate; adding one to a file does, and adding a file that is not in the census
at all does, because its implicit allowance is zero.

The three sites recorded as NARROWING REQUESTS below are counted (they exist
today) but are not endorsed; when their owners narrow them, lower the census
entry in the same change.
"""
import os
import re
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LUMENAIRY_DIR = os.path.join(REPO_ROOT, 'lumenairy')

# ---------------------------------------------------------------------------
# The census.  path (posix, relative to ``lumenairy/``) -> number of justified
# ``except Exception:`` clauses.  Grouped by the justification that licenses
# them; every group name is one of the sanctioned classes.
# ---------------------------------------------------------------------------

# (1) JAX-TRACER / CONCRETIZATION GUARDS -- the dominant class (34 of 53).
#     ``try: <materialize a value> except Exception: <conservative fallback>``
#     where the raised type is ``TracerArrayConversionError`` /
#     ``ConcretizationTypeError`` / ``TypeError`` depending on the jax version
#     and CANNOT be named without importing jax at module scope, which these
#     modules must not do (jax is optional).  Each one routes a traced input
#     to the general/exact path instead of a concrete-only fast path, so the
#     fallback is conservative by construction.
_TRACER_GUARDS = {
    'elements/_berreman_jax.py': 4,   # internal_field thicknesses; (3,3)-tensor
                                      # inspectability; is-traced; isotropy probe
    'elements/pmm/_core.py': 5,       # _resolve_incidence_checked, the jpmm
                                      # incidence guard, three _re_or_none/_n_or_none
    'elements/pmm/_jax_stack.py': 3,  # _concrete_real, _concrete_index, grazing guard
    'elements/pmm/_jax_stack2d.py': 1,
    'elements/pmm/_jax_twod.py': 3,   # _concrete + the host-guard pair
    'elements/pmm/oned.py': 2,        # _off_mag, pmm_jones_1d's scale probe
    'elements/pmm/stack.py': 3,       # add_layer traced thickness, solve's OOP
                                      # probe, _slices_consensus_check (a failed
                                      # clone solve is recorded as INFINITE
                                      # disagreement -- conservative)
    'elements/pmm/stack2d.py': 1,     # add_layer traced thickness
    'elements/rcwa/_core.py': 5,      # _is_traced, _require_jax_x64 (the
                                      # jax.config read API differs by version),
                                      # all-zero tensor probe, offplane-or-traced,
                                      # _reject_jax_offplane
    'elements/rcwa/oned.py': 2,       # _resolve_incidence_checked, _metallic
    'elements/rcwa/stack.py': 2,      # _layer_offplane_or_traced, _layer_eig_key
    'elements/rcwa/twod.py': 1,       # rcwa_jones_2d general-path routing
    'optimize/jax_merits.py': 1,      # _ensure_jax_x64
    'propagators/asymptotic_jax_twin.py': 1,  # _require_jax_x64
    'propagators/gbd.py': 1,          # _fft_reconstruct_applicable -> dense sum
}

# (2) OTHER UNTYPEABLE OPTIONAL-PACKAGE BOUNDARIES.
_OPTIONAL_PACKAGE_GUARDS = {
    # numba's build errors (TypingError / LoweringError / raw LLVM) are not a
    # stable public type set; on failure the numba path is disabled and the
    # exact NumPy dual runs.  (v5.28.0, roadmap R4.)
    'raytrace/differential.py': 1,
    # get_glass_index reaches the optional ``refractiveindex`` package, whose
    # lookup failures are not a declared type set.  One is a cache-key
    # fallback that degrades to an ('unresolved', name) key -- never a wrong
    # HIT -- and one RE-RAISES as ValueError, the sanctioned third tier.
    'elements/_lens_real.py': 2,      # _glass_key_value, _sag_callable_fingerprint
    'raytrace/exit_vertex.py': 1,     # resolve_exit_index -- re-raises ValueError
}

# (3) USER-SUPPLIED CODE AND BEST-EFFORT PROBES.  The guarded call is either a
#     caller's own callable or a probe whose only job is to answer "can this be
#     done?"; a raise there is data, not an error.
_PROBE_GUARDS = {
    'optimize/context.py': 2,         # pickle-ability probe (WARNS before
                                      # passing) and the merit-function probe call
    'io/storage.py': 3,               # h5py swmr_mode / dataset re-entry / zarr
                                      # append -- backend-specific raise types
    'analysis/psf_mtf_otf.py': 2,     # CubicSpline fit and brentq bracket in
                                      # sparrow_resolution -> nan / continue
    'elements/bor/coupled_radial_eigensolver.py': 1,  # step-index root census
    '_knobs.py': 1,                   # _same compares two knob values of
                                      # arbitrary type (an ndarray == ndarray
                                      # is not a bool); a non-comparison is
                                      # reported as "not equal"
}

# (4) TEARDOWN PATHS THAT MUST NOT RAISE.
_TEARDOWN_GUARDS = {
    '_context.py': 1,                 # atexit global-restore handler
    'optimize/driver.py': 1,          # __del__ dtype-restore guard
}

# (5) NARROWING REQUESTS -- counted because they exist, NOT endorsed.  Each is
#     a bare ``from X import Y`` (or a module attribute read) wrapped in
#     ``except Exception``, where ``except ImportError`` (plus AttributeError
#     for the attribute read) is the exact type set.  Recorded in
#     docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A15a_REPORT.md
#     as requests to the owning work packages; lower these entries when they land.
_NARROWING_REQUESTS = {
    'memory.py': 2,                   # set_low_memory's two fft_infra imports
    'elements/_lens_imap.py': 1,      # build_inverse_map's `from .. import memory`
}

_CENSUS: dict[str, int] = {}
for _group in (_TRACER_GUARDS, _OPTIONAL_PACKAGE_GUARDS, _PROBE_GUARDS,
               _TEARDOWN_GUARDS, _NARROWING_REQUESTS):
    for _k, _v in _group.items():
        _CENSUS[_k] = _CENSUS.get(_k, 0) + _v

# The scalar bar is the census total, not an independent number: 53 justified
# sites across 27 files, MEASURED 2026-09-12 on branch audit-fixes-2026-09.
# (History: 99 pre-sweep at v4.13.0; 48 was the last hand-maintained scalar,
# set at v5.28.0 and already 3 short of the tree at the 2026-09-11 audit base.)
_NON_UI_EXCEPT_BUDGET = sum(_CENSUS.values())

_PRE_SWEEP_COUNT = 99

_EXCEPT_RE = re.compile(r'^\s*except\s+Exception(\s+as\s+\w+)?\s*:')


def _census_except_exception_in_non_ui() -> Counter:
    """Walk ``lumenairy/*.py`` excluding ``ui/`` and count
    ``except Exception:`` / ``except Exception as ...:`` lines per file."""
    out: Counter = Counter()
    for root, dirs, files in os.walk(LUMENAIRY_DIR):
        if os.path.basename(root) == 'ui':
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d != 'ui' and not d.startswith('__')]
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, LUMENAIRY_DIR).replace(os.sep, '/')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if _EXCEPT_RE.match(line):
                            out[rel] += 1
            except OSError:
                pass
    return out


def _count_except_exception_in_non_ui() -> int:
    return sum(_census_except_exception_in_non_ui().values())


class TestExceptExceptionBudget:
    """The non-``ui/`` broad-``except`` count stays within budget."""

    def test_non_ui_except_exception_within_budget(self):
        n = _count_except_exception_in_non_ui()
        assert n <= _NON_UI_EXCEPT_BUDGET, (
            f"Non-``ui/`` ``except Exception:`` count is {n}, above the "
            f"budget of {_NON_UI_EXCEPT_BUDGET}.  A new broad-except crept "
            f"in.  See AUDIT_V4_12_1_2026_05_16.md L5 for the judgement rule "
            f"(NARROW > WARN-BEFORE-PASS > RE-RAISE > KEEP-AS-IS); only an "
            f"untypeable boundary (a jax tracer, numba's compilation errors, "
            f"user-supplied code) justifies KEEP-AS-IS + a budget bump.  "
            f"``test_non_ui_except_census_is_per_file`` names the file.")

    def test_non_ui_count_substantially_below_pre_sweep(self):
        """Pre-sweep (v4.13.0) was 99 non-ui clauses; the budget keeps the
        ~50% reduction pinned so a bulk reintroduction (e.g. a copy-paste
        from ``ui/``) gets caught."""
        n = _count_except_exception_in_non_ui()
        assert n <= _NON_UI_EXCEPT_BUDGET <= _PRE_SWEEP_COUNT

    def test_non_ui_except_census_is_per_file(self):
        """No file may carry MORE broad-excepts than its justified census
        entry, and a file absent from the census may carry none.

        This is the assertion the old scalar could not make.  A scalar budget
        with slack lets one module add a clause while another removes one, and
        the crept-in clause is never attributed -- which is exactly how the
        count reached 51 against a budget of 48 with the pin reading green on
        the way.  Per-file ``<=`` fails on the ADDITION regardless of what
        happened elsewhere, and narrowing a clause never fails the gate (the
        census entry is then stale-high; lower it in the same change).
        """
        actual = _census_except_exception_in_non_ui()
        over = {f: (n, _CENSUS.get(f, 0))
                for f, n in sorted(actual.items())
                if n > _CENSUS.get(f, 0)}
        assert not over, (
            "these lumenairy modules carry more broad ``except Exception:`` "
            "clauses than the justified census allows "
            "(file: actual vs allowed): "
            + ', '.join(f'{f}: {a} vs {b}' for f, (a, b) in over.items())
            + ".  Either NARROW the new clause to the exception types it "
            "actually needs, or -- if the raised type genuinely cannot be "
            "named without importing an optional package -- add it to the "
            "matching group in tests/unit/test_audit_except_budget.py with a "
            "one-line justification.  A bump without a justification is the "
            "defect this gate exists to catch.")

    def test_census_matches_the_tree_it_was_written_against(self):
        """Counter-pin: the census is not allowed to be uniformly slack.

        If every entry were set generously the per-file test above would pass
        for any addition.  This asserts the census TOTAL is not more than a
        small slack above the measured total, so the two tests together mean
        "each file is at its justified count, and the total is that sum".
        MEASURED 2026-09-12: census 53, tree 53, slack 0.
        """
        n = _count_except_exception_in_non_ui()
        assert _NON_UI_EXCEPT_BUDGET - n <= 3, (
            f"The per-file census allows {_NON_UI_EXCEPT_BUDGET} broad-excepts "
            f"but the tree only has {n}: {_NON_UI_EXCEPT_BUDGET - n} clauses of "
            f"unearned slack.  Clauses were narrowed (good) without lowering "
            f"their census entries, so the gate has gone loose.  Re-run the "
            f"census and lower the stale entries.")
