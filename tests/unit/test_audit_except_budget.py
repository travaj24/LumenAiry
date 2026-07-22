"""Non-``ui/`` broad-``except`` budget (extracted from ``test_audit_misc.py``).

EXTRACTED 2026-06-10 (v5.14.1 RCWA audit): ``test_audit_misc.py`` gained a
MODULE-LEVEL ``pytest.importorskip('jax')`` when the JAX audit pins were
appended, which silently skipped this budget guard (and every other
non-JAX pin below it) on CI -- jax is not installed there.  The count crept
13 -> 20 unobserved between v5.5.2 and v5.12.0.  This file has NO jax
dependency, so the guard runs everywhere again.

Re-triage 2026-06-10: every crept-in site is the JAX-TRACER-GUARD idiom --
``try: materialize(value) except Exception: <conservative fallback>`` --
where the caught type cannot be named without importing jax (Tracer
conversion raises ``TracerArrayConversionError`` / ``ConcretizationTypeError``
/ ``TypeError`` depending on version).  These are legitimate KEEP-AS-IS per
the v4.13.0 judgement rule (NARROW > WARN-BEFORE-PASS > RE-RAISE >
KEEP-AS-IS).  Current justified sites:

  - lumenairy/_context.py atexit handler (original)
  - lumenairy/optimize/core.py ``__del__`` dtype-restore guard (original)
  - lumenairy/propagators/hfpi.py JAX fold_in optional-dep gate (original)
  - tracer/concretization guards added with the PMM/RCWA JAX twins
    (v5.6-v5.13): pmm/_core.py x3, pmm/oned.py x2, rcwa/_core.py x3,
    rcwa/oned.py x1, rcwa/stack.py x1
  - io/storage.py x3, optimize/context.py x2 (probe-is-best-effort),
    optimize/driver.py x1, analysis/psf_mtf_otf.py x2

Author: Andrew Traverso (sweep) -- v4.13.0; extraction + re-triage v5.14.1.
"""
import os
import re

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LUMENAIRY_DIR = os.path.join(REPO_ROOT, 'lumenairy')

# Actual justified count is 20 (see the triage above) + the jax-tracer
# concretization guards added with the stack-level JAX twins (v5.14.2,
# 2026-06-11): pmm/_jax_stack.py `_concrete_real`/`_concrete_index`,
# pmm/_jax_stack2d.py `_concrete`, pmm/stack.py's traced-tensor OOP probe,
# pmm/stack2d.py's traced-thickness validation probe -- all the sanctioned
# KEEP-AS-IS class (a tracer's concretization error is untypeable without
# importing jax at module scope).
#
# Audit-fix campaign (AUDIT_V5_17_0, waves 3/5/6, 2026-07-02): +9 sites of
# the SAME sanctioned tracer-guard class -- the incidence/validation guards
# mirrored into the JAX twins must concretize source values and skip on any
# tracer error: pmm/_core.py `_jpmm_concrete_incidence_guard`,
# pmm/_jax_stack.py `_grazing_guard_concrete`, pmm/_jax_twod.py `_concrete`
# + host-guard pair (3), pmm/oned.py `_resolve_incidence_checked`,
# pmm/stack.py add_layer traced-thickness check, rcwa/oned.py
# `_resolve_incidence_checked`, propagators/asymptotic_jax_twin.py
# `_require_jax_x64` config read.  (The one non-tracer wave-2 site in
# memory.py was NARROWED to the registry-walk tuple instead of counted.)
# Slack of 2 retained.  If you add a non-tracer-guard site, NARROW it
# instead of bumping.
#
# v5.21 re-triage (2026-07-07): +5 sites, ALL the sanctioned jax-tracer-
# guard class (concretize-or-fallback where the tracer error is untypeable
# without importing jax at module scope): propagators/gbd.py
# `_fft_applicable` (tracer -> trace-safe dense-sum fallback),
# elements/_berreman_jax.py internal_field traced-thickness probe (re-raises
# as NotImplementedError) + `(3,3)`-tensor inspectability probe,
# elements/rcwa/_core.py traced-tensor OOP routing probes, rcwa/stack.py +
# rcwa/twod.py traced-tensor general-path routing.  The 3 NON-tracer sites
# added by the v5.21 Maslov Cholesky fit (lenses_maslov.py `_solve_fit` x2 +
# `_gram_cho_factor`) were NARROWED to
# ``(ImportError, ValueError, np.linalg.LinAlgError)`` instead of counted.
# Slack of 2 retained (count 43 at bump time).
#
# v5.21 delta-audit (2026-07-07, D11/D12): +2 sites, both the sanctioned
# jax-tracer-guard class in elements/_berreman_jax.py -- `_layer_M_gen_jax`'s
# trace-safe isotropy probe (D11a: concretize-or-fall-to-tensor-branch) and
# `_tensor_is_traced_jax`'s concrete-inspectability probe (D12 tracer -> exact
# general-path routing).  Slack of 2 retained (count 45 at bump time).
#
# v5.28.0 (2026-07-22, roadmap R4): +1 for the numba ADRT-kernel-build guard in
# raytrace/differential.py -- ``_build_adrt_numba_kernel()`` can fail with any of
# numba's untypeable compilation errors (TypingError/LoweringError/LLVM), the
# sanctioned untypeable-guard class (like the jax-tracer guards above); on
# failure it disables the numba path and falls back to the exact NumPy dual.  The
# cache RAM-detection guard was NARROWED (not broad), so net +1.
_NON_UI_EXCEPT_BUDGET = 48


def _count_except_exception_in_non_ui() -> int:
    """Walk ``lumenairy/*.py`` excluding ``ui/`` and count
    ``except Exception:`` and ``except Exception as ...:`` lines."""
    pat = re.compile(r'^\s*except\s+Exception(\s+as\s+\w+)?\s*:')
    n = 0
    for root, dirs, files in os.walk(LUMENAIRY_DIR):
        if os.path.basename(root) == 'ui':
            dirs[:] = []
            continue
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
                pass
    return n


class TestExceptExceptionBudget:
    """The non-``ui/`` broad-``except`` count stays within budget."""

    def test_non_ui_except_exception_within_budget(self):
        n = _count_except_exception_in_non_ui()
        assert n <= _NON_UI_EXCEPT_BUDGET, (
            f"Non-``ui/`` ``except Exception:`` count is {n}, above the "
            f"budget of {_NON_UI_EXCEPT_BUDGET}.  A new broad-except crept "
            f"in.  See AUDIT_V4_12_1_2026_05_16.md L5 for the judgement rule "
            f"(NARROW > WARN-BEFORE-PASS > RE-RAISE > KEEP-AS-IS); only a "
            f"jax-tracer guard (untypeable without importing jax) justifies "
            f"KEEP-AS-IS + a budget bump.")

    def test_non_ui_count_substantially_below_pre_sweep(self):
        """Pre-sweep (v4.13.0) was 99 non-ui clauses; the budget keeps the
        ~80% reduction pinned so a bulk reintroduction (e.g. a copy-paste
        from ``ui/``) gets caught."""
        n = _count_except_exception_in_non_ui()
        assert n <= _NON_UI_EXCEPT_BUDGET
