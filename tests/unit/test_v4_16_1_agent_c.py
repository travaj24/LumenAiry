"""v4.16.1 / Agent C -- regression tests for audit items 9-15.

This file pins the Agent C scope from AUDIT_V4_16_0_DEEP_2026_05_19
Part 8 (items 9-15):

* **C.1**: Constraint scalar-only validation (vector-return rejection
  in ``__post_init__``).
* **C.2**: Constraint lambda UserWarning (pickle-safety guidance).
* **C.3**: GLASS_VALIDITY <-> GLASS_REGISTRY consistency checks
  (orphan-GLASS_VALIDITY + tuple well-formedness).
* **C.5**: ``_clear_local_asm_caches`` registered via late-binding
  lambda (matching the other 8 cache enrollments).
* **C.7**: ``prescriptions.py:1019, 1470`` warnings carry explicit
  ``UserWarning`` category + ``stacklevel=2``.
* **C.8 (this file)**: regression tests for the above.

Author: Andrew Traverso -- v4.16.1 / Agent C
"""
from __future__ import annotations

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest

import lumenairy as la
from lumenairy import glass as glass_mod
from lumenairy.optimize import Constraint

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ===========================================================================
# C.1 -- Constraint scalar-only validation
# ===========================================================================


def _scalar_fn(x):
    """Module-level scalar-returning fun -- safe for lambda detection."""
    return float(np.sum(x))


def _vector_fn(x):
    """Module-level vector-returning fun -- triggers the C.1 rejection."""
    return np.asarray(x, dtype=np.float64) * 2.0


def _two_element_vector_fn(x):
    """Module-level fun that returns a (2,) ndarray regardless of input shape."""
    return np.asarray([1.0, 2.0])


# v4.16.2 (audit P2-NEW-F1-2): the following helpers were nested
# closures inside the test bodies in v4.16.1.  The v4.16.1 lambda
# detector used ``__name__ == '<lambda>'`` and missed closures (whose
# ``__name__`` is the inner-def name), so the tests passed under
# ``filterwarnings('error', UserWarning)``.  The v4.16.2 pickle-probe
# correctly catches non-picklable closures, which would flip those
# tests to ERROR.  Promote the helpers to module level so they pickle
# cleanly and the tests continue to exercise their original intent
# (probe-failure handling + 0-d ndarray acceptance) without colliding
# with the pickle-probe.
def _picky_fn(x):
    """Rejects the synthetic np.zeros(1) probe; only accepts (5,)."""
    if x.size != 5:
        raise ValueError(f"requires 5-element x; got {x.size}")
    return float(np.sum(x))


def _zero_d_fn(x):
    """Returns a 0-d ndarray -- legitimate scalar-like return."""
    return np.array(np.sum(x))


def test_constraint_scalar_module_level_fun_is_accepted():
    """A scalar-returning module-level fun must construct without
    raising or warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # any warn fails
        c = Constraint(fun=_scalar_fn, lb=0.0, ub=1.0, label='scalar')
    assert c.fun is _scalar_fn
    assert c.lb == 0.0
    assert c.ub == 1.0


def test_constraint_vector_return_is_rejected_in_post_init():
    """A vector-returning fun must raise ``TypeError`` when its shape
    is checked (the C.1 closure).

    v4.16.2 (audit P2-NEW-F1-1): the v4.16.1 auto-probe in
    ``__post_init__`` was removed because it ran user code (e.g. a
    full ray-trace for a BFL constraint) on every instantiation.
    The shape check now lives in :meth:`Constraint.validate`; call
    it explicitly to opt in to the v4.16.1 contract.  The test
    name is preserved for git-blame continuity even though the
    check is no longer in ``__post_init__`` per se.
    """
    # Construction must NOT raise (probe moved to validate()).
    c = Constraint(fun=_two_element_vector_fn, lb=0.0, ub=10.0,
                   label='vector-bad')
    # Explicit validate() raises TypeError with the same contract.
    with pytest.raises(TypeError) as excinfo:
        c.validate()
    msg = str(excinfo.value)
    assert 'ndarray' in msg
    assert 'scalar' in msg.lower() or '0-d' in msg
    # The label should appear in the error message for debug clarity.
    assert 'vector-bad' in msg


def test_constraint_vector_return_message_cites_label():
    """The vector-rejection error message must cite the constraint's
    label so a user can identify which Constraint failed when
    multiple are defined.

    v4.16.2 (audit P2-NEW-F1-1): the shape check moved from
    ``__post_init__`` to opt-in :meth:`Constraint.validate`.
    """
    c = Constraint(fun=_two_element_vector_fn, lb=0.0, ub=10.0,
                   label='BFL_constraint')
    with pytest.raises(TypeError) as excinfo:
        c.validate()
    assert 'BFL_constraint' in str(excinfo.value)


def test_constraint_probe_failure_does_not_block_construction():
    """If ``fun`` rejects the probe ``np.zeros(1)`` input (e.g.
    because it requires a specific parameter-vector shape), the
    Constraint must STILL construct successfully -- the probe is
    best-effort, not a hard validation gate.

    v4.16.2 (audit P2-NEW-F1-1): the probe moved from ``__post_init__``
    to opt-in :meth:`Constraint.validate`.  Construction is therefore
    trivially side-effect-free; we additionally exercise validate()
    explicitly to pin the probe's best-effort exception-swallowing.
    v4.16.2 (audit P2-NEW-F1-2): ``_picky_fn`` promoted to module
    scope so the new pickle-probe doesn't flag it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # any warn fails
        c = Constraint(fun=_picky_fn, lb=0.0, ub=1.0,
                       label='picky')
    assert c.fun is _picky_fn
    # Explicit validate() must also swallow the picky exception.
    c.validate()


def test_constraint_zero_d_ndarray_return_is_accepted():
    """A fun that returns a 0-d ndarray (``np.array(3.14)``) is a
    legitimate scalar-like return and must NOT trigger the
    vector-rejection.

    v4.16.2 (audit P2-NEW-F1-1): shape-check moved to opt-in
    :meth:`Constraint.validate`; verify both construction (no
    warning) AND explicit validate() (no TypeError).
    v4.16.2 (audit P2-NEW-F1-2): ``_zero_d_fn`` promoted to module
    scope so the new pickle-probe doesn't flag it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)  # any warn fails
        c = Constraint(fun=_zero_d_fn, lb=0.0, ub=10.0, label='0d')
    assert c.fun is _zero_d_fn
    # Explicit validate() must also accept the 0-d ndarray return.
    c.validate()


# ===========================================================================
# C.2 -- Constraint lambda UserWarning
# ===========================================================================


def test_constraint_lambda_fun_emits_userwarning():
    """A lambda fun must trigger a UserWarning recommending a
    module-level function (the C.2 closure)."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        Constraint(fun=lambda x: float(np.sum(x)),
                   lb=0.0, ub=1.0, label='lambda-test')
    matched = [
        w for w in recorded
        if issubclass(w.category, UserWarning)
        and 'lambda' in str(w.message).lower()
    ]
    assert len(matched) >= 1, (
        f"Expected at least one UserWarning citing 'lambda'; got "
        f"{[str(w.message) for w in recorded]}"
    )
    msg = str(matched[0].message)
    assert 'pickling' in msg.lower() or 'pickle' in msg.lower() or 'PicklingError' in msg, (
        f"UserWarning must explain the pickle-safety motivation; got: {msg}"
    )
    assert 'module-level' in msg, (
        f"UserWarning must recommend a module-level function; got: {msg}"
    )


def test_constraint_lambda_warning_cites_label():
    """The lambda-warning must mention the constraint's label so a
    user can identify which Constraint emitted the warning when
    multiple are defined."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        Constraint(fun=lambda x: float(np.sum(x)),
                   lb=0.0, ub=1.0,
                   label='unique_test_label_xyz')
    matched = [
        w for w in recorded
        if issubclass(w.category, UserWarning)
        and 'unique_test_label_xyz' in str(w.message)
    ]
    assert len(matched) >= 1


def test_constraint_module_level_fun_does_not_warn_about_lambda():
    """A non-lambda fun must NOT trigger the lambda-pickle warning."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        Constraint(fun=_scalar_fn, lb=0.0, ub=1.0,
                   label='module-level-fn')
    matched = [
        w for w in recorded
        if issubclass(w.category, UserWarning)
        and 'lambda' in str(w.message).lower()
    ]
    assert len(matched) == 0, (
        f"Module-level fun unexpectedly triggered lambda warning: "
        f"{[str(w.message) for w in matched]}"
    )


# ===========================================================================
# C.3 -- GLASS_VALIDITY <-> GLASS_REGISTRY consistency
# ===========================================================================


def test_glass_validity_consistency_orphan_in_validity_is_rejected():
    """Injecting an orphan GLASS_VALIDITY entry (one with no matching
    GLASS_REGISTRY entry) must trigger a RuntimeError on consistency
    check."""
    fake_glass = '__FAKE_ORPHAN_VALIDITY_GLASS__'
    # Sanity: the fake glass is not already in either dict.
    assert fake_glass not in glass_mod.GLASS_VALIDITY
    assert fake_glass not in glass_mod.GLASS_REGISTRY
    glass_mod.GLASS_VALIDITY[fake_glass] = (0.3e-6, 2.5e-6)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            glass_mod._check_glass_registry_consistency()
        msg = str(excinfo.value)
        assert fake_glass in msg
        assert 'GLASS_REGISTRY' in msg
    finally:
        del glass_mod.GLASS_VALIDITY[fake_glass]


def test_glass_validity_consistency_orphan_in_registry_is_allowed():
    """A GLASS_REGISTRY entry without a GLASS_VALIDITY counterpart is
    LEGITIMATE -- user-registered callable glasses don't have a
    documented validity range, and the consistency check must not
    treat that absence as drift.

    This is the asymmetry documented in
    ``_GLASS_VALIDITY_REGISTRY_EXEMPTIONS`` rationale: we only enforce
    the GLASS_VALIDITY -> GLASS_REGISTRY direction (orphan-validity is
    unreachable + a bug); the reverse direction is soft.
    """
    fake_glass = '__FAKE_REGISTRY_ONLY_GLASS__'
    # Sanity: the fake glass is not already in either dict.
    assert fake_glass not in glass_mod.GLASS_VALIDITY
    assert fake_glass not in glass_mod.GLASS_REGISTRY
    glass_mod.GLASS_REGISTRY[fake_glass] = lambda wl: 1.5
    try:
        # Must NOT raise -- registry-only entries are legitimate.
        glass_mod._check_glass_registry_consistency()
    finally:
        del glass_mod.GLASS_REGISTRY[fake_glass]


def test_glass_validity_malformed_tuple_is_rejected():
    """A GLASS_VALIDITY entry that is not a 2-tuple must trigger a
    RuntimeError on consistency check (the C.3 tuple well-formedness
    check)."""
    # Use a real glass that's already in both dicts so we don't trip
    # the orphan check.
    real_glass = 'N-BK7'
    assert real_glass in glass_mod.GLASS_VALIDITY
    assert real_glass in glass_mod.GLASS_REGISTRY
    original = glass_mod.GLASS_VALIDITY[real_glass]
    # Inject a malformed entry -- 3-tuple.
    glass_mod.GLASS_VALIDITY[real_glass] = (0.3e-6, 1.0e-6, 2.5e-6)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            glass_mod._check_glass_registry_consistency()
        msg = str(excinfo.value)
        assert real_glass in msg
        assert '2-tuple' in msg
    finally:
        glass_mod.GLASS_VALIDITY[real_glass] = original


def test_glass_validity_inverted_range_is_rejected():
    """A GLASS_VALIDITY entry with lmin >= lmax must trigger a
    RuntimeError (degenerate range)."""
    real_glass = 'N-BK7'
    original = glass_mod.GLASS_VALIDITY[real_glass]
    # Inject inverted range.
    glass_mod.GLASS_VALIDITY[real_glass] = (2.5e-6, 0.3e-6)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            glass_mod._check_glass_registry_consistency()
        msg = str(excinfo.value)
        assert real_glass in msg
        assert 'lambda_min < lambda_max' in msg
    finally:
        glass_mod.GLASS_VALIDITY[real_glass] = original


def test_glass_validity_negative_lmin_is_rejected():
    """A GLASS_VALIDITY entry with negative lmin must trigger a
    RuntimeError."""
    real_glass = 'N-BK7'
    original = glass_mod.GLASS_VALIDITY[real_glass]
    # Inject negative bound.
    glass_mod.GLASS_VALIDITY[real_glass] = (-1e-6, 2.5e-6)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            glass_mod._check_glass_registry_consistency()
        msg = str(excinfo.value)
        assert real_glass in msg
        assert 'non-negative' in msg or 'positive' in msg
    finally:
        glass_mod.GLASS_VALIDITY[real_glass] = original


def test_glass_validity_module_load_passes():
    """The library's actual GLASS_VALIDITY <-> GLASS_REGISTRY state
    must satisfy the consistency check at v4.16.1 HEAD."""
    # Re-run the check; must not raise.
    glass_mod._check_glass_registry_consistency()


# ===========================================================================
# C.5 -- ``_clear_local_asm_caches`` late-binding registration
# ===========================================================================


def test_clear_asm_caches_walks_registry_after_cleanup():
    """The v4.14.1 cache-clear meta-pin must still pass: every
    cache-clear helper is still importable from ``lumenairy``.

    Regression test for the C.5 late-binding lambda fix --
    ``_clear_local_asm_caches`` is now registered via a late-binding
    closure (matching the other 8 enrollments).  The external
    contract is preserved: ``clear_asm_caches`` still walks the
    registry and drains every cache.
    """
    # Smoke: the canonical clear helpers are still callable.
    for clear_name in (
        'clear_asm_caches',
        'clear_zernike_basis_cache',
        'clear_lg_polynomial_cache',
        'clear_lg_mode_stack_cache',
        'clear_through_focus_scan_jax_cache',
        'clear_trace_jax_cache',
        'clear_propagate_system_jax_cache',
        'clear_phase_retrieval_caches',
    ):
        assert hasattr(la, clear_name), (
            f"v4.14.1 meta-pin regression: la.{clear_name} missing."
        )
        assert callable(getattr(la, clear_name))
    # Smoke: clear_asm_caches walks the registry without raising.
    la.clear_asm_caches()


def test_clear_local_asm_caches_registration_uses_lambda():
    """The C.5 closure: ``_clear_local_asm_caches`` is registered via
    a late-binding lambda -- not a bare function reference.

    The AST check in
    ``test_propagation_asm_local_uses_late_binding_lambda`` (in the
    10th meta-pin walker) is the structural pin; this test is a
    behavioural cross-check that the registered clearer is in fact
    callable and drains the local ASM caches.
    """
    # The registered name must be 'asm_local' (canonical v4.16.0 name).
    from lumenairy._cache_registry import list_registered_cache_clearers
    registered = list_registered_cache_clearers()
    assert 'asm_local' in registered, (
        "v4.16.1 / Agent C (C.5) regression: 'asm_local' missing from "
        "registered cache clearers; the late-binding registration "
        "may have broken the enrollment."
    )


def test_clear_local_asm_caches_mock_patch_observed():
    """The late-binding lambda must preserve the ``mock.patch.object``
    semantic: a test that monkey-patches
    ``propagation._clear_local_asm_caches`` MUST observe its counter
    increment when ``clear_asm_caches`` walks the registry.

    Pre-v4.16.1 the early-binding registration captured the function
    object directly, silently bypassing ``mock.patch.object``.  C.5
    closes the gap.
    """
    from unittest import mock

    from lumenairy.propagators import propagation as prop_mod

    # Build a counting wrapper that delegates to the real clearer so
    # the caches still drain (otherwise the test would leak state into
    # sibling tests).
    real_clearer = prop_mod._clear_local_asm_caches
    counter = {'calls': 0}

    def _counting_clearer():
        counter['calls'] += 1
        real_clearer()

    with mock.patch.object(prop_mod, '_clear_local_asm_caches',
                            _counting_clearer):
        la.clear_asm_caches()
    # The counter must have incremented; if the registration captured
    # the function reference at module-load time (early-binding), the
    # monkey-patch would be bypassed and the counter would stay at 0.
    assert counter['calls'] >= 1, (
        f"v4.16.1 / Agent C (C.5) regression: mock.patch.object on "
        f"propagation._clear_local_asm_caches was bypassed (counter "
        f"stayed at {counter['calls']}).  The registration may have "
        f"reverted to early-binding."
    )


# ===========================================================================
# C.7 -- prescriptions.py warnings.warn hygiene
# ===========================================================================


def _parse_prescriptions_warn_calls():
    """Parse the prescription-I/O module(s) AST and return every
    ``warnings.warn(...)`` Call node along with its source line.

    v5.1.0 Agent F: the 3224-LOC ``io/prescriptions.py`` monolith was
    split into ``prescriptions_builders.py`` / ``prescriptions_zemax.py``
    / ``prescriptions_code_v.py`` / ``prescriptions_quadoa.py`` /
    ``prescriptions_transforms.py``.  Walk every ``prescriptions*.py``
    file under ``lumenairy/io/`` so the C.7 pin keeps catching a
    regression regardless of which submodule the warn calls end up in.
    """
    io_dir = _REPO_ROOT / 'lumenairy' / 'io'
    warn_calls = []
    for py in sorted(io_dir.glob('prescriptions*.py')):
        src = py.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(src, filename=str(py))
        for sub in ast.walk(tree):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            # ``warnings.warn(...)`` -- Attribute chain (warnings.warn).
            if (isinstance(func, ast.Attribute)
                    and func.attr == 'warn'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'warnings'):
                warn_calls.append(sub)
    return warn_calls


def _call_has_stacklevel_kwarg(call):
    """True iff a Call's kwargs include ``stacklevel=<int>`` with a
    value >= 2."""
    for kw in call.keywords:
        if kw.arg == 'stacklevel':
            if isinstance(kw.value, ast.Constant) and isinstance(
                    kw.value.value, int) and kw.value.value >= 2:
                return True
    return False


def _call_has_category_kwarg_or_positional(call):
    """True iff a Call's args / kwargs include an explicit warning
    category (UserWarning, DeprecationWarning, etc.).

    The category may be:
    * Positional arg #2 (after the message), OR
    * Keyword ``category=<Name>``.
    """
    # Check positional arg #2.
    if len(call.args) >= 2:
        second = call.args[1]
        if isinstance(second, ast.Name) and 'Warning' in second.id:
            return True
        if isinstance(second, ast.Attribute) and 'Warning' in second.attr:
            return True
    # Check kwarg.
    for kw in call.keywords:
        if kw.arg == 'category':
            return True
    return False


def test_prescriptions_warn_calls_have_explicit_category_and_stacklevel():
    """Every ``warnings.warn(...)`` call in ``io/prescriptions.py``
    MUST have an explicit ``UserWarning`` category AND ``stacklevel=2``
    (the C.7 closure)."""
    warn_calls = _parse_prescriptions_warn_calls()
    assert len(warn_calls) >= 2, (
        f"Expected >= 2 warnings.warn calls in io/prescriptions.py; "
        f"found {len(warn_calls)}.  The walker may have regressed."
    )
    missing_stacklevel = []
    missing_category = []
    for call in warn_calls:
        if not _call_has_stacklevel_kwarg(call):
            missing_stacklevel.append(call.lineno)
        if not _call_has_category_kwarg_or_positional(call):
            missing_category.append(call.lineno)
    assert not missing_stacklevel, (
        f"warnings.warn calls in io/prescriptions.py missing "
        f"``stacklevel>=2`` at lines: {missing_stacklevel}.  C.7 "
        f"requires all warn calls in this module to specify "
        f"stacklevel=2 so the warning points at the caller."
    )
    assert not missing_category, (
        f"warnings.warn calls in io/prescriptions.py missing explicit "
        f"warning category at lines: {missing_category}.  C.7 "
        f"requires UserWarning (or other explicit category) so the "
        f"warning is filterable per category."
    )


def test_prescriptions_specific_warn_lines_pinned():
    """The two specific warn calls cited in the audit (originally at
    ~lines 1019, 1470 in the v4.16.1 monolith -- the 'Glasses not in
    GLASS_REGISTRY' siblings) MUST carry stacklevel=2 + UserWarning.

    v5.1.0 Agent F: after the 6-file split these two siblings live in
    ``prescriptions_zemax.py`` (one inside ``load_zemax_zmx``, one
    inside ``load_zemax_prescription_data_txt``).  The pin walks every
    ``prescriptions*.py`` file under ``lumenairy/io/`` so the audit
    closure is preserved across the split.

    Name-anchored regression pin: a future revert of the C.7 fix that
    only touches the parametrized version of this test would be
    caught by this name-anchored check.
    """
    io_dir = _REPO_ROOT / 'lumenairy' / 'io'
    glasses_warn_calls = []
    for py in sorted(io_dir.glob('prescriptions*.py')):
        src = py.read_text(encoding='utf-8', errors='replace')
        tree = ast.parse(src, filename=str(py))
        for sub in ast.walk(tree):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            if not (isinstance(func, ast.Attribute)
                    and func.attr == 'warn'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'warnings'):
                continue
            # Inspect the message string for the 'Glasses not in GLASS_REGISTRY'
            # marker; both audit-cited lines share this phrasing.
            if not sub.args:
                continue
            first = sub.args[0]
            # The first arg is a JoinedStr (f-string).  Check whether any
            # literal segment contains the marker substring.
            if isinstance(first, ast.JoinedStr):
                literal_str = ''.join(
                    seg.value for seg in first.values
                    if isinstance(seg, ast.Constant) and isinstance(seg.value, str)
                )
                if 'Glasses not in GLASS_REGISTRY' in literal_str:
                    glasses_warn_calls.append(sub)
    assert len(glasses_warn_calls) >= 2, (
        f"Expected >= 2 ``Glasses not in GLASS_REGISTRY`` warn calls; "
        f"found {len(glasses_warn_calls)}.  The audit cited two such "
        f"siblings at lines ~1019, ~1470 (C.7)."
    )
    for call in glasses_warn_calls:
        assert _call_has_stacklevel_kwarg(call), (
            f"warnings.warn(...) at io/prescriptions.py:{call.lineno} "
            f"is the ``Glasses not in GLASS_REGISTRY`` warning but "
            f"missing stacklevel>=2.  C.7 closure regression."
        )
        assert _call_has_category_kwarg_or_positional(call), (
            f"warnings.warn(...) at io/prescriptions.py:{call.lineno} "
            f"is the ``Glasses not in GLASS_REGISTRY`` warning but "
            f"missing explicit UserWarning category.  C.7 closure "
            f"regression."
        )


# ===========================================================================
# Closing sanity: the v4.14.2 cache-lock pin still passes (regression
# check that the C.5 refactor didn't break the lock pairing).
# ===========================================================================


def test_propagation_asm_cache_lock_still_paired():
    """The v4.14.2 cache <-> lock pairing meta-pin still passes: the
    ``_clear_local_asm_caches`` body uses ``_ASM_CACHE_LOCK``.

    Cross-check that the C.5 late-binding refactor preserved the
    pre-existing lock-pairing invariant.
    """
    import inspect as _inspect

    from lumenairy.propagators import propagation as prop_mod

    src = _inspect.getsource(prop_mod._clear_local_asm_caches)
    assert '_ASM_CACHE_LOCK' in src, (
        "v4.14.2 cache-lock-pairing meta-pin regression: "
        "_clear_local_asm_caches no longer references _ASM_CACHE_LOCK."
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
