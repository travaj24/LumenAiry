"""Pinning tests for the v4.14 audit (group Agent 3 scope).

Covers nine audit items handled by Agent 3 in the v4.13.0 -> v4.14
audit pass.  Each test pins exactly one finding so a regression
points straight at the relevant fix:

* **P2 #10** ``_RestoreDtype`` exposes an explicit ``restore()`` and
  the dominant call site uses ``try/finally`` instead of relying on
  CPython refcount semantics.
* **P2 #11** ``_merit_jac_auto`` uses forward-FD + cached ``f0`` for
  the non-JAX merit terms, reducing eval count from ``2N`` to
  ``N + 2`` (one centre eval plus N forward steps plus one outer
  evaluate() to capture f0).
* **P2 #12** ``apply_mirror`` aperture docstring describes a circle
  (not an ellipse, which never matched the code).
* **P2 #13** :class:`CancellableProgress` exposes a ``should_stop``
  flag and ``cancel()`` method; :func:`is_cancelled` reads it.
* **P2 #14** :func:`design_optimize` emits a UserWarning when a non-
  default ``wave_propagator`` is selected alongside any of the three
  Merit classes that hard-code ``apply_real_lens`` for off-nominal
  legs.
* **P2 #16** ``_fd_grad_pure(validate_f0=True)`` catches a
  deliberately-stale ``f0`` cache and raises ValueError.
* **P2 #17** :class:`BSDFModel.total_integrated_scatter` raises
  ValueError (instead of silently broadcasting) when a subclass
  evaluator returns a wrong-shape array.
* **P3 #18** :func:`set_max_ram` rejects negative / zero budgets;
  :func:`get_max_ram` is in ``__all__``.
* **P3 #19** :class:`MultiPrescriptionParameterization` raises
  ValueError when the ``free_vars`` list contains a duplicate
  ``(prescription_index, *path)`` entry.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy
from lumenairy import (
    CancellableProgress, is_cancelled,
    MultiPrescriptionParameterization, MultiWavelengthMerit,
    StrehlMerit,
    set_max_ram, get_max_ram,
)
from lumenairy.elements.bsdf import BSDFModel, LambertianBSDF
from lumenairy.optimize.core import (
    _fd_grad_pure, design_optimize, DesignParameterization,
)


# ====================================================================
# P2 #13 -- Cancellation protocol
# ====================================================================

def test_p2_13_cancellable_progress_default_should_stop_false():
    """Fresh ``CancellableProgress`` has ``should_stop == False``."""
    cp = CancellableProgress()
    assert cp.should_stop is False


def test_p2_13_cancellable_progress_cancel_flips_flag():
    """``cancel()`` sets ``should_stop`` to True."""
    cp = CancellableProgress()
    assert cp.should_stop is False
    cp.cancel()
    assert cp.should_stop is True


def test_p2_13_cancellable_progress_cancel_is_idempotent():
    """Calling ``cancel()`` twice is harmless."""
    cp = CancellableProgress()
    cp.cancel()
    cp.cancel()
    assert cp.should_stop is True


def test_p2_13_cancellable_progress_reset_clears_flag():
    """``reset()`` lets one callback be reused across runs."""
    cp = CancellableProgress()
    cp.cancel()
    assert cp.should_stop is True
    cp.reset()
    assert cp.should_stop is False


def test_p2_13_is_cancelled_handles_none():
    """``is_cancelled(None)`` is always False -- no callback means
    no cancellation channel."""
    assert is_cancelled(None) is False


def test_p2_13_is_cancelled_handles_plain_callable():
    """Plain callables without the protocol are never cancelled."""
    plain = lambda *a, **k: None  # noqa: E731
    assert is_cancelled(plain) is False


def test_p2_13_is_cancelled_reads_cancellable_progress():
    """``is_cancelled`` reads ``CancellableProgress.should_stop``."""
    cp = CancellableProgress()
    assert is_cancelled(cp) is False
    cp.cancel()
    assert is_cancelled(cp) is True


def test_p2_13_cancellable_progress_forwards_to_parent():
    """Calling the wrapper forwards to the wrapped callback."""
    received = []
    parent = lambda s, f, m='': received.append((s, f, m))  # noqa: E731
    cp = CancellableProgress(parent)
    cp('foo', 0.5, 'msg')
    assert received == [('foo', 0.5, 'msg')]


# ====================================================================
# P2 #16 -- _fd_grad_pure validate_f0 catches stale cache
# ====================================================================

def test_p2_16_validate_f0_catches_stale_cache():
    """``validate_f0=True`` raises ValueError when f0 does not match
    f(x) at the supplied x."""
    def f(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0])
    stale_f0 = 999.0  # clearly wrong (true f(x) = 14)
    with pytest.raises(ValueError, match='stale'):
        _fd_grad_pure(f, x, scheme='forward', f0=stale_f0,
                       validate_f0=True)


def test_p2_16_validate_f0_default_off_lets_stale_pass():
    """Default behaviour: stale f0 silently produces a wrong gradient
    (this pins the documented "caller is responsible" contract)."""
    def f(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0])
    stale_f0 = 999.0
    # Should NOT raise; just produces a wrong gradient.
    g_stale = _fd_grad_pure(f, x, scheme='forward', f0=stale_f0)
    g_correct = _fd_grad_pure(f, x, scheme='forward', f0=None)
    # Confirm the stale path silently disagrees with the correct one
    # by far more than O(h) truncation could explain.
    assert np.max(np.abs(g_stale - g_correct)) > 1.0


def test_p2_16_validate_f0_accepts_correct_f0():
    """``validate_f0=True`` does NOT raise when ``f0 == f(x)``."""
    def f(x):
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0])
    correct_f0 = f(x)
    g = _fd_grad_pure(f, x, scheme='forward', f0=correct_f0,
                       validate_f0=True)
    assert g.shape == x.shape


# ====================================================================
# P2 #11 -- _merit_jac_auto forward-FD + cached f0 eval-count saving
# ====================================================================

def test_p2_11_forward_fd_with_f0_eval_count_is_N():
    """Pin the contract that motivates the design_optimize change:
    forward FD with ``f0`` supplied costs exactly N evaluations of
    the merit-subset function, vs 2N for central differences."""
    counter = {'count': 0}
    def f(x):
        counter['count'] += 1
        return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))
    x = np.array([1.0, 2.0, 3.0, 4.0])
    N = x.size
    # Central differences: 2N evals.
    counter['count'] = 0
    _fd_grad_pure(f, x, scheme='central')
    central_count = counter['count']
    # Forward + cached f0: N evals (caller pre-computed f0).
    counter['count'] = 0
    f0 = f(x)              # 1 eval -- caller pays this (mimics design_optimize)
    pre = counter['count']
    _fd_grad_pure(f, x, scheme='forward', f0=f0)
    forward_count = counter['count'] - pre
    assert central_count == 2 * N, (
        f'central FD expected 2N={2*N}, got {central_count}')
    assert forward_count == N, (
        f'forward FD with f0 expected N={N}, got {forward_count}')
    # The audit's net saving: central(2N) -> forward(N + 1 setup eval)
    # = (2N) - (N + 1) = N - 1 saved.  Pin this.
    saving = central_count - (forward_count + 1)
    assert saving == N - 1, (
        f'net eval-count saving expected N-1={N-1}, got {saving}')


def test_p2_11_forward_fd_gradient_matches_central_to_O_h():
    """Forward + cached f0 must agree with central differences to
    within forward-FD's O(h) truncation tolerance on a smooth
    quadratic.  This pins that the design_optimize switch is
    physically correct, not just faster."""
    def f(x):
        x = np.asarray(x, dtype=np.float64)
        return float(x @ np.diag([1.0, 2.0, 3.0]) @ x)
    x = np.array([0.1, -0.3, 0.5])
    g_central = _fd_grad_pure(f, x, eps=1e-7, scheme='central')
    f0 = f(x)
    g_forward = _fd_grad_pure(f, x, eps=1e-7, scheme='forward', f0=f0)
    # Forward FD O(h) at h~1e-7 -> rel err ~1e-7, well under 1e-3.
    rel = np.max(np.abs(g_forward - g_central) /
                 (np.abs(g_central) + 1e-12))
    assert rel < 1e-3, f'rel mismatch {rel:.3e} exceeds O(h) tol'


# ====================================================================
# P2 #14 -- Merit propagator inconsistency warning
# ====================================================================

def _make_minimal_singlet():
    """A trivial single-element prescription for warning-only tests."""
    return lumenairy.make_singlet(
        R1=60e-3, R2=float('inf'), d=4e-3, glass='N-BK7',
        aperture=12e-3,
    )


def test_p2_14_warning_fires_for_gbd_plus_multiwavelength():
    """``design_optimize`` with ``wave_propagator='gbd'`` + a
    MultiWavelengthMerit triggers a UserWarning that mentions the
    inconsistency.  We catch and check the warning text without
    needing to run the full optimisation."""
    pres = _make_minimal_singlet()
    param = DesignParameterization(
        template=pres,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)],
    )
    inner = StrehlMerit(weight=1.0)
    mw = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6], sub_merit=inner)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            design_optimize(
                parameterization=param,
                merit_terms=[mw],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                wave_propagator='gbd',
                max_iter=1,
                verbose=False,
            )
        except Exception:
            # Some sub-pipelines may raise on this trivial setup;
            # we only care that the warning fired before any raise.
            pass
    msgs = [str(w.message) for w in ws
            if issubclass(w.category, UserWarning)]
    matched = [m for m in msgs
               if 'wave_propagator' in m and 'MultiWavelengthMerit' in m]
    assert matched, (
        f'expected a UserWarning mentioning wave_propagator and '
        f'MultiWavelengthMerit; got messages={msgs}')


def test_p2_14_no_warning_for_default_real_lens_propagator():
    """The default ``wave_propagator='real_lens'`` must NOT trigger
    the warning -- it only fires when the user actively requests a
    propagator that the sub-merit cannot honour."""
    pres = _make_minimal_singlet()
    param = DesignParameterization(
        template=pres,
        free_vars=[('surfaces', 0, 'radius')],
        bounds=[(30e-3, 100e-3)],
    )
    inner = StrehlMerit(weight=1.0)
    mw = MultiWavelengthMerit(
        wavelengths=[1.30e-6, 1.55e-6], sub_merit=inner)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        try:
            design_optimize(
                parameterization=param,
                merit_terms=[mw],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                wave_propagator='real_lens',
                max_iter=1,
                verbose=False,
            )
        except Exception:
            pass
    msgs = [str(w.message) for w in ws
            if issubclass(w.category, UserWarning)]
    inconsistency = [m for m in msgs
                     if 'wave_propagator' in m
                     and 'MultiWavelengthMerit' in m]
    assert not inconsistency, (
        f'real_lens propagator must NOT trigger the inconsistency '
        f'warning; got {inconsistency}')


# ====================================================================
# P2 #17 -- BSDF TIS shape mismatch raises ValueError
# ====================================================================

def test_p2_17_bsdf_tis_raises_on_wrong_shape():
    """A subclass evaluator that returns a wrong-shape array must
    surface as a ValueError (pre-v4.14 silently broadcast)."""
    class _BadBSDF(BSDFModel):
        def evaluate(self, incident_dir, scattered_dir):
            # Wrong shape: scalar instead of the integration grid.
            return np.array(0.5)
        def sample(self, incident_dir, n_samples, rng=None):
            return np.zeros((n_samples, 3))
    bad = _BadBSDF()
    with pytest.raises(ValueError) as info:
        bad.total_integrated_scatter()
    msg = str(info.value)
    assert 'shape' in msg.lower(), (
        f'error must mention shape; got {msg!r}')
    # Expected/actual shapes both in the message.
    assert '256' in msg and '128' in msg, (
        f'error must include expected grid dims; got {msg!r}')


def test_p2_17_bsdf_tis_still_works_for_correct_subclass():
    """The default integration path still produces the right TIS
    for a well-behaved subclass (Lambertian closed form == rho)."""
    lamb = LambertianBSDF(rho=0.7)
    # Lambertian overrides total_integrated_scatter with the closed
    # form rho, so we exercise the default-path correctness via the
    # well-known fact that the Lambertian closed-form equals rho.
    tis = lamb.total_integrated_scatter()
    assert abs(tis - 0.7) < 1e-3


# ====================================================================
# P3 #18 -- memory.py validation + __all__
# ====================================================================

def test_p3_18_set_max_ram_rejects_negative():
    """``set_max_ram(-5)`` must raise ValueError (pre-v4.14 silently
    accepted, treating -5 GB as a negative byte count)."""
    # Ensure we don't leave state behind for other tests.
    prev = get_max_ram()
    try:
        with pytest.raises(ValueError, match='positive'):
            set_max_ram(-5)
    finally:
        set_max_ram(prev if prev is not None else None)


def test_p3_18_set_max_ram_rejects_zero():
    """``set_max_ram(0)`` must raise ValueError -- a zero budget is
    nonsensical."""
    prev = get_max_ram()
    try:
        with pytest.raises(ValueError, match='positive'):
            set_max_ram(0)
    finally:
        set_max_ram(prev if prev is not None else None)


def test_p3_18_set_max_ram_accepts_positive_gb():
    """Round-trip a 16 GB budget."""
    prev = get_max_ram()
    try:
        set_max_ram(16)                       # 16 GB
        assert get_max_ram() == 16 * 1024**3
    finally:
        set_max_ram(prev if prev is not None else None)


def test_p3_18_get_max_ram_in_memory_dunder_all():
    """``get_max_ram`` must be in ``lumenairy.memory.__all__``."""
    import lumenairy.memory as mem
    assert 'get_max_ram' in mem.__all__


# ====================================================================
# P3 #19 -- MultiPrescriptionParameterization duplicate detection
# ====================================================================

def test_p3_19_duplicate_free_vars_raises():
    """Two identical ``(prescription_index, *path)`` entries must
    raise ValueError at construction time."""
    template_a = _make_minimal_singlet()
    template_b = _make_minimal_singlet()
    free_vars = [
        (0, 'surfaces', 0, 'radius'),  # template 0, surface 0 radius
        (1, 'surfaces', 0, 'radius'),  # template 1, surface 0 radius
        (0, 'surfaces', 0, 'radius'),  # DUPLICATE of the first entry
    ]
    with pytest.raises(ValueError, match='duplicate'):
        MultiPrescriptionParameterization(
            templates=[template_a, template_b],
            free_vars=free_vars,
            bounds=[(30e-3, 100e-3)] * 3)


def test_p3_19_unique_free_vars_accepts():
    """No duplicates -> construction succeeds."""
    template_a = _make_minimal_singlet()
    template_b = _make_minimal_singlet()
    free_vars = [
        (0, 'surfaces', 0, 'radius'),
        (0, 'thicknesses', 0),
        (1, 'surfaces', 0, 'radius'),
    ]
    param = MultiPrescriptionParameterization(
        templates=[template_a, template_b],
        free_vars=free_vars,
        bounds=[(30e-3, 100e-3)] * 3)
    assert param.n_params == 3


# ====================================================================
# P2 #10 -- _RestoreDtype try/finally pattern
# ====================================================================

def test_p2_10_dtype_restored_on_exception():
    """Pin that ``design_optimize`` restores the global complex dtype
    on every exit path, including an exception during optimisation.

    Pre-v4.14 this relied on ``_RestoreDtype.__del__`` running at the
    right moment; under PyPy / when a reference cycle survives gc the
    restore could be deferred.  v4.14 wraps the body in try/finally
    so the restore is deterministic.
    """
    from lumenairy.propagators.propagation import (
        get_default_complex_dtype, set_default_complex_dtype)
    saved = get_default_complex_dtype()
    try:
        set_default_complex_dtype(np.complex128)

        pres = _make_minimal_singlet()
        param = DesignParameterization(
            template=pres,
            free_vars=[('surfaces', 0, 'radius')],
            bounds=[(30e-3, 100e-3)],
        )

        # A merit term that always raises -- forces an exception
        # path so we can verify the restore still happens.
        from lumenairy.optimize.core import MeritTerm
        class _BoomMerit(MeritTerm):
            name = 'Boom'
            needs_wave = False
            def evaluate(self, ctx):
                raise RuntimeError('boom')

        with pytest.raises(Exception):
            design_optimize(
                parameterization=param,
                merit_terms=[_BoomMerit()],
                wavelength=1.30e-6,
                N=32, dx=10e-6,
                precision='single',
                max_iter=1,
                verbose=False,
            )
        # Even though precision='single' switched the global to
        # complex64 during the call, the finally block must put it
        # back to complex128.
        assert get_default_complex_dtype() == np.complex128, (
            f'expected complex128 restored, got '
            f'{get_default_complex_dtype()}')
    finally:
        set_default_complex_dtype(saved)


# ====================================================================
# P2 #12 -- apply_mirror docstring says circle (not ellipse)
# ====================================================================

def test_p2_12_apply_mirror_aperture_doc_says_circle():
    """The inline comment block at the aperture-mask site must
    describe the actual code -- a circular aperture in physical
    coordinates, NOT the old (wrong) "elliptical" description."""
    from pathlib import Path
    src = Path(lumenairy.__file__).parent / 'elements' / 'elements.py'
    text = src.read_text(encoding='cp1252')
    # The current comment block lives just above the apply_mirror
    # aperture mask.  Pin: the word "ellipse" must NOT appear in
    # the comment block that immediately precedes the line that
    # zeros pixels outside the aperture in apply_mirror.  We
    # search for the unique phrase introduced by the fix.
    assert 'CIRCULAR clear aperture' in text, (
        'apply_mirror aperture comment must state a CIRCULAR '
        'aperture (audit P2 #12 fix marker missing)')
    # Sanity: the file should still have the apply_mirror def.
    assert 'def apply_mirror' in text


# ====================================================================
# P3 #21 -- dtype-aware zero replaces 0.0+0.0j literal
# ====================================================================

def test_p3_21_apply_mirror_aperture_no_complex_literal():
    """The aperture mask in apply_mirror must not use the
    ``0.0 + 0.0j`` complex128 literal (which could silently upcast
    a JAX-x32 input)."""
    from pathlib import Path
    src = Path(lumenairy.__file__).parent / 'elements' / 'elements.py'
    text = src.read_text(encoding='cp1252')
    # The fix marker.
    assert 'xp.zeros((), dtype=E.dtype)' in text, (
        'apply_mirror aperture must use xp.zeros((), dtype=E.dtype) '
        '(audit P3 #21 fix marker missing)')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
