"""w4 / P5 -- ``propagate()`` return contract: the F1 flip, EXECUTED (v5.30).

Closes the deferred F1 decision.  The roadmap costed four options; the owner
chose **option 4** -- "make the stable ``PropagationResult`` the default,
keeping ``return_result=False`` available" -- and then chose to EXECUTE the
flip in the same release that announced it, rather than ship a warning about a
change no caller could yet see (same call as the W5 shim-removal wave).

As shipped:

* ``propagate(...)`` with ``return_result`` unset returns a
  ``PropagationResult`` for **every** method.  One contract; the return shape
  no longer depends on ``z`` (the P5 finding).
* ``return_result=False`` returns the kernels' native shapes -- bare ndarray
  (asm / rs / maslov / gbd / hfpi / hf) or ``(E, dx_out, dy_out)`` (sas /
  fresnel / fraunhofer) -- bit-for-bit as before the flip.  Permanent,
  un-deprecated escape hatch.
* ``return_result=True`` is unchanged.
* The transition ``DeprecationWarning`` and its ``_caller_is_internal``
  external-caller predicate are RETIRED with the flip; the registry's
  ``API_TRANSITION_VERSION`` entry is completed (tombstoned) per the same
  convention the W5 wave used for ``REMOVAL_SCHEDULE``.
* The W3 grid-change ``UserWarning`` survives, but only on the
  ``return_result=False`` path -- the stable contract has no shape
  instability to report.
* P16 is unchanged: ``PropagationResult.__iter__`` stays 2-item, permanently.

This module SUPERSEDES the announcement-phase pins of ``3097cda``
(``TestWarnsOnTheDefaultUnstablePath``, ``TestBothExplicitModesAreSilent``,
``TestReturnShapesAreBitUnchanged``, ``TestRegistryScheduling``,
``TestTransitionIsDocumented``, ``TestInternalCallersAreExempt``), which
asserted that the sentinel default warns and returns the legacy contract.
Each superseding class names its predecessor.

Bit-identity is asserted IN-RUN against the kernels called directly (and
between the wrapped and unwrapped paths) rather than against frozen digests,
so the pins carry no platform-dependent float literals.  Out-of-band, the flip
was measured over a 55-record pre/post capture (every ``return_result=False``
and ``return_result=True`` record byte-identical -- 41/41 including the mhs,
algebra and sources call sites; the only 14 changed records are the 14 default
-path ones, each equal to that case's pre-flip ``return_result=True`` record).

Pre-fix (worktree at 3a1da2b) every class here except ``TestP16...`` fails:
the default returned the legacy contract, the DeprecationWarning machinery was
still present, and the internal call sites did not name a contract.

Author: audit wave 4 (P5 / roadmap F1 closure), wave 5 (the flip)
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy import _deprecation as dep
from lumenairy.propagators import dispatch
from lumenairy.propagators.dispatch import propagate
from lumenairy.propagators.result import PropagationResult

LAM = 633e-9
DX = 2e-6
N = 64

# Selector thresholds at N=64, dx=2 um, lambda=633 nm (same basis as the W3
# P5 pins): Q = lambda*z/(N*dx^2) crosses 1 at z = 4.04e-4 m and
# N_F = (N*dx/2)^2/(lambda*z) crosses 0.1 at z = 6.47e-2 m.
Z_SAME_GRID = 1e-4      # -> asm, natively a bare ndarray at the input pitch
Z_GRID_CHANGING = 1e-3  # -> sas, natively a 3-tuple at a kernel-chosen pitch
Z_FAR = 5.0             # -> fraunhofer, natively a 3-tuple


def _gauss():
    y, x = np.mgrid[0:N, 0:N]
    xc = (x - N / 2) * DX
    yc = (y - N / 2) * DX
    return np.exp(-(xc ** 2 + yc ** 2) / (20e-6) ** 2).astype(np.complex128)


def _catch(**kw):
    """Call ``propagate`` recording every warning; return (out, records)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        out = propagate(_gauss(), wavelength=LAM, dx=DX, **kw)
    return out, rec


def _quiet(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return propagate(_gauss(), wavelength=LAM, dx=DX, **kw)


def _deprecations(rec):
    return [r for r in rec if issubclass(r.category, DeprecationWarning)]


def _user_warnings(rec):
    return [r for r in rec if issubclass(r.category, UserWarning)
            and "method='auto'" in str(r.message)]


def _squash(text):
    return ' '.join((text or '').split())


def _same_bytes(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())


def _code_only(obj):
    """Source of ``obj`` with docstrings and comments removed.

    Lets a pin assert on what the CODE does without tripping over prose that
    legitimately names a retired mechanism.
    """
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if not isinstance(body, list) or not body:
            continue
        head = body[0]
        if (isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant)
                and isinstance(head.value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _dispatcher_calls(fn):
    """Every ``propagate(...)`` call inside ``fn``, as ``{lineno: kwargs}``."""
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'propagate'):
            out[node.lineno] = {k.arg: ast.unparse(k.value)
                                for k in node.keywords if k.arg}
    return out


# ======================================================================
# The default IS the stable contract
# ======================================================================

class TestDefaultIsTheStableContract:
    """SUPERSEDES ``TestWarnsOnTheDefaultUnstablePath`` (3097cda, the
    announcement phase), which pinned that the sentinel default returned the
    unstable legacy contract and emitted a ``DeprecationWarning`` about it.

    Option 4's whole point is that the caller who writes nothing gets ONE
    contract for every method, so the pins invert: same-grid (``asm``,
    natively a bare ndarray) and grid-changing (``sas`` / ``fraunhofer``,
    natively a 3-tuple at a different pitch) now come back as the same type,
    and nothing is deprecated because nothing needs migrating away from."""

    @pytest.mark.parametrize('z,expect_method', [
        (Z_SAME_GRID, 'asm'),
        (Z_GRID_CHANGING, 'sas'),
        (Z_FAR, 'fraunhofer'),
    ])
    def test_default_returns_a_result_for_every_auto_selection(
            self, z, expect_method):
        assert dispatch._auto_select_method(
            _gauss(), z=z, wavelength=LAM, dx=DX,
            prescription=None) == expect_method
        out, rec = _catch(z=z)
        assert isinstance(out, PropagationResult)
        assert out.method == expect_method
        assert not rec, [str(r.message)[:70] for r in rec]

    def test_the_shape_stable_attributes_are_defined_for_every_method(self):
        """``.field`` / ``.dx`` / ``.dy`` mean the same thing whichever kernel
        ran -- which is what makes the contract independent of z."""
        for z in (Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR):
            res = _quiet(z=z)
            assert isinstance(res.field, np.ndarray)
            assert res.field.shape == (N, N)
            assert float(res.dx) > 0.0 and float(res.dy) > 0.0
            assert res.dx_out == res.dx and res.dy_out == res.dy
            assert _same_bytes(np.asarray(res), res.field)

    def test_default_reports_the_kernel_chosen_output_pitch(self):
        """Independent oracle: for a grid-changing selection the wrapped
        ``.dx`` / ``.dy`` must be the kernel's OWN dx_out / dy_out, not the
        input pitch -- otherwise the stable contract would be stably wrong."""
        for z in (Z_GRID_CHANGING, Z_FAR):
            native = _quiet(z=z, return_result=False)
            res = _quiet(z=z)
            assert isinstance(native, tuple) and len(native) == 3
            assert float(res.dx) == float(native[1])
            assert float(res.dy) == float(native[2])
            assert float(res.dx) != DX, 'this z must re-grid, else no test'
        same = _quiet(z=Z_SAME_GRID)
        assert float(same.dx) == DX and float(same.dy) == DX

    def test_default_wraps_named_methods_too(self):
        """The flip is per-method-agnostic: naming ``sas`` does not opt back
        into its native 3-tuple."""
        out, rec = _catch(z=Z_GRID_CHANGING, method='sas')
        assert isinstance(out, PropagationResult)
        assert out.method == 'sas'
        assert not rec, [str(r.message)[:70] for r in rec]

    def test_default_wraps_the_z_none_passthrough(self):
        out, rec = _catch(z=None)
        assert isinstance(out, PropagationResult)
        assert not rec

    def test_no_deprecationwarning_anywhere_on_the_default_path(self):
        """The transition warning is retired: relying on the default is the
        recommended spelling now, not a deprecated one."""
        for kw in ({'z': Z_SAME_GRID}, {'z': Z_GRID_CHANGING},
                   {'z': Z_FAR}, {'z': None},
                   {'z': Z_GRID_CHANGING, 'method': 'sas'}):
            _out, rec = _catch(**kw)
            assert not _deprecations(rec), (
                kw, [str(r.message)[:70] for r in rec])

    def test_no_grid_change_userwarning_on_the_default_path(self):
        """The W3 diagnostic is about an unstable return SHAPE.  The default
        return has none, so it must not fire there (it stays on the
        ``return_result=False`` path -- pinned in the next class)."""
        for z in (Z_GRID_CHANGING, Z_FAR):
            _out, rec = _catch(z=z)
            assert not _user_warnings(rec), [str(r.message)[:70] for r in rec]

    def test_signature_default_is_still_the_sentinel(self):
        """The sentinel stays: "did not choose" remains distinguishable from
        "chose the wrapper", and the routing resolves it explicitly rather
        than relying on its truthiness (it is FALSY)."""
        sig = inspect.signature(propagate)
        default = sig.parameters['return_result'].default
        assert default is dep._NO_DEFAULT
        assert not default, 'the sentinel is falsy -- routing must not test it'
        src = inspect.getsource(propagate)
        assert 'wrap = True if return_result is _NO_DEFAULT' in src
        assert 'if not wrap:' in src
        assert 'if not return_result:' not in src, (
            'routing on the falsy sentinel would silently un-flip the default')


# ======================================================================
# The legacy contract is a permanent escape hatch
# ======================================================================

class TestLegacyContractIsAPermanentEscapeHatch:
    """SUPERSEDES ``TestBothExplicitModesAreSilent`` (3097cda), which pinned
    that both explicit values silenced the transition warning.  There is no
    transition warning any more; what needs pinning is that ``False`` still
    delivers the pre-flip shapes, is not deprecated, and keeps the one
    diagnostic that IS about those shapes."""

    def test_false_returns_the_native_bare_ndarray(self):
        out, rec = _catch(z=Z_SAME_GRID, return_result=False)
        assert isinstance(out, np.ndarray)
        assert not _deprecations(rec)

    @pytest.mark.parametrize('z', [Z_GRID_CHANGING, Z_FAR])
    def test_false_returns_the_native_triple(self, z):
        out = _quiet(z=z, return_result=False)
        assert isinstance(out, tuple) and len(out) == 3
        E, dxo, dyo = out           # the 3-item unpacking contract
        assert isinstance(E, np.ndarray)
        assert float(dxo) > 0.0 and float(dyo) > 0.0

    def test_false_is_not_deprecated(self):
        """Counter-pin: the escape hatch must not acquire a warning by
        accident.  Nothing in the registry schedules anything against it."""
        for z in (Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR):
            _out, rec = _catch(z=z, return_result=False)
            assert not _deprecations(rec), [str(r.message)[:70] for r in rec]
        assert 'return_result' not in str(dep.REMOVAL_SCHEDULE)

    def test_false_keeps_the_grid_change_userwarning(self):
        """The W3 UserWarning's sole remaining home: an ``auto`` selection
        that re-grids, delivered un-wrapped because the caller asked."""
        _out, rec = _catch(z=Z_GRID_CHANGING, return_result=False)
        hits = _user_warnings(rec)
        assert hits, 'the P5 UserWarning must survive on the legacy path'
        msg = str(hits[0].message)
        assert 'return_result=False' in msg, (
            'the message must name what the caller actually passed')
        assert 'return_result=True' in msg

    def test_false_userwarning_stays_silent_where_it_always_was(self):
        """Unchanged gating: same-grid selections and explicitly named
        kernels are silent even on the legacy path."""
        for kw in ({'z': Z_SAME_GRID}, {'z': Z_GRID_CHANGING,
                                        'method': 'sas'}):
            _out, rec = _catch(return_result=False, **kw)
            assert not _user_warnings(rec), kw

    def test_the_three_item_unpack_needs_the_escape_hatch(self):
        """P16's migration statement, made executable: ``E, dxo, dyo`` works
        on the legacy contract and raises on the wrapper (2-item iteration),
        which is why ``return_result=False`` is the documented answer."""
        E, dxo, dyo = _quiet(z=Z_GRID_CHANGING, method='sas',
                             return_result=False)
        assert isinstance(E, np.ndarray) and float(dxo) == float(dyo)
        with pytest.raises(ValueError):
            _a, _b, _c = _quiet(z=Z_GRID_CHANGING, method='sas')

    def test_false_is_reachable_through_any_falsy_value(self):
        """Pre-flip callers who passed ``0`` / ``None`` explicitly kept the
        legacy contract by truthiness; that must not change."""
        for value in (False, 0, None, ''):
            out = _quiet(z=Z_SAME_GRID, return_result=value)
            assert isinstance(out, np.ndarray), value


# ======================================================================
# return_result=True is untouched
# ======================================================================

class TestTrueIsUnchanged:

    @pytest.mark.parametrize('z', [Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR])
    def test_true_returns_a_result_and_warns_about_nothing(self, z):
        out, rec = _catch(z=z, return_result=True)
        assert isinstance(out, PropagationResult)
        assert not rec, [str(r.message)[:70] for r in rec]

    @pytest.mark.parametrize('z', [Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR])
    def test_true_and_the_default_are_the_same_result(self, z):
        """The flip's definition: the default now IS ``return_result=True``.
        Compared field-bytes-and-metadata, in-run."""
        a = _quiet(z=z, return_result=True)
        b = _quiet(z=z)
        assert type(a) is type(b) is PropagationResult
        assert _same_bytes(a.field, b.field)
        assert float(a.dx) == float(b.dx) and float(a.dy) == float(b.dy)
        assert a.method == b.method
        assert (a.z is None) == (b.z is None)
        assert sorted(a.metadata) == sorted(b.metadata)


# ======================================================================
# Bit-identity across the flip
# ======================================================================

class TestBitIdentityAcrossTheFlip:
    """SUPERSEDES ``TestReturnShapesAreBitUnchanged`` (3097cda), which pinned
    the default against explicit ``False``.  Post-flip the default is the
    wrapper, so the invariant moves: the legacy path must still be exactly
    the kernels' own output (that IS the pre-flip default, byte-for-byte),
    and the wrapper must carry those same bytes.

    Every comparison runs in-process against an independent oracle -- no
    frozen digests, so nothing here is platform-dependent."""

    def test_false_matches_the_kernels_called_directly(self):
        from lumenairy.propagators.propagation import (
            angular_spectrum_propagate,
            scalable_angular_spectrum_propagate,
        )
        E = _gauss()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            asm_out = propagate(E, wavelength=LAM, dx=DX, z=Z_SAME_GRID,
                                return_result=False)
            sas_out = propagate(E, wavelength=LAM, dx=DX, z=Z_GRID_CHANGING,
                                return_result=False)
        assert _same_bytes(
            asm_out, angular_spectrum_propagate(E, Z_SAME_GRID, LAM, DX))
        oracle = scalable_angular_spectrum_propagate(
            E, Z_GRID_CHANGING, LAM, DX)
        assert _same_bytes(sas_out[0], oracle[0])
        assert float(sas_out[1]) == float(oracle[1])
        assert float(sas_out[2]) == float(oracle[2])

    @pytest.mark.parametrize('z', [Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR])
    def test_the_wrapper_carries_the_legacy_bytes_unchanged(self, z):
        """The flip changed the container, not the content."""
        native = _quiet(z=z, return_result=False)
        field = native if isinstance(native, np.ndarray) else native[0]
        assert _same_bytes(_quiet(z=z).field, field)

    def test_prescription_methods_are_flipped_too_and_keep_their_bytes(self):
        """Not just the free-space kernels: a prescription-driven maslov call
        wraps as well, with the same field bytes."""
        pres = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                               glass='N-BK7', aperture=0.3e-3)
        n, dx = 48, 8e-6
        x = (np.arange(n) - n / 2) * dx
        xx, yy = np.meshgrid(x, x, indexing='xy')
        E = np.exp(-(xx ** 2 + yy ** 2) / (100e-6 ** 2)).astype(np.complex128)
        kw = dict(wavelength=LAM, dx=dx, prescription=pres, method='maslov',
                  ray_field_samples=6, ray_pupil_samples=6, n_v2=8)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            native = propagate(E, return_result=False, **kw)
            res = propagate(E, **kw)
        assert isinstance(native, np.ndarray)
        assert isinstance(res, PropagationResult)
        assert res.method == 'maslov'
        assert _same_bytes(res.field, native)


# ======================================================================
# The announcement machinery is retired, the registry entry completed
# ======================================================================

class TestTransitionMachineryIsRetired:
    """SUPERSEDES ``TestRegistryScheduling`` and
    ``TestInternalCallersAreExempt`` (3097cda), which pinned the scheduled
    horizon, the warning text, and the internal-caller exemption.

    A warning reading "the default WILL become a PropagationResult in vX"
    cannot outlive the version that makes it one, and an exemption predicate
    with no warning to suppress is dead code.  Both are gone; what must
    survive is the registry mechanism itself (for the NEXT transition) and a
    written record that this one was executed rather than dropped."""

    def test_the_warning_builder_and_its_helpers_are_gone(self):
        for name in ('_p5_transition_message', '_caller_is_internal',
                     '_P5_UNSTABLE_RETURN_TYPES', '_P5_DEPRECATED_SINCE'):
            assert not hasattr(dispatch, name), (
                f'{name} is announcement-phase machinery; it must be retired '
                f'with the flip')

    def test_dispatch_no_longer_imports_the_horizon(self):
        """The module must not resolve a transition version it has passed."""
        assert not hasattr(dispatch, 'API_TRANSITION_VERSION')
        assert not hasattr(dispatch, 'resolve_removal_version')
        assert 'DeprecationWarning' not in _code_only(propagate), (
            'no DeprecationWarning may be emitted from propagate() -- the '
            'docstring may still explain the retirement, the CODE may not '
            'raise it')

    def test_the_registry_slot_survives_and_schedules_nothing(self):
        """The mechanism is kept for the next transition; the executed entry
        is tombstoned in prose, exactly as REMOVAL_SCHEDULE's entries were."""
        assert hasattr(dep, 'API_TRANSITION_VERSION')
        assert 'API_TRANSITION_VERSION' in dep.__all__
        assert dep.API_TRANSITION_VERSION == dep.NEXT_REMOVAL_VERSION, (
            'the slot must stay bound to NEXT_REMOVAL_VERSION so '
            "check_removal_schedule()'s future-version invariant covers it")
        assert dep._version_tuple(dep.API_TRANSITION_VERSION) > \
            dep._version_tuple(la.__version__)
        assert dep.check_removal_schedule() == []

    def test_the_executed_entry_is_tombstoned_in_the_registry(self):
        src = _squash(inspect.getsource(dep))
        assert 'Tombstone, v5.30' in src
        assert 'schedules nothing' in src
        assert 'EXECUTED' in src
        assert 'PropagationResult' in src, (
            'the completed entry must still name what was flipped')
        assert 'return_result=False' in src, (
            'and the escape hatch that replaced the deprecation')

    def test_no_module_still_emits_the_transition_warning(self):
        """Whole-library counter-pin: the retired warning must not survive
        anywhere, for internal or external callers."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            propagate(_gauss(), wavelength=LAM, dx=DX, z=Z_SAME_GRID)
            propagate(_gauss(), wavelength=LAM, dx=DX, z=Z_GRID_CHANGING,
                      return_result=False)
        assert not [r for r in rec
                    if 'relying on the default' in str(r.message)]

    def test_this_module_is_an_external_caller(self):
        """The old exemption predicate classified callers by module name; the
        pin that mattered was that external callers were NOT silenced.  With
        the warning gone the meaningful residue is that the flip applies to
        external callers uniformly -- which this module demonstrates."""
        assert __name__ != 'lumenairy'
        assert not __name__.startswith('lumenairy.')
        assert isinstance(_quiet(z=Z_SAME_GRID), PropagationResult)


# ======================================================================
# The internal call sites named a contract (flip-day migration)
# ======================================================================

class TestInternalCallSitesAreMigrated:
    """The roadmap's F1 inventory listed three internal default-path callers.
    Measured at flip time: ``mhs.prescription_subdomain``'s two sites hand the
    dispatcher return straight to the MHS pipeline as a FIELD, and
    ``algebra.primitives.FreeSpace._apply`` was tolerant of the wrapper for
    the field but NOT for the anamorphic y-pitch (the wrapper reports
    ``dy == dx`` for a pitch-preserving kernel, by the documented v4.13.0 /
    DS-1 convention that ``Source.propagate`` works around explicitly).  All
    three now name ``return_result=False``; ``sources.core.Source.propagate``
    already named ``True``."""

    def test_mhs_subdomains_still_return_bare_fields(self):
        from lumenairy.propagators import mhs as mhs_mod
        HuygensSurface = mhs_mod.HuygensSurface
        prescription_subdomain = mhs_mod.prescription_subdomain
        pres = la.make_singlet(R1=50e-3, R2=-50e-3, d=3e-3,
                               glass='N-BK7', aperture=0.3e-3)

        def _field(n, dx):
            x = (np.arange(n) - n / 2) * dx
            xx, yy = np.meshgrid(x, x, indexing='xy')
            return np.exp(-(xx ** 2 + yy ** 2)
                          / (100e-6 ** 2)).astype(np.complex128)

        cases = {
            'maslov_same_grid': (48, 8e-6, 48, 8e-6, 'maslov'),
            'maslov_resampled': (64, 4e-6, 128, 2e-6, 'maslov'),
            'gbd_output_grid': (48, 8e-6, 48, 8e-6, 'gbd'),
        }
        for label, (n_in, dxi, n_out, dxo, method) in cases.items():
            in_s = HuygensSurface(z=0.0, dx=dxi, Ny=n_in, Nx=n_in)
            out_s = HuygensSurface(z=0.0, dx=dxo, Ny=n_out, Nx=n_out)
            kw = (dict(ray_field_samples=6, ray_pupil_samples=6, n_v2=8)
                  if method == 'maslov' else {})
            sub = prescription_subdomain(in_s, out_s, pres,
                                         wavelength=LAM, method=method, **kw)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = sub.propagator(_field(n_in, dxi), in_s, out_s,
                                     **sub.kwargs)
            assert isinstance(out, np.ndarray), (label, type(out).__name__)
            assert out.shape == (n_out, n_out), label

    def test_freespace_operator_preserves_the_anamorphic_pitch(self):
        """The measured regression the migration prevents: on the wrapped
        path this returned ``dy_out == dx`` for an anamorphic input, silently
        squaring an algebra chain's output pitch."""
        from lumenairy.algebra.primitives import FreeSpace
        dy = 3e-6
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E, dxo, dyo = FreeSpace(Z_GRID_CHANGING)._apply(
                _gauss(), dx=DX, dy=dy, wavelength=LAM)
        assert isinstance(E, np.ndarray)
        assert float(dxo) == DX
        assert float(dyo) == dy, (
            'FreeSpace must keep the caller y-pitch; the wrapped path reports '
            'dy == dx for pitch-preserving kernels')

    def test_freespace_grid_changing_reports_the_kernel_pitch(self):
        """Counter-pin to the above: a genuinely re-gridding selection must
        still report the KERNEL's output pitch, not the input one."""
        from lumenairy.algebra.primitives import FreeSpace
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _E, dxo, _dyo = FreeSpace(Z_GRID_CHANGING)._apply(
                _gauss(), dx=DX, dy=DX, wavelength=LAM)
            native = propagate(_gauss(), z=Z_GRID_CHANGING, wavelength=LAM,
                               dx=DX, return_result=False)
        assert float(dxo) == float(native[1]) != DX

    def test_source_propagate_still_returns_a_source(self):
        from lumenairy.sources.core import Source
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = Source(E=_gauss(), dx=DX, wavelength=LAM).propagate(
                z=Z_GRID_CHANGING)
        assert isinstance(out, Source)
        assert not _deprecations(rec)

    def test_every_internal_call_site_names_its_contract(self):
        """Source-level guard: a future edit must not put these back on the
        default path, where mhs would hand a PropagationResult to the MHS
        pipeline as a field."""
        from lumenairy.algebra import primitives
        from lumenairy.propagators import mhs
        from lumenairy.sources.core import Source
        mhs_calls = _dispatcher_calls(mhs.prescription_subdomain)
        assert len(mhs_calls) == 2, mhs_calls
        assert all(kw.get('return_result') == 'False'
                   for kw in mhs_calls.values()), mhs_calls
        fs_calls = _dispatcher_calls(primitives.FreeSpace._apply)
        assert len(fs_calls) == 1, fs_calls
        assert all(kw.get('return_result') == 'False'
                   for kw in fs_calls.values()), fs_calls
        src_calls = _dispatcher_calls(Source.propagate)
        assert len(src_calls) == 1, src_calls
        assert all(kw.get('return_result') == 'True'
                   for kw in src_calls.values()), src_calls

    def test_no_internal_default_path_callers_remain(self):
        """AST sweep of the shipped package: every dispatcher-``propagate``
        call inside ``lumenairy`` names ``return_result``.  A new internal
        default-path caller is not wrong per se, but it must be a decision --
        this pin makes it a visible one."""
        import ast
        import pathlib
        # ``asymptotic_aberration_tensor._measure_image_plane_waist`` takes a
        # ``propagate`` CALLBACK parameter, not the dispatcher; a name-based
        # sweep cannot tell them apart, so the file is excluded HERE and the
        # exclusion is justified structurally by the next assertion (rather
        # than by a line number that would rot on any edit to that file).
        callback_only = {'asymptotic_aberration_tensor.py'}
        root = pathlib.Path(inspect.getfile(la)).parent
        offenders = []
        for path in root.rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8',
                                            errors='replace'))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                is_prop = (
                    (isinstance(fn, ast.Name) and fn.id == 'propagate')
                    or (isinstance(fn, ast.Attribute) and fn.attr ==
                        'propagate' and isinstance(fn.value, ast.Name)
                        and fn.value.id in ('la', 'lumenairy', 'dispatch')))
                if not is_prop or path.name in callback_only:
                    continue
                if not any(k.arg == 'return_result' for k in node.keywords):
                    offenders.append(f'{path.name}:{node.lineno}')
        assert offenders == [], offenders

    def test_the_excluded_file_really_takes_a_propagate_callback(self):
        """Justifies the exclusion above: that module never imports the
        dispatcher -- its ``propagate`` is a parameter the caller supplies."""
        from lumenairy.propagators import asymptotic_aberration_tensor as aat
        fn = aat._measure_image_plane_waist
        assert 'propagate' in inspect.signature(fn).parameters
        assert 'propagate' not in getattr(aat, '__dict__', {})
        assert not hasattr(aat, 'propagate')


# ======================================================================
# P16 -- unchanged by the flip, as decided
# ======================================================================

class TestP16IterationStaysTwoItem:

    def test_iteration_is_still_exactly_two_items(self):
        res = PropagationResult(field=_gauss(), dx=DX, wavelength=LAM)
        assert len(list(iter(res))) == 2
        field, intermediates = res
        assert np.array_equal(field, res.field)
        assert intermediates == []
        with pytest.raises(ValueError):
            _a, _b, _c = res

    def test_the_flip_did_not_re_arity_iteration(self):
        """The collision P16 flagged is resolved by WHICH OBJECT you ask for,
        not by arity: the wrapper yields 2, the legacy return yields 3."""
        wrapped = _quiet(z=Z_GRID_CHANGING, method='sas')
        native = _quiet(z=Z_GRID_CHANGING, method='sas', return_result=False)
        assert len(list(iter(wrapped))) == 2
        assert len(native) == 3

    def test_no_registry_entry_schedules_an_iteration_change(self):
        assert '__iter__' not in str(dep.REMOVAL_SCHEDULE)
        src = _squash(inspect.getsource(dep))
        assert 'NOT scheduled here' in src
        assert '__iter__' in src, (
            'the registry entry must name the P16 decision explicitly')
        assert 'stays **2-item**' in src or 'stays 2-item' in src

    def test_the_decision_is_recorded_on_the_class(self):
        doc = _squash(PropagationResult.__doc__)
        assert 'P16 resolved' in doc
        assert 'stays 2-item' in doc or 'iteration stays 2-item' in doc
        assert 'NOT scheduled' in doc
        assert 'return_result=False' in doc, (
            'the class docstring must name the migration path for 3-tuple '
            'unpackers')

    def test_the_decision_is_recorded_on_iter(self):
        doc = _squash(PropagationResult.__iter__.__doc__)
        assert 'NOT scheduled to change' in doc
        assert 'return_result=False' in doc


# ======================================================================
# Documentation of the flip as shipped
# ======================================================================

class TestFlipIsDocumented:
    """SUPERSEDES ``TestTransitionIsDocumented`` (3097cda), which required the
    docstring to advertise a SCHEDULED transition and the version constant it
    was scheduled against.  Advertising a future change that has happened is
    the rot the registry exists to prevent, so the requirement inverts."""

    def test_docstring_states_the_settled_contract(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'Return contract, settled in v5.30' in doc
        assert 'EXECUTED' in doc
        assert 'option 4' in doc
        assert 'roadmap_deferred_2026_07_21.md' in doc
        assert 'return_result=True' in doc and 'return_result=False' in doc
        assert 'permanent, supported escape hatch' in doc
        # the P5 table and the deferred-decision record must survive
        assert "method='auto'`` return contract" in inspect.getdoc(propagate)
        assert 'deferred' in doc.lower()

    def test_docstring_no_longer_advertises_a_scheduled_flip(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'Scheduled API transition' not in doc
        assert 'API_TRANSITION_VERSION' not in doc
        assert 'Nothing has changed yet' not in doc
        assert 'is retired with the flip' in doc, (
            'the retirement must be stated, not silently dropped')

    def test_docstring_records_the_versionchanged(self):
        doc = inspect.getdoc(propagate)
        assert 'versionchanged:: 5.30' in doc
        assert 'restores the pre-v5.30 shapes' in _squash(doc)

    def test_warns_section_only_documents_the_userwarning(self):
        doc = _squash(inspect.getdoc(propagate))
        warns = doc.split('Warns -----')[1]
        assert 'UserWarning' in warns
        assert 'DeprecationWarning' not in warns, (
            'nothing about propagate() is deprecated any more')
        assert 'return_result=False' in warns

    def test_p16_is_still_documented_on_propagate(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'P16' in doc
        assert '2-item' in doc

    def test_result_module_documents_the_dispatcher_exception(self):
        from lumenairy.propagators import result as result_mod
        doc = _squash(result_mod.__doc__)
        assert 'BY DEFAULT' in doc
        assert 'return_result=False' in doc
