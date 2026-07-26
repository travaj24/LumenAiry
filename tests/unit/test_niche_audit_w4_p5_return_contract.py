"""w4 / P5 -- ``propagate()`` return-contract transition (roadmap Part F1).

Closes the deferred F1 decision.  The roadmap costed four options; the owner
chose **option 4** -- "make ``return_result=True`` the default over a
deprecation cycle, keeping ``return_result=False`` available", the one the
roadmap costs as least abrupt (options 2 and 3 break live call sites in the
release that ships them; option 4 breaks nobody now).

What v5.30 lands is therefore an ANNOUNCEMENT, not a behaviour change:

* ``propagate()``'s ``return_result`` default becomes the ``_NO_DEFAULT``
  sentinel, so "did not choose" is distinguishable from "chose False".  The
  sentinel is falsy, so every routing test inside ``propagate`` behaves exactly
  as ``False`` did -- pinned here by BYTE-identity of the returns.
* Relying on that default while the call returns the unstable legacy contract
  (bare ndarray **or** ``(E, dx_out, dy_out)`` triple) emits a
  ``DeprecationWarning`` whose horizon is read from the deprecation registry,
  so it cannot rot.
* Both explicit values silence it.  Neither is deprecated -- *not choosing* is.
* P16 is resolved in the same pass: ``PropagationResult.__iter__`` stays
  2-item, permanently, and is NOT scheduled to change at the flip.

Pre-fix (worktree at 865e922) every ``test_warns_*`` and every
``test_registry_*`` / ``test_p16_*`` documentation pin here fails: there was no
DeprecationWarning at all and ``API_TRANSITION_VERSION`` did not exist.

Author: audit wave 4 (P5 / roadmap F1 closure)
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
Z_SAME_GRID = 1e-4      # -> asm, bare ndarray at the input pitch
Z_GRID_CHANGING = 1e-3  # -> sas, 3-tuple at a kernel-chosen pitch
Z_FAR = 5.0             # -> fraunhofer, 3-tuple


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


def _deprecations(rec):
    return [r for r in rec if issubclass(r.category, DeprecationWarning)
            and 'return_result' in str(r.message)]


def _user_warnings(rec):
    return [r for r in rec if issubclass(r.category, UserWarning)
            and "method='auto'" in str(r.message)]


def _squash(text):
    return ' '.join((text or '').split())


# ======================================================================
# The warning fires on the default unstable path
# ======================================================================

class TestWarnsOnTheDefaultUnstablePath:
    """Option 4 flips the default for EVERY method, so the warning must
    cover the same-grid bare-ndarray case as well as the grid-changing
    3-tuple case -- i.e. strictly more than the P5 ``UserWarning``."""

    @pytest.mark.parametrize('z,expect_method,expect_type', [
        (Z_SAME_GRID, 'asm', np.ndarray),
        (Z_GRID_CHANGING, 'sas', tuple),
        (Z_FAR, 'fraunhofer', tuple),
    ])
    def test_warns_for_every_auto_selection(self, z, expect_method,
                                            expect_type):
        assert dispatch._auto_select_method(
            _gauss(), z=z, wavelength=LAM, dx=DX,
            prescription=None) == expect_method
        out, rec = _catch(z=z)
        assert isinstance(out, expect_type)
        hits = _deprecations(rec)
        assert len(hits) == 1, (
            f'expected exactly one return_result DeprecationWarning for '
            f'z={z}, got {[str(r.message)[:60] for r in hits]}')

    def test_warns_on_the_same_grid_case_the_user_warning_skips(self):
        """The delta vs the P5 ``UserWarning``: ``auto`` -> ``asm`` returns a
        bare ndarray, which the UserWarning deliberately stays silent about
        (nothing is re-gridded) but which option 4 still replaces."""
        out, rec = _catch(z=Z_SAME_GRID)
        assert isinstance(out, np.ndarray)
        assert not _user_warnings(rec), (
            'the P5 UserWarning must stay silent on the same-grid path')
        assert _deprecations(rec), (
            'the transition DeprecationWarning must fire on the same-grid '
            'path -- option 4 replaces the bare-ndarray default too')

    def test_warns_even_when_the_method_is_named_explicitly(self):
        """``method='sas'`` silences the *shape-instability* UserWarning (the
        caller named the kernel) but NOT the transition warning: naming a
        kernel says nothing about which return contract you want, and the
        default flips for named methods too."""
        out, rec = _catch(z=Z_GRID_CHANGING, method='sas')
        assert isinstance(out, tuple) and len(out) == 3
        assert not _user_warnings(rec)
        assert _deprecations(rec)

    def test_warns_on_the_z_none_passthrough(self):
        out, rec = _catch(z=None)
        assert isinstance(out, np.ndarray)
        assert _deprecations(rec)

    def test_warning_is_a_deprecationwarning_not_a_userwarning(self):
        """Category matters: the pre-existing P5 pins filter on
        ``issubclass(category, UserWarning)``, and DeprecationWarning is not
        a UserWarning subclass -- which is why adding this warning leaves
        those counter-pins green."""
        _out, rec = _catch(z=Z_SAME_GRID)
        hit = _deprecations(rec)[0]
        assert hit.category is DeprecationWarning
        assert not issubclass(DeprecationWarning, UserWarning)

    def test_pytest_warns_deprecation_contract(self):
        with pytest.warns(DeprecationWarning, match='return_result'):
            propagate(_gauss(), wavelength=LAM, dx=DX, z=Z_SAME_GRID)


# ======================================================================
# Both explicit modes are silent
# ======================================================================

class TestBothExplicitModesAreSilent:

    @pytest.mark.parametrize('z', [Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR])
    def test_return_result_true_is_fully_silent(self, z):
        out, rec = _catch(z=z, return_result=True)
        assert isinstance(out, PropagationResult)
        assert not _deprecations(rec)
        assert not _user_warnings(rec), (
            'return_result=True is the stable contract; neither warning '
            'applies')

    @pytest.mark.parametrize('z', [Z_SAME_GRID, Z_GRID_CHANGING, Z_FAR])
    def test_return_result_false_silences_the_transition_warning(self, z):
        """``False`` is an explicit choice of the legacy contract, kept
        available past the flip -- so no transition warning."""
        _out, rec = _catch(z=z, return_result=False)
        assert not _deprecations(rec)

    def test_return_result_false_keeps_the_p5_shape_instability_warning(self):
        """Counter-pin against over-silencing.  The two warnings are about
        different problems: opting into the legacy contract does not make a
        z-dependent return SHAPE stable, so the P5 UserWarning must survive
        an explicit ``return_result=False``."""
        _out, rec = _catch(z=Z_GRID_CHANGING, return_result=False)
        assert _user_warnings(rec), (
            'the P5 UserWarning must still fire for an explicit '
            'return_result=False on a grid-changing auto selection')

    def test_neither_value_is_itself_deprecated(self):
        """Not choosing is what is deprecated.  Pin that BOTH values are
        accepted without any DeprecationWarning of any origin."""
        for value in (True, False):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                propagate(_gauss(), wavelength=LAM, dx=DX,
                          z=Z_GRID_CHANGING, return_result=value)
            assert not [r for r in rec
                        if issubclass(r.category, DeprecationWarning)], (
                f'return_result={value} emitted a DeprecationWarning')


# ======================================================================
# Current return shapes are BIT-unchanged
# ======================================================================

class TestReturnShapesAreBitUnchanged:
    """The whole point of option 4: v5.30 changes nothing about the return.
    Compared in-run (no hard-coded magic numbers), so these hold on any
    platform."""

    @pytest.mark.parametrize('z,expect_type', [
        (Z_SAME_GRID, np.ndarray),
        (Z_GRID_CHANGING, tuple),
        (Z_FAR, tuple),
    ])
    def test_default_is_bit_identical_to_explicit_false(self, z, expect_type):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            default = propagate(_gauss(), wavelength=LAM, dx=DX, z=z)
            explicit = propagate(_gauss(), wavelength=LAM, dx=DX, z=z,
                                 return_result=False)
        assert type(default) is type(explicit) is expect_type
        if expect_type is np.ndarray:
            assert np.array_equal(default, explicit)
        else:
            assert len(default) == len(explicit) == 3
            assert np.array_equal(default[0], explicit[0])
            assert float(default[1]) == float(explicit[1])
            assert float(default[2]) == float(explicit[2])

    def test_sentinel_default_is_falsy_so_routing_is_unchanged(self):
        """The mechanism behind the byte-identity: ``propagate`` routes on
        ``if not return_result``, and the sentinel is falsy."""
        sig = inspect.signature(propagate)
        default = sig.parameters['return_result'].default
        assert default is dep._NO_DEFAULT
        assert bool(default) is False
        assert not default

    def test_default_matches_the_kernel_called_directly(self):
        """Independent oracle: the default path must still be a pure
        pass-through of the kernel's own return."""
        from lumenairy.propagators.propagation import (
            angular_spectrum_propagate,
            scalable_angular_spectrum_propagate,
        )
        E = _gauss()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            asm_out = propagate(E, wavelength=LAM, dx=DX, z=Z_SAME_GRID)
            sas_out = propagate(E, wavelength=LAM, dx=DX,
                                z=Z_GRID_CHANGING)
        assert np.array_equal(
            asm_out, angular_spectrum_propagate(E, Z_SAME_GRID, LAM, DX))
        oracle = scalable_angular_spectrum_propagate(
            E, Z_GRID_CHANGING, LAM, DX)
        assert np.array_equal(sas_out[0], oracle[0])
        assert float(sas_out[1]) == float(oracle[1])

    def test_the_flip_has_not_happened_yet(self):
        """Guard against landing the breaking half early."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert isinstance(
                propagate(_gauss(), wavelength=LAM, dx=DX, z=Z_SAME_GRID),
                np.ndarray)
            out = propagate(_gauss(), wavelength=LAM, dx=DX,
                            z=Z_GRID_CHANGING)
        assert isinstance(out, tuple) and len(out) == 3


# ======================================================================
# The horizon comes from the registry and cannot rot
# ======================================================================

class TestRegistryScheduling:

    def test_api_transition_version_is_a_registry_entry(self):
        assert hasattr(dep, 'API_TRANSITION_VERSION')
        assert 'API_TRANSITION_VERSION' in dep.__all__
        assert dep.API_TRANSITION_VERSION == dep.NEXT_REMOVAL_VERSION, (
            'API_TRANSITION_VERSION must stay bound to NEXT_REMOVAL_VERSION '
            'so check_removal_schedule()\'s future-version invariant covers '
            'it')

    def test_the_scheduled_version_lies_in_the_future(self):
        cur = dep._version_tuple(la.__version__)
        assert dep._version_tuple(dep.API_TRANSITION_VERSION) > cur, (
            f'API_TRANSITION_VERSION={dep.API_TRANSITION_VERSION} has '
            f'already shipped (running {la.__version__}) -- the registry rot '
            f'this mechanism exists to prevent')

    def test_registry_is_still_self_consistent(self):
        assert dep.check_removal_schedule() == []

    def test_message_quotes_the_registry_resolved_horizon(self):
        """Not a hand-written literal: the emitted version must equal what
        ``resolve_removal_version`` returns, so a re-scheduled horizon
        follows automatically."""
        live = dep.resolve_removal_version(dep.API_TRANSITION_VERSION)
        _out, rec = _catch(z=Z_SAME_GRID)
        msg = str(_deprecations(rec)[0].message)
        assert f'v{live}' in msg
        cur = dep._version_tuple(la.__version__)
        assert dep._version_tuple(live) > cur

    def test_message_names_both_escape_hatches_and_the_new_default(self):
        _out, rec = _catch(z=Z_SAME_GRID)
        msg = str(_deprecations(rec)[0].message)
        assert 'return_result=True' in msg
        assert 'return_result=False' in msg
        assert 'PropagationResult' in msg
        assert 'deprecated since v5.30' in msg
        # names the delivered legacy shape so the caller can see what it holds
        assert 'bare ndarray' in msg

    def test_message_names_the_delivered_shape_for_the_tuple_case(self):
        _out, rec = _catch(z=Z_GRID_CHANGING)
        msg = str(_deprecations(rec)[0].message)
        assert '3-tuple' in msg

    def test_message_points_at_the_roadmap_record(self):
        _out, rec = _catch(z=Z_SAME_GRID)
        msg = str(_deprecations(rec)[0].message)
        assert 'roadmap_deferred_2026_07_21.md' in msg
        assert 'F1' in msg


# ======================================================================
# P16 -- decided, and decided AGAINST changing iteration
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

    def test_no_registry_entry_schedules_an_iteration_change(self):
        """The decision was NOT to schedule it.  Guard that nobody adds a
        half-scheduled arity flip without updating this pin."""
        assert '__iter__' not in str(dep.REMOVAL_SCHEDULE)
        # ... while the registry's own prose records the decision explicitly,
        # so "not scheduled" is a stated choice rather than an omission.
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
# Documentation of the transition
# ======================================================================

class TestTransitionIsDocumented:

    def test_propagate_docstring_documents_the_transition(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'Scheduled API transition' in doc
        assert 'API_TRANSITION_VERSION' in doc
        assert 'return_result=True' in doc and 'return_result=False' in doc
        assert 'roadmap_deferred_2026_07_21.md' in doc
        assert 'option 4' in doc
        # the P5 table and the deferred-decision record must survive
        assert "method='auto'`` return contract" in inspect.getdoc(propagate)
        assert 'deferred' in doc.lower()

    def test_propagate_docstring_documents_the_deprecationwarning(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'DeprecationWarning' in doc
        # both fire-sites are described (grid-changing plus same-grid)
        assert 'same-grid bare-ndarray' in doc

    def test_propagate_docstring_records_the_p16_decision(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'P16' in doc
        assert '2-item' in doc

    def test_return_result_doc_says_not_choosing_is_deprecated(self):
        doc = _squash(inspect.getdoc(propagate))
        assert 'Leaving it unset is what is deprecated' in doc


# ======================================================================
# Library-internal calls are exempt
# ======================================================================

class TestInternalCallersAreExempt:
    """A warning fired from ``lumenairy.algebra.primitives`` names a file the
    user cannot edit and an argument the user never wrote.  Those sites
    migrate with the flip (inventoried in the roadmap), so they must not
    nag."""

    def test_freespace_operator_does_not_warn(self):
        from lumenairy.algebra.primitives import FreeSpace
        E = _gauss()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            FreeSpace(Z_GRID_CHANGING)._apply(
                E, dx=DX, dy=DX, wavelength=LAM)
        assert not _deprecations(rec)

    def test_source_propagate_does_not_warn(self):
        """``Source.propagate`` already passes ``return_result=True``, so it
        is silent by the ordinary explicit-choice route."""
        from lumenairy.sources.core import Source
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            Source(E=_gauss(), dx=DX, wavelength=LAM).propagate(
                z=Z_GRID_CHANGING)
        assert not _deprecations(rec)

    def test_caller_is_internal_classifies_module_names(self):
        assert dispatch._caller_is_internal.__doc__
        # Direct probe of the predicate via crafted caller globals.
        # ``depth=0`` asks about the DIRECT caller, i.e. ``_probe`` itself,
        # whose ``__name__`` we rebind below.  (``propagate`` passes
        # ``depth=1`` because it asks about ITS caller.)
        def _probe():
            return dispatch._caller_is_internal(0)

        for name, expect in (('lumenairy', True),
                             ('lumenairy.propagators.mhs', True),
                             ('lumenairy_other', False),
                             ('tests.unit.whatever', False),
                             ('__main__', False)):
            g = dict(_probe.__globals__)
            g['__name__'] = name
            fn = type(_probe)(_probe.__code__, g, '_probe')
            assert fn() is expect, (name, expect)

    def test_counter_pin_external_caller_does_warn(self):
        """This very test module is external, so the mechanism must NOT be
        so broad that it silences everyone."""
        assert __name__ != 'lumenairy'
        assert not __name__.startswith('lumenairy.')
        _out, rec = _catch(z=Z_SAME_GRID)
        assert _deprecations(rec)

    def test_unavailable_frame_fails_open(self):
        """When the frame cannot be reached the predicate must return False
        (i.e. warn) -- a missing introspection facility may only make the
        library noisier, never quieter.  Probed with a depth deeper than the
        stack, which is the ``ValueError`` arm of the same guard that catches
        ``AttributeError`` on a ``sys._getframe``-less interpreter."""
        assert dispatch._caller_is_internal(10 ** 6) is False
