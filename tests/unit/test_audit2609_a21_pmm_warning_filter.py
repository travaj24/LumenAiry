"""WP-A21: the staggered SEGMENT-grid redundancy warning is catchable in CI.

VERIFY-A13 recorded, while re-running ``test_pmm2d_staggered_mortar.py``, that
it had to count the warnings by grepping the message text because
``-W error::UserWarning:lumenairy.elements.pmm.twod_staggered`` "does not
reliably catch it" -- and left the question of what CI should do open.

This file settles it by measurement.  The finding is not a stacklevel bug to
be tuned away: Python matches a filter's ``module`` field against the
``__name__`` of the frame ``stacklevel`` selects, and a non-1 ``stacklevel``
exists precisely to select the CALLER, so a correctly attributed warning is
NEVER attributed to the module that raises it.  Any ``stacklevel`` that points
a user at their own code makes a module filter on the library module fail.  The
filter CI must use is therefore a MESSAGE filter, and that is what is pinned
here -- together with the counter-pin that the module filter does NOT work, so
that nobody re-arms the broken gate on the strength of the VERIFY-A13 note.

FIXTURE COST.  The warning is about a ~1000x cost cliff, so the obvious
fixture (the audit's 12x12 half-fill pillar) is measured at 498 s CPU / 8.4 GB
-- exactly what must not be run in a unit test.  The smallest fixture that
reaches the same code path is a 4x4 SEGMENT grid whose walls sit on the 2x2
lattice at M = 3, n_orders = 1: ``n_min = 2 < min(Nx, Ny) = 4`` makes
``redundant`` true, and the pencil is ``2 * (4*2)^2 = 128`` -- MEASURED 1.9 s
for this whole file, all four tests, on the calibration box.

No numeric tolerance appears anywhere below: every assertion is on whether a
warning was raised, on which module it was attributed to, or on an exact
integer.
"""
import re
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    _validate_stag_cost,
    pmm_efficiency_2d_staggered,
)

_P, _WL, _M = 1.2, 0.85, 3

#: The filter CI must use, in ``-W`` / pytest ``filterwarnings`` spelling.
#: ``action:message:category:module:lineno``; the message field is compiled
#: with ``re.compile(...).match(...)``, i.e. anchored at the start, which is
#: why it opens with ``.*`` -- the text begins with the entry point's own
#: ``fn_name`` and that differs per path.
CI_FILTER = 'error:.*eps_cell is a SEGMENT grid:UserWarning'

#: The module filter VERIFY-A13 tried.  Kept as a NEGATIVE control.
BROKEN_FILTER_MODULE = 'lumenairy.elements.pmm.twod_staggered'


def _cell4():
    """4x4 SEGMENT grid, walls on the 2x2 lattice -> the advice fires."""
    c = np.full((4, 4), complex(2.25))
    c[0:2, 0:2] = 6.0
    return c


def _direct():
    """The functional entry: warns from one frame below the caller."""
    pmm_efficiency_2d_staggered(_P, _P, _cell4(), 1.0, 1.0, 0.30, _WL,
                                n_modes=_M, n_orders=1)


def _deferred():
    """The shared-stack entry: the warn arm is deferred to ``solve``."""
    st = PMM2DStackPure(_P, n_modes=_M, n_orders=1)
    st.add_layer(0.30, eps_cell=_cell4())
    st.set_source(_WL, theta=0.0, phi=0.0)
    st.solve()


@pytest.mark.parametrize('entry,name', [(_direct, 'direct'),
                                        (_deferred, 'deferred')])
def test_the_documented_message_filter_turns_the_advice_into_an_error(entry,
                                                                      name):
    """The filter written into ``twod_staggered.py`` beside the warning works
    on BOTH entry points.

    This is the deliverable: CI can arm this gate, and the spelling that arms
    it is the one the source comment carries, not a paraphrase of it.
    """
    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter('ignore')
        action, message, category = CI_FILTER.split(':', 2)
        warnings.filterwarnings(action, message=message,
                                category=UserWarning)
        with pytest.raises(UserWarning, match='eps_cell is a SEGMENT grid'):
            entry()


@pytest.mark.parametrize('entry,name', [(_direct, 'direct'),
                                        (_deferred, 'deferred')])
def test_a_module_scoped_filter_on_this_module_does_not_catch_it(entry, name):
    """COUNTER-PIN, and the reason the message filter is the answer.

    ``-W error::UserWarning:lumenairy.elements.pmm.twod_staggered`` lets the
    warning through on every path -- MEASURED here, not assumed.  If a later
    change ever made this test fail, the warning would have been re-attributed
    to the library module, i.e. it would have stopped pointing at the caller's
    code, which is a REGRESSION even though it would make the module filter
    work.  So the assertion is deliberately two-edged: it pins both that CI
    must not use this filter and that the warning still points outward.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter('ignore')
        warnings.filterwarnings('error', category=UserWarning,
                                module=re.escape(BROKEN_FILTER_MODULE))
        entry()                              # must NOT raise
    assert not [w for w in caught
                if 'SEGMENT grid' in str(w.message)], (
        'the module-scoped filter is now catching the redundancy advice, '
        'which means the warning is being attributed to '
        f'{BROKEN_FILTER_MODULE} instead of to the caller.  A user reading '
        'the warning now gets a library line, not their own call site.')


def test_the_warning_is_attributed_to_the_caller_on_the_direct_path():
    """The direct functional entry points at the CALLER's line.

    ``stacklevel=3`` from ``_validate_stag_cost`` is frame 1 =
    ``_validate_stag_cost``, 2 = ``pmm_efficiency_2d_staggered``, 3 = this
    test -- so ``w.filename`` must be THIS file.  That is the property the
    module filter above is the price of.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _direct()
    hits = [w for w in caught if 'SEGMENT grid' in str(w.message)]
    assert len(hits) == 1, [str(w.message)[:60] for w in caught]
    assert hits[0].filename == __file__, (
        f'the advice is reported at {hits[0].filename}, not at the calling '
        f'line in {__file__}; a caller cannot see which of their calls is '
        f'expensive.')
    assert hits[0].message.args[0].startswith('pmm_efficiency_2d_staggered:')


def test_the_stacklevel_is_a_parameter_so_the_deferred_path_can_correct_it():
    """``_validate_stag_cost(..., stacklevel=n)`` exists and is honoured.

    The deferred ``PMM2DStackPure.solve`` path reaches the warning through one
    extra helper, so at the default 3 it lands on ``stack2d_pure.py``'s own
    line rather than on the user's ``solve()`` call.  Fixing that is a
    one-word change at a call site this work package does not own; the
    parameter is the half that is owned here, and this pins that it works --
    otherwise the request would land on a keyword that silently did nothing.

    Bar: at ``stacklevel=1`` the warning must be reported inside
    ``twod_staggered.py`` (the raising frame) and at ``stacklevel=2`` inside
    this file (the immediate caller).  Exact filenames, no tolerance.
    """
    import lumenairy.elements.pmm.twod_staggered as _ts

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _validate_stag_cost('probe', _M, _cell4(), check=('warn',),
                            stacklevel=1)
    hits = [w for w in caught if 'SEGMENT grid' in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == _ts.__file__, (
        f'stacklevel=1 must report inside {_ts.__file__}; got '
        f'{[w.filename for w in hits]}')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _validate_stag_cost('probe', _M, _cell4(), check=('warn',),
                            stacklevel=2)
    hits = [w for w in caught if 'SEGMENT grid' in str(w.message)]
    assert len(hits) == 1 and hits[0].filename == __file__, (
        f'stacklevel=2 must report at this call site; got '
        f'{[w.filename for w in hits]}')


def test_the_deferred_path_reports_at_the_callers_line():
    """The shared-stack entry must point at the user's ``solve()``.

    The direct path already does
    (``test_the_warning_is_attributed_to_the_caller_on_the_direct_path``); the
    deferred one reaches the warning through
    ``PMM2DStackPure._warn_stag_shared_redundancy``, one frame further down,
    so it needs ``stacklevel=4`` at that call site rather than the default 3.

    FAIL-BEFORE, measured on the pre-fix tree: ``hits[0].filename`` was
    ``stack2d_pure.py`` (the library's own ``solve`` line, 1441), not this
    file.  Bar: an exact filename, no tolerance.  The advice is about a
    ~1000x cost cliff, and a caller who cannot see WHICH of their calls is
    expensive cannot act on it.
    """
    import lumenairy.elements.pmm.stack2d_pure as _sp

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _deferred()
    hits = [w for w in caught if 'SEGMENT grid' in str(w.message)]
    assert len(hits) == 1, [str(w.message)[:60] for w in caught]
    assert hits[0].filename == __file__, (
        f'the deferred advice is reported at {hits[0].filename} '
        f'(the library\'s own line) rather than at the calling line in '
        f'{__file__}.  Pass stacklevel=4 at {_sp.__file__}, in '
        f'_warn_stag_shared_redundancy.')
    assert hits[0].message.args[0].startswith('PMM2DStackPure.solve:')
