# P2 DID-NOT-WARN diagnosis -- AXES 2 and 4: warning-CAPTURE mechanics.
#
# The brief's axis 2: "pytest.warns can miss a warning that IS emitted if the
# warning was already emitted once and the module's __warningregistry__ /
# 'default' filter dedup suppressed the repeat".  This file MEASURES that claim
# against the repo's own pytest configuration instead of asserting it.
#
# ``pyproject.toml`` sets ``filterwarnings = ["default", ...]`` -- and "default"
# IS the once-per-(text, category, lineno)-per-module dedup action, so the
# premise is live: outside a ``pytest.warns`` block a repeat IS swallowed.
#
# Run it:
#   python -m pytest validation/repro_traced_carrier_121/p2diag_capture.py -v
#
# (named ``p2diag_*`` so the suite's ``python_files = test_*.py`` never collects
# it; an explicit path on the command line bypasses that filter.)
import threading
import warnings

import pytest

_MSG = 'NOT dx-STABLE (p2diag synthetic, fixed text)'


# The real guard warns at ``stacklevel=3`` from inside
# ``_run_chain_dx_self_check`` <- ``propagate_traced_carrier_chain`` <- caller,
# so the registry that dedups it is the CALLER's module and the key's lineno is
# the CALLER's line.  Reproduce that shape exactly.
def _warn_lvl3(msg=_MSG):
    warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _mid(msg=_MSG):
    _warn_lvl3(msg)


def emit(msg=_MSG):
    """Callers of THIS get the registry entry (stacklevel 3 == our caller)."""
    _mid(msg)


def _emit_from_one_fixed_line(msg=_MSG):
    """Both probes below funnel through this ONE line, so the registry key
    (text, category, lineno) and the owning module are IDENTICAL between them
    -- the worst case the dedup hypothesis needs."""
    emit(msg)


# ===========================================================================
# CONTROL -- the 'default' action really does dedup, process-wide, when
# nothing resets the filters.  Without this row the probes below prove nothing.
# ===========================================================================
def test_control_default_action_dedups_within_one_filter_epoch():
    with warnings.catch_warnings(record=True) as wl:
        warnings.resetwarnings()
        warnings.simplefilter('default')
        _emit_from_one_fixed_line('control text')
        _emit_from_one_fixed_line('control text')   # same key -> swallowed
    assert len(wl) == 1, [str(w.message) for w in wl]


# ===========================================================================
# PROBE 1 -- warm the registry in one test, expect the warning in the NEXT
# test via pytest.warns.  This is axis 2 verbatim.
# ===========================================================================
def test_probe1a_warm_the_registry_under_the_ini_default_filter():
    """Emits the SAME text from the SAME line as probe 1b, with the repo's ini
    filters in force (pytest's per-item catch_warnings applies them)."""
    _emit_from_one_fixed_line()


def test_probe1b_pytest_warns_still_sees_it():
    """If axis 2 were the mechanism this would fail DID NOT WARN."""
    with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
        _emit_from_one_fixed_line()


# ===========================================================================
# PROBE 2 -- same, but WITHIN one test: emit once bare, then inside
# pytest.warns.  Removes the per-item catch_warnings reset from the picture,
# leaving only WarningsChecker.__enter__'s own simplefilter('always').
# ===========================================================================
def test_probe2_same_test_bare_then_pytest_warns():
    _emit_from_one_fixed_line()
    with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
        _emit_from_one_fixed_line()


# ===========================================================================
# PROBE 3 -- a leaked 'ignore' filter from an earlier test.
# ===========================================================================
def test_probe3a_leak_an_ignore_filter():
    warnings.simplefilter('ignore')          # deliberately NOT restored


def test_probe3b_pytest_warns_after_a_leaked_ignore():
    with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
        _emit_from_one_fixed_line()


# ===========================================================================
# PROBE 4 -- the ONE capture mechanism that CAN silently unhook pytest.warns:
# a background thread that entered warnings.catch_warnings() BEFORE the block
# and exits DURING it.  catch_warnings is process-global and NOT thread-safe:
# the thread's __exit__ restores warnings.showwarning to the value it saved on
# entry, ripping out the recorder pytest.warns installed.  Warnings emitted
# before the restore ARE recorded; ones after are LOST -- which is the exact
# shape of the CI report (early chain warnings present, the LAST warning of the
# call missing).
#
# xfail(strict) because it is a DEMONSTRATION of the failure mode: it is
# expected to raise Failed("DID NOT WARN").
# ===========================================================================
@pytest.mark.xfail(strict=True,
                   reason='demonstrates the thread/catch_warnings clobber; '
                          'this DID NOT WARN is the point of the probe')
def test_probe4_a_thread_leaving_catch_warnings_unhooks_pytest_warns():
    entered = threading.Event()
    release = threading.Event()

    def _worker():
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            entered.set()
            release.wait(10.0)
        # __exit__ here restores the PROCESS-WIDE showwarning/filters

    t = threading.Thread(target=_worker, name='p2diag_clobber')
    t.start()
    assert entered.wait(10.0)
    with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
        # a BYSTANDER, exactly like the chain's ray_density band warning: it
        # does NOT match the regex, and it lands in the recorder normally
        _emit_from_one_fixed_line("ray_density energy self-check FAILED -- "
                                  "P_out/P_ap = 0.8757")
        release.set()
        t.join(10.0)                        # recorder is unhooked HERE
        _emit_from_one_fixed_line()         # this one is LOST


# ===========================================================================
# PROBE 5 -- does pytest's per-item catch_warnings clear __warningregistry__
# between items?  (The C11 claim "warnings-filter leakage is already contained
# by pytest's own per-item catch_warnings".)  Measured, not assumed.
# ===========================================================================
_REG_SEEN = {}


def test_probe5a_record_registry_state():
    _emit_from_one_fixed_line('registry probe text')
    _REG_SEEN['after_first'] = dict(globals().get('__warningregistry__', {}))


def test_probe5b_registry_is_invalidated_by_the_next_item():
    import sys
    reg = getattr(sys.modules[__name__], '__warningregistry__', {})
    ver = reg.get('version')
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        _emit_from_one_fixed_line('registry probe text')
    assert len(wl) == 1, (
        'the repeat was swallowed across items -- registry version %r, '
        'entries %d' % (ver, len(reg)))
