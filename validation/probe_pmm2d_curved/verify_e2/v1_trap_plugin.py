"""E2 verifier item 1 -- MUTANT / trap plugin.

Loaded with ``-p v1_trap_plugin`` (PYTHONPATH containing verify_e2/): at
import it makes the Phase E2 curved-mortar kernel RAISE --
``_curvemortar.curved_cross_mass``, ``curved_cross_mass_adaptive`` and
``StagCrossOpsMapped.__init__`` -- so any test that reaches the new kernel
fails.  The unmapped and shared-map suites must stay GREEN under it; the E2
file must go RED (the trap fires).  A session-end line reports how often the
trap fired.
"""
from lumenairy.elements.pmm import _curvemortar as _CMO

FIRED = {"n": 0}


class V1Trap(RuntimeError):
    pass


def _trap(name):
    def f(*a, **kw):
        FIRED["n"] += 1
        raise V1Trap(f"V1 TRAP: {name} reached")
    return f


_CMO.curved_cross_mass = _trap("curved_cross_mass")
_CMO.curved_cross_mass_adaptive = _trap("curved_cross_mass_adaptive")
_CMO.StagCrossOpsMapped.__init__ = _trap("StagCrossOpsMapped.__init__")


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(f"V1 TRAP fired {FIRED['n']} times")
