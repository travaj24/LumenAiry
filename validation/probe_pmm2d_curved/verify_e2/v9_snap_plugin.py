"""E2 verifier item 9 -- the PROPOSED tolerance fix applied in-process:
``shapes2d._VERTEX_SNAP = shapes2d._WALL_SNAP`` (1e-12 instead of 1e-13).
Load with ``-p v9_snap_plugin`` to check the Phase C / D / E2 suites stay
green under it."""
from lumenairy.elements.pmm import shapes2d as _SH

_SH._VERTEX_SNAP = _SH._WALL_SNAP


def pytest_terminal_summary(terminalreporter):
    terminalreporter.write_line(
        f"V9 SNAP plugin: _VERTEX_SNAP = {_SH._VERTEX_SNAP!r}")
