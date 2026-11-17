"""The NumPy version floor, checked once at import.

``pyproject.toml`` declares the floor, but a package imported from a source
checkout (``PYTHONPATH``, an editable install made with ``--no-deps``, a
vendored copy) bypasses the installer that enforces it.  This module carries
the same floor as a constant and warns once, at import, when the running NumPy
is older.  A test pins this constant to the ``pyproject.toml`` and
``requirements.txt`` declarations so the three cannot drift.

WHY THE FLOOR IS 2.1 AND NOT 2.0.  NumPy builds made from source without an
external BLAS/LAPACK fall back to the bundled ``lapack_lite`` (the meson
option ``allow-noblas`` defaults to true, so the fallback is silent).
``lapack_lite`` is f2c-translated reference LAPACK in which every local
variable is ``static`` -- e.g. ``zgetrf_`` keeps its loop and block indices in
process-global storage -- and ``numpy.linalg`` releases the GIL around it.
Two threads inside ``numpy.linalg.inv`` (or ``solve``, ``eig``, ...) at once
then overwrite each other's indices and write out of bounds: an access
violation or heap corruption that kills the process.  NumPy 2.1.0 added a
global lock that serialises every ``lapack_lite`` call; 2.0.x and all 1.x
releases have none.  The library's threaded sweeps (``RCWAStack.
solve_vs_wavelength``, the PMM sweeps) and any caller that runs solves on its
own thread pool are therefore unsafe on such a build before 2.1.  Builds that
link an external LAPACK (every official NumPy wheel bundles OpenBLAS) are not
affected at any version; the floor is set where the source-build hazard ends
so one rule covers every way of obtaining NumPy.
"""
from __future__ import annotations

import warnings

NUMPY_FLOOR = "2.1"


def _version_tuple(version: str) -> tuple:
    """Leading numeric release components: ``'2.1.3'`` -> ``(2, 1, 3)``,
    ``'2.2.0rc1'`` -> ``(2, 2, 0)``, ``'2.3.dev0+git'`` -> ``(2, 3)``."""
    out = []
    for part in version.split("+")[0].split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        out.append(int(digits))
        if len(digits) != len(part):
            break
    return tuple(out)


def numpy_below_floor(version: str | None = None) -> bool:
    """True when ``version`` (default: the running NumPy) is older than
    :data:`NUMPY_FLOOR`."""
    if version is None:
        import numpy
        version = numpy.__version__
    return _version_tuple(version) < _version_tuple(NUMPY_FLOOR)


def check_numpy_floor(version: str | None = None) -> bool:
    """Warn (``RuntimeWarning``) when the running NumPy is below the floor.

    Returns True when a warning was issued.  Called once from
    ``lumenairy/__init__.py``; ``version`` exists for tests."""
    if version is None:
        import numpy
        version = numpy.__version__
    if not numpy_below_floor(version):
        return False
    warnings.warn(
        f"lumenairy requires numpy>={NUMPY_FLOOR}; this process has numpy "
        f"{version}, which the package metadata does not allow (it was "
        f"probably imported from a source checkout without installing its "
        f"dependencies).  Results are not supported.  In particular, a NumPy "
        f"older than 2.1 that was built without an external BLAS/LAPACK runs "
        f"numpy.linalg through a non-thread-safe fallback, and threaded "
        f"solves (RCWAStack.solve_vs_wavelength, PMM sweeps, or solves on "
        f"your own thread pool) can crash the process.  Upgrade with "
        f"`pip install -U \"numpy>={NUMPY_FLOOR}\"`.",
        RuntimeWarning, stacklevel=2)
    return True
