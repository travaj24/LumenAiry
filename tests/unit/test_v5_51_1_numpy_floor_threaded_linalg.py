"""NumPy floor 2.1 -- threaded ``numpy.linalg`` on a BLAS-less NumPy build.

THE DEFECT (outside this library, but it surfaced as a crash inside it).  A
NumPy built from source with no external BLAS/LAPACK silently falls back to
the bundled ``lapack_lite`` (meson ``allow-noblas`` defaults to true).
``lapack_lite`` is f2c-translated LAPACK whose locals are all ``static`` --
``zgetrf_``, which ``numpy.linalg.inv`` reaches through ``zgesv_``, keeps
``i__, j, jb, nb, iinfo`` in process-global storage -- and ``numpy.linalg``
releases the GIL around it.  Two threads inside it at once overwrite each
other's indices and write out of bounds.  NumPy 2.1.0 added a global lock
(``lapack_lite_lock`` in ``umath_linalg.cpp``) around every ``lapack_lite``
call; 1.x and 2.0.x have none.

MEASURED on Windows: a 4-thread ``numpy.linalg.inv`` loop on thread-PRIVATE
matrices (no shared arrays, no lumenairy) crashed 20 of 20 runs (access
violation / 0xc0000374 heap corruption) on a source-built, BLAS-less numpy
1.26.4 and 10 of 10 on a BLAS-less 2.0.2; it was clean on 30 of 30 on a
BLAS-less 2.1.3 build and on every official OpenBLAS wheel tested (1.24.4,
1.26.4, 2.0.2, 2.1.3, 2.2.6, 2.4.4, 2.5.5).  On the BLAS-less 1.26.4 build,
``RCWAStack.solve_vs_wavelength(max_workers=4)`` crashed 10 of 10 runs, and
so did independent ``RCWAStack.solve`` calls on a caller's 4-thread pool.  The library's own
threaded RCWA path shares no mutable state between workers (private stack
clones; the one shared cache is lock-guarded and read-only), and it calls no
native code other than NumPy/SciPy, so it cannot corrupt the heap by itself.

WHAT IS PINNED HERE.  The crash needs a particular NumPy BUILD, which no CI
leg has, so it cannot be reproduced deterministically in the suite.  What CAN
be pinned deterministically is the remedy: the floor is 2.1 in all three
places it is declared, and the import-time check warns below it (a source
checkout on ``PYTHONPATH`` bypasses the installer that enforces the floor --
which is how the crash was met).  The last test is an ENVIRONMENT CANARY, run
in a subprocess so a native crash fails one test instead of killing the run:
on any supported NumPy it passes; on an unsafe build it reports the cause.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

from lumenairy import _numpy_floor
from lumenairy._numpy_floor import (
    NUMPY_FLOOR,
    check_numpy_floor,
    numpy_below_floor,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _declared_numpy_floor(path):
    """The ``numpy>=X`` floor declared in a requirements-style text file or in
    ``pyproject.toml``'s dependency list (the first ``"numpy>=..."``)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r'^\s*"?numpy>=([0-9][0-9.]*)"?', text, flags=re.MULTILINE)
    assert m is not None, f"no numpy>= requirement found in {path}"
    return m.group(1)


def test_floor_constant_matches_every_declaration():
    """``_numpy_floor.NUMPY_FLOOR`` == pyproject == requirements.txt, so the
    runtime warning and the installer enforce the SAME floor."""
    assert NUMPY_FLOOR == "2.1"
    assert _declared_numpy_floor(os.path.join(REPO_ROOT, "pyproject.toml")) == NUMPY_FLOOR
    assert _declared_numpy_floor(os.path.join(REPO_ROOT, "requirements.txt")) == NUMPY_FLOOR


@pytest.mark.parametrize("version, below", [
    ("1.24.4", True),
    ("1.26.4", True),
    ("2.0.0", True),
    ("2.0.2", True),
    ("2.1.0rc1", False),
    ("2.1.0", False),
    ("2.1.3", False),
    ("2.2.6", False),
    ("2.10.0", False),
    ("3.0.0.dev0+git123", False),
])
def test_below_floor_comparison(version, below):
    """Numeric, not lexicographic (``2.10`` > ``2.1``), and tolerant of
    pre-release / local suffixes."""
    assert numpy_below_floor(version) is below


def test_check_warns_below_floor_and_names_the_hazard():
    with pytest.warns(RuntimeWarning, match=r"numpy>=2\.1.*numpy 2\.0\.2") as rec:
        assert check_numpy_floor("2.0.2") is True
    msg = str(rec[0].message)
    assert "thread" in msg and "BLAS" in msg


def test_check_is_silent_at_and_above_floor():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_numpy_floor("2.1.0") is False
        assert check_numpy_floor(np.__version__) is False


def test_running_numpy_satisfies_the_floor():
    """The suite itself must run on a supported NumPy -- otherwise every
    other result in it is from an unsupported configuration."""
    assert not numpy_below_floor(), (
        f"numpy {np.__version__} is below the declared floor {NUMPY_FLOOR}")


def test_import_calls_the_check():
    """The package import is what runs the check (the guard is only useful if
    it fires for a source-checkout import that bypassed the installer)."""
    import lumenairy
    src = open(lumenairy.__file__, encoding="utf-8").read()
    assert "_check_numpy_floor()" in src
    assert _numpy_floor.check_numpy_floor is check_numpy_floor


_CANARY = r"""
import numpy as np
from lumenairy.elements.rcwa import RCWAStack

def pillar(S, hw=0.2, eps=8.0, bg=2.1):
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    c = np.full((S, S), bg, complex)
    c[(np.abs(X) < hw) & (np.abs(Y) < hw)] = eps
    return c

st = RCWAStack(period=0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
               n_substrate=1.5, n_orders=5, n_orders_y=5)
st.add_layer(0.08e-6, eps=2.25)
st.add_layer(0.12e-6, eps_cell=pillar(40))
st.set_source(0.55e-6, theta=0.1, phi=0.2)
wls = np.linspace(0.50e-6, 0.70e-6, 8)
ser = st.solve_vs_wavelength(wls, max_workers=1)
for _ in range(3):
    par = st.solve_vs_wavelength(wls, max_workers=4)
    for a, b in zip(ser[1:], par[1:]):
        d = float(np.nanmax(np.abs(a - b)))
        assert d == 0.0, d
print("CANARY-OK")
"""


def test_threaded_rcwa_sweep_canary_in_subprocess():
    """Threaded ``solve_vs_wavelength`` (4 workers, 3 repeats) completes and is
    byte-identical to serial, in a SUBPROCESS so a native crash on an unsafe
    NumPy build fails this test instead of taking the run down.

    MEASURED, this exact script run 10 times per build: the process crashed
    10 of 10 times on BLAS-less numpy 1.26.4 and 10 of 10 on BLAS-less 2.0.2,
    and passed 10 of 10 on BLAS-less 2.1.3.  The test passes on the official
    wheels it was run on (Windows 2.1.3 and 2.4.4, Linux 2.4.6)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env["PYTHONFAULTHANDLER"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CANARY], cwd=REPO_ROOT, env=env,
        capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600)
    assert proc.returncode == 0 and "CANARY-OK" in proc.stdout, (
        f"threaded RCWA sweep failed in a subprocess (rc={proc.returncode}).  "
        f"A native crash here (access violation, heap corruption, rc<0 or a "
        f"large Windows status code) with numpy.linalg on the faulthandler "
        f"stack means this NumPy runs LAPACK through a non-thread-safe "
        f"fallback -- see the module docstring and lumenairy/_numpy_floor.py."
        f"\n--- stdout ---\n{proc.stdout[-2000:]}\n--- stderr ---\n"
        f"{proc.stderr[-4000:]}")
