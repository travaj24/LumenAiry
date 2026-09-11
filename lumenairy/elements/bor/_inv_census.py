"""The explicit-inverse CENSUS HOOK for the BOR cascades -- an instrument, not
a guard.

WHY THERE IS NO GUARD HERE, AND WHY THAT IS A MEASUREMENT.  The Cartesian
engines carry a conditioning REFUSAL on their cascade inverses
(``rcwa/_core._guarded_inverse``): a conjunction of an equilibrated reciprocal
1-condition below a per-site bar AND the inverse missing its own defining
equation ``A X = I`` by more than ``_INV_RESID_REFUSE = 1e-8``.  It is armed at
exactly one site, because only there did the population separate two-sided.

On the BOR production cascades -- staggered FD (``basis='fd'``) and SEM
(``basis='sem'``) -- it does not separate at all.  Measured over **2,031
inverses on 132 fixtures** built to contain every candidate broken family
(coincident uniform layers, a 1e-6 detune of them, exactly degenerate twin
layers, thin layers down to 1e-12, near-cutoff ``k0``, ``m`` up to 10, many
rings, SEM degrees 6 / 8 / 12 / 16 with ``elements_per_segment`` 1 and 3, and a
lossy metal ring;
``docs/audits/SCOPE_BOR_MULTILAYER_GUARDS_2026_09_12.md`` section 3.2):

    site                                     n    equilibrated rcond      resid
    zcascade.py solve(Wb,Wa)               150    1.986e-07 .. 5.368e-03  2.2e-13
    zcascade.py solve(Vb,Va)               150    2.261e-07 .. 2.490e-03  1.5e-13
    sem_radial.py alpha                    100    6.677e-07 .. 2.019e-03  6.5e-14
    sem_radial.py gamma                    100    1.368e-06 .. 4.914e-04  6.0e-14
    zcascade.py inv(a+b)                   150    1.957e-06 .. 1.000e+00  1.2e-14
    sem_radial.py inv(I + gamma alpha)     100    1.962e-06 .. 9.906e-02  1.5e-13
    sem_radial.py inv(Mz)                   99    2.600e-06 .. 8.547e-04  3.4e-13
    zcascade.py Redheffer denominators  360 ea    5.522e-06 .. 1.000e+00  5.5e-15
    coupled_radial_eigensolver.py Lei       62    2.616e-05 .. 1.898e-03  8.8e-14
    sem_radial.py mortar masses         100 ea    5.603e-02 .. 1.000e+00  2.5e-16

**Whole census: rcond 1.986e-07 .. 1.000, residual at most 3.408e-13.**  ONE
population, no second mode, and therefore no bar.  ``_INV_T22_RCOND_REFUSE =
1e-10`` sits three decades below the healthy floor, ``_MORTAR_RCOND_REFUSE =
1e-12`` five, and ``_MORTAR_RESID_REFUSE = 1e-6`` seven decades ABOVE the worst
residual.  A conjunction guard armed on this population would be dormant on
every fixture ever measured, which is not a guard but a liability: it would
add a decision that has never fired and therefore has never been tested against
the geometry it would one day refuse.

So what ships is the INSTRUMENT.  ``_BOR_INV_CENSUS`` is ``None`` by default and
costs exactly one ``is None`` test per inverse; assigning a list to it makes
every BOR cascade inverse record ``(site, n, rcond, residual)`` and nothing
else.  That is what lets a future population be MEASURED rather than assumed --
the same reason ``rcwa/_core._INV_CENSUS`` exists, and the instrument the
scoping's own populations were taken with.

THE ONE SITE WITH A REMEDY ALREADY IN PLACE is ``sem_radial.py``'s ``inv(Mz)``
(the SEM ``E_z`` elimination), which detects a near-singular elimination by an
LU-pivot ratio and falls back to an unreduced QZ pencil.  That is a REPAIR, not
a refusal -- a different shape from the Cartesian guard -- and it is correct for
its site.  Nothing is ported there.
"""
from __future__ import annotations

import numpy as np

#: MEASUREMENT-ONLY HOOK.  ``None`` (the default) costs one ``is None`` test per
#: inverse and changes nothing.  Assign a list to record
#: ``(site, n, rcond, residual)`` per call; it NEVER raises and NEVER changes a
#: returned value.  This is not a behaviour switch.
_BOR_INV_CENSUS = None


def _record(site, A, X):
    """Score one operand with the LIBRARY'S OWN instruments, so the BOR
    populations are directly comparable with the Cartesian censuses."""
    from ..rcwa._core import (
        _equilibrated_inverse_residual,
        _rcond_1_equilibrated,
    )
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return
    try:
        Xi = X if X is not None else np.linalg.inv(A)
        rc = float(_rcond_1_equilibrated(A, Xi))
    except Exception:                                  # noqa: BLE001
        rc = float("nan")
    try:
        rs = float(_equilibrated_inverse_residual(A))
    except Exception:                                  # noqa: BLE001
        rs = float("nan")
    _BOR_INV_CENSUS.append((str(site), int(A.shape[0]), rc, rs))


def census_inv(A, site):
    """``np.linalg.inv(A)``, recorded when the census is armed.

    The value returned is bit-identical to the bare call: the census reads the
    operand, it never repairs it."""
    X = np.linalg.inv(A)
    if _BOR_INV_CENSUS is not None:
        _record(site, A, X)
    return X


def census_solve(A, B, site):
    """``np.linalg.solve(A, B)``, recorded when the census is armed."""
    X = np.linalg.solve(A, B)
    if _BOR_INV_CENSUS is not None:
        _record(site, A, None)
    return X
