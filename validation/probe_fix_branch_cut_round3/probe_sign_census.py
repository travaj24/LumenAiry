"""ROUND 3 item 3: a BUILD-INDEPENDENT fail-before quantity for the two
engineered tests whose claims were made on amplified rounding.

THE QUANTITY.  During a solve, count the LAYER modes the branch selector
returns on the INCOMING root while being numerically ON THE CUT:

    on the cut   <=>  |Re(lam)| <= _CUT_BAND_REL * max(max|lam|, 1)
    incoming     <=>  Im(lam) < 0

A mode on the cut is PROPAGATING and LOSSLESS: its ``lam^2`` is a negative real
to within the eigensolver's backward error.  Its two roots are ``+i|kz|``
(OUTGOING -- the branch the S-matrix recursion requires of a layer's FORWARD
set) and ``-i|kz|`` (INCOMING).  Which one comes back is a SIGN, so the count
is a decision, not a reading.

Post-fix the count is ZERO BY CONSTRUCTION -- that is what the selector does.
Pre-fix the sign is the eigensolver's backward error, so it is a coin flip per
mode; over a ladder of truncations with tens of modes each, "at least one"
survives every BLAS kernel, where the closure MAGNITUDE does not.
"""
import os, sys, importlib, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import threadpoolctl
import lumenairy.elements.rcwa._core as _rc

_BOUND = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
          "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
          "lumenairy.elements.berreman")


def _pre_body(x, xp=None, band=1e-8):
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


class _arm:
    """Install ``body`` (or the shipped selector when None) wrapped in the
    census, at every module binding."""

    def __init__(self, body=None):
        self.body = body
        self.n_incoming = 0
        self.n_oncut = 0
        self.n_arrays = 0

    def __enter__(self):
        self._saved = []
        base = self.body

        def wrapped(x, xp=None, band=_rc._CUT_BAND_REL):
            fn = base if base is not None else self._shipped
            out = fn(x, xp, band)
            try:
                lam = np.asarray(out)
                if lam.size > 4 and np.all(np.isfinite(lam)):
                    scale = max(float(np.max(np.abs(lam))), 1.0)
                    oncut = np.abs(lam.real) <= _rc._CUT_BAND_REL * scale
                    self.n_arrays += 1
                    self.n_oncut += int(oncut.sum())
                    self.n_incoming += int((oncut & (lam.imag < 0)).sum())
            except Exception:
                pass
            return out

        for name in _BOUND:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._shipped = mod._sqrt_decay
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = wrapped
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


# ------------------------------------------------------- fixture A: spacer
from lumenairy.elements.pmm import PMM2DStackHybrid

_P, _WL, _D = 0.6e-6, 0.55e-6, 0.25e-6
_HOST, _WEAK, _N_SUB = 2.25, 2.26, 1.63


def _pmm_stack(n_orders, spacer=_HOST):
    e = np.full((6, 6), _HOST + 0j)
    e[2:4, 2:4] = _WEAK
    st = PMM2DStackHybrid(_P, _P, n_substrate=_N_SUB, n_superstrate=1.0,
                          degree=7, n_orders=n_orders, symmetry=False)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    st.add_layer(_D, eps_cell=e)
    if spacer is not None:
        st.add_layer(0.1e-6, eps=spacer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.set_source(_WL, theta=0.0).solve()


def census_spacer(body):
    with _arm(body) as a:
        for M in (3, 4, 5):
            _pmm_stack(M)
    return a


def census_spacer_detuned(body):
    with _arm(body) as a:
        for M in (3, 4, 5):
            _pmm_stack(M, spacer=_HOST * 1.01)
    return a


# --------------------------------------------------------- fixture B: thin
from lumenairy.elements.rcwa import rcwa_efficiency_1d

_THIN_LADDER = list(range(6, 31))


def census_thin(body, pol="te"):
    with _arm(body) as a:
        for M in _THIN_LADDER:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rcwa_efficiency_1d(10e-6, 1.55, 1.5, 1.5, 1.5,
                                       0.5e-6, 0.5, 700e-9, angle=0.0,
                                       n_orders=M, polarization=pol,
                                       stabilize=False)
            except Exception:
                pass
    return a


if __name__ == "__main__":
    arch = threadpoolctl.threadpool_info()[0].get("architecture", "?")
    print(f"# py{sys.version.split()[0]} np{np.__version__} arch={arch} "
          f"CORETYPE={os.environ.get('OPENBLAS_CORETYPE', '-')}")
    for nm, fn in (("A spacer coincident", census_spacer),
                   ("A spacer DETUNED  ", census_spacer_detuned),
                   ("B thin ladder TE  ", census_thin)):
        for arm, body in (("POST", None), ("PRE ", _pre_body)):
            a = fn(body)
            print(f"  {nm}  {arm}: arrays {a.n_arrays:4d}  on-cut "
                  f"{a.n_oncut:5d}  INCOMING {a.n_incoming:5d}")


# ------------------------------------------- the COINCIDENCE half: rcond(a+b)
def rcond_census(fn, body):
    """Worst LAPACK reciprocal condition recorded at any guarded inverse
    during the fixture, i.e. the mode-match ``a + b`` at its worst interface."""
    prev = _rc._INV_CENSUS
    _rc._INV_CENSUS = []
    try:
        fn(body)
        rows = [c for c in _rc._INV_CENSUS if np.isfinite(c[2])]
    finally:
        _rc._INV_CENSUS = prev
    return (min((c[2] for c in rows), default=float("nan")), len(rows))
