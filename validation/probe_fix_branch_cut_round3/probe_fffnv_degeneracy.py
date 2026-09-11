"""ROUND 3: the fff_nv file's mode-match-degeneracy gate, measured.

`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`
asserts that the index-coincident groove (`eps = 2.25 = no^2 = n_sub^2`) still
VIOLATES the 1-D lossless theorem by >= 1e5 x the clean fixture's closure, and
that a closure warning fires while it does.  Its own docstring says: "If the
solver is ever made degeneracy-robust this test fails, and that is the gate
working: it must then be re-derived (durability rule), not widened."

This probe measures both arms plus the ENGINEERED pre-round-1 arm, so the
re-derivation is made on numbers rather than on the fact that it stopped
failing.
"""
import os, sys, warnings, importlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import threadpoolctl
import lumenairy
from lumenairy.elements.rcwa import rcwa_jones_1d_segments

_BOUND = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
          "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
          "lumenairy.elements.berreman")

PX, WL, DEPTH = 0.7e-6, 1.0e-6, 0.5e-6
CLEAN, DEGEN = 2.10, 2.25


def _pre_body(x, xp=None, band=1e-8):
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


class _pre_arm:
    def __enter__(self):
        self._saved = []
        for n in _BOUND:
            m = importlib.import_module(n)
            if hasattr(m, "_sqrt_decay"):
                self._saved.append((m, m._sqrt_decay))
                m._sqrt_decay = _pre_body
        return self

    def __exit__(self, *a):
        for m, f in self._saved:
            m._sqrt_decay = f
        return False


def _rot(phi, no, ne):
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex) @ R.T


_ER = _rot(np.deg2rad(35.0), 1.5, 2.3)
LADDER = list(range(11, 42, 2))


def worst(eps_groove):
    """(worst |sum R + sum T - 2| over the ladder, warnings fired)."""
    eg = np.diag([eps_groove] * 3).astype(complex)
    out, warned = 0.0, 0
    for n in LADDER:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _o, R1, T1, _J = rcwa_jones_1d_segments(
                PX, [(0.5, _ER), (0.5, eg)], 1.5, 1.0, DEPTH, WL, theta=0.0,
                n_orders=n)
        warned += sum(1 for w in rec
                      if "lossless energy closure violated" in str(w.message))
        out = max(out, abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
    return out, warned


if __name__ == "__main__":
    print(f"# lumenairy {lumenairy.__file__}")
    print(f"# py{sys.version.split()[0]} np{np.__version__} "
          f"arch={threadpoolctl.threadpool_info()[0].get('architecture','?')} "
          f"CORETYPE={os.environ.get('OPENBLAS_CORETYPE','-')} "
          f"threads={os.environ.get('OPENBLAS_NUM_THREADS','-')}")
    c, cw = worst(CLEAN)
    d, dw = worst(DEGEN)
    with _pre_arm():
        pc, pcw = worst(CLEAN)
        pd, pdw = worst(DEGEN)
    print(f"  POST clean  2.10  worst {c:.6e}  warned {cw}/{len(LADDER)}")
    print(f"  POST degen  2.25  worst {d:.6e}  warned {dw}/{len(LADDER)}")
    print(f"  PRE  clean  2.10  worst {pc:.6e}  warned {pcw}/{len(LADDER)}")
    print(f"  PRE  degen  2.25  worst {pd:.6e}  warned {pdw}/{len(LADDER)}")
    print(f"  POST degen/clean ratio {d / max(c, 1e-13):.4e}   "
          f"PRE degen/clean ratio {pd / max(pc, 1e-13):.4e}")
