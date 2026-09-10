"""V13 -- where the stray LAPACK line comes from.

    ** On entry to DLASCL parameter number  4 had an illegal value

`DLASCL(TYPE, KL, KU, CFROM, CTO, M, N, A, LDA, INFO)` sets `INFO = -4` when
`CFROM` is zero or NaN, and LAPACK's error handler `XERBLA` then writes that
line to Fortran unit 6.  Unit 6 is buffered and flushed at interpreter exit, so
the line's position in a pytest log attributes nothing.

Bisected: on WSL the whole of `tests/unit/test_m1_conditioning_guard.py` emits
exactly two such lines and every other test file emits none; running its 27
node ids one at a time, the two lines come from ONE test,
`test_guarded_lstsq_stands_aside_on_a_non_finite_system`, which DELIBERATELY
hands a non-finite least-squares system to the guard.  This probe reproduces
each of that test's two calls in isolation and counts the lines, so the
attribution is per-CALL rather than per-test.

`A` is COMPLEX, so the driver is `zgelsd`; `zgelsd` scales the REAL singular
values with `DLASCL`, which is why the complaint names the double-REAL routine.

Usage:  python v13_dlascl.py <case>      case in {A, b, both, none}
"""
import sys

import numpy as np


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "both"
    from lumenairy.elements.pmm import _core as pc
    rng = np.random.default_rng(41)
    A = rng.standard_normal((12, 4)) + 1j * rng.standard_normal((12, 4))
    b = rng.standard_normal(12) + 0j
    if case in ("A", "both"):
        A_bad = A.copy()
        A_bad[3, 1] = np.nan
        try:
            pc._guarded_lstsq(A_bad, b, "probe")
            print("case A: NO raise")
        except np.linalg.LinAlgError as exc:
            print("case A: LinAlgError %s" % exc)
    if case in ("b", "both"):
        b_bad = b.copy()
        b_bad[5] = np.nan
        got = pc._guarded_lstsq(A, b_bad, "probe")
        print("case b: returned, %d NaN entries of %d"
              % (int(np.sum(np.isnan(got))), got.size))
    if case == "none":
        pc._guarded_lstsq(A, b, "probe")
        print("case none: clean system solved")
    print("done")


if __name__ == "__main__":
    main()
