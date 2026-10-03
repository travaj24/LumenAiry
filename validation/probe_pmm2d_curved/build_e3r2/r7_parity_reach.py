"""R7 (round-2 item 8): the per-fixture parity bars of the unit gate E3-2.
For every fixture of ``tests/unit/test_pmm2d_staggered_curved_e3.py::_PAR``
at M = 3: the twin's difference from the NumPy stack (max over R, T, J) and
the eig stage's own round-off REACH on the same fixture -- the NumPy stack
with its QZ replaced by the standard eig of G^-1 L (the twin's reduction,
an equally exact algorithm), minus the shipped NumPy stack.

    python r7_parity_reach.py
"""
import os
import sys

import scipy.linalg as _sla
from _r2 import HERE, dump, np

from lumenairy.elements.pmm import twod_staggered as TS

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                                "tests", "unit")))
import test_pmm2d_staggered_curved_e3 as E3  # noqa: E402


class _SE:
    """scipy.linalg stand-in whose eig(L, G) is the standard eig of G^-1 L."""

    def __getattr__(self, k):
        return getattr(_sla, k)

    @staticmethod
    def eig(a, b=None, **kw):
        if b is None:
            return _sla.eig(a, **kw)
        return np.linalg.eig(np.linalg.solve(b, a))


def amax3(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(a, b))


out = {"M": 3, "fixtures": {}}
for name, fx in sorted(E3._PAR.items()):
    kw = {k: v for k, v in fx.items() if k != "layers"}
    _o, R, T, J = E3._stack(fx["layers"], M=3, backend="numpy", **kw).solve()
    TS.sla = _SE()
    try:
        _o, R2, T2, J2 = E3._stack(fx["layers"], M=3, backend="numpy",
                                   **kw).solve()
    finally:
        TS.sla = _sla
    _o, Rj, Tj, Jj = E3._stack(fx["layers"], M=3, **kw).solve()
    rec = {"twin_numpy": amax3((Rj, Tj, Jj), (R, T, J)),
           "reach": amax3((R2, T2, J2), (R, T, J))}
    rec["ratio"] = rec["twin_numpy"] / max(rec["reach"], 1e-300)
    out["fixtures"][name] = rec
    print(name, "twin-numpy %.1e  reach %.1e  ratio %.2f" % (
        rec["twin_numpy"], rec["reach"], rec["ratio"]), flush=True)
print(dump("r7_parity_reach.json", out))
