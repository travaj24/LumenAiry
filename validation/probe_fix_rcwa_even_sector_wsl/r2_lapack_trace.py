"""R2 -- WHICH linear-algebra call first produces a non-finite number?

``DLASCL parameter number 4 had an illegal value`` is LAPACK's XERBLA report
that the scaling factor ``CFROM`` handed to ``DLASCL`` was zero or NaN.  It is
therefore a CONSEQUENCE: something upstream already produced a NaN/Inf (or an
exactly-zero norm) and fed it to a factorization.  This probe wraps every
``numpy.linalg`` entry point the RCWA solve can reach, records for each call
whether its INPUTS and its OUTPUTS are finite, and stops the recording at the
first call whose output is not finite -- with the Python traceback of that
call.  Run on both builds, the two logs localise the divergence to one line.
"""
from __future__ import annotations

import traceback

import _lib as L
import numpy as np

_LOG = []
_FIRST_BAD = {}
_WRAPPED = ("eig", "eigvals", "inv", "solve", "svd", "lstsq", "cond",
            "det", "slogdet", "qr", "pinv")


def _fin(a):
    try:
        return bool(np.all(np.isfinite(np.asarray(a))))
    except Exception:
        return True


def _amax(a):
    try:
        v = np.max(np.abs(np.asarray(a)))
        return float(v)
    except Exception:
        return float("nan")


def install():
    for name in _WRAPPED:
        orig = getattr(np.linalg, name, None)
        if orig is None:
            continue

        def make(name=name, orig=orig):
            def wrapper(*args, **kw):
                idx = len(_LOG)
                ins = [a for a in args if isinstance(a, np.ndarray)]
                in_fin = all(_fin(a) for a in ins)
                out = orig(*args, **kw)
                outs = out if isinstance(out, tuple) else (out,)
                out_fin = all(_fin(o) for o in outs)
                rec = dict(i=idx, fn=name,
                           shapes=[tuple(a.shape) for a in ins],
                           in_finite=in_fin, out_finite=out_fin,
                           in_max=[_amax(a) for a in ins],
                           out_max=[_amax(o) for o in outs])
                _LOG.append(rec)
                if (not out_fin or not in_fin) and "first" not in _FIRST_BAD:
                    _FIRST_BAD["first"] = dict(
                        rec, stack=traceback.format_stack(limit=14))
                return out
            return wrapper
        setattr(np.linalg, name, make())


def main():
    a = L.arm()
    install()
    from lumenairy.elements.rcwa import rcwa_jones_2d
    tc = L.even_sector_cell()
    kw = dict(n_orders_x=5, n_orders_y=5)
    out = {}
    for tag, sym in (("full", False), ("even", True)):
        _LOG.clear()
        _FIRST_BAD.pop("first", None)
        res = rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, 1.5, 1.0, 0.2e-6,
                            L.WL_DEFAULT, symmetry=sym, **kw)
        bad = [r for r in _LOG if not (r["in_finite"] and r["out_finite"])]
        big = [r for r in _LOG if any(m > 1e10 for m in r["out_max"]
                                      if np.isfinite(m))]
        out[tag] = dict(n_calls=len(_LOG), n_bad=len(bad),
                        first_bad=_FIRST_BAD.get("first"),
                        bad=bad[:6], huge_outputs=big[:12],
                        max_out_over_all=max(
                            (m for r in _LOG for m in r["out_max"]
                             if np.isfinite(m)), default=float("nan")),
                        R_max=float(np.max(np.abs(res[1]))),
                        R_sum=float(np.sum(res[1])))
        print("%-5s calls=%d bad=%d max|out|=%.4e Rsum=%.10f"
              % (tag, len(_LOG), len(bad), out[tag]["max_out_over_all"],
                 out[tag]["R_sum"]))
        if bad:
            print("   first bad:", bad[0]["fn"], bad[0]["shapes"],
                  "in_finite", bad[0]["in_finite"],
                  "out_finite", bad[0]["out_finite"])
    L.dump("r2_lapack_trace", out)
    print(a["build"], a["tree"])


if __name__ == "__main__":
    main()
