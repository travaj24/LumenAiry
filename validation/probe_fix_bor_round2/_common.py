"""Shared fixtures and arm bookkeeping for the ROUND 2 BOR guard probes.

Every probe in this directory prints the tree it loaded (``lumenairy.__file__``)
and the OpenBLAS kernel that ACTUALLY loaded (read back from ``threadpoolctl``,
never inferred from the ``OPENBLAS_CORETYPE`` request) before it measures
anything.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np


def arm():
    """The identity of the running arm: build, requested kernel, LOADED kernel,
    thread count, and the tree ``lumenairy`` was imported from."""
    import lumenairy
    rec = dict(
        tree=os.path.abspath(lumenairy.__file__),
        version=getattr(lumenairy, "__version__", "?"),
        python=sys.version.split()[0],
        platform=sys.platform,
        numpy=np.__version__,
        requested_coretype=os.environ.get("OPENBLAS_CORETYPE", "(unset)"),
        omp=os.environ.get("OMP_NUM_THREADS", "(unset)"),
        openblas_threads=os.environ.get("OPENBLAS_NUM_THREADS", "(unset)"),
    )
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
        rec["loaded"] = [dict(api=d.get("internal_api"),
                              arch=d.get("architecture"),
                              threads=d.get("num_threads"),
                              version=d.get("version")) for d in info]
        arch = [d.get("architecture") for d in info
                if d.get("internal_api") == "openblas"]
        rec["kernel"] = arch[0] if arch else "?"
        thr = [d.get("num_threads") for d in info
               if d.get("internal_api") == "openblas"]
        rec["threads"] = thr[0] if thr else None
    except Exception as exc:                      # pragma: no cover - probe
        rec["loaded"] = "threadpoolctl unavailable: %r" % (exc,)
        rec["kernel"] = "?"
    return rec


def tag(rec=None):
    r = rec or arm()
    build = "wsl" if r["platform"].startswith("linux") else "win"
    return "%s_%s_t%s" % (build, r.get("kernel", "?"), r.get("threads", "?"))


def dump(name, payload, rec=None):
    r = rec or arm()
    payload = dict(arm=r, **payload)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "%s_%s.json" % (name, tag(r)))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True, default=str)
    print("[probe] wrote %s" % (path,))
    return path


def banner(name):
    r = arm()
    print("=" * 78)
    print("%s  --  %s" % (name, tag(r)))
    print("  tree    : %s" % (r["tree"],))
    print("  version : %s   python %s   numpy %s" % (r["version"], r["python"],
                                                     r["numpy"]))
    print("  kernel  : requested %s -> LOADED %s, %s thread(s)"
          % (r["requested_coretype"], r.get("kernel"), r.get("threads")))
    print("=" * 78)
    return r


# --------------------------------------------------------------------------- #
#  permittivity profiles
# --------------------------------------------------------------------------- #
def uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def nodal_stack(e_hi, Rbig=4.0, N=200, k0=2.0, m=1, e_out=2.0 + 0j,
                basis="nodal", thickness=0.5, period=0.8):
    """The shipped gate's fixture (``test_structured_stack_energy_floor_nodal``)
    with the ring's HIGH permittivity and the outer medium left free."""
    from lumenairy.elements.bor.bor_solve import build_layer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, uni(e_out), k0, basis=basis),
                build_layer(m, Rbig, N, ring(period, e_out, e_hi), k0,
                            thickness=thickness, basis=basis),
                build_layer(m, Rbig, N, uni(e_out), k0, basis=basis)]


def uniform_stack(Rbig, N, k0, m, e_out=2.0 + 0j, e_mid=4.0 + 0j,
                  basis="nodal", thickness=0.4):
    from lumenairy.elements.bor.bor_solve import build_layer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, uni(e_out), k0, basis=basis),
                build_layer(m, Rbig, N, uni(e_mid), k0, thickness=thickness,
                            basis=basis),
                build_layer(m, Rbig, N, uni(e_out), k0, basis=basis)]


def solve_disarmed(layers, k0):
    """``bor_solve.solve`` with EVERY 5.45.1 guard off, so the row recorded is
    the number the solver RETURNS rather than the number the guard allows."""
    import lumenairy.elements.bor.bor_solve as bs
    prev = bs.BOR_NODAL_PASSIVITY_GUARD
    bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return bs.solve(layers, k0)
    finally:
        bs.BOR_NODAL_PASSIVITY_GUARD = prev


# --------------------------------------------------------------------------- #
#  the Bessel-zero channel-count oracle (depends on NO solver in this library)
# --------------------------------------------------------------------------- #
def exact_channel_count(m, n_index, k0, Rbig, nz=400):
    """``(count, basin)`` for a UNIFORM PEC-walled cylinder of index ``n``.

    TE modes satisfy ``J_m'(gamma Rbig) = 0``, TM modes ``J_m(gamma Rbig) = 0``,
    and a mode propagates when ``gamma < n k0``.  ``basin`` is the relative
    distance from the propagation cut ``x = n k0 Rbig`` to the nearest zero on
    either side: the count can only move if a zero CROSSES, so a wide basin is
    what makes the oracle build-free.
    """
    from scipy.special import jn_zeros, jnp_zeros
    x = float(np.real(n_index)) * float(np.real(k0)) * float(Rbig)
    z = np.sort(np.concatenate([jnp_zeros(m, nz), jn_zeros(m, nz)]))
    n = int(np.sum(z < x))
    below = x - z[n - 1] if n > 0 else x
    above = z[n] - x if n < z.size else float("inf")
    return n, float(min(below, above) / x)
