"""An INDEPENDENT oracle for the propagating-channel COUNT of a uniform
PEC-walled BOR half-space, and what it says about the nodal basis.

THE ORACLE.  A uniform cylinder of index ``n`` closed at ``r = Rbig`` by a
perfect electric conductor supports, at azimuthal order ``m``, two families of
guided modes:

* TE (``E_z = 0``): the transverse eigenvalue ``gamma`` satisfies
  ``J_m'(gamma Rbig) = 0``;
* TM (``H_z = 0``): ``J_m(gamma Rbig) = 0``.

A mode PROPAGATES along ``z`` when ``q^2 = n^2 k0^2 - gamma^2 > 0``, i.e. when
``gamma Rbig < n k0 Rbig``.  So the number of propagating channels is just the
number of Bessel-function (and Bessel-derivative) zeros below ``n k0 Rbig``.
That is a closed form: ``scipy.special.jn_zeros`` and ``jnp_zeros``.  It
depends on NO solver in this library, which is what makes it usable as the
reference for a claim about a solver's channel set.

WHY THIS MATTERS FOR THE 5.45.1 NODAL PASSIVITY BAR.  That bar decides on
``max(R + T) - 1``.  If the returned CHANNEL SET is wrong, ``R`` and ``T`` are
different observables from the ones the physics defines, and the energy excess
measures only the power the spurious extras happen to carry.  This probe asks
the direct question the energy cannot: does each basis return the right NUMBER
of channels, and does the answer track the passivity excess the guard reads?

Usage:  python v8_channel_oracle.py <pre|post> [outdir]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from scipy.special import jn_zeros, jnp_zeros

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402


def analytic_channels(m, n_index, k0, Rbig, nz=400):
    """Number of propagating (TE + TM) channels, from the Bessel zeros."""
    x_max = float(np.real(n_index)) * float(np.real(k0)) * float(Rbig)
    te = jnp_zeros(m, nz)
    tm = jn_zeros(m, nz)
    n_te = int(np.sum(te < x_max))
    n_tm = int(np.sum(tm < x_max))
    if m == 0:
        # m = 0 TE_0n: J_0'(x) = -J_1(x), whose first zero at x = 0 is the
        # trivial (identically zero) field and is not a mode; jnp_zeros(0, .)
        # already starts at the first NON-trivial root, so nothing is dropped.
        pass
    return dict(x_max=x_max, n_te=n_te, n_tm=n_tm, n_total=n_te + n_tm,
                te_last=float(te[max(n_te - 1, 0)]),
                te_next=float(te[n_te]) if n_te < nz else None,
                tm_last=float(tm[max(n_tm - 1, 0)]),
                tm_next=float(tm[n_tm]) if n_tm < nz else None)


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _solve(basis, m, Rbig, N, k0, eps_out=2.0, eps_mid=4.0, thick=0.4):
    import lumenairy.elements.bor.bor_solve as bs
    had = hasattr(bs, "BOR_NODAL_PASSIVITY_GUARD")
    prev = getattr(bs, "BOR_NODAL_PASSIVITY_GUARD", None)
    if had:
        bs.BOR_NODAL_PASSIVITY_GUARD = False      # measure, do not be refused
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            layers = [bs.build_layer(m, Rbig, N, _uni(eps_out), k0,
                                     basis=basis),
                      bs.build_layer(m, Rbig, N, _uni(eps_mid), k0,
                                     thickness=thick, basis=basis),
                      bs.build_layer(m, Rbig, N, _uni(eps_out), k0,
                                     basis=basis)]
            r = bs.solve(layers, k0)
        R, T = np.asarray(r["R"]), np.asarray(r["T"])
        E = R + T
        return dict(n=int(R.size),
                    excess=float(np.max(E)) - 1.0 if E.size else None,
                    deficit=1.0 - float(np.min(E)) if E.size else None)
    except BaseException as e:                     # noqa: BLE001
        return dict(raised=type(e).__name__, msg=str(e)[:150])
    finally:
        if had:
            bs.BOR_NODAL_PASSIVITY_GUARD = prev


def main():
    build = sys.argv[1]
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    rows = []
    k0 = 2.0
    print("  %-3s %-6s %-5s %-6s %-6s %-6s %-6s %-12s %-12s"
          % ("m", "R/lam", "N", "exact", "stag", "nodal", "extra",
             "nodal_excess", "guard"))
    for m in (0, 1, 2, 3):
        for rl in (0.5, 1.0, 2.0, 4.0, 8.0):
            for N in (120, 200):
                Rbig = rl * 2.0 * np.pi / k0
                ana = analytic_channels(m, np.sqrt(2.0), k0, Rbig)
                st = _solve("staggered", m, Rbig, N, k0)
                nd = _solve("nodal", m, Rbig, N, k0)
                exc = nd.get("excess")
                guard = ("refuse" if (exc is not None and exc > 1e-3) else
                         "warn" if (exc is not None and exc > 1e-6) else "ok")
                rec = dict(m=m, Rbig_over_lambda=rl, N=N, Rbig=Rbig,
                           analytic=ana, stag=st, nodal=nd,
                           guard_verdict=guard,
                           stag_matches_exact=(st.get("n") == ana["n_total"]),
                           nodal_matches_exact=(nd.get("n") == ana["n_total"]),
                           nodal_extra=(nd.get("n", 0) - ana["n_total"]))
                rows.append(rec)
                print("  %-3d %-6g %-5d %-6d %-6s %-6s %-6s %-12s %-12s"
                      % (m, rl, N, ana["n_total"], st.get("n"), nd.get("n"),
                         rec["nodal_extra"],
                         ("%.4g" % exc) if exc is not None else "-", guard),
                      flush=True)
    ns = sum(1 for r in rows if r["stag_matches_exact"])
    nn = sum(1 for r in rows if r["nodal_matches_exact"])
    print("\n  staggered matches the exact count on %d of %d rows" % (ns,
                                                                      len(rows)))
    print("  nodal     matches the exact count on %d of %d rows" % (nn,
                                                                    len(rows)))
    ok_rows = [r for r in rows if r["guard_verdict"] == "ok"]
    bad_ok = [r for r in ok_rows if not r["nodal_matches_exact"]]
    print("  rows the passivity guard calls OK whose NODAL channel count is "
          "still wrong: %d of %d" % (len(bad_ok), len(ok_rows)))
    o = sys.argv[2] if len(sys.argv) > 2 else "."
    _vh.dump("%s/v8_channel_oracle_%s_%s_%s_t%s.json"
             % (o, build, a["platform"], a["loaded_kernel"],
                a["blas_threads"]),
             dict(arm=a, build=build, rows=rows,
                  stag_exact=ns, nodal_exact=nn, n_rows=len(rows),
                  guard_ok_but_wrong_count=len(bad_ok), n_guard_ok=len(ok_rows)))


if __name__ == "__main__":
    main()
