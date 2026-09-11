"""STEP 3 instrument: the legacy NODAL cascade's passivity population, and the
STAGGERED twin of every row.

WHY PASSIVITY AND NOT CONDITIONING.  The scoping measured the nodal
``inv(a + b)``'s equilibrated residual at 5.998e-13 against the Cartesian
conjunction's ``_INV_RESID_REFUSE = 1e-8`` -- four to five decades below the
bar -- so ``_guarded_inverse`` as written refuses 0 of 6 broken rows on every
kernel.  The operator is not singular; it AMPLIFIES.  What separates is
``max(R + T)`` on a provably passive lossless stack.

WHAT THIS MEASURES, and why the scoping's own population is not enough.  The
scoping's broken arm was a FIVE-layer ring stack at ``m = 1``, ``N = 140``,
which reads ``max(R + T)`` from 5.06 to 966.7.  But the library also SHIPS a
gate (``test_bor_solve::test_structured_stack_energy_floor_nodal``) that
deliberately exercises the nodal basis on a THREE-layer stack at a small cell
and pins its documented "~1-4% floor".  A bar derived from the broken arm alone
would refuse that gate.  So this probe measures BOTH arms -- the broken family
AND every nodal fixture the shipped suite builds -- plus the staggered twin of
each, and the bar is derived from the gap between them.

Usage: ``python s3_nodal_passivity.py <tag>``.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def _uni(e):
    return lambda r: np.full_like(r, complex(e), dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    """The shipped test suite's own ring profile (``test_bor_solve._ring``),
    verbatim: a RADIAL GRATING of the given period, not a single core wall."""
    def f(r):
        e = np.full_like(r, complex(e_lo), dtype=complex)
        e[(r % period) < duty * period] = complex(e_hi)
        return e
    return f


def _solve(basis, m, Rbig, N, k0, layers_spec):
    """Solve with the passivity screen DISARMED, and separately record whether
    the ARMED screen would have refused.

    The population must be measured on the numbers the solver RETURNS, not on
    the ones it is now willing to return -- otherwise the census certifies the
    guard against itself."""
    from lumenairy.elements.bor import bor_solve as _bs
    from lumenairy.elements.bor.bor_solve import build_layer, solve
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        return _solve_inner(build_layer, solve, basis, m, Rbig, N, k0,
                            layers_spec, _bs)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev


def _solve_inner(build_layer, solve, basis, m, Rbig, N, k0, layers_spec, _bs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ls = []
        for prof, thk in layers_spec:
            ls.append(build_layer(m, Rbig, N, prof, k0, basis=basis,
                                  thickness=thk))
        res = solve(ls, k0)
    _ls = ls
    e = np.asarray(res["energy"])
    refused = None
    if basis == "nodal":
        _bs.BOR_NODAL_PASSIVITY_GUARD = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _bs._check_nodal_passivity(_ls, e)
            refused = False
        except _bs.BORNodalPassivityError:
            refused = True
        finally:
            _bs.BOR_NODAL_PASSIVITY_GUARD = False
    return dict(n_inc=int(len(res["inc"])),
                max_energy=float(np.max(e)) if e.size else None,
                min_energy=float(np.min(e)) if e.size else None,
                refused_when_armed=refused,
                warned=[str(x.message)[:60] for x in w],
                n_warn=len(w))


def broken_family():
    """The scoping's five-layer ring stack over Rbig = 1 .. 16 wavelengths."""
    rows = []
    m, N, k0 = 1, 140, 2.0
    lam = 2.0 * np.pi / k0
    spec = [(_uni(2.0), None),
            (_ring(0.8, 2.0, 6.0), 0.5),
            (_uni(2.0), 0.3),
            (_ring(1.2, 6.0, 2.0), 0.4),
            (_uni(2.0), None)]
    for nl in (1, 2, 4, 6, 8, 10, 12, 14, 16):
        Rbig = nl * lam
        row = dict(family="five_layer_ring", Rbig_over_lambda=float(nl),
                   Rbig=float(Rbig), N=N, m=m, k0=k0)
        for basis in ("nodal", "staggered"):
            try:
                row[basis] = _solve(basis, m, Rbig, N, k0, spec)
            except Exception as exc:                  # noqa: BLE001
                row[basis] = dict(error="%s: %s" % (type(exc).__name__, exc))
        rows.append(row)
    return rows


def refinement_sweep():
    """The same family at Rbig = 12 lambda over N, where the scoping reached
    max(R+T) = 966.7."""
    rows = []
    m, k0 = 1, 2.0
    Rbig = 12.0 * 2.0 * np.pi / k0
    spec = [(_uni(2.0), None),
            (_ring(0.8, 2.0, 6.0), 0.5),
            (_uni(2.0), 0.3),
            (_ring(1.2, 6.0, 2.0), 0.4),
            (_uni(2.0), None)]
    for N in (140, 200, 300):
        row = dict(family="refinement", N=N, Rbig=float(Rbig), m=m, k0=k0)
        for basis in ("nodal", "staggered"):
            try:
                row[basis] = _solve(basis, m, Rbig, N, k0, spec)
            except Exception as exc:                  # noqa: BLE001
                row[basis] = dict(error="%s: %s" % (type(exc).__name__, exc))
        rows.append(row)
    return rows


def shipped_fixtures():
    """Every nodal fixture the SHIPPED test suite builds, so the bar cannot be
    derived in ignorance of the behaviour the library already pins.

    ``test_bor_solve::test_structured_stack_energy_floor_nodal`` is the binding
    one: it deliberately runs the nodal basis on a small cell and asserts the
    documented ~1-4% floor.
    """
    rows = []
    cases = [
        # (label, m, Rbig, N, k0, spec)
        ("test_structured_stack_energy_floor_nodal", 1, 4.0, 200, 2.0,
         [(_uni(2.0), None), (_ring(0.8, 2.0, 6.0), 0.5), (_uni(2.0), None)]),
        ("uniform_small_cell", 1, 4.0, 120, 2.0,
         [(_uni(2.0), None), (_uni(2.5), 0.5), (_uni(2.0), None)]),
        ("uniform_2lambda", 1, 2.0 * np.pi, 120, 2.0,
         [(_uni(2.0), None), (_uni(2.5), 0.5), (_uni(2.0), None)]),
        ("ring_2lambda", 2, 2.0 * np.pi, 160, 2.0,
         [(_uni(2.0), None), (_ring(1.0, 2.0, 4.0), 0.4), (_uni(2.0), None)]),
        ("uniform_12lambda", 1, 12.0 * np.pi, 140, 2.0,
         [(_uni(2.0), None), (_uni(2.5), 0.5), (_uni(2.0), None)]),
        ("m0_small", 0, 4.0, 140, 2.0,
         [(_uni(2.0), None), (_ring(0.8, 2.0, 6.0), 0.5), (_uni(2.0), None)]),
    ]
    for label, m, Rbig, N, k0, spec in cases:
        row = dict(family="shipped", label=label, m=m, Rbig=float(Rbig), N=N,
                   k0=k0, Rbig_over_lambda=float(Rbig * k0 / (2.0 * np.pi)))
        for basis in ("nodal", "staggered"):
            try:
                row[basis] = _solve(basis, m, Rbig, N, k0, spec)
            except Exception as exc:                  # noqa: BLE001
                row[basis] = dict(error="%s: %s" % (type(exc).__name__, exc))
        rows.append(row)
    return rows


def floor_census():
    """The DOCUMENTED-FLOOR population: every SMALL-cell nodal configuration
    the shipped suite's own gate family covers, swept over ``m``, ``N`` and
    the cell radius up to 2 vacuum wavelengths.

    This is the population the refusal must NOT touch.  The nodal basis's
    documented accuracy floor is "~1-4%" and one shipped gate
    (``test_structured_stack_energy_floor_nodal``) pins it at ``< 5%``; a bar
    derived from the broken family alone would refuse it, which is why the
    census is measured FIRST and the bar is derived from the GAP.
    """
    rows = []
    for m in (0, 1, 2):
        for N in (120, 200):
            for rl in (0.5, 1.0, 1.5, 2.0):
                Rbig = rl * 2.0 * np.pi / 2.0
                for label, spec in (
                        ("uniform", [(_uni(2.0), None), (_uni(2.5), 0.5),
                                     (_uni(2.0), None)]),
                        ("ring", [(_uni(2.0), None),
                                  (_ring(0.8, 2.0, 6.0), 0.5),
                                  (_uni(2.0), None)]),
                ):
                    row = dict(family="floor", label=label, m=m, N=N,
                               Rbig_over_lambda=float(rl), Rbig=float(Rbig),
                               k0=2.0)
                    for basis in ("nodal", "staggered"):
                        try:
                            row[basis] = _solve(basis, m, Rbig, N, 2.0, spec)
                        except Exception as exc:      # noqa: BLE001
                            row[basis] = dict(
                                error="%s: %s" % (type(exc).__name__, exc))
                    rows.append(row)
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    print("TREE", C.pin_tree())
    print("KERNEL", C.kernel_tag())
    payload = dict(broken=broken_family(), refinement=refinement_sweep(),
                   shipped=shipped_fixtures(), floor=floor_census())

    def vals(rows, basis):
        return [r[basis]["max_energy"] for r in rows
                if isinstance(r.get(basis), dict)
                and r[basis].get("max_energy") is not None]

    allrows = (payload["broken"] + payload["refinement"]
               + payload["shipped"] + payload["floor"])
    nod_broken = vals(payload["broken"] + payload["refinement"], "nodal")
    nod_shipped = vals(payload["shipped"] + payload["floor"], "nodal")
    stag = vals(allrows, "staggered")
    payload["summary"] = dict(
        staggered_max=max(stag) if stag else None,
        staggered_worst_violation=(max(abs(v - 1.0) for v in stag)
                                   if stag else None),
        nodal_broken_min=min(nod_broken) if nod_broken else None,
        nodal_broken_max=max(nod_broken) if nod_broken else None,
        nodal_shipped_max=max(nod_shipped) if nod_shipped else None,
        nodal_floor_violations_top5=sorted(
            (round(v - 1.0, 6) for v in nod_shipped), reverse=True)[:5],
        n_floor_rows=len(nod_shipped),
        n_rows=len(allrows))
    print("SUMMARY", payload["summary"])
    for r in payload["broken"]:
        print("  Rbig/lam %5.1f  nodal %-14s staggered %-18s warned=%s"
              % (r["Rbig_over_lambda"],
                 "%.4g" % r["nodal"].get("max_energy", float("nan"))
                 if isinstance(r.get("nodal"), dict) else "ERR",
                 "%.13g" % r["staggered"].get("max_energy", float("nan"))
                 if isinstance(r.get("staggered"), dict) else "ERR",
                 r["nodal"].get("n_warn") if isinstance(r.get("nodal"), dict)
                 else "?"))
    for r in payload["shipped"]:
        print("  SHIPPED %-45s nodal %-10s staggered %s"
              % (r["label"],
                 "%.6g" % r["nodal"].get("max_energy", float("nan"))
                 if isinstance(r.get("nodal"), dict) else "ERR",
                 "%.13g" % r["staggered"].get("max_energy", float("nan"))
                 if isinstance(r.get("staggered"), dict) else "ERR"))
    C.dump("s3_nodal_%s.json" % (tag,), payload)


if __name__ == "__main__":
    main()
