"""V5 -- blast radius: what moved, by how much, and where NOTHING moved.

The A/B is taken WITHIN ONE INTERPRETER (``V.PreSqrtDecay`` reinstates the
``48c8747`` body in every module that binds ``_sqrt_decay``), for two reasons.
First, it isolates the change: the merge ``48c8747 -> 75a0c81`` also carries the
unrelated ``pmm/stack.py`` sliver round-3 work, so a cross-tree diff of a PMM
entry point would attribute that change to this one.  Second, it removes the
interpreter and the BLAS reduction order from the comparison entirely -- the two
arms run in the same process, microseconds apart.  ``v1_threads.py`` already
showed the transcribed body reproduces the real PRE tree to every digit
(``dR = 2.812111e-10``, closure ``-3.1995e-03`` / ``-3.3813e-04`` on Windows at
one thread, identical to the ``48c8747`` worktree).

Each surface is classified:

  a  off-coincidence lossless   -- expected to move at rounding level
  b  ON-coincidence lossless    -- the fix; expected to move, possibly a lot
  c  lossy                      -- expected BIT-IDENTICAL
  d  does not reach the patched ``_sqrt_decay`` at all (PMM's own copies)

and anything that moves against its class is a defect.

Usage: python v5_scope.py <out.json>
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

P, WL, DEPTH = V._P, V._WL, V._DEPTH


def _h(arrs):
    m = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _flat(res):
    """Every numeric array a result carries, in a stable order.

    ``RCWAStack.solve`` returns an ``RCWAResult`` -- not a tuple -- so its
    ``efficiencies()`` and Jones accessors are read explicitly.
    """
    if hasattr(res, "efficiencies"):
        cand = list(res.efficiencies())
        for n in ("jones_reflection", "jones_transmission"):
            try:
                cand.append(getattr(res, n))
            except Exception:                                 # pragma: no cover
                pass
        try:
            cand.append(res.absorptance())
        except Exception:                                     # pragma: no cover
            pass
    else:
        cand = list(res)
    out = []
    for x in cand:
        if x is None:
            continue
        a = np.asarray(x)
        if a.dtype.kind in "fciu" and a.size:
            out.append(a)
    return out


# --------------------------------------------------------------- the surfaces
def surfaces():
    from lumenairy.elements.berreman import berreman_jones_1d
    from lumenairy.elements.pmm import (
        PMM2DStackPure,
        pmm_efficiency_1d,
        pmm_efficiency_2d,
        pmm_efficiency_2d_staggered,
        pmm_jones_1d,
        pmm_jones_2d,
        pmm_jones_2d_staggered,
    )
    from lumenairy.elements.rcwa import (
        RCWAStack,
        rcwa_efficiency_1d,
        rcwa_efficiency_2d,
        rcwa_jones_1d,
        rcwa_jones_2d,
    )
    S = []

    def add(name, cls, fn):
        S.append((name, cls, fn))

    # ---- rcwa_jones_2d : anisotropic tensor, on and off the coincidence ----
    for tw in (0.0, 0.4, 0.7, 1.1):
        add("jones2d_coinc_tw%.1f" % tw, "b",
            lambda tw=tw: rcwa_jones_2d(P, P, V.uniaxial_cell(twist=tw), 1.5,
                                        1.0, DEPTH, WL, n_orders_x=5,
                                        n_orders_y=5))
        add("jones2d_offcoinc_tw%.1f" % tw, "a",
            lambda tw=tw: rcwa_jones_2d(P, P, V.uniaxial_cell(twist=tw), 1.63,
                                        1.0, DEPTH, WL, n_orders_x=5,
                                        n_orders_y=5))
    for sym in (True, False):
        add("jones2d_fold_%s" % sym, "b",
            lambda sym=sym: rcwa_jones_2d(P, P, V.uniaxial_cell(), 1.5, 1.0,
                                          DEPTH, WL, n_orders_x=5,
                                          n_orders_y=5, symmetry=sym))
    for th, ph in ((0.3, 0.0), (0.2, 0.7)):
        add("jones2d_obl_t%.1f_p%.1f" % (th, ph), "b",
            lambda th=th, ph=ph: rcwa_jones_2d(
                P, P, V.uniaxial_cell(), 1.5, 1.0, DEPTH, WL, theta=th,
                phi=ph, n_orders_x=4, n_orders_y=4))
    # ---- OOP / full-3x3: an out-of-plane-tilted director (eps_xz, eps_yz) --
    def _oop_cell(tilt=0.6):
        tc = V.uniaxial_cell(twist=0.0)
        no2, ne2 = 2.25, 2.89
        c, s = np.cos(tilt), np.sin(tilt)
        x = (np.arange(48) + 0.5) / 48 - 0.5
        m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
        tc[m, 0, 0] = ne2 * c * c + no2 * s * s
        tc[m, 2, 2] = ne2 * s * s + no2 * c * c
        tc[m, 0, 2] = tc[m, 2, 0] = (ne2 - no2) * c * s
        return tc
    add("jones2d_OOP_coinc", "b",
        lambda: rcwa_jones_2d(P, P, _oop_cell(), 1.5, 1.0, DEPTH, WL,
                              n_orders_x=4, n_orders_y=4))
    add("jones2d_OOP_offcoinc", "a",
        lambda: rcwa_jones_2d(P, P, _oop_cell(), 1.63, 1.0, DEPTH, WL,
                              n_orders_x=4, n_orders_y=4))
    # ---- lossy tensor cells: must be bit-identical -------------------------
    for im in (1e-2, 1e-4, 1e-6):
        add("jones2d_lossy_im%.0e" % im, "c",
            lambda im=im: rcwa_jones_2d(P, P, V.uniaxial_cell(eps_im=im), 1.5,
                                        1.0, DEPTH, WL, n_orders_x=5,
                                        n_orders_y=5))
    add("jones2d_metal_block", "c",
        lambda: rcwa_jones_2d(P, P, _metal_cell(), 1.5, 1.0, DEPTH, WL,
                              n_orders_x=4, n_orders_y=4))

    # ---- rcwa_efficiency_2d (scalar path) ----------------------------------
    add("eff2d_block4_coinc", "a",
        lambda: rcwa_efficiency_2d(P, P, V.scalar_cell(blk=4.0), 1.5, 1.0,
                                   DEPTH, WL, n_orders_x=5, n_orders_y=5))
    add("eff2d_weakmod_coinc", "b",
        lambda: rcwa_efficiency_2d(P, P, V.scalar_cell(blk=2.25 + 1e-6), 1.5,
                                   1.0, DEPTH, WL, n_orders_x=5, n_orders_y=5))
    add("eff2d_offcoinc", "a",
        lambda: rcwa_efficiency_2d(P, P, V.scalar_cell(blk=4.0), 1.63, 1.0,
                                   DEPTH, WL, n_orders_x=5, n_orders_y=5))
    add("eff2d_lossy", "c",
        lambda: rcwa_efficiency_2d(P, P, V.scalar_cell(blk=4.0, eps_im=1e-2),
                                   1.5, 1.0, DEPTH, WL, n_orders_x=5,
                                   n_orders_y=5))

    # ---- rcwa_efficiency_1d / rcwa_jones_1d --------------------------------
    for pol in ("te", "tm"):
        for duty in (0.02, 0.5):
            add("eff1d_%s_duty%.2f_coincgroove" % (pol, duty), "a",
                lambda pol=pol, duty=duty: rcwa_efficiency_1d(
                    P, 2.1, 1.5, 1.5, 1.0, DEPTH, duty, WL,
                    polarization=pol, n_orders=15))
        add("eff1d_%s_offcoinc" % pol, "a",
            lambda pol=pol: rcwa_efficiency_1d(P, 2.1, 1.35, 1.63, 1.0, DEPTH,
                                               0.5, WL, polarization=pol,
                                               n_orders=15))
    add("eff1d_oblique_te", "a",
        lambda: rcwa_efficiency_1d(P, 2.1, 1.5, 1.5, 1.0, DEPTH, 0.5, WL,
                                   theta=0.35, polarization="te",
                                   n_orders=15))
    add("eff1d_lossy_ridge", "c",
        lambda: rcwa_efficiency_1d(P, 2.1 + 0.05j, 1.5, 1.5, 1.0, DEPTH, 0.5,
                                   WL, polarization="tm", n_orders=15))
    add("jones1d_conical", "a",
        lambda: rcwa_jones_1d(P, np.array(4.41 + 0j), np.array(2.25 + 0j),
                              1.5, 1.0, DEPTH, 0.5, WL, theta=0.3,
                              n_orders=15))
    add("jones1d_conical_lossy", "c",
        lambda: rcwa_jones_1d(P, np.array(4.41 + 0.1j), np.array(2.25 + 0j),
                              1.5, 1.0, DEPTH, 0.5, WL, theta=0.3,
                              n_orders=15))

    # ---- RCWAStack ---------------------------------------------------------
    def _stack(n_sub, tensor=False, sym="auto"):
        st = RCWAStack(period=P, period_y=P, n_superstrate=1.0,
                       n_substrate=n_sub, n_orders=4, n_orders_y=4)
        st.add_layer(0.05e-6, eps=2.25)
        if tensor:
            st.add_layer(0.12e-6, eps_tensor_cell=V.uniaxial_cell(S=32))
        else:
            st.add_layer(0.12e-6, eps_cell=V.scalar_cell(S=32, blk=4.0))
        st.add_layer(0.06e-6, eps=2.25)
        return st.set_source(WL).solve(symmetry=sym)
    add("stack_iso_coinc", "a", lambda: _stack(1.5))
    add("stack_iso_offcoinc", "a", lambda: _stack(1.63))
    add("stack_tensor_coinc", "b", lambda: _stack(1.5, tensor=True))
    add("stack_tensor_coinc_full", "b",
        lambda: _stack(1.5, tensor=True, sym=False))
    add("stack_tensor_offcoinc", "a", lambda: _stack(1.63, tensor=True))

    # ---- Berreman (shares the patched _sqrt_decay) -------------------------
    def _berr(n_sub, tensor=True, lossy=False):
        eps = np.diag([2.89 + (0.1j if lossy else 0.0), 2.25, 2.25]) \
            if tensor else (2.25 + (0.1j if lossy else 0.0))
        return berreman_jones_1d([(eps, 0.2e-6), (2.25, 0.1e-6)], n_sub, 1.0,
                                 WL, angle=0.3)
    add("berreman_coinc", "b", lambda: _berr(1.5))
    add("berreman_offcoinc", "a", lambda: _berr(1.63))
    add("berreman_lossy", "c", lambda: _berr(1.5, lossy=True))
    add("berreman_iso", "a", lambda: _berr(1.5, tensor=False))

    # ---- PMM: its OWN _sqrt_decay copies, NOT the patched one --------------
    add("pmm_efficiency_2d_coinc", "d",
        lambda: pmm_efficiency_2d(P, P, 4.0, 2.25, (0.125e-6, 0.375e-6),
                                  (0.125e-6, 0.375e-6),
                                  1.5, 1.0, DEPTH, WL, degree=5, n_orders=5))
    add("pmm_efficiency_2d_weakmod", "d",
        lambda: pmm_efficiency_2d(P, P, 2.25 + 1e-6, 2.25, (0.125e-6, 0.375e-6),
                                  (0.125e-6, 0.375e-6), 1.5, 1.0, DEPTH, WL,
                                  degree=5, n_orders=5))
    # pmm_jones_2d's LAYER goes through rcwa._core._layer_eigenmodes_tensor,
    # i.e. through the PATCHED _sqrt_decay -- so it is class (b), not (d).
    add("pmm_jones_2d_coinc", "b",
        lambda: pmm_jones_2d(P, P, V.uniaxial_cell(S=32), 1.5, 1.0, DEPTH, WL,
                             degree=5, n_orders=5))
    add("pmm_jones_2d_staggered_coinc", "e",
        lambda: pmm_jones_2d_staggered(P, P, V.scalar_cell(S=8, blk=4.0),
                                       1.5, 1.0, DEPTH, WL, degree=3,
                                       n_orders=2))
    add("pmm_eff_2d_staggered_coinc", "e",
        lambda: pmm_efficiency_2d_staggered(P, P, V.scalar_cell(S=8, blk=4.0),
                                            1.5, 1.0, DEPTH, WL, degree=3,
                                            n_orders=2))
    add("pmm_efficiency_1d_coinc", "d",
        lambda: pmm_efficiency_1d(P, 2.1, 1.5, 1.5, 1.0, DEPTH, 0.5, WL,
                                  degree=12, polarization="te"))
    add("pmm_jones_1d_coinc", "d",
        lambda: pmm_jones_1d(P, np.diag([4.41 + 0j] * 3),
                             np.diag([2.25 + 0j] * 3), 1.5, 1.0, DEPTH, 0.5,
                             WL, theta=0.3, degree=12))

    def _pure():
        st = PMM2DStackPure(period_x=P, period_y=P, n_superstrate=1.0,
                            n_substrate=1.5, degree=3, n_orders=2)
        st.add_layer(0.12e-6, eps_cell=V.scalar_cell(S=8, blk=4.0))
        return st.set_source(WL).solve()
    add("pmm2d_stack_pure_coinc", "d", _pure)
    return S


def _metal_cell():
    tc = V.uniaxial_cell(twist=0.0)
    x = (np.arange(48) + 0.5) / 48 - 0.5
    m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
    for i in range(3):
        tc[m, i, i] = -20.0 + 1.5j
    tc[m, 0, 1] = tc[m, 1, 0] = 0.0
    return tc


def run_one(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _flat(fn())


def main():
    V.require_local_tree()
    out = sys.argv[1]
    rows = []
    for name, cls, fn in surfaces():
        row = dict(name=name, cls=cls)
        try:
            try:
                post = run_one(fn)
            except Exception as exc:
                row["post_raised"] = repr(exc)[:160]
                raise
            try:
                with V.PreSqrtDecay():
                    pre = run_one(fn)
            except Exception as exc:
                row["pre_raised"] = repr(exc)[:160]
                raise
            row["hash_post"] = _h(post)
            row["hash_pre"] = _h(pre)
            row["bit_identical"] = (row["hash_post"] == row["hash_pre"])
            mv = 0.0
            for a, b in zip(post, pre):
                a, b = np.asarray(a), np.asarray(b)
                if a.shape == b.shape:
                    mv = max(mv, float(np.max(np.abs(a - b))) if a.size else 0.0)
                else:
                    mv = float("inf")
            row["max_move"] = mv
        except Exception as exc:
            row["error"] = repr(exc)[:200]
        rows.append(row)
        print("%-34s cls=%s bitid=%-5s move=%s%s" % (
            name, cls, row.get("bit_identical", "-"),
            ("%.4e" % row["max_move"]) if "max_move" in row else "-",
            ("  PRE-ARM RAISED " if "pre_raised" in row else
             ("  POST-ARM RAISED " if "post_raised" in row else "")
             ) + (row.get("error", "")[:80] if "error" in row else "")))

    by = {}
    for r in rows:
        if "max_move" not in r:
            continue
        by.setdefault(r["cls"], []).append(r)
    summary = {}
    for k, v in sorted(by.items()):
        summary[k] = dict(
            n=len(v),
            n_bit_identical=sum(1 for r in v if r["bit_identical"]),
            max_move=max(r["max_move"] for r in v),
            min_nonzero_move=min([r["max_move"] for r in v
                                  if r["max_move"] > 0.0] or [0.0]),
            worst=max(v, key=lambda r: r["max_move"])["name"])
    print("\nBY CLASS")
    for k, s in summary.items():
        print("  %s: n=%-3s bit-identical=%-3s max_move=%.4e (%s)"
              % (k, s["n"], s["n_bit_identical"], s["max_move"], s["worst"]))
    V.dump(out, dict(rows=rows, summary=summary))


if __name__ == "__main__":
    main()
