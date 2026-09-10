"""B4 -- the bit-identity census for the CONSOLIDATION.

The round-2 change has two halves, and they carry different obligations.

*The RCWA half is a REFACTOR.*  ``rcwa/_core._sqrt_decay`` gained two
parameters (``xp``, ``band``) whose defaults reproduce the round-1 body
exactly.  Its obligation is BIT-IDENTITY on every input: the round-1 body,
transcribed here, and the shipped shared body must return the same bits over an
engineered corner set and over every RCWA surface.

*The PMM half is a FIX.*  Five private copies carrying the exact-zero pin were
deleted.  Its obligation is the two-sided one round 1 stated for itself:
LOSSY solves bit-identical (the sign was physics, so nothing may move);
off-coincidence LOSSLESS solves moving only at rounding level; on-coincidence
solves moving, because that is the repair, and moving TOWARDS the independent
oracle.

This probe measures both over >= 40 PMM fixtures spanning every caller of the
shared function -- the hybrid single-cell and stack paths, the tensor entry,
the pure staggered engine (vertical, slanted, magnetic, out-of-plane,
per-layer), the JAX twins against their NumPy siblings, and the RCWA surfaces
that share the function.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b4_census.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
NS, NS_OFF, HOST = 1.5, 1.63, 2.25
WEAK = HOST * (1.0 + 1e-6)
S6 = 6
LAYOUT6 = np.zeros((S6, S6), dtype=np.int64)
LAYOUT6[2:4, 2:4] = 1


def cell6(pillar, host=HOST, eps_im=0.0):
    c = np.full((S6, S6), host + 1j * eps_im, dtype=complex)
    c[2:4, 2:4] = pillar + 1j * eps_im
    return c


# ---------------------------------------------------------------- part 1: the
# RCWA half is a refactor: the round-1 body against the shipped shared body.
def round1_body(x):
    """The round-1 shipped source, re-typed.  ``_CUT_BAND_REL`` is read from
    the live module so this is a comparison of BODIES, not of constants."""
    from lumenairy.backend.array import array_namespace
    from lumenairy.elements.rcwa._core import _CUT_BAND_REL
    xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    on_cut = xp.abs(r.real) <= _CUT_BAND_REL * scale
    return xp.where(on_cut & (r.imag < 0), xp.conj(r), r)


def refactor_identity():
    from lumenairy.elements.rcwa import _core as rc
    rng = np.random.default_rng(20260911)
    mags = 10.0 ** rng.uniform(-15.0, 6.0, 4000)
    ang = rng.uniform(-np.pi, np.pi, 4000)
    vals = list(mags * np.exp(1j * ang))
    vals += [0 + 0j, 0 - 0j, 5e-324 + 0j, -1e-300 + 0j,
             -2.25 + 0j, -2.25 - 0j, -2.25 - 2.911e-15j,
             2.25 - 2.911e-15j, complex("nan"), 1e300 + 1e300j]
    z = np.array(vals, dtype=complex)
    a, b = np.asarray(rc._sqrt_decay(z)), np.asarray(round1_body(z))
    same = np.array_equal(np.nan_to_num(a, nan=-7.0),
                          np.nan_to_num(b, nan=-7.0))
    # and per-array, since the band is relative to the ARRAY
    per = []
    for n in (1, 2, 7, 64, 243):
        zz = z[:n]
        per.append(bool(np.array_equal(
            np.nan_to_num(np.asarray(rc._sqrt_decay(zz)), nan=-7.0),
            np.nan_to_num(np.asarray(round1_body(zz)), nan=-7.0))))
    return dict(n=int(z.size), bit_identical=bool(same),
                per_array_sizes=per,
                max_abs_diff=float(np.max(np.abs(
                    np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0)))))


# ------------------------------------------------------- part 2: the fixtures
def pmm_fixtures():
    from lumenairy.elements.pmm import (
        PMM2DStackHybrid,
        PMM2DStackPure,
        pmm_efficiency_1d,
        pmm_efficiency_2d_cell,
        pmm_efficiency_2d_staggered,
        pmm_jones_1d,
        pmm_jones_2d,
        pmm_jones_2d_staggered,
    )
    out = []

    def add(name, cls, kind, fn):
        out.append((name, cls, kind, fn))

    # -- hybrid single cell: lossless on/off the coincidence, lossy, oblique
    for tag, pillar in (("weak", WEAK), ("strong", 6.0)):
        for cls, nsub in (("coinc", NS), ("off", NS_OFF)):
            add("cell_%s_%s" % (tag, cls), cls, "eff",
                lambda p=pillar, n=nsub: pmm_efficiency_2d_cell(
                    PX, PX, cell6(p), n, 1.0, D, WL, degree=7, n_orders=4,
                    symmetry=False))
            add("cellsym_%s_%s" % (tag, cls), cls, "eff",
                lambda p=pillar, n=nsub: pmm_efficiency_2d_cell(
                    PX, PX, cell6(p), n, 1.0, D, WL, degree=7, n_orders=4,
                    symmetry=True))
    for ei in (1e-1, 1e-3, 1e-6):
        add("cell_lossy_%g" % ei, "lossy", "eff",
            lambda e=ei: pmm_efficiency_2d_cell(
                PX, PX, cell6(6.0, eps_im=e), NS, 1.0, D, WL, degree=7,
                n_orders=4, symmetry=False))
    add("cell_metal", "lossy", "eff",
        lambda: pmm_efficiency_2d_cell(
            PX, PX, cell6(-10.0 + 1.0j), NS, 1.0, D, WL, degree=7, n_orders=4,
            symmetry=False))
    add("cell_oblique_coinc", "coinc", "eff",
        lambda: pmm_efficiency_2d_cell(
            PX, PX, cell6(WEAK), NS, 1.0, D, WL, degree=7, n_orders=3,
            theta=0.3, symmetry=False))
    add("cell_conical_coinc", "coinc", "eff",
        lambda: pmm_efficiency_2d_cell(
            PX, PX, cell6(WEAK), NS, 1.0, D, WL, degree=7, n_orders=3,
            theta=0.2, phi=0.7, symmetry=False))

    # -- hybrid tensor entry
    add("jones2d_coinc", "coinc", "jones",
        lambda: pmm_jones_2d(PX, PX, F.tensor_cell(S=16), NS, 1.0, D, WL,
                             degree=7, n_orders=2))
    add("jones2d_off", "off", "jones",
        lambda: pmm_jones_2d(PX, PX, F.tensor_cell(S=16), NS_OFF, 1.0, D, WL,
                             degree=7, n_orders=2))
    add("jones2d_lossy", "lossy", "jones",
        lambda: pmm_jones_2d(PX, PX, F.tensor_cell(S=16, eps_im=1e-2), NS,
                             1.0, D, WL, degree=7, n_orders=2))

    # -- hybrid stack: uniform spacer, per-layer, slanted, lossy
    def hyb(layers, nsub, M=3, sym=False, theta=0.0):
        st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                              degree=7, n_orders=M, symmetry=sym)
        for kw in layers:
            st.add_layer(**kw)
        return st.set_source(WL, theta=theta).solve()

    SP = dict(thickness=0.1e-6, eps=HOST)
    PAT = dict(thickness=D, eps_cell=cell6(WEAK))
    PATS = dict(thickness=D, eps_cell=cell6(6.0))
    add("stack_spacer_coinc", "coinc", "jones",
        lambda: hyb([SP, PAT, SP], NS_OFF))
    add("stack_spacer_both", "coinc", "jones", lambda: hyb([SP, PAT, SP], NS))
    add("stack_nospacer_off", "off", "jones", lambda: hyb([PAT], NS_OFF))
    add("stack_spacer_detuned", "off", "jones",
        lambda: hyb([dict(thickness=0.1e-6, eps=HOST * 1.01), PAT,
                     dict(thickness=0.1e-6, eps=HOST * 1.01)], NS_OFF))
    add("stack_strong_off", "off", "jones", lambda: hyb([SP, PATS, SP],
                                                        NS_OFF))
    add("stack_perlayer", "off", "jones",
        lambda: hyb([dict(thickness=0.08e-6, eps_cell=cell6(4.0)),
                     dict(thickness=0.12e-6, eps_cell=cell6(6.0)),
                     dict(thickness=0.06e-6, eps=2.9)], NS_OFF))
    add("stack_slant", "off", "jones",
        lambda: hyb([dict(thickness=D, eps_cell=cell6(6.0), slant=0.15)],
                    NS_OFF))
    add("stack_slant_coinc", "coinc", "jones",
        lambda: hyb([SP, dict(thickness=D, eps_cell=cell6(WEAK), slant=0.15),
                     SP], NS))
    add("stack_lossy", "lossy", "jones",
        lambda: hyb([dict(thickness=0.1e-6, eps=HOST + 1e-2j),
                     dict(thickness=D, eps_cell=cell6(6.0, eps_im=1e-2))],
                    NS))
    add("stack_oblique_coinc", "coinc", "jones",
        lambda: hyb([SP, PAT, SP], NS, theta=0.25))
    add("stack_tensor", "off", "jones",
        lambda: hyb([dict(thickness=D,
                          eps_tensor_cell=F.tensor_cell(S=16))], NS_OFF, M=2))

    # -- PURE STAGGERED: vertical, slanted, magnetic, out-of-plane, per-layer
    def sc(p, ei=0.0):
        return F.stag_cell(host=HOST, pillar=p, eps_im=ei)

    for tag, p in (("weak", WEAK), ("strong", 6.0)):
        for cls, nsub in (("coinc", NS), ("off", NS_OFF)):
            add("stag_%s_%s" % (tag, cls), cls, "eff",
                lambda pp=p, n=nsub: pmm_efficiency_2d_staggered(
                    PX, PX, sc(pp), n, 1.0, D, WL, degree=6, n_orders=4))
    add("stag_lossy", "lossy", "eff",
        lambda: pmm_efficiency_2d_staggered(
            PX, PX, sc(6.0, 1e-2), NS, 1.0, D, WL, degree=6, n_orders=4))
    add("stag_slant", "off", "eff",
        lambda: pmm_efficiency_2d_staggered(
            PX, PX, sc(6.0), NS_OFF, 1.0, D, WL, degree=6, n_orders=4,
            slant=0.1))
    add("stagjones_tensor_coinc", "coinc", "jones",
        lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(), NS, 1.0,
                                       D, WL, degree=6, n_orders=3))
    add("stagjones_oop_coinc", "coinc", "jones",
        lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(oop=0.3),
                                       NS, 1.0, D, WL, degree=6, n_orders=3))
    add("stagjones_oop_lossy", "lossy", "jones",
        lambda: pmm_jones_2d_staggered(
            PX, PX, F.stag_tensor_cell(oop=0.3, eps_im=1e-2), NS, 1.0, D, WL,
            degree=6, n_orders=3))
    mu = np.zeros((4, 4, 3, 3), complex)
    for i in range(3):
        mu[:, :, i, i] = 1.0
    mu[1:3, 1:3, 0, 0] = 1.4
    mu[1:3, 1:3, 1, 1] = 1.4
    add("stagjones_magnetic_coinc", "coinc", "jones",
        lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(), NS, 1.0,
                                       D, WL, degree=6, n_orders=3,
                                       mu_cell=mu))

    def pure(layers, nsub):
        st = PMM2DStackPure(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                            degree=6, n_orders=4)
        for kw in layers:
            st.add_layer(**kw)
        return st.set_source(WL, theta=0.0).solve()

    add("pure_spacer_coinc", "coinc", "jones",
        lambda: pure([dict(thickness=0.1e-6, eps=HOST),
                      dict(thickness=D, eps_cell=sc(WEAK)),
                      dict(thickness=0.1e-6, eps=HOST)], NS))
    add("pure_off", "off", "jones",
        lambda: pure([dict(thickness=D, eps_cell=sc(6.0))], NS_OFF))
    add("pure_perlayer", "off", "jones",
        lambda: pure([dict(thickness=0.08e-6, eps_cell=sc(4.0)),
                      dict(thickness=0.12e-6, eps_cell=sc(6.0))], NS_OFF))
    add("pure_lossy", "lossy", "jones",
        lambda: pure([dict(thickness=D, eps_cell=sc(6.0, 1e-2))], NS))

    # -- the 1-D PMM (its forward branch is _forward_branch_flip; a control
    #    that the consolidation leaves alone)
    add("oned_te_coinc", "coinc", "eff",
        lambda: pmm_efficiency_1d(1.0e-6, 2.1 ** 0.5, 1.5, 1.5, 1.0, 0.4e-6,
                                  0.5, WL, degree=12, polarization="te"))
    add("oned_tm_lossy", "lossy", "eff",
        lambda: pmm_efficiency_1d(1.0e-6, (2.1 + 0.05j) ** 0.5, 1.5, 1.5, 1.0,
                                  0.4e-6, 0.5, WL, degree=12,
                                  polarization="tm"))
    add("onedjones_conical", "off", "jones",
        lambda: pmm_jones_1d(1.0e-6, 2.1 ** 0.5, 1.0, 1.5, 1.0, 0.4e-6, 0.5,
                             WL, degree=12, theta=0.25, phi=0.4))
    return out


def rcwa_fixtures():
    from lumenairy.elements.rcwa import (
        RCWAStack,
        rcwa_efficiency_1d,
        rcwa_efficiency_2d,
        rcwa_jones_2d,
    )
    out = []

    def add(name, cls, kind, fn):
        out.append((name, cls, kind, fn))

    add("rcwa_jones2d_coinc", "coinc", "jones",
        lambda: rcwa_jones_2d(PX, PX, F.tensor_cell(S=32), NS, 1.0, D, WL,
                              n_orders_x=4, n_orders_y=4, symmetry=False))
    add("rcwa_jones2d_off", "off", "jones",
        lambda: rcwa_jones_2d(PX, PX, F.tensor_cell(S=32), NS_OFF, 1.0, D, WL,
                              n_orders_x=4, n_orders_y=4, symmetry=False))
    add("rcwa_eff2d", "off", "eff",
        lambda: rcwa_efficiency_2d(PX, PX, F.pillar_cell(S=32), NS_OFF, 1.0,
                                   D, WL, n_orders_x=5, n_orders_y=5,
                                   polarization="te"))
    add("rcwa_eff1d_te", "coinc", "eff",
        lambda: rcwa_efficiency_1d(1.0e-6, 2.1 ** 0.5, 1.5, 1.5, 1.0, 0.4e-6,
                                   0.5, WL, polarization="te", n_orders=15))
    add("rcwa_eff1d_lossy", "lossy", "eff",
        lambda: rcwa_efficiency_1d(1.0e-6, (2.1 + 0.05j) ** 0.5, 1.5, 1.5,
                                   1.0, 0.4e-6, 0.5, WL, polarization="tm",
                                   n_orders=15))

    def stack():
        st = RCWAStack(PX, period_y=PX, n_substrate=NS,
                       n_superstrate=1.0, n_orders=3, n_orders_y=3)
        st.add_layer(0.1e-6, eps=HOST)
        st.add_layer(D, eps_cell=F.pillar_cell(S=32, pillar=WEAK))
        st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=0.0).solve().efficiencies()

    add("rcwa_stack_spacer", "coinc", "jones", stack)
    return out


def run(fn, kind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = fn()
        except Exception as exc:
            return dict(raised=type(exc).__name__, message=repr(exc)[:180])
    return dict(raised=None, rt=F.rt_vec(res).tolist(),
                closure=(F.closure_jones(res) if kind == "jones"
                         else F.closure_eff(res)))


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b4.json"
    ref = refactor_identity()
    print("RCWA refactor bit-identity vs the round-1 body: %s "
          "(n=%d, per-array %s, max|diff| %.3e)"
          % (ref["bit_identical"], ref["n"], ref["per_array_sizes"],
             ref["max_abs_diff"]))

    rows = {}
    fx = [("pmm", n, c, k, f) for n, c, k, f in pmm_fixtures()]
    fx += [("rcwa", n, c, k, f) for n, c, k, f in rcwa_fixtures()]
    print("%-28s %-6s %-7s %-13s %-13s %s"
          % ("fixture", "engine", "class", "PRE closure", "POST closure",
             "motion"))
    for engine, name, cls, kind, fn in fx:
        with F.PreSqrtDecayPMM():
            pre = run(fn, kind)
        with F.PostSqrtDecayPMM():
            post = run(fn, kind)
        mv = (F.motion(pre.get("rt"), post.get("rt"))
              if pre.get("rt") and post.get("rt") else None)
        rows[name] = dict(engine=engine, cls=cls, pre=pre, post=post,
                          motion=mv)
        print("%-28s %-6s %-7s %-13s %-13s %s"
              % (name, engine, cls,
                 "%+.4e" % pre["closure"] if pre["raised"] is None
                 else pre["raised"],
                 "%+.4e" % post["closure"] if post["raised"] is None
                 else post["raised"],
                 "%.3e" % mv if mv is not None else "-"))

    ok = {c: [] for c in ("lossy", "coinc", "off")}
    for name, r in rows.items():
        if r["motion"] is not None and r["cls"] in ok:
            ok[r["cls"]].append((name, r["motion"]))
    print("\nby class:")
    for c, lst in ok.items():
        if not lst:
            continue
        w = max(lst, key=lambda t: t[1])
        n_id = sum(1 for _n, m in lst if m == 0.0)
        print("  %-6s n=%-3d bit-identical %-3d worst %.4e (%s)"
              % (c, len(lst), n_id, w[1], w[0]))

    F.dump(out, dict(rows=rows, refactor=ref,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
