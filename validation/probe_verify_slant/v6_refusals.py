"""V6 -- THE REFUSALS, and an attempt to sneak a slanted layer past each one.

Two halves, both required:

  * every documented refusal raises, with the right TYPE and a message that
    names what was refused (a refusal whose message does not identify the
    offending construction is a support burden, not a gate);
  * the ACCEPTED shapes do not raise AND are numerically right -- an
    over-broad refusal and a silently-wrong acceptance are the two failure
    modes, and only measuring both catches either.

The sneak attempts probe the boundary the gate actually draws: ``_slant_is_zero``
is an EXACT ``== 0.0`` test, so a slant of 1e-17 is a SLANTED layer to every
routing decision in the tree.
"""
import warnings

import numpy as np
from _lib import arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    _stag_parity_gauge,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL = 0.68e-6
PX = PY = 1.10e-6
DEP = 0.34e-6
NSUP, NSUB = 1.0, 1.5
T35 = float(np.tan(np.deg2rad(35.0)))
SCA = np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex)
TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
OOPC = np.zeros((2, 2, 3, 3), dtype=complex)
OOPC[:, :] = AIR
OOPC[0, 0] = TIL
MUC = np.zeros((2, 2, 3, 3), dtype=complex)
MUC[:, :] = np.eye(3)
MUC[0, 0] = np.diag([1.4, 1.2, 1.1])
CENTRO = np.zeros((2, 2, 3, 3), dtype=complex)
CENTRO[:, :] = AIR
CENTRO[0, 0] = CENTRO[1, 1] = TIL


def expect_raise(name, fn, exc, must_contain=()):
    try:
        fn()
    except exc as e:
        msg = str(e)
        missing = [s for s in must_contain if s.lower() not in msg.lower()]
        return dict(raised=type(e).__name__, ok=not missing,
                    missing=missing, msg=msg[:400])
    except Exception as e:                      # noqa: BLE001
        return dict(raised=type(e).__name__, ok=False,
                    missing=["WRONG EXCEPTION TYPE"], msg=str(e)[:400])
    return dict(raised=None, ok=False, missing=["DID NOT RAISE"], msg="")


def _st(M=5, nord=3):
    return PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_modes=M, n_orders=nord)


# ------------------------------------------------------------------ refusals
def r_mixed_patterned():
    s = _st()
    s.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    s.add_layer(DEP, eps_cell=SCA, slant=(0.2, 0.0))
    s.set_source(WL, theta=0.2)
    s.solve(jones=True)


def r_vertical_patterned_counts():
    s = _st()
    s.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    s.add_layer(DEP, eps_cell=SCA)
    s.set_source(WL, theta=0.2)
    s.solve(jones=True)


def r_mixed_above_pattern():
    s = _st()
    s.add_layer(0.15e-6, eps=2.1 + 0j, slant=(T35, 0.0))
    s.add_layer(0.15e-6, eps=2.1 + 0j)
    s.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    s.set_source(WL, theta=0.2)
    s.solve(jones=True)


def r_mu_uniform():
    _st().add_layer(DEP, eps_cell=SCA, mu=1.2, slant=(T35, 0.0))


def r_mu_cell():
    _st().add_layer(DEP, eps_cell=SCA, mu_cell=MUC, slant=(T35, 0.0))


def r_solver_mu():
    Granet2DTransverseE(PX, PY, 2, 2, 4, SCA, mu_cell=MUC[:, :, 0, 0],
                        slant=(T35, 0.0))


def r_retain_internal():
    s = _st()
    s.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    s.set_source(WL, theta=0.2)
    s.solve(jones=True, retain_internal=True)


def r_efficiency_entry():
    pmm_efficiency_2d_staggered(PX, PY, SCA, NSUB, NSUP, DEP, WL, degree=4,
                                n_orders=3, slant=(T35, 0.0))


def r_three_vector():
    _st().add_layer(DEP, eps_cell=SCA, slant=(0.1, 0.2, 0.3))


def r_nonfinite():
    _st().add_layer(DEP, eps_cell=SCA, slant=(np.nan, 0.0))


def r_hybrid_slant_oop():
    s = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                         n_orders=5)
    s.add_layer(DEP, eps_tensor_cell=OOPC, slant=(T35, 0.0))
    s.set_source(WL, theta=0.2)
    s.solve()


def r_zero_ezz():
    c = np.zeros((2, 2, 3, 3), dtype=complex)
    c[:, :] = np.diag([2.0, 2.0, 0.0])
    Granet2DTransverseE(PX, PY, 2, 2, 4, c, slant=(T35, 0.0))


REFUSALS = {
    "mixed_patterned_slants": (r_mixed_patterned, NotImplementedError,
                               ("MIXED SLANTS", "PATTERNED")),
    "vertical_patterned_counts_as_zero": (r_vertical_patterned_counts,
                                          NotImplementedError,
                                          ("MIXED SLANTS",)),
    "mixed_frame_offset_above_pattern": (r_mixed_above_pattern,
                                         NotImplementedError,
                                         ("MIX of vertical and slanted",)),
    "mu_uniform_plus_slant": (r_mu_uniform, NotImplementedError, ("mu",)),
    "mu_cell_plus_slant": (r_mu_cell, NotImplementedError, ("mu",)),
    "solver_mu_cell_plus_slant": (r_solver_mu, NotImplementedError,
                                  ("mu_cell",)),
    "retain_internal_plus_slant": (r_retain_internal, NotImplementedError,
                                   ("retain_internal",)),
    "scalar_efficiency_entry": (r_efficiency_entry, NotImplementedError,
                                ("pmm_jones_2d_staggered",)),
    "three_component_slant": (r_three_vector, ValueError, ("slant",)),
    "nonfinite_slant": (r_nonfinite, ValueError, ("finite",)),
    "hybrid_slant_times_oop": (r_hybrid_slant_oop, NotImplementedError,
                               ("SLANTED", "OUT-OF-PLANE")),
    "zero_ezz_with_slant": (r_zero_ezz, (ValueError, NotImplementedError),
                            ()),
}


# ------------------------------------------------------------------ accepted
def a_one_slanted_among_uniform():
    s = _st()
    s.add_layer(0.15e-6, eps=2.1 + 0j)
    s.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    s.add_layer(0.11e-6, eps=2.1 + 0j)
    s.set_source(WL, theta=0.2)
    return s.solve(jones=True)


def a_two_patterned_same_slant():
    s = _st()
    s.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
    s.add_layer(DEP, eps_cell=OOPC, slant=(T35, 0.0))
    s.set_source(WL, theta=0.2)
    return s.solve(jones=True)


def a_uniform_only_mixed_slants():
    s = _st()
    s.add_layer(0.15e-6, eps=2.1 + 0j, slant=(T35, 0.0))
    s.add_layer(0.19e-6, eps=2.4 + 0j, slant=(0.0, 0.2))
    s.set_source(WL, theta=0.2, phi=0.4)
    return s.solve(jones=True)


def a_uniform_only_vertical():
    s = _st()
    s.add_layer(0.15e-6, eps=2.1 + 0j)
    s.add_layer(0.19e-6, eps=2.4 + 0j)
    s.set_source(WL, theta=0.2, phi=0.4)
    return s.solve(jones=True)


# ------------------------------------------------------------------ sneaks
def _solve_one(sl, M=5):
    s = _st(M)
    s.add_layer(DEP, eps_cell=SCA, slant=sl)
    s.set_source(WL, theta=0.2, phi=0.4)
    return s.solve(jones=True)


def main():
    out = {"refusals": {}, "accepted": {}, "sneaks": {}}
    for name, (fn, exc, need) in REFUSALS.items():
        out["refusals"][name] = expect_raise(name, fn, exc, need)
        r = out["refusals"][name]
        print(f"[R] {name:38s} {str(r['raised']):22s} "
              f"{'OK' if r['ok'] else 'FAIL ' + str(r['missing'])}")

    # ---- accepted shapes must not raise
    for name, fn in (("one_slanted_among_uniform", a_one_slanted_among_uniform),
                     ("two_patterned_same_slant", a_two_patterned_same_slant),
                     ("uniform_only_mixed_slants",
                      a_uniform_only_mixed_slants)):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                res = fn()
            out["accepted"][name] = dict(
                ok=True, warnings=[str(x.message)[:200] for x in w],
                closure=float(np.max(res[1].sum(1) + res[2].sum(1))))
        except Exception as e:                  # noqa: BLE001
            out["accepted"][name] = dict(ok=False, err=f"{type(e).__name__}: "
                                                       f"{e}")
        print(f"[A] {name:38s} {out['accepted'][name]}")

    # uniform-ONLY mixed slants is a physical NO-OP -- measure it
    o1, R1, T1, J1 = a_uniform_only_mixed_slants()
    o2, R2, T2, J2 = a_uniform_only_vertical()
    out["accepted"]["uniform_only_mixed_slants_is_a_noop"] = dict(
        dR=mx(R1, R2), dT=mx(T1, T2), dJones=mx(J1, J2))
    print("[A] uniform-only mixed slants vs all-vertical: "
          f"{out['accepted']['uniform_only_mixed_slants_is_a_noop']}")

    # ---- READING 2: one slanted layer of d == two of d/2 (the layer split)
    for th, ph, lab in ((0.0, 0.0, "normal"), (0.35, 0.6, "conical")):
        s1 = _st()
        s1.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
        s1.set_source(WL, theta=th, phi=ph)
        a = s1.solve(jones=True)
        s2 = _st()
        s2.add_layer(DEP / 2, eps_cell=SCA, slant=(T35, 0.0))
        s2.add_layer(DEP / 2, eps_cell=SCA, slant=(T35, 0.0))
        s2.set_source(WL, theta=th, phi=ph)
        b = s2.solve(jones=True)
        out["accepted"][f"layer_split_{lab}"] = dict(
            dR=mx(a[1], b[1]), dT=mx(a[2], b[2]), dJones=mx(a[3], b[3]),
            dJonesT=mx(s1.jones_transmission(), s2.jones_transmission()))
        print(f"[A] layer split {lab}: "
              f"{out['accepted'][f'layer_split_{lab}']}")

    # ---- READING 1 vs READING 2 on the SAME solid: a uniform film above one
    # slanted pattern, the film VERTICAL (Sh = 0) and the film SLANTED at the
    # same slant (Sh = t Z).  A uniform film is a translation-invariant medium,
    # so the two describe the same solid up to a rigid lateral translation ->
    # every efficiency and the zeroth-order Jones must agree.
    for th, ph, lab in ((0.2, 0.0, "oblique"), (0.35, 0.6, "conical")):
        sa = _st()
        sa.add_layer(0.15e-6, eps=2.1 + 0j)
        sa.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
        sa.set_source(WL, theta=th, phi=ph)
        A = sa.solve(jones=True)
        sb = _st()
        sb.add_layer(0.15e-6, eps=2.1 + 0j, slant=(T35, 0.0))
        sb.add_layer(DEP, eps_cell=SCA, slant=(T35, 0.0))
        sb.set_source(WL, theta=th, phi=ph)
        B = sb.solve(jones=True)
        out["accepted"][f"reading1_vs_reading2_{lab}"] = dict(
            dR=mx(A[1], B[1]), dT=mx(A[2], B[2]), dJones=mx(A[3], B[3]),
            dJonesT=mx(sa.jones_transmission(), sb.jones_transmission()))
        print(f"[A] reading1 vs reading2 {lab}: "
              f"{out['accepted'][f'reading1_vs_reading2_{lab}']}")

    # ---- SNEAKS
    tiny = (1e-17, 0.0)
    sn = {}
    # 1. does a 1e-17 slant count as SLANTED to the stack gate?
    sn["tiny_slant_trips_mixed_gate"] = expect_raise(
        "tiny", lambda: (lambda s: (s.add_layer(DEP, eps_cell=SCA, slant=tiny),
                                    s.add_layer(DEP, eps_cell=SCA),
                                    s.set_source(WL, theta=0.2),
                                    s.solve(jones=True)))(_st()),
        NotImplementedError, ("MIXED SLANTS",))
    # 2. and to the EFFICIENCY entry?
    sn["tiny_slant_efficiency_entry"] = expect_raise(
        "tiny_eff",
        lambda: pmm_efficiency_2d_staggered(PX, PY, SCA, NSUB, NSUP, DEP, WL,
                                            degree=4, n_orders=3, slant=tiny),
        NotImplementedError, ())
    # 3. does it change the answer / the routing?
    a = _solve_one(None)
    b = _solve_one(tiny)
    sn["tiny_slant_answer"] = dict(dR=mx(a[1], b[1]), dT=mx(a[2], b[2]),
                                   dJones=mx(a[3], b[3]))
    s_v = _st()
    s_v.add_layer(DEP, eps_cell=SCA)
    s_t = _st()
    s_t.add_layer(DEP, eps_cell=SCA, slant=tiny)
    sn["tiny_slant_routes_offplane"] = dict(
        vertical_offplane=bool(Granet2DTransverseE(
            PX, PY, 2, 2, 5, SCA).offplane),
        tiny_offplane=bool(Granet2DTransverseE(
            PX, PY, 2, 2, 5, SCA, slant=tiny).offplane))
    # 4. spellings that must be ACCEPTED as a real slant
    ok = {}
    ref = _solve_one((T35, 0.0))
    for sname, sv in (("list", [T35, 0.0]), ("nparray", np.array([T35, 0.0])),
                      ("scalar", T35), ("npfloat", np.float64(T35))):
        r = _solve_one(sv)
        ok[sname] = dict(dR=mx(r[1], ref[1]), dJones=mx(r[3], ref[3]))
    sn["spelling_equivalence"] = ok
    # 5. a UNIFORM layer with a slant: accepted, must be a no-op
    su = _st()
    su.add_layer(DEP, eps=2.25 + 0j, slant=(T35, 0.0))
    su.set_source(WL, theta=0.2, phi=0.4)
    U = su.solve(jones=True)
    sv2 = _st()
    sv2.add_layer(DEP, eps=2.25 + 0j)
    sv2.set_source(WL, theta=0.2, phi=0.4)
    V = sv2.solve(jones=True)
    sn["uniform_layer_slant_is_noop"] = dict(
        dR=mx(U[1], V[1]), dT=mx(U[2], V[2]), dJones=mx(U[3], V[3]),
        dJonesT=mx(su.jones_transmission(), sv2.jones_transmission()))
    # 6. the parity accelerator: does the gauge refuse a slanted solver?
    sol_v = Granet2DTransverseE(PX, PY, 2, 2, 6, CENTRO)
    sol_s = Granet2DTransverseE(PX, PY, 2, 2, 6, CENTRO, slant=(T35, 0.0))
    sn["parity_gauge"] = dict(
        vertical_gauge_is_none=_stag_parity_gauge(sol_v) is None,
        slanted_gauge_is_none=_stag_parity_gauge(sol_s) is None)
    # 7. symmetry='auto' on a slanted cell must equal symmetry=False
    ja = pmm_jones_2d_staggered(PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5,
                                n_orders=3, symmetry="auto",
                                slant=(T35, 0.0))
    jf = pmm_jones_2d_staggered(PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5,
                                n_orders=3, symmetry=False,
                                slant=(T35, 0.0))
    va = pmm_jones_2d_staggered(PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5,
                                n_orders=3, symmetry="auto")
    vf = pmm_jones_2d_staggered(PX, PY, CENTRO, NSUB, NSUP, DEP, WL, degree=5,
                                n_orders=3, symmetry=False)
    from _lib import sha
    sn["symmetry_auto_vs_false"] = dict(
        slanted_identical=(sha(ja[1], ja[2], ja[3])
                           == sha(jf[1], jf[2], jf[3])),
        vertical_identical=(sha(va[1], va[2], va[3])
                            == sha(vf[1], vf[2], vf[3])),
        slant_is_visible=mx(ja[3], va[3]))
    out["sneaks"] = sn
    for k, v in sn.items():
        print(f"[S] {k}: {v}")

    dump("v6_refusals", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
