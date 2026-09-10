"""V7 -- DURABILITY audit of the two new test files.

Every numeric constant in

  tests/unit/test_pmm2d_staggered_oop_block_eig.py
  tests/unit/test_pmm2d_staggered_oop_corner_convergence.py

is re-measured HERE on the test's own fixtures, so the table can state, for
each bar: what quantity it reads, the value the running build produces, the
gap to the bar, and the gap on the OTHER side (the smallest real signal the
bar must separate from).  The tests' fixture constructors are imported from
the file under audit -- the point of this script is the MARGIN of those exact
assertions, not a re-derivation of the physics (that is V2 / V3 / V4).

It also supplies the one number the build did not measure: the backward-error
RATIO when the reduction is FORCED onto a cell that does not carry the
structure, i.e. the lower-side signal for the ``bf <= 1e2 * bd`` bar.

Usage: python v7_test_durability.py
"""
import importlib.util
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import vfix as F  # noqa: E402

F.assert_arm("C:/tmp/lum_vacc")

from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = load("t_blockeig",
         os.path.join(ROOT, "tests", "unit",
                      "test_pmm2d_staggered_oop_block_eig.py"))
C = load("t_corner",
         os.path.join(ROOT, "tests", "unit",
                      "test_pmm2d_staggered_oop_corner_convergence.py"))


def main():
    rows = []

    def rec(test, bar, quantity, measured, other_side, note=""):
        rows.append(dict(test=test, bar=bar, quantity=quantity,
                         measured=measured, other_side=other_side, note=note))
        print(f"\n{test}\n   bar        {bar}\n   quantity   {quantity}\n"
              f"   MEASURED   {measured}\n   other side {other_side}"
              + (f"\n   note       {note}" if note else ""))

    print("=" * 96)
    print("test_pmm2d_staggered_oop_block_eig.py")
    print("=" * 96)

    # ---- 1. parity map / mass / derivative
    mt, mb, dodd, dwrong = [], [], [], []
    for Nx in (2, 3, 4):
        for M in (5, 6, 8):
            b = T.solver(T._tile(T.TIL, Nx), M).bx
            pt, st, pb, sb = TS._stag_parity_1d(b)
            Pt = np.zeros((pt.size, pt.size))
            Pt[pt, np.arange(pt.size)] = st
            Pb = np.zeros((pb.size, pb.size))
            Pb[pb, np.arange(pb.size)] = sb
            assert np.array_equal(Pt @ Pt, np.eye(pt.size))
            assert np.array_equal(Pb @ Pb, np.eye(pb.size))
            Mtt = b.mass(b.Btilde, b.Btilde)
            Mbb = b.mass(b.B, b.B)
            Cbt = b.mixed(b.B, b.Btilde)
            mt.append(np.max(np.abs(Pt.T @ Mtt @ Pt - Mtt)) / np.max(np.abs(Mtt)))
            mb.append(np.max(np.abs(Pb.T @ Mbb @ Pb - Mbb)) / np.max(np.abs(Mbb)))
            dodd.append(np.max(np.abs(Pb.T @ Cbt @ Pt + Cbt)) / np.max(np.abs(Cbt)))
            dwrong.append(np.max(np.abs(Pb.T @ Cbt @ Pt - Cbt)) / np.max(np.abs(Cbt)))
    rec("test_parity_map_is_an_exact_involution... (J^2)",
        "np.array_equal(P @ P, I)  -- EXACT, no tolerance",
        "|P^2 - I|", "0.0 on 12/12 (Nx, M)",
        "n/a (exact equality of a signed permutation squared)")
    rec("test_parity_map_is_an_exact_involution... (masses)",
        "<= 1e-12 * max|M|",
        "max|P^T M P - M| / max|M| for Mtt and Mbb",
        f"{min(mt + mb):.2e} .. {max(mt + mb):.2e}  (12 combos)",
        "the assembled generator reads 2.4e-02 .. 1.0e+00 when the parity "
        "genuinely fails (V2/V4)",
        f"gap above the measurement: {np.log10(1e-12 / max(mt + mb)):.1f} "
        f"decades")
    rec("test_parity_map_is_an_exact_involution... (derivative ODD)",
        "<= 1e-12 * max|C|  and  > 1e-3 * max|C| for the WRONG sign",
        "max|Pb^T C Pt +/- C| / max|C|",
        f"ODD {min(dodd):.2e} .. {max(dodd):.2e};  "
        f"EVEN(wrong) {min(dwrong):.2e} .. {max(dwrong):.2e}",
        "the wrong-sign reading is 2.00 exactly",
        f"gaps: {np.log10(1e-12 / max(dodd)):.1f} decades above the ODD "
        f"envelope; {np.log10(min(dwrong) / 1e-3):.1f} decades above the "
        f"1e-3 bar")

    # ---- 2. struct_dA on the test's own carrying fixtures
    car = []
    for Nx in (2, 3):
        for cellf in (lambda n: T._tile(T.TIL, n), lambda n: T.centro(T.TIL, n)):
            car.append(T.struct_dA(T.solver(cellf(Nx), 6)))
    bad = [T.struct_dA(T.solver(T.broken(n), 6)) for n in (2, 3)]
    bad += [T.struct_dA(T.solver(T.offcentre(T.TIL, n), 6)) for n in (2, 3)]
    rec("test_reduction_engages_and_splits_into_two_equal_sectors",
        "struct_dA <= 1e-10  (== the shipped _STAG_BLOCK_TOL)",
        "max|R A R + A| / max|A| on the assembled pencil",
        f"carrying {min(car):.2e} .. {max(car):.2e} (4 fixtures, M=6)",
        f"the file's own violating fixtures {min(bad):.2e} .. {max(bad):.2e}; "
        f"an ENGINEERED 1e-7-relative parity break reads 2.5e-09 (V2)",
        f"gap above {np.log10(1e-10 / max(car)):.1f} decades; gap below "
        f"{np.log10(min(bad) / 1e-10):.1f} decades on the file's fixtures but "
        f"only 1.4 decades on the engineered one")
    rec("test_reduction_engages_..._sectors (sector counts)",
        "pairs + fixed(+) == 2 q^2  and  sum(r[fixed]) == 0.0 -- EXACT",
        "counted from the SIGNED PERMUTATION, not from an eig",
        "exact on every fixture (V3 confirms tr R = 0 by construction: the "
        "E1/G2 and E2/G1 blocks carry the same permutation with opposite "
        "signs)",
        "n/a -- deterministic integer arithmetic, no build dependence")

    # ---- 3. ON vs OFF observables, the file's own 6 parametrizations
    dR = dT = dJ = 0.0
    for Nx in (2, 3):
        for tensor in (T.TIL, T.NREC, T.LOSSY):
            cell = T.centro(tensor, Nx)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                on = T.jones_call(cell, 6, True)
                off = T.jones_call(cell, 6, False)
            dR = max(dR, T.dmax(on[1], off[1]))
            dT = max(dT, T.dmax(on[2], off[2]))
            dJ = max(dJ, T.dmax(on[3], off[3]))
    rec("test_reduction_matches_the_dense_path_on_R_T_and_jones",
        "dR, dT, dJones <= 1e-11",
        "max|ON - OFF| per order, both polarizations, through the entry point",
        f"dR {dR:.2e}  dT {dT:.2e}  dJones {dJ:.2e}  (the file's own 6 params)",
        "a FORCED reduction on a structure-violating cell reads 2.4e-04 .. "
        "3.4e-01 (V4)",
        f"gap above {np.log10(1e-11 / max(dR, dT, dJ)):.1f} decades; gap "
        f"below {np.log10(2.4e-4 / 1e-11):.1f} decades")

    # ---- 4. eigenvalue set + split
    hd = 0.0
    for Nx in (2, 3):
        sol = T.solver(T.centro(T.NREC, Nx), 6)
        qq = sol.q * sol.q
        on = TS._region_modes_oop(sol, symmetry=True)
        off = TS._region_modes_oop(sol, symmetry=False)
        assert on[2].size == off[2].size == 2 * qq
        hd = max(hd, T.hausdorff(np.concatenate([on[2], on[5]]),
                                 np.concatenate([off[2], off[5]])))
    rec("test_reduction_reproduces_the_dense_eigenvalue_set_and_flux_split",
        "symmetric Hausdorff <= 1e-9;  counts == 2 q^2 EXACT",
        "set distance between the ON and OFF spectra on the same pencil",
        f"{hd:.2e} (the file's 2 params)",
        "a FORCED reduction moves the set by 7.7e-02 .. 1.1e+01 (V4)",
        f"gap above {np.log10(1e-9 / hd):.1f} decades; gap below "
        f"{np.log10(7.7e-2 / 1e-9):.1f} decades.  The COUNT is not a fragile "
        "census: _region_modes_oop RAISES if the split is not 2q^2/2q^2, so "
        "the assertion restates an invariant the library enforces")

    # ---- 5. backward error ratio, incl. the FORCED lower side
    ratios, dens = [], []
    for Nx in (2, 3):
        sol = T.solver(T.centro(T.TIL, Nx), 6)
        g = TS._stag_parity_gauge(sol)
        fac = TS._stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q, g)
        qd, Xd = T.dense_eig(sol)
        bd = T.backward_error(sol.Agen, sol.Bgen, qd, Xd)
        bf = T.backward_error(sol.Agen, sol.Bgen, fac[0], fac[1])
        ratios.append(bf / bd)
        dens.append((bd, 1e3 * 4 * sol.q * sol.q * np.finfo(float).eps))
    forced = []
    for cellf in (lambda: T.offcentre(T.TIL, 2), lambda: T.broken(3)):
        sol = T.solver(cellf(), 6)
        g = TS._stag_parity_gauge(sol)
        fac = TS._stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q, g,
                                 tol=np.inf)
        qd, Xd = T.dense_eig(sol)
        bd = T.backward_error(sol.Agen, sol.Bgen, qd, Xd)
        bf = T.backward_error(sol.Agen, sol.Bgen, fac[0], fac[1])
        forced.append(bf / bd)
    rec("test_factored_eigenpair_residual_is_at_the_dense_solve_own_scale",
        "bf <= 1e2 * bd   and   bd <= 1e3 * 4 q^2 * eps",
        "normwise eigenpair backward error, factored vs dense, same pencil",
        f"ratio {min(ratios):.2f} .. {max(ratios):.2f} (the file's 2 params); "
        f"19 of my own combinations give 0.44 .. 5.39 (V4); "
        f"bd {min(d for d, _ in dens):.2e} .. {max(d for d, _ in dens):.2e} "
        f"against its own bar {min(x for _, x in dens):.2e} .. "
        f"{max(x for _, x in dens):.2e}",
        f"FORCING the reduction on a structure-violating cell gives a ratio "
        f"{min(forced):.1f} .. {max(forced):.1f}",
        f"gap above the measured ratio "
        f"{np.log10(1e2 / max(max(ratios), 5.39)):.2f} decades; gap BELOW to "
        f"the smallest forced (structure-violating) ratio "
        f"{np.log10(min(forced) / 1e2):.1f} decades -- so the bar IS "
        "two-sided: a forced reduction blows the eigenpair backward error up "
        "by ~12 decades")

    # ---- 6. the tolerance-ladder test
    ok = T.solver(T.centro(T.TIL, 3), 6)
    bd3 = T.solver(T.broken(3), 6)
    rec("test_structure_bar_has_a_gap_on_both_sides_measured_through_the_gate",
        "engages at tol/1e2; refused at tol*1e3; engages at tol=1.0",
        "the SHIPPED gate walked through its own call-time tol",
        f"struct_dA(ok) {T.struct_dA(ok):.2e} < 1e-10 < "
        f"struct_dA(bad) {T.struct_dA(bd3):.2e}",
        "the tol=1.0 arm requires struct_dA(bad) < 1; my off-centre METAL "
        "fixture reads 1.033 and is still REFUSED at tol=1.0 (V4)",
        "not a defect -- the assertion would FAIL LOUDLY, not pass vacuously "
        "-- but ``tol=1.0`` is not a full disarm; ``np.inf`` is")

    # ---- 7. fail-before
    fb = []
    for cellf in (lambda: T.offcentre(T.TIL, 2), lambda: T.broken(3)):
        cell = cellf()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = T.jones_call(cell, 6, False)
            saved = TS._STAG_BLOCK_TOL
            try:
                TS._STAG_BLOCK_TOL = 1.0
                bad = T.jones_call(cell, 6, True)
            finally:
                TS._STAG_BLOCK_TOL = saved
        fb.append(max(T.dmax(bad[1], ref[1]), T.dmax(bad[2], ref[2]),
                      T.dmax(bad[3], ref[3])))
    rec("test_forcing_the_reduction_without_the_structure_is_wrong_by_decades",
        "> 1e-4",
        "max(dR, dT, dJones) with _STAG_BLOCK_TOL monkeypatched to 1.0",
        f"{min(fb):.2e} .. {max(fb):.2e} (the file's 2 params)",
        "the ACCEPTED agreement is <= 1e-11 (bar) / <= 4.9e-14 (measured)",
        f"gap below the measurement {np.log10(min(fb) / 1e-4):.1f} decades; "
        f"gap above the accepted agreement {np.log10(1e-4 / 1e-11):.0f} "
        "decades")

    # ---- 8. the corner-convergence test
    print()
    print("=" * 96)
    print("test_pmm2d_staggered_oop_corner_convergence.py")
    print("=" * 96)
    cell = C.l_cell()
    vals = []
    for M in (4, 5, 6, 7):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, _R, Tt, _J = C.pmm_jones_2d_staggered(
                C.PX, C.PY, cell, C.NSUB, C.NSUP, C.DEP, C.WL, degree=M,
                n_orders=5)
        vals.append(np.asarray(Tt))
    steps = [float(np.max(np.abs(vals[i + 1] - vals[i])))
             for i in range(len(vals) - 1)]
    ratios = [steps[i + 1] / steps[i] for i in range(len(steps) - 1)]
    rec("test_..._corner_ladder_steps_shrink_and_start_above_the_floor (a)",
        "steps[0] > 1e-5",
        "max per-order |T(M=5) - T(M=4)|",
        f"{steps[0]:.3e}  ({steps[0] / 1e-5:.0f}x the bar)",
        "a CONVERGED fixture would read ~1e-09",
        f"gap above the bar {np.log10(steps[0] / 1e-5):.2f} decades; gap to "
        f"the converged floor {np.log10(1e-5 / 1e-9):.0f} decades")
    rec("test_..._corner_ladder_steps_shrink_and_start_above_the_floor (b)",
        "steps[i+1] <= 0.5 * steps[i]   -- SUB-DECADE BAR",
        "successive step ratios of the per-order T ladder",
        f"{ratios[0]:.3f} and {ratios[1]:.3f}  (steps "
        f"{steps[0]:.3e}, {steps[1]:.3e}, {steps[2]:.3e})",
        "the cross-build spread of a fixed-M solve is ~1e-15 absolute, i.e. "
        "~1e-10 RELATIVE on these steps",
        f"headroom {0.5 / max(ratios):.2f}x = "
        f"{np.log10(0.5 / max(ratios)):.2f} decades -- sub-decade, but the "
        "quantity's own build spread is ~10 decades below the headroom, so "
        "the bar is not per-build; it is a convergence-rate assumption about "
        "THIS fixture")
    rec("test_..._corner_ladder_steps_shrink_and_start_above_the_floor (c)",
        "steps[-1] <= 0.1 * steps[0]",
        "end-to-end fall of the ladder",
        f"{steps[-1] / steps[0]:.4f}  ({steps[0] / steps[-1]:.0f}x)",
        "same as (b)",
        f"headroom {0.1 / (steps[-1] / steps[0]):.2f}x")

    with open(os.path.join(HERE, "results", "v7_test_durability.json"),
              "w") as fh:
        json.dump(dict(rows=rows, corner_steps=steps, corner_ratios=ratios,
                       forced_backward_ratio=forced), fh, indent=1)


if __name__ == "__main__":
    main()
