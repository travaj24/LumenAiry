"""V1 -- DEFAULT-ON MOVES NOTHING OUTSIDE ITS DOMAIN.

Three arms, hashed on (orders, R, T, Jones):

  A  this tip, ``symmetry='auto'``  (the shipped default)
  B  this tip, ``symmetry=False``   (the forced dense path)
  C  the READ-ONLY main clone at fb3fd93 (no ``symmetry`` keyword at all)

A vs B isolates the ACCELERATOR; B vs C isolates everything ELSE the merge
carries (the Wood-list unification, the magnetic weights); A vs C is the
composite the consumer actually sees.

Fixture classes:
  * scalar and in-plane-tensor cells (the in-plane pencil never enters the OOP
    solve, so all three arms must be bit-identical);
  * a magnetic cell (tip only -- ``mu_cell`` does not exist at fb3fd93);
  * OUT-OF-PLANE cells that the gate must REFUSE (oblique, conical, off-centre,
    parity-breaking tensor), where A must equal B bit-for-bit.

Usage:  python v1_default_on.py tip     (from the tip worktree)
        python v1_default_on.py main    (PYTHONPATH at the main clone)
        python v1_default_on.py compare
"""
import json
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import vfix as F  # noqa: E402

TIP = "C:/tmp/lum_vacc"
MAIN = "D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"

M_DEG = 6
NORD = 5


def build_cases():
    """(name, kind, kwargs) -- kind in {'eps', 'mu'}."""
    t = F.tensors()
    cases = []

    # ---- 1. SCALAR cells (2-D array): never out-of-plane
    scal2 = np.array([[2.25, 1.0], [1.0, 2.25]], dtype=complex)
    scal3 = np.array([[4.0, 1.0, 2.1],
                      [1.0, 6.25, 1.0],
                      [2.1, 1.0, 4.0]], dtype=complex)
    cases.append(("scalar (2,2) normal", dict(cell=scal2, theta=0.0, phi=0.0)))
    cases.append(("scalar (3,3) normal", dict(cell=scal3, theta=0.0, phi=0.0)))
    cases.append(("scalar (3,3) conical 25/40",
                  dict(cell=scal3, theta=np.deg2rad(25.0),
                       phi=np.deg2rad(40.0))))

    # ---- 2. IN-PLANE tensor cells (e13=e23=e31=e32=0 exactly)
    ip = F.inplane_tensor()
    ipc = F.tile(F.AIR, 3)
    ipc[1, 1] = ip
    ipc[0, 0] = ipc[2, 2] = 2.1 * np.eye(3)
    cases.append(("in-plane tensor (3,3) centro normal",
                  dict(cell=ipc, theta=0.0, phi=0.0)))
    ipa = F.tile(F.AIR, 2)
    ipa[0, 0] = ip                    # OFF-CENTRE in-plane: still untouched
    cases.append(("in-plane tensor (2,2) off-centre normal",
                  dict(cell=ipa, theta=0.0, phi=0.0)))
    cases.append(("in-plane tensor (3,3) centro oblique 20",
                  dict(cell=ipc, theta=np.deg2rad(20.0), phi=0.0)))

    # ---- 3. OUT-OF-PLANE cells the gate MUST REFUSE
    cases.append(("OOP off-centre pillar (2,2) normal",
                  dict(cell=F.offcentre(t["lc"], 2))))
    cases.append(("OOP off-centre pillar (3,3) normal",
                  dict(cell=F.offcentre(t["generic"], 3))))
    cases.append(("OOP parity-breaking tensor (2,2) normal",
                  dict(cell=F.parity_breaking_tensor(t["lc"], t["lc2"], 2))))
    cases.append(("OOP parity-breaking tensor (3,3) normal",
                  dict(cell=F.parity_breaking_tensor(t["lc"], t["nonrec"], 3))))
    cases.append(("OOP centro (2,2) OBLIQUE 25",
                  dict(cell=F.centro_pair(t["lc"], 2),
                       theta=np.deg2rad(25.0), phi=0.0)))
    cases.append(("OOP centro (3,3) CONICAL 25/40",
                  dict(cell=F.centro_pair(t["lc"], 3, centre=t["lossy"]),
                       theta=np.deg2rad(25.0), phi=np.deg2rad(40.0))))
    cases.append(("OOP centro (2,2) OBLIQUE 5 (small angle)",
                  dict(cell=F.centro_pair(t["lc"], 2),
                       theta=np.deg2rad(5.0), phi=0.0)))
    cases.append(("OOP centro (3,3) CONICAL 1/90 (tiny polar)",
                  dict(cell=F.centro_pair(t["lc"], 3, centre=t["lossy"]),
                       theta=np.deg2rad(1.0), phi=np.deg2rad(90.0))))

    return cases


def magnetic_cases():
    """Magnetic fixtures -- TIP ONLY (``mu_cell`` is new on this branch)."""
    out = []
    ip = F.inplane_tensor()
    ipc = F.tile(F.AIR, 3)
    ipc[1, 1] = ip
    ipc[0, 0] = ipc[2, 2] = 2.1 * np.eye(3)
    mu_s = np.array([[1.0, 1.4, 1.0],
                     [1.4, 0.8, 1.4],
                     [1.0, 1.4, 1.0]], dtype=complex)
    out.append(("MAGNETIC scalar-mu on in-plane tensor (3,3) normal",
                dict(cell=ipc, mu_cell=mu_s, theta=0.0, phi=0.0)))
    scal3 = np.array([[4.0, 1.0, 2.1],
                      [1.0, 6.25, 1.0],
                      [2.1, 1.0, 4.0]], dtype=complex)
    mu_t = np.zeros((3, 3, 3, 3), dtype=complex)
    mu_t[:, :] = np.eye(3)
    mt = np.eye(3, dtype=complex)
    mt[0, 0] = 1.3
    mt[1, 1] = 0.75
    mt[0, 1] = 0.22
    mt[1, 0] = 0.22
    mu_t[1, 1] = mt
    out.append(("MAGNETIC tensor-mu on scalar eps (3,3) conical 25/40",
                dict(cell=scal3, mu_cell=mu_t, theta=np.deg2rad(25.0),
                     phi=np.deg2rad(40.0))))
    return out


def run(arm):
    prefix = TIP if arm == "tip" else MAIN
    path, ver = F.assert_arm(prefix)
    from lumenairy.elements.pmm.twod_staggered import pmm_jones_2d_staggered

    def call(cell, theta=0.0, phi=0.0, sym=None, mu_cell=None):
        kw = dict(degree=M_DEG, n_orders=NORD, theta=theta, phi=phi)
        if sym is not None:
            kw["symmetry"] = sym
        if mu_cell is not None:
            kw["mu_cell"] = mu_cell
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pmm_jones_2d_staggered(F.PX, F.PY, cell, F.NSUB, F.NSUP,
                                          F.DEP, F.WL, **kw)

    rows = {}
    cases = build_cases()
    if arm == "tip":
        cases = cases + magnetic_cases()
    for name, kw in cases:
        cell = kw.pop("cell")
        rec = {}
        if arm == "tip":
            a = call(cell, sym="auto", **kw)
            b = call(cell, sym=False, **kw)
            rec["A_auto"] = F.sha(*a)
            rec["B_false"] = F.sha(*b)
            rec["A_eq_B"] = rec["A_auto"] == rec["B_false"]
            rec["dR_AB"] = F.dmax(a[1], b[1])
            rec["dT_AB"] = F.dmax(a[2], b[2])
            rec["dJ_AB"] = F.dmax(a[3], b[3])
            rec["sumT"] = float(np.sum(np.asarray(a[2])))
            rec["sumR"] = float(np.sum(np.asarray(a[1])))
        else:
            if "mu_cell" in kw:
                continue
            c = call(cell, **kw)
            rec["C_main"] = F.sha(*c)
            rec["sumT"] = float(np.sum(np.asarray(c[2])))
            rec["sumR"] = float(np.sum(np.asarray(c[1])))
        rows[name] = rec
        print(f"[{arm}] {name}: {rec}")

    out = dict(arm=arm, lumenairy=path, version=ver,
               numpy=np.__version__, rows=rows)
    with open(os.path.join(HERE, "results", f"v1_{arm}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


def compare():
    tip = json.load(open(os.path.join(HERE, "results", "v1_tip.json")))
    main = json.load(open(os.path.join(HERE, "results", "v1_main.json")))
    print(f"tip  {tip['lumenairy']} v{tip['version']}")
    print(f"main {main['lumenairy']} v{main['version']}")
    print()
    hdr = f"{'case':52s} {'A=B':5s} {'B=C':5s} {'A=C':5s}  dR(A,B)"
    print(hdr)
    print("-" * len(hdr))
    verdict = {"A_eq_B": [], "B_eq_C": [], "A_eq_C": []}
    for name, rec in tip["rows"].items():
        mrec = main["rows"].get(name)
        aeb = rec["A_eq_B"]
        if mrec is None:
            beq = aeq = None
        else:
            beq = rec["B_false"] == mrec["C_main"]
            aeq = rec["A_auto"] == mrec["C_main"]
        print(f"{name:52s} {str(aeb):5s} {str(beq):5s} {str(aeq):5s}  "
              f"{rec['dR_AB']:.3e}")
        verdict["A_eq_B"].append((name, aeb))
        if mrec is not None:
            verdict["B_eq_C"].append((name, beq))
            verdict["A_eq_C"].append((name, aeq))
    print()
    for k, v in verdict.items():
        bad = [n for n, ok in v if not ok]
        print(f"{k}: {sum(1 for _, ok in v if ok)}/{len(v)} identical"
              + (f"   FAILURES: {bad}" if bad else ""))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tip"
    if mode == "compare":
        compare()
    else:
        run(mode)
