"""V4 -- the SLANTED per-layer gate (the build's OWN open item) and MIXED
layer kinds on different grids.

The build doc's open item "NEW per-layer + SLANT" says the slant is plumbed
through the per-layer path but exercised only indirectly.  This is the
dedicated gate:

* ``bypass``  slanted PER-LAYER on grids that COINCIDE must be BIT-EXACT
  against ``layer_grids='shared'`` (the generalized-cascade bypass), and
  forced through the mortar must reduce to it at round-off.
* ``oracle``  a y-uniform SLANTED grating SPLIT into two layers at the SAME
  slant on NON-CONFORMING grids, PER ORDER against the shipped 1-D
  inclined-coordinate solver ``pmm_efficiency_1d_slanted`` -- with the
  WRONG-SIGN arm as the two-sided control (a slanted grating is not
  x-mirror-symmetric, so the sign is observable per order; energy is not).
* ``split``   the layer-split identity through the mortar: one slanted layer
  of depth ``d`` vs two of ``d/2`` on DIFFERENT grids.
* ``mixed``   OOP over scalar over MAGNETIC, each on its own grid, against
  the union-grid shared cascade.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_efficiency_1d_slanted
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}

# --- fixture: MY OWN, y-uniform slanted binary grating ---------------------
SPX = 0.80
SWL = 1.0
SDEP = 0.28
SNR, SNG = 2.0, 1.0
NSUP, NSUB = 1.0, 1.5
SCELL2 = np.array([[SNR ** 2, SNR ** 2], [SNG ** 2, SNG ** 2]], dtype=_C)
SCELL4 = np.array([[SNR ** 2] * 4, [SNR ** 2] * 4,
                   [SNG ** 2] * 4, [SNG ** 2] * 4], dtype=_C)   # duty 1/2


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def solve(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(**kw)


# ==========================================================================
def sec_bypass():
    """Conforming slanted grids: BIT-EXACT vs 'shared', and the forced mortar
    reduces to it."""
    out = {}
    t = float(np.tan(np.deg2rad(22.0)))
    for M, N, cell in ((6, 2, SCELL2), (5, 4, SCELL4)):
        def build(lg):
            st = PMM2DStackPure(SPX, SPX, n_superstrate=NSUP,
                                n_substrate=NSUB, n_modes=M, n_orders=2,
                                layer_grids=lg)
            extra = {"n_modes": M} if lg == "per-layer" else {}
            st.add_layer(SDEP / 2, eps_cell=cell, slant=(t, 0.0), **extra)
            st.add_layer(SDEP / 2, eps_cell=cell, slant=(t, 0.0), **extra)
            st.set_source(SWL, theta=0.19, phi=0.0)
            return st
        a = solve(build("shared"))
        b = solve(build("per-layer"))
        eq = all(sha(x) == sha(y) for x, y in zip(a[1:], b[1:]))
        mv = float(max(np.max(np.abs(a[1] - b[1])),
                       np.max(np.abs(a[2] - b[2])),
                       np.max(np.abs(a[3] - b[3]))))
        st = build("per-layer")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = st._solve_per_layer(jones=True, retain_internal=False,
                                    force_mortar=True)
        sc = max(float(np.max(np.abs(a[1]))), float(np.max(np.abs(a[2]))))
        fm = float(max(np.max(np.abs(a[1] - f[1])),
                       np.max(np.abs(a[2] - f[2])))) / sc
        out[f"N{N}_M{M}"] = {"sha_equal": eq, "max_move": mv,
                             "forced_mortar_rel": fm}
        print(f"[bypass] slanted N={N} M={M}: shared-vs-per-layer sha_equal="
              f"{eq} (move {mv:.2e});  FORCED mortar vs bypass {fm:.3e}",
              flush=True)
    RES["bypass"] = out


# ==========================================================================
def _oracle_1d_slanted(phi_deg, theta, pol, deg=22):
    if phi_deg == 0.0:
        o, R, T = pmm_efficiency_1d(SPX, SNR, SNG, NSUB, NSUP, SDEP, 0.5, SWL,
                                    angle=theta, polarization=pol,
                                    degree=deg, far_field_orders=15)
    else:
        o, R, T = pmm_efficiency_1d_slanted(
            SPX, SNR, SNG, NSUB, NSUP, SDEP, 0.5, SWL,
            np.deg2rad(phi_deg), angle=theta, polarization=pol, degree=deg,
            far_field_orders=15)
    return {int(m): (float(R[i]), float(T[i])) for i, m in enumerate(o)}


def _perlayer_slanted(slant, M_A, M_B, theta, n_orders=3):
    """The SPLIT slanted grating: half the depth on N=2, half on N=4 -- a
    NON-CONFORMING pair (N=2 walls {0, 1/2}; N=4 walls {0, 1/4, 1/2, 3/4}),
    which is exactly what the mortar has to carry."""
    st = PMM2DStackPure(SPX, SPX, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=max(M_A, M_B), n_orders=n_orders,
                        layer_grids="per-layer")
    st.add_layer(SDEP / 2, eps_cell=SCELL2, slant=slant, n_modes=M_A)
    st.add_layer(SDEP / 2, eps_cell=SCELL4, slant=slant, n_modes=M_B)
    st.set_source(SWL, theta=theta, phi=0.0)
    o, R, T, _J = solve(st)
    idx = {int(m): i for i, (m, n) in enumerate(o) if n == 0}
    return ({m: (float(R[0, i]), float(T[0, i])) for m, i in idx.items()},
            {m: (float(R[1, i]), float(T[1, i])) for m, i in idx.items()},
            float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))))


def _perorder(a, b, orders=(-1, 0, 1)):
    return max(max(abs(a[m][0] - b[m][0]), abs(a[m][1] - b[m][1]))
               for m in orders if m in a and m in b)


def sec_oracle():
    """THE DEDICATED SLANTED PER-LAYER GATE."""
    out = {}
    for phi_deg in (0.0, 15.0, 28.0):
        for theta, mount in ((0.0, "normal"), (np.deg2rad(22.0), "obl22")):
            orc_te = _oracle_1d_slanted(phi_deg, theta, "te")
            orc_tm = _oracle_1d_slanted(phi_deg, theta, "tm")
            t = float(np.tan(np.deg2rad(phi_deg)))
            arms = ([(None, "vertical")] if phi_deg == 0.0
                    else [((t, 0.0), "+tan"), ((-t, 0.0), "-tan")])
            rec = {}
            for sv, lab in arms:
                tm, te, clo = _perlayer_slanted(sv, 7, 5, theta)
                rec[lab] = {"te": _perorder(te, orc_te),
                            "tm": _perorder(tm, orc_tm), "closure": clo}
            good = "vertical" if phi_deg == 0.0 else "+tan"
            msg = (f"[oracle] slant {phi_deg:4.1f} deg {mount:6s} split "
                   f"N2|N4 (M 7|5): TE {rec[good]['te']:.3e}  "
                   f"TM {rec[good]['tm']:.3e}  closure "
                   f"{rec[good]['closure']:.2e}")
            if "-tan" in rec:
                ratio = rec["-tan"]["te"] / max(rec["+tan"]["te"], 1e-300)
                rec["wrong_sign_ratio_te"] = ratio
                msg += (f"  | WRONG-SIGN TE {rec['-tan']['te']:.3e} "
                        f"({ratio:.0f}x), closure "
                        f"{rec['-tan']['closure']:.2e}")
            out[f"phi{phi_deg}_{mount}"] = rec
            print(msg, flush=True)
    # a modal ladder at one setting, to show it is CONVERGING not coincident
    lad = {}
    for MA, MB in ((5, 4), (7, 5), (9, 6)):
        tm, te, clo = _perlayer_slanted((float(np.tan(np.deg2rad(28.0))), 0.0),
                                        MA, MB, np.deg2rad(22.0))
        orc_te = _oracle_1d_slanted(28.0, np.deg2rad(22.0), "te")
        lad[f"MA{MA}_MB{MB}"] = {"te": _perorder(te, orc_te), "closure": clo}
        print(f"[oracle] ladder MA={MA} MB={MB}: TE vs 1-D slanted "
              f"{lad[f'MA{MA}_MB{MB}']['te']:.3e}  closure {clo:.2e}",
              flush=True)
    out["ladder_phi28_obl22"] = lad
    RES["oracle"] = out


# ==========================================================================
def sec_split():
    """The layer-split identity THROUGH the mortar: one slanted layer of depth
    d against two of d/2 on DIFFERENT grids."""
    out = {}
    t = float(np.tan(np.deg2rad(24.0)))
    for M in (6, 7):
        st1 = PMM2DStackPure(SPX, SPX, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=M, n_orders=2, layer_grids="per-layer")
        st1.add_layer(SDEP, eps_cell=SCELL2, slant=(t, 0.0), n_modes=M)
        st1.set_source(SWL, theta=0.19, phi=0.0)
        a = solve(st1)
        st2 = PMM2DStackPure(SPX, SPX, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=M, n_orders=2, layer_grids="per-layer")
        st2.add_layer(SDEP / 2, eps_cell=SCELL2, slant=(t, 0.0), n_modes=M)
        st2.add_layer(SDEP / 2, eps_cell=SCELL4, slant=(t, 0.0),
                      n_modes=M + 1)
        st2.set_source(SWL, theta=0.19, phi=0.0)
        b = solve(st2)
        sc = max(float(np.max(np.abs(a[1]))), float(np.max(np.abs(a[2]))))
        d = float(max(np.max(np.abs(a[1] - b[1])),
                      np.max(np.abs(a[2] - b[2])))) / sc
        out[f"M{M}"] = {"one_vs_two_nonconforming_rel": d}
        print(f"[split] slanted M={M}: one layer(d) vs two(d/2) on N=2|N=4 "
              f"= {d:.3e} (relative)", flush=True)
    RES["split"] = out


# ==========================================================================
def _tilted(no, ne, tilt, azim):
    c, s = np.cos(tilt), np.sin(tilt)
    d = np.array([s * np.cos(azim), s * np.sin(azim), c])
    return no ** 2 * np.eye(3) + (ne ** 2 - no ** 2) * np.outer(d, d)


def sec_mixed():
    """OOP over SCALAR over MAGNETIC, each on its OWN grid, against the
    union-grid shared cascade at a converged modal count."""
    out = {}
    P, WL = 1.05, 0.72
    e33 = _tilted(1.5, 1.7, np.deg2rad(32.0), np.deg2rad(18.0))
    oop2 = np.zeros((2, 2, 3, 3), dtype=_C)
    oop2[:] = np.diag([2.10, 2.10, 2.10]).astype(_C)
    oop2[0, :] = e33
    sca3 = np.full((3, 3), 2.10 + 0j)
    sca3[1, 1] = 5.0
    mag2 = np.full((2, 2), 2.6 + 0j)
    mu2 = np.full((2, 2), 1.0 + 0j)
    mu2[0, 0] = 1.35

    def up(c, k):
        return np.repeat(np.repeat(c, k, axis=0), k, axis=1)

    def union(M):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1)
        st.add_layer(0.21, eps_cell=up(oop2, 3))
        st.add_layer(0.16, eps_cell=up(sca3, 2))
        st.add_layer(0.13, eps_cell=up(mag2, 3), mu_cell=up(mu2, 3))
        st.set_source(WL, theta=0.23, phi=0.55)
        return solve(st, jones=False)

    ref4 = union(4)
    ref5 = union(5)
    ref_gap = float(max(np.max(np.abs(ref4[1] - ref5[1])),
                        np.max(np.abs(ref4[2] - ref5[2]))))
    out["union_selfgap_M4_M5"] = ref_gap
    print(f"[mixed] union reference (N=6) self-gap M=4 vs M=5: {ref_gap:.3e}",
          flush=True)
    lad = {}
    for Ms in ((6, 5, 6), (8, 6, 8), (10, 7, 10)):
        t0 = time.time()
        st = PMM2DStackPure(P, n_modes=max(Ms), n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.21, eps_cell=oop2, n_modes=Ms[0])
        st.add_layer(0.16, eps_cell=sca3, n_modes=Ms[1])
        st.add_layer(0.13, eps_cell=mag2, mu_cell=mu2, n_modes=Ms[2])
        st.set_source(WL, theta=0.23, phi=0.55)
        o, R, T = solve(st, jones=False)
        err = float(max(np.max(np.abs(R - ref5[1])),
                        np.max(np.abs(T - ref5[2]))))
        clo = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0)))
        lad[str(Ms)] = {"err_vs_union_M5": err, "closure": clo,
                        "wall_s": time.time() - t0}
        print(f"[mixed] per-layer M={Ms}: err vs union(M=5) {err:.4e}  "
              f"closure {clo:.3e}  {time.time()-t0:.1f}s", flush=True)
    out["ladder"] = lad
    RES["mixed"] = out


SECTIONS = {"bypass": sec_bypass, "oracle": sec_oracle, "split": sec_split,
            "mixed": sec_mixed}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v4_slant_mixed.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:          # a partial write from a crashed run
            old = {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
