"""M6 -- spurious census in the CASCADE + energy closure at three depths.

The dispersion census (M1 T2) says what the spectrum contains; this says
whether the cascade SURVIVES it.  A mis-classified growing mode shows as
``exp(+|Re lam| k0 L)`` blow-up, which grows with depth -- so the test is a
DEPTH LADDER (0.25, 1, 3 wavelengths), not a single solve.

Measured at each depth, for each candidate:
  * ``|R + T - 1|`` for a HERMITIAN (lossless) OOP tensor -- two-sided: it must
    be small AND must not drift with depth;
  * ``1 - R - T`` (the absorbed fraction) for a lossy tensor -- must stay in
    ``[0, 1]``;
  * the largest S-matrix entry, and the largest forward-mode growth factor
    ``max exp(-Re(lam) k0 L)`` (>1 means a growing mode was classified forward);
  * the forward/backward split counts and the flux gap that separates them.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m6_cascade.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
NSUB, NSUP = 1.5, 1.0
DEPTHS = (0.25, 1.0, 3.0)


def pillar(er, eg):
    e = np.zeros((2, 2, 3, 3), dtype=complex)
    e[:, :] = eg
    e[0, 0] = er
    return e


def main():
    pc.banner("M6 -- cascade stability + closure")
    R = {}
    lossless = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
    lossy = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0, loss=0.15)
    herm = float(np.max(np.abs(lossless - lossless.conj().T)))
    print(f"  lossless tensor Hermiticity |eps - eps^H| = {herm:.2e} "
          f"(exactly Hermitian -> closure is a two-sided claim)")
    cases = (("uniform lossless OOP", pc.tile(lossless, 2, 2), True),
             ("pillar lossless OOP", pillar(lossless, np.eye(3) + 0j), True),
             ("pillar LOSSY OOP", pillar(lossy, np.eye(3) + 0j), False))
    for cname, ec, is_lossless in cases:
        print(f"\n=== {cname} ===")
        for th, ph in ((0.0, 0.0), (np.deg2rad(25.0), np.deg2rad(40.0))):
            print(f"  theta={np.rad2deg(th):.0f} phi={np.rad2deg(ph):.0f}")
            print("   cand  M  depth/wl   sum R      sum T     R+T-1      "
                 " max|S|    max fwd growth  fwd/bwd  flux gap")
            for cand in ("a", "d"):
                for M in (6, 7):
                    for dep in DEPTHS:
                        try:
                            o, Rm, Tm, J, ex = pc.solve_slab(
                                PX, PY, ec, NSUB, NSUP, dep, WL, M=M,
                                theta=th, phi=ph, candidate=cand,
                                return_modes=True)
                        except Exception as exc:          # noqa: BLE001
                            print(f"    ({cand}) {M:2d}  {dep:5.2f}  FAILED "
                                  f"{type(exc).__name__}: {exc}")
                            continue
                        Rt, Tt = Rm.sum(axis=1), Tm.sum(axis=1)
                        lam_f = -1j * ex["qv"][ex["fidx"]]
                        grow = float(np.max(np.exp(
                            -np.real(lam_f) * 2 * np.pi * dep)))
                        fl = pc.modal_flux(ex["lay"], ex["W"], ex["V"])
                        fl = fl / max(float(np.max(np.abs(fl))), 1e-30)
                        gap = (float(np.min(fl[ex["fidx"]])),
                               float(np.max(fl[ex["bidx"]])))
                        maxS = float(np.max(np.abs(J)))
                        print(f"    ({cand}) {M:2d}  {dep:5.2f}   "
                              f"{Rt[0]:.7f}  {Tt[0]:.7f}  "
                              f"{Rt[0]+Tt[0]-1:+.2e}   {maxS:.2e}  "
                              f"{grow:12.4e}   {ex['fidx'].size}/"
                              f"{ex['bidx'].size}  "
                              f"[{gap[0]:+.2e},{gap[1]:+.2e}]")
                        R[f"{cname}|th{int(np.rad2deg(th))}|{cand}|M{M}|"
                          f"d{dep}"] = dict(
                            R=float(Rt[0]), T=float(Tt[0]),
                            closure=float(np.max(np.abs(Rt + Tt - 1))),
                            absorbed=float(np.min(1 - Rt - Tt)),
                            grow=grow, nfwd=int(ex["fidx"].size),
                            nbwd=int(ex["bidx"].size))
    with open(os.path.join(OUT, "m6_cascade.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m6_cascade.json")


if __name__ == "__main__":
    main()
