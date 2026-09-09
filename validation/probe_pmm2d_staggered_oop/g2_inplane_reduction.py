"""GATE 2 -- the in-plane reduction of the integrated out-of-plane path.

Two claims, two arms, same build:

  (a) REDUCTION.  With the cross terms EXACTLY zero the 4 q^2 generator must
      reproduce the Stage-A 2 q^2 path.  The dispatch is forced by patching
      ``twod_staggered._tile_is_offplane`` to True, so the SAME cell runs both
      arms.  This is what pins ``_OOP_H_GAUGE`` (the constant relating the
      generator's ``G = i Z0 H`` state to the Eq.-25 partner the half-spaces
      use): a wrong value survives a pure out-of-plane stack and destroys a
      MIXED one, which is exactly this comparison.
  (b) DISPATCH FLOOR.  A 1e-16 stray in the xz slot must stay on the in-plane
      path and be BIT-IDENTICAL to the clean cell; a 1e-3 one must route to the
      generator.

Run:
  cd /c/tmp/lum_aniso_oopint && PYTHONPATH=/c/tmp/lum_aniso_oopint \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/g2_inplane_reduction.py
"""
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
DEPTH = 0.4
NSUB, NSUP = 1.5, 1.0

_LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_LC = np.array(_LC, dtype=complex)
_LC[0, 2] = _LC[1, 2] = _LC[2, 0] = _LC[2, 1] = 0.0        # EXACTLY zero
_GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)
_ISO = 4.0 * np.eye(3, dtype=complex)
_LOSSY = np.array([[2.25 + 0.2j, 0.0, 0.0], [0.0, 2.6 + 0.2j, 0.0],
                   [0.0, 0.0, 2.4 + 0.2j]], dtype=complex)


def cell(host, pillar, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:, :] = host
    c[0, 0] = pillar
    return c


def solve(ec, th, ph, M, force_oop):
    orig = TS._tile_is_offplane
    if force_oop:
        TS._tile_is_offplane = lambda t: True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TS.pmm_jones_2d_staggered(PX, PY, ec, NSUB, NSUP, DEPTH, WL,
                                             degree=M, n_orders=3,
                                             theta=th, phi=ph)
    finally:
        TS._tile_is_offplane = orig


def main():
    pc.banner("GATE 2 -- in-plane reduction of the 4 q^2 generator")
    res = {}
    for name, host, pil in (("in-plane uniaxial pillar", _ISO, _LC),
                            ("gyrotropic pillar", _ISO, _GYRO),
                            ("lossy diagonal pillar", _ISO, _LOSSY),
                            ("uniform in-plane uniaxial", _LC, _LC)):
        ec = cell(host, pil)
        for mount, th, ph in (("normal", 0.0, 0.0),
                              ("conical 20/35", np.deg2rad(20.0),
                               np.deg2rad(35.0))):
            for M in (5, 6):
                o1, R1, T1, J1 = solve(ec, th, ph, M, False)
                o2, R2, T2, J2 = solve(ec, th, ph, M, True)
                dR = float(np.max(np.abs(R1 - R2)))
                dT = float(np.max(np.abs(T1 - T2)))
                dJ = float(np.max(np.abs(J1 - J2)))
                print(f"  {name:26s} {mount:14s} M={M}  dR = {dR:.3e}  "
                      f"dT = {dT:.3e}  dJones = {dJ:.3e}")
                res[f"{name}|{mount}|M{M}"] = dict(dR=dR, dT=dT, dJ=dJ)

    print("\n  spectra: the generator's eigenvalue SET vs the E-form's "
          "{+q, -q}")
    for name, host, pil in (("in-plane uniaxial pillar", _ISO, _LC),
                            ("gyrotropic pillar", _ISO, _GYRO)):
        ec = cell(host, pil)
        for mount, a0x, a0y in (("normal", 0.0, 0.0),
                                ("oblique", 0.7, 0.4)):
            k0 = 2 * np.pi / WL
            sol_in = TS.Granet2DTransverseE(PX, PY, 2, 2, 6, ec,
                                            alpha0x=a0x, alpha0y=a0y, k0=k0)
            orig = TS._tile_is_offplane
            TS._tile_is_offplane = lambda t: True
            try:
                sol_oop = TS.Granet2DTransverseE(PX, PY, 2, 2, 6, ec,
                                                 alpha0x=a0x, alpha0y=a0y,
                                                 k0=k0)
                Wf, Vf, lf, Wb, Vb, lb = TS._region_modes_oop(sol_oop)
            finally:
                TS._tile_is_offplane = orig
            _W, _V, lam_in, _g2 = TS._region_modes(sol_in)
            ref = np.concatenate([1j * lam_in, -1j * lam_in])
            got = np.concatenate([1j * lf, 1j * lb])
            d = float(np.max(np.min(np.abs(got[:, None] - ref[None, :]),
                                    axis=1)))
            print(f"  {name:26s} {mount:9s} dim {got.size}  max dist to the "
                  f"nearest E-form eigenvalue = {d:.3e}")
            res[f"spectra|{name}|{mount}"] = d

    print("\n  DISPATCH FLOOR (relative 1e-12 * scale)")
    base = cell(_ISO, _LC)
    stray = base.copy()
    stray[0, 0, 0, 2] = 1e-16
    o1, R1, T1, J1 = solve(base, 0.0, 0.0, 5, False)
    o2, R2, T2, J2 = solve(stray, 0.0, 0.0, 5, False)
    bits = (R1.tobytes() == R2.tobytes() and T1.tobytes() == T2.tobytes()
            and J1.tobytes() == J2.tobytes())
    sol = TS.Granet2DTransverseE(PX, PY, 2, 2, 5, stray, k0=2 * np.pi / WL)
    big = base.copy()
    big[0, 0, 0, 2] = 1e-3
    sol_big = TS.Granet2DTransverseE(PX, PY, 2, 2, 5, big, k0=2 * np.pi / WL)
    print(f"  1e-16 stray: offplane = {sol.offplane}, BIT-IDENTICAL R/T/Jones "
          f"to the clean cell = {bits}")
    print(f"  1e-3  stray: offplane = {sol_big.offplane}")
    res["floor"] = dict(stray_offplane=bool(sol.offplane),
                        bit_identical=bool(bits),
                        big_offplane=bool(sol_big.offplane))
    with open(os.path.join(OUT, "g2_inplane_reduction.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote results/g2_inplane_reduction.json")


if __name__ == "__main__":
    main()
