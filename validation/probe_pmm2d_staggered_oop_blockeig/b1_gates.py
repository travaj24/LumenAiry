"""B1 -- the normal-incidence PARITY-sign block reduction for the PURE
staggered OUT-OF-PLANE region solve: structure residuals, agreement, fallback
bit-identity, the parity operator's own properties, and the fail-before.

Sections
--------
B0  the parity operator itself: ``J^2 = I`` and the eps-free mass maps.
B2  the structure residual ``max|R A R + A|/max|A|`` (and the B twin) on the
    ASSEMBLED pencil, carrying and violating cells -- the table the
    ``_STAG_BLOCK_TOL`` comment derives from.
B1  reduction ON vs OFF: R / T / Jones and the eigenvalue SET (symmetric
    Hausdorff, so a near-degenerate pair cannot fake a mismatch).
B3  fallback: an off-centre cell, an oblique mount and a parity-breaking
    tensor are BIT-IDENTICAL to symmetry=False (sha256 of R/T/Jones).
B5  fail-before: force the reduction through with ``tol = 1.0`` on a cell that
    does NOT carry the structure and show the answer is wrong by decades.

Run:
  cd /c/tmp/lum_oopfast && PYTHONPATH=/c/tmp/lum_oopfast OMP_NUM_THREADS=1 \\
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python validation/probe_pmm2d_staggered_oop_blockeig/b1_gates.py
"""
import hashlib
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    _stag_block_eig,
    _stag_parity_1d,
    _stag_parity_gauge,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUB, NSUP = 1.50, 1.0
K0 = 2.0 * np.pi / WL

TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
NREC = TIL.copy()                       # NON-reciprocal: e13 != e31, Hermitian
NREC[0, 2] = TIL[0, 2] + 0.30j
NREC[2, 0] = TIL[2, 0] - 0.30j
LOSSY = uniaxial_tensor(1.5 + 0.06j, 1.7 + 0.03j, np.deg2rad(35.0),
                        phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
ISO = 2.25 * np.eye(3, dtype=complex)

R = {"lumenairy": lumenairy.__file__, "version": lumenairy.__version__}


def tile(t, n):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = t
    return c


def centro_pillar(t, n):
    """A cell that IS its own parity image ``eps[i, j] == eps[n-1-i, n-1-j]``."""
    c = tile(AIR, n)
    if n == 2:
        c[0, 0] = t
        c[1, 1] = t
    else:                                # n = 3: centre (a fixed pixel) + pair
        c[1, 1] = t
        c[0, 0] = ISO
        c[2, 2] = ISO
    return c


def offcentre_pillar(t, n):
    """Its own parity image NOWHERE: one corner pixel only."""
    c = tile(AIR, n)
    c[0, 0] = t
    return c


def broken_tensor(n):
    """Parity-symmetric PATTERN, parity-BREAKING tensor grid: the two mirror
    pixels carry DIFFERENT out-of-plane tensors."""
    c = tile(AIR, n)
    if n == 2:
        c[0, 0] = TIL
        c[1, 1] = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0),
                                  phi=np.deg2rad(-70.0))
    else:
        c[1, 1] = TIL
        c[0, 0] = TIL
        c[2, 2] = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
    return c


def solver(cell, M, theta=0.0, phi=0.0):
    """Granet2DTransverseE built exactly as PMM2DStackPure.solve builds it
    (physical periods, k0 = 2 pi / wl, alpha0 = n sin(theta) ... k0)."""
    nre = float(np.real(NSUP))
    a0x = nre * np.sin(theta) * np.cos(phi) * K0
    a0y = nre * np.sin(theta) * np.sin(phi) * K0
    Nx = cell.shape[0]
    return Granet2DTransverseE(PX, PY, Nx, Nx, M, cell,
                               alpha0x=a0x, alpha0y=a0y, k0=K0)


def struct_resid(sol):
    """(dA, dB, engaged) for the assembled pencil."""
    g = _stag_parity_gauge(sol)
    if g is None:
        return None, None, False
    perm, r = g
    A, B = sol.Agen, sol.Bgen
    rr = r[:, None] * r[None, :]
    dA = (float(np.max(np.abs(rr * A[np.ix_(perm, perm)] + A)))
          / float(np.max(np.abs(A))))
    dB = (float(np.max(np.abs(rr * B[np.ix_(perm, perm)] - B)))
          / float(np.max(np.abs(B))))
    return dA, dB, _stag_block_eig(A, B, sol.q * sol.q, g) is not None


def rt_hash(res):
    h = hashlib.sha256()
    for a in res:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()[:16]


def dmax(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def hausdorff(a, b):
    """Symmetric set distance -- immune to the pairing ambiguity a plain sort
    introduces on a near-degenerate spectrum."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    d = np.abs(a[:, None] - b[None, :])
    return max(float(np.max(np.min(d, axis=1))),
               float(np.max(np.min(d, axis=0))))


# ===================================================================== B0
print("\n## B0  the parity operator's own properties")
print("   axis Nx  M   J^2 - I (Btilde / B)   mass-map (Mtt / Mbb)   "
      "deriv-odd <B|d|til>")
rows = []
for Nx in (2, 3, 4):
    for M in (5, 6, 7, 8):
        sol = solver(tile(TIL, Nx), M)
        b = sol.bx
        pt, st, pb, sb = _stag_parity_1d(b)
        Pt = np.zeros((pt.size, pt.size))
        Pt[pt, np.arange(pt.size)] = st
        Pb = np.zeros((pb.size, pb.size))
        Pb[pb, np.arange(pb.size)] = sb
        i2t = float(np.max(np.abs(Pt @ Pt - np.eye(pt.size))))
        i2b = float(np.max(np.abs(Pb @ Pb - np.eye(pb.size))))
        Mtt = b.mass(b.Btilde, b.Btilde)
        Mbb = b.mass(b.B, b.B)
        Cbt = b.mixed(b.B, b.Btilde)
        dmt = (float(np.max(np.abs(Pt.T @ Mtt @ Pt - Mtt)))
               / float(np.max(np.abs(Mtt))))
        dmb = (float(np.max(np.abs(Pb.T @ Mbb @ Pb - Mbb)))
               / float(np.max(np.abs(Mbb))))
        dcd = (float(np.max(np.abs(Pb.T @ Cbt @ Pt + Cbt)))
               / float(np.max(np.abs(Cbt))))
        print(f"   x    {Nx}  {M}   {i2t:.1e} / {i2b:.1e}          "
              f"{dmt:.2e} / {dmb:.2e}       {dcd:.2e}")
        rows.append(dict(Nx=Nx, M=M, J2t=i2t, J2b=i2b, mass_t=dmt,
                         mass_b=dmb, deriv_odd=dcd))
R["B0"] = rows

# ===================================================================== B2
print("\n## B2  structure residual on the ASSEMBLED pencil "
      "(the _STAG_BLOCK_TOL table)")
print("   cell                       grid  M  mount        "
      "|RAR+A|/|A|   |RBR-B|/|B|   engages")
rows = []
cases = []
for Nx in (2, 3):
    for M in (5, 6, 7, 8):
        cases += [
            ("uniform tilted uniaxial", tile(TIL, Nx), Nx, M, 0.0, 0.0),
            ("centro pillar", centro_pillar(TIL, Nx), Nx, M, 0.0, 0.0),
            ("centro NON-RECIPROCAL", centro_pillar(NREC, Nx), Nx, M,
             0.0, 0.0),
            ("centro LOSSY", centro_pillar(LOSSY, Nx), Nx, M, 0.0, 0.0),
        ]
for Nx in (2, 3):
    cases += [
        ("OFF-CENTRE pillar", offcentre_pillar(TIL, Nx), Nx, 6, 0.0, 0.0),
        ("parity-BREAKING tensor", broken_tensor(Nx), Nx, 6, 0.0, 0.0),
        ("uniform, OBLIQUE 25", tile(TIL, Nx), Nx, 6, np.deg2rad(25.0), 0.0),
        ("uniform, CONICAL 25/40", tile(TIL, Nx), Nx, 6, np.deg2rad(25.0),
         np.deg2rad(40.0)),
    ]
for name, cell, Nx, M, th, ph in cases:
    sol = solver(cell, M, th, ph)
    dA, dB, eng = struct_resid(sol)
    mount = "normal" if th == 0.0 else ("oblique" if ph == 0.0 else "conical")
    txt_A = "gauge=None" if dA is None else f"{dA:.3e}"
    txt_B = "" if dB is None else f"{dB:.3e}"
    print(f"   {name:26s} ({Nx},{Nx})  {M}  {mount:9s}  {txt_A:>12s}  "
          f"{txt_B:>12s}   {eng}")
    rows.append(dict(cell=name, Nx=Nx, M=M, mount=mount, dA=dA, dB=dB,
                     engaged=bool(eng)))
R["B2"] = rows
car = [r["dA"] for r in rows if r["engaged"]]
carB = [r["dB"] for r in rows if r["engaged"]]
vio = [r["dA"] for r in rows if r["dA"] is not None and not r["engaged"]]
print(f"   CARRYING envelope dA: {min(car):.2e} .. {max(car):.2e}  "
      f"dB: {min(carB):.2e} .. {max(carB):.2e}")
print(f"   VIOLATING (normal-incidence, gauge built) dA: "
      f"{min(vio):.2e} .. {max(vio):.2e}")
R["B2_envelope"] = dict(carrying=[min(car), max(car)],
                        carryingB=[min(carB), max(carB)],
                        violating=[min(vio), max(vio)])

# ===================================================================== B1
print("\n## B1  reduction ON vs OFF (agreement + eigenvalue SET)")
print("   cell                    grid  M   dR         dT         dJones     "
      "d(eig set)   min|q|/max|q|")
rows = []
for Nx in (2, 3):
    for M in (5, 6, 7):
        for name, cell in (("uniform tilted uniaxial", tile(TIL, Nx)),
                           ("centro pillar", centro_pillar(TIL, Nx)),
                           ("centro NON-RECIPROCAL", centro_pillar(NREC, Nx)),
                           ("centro LOSSY", centro_pillar(LOSSY, Nx))):
            on = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL,
                                        degree=M, n_orders=5, symmetry=True)
            off = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL,
                                         degree=M, n_orders=5, symmetry=False)
            dR, dT = dmax(on[1], off[1]), dmax(on[2], off[2])
            dJ = dmax(on[3], off[3])
            sol = solver(cell, M)
            g = _stag_parity_gauge(sol)
            fac = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q, g)
            assert fac is not None, (name, Nx, M)
            six_on = _region_modes_oop(sol, symmetry=True)
            six_off = _region_modes_oop(sol, symmetry=False)
            dset = hausdorff(np.concatenate([six_off[2], six_off[5]]),
                             np.concatenate([six_on[2], six_on[5]]))
            rq = (float(np.min(np.abs(fac[0])))
                  / float(np.max(np.abs(fac[0]))))
            print(f"   {name:23s} ({Nx},{Nx})  {M}   {dR:.3e}  {dT:.3e}  "
                  f"{dJ:.3e}  {dset:.3e}    {rq:.2e}")
            rows.append(dict(cell=name, Nx=Nx, M=M, dR=dR, dT=dT, dJones=dJ,
                             d_eigset=dset, gam_ratio=rq))
R["B1"] = rows
print(f"   WORST over {len(rows)} rows: dR {max(r['dR'] for r in rows):.3e}  "
      f"dT {max(r['dT'] for r in rows):.3e}  "
      f"dJones {max(r['dJones'] for r in rows):.3e}  "
      f"d(eig set) {max(r['d_eigset'] for r in rows):.3e}")
print(f"   min|q|/max|q| over the same rows: "
      f"{min(r['gam_ratio'] for r in rows):.2e} .. "
      f"{max(r['gam_ratio'] for r in rows):.2e}")
R["B1_worst"] = dict(dR=max(r["dR"] for r in rows),
                     dT=max(r["dT"] for r in rows),
                     dJones=max(r["dJones"] for r in rows),
                     d_eigset=max(r["d_eigset"] for r in rows),
                     gam_lo=min(r["gam_ratio"] for r in rows),
                     gam_hi=max(r["gam_ratio"] for r in rows))

# ===================================================================== B3
print("\n## B3  fallback is BIT-IDENTICAL to symmetry=False (sha256)")
print("   case                          grid  M  mount     "
      "sha(ON)          sha(OFF)         same   |RAR+A|/|A|")
rows = []
fb = [("OFF-CENTRE pillar", offcentre_pillar(TIL, 2), 2, 6, 0.0, 0.0),
      ("OFF-CENTRE pillar", offcentre_pillar(TIL, 3), 3, 6, 0.0, 0.0),
      ("parity-BREAKING tensor", broken_tensor(2), 2, 6, 0.0, 0.0),
      ("parity-BREAKING tensor", broken_tensor(3), 3, 6, 0.0, 0.0),
      ("uniform, OBLIQUE 25", tile(TIL, 2), 2, 6, np.deg2rad(25.0), 0.0),
      ("centro pillar, CONICAL 25/40", centro_pillar(TIL, 3), 3, 6,
       np.deg2rad(25.0), np.deg2rad(40.0))]
for name, cell, Nx, M, th, ph in fb:
    on = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M,
                                n_orders=5, theta=th, phi=ph, symmetry=True)
    off = pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL, degree=M,
                                 n_orders=5, theta=th, phi=ph, symmetry=False)
    ha, hb = rt_hash(on[1:]), rt_hash(off[1:])
    sol = solver(cell, M, th, ph)
    dA, _dB, eng = struct_resid(sol)
    mount = "normal" if th == 0.0 else ("oblique" if ph == 0.0 else "conical")
    txt = "gauge=None" if dA is None else f"{dA:.3e}"
    print(f"   {name:29s} ({Nx},{Nx})  {M}  {mount:8s}  {ha}  {hb}  "
          f"{str(ha == hb):5s}  {txt}")
    rows.append(dict(case=name, Nx=Nx, M=M, mount=mount, sha_on=ha,
                     sha_off=hb, identical=ha == hb, dA=dA,
                     engaged=bool(eng)))
R["B3"] = rows

# ===================================================================== B5
print("\n## B5  FAIL-BEFORE: force the reduction with tol = 1.0 on cells "
      "that do NOT carry the structure")
print("   case                          grid  M   |RAR+A|/|A|   "
      "d(eig set) forced vs dense")
rows = []
for name, cell, Nx, M, th, ph in fb:
    sol = solver(cell, M, th, ph)
    g = _stag_parity_gauge(sol)
    if g is None:
        print(f"   {name:29s} ({Nx},{Nx})  {M}   gauge=None (the "
              f"necessary-condition gate refuses before the pencil is built)")
        rows.append(dict(case=name, Nx=Nx, M=M, gauge=None))
        continue
    dA, _dB, _e = struct_resid(sol)
    forced = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q, g, tol=1.0)
    if forced is None:
        print(f"   {name:29s} ({Nx},{Nx})  {M}   {dA:.3e}   "
              f"refused for another reason")
        rows.append(dict(case=name, Nx=Nx, M=M, dA=dA, forced=False))
        continue
    six = _region_modes_oop(sol, symmetry=False)
    dset = hausdorff(np.concatenate([six[2], six[5]]), -1j * forced[0])
    print(f"   {name:29s} ({Nx},{Nx})  {M}   {dA:.3e}   {dset:.3e}")
    rows.append(dict(case=name, Nx=Nx, M=M, dA=dA, forced=True,
                     d_eigset=dset))
R["B5"] = rows

with open(os.path.join(OUT, "b1_gates.json"), "w") as f:
    json.dump(R, f, indent=1, default=str)
print("\nwrote results/b1_gates.json")
