"""M3 -- GENUINELY NON-CONFORMING grids on a 2-D (pillar) pair, against the
union-grid cascade on the COMMON REFINEMENT.

Granet's lattice is uniform, so the common refinement of two lattices N_a, N_b
is LCM(N_a, N_b).  Layer A's pillar has width 1/2 (representable on 2 and 6),
layer B's has width 1/3 (representable on 3 and 6) -- LCM = 6.  Holding the
reference at N=6 lets FOUR grid pairs be scored against ONE reference, which
separates the resolution effect from the non-conforming remainder:

    (6,6) conforming     -> identity control (mortar FORCED, not bypassed)
    (2,6) nested         -> only A coarse; A's walls lie on B's lattice
    (6,3) nested         -> only B coarse; B's walls lie on A's lattice
    (2,3) NON-conforming -> neither layer's walls lie on the other's lattice

The uniform lattice also fixes which pairs are ADMISSIBLE at all: a 1/2-wide
pillar exists on N in {2,4,6,...} and a 1/3-wide one on N in {3,6,...}, so
(3,6) is NOT a coarsening of this stack -- it is a DIFFERENT geometry (A's
pillar would become 2/3 wide).  Enumerating admissible per-layer grids is part
of the API contract, not a free choice.

Lossless closure is scored TWO-SIDED on both arms, and the last block repeats
the ladder with a HERMITIAN (gyrotropic, lossless-but-anisotropic) tensor in
layer A."""
import json
import sys
import time
import warnings
import numpy as np
from mortar2d import guard, MortarStack2D, refine_cell
print("lumenairy:", guard(), flush=True)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
PX = PY = 1.2e-6
WL = 0.85e-6
TH, PH = 0.18, 0.35
NORD = 2
tA, tB = 0.16e-6, 0.13e-6
EPS_H, EPS_P = 2.25, 9.0
GYRO = np.array([[6.0, 0.9j, 0.0], [-0.9j, 6.0, 0.0], [0.0, 0.0, 5.0]],
                dtype=complex)


def pillar(N, k, host=EPS_H, core=EPS_P):
    c = np.full((N, N), host + 0j)
    c[:k, :k] = core
    return c


def pillar33(N, k, host=EPS_H, core=GYRO):
    c = np.empty((N, N, 3, 3), dtype=complex)
    c[:, :] = np.diag([host, host, host]).astype(complex)
    c[:k, :k] = core
    return c


def obs(R, T):
    return np.concatenate([R.ravel(), T.ravel()])


def go(tensor, Ms):
    mk = pillar33 if tensor else pillar
    # union reference on N = 6 (A pillar 3/6 wide, B pillar 2/6 wide)
    out = []
    for M in Ms:
        t0 = time.perf_counter()
        ref = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
        ref.add_layer(tA, eps_cell=mk(6, 3))
        ref.add_layer(tB, eps_cell=pillar(6, 2))
        ref.set_source(WL, theta=TH, phi=PH)
        o0, R0, T0, J0 = ref.solve()
        t_ref = time.perf_counter() - t0
        sc = max(float(np.max(R0)), float(np.max(T0)))
        c0 = max(R0[p].sum() + T0[p].sum() - 1.0 for p in (0, 1))
        print(f"  [M={M}] union N=6 reference {t_ref:7.1f}s  "
              f"R+T-1 {c0:+.2e}", flush=True)
        for (Na, Nb) in ((6, 6), (2, 6), (6, 3), (2, 3)):
            ka = {6: 3, 2: 1}[Na]
            kb = {6: 2, 3: 1}[Nb]
            t0 = time.perf_counter()
            s = MortarStack2D(PX, PY, n_modes=M, n_orders=NORD)
            s.add_layer(tA, eps_cell=mk(Na, ka))
            s.add_layer(tB, eps_cell=pillar(Nb, kb))
            s.set_source(WL, theta=TH, phi=PH)
            o1, R1, T1, J1 = s.solve(force_mortar=(Na == Nb))
            dt = time.perf_counter() - t0
            assert np.array_equal(o0, o1)
            d = float(np.max(np.abs(obs(R0, T0) - obs(R1, T1)))) / sc
            dJ = float(np.max(np.abs(J0 - J1))) / float(np.max(np.abs(J0)))
            c1 = max(R1[p].sum() + T1[p].sum() - 1.0 for p in (0, 1))
            c1m = min(R1[p].sum() + T1[p].sum() - 1.0 for p in (0, 1))
            out.append(dict(tensor=tensor, M=M, Na=Na, Nb=Nb, d=d, dJ=dJ,
                            clo_ref=float(c0), clo_hi=float(c1),
                            clo_lo=float(c1m), t_ref=t_ref, t=dt))
            print(f"      grids({Na},{Nb})  d(R,T) {d:9.2e}  dJ {dJ:9.2e} | "
                  f"R+T-1 in [{c1m:+.1e}, {c1:+.1e}] | {dt:6.1f}s "
                  f"({t_ref/dt:5.2f}x)", flush=True)
    return out


Ms = [int(x) for x in (sys.argv[1:] or [4, 5, 6])]
print("SCALAR pillar pair")
rows = go(False, Ms)
import json as _j
_j.dump(rows, open("validation/probe_pmm2d_staggered_mortar/m3_nonconforming.json", "w"), indent=1)
print("\nHERMITIAN (gyrotropic) tensor in layer A -- lossless, two-sided")
rows += go(True, Ms[:2])
json.dump(rows, open(
    "validation/probe_pmm2d_staggered_mortar/m3_nonconforming.json", "w"),
    indent=1)
