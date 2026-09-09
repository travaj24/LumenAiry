"""M7 -- cost and the normal-incidence block structure.

T1  DIMENSION + EIG WALL TIME of the three paths at matched ``(Nx, M)``:
    in-plane E-form ``2 q^2``, candidate (a) ``4 q^2``, candidate (d) ``6 q^2``
    (a QZ, because the leading coefficient ``P2`` is singular).

T2  Does ``_generator_block_eig``'s ANTI-COMMUTING INVOLUTION exist on the
    staggered generator at NORMAL incidence?  The Fourier version uses
    ``R = S (I4 (x) F)`` with ``F`` the order flip ``(m, n) -> (-m, -n)`` and
    ``S = diag(I, I, -I, -I)``.  The staggered analogue of ``F`` is the exact
    PARITY signed permutation of the modified-Legendre dofs: ``x -> d - x``
    maps segment ``s -> N-1-s``, swaps the two half-hats and multiplies the
    degree-``a`` bubble by ``(-1)^a``, so the hats permute (node ``k -> N-k``)
    and the bubbles carry a sign -- exactly a signed permutation, PROVIDED
    ``tau = 1`` (normal incidence) and the cell is centro-symmetric.  This
    measures the structure residual ``||R A R + A|| / ||A||`` (and
    ``||R B R - B|| / ||B||``) on the ASSEMBLED pencil, the same
    verify-then-use discipline ``_generator_block_eig`` applies.

Run:
  cd /c/tmp/lum_aniso_oop && PYTHONPATH=/c/tmp/lum_aniso_oop \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/m7_cost.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
K0 = 2.0 * np.pi / WL


def parity_maps(b: pc.Basis1D):
    """``(Ptil, Pb)``: the exact PARITY (``x -> d - x``) signed permutations of
    the Btilde and B global sets.  Valid only at ``tau = 1``."""
    N, M = b.N, b.M
    # local: Ltilde_a(-u) expanded in Ltilde_b(u)
    T = np.zeros((M, M))
    T[0, 1] = T[1, 0] = 1.0
    for a in range(2, M):
        T[a, a] = (-1.0) ** a
    def stencil_parity(S):
        out = np.zeros_like(S)
        for s in range(N):
            out[N - 1 - s] = T @ S[s]
        return out
    def to_matrix(gset):
        A = np.array(gset)                       # (dim, N, M)
        flat = A.reshape(A.shape[0], -1)
        img = np.array([stencil_parity(S) for S in A]).reshape(A.shape[0], -1)
        # solve  P such that  img = P^T flat   (columns of the global set)
        sol, *_ = np.linalg.lstsq(flat.T, img.T, rcond=None)
        res = float(np.max(np.abs(flat.T @ sol - img.T)))
        return sol, res
    Pt, rt = to_matrix(b.Btilde)
    Pb, rb = to_matrix(b.B)
    return Pt, Pb, max(rt, rb)


def main():
    pc.banner("M7 -- cost and normal-incidence block structure")
    R = {}
    TIL = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)

    print("\n## T1  dimension and eig wall time (single-threaded)")
    print("   Nx  M    q   E-form 2q^2   (a) 4q^2      (d) 6q^2      "
          "ratio a/E   ratio d/a")
    for Nx in (2, 3):
        for M in (5, 6, 7, 8):
            q = Nx * (M - 1)
            if 6 * q * q > 3000:
                continue
            cell = pc.StaggeredCell(PX, PX, pc.tile(TIL, Nx, Nx), M, K0,
                                    0.3, 0.2)
            t = {}
            t0 = time.perf_counter()
            pc.eform_modes(cell)
            t["e"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            pc.modes_a(cell)
            t["a"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            pc.modes_d(cell)
            t["d"] = time.perf_counter() - t0
            print(f"   {Nx}  {M:2d} {q:4d}  {2*q*q:5d} {t['e']:7.2f}s  "
                  f"{4*q*q:5d} {t['a']:7.2f}s  {6*q*q:5d} {t['d']:7.2f}s   "
                  f"{t['a']/t['e']:8.1f}x  {t['d']/t['a']:8.1f}x")
            R[f"T1_Nx{Nx}_M{M}"] = dict(q=q, t_eform=t["e"], t_a=t["a"],
                                        t_d=t["d"],
                                        dim_eform=2 * q * q, dim_a=4 * q * q,
                                        dim_d=6 * q * q)

    print("\n## T2  anti-commuting involution on the staggered generator "
          "(normal incidence)")
    print("   cell               Nx  M   parity-map residual   "
          "||R A R + A||/||A||   ||R B R - B||/||B||")
    for kind in ("uniform OOP", "centro pillar OOP", "off-centre pillar OOP"):
        for Nx, M in ((2, 6), (2, 7), (4, 5)):
            ec = np.zeros((Nx, Nx, 3, 3), dtype=complex)
            ec[:, :] = np.eye(3)
            if kind == "uniform OOP":
                ec[:, :] = TIL
            elif kind == "centro pillar OOP":
                if Nx % 2:
                    continue
                ec[:, :] = np.eye(3)
                ec[Nx // 2 - 1:Nx // 2 + 1, Nx // 2 - 1:Nx // 2 + 1] = TIL
            else:
                ec[:, :] = np.eye(3)
                ec[0, 0] = TIL
            cell = pc.StaggeredCell(PX, PX, ec, M, K0, 0.0, 0.0)
            Ptx, Pbx, rx = parity_maps(cell.bx)
            Pty, Pby, ry = parity_maps(cell.by)
            # component spaces: V1 = B(x) x Btil(y), V2 = Btil(x) x B(y)
            P1 = np.kron(Pty, Pbx)
            P2 = np.kron(Pby, Ptx)
            qq = cell.qq
            Pst = np.zeros((4 * qq, 4 * qq), dtype=complex)
            Pst[:qq, :qq] = P1
            Pst[qq:2 * qq, qq:2 * qq] = P2
            Pst[2 * qq:3 * qq, 2 * qq:3 * qq] = P2      # G1 in V2
            Pst[3 * qq:, 3 * qq:] = P1                  # G2 in V1
            S = np.diag(np.concatenate([np.ones(2 * qq), -np.ones(2 * qq)]))
            Rop = S @ Pst
            A, B, _, _ = pc.generator_a(cell)
            dA = (float(np.max(np.abs(Rop @ A @ Rop + A)))
                  / float(np.max(np.abs(A))))
            dB = (float(np.max(np.abs(Rop @ B @ Rop - B)))
                  / float(np.max(np.abs(B))))
            inv = float(np.max(np.abs(Rop @ Rop - np.eye(4 * qq))))
            print(f"   {kind:20s} {Nx}  {M:2d}   {max(rx, ry):.2e}  "
                  f"(R^2-I: {inv:.1e})   {dA:.3e}          {dB:.3e}")
            R[f"T2_{kind}_Nx{Nx}_M{M}"] = dict(parity_res=max(rx, ry),
                                               involution=inv, dA=dA, dB=dB)

    print("\n## T2b  if the structure holds: one 2q^2 eig instead of 4q^2?")
    cell = pc.StaggeredCell(PX, PX, pc.tile(TIL, 2, 2), 7, K0, 0.0, 0.0)
    Ptx, Pbx, _ = parity_maps(cell.bx)
    Pty, Pby, _ = parity_maps(cell.by)
    qq = cell.qq
    P1 = np.kron(Pty, Pbx)
    P2 = np.kron(Pby, Ptx)
    Pst = np.zeros((4 * qq, 4 * qq), dtype=complex)
    Pst[:qq, :qq] = P1
    Pst[qq:2 * qq, qq:2 * qq] = P2
    Pst[2 * qq:3 * qq, 2 * qq:3 * qq] = P2
    Pst[3 * qq:, 3 * qq:] = P1
    S = np.diag(np.concatenate([np.ones(2 * qq), -np.ones(2 * qq)]))
    Rop = S @ Pst
    ev = np.linalg.eigvals(Rop)
    npos = int(np.sum(np.real(ev) > 0))
    print(f"   involution eigenvalues: {npos} at +1, {ev.size - npos} at -1 "
          f"(each sector 2q^2 = {2*qq}?  {'YES' if npos == 2*qq else 'NO'})")
    R["T2b_sector"] = dict(npos=npos, target=int(2 * qq))

    with open(os.path.join(OUT, "m7_cost.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    print("\nwrote results/m7_cost.json")


if __name__ == "__main__":
    main()
