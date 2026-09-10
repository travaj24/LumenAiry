"""V3 -- the INVOLUTION, re-derived from the assembly rather than read.

Three independent things are measured here:

A. The 1-D parity maps ``_stag_parity_1d`` returns are signed permutations with
   ``J^2 = I`` EXACTLY, and they act on the eps-free 1-D primitives the way the
   derivation claims: the two like-set MASSES are parity-EVEN and the directed
   derivative bracket ``<B | d Btilde>`` is parity-ODD.

B. The FOUR 2-D spaces (V1, V2, V3, Vw) carry the krons of those maps, and on
   the ASSEMBLED blocks of ``_assemble_oop`` -- rebuilt HERE from the same
   public primitives, never imported from the solver -- the NINE eps-weighted
   masses ``A11..A33`` (plus the three block Grams) are parity-EVEN while the
   FOUR single-derivative blocks ``P13, P23, CwE1, CwE2`` are parity-ODD.
   From those twelve facts the sign pattern is DERIVED symbolically by pushing
   the two eliminations through, and then CHECKED against the shipped pencil.

C. Every WRONG sign pattern is tried: all 16 block sign vectors on the right
   permutation, all 16 on a wrong permutation pattern, and four corruptions of
   the 1-D map itself.  Each must fail the structure check by decades.

Usage: python v3_involution.py
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import vfix as F  # noqa: E402

F.assert_arm("C:/tmp/lum_vacc")

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    _OOP_ROT_SIGN,
    Granet2DTransverseE,
    _stag_parity_1d,
)

K0 = 2.0 * np.pi / F.WL


def solver(cell, M):
    n = np.asarray(cell).shape[0]
    return Granet2DTransverseE(F.PX, F.PY, n, n, M, cell, k0=K0)


def signed_perm_matrix(perm, sign):
    P = np.zeros((perm.size, perm.size))
    P[perm, np.arange(perm.size)] = sign
    return P


def rel(Mm, Nn):
    d = float(np.max(np.abs(Mm - Nn)))
    s = float(np.max(np.abs(Mm)))
    return d / s if s else d


# --------------------------------------------------------------------------- #
# A. the 1-D maps
# --------------------------------------------------------------------------- #
def part_a():
    print("=" * 92)
    print("A. the 1-D parity maps: J^2 = I exactly, masses EVEN, derivative ODD")
    print("=" * 92)
    rows = []
    t = F.tensors()
    print(f"{'Nx':>3s} {'M':>3s} {'|Pt^2-I|':>9s} {'|Pb^2-I|':>9s} "
          f"{'Mtt EVEN':>10s} {'Mbb EVEN':>10s} {'dbt ODD':>10s} "
          f"{'dbt EVEN(wrong)':>16s}")
    for Nx in (2, 3, 4, 5):
        for M in (4, 5, 6, 7, 8):
            b = solver(F.tile(t["lc"], Nx), M).bx
            pt, st, pb, sb = _stag_parity_1d(b)
            Pt = signed_perm_matrix(pt, st)
            Pb = signed_perm_matrix(pb, sb)
            e_t = float(np.max(np.abs(Pt @ Pt - np.eye(pt.size))))
            e_b = float(np.max(np.abs(Pb @ Pb - np.eye(pb.size))))
            Mtt = b.mass(b.Btilde, b.Btilde)
            Mbb = b.mass(b.B, b.B)
            Cbt = b.mixed(b.B, b.Btilde)
            m_t = rel(Pt.T @ Mtt @ Pt, Mtt)
            m_b = rel(Pb.T @ Mbb @ Pb, Mbb)
            d_o = rel(Pb.T @ Cbt @ Pt, -Cbt)
            d_w = rel(Pb.T @ Cbt @ Pt, Cbt)
            rows.append(dict(Nx=Nx, M=M, PtPt=e_t, PbPb=e_b, mass_t=m_t,
                             mass_b=m_b, deriv_odd=d_o, deriv_even_wrong=d_w))
            print(f"{Nx:3d} {M:3d} {e_t:9.1e} {e_b:9.1e} {m_t:10.2e} "
                  f"{m_b:10.2e} {d_o:10.2e} {d_w:16.2e}")
    print()
    print(f"J^2 = I exactly on {sum(1 for r in rows if r['PtPt'] == 0.0 and r['PbPb'] == 0.0)}"
          f"/{len(rows)} (Nx, M) combinations")
    print(f"mass EVEN envelope   {min(min(r['mass_t'], r['mass_b']) for r in rows):.2e}"
          f" .. {max(max(r['mass_t'], r['mass_b']) for r in rows):.2e}")
    print(f"deriv ODD envelope   {min(r['deriv_odd'] for r in rows):.2e}"
          f" .. {max(r['deriv_odd'] for r in rows):.2e}")
    print(f"deriv EVEN (the WRONG sign) {min(r['deriv_even_wrong'] for r in rows):.2e}"
          f" .. {max(r['deriv_even_wrong'] for r in rows):.2e}  <- must be O(1)")
    return rows


# --------------------------------------------------------------------------- #
# B. the 2-D blocks, rebuilt here, and the DERIVED sign pattern
# --------------------------------------------------------------------------- #
def rebuild_blocks(sol):
    """Rebuild the twelve eps-free / eps-weighted blocks of ``_assemble_oop``
    from the SAME public primitives the assembly uses -- so the parity table
    below is a statement about the operators, not about a cached attribute."""
    bx, by = sol.bx, sol.by
    k0 = sol.k0
    sol._axis_mats()
    e = sol.eps_cell
    rot = _OOP_ROT_SIGN
    comp = dict(
        e11=e[..., 0, 0], e12=e[..., 0, 1], e13=rot * e[..., 0, 2],
        e21=e[..., 1, 0], e22=e[..., 1, 1], e23=rot * e[..., 1, 2],
        e31=rot * e[..., 2, 0], e32=rot * e[..., 2, 1], e33=e[..., 2, 2])
    Mtt_x, Mtt_y = sol.Mtt_x, sol.Mtt_y
    Mbb_x, Mbb_y = sol.Mbb_x, sol.Mbb_y
    dbt_x = bx.mixed(bx.B, bx.Btilde) / k0
    dbt_y = by.mixed(by.B, by.Btilde) / k0
    ew = sol._eps_weighted
    xB, xT, yB, yT = bx.B, bx.Btilde, by.B, by.Btilde
    mx, my = bx.m_ref, by.m_ref
    blocks = {
        # eps-free geometry: (matrix, test space, trial space, expected parity)
        "Ggram1": (np.kron(Mtt_y, Mbb_x), "V1", "V1", +1),
        "Ggram2": (np.kron(Mbb_y, Mtt_x), "V2", "V2", +1),
        "Gw": (np.kron(Mbb_y, Mbb_x), "Vw", "Vw", +1),
        "CwE1": (np.kron(dbt_y, Mbb_x), "Vw", "V1", -1),
        "CwE2": (np.kron(Mbb_y, dbt_x), "Vw", "V2", -1),
        "P13": (np.kron(Mtt_y, dbt_x), "V1", "V3", -1),
        "P23": (np.kron(dbt_y, Mtt_x), "V2", "V3", -1),
        # the NINE eps-weighted component masses
        "A11": (ew((bx, mx, xB, xB), (by, my, yT, yT), comp["e11"]),
                "V1", "V1", +1),
        "A12": (ew((bx, mx, xB, xT), (by, my, yT, yB), comp["e12"]),
                "V1", "V2", +1),
        "A13": (ew((bx, mx, xB, xT), (by, my, yT, yT), comp["e13"]),
                "V1", "V3", +1),
        "A21": (ew((bx, mx, xT, xB), (by, my, yB, yT), comp["e21"]),
                "V2", "V1", +1),
        "A22": (ew((bx, mx, xT, xT), (by, my, yB, yB), comp["e22"]),
                "V2", "V2", +1),
        "A23": (ew((bx, mx, xT, xT), (by, my, yB, yT), comp["e23"]),
                "V2", "V3", +1),
        "A31": (ew((bx, mx, xT, xB), (by, my, yT, yT), comp["e31"]),
                "V3", "V1", +1),
        "A32": (ew((bx, mx, xT, xT), (by, my, yT, yB), comp["e32"]),
                "V3", "V2", +1),
        "A33": (ew((bx, mx, xT, xT), (by, my, yT, yT), comp["e33"]),
                "V3", "V3", +1),
    }
    return blocks


def space_parities(sol):
    """The four 2-D signed permutations, in the module's ``kron(y, x)`` order.

    V1 = B(x) (x) Btilde(y)   V2 = Btilde(x) (x) B(y)
    V3 = Btilde(x) (x) Btilde(y)   Vw = B(x) (x) B(y)
    """
    ptx, stx, pbx, sbx = _stag_parity_1d(sol.bx)
    pty, sty, pby, sby = _stag_parity_1d(sol.by)
    q = sol.q

    def kron2(py_, sy_, px_, sx_):
        p = (py_[:, None] * q + px_[None, :]).ravel()
        s = (sy_[:, None] * sx_[None, :]).ravel()
        return signed_perm_matrix(p.astype(np.intp), s)

    return {"V1": kron2(pty, sty, pbx, sbx),
            "V2": kron2(pby, sby, ptx, stx),
            "V3": kron2(pty, sty, ptx, stx),
            "Vw": kron2(pby, sby, pbx, sbx)}


def part_b():
    print()
    print("=" * 92)
    print("B. the 2-D blocks: nine eps masses EVEN, four derivative blocks ODD")
    print("=" * 92)
    t = F.tensors()
    rows = []
    cells = [("centro pair generic (2,2)", F.centro_pair(t["generic"], 2), 6),
             ("(3,3) ring lc + interior lossy", F.centro_interior(t["lc"],
                                                                 t["lossy"]), 5),
             ("centro pair NON-RECIPROCAL (2,2)",
              F.centro_pair(t["nonrec"], 2), 7)]
    for name, cell, M in cells:
        sol = solver(cell, M)
        blocks = rebuild_blocks(sol)
        P = space_parities(sol)
        print(f"\n{name}, M={M}")
        print(f"  {'block':8s} {'test':4s} {'trial':5s} {'expect':>7s} "
              f"{'|P M P - s M|/|M|':>19s} {'|P M P + s M|/|M|(wrong)':>26s}")
        for bn, (Mm, ts, tr, sgn) in blocks.items():
            Pt_, Pr_ = P[ts], P[tr]
            img = Pt_.T @ Mm @ Pr_
            good = rel(img, sgn * Mm)
            bad = rel(img, -sgn * Mm)
            rows.append(dict(cell=name, M=M, block=bn, test=ts, trial=tr,
                             expect=sgn, good=good, wrong=bad))
            print(f"  {bn:8s} {ts:4s} {tr:5s} {sgn:+7d} {good:19.3e} "
                  f"{bad:26.3e}")
    ev = [r for r in rows if r["expect"] == +1]
    od = [r for r in rows if r["expect"] == -1]
    print()
    print(f"EVEN blocks ({len(ev)}): residual {min(r['good'] for r in ev):.2e}"
          f" .. {max(r['good'] for r in ev):.2e}; the ODD reading of the same "
          f"blocks {min(r['wrong'] for r in ev):.2e} .. "
          f"{max(r['wrong'] for r in ev):.2e}")
    print(f"ODD blocks  ({len(od)}): residual {min(r['good'] for r in od):.2e}"
          f" .. {max(r['good'] for r in od):.2e}; the EVEN reading "
          f"{min(r['wrong'] for r in od):.2e} .. "
          f"{max(r['wrong'] for r in od):.2e}")
    return rows


def part_b2():
    """The DERIVED consequence: e3 -> +P3 e3 and g3 -> -Pw g3, measured on the
    two elimination solves themselves (not on the final pencil)."""
    print()
    print("=" * 92)
    print("B2. the two ELIMINATIONS carry the derived signs (E3 EVEN, G3 ODD)")
    print("=" * 92)
    t = F.tensors()
    rows = []
    for name, cell, M in [("centro pair generic (2,2)",
                           F.centro_pair(t["generic"], 2), 6),
                          ("(3,3) ring lc + interior lossy",
                           F.centro_interior(t["lc"], t["lossy"]), 5)]:
        sol = solver(cell, M)
        bl = rebuild_blocks(sol)
        P = space_parities(sol)
        qq = sol.q * sol.q
        Z = np.zeros((qq, qq), dtype=complex)
        A31, A32 = bl["A31"][0], bl["A32"][0]
        A33 = bl["A33"][0]
        P13, P23 = bl["P13"][0], bl["P23"][0]
        CwE1, CwE2, Gw = bl["CwE1"][0], bl["CwE2"][0], bl["Gw"][0]
        E3S = np.linalg.solve(A33, np.concatenate(
            [-A31, -A32, P23.conj().T, -P13.conj().T], axis=1))
        G3S = np.linalg.solve(Gw, np.concatenate([-CwE1, CwE2, Z, Z], axis=1))
        # the state map R restricted to [E1;E2;G1;G2]
        Rst = np.zeros((4 * qq, 4 * qq))
        Rst[:qq, :qq] = P["V1"]
        Rst[qq:2 * qq, qq:2 * qq] = P["V2"]
        Rst[2 * qq:3 * qq, 2 * qq:3 * qq] = -P["V2"]
        Rst[3 * qq:, 3 * qq:] = -P["V1"]
        e3_even = rel(E3S @ Rst, P["V3"] @ E3S)
        e3_odd_wrong = rel(E3S @ Rst, -P["V3"] @ E3S)
        g3_odd = rel(G3S @ Rst, -P["Vw"] @ G3S)
        g3_even_wrong = rel(G3S @ Rst, P["Vw"] @ G3S)
        rows.append(dict(cell=name, M=M, e3_even=e3_even,
                         e3_odd_wrong=e3_odd_wrong, g3_odd=g3_odd,
                         g3_even_wrong=g3_even_wrong))
        print(f"{name:34s} M={M}  E3 -> +P3 E3 : {e3_even:.3e}   "
              f"(-P3 reading {e3_odd_wrong:.3e})")
        print(f"{'':34s}      G3 -> -Pw G3 : {g3_odd:.3e}   "
              f"(+Pw reading {g3_even_wrong:.3e})")
    return rows


# --------------------------------------------------------------------------- #
# C. every WRONG sign / permutation pattern
# --------------------------------------------------------------------------- #
def part_c():
    print()
    print("=" * 92)
    print("C. WRONG patterns.  dA = max|R A R + A| / max|A| ; dB likewise.")
    print("=" * 92)
    t = F.tensors()
    cell = F.centro_interior(t["lc"], t["lossy"])
    M = 5
    sol = solver(cell, M)
    A, B = sol.Agen, sol.Bgen
    qq = sol.q * sol.q
    P = space_parities(sol)
    sA, sB = float(np.max(np.abs(A))), float(np.max(np.abs(B)))

    def structure(perm_pattern, signs):
        R = np.zeros((4 * qq, 4 * qq))
        for k, sp in enumerate(perm_pattern):
            R[k * qq:(k + 1) * qq, k * qq:(k + 1) * qq] = signs[k] * P[sp]
        dA = float(np.max(np.abs(R @ A @ R + A))) / sA
        dB = float(np.max(np.abs(R @ B @ R - B))) / sB
        return dA, dB

    rows = []
    patterns = {
        "SHIPPED  blkdiag(P1,P2,P2,P1)": ("V1", "V2", "V2", "V1"),
        "WRONG-perm blkdiag(P1,P2,P1,P2)": ("V1", "V2", "V1", "V2"),
        "WRONG-perm blkdiag(P1,P1,P2,P2)": ("V1", "V1", "V2", "V2"),
        "WRONG-perm all-V3": ("V3", "V3", "V3", "V3"),
    }
    for pname, pp in patterns.items():
        print(f"\n{pname}")
        print(f"   {'signs':16s} {'dA':>11s} {'dB':>11s}  {'engages(1e-10)':14s}")
        for signs in itertools.product((+1, -1), repeat=4):
            dA, dB = structure(pp, signs)
            ok = dA <= 1e-10 and dB <= 1e-10
            rows.append(dict(pattern=pname, signs=list(signs), dA=dA, dB=dB,
                             passes=bool(ok)))
            star = "  <== PASSES" if ok else ""
            print(f"   {str(signs):16s} {dA:11.3e} {dB:11.3e}  "
                  f"{str(ok):14s}{star}")
    good = [r for r in rows if r["passes"]]
    print()
    print(f"{len(good)} of {len(rows)} (pattern, sign) combinations satisfy the "
          f"structure: {[ (r['pattern'], r['signs']) for r in good ]}")
    smallest_fail = min(r["dA"] for r in rows if not r["passes"])
    print(f"smallest FAILING dA {smallest_fail:.3e} -- "
          f"{np.log10(smallest_fail / 1e-10):.1f} decades above the 1e-10 bar")

    # ---- and corruptions of the 1-D map itself
    print()
    print("C2.  corruptions of _stag_parity_1d (the map, not the block signs)")
    ptx, stx, pbx, sbx = _stag_parity_1d(sol.bx)
    pty, sty, pby, sby = _stag_parity_1d(sol.by)
    q = sol.q

    def build(ptx_, stx_, pbx_, sbx_, pty_, sty_, pby_, sby_):
        def kron2(py_, sy_, px_, sx_):
            p = (py_[:, None] * q + px_[None, :]).ravel()
            s = (sy_[:, None] * sx_[None, :]).ravel()
            return signed_perm_matrix(p.astype(np.intp), s)
        p1 = kron2(pty_, sty_, pbx_, sbx_)
        p2 = kron2(pby_, sby_, ptx_, stx_)
        R = np.zeros((4 * qq, 4 * qq))
        R[:qq, :qq] = p1
        R[qq:2 * qq, qq:2 * qq] = p2
        R[2 * qq:3 * qq, 2 * qq:3 * qq] = -p2
        R[3 * qq:, 3 * qq:] = -p1
        dA = float(np.max(np.abs(R @ A @ R + A))) / sA
        dB = float(np.max(np.abs(R @ B @ R - B))) / sB
        r2 = float(np.max(np.abs(R @ R - np.eye(4 * qq))))
        return dA, dB, r2

    Nx, Mm = sol.bx.N, sol.bx.M
    variants = {}
    variants["shipped"] = (ptx, stx, pbx, sbx, pty, sty, pby, sby)
    # (i) bubbles carry NO sign
    variants["bubble sign -> +1"] = (ptx, np.ones_like(stx), pbx,
                                     np.abs(sbx), pty, np.ones_like(sty),
                                     pby, np.abs(sby))
    # (ii) bubble sign (-1)^(a+1) instead of (-1)^a
    variants["bubble sign flipped"] = (ptx, -stx * np.where(
        np.arange(stx.size) < Nx, -1.0, 1.0), pbx, sbx, pty, -sty * np.where(
        np.arange(sty.size) < Nx, -1.0, 1.0), pby, sby)
    # (iii) the B-set half-hats do NOT swap
    pbx_ns = pbx.copy()
    pby_ns = pby.copy()
    for s_ in range(Nx):
        base = s_ * (Mm - 1)
        img = (Nx - 1 - s_) * (Mm - 1)
        pbx_ns[base + 0] = img + 0
        pbx_ns[base + 1] = img + 1
        pby_ns[base + 0] = img + 0
        pby_ns[base + 1] = img + 1
    variants["half-hats do NOT swap"] = (ptx, stx, pbx_ns, sbx, pty, sty,
                                         pby_ns, sby)
    # (iv) identity map (no parity at all)
    ident_t = np.arange(ptx.size, dtype=np.intp)
    ident_b = np.arange(pbx.size, dtype=np.intp)
    variants["identity (no parity)"] = (ident_t, np.ones_like(stx), ident_b,
                                        np.ones_like(sbx), ident_t,
                                        np.ones_like(sty), ident_b,
                                        np.ones_like(sby))
    # (v) segment order NOT reversed (hats permuted, bubbles left in place)
    variants["bubbles not segment-reversed"] = (
        ptx, stx, pbx, sbx, pty, sty, pby, sby)
    del variants["bubbles not segment-reversed"]
    print(f"   {'variant':32s} {'dA':>11s} {'dB':>11s} {'|R^2-I|':>9s} "
          f"{'engages':8s}")
    crows = []
    for vn, args in variants.items():
        dA, dB, r2 = build(*args)
        ok = dA <= 1e-10 and dB <= 1e-10
        crows.append(dict(variant=vn, dA=dA, dB=dB, r2=r2, passes=bool(ok)))
        print(f"   {vn:32s} {dA:11.3e} {dB:11.3e} {r2:9.1e} {str(ok):8s}")
    return rows, crows


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    b2 = part_b2()
    c, c2 = part_c()
    with open(os.path.join(HERE, "results", "v3_involution.json"), "w") as fh:
        json.dump(dict(one_d=a, blocks=b, eliminations=b2, patterns=c,
                       corruptions=c2), fh, indent=1)
