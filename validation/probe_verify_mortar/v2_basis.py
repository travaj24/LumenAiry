"""V2 -- the NON-UNIFORM segment basis, re-measured.

Sections (``python v2_basis.py [scal fail invar parity derham]``):

* ``scal``   the three physical scalings (mass ``J_n``, stiffness ``1/J_n``,
  mixed ``1``) against an INDEPENDENT physical-space Gauss-Legendre oracle
  that evaluates the global functions directly, plus two ANALYTIC identities
  (hat partition of unity: the hat block of the mass sums to ``d`` exactly and
  the hat block of the stiffness sums to ``0``).
* ``derham`` ``d(Btilde) subset span(B)`` on ARBITRARY walls.
* ``fail``   the two-sided fail-before for each ``J -> J_n`` site: a scalar-``J``
  reimplementation swapped in in-process; the UNIFORM arm must stay
  sha256-identical and the NON-UNIFORM arm must move.
* ``invar``  two device-level invariances that a mis-scaled site cannot fake:
  MIRROR (x -> d - x at normal incidence) and CYCLIC TRANSLATION of a
  non-uniform wall set (position invariance of the total efficiencies).
* ``parity`` the FIFTH site: ``_stag_parity_1d`` must return ``None`` on a
  non-mirror-symmetric wall set.  Two-sided: forced past the guard, is the
  answer wrong?  And a MIRROR-symmetric NON-uniform wall set must be ACCEPTED
  and CORRECT.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time

import numpy as np
from numpy.polynomial.legendre import leggauss

import lumenairy
from lumenairy.elements.pmm import twod_staggered as ts
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {_ROOT}")
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
Basis1D = ts.Basis1D
RES = {}


def h(a):
    a = np.ascontiguousarray(a)
    m = hashlib.sha256()
    m.update(str(a.dtype).encode())
    m.update(str(a.shape).encode())
    m.update(a.tobytes())
    return m.hexdigest()


# ------------------------------------------------------------------ oracle
def eval_global(basis, S, x):
    """Value and d/dx of the global function with stencil ``S`` at points
    ``x`` -- evaluated DIRECTLY in physical space from the reference
    modified-Legendre functions and segment ``n``'s OWN affine map (Eq. 31).
    Independent of every assembly path in the library."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    xb = np.asarray(basis.xb, dtype=float)
    val = np.zeros(x.shape, dtype=_C)
    der = np.zeros(x.shape, dtype=_C)
    for n in range(basis.N):
        lo, hi = xb[n], xb[n + 1]
        sel = (x >= lo - 1e-15) & (x <= hi + 1e-15)
        if not np.any(sel):
            continue
        xs = x[sel]
        u = (2.0 * xs - (lo + hi)) / (hi - lo)
        V, Vp = ts._modleg_value_deriv(basis.M, u)          # (M, npts)
        val[sel] = S[n] @ V
        der[sel] = (S[n] @ Vp) * (2.0 / (hi - lo))
    return val, der


def oracle_matrix(basis, setL, setR, kind):
    """``INT conj(opL phi_i) (opR psi_j) dx`` by segment-wise Gauss-Legendre of
    order ``4M + 10`` -- exact for the polynomial integrands."""
    xg, wg = leggauss(4 * basis.M + 10)
    xb = np.asarray(basis.xb, dtype=float)
    dimL, dimR = len(setL), len(setR)
    out = np.zeros((dimL, dimR), dtype=_C)
    for n in range(basis.N):
        lo, hi = xb[n], xb[n + 1]
        J = 0.5 * (hi - lo)
        xs = 0.5 * (lo + hi) + J * xg
        VL = np.zeros((dimL, xs.size), dtype=_C)
        VR = np.zeros((dimR, xs.size), dtype=_C)
        u = (2.0 * xs - (lo + hi)) / (hi - lo)
        V, Vp = ts._modleg_value_deriv(basis.M, u)
        for i, S in enumerate(setL):
            VL[i] = (S[n] @ Vp) * (1.0 / J) if kind in ("stiff",) else S[n] @ V
        for j, S in enumerate(setR):
            VR[j] = ((S[n] @ Vp) * (1.0 / J) if kind in ("stiff", "mixed")
                     else S[n] @ V)
        out += (np.conj(VL) * (wg * J)) @ VR.T
    return out


WALLSETS = [
    ("uniform_N3", 1.0, np.array([0.0, 1 / 3, 2 / 3, 1.0])),
    ("arb_N3", 1.0, np.array([0.0, 0.2371, 0.6183, 1.0])),
    ("arb_N4_d0.83", 0.83, np.array([0.0, 0.11, 0.19, 0.55, 0.83])),
    ("mirror_N3", 1.0, np.array([0.0, 0.2, 0.8, 1.0])),
    ("skew_N5", 1.37, np.array([0.0, 0.05, 0.4, 0.42, 1.1, 1.37])),
]


def sec_scal():
    out = {}
    for name, d, xb in WALLSETS:
        for M in (4, 6):
            for tau in (1.0 + 0j, np.exp(-1j * 0.41)):
                b = Basis1D(d, xb, M, tau)
                row = {}
                pairs = [("tt", b.Btilde, b.Btilde), ("bb", b.B, b.B),
                         ("tb", b.Btilde, b.B), ("bt", b.B, b.Btilde)]
                for pname, sl, sr in pairs:
                    for kind, fn in (("mass", b.mass), ("stiff", b.stiff),
                                     ("mixed", b.mixed)):
                        lib = fn(sl, sr)
                        ora = oracle_matrix(b, sl, sr, kind)
                        sc = max(float(np.max(np.abs(ora))), 1e-300)
                        row[f"{kind}_{pname}"] = float(
                            np.max(np.abs(lib - ora)) / sc)
                # ANALYTIC: the N hats of Btilde (tau = 1) are a partition of
                # unity, so   sum_ij <T_i | T_j> = INT 1 dx = d   EXACTLY and
                # sum_ij <T_i' | T_j'> = INT 0 dx = 0.
                if tau == 1.0:
                    hats = b.Btilde[:b.N]
                    Mh = b.mass(hats, hats)
                    Sh = b.stiff(hats, hats)
                    row["pou_mass_rel"] = float(abs(Mh.sum() - d) / d)
                    row["pou_stiff_abs"] = float(abs(Sh.sum()))
                out[f"{name}_M{M}_tau{np.angle(tau):.2f}"] = row
    RES["scal"] = out
    worst = max(max(v for k, v in r.items() if k.endswith(("tt", "bb", "tb",
                                                           "bt")))
                for r in out.values())
    pou = max(r.get("pou_mass_rel", 0.0) for r in out.values())
    pous = max(r.get("pou_stiff_abs", 0.0) for r in out.values())
    print(f"[scal] worst lib-vs-oracle rel = {worst:.3e}; "
          f"partition-of-unity mass rel = {pou:.3e}, stiff abs = {pous:.3e}",
          flush=True)


def sec_derham():
    """d(Btilde) subset span(B) on ARBITRARY walls: the least-squares residual
    of each dBtilde against the B set, measured in the L2 (mass) metric."""
    out = {}
    for name, d, xb in WALLSETS:
        for M in (4, 6):
            b = Basis1D(d, xb, M, 1.0 + 0j)
            # <B_i | dBtilde_j> and <B_i | B_j>
            Cbd = b.mixed(b.B, b.Btilde)          # INT B_i* dBtilde_j
            Mbb = b.mass(b.B, b.B)
            coef = np.linalg.solve(Mbb, Cbd)
            # POINTWISE residual (the Galerkin energy form differences two
            # nearly-equal large numbers and floors at sqrt(eps) ~ 1e-8, which
            # the UNIFORM lattice reads too -- i.e. estimator noise, not a
            # non-uniformity defect).  Evaluate both sides on a dense
            # per-segment Gauss grid instead.
            xg, _wg = leggauss(4 * M + 10)
            xs = np.concatenate([0.5 * (xb[n] + xb[n + 1])
                                 + 0.5 * (xb[n + 1] - xb[n]) * xg
                                 for n in range(b.N)])
            dT = np.array([eval_global(b, S, xs)[1] for S in b.Btilde])
            Bv = np.array([eval_global(b, S, xs)[0] for S in b.B])
            resid = dT - coef.T @ Bv
            rel = (np.max(np.abs(resid), axis=1)
                   / np.maximum(np.max(np.abs(dT), axis=1), 1e-300))
            out[f"{name}_M{M}"] = float(np.max(rel))
    RES["derham"] = out
    print(f"[derham] worst relative L2 residual of d(Btilde) in span(B) = "
          f"{max(out.values()):.3e}", flush=True)


# ------------------------------------------------------- scalar-J variants
def scalar_J_global_matrix(self, ref, setL, setR, eps_seg=None):
    """The PRE-2026-09-11 site 1, verbatim in shape: ONE scalar jacobian."""
    N, _M = self.N, self.M
    J = float(self.Jn[0])
    if ref is self.m_ref:
        scale = J
    elif ref is self.s_ref:
        scale = 1.0 / J
    else:
        scale = 1.0
    L_ten = np.array(setL)
    R_ten = np.array(setR)
    w_seg = np.ones(N, dtype=_C) * scale
    if eps_seg is not None:
        w_seg = w_seg * np.asarray(eps_seg, dtype=_C)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)
    return np.einsum("isa,s,jsa->ij", np.conj(L_ten), w_seg, RR)


def scalar_J_pair_segmat(basis, ref, setL, setR):
    """The PRE-change site 2."""
    J = float(basis.Jn[0])
    if ref is basis.m_ref:
        scale = J
    elif ref is basis.s_ref:
        scale = 1.0 / J
    else:
        scale = 1.0
    L_ten = np.array(setL)
    R_ten = np.array(setR)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)
    return scale * np.einsum("isa,jsa->sij", np.conj(L_ten), RR)


def scalar_J_eps_dir(self, bx, lx, opx, rx, by, ly, opy, ry, wmap=None):
    """The PRE-change site 3."""
    def segmat(basis, lset, op, rset):
        Lt = np.array(getattr(basis, lset))
        Rt = np.array(getattr(basis, rset))
        if op == "m":
            RR = np.einsum("ab,jsb->jsa", basis.m_ref, Rt)
            scale = float(basis.Jn[0])
        elif op == "dL":
            RR = np.einsum("ab,jsb->jsa", basis.c_ref, Rt)
            scale = 1.0
        else:
            RR = np.einsum("ab,jsb->jsa", basis.c_ref.T, Rt)
            scale = 1.0
        return scale * np.einsum("isa,jsa->sij", np.conj(Lt), RR)
    Gx = segmat(bx, lx, opx, rx)
    Gy = segmat(by, ly, opy, ry)
    eps = self.eps_cell if wmap is None else wmap
    out = np.zeros((Gy.shape[1] * Gx.shape[1], Gy.shape[2] * Gx.shape[2]),
                   dtype=_C)
    for sx in range(bx.N):
        Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)
        out += np.kron(Wy, Gx[sx])
    return out


def scalar_J_fourier(basis, orders, alpha0=0.0):
    """The PRE-change site 4."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    nq = 2 * M + 8
    xg, wg = leggauss(nq)
    Vref, _ = ts._modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    J = float(basis.Jn[0])
    for seg in range(N):
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T

    def _assemble(global_set):
        S = np.array(global_set)
        return np.einsum("msa,jsa->mj", T_local, S)
    return _assemble


SITES = {
    "site1_global_matrix": ("Basis1D._global_matrix", scalar_J_global_matrix),
    "site2_pair_segmat": ("_global_pair_segmat", scalar_J_pair_segmat),
    "site3_eps_dir": ("Granet2DTransverseE._eps_dir", scalar_J_eps_dir),
    "site4_fourier": ("_stag_fourier_projection", scalar_J_fourier),
}


def _patch(site, on):
    if site == "site1_global_matrix":
        Basis1D._global_matrix = (scalar_J_global_matrix if on
                                  else _ORIG["site1_global_matrix"])
    elif site == "site2_pair_segmat":
        ts._global_pair_segmat = (scalar_J_pair_segmat if on
                                  else _ORIG["site2_pair_segmat"])
    elif site == "site3_eps_dir":
        ts.Granet2DTransverseE._eps_dir = (scalar_J_eps_dir if on
                                           else _ORIG["site3_eps_dir"])
    else:
        ts._stag_fourier_projection = (scalar_J_fourier if on
                                       else _ORIG["site4_fourier"])


_ORIG = {
    "site1_global_matrix": Basis1D._global_matrix,
    "site2_pair_segmat": ts._global_pair_segmat,
    "site3_eps_dir": ts.Granet2DTransverseE._eps_dir,
    "site4_fourier": ts._stag_fourier_projection,
}

WL = 1.0e-6
PX = 0.9e-6
PY = 0.9e-6


def _stripe_stack(walls, M=5, n_orders=3, theta=0.18, phi=0.35):
    """One patterned layer whose walls are ``walls`` (fractions) in x and the
    SAME in y (segment counts must match), between two half-spaces.  Only one
    grid exists, so no mortar is involved: this isolates the BASIS."""
    nseg = len(walls) + 1
    tile = np.full((nseg, nseg), 2.25 + 0j)
    tile[0, :] = 6.0
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=n_orders,
                        layer_grids="per-layer")
    st.add_layer(0.30e-6, eps_cell=tile,
                 x_walls=[w * PX for w in walls],
                 y_walls=[w * PY for w in walls])
    st.set_source(WL, theta=theta, phi=phi)
    return st


def sec_fail():
    out = {}
    uni = [1 / 3, 2 / 3]
    nu = [0.2371, 0.6183]
    for site in SITES:
        rec = {}
        for label, walls in (("uniform", uni), ("nonuniform", nu)):
            vals = []
            for on in (False, True):
                _patch(site, on)
                try:
                    st = _stripe_stack(walls)
                    o, R, T = st.solve(jones=False)
                    vals.append((h(R), h(T), float(np.abs(
                        R.sum(axis=1) + T.sum(axis=1) - 1.0).max()),
                        np.concatenate([R.ravel(), T.ravel()])))
                except Exception as exc:                # noqa: BLE001
                    vals.append(("ERR", "ERR", float("nan"),
                                 np.array([np.nan])))
                    rec[label + "_exc"] = f"{type(exc).__name__}: {exc}"
                finally:
                    _patch(site, False)
            same = (vals[0][0] == vals[1][0] and vals[0][1] == vals[1][1])
            if vals[0][3].shape == vals[1][3].shape:
                mv = float(np.max(np.abs(vals[0][3] - vals[1][3])))
            else:
                mv = float("inf")
            rec[label] = {"sha_equal": bool(same), "max_move": mv,
                          "closure_ref": vals[0][2],
                          "closure_scalarJ": vals[1][2]}
        out[site] = rec
        u, n = rec["uniform"], rec["nonuniform"]
        print(f"[fail] {site:22s} uniform sha_equal={u['sha_equal']} "
              f"move={u['max_move']:.2e} | nonuniform move={n['max_move']:.3e}"
              f"  closure {n['closure_ref']:.2e} -> {n['closure_scalarJ']:.2e}",
              flush=True)
    RES["fail"] = out


def sec_invar():
    """Two device invariances a mis-scaled per-segment site cannot fake."""
    out = {}
    # ---- MIRROR: x -> P - x at NORMAL incidence, y uniform ------------------
    # walls (fractions) and the mirrored wall set with the mirrored tile
    walls = [0.2371, 0.6183]
    mirror = [1.0 - walls[1], 1.0 - walls[0]]
    for M in (4, 5):
        nseg = 3
        tile = np.full((nseg, nseg), 2.25 + 0j)
        tile[0, :] = 6.0
        tile[2, :] = 3.1
        tile_m = tile[::-1, :].copy()
        st1 = PMM2DStackPure(PX, PY, n_substrate=1.45, n_modes=M, n_orders=3,
                             layer_grids="per-layer")
        st1.add_layer(0.30e-6, eps_cell=tile,
                      x_walls=[w * PX for w in walls],
                      y_walls=[w * PY for w in walls])
        st1.set_source(WL, theta=0.0, phi=0.0)
        o1, R1, T1 = st1.solve(jones=False)
        st2 = PMM2DStackPure(PX, PY, n_substrate=1.45, n_modes=M, n_orders=3,
                             layer_grids="per-layer")
        st2.add_layer(0.30e-6, eps_cell=tile_m,
                      x_walls=[w * PX for w in mirror],
                      y_walls=[w * PY for w in walls])
        st2.set_source(WL, theta=0.0, phi=0.0)
        o2, R2, T2 = st2.solve(jones=False)
        # order (m, n) -> (-m, n) under the x mirror
        idx = {(int(a), int(b)): i for i, (a, b) in enumerate(o2)}
        perm = [idx[(-int(a), int(b))] for a, b in o1]
        dR = float(np.max(np.abs(R1 - R2[:, perm])))
        dT = float(np.max(np.abs(T1 - T2[:, perm])))
        out[f"mirror_M{M}"] = {"dR": dR, "dT": dT,
                               "scale": float(max(R1.max(), T1.max()))}
        print(f"[invar] mirror  M={M}: dR = {dR:.3e}  dT = {dT:.3e}",
              flush=True)
    # ---- CYCLIC TRANSLATION of the wall set (position invariance) -----------
    # device: strips [0, .2371] eps 6, [.2371, .6183] eps 2.25, [.6183, 1] 3.1
    # shifted by s = 1 - 0.6183 so the third strip starts at 0
    for M in (4, 5):
        w = [0.2371, 0.6183]
        e = [6.0, 2.25, 3.1]
        # shift by s = 1 - w[0] so the wall at w[0] lands on the seam: the
        # shifted device is the SAME device translated, still 3 segments, and
        # still non-uniform.
        w_s = [w[1] - w[0], 1.0 - w[0]]
        e_s = [e[1], e[2], e[0]]
        tile = np.full((3, 3), 2.25 + 0j)
        for i, v in enumerate(e):
            tile[i, :] = v
        tile_s = np.full((3, 3), 2.25 + 0j)
        for i, v in enumerate(e_s):
            tile_s[i, :] = v
        res = []
        for ww, tt in ((w, tile), (w_s, tile_s)):
            st = PMM2DStackPure(PX, PY, n_substrate=1.45, n_modes=M,
                                n_orders=3, layer_grids="per-layer")
            st.add_layer(0.30e-6, eps_cell=tt,
                         x_walls=[q * PX for q in ww],
                         y_walls=[q * PY for q in w])
            st.set_source(WL, theta=0.21, phi=0.0)
            res.append(st.solve(jones=False))
        dR = float(np.max(np.abs(res[0][1] - res[1][1])))
        dT = float(np.max(np.abs(res[0][2] - res[1][2])))
        out[f"shift_M{M}"] = {"dR": dR, "dT": dT}
        print(f"[invar] xshift M={M}: dR = {dR:.3e}  dT = {dT:.3e}",
              flush=True)
    RES["invar"] = out


def sec_parity():
    """The FIFTH site, two-sided."""
    out = {}
    d = 1.0
    for name, xb, mirror_ok in (
            ("uniform_N4", np.linspace(0, 1, 5), True),
            ("mirror_nonuniform_0.2_0.8", np.array([0.0, 0.2, 0.8, 1.0]),
             True),
            ("mirror_nonuniform_4seg", np.array([0.0, 0.15, 0.5, 0.85, 1.0]),
             True),
            ("asym_N3", np.array([0.0, 0.2371, 0.6183, 1.0]), False),
            ("asym_N4", np.array([0.0, 0.1, 0.2, 0.7, 1.0]), False)):
        b = Basis1D(d, xb, 5, 1.0 + 0j)
        p = ts._stag_parity_1d(b)
        out[name] = {"accepted": p is not None, "expect": mirror_ok}
        print(f"[parity] {name:28s} accepted={p is not None} "
              f"(expected {mirror_ok})", flush=True)

    # ---- the FORCED arm: bypass the guard on an ASYMMETRIC wall set ---------
    # A patched _stag_parity_1d that SKIPS the non-uniform refusal is exactly
    # the pre-guard code.  Two questions: (a) does the reduction's own
    # structural residual catch it, and (b) forced past that too, is the
    # answer WRONG?
    orig_parity = ts._stag_parity_1d

    def parity_no_guard(basis):
        saved = basis.uniform
        try:
            basis.uniform = True             # bypass the fifth-site refusal
            return orig_parity(basis)
        finally:
            basis.uniform = saved

    tilt, azi = 0.61, 0.44
    dvec = np.array([np.sin(tilt) * np.cos(azi), np.sin(tilt) * np.sin(azi),
                     np.cos(tilt)])
    epsd = (1.5 ** 2) * np.eye(3) + (1.7 ** 2 - 1.5 ** 2) * np.outer(dvec,
                                                                     dvec)

    def oop_solve(walls, symmetry, M=4, tol=None):
        n = len(walls) + 1
        tile = np.zeros((n, n, 3, 3), dtype=_C)
        for i in range(n):
            for j in range(n):
                tile[i, j] = np.diag([2.25, 2.25, 2.25]).astype(_C)
        tile[1, 1] = epsd                      # centred -> eps IS its own
        st = PMM2DStackPure(PX, PY, n_substrate=1.45, n_modes=M, n_orders=2,
                            symmetry=symmetry, layer_grids="per-layer")
        st.add_layer(0.26e-6, eps_cell=tile,
                     x_walls=[w * PX for w in walls],
                     y_walls=[w * PY for w in walls])
        st.set_source(WL, theta=0.0, phi=0.0)
        return st.solve(jones=False)

    asym = [0.2371, 0.6183]
    mir = [0.2, 0.8]
    for label, walls in (("asym", asym), ("mirror_nonuniform", mir)):
        dense = oop_solve(walls, False)
        auto = oop_solve(walls, "auto")
        ts._stag_parity_1d = parity_no_guard
        try:
            forced = oop_solve(walls, "auto")
        finally:
            ts._stag_parity_1d = orig_parity
        sc = float(max(dense[1].max(), dense[2].max(), 1e-300))
        d_auto = float(max(np.max(np.abs(dense[1] - auto[1])),
                           np.max(np.abs(dense[2] - auto[2]))))
        d_forced = float(max(np.max(np.abs(dense[1] - forced[1])),
                             np.max(np.abs(dense[2] - forced[2]))))
        out[f"forced_{label}"] = {
            "shipped_vs_dense": d_auto, "forced_vs_dense": d_forced,
            "scale": sc}
        print(f"[parity] FORCED {label:20s} shipped-vs-dense {d_auto:.3e} | "
              f"guard-bypassed-vs-dense {d_forced:.3e} (scale {sc:.3f})",
              flush=True)

    # ---- does _stag_block_eig's own structural residual see it? ------------
    # Build the assembled OOP pencil directly and measure the two residuals
    # the reduction screens on, with the parity built from the ASYMMETRIC
    # wall set past the guard.
    for label, walls in (("asym", asym), ("mirror_nonuniform", mir)):
        n = len(walls) + 1
        tile = np.zeros((n, n, 3, 3), dtype=_C)
        for i in range(n):
            for j in range(n):
                tile[i, j] = np.diag([2.25, 2.25, 2.25]).astype(_C)
        tile[1, 1] = epsd
        wx = np.array([0.0] + [w * PX for w in walls] + [PX]) / WL
        sol = ts.Granet2DTransverseE(PX / WL, PY / WL, wx, wx, 4, tile,
                                     alpha0x=0.0, alpha0y=0.0, k0=2 * np.pi)
        ts._stag_parity_1d = parity_no_guard
        try:
            g = ts._stag_parity_gauge(sol)
        finally:
            ts._stag_parity_1d = orig_parity
        if g is None:
            out[f"struct_{label}"] = {"gauge": None}
            print(f"[parity] STRUCT {label}: gauge refused upstream")
            continue
        perm, r = g
        A, B = sol.Agen, sol.Bgen
        rr = r[:, None] * r[None, :]
        resA = float(np.max(np.abs(rr * A[np.ix_(perm, perm)] + A))
                     / max(float(np.max(np.abs(A))), 1e-300))
        resB = float(np.max(np.abs(rr * B[np.ix_(perm, perm)] - B))
                     / max(float(np.max(np.abs(B))), 1e-300))
        out[f"struct_{label}"] = {"resA_rel": resA, "resB_rel": resB,
                                  "tol": float(ts._STAG_BLOCK_TOL)}
        print(f"[parity] STRUCT {label:20s} |RAR+A|/|A| = {resA:.3e}  "
              f"|RBR-B|/|B| = {resB:.3e}  (tol {ts._STAG_BLOCK_TOL:.1e})",
              flush=True)
        # forced past BOTH: raise the tolerance and compare spectra
        fac = ts._stag_block_eig(A, B, sol.q ** 2, g, tol=1e6)
        if fac is None:
            out[f"struct_{label}"]["forced_eig"] = "refused anyway"
            print(f"[parity] STRUCT {label}: reduction refused even at "
                  f"tol=1e6")
        else:
            # ONE-SIDED HAUSDORFF: for every eigenvalue the reduction
            # produces, the distance to the NEAREST dense eigenvalue.  (A
            # sorted elementwise diff is meaningless here -- the two solvers
            # order a complex spectrum differently.)
            import scipy.linalg as sla
            lam_r = np.asarray(fac[0])
            lam_d = np.asarray(sla.eig(A, B, right=False))
            lam_d = lam_d[np.isfinite(lam_d)]
            # the reduction returns gamma = sqrt(mu); the dense pencil's
            # eigenvalue IS gamma for this first-order generator, so compare
            # directly and also against its negative (branch).
            dist = np.min(np.abs(lam_r[:, None] - lam_d[None, :]), axis=1)
            sc = max(float(np.max(np.abs(lam_d))), 1e-300)
            gap = float(np.max(dist) / sc)
            out[f"struct_{label}"]["forced_spectrum_rel_hausdorff"] = gap
            out[f"struct_{label}"]["n_reduced"] = int(lam_r.size)
            out[f"struct_{label}"]["n_dense"] = int(lam_d.size)
            print(f"[parity] STRUCT {label:20s} FORCED reduced spectrum vs "
                  f"dense (one-sided Hausdorff/|lam|max): {gap:.3e}  "
                  f"[{lam_r.size} vs {lam_d.size}]", flush=True)
    RES["parity"] = out


SECTIONS = {"scal": sec_scal, "derham": sec_derham, "fail": sec_fail,
            "invar": sec_invar, "parity": sec_parity}


def main():
    which = sys.argv[1:] or list(SECTIONS)
    for w in which:
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v2_basis.json")
    old = {}
    if os.path.exists(path):
        old = json.load(open(path))
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
