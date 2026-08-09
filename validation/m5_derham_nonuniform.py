"""M5 SPIKE 1 (N-1) -- non-uniform segment boundaries in the 2-D staggered Basis1D.

ANALYSIS-ONLY PROTOTYPE.  Nothing in ``lumenairy/`` is modified.  This script
COPIES the Granet-2023 staggered modified-Legendre construction out of
``lumenairy/elements/pmm/twod_staggered.py`` and generalizes the segment
boundaries from ``linspace`` + a SCALAR jacobian to an ARBITRARY partition +
a PER-SEGMENT jacobian vector, then measures the go/no-go gates named in
``docs/audits/PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md`` S4/M5:

  G1  de Rham residual   d(Btilde) subset span(B)   on uniform + non-uniform
  G2  Bloch periodic hat stays C0 (both the interior nodes and the seam)
  G3  uniform-boundary byte-identity of every assembled operator vs the library
  G4  conditioning vs segment-length ratio
  G5  convergence on an ANALYTIC case (exact 1-D lamellar Bloch dispersion)
  G6  full 2-D staggered solve on a non-uniform grid vs (a) the library uniform
      solve on a commensurate geometry and (b) RCWA on an "unrepresentable"
      wall position
  G7  the eig-dimension arithmetic for the 2 deg taper

Run:  python validation/m5_derham_nonuniform.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import scipy.linalg as sla
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm._core import (
    _guarded_lstsq,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
)
from lumenairy.elements.pmm.twod_staggered import (
    Basis1D,
    Granet2DTransverseE,
    _homog_geom_cache,
    _homog_region_modes,
    _kz_forward2,
    _modleg_value_deriv,
    _pmm2d_order_kz,
    _pmm2d_project_orders,
    _region_modes,
)
from lumenairy.elements.rcwa._core import _project_efficiency

_C = np.complex128


# ===========================================================================
# 1.  Basis1D generalized to an ARBITRARY partition
# ===========================================================================
class Basis1DNU(Basis1D):
    """``Basis1D`` with arbitrary segment boundaries ``xb`` and a PER-SEGMENT
    jacobian ``Jv[s] = 0.5 * (xb[s+1] - xb[s])``.

    The ONLY change vs the library class is that the three physical scale
    factors become per-segment vectors instead of scalars:

        mass       INT L R dx        = J_s   INT L R du
        stiffness  INT L' R' dx      = 1/J_s INT L' R' du
        mixed      INT L (dR/dx) dx  = 1     INT L (dR/du) du      (J cancels)

    ``_build_elementary`` and ``_build_sets`` are inherited UNCHANGED -- they
    are defined purely on the reference interval and never see a jacobian.
    That is the structural reason the generalization is small.
    """

    def __init__(self, d, xb, M, tau=1.0 + 0.0j):
        assert M >= 3, "Basis1D needs M>=3"
        xb = np.asarray(xb, dtype=float)
        if xb.ndim != 1 or xb.size < 2:
            raise ValueError("xb must be a 1-D array of >= 2 boundaries")
        if not np.all(np.diff(xb) > 0):
            raise ValueError("xb must be strictly increasing")
        self.d = float(d)
        self.xb = xb
        self.N = int(xb.size - 1)
        self.M = int(M)
        self.tau = _C(tau)
        self.hv = np.diff(xb)                       # (N,) segment lengths
        self.Jv = 0.5 * self.hv                     # (N,) per-segment dx/du
        # scalar forms deliberately ABSENT so any un-migrated site raises
        self.h = None
        self.J = None
        self._build_elementary()
        self._build_sets()

    # --- per-segment physical scale for each elementary matrix class --------
    def seg_scale(self, ref):
        if ref is self.m_ref:
            return self.Jv
        if ref is self.s_ref:
            return 1.0 / self.Jv
        return np.ones(self.N)                       # c_ref (one derivative)

    def _global_matrix(self, ref, setL, setR, eps_seg=None):
        w_seg = np.asarray(self.seg_scale(ref), dtype=_C)
        if eps_seg is not None:
            w_seg = w_seg * np.asarray(eps_seg, dtype=_C)
        L_ten = np.array(setL)
        R_ten = np.array(setR)
        RR = np.einsum("ab,jsb->jsa", ref, R_ten)
        return np.einsum("isa,s,jsa->ij", np.conj(L_ten), w_seg, RR)

    # ``mixed`` is inherited verbatim: its scale is 1 on EVERY segment.


def _global_pair_segmat_nu(basis, ref, setL, setR):
    """Per-segment contributions, per-segment jacobian (library sibling uses a
    scalar ``basis.J``)."""
    sc = np.asarray(basis.seg_scale(ref), dtype=_C)
    L_ten = np.array(setL)
    R_ten = np.array(setR)
    RR = np.einsum("ab,jsb->jsa", ref, R_ten)
    G = np.einsum("s,isa,jsa->sij", sc, np.conj(L_ten), RR)
    return G


class Granet2DTransverseENU(Granet2DTransverseE):
    """Granet 2-D staggered assembly on arbitrary per-axis partitions."""

    def __init__(self, px, py, xb, yb, M, eps_cell,
                 alpha0x=0.0, alpha0y=0.0, k0=2.0 * np.pi):
        self.k0 = float(k0)
        self.alpha0x = float(alpha0x)
        self.alpha0y = float(alpha0y)
        taux = np.exp(-1j * alpha0x * px)
        tauy = np.exp(-1j * alpha0y * py)
        self.bx = Basis1DNU(px, xb, M, taux)
        self.by = Basis1DNU(py, yb, M, tauy)
        self.eps_cell = np.asarray(eps_cell, dtype=_C)
        self.q = self.bx.dim
        if self.bx.dim != self.by.dim:
            raise ValueError("square-grid requirement Nx*(M-1) == Ny*(M-1)")
        self._assemble()

    def _eps_weighted(self, refx_pair, refy_pair):
        bx, refx, sLx, sRx = refx_pair
        by, refy, sLy, sRy = refy_pair
        Gx = _global_pair_segmat_nu(bx, refx, sLx, sRx)
        Gy = _global_pair_segmat_nu(by, refy, sLy, sRy)
        eps = self.eps_cell
        out = np.zeros((Gy.shape[1] * Gx.shape[1],
                        Gy.shape[2] * Gx.shape[2]), dtype=_C)
        for sx in range(bx.N):
            Wy = np.einsum("y,yij->ij", eps[sx, :], Gy)
            out += np.kron(Wy, Gx[sx])
        return out

    def _eps_dir(self, bx, lx, opx, rx, by, ly, opy, ry):
        def segmat(basis, lset, op, rset):
            sL = getattr(basis, lset)
            sR = getattr(basis, rset)
            Lt = np.array(sL)
            Rt = np.array(sR)
            if op == "m":
                RR = np.einsum("ab,jsb->jsa", basis.m_ref, Rt)
                sc = np.asarray(basis.Jv, dtype=_C)
            elif op == "dL":
                RR = np.einsum("ab,jsb->jsa", basis.c_ref, Rt)
                sc = np.ones(basis.N, dtype=_C)
            else:
                RR = np.einsum("ab,jsb->jsa", basis.c_ref.T, Rt)
                sc = np.ones(basis.N, dtype=_C)
            return np.einsum("s,isa,jsa->sij", sc, np.conj(Lt), RR)
        Gx = segmat(bx, lx, opx, rx)
        Gy = segmat(by, ly, opy, ry)
        out = np.zeros((Gy.shape[1] * Gx.shape[1],
                        Gy.shape[2] * Gx.shape[2]), dtype=_C)
        for sx in range(bx.N):
            Wy = np.einsum("y,yij->ij", self.eps_cell[sx, :], Gy)
            out += np.kron(Wy, Gx[sx])
        return out


def _stag_fourier_projection_nu(basis, orders, alpha0=0.0):
    """Per-segment-jacobian Fourier->Rayleigh projector (library sibling uses
    a scalar ``basis.J`` and a midpoint from a uniform ``xb``)."""
    d, N, M = basis.d, basis.N, basis.M
    G = 2.0 * np.pi / d
    xb = basis.xb
    nq = 2 * M + 8
    xg, wg = leggauss(nq)
    Vref, _ = _modleg_value_deriv(M, xg)
    orders = np.asarray(orders)
    T_local = np.zeros((len(orders), N, M), dtype=_C)
    for seg in range(N):
        Js = basis.Jv[seg]
        xphys = 0.5 * (xb[seg] + xb[seg + 1]) + Js * xg
        phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
        T_local[:, seg, :] = (Js / d) * (phase * wg) @ Vref.T

    def _assemble(global_set):
        S = np.array(global_set)
        return np.einsum("msa,jsa->mj", T_local, S)
    return _assemble


def _far_projector_2d_nu(bx, by, ox, oy, alpha0x=0.0, alpha0y=0.0):
    asmx = _stag_fourier_projection_nu(bx, ox, alpha0x)
    asmy = _stag_fourier_projection_nu(by, oy, alpha0y)
    P1 = np.kron(asmy(by.Btilde), asmx(bx.B))
    P2 = np.kron(asmy(by.B), asmx(bx.Btilde))
    return P1, P2


def solve_2d_nu(period_x, period_y, xb, yb, eps_cell, n_substrate,
                n_superstrate, depth, wavelength, *, M=8, n_orders=5,
                polarization="te", theta=0.0, phi=0.0):
    """``pmm_efficiency_2d_staggered`` on an ARBITRARY partition.  Structure is
    a line-for-line mirror of the library entry point minus the Wood guards."""
    eps_cell = np.asarray(eps_cell, dtype=_C)
    eps_sup = _C(n_superstrate) ** 2
    eps_sub = _C(n_substrate) ** 2
    wl = float(wavelength)
    k0 = 2.0 * np.pi / wl
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    alpha0x, alpha0y = kx0 * k0, ky0 * k0

    sol_l = Granet2DTransverseENU(period_x, period_y, xb, yb, M, eps_cell,
                                  alpha0x=alpha0x, alpha0y=alpha0y, k0=k0)
    Wl, Vl, lam_l, _ = _region_modes(sol_l)
    sol_h = Granet2DTransverseENU(
        period_x, period_y, xb, yb, M,
        np.full(eps_cell.shape, _C(eps_sup)),
        alpha0x=alpha0x, alpha0y=alpha0y, k0=k0)
    geom = _homog_geom_cache(sol_h)
    del sol_h
    Wsup, Vsup, _ = _homog_region_modes(geom, eps_sup)
    Wsub, Vsub, _ = _homog_region_modes(geom, eps_sub)

    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    order_x = np.tile(ox, len(oy))
    order_y = np.repeat(oy, len(ox))
    Nfo = len(order_x)
    P1, P2 = _far_projector_2d_nu(sol_l.bx, sol_l.by, ox, oy, alpha0x, alpha0y)
    qq = sol_l.q * sol_l.q
    Hsup = _pmm2d_project_orders(P1, P2, Wsup, qq)
    Hsub = _pmm2d_project_orders(P1, P2, Wsub, qq)
    kxv = kx0 + order_x * (wl / period_x)
    kyv = ky0 + order_y * (wl / period_y)
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        axu, ayu = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ayu, axu
            einc_sq = 1.0
        else:
            ex0, ey0 = axu, ayu
            kz_inc0 = float(np.real(_kz_forward2(eps_sup, kx0, ky0)))
            einc_sq = 1.0 + (kt / kz_inc0) ** 2
    delta = ((order_x == 0) & (order_y == 0)).astype(_C)
    rhs = np.concatenate([ex0 * delta, ey0 * delta])
    cinc = _guarded_lstsq(Hsup, rhs, "m5 2-D staggered NU far field")
    r_ord = Hsup @ (S11 @ cinc)
    t_ord = Hsub @ (S21 @ cinc)
    rx, ry = r_ord[:Nfo], r_ord[Nfo:]
    tx, ty = t_ord[:Nfo], t_ord[Nfo:]
    kz_ref, kz_trn, kz_inc, safe_r, safe_t = _pmm2d_order_kz(
        eps_sup, eps_sub, kxv, kyv, kx0, ky0)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R, T = _project_efficiency(np, kz_ref, kz_trn, kz_inc,
                               rx, ry, rz, tx, ty, tz, einc_sq)
    return np.stack([order_x, order_y], axis=1), R, T, 2 * qq


# ===========================================================================
# G1 / G2 -- de Rham residual and hat continuity
# ===========================================================================
def _eval_sets(basis, nq=None):
    """Values / physical derivatives of both global sets at Gauss points on
    every segment, plus the physical quadrature weights."""
    M, N = basis.M, basis.N
    nq = nq or (2 * M + 8)
    xg, wg = leggauss(nq)
    V, Vp = _modleg_value_deriv(M, xg)                    # (M, nq)
    Jv = basis.Jv
    w_phys = np.concatenate([wg * Jv[s] for s in range(N)])   # (N*nq,)

    def ev(global_set, deriv=False):
        S = np.array(global_set)                          # (dim, N, M)
        if deriv:
            base = np.einsum("jsa,aq->jsq", S, Vp)
            base = base / Jv[None, :, None]
        else:
            base = np.einsum("jsa,aq->jsq", S, V)
        return base.reshape(S.shape[0], N * nq)
    return ev, w_phys


def derham_residual(basis):
    """max_j  || d(Btilde_j)/dx - P_span(B) d(Btilde_j)/dx ||_2 / ||.||_2 .

    ``B`` is a BROKEN (segment-local) set, so the least-squares projection is
    block-diagonal and well conditioned; we still solve it globally to avoid
    baking the structure we are trying to test into the instrument."""
    ev, w = _eval_sets(basis)
    F = ev(basis.Btilde, deriv=True)                      # (dim, npts)
    Bv = ev(basis.B, deriv=False)
    sw = np.sqrt(w)
    Fw = F * sw[None, :]
    Bw = Bv * sw[None, :]
    # least squares  min_c || Bw^T c - Fw^T ||
    coef, *_ = np.linalg.lstsq(Bw.T, Fw.T, rcond=None)
    resid = Fw.T - Bw.T @ coef
    num = np.linalg.norm(resid, axis=0)
    den = np.linalg.norm(Fw.T, axis=0)
    rel = num / np.where(den == 0, 1.0, den)
    # converse control: d(B) should NOT be in span(Btilde) (the placement is
    # one-directional; a near-zero here would mean the test is vacuous)
    Fb = ev(basis.B, deriv=True) * sw[None, :]
    Tw = ev(basis.Btilde, deriv=False) * sw[None, :]
    c2, *_ = np.linalg.lstsq(Tw.T, Fb.T, rcond=None)
    r2 = Fb.T - Tw.T @ c2
    rel2 = np.linalg.norm(r2, axis=0) / np.maximum(
        np.linalg.norm(Fb.T, axis=0), 1e-300)
    return float(rel.max()), float(rel2.max())


def hat_c0_residual(basis):
    """Largest jump of any Btilde function across any interior node, and across
    the Bloch seam (where the expected ratio is ``tau``)."""
    M, N = basis.M, basis.N
    Ve, _ = _modleg_value_deriv(M, np.array([-1.0, 1.0]))
    vm1, vp1 = Ve[:, 0], Ve[:, 1]                      # values at u=-1, +1
    S = np.array(basis.Btilde)                          # (dim, N, M)
    left_val = np.einsum("jsa,a->js", S, vp1)           # value at right end of seg s
    right_val = np.einsum("jsa,a->js", S, vm1)          # value at left end of seg s
    worst_int = 0.0
    for node in range(1, N):
        jump = np.abs(left_val[:, node - 1] - right_val[:, node])
        worst_int = max(worst_int, float(jump.max()))
    # seam: physical Bloch condition  f(d) = tau * f(0)
    seam = np.abs(left_val[:, N - 1] - basis.tau * right_val[:, 0])
    scale = max(float(np.abs(np.array(basis.Btilde)).max()), 1.0)
    return worst_int / scale, float(seam.max()) / scale


# ===========================================================================
# G3 -- uniform-boundary byte-identity vs the library
# ===========================================================================
def byte_identity_vs_library(d, N, M, tau, exact_h=False):
    """``exact_h`` builds the uniform boundaries so that ``diff(xb)`` is
    BIT-EQUAL to the library's scalar ``h = d/N``.  Without it, ``linspace``
    returns segment lengths that differ from ``d/N`` in the last bit, which is
    the ONLY source of non-identity (see the report)."""
    lib = Basis1D(d, N, M, tau)
    if exact_h:
        xb = np.arange(N + 1, dtype=float) * (d / N)
        xb[-1] = d
    else:
        xb = np.linspace(0.0, d, N + 1)
    nu = Basis1DNU(d, xb, M, tau)
    if exact_h:
        nu.Jv = np.full(N, 0.5 * (d / N))
    out = {}
    pairs = [("Mtt", "mass", "Btilde", "Btilde"),
             ("Mbb", "mass", "B", "B"),
             ("Mtb", "mass", "Btilde", "B"),
             ("Stt", "stiff", "Btilde", "Btilde"),
             ("Sbb", "stiff", "B", "B"),
             ("Ctb", "mixed", "Btilde", "B"),
             ("Cbt", "mixed", "B", "Btilde"),
             ("Ctt", "mixed", "Btilde", "Btilde")]
    for name, op, sl, sr in pairs:
        a = getattr(lib, op)(getattr(lib, sl), getattr(lib, sr))
        b = getattr(nu, op)(getattr(nu, sl), getattr(nu, sr))
        out[name] = float(np.max(np.abs(a - b)))
    # eps-weighted mass (the per-segment weighting path)
    rng = np.random.default_rng(7)
    eps = rng.uniform(1.0, 4.0, N) + 0.1j * rng.uniform(0, 1, N)
    a = lib.mass(lib.Btilde, lib.Btilde, eps)
    b = nu.mass(nu.Btilde, nu.Btilde, eps)
    out["Mtt_eps"] = float(np.max(np.abs(a - b)))
    return out


def byte_identity_2d(px, py, N, M, eps_cell, a0x, a0y, k0):
    lib = Granet2DTransverseE(px, py, N, N, M, eps_cell,
                              alpha0x=a0x, alpha0y=a0y, k0=k0)
    nu = Granet2DTransverseENU(px, py, np.linspace(0, px, N + 1),
                               np.linspace(0, py, N + 1), M, eps_cell,
                               alpha0x=a0x, alpha0y=a0y, k0=k0)
    return {
        "Lmat": float(np.max(np.abs(lib.Lmat - nu.Lmat))),
        "Rmat": float(np.max(np.abs(lib.Rmat - nu.Rmat))),
        "Stt": float(np.max(np.abs(lib.Stt - nu.Stt))),
        "Schur": float(np.max(np.abs(lib.Schur - nu.Schur))),
    }


# ===========================================================================
# G5 -- analytic 1-D lamellar Bloch dispersion oracle
# ===========================================================================
def _te_pencil(basis, eps_seg, k0):
    """TE (E_y) 1-D modal pencil on the C0 set:  (k0^2 M_eps - S) c = g2 M c."""
    Mtt = basis.mass(basis.Btilde, basis.Btilde)
    Meps = basis.mass(basis.Btilde, basis.Btilde, eps_seg)
    S = basis.stiff(basis.Btilde, basis.Btilde)
    A = k0 * k0 * Meps - S
    return A, Mtt


def _lamellar_dispersion(g2, k0, eps1, eps2, a, b, bloch):
    """cos(bloch) - [cos k1 a cos k2 b - 0.5 (k1/k2 + k2/k1) sin k1 a sin k2 b]"""
    k1 = np.sqrt(_C(k0 ** 2 * eps1 - g2))
    k2 = np.sqrt(_C(k0 ** 2 * eps2 - g2))
    rhs = (np.cos(k1 * a) * np.cos(k2 * b)
           - 0.5 * (k1 / k2 + k2 / k1) * np.sin(k1 * a) * np.sin(k2 * b))
    return np.real(np.cos(bloch) - rhs)


def te_eig_error(basis, eps_seg, k0, eps1, eps2, a, b, bloch, n_check=4):
    """Absolute error of the n_check largest gamma^2 vs the exact transcendental
    root, found by bisection in a bracket around the numeric value."""
    A, Mb = _te_pencil(basis, eps_seg, k0)
    A = 0.5 * (A + A.conj().T)
    Mb = 0.5 * (Mb + Mb.conj().T)
    g2 = sla.eigh(A, Mb, eigvals_only=True)
    g2 = np.sort(np.real(g2))[::-1][:n_check]
    errs = []
    for gv in g2:
        span = max(abs(gv) * 1e-3, 1e-6 * k0 ** 2)
        lo, hi = gv - span, gv + span
        flo = _lamellar_dispersion(lo, k0, eps1, eps2, a, b, bloch)
        fhi = _lamellar_dispersion(hi, k0, eps1, eps2, a, b, bloch)
        tries = 0
        while flo * fhi > 0 and tries < 60:
            span *= 1.6
            lo, hi = gv - span, gv + span
            flo = _lamellar_dispersion(lo, k0, eps1, eps2, a, b, bloch)
            fhi = _lamellar_dispersion(hi, k0, eps1, eps2, a, b, bloch)
            tries += 1
        if flo * fhi > 0:
            errs.append(np.nan)
            continue
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fm = _lamellar_dispersion(mid, k0, eps1, eps2, a, b, bloch)
            if flo * fm <= 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        root = 0.5 * (lo + hi)
        errs.append(abs(gv - root) / max(abs(root), 1.0))
    return np.array(errs), g2


# ===========================================================================
# helpers
# ===========================================================================
def _partition(d, N, ratio, kind="geometric"):
    """A partition of [0,d] into N segments with h_max/h_min ~= ratio."""
    if kind == "geometric":
        r = ratio ** (1.0 / max(N - 1, 1))
        h = r ** np.arange(N)
    else:
        h = np.linspace(1.0, ratio, N)
    h = h / h.sum() * d
    return np.concatenate([[0.0], np.cumsum(h)])


def _fmt(x, n=3):
    if x is None:
        return "--"
    if isinstance(x, float) and (np.isnan(x)):
        return "nan"
    return f"{x:.{n}e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    res = {"platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__, "scipy": __import__("scipy").__version__}
    try:
        cfg = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
    except Exception:
        cfg = "unknown"
    res["blas"] = cfg
    print(f"# build: {res['python']} np{res['numpy']} sp{res['scipy']} "
          f"blas={cfg} {res['platform']}")

    # ---------------- G1 / G2 : de Rham + C0 --------------------------------
    print("\n== G1/G2  de Rham residual  d(Btilde) subset span(B),  and C0 ==")
    print(f"{'partition':<26}{'N':>3}{'M':>4}  {'deRham rel':>11} "
          f"{'converse dB':>11} {'C0 node':>10} {'C0 seam':>10} {'h_max/h_min':>12}")
    rows = []
    tau = np.exp(-1j * 0.37 * 2 * np.pi)
    cases = [("uniform (control)", None), ("mild geom  r=2", 2.0),
             ("strong geom r=10", 10.0), ("adversarial r=1e3", 1e3),
             ("adversarial r=1e6", 1e6)]
    for N in ([3, 6] if args.quick else [2, 3, 6, 11]):
        for M in ([4, 8] if args.quick else [3, 4, 8, 12]):
            for label, ratio in cases:
                d = 0.7
                xb = (np.linspace(0, d, N + 1) if ratio is None
                      else _partition(d, N, ratio))
                if N == 1 and ratio is not None:
                    continue
                b = Basis1DNU(d, xb, M, tau)
                dr, conv = derham_residual(b)
                c0i, c0s = hat_c0_residual(b)
                hr = float(b.hv.max() / b.hv.min())
                rows.append(dict(part=label, N=N, M=M, derham=dr,
                                 converse=conv, c0=c0i, seam=c0s, ratio=hr))
                print(f"{label:<26}{N:>3}{M:>4}  {_fmt(dr):>11} "
                      f"{_fmt(conv):>11} {_fmt(c0i):>10} {_fmt(c0s):>10} "
                      f"{hr:>12.3e}")
    res["derham"] = rows
    res["derham_max"] = max(r["derham"] for r in rows)
    res["converse_min"] = min(r["converse"] for r in rows)
    res["c0_max"] = max(max(r["c0"], r["seam"]) for r in rows)
    print(f"\n  WORST de Rham residual over all cases : {res['derham_max']:.3e}")
    print(f"  MIN converse residual (must be O(1))  : {res['converse_min']:.3e}")
    print(f"  WORST C0 residual (node or seam)      : {res['c0_max']:.3e}")

    # ---------------- G3 : uniform byte-identity ----------------------------
    print("\n== G3  uniform-boundary byte-identity vs the library ==")
    bi = {}
    print("  (i) xb = np.linspace(0,d,N+1)  -- diff(xb) != d/N in the last bit")
    for N, M in ([(3, 6)] if args.quick else [(2, 4), (3, 8), (5, 6)]):
        r = byte_identity_vs_library(0.7, N, M, tau)
        bi[f"linspace_N{N}_M{M}"] = r
        print(f"    Basis1D  N={N} M={M}: " +
              "  ".join(f"{k}={_fmt(v,1)}" for k, v in r.items()))
    print("  (ii) uniform fast path: Jv := 0.5*(d/N) exactly (the fix)")
    bi_exact = {}
    for N, M in ([(3, 6)] if args.quick else [(2, 4), (3, 8), (5, 6)]):
        r = byte_identity_vs_library(0.7, N, M, tau, exact_h=True)
        bi_exact[f"exact_N{N}_M{M}"] = r
        print(f"    Basis1D  N={N} M={M}: " +
              "  ".join(f"{k}={_fmt(v,1)}" for k, v in r.items()))
    epsc = np.array([[2.25, 1.0], [1.0, 4.0 + 0.2j]], dtype=_C)
    r2 = byte_identity_2d(0.7, 0.7, 2, 6, epsc, 0.3, 0.15, 2 * np.pi / 1.31)
    bi["granet2d_N2_M6_linspace"] = r2
    print("    Granet2D N=2 M=6 (linspace): " +
          "  ".join(f"{k}={_fmt(v,1)}" for k, v in r2.items()))
    allbi = [v for d_ in bi.values() for v in d_.values()]
    allex = [v for d_ in bi_exact.values() for v in d_.values()]
    res["byte_identity"] = bi
    res["byte_identity_exact"] = bi_exact
    res["byte_identity_max"] = float(max(allbi))
    res["byte_identity_exact_max"] = float(max(allex))
    print(f"  WORST |diff|, linspace boundaries : {res['byte_identity_max']:.3e}")
    print(f"  WORST |diff|, exact uniform Jv    : {res['byte_identity_exact_max']:.3e} "
          f"({'BYTE-IDENTICAL' if res['byte_identity_exact_max'] == 0.0 else 'NOT identical'})")

    # ---------------- G4 : conditioning vs ratio ----------------------------
    print("\n== G4  conditioning vs segment-length ratio ==")
    print(f"{'h_max/h_min':>12} {'cond Mtt':>11} {'cond Mbb':>11} "
          f"{'cond G(2D)':>12} {'cond Gw':>11}")
    condrows = []
    ratios = [1.0, 2.0, 5.0, 10.0, 100.0, 1e3, 1e4]
    for ratio in ratios:
        d = 0.7
        N, M = 3, 6
        xb = (np.linspace(0, d, N + 1) if ratio == 1.0
              else _partition(d, N, ratio))
        b = Basis1DNU(d, xb, M, tau)
        cMtt = float(np.linalg.cond(b.mass(b.Btilde, b.Btilde)))
        cMbb = float(np.linalg.cond(b.mass(b.B, b.B)))
        sol = Granet2DTransverseENU(d, d, xb, xb, M,
                                    np.full((N, N), _C(2.25)),
                                    alpha0x=0.0, alpha0y=0.0,
                                    k0=2 * np.pi / 1.31)
        cG = float(np.linalg.cond(-sol.Rmat))
        Gw = np.kron(b.mass(b.B, b.B), b.mass(b.B, b.B))
        cGw = float(np.linalg.cond(Gw))
        condrows.append(dict(ratio=float(b.hv.max() / b.hv.min()),
                             cMtt=cMtt, cMbb=cMbb, cG=cG, cGw=cGw))
        print(f"{b.hv.max()/b.hv.min():>12.3e} {cMtt:>11.3e} {cMbb:>11.3e} "
              f"{cG:>12.3e} {cGw:>11.3e}")
    res["conditioning"] = condrows

    # ---------------- G5 : analytic lamellar dispersion ---------------------
    print("\n== G5  analytic 1-D lamellar Bloch dispersion (exact oracle) ==")
    print("   period 0.7 um, eps 4.0 / 1.0, wl 1.31 um, Bloch phase 0.37*2pi")
    d, wl = 0.7, 1.31
    k0 = 2 * np.pi / wl
    eps1, eps2 = 4.0, 1.0
    bloch = 0.37 * 2 * np.pi
    tau_b = np.exp(-1j * bloch)
    g5 = []
    duty_cases = [0.5, 0.4, 0.371]
    Ms = [4, 6, 8] if args.quick else [4, 6, 8, 12, 16]
    for duty in duty_cases:
        a = duty * d
        b_w = d - a
        # exact-wall NON-UNIFORM partition: N=2, walls exactly on the boundary
        xb_nu = np.array([0.0, a, d])
        # uniform partitions that CAN represent the wall (if any small one does)
        Nu_ok = None
        for Ncand in range(1, 2001):
            if abs(duty * Ncand - round(duty * Ncand)) < 1e-12:
                Nu_ok = Ncand
                break
        for M in Ms:
            bn = Basis1DNU(d, xb_nu, M, tau_b)
            eps_nu = np.array([eps1, eps2])
            en, _ = te_eig_error(bn, eps_nu, k0, eps1, eps2, a, b_w, bloch)
            row = dict(duty=duty, M=M, dim_nu=bn.dim,
                       err_nu=float(np.nanmax(en)), N_uniform_needed=Nu_ok)
            # uniform at the SAME dim (cannot place the wall unless commensurate)
            Nsame = max(2, bn.dim // (M - 1))
            xu = np.linspace(0, d, Nsame + 1)
            eps_u = np.where(0.5 * (xu[:-1] + xu[1:]) < a, eps1, eps2)
            bu = Basis1DNU(d, xu, M, tau_b)
            eu, _ = te_eig_error(bu, eps_u, k0, eps1, eps2, a, b_w, bloch)
            row["N_uniform_same_dim"] = Nsame
            row["err_uniform_same_dim"] = float(np.nanmax(eu))
            # uniform at the SMALLEST commensurate N (exact walls, huge dim)
            if Nu_ok is not None and Nu_ok * (M - 1) <= (600 if args.quick else 1500):
                xu2 = np.linspace(0, d, Nu_ok + 1)
                eps_u2 = np.where(0.5 * (xu2[:-1] + xu2[1:]) < a, eps1, eps2)
                bu2 = Basis1DNU(d, xu2, M, tau_b)
                eu2, _ = te_eig_error(bu2, eps_u2, k0, eps1, eps2, a, b_w, bloch)
                row["dim_uniform_exact"] = bu2.dim
                row["err_uniform_exact"] = float(np.nanmax(eu2))
            g5.append(row)
    res["lamellar"] = g5
    print(f"{'duty':>7}{'M':>4}{'dim_NU':>8}{'err_NU':>11}"
          f"{'N_u(same dim)':>15}{'err_u':>11}"
          f"{'N_u exact':>11}{'dim_u':>8}{'err_u_exact':>13}")
    for r in g5:
        print(f"{r['duty']:>7.3f}{r['M']:>4}{r['dim_nu']:>8}"
              f"{_fmt(r['err_nu']):>11}{r['N_uniform_same_dim']:>15}"
              f"{_fmt(r['err_uniform_same_dim']):>11}"
              f"{str(r['N_uniform_needed']):>11}"
              f"{str(r.get('dim_uniform_exact','--')):>8}"
              f"{_fmt(r.get('err_uniform_exact')):>13}")

    # ---------------- G6 : full 2-D solve -----------------------------------
    print("\n== G6  full 2-D staggered solve on a non-uniform grid ==")
    px = py = 0.7
    wl2 = 1.31
    depth = 0.31
    eps_p, eps_b = 4.0, 1.0
    g6 = []
    # (a) commensurate walls at 0.25/0.75 -- library uniform N=4 vs NU N=3
    for M in ([5, 6] if args.quick else [4, 5, 6, 7]):
        from lumenairy.elements.pmm import pmm_efficiency_2d_staggered
        cell4 = np.array([[eps_b, eps_b, eps_b, eps_b],
                          [eps_b, eps_p, eps_p, eps_b],
                          [eps_b, eps_p, eps_p, eps_b],
                          [eps_b, eps_b, eps_b, eps_b]], dtype=_C)
        t0 = time.perf_counter()
        o_l, R_l, T_l = pmm_efficiency_2d_staggered(
            px, py, cell4, 1.0, 1.0, depth, wl2, degree=M, n_orders=3)
        t_lib = time.perf_counter() - t0
        xb3 = np.array([0.0, 0.25 * px, 0.75 * px, px])
        cell3 = np.array([[eps_b, eps_b, eps_b],
                          [eps_b, eps_p, eps_b],
                          [eps_b, eps_b, eps_b]], dtype=_C)
        t0 = time.perf_counter()
        o_n, R_n, T_n, dof_n = solve_2d_nu(px, py, xb3, xb3, cell3, 1.0, 1.0,
                                           depth, wl2, M=M, n_orders=3)
        t_nu = time.perf_counter() - t0
        i0l = int(np.where((o_l[:, 0] == 0) & (o_l[:, 1] == 0))[0][0])
        i0n = int(np.where((o_n[:, 0] == 0) & (o_n[:, 1] == 0))[0][0])
        g6.append(dict(case="commensurate", M=M,
                       lib_T0=float(T_l[i0l]), nu_T0=float(T_n[i0n]),
                       lib_R0=float(R_l[i0l]), nu_R0=float(R_n[i0n]),
                       lib_close=float(abs(R_l.sum() + T_l.sum() - 1)),
                       nu_close=float(abs(R_n.sum() + T_n.sum() - 1)),
                       t_lib=t_lib, t_nu=t_nu,
                       dof_lib=int(2 * (4 * (M - 1)) ** 2), dof_nu=int(dof_n)))
        print(f"  M={M}: lib(N=4,dof={2*(4*(M-1))**2:>6}) T0={T_l[i0l]:.9f} "
              f"|R+T-1|={abs(R_l.sum()+T_l.sum()-1):.2e} {t_lib:6.2f}s |  "
              f"NU(N=3,dof={dof_n:>6}) T0={T_n[i0n]:.9f} "
              f"|R+T-1|={abs(R_n.sum()+T_n.sum()-1):.2e} {t_nu:6.2f}s | "
              f"dT0={abs(T_l[i0l]-T_n[i0n]):.2e}")
    # (b) unrepresentable wall: pillar 0.317 .. 0.688 of the period, vs RCWA
    print("\n  (b) UNREPRESENTABLE walls (0.317 .. 0.688 of period) vs RCWA")
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    wa, wbnd = 0.317, 0.688
    xbU = np.array([0.0, wa * px, wbnd * px, px])
    cellU = np.array([[eps_b, eps_b, eps_b],
                      [eps_b, eps_p, eps_b],
                      [eps_b, eps_b, eps_b]], dtype=_C)
    nrast = 1000                       # walls land EXACTLY on 317/1000, 688/1000
    xs = (np.arange(nrast) + 0.5) / nrast
    mask = (xs >= wa) & (xs < wbnd)
    print(f"    raster realised walls: {np.argmax(mask)/nrast:.6f} .. "
          f"{(nrast-np.argmax(mask[::-1]))/nrast:.6f}")
    cell_r = np.where(np.outer(mask, mask), eps_p, eps_b).astype(_C)
    rc = {}
    for no in ([9, 13] if args.quick else [9, 13, 17, 21, 25]):
        o_r, R_r, T_r = rcwa_efficiency_2d(px, py, cell_r, 1.0, 1.0, depth, wl2,
                                           n_orders_x=no, n_orders_y=no)
        i0 = int(np.where((o_r[:, 0] == 0) & (o_r[:, 1] == 0))[0][0])
        rc[no] = (float(T_r[i0]), float(R_r[i0]))
        print(f"    RCWA n_orders={no:>3}: T0={T_r[i0]:.9f} R0={R_r[i0]:.9f}")
    ref_T0 = rc[max(rc)][0]
    for M in ([5, 6] if args.quick else [4, 5, 6, 7, 8, 9]):
        t0 = time.perf_counter()
        o_n, R_n, T_n, dof_n = solve_2d_nu(px, py, xbU, xbU, cellU, 1.0, 1.0,
                                           depth, wl2, M=M, n_orders=5)
        t_nu = time.perf_counter() - t0
        i0n = int(np.where((o_n[:, 0] == 0) & (o_n[:, 1] == 0))[0][0])
        g6.append(dict(case="unrepresentable", M=M, nu_T0=float(T_n[i0n]),
                       rcwa_T0=ref_T0, dof_nu=int(dof_n), t_nu=t_nu,
                       nu_close=float(abs(R_n.sum() + T_n.sum() - 1))))
        print(f"    NU staggered N=3 M={M} (dof={dof_n:>6}): T0={T_n[i0n]:.9f} "
              f"|dT0 vs RCWA|={abs(T_n[i0n]-ref_T0):.2e} "
              f"|R+T-1|={abs(R_n.sum()+T_n.sum()-1):.2e} {t_nu:6.2f}s")
    # NULL CONTROL: the alternative to N-1 -- snap the walls onto a uniform
    # grid the library CAN represent (N=3 -> 1/3, 2/3) and accept the error.
    print("    NULL CONTROL -- walls snapped to the nearest uniform N=3 grid "
          f"(1/3, 2/3): wall error {abs(wa-1/3)*px*1e3:.1f} nm / "
          f"{abs(wbnd-2/3)*px*1e3:.1f} nm")
    from lumenairy.elements.pmm import pmm_efficiency_2d_staggered as _lib2d
    for M in ([5, 6] if args.quick else [5, 6, 7, 8]):
        cell_s = np.array([[eps_b, eps_b, eps_b],
                           [eps_b, eps_p, eps_b],
                           [eps_b, eps_b, eps_b]], dtype=_C)
        o_s, R_s, T_s = _lib2d(px, py, cell_s, 1.0, 1.0, depth, wl2,
                               degree=M, n_orders=5)
        i0s = int(np.where((o_s[:, 0] == 0) & (o_s[:, 1] == 0))[0][0])
        g6.append(dict(case="snapped_uniform", M=M, T0=float(T_s[i0s]),
                       rcwa_T0=ref_T0))
        print(f"      lib uniform N=3 M={M}: T0={T_s[i0s]:.9f}  "
              f"|dT0 vs RCWA|={abs(T_s[i0s]-ref_T0):.2e}")
    # (c) POSITION INVARIANCE -- a property test, no oracle needed.  With
    # non-uniform segments the pillar can sit ANYWHERE, so this is a strictly
    # stronger test than the library's uniform-grid version.
    print("\n  (c) position invariance: slide the pillar (width 0.371) in the cell")
    w = 0.371
    piv = []
    Mp = 6
    for x0 in [0.05, 0.113, 0.2405, 0.4]:
        if x0 + w < 1.0:
            xbp = np.array([0.0, x0 * px, (x0 + w) * px, px])
            cellp = np.array([[eps_b, eps_b, eps_b],
                              [eps_b, eps_p, eps_b],
                              [eps_b, eps_b, eps_b]], dtype=_C)
        else:
            continue
        o_p, R_p, T_p, _ = solve_2d_nu(px, py, xbp, xbp, cellp, 1.0, 1.0,
                                       depth, wl2, M=Mp, n_orders=5)
        i0p = int(np.where((o_p[:, 0] == 0) & (o_p[:, 1] == 0))[0][0])
        piv.append(dict(x0=x0, T0=float(T_p[i0p]),
                        tot=float(R_p.sum() + T_p.sum())))
        print(f"    x0={x0:.4f}: T0={T_p[i0p]:.12f}  R+T={R_p.sum()+T_p.sum():.12f}")
    if piv:
        spread = max(r["T0"] for r in piv) - min(r["T0"] for r in piv)
        print(f"    T0 spread over pillar position: {spread:.3e}")
        res["position_invariance_spread"] = float(spread)
    res["position_invariance"] = piv
    res["twod"] = g6
    res["rcwa_ref"] = {str(k): v for k, v in rc.items()}

    # ---------------- G7 : eig-dimension arithmetic -------------------------
    print("\n== G7  eig-dimension arithmetic, 2 deg taper, 700 nm period ==")
    period_nm = 700.0
    thick_nm = 310.0
    tanphi = np.tan(np.deg2rad(2.0))
    arith = []
    for ns in [2, 4, 6, 8, 12]:
        off = thick_nm / ns * tanphi
        Nx_uni = int(np.ceil(period_nm / off))
        # NU: 2 walls per slice; per-layer window = 3 layers -> <= 6 walls +
        # cell edges; shared union = 2*ns walls
        Nx_nu_shared = 2 * ns
        Nx_nu_window = 6
        arith.append(dict(ns=ns, offset_nm=off, Nx_uniform=Nx_uni,
                          Nx_nu_shared=Nx_nu_shared,
                          Nx_nu_window=Nx_nu_window))
    print(f"{'n_slice':>8}{'wall offset':>13}{'Nx uniform':>12}"
          f"{'Nx NU shared':>14}{'Nx NU window':>14}")
    for r in arith:
        print(f"{r['ns']:>8}{r['offset_nm']:>11.3f}nm{r['Nx_uniform']:>12}"
              f"{r['Nx_nu_shared']:>14}{r['Nx_nu_window']:>14}")
    print(f"\n{'Nx':>6}{'M':>4}{'eig dim':>10}{'GB/matrix':>12}{'flop':>12}")
    tab = []
    for Nx in [3, 4, 6, 7, 8, 12, 13, 16, 25, 388, 390]:
        for M in [8]:
            n = 2 * (Nx * (M - 1)) ** 2
            gb = 16.0 * n * n / 1024 ** 3
            fl = 30.0 * n ** 3
            tab.append(dict(Nx=Nx, M=M, n=n, GB=gb, flop=fl))
            print(f"{Nx:>6}{M:>4}{n:>10}{gb:>12.4g}{fl:>12.3g}")
    res["arith_walls"] = arith
    res["arith_dim"] = tab

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"\n[json] {args.json}")
    return res


if __name__ == "__main__":
    main()
