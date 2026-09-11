"""E-EME -- measured probes for the lateral EME cascade
(``lumenairy/elements/eme/``).  MEASUREMENT ONLY; the library is untouched
(every instrument is a monkeypatch installed inside this process).

SITES
  class A : ``eme_2d._ky_forward``            eme_2d.py:129  exact ``ky.imag < 0.0``
            ``eme_diffraction.mode_match``    eme_diffraction.py:176 exact ``qz.imag < 0.0``
            (peers that DO carry a relative band, for contrast:
             ``eme_2d_vector._strip_split_forward`` eme_2d_vector.py:251,
             ``berreman._split_fwd_bwd`` berreman.py:170)
  class B : ``eme_2d._interface``  np.linalg.inv(a+b)   eme_2d.py:145    UNGUARDED
            ``eme_2d._star``       np.linalg.inv x2     eme_2d.py:159/160 UNGUARDED
            ``eme_diffraction.mode_match`` np.linalg.lstsq(rcond=None) eme_diffraction.py:192 UNGUARDED
  class C : no union grid, no mortar -- one uniform Nx x-grid shared by every
            strip and an ANALYTIC y direction.  Structural analogue = a
            vanishing strip height; ladder run as the instrument.

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=n MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_eme.py out.json
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

WL = 1.0
K0 = 2.0 * np.pi / WL
LX = 1.0
LY = 1.0
NX = 16


# ------------------------------------------------------------------ class A1
def a_branch(eps_im, nx=NX, uniform_loss=True, contrast=4.0):
    """One two-region strip.  ``strip_x_modes`` routes to ``scipy.linalg.eig``
    for ANY nonzero ``Im(eps)``, so ``Im(lam)`` is then the SUM of the physical
    loss and the eigensolver's backward error; once the loss falls below the
    backward error the EXACT test ``ky.imag < 0.0`` is decided by roundoff.

    ``uniform_loss=True`` puts the SAME ``Im(eps)`` on every pixel, which is a
    pure diagonal shift ``A + i c I`` whose eigenvalues move by EXACTLY ``i c``
    -- LAPACK reproduces that to full relative accuracy, so the pin is decided
    by physics however small ``c`` is.  ``uniform_loss=False`` puts the loss on
    ONE region, a genuine perturbation whose imaginary part must compete with
    the backward error."""
    from lumenairy.elements.eme import eme_2d as M
    eps_x = np.full(nx, 2.25 + 1j * eps_im, dtype=complex)
    eps_x[nx // 2:] = contrast + 1j * eps_im
    if not uniform_loss:
        eps_x[:nx // 2] = 2.25 + 0.0j
    lam = np.asarray(M.strip_x_modes(eps_x, LX, nx, K0)[0], dtype=complex)
    lam_h = np.sort(np.asarray(
        M.strip_x_modes(eps_x.real.astype(complex), LX, nx, K0)[0], float))
    raw = np.sqrt(lam + 0j)
    ky = M._ky_forward(lam, 0.0)
    flipped = ~np.isclose(raw, ky, rtol=0, atol=0)
    oncut = np.abs(raw.real) > 1e3 * np.abs(raw.imag)   # numerically real ky
    rel = np.abs(raw.imag) / np.maximum(np.abs(raw), 1e-300)
    sel = flipped & oncut
    return dict(
        eps_im=eps_im, nx=nx, uniform_loss=uniform_loss,
        route=("eigh (Hermitian, Im(lam) EXACTLY 0)" if eps_im == 0.0
               else "scipy.linalg.eig (Im(lam) = loss + backward error)"),
        physical_Im_lam=eps_im * K0 ** 2,
        min_abs_Im_lam=float(np.min(np.abs(lam.imag))),
        max_abs_Im_lam=float(np.max(np.abs(lam.imag))),
        eigh_gap_real=float(np.max(np.abs(np.sort(lam.real) - lam_h))),
        n_modes=int(lam.size), n_propagating=int(oncut.sum()),
        n_flipped=int(flipped.sum()),
        n_flipped_propagating=int(sel.sum()),
        worst_rel_Im_flipped=(float(np.max(rel[sel])) if sel.any() else None),
        flip_signature="".join("1" if f else "0" for f in flipped))


# ------------------------------------------------------------------ class A2
def a_indexmatched(detune, depth):
    """``diffraction_fd`` on a HOMOGENEOUS layer whose permittivity EXACTLY
    equals the superstrate's and the substrate's -- the index-matched slab
    whose exact answer is ``T_00 = 1``, ``R = 0``, ``energy = 1``.  Drives BOTH
    unguarded sites at once: ``mode_match``'s exact ``qz.imag < 0.0`` pin
    (class A) and its ``lstsq(rcond=None)`` (class B, instrumented)."""
    from lumenairy.elements.eme import eme_diffraction as D
    Nx = Ny = 12
    Mx = My = 1
    eps_sup = eps_sub = 2.25
    eps_xy = np.full((Nx, Ny), eps_sup * (1.0 + detune), dtype=complex)
    rec = {}
    orig = np.linalg.lstsq

    def spy(A, b, rcond=None):
        out = orig(A, b, rcond=rcond)
        A2, b2, x = np.asarray(A), np.asarray(b), out[0]
        s = np.asarray(out[3], float)
        nb = float(np.linalg.norm(b2))
        rec.update(n=int(min(A2.shape)), rank=int(out[2]),
                   svmin_over_svmax=(float(s[-1] / s[0])
                                     if s.size and s[0] > 0 else None),
                   relresid=(float(np.linalg.norm(A2 @ x - b2) / nb)
                             if nb else None),
                   cond=float(np.linalg.cond(A2)))
        return out

    np.linalg.lstsq = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = D.diffraction_fd(eps_xy, LX, LY, Nx, Ny, K0, eps_sup,
                                   eps_sub, depth, Mx, My)
    except Exception as exc:
        return dict(detune=detune, depth=depth, raised=repr(exc)[:200])
    finally:
        np.linalg.lstsq = orig
    orders = list(res["orders"])
    i0 = orders.index((0, 0))
    return dict(detune=detune, depth=depth, energy=float(res["energy"]),
                R00=float(res["R"][i0]), T00=float(res["T"][i0]),
                err_T00_vs_exact=(float(res["T"][i0] - 1.0) if detune == 0.0
                                  else None),
                lstsq=dict(rec))


# ------------------------------------------------------------------ class B
def b_census():
    """Population of (n, cond, relative inverse residual) at the three
    UNGUARDED cascade inverses over a real ``layer_modes`` solve, plus a
    BAND-EDGE sweep: ``V = Phi diag(i ky)`` loses rank as ``ky -> 0``, so
    ``b = solve(Vb, Va)`` and the mode-match ``a + b`` degrade with nothing
    watching."""
    from lumenairy.elements.eme import eme_2d as M
    pop = {"interface_apb": [], "interface_Vb": [], "star_D": [], "star_F": []}
    o_if, o_st = M._interface, M._star

    def _score(A):
        n = int(A.shape[0])
        try:
            X = np.linalg.inv(A)
            r = float(np.linalg.norm(A @ X - np.eye(n))
                      / max(1.0, np.linalg.norm(A)))
        except np.linalg.LinAlgError:
            r = float("inf")
        return (n, float(np.linalg.cond(A)), r)

    def spy_if(Wa, Va, Wb, Vb):
        pop["interface_Vb"].append(_score(Vb))
        a = np.linalg.solve(Wb, Wa)
        b = np.linalg.solve(Vb, Va)
        pop["interface_apb"].append(_score(a + b))
        return o_if(Wa, Va, Wb, Vb)

    def spy_st(SA, SB):
        A11, _A12, _A21, A22 = SA
        B11, _B12, _B21, _B22 = SB
        I = np.eye(A11.shape[0], dtype=complex)
        pop["star_D"].append(_score(I - B11 @ A22))
        pop["star_F"].append(_score(I - A22 @ B11))
        return o_st(SA, SB)

    e1 = np.full(NX, 2.25, dtype=complex)
    e2 = np.full(NX, 6.0, dtype=complex)
    e1[NX // 2:] = 6.0
    M._interface, M._star = spy_if, spy_st
    modes, raised = None, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modes = M.layer_modes([(e1, 0.5), (e2, 0.5)], LX, NX, LY, K0,
                                  (0.0, 40.0), n_scan=120)
    except Exception as exc:
        raised = repr(exc)[:200]
    finally:
        M._interface, M._star = o_if, o_st

    out = {}
    for k, v in pop.items():
        if not v:
            out[k] = dict(n=0)
            continue
        cs = np.array([x[1] for x in v])
        rs = np.array([x[2] for x in v])
        out[k] = dict(n=len(v), width=v[0][0], cond_min=float(cs.min()),
                      cond_median=float(np.median(cs)),
                      cond_max=float(cs.max()), resid_min=float(rs.min()),
                      resid_max=float(rs.max()),
                      n_cond_gt_1e12=int((cs > 1e12).sum()),
                      n_resid_gt_1e_8=int((rs > 1e-8).sum()))
    out["raised"] = raised
    out["n_modes_found"] = None if modes is None else int(np.size(modes))
    out["modes"] = (None if modes is None
                    else [float(v) for v in np.asarray(modes).ravel()[:12]])

    eu = np.full(NX, 4.0, dtype=complex)
    lam_u = np.sort(np.real(np.asarray(M.strip_x_modes(eu, LX, NX, K0)[0])))
    edge = float(lam_u[-1])
    rows = []
    for d in (1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-13, 0.0):
        q = edge - d
        sm = [(M.strip_x_modes(eu, LX, NX, K0), 0.5),
              (M.strip_x_modes(np.full(NX, 4.0 * (1 + 1e-6), complex),
                               LX, NX, K0), 0.5)]
        conds = []
        oi = M._interface

        def spy2(Wa, Va, Wb, Vb, _c=conds, _o=oi):
            _c.append((float(np.linalg.cond(Vb)),
                       float(np.linalg.cond(np.linalg.solve(Wb, Wa)
                                            + np.linalg.solve(Vb, Va)))))
            return _o(Wa, Va, Wb, Vb)

        M._interface = spy2
        try:
            sig, err = float(M.dispersion(sm, q, 0.0, LY)), None
        except Exception as exc:
            sig, err = None, repr(exc)[:120]
        finally:
            M._interface = oi
        rows.append(dict(dist_from_edge=d, qz2=q, sigma_min_M=sig, raised=err,
                         cond_Vb=(max(c[0] for c in conds) if conds else None),
                         cond_apb=(max(c[1] for c in conds) if conds else None)))
    out["band_edge_sweep"] = dict(edge_qz2=edge, rows=rows)
    return out


# ------------------------------------------------------------------ class C
def c_thin_strip():
    """Height ladder: the inserted strip pair's height -> 0 must converge to
    the two-strip cell (an exact limit)."""
    from lumenairy.elements.eme import eme_2d as M
    e1 = np.full(NX, 2.25, dtype=complex)
    e2 = np.full(NX, 6.0, dtype=complex)
    e1[NX // 2:] = 6.0
    rows = []
    for frac in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.0):
        h = frac * LY
        strips = ([(e1, 0.5 * LY), (e2, 0.5 * LY)] if h == 0.0 else
                  [(e1, 0.5 * LY - h), (e2, h), (e1, h), (e2, 0.5 * LY - h)])
        sm = [(M.strip_x_modes(e, LX, NX, K0), hh) for e, hh in strips]
        conds = []
        orig = M._interface

        def spy(Wa, Va, Wb, Vb, _c=conds, _o=orig):
            _c.append(float(np.linalg.cond(np.linalg.solve(Wb, Wa)
                                           + np.linalg.solve(Vb, Va))))
            return _o(Wa, Va, Wb, Vb)

        M._interface = spy
        try:
            sig, err = float(M.dispersion(sm, 5.0, 0.0, LY)), None
        except Exception as exc:
            sig, err = None, repr(exc)[:120]
        finally:
            M._interface = orig
        rows.append(dict(frac=frac, sigma_min_M=sig, raised=err,
                         max_iface_cond=(float(max(conds)) if conds else None)))
    return rows


def main():
    E.pin_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "e_eme.json"
    res = {}
    res["A_branch"] = (
        [a_branch(v) for v in (0.0, 1e-14, 1e-18, 1e-24, -1e-24)]
        + [a_branch(v, nx=nx, uniform_loss=False, contrast=12.0)
           for nx in (16, 64, 96, 128) for v in (1e-24, 1e-30)])
    res["A_indexmatched"] = [a_indexmatched(d, z)
                             for d in (0.0, 1e-6, -1e-6) for z in (0.2, 2.0)]
    res["B_census"] = b_census()
    res["C_thin_strip"] = c_thin_strip()
    E.dump(out, res)


if __name__ == "__main__":
    main()
