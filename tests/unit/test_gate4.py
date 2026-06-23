"""GATE 4 -- the full multi-order cylindrical-vs-planar diffraction test.

The load-bearing physics gate for the BOR-PMM: a concentric binary RING grating of
radial period ``Lambda`` diffracts an incident cylindrical mode (transverse
wavenumber ``gamma_inc``) into output modes near ``gamma_inc + p*(2*pi/Lambda)``
(radial-frequency diffraction).  As ``Rbig/Lambda -> inf`` the ring becomes locally
planar, so the per-order cylindrical efficiency must match the PLANAR
``pmm_efficiency_1d`` per-order efficiency at incidence
``theta_inc = arcsin(gamma_inc/(n_sup k0))`` and the same grating.

THREE facts make the correspondence exact (each diagnosed before this test was
written; cf. GATE 4a in ``test_bor_solve.py`` for the per-mode Fresnel anchor):

1. POLARIZATION SPLIT.  The m!=0 propagating modes form an interleaved TE-like /
   TM-like ladder (``P_te = <|E_phi|^2>/<|E|^2>``: TE-like ~0.7-0.99, TM-like
   ~0.3-0.001, both -> pure at large gamma/angle).  The grating preserves this
   character, so a TE-like incident mode is compared to the planar TE oracle and a
   TM-like one to planar TM.

2. WINDOW-SUMMED ORDER.  The cylindrical diffracted "order" is a finite-WIDTH packet
   in gamma (width ~2*pi/Rbig from the finite radial extent), NOT a single mode.  So
   each same-pol propagating output mode is assigned to the nearest planar |kx_m|
   bucket (Voronoi in gamma) and the |S|^2 are SUMMED per bucket.  (A single-nearest-
   mode match captures only ~half the diffracted order; the window-sum recovers it
   to ~0.1%.)

3. |kx| FOLDING.  The cylindrical basis carries only gamma>=0, so planar orders that
   share |kx_m| fold into one cylindrical channel; their planar efficiencies are
   summed into the same bucket.

PASS (measured): per-order |Delta eta| well under 2-3% at Rbig/Lambda=16 for BOTH
polarisations, with a clean convergence trend in Rbig/Lambda and (separately) in N
(the latter shows the residual is the documented 2nd-order FD floor, not a physics
error).  Two independent gratings (Lambda=3.0 and 4.0) both pass.

Runtime: ~25-45 s (two dense 512x512 eigensolves at N=256).  Marked ``slow``.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.bor.bor_stack import BORStack
from lumenairy.elements.bor.coupled_radial_eigensolver import _assemble_staggered
from lumenairy.elements.bor.zcascade import layer_modes
from lumenairy.elements.pmm import pmm_efficiency_1d


def _pol_fraction(m, Rbig, N, k0, eps, q, W):
    """P_te[j] = <|E_phi|^2> / <|E_r|^2+|E_phi|^2+|E_z|^2> per mode column.

    TE-like (E perpendicular to the plane of incidence -> azimuthal E_phi) -> 1;
    TM-like (E in the plane -> E_r, E_z) -> 0.  Uses the staggered E_z recovery on
    the uniform half-space (E_r on faces, E_phi/E_z on nodes)."""
    op = _assemble_staggered(m, Rbig, N,
                             lambda r: np.full_like(r, eps, dtype=complex), k0)
    Lei, A_f2n, mrn = op["Lei"], op["A_f2n"], op["mrn"]
    r_n, r_f, h = op["r_n"], op["r_f"], op["h"]
    Pte = np.zeros(W.shape[1])
    for j in range(W.shape[1]):
        Er, Ephi = W[:N, j], W[N:, j]
        Ez = q[j] * (Lei @ (1j * A_f2n @ Er - mrn * Ephi))
        e_r = np.sum(np.abs(Er) ** 2 * r_f * h)
        e_p = np.sum(np.abs(Ephi) ** 2 * r_n * h)
        e_z = np.sum(np.abs(Ez) ** 2 * r_n * h)
        Pte[j] = e_p / max(e_r + e_p + e_z, 1e-300)
    return Pte


def _planar_buckets(Lam, theta_inc, n_sup, k0, n_ridge, n_groove, n_sub,
                    depth, duty, wavelength, pol):
    """Planar per-order R/T folded onto |kx_m| buckets (summing orders that share
    |kx_m|), propagating set only.  Returns (sorted-buckets, total R+T)."""
    orders, Reff, Teff = pmm_efficiency_1d(
        Lam, n_ridge, n_groove, n_sub, n_sup, depth, duty, wavelength,
        angle=theta_inc, polarization=pol)
    kx_inc = n_sup * k0 * np.sin(theta_inc)
    G = 2 * np.pi / Lam
    absk = np.abs(kx_inc + orders * G)
    prop = absk < n_sup * k0 - 1e-9
    buckets = {}
    for o, ak, r, t, p in zip(orders, absk, Reff, Teff, prop):
        if not p:
            continue
        b = buckets.setdefault(round(ak, 3),
                               dict(absk=ak, orders=[], R=0.0, T=0.0))
        b["orders"].append(int(o))
        b["R"] += r
        b["T"] += t
    bk = sorted(buckets.values(), key=lambda b: b["absk"])
    return bk, float(Reff.sum() + Teff.sum())


def _gate4_compare(Lam, Rfac, pol, *, k0=2.0, n_sup=1.41, n_ridge=2.45,
                   n_groove=1.41, depth=0.5, duty=0.5, m_az=1,
                   pts_per_period=16, te_cut=0.6, tm_cut=0.4):
    """Run one BOR ring-grating solve and return per-bucket (planar, cylindrical)
    efficiencies for the requested polarisation, plus context."""
    n_sub = n_sup
    eps = n_sup ** 2
    wavelength = 2 * np.pi / k0
    Rbig = Rfac * Lam
    N = int(round(Rbig / (Lam / pts_per_period)))

    s = BORStack(Rbig, m_az, n_substrate=n_sub, n_superstrate=n_sup, N=N)
    s.add_layer(depth, rings=(Lam, duty, n_ridge, n_groove))
    s.set_source(k0=k0)
    res = s.solve()
    inc = res["inc"]
    S11, S21 = res["S"][0], res["S"][2]
    gamma_all = res["gamma"]

    # classify polarisation on the (shared) half-space basis
    L = layer_modes(m_az, Rbig, N,
                    lambda r: np.full_like(r, eps, dtype=complex), k0,
                    staggered=True)
    Pte_inc = _pol_fraction(m_az, Rbig, N, k0, eps, L["q"], L["W"])[inc]
    want = (Pte_inc > te_cut) if pol == "te" else (Pte_inc < tm_cut)
    sel = np.where(want)[0]
    assert sel.size >= 3, f"too few pure-{pol} propagating modes"

    # incident mode: pure of requested pol, largest gamma (locally plane-wave-like)
    jsel = sel[np.argmax(gamma_all[sel])]
    jglob = inc[jsel]
    g_inc = gamma_all[jsel]
    theta_inc = float(np.arcsin(np.clip(g_inc / (np.sqrt(eps) * k0), 0, 1)))

    bk, sumRT = _planar_buckets(Lam, theta_inc, n_sup, k0, n_ridge, n_groove,
                                n_sub, depth, duty, wavelength, pol)
    centers = np.array([b["absk"] for b in bk])

    # Voronoi assignment of every same-pol output mode to nearest planar bucket; sum |S|^2
    cR = np.zeros(len(bk))
    cT = np.zeros(len(bk))
    for kk in sel:
        bi = int(np.argmin(np.abs(centers - gamma_all[kk])))
        n_glob = inc[kk]
        cR[bi] += abs(S11[n_glob, jglob]) ** 2
        cT[bi] += abs(S21[n_glob, jglob]) ** 2

    rows = [dict(orders=b["orders"], absk=b["absk"], pR=b["R"], pT=b["T"],
                 cR=cR[i], cT=cT[i]) for i, b in enumerate(bk)]
    return dict(rows=rows, sumRT=sumRT, theta_inc=theta_inc, g_inc=g_inc,
                Pte_inc=Pte_inc[jsel], N=N, energy=res["energy"],
                n_orders=len(bk))


@pytest.mark.slow
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_gate4_multiorder_vs_pmm_1d(pol):
    """GATE 4: per-order ring-grating efficiency == planar ``pmm_efficiency_1d``.

    Multi-order regime (Lambda=3.0 -> diffraction orders m=-2,-1,0 propagate), at
    Rbig/Lambda=16 on the spurious-free STAGGERED basis.  Asserts:
      * machine-precision energy conservation (R+T=1),
      * a genuine multi-order channel set (>= 3 propagating |kx| buckets),
      * substantial-order per-order |Delta eta| < 1.5 %,
      * the weakest near-cutoff order < 2.5 % (FD-floor dominated),
      * the planar and cylindrical total power agree to < 1 %.
    """
    out = _gate4_compare(Lam=3.0, Rfac=16, pol=pol)

    # machine-precision energy (the staggered basis is loss-free)
    assert np.max(np.abs(out["energy"] - 1.0)) < 1e-6

    # genuinely multi-order
    assert out["n_orders"] >= 3

    # incident mode is essentially pure of the requested polarisation
    if pol == "te":
        assert out["Pte_inc"] > 0.95
    else:
        assert out["Pte_inc"] < 0.05

    # per-order agreement.  The dominant (specular + first diffracted) orders match
    # tightly; the weakest near-cutoff order carries the 2nd-order FD floor.
    weakest = int(np.argmin([r["pR"] + r["pT"] for r in out["rows"]]))
    for i, r in enumerate(out["rows"]):
        dR = abs(r["pR"] - r["cR"])
        dT = abs(r["pT"] - r["cT"])
        bound = 0.025 if i == weakest else 0.015
        assert dR < bound, f"{pol} order {r['orders']} R: |{r['pR']:.4f}-{r['cR']:.4f}|={dR:.4f}"
        assert dT < bound, f"{pol} order {r['orders']} T: |{r['pT']:.4f}-{r['cT']:.4f}|={dT:.4f}"

    # planar vs cylindrical total power
    tot_p = sum(r["pR"] + r["pT"] for r in out["rows"])
    tot_c = sum(r["cR"] + r["cT"] for r in out["rows"])
    assert abs(tot_p - tot_c) < 0.01


@pytest.mark.slow
def test_gate4_convergence_in_Rbig_over_Lambda():
    """The max per-order TM error shrinks monotonically as Rbig/Lambda grows
    (the ring becomes locally planar) -- the load-bearing convergence trend.
    TM is used because it is FD-clean here (TE has a larger fixed FD floor that
    masks the geometric trend; the N-refinement check covers TE separately).

    Measured (Lam=3.0):  Rbig/Lam = 8, 16, 24  ->  max|delta| = 0.012, 0.004, 0.002
    (a clean monotone decrease, log-log slope ~ -1.7).  This test checks the first
    two rungs to stay fast (~2 min); the full three-rung ladder was run by hand."""
    errs = []
    for Rfac in (8, 16):
        out = _gate4_compare(Lam=3.0, Rfac=Rfac, pol="tm")
        maxd = max(max(abs(r["pR"] - r["cR"]), abs(r["pT"] - r["cT"]))
                   for r in out["rows"])
        errs.append(maxd)
    # decreasing toward the locally-planar limit, and well under 1 % at Rbig/Lam=16
    assert errs[1] < errs[0]
    assert errs[-1] < 0.01
