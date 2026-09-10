"""ROUND 3 probe: the PRICE of the holomorphic flip -- census of |X| - 1 over
the round-1/round-2 fixture sets, where ``X = exp(-lam k0 L)`` is the layer
propagator.  ``conj(r)`` keeps ``Re(lam) >= 0`` so ``|X| <= 1`` exactly;
``-r`` hands back ``Re(lam) = -|Re(r)|`` for the flipped modes, so ``|X|``
exceeds 1 by ``exp(|Re(r)| k0 L) - 1``.  This measures the worst case.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from lumenairy.elements.rcwa import _core as C

CENSUS = []          # (site, n_modes, n_flipped, max|Re(r)| flipped, scale)
XCENSUS = []         # (max |X|-1, k0L, worst Re(lam))

_ORIG_DECAY = C._sqrt_decay
_ORIG_PSTAR = C._propagation_star
_ORIG_PSMAT = C._propagation_smatrix
_ORIG_PSTARG = C._propagation_star_general
_ORIG_PSMATG = C._propagation_smatrix_general

_MODE = os.environ.get("BC3_MODE", "neg")


def _decay(x, xp=None, band=C._CUT_BAND_REL):
    if xp is None:
        xp = C.array_namespace(x)
    x = xp.asarray(x).astype(C._C)
    r = xp.sqrt(x)
    scale = xp.maximum(xp.max(xp.abs(r)), 1.0) if r.size else 1.0
    flip = (xp.abs(r.real) <= band * scale) & (r.imag < 0)
    try:
        fl = np.asarray(flip)
        rr = np.asarray(r)
        if fl.any():
            CENSUS.append((int(rr.size), int(fl.sum()),
                           float(np.max(np.abs(rr.real[fl]))),
                           float(scale)))
    except Exception:
        pass
    if _MODE == "conj":
        return xp.where(flip, xp.conj(r), r)
    return r * xp.where(flip, -1.0, 1.0)


def _rec(lam, k0_L):
    try:
        la = np.asarray(lam)
        kl = float(np.real(k0_L))
        x = np.abs(np.exp(-la * k0_L))
        XCENSUS.append((float(np.max(x) - 1.0), kl,
                        float(np.min(np.real(la)))))
    except Exception:
        pass


def _pstar(S, lam, k0_L):
    _rec(lam, k0_L)
    return _ORIG_PSTAR(S, lam, k0_L)


def _psmat(lam, k0_L):
    _rec(lam, k0_L)
    return _ORIG_PSMAT(lam, k0_L)


def _pstarg(S, lam_f, lam_b, k0_L):
    _rec(lam_f, k0_L); _rec(lam_b, k0_L)
    return _ORIG_PSTARG(S, lam_f, lam_b, k0_L)


def _psmatg(lam_f, lam_b, k0_L):
    _rec(lam_f, k0_L); _rec(lam_b, k0_L)
    return _ORIG_PSMATG(lam_f, lam_b, k0_L)


C._sqrt_decay = _decay
C._propagation_star = _pstar
C._propagation_smatrix = _psmat
C._propagation_star_general = _pstarg
C._propagation_smatrix_general = _psmatg
for _m in ("lumenairy.elements.pmm.twod", "lumenairy.elements.rcwa.oned",
           "lumenairy.elements.rcwa.stack", "lumenairy.elements.berreman",
           "lumenairy.elements.rcwa.twod"):
    try:
        import importlib
        mod = importlib.import_module(_m)
        for _n, _f in (("_sqrt_decay", _decay), ("_propagation_star", _pstar),
                       ("_propagation_smatrix", _psmat),
                       ("_propagation_star_general", _pstarg),
                       ("_propagation_smatrix_general", _psmatg)):
            if hasattr(mod, _n):
                setattr(mod, _n, _f)
    except Exception as e:
        print("patch skip", _m, e)

# ------------------------------------------------------------------ fixtures
from lumenairy.elements.pmm import pmm_efficiency_2d, pmm_efficiency_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d

P, WL, DEP = 0.6e-6, 0.55e-6, 0.25e-6
XB = (0.2 * P, 0.6 * P)


def run():
    rows = []
    # -- hybrid 2-D PMM, the round-1/2 near-normal + oblique + conical set
    for th, ph, pol in ((0.0, 0.0, "te"), (1e-6, 0.0, "te"),
                        (0.3, 0.0, "te"), (0.3, 0.0, "tm"),
                        (0.4, 0.7, "te"), (0.4, 0.7, "tm"),
                        (0.8, 0.0, "te"), (1e-9, 0.5, "tm")):
        try:
            pmm_efficiency_2d(P, P, 6.0 + 0j, 1.0, XB, XB, 1.5, 1.0, DEP, WL,
                              theta=th, phi=ph, degree=5, n_orders=2,
                              polarization=pol)
            rows.append(("pmm2d", th, ph, pol, "ok"))
        except Exception as e:
            rows.append(("pmm2d", th, ph, pol, repr(e)[:50]))
    # -- the round-2 SPACER COINCIDENCE stack (uniform eps=2.25 | cell | uniform)
    try:
        from lumenairy.elements.pmm import PMM2DStackHybrid
        rows.append(("stack-hybrid", "-", "-", "-", "importable"))
    except Exception as e:
        rows.append(("stack-hybrid", "-", "-", "-", repr(e)[:50]))
    # -- ordinary 1-D RCWA + 1-D PMM (lossless, lossy, deep-cutoff)
    for th in (0.0, 1e-7, 0.2, 0.6):
        for epsp in (6.0 + 0j, 6.0 + 0.3j, 2.25 + 0j):
            try:
                rcwa_efficiency_1d(P, epsp, 1.0, 0.5, 1.5, 1.0, DEP, WL,
                                   theta=th, n_orders=6, polarization="te")
                rows.append(("rcwa1d", th, epsp, "te", "ok"))
            except Exception as e:
                rows.append(("rcwa1d", th, epsp, "te", repr(e)[:50]))
    for th in (0.0, 0.25):
        try:
            pmm_efficiency_1d(P, 6.0 + 0j, 1.0, 0.5, 1.5, 1.0, DEP, WL,
                              theta=th, degree=6, n_orders=4,
                              polarization="tm")
            rows.append(("pmm1d", th, "-", "tm", "ok"))
        except Exception as e:
            rows.append(("pmm1d", th, "-", "tm", repr(e)[:60]))
    return rows


if __name__ == "__main__":
    rows = run()
    nerr = sum(1 for r in rows if r[-1] not in ("ok", "importable"))
    print(f"fixtures run: {len(rows)}  ({nerr} raised)")
    for r in rows:
        if r[-1] not in ("ok", "importable"):
            print("   RAISED:", r)
    print(f"\nMODE = {_MODE}")
    if CENSUS:
        a = np.array([(c[0], c[1], c[2], c[3]) for c in CENSUS], float)
        print(f"on-cut FLIP census: {len(CENSUS)} arrays with >=1 flip; "
              f"total flipped modes {int(a[:, 1].sum())}")
        print(f"  worst |Re(r)| of a flipped mode : {a[:, 2].max():.6e}")
        print(f"  worst |Re(r)|/scale             : "
              f"{np.max(a[:, 2] / a[:, 3]):.6e}   (band = {C._CUT_BAND_REL:.0e})")
        print(f"  spectrum scale range            : "
              f"{a[:, 3].min():.4e} .. {a[:, 3].max():.4e}")
    else:
        print("on-cut FLIP census: NO flips recorded")
    if XCENSUS:
        b = np.array(XCENSUS, float)
        print(f"\npropagator census: {len(XCENSUS)} X-arrays")
        print(f"  worst  |X| - 1        : {b[:, 0].max():.6e}")
        print(f"  worst  min Re(lam)    : {b[:, 2].min():.6e}")
        print(f"  k0*L range            : {b[:, 1].min():.4e} .. "
              f"{b[:, 1].max():.4e}")
