"""TASK F / probe A -- RE-MEASURE every asserted quantity in STEP 1-2 of
tests/unit/test_fix_bor_multilayer_guards.py (the orientation kernel and the
classifier band), on the running build/arm.

Usage:  python vf_a_orient.py <pre|post> <tag>
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
_vh.require_tree(BUILD)
import lumenairy  # noqa: E402

print("lumenairy.__file__ =", lumenairy.__file__)
_WANT = "lum_vbor" if BUILD == "post" else "lum_vbor_pre"
assert pathlib.Path(lumenairy.__file__).resolve().parents[1].name.lower() == _WANT

from lumenairy.elements.bor import BORStack  # noqa: E402
from lumenairy.elements.bor import _orient as _or  # noqa: E402
from lumenairy.elements.bor.zcascade import layer_modes  # noqa: E402

_RBIG, _NFD, _NREF = 24.0, 120, 1.41
_EPS = _NREF ** 2


def _fd_modes(m, k0, eps=_EPS, N=_NFD):
    return layer_modes(m, _RBIG, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def _flux_and_norm(L):
    W, V = L["W"], L["V"]
    wq_f = np.real(np.asarray(L["wq_face"]))
    wq_n = np.real(np.asarray(L["wq_node"]))
    N = len(wq_f)
    flux = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
                   - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
    fnrm = (np.sum(np.abs(W[:N]) ** 2 * wq_f[:, None], axis=0)
            + np.sum(np.abs(W[N:]) ** 2 * wq_n[:, None], axis=0))
    return flux, fnrm


def _sigma(L, k0):
    q = np.asarray(L["q"])
    scale = max(float(np.max(np.abs(q))) if q.size else 0.0, float(k0))
    return q, np.abs(q.imag) / scale


def _gamma_of(m, idx=2):
    L = _fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * _EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def _cutoff_stack(m, k0):
    s = BORStack(_RBIG, m, n_substrate=_NREF, n_superstrate=_NREF, N=_NFD,
                 basis="fd")
    s.add_layer(0.4, eps=_EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=_EPS)
    s.set_source(k0=float(k0))
    return s.solve()


out = dict(arm=_vh.arm(), build=BUILD, tag=TAG,
           BOR_CUT_BAND_REL=_or._BOR_CUT_BAND_REL,
           BOR_FLUX_FALLBACK_REL=_or._BOR_FLUX_FALLBACK_REL)

# --- gate: test_near_cutoff_closure  (assert closure < 1e-8) --------------
gam = {m: _gamma_of(m) for m in (0, 1, 2)}
out["gamma_of"] = {str(k): v for k, v in gam.items()}
dl = 3.1622776601683795e-06
k0 = gam[0] / (_NREF * np.sqrt(1.0 - dl))
res = _cutoff_stack(0, k0)
e = np.asarray(res["energy"])
out["near_cutoff_closure"] = dict(
    n_energy=int(e.size), closure=float(np.max(np.abs(e - 1.0))),
    n_channels=int(np.size(res["R"])), k0=float(k0),
    hash=_vh.hash_arrays(np.asarray(res["R"]), np.asarray(res["T"])))

# --- gate: test_near_cutoff_channel_count_is_stable_over_the_ladder -------
counts, worst, rungs = set(), 0.0, []
for e_ in range(8, 21):
    d_ = 10.0 ** (-e_ / 2.0)
    kk = gam[0] / (_NREF * np.sqrt(1.0 - d_))
    r = _cutoff_stack(0, kk)
    c = int(np.size(r["R"]))
    counts.add(c)
    en = np.asarray(r["energy"])
    cl = float(np.max(np.abs(en - 1.0))) if en.size else float("nan")
    if en.size:
        worst = max(worst, cl)
    rungs.append(dict(e=e_, n=c, closure=cl))
out["ladder"] = dict(counts=sorted(counts), worst=worst, rungs=rungs)

# --- gate: test_no_forward_mode_..._carries_backward_flux -----------------
bad, n_checked, relmin_signal = [], 0, np.inf
for eps in (1.41 ** 2, 2.00 ** 2):
    for m in (0, 1, 2):
        for kk in (0.8, 2.0, 3.5):
            L = _fd_modes(m, kk, eps=eps)
            q, sig = _sigma(L, kk)
            flux, fnrm = _flux_and_norm(L)
            rel = np.abs(flux) / np.maximum(fnrm, 1e-300)
            prop = sig <= _or._BOR_CUT_BAND_REL
            signal = rel > 1e-10
            sel = prop & signal
            n_checked += int(np.sum(sel))
            if sel.any():
                relmin_signal = min(relmin_signal, float(np.min(rel[sel])))
            for j in np.where(sel & (flux < 0.0))[0]:
                bad.append("eps=%g m=%d k0=%g mode=%d flux=%.3e"
                           % (eps, m, kk, j, flux[j]))
out["backward_flux"] = dict(n_bad=len(bad), bad=bad[:10], n_checked=n_checked,
                            relmin_of_checked=float(relmin_signal))

# --- gate: test_band_two_sided_population ---------------------------------
band = _or._BOR_CUT_BAND_REL
worst_ord, n_ord, ord_rows = 0.0, 0, []
for eps in (1.41 ** 2, 1.50 ** 2, 2.00 ** 2):
    for m in (0, 1, 2):
        for kk in (0.8, 2.0, 3.5):
            L = _fd_modes(m, kk, eps=eps)
            q, sig = _sigma(L, kk)
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            if phys.any():
                w = float(np.max(sig[phys]))
                worst_ord = max(worst_ord, w)
                n_ord += 1
                ord_rows.append(dict(eps=eps, m=m, k0=kk, worst=w,
                                     n_phys=int(phys.sum())))
            else:
                ord_rows.append(dict(eps=eps, m=m, k0=kk, worst=None,
                                     n_phys=0))
out["noise_ordinary"] = dict(n=n_ord, worst=worst_ord, bar=band / 1e3,
                             rows=ord_rows)

worst_cut, n_cut, cut_rows = 0.0, 0, []
for m in (0, 1, 2):
    g = gam[m]
    for e_ in range(4, 27, 4):
        d_ = 10.0 ** (-e_)
        kk = g / (_NREF * np.sqrt(1.0 - d_))
        L = _fd_modes(m, kk)
        q, sig = _sigma(L, kk)
        phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
        if phys.any():
            w = float(np.max(sig[phys]))
            worst_cut = max(worst_cut, w)
            n_cut += 1
            cut_rows.append(dict(m=m, e=e_, worst=w, n_phys=int(phys.sum())))
        else:
            cut_rows.append(dict(m=m, e=e_, worst=None, n_phys=0))
out["noise_cutoff"] = dict(n=n_cut, worst=worst_cut, bar=band / 2.0,
                           rows=cut_rows)

sig_rows = {}
for imn, floor in ((1e-3, 1e2), (1e-6, 2.0)):
    smallest = np.inf
    per = []
    for m in (0, 1, 2):
        for kk in (2.0, 3.5):
            L = _fd_modes(m, kk, eps=(_NREF + 1j * imn) ** 2)
            q, sig = _sigma(L, kk)
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            if phys.any():
                s = float(np.min(sig[phys]))
                smallest = min(smallest, s)
                per.append(dict(m=m, k0=kk, smallest=s))
    sig_rows["imn_%g" % imn] = dict(smallest=float(smallest),
                                    bar=floor * band,
                                    ratio_to_band=float(smallest) / band,
                                    rows=per)
out["signal_side"] = sig_rows

# --- gate: test_widening_the_band_is_harmless -----------------------------
total, disagree, relmin = 0, 0, np.inf
for imn in (1e-6, 1e-8, 1e-10):
    for m in (0, 1, 2):
        L = _fd_modes(m, 2.0, eps=(_NREF + 1j * imn) ** 2)
        q = np.asarray(L["q"])
        flux, fnrm = _flux_and_norm(L)
        rel = np.abs(flux) / np.maximum(fnrm, 1e-300)
        for j in np.where(np.abs(q.real) > 10.0 * np.abs(q.imag))[0]:
            total += 1
            relmin = min(relmin, float(rel[j]))
            if (flux[j] >= 0.0) != (q[j].imag > 0.0):
                disagree += 1
out["widening"] = dict(total=total, disagree=disagree, relmin=float(relmin),
                       bar_total=300, bar_relmin=1e-3)

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_a_orient_%s_%s.json" % (BUILD, TAG)), out)
print("closure=%.4e  ladder_counts=%s  worst_ladder=%.4e"
      % (out["near_cutoff_closure"]["closure"], out["ladder"]["counts"],
         out["ladder"]["worst"]))
print("noise_ord=%.4e (bar %.1e)  noise_cut=%.4e (bar %.1e)"
      % (worst_ord, band / 1e3, worst_cut, band / 2.0))
print("signal", {k: (v["smallest"], v["bar"]) for k, v in sig_rows.items()})
print("widening total=%d disagree=%d relmin=%.4e" % (total, disagree, relmin))
