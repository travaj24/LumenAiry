"""STEP 2 instrument: the near-cutoff ladder and the classifier band's two
populations, RE-MEASURED on the tree being built.

Three measurements, all on the FD (staggered) basis whose ``layer_modes`` is
the cheapest route to a layer's raw modal spectrum:

* ``crossing`` -- the near-cutoff ladder.  The PEC-walled cylindrical spectrum
  is discrete, so an ordinary ``k0`` sweep cannot reach a cutoff; the ladder
  solves for the cutoff wavenumber ``gamma_j`` of one named radial order and
  approaches it geometrically, setting ``k0 = gamma_j / (n sqrt(1 - delta))``
  so ``qn = n sqrt(delta)`` exactly.  Per rung: the order's ``qn``, both
  discriminating ratios (the shipped per-mode ``rho`` and the spectrum-scaled
  ``sigma``), which class each band puts it in, whether the flux verdict and
  the ``Im q`` verdict agree, and the whole stack's closure and channel count
  so the consequence is visible in an OBSERVABLE.
* ``band_population`` -- the four sides of the two-sided bar: the NOISE side on
  ordinary geometry, the NOISE side at a deep cutoff (the binding one), and the
  SIGNAL side at ``Im(n)`` = 1e-3 and 1e-6.
* ``harmlessness`` -- over weakly lossy media inside the widened band, does the
  FLUX verdict ever disagree with the DECAY verdict?  That is what makes the
  widening harmless rather than merely wider.

Usage: ``python s2_band.py <tag>``; re-run under each kernel and thread count.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

RBIG = 24.0
NFD = 120
NREF = 1.41
EPS = NREF ** 2


def fd_modes(m, k0, eps=EPS, N=NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, RBIG, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def _flux_and_norm(L):
    """z-flux and ``r dr`` field norm per mode of a staggered FD layer."""
    W, V = L["W"], L["V"]
    wq_f = np.real(np.asarray(L["wq_face"]))
    wq_n = np.real(np.asarray(L["wq_node"]))
    N = len(wq_f)
    flux = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
                   - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
    fnrm = (np.sum(np.abs(W[:N]) ** 2 * wq_f[:, None], axis=0)
            + np.sum(np.abs(W[N:]) ** 2 * wq_n[:, None], axis=0))
    return flux, fnrm


def ratios(L, k0):
    """``(q, flux, relflux, rho, sigma)``.

    ``rho`` is the SHIPPED discriminating ratio ``|Im q| / max(|Re q|,1e-300)``;
    ``sigma`` the CANDIDATE ``|Im q| / max(max|q|, k0)``.
    """
    q = np.asarray(L["q"])
    flux, fnrm = _flux_and_norm(L)
    rel = np.abs(flux) / np.maximum(fnrm, 1e-300)
    rho = np.abs(q.imag) / np.maximum(np.abs(q.real), 1e-300)
    scale = max(float(np.max(np.abs(q))) if q.size else 0.0, float(k0))
    sigma = np.abs(q.imag) / scale
    return q, flux, rel, rho, sigma


def gamma_of(m, idx=2):
    """The cutoff wavenumber of one named radial order, from a reference solve."""
    L = fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def crossing(m, gamma, deltas):
    rows = []
    for dl in deltas:
        k0 = gamma / (NREF * np.sqrt(1.0 - dl))
        q, flux, rel, rho, sigma = ratios(fd_modes(m, k0), k0)
        qn = q / k0
        phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
        if not phys.any():
            continue
        j = np.where(phys)[0][int(np.argmin(np.abs(qn.real)[phys]))]
        rows.append(dict(
            delta=float(dl), k0=float(k0), m=m,
            qn=float(abs(qn[j].real)), rho=float(rho[j]),
            sigma=float(sigma[j]),
            shipped_calls_prop=bool(rho[j] < 1e-9),
            candidate_calls_prop=bool(sigma[j] <= 1e-8),
            flux_fwd=bool(flux[j] >= 0.0), im_fwd=bool(q[j].imag > 0.0),
            rules_agree=bool((flux[j] >= 0.0) == (q[j].imag > 0.0)),
            relflux=float(rel[j]),
            forward_carries_backward_flux=bool(
                q[j].real > 0.0 and flux[j] < 0.0),
            kept_by_channel_gate=bool(abs(qn[j].imag) < 5e-5
                                      and qn[j].real > 1e-6)))
    return rows


def stack_consequence(m, gamma, deltas):
    """The OBSERVABLE: a lossless three-layer stack at each cutoff rung."""
    from lumenairy import BORStack
    out = []
    for dl in deltas:
        k0 = gamma / (NREF * np.sqrt(1.0 - dl))
        s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                     basis="fd")
        s.add_layer(0.4, eps=EPS)                    # COINCIDENT spacer
        s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
        s.add_layer(0.4, eps=EPS)
        s.set_source(k0=float(k0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = s.solve()
        e = np.asarray(res["energy"])
        out.append(dict(
            delta=float(dl), k0=float(k0), m=m,
            n_orders=int(np.size(res["R"])),
            closure=float(np.max(np.abs(e - 1.0))) if e.size else None,
            sumR=float(np.sum(res["R"])), sumT=float(np.sum(res["T"]))))
    return out


def band_population():
    """The four sides of the two-sided bar, in ``sigma``."""
    out = {}

    # --- NOISE, ORDINARY geometry: lossless propagating modes ---------------
    worst = 0.0
    n = 0
    for eps in (1.41 ** 2, 1.50 ** 2, 2.00 ** 2):
        for m in (0, 1, 2):
            for k0 in (0.8, 2.0, 3.5):
                L = fd_modes(m, k0, eps=eps)
                q, _f, _r, rho, sigma = ratios(L, k0)
                prop = np.abs(q.real) > 10.0 * np.abs(q.imag)
                if prop.any():
                    worst = max(worst, float(np.max(sigma[prop])))
                    n += 1
    out["noise_ordinary"] = dict(n=n, worst_sigma=worst)

    # --- NOISE at a DEEP CUTOFF: the binding population ----------------------
    worst = 0.0
    n = 0
    for m in (0, 1, 2):
        g = gamma_of(m)
        for e in range(4, 27, 2):
            dl = 10.0 ** (-e)
            k0 = g / (NREF * np.sqrt(1.0 - dl))
            L = fd_modes(m, k0)
            q, _f, _r, rho, sigma = ratios(L, k0)
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            if phys.any():
                worst = max(worst, float(np.max(sigma[phys])))
                n += 1
    out["noise_deep_cutoff"] = dict(n=n, worst_sigma=worst)

    # --- SIGNAL: genuinely lossy media ---------------------------------------
    for imn, key in ((1e-3, "signal_1e-3"), (1e-6, "signal_1e-6")):
        smallest = np.inf
        n = 0
        for m in (0, 1, 2):
            for k0 in (2.0, 3.5):
                eps = (NREF + 1j * imn) ** 2
                L = fd_modes(m, k0, eps=eps)
                q, _f, _r, rho, sigma = ratios(L, k0)
                phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
                if phys.any():
                    smallest = min(smallest, float(np.min(sigma[phys])))
                    n += 1
        out[key] = dict(n=n, worst_sigma=float(smallest))
    return out


def harmlessness():
    """Inside the widened band, does the FLUX verdict ever disagree with the
    DECAY verdict on a physically propagating mode?"""
    total = 0
    disagree = 0
    relmin = np.inf
    for imn in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10):
        for m in (0, 1, 2):
            eps = (NREF + 1j * imn) ** 2
            L = fd_modes(m, 2.0, eps=eps)
            q, flux, rel, _rho, _sig = ratios(L, 2.0)
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            for j in np.where(phys)[0]:
                total += 1
                relmin = min(relmin, float(rel[j]))
                if (flux[j] >= 0.0) != (q[j].imag > 0.0):
                    disagree += 1
    return dict(n_modes=total, n_disagree=disagree,
                min_relflux=float(relmin))


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    print("TREE", C.pin_tree())
    print("KERNEL", C.kernel_tag())
    deltas = sorted({10.0 ** (-e / 2.0) for e in range(8, 21)}, reverse=True)
    payload = dict(crossings=[], stacks=[])
    for m in (0, 1, 2):
        g = gamma_of(m)
        payload["crossings"] += crossing(m, g, deltas)
        payload["stacks"] += stack_consequence(m, g, deltas)
    if "--fast" not in sys.argv:
        payload["band_population"] = band_population()
        payload["harmlessness"] = harmlessness()
    cl = [r["closure"] for r in payload["stacks"] if r["closure"] is not None]
    nn = sorted({r["n_orders"] for r in payload["stacks"]})
    payload["summary"] = dict(
        worst_closure=max(cl) if cl else None,
        channel_counts=nn,
        n_rungs=len(payload["stacks"]),
        n_forward_backward_flux=sum(
            1 for r in payload["crossings"]
            if r["forward_carries_backward_flux"]),
        n_prop_called_evanescent=sum(
            1 for r in payload["crossings"] if not r["shipped_calls_prop"]),
        n_candidate_calls_evanescent=sum(
            1 for r in payload["crossings"] if not r["candidate_calls_prop"]),
    )
    print("SUMMARY", payload["summary"])
    if "band_population" in payload:
        print("BAND", payload["band_population"])
        print("HARMLESS", payload["harmlessness"])
    C.dump("s2_band_%s.json" % (tag,), payload)


if __name__ == "__main__":
    main()
