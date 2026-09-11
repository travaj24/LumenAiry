"""TASK B -- the forward-orientation BAND, re-derived on MY OWN populations.

WHAT IS BEING DECIDED.  A layer's modal solve returns squared axial
wavenumbers; the square root's sign is fixed by a classifier:

    PROPAGATING  (oriented by the sign of its own r dr z-flux)  if
        |Im q| <= band * scale
    EVANESCENT   (oriented by decay, Im q > 0)                  otherwise

5.45.1 changed ``band * scale`` from ``1e-9 * max(|Re q|, 1e-300)`` -- the
mode's OWN collapsing real part -- to ``1e-8 * max(max|q| over the layer's
spectrum, k0)``.

WHY THE DISCRIMINATING RATIOS ARE BUILD-COMPARABLE.  Orientation multiplies a
mode by -1, so ``|Im q|``, ``|Re q|`` and ``max|q|`` are all INVARIANT under
it.  Both ``sigma = |Im q| / max(max|q|, k0)`` and ``rho = |Im q| /
max(|Re q|, 1e-300)`` can therefore be evaluated from the ORIENTED spectrum
either build returns, and the two classifiers compared on the same modes.
What is NOT invariant, and is the observable that matters, is the SIGN of the
returned mode's z-flux: a forward mode of a lossless layer must carry power in
+z, so a mode with ``prop`` true and ``P < 0`` in the returned basis is a mode
shipped backward.

Usage:  python v2_band.py <pre|post> [outdir] [--fast]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402

_BAND_NEW = 1e-8
_BAND_OLD = 1e-9


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def _stag_flux(W, V, wq_face, wq_node):
    N = len(wq_face)
    return np.real(
        np.sum(W[:N] * np.conj(V[N:]) * wq_face[:, None], axis=0)
        - np.sum(W[N:] * np.conj(V[:N]) * wq_node[:, None], axis=0))


def census_one(m, Rbig, N, prof, k0, tag, R_pml=None, staggered=True):
    """One layer's whole spectrum, with both classifiers evaluated on it.

    ``staggered=False`` is the legacy NODAL basis; it is the ONLY basis on
    which ``zcascade.layer_modes`` accepts ``R_pml`` (the staggered Yee basis
    refuses it by name -- "the radial PML is a NODAL-basis feature"), so the
    PML-adjacent population has to be taken there."""
    from lumenairy.elements.bor.zcascade import layer_modes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = layer_modes(m, Rbig, N, prof, k0, staggered=staggered,
                        wall=(None if R_pml is not None else "pec"),
                        R_pml=R_pml)
    q = np.asarray(L["q"])
    if staggered:
        P = _stag_flux(L["W"], L["V"], np.real(np.asarray(L["wq_face"])),
                       np.real(np.asarray(L["wq_node"])))
    else:
        Nn = L["N"]
        W, V, wq = L["W"], L["V"], np.asarray(L["wq"])
        P = np.real(np.sum((W[:Nn] * np.conj(V[Nn:])
                            - W[Nn:] * np.conj(V[:Nn]))
                           * wq[:, None], axis=0))
    scale_new = max(float(np.max(np.abs(q))), abs(float(np.real(k0))))
    sigma = np.abs(q.imag) / scale_new
    rho = np.abs(q.imag) / np.maximum(np.abs(q.real), 1e-300)
    prop_new = sigma <= _BAND_NEW
    prop_old = rho < _BAND_OLD
    # PHYSICALLY propagating, independent of either band: the returned root's
    # real part dominates its imaginary part.  In a lossless PEC-walled layer
    # q^2 is real, so every mode is either real (propagating) or imaginary
    # (evanescent) and this is a clean split; in a lossy layer it is the
    # travelling set.  This is the classification the two BANDS are judged
    # against -- ``worst sigma among modes the band calls propagating`` is
    # bounded by the band itself and proves nothing.
    phys_prop = np.abs(q.real) > np.abs(q.imag)
    return dict(tag=tag, phys_prop=phys_prop, m=m, Rbig=float(Rbig), N=int(N),
                k0=float(np.real(k0)), n_modes=int(q.size),
                staggered=bool(staggered),
                scale_new=scale_new,
                max_abs_q=float(np.max(np.abs(q))),
                sigma=sigma, rho=rho, P=P, q=q,
                prop_new=prop_new, prop_old=prop_old)


def _pack(rec, keep_modes=False):
    if rec.get("raised") or np.size(rec.get("sigma", [])) == 0:
        return {k: v for k, v in rec.items()
                if k in ("tag", "raised", "msg")}
    out = {k: v for k, v in rec.items()
           if k not in ("sigma", "rho", "P", "q", "prop_new", "prop_old",
                        "phys_prop")}
    s, r, P = rec["sigma"], rec["rho"], rec["P"]
    pn, po = rec["prop_new"], rec["prop_old"]
    pp = rec["phys_prop"]
    q = rec["q"]
    out.update(
        n_prop_new=int(pn.sum()), n_prop_old=int(po.sum()),
        n_class_disagree=int((pn != po).sum()),
        worst_sigma_prop_new=float(np.max(s[pn])) if pn.any() else None,
        min_sigma_evan_new=float(np.min(s[~pn])) if (~pn).any() else None,
        worst_rho_prop_old=float(np.max(r[po])) if po.any() else None,
        min_rho_evan_old=float(np.min(r[~po])) if (~po).any() else None,
        n_backward_in_forward_new=int(np.sum(pn & (P < 0.0))),
        n_backward_in_forward_old=int(np.sum(po & (P < 0.0))),
        # a mode the NEW band calls propagating that the OLD called evanescent
        n_new_prop_old_evan=int(np.sum(pn & ~po)),
        n_old_prop_new_evan=int(np.sum(po & ~pn)),
        worst_sigma_all=float(np.max(s)), min_sigma_all=float(np.min(s)),
        # the band-INDEPENDENT statistics
        n_phys_prop=int(pp.sum()),
        worst_sigma_phys_prop=float(np.max(s[pp])) if pp.any() else None,
        min_sigma_phys_evan=float(np.min(s[~pp])) if (~pp).any() else None,
        n_phys_prop_called_evan_new=int(np.sum(pp & ~pn)),
        n_phys_prop_called_evan_old=int(np.sum(pp & ~po)),
        n_phys_evan_called_prop_new=int(np.sum(~pp & pn)),
        n_phys_evan_called_prop_old=int(np.sum(~pp & po)),
        # THE OBSERVABLE: a forward mode of a lossless layer carrying power
        # in -z, in the basis the LIBRARY returned on this tree
        n_phys_prop_backward=int(np.sum(pp & (P < -0.5))),
        n_backward_any=int(np.sum(P < -0.5)),
        # a mode the band called propagating and then put on the GROWING
        # branch (Im q < 0): the cascade's stability guarantee
        worst_growing_imag=float(np.min(q.imag[pn])) if pn.any() else None,
        n_growing_prop=int(np.sum(pn & (q.imag < -1e-300))))
    if keep_modes:
        out["modes"] = [dict(q=[float(z.real), float(z.imag)],
                             sigma=float(a), rho=float(b), P=float(c))
                        for z, a, b, c in zip(rec["q"], s, r, P)]
    return out


# --------------------------------------------------------------------------- #
#  populations                                                                 #
# --------------------------------------------------------------------------- #
def ordinary_population():
    """>= 300 modes of ORDINARY lossless geometry -- the NOISE side the band
    must reach."""
    recs = []
    for m in (0, 1, 2, 5):
        for k0 in (0.8, 2.0, 3.5):
            recs.append(census_one(m, 6.0, 120, _uni(1.96), k0,
                                   "uniform_m%d_k%g" % (m, k0)))
            recs.append(census_one(m, 6.0, 120, _ring(1.2, 1.69, 3.24), k0,
                                   "ring_m%d_k%g" % (m, k0)))
    for Rb, N in ((2.0, 80), (12.0, 200), (24.0, 240)):
        recs.append(census_one(1, Rb, N, _uni(1.96), 2.0,
                               "uniform_R%g" % Rb))
    # a unit system six orders of magnitude away
    recs.append(census_one(1, 6.0e-6, 120, _uni(1.96), 2.0e6, "nmunits"))
    recs.append(census_one(1, 6.0e6, 120, _uni(1.96), 2.0e-6, "kmunits"))
    return recs


def cutoff_ladder(fast=False):
    """A radial order driven toward its OWN cutoff from BOTH sides.

    The PEC-walled cylindrical spectrum is discrete: for a uniform layer
    ``q_j^2 = eps k0^2 - gamma_j^2`` with ``gamma_j`` a k0-independent
    transverse eigenvalue of the radial operator.  Measuring ``gamma_j`` once
    lets ``k0`` be set so that a chosen order sits at ``qn = +- n sqrt(delta)``
    for any ``delta`` -- i.e. just ABOVE cutoff (propagating, ``q`` real) and
    just BELOW it (evanescent, ``q`` imaginary), which is the two-sided
    approach the band has to survive.
    """
    Rbig, N, n = 24.0, 120, 1.41
    eps = n ** 2
    recs = []
    k0ref = 2.0
    for m in (0, 1, 2):
        base = census_one(m, Rbig, N, _uni(eps), k0ref, "base_m%d" % m)
        q = base["q"]
        g2 = eps * k0ref ** 2 - q ** 2
        g = np.sqrt(g2[np.abs(g2.imag) < 1e-9 * np.abs(g2.real)].real)
        g = np.unique(np.round(g[g > 0.1], 10))
        if g.size == 0:
            continue
        gam = float(g[len(g) // 3])              # a mid-spectrum radial order
        deltas = ([1e-4, 1e-8, 1e-12, 1e-18, 1e-26] if fast else
                  [1e-4, 1e-5, 1e-6, 1e-8, 1e-10, 1e-12, 1e-15, 1e-18,
                   1e-22, 1e-26])
        for d in deltas:
            for side in (+1, -1):
                # +1: k0 above the cutoff (propagating); -1: below (evanescent)
                k0 = gam / (n * np.sqrt(1.0 - side * d))
                recs.append(census_one(
                    m, Rbig, N, _uni(eps), k0,
                    "cutoff_m%d_d%g_%s" % (m, d, "above" if side > 0 else
                                           "below")))
                recs[-1]["gamma"] = gam
                recs[-1]["delta"] = d
                recs[-1]["side"] = int(side)
    return recs


def lossy_ladder():
    """The SIGNAL side: genuinely lossy media, ``Im(n)`` from 1e-1 down to
    1e-9.  ``sigma`` is linear in the imaginary index, so this ladder locates
    the smallest real signal the band must NOT absorb."""
    recs = []
    for imn in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
        for m in (0, 1, 2):
            e = complex(1.4, imn) ** 2
            recs.append(census_one(m, 6.0, 120, _uni(e), 2.0,
                                   "lossy_m%d_im%g" % (m, imn)))
            recs[-1]["im_n"] = imn
    return recs


def pml_population():
    """PML-adjacent modes.  ``BORStack`` never sets ``R_pml``; the staggered
    Yee basis REFUSES it ("the radial PML is a NODAL-basis feature"), so this
    population exists only on the legacy nodal basis and is measured there --
    with a nodal control at the same settings so the PML's own effect is
    separable from the basis's."""
    recs = []
    for m in (0, 1):
        try:
            recs.append(census_one(m, 6.0, 120, _uni(1.96), 2.0,
                                   "nodal_ctrl_m%d" % m, staggered=False))
        except BaseException as e:                # noqa: BLE001
            recs.append(dict(tag="nodal_ctrl_m%d" % m,
                             raised=type(e).__name__, msg=str(e)[:200]))
        for rp in (0.7, 0.85):
            try:
                r = census_one(m, 6.0, 120, _uni(1.96), 2.0,
                               "pml%g_m%d" % (rp, m), R_pml=rp * 6.0,
                               staggered=False)
                r["R_pml_frac"] = rp
                recs.append(r)
            except BaseException as e:            # noqa: BLE001
                recs.append(dict(tag="pml%g_m%d" % (rp, m),
                                 raised=type(e).__name__, msg=str(e)[:200]))
    return recs


def channel_counts_near_cutoff(fast=False):
    """THE DECISION, end to end: the R/T channel count and the energy closure
    of a near-cutoff BORStack solve, which is what actually moved."""
    from lumenairy.elements.bor.bor_stack import BORStack
    Rbig, N, n = 24.0, 120, 1.41
    eps = n ** 2
    rows = []
    base = census_one(1, Rbig, N, _uni(eps), 2.0, "base")
    q = base["q"]
    g2 = eps * 2.0 ** 2 - q ** 2
    g = np.sqrt(g2[np.abs(g2.imag) < 1e-9 * np.abs(g2.real)].real)
    g = np.unique(np.round(g[g > 0.1], 10))
    gam = float(g[len(g) // 3])
    deltas = ([1e-4, 1e-6, 1e-8, 1e-10] if fast else
              [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12])
    for m in ((1,) if fast else (0, 1, 2)):
        for d in deltas:
            k0 = gam / (n * np.sqrt(1.0 - d))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s = BORStack(Rbig=Rbig, m=m, N=N, n_superstrate=n,
                                 n_substrate=n, basis="fd")
                    s.add_layer(0.5, segments=[(8.0, eps * 1.21),
                                               (Rbig, eps)])
                    s.set_source(k0=k0)
                    r = s.solve()
                R, T = np.asarray(r["R"]), np.asarray(r["T"])
                rows.append(dict(m=m, delta=d, k0=k0, n_channels=int(R.size),
                                 closure=float(np.max(np.abs(R + T - 1.0)))
                                 if R.size else float("nan"),
                                 sumR=float(np.sum(R)),
                                 hash=_vh.hash_arrays(R, T)))
            except BaseException as e:            # noqa: BLE001
                rows.append(dict(m=m, delta=d, k0=k0,
                                 raised=type(e).__name__, msg=str(e)[:200]))
    return rows


def unit_scaling_invariance(fast=False):
    """THE k0 FLOOR.  Scale the whole unit system by ``s`` (lengths x s,
    wavenumbers / s).  The physics is identical; a band whose floor were a
    literal 1.0 would classify differently at s = 1e-6 and 1e6.  The decision
    -- channel count, classifier verdicts, energy closure -- must not move."""
    from lumenairy.elements.bor.bor_stack import BORStack
    rows = []
    for s in (1e-6, 1e-3, 1.0, 1e3, 1e6):
        Rbig, k0 = 24.0 * s, 2.0 / s
        rec = dict(scale=s, Rbig=Rbig, k0=k0)
        try:
            c = census_one(1, Rbig, 120, _uni(1.96), k0, "unit_s%g" % s)
            rec.update(n_prop_new=int(c["prop_new"].sum()),
                       n_prop_old=int(c["prop_old"].sum()),
                       n_modes=c["n_modes"],
                       worst_sigma_prop=float(np.max(c["sigma"][c["prop_new"]]))
                       if c["prop_new"].any() else None,
                       n_backward_in_forward=int(np.sum(c["prop_new"]
                                                        & (c["P"] < 0))))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = BORStack(Rbig=Rbig, m=1, N=120, n_superstrate=1.4,
                              n_substrate=1.4, basis="fd")
                st.add_layer(0.5 * s, segments=[(8.0 * s, 3.24),
                                                (Rbig, 1.96)])
                st.set_source(k0=k0)
                r = st.solve()
            R, T = np.asarray(r["R"]), np.asarray(r["T"])
            rec.update(n_channels=int(R.size),
                       closure=float(np.max(np.abs(R + T - 1.0))),
                       R_sorted=[float(x) for x in np.sort(R)[::-1][:6]])
        except BaseException as e:                # noqa: BLE001
            rec.update(raised=type(e).__name__, msg=str(e)[:200])
        rows.append(rec)
    return rows


def summarize(recs, label):
    """The four-sided bar, from the records themselves."""
    ok = [r for r in recs if r.get("sigma") is not None
          and np.size(r.get("sigma", []))]
    if not ok:
        return dict(label=label, n=0)
    sig = np.concatenate([r["sigma"] for r in ok])
    rho = np.concatenate([r["rho"] for r in ok])
    P = np.concatenate([r["P"] for r in ok])
    pn = np.concatenate([r["prop_new"] for r in ok])
    po = np.concatenate([r["prop_old"] for r in ok])
    pp = np.concatenate([r["phys_prop"] for r in ok])
    qq = np.concatenate([r["q"] for r in ok])
    return dict(label=label, n_modes=int(sig.size),
                n_layers=len(ok),
                worst_sigma_prop_new=float(np.max(sig[pn])) if pn.any() else None,
                min_sigma_evan_new=float(np.min(sig[~pn])) if (~pn).any() else None,
                worst_rho_prop_old=float(np.max(rho[po])) if po.any() else None,
                n_prop_new=int(pn.sum()), n_prop_old=int(po.sum()),
                n_disagree=int((pn != po).sum()),
                n_backward_in_forward_new=int(np.sum(pn & (P < 0))),
                n_backward_in_forward_old=int(np.sum(po & (P < 0))),
                sigma_min=float(np.min(sig)), sigma_max=float(np.max(sig)),
                n_phys_prop=int(pp.sum()),
                worst_sigma_phys_prop=float(np.max(sig[pp])) if pp.any() else None,
                min_sigma_phys_evan=float(np.min(sig[~pp])) if (~pp).any() else None,
                n_phys_prop_called_evan_new=int(np.sum(pp & ~pn)),
                n_phys_prop_called_evan_old=int(np.sum(pp & ~po)),
                n_phys_evan_called_prop_new=int(np.sum(~pp & pn)),
                n_phys_evan_called_prop_old=int(np.sum(~pp & po)),
                n_phys_prop_backward=int(np.sum(pp & (P < -0.5))),
                n_backward_any=int(np.sum(P < -0.5)),
                n_growing_prop=int(np.sum(pn & (qq.imag < -1e-300))),
                worst_growing_imag=float(np.min(qq.imag[pn])) if pn.any() else None)


def main():
    build = sys.argv[1]
    fast = "--fast" in sys.argv
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    out = dict(arm=a, build=build, fast=fast,
               band_new=_BAND_NEW, band_old=_BAND_OLD)

    with _vh.timed("ordinary"):
        ordn = ordinary_population()
    out["ordinary"] = [_pack(r) for r in ordn]
    out["sum_ordinary"] = summarize(ordn, "ordinary lossless")
    print("  ordinary", out["sum_ordinary"], flush=True)

    with _vh.timed("cutoff"):
        cut = cutoff_ladder(fast)
    out["cutoff"] = [_pack(r) for r in cut]
    out["sum_cutoff"] = summarize(cut, "deep cutoff")
    print("  cutoff", out["sum_cutoff"], flush=True)

    with _vh.timed("lossy"):
        lo = lossy_ladder()
    out["lossy"] = [_pack(r) for r in lo]
    # the SIGNAL side, per Im(n): the smallest sigma a genuinely lossy layer
    # exhibits at that loss -- what the band must NOT reach
    sig_by_im = {}
    for r in lo:
        s = r["sigma"]
        # exclude the layer's own evanescent tail: take the modes whose
        # |Re q| dominates (physically propagating in a lossy medium)
        prop = r["phys_prop"]
        v = s[prop]
        if v.size:
            k = "%g" % r["im_n"]
            sig_by_im.setdefault(k, []).append(float(np.min(v)))
    out["signal_by_im_n"] = {k: dict(min_sigma=min(v), max_sigma=max(v))
                             for k, v in sig_by_im.items()}
    print("  signal", out["signal_by_im_n"], flush=True)

    with _vh.timed("pml"):
        pm = pml_population()
    out["pml"] = [_pack(r) for r in pm]
    out["sum_pml"] = summarize([r for r in pm if r.get("R_pml_frac")], "pml")
    out["sum_nodal_ctrl"] = summarize([r for r in pm if not r.get("R_pml_frac")], "nodal control")

    with _vh.timed("channels"):
        out["channels"] = channel_counts_near_cutoff(fast)
    print("  channel counts",
          sorted({r.get("n_channels") for r in out["channels"]}), flush=True)

    with _vh.timed("units"):
        out["units"] = unit_scaling_invariance(fast)

    o = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") \
        else "."
    _vh.dump("%s/v2_band_%s_%s_%s_t%s.json"
             % (o, build, a["platform"], a["loaded_kernel"],
                a["blas_threads"]), out)


if __name__ == "__main__":
    main()
