"""Q5c -- the MECHANISM behind audit item O2, measured at the mode level.

Hypothesis to test, stated before measuring: the blow-up is a FORWARD/BACKWARD
MISCLASSIFICATION in ``rcwa/_core._select_forward_flux``.  That selector
classifies a mode by its net Poynting z-flux, EXCEPT that a "STABILITY band"
(``|Re gam| > 0.5``) forces the decay-sign rule -- so a mode whose decay rate
sits just BELOW 0.5 while its projection-noise flux points the wrong way is
kept on the wrong side, and the propagation S-matrix then carries a GROWING
exponential ``exp(|Re gam| k0 L)``.  A slanted layer's generator is the 4N
convection generator, whose spectrum is denser and less real than a vertical
layer's, so it should meet that band more readily.

The probe monkeypatches ``_layer_modes_projected`` inside ``stack2d``'s
namespace (probe-side only, no library edit) and records, per layer:

  * how many FORWARD modes have ``Re(gam) < -tol``   (they GROW along +z),
  * how many BACKWARD modes have ``Re(gam) > +tol``  (they GROW along -z),
  * the largest ``|Re gam|`` among those and the growth factor it puts into the
    cascade, ``exp(|Re gam| k0 d)``,
  * the offending modes' normalized flux ``|Sz| / max|Sz|``, so the "the flux
    said forward" half of the hypothesis is measured and not assumed.
"""
from __future__ import annotations

import math
import time
import warnings

import _lib as L
import numpy as np

TOL = 1e-6


class ModeProbe:
    def __init__(self):
        self.rows = []

    def install(self, S2, k0, thicknesses):
        self._saved = S2._layer_modes_projected
        f0 = self._saved
        self._k0 = k0
        self._t = list(thicknesses)
        self._n = 0

        def patched(*a, **kw):
            out = f0(*a, **kw)
            if len(out) == 6:
                Wf, Vf, lf, Wb, Vb, lb = out
                d = self._t[min(self._n, len(self._t) - 1)]
                self.rows.append(self._row(Wf, Vf, lf, Wb, Vb, lb, d))
            self._n += 1
            return out

        S2._layer_modes_projected = patched

    def restore(self, S2):
        S2._layer_modes_projected = self._saved

    def _sz(self, W, V):
        n = W.shape[0] // 2
        Ex, Ey = W[:n, :], W[n:, :]
        Hx, Hy = V[:n, :] / 1j, V[n:, :] / 1j
        return np.real(np.sum(Ex * np.conj(Hy) - Ey * np.conj(Hx), axis=0))

    def _row(self, Wf, Vf, lf, Wb, Vb, lb, d):
        szf, szb = self._sz(Wf, Vf), self._sz(Wb, Vb)
        mx = max(float(np.max(np.abs(szf))), float(np.max(np.abs(szb))), 1.0)
        gf, gb = np.real(lf), np.real(lb)
        bad_f = np.where(gf < -TOL)[0]        # FORWARD but growing
        bad_b = np.where(gb > +TOL)[0]        # BACKWARD but growing
        worst = 0.0
        if bad_f.size:
            worst = max(worst, float(np.max(-gf[bad_f])))
        if bad_b.size:
            worst = max(worst, float(np.max(gb[bad_b])))
        return dict(
            n_forward=int(lf.size), n_backward=int(lb.size),
            n_growing_forward=int(bad_f.size),
            n_growing_backward=int(bad_b.size),
            worst_growth_rate=worst,
            growth_factor=float(math.exp(worst * self._k0 * d)),
            k0d=float(self._k0 * d),
            bad_forward_reGam=[round(float(x), 6) for x in gf[bad_f][:6]],
            bad_backward_reGam=[round(float(x), 6) for x in gb[bad_b][:6]],
            bad_forward_relflux=[round(float(abs(szf[i]) / mx), 8)
                                 for i in bad_f[:6]],
            bad_backward_relflux=[round(float(abs(szb[i]) / mx), 8)
                                  for i in bad_b[:6]],
            in_stability_band=[bool(abs(x) > 0.5) for x in
                               list(gf[bad_f][:6]) + list(gb[bad_b][:6])])


# the EXACT fixture the fix's probes rejected
WL = 0.68e-6
PX = PY = 1.20e-6
D1, DF, FEPS = 0.50e-6, 0.25e-6, 3.6
NSUP, NSUB = 1.0, 1.5
TSL = 0.5
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
MOUNTS = dict(normal=(0.0, 0.0),
              oblique25=(math.radians(25.0), 0.0),
              conical25_40=(math.radians(25.0), math.radians(40.0)),
              oblique40=(math.radians(40.0), 0.0))


def cell(n=6):
    c = np.ones((n, n), dtype=complex)
    c[:, 0:n // 2] = np.repeat(XPROF, n // 6)[:, None]
    return c


def run(mount, M, kind):
    from lumenairy.elements.pmm import stack2d as S2
    st = S2.PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                             n_orders=M)
    sl = (TSL, 0.0) if kind.startswith("slant") else None
    st.add_layer(D1, eps_cell=cell(), slant=sl)
    if kind.endswith("over_film"):
        st.add_layer(DF, eps=FEPS)
    th, ph = MOUNTS[mount]
    st.set_source(WL, theta=th, phi=ph)
    pr = ModeProbe()
    pr.install(S2, 2.0 * np.pi / WL, [D1, DF])
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _o, R, T, _J = st.solve()
        rec = dict(outcome="SOLVED",
                   RT=float(np.max(np.asarray(R).sum(axis=1)
                                   + np.asarray(T).sum(axis=1))),
                   warned=[str(x.message)[:70] for x in w])
    except Exception as e:                              # noqa: BLE001
        rec = dict(outcome="RAISE", exc=type(e).__name__, msg=str(e)[:160])
    finally:
        pr.restore(S2)
    rec["modes"] = pr.rows
    return rec


def main():
    t0 = time.time()
    res = {}
    for mount in MOUNTS:
        for M in (3, 5, 7, 9):
            for kind in ("slant_over_film", "slant_only"):
                k = "%s|%s|M%d" % (kind, mount, M)
                r = run(mount, M, kind)
                res[k] = r
                m = r.get("modes") or [{}]
                print("%-38s RT=%-12.5g grow_f=%s grow_b=%s worst=%s "
                      "factor=%s"
                      % (k, r.get("RT") or float("nan"),
                         [x.get("n_growing_forward") for x in m],
                         [x.get("n_growing_backward") for x in m],
                         ["%.4f" % x.get("worst_growth_rate", 0) for x in m],
                         ["%.3g" % x.get("growth_factor", 1) for x in m]))
    # the VERTICAL control: same cell, same grid, no shear
    for mount in MOUNTS:
        for M in (3, 5, 7, 9):
            k = "vertical_over_film|%s|M%d" % (mount, M)
            r = run(mount, M, "vertical_over_film")
            res[k] = r
            print("%-38s RT=%-12.5g generalized layers=%d"
                  % (k, r.get("RT") or float("nan"), len(r.get("modes") or [])))
    res["_seconds"] = round(time.time() - t0, 1)
    L.dump("q5c_mechanism", res)


if __name__ == "__main__":
    main()
