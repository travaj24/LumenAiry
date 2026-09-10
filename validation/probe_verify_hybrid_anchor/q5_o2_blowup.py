"""Q5 -- audit item O2: the slanted-layer-over-a-film cascade that blew up to
``sum R + T = 2.6e+27``.

Reproduce it, MAP which ``(n_orders, mount)`` pairs blow up on two fixtures,
find the mechanism BY MEASUREMENT rather than by reading, and settle whether
the library is LOUD about it (a warning / a refusal) or silently wrong.

The instrumentation monkeypatches the three GENERALIZED cascade helpers INSIDE
``stack2d``'s namespace (probe-side only, no library edit) and records, per
solve: the forward/backward eigenvalue split, ``cond`` of the mode matrix each
interface inverts, ``cond`` of the ``T22`` block, and the largest propagation
factor the star actually forms.
"""
from __future__ import annotations

import math
import time
import warnings

import _lib as L
import numpy as np

# fixture 1: the audit's own shape -- a cell CONTAINING eps = 1.0 (the
# superstrate), px = 1.2 um, wl = 0.68 um, so the +1 order sits at
# |alpha| = 0.9893 against a 1.0 cut-off.
F1 = dict(px=1.2e-6, py=1.2e-6, wl=0.68e-6, d=0.5e-6, slant=(1.0, 0.0),
          film=(0.15e-6, 2.25), n_sup=1.0, n_sub=1.0,
          cell=np.array([[1.00, 1.00, 3.24, 3.24],
                         [3.24, 1.00, 1.00, 1.44],
                         [1.00, 1.00, 1.00, 1.00],
                         [1.44, 2.10, 1.00, 1.00],
                         [1.00, 1.00, 2.56, 1.00],
                         [1.00, 1.00, 1.00, 1.00]]))
# fixture 2: LOWER contrast, no eps = 1.0 pixel, a different period so the
# near-cut-off order lands elsewhere.
F2 = dict(px=1.05e-6, py=1.05e-6, wl=0.68e-6, d=0.5e-6, slant=(0.8, 0.0),
          film=(0.15e-6, 2.05), n_sup=1.0, n_sub=1.0,
          cell=np.array([[1.30, 1.30, 1.95, 1.95],
                         [1.95, 1.30, 1.30, 1.60],
                         [1.30, 1.30, 1.30, 1.30],
                         [1.60, 1.72, 1.30, 1.30],
                         [1.30, 1.30, 1.80, 1.30],
                         [1.30, 1.30, 1.30, 1.30]]))

MOUNTS = dict(normal=(0.0, 0.0),
              oblique25=(math.radians(25.0), 0.0),
              conical25_40=(math.radians(25.0), math.radians(40.0)),
              oblique40=(math.radians(40.0), 0.0))


class Probe:
    """Records what the generalized cascade actually formed."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.if_cond_Mb = []
        self.if_cond_T22 = []
        self.lam_f_re = []
        self.lam_b_re = []
        self.max_prop = []

    def install(self, mod):
        import lumenairy.elements.rcwa._core as core
        self._saved = (mod._interface_smatrix_general,
                       mod._propagation_smatrix_general,
                       mod._propagation_star_general)
        i0, p0, s0 = self._saved

        def ifc(Ma, Mb):
            self.if_cond_Mb.append(float(np.linalg.cond(Mb)))
            T = np.linalg.solve(Mb, Ma)
            n2 = Ma.shape[0] // 2
            self.if_cond_T22.append(float(np.linalg.cond(T[n2:, n2:])))
            return i0(Ma, Mb)

        def prop(lam_f, lam_b, k0_L):
            self.lam_f_re.append((float(np.min(np.real(lam_f))),
                                  float(np.max(np.real(lam_f)))))
            self.lam_b_re.append((float(np.min(np.real(lam_b))),
                                  float(np.max(np.real(lam_b)))))
            self.max_prop.append(float(np.max(np.abs(
                np.concatenate([np.exp(-lam_f * k0_L),
                                np.exp(lam_b * k0_L)])))))
            return p0(lam_f, lam_b, k0_L)

        def star(S, lam_f, lam_b, k0_L):
            self.lam_f_re.append((float(np.min(np.real(lam_f))),
                                  float(np.max(np.real(lam_f)))))
            self.lam_b_re.append((float(np.min(np.real(lam_b))),
                                  float(np.max(np.real(lam_b)))))
            self.max_prop.append(float(np.max(np.abs(
                np.concatenate([np.exp(-lam_f * k0_L),
                                np.exp(lam_b * k0_L)])))))
            return s0(S, lam_f, lam_b, k0_L)

        mod._interface_smatrix_general = ifc
        mod._propagation_smatrix_general = prop
        mod._propagation_star_general = star
        del core

    def restore(self, mod):
        (mod._interface_smatrix_general, mod._propagation_smatrix_general,
         mod._propagation_star_general) = self._saved

    def summary(self):
        return dict(
            n_interfaces=len(self.if_cond_Mb),
            max_cond_Mb=max(self.if_cond_Mb) if self.if_cond_Mb else None,
            max_cond_T22=max(self.if_cond_T22) if self.if_cond_T22 else None,
            worst_lam_f_min_re=(min(a for a, _b in self.lam_f_re)
                                if self.lam_f_re else None),
            worst_lam_b_max_re=(max(b for _a, b in self.lam_b_re)
                                if self.lam_b_re else None),
            max_prop_factor=max(self.max_prop) if self.max_prop else None)


def solve_one(F, mount, n_orders, *, kind="slant_over_film", probe=None):
    from lumenairy.elements.pmm import stack2d as S2
    st = S2.PMM2DStackHybrid(F["px"], F["py"], n_superstrate=F["n_sup"],
                             n_substrate=F["n_sub"], n_orders=n_orders)
    sl = F["slant"] if kind in ("slant_over_film", "slant_only") else None
    st.add_layer(F["d"], eps_cell=F["cell"], slant=sl)
    if kind in ("slant_over_film", "vertical_over_film"):
        st.add_layer(F["film"][0], eps=F["film"][1])
    th, ph = MOUNTS[mount]
    st.set_source(F["wl"], theta=th, phi=ph)
    if probe is not None:
        probe.reset()
        probe.install(S2)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = st.solve()
        msgs = [str(x.message)[:150] for x in w]
        RT = float(np.max(np.asarray(out[1]).sum(axis=1)
                          + np.asarray(out[2]).sum(axis=1)))
        rec = dict(outcome="SOLVED", RT=RT, warnings=msgs)
    except Exception as e:                              # noqa: BLE001
        rec = dict(outcome="RAISE", exc=type(e).__name__, msg=str(e)[:200])
    finally:
        if probe is not None:
            probe.restore(S2)
    if probe is not None:
        rec["probe"] = probe.summary()
    return rec


def alphas(F, mount, n_orders):
    th, ph = MOUNTS[mount]
    a0x = math.sin(th) * math.cos(ph) * float(np.real(F["n_sup"]))
    a0y = math.sin(th) * math.sin(ph) * float(np.real(F["n_sup"]))
    out = []
    for m in range(-n_orders, n_orders + 1):
        for n in range(-n_orders, n_orders + 1):
            ax = a0x + m * F["wl"] / F["px"]
            ay = a0y + n * F["wl"] / F["py"]
            out.append((m, n, math.hypot(ax, ay)))
    cut = float(np.real(np.sqrt(complex(F["n_sup"]) ** 2)))
    near = sorted(out, key=lambda r: abs(r[2] - cut))[:3]
    return dict(cutoff=cut,
                nearest=[dict(m=m, n=n, alpha=a, gap=a - cut)
                         for m, n, a in near])


def main():
    t0 = time.time()
    res = {}
    for fname, F in (("F1_eps1_px1p2", F1), ("F2_lowcontrast_px1p05", F2)):
        grid = {}
        for mount in MOUNTS:
            for M in (3, 5, 7, 9, 11):
                r = solve_one(F, mount, M)
                grid["%s|M%d" % (mount, M)] = dict(
                    RT=r.get("RT"), outcome=r["outcome"],
                    warn=bool(r.get("warnings")),
                    warn0=(r.get("warnings") or [None])[0],
                    exc=r.get("exc"))
        res[fname] = dict(grid=grid,
                          alphas={m: alphas(F, m, 3) for m in MOUNTS})
        bad = {k: v for k, v in grid.items()
               if v["outcome"] == "RAISE" or (v["RT"] or 0) > 1.05}
        res[fname]["blowups"] = bad
        print("==", fname, "blowups:", len(bad), "of", len(grid))
        for k, v in sorted(bad.items()):
            print("   ", k, v["outcome"], v["RT"], "| warn:", v["warn0"])

    # ---- MECHANISM on the worst pair -------------------------------------
    mech = {}
    for fname, F in (("F1_eps1_px1p2", F1), ("F2_lowcontrast_px1p05", F2)):
        bad = res[fname]["blowups"]
        if not bad:
            mech[fname] = "no blow-up on this grid"
            continue
        key = max(bad, key=lambda k: (bad[k]["RT"] or 0.0))
        mount, M = key.split("|")
        M = int(M[1:])
        pr = Probe()
        rows = {}
        for kind in ("slant_over_film", "slant_only", "vertical_over_film"):
            rows[kind] = solve_one(F, mount, M, kind=kind, probe=pr)
        # the same pair one order lower / higher
        rows["slant_over_film_M-2"] = solve_one(F, mount, max(1, M - 2))
        rows["slant_over_film_M+2"] = solve_one(F, mount, M + 2)
        rows["alphas"] = alphas(F, mount, M)
        # the PURE engine on the same solid
        try:
            from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
            cell = F["cell"]
            n = max(cell.shape)
            sq = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    sq[i, j] = cell[i % cell.shape[0], j % cell.shape[1]]
            pu = PMM2DStackPure(F["px"], F["py"], n_superstrate=F["n_sup"],
                                n_substrate=F["n_sub"], n_modes=6, n_orders=3)
            pu.add_layer(F["d"], eps_cell=sq, slant=F["slant"])
            pu.add_layer(F["film"][0], eps=F["film"][1])
            th, ph = MOUNTS[mount]
            pu.set_source(F["wl"], theta=th, phi=ph)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                po = pu.solve()
            rows["pure_same_solid"] = dict(
                outcome="SOLVED",
                RT=float(np.max(np.asarray(po[1]).sum(axis=1)
                                + np.asarray(po[2]).sum(axis=1))),
                warnings=[str(x.message)[:120] for x in w])
        except Exception as e:                          # noqa: BLE001
            rows["pure_same_solid"] = dict(outcome="RAISE",
                                           exc=type(e).__name__,
                                           msg=str(e)[:200])
        mech[fname] = dict(worst_pair=key, rows=rows)
        print("-- mechanism", fname, key)
        for k, v in rows.items():
            print("   ", k, v)
    res["mechanism"] = mech
    res["seconds"] = round(time.time() - t0, 1)
    L.dump("q5_o2", res)


if __name__ == "__main__":
    main()
