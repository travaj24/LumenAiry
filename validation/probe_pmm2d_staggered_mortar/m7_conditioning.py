"""M7 -- CONDITIONING CENSUS of the 2-D mortar, in M1's instruments.

M1's finding stands: an EXPLICIT inverse is the exposed site, a backward-stable
``solve`` is not, and the discriminator is the EQUILIBRATED reciprocal
condition ``_rcond_1_equilibrated`` (M1 S2.2 -- exact, and free once the
inverse exists).  The 2-D mortar has three sites: the E-row solve
(``MassE_B W_B``), the H-row solve (``MassH_A V_A``) and the cascade
denominator ``I + BA`` (the only one that would carry a guard if this shipped).
This census reports all three across the M3 / M4 / M6 ladders, against M1's
shipped screen ``_INV_RCOND_SCREEN = 1e-8``, plus the far-field Rayleigh
projection's rank/residual (the T3-3 class) on each END grid.

Also measured here: the SEPARABLE Kronecker application against the dense
``kron`` -- the design claim that the 2-D cross-mass must never be
materialised."""
import json
import time
import warnings
import numpy as np
import mortar2d
from mortar2d import guard, MortarStack2D, GridOps, CrossOps, kron_apply
print("lumenairy:", guard(), flush=True)
from lumenairy.elements.rcwa import _core as _rc

warnings.simplefilter("ignore")
PX = PY = 1.2e-6
WL = 0.85e-6
EPS_H, EPS_P = 2.25, 9.0


def pillar(N, k):
    c = np.full((N, N), EPS_H + 0j)
    c[:k, :k] = EPS_P
    return c


def stripe(N, hi):
    c = np.full((N, N), 2.0 + 0j)
    c[:hi, :] = 6.0
    return c


CONFIGS = [
    ("M3 pillar (2,3)", [(0.16e-6, pillar(2, 1)), (0.13e-6, pillar(3, 1))],
     0.18, 0.35),
    ("M3 pillar (2,6)", [(0.16e-6, pillar(2, 1)), (0.13e-6, pillar(6, 2))],
     0.18, 0.35),
    ("M4 stripe (2,3)", [(0.20e-6, stripe(2, 1)), (0.15e-6, stripe(3, 1))],
     0.20, 0.0),
    ("M6 taper (2,3,6)", [(0.12e-6, pillar(2, 1)), (0.12e-6, pillar(3, 1)),
                          (0.12e-6, pillar(6, 1))], 0.18, 0.0),
]

by_site = {}
rows = []
for name, layers, th, ph in CONFIGS:
    for M in (4, 5, 6):
        mortar2d.CENSUS = []
        s = MortarStack2D(PX, PY, n_modes=M, n_orders=2)
        for t, c in layers:
            s.add_layer(t, eps_cell=c)
        s.set_source(WL, theta=th, phi=ph)
        try:
            s.solve(jones=False)
        except Exception as exc:                          # noqa: BLE001
            print(f"  {name} M={M}: REFUSED -- {type(exc).__name__}: {exc}")
            mortar2d.CENSUS = None
            continue
        for site, n, rc in mortar2d.CENSUS:
            by_site.setdefault(site, []).append(rc)
            rows.append(dict(cfg=name, M=M, site=site, n=n, rcond=rc))
        worst = min(rc for _s, _n, rc in mortar2d.CENSUS)
        print(f"  {name:20s} M={M}  calls {len(mortar2d.CENSUS):2d}  "
              f"worst equilibrated rcond {worst:.2e}", flush=True)
        mortar2d.CENSUS = None

print(f"\nSITE SUMMARY (M1 screen _INV_RCOND_SCREEN = "
      f"{_rc._INV_RCOND_SCREEN:.0e}; below it M1 pays for the confirming "
      f"residual, and NOTHING is refused -- the inverse refusal was withdrawn)")
summ = {}
for site, v in sorted(by_site.items()):
    summ[site] = dict(n=len(v), lo=float(min(v)), hi=float(max(v)),
                      below_screen=int(sum(1 for x in v
                                           if x < _rc._INV_RCOND_SCREEN)))
    print(f"  {site:38s} calls {len(v):3d}  rcond_eq "
          f"[{min(v):.2e}, {max(v):.2e}]  below screen: "
          f"{summ[site]['below_screen']}")

print("\nSEPARABLE vs DENSE cross-mass application")
sep_rows = []
taux = np.exp(-1j * 0.31 * 2 * np.pi)
for (Na, Nb, M) in ((2, 3, 6), (3, 6, 6), (4, 6, 8), (6, 12, 6)):
    ga = GridOps(PX, PY, Na, M, taux, 1.0 + 0j)
    gb = GridOps(PX, PY, Nb, M, taux, 1.0 + 0j)
    cr = CrossOps(ga, gb)
    X = np.random.default_rng(1).normal(size=(gb.qq, 8)) + 0j
    t0 = time.perf_counter()
    ysep = kron_apply(cr.C1[0], cr.C1[1], X)
    t_sep = time.perf_counter() - t0
    t0 = time.perf_counter()
    K = np.kron(cr.C1[0], cr.C1[1])
    ydense = K @ X
    t_dense = time.perf_counter() - t0
    rel = float(np.max(np.abs(ysep - ydense))) / float(np.max(np.abs(ydense)))
    mb = K.nbytes / 2 ** 20
    fmb = (cr.C1[0].nbytes + cr.C1[1].nbytes) / 2 ** 20
    sep_rows.append(dict(Na=Na, Nb=Nb, M=M, rel=rel, dense_MB=mb,
                         factors_MB=fmb, t_sep=t_sep, t_dense=t_dense))
    print(f"  grids({Na},{Nb}) M={M}: dense kron {mb:8.2f} MB vs factors "
          f"{fmb:6.3f} MB ({mb/fmb:7.1f}x)  apply {t_dense/t_sep:6.1f}x "
          f"faster  identity to {rel:.1e}")

json.dump(dict(rows=rows, summary=summ, separable=sep_rows),
          open("validation/probe_pmm2d_staggered_mortar/m7_conditioning.json",
               "w"), indent=1)
