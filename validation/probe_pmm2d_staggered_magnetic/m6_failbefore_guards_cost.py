"""Probe 6 -- FAIL-BEFORE (every new magnetic term is load-bearing, including
the R-vs-Gram separation), the guard census (G7) and the cost measurement.

The four knockouts, each applied through the shipped code (no forked copy):

  K1  the R-vs-GRAM separation: ``Ggram_blocks = None`` makes the Eq.-25 H
      recovery project with the PENCIL's R = C[chi_t]C instead of the plain
      block Gram -- the trap this build exists to avoid.
  K2  ``chi33 -> 1``: S_tt (the curl-curl operator) loses its permeability
      weight (Eq. 20 / A42).
  K3  ``chi_t -> I``: R (Eq. 24 / A39) and K_tz (Eq. 21 / A43) lose theirs,
      chi33 kept.
  K4  ``chi12 = chi21 = 0``: only the MIXED (C-rotated off-diagonal) blocks go.

K1-K3 are measured against the ANALYTIC Airy oracle on a uniform ISOTROPIC
magnetic slab (where an isotropic mu already exercises all three); K4 needs an
anisotropic mu, so all four are also measured by the DUALITY residual on a
uniform (eps, mu) tensor pair, where the intact build is spectral (~1e-14).
"""
import sys
import time
import tracemalloc

import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_mag"), \
    lumenairy.__file__

sys.path.insert(0, "validation/probe_pmm2d_staggered_magnetic")
import m2_analytic_slab as m2  # noqa: E402
import m3_duality as m3  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

P, WL, DEP = 0.70e-6, 0.55e-6, 0.28e-6
EYE = np.eye(3, dtype=complex)
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
LC2 = uniaxial_tensor(1.15, 1.35, np.pi / 2, phi=-0.30)

_ORIG_CHI = Granet2DTransverseE._chi_maps
_ORIG_ASM = Granet2DTransverseE._assemble


def _install(kind):
    if kind == "K1":
        def asm(self):
            _ORIG_ASM(self)
            self.Ggram_blocks = None          # H recovery projects with R
        Granet2DTransverseE._assemble = asm
        return
    def chi(self):
        c = _ORIG_CHI(self)
        if c is None:
            return None
        c11, c12, c21, c22, c33 = c
        if kind == "K2":
            return c11, c12, c21, c22, np.ones_like(c33)
        if kind == "K3":
            return (np.ones_like(c11), np.zeros_like(c12), np.zeros_like(c21),
                    np.ones_like(c22), c33)
        if kind == "K4":
            return c11, np.zeros_like(c12), np.zeros_like(c21), c22, c33
        raise AssertionError(kind)
    Granet2DTransverseE._chi_maps = chi


def _restore():
    Granet2DTransverseE._chi_maps = _ORIG_CHI
    Granet2DTransverseE._assemble = _ORIG_ASM


def analytic_residual():
    r = m2.row("eps=4 mu=2 th=0.35", 4.0, 2.0, 0.35, 8)
    return max(r["dR_te"], r["dT_te"], r["dR_tm"], r["dT_tm"])


def duality_residual():
    ue = np.broadcast_to(LC, (2, 2, 3, 3)).copy()
    um = np.broadcast_to(LC2, (2, 2, 3, 3)).copy()
    return m3.duality("uniform eps=LC, mu=LC2", ue, um, 0.30, 0.7, 7)[0]


if __name__ == "__main__":
    print("=== FAIL-BEFORE: knock each magnetic term out through the shipped "
          "code")
    base_a, base_d = analytic_residual(), duality_residual()
    print(f"  intact                       analytic={base_a:.3e}  "
          f"duality={base_d:.3e}")
    for k in ("K1", "K2", "K3", "K4"):
        _install(k)
        try:
            a = analytic_residual()
            d = duality_residual()
        finally:
            _restore()
        print(f"  {k} knocked out              analytic={a:.3e}  "
              f"duality={d:.3e}")
    print()

    print("=== G7 guards")
    def expect(tag, fn, exc):
        try:
            fn()
        except exc as e:
            print(f"  {tag:<44s} {type(e).__name__}: {str(e)[:62]}...")
            return
        except Exception as e:            # noqa: BLE001
            print(f"  {tag:<44s} WRONG TYPE {type(e).__name__}: {e}")
            return
        print(f"  {tag:<44s} *** NO RAISE ***")

    oop_mu = np.broadcast_to(EYE, (2, 2, 3, 3)).copy()
    oop_mu[..., 0, 2] = oop_mu[..., 2, 0] = 0.3
    oop_eps = np.broadcast_to(4.0 * EYE, (2, 2, 3, 3)).copy()
    oop_eps[..., 1, 2] = oop_eps[..., 2, 1] = 0.4
    tiny = np.broadcast_to(EYE, (2, 2, 3, 3)).copy()
    tiny[..., 0, 2] = 1e-16
    sing = np.broadcast_to(EYE, (2, 2, 3, 3)).copy()
    sing[..., 0, 1] = sing[..., 1, 0] = 1.0        # det[mu_t] = 0
    zz = np.broadcast_to(EYE, (2, 2, 3, 3)).copy()
    zz[..., 2, 2] = 0.0
    ec = np.broadcast_to(4.0 * EYE, (2, 2, 3, 3)).copy()
    J = pmm_jones_2d_staggered
    expect("OUT-OF-PLANE mu", lambda: J(P, P, ec, 1.0, 1.0, DEP, WL,
                                        mu_cell=oop_mu, degree=4),
           NotImplementedError)
    expect("mu with an OUT-OF-PLANE eps",
           lambda: J(P, P, oop_eps, 1.0, 1.0, DEP, WL,
                     mu_cell=np.broadcast_to(EYE, (2, 2, 3, 3)).copy(),
                     degree=4), NotImplementedError)
    expect("magnetic half-space",
           lambda: J(P, P, ec, 1.0, 1.0, DEP, WL, mu_superstrate=2.0,
                     degree=4), NotImplementedError)
    expect("singular [mu_t]", lambda: J(P, P, ec, 1.0, 1.0, DEP, WL,
                                        mu_cell=sing, degree=4), ValueError)
    expect("m33 = 0", lambda: J(P, P, ec, 1.0, 1.0, DEP, WL, mu_cell=zz,
                                degree=4), ValueError)
    expect("mu_cell of the wrong shape",
           lambda: J(P, P, ec, 1.0, 1.0, DEP, WL,
                     mu_cell=np.ones((2, 2, 2, 2)), degree=4), ValueError)
    expect("mu_cell grid != eps_cell grid",
           lambda: J(P, P, ec, 1.0, 1.0, DEP, WL, mu_cell=np.ones((3, 3)),
                     degree=4), ValueError)
    expect("non-square mu_cell",
           lambda: J(P, P, ec, 1.0, 1.0, DEP, WL, mu_cell=np.ones((2, 3)),
                     degree=4), ValueError)
    expect("zero scalar mu_cell",
           lambda: J(P, P, ec, 1.0, 1.0, DEP, WL, mu_cell=np.zeros((2, 2)),
                     degree=4), ValueError)
    expect("both mu and mu_cell",
           lambda: PMM2DStackPure(P, P).add_layer(DEP, eps=4.0, mu=2.0,
                                                  mu_cell=np.ones((2, 2))),
           ValueError)
    expect("_homog_geom_cache on a magnetic solver", lambda: __import__(
        "lumenairy.elements.pmm.twod_staggered", fromlist=["x"]
    )._homog_geom_cache(Granet2DTransverseE(
        P, P, 2, 2, 4, np.full((2, 2), 4.0 + 0j),
        mu_cell=np.full((2, 2), 2.0 + 0j))), ValueError)
    # the 1e-16 stray must NOT trip the block-form guard
    o, R, T, Jm = J(P, P, ec, 1.0, 1.0, DEP, WL, mu_cell=tiny, degree=5,
                    n_orders=2)
    o2, R2, T2, J2 = J(P, P, ec, 1.0, 1.0, DEP, WL,
                       mu_cell=np.broadcast_to(EYE, (2, 2, 3, 3)).copy(),
                       degree=5, n_orders=2)
    print(f"  {'1e-16 m13 stray: solves, dR vs exact-I':<44s} "
          f"{np.max(np.abs(R - R2)):.2e}")
    print()

    print("=== COST at (3,3), M=8 -- magnetic vs the eps-only tensor path")
    ec3 = np.empty((3, 3, 3, 3), dtype=complex)
    ec3[:] = LC
    ec3[0, 0] = 4.0 * EYE
    mc3 = np.empty((3, 3, 3, 3), dtype=complex)
    mc3[:] = LC2
    mc3[1, 1] = 1.6 * EYE
    for tag, mu in (("eps-only tensor", None), ("magnetic (eps + mu)", mc3)):
        tracemalloc.start()
        t0 = time.perf_counter()
        sol = Granet2DTransverseE(P / WL, P / WL, 3, 3, 8, ec3,
                                  k0=2 * np.pi, mu_cell=mu)
        t_asm = time.perf_counter() - t0
        t0 = time.perf_counter()
        _region_modes(sol)
        t_eig = time.perf_counter() - t0
        peak = tracemalloc.get_traced_memory()[1] / 2 ** 20
        tracemalloc.stop()
        print(f"  {tag:<22s} assemble {t_asm:6.2f} s  region-eig "
              f"{t_eig:6.2f} s  peak {peak:7.1f} MiB  "
              f"Ggram_blocks={'None' if sol.Ggram_blocks is None else '2 x q^2'}")
        del sol
