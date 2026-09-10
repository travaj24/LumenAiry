"""CLASS A -- the TWO-SIDED margins of the CANDIDATE band.

A bar without both sides measured is not a bar.  The candidate replaces

    prop = |Im q| < 1e-9 * max(|Re q|, 1e-300)          (the mode's OWN scale)
by
    prop = |Im q| <= band * max(max|q|, k0)             (the SPECTRUM's scale)

so the discriminating ratio changes from ``rho = |Im q| / |Re q|`` to
``sigma = |Im q| / max(max|q|, k0)``, and BOTH populations have to be
re-measured against ``sigma``, not against ``rho``:

  NOISE side (the band MUST reach it) -- a LOSSLESS propagating mode, whose
  ``Im q`` is nothing but the eigensolver's backward error.  Measured on the
  ordinary census fixtures AND at the deep cutoff, where the Cartesian band's
  noise side was the binding population.

  SIGNAL side (the band must NOT reach it) -- a genuinely LOSSY mode, whose
  ``Im q`` is physics.  Measured over an ``Im(n)`` ladder so the crossing
  imaginary index is located exactly, and reported as the smallest ``sigma``
  a real lossy mode reaches.

Also reported: the same two sides for the SHIPPED band on the same fixtures,
so the two bars are compared on one page rather than across two probes.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import dump, mode_table, pin_tree  # noqa: E402

print("TREE", pin_tree())
RBIG = 24.0
NFD = 120
BAND = 1e-8


def fd(m, k0, eps, N=NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, RBIG, N,
                       lambda r, e=eps: np.full_like(r, e, dtype=complex),
                       float(k0), staggered=True)


def sem(m, k0, eps, degree=8):
    from lumenairy.elements.bor.sem_radial import SemRadialMesh, sem_layer_modes
    n = abs(np.sqrt(complex(eps)).real)
    max_el = degree * (2 * np.pi / (k0 * max(n, 1e-3))) / 8.0
    ne = max(2, int(np.ceil(RBIG / max_el)))
    mesh = SemRadialMesh(np.linspace(0.0, RBIG, ne + 1),
                         [(complex(eps),) * 3] * ne, degree)
    return sem_layer_modes(mesh, m, float(k0))


def sigmas(L, k0):
    """(sigma, rho, physically-propagating mask) for one layer."""
    q, _flux, _rel, rho = mode_table(L, sem=("mesh" in L))
    scale = max(float(np.max(np.abs(q))), float(k0))
    sigma = np.abs(q.imag) / scale
    phys = np.abs(q.real) > np.abs(q.imag)
    return sigma, rho, phys, q


def gamma_of(m, eps, idx=2):
    q = np.asarray(fd(m, 2.0, eps)["q"])
    g = np.sqrt(2.0 ** 2 * complex(eps) - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    krn = os.environ.get("OPENBLAS_CORETYPE", "(default)")
    noise, signal = [], []

    # --- NOISE side, ORDINARY geometry ---------------------------------- #
    for basis in ("fd", "sem"):
        for n in (1.41, 1.50, 2.00):
            for m in (0, 1, 2):
                for k0 in (0.8, 2.0, 3.5):
                    L = (fd(m, k0, n ** 2) if basis == "fd"
                         else sem(m, k0, n ** 2))
                    s, r, phys, q = sigmas(L, k0)
                    if phys.any():
                        noise.append(dict(
                            kind="ordinary", basis=basis, n=n, m=m, k0=k0,
                            sigma_max=float(s[phys].max()),
                            rho_max=float(r[phys].max()),
                            n_phys=int(phys.sum())))

    # --- NOISE side, DEEP CUTOFF (the binding population) ---------------- #
    for basis in ("fd", "sem"):
        for m in (0, 1, 2):
            g = gamma_of(m, 1.41 ** 2)
            for dl in [10.0 ** (-e) for e in range(4, 27, 2)]:
                k0 = g / (1.41 * np.sqrt(1.0 - dl))
                L = (fd(m, k0, 1.41 ** 2) if basis == "fd"
                     else sem(m, k0, 1.41 ** 2))
                s, r, phys, q = sigmas(L, k0)
                if phys.any():
                    noise.append(dict(
                        kind="cutoff", basis=basis, n=1.41, m=m,
                        k0=float(k0), delta=float(dl),
                        sigma_max=float(s[phys].max()),
                        rho_max=float(r[phys].max()),
                        n_phys=int(phys.sum())))

    # --- SIGNAL side: an Im(n) ladder ----------------------------------- #
    for basis in ("fd", "sem"):
        for m in (0, 1, 2):
            for ex in range(1, 13):
                ni = 10.0 ** (-ex)
                eps = complex(1.41 + 1j * ni) ** 2
                L = fd(m, 2.0, eps) if basis == "fd" else sem(m, 2.0, eps)
                s, r, phys, q = sigmas(L, 2.0)
                if phys.any():
                    signal.append(dict(
                        basis=basis, m=m, imag_n=ni,
                        sigma_min=float(s[phys].min()),
                        rho_min=float(r[phys].min()),
                        n_phys=int(phys.sum())))

    payload = dict(kernel=krn, band=BAND, noise=noise, signal=signal)
    dump(f"a6_band_margins_{tag}_{krn}.json", payload)

    ordn = [x for x in noise if x["kind"] == "ordinary"]
    cutn = [x for x in noise if x["kind"] == "cutoff"]
    print("\n== NOISE SIDE (a lossless propagating mode: the band MUST "
          "reach it) ==")
    for nm, pop in (("ordinary geometry", ordn), ("deep cutoff", cutn)):
        sw = max(x["sigma_max"] for x in pop)
        rw = max(x["rho_max"] for x in pop)
        print(f"  {nm:<20} n={len(pop):4d}   "
              f"CANDIDATE sigma worst {sw:.4e} "
              f"({np.log10(BAND / sw):+.2f} decades below band={BAND:.0e})   "
              f"SHIPPED rho worst {rw:.4e} "
              f"({np.log10(1e-9 / rw):+.2f} decades below 1e-9)")
    print("\n== SIGNAL SIDE (a genuinely lossy mode: the band must NOT reach "
          "it) ==")
    print("  Im(n)      CANDIDATE sigma_min   decades above band   "
          "SHIPPED rho_min   decades above 1e-9")
    for ex in range(1, 13):
        ni = 10.0 ** (-ex)
        v = [x for x in signal if x["imag_n"] == ni]
        if not v:
            continue
        sm = min(x["sigma_min"] for x in v)
        rm = min(x["rho_min"] for x in v)
        print(f"  {ni:.0e}   {sm:.4e}          {np.log10(sm / BAND):+6.2f}"
              f"              {rm:.4e}      {np.log10(rm / 1e-9):+6.2f}")
    sw_all = max(x["sigma_max"] for x in noise)
    # the signal side that binds: the SMALLEST loss the band must still reject
    print("\n== THE TWO-SIDED GAP ==")
    print(f"  CANDIDATE band {BAND:.0e}:")
    print(f"    noise side worst  {sw_all:.4e}  -> "
          f"{np.log10(BAND / sw_all):.2f} decades of room ABOVE the noise")
    for ni in (1e-3, 1e-4, 1e-5, 1e-6):
        v = [x for x in signal if x["imag_n"] == ni]
        if v:
            sm = min(x["sigma_min"] for x in v)
            print(f"    signal side at Im(n)={ni:.0e}: {sm:.4e}  -> "
                  f"{np.log10(sm / BAND):.2f} decades of room BELOW the "
                  f"signal")
    rw_all = max(x["rho_max"] for x in noise)
    print(f"  SHIPPED band 1e-9:")
    print(f"    noise side worst  {rw_all:.4e}  -> "
          f"{np.log10(1e-9 / rw_all):.2f} decades")
    bad = [x for x in noise if x["rho_max"] >= 1e-9]
    print(f"    rungs whose noise side CROSSES the shipped band: "
          f"{len(bad)}/{len(noise)}")
    badc = [x for x in noise if x["sigma_max"] >= BAND]
    print(f"    rungs whose noise side CROSSES the candidate band: "
          f"{len(badc)}/{len(noise)}")


if __name__ == "__main__":
    main()
