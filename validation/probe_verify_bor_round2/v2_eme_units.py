"""V2 -- D13.  THE SAME EME CELL WRITTEN IN THREE UNIT SYSTEMS.

An independent cell (2 strips, Nx = 64, lambda = 1310 nm, a weak-gain high
region) -- not the round-2 probe's 1 um / 1550 nm / Nx = 96 cell -- plus a
SECOND cell deliberately sized so that ``max|ky|`` is sub-unity in MICROMETRES
as well, which is the population the literal 1.0 floor decided on and which the
round-2 probe's cell only reaches in nanometres.

Scaling: lengths x ``scale``, wavenumbers / ``scale``.  ``ky * scale`` is
therefore unit-free and MUST be identical in all three systems.  Spectra are
SORTED before differencing (an eigensolver's ordering is not stable between two
solves, so an elementwise difference of raw arrays measures a permutation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import lumenairy  # noqa: E402
from lumenairy.elements.eme import _branch  # noqa: E402
from lumenairy.elements.eme import eme_2d as E2  # noqa: E402
from lumenairy.elements.eme import eme_2d_vector as EV  # noqa: E402
from lumenairy.elements.eme.eme_diffraction import mode_match, plane_wave_orders  # noqa: E402

UNITS = dict(um=1.0, nm=1.0e3, m=1.0e-6)      # length multiplier from um


def _prov(name):
    import scipy
    import threadpoolctl
    return dict(tag=name, lumenairy_file=lumenairy.__file__,
                python=sys.version.split()[0], numpy=np.__version__,
                scipy=scipy.__version__,
                blas_arch=sorted({str(d.get("architecture"))
                                  for d in threadpoolctl.threadpool_info()}),
                blas_threads=sorted({int(d.get("num_threads", -1))
                                     for d in threadpoolctl.threadpool_info()}),
                env={k: os.environ.get(k) for k in
                     ("OPENBLAS_CORETYPE", "OMP_NUM_THREADS",
                      "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})


CELLS = {
    # (Lx_um, lam_um, Nx, eps_lo, eps_hi, depth_um, Ly_um)
    "A_ordinary": (1.0, 1.310, 64, 2.25, 9.0 - 1e-7j, 0.4, 1.0),
    # a LONG-period, low-contrast cell: max|ky| ~ 3.6 /um, so the literal 1.0
    # floor is silent in um, engages in nm.
    "B_subunit_in_nm": (2.0, 1.550, 48, 1.0, 2.1 - 1e-8j, 0.3, 2.0),
    # weak GAIN, which puts a population of roots just off the cut -- the
    # population whose ORIENTATION the band decides.
    "C_weakgain": (1.2, 1.550, 80, 1.0, 12.0 - 1e-6j, 0.5, 1.2),
}


def _strips(eps_lo, eps_hi, Nx):
    e = np.full(Nx, complex(eps_lo))
    e[Nx // 3: 2 * Nx // 3] = complex(eps_hi)
    return e


def _ky_fwd(lam_e, qz2, k0):
    """``_ky_forward`` on either tree: the PRE tree (f2d331c5) takes no ``k0``."""
    try:
        return E2._ky_forward(lam_e, qz2, k0)
    except TypeError:
        return E2._ky_forward(lam_e, qz2)


def _band(raw, k0):
    try:
        return float(np.real(_branch.cut_band(raw, k0=k0, xp=np)))
    except TypeError:
        return float(np.real(_branch.cut_band(raw, xp=np)))


def _cx(z):
    return complex(*z) if isinstance(z, (list, tuple)) else complex(z)


def _branch_root(z, k0):
    try:
        return _branch.forward_decaying_root(z, k0=k0, xp=np)
    except TypeError:
        return _branch.forward_decaying_root(z, xp=np)


def run_cell(key, scale, use_k0=True):
    Lx_um, lam_um, Nx, e_lo, e_hi, depth_um, Ly_um = CELLS[key]
    Lx, lam, depth, Ly = (Lx_um * scale, lam_um * scale,
                          depth_um * scale, Ly_um * scale)
    k0 = 2.0 * np.pi / lam
    eps_x = _strips(e_lo, e_hi, Nx)
    lam_e, Phi = E2.strip_x_modes(eps_x, Lx, Nx, k0)
    qz2 = 0.35 * (k0 ** 2)

    raw = np.sqrt(np.asarray(lam_e) - qz2 + 0j)
    ky = _ky_fwd(lam_e, qz2, k0 if use_k0 else None)

    # orientation census: what forward_decaying_root did to each raw root
    kept = neg = conj = other = 0
    for a, b in zip(raw, ky):
        if b == a:
            kept += 1
        elif b == -a:
            neg += 1
        elif b == np.conj(a):
            conj += 1
        else:
            other += 1

    band = _band(raw, k0 if use_k0 else None)
    top = float(np.max(np.abs(raw)))

    # mode_match T00 on a uniform-equivalent layer basis
    orders = plane_wave_orders(3, 0)
    Ny = 1
    ang = np.arange(Nx * Ny)
    Psi = np.exp(2j * np.pi * np.outer(ang, np.arange(len(orders))) / Nx)
    mm = mode_match(np.asarray(lam_e[:len(orders)]) - qz2, Psi[:, :len(orders)],
                    orders, kx0=0.0, ky0=0.0, k0=k0, eps_sup=1.0, eps_sub=1.0,
                    depth=depth, Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
    # the mode_match branch site has its OWN spectrum (qz, not ky); census it
    qz2_mm = np.asarray(lam_e[:len(orders)]) - qz2
    qz_raw = np.sqrt(qz2_mm + 0j)
    qz_out = _branch_root(qz_raw, k0 if use_k0 else None)
    mk = mn = mc = mo = 0
    for a_, b_ in zip(qz_raw, qz_out):
        if b_ == a_:
            mk += 1
        elif b_ == -a_:
            mn += 1
        elif b_ == np.conj(a_):
            mc += 1
        else:
            mo += 1
    i0 = [i for i, o in enumerate(orders) if o == (0, 0)][0]
    T00 = float(mm["T"][i0])
    R00 = float(mm["R"][i0])

    # vector strip split (its own cut_band site)
    vm = EV.strip_vector_modes(eps_x, Lx, Nx, k0, 0.0, qz2)
    kyv = np.asarray(vm[0] if isinstance(vm, tuple) else vm["ky"])

    return dict(
        scale=scale, k0=k0, band=band, band_unitfree=band * scale,
        top_absky=top, top_over_k0=top / k0,
        ky_sorted_unitfree=sorted((np.asarray(ky) * scale).tolist(),
                                  key=lambda z: (z.real, z.imag)),
        census=dict(kept=kept, negated=neg, conj=conj, other=other),
        mm_census=dict(kept=mk, negated=mn, conj=mc, other=mo),
        mm_band_unitfree=_band(qz_raw, k0 if use_k0 else None) * scale,
        mm_qz_sorted_unitfree=sorted((np.asarray(qz_out) * scale).tolist(),
                                     key=lambda z: (z.real, z.imag)),
        n_modes=int(len(ky)), T00=T00, R00=R00,
        energy=float(mm["energy"]),
        n_vec_modes=int(kyv.size),
        vec_ky_sorted_unitfree=sorted((kyv * scale).tolist(),
                                      key=lambda z: (z.real, z.imag)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    ap.add_argument("--no-k0", action="store_true",
                    help="call the band without k0 (the public default path)")
    a = ap.parse_args()
    out = dict(provenance=_prov(a.tag), use_k0=not a.no_k0, cells={})
    for key in CELLS:
        arms = {u: run_cell(key, s, use_k0=not a.no_k0)
                for u, s in UNITS.items()}
        base = arms["um"]
        cmp = {}
        for u in ("nm", "m"):
            A = np.array([_cx(z) for z in base["ky_sorted_unitfree"]])
            B = np.array([_cx(z) for z in arms[u]["ky_sorted_unitfree"]])
            n = min(A.size, B.size)
            d = np.abs(A[:n] - B[:n])
            tolr = 1e-9 * max(np.max(np.abs(A)), 1.0)
            cmp[u] = dict(
                n_roots=int(n),
                n_differing=int(np.sum(d > tolr)),
                worst_abs=float(np.max(d)) if n else 0.0,
                dT00=abs(arms[u]["T00"] - base["T00"]),
                dband_unitfree=abs(arms[u]["band_unitfree"]
                                   - base["band_unitfree"]),
                census_same=arms[u]["census"] == base["census"],
                mm_census_same=arms[u]["mm_census"] == base["mm_census"],
                mm_census=arms[u]["mm_census"],
                mm_dband_unitfree=abs(arms[u]["mm_band_unitfree"]
                                      - base["mm_band_unitfree"]),
                mm_n_differing=int(np.sum(np.abs(
                    np.array([_cx(z) for z
                              in base["mm_qz_sorted_unitfree"]])
                    - np.array([_cx(z) for z
                                in arms[u]["mm_qz_sorted_unitfree"]]))
                    > 1e-9 * max(1e-300, max(abs(_cx(z)) for z
                                             in base["mm_qz_sorted_unitfree"])))),
                nmodes_same=arms[u]["n_modes"] == base["n_modes"],
            )
        out["cells"][key] = dict(arms=arms, compare=cmp)
        print("%-16s top|ky| um=%.5g nm=%.5g m=%.5g   (literal-1.0 floor "
              "engages where this is < 1)  mm_census_um=%s"
              % (key, base["top_absky"], arms["nm"]["top_absky"],
                 arms["m"]["top_absky"], base["mm_census"]))
        for u in ("nm", "m"):
            c = cmp[u]
            print("   %s: ky differing %d/%d worst %.4g | mm qz differing %d "
                  "census %s (%s) | dT00 %.4g | census_same %s"
                  % (u, c["n_differing"], c["n_roots"], c["worst_abs"],
                     c["mm_n_differing"], c["mm_census_same"], c["mm_census"],
                     c["dT00"], c["census_same"]), flush=True)
    suffix = "_nok0" if a.no_k0 else ""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "v2_eme_units%s_%s.json" % (suffix, a.tag))
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, default=lambda o: [o.real, o.imag]
                  if isinstance(o, complex) else float(o))
    print("wrote", p)


if __name__ == "__main__":
    main()
