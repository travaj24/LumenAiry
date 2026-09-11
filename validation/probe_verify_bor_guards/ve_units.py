"""TASK E / claim (a) UNIT-SYSTEM check -- is ``ky`` DIMENSIONLESS?

``_branch.cut_band`` floors its band at a literal ``1.0`` and the docstring
justifies that with "``ky`` here is DIMENSIONLESS (the EME modules work in
``k0``-normalized units)".  If that is false the floor is unit-dependent, which
is the same class of defect the BOR peer fixed by flooring at ``k0``.

WHAT THE CODE ACTUALLY COMPUTES.  ``eme_2d.strip_x_modes`` assembles
``A = d2/dx2 + eps(x) k0^2`` on a grid of spacing ``h = Lx / Nx``, so ``lam``
carries units of ``1/length^2`` and ``ky = sqrt(lam - qz2)`` carries
``1/length``.  ``eme_diffraction.mode_match`` then forms ``exp(i qz * depth)``,
which is dimensionless only if ``qz`` is ``1/length``.  Neither is
dimensionless, and ``k0`` is a free argument with units, not a normalization.

THE TEST.  Re-express the SAME physics in a unit system whose length unit is
``s`` times smaller: ``Lx -> s Lx``, ``k0 -> k0 / s``, ``qz2 -> qz2 / s^2``,
``depth -> s depth``.  Every ``ky`` must then come back as ``ky / s`` and every
BRANCH DECISION must be identical.  A decision that moves is a unit bug.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _vh  # noqa: E402
import ve_fix  # noqa: E402


def decision(z0, got):
    """0 same, 1 negated, 2 conjugated, 3 other -- scale-free labels."""
    out = np.full(z0.size, 3, dtype=np.int8)
    out[np.abs(got - z0) == 0] = 0
    out[(np.abs(got + z0) <= 1e-13 * np.abs(z0)) & (out == 3)] = 1
    out[(np.abs(got - np.conj(z0)) <= 1e-13 * np.abs(z0)) & (out == 3)] = 2
    return out


def synthetic(eme_2d, out):
    """ONE root, TWO unit systems, one physical problem.

    System A: spectrum top ``M``, probe root ``a - i d``.
    System B: the identical physics with the length unit ``s`` times smaller,
    so every wavenumber is ``1/s`` times as large.
    """
    rows = []
    for M, a, d in ((200.0, 100.0, 5.0e-7), (200.0, 100.0, 1.0e-7),
                    (200.0, 100.0, 3.0e-6), (2000.0, 900.0, 3.0e-6)):
        for s in (1.0, 1e3, 1e-3):
            z = np.array([complex(M / s, 0.0), complex(a / s, -d / s)])
            got = np.asarray(eme_2d._ky_forward(z ** 2, 0.0))
            z0 = np.sqrt(z ** 2 + 0j)
            dec = decision(z0, got)
            rows.append(dict(M=M, a=a, d=d, s=s,
                             band=1e-9 * max(1.0, M / s),
                             band_in_A_units=1e-9 * max(1.0, M / s) * s,
                             dec=int(dec[1]),
                             dec_name={0: "same", 1: "NEG", 2: "conj",
                                       3: "other"}[int(dec[1])],
                             ky_scaled_re=float(got[1].real * s),
                             ky_scaled_im=float(got[1].imag * s)))
    out["synthetic_units"] = rows


def real_solve(eme_2d, out):
    """A real strip solve in three unit systems.  ``ky_s * s`` must reproduce
    ``ky_1`` and the decision vector must be identical."""
    rows = []
    Nx = 96
    for imag in (0.0, 1e-30, 1e-12, -1e-7, -1e-6, -1e-5, 1e-3):
        ref = None
        for s in (1.0, 1e3, 1e-3):
            Lx, k0 = 1.0 * s, 20.0 * np.pi / s
            eps = ve_fix.eps_split(Nx, imag=imag)
            lam = np.asarray(eme_2d.strip_x_modes(eps, Lx, Nx, k0, 0.0)[0],
                             dtype=complex)
            lam = lam[np.lexsort((lam.imag, lam.real))]
            z0 = np.sqrt(lam + 0j)
            got = np.asarray(eme_2d._ky_forward(lam, 0.0))
            dec = decision(z0, got)
            scaled = got * s
            if s == 1.0:
                ref = scaled
                ref_dec = dec
            row = dict(imag=imag, s=s, n=int(got.size),
                       band=1e-9 * max(1.0, float(np.max(np.abs(z0)))),
                       max_absz=float(np.max(np.abs(z0))),
                       n_same=int((dec == 0).sum()), n_neg=int((dec == 1).sum()),
                       n_conj=int((dec == 2).sum()),
                       n_dec_differs=int(np.sum(dec != ref_dec)),
                       worst_dky_vs_s1=float(np.max(np.abs(scaled - ref))),
                       worst_dky_rel=float(np.max(
                           np.abs(scaled - ref)
                           / np.maximum(np.abs(ref), 1e-300))),
                       n_root_differs=int(np.sum(
                           np.abs(scaled - ref)
                           > 1e-6 * np.maximum(np.abs(ref), 1.0))))
            rows.append(row)
    out["real_solve_units"] = rows


def mode_match_units(ed, out):
    """``mode_match``: ``qz`` multiplies ``depth``, so it is ``1/length``.  The
    same slab in three unit systems must give the SAME r/t."""
    rows = []
    orders = ed.plane_wave_orders(1, 1)
    Nx = Ny = 8
    for imqz2 in (0.0, -1e-12, -1e-9, -1e-7, -1e-5, -1e-3):
        ref = None
        for s in (1.0, 1e3, 1e-3):
            Lx = Ly = 1.0 * s
            k0 = 8.0 / s
            depth = 2.0 * s
            U, kx, ky = ed.pw_matrix(orders, 0.0, 0.0, Lx, Ly, Nx, Ny)
            qz2 = (2.25 * k0 ** 2 - kx ** 2 - ky ** 2).astype(complex)
            qz2 = qz2 + 1j * imqz2 / s ** 2
            res = ed.mode_match(qz2, U.copy(), orders, kx0=0.0, ky0=0.0,
                                k0=k0, eps_sup=1.0, eps_sub=1.0, depth=depth,
                                Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny)
            r = np.asarray(res["r"])
            z0 = np.sqrt(qz2 + 0j)
            if s == 1.0:
                ref = r
            rows.append(dict(im_qz2=imqz2, s=s,
                             band=1e-9 * max(1.0, float(np.max(np.abs(z0)))),
                             max_absqz=float(np.max(np.abs(z0))),
                             n_negIm=int(np.sum(z0.imag < 0)),
                             energy=float(res["energy"]),
                             T00=float(res["T"][orders.index((0, 0))]),
                             worst_dr=float(np.max(np.abs(r - ref)))))
    out["mode_match_units"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=["pre", "post"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    import lumenairy
    print("lumenairy.__file__ =", lumenairy.__file__)
    _vh.require_tree(a.build)
    from lumenairy.elements.eme import eme_2d, eme_diffraction

    out = dict(task="E", claim="3-units", build=a.build, tag=a.tag,
               arm=_vh.arm(), lumenairy_file=lumenairy.__file__)
    synthetic(eme_2d, out)
    real_solve(eme_2d, out)
    mode_match_units(eme_diffraction, out)
    _vh.dump(HERE / ("ve_units_%s.json" % a.tag), out)

    print("\nSYNTHETIC: one physical root, three unit systems")
    print("  %-8s %-8s %-10s %-6s %-12s %-12s %-6s %s"
          % ("M", "a", "d", "s", "band(s)", "band(A units)", "dec", "ky*s"))
    for r in out["synthetic_units"]:
        print("  %-8g %-8g %-10.3g %-6g %-12.4g %-12.4g %-6s %+.6g%+.3gj"
              % (r["M"], r["a"], r["d"], r["s"], r["band"],
                 r["band_in_A_units"], r["dec_name"], r["ky_scaled_re"],
                 r["ky_scaled_im"]))
    print("\nREAL SOLVE: same strip, three unit systems")
    print("  %-9s %-6s %-10s %-10s %-5s %-5s %-5s %-8s %-12s %s"
          % ("Im(eps)", "s", "max|z|", "band", "same", "neg", "conj",
             "dec_dif", "worst d(ky*s)", "root_dif"))
    for r in out["real_solve_units"]:
        print("  %-9g %-6g %-10.4g %-10.3e %-5d %-5d %-5d %-8d %-12.4e %d"
              % (r["imag"], r["s"], r["max_absz"], r["band"], r["n_same"],
                 r["n_neg"], r["n_conj"], r["n_dec_differs"],
                 r["worst_dky_vs_s1"], r["n_root_differs"]))
    print("\nmode_match: same slab, three unit systems")
    for r in out["mode_match_units"]:
        print("  Im(qz2)=%-10g s=%-6g band=%-10.3e max|qz|=%-9.4g negIm=%d "
              "T00=%.12f worst|dr|=%.3e"
              % (r["im_qz2"], r["s"], r["band"], r["max_absqz"], r["n_negIm"],
                 r["T00"], r["worst_dr"]))


if __name__ == "__main__":
    main()
