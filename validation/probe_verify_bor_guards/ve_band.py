"""TASK E / band probe -- the 1e-9 cut band, from BOTH sides.

1. THE DECISION AT THE EDGE.  Sweep a root's ``Im`` across the band and show
   what the selector returns just inside and just outside.  On the cut the
   answer must be the CONJUGATE (a move of ``2|Im z|``); off it the NEGATION (a
   move of ``2|z|``).  The jump between them at the edge is the band's
   discontinuity, and its size is what the margins have to keep populations
   away from.
2. THE TWO POPULATIONS.  For real strip solves, the largest ``|Im z| / scale``
   among PROPAGATING modes (the backward error the band must absorb) and the
   smallest among genuinely LOSSY / GAIN modes (the physics it must not),
   in decades either side of the band.
3. THE CASCADE GUARANTEE.  ``|exp(i ky h)| <= 1`` for every forward mode of
   every fixture -- for the scalar selector AND for the vector solver's own
   forward SET, which selects indices and does not move the root.
4. THE BAND EDGE OF THE LAYER (``ky^2 -> 0``), where a root's own magnitude
   collapses while the band, being relative to the spectrum's TOP, does not.
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


def sweep(eme_2d, out):
    """A spectrum whose top is ``M`` plus ONE probe root ``a - i d``; d sweeps
    across the band.  Driven through the PUBLIC path (``_ky_forward``), so PRE
    and POST are asked the same question."""
    rows = []
    for M, a in ((200.0, 100.0), (2.0e5, 1.0e5), (0.2, 0.1), (1.0, 0.5)):
        tol = 1e-9 * max(1.0, M)
        for mult in (1e-3, 1e-1, 0.5, 0.9, 0.999, 1.0, 1.001, 1.1, 10.0,
                     1e3, 1e6):
            d = tol * mult
            z = np.array([complex(M, 0.0), complex(a, -d)])
            lam = z ** 2                      # _ky_forward takes ky^2
            got = np.asarray(eme_2d._ky_forward(lam, 0.0))
            z0 = np.sqrt(lam + 0j)
            probe, p0 = got[1], z0[1]
            if abs(probe - p0) <= 0:
                dec = "same"
            elif abs(probe + p0) <= 1e-13 * abs(p0):
                dec = "NEG"
            elif abs(probe - np.conj(p0)) <= 1e-13 * abs(p0):
                dec = "conj"
            else:
                dec = "other"
            rows.append(dict(M=M, a=a, tol=tol, mult=mult, d=d, decision=dec,
                             re_out=float(probe.real),
                             im_out=float(probe.imag),
                             move=float(abs(probe - p0)),
                             move_over_absz=float(abs(probe - p0) / abs(p0))))
    out["edge_sweep"] = rows


def band_edge(eme_2d, out):
    """``qz2`` placed ON a strip eigenvalue, so that mode's ``ky^2 -> 0``.  Its
    own |z| collapses to ~sqrt(backward error) while the band stays tied to the
    spectrum's TOP, so the band is then astronomically wide RELATIVE to that
    mode -- and the backward error on ``Im(ky)`` is ~sqrt of the error on
    ``ky^2``, which is astronomically large relative to the band.  Both sides
    of the inequality move; this measures which wins."""
    rows = []
    for Nx, k0, imag in ((96, 20 * np.pi, 0.0), (96, 20 * np.pi, 1e-30),
                         (96, 20 * np.pi, 1e-12), (128, 40 * np.pi, 1e-30)):
        eps = ve_fix.eps_split(Nx, imag=imag)
        lam = np.asarray(eme_2d.strip_x_modes(eps, 1.0, Nx, k0, 0.0)[0],
                         dtype=complex)
        lam = lam[np.lexsort((lam.imag, lam.real))]
        for j in (Nx - 1, Nx - 5, Nx // 2):
            qz2 = float(lam[j].real)          # exactly ON the j-th band edge
            z0 = np.sqrt(lam - qz2 + 0j)
            got = np.asarray(eme_2d._ky_forward(lam, qz2))
            scale = max(1.0, float(np.max(np.abs(z0))))
            tol = 1e-9 * scale
            k = int(np.argmin(np.abs(z0)))    # the collapsed mode
            rows.append(dict(
                Nx=Nx, k0=k0, imag=imag, j=j, qz2=qz2,
                absz_min=float(np.abs(z0[k])), tol=tol,
                im_z0=float(z0[k].imag), re_z0=float(z0[k].real),
                oncut=bool(abs(z0[k].imag) <= tol),
                out_re=float(got[k].real), out_im=float(got[k].imag),
                move=float(abs(got[k] - z0[k])),
                move_rel=float(abs(got[k] - z0[k])
                               / max(abs(z0[k]), 1e-300)),
                min_imag_all=float(np.min(got.imag)),
                growth=float(np.max(np.abs(np.exp(1j * got * 0.5))))))
    out["band_edge"] = rows


def gain_scan(eme_2d, out):
    """How much GAIN does the band absorb?  ``Im(eps) < 0`` puts a PHYSICALLY
    negative ``Im(ky)`` on a propagating mode.  Below some |Im(eps)| the band
    calls it backward error and CONJUGATES (forward-propagating, decaying);
    above it the rule NEGATES (backward-propagating, decaying).  Both are 'gain
    is out of scope', but they are different answers, and the crossover is a
    property of the band, not of the physics."""
    rows = []
    Nx, k0 = 96, 20 * np.pi
    for g in (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        eps = ve_fix.eps_split(Nx, imag=-g)
        lam = np.asarray(eme_2d.strip_x_modes(eps, 1.0, Nx, k0, 0.0)[0],
                         dtype=complex)
        lam = lam[np.lexsort((lam.imag, lam.real))]
        z0 = np.sqrt(lam + 0j)
        got = np.asarray(eme_2d._ky_forward(lam, 0.0))
        scale = max(1.0, float(np.max(np.abs(z0))))
        neg = np.isclose(got, -z0, rtol=1e-13, atol=0)
        con = np.isclose(got, np.conj(z0), rtol=1e-13, atol=0) & ~neg
        prop = (z0.real > 0) & (np.abs(z0.imag) < 1e-6 * np.abs(z0.real))
        rows.append(dict(gain=g, n=int(lam.size), n_neg=int(neg.sum()),
                         n_conj=int(con.sum()),
                         n_conj_prop=int((con & prop).sum()),
                         band=1e-9 * scale,
                         min_ratio=float(np.min(np.abs(z0.imag)) / scale),
                         max_ratio=float(np.max(np.abs(z0.imag)) / scale)))
    out["gain_scan"] = rows


def vector_growth(ev, out):
    """The vector solver's forward SET: modes inside the band keep their root
    unchanged, so a mode with a slightly NEGATIVE Im inside the band is a
    forward mode whose propagator GROWS -- by exp(tol*h).  Measured."""
    rows = []
    for Nx, k0, imag in ((24, 8.0, 0.0), (24, 8.0, 1e-30), (24, 8.0, 1e-3),
                         (48, 20 * np.pi, 1e-30), (24, 8.0, -1e-6)):
        eps = ve_fix.eps_split(Nx, lo=2.25, hi=6.25, imag=imag)
        ky, W, V = ev.strip_vector_modes(eps, 1.0, Nx, k0, kx0=0.0, qz2=0.0)
        ky = np.asarray(ky)
        rows.append(dict(Nx=Nx, k0=k0, imag=imag, n=int(ky.size),
                         min_imag=float(np.min(ky.imag)),
                         tol=1e-9 * max(1.0, float(np.max(np.abs(ky)))),
                         growth_h05=float(np.max(np.abs(np.exp(1j*ky*0.5)))),
                         growth_h100=float(np.max(np.abs(
                             np.exp(1j * ky * 100.0))))))
    out["vector_growth"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, choices=["pre", "post"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    import lumenairy
    print("lumenairy.__file__ =", lumenairy.__file__)
    _vh.require_tree(a.build)
    from lumenairy.elements.eme import eme_2d, eme_2d_vector

    out = dict(task="E", claim="3-band", build=a.build, tag=a.tag,
               arm=_vh.arm(), lumenairy_file=lumenairy.__file__)
    sweep(eme_2d, out)
    band_edge(eme_2d, out)
    gain_scan(eme_2d, out)
    vector_growth(eme_2d_vector, out)
    _vh.dump(HERE / ("ve_band_%s.json" % a.tag), out)
    # a readable digest on stdout
    print("\nEDGE SWEEP (decision vs Im/band)")
    for r in out["edge_sweep"]:
        print("  M=%-8g a=%-8g Im/band=%-8g -> %-5s move=%.4e (%.3e x |z|)"
              % (r["M"], r["a"], r["mult"], r["decision"], r["move"],
                 r["move_over_absz"]))
    print("\nGAIN SCAN")
    for r in out["gain_scan"]:
        print("  |Im eps|=%-8g band=%.3e neg=%3d conj=%3d conj_prop=%3d "
              "min|Im z|/scale=%.3e" % (r["gain"], r["band"], r["n_neg"],
                                        r["n_conj"], r["n_conj_prop"],
                                        r["min_ratio"]))
    print("\nBAND EDGE (ky^2 -> 0)")
    for r in out["band_edge"]:
        print("  Nx=%3d im=%-8g j=%3d |z|min=%.3e Im z=%+.3e tol=%.3e "
              "oncut=%-5s move_rel=%.3e minImOut=%+.3e growth=%.6f"
              % (r["Nx"], r["imag"], r["j"], r["absz_min"], r["im_z0"],
                 r["tol"], r["oncut"], r["move_rel"], r["min_imag_all"],
                 r["growth"]))
    print("\nVECTOR forward-set growth")
    for r in out["vector_growth"]:
        print("  Nx=%3d im=%-8g minIm=%+.4e tol=%.3e growth(h=0.5)=%.16f "
              "growth(h=100)=%.16f" % (r["Nx"], r["imag"], r["min_imag"],
                                       r["tol"], r["growth_h05"],
                                       r["growth_h100"]))


if __name__ == "__main__":
    main()
