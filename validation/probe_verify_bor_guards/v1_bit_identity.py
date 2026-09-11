"""TASK A -- bit-identity of MY OWN fixture battery across 352173e -> c1189e3f.

>= 40 BOR fixtures (both bases, m = 0/1/2/5, lossless and lossy, ring
gratings, segment layers, the two uniform coincidences, thin rings, tapers at
4..64 slices, hp-refined/graded meshes, an anisotropic layer, an nm-unit
system) and >= 10 EME fixtures, hashed to the SHA-256 of the exact IEEE-754
bytes of the answer.

Usage:  python v1_bit_identity.py <pre|post> [outdir]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402


def _eps_strip(Nx, e_hi, e_lo, frac=0.5):
    x = (np.arange(Nx) + 0.5) / Nx
    return np.where(x >= frac, e_hi, e_lo).astype(complex)


def _strips(Nx, imag):
    hi = 12.0 + 1j * imag
    e1 = np.full(Nx, 2.25 + 0j); e1[Nx // 2:] = hi
    e2 = np.full(Nx, 2.25 + 0j); e2[Nx // 4:3 * Nx // 4] = hi
    return [(e1, 0.5), (e2, 0.5)]


def eme_fixtures():
    """>= 10 EME fixtures: strip root sets at three resolutions x two k0, the
    infinitesimal-loss discontinuity arms, two layer_modes counts, and two
    diffraction mode-match answers."""
    from lumenairy.elements.eme import eme_2d, eme_diffraction
    F = []

    def strip(Nx, kf, imag, name):
        def fn(Nx=Nx, kf=kf, imag=imag):
            e = _eps_strip(Nx, 12.0 + 1j * imag, 2.25 + 0j)
            lam = np.asarray(eme_2d.strip_x_modes(e, 1.0, Nx, kf * np.pi)[0],
                             complex)
            ky = np.asarray(eme_2d._ky_forward(lam, 0.37 ** 2))
            raw = np.sqrt(lam - 0.37 ** 2 + 0j)
            flip = ~np.isclose(raw, ky, rtol=0, atol=0)
            oncut = np.abs(raw.real) > 1e3 * np.abs(raw.imag)
            return dict(hash=_vh.hash_arrays(ky), n=int(ky.size),
                        n_flipped=int(flip.sum()),
                        n_flipped_propagating=int((flip & oncut).sum()),
                        n_neg_imag=int(np.sum(ky.imag < 0.0)),
                        max_abs=float(np.max(np.abs(ky))))
        F.append((name, fn))

    for Nx in (48, 96, 128):
        for kf in (20, 40):
            strip(Nx, kf, 0.0, "strip_Nx%d_k%dpi" % (Nx, kf))
    for imag in (1e-30, 1e-12, 1e-3):
        strip(96, 20, imag, "striplossy_im%g" % imag)

    def lm(Nx, n_scan, w, imag, name):
        def fn(Nx=Nx, n_scan=n_scan, w=w, imag=imag):
            q2 = np.asarray(eme_2d.layer_modes(
                _strips(Nx, imag), 1.0, Nx, 1.0, 20.0 * np.pi, w,
                kx0=0.0, ky0=0.37, n_scan=n_scan))
            return dict(n=int(q2.size), hash=_vh.hash_arrays(q2))
        F.append((name, fn))

    lm(96, 60, (26055.8, 28500.0), 0.0, "layermodes_96_real")
    lm(96, 60, (26055.8, 28500.0), 1e-30, "layermodes_96_im1e-30")
    lm(64, 60, (26055.8, 28500.0), 0.0, "layermodes_64_real")

    def diff(name, k0, eps_l, kx0, ky0, Nx):
        def fn(k0=k0, eps_l=eps_l, kx0=kx0, ky0=ky0, Nx=Nx):
            eps_xy = np.full((Nx, Nx), eps_l, dtype=complex)
            res = eme_diffraction.diffraction_fd(
                eps_xy, 2.0, 2.0, Nx, Nx, k0, 1.0, 2.25, 0.4, 2, 2,
                kx0=kx0, ky0=ky0)
            R = np.asarray(res["R"]); T = np.asarray(res["T"])
            return dict(hash=_vh.hash_arrays(R, T), n=int(R.size),
                        energy=float(res["energy"]))
        F.append((name, fn))

    diff("diffr_uniform_normal", 5.0, 4.0, 0.0, 0.0, 24)
    diff("diffr_uniform_oblique", 5.0, 4.0, 0.0, 2.0, 24)
    diff("diffr_lossy", 5.0, 4.0 + 0.02j, 0.0, 2.0, 24)

    return F


def main():
    build = sys.argv[1]
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    rows = {}
    for name, fn in _vh.bor_fixtures():
        rows[name] = _vh.solve_fixture(fn)
        r = rows[name]
        print("  BOR %-28s %s" % (name, r.get("raised") or r["hash"][:16]),
              flush=True)
    erows = {}
    for name, fn in eme_fixtures():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                erows[name] = fn()
                erows[name]["raised"] = None
            except BaseException as e:      # noqa: BLE001
                erows[name] = dict(raised=type(e).__name__, msg=str(e)[:300])
            erows[name]["warnings"] = [str(x.message)[:160] for x in w]
        print("  EME %-28s %s" % (name, erows[name].get("raised")
                                  or erows[name].get("hash", "?")[:16]),
              flush=True)
    out = sys.argv[2] if len(sys.argv) > 2 else "."
    _vh.dump("%s/v1_bit_identity_%s_%s_t%s.json"
             % (out, build, a["platform"], a["blas_threads"]),
             dict(arm=a, build=build, bor=rows, eme=erows))


if __name__ == "__main__":
    main()
