"""E-EME-FLIP -- does ``eme_2d._ky_forward``'s EXACT-ZERO pin change the ANSWER?

The pin (``eme_2d.py:129``, ``np.where(ky.imag < 0.0, -ky, ky)``) decides the
FORWARD lateral root of a PROPAGATING strip mode on the SIGN of an imaginary
part that, for such a mode, IS the eigensolver's backward error.

The wrong-answer test is A/B on IDENTICAL PHYSICS:

  arm A : ``eps`` REAL           -> ``strip_x_modes`` takes ``eigh``,
                                    ``Im(lam)`` EXACTLY 0, no mode flips;
  arm B : ``eps + i 1e-30``      -> any nonzero ``Im(eps)`` routes to
          on ONE region only        ``scipy.linalg.eig``; the physical
                                    ``Im(lam) ~ 4e-29`` is 13 decades BELOW the
                                    backward error and the pin is decided by
                                    roundoff.

The arms are the same optics to thirty decimal places.  Any difference in the
returned mode set is a DISCONTINUITY at ``Im(eps) = 0+`` -- the shape the
library already calls a defect on the RCWA side
(``_require_propagating_incidence``, audit P1 2026-06-10).

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=n MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_eme_flip.py out.json
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

LX = LY = 1.0
KY0 = 0.37                     # NONZERO: a flipped mode then roots elsewhere


def flip_scan():
    from lumenairy.elements.eme import eme_2d as M
    rows = []
    for nx in (24, 32, 48, 64, 96, 128):
        for kf in (20, 40, 60, 80):
            k0 = kf * np.pi
            e = np.full(nx, 2.25 + 0j)
            e[nx // 2:] = 12.0 + 1j * 1e-30
            lam = np.asarray(M.strip_x_modes(e, LX, nx, k0)[0], complex)
            raw = np.sqrt(lam + 0j)
            ky = M._ky_forward(lam, 0.0)
            flip = ~np.isclose(raw, ky, rtol=0, atol=0)
            oncut = np.abs(raw.real) > 1e3 * np.abs(raw.imag)
            rel = np.abs(raw.imag) / np.maximum(np.abs(raw), 1e-300)
            sel = flip & oncut
            rows.append(dict(
                nx=nx, k0_over_pi=kf, n_modes=int(lam.size),
                n_propagating=int(oncut.sum()),
                n_flipped_propagating=int(sel.sum()),
                worst_rel_Im_flipped=(float(np.max(rel[sel])) if sel.any()
                                      else None),
                physical_rel_Im=float(1e-30 * k0 ** 2
                                      / max(float(np.max(np.abs(raw))), 1e-300))))
    return rows


def ab_modes(nx, kf, window, n_scan=300):
    from lumenairy.elements.eme import eme_2d as M
    k0 = kf * np.pi
    out = {}
    for arm, im in (("A_real_eigh", 0.0), ("B_1e-30_eig", 1e-30)):
        e1 = np.full(nx, 2.25 + 0j)
        e1[nx // 2:] = 12.0 + 1j * im
        e2 = np.full(nx, 2.25 + 0j)
        e2[nx // 4:3 * nx // 4] = 12.0 + 1j * im
        lam = np.asarray(M.strip_x_modes(e1, LX, nx, k0)[0], complex)
        raw = np.sqrt(lam + 0j)
        ky = M._ky_forward(lam, 0.0)
        nflip = int(np.sum((~np.isclose(raw, ky, rtol=0, atol=0))
                           & (np.abs(raw.real) > 1e3 * np.abs(raw.imag))))
        t = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                q = np.sort(np.asarray(
                    M.layer_modes([(e1, 0.5 * LY), (e2, 0.5 * LY)], LX, nx, LY,
                                  k0, window, ky0=KY0, n_scan=n_scan),
                    float).ravel())
                err = None
            except Exception as exc:
                q, err = np.array([]), repr(exc)[:180]
        out[arm] = dict(n_flipped_propagating=nflip, raised=err,
                        secs=round(time.time() - t, 2), n_modes=int(q.size),
                        modes=[float(v) for v in q])
    a = np.asarray(out["A_real_eigh"]["modes"])
    b = np.asarray(out["B_1e-30_eig"]["modes"])
    if a.size and b.size:
        n = min(a.size, b.size)
        out["same_mode_count"] = bool(a.size == b.size)
        out["max_abs_gap"] = float(np.max(np.abs(a[:n] - b[:n])))
        out["max_rel_gap"] = float(np.max(np.abs(a[:n] - b[:n])
                                          / np.maximum(np.abs(a[:n]), 1.0)))
    out["nx"], out["k0_over_pi"], out["window"] = nx, kf, list(window)
    return out


def main():
    E.pin_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "e_eme_flip.json"
    res = {"flip_scan": flip_scan()}
    cand = [r for r in res["flip_scan"] if r["n_flipped_propagating"] > 0]
    cand.sort(key=lambda r: (r["nx"], r["k0_over_pi"]))
    res["chosen"] = cand[0] if cand else None
    if cand:
        nx, kf = cand[0]["nx"], cand[0]["k0_over_pi"]
        hi = 12.0 * (kf * np.pi) ** 2
        res["ab"] = ab_modes(nx, kf, (0.55 * hi, 0.75 * hi))
    E.dump(out, res)


if __name__ == "__main__":
    main()
