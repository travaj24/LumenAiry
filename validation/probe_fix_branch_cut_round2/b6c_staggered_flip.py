"""B6c -- the STAGGERED pencil's own band, on the quantity its selector reads.

CORRECTION TO A NATURAL MISTAKE.  ``b6_band_scale.py`` scores every population
with ``_sqrt_decay``'s discriminating ratio, ``|Re r| / max(max|r|, 1)`` on
``r = sqrt(lam^2)``.  For the staggered pencil that is the WRONG quantity,
because the two solvers use OPPOSITE conventions:

  * ``_sqrt_decay`` takes ``lam^2`` and a PROPAGATING mode has it real NEGATIVE
    (on the principal square root's cut), so the discriminator is the ROOT's
    REAL part;
  * ``_forward_branch_flip`` takes ``q = kz/k0 = sqrt(gamma^2)`` and a
    PROPAGATING mode has ``q`` real, i.e. ``gamma^2`` real POSITIVE, so its
    discriminator is ``|Im(q)|`` and its rule is
    ``flip <=> Im(q) < -tol  or  (|Im(q)| <= tol and Re(q) < 0)``.

The staggered engine never calls ``_sqrt_decay`` at all (measured: 0 calls on
every staggered surface), so its spectrum does not constrain that function's
band.  What it DOES constrain is ``_forward_branch_flip``'s own ``1e-8``, which
round 1 adopted as the shared convention -- so it is worth re-deriving here on
the quantity that selector actually thresholds.

Two-sided classification, made on the EIGENVALUE and never on the ratio:

  NOISE side  (the band must reach it):  ``gamma^2`` real POSITIVE to within the
              eigensolver's backward error -- a lossless PROPAGATING mode whose
              small imaginary part is rounding, and whose direction must then be
              decided by ``Re(q)``;
  SIGNAL side (the band must not reach it): everything else -- an evanescent or
              lossy mode, where ``Im(q)`` is physics.

Usage: OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b6c_staggered_flip.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
HOST = 2.25


#: How real ``gamma^2`` must be for its imaginary part to count as ROUNDING.
#: Reported at BOTH values because the classification is the only judgement in
#: this probe and its sensitivity has to be visible: ``1e-6`` is the convention
#: the ``_sqrt_decay`` census uses, but on THIS convention a relative ``1e-6``
#: on ``gamma^2`` admits modes with ``|Im q| / |q|`` up to 5e-7 -- six decades
#: above the eigensolver's own backward error, so it is an UPPER BOUND on the
#: noise side rather than the noise side.  ``1e-12`` is the defensible one.
_ROUND = (1e-6, 1e-12)


def score_q(q):
    """The ratio ``_forward_branch_flip`` thresholds, per mode, with its class."""
    q = np.asarray(q, dtype=complex).ravel()
    if not q.size:
        return []
    top = float(np.max(np.abs(q)))
    scale = max(top, 1.0)
    g2 = q ** 2
    out = []
    for i in range(q.size):
        qi, gi = q[i], g2[i]
        row = dict(ratio=float(abs(qi.imag)) / scale,
                   own=float(abs(qi.imag)) / max(abs(qi), 1e-300),
                   abs_q=float(abs(qi)),
                   rel_abs_q=float(abs(qi)) / max(top, 1e-300))
        for t in _ROUND:
            row["noise_%g" % t] = bool(gi.real > 0
                                       and abs(gi.imag) <= t * abs(gi.real))
        row["noise"] = row["noise_1e-12"]
        out.append(row)
    return out


def gaps(rows, key="noise"):
    noise = [r["ratio"] for r in rows if r[key]]
    sig = [r["ratio"] for r in rows if not r[key]]
    hi = max(noise) if noise else None
    lo = min(sig) if sig else None
    return dict(n=len(rows), n_noise=len(noise), n_signal=len(sig),
                noise_max=hi, signal_min=lo,
                gap_decades=(float(np.log10(lo / hi))
                             if (hi and lo and hi > 0) else None))


def collect():
    import lumenairy.elements.pmm._core as pc
    import lumenairy.elements.pmm.twod_staggered as ts
    from lumenairy.elements.pmm import (
        PMM2DStackPure,
        pmm_efficiency_1d,
        pmm_efficiency_2d_staggered,
        pmm_jones_1d,
        pmm_jones_2d_staggered,
    )
    seen = []
    orig = pc._forward_branch_flip

    def spy(q, xp=np):
        try:
            seen.append(np.asarray(q).astype(complex).copy())
        except Exception:                                 # pragma: no cover
            pass
        return orig(q, xp)

    saved = [(m, m._forward_branch_flip) for m in (pc, ts)
             if hasattr(m, "_forward_branch_flip")]
    for m, _ in saved:
        m._forward_branch_flip = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # degree 5/6 only: the degree-8 staggered pencil is a
            # 1568 x 1568 QZ per solve and adds nothing to a POPULATION census
            # (the ratio is a property of the eigenproblem, and degree only
            # changes how many modes it has).
            for pillar in (HOST * (1 + 1e-6), 6.0, -10.0 + 1.0j):
                for nsub in (1.5, 1.63):
                    for deg in (5, 6):
                        try:
                            pmm_efficiency_2d_staggered(
                                PX, PX,
                                F.stag_cell(host=HOST, pillar=pillar),
                                nsub, 1.0, D, WL, degree=deg, n_orders=4)
                        except Exception:
                            pass
            for ei in (0.0, 1e-2, 1e-6, 1e-10):
                try:
                    pmm_jones_2d_staggered(
                        PX, PX, F.stag_tensor_cell(eps_im=ei), 1.5, 1.0, D,
                        WL, degree=5, n_orders=3)
                except Exception:
                    pass
            for theta in (0.0, 0.25, 0.6):
                try:
                    pmm_efficiency_2d_staggered(
                        PX, PX, F.stag_cell(host=HOST, pillar=6.0), 1.5, 1.0,
                        D, WL, degree=6, n_orders=4, theta=theta)
                except Exception:
                    pass
            try:
                st = PMM2DStackPure(PX, PX, n_substrate=1.5,
                                    n_superstrate=1.0, degree=6, n_orders=4)
                st.add_layer(0.1e-6, eps=HOST)
                st.add_layer(D, eps_cell=F.stag_cell(host=HOST, pillar=6.0))
                st.add_layer(0.1e-6, eps=HOST)
                st.set_source(WL, theta=0.0).solve()
            except Exception:
                pass
            # the 1-D PMM shares the selector
            for nr in (1.45, 2.1 ** 0.5, 3.5):
                for pol in ("te", "tm"):
                    try:
                        pmm_efficiency_1d(1.0e-6, nr, 1.5, 1.5, 1.0, 0.4e-6,
                                          0.5, WL, degree=12,
                                          polarization=pol)
                    except Exception:
                        pass
            for ei in (0.05, 1e-4, 1e-8):
                try:
                    pmm_efficiency_1d(1.0e-6, (2.1 + 1j * ei) ** 0.5, 1.5,
                                      1.5, 1.0, 0.4e-6, 0.5, WL, degree=12,
                                      polarization="tm")
                except Exception:
                    pass
            try:
                pmm_jones_1d(1.0e-6, 2.1 ** 0.5, 1.0, 1.5, 1.0, 0.4e-6, 0.5,
                             WL, degree=12, theta=0.25, phi=0.4)
            except Exception:
                pass
    finally:
        for m, fn in saved:
            m._forward_branch_flip = fn
    rows = []
    for q in seen:
        rows.extend(score_q(q))
    return rows, len(seen)


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b6c.json"
    rows, n_arrays = collect()
    allg = {("noise_%g" % t): gaps(rows, "noise_%g" % t) for t in _ROUND}
    g = allg["noise_1e-12"]
    worst_own = max([r["own"] for r in rows if r["ratio"] <= 1e-8] or [0.0])
    print("staggered / 1-D PMM _forward_branch_flip population: %d arrays, "
          "%d modes" % (n_arrays, len(rows)))
    for k, gg in sorted(allg.items()):
        print("  [%s] noise n=%-6d max %-12s | signal n=%-6d min %-12s | "
              "gap %s dec"
              % (k, gg["n_noise"],
                 "%.4e" % gg["noise_max"] if gg["noise_max"] else "-",
                 gg["n_signal"],
                 "%.4e" % gg["signal_min"] if gg["signal_min"] else "-",
                 "%.2f" % gg["gap_decades"] if gg["gap_decades"] else "-"))
    print("  noise side (propagating, sign is rounding): n=%d max ratio %s"
          % (g["n_noise"],
             "%.4e" % g["noise_max"] if g["noise_max"] else "-"))
    print("  signal side (evanescent / lossy):           n=%d min ratio %s"
          % (g["n_signal"],
             "%.4e" % g["signal_min"] if g["signal_min"] else "-"))
    print("  two-sided gap: %s decades; bar 1e-8 sits %s above the noise and "
          "%s below the signal"
          % ("%.2f" % g["gap_decades"] if g["gap_decades"] else "-",
             "%.2f" % np.log10(1e-8 / g["noise_max"]) if g["noise_max"]
             else "-",
             "%.2f" % np.log10(g["signal_min"] / 1e-8) if g["signal_min"]
             else "-"))
    print("  worst |Im q|/|q| inside the band: %.4e" % worst_own)
    F.dump(out, dict(summary=g, by_threshold=allg, n_arrays=n_arrays,
                     worst_own_ratio_in_band=worst_own,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
