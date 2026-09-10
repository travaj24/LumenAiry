"""B6 -- the BAND SCALE decision: ARRAY-MAX versus PER-MODE.

TERMS.  The shipped band asks whether a root ``r = sqrt(lam^2)`` is
"numerically on the branch cut" by comparing its real part against the LARGEST
root of the SAME array:

    ARRAY-MAX shape      |Re r|  <=  C * max( max|r|, 1 )

so its verdict on one mode depends on the other modes of the same layer.  The
verification's D4 measured the consequence: at a mount driven onto a LAYER
CUTOFF a mode whose real part is 0.2 % of ITS OWN magnitude
(``|Re r| / |r| = 2.0751e-03``) is conjugated, because that mode sits at
``|lam| = 1.7e-07``, i.e. 3.8e-08 of the spectrum's top.  D3 measured the other
consequence: the noise side of the population reaches 6.7172e-09 at a cutoff
mount, only 1.5x under the ``1e-8`` bar.

The alternative judges each mode in its own terms, with a FLOOR so that a mode
whose own magnitude has collapsed to the eigensolver's backward error is not
divided by noise:

    PER-MODE shape       |Re r|  <=  C * max( |r|, floor )

FLOOR, derived.  ``eig`` returns ``lam^2`` with a backward error
``|d lam^2| ~ eps_mach * ||M||``, and ``r = sqrt(lam^2)``, so the smallest
magnitude a root can carry that still means anything is
``floor = sqrt(eps_mach * ||M||)``.  ``||M||`` is not visible inside the
selector, but ``max|r|^2`` is the spectrum's own scale for ``lam^2``, so
``floor = sqrt(eps_mach) * max|r|`` is the same statement written in the
quantities the function has.  With ``eps_mach = 2.22e-16`` that is
``1.49e-08 * max|r|`` -- i.e. the per-mode shape degenerates CONTINUOUSLY into
the array-max shape for a mode more than eight decades below the spectrum's
top, and differs from it only in the band between.

WHAT THIS PROBE DOES.  It censuses the same populations for BOTH shapes and
reports, per population and per build, the two-sided gap: the largest ratio of
a mode whose negative imaginary part is ROUNDING (the noise side, which the
band must reach) and the smallest ratio of a mode whose negative imaginary part
is PHYSICS (the signal side, which the band must not reach).  The decision is
the shape with the larger gap on BOTH the RCWA and the PMM populations.

Populations censused:
  * RCWA -- the round-1 / verification fixture families: lossless anisotropic
    2-D over twist x truncation x substrate, oblique and conical, scalar 2-D,
    1-D TE/TM, a LOSS LADDER, metals, and mounts driven onto a LAYER CUTOFF by
    trisection (the corner that sets the noise side);
  * PMM HYBRID -- the ``pmm/twod.py`` layer eigenproblem, whose spectrum is
    the SEM-projected ``P@Q`` and not the RCWA one;
  * PMM STAGGERED -- the ``gamma^2`` spectrum of the staggered generalized
    pencil (``_region_modes``), which is selected by ``_forward_branch_flip``
    rather than by ``_sqrt_decay`` but carries the same question about a
    relative band's scale.

The band ratio is computed from the RAW eigenvalue output, so it is a property
of the eigenproblem and is IDENTICAL on both arms of the change.

Usage:  OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b6_band_scale.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

EPS_MACH = float(np.finfo(np.float64).eps)
SQRT_EPS = float(np.sqrt(EPS_MACH))         # 1.4901e-08

WL, PX, D = F.WL, F.PX, F.DEPTH


# ------------------------------------------------------------------ scoring
def score(lam2):
    """Both candidate ratios for every ``Im(r) < 0`` mode of one eigenvalue
    array, with the class each mode belongs to.

    A mode is NOISE-side (the band must reach it) when its ``lam^2`` is real
    NEGATIVE to within the eigensolver's own backward error -- a lossless
    propagating mode whose imaginary sign is rounding.  It is SIGNAL-side (the
    band must NOT reach it) otherwise.  The classification is made on
    ``lam^2``, i.e. on the eigenproblem, and never on the ratio itself, so it
    is independent of the quantity under test.
    """
    x = np.asarray(lam2, dtype=complex).ravel()
    r = np.sqrt(x)
    neg = r.imag < 0
    if not np.any(neg):
        return []
    scale_arr = max(float(np.max(np.abs(r))), 1.0)
    top = float(np.max(np.abs(r)))
    floor = SQRT_EPS * max(top, 1.0)
    out = []
    for i in np.nonzero(neg)[0]:
        xi, ri = x[i], r[i]
        on_cut = (xi.real < 0
                  and abs(xi.imag) <= 1e-6 * abs(xi.real))
        out.append(dict(
            array_max=float(abs(ri.real)) / scale_arr,
            per_mode=float(abs(ri.real)) / max(abs(ri), floor),
            own=float(abs(ri.real)) / max(abs(ri), 1e-300),
            abs_r=float(abs(ri)),
            rel_abs_r=float(abs(ri)) / max(top, 1e-300),
            abs_lam2=float(abs(xi)),
            noise=bool(on_cut)))
    return out


def gaps(rows, key):
    noise = [r[key] for r in rows if r["noise"]]
    sig = [r[key] for r in rows if not r["noise"]]
    hi = max(noise) if noise else None
    lo = min(sig) if sig else None
    return dict(n_noise=len(noise), n_signal=len(sig),
                noise_max=hi, signal_min=lo,
                gap_decades=(float(np.log10(lo / hi))
                             if (hi and lo and hi > 0) else None))


# ------------------------------------------------------- RCWA eigen sources
class RcwaEigSpy:
    _MODULES = ("lumenairy.elements.rcwa._core",
                "lumenairy.elements.rcwa.oned")

    def __init__(self):
        self.seen = []
        self._saved = []

    def __enter__(self):
        import importlib

        from lumenairy.elements.rcwa import _core as rc
        orig = rc._eig_for
        seen = self.seen

        def factory(xp):
            base = orig(xp)

            def wrapped(A):
                w, v = base(A)
                seen.append(np.asarray(w).astype(complex).copy())
                return w, v
            return wrapped

        for name in self._MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_eig_for"):
                self._saved.append((mod, mod._eig_for))
                mod._eig_for = factory
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._eig_for = fn
        return False


def _aniso(S=32, twist=0.7, no=1.5, ne=1.7, bg=2.25, eps_im=0.0):
    return F.tensor_cell(S=S, host=bg + 1j * eps_im, no=no, ne=ne,
                         twist=twist, eps_im=eps_im)


def rcwa_fixtures():
    """(name, callable) covering the families the round-1 band was measured
    on plus the CUTOFF mounts that set its noise side."""
    from lumenairy.elements.rcwa import (
        rcwa_efficiency_1d,
        rcwa_efficiency_2d,
        rcwa_jones_2d,
    )
    out = []

    def add(n, f):
        out.append((n, f))

    for twist in (0.0, 0.4, 0.7, 1.1):
        for M in (3, 4, 5):
            for nsub in (1.5, 1.6):
                add("aniso_t%.1f_M%d_ns%.1f" % (twist, M, nsub),
                    lambda t=twist, m=M, n=nsub: rcwa_jones_2d(
                        PX, PX, _aniso(twist=t), n, 1.0, D, WL,
                        n_orders_x=m, n_orders_y=m, symmetry=False))
    add("aniso_oblique", lambda: rcwa_jones_2d(
        PX, PX, _aniso(), 1.5, 1.0, D, WL, theta=0.3, n_orders_x=4,
        n_orders_y=4, symmetry=False))
    add("aniso_conical", lambda: rcwa_jones_2d(
        PX, PX, _aniso(), 1.5, 1.0, D, WL, theta=0.2, phi=0.7, n_orders_x=4,
        n_orders_y=4, symmetry=False))
    for blk in (2.25 + 1e-6, 4.0, 12.25):
        add("scalar2d_%.4g" % blk, lambda b=blk: rcwa_efficiency_2d(
            PX, PX, F.pillar_cell(S=32, host=2.25, pillar=b), 1.5, 1.0, D, WL,
            n_orders_x=5, n_orders_y=5, polarization="te"))
    for pol in ("te", "tm"):
        for duty in (0.2, 0.5, 0.8):
            add("oned_%s_d%.1f" % (pol, duty),
                lambda p=pol, dd=duty: rcwa_efficiency_1d(
                    1.0e-6, 2.1 ** 0.5, 1.5, 1.5, 1.0, 0.4e-6, dd, WL,
                    polarization=p, n_orders=15))
    add("oned_highcontrast_21", lambda: rcwa_efficiency_1d(
        1.0e-6, 3.5, 1.0, 1.5, 1.0, 0.4e-6, 0.5, WL, polarization="te",
        n_orders=21))
    add("oned_highcontrast_31", lambda: rcwa_efficiency_1d(
        1.0e-6, 3.5, 1.0, 1.5, 1.0, 0.4e-6, 0.5, WL, polarization="tm",
        n_orders=31))
    for ei in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        add("lossy2d_%g" % ei, lambda e=ei: rcwa_jones_2d(
            PX, PX, _aniso(eps_im=e), 1.5, 1.0, D, WL, n_orders_x=4,
            n_orders_y=4, symmetry=False))
        add("lossy1d_%g" % ei, lambda e=ei: rcwa_efficiency_1d(
            1.0e-6, (2.1 + 1j * e) ** 0.5, 1.5, 1.5, 1.0, 0.4e-6, 0.5, WL,
            polarization="tm", n_orders=15))
    for em in (-2.0 + 0.1j, -10.0 + 1.0j, -50.0 + 3.0j, -100.0 + 5.0j):
        add("metal_%g" % em.real, lambda e=em: rcwa_efficiency_1d(
            1.0e-6, e ** 0.5, 1.0, 1.5, 1.0, 0.1e-6, 0.5, WL,
            polarization="tm", n_orders=11))
    # near-Wood mounts (a region order at grazing) and near-degenerate cells
    for th in (0.20135792079, 0.4302, 0.6155):
        add("wood_%.3f" % th, lambda t=th: rcwa_efficiency_1d(
            1.0e-6, 2.1 ** 0.5, 1.5, 1.5, 1.0, 0.4e-6, 0.5, WL, angle=t,
            polarization="te", n_orders=15))
    return out


def cutoff_mounts():
    """Mounts driven onto a LAYER CUTOFF (``min|lam^2| -> 0``) by trisecting
    the grating depth-free parameter that moves one modal eigenvalue through
    zero.  These set the noise side of the band on both shapes, which is the
    whole reason the shape is in question."""
    from lumenairy.elements.rcwa import rcwa_efficiency_1d

    def solve(nr, pol, M):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with RcwaEigSpy() as sp:
                try:
                    rcwa_efficiency_1d(1.0e-6, nr, 1.0, 1.5, 1.0, 0.4e-6, 0.5,
                                       WL, polarization=pol, n_orders=M)
                except Exception:
                    pass
            return sp.seen

    out = []
    for pol in ("te", "tm"):
        for M in (7, 11):
            lo, hi = 1.05, 2.6
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                seen = solve(mid, pol, M)
                m = min((float(np.min(np.abs(w))) for w in seen if w.size),
                        default=float("inf"))
                out.append(("cutoff_%s_M%d_n%.9f" % (pol, M, mid), seen, m))
                # walk toward the smallest |lam^2| by trisection on n_ridge
                a = 0.5 * (lo + mid)
                seen_a = solve(a, pol, M)
                ma = min((float(np.min(np.abs(w))) for w in seen_a if w.size),
                         default=float("inf"))
                if ma < m:
                    hi = mid
                else:
                    lo = mid
    return out


# ------------------------------------------------------------- PMM sources
def pmm_hybrid_populations():
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell, pmm_jones_2d
    rows = []
    cells = [("weak_region", F.pillar_cell(S=32, host=2.25,
                                           pillar=2.25 * (1 + 1e-6)), 1.5),
             ("strong_region", F.pillar_cell(S=32, host=2.25, pillar=6.0),
              1.5),
             ("strong_none", F.pillar_cell(S=32, host=2.25, pillar=6.0),
              1.63),
             ("metal", F.pillar_cell(S=32, host=2.25, pillar=-10.0 + 1.0j),
              1.5)]
    for name, cell, nsub in cells:
        for M in (3, 5, 7):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with F.PMMEigSpyRaw() as sp:
                    try:
                        pmm_efficiency_2d_cell(PX, PX, cell, nsub, 1.0, D, WL,
                                               degree=7, n_orders=M,
                                               symmetry=False)
                    except Exception:
                        pass
            for w in sp.arrays:
                rows.extend(score(w))
    for ei in (1e-2, 1e-6, 1e-10):
        cell = F.pillar_cell(S=32, host=2.25, pillar=6.0, eps_im=ei)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with F.PMMEigSpyRaw() as sp:
                try:
                    pmm_efficiency_2d_cell(PX, PX, cell, 1.5, 1.0, D, WL,
                                           degree=7, n_orders=5,
                                           symmetry=False)
                except Exception:
                    pass
        for w in sp.arrays:
            rows.extend(score(w))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with F.PMMEigSpyRaw() as sp:
            try:
                pmm_jones_2d(PX, PX, F.tensor_cell(), 1.5, 1.0, D, WL,
                             degree=7, n_orders=3)
            except Exception:
                pass
    for w in sp.arrays:
        rows.extend(score(w))
    return rows


def pmm_staggered_populations():
    """The staggered pencil's ``gamma^2`` spectrum, collected at
    ``_forward_branch_flip``'s own call sites -- a DIFFERENT population from
    both the RCWA and the hybrid one (it is the generalized pencil ``L x = g2 G
    x``, whose eigenvalues carry the SEM element scaling)."""
    import lumenairy.elements.pmm._core as pc
    import lumenairy.elements.pmm.twod_staggered as ts
    from lumenairy.elements.pmm import (
        pmm_efficiency_2d_staggered,
        pmm_jones_2d_staggered,
    )
    seen = []
    orig = pc._forward_branch_flip

    def spy(q, xp=np):
        try:
            seen.append(np.asarray(q).astype(complex).copy() ** 2)
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
            for pillar in (2.25 * (1 + 1e-6), 6.0, -10.0 + 1.0j):
                for nsub in (1.5, 1.63):
                    try:
                        pmm_efficiency_2d_staggered(
                            PX, PX, F.stag_cell(host=2.25, pillar=pillar),
                            nsub, 1.0, D, WL, degree=6, n_orders=4)
                    except Exception:
                        pass
            for ei in (0.0, 1e-2, 1e-8):
                try:
                    pmm_jones_2d_staggered(
                        PX, PX, F.stag_tensor_cell(eps_im=ei), 1.5, 1.0, D,
                        WL, degree=6, n_orders=3)
                except Exception:
                    pass
    finally:
        for m, fn in saved:
            m._forward_branch_flip = fn
    rows = []
    for g2 in seen:
        rows.extend(score(g2))
    return rows


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b6.json"
    pops = {}

    # ---- RCWA ordinary fixtures
    rows = []
    per_fixture = {}
    for name, fn in rcwa_fixtures():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with RcwaEigSpy() as sp:
                try:
                    fn()
                except Exception as exc:                  # pragma: no cover
                    per_fixture[name] = "RAISED " + type(exc).__name__
        sub = []
        for w in sp.seen:
            sub.extend(score(w))
        per_fixture.setdefault(name, len(sub))
        rows.extend(sub)
    pops["rcwa_ordinary"] = rows
    print("rcwa ordinary: %d fixtures, %d Im(r)<0 modes" % (len(per_fixture),
                                                            len(rows)))

    # ---- RCWA cutoff mounts
    crows = []
    cmin = float("inf")
    for _name, seen, m in cutoff_mounts():
        cmin = min(cmin, m)
        for w in seen:
            crows.extend(score(w))
    pops["rcwa_cutoff"] = crows
    print("rcwa cutoff: %d modes, smallest |lam^2| reached %.4e"
          % (len(crows), cmin))

    pops["pmm_hybrid"] = pmm_hybrid_populations()
    print("pmm hybrid: %d modes" % len(pops["pmm_hybrid"]))
    pops["pmm_staggered"] = pmm_staggered_populations()
    print("pmm staggered: %d modes" % len(pops["pmm_staggered"]))

    summary = {}
    for pop, rows in pops.items():
        summary[pop] = dict(
            array_max=gaps(rows, "array_max"),
            per_mode=gaps(rows, "per_mode"),
            own=gaps(rows, "own"),
            worst_own_ratio_caught_by_array_max=max(
                [r["own"] for r in rows if r["array_max"] <= 1e-8] or [None],
                key=lambda v: (-1 if v is None else v)),
            min_rel_abs_r=min([r["rel_abs_r"] for r in rows] or [None],
                              key=lambda v: (2 if v is None else v)),
        )
        for shape in ("array_max", "per_mode"):
            g = summary[pop][shape]
            print("  %-16s %-9s noise_max %-11s signal_min %-11s gap %s dec"
                  % (pop, shape,
                     "%.4e" % g["noise_max"] if g["noise_max"] else "-",
                     "%.4e" % g["signal_min"] if g["signal_min"] else "-",
                     "%.2f" % g["gap_decades"] if g["gap_decades"] else "-"))

    F.dump(out, dict(summary=summary,
                     cutoff_min_abs_lam2=cmin,
                     sqrt_eps_mach=SQRT_EPS,
                     rcwa_per_fixture=per_fixture,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
