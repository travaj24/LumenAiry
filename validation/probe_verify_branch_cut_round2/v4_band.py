"""TASK 3 -- the band scale, re-derived on MY OWN populations.

ARRAY-MAX (shipped)   ``|Re r| <= C * max(max|r|, 1)``
PER-MODE (rejected)   ``|Re r| <= C * max(|r|, sqrt(eps_mach) * max|r|)``

The census taps ``_sqrt_decay`` and records its RAW input, so every ratio is a
property of the EIGENPROBLEM and identical on both arms of the change.  Three
populations:

  1. RCWA ORDINARY -- 1-D TE/TM at three duties, 2-D scalar and anisotropic,
     oblique, conical, four metals, a loss ladder Im(eps) 1e-2 .. 1e-16, and
     near-Wood mounts.
  2. HYBRID PMM -- the SEM-projected ``P@Q`` spectrum, weak and strong
     modulation, on and off the coincidence, lossy, three truncations.
  3. RCWA LAYER CUTOFF -- >= 20 mounts found by a golden-section minimisation
     of ``min |lam^2|`` over the ridge index, each with a relative +/- ladder,
     driven as deep as the search will go.

Classification (made on ``lam^2`` and on the FIXTURE, never on the ratio under
test):
  NOISE side  -- a mode of a provably LOSSLESS fixture with ``Re(lam^2) < 0``
                 AND ``|Im(lam^2)| <= K eps_mach max|lam^2|`` (K = 1e3): a
                 PROPAGATING mode whose imaginary part is the eigensolver's own
                 backward error, so the band MUST reach it.
  SIGNAL side -- anything else in the ACTED-ON population (``Im(r) < 0``); the
                 band must NOT reach it.
  COMPLEX     -- reported separately: a mode of a LOSSLESS fixture with
                 ``Re(lam^2) < 0`` whose ``|Im(lam^2)|`` is ABOVE the backward
                 error.  These are genuine complex (leaky) modes of a lossless
                 cell.  They are counted with the SIGNAL side, and the count is
                 printed, because whether one calls them noise or signal is
                 exactly what makes a single "census every population the same
                 way" pass unreliable -- classifying them as noise on the
                 hybrid PMM population collapses the array-max gap from 6.6
                 decades to 2.2 and puts the "noise side" 4.5 decades ABOVE the
                 shipped bar.  BOTH readings are reported.
  A LOSSY fixture contributes NO noise-side modes by construction: there every
  imaginary part carries physics.
Region arrays (``-kz^2``, built in exact arithmetic, ``Im`` exactly zero) are
tagged and excluded from the eigenproblem populations.

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v4_band.py out.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v4.json"
WL = 0.5321e-6
EPSM = float(np.finfo(np.float64).eps)
SQEPS = float(np.sqrt(EPSM))


class SqrtTap:
    """Records the RAW input of every ``_sqrt_decay`` call, in every binding
    that exists on this tree."""

    def __init__(self):
        self.arrays = []
        self._saved = []

    def __enter__(self):
        import importlib

        import lumenairy.elements as EL
        base = Path(EL.__file__).parent
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(base).with_suffix("")
            name = "lumenairy.elements." + ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                mod = importlib.import_module(name)
            except Exception:
                continue
            fn = getattr(mod, "_sqrt_decay", None)
            if fn is None or not callable(fn):
                continue
            self._saved.append((mod, fn))

            def wrapped(x, *a, _fn=fn, **kw):
                try:
                    self.arrays.append(np.asarray(x).astype(np.complex128)
                                       .ravel().copy())
                except Exception:
                    pass
                return _fn(x, *a, **kw)

            mod._sqrt_decay = wrapped
        return self

    def __exit__(self, *exc):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        self._saved = []
        return False


def score(tagged, label):
    """Both shapes, over every acted-on mode of every EIGENPROBLEM array.

    ``tagged`` is a list of ``(array, lossless)`` pairs."""
    noise_am, sig_am, noise_pm, sig_pm = [], [], [], []
    loose_noise_am, loose_noise_pm = [], []
    n_complex = 0
    worst = {"rho": 0.0, "abs_r": None, "abs_im_r": None, "abs_lam2": None}
    closest_noise = {"ratio": 0.0}      # lossless propagating, nearest the bar
    closest_signal = {"ratio": float("inf")}   # lossy/evanescent, nearest bar
    n_pop = 0
    n_arrays = 0
    min_lam2 = float("inf")
    for x, lossless in tagged:
        if x.size == 0:
            continue
        if np.all(x.imag == 0.0):
            continue                     # region array, exact arithmetic
        n_arrays += 1
        r = np.sqrt(x)
        mx = max(float(np.max(np.abs(r))), 1.0)
        sel = r.imag < 0
        if not np.any(sel):
            continue
        nz = np.abs(x[np.abs(x) > 0])
        if nz.size:
            min_lam2 = min(min_lam2, float(nz.min()))
        rr = r[sel]
        xx = x[sel]
        n_pop += int(sel.sum())
        am = np.abs(rr.real) / mx
        floor = SQEPS * mx
        pm = np.abs(rr.real) / np.maximum(np.abs(rr), floor)
        bwe = 1e3 * EPSM * float(np.max(np.abs(x)))
        prop = (xx.real < 0)
        real_to_bwe = np.abs(xx.imag) <= bwe
        if lossless:
            is_noise = prop & real_to_bwe
            is_complex = prop & ~real_to_bwe
        else:
            is_noise = np.zeros(xx.shape, bool)
            is_complex = np.zeros(xx.shape, bool)
        n_complex += int(is_complex.sum())
        if np.any(is_complex):
            loose_noise_am.extend(am[prop].tolist())
            loose_noise_pm.extend(pm[prop].tolist())
        else:
            loose_noise_am.extend(am[is_noise].tolist())
            loose_noise_pm.extend(pm[is_noise].tolist())
        noise_am.extend(am[is_noise].tolist())
        sig_am.extend(am[~is_noise].tolist())
        noise_pm.extend(pm[is_noise].tolist())
        sig_pm.extend(pm[~is_noise].tolist())
        if np.any(is_noise):
            k = int(np.argmax(am[is_noise]))
            v = float(am[is_noise][k])
            if v > closest_noise["ratio"]:
                closest_noise = {
                    "ratio": v, "bar": 1e-8, "ratio_over_bar": v / 1e-8,
                    "abs_lam2": float(np.abs(xx[is_noise][k])),
                    "abs_r": float(np.abs(rr[is_noise][k]))}
        if np.any(~is_noise):
            k = int(np.argmin(am[~is_noise]))
            v = float(am[~is_noise][k])
            if v < closest_signal["ratio"]:
                closest_signal = {
                    "ratio": v, "bar": 1e-8, "bar_over_ratio": 1e-8 / max(v, 1e-300),
                    "abs_lam2": float(np.abs(xx[~is_noise][k])),
                    "abs_r": float(np.abs(rr[~is_noise][k])),
                    "lossless_fixture": bool(lossless)}
        acted = am <= 1e-8
        if np.any(acted):
            rho = np.abs(rr.real[acted]) / np.maximum(np.abs(rr[acted]),
                                                      1e-300)
            k = int(np.argmax(rho))
            if float(rho[k]) > worst["rho"]:
                worst = {"rho": float(rho[k]),
                         "abs_r": float(np.abs(rr[acted][k])),
                         "abs_im_r": float(np.abs(rr[acted][k].imag)),
                         "abs_lam2": float(np.abs(xx[acted][k])),
                         "lossless_fixture": bool(lossless)}
    def sm(v, f):
        return float(f(v)) if v else None
    out = {
        "label": label, "n_eig_arrays": n_arrays, "n_acted_on_population": n_pop,
        "min_abs_lam2": (None if min_lam2 == float("inf") else min_lam2),
        "arraymax_noise_max": sm(noise_am, max),
        "arraymax_signal_min": sm(sig_am, min),
        "permode_noise_max": sm(noise_pm, max),
        "permode_signal_min": sm(sig_pm, min),
        "worst_conjugated": worst,
        "closest_noise_to_bar": closest_noise,
        "closest_signal_to_bar": closest_signal,
        "n_noise": len(noise_am), "n_signal": len(sig_am),
        "n_complex_modes_of_a_lossless_cell": n_complex,
        "LOOSE_arraymax_noise_max": sm(loose_noise_am, max),
        "LOOSE_permode_noise_max": sm(loose_noise_pm, max),
    }
    for shape in ("arraymax", "permode"):
        nmax = out[f"{shape}_noise_max"]
        smin = out[f"{shape}_signal_min"]
        if nmax and smin and nmax > 0:
            out[f"{shape}_gap_decades"] = float(np.log10(smin / nmax))
            out[f"{shape}_decades_above_noise"] = float(np.log10(1e-8 / nmax))
            out[f"{shape}_decades_below_signal"] = float(np.log10(smin / 1e-8))
    return out


def collect(fns, label):
    """``fns`` is a list of ``(name, callable, lossless)``.  Each fixture is
    tapped SEPARATELY so its arrays carry its own lossless flag."""
    errs = []
    tagged = []
    for name, fn, lossless in fns:
        tap = SqrtTap()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with tap:
                try:
                    fn()
                except Exception as exc:
                    errs.append(f"{name}: {type(exc).__name__}: {exc}")
        tagged.extend((a, lossless) for a in tap.arrays)
    row = score(tagged, label)
    row["errors"] = errs
    row["n_fixtures"] = len(fns)
    return row


# ---------------------------------------------------------------------------
# population 1 -- RCWA ordinary
# ---------------------------------------------------------------------------
def rcwa_ordinary_fixtures():
    """(name, callable, LOSSLESS).  ``lossless`` is a property of the FIXTURE
    I built, not something read back from the solve."""
    from lumenairy.elements.rcwa import rcwa_efficiency_1d, rcwa_efficiency_2d, rcwa_jones_2d
    F = []
    for duty in (0.3, 0.5, 0.72):
        for pol in ("te", "tm"):
            for M in (7, 13, 21):
                F.append((f"1d_{pol}_{duty}_{M}",
                          lambda d=duty, p=pol, m=M: rcwa_efficiency_1d(
                              1.0e-6, 2.1, 1.0, 1.5, 1.0, 0.45e-6, d, WL,
                              polarization=p, n_orders=m), True))
    for th in (0.0, 0.21, 0.55):
        F.append((f"1d_obl_{th}",
                  lambda t=th: rcwa_efficiency_1d(
                      1.0e-6, 2.1, 1.0, 1.5, 1.0, 0.45e-6, 0.5, WL,
                      theta=t, n_orders=11), True))
        F.append((f"1d_obl_tm_{th}",
                  lambda t=th: rcwa_efficiency_1d(
                      1.0e-6, 2.1, 1.0, 1.5, 1.0, 0.45e-6, 0.5, WL,
                      theta=t, polarization="tm", n_orders=11), True))

    def cell2d(contrast, S=24):
        """A 4 x 4 block pattern sampled at ``S`` points per axis so the
        Fourier convolution is not refused (the entry needs >= 4M + 1)."""
        c = np.full((S, S), 1.0, dtype=complex)
        q = S // 4
        c[q:3 * q, q // 2:5 * q // 2] = contrast
        return c

    for con in (1.44, 4.0, 12.0):
        for M in (3, 5):
            F.append((f"2d_{con}_{M}",
                      lambda cc=con, m=M: rcwa_efficiency_2d(
                          0.62e-6, 0.58e-6, cell2d(cc), 1.5, 1.0, 0.3e-6, WL,
                          n_orders_x=m, n_orders_y=m), True))
    for tw in (0.0, 0.3, 1.1):
        tens = np.zeros((16, 16, 3, 3), dtype=complex)
        base = np.full((16, 16), 2.1, dtype=complex)
        base[3:10, 5:12] = 4.2
        ct, stt = np.cos(tw), np.sin(tw)
        tens[..., 0, 0] = base * ct ** 2 + 2.0 * stt ** 2
        tens[..., 1, 1] = base * stt ** 2 + 2.0 * ct ** 2
        tens[..., 2, 2] = base
        tens[..., 0, 1] = tens[..., 1, 0] = (base - 2.0) * ct * stt
        F.append((f"aniso_{tw}",
                  lambda t=tens: rcwa_jones_2d(
                      0.62e-6, 0.58e-6, t, 1.5, 1.0, 0.3e-6, WL,
                      n_orders=3), True))
        F.append((f"aniso_conical_{tw}",
                  lambda t=tens: rcwa_jones_2d(
                      0.62e-6, 0.58e-6, t, 1.5, 1.0, 0.3e-6, WL, n_orders=3,
                      theta=0.31, phi=0.77), True))
    # LOSS LADDER -- every rung is a LOSSY fixture, so it contributes only to
    # the SIGNAL side.  Below ~1e-14 the loss is under the eigensolver's own
    # backward error and no bar on any shape could separate it; those rungs are
    # therefore reported but flagged.
    for im in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
        F.append((f"loss1d_{im}",
                  lambda v=im: rcwa_efficiency_1d(
                      1.0e-6, np.sqrt(4.41 + 1j * v), 1.0, 1.5, 1.0, 0.45e-6,
                      0.5, WL, n_orders=11), False))
        F.append((f"loss1d_tm_{im}",
                  lambda v=im: rcwa_efficiency_1d(
                      1.0e-6, np.sqrt(4.41 + 1j * v), 1.0, 1.5, 1.0, 0.45e-6,
                      0.5, WL, polarization="tm", n_orders=11), False))
        F.append((f"loss2d_{im}",
                  lambda v=im: rcwa_efficiency_2d(
                      0.62e-6, 0.58e-6, cell2d(4.0 + 1j * v), 1.5, 1.0,
                      0.3e-6, WL, n_orders_x=3, n_orders_y=3), False))
    for eps in (-2.0 + 0.1j, -10.0 + 1.0j, -40.0 + 2.5j, -100.0 + 5.0j):
        F.append((f"metal_{eps.real}",
                  lambda e=eps: rcwa_efficiency_1d(
                      1.0e-6, np.sqrt(e), 1.0, 1.5, 1.0, 0.12e-6, 0.5, WL,
                      n_orders=11), False))
    for frac in (0.999, 1.0, 1.001):
        P = WL * 1.5 * frac
        F.append((f"wood_{frac}",
                  lambda p=P: rcwa_efficiency_1d(
                      p, 2.1, 1.0, 1.5, 1.0, 0.45e-6, 0.5, WL,
                      n_orders=9), True))
    return F


def hybrid_pmm_fixtures():
    from lumenairy.elements.pmm import pmm_efficiency_2d_cell, pmm_jones_2d
    F = []

    def cell(bg, rel):
        c = np.full((6, 6), float(bg), dtype=complex)
        c[1:3, 2:4] = bg * (1.0 + rel)
        return c


    for rel in (1e-6, 1e-2, 0.8):
        for nsub in (1.5, 1.63):
            for M in (3, 4, 5):
                F.append((f"hyb_{rel}_{nsub}_{M}",
                          lambda r=rel, s=nsub, m=M: pmm_efficiency_2d_cell(
                              0.62e-6, 0.58e-6, cell(2.25, r), s, 1.0,
                              0.23e-6, WL, n_orders=m, degree=7,
                              symmetry=False), True))
    for im in (1e-2, 1e-6, 1e-12):
        F.append((f"hyb_lossy_{im}",
                  lambda v=im: pmm_efficiency_2d_cell(
                      0.62e-6, 0.58e-6, cell(2.25, 1e-2) + 1j * v, 1.5, 1.0,
                      0.23e-6, WL, n_orders=4, degree=7,
                      symmetry=False), False))
    F.append(("hyb_metal",
              lambda: pmm_efficiency_2d_cell(
                  0.62e-6, 0.58e-6, cell(2.25, 1e-2) - 10.0, 1.5, 1.0,
                  0.12e-6, WL, n_orders=4, degree=7, symmetry=False), False))
    tens = np.zeros((6, 6, 3, 3), dtype=complex)
    base = np.full((6, 6), 2.25, dtype=complex)
    base[1:3, 2:4] = 2.25 * (1 + 1e-6)
    for i in range(3):
        tens[..., i, i] = base
    F.append(("hyb_tensor",
              lambda t=tens: pmm_jones_2d(0.62e-6, 0.58e-6, t, 1.5, 1.0,
                                          0.23e-6, WL, n_orders=3, degree=7,
                                          symmetry=False), True))
    F.append(("hyb_tensor_conical",
              lambda t=tens: pmm_jones_2d(0.62e-6, 0.58e-6, t, 1.5, 1.0,
                                          0.23e-6, WL, n_orders=3, degree=7,
                                          theta=0.31, phi=0.77,
                                          symmetry=False), True))
    return F


def min_lam2(n_ridge, pol="te", M=11, duty=0.5):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    tap = SqrtTap()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with tap:
            try:
                rcwa_efficiency_1d(1.0e-6, n_ridge, 1.0, 1.5, 1.0, 0.45e-6,
                                   duty, WL, polarization=pol, n_orders=M)
            except Exception:
                return float("inf")
    best = float("inf")
    for x in tap.arrays:
        if x.size == 0 or np.all(x.imag == 0.0):
            continue
        nz = np.abs(x[np.abs(x) > 0])
        if nz.size:
            best = min(best, float(nz.min()))
    return best


def hunt_cutoffs():
    """Golden-section on ``n_ridge`` to drive ``min|lam^2|`` down, seeded by a
    coarse scan; then a relative +/- ladder around each corner so a mount is a
    FAMILY, not one engineered point."""
    mounts = []
    for pol in ("te", "tm"):
        for M in (9, 11, 15):
            for duty in (0.4, 0.5, 0.62):
                grid = np.linspace(1.3, 3.4, 61)
                vals = [min_lam2(g, pol, M, duty) for g in grid]
                k = int(np.argmin(vals))
                lo = grid[max(k - 1, 0)]
                hi = grid[min(k + 1, len(grid) - 1)]
                # golden-section
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                a, b = lo, hi
                c, d = b - gr * (b - a), a + gr * (b - a)
                fc, fd = min_lam2(c, pol, M, duty), min_lam2(d, pol, M, duty)
                for _ in range(90):
                    if fc < fd:
                        b, d, fd = d, c, fc
                        c = b - gr * (b - a)
                        fc = min_lam2(c, pol, M, duty)
                    else:
                        a, c, fc = c, d, fd
                        d = a + gr * (b - a)
                        fd = min_lam2(d, pol, M, duty)
                    if abs(b - a) < 1e-15 * max(abs(a), 1.0):
                        break
                nr = 0.5 * (a + b)
                mounts.append((f"cut_{pol}_M{M}_d{duty}", nr, pol, M, duty,
                               min_lam2(nr, pol, M, duty)))
    return mounts


def cutoff_population(mounts):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    F = []
    for name, nr, pol, M, duty, _v in mounts:
        for rel in (0.0, 1e-3, -1e-3, 1e-6, -1e-6, 1e-9, -1e-9, 1e-12, -1e-12):
            F.append((f"{name}_{rel}",
                      lambda n=nr * (1 + rel), p=pol, m=M, d=duty:
                      rcwa_efficiency_1d(1.0e-6, n, 1.0, 1.5, 1.0, 0.45e-6, d,
                                         WL, polarization=p, n_orders=m),
                      True))
    return F


def run():
    payload = {}
    payload["pop1_rcwa_ordinary"] = collect(rcwa_ordinary_fixtures(),
                                            "RCWA ordinary")
    payload["pop2_hybrid_pmm"] = collect(hybrid_pmm_fixtures(), "hybrid PMM")
    mounts = hunt_cutoffs()
    payload["cutoff_mounts"] = [
        {"name": n, "n_ridge": float(nr), "pol": p, "M": M, "duty": d,
         "min_abs_lam2": float(v)} for n, nr, p, M, d, v in mounts]
    payload["pop3_rcwa_cutoff"] = collect(cutoff_population(mounts),
                                          "RCWA layer cutoff")
    VC.dump(OUT, payload)

    st = VC.stamp()
    print(f"ARM = {st['arm']}   python {st['python']}  numpy {st['numpy']}")
    print(f"{'population':22s} {'n':>6s} {'AM noise':>11s} {'AM signal':>11s} "
          f"{'AM gap':>7s} {'PM noise':>11s} {'PM signal':>11s} {'PM gap':>7s} "
          f"{'min|lam2|':>11s}")
    for k in ("pop1_rcwa_ordinary", "pop2_hybrid_pmm", "pop3_rcwa_cutoff"):
        r = payload[k]
        print(f"{k:22s} {r['n_acted_on_population']:6d} "
              f"{(r['arraymax_noise_max'] or float('nan')):11.4e} "
              f"{(r['arraymax_signal_min'] or float('nan')):11.4e} "
              f"{r.get('arraymax_gap_decades', float('nan')):7.2f} "
              f"{(r['permode_noise_max'] or float('nan')):11.4e} "
              f"{(r['permode_signal_min'] or float('nan')):11.4e} "
              f"{r.get('permode_gap_decades', float('nan')):7.2f} "
              f"{(r['min_abs_lam2'] or float('nan')):11.4e}")
        print(f"    decades above noise AM {r.get('arraymax_decades_above_noise')}"
              f"  below signal AM {r.get('arraymax_decades_below_signal')}")
        print(f"    worst conjugated: {r['worst_conjugated']}")
        print(f"    closest NOISE to bar: {r['closest_noise_to_bar']}")
        print(f"    closest SIGNAL to bar: {r['closest_signal_to_bar']}")
        print(f"    complex modes of a lossless cell: "
              f"{r['n_complex_modes_of_a_lossless_cell']}   LOOSE (count them "
              f"as noise) AM noise max {r['LOOSE_arraymax_noise_max']}  "
              f"PM {r['LOOSE_permode_noise_max']}")
        if r["errors"]:
            print(f"    errors: {r['errors'][:4]}")
    print("\ndeepest cutoff mounts:")
    for m in sorted(payload["cutoff_mounts"],
                    key=lambda z: z["min_abs_lam2"])[:8]:
        print(f"   {m['name']:22s} n_ridge={m['n_ridge']:.15f} "
              f"min|lam2|={m['min_abs_lam2']:.4e}")


if __name__ == "__main__":
    run()
