"""TASK 3 (continued) -- IS THE CUTOFF CORNER HARMLESS?

``v4_band.py`` drives ``min |lam^2|`` to ~3e-16 -- six decades deeper than the
round-2 document's own ladder (6.5341e-10) and 1.2 decades deeper than the
round-1 verification's (4.495e-15).  At that depth the ARRAY-MAX band's two
populations OVERLAP: its noise side reaches 3.9195e-09 while its signal side
falls to 1.9954e-09, and a mode whose real part is 99.76 % of its OWN magnitude
is conjugated.  The round-2 document asserts the consequence is nil ("it carries
no z-directed flux, and ``_inv_lam`` regularises it downstream, so no wrong
answer follows from either choice on any fixture measured").

This probe MEASURES that, rather than reasoning about it.  On each of the
deepest mounts, the same solve is run under FOUR branch rules:

  shipped   the round-2 body, band = 1e-8, ARRAY-MAX scale
  band0     band = 0 -- the conjugation never fires (the "do nothing" arm)
  permode   |Re r| <= 1e-8 max(|r|, sqrt(eps_mach) max|r|)
  prepin    the pre-ROUND-1 exact ``Re(r) == 0`` pin with the ``-r`` flip

and compared on
  * the lossless closure defect (an exact conservation law: an INDEPENDENT
    oracle whose error floor is the arithmetic);
  * the per-order efficiencies, arm against arm;
  * whether the solve RETURNS, WARNS or RAISES (the D5 shape).

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v6_cutoff_consequence.py out.json
"""
from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v6.json"
WL = 0.5321e-6
EPSM = float(np.finfo(np.float64).eps)
SQEPS = float(np.sqrt(EPSM))
_C = np.complex128


#: The shipped body, captured ONCE at import so the ``Arm`` monkeypatch below
#: cannot make ``rule_shipped`` call itself (it did on the first pass of this
#: probe, and every "shipped" row read RecursionError as a raise).
from lumenairy.elements.rcwa._core import _sqrt_decay as _SHIPPED_BODY  # noqa: E402


def rule_shipped(x):
    return _SHIPPED_BODY(x)


def rule_band0(x):
    x = np.asarray(x).astype(_C)
    return np.sqrt(x)


def rule_permode(x, band=1e-8):
    x = np.asarray(x).astype(_C)
    r = np.sqrt(x)
    mx = max(float(np.max(np.abs(r))), 1.0) if r.size else 1.0
    on_cut = np.abs(r.real) <= band * np.maximum(np.abs(r), SQEPS * mx)
    return np.where(on_cut & (r.imag < 0), np.conj(r), r)


def rule_prepin(x):
    x = np.asarray(x).astype(_C)
    r = np.sqrt(x)
    on_cut = r.real == 0
    return np.where(on_cut & (r.imag < 0), -r, r)


RULES = {"shipped": rule_shipped, "band0": rule_band0,
         "permode": rule_permode, "prepin": rule_prepin}


class Arm:
    def __init__(self, rule):
        self.rule = rule
        self._saved = []

    def __enter__(self):
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
            mod._sqrt_decay = (lambda x, *a, **kw: self.rule(x))
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        self._saved = []
        return False


def min_lam2(n_ridge, pol, M, duty):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    seen = []
    import lumenairy.elements.rcwa._core as rc
    saved = rc._sqrt_decay
    import lumenairy.elements.rcwa.oned as ro
    saved_o = ro._sqrt_decay

    def tap(x, *a, **kw):
        xx = np.asarray(x).astype(_C)
        if xx.size and not np.all(xx.imag == 0.0):
            nz = np.abs(xx[np.abs(xx) > 0])
            if nz.size:
                seen.append(float(nz.min()))
        return saved(x, *a, **kw)

    rc._sqrt_decay = tap
    ro._sqrt_decay = tap
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcwa_efficiency_1d(1.0e-6, n_ridge, 1.0, 1.5, 1.0, 0.45e-6, duty,
                               WL, polarization=pol, n_orders=M)
    except Exception:
        pass
    finally:
        rc._sqrt_decay = saved
        ro._sqrt_decay = saved_o
    return min(seen) if seen else float("inf")


def hunt(pol, M, duty):
    grid = np.linspace(1.3, 3.4, 61)
    vals = [min_lam2(g, pol, M, duty) for g in grid]
    k = int(np.argmin(vals))
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
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
    return nr, min_lam2(nr, pol, M, duty)


def solve(n_ridge, pol, M, duty):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    out = {}
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            o, R, T = rcwa_efficiency_1d(1.0e-6, n_ridge, 1.0, 1.5, 1.0,
                                         0.45e-6, duty, WL,
                                         polarization=pol, n_orders=M)
            R, T = np.asarray(R), np.asarray(T)
            out["closure"] = float(R.sum() + T.sum() - 1.0)
            out["R"] = R.tolist()
            out["T"] = T.tolist()
            out["status"] = "returned"
        except Exception as exc:
            out["status"] = "raised"
            out["error"] = f"{type(exc).__name__}"
            out["message_head"] = str(exc)[:120]
        out["warnings"] = sorted({type(w.message).__name__ for w in ws})
        if out["warnings"]:
            out["status"] = "warned"
    return out


def run():
    payload = {"mounts": []}
    for pol in ("te", "tm"):
        for M in (9, 11, 15):
            for duty in (0.4, 0.5, 0.62):
                nr, v = hunt(pol, M, duty)
                rows = {}
                for name, rule in RULES.items():
                    with Arm(rule):
                        rows[name] = solve(nr, pol, M, duty)
                # per-order motion, every arm against the shipped one
                base = rows["shipped"]
                for name, r in rows.items():
                    if name == "shipped" or "R" not in r or "R" not in base:
                        continue
                    dR = np.max(np.abs(np.asarray(r["R"])
                                       - np.asarray(base["R"])))
                    dT = np.max(np.abs(np.asarray(r["T"])
                                       - np.asarray(base["T"])))
                    r["vs_shipped_max_per_order"] = float(max(dR, dT))
                for r in rows.values():
                    r.pop("R", None)
                    r.pop("T", None)
                payload["mounts"].append(
                    {"pol": pol, "M": M, "duty": duty, "n_ridge": float(nr),
                     "min_abs_lam2": float(v), "arms": rows})
    VC.dump(OUT, payload)

    st = VC.stamp()
    print(f"ARM = {st['arm']}  python {st['python']}")
    print(f"{'mount':22s} {'min|lam2|':>11s} " + " ".join(
        f"{k:>26s}" for k in RULES))
    worst = 0.0
    for m in sorted(payload["mounts"], key=lambda z: z["min_abs_lam2"]):
        cells = []
        for k in RULES:
            r = m["arms"][k]
            c = r.get("closure")
            d = r.get("vs_shipped_max_per_order")
            cells.append(
                f"{r['status'][:4]} "
                f"{(c if c is not None else float('nan')):11.3e} "
                f"d={(d if d is not None else 0.0):9.2e}")
            if d:
                worst = max(worst, d)
        print(f"{m['pol']}_M{m['M']}_d{m['duty']:<6} "
              f"{m['min_abs_lam2']:11.3e} " + " ".join(cells))
    print(f"\nworst per-order motion between branch rules over all mounts: "
          f"{worst:.4e}")


if __name__ == "__main__":
    run()
