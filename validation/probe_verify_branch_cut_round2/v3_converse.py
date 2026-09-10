"""TASK 2, the CONVERSE -- can the PURE STAGGERED 2-D PMM or the 1-D PMM be
made to show the same coincidence pathology?

The round-2 claim is that they are IMMUNE because they never call
``_sqrt_decay`` (they select through ``pmm/_core._forward_branch_flip``, and on
the out-of-plane path through ``rcwa/_core._select_forward_flux``).  This probe
tries to BREAK that:

  * ``PMM2DStackPure`` with a UNIFORM SPACER of exactly the patterned layer's
    background permittivity, in-plane SCALAR and in-plane TENSOR (via
    ``mu``/``eps`` cells), on SHARED and PER-LAYER grids, at normal and
    conical incidence, weak and ordinary modulation;
  * region coincidence as well (``n_substrate^2`` and ``n_superstrate^2``
    equal to the background);
  * the 1-D ``PMMStack`` and ``pmm_efficiency_1d`` / ``pmm_jones_1d`` with the
    groove index equal to BOTH half-spaces (the X-1 shape) and with a uniform
    spacer layer of the groove index;
  * an explicit COUNTER of ``_sqrt_decay`` calls and of
    ``_forward_branch_flip`` calls made by each surface, so "never calls it" is
    a measurement rather than a reading of the source.

Everything is reported with the closure defect (an independent oracle) and,
where a reference exists, with a per-order comparison against RCWA.

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v3_converse.py out.json
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v3.json"
WL = 0.5321e-6


# ---------------------------------------------------------------------------
# call counters
# ---------------------------------------------------------------------------
class CallTap:
    """Counts calls to the branch selectors, in EVERY module binding that
    exists on this tree (the pre-round-2 tree has five private
    ``_sqrt_decay`` copies; the post tree has one)."""

    NAMES = ("_sqrt_decay", "_forward_branch_flip", "_select_forward_flux")

    def __init__(self):
        self.counts = {}
        self._saved = []

    def __enter__(self):
        import importlib

        import lumenairy.elements as EL
        base = Path(EL.__file__).parent
        mods = []
        for p in sorted(base.rglob("*.py")):
            rel = p.relative_to(base).with_suffix("")
            name = "lumenairy.elements." + ".".join(rel.parts)
            if name.endswith(".__init__"):
                name = name[: -len(".__init__")]
            try:
                mods.append(importlib.import_module(name))
            except Exception:
                pass
        for mod in mods:
            for nm in self.NAMES:
                fn = getattr(mod, nm, None)
                if fn is None or not callable(fn):
                    continue
                key = f"{mod.__name__}.{nm}"
                self.counts.setdefault(key, 0)
                self._saved.append((mod, nm, fn))

                def wrapped(*a, _fn=fn, _key=key, **kw):
                    self.counts[_key] = self.counts.get(_key, 0) + 1
                    return _fn(*a, **kw)

                setattr(mod, nm, wrapped)
        return self

    def __exit__(self, *exc):
        for mod, nm, fn in self._saved:
            setattr(mod, nm, fn)
        self._saved = []
        return False

    def nonzero(self):
        return {k: v for k, v in self.counts.items() if v}


import time as _time


def _run(fn, tap=True, name=""):
    ctx = CallTap() if tap else None
    row = {}
    t0 = _time.time()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            if ctx is not None:
                ctx.__enter__()
            row.update(fn())
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if ctx is not None:
                ctx.__exit__()
        row["warnings"] = sorted({type(w.message).__name__ for w in ws})
    if ctx is not None:
        row["calls"] = ctx.nonzero()
    row["seconds"] = round(_time.time() - t0, 2)
    print(f"  [{row['seconds']:8.2f}s] {name:32s} "
          f"closure={row.get('closure')} {row.get('error') or ''}",
          flush=True)
    return row


def weak(npx, bg, rel, blk):
    c = np.full((npx, npx), float(bg), dtype=complex)
    c[blk[0]:blk[1], blk[2]:blk[3]] = bg * (1.0 + rel)
    return c


# ---------------------------------------------------------------------------
# PURE STAGGERED stack -- try to break it on the SAME coincidence
# ---------------------------------------------------------------------------
#: The PURE STAGGERED engine costs ~40 s per solve at a 6-wall grid and
#: ``n_modes = 6`` on this box (measured 2.5 / 13.6 / 43.3 s at n_modes
#: 4 / 5 / 6, 2026-09-10), so the converse fixtures use a 4 x 4 cell.  The
#: coincidence being tested -- a UNIFORM SPACER of exactly the patterned
#: layer's background -- is unchanged by the grid size.
_PURE_NPX = 4


def pure_stack(bg, rel, nsub, nsup, spacers, grids, theta, phi, n_modes=5,
               n_orders=2, tensor=False, per_layer_walls=False):
    from lumenairy.elements.pmm import PMM2DStackPure
    cell = weak(_PURE_NPX, bg, rel, (1, 2, 1, 3))
    st = PMM2DStackPure(0.62e-6, 0.58e-6, n_superstrate=nsup,
                        n_substrate=nsub, n_modes=n_modes, n_orders=n_orders,
                        layer_grids=grids, symmetry=False)
    kw = {}
    if per_layer_walls:
        kw = dict(
            x_walls=np.linspace(0.0, 0.62e-6, _PURE_NPX + 1)[1:-1].tolist(),
            y_walls=np.linspace(0.0, 0.58e-6, _PURE_NPX + 1)[1:-1].tolist())
    if spacers:
        st.add_layer(0.12e-6, eps=bg)
    if tensor:
        mu_cell = np.ones_like(cell)
        st.add_layer(0.23e-6, eps_cell=cell, mu_cell=mu_cell, **kw)
    else:
        st.add_layer(0.23e-6, eps_cell=cell, **kw)
    if spacers:
        st.add_layer(0.09e-6, eps=bg)
    st.set_source(WL, theta=theta, phi=phi)
    res = st.solve()
    R = np.asarray(res[1] if isinstance(res, tuple) else res.R)
    T = np.asarray(res[2] if isinstance(res, tuple) else res.T)
    tgt = 2.0 if R.ndim == 2 else 1.0
    return {"sumRT": float(R.sum() + T.sum()),
            "closure": float(R.sum() + T.sum() - tgt),
            "R": R.tolist(), "T": T.tolist()}


def staggered_cell(bg, rel, nsub, nsup, theta, phi, jones=False):
    from lumenairy.elements.pmm import pmm_efficiency_2d_staggered, pmm_jones_2d_staggered
    cell = weak(_PURE_NPX, bg, rel, (1, 2, 1, 3))
    if jones:
        res = pmm_jones_2d_staggered(0.62e-6, 0.58e-6, cell, nsub, nsup,
                                     0.23e-6, WL, n_orders=2, degree=8,
                                     n_modes=5, theta=theta, phi=phi,
                                     symmetry=False)
        R = np.asarray(res[1])
        T = np.asarray(res[2])
        tgt = 2.0
    else:
        o, R, T = pmm_efficiency_2d_staggered(
            0.62e-6, 0.58e-6, cell, nsub, nsup, 0.23e-6, WL, n_orders=2,
            degree=8, n_modes=5, theta=theta, phi=phi)
        R, T = np.asarray(R), np.asarray(T)
        tgt = 1.0
    return {"sumRT": float(R.sum() + T.sum()),
            "closure": float(R.sum() + T.sum() - tgt),
            "R": R.tolist(), "T": T.tolist()}


def oned_stack(n_groove, n_ridge, nsub, nsup, spacer, M, pol="te",
               theta=0.0):
    """The X-1 shape one level up: a 1-D PMMStack whose UNIFORM SPACER has
    exactly the groove index, and whose half-spaces do too."""
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(1.0e-6, n_substrate=nsub, n_superstrate=nsup, degree=12,
                  far_field_orders=2 * M + 1)
    if spacer:
        st.add_layer(0.15e-6, eps=complex(n_groove) ** 2)
    st.add_layer(0.5e-6, segments=[(0.5, complex(n_ridge) ** 2),
                                   (0.5, complex(n_groove) ** 2)])
    if spacer:
        st.add_layer(0.15e-6, eps=complex(n_groove) ** 2)
    st.set_source(WL, theta=theta)
    res = st.solve()
    R = np.asarray(res[1] if isinstance(res, tuple) else res.R)
    T = np.asarray(res[2] if isinstance(res, tuple) else res.T)
    tgt = 2.0 if R.ndim == 2 else 1.0
    return {"sumRT": float(R.sum() + T.sum()),
            "closure": float(R.sum() + T.sum() - tgt),
            "R": R.tolist(), "T": T.tolist()}


def oned_cell(n_groove, n_ridge, nsub, nsup, M, pol="te", theta=0.0):
    from lumenairy.elements.pmm import pmm_efficiency_1d
    o, R, T = pmm_efficiency_1d(1.0e-6, n_ridge, n_groove, nsub, nsup,
                                0.5e-6, 0.5, WL, theta=theta,
                                polarization=pol, far_field_orders=2 * M + 1,
                                stabilize=False)
    R, T = np.asarray(R), np.asarray(T)
    return {"sumRT": float(R.sum() + T.sum()),
            "closure": float(R.sum() + T.sum() - 1.0),
            "R": R.tolist(), "T": T.tolist()}


def run():
    payload = {"pure_stack": {}, "staggered_cell": {}, "oned": {}}

    # ---- PURE STAGGERED, the coincidence in every shape I can build -------
    cases = {
        # name                         bg    rel    nsub  nsup  spacer grids
        "pure_spacer_shared":       (2.25, 1e-6, 1.63, 1.0, True, "shared",
                                     0.0, 0.0, False, False),
        "pure_spacer_perlayer":     (2.25, 1e-6, 1.63, 1.0, True, "per-layer",
                                     0.0, 0.0, False, False),
        "pure_spacer_region_both":  (2.25, 1e-6, 1.5, 1.5, True, "shared",
                                     0.0, 0.0, False, False),
        "pure_spacer_conical":      (2.25, 1e-6, 1.63, 1.0, True, "shared",
                                     np.deg2rad(15.0), np.deg2rad(33.0),
                                     False, False),
        "pure_spacer_tensor":       (2.25, 1e-6, 1.63, 1.0, True, "shared",
                                     0.0, 0.0, True, False),
        "pure_spacer_perlayerwalls": (2.25, 1e-6, 1.63, 1.0, True,
                                      "per-layer", 0.0, 0.0, False, True),
        "pure_spacer_eps4":         (4.0, 2e-6, 1.71, 1.0, True, "shared",
                                     0.0, 0.0, False, False),
        "pure_nospacer_ctrl":       (2.25, 1e-6, 1.63, 1.0, False, "shared",
                                     0.0, 0.0, False, False),
        "pure_spacer_strong":       (2.25, 1e-1, 1.63, 1.0, True, "shared",
                                     0.0, 0.0, False, False),
    }
    for name, (bg, rel, nsub, nsup, sp, gr, th, ph, tn, pw) in cases.items():
        payload["pure_stack"][name] = _run(
            lambda bg=bg, rel=rel, nsub=nsub, nsup=nsup, sp=sp, gr=gr,
            th=th, ph=ph, tn=tn, pw=pw:
            pure_stack(bg, rel, nsub, nsup, sp, gr, th, ph, tensor=tn,
                       per_layer_walls=pw), name=name)

    # ---- staggered single cell, with and without the region coincidence ---
    scases = {
        "stag_region_coinc":  (2.25, 1e-6, 1.5, 1.5, 0.0, 0.0, False),
        "stag_off":           (2.25, 1e-6, 1.63, 1.0, 0.0, 0.0, False),
        "stag_region_jones":  (2.25, 1e-6, 1.5, 1.5, 0.0, 0.0, True),
        "stag_conical_jones": (2.25, 1e-6, 1.5, 1.5, np.deg2rad(15.0),
                               np.deg2rad(33.0), True),
    }
    for name, (bg, rel, nsub, nsup, th, ph, j) in scases.items():
        payload["staggered_cell"][name] = _run(
            lambda bg=bg, rel=rel, nsub=nsub, nsup=nsup, th=th, ph=ph, j=j:
            staggered_cell(bg, rel, nsub, nsup, th, ph, jones=j), name=name)

    # ---- 1-D PMM: the X-1 shape, with and without a coincident spacer -----
    ocases = {}
    for M in (6, 11, 19, 21, 28):
        ocases[f"oned_stack_spacer_M{M}"] = (1.5, 1.55, 1.5, 1.5, True, M)
        ocases[f"oned_stack_nospacer_M{M}"] = (1.5, 1.55, 1.5, 1.5, False, M)
    for name, (ng, nr, nsub, nsup, sp, M) in ocases.items():
        payload["oned"][name] = _run(
            lambda ng=ng, nr=nr, nsub=nsub, nsup=nsup, sp=sp, M=M:
            oned_stack(ng, nr, nsub, nsup, sp, M), name=name)
    for M in (6, 11, 19, 21, 28):
        for pol in ("te", "tm"):
            payload["oned"][f"oned_cell_{pol}_M{M}"] = _run(
                lambda M=M, pol=pol: oned_cell(1.5, 1.55, 1.5, 1.5, M, pol),
                name=f"oned_cell_{pol}_M{M}")

    VC.dump(OUT, payload)

    st = VC.stamp()
    print(f"ARM = {st['arm']}  ({st['n_sqrt_decay_definitions']} definitions)")
    for sect in ("pure_stack", "staggered_cell", "oned"):
        print(f"\n== {sect}")
        for name, row in payload[sect].items():
            sd = sum(v for k, v in (row.get("calls") or {}).items()
                     if k.endswith("_sqrt_decay"))
            fb = sum(v for k, v in (row.get("calls") or {}).items()
                     if k.endswith("_forward_branch_flip"))
            sf = sum(v for k, v in (row.get("calls") or {}).items()
                     if k.endswith("_select_forward_flux"))
            print(f"  {name:30s} closure "
                  f"{row.get('closure', float('nan')):13.5e}  sumRT "
                  f"{row.get('sumRT', float('nan')):12.8f}  "
                  f"sqrt_decay={sd} flip={fb} flux={sf} "
                  f"{row.get('warnings') or ''} {row.get('error') or ''}")


if __name__ == "__main__":
    run()
