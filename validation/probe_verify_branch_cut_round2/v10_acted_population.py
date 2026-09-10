"""Is the ACTED-ON population really empty for a "lossy" stack?

The round-2 file's gate 6 says: "For a lossy layer the acted-on population is
EMPTY -- every mode's ``Im(lam^2)`` takes the sign the loss dictates -- so the
flip has nothing to act on."  That is a statement about the EIGENPROBLEM, so it
can be counted directly and is identical on both arms of the change.  Counted
here for loss placed in the CELL and in the SPACER separately.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

sys.path.insert(0, str(VC.TREE / "tests" / "unit"))
import test_verify_branch_cut_round2 as T  # noqa: E402

import lumenairy.elements.rcwa._core as rc  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "v10.json"


def count(**kw):
    """Modes the shipped band CONJUGATES, split by whether the array came from
    an eigensolve (a structured layer) or from the analytic Rayleigh helper (a
    region or uniform layer, whose Im is exactly zero)."""
    n_layer = n_exact = n_arrays = 0
    saved = rc._sqrt_decay

    def tap(x, xp=None, band=rc._CUT_BAND_REL):
        nonlocal n_layer, n_exact, n_arrays
        xx = np.asarray(x).astype(complex)
        if xx.size:
            n_arrays += 1
            r = np.sqrt(xx)
            mx = max(float(np.max(np.abs(r))), 1.0)
            acted = int(np.sum((np.abs(r.real) <= band * mx) & (r.imag < 0)))
            if np.all(xx.imag == 0.0):
                n_exact += acted
            else:
                n_layer += acted
        return saved(x, xp, band) if xp is not None else saved(x, band=band)

    import importlib

    import lumenairy.elements as EL
    base = Path(EL.__file__).parent
    patched = []
    for p in sorted(base.rglob("*.py")):
        rel = p.relative_to(base).with_suffix("")
        name = "lumenairy.elements." + ".".join(rel.parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if callable(getattr(mod, "_sqrt_decay", None)):
            patched.append((mod, mod._sqrt_decay))
            mod._sqrt_decay = tap
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            T._stack(**kw)
    finally:
        for mod, fn in patched:
            mod._sqrt_decay = fn
    return {"n_arrays": n_arrays, "acted_on_layer_modes": n_layer,
            "acted_on_exact_modes": n_exact}


def run():
    p = {}
    for name, kw in {
            "lossless": dict(),
            "lossy_cell_1e-2": dict(cell_loss=1e-2),
            "lossy_cell_1e-6": dict(cell_loss=1e-6),
            "lossy_spacer_1e-2": dict(spacer_loss=1e-2),
            "lossy_spacer_1e-6": dict(spacer_loss=1e-6),
            "lossy_both_1e-2": dict(cell_loss=1e-2, spacer_loss=1e-2),
            "lossy_cell_conical": dict(cell_loss=1e-2, theta=0.21, phi=0.35),
            "lossy_spacer_conical": dict(spacer_loss=1e-2, theta=0.21,
                                         phi=0.35),
    }.items():
        for M in (3, 4, 5):
            p[f"{name}_M{M}"] = count(n_orders=M, **kw)
    VC.dump(OUT, p)
    st = VC.stamp()
    print(f"BUILD py {st['python']} numpy {st['numpy']}")
    for k, v in p.items():
        print(f"  {k:26s} arrays {v['n_arrays']:3d}  acted-on LAYER modes "
              f"{v['acted_on_layer_modes']:3d}  acted-on EXACT modes "
              f"{v['acted_on_exact_modes']:3d}")


if __name__ == "__main__":
    run()
