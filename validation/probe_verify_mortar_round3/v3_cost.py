"""V3 -- what the round-3 residual screen COSTS, measured against the PRE-fix
tree on real operands of the generalized site.

Two stages.

``python v3_cost.py gen`` (this worktree only) captures one real generalized
mortar operand at each of five widths and writes them to
``C:/tmp/vmortar3_operands.npz`` (``/mnt/c/tmp/...`` from WSL), with a sha256
of each so the two TREES can be shown to be solving the SAME matrices.

``python v3_cost.py bench <tag> <root>`` loads them and times, INTERLEAVED,
minimum of N repetitions (a minimum is the only load-robust statistic on a
shared box):

* ``bare``      -- ``lu_factor`` + ``lu_solve``, no screen at all;
* ``guarded``   -- the tree's own ``_guarded_mortar_solve`` at its DEFAULT
  screen.  On the PRE tree that is round 2's ``gecon`` bar, which is what the
  generalized site actually paid before this round; on THIS tree the default
  is still ``gecon`` and the generalized site passes ``screen='residual'``;
* ``residual``  -- ``screen='residual'`` (this tree only): the shipped path;
* ``exact``     -- the same with the O(n^3) Frobenius residual forced, i.e.
  what the screen would cost if the probe were not there.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
NPZ = pathlib.Path("C:/tmp/vmortar3_operands.npz")
if not NPZ.parent.exists():
    NPZ = pathlib.Path("/mnt/c/tmp/vmortar3_operands.npz")


def _use_root(root):
    root = str(pathlib.Path(root).resolve())
    sys.path.insert(0, root)
    import lumenairy
    got = pathlib.Path(lumenairy.__file__).resolve()
    if pathlib.Path(root).resolve() not in got.parents:
        raise SystemExit(f"REFUSING: lumenairy resolved to {got}, not {root}")
    return lumenairy, root


def _sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def gen():
    sys.path.insert(0, str(HERE))
    import _path                            # noqa: F401,I001
    import _vfix as F
    from _capture import capture
    out, meta = {}, {}
    for M in (4, 5, 6, 7, 8):
        t0 = time.time()
        with capture() as recs:
            # BOTH-out-of-plane: healthy at every M under BOTH screens, so the
            # PRE tree's guarded path can actually be TIMED on it (the mixed
            # operand this round is about is REFUSED there from M = 5 up,
            # which is the defect, not a timing).
            F.build("ctrl_oop_both", M).solve(jones=False)
        r = recs[0]
        out[f"A{M}"], out[f"B{M}"] = r["A"], r["B"]
        meta[str(M)] = {"n": int(r["A"].shape[0]),
                        "sha_A": _sha(r["A"]), "sha_B": _sha(r["B"]),
                        "seconds": round(time.time() - t0, 2)}
        print(f"  M={M} n={r['A'].shape[0]} {meta[str(M)]['sha_A']} "
              f"{time.time() - t0:.1f}s", flush=True)
    np.savez(NPZ, **out)
    (HERE / "v3_operand_hashes_gen.json").write_text(
        json.dumps(meta, indent=1), encoding="cp1252")
    print(f"wrote {NPZ} and v3_operand_hashes_gen.json")


def bench(tag, root, reps=9):
    lum, root = _use_root(root)
    import scipy.linalg as sla              # noqa: I001
    from lumenairy.elements.pmm import _core as _pc
    z = np.load(NPZ)
    has_screen = "screen" in _pc._guarded_mortar_solve.__code__.co_varnames
    site = "pmm2d staggered GENERALIZED mortar interface"
    rows = []
    for M in (4, 5, 6, 7, 8):
        A, B = z[f"A{M}"], z[f"B{M}"]
        n = A.shape[0]
        def _gecon(A=A, B=B):
            """Round 2's screen, hand-assembled IDENTICALLY in both trees, so
            the two arms differ only in what this round added."""
            lu, piv = sla.lu_factor(A)
            anorm = float(np.max(np.sum(np.abs(A), axis=0)))
            gec = sla.get_lapack_funcs(("gecon",), (A,))[0]
            gec(lu, anorm, norm="1")
            return sla.lu_solve((lu, piv), B)

        arms = {"bare": lambda A=A, B=B: sla.lu_solve(sla.lu_factor(A), B),
                "gecon": _gecon,
                "guarded": lambda A=A, B=B: _pc._guarded_mortar_solve(
                    A, B, site)}
        if has_screen:
            arms["residual"] = lambda A=A, B=B: _pc._guarded_mortar_solve(
                A, B, site, screen="residual")

            def _exact(A=A, B=B):
                lu, piv = sla.lu_factor(A)
                anorm = float(np.max(np.sum(np.abs(A), axis=0)))
                gec = sla.get_lapack_funcs(("gecon",), (A,))[0]
                gec(lu, anorm, norm="1")
                X = sla.lu_solve((lu, piv), B)
                _pc._mortar_residual(A, X, B, probe=False)
                return X
            arms["exact"] = _exact
        best = {k: float("inf") for k in arms}
        for _ in range(reps):                    # INTERLEAVED, not blocked
            for k, fn in arms.items():
                t0 = time.perf_counter()
                fn()
                best[k] = min(best[k], time.perf_counter() - t0)
        row = {"M": M, "n": int(n), "sha_A": _sha(A), "sha_B": _sha(B),
               "reps": reps, "seconds": {k: best[k] for k in best},
               "ratio_vs_bare": {k: best[k] / best["bare"] for k in best},
               "ratio_vs_gecon": {k: best[k] / best["gecon"] for k in best}}
        rows.append(row)
        print(f"  n={n:5d} " + "  ".join(
            f"{k}={best[k]:.4f}s({best[k] / best['bare']:.3f}x)"
            for k in best), flush=True)
    import platform
    out = {"tag": tag, "root": root, "has_screen_kwarg": has_screen,
           "lumenairy": lum.__file__, "python": platform.python_version(),
           "numpy": np.__version__, "rows": rows}
    (HERE / f"v3_cost_{tag}.json").write_text(json.dumps(out, indent=1),
                                              encoding="cp1252")
    print(f"wrote v3_cost_{tag}.json")


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        gen()
    else:
        bench(sys.argv[2], sys.argv[3])
