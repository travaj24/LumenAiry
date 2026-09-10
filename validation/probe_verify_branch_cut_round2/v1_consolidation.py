"""TASK 1 -- consolidation.

(a) AST census: how many ``_sqrt_decay`` definitions exist in the library, and
    which modules CALL one, resolved from the source rather than from a grep of
    the word.
(b) Bit-identity of the shipped shared body against a TRANSCRIPTION of the
    round-1 ``rcwa/_core`` body, on MY OWN engineered value sets -- denormals,
    signed zeros, NaN/Inf, 1e300, and arrays whose ``max|r|`` is dominated by a
    single evanescent mode (the case the array-relative scale makes
    interesting).
(c) The JAX twins: ``jax.jit`` the shared body and each twin public surface,
    compare to the NumPy path.

Run:  PYTHONPATH=. python validation/probe_verify_branch_cut_round2/v1_consolidation.py out.json
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vcommon as VC  # noqa: E402

VC.pin_tree()

OUT = sys.argv[1] if len(sys.argv) > 1 else "v1.json"
TREE = VC.TREE
ELEMENTS = TREE / "lumenairy" / "elements"


# ---------------------------------------------------------------- (a) AST ---
def ast_census():
    defs, callers, importers, exact_zero_lines = [], [], [], []
    for p in sorted(ELEMENTS.rglob("*.py")):
        src = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(TREE)).replace("\\", "/")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == "_sqrt_decay":
                defs.append({"file": rel, "line": node.lineno,
                             "args": [a.arg for a in node.args.args]})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "_sqrt_decay":
                callers.append({"file": rel, "line": node.lineno,
                                "n_args": len(node.args),
                                "second_arg": (ast.unparse(node.args[1])
                                               if len(node.args) > 1 else None)})
            if isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == "_sqrt_decay":
                        importers.append({"file": rel, "line": node.lineno,
                                          "from": node.module})
            # an EXACT-ZERO branch pin on a real/imag part, anywhere
            if isinstance(node, ast.Compare) and len(node.ops) == 1 \
                    and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
                txt = ast.unparse(node)
                if (".real" in txt or ".imag" in txt) and \
                        ("== 0" in txt or "!= 0" in txt):
                    exact_zero_lines.append({"file": rel, "line": node.lineno,
                                             "src": txt})
    return {"definitions": defs, "call_sites": callers,
            "importers": importers, "exact_zero_pins": exact_zero_lines}


# ------------------------------------------------ (b) refactor bit-identity --
_C = np.complex128


def round1_body(x):
    """Transcription of the ROUND-1 ``rcwa/_core._sqrt_decay``, read out of the
    pre-round-2 tree at 2898767 and reproduced here so the comparison does not
    depend on that tree being mounted."""
    x = np.asarray(x).astype(_C)
    r = np.sqrt(x)
    scale = max(np.max(np.abs(r)), 1.0) if r.size else 1.0
    on_cut = np.abs(r.real) <= 1e-8 * scale
    return np.where(on_cut & (r.imag < 0), np.conj(r), r)


def engineered_values(rng):
    corners = [
        0 + 0j, -0.0 + 0j, 0 - 0.0j, -0.0 - 0.0j,
        5e-324, -5e-324, 5e-324 * 1j, -5e-324 * 1j,
        1e-300, -1e-300, 1e-300 - 1e-320j, -1e-300 + 1e-320j,
        -2.25 + 0j, -2.25 - 0.0j, -2.25 + 2.911e-15j, -2.25 - 2.911e-15j,
        +2.25 - 2.911e-15j, +2.25 + 2.911e-15j,
        np.nan, np.nan * 1j, complex(np.nan, np.nan),
        np.inf, -np.inf, complex(np.inf, np.inf), complex(-np.inf, 1.0),
        1e300 + 1e300j, -1e300 - 1e300j, 1e300 - 1e-300j,
        -1e-8 + 1e-30j, -1e-16 + 1e-32j, -1e8 - 1e-8j,
        1.0 + 0j, -1.0 + 0j, -1.0 - 0.0j,
        # exactly AT the band on a unit spectrum
        -1.0 + 2e-8j, -1.0 - 2e-8j, -1.0 - 2.0000001e-8j,
    ]
    mag = 10.0 ** rng.uniform(-15, 15, 4000)
    ph = rng.uniform(-np.pi, np.pi, 4000)
    rand = mag * np.exp(1j * ph)
    return np.concatenate([np.asarray(corners, dtype=_C), rand])


def dominated_arrays(rng):
    """Arrays whose ``max|r|`` is set by ONE huge evanescent mode, so the
    array-relative band is wide in absolute terms.  This is the shape a real
    layer spectrum has and the shape that makes the SCALE load-bearing."""
    out = {}
    for name, big in (("dom_1e4", 1e8), ("dom_1e8", 1e16),
                      ("dom_1e12", 1e24)):
        prop = -(10.0 ** rng.uniform(-3, 1, 31))          # lossless propagating
        eta = rng.normal(0, 1, 31) * 1e-16 * np.abs(prop)
        arr = prop + 1j * eta
        arr = np.concatenate([arr, [big + 0j]])            # the dominator
        out[name] = arr
    # a spectrum whose top is BELOW 1 (floor 1.0 engaged)
    out["subunit"] = np.asarray(
        [-1e-6 + 1e-22j, -4e-6 - 3e-22j, 9e-7 + 0j, -2.5e-7 - 1e-23j], dtype=_C)
    return out


def bit_identity():
    from lumenairy.elements.rcwa._core import _sqrt_decay
    rng = np.random.default_rng(20260910)
    vals = engineered_values(rng)
    rows = []
    for size in (1, 2, 3, 7, 64, 243, 1024, len(vals)):
        v = vals[:size] if size <= len(vals) else vals
        a = _sqrt_decay(v)
        b = round1_body(v)
        same = np.array_equal(a.view(np.float64), b.view(np.float64),
                              equal_nan=True)
        d = np.abs(a - b)
        d = d[np.isfinite(d)]
        rows.append({"size": int(len(v)), "bit_identical": bool(same),
                     "max_abs_diff": float(d.max()) if d.size else 0.0})
    dom = {}
    for name, arr in dominated_arrays(rng).items():
        a = _sqrt_decay(arr)
        b = round1_body(arr)
        same = np.array_equal(a.view(np.float64), b.view(np.float64),
                              equal_nan=True)
        n_flipped = int(np.sum(a != np.sqrt(arr.astype(_C))))
        dom[name] = {"n": int(arr.size), "bit_identical": bool(same),
                     "n_conjugated": n_flipped,
                     "max_abs_diff": float(np.nanmax(np.abs(a - b)))}
    allv = _sqrt_decay(vals)
    fin = np.isfinite(allv)
    return {"sizes": rows, "dominated": dom,
            "n_values": int(len(vals)),
            "min_Re": float(np.min(allv.real[fin])),
            "n_negative_Re": int(np.sum(allv.real[fin] < 0))}


# --------------------------------------------------------------- (c) JAX ----
def jax_trace():
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": repr(exc)}
    from lumenairy.elements.rcwa._core import _sqrt_decay
    rng = np.random.default_rng(7)
    vals = engineered_values(rng)
    # For the PARITY claim the population must be one both backends can even
    # represent: XLA FLUSHES SUBNORMALS TO ZERO and numpy's own
    # ``sqrt(0 + 5e-324j)`` overflows to ``inf``, so a set containing either
    # measures the backends' denormal handling, not this function.  Those
    # cases are reported separately in ``scale_contamination``.
    finite = vals[np.isfinite(vals) & (np.abs(vals) > 1e-290)
                  & (np.abs(vals) < 1e290)]

    f = jax.jit(lambda x: _sqrt_decay(x, jnp))
    g = jax.jit(_sqrt_decay)          # xp=None -> array_namespace on a TRACER
    jv = jnp.asarray(finite)
    a = np.asarray(f(jv))
    b = np.asarray(g(jv))
    n = _sqrt_decay(finite)
    ok = np.isfinite(n) & (np.abs(n) > 0)
    rel = np.abs(a[ok] - n[ok]) / np.abs(n[ok])
    # did the two paths make the SAME branch decision on every value?
    same_branch = np.sign(a.imag) == np.sign(n.imag)
    # gradient through the shared body, both call shapes
    def loss(scale, xp_explicit):
        v = jnp.asarray(finite.real * scale + 1j * finite.imag)
        r = _sqrt_decay(v, jnp) if xp_explicit else _sqrt_decay(v)
        return jnp.sum(jnp.abs(r[jnp.isfinite(r)]))
    try:
        g1 = float(jax.grad(lambda s: loss(s, True))(1.0))
        g2 = float(jax.grad(lambda s: loss(s, False))(1.0))
    except Exception as exc:
        g1 = g2 = repr(exc)
    return {
        "available": True,
        "jax_version": jax.__version__,
        "n_values": int(finite.size),
        "jit_explicit_jnp_vs_numpy_max_rel": float(rel.max()),
        "jit_autodetect_matches_explicit_bitwise": bool(np.array_equal(a, b)),
        "n_branch_disagreements_jax_vs_numpy": int(np.sum(~same_branch)),
        "n_conjugated_numpy": int(np.sum(n.imag != np.sqrt(
            finite.astype(_C)).imag)),
        "grad_explicit_jnp": g1,
        "grad_autodetect": g2,
    }


def scale_contamination():
    """The band scale is ``max(max|r|, 1)`` over the WHOLE array, so ONE
    non-finite root decides the verdict on every other mode of that layer.
    Measured here as a property of the shipped body (round 1 and round 2 share
    it bit for bit -- this is NOT a round-2 regression); the question is only
    whether a SOLVE can reach it."""
    from lumenairy.elements.rcwa._core import _sqrt_decay
    rng = np.random.default_rng(11)
    prop = -(10.0 ** rng.uniform(-2, 1, 24))
    eta = rng.normal(0, 1, 24) * 1e-16 * np.abs(prop)
    clean = (prop + 1j * eta).astype(_C)
    base = _sqrt_decay(clean)
    n_base = int(np.sum(base.imag != np.sqrt(clean).imag))
    rows = {}
    for name, extra in (("clean", None), ("plus_nan", np.nan),
                        ("plus_inf", np.inf),
                        ("plus_1e300", 1e300 + 0j),
                        ("plus_huge_denormal_imag", 5e-324j)):
        arr = clean if extra is None else np.concatenate(
            [clean, np.asarray([extra], dtype=_C)])
        out = _sqrt_decay(arr)
        r = np.sqrt(arr)
        scale = np.max(np.abs(r))
        rows[name] = {
            "scale": float(scale) if np.isfinite(scale) else str(scale),
            "n_conjugated_of_the_24_clean_modes":
                int(np.sum(out[:24].imag != np.sqrt(clean).imag)),
        }
    return {"n_conjugated_clean": n_base, "rows": rows}


WL = 0.5321e-6
PX, PY, DEPTH = 0.6e-6, 0.55e-6, 0.20e-6


def jax_twins():
    """Each JAX twin public surface against its NumPy sibling on a LOSSLESS
    COINCIDENCE fixture (a weakly modulated eps=2.25 cell, superstrate 1.5 so
    the region coincides).  The JAX path is entered by handing the surface a
    TRACED ``eps_cell`` (the library dispatches on ``is_jax_array``), which is
    why a concrete ``region_layout`` accompanies it."""
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell, pmm_jones_2d
    cell = VC.weak_cell(6, 6, 2.25, 1e-6)
    layout = np.zeros(cell.shape, dtype=int)
    layout[:2, :2] = 1
    out = {"available": True, "jax_version": jax.__version__}

    def eff(traced):
        ec = jnp.asarray(cell) if traced else cell
        o, R, T = pmm_efficiency_2d_cell(
            PX, PY, ec, 1.63, 1.5, DEPTH, WL, n_orders=3, degree=7,
            symmetry=False, region_layout=layout)
        return np.asarray(R), np.asarray(T)

    for name, fn, tgt in (("pmm_efficiency_2d_cell", eff, 1.0),):
        try:
            Rn, Tn = fn(False)
            Rj, Tj = fn(True)
            out[name] = {
                "max_abs_dR": float(np.max(np.abs(Rn - Rj))),
                "max_abs_dT": float(np.max(np.abs(Tn - Tj))),
                "closure_numpy": float(Rn.sum() + Tn.sum() - tgt),
                "closure_jax": float(Rj.sum() + Tj.sum() - tgt),
            }
        except Exception as exc:
            out[name] = {"error": repr(exc)}

    tens = np.zeros(cell.shape + (3, 3), dtype=complex)
    for i in range(3):
        tens[..., i, i] = cell

    def jones(traced):
        tc = jnp.asarray(tens) if traced else tens
        res = pmm_jones_2d(PX, PY, tc, 1.63, 1.5, DEPTH, WL, n_orders=3,
                           degree=7, region_layout=layout, symmetry=False)
        R = getattr(res, "R", None)
        if R is None:
            R, T = res[1], res[2]
        else:
            T = res.T
        return np.asarray(R), np.asarray(T)

    try:
        Rn, Tn = jones(False)
        Rj, Tj = jones(True)
        out["pmm_jones_2d"] = {
            "max_abs_dR": float(np.max(np.abs(Rn - Rj))),
            "max_abs_dT": float(np.max(np.abs(Tn - Tj))),
            "closure_numpy": float(Rn.sum() + Tn.sum() - 2.0),
            "closure_jax": float(Rj.sum() + Tj.sum() - 2.0),
        }
    except Exception as exc:
        out["pmm_jones_2d"] = {"error": repr(exc)}

    def stack(traced):
        ec = jnp.asarray(cell) if traced else cell
        st = PMM2DStackHybrid(PX, PY, n_superstrate=1.0, n_substrate=1.63,
                              degree=7, n_orders=3, symmetry=False)
        st.add_layer(0.10e-6, eps=2.25)
        if traced:
            st.add_layer(DEPTH, eps_cell=ec, region_layout=layout)
        else:
            st.add_layer(DEPTH, eps_cell=ec)
        st.add_layer(0.10e-6, eps=2.25)
        st.set_source(WL, theta=0.0, phi=0.0)
        r = st.solve()
        if hasattr(r, "R"):
            return np.asarray(r.R), np.asarray(r.T)
        return np.asarray(r[1]), np.asarray(r[2])

    try:
        Rn, Tn = stack(False)
        Rj, Tj = stack(True)
        out["PMM2DStackHybrid"] = {
            "max_abs_dR": float(np.max(np.abs(Rn - Rj))),
            "max_abs_dT": float(np.max(np.abs(Tn - Tj))),
            "closure_numpy": float(Rn.sum() + Tn.sum() - 2.0),
            "closure_jax": float(Rj.sum() + Tj.sum() - 2.0),
        }
    except Exception as exc:
        out["PMM2DStackHybrid"] = {"error": repr(exc)}

    # jax.grad through the traced twin, on the same fixture
    try:
        def loss(scale):
            ec = jnp.asarray(cell) * scale
            o, R, T = pmm_efficiency_2d_cell(
                PX, PY, ec, 1.63, 1.5, DEPTH, WL, n_orders=3, degree=7,
                symmetry=False, region_layout=layout)
            return jnp.sum(R)
        out["grad_dRdscale"] = float(jax.grad(loss)(1.0))
    except Exception as exc:
        out["grad_dRdscale"] = repr(exc)
    return out


if __name__ == "__main__":
    payload = {"ast": ast_census(), "bit_identity": bit_identity(),
               "scale_contamination": scale_contamination(),
               "jax_body": jax_trace(), "jax_twins": jax_twins()}
    VC.dump(OUT, payload)
    a = payload["ast"]
    print(f"definitions       : {len(a['definitions'])}  {a['definitions']}")
    print(f"call sites        : {len(a['call_sites'])}")
    for c in a["call_sites"]:
        print(f"    {c['file']}:{c['line']}  n_args={c['n_args']} "
              f"xp={c['second_arg']}")
    print(f"importers         : {len(a['importers'])}")
    print(f"exact-zero pins   : {a['exact_zero_pins']}")
    b = payload["bit_identity"]
    print(f"bit-identical     : {all(r['bit_identical'] for r in b['sizes'])} "
          f"max|diff| = {max(r['max_abs_diff'] for r in b['sizes']):.3e}")
    print(f"dominated arrays  : {b['dominated']}")
    print(f"min Re(lam)       : {b['min_Re']}   n(Re<0) = {b['n_negative_Re']}")
    print(f"scale contam      : {payload['scale_contamination']}")
    print(f"jax body          : {payload['jax_body']}")
    print(f"jax twins         : {payload['jax_twins']}")
