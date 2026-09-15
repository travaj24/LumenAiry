"""VERIFY-B11c, the hand-off red: BOUND the 4-ULP bar in
``test_audit2609_a4_verify_maslov_asymptotic.py::test_s10_vector_normalisation_is_one_joint_scale_for_the_pair``.

The bar is ``abs(ratios[mode] - r0) <= 4 * np.spacing(r0)`` on ``P_x/P_y``,
where each power is a 9216-term reduction of ``|E|**2``.  The WP-B11c report
re-measured 3 ULP for ``'power'`` and 1 for ``'peak'`` against a docstring that
records 0 and 1, and handed the re-derivation to the file's owner.  The
question a bar has to answer (``docs/TESTING_STANDARDS.md`` S4 / restatement 5)
is whether its pass/fail boundary sits inside the SPREAD of the quantity it
reads.  One reading cannot answer that; a family of readings can.

So this runs the test's own quantity over a FAMILY of fixtures the test is
entitled to be run on -- the exact one, then neighbours that vary only the
amplitude pair, the grid size, the quadrature node count and the polynomial
order -- and reports the ULP delta for each.  The spread across the family is
the envelope the bar needs a gap against, above and below.

It also runs the exact fixture with ``jax`` imported and 64-bit enabled first,
which is what the two files that run before it in the ten-file slice do.

argv: <tree-root> <output-json> [--jax-first]
"""
# ruff: noqa: E402, I001 -- the tree is bound before the library is imported.
from __future__ import annotations

import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
JAX_FIRST = "--jax-first" in sys.argv
sys.path.insert(0, TREE)

if JAX_FIRST:
    import jax
    jax.config.update("jax_enable_x64", True)

import vlib

la = vlib.anchor(TREE)

import numpy as np

from lumenairy.elements.lenses_maslov import apply_real_lens_maslov_vector

PRESC = {"surfaces": [
    {"radius": 8e-3, "conic": 0.0, "glass_before": "air",
     "glass_after": "N-BK7"},
    {"radius": -8e-3, "conic": 0.0, "glass_before": "N-BK7",
     "glass_after": "air"}],
    "thicknesses": [2.5e-3], "aperture_diameter": 2.4e-3}


def _fixture(N=96, dxg=16e-6, ax=0.8, ay=0.6, w=0.55e-3, n_v2=48, order=4,
             wl=1.55e-6):
    xs = (np.arange(N) - N // 2) * dxg
    X, Y = np.meshgrid(xs, xs)
    amp = np.exp(-(X ** 2 + Y ** 2) / w ** 2)
    E = np.stack([(ax * amp).astype(np.complex128),
                  (ay * amp).astype(np.complex128)], axis=0)
    return E, dict(prescription=PRESC, wavelength=wl, dx=dxg,
                   integration_method="quadrature", n_v2=n_v2,
                   poly_order=order)


def _ulps(E, kw):
    ratios = {}
    for mode in ("none", "power", "peak"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = apply_real_lens_maslov_vector(E.copy(),
                                                normalize_output=mode, **kw)
        px = float(np.sum(np.abs(out[0]) ** 2))
        py = float(np.sum(np.abs(out[1]) ** 2))
        ratios[mode] = px / py
    r0 = ratios["none"]
    sp = float(np.spacing(r0))
    return {"r0": repr(r0),
            "power_ulp": abs(ratios["power"] - r0) / sp,
            "peak_ulp": abs(ratios["peak"] - r0) / sp}


CASES = [
    ("exact_test_fixture", {}),
    ("amp_0.9_0.5", dict(ax=0.9, ay=0.5)),
    ("amp_0.6_0.8", dict(ax=0.6, ay=0.8)),
    ("amp_0.99_0.14", dict(ax=0.99, ay=0.14)),
    ("N_112", dict(N=112)),
    ("N_128", dict(N=128)),
    ("N_80", dict(N=80)),
    ("n_v2_40", dict(n_v2=40)),
    ("n_v2_56", dict(n_v2=56)),
    ("poly_order_5", dict(order=5)),
    ("waist_0.50mm", dict(w=0.50e-3)),
    ("waist_0.60mm", dict(w=0.60e-3)),
]

rows = {}
for name, kwargs in CASES:
    E, kw = _fixture(**kwargs)
    try:
        rows[name] = _ulps(E, kw)
    except Exception as exc:                     # noqa: BLE001 -- recorded
        rows[name] = {"error": f"{type(exc).__name__}: {exc}"}

ok = [r for r in rows.values() if "error" not in r]
summary = {
    "tree": TREE,
    "jax_first": JAX_FIRST,
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "bar_ulp": 4.0,
    "n_cases": len(rows),
    "power_ulp_min": min(r["power_ulp"] for r in ok) if ok else None,
    "power_ulp_max": max(r["power_ulp"] for r in ok) if ok else None,
    "peak_ulp_min": min(r["peak_ulp"] for r in ok) if ok else None,
    "peak_ulp_max": max(r["peak_ulp"] for r in ok) if ok else None,
    "cases_over_bar": sorted(n for n, r in rows.items()
                             if "error" not in r
                             and max(r["power_ulp"], r["peak_ulp"]) > 4.0),
    "rows": rows,
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
