"""VERIFY-B11c: the OBSERVABLE METADATA surface of every moved name.

The bit-identity probes answer "does the same call return the same bytes?".
This one answers the question a module split can fail without moving a byte:
what does INTROSPECTION see?  ``__module__`` and ``__qualname__`` are what
``pickle`` writes into its payload, what a traceback prints, what Sphinx
resolves and what ``inspect.getmodule`` returns.  A refactor whose stated
contract is "nothing observable moves" owes an answer here too.

Recorded, not asserted: the driver compares the two trees and the differences
are read in the report.  Run exactly like the bit-identity probes.

argv: <tree-root> <output-json>
"""
# ruff: noqa: E402, I001 -- the tree is bound before the library is imported.
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
sys.path.insert(0, TREE)

import vlib

la = vlib.anchor(TREE)

P = vlib.Probe()

MOVED_LENS = ("surface_sag_general", "_surface_sag_general",
              "surface_sag_biconic", "_fit_normaliser",
              "_multi_indices_total_degree", "_ensure_numexpr_loaded",
              "_ensure_cupy_loaded", "_is_cupy_array", "_load_numba",
              "_get_aspheric_sag_accum_numba", "check_grid_vs_apertures",
              "recommend_grid_for_prescription", "_collect_semi_diameters",
              "_warn_if_aperture_exceeds_grid")
MOVED_BLAS = ("set_blas_threads", "rcwa_blas_threads", "_get_blas_threads",
              "_blas_threads_quiet", "_blas_limit", "_with_blas_limit")


def _meta(obj):
    return (getattr(obj, "__module__", None),
            getattr(obj, "__qualname__", None),
            type(obj).__name__)


from lumenairy.elements import lenses as LE                        # noqa: E402
from lumenairy.elements.rcwa import _core as _core                 # noqa: E402

for _n in MOVED_LENS:
    P.add(f"L_{_n}", _meta(getattr(LE, _n, None)))
for _n in MOVED_BLAS:
    P.add(f"B_{_n}", _meta(getattr(_core, _n, None)))

# the public spellings a user reaches
for _n in ("surface_sag_general", "surface_sag_biconic", "set_blas_threads",
           "rcwa_blas_threads"):
    P.add(f"T_{_n}", _meta(getattr(la, _n, None)))

# what a traceback / Sphinx would report for the module objects themselves
# The AttributeError a typo produces: at the base commit CPython raised it for
# a plain module, here the facade raises it by hand, so the MESSAGE is part of
# the observable surface.  Two shapes: a name close to a real one (where the
# interpreter's own message machinery could differ) and a name close to none.
for _bad in ("surface_sag_generl", "zzz_no_such_name", "_NUMBA_AVAILABL"):
    P.call(f"E_attr_{_bad}", lambda n=_bad: getattr(LE, n))

P.add("M_lenses_type", type(LE).__name__)
P.add("M_lenses_file_tail", os.path.basename(LE.__file__))
P.add("M_lenses_name", LE.__name__)

P.write(OUT)
