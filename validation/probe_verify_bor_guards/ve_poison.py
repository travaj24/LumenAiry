"""TASK E -- does ONE non-finite eigenvalue change the branch decision of the
FINITE part of the spectrum?  ``cut_band`` takes ``max|z|`` over the WHOLE
spectrum, so an inf or nan anywhere sets the band for every mode."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import _vh  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True)
a = ap.parse_args()
import lumenairy

print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
from lumenairy.elements.eme import eme_2d

clean = np.array([100.0 - 0.5j, 50.0 - 1e-3j, 20.0 + 0.1j, 200.0 + 0j])
for name, extra in (("clean", None), ("plus_inf", np.inf + 0j),
                    ("plus_nan", np.nan + 0j), ("plus_1e300", 1e300 + 0j)):
    z = clean if extra is None else np.concatenate([clean, [extra]])
    with np.errstate(all="ignore"):
        got = np.asarray(eme_2d._ky_forward(z ** 2, 0.0))
    print("  %-10s finite part -> %s" % (name, np.round(got[:4], 6).tolist()))
