"""TASK E -- the DEGENERATE inputs of the three call sites, where 'cut_band
computes the identical quantity this line always did' can be checked exactly."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _vh  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True)
a = ap.parse_args()
import lumenairy

print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
from lumenairy.elements.eme import eme_2d
from lumenairy.elements.eme import eme_2d_vector as ev

CASES = {
    "empty": np.array([], dtype=complex),
    "one_real": np.array([3.0 + 0j]),
    "all_tiny": np.array([1e-14 + 0j, -1e-14 + 0j, 1e-14j, -1e-14j]),
    "nan": np.array([np.nan + 0j, 1.0 + 0j]),
    "inf": np.array([np.inf + 0j, 1.0 + 0j]),
    "negzero_imag": np.array([2.0 - 0.0j, 2.0 + 0.0j]),
}
for name, ky in CASES.items():
    try:
        idx = ev._strip_split_forward(ky)
        r = "idx=%s" % (list(map(int, idx)),)
    except Exception as e:  # noqa: BLE001
        r = "RAISES %s: %s" % (type(e).__name__, e)
    print("  _strip_split_forward[%-13s] %s" % (name, r))
print()
for name, z in (("empty", np.array([], dtype=complex)),
                ("negzero", np.array([4.0 - 0.0j])),
                ("scalar0d", np.array(4.0 - 1e-18j)),
                ("nan", np.array([np.nan - 1j, 4.0 + 0j])),
                ("inf", np.array([np.inf - 1j, 4.0 + 0j]))):
    try:
        got = eme_2d._ky_forward(z ** 2, 0.0)
        r = "-> %s" % (np.asarray(got).ravel().tolist(),)
    except Exception as e:  # noqa: BLE001
        r = "RAISES %s: %s" % (type(e).__name__, e)
    print("  _ky_forward[%-9s] %s" % (name, r))
