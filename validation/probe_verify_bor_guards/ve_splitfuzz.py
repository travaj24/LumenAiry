"""TASK E / claim (a) -- "cut_band computes the identical quantity this line
always did", fuzzed.  20000 random spectra spanning 14 decades of scale,
including sub-unit ones (where the 1.0 floor is what decides) and spectra with
values EXACTLY on the band, through eme_2d_vector._strip_split_forward on both
trees.  One hash over every index set."""
from __future__ import annotations
import argparse, hashlib, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import _vh  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--build", required=True)
a = ap.parse_args()
import lumenairy
print("lumenairy.__file__ =", lumenairy.__file__)
_vh.require_tree(a.build)
from lumenairy.elements.eme import eme_2d_vector as ev

rng = np.random.default_rng(20260912)
h = hashlib.sha256()
n_cases = 0
for trial in range(20000):
    n = int(rng.integers(1, 40))
    scale = 10.0 ** rng.uniform(-14, 14)
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * scale
    # push a third of the imaginary parts onto / near the band
    m = max(1.0, float(np.max(np.abs(z))))
    tol = 1e-9 * m
    k = rng.integers(0, n, size=max(1, n // 3))
    z[k] = z[k].real + 1j * tol * rng.choice([-1.0, 1.0, 0.0, 0.5, -0.5,
                                              1.0 + 1e-15, 1.0 - 1e-15],
                                             size=k.size)
    idx = ev._strip_split_forward(z)
    h.update(np.ascontiguousarray(np.asarray(idx, dtype=np.int64)).tobytes())
    h.update(b"|")
    n_cases += 1
print("  %d fuzz cases, index-set hash = %s" % (n_cases, h.hexdigest()))
