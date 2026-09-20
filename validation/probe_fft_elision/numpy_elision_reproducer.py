"""Minimal, lumenairy-free reproducer: a complex128 multiply is not invariant
under NumPy's temporary elision on some builds.  Prints False where it moves.

Run it as a script; no arguments, no third-party imports beyond NumPy.
"""
import numpy as np

n = 512
rng = np.random.default_rng(5)
A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
P = rng.standard_normal((n, n)) * 1e6
a, h = A * 1.0, np.exp(1j * P)      # both operands NAMED
named = a * h                       # nothing elidable
right = a * np.exp(1j * P)          # RIGHT operand an elidable temporary
print(np.__version__, np.array_equal(right, named),
      np.max(np.abs(right - named)) / np.max(np.abs(named)))
