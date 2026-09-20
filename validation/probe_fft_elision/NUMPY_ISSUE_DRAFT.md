# Draft NumPy issue -- complex128 `multiply` is not invariant under temporary elision on the manylinux wheel

Prepared 2026-09-15 for the maintainer to file upstream.  **Not filed by this work package.**
Everything below is measured; the reproducer is `numpy_elision_reproducer.py` in this directory.

## Title

`np.multiply` on complex128 gives different last bits when the RIGHT operand is an elidable
temporary (numpy 2.4.6, manylinux x86_64 wheel; not on the Windows wheel of the same version)

## Summary

NumPy's temporary elision (`temp_elide`, `can_elide_temp_unary` / the binop path in
`numpy/_core/src/multiarray/temp_elide.c`) rewrites `a * tmp` into an in-place operation on `tmp`
when `tmp` is an unreferenced, NumPy-owned temporary.  That rewrite is documented and intended as a
memory optimisation, i.e. it should be **numerically transparent**: `a * tmp` should give the same
bits as `out = a * b` with `b` named.

On the manylinux x86_64 wheel of numpy 2.4.6 it does not, for `complex128`, when the elided
temporary is the **right** operand.  The elided form differs from the named form in the last bits of
about one sixth of the doubles, for every array size from 128x128 up.  It also differs from the
explicit spelling the elision is supposed to be equivalent to (`np.multiply(a, b, out=b)`), which
DOES match the named form -- so the divergence is not "in-place complex multiply rounds
differently", it is specific to the elided path.

## Versions

| | affected | not affected | not affected |
|---|---|---|---|
| numpy | **2.4.6** | 2.4.6 | 2.4.4 |
| wheel | manylinux x86_64 (`scipy-openblas` 0.3.31.188.0, DYNAMIC_ARCH SkylakeX) | win_amd64 | win_amd64 |
| compiler (from `np.show_config`) | gcc 14.2.1 | msvc 19.44.35226 | msvc 19.44.35225 |
| python | 3.12.3 | 3.14.6 | 3.14.6 |
| SIMD baseline / found | X86_V2 / X86_V3 | X86_V2 / X86_V3 | X86_V2 / X86_V3 |

Host for all three: AMD Ryzen 9 5950X (Zen 3), Windows 11 with the Linux rows under WSL2.
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on every run.

The version axis is ruled out: numpy **2.4.6** installed into a clean Windows venv on the same box
prints the unaffected reading, and 2.4.4 and 2.4.6 on Windows agree with each other.  The
discriminator is the wheel/compiler, not the release.  The `SIMD Extensions` block is identical on
all three, so it is not a dispatch-level difference that `show_config` can see.

## Minimal code

```python
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
```

## Observed vs expected

**Expected** (all builds): `2.4.6 True 0.0` -- elision is a memory optimisation, so the two
spellings are the same computation.

**Observed**

| build | output |
|---|---|
| manylinux 2.4.6 / py3.12.3 | `2.4.6 False 1.7760122759024755e-16` |
| win_amd64 2.4.6 / py3.14.6 | `2.4.6 True 0.0` |
| win_amd64 2.4.4 / py3.14.6 | `2.4.4 True 0.0` |

Size dependence on the affected build (same script, `n` swept; `rel` is
`max|right-named| / max|named|`, `ndiff` counts differing float64 lanes):

| n | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| `right == named` | True | **False** | **False** | **False** | **False** |
| rel | 0.0 | 1.018e-16 | 9.651e-17 | 1.776e-16 | 1.787e-16 |
| ndiff / lanes | 0 / 8192 | 5426 / 32768 | 21698 / 131072 | 86794 / 524288 | 345396 / 2097152 |

n = 64 is below the elision size threshold (`NPY_MAX_NONZERO_...`/`temp_elide`'s array-size cut), so
the effect starting at 128 is consistent with elision being the trigger rather than the arithmetic.

## Which spellings agree (n = 512, affected build)

| spelling | equals `named` |
|---|---|
| `a * h` (both named) | -- (reference) |
| `(A * 1.0) * h` (LEFT operand an elidable temporary) | **True** |
| `a * np.exp(1j*P)` (RIGHT operand an elidable temporary) | **False** |
| `np.multiply(a, h2, out=h2)` (explicit in-place on the right) | **True** |
| `np.multiply(a, h3, out=o)` (explicit third array) | **True** |
| `np.multiply(a2, h, out=a2)` (explicit in-place on the left) | **True** |

The fourth row is the load-bearing one: the explicit in-place-on-the-right form -- what the elision
is meant to be a shorthand for -- matches the named form, while the elided form does not.  So this
is not a documented consequence of in-place complex arithmetic; the elided path is doing something
the explicit one is not.

Magnitudes are ~1 ULP of the array scale (`rel ~ 1e-16`).  Counted per element the ULP figure is
larger (up to ~2e5 ULP of the individual result) because `(ac - bd)` cancels for some entries; that
is expected amplification of a last-bit input difference, not an independent defect.

## Why it matters downstream

The two spellings are chosen by whether the left operand is NumPy-owned, which a library cannot
always control: an array returned as a non-owning view (an aligned FFT workspace, a `.T`, a slice)
is never elidable, so `f(x) * np.exp(...)` takes the right-elided path, while the same expression
with an owning `f(x)` takes the left-elided path.  Callers then see byte differences that track an
allocation detail rather than their inputs.  That is how this was found (a pyFFTW plan workspace
handed back as a view vs as a copy).

## What would close it

Either the elided binop path is made bit-identical to `np.multiply(a, b, out=b)` on this build, or
-- if some divergence is by design -- the elision documentation states that it is not required to be
bit-identical, so downstream projects can stop treating the two spellings as the same computation.
