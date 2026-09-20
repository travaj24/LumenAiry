# Wave 5 hygiene, part 2 -- the WP-B11 items never reached, and one durability defect

Scope: the three items `HANDOFF_2026_09_14.md` section 4.5 lists as not started
(item 14, the direct-matrix MFT branch; item 18, `_collins_transport` on JAX;
item 20's near-focus exact-kernel table), plus VERIFY-WP-B11c defect D3.

Four packages, labelled H2-1 .. H2-4 below.  Two of them ship code, one ships
code and a refusal, one ships only a measurement and its tests.  No default
moves in any of them.

## How every claim here was gated

* **Archive-to-archive, never worktree-to-worktree.**  Both arms of every
  bit-identity claim are read-only `git archive` extractions -- the base commit
  `f4f18851` into `C:/tmp/hyg2_arch/base` and this branch's tree object into
  `C:/tmp/hyg2_arch/branch`.  The probe runs in a child process whose `cwd` and
  `PYTHONPATH` name one tree, and `hlib.anchor` refuses to continue unless
  `lumenairy.__file__` resolves under it and prints what bound.  pytest is never
  involved, because pytest puts the repository root ahead of `PYTHONPATH` and
  both arms would then import the same tree.
* **Both builds, every time.**  Windows py3.14 (numpy 2.4.4, scipy 1.17.1,
  jax 0.11.0, CuPy 14.0.1 with a device) and WSL py3.12 (numpy 2.4.6 on
  scipy-openblas 0.3.31 SkylakeX, scipy 1.17.1, jax 0.10.2, no CuPy).
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
  line, `LUMENAIRY_MEM_BUDGET_MB=8192` pinned, `PYTHONHASHSEED=0`.
* **The digest is the whole record.**  Every bit-identity key folds the returned
  object's type name and bytes -- or the exception type and message -- plus every
  warning in EMISSION order.  NaN and signed zero are compared as bytes; no float
  `==` appears anywhere in the digest path.
* **Decisions, not readings.**  Every bar in the new test files is derived at
  runtime from the quantity it bounds (a summation growth factor, an FFT
  spread, an oracle's own piston and truncation error, a finite difference's
  `eps^(2/3)` floor) and carries its measurement and date.  Where a claim needs a
  premise -- that a span exists to order, that a guard's gate is reachable, that
  two association orders differ at all -- the premise is asserted first and fails
  loudly on its own.

Probes and JSON: `validation/probe_wave5_hyg2/`.
Tests: `tests/unit/test_wave5_h2_mft_direct.py`,
`tests/unit/test_wave5_h2_collins_jax.py`,
`tests/unit/test_wave5_h2_near_focus_table.py`, and the extended D3 group in
`tests/unit/test_verify_b11c_structure.py`.

---

## H2-1 (audit item 14) -- the direct-matrix MFT branch

### What was there

`lumenairy/propagators/mft.py`'s Notes and `_bluestein.py`'s module docstring have
both named "a direct matrix-Fourier transform (`O(N^2 M^2)`)" as the chirp-Z
reduction's alternative since the MFT propagators were written.  Neither shipped
one, and nobody had measured where the two cross.

The quoted complexity is also wrong in the direction that matters.  `O(N^2 M^2)`
is the cost of the UNFACTORED four-index sum.  The transform is separable -- the
existing code already knows this, it builds its own chirp kernel as
`h_y[:, None] * h_x[None, :]` -- so a dense evaluation is two matrix products at
`O(M N^2 + M^2 N)`, which is `O(N^3)` on a square grid and not `O(N^4)`.  That one
factor is the difference between "never worth it" and "the better route over most
of the shapes this propagator is written for".

### What ships

`_bluestein._direct_matrix_2d` -- ONE `xp`-parametrised implementation.  It takes
the four index centres, so the non-centred primitive's convention is the case
`cI = cO = 0` and both `_bluestein_2d` and `_bluestein_centred_2d` dispatch to the
same function (pinned: `test_the_dense_kernel_serves_both_index_conventions`).
The two products are taken in whichever order costs fewer multiply-adds, decided
from the four grid sizes alone.

Reached through `method=` on `_bluestein_2d`, `_bluestein_centred_2d` and the
three public entry points `fresnel_propagate_mft`, `fraunhofer_propagate_mft`,
`angular_spectrum_propagate_mft`.  Vocabulary `{'auto', 'bluestein',
'separable', 'direct'}`, CLOSED and checked -- an unrecognised value raises rather
than falling through to a default, the same rule `gap_kernel` was given after
defect D4.

`method='auto'` is the pre-5.48 dispatch exactly.

### Bit identity of the default

`validation/probe_wave5_hyg2/probe_mft_bitid.py`, 179 keys: the three propagators
over 4 shapes x 3 distances x 3 zoom factors, off-axis centres, anisotropic
pitches, complex64, back-propagation, `bandlimit=False`, the private
`_bluestein_separable` flag, five guard rows (two refusals, one replica warning),
`resample_field` on both legs, the two Bluestein primitives directly at 3 shapes x
2 signs x 2 `separable` settings x 3 centre conventions with three guard rows, and
the two carrier readouts that consume the separable route.

| build | keys | identical | differing | only-base | only-branch |
|---|---|---|---|---|---|
| WIN-py3.14 | 179 | **179** | 0 | 0 | 0 |
| WSL-py3.12 | 179 | **179** | 0 | 0 | 0 |

### The tolerance between the reductions, DERIVED

Each output point is a sum of `n = Ny*Nx` terms whose kernel has unit modulus, so
its summation condition number is `kappa = sum|E| / |F|` and a summation whose
growth factor is `g` commits at most `g * eps * sum|E|` of ABSOLUTE error.  Two
routes therefore agree to `(g_a + g_b) * eps * sum|E|`, with

* `g = log2(n/128) + 8` for NumPy's pairwise summation (blocked at 128),
* `g = 3 * log2(L^2)` for a chirp-Z route (three FFTs of length `L^2`),
* `g = sqrt(n)` for the dense route's two BLAS products.

The reference is the same sum evaluated one output point at a time with
`np.sum` -- pairwise, float64, no mpmath and no float128 -- with the phase built
the same way the dense route builds it, so what separates the reference from the
routes is the SUMMATION and nothing else.

The bar is ABSOLUTE, so the apples-to-apples reading against it is the max-abs
departure, not the relative L2.  Both are given; the bar column is what the test
asserts against (WIN / WSL):

| case | `kappa` max | route | max-abs vs reference | derived bar | decades of room |
|---|---|---|---|---|---|
| N=16 M=8 | 1.19e+02 | bluestein | 2.973e-14 / 2.975e-14 | 2.745e-12 | 1.96 |
| | | separable | 2.586e-14 / 2.892e-14 | 2.745e-12 | 1.98 |
| | | **direct** | **1.281e-14 / 1.256e-14** | 1.879e-12 | **2.17** |
| N=32 M=16 | 3.44e+02 | bluestein | 2.930e-13 / 2.845e-13 | 1.266e-11 | 1.64 |
| | | separable | 2.865e-13 / 2.879e-13 | 1.266e-11 | 1.64 |
| | | **direct** | **5.783e-14 / 5.753e-14** | 1.223e-11 | **2.33** |
| N=48 M=24 | 8.21e+02 | bluestein | 1.079e-12 / 1.038e-12 | 3.187e-11 | 1.47 |
| | | separable | 1.041e-12 / 1.055e-12 | 3.187e-11 | 1.48 |
| | | **direct** | **1.005e-13 / 8.556e-14** | 3.899e-11 | **2.59** |

Relative L2 against the same reference, for scale: bluestein 8.38e-16 /
2.24e-15 / 6.04e-15, separable 6.99e-16 / 2.27e-15 / 6.03e-15, direct 3.42e-16 /
3.92e-16 / 4.81e-16.  The two chirp-Z reductions against EACH OTHER: 6.68e-16 /
4.88e-16 / 5.49e-16 relative, 2.51e-14 / 5.71e-14 / 1.06e-13 max-abs, against
bars of 4.14e-12 / 1.91e-11 / 4.80e-11.

**Stated tolerance.**  The three routes agree to
`(g_a + g_b) * eps * sum|E|` and are measured **1.47 to 2.59 decades** inside
it.  The dense route is the most accurate of the three at every case, by a
factor of **2.4 to 12.6**, because it does not spend mantissa on a chirp phase.
They are NOT bit for bit equal and no caller may assume they are.

**CORRECTED, Round 2 (V-D8).**  An earlier wording here said "the margin
narrows with `n`, as it should, because the growth factors are upper bounds and
the actual errors grow more slowly".  That is backwards twice over: a bar that
grows faster than its quantity gives a WIDENING margin, and the two routes do
OPPOSITE things.  The DENSE route's margin WIDENS with `n` (1.84 -> 3.60
decades from 16x16->8x8 to 256x256->128x128, measured on both builds) because
its error grows like `sqrt(n)` against a bar that grows like `n`; the chirp-Z
routes' margin NARROWS (1.93 -> 0.73 over the same ladder) because theirs grows
faster than `3*log2(L^2) * eps * sum|E|`.  The bar is a bound for both at every
shape measured -- never crossed up to N = 256 on either build -- and the shapes
the shipped test file uses (max 32x24) are 1.6-2.4 decades inside it, so the
shipped assertions are sound.  The chirp-Z half of the bar is the one with a
finite lifetime: extrapolating the measured trend it would first be crossed
near `N ~ 4000-5000`, outside anything this file exercises but inside nothing
about the formula that says so.  Gated as a TREND by
`tests/unit/test_verify_wave5_hyg2.py::test_the_two_reductions_margins_move_in_opposite_directions_with_n`.

### The phase budget -- where the dense route is not merely an alternative

`_bluestein_2d` warns when `alpha * N_max^2 > 1e15`, advising a fall back to "a
regular FFT propagator".  The dense route reduces `t = alpha*(n - cI)*(k - cO)`
by `t - rint(t)` before calling `exp`, which is EXACT for `|t| <= 2**52`, so it
has no such phase.  Measured at N=24, M=12 against the pairwise reference
(WIN / WSL):

| phase budget | chirp-Z rel L2 | warned? | dense rel L2 | warned? |
|---|---|---|---|---|
| 9.0e+00 | 1.97e-15 / 1.98e-15 | no | 3.73e-16 / 3.72e-16 | no |
| 1.0e+12 | **1.92e-04** / 1.92e-04 | **no** | 3.65e-16 / 3.61e-16 | no |
| 1.0e+15 | **2.47e-01** / 2.47e-01 | **no** | 3.52e-16 / 3.50e-16 | no |
| 1.0e+17 | 1.56e+00 / 1.56e+00 | yes | 3.66e-16 / 3.59e-16 | no |

**A finding recorded, not fixed (see the decisions section):** the warning fires
three decades after the chirp route has already lost four digits.  At a budget of
1e12 the route returns an answer wrong in the fourth significant figure and says
nothing.  Moving the threshold changes warning behaviour on existing callers, so
it is a maintainer decision and not this package's to take.

**FIXED IN ROUND 2, AND IT WAS WORSE THAN THREE DECADES (V-D5).**  Against a
`math.fsum` correctly-rounded reference -- the pairwise reference above is the
wrong instrument here, because at a large budget its own chirp is as wrong as
the route's -- the error is LINEAR in the budget, `rel ~ eps * budget`,
measured over 11 decades at two geometries on both builds.  There is no cliff:
1.5e-08 already at a budget of 1e8, and the first budget that warned at all was
3.16e15, by which point the answer was 25 % wrong.  The threshold is now
`1e-6/eps = 4.5036e9`, the budget at which six significant figures remain
(measured 5.32e-07 there), and the message names the error the budget implies
and offers `method='direct'` beside the old advice.  This is a change in
WARNING behaviour and not a byte move; shipped callers stay silent, because at
the natural MFT grids `alpha = zoom/N` and `budget = zoom*N ~ 1e4`, which
`tests/unit/test_wave5_h2_mft_direct.py` asserts as its own claim.

The dense route is taken BEFORE that guard, deliberately: warning on the one
route the warning's own advice is the alternative to would be a false positive.

### A capability the tables do not show: the dense route needs no FFT

It is two matrix products, so it runs wherever `xp.matmul` does.  MEASURED
2026-09-19 on this box, whose cuFFT DLL is broken (the reason
`tests/unit/test_niche_k2_carrier_backends.py` skips its propagating CuPy arms):
`_direct_matrix_2d` with `xp=cupy` returns complex128 agreeing with the NumPy
route to **3.79e-16** relative on this fixture.  The reading is
FIXTURE-DEPENDENT and the single number needed its fixture (V-D16e):
independently re-measured at **2.803e-16** (16x16->8x8), **4.451e-16**
(64->32), **5.309e-16** (128x96->48x64) and **3.556e-16** (centred 32->16).
JAX also runs the route, eager and under `jit`, at 3.72e-16 / 3.61e-16.  The suite asserts the PROPERTY rather than the
reading -- the function takes no `fft2` / `ifft2` argument and its body mentions
no transform, no `next_fast_len` and no padding -- because asserting the CuPy
reading would mean skipping on a resource check.

### The crossover, in memory and in time

`validation/probe_wave5_hyg2/probe_mft_direct.py`, `N` in
{64,128,256,512,1024} x `M` in {32,64,128,256,512}, plus four shapes beyond that
box because the requested box does not contain the time crossover.  Best of five,
two cache regimes (`cold` = every library cache dropped before each repeat, which
is the production regime `_bluestein.py`'s own byte-cap comment records with
`hits = 0` across a full production order; `warm` = nothing cleared).  Timing and
`tracemalloc` are separate passes -- tracemalloc charges per allocation and the
chirp-Z route allocates far more objects, so timing inside it would bias the very
comparison the table is for.

**MEMORY.  `tracemalloc` peak, cold, MB.**

| N | M | chirp-Z 2-D | separable | dense | analytic (dense) |
|---|---|---|---|---|---|
| 64 | 32 | 1.1 | 0.2 | **0.1** | 0.1 |
| 256 | 32 | 11.7 | 2.4 | **0.5** | 0.5 |
| 512 | 32 | 43.0 | 9.0 | **0.9** | 0.9 |
| 1024 | 32 | 159.6 | 34.7 | **1.8** | 1.9 |
| 1024 | 128 | 186.8 | 37.8 | **7.4** | 7.6 |
| 1024 | 512 | 319.0 | 50.4 | **29.4** | 33.6 |
| 2048 | 1024 | 1275.4 | 201.5 | **117.5** | 134.2 |

The analytic count is a model read off the source (six live `(Ly, Lx)` arrays plus
the pre-chirped input for the 2-D route; three `(N, L)` arrays for the separable
one; two `M x N` kernels, the float64 phase scratch, one intermediate and the
output for the dense one), not a fit.  It tracks the measurement on the dense
route to a MEDIAN of 9.5 %, best 0.3 %, worst 14.3 % -- and it over-counts
wherever it is far off, because `tracemalloc` never sees the two kernels and the
float64 phase scratch all live at once (the scratch is freed inside the builder
before the second kernel is built).  The probe JSON prints both numbers at every
shape, so where the model and the measurement disagree the disagreement is
visible rather than hidden, and the ORDERING claim below is made on the MEASURED
peaks, not on the model.

**The memory ordering `dense < separable < chirp-Z 2-D` holds at all 29 shapes on
BOTH builds.**  That is build-free: it follows from the padding (`L =
next_fast_len(N + M - 1)` per axis) and not from a timing.

**TIME.  Best-of-five seconds, cold.**  This is a per-build reading and the report
says so explicitly.

Windows py3.14:

| N | M | chirp-Z 2-D | separable | dense | winner |
|---|---|---|---|---|---|
| 1024 | 32 | 0.1000 | 0.0381 | **0.0109** | direct |
| 1024 | 64 | 0.0620 | 0.0454 | **0.0196** | direct |
| 1024 | 128 | 0.0942 | 0.0550 | **0.0405** | direct |
| 1024 | 256 | 0.1656 | 0.1052 | **0.0963** | ~tie |
| 1024 | 512 | 0.1748 | **0.1293** | 0.1719 | separable |
| 1448 | 1448 | 1.5587 | **0.5899** | 1.4957 | separable |

WSL py3.12:

| N | M | chirp-Z 2-D | separable | dense | winner |
|---|---|---|---|---|---|
| 1024 | 32 | 0.1779 | 0.0103 | **0.0062** | dense |
| 1024 | 64 | 0.1460 | **0.0100** | 0.0118 | separable |
| 1024 | 128 | 0.1845 | **0.0093** | 0.0243 | separable |
| 1024 | 512 | 0.5758 | **0.0177** | 0.1225 | separable |
| 1448 | 1448 | 2.1609 | **0.5849** | 1.0211 | separable |

**The time crossover is PER-BUILD, and the two builds disagree by a factor of
TWO in where it sits, not four.**  (Re-measured in Round 2: at N = 1024 the
Windows crossover brackets `M/N` in (1/8, 1/4) and the WSL one in (1/16, 1/8)
-- adjacent octaves sharing the endpoint `M = N/8`.  The original "Windows
`M <~ N/4`, WSL `M ~ N/16`" quotes the OUTER edge of each bracket, which is
what made it read as 4x.  The decision the claim carries -- per-build,
therefore no threshold may ship -- is unchanged.)  On Windows the dense route is the fastest up to
roughly `M = N/4`; on WSL, where scipy's pocketfft drives the separable route's
1-D passes with its own worker pool (`SCIPY_FFT_WORKERS = -1`, which
`OMP_NUM_THREADS=1` does not constrain), the separable route wins from about
`M = N/16`.  The 2-D chirp-Z route is the slowest of the three at every shape on
WSL and at most shapes on Windows; at `M >= N/2` with `N >= 1448` the Windows
ordering between it and the dense route is inside the run-to-run spread of a
contended box (re-measured both ways on two runs -- 1448x1448 reads chirp-Z
1.5587 vs dense 1.4957 in the table above and dense 1.4120 vs chirp-Z 0.8540 on
the re-run) and is NOT build-free, unlike the MEMORY ordering.

That disagreement is the reason no threshold ships.  A threshold-automatic switch
keyed on time would be a per-build constant -- exactly the shape the testing
standards call S1, and the shape that burned five release tags.

### Decisions owed to the maintainer (H2-1)

1. **Threshold-automatic `method`?**  The table above is what it would have to be
   set from.  On MEMORY the rule is build-free and unambiguous -- the dense route
   is smaller at every shape measured, by 1.5x to 88x -- so a memory-driven
   automatic switch (take the dense route when the chirp-Z route's projected
   `16 * 6 * L^2` exceeds the RAM budget) is defensible and would need no
   per-build constant.  On TIME it is not: the crossover moved by 4x between two
   builds of the same library.  Whatever is chosen, it MOVES ANSWERS -- the three
   routes agree to round-off, not bit for bit -- so it needs a Migration note and
   cannot be a silent default change.  Shipped default is `'auto'`, byte-identical.
2. **The phase-budget warning threshold.**  1e15 is three decades late
   (measured above).  1e12 would be the first budget at which the chirp route's
   error is visible in four significant figures.  Moving it makes existing quiet
   callers noisy, which is a behaviour change.
3. **`resample_field`'s chirp-Z leg** already uses `method=` on the public name
   for a different axis (`'chirpz'` vs `'spline'`), so the dense route is NOT
   exposed there; if it should be, the keyword needs a different spelling.

### Tests

`tests/unit/test_wave5_h2_mft_direct.py`, 29 ids, all green on both builds.
Nothing in it pins a timing or a byte count read off one build; the memory claim
is asserted as an ORDERING on the two routes' analytic counts.

---

## H2-2 (audit item 18) -- `_collins_transport` on the field's own backend

### What was there

The transport called `np.asarray` / `np.ascontiguousarray(..., dtype=np.complex128)`
and then `_fft2`, `_collins_angle_support`, `_collins_exact_kernel_correction`,
`_collins_space_support`, `_collins_sampling_stats` and `_bluestein_centred_2d`.
WP-B11 part a measured that the brief's skip clause did not apply (jax IS
importable here) and that the work was "a chain, not a signature".

### What the chain actually needed, measured helper by helper

Three of the six needed nothing, and saying so is part of the answer:

* `_collins_power_marginals` already routes through `lumenairy.backend.to_numpy`
  and accumulates host-side in row bands, by design ("like every other measurement
  in this module").  So `_collins_space_support` and `_collins_angle_support`,
  which are thin wrappers over it, were ALREADY backend-agnostic -- what they were
  not is tracer-safe.  Giving them an `xp` and reducing on-device would change the
  NumPy summation order and break the bit-identity contract, so it is deliberately
  not done.
* `_collins_sampling_stats` takes only Python floats.

Three needed threading, and got it with no `_jax` twin of anything:

* `_collins_axis_chirp` gains `bld=np`.  The screen is FIELD-INDEPENDENT, so it is
  built on `bld` (host NumPy for a JAX field -- the module's standing S2-3 contract,
  because `k u^2 / 2R` reaches ~1e5 rad and must not be formed in float32) and
  moved by the caller's `_to_dev`.
* `_collins_exact_kernel_correction` gains `(xp, is_jax, bld) = (np, False, np)`.
  Its `exp(i*phase)` now goes through `_tf_phase_to_H` -- the builder
  `_exact_tf_2d_xp` and `_fresnel_tf_2d_xp` already share -- so the module has ONE
  such implementation and not a NumPy one beside a device one.  On `xp is np` that
  builder does exactly what the inline code did (`np.empty`, `np.cos(out=H.real)`,
  `np.sin(out=H.imag)`), which is why the NumPy path is byte-identical through it.
* `_collins_transport` itself: `_backend_of(env)` for the triple, `_fft2_pair` for
  the transforms, `_as_c_order` for the contiguity, `_to_dev` around both chirps,
  and `xp` / `fft2` / `ifft2` passed to `_bluestein_centred_2d`.

Two new helpers, each with a reason that is not stylistic:

* **`_fft2_pair(xp, is_jax)`** returns `lumenairy.backend.fft2`'s dispatch as the
  CALLABLES.  Identity is the point, not equivalence: `_bluestein_2d` keys its
  chirp-kernel FFT cache on `fft2 is fft_infra._fft2`, so routing the NumPy path
  through the `backend.fft2` wrapper would make that test false.  **CORRECTED,
  Round 2 (V-D9):** the sentence used to continue "and silently turn the cache
  off for every Collins leg", whose premise fails.  MEASURED over three
  identical legs on both builds: with the shipped
  `_EXACT_READOUT_SEPARABLE_BLUESTEIN = True` a Collins leg takes the SEPARABLE
  route (`_bluestein.py:561`), which never reaches that cache at all -- 0
  entries and 0 hits -- so the cache is off for every Collins leg today, and
  the identity is what keeps it available to the `separable=False` route and to
  every other caller of the 2-D arm (where a wrapper really does cost 1 entry /
  2 hits -> 0 / 0, worth 3.3x of wall time on WSL).  A test asserts the identity AND that the
  callable agrees with `backend.fft2` bit for bit on both backends, so there is one
  dispatch and not two.
* **`_as_c_order(a, dtype, xp)`** because `jax.numpy` has no `ascontiguousarray` at
  all (a JAX array exposes no strides to make contiguous, so `asarray` IS the
  contiguous form).  NumPy and CuPy keep the historical call.

`_is_traced` is the fourth addition; see below.

### Bit identity of the NumPy path

`validation/probe_wave5_hyg2/probe_collins_bitid.py`, 84 keys: the direct entry
with `transport='collins'` at four leg geometries x three `gap_kernel` values x
three output-reference spellings, the astigmatic arm (including its refusal of
`'exact'`), complex64, tilt on both kernels, the guard's three dispositions on a
leg that violates it, the Sziklas transport alongside, a coarse grid at a second
wavelength, both focus readouts, and the helper chain called directly (support
measurements, marginals, 18 axis-chirp cases, 6 exact-kernel corrections plus the
evanescent-tilt refusal, the sampling stats, the wrap ratio, the ABCD, and four
spellings of the private transport including `B = 0` and a `stats_out` dict).

| build | keys | identical | differing | only-base | only-branch |
|---|---|---|---|---|---|
| WIN-py3.14 | 84 | **84** | 0 | 0 | 0 |
| WSL-py3.12 | 84 | **84** | 0 | 0 | 0 |

### JAX parity, against a bar measured on the running build

The bar is not a remembered residual.  The only thing entitled to differ between
the two backends is the FFT -- NumPy's path is pyFFTW or scipy's pocketfft, JAX's
is XLA's own transform -- so the bar is measured first, from one forward transform
of the fixture through each backend, times an UPPER BOUND on the chain depth:
the measurement transform, the exact-kernel correction's inverse (on the
'exact' arm only) and the Bluestein primitive's three -- five on the 2-D arm,
six on the separable arm, so six is the bound used.

MEASURED 2026-09-15/19, Windows py3.14, 64x64 Gaussian at 8 um, `R` = -50 mm,
`z` = 5 mm:

| quantity | reading |
|---|---|
| single-FFT NumPy-vs-JAX spread | 2.6853e-16 |
| bar (x6 chain depth, PRE-FLOOR) | 1.6112e-15 |
| bar the test actually asserts against, `max(6*spread, 32*eps)` | **7.105e-15** |
| JAX vs NumPy, `gap_kernel='fresnel'` | 8.2711e-16 |
| JAX vs NumPy, `gap_kernel='auto'` / `'exact'` | 9.6583e-16 |
| smallest real signal on the fixture (exact vs fresnel kernel) | 1.0741e-06 |

Two-sided: the disagreement is below the bar, and the bar is nine decades below
the smallest real signal.  (V-D16f: the table's 1.6112e-15 is the PRE-FLOOR
term; `_fft_spread_bar` takes `max(6*spread, 32*eps)` and on this fixture the
floor dominates by 4.4x, so what is actually asserted against is 7.105e-15 and
the measured 8.2711e-16 is 0.94 decades inside it, not 0.29.  The floor is well
reasoned -- a spread measured as exactly zero would make the bar zero -- and
`32` has no stated origin either.)  The test asserts BOTH, so if the two ever met it would
say the comparison had become noise instead of passing.  The astigmatic arm and
the tilted exact kernel carry their own arms.  `jax_enable_x64` is forced on --
in float32 the comparison would measure JAX's dtype policy, not the port.

### Under a trace: refused, not defaulted

Two decisions in the transport are taken by MEASURING the envelope, and a Tracer
has no entries to measure:

* the `gap_kernel` resolution reads the envelope's angular half-width to decide
  whether the exact-kernel refinement can be applied at all (`k4`, the wrap ratio
  of its impulse response over the reduced frame);
* the Kelly K1/K2/K3 guard reads the envelope's measured support box.

Under `jax.jit` / `jax.grad` the transport refuses unless the caller has taken
both -- `gap_kernel='fresnel'` and `on_collins_sampling='ignore'` -- and names
each blocked decision and its remedy in the message.  `stats_out=` is refused for
the same reason.

The two alternatives were considered and are worse, on this repository's own
precedents.  Silently defaulting to `'fresnel'` is the shape of defect D4 (an
unrecognised `gap_kernel` falling through to the paraxial arm), which the
vocabulary gate exists to remove.  A conservative grid-edge guard -- evaluating
K1/K2/K3 at the grid half-width instead of the measured support, which needs no
field values -- would refuse legs whose measured departure from the analytic ABCD
field is 5.6e-08 of peak; `_collins_sampling_stats`'s own docstring records that
reading.

An EAGER JAX or CuPy array is not a Tracer and measures normally, including
raising the guard's warning.  That is asserted separately, so the refusal is
demonstrably about TRACING and not about JAX.

### The gradient, against a central difference on a ladder

A central difference has error `~ P''' h^2 / 6` plus `~ eps |P| / h`, so its
accuracy is a U-curve in `h` and no single step is "the" answer.  The ladder is
scanned and the claim is made at its best, against the smallest error a central
difference can achieve at all, `~ eps^(2/3) = 3.667e-11` relative, times ten.

| h | rel disagreement with `jax.grad` |
|---|---|
| 1e-02 | **6.4688e-13** |
| 3e-03 | 8.8375e-13 |
| 1e-03 | 2.0676e-12 |
| 3e-04 | 1.2140e-11 |
| 1e-04 | 1.2140e-11 |
| 3e-05 | 1.0626e-10 |
| 1e-05 | 7.2251e-10 |

Best 6.47e-13 against a bar of 3.67e-10 -- 567x inside it, and the U-curve's
upturn below `h = 1e-4` is visible, which is what tells the reader the ladder is
doing its job.  The merit is transported power, which the leg conserves to grid
accuracy, so the gradient also has an analytic SHAPE: it must be proportional to
the input amplitude.  Measured correlation 0.9999996831.  A falsification arm
asserts the gradient is neither all-zero nor constant, so the correlation claim
cannot pass on a degenerate fixture.

### Decisions owed to the maintainer (H2-2)

1. **Should the traced refusal instead offer a trace-safe guard?**  A
   `jax.pure_callback` could evaluate the Kelly conditions on a materialised copy
   at trace time, but its result still cannot be branched on, so it could only
   warn -- and warning once per TRACE rather than once per call is a different
   contract.  Not attempted; recorded.
2. **CuPy was not exercised on the device.**  `fft_infra.CUPY_AVAILABLE` reads
   **True** on this Windows build (cupy 14.0.1, one visible device) -- the
   earlier "reads False" was wrong, corrected in Round 2 (V-D10) -- but the
   FIRST TRANSFORM raises `ImportError: DLL load failed while importing cufft`,
   the known broken-DLL condition `test_niche_k2_carrier_backends.py` records.
   So the CuPy arm of the port is exercised structurally (the same code path,
   `bld is xp`) but not run on hardware.  The CONCLUSION stands; the evidence
   cited for it did not.  (`_as_c_order`'s CuPy half and H2-1's dense MFT
   kernel DO run on the device, because neither needs a transform.)

### Tests

`tests/unit/test_wave5_h2_collins_jax.py`, 21 ids, all green.  Includes a
structural inventory asserting no Collins helper carries a backend suffix and the
module imports no `*_jax*` sibling for this chain -- the "ONE implementation per
kernel" rule, gated rather than reviewed.

---

## H2-3 (audit item 20) -- the near-focus exact-kernel table

### What was missing, and what it turned out to be

WP-B11 section 2.20: the fixture was built and the dropped quartic computed, but
"`propagate_carrier_referenced` takes and returns an ENVELOPE referenced to a
carrier, and the first two spellings of the reference bookkeeping gave O(1)
residuals and then a blanket `ValueError`".

The bookkeeping is now right, and so is a second thing that was not flagged: the
first spelling of this round's own probe put a 6.5 mm output window on a 1.9 mm
chirp-Z period (K3 = 3.46) and read 2.83 -- a WRAPPED answer, not a wrong
bookkeeping.  That is recorded because it is the same class of mistake and it
took a measurement, not a reading, to tell the two apart.

### The fixture, derived

`theta = lambda / (pi w0)` fixes `lambda = pi w0 theta = 1.0000e-06 m`;
`zR = pi w0^2 / lambda = 795.77 um`; placing the input plane one focal length
before the waist fixes `q_in = -f - i zR`, hence
`R_in = -(f^2 + zR^2)/f = -20.0317 mm` and
`w_in = sqrt(lambda (f^2 + zR^2)/(pi zR)) = 400.03 um`.

**The bookkeeping in one sentence:** the input ENVELOPE is exactly the real
Gaussian `exp(-r^2/w_in^2)`, because `1/q = 1/R + i lambda/(pi w^2)` splits into
its real part (the carrier) and its imaginary part (the amplitude) with no cross
term.  So the carrier handed to the propagator IS the beam's own wavefront and
nothing has to be fitted.

Oracle: the whole-function `q` form
`E(r,z) = exp(i k z)/(1 + z/q_in) exp(i k r^2/(2(q_in + z)))`, the convention
WP-B11 section 2.17 established for this library.  It carries the absolute piston,
so every comparison is piston-included and a convention error cannot cancel.

Bar: the oracle's own floor, `eps * k * |z|` (the piston's representation error,
2.8e-11 at `z` = 20 mm) plus `exp(-(N dx/2)^2 / w^2)` (the Gaussian tail the grid
truncates and the oracle does not), times ten.

Grid: N = 512 at `dx` = 8 um in, output pitch derived per plane from three
constraints -- window (`6 w / N`), amplitude (`w/8`) and curvature
(`lambda |R| / (4 w)`), whichever binds.

### The bookkeeping, validated first

| case | spelling | rel L2 | bar |
|---|---|---|---|
| collimated, `gap_kernel='fresnel'` | reconstruct | **5.572e-12** | 1.555e-10 |
| collimated, `gap_kernel='auto'` | reconstruct | 1.838e-04 | (= the quartic, below) |
| converging, 5 mm from focus | `carrier_out=inf` | **4.801e-13** | 1.234e-03 |
| converging, 5 mm from focus | reconstruct | **4.801e-13** | 1.234e-03 |
| converging, 5 mm from focus | sziklas + reconstruct | **1.759e-11** | 2.711e-10 |
| converging, 5 mm from focus | **carrier applied twice** | **1.35** | -- |

Identical to all printed digits on both builds.  The two Collins spellings agree
with each other to **1.334e-05 relative** (6.4e-18 absolute on a 4.8e-13
quantity) -- "identical to all printed digits" is true and "exactly" was not
(V-D16a); the test's own bar (`<= 1e-3 * max`) was already the honest one.  The
wrong spelling -- one of the two that produced WP-B11's O(1) residuals -- is
twelve decades away from the right one.

The collimated `'auto'` row is the independent confirmation that the bookkeeping
is right: its residual is not a bug but the paraxial ORACLE's own error, and it is
the predicted size.  The beam's dropped quartic `k z theta^4 / 8` = 1.500e-04
against a measured 1.838e-04 -- ratio 1.225.

### The table

`gap_kernel` in {auto, fresnel, exact} x {sziklas, collins} x nine rungs from 1 um
to 5 mm short of the geometric focus.  Both builds agree to 1.5e-05 relative at
worst (on the smallest quantity in the table); every other row agrees to better
than 1e-9.

Relative L2 against the analytic Gaussian, Windows (WSL identical to the digits
shown):

| d to focus | sziklas auto | sziklas fresnel | collins auto | collins fresnel | `k\|z_eff\|theta_beam^4/8` |
|---|---|---|---|---|---|
| 1 um | 7.313e-04 | 7.313e-04 | 4.717e-06 | **9.787e-13** | 1.541 |
| 3 um | 7.310e-04 | 7.310e-04 | 4.444e-06 | **9.788e-13** | 1.452 |
| 10 um | 7.300e-04 | 7.300e-04 | 3.696e-06 | **9.789e-13** | 1.208 |
| 30 um | 7.269e-04 | 7.269e-04 | 2.495e-06 | **9.791e-13** | 0.815 |
| 100 um | 7.161e-04 | 7.161e-04 | 1.164e-06 | **9.811e-13** | 0.381 |
| 300 um | 1.157e-02 | 1.157e-02 | 4.576e-07 | **9.587e-13** | 0.150 |
| 1 mm | 1.432e-07 | 1.761e-08 | 1.419e-07 | **8.580e-13** | 0.046 |
| 3 mm | 4.319e-08 | 5.085e-11 | 4.320e-08 | **5.995e-13** | 0.014 |
| 5 mm | 2.296e-08 | 1.759e-11 | 2.297e-08 | **4.801e-13** | 0.008 |

`'exact'` reads identically to `'auto'` at every rung (see the gate below).

**What the oracle can and cannot referee.**  It is a PARAXIAL solution.  So it can
say that the propagator reproduces the paraxial truth when asked for the paraxial
kernel, and it can measure how far the exact kernel departs from that truth.  It
cannot say which of the two kernels is more physical.  Every claim below is
phrased accordingly.

**The publishable near-focus row.**  On `transport='collins'` with
`gap_kernel='fresnel'` the residual holds the oracle floor from 5 mm all the way
down to 1 um short of the focus.  The Sziklas transport reads 7.3e-04 within
100 um of the focus and 1.8e-11 at 5 mm -- and the reason is NOT a collapsed
co-moving grid: at 1 um the plain co-moving pitch would be `m*dx` = 1.30e-08 m and
the leg returns 1.9098e-06 m, i.e. the automatic carrier -> through-waist ASM
bridge -> carrier split engaged.  That bridge costs 7.3e-04 relative on this
fixture, and the Collins transport goes direct and does not pay it.

**The Sziklas column has TWO mechanisms and this paragraph named one (V-D16b).**
Its WORST rung is the one the prose skipped: **300 um reads 1.1565e-02 with the
bridge OFF**, because the co-moving window there holds only 1.99 beam radii and
the edge truncation is `exp(-3.97) = 1.879e-02`.  At 1 / 3 / 5 mm the window
opens to 4.1 / 5.0 / 5.1 radii and the reading tracks that truncation to within
3x.  So the column is "the bridge, within 100 um" AND "edge truncation of a
collapsing co-moving window, at 300 um", and the second is the larger of the
two.

### The correction to the framing -- WHOSE angle

WP-B11 section 2.20 quotes the dropped quartic as `k |z_eff| theta^4 / 8` with
`theta` = the BEAM's 20 mrad (the last column above).  For a carrier-referenced
transport that is the wrong angle: the exact-kernel refinement is applied to the
ENVELOPE, whose angular content is what remains after the carrier is divided out.
Measured by the library's own instrument (`stats_out`), the envelope's
containment half-width is 1.9531e-03 rad at every rung and its analytic `1/e^2`
half-angle `lambda/(pi w_in)` is 7.9512e-04 rad -- 25x below the beam's.  Since the
term is quartic, that is a factor of **4.0e+05**.

### The derived law

Two one-variable sweeps, each holding the other.

**`z_eff`,** over the distance ladder at fixed envelope (nine rungs, `z_eff`
0.0597 .. 12.27 m, a span of 205x):

| slope | worst relative deviation from the power law | points |
|---|---|---|
| **0.9999985** (WIN and WSL agree to 1e-11) | 1.30e-06 | 9 |

**`theta`,** at one distance with the envelope's width scaled, so the geometry and
`z_eff` are held (six rungs, 5.3x span of angle):

| against | slope | worst relative deviation | points |
|---|---|---|---|
| the analytic `1/e^2` half-angle | **3.99970** | 1.88e-04 | 6 |
| the library's measured containment radius | 4.05582 | 1.92e-01 | 6 |

(The first spelling of this sweep sized every rung's output window from the
FIXTURE's beam rather than from its own, which clipped the narrow-envelope rungs
and fitted 3.07 with 55 % scatter.  The clipping is what the scatter was.)

**Together, one constant:**

```
departure_relL2 = 1.2248 * k |z_eff| theta_env^4 / 8
```

`C = 1.2248` on the converging ladder, and `C = 1.2253` on the COLLIMATED leg of
the bookkeeping section -- a different transport, a different carrier and a
different geometry.  One constant serving both is what makes it a law rather than
a fit.

### The gate, and why `'auto'` never falls back

`gap_kernel='auto'` resolves to `'exact'` at every rung.  The gate is `k4 <= 1`,
where `k4` is the wrap ratio of the exact kernel's impulse response over the
reduced frame, formed from the ENVELOPE's measured angle:

| d to focus | `z_eff` (m) | `k4` | resolved kernel | K1 |
|---|---|---|---|---|
| 1 um | 12.266 | 2.231e-05 | exact | 0.0325 |
| 100 um | 3.028 | 5.507e-06 | exact | 0.0364 |
| 5 mm | 0.0597 | 1.086e-07 | exact | 0.2928 |

`k4` peaks four decades below its bar of 1.  The gate is not near firing, and the
departure it would be guarding against peaks at 4.717e-06 -- not the O(1) the
beam-angle quartic (1.541 rad) suggests.

### Decision owed to the maintainer (H2-3): VERIFY-B4 F3

**"Should `gap_kernel='auto'` fall back to `'fresnel'` near a focus?"**

Measured answer on this fixture: it does not today, and the cost of not doing so
is 4.7e-06 relative at one micron from the focus, falling monotonically to
2.3e-08 at 5 mm.  So this is a decision to CHANGE behaviour, not to ratify it.

If a phase-magnitude gate is wanted alongside the existing wrap gate, the derived
form is

```
fall back to 'fresnel' when   1.2248 * k |z_eff| theta_env^4 / 8  >  tau
```

with `theta_env` the envelope's own half-angle -- NOT the beam's.  Two readings to
set `tau` against:

* the fixture's worst case is `1.2248 * 3.851e-06 = 4.717e-06`, so with
  `tau = 1e-4` (a tenth of a per-mille of the field) this fixture never falls
  back and the rule is inert on it;
* with the chain's own `_GAP_ENV_PHI_TOL_DEFAULT = 0.3` as `tau`, the fixture
  would need `z_eff > 9.8e+05 m` to trip -- unreachable.

**A rule keyed on the BEAM's angle would fire five decades too eagerly** (the
beam-angle quartic is 1.541 at 1 um, 5.1x the 0.3 tolerance, while the actual
departure is 4.7e-06).

**WHOSE framing that is, corrected (V-D16c).**  The 4.0e5 arithmetic is right
for **WP-B11 sec. 2.20**, which did use the beam's 20 mrad.  **VERIFY-B4 F3
itself used the ENVELOPE's MEASURED containment angle** (2.8574e-03 rad), which
over-predicts by **33.5x -- 1.5 decades, not five**.  Attributing the
beam-angle framing to F3 was wrong; what remains true, and is the point, is
that the term has to be evaluated at the envelope's ANALYTIC `1/e^2`
half-angle, and that the measured containment radius is not that angle either.
F3's own reading (2.4e-3 against 1.7e-14 one micron from focus) was a different
fixture, and Round 2 DID reproduce it: rebuilt exactly, the law predicts
2.3496e-03 against a measured 2.3496e-03.

**Recommendation: no change.**  The existing `k4` wrap gate is keyed on the right
angle already and correctly does not fire.  Nothing in this package moves a
default.

**That recommendation is a defensible reading of THIS ladder and is not
defensible in general (V-D16d), and Round 2 restates it.**  The "4.7e-06
falling monotonically to 2.3e-08" bound is a bound on the ladder's closest
approach to `A = 0`, not on the fixture: the ladder measures `d` from the
WAIST, and this carrier's geometric focus sits 31.66 um further on, so `z_eff`
caps at 12.27 m.  Walked to its own carrier focus the SAME fixture pays
**1.5431e-04 at 1 um and 1.5431e-03 at 0.1 um**, 327x the published worst case;
and VERIFY-B4 F3's fixture, whose ladder does approach `A = 0`, pays
**2.3497e-03 at `z_eff` = 1600 m** against `'fresnel'` at **1.7042e-14**.  The
`k4` gate cannot serve there: it is 4.65 decades below its bar where the
departure is 4.7e-06 and still 2 decades below it where the departure is
2.3e-03, because it bounds REPRESENTABILITY and not accuracy.  Round 2 ships
the accuracy-keyed condition **behind a switch that is OFF**
(`carrier._GAP_KERNEL_ACCURACY_TAU = None`, one line to arm), with the decision
measured at `tau = 1e-4`: the hygiene-2 ladder stays inert and F3's 1 um and
10 um rungs fall back.  The default still moves nothing.

### Tests

`tests/unit/test_wave5_h2_near_focus_table.py`, 14 ids, all green.  The
bookkeeping claims are decisions against the derived floor bar with the
double-carrier failure as the falsification arm; the table's monotonicity and both
halves of the law are premise-gated (the departure must span two decades, `z_eff`
must span 100x, the angle must span 4x, the envelope's angle must be well below
the beam's -- each asserted first and failing on its own terms).

---

## H2-4 (VERIFY-WP-B11c D3) -- the stale lens patch is loud

### The defect, restated from the measurement

Eight names moved to `elements/_lens_kernels.py` are re-exported into `lenses` by
value -- `CUPY_AVAILABLE`, `_is_cupy_array`, `_ensure_cupy_loaded`, `_load_numba`,
`_get_aspheric_sag_accum_numba`, `_ensure_numexpr_loaded`,
`_collect_semi_diameters`, `_warn_if_aperture_exceeds_grid` -- and every one of them
is read at CALL TIME out of the leaf's globals.  At 5.47.0
`monkeypatch.setattr(lenses, '_is_cupy_array', fake)` therefore SUCCEEDED, bound a
shadow in `lenses.__dict__` and reached nothing.

`del lenses.CUPY_AVAILABLE` was worse: it removed the re-export outright, after
which the name raised `AttributeError` for the rest of the process.  That is not
a deduction -- the first run of `probe_lens_d3.py` against the base tree died at
exactly that point, which is why the probe now saves and restores around every
attempt.

### The choice, measured

The brief allowed either forwarding the write through `_LensesFacade` to the leaf,
or refusing it loudly.  Forwarding the write while the read still comes from this
module's dict ALIASES (`lenses.X` would return the original while
`_lens_kernels.X` held the fake), so forwarding only makes sense together with
removing the by-value re-export -- i.e. making all eight LIVE FORWARDS.  That
option (call it A) was rebuilt on synthetic modules with no lumenairy import, so
the measurement is about Python's machinery and not about this library:

| reading | by-value re-export (today) | served only by `__getattr__` (option A) |
|---|---|---|
| in `from ... import *` | **True** | **False** |
| in `dir()` | True | True |
| in `vars()` | True | False |
| a function in the shell reading the bare name | works | **`NameError`** |
| a write reaches the leaf | no | yes |

`import *` reads `__dict__` and consults neither `__getattr__` nor `__dir__`, and
`CUPY_AVAILABLE` is the one public name among the eight -- so option A is a
public-surface change inside a durability fix.  `LOAD_GLOBAL` likewise does not
consult `__getattr__`, so option A lays a trap for any future code in `lenses.py`
that calls one of the eight by bare name; an AST walk of the module finds zero
such sites today (recorded as a probe key), which is the only reason option A was
possible at all.

**Shipped: refuse the write.**  `_LensesFacade.__setattr__` and `__delattr__`
raise `AttributeError` for the eight, with `_lens_kernels` named as the address
that works.  Reading is untouched.  This reproduces the BLAS half's observable
contract -- a stale substitution raises `AttributeError` and leaves no shadow --
by a different mechanism, which is what a test author actually relies on.

The refusing set and its message live INSIDE `_LensesFacade` as a class attribute
and a staticmethod, not at module scope: `dir(lenses)` is a bit-identity key of
WP-B11c's own verification, and a module-level name -- private or not -- would have
grown it.  (It did, in the first spelling; that is how the placement was chosen.)

### Bit identity

`validation/probe_wave5_hyg2/probe_lens_d3.py`, 131 keys: **59** under `D3-`
(the eight names x six -- read, write, write-had-no-effect, delete,
delete-had-no-effect, restored -- with the module state put back around every
attempt; plus the eight live-forward writes, `dir`, the facade's type and its
MRO), **9** under `optA-` (the counterfactual, on synthetic modules and then
anchored to the real one), and **63** under `L-`: the lens FIXTURES
VERIFY-WP-B11c used, re-driven here rather than imported -- 20
`surface_sag_general` cases over a coefficient grid, 3 `surface_sag_biconic`,
the grid-vs-aperture census at three grids x two safety factors with its
recommendation and its warning, the semi-diameter collector, 12
`_multi_indices_total_degree` and 15 `_fit_normaliser` cases over three pad
values.

| build | keys | identical | differ (expected) | differ (unexpected) | only-base | only-branch |
|---|---|---|---|---|---|---|
| WIN-py3.14 | 131 | **99** | 32 | **0** | 0 | 0 |
| WSL-py3.12 | 131 | **99** | 32 | **0** | 0 | 0 |

The 32 are exactly the eight names x {write, write-had-no-effect, delete,
delete-had-no-effect}.  The driver reports "expected to differ but identical" as a
failure too, so the change is proved to have LANDED as well as to have been
contained.

### Tests

`tests/unit/test_verify_b11c_structure.py`'s two D3 tests became decisions: 16
parametrised ids asserting the refusal for all eight names on write and on delete
(exception type, the leaf named in the message, nothing moved on either module,
and the redirected statement against the leaf working), one falsification arm
asserting the facade still assigns / reads / deletes any OTHER name and still
forwards all eight live writes, and one symmetry claim that both halves of WP-B11c
now refuse a stale patch the same way.  The file's own module docstring is
rewritten to record the defect as closed and to carry the option-A measurement.
57 ids, all green.

---

## Files touched

Library:

* `lumenairy/propagators/_bluestein.py` -- `_direct_matrix_2d`, `_SUM_METHODS`,
  the `method=` selector on both primitives, `__all__`, module docstring.
* `lumenairy/propagators/mft.py` -- `method=` on the three public entry points and
  their Parameters sections, the Notes complexity correction.
* `lumenairy/propagators/carrier.py` -- `_is_traced`, `_as_c_order`, `_fft2_pair`;
  `bld` on `_collins_axis_chirp`; `(xp, is_jax, bld)` on
  `_collins_exact_kernel_correction`; `_collins_transport`'s body and its BACKENDS
  docstring paragraph.
* `lumenairy/elements/lenses.py` -- `_LensesFacade._LEAF_OWNED_NAMES`,
  `_leaf_owned_refusal`, the two refusing dunders.

History fingerprints re-recorded in the same commits, with reasons:
`docs/history/lumenairy.propagators.mft.md`, `docs/history/carrier.md`,
`docs/history/lumenairy.elements.lenses.md`.  `--check` green.

Tests: three new files, one extended.
Probes and JSON: `validation/probe_wave5_hyg2/` (7 probe / driver modules, 22 JSON).
CHANGELOG: four entries inside the existing `## [Unreleased]` block.

## Out of scope, and said so

The brief named these and this package does not touch them:

* the rest of the `rcwa/_core.py` split (WP-B11 part a section 2.2 carries the
  per-block hazard list: module-level mutable state, four monkeypatching test
  files);
* the whole-grid surface body (WP-B11 part a section 2.3 -- folding it into the
  band generator would move the numexpr gate, the `_ensure_full_grids` path and
  the Fresnel dtype-promotion point, so it is not a bit-identical refactor);
* the package-root 2-cycle (`lenses_maslov`'s `from .. import raytrace`).

## What could not be measured

* **CuPy on the device, for the Collins port.**  `fft_infra.CUPY_AVAILABLE` reads
  **True** on this Windows build (V-D10 -- the earlier "reads False" was wrong);
  what fails is the first transform, with the `ImportError: DLL load failed
  while importing cufft` that `tests/unit/test_niche_k2_carrier_backends.py`
  already records.  WSL has no CuPy at all.  The Collins chain needs a working cuFFT, so its CuPy arm is
  exercised structurally (`bld is xp`, `out=` supported, `cupy.fft.fftfreq`
  present, one shared implementation) but was not run on hardware.  H2-1's dense
  MFT kernel is the exception and WAS run there, because it uses no transform at
  all (3.79e-16 against NumPy; see H2-1).
* **An uncontended wall-clock ladder.**  This box ran 44 other Python
  interpreters during the first ladder.  The published timings are best-of-five
  with every repeat kept in the JSON, and the report leans the crossover decision
  on the MEMORY ordering (build-free, identical on both builds at all 29 shapes)
  rather than on the times (per-build, and the two builds' crossovers differ by
  about 2x -- see the correction in the crossover section).
* **A non-paraxial oracle for H2-3.**  The analytic Gaussian is a paraxial
  solution, so it cannot referee `'exact'` against `'fresnel'` on physical
  accuracy.  The report measures the exact kernel's DEPARTURE from the paraxial
  truth and derives its law; deciding which kernel is closer to the Helmholtz
  answer needs a different oracle and is not attempted.
* **VERIFY-B4 F3's own numbers.**  Its reading (2.4e-3 against 1.7e-14 one micron
  from focus) is a different fixture from the one WP-B11 section 2.20 specified
  and this report measures; no attempt was made to reproduce it, and this report
  does not claim its numbers are wrong -- only that the angle the quartic is
  evaluated at has to be the envelope's.

---

# Round 2 (VERIFY-WAVE5-HYGIENE2) -- 2026-09-19

Every one of the verification's 22 findings, closed or answered, on branch
`refactor/wave5-hygiene-2-round2` against `112c3049`.  Nothing in this round
moves a shipped default: **245 of 245 fixture keys are byte-identical
archive-to-archive on BOTH builds**, and the sensitivity control on the same
245 keys is 199 differing between the two builds.

## How this round was gated

* **Archive-to-archive, from `git archive`.**  Both arms are read-only
  extractions -- `112c3049` into `C:/tmp/r2_arch/base` and this tree's index
  object into `C:/tmp/r2_arch/branch` -- and the probe runs in a child process
  whose `cwd` and `PYTHONPATH` name one tree.  `hlib.anchor` prints what bound
  and refuses to continue outside it.  Probe and JSON:
  `validation/probe_round2_wave5_hyg2/`.
* **One harness, not a third copy.**  The round-2 probes import
  `validation/probe_wave5_hyg2/hlib.py` from the WORKTREE, so both arms fold
  their results through identical code and only `lumenairy` comes from the
  archive.
* **Both builds, every time.**  Windows py3.14 (numpy 2.4.4, scipy 1.17.1, jax
  0.11.0, CuPy 14.0.1 with a device and a broken cuFFT DLL) and WSL py3.12
  (numpy 2.4.6 on scipy-openblas SkylakeX, scipy 1.17.1, jax 0.10.2, no CuPy),
  with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  LUMENAIRY_MEM_BUDGET_MB=8192 PYTHONHASHSEED=0` on the COMMAND LINE and
  pytest run with `--capture=sys -p no:randomly`.
* **Decisions, not readings.**  Every bar this round adds is derived at runtime
  from a quantity the running build measures, two-sided, and premise-gated.

### The byte-identity count, stated

`validation/probe_round2_wave5_hyg2/r2_kernel_bitid.py`, **245 keys**:

| group | keys | what |
|---|---|---|
| `tf::exact::` | 5 grids x (4 reduced distances x 4 tilts + 3 refusals + complex64 + anisotropic pitch) | `_exact_envelope_tf_step`, every arm |
| `tf::xp::` | 3 grids x (3 tilts + 1 refusal) | `_exact_tf_2d_xp` on `xp = np` |
| `corr::` | 4 grids x (4 `z_eff` x 3 tilts + refusal + anisotropic pitch) | `_collins_exact_kernel_correction` |
| `transport::` | 3 envelopes x (3 `gap_kernel` x 2 references + astigmatic + astigmatic-refusal + tilted + off-centre) + 3 guard dispositions with their stats dicts + `B = 0` | `_collins_transport` |
| `public::` | 2 transports x (3 `gap_kernel` x 3 envelopes + a tilted second-wavelength leg + a collimated leg + `z = 0`) | `propagate_carrier_referenced` |
| `readout::`, `nearfocus::` | 2 tilts + 4 distances x 2 transports | the exact focus readout and the near-focus ladder |

| comparison | keys | identical | differing | only-base | only-branch |
|---|---|---|---|---|---|
| base -> branch, WIN-py3.14 | 245 | **245** | 0 | 0 | 0 |
| base -> branch, WSL-py3.12 | 245 | **245** | 0 | 0 | 0 |
| base -> the FINAL committed tree, WIN-py3.14 | 245 | **245** | 0 | 0 | 0 |
| base -> the FINAL committed tree, WSL-py3.12 | 245 | **245** | 0 | 0 | 0 |
| **control**: WIN vs WSL on the base | 245 | 46 | **199** | 0 | 0 |

The control is what makes the other rows mean something: the same instrument
separates two builds on 199 of 245 keys and cannot separate the two trees on
any.  The last two rows are re-taken on the tree as committed, after every
edit in this round including the accuracy-switch code, so the claim is about
what ships and not about a snapshot taken halfway through.

ONE observable behaviour change is intended and is NOT in these keys: the lens
facade's refusal MESSAGE (V-D21).  Its text is a digest key of
`validation/probe_wave5_hyg2/probe_lens_d3.py`, so that probe's 32
expected-to-differ keys differ by one more thing than they did; the refusal's
type, its target set and everything `dir()` / `vars()` / `import *` report are
untouched, and `tests/unit/test_verify_b11c_structure.py` reads 57 passed.

The chirp phase-budget threshold (V-D5) is the other intended behaviour change,
and it is a WARNING change: no route's arithmetic moves, which is why it is
inside the 245/245 rather than beside it.

---

## Per finding

### V-D1 -- BLOCKER, CLOSED.  `.test_durations`

The 121 ids of the branch's four new files were timed SERIALLY in one process
with BLAS pinned on the command line (121 passed in 28.89 s, 25.9 s of timed
work) and SPLICED into the committed JSON -- 121 insertions, 0 deletions,
16 265 -> 16 386 entries, re-parsed as JSON.  Every pre-existing entry is
untouched, which is what keeps the other weights on one scale.  Gate:
**WIN 4 passed in 83.14 s, WSL 4 passed in 60.81 s** (base read 4 passed,
branch read 1 failed / 3 passed).  Commit `0a75c5ad`.

### V-D2 -- BLOCKER, CLOSED.  Sixteen stale citations, and the tool that could not see them

All sixteen re-anchored BY CONTENT from `git show f4f18851:<path>`:
`mft.py` `:489->527`, `:769->807`, `:972->1050`, `:605-630->643-668` and the
hand-shifted range `:645-654->649-658`; `carrier.py` `:1966->2041`,
`:1564->1621`, `:1626->1683`, `:2266->2418`, `:1830->1895`, `:1700->1765` with
its bare sibling `` `:1736`->`:1801` ``, `:2127->2279`, `:1906->1971`,
`:1868->1933`, and `propagators/carrier.py:1068->1125`.  The two the branch had
re-anchored CORRECTLY (`mft.py:550->588`, `carrier.py:1541->1598`) are left
alone, which the tool decides rather than assumes.

The tool moved from `validation/probe_wp_b11c/` to `scripts/reanchor_citations.py`
and gained four things, each answering a measured blindness: `--base` and
`--block` as parameters; automatic resolution of any citation tail that names
exactly one module under `lumenairy/` (the hand-kept `OWNED` map saw 19 of the
5.47.0 block's citations, the resolver sees all 107); per-ENDPOINT anchoring of
ranges; and a `def NAME` fallback for a citation whose line is a definition
whose SIGNATURE changed (the only reason `_collins_axis_chirp` and
`_collins_exact_kernel_correction` could not be anchored).  The alignment is
now by citing-SENTENCE skeleton rather than by substituting the base token,
which is the only way to see a citation someone has already re-anchored to a
different wrong number -- the `+34` where the content moved `+38`.

Gated by three new ids in `tests/unit/test_v5_3_2_walker_source_line_citation.py`
(V18's own file, beside the blind spot they close).  **Before/after proved on
this tree: with the stale CHANGELOG restored, V18 still reads `ok=107 drift=0`,
rc=0 -- and the new id FAILS.**  Commit `33a69080`.

### V-D3 -- MAJOR, CLOSED.  The port now reaches the public leg

`_collins_carrier_leg` resolves `(xp, is_jax, bld)` with the same `_backend_of`
the transport uses, and `_collins_input_box` takes its own forward transform
through `_fft2_pair` / `_as_c_order`.  MEASURED on both archives, both builds
(`r2_reach_{win,wsl}_{base,branch}.json`):

| input | base `112c3049` | this tree |
|---|---|---|
| numpy | `ndarray` | `ndarray` |
| eager JAX | **`numpy.ndarray`** (silent host demotion) | **`jaxlib ArrayImpl`** |
| traced | raw `TracerArrayConversionError` | the designed `ValueError` |
| CuPy | `TypeError: Implicit conversion to a NumPy array is not allowed` | `ImportError: DLL load failed while importing cufft` -- i.e. it STAYED on the device to the transform |

and the JAX-vs-NumPy relative difference through the public leg reads **exactly
0.0 on base** -- bitwise equal, because both arms were NumPy -- against
4.1309e-16 (WIN) / 4.1251e-16 (WSL) now.  That `0.0` is why no value test could
see it.

`test_the_public_entry_agrees_across_backends[collins]` is restated: the type
check is now the PREMISE, with the failure message saying why a demoted arm
makes the comparison vacuous.  A new id asserts the traced refusal is reachable
through the public entry, is a clean `ValueError`, and names
`_collins_transport`, `dx_out`, `gap_kernel` and `on_collins_sampling` -- the
leg cannot offer a spelling that RUNS, because it resolves its own output
lattice and quadrature from the measured phase-space box, a third measured
decision with no caller alternative.  The CHANGELOG sentence is corrected in
place with what was actually true recorded beside it.  Commit `4638f601`.

### V-D22 -- CLOSED, and it is the item that made V-D4 load-bearing

`carrier.py::_exact_dispersion_phase` is now the ONE place the non-paraxial
dispersion is written.  It takes the caller's own frequency axes (the three
sites' layouts are not bit-identical to each other and each one's is part of
its byte-identity contract), and the in-place NumPy fast path that was
`_exact_envelope_tf_step`'s alone now lives inside it and serves all three
sites.  The two shipped evanescent-carrier phrasings are reproduced verbatim by
`_evanescent_tilt_message`, because a bit-identity probe folds an exception's
message into its digest.

**Byte-identity: 245/245 on both builds** (the table above).

**The guard this removes, measured.**  Mutation arms on this tree:

| arm | `b4::TestSameTheorem` | `test_wave5_h2_near_focus_table.py` |
|---|---|---|
| M-A: the ONE consolidated kernel negated | **9 passed** -- the agreement guard is GONE, because both transports now move together | **5 failed / 9 passed** |
| M-B: the Collins reduced distance negated (the verifier's own arm) | 2 failed / 7 passed | **1 failed / 13 passed** -- and the single failure IS the new sign assertion |
| M-C: BOTH callers' reduced distance negated | 2 failed / 7 passed | 2 failed / 12 passed |
| none | 9 passed | 14 passed |

M-A is the defect a single shared kernel newly makes possible and it is exactly
the conjugation: the Collins total becomes `exp(i(k z_eff - z_eff(root-root0)))`
and so does the Sziklas one, so a cross-IMPLEMENTATION agreement test cannot
see it.  That is V-D22's warning, reproduced.

Pinned by `test_the_exact_dispersion_is_written_once`, which censuses THREE
independent tokens (the `q = 0` subtraction, the shifted-frequency radical, the
evanescent guard) with docstrings stripped, and asserts all three sites still
call the one helper.  Commit `e9341583`.

### V-D4 -- CLOSED.  The sign the near-focus file could not see

Added to `test_the_measured_departure_is_the_quartic_times_one_constant`: the
intensity-weighted mean of `arg(exact/fresnel)` must be NEGATIVE (the exact
kernel RETARDS -- `sqrt(k^2-q^2) < k - q^2/(2k)` for every real `q`) and must
equal the derived `-k z_eff theta_env^4/16`, the `<u^4> = 1/2` moment beside
the `sqrt(<u^8>) = sqrt(3/2)` moment the magnitude law is fitted on.  Bar 1 %,
~0.7 decades above the worst of three independent fixtures' measured agreement
(0.002 %, 0.04 %, 0.2 %).  Before/after in the M-B row above.

### V-D5 -- CLOSED.  The chirp phase budget, 7.5 decades late

`_PHASE_BUDGET_MAX = 1e-6 / eps = 4.5036e9`, derived from the measured law
rather than from taste.  Against a `math.fsum` correctly-rounded reference at
N=24 M=12:

| budget | 1e6 | 1e8 | 1e9 | 4.5e9 | 1e10 | 1e12 | 1e15 |
|---|---|---|---|---|---|---|---|
| chirp-Z rel L2 | 1.98e-10 | 1.41e-08 | 2.10e-07 | **5.32e-07** | 1.90e-06 | 1.84e-04 | 2.48e-01 |
| dense rel L2 | 3.10e-16 | 3.20e-16 | 3.29e-16 | 1.84e-16 | 3.29e-16 | 3.20e-16 | 2.83e-16 |
| warned? | no | no | no | **no** | **yes** | yes | yes |

Three ids: the LAW (slope 1.000 +- 0.1 over four decades, dense immune, and the
threshold deliberately NOT pinned so it stays true whatever the threshold
becomes), the THRESHOLD two-sided, and the QUIET-CALLER side (four natural
focal-zoom grids, budget computed from the caller's own numbers, each >= 3
decades under the threshold and each run with `RuntimeWarning` promoted to an
error).  **This is a change in WARNING behaviour and not a byte move**; it is
recorded as such in the constant's own comment and in the Notes.

### V-D6 -- CLOSED.  The gradient's shape bar

`corr > 1.0 - 1e-6` is replaced by the leg's own measured departure from a
scaled isometry.  `c = <g,a>/<a,a>`, `r = |g - c a|/|c a|`, and Pearson's
shortfall is second order: `1 - corr ~ r^2/2`.  MEASURED on the shipped
fixture, both builds: `r = 4.998e-04`, `1 - corr = 3.1689e-07`,
`(1-corr)/r^2 = 1.2686`.  Bars `r^2/100 < 1 - corr < 10 r^2` -- 7.9x above,
127x below -- with `0 < r < 1e-1` asserted first as the premise the expansion
needs.

### V-D7 -- CLOSED.  The `eps^(2/3)` floor model

`P(a) = sum|L a|^2` is an exact quadratic form, so `P''' == 0`, so there is no
truncation branch to balance against cancellation and `eps^(2/3)` is ~4.7
decades loose.  The bar is now the cancellation branch itself, PER RUNG:
`rel <= 10 * eps |P| / (h |g|)`.  MEASURED over a ten-rung ladder extended
upward to `h = 3e-1`: the ratio of the disagreement to its own floor is
0.077 .. 0.737, so the factor of ten carries 13.6x at the worst rung; best
7.55e-15 at `h = 1e-1`, at the TOP of the ladder.

`P''' == 0` is MEASURED by its SCALING, not assumed: a five-point third
difference of a quadratic returns pure round-off, which grows as `eps|P|/h^3`
-- measured -7.11e-12, -2.84e-08, -1.42e-05, +7.11e-03 at `h = 1e-1 .. 1e-4`,
1e3 per decade to three digits with the sign flipping.  **And the instrument is
not blind**: the same ladder on `(|E(0,0)|^2)^3` returns a CONSTANT
`P''' = +2.235e-04` over four decades, a disagreement falling exactly 10x per
decade, a fitted slope of `h^2`, and a real U with its minimum at `h = 1e-4`;
the truncation model predicts it to three significant figures.

### V-D8 -- CLOSED.  See the corrected "Stated tolerance" paragraph above.

### V-D9, V-D10, V-D16a-f -- CLOSED.  Report prose

Each is corrected IN PLACE above, beside the sentence it replaces, so the
report reads as one document rather than a document plus errata: the
`_fft2_pair` cache consequence (measured 0/0 on the shipped separable route),
`CUPY_AVAILABLE` (reads **True**; the first transform is what fails), the two
Collins spellings' 1.334e-05 (not "exactly"), the Sziklas column's SECOND
mechanism (300 um reads 1.1565e-02 with the bridge OFF), whose angle VERIFY-B4
F3 actually used (the envelope's measured containment angle -- 33.5x, not five
decades), the "no change" recommendation (a bound on the ladder, not on the
fixture), the CuPy 3.79e-16 (fixture-dependent, 2.803e-16 .. 5.309e-16 over
four), and the JAX bar (the `32*eps` FLOOR dominates by 4.4x, so 7.105e-15 is
what is asserted against).  The timing claims are corrected too: the crossover
differs by 2x and not 4x, and "the 2-D chirp-Z route is the slowest at every
shape on both builds" does not survive on Windows.

### V-D11, V-D12, V-D13 -- CLOSED.  Three traced shapes

* **V-D11**: an ASTIGMATIC carrier with `gap_kernel='auto'` RUNS under a trace
  (max|grad| = 2.0008 on this fixture), because the kernel clause is guarded by
  `Ax == Ay` and there is no exact arm to resolve to.  The docstring now says
  so and a new id gates BOTH halves -- the traced arm runs, and the eager arm
  resolves the same configuration to `'fresnel'` anyway, which is why accepting
  it is benign rather than a silent default.
* **V-D13**: six scalar arguments are checked up front and refused by name.
* **V-D12**: a concrete `jnp` array CLOSED OVER by a jitted function is not a
  Tracer, but the measurement transform's OUTPUT is, so that is where the check
  went.  Both directions gated: the `jnp` constant is refused by name, a
  closed-over NUMPY constant still runs (measured 88.357... under `jit`).

### V-D14 -- CLOSED by citation, not by duplication

The `bld` threading in `_collins_axis_chirp` and `_as_c_order`'s contiguity are
invisible to every VALUE test, and that is measured: `_backend_of` returns
`bld = np` for NumPy AND for JAX, so only CuPy would differ, and `np.asarray` /
`np.ascontiguousarray` agree on values for every input.  They are closed by the
verification's own `test_the_axis_chirp_builds_on_the_bld_it_is_handed` and
`test_as_c_order_makes_the_numpy_path_contiguous`.  One gate per claim: the
H2-2 test file's docstring now points at them.

### V-D15 -- CLOSED.  Seven docstrings naming a release that does not exist

Dropped from all seven; the CHANGELOG carries the number, which is the practice
the repository's last five releases follow (0, 0, 0, 1 false positive, 0
mentions at the release commit's PARENT; the release commit touches exactly one
library file).  GATED by
`test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached`,
which exempts deprecation horizons and carries a falsification arm.  A token
qualifies only when it cannot be a float (three dotted components, or a `v`
prefix): MEASURED 1181 qualifying tokens across `lumenairy/`, 0 forward.

### V-D17, V-D18 -- CLOSED

`_bluestein_centred_2d` validates `sign` and then `method` before it reads
`E.shape`, so both primitives report the same first error; gated over three
`(sign, method)` combinations.  The three public MFT docstrings are widened to
the full `_SUM_METHODS`.

### V-D19 -- RESTATED, and the non-reproducing red is not chased

`b4::TestGateCTwoGroupChain::test_the_two_transports_agree_on_this_chain` is
split into a PREMISE and a CLAIM.  Premise: every Collins leg took the
transfer-function form AND its decision reading is clear of the threshold --
MEASURED on this fixture `K1 = 1.9167`, `K3 = 2.1387`, margin **2.139**, 114 %
above the threshold of 1, against a premise bar of 1.5.  Claim: the two fields
agree to `1e3 * eps` of the peak, a derived re-association bound for a
2048-point pairwise reduction.  MEASURED: **exactly 0.0, bit-identical**
(`r2_b4chain_win.json`).  The failure mode it must catch -- a quadrature switch
-- appeared once at 8.5e-06 of peak, 7.6 decades above the bar.  The
non-reproducing red is not chased and not rerun away; the shape that invited it
is gone.

### V-D20, V-D21 -- CLOSED

The probe harness's tree guard catches `ValueError: Paths don't have the same
drive` and says `WRONG TREE` instead (both copies, since they are two copies of
one function).  The lens facade's refusal message now says the read "is
unaffected by this refusal; note that it returns the object bound at IMPORT
time, so read it from `_lens_kernels` while a substitute is in place" --
verified live: while `_lens_kernels.X` holds a substitute, `lenses.X` returns
the ORIGINAL.

### The maintainer question -- ANSWERED, MEASURED, AND NOT SHIPPED

`gap_kernel='auto'` does NOT gain an accuracy-keyed fallback.  The condition is
in the source behind `carrier._GAP_KERNEL_ACCURACY_TAU = None`, which is one
line to arm and which, left alone, is not evaluated at all -- part of the
245/245.

The law it is keyed on is re-derived and re-measured on both fixtures:

```
departure_relL2 = sqrt(3/2) * k |z_eff| theta_env^4 / 8
```

with `theta_env` read from the envelope's own spectrum as `2 sqrt(<theta^2>)`,
which is exact for a Gaussian.  MEASURED, law vs the thing it predicts:

| fixture | rung | `z_eff` | law | measured |
|---|---|---|---|---|
| hygiene-2 | 1 um from focus | 12.2666 m | 4.7162e-06 | 4.7136e-06 |
| hygiene-2 | 100 um | 3.0277 m | 1.1641e-06 | 1.1641e-06 |
| VERIFY-B4 F3 | 1 um | 1599.96 m | **2.3496e-03** | **2.3496e-03** |
| VERIFY-B4 F3 | 10 um | 159.96 m | 2.3490e-04 | 2.3491e-04 |

and the measured half-angle equals the analytic `lambda/(pi w)` to 16 digits on
F3 and to 7 on the hygiene-2 fixture.  `tau = 1e-4` leaves the hygiene-2 ladder
INERT (worst 4.7e-06, 1.3 decades under) and makes F3's 1 um and 10 um rungs
fall back while its 100 um rung does not (2.3438e-05).  An EXPLICIT
`gap_kernel='exact'` is honoured even over `tau`, because silently replacing
what the caller asked for is the D4 shape.  Three ids measure all of this; one
of them asserts the switch is OFF in the shipped source and that `stats_out`
does not grow a `kernel_departure` key while it is.

## What could not be measured, this round

* **CuPy on the device for anything that transforms.**  `CUPY_AVAILABLE` is
  True and `cupy.fft.fft2` raises `ImportError: DLL load failed while importing
  cufft`; WSL has no CuPy.  V-D3's CuPy row is therefore a statement about
  WHERE the failure now happens (inside cuFFT rather than at a host conversion
  in `carrier.py`), which is the observable that moved, not a device run.
* **Whether V-D19's red is a branch regression.**  Unchanged from the
  verification: it appeared once in four sweep-scale runs, did not reproduce,
  and both prefixes ending at it pass.  The bisect that would settle it is ~12
  hours of machine time and was not attempted.  The assertion's SHAPE is fixed
  either way.
* **A non-paraxial oracle for the near-focus law.**  Unchanged.  The analytic
  Gaussian is paraxial, so it measures how far the exact kernel departs from
  the paraxial truth and cannot say which kernel is more physical -- which is
  half of why the `tau` rule is not shipped armed.
* **An uncontended Windows timing ladder.**  Unchanged; the crossover is a
  ratio taken under the same conditions for all three routes and is what the
  decision rests on.
