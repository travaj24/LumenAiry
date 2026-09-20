# VERIFY-WAVE5-HYGIENE2 ROUND 2 -- independent adversarial verification

**Subject.** The numerical and dispatch changes of Wave 5 hygiene part 2 round
2: branch `refactor/wave5-hygiene-2-round2`, 15 commits `0a75c5ad..936a7ca3`
on `112c3049`, merged into `wave5/audit-leftovers` at `3fce9ecb`.
**Author's report.** `WAVE5_HYGIENE2_REPORT.md`, addendum "Round 2
(VERIFY-WAVE5-HYGIENE2) -- 2026-09-19", and `VERIFY_WAVE5_HYGIENE2.md`.
**This document.** What I measured myself, on my own fixtures, on both builds.
**Probes and JSON.** `validation/probe_verify_hyg2_round2/`.
**New decision tests.** `tests/unit/test_verify_hyg2_round2.py` (7 ids, 2.2 s).
**Verification branch.** `verify/wave5-hygiene-2-round2`.

## How this verification was run

* **My own worktree**, `git -C /c/tmp/lum_wave5 worktree add -b
  verify/wave5-hygiene-2-round2 C:/tmp/lum_vhyg3 wave5/audit-leftovers`.
  Nothing under `lumenairy/` was edited; every mutation was applied to a
  scratch `git archive` export.
* **Both builds, every claim.**  Windows py3.14.6 (numpy 2.4.4, jax 0.11.0,
  CuPy 14.0.1 with a broken cuFFT DLL) and WSL py3.12.3 (numpy 2.4.6, jax
  0.10.2, no CuPy), with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1` and `LUMENAIRY_MEM_BUDGET_MB` on the COMMAND LINE, pytest
  with `--capture=sys`, and `lumenairy.__file__` printed and written into
  every JSON.
* **Re-measured, never read.**  Every number below came out of a probe I
  wrote.  Where a number of the author's is quoted it is marked as theirs.
  THREE of MY OWN readings were wrong and were caught the same way, and all
  three are written up rather than quietly fixed: a cross-backend spread
  measured through the wrong transform (D-5, demoted from MAJOR to a
  suggestion); a "no partial state" check whose caches were empty on both
  sides of the comparison (section 3); and a Windows mutation matrix that
  turned out to have been run by TWO concurrent harnesses on one scratch tree
  (section 8), which was re-run clean.
* **Two PRE trees**, both my own `git archive` exports: `112c3049` (the
  pre-round-2 tree) and `f4f18851` (before hygiene 2 at all).

## Verdict table

| # | claim | verdict | my numbers |
|---|---|---|---|
| 1 | the V-D22 consolidation is byte-identical | **CONFIRMED** | 342/342 keys identical `112c3049 -> 3fce9ecb`, WIN **and** WSL; 306/342 identical back to `f4f18851` (the 36 are the new `method=` keyword, a kind flip); control 294/342 differ WIN vs WSL |
| 2 | ONE kernel, and the sign is guarded | **CONFIRMED with one gap** | structural AST census: 3 owners at `112c3049`, **1** at the tip, all three sites call it once and unnegated. Negating the kernel: 15 ids fail (WIN) / 7 (WSL light). A SITE-LOCAL conjugation is still writable and IS caught (13 / 10 ids) -- the consolidation does not make it impossible, only visible. **The `(s.q)/N` term inside the kernel is gated by nothing (D-2).** |
| 3 | the public Collins leg on JAX | **CONFIRMED** | tip: `jaxlib._jax.ArrayImpl`, rel 5.486e-16 (WIN) / 4.029e-16 (WSL); base `112c3049`: `numpy.ndarray`, rel **exactly 0.0**. Traced -> designed `ValueError`, 10/10 needles (base: `TracerArrayConversionError`, 1/10). No state moved. Sziklas exact leg through the consolidated kernel: 5.055e-16 / 5.170e-16 (WIN), 4.224e-16 / 4.154e-16 (WSL) untilted / tilted. `jax.grad` at 0.0x..1.8x its own per-rung cancellation floor. CuPy: `TypeError` at a host boundary on the base, cuFFT's own `ImportError` on the tip. |
| 4 | the chirp phase-budget threshold | **CONFIRMED for the chirp-Z routes; the DENSE half is REFUTED (D-1)** | law slope 0.996 / 0.983 (my grids), 1.0045 (shipped geometry, exact-phase reference); threshold two-sided and `_PHASE_BUDGET_MAX * eps == 1e-6` exactly. **`method='direct'` obeys the SAME `eps*budget` law, slope 1.0038** -- 3.4e-05 at a budget of 1e12, not 3e-16 |
| 5 | the tau switch is OFF and inert | **CONFIRMED, and inert by EXECUTION** | `None` in the shipped source; a `sys.settrace` line trace runs the guard line and none of the rule's four lines; raising sentinels see zero calls. At `tau=1e-4`: hygiene-2 ladder all `'exact'` (worst 4.7161e-06), F3 1 um and 10 um `'fresnel'` (2.3496e-03, 2.3490e-04), 100 um `'exact'` (2.3437e-05); explicit `'exact'` honoured. Identical on both builds. |
| 6 | V-D19's restated premise and claim | **CONFIRMED** | 1 Collins stage, form `tf`, margin `max(K1,K3) = 2.1387` (bar 1.5); `max\|a-b\|` **exactly 0.0** against a derived 2.8307e-13; the 8.5e-06-of-peak quadrature signature reads 1.0836e-05 and fails by 7.58 decades; a 1-ULP re-association reads 2.4825e-16 and passes. Every one of those numbers is IDENTICAL on WSL |
| 7 | the S4 fix `936a7ca3` (the `P'''` bound) | **CONFIRMED and NECESSARY** | quadratic merit over its own floor: 0.362 / 1.45 / 0.724 / 0.362 (WIN), 0.362 / 1.09 / **0.000** / **0.000** (WSL) -- TWO exact zeros on WSL, not one. Cubic control on its own ladder 1.906e+04 .. 7.057e+08 |
| 8 | durability: mutation matrix, derived bars, environmentals | **CONFIRMED with three gaps** | 15 arms x 2 builds; 12 killed, **3 survive (M2, M11, M13)**. Every new bar derived; no cross-build pin; no `pytest.skip` on a resource. The WSL git-pointer condition reproduces verbatim in my worktree. **But two of round 2's OWN new gates are RED on the merged tip `3fce9ecb` on WINDOWS (D-A, D-B).** |

---

## 1. Byte identity (V-D22) -- CONFIRMED on 342 keys of my own

`validation/probe_verify_hyg2_round2/vh3_bitid.py` builds its own key set and
digests every reading to sha256 -- arrays by their RAW BYTES plus dtype and
shape, refusals by type plus message, `CarrierReferencedField` by envelope
bytes plus `R` plus pitch, stats dicts entry by entry.

| group | keys | what |
|---|---|---|
| `A.` | 119 | `_exact_envelope_tf_step`: 5 grids x 5 reduced distances x 4 tilts, 2 anisotropic pitches x 4 tilts, complex64 x 4 tilts, a real-amplitude input, 2 deterministic non-Gaussian grids, 4 evanescent-tilt refusals |
| `B.` | 53 | `_exact_tf_2d_xp` on NumPy (3 grids x 3 z x 2 tilts x 2 dtypes + a refusal) and on eager JAX (2 grids x 2 z x 2 tilts x 2 dtypes) |
| `C.` | 29 | `_collins_exact_kernel_correction`: 3 grids x 3 `z_eff` x 3 tilts, anisotropic pitch, refusal |
| `D.` | 27 | `_collins_transport`: 3 carriers x 2 distances x 3 `gap_kernel` with their stats dicts, tilted, astigmatic, astigmatic-exact refusal, 3 output lattices x 2 output references |
| `E.` | 44 | the PUBLIC `propagate_carrier_referenced`, both transports x 3 carriers x 3 distances x 2 kernels, explicit `'exact'`, 2 tilts, complex64 |
| `F.` | 22 | a 9-rung near-focus ladder across the geometric focus on `sziklas` (x2 kernels) and 4 rungs on `collins` |
| `G.` | 12 | `carrier_referenced_focus_readout` (3 kernels x 2 focal lengths + 2 tilts) and `carrier_referenced_exact_focus_readout` |
| `H.` | 36 | `_bluestein_centred_2d` on all four `method` values x 3 shapes x 3 alphas |

| comparison | keys | identical | differing | only-A | only-B |
|---|---|---|---|---|---|
| `112c3049` -> `3fce9ecb`, WIN py3.14 | 342 | **342** | 0 | 0 | 0 |
| `112c3049` -> `3fce9ecb`, WSL py3.12 | 342 | **342** | 0 | 0 | 0 |
| `f4f18851` -> `3fce9ecb`, WIN py3.14 | 342 | 306 | 36 | 0 | 0 |
| `f4f18851` -> `3fce9ecb`, WSL py3.12 | 342 | 306 | 36 | 0 | 0 |
| **control**: WIN vs WSL on the tip | 342 | 48 | **294** | 0 | 0 |

The control is the row that makes the others mean something: the same
instrument separates the two builds on 86 % of the keys and cannot separate
the two trees on one.  The 48 cross-build agreements are the 9 refusal
digests plus the 32x32 and complex64 rungs, where the last bits are rounded
away before the digest sees them.

The 36 that differ against `f4f18851` are all in group `H` and are all KIND
FLIPS (`TypeError` -> value): the `method=` keyword did not exist before
hygiene 2.  That is the intended API addition, and groups `A`-`G` -- every
carrier reading -- are identical all the way back to before hygiene 2.

**The author's 245/245 is reproduced and extended.**  My key set is 40 %
larger, is built independently, and includes arms theirs does not (a
real-amplitude input, two non-Gaussian grids, the public entry's complex64
arm, both focus readouts with three kernels, and the chirp-Z primitive on all
four routes).

### 1a.  What the key set does NOT reach (measured, not assumed)

Two arms of the kernel are not reached by ANY of my 342 keys, so a mutation
that deletes them moves none of them.  (One of the two, the clamp, IS reached
by two shipped near-focus ids -- see section 8 -- so the gap is in my
instrument, not in the suite.)

* the **evanescent clamp** bites only when `pi/dx > k`, i.e. `dx < lambda/2`;
  every fixture in MY key set has `dx` of micrometres against `lambda =
  1.55 um`, so `q_max/k = 0.25`.  `validation/probe_verify_hyg2_round2/
  vh3_edge.py` puts the pitch at `lambda/4` and `2 lambda/5`: with the clamp
  deleted the kernel and the step both return NaN, with it they are finite.
  Two shipped ids DO catch this mutant (section 8), so the gap is in the
  byte-identity instrument, not in the suite; the new id gates it directly,
  with the band crossing asserted as its premise.
* the **complex64 `mod 2*pi` fold** in `_tf_phase_to_H` is on the CuPy / JAX
  branch only and guards a float32 cast that happens AFTER `cos`/`sin`.  Host
  `cos`/`sin` (NumPy and JAX-on-CPU) do their own Payne-Hanek reduction, so
  deleting the fold is unobservable: at `k z = 4.05e+08` the complex64-vs-
  complex128 difference reads 1.4958e-07 with the fold and 1.4685e-07 without.
  **I could not measure this arm** on either build; it would need a working
  CuPy FFT.

---

## 2. ONE kernel (V-D22) -- CONFIRMED, with one gate missing

### 2a.  The census, written from the other end

`vh3_census.py` walks the AST of every module under `lumenairy/` (4546
function definitions on the tip), strips docstrings by re-parsing each body
from `ast.unparse`, and asks a STRUCTURAL question the shipped token census
does not: which functions subtract FROM `k**2`, as a `BinOp` or through
`np.subtract`?  A re-spelled copy is still counted.

| tree | carrier.py functions carrying the shifted-frequency radical |
|---|---|
| `112c3049` | `_exact_tf_2d_xp`, `_exact_envelope_tf_step`, `_collins_exact_kernel_correction` -- and all three own `root0`, `s2 < 1.0` and `ax * ax + ay * ay` |
| `3fce9ecb` | **`_exact_dispersion_phase` only** |

`_exact_dispersion_phase` is called from exactly three functions, once each,
and the unparsed call expressions carry no unary minus.  (`asm.py` x4,
`through_focus.py` x2, `mft.py` x1 and `jax_merits.py` x1 carry the PLAIN
`k^2 - q^2` ASM dispersion; that is a different kernel, not a fourth
transcription, and it is unchanged between the trees.)

### 2b.  The sign, by mutation

| arm | WIN, 5 files / 216 ids | WSL, light selection / 108 ids |
|---|---|---|
| M1: the ONE kernel negated | 15 failed | 7 failed |
| M3: ONE call site conjugated (`-_exact_dispersion_phase(...)` in `_collins_exact_kernel_correction`) | 13 failed | 10 failed |

Both are caught.  **The consolidation does not make a site-local conjugation
impossible** -- it is four characters, and it moves 63 of my 342 keys -- but
it does make it a single, visible edit, and 13 ids see it.  A structural gate
for it is now `test_verify_hyg2_round2.py::
test_no_call_site_negates_the_one_kernel`.

### 2c.  The author's M-A row, re-measured directly

| tree | `b4::TestSameTheorem` (9 ids) |
|---|---|
| the merged tip, unmutated | 9 passed |
| M1, the ONE consolidated kernel negated | **9 passed** |
| M3, ONE call site conjugated | 2 failed, 7 passed |

The author's central warning about what the consolidation costs is
**reproduced exactly**: with one kernel, both transports move together and a
cross-IMPLEMENTATION agreement test cannot see a sign flip of it.  A
conjugation of ONE site does still break that agreement, which is why M3 is
caught there as well as by the sign pins.

So the task brief's phrasing -- that negating the kernel should fail BOTH
`b4::TestSameTheorem` and the new sign pin -- is **refuted for the b4 half and
confirmed for the sign pin**, and the report says so itself.  What fails under
M1 is `test_wave5_h2_near_focus_table.py::
test_the_measured_departure_is_the_quartic_times_one_constant` (V-D4's
addition) and `test_verify_wave5_hyg2.py::
test_the_exact_kernel_retards_the_envelope_and_the_sign_is_gated`, together
with the departure-law ids around them.

---

## 3. The public Collins leg on JAX (V-D3) -- CONFIRMED

`vh3_jax.py`, the same probe run with `PYTHONPATH` pinned to each tree.

| reading | `112c3049` | `3fce9ecb` |
|---|---|---|
| eager JAX in, type out (WIN) | `numpy.ndarray` | `jaxlib._jax.ArrayImpl` |
| eager JAX in, type out (WSL) | `numpy.ndarray` | `jaxlib._jax.ArrayImpl` |
| JAX vs NumPy relative L2 (WIN) | **0.0** (bitwise -- both arms were NumPy) | 5.4861e-16 |
| JAX vs NumPy relative L2 (WSL) | **0.0** | 4.0289e-16 |
| traced envelope | `TracerArrayConversionError`, 1/10 needles | the designed `ValueError`, **10/10** needles |
| eager CuPy in (WIN, CuPy 14.0.1) | `TypeError: Implicit conversion to a NumPy array is not allowed` | `ImportError: DLL load failed while importing cufft` |

The CuPy row is the author's and it is reproduced here independently
(`vh3_cupy_reach.py`).  It is a statement about WHERE the failure now happens,
not a device run: on the tip the public leg's error is **character for
character the error a bare `cupy.fft.fft2(d)` raises on this box**, which is
the observable that moved -- the array reached the transform instead of being
converted at a host boundary inside `carrier.py`.

The ten needles asserted are `_collins_transport`, `dx_out`, `gap_kernel`,
`on_collins_sampling`, `Tracer`, `jax.jit`, `jax.grad`,
`gap_kernel='fresnel'`, `on_collins_sampling='ignore'` and `outside the
trace`.

**No partial state after the refusal.**  The chirp-Z kernel cache and its hit
counter, the pyFFTW plan cache, the ASM `H` cache and `warnings.filters` are
identical before and after, on both builds, and a caller-supplied `stats_out`
sentinel is untouched.

The priming took a correction of my own.  MEASURED 2026-09-20: a Collins leg,
a Sziklas leg and an exact-kernel leg leave **all four of those at zero** --
the carrier chain does not fill them; the MFT propagators do.  Priming through
the carrier leg, as my first cut did, would have made three quarters of this
reading the vacuous "empty before, empty after".  The probe now primes with
one `angular_spectrum_propagate_mft` call, which fills all four (chirp-Z
kernel cache 1, pyFFTW plans 2, ASM `H` 1, `warnings.filters` 10 on both
builds), and every one of them is then unchanged across the refusal.

**The bar.**  The shipped `_fft_spread_bar` measures one forward transform
through each backend, multiplies by a chain depth of 6, and floors the result
at `32 * eps`.  MEASURED here on the shipped fixture, both builds:

| quantity | WIN py3.14 | WSL py3.12 |
|---|---|---|
| library `_fft2` vs `jnp.fft.fft2` | 2.685e-16 | 2.509e-16 |
| `np.fft.fft2` vs `jnp.fft.fft2` | **exactly 0.0** | **exactly 0.0** |
| `_fft_spread_bar(env)` | 7.105427357601002e-15 | 7.105427357601002e-15 |
| is that `32 * eps` exactly? | **yes** | **yes** |

So the FLOOR is what is asserted against, by 4.41x over `6 x` the measured
spread -- **which is the ratio the author's own V-D16 entry records**, and my
measurement confirms it rather than correcting it.  The extra detail is where
the nonzero reading comes from: NumPy's and JAX's own transforms agree BIT FOR
BIT at this shape, and the entire 2.5-2.7e-16 is pyFFTW sitting on the library
side.  D-5 below is therefore a suggestion and not a correction.

A quantity that is a property of the LEG is its own response to a 1-ULP change
of its input: 3.454e-16 (WIN) / 3.469e-16 (WSL), against measured
cross-backend differences of 5.486e-16 / 4.029e-16.  The two-sided half is
comfortable either way: the smallest real signal on the fixture (the
exact-vs-paraxial kernel departure) is 3.078e-05, 9.95 decades above.

**The Sziklas exact leg through the consolidated kernel.**  5.055e-16
untilted / 5.170e-16 tilted (WIN) and **4.224e-16 / 4.154e-16** (WSL) -- the
author's claimed 4.2e-16 / 4.1e-16 reproduced to two digits on the WSL lane,
on my own fixture.  These readings are IDENTICAL on `112c3049` and the tip,
which is independent corroboration of the byte-identity claim.

**`jax.grad`.**  Through the PUBLIC Collins leg a gradient does not exist --
that leg refuses a trace by design -- so it is measured on the two legs that
do exist: the public SZIKLAS leg at `gap_kernel='exact'` (i.e. through the
consolidated kernel) and the private `_collins_transport`.  Seven rungs,
`h = 1e-1 .. 1e-4`, with the floor derived PER RUNG from the cancellation
that rung actually suffers -- `eps * max(|P+|,|P-|) / (2 h |g|)`, a reading of
its own two evaluations, not `eps^(2/3)`:

| build | leg | disagreement / its own floor, worst rung |
|---|---|---|
| WIN | public sziklas, exact | 0.6x |
| WIN | private collins | 1.1x |
| WSL | public sziklas, exact | 1.8x |
| WSL | private collins | 1.1x |

The gradient is non-zero and non-constant on every arm.

---

## 4. The chirp phase budget (V-D5) -- the law CONFIRMED, "dense is immune" REFUTED

### 4a.  The law, on my grids

`vh3_budget.py`, against a `math.fsum` reference, 11 budgets over 11 decades,
two geometries (20x10 and 28x14), both builds.  Fitted slope of
`log10(rel)` against `log10(budget)`: **0.996** and **0.983**.  The threshold
is two-sided and its derivation is exact:

| | budget | warned? | chirp-Z rel | six figures? |
|---|---|---|---|---|
| a decade below | 4.504e+08 | no | 5.295e-08 | yes |
| just below (0.9x) | 4.053e+09 | **no** | 5.055e-07 | yes |
| just above (1.1x) | 4.954e+09 | **yes** | 5.699e-07 | yes |
| a decade above | 4.504e+10 | yes | 9.386e-06 | no |

`_PHASE_BUDGET_MAX * eps == 1e-6` to the bit, and the warning message names
the budget, the implied error and `method='direct'`.

**The routes' arithmetic is untouched -- measured AT the budgets where the
warning differs.**  The 36 `H.` keys of the main byte-identity set all sit
five decades under both thresholds, so they cannot see a threshold change
either way.  `vh3_highbudget.py` therefore digests both primitives
(`_bluestein_2d` and `_bluestein_centred_2d`) x all four `method` values x two
shapes x budgets 1e8, 1e10, 1e12, 1e15 -- 64 value keys and 64 warning
counts -- archive-to-archive:

| | keys | differing |
|---|---|---|
| VALUE (the returned arrays, raw bytes), WIN | 64 | **0** |
| VALUE, WSL | 64 | **0** |
| WARNING COUNT, WIN | 64 | **36** |
| WARNING COUNT, WSL | 64 | **36** |

So the change is precisely and only a warning change, and it is one at the
budgets it was meant to be one at.  Identical on both builds.

### 4b.  D-1 -- the dense route is NOT immune, and the reference cannot see it

`_direct_matrix_2d` forms `t = alpha * n * k` in float64 and then reduces it
by `t - rint(t)`.  The second step is exact (Sterbenz).  The FIRST has already
discarded the low bits of a product that needs about 63 of them, and no later
reduction recovers a bit that is gone.  So the dense route's phase error is
`~eps * alpha * n * k <= eps * budget` -- the same law, with a smaller
constant.

`tests/unit/test_wave5_h2_mft_direct.py::_fsum_reference` forms `t` the SAME
way (`ty = alpha * ky * n_y`, then `ty - np.rint(ty)`).  `math.fsum` makes the
SUMMATION correctly rounded -- which is what that reference is for and is a
real improvement over the pairwise one -- but it leaves the reference's PHASE
exactly as wrong as the dense route's, so the two agree by construction.

`vh3_budget_exact.py` reduces the phase EXACTLY with `fractions.Fraction`
(`alpha` is a float64 and therefore an exact rational; `n` and `k` are
integers; the single float64 rounding happens on a number already inside
`[-1/2, 1/2)`).  On the SHIPPED geometry N=24 M=12, **identical to the digit
on both builds**:

| budget | chirp-Z vs EXACT | dense vs EXACT | `eps*budget` | dense vs NAIVE reference |
|---|---|---|---|---|
| 1e5 | 1.755e-11 | **2.292e-12** | 2.220e-11 | 3.379e-16 |
| 1e7 | 1.896e-09 | **2.008e-10** | 2.220e-09 | 3.288e-16 |
| 1e9 | 1.878e-07 | **2.751e-08** | 2.220e-07 | 3.143e-16 |
| 4.5036e9 | 9.186e-07 | **1.465e-07** | 1.000e-06 | 3.207e-16 |
| 1e12 | 1.484e-04 | **3.401e-05** | 2.220e-04 | 3.561e-16 |
| 1e15 | 1.834e-01 | **1.960e-02** | 2.220e-01 | 3.452e-16 |

Fitted slope: chirp-Z **1.0045**, dense **1.0038**.  The last column is the
reading the report quotes ("2.8e-16 .. 4.5e-16 at EVERY budget tested"); it is
a measurement of the instrument.

**What is wrong in the shipped tree, precisely:**

1. `lumenairy/propagators/_bluestein.py`, `_bluestein_2d` Notes: "The dense
   route (`method='direct'`) reads 2.8e-16 .. 4.5e-16 at EVERY budget tested,
   because it reduces `t` by `t - rint(t)` before calling `exp` and therefore
   has no chirp phase to lose."  The premise is true; the conclusion is not.
2. `lumenairy/propagators/_bluestein.py`, the WARNING MESSAGE a caller
   actually sees: "pass `method='direct'` (the dense route reduces its phase
   modulo one turn and measured 3e-16 at every budget tested)".  At the budget
   that triggers the warning the dense route is not at 3e-16; at 1e12 it is at
   3.4e-05.  This is caller-facing advice that does not deliver, and it is the
   REMEDY the warning offers.
3. `_direct_matrix_2d`'s docstring: "only a fractional turn reaches `exp`, so
   the `pi*alpha*N^2` phase-budget concern does not apply to this route".
4. `tests/unit/test_wave5_h2_mft_direct.py::
   test_the_chirp_phase_error_is_linear_in_the_budget_and_dense_is_immune` and
   `tests/unit/test_verify_wave5_hyg2.py::
   test_the_chirp_phase_error_is_linear_in_the_budget_and_dense_is_immune`
   assert `rd < 1e-14` against that reference.  The assertion cannot fail for
   any dense route that reduces its phase the way the reference does.

The LAW and the THRESHOLD are unaffected -- the dense route being 7x better
rather than 10 decades better does not move `1e-6/eps`.  What is affected is
the sentence that tells a caller what to do about it.

**Reproducer.** `validation/probe_verify_hyg2_round2/vh3_budget_exact.py`;
gated by `tests/unit/test_verify_hyg2_round2.py::
test_a_phase_budget_reference_must_not_reduce_its_phase_like_the_route`.

---

## 5. The near-focus accuracy switch -- CONFIRMED, and inert by execution

`vh3_tau.py`.

* **`None` in the SHIPPED SOURCE**, read from the file rather than the
  imported module, so a conftest that armed it could not hide.
* **INERT, PROVED BY EXECUTION.**  `sys.settrace` over three legs (`'auto'`,
  `'exact'`, `'fresnel'`) executes 207 lines of `carrier.py`.  The GUARD line
  (2450, `... and _GAP_KERNEL_ACCURACY_TAU is not None`) IS among them -- which
  is what makes the trace live rather than vacuous -- and the rule's own lines
  (2451 the half-angle measurement, 2452 the departure, 2454 the comparison,
  2474 the extra stats key) are not.  Independently, both helpers replaced by
  raising sentinels: **zero calls**, and `stats_out` never grows
  `kernel_departure`.  Identical on both builds.
* **At `tau = 1e-4`**, identical to five digits on both builds:

| fixture | rung | kernel | departure |
|---|---|---|---|
| hygiene-2 | 1 um | `exact` | 4.7161e-06 |
| hygiene-2 | 10 um | `exact` | 3.6956e-06 |
| hygiene-2 | 100 um | `exact` | 1.1641e-06 |
| hygiene-2 | 1 mm | `exact` | 1.4185e-07 |
| hygiene-2 | 5 mm | `exact` | 2.2960e-08 |
| VERIFY-B4 F3 | 1 um | **`fresnel`** | 2.3496e-03 |
| VERIFY-B4 F3 | 10 um | **`fresnel`** | 2.3490e-04 |
| VERIFY-B4 F3 | 100 um | `exact` | 2.3437e-05 |
| VERIFY-B4 F3 | 1 mm | `exact` | 2.2909e-06 |

  The ladder is inert (worst 4.7e-06, 1.3 decades under `tau`), F3's two near
  rungs fall back and its 100 um rung does not, and an EXPLICIT
  `gap_kernel='exact'` is honoured over an armed `tau` (F3 at 1 um, departure
  2.3496e-03, kernel `exact`).  `gap_kernel='fresnel'` stays `fresnel`.
* `tau` is restored to `None` by the probe and the module attribute is
  re-read afterwards.

---

## 6. V-D19 -- CONFIRMED on the id's own fixture

`vh3_v19_s4.py` calls `test_audit2609_b4_collins_transport.py::p5_arms`
directly (`__wrapped__`), so the numbers are the id's own and not a
re-implementation.

| reading | measured | the id's bar |
|---|---|---|
| Collins stages with a published form | 1 | -- |
| forms | `{'tf'}` | must be `{'tf'}` |
| `max(K1, K3)` | **2.1387** | premise `> 1.5` |
| peak | 1.274820e+00 | -- |
| `max\|a-b\|` | **0.0** exactly | `<= 1e3 eps peak = 2.8307e-13` |
| the quadrature-switch signature at 8.5e-06 of peak | 1.0836e-05 | **fails** by 7.58 decades |
| a 1-ULP re-association | 2.4825e-16 | **passes** |

Both the premise and the claim hold, and both sides of the bar are probed:
the failure mode it exists to catch fails it by 7.6 decades and the
re-association it must tolerate passes it by 3 decades.  The premise reading
2.1387 reproduces the author's exactly.

**BOTH BUILDS, identical to every digit printed** -- WSL py3.12 reads the same
`{'tf'}`, the same 2.1387, the same peak 1.274820e+00, the same
`max|a-b| = 0.000000e+00`, the same 1.0836e-05 and the same 2.4825e-16.  For a
2048x2048 traced-lens chain through two groups that is a strong statement in
its own right, and it is the opposite of the S1/S5 shape the restatement was
written to remove.

---

## 7. The S4 fix `936a7ca3` -- CONFIRMED, NECESSARY, and one label is wrong

`|P'''_est| / (eps |P| / h^3)`, the shipped quadratic merit
(`_merit_factory(gap_kernel='fresnel')`), bar 100:

| h | WIN py3.14 | WSL py3.12 |
|---|---|---|
| 1e-1 | 0.362 | 0.362 |
| 1e-2 | 1.45 | 1.09 |
| 1e-3 | 0.724 | **0.000** (exactly) |
| 1e-4 | 0.362 | **0.000** (exactly) |

The bound holds two-sidedly: the shipped cubic control
(`(|E(N/2,N/2)|^2)^3`) on ITS OWN ladder reads **1.906e+04 .. 7.057e+08**,
reproducing the report's 1.9e+04 .. 7.0e+08 exactly, and a second cubic
control of my own (`P^3`) reads 6.99e+10 .. 71.1.

The fix is **necessary**: on WSL the third difference is exactly 0.0 at TWO
rungs, not one, so the ratio form the round first shipped would have divided
by zero twice.  A bound is the right restatement and it is what landed.

Two small corrections (D-3, D-4) are in the defect list.

---

## 8. Durability -- the mutation matrix

Fifteen arms, each editing ONE shipped expression in a scratch `git archive`
export of the merged tip (never the worktree; every arm is restored from a
pristine copy before the next one).  Windows runs the five touched files in
full, 216 ids.  WSL runs a LIGHT selection, 108 ids -- the four hygiene-2 /
verify files plus `b4::TestSameTheorem`, `::TestKernelRefinement` and
`::TestQuadratureComplementarity` -- because this box was carrying another
agent's four-worker pytest sweep throughout and the full selection advanced at
about 6 % of one core there; `b4`'s four 2048x2048 gate classes are 235 s of
its 308 s and none of them reads the exact kernel.  V-D19's own numbers are
taken on WSL by `vh3_v19_s4.py` instead.

`validation/probe_verify_hyg2_round2/vh3_mutate.py`, JSON in
`mutation_win.json` / `mutation_wsl.json`.

**The Windows matrix was run TWICE, and the first pair of results was thrown
away.**  An early launch of the harness that I had recorded as dead ("the
background job produced no log, so it was killed with its shell") had in fact
survived, and ran the whole matrix CONCURRENTLY with the launch I was
watching -- both on the same scratch tree, each restoring and re-mutating the
files the other's pytest was importing.  I caught it by noticing that the
JSON's per-arm wall times (M0 310.60 s) did not match the console's
(M0 318.63 s), which two views of ONE run cannot do, and then finding the
stray log.  The two contaminated runs agreed on 14 of 15 arms and disagreed on
M5 (2 failed vs 4 failed), which is exactly the kind of silent, small
divergence a shared mutation tree produces.  The numbers below are from a
single clean process with nothing else of mine running; the contaminated JSON
is not committed.

| arm | what it breaks | WIN 216 ids | WSL 108 ids |
|---|---|---|---|
| M0 | (the unmutated tip -- the control) | 216 passed | 108 passed |
| M1 | the ONE kernel returns the conjugate phase | 15 failed | 7 failed |
| M2 | the `(s.q)/N` chief-ray subtraction is dropped | **0** | **0** |
| M3 | ONE call site conjugates the kernel | 13 failed | 10 failed |
| M4 | the `q = 0` value `k N` is left in | 10 failed | 7 failed |
| M5 | the evanescent band is not clamped | 2 failed | 2 failed |
| M6 | the `\|s\|^2 < 1` refusal admits `\|s\| = 1` | 1 failed | 1 failed |
| M7 | the phase budget reverts to 1e15 | 1 failed | 1 failed |
| M8 | the phase budget drops a decade below its derivation | 1 failed | 1 failed |
| M9 | the near-focus rule ships ARMED at `tau = 1e-4` | 1 failed | 1 failed |
| M10 | the public leg demotes an eager JAX array again | 1 failed | 1 failed |
| M11 | the input-box transform goes back to a host FFT | **0** | **0** |
| M12 | the designed traced `ValueError` is removed | 1 failed | 1 failed |
| M13 | the complex64 `mod 2*pi` fold is dropped | **0** | **0** |
| M14 | the phase-budget warning never fires | 2 failed | 2 failed |
| M15 | an armed `tau` would downgrade an explicit `'exact'` | 1 failed | 1 failed |

Twelve of fifteen are killed on both builds, and the two lanes agree on the
VERDICT for every single arm -- twelve killed, the same three alive -- which
is the strongest thing a 15-arm matrix can say about itself.  The three
survivors are D-2 (M2, a real gap) and D-6 (M11 and M13, both unobservable on
the backends available here).

The clean lane also settled the one disagreement between the two contaminated
runs: **M5 reads 2 failed**, not 4, matching WSL.

**Which mutants move bytes.**  A mutation that no test sees is only a finding
if it is LIVE, so each arm was also run through the 342-key byte-identity
probe:

| arm | keys moved (of 342) |
|---|---|
| M2 | **136** |
| M3 | 63 |
| M6 | 1 (a kind flip: the `\|s\| = 1` refusal becomes a value) |
| M5 | 0 -- no fixture of MINE reaches the band edge (section 1a); two shipped ids do, and catch it |
| M13 | 0 -- unobservable on a host `cos`/`sin` (section 1a) |

So M2 is a large, live numerical change with no gate.  M5 and M13 are
invisible to the byte-identity instrument, but M5 IS caught by two shipped ids
on both builds (`test_wave5_h2_near_focus_table.py::
test_the_exact_kernels_departure_on_a_collimated_leg_is_the_quartic` and
`::test_the_measured_departure_is_the_quartic_times_one_constant`), and the
arm's warning count rises from 4 to 6 -- i.e. at least one shipped fixture
does reach the band edge and produces `invalid value encountered in sqrt`
without the clamp.  My key set simply does not contain such a grid, which is
the gap section 1a records.

### 8a.  The author's M-A / M-B table

Reproduced exactly; see section 2c.

### 8b.  Testing-standards compliance of what round 2 added

| rule | verdict |
|---|---|
| derived-at-runtime bars, no prior version's numbers pinned | HOLDS -- `_PHASE_BUDGET_MAX` from `1e-6/eps` with the derivation asserted; the V-D6 shape bar from a measured `r`; the V-D7 floor from the rung's own `eps\|P\|/(h\|g\|)`; V-D19's from `1e3 eps peak` |
| bars with a gap on BOTH sides | HOLDS for every new bar I could re-measure (V-D19 7.6 decades; V-D4 0.7 decades stated and measured 0.002 %/0.04 %; the S4 bound 1.7-1.8 decades) -- with the two exceptions D-4 (a ladder-conditional separation, stated too broadly) and D-5 (a bar that reduces to its own floor, as V-D16 records) |
| no cross-build pin | HOLDS -- every reading I took is either identical on both builds or explicitly two-valued |
| never `pytest.skip` on a resource | HOLDS -- the environmental reds FAIL rather than skip, which is the repository's own rule and is why I could see them |
| the instrument is not blind | HOLDS -- the byte-identity control separates two builds on 294/342 keys |

### 8c.  The "environmental" WSL reds, reproduced

| id | WSL, merged tip | WSL, `112c3049` archive | WIN, merged tip |
|---|---|---|---|
| `test_public_api.py::test_installed_metadata_version_matches_source_version` | FAIL (a stale editable install; I did not read the venv's number) | FAIL | **FAIL** in my env (installed dist-info 5.47.0 against source 5.47.1, a global editable install); passes on the `112c3049` archive because that tree's `__version__` IS 5.47.0 |
| `test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught` | FAIL | **SKIP** ("git not available") | pass |
| `test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines` | FAIL (ENVIRONMENT message) | pass | **FAIL -- a REAL finding, D-B** |
| `...::test_v18_5_companion_reanchor_tool_exists_and_covers_the_cited_files` | FAIL (ENVIRONMENT message) | pass | pass |

The mechanism is reproduced verbatim in MY worktree: from WSL,

```
$ wsl -e bash -lc 'cd /mnt/c/tmp/lum_vhyg3 && git log --oneline -1'
fatal: not a git repository: /mnt/c/tmp/lum_vhyg3/D:/Metacept/.../worktrees/lum_vhyg3
```

-- a Windows worktree's `.git` file holds a Windows absolute path, and from
Linux that is a relative path under the worktree.  Every `git show` fails, so
every citation gate fails.

Two refinements to the report's account:

* on a plain `git archive` export (no `.git` at all)
  `test_v16_synthetic_fabrication_is_caught` **SKIPS** rather than fails --
  "git not available" is a different branch from "git present, repository
  unreachable".  The addendum's "on the base tree as well" is true of a base
  WORKTREE and not of an archive.
* the metadata id is environmental on Windows too, in my environment, for the
  same reason (a stale global editable install).  The addendum's "All four are
  green on Windows" is a statement about the author's Windows environment, not
  about the tree.

### 8d.  Two of round 2's own new gates are RED on the merged tip

This is the finding the round-2 report could not have made, because it
measured on the branch: see **D-A** and **D-B**.  Both are green on
`936a7ca3` and red on `3fce9ecb`, Windows py3.14, and in both cases the gate
that fires is one round 2 itself added.

## 9. Defects

Severity: **BLOCKER** = do not ship the merge as it stands; **MAJOR** = a
claim that does not hold or a gate that does not gate; **MINOR** = a number or
a sentence that is wrong beside a conclusion that is right; **SUGGESTION** /
**OBSERVATION** = nothing is wrong, and here is a measurement the next round
may want.

D-5 began as a MAJOR finding of mine and was demoted by re-measuring it: the
first cut used `np.fft.fft2` where the shipped helper uses the library's own
`_fft2`, and the two are not the same transform.  The author's number was
right and mine was an artefact of my instrument -- recorded here because
"re-measure, do not read" cuts both ways.

### D-A -- BLOCKER (merged state).  Round 2's own version gate is RED on `3fce9ecb`

`tests/unit/test_public_api.py::
test_no_shipped_source_claims_a_version_the_package_has_not_reached` is the
gate V-D15 added.  On the round-2 branch tip `936a7ca3` it reads **1 passed**.
On the merged integration tip `3fce9ecb`, **Windows py3.14**, it reads
**1 failed**:

```
2 shipped source line(s) name a version later than lumenairy.__version__ = 5.47.1
  lumenairy/propagators/gbd.py:3622 names 5.48.0 -- 'returned silently (VERIFY-WP-B12b D-4 / D-5, 5.48.0).  A prescription'
  lumenairy/propagators/gbd.py:3640 names 5.48.0 -- '.. versionchanged:: 5.48.0'
```

The merge brought in `fix/wp-b12b-gbd-round2` (`f5ad0f47`), which touched
`gbd.py` by 255 lines and left two forward version claims; the gate that
forbids them came in on the other side of the same merge.  Neither branch
alone is red.

REPRODUCER

```
cd <merge tip>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/unit/test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached -q --capture=sys
```

REQUESTED EDIT (one of the two, the maintainer's choice, but not neither):

* **preferred, and consistent with V-D15's own decision for seven other
  docstrings**: drop the number from both lines of
  `lumenairy/propagators/gbd.py` -- `:3622` becomes "returned silently
  (VERIFY-WP-B12b D-4 / D-5)" and the `.. versionchanged:: 5.48.0` directive
  at `:3640` goes, letting the CHANGELOG carry the version; or
* if `.. versionchanged::` is to be allowed to name an unreleased version,
  add that directive to the gate's exemption list in
  `tests/unit/test_public_api.py` **with the reason written down** -- and
  `:3622` still has to lose its number, because it is prose and not a
  directive.

### D-B -- BLOCKER (merged state).  Round 2's own citation gate is RED on `3fce9ecb`

`tests/unit/test_v5_3_2_walker_source_line_citation.py::
test_v18_5_the_5_47_0_block_citations_name_the_right_lines` is the gate V-D2
added.  On the merged tip, **Windows py3.14**, it reads **1 failed** with four
real findings:

```
4 citation(s) in the [5.47.0] CHANGELOG block do not name the line whose
CONTENT they named at f4f18851, and 0 could not be anchored at all.
  _lens_traced.py:3763                    -> _lens_traced.py:4130
  lumenairy/elements/_lens_traced.py:8539 -> lumenairy/elements/_lens_traced.py:8906
  raytrace/differential.py:557            -> raytrace/differential.py:614
  differential.py:748                     -> differential.py:805
```

Same mechanism as D-A: the merge moved `_lens_traced.py` by 467 lines and
`differential.py` by 71, and the gate that notices came in on the other side.
**The gate is working.**  V18 passes all four -- every one lands on some other
real line -- which is exactly the blind spot V-D2 built it for, so this is the
new instrument earning its place on its first real merge.

REQUESTED EDIT -- the tool round 2 shipped already does it; verified with
`--check` on this tree, which reports `4 re-anchored  (--check: nothing
written)`:

```
python scripts/reanchor_citations.py --base f4f18851 --block "[5.47.0]"
```

### D-1 -- MAJOR.  `method='direct'` is not immune to the chirp phase budget

Full measurement in section 4b.  Against a reference whose phase is reduced
EXACTLY, the dense route obeys `rel ~ eps * budget` with a fitted slope of
**1.0038** on the shipped N=24 M=12 geometry, reading **3.401e-05** at a
budget of 1e12 -- where the shipped id asserts `< 1e-14`.  The
`2.8e-16 .. 4.5e-16` reading is what a reference that forms `t = alpha*n*k` in
float64 and then reduces it measures about a route that does the same thing.

Identical to the digit on both builds.

REQUESTED EDITS

1. `lumenairy/propagators/_bluestein.py`, the `RuntimeWarning` message:
   replace "pass `method='direct'` (the dense route reduces its phase modulo
   one turn and measured 3e-16 at every budget tested)" with what it actually
   buys -- e.g. "pass `method='direct'`, which reduces its phase modulo one
   turn and measured about 7x smaller error at the same budget (still
   `~eps*budget`: 3.4e-05 at a budget of 1e12)".  This is the sentence a
   caller acts on, and it is currently wrong by up to 11 decades.
2. `lumenairy/propagators/_bluestein.py`, `_bluestein_2d`'s Notes: "The dense
   route (`method='direct'`) reads 2.8e-16 .. 4.5e-16 at EVERY budget tested,
   because it reduces `t` by `t - rint(t)` before calling `exp` and therefore
   has no chirp phase to lose."  The premise is true and the conclusion is
   not; state the measured constant (about 1/7 of the chirp-Z routes')
   instead.
2b. `_direct_matrix_2d`'s own "Phase construction" paragraph is the
   interesting one, because **it already contains the right answer**:

   > ``t = alpha*(n - cI)*(k - cO)`` is formed in float64 and then reduced by
   > ``t - rint(t)``, which is EXACT for ``|t| <= 2**52`` ... Only the
   > fractional turn reaches ``exp``, so the ``pi*alpha*N^2`` phase-budget
   > warning :func:`_bluestein_2d` carries does not apply to this route.  **The
   > irreducible error is the two roundings in forming ``t`` itself, amplified
   > by ``2*pi``.**

   The last sentence IS the law: two roundings in forming `t` cost
   `~eps |t| = eps * alpha * n * k`, and `2*pi` times that is exactly what I
   measure.  The sentence before it draws the opposite conclusion from the
   same mechanism.  The edit is to delete the "does not apply to this route"
   clause and let the sentence that follows stand, with the measured
   constant beside it.
3. `tests/unit/test_wave5_h2_mft_direct.py::_fsum_reference`: reduce the phase
   exactly (`fractions.Fraction`, as
   `validation/probe_verify_hyg2_round2/vh3_budget_exact.py` does -- the
   kernels are separable, so it is `N*M` exact reductions per axis and 0.2 s
   at these shapes) and re-derive the `rd` assertion from the new reading;
   or, if the cost is unwanted, rename the id and say in the docstring that
   the dense arm is a SUMMATION check and not a phase one.
4. The same paragraph in the round-2 addendum of `WAVE5_HYGIENE2_REPORT.md`
   and in `tests/unit/test_verify_wave5_hyg2.py::
   test_the_chirp_phase_error_is_linear_in_the_budget_and_dense_is_immune`.

The LAW and the THRESHOLD are unaffected: `1e-6/eps` is derived from the
chirp-Z routes' error, which is measured correctly.

### D-2 -- MAJOR.  The `(s.q)/N` term inside the ONE kernel is gated by nothing

Deleting the chief-ray subtraction from `_exact_dispersion_phase` (the
mutation `M2_linear_term_dropped`) moves **136 of my 342 byte-identity keys**
and leaves **all 216 ids of the five touched test files green on Windows and
all 108 of the light selection green on WSL**.  It is the one arm of the
consolidated kernel with no gate at all.

The mechanism: the shipped tilted fixtures are small-tilt, and the ids that
read a tilted leg either compare it to another leg carrying the same error or
read a magnitude the linear term barely moves.  A dropped linear term is
applied TWICE in the chain, because the caller already advances the chief ray
in real space (`x_c += L z / N`).

REQUESTED EDIT -- none in `lumenairy/`; the gate is added here:
`tests/unit/test_verify_hyg2_round2.py::
test_the_chief_ray_term_inside_the_one_kernel_is_gated_numerically` asserts
that the kernel has NO term linear in `q`, which is a physical property and
not a token.  The antisymmetric part over `max(eps k, |sym| q0/k)` reads
**0.000 .. 0.345** over four tilts and three `q0`, against a bar of 10, and
**1.0e+05 .. 2.3e+08** with the term deleted.  Please adopt it, or fold the
same statistic into `test_wave5_h2_collins_jax.py` beside the census.

### D-3 -- MINOR.  The WSL over-floor reading is attributed to the wrong rung

`936a7ca3`'s commit message, the addendum's V-D7 paragraph and
`test_wave5_h2_collins_jax.py::
test_the_central_difference_through_this_merit_has_no_truncation_branch`'s
docstring all say: "MEASURED at `h = 1e-1`, estimate over its own floor:
**0.36 (WIN) and 1.09 (WSL)**".  MEASURED here on the same fixture, the same
merit and the same ladder:

| h | WIN | WSL |
|---|---|---|
| 1e-1 | 0.362 | **0.362** |
| 1e-2 | 1.45 | **1.09** |
| 1e-3 | 0.724 | 0.000 |
| 1e-4 | 0.362 | 0.000 |

1.09 is the `h = 1e-2` rung.  At `h = 1e-1` the two builds agree to three
digits, which is a better advertisement for the bound than the sentence makes.
And the estimate is exactly 0.0 at **two** WSL rungs, not one, which
strengthens the case for the bound form.

REQUESTED EDIT: in the docstring and the addendum, "MEASURED over the ladder,
estimate over its own floor: 0.362 / 1.45 / 0.724 / 0.362 (WIN) and
0.362 / 1.09 / 0.000 / 0.000 (WSL) at h = 1e-1 .. 1e-4 -- exactly zero at two
rungs on WSL, which is why the ratio form was unusable."

### D-4 -- MINOR.  The falsification arm's separation is ladder-conditional

`test_the_same_ladder_does_find_a_truncation_branch_on_a_cubic_merit` says the
two readings are "two decades clear of the bar on one side and six on the
other, and the control cannot silently drift into the quadratic merit's
regime".  The control's ladder is `(1e-1, 3e-2, 1e-2, 3e-3)` and the quadratic
merit's is `(1e-1, 1e-2, 1e-3, 1e-4)` -- they are not "the same ladder".
Extending the cubic control by one decade to `h = 1e-4`, the same statistic
reads **0.3505**, inside the quadratic merit's own band, because a third
difference's round-off floor grows as `h^-3` and eventually swamps any real
`P'''`.

This is not a defect in the assertion, which is premise-gated at
`min(floors) > 1e4` over its own four rungs and therefore self-protecting.  It
is a defect in the sentence.

REQUESTED EDIT: say "over ITS ladder (`1e-1 .. 3e-3`)", and note that the
control's statistic falls below 100 by `h = 1e-4`, which is why the two ids do
not share a ladder.

### D-5 -- SUGGESTION.  The cross-backend bar reduces to its floor on both builds

`test_wave5_h2_collins_jax.py::_fft_spread_bar` multiplies the two backends'
measured FFT spread by a chain depth of 6 and floors it at `32 * eps`.
MEASURED here on the shipped fixture, both builds (section 3): the spread is
2.685e-16 (WIN) / 2.509e-16 (WSL), `6x` it is 1.61e-15 / 1.51e-15, and
`_fft_spread_bar(env)` returns **7.105427357601002e-15 == 32 * eps exactly**.
The floor dominates by 4.41x.

**This CONFIRMS the author's V-D16 entry** ("the `32*eps` FLOOR dominates by
4.4x, so 7.105e-15 is what is asserted against"), which I had expected to be
an approximation and which is exact.  I record it here only because it has a
consequence the file's docstring still claims otherwise for: the bar cannot
"track a numpy release, an XLA release, or a change of FFT backend" while it
is pinned to a constant -- a backend change would have to move the spread by
4.4x before the bar moved at all.

One more measurement, which is new: `np.fft.fft2` and `jnp.fft.fft2` agree
**BIT FOR BIT** at this shape on both builds, so the whole of the nonzero
spread is pyFFTW on the library side rather than any NumPy-vs-XLA difference.
A bar meant to bound "what two backends may legitimately differ by" is
therefore measuring, today, what two FFT WRAPPERS differ by.

SUGGESTED EDIT (not required for ship): either state in `_fft_spread_bar`'s
docstring that the floor is what is asserted against and the measurement is a
tripwire for a future backend, or derive the bar from a quantity that is a
property of the leg.  The leg's response to a 1-ULP input change is the
natural one -- 3.454e-16 (WIN) / 3.469e-16 (WSL) against cross-backend
differences of 5.486e-16 / 4.029e-16 --
and `tests/unit/test_verify_hyg2_round2.py::
test_the_cross_backend_bar_is_the_legs_own_last_bit_sensitivity` does it that
way and kills the M10 regression.

### D-6 -- OBSERVATION.  Two arms that no test on either build can see

* `M11_input_box_host_fft` (revert `_collins_input_box` to a host `_fft2`):
  **0 ids** on either build.  It is the shape V-D14 already documents --
  `np.asarray` and the device transform agree on VALUES for NumPy and JAX, so
  only CuPy would see it -- and CuPy's FFT is unavailable here.  Consistent
  with the author's own reasoning; recorded so the mutation count is honest.
* `M13_complex64_fold_removed`: **0 ids** on either build, 0 of my 342
  byte-identity keys, and no difference in a direct `complex64`-vs-
  `complex128` reading at `k z = 4.05e+08` (1.4958e-07 with the fold,
  1.4685e-07 without).  Host `cos`/`sin` reduce their own argument, so the
  fold is defensive rather than load-bearing on any backend available here.
  **I could not measure it**; it needs a working CuPy FFT.

### D-7 -- MINOR.  The evanescent-carrier REFUSAL is gated only by a token census

`M6_tilt_guard_weakened` (`if not (s2 < 1.0)` -> `if not (s2 <= 1.0)`, so the
grazing `|s| = 1` direction is admitted instead of refused) is caught by
exactly ONE id on each build, and on both it is
`test_wave5_h2_collins_jax.py::test_the_exact_dispersion_is_written_once` --
the single-definition census, which sees it only because the literal token
`s2 < 1.0` is one of the three strings it greps for.  No NUMERICAL id notices
that the refusal stopped refusing; my byte-identity set notices it as exactly
one kind flip out of 342.

That is a gate, and it is enough to stop this particular edit.  But it means
a re-spelling that preserves the behaviour change -- `if s2 >= 1.0: raise` --
would fail the census for the wrong reason, and a re-spelling that preserves
the census token while changing the comparison elsewhere would not be caught
at all.  A behavioural id is added here:
`tests/unit/test_verify_hyg2_round2.py::
test_the_evanescent_carrier_refusal_is_gated_by_behaviour` refuses five tilts
with `|s|^2 >= 1` (checking each refusal names its caller) and accepts three
with `|s|^2 < 1` including `|s|` one ULP below unity, and asserts that the two
shipped phrasings are still distinct -- they are reproduced verbatim on
purpose, because a byte-identity probe folds an exception's message into its
digest.  It kills M6.

## 10. New decision tests

`tests/unit/test_verify_hyg2_round2.py`, seven ids, 2.2 s for the file, every
one falsified against the mutation that motivates it before being written up.

| id | closes | seconds |
|---|---|---|
| `test_the_chief_ray_term_inside_the_one_kernel_is_gated_numerically` | D-2 | 0.004 |
| `test_no_call_site_negates_the_one_kernel` | section 2b -- a site-local conjugation | 0.37 |
| `test_the_evanescent_clamp_is_reached_and_holds_at_a_sub_wavelength_pitch` | section 1a -- the arm no fixture reaches | 0.004 |
| `test_the_evanescent_carrier_refusal_is_gated_by_behaviour` | D-7 | 0.004 |
| `test_the_accuracy_rule_is_never_executed_while_tau_is_none` | section 5 -- inert by EXECUTION, not by reading | 0.60 |
| `test_the_cross_backend_bar_is_the_legs_own_last_bit_sensitivity` | D-5 | 0.58 |
| `test_a_phase_budget_reference_must_not_reduce_its_phase_like_the_route` | D-1 | 0.16 |

FALSIFICATION, twelve arms in a scratch export, running this file alone:

| arm | this file |
|---|---|
| M1 kernel negated | 1 failed |
| M2 chief-ray term dropped | **1 failed** -- nothing else in the suite sees it |
| M3 one site conjugated | 1 failed |
| M4 `root0` not subtracted | 2 failed |
| M5 evanescent clamp removed | **2 failed** -- the byte-identity probe cannot see it |
| M6 tilt guard weakened | **1 failed** -- only a token census saw it before |
| M9 `tau` armed at 1e-4 | 1 failed |
| M10 public leg demotes again | 1 failed |
| M7, M11, M12, M15 | 7 passed -- outside these ids' scope, and M7/M12/M15 are caught by shipped ids |

Eight of the fifteen arms are killed by this one file, three of them (M2, M5,
M6) more tightly than by anything that shipped.

The seven ids are spliced into `.test_durations` (16588 -> 16595 entries,
re-parsed as JSON, no pre-existing entry touched); the staleness gate
`test_audit2609_a15a_durations_staleness.py` reads 4 passed.

## 11. What I could not measure

* **CuPy anywhere past a transform.**  `CUPY_AVAILABLE` is True on Windows and
  `cupy.fft.fft2` raises `ImportError: DLL load failed while importing cufft`;
  WSL has no CuPy.  So V-D3's CuPy row, the `bld` threading in
  `_collins_axis_chirp`, `_as_c_order`'s contiguity (M11) and the complex64
  `mod 2*pi` fold (M13) are all statements I can only make about where a
  failure moves, not about a device run.  Two of my three surviving mutants
  are in exactly this set.
* **Whether V-D19's one-off red is a branch regression.**  Unchanged from both
  earlier rounds.  I did not attempt the bisect; the assertion's SHAPE is
  sound either way and I verified both sides of its bar.
* **A non-paraxial oracle for the near-focus law.**  Unchanged.  The analytic
  Gaussian is paraxial, so the departure law measures distance from the
  paraxial truth and cannot referee which kernel is more physical -- which is
  the strongest single reason `tau` should stay `None`, and I agree with the
  decision.
* **An uncontended box.**  Another agent ran a four-worker pytest sweep on
  this workstation throughout.  Nothing in this document reads a wall clock as
  a claim, and the WSL mutation lane was reduced (section 8) rather than run
  contended; the seconds quoted for the new ids were taken in a focused run.
* **The full Windows selection on the WSL lane.**  For the same reason.  The
  three b4 classes that read the exact kernel ARE in the WSL selection; the
  four 2048x2048 gate classes are not, and V-D19's numbers were taken there by
  a probe instead of by pytest.
* **Whether the author's Windows environment is green on the metadata id.**  I
  can only say that mine is not, for an environment reason (section 8c), and
  that the tree is not implicated.
* **WHY the b4 file stalls on WSL.**  Measured as far as an idle
  `multiprocessing` spawn pool at id #57 with 95 GB of RAM free, reproducible
  3 of 3 (4 of 4 counting the control), and **identical on the `112c3049`
  archive -- the same 56 dots, the same id** (section 13).  So it is
  pre-existing and not this branch's, and the class runs fine in isolation on
  both trees.  What I did NOT do is find the leftover pool: that needs a
  whole-file bisect of a 15-minute stall, and nothing about round 2 turns on
  it.  It is worth someone's afternoon, because it means the b4 file cannot
  be run whole on a WSL CI lane.

## 12. Ship recommendation

**Do not ship the merge `3fce9ecb` as it stands.  Ship the round-2 branch's
content.**  The distinction matters and is the main finding of this
verification:

* Every numerical and dispatch claim of round 2 that I could re-measure is
  **CONFIRMED**, most of them on a larger instrument than the author's and
  with the same numbers to several digits.  The consolidation is byte-identical
  on 342 keys and two builds, the single-definition census holds structurally
  as well as by token, the V-D3 port reaches the public leg with no partial
  state on the refusal, the phase-budget LAW and threshold are correct and
  two-sided, the `tau` switch is off and provably not executed, V-D19's
  restatement is sound on both sides of its bar, and the S4 fix is both
  correct and necessary (two exactly-zero rungs on WSL, not one).
* **Two of round 2's own new gates are RED on the merge** (D-A, D-B), because
  the other side of the merge moved code they watch.  Both are one-line fixes,
  one of them by a tool round 2 itself shipped.  They are cheap, and leaving
  them red would retire two gates on their first real merge.
* **One shipped, caller-facing sentence is wrong** (D-1): the phase-budget
  warning tells a caller to escape the budget with `method='direct'`, and that
  route obeys the same `eps*budget` law.  The threshold is right; the remedy
  it offers is not.  This is the one item I would not let ship unedited,
  because it is a `RuntimeWarning` a user reads and acts on.
* **One arm of the consolidated kernel has no gate** (D-2), and one refusal
  beside it is gated only by a string the census greps for (D-7).  Tests for
  both are in this branch and kill their mutants on both builds.

ORDER OF WORK, smallest first:

1. `python scripts/reanchor_citations.py --base f4f18851 --block "[5.47.0]"`
   (D-B) -- one command, four citations, the gate goes green.
2. Drop the two `5.48.0` tokens from `lumenairy/propagators/gbd.py` (D-A) --
   two lines.
3. Re-word the `RuntimeWarning` message and the two Notes paragraphs in
   `lumenairy/propagators/_bluestein.py` (D-1 items 1 and 2).
4. Fix `_fsum_reference`'s phase reduction, or rename the id and say what it
   measures (D-1 item 3), and correct the report paragraph (item 4).
5. Adopt or re-home the two behavioural gates this branch adds:
   `test_the_chief_ray_term_inside_the_one_kernel_is_gated_numerically` (D-2)
   and `test_the_evanescent_carrier_refusal_is_gated_by_behaviour` (D-7).
6. The two sentence corrections, D-3 and D-4, and D-5's suggestion if
   the maintainer wants it -- D-5 is not a correction; it confirms V-D16.

Items 1-3 are what stands between this merge and a green gate on the tree it
was merged into.  Nothing in 1-6 touches an answer: the 342 keys stay
identical through all of them.

## 13. Test tails

The sweep is the 47 files the brief names: the touched files, my own new one,
`test_audit2609_b4_collins_transport.py`, every `*carrier*` and `*mft*` file,
the census / walker / dispatcher-pin / public-API / doc-consistency files,
`test_audit_except_budget.py` and `test_ci_kernel_consistency.py`.  BLAS pinned
on the command line, `--capture=sys`, `-p no:cacheprovider`, one process.

**WINDOWS py3.14, all 47 files**

```
3 failed, 1213 passed, 16 skipped, 29 warnings in 864.04s (0:14:24)

FAILED tests/unit/test_public_api.py::test_installed_metadata_version_matches_source_version
FAILED tests/unit/test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached
FAILED tests/unit/test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines
```

One environmental (a stale global editable install: dist-info 5.47.0 against
source 5.47.1), and **the two BLOCKERS, D-A and D-B**.  Nothing else in 1213
ids is red.

The 16 skips are all pre-existing -- round 2's test diff adds no
`pytest.skip`, `skipif` or `importorskip` at all, measured over the whole
`112c3049..936a7ca3` diff of `tests/`, 0 hits -- and 14 of them were
enumerated with `-rs` to check they are not the S3 shape:

* 8 in the walker CHANGELOG gates, all content-conditional ("the latest
  CHANGELOG block has no claim of this kind; nothing to verify").  A gate
  with nothing to gate is not a resource skip, though it is worth knowing
  that the `[5.47.1]` block currently silences eight of them.
* 6 in `test_v4_14_2_dispatcher_pin_cache_locks.py`, each a DECLARED
  exemption in a parametrised lock census (`_REGISTRY_LOCK`,
  `_ABANDONED_POOLS_LOCK`, `_PERSISTENT_POOL_LOCK`, `_BLAS_CONTROLLER_LOCK`,
  `_ZARR_MKDIR_PATCH_LOCK`).

None of the 14 is conditioned on a machine resource.

**WSL py3.12, 46 of the 47 files**

```
5 failed, 1084 passed, 17 skipped, 29 warnings in 610.63s (0:10:10)

FAILED tests/unit/test_public_api.py::test_installed_metadata_version_matches_source_version
FAILED tests/unit/test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached
FAILED tests/unit/test_v5_3_2_walker_source_line_citation.py::test_v18_5_companion_reanchor_tool_exists_and_covers_the_cited_files
FAILED tests/unit/test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines
FAILED tests/unit/test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught
```

Four of the five are the environmental pair (a stale editable install in
`~/lumvenv`, and the Windows-worktree `.git` pointer that no `git show` can
follow from Linux -- section 8c).  The fifth is **D-A, which is red on BOTH
builds** and is therefore not an environment story.

`test_audit2609_b4_collins_transport.py` is run SEPARATELY on this lane,
because on WSL the WHOLE FILE stalls.  What I measured, before deciding it is
not this branch's:

* it stalls at **id #57 of the file**, which the collection order makes
  `TestGateCTwoGroupChain::test_both_transports_reproduce_the_audit_s_own_
  readings` -- the `p5_arms` fixture, a 2048x2048 two-group traced-lens chain.
  **Three attempts out of three.**
* the stalled process is `State: S (sleeping)` with 50 threads in the main
  interpreter and five `multiprocessing.spawn` workers of 24 threads each --
  an IDLE spawn pool, not a busy loop.  It accumulated **1.3 CPU-seconds over
  17 minutes** on the first attempt and **0.2 over 3 minutes** on the second.
  95 GB of RAM free, so it is not the memory-pressure shape this repository
  has been bitten by before.
* **CONTROLS, all on WSL.**  `TestGateBMismatchMatrix` alone: 3 passed in
  3.65 s on the tip and 3.71 s on the `112c3049` archive.
  `TestGateCTwoGroupChain` alone -- the class that stalls inside the whole-file
  run -- **completes on the tip AND on the `112c3049` archive**.  And my own
  `vh3_v19_s4.py` builds that same fixture outside pytest on WSL and returns
  numbers identical to Windows to every digit (section 6).
* so it is a WHOLE-FILE interaction: a spawn pool left behind by the earlier
  gate classes, not the fixture itself, and not anything round 2 touched.
  **THE CONTROL THAT SETTLES IT**: the whole file run on the `112c3049`
  ARCHIVE, on WSL, under `timeout 1200`, stalls at **exactly the same 56
  dots** as the tip -- the same id, the same place.  Run side by side, the
  pre-round-2 tree and the merged tip are indistinguishable.  The stall is
  pre-existing.

I did not chase it further.  It costs hours to bisect, nothing in it is a
statement about this branch, and the numbers the class produces were taken on
this lane by a probe instead.  It is in section 11 as something I could not
measure.

**The new file, on its own, both builds**

```
WIN py3.14  7 passed in 2.15 s
WSL py3.12  7 passed in 2.25 s
```

**The durations gate**: `test_audit2609_a15a_durations_staleness.py`
4 passed in 40.27 s (WIN), after splicing seven ids into `.test_durations`
(16588 -> 16595 entries, re-parsed as JSON, no pre-existing entry touched).

**ruff 0.15.16 (WSL), repo-wide**: `All checks passed!`
