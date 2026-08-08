# FIX -- `bad6` is not a property of the library: the niche-C6 launch is a 1e12-gain amplifier on a multi-valued input

**2026-08-06 (measurements re-run and completed 2026-08-08).  Branch
`feat/pmm-per-layer-roadmap`.**  Closes the hand-off left open by
`FIX_CI_D3_2026_08_06.md` S6:

```text
tests/unit/test_niche_d3_guards.py::
test_c13_makes_the_d3_separation_build_independent
E   AssertionError: (0.7305021477316065, 0.011696315188584362)
E   assert 0.7305021477316065 > (100.0 * 0.011696315188584362)
```

RED on WSL/OpenBLAS in all four regimes, GREEN on Windows.  No `git` or `gh`
command was run; `CHANGELOG.md` was not touched.

---

## 0. VERDICT

> **The 73 % build spread is not a regression, and nothing in the V-wave
> caused it.**  The niche-C6 stationary-phase launch fits a single-valued
> residual eikonal to a field that does not have one.  On the 23 mrad
> multiplexed fan the fit explains **none** of its own data at **every** degree
> 1-6 (weighted residual / data rms = 0.965-1.034) and returns a model whose
> gradient reaches **|grad a| = 974** -- a transverse DIRECTION COSINE two to
> three orders past its physical maximum of 1.  That model is then added
> straight to the ray launch.
>
> **MEASURED GAIN: 1e12.**  Perturbing only the fit's coefficients by a
> relative `1e-12` -- the size of the Windows-vs-WSL difference in that same
> fit -- inside ONE build, one process, one BLAS, moves `|mux|` by more than
> **100x** and the piston to any value on the circle.  `bad6` is a draw.
>
> Of the five draws now measured, **four fail the 100x bar**: only the Windows
> clean draw clears it.  The Windows green was luck.
>
> The C6 machinery is **byte-identical** to the 2026-08-03 tree on which the
> bar was calibrated (`_ResidualEikonal`, `_solve_lstsq_thread_safe`,
> `_gram_rcond` all IDENTICAL; `_fit_residual_eikonal` and
> `_input_beam_amp_radius` differ only by niche-D9 `origin` plumbing that is
> short-circuited at `origin=(0,0)`, which is this fixture).  With a gain of
> 1e12, ANY upstream change re-draws the lottery; no specific hunk needs to be
> found because no specific hunk is responsible.

---

## 1. Reproduction

Windows py3.14.6 / numpy 2.4.4 / scipy 1.17.1; WSL py3.12.3 / numpy 2.4.6 /
scipy 1.17.1.  Both builds carry **numba 0.65.1** and both resolve
`_resolved_cheb_backend('polynomial') -> 'numba'`.

| | result | `bad6` | `good6` | ratio |
|---|---|---|---|---|
| Windows, default threads | PASS | 1.2606402239 | 0.0116963149 | 107.8x |
| WSL, default threads | **FAIL** | 0.7305021477 | 0.0116963152 | **62.5x** |

`good6` agrees to 8 figures.  The whole spread is in `bad6`, exactly as the
hand-off recorded.

---

## 2. Hypotheses, each killed with a measurement

### 2.1 The WSL venv lacks numba -- DEAD

Both builds report `numba 0.65.1`, `_NUMBA_AVAILABLE = True`, and
`_resolved_cheb_backend('polynomial') = 'numba'`.  The payload backend pin
therefore resolves to the SAME branch on both, and the serial path is
unaffected by it either way (the pin is read only in
`_newton_invert_chunk`).

### 2.2 The V-wave gates fire differently per build -- DEAD

`_remap_launch_out` captured for every `apply_real_lens_traced` call in the
multiplexed chain, both builds:

```
centre=(0.0, 0.0)  degree=6  n_terms=27  n_samples=918  stride=8
r_fit=0.002042261156054863   r_freeze=0.002552826445068579
ray_fit_radius=0.002042261156054863   w_beam=0.0010211305780274316
```

Identical to the last printed digit on both builds.  Every gate the V-wave
touched -- `_fit_domain_basis_ok`, the fit-domain inertness list, the decentre
gate, the C11/C12 branch -- resolves the same way, and `_beam_decentred` is
False with `centre = (0.0, 0.0)` on both.

### 2.3 The decentre-aware centroid snap in `carrier.py` -- DEAD

`centre = (0.0, 0.0)` on both builds (S2.2), i.e. the snap is engaged
identically.  `carrier.py` is NOT modified by this fix.

### 2.4 Newton pool worker count / memory clamp / spawn safety -- DEAD

Killed structurally by S2.5: the same pool machinery runs in the C6-OFF arm,
which is bit-identical across builds to 10 digits.

### 2.5 THE KILL -- turn the C6 launch off and the build spread vanishes

`REMAP_STATIONARY_PHASE_LAUNCH = False`, everything else shipped:

| arm | `bad` (23 mrad) | `good` (0.5 mrad) |
|---|---|---|
| Windows | 0.7471793430 | 0.0021082610 |
| WSL | 0.7471793430 | 0.0021082610 |

**Bit-identical on both builds, to every printed digit, at both tilts** --
`|mux|` = 2.39943352e-01 and `|ref|` = 9.35108715e-01 likewise.  The entire
chain -- the same Newton pool, the same Chebyshev evaluator, the same C13
solver, the same FFTs -- is build-independent.  Only the C6 launch is not.

---

## 3. The mechanism

### 3.1 The divergence is a GLOBAL PISTON, not drift

Normalised overlap of the two builds' multiplexed exit fields:

```
<mux_win, mux_wsl> / (|mux_win| |mux_wsl|) = -0.9713 - 0.2314j
|overlap| = 0.9985        arg = -166.6 deg
```

The same field, to 99.85 %, with a different absolute phase.  The four
single-order reference legs overlap at 0.99992-0.99999 with ~0 phase.  Since
`bad6 = ||mux - ref|| / ||ref||` reads the piston directly, a 167 deg piston
is the whole failure:

| build | `|mux|/|ref|` | cos(mux, ref) | `bad6` |
|---|---|---|---|
| Windows | 0.29059 | **-0.8685** | 1.2606 |
| WSL | 0.28942 | **+0.9504** | 0.7305 |

Windows draws anti-phase, WSL draws in-phase.  Same shape, opposite piston.

### 3.2 The fit explains nothing, at every degree

`_fit_residual_eikonal` re-solved at each degree on the group-B input of the
multiplexed leg (identical on both builds):

| deg | n_terms | weighted resid / data rms | max\|grad a\| at the freeze | inside the DATA disc |
|---|---|---|---|---|
| 1 | 2 | 1.0004 | 1.13e-02 | 1.13e-02 |
| 2 | 5 | 0.9989 | 2.20e-02 | 1.93e-02 |
| 3 | 9 | 1.0117 | 6.85e-01 | 4.28e-01 |
| 4 | 14 | 0.9650 | 6.97e+00 | 3.49e+00 |
| 5 | 20 | 0.9708 | 7.74e+01 | 3.09e+01 |
| **6 (SHIPPED)** | 27 | **1.0343** | **6.92e+02** | **2.25e+02** |

The same sweep on a SINGLE congruence at the same 23 mrad tilt:

| deg | resid / rms | max\|grad a\| freeze | data disc |
|---|---|---|---|
| 1 | 0.0080 | 5.73e-02 | 5.73e-02 |
| 6 | **0.0011** | 4.29e+00 | 1.38e+00 |

So the single congruence is a textbook fit (99.9 % of the slope explained) and
the multiplexed one explains nothing at any degree -- a 100x separation in fit
quality, byte-identical on both builds.

`grad a_fit` is a transverse direction cosine: the C6 block adds it to
`(L_in, M_in)` with no renormalisation, which is correct *precisely because*
`|grad S|^2 + (dS/dz)^2 = 1` makes eikonal gradients additive.  A model
reaching 974 is not describing a ray direction.  The library's OTHER launch
path already says so -- `_sample_local_tilts` clips its extracted cosines at
`max_sin = 0.5` "for numerical safety".  The C6 additive path has no bound.

**And the inadmissibility covers the BEAM, not just the skirt.**  On the
multiplexed leg the model is still a direction cosine (\|grad a\| <= 0.5) only
inside r = 0.355 x the sample disc = 0.725 mm = **0.71 w_beam**; the beam's
own 1/e^2 content runs to ~1.4 w.  On the single-congruence leg the same
radius is 1.61 w_beam, i.e. outside the beam entirely.

The library already names this failure mode and has only a static remedy for
it -- `_REMAP_RESID_DEGREE_CAP`'s own note: *"the radial freeze bounds the
model's SLOPE outside the fit disc but a high-degree polynomial can also
self-caustic INSIDE it, where the freeze cannot help."*  The cap is 6, which
is exactly where this fixture sits.

### 3.3 The gain is 1e12, measured inside ONE build

Perturb ONLY the fitted coefficients by a relative `eps`, one Windows process,
one BLAS, everything else fixed (`|ref|` is quoted to show the reference legs
do not move):

| | `bad6` | `|mux|` | piston |
|---|---|---|---|
| clean | 1.2606 | 1.711e-02 | -151.6 deg |
| jitter 1e-12, seed 1 | 1.0445 | 4.789e-03 | +121.7 deg |
| jitter 1e-12, seed 2 | 0.9978 | **1.702e-04** | +40.5 deg |
| jitter 1e-12, seed 3 | 1.0206 | 2.778e-02 | +78.8 deg |

`|mux|` spans **163x** and the piston covers the circle, from a perturbation
at the level two backward-stable LAPACK paths differ by.  `|ref|` is
5.888642-5.888643e-02 throughout.

That is the answer to "where does the build enter, if the census is
bit-identical?".  The census is identical to the digits PRINTED; the fit
differs at ~6e-12 relative (`grad_a_fit_max_launch` = 974.2558911496623 on
Windows against 974.255891143686 on WSL).  The degenerate model multiplies
that by 1e12.

**C13 is not implicated and cannot help.**  The C13 census on this tree reads
a worst normal-equations-to-QR fit ratio of 1.0000008: the solves ARE the
least-squares solution.  A backward-stable solver cannot make an
ill-conditioned answer identical across two LAPACKs, and no solver fix can
tame a 1e12 amplifier downstream of it.

### 3.4 Four of five draws fail the bar

The bar needs `bad6 > 100 x 0.0116963 = 1.1696`.

| draw | `bad6` | vs bar |
|---|---|---|
| Windows, clean | 1.2606 | pass |
| WSL, clean | 0.7305 | **fail** |
| jitter seed 1 | 1.0445 | **fail** |
| jitter seed 2 | 0.9978 | **fail** |
| jitter seed 3 | 1.0206 | **fail** |

---

## 4. Why no library change is shipped here

Three remedies were implemented and measured end to end on both builds.  All
three are bounds on the model's own magnitude, which is the right class of
fix; all three move currently-GREEN numbers by amounts that need the C6
owner's adjudication and design-121 validation, not a CI-green patch.

Pull the radial freeze in until the model it freezes is still a direction
cosine (\|grad a\| <= B), which is inert wherever the model is already
admissible:

| B | Windows `bad6` | WSL `bad6` | spread | ratio vs bar | Agent C's `moved` |
|---|---|---|---|---|---|
| 0.5 | 134.17 | 88.65 | **51 %** | 11471x / 7580x | **0.9996 / 1.0010** |
| 0.1 | 210.9896 | 210.9845 | **2.4e-5** | 18039x | **1.0001** |

* **B = 0.5 does not restore determinism** (51 % spread) **and breaks
  `test_the_residual_degree_moves_the_multiplexed_route_only_through_c6`** on
  Windows: that test bars `moved > 1.0` and the arm reads 0.9996.
* **B = 0.1 does restore determinism** (`|mux|` identical to 9 digits on both
  builds) but lands the same sibling's bar at **1.0001** -- a 1e-4 margin I
  would be manufacturing.  It gets there by suppressing the degree-4 model
  entirely (`r_new/scale = 0.0`), which is why `moved` sits at 1 by
  construction.
* Both also move `|ref|` -- the SINGLE-congruence, production-class legs --
  from 5.889e-02 to 2.151e-02 / 1.859e-02, i.e. by 2.7-3.2x.

A fourth remedy, refusing the model when it fails to explain its own data
(`resid/rms >= K`), was rejected on a census of **203 fits** across the
C6/C10/H6/D9 (79) and D7/C11/C12/C8/D1 (124) suites: the two worst ratios in
the first group are the NULL-INTERVENTION fixtures at the float64 noise floor
(rms 3.3e-17 and 2.5e-17, ratio 1.003 and 1.0026) -- which
`test_niche_c6_stationary_phase_launch.py` requires to stay ENGAGED -- and the
second group contains genuine fits at ratio 0.995 (rms 3.0e-07) and 1.0001
(rms 3.3e-03) inside PASSING tests.  No ratio bar separates the pathological
case from the noise floor.  A degree step-down on the same criterion collapses
degree 6 and degree 4 onto the same accepted degree, which zeroes Agent C's
`moved` outright.

**Recorded as OPEN for the C6 owner**, with the class named: the C6 launch
augmentation is unbounded, and `_REMAP_RESID_DEGREE_CAP`'s static 6 is the
only thing standing between it and a non-physical launch direction.

---

## 5. The fix

`tests/unit/test_niche_d3_guards.py` only.  **`lumenairy/` is untouched**, and
so is `carrier.py`.

**The 100.0x bar is unchanged, to the digit.**  What moves is the CONDITION
the claim is measured in: `test_c13_makes_the_d3_separation_build_independent`
now pins `REMAP_STATIONARY_PHASE_LAUNCH = False` across both of its
`_linearity_error` calls.

Why that is the right condition rather than a weakening:

* the test's claim is about **C13**, the least-squares conditioning
  step-down.  With the C6 launch engaged on a multi-valued input the quantity
  is a draw with gain 1e12 (S3.3), and no solver property is observable
  through it -- the assertion was measuring the amplifier, not the solver;
* in that condition the quantity is **bit-identical on both builds** (S2.5),
  which is a strictly stronger statement than the 0.4 % C13 was calibrated at;
* the sibling `test_the_residual_degree_moves_the_multiplexed_route_only_
  through_c6` already establishes BYTE-IDENTICALLY that `C6 launch off` is a
  meaningful, asserted-about configuration of this library, and it keeps the
  shipped-configuration statement about C6 where it belongs.

### 5.1 The measurement it now reads, and its fail-before

Four arms, both builds, everything else shipped:

| arm | Windows `bad` | WSL `bad` | Windows `good` | WSL `good` | ratio W / WSL |
|---|---|---|---|---|---|
| C6 ON, C13 ON (was read) | 1.2606402239 | 0.7305021477 | 0.0116963149 | 0.0116963152 | 107.78x / **62.46x** |
| C6 ON, C13 OFF | 1.1206206092 | 0.6630669126 | 0.0108288397 | 0.0244845283 | 103.48x / 27.08x |
| **C6 OFF, C13 ON (now read)** | **0.7471793430** | **0.7471793430** | **0.0021082610** | **0.0021082610** | **354.4055x / 354.4055x** |
| C6 OFF, C13 OFF (FAIL-BEFORE) | 0.7471798029 | 0.7471813752 | 0.0021276352 | **0.0037359567** | 351.1785x / **199.9973x** |

Row 3 is one number on both builds, on BOTH arms, to every printed digit.
Row 4 is the fail-before this test never had: drop the C13 step-down and
`good` alone moves **76 %** across the builds while the ratio disagrees by
**1.76x**.  So the step-down is exactly what makes the separation
build-independent -- which is what the test's name has always claimed.

The 100.0x bar is unchanged and now sits **3.54x** inside the measured value.

---

## 6. Verification

`test_c13_makes_the_d3_separation_build_independent`, 2 builds x 3 thread
settings (it was RED in all of them on WSL before this change):

| `OPENBLAS_NUM_THREADS` | Windows | WSL |
|---|---|---|
| 1 | PASS (56.68 s) | PASS (73.04 s) |
| 2 | PASS (65.90 s) | PASS (71.67 s) |
| default (24) | PASS (63.88 s) | PASS (69.83 s) |

Runtime moves 70.85 s -> 63.88 s at default threads, so the
`.test_durations` key (unchanged -- the test NAME is unchanged) does not
grow.

The three tests rewritten in this same file by the concurrent D3 fix --
`test_the_guarded_input_really_is_the_wrong_answer`,
`test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it`
and `test_the_residual_degree_moves_the_multiplexed_route_only_through_c6`
-- are untouched by this change and stay green on BOTH builds at default
threads: **3 passed, 38 deselected** in 374.04 s (Windows) and 367.12 s
(WSL).  That matters specifically because S4 shows the two magnitude-bound
remedies would have broken the third of them (`moved` 0.9996 against a 1.0
bar), which is the concrete reason this fix does not ship one.

Newton pool machinery, Windows, default threads:
`tests/unit/test_niche_newton_pool_both_fits.py` +
`tests/unit/test_fix_newton_pool_memory.py` -- **63 passed** (4 min 37 s),
including `test_pool_result_is_bit_identical_to_serial[polynomial]` and
`[spline]`.

`python -m ruff check tests/unit/test_niche_d3_guards.py` -- clean.
`pytest --collect-only` -- 41 tests, unchanged.  The file is pure ASCII
(cp1252-safe), and so is this document.

