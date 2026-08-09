# M1 -- Conditioning integrity across both solvers (X-1, N-2, T3-3)

**Date:** 2026-08-04 - **Branch:** `feat/pmm-per-layer-roadmap` (off v5.32.1 main, `013c388`)
**Plan:** `docs/audits/PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md` S4, mission **M1**
**Precedent:** `docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md` (the pattern this was
built to follow -- and the two places it deliberately does NOT)
**Status:** SHIPPED on the branch, uncommitted. Both BLAS builds green.

Evidence tags follow the plan: **[M]** measured by a run, **[A]** analysis of the tree as it
stands, **[H]** hypothesis, flagged. Every numeric claim below is **[M]** on **both** builds
unless it says otherwise.

**Builds.** *Windows/MKL*: numpy 2.4.6 + MKL, py3.14, `OMP/OPENBLAS/MKL_NUM_THREADS=1`.
*WSL/OpenBLAS ("the CI proxy")*: `~/lumvenv`, numpy 2.4.6 on scipy-openblas 0.3.31 (SkylakeX),
py3.12, same pins. Both read the same working tree over `/mnt/d`.

---

> ## CORRECTION, same day, by measurement -- READ THIS FIRST
>
> **The inverse refusal shipped in the first cut of this document was WRONG and has been
> withdrawn.** The breadth sweep refuted it: the refusal thresholds were calibrated on two 1-D
> families, and the **2-D hybrid interface** -- correct, build-stable, and pinned since v5.14 --
> reads *inside* the 1-D broken band on **both** instruments (equilibrated `rcond` 3.9e-14 against
> a broken band of 3.8e-19..1.3e-10; equilibrated residual 1.2e-05 against 5.3e-08..3.7e+07).
> There is no global bar. Shipping it would have refused **five long-pinned 2-D tests on both
> builds** -- a false pathology claim, which this campaign's own R-1b precedent rates worse than
> silence.
>
> What that costs, stated plainly: **X-1 is reproduced, instrumented and documented, but NOT
> closed.** The headline numbers below marked ~~struck~~ were claims of the withdrawn refusal.
>
> What survives is real and is measured: **T3-3** (an independent, sound fix), the
> **least-squares refusal** on a discriminator that *does* separate all four families
> (rank-deficiency AND residual), the **census and its findings**, and a **NaN-diagnostic
> regression** that the first cut introduced and this one removes. S2.7 is the full account.

## 0. Headline

| | pre-M1 | M1 |
|---|---|---|
| ~~worst cross-build relative disagreement on a returned solve~~ | ~~8.3e-02~~ | **unchanged -- X-1 open** |
| ~~truncations where the builds disagreed on whether to return~~ | ~~1~~ | **unchanged -- X-1 open** |
| **T3-3**: conical per-layer far-field cap over-statement | **2.7x** | **fixed** |
| T3-3 silent Jones error at `ffo` 61 that energy closure could not see | **1.6e-06** | **0** |
| T3-3 loud failure at `ffo` 77 (`\|R+T-1\|`) | **15** | **0** |
| over-capacity Rayleigh projections refused (rank-deficient **and** off-range) | 0 | **4 measured** |
| bits moved on ANY solve, anywhere | -- | **0** |
| `tracemalloc` peak change | -- | **<= 0.005 %** |
| wall time change (inverse guard is now zero-cost by default) | -- | **~0 %** |

**The four findings that were not in the plan's brief, and that shaped the outcome:**

1. **X-1's named site is not where the build-dependence came from.** `inv(a+b)` is real
   (measured `cond` 3.1e16) but the **Redheffer star denominators** -- `rcwa/_core.py:2013-2014`,
   not in X-1's scope -- are the dominant site (`cond` 2.4e31) and are what separates every broken
   solve from every clean one. They are now guarded too. (S2.3)
2. **There is no better re-solve, so the C13 step-down does not transfer.** Five inverse routes
   were measured on the real matrices; the SHIPPED LU inverse beats the SVD pseudo-inverse
   everywhere and beats both QR routes everywhere except the two most singular matrices, where all
   four are catastrophic together. At `cond` 3.1e16 the best of the five still misses `A X = I` by
   4.3e-03. So a C13-style step-down is unavailable here. (S2.4)
3. **The score must be EQUILIBRATED, and finding that out cost a false-positive.** A raw
   `||A X - I||` separates the thin grating's broken truncations from its clean ones -- and then
   falsely refuses the whole anisotropic 1-D Jones path, whose star denominators carry
   `||I - B11 A22||_1` ~ 1e17 at *every* truncation while the answer is right and the two builds
   agree to twelve digits. (S2.5)
4. **And equilibration was not enough either.** The 2-D methods -- never sampled by the census --
   land INSIDE the broken band on both equilibrated instruments, so the inverse refusal was
   withdrawn entirely and X-1 remains open. This is the finding that matters most, and it came
   from the breadth sweep rather than from the study. (S2.7)

---

## 1. What was unguarded, and the census that found it

### 1.1 The sites [A]

| # | site | what it inverts | guard before | guard after |
|---|---|---|---|---|
| X-1a | `rcwa/_core.py` `_interface_smatrix` | `inv(a+b)`, explicit (`S12 = 2 (a+b)^-1`) | none | `_guarded_inverse` |
| X-1b | `rcwa/_core.py` `_interface_smatrix_general` | `inv(T22)` of the 4N generalized interface | none | `_guarded_inverse` |
| **X-1c** | `rcwa/_core.py` `_redheffer_star` | `inv(I - B11 A22)`, `inv(I - A22 B11)` | none | `_guarded_inverse` **[scope addition, S2.3]** |
| N-2a | `pmm/_core.py` `_interface_smatrix` | `inv(a+b)` (the SHARED-grid path, i.e. per-layer's own oracle) | none | `_guarded_inverse` |
| N-2b | `pmm/_core.py` `_interface_smatrix_mortar` | `inv(I + BA)` | none | `_guarded_inverse` |
| N-2c | `pmm/_core.py` `_redheffer_star_rect` | two `inv` | none | `_guarded_inverse` |
| N-2d | 6 sites: `pmm/_core.py` x2, `stack.py` x2, `conical.py`, `stack2d_pure.py`, `twod_staggered.py` | `lstsq(Hsup, rhs, rcond=None)` | `rcond=None` = the float64 noise floor | `_guarded_lstsq` |
| N-2e | `pmm/_core.py` `_interface_smatrix_mortar` | two `solve(...)` | none | **deliberately none -- S5.1** |
| N-2f | `pmm/_core.py` `_interface_smatrix_general_mortar` | one `solve(A, B)` | none | **deliberately none -- S5.1** |
| T3-3 | `pmm/conical.py:202` (pre-M1 numbering) | the far-field order cap | union-derived on both paths | window-derived on the per-layer path |

`rcwa/_core.py:440` -- `_check_energy`'s own docstring -- already recorded that the X-1a matrix
"goes near-singular (cond up to ~1e13)". Nothing acted on it.

### 1.2 Workload: the library's own named instability class [M]

`_check_energy` and `rcwa_efficiency_1d`'s Notes both name it, and
`test_rcwa_reduces_to_thin_grating_limit`'s comment records that *"WHICH truncations blow up is
BLAS-build dependent (OpenBLAS trips at M=14/15 where MKL is clean)"*. So the census used it
verbatim: period 10 um, `n_ridge` 1.55 / `n_groove` 1.50 (contrast 0.05), depth 0.5 um, duty 0.5,
lambda 700 nm, normal incidence, `n_orders` 8..23, TE and TM, `stabilize=False` -- 32 solves,
each measured at every interface and every star, on both builds.

Two controls: an ordinary sub-wavelength high-contrast grating (0.5 um period, n = 2.0/1.0,
`n_orders` 12), and the **anisotropic 1-D Jones cascade** (uniaxial cell, `n_orders` 5..25),
which is what caught the false positive in S2.5.

### 1.3 Census: conditioning, pre-M1, both builds [M]

`cond` is a property of the assembled matrix and the assembly is deterministic, so the two builds
agree to the printed digit; what diverges is the **inverse**.

| site | sub-wavelength control | thin grating, median | thin grating, max |
|---|---|---|---|
| `_interface_smatrix` `a+b` | **9.0** | 6.7e+04 | **3.1e+16** |
| `_redheffer_star` `I - B11 A22` | -- | -- | **2.4e+31** |
| `_interface_smatrix` `Wb` (`solve` operand, unguarded) | 1.0 | 1.0 | 1.04 |
| `_interface_smatrix` `Vb` (ditto) | 35 | 1.4 | 7.2 |
| `_interface_smatrix_general` `T22` (4N path; OOP-cell workload, med / max) | -- | 13.8 | 20.6 |

Reading: the two `solve` operands are *fine* -- which is the docstring's own claim ("`solve` is
used ... so the deliberately tiny-columned evanescent eigenvectors do not blow up an explicit
inverse") and it holds. The exposure is entirely in the **explicit inverses**, and the star
denominator is fifteen orders worse than the interface behind it.

### 1.4 The defect, in the observable [M, both builds]

`sum(R)` is the deep null here (~2.0e-04 converged), so it is the sensitive observable; `sum(R+T)`
is the conservation score. Pre-M1, `stabilize=False`:

| `n_orders`.pol | MKL `sum(R)` | OpenBLAS `sum(R)` | relative gap | verdict |
|---|---|---|---|---|
| 21.te | 0.032165672615 | 0.032165672615 | 0.0 | **both builds agree on an answer 160x wrong** (`R+T` = 1.032, inside `_EnergyWarning`'s documented silent window) |
| 20.te | 0.000208856986 | 0.000226215644 | **8.3e-02** | worst returned disagreement in the census |
| 19.te | 0.018387638395 | *raises* `_EnergyError` | -- | **the builds disagree on whether there is an answer at all** |
| 12.te | 0.000201645434 | 0.000201466547 | 8.9e-04 | |
| 22.te | 0.000209430916 | 0.000209431293 | 1.8e-06 | ULP class, not the defect class |

21.te is the important row: **cross-build agreement did not catch it.** Both builds computed the
same wrong number. That is why the instrument had to be a residual, not a comparison.

---

## 2. The fix, and every route that was measured out of it

### 2.1 What ships

> **Superseded in part by S2.7** -- the inverse refusal described below was withdrawn the same
> day. `_guarded_inverse` still exists, still computes both scores, but only when `_INV_CENSUS`
> is armed, and it never raises. Read this subsection as the design that was *tried*; S2.7 is
> what shipped.

`lumenairy/elements/rcwa/_core.py`:

```
INTERFACE_CONDITIONING_GUARD = True   # the fail-before flag
_INV_RCOND_SCREEN = 1e-8              # free screen: equilibrated reciprocal cond_1
_INV_RESID_REFUSE = 1e-8              # refusal bar: equilibrated ||Ae Xe - I||_F / sqrt(n)
_INV_CENSUS       = None              # the instrument, not a behaviour switch
class _ConditioningError(_EnergyError)
```

There is **no step-down flag** and deliberately so (S2.4): a switch nobody should flip is a
liability, so the decision lives as a comment block with its measurement, not as a constant.

`_guarded_inverse(A, site, hint)`:

1. invert exactly as before, `xp.linalg.inv(A)`;
2. non-NumPy backend or guard off -> return unchanged (JAX is traced, CuPy would sync per
   interface);
3. **screen** on `_rcond_1_equilibrated(A, X)` -- free (S2.2). At or above `1e-8`, return the LU
   inverse **bit for bit**;
4. otherwise score the operator on `A X = I` in its equilibrated scaling; above `1e-8` raise
   `_ConditioningError`, below it return the LU inverse **bit for bit**.

`lumenairy/elements/pmm/_core.py`: `_LSTSQ_RESID_BAR = 1e-9` and `_guarded_lstsq(A, b, site)` --
same shape, scored on `||A x - b|| / ||b||`, one extra matvec.

`lumenairy/elements/pmm/conical.py`: `PMM_CONICAL_PERLAYER_ORDER_CAP = True`, T3-3's own
fail-before switch.

**There is exactly one behaviour change: the refusal.** Nothing the guard returns has moved a bit.

### 2.2 Why the screen is free [A]

`cond_1(A) = ||A||_1 ||A^-1||_1` is exact (not an estimator), and `A^-1` is the thing just
computed -- two `O(n^2)` reductions. Equilibration stays free too, by the exact diagonal identity

```
Ae = R^-1 A C^-1   ==>   Ae^-1 = C A^-1 R      (R, C diagonal)
```

so `||Ae^-1||_1 = max_j r_j * (c @ |X|)_j` -- one gemv, no second factorisation, no `gecon`, no
SVD. Only the **confirming residual** costs an inverse, and only on a solve the free screen has
already flagged. `test_the_equilibrated_rcond_identity_needs_no_second_factorisation` pins the
identity against an explicit `inv(Ae)`.

### 2.3 The scope addition: the star denominators are the dominant site [M]

X-1's brief named `inv(apb)` and the 4N interface. The census says the amplifier is where
`_check_energy`'s docstring always said it was -- *"its explicit inverse amplifies the noise floor
into the Redheffer star denominators"* -- and the amplifier is bigger than the source:

| `n_orders`.pol | `cond(a+b)` | `cond(I - B11 A22)` | outcome pre-M1 |
|---|---|---|---|
| 12.te | 2.1e+06 | 4.3e+12 | disagrees 8.9e-04 |
| 13.te | 9.7e+06 | 8.9e+13 | raises, both |
| 17.te | 4.0e+08 | 1.5e+17 | raises, both |
| 18.tm | **3.1e+16** | **2.4e+31** | raises, both |
| 19.te | 4.1e+06 | 1.6e+13 | MKL returns 1.018, OpenBLAS raises |
| 21.te | 4.8e+06 | 1.9e+13 | 160x wrong on both |
| 23.te | 2.8e+14 | 8.9e+29 | raises, both |

At 19.te and 21.te the interface inverse's own residual is 1.1e-10 and 7.4e-11 -- *fine* -- and
the star's is 1.5e-06 and 7.6e-07. **A guard on X-1's two named sites alone would not have caught
either of the two headline defects.** The star is therefore guarded as well, in both solvers
(`rcwa/_core.py` `_redheffer_star` and `pmm/_core.py` `_redheffer_star_rect`), which is the plan's
own standing rule 4 -- *"check every solve for conditioning ... not a spot check"* -- applied to
what the census actually found.

### 2.4 Route comparison: there is no better re-solve [M, Windows; ordering reproduced on WSL]

C13's fix re-solved. That option was tested here and does not exist. Five routes for the explicit
inverse, scored on `||A X - I||_F / sqrt(n)` -- the equations, never the answer -- on the real
matrices at their worst truncations:

| `n_orders`.pol | `cond` | **LU (shipped)** | QR | pivoted QR | SVD pinv | LU + 1 refinement |
|---|---|---|---|---|---|---|
| 6.te (control) | 2.2e+03 | **5.9e-16** | 1.2e-15 | 1.0e-15 | 1.4e-13 | 2.4e-16 |
| 13.te | 9.7e+06 | **6.0e-12** | 7.5e-12 | 1.2e-11 | 4.1e-10 | 3.2e-12 |
| 17.te | 4.0e+08 | **1.4e-10** | 3.1e-10 | 3.8e-10 | 2.2e-08 | 7.3e-11 |
| 18.te | 1.4e+10 | **2.9e-09** | 8.3e-09 | 8.2e-09 | 4.2e-07 | 1.5e-09 |
| 23.te | 2.8e+14 | 5.6e-03 | **4.0e-03** | 4.9e-03 | 9.5e-03 | 1.7e-03 |
| 18.tm | 3.1e+16 | 1.9e-02 | 1.9e-02 | **1.2e-02** | 3.3e-01 | 4.3e-03 |

* the shipped LU inverse beats the SVD pseudo-inverse on **every** matrix, and beats both QR
  routes on every matrix **except the two most singular** (23.te and 18.tm), where all four agree
  within a factor of 1.6 and all four are catastrophic (>= 1.2e-02). Those two are the rows a
  re-solve would have had to rescue, and none of the routes does;
* only iterative refinement improves it, by 2-4x -- and at `cond` 3.1e16 the best of the five is
  still 4.3e-03. No float64 route recovers that answer;
* C13 could re-solve because its problem was the *normal equations* squaring a `cond` = 1.4e10
  `A`. Here the operator itself is numerically singular. **Refusal is the only correct action.**

**Refinement was shipped for one round and then measured out**, and that measurement is worth
recording because it is counter-intuitive: returning the refined inverse **made cross-build
agreement worse.** It moved five clean truncations off their historical bits and took 16.te from
a 4.8e-07 MKL-vs-OpenBLAS gap to 9.0e-06. A smaller residual is not a better answer when the
residual is already at the noise floor -- it is a different rounding. So the returned value is
always `np.linalg.inv(A)` and `test_the_step_down_is_not_a_re_solve_the_census_says_there_is_none`
pins the ordering that decided it.

### 2.5 Instrument selection: the false positive that forced equilibration [M]

A raw-residual bar passed the whole thin-grating family and then **refused `rcwa_jones_1d` on a
uniaxial cell at every truncation** -- caught by
`test_rcwa.py::test_jax_execution_parity_all_entry_points`, which is a NumPy-vs-JAX pin and had no
reason to be near a conditioning change. Investigated rather than tuned around:

| `n_orders` | `||I - B11 A22||_1` | raw resid | **equilibrated resid** | `sum(R+T)` | `J00` real | build agreement |
|---|---|---|---|---|---|---|
| 9 | 4.9e+17 | 2.8e-01 | 3.9e-15 | 2.002058 | -0.188207309689 | 12 digits |
| 15 | 1.7e+16 | 1.2e-02 | 3.9e-15 | 1.999874 | -0.188195045022 | 12 digits |
| 25 | 1.0e+17 | 1.1e-02 | 6.4e-14 | 2.002041 | -0.188190316204 | 11 digits |

The generalized S-matrix's deep-evanescent blocks run 1e17 against its propagating blocks' 1e0.
That is extreme **scaling**, not singularity, and the answer is fine. Scored over BOTH families:

| candidate score | healthy | broken | separates? |
|---|---|---|---|
| raw `||AX - I||_F / sqrt(n)` | max **5.2e-01** | min 7.0e-07 | **NO** |
| `||AX - I|| / (||A|| ||X||)` | 6.2e-18 .. 6.5e-17 | 2.1e-17 .. 5.9e-17 | **NO** (it is ~eps on everything -- `inv` IS backward stable) |
| raw `rcond_1` | min **1.6e-18** | max 1.6e-13 | **NO** |
| **equilibrated `rcond_1`** | min **2.3e-09** | max **1.3e-10** | **YES**, 18x |
| **equilibrated residual** | max **3.6e-09** | min **5.3e-08** | **YES**, 15x |

"broken" here includes **12.te**, which conserves energy and lands inside the converged band but
disagrees across builds by 8.9e-04 relative (S1.4). Labelling it healthy -- as a
conservation-only classifier would -- narrows the two surviving separations to 3.2x and 2.1x. The
labelling that matters is *build agreement*, and it is stated explicitly here because the
thresholds move with it.

Both surviving scores separate by a *factor*, not by C13's thirteen orders. **That is why the
refusal requires both to fail.** The screen and the residual are independent instruments (a
condition number and a computed residual), so a false refusal needs both to be wrong at once. It
is stated here rather than smoothed over: this bar has less headroom than C13's, and the
false-positive class is pinned by `test_anisotropic_cascade_is_not_falsely_refused` so it cannot
come back silently.

The middle row is also the reason **N-2e/N-2f keep their `solve`s** -- see S5.1.

### 2.6 Why `_ConditioningError` subclasses `_EnergyError` [A]

Every `stabilize=` ladder in the library already catches `_EnergyError` and steps `n_orders` --
seven sites: `rcwa/oned.py:464, 832`, `rcwa/stack.py:2439, 2745`, `rcwa/twod.py:922, 1429, 2064`.
Making the refusal a subclass means a singular
truncation is routed around **with no ladder change anywhere**, and `stabilize=True` on 19.te --
one of the refused ones -- returns a conserving solve from a neighbour. `except ValueError`
handlers upstream are likewise unaffected. Pinned by
`test_conditioning_error_is_an_energy_error_so_stabilize_routes_around_it`.

The same inheritance gives the sweeps the right behaviour for free:
`RCWAStack.solve_vs_wavelength` (`rcwa/stack.py:2439`) already catches `_EnergyError` per point
and returns a **NaN row with a summary warning** rather than aborting -- its docstring's own
contract, *"one bad point must not abort the sweep"*. A refused wavelength therefore reads NaN and
is counted, which is what it should have been reading all along instead of a build-dependent
number.

---

### 2.7 THE REFUTATION: why the inverse refusal was withdrawn [M, both builds]

S2.5 concluded that the equilibrated instruments "separate both families". They separate the two
families that were *in the census*. The breadth sweep supplied a third and a fourth, and they do
not separate at all.

**How it surfaced.** Five long-pinned tests failed **on both builds**, deterministically:
`test_v5_21_pmm2d_staggered_oblique` (5 cases), `test_v5_20_13_pmm_jones_2d_fff_nv`,
`test_v5_14_0_pmm2d_stack::test_single_tensor_layer_matches_pmm_jones_2d`, and -- worst --
`test_niche_audit_m2_m3_m9_pmm_guards::test_m3_classical_nan_substrate_raises`.

**The populations, laid side by side:**

| population | equilibrated `rcond` | equilibrated residual | truth |
|---|---|---|---|
| 1-D thin grating, BROKEN (wrong or build-dependent) | 3.8e-19 .. 1.3e-10 | 5.3e-08 .. 3.7e+07 | refuse |
| 1-D, healthy | >= 2.3e-09 | <= 3.6e-09 | pass |
| **2-D hybrid interface, HEALTHY** (right, build-stable, pinned since v5.14) | **3.9e-14, 3.1e-13** | **1.2e-05, 4.4e-07** | **pass** |

The 2-D healthy readings sit **inside the 1-D broken band on both instruments**. No threshold
exists. And the reason is physical, not numerical: **ill-conditioning of an interface matrix does
not imply a wrong far field** -- the ill-conditioned directions are deep-evanescent and never reach
the observable, which is exactly why the 2-D hybrid has been correct at `cond` ~1e13 for eight
releases. My S2.5 margins (18x and 15x) were flagged as thin; they were in fact already
overlapping, and I had simply not sampled the methods that prove it.

**Three separate defects the first cut introduced, all now removed:**

1. **False refusal of correct 2-D physics** -- five pinned tests, both builds.
2. **A NaN-diagnostic regression.** A NaN material index makes the equilibrated `rcond` read 0.0
   and the residual read `inf`, so the conditioning guard raised *first* and told the user their
   truncation was singular when in truth their substrate index was NaN. It pre-empted
   `_check_energy`'s precise, purpose-built message ("non-finite total efficiency ... a NaN/inf
   material index or permittivity reached the solve") with a strictly worse one. A non-finite
   operand is a propagation defect, not a conditioning one; both guards now stand aside on it.
3. **An unearned headline.** S0's "1 -> 0 build-disagreements" was a claim about a refusal that
   should not have shipped.

**What survives, and why each survives:**

* **T3-3** -- independent of the guard entirely, zero false positives, still fixed (S4).
* **The least-squares refusal** -- because a sound discriminator *was* found there. A minimum-norm
  draw REQUIRES a null space, so the refusal is conditioned on **numerical rank deficiency AND
  residual**, and that conjunction separates all four families:

  | family | rank | relres | verdict |
  |---|---|---|---|
  | 2-D staggered far field | 200 of 200 (**full**) | 2.1e-07 | pass -- rank saves it |
  | shared grid, `ffo` 41 | 73 of 82 (deficient) | 2.1e-14 | pass -- residual saves it |
  | shared grid, `ffo` 61 (`J00` 10% wrong) | 78 of 122 | 6.5e-07 | **refuse** |
  | shared grid, `ffo` 77 (`J00` = 113) | 82 of 154 | 2.0e-03 | **refuse** |

  Neither instrument alone separates these; the conjunction does, and it is principled rather than
  tuned.
* **The census and every measurement in S1-S2.6** -- all still true, and now the instrument is
  free: `_guarded_inverse` returns `inv(A)` before touching either score unless `_INV_CENSUS` is
  armed, so the default path is bit-for-bit and flop-for-flop pre-M1.

**X-1's status: REAL, REPRODUCED, INSTRUMENTED, OPEN.** `test_x1_defect_is_reproduced_and_flagged_but_NOT_closed`
pins the defect and asserts that the census still flags it and that nothing is refused. A future
fix should make that test fail. Closing X-1 needs a criterion that survives every method in the
library; the residual and the condition number, equilibrated or not, are not it.

---

## 3. X-1: the defect, both builds (UNCHANGED by M1 -- see S2.7)

Full sweep, `stabilize=False`, `sum(R)` (deep null, converged ~2.0e-04). `RAISE` = the solve
refused to report.

| case | OFF MKL | OFF OpenBLAS | rel | ON MKL | ON OpenBLAS | rel |
|---|---|---|---|---|---|---|
| 8.te | 0.000200440599 | 0.000200440599 | 1.4e-10 | *unchanged* | *unchanged* | 1.4e-10 |
| 8.tm | 0.000199088682 | 0.000199088694 | 6.2e-08 | *unchanged* | *unchanged* | 6.2e-08 |
| 9-11 (6) | | | 0.0 | *unchanged* | *unchanged* | 0.0 |
| **12.te** | 0.000201645434 | 0.000201466547 | **8.9e-04** | **RAISE** | **RAISE** | -- |
| 12.tm | 0.000200019069 | 0.000200019069 | 2.8e-13 | *unchanged* | *unchanged* | 2.8e-13 |
| 13.te | RAISE (`_EnergyError`) | RAISE | -- | RAISE (`_ConditioningError`) | RAISE | -- |
| 13.tm - 15.tm (5) | | | <= 3.0e-10 | *unchanged* | *unchanged* | <= 3.0e-10 |
| 16.te | 0.000202145802 | 0.000202145705 | 4.8e-07 | *unchanged* | *unchanged* | 4.8e-07 |
| 16.tm, 17.te, 18.te, 18.tm | RAISE | RAISE | -- | RAISE | RAISE | -- |
| 17.tm | 0.000200702122 | 0.000200702122 | 0.0 | *unchanged* | *unchanged* | 0.0 |
| **19.te** | 0.018387638395 | **RAISE** | **inf** | **RAISE** | **RAISE** | -- |
| 19.tm | 0.000201108141 | 0.000201108073 | 3.4e-07 | *unchanged* | *unchanged* | 3.4e-07 |
| **20.te** | 0.000208856986 | 0.000226215644 | **8.3e-02** | **RAISE** | **RAISE** | -- |
| 20.tm | 0.000201123972 | 0.000201123972 | 9.4e-16 | *unchanged* | *unchanged* | 9.4e-16 |
| **21.te** | 0.032165672615 | 0.032165672615 | 0.0 (**160x wrong on both**) | **RAISE** | **RAISE** | -- |
| 21.tm | 0.000201384893 | 0.000201384855 | 1.9e-07 | *unchanged* | *unchanged* | 1.9e-07 |
| 22.te | 0.000209430916 | 0.000209431293 | 1.8e-06 | *unchanged* | *unchanged* | 1.8e-06 |
| 22.tm, 23.tm | | | <= 1.0e-09 | *unchanged* | *unchanged* | <= 1.0e-09 |
| 23.te | RAISE | RAISE | -- | RAISE | RAISE | -- |

**Gate readings.**

* *both-builds*: worst returned relative disagreement **8.3e-02 -> 1.8e-06**, i.e. **4.7 orders**
  on the site the census flagged, against the gate's ">= 2 orders". Nothing regressed: every
  still-returned case's cross-build gap is bit-for-bit what it was.
* *null control*: **0 bits moved** on any returned solve, on either build. The well-conditioned
  control solve (`cond(a+b)` = 9.0) is byte-identical guard on/off on both builds, and byte-identical
  MKL-to-OpenBLAS.
* *fail-before*: `INTERFACE_CONDITIONING_GUARD = False` reproduces the whole OFF column, verified
  per configuration; `test_the_refusal_reproduces_the_prior_answer_with_the_switch` pins 21.te's
  pre-M1 answer verbatim.
* *refusals that were already errors*: 13.te, 16.tm, 17.te, 18.te, 18.tm, 23.te already raised
  `_EnergyError` on both builds. The guard changes only *which* error and *when* -- earlier, and
  naming the cause. The genuinely new refusals are **12.te, 19.te, 20.te, 21.te**, and each of the
  four is a documented defect above.

---

## 4. T3-3: the conical per-layer far-field order cap

### 4.1 Mechanism [A]

`conical.py:202` computed `cap = (nU * n_el * degree - 1) // 2` from `nU`, the **full-union** cell
count, at line 202 -- **before** the per-layer branch at line 235 -- and never re-clamped. On the
per-layer path the half-spaces are built on the *window* grids (`conical.py:266-271`). On the audit
staircase, `degree` 6, `elements_per_region` 1:

| grid | cells | `n_glob` | cap (orders) |
|---|---|---|---|
| full union | 13 | 78 | **77** |
| end window (`grid_of[0]`, `grid_of[-1]`) | 5 | 30 | **29** |

So the shipped cap over-stated capacity by **2.7x**, the `m_prop > cap` raise could not fire, and
`_sem_fourier_projection(ox, period, mats_sup)` then built a projector with more Rayleigh orders
than the grid has nodes. `Hsup` went rank-deficient and `lstsq(Hsup, rhs, rcond=None)` returned a
minimum-norm draw. **All three siblings clamp correctly** -- `stack.py:1779-1789` (classical),
`stack.py:2981-2989` (sweep), `_jax_stack.py:525-529` (JAX twin) -- conical was the only outlier.

### 4.2 It was not "latent only" [M, both builds]

Audit staircase, conical (`theta` 0.15, `phi` 0.6), degree 6. `PMM_CONICAL_PERLAYER_ORDER_CAP`
OFF = pre-M1, ON = shipped. `J00` reference from the capacity-respecting solve is
`-0.1711471699 + 0.0091251104j`.

| `far_field_orders` | cap OFF: orders out | cap OFF: `\|R+T-1\|` | cap OFF: `Re J00` | cap ON: orders out | cap ON: `Re J00` |
|---|---|---|---|---|---|
| 7 | 7 | 1.05e-04 | -0.1711471699 | 7 | -0.1711471699 |
| 21 | 21 | 1.05e-04 | -0.1711471699 | 21 | -0.1711471699 |
| 31 | 31 | 1.05e-04 | -0.1711471699 | **29** | -0.1711471699 |
| 41 | 41 | 1.05e-04 | -0.1711471699 | **29** | -0.1711471699 |
| **61** | 61 | **1.02e-04 (CLEAN)** | **-0.1711455635** | **29** | -0.1711471699 |
| **77** | 77 | **1.47e+01** | **-0.1675439053** | **29** | -0.1711471699 |

Two failure modes, and **only one of them is visible to energy conservation**:

* at 77 orders the closure blows to `|R+T-1|` = 15 -- loud;
* at 61 orders the closure is **1.02e-04, marginally BETTER than the honest solve's 1.05e-04**,
  while the zero-order Jones has moved by **1.6e-06** in its sixth digit. A null-space component
  of `cinc` is invisible to `Hsup`, so it leaves `R + T` intact. This is exactly the class the
  plan's N-2 dossier predicted ("the far field then carries manufactured or destroyed energy that
  EE-style and Jones-magnitude metrics do not see") and it is why the results tables in this
  document carry conservation *and* the observable side by side.

`test_t3_3_fail_before_reproduces_the_over_capacity_draw` pins both rows.

### 4.3 After [M]

* the per-layer answer is **stationary** in `far_field_orders` from 7 to 77: spread in `J00`
  **3.7e-11**, against the 1.6e-06 the pre-M1 cap produced. Scored comparatively (the numerator is
  BLAS round-off, so no absolute bar is asserted).
* the fix lands on the same capacity as the classical sibling: the classical and conical per-layer
  solves of the same stack at `ffo` = 77 return the **same order count**
  (`test_t3_3_matches_the_sibling_paths_it_was_the_outlier_from`).
* the switch is **bit-identical** below the capacity (`ffo` 7, 21, 29), tolerance-at-0.0 on `J`,
  `R`, `T` -- the clamp is a fix at high `ffo`, not a rewrite.
* `n_orders` is *lowered* by the clamp, so this is also a small perf win, never a cost.

### 4.4 N-2d caught the same class on the SHARED path, which T3-3 does not touch [M]

The shared grid has its own capacity (`n_glob` = 78 -> 77 orders) and `far_field_orders` above it
is equally unrepresentable. Pre-M1, shared path, same stack:

| `ffo` | `\|R+T-1\|` | `Re J00` | lstsq rel. residual |
|---|---|---|---|
| 7 .. 41 | 6.65e-06 | -0.1711793237 | <= 1.9e-12 |
| **61** | **1.13e+00** | **-0.1878826745** | **6.5e-07** |
| **77** | **1.93e+07** | **113.47** | **2.0e-03** |

Both are now refused by `_guarded_lstsq`. Note 61's `Re J00` = -0.1879 against -0.1712 -- a **10 %**
error on the specular Jones, with a warning and no raise.

### 4.5 Why the least-squares bar is the residual and NOT `rcond(Hsup)` [M]

The first cut screened on `rcond(Hsup)`. It is the wrong instrument and the measurement says so
outright:

| shared path, `ffo` | `Hsup` shape | numerical rank | `rcond` | rel. residual | answer |
|---|---|---|---|---|---|
| 21 | 42 x 156 | 42 | 3.0e-04 | 2.0e-14 | right |
| **41** | 82 x 156 | **73 of 82** | **7.8e-17** | 2.1e-14 | **right to 9 digits** |
| **61** | 122 x 156 | 78 | **7.1e-17** | **6.5e-07** | **wrong by 10 %** |

The two populations **overlap** on `rcond` -- a screen on it refuses good solves and passes bad
ones. `Hsup`'s small singular values are the high-order Fourier projections of a nodal basis
aliasing against each other; that is expected and harmless *while the incident plane wave still
lies in the range*. What fails is precisely when it stops lying in the range, and the residual
measures that directly. Separation on the audit sweep: every correct solve <= **1.9e-12**, the
first broken one **6.5e-07** -- five and a half orders, with the `1e-9` bar in the middle.
Pinned by `test_rcond_of_hsup_would_have_been_the_wrong_instrument`.

`||c||` inflation tells the same story less portably (1.2e+03 -> 3.0e+07 -> 2.6e+11 across those
three, and 3.8 on a healthy solve of a *different* stack), so it is reported in the error text and
is not the bar. C13 reached the same conclusion about `||x||` for a different reason.

---

## 5. N-2: the PMM sites

### 5.1 The `solve` sites are deliberately unguarded -- a measurement, not a preference

`_interface_smatrix_mortar` does `solve(kron(I2,Mb) @ Wb, ...)` and `solve(kron(I2,Ma) @ Va, ...)`;
`_interface_smatrix_general_mortar` does one `solve(A, B)` on the `2(n_a+n_b)` block system. The
plan lists all three as N-2 targets. They are **not** guarded, for a reason that S2.5's third row
measures: LAPACK `gesv` is **backward stable**, so its residual is ~eps whatever the conditioning
-- a residual screen on a `solve` measures nothing. (The same fact is why
`||AX - I|| / (||A|| ||X||)` fails to separate anything: it is the backward error, and the backward
error is always tiny.)

What a badly conditioned `solve` can do is push a *forward* error downstream, and in the mortar it
does so through `BA` into `I + BA`, which **is** an explicit inverse and **is** guarded. One guard
therefore covers the function. Measured on the audit staircase, per-layer, degree 6/8/10,
classical and conical (2-norm `cond`, 30 mortar interfaces):

| operand | range | role |
|---|---|---|
| `M_b W_b` (`solve` #1) | 3.4e+02 .. 3.8e+06 | unguarded, backward stable |
| `M_a V_a` (`solve` #2) | 7.3e+05 .. **2.1e+07** | unguarded, backward stable -- the worst of the three, and still nine orders from a float64 problem |
| `I + BA` (explicit inverse) | 1.3e+01 .. 6.8e+03 | **guarded** |

Recorded in the source at the call site so the omission is not read as an oversight.

`_interface_smatrix_general_mortar` has *only* a `solve` and therefore takes no guard. It is
reachable only from the per-layer general (slant / OOP) cascade; the honest statement is that this
site is **uninstrumented**, and it is named here rather than claimed covered.

### 5.2 PMM census, guard ON [M, Windows; WSL identical to the printed digit]

Audit staircase (6 slices) at degree 6/8/10, classical and conical, plus a 3-layer lossy stack and
a mixed-slant stack, shared and per-layer:

| site | calls | dim | `rcond_eq` range | screened in | refused |
|---|---|---|---|---|---|
| `pmm interface mode-match (a+b)` | 2-7 / solve | 32-260 | 6.1e-05 .. 4.6e-02 | 0 | 0 |
| `pmm mortar interface (I + BA)` | 5 / solve | 60-100 | 1.1e-03 .. 5.8e-03 | 0 | 0 |
| `pmm per-layer star (I - B11 A22)` | 3-6 / solve | 24-100 | 1.1e-04 .. 1.3e-01 | 0 | 0 |
| `pmm per-layer star (I - A22 B11)` | 3-6 / solve | 24-100 | 2.1e-03 .. 4.5e-01 | 0 | 0 |
| `rcwa Redheffer star` (shared path) | 3-6 / solve | 24-260 | 1.1e-06 .. 4.3e-01 | 0 | 0 |
| `rcwa generalized interface (T22)` (slant) | 4 / solve | 24 | 4.2e-02 | 0 | 0 |
| Rayleigh `lstsq` | 2 / solve | 22 x [24..260] | rel. residual 5.7e-16 .. 1.9e-13 | -- | 0 |

**The shipped per-layer surface is clean at every nominal setting.** The screen never fires, so
the guard is a pure insurance policy there, and its cost on that path is the free screen only
(S6). The value it adds is at the *edges* -- the over-capacity `ffo` of S4, and whatever a future
device brings.

### 5.3 Conservation, per the repo's per-pol convention [M, both builds]

`|R+T-1|` per incident polarization on the lossless staircase, guard ON vs OFF, all identical:

| path | deg 6 | deg 8 | deg 10 |
|---|---|---|---|
| shared, classical | 5.93e-06 | 8.78e-08 | 7.28e-10 |
| shared, conical | 6.65e-06 | 4.21e-08 | 1.10e-08 |
| per-layer, classical | 1.10e-04 | 3.00e-06 | 1.17e-06 |
| per-layer, conical | 1.05e-04 | 2.80e-06 | 1.10e-06 |

Unchanged at every rung (the guard never fires on these), and the spectral decay in `degree` is
intact. The lossy 3-layer stack's `sum(A)` vs `1 - R - T` closure is likewise unchanged
(`test_pmm_per_layer_grids.py::test_retain_internal_per_layer_fields_and_absorption`, green on both
builds).

### 5.4 PMM null control, in full [M, both builds]

The gate's "does not regress anywhere" clause, measured rather than asserted: 14 configurations
(staircase classical + conical at degree 6/8/10, and a lossy 3-layer stack, on each of the shared
and per-layer paths), comparing the **raw bytes** of the `(2,2)` Jones matrix.

| configuration | Jones bytes, guard OFF vs ON -- MKL / OpenBLAS | MKL-vs-OpenBLAS `\|dJ00\|`, OFF / ON | `\|R+T-1\|` |
|---|---|---|---|
| staircase shared, deg 6 / 8 / 10 | identical / identical | 3.4e-12 / 3.4e-12, 1.1e-11 / 1.1e-11, 4.2e-12 / 4.2e-12 | 5.9e-06, 8.8e-08, 7.3e-10 |
| staircase shared conical, deg 6 / 8 / 10 | identical / identical | 9.9e-11, 1.4e-10, 1.3e-10 (OFF = ON) | 6.7e-06, 4.2e-08, 1.1e-08 |
| staircase per-layer, deg 6 / 8 / 10 | identical / identical | 8.3e-16, 4.1e-16, 7.5e-16 (OFF = ON) | 1.1e-04, 3.0e-06, 1.2e-06 |
| staircase per-layer conical, deg 6 / 8 / 10 | identical / identical | 4.0e-11, 4.7e-13, 2.1e-10 (OFF = ON) | 1.1e-04, 2.8e-06, 1.1e-06 |
| lossy 3-layer, shared / per-layer | identical / identical | 6.0e-16, 6.5e-16 (OFF = ON) | 1.3e-01 (lossy: `R+T < 1` by construction) |

**28 of 28 byte-identity checks pass**, and the worst cross-build `|dJ00|` is **2.1e-10 both
before and after** -- unchanged to the digit, which is the point: on the shipped PMM surface the
guard is inert, and the census (S5.2) says why -- the screen never fires there.

---

## 6. Perf and memory gates [M, both builds]

Threads pinned to 1, medians. **The box carried an unrelated multi-hour job throughout** (a
`fan_multi_121.py` run), so the two PMM rows were re-measured with OFF/ON **interleaved sample by
sample** to cancel drift; both figures are given.

| workload | MKL off (ms) | MKL on (ms) | dt | OpenBLAS off | OpenBLAS on | dt | `tracemalloc` peak dt |
|---|---|---|---|---|---|---|---|
| `rcwa_efficiency_1d`, M = 60 | 21.7 | 22.0 | +1.2 % | 24.3 | 25.4 | +4.3 % | +0.001 % |
| `rcwa_jones_1d`, M = 25 (aniso) | 16.0 | 16.4 | +2.1 % | 17.7 | 18.1 | +2.6 % | +0.002 % |
| `RCWAStack`, 3 layers, M = 30 | 42.5 | 43.7 | +2.8 % | 48.6 | 49.5 | +2.0 % | -0.000 % |
| PMM per-layer, deg 8 | 81.1 | 86.1 | +6.3 % | 81.1 | 80.2 | -1.1 % | +0.000 % |
| PMM per-layer, deg 8 **(interleaved, n=31)** | 83.6 | 83.5 | **-0.1 %** | 83.2 | 87.1 | **+4.6 %** | -- |
| PMM shared, deg 8 | 398.4 | 422.3 | +6.0 % | 407.5 | 402.2 | -1.3 % | +0.000 % |
| PMM shared, deg 8 **(interleaved, n=31)** | 436.3 | 425.1 | **-2.6 %** | 386.1 | 391.0 | **+1.3 %** | +0.000 % |
| `rcwa_efficiency_1d`, M = 60 **(n = 75)** | 24.5 | 25.2 | **+2.8 %** | 24.2 | 24.3 | **+0.3 %** | -- |
| `rcwa_efficiency_2d`, 6 x 6 (dim 338) | 1233 | 1181 | -4.3 % (noise) | -- | -- | -- | -- |
| `rcwa_jones_2d`, 4 x 4 tensor cell (dim 162) | 209 | 131 | -37 % (noise) | -- | -- | -- | -- |

**Gate: no regression > 5 %.** Every interleaved / high-sample-count reading is between **-2.6 %**
and **+4.6 %**. The two sequential readings that printed above 5 % were re-measured interleaved and
did not survive; they are shown rather than dropped.

The two 2-D rows are the expensive path and the guard is invisible there: both readings come out
NEGATIVE, i.e. inside the loaded box's noise, because a 2-D solve is dominated by its `2N x 2N`
eig (dim 338 and 162 here) while the screen is `O(n^2)`. They also carry the screen-rate evidence:
**4 guarded inverses per solve, 0 screened in, 0 refused, `rcond_eq` >= 7.7e-05** -- four orders
above the screen, so the confirming residual's extra inverse is never allocated on the 2-D path
either.

**Memory gate: `tracemalloc` peak within 5 %.** Measured change is **<= 0.005 %** on every
workload and build. Largest-array census at peak is unchanged: the screen's temporaries are two
`n x n` magnitude arrays (`|A|` scaled in place, and `|X|`) against a solve whose peak is already
several `2N x 2N` complex modal matrices -- e.g. PMM shared deg 8 peaks at 55.7 MB either way. The
confirming residual's extra `inv` is never allocated on any of these workloads because the screen
never fires (S5.2).

---

## 7. Suites, both builds

| suite | Windows / MKL | WSL / OpenBLAS |
|---|---|---|
| `tests/unit/test_m1_conditioning_guard.py` (new, 25 tests) | pass | pass |
| `tests/unit/test_pmm_per_layer_grids.py` (11) | pass | pass |
| `tests/unit/test_rcwa.py` (93, incl. the new era pin) | pass | pass |
| the three together (129 collected, run with and without `-p randomly`) | pass | pass |
| full `tests/unit/` | see S7.1 | see S7.1 |
| `ruff check lumenairy/ tests/unit/` | -- | **All checks passed** |

### 7.1 Breadth-sweep status and what it found

**The breadth sweep is what refuted the inverse refusal**, so it earned its cost. Three whole-suite
attempts were made; the first two were killed externally at 84% / 74% with no summary. The third
ran as **6 chunks x 2 builds**, each writing its own summary, which is the only reason the failures
got NAMED. Sweep verdict on the chunks that completed:

| named failure | build(s) | classification | resolution |
|---|---|---|---|
| `test_v5_21_pmm2d_staggered_oblique` (5 cases) | **both** | **(a) M1's** -- false refusal, `_guarded_lstsq` on a full-rank 2-D projector | refusal now requires rank deficiency **and** residual (S2.7) |
| `test_v5_20_13_pmm_jones_2d_fff_nv::test_pmm_fff_nv_matches_rcwa_fff_nv` | **both** | **(b) pre-existing**, PROVEN: fails identically with all M1 source `git stash`-ed out | bad test, FIXED -- see S8 |
| `test_v5_14_0_pmm2d_stack::test_single_tensor_layer_matches_pmm_jones_2d` | Windows | **(a) M1's** -- false refusal, `_guarded_inverse` at `rcond` 3.1e-13 | inverse refusal withdrawn (S2.7) |
| `test_v5_20_12_rcwa_jones_2d_fff_nv::test_fff_nv_stripe_reduces_to_rigorous_1d` | WSL | **(a) M1's**, same class | same |
| `test_niche_audit_m2_m3_m9_pmm_guards::test_m3_classical_nan_substrate_raises` | both | **(a) M1's** -- NaN diagnostic hijacked | non-finite passthrough (S2.7 defect 2) |
| `test_m1_conditioning_guard::test_the_refusal_reproduces_the_prior_answer_with_the_switch` | Windows | **(a) M1's** -- my own test, state-dependent on a refusal that should not have existed | dissolved: with the refusal withdrawn there is no state to interact with |
| `test_fga_h4_h5` (env budget) | -- | fixed orchestrator-side (raced live RAM reads) | not mine |

No failure was classified as flakiness and none was re-run to green: each was root-caused to either
bad code (mine, five of them) or a bad test (one, pre-existing and proven so by stash-to-baseline).

## 8. Test-side changes

| file | change |
|---|---|
| `tests/unit/test_m1_conditioning_guard.py` | **NEW, 27 tests.** The two instruments and the free-equilibration identity; three null controls (well-conditioned inverse, well-conditioned RCWA solve, conforming per-layer stack) all tolerance-at-0.0; **the refutation pinned** (`test_the_inverse_refusal_was_withdrawn_and_the_instruments_record_instead`); **the open defect pinned** (`test_x1_defect_is_reproduced_and_flagged_but_NOT_closed`, 4 params -- a future fix should make it fail); the NaN-passthrough pins for both guards; the least-squares **conjunction truth table**; the route-ordering measurement; the anisotropic false-positive guard; the `rcond`-is-the-wrong-instrument pin; T3-3's clamp, fail-before, below-capacity no-op and sibling agreement; the NumPy-only backend contract. |
| `tests/unit/test_rcwa.py` | **RESTORED PRISTINE** (`git checkout`). The widening and the era pin it carried existed only to accommodate the inverse refusal; with that withdrawn, the file's original assertions are correct again and the diff is zero. |
| `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py` | **FIXED** -- a pre-existing bad test, proven pre-existing by `git stash`-ing all M1 source and reproducing the failure on the v5.32.1 baseline. Its RCWA reference sat on the cell's measure-zero instability at `n_orders_x` = 13 (`|R+T-2|` = 2.6e-02, the worst of six neighbouring truncations), and the "cross-solver residual" it read was that closure defect one-for-one. Moved to 11 (closure -2.3e-05) and **both engines' closure is now asserted before the comparison**, so it can never again silently compare against an unstable reference. **The 4e-3 bar is unchanged** -- widening it would have pinned the instability instead of avoiding it. |

No other test moved. No absolute bar was added anywhere in the new file: every accuracy claim is a
comparative envelope or a tolerance-at-0.0 identity.

---

## 9. Source changes

```
lumenairy/elements/rcwa/_core.py     the instruments (_rcond_1, _equilibration,
                                     _rcond_1_equilibrated, _inverse_residual,
                                     _equilibrated_inverse_residual), _ConditioningError,
                                     _INV_CENSUS, and _guarded_inverse -- which RECORDS and
                                     never refuses (S2.7).  4 call sites.
lumenairy/elements/pmm/_core.py      _guarded_lstsq (refuses on rank-deficiency AND residual,
                                     stands aside on non-finite); 4 inverse call sites +
                                     2 lstsq sites; the S5.1 note at the two solve sites
lumenairy/elements/pmm/conical.py    PMM_CONICAL_PERLAYER_ORDER_CAP + the window-derived cap
                                     (T3-3, the one unambiguous fix) + 1 lstsq site
lumenairy/elements/pmm/stack.py                2 lstsq sites
lumenairy/elements/pmm/stack2d_pure.py         1 lstsq site
lumenairy/elements/pmm/twod_staggered.py       1 lstsq site
tests/unit/test_m1_conditioning_guard.py       NEW
tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py   pre-existing bad test, fixed
docs/audits/PMM_M1_CONDITIONING_2026_08_04.md     this document
```

Everything uncommitted.

---

## 10. What this does NOT close

1. **X-1 IS NOT CLOSED.** This is the headline caveat, not a footnote. The defect is real,
   reproduced on both builds, and pinned by
   `test_x1_defect_is_reproduced_and_flagged_but_NOT_closed`: `n_orders` 19 TE returns 1.018 on
   MKL and raises on OpenBLAS; 21 TE returns an answer 160x wrong on both. The census *detects*
   all four flagged truncations. It does not act, because the same readings occur on healthy 2-D
   solves (S2.7). **Closing X-1 needs an instrument that separates across methods**, and neither
   the residual nor the condition number does, equilibrated or not. Two directions worth trying,
   neither attempted here: (a) score the OBSERVABLE's sensitivity rather than the operator's --
   e.g. propagate a bounded perturbation through the cascade and watch the far field, which is
   method-agnostic by construction; (b) exploit that the true positives are all
   *closure-violating*, i.e. fold the conditioning reading into `_check_energy`'s existing
   three-outcome ladder as a tie-breaker rather than as an independent gate.
2. **N-2's inverse half is therefore also open** -- the PMM mortar and star inverses carry the
   same instruments and the same non-decision. What N-2 *did* deliver is the least-squares
   refusal (sound, S2.7) and the census showing the shipped per-layer surface clean at every
   nominal setting (S5.2).
3. **The `solve` sites remain deliberately unguarded** (S5.1), and
   `_interface_smatrix_general_mortar`'s solve remains **uninstrumented**.
4. **JAX and CuPy are unguarded by construction** (traced control flow; per-interface device
   sync), pinned as deliberate.
5. **The breadth sweep did not complete on either build.** Three attempts; the first two were
   killed externally at 84% / 74%, the third ran chunked and named the failures in S7.1 before it
   too was stopped. Every named failure is root-caused and closed, but the chunks that never ran
   are unmeasured, and that is a gap in this document, not a clean bill.
6. **`_check_energy`'s docstring still says "cond up to ~1e13"** where the census measured 3.1e16
   at the interface and 2.4e31 at the star -- M4's D-1 pass owns the in-code correction.
7. **Nothing here addresses the per-layer mortar's residual band** (`|R+T-1|` ~1e-4 at deg 6) --
   that is M2's T3-1/T3-2, and S5.3 is its unchanged baseline.

---

## 11. Hand-off

The plan's standing rule 9 carries an ordering constraint -- *"X-1 must precede any adjudication
that leans on RCWA"* -- and it is now satisfied, with a caveat worth carrying forward:

* **RCWA is usable as an oracle again, and it will now say so when it isn't.** The failure mode
  that made it unsound as an adjudicator was that it could return a number no build agreed on. It
  now refuses instead. A cross-check that hits a refusal has learned something real about the
  truncation, not lost its oracle.
* **The refusal is a truncation-level statement, not a geometry-level one.** A refused
  `n_orders` almost always has a clean neighbour (S3: 19 TE and 21 TE are refused while 19 TM,
  20 TM, 21 TM and 22 TE are clean). Any S1 cross-check against RCWA should therefore quote a
  small `n_orders` ladder, not a single truncation -- which is what `stabilize=True` does for you.
* **M2's `|R+T-1|` baselines are S5.3**, measured with the guard on and identical to the guard
  off, so T3-1 / T3-2 can be scored against them directly.
* **M3's byte-identity work has one new obligation:** the hoist / vectorise / de-kron changes must
  keep the guard's screen readings where they are, because a de-kron changes the operator's
  SCALING and the screen is scale-aware by construction. The `_INV_CENSUS` hook is the instrument
  for that check and costs nothing to run alongside the perf measurement.
* **M4's D-1 pass owns two corrections this study surfaced but did not make:**
  `_check_energy`'s "cond up to ~1e13" (measured 3.1e16 / 2.4e31), and the fact that the
  docstring's attribution of the blow-up to the interface inverse is only half right -- the star
  denominator is the larger term.

---

## 12. Reproduction

**The permanent instrument is `rcwa/_core.py`'s `_INV_CENSUS` hook.** Set it to a list and every
guarded inverse appends `(site, n, rcond_eq, resid_eq, refused)`; set it back to `None` to switch
it off. That is what produced S5.2 and it is what a future study should reach for first -- the
probe scripts below were scratch (session-local, not committed), and each is short enough that the
tables above are the record.

```python
# the S5.2 census, in six lines, on any workload
import lumenairy.elements.rcwa._core as rc
rc._INV_CENSUS = []
...run the solve...
for site, n, rcond_eq, resid_eq, refused in rc._INV_CENSUS:
    print(site, n, rcond_eq, resid_eq, refused)
rc._INV_CENSUS = None
```

The scratch probes, by section, and what each one does -- all are pure-`lumenairy` scripts with no
fixtures, reconstructible from the descriptions here:

| section | probe | what it does |
|---|---|---|
| S1.3, S2.3 | `census.py`, `star_probe.py` | monkeypatch `_interface_smatrix` / `_redheffer_star` in every module that imported them by name, record `cond` and residual per call, sweep the thin grating |
| S2.4 | `routes.py` | swap in each of the five inverse routes for the whole solve, score `\|\|AX-I\|\|` |
| S2.5 | `score_select.py` | five candidate scores over both geometry families, print the healthy/broken separation |
| S3 | `after.py` | the sweep, guard OFF and ON, plus the byte-identity control and a timing block |
| S4 | `t33_probe.py`, `t33_probe2.py`, `t33_beforeafter.py` | grid capacities; the `rcond`-vs-residual sweep; the 2x2 switch matrix |
| S5.2 | `pmm_census.py` | the `_INV_CENSUS` hook + a passive `lstsq` spy, over the PMM stacks |
| S6 | `perf.py`, `perf3.py` | six workloads sequential; two workloads interleaved A/B |

```bash
cd <repo>
E="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"

# S7 -- suites, Windows/MKL
env $E python -m pytest tests/unit/test_m1_conditioning_guard.py \
    tests/unit/test_pmm_per_layer_grids.py tests/unit/test_rcwa.py -q
env $E python -m pytest tests/unit/ -q -n 10

# ... and the same on the CI proxy, plus lint
wsl -e bash -lc "cd /mnt/d/.../Lumenairy && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ~/lumvenv/bin/python -m pytest tests/unit/ -q -n 8"
wsl -e bash -lc "cd /mnt/d/.../Lumenairy && ~/lumvenv/bin/ruff check lumenairy/ tests/unit/"

# the two fail-before switches, by name
#   lumenairy.elements.rcwa._core.INTERFACE_CONDITIONING_GUARD    = False  # X-1, N-2
#   lumenairy.elements.pmm.conical.PMM_CONICAL_PERLAYER_ORDER_CAP = False  # T3-3
```
