# The degree-6 cliff was never about degree: the traced fits' Gram matrix

Niche C13, 2026-08-03.  Branch `feat/d121-final-closure`, on top of the C12
tree (`_lens_traced.py 134c9e8f199c3413` before this study,
`06c590fe1bb800b7` after; `carrier.py 5a1b0d1021969df1`, **unmodified by this study**).

Predecessors: `C11_PHYSICAL_DECENTRE_GATE_2026_08_03` S9.5 (which named the
symptom and handed it over), `D121_RESIDUAL_CLOSURE_2026_08_02` (which shipped
degree 6), `C8_INVERSE_SUPPORT_BOUND_2026_08_01` and
`C6_FIT_GUARD_DECISION_2026_07_31` (the earlier build-divergence episodes in
this same fit family).

---

## 0. Headline

C11 S9.5 handed over a release blocker: `test_niche_d4_dgrating::
test_matches_the_manual_hand_split`, run ALONE, reads **5.93e-07 on Windows
and 8.80e-02 on Linux** -- a 148,000x route disagreement in the shipped
configuration, with a 37,000x CLIFF appearing at exactly
`_REMAP_RESID_EIKONAL_DEGREE = 6` and degrees 2-5 smooth.  The hypothesis it
was handed over with was that niche C10's degree-6 residual-eikonal fit is
ill-conditioned.

**It is not.**  That fit's design matrix reads `cond(A)` = **4.08e2** and its
Gram `cond(G)` = **1.67e5** -- identically on both builds -- and its
normal-equations solution matches an SVD reference to **2e-13**.  Degree 6 is
the STIMULUS, not the defect.

**The defect is one function downstream, and it is general.**
`_solve_lstsq_thread_safe` solves EVERY traced fit through the normal
equations, on an argument written into its own docstring: that `A` is *"a
well-conditioned normalised tensor-Chebyshev / monomial Vandermonde (~1.5x
oversampled), so squaring the condition number in `G` is safe"*.  That is true
of the concentric unweighted fits it was written for.  It is **false** of the
weighted ones D1/D7 introduced: `_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8` splits
the rows into two scales, and on D4's own chain the order-10 (66-term) OPL and
exit-coordinate fits read

| | `cond(A)` | `cond(G)` | normal-equations fit residual |
|---|---|---|---|
| the C10 residual-eikonal fit (27 terms) | 4.08e2 | 1.67e5 | at the minimum, both builds |
| **the weighted OPL / coordinate fits (66 terms)** | **1.4e10** | **>= 4e15 (formally 1.9e20)** | **1.05x the minimum on Windows, 14.8-23.0x on Linux** |

`cond(G)` past float64 means the Cholesky answer is an arbitrary draw from the
numerical null space.  Both builds take a garbage draw; Windows' happens to
fit, Linux's does not, and the exit field on the losing build is **speckled at
the pixel scale** (roughness `max|lap E|/peak` **0.2999** against 0.0353) --
numerical noise, not an approximation error.  Degree 6 moves the OPL samples
just enough to move which draw each build takes.

**The fix is a measured step-down, not a cap and not a new degree.**  Screen
the equilibrated Gram; where it is numerically singular, re-solve by
backward-stable Householder QR of `[A | b]` (`geqrf`, never `gelsd` -- B7's ban
survives) and keep whichever answer measurably fits the data better.  A fit
the normal equations already get right returns its **historical bits**.

**Result, both builds, shipped degree 6:**

| | Windows | Linux | agreement |
|---|---|---|---|
| D4 sphere arm, **before** | 5.9277e-07 | **8.7976e-02** | 148,000x apart |
| D4 sphere arm, **after** | 6.7181e-07 | 6.7186e-07 | **8e-05 relative** |
| D4 parabola arm, before | 3.6092e-07 | 2.4871e-07 | 45 % apart |
| D4 parabola arm, after | **1.1421e-10** | **1.1421e-10** | identical to 5 figures |
| the SAME exit field across builds, before | 6.79e-02 max / 5.37e-02 p99.9 | | |
| the SAME exit field across builds, after | **1.21e-10** max / 1.02e-10 p99.9 | | |

The degree sweep is smooth 2..6 on both builds, degree 6's per-order closure on
design 121 is reproduced to every printed digit on **both** builds (it had only
ever been measured on one), the production acceptance line
**3.350 um / 90.3 / 99.7 / 99.8** is unchanged including the peak, conservation
is 6 of 6, and the fail-before reproduces all six of design 121's recorded
degree-6 intensity hashes **bit for bit**.

**And a scope addition** (S10, adjudicated in S11): the ray-fit branch
selector was ordered ON for 5.32.1 mid-study -- for BOTH constants.  It lands
here because it runs straight through the machinery C13 had just made stable:
on design 121 the C12 spectrum is unresolved, so what the selector arbitrates
is a pair of least-squares fit residuals -- exactly the numbers that were
being answered by a null-space draw.

**Only one of the two survived.**  `DECENTRED_FIT_ARBITER` ships `True` and
reproduces C11's per-order table on both builds.  `DECENTRED_FIT_PREDICTOR`
was REVERTED to `False`: it reddens 9 tests across niches D6/D7, costs **32 %**
of the encircled energy on D6's analytic `K = -n^2` fixture (EE2/oracle 0.9819
-> 0.6670, spot 40x further off the Fermat focus) because its closed form
inverts the branch ranking that the arbiter measures 17,000x the other way --
and fires **zero** disagreement warnings across four design-121 runs.  Inert
where it was wanted, harmful where it is live.

---

## 1. Provenance

### 1.1 The two builds, named correctly

C11 S9.5 calls them "Windows / MKL" and "Linux / OpenBLAS".  **Neither uses
MKL.**  Both NumPy and SciPy are `scipy-openblas` wheels on both machines; what
differs is the `DYNAMIC_ARCH` kernel OpenBLAS selects, the Python, and the
NumPy patch level:

| | Windows | Linux (the CI proxy, `~/lumvenv`) |
|---|---|---|
| Python | 3.14.6 | 3.12.3 |
| NumPy | 2.4.4 | 2.4.6 |
| SciPy | 1.17.1 | 1.17.1 |
| BLAS/LAPACK | scipy-openblas 0.3.31.188.0 | scipy-openblas 0.3.31.188.0 |
| kernel | `Haswell MAX_THREADS=24` | `SkylakeX MAX_THREADS=64` |

This matters to the diagnosis rather than to the fix: the two builds differ
**only in rounding**, so a quantity that disagrees between them by 148,000x is
being computed by a process with no numerical stability at all.  The rest of
this document keeps the short names "Windows" and "Linux" and means these two.

### 1.2 What is measured against what

Every "before" number in this document is taken with
`LSTSQ_CONDITIONING_STEPDOWN = False`, which is the pre-C13 function bit for
bit, in the same process and on the same tree as the "after" number.  Nothing
is compared against a remembered value except where an earlier document's
recorded digits are quoted, and those are quoted to be REPRODUCED (S6.1).

The `A.shape[0] < m` guard in `_solve_lstsq_qr` was added after the design-121
tables of S6 were taken.  It is unreachable for every fit in this library --
every caller enforces a samples-per-term floor before it solves -- so no run in
this document can have taken it; it exists so the helper is total.

**Which selector every number was taken under.**  Half-way through this study
the ray-fit selector default was flipped on by an explicit user decision
(S10).  Sections 2-6 are all at the **v5.32 selector**
(`DECENTRED_FIT_PREDICTOR = False`, `DECENTRED_FIT_ARBITER = False`), which is
the configuration the blocker was reported in and the one C11 S9.5 measured;
they are the ATTRIBUTION and the fail-before.  Section 10 re-takes the
acceptance in the **shipped 5.32.1 configuration** (both `True`), which is
what CI will run.  Every table says which it is, and no claim is carried from
one to the other by assumption.

### 1.3 Null floors

Nothing below is reported without the floor it must beat.

| null | measured |
|---|---|
| same build, same process, two identical chain runs (`energy_stage_audit_121.py` NULL intervention, order (0,0), all six stages) | `array_equal = True`, `max abs dE` = **0.000e+00** |
| same build, the D4 chain re-run in a fresh process | bit-identical (`test_it_is_deterministic`) |
| the two builds on the **upstream** field the defect does not touch (chain A, group 1 only), after the fix | **4.08e-16** max, 2.24e-16 p99.9 -- i.e. machine precision |
| the two builds on that same field, **before** the fix | 1.38e-05 max, 1.25e-05 p99.9 |

The last row is worth stating plainly: even the single-group on-axis leg was
build-divergent at 1e-05 before this fix and is at 1e-16 after it.

### 1.4 Sampling

Field comparisons are reported as `max`, `p99.9`, `p99` and `rms` of
`abs(dE)` normalised by the reference peak, so no claim in this document rests
on one pixel.  Where only one number is quoted it is the max, and the p99.9
never disagrees with it by more than a factor 1.3 anywhere in S3.

---

## 2. The attribution

### 2.1 It is not the residual-eikonal fit

C11 S9.5's sweep is reproduced exactly (S3.1), so the question is only which
solve degree 6 breaks.  Instrumenting every `_solve_lstsq_thread_safe` call the
D4 sphere arm makes, at degree 6, on both builds:

| call | shape | `cond(A)` | `cond(G)` | NE vs SVD coefficients | NE vs SVD residual |
|---|---|---|---|---|---|
| **residual eikonal (C10)** | 1001 x 27 | **4.084e+02** | 1.668e+05 | 2.0e-13 (Win) / 6.7e-14 (Lin) | 2.2e-16 |
| OPL / coordinate fits | 141471 x 66 | **1.394e+10** | >= 4e15 | 0.85-1.75 (Win) / 3.6-15.7 (Lin) | see S2.2 |
| ray fits | 601 x 28 | 3.024e+10 | 8.9e15 | 0.06-2.37 | see S2.2 |

`cond(A)` and `cond(G)` for the residual-eikonal fit agree to four figures on
the two builds, and its coefficients match an SVD reference to 2e-13.  **The
27-term degree-6 gradient basis is one of the best-conditioned solves in the
traced pipeline.**  The 66-term weighted ones are the worst.

The same statement, made from the other side and pinned as a test
(`test_the_degree_six_residual_fit_passes_the_screen`): the residual-eikonal
Gram's equilibrated reciprocal condition number is **1.05e-02** at degree 4, 5
and 6 -- six orders CLEAR of the screen, at every degree the cap allows.

### 2.2 What "numerically singular" costs, in the currency a fit is defined by

A least-squares fit is defined by `||b - A x||`.  On D4's three
141471 x 66 OPL/coordinate fits, against the attainable minimum:

| | Windows | Linux |
|---|---|---|
| fit 1 | 1.048x | **14.78x** |
| fit 2 | 1.427x | **22.98x** |
| fit 3 | 1.234x | **2.48x** |

Same source, same degree, same library: the residual excess is a property of
the BLAS build, which is the definition of an unstable solve.  Windows is not
"right" here -- it is lucky, by 5-40 %.

### 2.3 Why it is the weighting and not the basis

`_FIT_DISC_OUTSIDE_WEIGHT_REL = 1e-8` gives out-of-disc rows a weight
`sqrt(1e-8 n_in/n_out)` ~ 1e-4 of the in-disc ones.  The in-disc rows alone do
not determine a total-degree-10 basis over the whole launch box -- the disc is
a small sub-region of it -- so the directions they leave over are pinned only
through the 1e-4-weighted rows, and the singular values split accordingly.
This is intrinsic to the D1/D7 restriction, not a normalisation bug: the same
basis, unweighted, over the same box reads a reciprocal condition number of
**2.7e-02** (`test_the_basis_is_healthy_and_the_weighting_is_what_kills_it`).

The file already half-knew this.  `_DECENTRED_FIT_POLY_ORDER`'s own note says
"order 14 starts to LOSE to conditioning (the normal-equations Gram matrix runs
1.0e10 -> 1.9e13 across the sweep)" -- it read the trend and stopped one step
short of the conclusion that order **10** is already past float64 once the
Gram squares it.

### 2.4 The failure has a visible signature

The Linux exit field at degree 6 is not smoothly wrong, it is speckled:

```
Windows |E| along the peak row   1.6801 1.6874 1.6893 1.6895 1.6882 1.6871 1.6803
Linux   |E| along the peak row   1.6923 1.7024 1.6789 1.7141 1.6700 1.6961 1.6987
```

with `max|lap E| / peak` = **0.0353** (Windows) against **0.2999** (Linux) and
the peak pixel itself displaced by 4 pixels.  Pixel-scale scatter of a few per
cent is what a per-pixel Newton pullback does when the surface it inverts has
been fitted by a null-space draw.  After the fix Linux reads **0.0348** and
Windows **0.0348**.

---

## 3. The fix

### 3.1 What was tried, and what each route measured

All routes below replace `_solve_lstsq_thread_safe` wholesale for the whole D4
sphere arm at degree 6, measured on both builds:

| route | Windows | Linux | roughness (Lin) | solve time, 14 solves (Win / Lin) |
|---|---|---|---|---|
| normal equations (shipped, pre-C13) | 5.9277e-07 | **8.7976e-02** | 0.2999 | 0.19 / 0.17 s |
| + Gram diagonal equilibration | 7.3740e-07 | 8.8369e-07 | 0.0350 | 0.46 / 1.17 s |
| + corrected semi-normal equations, 1 step | 2.0948e-06 | **6.1421e-02** | 0.2369 | 0.31 / 0.74 s |
| + corrected semi-normal equations, 2-4 steps | 1.9871e-06 | **6.1421e-02** | 0.2616 | 0.33 / 0.60 s |
| `gelsd` (`np.linalg.lstsq`) | 6.7183e-07 | 6.7183e-07 | 0.0348 | 1.35 / 1.51 s |
| `numpy.qr` + explicit `Q` | 6.7197e-07 | 6.7191e-07 | 0.0348 | 2.95 / 3.99 s |
| **Householder QR of `[A \| b]` (SHIPPED)** | **6.7195e-07** | **6.7185e-07** | **0.0348** | 1.37 / 2.61 s |

Three things were learned and two candidates died:

* **equilibration alone fixes the symptom and is not a fix.**  It brings D4
  back under the bar, but it leaves the Gram numerically singular
  (`cond` 2.8e18 after equilibration on D4's worst solve), and the two builds
  still disagree by 20 % where the stable routes agree to 4 figures.  It is
  a cheaper coin flip, not a solve;
* **corrected semi-normal equations DIVERGE here**, as theory says they must:
  refinement through a Cholesky factor of `G` contracts only while
  `eps * cond(G) < 1`, and `cond(G)` is >= 4e15.  On a synthetic replica with
  `cond(A)` = 3e5 the same code recovers the SVD residual to 1e-11, which is
  exactly why it had to be measured on the real matrices;
* **`gelsd` works and is banned.**  B7 removed `np.linalg.lstsq` from this
  module because its divide-and-conquer SVD spawns an OpenBLAS OpenMP pool that
  deadlocks nested inside JAX's.  Householder QR is a blocked BLAS-3
  factorisation (`geqrf`) and takes no such path; it is also the cheapest of
  the stable routes when the augmented form is used, because no `Q` is formed.

### 3.2 The certificate that was built, measured, and thrown away

The obvious quality score for a single answer is the least-squares
STATIONARITY residual `A^T (b - A x)`, which is exactly zero at the solution.
It was implemented and it is **wrong on these matrices**, because it is
computed by cancellation at the `||A||^2 ||x||` scale, so its float64 value is
rounding noise:

| D4's worst OPL fit, Linux | `||A^T r|| / (||A|| ||b||)` | actual fit residual |
|---|---|---|
| the null-space draw (bad) | 7.6e-11 | 1.557e-05 |
| the QR answer (good) | 2.97e-09 | **1.053e-06** |

The score prefers the wrong answer by 39x.  Worse, the natural denominator
`||A|| (||A|| ||x|| + ||b||)` is inflated by the very thing that goes wrong --
the bad draw has `||x||` = 151 against the good one's 9.85 -- so a
stationarity gate SILENTLY PASSED the broken solve and the first cut of this
fix changed nothing at all (D4 still read `0.08797626189418387`, to the last
digit).  Recorded here because it is a plausible-looking certificate that a
future reader will reach for.

`||b - A x||` has no such problem on these fits -- the residual sits only ~3
orders below `||b||`, so it is computed to ~1e-13 relative -- which is why the
shipped decision is a comparison of the two candidates on **that**.

### 3.3 What ships

```
LSTSQ_CONDITIONING_STEPDOWN = True     # the fail-before flag
_LSTSQ_GRAM_RCOND_MIN       = 1e-8     # the screen (cost, not correctness)
_LSTSQ_RESID_MARGIN         = 1e-6     # the decision's one-sided margin
```

1. solve the normal equations exactly as before (Cholesky, then LU);
2. **screen**: `_gram_rcond(G)` -- the reciprocal 2-norm condition number of
   the DIAGONALLY EQUILIBRATED Gram.  At or above `_LSTSQ_GRAM_RCOND_MIN`,
   return the normal-equations answer **unchanged, bit for bit**;
3. otherwise re-solve by `_solve_lstsq_qr` -- Householder QR of `[A | b]`,
   whose `R` carries both the R factor of `A` and `(Q^T b)[:m]`, so one
   `geqrf` and one triangular solve suffice and no `Q` is materialised;
4. keep the QR answer only if `||b - A x_qr|| < (1 - 1e-6) ||b - A x_ne||`.
   **Ties go to the shipped path.**

The screen is for cost and the margin is for build-independence, and both have
their room measured rather than assumed:

* the **screen** may not skip a solve whose two candidates could differ by more
  than the margin.  The normal equations lose ~`cond(G) * eps`, so at the
  screen (`cond` = 1e8) a skipped solve is off by at most ~1e-8 -- a hundredfold
  inside the margin.  A build that screened one way and another the other
  therefore cannot change the answer at the boundary.  Measured on D4's 54
  solves the populations are `cond` = **1.0e2** (skipped) against
  **1.7e15 / 5.5e15 / 1.5e16 / 2.8e18** (screened in): thirteen orders of
  separation with the screen in the middle;
* the **margin** must sit far above the noise in `||b - A x||` (~1e-13
  relative, so two builds' copies of the same candidate agree to about that)
  and far below the smallest real gap.  The smallest real gap measured anywhere
  in this campaign is **4.8e-02** (D4's first OPL fit, Windows).  1e-6 is
  ~7 orders above the noise and ~5 below the signal.

### 3.4 Cost

The screen is one `eigvalsh` on the `M x M` Gram (`M` ~ 15-120) -- microseconds
against the `A^T A` the same call already forms.  On a well-conditioned
141471 x 66 fit the shipped entry point is indistinguishable from the
pre-C13 one (0.065 s Windows / 0.046 s Linux, against 0.069 s for the bare
normal equations).  A REROUTED solve at that shape costs the QR: ~1.1 s
Windows / ~1.3 s Linux on a loaded box, which is the cost of being right.

End to end the price is modest but it is NOT free, and the screen does not
skip most of the library: on `focus_scan_121.py` it screens 24 of 32 solves IN
(S6.5), so the QR runs on most of them.  The measured wall-clock effect, on a
box loaded throughout by an unrelated 5-hour job (S8.7), is the degree sweep's
per-degree pair going **9-12 s -> 12-15 s** (Windows) and **20-27 s -> 22-25 s**
(Linux) -- i.e. tens of per cent at worst and inside the run-to-run spread on
the slower build.  No full-module pre-fix timing was taken, so no "the suite
costs the same" claim is made.

---

## 4. The degree sweep, on BOTH builds

`c13_degree_sweep.py`, the D4 chain, both arms, both flag states, one process
per build.

**Before (`LSTSQ_CONDITIONING_STEPDOWN = False`) -- the fail-before:**

| degree | 2 | 3 | 4 | 5 | **6 (SHIPPED)** |
|---|---|---|---|---|---|
| sphere, Windows | 8.5241e-07 | 5.0622e-07 | 1.4493e-06 | 1.6428e-06 | 5.9277e-07 |
| sphere, **Linux** | 4.8464e-07 | 9.6711e-07 | 2.1451e-06 | 2.3631e-06 | **8.7976e-02** |
| parabola, Windows | 3.6092e-07 | 3.6092e-07 | 3.6092e-07 | 3.6092e-07 | 3.6092e-07 |
| parabola, Linux | 2.4871e-07 | 2.4871e-07 | 2.4871e-07 | 2.4871e-07 | 2.4871e-07 |
| roughness, Linux | 3.72e-02 | 3.67e-02 | 3.70e-02 | 3.56e-02 | **3.00e-01** |

The Linux row reproduces C11 S9.5's table (4.85e-07 / 9.67e-07 / 2.15e-06 /
2.36e-06 / 8.80e-02) to every digit it printed, including the 8.7976e-02.

**After (shipped):**

| degree | 2 | 3 | 4 | 5 | **6** |
|---|---|---|---|---|---|
| sphere, Windows | 6.2341e-07 | 6.2529e-07 | 7.2352e-07 | 6.5757e-07 | 6.7181e-07 |
| sphere, Linux | 6.2336e-07 | 6.2536e-07 | 7.2358e-07 | 6.5763e-07 | 6.7186e-07 |
| **build spread** | 8e-05 | 1e-05 | 8e-06 | 9e-06 | **8e-05** |
| parabola, both | 1.1421e-10 | 1.1421e-10 | 1.1421e-10 | 1.1421e-10 | 1.1421e-10 |
| roughness, both | 3.71e-02 | 3.67e-02 | 3.70e-02 | 3.53e-02 | 3.48e-02 |

Three separate statements, all of them what a sound computation looks like:
**the cliff is gone**, the sweep is smooth 2..6 on both builds, and the two
builds now agree to 1e-5 at every degree instead of disagreeing by 1e5 at one
of them.

### 4.0.1 The parabola arm is the sharpest evidence, and it costs a docstring

The parabola arm never failed, so it was never suspected.  It improves by
**2,200x** (Windows) / **2,180x** (Linux) and becomes build-identical to five
figures.  That matters because of what D4's own docstring says that arm
MEASURES: *"it transports the leg in two pieces instead of one (measured 3.1e-8
under `carrier_reference='parabola'`, where that is the only difference)"*.

If the two-piece transport were really worth 3.1e-8, no change to a linear
solver could take it to 1.14e-10.  **It was not the transport.**  The recorded
3.1e-8 -- and the 3.6e-07 the same arm reads on this tree with the flag down --
was the solver's null-space draw, and the two-piece transport term is
1.14e-10, i.e. essentially exact.  A physical effect this campaign had
attributed, quantified, and written into a test's rationale was a numerical
artefact of the fit.  The docstring is corrected in this change (S7.3).

This is also the answer to the obvious objection -- *"a lower residual is not
the same as a better answer"*.  Four independent things move the right way at
once: two independent ROUTES to the same field agree (D4's whole point), on
two builds; the upstream leg goes to machine precision (S1.3); design 121's
per-order table against an EXACT-RAY oracle is reproduced (S6.2); and
conservation and the halo self-check do not move (S6.4).  A fit chosen to
minimise the wrong thing does not do that.

### 4.1 The exit fields, with percentiles

`abs(dE)` normalised by the reference peak, D4's own arms, degree 6:

| comparison | before: max / p99.9 / p99 / rms | after: max / p99.9 / p99 / rms |
|---|---|---|
| **same arm, two builds** (`doe`) | 6.79e-02 / 5.37e-02 / 4.43e-02 / 2.06e-02 | **1.21e-10** / 1.02e-10 / 8.25e-11 / 3.77e-11 |
| same arm, two builds (`manual`) | 7.79e-02 / 5.37e-02 / 4.10e-02 / 1.85e-02 | 1.17e-10 / 1.07e-10 / 9.48e-11 / 4.33e-11 |
| upstream (chain A), two builds | 1.38e-05 / 1.25e-05 / 7.69e-06 / 1.31e-06 | **4.08e-16** / 2.24e-16 / 1.61e-16 / 6.29e-17 |
| the D4 assertion, Windows | 5.93e-07 / 4.92e-07 / 3.76e-07 / 1.43e-07 | 6.72e-07 / 5.97e-07 / 3.95e-07 / 1.45e-07 |
| the D4 assertion, **Linux** | **8.80e-02** / 7.25e-02 / 5.81e-02 / 2.69e-02 | **6.72e-07** / 5.97e-07 / 3.95e-07 / 1.45e-07 |

Every conclusion survives at p99.9, so none of this is one pixel.

---

## 5. What the fix does NOT change

### 5.1 The null contract, bitwise

A fit whose Gram clears the screen returns **the same float64 words** it
returned before -- not "agrees to 1e-12".  Pinned two ways:

* `test_the_healthy_fit_is_bitwise_unchanged` -- an unweighted order-8
  Chebyshev fit over the whole box, `np.array_equal` against an independently
  computed `cho_solve` answer;
* `test_a_scaled_healthy_fit_still_returns_the_shipped_bits` -- the same after
  scaling the columns over twelve orders, so the null cannot be lost to units.

### 5.2 Ties keep the shipped path

`test_a_tie_keeps_the_shipped_path` hands the step-down a candidate that is not
better and asserts the normal-equations bits come back.  Without that, every
screened solve would reroute on noise and two builds could disagree about
which.

### 5.4 What it DOES change, unexpectedly: niche D1's defect is cured twice

The two-failure surprise of this study.  `test_niche_d1_tilted_carrier` carries
two FAIL-BEFORE witnesses that rebuild the pre-D1 library -- hard NaN sample
mask (`_FIT_DISC_OUTSIDE_WEIGHT_REL = 0`) plus the pre-C6 ray launch -- and
assert that the exit field then ghosts and that the ray-density fold detector
fires.  Both went RED on the shipped tree, and neither is a bar that drifted:

| witness | bar | pre-C13 | with C13's step-down |
|---|---|---|---|
| ghost power vs the spline oracle's true halo | `> 50x` | >1000x | **1.8x** (2.29e-07 against 1.30e-07) |
| fold-caustic warnings on the broken arm | `>= 1` | fires | **silent** |

**The fold does not happen any more.**  D1's cure and C13's cure are
independent solutions to the same defect: D1 removed the near-singularity from
the DESIGN MATRIX (by weighting the out-of-disc samples instead of masking
them), and C13 removed the instability from the SOLVE.  Either one alone
suppresses the ghost, so with C13 in force the pre-D1 configuration is no
longer reproducible by turning D1 off.

That is a claim about the mechanism, and it is the same mechanism this whole
document is about: the historical hard-mask fit was ill-conditioned, the normal
equations answered it with a null-space draw, and the draw is what folded the
inverse map and manufactured the lobe.  `_FIT_DISC_OUTSIDE_WEIGHT_REL`'s own
note already said as much -- *"the directions the disc leaves FREE are pinned
to the traced map instead of to fit noise"* -- it just did not connect "fit
noise" to the solver.

**Handled the C9/C10 way**: both witnesses now era-pin
`LSTSQ_CONDITIONING_STEPDOWN = False` alongside the selector flags, with their
assertions unchanged word for word.  A fail-before arm that inherits a default
is not a fail-before, and this study moved two defaults.

### 5.3 B7 survives

`test_it_never_reaches_gelsd_on_a_full_rank_matrix` detonates
`np.linalg.lstsq` and runs both `_solve_lstsq_qr` and the full shipped entry
point on the case that reroutes.  `lstsq` remains only as the last-resort
branch for an `A` whose `R` comes out exactly singular -- a path the traced
fits do not reach.

---

## 6. Design 121

### 6.1 The fail-before is bit-exact on the design, 6 of 6

`rc_resdeg_121.py` unedited, through `c13_with_stepdown.py`, `RN=1024`,
`rs=4`, `DEGS=4,6`, all six orders, with `STEPDOWN=0`.  The degree-6 intensity
hashes:

```
4e9effd4 / b2a8b150 / 88f726eb / cf0bc1f3 / 7845b7a8 / c4b850ea
```

-- **exactly** the six `C11` S6.6 (b) records, and the degree-4 line reproduces
`D121_RESIDUAL_CLOSURE` S3's `8db002a1` (on axis) and `5e855046` (at (-4,-2)).
The flag's OFF state is the pre-C13 library, on the design, to the bit.

### 6.2 The per-order table, on BOTH builds

C10's degree-6 closure had only ever been measured on Windows.  Here it is on
both, with the step-down ON.  EE3 area-exact against the CARRY=1 exact-ray
ceiling; the number is the residual left, in points (smaller is better).

| order | ceiling | deg 4 Win | deg 6 Win | deg 4 **Lin** | deg 6 **Lin** | C10 record (deg 4 / 6) |
|---|---|---|---|---|---|---|
| (0,0)   | 90.5324 | 0.0482 | -0.0477 | 0.0482 | -0.0477 | 0.048 / -0.048 |
| (-1,0)  | 90.5768 | 0.9344 | 0.0290 | 0.9344 | 0.0290 | 0.934 / 0.029 |
| (-2,0)  | 90.6650 | 0.7740 | 0.0634 | 0.7740 | 0.0634 | 0.774 / 0.063 |
| (-3,0)  | 90.6961 | 0.5274 | 0.0898 | 0.5274 | 0.0898 | 0.527 / 0.090 |
| (-4,0)  | 90.5035 | 0.3054 | 0.1410 | 0.3054 | 0.1410 | 0.305 / 0.141 |
| (-4,-2) | 90.1071 | 0.2786 | 0.1517 | 0.2786 | 0.1517 | 0.279 / 0.152 |

Every Linux cell equals its Windows cell **to all four decimals**, and both
equal `D121_RESIDUAL_CLOSURE` S6.2's recorded line to the three decimals it
printed.  C10's degree-6 closure -- the residual going from
`0.048 / 0.934 / 0.774 / 0.527 / 0.305 / 0.279` to
`-0.048 / 0.029 / 0.063 / 0.090 / 0.141 / 0.152`, every order inside
+-0.16 points of the exact-ray oracle -- **holds on OpenBLAS**, which had never
been measured.  The one cell that moves at all is (-4,0) at degree 6, and only
between the OFF arms: 0.1410 (Windows) against 0.1409 (Linux), i.e. 1e-4.

The per-order gains are therefore C10's, unchanged, on both builds and in both
step-down states -- see S6.6 for why that is the honest scope of the blocker.

### 6.3 Production acceptance -- unchanged to every printed digit

`focus_scan_121.py` **unedited**, pure library defaults (`CREF`/`AM`/`PIP`
unset), N=2048, `rs=4`, NFC=8192, WF=4.0, NOUT=2048, run twice through
`c13_with_stepdown.py` with the flag pinned either way so neither arm depends
on what the default happens to be:

| | **step-down ON (shipped)** | **step-down OFF (fail-before)** |
|---|---|---|
| `AT-PLANE` | 3.350 um / 90.3 / 99.7 / 99.8 | 3.350 um / 90.3 / 99.7 / 99.8 |
| `BEST-FOCUS[peak]` plane | dz = **+0 um** | dz = **+0 um** |
| FWHM / EE3 / EE6 / EE12 | 3.350 um / 90.3 / 99.7 / 99.8 | 3.350 um / 90.3 / 99.7 / 99.8 |
| peak | **5.529e+03** | **5.529e+03** |
| `dz = +5 um` | 3.450 um / 89.6 / 99.7 / 99.8 | 3.450 um / 89.6 / 99.7 / 99.8 |

**The recorded acceptance line is unchanged**, peak included -- the same line
`C11` S6.6 (a) and `D121_RESIDUAL_CLOSURE` S6.3 (a) record, so this is the
campaign's own measurement rather than a new one.

**It is unchanged METRICS, not untouched arithmetic**, and the difference is
measured rather than assumed: on this very runner the step-down reroutes 6 of
32 solves, some missing the attainable fit residual by 1072x (S6.5).  The
readout re-traces the final leg on a fine grid (`n_fine_cap` 12288) where those
coarse coordinate fits are not what limits the spot, which is why four
significant figures and the peak all survive.  Anyone who needs bit-identity on
the design has the flag.

### 6.4 Conservation and halo, 6 of 6, and the null is exactly zero

`energy_stage_audit_121.py` **unedited**, through `c13_with_stepdown.py`,
`RN=1024`, `rs=4`, six post-DOE groups, `final_leg='paraxial'`, `CONFIGS=ship`,
`NULL=1`, step-down ON.  Against `D121_RESIDUAL_CLOSURE` S5.3 / S6.3 (b)'s
recorded degree-6 numbers (the "record" columns, as quoted by `C11` S6.6 (c)):

| order | `P_out/P_in` C13 | (record) | `g4` C13 | (record) | `amax4` C13 | (record) | `r_rms` mm | NULL |
|---|---|---|---|---|---|---|---|---|
| (0,0)   | 0.994315 | 0.994315 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.8383 | 0.000e+00 |
| (-1,0)  | 0.994065 | 0.994065 | 2.663e-11 | 2.663e-11 | 1.309e-05 | 1.309e-05 | 0.8385 | 0.000e+00 |
| (-2,0)  | 0.994133 | 0.994133 | 7.659e-11 | 7.659e-11 | 3.326e-05 | 3.326e-05 | 0.8383 | 0.000e+00 |
| (-3,0)  | 0.994065 | -- | 1.567e-09 | -- | 5.824e-05 | -- | 0.8380 | 0.000e+00 |
| (-4,0)  | 0.994008 | -- | 9.778e-09 | -- | 1.058e-04 | -- | 0.8377 | 0.000e+00 |
| (-4,-2) | 0.993843 | 0.993843 | 9.694e-09 | 9.694e-09 | 1.117e-04 | 1.117e-04 | 0.8376 | 0.000e+00 |

Where the record has a number, C13 reproduces it **to every digit recorded**,
including `g4` = 2.663e-11 and `amax4` = 1.309e-05.  (C11 quoted only four
orders; the two dashes are orders it did not print, measured here for
completeness.)  `P_out/P_in` stays inside C2's [0.9850, 1.00050] on every
order, every `g4` is 1e-3 or less of its C3 bound, `amax4` stays 10x under the
C4 bound of 1.0e-03, and `r_rms` moves by <= 0.0002 mm against a C5 tolerance
of 0.030.

**The null column is the floor every one of those numbers is read against**:
two identical shipped runs, same process, all six stages, `array_equal = True`
and `max abs dE` = **0.000e+00**, on all six orders.  A chain that is bitwise
reproducible in-process is the precondition for reading anything off a 1e-5
delta at all, and it is asserted here rather than assumed.

**C7 and the element energy self-check stay silent.**
`grep -c "HALO self-check FAILED"` and `grep -c "energy self-check FAILED"`
both read **0** across the energy audit, all five `rc_resdeg_121.py` runs, both
production focus scans, the solver census and every suite in S7.

### 6.5 Where the step-down actually fires on the design

`c13_solver_census.py` wraps `_solve_lstsq_thread_safe` script-side and records
every call's equilibrated Gram condition number and, for the calls that screen
in, both candidates' fit residuals.

**On `focus_scan_121.py` (N=2048): 32 solves, 24 screen in, 6 REROUTE.**  The
rerouted ones are the 28-term coordinate fits (36541 x 28, 37673 x 28,
39381 x 28), and their normal-equations answers miss the attainable fit
residual by ratios running **1.0000 to 1072x**, median **145x**.  The Gram
condition numbers that runner produces span `rcond` 9.4e-11 (worst) to 9.9e-03
(best), median 1.6e-09 -- the traced pipeline lives on BOTH sides of the
screen, which is why there is a screen at all rather than "always QR".

This measurement replaced an assumption.  The first draft of
`_LSTSQ_GRAM_RCOND_MIN`'s note asserted that the concentric production route
was entirely on the skip side "which is why that acceptance is unchanged to
the bit".  It is not, and the note now says so.

### 6.6 Was design 121 ever broken by this?  No -- and that is worth stating

`rc_resdeg_121.py` with `STEPDOWN=0` on **Linux/OpenBLAS**, the build the
blocker fails on, all six orders (the campaign's first design-121 table taken
on that build at all -- S7.2):

| order | deg 4 | deg 6 | Windows/OFF deg 6 |
|---|---|---|---|
| (0,0)   | 0.0482 | -0.0477 | -0.0477 |
| (-1,0)  | 0.9344 | 0.0290 | 0.0290 |
| (-2,0)  | 0.7740 | 0.0634 | 0.0634 |
| (-3,0)  | 0.5274 | 0.0898 | 0.0898 |
| (-4,0)  | 0.3054 | **0.1409** | 0.1410 |
| (-4,-2) | 0.2786 | 0.1517 | 0.1517 |

**The design's own per-order table was already build-independent before the
fix**, to 1e-4 in the worst cell.  So the correct scope of the release blocker
is: an ill-conditioned solve that a SYNTHETIC two-route equivalence test caught
at 8.8e-02 and that design 121's EE metrics were insensitive to.  That does not
make it less of a blocker -- an unstable solve under every traced fit in the
library is a defect wherever it happens to be visible -- but the claim "design
121 was computing the wrong answer" is NOT made, because it was measured and it
is false.

---

## 7. What this change touches

### 7.1 The diff

| file | what |
|---|---|
| `lumenairy/elements/_lens_traced.py` | `LSTSQ_CONDITIONING_STEPDOWN`, `_LSTSQ_GRAM_RCOND_MIN`, `_LSTSQ_RESID_MARGIN`, `_gram_rcond`, `_solve_lstsq_qr`, `_lstsq_residual`, the step-down in `_solve_lstsq_thread_safe` -- and ONE selector default, `DECENTRED_FIT_ARBITER` (S10); `DECENTRED_FIT_PREDICTOR` is back at `False` with its note rewritten (S11) |
| `tests/unit/test_niche_c13_lstsq_conditioning.py` | 20 tests, ~1 s, no proprietary asset |
| `tests/unit/test_niche_c12_physics_fit_selection.py` | the default pin + two era-pinned arms (S10.2) + `_capture_traced_opl` era-pins both selector flags (S11.7) |
| `tests/unit/test_niche_c11_decentred_fit_arbiter.py` | the flip's default pin, era-pinned arm, and the module's `_apply` now pins BOTH flags (S10.2) |
| `tests/unit/test_niche_c1_consolidation.py`, `tests/unit/test_niche_c6_fit_guard.py` | era pins extended to both selector flags (S10.2) |
| `tests/unit/test_niche_d1_tilted_carrier.py` | TWO fail-before witnesses era-pin the selector AND the step-down; assertions unchanged (S5.4) |
| `tests/unit/test_niche_d3_guards.py` | ONE arm era-pins the step-down, assertions unchanged, PLUS a sibling asserting the new build-independence at 100x (S11.8) |
| `tests/unit/test_niche_d7_decentred_fit.py` | TWO fail-before witnesses era-pin the step-down, assertions unchanged, PLUS a new sibling asserting the cure (S11.5) |
| `tests/unit/test_niche_d4_dgrating.py` | ONE docstring corrected (S7.3); no assertion, fixture or bar moved |
| `validation/repro_traced_carrier_121/c13_with_stepdown.py` | pin the step-down and the selector through any runner |
| `validation/repro_traced_carrier_121/c13_solver_census.py` | which solves screen in, and by how much |
| `validation/repro_traced_carrier_121/_d121_common.py` | ONE literal: `D121_ROOT` (S7.2) |
| `docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md` | this document |

`lumenairy/propagators/carrier.py` is **unmodified BY THIS STUDY** -- it
read `5a1b0d1021969df1` (the C9/C11/C12 hash) throughout every
measurement in this document.  It has SINCE been changed by a
CONCURRENT workstream (niche C14, `ARCH_TRACED_ENCAPSULATION`), which
also added `lumenairy/elements/_traced_flags.py`.  Neither is C13's and
neither is in C13's inventory (S7.1).

That registry and this study AGREE, checked mechanically rather than by
reading: its `v5.32.1` column matches the live module defaults on all six
entries it declares -- `_REMAP_RESID_EIKONAL_DEGREE` 6,
`DECENTRED_FIT_ARBITER` True, **`DECENTRED_FIT_PREDICTOR` False**,
`REMAP_STATIONARY_PHASE_FIT_GUARD` False, `LSTSQ_CONDITIONING_STEPDOWN`
True, `SPHERE_PARAB_CONVERSION_EXACT` True -- including S11's reversion,
which it had already picked up.

`CHANGELOG.md` and `lumenairy/elements/pmm/**` are untouched.

The three new module-level constants are `bool`/`float` attributes of
`lumenairy.elements._lens_traced`, which is already in the C11 leak guard's
`_LEAK_GUARD_MODULES`, so a test that leaves any of them dirty now fails and
names itself.  That was free.

### 7.2 The design's runners now drive from the CI proxy

`_d121_common.py` hard-coded the Windows dev-box root, which is why every
design-121 measurement this campaign has taken -- including the degree-6
per-order table C13 exists to re-check -- had been taken on ONE BLAS build.
One `os.environ.get('D121_ROOT', <the old literal>)` removes that, and S6.2 /
S6.6 / S10.3 are the campaign's first design-121 tables measured on both.  The
default is the old literal, so nothing changes for a Windows run.

This is the third finding in two documents (with `C11` S9.2 and S9.5) where
"the evidence base is one build" was the actual defect, so the instrument is
worth more than the tables it produced.

### 7.3 The stale calibration this change retires

`test_niche_d4_dgrating::test_matches_the_manual_hand_split`'s docstring
attributed 3.1e-8 to the two-piece transport and 5.3e-7 to the sphere/parabola
conversion.  Post-C13 those read **1.14e-10** and **6.72e-07**, on both builds
(S4.0.1).  The docstring is corrected in place; no assertion, bar or fixture
moves.

### 7.4 Suites

All in the SHIPPED 5.32.1 configuration -- step-down ON, `DECENTRED_FIT_ARBITER`
ON, `DECENTRED_FIT_PREDICTOR` OFF (S11) -- unless a row says otherwise.

```
tests/unit/test_niche_{c1,c3,c5,c6,c7,c8,c9,c10,c11,c12,c13,
                       d1,d2,d3,d4,d5,d6,d7,s8}_*.py      THE GATE
  Windows  py3.14.6 numpy 2.4.4   ->  501 passed, 72 warnings in 2205.63s (36:45)
  Linux    py3.12.3 numpy 2.4.6   ->  501 passed, 72 warnings in 2516.92s (41:56)

tests/unit/test_niche_c13_lstsq_conditioning.py            (new, 20 tests)
  Windows  ->  20 passed in 0.93 s        Linux  ->  20 passed

tests/unit/test_niche_d4_dgrating.py                       (THE BLOCKER)
  Windows, SHIPPED   ->  59 passed (312 s)
  Linux,   SHIPPED, RUN ALONE, -k matches_the_manual_hand_split
                     ->  1 passed (563 s)
  Windows,           RUN ALONE, same -k  ->  1 passed
  (fail-before, Linux, step-down off:  sphere-reference arm 8.7976e-02 >= 1e-4)

tests/unit/test_niche_{d6,d7}_*.py       (the S11 adjudication's subjects)
  Windows  ->  75 passed in 314 s        (0 failed; 9 of the 11 were the
                                          predictor, 2 were C13-cured)

tests/unit/test_niche_d3_guards.py       (S11.8)
  Windows  ->  40 passed in 298 s        Linux  ->  40 passed in 240 s

ruff check lumenairy/ tests/unit/        (the CI invocation, CI's own ruff)
  Linux    ->  All checks passed
  (the Windows box has no ruff -- the WSL proxy IS the lint check here, per the
   standing note; `validation/` is outside CI's scope and the two runners added
   by this study match that directory's existing import convention)
```

**Getting to that gate took four rounds, and each round was a finding**, which
is the honest summary of S11: `11 failed` (the selector, S11.1-11.6), then
`2 failed` (C12's own positional capture index, S11.7), then `1 failed` on
Linux only (D3's pre-C13 bar, S11.8), then green on both. Every one was
adjudicated against the flags before anything was touched, and not one
assertion was weakened.

**A flip-specific hazard, checked rather than assumed.** `DECENTRED_FIT_ARBITER`
shipping `True` does not make the C12 disagreement `RuntimeWarning` fire --
that warning belongs to the predictor, which stayed off. `pyproject.toml`'s
`filterwarnings` promotes only `PytestReturnNotNoneWarning` and `SyntaxWarning`
to errors in any case.  Measured on the gate rather than argued: **72 warnings
on each build, the same 72**, and a census of the summary finds 24
`UserWarning` and 18 `RuntimeWarning` sites, every one of them the pre-existing
C5 aperture-exceeds-grid / exit-Nyquist family.  Grepping both logs for
`disagree|predictor|arbiter|conditioning|stepdown|lstsq` returns **0 lines**.
Neither the flip nor the step-down added a warning.

`validation/run_all.py` is unaffected for the reason `C11` S8.4 records: no
file it collects passes `beam_centre`, `fit_radius_beam_factor` or a
`TiltedCarrier`, so the decentred branch is never reached there in any flag
state.

---

## 8. What remains unresolved

1. **The weighted design matrix is still ill-conditioned.**  C13 makes the
   SOLVE stable; it does not make `cond(A)` = 1.4e10 go away.  The conditioning
   comes from `_FIT_DISC_OUTSIDE_WEIGHT_REL` = 1e-8 asking a total-degree-10
   basis over the whole launch box to be determined by a small off-centre disc,
   and the principled cure is to fit in a basis ORTHOGONAL ON THE WEIGHT (a
   disc-centred, disc-scaled one) rather than to solve a near-singular system
   carefully.  That is a bigger change than a release blocker should carry, and
   it is the natural successor.  The amplification it would remove is visible:
   the two builds agree to 1.2e-10 after the fix and to 4e-16 on the leg that
   does not take this branch, so ~6 orders of the launch's build-sensitivity
   are still conditioning rather than arithmetic.

   Note the counter-argument already in the file: `_DECENTRED_FIT_POLY_ORDER`'s
   own note rejected re-mapping the basis domain onto the off-centre disc,
   measured, because the Newton loop evaluates the fit over the WHOLE launch
   square where a re-mapped basis runs to `max|T_k|` = 5.7e8.  A weight-
   orthogonal basis has to answer that, and this study did not.

2. **`_DECENTRED_FIT_POLY_ORDER` = 10 was chosen on a sweep this document
   partly invalidates.**  That note reads 6/8/10/12/14 as "14 starts to LOSE to
   conditioning".  With the solve fixed, the sweep should be re-run: 12 and 14
   may now be usable and 10 may be under-fitting.  NOT DONE -- it moves a
   shipped constant on evidence this study did not take.

3. **The sibling solver is not covered.**  `lenses_maslov._solve_fit` is cited
   in `_solve_lstsq_thread_safe`'s own docstring as the precedent for the
   normal-equations choice ("measured 2.6e-15").  That measurement was taken on
   ITS matrices and nothing here says whether they are the concentric kind or
   the weighted kind.  Out of scope (this study owns `_lens_traced.py`); it
   should be checked.

4. **The energy audit did not exercise the step-down**, so its 6-of-6 agreement
   with the record is a statement that C13 is INERT on that instrument, not
   that C13 was scored by it.  The instruments that DID exercise it are
   `rc_resdeg_121.py`, `focus_scan_121.py` (S6.5) and D4 itself.

5. **CI has not run any of this.**  Both builds are local; the Linux one is the
   `~/lumvenv` CI proxy, not a runner, and the C11 campaign twice found a third
   build behaving like neither.  What makes this one different is that the fix
   REMOVES the build dependence rather than re-pinning a bar around it -- but
   that is an argument, not a CI leg.

6. **The 5.32.1 selector has a reachable discontinuity at residual degree 5**
   (and note S11 reverted the PREDICTOR half of that selector; the degree-5
   reading below was taken with `SELECTOR=arbiter` as well, so it is the
   ARBITER's discontinuity and it survives the reversion)
   (S10.6).  On D4's synthetic fixture at 0.62 w the crossover moves above the
   decentre at degree 5 only, the call routes to the CONCENTRIC fit, and the
   two-route comparison reads 6.4e-04 instead of ~6.7e-07.  It is C11's choice
   rather than C12's prediction (identical under `SELECTOR=arbiter`), both
   routes take the SAME branch so it is not a flip between them, and the
   SHIPPED degree 6 does not reach it.  NOT fixed here: it is a property of the
   selector the flip made reachable, not of the solve C13 repaired, and the
   right instrument for it is a decentre sweep of the two-route residual rather
   than a degree sweep of one fixture.

7. **A 74 GB orphan `fan_multi_121.py` was running on the box throughout**
   (pid 46684, ~5 h old, 78 % of a core, 14 GB of swap in use).  It was not
   killed -- it is live work belonging to someone else -- but every wall-clock
   cost in this document was measured against it, so the ABSOLUTE timings are
   pessimistic and only the ratios should be quoted.

---

## 9. Reproduction

All commands from the repo root unless stated.

```bash
# THE BLOCKER, both builds, run ALONE
python -m pytest tests/unit/test_niche_d4_dgrating.py -q -p no:randomly \
    -k matches_the_manual_hand_split
wsl -e bash -lc "cd <repo> && ~/lumvenv/bin/python -m pytest \
    tests/unit/test_niche_d4_dgrating.py -q -p no:randomly \
    -k matches_the_manual_hand_split"

# S4 -- the degree sweep, both flag states, both builds
python <scratch>/c13_degree_sweep.py both

# S3.1 -- the route comparison (ne | equilibration | csne | gelsd | qr)
python <scratch>/c13_diag5.py <route> 6

# S6.1 / S6.2 / S6.6 / S10.3 -- the design's per-order table.
# STEPDOWN=0 is C13's fail-before; SELECTOR=gate is the FLIP's fail-before.
cd validation/repro_traced_carrier_121
STEPDOWN=1 SELECTOR=predictor ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' \
    DEGS='4,6' python c13_with_stepdown.py rc_resdeg_121.py
#   ... and the same on the CI proxy, which is new:
wsl -e bash -lc "cd <repo>/validation/repro_traced_carrier_121 && \
    D121_ROOT=/mnt/d/.../Free_Space_Optics STEPDOWN=1 SELECTOR=predictor \
    ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' DEGS='4,6' \
    ~/lumvenv/bin/python c13_with_stepdown.py rc_resdeg_121.py"

# S6.3 / S10.4 -- production acceptance
STEPDOWN=1 SELECTOR=predictor python c13_with_stepdown.py focus_scan_121.py
STEPDOWN=0 python c13_with_stepdown.py focus_scan_121.py

# S6.4 -- conservation + halo + the null, all six orders
STEPDOWN=1 ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CONFIGS='ship' NULL=1 \
    python c13_with_stepdown.py energy_stage_audit_121.py

# S6.5 -- which solves screen in, on any runner
python c13_solver_census.py focus_scan_121.py

# S11.1 -- WHICH default each red arm was reporting.  Every rung pinned
# explicitly; this is what separates an era pin from a regression.
python <scratch>/c13_adjudicate2.py <nodeid> [...]      # pred+arb | arb | gate
python <scratch>/c13_adjudicate.py  <nodeid> [...]      # + the step-down axis

# S11.2 -- the analytic-oracle cost of the predictor, per ladder rung
python <scratch>/c13_d6_probe.py <pred:0|1> <arb:0|1>

# S11.5 -- the C13 cure on D7's hard-mask fixture, step-down off vs on
python <scratch>/c13_d7ghost_probe.py <stepdown:0|1>

# S11.4 -- did the predictor ever deviate on design 121?  (expect 0)
grep -c 'PREDICTOR and the niche-C11 ARBITER disagree' \
    validation/repro_traced_carrier_121/_c13_*_flip.txt

# S7.1 -- the C14 registry agrees with the live defaults, checked not read
python -c "import importlib, lumenairy.elements._traced_flags as TF; \
 [print(n, e['v5.32.1'], getattr(importlib.import_module(m), n)) \
  for (m, n), f in TF.FLAGS.items() if (e := f.eras) and 'v5.32.1' in e]"

# S7.4 -- the suites, both builds
python -m pytest tests/unit/test_niche_c13_lstsq_conditioning.py -q
python -m pytest tests/unit/test_niche_{c1,c3,c5,c6,c7,c8,c9,c10,c11,c12,d1,\
    d2,d3,d4,d5,d6,d7,s8}_*.py -q
ruff check lumenairy/ tests/unit/
```

### Files added by this study

```
tests/unit/test_niche_c13_lstsq_conditioning.py             20 tests
validation/repro_traced_carrier_121/c13_with_stepdown.py    pin step-down + selector
validation/repro_traced_carrier_121/c13_solver_census.py    where the step-down fires
docs/audits/C13_DEGREE6_CONDITIONING_2026_08_03.md          this document
```

### Files MODIFIED by this study

```
lumenairy/elements/_lens_traced.py           the fix + DECENTRED_FIT_ARBITER = True
                                             (and DECENTRED_FIT_PREDICTOR's note)
tests/unit/test_niche_c13_lstsq_conditioning.py   (new, above)
tests/unit/test_niche_c11_decentred_fit_arbiter.py  default pin + era pin + helper
tests/unit/test_niche_c12_physics_fit_selection.py  default pin + era pin + invariant
tests/unit/test_niche_c1_consolidation.py    era pin extended to both flags
tests/unit/test_niche_c6_fit_guard.py        era pin extended to both flags
tests/unit/test_niche_d1_tilted_carrier.py   2 witnesses era-pin selector + step-down
tests/unit/test_niche_d7_decentred_fit.py    2 witnesses era-pin the step-down,
                                             + test_c13_cures_the_hard_mask_fold...
tests/unit/test_niche_d4_dgrating.py         ONE docstring (S7.3)
validation/repro_traced_carrier_121/_d121_common.py   D121_ROOT (S7.2)
docs/audits/C11_PHYSICAL_DECENTRE_GATE_2026_08_03.md  S9.5 resolution pointer
```

**`test_niche_d6_exact_tilted_leg.py` is NOT in either list, deliberately.**
Its four red arms were the finding, not the debt: they are green again because
the predictor went back to `False` (S11), and not one line of that file moved.
A study that had pinned them would have shipped a 32 % EE loss with four
comments explaining why the tests no longer noticed.

### Scratch probes (not committed; reproduced in S9)

```
c13_diag2/4/5.py         conditioning + route comparison on the D4 chain
c13_degree_sweep.py      the degree sweep, both flag states
c13_routes.py            solve-route bench at the traced fits' worst shape
c13_adjudicate.py        PASS/FAIL vs the step-down AND selector axes
c13_adjudicate2.py       PASS/FAIL vs the three selector-ladder rungs (S11.1)
c13_d6_probe.py          D6's EE2/oracle per ladder rung, with u / u* / scores
c13_d7ghost_probe.py     D7's hard-mask ghost, step-down off vs on (S11.5)
```

---

## 10. The 5.32.1 selector flip (scope addition, user-ordered)

Half-way through this study the ray-fit branch selector was ordered ON for
5.32.1.  It lands here rather than in its own document because it runs through
the machinery C13 had just made stable, and because the standing rule is now
**no build-singular evidence** -- so it has to be re-measured on both builds,
which is this document's whole apparatus.

### 10.1 What was flipped, and why BOTH flags

```
DECENTRED_FIT_PREDICTOR = True      # was False (niche C12)
DECENTRED_FIT_ARBITER   = True      # was False (niche C11)
```

The C12 architecture is *predictor decides, arbiter cross-checks*, and the
cross-check is engaged INTERNALLY: the fit site enters the selector block on
`ARBITER or PREDICTOR`, computes the arbiter's measured verdict
unconditionally, and only then lets the predictor overwrite it.  So
`DECENTRED_FIT_ARBITER` is **not needed as a fallback path** while the
predictor is on -- `(True, True)` and `(True, False)` are bitwise the same
call, asserted by
`test_niche_c12::test_the_arbiter_flag_is_a_no_op_while_the_predictor_decides`.

**It ships `True` anyway, and the reason is the fall-back ladder.**  The
`RuntimeWarning` the predictor already raises on a disagreement tells the
reader *"set it False to fall back to the arbiter, or also
DECENTRED_FIT_ARBITER False for the v5.32 gate"*.  That sentence is only TRUE
if the arbiter is on underneath.  Shipping the predictor alone would have left
the library emitting an instruction that does not work.  The ladder as shipped:

```
PREDICTOR True                 ->  u <= u*      (niche C12)
PREDICTOR False, ARBITER True  ->  E_c <= E_o   (niche C11)
both False                     ->  the v5.32 gate, bit for bit
```

**C13 is a precondition for this flip, not merely adjacent to it.**  On design
121 the launch-box spectrum is UNRESOLVED (C12 S3.4), so the predictor falls
back to the MEASURED pair -- i.e. what this flag actually selects on the design
is C11's comparison of two least-squares fit residuals.  Those are the very
solves that were being answered by a null-space draw.  Flipped on before C13,
the selector would have been arbitrating between two numbers one of which was
a coin toss on 6 of 32 production solves (S6.5).

### 10.2 The tests that pinned the old default

Four files pinned "ships off", and each is re-pointed the C9/C10 way -- the
NEW default asserted in one place, the old assertions kept **verbatim** with
the flag state now set explicitly instead of inherited.  The table below is
the FINAL state, after S11 sent the predictor back to `False`:

| file | was | now |
|---|---|---|
| `test_niche_c12_physics_fit_selection.py` | `test_the_predictor_ships_off_and_is_never_even_computed` | `test_the_predictor_stays_off_and_the_arbiter_ships_on` (pins BOTH defaults and says why they differ) **and** `test_the_predictor_off_is_a_path_not_taken_and_never_even_computed` (era-pinned, body verbatim) **and** `test_the_arbiter_flag_is_a_no_op_while_the_predictor_decides` (new; now an invariant of the opt-in arm rather than of the shipped pair) |
| `test_niche_c11_decentred_fit_arbiter.py` | `test_the_arbiter_ships_off_as_an_opt_in` | `test_the_arbiter_ships_on_since_5_32_1` **and** `test_the_arbiter_off_is_a_path_not_taken` (era-pinned, body verbatim) |
| `test_niche_c11_...` `_apply` helper | pinned `ARBITER` only | pins `PREDICTOR = False` for the whole module -- **this file is the C11 era**, and with the predictor on every arm would have been measuring C12, including the `arbiter=False` "fail-before" arms, which the predictor enters on its own |
| `test_niche_c1_consolidation.py`, `test_niche_c6_fit_guard.py` | era-pinned `ARBITER = False` | era-pin BOTH flags, same reason |

That third row is the one that would have shipped a silent hole: five C11 tests
went red on the flip (`5 failed, 56 passed`) precisely because an
`ARBITER = False` arm is no longer a fail-before once `PREDICTOR` enters the
same block.  A file that pins one flag of a two-flag ladder is pinning nothing.

**The pins survived the reversion, and that is the point of pinning
explicitly.**  Every arm above states its flag state rather than inheriting it,
so sending `DECENTRED_FIT_PREDICTOR` back to `False` in S11 changed exactly one
line of test code -- the default assertion in the first row -- and no arm's
measurement.  The 9 tests that DID move (S11.1) are all in files that had no
pin at all.


### 10.3 The per-order table, SHIPPED configuration, both builds

`rc_resdeg_121.py` through `c13_with_stepdown.py` with `SELECTOR=predictor`,
`STEPDOWN=1`, `RN=1024`, `rs=4`, `DEGS=4,6`, all six orders.  Residual left
against the CARRY=1 exact-ray ceiling, in points; degree 6 is the shipped
residual-eikonal degree.

| order | ceiling | **flip Win** | **flip Lin** | v5.32 selector | delta | C11 record |
|---|---|---|---|---|---|---|
| (0,0) | 90.5324 | **-0.0477** | -0.0477 | -0.0477 | +0.0000 | -0.0477 |
| (-1,0) | 90.5768 | **+0.0552** | +0.0552 | +0.0290 | +0.0262 | +0.0552 |
| (-2,0) | 90.6650 | **+0.0464** | +0.0464 | +0.0634 | -0.0170 | +0.0464 |
| (-3,0) | 90.6961 | **+0.0375** | +0.0375 | +0.0898 | -0.0523 | +0.0375 |
| (-4,0) | 90.5035 | **+0.0308** | +0.0308 | +0.1410 | -0.1102 | +0.0308 |
| (-4,-2) | 90.1071 | **+0.0693** | +0.0693 | +0.1517 | -0.0824 | +0.0693 |

**Every cell reproduces `C11` S6.1's arbiter column exactly**, on BOTH builds
and to four decimals -- which is the first time that table has been measured
anywhere but the Windows box.  The two builds agree cell for cell.

* the accepted trade is there and it is the recorded size: **(-1,0) moves
  +0.0262 the wrong way** (0.0290 -> 0.0552), against C11's stated differential
  floor of 0.003-0.015, so it is a real move and not noise;
* the other four tilted orders improve by **0.017 / 0.052 / 0.110 / 0.082**;
* on axis the selector is **byte-identical** -- `(0,0)` reads the same sha
  (`8ecba1ec`) as the v5.32 arm, because the C1 null gate never opens there;
* worst-case residual **0.1517 -> 0.0693** and field-angle spread
  **0.1994 -> 0.1170**;
* the residual's monotone growth with field angle is gone: v5.32 runs
  0.029 / 0.063 / 0.090 / 0.141 / 0.152, the flip reads
  0.055 / 0.046 / 0.038 / 0.031 / 0.069 with no trend.  The exact-ray oracle
  says every order is equally diffraction-limited, so a residual with no
  field-angle dependence is the shape a correct chain should have.

At degree 4 the selector is worth much more than at degree 6 -- (-1,0) goes
0.9344 -> 0.3265, (-2,0) 0.7740 -> 0.3721 -- which is consistent with C12's
own account: the branch and the residual degree are both fixing the same
decentre-driven model error, so the second one to be applied buys less.

### 10.4 Production acceptance, conservation, C7

`focus_scan_121.py` **unedited**, pure library defaults, N=2048, `rs=4`,
NFC=8192, WF=4.0, through `c13_with_stepdown.py` with `SELECTOR=predictor`:

| | **SHIPPED 5.32.1** | v5.32 selector, step-down on | step-down off |
|---|---|---|---|
| `AT-PLANE` | 3.350 um / 90.3 / 99.7 / 99.8 | 3.350 / 90.3 / 99.7 / 99.8 | 3.350 / 90.3 / 99.7 / 99.8 |
| `BEST-FOCUS[peak]` | dz = **+0 um** | +0 um | +0 um |
| FWHM / EE3 / EE6 / EE12 | 3.350 um / 90.3 / 99.7 / 99.8 | same | same |
| peak | **5.529e+03** | 5.529e+03 | 5.529e+03 |
| `dz = +5 um` | 3.450 um / 89.6 / 99.7 / 99.8 | same | same |

**Three arms, one printed line**, peak included -- and it is the line `C11`
S6.6 (a) and `D121_RESIDUAL_CLOSURE` S6.3 (a) already record.  C11 predicted
this and gave the structural reason: `focus_scan_121.py` runs the single
ON-AXIS beam, the chief ray sits on the grid centre, the C1 null gate never
opens, and the selector therefore never runs.  Measured here rather than
inherited.

**Conservation and the halo, 6 of 6, shipped configuration.**
`energy_stage_audit_121.py` unedited, `CONFIGS=ship`, `NULL=1`, all six orders:

| order | `P_out/P_in` | (C11 arbiter) | `g4` | (C11) | `amax4` | (C11) | `r_rms` mm | NULL |
|---|---|---|---|---|---|---|---|---|
| (0,0) | 0.994315 | 0.994315 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.8383 | 0.000e+00 |
| (-1,0) | 0.994063 | 0.994063 | 1.962e-11 | 1.962e-11 | 1.716e-05 | 1.716e-05 | 0.8384 | 0.000e+00 |
| (-2,0) | 0.994132 | 0.994132 | 6.783e-11 | 6.783e-11 | 3.213e-05 | 3.213e-05 | 0.8382 | 0.000e+00 |
| (-3,0) | 0.994071 | -- | 1.302e-09 | -- | 5.625e-05 | -- | 0.8380 | 0.000e+00 |
| (-4,0) | 0.994004 | -- | 8.841e-09 | -- | 1.075e-04 | -- | 0.8376 | 0.000e+00 |
| (-4,-2) | 0.993826 | 0.993826 | 9.114e-09 | 9.114e-09 | 1.116e-04 | 1.116e-04 | 0.8375 | 0.000e+00 |

Where `C11` S6.6 (c) printed a row, the shipped 5.32.1 configuration reproduces
it **to every digit** -- `P_out/P_in`, `g4`, `amax4` and `r_rms` alike.  The
two orders C11 did not print are measured here.  Every bound holds: `P_out/P_in`
inside C2's [0.9850, 1.00050], every `g4` <= 1e-3 of its C3 bound, `amax4` 10x
under C4's 1.0e-03, `r_rms` within 0.0002 mm of C5's 0.030 tolerance.  The NULL
column is two identical shipped runs, all six stages, `array_equal = True`.

**C7 silent.**  `grep -c "HALO self-check FAILED"` and
`grep -c "energy self-check FAILED"` both read **0** across every run in this
section.

### 10.5 The flip's fail-before

`SELECTOR=gate` (both flags `False`) must restore the pre-flip library
exactly.  `rc_resdeg_121.py`, `DEGS=6`, all six orders, against the SAME runner
executed before the defaults moved:

```
gate arm    8ecba1ec  ae823405  5555e8b1  43777231  7ef1e5a5  b0f48388
pre-flip    8ecba1ec  ae823405  5555e8b1  43777231  7ef1e5a5  b0f48388
```

**6 of 6, bit for bit, on the design.**  At unit level the same statement is
`test_niche_c12::test_the_fail_before_restores_both_earlier_eras_bit_for_bit`,
which additionally pins that the two eras genuinely DIFFER on its fixture, so
neither identity is vacuous.

Note what the two fail-befores are, because they are different switches and
this document contains both:

| switch | restores |
|---|---|
| `LSTSQ_CONDITIONING_STEPDOWN = False` | the pre-C13 SOLVER (S6.1: six recorded hashes, bit for bit) |
| `DECENTRED_FIT_PREDICTOR = DECENTRED_FIT_ARBITER = False` | the v5.32 SELECTOR (this section) |

They are independent: the table above was taken with the step-down ON.

### 10.6 D4 and the degree sweep in the SHIPPED configuration

**D4, the blocker's own test, in the shipped configuration:**

```
Windows, SHIPPED selector   test_niche_d4_dgrating.py  ->  59 passed (312 s)
Linux,   v5.32 selector     (same tree, same solver)   ->  59 passed (158 s)
Windows, SHIPPED selector, run ALONE, the reported case:
                            -k matches_the_manual_hand_split  ->  1 passed
Linux,   SHIPPED selector, run ALONE, same -k          ->  1 passed (563 s)
```

**The degree sweep, step-down ON, shipped selector**, `c13_degree_sweep.py on`:

| degree | 2 | 3 | 4 | 5 | **6 (SHIPPED)** |
|---|---|---|---|---|---|
| sphere, Windows | 6.2341e-07 | 6.2529e-07 | 7.2352e-07 | **6.3808e-04** | 6.7181e-07 |
| sphere, Linux | 6.2336e-07 | 6.2536e-07 | 7.2358e-07 | **6.3766e-04** | 6.7186e-07 |
| parabola, Windows | 1.1388e-10 | 1.1388e-10 | 1.1388e-10 | 1.1388e-10 | 1.1388e-10 |
| roughness, Windows | 3.71e-02 | 3.67e-02 | 3.70e-02 | 3.70e-02 | 3.48e-02 |
**And the degree-5 outlier is BUILD-INDEPENDENT**: 6.3808e-04 (Windows) against
6.3766e-04 (Linux), agreeing to 0.07 %.  That is the load-bearing part.  A
deterministic branch decision that both builds take identically is a property
of the selector; the thing C13 removed was the opposite -- a quantity that
disagreed between the builds by 148,000x.  The two failures look alike in a
sweep table and are not the same class at all, and the way to tell them apart
is exactly the both-builds measurement that the standing rule now requires.

**Degree 5 is an outlier and it is NOT the solver.**  Reported rather than
smoothed, and attributed in four measurements on the D4 chain at `u` = 0.6163:

| arm | degree | `u*` | branch chosen | `rel` |
|---|---|---|---|---|
| `SELECTOR=predictor` | 4 | 0.0000 | OFF-CENTRE | 7.2352e-07 |
| `SELECTOR=predictor` | **5** | **0.8411** | **CONCENTRIC** | **6.3808e-04** |
| `SELECTOR=predictor` | 6 | 0.4686 | OFF-CENTRE | 6.7181e-07 |
| `SELECTOR=arbiter` | 5 | -- | CONCENTRIC | 6.3808e-04 |
| `SELECTOR=gate` | 5 | -- | (v5.32 branch) | 6.5757e-07 |

Read across: at degree 5 the crossover `u*` moves ABOVE the fixture's decentre,
so the selector routes that call to the CONCENTRIC fit -- and on the concentric
branch at 0.62 w this two-route comparison amplifies the arms' ~1e-6 input
difference by ~1000x.  It reproduces on BOTH builds to 0.07 %, and it is identical under `SELECTOR=arbiter`, so it is
**C11's choice, not C12's prediction**; it vanishes under `SELECTOR=gate`; and
BOTH arms take the same branch (`u*` agrees to 1e-5 between them), so it is not
a branch FLIP between the routes either.

Three things follow, and the third is the one that matters:

1. it is the selector's own documented discontinuity -- `C11` S5.2 says in as
   many words that C1 *relocated* its discontinuity rather than removing it --
   surfacing on a synthetic two-route equivalence test;
2. the concentric branch being chosen is the scorer working as specified: it
   picks concentric because the concentric fit reproduces the traced OPL BETTER
   there (`E_c` = 3.63e-10 against `E_o` = 1.11e-09).  Fitting the traced map
   better and being less sensitive to a 1e-6 input perturbation are not the
   same property, and this fixture separates them;
3. **the shipped degree is 6, where the selector picks OFF-CENTRE and the
   reading is 6.7181e-07.**  D4 passes in the shipped configuration on both
   builds, and no shipped configuration reaches the degree-5 cell.

It is left as an open item (S8) rather than fixed here: it is a property of the
selector that the flip made reachable, not of the solve C13 repaired, and the
fixture that shows it is synthetic.

### 10.7 The 11 red tests -- RESOLVED in S11, and not the way this section guessed

**Superseded by S11.**  This section is kept because its guess was wrong in an
instructive way, and the wrong guess is the thing worth recording.

Under the flip as first written the niche selection read **11 failed, 263
passed** on Windows.  All eleven are in two files:

```
test_niche_d6_exact_tilted_leg.py   4  exact leg reachable / beats paraxial /
                                       decentre-penalty envelope / refusal
                                       downgrade
test_niche_d7_decentred_fit.py      7  order raise flattens the wavefront
                                       [0.5, 1.0] / order rises only off centre
                                       / order steps down / the decentred path
                                       really did change / fold regularisation
                                       still load-bearing / hard-mask arm
                                       ghosts on every build
```

This section then assumed they were all era pins, on the strength of two of
them recovering when the selector was pinned off.  **They are not.**  Pinning
one flag off and observing recovery says only that the flag is implicated; it
does not say the test was wrong.  Adjudicated properly against all three rungs
(S11.1), nine of them recover with the PREDICTOR off and the ARBITER still on
-- i.e. they were reporting a real regression in one of the two flipped
constants -- and the other two are C13-cured (S11.5).  Nine era pins would have
buried the finding.

The original text of this section follows, unedited, because "what has to
happen before this ships" was the right instinct attached to the wrong
diagnosis:

**They are the C1/C6/D1 class, and that is measured rather than assumed**:
pinning `DECENTRED_FIT_PREDICTOR = DECENTRED_FIT_ARBITER = False` and re-running
two of them (`test_the_fit_order_actually_rises_only_off_centre`,
`test_the_decentred_path_really_did_change`) gives **2 passed in 14 s** with no
other change.  Each names a D6/D7 mechanism and then reaches it through a branch
the selector now chooses differently -- D7's order raise, for instance, happens
only on the OFF-CENTRE branch, and the predictor routes those fixtures to the
CONCENTRIC one, so the test measures a branch it was not written for.

**What has to happen before this ships**, per file rather than in bulk: each arm
gets the era pin its subject requires (`PREDICTOR`/`ARBITER` False for the D6/D7
era, and `LSTSQ_CONDITIONING_STEPDOWN` False as well for any arm that is a
pre-D1 fail-before -- S5.4 shows those exist), assertions verbatim.  Two of the
eleven (`test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order`,
`test_the_hard_mask_arm_ghosts_on_every_build`) are named like D1's witnesses
and should be checked against S5.4 specifically: they may be CURED by C13
rather than merely rerouted by the selector, and those two facts want different
comments.

Nothing in S10.3-S10.6 depends on this.  The design-121 per-order table, the
production acceptance, the conservation battery and both fail-befores were
taken on the library, not on these tests.

---

---

## 11. The predictor half of the flip was REVERTED, on evidence

S10 shipped two constants. Finishing S10.7's era-pin debt adjudicated them
separately, and they did not survive together.

### 11.1 What the 11 red tests actually were

S10.7 recorded 11 failures and guessed they were all era pins. They are not.
Running each under the three rungs of the ladder � every arm pinned
explicitly, nothing inherited:

| test | pred+arb | arb-only | gate |
|---|---|---|---|
| d6 `test_the_exact_leg_is_reachable_under_a_tilted_carrier` | FAIL | **PASS** | PASS |
| d6 `test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle` | FAIL | **PASS** | PASS |
| d6 `test_decentred_carrier_decentre_penalty_envelope` | FAIL | **PASS** | PASS |
| d6 `test_the_refusal_can_be_downgraded_and_then_it_is_worse` | FAIL | **PASS** | PASS |
| d7 `test_the_off_centre_fit_order_raise_flattens_the_exit_wavefront[0.5]` | FAIL | **PASS** | PASS |
| d7 `..._flattens_the_exit_wavefront[1.0]` | FAIL | **PASS** | PASS |
| d7 `test_the_fit_order_actually_rises_only_off_centre` | FAIL | **PASS** | PASS |
| d7 `test_the_order_steps_down_when_the_disc_cannot_constrain_it` | FAIL | **PASS** | PASS |
| d7 `test_the_decentred_path_really_did_change` | FAIL | **PASS** | PASS |
| d7 `test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order` | FAIL | FAIL | FAIL |
| d7 `test_the_hard_mask_arm_ghosts_on_every_build` | FAIL | FAIL | FAIL |

Two clean groups, and neither is what S10.7 assumed:

* **nine of them recover with `DECENTRED_FIT_PREDICTOR = False` and
  `DECENTRED_FIT_ARBITER` STILL TRUE.**  They are not era pins at all.  They
  are nine tests correctly reporting that one of the two flipped constants
  makes the library worse, and the arbiter is not the one;
* **two of them fail on every rung** and recover only with
  `LSTSQ_CONDITIONING_STEPDOWN = False`.  Those two are S5.4's finding again
  � C13-cured, not selector-rerouted (S11.4).

Pinning the first nine would have buried the finding under nine comments about
eras. That is exactly the failure mode this campaign keeps writing notes about,
and it was one commit away.

### 11.2 The predictor loses 32 % of the encircled energy on an ANALYTIC oracle

Niche D6's fixture is the `K = -n^2` conic stand-in whose truth is analytic and
decentre-invariant, scored against an inline exact conic raytrace that shares
no code with the library. At `|c|/w` = 1.0:

| `PREDICTOR` | `ARBITER` | EE2 / oracle | FWHM / oracle | spot off the Fermat focus |
|---|---|---|---|---|
| **True** | True | **0.6670** | 1.0952 | **3.96e-07 m** |
| **True** | False | **0.6670** | 1.0952 | **3.96e-07 m** |
| False | True | 0.9819 | 1.0000 | 9.87e-09 m |
| False | False | 0.9819 | 1.0000 | 9.87e-09 m |

The predictor is the sole variable: the arbiter flag moves nothing in either
direction, which is the S10.1 no-op invariant showing up as a two-row tie.
D6's own envelope test says it in words �
`decentred EE2 ratio 0.6670 is outside the MEASURED post-D7 envelope (0.9828
at |c|/w = 1.0). ... Below it: it regressed.`

### 11.3 The mechanism, out of the library's own mouth

On that call the library **already warns**, and then applies the losing choice:

```
the predictor selects the CONCENTRIC branch from |c|/w = 1.0001 against a
crossover u* = 1.4160 (spectral exponent m_eff = 8.003, spectrum resolved at
order 14, box-fit residual 8.855e-11 m, modelled OPL residuals 1.939e-07 m
concentric / 1.508e-06 m off-centre); the arbiter's own measured residuals are
1.368e-07 m concentric / 6.712e-09 m off-centre and select the OFF-CENTRE one.
The PREDICTOR's choice is applied.
```

Instrumented directly, the same call reads model `E_c` = 1.9364e-07 against
`E_o` = 1.5065e-06 (so `u*` = 1.4161 > `u` = 1.0001 -> concentric), while the
arbiter's MEASURED pair is 8.70e-11 off-centre against 1.5065e-06 concentric.
**Off-centre is 17,000x better and the closed form has the ordering inverted.**
C11's own concentric-arm sweep says the same thing from the other side: the
concentric fit is 26x its floor at 0.5 w and 312x at 1.0 w, and C11 S3.1
measured a residual of 67.3 points when that branch is forced at 0.965 w.

**Why C12's validation missed it.** C12 S3.2 validated three geometries � f/3
and f/6 singlets at two beam radii � and the model landed on the spline
oracle's crossover to 0.03 % on the one with a nonzero crossover (`u*` ~ 0.57).
D6's stand-in is a FOURTH geometry and sits at `u` = 1.0, past every crossover
in that table. Three designs is not a validation set for a closed form that
ships as a default, and the fourth was already in the suite.

### 11.4 ... and on design 121 it does nothing at all

C12 S3.4 says the launch-box spectrum on design 121 is UNRESOLVED, so the
predictor falls back to the arbiter's measured pair and is algebraically the
arbiter there. **Measured, not inferred**, and in the strongest available
form.

First, across four design-121 runs with the flag ON � `rc_resdeg_121` on both
builds, `focus_scan_121`, `energy_stage_audit_121` � the predictor/arbiter
disagreement warning fires **zero times**.

Second, and decisively: `rc_resdeg_121.py` re-run with `SELECTOR=arbiter`
against the `SELECTOR=predictor` run reproduces the degree-6 intensity hashes
**bit for bit, 6 of 6**:

| order | `SELECTOR=arbiter` (shipped) | `SELECTOR=predictor` (reverted) | `SELECTOR=gate` (v5.32) |
|---|---|---|---|
| (0,0)   | `8ecba1ec` -0.0477 | `8ecba1ec` -0.0477 | `8ecba1ec` -0.0477 |
| (-1,0)  | `e050db96` +0.0552 | `e050db96` +0.0552 | `ae823405` +0.0290 |
| (-2,0)  | `79f23160` +0.0464 | `79f23160` +0.0464 | `5555e8b1` +0.0634 |
| (-3,0)  | `06f13072` +0.0375 | `06f13072` +0.0375 | `43777231` +0.0898 |
| (-4,0)  | `6744ad41` +0.0308 | `6744ad41` +0.0308 | `7ef1e5a5` +0.1410 |
| (-4,-2) | `cd0b69d7` +0.0693 | `cd0b69d7` +0.0693 | `b0f48388` +0.1517 |

So on design 121 the predictor is **byte-inert**: reverting it changes nothing
on the design, to the last bit, at every order. The third column is the
arbiter doing real work against the v5.32 gate � which is why the arbiter
ships and the predictor does not. **Every design-121 number in S10.3-S10.4 is
therefore unaffected by the reversion, and that is a bit-identity rather than
an argument.**

So the flag is **inert where it was wanted and harmful where it is live**.
That is the whole case, and it is why `DECENTRED_FIT_PREDICTOR` goes back to
`False` while `DECENTRED_FIT_ARBITER` stays `True`.

**What still ships from S10, unchanged**: the arbiter, its per-order table
(S10.3, both builds), its production acceptance (S10.4), its conservation
battery, and its bit-exact fail-before (S10.5). Everything measured there was
the arbiter's, precisely because the predictor never deviated from it on that
design.

### 11.5 The two C13-cured witnesses, adjudicated and pinned

The remaining two are S5.4's D1 finding a second time. Both degenerate the
restriction to D1's hard NaN mask and require the result to ghost. Measured on
D7's own fixture, off-beam fraction of peak and fold-caustic warnings:

| step-down | degree | weights | off-beam | folds |
|---|---|---|---|---|
| OFF | 4 | regularised | 0.0002 | 0 |
| OFF | 4 | **hard mask** | **0.5213** | **1** |
| OFF | 6 | hard mask | 0.0002 | 1 |
| **ON** | 4 | **hard mask** | **0.0002** | **0** |
| ON | 6 | hard mask | 0.0002 | 0 |

At the era-pinned degree 4 the hard-mask ghost falls **2600x** and the fold
detector goes silent. `test_the_hard_mask_arm_ghosts_on_every_build`'s own
docstring had already diagnosed this and stopped one step short � *"whose
normal matrix is ill-conditioned BY CONSTRUCTION ... set by which side of the
instability that build's LAPACK lands on"* is C13's finding, written down
before C13 existed. The four-orders-of-magnitude build spread it records is a
property of the library BEFORE the step-down, which is now what that arm pins.

Handled the C9/C10 way, and NOT by blanket pin:

* both witnesses era-pin `LSTSQ_CONDITIONING_STEPDOWN = False` alongside their
  existing `_REMAP_RESID_EIKONAL_DEGREE = 4` pin, assertions word for word;
* a new sibling, `test_c13_cures_the_hard_mask_fold_at_the_d7_order`, asserts
  the CURE directly � fail-before live (`> 0.1`, fold fires), pass-after
  (`< 0.01`, no fold), and a `> 100x` separation between them. The improvement
  is asserted, not merely pinned away.

### 11.6 What this costs, and what it does not

`DECENTRED_FIT_PREDICTOR = False` restores C12's shipped state exactly, so
nothing in the C12 audit is retracted: its derivation, its `m_eff` algebra, its
crossover closed form and its three-geometry validation all stand, and the flag
remains available opt-in. What changes is only the DEFAULT, and the reason is a
fourth geometry that its validation set did not contain.

The 0.026-point (-1,0) trade S10 records is the ARBITER's and is unaffected �
that decision was taken on C11's evidence and still holds.

### 11.7 A third era pin, found by the full run: C12's own OPL capture

The 20-file niche run turned up two more, in a file nobody had touched:

```
test_niche_c12_physics_fit_selection.py
  test_the_spectral_tail_carries_the_whole_candidate_residual
  test_the_shell_spectrum_of_a_rotationally_symmetric_lens_is_even
```

They pass ALONE in all four flag configurations and fail IN FILE ORDER, which
is the signature of a harness assumption rather than a physics change.
Adjudicated before being touched:

* **not C13.**  The whole file fails identically with
  `LSTSQ_CONDITIONING_STEPDOWN` off (`2 failed, 18 passed` either way);
* **not this study's edits.**  `git show HEAD:` of that file, run against this
  working tree, fails the same two.

**The cause is a positional index in C12's own `_capture_traced_opl`**, which
takes `seen[2]` -- the third `_Cheb2DEvaluator` build of one element call --
and calls it "the UNMASKED traced OPL".  How many builds a call makes is
exactly what the selectors change, and C12's own S4.4 counts them: **3 on the
pure gate, 5 with the C11 arbiter, 9 with the C12 predictor.**  So `seen[2]`
is the box fit on the gate and a candidate trial fit on either selector, and
the two spectrum tests were quietly running their algebra on the wrong array
the moment `DECENTRED_FIT_ARBITER` shipped `True`.

Pinned the same way as everything else here: `_capture_traced_opl` now sets
BOTH selector flags `False` for the duration of its capture, alongside the
decentre-gate pins it already had, and restores all four. Assertions
untouched; the file reads `20 passed`.

**This is the fourth instance in two documents of one defect class** -- an
index, a bar or an arm expressed relative to a default and evaluated after the
default moved (`fc_production_taper`'s baseline row, D1's two witnesses, D7's
two witnesses, and now this). It is the class niche C14's flag registry exists
to close, and it is worth noting that the registry landed in the same tree
this was found in.

### 11.8 And a fourth, on the Linux build only: D3's degree-separation bar

The 20-file run was `500 passed` on Windows and `1 failed, 499 passed` on
Linux. The one is `test_niche_d3_guards::
test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it`,
and it is **C13's, not the selector's** -- it fails with either selector state
and passes with `LSTSQ_CONDITIONING_STEPDOWN` off, on both.

It is also the best single illustration of what the step-down does. The test
compares linearity errors at residual degree 4 against degree 6:

| | `bad4` | `bad6` | `bad4/bad6` | `bad6/good6` |
|---|---|---|---|---|
| Windows, step-down OFF | 23.2447 | 1.2135 | 19.15x | 80.54x |
| Linux, step-down OFF | 14.2337 | 0.9863 | 14.43x | 17.16x |
| **Windows, step-down ON** | 22.4613 | **3.3113** | 6.78x | **258.90x** |
| **Linux, step-down ON** | 10.6432 | **3.3246** | 3.20x | **259.94x** |

**`bad6` was a per-build lottery and is not one any more**: 1.2135 against
0.9863 is a 23 % build spread; 3.3113 against 3.3246 is **0.4 %**. The
degree-6 arm's linearity error was being SUPPRESSED by whichever null-space
draw that build's LAPACK happened to take, and C13 stopped it being suppressed.
`bad4` barely moves. So the ratio this test bars falls to 6.78x / 3.20x purely
because its DENOMINATOR became honest -- the mechanism claim ("the residual
degree is what moves the multiplexed route") is intact and still large.

The test's own docstring says the 5x bar sits "with 3x of headroom on the
weakest" build. That headroom is exactly what C13 consumed, so:

* the arm era-pins `LSTSQ_CONDITIONING_STEPDOWN = False`, **assertions word for
  word**, with the table above in its docstring;
* a new sibling, `test_c13_makes_the_d3_separation_build_independent`, asserts
  the SHIPPED era and asserts something **strictly stronger than this test
  could before**: `bad6 > 100 * good6`. Pre-C13 that bar was unreachable -- the
  weaker build read 17.2x -- and it now passes at 258.9x / 259.9x, one number
  on both builds instead of a 4.7x disagreement about it.

That is the pattern this whole document keeps arriving at: where C13 removes a
build-dependence, the right response is to assert the new invariant, not to
loosen the old bar.
