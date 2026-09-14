# VERIFY-WP-B10 -- independent adversarial re-verification of the disc-orthogonal traced fit basis

Subject: commit `f8d5466f` on `audit-fixes-2026-09` (`fit_basis='chebyshev' | 'zernike'` on the traced ray
fits, opt-in), its report `WP-B10_REPORT.md`, its changelog, and
`tests/unit/test_audit2609_b10_zernike_fit_basis.py`.

I did not write WP-B10.  Everything below is a measurement I took, on fixtures and with oracles I rebuilt,
on this machine, with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, Python 3.14.6, one
process at a time.  No git write command was issued; the only git commands used were
`git log` / `git show` / `git diff` / `git archive`, all read-only.

**Verdict in one line.**  The headline is right, and it is right for the reason the report gives: a change
of basis inside one span cannot move a least-squares answer, and the audit's premise that the SQUARE basis
is what couples marginal rays into defocus is refused by measurement, not by argument.  I could not break
it.  Four things the report states are weaker than it says, one of them materially: **the
`(R_data/R_disc)^2`-per-degree decay law that B10-D3 invites a future author to extrapolate from does not
extrapolate** -- measured on the report's own fixture the decay SATURATES at ~1.19 decades per degree from
`R_data/R_disc ~ 3.4` upward while the law keeps growing to 2.13 at 11.6.  Three defects of coverage (not of
behaviour) are fixed here: the disc the entry point normalises the basis to was pinned nowhere, where the
basis reaches the returned field was documented nowhere, and the decay law was pinned only loosely enough
to admit the wrong extrapolation.

---

## 1. Verdict table

| # | claim (as the report / changelog states it) | verdict | my oracle | my numbers |
|---|---|---|---|---|
| **B10-1** | the two bases span the same space, so the fit cannot move; the audit's premise is refused | **VERIFIED** | four launch lattices of my own (square-clipped-to-disc, disc-uniform sunflower, decentred + D1 skirt at `R_data/R_disc` 4.86, tiny-disc ghost at 12.14), rank + projector gap `\|P_c - P_z\|_2`; and the weighted LS answer solved by QR on the well-conditioned Chebyshev design as an independent reference | equal ranks and `\|P_c - P_z\|` = 1.3e-14 .. 6.9e-14 at order 6 on all four; both library answers score IDENTICALLY against the QR reference, to every printed digit, at every order 6..20 and on the disc alone (sec. 3, 4) |
| **B10-1a** | the decentred exit slope is the same number on both bases; **2.371 / 1.683 urad** at the shipped order | **VERIFIED, character for character** | D7's `K = -n^2` Fermat singlet rebuilt by its method, my own exit-slope estimator, the closed-form sphere | `2.371 / 2.371` at 0.5 w and `1.683 / 1.683` at 1.0 w; cross-basis 2.7e-12 and 1.9e-11 **urad** |
| **B10-1b** | the C11 arbiter's two candidate residuals agree across the bases, so its verdict cannot move | **VERIFIED** | my own N-SF11 singlet, three decentres, orders 6..20, the element's own `_decentred_fit_score` | identical to all 7 printed digits in 28 of 30 cells; the two exceptions are order 20 (3.671597e-12 vs 3.670871e-12, and 3.426621e-12 vs 3.128033e-12) where the Zernike Gram is already dead -- the verdict still does not move, the gap being 4 decades larger |
| **B10-2** | marginal rays couple into defocus identically in both bases; it is the weighted problem and the ORDER, not the basis | **VERIFIED, and strengthened** | the coupling measured as a FUNCTION of `_FIT_DISC_OUTSIDE_WEIGHT_REL`, swept x0 / x0.25 / x1 / x4 / x16, at orders 6 / 10 / 16 | the `Z(2,0)` shift is the same number in both bases to **1e-19 m** at every weight and every order; the coupling/base ratio is **1.0000** in all 15 cells.  The report measures the coupling at one weight; sweeping the weight is the stronger form of the same refusal and it holds |
| **B10-3** | what the basis buys is conditioning, where the samples are a disc | **VERIFIED** | the library's own `_gram_rcond` through the shipped solve path | on-axis / disc-masked fits: 1e-11-class -> 1e-1-class, step-down 4-5 of 5 solves -> **0 of 5** at orders 6-12 on my fixture; on a hard-masked concentric fit, 4.7 decades (sec. 4, 7) |
| **B10-3a** | ... and the advantage decays ~1.2 decades/degree because a disc column grows as `(r/R)^n`, crossing over near the shipped order 16 | **VERIFIED-WITH-NOTES** -- the crossover is real and lands where the report says, but the stated LAW is wrong away from the report's own ratio | the same `_gram_rcond`, six disc radii on ONE launch lattice, orders 6 -> 10 | slope 0.44 / **1.19 / 1.19 / 1.18 / 1.14 / 1.11** dec/deg at ratios 2.03 / 3.38 / 4.50 / 5.79 / 8.11 / 11.58 against a law predicting 0.61 / 1.06 / 1.31 / 1.53 / 1.82 / 2.13.  It saturates; the law does not.  **See sec. 8 -- this is the one finding against the report.** |
| **B10-4** | cost 1.12x at `ray_subsample=8`, 1.7x at `ray_subsample=1` | **VERIFIED** | interleaved medians of 7 on my own decentred call | **1.14x** at rs=8 (246.5 -> 280.2 ms) and **1.49x** at rs=1 (10.40 -> 15.49 s) |
| **BYTE** | `fit_basis='chebyshev'` is byte-identical, 15 configurations, 12 distinct fields, "a change that moved the default fit by one ULP would show here" | **VERIFIED** -- archive-to-archive, on my own 23-configuration battery, **with a positive control the report does not have** | `git archive f8d5466f^ lumenairy` vs `git archive f8d5466f lumenairy`, extracted read-only, child processes, `lumenairy.__file__` asserted, never through pytest | **23 configurations, 0 mismatches, 16 distinct fields.**  A tree in which every fit COEFFICIENT is moved by exactly 1 ULP is detected in **15 of 23** -- so the claim holds -- but 8 are structurally blind (sec. 6) |
| **REFUSALS** | `fit_basis='foo'` refuses with the sec. 2 prefix; `newton_fit='spline'` and `inversion_method != 'newton'` refuse | **VERIFIED** | six refusal probes through the entry point | all six raise `ValueError` starting `apply_real_lens_traced:` and naming the setting; `newton_fit='auto'` (the entry-point default) is correctly ACCEPTED, because the gate runs after `'auto' -> 'polynomial'` |
| **CONFIG** | `LensNumerics.fit_basis` round-trips, `_NUMERICS_FOR` carries it, the a16 census counts it | **VERIFIED**; the precedence-or-raise arm is **VACUOUS** and should be recorded as such | `LensConfig.to_kwargs`, `_NUMERICS_FOR`, `dataclasses.fields`, and a driven element call | field present, `to_kwargs` emits it for both entry points, census = **40**, `numerics=LensNumerics(fit_basis='zernike')` returns a field `np.array_equal` to the keyword call.  The conflict branch **cannot be reached** for a two-valued vocabulary whose default is one of the two (sec. 7) |
| **B10-D1** | the imap's exit fits and C12's box spectrum stay Chebyshev | **VERIFIED** | source + a driven call | `_decentred_fit_spectrum` constructs its own evaluator with no basis keyword, and reads `a + b` as the total degree -- which would be wrong for a Zernike `(n, m)`; it is unreachable, and that is worth knowing |
| **B10-D2 / D3 / D4 / D5** | deferred items | **ACCEPTED**, with D3 amended by sec. 8 | -- | -- |
| **NEW V-1** | which disc the ENTRY POINT normalises the basis to is pinned nowhere | **DEFECT (coverage), FIXED** | a mutation that forces the disc concentric passed 32 of 33 tests before the fix | sec. 9.1 |
| **NEW V-2** | where the basis reaches the returned field is documented nowhere, and on the DEFAULT `ray_subsample` it reaches nothing | **DEFECT (documentation), FIXED** | driven calls at rs 8 / 4 / 2 / 1 with the model's gate read from `_imap_out` | `np.array_equal` across the bases at rs 8, 4 and 2; differs by 2.2e-12 at rs 1.  sec. 9.2 |
| **NEW V-3** | the decay law is pinned only loosely enough to admit B10-D3's extrapolation | **DEFECT (coverage), FIXED** | sec. 8's six-ratio table | sec. 9.3 |

---

## 2. Isolation, and why these numbers are about this work package

Two trees, extracted READ-ONLY with `git archive` into
`.../scratchpad/verify_b10/{parent,head}`:

```
git archive f8d5466f^ lumenairy | tar -x -C .../verify_b10/parent
git archive f8d5466f  lumenairy | tar -x -C .../verify_b10/head
diff -rq parent/lumenairy head/lumenairy
  -> Files .../_lens_traced.py differ
  -> Files .../lens_config.py differ          (and nothing else)
```

So the two trees differ in exactly the two files this commit touches, and the shared working tree's
in-progress edits from other engineers (`carrier.py`, `pmm/*`, `system.py`, `_lens_real.py`,
`analysis/*`, `sources/*`, `mft.py`) cannot enter any comparison.  `git diff HEAD --` over
`_lens_traced.py`, `_lens_imap.py` and `lens_config.py` is empty, so the working tree carries the commit's
version of the files I own.

Every measurement in sections 3-8 was taken by a child process with `cwd` and `PYTHONPATH` set to one of
those extractions, with the editable-install finder stripped from `sys.meta_path` and
`lumenairy.__file__` asserted to lie inside the extraction before anything was measured.  Never through
pytest.

**My fixtures are not the engineer's.**  The element-level ladder uses an N-SF11 singlet at
`f = 5.00 mm`, `lambda = 1.55 um`, 3.60 mm aperture, a 0.80 mm beam on a 384 x 384 grid at 10 um --
a different glass, focal length, wavelength, aperture, beam and grid from D7's.  Its two oracles are the
closed-form Fermat sphere of the `K = -n^2` stigmatic stand-in and an inline exact conic trace (flat
entrance, exact even-conic sag, vector Snell on the gradient normal), both rebuilt by their METHOD.  Their
own cross-check, taken before either graded anything:

```
f_b = 0.005000000 m   stigmatic spread 5.2e-18 m
max |exact conic trace - closed-form sphere| = 8.674e-19 m
   against a float64 evaluation floor of 7.742e-19 m   (ratio 1.1)
```

The D7 fixture is also rebuilt, once, in section 4.2, to check the headline digits.

---

## 3. The span claim, re-derived on lattices the engineer did not use

Four launch lattices, both design matrices built by the library's own `_cheb_vand_2d` and
`_zernike_design`, orders 6..20.  `|P_c - P_z|_2` is the spectral norm of the difference of the two
orthogonal projectors onto the column spaces (computed from economy QR factors), which is the sine of the
largest principal angle: exactly zero if the spans coincide, and roundoff-sized if they coincide
numerically.

| lattice | `R_data/R_disc` | `\|P_c - P_z\|` o6 | o12 | o16 | o20 | ranks equal? |
|---|---|---|---|---|---|---|
| A square clipped to the disc (1 917 samples) | 1.00 | 3.665e-14 | 4.232e-12 | 1.081e-10 | 2.568e-09 | yes at every order |
| B disc-uniform sunflower (6 000) | 1.00 | 1.329e-14 | 2.980e-12 | 9.130e-11 | 2.736e-09 | yes at every order |
| C decentred disc + D1 skirt (19 881) | 4.86 | 6.919e-14 | 1.060e-10 | 2.373e-08 | 4.670e-06 | yes to o12 |
| D tiny-disc ghost (19 881) | 12.14 | 4.829e-14 | 9.273e-11 | 2.620e-08 | 3.992e-06 | yes to o8 |

The spans coincide.  The growth of the gap with the order on C and D is not a span statement but a SCALE
one: a disc-normalised column reaching `4.86^20 = 4e13` makes any rank or projector measured with a
relative tolerance a statement about that scale.  Measured on the shipped path instead -- the WEIGHTED
design the solver actually sees, on D7's own decentred geometry (`R_data/R_disc = 4.503`, D1's weights):

| order | terms | rank `A_c` | rank `A_z` | `sigma_min/sigma_max` `A_c` | `A_z` |
|---|---|---|---|---|---|
| 6 | 28 | 28 | 28 | 1.161e-05 | 1.563e-01 |
| 10 | 66 | 66 | 66 | 6.206e-06 | 3.664e-05 |
| 14 | 120 | 120 | 120 | 4.871e-06 | 7.129e-09 |
| **16 (shipped)** | 153 | 153 | **153** | 4.594e-06 | **9.581e-11** |
| 18 | 190 | 190 | **119** | 4.448e-06 | 1.276e-12 |
| 20 | 231 | 231 | **124** | 4.359e-06 | 1.672e-14 |

**Worth stating because the report does not:** at the SHIPPED order the opt-in design is still full rank
(`cond(A) ~ 1.0e10`); from order 18 up, on the decentred branch, it is numerically rank-deficient in
float64 and the weighted least-squares problem no longer determines its own coefficients.  The element
already warns there (`_solve_lstsq_thread_safe`'s "the Gram matrix screened as numerically singular"), so
nothing is silent, and the report's order-20 rows say the same thing in the Gram; but "the advantage is
negative above the crossover" understates it -- above the crossover the opt-in basis is not merely worse
conditioned, it is rank-deficient.  See sec. 10 for the recommendation.

**And the fit does not move.**  The same samples solved through the shipped `_Cheb2DEvaluator` on both
bases, each scored against an independent reference -- the weighted LS answer by QR on the CHEBYSHEV
design, whose condition number is 6.1e2 .. 1.3e3 on every row below, so the reference is the exact
minimiser to ~1e-13 relative:

| fixture C, `x_out` | o6 | o10 | o14 | o16 | o18 | o20 |
|---|---|---|---|---|---|---|
| chebyshev vs reference, over the launch square | 8.051e-01 | 1.273e-01 | 2.167e-02 | 9.015e-03 | 3.341e-03 | 1.231e-03 |
| zernike vs reference | 8.051e-01 | 1.273e-01 | 2.167e-02 | 9.015e-03 | 3.341e-03 | 1.231e-03 |
| **chebyshev vs zernike** | **5.7e-13** | **4.9e-12** | **7.8e-13** | **6.6e-10** | 3.4e-07 | 1.9e-08 |
| in-disc error, cheb / zern | 6.845e-04 / 6.845e-04 | 9.100e-05 / 9.100e-05 | 1.174e-05 / 1.174e-05 | 4.164e-06 / 4.164e-06 | 1.627e-06 / 1.627e-06 | 6.559e-07 / 6.559e-07 |

(fractions of the map's peak).  The two bases miss the reference by the same amount to every printed digit,
everywhere, and their mutual difference is 9 to 12 decades below the fit's own error at and below the
shipped order.  The OPL rows read the same.

---

## 4. The ladder, on my own singlet and on D7's

### 4.1 N-SF11 singlet, three decentres, orders 6..20, `ray_subsample=1`

Exit-slope rms over the beam core against the closed-form Fermat sphere; `sing` counts how many of the
call's least-squares solves screen singular (the C13 step-down firing); `arb resid` is the C11 arbiter's
two candidate residuals as the element computed them; `resid/QR` is the returned coefficients' residual
against an independent QR of the same system.

| decentre | order | urad cheb / zern | min rcond cheb -> zern | sing cheb -> zern | resid/QR | arbiter cheb vs zern |
|---|---|---|---|---|---|---|
| **0.25 w** | 6 | **21.762 / 21.762** | 4.972e-10 -> 6.631e-01 | 5/5 -> **0/5** | 1.000000 | identical to 7 digits |
| | 8 | **7.099 / 7.099** | 9.372e-11 -> 6.285e-03 | 5/5 -> **0/5** | 1.000000 | identical |
| | 12 | **6.050 / 6.050** | 2.724e-11 -> 4.012e-07 | 5/5 -> **0/5** | 1.000000 | identical |
| | 16 | **6.048 / 6.048** | 1.417e-11 -> 3.525e-11 | 5/5 -> 4/5 | 1.000000 | identical |
| | 20 | **6.048 / 6.048** | 9.982e-12 -> 3.729e-15 | 5/5 -> 4/5 | 1.000000 | identical |
| **0.75 w** | 6 | **24.355 / 24.355** | 4.444e-10 -> 4.276e-01 | 4/5 -> **0/5** | 1.000000 | identical |
| | 8 | **10.038 / 10.038** | 8.762e-11 -> 1.638e-03 | 4/5 -> **0/5** | 1.000000 | identical |
| | 12 | **8.422 / 8.422** | 2.640e-11 -> 3.805e-08 | 4/5 -> **0/5** | 1.000000 | identical |
| | 16 | **8.420 / 8.420** | 1.569e-11 -> 1.613e-12 | 4/5 -> 4/5 | 1.000000 | identical |
| | 20 | **8.420 / 8.420** | 1.109e-11 -> **7.186e-17** | 4/5 -> 4/5 | 1.000000 | 3.671597e-12 vs 3.670871e-12 |
| **1.25 w** | 6 | **81.324 / 81.324** | 4.066e-10 -> 1.553e-01 | 4/5 -> **0/5** | 1.000000 | identical |
| | 8 | **57.551 / 57.551** | 1.017e-10 -> 4.314e-04 | 4/5 -> **0/5** | 1.000000 | identical |
| | 12 | **57.506 / 57.506** | 3.148e-11 -> 4.181e-09 | 4/5 -> 4/5 | 1.000000 | identical |
| | 16 | **57.506 / 57.506** | 1.783e-11 -> 4.781e-14 | 4/5 -> 4/5 | 1.000000 | identical |
| | 20 | **57.506 / 57.506** | 1.267e-11 -> **0.000e+00** | 4/5 -> 4/5 | 1.000000 | 3.426621e-12 vs 3.128033e-12 |

**Is the Zernike basis EVER worse on my fixtures?**  Never in the answer: the exit slope is the same number
to the estimator's three decimals in all 30 cells.  Yes in the conditioning above the crossover, and the
worst case is the last row: at order 20 on the 1.25 w arm the Zernike Gram loses positive-definiteness
outright, the arbiter's concentric residual moves by 8.7 %, and the returned FIELD differs from the
Chebyshev one by **2.62e-06** -- the largest cross-basis difference I saw anywhere, and four decades larger
than the 2.6e-12 the same comparison reads at the shipped order 16.  It does not change the arbiter's
verdict (the gap it ranks is 8.9e-08, four decades above the movement) and it does not reach the default
path, but it is the shape of thing a caller who raised the order and opted in would get.

### 4.2 D7's own fixture, rebuilt: the headline digits

D7's fixture by its method (N-BK7, f = 3 mm, 3.40 mm aperture, 0.60 mm beam, N = 512, dx = 8 um,
`ray_subsample=1`, the shipped `_DECENTRED_FIT_POLY_ORDER = 16`, the phase-only call shape), my own
estimator and oracle:

```
f_b = 0.003000000 m   stigmatic spread 5.20e-18 m
decentre 0.5 w :  chebyshev 2.371 urad   zernike 2.371 urad   |d| 2.724e-12 urad
decentre 1.0 w :  chebyshev 1.683 urad   zernike 1.683 urad   |d| 1.866e-11 urad
```

**2.371 / 1.683, character for character, on both bases.**

One caveat on the report's phrasing: `test_niche_d7_decentred_fit.py` does NOT assert those digits.  It
pins RATIOS to the on-axis figure, deliberately ("so the test does not encode this machine's exact float
noise").  So "d7 slow: 38 passed -- the default is still 2.371 / 1.683 urad" is a report of the engineer's
own measurement, not something the suite checks; the independent check is the run above.

### 4.3 The coupling, with the marginal-ray weight SWEPT

The report measures the marginal-ray -> defocus coupling at the shipped skirt weight.  The stronger form of
the audit's claim is a SENSITIVITY -- if the disc basis decoupled marginal rays from defocus, the Zernike
column would be flat in `w_out` where the Chebyshev one is not.  Swept `_FIT_DISC_OUTSIDE_WEIGHT_REL` over
x0 (disc-only), x0.25, x1 (shipped, 1e-8), x4, x16, reading the fitted map's `Z(2,0)` on the fit disc by
exact quadrature:

| total degree | coupling, chebyshev | coupling, zernike | ratio |
|---|---|---|---|
| 6 | 4.456e-11 m | 4.456e-11 m | **1.0000** |
| 10 | 6.422e-12 m | 6.422e-12 m | **1.0000** |
| 16 | 5.681e-12 m | 5.681e-12 m | **1.0000** |

and the per-weight `Z(2,0)` values agree between the bases to **1e-19 m** in all 15 cells.  The audit's
premise is refused not just at one weight but as a function of the weight.  This is the single most direct
test of the audit's sentence and it goes the engineer's way.

---

## 5. Where the two bases COULD legitimately differ

The brief asked me to hunt for places a basis change could legitimately move an answer.  Six candidates,
all measured:

**5.1 The rows -- D1's skirt weights.**  `fit_basis` reaches only the columns: `_decentred_fit_restriction`
is called identically on both paths and its return feeds `weights=` unchanged; `_Cheb2DEvaluator` scales
`A *= w_keep[:, None]` after the design is built, on both branches.  No difference, by construction, and
sec. 4.3 confirms it downstream.

**5.2 The C13 conditioning step-down -- the one real asymmetry.**  The step-down fires when the
equilibrated Gram screens below `_LSTSQ_GRAM_RCOND_MIN = 1e-8`, re-solves by QR, and returns the QR answer
only if it beats the normal-equations one by `_LSTSQ_RESID_MARGIN`.  Because the two bases screen
DIFFERENTLY, the step-down can fire on one and not the other.  Measured (decentred fit, the step-down
forced both ways, difference of the evaluated maps as a fraction of peak):

| order | basis | rcond | step-down fires? | `\|on - off\|` / peak |
|---|---|---|---|---|
| 6 | chebyshev | 2.405e-10 | yes | 5.532e-08 |
| 6 | zernike | 1.675e-02 | **no** | 0.000e+00 |
| 12 | chebyshev | 2.851e-11 | yes | 9.025e-07 |
| 12 | zernike | 3.571e-10 | yes | 1.436e-10 |
| 16 | chebyshev | 1.878e-11 | yes | 3.735e-06 |
| 16 | zernike | 6.386e-15 | yes | 6.080e-07 |
| 20 | chebyshev | 1.616e-11 | yes | 1.732e-06 |
| 20 | zernike | 0.000e+00 | yes | 1.011e-04 |

So the report's "C13 is a no-op below order 20" is true of `||b - A x||` (which is what it measured: the
ratio reads 1.000000) and NOT true of the returned coefficients -- the step-down moves the evaluated map by
5.5e-08 to 3.7e-06 of peak, and at order 6 it moves the Chebyshev answer while leaving the Zernike one
untouched.  That is the same residual-vs-magnitude blindness A26-6 records, one level down.  It does not
break anything: both candidates are the least-squares answer, the movement is 5 to 7 decades under the
fit's own error against the oracle (sec. 3), and the cross-basis agreement WITH the step-down live is
5.7e-13.  Recorded as a note, not a defect.

**5.3 The Newton inversion.**  Identical census on both bases at `ray_subsample=1`: 147 456 finite pixels,
0 NaN, total power `9.951235187409395e+03` on both to all 16 digits, zero convergence warnings.  The
iteration budget and tolerance are not basis-sensitive on this fixture.

**5.4 dtype paths.**  `complex64` input: both bases run and return `complex64`, differing by 3.7e-09 (the
output type's own eps is 1.2e-07, so the basis is below the field's resolution there).  `sag_dtype=float32`:
both run, return `complex128`, differ by 1.9e-12.  The evaluator itself is float64 on both branches
(`A_full` and `coeffs` are `np.float64` unconditionally), so there is no reduced-precision Zernike path to
diverge.

**5.5 The Newton pool worker path.**  `n_workers=3` against `n_workers=1` on the opt-in basis is
`np.array_equal` -- `max|d| = 0.000e+00`.  The mechanism is sound: `_cheb_fit_state` adds `basis` and
`disc` only on the opt-in basis (so the default payload is the nine keys it always was, which I confirmed),
`from_state` does no arithmetic, and `_ev_zernike` has one implementation on every backend, so there is no
second floating-point order for a worker to disagree with its parent about.

**5.6 The inverse-map build cache.**  `parity_tag` at `:12968` is `(newton_fit, fit_poly_order, weights is
None, MAX_NEWTON_ITERS)` and does NOT name `fit_basis`, even though its own docstring says a knob that
changes the incumbent must enter the key or "the returned field would depend on the ORDER the calls were
made in".  It is covered anyway, by the OTHER half of the key: `incumbent_fp` hashes what the incumbent
ANSWERS, and the two bases' incumbents differ in the last bits.  Measured -- a Chebyshev call followed by a
Zernike call with everything else equal:

```
after cheb: {'size': 1, 'capacity': 4, 'hits': 0, 'misses': 1}
after zern: {'size': 2, 'capacity': 4, 'hits': 0, 'misses': 2}
```

No collision.  I record it because the module's stated doctrine is that the key errs toward a cold rebuild,
and adding `str(fit_basis)` to the `parity_tag` tuple would cost nothing and would not depend on the
incumbent's last bits staying different; see sec. 12.

---

## 6. Byte identity of the default, archive to archive, with a positive control

Twenty-three configurations of my own, driven identically in both extractions by a probe that passes no new
keyword anywhere (so it imports and runs on the pre-change tree), `lumenairy.__file__` asserted inside each
tree, md5 of the returned field's bytes:

```
01_concentric  02_decentred_1w  03_decentred_pre_d7  04_decentred_order20  05_spline
06_inverse_map_off  07_fwd_pre_d7  08_fwd_order20  09_fwd_concentric
10_sub1_concentric  11_sub1_decentred  12_sub1_decentred_pre_d7
13_inversion_fit  14_ghost_ray_density  15_pool_workers  16_complex64
17_sag_float32  18_no_fit_radius  19_amp_ray_density_axis  20_prepared_screen
21_fine_fwd_order6  22_fine_fwd_order12  23_fine_fwd_shipped

MISMATCHES parent vs head: 0
distinct fields (head): 16 of 23
```

**And the battery's sensitivity, measured rather than asserted.**  The report says "a change that moved the
default fit by one ULP would show here".  I built two more trees to check:

| mutant | detected in |
|---|---|
| **A** one design-matrix entry nudged by 1 ULP (`A_full[0,0] = nextafter(...)`) | **1 of 23** (`12_sub1_decentred_pre_d7`) |
| **B** every fit COEFFICIENT nudged by exactly 1 ULP (`c_np = nextafter(c_np, inf)`) | **15 of 23**, including the default `01_concentric` |

So the report's claim is **verified under its natural reading** (the fit moved by one ULP) and its floor is
exactly there: a perturbation smaller than one ULP of the coefficients does not reliably show.  The eight
configurations mutant B does not reach are `02`, `03`, `04`, `05`, `13`, `16`, `19`, `20` -- the ones where
the inverse-characteristic model supplies the answer, plus the `complex64` one whose output type cannot
resolve it.  That is the report's own "four cases share an md5" bullet, quantified: **a third of a
byte-identity battery of this shape proves the imap path unchanged, not the fit.**  Any future battery
should carry the `sub1_*` / `fwd_*` cases for that reason, and the report is right to say so.

---

## 7. The configuration seam

| probe | result |
|---|---|
| `fit_basis='foo'` / `'Zernike'` / `None` | `ValueError: apply_real_lens_traced: fit_basis=... is not a design basis for the traced ray fits` -- sec. 2 prefix, names the setting, lists the vocabulary |
| `fit_basis='zernike', newton_fit='spline'` | raises, naming `newton_fit='polynomial'` |
| `fit_basis='zernike', inversion_method='fit'` / `'backward_trace'` | raises, naming `inversion_method='newton'` |
| `fit_basis='zernike'` with the DEFAULT `newton_fit='auto'` | **accepted** -- the gate runs after `'auto' -> 'polynomial'` at `:9049`, which is the right order and worth pinning |
| `LensNumerics(fit_basis='zernike')` | present, round-trips, `LensConfig.to_kwargs()` and `to_kwargs(entry_point='apply_real_lens_traced')` both emit it |
| `_NUMERICS_FOR` for both traced entry points | carries `'fit_basis': 'fit_basis'` |
| a16 census | `sum(len(fields(d)) for _, d, _ in _GROUPS)` = **40** |
| `numerics=LensNumerics(fit_basis='zernike')` vs the keyword | `np.array_equal`, `max|d| = 0.000e+00` |
| `prepare_real_lens_traced(fit_basis=...)` | accepted; its config path equals its keyword path; refuses `zernike` + `spline` through the same gate |
| the precedence-or-raise contract | **VACUOUS for this field.**  `resolve_entry_point_kwargs` skips any field equal to its default, so a conflict needs two distinct NON-default values; `fit_basis` has one non-default value, so the raise branch is unreachable.  The mechanism itself is sound -- driven through the 3-valued sibling `newton_fit` it raises `... was passed explicitly but numerics.newton_fit=... A keyword and a config field must agree` |
| the JAX traced twin | `apply_real_lens_traced_jax` is keyword-only with an explicit signature and no `**kwargs`, so `fit_basis=` raises `TypeError`.  It does not share this vocabulary at all (its fit order is `cheb_order`, not `newton_poly_order`), so refusal by signature is the right answer; recorded because the report does not mention it |
| `_lens_imap`'s exit fits | stay Chebyshev, deliberately (B10-D1).  `_decentred_fit_spectrum` also stays Chebyshev and reads `a + b` from `ev._mi` as the total degree -- correct only for a Chebyshev `(kx, ky)`; it constructs its own evaluator with no basis keyword, so it is unreachable, but it is a latent trap for anyone who later plumbs `fit_basis` into it |

---

## 8. The crossover law, attacked at other `R_data/R_disc` -- the finding against the report

The report derives the decay of the disc basis's advantage from a mechanism -- "a degree-`n` Zernike column
normalised to a disc of radius `R` grows as `(r/R)^n` outside it ... the skirt's share of the Gram
therefore rises by `(r/R)^2 = 16.1` per degree ... at a rate of about 1.2 decades per degree", measured 0.95
at its own `R_data/R_disc = 4.01` -- and B10-D3 then says the crossover order "is a property of
`R_data / R_disc` ... through the `(R_data/R_disc)^2`-per-degree decay measured in section 6", inviting a
future author to re-derive a crossover for another geometry from it.

**It does not extrapolate.**  Measured on D7's OWN launch lattice and D1's own weights, varying only the fit
disc radius, over orders 6 -> 10 (both far above the float64 conditioning floor on every arm):

| `R_disc` | `R_data/R_disc` | zern rcond o6 | o10 | measured dec/degree | `2 log10(ratio)` |
|---|---|---|---|---|---|
| 2.00 mm | 2.03 | 8.520e-01 | 1.503e-02 | 0.44 | 0.61 |
| 1.20 mm | 3.38 | 5.636e-01 | 9.399e-06 | **1.19** | 1.06 |
| 0.90 mm | 4.50 | 5.930e-02 | 1.041e-06 | **1.19** | 1.31 |
| 0.70 mm | 5.79 | 1.585e-02 | 3.031e-07 | 1.18 | 1.53 |
| 0.50 mm | 8.11 | 2.193e-03 | 6.269e-08 | **1.14** | 1.82 |
| 0.35 mm | 11.58 | 6.981e-04 | 2.468e-08 | 1.11 | 2.13 |

Reproduced independently on my own N-SF11 lattice at ratios 2.66 / 3.55 / 5.33 / 7.11 / 10.66: 0.95 / 1.22 /
1.16 / 1.15 / 1.10 decades per degree.

The decay **saturates at ~1.19 decades per degree from `R_data/R_disc ~ 3.4` upward, and then very slowly
DECLINES**, while the stated law doubles over the same span.  At the report's own ratio the two agree to
20 %, which is why the mechanism looked confirmed; one ratio cannot separate `2 log10(x)` from a constant.

What the ratio DOES move is the OFFSET: the order-6 advantage falls 5.636e-01 -> 6.981e-04, 2.9 decades,
over ratios 3.38 -> 11.58.  Since the Chebyshev arm is nearly flat (4.6e-07 -> 8.6e-11 over the same span at
order 6, and 2.2e5-class `cond(A)` at every order), the crossover ORDER moves through that offset and not
through the slope -- roughly 6 + `log10(zern_o6 / cheb_flat) / 1.19`, which on these six arms runs from
about 18 down to about 13.

**Consequence for the report.**  The conclusion (opt-in, default unmoved, crossover near the shipped order
on D7) survives, and section 6's headline number 0.95 is a fair measurement of its own fixture.  What must
not survive is B10-D3's invitation to compute another geometry's crossover from `(R_data/R_disc)^2`: it
over-predicts the slope by 1.9x at ratio 11.6 and would put the crossover several degrees too low.  I have
pinned the saturation in the b10 file (sec. 9.3) so the next author meets the measurement before the law.
Recommended report edit in sec. 12.

---

## 9. Defects found and fixed

All three are defects of COVERAGE or DOCUMENTATION.  I found no defect of behaviour: nothing I did produced
a wrong answer from the shipped code, and the default basis is byte-identical on 23 configurations.

### 9.1 V-1 -- the disc the entry point normalises the basis to was pinned nowhere (FIXED)

`_fit_basis_disc_or_raise` refuses to invent a disc on the grounds that "the whole content of the basis is
WHICH disc it is orthogonal on".  The entry point resolves that disc at `:11792`: the BEAM's
`(bcx, bcy, _beam_fit_radius)` on the off-centre branch, `(0, 0, _fit_r_max)` concentric, with the
lattice's circumscribing disc as the fallback at `:12012`.  **Nothing tested it.**

**Fail-before, measured.**  An in-memory mutation that forces the basis disc concentric on the off-centre
branch left **32 of 33** tests in the b10 file green -- the one red was
`test_the_default_basis_ships_the_state_it_always_shipped`, and only incidentally, because it asserts a
literal disc tuple it passed in itself.  Nothing in the accuracy, coupling, arbiter or conditioning
sections noticed, and they cannot: the two discs span the same space, so the fitted map moves by
**6.048e-13 of peak** while the equilibrated Gram rcond moves by **4.7 decades** (5.930e-02 beam-centred
against 1.082e-06 concentric, at the arbiter's own order on the D7 fixture).  A wrong disc is invisible to
every pin the package shipped and costs the entire benefit the package exists for.

**Fix**: `test_the_entry_point_normalises_the_basis_to_the_beams_own_disc`, which drives the ELEMENT with
`fit_basis='zernike'` at a decentre, reads the disc off the evaluators the element actually constructed
(spying on `_Cheb2DEvaluator.__init__`, so the pin is on what the fit was normalised to and not on a copy
of the resolution logic), and asserts (a) an applied fit normalised to the beam's centre within 0.1 of the
disc radius with a radius at or under `fit_radius_beam_factor * w`, (b) the C11 arbiter's concentric
candidate scored in ITS own disc, and (c) the fail-before: the wrong disc moves the map by less than 1e-6
of peak and costs more than 2 decades of rcond.  After the fix the same mutation reddens it.

### 9.2 V-2 -- where the basis reaches the returned field was documented nowhere (FIXED)

On the entry point's DEFAULT `ray_subsample=8`, `fit_basis='zernike'` changes the returned field by
**nothing at all**, and says so nowhere:

| configuration | imap engaged (cheb/zern) | `array_equal` | `max\|d\|` |
|---|---|---|---|
| `ray_subsample=8` (the default) | True / True | **True** | 0.000e+00 |
| `ray_subsample=4` | True / True | **True** | 0.000e+00 |
| `ray_subsample=2` | True / True | **True** | 0.000e+00 |
| `ray_subsample=2, inverse_map=False` | False / False | False | 2.208e-12 |
| `ray_subsample=1` | False / False | False | 2.241e-12 |

`prepare_real_lens_traced` at `ray_subsample=2` likewise returns an identical screen on both bases.  This is
correct behaviour -- it is fix D5 / `FIX_G8_PROBE`'s finding for the fit's ORDER, restated for its basis,
and the report states it for `ray_subsample=2` in one bullet of section 4.  But the `fit_basis` parameter
documentation did not say it, and the module's own doctrine -- quoted in this package's own gate comment,
"a knob that is silently inert is the defect the `on_fit_domain_basis` ledger exists to record" -- makes
that omission the one worth closing.  A caller who reads "what it buys is conditioning" and opts in on a
default call gets a bit-identical field and no indication why.

**Fix**: a `WHERE IT REACHES THE RETURNED FIELD` paragraph on the `fit_basis` parameter in
`apply_real_lens_traced`'s docstring, with the measurement, and
`test_the_basis_reaches_the_returned_field_only_where_the_fits_do`, which asserts both arms and forces the
precondition rather than hoping for it (S4): each arm reads the model's engagement out of the element's own
`_imap_out` record and makes its claim only if the gate landed where that arm needs it.  I did not add a
warning: the fits ARE expressed in the basis and the conditioning and the arbiter's residuals do move, so a
per-call warning would be false as often as it was useful.

### 9.3 V-3 -- the decay law was pinned only loosely enough to admit B10-D3's extrapolation (FIXED)

`test_the_disc_basis_advantage_decays_with_the_order_on_a_square_of_data` asserts three decades of decay
over eight degrees (0.375 dec/degree) at ONE disc radius.  Everything in sec. 8 passes it, including a
future tree in which the slope really did track `2 log10(ratio)`.

**Fix**: `test_the_conditioning_advantage_does_not_follow_the_stated_ratio_law` measures the slope at two
ratios that differ by 3.4x (3.38 and 11.58 on this file's own fixture) and asserts that the two slopes agree
to 0.3 relative (measured 0.07) where the law predicts they differ by 2.0x, and that the order-6 OFFSET
moves by more than 1.5 decades (measured 2.9) -- so the test states, in one place, both halves of what
actually sets the crossover.  It refuses to run on an arm that reached the conditioning floor, so it cannot
degenerate into measuring noise, and its docstring carries the six-ratio table and the date.

### 9.4 Not fixed, recorded

* `parity_tag` omits `fit_basis` (sec. 5.6) -- covered by `incumbent_fp`; a one-token hardening, requested
  in sec. 12 rather than taken, because the call site is shared with the `newton_fit` contract another
  verifier is working next to.
* `_decentred_fit_spectrum` reads `a + b` from `ev._mi` as a total degree (sec. 7) -- correct today,
  unreachable from `fit_basis`, and a trap for whoever lifts B10-D1.
* Above the shipped order the opt-in design is rank-deficient on the decentred branch (sec. 3).  The
  element already warns.  Recommended documentation, sec. 12.

---

## 10. Follow-up: could the disc basis ever replace the arbiter / predictor apparatus?

**No, and the reason is structural rather than numerical.**  I tried to break the report's argument here and
could not.  What niche C11 arbitrates is not a parametrisation, it is two different WEIGHTED PROBLEMS -- the
off-centre candidate keeps the beam's disc at weight 1 and the launch square at `w_out = 1e-8`, the
concentric candidate keeps a different disc under a different restriction, and the two even run at
different orders (`_decentred_fit_restriction` returns its own order for each).  No change of basis inside
one span can equalise two different row sets.  Measured: the two candidates' residuals are identical across
the bases to seven printed digits in 28 of 30 cells of sec. 4.1, and the gap the arbiter ranks is four
decades larger than anything the basis moves, at every order and every decentre I ran.  The same holds for
the C12 predictor, whose input is the OPL's own spectral tail on the launch box -- a property of the data,
not of any basis it is expressed in.

**What a disc basis could replace is a different apparatus: the fit-radius restriction itself, if the
LAUNCH LATTICE were a disc.**  That is A26-7 from the other side.  Today the fit's worst data sits at 4.5
disc radii because the lattice is the square circumscribing a disc-shaped domain, the skirt is what keeps
the map single-valued out there, and the disc basis is therefore orthogonal under a measure the data does
not have.  On a disc-shaped launch congruence the retained samples ARE the disc, and sec. 4.1's concentric
column is what that looks like: Gram rcond 1e-11-class -> 1e-1-class and the step-down firing 0 of 5
instead of 4 of 5, at every order below the crossover, with the answer unchanged.

**The measurements that would justify moving the default**, in the order I would take them:

1. **A disc launch congruence.**  Trace the launch lattice on a disc (a Vogel / sunflower set of the same
   count is the cheapest, and `pattern='vogel'` already exists in the raytrace package from WP-B9).  Then
   re-measure sec. 8's table: the prediction is that `R_data/R_disc -> 1` and the decay disappears
   altogether, so there is no crossover and the disc basis is uniformly better conditioned.  Until that
   exists, no amount of basis work can help the decentred branch, and the report is right that this is not
   a basis change.
2. **The crossover as a function of the ORDER the element actually ships**, on more than one design.  The
   quantity to decide on is `order_crossover - _DECENTRED_FIT_POLY_ORDER`: today it is about `+2` on D7 at
   `fit_radius_beam_factor = 1.5` and about `-3` at `0.6` (sec. 8).  A default change needs that margin
   positive with room on every design in the validation set, measured, not extrapolated from the law.
3. **The exit slope at the crossover order, on both bases, at several decentres** -- i.e. sec. 4.1 repeated
   on design 121's chain.  My table says the answer does not move; the point of repeating it is that a
   default change must show the answer not moving on the designs that matter, and 121's chain is
   local-only (`validation/repro_traced_carrier_121/`), so I could not run it.
4. **The rank margin.**  Above the crossover the opt-in design is rank-deficient in float64 (sec. 3); any
   default change must carry a bar on `sigma_min/sigma_max` of the WEIGHTED design at the shipped order,
   with decades between it and `eps`, not merely a Gram rcond above `1e-8`.
5. **A numba kernel for the Zernike evaluation** (B10-D2) before, not after, any default change -- 1.49x on
   a `ray_subsample=1` call is a cost the default cannot carry, and the kernel would reintroduce exactly
   the second floating-point order the pool's bit-identity contract is about, so it needs
   `_resolved_cheb_backend`'s treatment and its own byte-identity battery.

My recommendation: **ship as-is, opt-in, default unmoved.**  The work package is correctly scoped and its
negative result is the valuable part of it.

---

## 11. Everything I ran

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-p no:randomly`, one process at a
time, from the repository root.  The pytest runs are on the working tree (which carries other engineers'
in-progress edits in other subsystems); the measurement runs in sections 3-8 are on the `git archive`
extractions, in child processes, as section 2 describes.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b10_zernike_fit_basis.py` (with my three new tests) | **33 passed** | 12.3 s |
| `pytest tests/unit/test_niche_c11_decentred_fit_arbiter.py test_niche_c12_physics_fit_selection.py test_niche_c13_lstsq_conditioning.py test_niche_c6_fit_guard.py test_fix_d5_fit_domain_basis.py test_audit2609_a26_decentred_exit_reference.py` | **129 passed** | 102.0 s |
| `pytest tests/unit -k a16_` | **168 passed, 1 skipped** (PySide6) -- the report's two KNOWN REDs are green now that the orchestrator's `lens_config.py` edits are in | 27.9 s |
| `pytest tests/unit/test_audit2609_a17_history_lint.py tests/unit/test_audit2609_a17_history_relocation.py` | **744 passed** | 40.6 s |
| `pytest tests/unit/test_niche_d1_tilted_carrier.py` | **33 passed** | 71.8 s |
| `pytest tests/unit -k real_lens` | **160 passed, 3 skipped** (PySide6, host-specific W5 digests, numexpr) | 262.8 s |
| `pytest tests/unit/test_niche_d7_decentred_fit.py` (slow) | **38 passed** | 314.3 s |
| `pytest tests/unit/test_niche_d6_exact_tilted_leg.py` (slow) | **38 passed** | 175.3 s |
| `python validation/run_all.py test_lenses` | **ALL 1 files passed** | 25.2 s |
| `ruff check lumenairy/elements/_lens_traced.py tests/unit/test_audit2609_b10_zernike_fit_basis.py` | **All checks passed!** | 1 s |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** -- including `lumenairy.elements._lens_traced.md`, which my docstring edit provably did not move (both fingerprints drop docstrings, which is the whole point of the pin) | 5 s |
| `python scripts/check_doc_identifiers.py` | **OK: every API-claiming backticked identifier resolves** (598) | 6 s |

Measurement runs (child processes on the archive extractions, section 2):

| probe | what it produced |
|---|---|
| `v1_span.py` | sec. 3's rank / projector table, four lattices, orders 6..20 |
| `v2_fit.py` | sec. 3's fit-vs-QR-reference table and sec. 5.2's step-down table |
| `v3_oracles.py` | sec. 2's oracle cross-check |
| `v4_ladder.py` | sec. 4.1's ladder, 30 element calls at `ray_subsample=1` |
| `v5_law.py` | sec. 4.3's swept-weight coupling and a five-ratio decay scan |
| `v6_seams.py`, `v6b_config.py` | sec. 5.3-5.6 and sec. 7 |
| `probe_default.py` x3 trees | sec. 6's byte identity and its two mutants |
| `v7_mutate.py` | sec. 9's mutation battery (six mutations x 33 tests) |
| `v8_cost_disc.py` | sec. 4's cost medians and sec. 9.1's fail-before |
| `v9_reach.py` | sec. 9.2's reachability table and sec. 5.3's Newton census |
| `v10_proto.py` | sec. 8's six-ratio table on D7's lattice |
| `v11_d7.py` | sec. 4.2's `2.371 / 1.683` |

Mutation battery result, for the record (newly red of 33, after my additions):

```
M1 zernike evaluated on UNSCALED coordinates          -> 15
M2 basis disc forced CONCENTRIC off-centre            ->  3   (was 1 before sec. 9.1)
M3 D1's skirt weight dropped (hard mask)              ->  9
M4 design columns PERMUTED against the evaluation     ->  9
M5 orthonormal NORMALISATION dropped                  ->  4
M6 radial factor perturbed 1e-9 per term              ->  4
```

Every mutation is caught by more than one test, and the baseline is 0 of 33 red.

---

## 12. Requested changes outside my ownership

**12.1 `WP-B10_REPORT.md` section 6 and B10-D3 -- the decay law.**  Section 6's sentence "the basis stops
being orthogonal under the sample measure at a rate of about 1.2 decades per degree" and B10-D3's "through
the `(R_data/R_disc)^2`-per-degree decay measured in section 6" should be amended to say that the rate is
measured at ONE ratio and does not extrapolate: from `R_data/R_disc ~ 3.4` upward the measured decay
saturates at ~1.19 decades per degree and slowly declines (sec. 8's table), so the ratio moves the crossover
through the order-6 OFFSET, not through the slope.  The same sentence appears in `WP-B10_CHANGELOG.md`
("decays by about 1.2 decades per degree -- `(r/R)^2 = 16` per degree, one extra power of the skirt's reach
per shell"); the mechanism is a fair description of what happens at low order and should keep its "about",
but the changelog should not imply the rate scales with the ratio.  The pin is
`test_the_conditioning_advantage_does_not_follow_the_stated_ratio_law`.

**12.2 `WP-B10_REPORT.md` section 11 -- the d7 line.**  "the default is still 2.371 / 1.683 urad" reads as
though the suite checked those digits; it pins ratios to the on-axis figure by design.  Suggest "the suite
is green and the digits were re-measured at 2.371 / 1.683" (sec. 4.2 re-measured them independently and
they hold).

**12.3 `lumenairy/elements/_lens_traced.py:12968` -- `parity_tag`.**  Add the basis to the tuple so the
inverse-map cache key names the knob as well as hashing the incumbent's answers:

```python
-            parity_tag=(str(newton_fit), int(_fit_poly_order),
-                        _fit_weights is None, int(MAX_NEWTON_ITERS)),
+            parity_tag=(str(newton_fit), int(_fit_poly_order),
+                        _fit_weights is None, int(MAX_NEWTON_ITERS),
+                        str(_fit_basis)),
```

Not taken here because the call site sits inside the `newton_fit` / G8 contract another verifier is working
next to, and because it is a hardening rather than a fix: measured, the two bases do NOT collide today
(sec. 5.6), since `incumbent_fp` hashes what the incumbent answers and the two answers differ in the last
bits.  The module's own doctrine is that the key should err toward a cold rebuild, which this restores.

**12.4 `WP-B10_REPORT.md` section 13 / B10-D1 -- a sentence for `_decentred_fit_spectrum`.**  Record that
its degree bookkeeping reads `a + b` from `ev._mi`, which is the total degree of a Chebyshev `(kx, ky)` and
NOT of a Zernike `(n, m)`.  It is unreachable today; it is the first thing that breaks if the deferral is
lifted.

**12.5 `docs/lens_configuration.md`.**  No change requested; the field, the row and the count are already
correct on this tree and the a16 census reads 40.

---

## 13. Files I changed

| file | what |
|---|---|
| `lumenairy/elements/_lens_traced.py` | ONE docstring paragraph: `WHERE IT REACHES THE RETURNED FIELD` on the `fit_basis` parameter of `apply_real_lens_traced` (`:8532-8547`).  No code, no behaviour -- proved by `scripts/record_history_fingerprints.py --check`, which is OK without a re-record because both fingerprints drop docstrings |
| `tests/unit/test_audit2609_b10_zernike_fit_basis.py` | three tests added (sec. 9.1-9.3) and two module-level helpers (`_prescription`, `_collimated_beam`) they need; nothing existing weakened or removed.  30 -> 33 tests |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B10.md` | this report |
| `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B10_CHANGELOG.md` | release text for the docstring |

`docs/history/lumenairy.elements._lens_traced.md` is NOT re-recorded, deliberately: the edit is
documentation-only and the gate proves it.
