# VERIFY -- round 3 of the PURE staggered 2-D PMM per-layer grids (L2 mortar)

**Date** 2026-09-11 · **Worktree** `C:/tmp/lum_vmortar3`, branch
`verify/mortar-round3`, created from the fix branch tip `5f01eea` · **PRE-fix
tree** `C:/tmp/lum_vmortar3_pre`, a second worktree detached at `1957da2`
(round 3's branch point = round 2 plus its verification)

**Under verification**
`docs/audits/FIX_PMM2D_MORTAR_ROUND3_2026_09_11.md` -- the P1 DEFECT V1
repair (`_MORTAR_RESID_REFUSE`), the S5.4 degradation-band warning, the
restored v8 table, the bit-identity battery and the round-3 gate file.

**Method** every number in this report was RE-MEASURED in this worktree by
`validation/probe_verify_mortar_round3/` (eight probes plus three helpers),
against fixtures written here, on BOTH builds except where S12 says
otherwise.  Nothing is read from the fix doc, from its probes or from its
tests: where one of their numbers is quoted it is labelled as theirs and
stands beside mine.  The one thing REUSED is the fix's own diff, which is
what is being checked.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Verdict

| # | claim (fix's wording) | verdict | my numbers |
|---|---|---|---|
| a1 | the generalized mortar operand is rank-deficient BY CONSTRUCTION when a promoted in-plane region meets a non-promoted one, near-null participation 0.000 / 1.000 | **CONFIRMED** | **exactly 0.000 / 1.000 on all 16 distinct mixed operands** across 6 fixture classes, `s_min/s_max` 4.3089e-14 .. 1.4856e-10; the 7 controls read 0.353-0.829 / 0.559-0.935 at `s_min/s_max` 2.6298e-05 .. 8.1053e-04.  WIN and WSL agree on every mixed `s_min/s_max` to 1e-3 relative |
| a2 | ... "whenever **EITHER** side of the interface is promoted" (S0, S3.1, and the site's own comment) | **REFUTED as worded / RESTATED** | with **BOTH** sides promoted the operand is HEALTHY: `s_min/s_max` 2.63e-05 / 2.42e-08 / 4.29e-07 at `M` = 4/5/6 and participation 0.884/0.468 -- 5.3 decades above the one-promoted operand and within 1.5 decades of the both-out-of-plane control.  The mechanism needs an ASYMMETRIC interface |
| a3 | the system is CONSISTENT anyway (residual is ordinary backward stability) | **CONFIRMED** | mixed residuals 8.4532e-15 .. 5.8482e-14 (WIN) / 8.7454e-15 .. 5.3637e-14 (WSL), indistinguishable from the controls' 5.9919e-15 .. 2.4471e-14 |
| a4 | range membership predicts the cross-build spread "within a factor 3" | **BOUNDED, and NOT discriminating** | prediction `eps/rangeB` = 4.5e-14 .. 3.5e-13; my measured WIN/WSL relative spread of `R00` on the same solves is 5.7e-15 .. 6.0e-11, i.e. within 1.4x on some rungs and 30x out on others.  The SAME formula returns 9.7e-14 .. 2.3e-13 on the HEALTHY controls, whose measured spread is 7.2e-14 .. 4.0e-13 -- so agreement "within 3x" is not evidence for the mechanism |
| b1 | `rcond` is REFUTED as an instrument at this site (healthy and sliver populations cross) | **CONFIRMED, and sharper** | healthy `rcond` **5.8555e-15 .. 4.7002e-05** over 43 operands, so the round-2 bar 1e-12 sits three decades inside the healthy population: **18 of my 43 ordinary operands (42 %) would be REFUSED by it**, the same 18 on both builds.  My PRE-tree run reproduces four of those refusals through the public API |
| b2 | `_MORTAR_RESID_REFUSE` = 1e-6 has 6.9 decades below / 4.7 above | **CONFIRMED, with my own (tighter) margins** | 43 healthy operands: worst **4.5550e-13** (WIN) / 4.4802e-13 (WSL) = **6.34 decades below**, not 6.9 -- the fix's population has no wide-angle case.  10 broken operands: closest **7.799e-02** (WIN) / 1.3079e-01 (WSL) = **4.89 decades above**.  Two-sided, worst of both builds |
| b3 | the residual makes the RIGHT distinction (rank-deficient-but-consistent passes, inconsistent refused) | **CONFIRMED** | same operand: consistent RHS 9.66e-15 ACCEPTED, inconsistent RHS 1.65e+01 REFUSED, exactly singular 8.40e-02 (WIN) / 1.73e-01 (WSL) REFUSED, zero column NaN REFUSED -- and a zero ROW, which the fix did not test, NaN REFUSED |
| b4 | the residual is NOT a sliver backstop; the width contract is the only line | **CONFIRMED** | S5: with the contract lifted the answer moves while the residual stays in the healthy band |
| b5 | the exactly-singular operand is SYNTHETIC; no public-API geometry reaches it | **CONFIRMED** | no geometry found; and at `delta` <= 1e-6 my sliver ladder is refused by a DIFFERENT guard (`_guarded_solve`), not this one |
| c | v8 restored: per-layer `M` = 4..7 solve where round 2 refused | **CONFIRMED, and the answers CONVERGE** | four previously-refused stacks now answer (S6); and the per-layer arm converges to its mortar-free twin: gap 1.41e-2 -> 8.04e-5 over `M` = 4..7, 12x INSIDE the arm's own last step |
| d1 | the band `[1e-3, 3e-2)` warns, with switch, once per solve, naming width / class / remedies | **CONFIRMED** | 15/15 message tokens on both builds; silent at 6e-2, warns at 2.9e-2 and 1.2e-3, refused at 9e-4; switch off -> 0, on -> 1; FOUR banded layers -> ONE warning |
| d2 | `3e-2` carries "4.65x measured cost", 2x first exceeded at 2e-1 | **CONFIRMED in direction, BOUNDED in magnitude** | my independent fixture: the edge carries **13.5x** (`M` = 7) / **22.9x** (`M` = 8), and 2x is first exceeded at **2e-1** on both rungs -- same crossing, 3-5x steeper |
| d3 | "3.6x below the narrowest ordinary geometry (1.25e-1 / 1.09e-1), so NO ordinary stack warns" | **half CONFIRMED, half REFUTED** | 0 warnings on my 32-geometry census, on both builds -- but the narrowest ORDINARY geometry I reach is **5.000e-02** (a duty-0.9 pillar) = **1.67x** the edge, and a 16-cell uniform lattice is 6.25e-02 = 2.08x.  The 3.6x is a property of the round-2 battery, not of the library |
| d4 | closing tapers at 32/64 slices land in the band by design | **CONFIRMED, and it starts earlier** | on my own 0.25 .. 0.75 closing taper, 7.8125e-03 at 32 slices and 3.9062e-03 at 64 (theirs, on a slightly different bottom width, reads 8.0094e-03 / 4.1047e-03) -- and the surface enters the band from **9 slices**, not 16: its narrowest is `w_bottom / (2 n_slices)`, so 8 slices is 3.1250e-02, only **1.04x** outside the edge |
| e1 | bit-identity 165/165, 0 warning-set differences | **CONFIRMED on an independent battery, both builds** | 37 fixtures / 120 hashes: **33 identical, 0 warning-set differences on WIN and again on WSL**; the only 4 that move are the previously-REFUSED mixed stacks, from a refusal to an answer, and their four `rcond` readings are identical across builds.  The harness is shown to reproduce its own hashes (37/37) before the comparison is believed |
| e2 | 7 suites WIN 185 / WSL 185; whole surface 367; the new warning fires on exactly one gate | **CONFIRMED** | WIN 185 passed / WSL 185 passed; surface **367 passed, 24 warnings**; `grep -c "degradation band"` over the surface log = **1**, on `test_the_mortars_own_algebra_is_exact_at_every_wall_separation` |
| f | cost: the probe is "0-6 % over the existing gecon screen" | **CONFIRMED to within the box's noise** | `residual / gecon` = 0.897 .. 1.101 (WIN) / 0.936 .. 1.213 (WSL) on five widths, against a `bare / gecon` noise floor of 0.834 .. 1.170 -- the overhead is BELOW what a shared box can resolve, and the FLOP argument (1.7/n) is the load-bearing one.  The EXACT residual costs 1.49 .. 2.10x the gecon screen, so skipping it is worth 49-110 % |

**Two DEFECTS found**, both P3, neither ship-blocking: see S8.

**Ship recommendation: SHIP in 5.45.0** (S9).

---

## S1. Terms, defined before they are used

**The generalized mortar site.**
`lumenairy/elements/pmm/_core.py::_interface_smatrix_general_mortar_2d`.  Its
operand is the square block

```
    A = [[E1, E2], [H1, H2]]      first block COLUMN  = side a's BACKWARD set
                                  second block COLUMN = side b's FORWARD set
```

An out-of-plane tensor layer or a slanted layer anywhere in a per-layer stack
puts the WHOLE cascade on this form (`any_oop` is a stack-level flag), so
every other layer and both half-spaces are PROMOTED.

**Promotion.**  `twod_staggered.py::_modes_as_general` writes a symmetric
in-plane region's modes as `(W, V, lam, W, -V, -lam)`.  Throughout this report
a side is called PROMOTED when its 6-tuple satisfies that identity BITWISE --
an exact test, not a tolerance.

**One-promoted vs both-promoted.**  An interface is ASYMMETRIC when exactly
one of its two sides is promoted (out-of-plane or slanted beside in-plane) and
SYMMETRIC when both are (two in-plane layers on different grids, with the
out-of-plane layer elsewhere in the stack).  S3 shows the distinction is the
whole mechanism.

**Near-null participation.**  Write `v` for the smallest right singular vector
of `A` and split it at the block-column boundary `m_a`.  `on_a = ||v[:m_a]||`
and `on_b = ||v[m_a:]||` say which side's modes the near-null combination
lives in.  `m_a` is read from the interface wrapper, not assumed to be `n/2`.

**Range membership.**  `||u_min^H B|| / ||B||`, `u_min` the smallest LEFT
singular vector: how much of the right-hand side lies along the direction the
operator almost cannot reach.

**The residual.**  `||A X - B|| / ||B||`, EXACT (Frobenius) unless the text
says PROBE.  Throughout, `X` is the answer the SHIPPED pair `lu_factor` +
`lu_solve` returns, not `np.linalg.solve`'s: on an exactly singular operand
`gesv` raises `LinAlgError` while `getrf` warns and returns a factor whose
answer is not finite, and the second is what the shipped code path produces.
(Measuring it the other way cost this verification one probe run.)

**The band.**  `[_STAG_MIN_SEG_FRAC, _STAG_SLIVER_BAND_FRAC)` = `[1e-3, 3e-2)`
of the period, both edges carrying 1e-9 relative slack.  Below it a stack is
REFUSED; inside it a stack WARNS; above it it is silent.

---

## S2. The two builds

| | Windows (WIN) | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 (scipy-openblas) | 1.17.1 (scipy-openblas) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1`, set in the SHELL | same |
| `lumenairy` | `C:\tmp\lum_vmortar3\lumenairy\__init__.py` 5.44.0 | `/mnt/c/tmp/lum_vmortar3/...` 5.44.0 |

Both link scipy-openblas, so every cross-build spread here is a LOWER bound --
the same caveat the build, both fixes and both verifications carry.

**A trap this verification hit, and it is worth recording because it would
have produced a whole run of authoritative-looking wrong numbers.**  A probe
run as `python validation/probe_.../x.py` gets the SCRIPT's directory as
`sys.path[0]`, not the working directory, so `import lumenairy` silently
resolved to a checkout installed on `D:` -- a DIFFERENT library, several
versions old, without `_guarded_mortar_solve` at all.  Every probe here now
imports `_path.py` first, which puts this worktree at the front of `sys.path`
and REFUSES to run if `lumenairy.__file__` resolves anywhere else; the
resolved path is recorded in every JSON.

---

## S3. Claim (a) -- the mechanism, on my own fixtures

`v1_mechanism.py`, 34 rows, 13 fixture/`M`/`n_orders` combinations, both
builds.  Period 0.87 um, wavelength 0.73 um, `theta` = 0.17, `phi` = 1.1,
host 1.96, pillar 8.41, walls (0.1873, 0.5412) and (0.3106, 0.8039) -- every
one of those different from the fix's fixture.

### 3.1 One promoted side: CONFIRMED, on six fixture classes

`n_orders` = 2 unless stated.  `p_a/p_b` is the promotion pair.

| fixture | `M` | `n` | `p_a/p_b` | `s_min/s_max` | `rcond` | residual | on_a / on_b |
|---|---|---|---|---|---|---|---|
| out-of-plane + UNIFORM SPACER | 4 | 324 | F/T | 1.409e-11 | 3.525e-12 | 1.051e-14 | **0.000 / 1.000** |
| | 5 | 576 | F/T | 3.679e-13 | 8.853e-14 | 2.414e-14 | **0.000 / 1.000** |
| | 6 | 900 | F/T | 4.309e-14 | 6.870e-15 | 3.429e-14 | **0.000 / 1.000** |
| | 7 | 1296 | F/T | 5.417e-14 | 8.597e-15 | 5.848e-14 | **0.000 / 1.000** |
| out-of-plane + in-plane PATTERNED | 4 | 324 | F/T | 1.356e-10 | 6.573e-12 | 8.598e-15 | **0.000 / 1.000** |
| | 6 | 900 | F/T | 2.068e-12 | 1.062e-13 | 3.504e-14 | **0.000 / 1.000** |
| out-of-plane + in-plane MAGNETIC | 4 | 324 | F/T | 7.826e-11 | 5.502e-12 | 8.453e-15 | **0.000 / 1.000** |
| SLANTED + uniform spacer | 4 | 324 | F/T | 4.779e-11 | 4.201e-12 | 9.480e-15 | **0.000 / 1.000** |
| | 6 | 900 | F/T | 1.970e-11 | 8.034e-13 | 2.921e-14 | **0.000 / 1.000** |
| STRONGLY out-of-plane + in-plane | 4 | 324 | F/T | 1.486e-10 | 7.607e-12 | 9.064e-15 | **0.000 / 1.000** |
| slant 1e-8 + uniform spacer | 4 | 324 | F/T | 4.260e-11 | 3.679e-12 | 8.591e-15 | **0.000 / 1.000** |
| **both out-of-plane** (control) | 4 | 324 | F/F | 8.105e-04 | 4.700e-05 | 5.992e-15 | 0.829 / 0.559 |
| | 6 | 900 | F/F | 2.630e-05 | 1.781e-06 | 2.447e-14 | 0.759 / 0.651 |
| **both slanted** (control) | 4 | 324 | F/F | 5.221e-04 | 3.260e-05 | 7.204e-15 | 0.786 / 0.618 |
| **both in-plane** (control) | 4 | 162 | -- | -- | -- | -- | reaches NO generalized site at all: two IN-PLANE mortar sites |

Every `s_min/s_max`, every `rcond` and every participation figure above is
**IDENTICAL on WIN and WSL to all four digits printed** (and every mixed
`s_min/s_max` agrees between the builds to 1e-3 relative); the residuals agree
to two digits.  Over the FULL set -- 16 distinct mixed operands across those
six classes -- the participation is **exactly 0.000 / 1.000 on every one**,
`s_min/s_max` runs 4.3089e-14 .. 1.4856e-10 and the residual 8.4532e-15 ..
5.8482e-14, while the 7 NEITHER-promoted controls run 0.353-0.829 / 0.559-0.935
at `s_min/s_max` 2.6298e-05 .. 8.1053e-04.  I could not refute the claim: no
mixed fixture gave anything but 0.000 / 1.000, and no control gave a localised
near-null.

**Two adversarial controls that could have refuted it, and did not.**

* **an OUT-OF-PLANE tensor that is in-plane to 1e-9** (`e_xz` = 3.10e-9, just
  above the 1e-12 relative floor of `_tile_needs_oop`, so it runs the `4 q^2`
  generator rather than being promoted).  It is physically the same
  near-symmetric region a promotion writes down -- and it is HEALTHY:
  `s_min/s_max` 6.806e-04 / 4.773e-05 at `M` = 4/5, participation 0.571/0.821
  and 0.353/0.935.  So the mechanism is the EXACT algebraic promotion, not
  near-symmetry of the physics.  This SHARPENS the fix's claim.
* **a slant of 1e-8 rad** beside a promoted spacer: still 0.000 / 1.000, so
  an arbitrarily small slant is enough to make the interface asymmetric.

### 3.2 Both promoted: the claim as WORDED is REFUTED

The fix says "whenever EITHER side of the interface is an in-plane region
promoted" (S0), and the site's own comment says "when either side is an
in-plane region promoted by `_modes_as_general`".  A three-layer stack --
out-of-plane on grid A, in-plane on grid B, in-plane on grid D -- builds TWO
generalized mortars, and the second has BOTH sides promoted:

| interface | `M` | `p_a/p_b` | `s_min/s_max` | `rcond` | residual | on_a / on_b |
|---|---|---|---|---|---|---|
| asymmetric (0) | 4 | F/T | 1.356e-10 | 6.573e-12 | 8.598e-15 | 0.000 / 1.000 |
| **symmetric (1)** | 4 | **T/T** | **2.631e-05** | 3.477e-06 | 3.358e-15 | **0.884 / 0.468** |
| asymmetric (0) | 5 | F/T | 8.371e-12 | 3.665e-13 | 2.087e-14 | 0.000 / 1.000 |
| **symmetric (1)** | 5 | **T/T** | **2.416e-08** | 2.973e-09 | 2.218e-14 | **0.914 / 0.406** |
| asymmetric (0) | 6 | F/T | 2.068e-12 | 1.062e-13 | 3.504e-14 | 0.000 / 1.000 |
| **symmetric (1)** | 6 | **T/T** | **4.295e-07** | 3.846e-08 | 1.160e-14 | **0.849 / 0.528** |

Identical on both builds.  A both-promoted interface sits **5.3 decades
above** the one-promoted one and **within 1.5 decades** of the
both-out-of-plane control, and its near-null direction is not localised.  The
correct statement is **"when EXACTLY ONE side is promoted"** -- the near-null
space is a property of the ASYMMETRY, not of promotion as such.

Nothing SHIPPED depends on this: the residual screen accepts all three
classes and the fix's own healthy population already included a both-promoted
middle interface (its S4 bullet list).  It is a documentation defect, because
a later reader deriving a bar from "any promoted side" would mis-predict by
five decades.  Gate `test_a_generalized_mortar_with_both_sides_promoted_is_not_rank_deficient`
in `tests/unit/test_verify_pmm2d_mortar_round3.py` pins the restatement.

### 3.3 The build-stability prediction is BOUNDED, and it does not discriminate

The fix predicts a relative cross-build agreement of `eps ||B|| / ||u_min^H B||`
and reports agreement "within a factor 3" against the round-2 verifier's
measurements.  Re-measured here, the prediction and my own WIN-vs-WSL relative
spread of `R00` (the gauge-invariant quantity; the mortar answer `X` itself is
only defined up to each build's eigenvector phases, so comparing `X` directly
is meaningless and reads O(1) even on healthy controls):

| fixture | `M` | `rangeB` | predicted | measured WIN/WSL `R00` spread |
|---|---|---|---|---|
| spacer | 4 | 6.32e-04 | 3.5e-13 | 3.8e-12 |
| spacer | 5 | 2.99e-04 | 7.4e-13 | 9.5e-12 |
| spacer | 6 | 1.05e-04 | 2.1e-12 | 6.0e-11 |
| spacer | 7 | 1.37e-04 | 1.6e-12 | 3.5e-11 |
| pattern | 4 | 3.19e-03 | 7.0e-14 | 3.6e-13 |
| pattern | 6 | 1.15e-03 | 1.9e-13 | 2.7e-13 |
| both out-of-plane (control) | 4 | 2.29e-03 | 9.7e-14 | 1.9e-13 |
| both out-of-plane (control) | 6 | 9.72e-04 | 2.3e-13 | 4.0e-13 |

So the prediction is right to within 1.4x on some rungs and low by 30x on
others -- an order-of-magnitude estimate, not a 3x law.  More importantly it
is **not discriminating**: it returns the same 1e-13 for the HEALTHY control,
whose measured spread is also 1e-13, because `rangeB` is ~1e-3 in both
populations.  What the formula DOES do, and this part stands, is beat the
naive condition bound: `eps / (s_min/s_max)` would predict 1.6e-05 for the
`M` = 4 spacer, which is nine decades wrong, while the range-membership form
is within a decade.  **BOUNDED**: read S3.2 of the fix as "the naive
conditioning bound does not apply and the range-membership one is the right
shape", not as a quantitative 3x agreement.

---

## S4. Claim (b) -- the residual bar, re-derived two-sidedly

`v2_populations.py`, run in three parts so that a failure in one cannot
destroy the others.

### 4.1 HEALTHY -- 43 operands from 35 ORDINARY per-layer stacks

Nothing in this population has a segment an ordinary geometry would not ask
for: the narrowest over all thirteen wall arrays is **0.0900 of the period**
(the wide-angle fixture's second layer) -- **3.0x** the band's upper edge and
**90x** the width contract -- and every one of the 43 solves is SILENT.  The stacks span six DEVICE variants -- period 0.52 / 0.87 / 0.95 /
1.10 / 1.40 um, wavelength 0.41 .. 1.55 um, incidence 0.0 .. 0.95 rad, pillar
permittivity 4.0 .. 30.0, four wall pairs -- crossed with seven stack shapes
(out-of-plane + uniform spacer, out-of-plane + in-plane patterned,
out-of-plane + in-plane MAGNETIC, both out-of-plane, both slanted, slanted +
spacer, and a SIX-layer stack on three grids that contributes five operands),
at `M` = 4..8 and `n_orders` = 1..3.

| | WIN (py3.14) | WSL (py3.12) |
|---|---|---|
| operands / distinct stacks | 43 / 35 | 43 / 35 |
| `n` | 324 .. 1764 | 324 .. 1764 |
| **residual, EXACT** | **3.4131e-15 .. 4.5550e-13** | **3.4149e-15 .. 4.4802e-13** |
| residual, PROBE | 4.3919e-15 .. 4.6538e-13 | 5.2926e-15 .. 4.6965e-13 |
| probe / exact | 0.680 .. 1.302 | 0.900 .. 1.550 |
| `rcond` | **5.8555e-15 .. 4.7002e-05** | 5.8555e-15 .. 4.7002e-05 |
| accepted by the shipped bar | 43 / 43 | 43 / 43 |

Two readings matter.

**The `rcond` refutation is reproduced, and it is sharper than the fix says.**
The healthy `rcond` population reaches down to **5.8555e-15**, so the in-plane
pair's 1e-12 bar sits three decades INSIDE it.  Counted directly: **18 of the
43 ordinary operands -- 42 % -- read below 1e-12 and would be REFUSED by the
round-2 screen**, and it is the SAME 18 on both builds, at `rcond` 5.8555e-15
.. 8.0465e-13.  Every one of them is an ASYMMETRIC interface (8 uniform
spacer, 7 in-plane patterned, 2 magnetic, 1 from the six-layer stack); no
both-out-of-plane or both-slanted operand is among them, which is the
mechanism of S3 showing through the population.  They spread over `M` = 4 (1),
5 (8), 6 (5), 7 (3) and 8 (1), so the defect is not confined to one modal
count.  There is no fixed `rcond` bar with a gap on both sides at this site.

**The residual bar's LOWER margin is 6.3 decades, not 6.9.**  My worst healthy
reading is 4.5550e-13 (WIN) / 4.4802e-13 (WSL), against the fix's 1.19e-13 --
because the fix's population contains no WIDE-ANGLE case, and the worst
reading on both my builds is `wide_angle/pattern` at `M` = 7, `theta` = 0.95
rad.  `log10(1e-6 / 4.555e-13)` = **6.34 decades (WIN) / 6.35 (WSL)**.  That is
still an enormous margin and it does NOT walk with `M` (the `M` = 8 rung is
not the worst), but the fix's "6.9" is its population's number, not the
family's.

**The closest healthy approach to the bar** is therefore 4.5550e-13, a factor
**2.2e+06** under it.


### 4.2 BROKEN, on operands of this site's own shape

Built FROM a real operand of the site (`mix_pattern`, `M` = 4, `n` = 324) and
residuated through the SHIPPED `lu_factor` + `lu_solve` pair:

| operand | residual (exact) WIN | probe WIN | `rcond` | shipped decision |
|---|---|---|---|---|
| rank-deficient but CONSISTENT (the V1 shape) | **9.656e-15** | 1.334e-14 | 1.01e-19 | **ACCEPTED** |
| EXACTLY singular, the site's real RHS | **8.397e-02** | 6.656e-02 | 1.01e-19 | REFUSED |
| INCONSISTENT RHS | **1.646e+01** | 1.211e+01 | 1.01e-19 | REFUSED |
| ZERO COLUMN | NaN | NaN | 0.0 | REFUSED |
| ZERO ROW (not in the fix's set) | NaN | NaN | 0.0 | REFUSED |
| near-repeated column, separation 1e-4 | 8.559e-02 | 9.113e-02 | 1.16e-19 | REFUSED |
| ... 1e-8 | 7.799e-02 | 8.135e-02 | 1.29e-19 | REFUSED |
| ... 1e-12 | 8.228e-02 | 1.459e-01 | 1.31e-19 | REFUSED |
| ... 1e-14 | 8.194e-02 | 9.808e-02 | 1.33e-19 | REFUSED |
| ... 1e-16 | 8.397e-02 | 6.656e-02 | 1.01e-19 | REFUSED |

The **closest broken reading from above is 7.799e-02**, i.e. the 1e-6 bar sits
**4.9 decades below** it.  The rank-deficient-but-consistent operand -- the
same matrix, a different right-hand side -- passes at 9.66e-15, which is the
distinction the whole round is about, reproduced independently.

The near-repeated-column ladder is new here and it matters for the
false-refusal question: a column pair only 1e-4 apart already residuates at
8.6e-02, so the screen refuses it -- and it is right to, because the answer
really is 8.6 % wrong.  There is no separation at which the residual sits
between the healthy band and the bar.

**The probe as an estimator on the WRONG side.**  The fix measured
probe/exact only on real (healthy) operands, where it reports 1.6x.  On the
broken population the probe runs **0.74x .. 1.77x** of the exact residual --
it can UNDER-read by 26 %.  That is harmless here (0.74 x 7.8e-2 is still 4.7
decades above the bar) and the design is safe by construction in the other
direction (an over-reading probe only triggers the exact re-measurement), but
it should be stated: the probe is a factor-2 estimator, not a factor-1.6 one.

### 4.3 The FAIL-BEFORE arm, reproduced against the PRE-fix tree

The bit-identity battery (S7) runs the SAME four mixed stacks against
`C:/tmp/lum_vmortar3_pre`.  They raise there and answer here:

```
MIXED_spacer_M5   pre: _ConditioningError ... reciprocal 1-condition 8.853e-14 against a 1e-12 bar
MIXED_spacer_M6   pre: _ConditioningError ... 6.870e-15
MIXED_pattern_M5  pre: _ConditioningError ... 3.263e-13
MIXED_pattern_M6  pre: _ConditioningError ... 1.443e-13
```

DEFECT V1 is therefore reproduced independently, on my geometry, at the
commit the fix branched from -- and the four `rcond` values match the ones my
mechanism probe measures on the same operands to every digit.

---

## S5. Claim (b4) -- the residual is not a sliver backstop

`v2_populations.py sliver`.  An out-of-plane patterned layer whose grid
carries two EXTRA walls a relative distance `delta` apart, both INSIDE the
pillar region, so the permittivity map is byte-identical at every `delta` and
the DEVICE cannot depend on the wall separation; the neighbour is a uniform
spacer on its own grid, so the interface is a generalized mortar.  The width
contract is LIFTED (`PMM2D_STAG_MIN_SEG_GUARD = False`) for the measuring arm,
and the SHIPPED library is asked separately what it would have done.

`M` = 5, WIN.  "shipped" is what the library does with the contract ARMED;
"moved" is the relative movement of `R00` from the widest (ordinary) rung.

| `delta` | residual (exact) | probe | `rcond` | `R00` moved | shipped | band warning |
|---|---|---|---|---|---|---|
| 1.0e-01 | 3.6744e-14 | 3.7502e-14 | 1.668e-14 | -- | accepted | no |
| 5.0e-02 | 4.3628e-14 | 4.4052e-14 | 9.374e-15 | 7.08e-06 | accepted | no |
| 3.0e-02 | 4.8969e-13 | 5.3863e-13 | **3.634e-16** | 4.71e-06 | accepted | no |
| 1.0e-02 | 3.7524e-14 | 3.8716e-14 | 8.113e-15 | 1.89e-05 | accepted | **yes (x)** |
| 3.0e-03 | 5.8322e-14 | 4.9562e-14 | 2.349e-15 | 3.27e-05 | accepted | **yes (x)** |
| 1.05e-03 | 3.5491e-13 | 3.0973e-13 | 2.336e-16 | 6.68e-06 | accepted | **yes (x)** |
| 1.0e-03 | 2.5141e-13 | 2.0399e-13 | 2.937e-16 | 2.16e-06 | accepted | yes (x) |
| 3.0e-04 | 8.0108e-14 | 7.1510e-14 | 1.122e-15 | 3.12e-05 | **REFUSED** (`ValueError`, the width contract) | -- |
| 1.0e-04 | 6.6980e-14 | 6.3010e-14 | 1.373e-15 | 3.46e-05 | **REFUSED** | -- |
| 1.0e-05 | 6.8076e-14 | 7.8181e-14 | 1.516e-15 | 3.65e-05 | **REFUSED** | -- |
| 1.0e-06 .. 1e-9 | -- | -- | -- | -- | **REFUSED** even with the width contract lifted, by `_guarded_solve` at the `rcwa generalized interface (T22)` site | -- |

The residual runs **3.55e-14 .. 4.90e-13** over the whole ladder -- the SAME
decade as the healthy population's 3.41e-15 .. 4.56e-13 -- while the answer,
on a device whose permittivity map is byte-identical at every `delta`, walks
monotonically to **3.65e-05** of itself.  Six decades of wall separation are
invisible to the residual.

The `rcond` column is the other half of claim (b1) and it lands: the sliver
population reaches **3.63e-16** while the healthy population's floor is
5.86e-15 and its ceiling 4.70e-05 -- so the two populations **overlap**, and no
fixed `rcond` bar separates them.  (The overlap here is 1.4 decades on my
fixtures against the fix's nine; the direction and the conclusion are the
same.)


**CONFIRMED.**  The residual stays inside the healthy band at every `delta`
the guard would refuse, while the answer moves.  The residual is not a
backstop for the width contract, exactly as the fix states, and my numbers say
so on a fixture built here.

**One reading the fix does not have.**  On MY fixture the ladder stops before
`delta` = 1e-6: below that the solve is refused by a DIFFERENT guard --
`_guarded_solve`, the non-mortar site -- not by the mortar screen.  So the
statement "no public-API geometry drives the generalized site to exact
singularity" (the fix's S10.4) holds here for a second reason as well: the
region solve gives out first.


---

## S6. Claim (c) -- the restored answers, and whether they are RIGHT

Accepting is only correct if the answers are correct, and a single modal
count cannot say so: the per-layer arm and the common-refinement (union) arm
are different discretisations of the same device.  `v7_convergence.py` runs
the ladder on both.  The union arm is CONFORMING, so it takes the plain square
modal match at every interface and builds NO mortar -- a numerically
independent path to the same device.

| device | per-layer `R00` at `M` = 4 / 5 / 6 / 7 | mortar-free twin | gap to the twin, `M` = 4 -> 7 | the arm's own last step |
|---|---|---|---|---|
| out-of-plane + in-plane patterned | 0.003821212 / 0.020272701 / 0.016839265 / 0.017820258 | 0.017900684 (`M` = 5, union grid) | 1.408e-02 -> 2.372e-03 -> 1.061e-03 -> **8.043e-05** | 9.810e-04 |
| out-of-plane + uniform spacer | 0.011033295 / 0.011030019 / 0.011035833 / 0.011036269 | 0.011039685 (`M` = 7, conforming) | 6.390e-06 -> 9.666e-06 -> 3.852e-06 -> **3.416e-06** | 4.363e-07 |
| SLANTED + uniform spacer | 0.017193228 / 0.015930308 / 0.015990792 | 0.015987868 (`M` = 6) | 1.205e-03 -> 5.756e-05 -> **2.924e-06** | 6.048e-05 |

**WSL reproduces every gap and every step in that table to four significant
figures** -- these are discretisation quantities, not build-sensitive ones.

The first and third arms converge onto the mortar-free twin by 175x and 412x
across the ladder, and at the last rung the gap is 12x and 21x INSIDE the
per-layer arm's own step.  The spacer arm's gap is flat at 3e-06 absolute on
an `R00` of 0.011 (3e-04 relative) and does not grow.  **The answers the
residual screen now accepts are right**, on three independent devices.

---

## S7. Claim (e1) -- bit-identity, on an independent battery

`v6_bitid.py`, 37 fixtures, **120 sha256 hashes** over `dtype|shape|tobytes`
of every returned array, plus each fixture's WARNING SET.  The battery is
written here, not borrowed: shared-grid scalar / `jones` / `M`=6 `N`=3 /
uniform tensor / out-of-plane / out-of-plane at NORMAL incidence / magnetic /
slanted / `retain_internal` + `layer_absorption` / per-order amplitudes /
lossy metal; per-layer conforming / non-conforming / non-conforming `M`=6 /
nested / union / per-layer `n_modes` / uniform spacer / uniform lattice /
magnetic / slant both / slant conforming / out-of-plane both / out-of-plane
conforming; `add_tapered_pillar` at 4, 8 (both rules) and 16 slices;
`add_tapered_pillars`; two 1-D `PMMStack` arms; and the six MIXED stacks.

```
WIN  post vs pre: 33 identical, 4 differing (of 37 fixtures, 120 hashes)
WSL  post vs pre: 33 identical, 4 differing (of 37 fixtures, 120 hashes)
  MIXED_spacer_M5   ERROR CHANGED: pre=_ConditioningError (rcond 8.853e-14) post=None
  MIXED_spacer_M6   ERROR CHANGED: pre=_ConditioningError (rcond 6.870e-15) post=None
  MIXED_pattern_M5  ERROR CHANGED: pre=_ConditioningError (rcond 3.263e-13) post=None
  MIXED_pattern_M6  ERROR CHANGED: pre=_ConditioningError (rcond 1.443e-13) post=None
warning sets: identical on all 37, on both builds
```

The four `rcond` readings above are identical on WIN and WSL to every digit --
the refused solves are not a build accident.

**Nothing else moved a bit, and nothing gained or lost a warning**, and the
four that moved moved only from a refusal to an answer.  CONFIRMED.

**A harness self-check, which caught a bug that reads exactly like a library
regression.**  The first run reported `shared_per_order_amplitudes` as a HASH
DIFF.  It was mine: `per_order_amplitudes()` returns a DICT, and
`np.asarray(dict)` is a 0-d OBJECT array whose `tobytes` is a POINTER, so it
hashes differently in every process.  `v6_bitid.py selfcheck` now runs the
whole battery TWICE in the SAME tree and reports **37/37 fixtures reproduce
their own hashes** before any cross-tree comparison is believed.  A
bit-identity battery that has not been shown to reproduce itself proves
nothing, and this one had one fixture that did not.

---

## S8. DEFECTS

### DEFECT 1 (P3, documentation) -- "either side promoted" over-states the mechanism

**Where** `docs/audits/FIX_PMM2D_MORTAR_ROUND3_2026_09_11.md` S0 and S3.1;
`lumenairy/elements/pmm/_core.py` `_MORTAR_RESID_REFUSE`'s docstring ("whenever
one side of the interface is an IN-PLANE region promoted"); the comment inside
`_interface_smatrix_general_mortar_2d` ("when either side is an in-plane
region promoted"); and the refusal MESSAGE, which repeats it to the user.

**What is wrong** the near-null space needs the interface to be ASYMMETRIC.
With BOTH sides promoted the operand is healthy (S3.2): `s_min/s_max`
2.631e-05 at `M` = 4 against the asymmetric interface's 1.356e-10 and the
both-out-of-plane control's 8.105e-04.

**Severity** P3.  Nothing shipped behaves differently -- the residual screen
accepts all three classes and the fix's healthy population already contained a
both-promoted interface.  The cost is to the next reader.

**Reproducer**
`tests/unit/test_verify_pmm2d_mortar_round3.py::test_a_generalized_mortar_with_both_sides_promoted_is_not_rank_deficient`,
and `validation/probe_verify_mortar_round3/v1_mechanism.py` rows
`both_promoted/ifc1`.

**Remedy** one word in three places: "whenever EXACTLY ONE side of the
interface is an in-plane region promoted".

> **CLOSED 2026-09-11 by ROUND 4**
> (`docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md` S3).  The wording is
> corrected in the constant, in `_guarded_mortar_solve`'s docstring, in the
> site comment and in the refusal MESSAGE, and in the round-3 fix doc and the
> CHANGELOG as dated CORRECTION notes.  The population behind it was re-made
> over 33 operands on both builds: one-promoted spread 1.3735e-08 ..
> 2.6587e-07, both-promoted 0.4442 .. 0.6218, neither-promoted 0.3777 ..
> 0.9902.  This gate's own `max(f_both.on) < 0.95` bar, which S14 flags, is
> re-derived there as a SPREAD comparison.

### DEFECT 2 (P3, false positive) -- the band warning names an axis that carries no mortar

**Where** `lumenairy/elements/pmm/twod_staggered.py::_warn_stag_sliver_band`
and `_stag_band_narrowest`.

**What is wrong** the warning is conditioned on the STACK building a
cross-grid interface *somewhere* (`force_mortar or len({g.key()}) > 1`), but
`_stag_band_narrowest` then scans BOTH axes of every grid.  A stack whose
layers differ on x and share the y wall array EXACTLY therefore warns about a
narrow y segment -- while the y mortar is the IDENTITY, so nothing on that
axis is projected across grids and nothing is degraded.  This is the same
shape as the still-open DEFECT V2 (a conforming stack refused by the width
contract), one level finer: conformity is per AXIS, and the warning tests it
per STACK.

**Measured** on a device whose permittivity is UNIFORM in both layers, so the
y walls carry no feature and the DEVICE cannot depend on their separation:

| y width | `R00` | movement vs the widest | band warnings | axis named |
|---|---|---|---|---|
| 3.0e-01 | 0.247088457739 | -- | 0 | -- |
| 1.0e-01 | 0.247088457739 | 1.7e-14 | 0 | -- |
| 5.0e-02 | 0.247088457739 | 8.5e-13 | 0 | -- |
| 3.0e-02 | 0.247088457739 | 2.8e-13 | 0 | -- |
| **1.0e-02** | 0.247088457739 | 4.2e-13 | **1** | **y** |
| **3.0e-03** | 0.247088457739 | 3.1e-13 | **1** | **y** |
| **1.2e-03** | 0.247088457739 | 3.9e-13 | **1** | **y** |

WSL reads the same `R00` to twelve figures with movements 6.4e-15 .. 1.3e-13.

The message claims the width costs "about 4-5x the error of the same device on
an ordinary partition" (and "about 5-6x" / "about 6x, its floor" on the
narrower rungs).  Stated as the ratio the message is about: the answer on a
band-width y segment divided by the answer on an ORDINARY (3e-1) one is
**1.000000000000** to twelve decimal places, on both builds.  The claimed
factor is not merely small here; it is absent to round-off.

**Severity** P3.  It is a warning, not a refusal and not a wrong answer, and
the remedies it prints are harmless.  But it is exactly the class of noise
that trains users to silence a warning, and this warning is the ONLY line
between a user and the real (x-axis) degradation.

**Reproducer**
`tests/unit/test_verify_pmm2d_mortar_round3.py::test_the_band_warning_fires_on_an_axis_that_carries_no_mortar`
(pins the current behaviour) and
`validation/probe_verify_mortar_round3/v5_falsepos.py fp2`.

**Remedy** condition per axis: compute the narrowest segment on the x axis
only over grids whose x wall arrays are not all equal, and likewise for y.
`_stag_band_narrowest` already walks `(("x", g.bx), ("y", g.by))`, so the fix
is to pass it two booleans instead of one, derived where `_solve_per_layer`
already computes `len({g.key() for g in gof}) > 1`.

> **CLOSED 2026-09-11 by ROUND 4**
> (`docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md` S4), by exactly that
> remedy: `_stag_mortared_axes(grids, force)` returns the `(x, y)` pair and
> `_stag_band_narrowest(grids, axes)` searches only live axes.  Measured
> two-sidedly against round 3's own rule, run in the SAME interpreter: 801
> leaves compared on each build, **0 answer-hash differences, 0 other
> differences, 23 warning differences**, every one of them this false positive
> stopping or the correct axis being named.  The reproducer named above is
> RE-PINNED to the fixed behaviour and renamed
> `test_the_band_warning_does_not_fire_on_an_axis_that_carries_no_mortar`.

### NOT DEFECTS -- three candidates raised and refuted

* **a fine UNIFORM lattice in the band.**  `_stag_band_narrowest` scores a
  uniform basis as `1/N`, so `N >= 34` would warn, and a uniform lattice is
  the opposite of a sliver.  It is unreachable: the staggered basis carries
  `q = N (M - 1)` per axis and a `2 q^2` region eig, so the smallest banded
  lattice is a **20808**-dimension dense eigenproblem (6.5 GiB of operand) at
  `M` = 4 and 57800 (49.8 GiB) at `M` = 6.  Bounded by
  `test_a_uniform_lattice_cannot_reach_the_band_through_the_public_api`.  At
  the counts that ARE reachable (`N` = 1, 2, 4, 6, 8 beside a 3-segment
  pillar layer, `M` = 5) the stack is SILENT, residuates at 3.0e-15 ..
  8.0e-15, and sits 3.3e-03 .. 8.4e-03 from its own conforming twin -- an
  ordinary discretisation difference (`v5_falsepos.py fp1`).
* **two grids with a razor-thin COMMON REFINEMENT.**  Each layer's own
  segments can be wide (0.31) while their union carries a segment of 1e-11 --
  neither the contract nor the band sees that.  Measured over offsets 3e-1
  down to 1e-11 at `M` = 5: `R00` converges monotonically to 0.015361176253
  (the aligned answer), residual 1.5e-14 .. 3.3e-14, `s_min/s_max` 2.1e-12 ..
  4.8e-11, closure 1e-05 .. 3e-05.  The mortar handles a vanishing overlap
  gracefully.  REFUTED.
* **a silent WRONG answer above the band.**  See S10.

---

## S9. Ship recommendation

**SHIP round 3 in 5.45.0.**

The P1 it repairs is real and is reproduced here against the pre-fix tree on
my own geometry (S4.3); the instrument it chose is the right one and its two
gaps, re-derived on populations built here, are **6.34 decades below and 4.89
above, worst of both builds** (S4); the answers it un-refuses converge to a
mortar-free twin of the same device (S6); it moves nothing else, on an
independent 37-fixture / 120-hash battery whose harness is shown to reproduce
its own hashes first (S7); its cost is below what a shared box can resolve
(S0 row f); and its 14 gates are decision-shaped, derived from the running
build, and carry no cross-build value pin (S14).

The two defects found are P3 and neither blocks: DEFECT 1 is one word in three
docstrings, DEFECT 2 is a warning that fires on a harmless axis.  Both have a
gate in `tests/unit/test_verify_pmm2d_mortar_round3.py`, and DEFECT 2's gate
PINS the current behaviour so it fails -- loudly, with the instruction to
re-pin -- when the axis conditioning is added.

**One thing to record in the release notes rather than fix.**  The
degradation the band warning exists for is already **13.5x** (`M` = 7) /
**22.9x** (`M` = 8) by the time a segment reaches the band's UPPER edge, and
grows only a further 1.24x from there to the contract (S11), so the warning
covers the flat tail and not the rise.  That is a consequence of the census constraint the fix
states plainly; it is not a defect, but a user reading "no warning" as "no
degradation" would be wrong, and the doc currently invites that reading.

---

## S10. The HUNT -- an ordinary geometry, a wrong answer, a small residual

`v5_falsepos.py hunt` scores eleven per-layer stacks against a mortar-free
twin built on the COMMON REFINEMENT of the SAME device: an extreme index
contrast (`eps` = 144), a lossy metal (-20 + 1.5i), five far-field orders,
`M` = 8, a dense superstrate (n = 2.4 / 3.5), near-Wood incidence, a
SIX-layer stack, a strongly out-of-plane tensor, and two posts whose width
sits just above the contract and just above the band.  Every one of them has a
generalized-mortar residual of **8.60e-15 .. 1.21e-13**, i.e. 7 decades under
the bar, and TEN of the eleven raise no warning at all (the exception is the
post just above the width contract, which is inside the band and warns).

Two looked like a find: a 3.2e-2-wide out-of-plane post (JUST above the band's
upper edge, so silent) read `R00` = 1.44e-04 at `M` = 5 against its union
twin's 1.71e-02 -- **99.2 % apart, silent, residual 1.8e-14**.

**It is under-convergence, not a silent wrong answer, and `v8_hunt_ladder.py`
settles it.**  A comparison at one modal count is not a wrongness test; on the
ladder both arms move and they move TOWARD each other:

| case (per-layer arm) | `M` = 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| just-above-band `R00` | 0.001691117 | 0.000143918 | 0.012992754 | 0.013906507 |
| its lossless closure | 1.00e-02 | 1.29e-03 | 1.95e-04 | **4.53e-06** |
| max abs gap to the union arm (`M` = 6) | 2.349e-02 | 2.993e-02 | 1.139e-02 | **9.156e-04** |
| the arm's OWN step | -- | 6.434e-03 | 1.853e-02 | 1.048e-02 |

At the last rung the gap to the mortar-free twin is **11x INSIDE the per-layer
arm's own step**, and the union arm is not converged either (its own last step
is 6.401e-03, seven times the gap).  An ORDINARY 15 %-wide post, run as a
control through the same ladder, behaves identically (gap 2.658e-02 ->
3.015e-04 with an own last step of 5.602e-03).  And the lossless closure falls
three decades across the ladder, so this is a rung that a user's own
convergence study -- and the energy tripwire -- both see.  That is the
opposite of the sliver band, where the closure stays pinned.

**No silent wrong answer was found.**  Two further candidates were raised and
refuted (S8, "NOT DEFECTS"): a fine uniform lattice in the band, which no
machine can build, and two grids whose COMMON REFINEMENT carries a 1e-11
segment while each grid's own segments are 0.31 wide -- the case the width
contract and the band warning are both blind to by construction.  Measured
over offsets 3e-1 .. 1e-11, `R00` converges monotonically onto the aligned
answer 0.015361176253 with the residual at 1.5e-14 .. 3.3e-14 throughout.  The
mortar handles a vanishing overlap gracefully; there is nothing there.


---

## S11. The band, re-measured

`v4_band.py`.  The device is a y-uniform 3-layer stack whose MIDDLE layer is
ALL HOST, so it cannot depend on the wall separation and every deviation is
numerical damage; the oracle is the exact 1-D `PMMStack` at degree 14, whose
own 12 -> 14 self-gap is measured beside the errors it is used to compare.
The fixture is independent of the fix's in every knob -- period 1.07 (theirs
0.93), wavelength 0.79 (0.66), `theta` 0.31 (0.19), `eps` 9.0 / 1.69 (6.25 /
2.1), walls .205/.495 and .365/.735 (.155/.585 and .315/.795), sliver centre
0.585 (0.41).

**Oracle self-gap (deg 12 -> 14): 4.0144e-05**, three decades under the `M` = 7
and `M` = 8 errors it is used to compare and two under the `M` = 6 ones.

Error against the oracle, and the ratio to the ORDINARY (3e-1) end of the same
ladder:

| narrowest / period | 3e-1 | 2e-1 | 1.5e-1 | 1e-1 | 7e-2 | 5e-2 | **3e-2** | 2e-2 | 1e-2 | 3e-3 | 1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `M`=6 err | 4.99e-3 | 2.88e-3 | 9.00e-4 | 4.30e-3 | 5.94e-3 | 6.75e-3 | 7.21e-3 | 7.25e-3 | 7.05e-3 | 6.46e-3 | 5.79e-3 |
| `M`=6 ratio | 1.00 | 0.58 | 0.18 | 0.86 | 1.19 | 1.35 | **1.45** | 1.45 | 1.41 | 1.30 | 1.16 |
| **`M`=7 err** | 4.52e-4 | 2.29e-3 | 3.48e-3 | 3.96e-3 | 4.54e-3 | 5.15e-3 | 6.10e-3 | 6.67e-3 | 7.23e-3 | 7.57e-3 | 7.56e-3 |
| **`M`=7 ratio** | 1.00 | **5.06** | 7.70 | 8.75 | 10.04 | 11.39 | **13.49** | 14.75 | 16.01 | 16.74 | 16.74 |
| `M`=8 err | 1.86e-4 | 1.15e-3 | 1.19e-3 | 1.94e-3 | 2.93e-3 | 3.64e-3 | 4.26e-3 | 4.51e-3 | 4.70e-3 | 4.78e-3 | 4.77e-3 |
| `M`=8 ratio | 1.00 | **6.21** | 6.39 | 10.45 | 15.73 | 19.56 | **22.92** | 24.25 | 25.26 | 25.68 | 25.62 |
| WARNS | no | no | no | no | no | no | no | **yes** | **yes** | **yes** | **yes** |

WSL reproduces the `M` = 7 row's first rung: `err(3e-1)` = 4.518999e-04 on
both builds, to seven significant figures.  It is a pure discretisation
quantity with no meaningful cross-build spread, which is what the fix says and
what makes a 2x bar on it a decision rather than noise.

**Three readings.**

1. **The `M` = 6 row is uninterpretable on this fixture and should not be
   read.**  Its ordinary arm has NOT converged (4.99e-3 at 3e-1 against
   9.00e-4 at 1.5e-1), so the ratios are contaminated by the baseline and go
   BELOW 1.  The fix makes the same point in the other direction ("the floor
   is only VISIBLE at rungs where the ordinary arm has converged below it");
   on my fixture only `M` = 7 and 8 qualify.  The round-3 GATE asserts its
   decision on the `M` = 6 rung of ITS fixture, where their ordinary arm
   evidently does converge -- but it is a rung whose ratio is fixture-sensitive
   by construction, and the gate's 1.15 bar against a measured 1.55 is the
   thinnest margin in that file.
2. **The 2x criterion is first crossed at 2e-1, on both rungs** -- exactly
   where the fix measures it on their fixture (2e-1 at `M` = 8 on theirs,
   1e-1 on the round-2 verifier's).  CONFIRMED.
3. **The band's upper edge sits in a region that is already 13.5x degraded**
   (`M` = 7) / 22.9x (`M` = 8), against the fix's 4.65x / 4.02x.  Direction
   CONFIRMED, magnitude fixture-dependent by 3-5x -- so read "4.65x" as one
   fixture's reading, not as the band's cost.

**And the consequence, which is the substantive limitation.**  On the `M` = 7
row the error has already grown **13.5x by the time the width reaches the
band's upper edge**, and it grows only a further **1.24x** from there to the
contract.  The warning therefore covers the FLAT TAIL of the degradation, not
its rise: by the time it fires, about 92 % of the damage (measured on a log
scale between 3e-1 and 1e-3) has already happened silently.  That is a direct
consequence of the census constraint the fix states plainly -- the accuracy
criterion alone would put the edge at 2e-1, on top of the ordinary population
-- so it is not a defect.  But "no warning" must not be read as "no
degradation", and S9 asks for one line in the release notes to say so.

### 11.1 The census, and the message

`v4_band.py rest`, 32 geometries built through the public builders, IDENTICAL
on both builds (it is a pure geometry reading):

| geometry | narrowest / period | x the edge | in band |
|---|---|---|---|
| duty-0.9 pillar | **5.0000e-02** | **1.67** | no |
| uniform lattice N=16 | 6.2500e-02 | 2.08 | no |
| uniform lattice N=12 | 8.3333e-02 | 2.78 | no |
| two pillars, 5 % gap | 1.0000e-01 | 3.33 | no |
| duty-0.1 pillar | 1.0000e-01 | 3.33 | no |
| nested refinement | 1.2500e-01 | 4.17 | no |
| straight taper, 4..128 slices | 2.502e-01 .. 2.563e-01 | 8.3 .. 8.5 | no |
| `add_tapered_pillars`, 4..32 slices | 1.4094e-01 .. 1.4750e-01 | 4.7 .. 4.9 | no |
| CLOSING taper, 4 slices | 6.2500e-02 | 2.08 | no |
| CLOSING taper, 8 slices | 3.1250e-02 | **1.04** | no |
| **CLOSING taper, 16 slices** | 1.5625e-02 | 0.52 | **yes** |
| **CLOSING taper, 32 slices** | 7.8125e-03 | 0.26 | **yes** |
| **CLOSING taper, 64 slices** | 3.9062e-03 | 0.13 | **yes** |
| **CLOSING taper, 128 slices** | 1.9531e-03 | 0.07 | **yes** |

**0 ordinary geometries land in the band, on both builds.**  CONFIRMED.

**But the margin is 1.67x, not 3.6x.**  The fix's "3.6x below the narrowest
ordinary geometry (1.2500e-01)" and the round-3 gate's `worst > 3.0 * edge`
are properties of the ROUND-2 BATTERY, which contains no high-duty pillar and
no fine uniform lattice.  A 90 %-duty pillar -- a 5 %-wide trench either side,
an entirely ordinary photonic feature -- sits at 1.67x, and a 94 %-duty one
would sit ON the edge.  The gate's docstring states the 3x of "ANY ordinary
per-layer geometry the library builds"; it is measured over a sample.  My gate
`test_ordinary_geometries_sit_closer_to_the_band_edge_than_3x` records the
scope.

**The closing taper reaches the band from NINE slices**, not from 32: its
narrowest sampled width is `w_bottom / (2 n_slices)`, so 8 slices is
3.1250e-02 (1.04x OUTSIDE) and 9 is 2.78e-02 (inside).  The fix's 8.0094e-03
at 32 and 4.1047e-03 at 64 are reproduced in kind (mine: 7.8125e-03 and
3.9062e-03 on a 0.25 .. 0.75 pillar), but the surface starts warning three
slice counts earlier than the doc suggests.

**The message.**  All **15/15** tokens present on both builds: the width, the
layer, the AXIS, the whole wall array, the contract, both band edges, the
degradation class, "FLOOR", "n_modes does NOT remove", "no energy tripwire can
see this", all five remedies, and the switch's fully-qualified name.  Two-sided
on both builds: silent at 6e-2, ONE warning at 2.9e-2, ONE at 1.2e-3, REFUSED
at 9e-4.  The switch: 0 warnings off, 1 on.  **Once per SOLVE**: a four-layer
stack with FOUR layers in the band and four distinct grids emits exactly ONE
warning.  A CONFORMING per-layer stack at the same width emits none while its
non-conforming twin emits one.  Across the whole 367-test `pmm2d` surface the
new warning appears exactly **once**, on
`test_the_mortars_own_algebra_is_exact_at_every_wall_separation` -- the round-2
gate that deliberately sweeps wall separations through the band.


---

## S12. What could NOT be verified

1. **A THIRD BLAS FAMILY.**  Both builds link scipy-openblas, as in the
   build, both fixes and the round-2 verification.  Every cross-build spread
   here is a LOWER bound.  It matters least for the residual bar (two-sided
   margins of 6.3 and 4.9 decades against a measured cross-build spread under
   2x on every reading) and most for S3.3.
2. **`M` > 8 at the generalized site.**  My healthy population reaches `M` = 8
   (`n` = 1764).  `M` = 9 on a 3-segment grid is a `4 q^2` = 2304 eig per
   layer and was not run.
3. **The `M` = 6 rung of the degradation ladder is uninterpretable on my
   fixture** (S11 reading 1), so my ladder confirms the 2x crossing and the
   edge's cost at `M` = 7 and 8 only.  The round-3 GATE asserts at `M` = 6 on
   its own fixture; I could not check that rung's robustness, because my
   fixture's `M` = 6 ordinary arm has not converged and running the fix's
   fixture would not be an independent measurement.
4. **`M` > 8 on the ladder**, where the floor would be more visible still: the
   `M` = 8 row alone is 32 minutes on this box, and its ordinary arm
   (1.86e-04) is already only 4.6x the oracle's own self-gap (4.01e-05), so an
   `M` = 9 row would need a degree-16 oracle as well.
5. **The WSL degradation ladder was run at `M` = 7 only** (the row that
   carries the argument), not at 6 and 8.  All eleven of its rungs reproduce
   WIN to SEVEN significant figures, which is the strongest cross-build
   statement in this report and the reason the other two rows were not
   repeated.
5b. **The HUNT's convergence ladder (`v8`) was run on WIN only.**  It is the
   most expensive probe here (its union arm is 200 s a rung at `M` = 6) and
   what it measures -- whether two discretisations of one device approach
   each other as `M` rises -- has no meaningful cross-build content: the
   quantity is a discretisation gap of 1e-3 .. 1e-2 against a cross-build
   spread of 1e-13.  Its sibling `v7` WAS run on both builds and reproduces
   to four significant figures, which is the evidence for that reading.  S10's
   conclusion is therefore single-build; S6's is not.
6. **The WSL sliver ladder was run at `M` = 5 only**, and the WSL population
   probe was killed once by memory pressure mid-run (another agent's pytest
   held 40 GB of a 128 GB box, leaving 45 MB free), which is why the
   populations are now written per PART rather than at the end.
7. **The fix's own probes and gate fixtures were not re-run.**  Every number
   in this report comes from fixtures written in
   `validation/probe_verify_mortar_round3/`; where the fix's number and mine
   disagree (S4.1's 6.34 vs 6.9 decades, S11's 13.5x vs 4.65x) the
   disagreement is a fixture difference, not a contradiction, and both are
   stated.
8. **DEFECT 2's remedy was not implemented or measured.**  I verified that the
   warning fires on a non-mortared axis and that the answer does not move
   there; I did not build the per-axis conditioning and check that it stays
   silent on the x-axis case while still firing on the y one.
9. **The MERGE.**  `wave2/pmm2d` was merged into this branch and the seven
   suites repeated on both builds (S13); the PROBES were not re-run on the
   merged tree, so every measured number above belongs to the fix branch tip
   `5f01eea`.  The merge changes `lumenairy/elements/pmm/stack.py` (the 1-D
   sliver arbiter) and `lumenairy/elements/rcwa/_core.py` (the modal
   branch-cut fix, which the RCWA verification measures at 1e-17 .. 5e-15 on
   lossless answers), so the one thing that could have moved under my numbers
   is the 1-D `PMMStack` ORACLE the degradation ladder is scored against.  It
   did not: re-run on the merged tree, its degree-12 -> 14 self-gap reads
   **4.0144485987e-05**, identical to the branch-tip reading to all eleven
   figures, and its `M` <= 1 reflectances are unchanged.

---

## S13. Runs

### 13.1 On the fix branch tip `5f01eea`

**The seven required suites, BOTH builds, one thread set in the SHELL**
(`test_fix_pmm2d_mortar_round3.py` + `test_fix_pmm2d_mortar_round2.py` +
`test_pmm2d_staggered_mortar.py` + `test_pmm2d_staggered_nonuniform.py` +
`test_verify_pmm2d_perlayer_slant.py` + `test_pmm2d_staggered_oop.py` +
`test_pmm2d_staggered_slant.py`):

```
WIN   185 passed, 3 warnings in 1225.73s (0:20:25)
WSL   185 passed, 3 warnings in 1074.17s (0:17:54)
```

**The whole `pmm2d` surface, Windows** (`test_pmm2d*.py` +
`test_fix_pmm2d*.py` + `test_verify_pmm2d*.py`).  This run was collected
BEFORE this verification's own gate file existed, so its 367 is directly
comparable with the fix's 367 -- the same tests, not the same count by
coincidence:

```
367 passed, 24 warnings in 2146.83s (0:35:46)
```

`grep -c "degradation band"` over that log is **1**, on
`test_fix_pmm2d_mortar_round2.py::test_the_mortars_own_algebra_is_exact_at_every_wall_separation`.

**This verification's own gates**, both builds:

```
WIN   4 passed in 15.61s
WSL   4 passed in 17.29s
```

Slowest gate 13.25 s (`..._sit_closer_to_the_band_edge_than_3x`), well inside
the 60 s budget; the four node ids are spliced into `.test_durations`
(12669 -> 12673 entries, sorted) at the MAX of the two builds.

### 13.2 After merging `wave2/pmm2d`

`git merge --no-edit wave2/pmm2d` at `2898767` (which carries the round-3 fix
merge `448ee85`, the sliver round-3 arbiter, the slant V1/V2/O2 fix and the
RCWA modal branch-cut fix).  `.test_durations` auto-merged to 12722 entries,
sorted, with all four of this verification's node ids present and all
fourteen of the round-3 gate's.

**The merge does not touch any of the three files round 3 changed**
(`git diff --stat 20e112b 852ea61 -- lumenairy/elements/pmm/{_core,twod_staggered,stack2d_pure}.py`
is empty), so nothing above can have moved under it; the two files it does
change are `lumenairy/elements/pmm/stack.py` and
`lumenairy/elements/rcwa/_core.py`, and the one thing of mine that reads
either -- the 1-D `PMMStack` oracle -- is re-measured unchanged to eleven
figures (S12 item 9).

The seven suites PLUS this verification's own file, both builds, one thread in
the shell:

```
WIN   189 passed, 3 warnings in 1154.39s (0:19:14)
WSL   189 passed, 3 warnings in 1191.33s (0:19:51)
```

189 = the seven suites' 185 plus this verification's 4, and the three warnings
are the same three as before the merge: the new band warning on
`test_the_mortars_own_algebra_is_exact_at_every_wall_separation` (one
occurrence, checked by `grep -c` on both logs) and round 2's two open-item
`RuntimeWarning`s.  **Nothing the merge brought in changes any reading in this
report**, and this verification's own four gates -- including the one that
PINS defect 2 -- pass on the merged tree on both builds.

**A merge mechanic worth recording.**  The first `git merge` printed
`fatal: update_ref failed for ref 'HEAD': couldn't set
'refs/heads/verify/mortar-round3'` while reporting `Merge made by the 'ort'
strategy` -- the index and worktree were merged but the commit was never
created.  The shared `.git` on `D:` is being written by several agents at
once, so ref-lock contention is the likely cause.  `git commit --no-edit`
completed it (`852ea61`).  A merge that says it succeeded is not one until
`git log` shows the commit.

### 13.3 `ruff`

```
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmortar3 && ~/lumvenv/bin/ruff check \
  lumenairy/ tests/ validation/probe_verify_mortar_round3/'
All checks passed!
```

### 13.4 Commands

```sh
# every shell command starts with the cd -- a pytest launched elsewhere
# prints "no tests ran in 0.02s" and exits 0
cd /c/tmp/lum_vmortar3

# probes, both builds (the tag is the build)
for s in v1_mechanism v7_convergence v8_hunt_ladder; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u validation/probe_verify_mortar_round3/$s.py win
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u validation/probe_verify_mortar_round3/v2_populations.py \
  win_healthy healthy          # ... and win_sliver sliver, win_broken broken
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u validation/probe_verify_mortar_round3/v4_band.py win rest
V4_LADDER_MS=6,7,8 ... v4_band.py win ladder
... v5_falsepos.py win fp1|fp2|hunt

# cost: generate the operands ONCE, then bench each tree against the SAME
# .npz (the sha256 of every operand is recorded in both runs)
python -u validation/probe_verify_mortar_round3/v3_cost.py gen
python -u validation/probe_verify_mortar_round3/v3_cost.py bench win_post \
  C:/tmp/lum_vmortar3
python -u validation/probe_verify_mortar_round3/v3_cost.py bench win_pre \
  C:/tmp/lum_vmortar3_pre

# bit-identity: the harness's OWN determinism first, then the comparison
python validation/probe_verify_mortar_round3/v6_bitid.py selfcheck \
  C:/tmp/lum_vmortar3
python validation/probe_verify_mortar_round3/v6_bitid.py post C:/tmp/lum_vmortar3
python validation/probe_verify_mortar_round3/v6_bitid.py pre  C:/tmp/lum_vmortar3_pre
python validation/probe_verify_mortar_round3/v6_bitid.py compare post pre

# WSL: the same, through the venv
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmortar3 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/lumvenv/bin/python -u \
  validation/probe_verify_mortar_round3/v1_mechanism.py wsl'

# the seven suites, BOTH builds, ONE thread -- on the COMMAND LINE, because
# the files' own os.environ.setdefault is a no-op in a multi-file run
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/unit/test_fix_pmm2d_mortar_round3.py \
  tests/unit/test_fix_pmm2d_mortar_round2.py \
  tests/unit/test_pmm2d_staggered_mortar.py \
  tests/unit/test_pmm2d_staggered_nonuniform.py \
  tests/unit/test_verify_pmm2d_perlayer_slant.py \
  tests/unit/test_pmm2d_staggered_oop.py \
  tests/unit/test_pmm2d_staggered_slant.py -q -p no:randomly --durations=15
```

**Three traps this verification hit, recorded because each produced
authoritative-looking output.**

1. **`sys.path[0]` is the SCRIPT's directory, not the working directory.**  The
   first mechanism run measured a checkout installed on `D:` and reported 31
   rows of `AttributeError` -- but a probe that merely READ numbers would have
   reported the wrong library's numbers with no symptom at all.  Every probe
   now imports `_path.py` first, which refuses to run unless
   `lumenairy.__file__` is inside this worktree, and records the resolved path
   in its JSON.
2. **`np.asarray` of a DICT is a 0-d object array whose `tobytes` is a
   pointer.**  `per_order_amplitudes()` returns a dict, and hashing it that
   way made the bit-identity battery report one HASH DIFF in a fixture the
   round-3 diff cannot touch -- a harness bug that reads exactly like a
   library regression.  `v6_bitid.py selfcheck` now runs the battery twice in
   ONE tree first and reports 37/37 reproducible before any cross-tree
   comparison is believed.
3. **`np.linalg.solve` is not the shipped path.**  On an exactly singular
   operand `gesv` raises `LinAlgError` while `getrf` + `getrs` warn and return
   a non-finite answer, and the second is what `_guarded_mortar_solve`
   residuates.  Measuring the broken population the other way killed a
   two-hour run on both builds.  The populations are now written per PART so
   one failure cannot destroy the others.

---

## S14. Test durability -- every constant in the round-3 gate file

`tests/unit/test_fix_pmm2d_mortar_round3.py`, 14 gates.  For each numeric
constant: where it comes from, what margin it carries against the value the
running build produces, whether that margin is FAMILY-scoped (re-derived from
the build on every run) or SAMPLE-scoped (a property of the fixture the gate
happens to use), and whether the quantity it reads has a cross-build spread at
all.

| constant | origin | measured margin | scope | build-sensitive? |
|---|---|---|---|---|
| `mixed.s_ratio < 1e-4 * ctrl.s_ratio` | the CONTROL's own reading, re-measured each run | 600x (mine: 1.36e-10 / 8.11e-04 = 1.7e-07) | FAMILY | no -- both sides move together |
| `max(mixed.on) > 0.95`, `min(mixed.on) < 0.05` | the localisation DECISION | exact: 1.000 / 0.000 on all 16 mixed operands I built, both builds | FAMILY | no |
| `max(ctrl.on) < 0.95` | the control must NOT localise | 0.12 on the gate's own control (0.829) -- but only **0.015** on a legitimate neighbouring fixture: an out-of-plane tensor that is in-plane to 1e-9 reads **0.935** | **SAMPLE** | no, but fixture-sensitive |
| `mixed.residual < 1e-2 * _MORTAR_RESID_REFUSE` | the shipped bar | 1e+6 (8.6e-15 against 1e-08) | FAMILY | no |
| `mixed.residual < 1e-11` | none stated -- an ABSOLUTE floor | 3 decades (8.6e-15).  The file's ONLY absolute bar; harmless because it is asserted at `M` = 4 only, so it cannot walk with `n eps` | SAMPLE | no |
| `closure < 1e-2` (x4) | "this is a real answer" | 3-6 decades | FAMILY | no |
| `step56 < 0.2 * step45` | the ladder's own steps | 12x (measured 61x against an asserted 5x) | FAMILY | no |
| `gap < 0.5 * step45` | the ladder's own coarsest step | 5.5x (measured 11x inside) | FAMILY | no |
| `abs(per5 - uni) < max(5 * step, 1e-2)` | the arm's own step, OR an absolute 1e-2 | the `1e-2` term is ~17 % of `R00` and would dominate if the step ever collapsed; on my own convergence data the gap is comfortably inside the `5 * step` term, so the floor is never reached | FAMILY, with a SAMPLE floor | no |
| `worst_healthy < 1e-4 * bar` | the healthy population, re-measured | 2.2e+06 (my worst healthy 4.56e-13) | FAMILY | no (WIN/WSL within 2 %) |
| `bad_singular > 1e4 * bar`, `bad_incons > 1e4 * bar` | the broken population, re-measured | 8x / 1.6e+03 (mine 7.8e-02 / 1.65e+01) | FAMILY | no |
| `true_singular > 1e2 * bar` | the least-squares residual | 12x measured | FAMILY | no |
| `ok_rank_def < 1e-4 * bar` | the V1 shape must PASS | 1e+07 (9.7e-15) | FAMILY | no |
| `0.01 < probe/exact < 100` | the estimator | 64x (mine 0.68 .. 1.55 healthy, 0.74 .. 1.78 broken) | FAMILY | no |
| `2.0e-2` / `6.0e-2` / `1.2e-3` / `9.0e-4` band widths | stated multiples of the two edges (1.5x inside, 2x outside) | they are INPUTS, not readings | FAMILY | no |
| `"2.000e-02" in msg` | a formatted echo of the gate's own input | exact | FAMILY | no |
| **`worst > 3.0 * edge`** (the census) | the ROUND-2 BATTERY's narrowest ordinary geometry | 1.2x on that battery (1.09e-1) -- but **0.56x** on a duty-0.9 pillar (5.0e-02), which the battery does not contain.  The docstring claims it of "ANY ordinary per-layer geometry the library builds" | **SAMPLE, stated as FAMILY** | no |
| `_STAG_MIN_SEG_FRAC <= f < edge` (closing tapers) | the library's own two constants | derived; reproduced (mine 7.81e-03 / 3.91e-03 at 32 / 64 slices) | FAMILY | no |
| `self_gap < 0.05 * abs(e_band - e_ord)` | the oracle's own 12 -> 14 gap | 500x on my fixture (4.01e-05 against a 7.1e-03 difference) | FAMILY | no |
| **`e_band > 1.15 * e_ord`** at `M` = 6 | the fix's own ladder on ITS fixture (1.55) | 1.35x -- **the thinnest margin in the file**, and it sits on the one rung whose ratio is fixture-sensitive by construction: on MY independent fixture the `M` = 6 ordinary arm has not converged and the same ratio reads 1.30 at 3e-3 and **0.18** at 1.5e-1 | **SAMPLE** | no -- the quantity is identical WIN/WSL to 7 significant figures |

**No cross-build value pin anywhere in the file.**  I checked every numeric
literal: the only fixed numbers are the library's own constants, the geometry,
the band widths (inputs), and the two SAMPLE-scoped bars called out above.
The one measured reading recorded in the file (`R00` at `M` = 7) is in a
comment and is explicitly not asserted, as the fix says.

> **CLOSED 2026-09-11 by ROUND 4**
> (`docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md` S5).  Both bars below
> are restated family-scoped.  `max(ctrl.on) < 0.95` becomes a SPREAD
> comparison (`min(on)/max(on)`, re-measured on BOTH sides each run) with 3.6
> decades / 1.5e+04x / 3.8x of margin over a 33-operand population; the
> `M` = 6 ratio bar becomes a two-rung assertion with the ordinary arm's
> convergence VERIFIED from the ladder itself (`e_ord(6) < 0.5 e_ord(5)`,
> measured 0.271).  The census's 3x is recorded at its real 1.67x in
> `_STAG_SLIVER_BAND_FRAC`'s comment and gated at 1.25x over the wider
> 47-geometry census.

**Two observations for the next round, neither ship-blocking.**

1. `max(ctrl.on) < 0.95` and `e_band > 1.15 * e_ord` are the two bars whose
   margin is a property of the chosen fixture rather than of the library.  The
   first would be crossed by a control whose tensor is nearly in-plane (0.935
   measured); the second by any fixture whose `M` = 6 ordinary arm has not
   converged (0.18 measured on mine).  Both are cheap to make family-scoped:
   assert the control's participation against the MIXED one's rather than
   against 0.95, and take the ladder's decision on the rung whose ordinary arm
   the gate itself verifies has converged.
2. `test_no_ordinary_geometry_the_library_builds_lands_in_the_band` states a
   family claim over a sample.  Adding a duty-0.9 pillar and a 16-cell uniform
   lattice to that census would make the assertion honest -- at `2.0 * edge`
   rather than `3.0 * edge`, which my census still clears.

**The thread-pinning trap the fix records is real and is in this file too.**
`os.environ.setdefault("OMP_NUM_THREADS", "1")` at import time does NOTHING in
a multi-file run, because numpy has already been imported by an earlier file.
Both the round-3 gate file and this verification's own gate file carry that
pattern; both carry a comment saying so; and every multi-file run in S13 sets
the three variables in the SHELL.
