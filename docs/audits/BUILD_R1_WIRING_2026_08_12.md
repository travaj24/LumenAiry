# BUILD -- R1, the angle-aware momentum kick, and the traced wiring

**2026-08-12.  Branch `feat/screen-r1-wiring`, cut from `fix/verify-arch`
@ `0f46efb`, worked in `C:/tmp/lum_r1`.  No commit, no push, no `gh`, no
`CHANGELOG`, no `xfail`, no `skip`.**

Two sequenced items on the analytic-lens path:

* **ITEM 1 (R1)** -- close the defect
  `BUILD_SCREEN_OBLIQUITY_2026_08_11` S5 named and left standing: the screen's
  momentum kick `-(n2-n1) grad sag` is ANGLE-BLIND, worth 0.090692 waves rms on
  design 121 group 5 against a tangent-facet arm's 0.000372.
* **ITEM 2 (wiring)** -- `VERIFY_ARCHITECTURE` F5/P2-9: ZERO `apply_real_lens`
  calls forwarded `carrier=`, so the whole angular correction was inert for
  every shipped consumer.

---

## 0. VERDICT

> **BOTH SHIPPED.  R1 TAKES DESIGN 121 GROUP 5 FROM 0.090692 TO 0.012398 WAVES
> RMS (7.3x, 20.8x AGAINST THE BLIND SCREEN), AND THE CORRECTION IS NOW
> REACHABLE FROM THE DELEGATE BRANCH, THE SYSTEM CHAIN AND THE JONES WRAPPER.**
>
> **1. THE DEFECT WAS MIS-ATTRIBUTED, AND THE MEASUREMENT SAYS SO.**  S5 of the
> build note called the residual a DEFLECTION error entering through the
> FOLLOWING gap.  Decomposing the exit-plane error into its two channels
> (S1.2) on the same g5 fixture: with the kick made EXACTLY right and the
> shipped OPD kept, the error is **0.089555 w** -- essentially unmoved.  With
> the OPD made exactly right and the blind kick kept, it is **0.012477 w**.
> The residual is 87 % OPD channel, not deflection channel.
>
> **2. AND THE OPD CHANNEL IS THE DEFLECTION DEFECT, SEEN THROUGH THE
> PRECEDING GAPS.**  The screen's kick error is `-Lam grad sag` with
> `Lam = (n2-n1) - dz`; its potential is the CARRIER-FREE field
> `E = Lam * sag`, which the ray samples where it actually crosses the
> surface -- and the carrier MOVES that crossing.  Measured directly by
> accumulating `sum_i [E(x_i^theta) - E(x_i^0)]` along the model's own rays:
> **0.081408 waves**, against the 0.086080 the channel decomposition demands.
> **95 % of the gap accounted for by one closed-form term.**
>
> **3. THE FIX IS ONE MORE SCREEN TERM, `-U . grad E`, WITH NO FITTED
> CONSTANT.**  `U` is the carrier-induced drift, accumulated on the grid beside
> the existing `p0`.  On design 121's own fixture, in the proto's own idiom
> (the arm that produced the 0.090692):
>
> ```
>   group 5, 3 mm   blind 0.25848  ->  eq (4) 0.090692  ->  + R1 0.012398   (20.8x)
>   group 5, 2 mm         0.10449  ->         0.032784  ->        0.004586  (22.8x)
>   group 5, 1 mm         0.02459  ->         0.006971  ->        0.000990  (24.8x)
>   group 2, 3 mm         0.01680  ->         0.000552  ->        0.000498  (33.7x)
>   group 3, 3 mm         0.00991  ->         0.000020  ->        0.000020  (496x)
>   group 4, 3 mm         0.00049  ->         0.000088  ->        0.000027  (18.1x)
>   groups 0, 1 (plates)  0.00000  ->         0.000000  ->        0.000000
> ```
>
> **0.012398 against the tangent-facet arm's 0.000372.**  The remaining 33x is
> named and bounded in S1.5: it is the deflection channel PROPER, which is not
> the gradient of any scalar and cannot be carried by a screen at all.  The
> 0.012477 measured in S1.2 with a perfect OPD and a blind kick is that floor,
> and R1 lands on it (0.012398).
>
> **4. THE PLATE IS EXACT AND THE BYTE-NULLS ARE STRUCTURAL.**  `E` is
> proportional to the sag, so a plate contributes nothing; the drift is skipped
> entirely at zero carrier (`_obl_q_zero`) and at surface 0, so the carrier-free
> and zero-angle-carrier calls never evaluate R1 at all rather than evaluating
> it to zero.  The plate's own walk-off is untouched: it is the gap's, and
> `_screen_drift_step` reproduces `t q / sqrt(n^2 - q^2)` exactly (pinned).
>
> **5. THE SIGN IS CONTROLLED THE SAME WAY EQUATION (4)'s WAS.**  Negating R1
> on a two-powered-surface element at 90 mrad lands ABOVE the no-R1 arm
> (0.032898 -> 0.011508 with it, 0.074988 against it -- a stable 2.27-2.30x of
> the no-R1 arm across a 3x angle range), so the term cancels a defect rather
> than merely having the right size.
>
> **6. THE WIRING IS DECIDED PER CALL SITE, AND THE ONE THAT MATTERS IS
> MEASURED.**  The traced path's four AMP legs must NOT forward: the traced
> exit is `E_analytic * exp(i(k0 opl - phase_analytic_lens))` and BOTH factors
> come from `apply_real_lens`, so the correction is subtracted straight back
> out -- forwarding to all of them moves the traced answer by **8.7e-05 waves**
> where the correction itself is **4.3e-03** (a 49x cancellation), and
> forwarding to only the `E_in` leg injects the FULL 4.3e-03 into an answer
> that is already exact.  The DELEGATE branch, where the analytic model IS the
> answer, now forwards.
>
> **7. AND THE DELEGATE'S "WRONG CARRIER" CASE IS ADJUDICATED BY MEASUREMENT,
> NOT BY ASSUMPTION.**  That branch fires precisely because the carrier does
> not describe the field.  The forward is gated on the guard's OWN F1
> statistic: forward iff removing the carrier REDUCES the input's angular
> spread.  A carrier of the right size and the wrong sign raises the residual
> (0.1065 vs a raw 0.0619) and is REFUSED, byte-identically to the carrier-free
> answer, with the refusal named in the warning.
>
> **8. THE DESIGN-121 BANNER CANNOT HAVE MOVED, AND THAT IS MEASURED.**  The
> traced call the acceptance chain makes is **byte-identical** to the branch
> base on groups 2 and 5 (sha256 of the complex exit field).

---

## 1. THE DERIVATION

### 1.1 What equation (4) does and does not fix

`BUILD_SCREEN_OBLIQUITY_2026_08_11` derives the axial-translation identity: a
facet at height `s` costs the exit-referenced eikonal exactly
`s (pz1 - pz2)`, so the eikonal-exact screen OPD is `dz * sag` with
`dz = pz2 - pz1` from exact vector Snell at the local facet normal.  The
shipped correction is that MINUS its carrier-free value,

```
    dOPD_i = [ dz(p0 + q) - dz(p0) ] * sag_i .                             (4)
```

That fixes the screen's OPD VALUE.  A screen also has to DEFLECT, and its kick
is the gradient of whatever it imprints:

```
    K_screen = -grad[(n2-n1) sag] = -(n2-n1) grad sag        (angle-blind)
    K_facet  = -dz grad sag                                  (exact)
    K_screen - K_facet = -Lam grad sag ,   Lam = (n2-n1) - dz .            (5a)
```

`Lam` depends on the local ray momentum through `dz`, so the kick error is
angle-dependent -- that is R1 as S5 named it.

### 1.2 The two channels, and which one actually carries the residual

Score the model against exact rays the way the oracle does: carry the model's
eikonal to the exact ray's landing point at the model's own exit momentum.
To first order in the per-surface screen perturbation that is

```
    D = dLam - p . dx ,                                                    (5b)
```

an **OPD channel** (`dLam`, the screen's value errors along the ray) plus a
**DEFLECTION channel** (`p . dx`, the landing error the kick errors cause).
Both are measurable independently by building a ray arm that takes its OPD
from one model and its kick from another.  Design 121 group 5, 3 mm pupil,
54.87 mrad, `validation/.../screen_obliquity_derive.py`-class arms:

| arm (OPD / kick) | exit-plane angular error |
|---|---|
| blind screen / blind screen | 0.258480 w |
| eq (4) / eq (4)'s own gradient | 0.086980 w |
| **eq (4) / EXACT tangent-facet kick** | **0.089555 w** |
| **exact translation OPD / blind screen kick** | **0.012477 w** |
| exact translation OPD / exact kick (the tangent-facet arm) | 0.000372 w |

**Making the kick exact does not help (0.0870 -> 0.0896).  Making the OPD
exact does (0.0870 -> 0.0125).**  The defect S5 named is real, but 87 % of
what it costs arrives through the OPD channel, not the deflection channel.

### 1.3 Why: a carrier-free error, sampled at a carrier-shifted point

The difference between the "exact translation OPD" arm and the eq-(4) arm is
exactly `Lam * sag` -- carrier-FREE, and therefore invisible to the
common-mode control **if the two arms sampled it at the same place**.  They do
not: the carrier displaces the ray's crossing of every surface after the first.
Define

```
    E_i(x) = Lam_i(x) * sag_i(x)                                           (5)
    U_i    = sum_{j<i} t_j [ (p0+q)/pz_a - p0/pz_b ]_j                     (6)
```

-- `E` is the potential of the angle-blind kick error (5a), and `U` is the
transverse drift the carrier adds over the gaps BEFORE surface `i`.  Then the
angular error the eq-(4) screen still carries is `sum_i [E(x_i^th) - E(x_i^0)]
= sum_i U_i . grad E_i`.

Measured directly, by accumulating `E` along the model's own traced rays at
54.87 mrad and at 0, piston and tilt removed: **0.081408 waves rms**.  The
channel decomposition of S1.2 requires `sqrt(0.086980^2 - 0.012477^2) =
0.086080`.  **95 %**, from a term with no free parameter.

So the R1 screen term is

```
    dOPD_R1,i(x) = - U_i . grad E_i(x)                                     (7)
```

with three properties that are structural rather than numerical:

1. `U == 0` (no carrier; surface 0, which has no gap in front of it) -> zero;
2. `sag == 0` (a plate face, a cemented plano) -> `E == 0` -> zero;
3. it needs no ray trace, no fit and no cache -- `U` accumulates on the grid
   beside the `p0` the correction already carries.

### 1.4 The drift, and the one feedback that matters

`U` is not simply `t q / pz` summed over gaps: the element re-images its own
drift, so the CARRIER-FREE arm's momentum must be read at ITS own position,
`p0(x - U) ~ p0(x) - (U . grad) p0(x)`.  Measured on g5, dropping that
feedback costs 14 % of the term (grid-consistent arm: 0.022158 vs 0.019049
waves).  `grad p0` is one `gradient` call per component on the accumulator the
loop already maintains -- no Hessian of the sag is formed.

Two forms were measured and the cheaper one is also the better one.  The
LINEAR term `-U . grad E` reads 0.019049 w; the exact shift `E(x-U) - E(x)`
reads 0.033671 w.  The linear term is what the first-order response (5b)
actually asks for; the shift's second-order piece has no counterpart in the
score and makes it worse.  **Nothing here is fitted**: a scale scan on the
term peaks near 0.85-0.90 (0.0141 w) and unit coefficient is what ships.

### 1.5 What R1 does NOT fix, and why no screen can

With (7) applied, the exit-plane error lands on the **0.012477 w** the S1.2
decomposition measured for a PERFECT OPD with the blind kick.  That is the
deflection channel proper, `sag grad dz` acting through the gaps AFTER the
surface.  It cannot be carried by a screen: a screen imparts `-grad Psi`, so
the momentum field it can deliver is curl-free, and

```
    curl( dz grad sag ) = grad dz x grad sag  =/=  0
```

whenever the carrier breaks the surface's rotational symmetry (for a
rotationally-symmetric sag and a collimated `q` the z-component is
`-2c(n2-n1) u(r)^2 (r x q)_z`, which vanishes only on the meridian).  Closing
it is the transverse-walk REMAP axis (`surface_model='displaced'`), not this
one.

---

## 2. THE MEASUREMENTS

The oracle is the shipped exact ray tracer; the model arm is
`apply_real_lens`'s OWN split-step system traced as a Hamiltonian ray system,
differenced at the exit plane with the common-mode `D(theta) - D(0)` and piston
and tilt removed.  Reproduce with

```
python screen_obliquity_derive.py plate    # the plate zero, WITH R1 on
python screen_obliquity_derive.py sphere   # single facets: R1 is exactly zero
python screen_obliquity_derive.py d121     # all six groups, blind / eq4 / +R1
python screen_obliquity_derive.py ablate   # per-surface, + the tangent facet
python screen_obliquity_derive.py guard    # the estimator on d121
```

### 2.1 Design 121, all six post-DOE groups (waves rms)

| group | prescription | tilt | pupil | SHIPPED | eq (4) | **+ R1** | gain |
|---|---|---|---|---|---|---|---|
| 0 | plate N-SF1 | 51.50 mrad | 3 mm | 0.00000 | 0.000000 | **0.000000** | -- |
| 1 | plate N-BK7 | 51.50 mrad | 3 mm | 0.00000 | 0.000000 | **0.000000** | -- |
| 2 | doublet PK52A/SF57 | 51.50 mrad | 3 mm | 0.01680 | 0.000552 | **0.000498** | 33.7x |
| 3 | singlet LAK8 | 46.69 mrad | 3 mm | 0.00991 | 0.000020 | **0.000020** | 496x |
| 4 | singlet LAK9 | 7.38 mrad | 3 mm | 0.00049 | 0.000088 | **0.000027** | 18.1x |
| **5** | **doublet SK2/SF57** | **54.87 mrad** | **3 mm** | **0.25848** | **0.090692** | **0.012398** | **20.8x** |
| 5 | doublet SK2/SF57 | 54.87 mrad | 2 mm | 0.10449 | 0.032784 | **0.004586** | 22.8x |
| 5 | doublet SK2/SF57 | 54.87 mrad | 1 mm | 0.02459 | 0.006971 | **0.000990** | 24.8x |

**The achieved g5 number is 0.012398 waves rms**, against the 0.090692 it had
to beat and the 0.000372 tangent-facet arm it is measured against.  The 1/2/3
mm ladder scales as the pupil the same way the blind column does, so this is
the same function of the pupil with a 20x smaller coefficient, not a
cancellation at one radius.

Group 2 at the smaller pupils is the one place R1 is not an improvement:
0.000020 -> 0.000048 (1 mm) and 0.000159 -> 0.000204 (2 mm).  Those are
2-5e-05 waves, i.e. 1/500 of the tolerance, and are recorded rather than
hidden.

### 2.2 The plate, and the single facets: exactly unchanged

`plate`: with R1 ON, the corrected eikonal is **byte-identical** to the blind
one at 0 / 10 / 20 / 41.5 / 100 mrad (`correction - shipped = 0.000e+00 m`,
`byte_identical True`), and the shipped screen's own angular error on a plate
stays 1.413e-26 w.

`sphere`: all 20 rows of the single-facet ladder (N-BK7 / N-SF11, R = +/-25 to
+/-50 mm, 10 to 100 mrad) read **the same digits with R1 on as with it off** --
0.000002 .. 0.000776 w.  A single curved facet has no gap in front of it and a
plano exit face has no coefficient error, so `U . grad E == 0` at every
surface.  **The 155-891x ladder of the previous build is untouched by
construction, not by tuning.**

### 2.3 The ablation, on g5

| corrected at | eq (4) | **eq (4) + R1** |
|---|---|---|
| nothing (shipped) | 0.25848 | 0.25848 |
| surface 0 only | 0.30899 | 0.30899 |
| surface 1 only | 0.26578 | 0.26098 |
| surface 2 only | 0.04233 | 0.07586 |
| surfaces 0, 1 | 0.31722 | 0.31208 |
| surfaces 0, 2 | 0.08148 | **0.01405** |
| surfaces 1, 2 | 0.04559 | 0.07149 |
| **surfaces 0, 1, 2** | 0.09069 | **0.01240** |
| tangent facet, no translation OPD | 0.29119 | -- |
| tangent facet + translation OPD (2) | 0.000372 | -- |

Surface 0 carries no R1 term (no preceding gap), so the `(0,)` and `(none)`
rows are identical to the previous build's -- the arithmetic check that the
term is where the derivation says it is.

### 2.4 The sign control

Same control equation (4) carries, on a two-powered-surface element
(R = +/-25 mm N-SSK2, 4 mm thick, 3 mm pupil) where R1 is nonzero:

| tilt | eq (4) alone | **+ R1** | **- R1 (negated)** | negated / eq (4) |
|---|---|---|---|---|
| 54.9 mrad | 0.018966 | **0.005698** | 0.043021 | 2.27x |
| 90 mrad | 0.032898 | **0.011508** | 0.074988 | 2.28x |
| 150 mrad | 0.062676 | **0.027155** | 0.144174 | 2.30x |

Negating lands ABOVE the no-R1 arm at every tilt -- and at a stable 2.27-2.30x
of it across a 3x angle range, which is the "wrong by twice the term" signature
of a correction that otherwise cancels the defect it targets, not merely one of
the right size (`test_the_r1_sign_is_load_bearing`).

### 2.5 Where the correction does NOT help, recorded

An R = +/-12 mm biconvex N-SSK2 element (slope 0.25 at 3 mm) at large angle is
outside the analytic model's envelope, and equation (4) alone makes it WORSE
there.  R1 recovers it, but the whole thing is 0.4 waves off truth and the
guard exists to say so:

| tilt | blind | eq (4) | + R1 |
|---|---|---|---|
| 20 mrad | 0.048735 | 0.080936 | **0.011037** |
| 54.9 mrad | 0.144106 | 0.226307 | **0.080976** |
| 90 mrad | 0.266234 | 0.383993 | **0.219290** |
| 120 mrad | 0.399269 | 0.533027 | **0.394817** |

**R1 improves on equation (4) in every row measured in this campaign except
design 121 group 2 at the 1 mm and 2 mm pupils** (S2.1: 0.000020 -> 0.000048
and 0.000159 -> 0.000204 waves, i.e. 1/500 of the tolerance on both arms) --
including, here, the rows where equation (4) is itself a regression.

---

## 3. THE GUARD, RECALIBRATED

`_SCREEN_OBLIQUITY_RESIDUAL_FRAC` **0.40 -> 0.10**.

* The ESTIMATOR is unchanged and still scores equation (4) alone: 0.23910
  waves on d121 g5 against the exact-ray truth 0.25848 (7.5 % low), silent on
  the other five groups.  Scoring `eq (4) + R1` instead was measured and
  REJECTED: it reads 0.39530 w against a 0.25848 truth, because R1 is a second
  correction to the SAME defect and partially cancels the first, so summing
  their magnitudes double-counts.
* The BUDGET is what moved.  Worst measured residual/uncorrected ratio across
  the campaign's powered cases is now **0.048** (g5, the binding case) and
  0.055 on g4 where the error is 0.0005 w and the ratio is not meaningful;
  single facets read 0.001-0.007.  Rounded UP to 0.10 -- a 2x margin, because
  the leftover is now the deflection channel proper, which has NOT been
  measured on a decentred / tilted / freeform element.

**One pin moved, and it is adjudicated against the oracle rather than
retuned.**  `test_guard_fires_on_the_steep_large_angle_case` used an
R = 15 mm plano singlet at 90 mrad, whose estimate is 0.1286 w: at 40 % that
budgeted 0.0514 w and fired, at 10 % it budgets 0.0129 w and does not.  The
exact-ray truth on that fixture is **0.001381 waves** -- 36x INSIDE the
lambda/20 tolerance -- so the old warning was a false alarm by 37x and silence
is the correct behaviour.  Replaced by two tests:

* `test_the_guard_does_not_warn_about_a_call_the_correction_fixed` -- asserts
  BOTH the measured 0.0014 w and the silence, so neither a re-loosened budget
  nor a regressed correction can pass;
* `test_guard_fires_with_the_correction_ON_when_it_cannot_rescue_the_call` --
  the ON-branch on the S2.5 biconvex at 120 mrad, where the corrected truth is
  **0.395 w**, 8x over the tolerance.  `test_guard_policies` moved to the same
  fixture for the same reason.

---

## 4. THE WIRING (ITEM 2)

`VERIFY_ARCHITECTURE` F5/P2-9 stands as reported: before this branch, zero
`apply_real_lens` calls anywhere in the library forwarded `carrier=`.  Every
call site was surveyed -- an AST sweep of `lumenairy/` for real `Call` nodes
named `apply_real_lens`, not a grep over prose, which finds **34 call sites in
13 modules** -- and each is decided below.

| # | call site | caller has a carrier? | verdict |
|---|---|---|---|
| 1 | `_lens_traced.py:8018` delegate branch (`on_noncollimated='delegate'`) | YES | **FORWARDED**, gated on the F1 statistic (S4.1) |
| 2 | `_lens_traced.py:8147` `_amp_call` (parallel, `E_in`) | yes | **adjudicated NOT** (S4.2) |
| 3 | `_lens_traced.py:8172` `_amp_pw_call` (parallel, reference) | yes | **adjudicated NOT** (S4.2) |
| 4 | `_lens_traced.py:8192` sequential amp (`E_in`) | yes | **adjudicated NOT** (S4.2) |
| 5 | `_lens_traced.py:8221` sequential amp (reference) | yes | **adjudicated NOT** (S4.2) |
| 6 | `_lens_traced.py:11747` `PreparedTracedLens.__call__` | yes | **adjudicated NOT** -- same leg as 2-5, and its screen is input-independent by construction (S4.2) |
| 7 | `propagators/system.py:847` `'real_lens'` chain element | YES (element dict) | **FORWARDED** (S4.3) |
| 8 | `elements/polarization.py:457/466` `JonesField.apply_real_lens` | not exposed | **SURFACE ADDED + forwarded to both components** (S4.4) |
| 9 | `propagators/fga.py:3066/3070` router analytic members | no (documented: "this router cannot provide it, it has no q-trace") | not forwarded; **already reachable** -- the router splats caller `**extra` into the call, so `carrier=` passes through today |
| 10 | `analysis/coherence.py:113/175` | no -- illumination realisations, no congruence statement | nothing to forward |
| 11 | `analysis/polychromatic.py:207/381` | no | nothing to forward |
| 12 | `analysis/through_focus.py` (8 sites) | no | nothing to forward |
| 13 | `optimize/driver.py:143`, `optimize/wrapper_merits.py` (3 sites) | no | nothing to forward |
| 14 | `elements/_lens_jax.py` (3 sites) | no -- the JAX twin's parity references | nothing to forward; forwarding would break the twin's parity target |
| 15 | `elements/_lens_real.py:298/301` (the float32-sag self-check's own two arms) | no | internal control, must stay carrier-free |
| 16 | `ui/waveoptics_dock.py:730`, `ui/jones_pupil_dock.py:319` (+ `_context.py` / `io/codegen.py`, which EMIT a call rather than make one) | no | interactive / generated code, no congruence to state |

### 4.1 The delegate branch, and the wrong-carrier adjudication

This is the one site where the analytic model IS the answer -- it `return`s
`apply_real_lens(...)` and there is no traced leg behind it.  It is also, by
construction, the site where the carrier is least trustworthy: it fires
because `_carrier_residual_rms(E_in, W) > 0.02 rad`, i.e. because the carrier
does not describe the field.

Forwarding a wrong carrier IS worse than none: the correction's cross term
`2 p0 . q` flips with the carrier's sign, which is the same "wrong by twice
the term" signature the refutation used to kill the previous design.  So the
forward is gated on the guard's OWN statistic, not a new heuristic:

```
    forward iff  _carrier_residual_rms(E_in, W)  <  _input_tilt_stats(E_in)[0]
```

-- the residual WITH the carrier removed versus the same statistic with
NOTHING removed, which is exactly the `q = 0` arm.  Measured on a tilted +
strongly-diverging beam (`lam/2dx = 0.164 rad`, so both arms are inside the
wrapping-safe estimator's range):

| carrier | residual | raw | forwarded? | delegate output |
|---|---|---|---|---|
| `+0.05` (right) | 0.0366 | 0.0619 | **YES** | byte-identical to `apply_real_lens(carrier=...)` |
| `-0.05` (reversed) | 0.1065 | 0.0619 | **NO** | byte-identical to `apply_real_lens()` (carrier-free) |

Both are pinned (`test_the_delegate_branch_forwards_a_carrier_that_describes_
the_field`, `..._refuses_a_carrier_that_does_not_describe_it`).  The warning
now states which way it went instead of listing `carrier` among the
DISCARDED arguments; a delegate call with a carrier still always warns, so no
caller loses the model-swap notice.

**Known limit, not introduced here:** the F1 estimator is a nearest-neighbour
phase-increment statistic, so it aliases above `lambda / (2 dx)`.  Above that
pitch both arms of the comparison are unreliable -- the same caveat the
existing guard carries.

### 4.2 The amp legs: adjudicated NOT to forward, and measured

The traced exit is

```
    E_out = E_analytic * exp( i ( k0 * opl_traced - phase_analytic_lens ) )
```

with `E_analytic = apply_real_lens(E_in)` and `phase_analytic_lens =
angle(apply_real_lens(reference))`.  The analytic lens phase therefore enters
TWICE with opposite signs and is designed to cancel -- the traced OPL replaces
it.  Improving it is at best a no-op.  Measured on an R = 19.6 mm N-SSK2
singlet at 100 mrad, N = 1536, `dx = 4 um`, phase rms over the probe disc:

| what was forwarded | change to the traced output |
|---|---|
| all amp legs (symmetric) | **8.676e-05 waves** |
| the `E_in` leg only (asymmetric) | **4.250e-03 waves** |
| (for scale: the correction's own effect on the analytic field) | 4.250e-03 waves |

**49x cancellation when symmetric; the FULL correction injected when not.**
So the hazard is not "forwarding is useless", it is "forwarding half of it
corrupts an answer that is already exact" -- and the reference leg is
deliberately INPUT-INDEPENDENT (built on a `ones` placeholder) so the
prepared-screen and multi-order reuse paths stay byte-identical, which a
carrier would break.  Cost is real too: 2.5x -> 3.2x of the analytic wall
clock, on two full-grid passes, for 8.7e-05 waves.

Pinned structurally by `test_the_amp_legs_do_not_forward_the_carrier`, which
spies on every amp-leg call and asserts none of them was handed a `carrier`.

### 4.3 The system chain's `'real_lens'` element

A chain element dict accepted a `carrier` key and DROPPED IT IN SILENCE -- the
same class v5.31's W9-11 closed for the `'real_lens_traced'` sibling, where
the key IS honoured.  `carrier` / `screen_obliquity` / `on_screen_obliquity`
now join the forwarded set; absent them the call is byte-identical to before.
Pinned by `test_the_system_chain_forwards_a_real_lens_elements_carrier`.

### 4.4 `JonesField.apply_real_lens`

`BUILD_SCREEN_OBLIQUITY` S9 recorded this as deliberately NOT widened.  With
the wiring decision made, it is widened: the signature gains `carrier` /
`screen_obliquity` / `on_screen_obliquity`, all defaulting to the pre-5.35
behaviour, and forwards ONE carrier to both components -- which is the same
statement the module already makes ("Ex and Ey are dispatched with the same
carrier").  The guard is silenced on the Ey leg only, because it would print
the identical estimate for the identical congruence twice.  Two pins: each
component byte-identical to the scalar call, and byte-identical to pre-5.35
without a carrier.

---

## 5. COST AND SCOPE

* **Cost, measured** on a three-surface cemented element, best of 3, guard on,
  same fixture on both trees:

  | N | carrier-free | carriered, eq (4) only (base) | carriered, + R1 |
  |---|---|---|---|
  | 512 | 0.127 s | 0.370 s (2.48x) | 0.406 s (**3.19x**) |
  | 1024 | 0.590 s | 1.624 s (2.56x) | 1.855 s (**3.15x**) |

  **R1 adds ~0.6-0.7x of the carrier-free call** on top of equation (4)'s
  ~1.5x.  Per powered surface it is one extra `_facet_axial_momenta`, one
  `gradient` of `E`, and two multiply-adds; per GAP it is two `gradient` calls
  on the `p0` accumulator plus ~10 grid ops.  No ray trace, no fit, no cache,
  no I/O.
* **Memory**: two more full-grid arrays (`_obl_ux` / `_obl_uy`), and they stay
  SCALAR (a numpy 0-d) until a POWERED surface makes `p0` a field -- so a
  leading plate, a single facet and every zero-angle carrier allocate nothing.
  The gradients R1 needs are transients, `del`-ed inside the loop body.
* **Path**: unchanged -- `_obl_active` already routes the surface loop to the
  whole-grid path (the row-banded sag path carries no halo for a gradient).
* **`_split_mode`**: the drift uses the model's OWN gap geometry, so the
  paraxial-equivalent factorisation drifts through its reduced distance `t/n`
  in air rather than `t` in glass.
* **Refusals unchanged**: `screen_obliquity=True` without a carrier still
  raises; `carrier=` with `surface_model='displaced'` still raises.
  Non-propagating pixels take a ZERO drift step rather than a clamped one.

---

## 6. THE COMPATIBILITY / ADJUDICATION LEDGER

| item | status | evidence |
|---|---|---|
| carrier-free byte-null CROSS-TREE vs the branch base (`fix/verify-arch` @ `0f46efb`, worktree `C:/tmp/lum_varch`) | **HELD on 10/10 option combinations** | a conic+asphere singlet (both faces powered, both aspheric), sha256 of the complex exit field: `default`, `slant_correction`, `fresnel`, both, `sag_chunk_rows=32`, `seidel_correction`, `surface_model='displaced'`, `displaced`+`conjugate=0.5`, `sag_dtype=float32`, complex64 input -- all identical |
| carrier-free byte-null | **HELD** | `test_carrier_free_call_is_byte_identical`; R1 is skipped structurally (`_obl_drift_live` is never set) |
| zero-angle-carrier byte-null | **HELD** | `test_zero_angle_carrier_is_byte_identical`; `_obl_q_zero` blocks the drift, and `_screen_drift_step` returns EXACT zeros anyway (`test_r1_is_exactly_zero_where_there_is_no_drift`) |
| plane-plate byte-null WITH a carrier | **HELD at 5 tilts to 100 mrad** | `test_plane_plate_correction_is_exactly_zero`; `test_r1_is_zero_on_a_plate_at_every_tilt` on the shipped expression |
| the plate's analytic walk-off | **EXACT** | `test_the_drift_is_the_carriers_own_transverse_walk` pins `t q / sqrt(n^2-q^2)` to 1e-18 |
| single-facet ladder (the 155-891x table) | **UNCHANGED, structurally** | `test_r1_does_not_move_a_single_facet_element`; `sphere` mode reprints the same 20 rows |
| **design-121 acceptance banner** | **BYTE-IDENTICAL to the base** | sha256 of the traced exit field on groups 2 and 5, `apply_real_lens_traced` with the chain's own kwargs: `9f5614bb...` / `8067eb98...` on both trees (S0.8) |
| `test_guard_fires_on_the_steep_large_angle_case` | **MOVED, adjudicated** | oracle says 0.001381 w, 36x inside tolerance -> silence is correct; replaced by two tests, one asserting the measurement (S3) |
| `test_guard_policies` fixture | **MOVED with it** | same reason; now on a fixture whose corrected truth is 0.395 w |
| the guard's behaviour ON DESIGN 121 | **estimate unchanged (0.23910 w), group 5's ON-branch now SILENT** | `_screen_obl_guard.json` moves by exactly one field, `fires_on: true -> false` on g5.  Correct: the corrected residual there is 0.012398 w, 4x inside the lambda/20 tolerance.  `fires_off` still true on g5 and the other five groups are still silent on both arms |
| `E-M2` / `E-L22` delegate contracts | **PASS unchanged** | `carrier=None` in both, so the new branch is not taken; the file goes 36 -> 39 passed, all three new |
| relaxed / retargeted assertions elsewhere | **ZERO** | -- |
| `xfail` / `skip` added | **ZERO** | -- |
| `prepare_real_lens` / `PreparedAnalyticLens` | **UNTOUCHED** | no new kwarg; a prepared screen is input-independent and a carrier is not |
| `CHANGELOG` | **NOT TOUCHED** | as instructed |

---

## 7. FILES

| file | change |
|---|---|
| `lumenairy/elements/_lens_real.py` | the R1 derivation block (eqs 5-7), `_screen_coeff_error` / `_screen_drift_step` / `_screen_drift_opd`, `_SCREEN_DRIFT_MIN_PZ_SQ`, the drift accumulators + gap advance in `apply_real_lens`, `_SCREEN_OBLIQUITY_RESIDUAL_FRAC` 0.40 -> 0.10, guard message |
| `lumenairy/elements/_lens_traced.py` | the delegate branch's adjudicated `carrier=` forward + its warning |
| `lumenairy/propagators/system.py` | `'real_lens'` element forwards `carrier` / `screen_obliquity` / `on_screen_obliquity` |
| `lumenairy/elements/polarization.py` | `JonesField.apply_real_lens` gains the three keywords |
| `tests/unit/test_screen_obliquity.py` | R1-aware exact-ray arm; 8 R1 pins, 2 guard adjudications, 3 consumer-reachability pins (34 -> 47) |
| `tests/unit/test_niche_audit_e_prepared_and_enums.py` | 3 wiring pins (36 -> 39) |
| `validation/repro_traced_carrier_121/screen_obliquity_derive.py` | `coeff_error` / `carrier_drift` / `r1_opd`, an `r1=` arm through `screen_trace` / `model_error` / `angular_error`, and the R1 column in `plate` / `sphere` / `d121` / `ablate` |
| `validation/repro_traced_carrier_121/_screen_obl_{d121,ablate,sphere,guard}.json` | results of record, regenerated, and the diffs are ADDITIVE: `_screen_obl_plate.json` is unchanged on disk; `sphere` gains only `r1_rms` (== `snell_rms` on 20/20 rows); `d121` gains only `r1_rms` / `r1_max`; `ablate` gains 8 R1 rows; `guard` moves exactly one field (`fires_on` on g5).  **ZERO pre-existing values moved in any of them**, checked by row identity rather than by position |
| `docs/audits/BUILD_R1_WIRING_2026_08_12.md` | this note |

---

## 8. WHAT IS NOT CLAIMED

* **The tangent-facet arm's 0.000372 w is not reached, and cannot be by a
  screen.**  R1 lands on 0.012398, which IS the measured perfect-OPD /
  blind-kick floor (0.012477).  The remaining 33x is the deflection channel
  proper; S1.5 gives the curl that forbids it.
* **R1 is measured only on rotationally symmetric surfaces.**  It takes
  `grad sag` and `grad p0` from `xp.gradient` AFTER the decenter shift, tilt
  ramp, form-error map and freeform departure are folded into `sag`, so it is
  structurally correct for those -- but no oracle run was made on a decentred,
  tilted, biconic or freeform element.
* **Nothing is measured on the GPU path.**  The code is `xp`-generic and
  `cupy` has `gradient`, but no CuPy run was made.
* **The delegate's F1 gate inherits the F1 estimator's aliasing limit** above
  `lambda / (2 dx)`; it is not introduced here and is not fixed here.
* **The guard's estimator is still a leading-order SIZE, not a bound** (7.5 %
  low on the one case where both were measured, and 4x low on the
  out-of-envelope biconvex of S2.5).
* **`prepare_real_lens` still does not support the correction**, and the
  `'auto'` / explicit-wavefront carriers are exercised but not measured
  against an oracle -- both carried over from the previous build.

---

## 9. SUITES

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X, 24 logical CPUs
python 3.14.6   numpy 2.4.4      lumenairy 5.34.0 (worktree C:/tmp/lum_r1,
                                 branch feat/screen-r1-wiring off
                                 fix/verify-arch 0f46efb)
WSL / OpenBLAS proxy             ~/lumvenv, numpy 2.4.6, BLAS pinned to 1 thread
```

| gate | result |
|---|---|
| `test_screen_obliquity.py` (Windows) | **47 passed** (34 before) |
| `test_niche_audit_e_prepared_and_enums.py` (Windows) | **39 passed** (36 before) |
| consumer set A+B (Windows) -- 36 files: every `tests/unit` file with a real `apply_real_lens(` call site plus `test_niche_c13_lstsq_conditioning` / `test_niche_c14_encapsulation` / `test_audit_lens` / `test_audit_polarization` / `test_v5_1_0_agent_e_split` / `test_niche_audit_w9_dispatch` | **1231 passed** in 2267.4 s |
| final-tree re-verify (Windows) -- the FOUR files that actually exercise `carrier=`: the two changed ones plus `test_niche_d9_grid_origin` and `test_niche_p1_traced_tiltaware` | **115 passed** in 89.3 s |
| WSL / OpenBLAS proxy, 11 files (the two changed files + c13 + c14 + `test_audit_lens` + the system-chain, dispatch, polarization and hammer H1 / H2 / H6 files) | **292 passed, 4 skipped** in 1540.7 s -- the 4 skips are `test_audit_lens.py`'s pre-existing `PySide6` GUI-dep skips, not added here |
| WSL re-verify on the FINAL tree (the two changed test files) | **86 passed** in 350.2 s |
| `ruff check lumenairy/ tests/unit/` + the validation file, Windows AND WSL | **All checks passed** |
| cp1252 decode + pure-ASCII added lines, all 8 changed files | **clean** |
| `xfail` / `skip` added by this work | **ZERO** |
| `CHANGELOG` | **NOT TOUCHED** |

Reproducing the study:

```
cd validation/repro_traced_carrier_121
python screen_obliquity_derive.py plate    # the plate zero, unchanged by R1
python screen_obliquity_derive.py sphere   # single facets, unchanged by R1
python screen_obliquity_derive.py d121     # S2.1, blind / eq (4) / +R1
python screen_obliquity_derive.py ablate   # S2.3, per-surface + tangent facet
python screen_obliquity_derive.py guard    # S3, the estimator on d121
```
