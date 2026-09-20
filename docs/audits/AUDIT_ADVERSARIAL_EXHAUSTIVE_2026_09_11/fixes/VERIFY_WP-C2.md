# VERIFY-WP-C2 -- independent adversarial verification of the ray tracer's two default flips

Subject: branch `feat/c2-analytic-normal-default`, ten commits `bdb3ae73`..`eadc67ba`
on `49ddf4bd`; report `WP-C2_ANALYTIC_NORMAL_REPORT.md`; tests
`tests/unit/test_c2_analytic_normal_default.py`; probes
`validation/probe_c2_analytic_normal/`.

Verification worktree `C:/tmp/lum_vc2`, branch `verify/c2-analytic-normal`.
PRE tree = this verifier's own `git archive 49ddf4bd` extracted read-only to
`C:/tmp/lum_vc2_pre`.  Builds: **Windows py3.14.6 / numpy 2.4.4** and
**WSL py3.12.3 / numpy 2.4.6**, every probe on both, every probe in a child
process with `LUMENAIRY_ROOT` on `sys.path` and `lumenairy.__file__` asserted
inside it, all with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line.  Nothing below is read from the WP-C2
report: every number is re-measured on this verifier's own sphere set, its own
60-digit oracles and its own prescriptions.  Evidence:
`validation/probe_verify_c2/` (11 probes, JSON per build).

---

## 0. Verdict table

| # | Claim under test | Verdict | This verification's numbers |
|---|---|---|---|
| 1 | closed form within 1.75 ULP to `0.95 R`, never worse by > 1 ULP there; above it neither dominates | **CONFIRMED, with better numbers** | on 2240 of my own points and an EXACT-input oracle: **1.00 vs 1.75** out to `0.95 R`, **0 of 1280** worse by > 1; whole set **41.47 vs 75.66**; 28 of 2240 worse, by up to 16.08.  Both builds identical to the last digit |
| 2a | what `renormalize='exit'` changes numerically | **ANSWERED** (the report does not) | history `abs(abs(d)-1)` 2.22e-16 -> 6.66e-16 (3 surf) .. **1.67e-15** (13); the ray-sphere quadratic hard-codes `a = abs(d)**2 = 1` and the OPL leg is `n t`, so the drift is a FIRST-ORDER error with a measured, LINEAR coefficient: **1.786e-3 m** position and **8.95e-4 m** OPL per unit drift -> induced error **3.0e-18 m / 1.5e-18 m** at 13 surfaces |
| 2b | against a 60-digit END-TO-END oracle, is `'exit'` worse? | **NO** | all four combinations within **5.2e-18 m** position and **7.0e-17 m** OPL of the truth on three prescriptions; which is closest flips with prescription AND build |
| 2c | WP-B9's 1.03x-1.10x "does not reproduce" | **CONFIRMED as a measurement, and now BOUNDED** | timed, sphere-bearing prescriptions: 0.889x-1.088x (Win, median 1.006x), 0.990x-1.053x (WSL, median 0.997x).  This switch has NO control arm -- it applies to every prescription, sphere or not -- so the conic and aspheric prescriptions are extra samples, not controls, and they read 1.076x / 1.660x (Win) and 1.000x / 1.056x (WSL), i.e. the spread of the METHOD swamps the effect.  Deterministic element-op count: **0.9910x at 2 surfaces (a LOSS), 1.0050x at 3, 1.0189x at 7, 1.0237x at 13**.  B9's floor is reachable; **its ceiling of 1.10x is not** |
| 2d | KEEP or REVERT `44151397` | **KEEP** -- see section 3 | |
| 3 | analytic normal 1.08x-1.44x, profile share 16.1 % -> 9.8 % / 19.6 % -> 10.4 % | **CONFIRMED** | CPU **1.121x-1.359x median 1.147x** (Win), **1.105x-1.224x median 1.181x** (WSL); profile share **10.04-11.67 % -> 7.26-9.80 %** (Win), **10.31-13.61 % -> 6.63-10.43 %** (WSL); element-op count **1.1612x-1.1973x**, controls **exactly 1.0000x** |
| 4a | the rim band, and 360 000 rays move nothing | **CONFIRMED, and the band is NARROWER than stated** | both gates LOCATED by bisection are the **same float** (`0.9999499987499374`) at all 8 radii ON THE AXIS -- the band is not an annulus, it exists only at azimuths where the two expressions round apart.  Straddle found at 2 of 6 radii, reaches `_refract` (code 4 vs 0) and `trace` (code 4 vs 1).  **580 000 rays over twelve combinations: zero alive flags, zero error codes**, both builds, against both the new pair and the full pre-5.49.0 pair |
| 4b | can a REAL design land in the band? | **ANSWERED** | a fast singlet cannot (`max h/abs(R) = 0.495` at f/1).  A **BALL LENS** and a **HEMISPHERE** -- catalogue parts with semi-diameter `abs(R)` -- reach 0.999999 and DO cross the clamp: **2930 of 60 000** rim-packed rays die RAY_NAN on BOTH routes.  The 1-ULP straddle inside that region is not reachable by sampling (0 of 580 000 within 4 ULP of the gate).  **Migration-note fact: the reachable part is the PRE-EXISTING clamp, not the band** |
| 5a | the way back is byte-identical to the pre-5.49.0 arithmetic | **CONFIRMED, archive to archive** | **594 of 594 arrays identical** on both builds -- the PRE tree (this verifier's own `git archive 49ddf4bd`, its own root, its own process) against the branch with both old keywords forced, over five prescriptions x two fields x `'last'` and `'all'` with every history bundle recorded.  At the DEFAULTS the same set reads 172/594 identical, 422 moved, worst 3.3e-16 |
| 5b | the SIX entry points with no way back | **REFUTED -- there are sixteen** | an AST census of the package finds **21** exported, directly-tracing entry points carrying neither keyword; 5 reach `trace_jax` (which has neither switch by design), leaving **16 CPU-affected** against the report's six.  A further **46** reach a tracer transitively. **DEFECT D4** |
| 6a | CPU/JAX parity 3.5e-18 m unmoved under all four CPU settings | **CONFIRMED** | **3.469e-18 m** position, 5.55e-17 m (Win) / 5.20e-17 m (WSL) OPL, alive masks equal, **identical under all five CPU settings** including the library default |
| 6b | does the JAX tracer have the same clamp? | **NO -- and the backends' vignetting bands differ by a whole annulus** | `_refract_jax` applies NO domain gate.  Ball lens, 40 000 rim-packed rays: **1991 past the clamp; CPU kills all 1991 on both routes, JAX keeps all 1991**.  Pre-existing, unpinned. **DEFECT D6** |
| 7a | both B9 reports' "basin flip / bimodal" mechanism is wrong | **CONFIRMED independently** | **71.5-75.6 %** of the 1024 pixels are above 10 % of the maximum where a one-knife-edge-pixel story predicts **0.098 %**; 91.5 % above 1 %; median/max 0.208; exactly ONE pixel within 1 % of the max |
| 7b | `w6_a2`'s root is genuinely off centre by `H^-1 r(v_c)` | **CONFIRMED** | offset == the one-Newton step to **9.86e-32**; the float64 step agrees with a 60-digit `mpmath` solve to **3.08e-33**; a SECOND Newton step is **2.0e-7** of the first; `abs(r(v_c))` 8.415e-08, `sigma_min(H)` 9.7413e+07, normalised offset 4.2723e-14 against a derived bound of 4.2737e-14 |
| 7c | the restated pins' bars are runtime-derived with 11.1x-13.7x margins | **PARTLY -- the margin is seed-dependent** | `kappa` depends on the RANDOM DIRECTION `_conditioning_bar` draws: shipped seed **6.2423e+06**, a different seed **1.6406e+06** (both builds, ladder linear to 1.2 %).  The margin the report gives as 11.1x-13.7x reads **2.92x-3.59x** with the other direction. **DEFECT D2** |
| 7d | `trace`'s `<= 1e-15` history bound was a reading and is exceeded | **CONFIRMED** | exceeded at **SEVEN** surfaces on my ladder (1.22e-15), 1.67e-15 at 13.  The derived `n_surfaces * eps` envelope HOLDS on every rung, at ratios 1.00 (n=3) down to 0.577 (n=13) -- so the corrected docstring's "about 0.6 n eps" is a long-end reading, not the shape |
| 8 | the two d3 movers, restated against a one-ULP floor | **READINGS REPRODUCE; the bar has no gap below** | `moved` 0.8360 vs 0.836, floor 0.1404 vs 0.1404, degree effect 0.16894 vs 0.1689, floor 0.002586 vs 0.0026.  Floors ARE measured in process; the multipliers (10x, 3x) are CHOSEN; neither arm is two-sided.  **The arm-2 floor's own spread over four one-ULP directions is 3.22x (Win) / 4.79x (WSL) -- at or above the 3.0 multiplier on it. DEFECT D8** |
| 9 | `analysis.ghost` keeps the generic normal | **CONFIRMED, and it is a defect** | three 2-bounce ghost paths of a spherical doublet move by up to **2.13e-14 mm** of RMS spot radius (normal only) / 4.26e-14 mm (both keywords); transmittance, energy fraction and ray counts unchanged. **DEFECT D5** |
| 10 | the `EDITED_IN_PLACE` map is guarded | **PARTLY -- the guard is one-sided** | it REFUSES an unrelated line and an out-of-range coordinate, and it ACCEPTS a silently reverted default, a nonsense value, and a stale copy left at the mapped line.  It is not version-pinned. **DEFECT D7** |
| 11 | nine pre-existing reds at 49ddf4bd | **9 of 9 CONFIRMED on my own archive** | eight fail identically on `git archive 49ddf4bd` under the Windows build; the ninth (`test_installed_metadata_version_matches_source_version`) is green on the Windows mount and RED on the WSL one -- installed metadata 5.11.0 against a source 5.48.1 -- and red on the ARCHIVE there too, so it is a stale editable install on that mount, pre-existing exactly as the report classifies it |
| 12 | the mutation matrix | **8 of 9 caught; one gap** | defaults reverted (signature AND in-loop), clamp moved, mirror `nz` flipped, `nz` flipped everywhere, clamp on the analytic route only, exit rescale never runs, predicate accepts conics -- **all caught, both builds**.  The report's signature/call SEPARATION is real: an in-loop revert wrapped in `functools.wraps`, so `inspect.signature` sees nothing, is caught first by `..._defaults_are_what_a_call_actually_takes` (and by three others), not by the signature arm.  **`jax_gets_a_clamp` survives 294 raytrace and parity tests** (the mutation was verified to bite: 1901 of 2000 alive instead of 2000).  `whole_normal_sign_flipped` survives CORRECTLY -- a negative control, since the shared core orients the normal against the ray |

---

## 1. What this verification did differently

Four instruments the shipped work package does not have, and each of them
changed an answer.

**(a) An exact-input oracle.**  `probe_c2_analytic_normal/sphere_oracle.py`
converts its inputs with `ctx.create_decimal(repr(float(x)))` -- the shortest
ROUND-TRIPPING decimal, not the exact binary value the library was handed.
`Decimal(float(x))` is exact; `Decimal(repr(float(x)))` is not, and differs
from it by up to half an ULP of the input.  `nz = sqrt(1 - u)` amplifies a
relative input perturbation by `u / (2 (1 - u))`, so that conversion alone
contributes, in the probe's own units:

| `h/abs(R)` | 0.5 | 0.9 | 0.95 | 0.99 | 0.999 | 0.9999 | 0.99994 |
|---|---|---|---|---|---|---|---|
| oracle's own input-conversion error | 0.50 | 0.75 | **1.00** | **4.25** | 8.59 | 25.77 | **41.58** |
| the closed form's measured error (exact-input oracle) | 0.50 | 1.00 | 1.00 | 2.50 | 11.25 | 32.24 | 41.47 |

Above `h = 0.9 abs(R)` the shipped probe is measuring its own oracle as much as
the library.  Its headline "1.75 ULP at 0.95" and "57 vs 76 at the clamp" are
contaminated by it.  Re-measured with the exact conversion the conclusion does
not just survive, it strengthens: **1.00 against 1.75 out to 0.95 abs(R)** and
**41.47 against 75.66** over the whole set.  The oracle is shown converged
(`VC2_PREC=120` reproduces every summary field of `VC2_PREC=60` exactly).

**(b) An end-to-end trace oracle.**  The shipped ladder measures the
DIFFERENCE between the two renormalise modes, which says how far apart they
are but not which is closer to the truth.  `vc2_trace_truth.py` traces the
same geometry in `decimal` at 60 digits -- ray-sphere intersection, vector
Snell with the outward sphere normal, vertex-plane transfer, `opd += n t` --
fed exactly-unit axial rays.  (Only an axial launch has an exactly-unit
float64 direction: a dyadic `(L, N)` with `L**2 + N**2 = 1` exactly forces
`L = 0`.  Height is swept instead, which is the marginal-ray sweep the rim
question needs anyway.)

**(c) A deterministic cost instrument.**  Three timing instruments were tried
against `renormalize` on this box and all three moved under load:

* batched CPU time -- Windows' process clock ticks at 15.6 ms, which on a
  3.5 ms body quantises at 0.9 % and read the 9.6 % renormalise block as
  **exactly zero**;
* wall-clock minima -- the SAME two arms came out 9.6 % apart on one run and
  5.7 % apart **with the sign reversed** on the next.  `renormalize=True`
  cannot be faster than `renormalize=False`; it does strictly more work.  That
  reading is the instrument failing, and the WSL run reproduced the inversion;
* cProfile shares -- stable, but blind to code inlined in `_refract`.

`vc2_opcount.py` does not measure time at all.  An `ndarray` subclass
implementing BOTH `__array_ufunc__` and `__array_function__` counts every
element-wise operation.  (`__array_ufunc__` alone is not enough: `np.where`
dispatches through `__array_function__` and, without it, returns a BASE
`ndarray`, after which everything downstream is invisible -- which
under-counts exactly the branch that uses `np.where` most and makes the census
come out backwards.  `np.asarray` on a subclass returns a base array by
contract and `_surface_sag_derivative` opens with that call, so a shim
re-views it.)  The result is a COUNT, identical on both builds to the last
digit, and it reproduces the code exactly: at seven surfaces the delta is
**+18 divide, +6 maximum, -3 square, -2 add, -1 sqrt**, which is
`n_refracting * (1 maximum + 3 divides)` removed against one
`_normalize_directions` (3 squares + 2 adds + 1 sqrt + 1 maximum + 3 divides)
added.

**(d) A census taken from the package.**  `vc2_entrypoints.py` walks every
module's AST rather than reading the report's list.  A bare NAME counts as
well as a call, because `ray_fan_data` PASSES `trace` to a helper rather than
calling it -- a census that reads only `Call` nodes misses exactly that shape,
and misses `ray_fan_data`, `opd_fan_data` and both `_world` twins with it.

---

## 2. Item by item

### 2.1 The normal (items 1 and 3)

My sphere set shares nothing with the C2 probe's: eight radius MAGNITUDES
(1.5, 7.3, 25.0, 43.7, 62.9, 101.3, 250.0, 777.0 mm) of both signs, fourteen
heights from the vertex to `0.99994 abs(R)`, five azimuths, each evaluated
twice -- once refracting, once with the surface declared a MIRROR -- **2240
points**.

| quantity (the C2 probe's own metric: worst component error / `2**-52`) | Windows | WSL |
|---|---|---|
| closed form, worst out to `h = 0.95 abs(R)` (1280 points) | **1.00** | **1.00** |
| generic route, same | 1.75 | 1.75 |
| points where the closed form is worse there by > 1 | **0** | **0** |
| closed form, worst over the whole set | 41.47 | 41.47 |
| generic route, same | 75.66 | 75.66 |
| points where the closed form rounds worse (whole set) | 28 of 2240 | 28 |
| worst deficit | 16.08 | 16.08 |
| unit-vector defect of the closed form | 1.0 ULP | 1.0 ULP |
| mirror changes the normal | **no** (0 of 560 pairs) | no |
| domain-gate disagreements on this grid | 0 | 0 |
| `_surface_normal(analytic_sphere=True)` == `_sphere_normal` | yes, every point | yes |

A note on the ULP unit, because it matters to how the claim reads.  Both the
shipped probe and this one divide the worst ABSOLUTE component error by a
fixed `2**-52`, i.e. an ULP at 1.0.  Measured in ULPs of each COMPONENT
instead, the closed form reads 3.86 out to `0.95 abs(R)` against the generic
route's 6.74, and there are 16 of 1280 points where the closed form is worse
by up to 3 component-ULPs.  Neither reading contradicts the other; for a unit
direction vector the error relative to 1 is the right measure, so the
report's metric is the right one -- but the docstring and CHANGELOG say "ULP"
without saying which, and a reader checking the number with `math.ulp` will
get a different answer.

**Timing.**  Three instruments, all on my own five prescriptions:

| instrument | Windows | WSL |
|---|---|---|
| CPU time, min over 7 interleaved repeats of a >= 1.5 s batch | **1.121x .. 1.359x, median 1.147x** | **1.105x .. 1.224x, median 1.181x** |
| controls (no pure sphere) | 1.023x, and one 0.818x contention spike | **1.010x and 0.990x** |
| profile share of the normal block | 10.04-11.67 % -> 7.26-9.80 % | 10.31-13.61 % -> 6.63-10.43 % |
| element-op count (deterministic) | **1.1612x .. 1.1973x** | identical |
| element-op count, controls | **exactly 1.0000x** (0 elements) | exactly 1.0000x |

The report's 1.08x-1.44x is inside this.  The WSL controls at 1.010x and
0.990x are tighter than the report's "+-7 %", which is a property of the box's
load on the day rather than of the method.

### 2.2 The rim band (item 4)

**The band is narrower than the work package says.**  Located by 80-step
bisection on the running build, the two routes' gates are the SAME float at
every radius tested ON THE AXIS:

| `R` (m) | 0.0020 | -0.0020 | 0.0125 | -0.0125 | 0.0515 | -0.1200 | 0.5 | -1.0 |
|---|---|---|---|---|---|---|---|---|
| analytic gate, `h/abs(R)` | 0.9999499987499374 | same | same | same | same | same | same | same |
| generic gate | **identical** | identical | identical | identical | identical | identical | identical | identical |
| band width | **0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The band is therefore not an annulus at `0.99995 abs(R)`: it exists only at
azimuths where `(x*x + y*y)` and `sqrt(x*x + y*y)**2` round apart, and a
bundle confined to a meridional fan cannot enter it at all.  A directed
`nextafter` walk (8 azimuths, 400 steps each side, 6 radii) finds a straddle
at **2 of the 6** radii, both negative: `R = -0.0125 m`,
`x = 0.012423829986366148`, `y = 0.0013721601473112902`,
`h/R = 0.9999499987499375`.  It reaches `_refract` (generic `alive=False,
code=4`; analytic `alive=True, code=0, L=0.7408, N=0.6667`) and the public
`trace` (`code=4` against `code=1`).  That reproduces the report's finding at
a different radius.

**Nothing that is not aimed at it lands in it.**  580 000 rays over twelve
combinations -- a ball lens rim-packed and full-fan, a hemisphere rim-packed,
an over-filled aperture at 0 and 5 degrees, `make_singlet` 20/-20 at 0 and 8
degrees and rim-packed, `make_singlet` 51.5/inf, `make_doublet` at 0 and 5
degrees and rim-packed -- move **zero** alive flags and **zero** error codes,
on both builds, against the new pair AND against the full pre-5.49.0 pair.
Zero rays landed within 4 ULP of either gate.

**Can a real design land there?**  The question has two answers and the report
gives neither.

| prescription | max `h/abs(R)` reached | crosses the clamp? |
|---|---|---|
| `make_singlet` 20/-20, 20 mm aperture (about f/1) | 0.4950 | no |
| the same at 8 degrees | 0.5150 | no |
| `make_doublet` 51.7/-34.5/-120 | 0.3464 | no |
| **ball lens**, R = 12.5 mm, semi-diameter 12.5 mm | **0.999999** | **yes -- 2930 of 60 000 rim-packed rays die RAY_NAN** |
| **hemisphere** (half-ball), same | **0.999999** | **yes, same 2930** |

A fast singlet cannot reach the clamp: at f/1 its marginal ray is at half the
radius.  A BALL LENS or HEMISPHERE can, because its clear aperture IS `abs(R)`
by construction -- and both are catalogue parts (fibre couplers, endoscope
objectives, immersion optics).  For such a design 4.9 % of a rim-packed bundle
already dies with `RAY_NAN` -- a NUMERICAL-FAULT code, not `RAY_APERTURE` --
on BOTH normal routes, before and after WP-C2.  So the Migration-note fact is
not the 1-ULP band (which sampling cannot reach) but the pre-existing clamp
that WP-C2 correctly did not move.  See **D9**.

### 2.3 The two pins (item 7)

**The mechanism.**  Both B9 reports explain the ModalAsymptotic pin as "one
knife-edge pixel changing saddle basin" on a bimodal quantity.  Measured on
the per-pixel disagreement distribution, independently of the C2 probe:

| statistic (1024 pixels) | Windows | WSL | a one-knife-edge-pixel story predicts |
|---|---|---|---|
| fraction above 10 % of the maximum | **0.715 - 0.756** | 0.718 - 0.744 | **0.00098** |
| fraction above 1 % of the maximum | 0.915 | ~0.92 | 0.00098 |
| fraction above 50 % of the maximum | 0.104 | ~0.10 | 0.00098 |
| median / maximum | 0.208 | ~0.21 | ~0 |
| pixels within 1 % of the maximum | **1** | 1 | 1 |
| fraction below 1e-12 relative | 0.027 | ~0.03 | 0.999 |

The measured shape is **730x** the knife-edge prediction.  WP-C2's correction
of both B9 reports is confirmed by an independent statistic.

**`w6_a2`.**  Confirmed in every particular, with an independent 60-digit
solve of the Gauss-Newton system and a nonlinear convergence check the C2
probe does not have:

| quantity | Windows | WSL |
|---|---|---|
| `abs(v2*)` at the fit centre, four default combinations | 4.59e-16 .. 8.64e-16 | 4.09e-16 .. 6.40e-16 |
| `offset` vs the single Newton step `-H^-1 r(v_c)` | **9.86e-32** | 0 .. 9.86e-32 |
| the float64 Newton step vs a 60-digit `mpmath` `lu_solve` of the same `H`, `r` | **3.08e-33** | `mpmath` absent |
| a SECOND Newton step from `v*`, as a fraction of the first | **2.0e-07** | 6.7e-08 .. 1.2e-06 |
| `abs(r(v_c))` | 8.415e-08 | (same shape) |
| `sigma_min(H)` | 9.7413e+07 | 9.7413e+07 |
| normalised offset / derived `abs(r)/sigma_min` bound | 0.99967 | ~1.0 |
| would the retired `1e-15` bar pass here? | **yes, all four combinations** | yes |

The last row is the useful one: on THIS box the pre-WP-B9 `1e-15` pin is green
under every default combination, while WP-B9 measured 1.150e-15 on its box.
That is what "a coin" looks like, and it confirms the restatement was
warranted -- but it also means the old pin's failure was never evidence about
the default flip.

### 2.4 The d3 movers (item 8)

Both readings reproduce to four figures on Windows.  The restatement is a
genuine improvement -- the floors are measured in process and the comparator
is like for like.  Two things it does not do:

| | arm 1 (`degree_effect > 10 x noise`) | arm 2 (`moved > 3 x noise`) |
|---|---|---|
| reading (Win / WSL) | 0.16894 / 0.16905 | 0.8360 / 0.9868 |
| shipped floor (Win / WSL) | 0.002586 / 8.42e-05 | 0.1404 / 0.1108 |
| margin | 65.3x / 2008x | 5.96x / 8.91x |
| headroom over its own bar | 6.53x / 200.8x | **1.99x** / 2.97x |
| the FLOOR's own spread over four one-ULP directions | -- | **3.22x (Win), 4.79x (WSL)** |
| margin at the WORST of those floors | -- | 4.69x / 8.59x (still passes) |
| two-sided? | no | no (the "same degree twice" control reads exactly 0.0 on both builds and is not asserted) |

Arm 2's bar multiplier (3.0) is smaller than the spread of the very floor it
sits on.  See **D8**.

---

## 3. `renormalize='exit'` (`44151397`) -- KEEP

**Recommendation: KEEP the commit.**  Not because the speed-up reproduces --
it does not, and the work package is right that it is below what this box can
resolve -- but because the three things a maintainer would weigh all come out
in its favour once they are measured rather than argued.

**Accuracy: it costs nothing measurable.**  Against a 60-digit end-to-end
`decimal` trace on three prescriptions, all four `(renormalize,
sphere_normal)` combinations land within **5.2e-18 m** in position and
**7.0e-17 m** in OPL of the truth, and which combination is CLOSEST flips with
the prescription and with the build:

| prescription | surface/generic | surface/analytic | exit/generic | exit/analytic |
|---|---|---|---|---|
| doublet, 3 surfaces (Win) `dx` | 2.17e-19 | 2.17e-19 | 2.17e-19 | 4.34e-19 |
| stack, 7 surfaces (Win) `dx` | 1.30e-18 | 1.73e-18 | 1.73e-18 | 1.95e-18 |
| ladder, 13 surfaces (Win) `dx` | 3.04e-18 | 4.34e-18 | 3.47e-18 | **1.73e-18** |
| ladder, 13 surfaces (WSL) `dx` | 1.73e-18 | 4.34e-18 | 5.20e-18 | 1.73e-18 |

The structural worry is real and is now a number.  The ray-sphere quadratic
hard-codes `a = abs(d)**2 = 1` (`intersection.py`, the Newton-skip fast path)
and the OPL leg is `opd += n * t` with `t` parametric, so a direction-norm
drift is a FIRST-ORDER error in both.  Measured by injecting a drift nine
decades above the noise and scaling back, on both builds:

* `d(position) / d(drift)` = **1.786e-3 m**, `d(OPL) / d(drift)` = **8.95e-4 m**;
* LINEAR: the coefficients from injections of 1e-12 and 1e-10 agree to 0.09 %
  and 0.40 %;
* the measured 13-surface history drift is 1.665e-15, so the induced error is
  **3.0e-18 m** in position and **1.5e-18 m** in OPL -- at the trace's own
  distance from the truth (1.7e-18 .. 5.2e-18 m), not above it.

**Speed: small, positive from three surfaces up, negative below.**  The
deterministic element-op count, identical on both builds:

| prescription | surfaces | elements saved | predicted |
|---|---|---|---|
| two mirrors | 2 | **-8 192** | **0.9910x (a LOSS)** |
| doublet | 3 | +8 192 | 1.0050x |
| stack | 5 | +40 960 | 1.0147x |
| stack | 7 | +73 728 | 1.0189x |
| stack | 9 | +106 496 | 1.0212x |
| ladder | 13 | +172 032 | **1.0237x** |

The arithmetic is exactly what the code says: the hoist removes
`n_refracting * (1 maximum + 3 in-place divides)` = `4 n` element passes and
adds one `_normalize_directions` = 10, so it breaks even between two and three
surfaces.  **WP-B9's 1.03x-1.10x therefore has a reachable floor and an
unreachable ceiling**: 1.10x is above what the change can produce on any
surface count measured, and WP-C2's inability to see the effect is consistent
with an effect of 0.5 %-2.4 %.  The profile share of the hoisted block agrees:
0.47-2.35 % of `trace`'s tottime on Windows and 0.02-1.12 % on WSL (one
Windows control reads -0.65 %, which is the profiler's own noise on a
prescription where the block is not there to be removed).

**Cost: one documented behaviour moves, and it is now correctly bounded.**
The intermediate `ray_history` bundles are no longer unit.  Measured on my own
ladder, both builds identical:

| surfaces | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|
| `'exit'`, worst history `abs(abs(d)-1)` | 6.66e-16 | 9.99e-16 | **1.22e-15** | 1.44e-15 | 1.67e-15 | 1.67e-15 |
| as a fraction of `n_surfaces * eps` | **1.00** | 0.90 | 0.79 | 0.72 | 0.68 | 0.58 |
| `'surface'`, same | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 | 2.22e-16 |
| final bundle, either mode | 1.1e-16 | 1.1e-16 | 1.1e-16 | 2.2e-16 | 1.1e-16 | 2.2e-16 |

The retired `<= 1e-15` is first exceeded at **seven** surfaces, not eight.
The `n_surfaces * eps` envelope holds on every rung.  The docstring's "about
0.6 `n_surfaces * eps`" is a reading at the LONG end: at three surfaces the
ratio is exactly 1.00, so a consumer sizing a tolerance from 0.6 would be
under by 40 % on a triplet.  See **D3**.

**So: keep it, and fix the docstring's coefficient.**  If a maintainer weighs
it the other way, the case for reverting is that the change is worth at most
+2.4 % and is negative on two-surface systems, in exchange for a history
contract that every downstream consumer of `output_filter='all'` now has to
know about.  That is a defensible call; the numbers above are what it should
be made on, and neither the accuracy nor the vignetting argues for reverting.

---

## 4. Defects

Requested edits are exact.  Nothing under `lumenairy/` was edited by this
verification.

### D1 (P2, probe) -- the shipped oracle's input conversion is not exact

`validation/probe_c2_analytic_normal/sphere_oracle.py:51`

```python
    X, Y, RR = dec(repr(float(x))), dec(repr(float(y))), dec(repr(float(R)))
```

`repr` gives the shortest ROUND-TRIPPING decimal, which differs from the
float's exact value by up to half an ULP; `sqrt(1 - u)` amplifies that by
`u / (2 (1 - u))`.  Measured contribution in the probe's own units: 1.00 at
`h = 0.95 abs(R)` (where it reports the closed form at 1.75), 4.25 at 0.99,
25.8 at 0.9999, 41.6 at the clamp.  Above `h = 0.9 abs(R)` the probe measures
its own oracle.

Requested edit:

```python
    X, Y, RR = (decimal.Decimal(float(x)), decimal.Decimal(float(y)),
                decimal.Decimal(float(R)))
```

`prec` may stay at 60 -- this verification re-ran its whole sweep at
`prec = 120` and every summary field was identical, so 60 is demonstrably
enough for the exact-input form -- but raising it to 80 costs nothing and
removes the last question (the exact decimal expansion of a float near 1e-3
runs to about 60 significant digits, so 60 sits exactly at the boundary).  The report's section 2.1
table and the `trace` docstring's "1.75 ULP" / "57 to 76 at the clamp" should
be re-run and restated; with the exact conversion they become **1.00 / 1.75**
and **41.47 / 75.66**, i.e. the conclusion is unchanged and stronger.

### D2 (P2, test) -- the conditioning bar's `kappa` depends on a random draw

`tests/unit/test_audit_propagation.py`, `_conditioning_bar`:

```python
        rng = np.random.default_rng(20260920)
        direction = rng.normal(size=np.asarray(fit.coef_phi).shape)
```

`kappa` is measured along ONE random direction in coefficient space.
Re-measured with `default_rng(7770001)` and a different four-point ladder, on
both builds: **1.6406e+06** against the shipped seed's **6.2423e+06** -- a
factor of 3.8, with the ladder linear to 1.2 % in both cases, so this is not
noise but the operator's directional spread.  The bar is `100 * eps * kappa`,
so the margin the report presents as 11.1x-13.7x reads **2.92x-3.59x** along
the other direction, and a future seed 3.8x lower again takes the arm red
without anything in the library moving.  TESTING_STANDARDS rule 5 asks for a
measured envelope below the bar; the direction dependence is that envelope and
it is unmeasured.

Requested edit -- measure the envelope and bar the WORST case:

```python
        deltas = (1e-12, 1e-11, 1e-10)
        # WP-C2 / VERIFY-WP-C2 (2026-09-20): kappa is DIRECTIONAL.  One
        # random direction reads 6.2423e+06 and another 1.6406e+06 on the
        # same fixture and the same build, so a bar derived from one draw
        # carries a 3.8x hidden envelope.  Measure several and take the
        # SMALLEST kappa (the tightest floor), and assert the spread so
        # the envelope is a number in the test rather than an assumption.
        kappas_by_direction = []
        for seed in (20260920, 7770001, 424242):
            rng = np.random.default_rng(seed)
            direction = rng.normal(size=np.asarray(fit.coef_phi).shape)
            direction /= np.linalg.norm(direction)
            ladder = []
            for delta in deltas:
                alt = dataclasses.replace(
                    fit,
                    coef_phi=np.asarray(fit.coef_phi) * (1.0 + delta * direction))
                moved = propagate_modal_asymptotic(alt, **kwargs)
                ladder.append(float(np.max(np.abs(moved - base))) / peak / delta)
            assert max(ladder) / min(ladder) < 1.5, (seed, ladder)
            kappas_by_direction.append(float(np.median(ladder)))
        kappa = min(kappas_by_direction)
        spread = max(kappas_by_direction) / min(kappas_by_direction)
        assert spread < 100.0, (
            f'kappa varies by {spread:.1f}x over the directions sampled; '
            f'the eps * kappa floor no longer has a single value.  '
            f'Measured 3.8x on 2026-09-20 (6.2423e+06 / 1.6406e+06).')
```

and restate the docstring's "7.4x to 9.0x that floor" as the measured range
over directions (**7.4x to 34.3x** across the two seeds measured here).

### D3 (P3, docs) -- the history-drift coefficient is a long-end reading

`lumenairy/raytrace/trace.py`, the `renormalize` docstring:

> measured 6.7e-16 on a 3-surface stack rising to **1.8e-15 on a 13-surface
> stack** (about 0.6 ``n_surfaces * eps``...)

The BOUND is right; the coefficient is not.  Measured on my own ladder, both
builds: the ratio to `n_surfaces * eps` is **1.00 at three surfaces** and falls
to 0.58 at thirteen.  A consumer sizing a tolerance from "0.6 n eps" is under
by 40 % on a triplet -- the commonest case.

Requested edit: replace "about 0.6 ``n_surfaces * eps``" with

```
        -- measured at ``1.00 * n_surfaces * eps`` on a 3-surface stack,
        falling to ``0.58`` of it by 13 surfaces, so ``n_surfaces * eps`` is
        the BOUND and the coefficient is not constant.  ``1e-15`` is first
        exceeded at the SEVENTH surface (1.22e-15), not the eighth.
```

and correct the same "by the eighth surface" claim in the report's section 0
item 7 and section 3.2.

### The way back itself is exact (not a defect -- the strongest result here)

Before the defect, the claim that matters most to a user.  `vc2_wayback.py`
dumps 594 arrays -- five prescriptions (doublet, 7-surface stack, 13-surface
ladder, a conic stack, a two-mirror stack) at two field angles under both
`output_filter` modes, every history bundle, `x/y/z/L/M/N/opd/alive/code` --
from the PRE tree in its own process with its own `PYTHONPATH`, and from the
branch with both old keywords forced.

| | Windows | WSL |
|---|---|---|
| `sphere_normal='generic', renormalize='surface'` vs the 49ddf4bd archive | **594 / 594 byte-identical** | **594 / 594** |
| the shipped DEFAULTS vs the same archive | 172 / 594 identical, 422 moved | 172 / 594, 422 moved |
| worst absolute move at the defaults | 3.33e-16 | 3.33e-16 |

The way back is exact, and it is exact through a real second process on a
read-only archive rather than by self-consistency.  Note the second row's
company: the CONIC prescription moves too, because `renormalize='exit'`
applies to every prescription whether or not it contains a sphere -- which is
why that switch has no control arm, as the WP-C2 report itself observes.

### D4 (P2, API) -- sixteen entry points have no way back, not six

The campaign's rule is that every moved public entry point has a one-keyword
way back.  An AST census of the package (`vc2_entrypoints.py`, both builds
identical) finds **21** exported, directly-tracing entry points whose
signature carries neither `sphere_normal` nor `renormalize`.  Five of them
reach `trace_jax`, which has neither switch by design and is unaffected by the
flip; **sixteen are CPU-affected**:

| already named in the report (6) | NOT named (10, plus 5 jax twins) |
|---|---|
| `trace_prescription` | `apply_real_lens_traced` |
| `raytrace_system` | `apply_real_lens_maslov` |
| `ray_fan_data` | `ray_transfer_jacobian` |
| `opd_fan_data` | `eval_image_plane_wfe` |
| `through_focus_rms` | `caustic_diagnostic` |
| (`spot_rms` / `spot_geo_radius` / `refocus` -- these CONSUME a `TraceResult` and do not trace; their bytes move because their input does) | `plot_lens_layout` |
| | `fit_canonical_polynomials`, `fit_hf_polynomials` |
| | `paraxial_focus_world` |
| | `ray_fan_data_world`, `opd_fan_data_world` |
| | (jax: `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`, `fit_canonical_polynomials_jax`, `ray_transfer_jacobian_jax`) |

A further **46** exported functions reach a tracer transitively with no way
back, including `apply_real_lens_auto`, `apply_real_lens_universal`,
`propagate_through_system`, `monte_carlo_tolerancing` and
`optimize_traced_geometry`.

The two the report's own byte-identity census could not have seen are the ones
that matter most: `apply_real_lens_traced` and `apply_real_lens_maslov` are the
library's headline lens propagators.

Requested edits.  The cheapest complete fix is one keyword pair threaded
through the five `raytrace/` entry points plus a single documented way back
for the rest:

```python
# lumenairy/raytrace/trace.py -- trace_prescription, raytrace_system
def trace_prescription(..., renormalize: str = 'exit',
                       sphere_normal: str = 'analytic'):
    ...
    return trace(rays, surfaces, wavelength, ...,
                 renormalize=renormalize, sphere_normal=sphere_normal)

# lumenairy/raytrace/ray_fan.py -- ray_fan_data, ray_fan_data_world,
#                                  opd_fan_data, opd_fan_data_world,
#                                  through_focus_rms
#   same two keyword-only parameters, forwarded to every internal
#   trace / trace_world / _trace_fan_set call.

# lumenairy/raytrace/world.py -- paraxial_focus_world
# lumenairy/raytrace/differential.py -- ray_transfer_jacobian
# lumenairy/analysis/aberration.py -- caustic_diagnostic
# lumenairy/analysis/image_plane_wfe.py -- eval_image_plane_wfe
# lumenairy/analysis/plotting.py -- plot_lens_layout
# lumenairy/propagators/asymptotic_canonical_fit.py
#     -- fit_canonical_polynomials, fit_hf_polynomials
# lumenairy/elements/_lens_traced.py -- apply_real_lens_traced
# lumenairy/elements/lenses_maslov.py -- apply_real_lens_maslov
```

If threading sixteen signatures is judged too large, the alternative that
still satisfies the one-keyword rule is a documented module-level override in
`raytrace/trace.py` -- but it must be explicit, not a global mutable:

```python
@contextlib.contextmanager
def raytrace_defaults(*, renormalize=None, sphere_normal=None):
    """Temporarily restore the pre-5.49.0 arithmetic for every internally
    tracing entry point.  Not thread-safe; use it around a call, not around
    a program."""
```

Either way, `test_verify_c2_analytic_normal.py::test_vc2_the_entry_points_without_a_way_back_are_pinned`
pins the census so the list can only move deliberately, and section 8 item 1
of the WP-C2 report should be corrected from "six" to the measured sixteen.

### D5 (P3, consistency) -- `analysis.ghost` refracts off a different normal

`lumenairy/analysis/ghost.py:934,955`.  Measured on three 2-bounce ghost paths
of a spherical doublet, 256 rays, both builds: forcing the public default's
normal moves the ghost RMS spot radius by up to **2.13e-14 mm**
(= 2.13e-17 m) and the FWHM by 1.42e-14 mm; total transmittance, energy
fraction and ray counts do not move.  Small, but ONE implementation says the
ghost path should ask the same default as `trace`.

Requested edit (the normal only -- `renormalize` must STAY `True` there,
because ghost owns its own loop and has no exit pass to hoist to):

```python
            _reflect(rays, surfs[s_idx], sphere_normal='analytic')
...
            _refract(rays, surfs[s_idx], n1, n2, sphere_normal='analytic')
```

### D6 (P2, cross-backend) -- the JAX tracer has no domain clamp, and nothing pins it

`lumenairy/raytrace/jax_trace.py:417-420` builds the pure-spherical normal as
`(x, y, z - R) / R` and applies no `h**2/R**2 < 0.9999` gate.  The NumPy side
applies that gate on BOTH routes.  So the two backends' vignetting differs by
the WHOLE outer annulus `h > 0.99995 abs(R)`, not by one ULP.

Measured on a ball lens (R = 12.5 mm, clear semi-diameter 12.5 mm), 40 000
rim-packed rays, both builds:

| | rays past the clamp | kept alive |
|---|---|---|
| CPU, `surface`/`generic` | 1991 | **0** (all RAY_NAN) |
| CPU, `exit`/`analytic` | 1991 | **0** (all RAY_NAN) |
| **JAX** | 1991 | **1991** |

1991 alive-flag disagreements on a catalogue part.  This is PRE-EXISTING --
identical under all four CPU settings -- so WP-C2 did not cause it, but the
work package's item 4 says "the flip moves the CPU tracer TOWARD the JAX one"
without noting that the two still disagree by an annulus.  And nothing pins
it: the `jax_gets_a_clamp` mutation survives **294** raytrace and parity tests.

Requested edits.  (a) The report's section 4 gains the sentence: "the two
backends nonetheless gate the sphere's domain differently -- NumPy is NaN
outside `h**2/R**2 < 0.9999` and JAX has no gate -- so on a prescription whose
aperture reaches the rim they disagree on an annulus, measured 1991 of 40 000
rays on a ball lens."  (b) The divergence is now pinned by
`test_verify_c2_analytic_normal.py::test_vc2_the_jax_backend_does_not_apply_the_numpy_domain_clamp`.
(c) Whoever owns `jax_trace.py` decides whether JAX should gain the clamp or
NumPy should lose it -- that is the "separate vignetting decision nobody has
taken" the WP-C2 brief already names, and it now has a number attached.

### D7 (P2, tooling) -- the `EDITED_IN_PLACE` override is one-sided

`scripts/reanchor_citations.py::_edited_in_place`.  The guard compares only
the text BEFORE the first `=`:

```python
    lead = want.strip().split('=')[0].strip()
    if not lead or not got.strip().startswith(lead):
        return None, None
```

Abused (`vc2_ghost_and_reanchor.py`, each result reproduced on both builds):

| doctored current line | override |
|---|---|
| `sphere_normal: str = 'analytic',` (the shipped state) | fires -- correct |
| **`sphere_normal: str = 'generic',`** (the default silently REVERTED) | **fires** |
| **`sphere_normal: str = 'not-a-route',`** | **fires** |
| **the declaration MOVED and a stale copy left at line 61** | **fires**, pointing at the stale copy |
| `renormalize: str = "exit",` (an unrelated line) | refused -- correct |
| the file truncated to 30 lines | refused -- correct |

So the map cannot hide a citation that has slid onto an UNRELATED line, which
is what its comment claims, but it silently re-anchors a citation whose claim
has become FALSE.  It is also not version-pinned: the release number lives only
in the human-readable reason string, and nothing compares it to `__version__`
or to the CHANGELOG block being re-anchored, so the map keeps firing for every
release after 5.49.0.

Requested edit -- pin the NEW content, not just the old token:

```python
EDITED_IN_PLACE = {
    ('lumenairy/raytrace/trace.py', 61): (
        61, "WP-C2 5.49.0: sphere_normal default 'generic' -> 'analytic'",
        "sphere_normal: str = 'analytic',"),
    ...
}


def _edited_in_place(path, base_num, base):
    entry = EDITED_IN_PLACE.get((path, base_num))
    if entry is None:
        return None, None
    new_num, reason, want_new = entry
    want, _ctx = base_line(path, base_num, base)
    hay = lines(path)
    if want is None or not (1 <= new_num <= len(hay)):
        return None, None
    got = hay[new_num - 1]
    lead = want.strip().split('=')[0].strip()
    if not lead or not got.strip().startswith(lead):
        return None, None
    # VERIFY-WP-C2 (2026-09-20): also require the line to be the EXACT
    # content this entry says the release produced.  Without it the
    # override accepts the default being reverted, set to nonsense, or
    # left behind as a stale copy while the declaration moves -- all three
    # demonstrated in validation/probe_verify_c2/.
    if got.strip() != want_new.strip():
        return None, None
    return new_num, f'edited in place ({reason})'
```

That turns the map from a pattern into a pinned fact and makes it fail loudly
the day the default moves again -- which is exactly when a human should look.

### D8 (P3, test) -- the d3 arm-2 bar is smaller than its own floor's spread

`tests/unit/test_niche_d3_guards.py::test_the_residual_degree_moves_the_multiplexed_route_only_through_c6`:
`assert moved > 3.0 * noise`, with `noise` measured from ONE perturbation
(`np.nextafter(..., +inf)` on both parts of the envelope).  Measured over four
one-ULP directions:

| nudge | Windows | WSL |
|---|---|---|
| up (the shipped floor) | 0.1404 | 0.1108 |
| down | **0.1782** | **0.1149** |
| real part only | 0.1404 | 0.1108 |
| one element | **0.0554** | **0.0240** |
| spread | **3.22x** | **4.79x** |

The multiplier is 3.0.  The arm still passes at every floor measured (margin
4.69x Windows, 8.59x WSL), so it is not red -- but by rule 5 there is no gap
on the lower side.  Neither arm is two-sided either: the natural control ("the
same degree twice") reads exactly **0.0** on both builds and is not asserted.

Requested edit:

```python
    # VERIFY-WP-C2 (2026-09-20): measure the floor over SEVERAL one-ULP
    # directions and bar against the worst.  One direction is not a floor:
    # measured 0.0554 to 0.1782 (3.22x, Windows) and 0.0240 to 0.1149
    # (4.79x, WSL) on the same reading, i.e. a spread larger than the
    # multiplier that used to sit on a single draw.
    noise = max(_mux_last_bit_noise(0.023, degree=6, launch=True, kind=k)
                for k in ('up', 'down', 'one_element'))
    assert moved > 3.0 * noise, (...)
    # and the two-sided half: holding the degree FIXED must land AT the
    # floor, not above it.  Measured 0.0 exactly on both builds.
    assert _same_degree_twice(0.023) <= noise, (...)
```

with `_mux_last_bit_noise` gaining a `kind` parameter selecting the
perturbation.  The same applies to the sibling arm's `10.0 * noise`, whose
floor differs by **31x** between the two builds (2.59e-3 Windows against
8.42e-5 WSL) -- absorbed there by a 65x-2008x margin, but it should be stated.

### D9 (P3, docs) -- the Migration note's rim-band paragraph is right but incomplete

Credit where it is due: the Migration Guide already tells a user who works rays
past `0.9999 R^2` to give the surface an explicit clear aperture rather than
rely on the clamp, and already says neither route resolves the normal there.
Two facts it does not carry, both measured here, and both of which change how a
reader sizes the risk:

* the band **does not exist on the meridian at all** -- the two gate
  expressions agree EXACTLY at `y = 0`, at all eight radii bisected, so a
  meridional fan cannot enter it;
* what a real design actually meets is the **clamp**, not the band, and it is
  quantifiable: a ball lens or hemisphere (semi-diameter `abs(R)`) loses
  **4.9 %** of a rim-packed bundle to `RAY_NAN` -- a numerical-fault code, not
  `RAY_APERTURE`, so it lands in `raytrace.layout`'s fault histogram rather
  than its vignetting one -- on both routes, before and after WP-C2.

Requested addition to the Migration Guide's 5.49.0 `sphere_normal`
section, after the existing rim-band paragraph:

```
Both routes refuse a ray above ``h = 0.99995 |R|`` and report it as
``RAY_NAN``.  That clamp is UNCHANGED by this release, but it is what a
hemispherical or ball lens actually meets: with the clear aperture at
``|R|``, 4.9 % of a bundle packed into the outer 0.1 % of the aperture dies
there (measured, 60 000 rays).  The one-ULP band where the two routes
disagree sits inside that region and is not reachable by sampling -- 580 000
rays aimed at the rim moved zero flags -- and it does not exist on the
meridian at all, because the two gate expressions agree exactly at ``y = 0``.
```

### D10 (P3, process) -- the 19 new test ids are not in `.test_durations`

`git diff 49ddf4bd..eadc67ba -- .test_durations` is empty, and
`test_c2_analytic_normal_default.py`'s 19 ids are absent from the file.  The
durations file is what splits the CI lanes; 19 unlisted ids are scheduled
blind.  This verification's 16 ids ARE spliced (see the commit).

Requested edit: run
`python -m pytest tests/unit/test_c2_analytic_normal_default.py -p no:randomly -vv --durations=0`
and splice its 19 ids into `.test_durations`.

### D11 (P1, docs) -- four private-layer docstring sentences still say the shipped default is the GENERIC route

The work package rewrote the two PUBLIC docstrings and left the private ones
behind.  Four of their sentences are now false, and the first is the
one a reader reaches for when asking exactly the question the Migration note
raises:

`lumenairy/raytrace/surface.py:722-724` (`_sphere_normal`):

> That band is reachable only under ``sphere_normal='analytic'``; **the
> shipped default is the generic route on both sides.**

The shipped default IS the analytic route, so the band is reachable AT THE
DEFAULT.  This sentence tells a reader the opposite of what the CHANGELOG and
the Migration Guide tell them, in the very function whose docstring they would
open to check.

`lumenairy/raytrace/surface.py:749-750` (`_surface_normal`):

> The default is the generic sag-derivative route, which is **the arithmetic
> every caller has always got**

The PRIVATE default is still `False`, so the first clause is right; "every
caller" is not -- `trace` and `trace_world` now pass `analytic_sphere=True` at
every pure sphere.

`lumenairy/raytrace/intersection.py:159-160` (`_intersect_surface`), and the
same phrase in `_refract`'s `sphere_normal` parameter block:

> It is **opt-in** because it differs in the last bit from the sag-derivative
> route **every caller has been getting**.

It is opt-OUT for the two public tracers now.

Neither the doc-consistency gate nor the walker citation gate reads prose, so
nothing caught these.  Requested edits:

```python
# surface.py, _sphere_normal
    ``RAY_NAN`` kill through the generic route.  That band is reachable
    only under ``sphere_normal='analytic'`` -- which is the SHIPPED
    DEFAULT of ``trace`` / ``trace_world`` since 5.49.0, so it is
    reachable by default; ``sphere_normal='generic'`` is the way back.
    Measured: 580 000 rays over twelve prescription and field
    combinations move zero alive flags, and the band does not exist on
    the meridian at all (the two gate expressions agree exactly at
    ``y = 0``).

# surface.py, _surface_normal
    analytic_sphere : bool, default False
        ...  This PRIVATE default did not move in 5.49.0 -- it is what
        ``analysis.ghost`` and the finite-difference differential path
        get, so their arithmetic is unchanged by construction -- but
        ``trace`` / ``trace_world`` now pass ``True`` at every pure
        sphere.

# intersection.py, _intersect_surface
    It is the DEFAULT for ``trace`` / ``trace_world`` since 5.49.0 and
    opt-in for every direct caller of ``_refract`` / ``_reflect``,
    because it differs in the last bit from the sag-derivative route.

# intersection.py, _refract's sphere_normal block
    ``'generic'`` is the sag-derivative dispatch and the default of THIS
    private helper; ``trace`` passes ``'analytic'``.
```

Pinned by
`test_verify_c2_analytic_normal.py::test_vc2_no_private_docstring_claims_the_generic_route_is_shipped`.

### D12 (P2, docs) -- the CHANGELOG and Migration Guide carry a byte-identity count that the probe's own JSON contradicts

`CHANGELOG.md:50` and `Migration-Guide.md:1795-1797` both say:

> explicitly, **938 of 1008** are byte-identical and **the 70** that are not are
> exactly [the entry points listed]

`WP-C2_ANALYTIC_NORMAL_REPORT.md:539,541` says 934 / 1008 (Windows) and
935 / 1008 (WSL), with 73-74 moving.  The probe's own committed output settles
it: `validation/probe_c2_analytic_normal/byte_identity_old_kw_win.json` has
`n_moved = 74` and the WSL file `n_moved = 73`, i.e. **934** and **935**
identical.  Summing the moved keys in the Windows file by entry point gives
`trace_prescription` 27 + `refocus` 26 + `ray_fan_data` 8 + `opd_fan_data` 8 +
`spot_rms` 4 + `through_focus` 1 = **74**, which is the report's number and not
the guide's.

The two USER-FACING documents are the ones that are wrong, and they are the
ones a reader will quote.

Requested edits, in both files:

```
explicitly, 934 of 1008 are byte-identical on Windows (935 on WSL) and the
74 that are not (73 on WSL) are exactly [the entry points listed]
```

and, while that line is being touched, the entry-point list itself needs the
correction in **D4** -- the Migration Guide's "No keyword there" paragraph
names the same six and is short by ten.

---

## 5. Gate runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line, `-q -p no:randomly --capture=sys -rf`.

| run | files | result |
|---|---|---|
| Windows: the trace-touching family + both C2 test files + the B9 family + `test_niche_d3_guards.py` + `test_audit_propagation.py` + `test_niche_audit_w6_asymptotic.py` + the census / walker / dispatcher-pin / public-API / doc-consistency / history-relocation gates + `test_audit_except_budget.py` | 29 | **2000 passed, 14 skipped, 0 failed** (14:40) |
| WSL: both C2 test files + the B9 family + `test_public_api.py` + `test_audit_except_budget.py` | 5 | **123 passed, 1 failed** -- `test_installed_metadata_version_matches_source_version`, which is the ninth pre-existing red and fails on the 49ddf4bd archive on that mount too |
| WSL: `test_niche_d3_guards.py` + `test_audit_propagation.py` + `test_niche_audit_w6_asymptotic.py` | 3 | **198 passed, 0 failed** (12:04) |
| `tests/unit/test_verify_c2_analytic_normal.py` alone | 1 | **17 passed**, 5.8 s Windows / 3.5 s WSL |
| the nine "pre-existing" ids, on this verifier's own `git archive 49ddf4bd` | 9 | 8 failed identically (Windows), the ninth failed identically (WSL) |
| mutation matrix, both builds | 11 arms | control 94 passed; 8 mutations caught; `jax_gets_a_clamp` survives (also survives a 294-test raytrace and parity sweep); `whole_normal_sign_flipped` survives correctly |

WSL `ruff check lumenairy/ tests/`: **All checks passed!**
`python -m mypy` (no args): **Success: no issues found in 33 source files.**
`python scripts/record_history_fingerprints.py --check`: **OK: every history
document matches its module.**

---

## 6. Ship recommendation

**SHIP**, with D11, D1, D3, D4, D7 and D12 actioned before the release note
is written, and D2, D5, D6, D8, D9, D10 filed.  D11 is the only P1: a reader
who opens `_sphere_normal` to ask whether the rim band is reachable is told,
in that docstring, that it is not -- which is the opposite of what the release
actually did.  D12 is two numbers in two user-facing files and costs nothing to
fix.

The two default flips are sound and this verification strengthened rather than
weakened both:

* `sphere_normal='analytic'` is better or equal to the generic route over the
  whole working aperture against a CORRECTED oracle (1.00 against 1.75 out to
  `0.95 abs(R)`, never worse by more than one unit at any of 1280 points), is a
  unit vector to 1 ULP by construction, removes **16 %** of the trace's array
  work by a deterministic count with an exactly-1.0000x control, and moves zero
  alive flags and zero error codes across 580 000 rays over twelve
  prescription and field combinations on two builds.
* `renormalize='exit'` costs no measurable accuracy against a 60-digit
  end-to-end oracle, and its induced error is bounded at 3.0e-18 m by a
  measured linear sensitivity coefficient.  Its speed benefit is real but
  small and sign-dependent on surface count; **KEEP** (section 3).

The six items that must not go out unqualified are the four private-docstring
sentences that still say the generic route is shipped (**D11, P1**), the entry
points with no way back (**sixteen**, not six -- D4), the oracle whose own
error exceeds what it reports above `h = 0.9 abs(R)` (D1), the `0.6 n eps`
coefficient (D3), the citation override that accepts a reverted default (D7)
and the byte-identity count the CHANGELOG and Migration Guide get wrong (D12).

**And the two results the release notes should gain, because they are better
than what is in them now.**  The way back is byte-identical archive to archive
in a second process (594 / 594 arrays, both builds), and the closed form's
accuracy against a corrected oracle is 1.00 against 1.75 rather than 1.75
against 2.00-2.25.  Both are stronger claims than the ones currently shipped.

---

## 7. What could not be measured

1. **The 1008-array byte-identity census was not re-run.**  The AST entry-point
   census supersedes its CONCLUSION (and finds ten more entry points than it
   did), but the per-array counts -- "934/1008 identical", "73-74 move" -- are
   not independently reproduced here.  The claim they support is verified by a
   different route; the counts themselves are taken on trust.
2. **The 368-file blast radius was not re-run.**  It cost the WP-C2 agent
   about seven hours across seven shards.  What was re-run: the two C2 test
   files, the B9 family, `test_niche_d3_guards.py`, `test_audit_propagation.py`,
   `test_niche_audit_w6_asymptotic.py`, nine raytrace-touching files, and the
   census / walker / dispatcher-pin / public-API / doc-consistency /
   except-budget gates -- plus the nine "pre-existing" ids against this
   verifier's own archive.
3. **A clean absolute timing number.**  Three instruments failed under load on
   this box, including one that reversed sign between runs.  The element-op
   count replaces them for both switches, but it is a COUNT with a stated
   first-order cost model (equal cost per element-op), not a time.
4. **`mpmath` on WSL.**  The 60-digit cross-check of the `w6_a2` Newton step
   ran on Windows only (3.08e-33); the WSL venv has no `mpmath`.  Every other
   `w6_a2` number is measured on both builds.
5. **Whether the rim band matters to any user's prescription.**  Unchanged
   from the work package's own answer: it is about one ULP wide where it
   exists at all, sampling cannot find it, and only a directed walk reaches it.
   What this verification adds is that the CLAMP around it is reachable by a
   real design, which is a different and larger fact (D9).
