# The decentred beam's ray-fit branch: measuring it instead of guessing it

Niche C11, 2026-08-03.  Branch `feat/d121-final-closure`, on top of
`5af1edf` (the C9 + C10 tree).

Predecessors: `D121_RESIDUAL_CLOSURE_2026_08_02` (S4.2, S5, S7 items 1/2/5),
`C6_FIT_GUARD_DECISION_2026_07_31`, `C8_INVERSE_SUPPORT_BOUND_2026_08_01`.

---

## 0. Headline

`_DECENTRE_GATE_W_FRAC = 0.05` chooses, per element call, between the
historical CONCENTRIC ray fit and the D1/D7 OFF-CENTRE one.  Its own note says
it was set to kill a discontinuity at NULL decentre, not to find the
crossover.  It does not find the crossover.

**The crossover is not a number.**  Measured against a fit-domain-free oracle
it is at **0.55 w** on a synthetic f/3 N-BK7 singlet, at **0 w** on an f/6 one,
and it lands anywhere in **0.46-0.69 w** across design 121's six groups.  It is
not a property of the decentre at all: it is where two different approximations
of the same traced map happen to be equally good, which depends on how much
aberration the concentric branch's lower-order fit leaves over the beam.  Four
closed-form predictors were derived and three of them land in the right band on
design 121 -- and every one of them is wrong on at least one of the synthetic
geometries, for the same reason: they all predict a constant, and the quantity
is not one.

**So the branch is now ARBITRATED.**  At the fit site the rays are already
traced, so both candidates are BUILT and COMPARED there: fit the OPL each way,
score each against the traced samples themselves weighted by the beam's own
intensity, take the smaller.  One extra Chebyshev fit per call, taken only
above the C1 null gate.  **`DECENTRED_FIT_ARBITER` ships `False` -- OPT-IN**
(S6.3); `True` engages it, and with it `False` the whole layer is
byte-identical to the pure-gate selector, measured on the design (S6.5).

**What it buys, on the acceptance target.**  Design 121, EE3 area-exact against
the exact-ray CARRY=1 ceiling, residual in points:

| order | last-group `\|c\|/w` | **C11** | v5.32 | delta |
|---|---|---|---|---|
| (0,0)   | 0.000 | **-0.0477** | -0.0477 | *byte-identical* |
| (-1,0)  | 0.241 | **+0.0552** | +0.0290 | **+0.026 WORSE** |
| (-2,0)  | 0.481 | **+0.0464** | +0.0634 | -0.017 |
| (-3,0)  | 0.723 | **+0.0375** | +0.0898 | -0.052 |
| (-4,0)  | 0.965 | **+0.0308** | +0.1410 | -0.110 |
| (-4,-2) | 1.079 | **+0.0693** | +0.1517 | -0.082 |

Worst-case residual **0.152 -> 0.069** (2.2x), field-angle spread
**0.200 -> 0.117**, and the monotone growth of the residual with field angle --
the signature the whole D121 campaign has been chasing -- is gone: the tilted
orders now read 0.031-0.069 with no trend, against 0.029-0.152 rising.

**And one order moves the wrong way, by 0.026 points against a 0.003-0.015
differential floor.**  It is stated here rather than averaged away, it is traced
to a mechanism this study could separate but not fix (S6.4), and **it is why
the arbiter ships OPT-IN**: a default that fails a per-order "improve or hold"
bar at one order is a judgement about a design, not a library fact, and it
should be taken with this table in front of the decision rather than silently
in a patch release (S6.3).

**Two things were also found on the way that are not about the gate.**

* Niche C10's degree raise made the CONCENTRIC branch far MORE dangerous at
  large decentre -- the forced-concentric arm's residual at (-4,0) goes from
  19.9 points (degree 4) to **67.3** (degree 6).  The branch the old gate
  selects below 0.05 w is now a worse thing to fall back to than it was when
  the gate was set.
* The prior study's headline "the concentric branch is worth 0.9 points at
  0.24 w" is, on the current tree, **85 % the niche-C6 residual-eikonal fit
  DOMAIN and 15 % the ray-fit branch** (S6.4).  Those two moved together in
  `rc_gate_121.py` and `D121_RESIDUAL_CLOSURE` S7 item 5 names them as
  unseparated; they are separated here, and the separation reverses the
  attribution.

**The v5.32.1 CI reconciliations are in S9**, and they resolve into three
findings.  Two are REPRODUCED in the Linux/OpenBLAS proxy, attributed and
reconciled: D3's linearity separation (caused by C10 -- **niche C9 is inert on
it**, its rows are bit-identical) and a class the brief did not name, found by
running the affected FILES rather than the named cases.  The third, the P2 dx
self-check, is **measured healthy on both builds with a 10.5x margin and every
C9/C10/C6 knob inert**, so nothing was changed there and the reason CI saw it
red is reported as unexplained rather than guessed at.

---

## 1. Provenance, and the floors every delta is measured against

### 1.1 Which library each number was taken against

| | |
|---|---|
| repo | `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy` |
| branch / base | `feat/d121-final-closure`, `5af1edf` |
| `lumenairy` | 5.32.0 |
| `_lens_traced.py` BEFORE | `34ef5a9d95279b8f` (the C10 hash) |
| `_lens_traced.py` AFTER | `8b13259592233b52` |
| `carrier.py` | `5a1b0d1021969df1` **unmodified**, both states |
| CI proxy | WSL Ubuntu, `~/lumvenv`, Python 3.12.3, numpy 2.4.6, **OpenBLAS** 0.3.31 |
| dev box | Windows 11, Python 3.14, MKL |

Every runner forces `LUMEN_PIN=0` (the live working tree; `approx_common`'s
default still resolves the stale `pin_d2e60ca` export, which is v5.31).

**The chain-A cache was NOT invalidated, deliberately.**
`_chainA_1024_2000nm_rs4.npz` is dated 2026-07-30, i.e. pre-C7/C8/C9/C10 --
exactly the trap `D121_RESIDUAL_CLOSURE` S9 item 1 records.  It is kept because
every number in this document is DIFFERENTIAL against arms that share it, and
because the recorded per-order table this study must not regress was taken
against the same file.  Re-generating it would have made the comparison
meaningless in the other direction.  It is a pre-DOE, on-axis chain, so the
element gate never engages inside it.  **Stated, not hidden**; S10 item 5.

### 1.2 Differential floors, before any delta

| instrument | null | reading |
|---|---|---|
| `c11_gate_arms_121.py`, `v532` arm, all six orders | vs the recorded C10 table | **-0.0477 / 0.0290 / 0.0634 / 0.0898 / 0.1410 / 0.1517** against the audit's **-0.048 / 0.029 / 0.063 / 0.090 / 0.141 / 0.152** -- every printed digit |
| ... and its intensity sha256 | vs `rc_gate_121.py`'s own `shipped` arm, different process | `4e9effd4` / `b2a8b150` / `88f726eb` / `cf0bc1f3` / `7845b7a8` / `c4b850ea` -- **6 of 6 bit-identical** |
| on-axis order, both arms | zero decentre, no branch to choose | **byte-identical** (`4e9effd4` twice) |
| `c11_synth_sweep.py` spline reference | its own residual gate dependence (the reference re-run with the branch forced the other way) | **2.6e-18 waves**, i.e. 9 orders below the smallest branch difference measured |
| `c11_synth_sweep.py` sampling adequacy | power-weighted p99.9 wrapped nearest-neighbour step of every scored residual | **0.000 rad** against `pi` on every row of every geometry |
| campaign estimator noise (inherited) | `D121_RESIDUAL_CLOSURE` S1.3 | **0.003-0.015 points** |

The 0.026-point counter-movement at (-1,0) is therefore **2-9x the floor**: it
is a real move, and it is reported as one.

### 1.3 What was thrown away first

`rc_gate_121.py`'s recorded output (`_rc_gate.txt`) is from the **pre-C10**
tree (`_lens_traced.py 9717ad88dd959889`).  Its table is the one the brief
quotes, and re-using it would have priced a branch decision against a library
two changes old.  It was re-run in full on the current tree (S3.1) before
anything else, and **the crossover it reports moved**: the concentric branch's
failure above 0.72 w got 3.4x WORSE and its advantage below 0.48 w shrank from
0.89 points to 0.13.  Every branch statement below is from the re-run.

---

## 2. The consumer map: everything `_beam_decentred` moves

Enumerated before touching anything, because the brief is right that a gate
change moves all of them.  Line numbers are pre-change.

| # | site | what it decides | side of the RAY TRACE |
|---|---|---|---|
| 1 | 5321 | the CENTRE the beam radius `_w_in_beam` is measured about (beam vs grid origin) | before |
| 2 | 5323-5330 | the stage-2 gate itself, and the origin re-measure that makes the fall-back byte-identical | before |
| 3 | 5331 | `_beam_fit_radius = min(frbf * w, launch_radius)` -- the P2 disc RADIUS | before |
| 4 | 5333-5353 | (indirectly) whether the aperture:beam `on_aperture_beam` warning is reachable at all -- it is skipped when `_beam_fit_radius` is not `None` | before |
| 5 | 5359-5372 | `_fit_r_max`, and `_fit_r_about_beam` -- the radius niche C6's residual-eikonal FREEZE circle has to clear | before |
| 6 | **5390-5397** | niche C6's `_res_w` and `ray_fit_radius`: **the domain the input residual eikonal `a_fit` is fitted over** | before |
| 7 | 5908 | the ray-fit DISC: beam-centred at `_beam_fit_radius`, or origin-centred at `_fit_r_max` | after |
| 8 | 5936-5973 | the RESTRICTION (D1 weighted vs historical hard NaN mask) and the ORDER (D7's `_DECENTRED_FIT_POLY_ORDER` raise, with its sample-count step-down) | after |
| 9 | 5993 | the wording of the "restriction ABANDONED" warning | after |

**The structural fact that decides the whole design.**  Consumer 6 is not a
diagnostic: `_resid_eik` is consumed at 5506-5521, where it augments EVERY ray
direction by `grad(a_fit)`, and at 5645-5655, where it is added to the OPD.  It
is baked into the trace.  **No post-trace decision can change it.**  Consumers
1-6 are therefore committed before a single ray is launched, and only 7-9 are
still open at the fit site.

Two things that are NOT consumers, checked explicitly:

* **`carrier.py`'s `on_decentred_fit` / `decentre_fit_frac` tier.**  It reads
  the chief-ray offset directly (`_check_decentred_fit`, carrier.py 3927) and
  never reads the element gate; it changes no behaviour, only warns.  Its
  default `_DECENTRE_FIT_FRAC_DEFAULT = 0.5` happens to sit inside the
  crossover band this study measures, which is a coincidence worth recording
  and not a coupling.  Its docstring is a v5.32-era description of the
  decentred fit; it is **left alone** (carrier.py is unmodified) and flagged in
  S10 item 4.
* **`REMAP_STATIONARY_PHASE_FIT_GUARD` (`_c6_fit_guard`).**  It is OR'd with
  `_beam_decentred` at consumer 8, but it is independently gated and defaults
  `False`.  The arbiter mirrors it exactly when scoring the concentric
  candidate, so the scored configuration is the applied one in both states.

---

## 3. What the crossover depends on

### 3.1 The branches, re-measured on the current tree

`rc_gate_121.py`, unedited, all six orders, three arms
(`shipped`; `_DECENTRE_GATE_W_FRAC = inf` = every call CONCENTRIC;
`_DECENTRE_GATE_W_FRAC = _DECENTRE_GATE_PIXELS = 0` = every call OFF-CENTRE).
EE3 area-exact against the CARRY=1 exact-ray ceiling; residual in points:

| order | last-grp `\|c\|/w` | oracle | shipped | **concentric** | offcentre | res ship | **res conc** | res off |
|---|---|---|---|---|---|---|---|---|
| (0,0)   | 0.000 | 90.5324 | 90.5801 | 90.5801 | 90.5801 | -0.0477 | **-0.0477** | -0.0477 |
| (-1,0)  | 0.241 | 90.5768 | 90.5477 | 90.6753 | 90.5493 | 0.0290 | **-0.0985** | 0.0275 |
| (-2,0)  | 0.481 | 90.6650 | 90.6016 | 90.7918 | 90.6028 | 0.0634 | **-0.1267** | 0.0623 |
| (-3,0)  | 0.723 | 90.6961 | 90.6062 | 88.6630 | 90.6052 | 0.0898 | ***2.0331*** | 0.0909 |
| (-4,0)  | 0.965 | 90.5035 | 90.3625 | 23.1918 | 90.3625 | 0.1410 | ***67.3117*** | 0.1410 |
| (-4,-2) | 1.079 | 90.1071 | 89.9554 | 10.9619 | 89.9554 | 0.1517 | ***79.1452*** | 0.1517 |

On axis all three are byte-identical (`4e9effd4` x3).  At (-4,0) and (-4,-2)
`offcentre` is byte-identical to `shipped` (`7845b7a8`, `c4b850ea`), because no
group of those orders sits below 0.05 w.

**Both halves of the brief's premise moved under C10.**

* The concentric branch's advantage below the crossover collapsed: **0.89
  points at (-1,0) at degree 4, 0.13 at degree 6**, and it is now an
  OVERSHOOT of the ceiling (-0.099) rather than a recovery to it.
* The concentric branch's failure above the crossover got much worse:
  (-3,0) / (-4,0) / (-4,-2) go from 9.77 / 19.93 / 11.25 points at degree 4 to
  **2.03 / 67.31 / 79.15** at degree 6.

The second is mechanistically expected and worth stating on its own: the C6
launch augments every ray direction by `grad(a_fit)` of a NON-radial
polynomial, and the concentric hard-mask branch's whole safety argument is the
radial symmetry of its unconstrained directions (`_FIT_DISC_OUTSIDE_WEIGHT_REL`
says so, and `REMAP_STATIONARY_PHASE_FIT_GUARD` exists because of it).  A
higher-degree `a_fit` carries more non-radial content, so it breaks that
symmetry harder.  **C10 made the branch the old gate falls back to more
fragile, and the gate was calibrated before that.**

### 3.2 Where the crossover actually is, on designs that share nothing with 121

`c11_synth_sweep.py`.  Self-contained N-BK7 singlets built inline, no
prescription asset.  Both branches forced through the library's own module
attributes; scored against `newton_fit='spline'` -- a LOCAL bicubic of the
traced map which skips the polynomial fit and its disc restriction entirely
(the disc block is gated on `newton_fit != 'spline'`), so it is independent of
both candidates.  `sig` is the equivalent rms wavefront error in WAVES, piston
and tilt projected out without any phase unwrapping.

f/3 singlet (`R = +-30 mm`, aperture 10 mm, `w = 1.0 mm`, `dx = 30 um`,
`ray_subsample=8`, `fit_radius_beam_factor=2`):

| `\|c\|/w` | inflation | sig CONCENTRIC | sig OFF-CENTRE | winner |
|---|---|---|---|---|
| 0.00 | 1.000 | 9.37e-08 | 9.37e-08 | *tie (no branch)* |
| 0.05 | 1.002 | **9.40e-08** | 6.57e-07 | **conc** |
| 0.20 | 1.039 | **7.91e-08** | 6.54e-07 | **conc** |
| 0.40 | 1.149 | **1.69e-07** | 6.22e-07 | **conc** |
| 0.50 | 1.225 | **4.13e-07** | 6.36e-07 | **conc** |
| 0.60 | 1.311 | 7.59e-07 | **6.18e-07** | **off** |
| 0.75 | 1.458 | 1.71e-06 | **6.00e-07** | **off** |
| 1.00 | 1.732 | 5.38e-06 | **5.26e-07** | **off** |
| 1.50 | 2.345 | 3.73e-05 | **2.89e-07** | **off** |

f/6 singlet, same beam, and the same f/6 at `w = 1.4 mm` (the geometry the C1
note's own crossover sweep used): **off-centre wins at every nonzero
decentre**, by 3-1000x, and the crossover is therefore at 0 w.

**The shape is the same on all three, and it is the whole mechanism:**

* the OFF-CENTRE branch's error is **FLAT in decentre** -- 6.2e-7 waves across
  0-1.5 w on the f/3, 1.8e-9 on the f/6w, and monotone 0.028 -> 0.152 EE3
  points across design 121's whole fan.  Its disc is beam-sized and
  beam-centred at every offset, so nothing about it changes as the beam moves;
* the CONCENTRIC branch's error **GROWS**, 400x on the f/3 and 1400x on the
  f/6w across the same sweep, because its disc is sized from the
  ORIGIN-referenced second moment `sqrt(2 c^2 + w^2)`.  That is an artefact of
  measuring about the wrong point -- the D1 defect -- not a physical radius, so
  the same total-degree budget is spread over a disc inflated by
  `sqrt(1 + 2 (c/w)^2)`: 1.22x at 0.5 w, 1.73x at 1.0 w, 2.35x at 1.5 w.  That
  is the P2 aperture:beam cliff re-entering through the back door
  (`_FIT_RADIUS_BEAM_FACTOR_DEFAULT`).

**The crossover is therefore where a GROWING curve crosses a FLAT one, and its
position is set by the OFFSET between them** -- i.e. by whether an
order-`newton_poly_order` fit is already enough for that lens.  On the f/3 the
concentric floor is 7x BELOW the off-centre one, so it takes 0.55 w of
inflation to lose; on the f/6 it is above it from the first pixel.  Nothing
about that is a function of `|c|/w`.

### 3.3 The compound intervention, separated

Forcing a branch moves four things at once (`D121_RESIDUAL_CLOSURE` S5).  Two
of them -- the restriction and the order -- are separable at the element, and
the separation matters because it is the reason a predictor cannot work:

| `\|c\|/w` | conc (mask, ord 6) | off (weighted, ord 10) | **off6** (weighted, ord 6) | **conc10** (mask, ord 10) |
|---|---|---|---|---|
| f/3 0.00 | 9.37e-08 | 9.37e-08 | 9.37e-08 | *1.57e-05* |
| f/3 0.20 | 7.91e-08 | 6.54e-07 | 1.25e-05 | *1.98e-05* |
| f/3 0.50 | 4.13e-07 | 6.36e-07 | 1.67e-05 | 8.32e-06 |
| f/3 1.00 | 5.38e-06 | 5.26e-07 | 2.71e-05 | 2.10e-06 |
| f/3 1.50 | 3.73e-05 | 2.89e-07 | 3.50e-05 | 1.59e-07 |
| f/6w 0.00 | 3.01e-09 | 3.01e-09 | 3.01e-09 | *3.28e-06* |
| f/6w 1.00 | 6.26e-07 | 1.79e-09 | 3.21e-07 | 6.14e-08 |

Two readings.

**(a) D7's order raise is load-bearing and it is not optional.**  `off6` --
D1's weighted restriction at the caller's own order -- is **19x to 310x worse**
than `off` on every nonzero-decentre row of both geometries (19x at f/3 0.20 w,
121x at f/3 1.50 w, 310x at f/6w 0.20 w).  The weighted branch without the
raise is the worst configuration measured.

**(b) The hard mask at order 10 is unstable, which is D1's finding reproduced
from a new direction.**  `conc10` is 167x worse than `conc` at zero decentre on
the f/3 and 1000x worse on the f/6w, and it is NON-MONOTONE in decentre
(4.3e-4 at 0.4 w, 8.3e-6 at 0.5 w, 2.1e-6 at 1.0 w).  A hard mask leaves the
fit's remaining freedom unconstrained, and more freedom is worse.

So the two branches are each internally coherent -- mask with low order,
weights with high order -- and the crossover is between two *packages*, not
between two discs.

### 3.4 Four predictors, derived, and why none of them ships

The brief asks for a predictor from quantities in hand rather than a tuned
constant.  Four were derived and priced.  `f` is `fit_radius_beam_factor`,
`u = |c|/w`, `Lr` the launch radius.

**P1 -- REACH.**  The off-centre disc reaches `|c| + f w` into the aperture;
the concentric one reaches `f sqrt(2 c^2 + w^2)`.  Prefer the one that reaches
less far into aberrated territory.  They cross at

>  `u* = 2f / (2 f^2 - 1)`  =  **0.5714** at the default `f = 2`.

**P2 -- INFLATION against the P2 cliff.**  Keep the concentric disc while it
has not inflated past the aperture:beam cliff, whose own note puts the cliff at
"the launch square growing past ~2.5x the beam radius" against a requested
`2.0`:

>  `sqrt(1 + 2 u^2) <= 2.5 / 2.0`  ->  `u* = sqrt((1.5625 - 1)/2)` = **0.5303**.

**P3 -- CONTAINMENT.**  Require the concentric disc to hold the whole beam
disc.  Algebraically identical to P1 (`f sqrt(2c^2+w^2) >= c + f w`), so the
same 0.5714 -- and REFUTED on inspection: at small `c` the concentric radius
grows as `O(c^2)` while the beam's far edge grows as `O(c)`, so containment
fails immediately above zero and recovers at 0.5714 w.  It is the same
inequality read with the wrong sign; recorded so the next reader does not
re-derive it.

**P4 -- SAMPLE COUNT.**  The in-disc counts (the quantity D7's step-down
already uses) do not separate the branches at all: on the f/3 they are 213 /
221 at 0.06 w and 217 / 657 at 1.0 w, i.e. the concentric disc has MORE samples
exactly where it is worse.

**How they score.**  P1 and P2 land at 0.571 and 0.530, both inside design
121's measured (0.48, 0.72) bracket and both within 4 % of the f/3 singlet's
measured 0.545 -- which is a genuinely good showing for two tuning-free
derivations, and is worth recording as such.  **And both are simply wrong on
the f/6 geometries, where the measured crossover is 0 w and they would keep the
concentric branch out to 0.53-0.57 w** -- a branch that is 3-1000x worse there
from the first pixel.  They are also wrong per-group on design 121, whose six
groups cross at 0.46, 0.48, 0.60, 0.61, 0.69 and (for the low-decentre groups)
never.

A constant cannot express "it depends on whether order 6 was enough for this
lens", and that is what the quantity is.  **Defended negative on the predictor
route.**

---

## 4. The arbiter

### 4.1 What it does

At the fit site the rays have already been traced and both discs already exist.
So both candidates are built there and compared:

```
w_i    = exp(-2 |r_i - c|^2 / w^2)            beam intensity on the launch lattice
S(cand)= sqrt( sum_i w_i (P_cand(r_i) - opl_i)^2 / sum_i w_i )
```

where `P_cand` is the OPL fit built exactly as that candidate would be applied
-- hard NaN mask at `newton_poly_order`, or D1's weighted restriction at D7's
raised order with its sample-count step-down -- and `opl_i` is the UNMASKED
traced OPL.  Smaller `S` wins; an exact tie keeps the concentric candidate.

Three properties, all deliberate:

* **it is scored on the beam, not on the candidate's own domain.**  A fit that
  is clean only where it was allowed to look scores badly, which is the failure
  mode of the concentric mask;
* **the candidate scored is the candidate applied.**  Both go through the same
  `_decentred_fit_restriction`, so an order-6 trial can never be applied at
  order 10.  Pinned by a test;
* **it is gated on `_beam_decentred`.**  Below the C1 null gate it does not
  run at all, so every C1 byte-identity contract is untouched by construction,
  in both flag states.

Cost: one extra `_Cheb2DEvaluator` OPL fit per branch (the rays, the trace and
both discs already exist) plus one extra second-moment pass over `|E_in|^2`,
taken only above the C1 gate.  Measured as an exact count in the test suite:
**5 evaluator builds where the pure gate makes 3**.

**Why the OPL alone.**  It is the quantity that becomes phase.  The transverse
map residual was measured alongside it on every synthetic row and ranks the
branches identically; adding it would mean combining two quantities with
different units and no principled scale.  Recorded as measured-and-not-used.

### 4.2 It agrees with the oracle 42 times out of 42

`c11_synth_sweep.py`, three geometries x 14 decentres, the arbiter's pick
against the spline oracle's verdict:

| geometry | rows | agreements | crossover, oracle | crossover, arbiter |
|---|---|---|---|---|
| f/3, `w` = 1.0 mm | 14 | **14** | between 0.50 and 0.60 w | **0.5454** (bisected) |
| f/6, `w` = 1.0 mm | 14 | **14** | 0 w | 0 w |
| f/6, `w` = 1.4 mm | 14 | **14** | 0 w | 0 w |

Not just the sign: the magnitudes track.  On the f/3 the arbiter's concentric
score runs **1.6e-7 -> 1.05e-4 waves** across the sweep while the oracle's
concentric error runs **9.4e-8 -> 3.7e-5** -- three decades of growth, tracked
with the oracle/score ratio staying inside 0.36-0.60 throughout; the off-centre
score is flat at 7-8e-7 against a flat 6e-7.  **The arbiter is an estimator of
the exit wavefront error good to a factor under 2 over three decades, not
merely a tie-break** -- which is why its verdict at an 18x margin (S4.3) is
worth acting on and its verdict at a 1.1x margin (S6.4) is not.

### 4.3 On design 121, group by group

`c11_discrim_121.py`.  Each of the chain's six post-DOE element calls is
captured once and replayed with each branch forced; the replay is aborted as
soon as the three fits exist.  Beam-weighted OPL residual, in waves:

| order | grp | `\|c\|/w` | concentric | off-centre | ratio | pick |
|---|---|---|---|---|---|---|
| (-1,0) | 0-5 | 0.013 .. 0.241 | 1.1e-4 .. 6.0e-3 | 1.9e-4 .. 8.0e-3 | 0.21-0.91 | **conc x6** |
| (-2,0) | 4 | 0.398 | 6.79e-3 | 6.96e-3 | 0.975 | conc |
| (-2,0) | 5 | **0.481** | 1.73e-2 | 7.80e-3 | 2.215 | **off** |
| (-3,0) | 3 | 0.459 | 2.29e-3 | 4.34e-3 | 0.527 | conc |
| (-3,0) | 4 | **0.597** | 7.41e-3 | 6.74e-3 | 1.099 | **off** |
| (-4,0) | 2 | 0.249 | 2.89e-3 | 1.01e-2 | 0.287 | conc |
| (-4,0) | 3 | **0.613** | 5.04e-3 | 4.79e-3 | 1.052 | **off** |
| (-4,0) | 4 | 0.796 | **1.22e-1** | 6.78e-3 | **17.99** | **off** |
| (-4,-2) | 2 | 0.279 | 3.30e-3 | 1.17e-2 | 0.282 | conc |
| (-4,-2) | 3 | **0.685** | 3.62e-2 | 5.06e-3 | 7.157 | **off** |

The per-group flips land at **0.46-0.69 w**, reproducing the (0.48, 0.72)
bracket the chain-level arms give -- from the fits alone, with no constant, on
a design whose regime the spline oracle cannot reach at all.  And at (-4,0)
group 4 the catastrophe that costs 67 EE3 points downstream is visible at the
decision site as an **18x** margin, before any field is reconstructed.

---

## 5. The boundary, held to niche C1's own standard

C1's finding was a discontinuity: **8.32e-6 of peak at 1e-9 PIXELS**, "100x the
pipeline's ~1e-7 roundoff floor and bought by nothing".  `c11_boundary_step.py`
measures the same quantity at the new boundary.

### 5.1 The null contract survives, bitwise, in both states

f/6 and f/3 fixtures, C1's own offsets, `max|dE|/max|E|` against the
origin-referenced arm AND across the flag:

| offset | vs origin-referenced | vs arbiter OFF |
|---|---|---|
| 1e-9 px | **BYTE-IDENTICAL** | **BYTE-IDENTICAL** |
| 0.4 px | **BYTE-IDENTICAL** | **BYTE-IDENTICAL** |
| 1 px | **BYTE-IDENTICAL** | **BYTE-IDENTICAL** |
| 0.02 w | **BYTE-IDENTICAL** | **BYTE-IDENTICAL** |
| 0.049 w | **BYTE-IDENTICAL** | **BYTE-IDENTICAL** |

10 of 10 on each fixture.  This is structural, not lucky: the arbiter is gated
on `_beam_decentred`.

### 5.2 C1 RELOCATED its discontinuity; it did not remove it

Measured now, on the f/3 fixture, across `0.05 w` at `+-1e-6` relative:

| | step at the 0.05 w gate |
|---|---|
| arbiter **OFF** (v5.32) | **8.338e-06** |
| arbiter **ON** | **8.596e-08** |

**8.338e-6 is the C1 note's own 8.32e-6**, to three digits -- the same jump,
still there, just moved from 1e-9 px to 0.05 w, where nothing physical happens
either.  The arbiter takes it down **97x**, to the geometry's own smoothness
(8.7e-8 and 1.1e-7 are what the same measurement reads on the f/6 fixtures,
where no branch changes).

### 5.3 The step at the new boundary, and why it is bounded

f/3, boundary bisected to `|c|/w = 0.54544 +- 7.6e-6`:

| about the boundary | `max\|dE\|/max\|E\|` | candidate OPL residuals there |
|---|---|---|
| `+-1e-6` | **6.131e-06** | 9.350e-13 / 1.001e-12 m |
| `+-1e-4` | 9.340e-05 | 9.351e-13 / 1.001e-12 m |
| `+-1e-3` | 9.323e-04 | 9.351e-13 / 1.001e-12 m |
| `+-1e-2` | 9.324e-03 | 9.353e-13 / 9.940e-13 m |
| `+-1e-3`, arbiter OFF | 9.323e-04 | *(no swap there)* |

The `+-1e-3` row is **identical with and without the arbiter**, so from 1e-3
outwards the number is the beam physically moving, not the branch swapping.
Extrapolating that linearly to `+-1e-6` gives ~9.3e-7, so the swap itself
contributes **~5.2e-6 of peak**.

**And that is exactly the two candidates' shared accuracy.**  At the boundary
they score 9.35e-13 and 1.00e-12 m of OPL residual -- within 7 % of each other,
by construction -- which is 7.1e-7 and 7.6e-7 waves, i.e. **4.5e-6 and 4.8e-6
radians**.  A phase disagreement of 4.5e-6 rad is a field disagreement of
4.5e-6 of peak.  The measured 5.2e-6 is that number.

This is the structural argument, and it is the one thing a constant gate can
never have: **the arbiter's boundary is, by definition, the locus where the two
models are equally good, so the step across it is bounded by the accuracy they
share there.**  A fixed gate's step is whatever the two branches happen to
differ by at that decentre -- which on design 121 at 0.965 w is 67 EE3 points.

---

## 6. Design 121

### 6.1 The per-order table

`c11_gate_arms_121.py` (a copy of `rc_gate_121.py` with only `ARMS` changed, so
the instrument, readout and scoring are the campaign's own), `RN=1024`, `rs=4`,
`NLO=321`, area-exact, both arms pinned EXPLICITLY so neither depends on what
the default happens to be:

| order | `\|c\|/w` | oracle | **C11** | v5.32 | **res C11** | res v5.32 | delta |
|---|---|---|---|---|---|---|---|
| (0,0)   | 0.000 | 90.5324 | 90.5801 | 90.5801 | **-0.0477** | -0.0477 | *byte-identical* |
| (-1,0)  | 0.241 | 90.5768 | 90.5215 | 90.5477 | **+0.0552** | +0.0290 | **+0.0262** |
| (-2,0)  | 0.481 | 90.6650 | 90.6186 | 90.6016 | **+0.0464** | +0.0634 | -0.0170 |
| (-3,0)  | 0.723 | 90.6961 | 90.6586 | 90.6062 | **+0.0375** | +0.0898 | -0.0523 |
| (-4,0)  | 0.965 | 90.5035 | 90.4727 | 90.3625 | **+0.0308** | +0.1410 | -0.1102 |
| (-4,-2) | 1.079 | 90.1071 | 90.0378 | 89.9554 | **+0.0693** | +0.1517 | -0.0824 |

* **worst-case residual 0.1517 -> 0.0693**, 2.2x;
* **field-angle spread 0.1994 -> 0.1170**;
* the residual's monotone growth with field angle is gone: v5.32 rises
  0.029 -> 0.063 -> 0.090 -> 0.141 -> 0.152; C11 reads 0.055 / 0.046 / 0.038 /
  0.031 / 0.069, with no trend.  The exact-ray oracle says every order is
  EQUALLY diffraction-limited, so a residual with no field-angle dependence is
  the shape a correct chain should have, and a rising one is the shape of a
  decentre-driven model error.

### 6.2 The arbiter beats BOTH fixed branches at four of five tilted orders

Against S3.1's forced arms, residual in points (smaller magnitude better):

| order | **C11** | forced concentric | forced off-centre | best fixed branch | C11 - best fixed |
|---|---|---|---|---|---|
| (-1,0)  | 0.0552 | -0.0985 | **0.0275** | 0.0275 | **+0.028** |
| (-2,0)  | **0.0464** | -0.1267 | 0.0623 | 0.0623 | **-0.016** |
| (-3,0)  | **0.0375** | 2.0331 | 0.0909 | 0.0909 | **-0.053** |
| (-4,0)  | **0.0308** | 67.3117 | 0.1410 | 0.1410 | **-0.110** |
| (-4,-2) | **0.0693** | 79.1452 | 0.1517 | 0.1517 | **-0.082** |

At (-4,0) and (-4,-2) v5.32 IS the off-centre branch byte-identically, so the
0.110 and 0.082 are bought entirely by routing the SMALL-decentre groups (0-2,
at 0.05-0.28 w) onto the concentric fit -- which is the half of the brief's
premise that survives.

### 6.3 The one order that moves the wrong way -- and why the flag ships OFF

**(-1,0) gets worse by 0.0262 points**, against a differential floor of
0.003-0.015.  That is a real move, 2-9x the floor.  It is not noise and it is
not being averaged into the headline.

If the acceptance bar "improve or hold at EVERY order" is read per-order, this
change **does not meet it at (-1,0)**.  If it is read on the table's range
(`-0.048 .. 0.152` -> `-0.048 .. 0.069`) it improves it strictly.

**Resolved by shipping the arbiter OPT-IN** (`DECENTRED_FIT_ARBITER = False`),
for two reasons that point the same way:

* the per-order bar is the stricter reading and this change fails it, so the
  default flip is a judgement about design 121 rather than a library fact, and
  it belongs to an explicit decision with this table in front of it;
* flag-off is a genuine NO-OP, not a fall-back (S6.5 measures it on the
  design, and a test asserts the scoring function is never even CALLED), so the
  whole C11 layer -- and the four test reconciliations in S8.3 and the three CI
  ones in S9 -- can ride a patch release without a physics re-verification
  cycle.

`D121_RESIDUAL_CLOSURE` S6.3 (e) is the precedent for shipping WITH a stated
counter-movement (C10 shipped although the last group's element pass got
0.001-0.005 waves worse).  The difference here is that C10's counter-movement
was in a different currency from its gain; this one is in the SAME currency at
the SAME orders, which is what makes it a decision rather than a trade.

### 6.4 Why (-1,0) moves the wrong way -- the C6 domain, separated at last

At (-1,0) all six groups pick concentric, so C11 differs from the forced
`concentric` arm in exactly ONE thing: the niche-C6 residual-eikonal fit domain
(consumer 6), which is beam-centred whenever `_beam_decentred` and cannot be
revisited after the trace.  Three arms, same order, same everything else:

| ray fit | C6 residual-eikonal domain | residual |
|---|---|---|
| OFF-CENTRE | beam-centred | **0.0275** |
| CONCENTRIC | beam-centred (**this is C11**) | **0.0552** |
| CONCENTRIC | origin-referenced | **-0.0985** |

Two readings, and they are the point of this section.

**(a) The prior study's attribution is reversed.**  Moving the C6 domain alone
is worth **0.154 points**; moving the ray-fit branch alone (C6 held
beam-centred) is worth **0.028**.  `D121_RESIDUAL_CLOSURE` S5 said the
"concentric" intervention was compound and listed the C6 domain as unseparated
(its S7 item 5).  Separated, **85 % of "the concentric branch wins at 0.24 w"
is the C6 fit domain, not the ray fit.**

**(b) The arbiter's pick at (-1,0) is, by this measurement, wrong** -- with C6
held fixed the off-centre ray fit is better by 0.028 points, while the arbiter's
own scores prefer concentric by 1.1-4.8x.

Two things about that are worth separating.

The score is a PROXY: it measures how well each polynomial reproduces the
traced OPL over the beam, and the chain's EE3 also depends on how the Newton
inversion uses those polynomials over the whole launch square (which the
intensity weight, ~0 beyond 2 w, cannot see) and on the map's derivatives (the
ray-density amplitude).  So the proxy is expected to be reliable when the
margin is LARGE and unreliable near a tie -- and that is exactly what the data
show.  Collecting every case where the chain-level verdict is known:

| where | arbiter margin | chain-level outcome |
|---|---|---|
| (-4,0) grp 4 | **18.0x** | correct; part of a 0.110-point gain |
| (-4,-2) grp 3 | **7.2x** | correct; part of a 0.082-point gain |
| (-2,0) grp 5 / (-3,0) grp 4-5 | 1.1-2.4x | correct; 0.017 and 0.052 gains |
| (-1,0) all groups | **1.1-4.8x, mixed** | **wrong by 0.028 points** |
| f/3 singlet, 14 rows | 1.02x at the crossover to 316x at 1.5 w | **14/14 correct** |

**Every decision at a large margin is right, and the one wrong answer is on a
set of near-ties.**  That is the behaviour a proxy should have, and it bounds
the damage: a near-tie is by definition a place where the two candidates are
close, so picking the worse one costs little -- 0.028 points here against the
67 points the same mechanism avoids at (-4,0).

Whether it is FIXABLE is a different question and the answer is not obviously
yes.  The natural candidate -- an admissibility check rejecting a candidate
whose fitted map is non-monotone over the launch square (D1's own detector) --
would not change this case: (-1,0)'s groups sit at 0.013-0.241 w, where the
concentric fit is nowhere near folding.  Re-weighting to reach outside the beam
would re-import the aperture:beam cliff the weight exists to avoid.  So this
may simply be the level at which a fit-residual proxy stops tracking a
chain-level image metric, and the honest statement is that it is 0.03 points.
S10 item 1.

### 6.5 The shipped default is a NO-OP on the design, bit for bit

`c11_gate_arms_121.py` with a third arm that patches NOTHING, so the `default`
column is whatever the module ships:

| order | oracle | default | C11 (`True`) | v532 (`False`) | sha `default` / `C11` / `v532` |
|---|---|---|---|---|---|
| (0,0)   | 90.5324 | 90.5801 | 90.5801 | 90.5801 | `4e9effd4` / `4e9effd4` / `4e9effd4` |
| (-1,0)  | 90.5768 | 90.5477 | 90.5215 | 90.5477 | `b2a8b150` / `098aa4a2` / `b2a8b150` |
| (-2,0)  | 90.6650 | 90.6016 | 90.6186 | 90.6016 | `88f726eb` / `340d7b32` / `88f726eb` |
| (-3,0)  | 90.6961 | 90.6062 | 90.6586 | 90.6062 | `cf0bc1f3` / `71af633a` / `cf0bc1f3` |
| (-4,0)  | 90.5035 | 90.3625 | 90.4727 | 90.3625 | `7845b7a8` / `fe6b14c8` / `7845b7a8` |
| (-4,-2) | 90.1071 | 89.9554 | 90.0378 | 89.9554 | `c4b850ea` / `be68c661` / `c4b850ea` |

**`default` == `v532` on 6 of 6 orders, bit for bit** -- and both reproduce
`rc_gate_121.py`'s `shipped` shas from BEFORE any C11 code existed
(`_c11_gate_deg6.txt`, a different runner in a different process on the
pre-change tree).  So the shipped default is byte-identical to v5.32.0-class
behaviour **on the design**, not merely by inspection of the guards, and the
`C11` column differs on every tilted order, which is what makes the opt-in a
real switch rather than a dormant one.

A unit test asserts the same thing from the other side: with the flag `False`
the arbiter's scoring function is **never called** on a call that would
otherwise reach it (`test_the_arbiter_ships_off_as_an_opt_in`), so this is a
path that is not taken rather than a computation whose result is discarded.

### 6.6 Acceptance

**(a) Production acceptance does not regress -- it is identical, to every
printed digit including the peak.**  `focus_scan_121.py` **unedited**, pure
library defaults (`CREF`/`AM`/`PIP` unset), N=2048, `rs=4`, NFC=8192, WF=4.0,
NOUT=2048, run twice through `c11_with_arbiter.py` with the flag pinned either
way so neither arm depends on what the default happens to be:

| | **arbiter ON (shipped)** | **arbiter OFF (fail-before)** |
|---|---|---|
| `AT-PLANE` | 3.350 um / 90.3 / 99.7 / 99.8 | 3.350 um / 90.3 / 99.7 / 99.8 |
| `BEST-FOCUS[peak]` plane | dz = **+0 um** | dz = **+0 um** |
| FWHM / EE3 / EE6 / EE12 | 3.350 um / 90.3 / 99.7 / 99.8 | 3.350 um / 90.3 / 99.7 / 99.8 |
| peak | **5.529e+03** | **5.529e+03** |
| `dz = +5 um` | 3.450 um / 89.6 / 99.7 / 99.8 | 3.450 um / 89.6 / 99.7 / 99.8 |

Both arms also reproduce `D121_RESIDUAL_CLOSURE` S6.3 (a)'s recorded degree-6
line exactly, peak included -- so this is the campaign's own measurement, not a
new one.  **The recorded acceptance line is unchanged.**

That it is IDENTICAL rather than merely close is structural: `focus_scan_121.py`
runs the single on-axis beam, where the chief ray is on the grid centre, the C1
null gate never opens and the arbiter therefore never runs.  The production
route also re-traces the last group on a fine grid where much of this is inert
(`_FIT_DISC_OUTSIDE_WEIGHT_REL`'s own note).

**(b) The fail-before is bit-exact on the DESIGN.**  `DECENTRED_FIT_ARBITER =
False` reproduces all six orders' intensity arrays bit for bit against
`rc_gate_121.py`'s independently-run `shipped` arm: `4e9effd4` / `b2a8b150` /
`88f726eb` / `cf0bc1f3` / `7845b7a8` / `c4b850ea`, 6 of 6, two processes, two
runners.

**(c) Conservation and halo, on the campaign's own instrument, 4 of 4.**
`energy_stage_audit_121.py` **unedited**, through `c11_with_arbiter.py`,
`RN=1024`, `rs=4`, six post-DOE groups, `final_leg='paraxial'`, with the
arbiter ON.  Against `D121_RESIDUAL_CLOSURE` S5.3 / S6.3 (b)'s recorded
degree-6 numbers:

| order | `P_out/P_in` C11 | (C10 record) | `g4` C11 | (record) | `amax4` C11 | (record) | `r_rms` mm |
|---|---|---|---|---|---|---|---|
| (0,0)   | 0.994315 | 0.994315 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.8383 |
| (-1,0)  | 0.994063 | 0.994065 | 1.962e-11 | 2.663e-11 | 1.716e-05 | 1.309e-05 | 0.8384 |
| (-2,0)  | 0.994132 | 0.994133 | 6.783e-11 | 7.659e-11 | 3.213e-05 | 3.326e-05 | 0.8382 |
| (-4,-2) | 0.993826 | 0.993843 | 9.114e-09 | 9.694e-09 | 1.116e-04 | 1.117e-04 | 0.8375 |

`P_out/P_in` moves by **<= 1.7e-05** and stays inside C2's [0.9850, 1.00050];
every `g4` is 1e-3 or less of its C3 bound and two of the three nonzero ones
move DOWN; `amax4` stays 10x under the C4 bound of 1.0e-03; `r_rms` moves by
<= 0.0001 mm against a C5 tolerance of 0.030.  **Every bound is met on every
order, and the halo currency does not see the branch change** -- which is the
matched-pair statement the campaign has been making since the energy audit,
now made in the reassuring direction.

**(d) The C7 halo self-check and the element energy self-check stay silent.**
`grep -c "HALO self-check FAILED"` and `grep -c "energy self-check FAILED"`
both read **0** across the energy audit, the 386-test niche selection, the
146-test CI-proxy run and both production focus scans.

**(g) Lint.**  `python -m ruff check lumenairy/ tests/unit/` -> All checks
passed.

---

## 7. Generality: it is not tuned to design 121

Nothing in the arbiter has a constant to tune, so "tuned to one design" is not
available to it in the way it would be to a gate value.  What can still be
wrong is the SCORE, and that was exercised on geometries that share no code,
no prescription and no regime with 121:

| geometry | type | aperture:beam | what the arbiter does | checked against |
|---|---|---|---|---|
| f/6 N-BK7 singlet, `w` = 1.0 mm | spherical | 5.0 | off-centre at every nonzero decentre | spline oracle, 14/14 |
| f/6 N-BK7 singlet, `w` = 1.4 mm | spherical | 3.6 | off-centre at every nonzero decentre | spline oracle, 14/14 |
| f/3 N-BK7 singlet, `w` = 1.0 mm | spherical | 5.0 | concentric to 0.545 w, off-centre above | spline oracle, 14/14 |
| niche-D6 `K = -n^2` conic, `f` = 3 mm | **aspheric**, exit NA 0.20 | 2.8 | off-centre, margin **27.7x** at 1.0 w and **30.3x** at 1.5 w | see below |
| design 121, six groups x six orders | 8-surface relay, exit NA 0.41 | 3.3 | per-group flips at 0.46-0.69 w | chain-level EE3, S6 |

**The conic could not be scored against the spline oracle** and this is not a
gap in the instrument but the documented behaviour of that oracle: niche C2's
note in `_FIT_DISC_OUTSIDE_WEIGHT_REL` records that `newton_fit='spline'`
returns an ALL-ZERO field when it cannot converge, silently under
`on_undersample='silent'`.  It does so here at N=1024 AND at N=2048 (whose
Nyquist direction cosine 0.224 clears the 0.20 exit NA) AND on a slowed `f` = 6
mm variant at NA 0.10 -- so the failure is not purely the NA.  The runner
detects the all-zero return and REFUSES to score rather than reporting the
comparison against nothing, which is the trap that note exists to warn about.

For the conic the verdict therefore comes from niche D6's own suite, whose
oracle is lumenairy-free (an exact conic raytrace plus a direct
Rayleigh-Sommerfeld sum) and whose stand-in is stigmatic at EVERY decentre --
so its truth is decentre-invariant by construction.  See S8.3.

**The two spherical singlets disagree with each other**, which is the load
this study most needed: same beam, same decentre, same gate, same
`fit_radius_beam_factor`, two lenses, two different correct answers.  That is
pinned as a test (`test_the_crossover_is_design_dependent_not_a_constant`) as
an exact identity rather than a threshold.

---

## 8. What shipped

### 8.1 The diff

| file | before | after | what |
|---|---|---|---|
| `lumenairy/elements/_lens_traced.py` | `34ef5a9d95279b8f` | `8b13259592233b52` | one flag (shipped `False`) + two helpers + the arbiter block |
| `lumenairy/propagators/carrier.py` | `5a1b0d1021969df1` | `5a1b0d1021969df1` | **unmodified** |
| `tests/unit/test_niche_c11_decentred_fit_arbiter.py` | | | new |
| `tests/unit/test_niche_c1_consolidation.py` | | | one era pin (assertions verbatim) |
| `tests/unit/test_niche_c6_fit_guard.py` | | | one pinned attribute in one arm |
| `tests/unit/test_niche_d7_decentred_fit.py` | | | two instrument-scope fixes, kept LIVE |
| `tests/unit/test_niche_d3_guards.py` | | | one era pin + one sibling (S9.1) |
| `tests/conftest.py` | | | autouse module-flag leak guard (S9.5) |
| `CHANGELOG.md`, `lumenairy/elements/pmm/**` | | | **untouched** |

`tests/conftest.py` is the only change outside the niche files, and it is
test-infrastructure: an autouse fixture that restores physics-mode flags after
each test.  It has no effect on any library code path.

No signature moved, no public entry point added, no other default flipped.
`_DECENTRE_GATE_PIXELS` and `_DECENTRE_GATE_W_FRAC` are **unchanged at 0.5 and
0.05** -- they keep exactly the job C1 gave them (the null floor) and lose the
job they were never fit for.

The library change is:

* `DECENTRED_FIT_ARBITER = True`, with the derivation, the measurements and
  the fail-before in its note;
* `_decentred_fit_restriction(disc, weighted, base_order, dec_order)` -- D1's
  weight formula and D7's order step-down, factored out of the fit site so the
  arbiter scores what it applies.  The fit site now calls it, so the arithmetic
  exists once;
* `_decentred_fit_score(xs_in, opl_grid, weight, disc, weights, order)` --
  the beam-weighted OPL residual, returning `inf` for an unscoreable candidate
  so an inadmissible one can never win;
* at the gate site, the origin-referenced radius is ALSO measured when (and
  only when) the arbiter can run;
* at the fit site, the two candidates are built, scored and one is applied.

### 8.2 Tests added -- 20, ~90 s, no proprietary asset

`tests/unit/test_niche_c11_decentred_fit_arbiter.py`.  Every numeric bar is a
RATIO between two arms measured in the same process on the same fixture, or an
exact-arithmetic identity; **there is no absolute bar on a BLAS-dependent
magnitude anywhere.**

1. the flag ships `True`;
2. **the fail-before, both ways**: `False` changes the f/3 fixture at 0.2 w and
   is bit-identical on the f/6 one at the same decentre -- so the switch is
   live AND the arbiter is not a blanket branch revert;
3-12. **the C1 null contract**, five offsets x two fixtures, `array_equal`
    against the origin-referenced arm and across the flag;
13. **the step moves off the gate**: the 0.05 w jump with the arbiter off must
    exceed the geometry's own smoothness (measured at 0.10 w, same process) by
    10x, and the arbiter must take it below 0.1x of that and within 3x of the
    smoothness;
14. **the crossover is design-dependent**, as an exact identity: at 0.2 w the
    f/3 field IS the concentric arm bitwise and the f/6 field is not;
15. the f/6 fixture still routes to the weighted raised-order path (the live
    scoping of C1's own claim);
16-17. **scored == applied**: exactly 2 scores and 5 evaluator builds, and the
    last three builds carry the WINNING trial's `(order, weighted)` pair, with
    the winner read from the library's own scores;
18. the score ranks an order-2 fit of a degree-2 map `1e-6` below an order-1
    one, weighted and unweighted;
19. an unscoreable candidate returns `inf`;
20. `_decentred_fit_restriction` reproduces D1's weight exactly
    (`array_equal`) and D7's step-down at its exact sample-count boundary, and
    never steps below the caller's own order.

### 8.3 Existing tests changed

**`test_niche_c1::test_a_genuine_decentre_still_routes_to_the_weighted_raised_order_path`.**
ERA-PINNED at `DECENTRED_FIT_ARBITER = False`, **assertions kept word for
word**, reason in its docstring.  Nothing was relaxed and no threshold moved.
What broke is the INSTRUMENT, not the claim: the test spies on every
`_Cheb2DEvaluator` build and asserts `all(...)` over them, and the arbiter adds
two TRIAL builds the spy was never written to see.  The claim itself is intact
on that fixture -- the arbiter picks off-centre at all three offsets -- and is
kept LIVE on the shipped default, scoped to the fits the Newton inversion is
handed, as C11 test 15.

**`test_niche_d7::test_no_fold_and_no_ghost_across_the_adversarial_geometries`
(11 cases) and `::test_the_fit_order_actually_rises_only_off_centre`.**  Same
cause, and this one was nearly a false alarm worth recording in full, because
for twenty minutes it read as **the arbiter causing D1's fold**.

Eleven adversarial geometries failed with `the fitted forward map folds (491)`.
That is the exact failure D1 exists to prevent, and the arbiter's known blind
spot (S10 item 1) is exactly "it cannot see out-of-beam behaviour", so the
diagnosis wrote itself.  It was wrong.

Two things did not fit.  The fold counts (455-573 on a 451x451 lattice) are
what ONE zero-crossing line produces, not a fold; and the test's other three
fold indicators -- the `fold caustic` warning, the off-beam lobe bound
(`< 0.02` of peak) and the off-beam power bound (`< 1e-5` of `P_in`) -- are
asserted BEFORE the sign-change one and all PASSED.  A field with a real fold
does not pass those.

Measured directly on the `x+` case:

| evaluator | order, weighted | sign changes in `d/dx` |
|---|---|---|
| `seen[0]` -- what the test read | (10, True) | **491** |
| `seen[-3]` -- the APPLIED `x_out` fit | (10, True) | **0** |

The test takes `Sx, xs = seen[0]` because a call used to build exactly three
evaluators.  The arbiter builds two TRIAL **OPL** fits ahead of them, and an
OPL has a vertex -- so `d(OPL)/dx` crosses zero by construction, once, along a
line.  The test was scoring the sign changes of an optical path length and
calling it a folded forward map.

**The arbiter picked the OFF-CENTRE branch on all eleven** (the applied fits
read `(10, True)`), which is what D1/D7 require, and the applied map does not
fold.  Both tests are therefore kept LIVE on the shipped default with the
handle moved to `seen[-3]` / `seen[-3:]` and the measurement above in the
comment -- no era pin, no threshold moved, no assertion relaxed.

**`test_niche_d7::test_c10_shrinks_this_fixtures_hard_mask_ghost` ->
`test_the_hard_mask_arm_ghosts_on_every_build`.**  Its magnitude assertion was
removed outright after a third build showed a third answer; see S9.2.  This is
the one place in this study where a claim was RETIRED rather than re-scoped,
and the reason is that the quantity is the conditioning of a deliberately
ill-conditioned solve.

**`test_niche_d3_guards::test_the_separation_survives_...`.**  Its own
mechanism assertion moved from the passed pair (1.04-1.79x across builds) to
the refused fan (14-19x); see S9.1.

**`test_niche_c6_fit_guard::test_guard_reproduces_the_offcentre_branch_exactly`.**
Its device is to drive the C1 gates NEGATIVE so a null decentre counts as off
centre, which reaches the weighted branch with the disc unchanged.  That device
now also wakes the arbiter, which at a null decentre (identical discs, guard
off in that arm) correctly prefers the plain concentric candidate -- so the
device stops reaching the branch it names.  ONE pinned attribute was added to
that arm (`DECENTRED_FIT_ARBITER = False`) and nothing else; the assertion is
the original.  The other arm is untouched because the guard path has
`_beam_decentred = False` and the arbiter never runs there.

This is the third and fourth instance in four days of the pattern
`D121_RESIDUAL_CLOSURE` S9 item 9 named (a witness eroding because the library
changed under it).  Both are the *other* failure mode of that pattern: not a
guard going quiet, but a positional instrument (`seen[0]`, a negative-gate
device) becoming ambiguous when a second candidate appeared.  **The lesson is
narrower and more actionable than the original: an instrument that indexes into
a list of side effects is pinning an implementation's SHAPE, not its claim, and
the fix is to index from the end that the claim is about.**

### 8.4 Suites

**Every file this change touches, run on BOTH builds:**

```
pytest tests/unit/test_niche_{p2_guards,d3_guards,d7_decentred_fit,
                              c1_consolidation,c6_fit_guard,c11_*}.py -q
  Windows / MKL      py3.14.6  numpy 2.4.4   ->  158 passed, 20 warnings
  Linux / OpenBLAS   py3.12.3  numpy 2.4.6   ->  158 passed, 20 warnings

... and after the S9.1 / S9.2 bar revisions:
  d3_guards + d7_decentred_fit (Windows/MKL)  ->  75 passed, 1 warning
  c11 arbiter suite (Windows/MKL)             ->  20 passed
  p2_guards (Linux/OpenBLAS)                  ->  12 passed

ruff check lumenairy/ tests/unit/
  Windows           ->  All checks passed
  Linux (CI's own)  ->  All checks passed
```

**Shard compositions, end to end, this box's split:**

```
pytest tests/unit -m "not integration and not slow" --splits 3 --group 3 -q
  ->  2122 passed, 12 skipped, 2 failed   (46m58s)
      the 2 are test_v5_20_2_pmm_jones_2d_jax[inplane] and
      test_v5_20_8_rcwa_threaded_sweep -- PRE-EXISTING, see S9.5

shard 2/3, the 12-file / 327-test prefix through the reported victim
(test_niche_d4_dgrating::...::test_matches_the_manual_hand_split):
  ->  327 passed            (normal)
  ->  327 passed            (LUMEN_TEST_FLAG_LEAK_STRICT=1)
```

The prefix is the causally relevant part of shard 2: tests that run AFTER the
victim cannot have leaked into it.  The full 3670-test shard and a strict-mode
sweep of every flag-writing file (`tests/unit/test_niche_*.py`, Linux/OpenBLAS)
were both still running when this was written -- **zero failures at 84 %** on
the latter.  S10 item 11.

### 8.5 Cost

The arbiter adds, per element call ABOVE the C1 gate: one `_input_beam_amp_radius`
pass over `|E_in|^2` (banded, no full-grid temporary) and two
`_Cheb2DEvaluator` OPL fits on the coarse launch lattice.  Measured on the f/3
fixture (N=512, `ray_subsample=8`, 64x64 lattice), after numba warm-up and with
the box otherwise loaded, three alternating runs read **0.345 / 0.361 / 0.335 s**
with the flag on / off / on -- i.e. inside the run-to-run spread.  The fit
stage's evaluator count goes 3 -> 5 SEQUENTIALLY, so peak memory is unchanged;
on a large configuration (N=8192, `ray_subsample=8`) each design matrix is
~0.5 GiB and there are now five of them in sequence rather than three, which is
a wall-clock cost on the fit stage and not a footprint one.

`validation/run_all.py` is **unaffected**: no file it collects
(`rglob("test_*.py")` under `validation/`) passes `beam_centre`,
`fit_radius_beam_factor` or a `TiltedCarrier`, so the decentred branch is never
reached there in either flag state.

---

## 9. The v5.32.1 CI reconciliations (separate from the gate work)

PR #25 at `5af1edf` failed 4 shards across py3.10/3.12/3.13.  Both classes were
attacked with the same rule: reproduce, adjudicate against something
independent, then either re-pin with the calibration recorded or report a
regression.

### 9.1 D3 linearity -- REPRODUCED, ATTRIBUTED, RECONCILED

`tests/unit/test_niche_d3_guards.py::test_the_guarded_input_really_is_the_wrong_answer`.

It passes on Windows/MKL and fails on Linux/OpenBLAS -- the documented blind
spot -- so it was reproduced in the WSL CI proxy first:

```
E  AssertionError: a near-collinear pair the gate PASSES violated linearity
E  by 0.0575; the gate would then be missing a real failure
E  assert 0.0574576819618604 < 0.05
```

`c11_ci_recon.py` then swept the two knobs that moved plus the launch they sit
on.  `bad` is the 23 mrad fan the gate REFUSES, `good` the 0.5 mrad pair it
PASSES; both are relative-L2 violations of `chain(sum E_k) == sum chain(E_k)`,
which is exact for a passive optic:

| state | `bad` | `good` | ratio |
|---|---|---|---|
| C6 launch OFF | 0.623589 | **0.003766** | 165.6 |
| C6 on, resid degree 4 | 14.233651 | **0.032148** | 442.8 |
| C6 on, resid degree 6 (**shipped**) | 0.986256 | **0.057458** | 17.2 |
| C9 off, degree 6 | 0.986256 | 0.057458 | 17.2 |
| C9 off, degree 4 | 14.233651 | 0.032148 | 442.8 |

**Verdict: not a regression on any supported input; a real and attributed
increase in the error of the MULTIPLEXED route.**

* **niche C9 is NOT involved.**  Its two rows are identical to every printed
  digit.  The coordinator's hypothesis that C9 was implicated is refuted.
* the test's own docstring quotes `bad` = 62 %, which is the **C6-OFF** row
  (0.6236) exactly -- so this case was calibrated before the niche-C6
  stationary-phase launch existed.  C6 already moved `good` 0.0038 -> 0.0321;
  **C10's degree raise moved it 0.0321 -> 0.0575, across the 0.05 bar**;
* the mechanism is the expected sign, not a defect.  The C6 launch makes
  `apply_real_lens_traced` deliberately nonlinear in its input (it fits the
  input's residual eikonal and launches along `grad(W + a_fit)`).  A
  multiplexed input carries beat fringes that no single-valued residual model
  represents, so the better that model gets, the further the multiplexed route
  departs from the linear superposition -- while the superposition itself, the
  reference, got MORE accurate under C10 (that is C10's whole result).
  Degree 6 is also far LESS pathological on the fan the gate actually refuses
  (`bad` 14.23 -> 0.99);
* **the second assertion was broken too and would have been the next CI
  round**: `bad > 20 * good` reads 17.2 at degree 6.  It was never reached
  because the run stopped at the first failure.  Fixing only the reported one
  would have cost exactly the extra round the brief warns about.

**Reconciled** on the C9/C10 precedent: the case is era-pinned at
`_REMAP_RESID_EIKONAL_DEGREE = 4` with **all three assertions verbatim** and
the table above in its docstring, and its detector-agreement half -- which
reads only the INPUT's congruence statistics and is independent of the residual
model -- stays on the shipped default.  A new sibling,
`test_the_separation_survives_the_c10_residual_degree_and_is_caused_by_it`,
carries the shipped-era statement COMPARATIVELY -- and its own first revision
was ALSO a cross-build casualty, which is worth recording because the fix is
the interesting part.

The mechanism (the residual degree moves the multiplexed route) can be read on
either arm, and the two arms do not carry it with the same signal:

| `deg 4 -> deg 6` | Windows/MKL | WSL/OpenBLAS | CI/OpenBLAS |
|---|---|---|---|
| `good` (0.5 mrad, the PASSED pair) | 1.19x | 1.79x | **1.04x** |
| `bad` (23 mrad, the REFUSED fan) | **19.2x** | **14.4x** | -- |

A bar on `good` must live inside a 1.04-1.79x spread; the first revision put it
at 1.10x, which passed two builds and failed CI's by 6 %.  There is no value
that is both meaningful and safe.  **The same mechanism read on `bad` is 14-19x
on every build measured**, so the assertion moved there:
`bad4 > 5.0 * bad6` (3x headroom on the weakest build) plus
`bad6 > 5.0 * good6` for the separation itself (measured 17x-92x).  Nothing was
weakened -- the claim moved to where it is large.  Both green on both builds.

**What is NOT fixed by this, and should be someone's next question:** on the
shipped tree the D3 gate PASSES a 0.5 mrad pair whose answer is 5.7 % wrong in
L2.  The test's own logic calls that "missing a real failure".  The guard
THRESHOLDS (`_NONCOLLIMATED_RESID_THRESH`, `_MULTI_CONGRUENCE_MV_THRESH`) live
in `carrier.py`, which this study did not modify, and re-calibrating them is a
guard-behaviour change with its own blast radius.  **Flagged, not fixed.**

### 9.2 The D7 hard-mask ghost -- a BLAS lottery, now asserted nowhere

Found by running the affected files in the CI proxy rather than only the two
named cases.  `test_niche_d7::test_c10_shrinks_this_fixtures_hard_mask_ghost`,
added by niche C10 the day before, fails on Linux/OpenBLAS and passes on
Windows/MKL.

**It PREDATES niche C11.**  Driven directly with the flag pinned each way in
one process it fails identically in both states, so it is not the arbiter.

**Adjudication -- and a re-pin that was itself wrong.**  The fixture
DELIBERATELY degenerates `_FIT_DISC_OUTSIDE_WEIGHT_REL` to 0.0, i.e. D1's hard
NaN mask, whose documented property is an ill-conditioned normal matrix.  The
size of the ghost surviving such a solve is set by which side of the
instability that build's LAPACK lands on, and three builds give three answers:

| build | `r_old` (degree 4) | `r_new` (degree 6) | shrink |
|---|---|---|---|
| Windows / MKL | ~0.35 | ~1.8e-04 | **~1900x** |
| WSL Linux / OpenBLAS | 0.9970 | 0.5216 | **1.9x** |
| **CI Linux / OpenBLAS** | ~1.0 | **0.9998** | **1.0x -- none** |

**Four orders of magnitude, including one build that shows no shrink at all.**
This study's first re-pin moved the bar `0.1x -> 0.8x`, set below the weaker of
the TWO builds it could measure -- and CI's third build then read 0.9998, which
fails 0.8x too.  Two successive bars, each calibrated on the builds in hand and
each broken by the next one, is the definition of a quantity that must not
carry an assertion.

**The magnitude is now RECORDED and asserted nowhere.**  What the test still
asserts is the part that is build-invariant and that the era-pinned sibling
actually depends on: the degree-4 hard-mask arm GHOSTS on every build
(0.35 / 0.997 / ~1.0 against a 0.1 floor), and the degree-6 arm still returns a
finite, non-empty field on the same degenerate fixture.  Renamed
`test_the_hard_mask_arm_ghosts_on_every_build` to say what it now checks.

**The lesson, which is the third instance of it in this document:** a fixture
built to be ill-conditioned ON PURPOSE can be a liveness witness or a direction
witness, but never a magnitude one -- the magnitude is the conditioning, and
the conditioning is the build.

### 9.3 The P2 dx self-check -- LOCATED, ADJUDICATED, NOT STALE

`tests/unit/test_niche_p2_guards.py::test_self_check_dx_flags_a_non_convergent_chain`
and its sibling `::test_self_check_tolerance_is_honoured`.  The file never
names `amplitude_model='ray_density'` -- the fixture reaches that path through
chain defaults -- which is why S9's own grep-driven sweep of 21 files missed
it, and the `P_out/P_ap` band text in the CI log was a BYSTANDER warning inside
a `DID NOT WARN` failure report rather than the failure itself.

The hypothesis to test was that C9/C10 STABILISED the deliberately
beyond-Nyquist fixture (N=768 / dx=4 um / `r_in=+3 mm`), so its ~50 % drift
fell under `self_check_tol = 0.05` and the guard correctly stopped warning --
making the fixture's premise stale.

**`pytest.warns` cannot answer that**: it is binary, it reports fired / did not
fire, and it never reports the margin.  `c11_p2dx_recon.py` measures the margin
instead, running the guard's OWN comparison (`_chain_result_metrics` at
`N -> 2 round(N sqrt2 / 2)`, same physical extent) with the library's own
helpers, across every knob that moved.

**Both fixtures, both builds, all five states -- identical to every printed
digit:**

| fixture | tol | power | peak | r50 | **max** | **margin** | fires |
|---|---|---|---|---|---|---|---|
| non-convergent (N=768, `r_in`=+3 mm) | 0.05 | 52.457 % | 50.026 % | 3.584 % | **52.457 %** | **10.49x** | YES |
| dx-stable at `self_check_tol=1e-4` | 1e-4 | 0.102 % | 0.105 % | 0.001 % | **0.105 %** | **10.52x** | YES |

* Windows/MKL py3.14 / numpy 2.4.4 and Linux/OpenBLAS py3.12 / numpy 2.4.6
  agree to **five significant figures** on every entry.  This quantity is not
  BLAS-sensitive at all, which is what distinguishes it from S9.2's
  hard-mask ghost.
* `SPHERE_PARAB_CONVERSION_EXACT` (C9), `_REMAP_RESID_EIKONAL_DEGREE` (C10) and
  `REMAP_STATIONARY_PHASE_LAUNCH` (C6) are **completely inert** on both
  fixtures: all five rows are bit-for-bit the same number.  The fixture's
  chain is a single slow singlet with an on-axis beam, so it never opens the
  C1 decentre gate and never engages the residual-eikonal launch.
* the whole file passes in the CI proxy on this tree: **12 passed in 12.70 s**.

**Verdict: neither of the two dispositions the brief anticipated applies.**
The premise is not stale -- it holds with a 10.5x margin on both builds -- and
the guard has not lost detection capability; it fires in every state on every
build.  **Nothing was changed**, because strengthening a fixture that measures
10.5x its own threshold, identically on two BLAS builds, would be churn against
no evidence, and relaxing a bar that is passing would be worse.

**What this study therefore cannot explain**, and reports rather than papers
over: why CI saw `DID NOT WARN` at `5af1edf`.  Two mechanisms remain open and
neither is decidable from here (this study does not read CI logs):

1. **shard-ordering state leak.**  These are the only tests in the selection
   that assert a warning FIRES from a chain-level guard, and the drift they
   measure is deterministic -- so a `DID NOT WARN` with the physics unchanged
   points at warning-filter or module-attribute state left behind by an earlier
   test in the same shard, which is by construction shard-layout dependent and
   would explain "4 shards, 3 pythons" better than a numerical cause;
2. **misattribution in the log.**  The band text was already established to be
   a bystander; the `DID NOT WARN` line may belong to a different case.

The instrument to settle it is committed (`c11_p2dx_recon.py`) and takes ~2
minutes per build.

### 9.4 The `ray_density` energy band -- accounted for, not a failure

With S9.3 located, the `P_out/P_ap` = 0.8757 text is explained: it is a
**bystander warning printed inside the `DID NOT WARN` report**, not an
assertion that failed.  That is consistent with this study's inability to find
any test asserting on it -- all 21 unit files naming
`amplitude_model='ray_density'` were run in the CI proxy (287 + 207 = 494
tests) and none trips the `[0.8900, 1.0500]` band.

**The band itself was not touched**, and should not be: its calibration
argument is unchanged (`_RD_ENERGY_DEFICIT_BASE`'s note -- design-battery
envelope 0.9535-0.9920 at `ray_subsample=8`, converging to 0.9569-1.0000 at 1),
and moving a band edge to chase a warning that was never the failure is exactly
the class of change this campaign has paid for four times.

### 9.5 The state-leak class, and the guard that closes it permanently

A second witness arrived on `a6f7875`:
`test_niche_d4_dgrating::TestDoeChainBookkeeping::test_matches_the_manual_hand_split`
reading `max|dE|` = **0.0661** against a 1e-4 bar on CI while passing in
isolation.  0.066 is a physics-mode-sized delta, not round-off, and together
with S9.3's `DID NOT WARN` it is the signature of a same-shard test leaving a
module-level flag dirty.

**What was hunted, and what was found.**

* The failing test sits at position 291 of shard 2/3 on this box's split, with
  only **12 files** ahead of it.  Run in that exact order: **327 passed** --
  the leak does not reproduce under this build's shard composition (CI's split
  differs by Python version and durations file, so its shard 2/3 on 3.10 and
  shard 3/3 on 3.13 are different sets).
* A static sweep of every direct module-attribute write in `tests/unit`
  found **no unprotected site**: `test_niche_c5` (4 sites), `c6_fit_guard`,
  `c6_stationary_phase_launch`, `c7`, `c8`, `c9` (4 sites, all inside the
  restoring `exact_off` fixture), `c10`, `c11`, `s8` and `audit_w3` all
  save-and-restore in a `finally` or a fixture teardown.
* The same 12 files re-run with the guard in STRICT mode -- which FAILS any
  test that leaves a flag dirty -- also read **327 passed**.  There is no
  module-flag leak in that set.

**So the leaker was not found, and the class was closed anyway.**
`tests/conftest.py` now carries an autouse `_module_flag_leak_guard` that
snapshots every module-level mode flag before each test and restores it after,
making the suite order-independent whether or not a leaker exists:

* the flag set is **DISCOVERED, not enumerated** -- every module-level scalar
  (`bool`/`int`/`float`/`str`/`None`) with an upper-case name in
  `elements/_lens_traced` and `propagators/carrier`, which is **62 flags**
  against the 26 a hand-written list had.  A hand list is itself a defect
  surface: it stops covering a flag the day someone adds one, which is the
  class being closed;
* it restores **silently** by default -- the goal is an order-independent
  suite, not a red one -- and `LUMEN_TEST_FLAG_LEAK_STRICT=1` turns it into a
  failure that NAMES the leaking test and the flags it left dirty.  That is
  the instrument for the next occurrence, and it costs one environment
  variable;
* it is an autouse fixture, so it is set up first and torn down LAST -- after
  a `monkeypatch` undo -- and therefore sees genuine leakage rather than a
  pending restore.

**Two PRE-EXISTING failures were surfaced by running the shard end to end**,
and they are neither this campaign's nor the leak class's:
`test_v5_20_2_pmm_jones_2d_jax::test_pmm_jones_2d_jax_forward_matches_numpy[inplane]`
and `test_v5_20_8_rcwa_threaded_sweep::test_threaded_sweep_is_byte_identical_to_serial`
fail in shard 3/3 on Windows/MKL py3.14 -- **and fail identically when the two
files are run ALONE, and identically again with the ORIGINAL `HEAD`
`tests/conftest.py` restored**.  So they are not order-dependent, not caused by
the guard and not caused by anything here; they are a local PMM-JAX /
RCWA-threading environment issue in a subtree this study is barred from
editing, and CI does not report them.  Recorded, not touched.

**Two things this guard does NOT cover, stated so the next reader does not
assume otherwise.**  Warnings-filter leakage is already contained by pytest's own
per-item `catch_warnings`, so it is not the mechanism for S9.3 either; and the
guard restores SCALARS only -- a mutated module-level container would pass it.
The only module-level container in the two modules is
`_TRACED_KWARG_DEFAULTS_CACHE`, which is introspection, not physics.

---

## 10. What remains unresolved

1. **The arbiter's blind spot, and the (-1,0) mis-pick.**  The score is
   weighted by beam intensity, so it cannot see the fitted map's behaviour
   OUTSIDE the beam -- which is where D1's fold lives and which the Newton
   inversion evaluates over the whole launch square.  S6.4 shows this costing
   0.028 points at design 121's first order.  The fix is an ADMISSIBILITY check
   (reject a candidate whose fitted map is non-monotone over the launch
   square -- D1's own detector, and it needs the `x_out` fit which the arbiter
   currently does not build) rather than a re-weighting, because a weight that
   reaches outside the beam would re-import the aperture:beam cliff.  Not
   built, not measured.
2. *(CLOSED -- S6.6 (a).)*  Production acceptance re-measured both ways and
   identical to every printed digit including the peak, because the on-axis
   configuration never opens the C1 null gate.
3. *(CLOSED -- S8.4, S6.6 (c), (d).)*  Full niche selection 386 passed / 0
   failed with D6's independent oracle inside it; conservation and halo 4 of 4
   with every bound met; both self-checks silent.  What is NOT run is the
   WIDER `-k "traced or carrier or ..."` sweep and a full `tests/unit` pass in
   the CI proxy -- the proxy run of the five touched files is 146 passed, but
   a file outside them that reads a fit-count or a branch positionally would
   break the way S8.3's four did, and only a full sweep would find it.
4. **`carrier.py`'s `on_decentred_fit` docstring is now partly stale.**  It
   describes the decentred fit's residual in v5.32 terms and cites D7's
   0.90 urad figure; the branch it describes is now chosen differently.  No
   behaviour depends on it and `carrier.py` was deliberately left unmodified,
   but a reader will find the two documents disagreeing.
5. **The chain-A cache is pre-C7/C8/C9/C10** (S1.1).  Every arm here shares it
   so no comparison is affected, but the ABSOLUTE per-order numbers in S6.1
   inherit whatever it carries, exactly as the recorded table does.
6. **The state leak has a guard but no culprit** (S9.5).  Neither the
   12-file shard prefix nor a strict-mode run of it reproduces a leak on this
   build, and no unprotected write survives a static sweep -- so the guard
   closes the class without the leaker ever being named.  If it recurs, the
   one command that will name it is
   `LUMEN_TEST_FLAG_LEAK_STRICT=1 pytest <the shard>`.  The same applies to
   S9.3's `DID NOT WARN`: the fixture is healthy on both builds (10.5x margin,
   every knob inert) and nothing was changed there either.
7. **The transverse-map residual is measured and unused.**  It ranks the
   branches identically to the OPL on every synthetic row, so combining them
   was not needed -- but that also means the arbiter has never been tested on a
   geometry where the two disagree, and such a geometry would be the obvious
   adversarial target.
8. **The arbiter is scored on ONE quantity at ONE weighting.**  `exp(-2 r^2 /
   w^2)` about the measured chief ray is an idealisation of the input's own
   intensity; using `|E_in|^2` sampled on the launch lattice is strictly more
   faithful, costs a nearest-neighbour gather, and was not tried.  It would
   matter for a non-Gaussian input, which nothing here uses.
9. **`_DECENTRED_FIT_POLY_ORDER = 10`'s own anomaly is untouched.**
   `D121_RESIDUAL_CLOSURE` S7 item 1 records that orders 6, 8 and 12 all close
   the (-1,0) residual and 10 does not.  S3.3 here adds that the raise is
   worth 10-200x on the synthetic geometries, so the constant is doing real
   work -- but the anomaly at 10 on design 121 is still unexplained, and it is
   in the same neighbourhood as S10 item 1.
11. **Two long regression runs were still in flight** when this document was
    written: the full 3670-test shard 2/3 on Windows, and a strict-mode
    (`LUMEN_TEST_FLAG_LEAK_STRICT=1`) sweep of every flag-writing test file on
    Linux/OpenBLAS, which stood at 84 % with zero failures.  Neither is
    load-bearing for a conclusion here -- the causally relevant prefix of
    shard 2 is green both ways (S8.4) and the leak class is closed by
    construction rather than by that sweep (S9.5) -- but neither is finished
    either.

10. **Nothing here was run on GPU, on `inversion_method='fit'`, or through
    `apply_real_lens_traced_multi`'s prepared-screen reuse.**  The arbiter adds
    an input-dependent decision to the fit site; the prepared-screen path's
    input-independence was already broken by the P2 beam-relative radius, but
    that is an argument from an existing hole, not a measurement.

---

## 11. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Every runner forces
`LUMEN_PIN=0` and prints the library version, path and file hashes.

```bash
# S3.1 -- the branches, re-measured on the C10 tree
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python rc_gate_121.py

# S3.2 / S3.3 / S4.2 -- the synthetic sweeps, the decomposition, the arbiter
GEOMS=f6,f3,f6w python c11_synth_sweep.py
GEOMS=f3,f6w ARMS=conc,off,off6,conc10 python c11_synth_sweep.py

# S4.3 -- the arbiter on design 121's own element calls
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python c11_discrim_121.py

# S5 -- the null contract and the step at the boundary
GEOMS=f3,f6w python c11_boundary_step.py

# S6 -- THE RESULT: the per-order table, both arms pinned
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python c11_gate_arms_121.py

# S9.3 -- the P2 dx self-check MARGIN (run on BOTH builds; ~2 min each)
LUMEN_PIN=0 python c11_p2dx_recon.py                  # the non-convergent one
LUMEN_PIN=0 WHICH=stable python c11_p2dx_recon.py     # the sibling

# S9.5 -- the state-leak class.  This is the command that NAMES a leaker:
#   from the repo root, on whatever shard composition went red
LUMEN_TEST_FLAG_LEAK_STRICT=1 python -m pytest tests/unit     -m "not integration and not slow" --splits 3 --group <N> -q
#   without the variable the guard restores silently and the suite is simply
#   order-independent; with it, the leaking TEST fails and names the flags

# S9.1 -- the D3 reconciliation, in the CI proxy (Linux/OpenBLAS)
wsl -e bash -lc "cd <repo> && ~/lumvenv/bin/python -m pytest \
    tests/unit/test_niche_d3_guards.py -q -k guarded_input"
wsl -e bash -lc "cd <repo>/validation/repro_traced_carrier_121 && \
    LUMEN_PIN=0 WHAT=d3 ~/lumvenv/bin/python c11_ci_recon.py"

# S10 items 2, 3 -- NOT RUN, and the commands that would close them
DECENTRED_FIT_ARBITER is a module attribute; pin it both ways through
rc_with_gate.py the way RESID_DEG is pinned:
    python rc_with_gate.py focus_scan_121.py
    ORDERS='0,0 -1,0 -2,0 -4,-2' CONFIGS='ship' NULL=0 \
        python rc_with_gate.py energy_stage_audit_121.py
    python -m pytest tests/unit/test_niche_{c1,c3,c5,c6,c7,c8,c9,c10,c11,d1,d2,d3,d6,d7,s8}_*.py -q
```

### Files added by this study

```
validation/repro_traced_carrier_121/c11_synth_sweep.py     the synthetic sweeps + the discriminator
validation/repro_traced_carrier_121/c11_discrim_121.py     the arbiter on 121's element calls
validation/repro_traced_carrier_121/c11_boundary_step.py   the null contract + the boundary step
validation/repro_traced_carrier_121/c11_gate_arms_121.py   the per-order table, both arms
validation/repro_traced_carrier_121/c11_ci_recon.py        the D3 / C9-C10 CI adjudication
validation/repro_traced_carrier_121/c11_p2dx_recon.py      the P2 dx-drift MARGIN, both builds
tests/conftest.py                                          the module-flag leak guard (S9.5)
tests/unit/test_niche_c11_decentred_fit_arbiter.py         20 tests
docs/audits/C11_PHYSICAL_DECENTRE_GATE_2026_08_03.md       this document
```
