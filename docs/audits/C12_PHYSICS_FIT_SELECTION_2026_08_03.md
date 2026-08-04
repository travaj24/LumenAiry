# The decentred beam's ray-fit branch, DERIVED: the lens's own spectral tail

Niche C12, 2026-08-03.  Branch `feat/d121-final-closure`, on top of the C11
tree (`_lens_traced.py 8b13259592233b52`).

Predecessors: `C11_PHYSICAL_DECENTRE_GATE_2026_08_03` (the whole thing),
`ARCH_TRACED_ENCAPSULATION_2026_08_03` S3.2 (UNIT B),
`C6_FIT_GUARD_DECISION_2026_07_31`, `D121_RESIDUAL_CLOSURE_2026_08_02`.

---

## 0. Headline

Niche C11 replaced a constant gate with an ARBITER: build both candidate ray
fits, score each against the traced OPL over the beam, take the smaller.  It
works, and its own note says what it is -- *a measurement, not a model*.  It
cannot say why the crossover sits where it does, and it says nothing about any
decentre other than the one in front of it.

**It is derivable, and the derivation is short.**  The traced OPL is a fixed
function of the ENTRANCE position: moving the beam moves neither it nor the
launch grid.  A total-degree-`m` least-squares fit reproduces the degree-`<= m`
part *exactly*, so each candidate's residual is **identically** the residual of
fitting its own Chebyshev tail `W_>m`.  That tail is decentre-free, so the
entire `u`-dependence is geometric: the concentric disc is sized from the
ORIGIN-referenced second moment and therefore inflates by
`rho = sqrt(1 + 2 u^2)`, while the off-centre disc and the beam translate
together and its residual is flat.  Each shell scales as `s^n`, so the
concentric residual runs as `rho^m_eff` with `m_eff` the tail's own spectral
first moment, and the crossover follows in closed form:

>   `rho* = rho(u) (E_off / E_conc)^(1/m_eff)` ,  `u* = sqrt((rho*^2 - 1)/2)`

No fitted constant appears anywhere: `S_n` is the lens's measured spectrum,
`sigma` and `rho` are geometry, and `p`, `P`, `eps` are library constants.

**It predicts all three designs, and it beats the arbiter against the only
oracle that can adjudicate.**  Bisected on the EXIT FIELD against the
fit-domain-free `newton_fit='spline'` reference, in one process
(`c12_oracle_crossover.py`):

| geometry | ORACLE `u*` | ARBITER `u*` (C11) | **PREDICTOR `u*` (C12)** | `m_eff` |
|---|---|---|---|---|
| f/6 N-BK7 singlet, `w` = 1.0 mm | 0.0000 | 0.0000 | **0.0000** | 8.000 |
| f/3 N-BK7 singlet, `w` = 1.0 mm | **0.5715** | 0.5453 (**-4.6 %**) | **0.5717 (+0.03 %)** | 8.000 |
| f/6 N-BK7 singlet, `w` = 1.4 mm | 0.0000 | 0.0000 | **0.0000** | 8.000 |

**And that corrects niche C11.**  Its "0.55 w" was the ARBITER's crossover, not
the oracle's; C11 sampled the oracle at 0.50 and 0.60 and could not separate
them.  Bisected to 3.5e-4 w the oracle sits at **0.5715**.  The two selectors
therefore disagree on `0.5453 < u < 0.5717`, and the oracle backs the PREDICTOR
across all but the last 2e-4 of that band -- four measured rows in S3.3, at
separations nine orders above the reference's own floor.

**On design 121 the spectral half does not apply, and the predictor says so.**
Its groups' launch square is 47 mm against a 6.3 mm beam (7.5:1) with ~0.5 % of
it carrying no ray at all, and the launch-box Chebyshev expansion does not
converge there: the shells sit flat at ~1e-3 out to degree 20 and the order-14
box fit leaves 1e-5 m over the beam, four decades ABOVE the candidate residuals
it would have to rank.  The predictor tests exactly that and falls back to the
measured pair, so on all **26** of design 121's arbitrated element calls it
reproduces the arbiter's 26 decisions with **zero** disagreement warnings -- and
the per-order table is C11's, bit for bit.

**The two things the brief asked for that came back NEGATIVE, both measured.**

* **The scorer's blind spot is not the (-1,0) regression's cause.**  Flooring
  the score weight over the full launch square flips **every one of the 26
  calls** to the off-centre branch at any floor `>= 1e-8` -- the concentric
  candidate is a hard NaN mask, so beyond its disc the fit is pure
  extrapolation and any floor at all lets that dominate.  The selector then IS
  the v5.32 gate and every C11 gain is lost.  Restricting to the illuminated
  support moves no verdict (26/26 unchanged at floor 1e-4); amplitude
  weighting flips four calls and produces a pattern that is NON-MONOTONE in
  `|c|/w`.  `_DECENTRED_FIT_SCORE_FLOOR` ships **0.0**, and its note records
  the sweep.
* **No per-call selector can meet a per-order "improve or hold" bar on design
  121.**  Proven, not asserted (S5): group 2 is preferred CONCENTRIC by
  **4.77x** at (-1,0) (`u` = 0.062) and by **4.24x** at (-2,0) (`u` = 0.125),
  and the chain wants OFF-CENTRE at the first and CONCENTRIC at the second.
  No rule monotone in the margin, in `u` or in `u/u*` can produce that pair.
  The chain-level EE3 is not separable across the six groups: measured at
  (-1,0), the mixed patterns `ccoo` (0.069) and `cooo` (0.067) are **worse**
  than either `ccco` (0.055) or `oooo` (0.029).

**And one attribution in C11 is reversed.**  Its S6.4 concluded that "85 % of
'the concentric branch wins at 0.24 w' is the niche-C6 fit domain, not the ray
fit".  That rests on the premise that the arbiter picks concentric at all six
of (-1,0)'s groups.  Instrumented in process, **it does not**: only four groups
are arbitrated there at all and one of them picks off-centre.  Forcing the ray
fit alone, with the C6 domain held exactly as shipped, moves (-1,0) from
**+0.0552 to -0.0985** -- the whole 0.154 points.  **The ray-fit branch is the
driver and the C6 domain is inert at that order** (S6).

**`DECENTRED_FIT_PREDICTOR` ships `False`.**  The fail-before is the v5.32
gate, bit for bit, on the design (S7.1).

---

## 1. Provenance, and what every delta is measured against

### 1.1 Which library

| | |
|---|---|
| repo | `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy` |
| branch | `feat/d121-final-closure` |
| `lumenairy` | 5.32.0 |
| `_lens_traced.py` BEFORE | `8b13259592233b52` (the C11 hash) |
| `_lens_traced.py` AFTER | `d0d8da8d494e95f7` |
| ... intermediate | `dafdf9b27a6bb214` -- see the note below |
| `carrier.py` | `5a1b0d1021969df1` **unmodified**, both states |
| CI proxy | WSL Ubuntu, `~/lumvenv`, Python 3.12.3, numpy 2.4.6, **OpenBLAS** |
| dev box | Windows 11, Python 3.14, MKL |

Every runner forces `LUMEN_PIN=0`.  `CHANGELOG.md` and `lumenairy/elements/pmm/**`
are untouched.

**The intermediate hash, stated rather than glossed.**  The design-121 arms
table (S7.1) was taken at `dafdf9b27a6bb214`; the file then gained two things
that are not on any numeric path and cannot be -- an `np.isfinite(_sp_resid)`
conjunct in the resolution guard (true whenever the box fit was built at all,
which is the only state in which the guard is reached), and two extra FIELDS in
the disagreement warning's text.  It is checked rather than argued: S7.2
re-runs the hardest order at the final hash and compares intensity shas.

**Environment note.**  An unrelated 85 GiB `fan_multi_121.py` belonging to
another session was resident for the whole of this study's measurement window.
It moves wall-clock times (the arms table's per-order 269-295 s against a
~150 s baseline) and it is why some RAM-guarded tests skipped on the first
Windows pass; it cannot move a number.

### 1.2 Floors

| instrument | null | reading |
|---|---|---|
| `c12_gate_arms_121.py`, `default` arm | vs C11's recorded `default`/`v532` shas, a different runner in a different process on the pre-C12 tree | **6 of 6 bit-identical** (S7.1) |
| `c12_predict_synth.py`, traced-map invariance | the same lens captured at `u` = 0.2 and `u` = 1.0 | **`array_equal` True** on all three geometries -- the decentre-freedom the whole derivation rests on is a measurement, not an argument |
| `c11_synth_sweep.py` spline reference | its own residual gate dependence | **2.6e-18 waves** (C11 S1.2, unchanged) |
| sampling adequacy | power-weighted p99.9 wrapped nearest-neighbour step of every scored residual | **0.000 rad** against `pi` on every oracle row taken here |
| campaign estimator noise | `D121_RESIDUAL_CLOSURE` S1.3 | **0.003-0.015 points** |

---

## 2. The derivation

### 2.1 The exact identity that makes it a model rather than a fit

Let `W` be the traced OPL on the coarse launch lattice and `Pi_{m,D}` the
weighted least-squares projection onto total degree `<= m` over a restriction
`D` (a hard NaN mask, or D1's weights).  For any polynomial `q` of degree
`<= m`, `Pi_{m,D} q = q`.  Split `W` in the launch-box Chebyshev basis at
degree `m`:

```
W = W_<=m + W_>m
(I - Pi_{m,D}) W = (I - Pi_{m,D}) W_<=m + (I - Pi_{m,D}) W_>m
                 = 0                     + (I - Pi_{m,D}) W_>m
```

**A candidate's residual IS the residual of fitting its own spectral tail.**
Exactly -- not to leading order.  That is the whole content of the phrase "the
aberration content beyond the polynomial budget is the lens's own Chebyshev
spectral tail", and it is why the tail is the right object to reason about.

`_decentred_fit_spectrum` builds `W_>m` from ONE unweighted order-`q` box fit
by zeroing every coefficient of total degree `<= m`.  The identity is pinned as
a test on a real traced map (`test_the_spectral_tail_carries_the_whole_candidate_residual`).

### 2.2 The tail does not move when the beam does

`W` is a function of the ENTRANCE position on a launch grid whose radius comes
from the aperture (`launch_radius = 0.75 * aperture`), not from the beam.
Neither depends on `beam_centre`.  **Measured, not assumed**: the same lens
captured at `|c|/w` = 0.2 and 1.0 returns `array_equal` OPL grids on all three
synthetic geometries (S1.2).

So the two candidates' `u`-dependence is purely geometric.

* **OFF-CENTRE.**  Disc radius `frbf * w` about the measured chief ray; the
  score weight `exp(-2 |r - c|^2 / w^2)` is about the same point.  Disc, weight
  and beam translate together -- the whole configuration is covariant under the
  shift -- so `E_off` is FLAT in `u`.  (Measured: 1.06e-12 -> 7.78e-13 m across
  `u` = 0.02 .. 1.0 on the f/3, i.e. a 27 % drift over a 50x change in decentre,
  against `E_conc` growing 62x on the same rows.)
* **CONCENTRIC.**  Disc radius `frbf * sqrt(2 c^2 + w^2)` about the GRID
  ORIGIN, i.e. `frbf * w * rho` with

  >   `rho(u) = sqrt(1 + 2 u^2)`

  -- the D1 defect, restated: an origin-referenced second moment of a
  decentred beam reads `sqrt(2 c^2 + w^2)`, so the same total-degree budget is
  spread over a disc `rho` times bigger.

### 2.3 The inflation law and its exponent

The box-normalised Chebyshev basis is degree-graded, so restricting a
degree-`n` shell to a concentric sub-domain of relative radius `s` scales its
contribution by `s^n`.  With `sigma = frbf * w / R_box` the beam disc in box
units, the concentric candidate's tail energy at decentre `u` is

```
T(p, sigma rho)^2 = sum_{n>p} (S_n sigma^n)^2 rho^{2n}
```

so, differentiating at `rho = 1`,

>   `d log T / d log rho  =  m_eff  =  sum_{n>p} n (S_n sigma^n)^2 /
>                                      sum_{n>p} (S_n sigma^n)^2`

-- the tail's **spectral first moment**, computed by
`_decentred_fit_spectral_moment` and pinned by exact arithmetic in the test
suite.  It is the degree of the first shell the fit cannot reach, softened by
whatever the shells above it carry.  Measured it is **8.000** on all three
synthetic geometries (the first EVEN shell above `newton_poly_order = 6`; a
centred spherical singlet's OPL is even in both launch coordinates, so odd
shells are at round-off -- also pinned as a test), and **7.15 - 8.05** on all
26 of design 121's arbitrated element calls.  It is set mainly by the BUDGET
rather than by the lens: the f/3 and f/6 spectra differ by two decades in decay
rate and give the same 8.000, and design 121 -- whose off-axis groups DO carry
odd shells -- moves it by at most 11 %.  That stability is why the inflation
law is transferable at all.

Hence `E_conc(u) = E_conc(0) rho(u)^m_eff`, and setting that equal to the flat
`E_off`:

>   `rho* = rho(u) (E_off / E_conc(u))^(1/m_eff)`
>   `u*   = sqrt((rho*^2 - 1) / 2)`

`_decentred_fit_crossover` is that, with the degenerate cases spelled out
(`0.0` when the off-centre candidate already wins at zero decentre, `inf` when
the concentric one cannot lose, `nan` when there is no prediction to make).

### 2.4 What the closed form buys, stated honestly

For a positive exponent, `u <= u*` and `E_conc <= E_off` are the SAME test --
pinned as a 400-sample property test.  So the closed form does not by itself
change a verdict; what it buys is:

* a NUMBER, `u*`, per lens, that the arbiter never produces -- which is what
  makes the selection reportable and transferable to a decentre nobody ran;
* the ability to evaluate the decision from the SPECTRAL TAIL alone, without
  building the concentric candidate at all -- and that arm is genuinely
  independent of the measured pair, which is what makes the runtime check in
  S4.3 substantive rather than an identity.

The predictor uses the spectral arm where it is RESOLVED and the measured pair
otherwise.  Both states are named in the diagnostics; neither is silent.

---

## 3. The three-design prediction table

### 3.1 The model reproduces the measured residuals with no constant

`c12_predict_synth.py`.  ONE order-16 box fit per geometry; the two candidates'
residuals are then re-derived from the surrogate tails and compared with the
library's own measured scores at the same decentre.  `K = measured / modelled`:

| geometry | `\|c\|/w` | `E_c` model | `E_c` meas | `K_c` | `E_o` model | `E_o` meas | `K_o` |
|---|---|---|---|---|---|---|---|
| f/3 | 0.100 | 2.650e-13 | 2.231e-13 | 0.842 | 1.175e-12 | 1.048e-12 | 0.892 |
| f/3 | 0.400 | 6.636e-13 | 4.034e-13 | 0.608 | 1.122e-12 | 1.001e-12 | 0.892 |
| f/3 | 0.500 | 1.020e-12 | 7.494e-13 | 0.735 | 1.126e-12 | 1.008e-12 | 0.895 |
| f/3 | 0.600 | 1.686e-12 | 1.390e-12 | 0.824 | 1.077e-12 | 9.690e-13 | 0.900 |
| f/3 | 1.000 | 1.462e-11 | 1.314e-11 | 0.898 | 8.355e-13 | 7.779e-13 | 0.931 |
| f/6w | 0.500 | 1.069e-13 | 1.068e-13 | 0.998 | 3.710e-16 | 1.167e-15 | 3.14 |
| f/6w | 1.000 | 1.511e-12 | 1.511e-12 | 1.000 | 4.962e-16 | 1.491e-15 | 3.01 |
| f/6 | 1.000 | 1.040e-13 | 1.046e-13 | 1.005 | 1.534e-16 | 2.179e-15 | 14.2 |

Two readings, and the second is the honest limit.

* **Where a residual is truncation- or leak-limited the model is right with no
  fitted constant**: `K = 0.61-1.00` for the concentric candidate on both
  geometries where it matters, and `K = 0.89-0.93` for the off-centre one on
  the f/3.
* **Where a candidate sits on the fit's own NUMERICAL FLOOR the model
  under-predicts** -- the f/6 pair are both at 1e-15 .. 1e-13 m (1e-9 waves),
  and no truncation model reaches below a floor it does not contain.  It costs
  nothing there because the ordering is unambiguous by 100x, but it is the
  reason the predictor tests resolution rather than trusting the model.

The shells themselves, for the record (order-16 box fit, even degrees; odd ones
are at 1e-18):

```
f/3   9.05e-04 6.39e-04 4.10e-06 3.24e-06 5.25e-07 7.56e-08 1.09e-08 1.58e-09 2.17e-10
f/6   4.62e-04 3.28e-04 7.39e-07 3.81e-08 1.49e-09 4.53e-11 1.32e-12 3.87e-14 1.13e-15
```

-- the f/3's tail decays 2.9x per two degrees at the top, the f/6's 34x.  That
one number is the whole difference between a crossover at 0.57 w and one at 0.

### 3.2 The crossovers, bisected three ways in one process

`c12_oracle_crossover.py`, 12 bisection steps (resolution 3.5e-4 in `u`):

| geometry | ORACLE `u*` | ARBITER `u*` | **PREDICTOR `u*`** | `m_eff` | spectrum |
|---|---|---|---|---|---|
| f/6, `w` = 1.0 mm | 0.0000 | 0.0000 | **0.0000** | 8.000 | unresolved (floor-limited) |
| f/3, `w` = 1.0 mm | 0.5715 | 0.5453 | **0.5717** | 8.000 | **resolved** |
| f/6, `w` = 1.4 mm | 0.0000 | 0.0000 | **0.0000** | 8.000 | unresolved (floor-limited) |

The ORACLE column is bisected on the EXIT FIELD -- equivalent rms wavefront
error against `newton_fit='spline'`, piston and tilt projected out without any
unwrapping, the C11 instrument verbatim -- so it shares no code with either
selector.  The bisection starts at `u` = 1e-4 for the oracle and at `u` = 0.06
for the two selectors, because below niche C1's null gate (0.05 w) neither
selector runs at all and a bisection started there would report a choice that
was never made.

**The predictor lands on the oracle to 0.03 % on the one geometry where the
crossover is not at zero, and reproduces both zeros exactly.**

### 3.3 ... and the arbiter is wrong on a 0.026-w band

The oracle sweep around the crossover, on the f/3 fixture, scored the C11 way:

| `\|c\|/w` | sig CONCENTRIC | sig OFF-CENTRE | oracle | C11 arbiter | **C12 predictor** |
|---|---|---|---|---|---|
| 0.545 | **5.2338e-07** | 6.3484e-07 | conc | conc | **conc** |
| 0.550 | **5.7501e-07** | 6.3490e-07 | conc | *off* | **conc** |
| 0.560 | **5.7369e-07** | 6.1523e-07 | conc | *off* | **conc** |
| 0.565 | **5.7308e-07** | 6.1524e-07 | conc | *off* | **conc** |
| 0.570 | **5.7249e-07** | 6.1519e-07 | conc | *off* | **conc** |
| 0.572 | 6.3579e-07 | **6.1516e-07** | off | off | off |
| 0.578 | 6.3467e-07 | **6.1507e-07** | off | off | off |

Four rows -- the whole 0.5453 .. 0.5717 band -- on which C11's arbiter takes the
branch the exit field does not want, and the predictor takes the branch it
does.  The differential floor there (the spline reference re-run with the
branch forced the other way) is 2.6e-18 waves, so the 6.0e-8 .. 4.3e-8
separations are nine orders above it.  `sig` is the equivalent rms wavefront
error in waves; smaller is better.

This does not contradict C11's "42 of 42": that agreement was scored at 14
decentres per geometry, of which the two straddling the f/3 crossover are 0.50
and 0.60.  Both selectors are right at both.  **It takes a bisection to see the
difference, and this is the first time one was run.**

### 3.4 Design 121: the spectral arm does not apply, and the predictor knows it

`c12_arb_trace_121.py` captures the arbiter's OWN arguments in process, with
the flag ON.  The launch lattice there is 231x231 .. 265x265 over a 47 mm
launch square, against a 6.3 mm beam, with ~0.5 % of the lattice carrying no
finite OPL.  The box expansion does not converge:

| order-`q` box fit, (-1,0) group 2 | shells (even degrees 0..q) | beam-weighted residual |
|---|---|---|
| `q` = 10 | 2.1e-04 8.5e-04 1.8e-03 1.2e-03 6.5e-04 2.3e-04 | 4.018e-05 m |
| `q` = 14 | 3.0e-05 1.5e-03 2.8e-03 2.3e-03 1.6e-03 9.3e-04 4.4e-04 1.4e-04 | 1.916e-05 m |
| `q` = 20 | 6.7e-04 3.2e-03 5.5e-03 5.3e-03 4.4e-03 3.3e-03 2.2e-03 1.3e-03 6.8e-04 2.9e-04 8.3e-05 | 8.564e-06 m |

The shells RISE with degree and then decay slowly, and -- decisively -- they
are not stable in `q`: the degree-4 shell reads 1.8e-3 / 2.8e-3 / 5.5e-3 at
`q` = 10 / 14 / 20.  A coefficient that depends on where you truncate is not a
coefficient.  The candidate residuals it would have to rank are 2e-9 .. 1e-7 m.

**The resolution test.**  The model's error is exactly what a degree-`oc` fit
over the concentric disc CANNOT absorb of `W - W^(q)`, and by the S2.1 identity
that is `(I - Pi)(W - tails[oc])` -- one more score of the same kind, no new
machinery.  The model is used only while that gap is below the residuals it
would rank.  Measured:

| | gap | modelled pair | resolved |
|---|---|---|---|
| f/3, `u` = 0.2 | 1.11e-13 | 1.48e-13 / 6.90e-13 | **yes** |
| f/3, `u` = 0.5 | 2.75e-13 | 4.92e-13 / 6.76e-13 | **yes** |
| f/6w, `u` = 1.0 | 1.95e-16 | 1.51e-12 / 4.88e-16 | **yes** |
| f/6, `u` = 0.5 | 8.53e-14 | 7.23e-15 / 2.05e-16 | no |
| 121 (-1,0) grp 2 | 1.77e-06 | 1.77e-06 / 4.29e-07 | no |
| 121 (-4,0) grp 4 | 2.66e-06 | 2.74e-06 / 2.76e-07 | no |

**All 26 of design 121's arbitrated calls read UNRESOLVED**, so the predictor
falls back to the measured pair there.  Its per-call crossover, from the closed
form, then reads:

| order | arbitrated groups (`\|c\|/w`) | `u*` per call | picks |
|---|---|---|---|
| (-1,0) | 2,3,4,5 (0.062 .. 0.241) | 0.493 / 0.316 / 0.258 / 0.201 | c c c **o** |
| (-2,0) | 1..5 (0.099 .. 0.481) | 0.319 / 0.488 / 0.430 / 0 / 0 | c c c **o o** |
| (-3,0) | 1..5 (0.148 .. 0.723) | 0.350 / 0.489 / 0 / 0 / 0.269 | c c **o o o** |
| (-4,0) | 0..5 (0.051 .. 0.965) | 0.282 / 0.407 / 0.516 / 0 / 0 / 0.431 | c c c **o o o** |
| (-4,-2) | 0..5 (0.057 .. 1.079) | 0.282 / 0.436 / 0.539 / 0.194 / 0 / 0.519 | c c c **o o o** |

**26 of 26 identical to the arbiter's verdicts, zero disagreement warnings.**

Two things follow that are corrections rather than results.

* **C11's own per-group table (its S4.3) is wrong**, and so is the "0.46-0.69 w"
  band it reports.  That table was built SCRIPT-side from two separately-FORCED
  traces, which also moves niche C6's residual-eikonal domain; the library
  builds both candidates from ONE trace and additionally intersects the
  concentric disc with the R7 carrier disc (`_fit_r_max_conc`).  Instrumented
  in process, the shipped arbiter's per-call crossovers land at **0.20-0.54**,
  and at (-1,0) only FOUR groups are arbitrated at all -- groups 0 and 1 sit
  below the C1 null gate.
* **The band is per (group, ORDER), not per group.**  Each diffraction order
  reaches a group at a different incidence, so the traced map differs; the
  "same group at a larger decentre" comparison the band assumes does not exist
  on this design.

---

## 4. What shipped

### 4.1 The diff

| file | before | after | what |
|---|---|---|---|
| `lumenairy/elements/_lens_traced.py` | `8b13259592233b52` | `d0d8da8d494e95f7` | two constants (both inert), four helpers, one flag, the predictor block |
| `lumenairy/propagators/carrier.py` | `5a1b0d1021969df1` | `5a1b0d1021969df1` | **unmodified** |
| `tests/unit/test_niche_c12_physics_fit_selection.py` | | | new |
| `CHANGELOG.md`, `lumenairy/elements/pmm/**` | | | **untouched** |

No signature moved, no public entry point added, no existing default flipped.
`_DECENTRE_GATE_PIXELS`, `_DECENTRE_GATE_W_FRAC`, `_DECENTRED_FIT_POLY_ORDER`,
`_FIT_DISC_OUTSIDE_WEIGHT_REL` and `DECENTRED_FIT_ARBITER` are all unchanged.

Added:

* `_DECENTRED_FIT_SCORE_FLOOR = 0.0` and `_decentred_fit_score_weight(...)` --
  the score weight gets one home and the floor gets a documented, inert knob
  (S5.1);
* `_DECENTRED_FIT_SPECTRUM_ORDER = 14` and `_decentred_fit_spectrum(...)` --
  the launch-box shell spectrum, the tail surrogates, and the box fit's own
  residual under the score weight;
* `_decentred_fit_spectral_moment(S, m, sigma)` -- `m_eff`;
* `_decentred_fit_crossover(u, e_conc, e_off, m_eff)` -- `u*`;
* `DECENTRED_FIT_PREDICTOR = False` -- the opt-in.

### 4.2 Precedence, and the fail-before

```
DECENTRED_FIT_PREDICTOR  True  ->  u <= u*        (niche C12; arbiter checks)
DECENTRED_FIT_ARBITER    True  ->  E_c <= E_o     (niche C11)
neither                        ->  |c| > max(0.5 dx, 0.05 w)   (niche C1/v5.32)
```

With both `False` the whole C12 layer is a path NOT TAKEN, not a computation
discarded: the gate site tests the flags before measuring the origin-referenced
radius, and the fit site tests them before building any candidate.  A test
asserts `_decentred_fit_crossover` is never CALLED on a call that reaches the
branch decision by every other route.

`_decentred_fit_score_weight(..., floor=0.0)` returns the bare Gaussian by
`array_equal` -- `np.maximum` is skipped, not applied with a zero -- so the C11
arbiter path is bitwise what it was.

### 4.3 The runtime check

Niche C11's raw comparison is computed on every predictor call and is the
CHECK.  When the spectrum is RESOLVED the two are independent -- the predictor
reads the modelled pair, the arbiter the measured one -- and a disagreement is
real; on the f/3 fixture it is exactly the 0.026-w band of S3.3.  When the
spectrum is unresolved the predictor uses the measured pair and the two
coincide by construction; **that is stated in the warning text and in the flag
note rather than left for a reader to discover.**

A disagreement raises a `RuntimeWarning` naming both score pairs, `u`, `u*`,
`m_eff`, the resolution state and the box fit's own residual, and says which
one was applied.  Verbatim, from the f/3 fixture at `u` = 0.560:

```text
apply_real_lens_traced: the niche-C12 ray-fit PREDICTOR and the niche-C11
ARBITER disagree on this call. The predictor selects the CONCENTRIC branch
from |c|/w = 0.5600 against a crossover u* = 0.5709 (spectral exponent m_eff
= 8.000, spectrum resolved at order 14, box-fit residual 1.616e-10 m,
modelled OPL residuals 6.131e-13 m concentric / 6.513e-13 m off-centre); the
arbiter's own measured residuals are 1.022e-12 m concentric / 9.688e-13 m
off-centre and select the OFF-CENTRE one. The PREDICTOR's choice is applied.
See DECENTRED_FIT_PREDICTOR; set it False to fall back to the arbiter, or
also DECENTRED_FIT_ARBITER False for the v5.32 gate.
```

That row is line 3 of the S3.3 table: the oracle says CONCENTRIC there, so the
warning is firing on a call where the predictor is right and the arbiter is
wrong.  The box-fit residual it reports (1.6e-10 m) is three decades ABOVE the
candidate residuals and yet the spectrum is still `resolved` -- because what
the resolution test measures is not the box fit's error but the part of it a
degree-6 fit over the disc cannot absorb (1.1e-13 .. 2.8e-13 m here).  Both
numbers are printed so the distinction is visible at the point of use.

### 4.4 Cost

Per element call ABOVE the C1 null gate, with the predictor engaged: one
order-`q` `_Cheb2DEvaluator` box fit, two surrogate fits and one resolution
probe on top of niche C11's two.  Counted exactly on the f/3 fixture
(`ray_subsample = 8`, 63x63 lattice) -- **9 evaluator builds where the pure
gate makes 3 and the C11 arbiter makes 5**, at orders
`10, 6, 14, 6, 10, 6, 6, 6, 6`.  Wall clock on the same fixture, after numba
warm-up, three alternating runs each:

```
gate  0.365 / 0.415 / 0.419 s
C11   0.437 / 0.360 / 0.410 s
C12   0.364 / 0.436 / 0.365 s
```

-- i.e. inside the run-to-run spread at this size, where the Newton inversion
dominates.  They are SEQUENTIAL, so peak memory is
set by the largest single design matrix; at `q` = 14 that is 120 basis terms
against the off-centre candidate's 66, i.e. 1.8x the largest matrix C11 builds.
On a large configuration (N = 8192, `ray_subsample` = 8) each order-10 design
matrix is ~0.5 GiB, so the order-14 one is ~0.9 GiB.  **Stated; it is why the
flag is opt-in on cost grounds as well as on physics grounds, and why
`_DECENTRED_FIT_SPECTRUM_ORDER = 0` disables the spectral half outright.**

---

## 5. The scorer, and why it is not the fix

### 5.1 The floor sweep

`c12_arb_trace_121.py` captures the shipped arbiter's own `(xs, opl, weight,
disc, weights, order)` for all 26 arbitrated calls; `c12_scorer_sweep.py`
re-scores those exact inputs under a family of weights.  Concentric/off-centre
score ratio, `< 1` picks concentric:

| order | pair | `\|c\|/w` | c11 | fl 1e-8 | fl 1e-6 | su 1e-4 | amp |
|---|---|---|---|---|---|---|---|
| (-1,0) | 0 | 0.062 | **0.210** | 2.563 | 2.587 | 0.141 | 0.439 |
| (-1,0) | 1 | 0.153 | **0.582** | 12.974 | 14.250 | 0.069 | 0.937 |
| (-1,0) | 2 | 0.199 | **0.827** | 4.019 | 4.030 | 0.060 | 0.882 |
| (-1,0) | 3 | 0.241 | 1.126 | 2.888 | 3.091 | 2.254 | 1.127 |
| (-2,0) | 0 | 0.099 | **0.520** | 9.046 | 10.902 | 0.000 | 3.106 |
| (-2,0) | 1 | 0.125 | **0.236** | 2.608 | 2.637 | 0.159 | 0.479 |
| (-2,0) | 2 | 0.306 | **0.578** | 12.576 | 13.678 | 0.121 | 0.923 |
| (-4,0) | 0 | 0.051 | **0.568** | 8.195 | 11.611 | 0.001 | 3.046 |
| (-4,0) | 1 | 0.197 | **0.443** | 8.124 | 9.936 | 0.000 | 2.684 |
| (-4,0) | 2 | 0.249 | **0.288** | 2.810 | 2.881 | 0.217 | 0.539 |
| (-4,0) | 4 | 0.796 | 28.653 | 3.946 | 3.160 | 80.357 | 3.804 |

* **`fl` (a floor over the whole launch square) flips every concentric pick to
  off-centre, at every floor from 1e-8 up.**  The mechanism is not subtle: the
  concentric candidate is a hard NaN MASK, so beyond its disc the fit is pure
  extrapolation over a square 7.5x the beam, and its residual there is 4-6
  decades above its in-beam one.  Any floor at all lets that dominate.  The
  selector becomes "always off-centre" -- which on this design IS the v5.32
  gate, so every niche-C11 gain is lost and nothing is bought.
* **`su` (the same floor, but ZERO outside the illuminated support) moves no
  verdict**: 26 of 26 picks unchanged at `F` = 1e-4.  It makes the concentric
  candidate look BETTER, because it removes exactly the skirt the extension was
  supposed to add.
* **`amp` (amplitude rather than intensity weighting) flips four calls and is
  non-monotone**: at (-4,0) it prefers off-centre at `u` = 0.051 and 0.197 and
  concentric at `u` = 0.249, on the same chain.  A selector with that shape is
  not a selector.

`_DECENTRED_FIT_SCORE_FLOOR` therefore ships **0.0**, with the sweep in its
note.  It exists so the next reader does not re-run it.

### 5.2 The deeper reason: the chain is not separable across groups

`c12_group_arms_121.py` forces the branch PER ARBITRATED CALL -- at the
decision, by returning the two scores as 0/1 in the wanted direction, so both
candidates are still built exactly as shipped and everything upstream of the
fit site (including niche C6's residual-eikonal domain, which is committed
before the trace) is untouched.  EE3 area-exact residual against the
exact-ray CARRY=1 ceiling, in points:

**(-1,0)**, four arbitrated calls (groups 2..5):

| pattern | `cccc` | `ccco` (**C11**) | `ccoo` | `cooo` | `oooo` (**v5.32**) |
|---|---|---|---|---|---|
| residual | **-0.0985** | **+0.0552** | +0.0689 | +0.0673 | **+0.0290** |

**(-2,0)**, five arbitrated calls (groups 1..5):

| pattern | `ccccc` | `cccco` | `cccoo` (**C11**) | `ccooo` | `coooo` | `ooooo` (**v5.32**) |
|---|---|---|---|---|---|---|
| residual | +84.7965 | +15.9866 | **+0.0464** | +0.0549 | +0.0630 | +0.0634 |

Two cross-checks pass before anything is read off these: `oooo` / `ooooo`
reproduce the `v532` arm's intensity sha bit for bit (`b2a8b150`, `88f726eb`),
and `cccoo` reproduces C11's own `340d7b32`.

**At (-1,0) the response is NON-MONOTONE in the pattern.**  Flipping group 4
from concentric to off-centre makes it WORSE (0.0552 -> 0.0689) and flipping
group 3 as well recovers almost nothing (0.0673); only the fully off-centre
pattern reaches 0.0290.  At (-2,0) the response IS monotone and C11's mixed
pattern is the best of the six.  **The chain-level metric is not a sum of
per-group terms**, and a fit-site selector only ever chooses per call.

### 5.3 The impossibility, stated as arithmetic

Any selector of the form "concentric iff `u <= lambda u*`" assigns, per order,
the prefix pattern set by `lambda` against the sorted `u/u*` of that order's
calls.  Measured:

| order | sorted `u/u*` | pattern as `lambda` falls |
|---|---|---|
| (-1,0) | 0.126, 0.484, 0.770, 1.197 | `cccc` -> `ccco` -> `ccoo` -> `cooo` -> `oooo` |
| (-2,0) | 0.255, 0.309, 0.712, inf, inf | `ccccc` -> ... -> `ooooo` |

`lambda = 1` is C11 exactly, at both orders.  To reach `oooo` at (-1,0) -- the
only pattern there that improves on v5.32 -- requires `lambda < 0.126`, and at
(-2,0), (-3,0), (-4,0) and (-4,-2) every `u/u*` exceeds 0.126 (the smallest are
0.255, 0.383, 0.181 and 0.203), so the same `lambda` gives all-off at every one
of them, i.e. **v5.32 exactly**, and every C11 gain
(0.017 / 0.052 / 0.110 / 0.082 points) is lost.

One honest caveat on the family: at (-1,0) `u/u*` is monotone in the group
index, so the reachable patterns really are the five measured prefixes.  At
(-2,0) it is not (group 2 reads 0.255 against group 1's 0.309), so a `lambda`
in (0.255, 0.309) would produce the non-prefix pattern `ocooo`, which was not
run.  It cannot rescue the argument -- reaching `oooo` at (-1,0) still needs
`lambda < 0.126`, which is below every (-2,0) value and therefore still gives
`ooooo` -- but the (-2,0) reachable set is five of six measured, not six.

The sharper version needs no family at all.  **Group 2 is preferred CONCENTRIC
by 4.77x at (-1,0) (`u` = 0.062) and by 4.24x at (-2,0) (`u` = 0.125)** -- the
same group, the LOWER decentre carrying the LARGER margin -- and the chain
wants OFF-CENTRE at the first and CONCENTRIC at the second.  No rule monotone
in the margin, in `u`, or in `u/u*` can produce that pair.

**So "improve or hold at every order" and "keep the C11 gains" are mutually
exclusive on design 121, for any per-call selector.**  That is the reason
`DECENTRED_FIT_PREDICTOR` ships `False`, and it is a stronger reason than
C11's: not "the proxy is imperfect near a tie" but "no per-call proxy exists".

---

## 6. The C11 attribution, reversed

`C11_PHYSICAL_DECENTRE_GATE` S6.4 reads:

> Moving the C6 domain alone is worth **0.154 points**; moving the ray-fit
> branch alone (C6 held beam-centred) is worth **0.028**. ... Separated, **85 %
> of "the concentric branch wins at 0.24 w" is the C6 fit domain, not the ray
> fit.**

Its premise is "at (-1,0) all six groups pick concentric".  Instrumented in
process (`c12_arb_trace_121.py`), **only four groups are arbitrated at (-1,0)**
-- groups 0 and 1 sit below niche C1's null gate -- **and one of the four picks
OFF-CENTRE** (group 5, at a 1.126 ratio).  So the arm C11 labels "CONCENTRIC,
beam-centred C6" is not the all-concentric arm.

Measured with the C6 domain held EXACTLY as shipped and only the fit-site
decision forced (S5.2):

| ray fit | C6 residual-eikonal domain | residual |
|---|---|---|
| off-centre, all four | as shipped | **+0.0290** |
| C11's own verdict (`ccco`) | as shipped | **+0.0552** |
| concentric, all four (`cccc`) | **as shipped** | **-0.0985** |

`-0.0985` is `rc_gate_121.py`'s forced-concentric arm to four decimals -- and
that arm ALSO moves the C6 domain to origin-referenced.  The two agree, so **the
C6 domain moves (-1,0)'s EE3 by less than 0.0001 points and the whole 0.154 is
the ray-fit branch**, specifically group 5's.

This does not change any C11 number; it changes what they mean.  C11 S10 item 1
("the arbiter's blind spot costs 0.028 points") should be read as S5.2 above:
the cost is 0.026 points and it is the ray-fit branch at three groups, not a
weighting defect.

---

## 7. Acceptance

### 7.1 Per-order table, design 121

`c12_gate_arms_121.py` (`c11_gate_arms_121.py` with only `ARMS` changed, so the
instrument, readout and scoring are the campaign's own), `RN=1024`, `rs=4`,
`NLO=321`, EE3 area-exact against the exact-ray `CARRY=1` ceiling, every arm
pinned EXPLICITLY so none depends on what the defaults happen to be:

| order | oracle | `default` | **C12** | C11 | v5.32 | res `default` | **res C12** | res C11 | res v5.32 |
|---|---|---|---|---|---|---|---|---|---|
| (0,0)   | 90.5324 | 90.5801 | 90.5801 | 90.5801 | 90.5801 | -0.0477 | **-0.0477** | -0.0477 | -0.0477 |
| (-1,0)  | 90.5768 | 90.5477 | 90.5215 | 90.5215 | 90.5477 | +0.0290 | **+0.0552** | +0.0552 | +0.0290 |
| (-2,0)  | 90.6650 | 90.6016 | 90.6186 | 90.6186 | 90.6016 | +0.0634 | **+0.0464** | +0.0464 | +0.0634 |
| (-3,0)  | 90.6961 | 90.6062 | 90.6586 | 90.6586 | 90.6062 | +0.0898 | **+0.0375** | +0.0375 | +0.0898 |
| (-4,0)  | 90.5035 | 90.3625 | 90.4727 | 90.4727 | 90.3625 | +0.1410 | **+0.0308** | +0.0308 | +0.1410 |
| (-4,-2) | 90.1071 | 89.9554 | 90.0378 | 90.0378 | 89.9554 | +0.1517 | **+0.0693** | +0.0693 | +0.1517 |

Intensity shas, `default` / C12 / C11 / v5.32:

| order | shas |
|---|---|
| (0,0)   | `4e9effd4` / `4e9effd4` / `4e9effd4` / `4e9effd4` |
| (-1,0)  | `b2a8b150` / `098aa4a2` / `098aa4a2` / `b2a8b150` |
| (-2,0)  | `88f726eb` / `340d7b32` / `340d7b32` / `88f726eb` |
| (-3,0)  | `cf0bc1f3` / `71af633a` / `71af633a` / `cf0bc1f3` |
| (-4,0)  | `7845b7a8` / `fe6b14c8` / `fe6b14c8` / `7845b7a8` |
| (-4,-2) | `c4b850ea` / `be68c661` / `be68c661` / `c4b850ea` |

Every `default` sha is C11 S6.5's recorded `default`/`v532` value and every C12
sha is its recorded `C11` value, from a different runner in a different process
on a tree where no C12 code existed: **6 of 6 and 6 of 6.**

**Two identities, both bit for bit and both on the design.**

* **`C12 == C11` on every order.**  The predictor's spectral arm is unresolved
  on all 26 arbitrated calls, so it decides from the measured pair, and the
  closed form is algebraically that comparison (S2.4).  This is the measurement
  that turns the argument into a fact.
* **`default == v5.32` on every order**, and both reproduce the shas C11's own
  runner recorded BEFORE any C12 code existed.  The fail-before is real.

**Against the acceptance bar, stated both ways.**

* The SHIPPED default is v5.32 bit for bit, so "improves or holds at every
  order" holds trivially and vacuously -- it is the baseline.
* With the predictor ON the table is C11's, which **fails** the per-order bar
  at (-1,0) by +0.0262 against a 0.003-0.015 differential floor, while
  improving the other four tilted orders by 0.017 / 0.052 / 0.110 / 0.082 and
  taking the worst-case residual from 0.152 to 0.069.
* **S5.3 proves that no per-call selector can do better**, so this is not a
  gap this niche left open; it is a result.

### 7.2 The intermediate-hash check -- bit-identical

The table above was taken at `dafdf9b27a6bb214`; the file's final state is
`d0d8da8d494e95f7` (S1.1).  `c12_gate_arms_121.py` re-run for (-1,0) -- the
order that carries the acceptance question -- at the FINAL hash:

```text
  _lens_traced.py d0d8da8d494e95f7
    -1,0   90.5768   90.5477   90.5215   90.5215   90.5477
           shas b2a8b150/098aa4a2/098aa4a2/b2a8b150   [295s]
```

Four arms, four intensity shas, **all four bit-identical to the
`dafdf9b27a6bb214` run**.  The diff is inert, measured rather than argued.

### 7.3 Production acceptance -- unchanged, to every printed digit

`focus_scan_121.py` **unedited**, pure library defaults (`CREF`/`AM`/`PIP`
unset), N=2048, `rs=4`, NFC=8192, WF=4.0, NOUT=2048, run through
`c12_with_predictor.py` with both flags pinned EXPLICITLY so neither arm
depends on what the defaults happen to be:

| | **predictor ON** (`PRED=1 ARBITER=0`) | **fail-before** (`PRED=0 ARBITER=0`) |
|---|---|---|
| `AT-PLANE` dz=0 | 3.350 um / 90.3 / 99.7 / 99.8 | 3.350 um / 90.3 / 99.7 / 99.8 |
| `BEST-FOCUS[peak]` plane | dz = **+0 um** | dz = **+0 um** |
| peak | **5.529e+03** | **5.529e+03** |
| `dz = +5 um` | 3.450 um / 89.6 / 99.7 / 99.8, pk 5.312e+03 | 3.450 um / 89.6 / 99.7 / 99.8, pk 5.312e+03 |

**The recorded acceptance line is unchanged: 3.350 um / 90.3 / 99.7 / 99.8.**
`diff` over every `AT-PLANE` / `dz` / `BEST-FOCUS` row of the two arms produces
no output, and the same `diff` against C11's own arbiter-ON scan also produces
no output -- so this is the campaign's own measurement, not a new one.  Zero
predictor/arbiter disagreement warnings and zero self-check failures in either
scan.

That it is IDENTICAL rather than merely close is structural and was checked
from a second direction: `focus_scan_121.py` runs the single ON-AXIS beam,
where the chief ray is on the grid centre, niche C1's null gate never opens and
neither selector ever runs -- and the (0,0) row of S7.1 is the same statement
through a different instrument, `4e9effd4` on all four arms.

### 7.4 Conservation and the self-checks -- 6 of 6

`energy_stage_audit_121.py` **unedited**, through `c12_with_predictor.py` with
`PRED=1 ARBITER=0`, `RN=1024`, `rs=4`, six post-DOE groups,
`final_leg='paraxial'`, all six orders (C11 ran four):

| order | `P_out/P_in` | `g4` | `amax4` | `r_rms` mm |
|---|---|---|---|---|
| (0,0)   | 0.994315 | 0.000e+00 | 0.000e+00 | 0.8383 |
| (-1,0)  | 0.994063 | 1.962e-11 | 1.716e-05 | 0.8384 |
| (-2,0)  | 0.994132 | 6.783e-11 | 3.213e-05 | 0.8382 |
| (-3,0)  | 0.994071 | 1.302e-09 | 5.625e-05 | 0.8380 |
| (-4,0)  | 0.994004 | 8.841e-09 | 1.075e-04 | 0.8376 |
| (-4,-2) | 0.993826 | 9.114e-09 | 1.116e-04 | 0.8375 |

* **`P_out/P_in` is inside C2's [0.9850, 1.00050] on all six**, and every value
  it shares with C11's recorded table is identical to six digits (0.994315 /
  0.994063 / 0.994132 / 0.993826);
* `amax4` peaks at **1.116e-04** against C4's 1.0e-03 bound -- 9x under;
* `r_rms` moves by **0.0009 mm** across the whole fan against C5's 0.030 mm;
* `g4` is 1e-3 or less of its C3 bound everywhere.

**The predictor and the arbiter agreed on all 26 arbitrated calls**:
`grep -c "PREDICTOR and the niche-C11 ARBITER disagree"` reads **0** across the
whole audit, which is the design-121 fall-back of S3.4 measured end to end
rather than asserted.

**(d) The C7 halo self-check and the element energy self-check stay silent.**
`grep -c "HALO self-check FAILED"` and `grep -c "energy self-check FAILED"`
both read **0** across the conservation audit, the four-arm per-order table,
the per-group pattern runs and both production focus scans.

### 7.5 Suites, both builds

| | |
|---|---|
| Windows / MKL, py3.14 -- `p2_guards`, `d3_guards`, `d7_decentred_fit`, `c1_consolidation`, `c6_fit_guard`, `c11_*` | **158 passed, 20 warnings** (13m35s) -- C11's own recorded count, unchanged |
| Windows / MKL -- **the same six + `c12_*`, at the FINAL hash** | **176 passed, 20 warnings, 0 skipped** (10m44s) |
| Linux / OpenBLAS, py3.12.3, numpy 2.4.6 -- the same seven files | **174 passed** of 176 (15m12s); the two failures were C12's own band-pinned tests, since re-derived in process (S7.6) |
| Linux / OpenBLAS -- `c12_*`, after the S7.6 fix | **18 passed** (39s) |
| `ruff check lumenairy/ tests/unit/` | **All checks passed** on BOTH builds |

`validation/run_all.py` is **unaffected**, for C11's reason unchanged: no file
it collects passes `beam_centre`, `fit_radius_beam_factor` or a
`TiltedCarrier`, so the decentred branch is never reached there in any flag
state.

### 7.6 The one test that was calibrated and had to stop being

`test_the_predictor_holds_the_concentric_branch_where_the_arbiter_switches` and
`test_a_disagreement_is_never_silent_and_names_both_scores` first pinned the
disagreement band at `|c|/w = 0.555`, measured on MKL.  They passed there and
**failed on OpenBLAS**, because the ARBITER's crossover is a difference of two
nearly-equal least-squares residuals and moves with the BLAS:

| build | oracle | arbiter | **predictor** |
|---|---|---|---|
| Windows / MKL | 0.5715 | 0.5453 | **0.5717** |
| Linux / OpenBLAS | 0.5906 | 0.5555 | **0.5717** |

0.555 sits inside the MKL band and outside the OpenBLAS one.  Both tests now
BISECT the two crossovers in process and test at their midpoint, so what is
asserted is the ORDERING (`u*_predictor > u*_arbiter`) plus an exact bitwise
identity at a decentre the same process chose.  Nothing was relaxed.

**And the incident is itself a result**: the predictor's crossover is
build-invariant to four digits while the arbiter's moves 1.9 % and the oracle's
3.3 %.  It reads the spectrum, not a near-cancellation.

### 7.7 The acceptance list, item by item

| asked | result |
|---|---|
| per-order table improves-or-holds at EVERY order incl. (-1,0) | **shipped default: yes, bit-identically (it IS v5.32).  Predictor ON: no at (-1,0), +0.026 -- and S5.3 proves no per-call selector can do otherwise on this design** |
| production acceptance 3.350 / 90.3 / 99.7 / 99.8 unchanged | **yes**, both arms, every printed digit including the peak (S7.3) |
| conservation 6/6 | **yes**, all six orders, every C2-C5 bound met (S7.4) |
| C7 silent | **yes**, 0 halo and 0 energy self-check failures across every runner here |
| niche suites green | **yes** -- 158 (Windows) / 174 of 176 then 18/18 after the S7.6 fix (Linux) |
| both-build runs | **yes**, Windows/MKL py3.14 + WSL/OpenBLAS py3.12, including the crossover bisection itself |
| ruff clean | **yes**, both builds |
| null floors | **yes** -- 10 of 10 byte-identity rows below the C1 gate across four flag states x two fixtures x three offsets (test 11), and `default == v5.32` on 6 of 6 design orders |
| sampling adequacy p99.9 | **0.000 rad** against `pi` on every oracle row scored here (S1.2) |
| fail-before restoring current behaviour bit-for-bit | **yes** (S7.1, S7.2, and test 2) |

---

## 8. What remains unresolved

1. **The spectral arm does not reach design 121, and the fix is named but not
   built.**  The surrogate is measured on the LAUNCH BOX, and on 121 that box
   is 7.5x the beam with 0.5 % of it unfilled, so the expansion does not
   converge (S3.4).  The obvious repair is to normalise the surrogate's basis
   to a BEAM-scale domain instead -- the projection identity of S2.1 holds for
   any degree-`<= m` subtraction, so the surrogate is free to be built on a
   better-conditioned domain than the one the candidates' basis uses.  Not
   attempted here.  Until it is, C12 on design 121 is C11 with a number
   attached.
2. **The order-`q` box fit is paid even when its answer is thrown away.**  Nine
   evaluator builds where C11 makes five and the pure gate makes three, and on
   the design that matters all four extra ones end in `resolved = False`.  A
   cheap PRE-test for resolution -- two box fits at `q` and `q-2` compared
   shell by shell, or one on a decimated lattice -- would let the expensive
   path be skipped, and was not built.  `_DECENTRED_FIT_SPECTRUM_ORDER = 0`
   skips it outright, at the price of making the predictor exactly the
   arbiter.
3. **The (-1,0) counter-movement is now proven unfixable AT THE FIT SITE**
   (S5.3), which moves the question upstream rather than answering it.  The
   remaining levers are the six consumers that are committed BEFORE the ray
   trace (C11 S2, items 1-6) and a per-CHAIN rather than per-call selection --
   the chain knows all six groups' candidates at once, and S5.2 shows the
   metric is a property of the SET.  Nothing here evaluates either.
4. **The oracle itself is BLAS-dependent and the predictor is not.**  The f/3
   crossover reads 0.5715 on MKL and 0.5906 on OpenBLAS (3.3 %) against a
   predictor that reads **0.5717 on both**.  Which build's oracle is closer to
   the physics is not decidable from here -- both are the same code on the same
   fixture -- so the predictor is quoted as 0.03 % from the MKL bracket and
   3.2 % from the OpenBLAS one, and the arbiter as 4.6 % and 5.9 %.  The
   predictor is the better of the two on both builds and the more STABLE of
   the two by two orders of magnitude, and that is the whole claim.
5. **The 26-call ground truth is one `ray_subsample` and one cache.**  It was
   captured at `rs = 4` against `_chainA_1024_2000nm_rs4.npz` (dated 2026-07-30,
   i.e. pre-C7/C8/C9/C10 -- C11 S1.1's deliberate trap, inherited unchanged so
   every comparison here stays differential).  D7's order step-down is
   sample-count driven, so a different `ray_subsample` can change which order
   each candidate is scored at and therefore which one wins.  Not swept.
6. **Niche C11's S4.3 per-group table and its 0.46-0.69 w band are superseded**
   (S3.4).  This document does not re-derive the rest of C11's conclusions from
   the corrected picks; the ones it does re-derive (S6) reverse.
7. **The transverse-map residual is still measured and unused** (C11 S10 item
   7), and the model of S2 says nothing about it -- it is a different tail of
   a different function, and combining the two would need a scale neither
   supplies.
8. **`carrier.py`'s `on_decentred_fit` docstring** describes the v5.32 branch
   selection and is now two niches stale (C11 S10 item 4).  `carrier.py` is
   deliberately unmodified.
9. **`_DECENTRED_FIT_POLY_ORDER = 10`'s own anomaly is untouched** (C11 S10
   item 9): orders 6, 8 and 12 close the (-1,0) residual and 10 does not.  The
   model of S2 gives a language for asking why -- `P` enters the crossover only
   through which shell the off-centre candidate's tail starts at -- but nothing
   here tests it.
10. **Nothing was run on GPU, on `inversion_method='fit'`, or through
    `apply_real_lens_traced_multi`'s prepared-screen reuse.**  The predictor
    adds an input-dependent decision to the fit site exactly as C11's arbiter
    does, so it inherits C11 S10 item 10 unchanged.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`.  Every runner forces
`LUMEN_PIN=0` and prints the library version, path and file hashes.

```bash
# S3.1 / S3.4 -- the model against the measured residuals, and m_eff
GEOMS=f6,f3,f6w python c12_predict_synth.py

# S3.2 -- THE TABLE: oracle vs arbiter vs predictor, bisected in one process
GEOMS=f6,f3,f6w python c12_oracle_crossover.py

# S3.4 -- the arbiter's OWN arguments on design 121, captured in process
ORDERS='-1,0 -2,0 -3,0 -4,0 -4,-2' python c12_arb_trace_121.py
python c12_predict_121.py

# S5.1 -- the scorer floor sweep, replayed on those captures (no ray trace)
TAGS='m1_0 m2_0 m3_0 m4_0 m4_m2' python c12_scorer_sweep.py
ORDERS='-1,0 -2,0' python c12_scorer_121.py        # the same, from live calls

# S5.2 -- the per-group branch patterns, priced at the chain level
ORDERS='-1,0' PATTERNS='v532=OFF,C11=-,allc=cccc,allo=oooo' \
    python c12_group_arms_121.py
ORDERS='-2,0' PATTERNS='cccco=cccco,ccooo=ccooo,coooo=coooo' \
    python c12_group_arms_121.py

# S7.1 / S7.2 -- the per-order table, all four arms pinned explicitly
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python c12_gate_arms_121.py
ORDERS='-1,0' python c12_gate_arms_121.py          # the hash re-check

# S7.3 / S7.4 -- production acceptance and conservation, both arms pinned
PRED=1 ARBITER=0 python c12_with_predictor.py focus_scan_121.py
PRED=0 ARBITER=0 python c12_with_predictor.py focus_scan_121.py
PRED=1 ARBITER=0 ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CONFIGS=ship NULL=0 \
    python c12_with_predictor.py energy_stage_audit_121.py

# S7.5 -- the suites, on both builds
python -m pytest tests/unit/test_niche_{p2_guards,d3_guards,d7_decentred_fit,\
c1_consolidation,c6_fit_guard,c11_*,c12_*}.py -q
wsl -e bash -lc "cd <repo> && LUMEN_PIN=0 ~/lumvenv/bin/python -m pytest \
    tests/unit/test_niche_c12_physics_fit_selection.py -q"

# S7.6 -- the crossovers in the CI proxy (this is what caught the pinned band)
wsl -e bash -lc "cd <repo>/validation/repro_traced_carrier_121 && \
    LUMEN_PIN=0 GEOMS=f3 ~/lumvenv/bin/python c12_oracle_crossover.py"
```

### Files added by this study

```
validation/repro_traced_carrier_121/c12_predict_synth.py     the model vs the measurement
validation/repro_traced_carrier_121/c12_oracle_crossover.py  oracle / arbiter / predictor
validation/repro_traced_carrier_121/c12_arb_trace_121.py     the arbiter's own inputs, in process
validation/repro_traced_carrier_121/c12_predict_121.py       the model on those inputs
validation/repro_traced_carrier_121/c12_scorer_121.py        the floor sweep, from live calls
validation/repro_traced_carrier_121/c12_scorer_sweep.py      the floor sweep, from captures
validation/repro_traced_carrier_121/c12_group_arms_121.py    per-group branch patterns
validation/repro_traced_carrier_121/c12_gate_arms_121.py     the per-order table
validation/repro_traced_carrier_121/c12_with_predictor.py    the flag-pinning wrapper
tests/unit/test_niche_c12_physics_fit_selection.py
docs/audits/C12_PHYSICS_FIT_SELECTION_2026_08_03.md          this document
```
