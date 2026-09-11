# ROUND 3 (closing) of the 5.45.1 BOR/EME guards — 2026-09-12

> **STATUS — FIX.** Closes the six gaps the independent verification of round 2
> (`docs/audits/VERIFY_BOR_GUARDS_ROUND2_2026_09_12.md`) left open: GAP 2 (P1,
> library), GAP 3 (P2 inherited, but a KERNEL-DEPENDENT verdict, so P1 by this
> project's standard), GAP 4 (P3, library), GAP 5 (P2, test), GAP 1 (P2, test)
> and GAP 6 (P3, documentation) — plus the restatement of D4's ladder claim,
> which the verification refuted.
>
> Tree: `C:\tmp\lum_bor3`, branch `fix/bor-guards-round3`, forked from
> `verify/bor-guards-round2` (`1ac6de7e`), which is the bit-identity baseline
> throughout.  Probes and per-arm JSON: `validation/probe_fix_bor_round3/`.
> Binding: `docs/TESTING_STANDARDS.md`.
>
> **Every number below was measured on this tree by probes written for this
> round.**  Where a round-2 number is repeated it is labelled as theirs and is
> either reproduced independently here or read back out of their own JSON.

---

## 1. What changed, in one table

| gap | P | where | the change | the decision it moves |
|---|---|---|---|---|
| **2** | P1 | `bor_solve._stack_media_are_passive` and `_index_ceiling_slack` (both new), `_check_nodal_passivity` | the incidence-lossless conjunct gated BOTH detectors through one early return; it now gates the ENERGY detector alone (GAIN still disarms both), and the ceiling's own per-half-space gate moves from LOSSLESS to PASSIVE at a DERIVED slack | ladder D **4/13 → 13/13 refused** (4 energy, 9 ceiling), **0 refusals and 0 warnings on 52** screened-healthy rows, **0 false positives on 600** undamaged census rows |
| **3** | P1-by-standard | `_sem_contract.verdict`, `measure_layer` | a NON-FINITE `q_excess` read as *not hot*; a ratio FORMED from a real spectrum that comes back non-finite is now HOT, one that was never formed stays cold (`q_measurable`) | the axis liner at 1e-8 of `Rbig` reads `warn_own` on **all four kernels** instead of `ok` on three and `warn_own` on one |
| **4** | P3 | `_sem_contract._BOR_FRAC_DEADBAND`, `_below` (new) | the three width comparisons were STRICT against a quantity the geometry reproduces only to round-off; they now carry the library's 16-ULP relative deadband | the same physical liner decides identically at the axis, in the interior and at the outer wall — **0 of 9 rungs disagree**, was 4 of 9 |
| **5** | P2 | `test_fix_bor_multilayer_guards` | `_gamma_of(m, idx=2)`'s default was load-bearing and undocumented; the gate now sweeps `m` 0..3 × `idx` 1..3, asserts the channel count is `idx + 1`, and scopes its energy bar to the rungs it has a two-sided gap on | 12 ladders instead of 3, with the deep rungs bounded rather than unasserted |
| **1** | P2 | `test_fix_bor_guards_round2` | the gate protecting the incidence conjunct could not fail (its 1e-3 loss emptied the channel set); replaced by the verification's live gate, restated for GAP 2's split | a gate that flips when the conjunct is deleted |
| **6** | P3 | `docs/audits/FIX_BOR_GUARDS_ROUND2_2026_09_12.md` §6.2 | two numbers attributed to the wrong probe | prose only; the conclusion and the shipped code comment were already right |

Nothing else in `lumenairy/` is touched.

---

## 2. Terms

**Passive medium** - `Im eps >= 0` in this library's `exp(-i omega t)`
convention: absorbing or lossless, but not amplifying. For a stack of passive
media `R + T + A = 1` with `A >= 0`, so `R + T <= 1` is a theorem; inside a PEC
wall a LOSSLESS passive stack has `R + T = 1` as an equality.

**The two detectors.** *The energy screen* reads `R + T` and needs the incidence
medium's flux to be a conserved power. *The index ceiling* reads no energy at
all: a returned channel whose axial index `Re qn` exceeds its own half-space's
`Re sqrt(eps)` has `gamma^2 < 0`, a transverse eigenvalue the PEC-walled
cylinder does not have. It is a contradiction, not a magnitude.

**Set-right / set-wrong** - a nodal row whose `(incidence, exit)` channel counts
do (do not) match the div-conforming STAGGERED twin's on the identical geometry.
A definition of damage that reads no energy, so an energy bar can be scored
against it without assuming its conclusion.

**Arm** - a build (Windows py3.14.6 / numpy 2.4.4 / scipy 1.17.1, or WSL
py3.12.3 / numpy 2.4.6 / scipy 1.17.1) x an `OPENBLAS_CORETYPE` x a thread
count. Every command carries `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` on the command line; the **loaded** kernel is read back from
`threadpoolctl` into every JSON and never inferred from the request.

---

## 3. GAP 2 (P1) - D1's hole, relocated by round 2, closed

### 3.1 What was wrong

`_stack_is_provably_passive` required `Im eps >= 0` on every layer **and**
`_layer_is_lossless(layers[0])`, and `_check_nodal_passivity` took that one
predicate as a single early return for BOTH detectors. So a loss of any size on
the incidence half-space - 3e-12 included, the exact rung defect D1 was named
for - disarmed the INDEX CEILING along with the energy screen.

The conjunct itself is right about the energy: `R` and `T` are formed from a
basis normalised to unit `|z-flux|` per mode, so in an absorbing incidence
medium they are not power fractions and no energy bar can mean anything. It says
nothing about the ceiling, whose Rayleigh argument concerns one half-space's own
`eps`.

**The cost, re-measured here** (`g2_ladder_d.py`, ladder D: a damaging ring stack
between two `eps = 2` half-spaces, the loss on `layers[0]` ALONE):

| `Im/Re` on `layers[0]` | channels | nodal `max(R+T)` | staggered twin | PRE | POST |
|---|---|---|---|---|---|
| 0 .. 1e-12 | 12 | 2.41297 | 1 + 1.03e-12 | REFUSED (energy) | REFUSED (energy) |
| **3e-12** | 12 | **2.41297** | 1 + 2.00e-12 | **returned** | **REFUSED (ceiling)** |
| 1e-11 | 12 | 2.41297 | 1 + 9.46e-12 | returned | **REFUSED (ceiling)** |
| 1e-9 | 12 | 2.41297 | 1 + 8.75e-10 | returned | **REFUSED (ceiling)** |
| **1e-6** | 12 | **2.41297** | **1 + 8.75e-07** | **returned** | **REFUSED (ceiling)** |
| 1e-4 .. 1e-1 | **0** | - | - | returned | **REFUSED (ceiling, exit side)** |

**4 of 13 -> 13 of 13**, 4 by the energy detector and 9 by the ceiling -
measured on BOTH trees by the same probe (`LUM_PROBE_TAG=BASE` against
`C:/tmp/lum_bor3_base`, detached at `1ac6de7e`): BASE reads
`{'refused': 4, 'by_detector': {'energy': 4, 'ceiling': 0}}` and round 3
`{'refused': 13, 'by_detector': {'energy': 4, 'ceiling': 9}}`, with ladder H
reading 52 rows / 0 refused / 0 warnings on BOTH. At
`Im/Re = 1e-6` the div-conforming twin measures the legitimate budget at
**8.75e-07** against a returned **2.41297** - 6.2 decades the flux bookkeeping
cannot account for; at 3e-12 the twin closes to 1 exactly, so the whole 1.413 is
unexplained. The five rungs with NO channels are refused on the EXIT
half-space's own returned channel set: `R` and `T` are empty there, but the
`S`-matrix the caller receives is built from the same modes, and that set
contains a channel 7.3118e-06 above the medium's index ceiling.

### 3.2 The counter-population: 0 false refusals on healthy lossy-incidence stacks

"Healthy" is SCREENED, not assumed. An accurate energy is not health: the
round-2 census's own headline row (`uniform`, `m = 1`, `N = 200`, `rbl = 2`)
closes `R + T` to 1 and returns 12 channels where its staggered twin returns 10,
and the ceiling refuses it - correctly, and already on the round-2 tree. Ladder
H therefore admits a geometry only if, at zero loss, the nodal cascade returns
the SAME channel counts as its staggered twin AND its ceiling is silent.

**Ladder H: 52 rows (4 screened geometries x 13 rungs), 0 refused, 0 warnings.**

### 3.3 Does the index ceiling HOLD on a lossy half-space? - the measurement the verification could not make

The round-2 verification listed this under "what I could not verify", and GAP
2's suggested remedy assumed it. It is now measured: `g2_lossy_ceiling.py`, **640
solves** - 2 families x `m` 0..3 x `N` 120/200 x `Rbig/lambda` 0.5/1/2/4 x
{loss on `layers[0]`, loss on BOTH half-spaces} x `Im/Re` in {1e-12, 1e-9, 1e-6,
1e-3, 1e-1}, each geometry solved in BOTH bases with the guard disarmed. 384
rows return channels; the channel gate empties the rest at `Im/Re >= 1e-3`.

| population | n | worst `Re qn - n_max` |
|---|---|---|
| NODAL, channel set RIGHT | 216 | **-1.999276e-03** |
| STAGGERED, every row | 384 | **-1.712581e-04** |
| NODAL, channel set WRONG | 168 | fires on **144**, mildest **+7.108677e-06** |

* **Zero false positives on 600 undamaged rows** (216 nodal set-right + 384
  staggered), against the >= 40 this round was asked for;
* the two populations are on **opposite sides of zero**;
* **the two-sided margin**: the mildest violation is **4.15 decades above** the
  base slack `_BOR_INDEX_CEILING_SLACK = 5e-10` and **3.75 decades above** the
  widened one; the closest undamaged approach is **2.70 decades below zero**
  (nodal) and 3.77 (staggered);
* it is the same shape as the LOSSLESS census, and the mildest violation is the
  same number to five digits (round 2 measured +7.1087e-06 on lossless rows),
  because the excess is a property of the discretisation and not of the loss;
* **no false positive appeared**, so the gate is NOT conditioned on a measured
  exception; it is conditioned on the DERIVED bound below.

### 3.4 What ships, and why the ceiling is armed on an absorbing half-space

Two changes, and the second is the one the measurement above licenses.

**(a) The predicate splits.** `_stack_media_are_passive(layers)` is the media
half - `Im eps >= 0` everywhere, to `_BOR_PASSIVE_DEADBAND` - and gates BOTH
detectors, because GAIN leaves neither the energy theorem nor the Rayleigh bound
standing. `_stack_is_provably_passive` is that AND `_layer_is_lossless(layers[0])`
and now gates the ENERGY detector alone.

**(b) The ceiling's per-half-space gate moves from LOSSLESS to PASSIVE, at a
DERIVED slack.** Round 2 silenced the conjunct wherever `_layer_is_lossless` was
false, on the reasoning that with a complex `eps` the operator is
complex-symmetric and `Re q <= n k0` is "only approximate". The first half is
right; the second is the absence of a statement. Here it is:

* `q^2` is an eigenvalue of `eps k0^2 + D` with `D` real symmetric NEGATIVE
  SEMI-DEFINITE. An eigenvalue lies in the numerical range, and for any unit `x`,
  `Re(x* (eps k0^2 + D) x) = k0^2 sum Re(eps_i) |x_i|^2 + x* D x <= k0^2 max(Re eps)`.
  So **`Re(q^2) <= max(Re eps) k0^2` EXACTLY on any passive medium** - the
  Rayleigh argument never needed losslessness. What losslessness bought was the
  step from `Re(q^2)` to `Re q`.
* That step: `Re(qn)^2 - Im(qn)^2 = Re(qn^2) <= n^2`, and the R/T channel gate
  returns only modes with `|Im qn| < _BOR_CHANNEL_IMAG_BAR`, so
  **`Re qn < sqrt(n^2 + imag_bar^2) <= n + imag_bar^2 / (2 n)`**.

`_index_ceiling_slack(L, n_max)` therefore returns the base 5e-10 on a lossless
half-space and `5e-10 + _BOR_CHANNEL_IMAG_BAR**2 / (2 n)` on an absorbing one.
At `imag_bar = 5e-5` the extra term is **1.25e-09 / n** - at most 1.25e-09,
built from two library constants and the medium's own index, calibrated against
nothing. It sits **3.75 decades below** the mildest violation measured in §3.3
and **5.1 decades above** nothing either population reaches, so it can only ever
decide a case neither population contains.

**What it buys, measured:** of the 168 set-wrong rows in the census, **72** are
refused ONLY because the conjunct is armed on a lossy half-space - the rows
where BOTH half-spaces absorb, which a `lossless` gate silences on both sides at
once.

**What it does not claim.** `eps_ceiling` is the complex `eps` at the argmax of
`Re eps`, and the code's `n_max = Re sqrt(eps_ceiling)` is >= `sqrt(max Re eps)`
for a passive medium, so the implemented ceiling is at or above the derived one
- conservative in the direction that matters (fewer false positives), which is
why the derivation is quoted as a bound rather than as an equality.

---

## 4. GAP 3 (kernel-dependent verdict) - a non-finite `q_excess` read as benign

`_sem_contract.verdict` computed

    hot = np.isfinite(excess) and excess > _BOR_Q_EXCESS

so a spectrum that had actually blown up (`excess = inf`) made `hot` **False**
and the contract fell silent exactly where the damage is worst. And because
`inf` is a backward-error outcome, the VERDICT moved with the BLAS kernel.

**Re-measured here, before the change** (`g3_sem_ladder.py`: a caller-prescribed
liner at the axis / interior / outer wall, 9 widths from `1e-9` to `1e-5` of
`Rbig`; `BORStack(Rbig=24, m=1, N=120, basis='sem', degree=8)`, Windows /
Haswell / 1 thread):

| width / `Rbig` | axis `q_excess` | axis | middle | outer |
|---|---|---|---|---|
| 1e-9 | **inf** | **ok** | `warn_own` | `warn_own` |
| 3e-9 | **inf** | **ok** | `warn_own` | `warn_own` |
| 1e-8 | **inf** | **ok** | `warn_own` | `warn_own` |
| 3e-8 | 2.965e+07 | `warn_own` | `warn_own` | `warn_own` |
| 1e-7 | 8.895e+06 | `warn_own` | `warn_own` | `warn_own` |
| 3e-7 | 2.965e+06 | `warn_own` | `warn_own` | `warn_own` |
| **1e-6 (the edge)** | 8.895e+05 | **`ok`** | `warn_own` | `warn_own` |
| 3e-6 | 2.965e+05 | `ok` | `ok` | `ok` |
| 1e-5 | 8.895e+04 | `ok` | `ok` | `ok` |

Three of the four disagreeing rungs are GAP 3; the fourth is GAP 4. The round-2
verification measured the same geometry reading `inf` / `ok` on Haswell, Nehalem
and Katmai and `8.89487e+07` / `warn_own` on Sandybridge, on both builds - a
mesh-contract decision moving with the kernel for a geometry the caller wrote
once, which is the shape `docs/TESTING_STANDARDS.md` exists to forbid.

**What ships.** `verdict` now reads

    hot = (excess > _BOR_Q_EXCESS if np.isfinite(excess)
           else bool(rec.get("q_measurable", False)))

with `measure_layer` recording `q_measurable = bool(qa.size and den > 0.0)`. The
distinction is the point: a ratio **formed** from a real spectrum that comes back
`inf` or `nan` is past every bar there is, so it is HOT; a ratio that was never
formed (no modes, or a zero `n_max k0` denominator) is evidence in neither
direction and stays COLD - the conservative reading, and the pre-round-3
behaviour for that case. Conflating the two was the defect.

**After**: the axis liner reads `warn_own` at all three non-finite rungs, on
every arm, which is what Sandybridge already read. The cost of the old reading
was bounded - a missing `UserWarning`, never a wrong returned number - which is
why the verification filed it P2; it is fixed with the round because a
kernel-dependent verdict is P1 by this project's standard.

---

## 5. GAP 4 (P3) - the `warn_own` edge decided by the representation of the width

`verdict`'s three arms compared with a **strict** `<` against
`_BOR_MIN_ELEM_FRAC` and `_BOR_SLIVER_BAND_FRAC` - exact decimal literals -
while the left-hand side is a difference of two mesh breakpoints. At the edge:

| position | the liner's two walls | `w_min_own_frac` | PRE verdict |
|---|---|---|---|
| axis | `0`, `w` | **1.000000000000e-06** exactly | **`ok`** |
| middle | `6.0`, `6.0 + w` | 9.999999999917482e-07 | `warn_own` |
| outer | `Rbig - w`, `Rbig` | 9.999999999917482e-07 | `warn_own` |

The axis liner's walls difference to `w` exactly; the other two lose **8.2518e-12
relative - 38,968 ULP** - to the subtraction of two numbers of order `Rbig`.
Neither reading is wrong about the geometry; the EDGE was wrong to be a strict
comparison at the representation limit.

**What ships.** `_BOR_FRAC_DEADBAND = 16 * eps` (3.5527e-15 relative) and a
`_below(x, bar)` helper all three comparisons go through. 16 ULP is the sizing
constant the library's other at-the-limit bars already use
(`pmm/stack._PASSIVE_ANTIHERM_DEADBAND`, `_WALL_SNAP_DEADBAND`,
`bor_solve._BOR_PASSIVE_DEADBAND`). It is **symmetric**: every `x` within 16 ULP
of `bar` on either side decides the same way, and the bar itself resolves to the
INFORMATIVE side (the narrow one).

**It does not need to span the 38,968-ULP spread** - only the exact tie AT the
bar, the one representation the strict comparison excluded; the other two were
already below it. **Two-sided:** the nearest ladder rung above the edge is `3e-6`
(3.0x) and below it `3e-7` (0.30x), so the deadband sits **14.4 decades inside**
the closest width any rung actually asks for. It can only ever decide a tie.

**After**: 0 of 9 rungs disagree across the three positions, on every arm.

---

## 6. GAP 5 (P2, test) - the near-cutoff bar's margin was a property of an un-swept axis

`_gamma_of(m, idx=2)` fixed the RADIAL cutoff index by a default no caller
overrode and no docstring mentioned, so `_CUTOFF_LADDER_BAR = 1e-5`'s "7.86x over
the family" was a statement about `m` alone.

**Re-measured on an independently written probe** (`g5_cutoff_family.py`,
`g5b_cutoff_mechanism.py`: `m` 0..3 x `idx` 1..3 x the same floor-derived 13-rung
ladder = 156 solves; Windows / Haswell / 1 thread), worst lossless closure per
ladder:

| `m` | `idx=1` | `idx=2` (the gate's own) | `idx=3` |
|---|---|---|---|
| 0 | 5.548884e-05 | 1.965486e-07 | 7.091522e-05 |
| 1 | **1.787437e-04** | 3.626719e-08 | 2.760941e-06 |
| 2 | 2.068969e-05 | 5.527032e-08 | 1.895889e-05 |
| 3 | 8.507884e-06 | 6.914638e-09 | 9.329587e-06 |

**5 of 12** combinations exceed 1e-5, the worst by **17.9x** - the round-2
verification's 1.787437e-04 reproduced to seven digits on a fixture written
independently of theirs. The worst rung is the DEEPEST of every ladder.

### 6.1 Why the bar could not simply be widened

**The PRE-fix defect this gate exists to refuse is SMALLER than the residual.**
The shipped per-mode band read `|R + T - 1| = 1.2167e-04` at `qn ~ 2.5e-03`
(`test_near_cutoff_closure`, and the CHANGELOG entry's own number), where this
tree reads 2.3e-11. A scalar bar between 1.787437e-04 and 1.2167e-04 does not
exist: widening to cover the residual puts the defect inside the bar.

The populations separate on an axis, not on a scalar, and the axis is the rung's
own distance from cutoff - `qn_marginal = n sqrt(delta)` - in units of the R/T
channel gate's floor `_orient._BOR_CHANNEL_REAL_FLOOR` (1e-6):

| regime | rows | worst closure |
|---|---|---|
| `qn >= 100x` the channel floor | 9 rungs x 12 ladders = **108** | **6.889470e-07** |
| `qn <  100x` the channel floor | 4 rungs x 12 ladders = **48** | **1.787437e-04** |

**2.41 decades apart.** 100x is the measured knee, not a round number: at 100x
(`qn` = 1.41e-04) the family reads 6.889e-07 and at the next rung down (79x,
`qn` = 7.93e-05) it reads 2.604e-06 - the first row over 1e-06 anywhere in the
grid.

### 6.2 Is 1.79e-04 the band's noise side or a residual defect? - measured

**It is a residual, it is deterministic, and it is NOT the orientation band.**
Three measurements at the worst rung (`m=1`, `idx=1`, `delta` = 1e-10,
`qn` = 1.410e-05):

1. **The channel COUNT - the integer the band exists to protect - is one number
   over the whole ladder on 12 of 12 `(m, idx)` combinations**, and it is the
   RIGHT number, `idx + 1`, on every one.
2. **Zero modes inside the classifier band carry BACKWARD flux** - the 5.45.1
   defect's own signature - in either half-space. (84 and 119 of the 240 modes
   are backward overall, as they must be; none is in the band.)
3. **The marginal channel's flux has collapsed, which is the mechanism.** The
   weakest in-band `|flux|` is **5.5399e-08** against the strongest
   **4.5564e-03** in the superstrate - five decades. The ladder drives one
   channel to cutoff, so its `qn -> 0` and its `r dr` z-flux goes with it
   (`P/fnrm ~ qn` for the limiting polarization family); the modal basis
   normalizes by `1/sqrt(|P|)`, so that ONE column's amplitude diverges and the
   cascade loses digits in it. The excess lands on the marginal channel's own row
   of `R + T` on **17 of the 28** rows whose closure exceeds 1e-07.

So it is the same population `_CUTOFF_LADDER_FLOOR_MULT` exists to keep this gate
away from, one effect earlier: the COUNT survives down to 10x the channel floor,
the CLOSURE leaves its round-off floor at about 100x. **It is a residual, and it
is on the noise side of the BAND CLAIM** - the band claim is about orientation
and count, and both are intact at the rung where the closure is worst.

### 6.3 What ships

* `_gamma_of`'s docstring names `idx`, says what it selects (`idx + 1` channels)
  and records that the default is no longer load-bearing;
* `test_near_cutoff_channel_count_is_stable_over_the_ladder` is parametrized over
  `m` 0..3 x `idx` 1..3 - **12 ladders, 156 solves** - and makes three claims in
  decreasing strength: the count is ONE number (unconditional), the count is
  `idx + 1` (unconditional), and the closure is inside its bar;
* `_CUTOFF_LADDER_BAR` is **unchanged at 1e-5** and now SCOPED, with the
  measurement and the reason it could not move: **14.5x (1.16 decades) above**
  the measured family envelope 6.8895e-07 and **12.2x (1.09 decades) below** the
  pre-fix defect 1.2167e-04 - two-sided;
* `_CUTOFF_ENERGY_FLOOR_MULT = 100.0` carries the scope and the knee;
* `_CUTOFF_DEEP_BAR = 2e-3` bounds the deep rungs, **11.2x (1.05 decades)** above
  their measured envelope, declared ONE-SIDED by construction: there is no defect
  population above it, because on those rungs the pre-fix band's failure shows up
  in the COUNT, which is asserted unconditionally at every rung.

`test_verify_bor_guards_round2._CUT_FAMILY_BAR` keeps the same 2e-3 and now
cross-references this measurement; that gate becomes an independent cross-check
rather than the only place the axis is exercised.

---

## 7. GAP 1 (P2, test) - a gate that could not fail, replaced

`test_the_screen_disarms_on_an_absorbing_incidence_medium` built its superstrate
at `eps = 2 + 2e-3j`, i.e. `Im/Re = 1e-3`. `_orient.channel_core` drops any mode
whose `|Im qn|` exceeds `_BOR_CHANNEL_IMAG_BAR` (5e-5), so at that loss the
channel set is **EMPTY**, `R + T` is empty, and `_check_nodal_passivity` returned
at its `e.size == 0` line **without ever calling the predicate**. The round-2
verification proved it by mutation: deleting the incidence conjunct left the gate
passing.

It is **retired**. Its replacement -
`test_the_energy_screen_disarms_on_an_absorbing_incidence_medium_but_the_ceiling_does_not`,
taken from the verification's
`test_the_absorbing_incidence_gate_needs_a_non_empty_channel_set` and restated
for GAP 2's split - runs the same geometry at `Im/Re = 1e-7`, two decades INSIDE
the channel gate, and asserts four things:

* its own premise (the channel set survives, so the predicate is reached);
* `_stack_is_provably_passive` is **False** (the energy theorem is off);
* `_stack_media_are_passive` is **True** (the Rayleigh bound is intact) -
  deleting the conjunct collapses these two and flips the gate;
* the caller receives **no energy verdict and no energy warning**, and
  (premise-gated on this arm's cascade actually returning a channel above the
  ceiling) the INDEX CEILING **does** refuse.

The verification file keeps its other gates; a comment there records the move.

**And the replacement is PROVED live by the same mutation that convicted the
one it replaces** (`g9_mutation.py`, which patches
`_stack_is_provably_passive = _stack_media_are_passive` at runtime -- the
predicate with its incidence-lossless conjunct removed -- rather than editing
the library):

    intact  -> PASS
    mutated -> FAIL: the predicate called a stack with an ABSORBING incidence
               medium provably passive; R and T come from a unit-|z-flux| ...
    VERDICT: LIVE

A gate that asserts a conjunct is worth nothing until the conjunct's removal is
shown to break it; the retired gate failed exactly that test in round 2.

---

## 8. GAP 6 (P3, doc) - a number attributed to the wrong probe

Round 2's section 6.2 read *"Re-measured, `r8_band_sides.py`: MIN `sigma` =
2.3820e-05 at `Im(n)` = 1e-3 and 2.3820e-08 at 1e-6, i.e. 3.38 and 0.38
decades."* That probe's own JSON
(`validation/probe_fix_bor_round2/r8_band_sides_win_Haswell_t1.json`,
`summary.signal`) reports minima of **3.7752e-05** and **3.7752e-08** - 3.577 and
0.577 decades. 2.3820e-08 is the ROUND-1 verification's own number, from its own
battery.

The prose is corrected in place with a dated note. **The conclusion does not
move**: the minimum over the union of the two batteries is the smaller of the
two, so 2.3820e-08 / 0.38 decades is still the figure the band carries, and
`_orient.orient_band_scale`'s docstring already attributed both numbers to their
own probes correctly. Recorded because it is exactly the
right-conclusion-wrong-numbers shape `docs/TESTING_STANDARDS.md` names as the
most dangerous.

---

## 9. D4's ladder claim - restated to what holds

Round 2's report generalised two passing rungs into *"the three positions behave
identically over a width ladder"*. The verification refuted that as a ladder
claim (2 of its 11 rungs disagreed, identically on both trees), and this round
reproduces the refutation at **4 of 9 rungs** on its own ladder - three of them
GAP 3 and one GAP 4, as sections 4 and 5 show.

**Both causes are now fixed, so the generalised claim is true as stated**, and it
is pinned as a ladder rather than as two rungs:
`test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r` now runs six
widths DERIVED from `_BOR_MIN_ELEM_FRAC` (1e-3x, 1e-2x, 0.1x, 1x, 3x and 100x of
the edge) at all three positions and asserts, at every rung, that the three
agree, that the verdict is the right one, that a caller-prescribed liner is never
attributed to the union, and that it always has an owner where the contract must
speak. The round-2 report's D4 section carries a dated restatement of both the
refutation and the fix.

---

## 10. Runs

### 10.1 The 21-file BOR/EME set, four kernels x 1 thread + Haswell x 4, both builds

The round-2 verification's 20-file set plus
`tests/unit/test_verify_bor_guards_round2.py`, whose three strict xfails round 3
flips to live gates. Every arm records `lumenairy.__file__` and the LOADED
kernel from `threadpoolctl` before pytest runs; scripts are
`validation/probe_fix_bor_round3/runs/arm_win.sh` and `arm_wsl.sh`, logs
`set_<build>_<CORETYPE>_t<threads>.log` beside them.

| # | build | requested | **loaded** | thr | passed | skipped | failed | errors | time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Windows py3.14 | HASWELL | Haswell | 1 | **403** | 0 | 0 | 0 | 1431.14 s |
| 2 | Windows py3.14 | NEHALEM | Nehalem | 1 | **403** | 0 | 0 | 0 | 1985.07 s |
| 3 | Windows py3.14 | KATMAI | Katmai | 1 | **403** | 0 | 0 | 0 | 2033.20 s |
| 4 | Windows py3.14 | SANDYBRIDGE | Sandybridge | 1 | **402** | 1 | 0 | 0 | 1804.37 s |
| 5 | Windows py3.14 | HASWELL | Haswell | 4 | **403** | 0 | 0 | 0 | 1240.23 s |
| 6 | WSL py3.12 | HASWELL | Haswell | 1 | **403** | 0 | 0 | 0 | 1321.02 s |
| 7 | WSL py3.12 | NEHALEM | Nehalem | 1 | **403** | 0 | 0 | 0 | 1794.68 s |
| 8 | WSL py3.12 | KATMAI | Katmai | 1 | **403** | 0 | 0 | 0 | 1967.47 s |
| 9 | WSL py3.12 | SANDYBRIDGE | Sandybridge | 1 | **402** | 1 | 0 | 0 | 1703.54 s |
| 10 | WSL py3.12 | HASWELL | Haswell | 4 | **403** | 0 | 0 | 0 | 1019.53 s |

**Zero failures and zero errors on every one of the ten**, every tail carrying a
real summary line and no arm reporting `no tests ran`. The count is 403 against
the round-2 verification's 389 on the same battery: +5 for
`test_verify_bor_guards_round2.py` joining the set (its 6 gates minus the one
that moved), +9 for the near-cutoff gate going from 3 parametrizations to 12,
and the retired vacuous gate replaced one-for-one.

**The one skip is GAP 3's own premise gate, on both Sandybridge arms**, and it
carries its reading:

    SKIPPED tests/unit/test_verify_bor_guards_round2.py:261: premise absent on
    this arm: the axis liner's q_excess is finite (8.895e+07), so the
    non-finite branch is not exercised

That is the premise gate working, and it is also an independent cross-check:
8.895e+07 reproduces the round-2 verification's Sandybridge reading
(8.89487e+07) to four digits, on both builds, a round later. Every other
pathology-asserting gate in the changed files held its premise on all ten arms.
No strict-xfail escape occurred anywhere, because round 3 removed the three
strict xfails it closed.

> **A reading error worth recording, because it nearly became a finding.**
> The arm scripts pipe pytest through `tee`, so pytest's progress dots are
> BLOCK-buffered: the log file's byte count lags the run by kilobytes and is not
> a live progress indicator. Watching it mid-round produced an apparent stall
> (3 dots in 8 minutes) that looked like cross-build I/O contention, and an
> apparent 180x "recovery" after a WSL arm was stopped that was simply the next
> buffer flush. The process was CPU-bound the whole time -- **1950 s of CPU in
> 1953 s of wall clock** -- and the arm finished in 2033 s against the round-2
> verification's 2534 s on the same kernel. The two builds' arms run
> concurrently without interfering: Windows and WSL Haswell/1 took 1431 s and
> 1321 s here against 1841 s and 1708 s there. Nothing in this round rests on
> that non-finding, and it is recorded because a log tail read as a measurement
> is exactly the shape `docs/TESTING_STANDARDS.md` warns about.

### 10.1b The JAX-guarded BOR / EME files, Windows

`test_bor_sem_jax.py`, `test_eme_jax_modes.py`, `test_v5_20_11_bor_jax.py` with
jax 0.11.0 present on the Windows build (so nothing is skipped for a missing
backend): **22 passed in 39.79 s**, Haswell / 1 thread. These three files are
also inside the 21-file set above; this run exists to record that the JAX twins
were exercised rather than skipped.

### 10.2 The census / walker / dispatcher-pin sweep

    python -m pytest $(ls tests/unit | grep -iE 'walker|census|dispatcher_pin|public_api|doc_consistency'                        | sed 's#^#tests/unit/#') -q -x -p no:randomly

**1288 passed, 12 skipped** (Windows / Haswell / 1 thread). Every skip carries
its reading, and all 12 are the CHANGELOG walkers declining to enforce on the
`## [5.45.0]` block (no audit closures, no self-citation, no line-count claim
declared there) - unchanged by this round, which adds only `[Unreleased]`
paragraphs and no version header.

### 10.3 ruff

    wsl -e bash -lc 'cd /mnt/c/tmp/lum_bor3 && ~/lumvenv/bin/ruff check                      lumenairy/ tests/ validation/probe_fix_bor_round3/'
    -> All checks passed!

### 10.4 `.test_durations`

Re-measured by `validation/probe_fix_bor_round3/g8_durations.py`, which runs the
three touched files with `--durations=0 -vv` (deliberately WITHOUT `-q`: `-q`
decrements pytest's verbosity below the threshold at which sub-5 ms durations
are printed, and the first pass of this splice silently lost 12 fast keys that
way), drops every key belonging to those files, re-adds the measured set,
re-sorts and re-validates.

| | |
|---|---|
| entries before / after | 12,886 / **12,894** |
| keys dropped / added | 83 / **91** |
| keys per file | 13 / 73 / 5 - one per collected test, none defaulted |
| sorted, JSON-valid after the splice | **yes** (asserted in the probe) |
| retired key `test_the_screen_disarms_on_an_absorbing_incidence_medium` | **gone** |
| moved key `test_the_absorbing_incidence_gate_needs_a_non_empty_channel_set` | **gone** |
| the 12 `near_cutoff_channel_count...[m-idx]` keys | 5.77 - 6.31 s each |
| `test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r` | 5.81 -> **18.0 s** (two rungs -> six) |
| slowest key this round wrote | `test_ordinary_geometry_census[mesh-12]`, **51.67 s** (86 % of the cap) |
| keys this round wrote that exceed the 60 s shard cap | **0** |

The 26 entries elsewhere in the file that exceed 60 s are pre-existing and
untouched.

---

## 11. Bit identity - and why one battery of it is not probative

**The round-2 verification's own battery** (46 BOR + 19 EME fixtures, each
hashed to the SHA-256 of the exact IEEE-754 bytes of its answer, a raising
fixture recorded as its exception class), re-run against `1ac6de7e`:

| build / loaded kernel | common | identical | moved | illegal |
|---|---|---|---|---|
| Windows py3.14 / Haswell / 1 | 65 | **65 (BOR 46 / EME 19)** | 0 | 0 |
| WSL py3.12 / Haswell / 1 | 65 | **65 (BOR 46 / EME 19)** | 0 | 0 |

**That result is honest and NOT probative for round 3**, and the reason is in
the battery: `v7_identity._bor_nodal_fixture` puts its `im_rel` on the MIDDLE
layer only - both half-spaces are `F.uniform(2.0)`, exactly lossless, on every
one of its 22 legacy-nodal rows. So it contains no row of the population round 3
changed. Reporting "65 of 65 identical" as evidence that GAP 2's fix is safe
would be the same mistake round 2's own §7 warned about when its battery turned
out not to contain D13's population.

**So round 3 adds the battery that IS that population**
(`g7_lossy_identity.py`: 144 legacy-nodal fixtures - 3 profile families x `m`
0..2 x `Rbig/lambda` 0.5/2 x {loss on `layers[0]`, loss on BOTH half-spaces} x
`Im/Re` in {1e-12, 1e-9, 1e-6, 1e-3}):

| | BASE (`1ac6de7e`) | ROUND 3 |
|---|---|---|
| fixtures | 144 | 144 |
| refused | 28 | **58** |
| identical | - | **114** |
| moved | - | **30** |
| of which `HASH -> BORNodalPassivityError` | - | **30 of 30** |
| **illegal moves (a changed ANSWER)** | - | **0** |

Every one of the 30 is a row whose loss sits on a half-space, at
`Im/Re` in {1e-9, 1e-6, 1e-3}, on `Rbig/lambda = 2` - the damaging cell size -
across all three profile families. **No BOR answer changed value on either
battery.**

---

## 12. What I could not measure

* **The CI's actual kernels.** `ZEN` aliases Haswell and `SKYLAKEX` dies on this
  host (Ryzen 9 5950X, Zen 3); the EPYC 9V74 / 7763 pool is not reproducible
  here. Every gate added or changed asserts its invariant unconditionally and
  premise-gates every pathology claim.
* **Whether the derived lossy-ceiling term is TIGHT.** §3.4 derives
  `Re qn <= n + imag_bar^2 / (2 n)` and the census measures the nearest
  undamaged approach 5.1 decades away, so the term is never exercised by any
  row measured. It is a bound that has not been made to bind; a fixture that
  drives a passive lossy half-space's returned channel to within 1e-9 of its own
  ceiling would decide whether it is tight or merely safe, and I did not
  construct one.
* **The `4 of 44` population both detectors miss.** Unchanged from round 2 and
  its verification: it lives at `m = 3` on a near-degenerate radial pair, and
  round 3 neither closes it nor exercises it. The lossy census here reaches
  `m = 3` and finds the same shape (the ceiling misses 24 of 168 set-wrong
  rows), so the blind spot is a property of the basis, not of the gating.
* **An external accuracy oracle for the SEM answer**, PML, and the anisotropic
  Class-C populations. Unchanged from all three prior documents.
* **`.test_durations` on a quiet box.** The timings spliced here were taken with
  several arms in flight, so they are upper bounds - the safe direction for a
  shard balancer, but not a clean measurement.
* **Whether `_CUTOFF_DEEP_BAR` has a signal side.** It is declared one-sided by
  construction (§6.3) because on the deep rungs the pre-fix band's failure shows
  in the COUNT. Measuring what the PRE-fix band's CLOSURE would have been at
  those rungs would need the pre-fix band re-implemented, which this round did
  not do.
