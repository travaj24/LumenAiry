# VERIFY -- round 4 of the PURE staggered 2-D PMM per-layer grids (L2 mortar)

**Date** 2026-09-11 · **Worktree** `C:/tmp/lum_vmortar4`, branch
`verify/mortar-round4`, created from `b7239bf` (the wave2/pmm2d tip, which
carries round 4).  The pre-round-4 tree is `15af675` at
`C:/tmp/lum_vmortar4_pre`.

**Under verification**
`docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md` -- the
exactly-one-promoted-side WORDING, the PER-AXIS band warning, and the two
restated test bars.

**Method** every number below was RE-MEASURED in this worktree through
`validation/probe_verify_mortar_round4/`, on fixtures written for this
verification and disjoint from the fix's: period 1.24e-6 m, wavelength
0.905e-6 m, `theta` 0.29, `phi` 0.71, substrate 1.45, host 2.31, pillars 7.29
/ 5.76 / 6.76 / 5.29, its own out-of-plane and magnetic tensors, its own wall
arrays, slant (0.13, 0.06), and its own layer thicknesses.  **Nothing is read
from the fix's JSON.**

The fix's own probe runs both rules in ONE interpreter by replacing
`_stag_mortared_axes`.  This verification does the opposite: **the same probe
file is run against BOTH TREES, in separate interpreters, on both builds**, so
the comparison holds nothing fixed by construction.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Verdict

| # | round-4 claim | verdict |
|---|---|---|
| 1 | the mechanism needs an ASYMMETRIC interface -- EXACTLY ONE promoted side; both-promoted is HEALTHY | **CONFIRMED** (S3) |
| 1b | the wording is corrected everywhere a user or a later reader meets it | **CONFIRMED**, one nit (S3.2, DEFECT 3) |
| 2a | no ANSWER moves across the change | **CONFIRMED** -- 782 leaves per build, **0** answer differences, **0** other differences, on BOTH builds (S4.1) |
| 2b | the false positive on a non-mortared axis stops | **CONFIRMED** -- 5 fixtures stop warning, identical set on both builds (S4.1) |
| 2c | a stack narrow on both axes now names the MORTARED axis, once | **CONFIRMED** -- `y` @ 3.000e-03 -> `x` @ 9.000e-03, count 1 (S4.1) |
| 2d | every warning that fired for a real reason still fires | **CONFIRMED** -- 10 fixtures, unchanged count/axis/width (S4.1) |
| 2e | the per-axis narrowest is `>=` the all-axes narrowest, so the change can never ADD a warning | **CONFIRMED** -- 35/35, and it is a min-over-a-subset identity (S4.2) |
| 2f | the axis the rule SKIPS is genuinely harmless | **CONFIRMED and STRENGTHENED** -- the fix's evidence is ONE-ARMED; the two-armed measurement is S5 (DEFECT 2, P4) |
| 3a | `max(ctrl.on) < 0.95` -> the SPREAD statistics | **CONFIRMED** -- 3.56 decades / 1.49e+03x / 4.07x on 39 independent operands (S6.1) |
| 3b | the `M` = 6 ratio gate -> "M = 5 AND 6 with a convergence precondition", stated as removing the SAMPLE scope | **REFUTED AS STATED / BOUNDED** -- the new precondition is read AT rung 6 and certifies nothing about rung 5; an independent fixture satisfies it by 8.4x and reads **0.4733** at `M` = 5, 2.4x BELOW the bar asserted there (S6.2, DEFECT 1, P3) |
| 3c | the census margin is 1.67x, not 3.6x, and is SAMPLE-scoped | **CONFIRMED** (S6.3) |
| 4 | "no warning" is not "no degradation" | **CONFIRMED and corroborated** -- on this verification's own fixture **86 %** of the total damage is already present AT the edge (S5.3) |
| B(iv) | `window_halfwidth` interactions | **NOT APPLICABLE** -- `PMM2DStackPure` RAISES on the keyword; no interaction exists (S4.4) |

**SHIP for 5.45.0.**  Round 4 is a warning-SCOPE change; it moves no answer on
either build, the scope it narrows to is the correct one, and the two
remaining findings are a test-durability P3 and two P4 documentation items.
None of them blocks a release.  DEFECT 1 is now gated in
`tests/unit/test_verify_pmm2d_mortar_round4.py`, so it cannot be lost.

---

## S1. Terms, defined before they are used

**Mortar.**  The L2 projection coupling two adjacent per-layer element grids.
It exists on an interface only where the two layers' wall arrays DIFFER
(`stack2d_pure.py`: `same = ga.key() == gb.key()`).

**A MORTARED AXIS.**  An axis (x or y) on which some adjacent pair of layers
carries different wall arrays.  On a CONFORMING axis the cross-mass is the
layer's own mass matrix and the projection is the identity.

**The band.**  `[_STAG_MIN_SEG_FRAC, _STAG_SLIVER_BAND_FRAC)` = `[1e-3, 3e-2)`
of the period, both edges carrying 1e-9 relative slack.  Below it a stack is
REFUSED; inside it the solve proceeds and WARNS; above it it is silent.

**A PURE PARTITION wall.**  A wall separating two segments of the SAME
permittivity.  The device cannot depend on where it sits, so ANY movement of
the answer as its segment narrows is numerical damage rather than physics.
Every ladder in S5 sweeps such a wall, and the quantity reported is
`max |R(g) - R(3e-1)|`, `max |T(g) - T(3e-1)|` -- an ABSOLUTE efficiency move
against the widest rung of the same device.

**PROMOTION, and an ASYMMETRIC interface.**  `_modes_as_general` rewrites a
symmetric in-plane region's modes as `(W, V, lam, W, -V, -lam)`.  An interface
is ASYMMETRIC when EXACTLY ONE side is promoted.  The test used here is
STRUCTURAL and exact (`W2 is W` and the bitwise negations), written
independently in `v2_bars.py`, not imported from the fix's instrument.

---

## S2. The two builds

| | Windows (WIN) | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 (scipy-openblas) | 1.17.1 (scipy-openblas) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1`, on the COMMAND LINE | same |
| `lumenairy` | `C:\tmp\lum_vmortar4\lumenairy\__init__.py` 5.44.0 | `/mnt/c/tmp/lum_vmortar4/...` 5.44.0 |

Both link scipy-openblas, so **every cross-build spread quoted here is a LOWER
bound** -- the caveat rounds 2, 3 and 4 and both earlier verifications carry.
Every probe imports `_path.py` first, which pins the requested tree and
REFUSES to run if `lumenairy.__file__` resolves anywhere else; the resolved
path is recorded in every JSON.

Cross-build, on the round-4 tree: answer sha256s DIFFER on all 35 fixtures (as
they must -- two BLAS builds), `R00` agrees to **7.7e-10 relative at worst**,
and **every warning field is identical on every fixture**.  The geometry
section is identical byte for byte.

---

## S3. Claim 1 -- the ASYMMETRIC-interface wording

### 3.1 The measurement, re-made on 39 independent operands

`v2_bars.py --parts p`, over 12 fixtures at `M` = 4, 5, 6 -- an out-of-plane
layer beside a uniform spacer, beside an in-plane patterned layer, beside an
in-plane MAGNETIC one, a slanted layer beside an in-plane one, a strongly
out-of-plane layer, a 1e-8-rad slant, a three-layer stack whose middle
interface has BOTH sides promoted, and the both-out-of-plane / both-slanted /
in-plane-to-1e-9 controls:

| class | operands | `s_min/s_max` | near-null `max(on)` | SPREAD `min(on)/max(on)` |
|---|---|---|---|---|
| **EXACTLY ONE promoted** | 27 | 1.6178e-15 .. 5.7534e-10 | 0.999999999999962 .. 1.000000000000000 | **2.4126e-09 .. 2.7227e-07** |
| **BOTH promoted** | 3 | 2.2512e-07 .. 2.6466e-06 | 0.8617 .. 0.9264 | **0.4065 .. 0.5888** |
| NEITHER promoted (controls) | 9 | 1.5176e-05 .. 1.3779e-03 | 0.7312 .. 0.8787 | 0.5432 .. 0.9328 |

WIN and WSL agree to **12 significant figures** on every entry except the two
that are themselves at the round-off floor -- the worst one-promoted SPREAD
(2.722719580e-07 WIN / 2.722729182e-07 WSL, 6 figures) and `s_min/s_max` on
the one-promoted operands (4 figures at 1.6e-15) -- which is what a near-null
direction is entitled to.  The residual of every operand in all three classes
is `<= 3.4e-14` -- the
generalized system is CONSISTENT throughout, which is why the residual screen
accepts all three and why nothing shipped behaves differently.

**The classes are cleanly separated and the both-promoted class sits with the
CONTROLS, not with the defect.**  The worst one-promoted spread (2.72e-07) and
the best both-promoted spread (0.4065) are **6.2 decades** apart; in
`s_min/s_max` the two classes are **2.59 decades** apart at their closest
(5.7534e-10 against 2.2512e-07), and the both-promoted class's best reading
sits **0.76 decades** below the controls' worst (2.6466e-06 against
1.5176e-05).
A reader deriving a bar from "either side promoted" would mis-predict a
both-promoted interface by six decades in the quantity that matters.  Round
4's correction is **CONFIRMED**, on a population that shares no fixture with
the one it was derived from.

### 3.2 The wording, as a user meets it

The refusal message was triggered at RUNTIME (an exactly singular operand with
a right-hand side outside its range, through `_guarded_mortar_solve(...,
screen='residual')`) and reads

> `... This site's operand is RANK-DEFICIENT BY CONSTRUCTION whenever EXACTLY
> ONE side of the interface is an in-plane region promoted to the generalized
> 6-tuple form ...`

`"whenever one side"` and `"either side"` do not appear in it.  A tree-wide
grep for `either side` / `whenever (one|either) side` finds no LIVE statement
of the mechanism in the old form: every remaining hit is either unrelated
(`min_feature`, the lens code, the band's own floating-point slack) or a DATED
CORRECTION that quotes the old wording in order to correct it -- in
`_core.py`, `test_fix_pmm2d_mortar_round3.py`,
`test_verify_pmm2d_mortar_round3.py`, the round-3 fix doc and the CHANGELOG.

**Nit (DEFECT 3, P4).**  `docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md`
line 562 carries a forward-note (the dated round-3 superseding block appended
to the round-2 doc) that still states the mechanism as *"RANK-DEFICIENT BY
CONSTRUCTION whenever one side of the interface is an in-plane region
promoted"*, uncorrected.  Round 4 corrected the round-3 doc and the CHANGELOG
but not this one.  It is a historical audit doc, so this is a completeness
nit, not a defect in anything shipped.

---

## S4. Claim 2 -- the band warning, per AXIS

### 4.1 Bit-identity across the change, on 35 fixtures, both builds

`v1_axis_identity.py --parts fgw` run against `15af675` and against `b7239bf`
in separate interpreters, then `v4_compare.py` flattens both result trees and
compares every leaf by name (timings and provenance excluded BY NAME, not by
tolerance).  The family is 32 stacks -- fully conforming (2- and 3-layer);
non-conforming on **x only**, on **y only**, on **both**; mixed
in-plane/out-of-plane in four shapes; a MAGNETIC in-plane layer beside an
out-of-plane one, and its conforming control; a SLANTED layer beside a uniform
spacer, and one with a narrow shared y wall; closing tapers at **8, 9 and 16**
slices; and the five ADVERSARIAL stacks of S4.3 -- plus 3 solved again with
`force_mortar=True`.

| | WIN | WSL |
|---|---|---|
| leaves compared | 782 | 782 |
| **ANSWER differences** (sha256 of `(orders, R, T)`, `R00` both rows, `T00`, closure) | **0** | **0** |
| warning differences | 17 | 17 |
| other differences (geometry, per-axis reading, band edges, ...) | **0** | **0** |

The 17 are the same 17 on both builds, on 6 fixtures, and every one is a field
of a WARNING:

| fixture | shape | ROUND 3 | ROUND 4 | classification |
|---|---|---|---|---|
| `x_only_narrow_y` | layers differ on x, y wall array SHARED, narrow y | 1, y, 3.000e-03 | **silent** | count + axis + width |
| `y_only_narrow_x` | the MIRROR: differ on y, x SHARED, narrow x | 1, x, 3.000e-03 | **silent** | count + axis + width |
| `oop_narrow_y_conf` | the same with an OUT-OF-PLANE layer | 1, y, 3.000e-03 | **silent** | count + axis + width |
| `magnetic_narrow_y_conf` | the same with a MAGNETIC neighbour | 1, y, 3.000e-03 | **silent** | count + axis + width |
| `advB_ymortar_narrow_x` | y mortared, narrow x a pure PARTITION | 1, x, 3.000e-03 | **silent** | count + axis + width |
| `x_only_mixed` | x mortared @ 9e-3, y conforming @ 3e-3 | 1, **y**, 3.000e-03 | 1, **x**, 9.000e-03 | axis + width, count UNCHANGED |

`y_only_narrow_x` is the MIRROR of the fix's own DEFECT-2 shape; the fix
measured only the y-conforming direction.  The rule is symmetric under x <-> y,
measured.

The warnings that fired for a real reason are **unchanged in count, axis and
width** across the change on all of: `taper9` (1, x, 2.889e-02), `taper16`
(1, x, 1.625e-02), `x_only_narrow_x` (1, x, 5.000e-03), `y_only_narrow_y`
(1, y, 5.000e-03), `both_narrow_x` (1, x, 4.000e-03), `both_narrow_y`
(1, y, 4.000e-03), `slant_spacer_narrow_y` (1, y, 3.000e-03), `advB_i`
(1, x, 6.000e-03), `advB_ii` (1, y, 4.000e-03), `advB_far` (1, x, 5.000e-03),
plus both `force_mortar` fixtures (which make BOTH axes live, as round 3 did).
`taper8` is 3.250e-02 on this verification's taper (`w_bottom` = 0.52), i.e.
**1.08x OUTSIDE** the edge and silent, while `taper9` at 2.889e-02 is inside
-- the "enters the band from NINE slices" claim reproduced on a different
taper width.

**Any answer difference would be a defect.  There is none, on either build.**

*(Note on the lossless closure: these fixtures run `n_orders = 1`, so higher
propagating orders are truncated out of the sum and the closure is not a
tripwire here -- it reaches 2.17e-01 on `taper16`.  It is reported only
because it is identical across the two trees, which is the point.)*

### 4.2 The rule itself, re-derived rather than read

`v1_axis_identity.py --parts g` computes, for every fixture, the narrowest
segment per axis and which axes are mortared **from the fingerprint PAIR
`StagGridOps.key()` directly**, not through `_stag_mortared_axes`, and then
applies both rules by hand.  On all 35 entries:

* the independently-derived `(x, y)` liveness pair **equals** the library's
  `_stag_mortared_axes` reading, including under `force`;
* the independently-derived round-4 DECISION (warn / axis / width) equals the
  warning the solve actually raised, on both builds;
* the **per-axis narrowest is `>=` the all-axes narrowest on every fixture**.
  That is a min-over-a-subset identity, so it is not merely observed: the
  round-4 search runs over a SUBSET of the axes round 3 searched, and
  therefore **can never ADD a warning**.  CONFIRMED.

### 4.3 Trying to break the per-axis rule (task B)

| # | shape built | what the rule must do | measured DECISION | deserved? |
|---|---|---|---|---|
| (i) | `advB_i` -- x wall arrays differ by a wall that is NOT the narrow one; the narrow x segment (6e-3) is IDENTICAL in both layers | still WARN, naming x | **warns, x, 6.000e-03**, both rules, both builds | **YES** -- the same shape as a ladder moves the answer 3.87e-06, **20x** the no-mortar floor (1.89e-07) |
| (ii) | `advB_ii` -- three layers: A and B share the y walls and differ on x, B and C differ on y; the narrow y segment (4e-3) is common to all three | see the B-C interface's y mortar and WARN, naming y | **warns, y, 4.000e-03** | **YES** -- moves 6.71e-07, 3.5x the floor |
| (iii) | `advB_iii` -- four layers, ALTERNATING conformity (share x / differ y, then differ x / share y, then share both) | both axes live, no band segment -> silent | **silent**, `mortared_axes = (True, True)` | n/a |
| (iii') | `advB_far` -- the narrow x segment is in L0 whose BOTH neighbours conform on x; the only x mortar is at the far L2-L3 interface | (the rule asks the STACK per axis) WARN | **warns, x, 5.000e-03** | **OVER-warns**, the SAFE direction.  A per-INTERFACE rule would be silent here; the set reading is a conservative envelope of it |
| (iv) | `window_halfwidth = 1` and `= 2` on `PMM2DStackPure` | -- | **RAISES** ("no meaning in the staggered basis ... drop the keyword") | no interaction can exist |
| (v) | `advB_ymortar_narrow_x` -- the MISS candidate: y carries the mortar, x CONFORMS and carries a 3e-3 segment that is a pure PARTITION | silent, IF the segment is really harmless | **silent** under round 4, warned under round 3 | **YES, silent is right** -- see S5 |

The one asymmetry worth recording is (iii'): `_stag_mortared_axes` asks the
question of the STACK per axis, not of each INTERFACE, so a narrow segment
whose own neighbours conform on that axis still warns when some OTHER pair
differs on it.  That is strictly more conservative than the mechanism
requires and cannot silence a real warning, so it is not a defect -- but the
docstring's "some ADJACENT pair differs" is the reason the rule is SOUND, not
a description of the segment it names.

### 4.4 `window_halfwidth`

`PMM2DStackPure.__init__` accepts the keyword only to REFUSE it, for both 1
and 2, with a message explaining that a per-layer grid here is a segment
PARTITION so a window would mean the union grid.  Task B(iv) therefore has no
content: **no interaction is reachable.**

---

## S5. The decision the rule takes, measured against the DAMAGE

This is the part the fix's evidence does not cover, and it is where this
verification adds the most.

### 5.1 DEFECT 2 (P4, evidence) -- the fix's DEFECT-2 measurement is ONE-ARMED

Round 4's S4.3 sweeps the SHARED y wall pair of a two-layer stack of UNIFORM
permittivity (2.5 beside 3.5) whose layers differ on x, and reports that
`R00` is 0.247088457739 at every rung with a band/ordinary ratio of
1.000000000000.  **It never sweeps the MORTARED axis of that same device**,
which is the arm that says whether the device can move at all.  As stated,
"the answer does not move" is compatible with "this device cannot move".

Reproduced here knob for knob (`_vfix4.fixdev`, the same period/wavelength
ratio in metres), and the missing arm run:

| the fix's own device | axis swept | `M` | move @ 3e-2 | move @ 1.2e-3 | warns |
|---|---|---|---|---|---|
| their arm | y (CONFORMING) | 4 | 2.213e-10 | 2.179e-10 | no |
| their arm | y (CONFORMING) | 5 | 9.824e-13 | 9.446e-13 | no |
| **the missing arm** | **x (MORTARED)** | 4 | 5.534e-05 | **6.765e-05** | yes, x |
| **the missing arm** | **x (MORTARED)** | 5 | 4.276e-07 | **5.586e-07** | yes, x |

WSL reproduces every entry to four significant figures.  So the fix's device
DOES separate the two axes -- by **5.7 decades** at `M` = 5 -- and its
conclusion stands.  What was missing is the arm that establishes it.  P4:
evidence completeness, no shipped consequence.

### 5.2 The two-armed measurement, on a stack where the mortar does real work

`v3_ladder.py`.  Three layers, the middle one uniform host; the permittivity
depends on ONE axis only, so every wall on the OTHER axis is a pure PARTITION
choice; the swept wall is always such a wall.  `M` = 4 and 6, six rungs from
3e-1 to 1.2e-3.  Move = `max |R(g) - R(3e-1)|, |T(g) - T(3e-1)|`.

| shape | the swept axis | `M` = 4 move @ 1.2e-3 | `M` = 6 move @ 1.2e-3 | round-4 decision |
|---|---|---|---|---|
| `Xmort` | **MORTARED** | **7.6103e-03** | **1.7308e-02** | warns, x |
| `Xconf_y` | CONFORMING (x is mortared) | 2.3704e-07 | 4.3913e-10 | silent |
| `Xnomort` | no mortar ANYWHERE | 1.5429e-07 | 2.9834e-10 | silent |
| `Ymort` | **MORTARED** | **7.2277e-03** | **3.1011e-02** | warns, y |
| `Yconf_x` | CONFORMING (y is mortared) -- the MIRROR | 1.1077e-07 | 3.9819e-10 | silent |

WIN and WSL agree to **four significant figures on every entry** (worst
disagreement 7e-04 relative, on the 1e-10 rows; the 1e-02 / 1e-03 rows agree
to all five figures printed).

Three things follow, and all three are DECISIONS rather than readings:

1. **The axis the rule KEEPS is damaged**: 7.6e-03 / 7.2e-03 at `M` = 4, on a
   device that cannot depend on the swept wall at all.
2. **The axis it SKIPS is not**: 2.4e-07 / 1.1e-07, which is
   **3.1e-05 / 1.5e-05 of the mortared arm** -- four to five decades down --
   and within **1.54x** of a stack that builds NO mortar anywhere.  It is not
   merely small; it is what the no-mortar control costs.
3. **The separation is a FLOOR beside a DISCRETISATION.**  From `M` = 4 to 6
   the mortared arm GROWS (7.6e-03 -> 1.7e-02, and 7.2e-03 -> 3.1e-02) while
   the conforming arm and the no-mortar control FALL by ~3 decades
   (2.4e-07 -> 4.4e-10; 1.5e-07 -> 3.0e-10).  A convergence study removes the
   conforming-axis residue and does not touch the mortared one.

Round 4's DECISION is therefore **CONFIRMED on a fixture family that
discriminates**, in both axis orientations, on both builds.  Gated by
`tests/unit/test_verify_pmm2d_mortar_round4.py::
test_the_axis_the_rule_skips_is_harmless_and_the_one_it_keeps_is_not`, which
also carries the FAIL-BEFORE arm (under round 3's rule both conforming arms
warn, naming the conforming axis).

### 5.3 "No warning" is not "no degradation" -- corroborated independently

The claim is in the CHANGELOG, in `_warn_stag_sliver_band`'s docstring and in
`_STAG_SLIVER_BAND_FRAC`'s comment, and is gated.  Its substance also falls
out of S5.2 without using the fix's numbers: on `Xmort`, the move at the
band's upper EDGE (3e-2) is 6.5133e-03 against 7.6103e-03 at the width
contract -- **86 % of the total damage is already present at the edge**, and
only a further **1.17x** is added over the 1.4 decades below it (1.47x at
`M` = 6).  The fix quotes 1.24x for that last stretch on its own fixture.
CONFIRMED.

---

## S6. Claim 3 -- the two restated bars

### 6.1 The SPREAD statistics -- CONFIRMED

Re-derived on this verification's own 39 operands (S3.1); each bar is scored
at the WORST reading of the whole population, which is stricter than the gate
(the gate compares two operands of one solve):

| bar | population | measured worst | margin | round 4 claimed |
|---|---|---|---|---|
| `s_mixed < 1e-3` | 27 one-promoted | 2.7227e-07 | **3.56 decades** | 3.6 decades |
| `s_ctrl > 1e3 * s_mixed` | worst control vs worst mixed | 0.40653 vs 1e3 x 2.7227e-07 | **1.49e+03x** | 1.5e+04x (on the gate's own pair) |
| `s_ctrl > 0.1` | 9 neither- + 3 both-promoted | 0.40653 | **4.07x** | 3.8x |

WIN and WSL agree to 12 significant figures on the two control readings and
to 6 on the worst mixed one, which is at the round-off floor.  The middle row
is
an order of magnitude thinner than the fix's number only because it is scored
across fixtures rather than within one solve; it is still three decades of
margin.

The complaint the restatement answers is also reproduced: the OLD bar
`max(ctrl.on) < 0.95` has only **0.0236** of slack on this verification's own
control population (worst legitimate reading 0.92638, on a both-promoted
interface), against the fix's 0.015 on theirs.  The old bar was
sample-scoped; the restatement is not.  **CONFIRMED.**

### 6.2 DEFECT 1 (P3) -- the `M` = 5 rung is asserted at a rung the new precondition does not certify

The restated gate
(`test_fix_pmm2d_mortar_round3.py::test_the_band_the_warning_names_carries_a_
measurable_cost`) now asserts

```py
assert self_gap < 0.05 * abs(e_band - e_ord)          # oracle
assert ladder[6][0] < 0.5 * ladder[5][0]              # NEW precondition
for M, (eo, eb) in sorted(ladder.items()):            # M = 5 AND 6
    assert eb > 1.15 * eo
```

The precondition is read **at rung 6** -- it says the ordinary arm is still
falling *there*.  It certifies nothing about rung 5, and rung 5 is the rung
the restatement ADDED.  S14's complaint was precisely that a ratio at an
UNVERIFIED rung is contaminated.

Re-measured on an independent fixture (`v2_bars.py --parts l`): period
1.19e-6 m, wavelength 0.83e-6 m, `theta` 0.21, a y-INVARIANT three-layer
grating whose middle layer is uniform host, driven at `phi` = 0 so the exact
1-D `PMMStack` -- a different assembly with no mortar and no element grid --
is an INDEPENDENT oracle whose truth does not move with the swept width.
Oracle self-gap (degree 12 -> 14) **1.193027e-06**:

| `M` | `e_ord` (3e-01) | `e_band` (3e-03) | ratio | ordinary arm's fall |
|---|---|---|---|---|
| 5 | 1.647715e-01 | 7.799080e-02 | **0.4733** | -- |
| 6 | 9.780604e-03 | 1.456122e-02 | 1.4888 | **16.85x** |
| 7 | 2.499662e-03 | 7.225423e-03 | 2.8906 | 3.91x |

WIN and WSL agree to **13 significant figures** -- it is a pure discretisation
quantity with no meaningful cross-build spread, so this is not a flake.

* the shipped precondition **PASSES with 8.4x of margin** (`e_ord(6)/e_ord(5)`
  = 5.9359e-02 against 0.5);
* the shipped oracle precondition passes by 4007x at `M` = 6;
* and the `M` = 5 ratio reads **0.4733**, **2.4x BELOW** the 1.15 asserted
  there.

So the restatement does not achieve what it claims: adding a second rung did
not make the bar fixture-independent, because the added rung is the LESS
converged one and carries no precondition of its own.  On the fix's own
fixture `M` = 5 happens to read 1.611; that is a property of that fixture, not
of the library.

**Severity P3 -- test durability, not a library defect.**  The gate runs a
fixed fixture and will not flake; what is wrong is the claim that the bar is
now family-scoped.  **Reproducer**

```sh
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VMORTAR4_TREE=C:/tmp/lum_vmortar4 python \
  validation/probe_verify_mortar_round4/v2_bars.py --tag repro --parts l \
  --ladder-M 5,6
```

**Gated** by `tests/unit/test_verify_pmm2d_mortar_round4.py::
test_the_ordinary_arm_precondition_does_not_certify_the_m5_rung` (20 s WIN /
23 s WSL), which asserts that this fixture satisfies the precondition by 4.2x
AND reads below 0.9 x `e_ord` at `M` = 5.  **If the gate is later given a
per-rung precondition, or drops the `M` = 5 rung, that test fails -- and the
failure is the gate working.  Re-pin it, do not relax it.**

The **remedy** is small and does not need a new measurement: either drop the
`M` = 5 assertion (keeping the rung as the precondition's input, which is what
it is actually good for), or give each asserted rung its own convergence
precondition (`e_ord(M) < 0.5 e_ord(M-1)` for every `M` the loop asserts).

### 6.3 The census margin -- CONFIRMED

The 1.67x is arithmetic and reproduces: a duty-`d` pillar partitions the
period as `((1-d)/2, d, (1-d)/2)`, so a duty-0.9 pillar's narrowest segment is
**5.0000e-02 = 1.667x** the 3e-2 edge, a 16-cell uniform lattice is
`1/16 = 6.2500e-02 = 2.083x`, and a **94 %-duty pillar sits exactly ON the
edge** (`(1-0.94)/2 = 3.0e-2`).  Every ordinary geometry in this
verification's own 35-fixture family also stays above the edge on both builds,
and 0 of them warn.  The restatement from 3.6x (battery-scoped) to 1.67x
(family-scoped), gated at 1.25x, is **CONFIRMED** -- and it is right to call
it SAMPLE-scoped, because it is still a property of whatever census is run.

---

## S7. Defects

| # | severity | what | where | status |
|---|---|---|---|---|
| **1** | **P3** | the restated ladder gate asserts `e_band > 1.15 e_ord` at `M` = 5, a rung its new precondition does not certify; an independent fixture satisfies the precondition by 8.4x and reads 0.4733 there | `tests/unit/test_fix_pmm2d_mortar_round3.py::test_the_band_the_warning_names_carries_a_measurable_cost` | **GATED** here (S6.2); remedy stated |
| **2** | **P4** | the fix's DEFECT-2 (a) measurement is ONE-ARMED -- the mortared axis of the same device is never swept, so "the answer does not move" is not distinguished from "this device cannot move" | `docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md` S4.3 | missing arm MEASURED here (S5.1); conclusion stands |
| **3** | **P4** | the round-2 fix doc's forward-note still states the mechanism as "whenever one side of the interface is an in-plane region promoted" | `docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md` line 562 | one dated correction line would close it |

No P0, P1 or P2.  **No answer moves anywhere in this change, on either
build.**

---

## S8. What could NOT be verified

1. **A THIRD BLAS FAMILY.**  Both builds link scipy-openblas; every
   cross-build spread here is a LOWER bound.
2. **`M` = 8 on the patterned ladders.**  Run at `M` = 4 and 6, where the
   mortared/conforming separation already widens from 2.7e+04x to 3.9e+07x.
   `M` = 8 on a three-layer 3x3 stack is a 882-dimension region eigenproblem
   per layer and was not affordable beside the rest of this budget; the
   trend across the two rungs measured is the evidence for it.
3. **The fix's 801-leaf comparison was not reproduced leaf for leaf.**  This
   verification ran its OWN 782-leaf comparison on its own fixtures, against
   the pre-round-4 TREE rather than against an in-process emulation.  The two
   are independent measurements of the same claim, not a re-run of theirs.
4. **The fix's 47-geometry census was not re-run through the shipped
   builders.**  Its binding number (the duty-0.9 pillar's 5.0e-2) is
   arithmetic and was confirmed as such; the wider census's *composition* is
   taken as reported.
5. **The whole `pmm2d` surface was not re-run per ARM.**  The two-tree
   comparison covers 35 fixtures; the suites in S9 were run on the round-4
   tree only.  What bounds the rest is the bit-identity result plus the fact
   that the change touches only which axes a WARNING scans -- there is no
   path from it to an answer.
6. **The `M` = 5 rung's behaviour on a THIRD ladder fixture.**  DEFECT 1 is
   established by one independent fixture reading 0.4733 where the shipped one
   reads 1.611.  That is enough to refute "family-scoped"; it is not a
   distribution.

---

## S9. Runs

| run | WIN | WSL |
|---|---|---|
| the five suites the task names (`_run_5suites_{win,wsl}.txt`) | **72 passed** in 495.77 s | **72 passed** in 502.22 s |
| the same five plus this verification's own file (`_run_6suites_{win,wsl}.txt`) | **74 passed** in 661.77 s | **74 passed** in 601.25 s |
| `tests/unit/test_verify_pmm2d_mortar_round4.py` alone | **2 passed** in 28.43 s (8.11 s + 20.15 s) | **2 passed** in 31.03 s (7.24 s + 23.47 s) |
| `ruff check lumenairy/ tests/ validation/probe_verify_mortar_round4/` | -- | **All checks passed!** |

`.test_durations` carries both new tests (8.11 / 20.15), spliced in sorted
position and JSON-validated; the diff is 2 insertions and nothing else.

## S10. Commands

Every command is in `validation/probe_verify_mortar_round4/README.md`, which
also lists what each probe measures and where its logs are.
