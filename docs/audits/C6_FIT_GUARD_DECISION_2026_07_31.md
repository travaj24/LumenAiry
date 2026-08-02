# `REMAP_STATIONARY_PHASE_FIT_GUARD` measured end to end - the decision

**Date** 2026-07-31 - **Scope** the working tree of
`fix/pmm-union-grid-conditioning` @ `d2e60ca` plus the uncommitted C4/C5/C6 and
the two 2026-07-31 audits - **Subject** design 121's post-DOE chain, orders
(0,0), (-1,0), (-2,0), (-3,0), (-4,0), (-4,-2), at `ray_subsample` 4 and 2 -
**Question** does turning the fit guard on make the concentric branch pass the
energy audit's six bounds, and what does it cost?

---

## 0. Headline

**The guard does exactly what the energy audit hoped on (0,0), does nothing at
all on (-4,0)/(-4,-2), and DOES NOT FIX (-2,0) - which the energy audit left
open (its S7.5) and which is the reason this is not a default flip on the
conservation evidence alone.**

Scored against the energy audit's six proposed bounds, guard OFF -> ON, at
`ray_subsample = 4` and `2`, in the diagnostic (relay-only) configuration:

| order | rs=4 OFF | rs=4 ON | rs=2 OFF | rs=2 ON | what moved |
|---|---|---|---|---|---|
| **(0,0)** | **2 of 6** | **6 of 6** | **0 of 6** | **6 of 6** | **fixed outright** |
| (-1,0) | 6 of 6 | 6 of 6 | 6 of 6 | 6 of 6 | nothing |
| **(-2,0)** | 4 of 6 | **4 of 6** | 4 of 6 | **4 of 6** | **nothing - still fails C3+C4** |
| (-3,0) | 4 of 6 | 5 of 6 | 4 of 6 | 4 of 6 | C3 flips at rs=4 only |
| (-4,0) | 6 of 6 | 6 of 6 | 6 of 6 | 6 of 6 | byte-identical |
| (-4,-2) | 6 of 6 | 6 of 6 | 6 of 6 | 6 of 6 | byte-identical |

On (0,0) the guard takes the last group's `g4` from 3.400e-03 to **exactly
0.000e+00** (`rs=4`) and 4.495e-03 to **exactly 0.000e+00** (`rs=2`), `amax4`
from 0.770 / 0.978 to **exactly 0**, restores the discretisation deficit floor
from 0.15x of the C6-off reference to **0.98x / 0.99x**, and pulls the second
moment from +11.20 % / +12.73 % against the exact ray trace to **-0.27 % /
-0.52 %**. At `rs=2` it removes the above-unity element ratio (1.003696 ->
0.999199) and with it the above-unity chain (1.003186 -> 0.998692).

**Three findings the record did not have.**

1. **"Byte-identical on every tilted order" is FALSE at chain level.** The
   branch gate is per GROUP, not per order: a group takes the concentric branch
   whenever its own beam decentre is <= `_DECENTRE_GATE_W_FRAC` = 0.05 w, and
   the early groups of a mildly tilted order are. Measured by replaying every
   group on its own captured input (S2), the guard moves **all six** groups of
   (0,0), the **first two** of (-1,0), the **first** of (-2,0) and (-3,0), and
   **none** of (-4,0)/(-4,-2). Byte-identity of the chain holds on the last two
   orders only; on the other three the first group moves by 2.2e-04 of peak and
   cascades to up to 2.0e-03 at the chain exit.
2. **It cannot fix (-2,0) or (-3,0), and the reason is structural.** Those
   orders' last group sits at 0.481 w / 0.723 w - already on the off-centre
   weighted branch - so the guard is inert exactly where their lobe is made.
   That closes the energy audit's open item S7.5 ("(-2,0)'s fit branch is not
   established") with the answer **off-centre**. Their lobe is C6's and it is
   made on the weighted branch, so it is a *different* defect from the
   concentric one this flag addresses.
3. **The counter-example that keeps it opt-in was re-measured on the current
   tree and it stands.** On `probe_ghost_synthetic.py`'s design-121-scale
   stand-in singlet the hard mask is exactly clean (`P/Pin` 0.99828, halo
   0.000e+00) and the guard manufactures energy: **`P/Pin` = 1.00697** with
   8.7e-03 of the input power beyond the exact-ray exit support at **88 % of
   peak**. A second fixture ('medium, finer grid') goes from 6.9e-06 to
   **4.6e-02** of peak. Two of six fixtures regress. Design 121 itself shows no
   such regression anywhere - the real design was checked, not only the
   synthetic - but a library-wide default is not answerable to one design.

**The EE half of the trade, in the production configuration (S4).** On (0,0)
the guard keeps **+0.727** of C6's **+1.691** on-axis EE3 points while removing
**100 %** of the manufactured energy - but it costs **-0.590 EE6 points against
the shipped path and -0.581 against C6-off**, a regression neither of the other
two configurations pays. On (-2,0) it costs **nothing at all**: EE3, EE6, EE12,
`P_tile`, `elem(last)` and `r_rms` are identical to every printed digit.

**DECISION: the default stays `False`; the flag stays opt-in.** Two
independent reasons, either of which would be sufficient:

1. it trades this design's artefact for another design's - on two of six
   synthetic fixtures it *manufactures* the energy the hard mask does not, one
   of them to `P/Pin` = 1.00697;
2. even on the order it repairs it is not an unambiguous EE win - **-0.59 EE6
   points** against both the shipped path and C6-off, i.e. it moves ~0.6 % of
   the power from inside 6 um to between 6 and 12 um.

What ships instead is the instrument that makes either choice *visible*: a
**halo-amplitude term on the ray-density self-check** (S5.2), calibrated over
180 element calls, which fires on design 121's on-axis relay exit with the
guard OFF and is silent with it ON - and fires on the guard's own regression
fixture, so a caller who takes the lever is told when it backfires. **On its
first run through the niche suite it also found a manufactured lobe nobody had
reported**, in niche D6's exact-tilted-leg retrace: 0.641 of peak beyond
2.020 mm where the exact ray trace reaches 1.616 mm, in a fixture two green
tests depend on (S5.2).

**And a precise negative on the other half of the brief (S5.1): the energy
self-check's GAIN tolerance cannot be tightened at all.** A currently-green P2
battery cell reads `P_out/P_ap` = **1.04374** at the subsample CI runs it at -
the same magnitude as the defects the audit wanted caught (1.03317, 1.04593).
The observable is exhausted; the constant is left at 0.050 and the reason is
now in the source.

---

## 1. Provenance, instruments, floors, sampling

### 1.1 What was measured, and against which library

Every runner prints the sha256 of the file it imported.

| state | `elements/_lens_traced.py` | `propagators/carrier.py` |
|---|---|---|
| the energy audit's L2 = this study's START | `adcd1cee14dc7a0a` | `1a90453a4ef65399` |
| after the halo self-check first landed (S5.2) | `1d8e7f34182eb888` | `1a90453a4ef65399` |
| **final** (the decline condition + documentation) | **`f789409298b3acc6`** | `1a90453a4ef65399` |

**All conservation, halo and second-moment numbers in S3 were taken on the
START state** (`adcd1cee14dc7a0a`), i.e. on exactly the library the energy
audit's L2 tables were taken on - which is why every guard-OFF row here
reproduces that document to the digit (`(0,0)` ship `elem(5)` 0.999371, `g4`
3.400e-03, `amax4` 7.701e-01, `r_rms` 0.9349 mm; `noC6` 0.995901 / 3.606e-11 /
1.395e-05 / 0.8422 mm).

**The library change made afterwards is proved to move nothing** (S6.1):
`probe_c6_byte_identity.py` reproduces the *committed* `HEAD` library bit for
bit on all 12 synthetic cases and all 14 design-121 chain cases, and the
`energy_ee_vs_conservation_121.py` NULL intervention is `array_equal = True`,
`max|dE| = 0.000e+00` on the new library.

`carrier.py` was **not modified by this study at all.**

### 1.2 Instruments

| instrument | what it measures | new? |
|---|---|---|
| `energy_stage_audit_121.py` | per-stage chain energy budget + halo; **+ a `guard` config axis and a pairwise per-stage byte-identity line** | extended |
| `fitguard_branch_map.py` | **which groups the guard actually touches**, by per-group replay | **new** |
| `halo_calibration.py` | the halo statistic over the P2 battery, the synthetic fixtures and design 121, at five radius factors from one pass | **new** |
| `energy_ee_vs_conservation_121.py` | both currencies on ONE production field; **+ the guard axis and a halo-warning column** | extended |
| `energy_hull_121.py` | the exact traced exit hull + halo ceiling + r2m reference (run for the NEW order (-3,0)) | reused |
| `probe_ghost_synthetic.py` | the guard's own regression fixtures | reused verbatim |
| `probe_c6_byte_identity.py`, `probe_c6_tilted_failbefore.py` | the fail-before contracts | reused verbatim |
| `focus_scan_121.py` | the single-beam acceptance | reused |

### 1.3 Differential floors - bit-exact, established before any delta

| instrument | null intervention | reading |
|---|---|---|
| `energy_stage_audit_121.py` | two identical shipped chain runs | **all six stages `array_equal = True`, `max\|dE\| = 0.000e+00`**, on every order measured |
| `energy_ee_vs_conservation_121.py` | two identical production runs, on the NEW library | **`array_equal = True`, `max\|dE\| = 0.000e+00`** |
| `probe_ghost_synthetic.py` | two identical runs per fixture | `array_equal = True`, all six fixtures |
| `fitguard_branch_map.py` | guard-off vs guard-off replay is the same call | (the guard-ON/OFF pair is itself the measurement; a `no` row IS a zero-floor result) |

Every delta below is against a floor of exactly zero.

### 1.4 Sampling adequacy - unchanged, and still saturated

Stated as an amplitude-weighted wrapped nearest-neighbour step at **p99.9**
against pi, never a max. Re-measured on the new order:

| field | p99.9 | limit |
|---|---|---|
| last-group `E_in`, (-3,0) | **3.1368 rad** | pi = 3.1416 |

The last group's exit NA at (-3,0) is 0.2962 against this grid's Nyquist NA of
0.0197 (`dx` 33.211 um) - **15.0x short** of `lambda/(2 NA_exit)`, with
**96.65 %** of the exit power above the grid's Nyquist angle. This is the same
condition the energy audit documented on the other five orders and it carries
the same restriction: **every number in this document is a power measurement of
a returned array or an exact geometric ray trace. No wave or WFE claim is made
anywhere.** Halo radii at an element's own exit plane are set by the Newton
inversion of the traced ray map - a geometric placement, not a propagation - so
they are exact for the returned array; where a manufactured lobe ends up after
further propagation is *not* determined here.

### 1.5 The conservation and halo reference

`energy_hull_121.py`, `NL=801`, the element's own carrier and input amplitude,
traced with the Zemax-validated skew trace. The five previously-measured orders
are carried from the energy audit S3.1; **(-3,0) is new**:

| order | **exact-ray `g4` ceiling** | C3 bound (3x) | **exact-ray `r_rms`** | 3w exit hull |
|---|---|---|---|---|
| (0,0) | 3.5641e-10 | 1.069e-09 | 0.8407 mm | 3.755 mm |
| (-1,0) | 1.0153e-08 | 3.046e-08 | 0.8413 | 3.807 |
| (-2,0) | 3.0443e-08 | 9.133e-08 | 0.8427 | 3.865 |
| **(-3,0)** | **5.9279e-08** | **1.778e-07** | **0.8450** | **3.931** |
| (-4,0) | 7.5559e-08 | 2.267e-07 | 0.8483 | 4.004 |
| (-4,-2) | 7.4639e-08 | 2.239e-07 | 0.8503 | 4.042 |

(-3,0) slots monotonically between (-2,0) and (-4,0) on every column, which is
the cheap consistency check on the new row.

---

## 2. What the guard actually touches - the reach map

`fitguard_branch_map.py`. One shipped chain run per order is captured; then
**each group's element call is replayed on its own captured input** with the
guard off and on. A chain cascades - if group 0 moves, groups 1-5 move whether
or not the guard touched them - so only a per-group replay can answer this.

| order | grp0 | grp1 | grp2 | grp3 | grp4 | grp5 | groups the guard MOVES |
|---|---|---|---|---|---|---|---|
| (0,0) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **all six** |
| (-1,0) | 0.0128 | 0.0493 | 0.0623 | 0.1530 | 0.1989 | 0.2406 | **0 and 1** |
| (-2,0) | 0.0255 | 0.0986 | 0.1246 | 0.3061 | 0.3978 | 0.4813 | **0 only** |
| (-3,0) | 0.0383 | 0.1480 | 0.1869 | 0.4593 | 0.5968 | 0.7227 | **0 only** |
| (-4,0) | 0.0511 | 0.1974 | 0.2493 | 0.6126 | 0.7960 | 0.9647 | **none** |
| (-4,-2) | 0.0571 | 0.2207 | 0.2788 | 0.6850 | 0.8900 | 1.0791 | **none** |

Numbers are the group's beam decentre in input beam radii. The gate is
`_DECENTRE_GATE_W_FRAC = 0.05 w`; every group at or below it takes the
CONCENTRIC branch and every group above it is already on the weighted one.

Three consequences.

1. **The reach is a per-group property, not a per-order one.** The record's
   "byte-identical on every tilted order" was an element-level inference from
   the LAST group and it does not survive at chain level. `(-4,0)` and
   `(-4,-2)` are byte-identical (`array_equal` on all six stages, at `rs` 4 AND
   2, `max|dE| = 0.000e+00`); `(-1,0)`, `(-2,0)` and `(-3,0)` are not.
2. **What moves on those three is the FIRST group, by 2.2e-04 of peak**, and
   the chain-exit difference (up to 2.0e-03 of peak at (-2,0), rs=4) is that
   perturbation cascaded. It carries no conservation signal: `elem(5)` agrees
   to all six printed digits on (-1,0), (-2,0) at rs=4 and (-3,0), and to five
   at (-2,0) rs=2.
3. **The guard is inert where (-2,0)/(-3,0)'s lobe is made.** Their last group
   is at 0.481 w / 0.723 w. Their halo warning fires with the guard OFF *and*
   ON (the `HALO off/on` column of the reach map reads `1/1` on both, against
   `1/0` at (0,0)). This is what closes S7.5 of the energy audit.

---

## 3. The six-bound table, guard OFF vs guard ON

`energy_stage_audit_121.py`, `RN=1024`, six post-DOE groups,
`final_distance=0`, `final_leg='paraxial'` - the diagnostic configuration, so
nothing but the relay is scored. `ship` = C5+C6 as shipped, guard OFF;
`shipG` = the same with `REMAP_STATIONARY_PHASE_FIT_GUARD = True`; `noC6` = the
C6-off reference, which is what C1b's deficit floor is measured against.

Bounds as proposed in `ENERGY_CONSERVATION_AUDIT_2026_07_31.md` S6:
**C1a** `P_out/P_ap` in [0.9900, 1.00020] for every group -
**C1b** the last group's deficit >= 0.5x the same order's C6-off deficit -
**C2** end to end in [0.9850, 1.00050] -
**C3** `g4 <= 3 x g4_exact(order)` -
**C4** `amax4 <= 1.0e-03` -
**C5** `|r_rms - r_rms_exact| / r_rms_exact <= 0.030`.

### 3.1 `ray_subsample = 4`

| order | cfg | `elem(5)` | deficit / floor | end to end | `g4` | `g4` / bound | `amax4` | `r_rms` | dev | **score** |
|---|---|---|---|---|---|---|---|---|---|---|
| (0,0) | ship | 0.999371 | **0.15x** | 0.997750 | **3.400e-03** | **3.2e6** | **7.70e-01** | 0.9349 | **+11.20 %** | **2/6** |
| (0,0) | **shipG** | 0.995976 | **0.98x** | 0.994360 | **0.000e+00** | **0.00** | **0.00e+00** | 0.8384 | **-0.27 %** | **6/6** |
| (0,0) | noC6 | 0.995901 | 1.00x | 0.994281 | 3.606e-11 | 0.03 | 1.40e-05 | 0.8422 | +0.18 % | 6/6 |
| (-1,0) | ship | 0.996017 | 1.00x | 0.994074 | 3.318e-10 | 0.01 | 8.24e-05 | 0.8384 | -0.34 % | 6/6 |
| (-1,0) | shipG | 0.996017 | 1.00x | 0.994074 | 3.304e-10 | 0.01 | 8.21e-05 | 0.8384 | -0.34 % | 6/6 |
| (-1,0) | noC6 | 0.996006 | 1.00x | 0.994060 | 1.425e-09 | 0.05 | 9.84e-05 | 0.8429 | +0.19 % | 6/6 |
| (-2,0) | ship | 0.996043 | 1.00x | 0.994131 | **2.270e-07** | **2.49** | **5.73e-03** | 0.8382 | -0.53 % | **4/6** |
| (-2,0) | **shipG** | 0.996043 | 1.00x | 0.994131 | **2.274e-07** | **2.49** | **5.75e-03** | 0.8382 | -0.53 % | **4/6** |
| (-2,0) | noC6 | 0.996057 | 1.00x | 0.994142 | 7.750e-09 | 0.08 | 1.60e-04 | 0.8443 | +0.19 % | 6/6 |
| (-3,0) | ship | 0.995917 | 1.01x | 0.994064 | **2.234e-07** | **1.26** | **5.87e-03** | 0.8380 | -0.83 % | **4/6** |
| (-3,0) | **shipG** | 0.995917 | 1.01x | 0.994064 | 1.633e-07 | 0.92 | **5.68e-03** | 0.8379 | -0.84 % | **5/6** |
| (-3,0) | noC6 | 0.995977 | 1.00x | 0.994121 | 1.918e-08 | 0.11 | 2.06e-04 | 0.8467 | +0.20 % | 6/6 |
| (-4,0) | ship | 0.995906 | 1.05x | 0.993992 | 2.628e-08 | 0.12 | 2.37e-04 | 0.8376 | -1.26 % | 6/6 |
| (-4,0) | **shipG** | *byte-identical to `ship`* | | | | | | | | **6/6** |
| (-4,0) | noC6 | 0.996084 | 1.00x | 0.994169 | 2.831e-08 | 0.12 | 2.00e-04 | 0.8500 | +0.20 % | 6/6 |
| (-4,-2) | ship | 0.996036 | 1.04x | 0.993816 | 2.653e-08 | 0.12 | 2.34e-04 | 0.8375 | -1.51 % | 6/6 |
| (-4,-2) | **shipG** | *byte-identical to `ship`* | | | | | | | | **6/6** |
| (-4,-2) | noC6 | 0.996185 | 1.00x | 0.993966 | 2.507e-08 | 0.11 | 2.26e-04 | 0.8519 | +0.19 % | 6/6 |

Per-bound, guard OFF -> ON:

| bound | (0,0) | (-1,0) | (-2,0) | (-3,0) | (-4,0) | (-4,-2) |
|---|---|---|---|---|---|---|
| C1a per-element | pass -> pass | pass -> pass | pass -> pass | pass -> pass | pass -> pass | pass -> pass |
| C1b deficit floor | **FAIL -> pass** | pass -> pass | pass -> pass | pass -> pass | pass -> pass | pass -> pass |
| C2 end to end | pass -> pass | pass -> pass | pass -> pass | pass -> pass | pass -> pass | pass -> pass |
| C3 halo fraction | **FAIL -> pass** | pass -> pass | **FAIL -> FAIL** | **FAIL -> pass** | pass -> pass | pass -> pass |
| C4 halo amplitude | **FAIL -> pass** | pass -> pass | **FAIL -> FAIL** | **FAIL -> FAIL** | pass -> pass | pass -> pass |
| C5 second moment | **FAIL -> pass** | pass -> pass | pass -> pass | pass -> pass | marginal | marginal |

### 3.2 `ray_subsample = 2`

The finer subsample is where the on-axis defect stops being masked: the
surrounding discretisation deficit falls ~5x while C6's on-axis gain grows, so
the guard-OFF chain ends **above unity**.

| order | cfg | `elem(5)` | deficit / floor | end to end | `g4` | `g4` / bound | `amax4` | `r_rms` | dev | **score** |
|---|---|---|---|---|---|---|---|---|---|---|
| (0,0) | ship | **1.003696** | **negative** | **1.003186** | **4.495e-03** | **4.2e6** | **9.78e-01** | 0.9477 | **+12.73 %** | **0/6** |
| (0,0) | **shipG** | 0.999199 | **0.99x** | 0.998692 | **0.000e+00** | **0.00** | **0.00e+00** | 0.8363 | **-0.52 %** | **6/6** |
| (0,0) | noC6 | 0.999189 | 1.00x | 0.998680 | 4.559e-11 | 0.04 | 1.47e-05 | 0.8403 | -0.05 % | 6/6 |
| (-1,0) | ship | 0.999174 | 1.01x | 0.998652 | 3.952e-10 | 0.01 | 9.71e-05 | 0.8362 | -0.61 % | 6/6 |
| (-1,0) | shipG | 0.999174 | 1.01x | 0.998652 | 3.957e-10 | 0.01 | 9.73e-05 | 0.8362 | -0.61 % | 6/6 |
| (-1,0) | noC6 | 0.999185 | 1.00x | 0.998661 | 1.559e-09 | 0.05 | 9.42e-05 | 0.8408 | -0.06 % | 6/6 |
| (-2,0) | ship | 0.999195 | 0.99x | 0.998681 | **7.076e-06** | **77** | **5.02e-02** | 0.8365 | -0.74 % | **4/6** |
| (-2,0) | **shipG** | 0.999191 | 1.00x | 0.998677 | **2.790e-06** | **31** | **3.23e-02** | 0.8362 | -0.77 % | **4/6** |
| (-2,0) | noC6 | 0.999190 | 1.00x | 0.998675 | 8.251e-09 | 0.09 | 1.60e-04 | 0.8422 | -0.06 % | 6/6 |
| (-3,0) | ship | 0.999181 | 1.01x | 0.998669 | **5.694e-07** | **3.20** | **1.39e-02** | 0.8358 | -1.09 % | **4/6** |
| (-3,0) | **shipG** | 0.999181 | 1.01x | 0.998669 | **6.153e-07** | **3.46** | **9.92e-03** | 0.8358 | -1.09 % | **4/6** |
| (-3,0) | noC6 | 0.999192 | 1.00x | 0.998680 | 2.041e-08 | 0.11 | 2.08e-04 | 0.8445 | -0.06 % | 6/6 |
| (-4,0) | ship | 0.999175 | 1.03x | 0.998654 | 2.794e-08 | 0.12 | 2.51e-04 | 0.8354 | -1.52 % | 6/6 |
| (-4,0) | **shipG** | *byte-identical to `ship`* | | | | | | | | **6/6** |
| (-4,0) | noC6 | 0.999198 | 1.00x | 0.998676 | 2.986e-08 | 0.13 | 2.73e-04 | 0.8478 | -0.06 % | 6/6 |
| (-4,-2) | ship | 0.999164 | 1.04x | 0.998628 | 2.739e-08 | 0.12 | 2.32e-04 | 0.8352 | -1.78 % | 6/6 |
| (-4,-2) | **shipG** | *byte-identical to `ship`* | | | | | | | | **6/6** |
| (-4,-2) | noC6 | 0.999196 | 1.00x | 0.998661 | **6.322e-06** | **28** | **7.84e-02** | 0.8498 | -0.06 % | **4/6** |

The last row is the energy audit's reversal and it survives: at `rs=2` on
(-4,-2) it is **C6-OFF** that violates C3 and C4, and C6-on that is clean. The
guard does not change that either way, because it never touches that order.

### 3.3 The answer to the brief's question

> *does the guard make (0,0) and (-2,0) pass without costing the tilted
> orders?*

**(0,0): yes, completely, at both subsamples.** 2/6 -> 6/6 at `rs=4` and 0/6 ->
6/6 at `rs=2` (and 1/6 -> 6/6 in the PRODUCTION configuration, where
`elem(last)` = 1.000741 breaks C1a as well - S4.1). Every bound the on-axis
order failed is repaired, and the two that are hardest to satisfy - the halo
pair - go to *exactly zero*, not merely under bound.

**(-2,0): no, and it never could.** 4/6 -> 4/6 at both subsamples, failing C3
and C4 identically. Its last group is at 0.481 w, on the weighted branch
already; the guard's only reach at that order is group 0. The `g4` numbers move
(2.270e-07 -> 2.274e-07 at `rs=4`, 7.076e-06 -> 2.790e-06 at `rs=2`) but that
is the quantity's own ill-conditioning, not a repair - the energy audit
measured the same number spreading 4.9x under ~1e-06-level input perturbations.
**(-3,0) behaves the same way** and is added here for exactly that reason: it
confirms (-2,0) is not a one-order accident but the signature of the
off-centre-branch lobe.

**The tilted orders: no conservation cost anywhere.** (-4,0) and (-4,-2) are
byte-identical at both subsamples. (-1,0), (-2,0) and (-3,0) score identically
on all six bounds with one exception, and that exception is an *improvement*
((-3,0) C3 at `rs=4`, `g4` 2.234e-07 -> 1.633e-07, crossing the bound from
1.26x over to 0.92x under). No bound is worsened on any order.

---

## 4. The EE half of the trade

`energy_ee_vs_conservation_121.py`, the **production configuration** - the same
`approx_common.run_chain` the campaign's EE numbers came from: six post-DOE
groups, the 7.7058 mm trailing leg, `final_leg='exact'`, exact Bluestein
readout, `NOUT=192`, `dx_out=0.1 um`, `n_fine_cap=12288`, `window_factor=4.0`,
fixed lattice on the order's chief ray. Both currencies come from the same run.

Differential floor first: **two identical baseline runs give
`array_equal = True`, `max|dE| = 0.000e+00`** on the library that carries the
halo check.

### 4.1 Order (0,0) - the order the guard is for

| config | EE3 % | EE6 % | EE12 % | `P_tile` % | `elem(last)` | `g4` | `amax4` | `r_rms` mm | dev vs exact ray |
|---|---|---|---|---|---|---|---|---|---|
| **ship** (guard OFF) | **88.400** | 98.646 | 98.803 | 98.8046 | **1.000741** | **4.715e-03** | **8.32e-01** | 0.9644 | **+14.71 %** |
| **shipG** (guard ON) | **87.436** | **98.056** | 98.798 | 98.7988 | **0.995984** | **0.000e+00** | **0.00e+00** | 0.8381 | **-0.31 %** |
| noC6 | 86.709 | 98.637 | 98.766 | 98.7691 | 0.995883 | 0.000e+00 | 0.00e+00 | 0.8419 | +0.14 % |

`ship` and `noC6` reproduce the energy audit's S4.1 to every digit, which is
the cross-check that this is the same instrument on the same path.

**The guard's EE cost, and it is not one-sided:**

| | vs `ship` (guard OFF) | vs `noC6` (C6 off) |
|---|---|---|
| EE3 | **-0.964 pts** | **+0.727 pts** |
| EE6 | **-0.590 pts** | **-0.581 pts** |
| EE12 | -0.005 | +0.032 |
| `P_tile` | -0.006 | +0.030 |
| `elem(last)` | **1.000741 -> 0.995984** | 0.995883 -> 0.995984 |
| `g4` | **4.715e-03 -> 0** | 0 -> 0 |
| `amax4` | **0.832 -> 0** | 0 -> 0 |
| `r_rms` dev | **+14.71 % -> -0.31 %** | +0.14 % -> -0.31 % |

Read plainly: **the guard keeps 43 % of C6's on-axis EE3 gain (+0.727 of
+1.691) and removes 100 % of the manufactured energy** - `elem(last)` comes
back under unity and onto the C6-off value to five decimals, the halo goes to
exactly zero, and the second moment lands on the exact-ray reference. That is
a good trade on EE3.

**But it costs 0.59 EE6 points against BOTH other configurations** (98.056 vs
98.646 and 98.637), while EE12 and `P_tile` are unchanged to within 0.03. So it
moves ~0.6 % of the power from inside 6 um to between 6 and 12 um: a real
spot-quality cost at the 6 um radius that neither `ship` nor `noC6` pays.
**On design 121's own on-axis order the guard is therefore not an unambiguous
EE improvement over C6-off either** - better on EE3, worse on EE6 - and that,
independently of the synthetic counter-example, is why this is not a default
flip.

### 4.2 Order (-2,0) - a tilted order the guard reaches (group 0 only)

| config | EE3 % | EE6 % | EE12 % | `P_tile` % | `elem(last)` | `g4` | `amax4` | `r_rms` mm |
|---|---|---|---|---|---|---|---|---|
| ship | 87.824 | 98.096 | 98.760 | 98.7610 | 0.995730 | 9.998e-07 | **1.63e-02** | 0.8380 |
| **shipG** | **87.824** | **98.096** | **98.760** | **98.7610** | **0.995730** | 8.851e-07 | **1.48e-02** | 0.8380 |

**The EE cost on the tilted order is zero to every printed digit** - EE3, EE6,
EE12, `P_tile`, `elem(last)` and `r_rms` are identical - even though the fields
are not byte-identical (the guard moved group 0). Only the halo moves, and it
moves within its own ill-conditioning (1.63e-02 -> 1.48e-02 of peak at 4 mm,
both an order of magnitude over the C4 bound).

This is the measurement the audit called for and it is unambiguous: **the guard
costs the tilted orders nothing measurable in either currency.** The entire
case against flipping it rests on (a) the 0.59 EE6 points it costs on axis and
(b) the synthetic regression, not on any tilted-order damage.

---

## 5. What shipped, and one thing that deliberately did not

### 5.1 The energy self-check's GAIN side: measured for tightening, LEFT AT 0.050

The brief asked for the `ray_density` energy self-check to be tightened or
given the halo term the energy audit shows is necessary. The tightening was
attempted first, because it is the cheaper change, and **it must not be made**.

The band's documented calibration is the P2 design battery at **N = 512**,
where the largest ratio anywhere is 1.00003 - which makes the +0.050 gain
tolerance look like ~1600x of unused headroom. Re-measured on the SAME battery
at the **N = 1024 and `ray_subsample = 4` that `tests/unit/
test_niche_p2_design_battery.py` actually runs**, driving the tolerances
negative so every call prints its own ratio:

| battery cell | grp0 | grp1 | grp2 |
|---|---|---|---|
| triplet, w0 1.6 mm, aperture:beam **2.5x**, rs=8 | 0.99731 | **1.38446** | **1.30097** |
| triplet, w0 1.6 mm, aperture:beam **2.5x**, **rs=4 (CI)** | 0.99933 | 0.94477 | **1.04374** |
| triplet, w0 1.6 mm, aperture:beam 2.5x, rs=2 | 0.99984 | 1.00069 | 1.00793 |
| triplet, w0 1.6 mm, aperture:beam 2.5x, rs=1 | 0.99996 | 0.99996 | 0.99998 |
| triplet, w0 1.6 mm, aperture:beam 1.2x, rs=4 | 0.99837 | 0.98578 | 1.00095 |
| relay, w0 2 mm, aperture:beam 2.5x, rs=8 | 0.99724 | **1.16995** | - |
| every other cell, rs=4 | 0.99804 .. 0.99933 | 0.94477 .. 0.99669 | - |

So a **currently-green CI cell reads 1.04374**, and any gain tolerance below
0.044 would warn on it. That reading is a real defect - it converges to 1.00000
as `ray_subsample -> 1`, and the library's fold-caustic warning already fires
on every group of that design - but it is **the same magnitude as the defects
the energy audit wanted the band to catch** (its S5.3 rows at 1.03317 and
1.04593). The two populations overlap on this observable. The honest conclusion
is not "tighten it" but "**this observable is exhausted**", and tightening it
would have been exactly the failure this project has committed before: a guard
that refuses its own shipped configuration.

Nor would tightening have caught what prompted the review: the shipped on-axis
production call reads **1.000741** while carrying a lobe at 83 % of peak, and
the energy audit's S2.4 shows the same defect reading 1.001058 before a library
edit and 0.999371 after, with the lobe unchanged.

`_RD_ENERGY_GAIN_TOL` stays **0.050**; `_RD_ENERGY_DEFICIT_BASE` stays 0.080
and `_RD_ENERGY_DEFICIT_PER_SUB` 0.010 (the deficit side's P2 calibration is
sound and the measured worst cell at rs=4 is 0.94477 against a 0.890 floor).
The measurement and the reasoning are now in the source above those constants,
so the next person to look at them does not repeat the attempt.

### 5.2 SHIPPED: the halo-amplitude term

`lumenairy/elements/_lens_traced.py`, ~55 executable lines in two places, plus
four constants and a policy knob:

- `_RD_HALO_AMP_CONTOUR = 9.0` - the `e^-9` amplitude contour (`r = 3w` for a
  Gaussian, interior holding 1 - 1.5e-08 of the beam power);
- `_RD_HALO_RADIUS_FACTOR = 1.25`;
- `_RD_HALO_AMAX_TOL = 1.0e-03`;
- `RAY_DENSITY_HALO_CHECK = 'warn'` (`'warn'` | `'silent'`; never an error).

**Unchanged:** `_RD_ENERGY_DEFICIT_BASE`, `_RD_ENERGY_DEFICIT_PER_SUB`,
`_RD_ENERGY_GAIN_TOL` (S5.1) and every other constant; no signature moved, no
default flipped, no public entry point added.

**What it measures.** Immediately after the launch lattice is traced - *before*
any fit-domain restriction, so it reads the optics and not the model - the
exact exit positions of every alive launch ray carrying input amplitude above
the contour are reduced to an amplitude-weighted exit centroid and a support
radius `r_hull`. Then, on the returned field,
`amax_halo` = max `|E_out|` beyond `1.25 x r_hull` of that centroid, over the
peak. Both the hull and the halo are referenced to the same traced centroid, so
the statistic does not care where the beam sits on the grid.

**Why amplitude and not power.** A power fraction is only as sensitive as the
ghost is large. The lobe this catches on design 121 carries 0.34 % of the power
and stands at **77 % of the peak**. `g_halo` is reported in the message for
context and is deliberately not part of the bar.

**It declines rather than guess when the annulus is not a genuine annulus.**
The bound circle must fit inside the grid about the traced exit centroid;
otherwise all that survives of the annulus is a sliver of grid corners, and the
statistic measured there is unreliable **in both directions** - measured twice,
in S5.2.1 below. This is one line and it costs no calibrated defect.

**Calibration - 180 element calls (177 readings), two populations, one pass.**
`validation/repro_traced_carrier_121/halo_calibration.py` forces the tolerance
negative so every call prints its own reading (including the support radius and
centroid), captures the field alongside, and re-scores at every radius factor
offline - so the factor sweep costs nothing and no library constant is swept.

*Clean* = the CI-safe P2 battery (four designs x two beam sizes x two
aperture:beam ratios, every group, at `ray_subsample` 1, 4 and 8 - 54 calls, 51
readings, 3 declined), the synthetic C6 fixtures on their clean branches (18
calls), and every design-121 call under its own exact-ray ceiling (108 calls,
none declined). *Defective* = lobes confirmed manufactured against an exact ray
trace.

| radius factor | worst CLEAN | mildest DEFECT | separation |
|---|---|---|---|
| 1.00 | 2.270e-04 | 5.727e-03 | 25x |
| 1.10 | 1.046e-04 | 5.727e-03 | 55x |
| **1.25** | **4.622e-05** | **5.684e-03** | **123x** |
| 1.50 | 1.246e-05 | 5.684e-03 | 456x |
| 2.00 | ~1e-16 | **MISSED** (1.4e-04 / 0.0) | - |

The clean column is the ray-density field's bilinear-upsample spill past the
last traced ray; it dies by 1.25. Beyond 1.50 the bound starts stepping over
real defects - at 2.00 both the (-2,0)-class lobe and the synthetic one vanish
entirely. **1.25 is the smallest factor that clears the spill by more than an
order of magnitude while still reading every measured defect at full
amplitude**, and `1.0e-03` then sits inside a gap of 123x, at 21.6x above the
worst clean reading and 5.7x below the mildest real defect.

**What it fires on, stated so nobody is surprised.**

| configuration | fires? | true positive? |
|---|---|---|
| every P2 battery cell, every group, rs 1/4/8 | **no** (worst reading 1.6e-05) | - |
| design 121 relay, all six orders, `noC6` | **no** | - |
| design 121 relay, (0,0) last group, shipped (guard OFF) | **YES** (0.770) | yes - 9.5e6x its exact-ray ceiling |
| design 121 relay, (0,0) last group, guard ON | **no** (1.5e-05) | - |
| design 121 relay, (-2,0) / (-3,0) last group, guard OFF or ON | **YES** (5.7e-03 / 5.9e-03) | yes - 7.5x / 3.8x their ceilings, and exactly zero with C6 off |
| synthetic 'design-121-scale stand-in', guard ON | **YES** (0.881) | yes - `P/Pin` 1.00697 |
| synthetic 'medium, finer grid', guard ON | **YES** (4.6e-02) | yes |
| the same two synthetic fixtures, guard OFF | **no** | - |
| **niche D6's exact-tilted-leg RETRACE** | **YES** (0.641) | **yes, and NEW - see below** |

So the check fires on 3 of design 121's 36 relay element calls and on none of
the battery's 54. Every firing is a lobe independently confirmed against an
exact ray trace, and **the energy self-check is silent on every one of them** -
which is the entire argument for the term existing.

**It found something on its first run through the suite.** Two tests in
`tests/unit/test_niche_d6_exact_tilted_leg.py` -
`test_the_tilted_exact_leg_conserves_power_like_the_paraxial_one` and
`test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle` - make
the exact-leg **retrace** call warn: `amax_halo` = **6.405e-01 of peak beyond
2.0202 mm** against an exact-ray support of **1.6161 mm**, `g_halo` =
**6.449e-04**, on a grid reaching **2.4341 mm** from the centroid, i.e. a FULL
annulus and not a corner sliver. That is a previously unreported manufactured
lobe in a green CI fixture. Three things about it:

- **the library already half-knew.** The fold-caustic warning fires on the same
  call, so the two diagnostics agree on the mechanism (`det J -> 0` inflating
  the capped ray-density amplitude); what the halo term adds is *where* it
  lands and *how much* of the peak it is;
- **the D6 tests still pass** (38 passed). They assert spot FWHM/EE against an
  independent oracle and the stage power - all of which the lobe is outside.
  That is the same blindness the energy audit found in EE3/EE6/`P_tile`, on a
  different fixture;
- **it is recorded, not silenced.** No test was changed to hide it and the
  warning is left in place. It belongs to D6's owner, and it is in S7.

### 5.2.1 The limitation, measured rather than assumed

The production run
reported ZERO halo warnings on the `ship` (0,0) field whose halo at 4 mm is
8.32e-01 of peak - which looked like a plumbing failure and is not.
`halo_production_probe.py` shows why: in the production configuration the last
group is re-run on the `n_fine_cap = 12288` **fine grid**, whose half-width is
*narrower than the group's own exit fan*:

| order | grid half-width | `r_hull` | `1.25 r_hull` | annulus | reading |
|---|---|---|---|---|---|
| (0,0), production readout leg | 6.277 mm | 7.136 mm | 8.920 mm | **none** | - |
| (-2,0), production readout leg | 7.771 mm | 7.827 mm | 9.784 mm | corners only | 4.5e-04 (OFF) / 1.4e-03 (ON) |
| (0,0), relay exit | 17.0 mm | 2.993 mm | 3.741 mm | full | **0.770** |
| (-2,0), relay exit | 17.0 mm | 4.817 mm | 6.021 mm | full | **5.7e-03** |

So **the check covers the relay, not the readout leg** - and in the
corners-only regime it was unreliable in both directions: at (-2,0) it warned
with the guard ON and was silent with it OFF on two fields that are *both*
defective at 4 mm (1.63e-02 / 1.48e-02 of peak).

**That is why the shipped check declines there.** It requires the bound circle
to fit inside the grid about the traced exit centroid; both readout-leg rows
above now report nothing rather than a number that cannot be trusted either
way. The condition was verified not to cost a single calibrated defect - the
full calibration was re-run under it and every defective row still reports at
full amplitude (clean max 4.622e-05, mildest defect 5.684e-03, separation
still 123x); the only rows it removes are **3 of the 54 P2 battery calls**,
all of which were reading clean anyway. A second measured case is niche D6's
*coarse* call (a 3.4 mm aperture on a 3.6 mm grid, `r_hull` 1.619 mm against a
1.239 mm centroid-to-edge distance), whose corners-only reading was 0.841 of
peak at the corner diagonally opposite the beam - now declined, while D6's
*retrace* call, which has a genuine full annulus, still warns.

Declining when the support does not fit the grid is the correct behaviour - a
grid comparable to its own exit fan cannot support a halo statement - and both
halves of it (no annulus, and corners-only) are pinned by tests. The
alternative, tightening the amplitude contour until a hull always fits, buys
that coverage with false positives on every legitimate skirt. Recorded in the
constant's SCOPE (d) and in S7.

**It is a diagnostic and it moves nothing.** Proved two ways: `'warn'` vs
`'silent'` is `array_equal`, and so is a run that fires against one that does
not (tolerance-driven, so the two runs differ in nothing else). At library
level, `probe_c6_byte_identity.py` reproduces the committed `HEAD` bit for bit
on all 26 cases (S6.1).

### 5.3 Documentation changes, no behaviour

- `REMAP_STATIONARY_PHASE_FIT_GUARD`'s note now carries the reach map, the
  (-2,0)/(-3,0) non-fix, the (0,0) repair table at both subsamples, and the
  re-measured synthetic counter-example - and states plainly that the default
  is confirmed `False` on a chain measurement rather than an element-level
  inference.
- `tests/unit/test_niche_c6_fit_guard.py`'s module docstring had claimed the
  decentre inertness covers "every tilted order". Corrected, with the reach map
  and the reason (the gate is per group). No test changed: the element-level
  pin was always a statement about a decentred *beam* and is unaffected.

### 5.4 Tests added

`tests/unit/test_niche_c7_ray_density_halo_check.py`, 15 tests, ~7 s, no
proprietary asset:

1. the policy defaults to `'warn'`, and the constants sit inside the measured
   gap (so a future edit has to come back through the calibration);
2. **field neutrality both ways** - `'warn'` vs `'silent'`, and a call that
   fires vs one that does not;
3. no false positive on a clean call (C6 launch on and off), on a **decentred**
   beam (the case a grid-referenced radius would get wrong), or in
   `amplitude_model='screen'`;
4. **fail-before**: it fires on the `REMAP_STATIONARY_PHASE_FIT_GUARD`
   regression fixture and is silent with the guard off - *and* the energy band
   is quiet in both directions, which is the pin that this is not a
   re-statement of the power check;
5. the radius really tracks the traced ray support (drive the factor below 1
   and real light falls in the annulus);
6. **both halves of the decline condition** - no annulus at all (factor 50),
   and the harder one, corners-only (a factor whose bound is outside the
   grid's half-width but inside its half-diagonal, asserted to be so, so the
   test cannot pass for the trivial reason);
7. the message carries radius, amplitude, power, the grid's own reach and the
   suppression knob; the category is `RuntimeWarning` and it never raises.

---

## 6. Re-established contracts

### 6.1 Byte-identity and the fail-before switches

Both re-run on the FINAL library (`_lens_traced.py` `f789409298b3acc6`), after
every change described above:

| probe | arms | result |
|---|---|---|
| `probe_c6_byte_identity.py` | (a) 12 synthetic cases, all `preserve_input_phase` x `amplitude_model` combinations + `'remap'` + lattice + no-carrier, at `rs` 1 and 4; (b) 7 design-121 chain cases at order (0,0) (two grids, two subsamples, 3- and 5-group runs, both readout paths, `final_leg='exact'`); (c) the same 7 at (-4,-2); (d) the C5 contract with C6 shipped | **OK - `array_equal=True`, `max\|dE\| = 0.000e+00` on all 29 comparisons** |
| `probe_c6_tilted_failbefore.py` | orders (-4,-2), (-4,0), (-1,0), (0,0), C6-OFF vs the committed library | **OK - `array_equal=True`, `max\|dE\| = 0.000e+00` on all four** |

Because the default was **not** flipped, the untilted path's byte-identity is
not merely preserved by argument - it is the measured result above, on the
library that now carries the halo check. The halo term is gated on
`amplitude_model='ray_density'` and emits a warning only; the fit guard is
gated on `_resid_eik is not None`, which requires `REMAP_STATIONARY_PHASE_
LAUNCH`, so both probes' C6-OFF arms exercise the guard's inert path as well.

**This is also the proof that the halo term is a diagnostic and not a filter**
at the level that matters: the reference is not a re-run of the same code but
`git show HEAD:lumenairy/elements/_lens_traced.py` imported as a shadow module,
so a single changed bit anywhere in the returned field would show. The unit
tests pin the same property two more ways (`'warn'` vs `'silent'`, and a call
that fires vs one that does not).

The single-beam acceptance in S6.2 and the S3 conservation tables were taken on
`1d8e7f34182eb888`, before the decline condition and the message change; the
byte-identity above is what carries them to the final library.

### 6.2 Single-beam acceptance

`focus_scan_121.py`, pure library defaults (`CREF`/`AM`/`PIP` unset), `N=2048`,
`rs=4`, `NFC=8192`, `WF=4.0`, `NOUT=2048`. The line to read is
`BEST-FOCUS[peak]`, because EE6 saturates at ~99.7 % and mis-selects.

| | before | **after** |
|---|---|---|
| `BEST-FOCUS[peak]` plane | dz = +0 um (nominal) | **dz = +0 um (nominal)** |
| FWHM | 3.450 um | **3.450 um** |
| EE3 | 90.2 % | **90.2 %** |
| EE6 | 99.7 % | **99.7 %** |
| EE12 | - | 99.8 % |
| peak | - | 5.473e+03 |

**No change, on any digit reported.** Expected: the default was not flipped and
the halo term is proved field-neutral (S6.1). The `BEST-FOCUS` (EE6-selected)
line still mis-selects to dz = +10 um / FWHM 3.650 / EE3 85.3, exactly as
documented - that is the reason the `[peak]` line is the one quoted.

### 6.3 Niche suites and lint

```bash
python -m pytest tests/unit/test_niche_c1_*.py tests/unit/test_niche_c3_*.py     tests/unit/test_niche_c5_*.py tests/unit/test_niche_c6_*.py     tests/unit/test_niche_c7_*.py tests/unit/test_niche_d1_*.py     tests/unit/test_niche_d2_*.py tests/unit/test_niche_d3_*.py     tests/unit/test_niche_d6_*.py tests/unit/test_niche_d7_*.py -q
-> 316 passed, 73 warnings in 828.18s        (the FINAL library)

python -m ruff check lumenairy/ tests/unit/
-> All checks passed!
```

**316 passed, zero failures.** Across all 316 tests the suite emits exactly
**one** new halo warning - niche D6's exact-tilted-leg retrace (S5.2), a true
positive that is reported rather than silenced. The other 73 warnings are the
pre-existing physics diagnostics the suite is documented to emit.

An earlier revision of the check produced **three** halo warnings here; two
were corners-only readings, and they are why the shipped check declines when
the annulus is not a genuine annulus (S5.2.1, and artefact 9 in S8).

---

## 7. What remains unmeasured

1. **Where the halo goes.** Every halo figure is at an element's own exit
   plane. The co-moving grid is ~15x short of `lambda/(2 NA_exit)` and ~96.7 %
   of the exit power is above its Nyquist angle, so the image-plane fate of a
   manufactured lobe is not determined by these numbers. This does not weaken
   the conservation conclusions - manufactured energy is manufactured wherever
   it lands - but no image-plane r2m bound can be set on this grid.
2. **The (-2,0)/(-3,0) lobe has a mechanism only by elimination.** It is made
   on the off-centre weighted branch (the guard cannot reach it), it is C6's
   (exactly zero with the launch off), and it is above the exact-ray ceiling.
   Which of `_FIT_DISC_OUTSIDE_WEIGHT_REL`, the D7 order raise or the residual
   freeze admits it was not separated. The post-C6 audit's finding that every
   knob on that fit is a conditioning lottery with no monotone structure
   suggests a sweep would not settle it either; the structural cure named there
   - bounding the Newton inverse to the traced samples' own support - remains
   unattempted.
3. **The EE cost was priced on two orders**, (0,0) and (-2,0), not on all six.
   The production run is ~5 min and ~50 GB resident per configuration.
4. **`ray_subsample` was measured at 4 and 2 only.** The library's shipped
   default is 8; design 121 does not use it, and the guard's behaviour there is
   unmeasured on this design (the halo check was calibrated at 8 on the P2
   battery).
5. **The halo check's radius factor was calibrated on one class of defect** -
   the D1 spurious-root lobe. A defect that deposits energy *inside*
   `1.25 x r_hull` is invisible to it by construction, and C1b (the deficit
   floor, the criterion that catches the on-axis defect in every library state)
   is **not** implementable as a library self-check at all, because it needs a
   same-design C6-off reference run the element does not have.
6. **The check does not reach a call whose traced exit fan is wider than its
   own grid**, which on design 121 is the production readout leg (S5.2.1).
   What would extend it is a halo statement referenced to something other than
   the grid - the readout's own fine lattice has the resolution but not the
   extent - and that was not attempted. The corners-only regime between "full
   annulus" and "no annulus" was measured on three fields and found unreliable
   in both directions; the check now declines there and no bound is claimed.
7. **Whether the guard's -0.59 EE6 points on axis is a property of the guard or
   of design 121's on-axis branch** is not established. It is the same size
   against `ship` and against `noC6`, which suggests the weighted+order-10 fit
   itself, but no second design was priced in EE.
8. **Only the post-DOE relay is scored.** Chain A (source -> DOE) and
   `propagate_traced_carrier_chain_multi`'s readout tiling and recombination
   are untouched.
9. **The guard's behaviour on the P2 battery was not measured**, because the
   battery has no carrier residual to fit and C6 never engages there.
10. **The D6 exact-tilted-leg retrace lobe is reported, not explained.** It is
    a full-annulus reading of 0.641 of peak beyond 2.020 mm carrying 6.449e-04
    of the input power, with a fold-caustic warning on the same call, in a
    fixture two green tests depend on for spot metrics. Whether it is the
    fold, the retrace's widened axis-centred window, or the fit disc on that
    decentred beam was not separated, and neither was its effect on anything
    D6 asserts (nothing measurable: both tests pass with margin). It belongs
    to niche D6.

---

## 8. Artefacts found and killed in my own instruments

1. **The first version of the halo constants' calibration note was written
   before the calibration ran**, with plausible invented numbers ("the worst
   clean reading at factor 1.00 is 1.9e-03, i.e. the bound WOULD fire
   spuriously there"). The measurement then said 2.270e-04 and the argument
   reversed - factor 1.00 would have been *safe*, and the real reason to prefer
   1.25 is margin, not necessity. Caught by running the calibration before
   committing to the text. **Write the note after the measurement.**
2. **The gain-tolerance tightening was implemented, tested and reverted.**
   `_RD_ENERGY_GAIN_TOL = 0.005` was in the tree for about twenty minutes with
   a confident derivation attached ("167x the largest energy gain ever measured
   on a clean configuration"). The battery at its own N and subsample then read
   1.04374. The derivation had inherited the source comment's N = 512 envelope
   without checking that the CI fixture runs N = 1024.
3. **The chain-level "all six stages moved" was nearly reported as the guard's
   reach.** It is a cascade artefact: at (-2,0) the guard touches ONE group and
   the other five move because their input did. Settled by per-group replay
   (S2), which turned a wrong headline into the reach map.
4. **`halo_calibration.py` initially captured nothing on the synthetic
   fixtures.** `probe_ghost_synthetic.run` calls `la.apply_real_lens_traced`
   directly, so the `lumenairy.elements` monkeypatch never saw it and the
   script died on an empty list rather than silently reporting zeros - the
   right failure, but only by luck.
5. **The (-3,0) row of the exact-ray reference was checked for monotonicity
   against its neighbours before use** (`g4` ceiling 3.04e-08 -> **5.93e-08** ->
   7.56e-08, `r_rms` 0.8427 -> **0.8450** -> 0.8483, 3w hull 3.865 -> **3.931**
   -> 4.004 mm). A new reference row that did not interleave would have meant a
   bad cache or a mis-specified order.
6. **The `r_hull` / `_amp` pairing is transpose-sensitive** and was written
   against niche C4's note rather than by inspection: the launch grid is
   `indexing='ij'` so a ray's flat index is x-major, and pairing the amplitude
   with the transposed ray is invisible on a rotationally symmetric beam and
   wrong on every other. The decentred-beam test in S5.4 exists to catch that
   class.
7. **"The halo check did not fire in the production run" was nearly filed as a
   plumbing bug.** The `ship` (0,0) production row reported zero halo warnings
   on a field whose halo at 4 mm is 8.32e-01 of peak, and the check demonstrably
   fires on the same group in the relay configuration - which reads as a
   swallowed warning. It is not: the production readout leg re-runs that group
   on a 12288-point fine grid whose half-width (6.277 mm) is *narrower than the
   group's own exit fan* (7.136 mm), so no pixel lies outside the traced
   support and there is nothing to test. Settled by writing
   `halo_production_probe.py` to print the per-call grid, hull and centroid
   rather than by reasoning about the harness. **The instrument disagreed with
   itself and the grid was the reason** - and the limitation is now a documented
   SCOPE clause and a test, instead of a silent hole.
8. **`halo_calibration.py`'s design-121 arm measures the RELAY, not the shipped
   production path**, and the first draft of S5.2 said "design 121's shipped
   post-DOE chain warns on ...". Corrected to "relay" everywhere after item 7,
   because the two configurations do not give the same answer on the last
   group and conflating them would have overstated the check's coverage.
9. **The first version of the check shipped without the decline condition and
   immediately warned three times inside the green niche suite.** Two of those
   were corners-only readings (niche D6's coarse call at 0.841 of peak, and
   design 121's (-2,0) readout leg, where the *same* physical situation read
   silent with the fit guard off and warned with it on). Rather than accept
   them or tune the tolerance around them, the annulus was required to be a
   genuine annulus - and the whole calibration was re-run to prove that costs
   no defect (it removes 3 of 54 battery readings, all clean, and leaves the
   clean max, the defect min and the 123x separation unchanged). **A guard
   that fires in a regime where it has already been measured to be wrong in
   both directions is not a guard.**
10. **The THIRD firing survived that condition and is real.** It would have
    been convenient to treat all three D6/production firings as the same
    corners-only artefact and suppress them together. The retrace call has a
    full annulus (bound 2.020 mm inside a 2.434 mm reach) and its 0.641-of-peak
    lobe stands. It is reported as a finding, the test is not touched, and the
    warning is not silenced. The message now also prints how far the grid
    reaches from the centroid, so the next reader can tell the two regimes
    apart without re-deriving them.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`. Every runner prints
the sha256 of the library file it imported.

```bash
# S3.1 -- the six-bound table, guard OFF vs ON, ray_subsample 4
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CONFIGS='ship,shipG,noC6' \
    python energy_stage_audit_121.py

# S3.2 -- the same at ray_subsample 2 (chain A is cached at rs=2)
RS=2 NULL=0 ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CONFIGS='ship,shipG,noC6' \
    python energy_stage_audit_121.py

# S2 -- which groups the guard actually touches
ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python fitguard_branch_map.py

# S1.5 -- the exact-ray hull / halo ceiling / r2m reference for the new order
ORDERS='-3,0' NL=801 python energy_hull_121.py

# S5.2 -- the halo-check calibration, three populations
PART=synth python halo_calibration.py
RS=8 PART=batt python halo_calibration.py      # also RS=4, RS=1
PART=d121 ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' CFGS='ship,shipG,noC6' \
    python halo_calibration.py

# S0 -- the guard's own counter-example, re-measured on the current tree
python probe_ghost_synthetic.py

# S4 -- both currencies on ONE production field.  ~5 min and ~50 GB per row:
#       do NOT run concurrently with another chain batch.
LUMEN_PIN=0 NULL=1 ORDERS='0,0' CONFIGS='ship,shipG,noC6' \
    python energy_ee_vs_conservation_121.py
LUMEN_PIN=0 NULL=0 ORDERS='-2,0' CONFIGS='ship,shipG' \
    python energy_ee_vs_conservation_121.py

# S5.2 -- why the production readout leg reports nothing (per-call grid/hull)
LUMEN_PIN=0 ORDERS='0,0' CFGS='ship' FACTOR=0.05 python halo_production_probe.py

# S6 -- the contracts
python probe_c6_byte_identity.py
python probe_c6_tilted_failbefore.py
python focus_scan_121.py
python -m pytest tests/unit/test_niche_c[1356]_*.py \
    tests/unit/test_niche_c7_*.py tests/unit/test_niche_d[12367]_*.py -q
python -m ruff check lumenairy/ tests/unit/
```

### Files added by this study

`validation/repro_traced_carrier_121/fitguard_branch_map.py`,
`validation/repro_traced_carrier_121/halo_calibration.py`,
`validation/repro_traced_carrier_121/halo_production_probe.py`,
`tests/unit/test_niche_c7_ray_density_halo_check.py`, and this document.
`energy_stage_audit_121.py` and `energy_ee_vs_conservation_121.py` gained a
guard axis and identity/warning columns; nothing else about them changed.

Raw logs: `_fg_stage_rs4.txt`, `_fg_stage_rs2.txt`, `_fg_halo_batt.txt`,
`_fg_halo_d121.txt`, `_fg_ee_00.txt`, `_fg_ee_m20.txt`, `_fg_byteid.txt`.

### Library changes

`lumenairy/elements/_lens_traced.py` only: the halo self-check (behaviour,
warning-only, proved field-neutral) and documentation. **No default was
flipped.** `lumenairy/propagators/carrier.py`, `CHANGELOG.md` and
`lumenairy/elements/pmm/**` were not touched.
