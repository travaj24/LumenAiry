# Energy-conservation audit of the shipped traced-carrier path (design 121)

**Date** 2026-07-31 - **Scope** the working tree of
`fix/pmm-union-grid-conditioning` @ `d2e60ca` plus the uncommitted niche C6 and
post-C6 audit work, i.e. **C4 + C5 + C6 as they currently sit** -
**Subject** design 121's post-DOE chain, orders (0,0), (-1,0), (-2,0), (-4,0),
(-4,-2) - **Question** is this path fit to be a production ORACLE *on
conservation grounds*, and what acceptance bar should replace the EE-only one?

---

## 0. Headline

**The shipped configuration manufactures energy on the on-axis order, and no
metric the campaign used - EE3/EE6/EE12, `P_tile`, or the library's own
ray-density energy self-check - reports it.**

In the **production configuration** (the complete shipped path the campaign's
EE numbers come from: six post-DOE groups, the 7.7058 mm trailing leg,
`final_leg='exact'`, the Bluestein readout) design 121's last group at order
(0,0) returns `P_out / P_ap` = **1.000741**. Against the same chain with niche
C6 disabled (0.995883) the flag's on-axis cost is **+0.486 % of the last
group's input power**, and it deposits **4.715e-03 of the input power beyond
4 mm at 83 % of peak amplitude** - where the exact ray trace of the same launch
says at most **3.56e-10** can land, and where the C6-off field's halo is
**exactly zero**. The second moment moves from 0.8419 mm to 0.9644 mm
(**+14.7 %** against an exact-ray reference the C6-off field matches to
0.14 %); the *core* second moment moves **+0.05 %**, so the spot never moves
and encircled energy is blind by construction.

**Priced in both currencies on one field** (S4.1): C6 buys **+1.691 EE3
points** (86.709 -> 88.400) for that half-percent of manufactured energy, and
the rest of the EE family reported the transaction as EE6 **+0.009**, EE12
**+0.037**, `P_tile` **+0.036** points. **No member of the EE family - including
total power in the readout tile - could have caught it**: the tile is 19.2 um
across and the ghost is at 4-8 mm, 300x outside it.

Five supporting results:

1. **The physical reference is exactly 1.** The independent exact-ray +
   Rayleigh-Sommerfeld oracle reports **live power 100.0000 %** on every order
   (99.9999 % at (-4,-2)); the chain's aperture clipping never exceeds
   **1.13e-05**. There is no vignetting and no modelled Fresnel loss in this
   path, so **every deviation from 1.0000 is numerical**.
2. **The deficit is discretisation and converges; the gain is not and does
   not.** Halving `ray_subsample` shrinks the clean last-group deficit **5.1x**
   (4.10e-03 -> 8.11e-04) while C6's on-axis gain *grows* (3.47e-03 ->
   4.51e-03), so at `rs=2` the gain dominates and **the whole post-DOE chain
   ends above unity at 1.003186** (S2.3).
3. **The guard works and is still far too loose.** It fires correctly at
   `P/Pin` 1.085 and 1.488. But its +0.050 gain tolerance passed a field
   carrying **4.6e-02 of the input power as a lobe at 96 % of peak with the
   second moment at 2.2x**, and it passes the shipped on-axis defect with a
   factor of **67** to spare (S5). Total power is the wrong observable: it
   separates clean from ghosted by 5e-03, while the halo fraction separates
   them by **eight orders of magnitude**.
4. **C5 is load-bearing for conservation, not only for EE.** C5-off + C6-on
   gains **1.12 %** of the input power at (-4,0) and **0.92 %** at (-4,-2),
   with `g4` at 1.5e-02 / 1.3e-02 and the second moment at **+88 % / +77 %**.
   C5-off + C6-off is clean. The pathology is the *interaction*.
5. **The ghost's magnitude is chaotically sensitive; its presence is not.**
   Four independent ~1e-06-level perturbations of the last group's input
   (element replay vs chain; a mid-audit library edit; the diagnostic vs the
   production final leg) move on-axis `g4` over **1.03e-03 .. 5.09e-03**, and
   at (-2,0) the library edit alone moved it **202x** (4.58e-05 -> 2.27e-07).
   Over the very same perturbations the C6-*off* value never moved from
   3.606e-11 by a single bit.
   **A magnitude this ill-conditioned cannot be tuned or bounded; only a
   threshold on its presence is meaningful** - which is exactly what the
   proposed criterion is.

**Verdict on oracle fitness: NOT YET, on two of five orders.** The path is fit
as shipped on (-1,0), (-4,0) and (-4,-2). It is **not** fit on (0,0), and is
marginal at (-2,0), without `REMAP_STATIONARY_PHASE_FIT_GUARD = True` - which
already exists in the tree, acts only on the concentric (on-axis) fit branch,
is already pinned byte-identical on every tilted order, and is shipped
default-OFF. Its chain-level cost has still never been measured (S7.4).

**And the honest statement of the trade, unsmoothed: on the on-axis order C6
improves EE3 by 1.7 points and degrades conservation outright. On the tilted
orders it improves EE3 by 15.3 points at no conservation cost, and at
`ray_subsample=2` it is the only reason (-4,-2) stays inside the halo
criterion at all. The flag should not be reverted; it should be gated on the
branch that fails, which is what the existing fit guard does.**

---

## 1. Provenance, instruments, floors, sampling

### 1.1 What was actually measured - and the library moved mid-audit

Every runner prints the sha256 of the two files it imported.

| state | `elements/_lens_traced.py` | `propagators/carrier.py` | when |
|---|---|---|---|
| **L1** | `b745859f65f5b6c2` | `1a90453a4ef65399` | audit start, 00:20-00:50 |
| **L2** (current tree) | **`adcd1cee14dc7a0a`** | `1a90453a4ef65399` | 00:42:38 onward |
| post-C6 audit's pin, for reference | `f06da6ab8e15ce2a` | `2d30f1ed7beb3c7e` | 2026-07-30 |

**`_lens_traced.py` is owned by another agent and was edited at 00:42:38,
during this audit.** That is recorded rather than worked around, and the
consequences were measured rather than assumed. Re-running the entire cheap
matrix against L2 gives:

- **The production configuration is bit-identical between L1 and L2.** The
  S4.1 table - the complete shipped path with the trailing leg and the exact
  readout, which is where the headline lives - reproduces to every digit on
  both (EE3 88.400 / 86.709, `elem(last)` 1.000741 / 0.995883, `g4`
  4.715e-03 / 0.000e+00). **Nothing in S0 or S4.1 depends on which state you
  run.**
- **Every C6-OFF row is bit-identical between L1 and L2** - 10 of 10 rows,
  all six stages each. The edit touches only the C6 path.
- **The `rs=2` rows are bit-identical between L1 and L2.**
- What did move is the C6 rows of the *diagnostic* per-stage configuration
  (`final_distance=0`, `final_leg='paraxial'`, used in S2 because it isolates
  the relay from the readout). Quantified in S2.4, where it becomes a result
  in its own right.

**All primary tables below are L2, the current tree.** L1 appears only in S2.4
as a controlled comparison.

Flags as shipped: `TILTED_CARRIER_EXACT_EIKONAL = True` (C5),
`REMAP_STATIONARY_PHASE_LAUNCH = True` (C6),
`REMAP_STATIONARY_PHASE_FIT_GUARD = False`. All three, and the three guard
constants, are at identical values and line numbers in L1 and L2.

As a cross-check on the prior audit's claim that its own edits move no returned
array, replaying the last-group element call reproduces its S3.1 table exactly
on both L1 and L2 - C6 OFF: `P/Pin` 0.99590, `g4` 3.606e-11, `amax4` 1.40e-05,
`r_rms` 0.8422 mm; C6 deg4: 0.99701, 1.034e-03, 3.30e-01, 0.8638 mm.

### 1.2 Instruments

All new, all script-side, none edits the library; C5/C6 are module attributes
set inside `try/finally`, and the element capture is `wfe_probe_common`'s
existing monkeypatch extended only to census warnings **per call** so the
energy self-check can be attributed to a stage.

| instrument | what it measures | cost |
|---|---|---|
| `energy_recon.py` | library identity, group apertures, the guard's denominator vs the raw grid | 30 s |
| `energy_stage_audit_121.py` | the per-stage chain energy budget + halo, per order x {ship, noC6, noC5, noC56} | 8-40 s per run |
| `energy_guard_probe.py` | element-level conservation + halo + **whether the guard fired** | 2-4 s per case |
| `energy_hull_121.py` | the exact traced exit hull, the exact-ray halo ceiling and the exact-ray second moment | 40 s per order |
| `energy_sampling_check.py` | sampling adequacy of the fields the audit touches | 60 s |
| `energy_ee_vs_conservation_121.py` | both currencies on ONE field, complete shipped path | 4-8 min per run |
| `exact_ray_oracle_121.py` (existing, reused verbatim) | the conservation reference | 3 s per order |

### 1.3 Differential floors - established, bit-exact, before any delta

| instrument | null intervention | reading |
|---|---|---|
| `energy_stage_audit_121.py` | two identical shipped chain runs | **all six stages `array_equal = True`, `max\|dE\| = 0.000e+00`**, on all five orders |
| `energy_guard_probe.py` | two identical shipped element runs | `array_equal = True`, `max\|dE\| = 0.000e+00`, on all orders, both library states |
| `energy_ee_vs_conservation_121.py` | two identical baseline runs | `array_equal = True`, `max\|dE\| = 0.000e+00` |

Every delta quoted below is against a floor of exactly zero.

### 1.4 Sampling adequacy - stated as an amplitude-weighted p99.9, and it is SATURATED

Reported as the amplitude-weighted wrapped nearest-neighbour step at **p99.9**
against pi, never a max (SCOPE S8 artefact 5(a), DIAG artefact 4). The weight
of a step is the smaller of the two pixels' powers.

| field | p50 | p99 | **p99.9** | limit |
|---|---|---|---|---|
| last-group `E_in`, (0,0), brightest 86 % of power | 1.5940 | 3.1258 | **3.1276** | pi = 3.1416 |
| last-group `E_out`, (0,0), brightest 86 % | 1.5919 | 3.0852 | **3.0856** | pi |
| last-group `E_in`, (-4,-2), brightest 86 % | 1.5650 | 3.0973 | **3.1312** | pi |
| last-group `E_out`, (-4,-2), brightest 86 % | 1.5986 | 3.1318 | **3.1410** | pi |
| oracle integrand step between adjacent launch rays (amplitude-weighted) | - | - | **0.0261-0.0262 cycles** | 0.25 cycles |

**A median of 1.57 rad is exactly the median of a uniform distribution on
[-pi, pi].** These fields' raw phase is not marginal, it is fully unresolved,
and restricting to any amplitude contour from `1e-6 * pk` to `e^-1 * pk` does
not move the statistic - so this is a property of the field, not of the metric
(S8.2). It is also expected: the exact trace puts the last group's exit NA at
**0.288-0.298** against this grid's Nyquist NA of **0.0197** (`dx` = 33.211 um),
i.e. the co-moving grid is **14.6-15.1x short** of `lambda/(2 NA_exit)` and
**96.5-96.8 % of the exit power sits above the grid's Nyquist angle**. The
library's own guard says so on 4 of the 6 groups of every run.

**What this permits and forbids.** Every conservation number in this document
is a *power* measurement of a returned array - `sum |E|^2 dx^2`, and
`|E|^2`-weighted radial shells and moments - or an exact geometric ray trace.
None of them reads a phase. **No wave or WFE claim is made anywhere in this
audit.** The halo's radial position at the element exit plane is set by the
Newton inversion of the traced ray map, a geometric placement, not by a
propagation, so it is not aliased. What is *not* determined by these numbers is
**where the halo ends up after further propagation** (S7.1). Halo figures are
therefore exact for the returned array and are a statement about manufactured
energy, not a prediction of an image-plane r2m.

**One tautology is flagged rather than reported as evidence.** The free-space
legs between groups conserve to `|leg - 1| <= 1.1e-15` in all 40 chain runs.
That is not a result: the co-moving transport is FFT-based and therefore
exactly unitary on the grid. It means only that an aliased annulus is not
*lost*, it is relocated - which is precisely the library's warning.

### 1.5 The conservation reference

`exact_ray_oracle_121.py`, `ORD='0,0;-1,0;-4,0;-4,-2' NL=161 NOUT=61 DXO=0.2
CLIP=3.0`. Rays launched from the DOE plane on the exact carrier sphere plus
the order's grating tilt, traced through all six post-DOE groups with the
Zemax-validated skew trace, then a first Rayleigh-Sommerfeld integral with an
energy-conserving ray-density amplitude.

| order | **live power** | dead rays | exit NA | integrand step (amp-weighted) | EE3 % | EE6 % |
|---|---|---|---|---|---|---|
| (0,0) | **100.0000 %** | 13356 | 0.3596 | 0.0262 cyc | 89.88 | 99.97 |
| (-1,0) | **100.0000 %** | 13441 | 0.3616 | 0.0262 | 90.04 | 99.97 |
| (-4,0) | **100.0000 %** | 13726 | 0.3667 | 0.0261 | 90.45 | 99.97 |
| (-4,-2) | **99.9999 %** | 13777 | 0.3674 | 0.0261 | 90.14 | 99.96 |

(The "dead rays" are the corners of the launch square, at input amplitude
~`e^-59`; they carry no power, which is why live power is 1.0000 despite them.)

**The reference value for `P_out/P_in` is therefore 1.0000 exactly**, on every
order, with no legitimate loss channel to subtract. The chain's own aperture
clipping never exceeds **1.13e-05** across all 20 runs and all six groups.
Fresnel reflection is not modelled in this path at all. The oracle's EE3 is
also flat across the fan (89.88-90.45 %), i.e. the physics carries essentially
no field-angle penalty.

---

## 2. The per-order conservation table

`energy_stage_audit_121.py`, library **L2**, `RN=1024`, `ray_subsample=4`, six
post-DOE groups, `final_distance=0`, `final_leg='paraxial'` - the *diagnostic*
configuration, chosen so nothing but the relay is scored. (The *production*
configuration, with the trailing leg and the exact readout, is S4.1.) Powers
are `sum |E|^2 dx^2`, the library's own `_chain_envelope_stats` convention,
needed because the co-moving frame rescales `dx` per stage. Each run's
arithmetic is cross-checked against the chain's own `res.stages[k]['power']`
and agrees to all printed digits.

Definitions, so the denominator is never in doubt:

- `P_ap(k)` = the input power group *k* **admits** (inside its
  `aperture_diameter`). This is the denominator the library's own energy
  self-check uses, so `elem(k)` is exactly the quantity the guard polices.
- `clip(k)` = `1 - P_ap(k)/P_in(k)` - legitimate vignetting, reported
  separately and **never folded into the ratio**.
- `elem(k)` = `P_out(k)/P_ap(k)`; `leg(k)` = `P_in(k+1)/P_out(k)`.
- End to end factorises exactly as `prod elem * prod (1-clip) * prod leg`, and
  the script prints each factor (the check column matches the direct ratio to
  6 digits in all 20 runs).

### 2.1 Per-group element ratios `P_out/P_ap`

| order | config | grp 0 | grp 1 | grp 2 | grp 3 | grp 4 | **grp 5** | **end to end** |
|---|---|---|---|---|---|---|---|---|
| (0,0) | **ship** | 0.999846 | 0.999988 | 0.999584 | 0.999544 | 0.999414 | **0.999371** | **0.997750** |
| (0,0) | noC6 | 0.999846 | 0.999988 | 0.999584 | 0.999543 | 0.999412 | 0.995901 | 0.994281 |
| (0,0) | noC5 | 0.999846 | 0.999988 | 0.999584 | 0.999544 | 0.999414 | 0.999371 | 0.997750 |
| (0,0) | noC56 | 0.999846 | 0.999988 | 0.999584 | 0.999543 | 0.999412 | 0.995901 | 0.994281 |
| (-1,0) | **ship** | 0.999753 | 0.999736 | 0.999599 | 0.999546 | 0.999414 | 0.996017 | 0.994074 |
| (-1,0) | noC6 | 0.999753 | 0.999736 | 0.999599 | 0.999545 | 0.999411 | 0.996006 | 0.994060 |
| (-1,0) | noC5 | 0.999753 | 0.999736 | 0.999599 | 0.999546 | 0.999414 | 0.996011 | 0.994069 |
| (-1,0) | noC56 | 0.999753 | 0.999736 | 0.999599 | 0.999545 | 0.999412 | 0.996023 | 0.994077 |
| (-2,0) | **ship** | 0.999646 | 0.999864 | 0.999609 | 0.999544 | 0.999416 | 0.996043 | 0.994131 |
| (-2,0) | noC6 | 0.999646 | 0.999864 | 0.999609 | 0.999544 | 0.999414 | 0.996057 | 0.994142 |
| (-2,0) | noC5 | 0.999646 | 0.999864 | 0.999609 | 0.999544 | 0.999417 | 0.996618 | 0.994705 |
| (-2,0) | noC56 | 0.999646 | 0.999864 | 0.999609 | 0.999544 | 0.999414 | 0.996059 | 0.994144 |
| (-4,0) | **ship** | 0.999818 | 0.999719 | 0.999588 | 0.999545 | 0.999409 | 0.995906 | 0.993992 |
| (-4,0) | noC6 | 0.999818 | 0.999719 | 0.999588 | 0.999545 | 0.999408 | 0.996084 | 0.994169 |
| (-4,0) | noC5 | 0.999818 | 0.999719 | 0.999588 | 0.999545 | 0.999410 | **1.011233** | **1.009290** |
| (-4,0) | noC56 | 0.999818 | 0.999719 | 0.999588 | 0.999545 | 0.999409 | 0.996046 | 0.994132 |
| (-4,-2) | **ship** | 0.999617 | 0.999595 | 0.999612 | 0.999545 | 0.999411 | 0.996036 | 0.993816 |
| (-4,-2) | noC6 | 0.999617 | 0.999595 | 0.999612 | 0.999546 | 0.999410 | 0.996185 | 0.993966 |
| (-4,-2) | noC5 | 0.999617 | 0.999595 | 0.999612 | 0.999545 | 0.999411 | **1.009205** | **1.006955** |
| (-4,-2) | noC56 | 0.999617 | 0.999595 | 0.999612 | 0.999545 | 0.999410 | 0.996137 | 0.993918 |

`ship` = C5+C6 (the shipped tree). `noC6` / `noC5` / `noC56` ablate
`REMAP_STATIONARY_PHASE_LAUNCH` / `TILTED_CARRIER_EXACT_EIKONAL` / both.
`noC5 == ship` bit-for-bit at (0,0) because C5's exact eikonal is gated on
`(L, M) != (0, 0)` - a free correctness check on the toggle harness.

### 2.2 What the table says

**Vignetting is not a factor.** Maximum `clip` anywhere: **1.13e-05**;
`prod(1-clip)` rounds to 1.000000 in every run. With the oracle's 100.0000 %
live power, **there is no legitimate loss channel**, and the reference for
every cell above is 1.0000.

**The loss is one group, and it is discretisation.** Groups 0-4 sit at
0.999408-0.999988. Group 5 sits at **0.995901-0.996185** in every clean
configuration - a deficit of 3.815e-03 to 4.099e-03, **six to three hundred
times any other group** - and it is the group with the highest exit NA. This is
the documented `ray_subsample` loss concentrating where the map is steepest.
The clean envelopes, over 60 clean element calls:

| envelope (clean = `noC6` + `noC56`, `rs=4`) | range | width |
|---|---|---|
| per-element `elem(k)`, all 60 calls | [0.995901, 0.999988] | 4.09e-03 |
| last group `elem(5)`, 10 calls | **[0.995901, 0.996185]** | **2.84e-04** |
| end to end | **[0.993918, 0.994281]** | **3.63e-04** |

**The on-axis shipped row's deficit is 6.5x too SMALL.** `elem(5)` = 0.999371
is a deficit of 6.29e-04 where every clean configuration of the same group,
same grid, same subsample gives 3.82e-03 to 4.10e-03. The discretisation loss
is a real property of the ray lattice and does not go away because C6 is on -
so a configuration that returns six times less of it is not more accurate, it
is **adding energy back**. This is the signature that survives everything
(S2.4), and it is the basis of criterion C1b.

**C5-off + C6-on manufactures energy at the extreme tilted orders.** 1.011233
at (-4,0) and 1.009205 at (-4,-2), with `g4` 1.53e-02 / 1.31e-02 at 91 % / 79 %
of peak and the second moment at **+88 % / +77 %** (S3.2). `noC56` is clean
everywhere. The defect is the *interaction*, not either flag: **C6's
`grad(a_fit)` launch augmentation is only safe on top of C5's exact eikonal
reference.** C5 was justified on EE3 (+20.32 points at (-4,-2)); it is here
shown to be load-bearing for conservation as well.

### 2.3 `ray_subsample` convergence - the deficit converges away, the gain does not

`RS=2 NULL=0 ORDERS='0,0 -4,-2' CONFIGS='ship,noC6'`. **Bit-identical between
L1 and L2.** The entire justification for tolerating a 4e-03 deficit is that it
is discretisation and shrinks as `ray_subsample -> 1`; this tests it.

| quantity | order | `rs=4` | `rs=2` | ratio |
|---|---|---|---|---|
| clean (`noC6`) `elem(5)` **deficit** | (0,0) | 4.099e-03 | **8.11e-04** | **5.1x** |
| clean `elem(5)` deficit | (-4,-2) | 3.815e-03 | **8.04e-04** | **4.7x** |
| clean end-to-end deficit | (0,0) | 5.719e-03 | **1.320e-03** | 4.3x |
| clean end-to-end deficit | (-4,-2) | 6.034e-03 | **1.339e-03** | 4.5x |
| **C6's on-axis GAIN** (ship - noC6, `elem(5)`) | (0,0) | +3.470e-03 | **+4.507e-03** | **1.3x** |

**The deficit falls 4.3-5.1x for a 2x reduction in `ray_subsample`** - steeper
than linear, confirming for design 121 what the library documents only for the
P2 design battery, and consistent with the oracle's 100.0000 % live power.

**C6's on-axis gain does not converge - it grows slightly.** The consequence is
that at `rs=2` the gain dominates and **the entire post-DOE chain ends above
unity: `P_out/P_in` = 1.003186 on the shipped on-axis order**, failing
criterion C2 outright. At `rs=4` the same defect is partly masked because five
per-element deficits absorb it.

**A reversal at (-4,-2).** At `rs=2` the halo verdict on the tilted order
*inverts*: `noC6` carries `g4` = 6.322e-06 at `amax4` = 7.84e-02 - **85x over
its exact-ray ceiling of 7.4639e-08** - while `ship` (C6 on) is clean at
2.739e-08, below the ceiling. At `rs=4` both were clean (~2.5e-08). **On the
tilted order at `rs=2`, C6 is the thing that keeps the field inside the halo
criterion.** This also bounds, for the first time, the prior audit's open item
that the ghost's `ray_subsample` dependence was unmeasured: it is
`rs`-dependent **in both directions**, and any acceptance criterion must
therefore name the `ray_subsample` it is evaluated at.

| order | config | `elem(5)` | end to end | `g4` | `amax4` | `r_rms` |
|---|---|---|---|---|---|---|
| (0,0) | **ship** | **1.003696** | **1.003186** | 4.495e-03 | 9.78e-01 | 0.9477 |
| (0,0) | noC6 | 0.999189 | 0.998680 | 4.559e-11 | 1.47e-05 | 0.8403 |
| (-4,-2) | **ship** | 0.999164 | 0.998628 | 2.739e-08 | 2.32e-04 | 0.8352 |
| (-4,-2) | noC6 | 0.999196 | 0.998661 | **6.322e-06** | **7.84e-02** | 0.8498 |

### 2.4 The library moved mid-audit, and what that measured

`_lens_traced.py` went from L1 `b745859f65f5b6c2` to L2 `adcd1cee14dc7a0a` at
00:42:38. The full 20-run matrix was re-run against L2; `carrier.py` never
changed. This is reported because it is *evidence*, not merely a caveat.

| row (diagnostic config) | `elem(5)` L1 -> L2 | `g4` L1 -> L2 | `amax4` L1 -> L2 |
|---|---|---|---|
| (0,0) noC6 / noC56 | 0.995901 -> **0.995901** | 3.606e-11 -> **3.606e-11** | 1.40e-05 -> **1.40e-05** |
| (-1,0) noC6 / noC56 | 0.996006 -> **0.996006** | 1.425e-09 -> **1.425e-09** | (identical) |
| (-2,0), (-4,0), (-4,-2) noC6 / noC56 | (identical) | (identical) | (identical) |
| **(0,0) ship** | 1.001058 -> **0.999371** | 5.089e-03 -> **3.400e-03** | 9.21e-01 -> 7.70e-01 |
| **(-2,0) ship** | 0.996083 -> 0.996043 | 4.582e-05 -> **2.270e-07** | 1.10e-01 -> 5.73e-03 |
| **(-1,0) noC5** | **1.123769 -> 0.996011** | 1.278e-01 -> **5.314e-10** | 9.96e-01 -> 8.39e-05 |
| **(-4,0) noC5** | 1.008448 -> **1.011233** | 1.251e-02 -> **1.529e-02** | 9.99e-01 -> 9.05e-01 |
| (-4,-2) noC5 | 1.040045 -> 1.009205 | 4.397e-02 -> 1.313e-02 | 9.58e-01 -> 7.89e-01 |

Three things follow.

1. **All 10 C6-off rows are bit-identical; all 10 C6-on rows moved.** The edit
   is confined to the C6 path. That is a clean, unplanned confirmation that
   every "C6 vs no-C6" delta in this document is attributable to C6.
2. **The on-axis total-power signature was erased while the defect remained.**
   `elem(5)` went from 1.001058 (above unity, visibly wrong) to 0.999371
   (inside every absolute band, including the tight one proposed in S6) - while
   the manufactured lobe stayed at **3.400e-03 of the input power at 77 % of
   peak**, 9.5e6 times the exact-ray ceiling. **A criterion written on total
   power alone would have declared this fixed. The halo criterion does not.**
   This single row is the strongest argument in the document for C3/C4 being
   the load-bearing bounds, and for C1b (the deficit floor, which still flags
   the L2 row at 6.5x too small) rather than an absolute band alone.
3. **The magnitude is ill-conditioned; the presence is not.** On the on-axis
   order the four ~1e-06-level input perturbations available here - element
   replay (1.034e-03), diagnostic chain on L1 (5.089e-03), diagnostic chain on
   L2 (3.400e-03), production chain (4.715e-03) - spread `g4` over **4.9x**;
   at (-2,0) the L1 -> L2 edit alone moved it **202x**
   (4.582e-05 -> 2.270e-07). Over the very same perturbations the C6-off value
   never moved from 3.606e-11 by a bit. **Tuning against this magnitude is not
   possible; thresholding its presence is** - which is why C3 and C4 are stated
   as thresholds against an exact-ray ceiling and not as tolerances.

---

## 3. The halo / second-moment table, against an exact-ray ceiling

### 3.1 Choosing the radius - derived, not picked

`energy_hull_121.py` traces the last group's launch plane exactly, with the
element's own carrier and input amplitude, on an 801x801 lattice over the full
co-moving grid, and asks how far out the illuminated pupil can land.

| order | 1w hull | **2w hull** (= the ray-fit disc, `fit_radius_beam_factor=2.0`) | **3w hull** | 4w hull | radius holding 1 - 1e-6 of launched power |
|---|---|---|---|---|---|
| (0,0) | 1.189 mm | 2.423 mm | 3.755 mm | 5.260 mm | 2.901 mm |
| (-1,0) | 1.195 | 2.444 | 3.807 | 5.366 | 2.913 |
| (-2,0) | 1.201 | 2.470 | 3.865 | 5.481 | 2.939 |
| (-4,0) | 1.219 | 2.533 | 4.004 | 5.757 | 3.003 |
| (-4,-2) | 1.227 | 2.549 | 4.042 | 5.832 | 3.016 |

The 3w contour is the `e^-9` amplitude / `e^-18` intensity contour; everything
inside it lands within **3.76-4.04 mm**. **4 mm is therefore the smallest
radius beyond which essentially no illuminated input can arrive at any order**,
which is why `g4` is the halo metric here - the radius is a property of the
optics, not a choice.

That is turned into a quantitative ceiling by weighting the same exact trace
with the element's own input intensity:

| order | **exact-ray `g1`** | **`g2`** | **`g4`** | **`g8`** | **exact-ray `r_rms`** | core (<1 mm) |
|---|---|---|---|---|---|---|
| (0,0) | 2.4104e-01 | 4.2998e-03 | **3.5641e-10** | 4.5714e-12 | **0.8407 mm** | 0.6185 mm |
| (-1,0) | 2.4125e-01 | 4.3360e-03 | **1.0153e-08** | 2.9408e-10 | **0.8413** | 0.6188 |
| (-2,0) | 2.4235e-01 | 4.4240e-03 | **3.0443e-08** | 1.4839e-09 | **0.8427** | 0.6191 |
| (-4,0) | 2.4662e-01 | 4.7731e-03 | **7.5559e-08** | 6.2887e-09 | **0.8483** | 0.6202 |
| (-4,-2) | 2.4815e-01 | 4.9057e-03 | **7.4639e-08** | 5.7758e-09 | **0.8503** | 0.6206 |

### 3.2 The measured halo, at the last group's exit, in the production chain

Library L2, `rs=4`. `gN` = returned power beyond N mm of the traced exit chief
ray, over that group's input power. `amax4` = largest `|E|` beyond 4 mm over
the peak. `r_rms` = power-weighted second moment of the whole returned field;
`core` = the same restricted to r < 1 mm. `g4/ceil` is against S3.1;
`dr_rms` is against the S3.1 exact-ray second moment.

| order | config | g1 | g2 | **g4** | g8 | **amax4** | **r_rms** | core | **g4/ceil** | **dr_rms** |
|---|---|---|---|---|---|---|---|---|---|---|
| (0,0) | **ship** | 2.409e-01 | 7.195e-03 | **3.400e-03** | 0 | **7.70e-01** | **0.9349** | 0.6204 | **9.5e+06** | **+11.20 %** |
| (0,0) | noC6 | 2.396e-01 | 4.242e-03 | 3.606e-11 | 0 | 1.40e-05 | 0.8422 | 0.6201 | 0.10 | +0.18 % |
| (0,0) | noC56 | 2.396e-01 | 4.242e-03 | 3.606e-11 | 0 | 1.40e-05 | 0.8422 | 0.6201 | 0.10 | +0.18 % |
| (-1,0) | **ship** | 2.388e-01 | 3.808e-03 | 3.318e-10 | 0 | 8.24e-05 | 0.8384 | 0.6196 | 0.03 | -0.34 % |
| (-1,0) | noC6 | 2.414e-01 | 4.360e-03 | 1.425e-09 | 0 | 9.84e-05 | 0.8429 | 0.6194 | 0.14 | +0.19 % |
| (-1,0) | noC5 | 2.389e-01 | 3.797e-03 | 5.314e-10 | 0 | 8.39e-05 | 0.8384 | 0.6196 | 0.05 | -0.34 % |
| (-1,0) | noC56 | 2.413e-01 | 4.370e-03 | 2.472e-09 | 0 | 9.69e-05 | 0.8428 | 0.6194 | 0.24 | +0.18 % |
| (-2,0) | **ship** | 2.388e-01 | 3.764e-03 | **2.270e-07** | 2.217e-07 | **5.73e-03** | 0.8382 | 0.6195 | **7.5** | -0.53 % |
| (-2,0) | noC6 | 2.426e-01 | 4.461e-03 | 7.750e-09 | 0 | 1.60e-04 | 0.8443 | 0.6196 | 0.25 | +0.19 % |
| (-2,0) | noC5 | 2.396e-01 | 4.360e-03 | **5.664e-04** | 5.664e-04 | **4.00e-01** | 0.8753 | 0.6195 | **1.9e+04** | +3.87 % |
| (-2,0) | noC56 | 2.423e-01 | 4.431e-03 | 6.657e-09 | 0 | 1.58e-04 | 0.8439 | 0.6195 | 0.22 | +0.14 % |
| (-4,0) | **ship** | 2.386e-01 | 3.739e-03 | 2.628e-08 | 0 | 2.37e-04 | 0.8376 | 0.6191 | 0.35 | -1.26 % |
| (-4,0) | noC6 | 2.473e-01 | 4.857e-03 | 2.831e-08 | 0 | 2.00e-04 | 0.8500 | 0.6204 | 0.37 | +0.20 % |
| (-4,0) | noC5 | 2.547e-01 | 1.909e-02 | **1.529e-02** | 1.529e-02 | **9.05e-01** | **1.5930** | 0.6193 | **2.0e+05** | **+87.79 %** |
| (-4,0) | noC56 | 2.461e-01 | 4.721e-03 | 3.622e-08 | 0 | 2.45e-04 | 0.8484 | 0.6201 | 0.48 | +0.01 % |
| (-4,-2) | **ship** | 2.384e-01 | 3.720e-03 | 2.653e-08 | 0 | 2.34e-04 | 0.8375 | 0.6193 | 0.36 | -1.51 % |
| (-4,-2) | noC6 | 2.486e-01 | 4.986e-03 | 2.507e-08 | 2.29e-12 | 2.26e-04 | 0.8519 | 0.6210 | 0.34 | +0.19 % |
| (-4,-2) | noC5 | 2.525e-01 | 1.695e-02 | **1.313e-02** | 1.313e-02 | **7.89e-01** | **1.5061** | 0.6195 | **1.8e+05** | **+77.13 %** |
| (-4,-2) | noC56 | 2.471e-01 | 4.788e-03 | 3.152e-08 | 0 | 2.39e-04 | 0.8498 | 0.6206 | 0.42 | -0.06 % |

### 3.3 The reference validates the clean rows and convicts the rest

**Every clean row sits below its exact-ray halo ceiling** - `g4/ceil` spans
**0.10 to 0.48** across all ten of them - and **every clean row's second moment
agrees with the exact-ray reference to within -0.06 % to +0.20 %**, a
predominantly *positive* offset of the sign and roughly the size the
diffractive skirt the ray model omits requires. `g1` and `g2` agree to
0.06-1.8 % as well. **The instrument and the clean configurations validate each
other**, which is what licenses using the reference to convict the rest:

| row | `g4` | ceiling | **factor over ceiling** |
|---|---|---|---|
| **(0,0) SHIPPED** | 3.400e-03 | 3.5641e-10 | **9.5e6** |
| (-4,0) noC5 | 1.529e-02 | 7.5559e-08 | 2.0e5 |
| (-4,-2) noC5 | 1.313e-02 | 7.4639e-08 | 1.8e5 |
| (-2,0) noC5 | 5.664e-04 | 3.0443e-08 | 1.9e4 |
| **(-2,0) SHIPPED** | 2.270e-07 | 3.0443e-08 | **7.5** |

**`amax4` is the sharpest discriminator.** Clean rows span
1.40e-05 to 2.45e-04; violating rows span 5.73e-03 to 9.05e-01. It separates a
*lobe* from a *skirt* in a way a power fraction cannot: the (0,0) shipped ghost
carries 0.34 % of the power but stands at **77 % of the peak amplitude**.

**The core never moves.** `core` r_rms spans 0.6191-0.6210 mm across all 20
rows, clean and grossly ghosted alike - a 0.31 % spread. This is the mechanical
reason EE3/EE6/EE12 are blind: the spot is untouched, the energy is elsewhere.

---

## 4. C6 scored on conservation: does it improve, worsen, or leave unchanged?

**Split, and the split is by order.** Diagnostic configuration, L2, `rs=4`:

| order | `elem(5)` noC6 -> ship | deficit vs the 3.8-4.1e-03 floor | `g4` noC6 -> ship | `r_rms` dev noC6 -> ship | verdict |
|---|---|---|---|---|---|
| **(0,0)** | 0.995901 -> **0.999371** | **6.29e-04, 6.5x too small** | 3.61e-11 -> **3.40e-03** | +0.18 % -> **+11.20 %** | **WORSE - adds energy, 9.5e6x ceiling** |
| (-1,0) | 0.996006 -> 0.996017 | 3.98e-03, normal | 1.43e-09 -> 3.32e-10 | +0.19 % -> -0.34 % | better |
| **(-2,0)** | 0.996057 -> 0.996043 | 3.96e-03, normal | 7.75e-09 -> **2.27e-07** | +0.19 % -> -0.53 % | **WORSE on halo, 7.5x ceiling** |
| (-4,0) | 0.996084 -> 0.995906 | 4.09e-03, normal | 2.83e-08 -> 2.63e-08 | +0.20 % -> -1.26 % | unchanged |
| (-4,-2) | 0.996185 -> 0.996036 | 3.96e-03, normal | 2.51e-08 -> 2.65e-08 | +0.19 % -> -1.51 % | unchanged |

**Answer to the brief's question.** On the tilted orders C6 was designed for,
C6 **leaves the energy budget unchanged** - the element ratio moves by at most
1.8e-04, twenty times smaller than the 4e-03 discretisation deficit it sits on,
and `g4` stays well under the exact-ray ceiling. On the on-axis order it
**worsens conservation outright**: it returns six times less discretisation
deficit than physics allows and puts a 77 %-of-peak lobe 9.5e6 times over the
ceiling. At (-2,0) it produces a smaller lobe, 7.5x over the ceiling at 0.6 %
of peak.

**It is not a discretisation artefact.** At `rs=2` the surrounding deficit
falls 5.1x while C6's on-axis gain grows to 4.51e-03, so the shipped on-axis
chain ends above unity at 1.003186 (S2.3). Nothing converges this away.

**The tilted case is stronger for C6 than the `rs=4` table alone shows.** At
`rs=2` on (-4,-2), C6-*off* is the configuration that violates the halo ceiling
(`g4` 6.32e-06, 85x over, at 7.8 % of peak) and C6-on is clean. C6 is not
merely harmless on the tilted orders; at the finer subsample it is what keeps
them inside the criterion.

**Is that acceptable for oracle use? No on (0,0), marginal on (-2,0), yes on
the rest.** An oracle whose on-axis call invents half a percent of the light
cannot certify anything on that order, however good its EE3.

**The remedy already exists in the tree.** `REMAP_STATIONARY_PHASE_FIT_GUARD`
acts only on the concentric fit branch - the on-axis one - and the prior audit
pinned it byte-identical on all tilted orders. On the evidence here, turning it
on is **free on (-1,0)/(-4,0)/(-4,-2) and exactly what (0,0) needs**. Two
caveats keep this a recommendation, not a conclusion: (-2,0)'s decentre is
0.482 w, so which branch it takes is not established here; and the flag's
chain-level cost has never been measured (S7.4).

### 4.1 The trade, priced in both currencies on ONE field

`energy_ee_vs_conservation_121.py`, order (0,0), the **production
configuration** - the same `approx_common.run_chain` the campaign's EE numbers
came from: exact Bluestein readout, `NOUT=192`, `dx_out=0.1 um`,
`n_fine_cap=12288`, `window_factor=4.0`, `final_leg='exact'`, fixed lattice on
the chief ray. Both currencies are read from the same run. **This table is
bit-identical on library L1 and L2.**

| config | EE3 % | EE6 % | EE12 % | `P_tile` % | `elem(last)` | `g4` | `amax4` | `r_rms` mm |
|---|---|---|---|---|---|---|---|---|
| **ship** | **88.400** | 98.646 | 98.803 | 98.8046 | **1.000741** | **4.715e-03** | **8.32e-01** | **0.9644** |
| noC6 | 86.709 | 98.637 | 98.766 | 98.7691 | 0.995883 | **0.000e+00** | 0.00e+00 | 0.8419 |
| **C6 buys / costs** | **+1.691 pts** | +0.009 | +0.037 | +0.036 | **+0.486 %** | **+4.7e-03** | **+0.83** | **+14.6 %** |

**This single table is the argument of this audit.** On the on-axis order,
niche C6 buys **1.691 EE3 points** and pays with **0.486 % of the input power
manufactured** - `elem(last)` **above unity at 1.000741** - deposited as a lobe
at **83 % of peak amplitude** where the exact ray trace permits at most
3.56e-10, a factor of **1.3e7**, moving the second moment **+14.6 %** against a
reference the C6-off field matches to 0.14 %. **`noC6`'s halo beyond 4 mm is
exactly zero.**

And every metric the campaign used reported that transaction as a modest win:
EE3 **+1.7 points**, EE6 **+0.009**, EE12 **+0.037**, `P_tile` **+0.036**.
`P_tile` is the *total power in the readout tile* and is also blind, because
the tile is 19.2 um across and the ghost is at 4-8 mm - **300x outside it**.
There is no member of the EE family, including total-power-in-tile, that could
have caught this.

**One unresolved observation, recorded rather than claimed.** On every tilted
order C6 removes ~25 % of the 2-4 mm skirt (`g2` 4.99e-03 -> 3.72e-03 at
(-4,-2)) and pulls `r_rms` **below** the exact-ray reference (-1.26 % / -1.51 %
at (-4,0) / (-4,-2)), where every clean row sits at -0.06 % to +0.20 %. The
exact-ray reference is launched along `grad W` only, so it is matched to C6-off
and biased against C6-on, and a launch correction of C6's magnitude should move
the exit-plane second moment by ~0.3 %, not 1.5 %. This audit cannot separate
"a real correction the ray reference does not model" from "a second, milder
manifestation of the same fit conditioning" (S7.2).

---

## 5. Would the library's own guards catch a violation?

The post-hoc self-check in `_lens_traced.py` fires when `P_out / P_ap` leaves
`[1 - (0.080 + 0.010*(sub-1)), 1 + 0.050]` = **[0.8900, 1.0500]** at
`ray_subsample=4`.

### 5.1 It fires - the premise in the brief is corrected

`energy_guard_probe.py` replays the last-group call under seven configurations
per order on library L2 and records whether the warning was raised.

| order | config | **guard's ratio** | **guard fired?** | fold | `g4` | `amax4` | `r_rms` |
|---|---|---|---|---|---|---|---|
| (0,0) | C5on C6off | 0.99590 | - (in band) | 0 | 3.61e-11 | 1.40e-05 | 0.8422 |
| (0,0) | C5on C6on (element replay) | 0.99701 | - (in band) | 0 | 1.03e-03 | 3.30e-01 | 0.8638 |
| (0,0) | fit order 10 | **1.08514** | **WARN** | 1 | 8.92e-02 | 9.30e-01 | 2.4192 |
| (0,0) | fit order 14 | 0.99773 | - (in band) | 1 | 1.42e-03 | 4.05e-01 | 0.8927 |
| (0,0) | **C6off, fit order 14** | **1.04593** | **- (in band)** | 1 | **4.62e-02** | **9.62e-01** | **1.8263** |
| (-4,0) | C5on C6on (shipped) | 0.99591 | - | 0 | 2.62e-08 | 2.70e-04 | 0.8369 |
| (-4,0) | **C5off C6on** | **1.01667** | **- (in band)** | 0 | **2.08e-02** | **9.95e-01** | **1.7764** |
| (-4,0) | **fit order 14** | **1.03317** | **- (in band)** | 0 | **3.73e-02** | **9.62e-01** | **1.4699** |
| (-4,-2) | C5off C6on | 0.99862 | - (in band) | 0 | 2.60e-03 | 6.10e-01 | 1.0065 |
| (-4,-2) | **fit order 14** | **1.48807** | **WARN** | 0 | 4.92e-01 | 9.94e-01 | 4.8448 |

**Does it fire on the `P/Pin` 1.82 / 2.21 case? Yes, when the ratio is that
large.** On library L1 the same `fit order 14` configuration read **2.20714**
at (-4,0) and warned. On L2 it reads **1.03317** at that order and **does not
warn** - while still carrying **3.73e-02 of manufactured energy at 96 % of
peak** and a second moment at 1.73x. Across both states the guard produced
**zero false-negatives relative to its own band**: every out-of-band ratio
warned, every in-band ratio did not. **The guard is not broken. Its band is
the problem.**

### 5.2 Does it fire anywhere in the shipped 121 configuration? No

Across all 20 L2 chain runs x 6 groups = 120 element calls, the energy
self-check fired **zero times** - including on (-4,0) `noC5` at ratio 1.011233
carrying `g4` = 1.53e-02 at 91 % of peak with the second moment at +88 %, and
including the shipped on-axis production call at 1.000741. (On L1 it fired
exactly once in 120, on (-1,0) `noC5` at 1.123769.) **In the `ship`
configuration it never fires on any group of any order, in either library
state.**

### 5.3 The band is wide enough to hide real defects - by ~5 orders of magnitude

| what got through the band | ratio | margin to the +0.050 limit | `g4` | `amax4` | `r_rms` vs exact ray |
|---|---|---|---|---|---|
| (0,0) C6off fit-order-14, element | 1.04593 | 0.004 to spare | **4.62e-02** | 0.962 | 2.17x |
| (-4,0) fit-order-14, element | 1.03317 | 0.017 | 3.73e-02 | 0.962 | 1.73x |
| (-4,0) C5off C6on, element | 1.01667 | 0.033 | 2.08e-02 | 0.995 | 2.09x |
| (-4,0) noC5, chain | 1.011233 | 0.039 | 1.53e-02 | 0.905 | 1.88x |
| (-4,-2) noC5, chain | 1.009205 | 0.041 | 1.31e-02 | 0.789 | 1.77x |
| **(0,0) SHIPPED, production chain** | **1.000741** | **0.049 to spare** | **4.72e-03** | **0.832** | **1.15x** |

**A field may manufacture up to 5 % of the input power as a lobe at 96 % of
peak, more than double its own second moment, and pass.** The shipped on-axis
defect passes with a factor of **67** in hand.

The reason is structural, not a mis-set constant: **total power is the wrong
observable for this failure mode.**

- Total power separates clean from ghosted by the *size* of the ghost:
  4.7e-03 at (0,0), against a tolerance of 5.0e-02 - 11x inside.
- The halo fraction separates the same two fields by
  **3.6e-11 vs 4.7e-03 = 1.3e8**, and `amax4` by **1.4e-05 vs 8.3e-01 = 6e4**.

A guard on a scalar sum can only ever be as sensitive as the ghost is large. A
guard on the halo is sensitive to *where* the energy is, which is the defect.
S2.4 shows this is not hypothetical: a library change removed the on-axis
above-unity signature entirely while leaving the lobe in place.

### 5.4 Three further structural limits

1. **The guard is per-element and re-references to each element's own input**,
   so it cannot see cumulative drift. Six groups at 0.9995 give 0.9970 end to
   end with every factor comfortably in band; thirty groups would give 0.985
   and still never warn.
2. **Its denominator excludes aperture-clipped power** (correctly, to avoid
   flagging legitimate vignetting) - so **vignetting is invisible to it by
   construction**. Harmless on design 121, where clipping is <= 1.13e-05, but
   it means the guard cannot be read as a statement about throughput.
3. **The deficit side is ~300x looser than this design needs.** The band allows
   0.890; design 121's clean per-element envelope is [0.995901, 0.999988] and
   its clean end-to-end envelope is [0.993918, 0.994281], width 3.6e-04. The
   wide band is justified in the source by the P2 design battery (worst cell
   0.9535, a genuinely diverging exit fan leaving the window). That
   justification is sound **for a library default**; it is not a reason for a
   design-121 oracle to accept 0.890.

**Assessment: the guard is correctly implemented, correctly documented, and
appropriate as a library-wide default that must not warn spuriously on any
design. It is not, and cannot be, an oracle acceptance criterion.** Keep it
exactly as it is; add the criterion below alongside it.

---

## 6. The proposed conservation acceptance criterion

Every bound below is derived from a measurement in this document, with the
measurement named. Nothing is rounded to a pleasing figure.

**The criterion is evaluated at the configuration's own `ray_subsample`, which
must be stated with the result.** S2.3 shows both the deficit and the halo are
`ray_subsample`-dependent - the deficit by 5x between `rs=4` and `rs=2`, the
halo by enough to *invert* which configuration passes at (-4,-2). C1/C2's
bounds are set from the `rs=4` clean envelope; a run at a different subsample
must re-derive them from its own clean envelope. C1b and C3/C4/C5 are
referenced to an exact ray trace or to a same-configuration control rather than
to an observed spread, and carry over unchanged.

### C1a - Per-element conservation, absolute

> For every group *k*, `P_out(k) / P_ap(k)` must lie in **[0.9900, 1.00020]**,
> with `P_ap` the aperture-transmitted input power and aperture clipping
> reported separately.

- **Upper 1.00020.** The physical reference is exactly 1.0000 (oracle live
  power 100.0000 %, S1.5); ray-tube transport cannot create energy. The
  largest ratio in the 60-call clean matrix is **0.999988**, so any value above
  1.0000 is already outside the clean envelope. 2.0e-04 of headroom is **18x**
  the largest aperture clip measured (1.13e-05) and far above float-summation
  noise. It rejects the shipped on-axis production row (1.000741) by 3.7x.
- **Lower 0.9900.** The clean envelope is [0.995901, 0.999988]. 0.9900 sits
  5.9e-03 below the worst clean element, permitting the last group's 4.1e-03
  deficit to grow **2.4x** before firing - enough to survive a grid or
  subsample change, tight enough to catch an order-of-magnitude regression.
  The shipped library guard permits 26x.

### C1b - Per-element DEFICIT FLOOR (the criterion that survives)

> For the highest-NA group, `1 - P_out/P_ap` must not fall below **0.5x** the
> value the same design, grid and `ray_subsample` returns with the ray-density
> transport alone (the C6-off reference configuration).

This is the criterion S2.4 demands. The discretisation deficit is a real
property of the ray lattice, not of the model layered on top of it: measured at
**3.815e-03 to 4.099e-03** across five orders at `rs=4`, and
**8.04e-04 to 8.11e-04** at `rs=2` - a tight, reproducible, subsample-scaling
floor. **A configuration that returns materially less of it is not more
accurate; it is adding energy back.**

- **The factor 0.5.** The clean floor's own spread across five orders is
  2.84e-04, i.e. 7 % of its value; 0.5x is **seven times** that spread, so it
  cannot fire on order-to-order variation.
- It flags the on-axis shipped row at **6.29e-04 against a 4.10e-03 floor -
  0.15x, i.e. 6.5x too small** - on library L2, where the absolute bound C1a
  passes. On library L1 the same row read 1.001058 and C1a caught it. **C1b
  catches it in both states, at both subsamples, and in both the diagnostic and
  production configurations.** No other conservation bound proposed here does.

### C2 - End-to-end conservation

> `P_out(last) / P_in(first)` must lie in **[0.9850, 1.00050]**, with the
> aperture-clipped fraction reported alongside, not folded in.

- **Lower 0.9850.** Clean end to end is [0.993918, 0.994281], width 3.63e-04.
  0.9850 is 8.9e-03 below the worst clean value, permitting the 6.1e-03 total
  deficit to grow **2.4x** - the same multiple as C1a, applied to the product
  of six factors so the two are consistent.
- **Upper 1.00050.** Six elements each allowed 2.0e-04 would compound to
  1.0012; 1.00050 is deliberately tighter, because a *systematic* per-element
  gain is exactly what C1 exists to catch and must not be laundered by the
  end-to-end bound. It rejects (-4,0) `noC5` (1.009290), (-4,-2) `noC5`
  (1.006955), and the shipped on-axis chain at `rs=2` (1.003186).

### C3 - Halo fraction (the criterion the campaign was missing)

> At each element's exit plane, `g4` - returned power beyond 4 mm of the traced
> exit chief ray, over that element's input power - must satisfy
> **`g4 <= 3 x g4_exact(order)`**, where `g4_exact` is the exact ray trace of
> the same launch weighted by the element's own input amplitude (S3.1). Where a
> per-order trace is unavailable, the fixed bound **`g4 <= 3.0e-07`** applies.

- **The radius 4 mm is derived**, not chosen: the 3w (`e^-18` intensity) launch
  contour lands within 3.755-4.042 mm at every order (S3.1), so beyond 4 mm
  essentially no illuminated input can arrive.
- **The factor 3.** Clean rows sit at **0.10x to 0.48x** their own ceiling
  across all ten (S3.2). A factor of 3 is **6.3x** the largest clean excursion
  and still rejects the shipped (0,0) row by **3.2e6** and the shipped (-2,0)
  row by 2.5x.
- **The fixed 3.0e-07** is 4.0x the largest per-order ceiling (7.5559e-08) and
  8.3x the largest clean measurement (3.622e-08).

### C4 - Halo amplitude

> **`amax4 <= 1.0e-03`** - the largest `|E|` beyond 4 mm, over the peak.

Clean rows span **1.40e-05 to 2.45e-04** across five orders and two clean
configurations; violating rows span **5.73e-03 to 9.05e-01**. The two
populations are separated by a factor of **23** with nothing in between, and
the bound sits inside that gap at **4.1x** above the worst clean row and
**5.7x** below the mildest violation. This is the criterion that distinguishes
a *lobe* from a *skirt*, and it catches the on-axis production defect most
decisively (8.32e-01 against 1.0e-03: **832x**).

### C5 - Second moment

> `|r_rms - r_rms_exact| / r_rms_exact <= 0.030`, with `r_rms_exact` the
> exact-ray second moment of the same launch (S3.1), and the core moment
> `r_rms(r<1mm)` reported alongside.

- Clean rows agree with the reference to **-0.06 % to +0.20 %** across all ten
  (S3.2). A 3.0 % bound is **15x** the worst clean deviation, which absorbs the
  reference's own ray-optics bias with room to spare.
- It rejects (0,0) production ship (+14.6 %), (-4,0) `noC5` (+87.8 %),
  (-4,-2) `noC5` (+77.1 %), (-2,0) `noC5` (+3.9 %).
- **It does not reject the tilted `ship` rows** at -1.26 % and -1.51 %, which
  is intended given S4.1's unresolved reading of that shift: the criterion
  should flag it as marginal, not fail it, until the launch-reference question
  is settled.
- **The core moment must be reported but must NOT be part of the bar.** It
  spans 0.6191-0.6210 mm over every row in this audit, clean and grossly
  ghosted alike - it is precisely as blind as EE3, and including it would
  reintroduce the defect this criterion exists to remove.

### C6 - Mandatory reporting, not a bound

Any future change must additionally report, and may not silently omit:
(a) the aperture-clipped fraction per group; (b) the count of energy
self-check, fold-caustic and exit-NA warnings per group; (c) the
sampling-adequacy p99.9 of any field it makes a *wave* claim about; (d) a
bit-exact null intervention; (e) **the sha256 of the library files actually
imported** - S2.4 is what that clause is for.

### 6.1 How the shipped state scores

Production configuration where available, diagnostic otherwise; library L2,
`rs=4`.

| criterion | (0,0) | (-1,0) | (-2,0) | (-4,0) | (-4,-2) |
|---|---|---|---|---|---|
| C1a per-element absolute | **FAIL** (1.000741) | pass | pass | pass | pass |
| C1b deficit floor | **FAIL** (0.15x floor) | pass (1.00x) | pass (1.00x) | pass (1.05x) | pass (1.04x) |
| C2 end to end | pass (0.997750) | pass | pass | pass | pass |
| C3 halo fraction | **FAIL** (3.2e6x bound) | pass | **FAIL** (2.5x bound) | pass | pass |
| C4 halo amplitude | **FAIL** (832x) | pass | **FAIL** (5.7x) | pass | pass |
| C5 second moment | **FAIL** (+14.6 %) | pass (-0.34 %) | pass (-0.53 %) | marginal (-1.26 %) | marginal (-1.51 %) |

**Three of five orders pass every criterion. (0,0) fails five of six and
(-2,0) fails two.** Note C2 *passes* on the worst order at `rs=4`: the
end-to-end number launders a per-element gain against five per-element
deficits. That is why C1 exists and why C2 alone is insufficient - and at
`rs=2`, where the deficits are 5x smaller, the laundering stops working and C2
fails too (1.003186 > 1.00050).

At `ray_subsample=2`, the two orders measured:

| criterion | (0,0) ship | (0,0) noC6 | (-4,-2) ship | (-4,-2) noC6 |
|---|---|---|---|---|
| C1a per-element | **FAIL** (1.003696) | pass (0.999189) | pass (0.999164) | pass (0.999196) |
| C1b deficit floor | **FAIL** (negative) | pass | pass (1.04x) | pass |
| C2 end to end | **FAIL** (1.003186) | pass (0.998680) | pass (0.998628) | pass (0.998661) |
| C3 halo fraction | **FAIL** (4.2e6x bound) | pass | pass (0.12x bound) | **FAIL** (28x bound) |
| C4 halo amplitude | **FAIL** (978x) | pass | pass | **FAIL** (78x) |
| C5 second moment | **FAIL** (+12.7 %) | pass (-0.05 %) | pass (-1.8 %) | pass (-0.06 %) |

**This is the clearest statement of the trade: at `rs=2`, C6 fails six of six
on axis and is the only reason (-4,-2) passes C3 and C4.**

---

## 7. What I could not measure

1. **Where the halo goes.** Every halo figure is at an element's own exit
   plane, where it is an exact property of the returned array. The co-moving
   grid is 14.6-15.1x short of `lambda/(2 NA_exit)` and 96.5-96.8 % of the exit
   power is above its Nyquist angle (S1.4), so **the image-plane fate of a
   manufactured lobe is not determined by these numbers**. This does not weaken
   the conservation conclusions - manufactured energy is manufactured wherever
   it lands - but **no image-plane r2m bound can be set on this grid**, which
   is why C5 is stated at the exit plane. The library's own warning fires on 4
   of 6 groups of every run and says exactly this.
2. **The C6 tilted second-moment shift (-1.26 % / -1.51 %) is unexplained.**
   The exact-ray reference launches along `grad W` only, so it is matched to
   C6-off; a launch correction of C6's magnitude should move the exit second
   moment by ~0.3 %, not 1.5 %. Resolving it needs a reference traced along
   `grad(W + a_fit)`; `wfe_probe_remap` already does this and it was not run.
3. **`ray_subsample` was measured at 2 and 4 only**, and at `rs=2` only on
   (0,0) and (-4,-2). `rs=8` - the *library's* shipped default, though not
   design 121's configuration - is not measured and would need a fresh chain A.
   The (-4,-2) halo inversion at `rs=2` is therefore not known to generalise
   across the fan.
4. **The chain-level cost of `REMAP_STATIONARY_PHASE_FIT_GUARD = True` is
   still unmeasured**, as the prior audit recorded. S4 recommends it on an
   element-level byte-identity argument, which is evidence but not a chain
   measurement.
5. **(-2,0)'s fit branch is not established.** Its decentre is 0.482 w;
   whether it takes the concentric or off-centre branch decides whether the fit
   guard would help it.
6. **The element-level ghost is measured at `rs=4` only**, as in the prior
   audit - the element captures are cached at that value.
7. **Only the post-DOE relay is scored.** Chain A (source -> DOE) and
   `propagate_traced_carrier_chain_multi`'s readout tiling and recombination
   are untouched.
8. **S4.1 covers the on-axis order only.** The tilted EE/conservation pair was
   budgeted and not run (~8 min per row, ~48 GB resident). The tilted EE case
   for C6 is therefore quoted from the prior audit's +15.27 EE3 points at
   (-4,-2) rather than re-measured; only the conservation half of that trade is
   this audit's own measurement.
9. **What the L1 -> L2 library edit actually changed is not known** - only its
   effects were measured (S2.4). The file is owned by another agent and was not
   read for intent, only hashed.

---

## 8. Artefacts found and killed in MY OWN instruments

Recorded because this project has now had ~25 artefacts pass as findings.

1. **The library changed underneath the audit and I nearly published across
   two states.** All of S2/S3 was measured on L1; `_lens_traced.py` was edited
   at 00:42:38; the first draft of this document quoted L1 numbers as "the
   current tree". Caught by hashing the file at the *end* as well as the start.
   The fix was to re-run the whole cheap matrix on L2 and demote L1 to a
   controlled comparison - which turned the accident into S2.4, the strongest
   evidence in the document. **Hash at the end, not only at the beginning.**
2. **"The element input's phase is at Nyquist" was nearly reported as a
   finding.** The first pass measured an amplitude-weighted p99.9 of 3.13 rad
   and it read as a defect. It is not: it is expected from a co-moving grid
   15x short of the exit NA, and it is **irrelevant to every number here**, all
   of which are power measurements. Settled by restricting to four amplitude
   contours from `1e-6 pk` to `e^-1 pk` and finding the statistic unmoved -
   proving it is a property of the field, not of the percentile.
3. **`r_rms(exact ray)` returned `NaN` on two of five orders and a finite
   number on the other three.** Dead rays return `NaN` coordinates and
   `0.0 * NaN` is `NaN`, so the weighted second moment was poisoned while the
   *shell* sums - which compare `re > r`, `False` for `NaN` - were silently
   unaffected. Had only the three finite orders been measured, a poisoned
   reference would have been quoted. Fixed with an explicit finite-radius mask
   that reports the retained weight (**1.000000000** on every order).
4. **The free-space legs conserving to 1e-15 was nearly reported as evidence
   that the transport is sound.** It is a tautology - FFT propagation is
   exactly unitary on the grid - and says nothing about whether the transported
   field is physically right. Demoted to a flagged non-result (S1.4). The
   "lossless trap" in a new costume.
5. **The element-level replay understates the ghost by 3-5x.** Replaying a
   C6-*off*-captured input through the last group gives `g4` 1.03e-03; the
   production chain gives 4.72e-03 from an input differing by ~1e-06 relative.
   Had this audit measured only at element level - as the prior one did - it
   would have reproduced 0.103 % and missed 0.486 %. **Element-level replay is
   not a substitute for the chain.**
6. **The S4.1 instrument threw away a 55-minute run at its last statement.** On
   the on-axis order the chain hands the element a bare `float` R rather than a
   `TiltedCarrier`, and the halo step dereferenced `car.x0` *after* all four
   chain runs completed. `probe_c6_element.get_call` already guards this and
   the guard was not copied. Cost was schedule only; no number was affected.
   **The cheap assertion belongs before the expensive computation.**
7. **Quoting halo exceedances against two different denominators.** An early
   draft reported `rs=4` exceedances against the *bound* and `rs=2` against the
   *ceiling*, making the latter look 3x worse. Normalised to "over the bound"
   throughout.
8. **A grep that ate its own table.** The first `energy_guard_probe` run
   appeared to produce a single row; `grep -v "^  "` was stripping every
   right-aligned label. No number was wrong, but a one-row table was briefly
   read as a crash.
9. **`(-1,0)` has no element-call cache**, so its element-level rows are absent
   from S5.1 while its chain rows are complete. Stated rather than papered over
   by substituting a neighbouring order.

---

## 9. Reproduction

All commands from `validation/repro_traced_carrier_121/`. Every runner prints
the sha256 of the library file it imported - **check it against S1.1 before
comparing any number below to this document.** Total wall time ~25 min.

```bash
# S1.1 -- library identity, apertures, the guard's denominator
ORDERS='0,0 -4,0 -4,-2' python energy_recon.py

# S2 / S3.2 -- the per-stage conservation table + halo, all configs.
#              ~8-40 s per run; each run holds well under 10 GB.
ORDERS='0,0 -1,0 -2,0 -4,0 -4,-2' CONFIGS='ship,noC6,noC5,noC56' \
    python energy_stage_audit_121.py

# S2.3 -- the ray_subsample convergence control (chain A is cached at rs=2)
RS=2 NULL=0 ORDERS='0,0 -4,-2' CONFIGS='ship,noC6' python energy_stage_audit_121.py

# S3.1 / S3.3 -- the exact traced exit hull, the halo ceiling, the r2m reference
ORDERS='0,0 -1,0 -2,0 -4,0 -4,-2' NL=801 python energy_hull_121.py

# S5 -- element-level conservation + halo + DID THE GUARD FIRE
ORDERS='0,0 -4,0 -4,-2' python energy_guard_probe.py

# S1.4 -- sampling adequacy, four amplitude contours
ORDERS='0,0 -4,-2' python energy_sampling_check.py

# S1.5 -- the conservation reference (existing script, reused verbatim)
ORD='0,0;-1,0;-4,0;-4,-2' NL=161 NOUT=61 DXO=0.2 CLIP=3.0 python exact_ray_oracle_121.py

# S4.1 -- both currencies on one field, PRODUCTION path.  ~4-8 min per run,
#         ~48 GB resident: do NOT run concurrently with another chain batch.
LUMEN_PIN=0 NULL=0 ORDERS='0,0' CONFIGS='ship,noC6' PYTHONUNBUFFERED=1 \
    python energy_ee_vs_conservation_121.py
```

### Files added by this study (none are library code)

`energy_recon.py`, `energy_stage_audit_121.py`, `energy_guard_probe.py`,
`energy_hull_121.py`, `energy_sampling_check.py`,
`energy_ee_vs_conservation_121.py`, and this document.

Raw logs: `_energy_stage_v2.txt` (L2, primary), `_energy_stage_A.txt` /
`_energy_stage_B.txt` (L1, S2.4 comparison), `_energy_rs2_v2.txt` /
`_energy_rs2.txt`, `_energy_guard_v2.txt`, `_energy_hull4.txt`,
`_energy_oracle.txt`, `_energy_ee_v2.txt` / `_energy_ee2.txt`.

### Library changes

**None.** `lumenairy/elements/_lens_traced.py` and
`lumenairy/propagators/carrier.py` were read only; C5/C6 and every other
constant were toggled as module attributes inside `try/finally` at runtime.
`CHANGELOG.md` and `lumenairy/elements/pmm/**` were not touched.
