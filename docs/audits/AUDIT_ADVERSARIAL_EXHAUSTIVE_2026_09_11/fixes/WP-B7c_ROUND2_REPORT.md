# WP-B7c round 2 -- the PIXEL-HALVING ARBITER, and the claims round 1 could not support

Wave 5 item C, round 2, answering `fixes/VERIFY_WP-B7c.md`.  Branch
`fix/wp-b7c-round2` off `verify/wp-b7c` (`250421ed`), so the verification's
own gates are inherited.  Worktree `C:/tmp/lum_mb2`; the round-1 tree
`C:/tmp/lum_mb2_ref` (a `git archive` of `verify/wp-b7c`) and the audit base
`C:/tmp/lum_mb2_pre` (`96cb2096`).

| build | interpreter | numpy | tree |
|---|---|---|---|
| **win** | Windows py3.14.6 | 2.4.4 | `C:/tmp/lum_mb2` |
| **wsl** | WSL py3.12.3 | 2.4.6 | `/mnt/c/tmp/lum_mb2` |

Probes and their JSON: `validation/probe_wp_b7c_round2/`.  Decision tests:
`tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py` (12 ids, 9.2 s).

---

## 0. What the verification established, and what this round does about it

VERIFY-WP-B7c **CONFIRMED** the mechanism WP-B7c diagnosed -- the branch sum's
point-sampled area quadrature losing unbiasedness when a ring collapses onto a
handful of pixels -- and made the case more strongly than WP-B7c did, with a
control WP-B7c had not run: hold the optic, the plane, the LAUNCH LATTICE and
the physical window fixed, halve the OUTPUT PIXEL, and the excess divides by
exactly 4 (S 107 -> 27.3 -> 7.40; M 504 -> 127 -> 32.2; Q 11238 -> 2810 ->
703; F 2.98 -> 1.42 -> 1.04) while a healthy plane is invariant
(0.937 -> 0.937 -> 0.938), identically on both builds.

It **REFUTED** the guard built on that diagnosis.  The shipped bar reads the
deposited power against the LAUNCHED power, and on a wider optic set its
accepted and broken populations overlap (largest accepted 0.9804, smallest
broken 1.106 / 1.151, a gap of 1.13x under a 2.0 bar), the interval reported
as empty is populated continuously, D1's plane returns a wrong field under
healthy diagnostics 1.74x inside the bar, and D2's decision follows the output
grid rather than the field.

**This round makes the control the decision.**  The reading is

    pixel_continuity = p_out(dx) / p_out(dx / 2)

taken over the same physical window from the same launch lattice.  A converged
point-sampled quadrature deposits the same total power at any pitch -- its
expectation is the launched energy and does not contain `dx` -- so this reads
**1 exactly**, on any optic, at any plane, at any grid.

> **CORRECTED 2026-09-19 (WP-B7c round 3, R3-5; claim 1b of
> `VERIFY_WP-B7c_ROUND2.md`).**  It does not read 1 exactly, for a structural
> reason: the two renders' pixel lattices do not have the same Voronoi hull,
> so where light reaches the window edge they are not integrating the
> identical rectangle (VERIFY round 2, E5).  Measured on **102 planes over
> seventeen optics** chosen by a geometric criterion -- at least a quarter of
> the paraxial focal distance short of the interior fold, so no reading
> selects the planes its own spread is measured on -- a converged render reads
> **0.9994103 .. 1.0004414**, rms 1.48e-4.  A window ladder isolates the
> mechanism: at a fixed pitch the reading is 1.4e-3 from 1 on a window too
> small to contain the field and 1.2e-4 on every larger one, bit for bit.  And
> the reference does NOT hold where the guard works: on healthy FOLD planes
> the same reading spreads by up to 2.1e-2, 36x more.  See
> `WP-B7c_ROUND3_REPORT.md` section 4 and
> `_lens_traced_multibranch._HALF_PITCH_CENTRE_OFFSET`.  A quadrature that has
stopped being unbiased deposits a power proportional to the PIXEL AREA and
reads ~4 per halving.  That fixed reference is what the launched-power ratio
never had, and it is why a bar on this quantity is a tolerance on a known
value rather than a boundary between two moving populations.

---

## 1. The fixtures

Eight optics.  VERIFY-B7b's own fold fixture (which WP-B7c derived on) and
VERIFY-WP-B7c's four oracle-valid ones are carried verbatim, so the round-2
population CONTAINS the population the round-1 bar was refuted on; three are
new to the campaign.

| id | prescription | lambda | grid | `y_max/z` | why |
|---|---|---|---|---|---|
| **V** | N-BAF10 biconvex R = +/-2.6 mm, 0.90 mm ap. | 1.064 um | 512 x 2.20 um | 0.133 | VERIFY-B7b's fixture; WP-B7c's own blow-up ladder |
| **S** | N-SK16 plano-convex CONVEX-first R = +2.4 mm | 532 nm | 512 x 1.50 um | 0.133 | VERIFY-WP-B7c |
| **M** | N-SF6 positive MENISCUS R = +1.6/+4.2 mm | 2.0 um | 512 x 3.60 um | 0.200 | VERIFY-WP-B7c |
| **Q** | CEMENTED doublet N-BK7 / N-SF6, 3 surfaces | 1.31 um | 512 x 2.60 um | 0.108 | VERIFY-WP-B7c; OVERCORRECTED SA |
| **F_alt** | N-LASF9 biconvex stopped to f/2.0 | 633 nm | 640 x 1.40 um | 0.231 | VERIFY-WP-B7c; carries D1 and D2 |
| **A** | **AIR-SPACED** doublet N-BK7 + / N-SF11 -, 4 surfaces | 1.064 um | 512 x 3.00 um | 0.151 | NEW -- a real air gap; two glass-air boundaries between the elements |
| **K** | N-LAK22 biconvex with a **CONIC** first surface (k = -0.55) | 1.31 um | 512 x 2.00 um | 0.222 | NEW -- every published fixture is all-spherical; a conic partly corrects the SA, so the fold is much tighter |
| **G** | N-SF11 biconvex stopped to **NA 0.33** | 850 nm | 768 x 1.20 um | 0.354 | NEW -- above the 0.12-0.29 envelope every published fixture sits in, and above the 0.3 the brief asks for |

Prescriptions in `validation/probe_wp_b7c_round2/fixtures.py`; `P` (f/1.2)
stays excluded for the reason VERIFY-WP-B7c gives and is reported only on the
oracle-side axes of section 7.

---

## 2. What shipped

### 2.1 The reading, and where it is taken

`_lens_traced_multibranch` factors its rasteriser into a nested `_render(N_r,
dx_r)` parameterised by the OUTPUT sampling alone -- the trace, the
triangulation, the mapped areas and the degeneracy census are properties of
the launch lattice and the optic, computed once and closed over.  The public
`apply_real_lens_traced_multibranch` calls it once with `(N, dx)`, exactly as
before; a new module-private `_multibranch_render(..., pixel_halving_arbiter)`
calls it a second time with `(2N, dx/2)` and hands the half-pitch render out
in the diagnostics.

`apply_real_lens_traced_uniform` is the only caller that asks.  It completes
the half-pitch render with the **same** fold parameters (`r_c`, `kappa`, the
cone phase and the two fitted CFU coefficients) and reads the continuity of
the field it is about to RETURN -- the completed fold field where the
completion applies, the plain branch sum on every fallback path.

**Reading the branch sum instead would refuse fields that are right**, and
this is measured, not argued: the CFU swap rewrites the fold band and the
whole dark side, which is exactly where a modest branch-sum excess sits.  On
VERIFY-B7b's own fixture at z = 1761 um the branch sum reads **1.0636** while
the completion built on it reads **1.0020**, and that completed field's oracle
fidelity is 0.9878 with its power 0.994x the oracle's.  Over the fold-ring
population the branch-sum reading spans 0.9419 .. 1.0636 on ACCEPTED planes
where the completion-level reading spans 0.9860 .. 1.0221 -- on the same
planes, the widest departure from 1 falls from **6.4 % to 2.2 %**, a factor of
2.9, and that is the whole margin the bar has to live in.

Re-using the coarse fit on the fine grid is deliberate: it isolates the
RASTERISATION, which is what the control measures, instead of adding a second
least-squares fit's own noise to the comparison.

### 2.2 The decision

| decision | when | what happens |
|---|---|---|
| `ok` | `1/1.06 <= C <= 1.06` | returned |
| `not_converged_gain` | `C > 1.06` | **REFUSED**, naming both readings, the branch count and a member that works |
| `not_converged_loss` | `C < 1/1.06` | REPORTED; the field is returned |
| `not_measured` | fine grid past `_ARBITER_MAX_FINE_ENTRIES`, or a zero render | reported as such, never a silent `None` |
| `not_requested` | the public branch sum, which never asks | -- |

The launched-power tripwire (`_MB_POWER_RATIO_MAX = 2.0`) is kept as the first,
free arm and fires first where both would; it costs nothing and catches the far
tail before the fold trace runs.

**Why the loss arm reports and does not refuse** -- the asymmetry is physical.
The completion keeps the branch sum's BRIGHT side verbatim, so a bright-side
excess reaches the caller's field; the DARK side is exactly what the completion
replaces, so a dark-side deficit does not.  Measured on the air-spaced doublet
at z = 2903 um: the branch sum reads 0.9419 (6 % short) while the completed
field's power is 0.997x the oracle's and its fidelity 0.9910.  Over the whole
round-2 population no fold-ring plane is refused on this arm and none needs to
be; the only loss-side readings that correspond to a wrong field are on
FALLBACK planes, and there the nearest returned reading (0.854, fidelity 0.904)
and the nearest wrong one (0.726, fidelity 0.837) leave 1.18x on ONE optic --
not a population a bar can be derived on.

### 2.3 Diagnostics

On the completion: `pixel_continuity`, `pixel_continuity_of`,
`pixel_continuity_band`, `pixel_continuity_decision`,
`multibranch_pixel_continuity` (the branch sum's own reading, carried
bit-for-bit).  On the branch sum: `pixel_continuity`,
`pixel_continuity_band`, `pixel_continuity_decision`, `pixel_halved_power`,
`pixel_halved_field`, `grid_power`.

### 2.4 One seam, and why five existing tests moved with it

The completion now reaches the branch sum through the module-private
`_multibranch_render` rather than the public
`apply_real_lens_traced_multibranch`, because it asks for a reading the public
signature does not carry.  Five existing tests inject a synthetic multibranch
field through that call to drive the cusp and higher-catastrophe routes
(`test_niche_r2_pearcey_cusp.py::test_uniform_routes_cusp_to_pearcey` and
`::test_cusp_fallbacks_finite`;
`test_niche_r5_gbd_vector_catastrophe.py::test_a4_higher_catastrophe_routes_finite_named`
at `n_turn` 3 / 4 / 5), and they were re-pointed at the new name.

The failure that found them was LOUD, and deliberately so: with only ONE name
reachable from the completion module, patching the old one raises
`AttributeError`.  Re-importing the public name here as an alias would have
kept those five green while silently making them test nothing, which is the
worse outcome -- so the single-seam property is now pinned by
`..._the_branch_sum_is_reached_through_exactly_one_seam`, which asserts that
the private name exists, that the public one is NOT reachable from this
module, and that patching the private one takes effect with
`pixel_halving_arbiter=True`.

Proof of patch: `test_niche_r2_pearcey_cusp.py` +
`test_niche_r5_gbd_vector_catastrophe.py` **25 passed** (119 s) after the
re-point, against 5 failed before it; each of the five asserts a route
(`reason == 'cusp_ring'`, or the named higher-catastrophe warning) that only
the injected field can produce, so a patch that did not take effect would fail
them.

---

## 3. The bar, derived

`probe_wp_b7c_round2/scan.py` on HEAD (the reading and the decision; at a
refused plane the reading is read back from the refusal message, which names
it) joined by `summarise.py` to `arbiter.py` on the audit base `96cb2096`
(which refuses nothing, so every plane yields a field to score).  **82
oracle-scored fold-ring planes on 8 optics**, plus 22 fallback planes.

### 3.1 The populations -- the guard's own split, with no accept bar chosen

| population | n | continuity | oracle fidelity of the field |
|---|---|---|---|
| **RETURNED** | **67** | **0.9860 .. 1.0221** | **0.9593 .. 0.9985** |
| **REFUSED** | **15** | **1.092 .. 3.998** | **0.0173 .. 0.9302** |

**The two fidelity populations do not overlap.**  That statement chooses no
accept criterion: the split is the guard's own, and the fidelities fall out of
it.  Round 1's reading cannot make it -- its accepted and broken populations
overlap at every bar.

> **CORRECTED 2026-09-19 (WP-B7c round 3, R3-5; claim 1d of
> `VERIFY_WP-B7c_ROUND2.md` and E1).**  They overlap.  The round-2
> verification narrowed the separation from (0.9593, 0.9302) here to (0.9747,
> 0.9578) on 124 fold-ring planes over seven optics and warned it was
> narrowing; on **394 fold-ring planes over sixteen optics** it closes -- the
> worst RETURNED field scores **0.9421** (an NA 0.41 plano-convex at
> z = 2069.3 um, reading 1.0133, carrying 1.27x the oracle's energy) against a
> best REFUSED of **0.9520** (a plano-first N-BAK4 at z = 4929.0 um, reading
> 1.0812).  It is not a knife edge: that optic's WHOLE fold ring is returned
> at fidelity 0.9421-0.9564 with 1.24x-1.27x the oracle's power.  So the
> guard's own split no longer separates right fields from wrong ones without
> an accept criterion, and choosing that criterion is a maintainer decision.
> `WP-B7c_ROUND3_REPORT.md` section 2.

The extremes, which is where a bar of this width lives or dies:

* worst RETURNED: S z = 3230 um, C = 0.9981, fidelity 0.9593, power 1.0285;
  then F_alt z = 1073 um, C = **1.0221**, fidelity 0.9742, power 1.0506;
* best REFUSED: F_alt z = 1074 um, C = **1.092**, fidelity 0.9302, power
  1.1538; then G z = 972 um, C = 1.142, fidelity 0.9184, power 1.2059.

### 3.2 The bar

Gap between the largest returned reading (1.0221) and the smallest refused one
(1.092): **1.0683x**, geometric centre **1.0564** -- **1.06 to three figures**,
which is the shipped `_PIXEL_CONTINUITY_MAX`.

**Margins: 1.037x above the largest returned reading, 1.030x below the
smallest refused one.**  That is a 7 % gap, not decades, and it is stated
rather than papered over.  What makes it usable where round 1's was not:

> **CORRECTED 2026-09-19 (WP-B7c round 3, R3-5; claim 1c of
> `VERIFY_WP-B7c_ROUND2.md` and E1).**  The gap is a reading of the z LADDER,
> not a property of the quantity, and neither 1.0683x nor the verification's
> 1.0376x is a measurement of the bar.  The reading is not smooth in z near a
> fold onset (the verification measured 1.0002 -> 1.1810 -> 0.9989 over 20 nm
> of defocus), so refining the ladder keeps finding readings closer to the
> bar.  Scanning the SAME sixteen optics three times, each ladder 10-30x
> finer: the fold-ring gap reads **1.371x at 468 planes, 1.032x at 882 and
> 1.0045x at 1304**, with the two-sided margin falling 1.040x -> 1.012x ->
> **1.00047x** and no sign of a floor.  The bar is kept at 1.06 -- which is
> the geometric centre (1.0600253) of its own gap on that population, to three
> figures -- but it is derived from what it COSTS and not from a gap: at an
> accept criterion of fidelity 0.95 it costs 1 false refusal and 2 misses over
> 394 fold-ring planes, against 5 false refusals for 1.04 and 4 more misses
> for 1.08.  `WP-B7c_ROUND3_REPORT.md` sections 3 and 5.

* the quantity has a FIXED reference (a converged render reads exactly 1), so
  the bar is a tolerance, not a boundary between two moving populations;
* the gap EXISTS.  Round 1's populations overlap: no value of that constant
  separates them.

### 3.3 Confusion, at both accept criteria

WP-B7c's own accept bar was fidelity >= 0.883 -- the minimum of ITS accepted
cluster, which was bimodal.  This population is continuous, so both that bar
and a stricter 0.95 are reported.

| population | accept bar | refused-broken | **FALSE refusals** | misses | returned-accepted |
|---|---|---|---|---|---|
| fold-ring (82) | fid >= 0.883 | 12 | **3** | 0 | 67 |
| fold-ring (82) | fid >= 0.95 | 15 | **0** | 0 | 67 |
| all planes (104) | fid >= 0.883 | 20 | **5** | 8 | 71 |
| all planes (104) | fid >= 0.95 | 25 | **0** | 12 | 67 |

> **CORRECTED 2026-09-19 (WP-B7c round 3, R3-5; claim 5 of
> `VERIFY_WP-B7c_ROUND2.md`).**  "0 false refusals at fidelity >= 0.95" is a
> property of this 82-plane population.  The round-2 verification found 2 of
> 124; round 3 finds **1 of 394** on the fold ring (`W_alt` z = 4929.0 um,
> reading 1.0812, fidelity 0.9520) and **2 misses** at the same criterion, and
> over all 1304 planes 1 false refusal and 265 misses -- 263 of the misses on
> the FALLBACK route, which this arm does not claim to arbitrate.  The full
> confusion by fidelity band, for 1.04 / 1.06 / 1.08 and three bars derived
> from the converged reading's own spread, is `WP-B7c_ROUND3_REPORT.md`
> section 5.

The five refusals that are "false" at the 0.883 bar, named:

| plane | C | fidelity | power / oracle | route |
|---|---|---|---|---|
| F_alt z = 1074 um | 1.092 | 0.9302 | **1.154** | fold ring |
| F_alt z = 1075 um | 1.139 | 0.9026 | **1.214** | fold ring |
| G z = 972 um | 1.142 | 0.9184 | **1.206** | fold ring |
| M z = 2300 um | 1.071 | 0.9000 | **0.806** | fallback (`too few bright-band pixels`) |
| S z = 3260 um | 1.065 | 0.8890 | **0.816** | fallback (`fold Airy scale under-resolved`) |

Every one carries a 15-21 % ENERGY error against the oracle, and the two
fallback planes are on ladders whose immediate neighbours are broken by any
bar, on a route where the module has already warned that the completion does
not apply.  Whether they are "false" is exactly the choice of accept bar, and
that is why both are reported.

### 3.4 What the arbiter does NOT catch

On FALLBACK planes -- where the field returned is the bright-side-only branch
sum -- the returned and refused fidelity populations DO overlap (returned down
to 0.764, refused up to 0.930), because what is wrong there is the missing
dark-side tail and not the quadrature.

> **CORRECTED 2026-09-19 (WP-B7c round 3, R3-4; E7).**  "Returned down to
> 0.764" is a property of this population's 22 fallback planes.  Over the
> **476 fallback planes the shipped bars RETURN**, on sixteen optics, the
> range is **0.5358 .. 0.9999** with 263 below fidelity 0.95 and 152 below
> 0.883.  A reading that orders that population DOES exist and is already
> reported: the LOSS side of this arm flags 192 planes, every one of them
> wrong, with zero false alarms, and the launched-power bracket below 0.889
> flags 254 with none false and only 9 wrong ones missed.  Their bars are set
> for the fold-ring route, where the completion replaces the dark side -- an
> argument that is false on the fallback route, where the deficit reaches the
> caller verbatim.  Thirteen returned planes are left that nothing sees, the
> worst at fidelity 0.5358.  `WP-B7c_ROUND3_REPORT.md` section 7.  **This arm is a detector of ONE
failure mode; a plane it accepts is not thereby certified.**  Twelve such
planes are misses at the 0.95 bar and they are listed in
`joined_head_win.json`.

### 3.5 Two grids

Every optic is also read on a second grid (`*_alt`, N and dx both changed) in
the bit-identity matrix of section 6, and the D2 ladder of section 4.2 is the
two-grid question asked cleanly -- optic, plane, launch lattice and window all
held, only `dx` changed.

---

## 4. The targets

### 4.1 D1 -- REFUSED

VERIFY-WP-B7c's single sharpest counterexample, `F_alt` z = 1076 um:
`reason='fold_ring'`, `fell_back=False`, `zeta_extrapolation=1.929`,
`power_ratio_decision='ok'`, bracketed launched-power reading 1.1513 -- and a
returned field at oracle fidelity **0.8580** carrying **1.284x** the oracle's
power.

**It now reads a continuity of 1.185 and is REFUSED.**  So are the three
doublet planes of the same class: Q z = 5400 / 5410 / 5420 um read 1.100 /
1.265 / 1.636 at fidelity 0.837 / 0.772 / 0.666, every one of which round 1
returned with `power_ratio_decision == 'ok'`.

### 4.2 D2 -- the decision no longer follows the grid

Round 1, on `F_alt` z = 1080 um: bracketed reading **2.982 at dx = 1.40 um
(REFUSED)**, **1.420 at dx = 0.70 um (ACCEPTED)**.  One halving of the pixel
removed the guard, in the direction this module's own `l_airy` gate tells a
caller to refine.

| dx [um] | launched-power reading | round 1 | continuity | round 2 |
|---|---|---|---|---|
| 1.40 | 2.982 | REFUSED | -- | REFUSED (power arm) |
| 0.70 | 1.420 | **accepted** | **1.346** | **REFUSED** |

The reading still falls by ~4 per halving -- that is the mechanism, and the
test still asserts it -- but the decision is a RATIO of two renders one
halving apart, taken at the caller's own pitch, so it does not fall with it.

The companion case is the one that shows the guard tracking the render rather
than punishing refinement: `F_alt` z = 1076 um is refused at dx = 1.40 um
(C = 1.185) and RETURNED at dx = 0.70 um with C = 1.0015 and a bracketed
reading of 0.971 -- there the finer render really has converged, and the guard
says so.

`_ARBITER_MAX_FINE_ENTRIES` was set to 7e6 (fine grid up to N = 2644, i.e.
caller grids to N = 1322) for exactly this reason: a cap at N = 1024 would
have stopped reporting on the refined grid of a 640-grid fixture and
reproduced D2 as a hole in the measurement.

### 4.3 D3 -- the refusal's mechanism sentence

The message's mechanism clause is now conditioned on the plane's own
`n_branch_max`.  With three or more branches on a pixel it names the ring
coalescence; with one or two it says the map is NOT multi-valued, names the
degenerate-triangle census, and points at `['pixel_continuity']`.  The doublet
at z = 5.460 mm (`n_branch_max == 1`) no longer prints "up to 1 branches on
one pixel" and "a whole RING of branches coalesces" in consecutive clauses.

### 4.4 D4 -- the bracket's price, stated

`_MB_POWER_RATIO_MAX`'s comment now carries it: on the geometry the bracket
exists for, the two denominators separate by up to **7.85x** (20 planes of the
delta audit's D3 fixture, `power_ratio` 2.02-3.76 against
`power_ratio_triangles` 0.45-0.90), so that arm's detection floor is the bar
times that spread -- up to ~15.7x.  The bracket is still right; the continuity
arm has no such denominator at all.

### 4.5 D5 -- the oracle's NA ceiling, measured

See section 7.

---

## 5. Restatements

| claim | was | now |
|---|---|---|
| `_MB_POWER_RATIO_MAX`'s band | "accepted 0.816-1.246, smallest broken 5.848, nothing between, margins 1.60x / 2.92x in a 4.69x gap" | scoped to WP-B7c's five undercorrected singlets, with VERIFY-WP-B7c's overlap (0.9804 against 1.106 / 1.151) recorded beside it and the constant described as **a far-tail tripwire with a measured margin below (2.04x) and none above, not a classifier**.  Left at 2.0: lowering it toward 1.106 starts refusing fields the oracle accepts at 0.9804 |
| `_ZETA_EXTRAPOLATION_MAX` | "a SIGN boundary: signed and centred below, one-sided gain above" | only the ABOVE-bar half, which reproduces (0 of 13 rungs negative on four more optics, mirroring 0 of 8).  "Signed and centred below" is dropped: 1 of 17 negative, mean +7.00 %, and the largest excursion in either study (+28.4 %) is BELOW the bar.  `sqrt(5.653 * 11.05) = 7.90` is kept as the provenance of where 8.0 came from, not as a two-sided calibration |
| `test_audit2609_b7c_multibranch_envelope.py`'s two literal pins | `largest_accepted = 1.246` / `smallest_broken = 5.848` asserted as an envelope with 1.5x / 2.5x margins | replaced by the measured overlap and the SHAPE of the constraint: the populations do not separate (`smallest_broken / largest_accepted < 1.25`), the bar has a margin below and none above, and the arm that decides inside it is the continuity one.  The zeta test is renamed `..._is_the_onset_of_a_one_sided_gain` and pins the surviving half |
| `test_verify_b7c_multibranch.py` D1 / D2 / D3 | three defects, pinned as defects | the same three, restated as the DECISIONS that replaced them (refused / still refused at half the pitch / the message no longer claims a ring), each keeping the invariant arm |
| `caustic_fold_truth.py`'s `J0` note | "exact to O((y rho / R0^2)^2 k) which is < 1e-2 rad here" | a measured NA ceiling with the term named, the envelope it holds over, and the exact-azimuth control that settles it (section 7) |

---

## 6. Bit identity

`bitid.py`: a 30-fixture matrix spanning, for each of P / Q / M / S / F / V, a
healthy fold plane, a fallback, the exit vertex (`output_plane_distance = 0`),
the branch sum alone, `caustic_band='plain'`, `ray_subsample=4`, a complex64
input, the alternate grid of every optic, and the planes round 1 refused.
A CHILD process per tree with `cwd` and `PYTHONPATH` set to it and
`lumenairy.__file__` ASSERTED under it; SHA-256 over `ndarray.tobytes()`.

`git archive verify/wp-b7c` (round 1) against HEAD:

**26 identical, 0 moved, 4 newly refused.**

| newly refused | round-1 grid power | round-2 |
|---|---|---|
| `Q_uni_gap` (z = 5400 um, fidelity 0.837) | 3.945e-07 W | `RuntimeError` |
| `Q_uni_gap2` (z = 5412 um, fidelity 0.772) | 4.822e-07 W | `RuntimeError` |
| `Q_uni_gap3` (z = 5420 um, fidelity 0.666) | 6.514e-07 W | `RuntimeError` |
| `F_uni_d1` (z = 1076 um, fidelity 0.858) -- **D1's plane** | 9.352e-08 W | `RuntimeError` |

Every plane the arbiter accepts is byte-identical to round 1; only newly
refused planes change, and all four are fields the oracle scores at 0.666-0.858
where the ray-to-wave hand-off at the same plane scores 0.998.

The rasteriser refactor was verified byte-identical on the same matrix BEFORE
the arbiter was added (30 of 30, `bitid_ref_win.json` against
`bitid_refactor_win.json`), so the extraction of `_render` and the split of
the entry point are separately established not to have moved anything.

---

## 7. The oracle's NA ceiling (D5), measured rather than bounded

`validation/probe_wp_b7c_round2/oracle.py` adds an EXACT azimuthal quadrature
beside the shared oracle's Debye `J0` form.  The two differ in exactly one
term, so their difference IS the ceiling.

The azimuthal integrand oscillates like `exp(-i b cos phi)` with
`b = k y rho / R0`, and `b` runs from 0 on the axis to ~1000 rad at the corner
of a 512-pixel grid -- which is why the shared oracle does this integral in
closed form.  The node count is therefore chosen PER RHO BLOCK from that
block's own `b` (the midpoint rule aliases onto `sum_m J_{m n_phi}(b)`, so
`n_phi` must clear `b` by a few `b^{1/3}`), and the exact quadrature is applied
inside the radius holding 99.95 % of the field's energy, with the `J0` form
kept outside it.  Both the radius and that fraction are reported.

FLOOR AND CEILING, `oraclefloor_win.json`:

| optic | plane | `y_max/z` | rms radius | `eps` = `k y^2 rho^2 / 2 R0^3` | core (99.95 % of the energy) | Debye vs exact: rel L2 | fidelity | power | `J0` closure |
|---|---|---|---|---|---|---|---|---|---|
| Q (doublet) | 5680 um | 0.108 | 13.25 um | **0.0009 rad** | 143.0 um | 0.0003 | 1.00000 | 1.00008 | 0.99999 |
| S (plano-cx) | 3214.78 um | 0.133 | 6.58 um | 0.0014 | 43.4 um | 0.0005 | 1.00000 | 0.99990 | 0.99982 |
| A (air doublet) | 2800 um | 0.151 | 18.40 um | 0.0079 | 111.0 um | 0.0029 | 0.99999 | 0.99958 | 0.99953 |
| M (meniscus) | 2201.74 um | 0.200 | 20.19 um | 0.0110 | 154.8 um | 0.0042 | 0.99996 | 0.99924 | 0.99921 |
| K (conic) | 2000 um | 0.222 | 21.76 um | 0.0259 | 101.5 um | 0.0111 | 0.99988 | 0.99886 | 0.99881 |
| V (N-BAF10) | 1750 um | 0.224 | 11.08 um | 0.0096 | 95.0 um | 0.0036 | 0.99995 | 0.99946 | 0.99938 |
| F_alt (f/2.0) | 1056 um | 0.231 | 11.91 um | 0.0328 | 60.3 um | 0.0139 | 0.99980 | 0.99880 | 0.99875 |
| **G (NA 0.33)** | 972 um | 0.346 | 21.48 um | **0.1769** | 38.1 um | **0.0629** | **0.99705** | 0.99670 | 0.99676 |
| **G (NA 0.33)** | 950 um | 0.354 | 24.21 um | **0.2388** | 44.3 um | **0.0875** | **0.99397** | 0.99611 | 0.99616 |
| **P (f/1.2)** | 888 um | 0.452 | 40.66 um | **1.4231** | 66.5 um | **0.5371** | **0.79403** | 0.99007 | 0.98990 |

Convergence of the exact arm, every row: doubling its azimuthal safety factor
moves it by **1.7e-14 .. 1.3e-13** and doubling `n_fan` by **6.0e-05 ..
3.6e-04**, so the Debye-vs-exact column is the `J0` truncation and nothing
else.  The rule that falls out is **rel L2 ~ 0.37 `eps`**, holding over three
decades of `eps`.

> **CORRECTED 2026-09-19 (WP-B7c round 3, R3-5; claim 6b of
> `VERIFY_WP-B7c_ROUND2.md` and E6).**  The column and the rule are measured
> inside the 99.95 %-energy CORE, outside which the two compared fields are
> identical by construction while the dropped quadratic term is largest.
> Re-measured with the exact azimuth at EVERY radius, pointwise at a
> radius-stratified importance-weighted sample of the output grid's own pixels
> (no radial reconstruction on either side): the doublet row reads **0.0563**
> where this table reads 0.0003 and where the rule predicts 3.2e-4 -- 178x --
> and over three decades of `eps` the full-radius error moves by a factor of
> ten and is not monotone in `eps`.  The rule is deleted from
> `validation/oracles/caustic_fold_truth.py` rather than refitted.  The
> FIDELITY conclusion survives and is the one the studies use: 5e-4 .. 4.5e-3
> below `y_max/z` 0.21, 7.7e-3 at 0.354, 3.65e-2 at 0.461.  The energy-closure
> column of this table is CONFIRMED (0.99992 / 0.99913 / 0.99616 / 0.99222),
> against the round-2 verification's E6, which read 0.9972 / 0.9741 from its
> own 256-radius reconstruction -- that loses energy on both arms equally
> (0.894 on each).  `WP-B7c_ROUND3_REPORT.md` section 8.

Three consequences:

* **no published number is affected.**  Every fixture in WP-B7b, VERIFY-B7b,
  WP-B7c and VERIFY-WP-B7c sits at `y_max/z` 0.108-0.231, where the `J0` form
  costs under 1.5 % in relative L2 and under 3e-4 in fidelity -- inside those
  studies' own sampling floors;
* **NA 0.33 needed the exact arm, and got it.**  At `G` the `J0` form costs
  6-9 % in relative L2 and 3e-3-6e-3 in fidelity, which is larger than the
  fidelity differences this study reads, so `G`'s 11 planes are scored against
  the exact azimuthal quadrature and not the `J0` form.  That is how the
  brief's "NA above 0.3 with an oracle that holds there" is met rather than
  waived;
* **VERIFY-WP-B7c's D5 understated it, and misattributed it.**  At f/1.2 the
  `J0` form is wrong by **54 % in relative L2 and a fidelity of 0.794**.  D5
  recorded that the library's own `uniform` and `wave` members disagree with
  the oracle at fidelity 0.737 / 0.722 there and called it a model gap in the
  library.  It is THIS ORACLE'S gap: the members were closer to the truth than
  the thing scoring them.  `caustic_fold_truth.py` now carries the ceiling,
  the `eps` formula, this table and the instruction not to score against the
  `J0` form above NA ~0.3, and notes that its own `energy_closure` column sees
  the same thing from the other side (0.99999 at `y_max/z` 0.108 falling to
  0.98990 at 0.452).

The exact arm's own convergence is reported beside every row (doubling the
azimuthal safety factor and doubling `n_fan`), so the Debye-vs-exact
difference can be attributed.

---

## 8. Cost

One extra rasterisation of the SAME mapped triangles -- never a second ray
trace, a second KMAH / Jacobian pass or a second meridional fold trace.  That
is asserted structurally, not by wall clock:
`..._costs_one_rasterisation_and_not_a_second_trace` counts calls to
`_trace_launch_grid` and requires exactly one with the arbiter on and with it
off.

Measured (`cost.py`, best of 3, round-1 tree against HEAD, same box, other
jobs running so the spread is generous):

| fixture | plane | N | round-1 `uniform` | round-2 `uniform` | ratio | round-2 `multibranch` |
|---|---|---|---|---|---|---|
| V | z = 1758 um (fold ring) | 512 | 0.319 s | 0.715 s | **2.24x** | 0.296 s |
| V | z = 0 (exit vertex, near-identity map -- the worst case for the extra render) | 512 | 0.558 s | 1.280 s | **2.29x** | 0.297 s |
| S | z = 3214.78 um | 512 | 0.930 s | 1.978 s | 2.13x | 0.802 s |
| F_alt | z = 1063 um | 640 | 0.526 s | 0.944 s | 1.79x | 0.365 s |
| G | z = 950 um | 768 | 1.484 s | 1.650 s | **1.11x** | 0.742 s |
| A | z = 2800 um | 512 | 0.264 s | 0.495 s | 1.88x | 0.202 s |
| K | z = 2000 um | 512 | 0.389 s | 1.153 s | 2.96x | 0.322 s |
| Q | z = 5680 um | 512 | 0.584 s | 0.983 s | 1.68x | 0.590 s |

So **1.1x to 3.0x of the completion, typically ~2x**, and the branch sum
unchanged (0.20-0.80 s against the round-1 tree's 0.27-0.86 s, which is the
timing noise of a box with other jobs on it).  The spread across optics is the
map's compression: where the triangles are already sub-pixel the half-pitch
render costs what the coarse one does, and where the map is near-identity
(the exit vertex) it costs four times as much.

`apply_real_lens_traced_multibranch` never asks for the reading and is
unchanged in cost and in bits.

---

## 9. Runs

All from `cd /c/tmp/lum_mb2`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, `--capture=sys`, and `PYTHONPATH` pinning the tree.

| run | build | result | duration | log |
|---|---|---|---|---|
| `b7c2_pixel_halving_arbiter` alone | win py3.14.6 | **12 passed** | 9.2 s | -- |
| `b7c2` + `verify_b7c` + `b7c` + `b7_asymptotic` + `b7b_caustic_routing` + `niche_k4_uniform_caustic` + `niche_audit_w9_dispatch2` + `a3_caustic_siblings` + `a3_verify_traced` | win | **195 passed** | 387 s | `pytest_core_head_win.txt` |
| `b7c2` + `verify_b7c` + `b7c` + `b7_asymptotic` + `b7b_caustic_routing` + `niche_k4_uniform_caustic` + `niche_audit_w9_dispatch2` | **wsl py3.12.3** | **151 passed** | 261 s | `pytest_core_head_wsl.txt` |
| `niche_r2_pearcey_cusp` + `niche_r5_gbd_vector_catastrophe` (the re-pointed seam) | win | **25 passed** | 119 s | -- |
| `a16` config bit-identity + round-trip + verify, `niche_k1` / `r2` / `r5` / `w4`, `v5_21` delta audit + lens accuracy | win | **473 passed**, 2 deselected | 298 s | `pytest_a16_niche_win.txt` |
| `-k census` (whole suite) | win | **45 passed**, 16205 deselected | 726 s | `pytest_census_win.txt` |
| `-k walker` (whole suite) | win | **118 passed, 6 skipped**, 16126 deselected | 57 s | `pytest_walker_win.txt` |
| `-k dispatcher_pin` (whole suite) | win | **510 passed, 5 skipped**, 15735 deselected, 7 warnings | 59 s | `pytest_dispatch_win.txt` |
| `test_public_api.py` + `test_v4_16_2_dispatcher_pin_doc_consistency.py` + `test_audit_except_budget.py` | win | **19 passed** | 12.8 s | `pytest_publicapi_win.txt` |
| `ruff check lumenairy/ tests/ validation/probe_wp_b7c_round2/` | wsl | **All checks passed** | -- | -- |

No red anywhere, on either build.  WP-B7c's environmental red
(`test_installed_metadata_version_matches_source_version`) does not reproduce
here either; the editable install reads 5.47.0.

`.test_durations` carries the twelve new ids with their measured values
(1.79 / 1.49 / 1.27 / 0.94 / 0.84 / 0.81 / 0.55 / 0.43 / 0.43 / 0.35 / 0.14 /
0.01 s; **9.05 s for the file**, inside the 60 s budget), the renamed zeta
pin, and the three restated verify ids re-timed; the file is JSON-validated
after the edit (16 212 -> 16 214 ids).

```
python -m pytest tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py \
  tests/unit/test_verify_b7c_multibranch.py \
  tests/unit/test_audit2609_b7c_multibranch_envelope.py \
  tests/unit/test_audit2609_b7_asymptotic.py \
  tests/unit/test_audit2609_b7b_caustic_routing.py \
  tests/unit/test_niche_k4_uniform_caustic.py \
  tests/unit/test_niche_audit_w9_dispatch2.py \
  tests/unit/test_audit2609_a3_caustic_siblings.py \
  tests/unit/test_audit2609_a3_verify_traced.py -q --capture=sys
python -m pytest tests/ -q --capture=sys -k census
python -m pytest tests/ -q --capture=sys -k walker
python -m pytest tests/ -q --capture=sys -k dispatcher_pin
python -m pytest tests/unit/test_public_api.py \
  tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
  tests/unit/test_audit_except_budget.py -q --capture=sys
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mb2 && ~/lumvenv/bin/ruff check lumenairy/ tests/ validation/probe_wp_b7c_round2/'
```

---

## 10. What could not be measured

* **whether the 7 % gap survives a ninth optic.**  The bar's margins are
  1.037x / 1.030x on eight optics, and a ninth landing inside the gap would
  move it.  The quantity's fixed reference (a converged render reads exactly
  1) is what keeps that a re-derivation rather than a collapse;

  > **ANSWERED 2026-09-19 (WP-B7c round 3).**  It does not, and the reason is
  > not a ninth optic: the gap is a reading of the z ladder (1.371x / 1.032x /
  > 1.0045x over three refinements of the SAME sixteen optics), and the fixed
  > reference is 1 only to O(1/N) and only far from the caustic.  The bar
  > survives as a COST optimum, not as the centre of a gap.
* **the population on the second build.**  The 104-plane oracle-scored
  population was measured on win only; the wsl build was covered by the
  DECISION tests (151 passed), which re-derive every premise on the running
  build and would fail if the readings had moved across the bar.  What bounds
  the cross-build spread of these quantities at ~1e-13 is VERIFY-WP-B7c's own
  two-build measurement of the same readings, not a second run of this
  population;
* **the accept criterion.**  This population is continuous in fidelity, so
  there is no bimodal split to read a bar off, and both WP-B7c's own 0.883 and
  a stricter 0.95 are reported rather than one being chosen.  The
  criterion-free statement of section 3.1 -- the guard's own split separates
  the fidelity populations on fold rings -- is what does not depend on it;
* **the fallback route.**  The arbiter reads the branch sum there, and that
  reading does not order the returned field's fidelity, because the dominant
  error is the missing dark tail.  Closing that needs the fold's own accuracy
  gate, not this one;
* **the member question**, unchanged from WP-B7c and VERIFY-WP-B7c: the oracle
  builds its boundary field from geometrical optics at the exit vertex, so a
  hand-off-vs-oracle fidelity partly measures two propagators of the same
  boundary field agreeing.  Nothing in this round rests on that comparison;
* **the quadrature itself.**  The arbiter DETECTS the failure; it does not fix
  it.  An area-weighted splat would make `ray_subsample` a convergence knob
  instead of a divergence one and would move every multibranch field near a
  caustic -- WP-B7c's open item 1, still open, and now with a deterministic
  detector to regression-test it against.
