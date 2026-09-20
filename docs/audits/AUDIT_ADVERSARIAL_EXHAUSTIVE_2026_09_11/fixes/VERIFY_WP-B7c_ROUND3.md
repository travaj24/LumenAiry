# VERIFY-WP-B7c round 3 -- the mechanism, the ladder and the lattice all reproduce; the bar's DERIVED CENTRE does not, the converged residue is not the window-edge term round 3 says it is, and the fold ring's two fidelity populations overlap four times wider than round 3 measured

Independent adversarial verification of WP-B7c **round 3**: branch
`fix/wp-b7c-round3`, twelve commits `fc66a8f8 .. bf0b358b` off
`wave5/audit-leftovers` at `dcaa21f0`.  Verified against
`fixes/WP-B7c_ROUND3_REPORT.md`, `tests/unit/test_wp_b7c_round3.py` and
`validation/probe_wp_b7c_round3/`.

Worktree `C:/tmp/lum_vmb3` (`verify/wp-b7c-round3`).  Three more trees, all
`git archive` EXPORTS and not worktrees: `C:/tmp/lum_vmb3_pre` (`dcaa21f0`),
`C:/tmp/lum_vmb3_post` (`bf0b358b`) for bit identity, and
`C:/tmp/lum_vmb3_mut` / `C:/tmp/lum_vmb3_mutw` for the mutation matrix.

| build | interpreter | numpy | scipy | tree |
|---|---|---|---|---|
| **win** | Windows py3.14.6 | 2.4.4 | 1.17.1 | `C:/tmp/lum_vmb3` |
| **wsl** | WSL py3.12.3 | 2.4.6 | 1.17.1 | `/mnt/c/tmp/lum_vmb3` |

Probes and JSON: `validation/probe_verify_b7c_round3/`.  New decision tests:
`tests/unit/test_verify_b7c_round3.py` (5 ids).

**Nothing in `lumenairy/` was edited.**

---

## 0. How this was measured, and why it is independent

WP-B7c round 3 imported the round-2 verifier's oracle **verbatim**
(`r3oracle.py` *is* `probe_verify_b7c_round2/vroracle.py`, plus an aspheric
term), so re-using it here would have made this verification a re-run of
arithmetic that has already been used to produce the numbers under test.
`v3oracle.py` is written for this verification and differs in METHOD at every
step: a vectorised damped-Newton intersection with a bisection fallback rather
than a fixed-iteration scalar Newton; ANGLE-form Snell rather than the vector
form; a 4th-order ray-map Jacobian rather than `np.gradient`; a composite
SIMPSON radial quadrature in the launch height rather than trapezoid ring
weights; a GAUSS-LEGENDRE azimuthal quadrature rather than the midpoint rule;
and an angular-spectrum source laid down by CUBIC interpolation of AMPLITUDE
and OPTICAL PATH rather than linear interpolation of the real and imaginary
parts (which at NA 0.4 commits ~4e-3 of relative error, because the field's
phase turns by ~0.18 rad between source samples).

Controls on the oracle itself, all on this build:

| control | measured |
|---|---|
| this module's Sellmeier vs `lumenairy.glass.get_glass_index`, 10 glasses x 7 wavelengths | **2.22e-16** worst |
| the even-aspheric term is bit-inert on a spherical (`V`) and on a conic (`VC`) prescription | **0.0** on every column |
| exact azimuthal quadrature, safety factor doubled (`Q` z = 5680 um) | **1.36e-12** |
| exact azimuthal quadrature, ring count doubled | **1.24e-04** |
| ASM vs exact quadrature once the band limit clears the largest source-to-pixel angle | **0.00196** rel L2, **0.9999980** fidelity |
| ASM fine window TRIPLED, 24 planes stratified over fidelity 0.015 .. 0.996 | self-fidelity **>= 0.999988**, power shift **<= 1.8e-5** |

### The population

Nine prescriptions and seventeen (prescription, grid) pairs: fifteen carry the
three-level z ladder, one more (`VS`) is swept through its own focus for the
fallback route, and `Q` is used for the oracle floor.  **FOUR prescriptions are
new to the campaign** -- they appear in no fixture module of WP-B7b, VERIFY-B7b,
WP-B7c, VERIFY-WP-B7c, WP-B7c round 2, VERIFY round 2 or WP-B7c round 3 --
and four are re-typed from the published prescriptions the claims are stated
ON, so a claim about a specific optic is re-measured on that optic's geometry.

| optic | what it is | NA | turns in `f(h)` | grid | provenance |
|---|---|---|---|---|---|
| `VA` / `VA_alt` | N-BK7 biconvex with an even-ASPHERIC departure (`a4 = -1.0e7`, `a6 = +1.2e14`) on the **EXIT** surface | 0.195 | **1** | 512 x 2.30 um / 640 x 1.85 um | **NEW** |
| `VX` / `VX_alt` | N-SF6 biconvex at **NA 0.437** | 0.437 | 0 | 640 x 1.60 um / 512 x 2.00 um | **NEW** |
| `VC` / `VC_alt` | N-LAK22 plano-convex with a **HYPERBOLIC** conic `k = -2.20` | 0.169 | 0 | 512 x 2.00 um / 640 x 1.60 um | **NEW** |
| `VD` | AIR-SPACED pair run **negative element first** | 0.173 | 0 | 512 x 2.10 um | **NEW** |
| `V` / `V_alt` | VERIFY-B7b's N-BAF10 biconvex (the fixture the bar is argued on) | 0.216 | 0 | 512 x 2.20 um / 400 x 2.80 um | re-typed |
| `HN` | round 3's NA-0.41 N-LASF9 plano-convex (claim 2 is about it) | 0.413 | 0 | 640 x 2.00 um | re-typed |
| `W` / `W_alt` | the round-2 verification's plano-first N-BAK4 | 0.107 | 0 | 512 x 2.40 um / 640 x 1.90 um | re-typed |
| `Q` | the cemented doublet of round 2's own oracle-floor row | 0.109 | 0 | 512 x 2.60 um | re-typed |
| `VS`, `VA_c`, `HN_c`, `W_c` | a slow control at NA 0.027 scanned THROUGH its focus, and three deliberately COARSE grids | -- | -- | -- | **NEW** (fallback enrichment) |

`VA`'s focal locus is the one with an interior stationary point: `f(h)` falls
2773.9 -> 2728.3 um at `h = 386 um` of a 650 um clear semi-aperture and then
rises to 2907.4.  Every spherical, conic, cemented and air-spaced optic here
reads ZERO turns, which is the control that the asphere is the thing being
probed.  The re-typing is cross-checked by the geometry: `V` reads NA 0.2163
and `f_par` 1876.1 um against round 3's 0.216 and "~1876 um", `HN` 0.4125
against 0.412, `W` 0.1074 against 0.107, `Q` 0.1085 against 0.108.

**1141 oracle-scored planes** over sixteen (prescription, grid) pairs -- 661
on the completion route and 480 on the fallback route -- and the z ladders
carry **1172** planes in total, 897 of them on the completion route.  The
ladder population is the fifteen pairs; `VS` adds 40 swept planes and `Q` is
scored only at the oracle-floor row.

---

## 1. Verdict table

| # | claim | verdict | my numbers |
|---|---|---|---|
| **1** | the fold-ring gap is a reading of the z ladder, not a property (1.371x -> 1.032x -> 1.0045x; margin 1.040x -> 1.012x -> 1.00047x; no floor) | **CONFIRMED, and sharpened** | on the SAME fifteen (prescription, grid) pairs, three ladders: fold gap **1.07810 -> 1.03669 -> 1.02067**, two-sided margin **1.019x/1.058x -> 1.019x/1.017x -> 1.016x/1.005x**, and the gap falls monotonically on every one of the ten pairs that has a crossing at all.  Sharper: at a SINGLE crossing the gap does NOT go to 1 -- bisecting `V` between 1761 and 1764 um to **2.9e-3 nm** leaves the reading stepping 0.99322 -> 1.38848 across the bar.  The reading is DISCONTINUOUS in z; the population gap falls because refinement adds crossings, not because any crossing closes (section 2) |
| **2** | the fold ring's fidelity populations OVERLAP (worst returned 0.9421 vs best refused 0.9520); `HN`'s whole ring returned at 0.9421-0.9564 with 1.24-1.27x the oracle's power | **CONFIRMED, four times wider** | 661 oracle-scored fold-ring planes: RETURNED **0.9331 .. 0.9965**, REFUSED **0.0239 .. 0.9740** -- an overlap of **0.041** against round 3's 0.010.  `HN`'s whole fold ring is returned at **0.9331 .. 0.9557** with **1.218 .. 1.264x** the oracle's power.  The best REFUSED is `W` z = 4925.69 um, reading 1.1072, fidelity **0.9740** -- the same optic and within 4 um of round 3's own `W_alt` counterexample (section 3) |
| **3** | the bar-cost table: fold ring at 0.95, 1.06 costs 1 false refusal / 2 misses; 1.0018 costs 63/1; 1.04 5/2; 1.08 1/6 | **CONFIRMED** | fold ring `n = 661`: **1.0018 -> 231/1, 1.0059 -> 71/1, 1.0177 -> 17/1, 1.04 -> 5/2, 1.06 -> 1/2, 1.08 -> 1/4** -- the 1.04 and 1.06 columns land on round 3's own numbers.  1.06 is the knee and the best neighbour on both axes (section 4) |
| **3b** | the derived centre over all planes is **1.0600253** and the fold-ring-only one **1.0619044**, so "1.06 to three figures IS the derived centre" | **PARTLY CONFIRMED** -- one of the two centres reproduces, the other does not | over all 1141 planes I read **1.0571316**, which IS 1.06 to three figures, so round 3's headline construction survives an independent population.  The FOLD-RING centre does not: **1.0543406** against round 3's 1.0619044, i.e. 1.05 against 1.06.  Both are readings of the population -- the two constructions disagree by 0.26 % within ONE population and the same construction moves 0.7 % between two -- so which one the sentence means is load-bearing, and it does not say (section 4.1) |
| **4** | `_PIXEL_CONTINUITY_MAX` not moved because the two-sided margin (0.047 %) is below the 1 % rule and 1.06 IS the derived centre | **CONFIRMED in outcome; the reasoning is weaker than it reads** | the derivation comment DOES name its population ("1304 oracle-scored planes", "SIXTEEN prescriptions and twenty-one (prescription, grid) pairs") and the 1 %-rule arithmetic is applied exactly as stated (0.047 % < 1 %).  Two caveats: the "1 % rule" appears nowhere in `docs/TESTING_STANDARDS.md`, in the constant, or in any earlier round -- it is introduced by the document that applies it -- and the quantity it is applied TO is the margin claim 1 establishes cannot be measured.  Keeping the bar is right; the reason that survives is the COST table (section 4.1, defect **D-6**) |
| **5** | converged reading 0.9994103 .. 1.0004414 (rms 1.48e-4, worst 5.9e-4); 1.4e-3 on a too-small window, 1.2e-4 on larger ones, bit-identical for N = 512/640/896; 2.1e-2 on healthy FOLD planes (36x) | **CONFIRMED** | 96 planes over 12 pairs, chosen geometrically: **0.9997692 .. 1.0002647**, rms **8.50e-5**, worst **2.65e-4**.  Window ladder at a fixed 3.00 um pitch: **4.96e-4** at N = 192, then **7.2986e-05 bit-identically at N = 384, 512, 640 and 896**.  Healthy FOLD planes: worst **8.59e-3**, i.e. **32x** the converged spread.  Identical to the printed digit on BOTH builds (section 5) |
| **5b** | the residue is "the window-edge term `_HALF_PITCH_CENTRE_OFFSET` documents" / "exactly what a Voronoi-hull mismatch at the window EDGE predicts" | **REFUTED** | a term living at the window edge must SHRINK as the window grows past the field.  It does not move by one bit over a 2.3x range of window width.  What moves it is the PITCH: holding the window and refining the grid takes the deviation 7.85e-4 -> 1.26e-5.  The residue is the two point-sampled quadratures' INTERIOR difference; the hull mismatch explains only the excess on an UNDERSIZED window (section 5.1, defect **D-4**) |
| **6** | E5's hull-aligned lattice was built, measured and REJECTED (unbounded: 7343.9 where nesting reads 3.998); what shipped is NESTING plus one `half_pitch_centres` definition; a converged render reads 1 to O(1/N) | **CONFIRMED to the digit** | on an independently typed `V`: nested branch sum **3.99837**, hull-aligned **7343.92932** at z = 1768 um -- round 3's own two numbers.  At z = 1761 um the branch sum moves 1.06360 -> 1.01803 and the completion 1.00213 -> 0.97970 with the field's fidelity unchanged at 0.98789.  The unboundedness generalises: 2728.97 on `VX`, 627.44 on `VA`.  The RETURNED FIELD is bit-identical under both lattices on all seven cases, so the lattice really is the arbiter's alone.  Identical on both builds (section 6) |
| **7** | bit identity 157 cases / 17 optics: 116 identical, 0 moved, 0 newly refused, 0 newly returned, 41 refused on both | **CONFIRMED** | my own matrix, **117 cases over 13 optics**, archive to archive, a child process per tree with `lumenairy.__file__` asserted under it and `LUMENAIRY_MEM_BUDGET_MB=2048` pinned: **93 identical, 0 MOVED, 0 newly refused, 0 newly returned, 24 refused on both** -- and the same tally on wsl.  The two trees differ in exactly the two files round 3 claims (section 7) |
| **8** | E7: on 476 returned fallback planes 263 score below 0.95; the continuity loss arm flags 192 wrong / 0 right, the power loss arm 47/0, `< 0.889` 254/0 with 9 left; 13 unordered; the scope enum + note shipped, unknown scope raises | **the machinery and the "zero false alarms" CONFIRMED; the arms' SENSITIVITY REFUTED** | on 350 returned fallback planes over sixteen pairs (including a slow control at NA 0.027 scanned through focus and three deliberately coarse grids) the pure loss arms flag 11 and 16 planes and **every one is wrong** -- I could not refute the zero-false-alarm property.  But they catch 11-16 of 50 wrong planes where round 3's catch 192-254 of 263, their Spearman against fidelity is **-0.044** and **+0.308** against round 3's 0.591 and 0.846, and the **23** planes nothing sees include the worst field in this study: oracle fidelity **0.2915** with `pixel_continuity` 0.99896 and the bracket 0.9999 (section 8) |
| **9** | R3-5's six corrected paragraphs; the `0.37 eps` rule deleted; the correction AGAINST the round-2 verification's E6 energy closure; E6 re-measured full-radius (`Q`: rule 3.2e-4, table 0.0003, measured 0.0563 = 178x; ASM vs exact 0.00068-0.00738) | **CONFIRMED (the measurement), with one arithmetic slip and one stale set of counts** | at round 2's own `Q` row my independent oracle reads **0.05628** full radius against round 3's 0.0563, **0.005367** core-confined against 0.00537, fidelity **0.99843** against 0.99842, and the rule predicts **3.172e-4**.  The closure adjudication is confirmed by direct measurement: a 256-radius profile closes at **0.97968 (exact) / 0.97976 (`J0`)** -- equal on both arms -- rising together to 0.99903 / 0.99911 at 1200 radii.  The ASM bracket reproduces once the band limit clears (0.00196 rel L2, 0.9999980 fidelity).  BUT the six dated CORRECTED paragraphs carry the counts of an intermediate 1196-plane pass, not the closed population (section 9, defect **D-3**) |
| **10** | R3-6: 18/18 mutations caught on both builds; M17 exposed the converged-spread pin not firing | **CONFIRMED exactly** | my own matrix, same gate: **18 of 18 caught**, with the pass/fail counts identical to round 3's table on every row (M1 24/43, M2 1/41, M3 1/38/3, M4 19/22/1, M5 4/38, M6 4/38, M7 14/26/2, M8 18/24, M9 10/29/3, M10 5/37, M11 2/40, M12 1/34/7, M13 2/39/1, M14 3/37/2, M15 1/41, M16 2/40, M17 6/36, M18 1/41).  Of my FIVE new mutations, **three are caught and two are not** (section 10) |
| **11** | fingerprints deliberately not re-recorded; `_lens_traced` is the only module in this family with a history document | **CONFIRMED** | `scripts/record_history_fingerprints.py --check` reads `OK: every history document matches its module`; `git diff dcaa21f0..bf0b358b -- lumenairy/elements/_lens_traced.py` is empty (0 lines); `docs/history/` carries `lumenairy.elements._lens_traced.md` and NO document for `_lens_traced_multibranch` or `_lens_traced_uniform` |
| **12** | `.test_durations` 16 310 with the 8 new ids; the durations-staleness gate green | **CONFIRMED** | the file reloads as valid JSON with **16 310** entries, the eight `test_wp_b7c_round3` ids are present with plausible values (0.0008 .. 2.65 s), and `test_audit2609_a15a_durations_staleness.py` is **4 passed** |

---

## 2. Claim 1 -- the ladder, and what is under it

Three ladders on the same fifteen (prescription, grid) pairs, each derived
from the geometry and then refined: **L1** = 34 planes over the completion
band the build itself takes plus a 12-plane tail over the rest of the
interior-turning-point window; **L2** = 21 planes between every ADJACENT PAIR
of L1 fold-ring planes that straddles the bar (~20x the step); **L3** = the
same rule on L2's own straddling pairs (~10x again).

| ladder | planes | fold-ring planes | fold gap | margin above | margin below | all-planes gap |
|---|---|---|---|---|---|---|
| band + tail | 690 | 415 | **1.07810** | 1.01917 | 1.05782 | 1.01627 |
| + pass 1 | 963 | 688 | **1.03669** | 1.01917 | 1.01719 | 1.01627 |
| + pass 2 | **1172** | **897** | **1.02067** | 1.01569 | 1.00490 | 1.01531 |

It is not one fixture's trend.  Per (prescription, grid) pair, the fold gap
over the three ladders:

| pair | L1 | + pass 1 | + pass 2 |
|---|---|---|---|
| `HN` (NA 0.41) | 3.9744 | 2.5483 | **1.9287** |
| `V` | 3.9556 | 3.5858 | **2.0053** |
| `VA` (the asphere) | 3.7902 | 1.2822 | **1.0580** |
| `VA_alt` | 3.7804 | 1.2224 | **1.0890** |
| `VC` (hyperbolic conic) | 3.9568 | 2.7367 | **1.0989** |
| `VC_alt` | 3.9704 | 3.2378 | **2.2931** |
| `VX` (NA 0.44) | 3.8622 | 1.4046 | **1.0357** |
| `VX_alt` | 1.3528 | 1.2468 | 1.2468 |
| `V_alt` | 1.1092 | 1.1090 | **1.0867** |
| `W` | 2.7565 | 1.0677 | **1.0544** |

Nine of the ten strictly narrow, none widens (none can: a refinement adds
planes, so the minimum refused reading can only fall and the maximum returned
one can only rise), and `VD`, `W_alt`, `W_c`, `VA_c` and `HN_c` produce no
crossing at all.  Round 3's "the gap is a reading of the z ladder" reproduces
on optics it never ran, and so does "there is no sign of a floor" as a
statement about the POPULATION.

**But the mechanism round 3 gives for it is not quite the mechanism.**  Round
3 says "the reading is not smooth in z near a fold onset ... so refining the
ladder keeps finding readings closer to the bar", which reads as a steep but
continuous crossing that a fine enough ladder resolves.  It is not continuous.
Bisecting `V` between the straddling pair 1761 / 1764 um, twenty halvings:

```
 0  dz=1500 nm     returned 1.002134   refused 3.977097
 4  dz=93.75 nm    returned 0.993713   refused 2.031655
 8  dz=5.86 nm     returned 0.993220   refused 1.389371
12  dz=0.366 nm    returned 0.993220   refused 1.388492
16  dz=0.0229 nm   returned 0.993219   refused 1.388488
19  dz=0.00286 nm  returned 0.993219   refused 1.388484
```

At a plane separation of **three picometres** the reading still STEPS by
1.3980x across the bar, and the step has converged -- it stops moving after
about ten halvings.

**And the whole diagnostics say what the step IS.**  Reading every key on both
sides of that 2.861 pm bracket (`control_jump_win.json`,
`control_jump_wsl.json`, identical on both builds to the last bit but one):

| | returned side | refused side |
|---|---|---|
| `z` [um] | 1761.6225128173824 | 1761.6225156784053 |
| `pixel_continuity` | **0.9932194239** | **1.3884836045** |
| `n_branch_max` | **2** | **9** |
| `n_triangles_degenerate` | 8 | 8 |
| `r_c` [m] | 9.772659106e-06 | 9.772658737e-06 |
| `kappa` | 10.55059183 | 10.55059189 |
| `zeta_extrapolation` | 0.3320488291 | 0.3320488172 |
| `reason` | `fold_ring` | `fold_ring` |

Every CONTINUOUS quantity agrees to seven or eight figures -- the fold radius
moves by 4e-8 of itself, the curvature by 6e-9 -- and the only thing that
changes is `n_branch_max`, **2 -> 9**, which is an INTEGER.  The reading is a
ratio of two rasterisations of the mapped triangles, and a ray map that puts
nine branches on a pixel instead of two rasterises differently at the two
pitches.  So the discontinuity is not an artefact of the ladder's resolution;
it is the branch census, and no amount of refinement removes it.

The population gap therefore shrinks not because any crossing closes but
because each refinement finds MORE crossings and the minimum over them keeps
falling.  That is a stronger statement of round 3's conclusion -- no margin
exists to be measured, at any ladder density -- and it is what
`test_vb7c3_the_bar_has_no_margin_because_the_crossing_is_a_jump_in_z` pins.

---

## 3. Claim 2 -- the two fidelity populations

| population | n | reading | oracle fidelity |
|---|---|---|---|
| **RETURNED** | 608 | 0.69459 .. 1.04360 | **0.9331** .. 0.9965 |
| **REFUSED** | 53 | 1.06519 .. 3.99863 | 0.0239 .. **0.9740** |

They overlap by **0.041** in fidelity, four times round 3's 0.010 and on a
population that shares one optic with round 3's.

| | optic | z [um] | reading | fidelity | power / oracle |
|---|---|---|---|---|---|
| worst RETURNED | `HN` (NA 0.41) | 2081.58 | 1.02083 | **0.9331** | 1.2640 |
| | `HN` | 1719.57 | 0.99613 | 0.9496 | 1.2339 |
| | `HN` | 1802.12 | 1.00202 | 0.9503 | 1.2369 |
| best REFUSED | `W` | 4925.69 | 1.10720 | **0.9740** | 1.1844 |
| | `VX` (NA 0.44) | 807.69 | 1.06519 | 0.9491 | 1.0940 |
| | `W` | 4921.15 | 1.12339 | 0.9457 | 1.1936 |

Two things are worth naming.

**`HN`'s miss class reproduces on an independent oracle.**  Its WHOLE fold
ring is returned, at fidelity 0.9331-0.9557 with the returned field carrying
**1.218x to 1.264x** the oracle's energy, and the reading never leaves
0.9961-1.0208.  Round 3 measured 0.9421-0.9564 at 1.24-1.27x.  On the fastest
optics in either study the arbiter sees nothing wrong while a quarter of the
returned energy is spurious, and that is invisible below NA 0.4.

**The single false refusal at 1.06 is the same optic round 3 found.**  Round 3
names `W_alt` z = 4928.969 um (reading 1.0812, fidelity 0.9520); mine is `W`
z = 4925.69 um (reading 1.1072, fidelity 0.9740), 3 um away on the other grid
of the same plano-first N-BAK4.  Two independent oracles and two independent
ladders land on the same plane, which is the strongest single corroboration in
this verification.

On `W` the two populations overlap WITHIN one optic: worst returned 0.9699 at
z = 4925.43 um against best refused 0.9740 at z = 4925.69 um -- 0.26 um apart.

Per optic, the fold ring at the shipped bar:

| pair | n | RETURNED reading / fidelity / power | REFUSED reading / fidelity |
|---|---|---|---|
| `HN` | 42 | 0.9961..1.0208 / **0.9331..0.9557** / **1.218..1.264** | 1.9689..3.9985 / 0.0239..0.5329 |
| `V` | 35 | 0.8426..1.0106 / 0.9843..0.9927 / 0.980..1.059 | 2.0264..3.9974 / 0.1852..0.6669 |
| `VA` | 61 | 0.9556..1.0411 / 0.9558..0.9964 / 1.002..1.063 | 1.1014..3.9456 / 0.1278..0.9161 |
| `VA_alt` | 54 | 0.9783..1.0436 / 0.9572..0.9964 / 1.006..1.068 | 1.1365..3.9545 / 0.1076..0.9204 |
| `VA_c` | 38 | 0.9771..1.0401 / 0.9714..0.9963 / 1.005..1.070 | -- |
| `VC` | 45 | 0.9908..1.0099 / 0.9845..0.9936 / 0.997..1.054 | 1.1097..3.9984 / 0.0501..0.9457 |
| `VC_alt` | 50 | 0.9076..1.0104 / 0.9852..0.9928 / 1.001..1.047 | 2.3171..3.9986 / 0.0411..0.5247 |
| `VD` | 30 | 0.9685..1.0068 / 0.9917..0.9959 / 0.985..1.025 | -- |
| `VX` | 42 | 0.9665..1.0285 / 0.9731..0.9964 / 1.004..1.045 | 1.0652..3.9795 / 0.0709..0.9491 |
| `VX_alt` | 66 | 0.9926..1.0103 / 0.9915..0.9965 / 1.003..1.024 | 1.2597..3.9737 / 0.0785..0.8597 |
| `V_alt` | 95 | 0.9399..1.0243 / 0.9736..0.9910 / 0.996..1.081 | 1.1132..1.1975 / 0.8769..0.9255 |
| `W` | 72 | 0.8545..1.0226 / 0.9699..0.9926 / 0.962..1.070 | 1.0782..2.7831 / 0.3403..**0.9740** |
| `W_alt` | 31 | 0.6946..1.0094 / 0.9834..0.9915 / 0.964..1.066 | -- |

`HN` is the only optic whose returned fold ring sits below 0.96 everywhere,
and the only one carrying more than 7 % extra energy on every returned plane.
`VX`, at a HIGHER NA (0.437) on a different glass and a different form,
returns 0.9731-0.9964 -- so the miss class is not predicted by speed alone.

---

## 4. Claim 3 -- the bar-cost table, rebuilt

Fold ring, `n = 661`.  "False refusal" = refused with fidelity above the
criterion; "miss" = returned with fidelity below it.  `s_conv` = **2.647e-4**,
this population's worst converged deviation from 1 (section 5).

| bar | refused | returned | margin above | margin below | false refusals @ 0.95 | misses @ 0.95 | false @ 0.883 | misses @ 0.883 |
|---|---|---|---|---|---|---|---|---|
| 1.0018 (`1 + 7 s_conv`) | 284 | 377 | 1.00001 | 1.00001 | **231** | 1 | 246 | 0 |
| 1.0059 (`1 + 22 s_conv`) | 124 | 537 | 1.00013 | 1.00006 | **71** | 1 | 86 | 0 |
| 1.0177 (`1 + 67 s_conv`) | 70 | 591 | 1.00058 | 1.00027 | **17** | 1 | 32 | 0 |
| 1.04 | 57 | 604 | 1.00036 | 1.00006 | 5 | 2 | 19 | 0 |
| **1.06 (shipped)** | 53 | 608 | 1.01572 | 1.00490 | **1** | **2** | 15 | 0 |
| 1.0600253 | 53 | 608 | 1.01574 | 1.00487 | 1 | 2 | 15 | 0 |
| 1.0619 | 53 | 608 | 1.01754 | 1.00310 | 1 | 2 | 15 | 0 |
| 1.08 | 51 | 610 | 1.00165 | 1.01983 | 1 | **4** | 13 | 0 |

All planes, `n = 1141`: 1.06 costs 1 false refusal and 52 misses at 0.95;
1.0018 costs 269 and 31; 1.04 costs 6 and 50; 1.08 costs 1 and 57.  As in
round 3, the all-planes miss column is not this arm's failure -- 50 of the 52
are on the FALLBACK route, where the module has already warned that the
completion does not apply.

**The shape round 3 reports is exactly right, and two of its columns land on
the same integers.**  1.06 sits at the knee: 1.04 costs FIVE times the false
refusals for no fewer misses (round 3: five), 1.08 costs twice the misses for
no fewer false refusals (round 3: three times), and every bar derived from the
converged reading's own spread costs 17-231 false refusals (round 3 measured
9-63).  Nothing measured here does better than 1.06 on both axes.

### 4.1 The derived centre, and what about it does not reproduce

Round 3 keeps the bar with two arguments: the two-sided margin at 1.06 is
0.047 %, below "the 1 % that would call for re-centring"; and the geometric
centre of the gap the bar sits in "IS the derived centre to three figures".

It computes that centre TWICE -- **1.0600253** over all 1304 planes and
**1.0619044** on the fold ring alone -- and reports that moving to the
fold-ring one "changes no fold-ring decision at all and returns one more wrong
field over all planes".  On my population the same two constructions give
**1.0571316** over all 1141 planes and **1.0543406** on the fold ring (the
largest returned reading is 1.04360 and the smallest refused 1.06519, whose
geometric mean is 1.0543).

So the HEADLINE construction reproduces: 1.0571 is 1.06 to three figures, as
1.0600 is.  What does not reproduce is the stability the sentence implies.  The
two centres round 3 itself computes differ by 0.18 % on its population and
0.26 % on mine, and the SAME construction moves 1.0619 -> 1.0543 (0.7 %)
between two populations that share an optic.  "1.06 to three figures IS the
derived centre" is therefore true of one construction on both populations and
false of the other on mine, which makes the choice load-bearing in a sentence
written as if it were not -- the same shape claim 1 disposes of for the
margin.

The CONCLUSION is unaffected and I agree with it: re-centring to 1.0543 would
change no decision on either population (the gap is empty between 1.0436 and
1.0652 here, and the cost table reads the same integers at 1.06, 1.0600253 and
1.0619).  What should change is the REASON written in the constant: the bar is
kept because it is the cost optimum, and the derived-centre sentence should
name the construction and the population it belongs to.  That is defect
**D-6**.

I could not find the "1 % rule" stated anywhere outside this report --
`docs/TESTING_STANDARDS.md` has no such rule, nor does the constant's own
comment, nor the earlier rounds.  The arithmetic is applied as stated (0.047 %
< 1 %), but a threshold invented in the document that applies it is not a
rule, and the report presents it as one.

---

## 5. Claim 5 -- the converged reference

96 planes over the twelve pairs, chosen by a GEOMETRIC criterion (at least a
quarter of the paraxial focal distance short of the interior fold's onset), so
no reading selects the planes its own spread is then measured on.

| population | n | reading | rms deviation | worst |
|---|---|---|---|---|
| converged, 12 pairs | 96 | **0.9997692 .. 1.0002647** | **8.498e-05** | **2.647e-04** |
| healthy FOLD planes (fidelity >= 0.99 AND power within 2 % of the oracle) | 44 | 0.9914106 .. 1.0035878 | -- | **8.589e-03** |

**32.4x**, against round 3's 36x.  A bar derived from the converged spread is
unusable where the guard works, and section 4 measures what it would cost.
Every number in this table is identical to the printed digit on win and on
wsl.

### 5.1 The residue does not follow the window (defect D-4)

| WINDOW ladder (`dx` held at 3.00 um, `N` grows) | 192 | 256 | 384 | 512 | 640 | 896 |
|---|---|---|---|---|---|---|
| window [um] | 576 | 768 | 1152 | 1536 | 1920 | 2688 |
| deviation from 1 | 4.959e-4 | 2.386e-4 | **7.2986e-05** | **7.2986e-05** | **7.2986e-05** | **7.2986e-05** |

| GRID ladder (window held, `N` and `dx` both scaled) | 128 | 192 | 256 | 384 | 512 | 768 | 1024 |
|---|---|---|---|---|---|---|---|
| deviation from 1 | 7.85e-4 | 2.97e-4 | 2.75e-4 | 3.51e-4 | 1.37e-4 | **1.26e-5** | 5.81e-5 |

Round 3 reads the first of these as "exactly what a Voronoi-hull mismatch at
the window EDGE predicts" and writes that mechanism into
`_HALF_PITCH_CENTRE_OFFSET`: "Where light reaches the window edge the two
integrals are therefore not over the identical rectangle, and a converged
render reads 1 to O(1/N) rather than exactly."

That is refuted by the ladder itself.  The hull mismatch is a fixed `dx/4` at
the boundary; what multiplies it is the light that REACHES the boundary, and
that falls to zero as the window grows.  An edge term must therefore SHRINK
along this ladder.  It does not move **by a single bit** across a 2.3x range
of window width -- and round 3's own table shows the same plateau (1.18e-4,
"bit for bit, for N = 512, 640 and 896") and calls it confirmation.

The decisive reading is the last one.  At `N = 896` the window is
+/-1344 um and the field at that plane is a converging beam of geometric
radius ~200 um: there is no light within 1 mm of the boundary, so ANY term
that lives at the boundary is identically zero there.  The reading is still
**7.2986e-05** from 1.  A residue that survives with nothing at the edge is
not an edge term.

What the residue does follow is the PITCH: the grid ladder takes it from
7.85e-4 to 1.26e-5 over 8x in `N` at a fixed window.  It is the difference
between two point-sampled quadratures of the same interior field at two
pitches -- the thing the reading is BUILT from -- and the hull mismatch
explains only the EXCESS on a window too small to hold the field (4.96e-4
against 7.30e-5, a factor of 6.8; round 3 measured 12x).

This is a P3: it changes no decision, but it is a mechanism written into a
shipped constant that its own evidence contradicts, and the next reader will
try to remove the residue by growing the window.  Pinned by
`test_vb7c3_the_converged_readings_residue_does_not_follow_the_window`.

---

## 6. Claim 6 -- the lattice

Both conventions run on the same planes, on independently typed
prescriptions, with the RETURNED field compared byte for byte.  The
hull-aligned offset is DERIVED here (setting the fine hull's lower edge equal
to the coarse one's gives `o = -1/2`), not copied.

| case | nested (shipped) | hull-aligned | field bit-identical | fidelity |
|---|---|---|---|---|
| `V` z = 1768 um, blow-up, branch sum | **3.99837** | **7343.92932** | yes | 0.16756 both |
| `V` z = 1764 um, branch sum | 3.99710 | 4323.44634 | yes | 0.18915 both |
| `V` z = 1761 um, branch sum | **1.06360** | **1.01803** | yes | 0.98789 both |
| `V` z = 1761 um, completion | **1.00213** | **0.97970** | yes | 0.98789 both |
| `V` z = 1000 um, far from every caustic | 1.00016 | 1.00026 | yes | 0.99666 both |
| `VX` z = 900 um (NA 0.44) | 3.99565 | **2728.97155** | yes | 0.15893 both |
| `VA` z = 2900 um (the asphere) | 3.98588 | **627.44063** | yes | 0.14712 both |

`3.99837` and `7343.92932` are round 3's own two numbers, reproduced to every
printed digit from a prescription typed independently.  So are 1.0636 ->
1.0180 and 1.0021 -> 0.9797 at z = 1761 um with the field's fidelity unchanged
at 0.9879.  The unboundedness is not a property of that one fixture: the
hull-aligned lattice reads 2729 and 627 where nesting reads 3.996 and 3.986.
Every nested reading in the study is **below 4**, which is the `~4 per
halving` bound nesting buys and the hull-aligned convention destroys.

The claim that the lattice is the ARBITER's alone is measured, not assumed:
the returned field is bit-identical under the two conventions on all seven
cases.  Identical on both builds.

The `O(1/N)` statement is true as arithmetic at a fixed window (`1/N` and `dx`
are the same thing there) but its stated MECHANISM is wrong -- see 5.1.

---

## 7. Claim 7 -- bit identity

`v3bitid.py`, **117 cases over 13 optics**: for each optic a healthy fold
plane, a plane at the fold band's far edge, a blown-up plane, a plane short of
the fold, the exit vertex (`output_plane_distance = 0`), `caustic_band =
'plain'`, `ray_subsample = 4`, a complex64 input and the branch sum alone.

Both trees are `git archive` exports differing in exactly the two files round
3 changed (`diff -rq` on `lumenairy/` names `_lens_traced_multibranch.py` and
`_lens_traced_uniform.py` and nothing else).  A CHILD process per tree with
`cwd` and `PYTHONPATH` set to it and `lumenairy.__file__` ASSERTED under it;
`LUMENAIRY_MEM_BUDGET_MB=2048` pinned; SHA-256 over `ndarray.tobytes()` with
the dtype, the shape and the DECISION recorded beside it.

| | win | wsl |
|---|---|---|
| identical | **93** | **93** |
| **MOVED** | **0** | **0** |
| newly refused | **0** | **0** |
| newly returned | **0** | **0** |
| refused on both | 24 | 24 |

Round 3's "nothing moves" is confirmed on optics it never ran, on both builds.

---

## 8. Claim 8 -- the fallback route

### 8.1 What reproduces

The shipped machinery is exactly as described: `pixel_continuity_of`,
`pixel_continuity_scope` (a three-value enum), `pixel_continuity_scope_note`,
the three scopes distinct and documented, the keys seeded on every dict the
function builds, and an unknown scope refused inside `_arbitrate` rather than
reaching a caller.  I exercised the guard by removing the name the completion
route uses from the enum through the module's own surface: the call raises an
`AssertionError` that names the offending scope and lists the legal ones.

The population had to be BUILT for this question rather than taken from the
ladders.  A fallback population sampled only where the ladders run is
dominated by planes far from any caustic (`reason = 'no_fold'`, median
fidelity 0.994), which is not the regime E7 is about.  Four fixtures were
added for it: **`VS`**, a slow N-BK7 plano-convex at **NA 0.027** scanned
through its own focus -- the regime the round-2 verification's own E7 case
(`C` at z = 22.8 mm, fidelity 0.4639) lives in -- and three deliberately
COARSE grids (`VA_c` 256 x 4.60 um, `HN_c` 320 x 4.00 um, `W_c` 256 x 4.80
um), because the commonest reason the completion declines near a caustic is
that the grid cannot resolve the fold's Airy scale.

### 8.2 The challenge, and what it found

**The "zero false alarms" property survives.**  I could not find a single
fallback plane on which `pixel_continuity < 1/1.06` or
`multibranch_power_ratio_bracketed < 0.889` would refuse a RIGHT field.  Both
arms flag only wrong ones here, exactly as round 3 reports.

**Their SENSITIVITY does not survive, and the worst failure class is invisible
to them.**

| reading the library ALREADY reports | flagged | of which wrong | of which RIGHT | not flagged | of which wrong |
|---|---|---|---|---|---|
| continuity LOSS arm (`< 1/1.06`) | 11 | **11** | **0** | 339 | 39 |
| launched-power LOSS arm (`< 0.5`) | 1 | 1 | 0 | 349 | 49 |
| the same at `< 0.889` | 16 | **16** | **0** | 334 | 34 |
| either loss arm | 11 | 11 | 0 | 339 | 39 |
| **either loss arm OR `reason != 'no_fold'`** | 49 | 27 | **22** | 301 | 23 |

Round 3's arms flag 192 and 254 of 263 wrong planes; mine flag 11 and 16 of
50.  And the 23 planes nothing sees include every one of the sixteen slow-
control planes whose fields score **0.2915 .. 0.6906** with `pixel_continuity`
between 0.992 and 1.017 and the bracket between 0.988 and 1.019 -- i.e. the
worst returned field in this study, at **oracle fidelity 0.2915** with every
reading nominal, is a plane BOTH loss arms call healthy.  Round 3's own worst
unordered plane is 0.5358 and the round-2 verification's is 0.4639, so E7's
failure mode is worse than either measured, and a loss-arm refusal does not
touch it.

The ordering the two arms give is weak on this population -- Spearman against
oracle fidelity: `pixel_continuity` **-0.044**, `multibranch_power_ratio_
bracketed` **+0.308**, `power_ratio` +0.293, `n_triangles_degenerate`
**-0.463** -- against the 0.591 and 0.846 round 3 measures.

**The union that WOULD be sensitive refuses right fields.**  Adding `reason !=
'no_fold'` takes the flag count from 11 to 49 and the wrong ones caught from
11 to 27, at the cost of **22 RIGHT fields**, the best at oracle fidelity
**0.9842** (`VA_c` z = 3253.28 um, reason `fold Airy scale under-resolved by
the grid`), then `VA_c` 0.9838 and 0.9813 and `W_c` 0.9727 and 0.9723 -- all
five on the deliberately coarse grids, i.e. exactly where a caller who cannot
afford a finer grid lives.  Round 3's own table reports 17 false alarms for that same rule, so
this is agreement rather than a new finding -- but it is the rule a maintainer
would reach for if the pure arms' sensitivity mattered, and it is the one that
must not refuse.

By fallback `reason`, which orders it about as well as anything:

| reason | n | fidelity |
|---|---|---|
| `no_fold` | 306 | 0.2915 .. 0.9983, median 0.9938 |
| `fold Airy scale under-resolved by the grid` | 39 | 0.8202 .. 0.9842, median 0.9511 |
| `cusp: non-finite field (guarded)` | 2 | 0.9132 .. 0.9307 |
| `bad_kappa` | 2 | 0.9072 .. 0.9168 |
| `branch_undersampled` | 1 | 0.9603 |

Note that `no_fold` -- the reason round 3 treats as the benign one -- carries
both the healthiest planes (median 0.9938) and every one of the worst.

The 23 planes nothing orders, in full: **sixteen** on the slow control
(fidelity 0.2915 .. 0.6906, reading 0.99201 .. 1.01698, bracket 0.9876 ..
1.0185), and seven at the criterion rather than past it -- `HN` at 0.9372,
0.9493 and 0.9497, `HN_c` at 0.9464, 0.9485 and 0.9492, and `W` at 0.9489.
Round 3's thirteen are nine on its asphere, its slow control and its NA-0.41
plano-convex twice; mine are its slow-control analogue and the NA-0.41
plano-convex on two grids.  Both studies land on the same two shapes: an optic
so slow that geometrical optics has no answer near its focus, and the fastest
optic in the population.

---

## 9. Claim 9 -- the oracle floor (E6)

At round 2's own `Q` row (the cemented doublet at z = 5680 um, `y_max/z` =
0.1084), measured pointwise at a radius-stratified importance-weighted sample
of 480 of the output grid's own pixels, with an exact GAUSS-LEGENDRE azimuthal
quadrature and no reconstruction on either side:

| quantity | round 3, `Q` | this verification, `Q` | round 3, `W` | this verification, `W` |
|---|---|---|---|---|
| `y_max/z` | 0.108 | **0.10838** | 0.109 | **0.10889** |
| `eps` at the rms radius | 8.5e-04 | **8.572e-04** | 1.5e-03 | **1.451e-03** |
| the rule `0.37 eps` predicts | 3.2e-4 | **3.172e-4** | -- | **5.369e-4** |
| `J0` vs exact, FULL radius | **0.0563** | **0.05628** | **0.0952** | **0.09503** |
| ... as fidelity | 0.99842 | **0.99843** | 0.99551 | **0.99553** |
| `J0` vs exact, CORE only | 0.00537 | **0.005367** | 0.01123 | **0.011235** |
| ASM vs exact, rel L2 | 0.00188 | **0.00196** (see below) | 0.00470 | 0.02588 (same cause) |
| ... as fidelity | 0.999998 | **0.9999980** | 0.999989 | 0.999665 |
| exact quadrature, safety x2 | 4.7e-14 .. 1.0e-13 | 1.36e-12 | | 3.93e-12 |
| exact quadrature, rings x2 | 1.1e-4 .. 7.0e-4 | **1.236e-4** | | **2.444e-4** |

Both published `J0` rows reproduce to **three significant figures** against an
independently written quadrature, sample and ray trace.  The full-radius
number is **177x** the rule's prediction at `Q` (round 3: 178x) and **177x**
at `W`, and the core-confined value is **10.5x** and **8.5x** smaller than the
full-radius one on the same sample -- which is E6's mechanism showing
directly.  **R3-5's deletion of the `rel L2 ~ 0.37 eps` rule from
`validation/oracles/caustic_fold_truth.py` is correct and is confirmed.**

The ASM row needed one extra control.  At my first fine-window setting the ASM
read 0.0144 against the exact quadrature, not 0.0020.  The cause is the
Matsushima band limit, not the ASM: it removes every plane wave whose ray
traverses more than half the fine window in `z`, and at that window the limit
was 0.137/um against a largest source-to-pixel angle of 0.206/um.  Widening
the window until the limit clears takes it to **0.00198** and then it does not
move (0.00196 at a further 1.5x, 0.00131 at `refine = 4`).  So round 3's
bracket is right, and the lesson for both studies is that the ASM's fine
WINDOW, not its pitch, is the parameter that has to be controlled at long
propagation distances -- which neither round 3's `_asm_grid` nor my first one
did from the geometry.  `asmbandlimit_win.json` has the ladder.

**The correction AGAINST the round-2 verification is confirmed by direct
measurement, not by attribution.**  The `J0` arm's energy closure from a
radial profile, at three resolutions:

| radii | `Q`: exact arm | `Q`: `J0` arm | `W`: exact arm | `W`: `J0` arm |
|---|---|---|---|---|
| 256 | **0.97968** | **0.97976** | **0.98896** | **0.98886** |
| 600 | 0.99637 | 0.99645 | 0.99884 | 0.99875 |
| 1200 | **0.99903** | **0.99911** | **0.99965** | **0.99955** |

The two arms lose energy EQUALLY at every resolution (they differ by 8e-5 at
`Q` and 1e-4 at `W`, at every one of the three resolutions) and converge
together towards 1.  That is exactly round 3's
diagnosis of the round-2 verification's 0.9972 / 0.9741: a coarse radial
reconstruction, not a propagator defect.  The published closure column is
closer to right than the verification concluded, and R3-5's recording of that
in `VERIFY_WP-B7c_ROUND2.md` is correct.

---

## 10. Claim 10 -- the mutation matrix

The whole five-file gate, one mutation per run, a separate export per build,
restored BY COPY from a snapshot taken at import.

**18 of 18 caught, on both builds.**  On **win**, against round 3's gate
exactly as shipped (its own five files, this verification's not present),
every pass/fail count matches round 3's published table to the integer: M1
`24 failed, 43 passed`; M2 `1 failed, 41 passed`; M3 `1 failed, 38 passed, 3
skipped`; M4 `19/22/1`; M5 `4/38`; M6 `4/38`; M7 `14/26/2`; M8 `18/24`; M9
`10/29/3`; M10 `5/37`; M11 `2/40`; M12 `1/34/7`; M13 `2/39/1`; M14 `3/37/2`;
M15 `1/41`; M16 `2/40`; M17 `6/36`; M18 `1/41`.  That is a complete
reproduction of R3-6.  On **wsl** the same eighteen were run against that gate
PLUS this verification's file, so the counts are higher by this file's own
catches (M2 `3 failed, 44 passed`, M4 `22/24/1`, M10 `6/41`, M13 `3/43/1`),
and all eighteen are caught there too.

Five mutations of my own, against round 3's gate as shipped:

| # | regression | caught by round 3's gate? |
|---|---|---|
| **N1** | the unknown-scope GUARD inside `_arbitrate` is deleted | **NO** -- `42 passed` |
| **N2** | `half_pitch_centres` is off by ONE FINE PIXEL | yes (`4 failed, 37 passed, 1 skipped`) |
| **N3** | the nesting breaks on ODD `N` only | yes (`1 failed, 41 passed`) -- because round 3's lattice test includes `N = 37`, which is the right instinct |
| **N4** | the derivation comment's POPULATION COUNT is edited (`1304` -> `204`) | **NO** -- `42 passed` |
| **N5** | the scope NOTE contradicts the constant it summarises (`flags 192` -> `flags 0`) | **NO** -- `42 passed` |

With `tests/unit/test_verify_b7c_round3.py` added, **all five are caught**
(N1 `1 failed, 46 passed`; N2 `6/40/1`; N3 `1/46`; N4 `1/46`; N5 `1/46`), on
both builds.

---

## 11. Defects

| # | severity | defect | reproducer |
|---|---|---|---|
| **D-1** | **P3** | **A shipped derivation comment contradicts the same round's own finding, and cites a population that does not exist.**  `_lens_traced_uniform._PIXEL_CONTINUITY_SCOPES`'s comment says the enum was "MEASURED on the round-3 population (**642** oracle-scored planes, 16 optics)" -- the population closed at **1304** -- and then says "on the fallback route **NOTHING** already in these diagnostics orders the returned field's fidelity".  Section 7 of the round-3 report says the opposite in its own heading ("Does anything order it?  **Yes** -- two readings already reported"), so does `_lens_traced_multibranch._PIXEL_CONTINUITY_MIN` ("a reading that orders the fallback route's accuracy **DOES exist** and is already reported"), and so does the enum's own note fifteen lines below it | `grep -n '642\\|NOTHING already' lumenairy/elements/_lens_traced_uniform.py` |
| **D-2** | **P3** | **The shipped test file's docstring carries the same stale numbers.**  `tests/unit/test_wp_b7c_round3.py` says "642 oracle-scored planes over sixteen optics", "433 fallback planes", "oracle fidelity 0.4639" (round 3's own worst RETURNED fallback plane is 0.5358; 0.4639 is the round-2 verification's number at a different plane) and "nine candidate readings ... none of which orders that population" | `grep -n '642\\|433\\|0.4639\\|none of which' tests/unit/test_wp_b7c_round3.py` |
| **D-3** | **P3** | **The six dated CORRECTED paragraphs -- the corrections of record -- carry an intermediate pass's counts.**  In `WP-B7c_ROUND2_REPORT.md` they say **367** fold-ring planes, **1196** planes, **437** returned fallback planes, **161** / **215** flagged, **224** below 0.95, **142** below 0.883 and **226** misses.  The closed population is 394 / 1304 / 476 / 192 / 254 / 263 / 152 / 265, and commit `257ced0e` / `00f821e0` claim "every count in the report and the constants is re-read from it" | `grep -n '367\\|1196\\|437\\|161 planes\\|flags 215' docs/audits/.../WP-B7c_ROUND2_REPORT.md` |
| **D-4** | **P3** | **The converged residue's MECHANISM is written into a shipped constant and its own evidence contradicts it.**  `_HALF_PITCH_CENTRE_OFFSET` and section 4 of the report attribute the residue to the two lattices' Voronoi-hull mismatch at the window EDGE.  An edge term must shrink as the window grows past the field; this one does not move by a bit over a 2.3x range of window width (mine) or a 1.75x range (round 3's own table).  The residue follows the PITCH | section 5.1; `control_windowladder_win.json`, `control_gridladder_win.json` |
| **D-5** | **P4** | **A shipped test docstring misattributes the hull-aligned offset.**  `test_b7c3_the_arbiter_lattice_never_moves_the_returned_field` says it compares "the shipped value and round 2's".  Round 2's lattice was `(m - N) dx/2`, i.e. offset **0** -- the SHIPPED value.  `-0.5` is the hull-aligned ALTERNATIVE (the round's own `M14`), so the docstring names the wrong thing and hides that the test is a second measurement of the rejected convention | `tests/unit/test_wp_b7c_round3.py`, the `-0.5` monkeypatch; compare `git show dcaa21f0:lumenairy/elements/_lens_traced_uniform.py \| grep "_N_h / 2.0"` |
| **D-6** | **P4** | **The constant's second reason for not moving is weaker than it reads.**  The all-planes derived centre reproduces (1.0571316 here against 1.0600253 -- both 1.06 to three figures), but the FOLD-RING centre round 3 also computes reads **1.0543406** here against its 1.0619044, so the two constructions straddle 1.055 and the sentence does not say which it means.  The decision is right; the sentence should name its construction and its population, and the "1 % rule" it is paired with is stated nowhere but in the document that applies it, on a margin claim 1 shows cannot be measured | section 4.1; `joined_win.json` keys `derived_centre_all`, `derived_centre_fold` |
| **D-7** | **P4** | **`_N_h` is dead after the E5 refactor.**  `_lens_traced_uniform` still computes `_N_h, _dx_h = 2 * N, 0.5 * dx` but only `_dx_h` is used now that `_xh` comes from `half_pitch_centres`; ruff does not flag it because it is a tuple unpack | `lumenairy/elements/_lens_traced_uniform.py`, the `_E_half` block |
| **D-8** | **P4** | **Three gaps in the gate**, all closed by this verification's file: the unknown-scope guard (N1), the derivation comment's population count (N4) and the scope note's own numbers (N5) each leave the round-3 gate at `42 passed` | section 10 |

### Exact requested edits

1. **D-1**, `lumenairy/elements/_lens_traced_uniform.py`, the
   `_PIXEL_CONTINUITY_SCOPES` comment.  Replace

   > `MEASURED on the round-3 population (642 oracle-scored planes, 16
   > optics; validation/probe_wp_b7c_round3/fallback_win.json): on the
   > fallback route NOTHING already in these diagnostics orders the returned
   > field's fidelity -- see WP-B7c_ROUND3_REPORT.md section 7 for the
   > candidates tried and what each one's best threshold achieves -- so the
   > honest thing the module can do there is say that the field is
   > unarbitrated for accuracy, which is what this key does.`

   with

   > `MEASURED on the round-3 population (1304 oracle-scored planes, 16
   > prescriptions, of which 476 are fallback planes the shipped bars RETURN;
   > validation/probe_wp_b7c_round3/fallback_win.json).  On the fallback
   > route the reading is of the field this call returns but the dominant
   > error is the dark tail the completion did not build, which this arm
   > cannot see.  Two readings the module ALREADY reports do order that
   > population -- the LOSS side of this arm and of the launched-power
   > bracket, whose bars are set for the fold-ring route where a dark deficit
   > does not reach the caller (see _lens_traced_multibranch.
   > _PIXEL_CONTINUITY_MIN for the confusion table) -- and neither refuses
   > there, so what the module can say honestly is that the returned field's
   > dark tail is NOT arbitrated, which is what this key does.`

2. **D-2**, `tests/unit/test_wp_b7c_round3.py`, module docstring: `642` ->
   `1304`, `433 fallback planes` -> `476 returned fallback planes`,
   `0.4639` -> `0.5358`, and replace "nine candidate readings already in these
   diagnostics, none of which orders that population" with "fourteen candidate
   readings, of which the two LOSS arms do order it -- and neither refuses on
   that route".

3. **D-3**, `WP-B7c_ROUND2_REPORT.md`, the six `CORRECTED 2026-09-19`
   paragraphs: `367` -> `394`, `1196` -> `1304`, `437` -> `476`, `161` ->
   `192`, `215` -> `254`, `224 below fidelity 0.95` -> `263`, `142 below
   0.883` -> `152`, `226 misses` -> `265 misses`.

4. **D-4**, `lumenairy/elements/_lens_traced_multibranch.py`,
   `_HALF_PITCH_CENTRE_OFFSET`.  Replace

   > `Where light reaches the window edge the two integrals are therefore not
   > over the identical rectangle, and a converged render reads 1 to O(1/N)
   > rather than exactly.`

   with

   > `Where light reaches the window edge the two integrals are therefore not
   > over the identical rectangle, and a converged render reads 1 only
   > approximately.  TWO terms, and they are distinguishable: the hull
   > mismatch is what an UNDERSIZED window costs (measured 1.4e-3 against
   > 1.2e-4 once the window holds the field, WP-B7c round 3 section 4; the
   > round-3 verification measures 5.0e-4 against 7.3e-5), and it vanishes
   > once the window holds the field -- growing the window further does not
   > move the reading by a bit.  What is LEFT is the two point-sampled
   > quadratures' INTERIOR difference, which follows the PITCH and not the
   > window: holding the window and refining the grid 8x takes it from 7.9e-4
   > to 1.3e-5 (VERIFY round 3, control_gridladder_win.json).`

   and the corresponding sentence in section 4 of `WP-B7c_ROUND3_REPORT.md`
   ("That is exactly what a Voronoi-hull mismatch at the window EDGE
   predicts") should say that the window ladder bounds the hull term and that
   the plateau is the interior term.

5. **D-5**, `tests/unit/test_wp_b7c_round3.py`,
   `test_b7c3_the_arbiter_lattice_never_moves_the_returned_field`: replace
   "the shipped value and round 2's" with "the shipped value and the
   HULL-ALIGNED alternative E5 asked for (`-0.5`, the round's own M14)".

6. **D-6**, `lumenairy/elements/_lens_traced_multibranch.py`,
   `_PIXEL_CONTINUITY_MAX`: replace "the geometric centre of the gap it sits
   in is **1.0600253** over all 1304 planes -- i.e. 1.06 to three figures IS
   the derived centre, and the constant does not move" with "the geometric
   centre of the gap it sits in reads 1.0600253 over all 1304 planes here and
   1.0571316 on the verification's independent 1141 -- both 1.06 to three
   figures -- while the FOLD-RING-only centre reads 1.0619044 here and
   1.0543406 there.  The centre is a reading of the population and the two
   constructions differ at the third figure, so what keeps the constant where
   it is, is the COST table below, which reads identical integers at 1.06,
   1.0600253 and 1.0619 on both populations.".

7. **D-7**: drop `_N_h` from `_N_h, _dx_h = 2 * N, 0.5 * dx`.

---

## 12. Ship recommendation

**Ship it, after edits 1, 2, 3 and 4.**

Everything this round set out to do is done, and the parts that matter most
are right: the mechanism, the placement of the reading, the lattice
convention and the refactor that gives it one definition, the bit identity,
the mutation matrix, and the two labelling fixes.  Two of round 3's central
corrections against its own earlier rounds -- the fidelity populations
overlapping and the gap being a reading of the ladder -- reproduce on optics
it never ran, and one of them (the ladder) is if anything understated.  The
oracle-floor re-measurement reproduces to three significant figures against an
independently written quadrature, and its correction AGAINST the round-2
verification is right for the reason it gives.

Edits 1-3 are documentation, but they are the kind
`docs/TESTING_STANDARDS.md` calls a defect: a constant's stated origin is part
of the constant, and here one comment cites a population that never existed
and states a conclusion the same round refutes twelve lines further down.
Edit 4 is a mechanism written into a shipped constant that its own ladder
contradicts.  None of them changes a decision, so none of them is a blocker
for correctness -- but they are cheap and the next reader will act on them.

Edits 5-7 are optional.

---

## 13. The two maintainer questions

### 13.1 The accept criterion, and the bar

**Recommendation: keep the bar at 1.06, take 0.95 as the accept criterion, and
stop justifying the bar by a centre or a margin -- justify it by the cost
table alone.**  On 661 fold-ring planes over fifteen (prescription, grid)
pairs this verification measures 1.06 costing ONE false refusal (`W`
z = 4925.69 um, reading 1.1072, fidelity 0.9740) and TWO misses at 0.95,
against 5 and 2 for 1.04, 17 and 1 for 1.0177, 71 and 1 for 1.0059, and 1 and
4 for 1.08 -- the same knee round 3 reports, on an independent population and
an independent oracle, with the 1.04 and 1.06 columns landing on the same
integers, and nothing measured better on both axes.  What does NOT survive
intact is either geometric argument for the value: the two-sided margin is a reading of
the z ladder (1.019x/1.058x -> 1.016x/1.005x over three refinements, and a
picometre bisection shows the crossing is a JUMP, so no margin exists at any
ladder density), and the derived centre is a reading of the POPULATION
(round 3's own two constructions read 1.0600253 and 1.0619044; mine read
**1.0571316** and **1.0543406**, so the all-planes centre reproduces to three
figures while the fold-ring one moves from 1.06 to 1.05).  Since the cost
table reads identical integers at 1.06, 1.0600253 and 1.0619 and re-centring
changes no decision on either population, the constant should stay at 1.06
with the derived-centre sentence naming its construction (edit 6).  One caveat
the accept criterion cannot fix: at 0.95 the criterion RETURNS `HN`'s entire
fold ring at 0.9331-0.9557 with 1.22-1.26x the oracle's energy, so a maintainer
choosing 0.95 is choosing to accept a quarter of the returned energy being
spurious above NA 0.4 -- which is a statement about the completion's validity
at high NA, not about this bar.

### 13.2 Should a loss arm REFUSE on the fallback route?

**Recommendation: no -- REPORT the two loss readings' verdicts in the scope
note and leave them non-refusing, because on an independent population they
are safe but nearly inert, and the failure mode E7 names is invisible to
them.**  Round 3's zero-false-alarm property SURVIVES my attempt to refute it:
over 350 returned fallback planes on sixteen (prescription, grid) pairs --
including a slow control at NA 0.027 scanned through its own focus and three
deliberately coarse grids chosen so the completion declines for the commonest
real reason -- I could not find one plane on which `pixel_continuity < 1/1.06`
or `multibranch_power_ratio_bracketed < 0.889` would refuse a field scoring
above 0.95; both arms flag 11 and 16 planes respectively and every one of them
is wrong.  What does not survive is the SENSITIVITY: round 3's arms flag 192
and 254 of 263 wrong planes, mine flag 11 and 16 of 50, and the 23 planes
nothing sees include the worst field in this whole study -- the slow control
at z = 26620.8 um, oracle fidelity **0.2915**, `pixel_continuity` **0.99896**,
bracket **0.9999**, power 1.005x the oracle's, `reason = 'no_fold'`, no
warning.  That is E7's case, worse than round 3's 0.5358 and the round-2
verification's 0.4639, and BOTH loss arms call it healthy; so refusing on them
would pay a Migration cost for a rule that does not address the defect it is
being introduced for.  The union that WOULD be sensitive -- `either loss arm
OR reason != 'no_fold'`, which flags 49 and catches 27 of the 50 -- refuses 22
RIGHT fields here, the best five at fidelity **0.9842, 0.9838, 0.9813, 0.9727
and 0.9723** and all five on the deliberately coarse grids, which round 3's
own table already shows (17 false alarms) and which is the rule a maintainer
reaches for once the pure arms turn out to be inert.  What I would ship instead costs
nothing and carries the whole finding: put the two loss readings' own
verdicts into `pixel_continuity_scope_note` on the fallback route, together
with the sentence that on that route NEITHER arm is a certificate -- a plane
both call healthy has been measured at oracle fidelity 0.29.

---

## 14. Runs

All from `cd /c/tmp/lum_vmb3`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, `--capture=sys` and `-p no:randomly`; every probe prints
`lumenairy.__file__` as its first line and `PYTHONPATH` pins the tree.

| run | build | result | log |
|---|---|---|---|
| `test_verify_b7c_round3.py` alone | win | **5 passed**, 18.8 s | -- |
| the six-file b7c gate + `a16` + `niche_r2` + `niche_r5` + `v5_21` + the durations gate | win | **413 passed**, 2 deselected, 51 warnings, 1513.4 s | `pytest_core_win.txt` |
| the same | **wsl** | **412 passed, 1 skipped** (optional `astropy` absent), 2 deselected, 52 warnings, 1521.6 s | `pytest_core_wsl.txt` |
| `-k census` (whole suite) | win | **45 passed**, 16 349 deselected, 541.8 s | `pytest_census_win.txt` |
| `-k walker` (whole suite) | win | **116 passed, 8 skipped**, 16 270 deselected, 54.2 s | `pytest_walker_win.txt` |
| `-k dispatcher_pin` (whole suite) | win | **511 passed, 6 skipped**, 15 877 deselected, 68.1 s | `pytest_dispatcher_pin_win.txt` |
| `test_public_api.py` + `test_v4_16_2_dispatcher_pin_doc_consistency.py` + `test_audit_except_budget.py` | win | 18 passed, **1 failed** (see below) | `pytest_publicapi_win.txt` |
| `test_audit2609_a15a_durations_staleness.py` after splicing the new ids | win | **4 passed**, 85.6 s | -- |
| the 23-mutation matrix against round 3's gate as shipped | win | **18 of 18 round-3 mutations caught; 20 of 23 overall** | `mutation_round3gate_win.json`, `mut_*_win.txt` |
| the five new mutations against that gate PLUS this file | win | **5 of 5 caught** | `mutation_fullgate_win.json`, `mut_*_win2.txt` |
| the whole 23-mutation matrix against that gate plus this file | **wsl** | **23 of 23 caught** | `mutation_fullgate_wsl.json`, `mut_*_wsl.txt` |
| bit identity, archive to archive, 117 cases | win / **wsl** | **93 identical, 0 moved, 0 newly refused, 0 newly returned, 24 refused on both** on each | `bitid_win.json`, `bitid_wsl.json` |
| the converged reference, the grid ladder, the window ladder, the lattice A/B, the picometre jump | win / **wsl** | identical to the printed digit on both (the jump's two readings differ by one ULP) | `control_*_win.json`, `control_*_wsl.json` |
| `ruff check lumenairy/ tests/ validation/probe_verify_b7c_round3/ validation/oracles/` | **wsl** | **All checks passed!** | -- |
| `scripts/record_history_fingerprints.py --check` | win | **OK: every history document matches its module** | -- |

**The one red is not this round's and predates it.**
`test_installed_metadata_version_matches_source_version` fails because the
editable install's distribution metadata reads 5.47.0 while
`lumenairy.__version__` reads 5.47.1.  Run against the PRE archive
(`C:/tmp/lum_vmb3_pre`, `dcaa21f0`, with none of round 3's changes) it fails
identically (`assert '5.47.0' == '5.47.1'`), which is exactly what round 3
reports and what the test is for.

```
python -m pytest tests/unit/test_wp_b7c_round3.py tests/unit/test_verify_b7c_round3.py \
  tests/unit/test_verify_b7c_round2.py tests/unit/test_verify_b7c_multibranch.py \
  tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py \
  tests/unit/test_audit2609_b7c_multibranch_envelope.py \
  tests/unit/test_audit2609_a16*.py tests/unit/test_niche_r2*.py \
  tests/unit/test_niche_r5*.py tests/unit/test_v5_21*.py \
  tests/unit/test_audit2609_a15a_durations_staleness.py -q --capture=sys -p no:randomly
python -m pytest tests/ -q --capture=sys -p no:randomly -k census   # walker, dispatcher_pin
python -m pytest tests/unit/test_public_api.py \
  tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
  tests/unit/test_audit_except_budget.py -q --capture=sys -p no:randomly
MUT_TREE=C:/tmp/lum_vmb3_mut python validation/probe_verify_b7c_round3/v3mutate.py out.json
python validation/probe_verify_b7c_round3/v3bitid.py C:/tmp/lum_vmb3_pre C:/tmp/lum_vmb3_post bitid_win.json
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmb3 && ~/lumvenv/bin/ruff check lumenairy/ tests/ \
  validation/probe_verify_b7c_round3/ validation/oracles/'
python scripts/record_history_fingerprints.py --check
```

`.test_durations` carries the five new ids with their measured values and
reloads as valid JSON (**16 315** entries), so a sharded CI run schedules none
of them as unknown-duration.

---

## 15. What I could not measure

* **the oracle-scored population on the wsl build.**  As in both earlier
  rounds it is a win measurement.  What IS measured on both builds is every
  DECISION probe -- the converged reference, the two ladders, the lattice A/B,
  the bit-identity matrix and the whole mutation matrix -- and all of them
  agree to the printed digit;
* **round 3's own 1304-plane population.**  I re-derived the claims on my
  own 1141-plane one rather than re-running theirs, which is the point of an
  independent verification but means "1304" itself is a count I take on
  trust;
* **a fourth ladder refinement.**  Each pass still found new crossings, and
  the picometre bisection shows why: the limit is not a resolution but a
  discontinuity, so a fourth pass would lower the population gap again without
  telling anyone anything new;
* **the oracle floor on more than the two rows the brief asks for.**  `Q`
  (z = 5680 um) and `W` (z = 4900 um) completed and both reproduce to three
  significant figures; a third row at NA 0.44 (`VX` z = 900 um) was still
  running when this closed.  The exact full-radius quadrature needs ~4000
  azimuthal Gauss-Legendre nodes per radius at that speed, and costs 25-60
  minutes per plane;
* **`VX`'s and `VA`'s rows of the `J0` ceiling.**  The two rows I have span
  `y_max/z` 0.108-0.109 only, so the NA axis of E6's table is taken on trust
  above that.  What IS measured across the whole NA range is the arm the
  population is actually scored with: the angular spectrum moves by 1.3e-5 in
  fidelity when its fine window is tripled, on 24 planes spanning fidelity
  0.015 to 0.996;
* **the FALLBACK population on round 3's own optics.**  Mine reaches the
  near-caustic fallback regime through three deliberately coarse grids and a
  slow control rather than through their `M`, `S`, `P`, `Z` and `C`, so the
  two fallback populations differ in composition (306 of my 350 returned
  planes read `reason = 'no_fold'` against 271 of their 476).  That difference
  is the reason the loss arms' SENSITIVITY differs, and it is also why I would
  not make either arm refuse on the strength of one sampling;
* **an accept criterion independent of the oracle's fidelity.**  Energy error
  tracks fidelity closely here (the worst returned fold-ring fields carry
  1.22-1.26x the oracle's power), but the two are not independent axes;
* **whether `HN`'s whole-ring miss is a property of NA or of that
  prescription.**  Both round 3's `HN` and my `VX` (NA 0.437, a different
  glass and a different form) return fold rings the arbiter is silent on, but
  `VX`'s returned fields score 0.973-0.996 where `HN`'s score 0.933-0.956, so
  speed alone does not predict it;
* **the wall-clock cost of the change**, which round 3 argues is zero by
  construction and which my bit-identity matrix supports (0 moved of 117), so
  I did not time it either.
