# VERIFY-WP-B7c -- independent adversarial re-verification of WP-B7c (the multibranch blow-up and the fold envelope's two constants)

Branch `verify/wp-b7c` off `fix/multibranch-zeta-envelope` (`f6c7eec7`,
`e3a7ee98`, `a985bf42`, `119b8bb6`), base `96cb2096`.  Worktree
`C:/tmp/lum_vmb`; the base tree `C:/tmp/lum_vmb_pre`.  I did not write WP-B7c.

Every number below is mine, taken on **five prescriptions WP-B7c never used**
and on **two builds**:

| build | interpreter | numpy | tree |
|---|---|---|---|
| **win** | Windows py3.14.6 | 2.4.4 | `C:/tmp/lum_vmb` (head) and `C:/tmp/lum_vmb_pre` (`96cb2096`) |
| **wsl** | WSL py3.12.3 | 2.4.6 | `/mnt/c/tmp/lum_vmb` |

Probes and their JSON: `validation/probe_verify_b7c/`.  Decision tests:
`tests/unit/test_verify_b7c_multibranch.py`.

---

## 0. The fixtures, and why these

WP-B7c derived on V (N-BAF10 biconvex / 1.064 um), C (N-BK7 plano-convex,
flat-first / 780 nm), D (N-SF11 biconvex / 1.55 um) and WP-B7b's two N-LAK22
singlets (850 nm, 1.55 um) -- five biconvex-or-plano SINGLETS, all with
UNDERCORRECTED spherical aberration, all scored on ONE grid each.  Mine share
no glass, wavelength, grid or surface count with any of them, and two are
shapes that study did not have:

| id | prescription | lambda | grid | why |
|---|---|---|---|---|
| **Q** | **cemented DOUBLET** N-BK7 R = +3.0 / -2.0 mm + N-SF6 -2.0 / -6.0 mm, 1.40 mm aperture | 1.31 um | N = 512, dx = 2.60 um | three surfaces, a buried cemented interface, and **OVERCORRECTED** SA -- marginal focus 5642 um BEYOND paraxial 5444 um, the opposite sign to every optic in the study |
| **M** | positive **MENISCUS** N-SF6 R = +1.6 / +4.2 mm, 1.10 mm aperture | 2.0 um | N = 512, dx = 3.60 um | both centres of curvature on one side; longest wavelength in either study |
| **S** | N-SK16 **plano-convex, CONVEX side first**, R = +2.4 mm, 1.00 mm aperture | 532 nm | N = 512, dx = 1.50 um | shortest wavelength in either study, and the orientation opposite to WP-B7c's own plano-convex |
| **F** | fast N-LASF9 biconvex R = +/-2.2 mm stopped to 0.60 mm (**f/2.0**) | 633 nm | N = 640, dx = 1.40 um (`F_alt`) and N = 512, dx = 1.80 um | the fast singlet the brief asks for, inside the oracle's envelope |
| **P** | the same N-LASF9 biconvex at its full 1.00 mm aperture (**f/1.2**) | 633 nm | N = 512, dx = 1.80 um | kept as the case that exposes the shared oracle's NA limit -- see section 1 |

Two grids per optic (`*_alt`), and the band is read on both -- see section 3.4.

---

## 1. The oracle, and its floor

`validation/oracles/caustic_fold_truth.py` -- the same lumenairy-free direct
Rayleigh-Sommerfeld ring integral over an exact meridional conic trace that
WP-B7b, VERIFY-B7b and WP-B7c scored against.  My driver
(`validation/probe_verify_b7c/oracle.py`) shares no code with WP-B7c's:
Sellmeier coefficients are typed here from the Schott catalogue (control
against `lumenairy.glass.get_glass_index`: **delta = 0.0** for N-LASF9, N-BK7,
N-SF6 and **2.2e-16** for N-SK16), the doublet's buried interface is handled,
and the radial answer is rotated onto the library's own grid.

Floor, measured per fixture (`probe5_oraclefloor.py` / `oraclefloor_win.json`):

| fixture | z [um] | `y_max/z` | energy closure | `n_fan` 6000 -> 12000 rel L2 | `n_rho` 2400 -> 4800 rel L2 | rms radius of the field [um] | Debye `J0` phase error [rad] |
|---|---|---|---|---|---|---|---|
| **S** | 3214.78 | 0.152 | 0.99962 | 2.0e-04 | 2.4e-03 | 6.4 | **0.0017** |
| **M** | 2242.17 | 0.240 | 0.99916 | 2.3e-04 | 2.2e-03 | 16.8 | **0.011** |
| **Q** | 5680.00 | 0.121 | 0.99978 | 1.7e-04 | 8.5e-04 | 12.6 | **0.00098** |
| **F_alt** | 1056.00 | 0.278 | 0.99868 | 3.1e-04 | 7.0e-03 | 11.7 | **0.050** |
| **P** | 888.00 | **0.552** | **0.98989** | 3.0e-04 | **1.9e-02** | 40.3 | **2.76** |

So on the four optics I score, the oracle is converged to **rel L2 ~1e-3 to
7e-3** (the radial sampling dominates; the ray fan is already converged to
3e-4), conserves energy to **0.1 % or better**, and its `J0` expansion
parameter stays below **0.05 rad**.  A fidelity DIFFERENCE below about
**4e-4** is inside that floor and I do not read one.  P is the outlier on
every column at once.

**The oracle has an NA ceiling, and P is past it.**  The azimuthal integral is
taken in the Debye `J0` form, exact to `O(k (y rho / R0^2)^2)`.  At P's
f/1.2 that parameter reads **2.76 rad** at the rms radius the light actually
occupies -- 56x the worst of the other four and 1600x S's -- and the consequence is measurable: at z = 888 um on P the library's
OWN two members disagree with EACH OTHER at fidelity **0.737**
(`caustic='uniform'` vs `caustic='wave'`), and the oracle sits between them
(0.791 / 0.722) -- stable to 1e-3 under a 16x refinement of `n_fan` and
`n_rho`, so this is a model gap, not sampling.  **The four optics I score span `y_max/z` 0.12-0.28, which brackets
WP-B7c's own five (semi-aperture over focal length, 0.12-0.29); P is at
0.55.**  P is therefore
excluded from the accept/broken classification and reported only on the
library-side axes (the ratio, the decision, bit identity), with the limitation
recorded as a finding of its own (D5 below).

---

## 2. Verdict table

Legend: **CONFIRMED** = reproduces on my fixtures with margin; **BOUNDED** =
reproduces, but only over a narrower population than the report states;
**RESTATED** = true as measured, false as written; **REFUTED** = my measurement
contradicts it.

| # | WP-B7c's claim | verdict | my numbers (win; wsl where run) |
|---|---|---|---|
| 1 | the blow-up is the rasteriser's POINT-SAMPLED AREA QUADRATURE losing unbiasedness when a ring collapses onto a handful of pixels | **CONFIRMED, and established more strongly than the report does** | 100 % of launch triangles sub-pixel at EVERY plane on all four optics (median mapped area 1/188 to 1/10 331 of a pixel); a control WP-B7c did not run settles it -- holding the launch lattice and the window fixed and halving the PIXEL divides the excess by exactly 4 (S 107.3 -> 27.3 -> 7.40; M 504 -> 127 -> 32.2; Q 11238 -> 2810 -> 703; F_alt 2.98 -> 1.42 -> 1.04) while a healthy plane is invariant (0.937 -> 0.937 -> 0.938).  Identical on wsl to all printed digits |
| 1a | control: `caustic_band='plain'` reproduces to 0.1 % | **CONFIRMED at the blow-up plane, RESTATED elsewhere** | blow-up planes 0.9 % / 2.0 % / 2.1 % / 3.1 % (M / S / F_alt / Q) -- the ordering is not the cause.  But at HEALTHY fold planes the same switch moves the reading by **4.6 % .. 55 %** (Q: ludwig 0.823 vs plain 1.277), i.e. the quantity the bar reads depends on a user-settable argument by more than the bar's own margin |
| 1b | control: `min_area_ratio` 1e-8 vs 1e-6 bit-identical | **REFUTED as general; reproduces on 2 of 4** | identical on M and F_alt; on **Q** 1e-8 reads **2.22e+06** against 1e-6's 11238 (197x) and on **S** 220.2 against 107.3 (2.05x).  The clip is operative at the default on two of my four optics |
| 1c | control: refining `ray_subsample` 2 -> 1 makes it 7.4x WORSE | **REFUTED as general; reproduces on 3 of 4, inverts on the 4th** | S 16.2 -> 107 -> 8805 and M 36.8 -> 504 -> 20055 (slope -4.54 in the launch pitch, i.e. worse than the count argument predicts), F_alt 1.20 -> 2.98 -> 11.3; but **Q** 39543 -> 11238 -> **1.80**, i.e. refining makes it 6200x BETTER.  Healthy planes invariant on all four (slope 0.0000 .. -0.017) |
| 2 | the fix REFUSES outside a derived band and the fallback is not a remedy | **CONFIRMED -- BOUNDED by D1 / D2** | 18 refused planes over 5 optics, every one genuinely broken (oracle fidelity **0.181 .. 0.641**, or 0.292 .. 0.637 excluding P, against the hand-off's 0.986 .. 0.999 at the SAME planes); **0 false refusals in 71 scored planes**.  Fallback confirmed useless: M z = 2330 / 2342.24 and S z = 3253.91 / 3273.95 fall back and return fields at fidelity 0.778-0.847 |
| 2a | the band: 42 accepted read 0.816-1.246, 9 broken read >= 5.848, "between 1.246 and 5.848 there is nothing", margins 1.60x / 2.92x in a 4.69x gap | **REFUTED** | on 54 oracle-scored planes (4 optics, P excluded) the two populations **overlap**: largest ACCEPTED bracketed reading **0.9804**, smallest BROKEN reading above it **1.106** (all planes) / **1.151** (fold-ring only).  **Gap 1.13x / 1.17x**, against the brief's 2x defect threshold.  The bar's margin BELOW is 2.04x; its margin ABOVE is **0.55x-0.58x, i.e. the smallest broken reading sits 1.74x-1.81x INSIDE the bar**.  The "empty gap" is an artefact of plane sampling: on four optics the reading passes continuously through (1.25, 5.85) -- P 1.34 / 1.79 / 1.82 / 2.25 / 4.49, Q 1.35 / 1.53 / 1.83 / 2.04 / 2.72 / 4.07 / 5.31, M 4.53 / 4.76, F 2.01 / 1.23 / 4.15 / 1.51 / 3.48 |
| 3 | `_ZETA_EXTRAPOLATION_MAX = 8.0` is a SIGN boundary: signed and centred below (-2.18 .. +4.14 %, 12/22 negative, mean +0.57 %), one-sided gain above (+3.64 .. +30.13 %, 0/8) | **REFUTED below the bar, CONFIRMED above it** | 30 fold rungs on 4 optics, completed power / oracle power: BELOW the bar (17 rungs, `zeta_x` 0.64 .. 7.64) **-1.02 % .. +28.44 %, 1 of 17 negative, mean +7.00 %**; ABOVE (13 rungs, 10.8 .. 1779) **+3.92 % .. +12.43 %, 0 of 13, mean +8.20 %**.  Excluding the two rungs contaminated by the section-1 quadrature (`n_branch_max > 2`): below **-1.02 % .. +17.39 %, 1 of 15**.  The **largest excursion in the whole study is BELOW the bar** (F_alt, `zeta_x` = 1.93, +28.4 %); the two populations overlap completely |
| 4 | member default: both orderings reproduce on their own fixtures; the hand-off is closer off WP-B7b's optic; "beats plain multibranch" holds at 40/42 | **CONFIRMED (RESTATEMENT HOLDS)** | on my 29 accepted planes the hand-off is closer at **28** (by 0.0010 .. 0.0242), the completion at **1** (S z = 3188.70, by 0.0003 -- inside the oracle's own floor, so a tie); the completion beats the plain branch sum at **29 of 29**.  The hand-off's power against the oracle stays in **0.973 .. 1.019** at every one of the 54 planes, including all 18 the completion is refused at |
| 5 | `zeta(r)` beyond the band: `kappa_eff/kappa` 1.068 -> 0.489; a second fold parameter is the repair | **NOT VERIFIED** -- outside the tasks I was given; no independent `kappa_eff` fit was run |
| 6 | `_AIRY_TAIL_CELLS` clip refused: > 94 % of the excess is inside 3 Airy lengths; `zeta_linear_range` is REPORTED, not applied | **CONFIRMED, both arms, by construction** | 15 fold planes on M / S / Q: the annulus ratio is flat beyond 3 cells (worst case 2.074 at 3 against 1.977 at 20, 4.9 %), so **>= 95 % of the excess is inside 3 Airy lengths** at every plane where there is an excess to attribute.  "Never applied" proved by construction: `_trace_meridional_fold` monkey-patched to return `zeta_linear_range` = 1e-12 m and 1e+6 m returns a **byte-identical** field at all 15 planes.  `u*/l_airy` reads **7.74 .. 32.0** (report: 6.93 .. 144) and is SHORTER than the 20-cell fill on the doublet at all three of its fold planes |
| 7 | movers 11/13 byte-identical; the 2 movers are the blow-up planes, now raising | **CONFIRMED on a larger matrix** | archive-to-archive `96cb2096` vs `119b8bb6`, child processes with `cwd` and `PYTHONPATH` set to the archive and `lumenairy.__file__` asserted under it, SHA-256 over `tobytes()`: **29 of 32 identical, 3 moved**, and all three are refused planes (base returned a field carrying 3.2x-504x the launched power; head raises).  Includes both grids, `caustic_band='plain'`, complex64, `output_plane_distance=0`, `ray_subsample=4`, and the two "gap" planes -- which are byte-identical |
| 8 | one environmental red: `test_public_api.py::test_installed_metadata_version_matches_source_version` | **NOT REPRODUCED** | `tests/unit/test_public_api.py` + `test_v4_16_2_dispatcher_pin_doc_consistency.py`: **15 passed**.  `importlib.metadata.version('lumenairy')` on this interpreter reads **5.47.0**, equal to `__version__`; the editable install has been refreshed since WP-B7c ran |

---

## 3. Task A -- the band, re-derived

`probe1_band.py` on both trees, joined on `z` by `summarise.py`.  The
classification ("does the oracle accept this completed field") can only be
read on a build that still RETURNS the field, so the base tree supplies
`fid_uni` at the planes the head refuses and the head supplies the DECISION.
Accept bar: the report's own, fidelity >= 0.883.

### 3.1 The populations

| population | n | bracketed multibranch reading | oracle fidelity |
|---|---|---|---|
| accepted | 29 | **0.743 .. 0.980** | 0.962 .. 0.993 |
| broken, gain side | 3 | **1.106 / 1.151 / 1.226** and up | 0.858 / 0.837 / 0.787 |
| broken, loss side | 6 | 0.414 .. 0.762 | 0.778 .. 0.847 |
| refused by the head | 18 | 2.04 .. 1.12e+04 | 0.181 .. 0.641 |

**Closest accepted from below: 0.9804** (M, z = 2201.74 um, fidelity 0.9798).
**Mildest broken from above: 1.151** (F_alt, z = 1076 um, fidelity 0.8580) on
fold-ring planes, **1.106** (Q, z = 5400 um, fidelity 0.8366) over all planes
the guard can fire on.  **Gap 1.17x / 1.13x.**  The brief's criterion -- "if
the gap closes below 2x, the bar is a defect" -- is met.

### 3.2 The single sharpest counterexample

**F_alt, z = 1076.0 um.**  `reason='fold_ring'`, `fell_back=False`,
`zeta_extrapolation = 1.929` (inside the docstring's own best band),
`power_ratio_decision = 'ok'`, bracketed reading **1.1513** -- and the returned
field has oracle fidelity **0.8580** and carries **1.284x** the oracle's power,
with `n_branch_max = 37`.  This is exactly the R-5 defect class WP-B7c set out
to close ("its own diagnostics read their best values on a broken field"),
surviving the fix at a reading 1.74x below the bar.  The one diagnostic that
moves is `fit_residual` (0.1006 against 0.0608 one plane earlier), still inside
the module's own 0.15 gate.

### 3.3 The gap is a sampling artefact

Stepping `z` finely through the transition, the reading passes continuously
through the interval WP-B7c reports as empty:

| optic | readings measured inside (1.246, 5.848) |
|---|---|
| Q (doublet) | 1.289, 1.352, 1.428, 1.532, 1.659, 1.826 (returned) / 2.036, 2.324, 2.718, 3.262, 4.071, 5.306, 5.558, 4.133, 3.216, 2.584 (refused) |
| P | 1.340, 1.771, 1.787, 1.802, 1.818 (returned) / 2.247, 3.814, 4.490, 4.548 (refused) |
| M | 4.533, 4.763 (refused) |
| F | 1.229, 1.262, 1.514, 1.696 (returned) / 2.008, 4.146, 3.480 (refused) |

and on **F** the decision is not even monotone in `z`: at 0.5 um steps the
sequence over z = 1077.0 .. 1080.0 um is refuse (2.008), return (1.229),
return (1.262), refuse (4.146), return (1.514), return (1.696), refuse (3.480).

### 3.4 Two grids

The band was read on two grids in three ways, and all three say the same thing.

1. **Oracle-scored, two grids on the same optic.**  The fast singlet at
   dx = 1.80 um (`F`, N = 512) never produces an accepted `fold_ring` plane --
   its Airy layer is under-resolved and the module falls back over the whole
   window -- while at dx = 1.40 um (`F_alt`, N = 640) it gives a 13-rung fold
   ladder.  Same optic, same planes, different guard population.
2. **Byte identity on the alternate grid of every optic.**  `P_alt`, `Q_alt`,
   `M_alt`, `S_alt` (N and dx both changed) are in the 32-fixture matrix of
   section 8 and are byte-identical base-to-head.
3. **The pixel-refinement ladder of section 4.3**, which is the two-grid
   question asked cleanly -- optic, plane, launch lattice and window all held,
   only `dx` changed.  The reading the bar reads moves by 4x per halving at a
   blow-up plane and not at all at a healthy one.

So the grid is not a tie-breaker between readings; it IS the variable the
reading is most sensitive to near the bar.  That is defect **D2**.

---

## 4. Task B -- the mechanism, re-measured

`probe2_mech.py`, four arms, on S / M / Q / F_alt.  S was re-run in full on
**wsl py3.12.3 / numpy 2.4.6** and every figure below reproduces to all printed
digits (`mech_S_head_wsl.json` against `mech_S_head_win.json`), so none of this
is a BLAS artefact.

### 4.1 The sub-pixel statistic, derived independently

For a rotationally-symmetric collimated launch the mapped-area ratio at radius
`r` is `|y dy/dr / r|` with `y(r)` the geometric landing height, taken from the
ORACLE's exact meridional trace, not from the library.  The mapped area in
pixels is `ratio * 0.5 h^2 / dx^2`, `h = ray_subsample * dx`:

| optic | plane | `Apx` p05 / p50 / p95 | fraction sub-pixel | max `1/sqrt(ratio)` |
|---|---|---|---|---|
| M | healthy | 1.9e-04 / **2.34e-03** / 7.1e-03 | **100 %** | 1184 |
| M | blow-up | 8.4e-05 / 5.08e-04 / 1.4e-03 | 100 % | 2.3e+04 |
| S | healthy | 3.7e-05 / 4.62e-04 / 1.5e-03 | 100 % | 2700 |
| S | blow-up | 1.6e-05 / 9.68e-05 / 2.7e-04 | 100 % | 4573 |
| Q | healthy | 1.4e-04 / 1.10e-03 / 2.5e-03 | 100 % | 2058 |
| Q | blow-up | 4.5e-06 / 8.62e-04 / 5.2e-03 | 100 % | 1.2e+04 |
| F_alt | healthy | 3.8e-04 / 5.33e-03 / 1.5e-02 | 100 % | 3199 |
| F_alt | blow-up | 3.0e-04 / 2.79e-03 / 9.7e-03 | 100 % | 1733 |

M's healthy median is **1/427 of a pixel**, which is WP-B7c's own 1/430 on a
different optic at a different wavelength -- the statistic is a property of the
lattice-to-pixel ratio, exactly as the report says.  100 % sub-pixel at healthy
planes too, so sub-pixel area alone is not the pathology.

### 4.2 The `ray_subsample` inversion

| optic | blow-up: 4 / 2 / 1 | log-slope vs pitch | healthy: 4 / 2 / 1 | slope |
|---|---|---|---|---|
| S | 16.2 / 107.3 / 8805 | **-4.54** | 0.9257 / 0.9271 / 0.9275 | -0.0014 |
| M | 36.8 / 504.4 / 20055 | **-4.54** | 0.9154 / 0.9153 / 0.9151 | +0.0003 |
| F_alt | 1.20 / 2.98 / 11.26 | -1.61 | 0.9366 / 0.9371 / 0.9372 | -0.0004 |
| Q | 39543 / 11238 / **1.80** | +7.21 | 0.8078 / 0.8230 / 0.8270 | -0.017 |

The inversion reproduces on three of four and is STEEPER than WP-B7c's 7.4x
(S and M lose 4.5 decades per decade of pitch).  On the doublet it inverts the
other way.  Either way the healthy arm is invariant, which is the two-sided
control the claim needs.

### 4.3 The control WP-B7c did not run -- and the cleanest evidence for its own diagnosis

Hold the optic, the plane, the LAUNCH LATTICE (`ray_subsample * dx`) and the
physical window fixed; refine only the OUTPUT PIXEL:

| optic | plane | x1 | x2 | x4 | `n_branch_max` |
|---|---|---|---|---|---|
| S | blow-up | 107.3 | 27.32 | 7.399 | 37 at all three |
| M | blow-up | 504.4 | 126.6 | 32.15 | 105 at all three |
| Q | blow-up | 11238 | 2810 | 702.8 | 45 at all three |
| F_alt | blow-up | 2.982 | 1.420 | 1.036 | 61 at all three |
| S | healthy | 0.9271 | 0.9129 | 0.9129 | 2 |
| M | healthy | 0.9153 | 0.9108 | 0.9224 | 2 |
| Q | healthy | 0.8230 | 0.8009 | 0.8004 | 2 |
| F_alt | healthy | 0.9371 | 0.9365 | 0.9376 | 2 |

Exactly `1/4` per halving of `dx`, at constant branch count.  That is the
quadrature identity: each triangle deposits `dx^2 |E|^2 / ratio` wherever it
catches a pixel CENTRE, `ratio` is scale-free once the launch lattice is
fixed, and the number of catches does not fall -- so the written energy scales
with the PIXEL AREA and not with the mapped area.  WP-B7c's diagnosis is
right, and this is a stronger statement of it than the report makes.  It is
also the reason for defect **D2**.

### 4.4 The three alternative hypotheses, tested independently

* **fold-member ordering** -- ruled out at the blow-up planes on all four
  optics (`plain` vs `ludwig` 0.9-3.1 %), but NOT a negligible knob elsewhere:
  4.6 % / 6.1 % / 7.7 % / 55 % at the healthy planes of M / F_alt / S / Q;
* **the Jacobian clip** -- NOT inoperative on Q and S (section 2, claim 1b);
* **branch selection** -- ruled out by construction: the doublet refuses at
  `n_branch_max == 1` (no coalescence at all), so a wrong branch cannot be the
  mechanism there, and the pixel-refinement scaling above holds at fixed
  branch count.

---

## 5. Task C -- the zeta envelope's sign boundary

Ladders on F_alt (13 rungs, `zeta_x` 1.93 .. 1779), M (7, 2.29 .. 156),
S (7, 3.63 .. 246), Q (3, 0.64 .. 0.81); completed-field power against the
oracle's.

| | rungs | range | negatives | mean |
|---|---|---|---|---|
| below `_ZETA_EXTRAPOLATION_MAX` | 17 | **-1.02 % .. +28.44 %** | **1 of 17** | **+7.00 %** |
| above it | 13 | **+3.92 % .. +12.43 %** | 0 of 13 | +8.20 % |

Per optic, the largest below-bar excursion is +28.4 % (F_alt), +17.4 % (M),
+14.1 % (S), +4.9 % (Q); the largest above-bar excursion is +12.4 % (M),
+12.2 % (S), +5.3 % (F_alt).  **8.0 does not sit inside a signed/one-sided gap
on my optics: there is no such gap.**  What does reproduce is the ABOVE-bar
half of the claim -- 0 of 13 rungs negative, mirroring WP-B7c's 0 of 8.

Read instead against the branch sum's own `launched_power` (no oracle at all)
the picture is the same: below-bar -1.7 % .. +24.8 %, above-bar +1.7 % ..
+11.6 %.

---

## 6. Task D -- member selection

Scored at every accepted plane: the completion (`caustic='uniform'`), the
hand-off (`caustic='wave', amplitude_model='ray_density'`), the plain branch
sum, and the screen hand-off.

* the hand-off is the closer member at **28 of 29** accepted planes; the
  completion at **1** (S z = 3188.70, by 0.0003).  Margins 0.0003 .. 0.0242 --
  and the single completion win is INSIDE the oracle's own floor: a relative
  L2 change of 2.4e-3 in the reference moves a fidelity of 0.986 by up to
  ~4e-4, so that plane is a tie, not a win.  The hand-off's 28 wins are
  0.0010 .. 0.0242, i.e. 2.5x to 60x the floor;
* the completion beats the plain branch sum at **29 of 29**;
* the hand-off's power against the oracle never leaves **0.973 .. 1.019**,
  including at all 14 refused planes of the four oracle-valid optics, where
  its fidelity is **0.986 .. 0.999** against the completion's 0.29 .. 0.64.

So the docstring's restatement holds as written: the ranking is optic-dependent
and the completion wins only on WP-B7b's own optic, while the energy statement
("the hand-off conserves where the completion above the bar does not") is
general on my fixtures too.  WP-B7c's caveat about the oracle not being neutral
between the hand-off and the completion applies unchanged to my measurement and
I record it rather than claim past it.

---

## 7. Task E -- `zeta_linear_range` and the fill depth

`probe3_tail.py`, 15 fold planes on M / S / Q.

* **reported**: `zeta_linear_range`, `zeta_curvature`, `zeta_linear_resid`,
  `dark_fill_depth` and `l_airy` present, finite and positive at every fold
  plane.  `u*/l_airy` = **7.74 .. 32.0**; shorter than the 20-cell fill on all
  three doublet planes (7.74 / 8.73 / 9.67), as the report predicts happens on
  some optics;
* **never applied**: `_trace_meridional_fold` monkey-patched to return
  `zeta_linear_range = 1e-12` m (far inside one Airy length) and `1e+6` m, with
  every other key untouched -- **byte-identical field at all 15 planes, on both
  arms**, while the diagnostic reports the injected value.  A mutant tree that
  DOES apply it (`dark = ... r_c + min(_AIRY_TAIL_CELLS*l_airy, u*)`) turns the
  corresponding decision test red at byte 1855376, so the test is not passing
  for the wrong reason;
* **the 3-Airy concentration**: the completed/oracle annulus ratio is flat
  beyond 3 cells -- e.g. M z = 2201.74: 2.238 / 2.143 / **2.074** / 2.026 /
  1.994 / **1.977** at 1 / 2 / 3 / 5 / 10 / 20 cells (4.9 % from 3 to 20).  The
  fraction of the excess written inside 3 Airy lengths is **>= 1.0** at 11 of
  15 planes and 0.71 at one; the remaining three have essentially no excess to
  attribute (ratio within 4 % of 1) and the fraction there is not meaningful.
  **The premise WP-B7b proposed clipping on does not hold, exactly as WP-B7c
  reports.**

---

## 8. Task F -- bit identity on my own fixtures

`bitid.py`.  `git archive 96cb2096 lumenairy` and
`git archive 119b8bb6 lumenairy` extracted to two scratch trees; a CHILD
process per tree with `cwd` AND `PYTHONPATH` set to it and `lumenairy.__file__`
ASSERTED under it before any field is built; SHA-256 over
`numpy.ndarray.tobytes()`.

**32 fixtures, 29 identical, 3 moved.**  The 32 span, for each of P / Q / M / S:
a healthy fold plane, a fallback plane, the exit vertex (`output_plane_distance
= 0`), the branch sum at the fold plane and the branch sum at the blow-up
plane; plus all four alternate grids, `caustic_band='plain'`, a complex64
input, `ray_subsample=4`, the two planes inside the "empty" ratio band, and the
three refused planes.

| mover | base | head |
|---|---|---|
| `P_uni_refused` (z = 1015.34 um) | field, grid power 8.051e-07 W, `power_ratio` 4.490, 3 warnings | `RuntimeError` |
| `Q_uni_refused` (z = 5460 um) | field, 1.147e-06 W, `power_ratio` 3.216, 2 warnings | `RuntimeError` |
| `P_uni_blow` (z = 1016 um) | field, 9.852e-06 W, `power_ratio` 54.94, 3 warnings | `RuntimeError` |

Every other path is byte-identical, including the branch sum at the SAME
blow-up planes, both "gap" planes (`P_uni_gap` 1.34, `Q_uni_gap` 1.83 -- both
returned unchanged) and the complex64 input.

**Fail-before, on the base tree** (`failbefore_base_win.json`, child process,
`lumenairy.__file__` asserted): at all five of P 1015.34 / P 1016 / Q 5460 /
M 2343 / S 3274.02 the base returns a field carrying **3.2x to 504x** the
launched power, with `power_ratio_decision`, `multibranch_power_ratio` and
`n_branch_max` **absent from the diagnostics entirely**.

---

## 9. Task G -- durability of the constants in `test_audit2609_b7c_multibranch_envelope.py`

| constant | origin | two-sided margin | premise gating | verdict |
|---|---|---|---|---|
| `_WL/_DX/_N/_W0`, `_HEALTHY_Z`, `_BLOWUP_LADDER` | VERIFY-B7b's fixture, restated | n/a (fixture) | -- | sound |
| `largest_accepted = 1.246` | WP-B7c's own 51-plane study, a LITERAL, not re-measured at runtime | asserted `> 1.5x` below the bar | none (build-free) | **not durable -- my measurement puts the largest ACCEPTED reading at 0.9804 and a BROKEN one at 1.106/1.151, so the literal is not an envelope and `_MB_POWER_RATIO_MAX / largest_accepted > 1.5` is testing a number the world does not obey.** The test is build-free, so it will never go red; it pins a claim rather than a decision |
| `smallest_broken = 5.848` | same | asserted `> 2.5x` above the bar | none | **not durable -- my smallest broken reading is 1.151, so `smallest_broken / _MB_POWER_RATIO_MAX > 2.5` passes only because the literal is 5.1x too large** |
| `last_signed = 5.653`, `first_gain = 11.05` | WP-B7c's three-optic ladder, literals | asserted 1.4x / 1.3x | none | **not durable -- section 5 finds no signed/one-sided transition at all on four other optics** |
| `_AIRY_TAIL_CELLS` arm: `Ai(x)/Ai(0) < 1e-12` inside the fill; `Ai(3)/Ai(0) > 1e-4` | SOLVED on the running build with scipy | `Ai(3)/Ai(0) = 1.85e-02` against a 1e-4 bar, two decades | unconditional | **sound -- the one derived-at-runtime constant in the file** |
| member arm: `abs(p_wv/p_in - 1) < 0.02`, `p_uni/p_in > p_wv/p_in + 0.02`, `abs(p_uni/p_in - 1) < 0.05`, planes 1594.9 / 1683.4 um, aperture factor 0.98 | measured on V; `0.98` matches the library's own `launch_radius = 0.5*aperture*0.98` | the `+0.02` separation is measured at ~+6 % on V | the plane's side of `_ZETA_EXTRAPOLATION_MAX` is asserted first | sound in shape; the bars are one-optic numbers but the premise gate protects them |
| the `pytest.skip` in `..._refusal_is_never_the_first_diagnostic` and `..._public_caustic_uniform_entry_point_refuses_too` | premise gate | -- | **no invariant arm** -- if the ladder stops firing, these two tests become no-ops with nothing asserted (unlike `..._is_refused_not_passed_through`, which has one) | weak; note, not a defect |

**Mutation matrix for my own decision tests** (each mutant a `git archive` of
`119b8bb6` with one edit, run in a child process with `lumenairy.__file__`
asserted):

| mutant | tests killed | collateral |
|---|---|---|
| the dark fill clipped by `zeta_linear_range` | `..._zeta_linear_range_is_reported_and_never_applied` | none (5 pass) |
| `_mb_bracket = max(...)` instead of `min(...)` | `..._gain_bracket_headroom_exceeds_the_refusal_bar`, `..._refusal_fires_with_a_single_valued_ray_map` | none (4 pass) |
| `_MB_POWER_RATIO_MAX = 1.0` | `..._an_accepted_field_reads_above_the_pinned_accepted_ceiling`, `..._refusal_follows_the_pixel_not_the_field` | none (4 pass) |

Every one of my six tests is killed by at least one mutation.

---

## 10. Defects

| id | severity | statement | reproducer |
|---|---|---|---|
| **D1** | **P2** | **The refusal bar has no upper margin.**  The smallest BROKEN bracketed reading I measured is **1.151** (F_alt z = 1076 um, `reason='fold_ring'`, `fell_back=False`, `zeta_extrapolation=1.93`, `power_ratio_decision='ok'`, oracle fidelity **0.858**, power **1.284x** the oracle's) -- 1.74x INSIDE the 2.0 bar.  Over all planes the guard can fire on, the smallest broken reading is 1.106 (Q z = 5400 um, fidelity 0.837) and the largest accepted is 0.9804: **the populations overlap and the gap is 1.13x**, against the report's claimed 4.69x.  The shipped bar therefore does not separate them; it only catches the far tail.  WP-B7c's open item 6 asked for an optic reading 1.25-2.0 with a HEALTHY completion (bar up) or 2.0-5.85 with a BROKEN one (bar down); what I found is the second, on three optics (F_alt 2.98, M 4.53/4.76, S 4.42, Q 2.04-5.31, P 2.25-4.55, all broken), AND readings of 1.11-1.83 that are ALSO broken -- so the bar should move down, but there is nowhere below it to move to: the accepted population runs to 0.98 | `validation/probe_verify_b7c/probe1_band.py F_alt out.json 1072 1076 1080` on both trees, then `summarise.py`; or `tests/unit/test_verify_b7c_multibranch.py::test_vb7c_an_accepted_field_reads_above_the_pinned_accepted_ceiling` |
| **D2** | **P2** | **The refusal is a property of the OUTPUT GRID, not of the field.**  Holding the optic, plane, launch lattice and window fixed and halving `dx` divides the reading by 4 (section 4.3), so the same physical plane is refused at one grid and accepted at the next: F_alt z = 1080 um reads 2.982 (REFUSED) at dx = 1.40 um, **1.420 (accepted)** at dx = 0.70 um and 1.036 at dx = 0.35 um -- one halving of the pixel is enough to remove the guard.  The bar's derivation used one grid per optic, so its margins are margins of a grid.  A caller who refines their grid to resolve the Airy layer -- which this very module's `l_airy` gate tells them to do -- crosses the bar in the direction that removes the guard | `tests/unit/test_verify_b7c_multibranch.py::test_vb7c_the_refusal_follows_the_pixel_not_the_field`; full ladder in `mech_*_head_win.json` |
| **D3** | **P3** | **The refusal message states a mechanism its own reading contradicts.**  At Q z = 5422 / 5460 um the text reads "with up to **1** branches on one pixel.  The output plane is at or near the AXIAL point focus, where a whole RING of branches coalesces..." -- the map is single-valued there and there is no ring (the refusal itself is correct: fidelity 0.637 / 0.292).  The narrative is unconditional, so a caller cannot use it to diagnose their own plane | `tests/unit/test_verify_b7c_multibranch.py::test_vb7c_the_refusal_fires_with_a_single_valued_ray_map`; message verbatim in section 2 of `log_bitid_win.txt` / the test's assertion |
| **D4** | **P3** | **The bar's denominator has more headroom than the bar.**  The decision reads `min(power_ratio, power_ratio_triangles)`; on the D3 geometry the two separate by up to **7.85x** (20 planes where `power_ratio` reads 2.02-3.76 while the bracket reads 0.45-0.90 and the completion returns with `'ok'` / `'energy_loss'`).  That bracket is correct and the fields there are not blown up -- but it means the guard's detection floor on such a geometry is `2.0 x spread`, i.e. up to ~15.7x, which is not stated anywhere.  The multibranch's own justification for the bracket ("costs no detection power, because a real blow-up is 1e5x") was written for a WARNING read in decades, not for a REFUSAL at 2.0 | `validation/probe_verify_b7c/probe4_bracket.py out.json`; `tests/unit/test_verify_b7c_multibranch.py::test_vb7c_the_gain_bracket_headroom_exceeds_the_refusal_bar` |
| **D5** | **P3** | **The shared fold oracle has an unstated NA ceiling.**  Its azimuthal integral is the Debye `J0` form; at f/1.2 (NA 0.45) the library's own `uniform` and `wave` members disagree with each other at fidelity 0.737 and the oracle sits between them, stable under 16x refinement.  Every optic in WP-B7b / VERIFY-B7b / WP-B7c is at NA 0.12-0.29, so no published number is affected -- but nothing in `caustic_fold_truth.py` records the limit, and the next study that reaches for a fast lens will read a fidelity of 0.6 and blame the propagator | `validation/probe_verify_b7c/probe5_oraclefloor.py out.json P:888 S:3214.78`; the `debye_J0_phase_error_rad` column |
| **D6** | **P4** | Three of WP-B7c's control statements are fixture-specific and read as general in the shipped comment of `_ENERGY_BLOWUP_FACTOR`: `min_area_ratio` 1e-8 vs 1e-6 is NOT identical on two of my four optics (197x and 2.05x apart), `ray_subsample` refinement INVERTS on the doublet, and `caustic_band` moves a healthy plane's reading by up to 55 % | `mech_Q_head_win.json`, `mech_S_head_win.json` |
| **D7** | **P4** | An unattributed numpy `RuntimeWarning: invalid value encountered in subtract` leaks from the lens-traced chain at the transition planes on P / M / S, on the base tree as well as the head, so it predates WP-B7c | `failbefore_base_win.json`, the `warnings` field |

Nothing outside my ownership was edited: `lumenairy/` is untouched on this
branch (`git diff 119b8bb6 -- lumenairy/` is empty).

---

## 11. Ship recommendation

**Ship, with D1 and D2 recorded as follow-ups and the report's section 3.1
numbers scoped.**

What is right and worth shipping as it stands:

* the mechanism is correctly diagnosed, and my pixel-refinement control makes
  the case more strongly than the report does;
* the refusal never removed a good field: **0 false positives in 71 scored
  planes on five optics and two builds**, and every one of the 18 refusals is
  a field at fidelity 0.18-0.64 while the hand-off at the same plane is at
  0.986-0.999.  This is a real improvement on a real defect (the base returns
  those fields silently but for another module's warning);
* the diagnostics, the byte-identity discipline and the `zeta_linear_range`
  report-don't-apply decision all verify cleanly;
* the member restatement and the `_AIRY_TAIL_CELLS` refusal verify cleanly.

What must not ship as a *claim*:

1. **the band's margins.**  "0.816 .. 1.246 accepted, >= 5.848 broken, nothing
   between, 1.60x / 2.92x in a 4.69x gap" is a property of 51 fold planes on
   five undercorrected singlets, not of the quantity.  Re-scope it to that
   population in the constant's comment, in the CHANGELOG and in
   `test_audit2609_b7c_multibranch_envelope.py`, and say what the guard is:
   **a far-tail tripwire with no measured lower margin, not a classifier.**
   Leaving 2.0 where it is is the right call -- it is the only value in the
   library for this quantity and lowering it toward the measured broken floor
   (1.106) would start refusing accepted fields at 0.98 -- but the two literal
   pins in the test file assert an envelope that does not exist and should be
   replaced by the scoped statement;
2. **`_ZETA_EXTRAPOLATION_MAX` as a sign boundary.**  The ABOVE-bar
   one-sidedness reproduces; "signed and centred below" does not (1 of 17
   negative, mean +7.00 %, largest excursion in the study below the bar).  The
   docstring and the constant's comment should keep the above-bar half and
   drop the below-bar half, or scope it to WP-B7c's three optics.

Neither of those is a code change; both are claims that a future reader would
otherwise rely on.  D2 (grid dependence) is the one that deserves a real
follow-up: until the quadrature itself is fixed (WP-B7c's own open item 1),
any bar on this quantity is a bar on a grid, and that belongs in the refusal's
own message so the caller knows refining is not a workaround.

---

## 12. What I could not verify

* **claim 5** (`kappa_eff/kappa` 1.068 -> 0.489 off the oracle's dark-side
  decay).  Outside the tasks I was given; I ran no independent `kappa_eff` fit,
  so the report's section 6.1 stands unexamined;
* **the member question itself.**  My oracle is the same one, so WP-B7c's own
  caveat -- that a hand-off-vs-oracle fidelity partly measures two propagators
  of the same geometrical boundary field agreeing -- applies to my 28-of-29
  exactly as it does to theirs.  Settling it needs an oracle whose boundary
  field is not geometrical;
* **whether F_alt z = 1076 um is "broken" in a sense a user would care about.**
  Its fidelity is 0.858 and its power 1.284x the oracle's; I classified it by
  the report's own 0.883 bar.  A different accept bar moves D1's numbers but
  not its direction -- the populations overlap at any bar between 0.86 and
  0.96;
* **the fast singlet at high NA.**  P (f/1.2) could not be scored, so the
  "fast singlet" arm of Task A rests on F at f/2.0.  Whether the band behaves
  differently above NA 0.3 is unmeasured, and the oracle available to this
  campaign cannot answer it;
* **anything about a build with a different LAPACK.**  Two builds agree to all
  printed digits on every number in this report, which bounds the cross-build
  spread of these quantities at ~1e-13 -- so none of the bars above is at risk
  from BLAS, and none of my verdicts depends on that spread.

---

## 13. Runs

| run | build | result | duration | log |
|---|---|---|---|---|
| `test_verify_b7c_multibranch.py` alone | win py3.14.6 | **6 passed** | 15.9 s | -- |
| `verify_b7c` + `b7c` + `b7_asymptotic` + `b7b_caustic_routing` + `niche_k4_uniform_caustic` + `niche_audit_w9_dispatch2` + `a3_caustic_siblings` + `a3_verify_traced` | win | **185 passed** | 130 s | `pytest_core_head_win.txt` |
| `verify_b7c` + `b7c` + `b7_asymptotic` + `b7b_caustic_routing` + `niche_k4_uniform_caustic` + `niche_audit_w9_dispatch2` | wsl py3.12.3 | **141 passed** | 180 s | `pytest_core_head_wsl.txt` |
| `-k census` (whole suite) | win | **45 passed**, 16195 deselected | 758 s | `pytest_census_win.txt` |
| `-k walker` (whole suite) | win | **118 passed, 6 skipped**, 16116 deselected | 61 s | `pytest_walker_win.txt` |
| `-k dispatcher_pin` (whole suite) | win | **510 passed, 5 skipped**, 15725 deselected, 7 warnings | 56 s | `pytest_dispatch_win.txt` |
| `test_public_api.py` + `test_v4_16_2_dispatcher_pin_doc_consistency.py` | win | **15 passed** (WP-B7c's environmental red does NOT reproduce) | 2.3 s | `pytest_publicapi_win.txt` |
| mutation matrix, 3 mutant trees x 6 decision tests | win | 1 / 2 / 2 killed, no collateral | 27 / 15 / 15 s | `mutations_win.txt` |
| `ruff check lumenairy/ tests/ validation/probe_verify_b7c/` | wsl | **All checks passed** | -- | -- |

No red anywhere, on either build.  `.test_durations` carries the six new ids,
spliced in sorted position with the measured values (6.31 / 2.95 / 2.80 /
2.03 / 0.96 / 0.05 s; 15.9 s for the file, inside the 60 s budget).

Commands (every one from `cd /c/tmp/lum_vmb`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, `--capture=sys`, and `PYTHONPATH` pinning the tree):

```
python -m pytest tests/unit/test_verify_b7c_multibranch.py tests/unit/test_audit2609_b7c_multibranch_envelope.py \
  tests/unit/test_audit2609_b7_asymptotic.py tests/unit/test_audit2609_b7b_caustic_routing.py \
  tests/unit/test_niche_k4_uniform_caustic.py tests/unit/test_niche_audit_w9_dispatch2.py \
  tests/unit/test_audit2609_a3_caustic_siblings.py tests/unit/test_audit2609_a3_verify_traced.py -q --capture=sys
python -m pytest tests/ -q --capture=sys -k census
python -m pytest tests/ -q --capture=sys -k walker
python -m pytest tests/ -q --capture=sys -k dispatcher_pin
python -m pytest tests/unit/test_public_api.py tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py -q --capture=sys
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmb && ~/lumvenv/bin/ruff check lumenairy/ tests/ validation/probe_verify_b7c/'
```
