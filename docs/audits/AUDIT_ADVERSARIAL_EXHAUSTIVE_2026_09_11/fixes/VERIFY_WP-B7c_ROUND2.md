# VERIFY-WP-B7c round 2 -- the arbiter verifies; the 7 % gap its bar sits in does not survive a ninth optic, and two plausible regressions of it leave the gate green

Independent adversarial verification of `fix/wp-b7c-round2` (`fcb9ed12`
library + tests + oracle, `e8ddec9d` report + CHANGELOG + probes), answering
`fixes/WP-B7c_ROUND2_REPORT.md`.  Branch `verify/wp-b7c-round2` off
`fix/wp-b7c-round2`; worktree `C:/tmp/lum_vmb2`, with two more used read-only:
`C:/tmp/lum_vmb2_ref` (`250421ed`, round 1) for bit identity and cost, and
`C:/tmp/lum_vmb2_mut` (detached `e8ddec9d`) for the mutation matrix.
`lumenairy/` on this branch is **untouched** (`git diff e8ddec9d --
lumenairy/` is empty).

| build | interpreter | numpy | scipy | tree |
|---|---|---|---|---|
| **win** | Windows py3.14.6 | 2.4.4 | 1.17.1 | `C:/tmp/lum_vmb2` |
| **wsl** | WSL py3.12.3 | 2.4.6 | 1.17.1 | `/mnt/c/tmp/lum_vmb2` |

Probes and JSON: `validation/probe_verify_b7c_round2/`.  New decision tests:
`tests/unit/test_verify_b7c_round2.py` (6 ids).

**The box was running unrelated jobs throughout this verification** (two other
pytest sessions and a long numerical script), so every wall-clock number below
is an upper bound and is labelled as such.  No claim rests on one.

---

## 0. Verdict table

| # | claim | verdict | my numbers |
|---|---|---|---|
| **1a** | the arbiter reads `p_out(dx)/p_out(dx/2)` on the RETURNED field, after completion | **CONFIRMED** | `_arbitrate(_uni_cont, 'the completed fold field')` is called on `E_out` after `_fill` and before the dtype cast; `pixel_continuity_of` reads `'the completed fold field'` on all 124 fold-ring planes I scored and `'the plain multibranch field'` on every fallback |
| **1b** | a converged quadrature reads **exactly 1** | **CONFIRMED to 5e-4; not exact** | on the slow control `C` (NA 0.034) the reading is **0.999529 .. 1.000207** on the eight planes more than 1.2 mm from its focus, spanning z = 12-30 mm.  It is not exact because the two renders' pixel CENTRES do not span the same window -- see **E5**.  Inside 1.2 mm of the focus, where geometrical optics itself fails, it reaches 1.0648 |
| **1c** | the bar 1.06 is the geometric centre of a **1.0683x** gap between a returned population (0.9860..1.0221, fid 0.9593..0.9985, n = 67) and a refused one (1.092..3.998, fid 0.0173..0.9302, n = 15) | **the builder's two extremes REPRODUCE; the GAP does not survive** | on `F_alt` I measure 1.0221/0.9747 and 1.0921/0.9306 -- the builder's own extremes to four decimals, on an oracle that shares no code with theirs.  But on **124 fold-ring planes over 7 optics** (5 of them mine) the gap is **1.02212 -> 1.06054 = 1.0376x**, margins **1.037x above / 1.00051x below**.  Over all **328** scored planes it is **1.0565 -> 1.06054 = 1.0038x** |
| **1d** | on YOUR optics, do the fidelity populations still not overlap? | **YES on the fold ring, NO overall** | fold ring: returned 0.9747..0.9954 against refused 0.0653..**0.9578** -- still disjoint, but the separation narrows from (0.9593, 0.9302) to (0.9747, 0.9578).  All planes: returned 0.4639..0.9976 against refused 0.0530..0.9578 -- **overlapping**, as the constant's own "WHAT IT DOES NOT CATCH" paragraph scopes |
| **1e** | does any of your planes land INSIDE the claimed gap (a 7 % margin, not decades)? | **YES, four** | `X` z = 992.30 (**1.06244**, REFUSED, fid 0.9578), `X` z = 992.90 (**1.06054**, REFUSED, fid 0.9564), `Z` z = 3442 (**1.0565**, returned, fid 0.9378), `Y` z = 2221.4 (1.05408, returned, fid 0.9500).  **Two are on the FOLD RING**, which is the population the gap is claimed on |
| **2** | reading the branch SUM instead would refuse good fields: 1.0636 vs 1.0020 at fidelity 0.9878 on VERIFY-B7b's fixture | **CONFIRMED exactly** | `V` z = 1761 um: branch sum **1.06360**, completion **1.00213**, my-oracle fidelity **0.9879**, power 0.988x.  Identical on both builds |
| **3** | bit identity 26 identical / 0 moved / 4 newly refused of 30 | **CONFIRMED in shape, on my own matrix** | 57 cases, child process per tree, `LUMENAIRY_MEM_BUDGET_MB=2048` pinned, `lumenairy.__file__` asserted under each tree: **44 identical, 0 MOVED, 6 newly refused, 0 newly returned, 7 refused on both** |
| **4a** | D1 (`F_alt` z = 1076, fid 0.858) is now refused | **CONFIRMED** | reads 1.18506 on both builds and raises; my oracle scores the field it would have returned at **0.8583** (builder: 0.8580) |
| **4b** | D2: the decision no longer follows the grid | **CONFIRMED** | `F_alt` z = 1080 reads a bracketed launched-power ratio of **2.982** (the builder's exact number) and a continuity of **2.0352** -- refused on both arms; the shipped test's refined-grid arm (N = 1280, dx = 0.70) passes on both builds |
| **4c** | the mechanism (the reading falls ~4x per halving) still holds | **CONFIRMED** | **29 planes across five optics** (`V`, `W`, `X`, `Y`, `Z`) read **3.9235 .. 4.0000** in the blow-up regime, against 0.956 .. 1.022 on healthy ones; 38 read above 3.5 |
| **5** | confusion: **0** false refusals at fid >= 0.95 on the fold ring, 3 at 0.883, **0/5** on all 104 planes | **REFUTED** | fold ring (124): **2** false refusals at fid >= 0.95 and **6** at 0.883.  All planes (328): **2** at 0.95 and **37** at 0.883, with **62** misses at 0.95 against the builder's 12 |
| **6a** | the Debye `J0` form is wrong by 54 % / fid 0.794 at f/1.2, so VERIFY-B7c's "members disagree at 0.737" was the oracle's gap | **CONFIRMED in direction, UNDERSTATED in size** | see section 8 and **E6** |
| **6b** | the rule `rel L2 ~ 0.37 eps`, "holding over three decades of `eps`" | **REFUTED** | on the builder's OWN fixture `Q` at its OWN plane (z = 5680 um, `y_max/z` 0.108, `eps` 0.0009) the rule predicts 0.00033 and the published row reads 0.0003.  **Their own `oracle_field(phi='exact')` gives 0.00461** there, and an exact azimuth applied at EVERY radius gives **0.01225** (quadrature) / **0.01446** (angular spectrum) -- 14x and 37x-44x.  The mechanism is in **E6** |
| **6c** | "no published number is affected ... under 1.5 % in relative L2, under 3e-4 in fidelity" over `y_max/z` 0.108-0.231 | **the L2 half REFUTED, the FIDELITY half CONFIRMED** | 1.2-2.7 % in relative L2 over `y_max/z` 0.102-0.209 (above the stated 1.5 %), but 7e-5 .. 3.6e-4 in fidelity -- so no published FIDELITY moves.  The published `power / oracle` columns do move: the `J0` oracle's own energy closure is 0.9972 at `y_max/z` 0.108 and **0.9741** at 0.209, not the 0.9992-0.99999 tabulated |
| **7** | cost 1.1x-3.0x of the completion, ONE extra rasterisation, never a second trace (assert structurally, and time it) | **structurally CONFIRMED; the wall-clock range is NOT RESOLVABLE on this box** | `_trace_launch_grid` and `_kmah_free_leg` are called **exactly once** with the arbiter on and with it off, on four optics; `pixel_halved_field` is `(2N, 2N)` with it on and `None` with it off, and never reaches the caller.  Wall clock (best of 7, round-1 tree against HEAD): uniform **1.23x .. 3.70x**, but the same-code control (the public branch sum, byte-identical) reads **0.88x .. 1.57x** on the same runs, so the measurement cannot separate 3.7 from 2.3 |
| **8** | do the five re-pointed tests and the single-seam pin FAIL under a plausible regression? | **the seam pin DOES; the bar and the reading's placement do NOT** | 13 mutations, all against the three shipped decision files: **11 caught, 2 NOT**.  The alias bypass (M1) is caught loudly -- 14 failed, the seam pin and all five re-pointed niche ids among them.  **M2 (read the branch sum) -> 28 passed.  M3 (bar 1.20) -> 26 passed, 2 skipped** |
| **9** | every new test's bar is derived, two-sided, premise-gated, and FAILS (not skips) on a silent regression | **REFUTED for two of the twelve** | under M3 the two tests that encode the claim `pytest.skip` on exactly the premise the regression removes, and the file reads green |
| **10** | the JAX twin, if `_lens_traced_multibranch` has one: same arbiter, same decision | **N/A** | there is no JAX twin of the branch sum or of the uniform completion.  `lumenairy/elements/_lens_jax.py` carries Maslov / GBD only; `apply_real_lens_traced_jax` has no `caustic='multibranch'` or `'uniform'` route |
| -- | the fallback route: how often is a bad field RETURNED silently, and what do the diagnostics say? | **measured** | of the 204 non-fold-ring planes the completion returns **62** fields below fidelity 0.95 and **11** below 0.883, down to **0.4639** -- at which the diagnostics read `pixel_continuity` 1.00201, `power_ratio` 1.005, `pixel_continuity_decision == 'ok'`, `power_ratio_decision == 'ok'`, and no warning from either arm |

---

## 1. Method, and the oracle

Nothing here reads a builder number and agrees with it.

**The optics are mine.**  Five prescriptions the campaign has never run,
chosen against the brief:

| id | prescription | lambda | grid | NA | why |
|---|---|---|---|---|---|
| `W` | N-BAK4 plano-convex run **PLANO-FIRST** (R1 = inf, R2 = -2.9 mm) | 780 nm | 512 x 2.40 um | 0.107 | inside the published `y_max/z` envelope (0.102-0.113); every shared-oracle fixture is convex-first or biconvex |
| `X` | N-SF10 biconvex R = +/-2.0 mm stopped to 1.15 mm | 1.064 um | 640 x 1.40 um | **0.3865** | **above the 0.33 the brief asks for**; measured `y_max/z = 0.461` at its fold planes -- higher than every fixture in either study, `P` (0.452) included |
| `Y` | N-BK7 biconvex with an **OBLATE** conic `k2 = +0.90` on surface 2 | 633 nm | 512 x 1.80 um | 0.197 | conic-surfaced, and on the opposite side and sign from the builder's `K` |
| `Z` | **CEMENTED doublet** N-BAK4 / N-F2, 3 surfaces | 532 nm | 512 x 1.60 um | 0.119 | a doublet with glasses the campaign has not used; its SA is so small (f_marginal 3488.5 um against f_paraxial 3525.9 um) that NO plane takes the completion route |
| `C` | N-BK7 plano-convex R1 = +12.0 mm, 1.60 mm aperture | 1.064 um | 512 x 4.00 um | 0.034 | the **CONVERGED CONTROL**: far from any caustic, so the reading must be 1 |

`V` and `F_alt` from the builder's archive are run as well, because claims 2
and 4 are stated ON those geometries and must be re-measured there.

**The oracle is mine** (`vroracle.py`), sharing no code with
`validation/oracles/caustic_fold_truth.py` or with either builder probe:

* Sellmeier coefficients for ten Schott glasses typed here.  The control
  against `lumenairy.glass.get_glass_index` over six wavelengths is
  **2.2e-16** at worst, so a typo would have shown as a control failure and
  not as a silent oracle bias;
* an exact sequential meridional **conic** ray trace (Newton intersection,
  vector Snell), written here;
* **three** propagators of the traced exit field -- `rs_j0` (the shared
  oracle's Debye `J0` ring integral, re-implemented term for term so the
  comparison is like for like), `rs_exact` (the same integral with the
  azimuth kept EXACTLY: midpoint rule on `(0, pi)`, node count derived per
  rho from that rho's own `b = k y rho / R0`, applied at **every** radius out
  to the grid corner), and `asm_field` (a band-limited **angular spectrum** --
  a different integral altogether, exact for the scalar Helmholtz equation
  given the boundary field, with no Debye, paraxial or azimuthal
  approximation anywhere).

Cross-validation (`oraclefloor_win.json`, `oraclefloor_builder_win.json`):

* `rs_exact` reproduces `rs_j0` at `rho = 0` to **1e-15**, where the two are
  identical by construction;
* doubling the exact arm's azimuthal safety factor moves it by **2e-13 ..
  5e-13**; doubling its ring count by 6e-4 .. 1e-3; doubling the ASM's
  refinement (4 -> 6) by 5e-4 .. 1e-3 and its ring count by 1.2e-4.  Both
  exact arms are converged at least a decade below the effect they measure;
* the ASM arm closes energy at **0.99849 .. 0.99977** of the launched power
  where the `J0` arm closes at 0.9741 .. 0.9972;
* the library's own `caustic='wave'` hand-off scores **0.9980** against my ASM
  oracle at `F_alt` z = 1063 and z = 1076 -- an independent propagator of the
  same boundary field, agreeing with my oracle at the two planes the whole
  argument turns on.

**The scan design differs from the builder's on purpose.**  `vscan.py` makes
ONE call per plane on ONE tree with both refusal bars patched to `inf` at run
time (`lumenairy/` is not edited), so the reading and the field it would have
refused come from the SAME call; the shipped decision is re-derived from the
constants imported before the patch.  The builder joined a head-tree reading
to a base-tree field across two processes, which by construction cannot see a
reading that disagrees with the field it was taken on.

**Both builds.**  The oracle never imports lumenairy, so what has to be shown
twice is the READING and the DECISION.  `vreadings.py` takes the 23 readings
every claim in this report rests on -- including both false refusals, the
claim-2 split and D1 -- on win and on wsl: **every reading identical to
4.1e-16 or exactly, every decision identical** (`readings_win.json` against
`readings_wsl.json`).

---

## 2. The populations, the gap and the bar

`joined_win.json`: **328 oracle-scored planes on 7 optics**, of which **124**
take the completion route (`reason == 'fold_ring'`, `fell_back == False`).

### 2.1 Fold ring (124 planes)

| population | n | continuity | oracle fidelity |
|---|---|---|---|
| RETURNED | 89 | 0.95580 .. **1.02212** | **0.9747** .. 0.9954 |
| REFUSED | 35 | **1.06054** .. 3.99877 | 0.0653 .. **0.9578** |

Per optic:

| optic | returned C / fidelity | refused C / fidelity |
|---|---|---|
| `V` (builder) | 0.9868..1.0213 / 0.9809..0.9917 | 3.8770..3.9981 / 0.1675..0.2589 |
| `F_alt` (builder) | 0.9926..**1.0221** / **0.9747**..0.9901 | 1.0921..2.5059 / 0.4884..0.9306 |
| `W` (plano-first) | 0.9746..1.0131 / 0.9870..0.9928 | 3.9988 / 0.0780 |
| `X` (**NA 0.3865**) | 0.9558..1.0046 / 0.9912..0.9954 | **1.0605**..3.9808 / 0.0653..**0.9578** |
| `Y` (oblate conic) | 0.9919..1.0081 / 0.9937..0.9946 | 3.9920 / 0.1009 |
| `Z` (cemented) | -- | -- |

`Z` produces no fold-ring plane at all: on this doublet every fold plane falls
back on `fold Airy scale under-resolved by the grid`.  That is itself worth
recording -- the completion route the bar is derived on is not reached by a
well-corrected doublet at a sane grid.

**The criterion-free claim survives.**  The two FIDELITY populations still do
not overlap on the fold ring, with no accept bar chosen (0.9578 < 0.9747).
That is the report's strongest statement and it reproduces on five optics it
was not derived on.

**The gap does not.**  `1.02212 -> 1.06054` is **1.0376x**, and the bar sits
**1.00051x** below the smallest refused reading -- 0.05 %, where the report
claims 3.0 %.  Over all 328 planes the gap is **1.0038x**.  Section 10 of the
round-2 report names exactly this risk ("whether the 7 % gap survives a ninth
optic"); on a ninth and a tenth, it does not.

### 2.2 Confusion

| population | accept bar | false refusals | misses | refused-broken | returned-accepted |
|---|---|---|---|---|---|
| fold ring (124) | fid >= 0.883 | **6** | 0 | 29 | 89 |
| fold ring (124) | fid >= 0.95 | **2** | 0 | 33 | 89 |
| all planes (328) | fid >= 0.883 | **37** | 11 | 97 | 183 |
| all planes (328) | fid >= 0.95 | **2** | **62** | 132 | 132 |

The two false refusals at the 0.95 bar, both on the FOLD RING, both on `X`:

| plane | C | fidelity | power / oracle |
|---|---|---|---|
| `X` z = 992.30 um | **1.06244** | **0.9578** | 1.1046 |
| `X` z = 992.90 um | **1.06054** | **0.9564** | 1.1022 |

Both carry a ~10 % ENERGY error, so a stricter accept criterion would call
them correct refusals -- exactly the ambiguity the report acknowledges for its
own three.  What does not depend on the criterion is that they are the
smallest refused readings in the study and sit **1.0005x above the bar**,
inside the gap the bar was centred in.

### 2.3 The reading is not smooth in z near a fold onset

On `X` at 0.01 um steps (`fine_X2_win.json`):

```
z [um]    992.22   992.23   992.24   992.25 .. 992.29   992.30   992.31
C         1.0002   1.1810   0.9989   0.9991 .. 1.0046   1.0624   0.9988
fidelity  0.9943   0.8949   0.9949   0.9952 .. 0.9912   0.9578   0.9953
```

The reading spikes to 1.181 and returns, then to 1.062 and returns, inside
80 nm of defocus -- an 18 % excursion and a 6 % one, both isolated to a single
10 nm rung.  The
FIELD moves with it (0.9943 -> 0.8949 -> 0.9949), so the arbiter is tracking
something real -- the collapsing ring crossing pixel centres, which is the
mechanism it is built to detect.  But it means the bar's margin is smaller
than the reading's own plane-to-plane excursion in this regime, and a
7 %-wide band cannot be a stable classifier there.

### 2.4 Is the band a property of the quantity, or of the fixture set?

Every published reading is taken at the default `ray_subsample=2` on one grid
per optic, and the estimator argument behind "a converged render reads 1" is
that the mapped triangles are spread over many pixels -- a joint property of
the LAUNCH lattice and the OUTPUT pitch, neither of which the derivation
varies.  `vsweep.py` varies both at three healthy fold planes of `W`
(`sweep_W_win.json`):

| axis | range | reading | fidelity |
|---|---|---|---|
| `ray_subsample` 1, 2, 3, 4, 6, 8 at N = 512 | z = 4870 | 0.9783 .. 1.0106 | 0.9894 .. 0.9915 |
| the same at z = 4900 | | 1.0036 .. 1.0129 | 0.9926 .. 0.9931 |
| the same at z = 4930 | | 1.0049 .. **1.7402** | **0.7399** .. 0.9896 |
| `dx` x0.5, x0.75, x1 (fold ring) | z = 4870-4930 | 0.9934 .. 1.0091 | 0.9870 .. 0.9922 |
| `dx` x1.5, x2 (the completion falls back) | z = 4870-4930 | 0.9739 .. **1.1114** | **0.7947** .. 0.9176 |

**The arbiter passes this test.**  Across six launch lattices the reading
stays inside the band wherever the field is right, and the one rung where the
field collapses (`ray_subsample=6` at z = 4930, fidelity 0.7399) is the one
rung the reading refuses (1.7402).  Coarsening `dx` past the point where the
completion falls back moves both together in the same direction.

It also adds one data point to **E7**: at `dx` x1.5 the reading is 1.0113 and
the returned field scores 0.8520 -- accepted, and wrong, because
under-resolving the Airy layer is a different failure mode from the one this
arm detects.

---

## 3. Defects

| id | severity | statement | reproducer |
|---|---|---|---|
| **E1** | **P2** | **The bar's margins are a property of the eight optics it was derived on.**  On a ninth (`X`, NA 0.3865) and a tenth (`Z`) the fold-ring gap falls from 1.0683x to **1.0376x** and the all-planes gap to **1.0038x**; the bar's margin BELOW the smallest refused reading falls from 1.030x to **1.00051x**, and two fold-ring planes are refused at fidelity 0.9578 / 0.9564.  The reading is also not smooth in `z` near a fold onset (section 2.3), so the 7 % band is narrower than the quantity's own excursion there.  The comment, the CHANGELOG and the report all state the 1.0683x gap and its two margins as measured properties of the quantity | `python vscan.py X "992.22:992.40:19" out.json` and `"940:990:6,991:999:9"`; `fine_X2_win.json`, `fine_X_win.json`, `joined_win.json` |
| **E2** | **P2** | **Nothing in the gate holds `_PIXEL_CONTINUITY_MAX`.**  Setting it to **1.20** -- which RETURNS D1's plane, the R-5 counterexample the round exists to close, at oracle fidelity 0.858 -- leaves the three shipped decision files at **`26 passed, 2 skipped`**.  Every constant assertion is a shape (`1.0 < MAX < _ENERGY_BLOWUP_FACTOR`, `MIN == 1/MAX`), and the one behavioural test that would have seen it -- `..._a_plane_inside_the_power_ratio_bar_is_refused_on_continuity`, which IS D1 -- `pytest.skip`s on precisely the premise the regression removes (`'this build returns the pinned plane; its continuity reads 1.1851, inside the band'`), after an "invariant arm" that a loosened bar satisfies trivially.  The loss-arm test skips too, for the mirrored reason (`MIN` moves to 1/1.20 = 0.833, below the ladder's 0.854).  That is the fail-open shape `docs/TESTING_STANDARDS.md` rule 4 forbids | `python vmutate.py out.json M3_bar_1_20`; `mut_M3_bar_1_20_win.txt` |
| **E3** | **P2** | **Nothing in the gate holds WHERE the reading is taken.**  Replacing `_arbitrate(_uni_cont, ...)` by `_arbitrate(_mb_cont, ...)` -- deciding on the branch sum, which by the module's own measurement would refuse `V` z = 1761 at oracle fidelity 0.988 -- leaves the three shipped files at **`28 passed`**.  This is the one design decision round 2 argues for at length, and the argument for it is right (claim 2 reproduces exactly); it is simply not held | `python vmutate.py out.json M2_reads_branch_sum`; `mut_M2_reads_branch_sum_win.txt` |
| **E4** | **P3** | **The Pearcey cusp route labels the branch sum's reading as the completed field's.**  `_arbitrate(_mb_cont, 'the Pearcey cusp field')` records `pixel_continuity_of = 'the Pearcey cusp field'` while the number is the branch sum's.  The code comment beside it says why (there is no half-pitch Pearcey without a second cusp trace) -- but by the module's own claim-2 argument that is the reading that "would refuse fields that are right", so this route carries the very exposure the design decision removes, under a label that says it does not | `lumenairy/elements/_lens_traced_uniform.py` line 1492 |
| **E5** | **P3** | **The two renders do not span the same window, so a converged render does not read 1 exactly.**  `_render(N, dx)` puts pixel centres on `(j - N/2) dx`, `_render(2N, dx/2)` on `(m - N) dx/2`: the fine grid's centres extend half a coarse pixel further out on the `+x` and `+y` edges, so it catches deposits the coarse one cannot whenever light reaches the far edge.  Measured floor on the slow control: **0.999529 .. 1.000207** over eight planes and 18 mm of propagation.  60x smaller than the bar's margin, so it changes no decision measured here -- but "reads **1 exactly**, on any optic, at any plane, at any grid" is not what the code computes, and the claim is load-bearing (it is what makes the bar "a tolerance on a known value") | `band_C_win.json`; `_lens_traced_multibranch._render` |
| **E6** | **P3** | **The oracle-floor table's "Debye vs exact" column is measured only where the truncation is smallest.**  `probe_wp_b7c_round2/oracle.py` substitutes the exact azimuth ONLY inside `r_core` (the 99.95 %-energy radius) and keeps the `J0` form outside, then reports the relative L2 of the whole 2-D field -- which is identically zero outside `r_core` by construction.  But the dropped quadratic term grows with `rho`: on `Q` at z = 5680 um it is 0.0009 rad at the rms radius and several radians at the grid corner.  Three measurements at that plane: the published row reads **0.0003**; running the builder's OWN `oracle_field(phi='exact')` gives **0.00461** (i.e. the published value does not reproduce from their own code at their own defaults); and an exact azimuth applied at EVERY radius gives **0.01225**, with an independent angular spectrum at **0.01446**.  The fitted rule `rel L2 ~ 0.37 eps` is fitted to the core-confined number.  The headline conclusion is unaffected and in fact strengthened -- the `J0` form's NA ceiling is WORSE than reported -- but the envelope "under 1.5 % in relative L2" is not right.  Separately, the `J0` field's own energy closure against the launched power is 0.9986 (their code) / 0.9972 (mine) at `y_max/z` 0.108 and **0.9741** at 0.209, where the table's `J0 closure` column reads 0.99999 and 0.99921; I could not reconcile those definitions, so the studies' `power / oracle` columns should be re-checked against an energy-conserving arm.  **CORRECTED 2026-09-19 (WP-B7c round 3, R3-5): the published closure column is right and THIS one is not.**  Measured from a well-resolved radial profile the `J0` arm closes at **0.99992** at `y_max/z` 0.108 and **0.99916** at 0.209; the 0.9972 / 0.9741 here are this verification's own radial RECONSTRUCTION losing energy, which it does on both arms equally (a 256-radius profile closes at 0.894 on the `J0` and the exact arm alike).  The full-radius Debye-vs-exact finding is unaffected and is strengthened -- round 3 measures **0.0563** at the doublet row against 0.01225 here and 0.0003 published, pointwise at the output grid's own pixels with no reconstruction on either side | `python voracle_check.py out.json B:Q 5680 W 4900 Y 1980`; `log_builder_oracle_cross_win.txt`; `oraclefloor_win.json`, `oraclefloor_builder_win.json` |
| **E7** | **P4** | **On the fallback route a wrong field is returned silently with both arms at their nominal values.**  `C` z = 22800 um returns a field at oracle fidelity **0.4639** with `pixel_continuity` 1.00201, `multibranch_power_ratio_bracketed` 1.005, both decisions `'ok'` and no warning.  The constant's comment scopes this ("a plane it accepts is not thereby certified") and quotes "returned down to 0.764"; over 328 planes the floor is **0.4639**, and **62** returned planes are below fidelity 0.95 | `vreadings.py` case `C_22800`; `band_C_win.json` |

Requested edits, in the order I would apply them:

1. **E2 / E3** -- add the two decision tests this verification ships
   (`test_vb7c2_the_bar_is_derived_against_the_field_on_both_sides` and
   `test_vb7c2_the_reading_is_of_the_returned_field_not_of_the_branch_sum`),
   or equivalents.  They check the bar against the FIELD rather than against
   a remembered number, and they fail rather than skip when the guard stops
   refusing.
2. **E1** -- in `_lens_traced_multibranch._PIXEL_CONTINUITY_MAX` and in the
   mirror block of `_lens_traced_uniform._MB_PIXEL_CONTINUITY_MAX`, replace
   "Margins: 1.037x above the largest returned reading and 1.030x below the
   smallest refused one ... this is a 7 % gap" with:

   ```
   # RE-MEASURED (VERIFY round 2, 2026-09-15) on 124 fold-ring planes over
   # seven optics, five new -- an N-SF10 biconvex at NA 0.3865 (y_max/z
   # 0.461), an oblate-conic N-BK7, a plano-first N-BAK4, a cemented
   # N-BAK4 / N-F2 doublet and a slow N-BK7 control:
   #   * RETURNED: 0.9558 .. 1.0221, fidelity 0.9747 .. 0.9954  (n = 89)
   #   * REFUSED:  1.0605 .. 3.9988, fidelity 0.0653 .. 0.9578  (n = 35)
   # The two FIDELITY populations still do not overlap.  The gap in the
   # READING is 1.0376x, not 1.0683x: the margin ABOVE the accepted
   # population is 1.037x and the margin BELOW the refused one is 1.0005x,
   # i.e. none.  Two fold-ring planes on the fast singlet are refused at
   # fidelity 0.958 / 0.956 carrying a 10 % energy error, and near a fold
   # onset the reading is not smooth in z (1.0002 -> 1.1810 -> 0.9989 over
   # 20 nm of defocus), so this bar separates the FIELDS but does not have a
   # margin in the reading on its refusal side.
   ```

   and change the CHANGELOG's and the report's "a 7 % gap" accordingly.
3. **E6** -- in `probe_wp_b7c_round2/oracle.py`, either extend the exact arm
   to the grid corner (it is affordable at a reduced ring count -- mine runs
   1500 rings x 320 radii in 40-90 s per plane) or relabel the published
   column `core_relL2_vs_debye`, which is what it measures.  In
   `validation/oracles/caustic_fold_truth.py`, drop "holding over three
   decades of `eps`" and restate the envelope in FIDELITY (the quantity the
   studies read): under 3.6e-4 for `y_max/z <= 0.21`, measured by two
   independent exact propagators -- and record that the `J0` form's own
   energy closure is 0.997 at `y_max/z` 0.11 and 0.974 at 0.21, so
   `power / oracle` columns carry that bias.
4. **E4** -- either build the half-pitch Pearcey field, or label it honestly:
   `_arbitrate(_mb_cont, 'the branch sum the Pearcey cusp field is built on')`.
5. **E5** -- one sentence in `_render`'s docstring and in
   `_PIXEL_CONTINUITY_MAX`'s "WHAT IT MEASURES" block: the fine grid's pixel
   centres extend half a coarse pixel further out on two edges, so a
   converged render reads 1 to O(1/N) -- measured 5e-4 at N = 512 -- rather
   than exactly.
6. **E7** -- no code change requested; "returned down to 0.764" in the
   constant's "WHAT IT DOES NOT CATCH" paragraph should read 0.464 over 328
   planes.

---

## 4. The mutation matrix (claims 8 and 9)

Thirteen plausible regressions, each applied to `C:/tmp/lum_vmb2_mut`
(detached `e8ddec9d`) and run against the three shipped decision files
(`test_audit2609_b7c2_pixel_halving_arbiter.py`,
`test_verify_b7c_multibranch.py`,
`test_audit2609_b7c_multibranch_envelope.py`; the seam mutation additionally
against `test_niche_r2_pearcey_cusp.py` and
`test_niche_r5_gbd_vector_catastrophe.py`), with `PYTHONPATH` pinning the
mutant tree, `--capture=sys` and `-p no:randomly`.
`mutation_win.json`, `mutation2_win.json`, `mutation3_win.json`,
`mut_*_win.txt`.

| id | regression | caught? | tail |
|---|---|---|---|
| **M1** | an ALIAS that bypasses `_multibranch_render` (the completion reaches the public entry point again) | **yes** | 14 failed, 39 passed -- the seam pin and all five re-pointed niche ids among them |
| **M2** | the arbiter reads the BRANCH SUM instead of the returned field | **NO** | **28 passed** |
| **M3** | the bar is loosened to **1.20** | **NO** | **26 passed, 2 skipped** |
| **M4** | the second render is taken at `(N, dx)` instead of `(2N, dx/2)` | yes | 4 failed, 22 passed, 2 skipped |
| **M5** | the LOSS arm is made to refuse too | yes | 1 failed, 27 passed |
| **M6** | the band stops being symmetric in the log (`MIN = 0.5`) | yes | 2 failed, 26 passed |
| **M7** | entry cap 1e6 (excludes every published fixture) | yes | 6 failed, 20 passed, 2 skipped |
| **M8** | the completion stops asking for the reading | yes | 9 failed, 19 passed |
| **M9** | the ratio is inverted | yes | 5 failed, 21 passed, 2 skipped |
| **M10** | the FALLBACK path stops arbitrating | yes | 2 failed, 26 passed |
| **M11** | an unmeasurable reading silently becomes `'ok'` | yes | 1 failed, 27 passed |
| **M12** | the launched-power tripwire is disabled | yes | 1 failed, 20 passed, 7 skipped |
| **M13** | entry cap **2e6** -- keeps every published fixture, excludes only D2's refined grid | yes | 1 failed (`test_vb7c_the_refusal_follows_the_pixel_not_the_field`), 26 passed, 1 skipped |

**The seam pin does fail under its own regression.**  M1 is the one the
report names, it is caught loudly, and the failure mode is the one the report
describes: with one name reachable, patching the wrong one raises.  What the
seam was built to carry is not held: the BAR and the SIDE of the seam the
reading is taken on.

M3's two skips are the durability finding.  `26 passed, 2 skipped` is
indistinguishable from green in CI, and the first of the two skipped ids is
the one that encodes D1 -- the counterexample the whole round exists to close
(`mut_M3_bar_1_20_win.txt`, verbatim):

```
SKIPPED tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py:435:
    this build returns the pinned plane; its continuity reads 1.1851,
    inside the band
SKIPPED tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py:545:
    no plane of this ladder reads below the loss arm on this build
```

On HEAD the same three files report **28 passed, 0 skipped** with `-rs`
(`pytest_skips_head_win.txt`), so the fail-open shape is latent: it is
invisible on a healthy build and appears only on the build that has the
defect.

The first reaches the skip through an "invariant arm" that a loosened bar
satisfies trivially: `ud['pixel_continuity'] <= _MB_PIXEL_CONTINUITY_MAX` is
true by construction once the call returned at all.  The second is the loss
arm going quiet for the mirrored reason (`MIN` moves with `MAX` to
1/1.20 = 0.833, below the 0.854 the ladder reads).  `..._the_decision_survives_halving_the_output_pixel` still passes: its two
readings (2.04 and 1.35) are above 1.20 as well, so the grid claim survives a
bar this size -- which is exactly why the bar needs a test of its own.

**The two gaps are closed by the tests in section 5**, proved the same way --
the same mutations re-run with `tests/unit/test_verify_b7c_round2.py` added
to the same three files:

| id | with the shipped three | with my file added | the id that fires |
|---|---|---|---|
| **M2** read the branch sum | 28 passed | **1 failed, 33 passed** | `test_vb7c2_the_reading_is_of_the_returned_field_not_of_the_branch_sum` |
| **M3** bar 1.20 | 26 passed, 2 skipped | **1 failed, 30 passed, 3 skipped** | `test_vb7c2_the_bar_is_derived_against_the_field_on_both_sides` |
| **M4** reading at `(N, dx)` | 4 failed | **6 failed** | both of the above, in addition |
| **M13** entry cap 2e6 | 1 failed | **2 failed** | `test_vb7c2_the_entry_cap_survives_the_refinement_d2_is_about`, in addition |

---

## 5. New decision tests

`tests/unit/test_verify_b7c_round2.py`, **6 passed in 24.5 s** on win and, run
with the three shipped files, **34 passed in 120.7 s** on win / **102.1 s** on
wsl.  The slowest id is 18.4 s (win) / 27.1 s (wsl), inside the 60 s budget:

| id | closes | shape |
|---|---|---|
| `test_vb7c2_the_bar_is_derived_against_the_field_on_both_sides` | **E2 / M3** | two-sided, derived on the running build: on `F_alt`'s three-plane ladder every plane whose FIELD is good (fidelity >= 0.97 against the `caustic='wave'` hand-off) reads BELOW the bar and every plane whose field is broken (<= 0.94) reads ABOVE it.  When no broken plane exists it **fails** with an instruction to extend the ladder, rather than skipping |
| `test_vb7c2_the_reading_is_of_the_returned_field_not_of_the_branch_sum` | **E3 / M2** | unconditional arm: the recorded reading is not the branch sum's and `pixel_continuity_of` names the completed field.  Premise-gated arm: on this build the two straddle the bar, so the substitution is a DECISION difference, and the returned field scores > 0.95 against the hand-off |
| `test_vb7c2_the_entry_cap_survives_the_refinement_d2_is_about` | M13, belt and braces | derived: `4 N^2 <= _ARBITER_MAX_FINE_ENTRIES` for N = 768 (the campaign's largest) and N = 1280 (D2's refined grid), with a behavioural arm that the reading IS taken at N = 640.  M13 is already caught by a round-1 test, but only through a ladder that can itself skip; this arm cannot, and under M13 the two fail together |
| `test_vb7c2_an_unmeasurable_reading_is_never_reported_as_ok` | M11, premise ENGINEERED | the cap is lowered on the running build so the plane under test is past it, so the claim holds whatever the shipped constant is |
| `test_vb7c2_a_fallback_plane_is_arbitrated_on_the_field_it_returns` | M10 | on a fallback the returned field IS the branch sum, so the two readings must be the same number to the bit and `pixel_continuity_of` must say so |
| `test_vb7c2_the_bar_is_one_constant_and_the_band_is_its_reciprocal` | M6 | unconditional, build-free |

No test here pins a number this verification measured.  The accuracy axis is
the `caustic='wave'` hand-off, measured on the running build at the same
plane (and independently scored at 0.998 against my oracle); the basins
0.97 / 0.94 sit inside the spacing the ladder itself exhibits (0.9747 against
0.9306).

---

## 6. Bit identity (claim 3)

`vbitid.py`: **57 cases** -- for each of `W`, `X`, `Y`, `Z`, `C` a healthy
fold plane, a blown-up plane, a fallback, a gap-edge plane, the branch sum
alone, the exit vertex (`output_plane_distance = 0`), `caustic_band='plain'`,
`ray_subsample=4`, a complex64 input and the alternate grid; plus the
builder's `V` at 1758 / 1761 / 1768 and `F_alt` at 1063 / 1073 / 1076 / 1080.
A CHILD process per tree with `cwd` and `PYTHONPATH` set to it,
`lumenairy.__file__` ASSERTED under it, **`LUMENAIRY_MEM_BUDGET_MB=2048`
pinned** (VERIFY-WP-B12 D-4: unpinned digests are chunking-dependent),
SHA-256 over `ndarray.tobytes()`.

`250421ed` against `e8ddec9d`: **44 identical, 0 MOVED, 6 newly refused, 0
newly returned, 7 refused on both.**

| newly refused | round-2 reading | oracle fidelity of the round-1 field |
|---|---|---|
| `W_uni_fb` (z = 5210, fallback) | 1.0760 | 0.9061 |
| `Y_uni_fb` (z = 2220, fallback) | 1.0708 | 0.9429 |
| `Z_uni_fb` (z = 3550, fallback) | 1.0716 | 0.8875 |
| `Z_uni_edge` (z = 3510) | **3.0268** | **0.6470** |
| `C_uni_bad` (z = 23200) | 1.0648 | 0.4138 |
| `BF_uni_1076` -- **D1's plane** | 1.1851 | 0.8583 |

`Z_uni_edge` is the strongest independent evidence FOR the fix in this
report.  It is a **single-valued map** (`n_branch_max == 1`) whose
launched-power bracket reads **1.021** -- as healthy as a reading gets, and
1.96x inside the 2.0 bar -- returning a field at oracle fidelity **0.647**.
Only the continuity arm sees it.  Round 1 returned it silently.

---

## 7. Cost (claim 7)

**Structural, on HEAD, four optics** (`cost_win.json`).  With the arbiter ON
and with it OFF, `_lens_traced_multibranch._trace_launch_grid` is called
**exactly once** and `_kmah_free_leg` **exactly once**; the completion's
`_trace_meridional_fold` is called once on the completion route and zero
times from the branch sum.  With the arbiter on, `pixel_halved_field` is
present at shape `(2N, 2N)`; with it off it is `None` and the decision is
`'not_requested'`; and it is `pop`-ed before the caller's diagnostics dict is
built (`'pixel_halved_field' not in ud` on every route).  **One extra
rasterisation, no second trace, no second KMAH pass, no second fold trace --
confirmed.**

**Wall clock**, best of 7, round-1 tree against HEAD, same box, `cost2_win.json`:

| fixture | plane | round-1 `uniform` | round-2 `uniform` | ratio | `multibranch` ratio (must be 1) |
|---|---|---|---|---|---|
| `W` | z = 4870 (fold ring) | 0.410 s | 0.505 s | **1.23x** | 0.99x |
| `W` | z = 0 (exit vertex) | 0.472 s | 1.743 s | **3.70x** | 1.57x |
| `X` | z = 960 (N = 640) | 2.191 s | 3.476 s | 1.59x | 1.35x |
| `Y` | z = 1970 | 0.444 s | 0.974 s | 2.20x | 0.88x |
| `Z` | z = 3430 (fallback) | 0.632 s | 1.298 s | 2.05x | 1.18x |
| `C` | z = 16000 | 0.367 s | 0.765 s | 2.08x | 1.27x |

The `multibranch` column is the noise control: that entry point is
byte-identical between the two trees and must read 1.00x.  It reads
**0.88x .. 1.57x**, so this box cannot resolve the uniform ratio better than
about +/-50 %.  The claimed "1.1x-3.0x, typically ~2x" is **consistent with**
these numbers and **not confirmed by** them; the exit-vertex case reads 3.70x
here against the report's 2.29x, and the difference is inside the noise.

---

## 8. The oracle's NA ceiling (claim 6)

`voracle_check.py` measures the `J0` form against TWO independent exact
propagators of the same exit field, each with its own convergence control,
and -- unlike the builder's probe -- applies the exact azimuth at **every**
radius out to the grid corner.

| optic | plane | `y_max/z` | `eps` (rms) | `J0` vs exact azimuth | fid | `J0` vs ASM | fid | `J0` energy / `P_in` | ASM energy / `P_in` |
|---|---|---|---|---|---|---|---|---|---|
| `W` (mine) | 5220 um | 0.102 | 0.0012 | 0.01650 | 0.99986 | 0.01889 | 0.99983 | 0.9930 | 0.99965 |
| `Q` (builder's doublet) | 5680 um | 0.108 | 0.0009 | **0.01225** | 0.99993 | 0.01446 | 0.99990 | 0.9972 | 0.99977 |
| `W` (mine) | 4900 um | 0.109 | 0.0015 | 0.01844 | 0.99983 | 0.01975 | 0.99981 | 0.9924 | 0.99966 |
| `S` (builder's plano-cx) | 3214.78 um | 0.133 | 0.0014 | 0.01337 | 0.99991 | 0.01583 | 0.99988 | 0.9922 | 0.99953 |
| `Y` (mine) | 1980 um | 0.209 | 0.0264 | **0.02738** | 0.99964 | 0.03451 | 0.99952 | **0.9741** | 0.99849 |
| **`X` (mine, NA 0.3865)** | 960 um | **0.461** | 0.7158 | **0.2849** | **0.97067** | **0.3678** | **0.95577** | 0.9732 | 0.99924 |

The high-NA row is the one that matters for the brief's "an oracle that holds
above NA 0.33": at `y_max/z = 0.461` the `J0` form costs **0.029 .. 0.044 in
fidelity**, an order of magnitude more than the fidelity differences this
study reads (0.9578 against 0.9953).  Scoring `X` against it would have been
meaningless.  The ASM arm used for that scoring is converged there to
**5.1e-4** (refinement 4 -> 6) and **3.7e-4** (ring count doubled), three
decades below the differences read.

**How `X` is scored.**  Every fidelity in this report -- `X` at NA 0.3865 and
`y_max/z = 0.461` included -- is taken against the ANGULAR-SPECTRUM arm, which
makes no azimuthal approximation at any NA, so "an NA above 0.33 with an
oracle that holds there" is met by construction rather than by an envelope
argument.  The `J0` arm appears in this report only as the quantity under
test.

Convergence of both exact arms on every row: doubling the azimuthal safety
factor moves the exact quadrature by **2e-13 .. 5e-13**, doubling its ring
count by 6e-4 .. 1e-3; doubling the ASM's refinement by 5e-4 .. 1e-3.  So the
`J0`-vs-exact column is the `J0` truncation and nothing else, and the two
independent methods bracket it consistently.

Three findings:

* **the direction is right and the ceiling is real.**  D5's suspicion and the
  round-2 report's measurement both hold: the `J0` form has a genuine NA
  ceiling, and "the members were closer to the truth than the thing scoring
  them" is the right reading of VERIFY-B7c's 0.737;
* **the published table under-reports it**, for the reason in **E6**: the
  builder's exact arm is confined to the 99.95 %-energy core, outside which
  the two fields are identical by construction, while the dropped quadratic
  term grows with `rho`.  At `Q` z = 5680 um the published row reads 0.0003,
  the builder's own `oracle_field(phi='exact')` gives 0.00461 at its
  defaults, and an exact azimuth at every radius gives 0.01225 with an
  independent angular spectrum at 0.01446.  The three differ by 14x-44x and
  the two full-radius methods agree with each other;
* **the conclusion that matters survives.**  The FIDELITY cost of the `J0`
  form is 7e-5 .. 3.6e-4 over `y_max/z` 0.10-0.21, below every fidelity
  difference either study reads, so **no published fidelity moves**.  What
  does move is the `power / oracle` column: the `J0` oracle loses 0.3 % of
  the energy at `y_max/z` 0.108 and **2.6 %** at 0.209, where its tabulated
  closure reads 0.99999 and 0.99921.

---

## 9. Runs

All from `cd /c/tmp/lum_vmb2`, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line, `--capture=sys`, `-p no:randomly`, and `PYTHONPATH` pinning the tree.
**The box was running two other pytest sessions and a long numerical script
throughout**, so every duration is an upper bound.

| run | build | result | duration | log |
|---|---|---|---|---|
| `test_verify_b7c_round2.py` alone | win py3.14.6 | **6 passed** | 24.5 s | `pytest_new3_win.txt` |
| `verify_b7c_round2` + `b7c2` + `verify_b7c` + `b7c_envelope` | win | **34 passed** | 120.7 s | `pytest_core_head_win.txt` |
| the same four | **wsl py3.12.3** | **34 passed** | 102.1 s | `pytest_core_head_wsl.txt` |
| the three SHIPPED files alone, before my file existed | win | **28 passed** | 115.7 s | `pytest_base_pinned_win.txt` |
| `a16` bit-identity + round-trip + verify, `niche_k1` / `r2` / `r5`, `v5_21` delta audit + lens accuracy (the builder's own scope) | win | **345 passed**, 2 deselected, 51 warnings | 1827.6 s | `pytest_a16_niche_narrow_win.txt` |
| `test_audit2609_a16*` + **all 117** `test_*niche*` + `test_v5_21*` (the brief's wider glob) | win | INCOMPLETE -- see below | -- | `pytest_a16_niche_win.txt` |
| `-k census` (whole suite) | win | **45 passed**, 16213 deselected | 770.9 s | `pytest_census_win.txt` |
| `-k walker` (whole suite) | win | **118 passed, 6 skipped**, 16134 deselected | 177.8 s | `pytest_walker_win.txt` |
| `-k dispatcher_pin` (whole suite) | win | **510 passed, 5 skipped**, 15743 deselected, 7 warnings | 267.3 s | `pytest_dispatch_win.txt` |
| `test_public_api.py` + `test_v4_16_2_dispatcher_pin_doc_consistency.py` + `test_audit_except_budget.py` | win | **19 passed** | 3.7 s | `pytest_publicapi_win.txt` |
| the three SHIPPED files with `-rs`, on HEAD | win | **28 passed, 0 skipped** | 63.1 s | `pytest_skips_head_win.txt` |
| the 13-mutation matrix (13 x 3 or 5 files, on a detached tree) | win | 11 caught / 2 not | ~35 min | `mut_*_win.txt` |
| the 4 mutation re-runs WITH my file added (M2 / M3 / M4 / M13) | win | **4 of 4 caught** | ~10 min | `log_mutation_plus_win.txt`, `mutation_plus_win.json` |
| `ruff check lumenairy/ tests/ validation/probe_verify_b7c_round2/` | **wsl** | **All checks passed** | -- | -- |

No red anywhere, on either build.  The one incomplete row is the brief's
wider `test_*niche*` glob (117 files): it reached 71% after 100 minutes
and was still advancing, on a box carrying 38 resident python processes
including three other multi-hour pytest sessions.  The builder's own scope
ran to completion green (345 passed) and covers what this verification needs
from that group -- `niche_r2` and `niche_r5`, the five tests the seam re-point
moved.

```
python -m pytest tests/unit/test_verify_b7c_round2.py \
  tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py \
  tests/unit/test_verify_b7c_multibranch.py \
  tests/unit/test_audit2609_b7c_multibranch_envelope.py -q --capture=sys
python -m pytest tests/unit/test_audit2609_a16*.py tests/unit/test_*niche*.py \
  tests/unit/test_v5_21*.py -q --capture=sys
python -m pytest tests/ -q --capture=sys -k census
python -m pytest tests/ -q --capture=sys -k walker
python -m pytest tests/ -q --capture=sys -k dispatcher_pin
python -m pytest tests/unit/test_public_api.py \
  tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py \
  tests/unit/test_audit_except_budget.py -q --capture=sys
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmb2 && ~/lumvenv/bin/ruff check lumenairy/ tests/ validation/probe_verify_b7c_round2/'
```

`.test_durations` carries the six new ids with their measured values and is
JSON-validated after the edit.

---

## 10. Ship recommendation

**Ship the library change.  Do not ship the bar's margins or the oracle-floor
table as stated, and add a gate that holds the bar and the side of the seam
the reading is taken on.**

What verifies and is worth shipping exactly as it stands:

* **the mechanism**, independently and on five new optics: 29 blown-up planes
  across five optics read 3.9235-4.0000 and healthy ones 0.956-1.022, and a
  slow control reads 0.999529-1.000207 across 18 mm of propagation;
* **the reading's placement.**  Claim 2 reproduces to four decimals on an
  oracle that shares no code with the builder's: the branch sum reads 1.0636
  where the completion reads 1.0021 at fidelity 0.9879.  Reading the returned
  field is right, and the argument for it is correct;
* **the R-5 class is genuinely closed.**  D1's plane is refused, and my own
  `Z` z = 3510 -- a single-valued map with a launched-power bracket of
  **1.021** returning a field at fidelity **0.647** -- is caught by this arm
  and by nothing else in the library.  That case alone justifies the change;
* **bit identity**: 0 moved of 57 on a matrix the builder did not choose,
  with the memory budget pinned;
* **cost**: one rasterisation, one trace, structurally proven on four optics,
  and the half-pitch render never reaches the caller;
* **the diagnostics**: `not_measured` is never silently `ok`, every return
  path carries the reading, and the refusal message names both readings, the
  branch count and a member that works.

What must not ship as a *claim*:

1. **"a 7 % gap, measured on eight optics and two builds, and a ninth optic
   could narrow it"** -- a ninth and a tenth narrow it to **3.8 %** on the
   fold ring and **0.4 %** over all planes, and the margin below the smallest
   refused reading falls to **0.05 %** (E1);
2. **"0 false refusals at fidelity >= 0.95"** -- two, on the fold ring, on the
   NA 0.3865 singlet, at 0.9578 and 0.9564 (claim 5 / E1);
3. **"reads 1 exactly, on any optic, at any plane, at any grid"** -- 5e-4,
   for a stated structural reason (E5);
4. **the oracle-floor table's "Debye vs exact" column and the
   `rel L2 ~ 0.37 eps` rule** -- under-reported 6x-41x by a measurement
   confined to the energy core (E6).  The NA-ceiling conclusion is
   strengthened, not weakened, by the correction.

And the gate must grow two tests before the next change to this module can be
trusted (E2, E3).  They are in `tests/unit/test_verify_b7c_round2.py`, pass on
both builds (34 passed with the shipped three on win and on wsl), and are
proved to close exactly those two gaps: with the file added, M2 goes from
`28 passed` to `1 failed, 33 passed` and M3 from `26 passed, 2 skipped` to
`1 failed, 30 passed, 3 skipped`, each on the intended id.

---

## 11. What I could not measure

* **the builder's own 30-case bit-identity matrix.**  I ran my own 57-case
  matrix instead -- a broader check, but not the same one.  If "26 / 0 / 4" is
  to stand as a pinned claim it should be re-run from their `bitid.py` on
  both trees;
* **the 328-plane population on the wsl build.**  As in round 2, the
  oracle-scored population was measured on win only.  What I did measure on
  both builds is the 23 READINGS every claim rests on -- both false refusals,
  the claim-2 split, D1, D2, the `Z` R-5 plane and the converged control --
  and they are **identical to 4.1e-16 or exactly, with every decision
  identical**.  The oracle never imports lumenairy, so the fidelity column
  cannot move across builds;
* **whether the two `X` refusals are "false".**  Both carry a 10 % energy
  error, so a stricter accept criterion would call them correct.  What does
  not depend on the criterion is that they are the smallest refused readings
  in the study and sit 1.0005x above the bar;
* **an accept criterion of my own.**  I report the report's two (0.883 and
  0.95) rather than inventing a third;
* **the wall-clock cost range.**  The same-code control reads 0.88x-1.57x on
  this box, so 1.1x-3.0x can be neither confirmed nor refuted here; it needs
  an idle machine;
* **the brief's wider `test_*niche*` glob** (117 files).  It reached 71 %
  after 100 minutes and was still advancing when this was written, on a box
  carrying 38 resident python processes including three other people's
  multi-hour pytest sessions.  The builder's own scope -- which contains
  `niche_r2` and `niche_r5`, the two files the seam re-point moved -- ran to
  completion green (345 passed, 1827.6 s).  Nothing in this verification
  depends on the remainder, but it is not measured;
* **why the `X` reading spikes at isolated planes** (1.0002 -> 1.1810 ->
  0.9989 over 20 nm).  It is consistent with the collapsing ring crossing
  pixel centres -- the mechanism the arbiter detects -- and the FIELD moves
  with it, so it is not a false alarm; but whether a 10 nm-wide spike is a
  property of the quadrature or of the meridional fold trace is not settled
  here;
* **the quadrature itself**, unchanged from the report: the arbiter detects
  the failure, it does not fix it.  WP-B7c's open item 1 remains open, now
  with a deterministic detector to regression-test a fix against.
