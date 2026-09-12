# VERIFY-A3 — adversarial re-verification of WP-A3 (the traced lens family)

Independent verifier.  Diff under test: commit `37d8afe7`, base `37d8afe7^`
(`_lens_traced.py` +1995, `_lens_traced_multibranch.py` +448,
`_lens_traced_uniform.py` +65, `_lens_imap.py` +122, 10 test files, 2 new).
Branch `audit-fixes-2026-09`; no git write commands were run.
Every number below was MEASURED on this box with `OPENBLAS_NUM_THREADS=1`,
either by re-running the audit's own repro scripts or with an oracle written
for this verification.

**The oracles I used that the library did not produce.**

1. `scratchpad/va3/oracle_trace.py` — a self-contained Newton-intersection +
   vector-Snell tracer (conic sag, `(-dz/dx, -dz/dy, 1)` normal,
   `t = mu d + (mu cos_i - sqrt(1 - mu^2 sin^2_i)) n`), written from the
   equations.  Only the glass dispersion comes from `lumenairy` (data, not
   algorithm).  A trimmed copy of it is embedded in the new test file.
2. `scratchpad/va3/p7_rs.py` — a BRUTE-FORCE Rayleigh–Sommerfeld quadrature
   (no FFT, no paraxial approximation, no ASM) from the exit-vertex plane,
   `E(r) = (1/i lambda) ∫ U(r') (z/R)(e^{ikR}/R) r' dr' dphi`, with the pupil
   field built from oracle 1 (ray-tube amplitude + `k·OPL`).  **Calibrated**:
   against `validation/oracles/caustic_fold_ref.npz` (the shipped independent
   Huygens reference) it agrees to **2e-4 of peak**, r2m 11.217 vs 11.219 µm,
   EE50 9.676 vs 9.677, EE80 13.404 vs 13.407 µm.
3. Closed forms: the tilted-congruence eikonal gradient
   `g = sign(R) u/|(u, R)|` with `u = x + R(L, M)/N`; the `form_error` screen
   `-k0 (n_after - n_before) · form_error`; the Chebyshev derivative
   Vandermonde by the second-kind recurrence.

---

## 1. Verdicts

| ID | verdict | what I re-measured | value |
|---|---|---|---|
| **S1** (P0) | **VERIFIED** | `repro/TR-SIBLINGS/repro_vertex.py`; `repro_vertex_field.py`; brute-force RS on CURVED-REAR fixtures | transverse 1.735e-12 µm / OPD 4.428e-12 waves at R2 = −25 mm (report's after-numbers reproduced digit for digit); field vs single-valued 0.0003 rad rms curved / 0.0050 flat; RS oracle: `caustic='uniform'` +4.3 % r2m / +7.3 % EE80 on R2 = −50 mm, the SAME envelope as on the plano fixture |
| **T1** (P1) | **VERIFIED** | `verify_trmain2.py` item 1; my own real/float32/non-contiguous/strided sweep, whole-grid AND banded | `complex128` out; **0.000e+00** relative difference against the complex-input call on 4 dtype × 2 band-mode combinations |
| **T2** (P1) | **VERIFIED-WITH-NOTES** | `verify_trmain2.py` item 3; `p3b_scan.py`; a 32-point (plane × launch-density) sweep on the D3 geometry | refusal at the BFL with the named remedy; 4×/8× band warns (3.988 / 8.102 / 13.07 / 61.78 / 106.5, identical to the report); **but I found 4 gain-side FALSE POSITIVES the report's sweep missed — fixed, see §3.2** |
| **T3** (P1) | **VERIFIED** | `verify_trmain2.py` item 2; a NEW N-SF11 prescription at 1.064 µm with two vignetting levels | spline 0.9501 / 17 493 px (report's number); on my fixture 0.9980 and 0.8657 of input power with the named warning, against the polynomial control's 0.9983 |
| **T4** (P1) | **VERIFIED** | `repro/TR-INFRA/p3b_fastphase.py` | runs on a refracting prescription on all three `preserve_input_phase` settings |
| **T5** (P1) | **VERIFIED** | a NEW trefoil figure map (90 nm, `cos 3θ·r³`) at λ = 1.064 µm on an N-BK7 ±80 mm / 2.5 mm singlet, against the closed-form screen | applied−exact **0.10 %** of the screen on S1 (analytic sibling 3.22 %), **0.00 %** on S2; sign check on a 100 nm bump: −0.29917 rad measured vs −0.29918 rad closed form; both caustic modes refuse |
| **T6** (P1) | **VERIFIED** | `repro/TR-INFRA/p4_reverse.py` | aspheric sag error **0.0e+00**; fwd/bwd OPL Δ 5.204e-18 m under BOTH thickness conventions; ray returns to its launch height exactly; every sag term negated; all top-level keys kept, `stop_index` remapped |
| **T7** (P2) | **VERIFIED** | `repro/TR-INFRA/p7_tilts.py`, `p8_misc.py` | whole-grid `max|L−L0|` **1.232e-15** (σ=0) / **2.220e-16** (σ=4); half-pixel bias **+4.48e-21** / **−2.60e-22**; ndarray index offset **−0.0000 px** |
| **T8** (P2) | **VERIFIED** | the WP's tests + signature derivation | `inspect.signature`-derived refusal; `segmented(dy≠dx)` raises |
| **T9** (P2) | **VERIFIED** | my own chunked-vs-whole-grid rebuild + tracemalloc | `np.array_equal` **True** on all three outputs at 250 kpt and 1 Mpt; peak 912 MB (114 float64/pt) → 104 MB (13.0/pt), **8.8×**, bitwise identical |
| **T10** (P2) | **VERIFIED** | the audit's own `repro/TR-INFRA/guard/g*.py` corpus + the two real drivers | g6 **False**, g9 **True**, g12 **False**, g1/g2/g3 True, g5/g7/g8/g10/g11 False, gbad True; `capstone_stageB.py` and `focus_scan_121.py` both **True** |
| **T11** (P3) | **VERIFIED** | `_cheb_dvander` against a re-created pre-fix copy | `max|Δ| = 0.0` at degrees 0/1/2/6/8/12/14, C-contiguous |
| **T12** (P2) | **VERIFIED** | the closed-form tilted congruence on THREE new (R, L, M) triples, both axes, a 4× ladder | `TiltedCarrier` gradient error **0.0** at every rung; ndarray error / derived `(7/24)dx²max|g''|` = **0.90–1.12**, scaling exactly as dx² |
| **T13** (P2) | **VERIFIED-WITH-NOTES** | my Snell tracer on 6 prescriptions (thin, thick-fast, diverging, vignetted, immersed, stop-on-S2) | the statistic is the ENTRANCE-disc max to **6 digits** on every fixture — the 3.14× overstatement is gone.  **But the code claims it gates on "the same disc the output mask uses", which is the EXIT disc, and on a thick fast element the two differ by 1.72× in the unsafe direction — fixed, see §3.1** |
| **T14/T15/T16** | **VERIFIED** | doc/notice items; the delegate list; `return_screen`+delegate; `float(sum(thicknesses))` | all present and behaving as described; `on_noncollimated='delegate'` always announces |
| **S8** (P2) | **VERIFIED** | `repro/TR-SIBLINGS/imap_cache.py`; an end-to-end flag flip through the public API | `key(det=True)==key(det=False)` **False**, same object **False**; `traced_flags(DETERMINISTIC_TRACED_FIT=False)` now moves the returned FIELD by 5.545e-14 where a cache hit would have made it 0; accuracy re-measured **3.598e-12 vs 1.845e-08 waves** (the report's numbers) |
| **S9** (P2) | **VERIFIED-WITH-NOTES** | the dark fill differenced on the FIELD; the rasteriser through every caustic suite | annulus vs whole-outside `max|ΔE|` **3.16e-25 (9.7e-27 of peak)**, `P_ann/P_full = 1.0000000000`; the 1.5 s-vs-25.4 s claim is an N = 2048 number — at N = 384/512 I measure only **1.1–1.2×**, so the headline is grid-size-specific (not wrong, but do not quote it without N) |
| **S11** (P3) | **VERIFIED-WITH-NOTES** | `_shift_clamped`, the turning-point counter | the clamped shift and the counter are correct; **the report's claim that `_count_interior_turning_points` "is now called by both meridional traces" is FALSE** — `_trace_meridional_cusp` still uses the raw `np.diff(np.sign(dxo))` form (`_lens_traced_uniform.py:624–627`).  Harmless (the cusp path needs `turns.size == 2`, positions and count together) but the report is wrong |
| **§15.9 `caustic='wave'`** | **VERIFIED** | d = 0 bit-identity; an independent `apply_real_lens` + ASM oracle; a brute-force RS oracle at three fold planes; an adversarial input sweep | `array_equal` **True** at d = 0; peak 7.90079 vs oracle 7.89709, `P_wave/P_oracle = 1.000000`; against the RS oracle **+2.8 / +3.0 / +3.5 % r2m and +0.8 / +1.0 / +1.6 % EE80** on plano and two curved-rear fixtures — the best of the three modes everywhere, and the only one with an answer at the axial focus; robust on complex64, real float32/64, Fortran-order, strided views and odd N (`P_out/P_in = 1.000000` in every case) |
| **WP-A2 §5.3 `stop_index`** | **VERIFIED-WITH-NOTES** | 15 spellings through the public API | `None ≡ 0 ≡ -2 ≡ np.int64(0)` and `1 ≡ -1 ≡ np.int32(1)` bit for bit; `2, 5, -3, 0.0, 1.5, 'first', True` all raise `apply_real_lens_traced: …`.  **The report's "Every in-range spelling returns a bit-identical field" is over-general**: it holds within each equivalence class, not across them (moving the stop from the entrance to the rear legitimately changes the field — measured non-identical on a 2-surface BK7 singlet) |
| **WP-A2 §5.5 aperture notice** | **VERIFIED** | `_warn_if_aperture_exceeds_grid` with both spellings | old spelling (Ny, dx) **0** warnings, new spelling (Nx, dx, N_y, dy) **1**, naming `N=64` |

Verified-correct items re-checked and still holding: `caustic='single'`
byte-identical to the default (`np.array_equal` True); masked pixels exactly
zero with no NaN/inf; screen-model energy within 1e-4 of the
aperture-transmitted input; the deterministic solve **byte-identical across
`n_workers` 1 / 2 / 4** (`max|Δ| = 0.000e+00`) with the fail-before flag
genuinely moving the answer (5.545e-14); `_compute_carrier('auto')` and
`TiltedCarrier` exact on spheres; the segmented partition reconstructing the
input to 5.55e-16 at `min_segment_power` 0 and 1e-3; the inverse map beating
the incumbent 5.1e3× on OPL.

---

## 2. The independent wave check on S1 (the P0)

The audit's S1 evidence and the WP's verification are both GEOMETRIC (ray
positions and OPL).  I added the missing wave-domain arm, because that is
where the defect would have shown up for a user.

`scratchpad/va3/p7_rs.py` builds the exit-vertex pupil field from my own Snell
trace and propagates it with a direct Rayleigh–Sommerfeld quadrature.  On the
shipped plano-rear `caustic_fold_ref` fixture the integrator reproduces the
library's independent Huygens reference to 2e-4 of peak (§0), so it is a
calibrated oracle.  Run at the same fraction of each lens's OWN back focal
length (`4.3704/4.565175 = 0.95733`):

| rear surface | mode | r2m vs RS | EE50 vs RS | EE80 vs RS |
|---|---|---|---|---|
| plano (the ONLY geometry the shipped corpus has) | multibranch | −12.0 % | −2.2 % | −8.8 % |
| | uniform | +3.8 % | +10.9 % | +6.9 % |
| | wave | +2.8 % | −7.1 % | +0.8 % |
| **R2 = −50 mm** (sag ≠ 0: the S1 geometry) | multibranch | −13.9 % | −4.8 % | −10.9 % |
| | **uniform** | **+4.3 %** | +10.9 % | **+7.3 %** |
| | wave | +3.0 % | −7.1 % | +1.0 % |
| **R2 = −20 mm** (3× the sag) | multibranch | −12.6 % | −2.9 % | −10.7 % |
| | **uniform** | **+3.4 %** | +9.8 % | **+5.4 %** |
| | wave | +3.5 % | −6.6 % | +1.6 % |

The curved-rear rows sit inside the plano-rear envelope, i.e. the sag no longer
enters the answer.  Pre-fix the fold radius itself was resolved on the sag
surface (477 µm / 3501 waves of transfer error at R2 = −25 mm), so nothing in
these columns could have been within 4 % of the oracle.  `caustic='uniform'`
reproduces the K4 docstring's own claim ("closes K1's −14.8 % r2m gap") on a
geometry no shipped fixture has.  **S1 VERIFIED in the wave domain.**

---

## 3. Defects I found and FIXED in WP-A3's own files

Both are in files I own for this task; both have their own regression test in
`tests/unit/test_audit2609_a3_verify_traced.py`; `ruff check` is clean.

### 3.1 T13's gate is the ENTRANCE disc, not the one the output mask uses (P2)

`_lens_traced.py` intersects the exit-NA significance mask with
`h_x² + h_y² <= (aperture/2)²` — the LAUNCH heights — under a comment saying
"Gate on the same disc the output mask uses".  The output mask is applied on
the OUTPUT grid coordinate (`np.copyto(E_out, 0, where=x² + y² > (a/2)²)`), and
the two sets are not the same: a ray entering outside `aperture/2` can still
land inside it.  Measured with my Snell tracer (λ = 1.0 µm, the element's own
launch lattice):

| prescription | entrance disc | output disc | all alive | ratio |
|---|---|---|---|---|
| R = ±51.68 mm, t = 4 mm, ap 24 mm (the WP's f/5 fixture) | 0.246183 | 0.257506 | 0.812199 | 1.05 |
| meniscus R = +30/+60 mm, t = 3 mm, ap 20 mm | 0.092428 | 0.094121 | 0.255092 | 1.02 |
| **R = ±20 mm, t = 12 mm, ap 18 mm (thick, f/1.1)** | **0.498089** | **0.858017** | 0.858017 | **1.72** |
| diverging R = ∓40 mm, ap 16 mm | 0.215007 | 0.211099 | 0.592113 | 0.98 |
| vignetted rear (semi 3.6 mm) | 0.071348 | 0.071348 | 0.071348 | 1.00 |
| immersed rear (glass_after = N-SF11) | 0.039964 | 0.040931 | 0.062353 | 1.02 |

The library's `na_exit` matched my oracle's ENTRANCE-disc column to six digits
on all six, so the implementation is exactly what the code does — but on the
thick fast element the guard's own advice, `dx <= lambda/(2·NA_exit)`, read
**1.00 µm where 0.58 µm is required**.  Understating is the unsafe direction
for a Nyquist guard.

**Changed** (`_lens_traced.py`, the `_sig` block): the comment now states which
disc is which and carries the measurements; `_exit_na_out` gains
`na_exit_entrance_disc`, `na_exit_output_disc` and `na_exit_guard`; and the
undersample WARNING is decided on `max(entrance, output)`.  `na_exit` itself is
deliberately UNMOVED — `propagators/carrier.py`'s `on_tilt_exact_grid` (default
action `'error'`) and `test_niche_c1_consolidation.py`'s restated relation are
calibrated against it, and a warning is never a returned field.

**Verified.** After: `na_exit 0.502114 / output 0.865101 / guard 0.865101`, and
the emitted message quotes `dx <= … = 0.58 um` where it used to quote 1.00 µm.
On the thin fixtures the guard moves by ≤ 5 % and nothing else changes.
Tests: `::test_exit_na_statistics_match_an_independent_snell_trace` (3 fixtures)
and `::test_the_undersample_guard_is_not_understated_on_a_thick_fast_element`.

### 3.2 The re-normalised energy tripwire still has gain-side false positives (P2)

The T2 follow-up replaced the denominator with "the launched power that reaches
the grid" and swept 8 planes × 4 launch densities on the D3 geometry, finding
[0.70, 1.16] with 0 warnings.  I swept the same geometry over a **wider plane
range** and found the band is still reachable, because `p_in` counts a launch
node only when its OWN mapped point lands on the grid while `p_out` counts every
pixel a triangle covers — including triangles that STRADDLE the grid boundary
with their nodes outside:

| launch density | plane | ratio | n_branch | degenerate | pre-fix verdict |
|---|---|---|---|---|---|
| `ray_subsample=8` | 100 mm | **3.257** | 1 | 0 / 1168 | **spurious warning** |
| | 110 mm | **2.563** | 1 | 0 / 1168 | **spurious warning** |
| | 120 mm | **2.069** | 1 | 0 / 1168 | **spurious warning** |
| `ray_subsample=4` | 200 mm | **2.736** | 1 | 0 / 5264 | **spurious warning** |

One branch per pixel and zero degenerate triangles: there is no coalescence
anywhere in those fields — this is the exact failure mode ("the give-away is in
the message itself: *1* branch per pixel") the T2 follow-up was written to
remove, merely moved to other planes.

**Changed** (`_lens_traced_multibranch.py`): a second, independent denominator
`_p_in_tri` — the launch power of the triangles that rasterise onto the grid,
`Σ (|E0|²+|E1|²+|E2|²)/3 · (h²/2)`.  Over the interior of the launch lattice it
is algebraically the node sum (each node is shared by six triangles, each
triangle averages three nodes); where a triangle straddles the boundary it
counts the whole triangle where the node sum counts none of it, so it BRACKETS
the true launched power from above.  The GAIN arm now requires
`p_out > 2 · max(p_in, _p_in_tri)`.  The collapse arm is unchanged (it is
already priced on the lower bound).  `power_ratio` is unchanged;
`power_ratio_triangles` is added to `return_diagnostics`.

**Verified.** All four false positives are silent
(triangle-ratios 0.865 / 0.681 / 0.549 / 0.694), and **no detection is lost**:
on the same fixture the true blow-up reads **1.803e+05 / 2.331e+05 / 1.291e+05
on BOTH denominators** at 0.98 / 0.99 / 0.995 of the ray-traced BFL (they agree
to 3 %), and the audit's silent 4×/8× band is unmoved (3.988 / 8.102 …).
Test: `::test_the_gain_arm_is_bracketed_against_boundary_straddling_triangles`
plus `::test_the_new_normaliser_separates_geometry_from_a_real_blow_up`.

### 3.3 One WP test weakened a decision it was meant to pin (P3)

`test_audit2609_a3_caustic_siblings.py::test_t2_energy_tripwire_is_two_sided_
and_sees_the_pre_focus_band` accepted ANY warning containing `'multibranch'` as
proof that the energy arm fired — including the degenerate-triangle census
warning, which is emitted from the same module with a different message and at
overlapping planes.  A broken energy tripwire would have passed it.  Narrowed
to `'reconstructed grid power is'`, with the other warnings printed in the
failure message.  The test still passes.

---

## 4. Tests I added

`tests/unit/test_audit2609_a3_verify_traced.py` — **17 tests, 4.3 s**, all with
derived two-sided bars and their measurements dated in the docstrings, no
wall-clock assertions, no `pytest.skip`:

* `_oracle_trace` — the embedded Newton + vector-Snell tracer (no
  `lumenairy.raytrace` anywhere in it).
* `test_exit_na_statistics_match_an_independent_snell_trace[3 fixtures]` — both
  NA statistics are what they claim, to 1 % against the oracle (oracle floor
  2e-3, separation 4 %–72 %).
* `test_the_undersample_guard_is_not_understated_on_a_thick_fast_element` — the
  fail-before for §3.1, including the micron figure quoted in the message.
* `test_the_two_normalisers_agree_when_the_grid_holds_the_aperture` — 1e-3
  relative, measured 6e-5, against a 130 % separation on the pathological grid.
* `test_the_new_normaliser_separates_geometry_from_a_real_blow_up` — two-sided;
  the blow-up plane is found by scanning fractions of the BFL the ORACLE
  measures, never a constant.
* `test_the_gain_arm_is_bracketed_against_boundary_straddling_triangles` — the
  fail-before for §3.2, asserting one branch and zero degenerate triangles so it
  cannot pass on a fixture that has developed a real caustic.
* `test_the_airy_annulus_does_not_move_the_returned_field` — S9 measured on the
  FIELD (the shipped test argues from `Ai(x)` and the constant): 1e-12 bar,
  measured 1.1e-26, with the completion proved to have ENGAGED.
* `test_ndarray_carrier_is_discretisation_limited_in_both_axes[2 fixtures]` —
  the `(7/24)dx²max|g''|` band in x AND y on curvature/tilt pairs the D1 pin
  does not use.
* `test_stop_index_spellings_collapse_into_their_equivalence_classes` — and it
  asserts the two classes DIFFER, so it cannot pass on a build that ignores the
  key.
* `test_stop_index_is_refused_up_front_with_this_functions_own_name[6]`.

---

## 5. Tests run

All with `OPENBLAS_NUM_THREADS=1`; no full-suite run.

| command | result | duration |
|---|---|---|
| `pytest test_audit2609_a3_traced_lens.py test_audit2609_a3_caustic_siblings.py` | **55 passed** | 14.5 s |
| `pytest test_niche_k4_uniform_caustic.py test_niche_r2_pearcey_cusp.py test_niche_s10_sibling_patterns.py test_v5_21_delta_audit.py test_niche_c1_consolidation.py` | **101 passed** | 276 s |
| `pytest test_fix_newton_pool_memory.py test_niche_d14_*.py test_niche_d15_*.py -s` | **91 passed, 3 skipped** | 190 s |
| `pytest test_fix_newton_pool_memory.py` (no `-s`) | **53 passed, 3 skipped** | 17.4 s |
| `pytest` a3 ×3 + delta-audit + k1 + k4 + r2 + s10 + c1 (AFTER my two module fixes) | **184 passed** | 298 s |
| `pytest` c9 + s8-sphere + tangent-facet + c15-inverse-map + banded-ray-density + p11 + v5_21_lens_accuracy_extensions | **184 passed, 2 deselected** | 241 s |
| `pytest tests/unit -k real_lens_traced` | **38 passed, 1 skipped** (PySide6 absent) | 25.4 s |
| `pytest test_niche_d1_tilted_carrier.py -s` | **33 passed** | 80 s |
| `pytest test_audit2609_a3_verify_traced.py` | **17 passed** | 4.3 s |
| `python validation/run_all.py test_lenses` | **PASS** (all checks) | 28.6 s |
| `ruff check` over all four modules and both edited/new test files | clean | — |

Repro scripts re-run: `repro/TR-SIBLINGS/repro_vertex.py`,
`repro_vertex_field.py`, `imap_cache.py`; `repro/orch/verify_trmain2.py`;
`repro/TR-INFRA/p4_reverse.py`, `p7_tilts.py`, `p8_misc.py`,
`p3b_fastphase.py`, `guard/g*.py`; `repro/TR-MAIN-2/p3b_scan.py`.  Every
after-number quoted in the WP report that I re-ran came back identical, with
the two exceptions recorded in §6 (OI-6, OI-7).

**Environment notes.**

* The `WinError 6` class the WP report describes did **not** reproduce in my
  shell: `test_fix_newton_pool_memory.py` passes with and without `-s`
  (53 passed / 3 skipped either way), as do `test_niche_d14_*` and
  `test_niche_d15_*`.  The three skips are `threadpoolctl` absent (an optional
  dependency, not a resource precondition).
* `test_niche_d1_tilted_carrier.py` is **no longer red**: 33 passed.  The
  containment floor the brief flagged (fixture 1.033 vs floor 2.880) has been
  resolved upstream since the brief was written.
* `validation/run_all.py test_lenses` now passes outright; the WP report's one
  failure (`apply_real_lens_traced_jax` hitting WP-A4's x64 refusal) is gone.
* No pre-existing failure of any kind was seen in 600+ tests across the traced
  layer.

---

## 6. Open items for the orchestrator

| # | severity | item |
|---|---|---|
| **OI-1** | P2 — **fixed here** | T13's entrance-vs-output disc (§3.1).  `na_exit` is unchanged, so `propagators/carrier.py` needs no action; but WP-A6 may prefer to read the new `na_exit_guard` key for `on_tilt_exact_grid`, which is the conservative statistic. |
| **OI-2** | P2 — **fixed here** | the tripwire's boundary-straddle false positives (§3.2). |
| **OI-3** | **P2 — NOT fixed** | The exit-NA Nyquist guard omits `n_exit`. It tests `dx > lambda/(2·NA)` with the VACUUM wavelength while `NA` is the bare exit direction cosine `sin θ`, so on a prescription ending in glass the sampling requirement is understated by exactly `n_exit` (the in-medium wavelength is `lambda/n`). Measured on an immersed rear (`glass_after='N-SF11'`, n = 1.75588, R = ±30 mm, ap 16 mm): the guard's statistic is `sin θ = 0.040931` while the criterion needs `n·sin θ = 0.071870`, so the advised `dx` is **1.76× too coarse**. Pre-existing (the guard never had the factor) and NOT part of T13, so I left it: the fix is one multiply, but it changes when an existing guard fires on every immersed design and belongs with whoever owns that calibration. |
| **OI-4** | **P3 — NOT fixed** | `_spectral_gap_cuts` was CHANGED, not just re-documented: the flanking-peak test is now restricted to the occupied band. Demonstrated difference on a synthetic marginal with out-of-band leakage — pre-fix `[-40.0]`, shipped `[]` — so `apply_real_lens_traced_segmented` can now produce a different segment count on the same field. There is **no changelog section and no test** for it (25 sections in `WP-A3_CHANGELOG.md`, none mentioning it), which is COMMON rule 8. The exact-reconstruction contract still holds (`max|Σsegments − E|/peak = 5.55e-16` at `min_segment_power` 0 and 1e-3). Add a changelog line and a pin, or revert the code and fix the docstring instead. |
| **OI-5** | **P3 — NOT fixed** | `_sample_local_tilts` silently CLIPS an aliased input tilt. The WP's own repro prints it: `repro/TR-INFRA/p7_tilts.py` §7c, launch tilt 0.8 at dx = 4 µm (grid Nyquist `sin = 0.164`) returns `max|L| = 0.1450` with the clip at 0.5 and **no warning**. T7 fixed the wrap and the half-pixel offset but not this. One `RuntimeWarning` when the raw estimate saturates against `max_sin` would close it. |
| **OI-6** | P3 — report correction | `WP-A3_REPORT.md` §2 S11 says `_count_interior_turning_points` "is now called by both meridional traces". It is called by `_trace_meridional_fold` only; `_trace_meridional_cusp` still counts with `np.diff(np.sign(dxo))` (`_lens_traced_uniform.py:624–627`). Harmless today — the cusp path needs both the count AND the positions and takes `turns.size != 2` as its gate — but the sentence should be corrected. |
| **OI-7** | P3 — report correction | `WP-A3_REPORT.md` "Every in-range spelling returns a **bit-identical** field" (WP-A2 §5.3) is over-general. Measured on a 2-surface BK7 singlet with a 0.5 mm beam in a 2 mm aperture: `None ≡ 0 ≡ -2` and `1 ≡ -1` are each bit-identical, but the two CLASSES differ (as they physically must — the stop moves through 3 mm of glass). Restate as "every spelling of the same surface". |
| **OI-8** | P3 | The S9 headline "25.4 s → ~1.5 s" is an N = 2048 number. At N = 384 / 512 with `ray_subsample=1` the annulus buys only **1.1–1.2×** (2.94 vs 3.11 s, 2.92 vs 3.59 s), because the dark fill is a small share of the call there. Correct, but quote it with its grid size. I did not re-run the N = 4096 / 12 GB rasteriser fixture (shared machine, COMMON rule 5). |
| **OI-9** | P3 — not WP-A3 | `test_niche_k4_uniform_caustic.py:296` calls `pytest.skip` on `available_memory_bytes()`, which `docs/TESTING_STANDARDS.md` §4 forbids ("Never `pytest.skip` on a resource check"). Pre-existing and outside the WP's hunks in that file; worth a ticket. |
| **OI-10** | P3 — pre-existing, cross-file | `_lens_traced._compute_carrier`'s ndarray branch is single-pitch: `np.gradient(W_full, dx, dx)` and `/dx` on both axes. `apply_real_lens` (`_lens_real.py`, six call sites) DOES support `dy ≠ dx`, so an anamorphic analytic call with `conjugate=<ndarray>` differentiates and samples y with the x pitch. Unreachable through `apply_real_lens_traced` (square-pixel refusal). Fix is an optional `dy=None` parameter defaulting to `dx`; it touches WP-A2's call sites, so I did not make it. |
| **OI-11** | P3 — observation, not a defect | On a prescription whose vignetting comes from a per-surface `semi_diameter`, the DEFAULT (`newton_fit='polynomial'`) path returns unvignetted power: the least-squares fit extrapolates smoothly over the dead samples and the amplitude leg (`apply_real_lens`) does not read `semi_diameter` at all. Measured: an N-SF11 singlet with `semi_diameter=1.4 mm` inside a 5 mm aperture gives `P/P_in = 0.9983` and 21 821 non-zero pixels — identical to the un-vignetted run — while the (now working) spline path gives 0.8657 / 6 637. Consistent with `semi_diameter` being a ray-trace-only key, but a caller reading T3's story could reasonably expect the default path to show the vignetting. |

## 7. Files I touched

Modified (all WP-A3-owned):

* `lumenairy/elements/_lens_traced.py` — §3.1 (the two-disc comment, the three
  new `_exit_na_out` keys, the guard priced on `max`).
* `lumenairy/elements/_lens_traced_multibranch.py` — §3.2 (`_p_in_tri`, the
  bracketed gain arm, `power_ratio_triangles`).
* `tests/unit/test_audit2609_a3_caustic_siblings.py` — §3.3.

New:

* `tests/unit/test_audit2609_a3_verify_traced.py` (17 tests).
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-A3.md`.

Scratch probes (untracked, kept so every number here can be re-measured):
`scratchpad/va3/oracle_trace.py`, `p1_quick.py`, `p2_na_exit.py`,
`p3_tripwire.py`, `p3b_d3.py`, `p4_misc.py`, `p7_rs.py`, `p7b_rs.py`,
`p8_darkfill.py`, `p9_t5.py`, `p10_seg.py`.

## 8. Changelog text for my two fixes

### Fixed -- traced lens: the exit-NA undersample guard was priced on the entrance disc (VERIFY-A3)

`apply_real_lens_traced`'s exit-NA statistic is intersected with the ENTRANCE
aperture disc (audit T13, which removed a 3.14x overstatement), but the mask the
returned field carries is applied on the OUTPUT grid.  On a thick fast element
those sets differ: measured against an independent Newton+Snell trace at
lambda = 1.0 um, `R = +-20 mm / t = 12 mm / aperture 18 mm` reads 0.49809 on the
entrance disc against 0.85802 on the output disc, so the guard's advice
`dx <= lambda/(2*NA_exit)` said 1.00 um where 0.58 um is required -- 1.72x too
coarse, in the unsafe direction.  The WARNING is now decided on the larger of
the two, and `_exit_na_out` reports `na_exit_entrance_disc`,
`na_exit_output_disc` and `na_exit_guard` alongside the unchanged `na_exit`
(which `propagators/carrier.py`'s `on_tilt_exact_grid` reads and which is
therefore deliberately not moved).  Thin elements move by <= 5 %
(0.246183 -> 0.257506 on the audit's f/5 fixture).

### Fixed -- caustic siblings: the energy tripwire's gain arm is bracketed against boundary-straddling triangles (VERIFY-A3)

The launched-power normaliser counts a launch node only when its own mapped
point lands on the output grid, while the reconstructed power counts every pixel
a triangle covers -- including triangles that straddle the grid boundary with
their nodes outside.  On a coarse launch lattice over a grid much smaller than
the beam that mismatch alone reached 3.26x: measured on the delta-audit's D3
fixture (a 6 mm aperture on a 1.2 mm grid) at `ray_subsample=8`,
z = 100 / 110 / 120 mm gives 3.257 / 2.563 / 2.069 with `n_branch = 1` and ZERO
degenerate triangles -- three spurious `RuntimeWarning`s with no coalescence
anywhere, from the same geometry the launched-power normaliser was introduced to
quieten.  The gain arm now requires the excess to clear a second denominator,
the launch power of the triangles that rasterise onto the grid, which counts a
straddling triangle whole and so bounds the launched power from above.  All four
false positives go silent (bracketed ratios 0.549..0.865) while the real
blow-up is unmoved: 1.803e+05 / 2.331e+05 / 1.291e+05 at 0.98 / 0.99 / 0.995 of
the ray-traced BFL on BOTH denominators, and the audit's silent 4-8x pre-focus
band still warns (3.988 / 8.102 / 13.07).  `return_diagnostics` gains
`power_ratio_triangles`; `power_ratio` is unchanged.

---

# 9. Follow-up (coordinator rulings 1–7, 2026-09-12)

All seven rulings implemented or actioned.  Ownership as granted: the four
traced modules, the A3 test files, `_lens_real.py`'s six `_compute_carrier`
call sites (OI-10 only), `test_niche_k4_uniform_caustic.py`, and the two WP-A3
report/changelog documents.  No git write commands.  `ruff check` clean and
`import lumenairy` OK on every file touched.

| ruling | item | status | headline measurement |
|---|---|---|---|
| 1 | **OI-3** in-medium exit-NA guard | **implemented** | immersed N-SF11 rear: advised `dx` **12.216 → 6.957 µm** (ratio 1.75588 = `n_exit` exactly); N-BK7 rear 5.252 → 3.484 µm; **air rear bit-identical** (`n_exit == 1`) |
| 2 | **OI-4** `_spectral_gap_cuts` | **documented + pinned** | pre-fix `[-40.0]` → shipped `[]` on an out-of-band-leakage marginal; a real two-lobe gap still `[0.0]` under both; segments still sum to the input at 5.55e-16 |
| 3 | **OI-5** tilt-saturation warning | **implemented** | fires on the audit's own `p7_tilts` §7c case (reading 0.885 of the 0.16375 Nyquist, `max_sin=0.5` unreachable) and on a genuine clip (50.00 % replaced); silent at 0.183 and 0.611 of Nyquist and on a collimated field |
| 4 | **OI-9** k4 resource skips | **replaced** | both `pytest.skip`s → asserted preconditions; the bound is now the MEASURED 0.229 GB peak doubled, against the 1.58 GB the skip demanded (7× too conservative); both gates now RUN |
| 5 | **OI-10** `dy` through the ndarray carrier | **implemented** | y-gradient error at dy = 3 dx **1.051e-01 → 1.198e-07**; at dy = 0.4 dx **1.936e-02 → 5.622e-09**; `dy is None` / `dy == dx` bit-identical |
| 6 | **OI-11** vignetting on the default path | **implemented (the preferred arm)** | polynomial `P/P_in` **0.9983 / 21 821 px → 0.8840 / 7 093 px**, within 2 % of the spline's 0.8657 / 6 637; un-vignetted call bit-identical; rim case (semi = aperture/2) unmoved to 1e-4 and silent |
| 7 | **OI-6 / OI-7 / OI-8** | **corrected** | the two WP-A3 report sentences rewritten in place with a dated correction note; the S9 headline now carries its grid size in both the report and the changelog |

## 9.1 What changed, and how each was verified

**Ruling 1 — OI-3, the in-medium Nyquist guard** (`_lens_traced.py`, the `_sig`
block).  The exit leg runs in the medium after the last surface, so a ray at
angle θ carries transverse spatial frequency `n_exit·sinθ/λ_vac`; the guard
compared the bare direction cosine against `λ/(2dx)`.  `_na_guard` is now
`n_exit · max(entrance-disc, output-disc)`, the message names both factors and
says which medium, and `power_frac_above_nyquist` is compared in the same
units.  Measured (λ = 1.0 µm, R = ±30 mm, aperture 16 mm, N = 512):

| rear medium | `n_exit` | bare sin θ | guard NA | advised dx |
|---|---|---|---|---|
| air | 1.00000 | 0.257506 | 0.257506 | 1.942 µm (**unchanged**) |
| N-BK7 | 1.50750 | 0.095201 | 0.143515 | 5.252 → **3.484 µm** |
| N-SF11 | 1.75588 | 0.040931 | 0.071870 | 12.216 → **6.957 µm** |

Pinned two-sided by `::test_the_exit_na_guard_is_priced_on_the_in_medium_
numerical_aperture`: the immersed arms require the ratio to be exactly `n_exit`
(rtol 1e-9 — it is one multiply, so the only error is float64 round-off ~1e-16
against a 1.76× effect) and the message to name the medium; the air arm
requires `na_exit_guard == bare` bit-identically, which is the no-op every
fixture in the tree exercises.  `na_exit` / `na_exit_entrance_disc` /
`na_exit_output_disc` stay bare direction cosines, so nothing calibrated
against them (`propagators/carrier.py`'s `on_tilt_exact_grid`,
`test_niche_c1_consolidation.py`'s NA relation) moves; `n_exit` is now reported
so the two are convertible.  Migration note in the changelog.

**Ruling 2 — OI-4, `_spectral_gap_cuts`.**  Code kept.  A changelog section was
added (naming audit T11 as the origin and stating the behaviour change on
`apply_real_lens_traced_segmented` with a migration note), and two pins:
`::test_spectral_gap_cuts_only_counts_peaks_inside_the_occupied_band` re-creates
the pre-fix predicate inline and asserts `[-40.0]` → `[]` on a synthetic
marginal (one in-band lobe at f = +20, one out-of-band at f = −80, support
declared [−40, +40]) while a genuine two-lobe gap is `[0.0]` under both; and
`::test_the_angular_segmentation_still_sums_back_to_the_input` pins the
contract the change must not touch — `max|Σsegments − E|/peak = 5.55e-16` at
`min_segment_power` 0 and 1e-3, bar 1e-13.

**Ruling 3 — OI-5, the tilt-estimator saturation notice**
(`_lens_traced.py::_sample_local_tilts`).  The estimator reads a wrapped phase
difference, so it cannot return a cosine above `λ/(2dx)`; a steeper tilt folds
in.  A single `RuntimeWarning` now fires in two situations — the `max_sin` clip
actually biting (naming the clipped fraction), or `max_sin` being beyond what
the grid can carry while the reading has run to within 20 % of the fold (naming
both numbers).  Measured on the audit's own fixture family (λ = 1.31 µm,
dx = 4 µm, Nyquist sin = 0.16375):

| case | reading | fraction of Nyquist | warns |
|---|---|---|---|
| §7c, launch tilt 0.8, `max_sin=0.5` | 0.1450 | **0.885** | **yes** |
| tilt 0.8, `max_sin=0.10` (a real clip) | 0.1000 | — | **yes** (50.00 % replaced) |
| tilt 0.10, `max_sin=0.5` | 0.1000 | 0.611 | no |
| tilt 0.03, `max_sin=0.5`, σ = 0 and σ = 4 | 0.0300 | 0.183 | no |
| collimated, σ = 4 | 0.0 | 0 | no |

1.4× of gap above the bar and 1.3× below.  Confirmed end to end: running the
audit's own `repro/TR-INFRA/p7_tilts.py` under `-W always` now emits the
warning exactly once — on 7c — naming `lambda/(2 dx) = 0.16375`; 7a and 7b stay
silent.  (The repro's own printed string still reads "silently WRONG, no
warning"; it is a hard-coded literal, not a measurement, and the script does not
capture warnings.)  Pinned by
`::test_the_local_tilt_estimator_says_when_it_is_at_its_sampling_limit`.

**Ruling 4 — OI-9, the k4 resource skips.**  Both sites replaced by asserted
preconditions.  The skip demanded `8·N²·16 + 1.5` GB = 1.58 GB; the MEASURED
tracemalloc peak of the N = 768 uniform call is **0.229 GB** (24 grid-units of
`16N²`; the complex128 field itself is 0.0094 GB).  The requirement is that
measurement doubled and it fails loudly with the number.  `pytest.skip` no
longer appears in the file, and all 11 tests — including both
`caustic_fold_ref` gates — run and pass.

**Ruling 5 — OI-10, `dy` through the ndarray carrier.**
`_compute_carrier(..., dy=None)`; the ndarray branch differentiates with
`np.gradient(W, dy, dx)` and indexes each axis with its own pitch AND its own
sample count (`N = X.shape[0]` was the row count, used for both).
`_lens_real.py`'s six call sites forward their `dy`.  Verified against the
closed-form tilted congruence (R = −30 mm, L = 0.046, M = 0.031), with the bar
extended to carry the cross term the anamorphic case needs —
`bar_x = (7/24)dx²|∂²g_x/∂x²| + (1/8)dy²|∂²g_x/∂y²|`, and the mirror for y:

| grid | analytic err | ndarray err_x / bar | ndarray err_y / bar | dy := dx err_y |
|---|---|---|---|---|
| dx = 20, dy = 60 µm | 0.0 | 3.51e-08 / 4.10e-08 = 0.855 | 1.198e-07 / 1.358e-07 = 0.882 | **1.051e-01** |
| dx = 25, dy = 10 µm | 0.0 | 2.71e-08 / 2.88e-08 = 0.941 | 5.62e-09 / 5.90e-09 = 0.953 | **1.936e-02** |
| dx = dy = 20 µm | 0.0 | 1.89e-08 / 2.05e-08 = 0.922 | 1.36e-08 / 1.48e-08 = 0.920 | identical |

Pinned by `::test_the_ndarray_carrier_uses_its_own_pitch_on_each_axis` (two
anamorphic grids, with a fail-before arm requiring the `dy := dx` call to
overshoot by > 30×) and `::test_the_ndarray_carrier_is_bit_identical_on_a_
square_grid` (`{}`, `dy=None` and `dy=dx` give `array_equal` gradients and
eikonal).  `test_audit2609_a2_analytic_lens.py` passes unchanged.

**Ruling 6 — OI-11, vignetting on the default path.**  Took the preferred arm.
A `_dead_launch_mask` is now built whenever any launch ray dies, and BOTH fits
reject an output pixel whose converged entrance solution lands on a dead node
(the spline already did this for its filled nodes; the helper and the Newton
payload key are shared, so pool and serial stay bit-identical).  Measured on an
N-SF11 singlet, N = 192, dx = 30 µm, λ = 1.064 µm, aperture 5 mm:

| rear `semi_diameter` | polynomial before | polynomial after | spline | notice |
|---|---|---|---|---|
| none | 0.9983 / 21 821 px | **0.9983 / 21 821** (bit-identical: the mask is `None`) | 0.9983 / 21 821 | silent |
| 2.5 mm (= aperture/2) | 0.9983 / 21 821 | **0.9983 / 21 821** (unmoved to 1e-4) | 0.9980 / 21 319 | silent |
| 2.0 mm | 0.9983 / 21 821 | **0.9936 / 17 457** | — | 468 rays |
| 1.4 mm | 0.9983 / 21 821 | **0.8840 / 7 093** | 0.8657 / 6 637 | 916 rays |

The notice counts only dead nodes INSIDE the clear aperture, because the launch
square's corners sit at 1.06 aperture radii and die against any
`semi_diameter ≤ aperture/2` — on every ordinary prescription, and entirely
outside the disc the output is masked to.  Pinned two-sided by
`::test_a_vignetting_semi_diameter_reaches_the_default_polynomial_answer`.
Residual, recorded rather than fixed: the CuPy branch (`use_gpu=True`,
polynomial only) keeps the historical extrapolating behaviour, because
`_landed_on_filled_node` is a NumPy kernel — stated in the code and the
changelog.

**Ruling 7 — the report corrections.**  `WP-A3_REPORT.md`: the S11 sentence now
reads "called by `_trace_meridional_fold`" with a dated correction note
explaining why the cusp trace keeps the raw form; the `stop_index` sentence now
reads "every spelling of the SAME surface", with the equivalence classes spelt
out and the reason the two classes must differ; the S9 row carries "the
25.4 s → ~1.5 s headline is an **N = 2048** number" with the 1.1–1.2× measured
at N = 384/512 and the 9.7e-27 field bound.  `WP-A3_CHANGELOG.md`: the same
qualification on its S9 line, and my §8 text plus the five new items appended as
a "VERIFY-A3 follow-up" block (33 `###` sections total).

## 9.2 Tests run after the follow-up

| command | result | duration |
|---|---|---|
| `pytest` a3 ×3 + delta-audit + k1 + k4 + r2 + s10 + c1 | **193 passed** | 309 s |
| `pytest` c9 + s8-sphere + tangent-facet + c15-inverse-map + banded-ray-density + p11 + v5_21_lens_accuracy_extensions + **d1** | **217 passed, 2 deselected** | 313 s |
| `pytest` newton-pool + d14 + d15 + **test_audit2609_a2_analytic_lens** (the `_lens_real.py` consumer) `-s` | **140 passed, 3 skipped** | 196 s |
| `pytest tests/unit -k real_lens_traced` | **38 passed, 1 skipped** | 30.7 s |
| `pytest test_audit2609_a3_verify_traced.py` | **25 passed** (17 → 25) | 6.6 s |
| `pytest test_niche_k4_uniform_caustic.py` | **11 passed, 0 skipped** (was 11 with 2 resource skips available) | 5.7 s |
| `python validation/run_all.py test_lenses` | **PASS** | 30.0 s |
| `repro/TR-SIBLINGS/repro_vertex.py`, `repro/orch/verify_trmain2.py`, `repro/TR-INFRA/p7_tilts.py`, `p8_misc.py` | unchanged numbers; p7_tilts now warns on 7c | — |
| `ruff check` + `python -c "import lumenairy"` on every file touched | clean / OK | — |

No failures, no new skips, no pre-existing failures.  `test_niche_d1_tilted_
carrier.py` is included in the second batch and passes (33 tests).

## 9.3 Files touched in the follow-up

Modified:

* `lumenairy/elements/_lens_traced.py` — OI-3 (in-medium guard + `n_exit` key),
  OI-5 (the saturation notice), OI-10 (`dy=` on `_compute_carrier`), OI-11
  (`_dead_launch_mask` on both fits + the clear-aperture-gated notice).
* `lumenairy/elements/_lens_real.py` — OI-10, the six `_compute_carrier` call
  sites only (granted scope; +17/−6 lines, all of them that call).
* `tests/unit/test_niche_k4_uniform_caustic.py` — OI-9, both resource skips.
* `tests/unit/test_audit2609_a3_verify_traced.py` — 8 new tests (17 → 25).
* `docs/.../fixes/WP-A3_REPORT.md` — OI-6 / OI-7 / OI-8 corrections.
* `docs/.../fixes/WP-A3_CHANGELOG.md` — the S9 qualification and the appended
  "VERIFY-A3 follow-up" block (7 new sections).
* `docs/.../fixes/VERIFY_WP-A3.md` — this section.

`lumenairy/elements/_lens_traced_multibranch.py` and
`tests/unit/test_audit2609_a3_caustic_siblings.py` carry my §3.2 / §3.3 fixes
and were committed by the coordinator before this follow-up began; they are
unchanged since.

## 9.4 Open items after the follow-up

All of §6 is now closed except:

* **OI-11 residual (P3)** — `use_gpu=True` + `newton_fit='polynomial'` +
  vignetting keeps the extrapolating behaviour, because the rejection kernel is
  NumPy.  Stated in the code and the changelog; a CuPy twin is a small,
  separate job for whoever owns the GPU path.
* **OI-8 (P3, informational)** — I still have not re-run the N = 4096 / 12 GB
  rasteriser fixture (shared machine, COMMON rule 5), so the S9 memory
  headline remains the WP's own measurement, now quoted with its grid size.
