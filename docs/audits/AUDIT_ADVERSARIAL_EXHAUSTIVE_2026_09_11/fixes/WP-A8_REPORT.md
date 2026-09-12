# WP-A8 report — thin elements, DOEs, materials (audit 2026-09-11 §5, rows E1–E3, E5–E7)

Branch `audit-fixes-2026-09`, working tree, 2026-09-12.  Environment: CPython
3.14.6, numpy 2.4.6, `refractiveindex` installed (database commit
`a66ef8805cdb200973fc7ae9181587e1d89d14eb`), jax installed, **cupy not
installed** (CuPy paths desk-checked only).  Every python invocation ran with
`OPENBLAS_NUM_THREADS=1`.

## 1. Summary

| ID | Status | Files:lines | Tests | Oracle | Measured before → after |
|---|---|---|---|---|---|
| **E1** (P0) wrong Sellmeier rows | **fixed** | `lumenairy/glass.py:126-146` | `tests/unit/test_audit2609_a8_glass.py::test_e1_bundled_row_reproduces_the_manufacturer_data_sheet`, `::test_e1_repaired_rows_are_not_the_pre_fix_rows` | SCHOTT data-sheet n_d / V_d (independent of the coefficients) + refractiveindex.info SCHOTT-optical | N-BAF52 n_d 1.637147 → **1.608631** (sheet 1.60863), V_d 42.469 → **46.597** (46.60); N-LAK33A 1.754279 → **1.753930**, V_d 53.031 → **52.271**; N-LAK33B 1.755294 → **1.755000**, V_d 52.940 → **52.300**.  N-BK7 / N-SF11 unchanged and exact |
| **E1** value cross-check | **fixed (added)** | `glass.py:1079-1253`, `:1255` | `::test_e1_whole_bundled_table_agrees_with_refractiveindex_info`, `::test_e1_value_gate_rejects_the_pre_fix_rows`, `::test_e1_import_time_check_stays_structural_only` | refractiveindex.info, n_d to 5e-5 and V_d to 1e-3 relative | 0 rows checked by value → **76 rows** checked in 0.46 s, 0 discrepancies; the pre-fix rows re-injected flag exactly those 3 and raise |
| **E2** (P1) `NoExtinctionCoefficient` | **fixed** | `glass.py:1896-1946` | `::test_e2_complex_index_never_raises_for_a_catalogue_glass`, `::test_e2_extinction_is_finite_and_non_negative`, `::test_e2_missing_kappa_warns_once_and_returns_zero`, `::test_e2_pages_that_do_carry_k_keep_their_value_and_sign` | exhaustive sweep of every tuple-registered glass at 1.31 / 1.55 µm | **7 of 44 raised → 0**; and a second arm found here: `E-BAK1`, `E-LAK04` returned `n + nan·j` → now warn + κ = 0.  N-BK7 κ unchanged (1.4361e-7 at 1.55 µm) |
| **E3** (P1) turbulence √2 | **fixed** (default change) | `elements/elements.py:1333`, `:1151` | `tests/unit/test_audit2609_a8_turbulence.py` (18 tests) | the lattice's exact discrete structure function; Kolmogorov 6.88 (r/r0)^(5/3); an explicit Schmidt `ft_phase_screen` reference | D_meas/D_lattice at r = 5 mm **1.987 → 0.994**; variance/ΣPSD·df² **1.976 → 0.988** (orch repro); D/D_Kolm at 0.05 r0 **1.584 → 0.798** |
| **E3** subharmonics | **fixed (added)** | `elements/elements.py:1168`, `:1352` | `::test_e3_subharmonics_recover_the_large_scale_structure_function`, `::test_e3_subharmonic_sum_is_the_separable_form_of_the_direct_sum` | Kolmogorov structure function, 40 seeds | D/D_Kolm at r = 3.2 r0 **0.463 → 0.777** with `subharmonics=3`; default 0, bit-identical |
| **E5** (P1) GRIN power | **fixed** (default change) | `elements/_lens_thin.py:1186`, `:1258-1281`, `:1305` | `tests/unit/test_audit2609_a8_thin_elements.py::test_e5_*` (8 tests) | the rod's exact ABCD `C = −n0 g sin(gd)`; an independent ASM focus scan | screen power ratio to exact **0.6366 → 1.0000** at the quarter pitch; ASM focus/f_exact 0.637 → **0.994** |
| **E6** HS TIS | **fixed** | `elements/bsdf.py:503-528`, `:61`, `:130-201` | `::test_e6_harvey_shack_tis_*`, `::test_e6_base_class_quadrature_*` | closed form + a 200 001-point geometric-grid trapezoid integral built in the test | TIS(l = 1e-3) **3.54486e-5 → 4.34027e-5** (−18.3 % → −9.3e-8); l = 1e-4 −31.8 % → −7.0e-6; base quadrature worst error **−32 % → 1.4e-6** at the same node count |
| **E6** `make_bsdf` keys | **fixed** | `elements/bsdf.py:617-703` | `::test_e6_make_bsdf_*` (4 tests) | direct construction | `{'sigma','scatter_fraction'}` silently → 10× wider / 50× weaker lobe; now raises.  `A`/`B`/`C` aliases accepted |
| **E6** Klein-Cook guard | **fixed (added)** | `elements/thin_grating.py:1-108`, `:196` | `::test_e6_klein_cook_guard_fires_in_the_bragg_regime`, `::test_e6_short_period_guard_fires_when_q_is_small`, `::test_e6_grating_fourier_coefficients_are_untouched` | Klein-Cook Q; a direct 8192-point FFT of the transmittance (unchanged) | **0 warnings at Q = 62.8 → 1**; silent at Q = 0.13; ΣT = 1.0000 throughout (why closure cannot be the check) |
| **E6** MLA separable | **fixed** | `elements/doe.py:256-284` | `::test_e6_microlens_array_separable_form_is_bit_identical`, `::test_e6_microlens_array_allocates_a_handful_of_grids_not_ten`, `::test_e6_microlens_array_physics_is_untouched` | the pre-separable implementation written out in the test | peak **14.13 → 3.25** float64 grids (identical at N = 1024 and 2048, one implementation per process — VERIFY-A8 correction of the 10.13 first reported); median 426 → **144 ms** at N = 2048; output bit-identical |
| **E6** `sample_scatter_rays` | **fixed** | `elements/bsdf.py:113-127`, `:306-326`, `:407-426`, `:539-566`, `:582-615`, `:753-771` | `::test_e6_sample_scatter_rays_is_vectorised_and_distribution_preserving`, `::test_e6_batched_rotation_equals_the_per_ray_reference`, `::test_e6_harvey_shack_sampler_matches_its_power_weighted_density`, `::test_e6_harvey_shack_sampler_makes_no_rejection_loop` | the per-ray loop it replaces; a numerically-integrated CDF | 20 000 rays: **113× / 327× / 377×** (lambertian / gaussian / harvey_shack).  HS `sample()` of 20 k: 5.47 → 3.73 ms, 7.33 → 3.28 ms, **24.93 → 3.58 ms**, rejection eliminated (acceptance was 46 % / 9.2 % / 1.3 %) |
| **E7** Noll pointer | **fixed** | `elements/elements.py:476-484` | `::test_e7_zernike_docstring_no_longer_points_noll_users_at_the_osa_map` | `zernike_index_to_nm(5) == (2, +2)` (OSA) vs Noll's (2, −2) | docstring corrected; no converter is claimed |
| **E7** `'air'`/`'vacuum'`/`'__MIRROR__'` | **fixed** | `glass.py:1049-1076` (comment), `:1650-1662`, `:1852-1866` | `::test_e7_air_defaults_to_one_for_every_spelling`, `::test_e7_registered_air_callable_is_honoured`, `::test_e7_documented_non_entries_really_are_absent`, `::test_e7_exemption_list_documents_only_legal_names` | Edlén ambient model | a registered `'air'` callable was ignored (n = 1.000000 vs 1.000274 at 1.064 µm, **274 µm/m of OPD**); now honoured.  Defaults unchanged, `list_glasses()` still 77 names |
| **E7** non-square DOE cell | **fixed** | `elements/doe.py:155-163` | `::test_e7_periodic_phase_mask_rejects_a_non_square_cell`, `::test_e7_periodic_phase_mask_square_behaviour_is_unchanged` | direct occupancy count | `(4,8)`: half the design silently unused → named `ValueError`; `(8,4)`: bare `IndexError` → named `ValueError` |
| **E7** grey-pixel aperture | **fixed (added)** | `elements/elements.py:226-360` | `::test_e7_aperture_gray_edge_removes_the_area_quantisation`, `::test_e7_aperture_hard_edge_is_the_default_and_unchanged`, `::test_e7_aperture_gray_edge_has_jax_parity`, `::test_e7_aperture_gray_edge_preserves_dtype_and_validates_its_kwargs` | analytic disc area | D/dx = 50 px on ONE grid: **−0.5345 % → +0.0384 %** (`edge='gray'`), −0.00056 % at 16×16; rms over the 12 rim placements the test sweeps 0.386 % → 0.044 %; 200 px 0.031 % → 0.0041 % (VERIFY-A8 re-measurement — the −0.942 % first paired here is the audit's hard reading on a different grid).  Default unchanged, bit-identical |
| **E7** reference-plane doc | **fixed** | `elements/_lens_thin.py:552-578` | `::test_e7_spherical_vs_real_lens_reference_plane_is_documented` | re-run of `repro/THIN-ELEMENTS-GLASS/p_real.py` | re-measured 19.3111 mm vs 18.8080 mm = **503 µm (2.6 %)**, 1.708 rad rms — now stated in the See Also |
| **E7** `surface_sag_general(R=0)` | **not implemented — outside ownership** | `elements/lenses.py:241, :249` (WP-A2) | — | reproduced on HEAD | `[nan nan]` plus four anonymous numpy RuntimeWarnings; see §5 |
| **E6** `surface_sag_general` conic memory | **not mine** | — | — | — | the WP file assigns it to WP-A2 |

Also verified and reported below: **all 24 formula-3 `POLYNOMIAL_COEFFICIENTS`
rows** — the partition report's "single biggest remaining gap" — are exact to
0.00e+00 against their catalogue pages.

## 2. Per finding

### E1 (P0) — three bundled Sellmeier rows are a different glass

**What was wrong.**  `N-BAF52`, `N-LAK33A` and `N-LAK33B` are registered
`'__sellmeier__'`, and that branch fires before any refractiveindex.info lookup,
so the bundled row was what *every* install returned.  Reproduced on HEAD with
`repro/orch/verify_rt_glass.py`: bundled n_d 1.637147 / 1.754279 / 1.755294 vs
catalogue 1.608631 / 1.753930 / 1.755000.

Reading the rows shows the mechanism: `N-LAK33A/B` had the correct B₁, B₂, B₃, C₁
and C₂ and a **wrong C₃** (107.097324 and 101.736644 µm² instead of 80.9379555
and 80.7407701) — a transcription slip in the IR pole.  `N-BAF52` had the correct
B₁ only; B₂, B₃, C₂, C₃ all belong to something else.  I searched all 655
formula-2 rows in the database for a coefficient match and for a curve match: no
verbatim source exists, and the closest *curve* is SCHOTT N-KZFS11 at
max|Δn| = 1.03e-3 over 0.45–1.55 µm (next: OHARA PBM6 at 3.1e-3).  So the
auditor's "looks like a mis-copied N-KZFS11 row" is right about the neighbourhood
but the row is not a copy of any catalogue row — it is simply wrong.

**What I changed.**  Replaced all three rows from the SCHOTT Zemax 2017-01-20b
YAMLs (refractiveindex.info `specs/SCHOTT-optical`, formula 2, whose C_i are
already in µm² and so map 1:1 onto this table's format).  Each row carries a
why-comment naming its source page and the data-sheet n_d / V_d it must
reproduce.

**Verification.**  Two oracles.  (a) The manufacturer data sheet — n_d and V_d
are published numbers, independent of the dispersion coefficients, and encoded a
second time in each glass's 6-digit glass code (609466.305 / 754523.422 /
755523.422).  Measured after the fix: Δn_d = +1.01e-6 / +8.7e-8 / +1.65e-7 and
ΔV_d = −0.003 / +0.001 / +0.000, i.e. at the data sheet's own quote floor.
(b) `repro/orch/verify_rt_glass.py` re-run: bundled and catalogue now agree to
the printed 6 decimals on all five glasses it checks, N-BK7 and N-SF11 included.
`repro/THIN-ELEMENTS-GLASS/p_glass3.py` re-run prints `FLAGGED: []`.

**Residual risk.**  The rows are only as good as the 2017 SCHOTT catalogue; a
future catalogue revision would move them.  The new value check makes that
visible rather than silent.

### E1 (continued) — the value cross-check that closes the class

`_check_glass_registry_consistency` had six checks, all structural: they prove a
row exists and is reachable.  None could see a row that is present, reachable,
evaluates cleanly and holds the wrong glass — which is exactly what happened
twice now (v4.11.2's S-LAH64/S-LAH79, and these three).

I added `_cross_check_bundled_values()`, reached through a new
`_check_glass_registry_consistency(check_values=True)`.  It re-derives n_d and
V_d for every `SELLMEIER_COEFFICIENTS` **and** `POLYNOMIAL_COEFFICIENTS` row and
compares against the refractiveindex.info page the row was sourced from,
resolving that page from the registry tuple where there is one, from the new
`_BUNDLED_ROW_SOURCE` map for the literature-shelf rows (whose page is named
after the author, not the glass), and otherwise by scanning the five
manufacturer books.

**Bars, derived.**  Over the repaired table, 76 rows resolve and the worst
residual is |Δn_d| = 4.18e-6 and |ΔV_d|/V_d = 1.13e-4, both N-LASF40 (its
catalogue page carries a marginally different fit).  46 of 49 Sellmeier rows and
**all 24 polynomial rows are exact to ≤ 2.2e-16**.  The bars (5e-5 and 1e-3) sit
~1 decade above that floor and ~1 decade below the weakest defect they must
catch (N-LAK33A: 2.9e-4 in n_d, 1.2e-2 relative in V_d).

**Not at import.**  The check parses one catalogue YAML per row.  Measured: the
whole sweep is 0.46 s (the package caches the catalogue index), but that has no
place in `import lumenairy`, so `check_values` defaults to False and the
module-level call is unchanged.  `test_e1_import_time_check_stays_structural_only`
pins both facts.

**Fail-before.**  `test_e1_value_gate_rejects_the_pre_fix_rows` injects the exact
pre-fix coefficients through the public table and asserts the gate flags exactly
those three names and raises.  Measured output:

```
N-BAF52:  dn_d = +2.852e-02, tol 5.0e-05; dV_d/V_d = -8.859e-02, tol 1.0e-03
N-LAK33A: dn_d = +3.489e-04, tol 5.0e-05; dV_d/V_d = +1.455e-02, tol 1.0e-03
N-LAK33B: dn_d = +2.942e-04, tol 5.0e-05; dV_d/V_d = +1.223e-02, tol 1.0e-03
```

**Two things the new check surfaced.**

* `BaF2`'s row comment credited "the authoritative Li 1980 fit
  (refractiveindex.info main/BaF2/Li)".  The coefficients are **Malitson & Dodge
  1972**: the row reproduces `main/BaF2/Malitson` to 2.2e-16 across 0.4–1.6 µm and
  differs from `main/BaF2/Li` by 8.1e-5 in n_d.  I corrected the attribution and
  left the data alone (it is a valid, well-known fit; changing it would be a
  silent physics change with no finding behind it).
* `CaF2`'s bundled row is Malitson's fit while `GLASS_REGISTRY['CaF2']`
  dispatches to `main/CaF2/Daimon-20`, so the returned index depends on whether
  the optional package is installed: 1.4338493 vs 1.4338769 at the d-line
  (2.76e-5), 3.78e-5 maximum over 0.4–1.6 µm.  Both are legitimate published CaF₂
  fits and choosing between them is a policy call, so I documented it in the row
  comment and pinned the deviation
  (`test_caf2_bundled_fallback_vs_catalogue_dispatch_deviation_is_bounded`, bar
  1e-4) rather than changing a default.  Every other tuple-registered bundled row
  agrees with its own dispatch page to < 1e-5.  See §6.

### E2 (P1) — `get_glass_index_complex` raised instead of the documented κ = 0

**Reproduced on HEAD** with `repro/THIN-ELEMENTS-GLASS/p_glass2.py`: 7 of 44
tuple-registered glasses raised at 1.31 µm — `CaF2`, `FUSED_SILICA`, `F_SILICA`,
`MgF2`, `SILICA`, `SILICON`, `SiO2`.  `NoExtinctionCoefficient`'s MRO is
`(NoExtinctionCoefficient, Exception, BaseException, object)`, so it is not in
the caught tuple; `_warn_missing_kappa_once` was unreachable on that path.

**Changed.**  `_missing_kappa_exceptions()` resolves the package's class lazily
from the loaded module (falling back to the top-level export, then to nothing)
and appends it to the built-in tuple, so the package stays optional per
CONVENTIONS §10.  Re-run of `p_glass2.py`: **0 of the tuple glasses raise**.

**A second arm of the same contract, found while verifying.**  The package's
interpolator returns **NaN** rather than raising when the requested wavelength is
outside a page's tabulated-k range, so `get_glass_index_complex('E-BAK1', 1.31e-6)`
returned `1.5560943 + nan·j` and `E-LAK04` likewise — silently poisoning any
`exp(−2πκt/λ)` downstream.  A non-finite κ now takes the same warn-once + κ = 0
path.  Pinned by `test_e2_extinction_is_finite_and_non_negative`, which sweeps
every tuple glass at two wavelengths.

**Guard against "fix by swallowing".**  `test_e2_pages_that_do_carry_k_keep_their_value_and_sign`
holds N-BK7 at 1.55 µm to 1.5006520 + 1.4361e-7 j (the audit-verified value and
sign convention) to 1e-4 relative.

**Residual risk.**  The catch is still a fixed tuple plus one resolved class; a
future package version could invent another signal.  The NaN arm now covers the
most likely one (silent bad data rather than an exception).

### E3 (P1) — the spurious √2 in `generate_turbulence_screen`

**Reproduced on HEAD**: `repro/orch/verify_turb.py` → variance / ΣPSD·df² =
**1.976** (40 seeds, N = 256); `repro/THIN-ELEMENTS-GLASS/p_turb.py` →
D_meas/D_lattice = 1.987 / 1.984 / 1.979 at r = 5 / 10 / 20 mm.

**Changed.**  `amplitude = np.sqrt(psd) * df`.  The derivation is now in the
Notes: with `c_k = (a_k + i b_k)·A_k` and independent standard normals,
`Var(Re c_k) = A_k²` already, so the real part costs no factor of two.  The PSD
expression moved into a shared `_turbulence_psd` helper so the FFT grid and the
new subharmonic grids cannot drift apart.

**Verified.**  After: `verify_turb.py` → **0.988**; `p_turb.py` →
D_meas/D_lattice 0.9937 / 0.9922 / 0.9896 at the same three separations, i.e.
→ 1.000 as r → 0 exactly as the WP asked.  Against the continuum target the
screen went from **1.584 (above Kolmogorov — unphysical for a band-limited
screen)** to 0.798 at r = 0.05 r0.  The change is exactly a √2 on the same RNG
stream: `max|new·√2 − old| / max|old| = 5.5e-16`.

**Subharmonics.**  Added `subharmonics=p` (default 0, so the shipped screen is
bit-identical apart from the amplitude).  Each level is a 3×3 grid at spacing
`1/(3^p·N·dx)` with its DC killed, summed in the separable `e.T @ cn @ e` form
(3 length-N exponentials per level instead of 9 full-grid ones); verified equal
to the literal double loop to 1.8e-15 relative.  D/D_Kolmogorov over 40 seeds
(N = 512): 0.798 → **0.880** at r = 0.05 r0 and 0.463 → **0.777** at 3.2 r0, for
+23–28 % wall time (N = 512/1024/2048) and no extra peak memory.  I left the
default off because it is a new model component, not a defect fix, and because
"off" keeps the entry point bit-identical to what the amplitude fix alone gives.

**Test that pinned the defect.**  `tests/unit/test_niche_audit_w3_elements.py::_turbulence_reference`
— the E-L11 test's "independent" oracle carried the same spurious √2, so it would
have failed the corrected code.  Corrected in place (one line) with a why-comment;
the test's actual purpose (integer DC anchor, bit-identity for even N) is
untouched and still passes for N = 64/65/128/129/257.

**Residual risk.**  A default change.  Anyone who calibrated a run by eye against
the old screen now gets weaker turbulence; the migration note in the changelog
gives the exact equivalent (`r0 · 2^(−3/5)`).

### E5 (P1) — `apply_grin_lens` at the quarter pitch

**Reproduced**: `repro/THIN-ELEMENTS-GLASS/p_thinlens.py` shows
f_code/f_exact = sin(gd)/(gd) = 0.9996 / 0.9851 / 0.9003 / **0.6366** / 0.1093 at
g·d = 0.05 / 0.3 / π/4 / π/2 / 0.9π, and the ASM peak-intensity plane tracked
f_code.

**Changed.**  The screen now carries the rod's exact paraxial power
`n0·g·sin(g·d)` (the ABCD `−C`).  `thin_form=True` restores the short-rod form
bit-identically for anyone who needs the old numbers and warns above g·d = 0.2,
quoting the measured `sin(gd)/(gd)` factor.  On the exact path, g·d past the
quarter pitch warns that a single screen carries power only — that power passes
through zero at the half pitch where the rod reimages 1:1 inverted.  The Notes
now also state the reference-plane limitation explicitly (the rod's BFL is
`cos(gd)/(n0 g sin(gd))` from the exit face, a thin screen focuses at `f` past
itself, and the two differ by `(1−cos gd)/(n0 g sin gd)`).

**Verified.**  (a) The screen's quadratic coefficient recovered from the phase
matches `n0 g sin(gd)` to < 1e-6 relative at all four pitches.  (b) An
independent ASM focus scan (n0 = 1.6, g = 30 /m, quarter pitch, w0 = 0.5 mm,
NA 0.024, N = 512, dx = 6 µm) puts the peak at 0.994 × f_exact — the residual is
the paraxial screen's own spherical-aberration shift, the same class as
`apply_thin_lens(paraxial)`'s −40 µm at f = 20 mm.  Re-measured at higher NA
(g = 300 /m): focus/f_exact = 1.0000 / 0.9955 / 0.9940 / 0.9940 at
g·d = 0.05 / 0.3 / π/4 / π/2 — all tracking f_exact, none tracking f_thin.

Note that `p_thinlens.py`'s own quarter-pitch line now reads 1.7242 mm: its scan
window is `linspace(0.8·f_code, 1.3·f_code)` = 1.06–1.72 mm, anchored on the
*pre-fix* focal length, so the true focus (2.083 mm) is outside the window and
the reported number is the window's top edge.  That is the fix showing through a
fixture that was built around the defect, not a residual error; my own scan uses
a window around f_exact.

**Residual risk.**  A default change, documented with a `thin_form` escape hatch.
CuPy path desk-checked: `power` is a Python float and `r_sq` stays `xp`, so the
CuPy branch is unaffected; `np.sin` of a Python float is host arithmetic.

### E6 (P2) — five items

1. **Harvey-Shack TIS.**  Added the exact closed form
   `π b0 l²[(1+1/l²)^(1−s/2) − 1]/(1 − s/2)` (`π b0 l² ln(1+1/l²)` at s = 2),
   derived by the substitution `t = 1 + (u/l)²` under `u = sin θ`.  Verified
   against a 200 001-point geometric-grid trapezoid integral built in the test
   (its own error floor ~1e-8): agreement ≤ 1.1e-7 across l = 1e-1…1e-4 and
   s = 1.5/2/2.5, and against the published s = 2 form to 1e-12.  Re-run of
   `p_bsdf2.py`: every Harvey-Shack row now reads **−0.00 %** against its dense
   reference (was −0.72 % at the defaults, −18.33 % at l = 1e-3).

   I also rebuilt the **base-class** quadrature, which is what a user subclass
   inherits: the variable is `u = sin θ`, the cells are geometric in `ln u` from
   1e-7 to 1 with two-point Gauss-Legendre inside each, and the `u < 1e-7` disc
   is added analytically.  Same node count (256 × 128).
   Measured worst relative error across Lambertian, Gaussian (σ = 1e-2 and 0.3)
   and Harvey-Shack (l = 1e-1…1e-4, s = 1.5/2/2.5): **1.4e-6**, typical 1e-9,
   against −32 % / −18 % / −0.7 % for the linear-θ grid it replaces.  All three
   shipped models override TIS, so no library number moves except Harvey-Shack's.

   **Correction (VERIFY-A8, 2026-09-12).**  This paragraph originally claimed
   "the weights sum to `∫u du` exactly (so a flat lobe integrates with zero
   quadrature error)".  That is false.  Two-point Gauss-Legendre is exact
   through cubic order in `v = ln u`, but the flat-lobe integrand is
   `u² = e^{2v}`, not a cubic, so the textbook quartic residual
   `dv⁴·2⁴/4320` survives: with `dv = ln(1e7)/128 = 0.1259232` that predicts
   **9.312e-07**, and the measurement is `Σw/(1/2) − 1 = −9.2935e-07` — the
   two agree to 0.2 %.  The 1.4e-6 above is also the worst of *those* lobes,
   not a bound: an independent sweep measured **1.3e-5** on `B = 1 − u²` and
   6.4e-6 at l = 1e-6.  Budget ~1e-5 for a broad smooth lobe.  Nothing about
   the fix changes — it is still four to five decades better than the grid it
   replaces — only the claim.  The source docstring and
   `test_e6_base_class_quadrature_is_exact_for_a_flat_lobe` now carry the
   derivation and the measurement.

2. **`make_bsdf` key validation.**  Unknown keys now raise, naming the accepted
   set; `A`/`B`/`C` are accepted as aliases for `b0`/`l`/`s`; both spellings of
   one parameter raises.  The audit's two examples are now errors instead of
   silent 10×-wide / 50×-weak lobes and all-defaults.  No in-library caller
   passes extra keys (grepped `lumenairy/`, `tests/`, `validation/`).

3. **Klein-Cook guard.**  `_warn_thin_grating_validity` mirrors
   `emt._warn_rytov_validity` (same shape, same stacklevel, same "hand off to
   rcwa/pmm" wording).  Fires on `Q = 2πλd/(n̄Λ²) > 1`, else on `Λ < 10λ`.  n̄ is
   the duty-weighted mean of the real parts of the two indices (the modulated
   layer's mean index).  Verified: silent at Q = 0.13, one warning at Q = 12.6 and
   Q = 50.3; ΣT = 1.0000 in all three, which is the point — energy closure cannot
   be the validity check here.  The analytic Fourier coefficients are unchanged
   and still exactly 4/π² and 4/(9π²).  The module docstring's garbled `t_m`
   formula is replaced with the correct expression.

4. **MLA separable phase.**  The snap, the local coordinate and the footprint
   test are length-N vectors; only `r_sq` is a full grid, and `cos`/`sin` write
   directly into the output's real/imaginary views instead of going through
   `np.exp(1j·phase)`.  Peak tracemalloc **14.13 → 3.25** float64 grids (identical
   at N = 1024 and N = 2048), median wall 426 → 144 ms at N = 2048, output
   **bit-identical** (`np.array_equal` over three N / pitch / lenslet-count
   combinations).  The audit's verified-correct properties (|T| = 1, exactly zero
   steer at every lenslet centre, fractional pitch) re-measured and unchanged.

   **Correction (VERIFY-A8, 2026-09-12).**  The pre-fix peak is quoted here as
   14.13 grids, not the 10.13 this report first carried: measured against a
   verbatim transcription of the lines the diff removes, **one implementation
   per process** so neither is measured in the other's allocator wake, and
   stable at both N = 1024 and N = 2048.  The post-fix 3.25 reproduces exactly
   either way, so the improvement is **4.35×**, not 3.1×.

5. **`sample_scatter_rays` vectorisation.**  Split each model's lobe-local draw
   into `_sample_local(n, rng)` (incidence-independent by construction) and moved
   the frame build into one batched `_rotate_local_to_specular`, which
   reproduces the per-ray code it replaces on four incidences including
   the |spec_z| ≥ 0.999 pole branch.

   **Correction (VERIFY-A8, 2026-09-12).**  This said **bit-identical**; it is
   not, in general.  Swept over 407 incidences (400 uniform random plus 7 at
   and around the 0.999 threshold), the batched and per-ray forms agree
   exactly on **350** and differ by at most **4.441e-16** — 2 ULP of a unit
   direction cosine — on the other 57, because `np.linalg.norm(v)` on a 1-D
   vector can dispatch to a BLAS `nrm2` while `np.linalg.norm(v, axis=-1)` is
   a ufunc reduction.  The four fixtures quoted happen to fall in the exact
   set, which is the per-build knife edge TESTING_STANDARDS S4 warns about;
   the test now asserts a derived `8·eps` bar instead of `np.array_equal`.

   `sample_scatter_rays` then draws the whole
   bundle in one call: 20 000 rays measured at **113× / 327× / 377×**.
   Harvey-Shack's rejection sampler is replaced by its exact inverse CDF
   (derived in the same substitution as the TIS closed form), which also makes
   the draw a fixed-length RNG call: 24.93 → 3.58 ms for 20 k at l = 1e-3, and
   the rejection loop and its Python list are gone.  Distribution verified
   against a numerically-integrated CDF (KS = 0.0020 at n = 2e5, 95 % band
   0.0030) and against the per-ray loop's own sample mean at 6σ.

   Behaviour note: seeded `sample()` / `sample_scatter_rays` output differs from
   before (different algorithm, different per-ray ordering).  Distributions are
   unchanged; the changelog carries the migration note.

### E7 (P3) — the remainder

* **Noll pointer** corrected; the docstring now states that the library ships no
  Noll converter and that `zernike_index_to_nm` is OSA, with the j = 5
  counter-example.  Pinned, including a live assertion that
  `zernike_index_to_nm(5) == (2, +2)`.
* **`'air'`/`'vacuum'`/`'__MIRROR__'`** — the exemptions comment now describes
  what these names actually are (none of the three is a registry entry), and the
  `'air'` short-circuit consults `GLASS_REGISTRY['air']` before returning 1.0, so
  the Edlén model the comment described is implementable.  Verified: default
  1.0 for `'air'`/`'AIR'`/`'Air'` unchanged, a registered Edlén callable returns
  1.000273988 (274 µm/m of OPD), `list_glasses()` unchanged at 77 names,
  `'vacuum'`/`'MIRROR'`/`'__MIRROR__'` still raise.  I deliberately did **not**
  register real `'air'`/`'vacuum'` entries: that would put them in
  `list_glasses()`, which feeds `analysis/plotting.py::plot_glass_map` and
  `ui/glass_map_dock.py` (files I do not own), where V_d = 0/0 on an n ≡ 1
  material.  See §5.
* **Non-square DOE cell** — named `ValueError` for both orientations, with the
  reason; square behaviour re-measured unchanged (uniform per-cell-pixel
  occupancy `[32]*8`, 0.0000 % power off the order lattice).
* **Grey-pixel aperture** — `edge='gray'` / `edge_samples`.  Area error at
  D/dx = 50 px, **all three readings on one grid** (N = 1024, dx = 1 µm,
  D = 50 µm; VERIFY-A8 re-measurement 2026-09-12):
  **−0.5345 % → +0.0384 % (4×4) → −0.00056 % (16×16)**.  The −0.942 % this
  report first paired with them is the audit's own hard-edge reading on a
  different grid — the hard-edge error depends on where the rim falls on the
  pixel lattice, so a before/after pair must come from one fixture.  Over the
  12 sub-pixel rim placements the test sweeps, the rms is 0.386 % hard →
  0.044 % gray(4).  Default `'hard'` bit-identical on all three shapes,
  dtype-preserving, JAX parity exact (0.00e+00).
* **Reference-plane difference** — re-measured with `p_real.py` and written into
  `apply_spherical_lens`'s See Also block with the numbers.

**`surface_sag_general(R=0)`** is the one E7 item I did not implement: the code
is in `elements/lenses.py`, which WP-A2 owns.  Reproduced on HEAD and requested
in §5.

## 3. Files touched

Library (all within my ownership):

* `lumenairy/glass.py` — E1 rows + value cross-check, E2 exception widening +
  NaN-κ arm, E7 air/vacuum/mirror comment and the `'air'` registry consult,
  BaF2/CaF2/MgF2 provenance comments.
* `lumenairy/elements/elements.py` — E3 amplitude + `_turbulence_psd` +
  `_turbulence_subharmonics` + the `subharmonics` kwarg; E7 grey-pixel
  `apply_aperture`; E7 `zernike` Noll docstring.
* `lumenairy/elements/_lens_thin.py` — E5 `apply_grin_lens` exact power,
  `thin_form` kwarg and the two guards; E7 `apply_spherical_lens` See Also.
* `lumenairy/elements/bsdf.py` — E6 TIS closed form + base quadrature,
  `_sample_local` split, `_rotate_local_to_specular`, `make_bsdf` validation and
  aliases, vectorised `sample_scatter_rays`, HS inverse-CDF sampler.
* `lumenairy/elements/thin_grating.py` — E6 Klein-Cook / short-period guard,
  P3 module-docstring formula.
* `lumenairy/elements/doe.py` — E6 separable MLA, E7 square-cell guard.

Tests:

* **new** `tests/unit/test_audit2609_a8_glass.py` (28 tests)
* **new** `tests/unit/test_audit2609_a8_turbulence.py` (18 tests)
* **new** `tests/unit/test_audit2609_a8_thin_elements.py` (61 tests)
* **edited** `tests/unit/test_niche_audit_w3_elements.py` — one line in
  `_turbulence_reference` (it pinned the √2 defect in its own oracle), plus a
  why-comment.  This file is shared with WP-A2/A3/A4 material (`_lens_real`,
  `_lens_traced*`, `lenses_gbd`); I touched only the E-L11 turbulence helper.

No other file was modified.  No git write command was run.

## 4. Tests run

| Command | Result | Duration |
|---|---|---|
| `python -m pytest tests/unit/test_audit2609_a8_glass.py -q ...` | 28 passed | 0.9 s |
| `python -m pytest tests/unit/test_audit2609_a8_turbulence.py -q ...` | 18 passed | 1.6 s |
| `python -m pytest tests/unit/test_audit2609_a8_thin_elements.py -q ...` | 61 passed | 10.9 s |
| `python -m pytest tests/unit/test_v4_16_0_agent_d_glass_catalogues.py test_audit_p1_glass_registration.py test_audit_w4_glass_registry_meshgrid.py test_v5_2_glass_formula3.py test_v5_6_glass_memo.py test_v5_4_6_wave3_coatings_glass.py -q ...` | 171 passed, 2 skipped | 4.2 s |
| `python -m pytest tests/unit/test_niche_audit_w3_elements.py test_audit_w5_elements_misc.py test_audit_w6_analysis_elements.py test_v4_15_3_agent_a.py -q ...` | 117 passed | 9.4 s |
| `python -m pytest tests/unit/test_v4_15_agent_e.py test_v4_15_1_agent_e.py test_v5_21_2_subsystem_audits.py test_v5_4_6_wave5_delegated.py test_audit_w6_raytrace.py -q ...` | 120 passed, 20 skipped | 17.5 s |
| `python -m pytest tests/unit/test_doe_rcwa.py test_niche_audit_w9_hfpi_doe.py test_v5_4_zernike_normalization_weighting.py test_v4_16_0_walker_xp_of_dispatch.py test_niche_audit_w4_input_kind.py test_elements_lens.py -q ...` | **2 failed**, 332 passed | 12.8 s |
| `python -m pytest tests/unit/test_thin_lens_audit_2026_07_18.py test_niche_audit_ec_thin_lens_claims.py test_perf_v4_12_0_zernike_cache.py -q ...` | 28 passed | 14.3 s |
| `python -m pytest tests/unit/test_audit_misc.py -q ...` | **1 failed**, 224 passed, 3 skipped | 95.7 s |
| consolidated re-run of all 22 files above (minus `test_niche_audit_w4_input_kind.py`) | **1 failed**, 599 passed, 3 skipped | 477.6 s |
| `python validation/run_all.py test_elements test_doe test_features test_ao test_analysis --quiet` | ALL 5 files passed | 28.9 s |

### Pre-existing failures — none related to WP-A8

1. `tests/unit/test_niche_audit_w4_input_kind.py::test_all_sixty_eight_sites_are_wired`
   and `::test_wired_site_declares_expected_input_kind[beam_stats.py::beam_d4sigma(E)->field]`
   — the `_check_2d_scalar_field` census expects 69 call sites and finds 70.
   `git diff` shows another WP added the guard to
   `lumenairy/analysis/beam_stats.py::beam_d4sigma`; my diff adds **zero**
   `_check_2d_scalar_field` call sites (verified with
   `git diff -- <my files> | grep -c '^+.*_check_2d_scalar_field'` → 0).
   Whoever owns `beam_stats.py` must bump the count and `_WIRED_SITES`.
2. `tests/unit/test_audit_misc.py::TestAuditFixesV4_12_1_coverage_StopIndexWarn::test_traced_emits_warning_for_stop_index_2`
   — `apply_real_lens_traced` now raises `ValueError` where the test expects a
   warning, from a new message in `_lens_real.py` ("An out-of-range stop matches
   no surface AND suppresses the entrance aperture").  `_lens_real.py` and
   `lenses.py` are both `M` in `git status` and the message is present in the
   stale `.pyc` but not in the current `.py`, i.e. WP-A2 is mid-edit.  Not mine.
3. `tests/unit/test_audit_glass.py::TestAuditFixesV4_11_2_track_a_SeidelCorrectionSignAgainstGroundTruth::test_seidel_does_not_move_the_focus`
   — "seidel_correction=True moved best focus from 50.6656 mm to 50.9215 mm
   (+0.51 %); a rho**2 term is back in the fit."  This is finding L1 / F-O1 in
   `_lens_real.py`, and the WP file says WP-A2 is rewriting a test in exactly
   this file.  Not mine; I did not edit `test_audit_glass.py`.

I also hit one transient `SyntaxError` in `lumenairy/_cache_registry.py` while
another agent was mid-write; it recompiled cleanly on the next attempt.  And
`p_misc.py`'s last section now ends `apply_real_lens: KeyError 'thicknesses'`,
which is another in-flight `_lens_real.py` change, not a regression from this WP.

### Fail-before demonstrations

Each behavioural fix was reverted **in-process** with a pytest plugin (kept in the
scratchpad, not in the repo) and the new files re-run:

* pre-fix glass (rows re-injected + narrow exception tuple): **16 of 28** glass
  pins fail — the three data-sheet rows, the three not-pre-fix arms, the table
  gate, the value gate, both no-raise sweeps, both NaN sweeps and all four
  warn-once arms.
* pre-fix turbulence (`× √2` shim): **11 of 18** turbulence pins fail — all four
  Schmidt-form parities, the √2 relation, all three lattice ratios, the
  Kolmogorov-side test and the subharmonic comparison.
* pre-fix thin elements (short-rod GRIN, HS TIS override removed + linear-θ base
  grid, key validation off, Klein-Cook guard off, full-grid MLA): **24 of 61**
  pins fail, covering every behavioural item.

The remaining new pins (alias acceptance, the square-cell guard, `edge='gray'`,
the four docstring contracts, the no-rejection-loop check) fail pre-fix trivially
because the kwarg / guard / text did not exist.

## 5. Requested changes outside my ownership

1. **`lumenairy/elements/lenses.py` (WP-A2) — `surface_sag_general(R=0)` should
   raise.**  Reproduced on HEAD:

   ```
   surface_sag_general(np.array([0.0, 1e-6]), R=0.0, conic=0.0) -> [nan nan]
     RuntimeWarning: divide by zero encountered in divide   (lenses.py:241)
     RuntimeWarning: invalid value encountered in divide     (lenses.py:241)
     RuntimeWarning: divide by zero encountered in divide    (lenses.py:249)
     RuntimeWarning: invalid value encountered in divide     (lenses.py:249)
   R=inf  -> [0. 0.]        (correct)
   R=None -> [0. 0.]        (correct)
   ```

   Requested change: at the top of `surface_sag_general`, after the `R is None` /
   `isinf(R)` handling, add

   ```python
   if R == 0:
       raise ValueError(
           f"surface_sag_general: R = 0 is not a surface (curvature 1/R is "
           f"infinite).  Use R = np.inf or R = None for a flat surface.")
   ```

   This is the same class the v5.32 / audit W5-2 change fixed for
   `apply_thin_lens(f=0)` and `apply_cylindrical_lens(f=0)`; the sag entry point
   was not in that sweep.  I did not implement it because `lenses.py` is WP-A2's
   and is currently modified in the working tree.

2. **`lumenairy/analysis/beam_stats.py` owner — bump the input-kind census.**
   Adding `_check_2d_scalar_field` to `beam_d4sigma` broke
   `tests/unit/test_niche_audit_w4_input_kind.py` (69 → 70 sites, and the new
   site is not in `_WIRED_SITES`).  Both need updating in the same change, per
   that test's own message.

3. **`lumenairy/propagators/system.py` (optional, low priority).**  The
   `'turbulence'` element type calls `generate_turbulence_screen` and could pass
   the new `subharmonics` through:
   `subharmonics=elem.get('subharmonics', 0)` at `system.py:948`.  Default 0
   keeps behaviour identical, so nothing breaks without it.

4. **`lumenairy/analysis/plotting.py` / `lumenairy/ui/glass_map_dock.py`
   (informational).**  If anyone later registers real `'air'` / `'vacuum'`
   entries in `GLASS_REGISTRY`, `plot_glass_map` will try to compute
   `V_d = (n_d − 1)/(n_F − n_C) = 0/0` for them (`glass_map_dock` already skips
   `'air'` but not `'vacuum'`).  I kept them unregistered partly for this reason.

## 6. Deferred, with designs

1. **The CaF₂ bundled-fallback vs catalogue-dispatch split (2.76e-5 in n_d).**
   Not an audit finding; surfaced by the new value check.  The library returns
   Malitson's CaF₂ on a minimal install and Daimon-20 with
   `lumenairy[glass]`.  Design: pick one — either re-transcribe the bundled row
   from `main/CaF2/Daimon-20` (formula-2 conversion is mechanical; ~1 h including
   a validity-range update and a changelog migration note), or point
   `GLASS_REGISTRY['CaF2']` at `main/CaF2/Malitson`.  Either is a user-visible
   index change of 2.8e-5 and wants a deliberate decision, so I documented and
   pinned the gap instead.  MgF₂, SiO₂ and the silica aliases have no such split
   (agreement ≤ 4.1e-9).
2. **`thin_grating` warning volume in wavelength sweeps.**
   `grating_efficiency_vs_wavelength` loops over wavelengths, and Q ∝ λ, so a
   sweep that crosses the threshold emits one warning per wavelength (Python's
   default filter will not dedupe them because the message carries the value).
   I matched `emt._warn_rytov_validity`'s policy, which the audit named as the
   model to mirror.  Design if it becomes noisy: a `_warned` set keyed on
   `(round(log10 Q, 1), period, depth)` as `glass._validity_warned` does; ~15
   lines.
3. **Base-class TIS inner truncation.**  The `u < 1e-7` disc is completed
   analytically assuming a flat lobe there.  A subclass with structure below
   `sin θ = 1e-7` (57 µrad) would be mis-integrated.  Measured contribution of
   that disc for the shipped models: ~π·B(0)·1e-14.  Design if ever needed: make
   `_TIS_U_MIN` a class attribute so a subclass can lower it; ~5 lines.
4. **`makedammann2d` / `create_kinoform` IFTA suspicions** (the partition
   report's "Unverified suspicions") are not in my finding list and I did not
   run 3000 IFTA iterations.  The two flagged sites
   (`doe.py:922`'s divide by a far field that can contain exact zeros, and the
   `fftshift`/`ifftshift` asymmetry) are still open.
5. **numba `fastmath` on `_aspheric_sag_accum_numba`** — `lenses.py`, WP-A2's.

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A8_CHANGELOG.md`
