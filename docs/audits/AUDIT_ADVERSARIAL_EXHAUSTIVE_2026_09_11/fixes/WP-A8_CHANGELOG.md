# WP-A8 changelog text — thin elements, DOEs, materials (audit 2026-09-11, §5 rows E1–E3, E5–E7)

### Fixed -- glass: three bundled Sellmeier rows held a DIFFERENT GLASS, and they were the only dispatch path for those names (E1, P0)

`N-BAF52`, `N-LAK33A` and `N-LAK33B` are registered `'__sellmeier__'`, and that
branch fires **before** any refractiveindex.info lookup — so the bundled row is
what every install returned, package present or not.  Measured against the
refractiveindex.info SCHOTT-optical catalogue (database commit `a66ef88`) and
against SCHOTT's own published data sheet:

| glass | n_d before | n_d after | data sheet | V_d before | V_d after | data sheet |
|---|---|---|---|---|---|---|
| N-BAF52 | 1.637147 | **1.608631** | 1.60863 | 42.469 | **46.597** | 46.60 |
| N-LAK33A | 1.754279 | **1.753930** | 1.75393 | 53.031 | **52.271** | 52.27 |
| N-LAK33B | 1.755294 | **1.755000** | 1.75500 | 52.940 | **52.300** | 52.30 |

The `N-BAF52` row was off by **2.85e-2 in index (1.8 %)** — a ~1.9 % focal-length
error on any singlet and a wholly wrong Abbe number for any achromat — with the
nearest catalogue curve being N-KZFS11 (1.63775 / 42.41), i.e. the same
mis-copied-neighbouring-row failure the v4.11.2 sweep fixed for S-LAH64 /
S-LAH79 without re-checking the rest of the table.  `N-LAK33A/B` carried the
wrong third Sellmeier pole (C3 = 107.10 / 101.74 µm² instead of 80.938 /
80.741), worth 2.8e-3 / 2.3e-3 of index across 0.4–1.6 µm.  All three rows are
replaced from the SCHOTT Zemax 2017-01-20b catalogue and now reproduce the data
sheet's n_d to 1.0e-6 and V_d to 0.003.  `N-BK7` and `N-SF11` are unchanged and
still exact.

Files: `lumenairy/glass.py:125-145`.
Tests: `tests/unit/test_audit2609_a8_glass.py::test_e1_bundled_row_reproduces_the_manufacturer_data_sheet`,
`::test_e1_repaired_rows_are_not_the_pre_fix_rows`.

### Added -- glass: a VALUE cross-check over every bundled dispersion row, so this class cannot recur (E1)

`_check_glass_registry_consistency` proved a row *exists* and is *reachable*; it
could not see a row that is present, reachable, evaluates cleanly and holds the
wrong glass.  It gains an opt-in seventh check,
`_check_glass_registry_consistency(check_values=True)`, delegating to the new
`_cross_check_bundled_values()`: every `SELLMEIER_COEFFICIENTS` **and**
`POLYNOMIAL_COEFFICIENTS` row is re-derived at the d/F/C lines and compared to
the refractiveindex.info page it was sourced from, to 5e-5 in n_d and 1e-3
relative in V_d.  Provenance for the literature-shelf rows (whose catalogue page
is named after the author, not the glass) is recorded in the new
`_BUNDLED_ROW_SOURCE` map.

Bars derived from the measured residual of the whole repaired table (76 rows
resolved, 0.46 s): worst |Δn_d| 4.18e-6 and worst |ΔV_d|/V_d 1.13e-4 (both
N-LASF40), with 46 of 49 Sellmeier rows and **all 24 formula-3 polynomial rows
exact to ≤ 2.2e-16** — so each bar sits ~1 decade above the honest-row floor and
~1 decade below the weakest real defect (N-LAK33A's 2.9e-4 / 1.2e-2).  The check
is **not** run at import (it parses one catalogue YAML per row); the test suite
runs it.  Injecting the pre-fix coefficients makes it flag exactly those three
rows and raise.

Files: `lumenairy/glass.py:1031-1279` (new constants, `_bundled_row_catalogue_source`,
`_catalogue_index_fn_from_entry`, `_catalogue_index_fn`, `_nd_vd`,
`_cross_check_bundled_values`), `:1255` (`check_values` parameter).
Tests: `tests/unit/test_audit2609_a8_glass.py::test_e1_whole_bundled_table_agrees_with_refractiveindex_info`,
`::test_e1_value_gate_rejects_the_pre_fix_rows`,
`::test_e1_import_time_check_stays_structural_only`.

### Fixed -- glass: `get_glass_index_complex` RAISED instead of the documented κ = 0 fallback, for exactly the common bulk materials (E2, P1)

With the recommended `pip install lumenairy[glass]` extra installed, a catalogue
page carrying no k-table raises `refractiveindex.refractiveindex.NoExtinctionCoefficient`,
which subclasses `Exception` **directly** and so was not in the caught tuple
`(AttributeError, NotImplementedError, KeyError, ValueError, TypeError)`.
Measured over all tuple-registered glasses at 1.31 µm: **7 raised** — `CaF2`,
`FUSED_SILICA`, `F_SILICA`, `MgF2`, `SILICA`, `SILICON`, `SiO2`, i.e. every
`main`-shelf window material — now **0**.  The class is resolved from the
installed package rather than imported (CONVENTIONS §10) by the new
`_missing_kappa_exceptions()`.

The same sweep found a second silent-wrong arm of the same contract: a page whose
tabulated k does not span the requested wavelength **interpolates to NaN** rather
than raising, so `get_glass_index_complex('E-BAK1', 1.31e-6)` returned
`1.5560943 + nan·j` and poisoned every downstream absorption product.  A
non-finite κ now takes the same warn-once + κ = 0 path as a missing one.
`N-BK7`'s real extinction and its sign convention are unchanged
(1.5006520 + 1.4361e-7 j at 1.55 µm).

Files: `lumenairy/glass.py:1959-2009`.
Tests: `tests/unit/test_audit2609_a8_glass.py::test_e2_complex_index_never_raises_for_a_catalogue_glass`,
`::test_e2_extinction_is_finite_and_non_negative`,
`::test_e2_missing_kappa_warns_once_and_returns_zero`,
`::test_e2_pages_that_do_carry_k_keep_their_value_and_sign`.

### Changed -- glass: a catalogue lookup outside the page's data range now RAISES instead of returning NaN (E2, second arm) — **default change**

The κ arm above was half the contract.  A refractiveindex.info page whose data
does not span the requested wavelength interpolates the REAL index to NaN as
well, and that one was never guarded: `get_glass_index('SILICON', 633e-9)`
returned `nan` and `get_glass_index_complex` returned `nan + 0j`, with only a
validity *warning* to show for it — measured at 5 of 8 wavelengths swept
(0.35 / 0.4 / 0.5876 / 0.633 / 1.064 µm; the `main/Si/Li-293K` page starts at
1.2 µm).  One multiplication later the NaN is across the whole field.

It is now a named `ValueError` (CONVENTIONS §2 prefix) quoting the page's own
wavelength range and the offending wavelength, e.g.

```
get_glass_index: the refractiveindex.info page for 'SILICON' has no data at
6.3300e-07 m; its wavelength range is [1.200e-06, 1.400e-05] m. ...
```

Only the live-catalogue (tuple-registered) path can produce it; the bundled
Sellmeier and polynomial evaluators are closed forms with their own guards and
are untouched.  **Migration.**  A caller that relied on the NaN as an
in-band "no data" signal must catch `ValueError` (or test the range first);
in-range lookups are bit-identical (SILICON 3.5003 at 1.31 µm, 3.4757 at
1.55 µm, 3.4150 at 10 µm) and every other catalogue glass is unaffected.

Files: `lumenairy/glass.py` (`_catalogue_page_wavelength_range_m`,
`_require_finite_catalogue_index`, and the tuple-path return of
`get_glass_index`).
Tests: `tests/unit/test_audit2609_a8_verify.py::test_verify_a8_e2_out_of_range_catalogue_lookup_refuses_instead_of_nan`.

### Changed -- glass: the two new catalogue-lookup guards catch named exception tuples, not bare `Exception` (E1)

`_catalogue_index_fn_from_entry` and `_cross_check_bundled_values` shipped with
`except Exception`, which took the non-`ui/` broad-except census from 48 to 50
against a budget of 48 (`tests/unit/test_audit_except_budget.py`).  Both are
narrowed to `_catalogue_lookup_exceptions()` — an explicit built-in tuple plus
the optional package's own exception classes resolved from the installed
module, exactly as `_missing_kappa_exceptions()` already does (CONVENTIONS §10,
no hard dependency).  The one case that genuinely could not be named — the
package raising a BARE `Exception` from `get_refractive_index` for a page that
carries only tabulated k — is now TESTED for (`_n_func is None`) rather than
caught, with the forced evaluation kept as the fallback if a future version
renames that attribute.  Coverage is unchanged: 76 of 76 bundled rows still
resolve, 0 problems, same residuals; census contribution back to **0**.

Files: `lumenairy/glass.py` (`_CATALOGUE_LOOKUP_BUILTIN_EXCEPTIONS`,
`_NoSuchAttributeSentinel`, `_catalogue_lookup_exceptions`,
`_catalogue_index_fn_from_entry`, `_cross_check_bundled_values`).

### Changed -- elements: `generate_turbulence_screen` delivered 2× the requested phase variance; the spurious `sqrt(2)` is gone (E3, P1) — **default change**

The amplitude was `sqrt(2 * PSD) * df`.  With `c_k = (a_k + i b_k)·A_k` and `a`,
`b` independent standard normals, `Var(Re c_k) = A_k²` already — taking the real
part costs no factor of two, so the `sqrt(2)` doubled the screen's variance and
its structure function.  The correct amplitude is `sqrt(PSD) * df` (Schmidt
2010, `ft_phase_screen`).

Measured (N = 512, dx = 5 mm, r0 = 0.1 m, 60 seeds) against the exact discrete
structure function of the code's own lattice:

| r [m] | D_meas/D_lattice before | after |
|---|---|---|
| 0.005 | 1.987 | **0.994** |
| 0.010 | 1.984 | **0.992** |
| 0.020 | 1.979 | **0.990** |

and the screen variance / ΣPSD·df² over 40 seeds at N = 256 went **1.976 → 0.988**
(orchestrator repro `verify_turb.py`).  Against the continuum Kolmogorov target
`D = 6.88 (r/r0)^(5/3)` the screen read **1.584 at r = 0.05 r0 — above the
continuum**, which is unphysical for a band-limited screen; it now reads 0.798,
below it, as an FFT screen must be.

**Migration.**  Every screen from this entry point is now smaller by exactly
`sqrt(2)` on the same seed (verified: `new * sqrt(2) == old` to 5.5e-16
relative).  Runs that were tuned by eye against the old screen were running
1.5× stronger turbulence than requested at small separations (effective
`r0 = r0/2^{3/5} = r0/1.516`) and a φ_rms `sqrt(2)` too large; to reproduce the
old numbers exactly, ask for `r0_old_equivalent = r0 * 2**(-3/5)`.  The in-code
claim that the `sqrt(2)` was "verified against D(r=r0) = 6.88" was false: the
agreement it cited happens only near r ≈ 3.2 r0, where the 2× excess crosses the
FFT screen's own low-frequency deficit.

The PSD *shape* (the 0.023 f^-11/3 constant in cycles/m, the von Kármán knee, the
inner-scale cutoff and the v5.30 integer `N//2` DC anchor) is unchanged and still
audit-verified correct; only the amplitude moved.

Files: `lumenairy/elements/elements.py:1186` (re-anchored 2026-09-12 after the history relocation moved lines); PSD extracted to the shared
`_turbulence_psd` helper at `:1151`.
Tests: `tests/unit/test_audit2609_a8_turbulence.py` (whole file);
`tests/unit/test_niche_audit_w3_elements.py::_turbulence_reference` updated —
its independent E-L11 oracle carried the same spurious `sqrt(2)`, i.e. it pinned
the defect.

### Added -- elements: `generate_turbulence_screen(subharmonics=N)`, the Lane low-frequency correction (E3)

An FFT screen cannot hold eddies larger than the grid, so even with the
amplitude fixed its structure function falls below Kolmogorov at large
separations.  `subharmonics=p` adds `p` levels of 3×3 frequency grids at
spacing `1/(3**p · N · dx)` (Lane, Glindemann & Dainty 1992; Schmidt 2010
`ft_sh_phase_screen`), summed in the separable `e.T @ cn @ e` form (three
length-N exponentials per level, not nine full-grid ones).

Default **0** — the shipped behaviour is unchanged and bit-identical for
`subharmonics=0`.  Measured D/D_Kolmogorov over 40 seeds (N = 512, dx = 5 mm,
r0 = 0.1 m): 0.798 → **0.880** at r = 0.05 r0 and 0.463 → **0.777** at
r = 3.2 r0, for +23–28 % wall time and no extra peak memory.

Files: `lumenairy/elements/elements.py:1186` (re-anchored 2026-09-12 after the history relocation moved lines) (signature + validation),
`:1352` (`_turbulence_subharmonics`).

### Fixed -- elements: `apply_grin_lens` was 36 % wrong at the quarter pitch its own Notes recommend (E5, P1) — **default change**

The screen applied the short-rod power `n0·g²·d`; the rod's exact paraxial ABCD
has `C = −n0·g·sin(g·d)`, so the two differ by `sin(gd)/(gd)`.  Measured
`f_code/f_exact`: 0.9996 at g·d = 0.05, 0.9851 at 0.30, 0.9003 at π/4,
**0.6366 at the quarter pitch π/2** and 0.1093 at 0.9π.  An ASM focus scan
confirmed the screen tracked the wrong focal length, not just the wrong formula.
The screen now carries `φ = −k·n0·g·sin(g·d)·r²/2`, exact at any pitch and the
same cost.  Re-measured ASM focus / f_exact after the fix: 1.0000 / 0.9955 /
0.9940 / 0.9940 at g·d = 0.05 / 0.30 / π/4 / π/2 (the residual is the paraxial
screen's own spherical-aberration focal shift at these NAs).

**Migration.**  `thin_form=True` reproduces the previous screen bit-identically
and warns above g·d = 0.2, naming the measured `sin(gd)/(gd)` factor.  On the
exact path, g·d beyond the quarter pitch now warns that a single screen carries
the rod's power only — that power passes through zero at the half pitch, where
the rod instead reimages 1:1 inverted.  The Notes now also state what a single
screen cannot do: the rod's back focal distance is `cos(gd)/(n0 g sin(gd))` from
the exit face while a thin screen focuses a collimated input at `f` past itself,
so the focus POSITION relative to the rod faces needs a ray/split-step model.

Files: `lumenairy/elements/_lens_thin.py:1096` (`thin_form` kwarg), `:1258-1281`
(guards), `:1305` (the power).
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e5_*`.

### Fixed -- bsdf: Harvey-Shack TIS under-read by up to 32 %, and the inherited quadrature could not resolve any realistic lobe (E6, P2)

`HarveyShackBSDF` was the only shipped model without a closed-form TIS, so it fell
back to the base class's 256-point **linear-θ** grid — Δθ = 6.16 mrad, coarser
than the whole lobe, whose shoulder `l` defaults to 0.01.  Measured against a
dense reference:

| model | TIS before | TIS after | reference | error before → after |
|---|---|---|---|---|
| HarveyShack(l = 1e-2, s = 2) *(the defaults)* | 0.00287272 | **0.00289355** | 0.00289355 | −0.72 % → −1.4e-9 |
| HarveyShack(l = 1e-3, s = 2) | 3.54486e-5 | **4.34027e-5** | 4.34027e-5 | **−18.3 %** → −9.3e-8 |
| HarveyShack(l = 1e-4, s = 2) | 3.94725e-7 | **5.78703e-7** | 5.78703e-7 | **−31.8 %** → −7.0e-6 |
| HarveyShack(l = 1e-2, s = 1.5) | 0.0112895 | **0.01131** | 0.01131 | −0.18 % → −3.6e-10 |

`HarveyShackBSDF.total_integrated_scatter` is now the exact closed form
`π b0 l² [(1+1/l²)^(1−s/2) − 1]/(1 − s/2)` (`π b0 l² ln(1+1/l²)` at s = 2),
O(1) and exact.

The **base-class** quadrature — what any user subclass gets — is rebuilt on the
same node count (256 × 128): the variable is `u = sin θ` (for which
`BSDF cos θ dΩ = BSDF(u) u du dφ` exactly), the cells are geometric in `ln u`
from 1e-7 to 1 with two-point Gauss-Legendre inside each, and the `u < 1e-7`
disc is added analytically.  Worst-case relative error across Lambertian,
Gaussian (σ = 1e-2 and 0.3) and Harvey-Shack (l = 1e-1…1e-4, s = 1.5/2/2.5):
**1.4e-6** (typical 1e-9), against −32 % / −18 % for the grid it replaces.

That 1.4e-6 is the worst of those lobes, not a bound.  The rule is exact
through cubic order in `v = ln u`, so the residual is the quartic term
`dv⁴·2⁴/4320`; with `dv = ln(1e7)/128` that is **9.3e-7 even for a FLAT lobe**
(whose integrand `u² = e^{2v}` is not a cubic), measured at −9.2935e-07 for
`B = const` and 1.3e-5 for `B = 1 − u²`.  Budget ~1e-5 for a broad smooth lobe
— still four to five decades better than the grid this replaces.

Files: `lumenairy/elements/bsdf.py:61` (`_TIS_U_MIN`), `:130-201` (base
quadrature), `:503-528` (the closed form).
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e6_harvey_shack_tis_*`,
`::test_e6_base_class_quadrature_*`, `::test_e6_shipped_model_tis_values_are_unchanged`.

### Fixed -- bsdf: `make_bsdf` silently ignored unknown dict keys, including the `A`/`B`/`C` aliases its own docstrings teach (E6, P2)

`make_bsdf({'kind':'gaussian','sigma':0.001,'scatter_fraction':0.5})` built
`GaussianBSDF(sigma_rad=0.01, scattered_fraction=0.01)` — a 10× wider lobe and
50× less scatter than requested, silently; and
`make_bsdf({'kind':'harvey_shack','A':1e-3,'B':0.02})` returned all defaults even
though `A` / `B` are the alias names the `HarveyShackBSDF` docstring itself
introduces.  Unknown keys now raise, naming the accepted set; `A` / `B` / `C` are
accepted as aliases for `b0` / `l` / `s`, and supplying both spellings of one
parameter raises rather than silently picking one.

**Migration.**  This is a strictness change, not only an alias addition: ANY key
the chosen `kind` does not consume is now a `ValueError`, so a surface `bsdf`
spec that carried a free-text annotation (`'comment'`, `'source'`, `'notes'`)
alongside its parameters used to be accepted and is now rejected.  No in-repo
caller passes an extra key (grepped `lumenairy/`, `tests/`, `validation/`,
`examples/`); prescriptions written outside the repo may.  Move such keys out
of the spec dict — the alternative, a silently-ignored key, is the defect this
entry is about.

Files: `lumenairy/elements/bsdf.py:628-715`.
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e6_make_bsdf_*`.

### Performance -- bsdf: `sample_scatter_rays` vectorised over the bundle (113×–377×), and the Harvey-Shack rejection sampler replaced by its exact inverse CDF (E6)

`sample_scatter_rays` made one Python `sample()` call per incident ray, each with
its own `np.array` build and — for Harvey-Shack — its own rejection loop with a
Python `list.extend`.  The lobe-local draw is incidence-independent, so the whole
bundle is now drawn in one call and rotated by a batched frame build; the
rotation reproduces the per-ray code it replaces, including the near-pole
branch, to **4.441e-16** — 2 ULP of a unit direction cosine — and exactly on
350 of 407 incidences swept (the residual is `np.linalg.norm(v)` on a 1-D
vector dispatching to a BLAS `nrm2` where `norm(v, axis=-1)` is a ufunc
reduction; it is not bit-identity and must not be asserted as such).
Measured on 20 000 incident rays, `n_per_ray = 1`:
**113× (lambertian), 327× (gaussian), 377× (harvey_shack)**.

**Migration.**  `sample_scatter_rays` prefers a new batched
`BSDFModel._sample_local` hook, which the three shipped models implement.  A
user subclass that implements only the ABC's abstract `sample` keeps working:
it has no hook, so the function falls back to the per-ray loop for it.

`HarveyShackBSDF` draws `u = sin θ` from the closed-form inverse CDF
(`t = T^ξ` at s = 2, `t = (1 + ξ(T^p − 1))^{1/p}` otherwise, `u = l√(t−1)`),
replacing a rejection sampler whose acceptance was measured at 46 % / 9.2 % /
1.3 % at l = 0.1 / 0.01 / 0.001.  `sample()` of 20 000 directions:
5.47 → 3.73 ms, 7.33 → 3.28 ms, 24.93 → **3.58 ms** — and now constant-time.
Distribution verified against a numerically-integrated CDF (KS = 0.0020 at
n = 2e5, inside the 0.0030 95 % band).

**Migration.** `sample()` on all three models and `sample_scatter_rays` draw
different random numbers than before for a given seed (a different algorithm and
a different per-ray ordering); the sampled distributions are unchanged.

Files: `lumenairy/elements/bsdf.py:113-127` (`_sample_local` contract),
`:306-326`, `:407-426`, `:539-566` (the three local samplers), `:582-615`
(`_rotate_local_to_specular`), `:753-771` (`sample_scatter_rays`).

### Added -- thin_grating: a Klein-Cook / short-period validity guard (E6, P2)

`thin_grating_efficiency_1d` had no regime check, and the failure is invisible in
its output: measured at λ = 1 µm, depth = 10 µm, **Q = 0.16 / 15.7 / 62.8** at
Λ = 20 / 2 / 1 µm produced **zero warnings and ΣT = 1.0000 every time**, so
energy closure cannot be used as the tripwire.  A `UserWarning` now fires when
`Q = 2πλd/(n̄Λ²) > 1` (Bragg regime), or failing that when `Λ < 10λ`, mirroring
`emt.py`'s `_warn_rytov_validity` so the diagnostic policy is consistent inside
the package.  Verified silent at Q = 0.13 and firing at Q = 12.6 and 50.3
(n̄ = duty-weighted 1.25 for the audit's fixture).  The analytic Fourier
coefficients are untouched and still exact (η₊₁ = 4/π² = 0.405285,
η₊₃ = 4/(9π²) = 0.045032 for the 50 %-duty π-step grating).  The module
docstring's garbled `t_m` formula is replaced with the correct expression.

Files: `lumenairy/elements/thin_grating.py:1-108` (docstring, thresholds,
`_warn_thin_grating_validity`), `:196` (the call).
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e6_klein_cook_guard_fires_in_the_bragg_regime`,
`::test_e6_short_period_guard_fires_when_q_is_small`,
`::test_e6_grating_fourier_coefficients_are_untouched`,
`::test_e6_grating_module_docstring_formula_is_well_formed`.

### Performance -- doe: `create_microlens_array` rebuilt on the separable phase — 4.3× memory, 3.0× time, bit-identical (E6, P2)

The lenslet phase `−k/(2f)·(dX² + dY²)` is separable and the footprint test is
too, but `X, Y, in_mla, jx, jy, xc, yc, dX, dY, r_sq, phase` were all full N×N
grids.  The snap, the local coordinate and the footprint test are now length-N
vectors; only `r_sq` is a full grid, and `cos`/`sin` are written straight into
the output's real/imaginary views instead of going through `np.exp(1j·phase)`.
Measured tracemalloc peak in units of one N² float64 grid, one implementation
per process and identical at N = 1024 and N = 2048: **14.13 → 3.25**; median
wall time at N = 2048: **426 → 144 ms**.
Output is bit-identical (`np.array_equal`, several N / pitch / lenslet-count
combinations).  `|T| = 1` everywhere and the exactly-zero steer at each lenslet
centre are unchanged.

Files: `lumenairy/elements/doe.py:251-279`.
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e6_microlens_array_separable_form_is_bit_identical`,
`::test_e6_microlens_array_allocates_a_handful_of_grids_not_ten`,
`::test_e6_microlens_array_physics_is_untouched`.

### Fixed -- doe: `create_periodic_phase_mask` assumed a square cell without checking (E7, P3)

`cell_N = phase_cell.shape[0]` indexed both axes: a `(4, 8)` cell silently
sampled only columns 0–3 — half the design never appeared in the mask — and an
`(8, 4)` cell raised a bare `IndexError` naming neither the function nor the
argument.  Both now raise a named `ValueError`.  Square-cell behaviour is
unchanged (per-cell-pixel occupancy still exactly uniform, 0.0000 % of power off
the order lattice).

Files: `lumenairy/elements/doe.py:155-163`.
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e7_periodic_phase_mask_*`.

### Added -- elements: `apply_aperture(edge='gray')`, an area-weighted aperture edge (E7, P3)

The aperture family offered a hard binary mask only, so an under-sampled stop
carried a systematic throughput bias plus extra Gibbs ringing.  `edge='gray'`
gives each boundary pixel its `edge_samples**2`-supersampled open-area fraction
(default 4 → 16 sub-samples), accumulated one sub-mask at a time so the peak
cost does not grow with `edge_samples` (measured 6.0 float64 grids at N = 2048
for every `n_sub`, against 5.0 for the hard edge).  Measured transmitted-area
error against the analytic disc area, **before and after on the same grid**
(N = 1024, dx = 1 µm): at D/dx = 50 px **−0.5345 % → +0.0384 %**, at 200 px
−0.0220 % → −0.0006 %, at 800 px −0.0075 % → +0.0005 %; `edge_samples=16`
reaches −0.00056 % at 50 px.  The hard-edge error depends on where the rim
falls on the pixel lattice, so the honest summary is the rms over sub-pixel rim
placements: **0.386 % → 0.044 %** at 50 px and 0.031 % → 0.0041 % at 200 px,
over `linspace(0, 0.95, 12)`.  Those are CIRCULAR-rim figures; an axis-aligned
(rectangular) rim is quantised rather than sampled and improves as `1/n_sub`
exactly (0.639 % at 4 → 0.158 % at 16 on a 40.3 × 17.7 px stop).  Default
`'hard'` is bit-identical to before on circular / annular / rectangular,
dtype-preserving, and the JAX path matches NumPy exactly (0.00e+00) for both
edges.  A fully blocked pixel comes out exactly zero for any input, including a
field carrying NaN or inf outside the stop.

Files: `lumenairy/elements/elements.py:227` (re-anchored 2026-09-12 after the history relocation moved lines).
Tests: `tests/unit/test_audit2609_a8_thin_elements.py::test_e7_aperture_*`.

### Fixed -- documentation: four claims in `elements/` and `glass.py` that said the opposite of the code (E7, P3)

* `elements.zernike` told Noll users to map `j_Noll -> (n, m)` with
  `lumenairy.analysis.zernike_index_to_nm`, which is the **OSA** map
  (`m = 2j − n(n+2)`): OSA j = 5 is (2, +2), Noll j = 5 is (2, −2), so a reader
  following the pointer got the wrong polynomial for every j ≥ 5.  The docstring
  now says the library ships no Noll converter and that only the single-index
  ORDERING differs.
* `apply_spherical_lens`'s See Also now states the **reference-plane** difference
  from `apply_real_lens`, re-measured for this fix: on n = 1.5168, R1 = +20 mm,
  R2 = −20 mm, d = 3 mm at 1 µm, thin-lens f = 19.3498 mm, thick EFL 19.8573 mm,
  BFL 18.8424 mm, `apply_spherical_lens` focuses at 19.3111 mm and
  `apply_real_lens` at 18.8080 mm — **503 µm apart on a 19 mm lens (2.6 %)**, with
  1.708 rad rms (0.272 waves) of phase difference across the illuminated pupil.
* `glass.py`'s `_GLASS_VALIDITY_REGISTRY_EXEMPTIONS` comment described `'air'`,
  `'vacuum'` and `'__MIRROR__'` as registry entries; none of the three is one.
  The comment now describes what they actually are.
* The `BaF2` Sellmeier row's comment credited "the authoritative Li 1980 fit
  (main/BaF2/Li)"; the coefficients are Malitson & Dodge 1972, which the row
  reproduces to 2.2e-16 while differing from Li by 8.1e-5 in n_d.  Attribution
  corrected; the data is unchanged.  The `CaF2` row's comment now records that
  the registry dispatches to a *different* published fit (`main/CaF2/Daimon-20`,
  2.8e-5 away in n_d) when `refractiveindex` is installed, so the bundled
  fallback is reached only on a minimal install.

### Changed -- glass: the `'air'` short-circuit now consults the registry, so an ambient model is implementable (E7, P3)

`get_glass_index` / `get_glass_index_complex` returned 1.0 for any spelling of
`'air'` **before** the registry lookup, so a user who registered the Edlén
callable the exemptions comment described had it silently ignored — n = 1.000000
instead of 1.000274 at 1 atm / 1.064 µm, i.e. 274 µm of OPD per metre of air path
(≈ 256 waves).  The short-circuit now honours a callable registered under the
canonical lower-case key `'air'` and falls back to 1.0 otherwise.  **No default
changes**: `'air'` ships unregistered, `list_glasses()` is unchanged (77 names,
no `'air'`), and `'vacuum'` / `'__MIRROR__'` still raise as before.

**Migration.**  The consult is by the CANONICAL LOWER-CASE key only.
`get_glass_index('AIR')` and `get_glass_index('Air')` honour
`GLASS_REGISTRY['air']`, but a callable stored under a mixed-case key
(`GLASS_REGISTRY['Air'] = ...`) is still silently ignored and the lookup
returns 1.0 — the name is case-folded before the short-circuit, so the registry
is only ever consulted at `'air'`.  Register ambient models at `'air'`.

Files: `lumenairy/glass.py:1721-1733`, `:1852-1866`.
Tests: `tests/unit/test_audit2609_a8_glass.py::test_e7_*`.

### Verified -- glass: the 24 formula-3 polynomial rows the auditor could not reach

The partition report listed the 4 CDGM + 10 Hikari + 10 Sumita `POLYNOMIAL_COEFFICIENTS`
rows as its single biggest remaining gap ("given that 3 of 49 Sellmeier rows are
wrong, the polynomial table deserves the same treatment").  All 24 are now
value-checked by the new cross-check and every one is **exact to 0.00e+00** in
both n_d and V_d against its catalogue page.
