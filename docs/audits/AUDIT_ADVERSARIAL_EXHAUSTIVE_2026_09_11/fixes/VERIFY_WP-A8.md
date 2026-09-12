# VERIFY_WP-A8 — independent adversarial re-verification of WP-A8

Verifier: VERIFY-A8 (did not write the fixes).  Date 2026-09-12, branch
`audit-fixes-2026-09`.  WP commit under test: **`0067d63b`** ("fix(elements,materials):
WP-A8 …"), diff base **`658e6142`**.  Environment: CPython 3.14.6, numpy 2.4.6, scipy,
jax 0.10.1 (x64 enabled where used), `refractiveindex` installed with database commit
`a66ef88`, cupy absent.  Every python invocation ran with `OPENBLAS_NUM_THREADS=1`.
Other engineers are mid-edit on `_lens_*`, `pmm/`, `rcwa/`, `bor/`, `eme/`, `analysis/`,
`propagators/` — those changes were ignored and never touched.

## 0. Verdict table

| finding | verdict | the measurement that decides it |
|---|---|---|
| **E1** three wrong Sellmeier rows | **VERIFIED** | hand-evaluated SCHOTT formula 2 from the raw catalogue YAML: agreement 2.2e-16; data sheet 1.0e-6 / 8.7e-8 / 1.6e-7 in n_d |
| **E1** value cross-check | **VERIFIED** | 76 / 76 rows resolve, 0 skipped; two-sided on `N-SK16`, `N-BAK4`, `F2` and two polynomial rows (silent at Δn_d 4.2e-5, fires at 4.2e-4) |
| **E2** `NoExtinctionCoefficient` | **VERIFIED** | 616 calls over all 78 registry names × 8 wavelengths: **0 raises, 0 negative κ**, NaN-κ arm holds |
| **E3** turbulence √2 | **VERIFIED** | D/D_lattice 0.977–1.029 ± 0.011–0.017 at 8 grid/parameter sets incl. odd N = 65 / 127 / 301 (2.0 = pre-fix) |
| **E3** subharmonics | **VERIFIED** | odd-N D/D_Kolm 0.364 → 0.748 (N = 65) and 0.338 → 0.741 (N = 127); `subharmonics=0` bit-identical; correction exactly zero-mean |
| **E5** GRIN power | **VERIFIED** | screen power = −C of the rod ABCD **integrated by RK4** to ≤ 3.7e-14 at 15 (n0, g, g·d); my own ASM scans put the focus at 0.996–0.998 × f_exact |
| **E6** Harvey-Shack TIS closed form | **VERIFIED** | 1.8e-14 worst over 75 (b0, l, s) triples vs `scipy.integrate.quad` |
| **E6** base-class quadrature | **VERIFIED-WITH-NOTES** | correct and 4–5 decades better, but **"exact for a flat lobe" is false** (−9.29e-7) and the worst error on a broad lobe is 1.3e-5, not the 1.4e-6 quoted |
| **E6** `make_bsdf` keys | **VERIFIED** | 9 adversarial specs, all as intended |
| **E6** Klein-Cook guard | **VERIFIED** | silent at Q = 0.013 / 0.126, fires at Q = 12.6 / 50.3 and at Λ/λ = 5; η₊₁ = 4/π² to 5.6e-17 |
| **E6** MLA separable | **VERIFIED** | bit-identical at 5 fixtures incl. odd N and negative f; peak 3.25 grids (n_sub-independent) |
| **E6** `sample_scatter_rays` | **REGRESSION — fixed here** | a `BSDFModel` subclass implementing only the ABC's `sample()` raised `NotImplementedError` where it used to work; also the "bit-identical" rotation claim holds on only 350 / 407 incidences |
| **E7** Noll pointer | **VERIFIED** | `zernike_index_to_nm(5) == (2, +2)`, docstring corrected |
| **E7** air / vacuum / `__MIRROR__` | **VERIFIED** | Edlén callable honoured for every spelling (1.000273975, 274 µm/m); `list_glasses()` still 77; vacuum/MIRROR still raise |
| **E7** non-square DOE cell | **VERIFIED** | 6 cell shapes incl. 1-D and 3-D all classified correctly |
| **E7** grey-pixel aperture | **VERIFIED-WITH-NOTES — defect fixed here** | 6.6× / 110× rms area improvement at fixtures the WP did not use, but a blocked pixel carrying **NaN/inf came back NaN** (`0 * nan`) |
| **E7** reference-plane doc | **VERIFIED** | `p_real.py` re-run reproduces 19.3111 / 18.8080 mm and 1.7076 rad rms to the printed digits |
| **E7** `surface_sag_general(R=0)` | correctly deferred | still `[nan nan]` + 4 anonymous RuntimeWarnings on HEAD; `lenses.py` is WP-A2's |

**Two regressions the WP did not report, in addition to the per-finding items:**

* **R1 — `tests/unit/test_audit_lens.py` was RED on `0067d63b`** (2 failures). Both tests
  pinned the library's TIS to the pre-fix linear-θ quadrature. Fixed here; file now
  48 passed / 4 skipped.
* **R2 — `tests/unit/test_audit_except_budget.py` is RED and WP-A8 alone breaks it**:
  the non-`ui/` `except Exception` census goes **48 → 50** across `658e6142..0067d63b`,
  both new clauses in `lumenairy/glass.py`, against a budget of 48. Not fixed here
  (cross-WP gate — see §5, item **O1**).

---

## 1. Method

For every finding: (a) re-run the audit's own repro script and compare with the WP's
claimed after-numbers; (b) build at least one NEW oracle on a fixture the WP did not use;
(c) read the new/changed tests against `docs/TESTING_STANDARDS.md`; (d) run the existing
test files of the touched modules and their consumers; (e) attack with adversarial inputs
(complex64, JAX x64, F-order and strided views, odd N, anamorphic `dy ≠ dx`, zero and
negative parameters, non-finite fields).

All scratch scripts live in the session scratchpad (not in the repo). The pins that
survive are in the new file `tests/unit/test_audit2609_a8_verify.py` (36 tests, 11 s).

---

## 2. Per finding

### E1 (P0) — the three repaired Sellmeier rows — **VERIFIED**

**Repro re-run.** `repro/orch/verify_rt_glass.py`: bundled and catalogue now print
identical n_d for all five glasses it checks —

```
N-BK7     1.516800 / 1.516800      N-BAF52   1.608631 / 1.608631
N-LAK33A  1.753930 / 1.753930      N-LAK33B  1.755000 / 1.755000
N-SF11    1.784720 / 1.784720
```

`repro/THIN-ELEMENTS-GLASS/p_glass3.py` prints `FLAGGED: []`. Both match the report.

**New independent oracle (the WP used the `refractiveindex` *API*; I did not).** I read
the catalogue YAML as raw text, pulled the `formula 2` coefficient line with a regex, and
evaluated `n² − 1 = Σ Bᵢλ²/(λ² − Cᵢ)` in my own code at nine wavelengths from 0.4 to
2.0 µm:

| glass | max \|hand − `get_glass_index`\| | Δn_d vs data sheet | ΔV_d vs data sheet | glass code |
|---|---|---|---|---|
| N-BAF52 | **2.22e-16** | +1.012e-06 | −0.0026 | 609466.305 ✓ |
| N-LAK33A | **2.22e-16** | +8.724e-08 | +0.0008 | 754523.422 ✓ |
| N-LAK33B | **2.22e-16** | +1.645e-07 | +0.0001 | 755523.422 ✓ |
| N-BK7 | **0.00e+00** | +3.450e-08 | −0.0027 | 517642.251 ✓ |
| N-SF11 | **2.22e-16** | −5.769e-08 | −0.0000 | 785257.322 ✓ |

The same hand oracle on the pre-fix rows gives n_d 1.637147 / 1.754279 / 1.755294 and
V_d 42.469 / 53.031 / 52.940 — Δn_d +2.85e-2 / +3.49e-4 / +2.94e-4, exactly the report's
and the audit's numbers. The audit's "mis-copied N-KZFS11" hypothesis: N-KZFS11's own
n_d is 1.637750, i.e. 6.0e-4 from the pre-fix N-BAF52 row — near, not identical, which
matches the WP's finding that no verbatim source row exists.

Also re-measured independently: **N-LASF40** is the worst honest row, |Δn_d| = 4.2e-6
against its catalogue page and up to 5.35e-5 across 0.4–2.0 µm — the WP's stated noise
floor is right.

*Aside, not a WP-A8 issue:* `repro/.../p_glass2.py`'s own `PUB_VD` table is wrong for at
least three glasses (N-BAK4 43.87, H-ZK9B 50.40, S-LAH64 40.83 where the manufacturers
publish 55.98, 60.32, 47.37), so its "<<<<" flags on those rows are fixture errors, not
library errors. The library agrees with the catalogue on all three.

Pins added: `test_verify_a8_e1_bundled_row_is_the_published_schott_row`,
`::_row_reproduces_the_data_sheet_nd_and_vd`,
`::_pre_fix_rows_would_fail_both_oracles` (the published coefficients are literals in the
test, so the pin does not need the optional package).

### E1 (continued) — the value cross-check — **VERIFIED**

**Coverage.** `_cross_check_bundled_values()` resolves **76 of 76** rows (52 Sellmeier +
24 polynomial) and skips none. A gate that silently skips is a gate that does not gate;
this one does not skip. Residual spread over the repaired table: worst |Δn_d| 4.176e-6
(N-LASF40, 2nd 2.723e-6 N-LASF45), worst |ΔV_d|/V_d 1.128e-4 (N-LASF40), **median 0.0**
on both — the bars 5e-5 / 1e-3 sit ~1 decade above.

**Sensitivity on rows WP-A8 never touched** (the WP only demonstrated the gate on the
three it repaired):

```
N-SK16   B1 x (1+1e-6/1e-5/1e-4/1e-3) -> dn_d 4.23e-7 / 4.23e-6 / 4.23e-5 / 4.23e-4
                                flagged  False       False       False       True
N-BAK4   same ladder                     False       False       False       True
F2       same ladder                     False       False       False       True
F1-CDGM  (formula-3 polynomial row)      False       False       False       True
F2-CDGM  (formula-3 polynomial row)      False       False       False       True
```

Two-sided, with the transition exactly at the 5e-5 bar, and it reaches the polynomial
table the partition report called its biggest gap. `_check_glass_registry_consistency(
check_values=True)` raises `RuntimeError` naming the offending glass and restores clean.
Import is unaffected (`import lumenairy` = 6.60 s wall on this box, dominated by numpy /
scipy; the module-level call is still the no-argument one).

Pin added: `test_verify_a8_e1_value_gate_is_two_sided_on_a_glass_the_wp_never_touched`.

### E2 (P1) — `get_glass_index_complex` — **VERIFIED**

`repro/.../p_glass2.py` re-run: **0 of the tuple glasses raise** (was 7). I then swept
*every* registry name, not just the tuple ones, at 8 wavelengths from 0.35 to 2.2 µm —
**616 calls, 0 raises, 0 negative κ**. `E-BAK1` and `E-LAK04` at 1.31 µm (the NaN-κ arm
the WP found while fixing E2) return `1.5560943+0j` / `1.6339000+0j` with one warn-once
message each and none on the second call. `N-BK7` at 1.55 µm is unchanged at
**1.5006520 + 1.436132e-07 j**, the audit-verified value and sign. The package-absent
path still works (monkeypatched `_ensure_refractiveindex_loaded → False`:
`1.5006520 + 1.4361e-07 j`, no warning), so CONVENTIONS §10 is honoured.

Residual, **pre-existing and not a WP-A8 regression** (recorded because the sweep found
it): `get_glass_index_complex('SILICON', λ)` returns `nan + 0j` for λ < 1.2 µm — the
**real** part is NaN because the `main/Si/Li-293K` page's n-table does not span there.
It is *warned* (`get_glass_index: SILICON Sellmeier validity is [1.200e-06, 1.400e-05]`),
so it is not silent, but it is the same silent-wrong shape one level over from the κ arm
WP-A8 closed. See §5 **O5** — and note that §8 then **closed** it on the coordinator's
ruling, so those five NaN readings are now a named `ValueError`. The "0 raises" above is
the measurement as the WP shipped it; re-run today the same sweep raises on exactly
those five out-of-range SILICON calls and on nothing else. E2's own contract — the
κ = 0 fallback for a page with no k-table — is unaffected either way, and the two
wavelengths its tests use (1.31 and 1.55 µm) are inside every page's range.

### E3 (P1) — the spurious √2 — **VERIFIED**

**Repro re-run.** `repro/orch/verify_turb.py`: variance / ΣPSD·df² = **0.988** (was
1.976). `repro/.../p_turb.py`: D_meas/D_lattice = **0.9937 / 0.9922 / 0.9896** at
r = 5 / 10 / 20 mm (was 1.987 / 1.984 / 1.979), and D/D_Kolmogorov = 0.792 at
r = 0.05 r0 — below the continuum, as a band-limited screen must be. All match the
report.

**New oracle at grids and parities the WP's tests do not use** — the exact discrete
structure function and the exact lattice variance, built from the documented PSD:

| N | dx | r0 | L0 | l0 | seeds | sep | D/D_lattice | var/lat-var |
|---|---|---|---|---|---|---|---|---|
| 127 (odd) | 2.0e-3 | 0.05 | ∞ | 0 | 60 | 1 | 0.991 ± 0.012 | 0.958 ± 0.058 |
| 127 (odd) | 2.0e-3 | 0.05 | ∞ | 0 | 60 | 3 | 0.988 ± 0.017 | — |
| 200 | 1.0e-2 | 0.30 | ∞ | 0 | 60 | 2 | 1.011 ± 0.014 | 1.007 ± 0.069 |
| 301 (odd) | 4.0e-4 | 0.02 | ∞ | 0 | 40 | 1 | 0.991 ± 0.010 | 0.966 ± 0.071 |
| 64 | 2.5e-2 | 1.00 | ∞ | 0 | 80 | 1 | 0.977 ± 0.014 | 0.929 ± 0.050 |
| 128 | 5.0e-3 | 0.10 | 2.0 m | 0 | 60 | 2 | 1.016 ± 0.014 | 1.065 ± 0.051 |
| 128 | 5.0e-3 | 0.10 | 2.0 m | 5 mm | 60 | 2 | 1.017 ± 0.015 | 1.065 ± 0.051 |
| 65 (odd) | 5.0e-3 | 0.10 | ∞ | 0 | 80 | 1 | 1.029 ± 0.013 | 1.095 ± 0.050 |

Every entry is 1.0 within a few standard errors and none is anywhere near 2.0; the
von Kármán and inner-scale arms are covered too.

**Subharmonics.** `subharmonics=0` is bit-identical to the default. The correction is
exactly zero-mean (|mean| ≤ 1.3e-14 against an rms of 1.9–6.8 rad at p = 1…5). At odd N
it still works: D/D_Kolmogorov at r = N/4 pixels goes **0.364 → 0.748** (N = 65) and
**0.338 → 0.741** (N = 127) with `subharmonics=3`. I re-derived the separable
`e.T @ cn @ e` form independently — it is the correct double sum for a square grid.

The WP's edit of `tests/unit/test_niche_audit_w3_elements.py::_turbulence_reference` is
exactly what it claims: one line, `sqrt(2.0*psd)` → `sqrt(psd)`, in an "independent"
oracle that carried the same defect. Correct per COMMON §"a test that asserts the OLD
behaviour".

Pins added: `test_verify_a8_e3_structure_function_at_other_grids_and_parities`
(3 parametrisations), `::_subharmonics_keep_the_correction_zero_mean`.

### E5 (P1) — `apply_grin_lens` — **VERIFIED**

**New oracle that does not use `sin(g d)` at all.** I integrated the paraxial ray
equation `y'' = −g² y` for the parabolic rod with RK4 and read the ABCD element `C`
directly. The screen's recovered quadratic coefficient against `−C`:

| n0 | g [1/m] | g·d | P(screen) | −C (RK4) | rel | sin(gd)/(gd) = f_thin/f_exact |
|---|---|---|---|---|---|---|
| 1.60 | 300 | 0.050 | 23.9900012 | 23.9900012 | 1.1e-15 | 0.9996 |
| 1.60 | 300 | 0.300 | 141.8496992 | 141.8496992 | 3.7e-14 | 0.9851 |
| 1.60 | 300 | π/4 | 339.4112550 | 339.4112550 | 7.9e-15 | 0.9003 |
| 1.60 | 300 | π/2 | 480.0000000 | 480.0000000 | 7.8e-15 | **0.6366** |
| 1.60 | 300 | 1.200 | 447.3787613 | 447.3787613 | 1.2e-14 | 0.7767 |

(the same at n0 = 1.48 / g = 55 and n0 = 2.00 / g = 1200 — 15 combinations, worst
3.7e-14).

**My own ASM focus scan**, at grids, NA and wavelength the WP did not use (λ = 1.064 µm,
N = 640–768, dx = 5–6 µm), with the window centred on **f_exact** rather than f_code:

| n0 | g | g·d | f_exact | f_thin | ASM peak | peak/f_exact | peak/f_thin |
|---|---|---|---|---|---|---|---|
| 1.52 | 180 | 0.300 | 12.3679 mm | 12.1832 mm | 12.3408 mm | **0.9978** | 1.0129 |
| 1.52 | 180 | π/4 | 5.1689 mm | 4.6537 mm | 5.1496 mm | **0.9963** | 1.1066 |
| 1.52 | 180 | π/2 | 3.6550 mm | 2.3268 mm | 3.6405 mm | **0.9960** | 1.5646 |
| 1.75 | 90 | 1.200 | 6.8122 mm | 5.2910 mm | 6.7935 mm | **0.9973** | 1.2840 |

The 0.3–0.4 % residual is the paraxial screen's own spherical-aberration focal shift, the
same class as `apply_thin_lens(paraxial)`'s −40 µm at f = 20 mm.

**The WP's explanation of the repro fixture checks out.** `p_thinlens.py` computes
`f_code = 1/(n0 g² d)` *in the script* (line 66) and scans
`linspace(0.8·f_code, 1.3·f_code)` (line 73). At the quarter pitch that window is
1.061–1.724 mm; it prints 1.7242 mm = exactly 1.3 × 1.3263, the window's top edge. The
fixture is now blind, not the code.

Adversarial: `g = 0` and `d = 0` give exactly zero phase and no warning; `g < 0` gives
the same (positive) power as `+g`, `d < 0` flips it — same as pre-fix. Warning matrix
verified two-sided over 7 (g·d, `thin_form`) combinations; `thin_form=True` reproduces
the pre-fix screen **bit-identically** (`max|diff| = 0.0`); complex64 preserved; F-order
input matches C-order exactly.

Pin added: `test_verify_a8_e5_screen_power_equals_the_integrated_rod_abcd`
(3 parametrisations; the RK4 oracle's own floor is measured in-test by Richardson).

### E6 — Harvey-Shack TIS closed form — **VERIFIED**

`repro/.../p_bsdf2.py` re-run: at l = 1e-3, `analytic = lib = ref = 4.34027e-05`,
`lib/analytic = 1.0000` (pre-fix 3.54486e-5, −18.33 %).

**New oracle**, `scipy.integrate.quad` split at the shoulder with its own reported
absolute error, over **75 triples** `b0 ∈ {1, 0.025, 7.3} × l ∈ {3e-1 … 1e-5} ×
s ∈ {1.2, 1.7, 2.0, 2.3, 3.1}` — parameters outside the WP's fixture set, including
b0 ≠ 1 and s off the 1.5/2/2.5 grid: **worst relative difference 1.85e-14**, typical 0.

Pin added: `test_verify_a8_e6_harvey_shack_tis_at_parameters_the_wp_did_not_use`.

### E6 — base-class quadrature — **VERIFIED-WITH-NOTES**

Against a 9600-node composite geometric Gauss-Legendre rule built here:

| lobe (none of these is in the WP's fixture set) | base quadrature | GL reference | rel err |
|---|---|---|---|
| HS l = 2e-1, s = 1.3 | 7.63967895e-01 | 7.63967912e-01 | 2.2e-08 |
| HS l = 7e-2, s = 2.0 | 8.19475010e-02 | 8.19475010e-02 | 8.2e-10 |
| HS l = 5e-3, s = 2.7 | 2.18900307e-04 | 2.18900307e-04 | 3.5e-10 |
| HS l = 8e-4, s = 1.9 | 4.18323757e-05 | 4.18323757e-05 | 1.2e-11 |
| HS l = 1e-6, s = 2.2 | 2.94335267e-11 | 2.94337156e-11 | 6.4e-06 |
| B(u) = 1 (flat) | 3.14158973e+00 | 3.14159265e+00 | **9.3e-07** |
| B(u) = 1 − u² | 1.57081662e+00 | 1.57079633e+00 | **1.3e-05** |
| B(u) = (1 − u²)⁸ | 3.49065863e-01 | 3.49065850e-01 | 3.7e-08 |

The fix is real and large (the grid it replaces read −32 % / −18 % on narrow lobes), but
**two documented claims are wrong**:

1. *"the cell weights sum to `∫u du` exactly (so a flat lobe integrates with zero
   quadrature error)"* — WP report §2 item 1 and `WP-A8_CHANGELOG.md`. Measured
   `sum(w) = 0.49999953532703` against `∫₀¹u du = 0.5`: **−9.2935e-07 relative**. This is
   not a bug, it is the textbook 2-point Gauss-Legendre residual: the flat-lobe integrand
   in `v = ln u` is `e^{2v}`, not a cubic, and `dv⁴·2⁴/4320 = 9.3122e-07` with
   `dv = ln(1e7)/128` — agreement with the measurement to 0.2 %. But the claim as written
   is false and the test built on it (`test_e6_base_class_quadrature_is_exact_for_a_flat_lobe`)
   carried a bare `< 1e-5` with no derivation, which TESTING_STANDARDS calls a defect.
   **Fixed here** (§4).
2. *"worst 1.4e-6"* — true for the listed models; a smooth broad lobe (`1 − u²`) reads
   **1.3e-5**, and a lobe narrower than `10 × _TIS_U_MIN` reads 6.4e-6. The docstring is
   scoped, so this is a note rather than an error, but the number should not be read as a
   guarantee.

### E6 — `make_bsdf` key validation — **VERIFIED**

Nine adversarial specs, all as intended: `{'sigma','scatter_fraction'}` → raises,
`{'A','B'}` → `b0=1e-3, l=0.02`, `{'C': 2.5}` → `s=2.5`, `{'A','b0'}` → raises,
`{'Rho'}` → raises, a stray `'comment'` key → raises, `{'kind':'gaussian'}` alone → OK.
No in-repo caller passes an extra key (grepped `lumenairy/`, `tests/`, `validation/`,
`examples/`). Behavioural break for user prescriptions noted in §5 **O3**.

### E6 — Klein-Cook guard — **VERIFIED**

| Λ | depth | λ | Q | Λ/λ | warnings | ΣT |
|---|---|---|---|---|---|---|
| 20 µm | 10 µm | 1 µm | 0.126 | 20.0 | **0** | 1.000000 |
| 2 µm | 10 µm | 1 µm | 12.566 | 2.0 | **1** (Klein-Cook) | 1.000000 |
| 1 µm | 10 µm | 1 µm | 50.265 | 1.0 | **1** (Klein-Cook) | 1.000000 |
| 20 µm | 1 µm | 1 µm | 0.013 | 20.0 | **0** | 0.959605 |
| 5 µm | 0.2 µm | 1 µm | 0.040 | 5.0 | **1** (short period) | — |
| 100 µm | 2 µm | 1.55 µm | 0.003 | 64.5 | **0** | — |

Two-sided, and both arms are exercised. ΣT = 1.0000 in the Bragg cases confirms the WP's
point that energy closure cannot be the tripwire. The analytic coefficients are untouched:
η₊₁ = 0.405284734569351 vs 4/π² (**5.6e-17**), η₊₃ = 0.045031637174372 vs 4/(9π²)
(**6.9e-18**), η₀ = 2.6e-32.

### E6 — `create_microlens_array` — **VERIFIED**

Bit-identical to the pre-separable implementation (transcribed from the diff) at five
fixtures the WP did not use, including **odd N** and a **negative focal length**:

```
N=256 pitch=40.0um n=5   max|a-b| = 0.000e+00      N=255 (odd)  0.000e+00
N=512 pitch=101.3um n=4  max|a-b| = 0.000e+00      N=128        0.000e+00
N=129 (odd) pitch=33.7um n=3, f = -1 mm            0.000e+00
```

`|T| = 1` everywhere in all five. tracemalloc peak, in units of one N² float64 grid:
**3.25 new** (identical at N = 1024 and 2048) — exactly the WP's number. My transcription
of the old code peaks at 14.13 grids rather than the 10.13 the WP reports; the *new*
figure, which is the one that matters, reproduces exactly. `p_doe.py` re-run: N = 2048
now **0.372 s** (the audit measured 1.25 s), `|T|` unity, zero steer at every lenslet
centre, fractional pitch fine.

### E6 — `sample_scatter_rays` — **REGRESSION (fixed here)** + a claim to correct

**Distributions: correct.** My own KS tests against the analytic CDF, n = 2e5, 99 %
critical 0.00364: Harvey-Shack at (l, s) = (1e-1, 2), (1e-2, 1.5), (1e-3, 2.5),
(5e-4, 2), (2e-2, 3) all give **D = 0.00231**; Gaussian at σ = 1e-3, 1e-2, 0.2 all give
**D = 0.00190**; Lambertian **D = 0.00104**. The *identity across parameters* is itself
the proof: if `u = F⁻¹(ξ)` exactly then `F(u_i) = ξ_i` and the KS statistic collapses onto
the uniform stream's own, which no rejection sampler can do. Perf re-measured (median of
5 interleaved runs, 20 000 rays): **202× / 660× / 306×** (lambertian / gaussian /
harvey_shack) against a per-ray loop, and `HarveyShackBSDF.sample(20 000)` is now
constant-time in `l` (4.08 / 5.77 / 5.39 ms at l = 1e-1 / 1e-2 / 1e-3, against the audit's
5.47 / 7.33 / 24.93 ms).

**Regression.** `sample_scatter_rays` called `bsdf._sample_local(...)` unconditionally.
`BSDFModel` is a public export (`lumenairy.__all__`, `lumenairy.elements.__all__`) and the
README documents it as *the* subclassing entry point; its only abstract draw method is
`sample`. A subclass written against that ABC therefore raised
`NotImplementedError: …must implement _sample_local…` where the pre-vectorisation code
worked. Reproduced with a minimal `evaluate` + `sample` subclass. **Fixed** (§4) with a
per-ray compatibility path selected by `type(bsdf)._sample_local is
BSDFModel._sample_local`; the three shipped models all override it and keep the fast path.

**Claim to correct.** The report and changelog say the batched rotation is
*"bit-identical to the per-ray code it replaces"*. Over 407 incidences (400 uniform random
plus 7 at and around the `|spec_z| = 0.999` threshold) it is bit-identical on **350** and
differs by up to **4.441e-16** (2 ULP of a unit direction cosine) on the other 57 —
`np.linalg.norm(v)` on a 1-D vector can go through BLAS `nrm2` while
`np.linalg.norm(v, axis=-1)` is a ufunc reduction. Physically irrelevant; but the WP's
test asserted `np.array_equal` on four fixtures that happen to land in the exact set,
which is the per-build knife edge TESTING_STANDARDS S4 warns about. **Fixed** (§4).

### E7 — the remainder

* **Noll pointer — VERIFIED.** `zernike_index_to_nm(5) == (2, +2)`; docstring now states
  the library ships no Noll converter.
* **`'air'` / `'vacuum'` / `'__MIRROR__'` — VERIFIED.** Defaults unchanged
  (`'air'`/`'AIR'`/`'Air'`/`'aIr'` → 1.0; `'vacuum'`/`'VACUUM'`/`'MIRROR'`/`'__MIRROR__'`
  → `ValueError`); `len(list_glasses()) == 77` with no `'air'`. With my own Edlén 1966
  callable registered: **1.000273975** for every spelling, matching my hand evaluation
  exactly — **274 µm/m of OPD** that the short-circuit used to discard. A
  complex-returning callable keeps its κ (`1.000273+3e-09j`) and `get_glass_index` takes
  the real part. Note in §5 **O6**: the consult is lower-case-key only (documented).
* **Non-square DOE cell — VERIFIED.** `(4,8)`, `(8,4)`, `(8,)` and `(2,2,2)` all raise a
  named `ValueError`; `(8,8)` and `(2,2)` still work.
* **Grey aperture — VERIFIED-WITH-NOTES, one defect fixed.** See below.
* **Reference-plane doc — VERIFIED.** `p_real.py` re-run prints focus(screen)
  = 19.3111 mm, focus(real_lens) = 18.8080 mm, rms 1.7076 rad = 0.2718 waves, PV
  8.1737 rad = 1.3009 waves, thin f 19.3498 / thick EFL 19.8573 / BFL 18.8424 mm — every
  number in the new See Also block, to the printed digits.
* **`surface_sag_general(R=0)`** still returns `[nan nan]` with four anonymous
  RuntimeWarnings on HEAD. Correctly left to WP-A2 and correctly requested.

### E7 — grey-pixel aperture, in detail

Transmitted area against the analytic area, at diameters, sub-pixel centre offsets and
**anamorphic `dy ≠ dx`** the WP did not use (N = 512, dx = 1 µm):

| shape | D/dx | dy/dx | offset | hard | gray(4) | gray(16) |
|---|---|---|---|---|---|---|
| circular | 37 | 1.0 | 0.00 | +9.105e-03 | −1.954e-04 | +2.259e-05 |
| circular | 37 | 1.0 | 0.37 | −4.846e-03 | −8.348e-04 | +8.072e-05 |
| circular | 91 | 1.0 | 0.50 | −2.442e-03 | +1.334e-04 | +3.013e-05 |
| circular | 63 | 1.0 | 0.13 | +1.846e-03 | −1.388e-04 | +1.404e-05 |
| circular | 63 | **2.5** | 0.13 | −7.203e-04 | +8.170e-05 | −1.541e-05 |
| circular | 145 | **0.4** | 0.29 | +6.078e-05 | +3.050e-05 | −1.011e-06 |
| circular | 23 | 1.0 | 0.50 | −1.799e-02 | +3.067e-03 | −1.670e-04 |
| annular | 30/75 | 1.0 | 0.21 | +2.693e-03 | −3.539e-05 | −1.012e-05 |
| rectangular | 40.3 × 17.7 | 1.0 | 0.00 | −2.287e-02 | −6.393e-03 | +1.581e-03 |

rms over the seven circular fixtures: **hard 7.93e-3 → gray(4) 1.21e-3 (6.6×) →
gray(16) 7.20e-5 (110×)**. On the audit's own three sizes the WP's *after* numbers
reproduce exactly (+0.0384 % at 50 px, −0.00056 % at 16 sub-samples, −0.0006 % at 200 px).

Notes:
* **Defect found and fixed.** The grey return was `E_in * frac`; `frac` is exactly 0
  outside the opening and `0.0 * nan == nan`, so a field carrying NaN or inf outside the
  stop came back with the whole blocked region reading `nan+nanj` — and non-finite values
  outside a stop are ordinary here (`surface_sag_general` returns NaN outside the conic
  domain, `apply_thin_lens(model='aplanatic')` leaves a sentinel). One FFT and the NaN
  covers the plane. **Fixed** (§4): select rather than scale outside the opening.
* The rectangular arm converges only as `1/n_sub` (6.39e-3 → 1.58e-3, a factor 4.0 from 4
  to 16 sub-samples), because an axis-aligned rim is quantised, not sampled. Correct
  behaviour, but the docstring's `~1/n_sub**1.5` and its numbers are circular-rim
  specific. Pinned so the docstring cannot be read as a rectangle guarantee.
* The code comment claimed the peak cost is "a single float grid plus a single boolean
  grid". Measured at N = 2048: **6.00 float64 grids for gray against 5.00 for hard**, and
  — the part that actually matters — **identical at n_sub = 2, 4 and 8**. Comment
  corrected in place.
* Adversarial inputs all clean: complex64 → complex64 (numpy and JAX), F-order and
  strided non-contiguous views give results identical to the C-order call, odd 95 × 93
  grid with `dy = 2.3·dx` runs, `edge='soft'` / `edge_samples` 0, 2.5, −3 all raise with
  the `apply_aperture:` prefix, **JAX x64 parity exactly 0.00e+00** for both edges.

---

## 3. Tests audited against `docs/TESTING_STANDARDS.md`

Read all 107 WP-A8 tests. Overall they are unusually good: derived bars with measured
values and dates, independent oracles, decisions rather than readings where possible, and
the JAX-parity test correctly asserts the absence fact instead of `pytest.skip`. Items
found:

| # | test | issue | action |
|---|---|---|---|
| 1 | `..._thin_elements.py::test_e6_base_class_quadrature_is_exact_for_a_flat_lobe` | bar `1e-5` with **no derivation and no measured value** ("a numeric constant in a test without a stated origin is a defect"), and the stated property is false | **fixed**: bar 3e-6, derivation + measurement in the docstring, plus a lower arm pinning that the residual equals the predicted GL term |
| 2 | `..._thin_elements.py::test_e6_batched_rotation_equals_the_per_ray_reference` | `np.array_equal` on a quantity a BLAS build may move (S4/S5) | **fixed**: bar `8·eps` derived from the 407-incidence sweep |
| 3 | `..._thin_elements.py::test_e6_sample_scatter_rays_is_vectorised...` | asserts on **source text** (`'for i in range(n_rays)' not in src`); tests the implementation, not the property — and the compatibility path added in §4 legitimately contains that loop | **fixed**: counts `_sample_local` calls instead (must be exactly one for the whole bundle) |
| 4 | `..._turbulence.py::test_e3_shared_psd_helper_is_used_by_both_paths` | also a source-text assertion (`'0.023' not in src.split('"""')[-1]`) | left — it is an anti-duplication guard, not a physics pin, and it is honest about that |
| 5 | `..._glass.py` (4 sites) | `pytest.fail('refractiveindex is required…')` makes a **minimal install go red**; sibling WPs use `pytest.importorskip` for zarr/h5py/jax and CONVENTIONS §10 lists `refractiveindex` as optional | left — cross-cutting policy call; see §5 **O2** |
| 6 | `..._thin_elements.py::test_e6_microlens_array_allocates_a_handful_of_grids_not_ten` | tracemalloc bar, but deterministic, two-sided (1.85× / 1.69×) and the docstring says so explicitly | OK |

---

## 4. Defects I fixed in WP-A8's files (each with its own verification)

**F1 — `lumenairy/elements/bsdf.py::sample_scatter_rays`: restore the public-ABC
contract.** A `BSDFModel` subclass implementing only `evaluate` + `sample` raised
`NotImplementedError`. Added a per-ray compatibility path chosen by
`type(bsdf)._sample_local is BSDFModel._sample_local`, plus a Notes paragraph.
*Verified*: new pin
`test_audit2609_a8_verify.py::test_verify_a8_sample_scatter_rays_still_serves_a_sample_only_subclass`
(a `_SampleOnly` subclass, 64 rays × `n_per_ray=3`, unit-norm directions in the outgoing
hemisphere) plus an assertion that all three shipped models still override
`_sample_local`; and the re-measured speedups above are unchanged (the shipped models
never reach the fallback).

**F2 — `lumenairy/elements/elements.py::apply_aperture(edge='gray')`: a blocked pixel must
be exactly zero.** Changed the return to
`xp.where(frac > 0, E_in * frac, zeros(dtype=E_in.dtype))`, and corrected the memory
comment to the measured numbers.
*Verified*: new pin
`::test_verify_a8_e7_gray_edge_zeroes_a_blocked_pixel_even_if_it_is_not_finite`
(NaN, +inf, −inf); re-measured areas unchanged to the last printed digit
(D = 37 px gray(4) −1.954e-04 before and after); dtype preservation and JAX x64 parity
still `0.0e+00` for complex64 and complex128.

**F3 — `tests/unit/test_audit_lens.py`: two tests that pinned the pre-fix defect and were
FAILING on `0067d63b`.** This is the report §14 "tests that pin defects" pattern, and
WP-A8 did not run this file.
* `test_vectorised_matches_scalar_loop_harvey_shack` asserted
  `|closed_form − linear_θ_loop| ≤ 1e-10`; measured difference **1.003e-06** at
  (b0, l, s) = (0.05, 0.02, 2). Re-oracled onto the analytic
  `π b0 l² ln(1 + 1/l²)` (agreement **0.00e+00**), with the linear-θ reading
  (−2.0397e-03 relative) asserted as the thing that must not come back.
* `test_vectorised_matches_scalar_loop_default_lambertian` asserted the same 1e-10 against
  the linear-θ loop; measured difference **3.514e-06** at ρ = 0.3. Re-oracled onto the
  analytic answer (TIS of a `ρ/π` lobe is exactly ρ): new grid **−9.2935e-07** relative,
  old grid **−1.2642e-05**, i.e. 13.6× better; the surviving arm is the decision "closer
  to ρ than the grid it replaced".
* `test_default_tis_against_lambertian_closed_form`'s 1e-3 bar and its "achievable bound
  without going to a Gauss-Legendre rule" comment are now stale — tightened to 3e-6 and
  run at two ρ.
* `_reference_scalar_tis` keeps a note saying it is no longer an oracle.
*Verified*: `tests/unit/test_audit_lens.py` → **48 passed, 4 skipped** (was 2 failed,
46 passed).

**New file — `tests/unit/test_audit2609_a8_verify.py`, 36 tests, 11 s.** Independent
oracles only: the published SCHOTT coefficients as literals evaluated in-test, the
value-gate ladder on `N-SK16` and a polynomial row, the exact lattice structure function
at three new grids, the RK4 rod ABCD with an in-test Richardson floor, a 9600-node
Gauss-Legendre TIS, the inverse-transform invariance of the sampler, the subclass
fallback, the NaN-zeroing, and the anamorphic/offset aperture areas. No `pytest.skip`.

---

## 5. Open items for the orchestrator

| id | sev | item |
|---|---|---|
| **O1** | **high** | `tests/unit/test_audit_except_budget.py` is RED, and **WP-A8 alone takes the non-`ui/` `except Exception` census from 48 to 50 against a budget of 48** — both new clauses are in `lumenairy/glass.py` (`_catalogue_index_fn_from_entry`'s page-resolution guard and `_cross_check_bundled_values`'s per-row guard). Measured per commit: base `658e6142` = 48, WP head `0067d63b` = 50, current working tree = 55 (the other 5 are `_lens_imap.py` +1, `_lens_jax.py` +2, `_lens_real.py` +2, `raytrace/exit_vertex.py` +1, `propagators/hfpi.py` −1 — other WPs, still uncommitted). I did not touch it because the budget constant is a cross-WP gate. Remedies, in the census's own order of preference: **NARROW** both clauses to an explicit tuple plus the package's classes resolved the way `_missing_kappa_exceptions()` already does in the same file; or **bump** `_NON_UI_EXCEPT_BUDGET` once for all WPs with the justification recorded (both clauses sit on an optional-package boundary and neither runs at import or in library dispatch, which is the same "untypeable without importing the optional package" argument the jax-tracer exemption rests on). |
| **O2** | medium | Four `pytest.fail('refractiveindex is required…')` sites in `tests/unit/test_audit2609_a8_glass.py` (lines ~151, ~279, ~307) turn a **minimal install** red. `refractiveindex` is an optional extra (CONVENTIONS §10) and sibling WPs use `pytest.importorskip`. Suggested: use the pattern the same file already uses at line 124 — assert the absence fact (`_cross_check_bundled_values()[0] == 0`) and return — which satisfies both TESTING_STANDARDS rule 4 and the optional-dependency contract. |
| **O3** | medium | `make_bsdf` now raises on ANY key the chosen kind does not consume. No in-repo caller is affected, but a user prescription carrying an annotation key (`'comment'`, `'source'`, `'notes'`) on a surface `bsdf` dict now fails where it used to be ignored. The changelog documents the aliases but not the new strictness as a migration item; it should. |
| **O4** | low | `lumenairy/propagators/system.py:948` could pass `subharmonics=elem.get('subharmonics', 0)` through the `'turbulence'` element type (WP-A8's own request 3). Default 0 keeps behaviour identical. |
| **O5** | low | `get_glass_index_complex('SILICON', λ)` returns `nan + 0j` (NaN **real** part) below 1.2 µm — the page's n-table does not span there. It warns via the existing validity message, so it is not silent, and it predates WP-A8; but it is the same shape as the NaN-κ arm E2 closed, one level over. A `NaN → named ValueError` (or an explicit "outside the page's tabulated range" error) would finish the class. |
| **O6** | low | The `'air'` registry consult uses the canonical lower-case key only, so `GLASS_REGISTRY['Air'] = callable` is still silently ignored (verified: returns 1.0). The docstring says "lower-case key", so this is documented rather than wrong — worth one sentence in the changelog's migration note. |
| **O7** | low | Documentation accuracy in `WP-A8_REPORT.md` / `WP-A8_CHANGELOG.md`, for the orchestrator to correct while assembling the CHANGELOG: (a) *"the cell weights sum to `∫u du` exactly"* / *"a flat lobe integrates exactly"* — measured −9.29e-7 (see E6 above); (b) the batched rotation is *"bit-identical"* on 350 of 407 incidences, not all (worst 4.44e-16); (c) the MLA memory pair "10.13 → 3.25 grids" — the 3.25 reproduces exactly, my transcription of the pre-fix code peaks at 14.13; (d) the aperture pair "−0.942 % → +0.038 % at D/dx = 50 px" mixes the audit's hard reading with a gray reading taken on a different grid (on the grid that gives +0.0384 %, the hard reading is −0.5345 %); (e) `apply_aperture`'s docstring quotes "rms over 20 sub-pixel rim placements" while the test uses 12. |
| **O8** | info | The new Klein-Cook / short-period guard fires inside an existing fixture (`tests/unit/test_audit_lens.py::test_vectorised_shape_and_order_axis`, Λ/λ = 3.16). Harmless, and it is the WP's own deferred item 2 (warning volume in wavelength sweeps) showing up in the test suite. |

---

## 6. Commands run (all with `OPENBLAS_NUM_THREADS=1`)

| command | result | duration |
|---|---|---|
| `repro/orch/verify_turb.py` | ratio 0.988 | 8 s |
| `repro/orch/verify_rt_glass.py` | bundled == catalogue on all 5 | 20 s |
| `repro/THIN-ELEMENTS-GLASS/p_turb.py`, `p_glass2.py`, `p_glass3.py`, `p_doe.py`, `p_thinlens.py`, `p_real.py`, `p_bsdf2.py` | all as quoted above | — |
| `pytest tests/unit/test_audit2609_a8_{glass,turbulence,thin_elements}.py` (as delivered) | **107 passed** | 10.9 s |
| `pytest` × 16 existing element / glass files (`test_v4_16_0_agent_d_glass_catalogues`, `test_audit_p1_glass_registration`, `test_audit_w4_glass_registry_meshgrid`, `test_v5_2_glass_formula3`, `test_v5_6_glass_memo`, `test_v5_4_6_wave3_coatings_glass`, `test_niche_audit_w3_elements`, `test_audit_w5_elements_misc`, `test_audit_w6_analysis_elements`, `test_doe_rcwa`, `test_niche_audit_w9_hfpi_doe`, `test_elements_lens`, `test_thin_lens_audit_2026_07_18`, `test_niche_audit_ec_thin_lens_claims`, `test_niche_d4_dgrating`, `test_niche_p6_astigmatic_aperture`) | **428 passed, 2 skipped** | 511 s |
| `pytest` × 7 BSDF-adjacent files (`test_audit_lens`, `test_v4_15_agent_e`, `test_v4_15_1_agent_e`, `test_v4_16_1_agent_d`, `test_v5_21_2_subsystem_audits`, `test_v5_4_6_wave5_delegated`, `test_audit_w6_raytrace`) | **2 failed**, 195 passed, 24 skipped → both failures are R1, fixed | 55 s |
| `pytest tests/unit/test_audit_except_budget.py` | **2 failed** (55 > 48) — see O1 | 2.6 s |
| `pytest tests/unit/test_audit_lens.py` (after F3) | **48 passed, 4 skipped** | 44 s |
| `pytest` × 10 files incl. the new verify file, after F1–F3 | **334 passed, 4 skipped** | 376 s |
| `python validation/run_all.py test_elements test_doe test_features test_ao test_analysis --quiet` | **ALL 5 passed** | 96 s |

Pre-existing failures NOT related to WP-A8 (other WPs are mid-edit and I did not touch
their files): `test_niche_audit_w4_input_kind.py` (the `beam_stats.py` census, WP-A8's own
request 2) and `test_audit_glass.py::…test_seidel_does_not_move_the_focus` (WP-A2's
`_lens_real.py`) — both already recorded in `WP-A8_REPORT.md` §4 and both still open. The
except-budget overshoot (O1) is partly WP-A8's and partly other WPs', with the split
measured above.

## 7. Files I changed

* `lumenairy/glass.py` — O1 (both new broad excepts narrowed), O5 (out-of-range
  catalogue lookup refuses instead of returning NaN).
* `lumenairy/elements/bsdf.py` — F1 (subclass compatibility path + Notes), O7 (the
  base-quadrature accuracy claim).
* `lumenairy/elements/elements.py` — F2 (`edge='gray'` zeroes blocked pixels; memory
  comment corrected to the measurement), O7 (the aperture docstring's rim-placement
  numbers and the circular-vs-rectangular scaling note).
* `tests/unit/test_audit2609_a8_glass.py` — O2 (three `pytest.fail` sites replaced).
* `tests/unit/test_audit2609_a8_thin_elements.py` — three test corrections (§3 items 1–3)
  plus O7 (the aperture test's quoted rms figures).
* `tests/unit/test_audit_lens.py` — F3 (three tests re-oracled; two were failing on HEAD).
* **new** `tests/unit/test_audit2609_a8_verify.py` — 36 independent pins.
* `docs/audits/.../fixes/WP-A8_REPORT.md`, `.../WP-A8_CHANGELOG.md` — O3, O5, O6, O7.

No other file was modified. No git write command was run.

---

# 8. Open-item resolution (coordinator rulings, 2026-09-12)

Rulings received after the report above was filed. O4 and O8 are left for the final
pass, as instructed. Everything else is implemented, pinned and re-measured below.

## O1 — both new `except Exception` clauses narrowed; WP-A8's census contribution is 0

**What the package actually raises**, read out of the installed
`refractiveindex/refractiveindex.py`: `KeyError` ("Material not found", line 273),
`ValueError` (unknown formula id / unit, lines 190, 226, 234), its own
`NoExtinctionCoefficient` (line 352) — and, at line 340, a **bare
`Exception("No refractive index specified for this material")`** for a page that
carries only tabulated `k`. That last one is precisely the case
`_catalogue_index_fn_from_entry` was forcing an evaluation to detect, and no narrow
except clause can name it.

**Resolution.** Both clauses now catch `_catalogue_lookup_exceptions()` — an explicit
built-in tuple (`KeyError, ValueError, TypeError, AttributeError, IndexError,
ArithmeticError, NotImplementedError, OSError`) plus the optional package's own
exception classes, resolved from the loaded module exactly as
`_missing_kappa_exceptions()` already does (and restricted to classes actually
*defined* in that module, so nothing imported into its namespace is swept in).
Measured resolution: `['KeyError', 'ValueError', 'TypeError', 'AttributeError',
'IndexError', 'ArithmeticError', 'NotImplementedError', 'OSError',
'NoExtinctionCoefficient']`.

The un-nameable bare `Exception` is **tested for instead of caught**: the page's
index model is `material._n_func`, and `_n_func is None` is exactly the condition
behind line 340. Verified on the four `main/BaF2/Bosomworth-*` pages, of which two
are k-only:

```
('main','BaF2','Bosomworth-5K')  : _n_func=None      forced eval -> Exception: No refractive index...
                                   _catalogue_index_fn_from_entry -> None   (rejected, nothing caught)
('main','BaF2','Bosomworth-80K') : _n_func=function  forced eval -> no raise
                                   _catalogue_index_fn_from_entry -> <callable>
```

`_n_func` is private, so a `_NoSuchAttributeSentinel` (CONVENTIONS §9: `__slots__`,
custom `__repr__`, `is`-comparison only) distinguishes "absent" from "None" and the
forced evaluation is kept as the fallback if a future version renames it; a bare
`Exception` escaping *there* is a loud API-drift signal, which is the right outcome
for a check that gates the whole table. A yaml parse error is deliberately outside
the tuple: a corrupt database must be loud, not filed as "row unresolvable".

**No loss of coverage**: 76 of 76 bundled rows still resolve, 0 problems, worst
|Δn_d| 4.176e-6 and worst |ΔV_d|/V_d 1.128e-4 (both N-LASF40) — identical to before
the narrowing. `verify_rt_glass.py` and `p_glass3.py` (`FLAGGED: []`) unchanged.

**Census, re-measured per commit** (the same script as §5): base `658e6142` = 48,
WP head `0067d63b` = 50, **working tree 53 with `lumenairy/glass.py: 2 → 0`**. WP-A8
now contributes **zero**. The remaining +5 over the budget are other WPs still in
flight (`_lens_imap.py` +1, `_lens_jax.py` +2, `_lens_real.py` +2,
`raytrace/exit_vertex.py` +1, `propagators/hfpi.py` −1), so
`test_audit_except_budget.py` is still red — now entirely as the cross-WP
reconciliation the tests/CI package owns. The budget constant was **not** touched.

## O2 — the `pytest.fail` sites replaced with assert-the-absence-fact

Correction to my own §5 count: there are **three** `pytest.fail` sites, not four —
the fourth `_REFRACTIVEINDEX_AVAILABLE` guard (in
`test_e1_whole_bundled_table_agrees_with_refractiveindex_info`) already used the good
pattern, which is what the three now follow.

| test | minimal-install arm now asserts |
|---|---|
| `test_e1_value_gate_rejects_the_pre_fix_rows` | that the gate is **vacuous, not wrong**: with the pre-fix rows injected, `_cross_check_bundled_values()` resolves 0 rows and reports no problems, and `_check_glass_registry_consistency(check_values=True)` does not raise on rows it cannot reach. (A gate that claimed to have checked rows it could not resolve would be the worse failure, so this is worth pinning in its own right.) |
| `test_caf2_bundled_fallback_vs_catalogue_dispatch_deviation_is_bounded` | that with no catalogue dispatch, `get_glass_index('CaF2', λ_d)` **is** the bundled Malitson row, exactly — the half of the documented split that is checkable without the package |
| `test_baf2_row_is_the_malitson_fit_its_comment_now_names` | the coefficients themselves against Malitson & Dodge 1972 as published (B = 0.643356, 0.506762, 3.8261; resonances 0.057789, 0.10968, 46.3864 µm) — the literals the corrected comment names, and the stronger of the two statements |

**Verified** by simulating a minimal install in-process (`_REFRACTIVEINDEX_AVAILABLE
= False` **and** `_ensure_refractiveindex_loaded → False`, caches cleared) and calling
all four functions directly: **4 of 4 pass, none skipped**. With the package present
the full file is unchanged at 28 passed.

## O3 + O6 — migration notes added to the changelog

* **`make_bsdf` strictness** — the entry now says explicitly that this is a
  strictness change and not only an alias addition: *any* key the chosen `kind` does
  not consume is a `ValueError`, so a spec carrying a free-text annotation
  (`'comment'`, `'source'`, `'notes'`) alongside its parameters used to be accepted
  and is now rejected; no in-repo caller is affected, prescriptions written outside
  the repo may be.
* **`'air'` lower-case-only consult** — the entry now says the registry is only ever
  consulted at the canonical key `'air'` (the name is case-folded before the
  short-circuit), so `GLASS_REGISTRY['Air'] = callable` is still silently ignored and
  the lookup returns 1.0. Re-verified: 1.0 with a mixed-case key, 1.000273975 for
  `'air'` / `'AIR'` / `'Air'` when registered at `'air'`.

The `sample_scatter_rays` subclass fallback (F1) also gained a migration note in the
same pass, since it changes what a `BSDFModel` subclass must provide.

## O5 — an out-of-range catalogue lookup raises instead of returning NaN

`get_glass_index` grew `_require_finite_catalogue_index`, applied only to the
live-catalogue (tuple-registered) return; `get_glass_index_complex` inherits it
because it obtains `n_real` through `get_glass_index`. The message carries the §2
prefix, the offending wavelength (or, for arrays, how many and their span) and the
page's own range, read from `material._wl_range` with `GLASS_VALIDITY` as the
fallback:

```
get_glass_index: the refractiveindex.info page for 'SILICON' has no data at
6.3300e-07 m; its wavelength range is [1.200e-06, 1.400e-05] m.  The page's
interpolator returns NaN outside that range rather than raising, so this is
refused instead of being handed back as a silent NaN.  ...
```

Measured, two-sided:

| call | before | after |
|---|---|---|
| `get_glass_index('SILICON', 633 nm)` | `nan` + a validity warning | **ValueError** |
| `get_glass_index('SILICON', 1064 nm)` | `nan` | **ValueError** |
| `get_glass_index('SILICON', 20 µm)` | `nan` | **ValueError** (upper edge too) |
| `get_glass_index('SILICON', 1.31 / 1.55 / 10 µm)` | 3.5003 / 3.4757 / 3.4150 | **unchanged** |
| `get_glass_index_complex('SILICON', 633 nm)` | `nan + 0j` | **ValueError** |
| `get_glass_index_complex('SILICON', 1.55 µm)` | 3.4757+0j | **unchanged** |
| array `[1.3 µm, 633 nm]` | `[3.50, nan]` | **ValueError** naming the offender |
| `N-BK7` / `CaF2` / `MgF2` / `SiO2` in range | finite | **unchanged, finite** |

Pinned by `test_audit2609_a8_verify.py::test_verify_a8_e2_out_of_range_catalogue_lookup_refuses_instead_of_nan`,
which asserts both arms, that the message names the range, and that every other
catalogue-dispatched glass is untouched. Documented as a **default change** in the
changelog with the migration note (catch `ValueError`, or range-check first).

## O7 — the five documentation inaccuracies corrected

| # | was | now, as measured |
|---|---|---|
| 1 | "the cell weights sum to `∫u du` **exactly** (so a flat lobe integrates with zero quadrature error)" | the residual is the 2-point Gauss-Legendre quartic term `dv⁴·2⁴/4320`; with `dv = ln(1e7)/128` that **predicts 9.312e-07** and the measurement is **−9.2935e-07** (agreement 0.2 %). Also recorded: the quoted "worst 1.4e-6" is the worst of *those* lobes — **1.3e-5** on `B = 1 − u²`, 6.4e-6 at l = 1e-6. Corrected in `bsdf.py`'s docstring, the report, the changelog and the test. |
| 2 | the batched rotation is "**bit-identical** to the per-ray code" | exact on **350 of 407** incidences, and within **4.441e-16** (2 ULP of a unit direction cosine) on the other 57; the cause (`np.linalg.norm` 1-D BLAS path vs ufunc reduction) is named. Corrected in the report and the changelog; the test already carries the derived `8·eps` bar. |
| 3 | MLA pre-fix peak "10.13 → 3.25 grids", 3.1× | re-measured **one implementation per process** so neither is read in the other's allocator wake: **14.13 → 3.25**, stable at N = 1024 and N = 2048, i.e. **4.35×**. The post-fix 3.25 reproduces exactly either way. Corrected in the report (table + §2), the changelog body and its heading. |
| 4 | aperture "−0.942 % → +0.038 % at D/dx = 50 px" — two grids | one grid (N = 1024, dx = 1 µm, D = 50 µm): **−0.5345 % → +0.0384 % → −0.00056 %** at 4× and 16×; and the honest summary, rms over the 12 rim placements the test sweeps: **0.386 % → 0.044 %** at 50 px, 0.031 % → 0.0041 % at 200 px. Corrected in the report (table + §2) and the changelog. |
| 5 | `apply_aperture` docstring: "rms over **20** sub-pixel rim placements", figures 0.342 / 0.059 / 0.041 / 0.0069 %, `edge_samples` ladder 0.144 / 0.054 / 0.021 / 0.0033 %, "~1/n_sub**1.5" | **12** placements (`linspace(0, 0.95, 12)`), re-measured 0.386 % hard (range −0.535 %..+0.805 %) → 0.044 % gray(4); ladder **0.108 / 0.044 / 0.021 / 0.0055 %** at n_sub = 2/4/8/16 — "a little better than `1/n_sub` and short of `1/n_sub**1.5`" — plus the new note that these are circular-rim numbers and an axis-aligned rim improves as `1/n_sub` exactly (0.639 % → 0.158 % from 4 to 16 on a 40.3 × 17.7 px stop). The test docstring carries the same re-measurement; its bars were already clear and are unchanged. |

## Follow-on found while implementing these

`edge='gray'` emitted a numpy `RuntimeWarning: invalid value encountered in multiply`
on a field carrying ±inf (from `inf * 0` on the blocked pixels). Blanking before
scaling removes it, but measured **8.0 float64 grids against 6.0 at N = 2048** — two
extra full-grid temporaries to quieten a diagnostic that the OPEN region's own
`inf * frac` raises anyway. Kept the 6-grid form and documented the trade-off in
place. A NaN field is quiet either way, and the *values* are correct in every case
(blocked pixels exactly `0j` for NaN, +inf, −inf and finite input alike). Peak
re-measured with the field hoisted out of the traced region: hard 5.00, gray 6.00 at
n_sub = 2, 4 and 8 — n_sub-independent, as the comment claims.

## Re-runs after these changes (all with `OPENBLAS_NUM_THREADS=1`)

| command | result |
|---|---|
| `pytest` × the 5 WP-A8/verify/lens files + `test_audit_except_budget.py` | **190 passed, 4 skipped**, plus the 2 expected except-budget failures (now 53, none of them WP-A8's) |
| minimal-install simulation of the 4 optional-dependency arms | **4 / 4 pass, 0 skipped** |
| `repro/orch/verify_rt_glass.py` | bundled == catalogue on all 5 glasses |
| `repro/THIN-ELEMENTS-GLASS/p_glass3.py` | `FLAGGED: []` |
| `_cross_check_bundled_values()` coverage + two-sided ladder | 76 / 76 resolved, 0 problems, ladder unchanged |
| `python validation/run_all.py test_elements test_doe test_features test_ao test_analysis` | **ALL 5 passed** |
| `pytest` × 22 collateral glass / element / BSDF-adjacent files | **575 passed, 22 skipped, 2 failed** — both traced to other WPs still in flight (below) |
| `sample_scatter_rays` perf re-check after the F1 fallback | 20 000 rays: 1.97 / 8.32 / 6.82 ms, fast path confirmed taken for all three shipped models |

### The 2 collateral failures are not WP-A8's and not mine

Both files were **green** in the §6 runs before these changes, so each was chased to
its cause rather than assumed:

1. `test_audit_w4_glass_registry_meshgrid.py::TestP242WalkerCaseInsensitive::test_main_meta_pin_passes_with_broadened_filter`
   — despite the file name, this is a meta-pin that runs
   `test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`, and that pin now reports
   `lumenairy/elements/lenses_maslov.py:205 _local_window_1d (lru_cache): cache owner
   does not call register_cache_clearer(...)`.  `lenses_maslov.py` is `M` in the working
   tree and belongs to **WP-A4**; a new `lru_cache` there needs either a
   `register_cache_clearer` registration or a cited `_CACHE_REGISTRY_EXEMPTIONS` entry.
   Nothing in glass / elements / bsdf is involved.
2. `test_v5_21_2_subsystem_audits.py::test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude`
   — `make_lg_aberration_merit_jax` returns **−7.05e13** where the pin wants ~1.  Chased
   by instrumenting every function VERIFY-A8 touched and re-running the test: it makes
   **0 calls** to `get_glass_index`, `apply_aperture` and `sample_scatter_rays`, so no
   edit in this pass can reach it.  The diverged magnitude points at `optimize/` /
   `analysis/` (both `M` in the working tree, other WPs).

Recorded for the orchestrator as **O9 (medium, WP-A4)** and **O10 (medium, owner of
`optimize/`+`analysis/` LG-JAX merit)** — neither is actionable from WP-A8's files.
