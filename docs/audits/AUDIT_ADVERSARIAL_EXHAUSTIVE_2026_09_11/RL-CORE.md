# RL-CORE audit — `lumenairy/elements/_lens_real.py`, the analytic split-step propagator

All line numbers are `lumenairy/elements/_lens_real.py` unless stated otherwise.
Every repro script is under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RL-CORE/`
(referred to below as `<S>/`). Environment: CPython 3.14, numpy 2.4.6, **numexpr NOT installed**
(so every measurement below is on the numpy fallback branch; the numexpr branch is desk-checked only).

## Scope read (line by line)

* 1–531 module head: cupy/numexpr lazy loaders, `_drop_numexpr_out_retention`, `_NUMEXPR_MIN_SIZE`,
  `_LENS_SAG_DTYPE` / `set_lens_sag_dtype` / `_resolve_sag_real`, `_resolve_sag_chunk_rows`,
  `lens_sag_float32_opd_error` (208–531).
* 1868–2136: `_propagate_through_glass`, `_check_apply_real_lens_kwarg_combination`,
  `_check_no_silent_fold_drop`, the SCREEN-OBLIQUITY derivation comment (eq. 1–7).
* 3211–3773: `_check_screen_obliquity_support`, `_WARN_STACKLEVEL`, `_AccumulatorStore` (3284–3543),
  `_VALID_*` tables, `_check_displaced_support`.
* 3773–4524: `apply_real_lens` (signature + full docstring + body).
* 4524–6663: `_apply_real_lens_impl` — **every line**, including the banded `_narrow_chunk` and
  `_slant_narrow_chunk` arms, the whole-grid surface body, the Seidel block and the obliquity guard.
* 6664–6863: `PreparedAnalyticLens`, `prepare_real_lens`.
* Dependencies as needed: `lenses.py::surface_sag_general` / `_warn_if_aperture_exceeds_grid`,
  `propagators/asm.py::angular_spectrum_propagate` + `_asm_H_from_kz` + the H-cache key,
  `glass.py::get_glass_index(_complex)`, `raytrace/trace.py::surfaces_from_prescription`.
* Cross-checked against `CONVENTIONS.md` §7, `docs/audits/AUDIT_REPORT_2026_05_16.md` (C-LR-1),
  `AUDIT_ROUND3_2026_05_16.md`, `AUDIT_VERIFICATION_2026_05_16.md` (M-LR-2),
  `BUILD_SCREEN_OBLIQUITY_2026_08_11.md`, `AUDIT_V5_2_3_2026_05_21.md`,
  `tests/unit/test_v5_2_off_axis_conic_surface_frame.py`, `tests/unit/test_audit_glass.py`.

**Not covered** (out of partition, read only far enough to know how the impl calls them):
the `displaced` / `tangent_facet(_remap)` / `screen_obliquity` model helpers at 532–3211 —
RL-MODELS owns those. I did verify the impl's *wiring* of the accumulator store and the banded
halos. I did not exercise `use_gpu` (no cupy), `wave_propagator in {'sas','fresnel','rs'}`
beyond dispatch reading, or the `freeform_type='q_bfs'/'q_con'` branch.

---

## Findings

### [P1] `slant_correction=True` uses NORMAL-referenced cosines where the exact identity needs Z-AXIS-referenced ones — the obliquity term comes out with the **wrong sign** and 1.94× the magnitude, making the OPD **2.94× worse** than leaving it off

`_lens_real.py:6156` (whole grid) and the byte-identical banded copy at `:5780`.

```python
cos_ti = 1.0 / xp.sqrt(one_plus_g)          # cos(angle between the NORMAL and z)
sin2_tt = (n1r / n2r) ** 2 * sin2_ti        # Snell from the NORMAL
cos_tt = xp.sqrt(xp.maximum(1.0 - sin2_tt, 0.0))
opd = (n2r * cos_tt_safe - n1r * cos_ti_safe) * sag
```

The module's own derivation, 4000 lines above at `:2040–2065` (eq. (3) at `:2061`, the warning at `:2052`), states the exact axial-translation
identity and explicitly warns about this: `OPD_i = (pz2 - pz1) * sag_i` with
`pz1 = n1 cos(alpha_in)`, `pz2 = n2 cos(alpha_out)`, "**both measured to the Z-AXIS (not to the
facet normal)**". For a collimated (axial) input `alpha_in = 0` and `alpha_out = theta_i - theta_t`,
so the correct coefficient is `n2 cos(theta_i - theta_t) - n1`, not `n2 cos(theta_t) - n1 cos(theta_i)`.

Expanding both to O(theta²) with n1 = 1:

* exact: `(n2-1) - theta_i^2 (n2-1)^2 / (2 n2)`
* paraxial: `(n2-1)` — missing that term
* slant: `(n2-1) + theta_i^2 (n2-1) / (2 n2)` — **opposite sign**, 1/(n2-1) = 1.94× too big

so the slant total error is `n2/(n2-1)` = **2.9414×** the paraxial error for N-BK7.

**Evidence** (`<S>/p5c_slant_sign.py`) — one N-BK7 refracting surface, plane wave in, compared
against the exact vertex-plane eikonal from an independent Newton-intersection + vector-Snell trace
(`<S>/oracle.py`). rms OPD error, piston-free:

| R [mm] | h_max [mm] | theta_i | paraxial [nm] | **slant [nm]** | eq.(3) z-axis [nm] | slant/paraxial |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 4.0 | 0.0400 | 2.993 | **8.804** | 0.0007 | 2.9414 |
| 50 | 4.0 | 0.0801 | 24.022 | **70.659** | 0.0234 | 2.9415 |
| 30 | 3.0 | 0.1002 | 35.274 | **103.760** | 0.0540 | 2.9415 |
| 20 | 3.0 | 0.1506 | 120.067 | **353.196** | 0.4179 | 2.9417 |

Predicted ratio n2/(n2-1) = 2.9414 — matched to 5 digits at every angle. The correct
coefficient is 290× to 4000× better than paraxial.

End-to-end against the same oracle at Nyquist sampling (`<S>/p5_slant.py`), rms exit-plane OPD:

| case | default [nm] | slant [nm] | gain |
|---|---:|---:|---:|
| plano-convex curved-first | 0.848 | 2.487 | 0.341 |
| plano-convex flat-first | 1.177 | 2.282 | 0.516 |
| biconvex R=±60 | 1.829 | 1.133 | 1.614 |
| biconvex R=±25 | 25.591 | 17.588 | 1.455 |
| meniscus R=20/25 | 870.79 | 875.67 | 0.994 |
| parabolic asphere (k=-1, A4, A6) | 7.764 | 22.846 | 0.340 |

The biconvex "gains" are cancellation between two oppositely-signed surfaces, not correctness.
The docstring's specific guidance ("helpful in a few specific geometries (asymmetric meniscus,
very steep asphere)") is contradicted: the meniscus is neutral (0.994×) and the steep asphere is
2.9× **worse**.

**Impact.** Any caller who turns on `slant_correction` to get closer to a ray trace gets ~3× further
away on a single powered face, and the v5.25.0 in-code comment ("cosines in the NUMERATOR ... the
plane-parallel-plate result") mis-cites the tilted-plate formula for a facet that is displaced along
**z**, not along its own normal.

**Fix (free).** `sin2_ti` and `sin2_tt` are both already materialised, so
`cos(theta_i - theta_t) = cos_ti*cos_tt + sqrt(sin2_ti*sin2_tt)`:

```python
opd = (n2r * (cos_ti_safe * cos_tt_safe
              + xp.sqrt(sin2_ti * sin2_tt)) - n1r) * sag
```

(two extra sqrt + two mults). That is exact for a collimated input; for a general input the ray angle
must come from `carrier=` / `surface_model='displaced'`, which already implement the identity.
Whichever is chosen, the flag's docstring must stop advertising the current form.

---

### [P1] `surface_frame=True` silently DELETES the surface tilt to first order — and a regression test pins the missing term

`_lens_real.py:5928–5950` (the inverse rigid-body map) and `:6056–6057` (the ramp suppression).

```python
Xs = cy_f * _dx_local + sx_f * sy_f * _dy_local
Ys = cx_f * _dy_local
h_sq = Xs ** 2 + Ys ** 2
...
if (tilt[0] != 0.0 or tilt[1] != 0.0) and not _sf_active:
    sag = sag + tilt[0] * Xs + tilt[1] * Ys        # ramp suppressed in surface frame
...
opd = (n2r - n1r) * sag                            # sag evaluated in the SURFACE frame
```

The comment says "z_s is discarded". But z_s is exactly where the tilt lives. For a surface
`z_s = g(x_s, y_s)` rigid-rotated by `R = Rx(a) Ry(b)`, the FIELD-frame height is, to first order,
`z_f(x, y) = g(x, y) - b*x + a*y`. The code evaluates `g(x_s, y_s)` and drops `-b*x + a*y`
entirely — the leading term, the one that deviates the beam.

**Evidence 1 — a tilted flat face is a literal no-op** (`<S>/p8_tilt_frames.py`). N-BK7 plate,
first face tilted 5 mrad, exit-field ray direction from the phase gradient:

```
exact thin tilted flat surface: (n-1)*theta = +2.575446e-03 rad
surface_frame=False   (Lx, Ly) = (-2.574435e-03, -2.08e-20)   <- correct to 0.04 %
surface_frame=True    (Lx, Ly) = (-1.517131e-20, -1.52e-20)   <- ZERO
max|E(tilted, surface_frame) - E(untilted)| = 0.000e+00   bytes identical: True
```

A tilted sphere behaves the same: d(OPD)/dx over ±1 mm is `-3.80e-21` where it should be
`±(n-1)*theta = 2.575e-03`.

**Evidence 2 — exact rigid-body geometry** (`<S>/p8b_surfaceframe_exact.py`). A sphere rotated about
its vertex is still a sphere with its centre at `(R sin th, 0, R cos th)`, so the exact field-frame
height is closed form. Over x ∈ [-2, 2] mm on R = 50 mm:

| tilt | max\|z_exact − z_code\| | OPD = (n−1)·d | residual after adding back −th·x |
|---:|---:|---:|---:|
| 1 mrad | 2.002 µm | 1.031 µm = **1.63 waves** | 1.66 nm |
| 5 mrad | 10.011 µm | 5.156 µm = **8.15 waves** | 9.60 nm |
| 20 mrad | 40.077 µm | 20.643 µm = **32.6 waves** | 61.5 nm |

The missing term is exactly the linear ramp.

**The existing test pins the defect.** `tests/unit/test_v5_2_off_axis_conic_surface_frame.py:263–268`
asserts `abs(slope_sf) < 0.2 * abs((n2-n1)*tx)` with the comment "the ramp is gone, only the
parabola's residual sag-vs-x remains". The test's premise — that a rigid-body rotation removes the
tilt ramp — is wrong; a rotation re-expresses the ramp, it does not delete it.
`test_tilted_parabola_branches_differ`'s "not a no-op" assertion passes only because
`Ys = cos(tx)*Y` is a second-order coordinate scaling; on a FLAT face the branch is provably a no-op.

**Impact.** `surface_frame=True` is documented as the *more accurate*, Optiland/Zemax-style treatment
("Use for off-axis aspheres, decentered parabolas"). It is strictly **less** accurate than the default
for any non-zero tilt: 8 waves of missing OPD at 5 mrad on a slow 50 mm surface. `decenter` is
unaffected (both branches agree, verified).

**Fix.** Compute the field-frame height instead of the surface-frame sag:
`z_f = (R @ (x_s, y_s, g(x_s, y_s)))_z`, i.e. `opd = (n2-n1) * z_f`. That is one extra multiply-add
per component, needs no new state, and reduces to `g - b*x + a*y` at first order. The test's two
tilt assertions then have to be re-derived.

---

### [P1] `seidel_correction=True` injects a spurious defocus and makes the field 10×–3600× worse; the docstring recommends it precisely where it is worst

`_lens_real.py:6499–6614`.

Three independent defects compound:

**(a) the fit basis starts at ρ², i.e. it contains DEFOCUS** — `:6586`
`even_powers = np.arange(2, max_order + 2, 2)` → `[2, 4, 6]`. A "Seidel-style high-order residual"
must start at ρ⁴; with ρ² in the basis every reference-frame mismatch below is absorbed as focus and
imprinted on the field. Measured fitted coefficients: plano-convex `[+4.20e-07, -5.10e-09, -8.33e-12]`
(the ρ² term alone is **420 nm** at the rim); doublet `[-5.51e-06, -5.60e-07, -1.28e-08]` (ρ² term
**-5.5 µm**). The ρ⁴/ρ⁶ terms are 1–2 orders smaller, which is why `seidel_poly_order=4` and `=6`
give identical answers to 5 digits.

**(b) the ray OPL is referenced to the LAST SURFACE, the analytic OPL to the exit vertex plane** —
`:6520` `opl_ray = final_fan.opd[alive_fan]`. `surfaces_from_prescription` gives the last surface
thickness 0, so `image_rays.opd` is the OPL at the ray's landing point ON the last surface, not
back-projected to the vertex plane. Measured (`<S>/p6b_seidel_internals.py`) against my independent
exit-vertex-plane oracle: the difference is **exactly 0.000 nm** when the last surface is flat
(plano-convex) and **−17.05 µm at h = 3.6 mm** on the cemented doublet whose last surface is
R = −291 mm. The whole last-surface sag leaks into `correction`.

**(c) what is fitted is not a residual of the wave model at all.** `opl_analytic` is
`sum (n2-n1)*sag_i(h)` with every surface evaluated at the SAME entrance height and no propagation.
The wave model does not make that approximation: the in-glass ASM carries the slab obliquity exactly.
For the plano-convex the `correction` is `337.075 nm` at h = 1.8 mm, against the analytic prediction
`n t theta^2 / 2 = 1.515 × 3 mm × (1.225e-2)^2 / 2 = 341 nm` — i.e. it **is** the in-glass obliquity
path the ASM already supplies. The measured true residual of the wave model at the exit plane on that
lens is **0.85 nm rms**. The Seidel screen adds ~160 nm rms of double count.

**Evidence — exit-plane OPD vs the independent ray oracle** (`<S>/p6_seidel.py`), rms piston-free:

| prescription | seidel OFF [nm] | seidel ON [nm] | factor |
|---|---:|---:|---:|
| plano-convex 4 mm | 0.848 | 88.87 | **105× worse** |
| biconvex R=±60 4 mm | 1.829 | 6599.4 | **3608× worse** |
| biconvex R=±30 4 mm | 14.76 | 12128.7 | **822× worse** |
| cemented doublet 4 mm | 10.86 | 311.6 | **28.7× worse** |
| cemented doublet 8 mm | 173.5 | 1430.0 | **8.2× worse** |

**Evidence — unwrap-free confirmation** (`<S>/p6d_psf.py`): propagate the exit field and scan for best
focus. No phase unwrapping, no oracle:

| | peak \|E\|² off | peak \|E\|² on | best focus off | best focus on |
|---|---:|---:|---:|---:|
| plano-convex 4 mm | 41849.17 | 40157.39 (−4.0 %) | 95.0657 mm | 96.9670 mm (**+2.0 %**) |
| doublet 4 mm | 111324.86 | 81436.98 (**−26.8 %**) | 50.6291 mm | 49.3570 mm (−2.5 %) |

The 2 % focal-length shift is the ρ² term.

**The 5 nm gate never fires as intended** — `:6601` `if corr_rms > 5e-9`. Measured `corr_rms` is
158.6 nm (plano-convex) and 2225 nm (8 mm doublet); the gate passes everything.

**The regression test that "locks in the sign" cannot fail.** `tests/unit/test_audit_glass.py:92–152`
computes `phase_diff = np.angle(np.exp(1j*phase_diff))` (so |phase_diff| ≤ π), then
`rms_waves = rms/(2π)` (so ≤ 0.5 by construction), then asserts `rms_waves < 50.0`. The assertion is
vacuous — it is what let (a)/(b)/(c) survive the round-3 sign fix.

**Impact.** The docstring says "Captures ~3-5x improvement on cemented doublets at essentially no
extra cost ... **Recommended: turn on for AC254\*-class cemented doublets**". Measured on an
AC254-050-like cemented doublet it is 8×–29× worse and costs 27 % of the focal peak intensity.

**Fix.** Either deprecate the flag and point at `apply_real_lens_traced`, or: back-project the ray
OPD to the exit vertex plane; build the model reference from the model's OWN exit-plane OPL
(which includes the exact slab obliquity, e.g. by running the analytic leg and differencing) rather
than from `sum (n2-n1) sag`; and fit from ρ⁴ up. After (a)+(b)+(c) the residual it should see is the
0.85 nm the model actually has, and the 5 nm gate would then correctly skip it.

---

### [P1] `fresnel=True` applies |t|² instead of the power transmittance `T = (n2 cos θt)/(n1 cos θi) |t|²`

`_lens_real.py:6366–6367` (whole grid) and `:5842–5843` (banded copy).

```python
T_eff = 0.5 * (xp.abs(t_s) ** 2 + xp.abs(t_p) ** 2)
E = E * xp.sqrt(T_eff)
```

`t_s`, `t_p` are AMPLITUDE coefficients. The library's `|E|²` is treated as irradiance everywhere
(the ASM is Parseval-unitary and `_propagate_through_glass` adds no impedance factor), so crossing an
index step needs the **power** transmittance, which carries an extra `(n2 cos θt)/(n1 cos θi)`.

**Evidence** (`<S>/p3_fresnel_energy.py`), N-BK7 at 632.8 nm, n = 1.515089:

| configuration | measured \|E_out\|²/\|E_in\|² | correct |
|---|---:|---:|
| single flat AIR→BK7 interface | **0.632344** | 0.958057 |
| prescription ENDING in glass | **0.632344** | 0.958057 |
| bare cemented BK7→SF11 interface | **0.846390** | 0.993599 |
| AIR→BK7→AIR plate (both faces) | 0.917873 | 0.917873 ✓ |
| R=±20 biconvex, 8 mm aperture | 0.924919 | 0.917873 (+0.77 %) |

The plate is right because the `n2/n1` factors telescope to `n_last/n_first = 1`. Everything that does
not start and end in the same index is wrong — by **−34 %** for a single air-glass interface and
−15 % for a bare cemented interface. Even for an air-to-air element the `cos θt/cos θi` half of the
factor is dropped and shows up as +0.77 % on an f/1.25 biconvex (it grows with NA).

The docstring's "~4 % loss per uncoated air-glass interface" is not what the code does per interface
(it is −37 % then +45 %); only the product is right.

Prior audit `AUDIT_VERIFICATION_2026_05_16.md:148` (M-LR-2) verified only the s/p *averaging*
(intensity vs amplitude average) and did not look at the impedance factor — so this is not a
documented accepted limitation.

**Fix.** `T_eff = 0.5*(|t_s|**2 + |t_p|**2) * (n2r * cos_tt_safe) / (n1r * cos_ti_safe)` in both
copies. That also makes the shipped per-interface loss equal the 4.2 % the docstring claims.

---

### [P1] `stop_index` out of range — or negative — silently removes ALL aperture clipping

`_lens_real.py:5478` (`if aperture is not None and stop_index is None:` → entrance aperture skipped)
and `:6390` (`if stop_index is not None and i == stop_index and aperture is not None:` → never
matches). No validation, no warning.

**Evidence** (`<S>/p7b.py`), transmitted power fraction, 3 mm `aperture_diameter` on a 5.12 mm grid:

```
stop_index=None   P/P0 = 0.26930     stop_index=2    P/P0 = 1.00000   warnings=[]
stop_index=0      P/P0 = 0.26930     stop_index=5    P/P0 = 1.00000   warnings=[]
stop_index=1      P/P0 = 0.27880     stop_index=-1   P/P0 = 1.00000   warnings=[]
                                     stop_index=-2   P/P0 = 1.00000   warnings=[]
```

`stop_index=-1` ("the last surface", the natural Python spelling) throws the stop away and transmits
3.7× the energy. `prepare_real_lens:6779` refuses `stop_index` outright, so the two entry points
disagree about the same key.

**Fix.** In the impl, right after `stop_index = prescription.get('stop_index')`:
normalise negatives (`stop_index += len(surfaces)`) and raise a precise `ValueError` when the result
is outside `[0, len(surfaces))`.

---

### [P2] `form_error` has no shape or dtype validation; a 1-D map is silently broadcast

`_lens_real.py:6062` `sag = sag + form_err` — the raw array straight out of the surface dict.

**Evidence** (`<S>/p9_formerr.py`), N = 128 field:

```
form_error shape (128, 128)  ACCEPTED  (correct)
form_error shape (128,)      ACCEPTED  -> silently broadcast across every row
form_error shape (64, 64)    ValueError: operands could not be broadcast together ...
form_error shape (128,128,1) survives the lens, dies later inside angular_spectrum_propagate
```

A `(N,)` map is an obviously-wrong figure map (one row replicated N times) and is applied without any
diagnostic; the two failing shapes produce raw numpy/downstream errors where every other input in this
module gets a precise `apply_real_lens: ...` message. Value and sign are correct for a well-shaped map
(measured Δphase = −0.511441 rad for a uniform 100 nm map vs `-k0 (n-1) 100 nm` = −0.511441).

Related, documentation-only: `form_error` is added AFTER `Xs`/`Ys` are consumed, so it is a
**field-frame** map that is never shifted by `decenter` nor rotated by `surface_frame` even though the
docstring lists it beside those keys as a per-surface property.

**Fix.** One `if np.shape(form_err) != (Ny, Nx): raise ValueError(...)` next to the other guards, and
one docstring sentence saying the map is in the field frame.

---

### [P2] `surface_sag_general` costs 5.1 full float64 grids and 4.5× the time of the same arithmetic written in place

`lenses.py:180–236`. `norm`, `valid` (bool), `denom_arg` (via `where`), `sqrt`, `1 + ...`,
`R*(...)`, `h_sq/(...)`, then `where(valid, ..., nan)` — every one a fresh full-grid allocation.

**Measured** (`<S>/p17e.py`) at N = 2048, sphere + conic:

```
surface_sag_general       248.18 ms   tracemalloc peak 171.97 MB = 5.13 float64 grids
in-place equivalent        54.82 ms   tracemalloc peak  67.11 MB = 2.00 float64 grids
```

4.5× faster, 2.6× leaner, bit-identical arithmetic. In the N=2048 cProfile of the default 3-surface
call it is **1.247 s of 5.61 s cumulative (22 %)**. At N = 32768 the difference is ~27 GB of transient
per surface on the whole-grid path (which any decenter / tilt / clear_aperture / stop / fresnel /
slant / surface-frame surface takes — `sag_chunk_rows` does not help those).

---

### [P2] No early-out for a FLAT surface on the default screen

`_lens_real.py:6158` + `:6347–6350`. A plano face has `sag ≡ 0` and `opd ≡ 0`, and the code still
evaluates `xp.exp(-1j*k0*opd)` and multiplies the whole field by ones. The tangent-facet block
(`:6170`) and the obliquity block (`:6270`) both already guard on `bool(xp.any(sag))`; the default
screen does not.

**Measured** at N = 2048, complex128 (`<S>/p7b.py`): `np.exp(-1j*k0*0)` = **471 ms**, `E * ones` =
95 ms, 134 MB of temporaries — per flat face, for an identity operation. Almost every real
prescription has at least one (plano-convex singlets, cemented planos, windows, the stop face).

**Fix.** `if not (R_y is not None or ft or form_err is not None or tilt_nonzero or asph) and
not xp.any(opd): pass` — or simply reuse the existing `bool(xp.any(sag))` reduction to skip the
screen application.

---

### [P2] `sag_chunk_rows` banding is **not** wall-clock neutral below the auto threshold

The `sag_chunk_rows` docstring says the banded path "is BYTE-IDENTICAL to the whole-grid path and
wall-clock neutral", unqualified.

**Measured** (`<S>/p17_perf.py`), 3-surface element, best of 3:

| N | whole-grid | auto | band=256 |
|---:|---:|---:|---:|
| 512 | 144.99 ms | 199.18 ms | 186.03 ms (**+28 %**) |
| 1024 | 1149.39 ms | 1173.93 ms | 1210.79 ms (+5 %) |
| 2048 | 4010.12 ms | 4282.55 ms | 4386.92 ms (+9 %) |

(Memory is the payoff and it is real: peak drops from 16.13 to 6.4–6.8 float64 grids, 2.4×.)
The AUTO default only bands at N ≥ 4096 so shipped behaviour is safe, but a caller who passes an
explicit `sag_chunk_rows` on a small grid pays up to 28 % on the strength of that sentence.

---

### [P3] `assert` used for the `thicknesses` / `surfaces` length contract

`_lens_real.py:4724`. Stripped under `python -O`. Measured with `-O`: 3 surfaces + 1 thickness →
`IndexError: list index out of range` from inside the loop (not the precise message); 2 surfaces +
2 thicknesses → silently accepted. `prepare_real_lens:6774` raises a proper `ValueError` for the same
condition, and CONVENTIONS §2 requires the `f"{fn_name}: ..."` form. Make it a `ValueError`.

### [P3] Field-frame `tilt` axis convention contradicts CONVENTIONS §7 and the surface-frame branch

`:6057` `sag = sag + tilt[0] * Xs + tilt[1] * Ys`. A right-hand rotation about **+x** by θx maps
`z=0` to `z = y tan θx` (a **y**-ramp), and about +y to `z = -x tan θy`. So the field-frame
`tilt = (t0, t1)` means `(t0, t1) = (-θ_y, +θ_x)` — components swapped and one sign flipped relative
to CONVENTIONS §7's "`tilt=(theta_x, theta_y, theta_z)` … right-hand rotation around +x, +y, +z".
The surface-frame branch (`R = Rx(tx) @ Ry(ty)`, `:5940–5946`) uses the documented convention.
Measured: field-frame `tilt=(θ,0)` deflects in **x**, `tilt=(0,θ)` deflects in **y**. Flipping
`surface_frame` therefore changes which axis the tilt is about — masked today only because the
surface-frame branch drops the tilt entirely (finding 2).

### [P3] The numexpr and numpy phase-screen paths are not bit-identical for complex64

`:6332` vs `:6347`. numexpr evaluates at complex128 internally and casts at the `out=` store; the
numpy fallback casts `phase_exp` to `E.dtype` **before** the multiply. Emulated exactly
(`<S>/p12_numexpr_emul.py`, N = 1024 so the `E.size >= 1<<20` gate fires): max relative difference
**1.05e-07**, rms 2.4e-08 — about one float32 ULP. Which path runs depends on whether numexpr is
importable and on `E.size`. Given how many byte-identity claims this module makes
(`sag_chunk_rows`, `accumulator_store`, the carrier byte-null), an undocumented
environment-dependent difference deserves a line in the docs. Corollary: `PreparedAnalyticLens`
always takes the cast-then-multiply route (`:6721` `sc = screen.astype(E.dtype); E = E * sc`), so on
a numexpr build `prepared(E) != apply_real_lens(E)` bit-for-bit for complex64 at N ≥ 1024 — it *is*
byte-identical here, where numexpr is absent (verified, `<S>/p10_banded_identity.py`).

### [P3] `absorption` uses the AXIAL gap thickness, not the local glass path

`:1910` `E = E * xp.exp(-k0 * n_medium_kappa * thickness)`. The amplitude factor itself is right
(verified to 9 digits, below), but every pixel is attenuated by the on-axis thickness. A biconvex with
6 mm centre / 5.2 mm edge thickness loses ~13 % of its absorption apodisation; the error grows with
sag. Undocumented — the docstring only says "bulk attenuation … between surfaces". Cheap partial fix:
multiply by `exp(-k0 kappa (t - sag_i + sag_{i+1}))`, which the loop already has the sags for.

### [P3] `_warn_if_aperture_exceeds_grid` is called with the wrong extent on anamorphic grids

`:4681` passes `N_grid = np.shape(E_in)[0]` (i.e. **Ny**) together with `dx`. `lenses.py:769` then
computes `grid_semi = 0.5*N*dx`. On a non-square grid or with `dy != dx` (both supported throughout
the rest of this function) the guard checks a semi-extent that exists on neither axis.

### [P3] `fresnel` takes the refraction angle from the real index only

`:6089` `sin2_tt = (n1r/n2r)**2 * sin2_ti` while `:6361–6365` use the complex `n1c`/`n2c` in the
Fresnel coefficients. Standard weakly-absorbing approximation, but "works naturally with complex
refractive indices" oversells it. Also `cos_ti` is the AOI of an **axial** ray at the local normal, so
the second surface of a converging element is given the wrong incidence — the same normal-incidence
ceiling the OPD screen has, and it is not mentioned in the `fresnel` parameter text.

### [P3] `_VALID_SCREEN_OBLIQUITY` (`:2136`) is dead

`_check_screen_obliquity_support` validates by hand (deliberately, per its comment, to keep `1`/`0`
from masquerading as `True`/`False`) and never reads the tuple. Either wire it or delete it.

---

## Performance opportunities

Baseline: 3-surface N-BK7/N-SF11 element, default path, numexpr absent, measured `<S>/p17_perf.py`.
N = 2048 whole grid: **4.01 s**, tracemalloc peak **16.13 float64 grids** (= 8.06 complex128 grids =
541 MB). cProfile top by cumulative: `_apply_real_lens_impl` 5.61 s (tottime 2.24 s),
`_propagate_through_glass` 1.84 s (2 calls), `surface_sag_general` **1.25 s (3 calls)**,
`_ifft2` 0.82 s, `_fft2` 0.35 s, `np.roll` 0.30 s (4 calls, inside the ASM's fftshift pair).

1. **In-place `surface_sag_general`** (finding above): 4.5× faster, 5.13 → 2.00 float64 grids.
   ~22 % of the N=2048 call; ~27 GB less transient per surface at N = 32768. Highest value/effort ratio
   in the partition.
2. **cos/sin phase screen instead of `np.exp` of a complex array.** Measured at N = 2048
   (`<S>/p17d_screen_bench.py`), bit-identical (`max|d| = 0.0`):
   `E * np.exp(-1j*k0*opd)` 376 ms / 3.00 complex128 grids;
   cos+sin written into a preallocated complex view then `E *= ph` **305 ms / 2.50 grids**
   (1.23× faster, −17 % peak). Separately, `E *= ph` instead of `E = E * ph` is 1.33× faster on its
   own (616 → 463 ms in a second run). At N = 32768/complex64 the half-grid saving is 8.6 GB per
   surface.
3. **Skip flat surfaces** (finding above): 566 ms and 134 MB per plano face at N = 2048.
4. **`np.exp` of a complex array is the single hottest primitive** in the whole element (471 ms for
   4.2 M points at N = 2048 = 112 ns/element). Items 2+3 together cut the screen cost roughly in half.
5. **ASM fftshift/ifftshift pair per gap.** `np.roll` shows 4 calls / 0.30 s at N = 2048, and each
   `fftshift` is a full complex copy — 2 extra complex grids per in-glass propagation (17.2 GB each at
   N = 32768/complex128). Not my partition (asm.py), but `_propagate_through_glass` is what triggers
   it; worth handing to the ASM auditor: the shift can be folded into the transfer function as a
   checkerboard sign, or the H kernel can be built in natural order (which `_get_asm_H_natural`
   already suggests exists).
6. **Fixed per-call overhead is 3.4 ms at N = 64** (200 calls in 676 ms). The validators are not the
   cost — `_apply_real_lens_impl` tottime is 0.62 ms/call; the rest is the ASM's roll/fftshift
   (0.8 ms/call) and scipy FFT setup. Matters for tolerancing/optimizer loops; `prepare_real_lens`
   does not help because it still calls `_propagate_through_glass`.
7. **H-cache across surfaces**: the key (`asm.py:341`) is
   `(Ny, Nx, dy, dx, wavelength, z, bandlimit, dtype)` and `_propagate_through_glass` passes
   `wavelength/n`, so two gaps of the same thickness AND the same glass do share a kernel — correct.
   I could not resolve a wall-clock difference between equal and unequal gaps above run-to-run noise
   on this (heavily loaded) box (4136 vs 3686 ms), so I make no claim about the size of that win.
8. `E = E_in.copy()` at `:5464`/`:5470` is required (the function must not mutate the caller's array) — not a
   waste. But the subsequent `E = xp.where(...)` at the entrance aperture, `E = E * sqrt(T_eff)`,
   `E = xp.where(sin2_tt < 1.0, ...)` and the two aperture `xp.where`s each allocate a fresh full
   complex grid where `np.multiply(..., out=E)` / `E[mask] = 0` would not. Five avoidable full-grid
   complex allocations per surface on the fresnel/slant/aperture path.

## Alternative algorithms / methods

1. **Use the module's own axial-translation identity for `slant_correction`** (see finding 1).
   Measured 290×–4000× better than the paraxial screen on a single surface, at the cost of two
   sqrt and two multiplies on quantities already computed. This is the highest-value algorithmic
   change in the partition and it needs no new machinery — the derivation, the code
   (`_facet_axial_momenta`) and the tests already exist for `screen_obliquity`.
2. **Symmetric (Strang) split-operator, half-screen / propagate / half-screen — will NOT help, and I
   can show it.** The residual is not a commutator error: measured on a SINGLE refracting surface with
   no propagation at all (`<S>/p5c`), the paraxial screen already carries 3.0–120 nm rms, and the
   full-element residual (`<S>/p1b`) scales as `0.049 · sag · NA²` with the *same* prefactor from
   R = 240 mm to R = 15 mm. It is a per-facet thickness error, so reordering the operators cannot
   remove it; only a better facet coefficient (item 1) or the tangent-facet models can. Recommend
   *not* spending effort here.
3. **WPM (Brenner & Singer, Appl. Opt. 32(26), 4984 (1993))** — propagate each plane-wave component
   with the local index. For a step-index air-glass interface this needs sub-wavelength slabs, exactly
   as the in-code comment at `:6119` already argues for BPM. The library's own `tangent_facet` /
   `tangent_facet_remap` routes reach the same accuracy for O(1) screens. No recommendation to adopt.
4. **Collins/ABCD-Fresnel (Collins, JOSA 60, 1168 (1970)) via a single Bluestein/chirp-z step for the
   in-glass gap.** The ASM is already *exact* for a homogeneous slab, so there is no accuracy gain;
   the only benefit would be a changed output pitch, which this function does not want. Not
   recommended. (It would be a real win for the AIR gaps between elements — a different partition.)
5. **1-D radial sag + interpolation for rotationally symmetric surfaces — measured NOT worth it.**
   The direct conic evaluation written with `out=` is 54.8 ms / 2.00 float64 grids at N = 2048
   (item 1 above); an `np.interp` over 4.2 M points plus the `h²` grid cannot beat that on either
   axis. The memory win people expect from a radial LUT is already available from the in-place
   rewrite plus the existing row banding. Recommend the in-place rewrite instead.
6. **Fresnel transmittance:** if polarised throughput ever matters, the right move is to stop
   averaging and route `t_s`/`t_p` through the Jones pipeline (already flagged in
   `AUDIT_V5_4_5_2026_05_26_DEEP.md`); the scalar path only needs the impedance factor (finding 4).

## Code organization observations

* `_apply_real_lens_impl` is **2140 lines** with ~25 closures defined inside it (`_obl_*`, `_tf_*`,
  `_ensure_full_grids`, `_band_any_sag`, …), all capturing ~40 nonlocals. The three surface bodies
  (`_narrow_chunk`, `_slant_narrow_chunk`, whole grid) are near-duplicates that must be kept
  byte-identical by hand; the file says so repeatedly ("keep byte-identical to the whole-grid copy
  below"). The `slant_correction` bug is duplicated verbatim in two of them, which is exactly the
  failure mode this structure invites. The three bodies differ only in how `sag` is produced and how
  the screen is applied — a single `for band in bands(...)` generator that yields `(r0, r1, sag)` and
  a whole-grid degenerate case would collapse them.
* Comment-to-code ratio in the docstring region is extreme: `apply_real_lens`'s docstring is
  **751 lines** (3773–4524) for a 40-line body, and much of it is measurement history that belongs in
  `docs/audits`. Three of the five P1s above are in behaviour the docstring describes *incorrectly*
  (`slant_correction` guidance, `seidel_correction` recommendation, `fresnel` per-interface loss), so
  the volume is not buying accuracy.
* `_check_apply_real_lens_kwarg_combination` validates `seidel_poly_order`, propagator and
  slant/propagator compatibility but does **not** reject `slant_correction=True` together with
  `seidel_correction=True`, even though the Seidel block's `opl_analytic` (`:6530`) hard-codes the
  paraxial `(n2-n1)*sag` reference and is therefore inconsistent with the slant screen that was
  actually applied. Measured (`<S>/p6_seidel.py`, 8 mm doublet): 173.5 → 1488.6 nm with both on.
* `prepare_real_lens` skips `_check_no_silent_fold_drop` and `_warn_if_aperture_exceeds_grid` that
  `apply_real_lens` runs, and refuses `stop_index` that `apply_real_lens` accepts. Two entry points,
  two validation policies.
* `_AccumulatorStore._drop` can sleep up to 0.15 s and `close()` up to 1.1 s inside a
  `weakref.finalize` callback, i.e. at an arbitrary GC point (possibly on another thread). Only on the
  opt-in memmap path, but worth a comment.
* `_LENS_SAG_DTYPE` is mutable process-wide global state with no lock; `set_lens_sag_dtype` racing a
  running `apply_real_lens` is benign only because `_resolve_sag_real` reads it once at the top.
  Documented as process-wide; no change needed, but the `PreparedAnalyticLens` freeze note is the only
  place that says so.

## Unverified suspicions

* **`_obl_q_whole()` (`:5024`) omits the `xp.asarray` promotion** that the first-call site at `:4894`
  performs, so on a CuPy backend it would return host arrays into device arithmetic. Unreachable today
  (it only runs when `_obl_q_rows_fn is not None`, which requires `_chunk_grids`, which requires
  `xp is np`), and I have no cupy to test with. Latent.
* **`sag_callable` (`:6045`) uses `np.asarray`**, which would host-copy a CuPy result. Unreachable
  because `_check_displaced_support` refuses `use_gpu` with `surface_model='displaced'`. Latent.
* The numexpr `out=Eb` writes into a *view* of `E` that the same expression reads. numexpr is
  element-wise so this should be safe, but I could not execute it (numexpr not installed) and the
  banded path does it for every band. Confirm by installing numexpr and re-running
  `<S>/p10_banded_identity.py` — the banded/whole-grid byte-identity matrix would catch a violation.
* **`screen_obliquity`'s correction did not measurably help on my fixture, but my fixture is not the
  right one to judge it by.** `<S>/p_obl.py` (tilted plane wave in, exact tilted-input ray oracle,
  piston+tilt removed, R = ±40 mm N-BK7 biconvex, 3 mm pupil) gives exit-plane rms:
  θ = 0.020 blind 2.333 nm / corrected 2.020 nm; θ = 0.050 blind 15.135 / corrected 15.812;
  θ = 0.100 blind 544.729 / corrected 536.605 — i.e. ±4 %, nothing like the 13.6× the docstring
  reports for design 121 group 5. That is **not** a refutation: this element's residual is dominated
  by the normal-incidence sag·θ² floor, which equation (4) does not touch by construction, so the
  angular term it does correct is a small part of what I measured. RL-MODELS should build a fixture
  where the angular term dominates (fast surfaces, large carrier angle, and score it the way the
  guard does) before anyone concludes either way. What this run *does* pin cleanly is the
  estimator-only contract: `screen_obliquity=False` with a carrier reproduces the carrier-free field
  to every printed digit at all three angles.
* `bandlimit=True` in `_propagate_through_glass` uses `f_max = L/(2 λ_medium |z|)` from the ASM. For a
  very thin gap (t → 0) that cutoff exceeds Nyquist and is inert, but for a thick in-glass gap at a
  large aperture it could clip real propagating content. Not measured.

## Checked and found correct

* **Default 'thin' path OPD vs an independent ray oracle** (`<S>/oracle.py` + `<S>/p1b`; Newton conic
  intersection + vector Snell + back-projection to the exit vertex plane; Nyquist-sampled from the
  traced exit NA). rms piston-free / PV: plano-convex curved-first **0.848 / 3.28 nm**; flat-first
  1.177 / 4.43; biconvex R=±60 1.829 / 6.87; cemented doublet 10.86 / 55.4 (all at 4 mm aperture).
* **The docstring's `sag·θ²` bound holds with a constant prefactor.** rms/(sag·NA²) = 0.0695, 0.0486,
  0.0480, 0.0492, 0.0516 for R = 240/120/60/30/15 mm (NA 0.0085 → 0.133, sag 8.3 → 133 µm) — a clean
  0.049 over 4 decades of residual (0.042 → 120.8 nm).
* **Sign of the imprinted phase.** Every converging element tested has φ(edge) − φ(axis) < 0
  (−676.8 rad at 0.8·ap on the plano-convex), i.e. φ = −k·n·sag under exp(+ikz) — CONVENTIONS §7.
* **In-glass propagation** (`<S>/p2_glass.py`, `<S>/p2b_glass.py`): the absolute axial piston
  reproduces `n·k0·t mod 2π` to **5.9e-13 / 1.8e-12 rad** at t = 1 / 3 mm (so chained elements do get a
  consistent absolute phase); a tilted plane wave's axial phase matches `sqrt((n k0)² − kx²)·t` to
  ≤ 1.9e-3 rad at sin θ = 0.2 and 0.6; transverse k is preserved; and the **evanescent cutoff is at
  n·k0**, not k0 (|kx| = 1.25 k0 propagates in n = 1.515 and decays in air; 1.8 k0 decays in both).
  The field ends on the last surface's vertex plane with no trailing thickness applied, consistent
  with `len(thicknesses) == len(surfaces) - 1`.
* **`absorption=True` amplitude factor** (`<S>/p4`): measured `|E| = 0.284609543` against
  `exp(-k0·κ·t) = 0.284609543` (9 digits) for κ = 1e-4, t = 2 mm, λ = 1 µm, i.e. intensity
  `exp(-4πκt/λ)`. No attenuation after the last surface (measured exactly 1.0). The gap's κ comes from
  `n2c.imag` of the gap medium, verified with a two-medium stack.
* **Row-banded vs whole-grid BYTE identity** (`<S>/p10_banded_identity.py`, `np.array_equal` on the
  raw `uint8` view): held for **5 prescriptions × 7 band sizes** (plain 3-surface conic+asphere;
  decentered mid surface; per-surface `clear_aperture`; mid-train `stop_index`; a MIXED element where
  surface 0 bands, surface 1 falls through to whole-grid on decenter and surface 2 on clear_aperture)
  at `sag_chunk_rows ∈ {None, 1, 7, 64, 256, 1024, 4096}`, plus the `_slant_narrow_chunk` arm for
  `fresnel` / `slant` / both × {1, 33, 256}. **No state leak** between banded and whole-grid surfaces.
* **`accumulator_store='memmap'` is byte-identical to `'ram'`** on the `tangent_facet` route with a
  finite-radius carrier, and the scratch directory is empty after a normal return **and** after a
  ValueError raised mid-prescription (the `with` block in `apply_real_lens` does its job on Windows).
* **`prepare_real_lens` reproduces `apply_real_lens` byte-for-byte** (complex128 and complex64) on this
  numexpr-free build; the documented freeze w.r.t. the glass catalogue and w.r.t. later mutation of the
  prescription dict behaves exactly as the docstring says.
* **complex64**: max relative field error **6.7e-07** vs complex128 on a 3-surface, 5 mm-thick element
  (phase rms 1.7e-3 rad). The dangerous piece — the ~4.5e+04 rad `n·k0·t` piston — is folded
  `mod 2π` inside `asm.py::_asm_H_from_kz` *before* the float32 cast, so it does not degrade; the
  screen `exp` is evaluated in complex128 and only the stored product is narrowed. Output dtype
  follows the input.
* **Mirror guards**: `is_mirror=True` and `glass_after='MIRROR'` both raise a precise ValueError from
  `_apply_real_lens_impl`; `_check_no_silent_fold_drop` fires for `elements`-borne mirrors and honours
  `allow_unfolded_equivalent`.
* Biconic `radius_y` produces a genuinely anamorphic screen; anamorphic `dy != dx` is accepted and the
  surface-normal gradient correctly uses `(dy, dx)` in that order.
* `_check_screen_obliquity_support`'s value-vs-identity validation (`screen_obliquity == 'auto'` but
  `is True` / `is False`) is right, and `_check_displaced_support`'s refusal matrix is consistent with
  the models' documented envelopes.
* `_AccumulatorStore` hands out `np.asarray(mm)` base-class views, so no ufunc can dispatch on the
  memmap subclass; the per-view `weakref.finalize` reaper does release mappings before `close()`.
