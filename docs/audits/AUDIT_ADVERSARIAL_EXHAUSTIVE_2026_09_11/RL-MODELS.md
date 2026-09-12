# RL-MODELS audit — opt-in refraction-OPD models of `apply_real_lens` (`displaced` / `tangent_facet` / `tangent_facet_remap`, `carrier=`/`screen_obliquity`, their caches)

All measurements on the stated environment (CPython 3.14, numpy 2.4.6, scipy 1.17.1, Windows 11).
Every repro script is under
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RL-MODELS/`.
Repo files were only read; nothing in the repository was created, modified or deleted.

## Scope read (line by line)

`lumenairy/elements/_lens_real.py`:
* 1–260 module header, cupy/numexpr shims, `_LENS_SAG_DTYPE` / `set_lens_sag_dtype` / `_resolve_sag_chunk_rows`
* 208–531 `lens_sag_float32_opd_error` (probe 8)
* 481–800 the `displaced` derivation block, `_build_displaced_cos_luts`, `_displaced_opd`, `_displaced_carrier_slope_fn`, `_DISPLACED_LUT_CACHE` + `_displaced_geom_key` + `_get_displaced_cos_luts`
* 800–1340 `_displaced_carrier_dir_fn`, `_disp_surface_z_grad`, `_build_displaced_cos_grid` (`_interp2_structured` / `_interp2_delaunay` / `_upsample_coarse`), `_DISPLACED_COS_GRID_CACHE` + budget API + `_displaced_cos_grid_key` + `_get_displaced_cos_grid`, `_element_is_asymmetric`, `_resolve_displaced_obliquity`
* 1340–1830 `_displaced_eikonal_fn`, `_displaced_carrier_dir_eik_fn`, `_build_displaced_ray_map`, `_apply_displaced_remap`, `_build_displaced_ray_map_2d`, `_apply_displaced_remap_2d`
* 1830–2030 `_propagate_through_glass`, `_check_apply_real_lens_kwarg_combination`, `_check_no_silent_fold_drop`
* 2030–2605 the screen-obliquity derivation, `_SCREEN_OBLIQUITY_*` constants, `_screen_obliquity_angle_field`, `_facet_axial_momenta`, `_screen_obliquity_delta`, `_screen_coeff_error`, `_screen_drift_step`, `_screen_drift_opd(_rows)`, `_screen_obliquity_row_evaluator`, `_screen_obliquity_rows_any`, `_screen_obliquity_pupil_radius`, `_screen_obliquity_rms_waves`
* 2605–3210 route-3 derivation + halo derivation, `_tf_sl`, `_tf_rows_grad`, `_tangent_facet_screen(_rows)`, `_tangent_facet_transport(_rows)`, the remap derivation, `_tangent_facet_remap_screen`, `_tf_remap_quadratic_eikonal`, `_tf_remap_phi`, `_tangent_facet_remap_apply`
* 3210–3280 `_check_screen_obliquity_support`; 3544–3780 `_check_displaced_support`
* 4090–4420 the public docstrings for `surface_model` / `conjugate` / `displaced_mode` / `displaced_obliquity` / `carrier` / `remap_order` / `screen_obliquity` / `on_screen_obliquity`
* dispatch sites in `_apply_real_lens_impl`: 4593–4630 (gates), 4690–4900 (model set-up, `_r_max`, grids, `_obl_*` seeding), 5014–5250 (`_obl_band_*`, `_obl_gap_advance`), 5250–5455 (`_tf_*` banded closures, `_tf_price_walk_halo`), 5455–5600 (entrance aperture, remap early returns, `_narrow_chunk`), 5660–5740 (`_slant_narrow_chunk` gate), 5900–6060 (whole-grid sag/decenter/tilt/sag_callable), 6090–6480 (screen dispatch, tangent-facet block, obliquity block, phase screen, remap apply, gap transport), 6620–6660 (the guard)

Docs read: `docs/audit_real_lens_displaced_2026_07_19.md` (P2/P3/P10 sections), `docs/audits/BUILD_TF_BANDED_2026_08_16.md`, `BUILD_TF_REMAP_2026_08_16.md`, `BUILD_TANGENT_FACET_2026_08_16.md` (grep + targeted reads), plus greps of `CHANGELOG.md` and `docs/audits/*`.

**Not reached:** `_AccumulatorStore`'s memmap internals (3420–3544, only its API contract was read); the `'auto'` carrier FIT itself inside `_lens_traced._compute_carrier` beyond the branch that fixes the units question (finding 2); the Seidel / Maslov blocks; the GPU (`cp`) arms (uninstallable here) — all desk-checked only.

---

## Findings

### [P1] Both `displaced` remap paths silently DISCARD the input field's phase — and the 2-D one is the DEFAULT routing for any asymmetric element
`lumenairy/elements/_lens_real.py:1612` (`amp = map_coordinates(np.abs(E_in), ...)`), `:1809` (same, 2-D), routing at `:4756-4761` and the early returns at `:5492` / `:5510`.

**What is wrong.** `_apply_displaced_remap` and `_apply_displaced_remap_2d` sample only `np.abs(E_in)` and rebuild the exit phase from the ray eikonal, `E_out = amp · jac · exp(+i k0 (opl − opl_ref))`. The eikonal is the trace of the congruence named by `conjugate=` — **default `None` = collimated**. Any phase the caller's field actually carries (curvature, tilt, aberration, an upstream element's residual) is thrown away with no warning. The private docstring of `_apply_displaced_remap` says "assumes the input phase matches the specified congruence", but the **public** `conjugate` docstring (`:4248-4250`) asserts the opposite:

> "The wave field itself (`E_in`) already carries the input curvature in its phase -- `conjugate` ONLY informs the per-surface obliquity cosines; it adds no reference phase and does not modify `E_in`."

That is true of the screen paths and false of the remaps. And this is not an obscure opt-in: `_disp_2d_remap = _disp_asym and (displaced_mode == 'remap' or (displaced_mode == 'screen' and displaced_obliquity == 'auto'))` — i.e. with the **documented defaults** `displaced_mode='screen'`, `displaced_obliquity='auto'`, any surface carrying a `decenter`, a `tilt` or a `sag_callable` routes to the phase-discarding 2-D remap.

**Evidence** (`t5_phase_discard.py`, `t6_phase_discard2.py`):

```
input phase p-v over pupil: 35.40 waves (200 mm defocus + 4 mrad tilt)
                                        max|E(flat) − E(phased)| / max|E|   phase-diff ptp
1-D remap (displaced_mode='remap')                       4.749e-16            0.0000 rad
2-D remap (DEFAULT for a decentered element)             4.749e-16            0.0000 rad
pointwise SCREEN (displaced_obliquity='pointwise')       1.977e+00            6.2831 rad
thin (reference)                                         1.978e+00            6.2823 rad
displaced screen (symmetric)                             1.978e+00            6.2830 rad
```

Physical consequence — a 150 mm **diverging** source through a decentered N-BK7 singlet (f = 22.1 mm), best-focus z found by a 36-plane ASM scan:

```
thin                                            25.000 mm   (correct: 25.93 mm predicted)
displaced 2-D remap, conjugate=None (DEFAULT)   21.000 mm   <-- the COLLIMATED focus
displaced 2-D remap, conjugate='auto'           25.000 mm
displaced 2-D remap, conjugate=+0.150           25.000 mm
pointwise SCREEN, conjugate=None                25.000 mm
```

A 4 mm / 19 % focal-position error, silent.

**Impact.** Any chained use (`apply_real_lens` on element *k* of a multi-element system, where the incoming field carries element *k−1*'s exit wavefront) is wrong on the default asymmetric path: the element is re-illuminated by an idealised collimated congruence. The failure is invisible — no warning, no NaN, no energy loss.

**Fix.** (a) Correct the public `conjugate` docstring and add the limitation to `displaced_mode` / `displaced_obliquity`. (b) Carry the input phase: the remap already resamples at the entrance ray positions, so `amp_in` can become `E_in` sampled complex (with the ray eikonal ADDED to `angle(E_in)/k0` rather than replacing it) — interpolate `|E|` and `unwrap(angle(E)) − k0·W_conj` separately, as the 2-D path already does for amplitude and OPL. (c) Failing that, measure the residual `angle(E_in)/k0 − W_conj(x,y)` over the bright support and raise/warn above a fraction of a wave, with a policy kwarg — the library already has `_guard_dispose` for exactly this shape of guard.

---

### [P1] `_screen_obliquity_angle_field` multiplies by `n1` for *every* carrier vocabulary, double-counting it for `carrier='auto'` and `carrier=<ndarray>`
`lumenairy/elements/_lens_real.py:2244` (collimated `TiltedCarrier` fast path) and `:2250-2251`.

**What is wrong.** The function's own docstring is explicit that the consumer needs the transverse **optical momentum** `q = n1·(L, M)` and that `(L, M)` from `_compute_carrier` are **unit-ray direction cosines**. That holds for the two *geometric* congruences — a `TiltedCarrier` (whose `L`, `M` the traced module launches unit rays along) and a scalar conjugate (whose `W = sign(s)(sqrt(x²+y²+s²) − |s|)` has `grad W = sin α`). It does **not** hold for the other two:

* `carrier='auto'` — `_compute_carrier` fits `Lx = angle(E[:,1:]·conj(E[:,:-1])) / (k0·dx)` (`_lens_traced.py:4584`). The field's phase is `k0·S` with `S` the *optical* path, so this reading **is** `p_x` already.
* `carrier=<ndarray>` — the wavefront `W` is documented as "reference phase = `k0 * W`", so `grad W` is likewise the optical momentum.

Multiplying either by `n1` over-counts by exactly `n1`.

**Evidence** (`t19_units.py`, `t20_units2.py`). Field = an exact plane wave in N-BK7 at 50 mrad *in the glass*; true transverse optical momentum `n1·sin θ = 0.075808`:

```
TiltedCarrier(L=sin θ)  [direction cosine]  -> qx = 0.075808   OK
TiltedCarrier(L=n1 sin θ)                   -> qx = 0.114986   x1.5168
'auto'  (fit of the real in-glass field)    -> qx = 0.114986   x1.5168
ndarray W = p_true * X                      -> qx = 0.114986   x1.5168
```

Downstream, on an immersed R = 25 mm N-SSK-class surface (n1 = 1.5168 → n2 = 1.0), the applied correction (eq. 4) against the exact `(T1)` target:

```
theta   correction with the TRUE q   with the shipped n1*q   ratio   residual vs exact (T1)
0.020   0.00700 waves rms            0.01170 waves rms       1.67    0.0000 -> 0.0050 waves
0.050   0.02391                      0.04770                 1.99    0.0000 -> 0.0247
0.100   0.07839                      0.17426                 2.22    0.0000 -> 0.0971
```

With the correct `q` the residual is exactly zero (eq. 4 reduces to `(T1)` on a single surface — verified separately in `t11_obl_newton.py`). With the shipped `q` the "corrected" screen at 100 mrad carries **0.0971 waves** — *worse than the uncorrected screen's 0.0784*. The same inflated field feeds `_obl_total`, so the accuracy guard is inflated too.

**Impact.** Only bites when `surfaces[0]['glass_before']` is not air — which is precisely the case the `n_medium` argument was added for, and the docstring's own measured table (`air / N-BK7 / N-SF57`, "2.2x -> 474x") was evidently taken with a `TiltedCarrier`. For an immersed prescription driven by `carrier='auto'` (the natural choice when the incoming field comes from an upstream element) the correction is 1.7–2.2x too large.

**Fix.** Normalise the convention in one place. Either have `_compute_carrier` return a geometric gradient for all four branches (divide the `'auto'`/ndarray gradients by the medium index, which the caller must then supply), or — cheaper and local — have `_screen_obliquity_angle_field` apply `n1` only for the `TiltedCarrier` and scalar-conjugate branches and pass the `'auto'`/ndarray gradients through unscaled, with a comment recording why the two families differ. Add a two-line unit test with a genuine in-glass plane wave (the one above) so the two conventions stay pinned.

Related, same class, lower exposure: `_displaced_carrier_slope_fn` (`:722-737`) and `_displaced_carrier_dir_fn` (`:875-889`) feed `grad_fn`'s output into `(dz, dy) = (1, g)/sqrt(1+g²)`, i.e. treat `g` as a **tangent**; for the scalar conjugate `g = h/s` is a tangent (correct), for `'auto'` it is `sin α` (a third-order error, ~1.3e-3 relative at 50 mrad) and, in glass, additionally off by `n1`.

---

### [P1] `_DISPLACED_COS_GRID_CACHE` returns a stale grid when a `sag_callable`'s internal state changes (identity keying)
`lumenairy/elements/_lens_real.py:1271` (`s.get('sag_callable'),   # by identity (held -> no GC)`), used by `_get_displaced_cos_grid` (`:1281`).

**What is wrong.** The key docstring reasons only about the *miss* direction ("a fresh callable each call simply misses (correct -- two callables cannot be proven equal)"). The *hit* direction is unguarded: object identity does not imply value equality for a **mutable** callable, and the cache is sold for exactly the workload that mutates one — "a decentered-design iteration loop that only moves the field re-uses the ~3.9 s trace".

**Evidence** (`t7_misc.py` §C, `t8_follow.py` §C2/C3). A freeform `class FF: __call__ = lambda self,x,y: self.a*(x²+y²)` whose `a` goes 5.0 → −5000.0 between two `apply_real_lens(..., surface_model='displaced', displaced_obliquity='pointwise')` calls, with the cache enabled (`set_pointwise_cos_grid_cache_budget(64)`):

```
_build_displaced_cos_grid depends on the state:  a=5.0 -> surf0 cos_out mean 0.999999745
                                                 a=-5000 -> surf0 cos_out mean 0.894790600
public API:  |E_stale - E_correct|max / |E_correct|max = 1.6377e+00   (164 % of peak)
             |E(a=5) - E(a=-5000)|max / |E|max         = 3.0919e+00   (the real change)
```

**Impact.** Silently wrong exit field, at full amplitude scale, in a design loop. Mitigated only by the cache shipping at `max_bytes=0`; it is one `set_pointwise_cos_grid_cache_budget(...)` call away from live.

**Fix.** Key on a *value* fingerprint rather than identity: e.g. `(id(cb), getattr(cb, '__lumenairy_version__', None))` plus a cheap probe — evaluate the callable on a small fixed stencil (say 8 points spanning `±r_max`) and hash the resulting float64 bytes into the key. That costs microseconds against a multi-second trace and makes a state change a miss instead of a stale hit. Failing that, document loudly on `set_pointwise_cos_grid_cache_budget` that a `sag_callable` must be treated as immutable while the cache is enabled, and offer `clear_pointwise_cos_grid_cache()` as the required per-iteration call.

---

### [P2] `tangent_facet_remap`'s fold / pull-back guards are WHOLE-GRID reductions, so ordinary padded grids are refused on dark corner pixels
`lumenairy/elements/_lens_real.py:3107` (`d_min = float(np.min(det))`) and `:3145-3160` (`step = max(float(np.max(np.abs(nix - ix))), ...)` over the full grid).

**What is wrong.** `det(I + dW/dx)` is evaluated over every pixel of the grid, including the region outside the clear aperture where the field is exactly zero (the entrance aperture at `:5486-5490` has already zeroed it) and where `sag(r)` and `grad sag` grow without bound. A converging beam needs a padded grid; padding therefore *causes* refusals.

**Evidence** (`t9_fold.py`). N-BK7 biconvex, R = +19.6 / −27.4 mm, t = 2.5 mm, **`aperture_diameter = 2.0 mm`**, collimated Gaussian w0 = 0.67 mm:

```
 pad   N      dx[um] window[mm] corner[mm]  result
  2.0  1024     4.0      4.10      2.90     OK
  4.1  2048     4.0      8.19      5.79     OK
  8.2  4096     4.0     16.38     11.59     REFUSED: "the transverse-walk map folds.  min det = -0.867747"
  4.1  1024     8.0      8.19      5.79     OK
  8.2  2048     8.0     16.38     11.59     REFUSED: min det = -0.821741

min det, surface 1, N = 1024:        whole grid      inside the 2 mm pupil
   dx =  4 um                           0.9883             0.9986
   dx =  8 um                           0.9508             0.9986
   dx = 16 um                           0.7447             0.9986
   dx = 24 um                          -0.6622             0.9986
```

An 8x pad — a completely standard choice for a converging beam — is refused while the illuminated pupil sits at `det = 0.9986`, three orders from the `1e-4` bar. At `dx = 30 um` the *pull-back* guard fires instead ("did not converge in 64 iterations"), same cause.

The refusal message blames the physics and tells the user to change model ("use `surface_model='tangent_facet'`, `apply_real_lens_traced`, or `apply_real_lens_maslov`"); the actual remedy is to shrink the grid. That is a misdiagnosis, and the code has the pupil in hand (`prescription['aperture_diameter']`, and `_screen_obliquity_pupil_radius` already implements the aperture→semi-diameter→inscribed-radius fallback).

**Fix.** Score both reductions over the support that matters. Minimal change: build `pup = h_sq_axis <= r_pupil**2` (reusing `_screen_obliquity_pupil_radius`) or, more robustly, `supp = np.abs(E) > 0`, and take `d_min = float(np.min(det[supp]))` / `step = max(np.max(np.abs((nix-ix)[supp])), ...)`. Outside the support, clamp `W` to zero (the field there is zero, so the walk is unobservable) so the resampling stays well-defined. Keep the whole-grid `det` in the message as a diagnostic.

---

### [P2] `_build_displaced_ray_map` / `_build_displaced_ray_map_2d` hard-code `n = 1` for the exit referencing leg
`lumenairy/elements/_lens_real.py:1566` and `:1766` — `opl = opl + 1.0 * t_f    # exit gap is air (n = 1)`.

**What is wrong.** After the last surface the ray is walked back (or forward) to the exit vertex plane `z = sum(thicknesses)`; that leg is travelled in `surfaces[-1]['glass_after']`, not necessarily air. `t_f` is of order the last surface's sag, so the error is `(n_after − 1) · |sag_last| / cos`.

**Evidence** (`t8_follow.py` §D2). Same singlet with `surfaces[-1]['glass_after'] = 'N-BK7'`, r_max = 1 mm, compared against an independent meridional trace with the correct exit index:

```
model OPL vs manual trace, n_exit = n(glass_after) = 1.5168 :  max|d| = 8.6723 um = 15.77 waves
model OPL vs manual trace, n_exit = 1.0                     :  max|d| = 0.0000e+00 um
```

i.e. the function reproduces the `n=1` trace exactly and is 15.8 waves away from the correct one.

**Impact.** Wrong exit wavefront for any `displaced_mode='remap'` (1-D or 2-D) call on an immersed-exit prescription. Rare, but nothing refuses it.

**Fix.** `n_exit = float(get_glass_index(surfaces[-1]['glass_after'], wavelength)); opl = opl + n_exit * t_f` in both functions. `idx[-1][1]` is already computed.

---

### [P2] `_DISPLACED_LUT_CACHE` (on by default) keys glasses by NAME, not by resolved index
`lumenairy/elements/_lens_real.py:767` (and the same at `:1272` for the cos-grid key).

**What is wrong.** `str(s.get('glass_before')), str(s.get('glass_after'))` identifies the glass by its registry key. `lumenairy.glass.GLASS_REGISTRY` is a documented *mutable* user extension point (`GLASS_REGISTRY['MY_GLASS'] = lambda wl: ...`), so re-pointing an entry under the same name leaves the key unchanged. Nothing invalidates `_DISPLACED_LUT_CACHE` on a registry write.

**Evidence** (`t13_glasscache.py`): `GLASS_REGISTRY['MYGLASS']` moved 1.50 → 1.90 between two calls (glass value cache cleared in between so the *index* is re-read correctly):

```
cached LUT == pre-repoint LUT:        True
cold rebuild differs from cached:     True   max|dcos_in| = 1.476e-04
public API: |E_stale - E_correct|max / |E|max = 1.0558e-02
```

**Impact.** 1 % of peak amplitude — bounded, because only the obliquity *cosines* are stale (`n1r`/`n2r` are re-resolved each call). Still a silently wrong result from an always-on cache.

**Fix.** Key on the resolved values, not the names: `float(get_glass_index(s['glass_before'], wavelength))`, likewise `glass_after`. That is one lookup per surface (already memoised by `_GLASS_VALUE_CACHE`), removes the name-aliasing hazard entirely, and is strictly more correct (two names with identical index at this wavelength then share an entry).

---

### [P2] The 2-D displaced remap's transverse resolution is fixed by a hard-coded `n_side=181`, independent of N and dx
`lumenairy/elements/_lens_real.py:1650` (`n_side=181`), consumed at `:1830-1852`.

**What is wrong.** `_apply_displaced_remap_2d` reconstructs the **entire** exit field by Delaunay-interpolating `(amp_out, OPL)` from at most 181² scattered exit points onto the full `N²` grid. For a 2 mm aperture the launch pitch is 11.4 µm regardless of whether `dx` is 8 µm or 0.5 µm. `n_side` is not exposed as a kwarg.

**Evidence** (`t18_last.py` §A, `t17_res.py`). Input `Gaussian · (1 + 0.5 cos(2πr/Λ))`, N = 1024, dx = 2 µm, decentered singlet; residual ripple contrast of `|E_out|` on the central row:

```
ripple 200 um (17.5 launch samples/period):  screen 0.1027   remap 0.1060
ripple  60 um ( 5.2 launch samples/period):  screen 0.3849   remap 0.5459
ripple  25 um ( 2.2 launch samples/period):  screen 1.2623   remap 0.5063
```

At 2.2 samples/period the remap has lost/aliased structure the screen path (which lives on the field grid) resolves. The 60 µm row also shows the other half of the trade: the remap is a *purely geometric* transfer with **no in-glass diffraction at all**, so it over-preserves a ripple the screen+ASM chain correctly washes out.

**Impact.** Structured pupils (hard stops, obscurations, segmented apertures, speckle, an upstream DOE) are smoothed to the 181-ray lattice. Nothing warns.

**Fix.** Scale `n_side` with the grid — e.g. `n_side = min(1025, max(181, int(2*r_max/dx) + 1))` — and expose it. Better still: the launch fan is a **regular** grid, so the exit map is a smooth curvilinear grid and the module *already contains* the right machinery (`_interp2_structured`, `:1046-1090`) to invert it with `map_coordinates` instead of QHull. Reusing it would remove both the triangulation build (measured 16.4 s vs 8.4 s for structured on the cos grid) and the resolution ceiling.

---

### [P2] The `displaced` remaps have no fold guard at all, while `tangent_facet_remap` refuses — inconsistent safety for the same hazard
`lumenairy/elements/_lens_real.py:1825` (`jac_amp = 1.0 / np.sqrt(np.maximum(np.abs(det), 1e-30))`) and `:1593` (`keep = np.concatenate(([True], np.diff(ho) > 0))`).

`_apply_displaced_remap_2d` takes `|det|`, so an orientation-**reversed** (folded) cell still contributes amplitude, and `LinearNDInterpolator` over a self-overlapping point set returns whichever branch the triangulation picked. `_apply_displaced_remap` silently drops every ray whose exit height is not strictly increasing — a fold is discarded without a diagnostic — and then computes `np.gradient(ho, hi)` on an `hi` that is no longer monotone once the map has folded. The module's own remap rung (`:3096-3120`) argues at length that a folded map "is not approximated, it is declined"; the same argument is not applied here.

I did **not** reproduce a fold on the prescriptions tried (`t18_last.py` §B: at `conjugate = None / −3.0 mm / −1.5 mm`, all 1025 fan rays survive), so this is reported as a missing guard rather than an observed wrong answer.

**Fix.** Add the same `min(det) > 0` / `all(diff(ho) > 0)` refusal (or at minimum a `RuntimeWarning` naming the number of dropped rays and the minimum determinant) to both displaced remaps, with the same `on_*` policy vocabulary the module already uses.

---

### [P2] The pointwise obliquity cos-grid is built on a fixed 384-sample coarse grid spanning the FIELD extent, so its accuracy degrades linearly with grid padding
`lumenairy/elements/_lens_real.py:914` / `:1283` (`n_coarse=384`), `:985-990` (`xcoarse = np.linspace(xax[0], xax[-1], _ncx)`).

The comment claims "``n_coarse`` samples over the field extent resolve the aperture-scale cos variation to well under the obliquity tol". That is true for an unpadded grid and false in proportion to the pad factor: the 384 samples are spread over `N·dx`, not over the traced aperture.

**Evidence** (`t14_coarse.py`), fixed 2 mm pupil, cos grid compared against the same trace at `n_coarse=4096`:

```
window[mm]  coarse pitch[um]  samples across the 2 mm pupil |  max|Δcos_out| inside the pupil
     2.048            5.35                         374.0    |  1.514e-06
     4.096           10.69                         187.0    |  2.251e-06
     8.192           21.39                          93.5    |  4.831e-06
    16.384           42.78                          46.8    |  1.846e-05
    32.768           85.56                          23.4    |  3.632e-05
```

24x degradation over a 16x pad. At `sag ≈ 20 µm` and `n ≈ 1.5` that is ~0.002 waves of OPD — small in absolute terms, but the same order as the 0.001-wave bar the tangent-facet ladder is graded against, and it grows with padding rather than with anything physical.

**Fix.** Build the coarse grid over the traced extent (`±r_max`, padded a few cells) and clamp outside it, or set `_ncx = min(Nx, max(384, int(384 * (Nx*dx)/(2*r_max))))`. Either decouples the accuracy from the pad factor.

---

### [P2] Every geometric trace in the partition runs a FIXED 24 Newton iterations; the residual is exactly zero after 2
`lumenairy/elements/_lens_real.py:598`, `:1105`, `:1517`, `:1721` — all `for _ in range(24):` with no residual test.

**Evidence** (`t11_obl_newton.py`), replicating `_build_displaced_cos_grid`'s intersection loop on a 257² launch fan against a spherical first surface:

```
iter  1: max|residual| = 1.231e-04 m
iter  2: max|residual| = 0.000e+00 m      (and every iteration thereafter)
_build_displaced_cos_grid(n_launch=257, N=1024) : 8.434 s   (interp_method='structured')
                                  ... 'delaunay': 16.440 s
```

Each iteration of the 2-D loops calls `_disp_surface_z_grad`, which itself evaluates `_surface_sag_general` **three** times (value + the two finite-difference arms). 24 iterations × 3 evaluations × 66 049 rays × 2 surfaces is ~9.5 M sag evaluations where ~0.8 M would do. A `if max|g| < tol: break` (or a residual-masked update) is a ~10x saving on the dominant cost of the pointwise path.

---

### [P3] `lens_sag_float32_opd_error`'s `on_partial_aperture` warning is unreachable with default arguments, contradicting its own docstring
`lumenairy/elements/_lens_real.py:401-403` (`dx_fc = ... else float(ap) / (0.8 * n_fc)`) and `:420-424` (`cover = window / ap`; `covers = cover >= 1.0`).

With `field_check_dx=None` the pitch is chosen so the aperture spans 80 % of the window, hence `cover` is **always** 1.25 and `covers` is always `True`. The warning therefore never fires on a default call, yet the docstring says:

> "`'warn'` is the default because the shipped default `field_check_n=512` is such a proxy on any real lens, and reading its `ok` as a production sign-off is the mistake this exists to stop."

**Evidence** (`t17_res.py`):

```
default call:   aperture_cover=1.250  covers=True   warnings=0  ok=True   field_check_dx=4.883 um
production dx:  aperture_cover=0.230  covers=False  warnings=3  ok=True
```

The docstring also contradicts itself: its own "WHAT COVER >= 1 BUYS" paragraph argues that cover ≥ 1 is the sufficiency criterion, which the default satisfies. The substantive gap is that the guard checks **cover** while the default configuration's actual weakness is **pitch** (4.88 µm here against a production 0.9 µm), and the docstring's own "IMPORTANT: the field-level error is CONFIG-DEPENDENT" says pitch is what matters.

**Fix.** Either drop the claim that the default warns, or add a second condition on the pitch (e.g. warn when `field_check_dx` was auto-chosen at all, since that is by construction not the caller's production sampling).

---

### [P3] Dead branch: `_split_mode` inside `_obl_gap_advance`
`lumenairy/elements/_lens_real.py:5191-5192` — `if _split_mode: _t_gap, _n_gap = _t_gap / n2r, 1.0`.

`_obl_gap_advance` runs only when `_obl_active`, which requires `carrier is not None`; `_check_screen_obliquity_support` (`:3258-3267`) raises `ValueError` for `carrier=` with any `surface_model != 'thin'`, and `_split_mode` requires `surface_model == 'displaced'`. The branch is unreachable. The comment above it ("the 'split' factorisation drifts through its reduced distance", `:6437`) describes behaviour that cannot occur.

---

### [P3] `_tangent_facet_remap_apply` uses a different coordinate-origin convention from the rest of the module
`lumenairy/elements/_lens_real.py:3127-3128` — `x_ax = (np.arange(nx) - nx // 2) * dx` (integer `//`), against `x = (xp.arange(Nx) - Nx / 2) * dx` everywhere else (`:4835`). Identical for even `N`, off by `dx/2` for odd `N`. It is self-consistent inside the function (the quadratic-eikonal fit, the demodulation and the `x_src` remodulation all use `x_ax`), so no bug follows — but the divergence is gratuitous and would become one if the fitted coefficients were ever exported.

---

### [P3] Documented performance figures for `screen_obliquity` not reproduced; the documented trend has the wrong sign
`lumenairy/elements/_lens_real.py:4377-4383`: "measured **2.2x / 2.9x / 3.6x** the carrier-free call wall-clock at N = 512 / 1024 / 2048 on a three-surface cemented element ... peak memory is that of the unbanded path plus **three float geometry grids** -- four more for a non-collimated carrier".

**Measured** (`t12_perf2.py`; N-SSK2 biconvex, `sag_chunk_rows=0` on both arms, every path warmed at the same N first, `tracemalloc` peak in units of one float64 `N²` grid):

```
N=512    thin 0.209 s  16.14 grids | +collimated carrier 3.88x, +11.13 grids | +finite-R 3.40x, +13.13
N=1024   thin 0.407 s  16.13 grids | +collimated carrier 5.09x, +11.13 grids | +finite-R 6.14x, +13.13
N=2048   thin 3.084 s  16.13 grids | +collimated carrier 2.38x, +11.13 grids | +finite-R 2.43x, +13.13
```

* Wall clock: my ratios **fall** with N (3.88 → 2.38), the documented ones **rise** (2.2 → 3.6). A fixed O(N²) per-surface addition measured against an O(N² log N) FFT baseline must fall; the documented trend is the wrong sign and is worth re-deriving. (My N = 1024 row is an outlier because the thin baseline there is unusually fast — FFT-size luck — so treat the trend, not the individual numbers.)
* Memory: "+3 grids" measured as **+11.13**, and the non-collimated surcharge as **+2.00**, not +4. (`on_screen_obliquity='silent'` was used, which skips `_obl_total`; with the default policy it is +12.13.) The "+3" may be counting only the persistent accumulators, but the sentence says "peak memory".

The other documented figures in the partition that I *could* test came out right — see "Checked and found correct".

---

## Performance opportunities

Measured on the fixtures above (warmed, `sag_chunk_rows=0`, 2-surface N-SSK2 biconvex).

| Opportunity | Where | Estimated gain | How estimated |
|---|---|---|---|
| Break the intersection Newton on a residual test instead of 24 fixed sweeps | `:598`, `:1105`, `:1517`, `:1721` | ~10x on the pointwise / remap geometric traces (8.4 s → ~1 s at N = 1024) | residual is exactly 0 at iteration 2 (`t11_obl_newton.py`); the loop body is 3 `_surface_sag_general` calls |
| Pre-allocate the `crd` buffer in the pull-back loop instead of `np.stack([iy.ravel(), ix.ravel()])` per iteration | `:3146` | 8–12 × 2 full-grid (float64) allocations per surface removed; ~5–10 % of `_tangent_facet_remap_apply` | pull-back measured at 1.79 s (surface 0) + 5.30 s (surface 1) of a 15.7 s call (`t15_profile.py`) |
| Build `iu`/`iv` by broadcasting instead of `np.arange(...)[None,:] + np.zeros((ny,1))` | `:3141-3142` | 2 full float64 grids per surface | code read |
| Replace `RegularGridInterpolator` + the `(N², 2)` query stack in `_upsample_coarse` with `map_coordinates(order=1)` on computed fractional indices; and skip `Xg, Yg = np.meshgrid(...)` (`:984`) when `_ncx == Nx` | `:1000-1015` | ≥4 full float64 grids at the pointwise path's peak (~0.5 GB each at N = 8192) | code read: `Xg`,`Yg`,`pq` are all `N²`-scale and `pq` is `2N²` |
| Replace the 2-D remap's `LinearNDInterpolator` with the structured Newton inversion already implemented at `:1046` | `:1846-1852` | 2.0x on the cos-grid analogue (16.44 s Delaunay vs 8.43 s structured, `t11_obl_newton.py`), plus removal of the 181-ray resolution ceiling | direct measurement of the two backends |
| Let `surface_model='displaced'` take the row-banded path (`_narrow_chunk` at `:5563` excludes it with `and not _displaced`, and `_slant_narrow_chunk` requires `slant_correction or fresnel`, which `displaced` refuses) | `:5563` | at N ≥ 4096 `thin` auto-bands and `displaced` does not; the displaced screen is pointwise in `sag` and needs **no** halo, so it is the easiest model in the file to band | code read; `displaced` measured at thin + 1.00 grid whole-grid, so the banded saving is the full whole-grid transient set |
| Reuse the sag and its gradient across the route-3 / remap blocks and the obliquity block | `:6182` / `:6275` | one `xp.gradient` (2 full grids) per powered surface when both are live | code read; note the two are mutually exclusive today, so this only bites a future combination |

Raw cost ladder for the record (same fixture, whole-grid, wall clock / tracemalloc peak in float64 grids):

```
                       N=512            N=1024            N=2048
thin                 0.209s 16.14     0.407s 16.13      3.084s 16.13
displaced            0.190s 17.14     1.321s 17.13      2.825s 17.13
tangent_facet        1.003s 26.14     4.989s 26.13      7.418s 26.13
tangent_facet_remap  4.502s 32.14    14.798s 32.13     47.861s 32.13
remap_order 1/3/5 at N=1024: 11.0 / 15.7 / 22.9 s
```

## Alternative algorithms / methods

1. **WPM (wave-propagation method) does not subsume this ladder, and the cost is quantifiable.** Brenner & Singer, *Appl. Opt.* **32**, 4984 (1993); Fertig & Brenner, *JOSA A* **27**, 709 (2010) (tilted-plane-wave decomposition). WPM replaces BPM's single reference index by the exact local `k_z` per medium, so it *does* handle the air→glass step the module's own BPM note rejects. But the error being corrected here is not the propagation: it is that the *sag surface* crosses the slab. A WPM only removes it if the slab thickness is small against the sag depth, i.e. `Δz ≲ sag/M`. With sag depths of 20–140 µm on the fixtures above and `M ≈ 10`, that is 2–14 µm slabs, i.e. 200–1250 slabs across a 2.5 mm gap, each an FFT pair. Against the one FFT pair the screen model pays, that is **two to three orders of magnitude** more work for an accuracy the tangent-facet screen already reaches at 8e-10…2e-5 waves (measured, `t3b.py`). Recommendation: keep the screen ladder; WPM is only interesting as an independent *oracle* at small N.
2. **Tilted-plane angular spectrum (Matsushima).** Matsushima, Schimmel & Wyrowski, *JOSA A* **20**, 1755 (2003); Matsushima, *Appl. Opt.* **47**, D110 (2008). The exact spectral rotation with its Jacobian is the frequency-domain statement of the same axial-translation identity `(T1)` this module derives in real space — so it would reproduce, not improve, the tangent-facet screen for a *single* tilt. To cover a curved surface you would bin the pupil by local facet tilt and pay one rotated-ASM per bin; for a lens with a 0.24 slope range and a 1e-3-wave bar that is tens of bins per surface. Not competitive with the pointwise screen, but it *would* give an independent spectral cross-check of `(T1)`+`(T2)` at zero new physics.
3. **Type-2 NUFFT for the remap resampling.** Barnett, Magland & af Klinteberg, *SIAM J. Sci. Comput.* **41**, C479 (2019) (FINUFFT); Greengard & Lee, *SIAM Rev.* **46**, 443 (2004). The pull-back in `_tangent_facet_remap_apply` is exactly "evaluate a band-limited field at `N²` non-uniform points", which is a type-2 NUFFT: `O(N² log N + N²|log ε|)`, spectrally exact to a chosen `ε`, and it **removes the need for the demodulating quadratic-eikonal fit entirely** (the NUFFT does not care how fast the phase oscillates, only that the field is band-limited). That would delete `_tf_remap_quadratic_eikonal`, `_tf_remap_phi`, the `spline_filter` prefilters, and the `remap_order` kwarg with its measured 1.37e-4-of-peak order-3-vs-5 gap, at comparable wall clock. It also dissolves finding [P2] above about `spline_filter`'s IIR halo: an FFT is globally coupled anyway, so nothing is lost by admitting it. Caveat: the field must be band-limited on the grid, which the module already requires for the ASM legs.
4. **Structured inversion for the 2-D displaced remap** (see Performance table) — the module already owns the implementation.

## Code organization observations

* `_build_displaced_cos_luts` (`:530`), `_build_displaced_ray_map` (`:1465`) and the 2-D pair `_build_displaced_cos_grid` (`:913`) / `_build_displaced_ray_map_2d` (`:1649`) contain **four near-verbatim copies** of the same vectorised Newton-intersection + vector-Snell loop (~45 lines each, including the identical `e = max(1e-9, 1e-6*r)` finite-difference gradient and the identical `dgdt` floor). Any fix to the Newton (e.g. the convergence break above, or the exit-index fix) has to be made four times; the exit-index bug is present in exactly the two copies that accumulate OPL. One shared `_trace_fan(surfaces, thicknesses, wavelength, state, want_opl)` would collapse them.
* The module-level comment blocks are exceptional as derivations and are also the reason the file is 6 863 lines: the `screen obliquity` block alone is 130 lines of prose before the first `def`, and the route-3 / remap blocks another 200. They are worth keeping, but they belong in `docs/` with a one-paragraph pointer here — as it stands, `_lens_real.py` is ~35 % prose and the executable logic of the four models is scattered through it.
* `_apply_real_lens_impl` is ~2 100 lines with three near-duplicate surface loops (`_narrow_chunk`, `_slant_narrow_chunk`, whole-grid), each of which has to re-implement every model's dispatch. The `_tf_*` / `_obl_*` closures exist precisely to let the three loops share code; the same treatment has not been applied to the sag construction, which is why `_narrow_chunk` carries an eleven-clause exclusion list.
* Naming: `_screen_obliquity_delta` implements eq. (4), `_screen_drift_opd` eq. (7), `_screen_coeff_error` eq. (5), `_facet_axial_momenta` eq. (3)/(T1). The equation numbers live only in comments; a docstring cross-reference table at the top of the block would make the call graph readable.
* The public API now has five interacting selectors (`surface_model`, `displaced_mode`, `displaced_obliquity`, `carrier`/`screen_obliquity`, `conjugate`) whose legal combinations are enforced by two ~200-line validators that raise on most of the product space. The routing rule that actually matters — "an asymmetric element under the defaults silently becomes a geometric remap" — is stated only inside a comment at `:4752-4756`, not in the parameter that selects it.

## Unverified suspicions

* **Non-monotone LUT crossing heights.** `_build_displaced_cos_luts` sorts by `h_cross = |py|` and feeds the result to `np.interp`, which requires a strictly increasing abscissa. If a converging congruence makes two fan rays cross surface *i* at the same height (or the fan crosses the axis inside the glass), `np.interp` silently returns a blend. I could not produce it: at `conjugate = −30 / −20.5 / −3.0 / −2.0 / −1.2 mm` all 257 crossings stayed monotone (`t7_misc.py` §E, `t8_follow.py` §E2). Would be confirmed by a prescription with a strong internal intermediate focus. Cheap defensive fix regardless: assert monotonicity and fall back to the paraxial coefficient if it fails, as the `h_lut.size == 0` branch already does.
* **`_displaced_opd` clamps the cosines outside the traced fan** (`np.interp(..., left=cin_lut[0], right=cout_lut[-1])`, `:665-666`) with no comment and no guard. At `conjugate = −2 mm` the surface-1 crossing heights span only 0.5–173 µm while the field grid runs to ~1.4 mm, so the outer 88 % of the pupil gets the marginal ray's cosine. The field there is small, so I could not show it matters, but the extrapolation is silent and unbounded.
* **`_r_max` fallback** (`:4778-4783`) uses `0.5 * max(Nx*dx, Ny*dy)` when there is neither an aperture nor a semi-diameter, which never reaches the grid *corner* radius `0.5*sqrt((Nx dx)² + (Ny dy)²)`. Combined with the clamping above, the corners always run on extrapolated cosines. Untested.
* **`_apply_displaced_remap`'s central hole.** For `r_out < ho[0]` the code clamps `rc` and computes `scale = hin_of/r_out`, so the on-axis pixels sample the input envelope at a *fixed* radius `hi[0] = r_max/n_fan`. For a smooth envelope that is benign (and `n_fan = 1025` makes `hi[0]` ~1 µm), but for a structured on-axis input it is not. Not measured.
* Doc claim I could not test at this scale: `'tangent_facet'` 17.6x better than the `carrier`-corrected thin screen (0.00008 vs 0.00141 waves). On my plano-convex fixture `displaced`, `tangent_facet` and `tangent_facet_remap` all land at 0.0007 waves against an exact ray trace, i.e. indistinguishable — but that is my comparison floor (unwrap + oracle interpolation), not evidence against the claim. The screen-level oracle (`t3b.py`) does separate them.

## Checked and found correct

* **The axial-translation identity** `Λ(s) − Λ(0) = s(pz1 − pz2)` and therefore `OPD = (pz2 − pz1)·sag` — verified against an explicit 3-D ray trace through a tilted plane facet over 400 random `(n1, n2, θ ≤ 0.45, slope ≤ 0.30, s)` draws: **worst relative error 1.14e-12**. `_facet_axial_momenta` returns exactly `Γ·ν_z = pz2 − pz1` (`t1_facet_identity.py`).
* **The walk** `W = s_hit·(p/pz1 − p_out/pz2)` against the same trace: **worst relative error 1.14e-13**. Sign and reference plane both correct.
* **The (T1) → route 3 → remap ladder** against an exact ray oracle built by inverting the landing map `x + W(x) = u` on a single spherical surface (`t3b.py`), waves rms, piston+tilt removed:
  ```
      R      n1   n2   θ[mrad]  ap[mm] |  (T1)      route3     remap(R1)
    19.6   1.00 1.62     0.0     1.00  | 1.079e-06  8.122e-10  4.473e-10
    19.6   1.00 1.62    80.0     2.00  | 2.337e-04  5.181e-07  1.933e-08
   -19.6   1.62 1.00    80.0     2.00  | 1.347e-03  1.745e-05  4.719e-08
    12.0   1.00 1.50    80.0     2.00  | 1.215e-03  9.853e-06  1.921e-07
  ```
  The documented ordering and the documented magnitudes both hold; the walk itself reproduces to 0.03–6 nm.
* **On one surface, `thin + eq(4)` differs from `(T1)` by exactly the carrier-free coefficient error** `[(n2−n1) − dz(0)]·sag`, independent of θ (3.51e-08 m at 10, 55 and 150 mrad) — i.e. the screen-obliquity correction fixes only the angular part and leaves the normal-incidence steep-facet ceiling, exactly as the module claims.
* **Banded == whole-grid, BYTE-IDENTICAL**, N = 1024, 3-surface element, explicit `sag_chunk_rows ∈ {37, 128, 256}` against `sag_chunk_rows=0`, for `thin`, `thin + collimated carrier`, `tangent_facet` and `tangent_facet_remap`: `np.array_equal` **True**, `maxdiff 0.0` in all 12 combinations (`t10_band_perf.py`). The halo derivation (3 sag / 2 accumulator rows for route 3, 2 / 0 for the remap, 1 for the gap transport) is sound.
* **Energy.** Exit power / apertured input power at N = 1024: `thin` 1.000000, `displaced` 1.000000, `displaced_mode='split'` 1.000000, `tangent_facet` 1.000000, `displaced_mode='remap'` 0.999923, `tangent_facet_remap` 0.999976. The amplitude Jacobians conserve energy as derived.
* **Cross-model agreement against an exact meridional ray trace**, plano-convex N-BK7 R = 12 mm, t = 3 mm, ap = 2.4 mm, N = 2048, dx = 1.5 µm (Nyquist-compliant for the exit NA), exit wavefront rms error in waves (`t16b.py`):
  ```
  orientation     thin      displaced  tangent_facet  tangent_facet_remap
  curved-first   0.01357     0.00069      0.00069           0.00063
  flat-first     0.02054     0.00076      0.00076           0.00073
  ```
  The three angle-aware models agree to ~1e-5 waves and beat `thin` by 20–30x in both orientations. (`displaced` and `tangent_facet` coinciding is expected and is a good cross-check: at surface 0 under a collimated fan, `cos_in = 1` so eq. (1) *is* `(T1)`.)
* **`_tf_remap_quadratic_eikonal`'s 5×5 normal equations** — re-derived by hand from `∂/∂(c0,c1,c2,d0,d2) Σ w[(Φx−px)² + (Φy−py)²]`; every entry of the matrix and the RHS matches, including the shared cross term `myy+mxx` on the `c2` row that enforces curl-freedom.
* **The demodulate/remodulate sign and evaluation point** in `_tangent_facet_remap_apply`: `f = E·jac·exp(−i k0 Φ(x))`, resampled at the source index, then `×exp(+i k0 Φ(x_src))` — the remodulation is at the *source* point, so it exactly undoes the demodulation and the transported value is `E(x_src)`, as required for Lagrangian transport. The amplitude Jacobian `1/sqrt(det A)` is likewise multiplied in *before* the resample, hence evaluated at the source point, matching the derivation.
* **`_interp2_structured`'s Newton inversion algebra** (axis order of `np.gradient` vs the `meshgrid` layout, the `(v+r_max)/_du` index map, the 2×2 inverse) — all correct.
* **Pull-back convergence**: 8 and 12 iterations on the two surfaces of a biconvex singlet at N = 1024, well inside the 64 cap (`t15_profile.py`). The `_TF_REMAP_MAX_ITERS` ceiling is not being hit in normal operation.
* **`lens_sag_float32_opd_error` does not leak the process-global sag dtype** — it drives the A/B through the `sag_dtype=` kwarg only, never `set_lens_sag_dtype`, so no `try/finally` is needed (probe 8's concern is unfounded). The float32 arm is genuinely float32 (`h_sq = (r.astype(np.float32))**2` keeps the whole chain narrow under NEP 50), and `ok` is `max_waves < 0.02` (λ/50) **and** `field_rel < max_field_rel_error`, as documented.
* **Cache locking**: `_DISPLACED_LUT_CACHE` is guarded by its own `threading.Lock` for get/put/clear; `ByteBudgetedLRU.get`/`put`/`set_budget`/`clear` all take the shared `_BUDGET_MUTEX`. No lost-update or torn-read hazard found.
* **`_screen_obliquity_rms_waves`**: the masked 3×3 normal equations reproduce a piston+tilt least-squares over the pupil disc exactly (zeros outside the mask contribute nothing to either `A`, `b` or the residual, and `n` counts only the disc).
* **`_displaced_geom_key` / `_displaced_cos_grid_key` field coverage** — apart from the glass-name issue above, both keys carry every quantity the traces actually read (radius, conic, aspheric dict, decenter, tilt, `sag_callable`, thicknesses, wavelength, `r_max`, conjugate, and for the grid `Nx/Ny/dx/dy/n_launch/n_coarse/interp_method`). `form_error`, `radius_y` and `freeform_type` are refused upstream, so their absence from the key is safe.
* The `'displaced'`/`'tangent_facet'` mutual-exclusion validators (`_check_displaced_support`, `_check_screen_obliquity_support`) — I could not find a combination that reaches a model with a kwarg it would silently ignore; in particular `displaced_obliquity='meridional'` on an asymmetric element correctly raises, and `carrier=` + `'displaced'` correctly raises rather than double-counting.
