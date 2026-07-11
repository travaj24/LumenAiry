# Coatings + Elements Audit — 2026-07-09

Scope: full line-level reads of `elements/coatings.py` (725 — the
transfer-matrix thin-film model, its JAX twin, and the coating-material
registry) and `elements/elements.py` (1,159 — mirrors, apertures,
masks, the Zernike element builder, coronagraph templates, and the
turbulence phase-screen generator).  Continuation of the `elements/`
sweep after the DOE/grating/freeform tranche.  Read-only; the TMM
algebra and the mirror/turbulence physics re-derived by hand.

---

## 1. Verdict

**Both modules are correct.**  Independently verified this pass:

* **`coating_reflectance` TMM** — the full Macleod characteristic-matrix
  chain: the layer matrix
  `[[cos δ, −i sin δ/η],[−i η sin δ, cos δ]]`, the tilted admittances
  (`η_s = n cos θ`, `η_p = n/cos θ`), the tournament matrix product
  (verified left-to-right order-preserving, ambient-side first), the
  `[B;C] = M·[1; η_sub]` reflection algebra, `r = (η_amb B − C)/
  (η_amb B + C)`, `t = 2η_amb/(η_amb B + C)`, and the power
  transmittance `T = Re(η_sub)/Re(η_amb)·|t|²` (Macleod 2.99) all match.
* **The v5.6 complex-Snell branch** — `cos θ_j = √(1 − (n₀sinθ₀/n_j)²)`
  on the decaying-evanescent branch `Im(n_j cos θ_j) ≥ 0`, with the
  exact-critical `|cos θ|<1e-12` guard against the p-pol `n/cos θ`
  blow-up.  The real-subcritical gate (all-real indices AND subcritical
  everywhere) correctly preserves the bit-identical libm path; the
  cap-free complex path gives the physically-correct `R→1`, `T→0` at
  TIR.  The P1-NEW-2 complex-sum reflection-phase aggregation
  (`angle(½(r_s+r_p))`, Brewster-branch-cut-robust) is sound.
* **`apply_mirror`** — the double-pass sag phase `φ = −k·2·sag`
  reduces paraxially to `−k r²/R` = a converging lens of `f = R/2`
  for concave (wave-side R>0) ✓; the wave-side-vs-Welford sign note is
  accurate; the P1-NEW-F NaN-sag guard prevents `exp(i·NaN)` poisoning.
* **`zernike`** — OSA/Born-Wolf normalisation (`√(n+1)` for m=0,
  `√(2(n+1))` otherwise) matches `analysis.zernike_polynomial` exactly
  (round-trip-safe), radial recurrence correct, `rho>1` zeroed.
* **`generate_turbulence_screen`** — the spectral shape is correct: the
  `f^(−11/3)` Kolmogorov slope, the von Karman outer-scale knee at
  `f = 1/L0` (equivalently `κ = 2π/L0`), and the inner-scale cutoff at
  `κ_m = 5.92/l0` (the `2π` in `exp(−(2πf·l0/5.92)²)` correctly maps
  cyclic f to angular κ) are internally consistent on the cyclic-f
  grid.  The absolute normalisation is documented as pinned to
  `D(r0)=6.88` (the √2 real-part-variance factor is the right idea);
  I did not re-derive the (2π) constant bookkeeping — flagged as
  verified-by-shape, calibration-by-citation.
* The coronagraph templates (Lyot FPM hard/gaussian/sin2, scalar
  vortex, FQPM `X·Y<0`, 8OPM octant alternation, Lyot stop, cos²/
  cos^n/gaussian/sonine apodizers) are all standard forms; the
  xp-dispatch, dtype-aware zeros, and `dy` support are uniformly
  present.
* **`coating_reflectance_jax`** — the thickness-differentiable path is
  structurally correct (concrete complex `cos θ` chain, δ carries the
  differentiated `d`, ambient-first `M @ Mj`).

One real finding and a set of nits follow.

---

## 2. Findings

### COAT-1 (P4) — `coating_reflectance` holds layer indices fixed across the wavelength sweep, and can't compose with the dispersion helper
The function accepts a wavelength ARRAY and returns spectral `R(λ)`,
but each layer carries a single scalar index `n_j`; only the phase
thickness `δ = 2π n_j d_j cos θ_j / λ` varies with λ (through `1/λ`
and the fixed `n_j`).  So a spectral sweep is computed with
**non-dispersive** indices — the layer `n` at 400 nm and at 1600 nm
are identical unless the caller slices the sweep and calls once per
λ.  The module ships `get_coating_material_index(material, wavelength)`
precisely to supply `n(λ)`, but the two cannot be composed over an
array in one call (the TMM takes one `n` per layer, not `n(λ)`).  For
weakly-dispersive dielectrics over a modest AR/HR band the error is
small; over a wide band, or for high-index/dispersive layers, it is
material — and nothing in the docstring flags it, while the array-λ
signature actively invites the spectral-sweep reading.  **Fix**:
document the non-dispersive assumption, and/or accept a per-layer
`n(λ)` callable (or an `(n_layers, n_wv)` index array) so the
dispersion helper can feed it.

### Nits
* `quarter_wave_ar` returns `n = √(n_substrate)` — the ideal only for
  an **air** ambient (`√(n_sub·n_ambient)` in general).  The function
  takes no `n_ambient`, so a non-air ambient gives a mistuned layer;
  worth a docstring note (or the general form).
* `get_coating_material_index` fires the "extrapolated value may not
  be physical" out-of-range `UserWarning` even for the **constant-n**
  materials (MgO, ZnS, Al2O3, …) that have no Sellmeier — where the
  returned value is flat and no extrapolation actually occurs.
  Misleading; gate the warning on `'sellmeier' in entry`.
* `coating_reflectance_jax`'s local list `n_re` actually holds the
  **full complex** index (`n_re.append(complex(n_layer))`), not the
  real part — a misnomer that reads as a bug on skim (the value is
  correct).
* A negative wavelength into `get_coating_material_index(...,
  sellmeier)` triggers two `UserWarning`s (one here, one inside
  `_coating_sellmeier`'s shared `_guard_wavelength`).
* `apply_aperture(shape='annular')` / `apply_lyot_stop` have no
  `inner < outer` guard: an inverted annulus silently returns an
  all-zero field (same class as SRC-2's `create_annular_beam`).
* `apply_gaussian_aperture` has no `sigma > 0` guard: `sigma=0` →
  divide-by-zero → NaN field.
* `create_eight_octant_phase_mask`: a ray of pixels at exactly
  `θ = π` lands in `octant = 8` (phase 0), a one-pixel-wide seam;
  cosmetic.

---

## 3. Coverage statement

Every line of `coatings.py` and `elements.py` read.  Cross-referenced:
`polarization.py`'s `apply_real_lens(fresnel=True)` s/p-averaging
caveat (P2-3, prior tranche) is the JonesField-side consequence of
`coating_reflectance`'s `polarization='avg'` power average — both
honestly documented.  Not audited here: `bsdf.py` (591),
`materials.py` (104), `segment_geometry.py` (558),
`lenses_maslov.py` (3,503), `eme/` (1,834), the `io/` siblings, and
`optimize/` — the natural next tranches.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_DOE_GRATING_FREEFORM_2026_07_09.md`,
`AUDIT_GLASS_POLARIZATION_2026_07_08.md` (the polarization s/p bridge),
`AUDIT_IO_ZEMAX_2026_07_08.md`, `AUDIT_RAYTRACE_CORE_2026_07_08.md`,
and the 07-07 set.*
