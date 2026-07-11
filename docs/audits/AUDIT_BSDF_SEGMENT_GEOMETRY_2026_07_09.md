# BSDF + Segment-Geometry Audit — 2026-07-09

Scope: full line-level reads of `elements/bsdf.py` (591 — the
Lambertian / Gaussian / Harvey-Shack scatter models + the scatter-ray
spawner) and `elements/segment_geometry.py` (558 — the exact interval-
arithmetic z-stack geometry builder feeding PMM/RCWA).  This tranche
**completes the `elements/` directory** (PMM/RCWA, lenses, DOE/grating/
freeform, coatings, polarization, Maslov, EME, and now these two are
all line- or deep-audited).  Read-only; scatter sampling and the
morphological-dilation coat re-derived by hand.

---

## 1. Verdict

Both modules are sound; **one real sampling bug** in the Gaussian BSDF.

Verified correct:

* **BSDF `evaluate` / TIS**: Lambertian `ρ/π` with closed-form TIS = ρ
  (I verified `∫(ρ/π)cosθ sinθ dθ dφ = ρ`); Gaussian small-angle
  normalisation `A = f/(2πσ²)` (from `∫A e^{−θ²/2σ²}θ dθ dφ = f`) with
  the v5.17/P3-15 `|cos θ_spec|` incidence correction; Harvey-Shack ABC
  `b0/(1+(sinθ/l)²)^{s/2}` and its `(λ_ref/λ)²` smooth-surface scaling.
  The base-class numerical TIS's shape-mismatch `ValueError` guard
  (P2-17) is a good defensive check.
* **Lambertian & Harvey-Shack sampling**: both draw directions with the
  correct projected/power weighting.  Lambertian's cosine-weighted
  `θ = arcsin√ξ` gives density ∝ cos θ ∝ BSDF·cos θ ✓; Harvey-Shack's
  rejection weight `w(u) = u·profile(u)` (u = sin θ) reproduces
  ∝ BSDF·cos θ (I traced the `du = cos θ dθ` Jacobian), and its
  envelope `peak_val` at `u* = l/√(s−1)` is a valid upper bound even
  when `u* > 1` (monotone case).
* **Segment geometry** (`segment_geometry.py`): the interval primitives
  (`_norm_intervals` merge, `_complement`, `_union`, and the wrap-aware
  L∞ `_dilate_x` — verified the `b−a ≥ period → full`, sub-0 and
  over-period wrap splits), the conformal `coat` (split-at-boundaries-±t
  then per-band dilation over the ±t z-window, plus the substrate-floor
  full-coat), the `line_interface` vertical + horizontal carving with
  the P3-39 thin-band omission warning, and the exact-boundary RCWA
  pixelation all check out.  This module's stated purpose — removing the
  geometry-classification bug class by hand — is met.

## 2. Findings

### BSDF-1 (P3) — `GaussianBSDF.sample` omits the sin θ solid-angle Jacobian, so sampled rays don't reproduce the lobe (biased toward specular)
`sample` draws the offset angle as a **half-normal**,
`θ = |normal(0, σ)|`, giving a per-θ density ∝ `exp(−θ²/2σ²)`.  But to
reproduce the Gaussian BSDF *lobe* as a Monte-Carlo direction
distribution, the per-θ density must carry the solid-angle (and
projected-power) weight: ∝ `BSDF(θ)·cosθ·sinθ ≈ θ·exp(−θ²/2σ²)`
(small-angle) — a **Rayleigh** distribution, not a half-normal.  The
missing `sinθ ≈ θ` factor over-concentrates samples at small angles:
the sampled mean scatter angle is the half-normal's `σ√(2/π) ≈ 0.80σ`
where the true lobe's is the Rayleigh `σ√(π/2) ≈ 1.25σ` — the drawn
directions cluster ~35% closer to specular than the model specifies, so
`sample_scatter_rays` produces a systematically too-narrow stray-light
cone.  The sibling `HarveyShackBSDF.sample` includes exactly this weight
(`w = u·profile`), and `LambertianBSDF.sample` is cosine-weighted —
confirming the intended `∝ BSDF·cosθ` convention that the Gaussian path
alone violates.  Neither `evaluate` nor `total_integrated_scatter`
(closed-form `= scattered_fraction`) is affected — the bug is isolated
to the MC draw.  **Fix**: sample `θ` from the Rayleigh form,
`θ = σ√(−2 ln(1−ξ))` (clipped to `π/2`), or rejection-sample with the
`sinθ` weight as Harvey-Shack does.

### Nits
* `HarveyShackBSDF.evaluate` indexes `inc[0]/inc[1]/inc[2]`
  (single-incidence only), while `GaussianBSDF.evaluate` was made
  batch-safe with `inc[..., k]` (F-22).  A batched-incidence call to the
  Harvey-Shack evaluator would break broadcasting — an unmirrored fix.
* `HarveyShackBSDF` has no closed-form `total_integrated_scatter`
  (falls back to the base numerical integrator, correct but slow); its
  TIS can exceed 1 for large `b0` with no guard (user's responsibility,
  but unvalidated).
* `to_rcwa_stack` seeds the pixel column with `NaN` and assigns by
  half-open `[lo, hi)`; if an (externally-built) interval set fails to
  tile `[0, period)` exactly, residual `NaN` eps flows to the RCWA
  layer silently — a post-assignment `isnan` check would fail loud.
* `LambertianBSDF.evaluate`'s hemisphere test is frame-sensitive
  (documented, with an advisory `RuntimeWarning` for likely
  world-frame inputs) — not a bug, noted for completeness.

## 3. Coverage statement

Every line of `bsdf.py` and `segment_geometry.py` read.  With this
tranche the **entire `elements/` directory is line- or deep-audited**
(pmm/, rcwa/, berreman, emt, bor, lenses + `_lens_*`, doe, thin_grating,
freeform, coatings, elements, polarization, lenses_maslov, eme/, bsdf,
segment_geometry, materials via cross-ref).  Not yet audited: the `io/`
siblings (`prescriptions_code_v.py`, `prescriptions_quadoa.py`,
`prescriptions_transforms.py`, `prescriptions_builders.py`,
`storage.py`, `codegen.py`) and the `optimize/` subsystem (~7k) — the
remaining ground.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_EME_2026_07_09.md`, `AUDIT_MASLOV_2026_07_09.md`,
`AUDIT_COATINGS_ELEMENTS_2026_07_09.md`,
`AUDIT_DOE_GRATING_FREEFORM_2026_07_09.md`.*
