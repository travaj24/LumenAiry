# Optimize Merit-Terms Audit — 2026-07-09

Scope: the merit-physics core of the `optimize/` subsystem —
`merit_terms.py` (1,348 — every leaf `MeritTerm`) and `jax_merits.py`
(487 — `JaxMeritTerm`, `make_lg_aberration_merit_jax`,
`optimize_traced_geometry`).  Cross-referenced
`propagators/asymptotic_jax_twin.py:aberration_tensor_lg00_jax` to
settle the LG-merit sign.  First tranche of `optimize/`; the driver /
core / context / parameterizations / wrapper-merits plumbing is a later
pass.  Read-only; penalty formulas and the merit directions re-derived.

---

## 1. Verdict

**The NumPy merit terms are correct; one JAX merit optimises the wrong
direction.**  Verified this pass:

* **Geometric merits** — `FocalLength` / `BackFocalLength` normalised
  squared error with the afocal `target==0` → `(1/efl)²` power fallback;
  `SphericalSeidel` `S_I²`; the constraint merits `MinThickness`
  (glass-only via `glass_after[i]`, air-gap skip), `MaxThickness`,
  `MinBFL`, `MaxFNumber` — all correct squared-excess/deficit forms with
  the 4.10/5.17 sentinel guards (`ctx_is_valid`, the `1e9`-BFL/EFL
  fail-loud replacements, the P3-48 `aperture_diameter is None` route).
* **Wave merits** — `Strehl` `max(0, min−best)²`, `RMSWavefront` /
  `SpotSize` excess-squared, with the honest F-4 note that the
  contiguous-OSA `exclude_low_order` slice cannot drop defocus while
  keeping both astigmatism orientations.
* **Field-matching kernels** (`MatchIdealSystem`) — I checked all four:
  `field_overlap` `1−|⟨E_i,E_r⟩|²/(P_iP_r)` (coupling efficiency),
  `field_mse` (power-normalised, global-phase-aligned residual /P_i),
  `intensity_mse` (equal-power-normalised), `intensity_overlap`
  (normalised cross-correlation) — all standard, with correct
  zero-field guards; the ASM focus-search picks min-penalty over a z
  sweep.
* **`MatchTargetOPD` / `ZernikeCoefficient`** — Zernike-decompose the
  residual/OPD, exclude-low-order or per-mode target, RMS-in-waves²
  (unit handling `coeffs[m]−target` in metres `/wavelength` → waves ✓),
  NaN-clean + decompose-failure fallbacks.
* **`LGAberrationMerit`** (NumPy) — correct: penalises
  `Σ_targets wgt·|L[(p,ell),(0,0)]|²` where the **non-(0,0)** channels
  are aberration content (→0 for a perfect system), the right
  direction; the fit cache and the fail-loud `1e20` penalties are sound.
* **`JaxMeritTerm`** — forward `evaluate` + `jax.grad` `gradient_at_x`
  plumbing (`weight·|fn|` or `weight·real(fn)`), the JAX→NumPy boundary
  bridge, and `supports_jax_grad` are correct.
* **`optimize_traced_geometry`** — the Adam loop and (crucially) the
  DEFAULT merit `−max|E_foc|²/ΣΙ₀` MINIMISE → **maximise** peak focal
  intensity — the correct focus-sharpening direction, and the exact
  contrast that makes the LG-merit bug below unambiguous.

## 2. Findings

### OPT-1 (P3) — `make_lg_aberration_merit_jax` optimises the Strehl in the WRONG direction (and its documented example is rejected at construction)
`aberration_tensor_lg00_jax` returns "the leading **Strehl amplitude**"
(`asymptotic_jax_twin.py:330`) — `|res|² → 1` for a perfect system,
`→ 0` as aberration grows.  The merit's inner `fn` returns
`total = Σ_field piston_weight·|res|²`, and `JaxMeritTerm(real_part=True)`
makes the merit value `weight·total ≥ 0`.  `design_optimize` **minimises**
the weighted merit sum, so this term is minimised at `|Strehl| = 0` —
i.e. it drives the design toward **maximum aberration**, the opposite of
every other merit in the module (all of which → 0 when the design is
*good*), and the opposite of `optimize_traced_geometry`'s own
peak-intensity merit.  A correct Strehl merit would minimise
`(1 − |res|²)` (or `(target − |res|²)²`).  Compounding it, the
constructor **rejects every non-(0,0) target** (`aberration_tensor_lg00_jax`
computes only the (0,0) channel), so:

* the class docstring's headline example — `targets={(2, 0): 1.0}`,
  "minimise primary spherical aberration" — raises `NotImplementedError`
  at construction (an unrunnable documented example); and
* the only accepted form, `targets={(0, 0): w}`, minimises the Strehl.

So the JAX LG merit cannot fulfil its stated purpose: it either raises
(non-(0,0) target) or worsens focus ((0,0) target).  The prior fix
history (the comment at ~L326 patched the *weight scaling* — "pre-fix
code ignored wgt") shows the surface was patched without catching the
direction.  **Fix**: return `piston_weight·(1 − |res|²)` (Strehl
deficit), and either implement the non-(0,0) aberration channels in
`aberration_tensor_lg00_jax` or correct the docstring example to
`{(0, 0): w}`.  The NumPy `LGAberrationMerit` is unaffected (its
targets are the aberration channels, minimised correctly); users who
need differentiable aberration control should be pointed there (FD path)
until this is fixed.

### Nits
* `make_lg_aberration_merit_jax` and `optimize_traced_geometry` call
  `jax.config.update('jax_enable_x64', True)` as a side effect of
  construction / invocation — a **process-wide** JAX state mutation
  (same anti-pattern class as the pre-fix `np.random.seed` in
  `makedammann2d`).  A caller relying on default float32 JAX elsewhere
  is silently switched to float64.  Document it or scope it.
* `JaxMeritTerm.gradient_at_x` builds `x_jax = jnp.asarray(x)` at JAX's
  default dtype (float32 unless x64 is on) while the forward `evaluate`
  bridges through `float()`; for a generic (non-LG) JaxMeritTerm the
  gradient is then single-precision, a possible accuracy mismatch with
  the double-precision forward value.

## 3. Coverage statement

Deep-read: `merit_terms.py` — all geometric/wave/constraint merits, the
`MatchIdealSystem` metric kernels + focus search, `MatchTargetOPD`,
`ZernikeCoefficient`, `ChromaticFocalShift`, and `LGAberrationMerit`;
`jax_merits.py` in full.  Cross-referenced `aberration_tensor_lg00_jax`
for OPT-1.  Structurally covered (setup/plumbing, not line-verified):
`MatchIdealThinLensMerit` / `MatchIdealSystemMerit` construction
(`_make_source`/`_build_real_elements`/`_propagate`, ~210-640) and the
`Composite`/`Callable` wrappers.  **Not audited**: the rest of
`optimize/` — `driver.py` (1,413), `core.py` (419), `context.py` (591),
`parameterizations.py` (473), `wrapper_merits.py` (992),
`multi_objective.py` (396), `multiconfig.py` (440), `_merit_jit.py`
(263) — the driver / parameter-mapping / multi-config plumbing, a
natural next tranche.  Also unaudited: the `io/` siblings.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_BSDF_SEGMENT_GEOMETRY_2026_07_09.md` (completes
`elements/`), `AUDIT_EME_2026_07_09.md`, `AUDIT_MASLOV_2026_07_09.md`.*
