# Optimize Wrapper-Merits + Context Audit — 2026-07-09

Scope: `optimize/wrapper_merits.py` (992 — `MultiWavelengthMerit`,
`MultiFieldMerit`, `ToleranceAwareMerit`, the aperture-mask cache) and
`optimize/context.py` (591 — `EvaluationContext`, `MeritTerm` base,
`ctx_is_valid`, the failure sentinels, `Constraint`).  Third `optimize/`
tranche.  Read-only; the three aggregators' per-leg wave re-evaluation
and averaging re-derived, cross-checked against the leaf merits they
drive.

---

## 1. Verdict

**Correct except one aggregator drops two sub-context fields.**
Verified this pass:

* **`ctx_is_valid`** — AttributeError / None / non-finite /
  `≥ 0.5·_INVALID_FL_SENTINEL` guards; the exact gate the leaf
  constraint merits rely on.  `EvaluationContext.rms_wavefront_waves`
  wraps the (already-audited) `zernike_decompose`, excludes low-order,
  RMS-in-waves — correct math.  The failure sentinels
  (`_InvalidFocalLength`, `_FailedScanStrehl`, `_ZeroApertureMask`) with
  their `__float__` collapses are a sound way to keep an identity-check
  available while the arithmetic path stays finite.
* **`MultiWavelengthMerit`** — the 4.10 fix (re-run the wave leg at each
  wavelength, vs the pre-4.10 no-op that averaged the same
  single-wavelength number) is in place; the v4.16.1 **SUM→AVG** fix
  (divide by `len(wavelengths)`, matching the docstring and both
  siblings) with its one-shot `FutureWarning` migration notice is
  correct; the degenerate-ABCD sentinel, the cached aperture mask
  (incl. the `_ZERO_APERTURE_MASK` deliberate-zero branch), and the
  nanargmax best-focus all check out.  It populates **strehl + rms +
  opd** on the sub-context.
* **`MultiFieldMerit`** — the aperture-clipped tilted plane wave (with
  the C-P0-2 both-X-and-Y tilt fix — the X term was previously dropped),
  the per-field through-focus, and the AVG-over-fields aggregation are
  correct; it populates **rms_radius_best (nanargmax) + opd_map (when
  the sub-merit needs it)**.
* **`ToleranceAwareMerit`** — the Monte-Carlo structure is right: a
  per-trial `default_rng(seed+t)` (reproducible), a deterministic
  form-error seed, `apply_perturbations`, a through-focus scan around
  the **perturbed** BFL (not the nominal — the key detail that keeps
  the perturbed Strehl honest), and the MEAN-over-trials aggregation.

## 2. Findings

### OPT-2 (P3) — `ToleranceAwareMerit` only populates `strehl_best`, so every OPD-/spot-based sub-merit (incl. the docstring-named `RMSWavefrontMerit`) degenerates to ∞
The per-trial `sub_ctx` (constructed at ~L964) sets `prescription`,
`wavelength`, `N`, `dx`, `efl_p`, `bfl_p`, `x`, and then the
through-focus block sets **only `sub_ctx.strehl_best`** (L980).  It
never sets `rms_radius_best` (stays the `EvaluationContext` default
`np.inf`) and never builds `opd_map` (stays `None`).  Consequences for
the sub-merits the class is meant to wrap:

* **`RMSWavefrontMerit`** (named in this class's own docstring as a
  "typical" sub_merit) reads `ctx.rms_wavefront_waves()`, which returns
  `np.inf` when `opd_map is None` → `excess = max(0, ∞−target) = ∞` →
  the merit is `weight·∞`.  Summed over trials, the whole tolerance
  merit is `∞`, which stalls/breaks the optimiser.
* **`SpotSizeMerit`** reads `rms_radius_best` (default `∞`) → `∞`.
* **`MatchTargetOPDMerit` / `ZernikeCoefficientMerit`** read `opd_map`
  (None) → return `0.0` (silently inert — a robustness merit that does
  nothing).

Only `StrehlMerit` works.  Both sibling aggregators
(`MultiFieldMerit`, `MultiWavelengthMerit`) populate `rms_radius_best`
via nanargmax AND `opd_map` via `wave_opd_2d`, so the identical
sub_merit that optimises fine under those two returns `∞`/inert under
`ToleranceAwareMerit`.  **Fix**: mirror the siblings — after the
per-trial `find_best_focus`, set `sub_ctx.rms_radius_best` from
`scan.rms_radius[nanargmax]`, and build `sub_ctx.opd_map` via
`wave_opd_2d(E_exit, ..., focal_length=bfl_p)` when
`self.sub_merit.needs_wave`.

### Nits
* `EvaluationContext.rms_wavefront_waves` returns `np.inf` (not a
  large-but-finite penalty) when `opd_map is None` or the aperture is
  absent.  Every other leaf-merit failure path returns a finite
  `weight` sentinel; this one propagates `∞` into `RMSWavefrontMerit`,
  which is what turns OPT-2 from "large finite garbage" into a hard
  `∞` that scipy handles poorly.  A finite fallback here would soften
  OPT-2 even before its root-cause fix.
* `MultiWavelengthMerit` keeps its `_MULTIWL_AVG_WARNED` latch mirrored
  across both this module and the `optimize.core` re-export alias (for
  test-fixture resets) — correct but intricate dual-write; noted for
  the next maintainer.

## 3. Coverage statement

Deep-read: `context.py` in full (sentinels, `ctx_is_valid`, `MeritTerm`
base, `EvaluationContext` + `rms_wavefront_waves`, `DesignResult`); the
three `evaluate` methods of `wrapper_merits.py` and their aggregation.
Structurally covered: the aperture-mask cache infrastructure
(`_wrapper_merit_aperture_key`, `_get_wrapper_merit_cache`,
`_clear_wrapper_merit_cache`, the `_ZERO_APERTURE_MASK` sentinel, byte
budget) and the `_merit_jit` tilt-phasor kernel dispatch (verified only
at the call site).  `Constraint.__post_init__`/`validate`/`to_scipy`
(context.py 335-591) read for shape but not derivation-checked.  **Not
audited**: `multi_objective.py` (396), `multiconfig.py` (440),
`_merit_jit.py` (263, the Numba tilt kernel), `core.py` (419), and the
`io/` siblings — the remaining ground.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_OPTIMIZE_DRIVER_2026_07_09.md`,
`AUDIT_OPTIMIZE_MERITS_2026_07_09.md` (OPT-1).*
