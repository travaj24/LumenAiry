# RCWA Module Integration Audit — LumenAiry v5.5.0

**Module under review:** `lumenairy/elements/rcwa.py` (1844 lines), shipped as
the v5.5.0 RCWA feature (commit `7b3c742`).
**Date:** 2026-05-31
**Scope:** RCWA ↔ library **integration** gaps, synergy/compatibility
opportunities, library organization, and code quality.
**Companion:** `AUDIT_V5_5_0_2026_05_31.md` is a separate **adversarial
bug-hunt** of the same release (P1/P2 correctness focus). This report is the
**integration/architecture/quality** view; the two are complementary. See
§7 for cross-references and triangulated corrections to two of the companion's
correctness claims.

**Method.** 8 parallel dimension-reviewers (backend/GPU, field pipeline,
grating/coatings unification, ray-trace/prescription, conventions, code
quality, organization, API ergonomics) read the source with file:line
citations; **52 candidate findings**, each adversarially verified by an
independent reviewer (**47 survived**; 5 refuted). High-severity items were
runtime-corroborated by the lead. Survived by category: integration-gap 23,
convention 15, code-quality 7, organization 2.

> **Headline:** The solver *math* is correct (the companion's P1-A "oblique-TM
> split" is a convergence artifact, not a bug — triangulated in §7). The real
> issues are **architectural/connective**: RCWA is a well-built *island*, plus
> one genuine helper bug (`apply_reflection` mutates its input, §7 / companion
> P2-C) and a cluster of missing input validation.

---

## 1. Executive summary

The two highest-leverage gaps are architectural: (1) the module is pure-NumPy
with no `backend.array_namespace` dispatch and no `use_gpu` kwarg — the sole
major element that cannot run on CuPy/JAX — with `rcwa_efficiency_1d_jax`
(`rcwa.py:1412-1559`) a full ~150-line copy-paste solver **fork** rather than a
dispatch variant; and (2) RCWA results are a dead-end relative to the
field-propagation pipeline — `RCWAResult` exposes only a uniform zeroth-order
2×2 Jones, with no path to a propagatable `JonesField`, no higher-order field
reconstruction, and no `'rcwa'` element type in `propagate_through_system`. The
most dangerous correctness exposure is **missing input validation** (silent
wrong answers / cryptic linalg errors). Quick wins are small, non-breaking
validation/documentation fixes; the keystone is a single backend-dispatched
solver that kills the JAX fork, unlocks GPU/JAX, and removes the per-function
`_jax` twin gap.

---

## 2. RCWA ↔ library integration gaps

### A. Backend / GPU / JAX architecture — **HIGH (root structural gap)**
`rcwa.py` imports only `import numpy as np` (`:69`) and uses `np.*` in every
solver (`:380, :678, :926, :1043, :1230, :1760`). It is the only major element
not following the `array_namespace`/`to_backend`/`is_jax_array` + `use_gpu`
pattern (`propagators/asm.py:143,246-256`, `fresnel.py:101-103`). No `use_gpu`;
JAX/CuPy arrays silently demote to NumPy. The custom-VJP-eig counter-argument
only justifies *conditional* JAX inside a dispatched solver, not a separate
function.

### B. Field-pipeline dead-end — **HIGH**
`RCWAResult` (`:1592-1636`) exposes efficiencies/absorptance/jones, but no
method producing a 2-D complex field / `JonesField` for propagation. Per-order
amplitudes (`rx/ry/tx/ty`, `:1815-1826`) are discarded; only the (0,0) order is
kept. `source → RCWA → propagate → detector` cannot be expressed.

### C. Uniform-Jones vs. spatially-varying metasurface — **HIGH (docs risk)**
`apply_reflection` broadcasts a single `(2,2)` to every pixel — correct for a
uniform periodic cell, but a metalens varies per pixel and `apply_jones_matrix`
already supports spatially-varying callables (`polarization.py:541-562`) that
RCWA never exposes. The module docstring oversells "drops straight into the
JonesField pipeline" (`:13-14`); the limitation is undocumented.

### D. Not composable into systems / prescriptions — **HIGH / MED**
`propagate_through_system` (`system.py:522`) has no `'rcwa'`/`'grating'` branch;
Zemax loader emits only `'mirror'`/`'surface'`. RCWA has zero composition
pathway.

### E. Missing input validation — **HIGH (correctness; quick fix)**
Runtime-confirmed: `depth<0` → **silently returns a wrong answer**
(`R.sum=0.1986`, no error); `period=0` → `ZeroDivisionError`; `n_orders<0` →
cryptic `zero-size array` ValueError; 2-D undersized `eps_cell` (Sx/Sy <
4·n_orders) → `LinAlgError` or silent aliasing. Only `polarization`/`duty_cycle`
are validated (`:437-443`); geometry is unguarded across all entry points and
`RCWAStack`. No tests cover these.

### F. Convention slips introduced in v5.5.0 — **HIGH severity, small effort**
- `rcwa_efficiency_1d_jax` error message lacks the `fn_name:` prefix (`:1447`,
  CONVENTIONS §2).
- JAX imported via bare `import jax` (`:1370`) with no `_JAX_AVAILABLE`/
  `find_spec` sentinel + no `pip install lumenairy[jax]` hint (§10).
- JAX path silently drops the duty-cycle/formulation validation **and** the
  Wood-anomaly regularization (`:492-501`) the NumPy path has.

---

## 3. Synergy / compatibility opportunities

1. **`RCWAResult.to_jones_field(nx, ny, dx, z=0)`** — reconstruct a sampled
   `JonesField` from the modal amplitudes the solver already holds. Closes B +
   the higher-order limitation. Pure addition.
2. **`tile_jones_response([RCWAResult], …)`** → spatially-varying callable Jones
   for `apply_jones_matrix` → metalenses/deflectors.
3. **Backend/GPU unification (keystone)** — one `array_namespace`-dispatched
   solver gives CuPy + JAX for free, deletes the fork, auto-provides JAX for
   2-D/Jones/stack.
4. **Grating/coatings/thin_grating unification** — `coatings` uses `s`/`p`/`avg`
   (`coatings.py:52-53`); `thin_grating`/RCWA use `te`/`tm`. Accept both aliases
   in RCWA, document the bridge in CONVENTIONS §7, add a cross-solver overlap
   test (RCWA↔TMM exists `test_rcwa.py:62-71`; RCWA↔thin_grating missing).
5. **`rcwa_to_ray_orders(result, …)`** — `raytrace/trace.py:806-950` already
   implements the grating equation; a converter feeds rigorous efficiencies +
   order angles to the tracer.
6. **`'rcwa'` element type** in `propagate_through_system`.

---

## 4. Library organization

- **Placement correct.** `elements/rcwa.py` at 1844 lines is within library
  norms (analysis modules run 1700-2300 un-split). No split for v5.5.0; promote
  to a `rcwa/` subpackage at v5.6+ if concerns diverge.
- **Re-export symmetry clean** (all names in `__all__` + `__init__`).
- **JAX surface asymmetric** — `rcwa_efficiency_1d_jax` is public with no
  2-D/Jones/stack twin and no "1-D only" docstring note; resolve via the backend
  refactor.

---

## 5. Code quality

- **Solver duplication (dominant).** `rcwa_efficiency_1d_jax` (`:1412-1559`)
  reimplements the whole 1-D algorithm with 5 local nested functions that
  duplicate the shared NumPy helpers and call **zero** of them → caused **real
  drift** (Wood-anomaly + validation absent from the JAX path).
- **Repeated physics blocks.** The R_eff/T_eff normalization block is duplicated
  across **7 sites**; the single-layer S-matrix triple across **5 functions +
  the `RCWAStack` loop**. Extract `_compute_efficiencies` / `_assemble_stack_smatrix`
  — but 1-D/2-D-aware (1-D anisotropic omits the `ky*ry` term).
- **Minor:** `add_layer` 4 mutually-exclusive specs; `period` vs `period_x`;
  1-D `angle` vs 2-D `theta`/`phi` (defensible but undocumented).

---

## 6. Prioritized action list

### Tier 1 — quick wins (small, non-breaking → v5.5.1)
1. Geometric validation (`period>0`, `depth>0`, `n_orders>=1`, `thickness>0`)
   with `fn_name:` prefixes on all functions + `RCWAStack`.
2. Enforce the 2-D aliasing bound (`eps_cell.shape >= 4*n_orders+2`).
3. Fix the JAX error prefix (`:1447`); add the missing JAX-path validation.
4. JAX lazy-import sentinel + `ImportError` hint (§10).
5. Document the uniform/zeroth-order/periodic limitation; soften the
   "drops-straight-in" claim; add a sign-convention note to `RCWAResult`.
6. **Fix `apply_reflection` in-place mutation** (apply to `jones_field.copy()`)
   — see §7 / companion P2-C.
7. Tests for bad geometry + aliasing rejection.
8. Accept `s`/`p` aliases for `te`/`tm`; CONVENTIONS §7 bridge.
9. **Qualify the CHANGELOG "<2e-3 per order" claim** — it is config-dependent;
   for high-contrast oblique TM, agreement with an under-converged oracle at
   matched truncation is worse (the solver is still correct — §7).

### Tier 2 — synergy additions (medium, pure additions → v5.5.x/5.6)
`RCWAResult.to_jones_field`; `'rcwa'` system element; `tile_jones_response` /
`rcwa_to_ray_orders`; cross-solver validation test + "which solver when" docs.

### Tier 3 — keystone refactor (large → v5.6)
`array_namespace`/`use_gpu` dispatch (folds the JAX fork, unlocks GPU,
auto-provides JAX to 2-D/Jones/stack; keep `rcwa_efficiency_1d_jax` as a
deprecated alias); then extract the shared 1-D/2-D-aware helpers.

---

## 7. Cross-references + triangulated correctness notes

The companion adversarial audit (`AUDIT_V5_5_0_2026_05_31.md`) raised two RCWA
correctness claims. The lead independently re-tested both:

- **P1-A (oblique-TM multi-order split allegedly wrong) — REFUTED (false
  positive).** Config Λ=1.2µm, λ=0.55µm, n=2.5/1.0, d=0.4µm, dc=0.4, θ=15°, TM.
  Triangulating three independent solvers to convergence: **lumen** (Li inverse
  rule) is *flat/converged* at Rtot≈0.5317 (M=40/80/120 → 0.5309/0.5316/0.5317);
  **grcwa** (Laurent) converges *downward toward it* (nG 251/401/551 →
  0.544/0.539/0.537); **inkstone** converges *upward toward it* (numG 101/201/301
  → 0.424/0.475/0.494). Both oracles approach lumen's value from opposite sides,
  so lumen is the correct converged answer and the companion compared to an
  **under-converged grcwa** (nG=151-251). This is the classic fast-Li-rule vs
  slow-Laurent-oracle convergence trap. **Action:** none on the solver; only
  *qualify the CHANGELOG "<2e-3 per order" claim* (Tier-1 #9) — it holds for the
  validated configs but is config-dependent.
- **P2-C (`apply_reflection` mutates its input JonesField) — CONFIRMED.** Runtime
  check: `res.apply_reflection(jf)` changes `jf.Ex` in place and returns the same
  object (`out is jf`). Same in-place class as the (already-fixed) PBS bug;
  `RCWAResult.apply_reflection` must operate on `jones_field.copy()`. **Action:**
  Tier-1 #6.

(The companion's P1-B substrate-cache-key collision and its P2-A/P2-B grazing/
non-propagating findings were not independently re-tested here; defer to that
report.)

---

*Generated from an 8-dimension multi-agent review with per-finding adversarial
verification (47/52 survived) plus parent-agent runtime corroboration and a
three-solver convergence triangulation of the companion's P1-A claim.*
