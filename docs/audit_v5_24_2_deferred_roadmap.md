# Deferred-items roadmap — v5.24.2 exhaustive audit

Companion to `docs/audits/AUDIT_V5_24_2_2026_07_17_EXHAUSTIVE.md`.

The v5.24.4 release remediated **88** audit findings (P1 ×11, P2 ×23, P3 ×54).
**10 P3 items were deliberately deferred**, each with a written rationale (also
recorded in the `cfc02d4` commit message). This document is the actionable
backlog for those 10: what each is, *why* it was held, the implementation
approach, the specific challenges, and an effort/risk estimate.

Deferral was governed by two rules from the remediation pass:

- **Byte-identical rule** — a "de-duplication" may only ship if the extracted
  helper reproduces every former call site to machine precision. Where the
  "duplicate" copies do **not** actually agree, consolidating would either bury
  a latent bug or silently change numerics, so it was stopped.
- **Hygiene-scope rule** — a P3 hygiene pass does not make breaking public-API
  changes or take on multi-release deprecation cycles; those are flagged here.

**Recommended order:** do **S2-14-FGA first** — it is the only item deferred for
a now-obsolete reason and is a clean byte-identical win. Then the two low-risk
numeric-reconciliations (S3-19-metrics, S2-10). The API-change items
(S3-16, S5-10-BOR, S4-18-scale) should be batched into a single deprecation
cycle. S4-20 needs only a confirming comment.

---

## Category A — not (yet) byte-identical-consolidatable

### A1. S2-14-FGA — consolidate the FGA HK-prefactor + coeff-prune duplication  ⭐ NOW UNBLOCKED
- **Where:** `propagators/fga.py` — scalar/vector Herman–Kluk prefactor
  (`~:706` / `~:1012`) and the coefficient-prune pre-pass (`~:643` / `~:945`).
- **Why deferred:** `fga.py` was the actively-developed file on the release
  branch (`feat/fga-coarse-lattice-cache`); consolidating mid-flight risked
  colliding with the in-flight FGA-lever work. **That reason is now gone.**
- **Approach:** extract the HK-prefactor `a = sqrt(det Z)` (with the branch
  `a = where(a.real<0, -a, a)`) and the coeff-prune pre-pass into shared helpers;
  route the scalar and vector paths through them.
- **Challenges:** the scalar path uses a scalar Jacobian while the vector path
  carries a 2×2 (see the still-open S2-16); confirm the *prefactor* computation
  is genuinely identical between them before merging, and keep the Jacobian
  difference in the caller, not the shared helper. Guard the byte-identical
  assertion behind the `numba` importorskip that the FGA tests already use.
- **Effort:** low–moderate. **Risk:** low (byte-identical verifiable).

### A2. S2-10 — consolidate the ASM transfer-function construction (~6 copies)
- **Where:** `propagators/asm.py:116,184,224,842(tilted)`,
  `propagators/mft.py:190,220`.
- **Why deferred:** the copies **do not agree today**. The S2-3 mod-2π
  complex64 mitigation landed in only one copy (`_get_asm_H_natural`'s c64
  branch, ~`asm.py:253`, and the tiled branch fold the phase mod 2π before the
  float32 cast); the other builders do not. A naive merge would either bury the
  divergence or silently propagate the fix everywhere (a numeric change).
- **Approach:** (1) first *decide* whether the mod-2π-before-f32 mitigation
  should apply to all H builders (it almost certainly should — it is a
  correctness fix, not a per-site choice); (2) apply it uniformly with a
  physics test at the f32 boundary; (3) *then* extract the one parametrized
  `_build_asm_H(...)` builder.
- **Challenges:** the consolidation is **not** byte-identical — it fixes a real
  cross-copy divergence, so it needs a numerical-change regression test (f32
  phase-wrap at large `z`), not a byte-identical oracle. Sequence matters:
  reconcile → validate → consolidate.
- **Effort:** moderate. **Risk:** low–medium (ASM H at the f32 boundary).

### A3. S3-10 — de-duplicate conic-intersection + Snell (4 sites)
- **Where:** `raytrace/intersection.py`, `raytrace/differential.py`,
  `raytrace/seidel.py` (paraxial ×2), `raytrace/jax_trace.py`.
- **Why deferred:** the four are **intentionally different numerical
  strategies**, not copies: `intersection.py` is the exact NumPy solver
  (spherical fast-path + Newton), `differential.py` the ADRT differential form,
  `seidel.py` the paraxial reduction, `jax_trace.py` the JAX-traceable form.
  Forcing one strategy onto all would change numerics and/or differentiability.
- **Approach:** do **not** unify the strategies. Instead extract a small
  backend-agnostic *shared core* — the conic sag `z(r)` and Snell refraction —
  that each site calls while keeping its own outer strategy (exact vs paraxial
  vs differential vs jax).
- **Challenges:** the shared core must run on both NumPy and JAX (no in-place
  writes); it must not impose the exact-solver's Newton loop on the paraxial
  site. The removed Seidel `_paraxial_trace` (audit RT-3) is prior evidence this
  family has shipped bugs, so a sag+Snell shared core is worth doing — carefully.
- **Effort:** moderate–high. **Risk:** medium (core ray tracing).

### A4. S3-19-metrics — unify the JAX inline single-plane metrics with `single_plane_metrics`
- **Where:** `analysis/through_focus.py:~1263` (JAX inline block) vs
  `single_plane_metrics` / `radial_power_bands`.
- **Why deferred:** genuine latent divergence — `power_in_bucket` uses a strict
  `r < bucket_radius` mask in the JAX block vs `R² <= r²` (**`<=`**) in
  `radial_power_bands` (they differ at a pixel exactly on the boundary); and
  `rms_radius` is an *independent* second moment in the JAX block vs
  `beam_d4sigma`-derived in the NumPy twin (agree ~1e-15, not byte-identical).
  v5.24.4 documented the divergence in-code and added a smooth-field parity test
  rather than merging.
- **Approach:** pick the canonical boundary convention (`<` vs `<=`) and the
  canonical `rms_radius` definition (independent 2nd moment vs d4σ), align both
  implementations to it, then consolidate.
- **Challenges:** this is a numerical-*convention* decision that changes results
  at band edges; ship it with a parity test at an agreed tolerance, not a
  byte-identical oracle, and note the band-edge change in the CHANGELOG.
- **Effort:** low–moderate. **Risk:** low (boundary pixel only).

---

## Category B — public-API changes needing a deprecation cycle

Batch these into one coordinated deprecation release (accept old **and** new
forms with a `DeprecationWarning`, update tests, remove the old form a release
later).

### B1. S3-16 — source-factory convention normalization
- **Where:** `sources/core.py` — `seed` kwarg violates the CONVENTIONS.md `rng`
  rule (`:1966,2128,2238,2956,3060`); normalization inconsistent across the
  factory zoo (peak vs power vs raw, `:223,384,542,943,1005,1437`); `sigma` vs
  `w0` naming (`:218`). *(The dead `N` param in `_schell_phase_realizations`,
  `:1890`, was already removed in v5.24.4.)*
- **Approach:** (1) add `rng=` alongside `seed=` (deprecate `seed`); (2) add an
  explicit `normalize=` kwarg with a default that preserves current behavior,
  deprecating the implicit per-factory convention; (3) add a `w0=` alias for
  `sigma=`.
- **Challenges:** ~5–8 factories, each with tests pinning the current
  `seed=`/`sigma=`/normalization contract; must not break existing user code —
  hence the two-release deprecation cycle.
- **Effort:** high. **Risk:** medium (public API surface).

### B2. S5-10-BOR — BOR result/terminology/export normalization
- **Where:** `elements/bor/*` — mode/order terminology inconsistency, dict
  return shape, and BOR not top-level exported. *(The incidence-angle and
  te/tm-vs-x/y Jones-basis parts of S5-10 were fixed in CONVENTIONS.md + the
  2-D docstrings in v5.24.4.)*
- **Approach:** settle canonical BOR terminology (mode vs order), migrate the
  dict return to a structured result container matching the other engines
  (coordinate with the still-open S5-6 solve-result-container unification), and
  add top-level re-exports.
- **Challenges:** the return-shape change is breaking; do it together with S5-6
  so BOR/RCWA/PMM/Berreman converge on one container rather than churning twice.
- **Effort:** moderate. **Risk:** medium (API + return shape).

### B3. S4-18-scale — merit-family scale unification (opt-in only)
- **Where:** `optimize/merit_terms.py:66,1208` — merit families on wildly
  different scales (normalized vs dioptre² vs absolute m²).
- **Why deferred:** re-scaling any family would silently change **every existing
  weight calibration** users have tuned.
- **Approach:** introduce an **opt-in** normalization (a scale-aware merit
  wrapper or a `normalized_scale=` flag) that rescales families to a common
  scale, leaving the default untouched; document each family's native scale.
- **Challenges:** defaults must not change; the opt-in path needs its own
  calibration guidance so users can migrate deliberately.
- **Effort:** moderate. **Risk:** low (opt-in, default unchanged).

---

## Category C — dedicated work / already-correct

### C1. S4-19-storage — faithful nested-metadata round-trip
- **Where:** `io/storage.py:821,1184` — `list -> ndarray` coercion on the h5
  backend; un-reversed dict flattening. *(The `replay_run` stored-wavelength and
  stale-`.lock`-docstring parts of S4-19 were fixed in v5.24.4.)*
- **Approach:** define a canonical metadata serialization contract (preserve
  Python `list` vs `ndarray`; preserve dict ordering) honored identically by the
  h5 **and** zarr backends; add round-trip fidelity tests on both.
- **Challenges:** h5 and zarr have different native type handling, so a shared
  type-coercion/serialization layer is required; a partial (one-backend) fix
  would create backend-divergent behavior — which is why a drive-by fix was
  refused.
- **Effort:** high. **Risk:** medium (persisted format).

### C2. S4-19-user_library — replace `eval()` on expression masks
- **Where:** `io/user_library.py:585` (`eval()` with full `np` exposed —
  flagged "known/tracked" in the finding) and `:74` (silent clobber +
  sanitized-name collisions).
- **Approach:** replace `eval()` with a restricted AST-based safe evaluator
  (whitelisted numpy ops) or a `numexpr`/parser-based expression engine; add a
  collision warning for sanitized-name clashes.
- **Challenges:** the `eval()` is a *feature* — it lets users write arbitrary
  mask expressions. A safe replacement must preserve that expressiveness while
  blocking arbitrary code execution, which is a genuine security-design task
  (whitelist scope, error messages), not a one-liner. Deferred pending that
  dedicated review.
- **Effort:** high. **Risk:** medium (security-sensitive + must preserve feature).

### C3. S4-20-lensmaker — verify-and-document only (likely not a defect)
- **Where:** `ui/main_window.py:2288` — lensmaker seed `R = f(n-1)`.
- **Assessment:** the cited plano-convex seed is **already thick-lens-exact**
  (`R2 = inf` zeroes the thick-lens term); the biconvex/other builders carry an
  **intentional** thin-lens *seed* approximation that the optimizer then refines.
- **Approach:** no code change required. Optionally add a one-line comment at the
  seed noting it is a deliberate thin-lens initial guess (not a target), so a
  future reader does not "fix" a non-defect.
- **Effort:** trivial. **Risk:** none.

---

## Summary table

| Item | Category | Effort | Risk | Blocking reason |
|------|----------|--------|------|-----------------|
| **S2-14-FGA** | A (dedup) | low–med | low | *none — unblocked* |
| S2-10 | A (dedup) | moderate | low–med | copies disagree (mod-2π c64) |
| S3-10 | A (dedup) | mod–high | medium | intentionally different strategies |
| S3-19-metrics | A (dedup) | low–med | low | `<` vs `<=` band-edge divergence |
| S3-16 | B (API) | high | medium | needs deprecation cycle |
| S5-10-BOR | B (API) | moderate | medium | breaking return-shape (pair with S5-6) |
| S4-18-scale | B (API) | moderate | low | would break weight calibrations |
| S4-19-storage | C | high | medium | dual-backend serialization contract |
| S4-19-user_library | C | high | medium | eval() security review |
| S4-20-lensmaker | C | trivial | none | already correct (verify+document) |
