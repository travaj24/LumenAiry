# Deferred-Items Plan: KMAH caustic sum, GPU backends, perf/memory (2026-07-21)

Status: PLAN — campaign on `feat/kmah-gpu-perf` (base main @ a098091, v5.26.0).

User request (2026-07-20, after the v5.26.0 release): tackle the two deferred
items from the accuracy-niches campaign plus a perf/memory sweep, with Opus
subagents and the same impl → adversarial-verify → fix harness.

Method rules (binding, inherited from the accuracy-niches campaign):

- Fail-before / pass-after oracle-backed tests for every behavior change.
- Defaults byte-identical unless a validated accuracy/perf improvement (pinned);
  and byte-identity is asserted with a TOLERANCE (1e-10·scale), never
  `array_equal`, between two live FFT/cache propagator calls (cache-warmth ~1 ULP
  lesson from the v5.26.0 CI).
- Every perf/memory claim MEASURED (wall-clock + peak bytes), and the adversarial
  verifier independently reproduces the number.
- Every new cache bounded + registry-enrolled + releasable.
- Big-N tests carry a RAM-skip guard (`lumenairy.memory.available_memory_bytes`)
  so a 16 GB CI runner never OOMs (v5.26.0 lesson).
- ASCII-only source; single pytest runs < 10 min, < 16 GB.
- Independent oracles: fold-caustic ground truth (`validation/oracles/
  caustic_fold_truth.py` / `caustic_fold_ref.npz`), Debye/geometric raytrace,
  ZOS-API POP + Huygens PSF, ABCD q-traces, energy conservation.

---

## N13. Multibranch KMAH / Maslov caustic amplitude for traced ray-density

- **Current state (v5.26.0, N12):** `apply_real_lens_traced(amplitude_model=
  'ray_density')` uses the single-branch geometric ray-tube amplitude
  `|E_in| / sqrt(|det J|)`, and at a fold (det J → 0) it only DETECTS +
  floors/caps + warns.  So it cannot represent a FOCAL decentered PSF (the focus
  is a caustic) or a genuinely multi-valued fold/cusp caustic.
- **Approach:** connect the ray-density amplitude to the EXISTING
  `traced_multibranch` / det-Q KMAH machinery (from the caustic-papers round).
  Where the ray map folds, gather ALL real ray branches reaching each output
  point; weight each branch `|E_in(x_in^b)| / sqrt(|det J_b|)`; apply the Maslov
  phase `exp(-i·(π/2)·KMAH_b)` where the KMAH index is the number of sign changes
  of det J along that branch's ray; SUM COHERENTLY.  Expose it as an opt-in
  refinement of the `ray_density` mode (e.g. `caustic='multibranch'`), default
  single-branch (byte-identical).  Reuse the existing multibranch branch-finder
  and KMAH counter — do NOT reimplement.
- **Adversarial validation:**
  1. **Fold ground truth** (`caustic_fold_ref.npz`): the multibranch ray-density
     field at the fold plane must beat the current single-branch exit-field+ASM
     (~3%), targeting < ~1–2% windowed r2m/EE, and must NOT blow up.
  2. **Airy transition at the fold:** the √-singularity of the single-branch
     amplitude must resolve into the finite fold-diffraction (Airy-function)
     profile — check the peak stays finite and the fringe spacing matches the
     Pearcey/Airy scaling.
  3. **KMAH correctness:** the per-branch Maslov phase must be right — verify by
     comparing a two-branch region's coherent sum against the direct Huygens
     integral (a wrong ±π/2 flips a bright fringe to dark).
  4. **Focal decentered PSF:** with multibranch, traced at the IMAGE plane of a
     decentered element must broaden to within ~15% of GBD / geom / ZOS (the N12
     gap) — or, if a residual remains, document the measured envelope honestly.
  5. **Energy** conserved through the fold (< 0.5%); single-branch default
     byte-identical (tolerance-pinned); collimated unaberrated unchanged.
- **Risk (honest):** branch-finding across the full 2-D grid at production N may
  be expensive; if so, measure the cost and gate it (opt-in + a coarse-grid
  branch map).  If the coherent sum cannot reach the fold-truth target, document
  the residual and keep GBD/FGA as the caustic reference — never inf/nan.

## N14. CuPy + JAX backends for the astigmatic / apertured carrier ASM

- **Current state (v5.26.0, N7):** `propagators/carrier.py` (Sziklas–Siegman
  pilot-beam, astigmatic `carrier=(R_x, R_y)` + focus crossing + aperture re-fit)
  is NumPy-only.
- **Approach:** backend-abstract the array namespace (`xp` ∈ {numpy, cupy,
  jax.numpy}) and route the FFTs + envelope-grid ops through it; keep the
  algorithm identical.  Handle: complex dtype preservation; the focus-crossing
  split control flow (straightforward eager for CuPy; JAX needs `lax.cond` or an
  eager/host-branch fallback — the split is data-dependent, so a jit-compatible
  form or a documented eager path); FFT-plan reuse; the per-axis astigmatic
  magnification.  JAX additionally yields differentiability (grad through a
  carrier leg) — validate a gradient if cheap.  Backend selected via the existing
  library backend switch (mirror how RCWA/PMM pick numpy/jax) — do NOT invent a
  new mechanism.
- **Adversarial validation:**
  1. **Backend parity:** CuPy and JAX results match NumPy to < 1e-6 (relative) on
     the astigmatic line-foci + circle-of-least-confusion case AND an apertured
     converging leg; the NumPy path stays byte-identical (tolerance-pinned).
  2. **Speedup + memory** measured on a representative grid (e.g. N=2048–8192):
     report wall-clock and peak bytes per backend; the verifier reproduces them.
     (If no GPU is present in the harness, CuPy parity/speed is measured where
     possible and otherwise documented as untested-here with the code path
     import-guarded; JAX runs CPU-XLA at minimum.)
  3. **Focus crossing** works on every backend (both line foci + the waist);
     isotropic input reduces to the scalar path on every backend.
  4. **No silent dtype upcast** (the v4.14.1 audit contract) on any backend.
- **Out of scope:** new physics — this is a backend port, accuracy-neutral.

## N15. Perf / memory sweep across the accuracy-niches updates

- **Scope:** profile and optimize the hot paths added by v5.26.0 where it is
  free (byte-identical) or a validated improvement — the traced ray-density
  amplitude + decenter geometry, the displaced 2-D remap scatter, the GBD
  decenter path, the carrier ASM, the Seidel gate, and adaptive FGA.
- **Approach:** profile (cProfile + tracemalloc / peak-RSS) each path at a
  representative grid to find the real hot spots; then apply the free wins —
  candidates (measure, don't assume): cache the ray-map Jacobian / reuse the
  Newton fit between the OPL and the ray-density amplitude (they already share
  the entrance→exit fit — confirm no double-trace); vectorize the pointwise 2-D
  obliquity; reduce the 2-D remap scatter's peak memory (chunk the scatter /
  in-place accumulation / avoid a full dense intermediate); pre-broadcast reuse
  in the astigmatic ASM; bound/optimize any per-call allocations.  Each new or
  modified cache stays bounded + registered + releasable.
- **Adversarial validation:** every optimized path is byte-identical
  (tolerance-pinned) OR a validated improvement with an oracle; the verifier
  INDEPENDENTLY reproduces each speedup / memory-reduction number (a perf claim
  that does not reproduce is a kill); no accuracy regression on the full
  accuracy-niches test set; a perf regression test (or a documented measurement)
  guards the win.
- **No silent caps:** if an optimization changes a default (e.g. a new cache), it
  is documented and the memory footprint measured.

---

## Execution

| Phase | Item | Depends on |
|---|---|---|
| K1 | N13 multibranch KMAH ray-density caustic sum | — |
| K2 | N14 CuPy + JAX carrier-ASM backends | — |
| K3 | N15 perf / memory sweep | K1, K2 (so it profiles the final code) |

Sequential single-writer Opus agents; each phase = implementer → adversarial
verifier → (on kills) fixer → re-verify, max two rounds; unresolved kills are
documented open findings.  Checkpoint commit after each phase; run the touched
subsystem + a cross-file batch at each checkpoint (the v5.26.0 CI lesson).
Release only on explicit user approval.
