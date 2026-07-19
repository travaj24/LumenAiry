# Real-lens hammer campaign — findings + resolutions (2026-07-19)

**Scope:** adversarial validation of the real-lens propagator family
(`apply_real_lens` analytic, `_traced`, `_gbd`, `_fga`) against external
oracles, following the 2026-07-18 thin-lens audit's item 5.

**Oracles (two, fully independent):**

1. **Zemax OpticStudio POP via ZOS-API** (zospy 2.1.5, py3.13 venv) —
   prescriptions built with *dispersionless model glass* (vd=0) and the
   exact per-surface indices read back via `INDX` operands, EFL
   cross-checked via `EFFL` (agreed with hand thick-lens calculation to
   6 digits). No index/geometry ambiguity by construction.
2. **Exact-raytrace Debye/Huygens integral** (`debye_oracle.py`,
   session scratchpad) — a self-contained meridional exact raytrace +
   ring-Huygens sum. No FFT, no lumenairy, no Zemax.

Oracle cross-agreement: **0.5–8 % on r2m across every case** (plano-convex
good orientation: 43.22 vs 43.00 µm — 0.5 %). The POP pilot-beam caveat
was explicitly checked and cleared by the second oracle.

**Test cases:** biconvex singlet R = ±51.68 mm, t = 5 mm, n = 1.5168 at
λ = 1.31 µm (benign w0 = 0.5 mm and f/5 w0 = 5 mm illumination);
plano-convex f ≈ 51.7 mm in both orientations (textbook ~4× spherical
penalty reproduced: oracle 43/128 µm); through-focus scans.

## Final verdicts

| model | verdict |
|---|---|
| `traced` | **Validated: 99.7 %** of dual-oracle r2m (64.77 vs 64.98 µm) at rule-compliant sampling; exact ray OPL to λ/44 |
| `gbd` | **Validated: 98.4 %**, grid-insensitive (63.96/63.97 µm at dx = 3/6 µm) |
| `fga` | Feasible + sane by default after H4/H5; r2m within ~2 % with matched sampling; GBD/traced remain preferable for smooth single-valued caustics (cost) |
| `analytic` | Honest model plateau: converged 40.5 µm vs oracle 65 (H2); ~50 µm with the H1-fixed slant screen. Cannot represent orientation-dependent aberration (60.4/60.9 µm where truth is 43/128) |
| benign regime | all models exact to 4 digits vs Zemax (29.981 vs 29.979 µm) |

## Findings

### H1 — FIXED: `slant_correction` had inverted cosines (physics-derivation error)

The slant OPD computed `n·sag/cosθ` (ray path-length through a slab)
where the wavefront OPD of a locally tilted refracting facet is
`(n₂cosθ_t − n₁cosθ_i)·sag` — cosines in the **numerator**. The inversion
sign-flips the leading obliquity/spherical term; on a symmetric biconvex
the two wrong-signed corrections cancelled the pupil SA → an impossible
3.6 µm near-diffraction-limited spot vs the 65 µm truth. Fixed in both
byte-identical copies (`_lens_real.py` banded + whole-grid) + docstrings
+ UI tooltip. Corrected slant measures 50.3 µm — the correct-signed term
moves the analytic model *toward* the oracle. `analysis/through_focus.py`
used `slant_correction=True` internally as its "ideal pupil" (it looked
sharp *because* of the bug); goldens derived from it shift.
Tests: `test_hammer_h1_slant_obliquity.py`.

### H2 — QUANTIFIED: analytic sag-screen validity envelope

Per-surface phase screens structurally cannot represent the transverse
ray-displacement physics of orientation-dependent aberration (same
conclusion as the 2026-07-18 audit, now dual-oracle-quantified). Use
`traced`/`gbd` for absolute spot fidelity on aberrated systems; analytic
remains exact in the benign regime.

### H3 — FIXED: traced's sampling rule now enforced (model itself exact)

`traced`'s documented critical-sampling rule (`dx ≤ λ·f/aperture`) was
never enforced; violating it silently aliases the exit converging
wavefront beyond grid Nyquist (folds 8.5 % of energy → r2m reads 37 %
low while EE50/EE80 stay plausible). At compliant dx the model is
**99.7 %** of the oracle. Added an amplitude-aware exit-NA Nyquist
warning (rays ≥ e⁻⁴ of peak input amplitude define NA_exit; warns when
`dx > λ/(2·NA_exit)`; `on_undersample='silent'` suppresses; never
raises). Tests: `test_hammer_h3_traced_nyquist_guard.py`.

### H4 — FIXED: FGA auto-path memory wall

Peak memory ≈ Nq·Np·(80–130 B) with the (Nq, Np) coefficient matrix
allocated unchunked (29.2 GiB at N=8192) because (a) the existing
byte-identical chunk loop was gated on `mem_budget_mb=None`, (b) the
cost model ignored the 9-ray FD Jacobian bundle (~6× under-count),
(c) the FD bundle itself is unnecessary for all-conic prescriptions.
Fixed: default RAM-fraction budget, FD-bundle-aware cost model,
`exact_jacobian` auto-default for conic-only prescriptions.

### H5 — FIXED: FGA auto-sampler halo on near-collimated strong-focusing input

`coarse_stride` was **exonerated** (byte-faithful, rel L2 2.4e-4). The
5×-wrong result came from `_resolve_sampling` flooring `p_max` at the
S2-2 beamlet-completeness width `3/(k·w0)` — 32–130× the field's real
angular content for near-collimated input — whose excess-momentum
beamlets spray a halo ~`efl·p_max` through a strong lens. Fixed: when
floor/content > 10, p_max is content-sized (×3 tail) with the
completeness energy restored via output power normalization + a
RuntimeWarning. S2-2 identity-power and diverging-beam contracts
unchanged and re-verified. Tests: `test_fga_h4_h5.py`.

## Method lessons (for future campaigns)

- **Windowed r2m is domain-sensitive; EE curves + through-focus r2m
  curves are the robust cross-tool metrics.** A full-grid second moment
  is dominated by r²-weighted background at 1e-6 relative intensity.
- **Grid-coverage trap:** the input beam must fit the grid
  (±>2.4 w0); `P_out/P_in` computed post-grid hides input truncation.
- **Beamlet methods (GBD/FGA) are output-grid-insensitive** (phase rides
  analytically); phase-screen models are not — sampling rules differ
  fundamentally between the two families.
- **Best-focus curves discriminate mechanisms:** a shifted curve means a
  reference offset; a deeper minimum means lost aberration.
- **Platform notes:** Windows `np.longdouble` IS float64 (oracle
  pitfalls); Linux glibc's complex exp can beat an f64 mod-2π fold —
  assert platform-independent *bounds*, not per-platform orderings.

## Follow-ups (tracked, not blocking)

- Extended oracle matrix: 4f relay at finite conjugates (121-chain
  class), cemented doublet, point-source (ZOS Huygens PSF oracle),
  FGA's true multi-valued-caustic specialty, universal-dispatcher
  routing (must not route heavily-aberrated prescriptions to analytic).
- Re-baseline any goldens derived from `through_focus`'s internal
  slant-corrected reference fields (H1 blast radius).
- The traced JAX×OpenBLAS lstsq deadlock library-side mitigation
  (threadpool_limits(1) when jax is co-imported) — from the v5.24.4
  cycle, still open.
