# RCWA roadmap — post-v5.14.0 audit (2026-06-10)

Companion to `docs/pmm_roadmap_v5_14.md`. Source: the 2026-06-10 RCWA
accuracy/generalizability/speed audit (46 findings; the verification phase was
cut short, so every item below that shipped in v5.14.1 was **hand-verified**
with independent probes before landing — and several audit numbers were
corrected in the process).

## Shipped in v5.14.1 (audit P1/P2 fixes — see CHANGELOG)

- **F1 inverse-z-rule** (P1): 2-D `'li'`/`'fff'`/`'auto'` is now the Li-1997
  sequential rule (inverse along the component's own axis, direct along the
  other, **direct-rule E_z elimination** — Li 1997 Eqs. 8/9/27); the shapes
  solver, `fff_nv`'s `EZZ`, and shapes layers in `RCWAStack` switched to the
  direct z-rule. Hand-verified: y-/x-uniform metal stripes reduce to rigorous
  1-D `'li'` per-order (5e-5 at S=256, second-order in S); metal-stripe
  absorptance error at M=16 drops +0.345 → +4.6e-3; on a true 2-D metal pillar
  the Richardson 1/M extrapolation of `'laurent'` (0.5717) lands on the
  sequential rule's converged value (0.5703), while inkstone at num_g≤1400 is
  still oscillating below — the old "gold" agreement was mutual unconvergence.
  The sequential rule also wins on STAIRCASED CURVES (a pixel cell is
  axis-aligned by construction): on the audited metal disk `'li'` is already
  in the converged absorptance band at M=13 (0.0351 vs converged ~0.029)
  while `'laurent'` is 2× high (0.0614) and crashes onto `'li'` by M=21
  (0.02949 vs 0.02890) — so `'li'` is the recommended formulation for ALL
  pixelated metal cells, straight or curved.
- **Gain superstrate** (P1): rejected at entry everywhere
  (`_require_propagating_incidence`); `_check_energy` is two-sided.
- **Silent sub-tripwire window** (P1): provably-lossless solves violating
  closure beyond 1e-6 emit `_EnergyWarning`; the `stabilize=` ladders treat
  that warning as a failed attempt (previously they returned the
  byte-identical wrong answer).
- **2-D stabilize wrong-abort** (P2): bumps past the cell-sampling bound are
  pre-filtered; the ladder keeps its documented `_EnergyError` contract.
- **Berreman conical oracle** (P2): `tests/unit/_berreman4x4._berreman_delta`
  3-entry `±Kx·Ky` fix (rotation covariance now 4e-16; RCWA 1-D conical
  matches the corrected oracle to 3.9e-15 incl. out-of-plane lossy tensors).
- **Docs** (P3): metal + explicit-`'laurent'`+TM warning in 1-D (unconverged
  at n_orders=128 on Ag, ~2-3e-2 off `'li'`).

## Performance levers (probed, not yet landed)

1. **1-D planar TE/TM decouple** (`RCWA-LEV-1`, P2): at Ky=0 the 2N system is
   exactly block-diagonal; a fixed-polarization call excites one N-block.
   Prototype reproduces shipped results; ~4-8× on large-N 1-D solves.
   `rcwa_jones_1d` keeps 2N (tensor cross-blocks couple).
2. **Diagonal-aware propagation star** (`RCWA-LEV-2`, P2): `_redheffer_star`
   with a propagation factor multiplies literal identity/zero blocks; the
   diagonal-aware form is algebraically identical (9e-16) and saves ~10% per
   2-D solve, ~20-30% per stack solve. Zero numerical risk.
3. **Symmetry scope** (`RCWA-LEV-3`, P2): the even-parity path is ×3.9-4.0
   end-to-end but covers only scalar single-layer `rcwa_efficiency_2d`.
   Extending to `RCWAStack.solve` (and `rcwa_jones_2d` at normal incidence)
   is the common metasurface case. NEW sub-item (v5.14.1): the sequential-rule
   `'li'` now routes through the tensor eigensolver, which the even-sector
   fold cannot represent — extending the fold to per-component tensor
   operators (Cxy=Cyx=0) would restore the ×4 for the recommended metal path.
4. `RCWA-LEV-4/5` (P3): analytic homogeneous-layer modes in stacks; diagonal
   K-matrix assembly micro-wins.
5. **JAX positioning** (`JAX-SPEED-HONEST`, P2): jit-warm forward is only
   ~1-2.4× wall (XLA intra-op parallelism, equal work, more CPU-seconds).
   Document JAX as the GRADIENT path; numpy + prepare/sweep for forward scans.

## Capability gaps (by leverage)

1. **Per-layer formulation in `RCWAStack`** (`GAP3`, P2, LOW-MODERATE):
   patterned iso layers inside stacks are Laurent-only — not even the (new,
   rigorous) `'li'`. Plumb `formulation=` through `add_layer`/`_layer_modes`;
   measurably faster metal multilayer convergence.
2. **Stack dispersion sweeps** (`GAP5`, P2, LOW-MODERATE):
   `RCWAStack.solve_vs_wavelength(wavelengths)` + `wl -> eps` callables in
   `add_layer` (1-D jones sweeps already accept callables; the stack and the
   2-D paths don't).
3. **2-D out-of-plane tensors** (`GAP2`, P2, MODERATE): feasibility proven —
   pointwise ezz-Schur fold on the cell + direct-rule convolutions of the
   effective components into `_layer_eigenmodes_tensor`'s existing 6-tuple
   branch + the generalized cascade (the identical pattern `pmm_jones_2d`
   shipped in v5.14.0; reuse its tests/oracles).
4. **Slant/shear walls** (`GAP1`, P2, LOW): `add_tapered_grating` only does
   symmetric trapezoids; a `shear`/`wall_angle` parameter (shifted-centre
   profile) covers sheared walls. Staircase accuracy class ~1e-3 at 16-32
   slices — document it.
5. **`fff_nv` rework-or-retire** (`F2`, P2): its NV field uses a fixed PIXEL
   smoothing width (`sigma_px=1.5`), so the answer depends on the sampling
   resolution S and converges to the in-plane-Laurent result as S grows; with
   the corrected direct z-rule it no longer beats `'li'` anywhere measured
   (axis-aligned: `'li'` is rigorous and faster-converging; curved: the
   cross-term mis-split still raises by default). Either make the smoothing
   width physical + validate a curved-wall cross term (research-grade), or
   deprecate in favour of `'li'` + the shapes solver.
6. `GAP4` µ/bianisotropic, `GAP6` hex lattices + parallelogrammic truncation
   (P3, research-class).

## P3 hygiene backlog

- `GPU-DLL-IMPORTERROR-NOT-CLEAN`: probe a trivial CuPy device op inside the
  `use_gpu` guard and re-raise the friendly RuntimeError (partial CUDA wheel
  installs currently die deep in cufft).
- `GPU-EIG-CUPY14-PIN`; `SHAPES-JAX-REJECT-DEAD-CODE`;
  `STACK-RETAIN-INTERNAL-SILENT-DROP`; `SWEEP-JAX-SILENT-NUMPY`;
  `RCWA-NAN-SUBSTRATE-SILENT`.

## Validated-clean (no action)

1-D core vs Fresnel/Berreman at machine precision; non-separable fff_nv guard
works; seam checks exact; energy conservation drift none beyond the audited
window; JAX twins value-match 5e-15.
