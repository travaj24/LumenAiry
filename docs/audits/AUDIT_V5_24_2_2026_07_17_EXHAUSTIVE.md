# Exhaustive Library Audit -- v5.24.2 -- 2026-07-17

**Scope.** Whole-library deep audit of lumenairy at v5.24.2 (branch
`fix/audit-followups-2026-07-17`, HEAD `3b293ca`): feature gaps and seams
between subsystems, physical correctness, code-convention compliance,
library organization, and speed / memory-footprint opportunities that do
not sacrifice accuracy.

**Method.** Staged audit: one reconnaissance pass, then five focused
stages of at most five Opus auditors each (electromagnetic engines;
wave propagators and lens models; rays / sources / analysis / IO;
optimization / GUI seam / infrastructure; cross-cutting seams, feature
gaps, performance), followed by an adversarial verification pass on
every P0/P1 and every load-bearing P2 before it is accepted into this
document.  Auditors were instructed: read-only; cite `file:line` at
HEAD; confirm physics findings with small numerical probes where
feasible; prefer independent oracles over energy-conservation-only
arguments (the "lossless trap"); explicitly hunt duplicated physics
blocks (history: fixes that land in one of two copies); check the
module roadmap docs so known-deferred items are not re-reported as
discoveries.

**Environment.** Windows 11, Python 3.14.6, `import lumenairy` resolves
to this working tree; jax 0.10.0, numba 0.65.1, pyfftw 0.15.1,
numexpr 2.14.1, zarr 3.1.6, PySide6 6.11.0 all importable.

**HEAD-moved note.** Mid-audit (during Stage 1/2), the working tree
advanced from `3b293ca` (v5.24.2, branch `fix/audit-followups-2026-07-17`)
to `2628426` (v5.24.3, branch `feat/fga-adaptive-sampling`).  The delta
touches ONLY the FGA module (auto phase-space sampling).  Stage 1
findings are therefore valid at both; Stage 2 onward audited the code
actually present at `2628426`, including the new FGA auto-sampler.

**Severity.**

| Sev | Meaning |
|-----|---------|
| P0  | Wrong physics or data loss on a default, documented path |
| P1  | Wrong physics or silent accuracy loss reachable through the public API |
| P2  | Edge-case correctness, robustness, or a no-loss perf / memory win |
| P3  | Hygiene, documentation, conventions, organization |

Categories: `physics`, `seam`, `gap`, `perf`, `memory`, `convention`,
`organization`.

**Status: COMPLETE (2026-07-17).** 30 auditors + 5 adversarial
verifiers across 6 stages.  12 P1 (10 code + 2 test-oracle, ALL
independently verified), 34 P2, ~60 P3.  Zero P0.  Remediation NOT
started.  Executive summary in the Rollups section at the end.
Final HEAD during verification: `826e606` (v5.24.3 + CI commit).

---

## 0. Reconnaissance summary

Package layout (lines of Python, `__pycache__` excluded):

| Subtree | LOC | Contents |
|---------|-----|----------|
| `lumenairy/elements/` | 49,120 | RCWA, PMM (1-D/2-D), Berreman, BOR, EME, lens models (traced / maslov / real), DOE, polarization |
| `lumenairy/ui/` | 32,063 | Qt designer GUI |
| `lumenairy/propagators/` | 21,850 | ASM/fft_infra, GBD, FGA, system graph, HFPI, asymptotic fits |
| `lumenairy/analysis/` | 16,290 | field metrics, PSF/MTF/OTF, through-focus, AO, plotting |
| `lumenairy/raytrace/` | 9,519 | numpy trace, jax twin, Seidel |
| `lumenairy/optimize/` | 6,991 | design optimizer, merit terms |
| `lumenairy/io/` | 6,733 | Zemax prescriptions, storage (zarr), user library |
| `lumenairy/sources/` | 3,153 | source factories |
| `lumenairy/algebra/`, `backend/`, `_math/` | 3,398 | field algebra, xp backend, math kernels |
| root modules | ~8,000 | `__init__` (1,894), `glass.py` (1,695), validation, caches, logging |

Tests: 288 unit files + 3 integration + 45 validation files (separate
oracle suite under `validation/` with `run_all.py`).  ~30 prior audit
documents under `docs/audits/`; module roadmaps under `docs/`
(`lens_propagators_roadmap.md`, `PMM_ROADMAP.md`, `rcwa_roadmap_v5_14.md`,
`pmm_bor_axisymmetric_roadmap.md`, `audit_universal_dispatcher_2026_07_16.md`).

Working-tree note: one uncommitted modification predates this audit
(`docs/audits/AUDIT_DYNAMETA_CONSUMER_API_GAPS_2026_07_13.md`); left
untouched.

---

## Stage 1 -- Electromagnetic engines (RCWA, PMM 1-D, PMM 2-D, Berreman/EMT/polarization/glass, BOR/EME)

Five auditors (RCWA; PMM 1-D; PMM 2-D; Berreman+EMT+polarization+glass;
BOR+EME).  Net: the engines' core physics is in strong shape -- every
historically hot area (OOP factor-i, A|B mirror kernel, slant signs,
grazing cutoff, loss convention) was re-probed against independent
oracles and holds.  One P1, three P2, and a set of P3s below.

### S1-1 [P1][physics/seam] `PMMStack.prepare()` silently truncates out-of-plane tensors -- `elements/pmm/stack.py:2112,2176`

`_PreparedPMMStack` has no OOP guard: `_layer_eig` builds
`_build_sem_tensor_segments` (reads only `exx,exy,eyx,eyy,ezz`) and
diagonalizes with the in-plane 2n solver, silently dropping
`exz/eyz/ezx/ezy`.  Probe: tilted-uniaxial layer, `solve()` vs
`prepare().solve()` gives `max|dJ| = 5.9e-3` and both conserve energy
(so `_warn_stack_energy` never trips).  Tilted-LC-director sweep (the
documented use case of `prepare()`, stack.py:1950) shows 0.34 deg
retardance error, exactly zero at tilt=0.  `solve()` routes OOP
correctly and `solve_vs_wavelength` RAISES for the identical input --
`prepare()` is the one silent-wrong path of the three.
Fix: reject OOP tensors in `_PreparedPMMStack.__init__`/`_resolve`
(mirror the `solve_vs_wavelength` guard) or promote the prepared path
to the generalized cascade.  Confidence: high (probe-pinned).

### S1-2 [P2][physics] RCWA lossless per-order tripwire absent on `RCWAStack.solve` and `rcwa_jones_1d` -- `elements/rcwa/stack.py:2208`, `elements/rcwa/oned.py:1041`

`_check_energy(..., lossless=)` warns when a provably-lossless solve
violates closure in the silent `1e-6..0.05` window (the 2026-06-10
audit-P1 guard).  `rcwa_efficiency_1d/2d`, `rcwa_jones_2d`, and
`_shapes` pass `lossless=`; `RCWAStack.solve` and `rcwa_jones_1d` do
not (flag defaults False, so the guard is dead there).  A lossless
stack solved at an unstable `(period, n_orders)` coincidence returns
per-order efficiencies wrong by up to a few percent with no warning.
Fix: compute joint losslessness (`_layer_eps_reals` at stack.py:1414
already enumerates layer eps) and pass the flag at both call sites.

### S1-3 [P2][physics] PMM-2D hybrid returns non-physical energy silently at the order-representability boundary -- `elements/pmm/twod.py:1043,955,1071`

`_validate_cell_orders` raises only when `2*n_orders+1 > nodes`;
at/near equality the Fourier projector pinv is ill-conditioned and,
with default `stabilize=False`, there is no passivity check.  Probe
(cell path, degree=11, nodes=33): `n_orders=15` -> E=1.303,
`n_orders=16` -> E=3.884, both returned silently; 17 raises.  The
docstrings advise raising `n_orders` at steep incidence, walking users
into the corner.  Fix: require conditioning margin (e.g.
`nodes >= 1.3*(2*n_orders+1)`) and warn unconditionally on
`|R+T-1|` for lossless cells even when `stabilize=False`.

### S1-4 [P2][seam/robustness] EME scalar `ref_2d_modes` default shift-invert sigma sits ON the flat-mode eigenvalue -- `elements/eme/eme_2d.py:365`

Default `sigma = max(eps)*k0**2` is the band top; a cell with an
`n_max` region spanning the full x-extent has a Bloch mode exactly
there, making ARPACK's `(A - sigma I)` singular -- the identical root
cause commit 9759796 just fixed in `_fd_eig_dist` and that the vector
sibling avoids with a band-centre sigma (eme_2d_vector.py:887).  Not
reproducible on this machine's MKL (5 stable runs); the risk is
cross-BLAS (the fixed flake diverged on CI OpenBLAS).  Fix: one line,
match the vector sibling's band-centre sigma.  Confidence: medium.

### S1-P3 bundle (hygiene / conventions / latent seams)

- S1-5 [P3][convention] **CONVENTIONS.md waveplate slow-axis sign row is
  stale and actively dangerous** -- `CONVENTIONS.md:157` says
  `exp(-i*retardance)` and declares itself source-of-truth ("fix the
  call site to match this table"), but `polarization.py:723` correctly
  ships `exp(+i*retardance)` (the v5.17.0 P2-15 realignment, verified
  against Berreman handedness).  Obeying the table would reintroduce
  the fixed bug.  Update the row + citation.
- S1-6 [P3][convention] `berreman.py:256` `_solve_core` docstring claims
  eps arrives "already-conjugated (INTERNAL)"; every caller passes raw
  PUBLIC eps (verified numerically) while `_offplane_oblique_solve`
  genuinely does conjugate -- opposite gauges in the two internal
  paths make the stale docstring an active refactoring trap.
- S1-7 [P3][seam] `PMMStack.set_source` bypasses
  `_resolve_incidence_checked` (`elements/pmm/stack.py:514`): back-side
  angle `2.5 rad` silently solves the supplementary front-side geometry
  (byte-identical to `pi-2.5`); the 1-D entry points raise.
- S1-8 [P3][organization] Forward-mode branch selector copy-pasted
  across 5 NumPy + 2 JAX PMM generator sites with a drifted
  propagating/evanescent tolerance (`1e-8` scalar-vertical vs `1e-7`
  slant/tensor/metric) -- `elements/pmm/_core.py:544,592,1225,1343,
  3217,3253,3304,3858,4238`.  The exact multi-copy pattern that
  produced the six-copy factor-i defect.
- S1-9 [P3][organization] RCWA efficiency-projection physics duplicated
  at ~9 sites (`oned.py:681,1027; twod.py:940,1055,1598,1840;
  stack.py:319,2091`); all agree today.  Extract one
  `_project_efficiency` helper.
- S1-10 [P3][organization] PMM-2D far-field projection `_proj` closure +
  kz/RT/einc block duplicated between `stack2d_pure.py:323-374` and
  `twod_staggered.py:1003-1060`.
- S1-11 [P3][perf] Berreman OOP-oblique generalized path bypasses the
  wl-independent eig + interface caches the native path uses
  (`berreman.py:428-472`); a fixed-angle wavelength sweep of a
  tilted-director stack recomputes every eig.
- S1-12 [P3][perf] Off-plane tile detection uses strict `>0.0`
  (`elements/pmm/twod_jones.py:97,120`): 1e-17 rotation noise routes
  in-plane cells to the ~8x-slower 4Nf generator and disables the
  even-parity fold.  Threshold relative to tensor scale.
- S1-13 [P3][seam] Berreman numpy vs JAX forward/backward split diverge
  in the non-2+2 fallback branch (`berreman.py:177` decay-ranked vs
  `_berreman_jax.py:72` index-ordered); unreachable for physical media
  tested, latent for degenerate bianisotropic inputs.
- S1-14 [P3][robustness] `rcwa_jones_1d` crashes with `IndexError` on
  scalar eps (`elements/rcwa/oned.py:1146`) and `RCWAStack.add_layer`
  crashes on a uniform (3,3) tensor (`stack.py:1420`) -- no clean
  public uniform-tensor entry to RCWA, blocking API-level reproduction
  of the Berreman cross-validation.
- S1-15 [P3][gap] BOR `basis="nodal"` still executes on large cells and
  returns the documented ~1e29 cascade blow-up silently
  (`elements/bor/bor_solve.py:86-90`); guard or warn.
- S1-16 [P3][seam] The three BOR mode classifiers are not in lockstep
  despite comments asserting it (`bor_solve.py:112` has reldiv leg, no
  index ceiling; `bor_stack.py:351` + `_jax_bor.py:189` the reverse).
- S1-17 [P3][gap] `eme_diffraction.diffraction_fd/eme` silently return
  non-convergent efficiencies for structured layers (documented
  negative result, unguarded) -- `elements/eme/eme_diffraction.py:146,169`.
- S1-18 [P3][perf] BOR nodal `build_layer` runs two byte-identical dense
  eigensolves (modes + reldiv harvest) -- `bor_solve.py:87-90`, ~2x.
- S1-19 [P3][convention] `rcwa_jones_2d` docstring claims OOP tensor
  components are ignored; they are fully supported since v5.14.1
  (`elements/rcwa/twod.py:1287`).  Also stale BOR README
  (`elements/bor/README.md:9,21`) and `_tensor_convolutions` docstring
  misstating the isotropic reduction (`elements/rcwa/_core.py:1705`).
- S1-20 [P3][gap] `truncation='circular'` exposed on scalar 2-D RCWA
  entries but not `rcwa_jones_2d`/`RCWAStack`
  (`elements/rcwa/twod.py:1432`, `stack.py:1526,1894`).

### Stage 1 verified sound (oracle-backed, spot list)

- RCWA OOP extraordinary dispersion exact at oblique: eig matches the
  Christoffel quartic to 1.3e-15; full end-to-end OOP Jones vs
  independent 4x4 transfer matrix to 8.5e-16 (lossless) / 1.3e-15
  (lossy).  Factor-i fix live in RCWA, PMM metric generator
  (vs RCWA 1.3e-5..2.3e-6 incl. magneto-optic), 2-D PMM OOP
  (vs Berreman ~1e-15 conical), and all three Berreman copies
  (numpy vs JAX generators to 1e-15).
- RCWA 1-D fast path vs 2N Jones path 1e-14; 2-D Li-1997 sequential
  rule converges 2nd-order onto rigorous 1-D; S-matrix stable for
  5-8 um evanescent/lossy layers; eig/interface cache keys complete;
  numpy/JAX forward-mode selectors provably pick the same set.
- PMM 1-D: slant sign conventions consistent (no mirrored-order bug);
  binary slanted == 2-segment segments bit-identical; internal-field
  reconstruction divergence-free (flux constant across depth);
  prepared-stack in-plane solve bit-identical to direct; conical
  patterned+OOP correctly raises.
- PMM 2-D: y-uniform limit matches 1-D engine 2e-6..8e-4; pure stack
  == staggered single layer exactly (0.0); A|B cascade vs 1-D
  multilayer ~1e-4 (TE+TM) corroborated by RCWAStack; mirror symmetry
  3.5e-15; layer_absorption closure 6e-6; even-parity fold ~1e-14;
  hybrid FMM-floor honestly documented, no silent routing to hybrid.
- Berreman: isotropic vs Fresnel/TMM ~1e-16 lossless+lossy; loss-sign
  convention consistent Berreman <-> scalar coatings TMM (~1e-15);
  z-anisotropic oblique vs admittance TMM 1e-16; gyrotropic Hermitian
  slab conserves 1e-7; thick-slab evanescent stable to 200 um;
  absorption closure 4e-16; theta->0 native/generalized handover
  continuous 1e-17.
- EMT (Rytov): order-2 tensor matches hand-derivation; rigorous
  grating converges monotonically onto it as period->0; validity
  warnings fire.
- glass.py: Sellmeier/polynomial catalogs match published values
  <1e-5 and are bit-identical to live refractiveindex.info
  coefficients; unit handling and out-of-band guards correct.
- BOR: grazing-cutoff fix holds (1.22e-11 closure at 319 modes);
  Fresnel+TIR oracle 5.3e-15; m=0..5 axis regularity clean; JAX twin
  9e-14.  EME: de-flake fix is in library code (verified); scalar JAX
  twin 8e-13, vector twin bit-identical; non-Hermitian strips routed
  off `eigh` correctly.

### Stage 1 feature gaps (ranked)

1. 2-D anisotropic cells in the no-floor staggered/pure PMM engine
   (today anisotropic 2-D exists only on the FMM-floored hybrid;
   pure path correctly raises).  Roadmap Phase C -- highest value.
2. Mixed Li-1997 crossed-tensor rule for `pmm_jones_2d` (lifts
   patterned tensor cells off the Laurent ~1e-3 floor).
3. Even-parity fold extension to `RCWAStack.solve` and tensor
   operators (common-metasurface speed win, roadmap LEV-3).
4. Circular truncation on the anisotropic/multilayer RCWA APIs (S1-20).
5. 2-D slant/curved PMM (roadmap Phases D/E, correctly deferred).

---

## Stage 2 -- Wave propagators and lens models (ASM/fft_infra/system, GBD, FGA, traced/Maslov/Levin, real-lens/dispatcher/DOE/HFPI)

Five auditors (ASM/fft infrastructure; GBD; FGA; traced/multibranch/
Maslov/Levin; real-lens+dispatcher+DOE -- the last re-run, reported
below the first four).  This stage found the audit's most serious
defects: three P1s, all probe-pinned.

### S2-1 [P1][physics] Multibranch rasterizer double-counts shared triangle-edge pixels: spurious energy up to +53% and 2x-amplitude hot pixels -- `elements/_lens_traced_multibranch.py:583`

The barycentric coverage test `inside = (a0>=0)&(a1>=0)&(a2>=0)` is
CLOSED for both triangles of every launch cell; contributions are then
accumulated coherently via `np.add.at` (line 619) with no half-open
tie-break or (pixel, sheet) dedup.  Pixels on a shared edge (notably
every cell diagonal) get ~identical contributions from both triangles:
2x amplitude, 4x intensity.  Probes: flat BK7 plate (identity map)
returns `power/Pin = 1.531` with 206/5233 covered pixels multi-counted
(single-branch pixels exact, 0.9999); weak f=20 mm singlet 1.51,
growing with output distance (1.59 at 2 mm, 1.69 at 5 mm); STABLE
across `ray_subsample` 1/2/4 (grids near-commensurate by construction,
so edge hits are systematic).  This is the regime the roadmap matrix
marks as the safe "reduces to plain sum" case -- the artifact hides in
the easy regime.  The sole energy test
(`tests/unit/test_v5_21_lens_accuracy_extensions.py:623`) passes only
because its two probe planes happen to have zero shared-edge pixels
(d=2.8 mm on the same config gives 1.098 and would fail its 8% gate).
Fix: half-open (top-left-rule) coverage or dedup (pixel, sheet) before
the coherent sum; then re-gate the test at a gentle-map plane with a
tight tolerance.  Confidence: high (probe-pinned).

### S2-2 [P1][physics] FGA v5.24.3 auto-sampling: 30-35% power deficit on the resolution of identity / near-axial free space -- `propagators/fga.py:521` (REGRESSION, new in v5.24.3)

The new `_resolve_sampling` floors `p_max` at `max(0.03, 1.5*content)`.
For a near-collimated beam the completeness requirement is
`p_max >~ 2/(k*w0)` (the beamlet momentum width), which exceeds the
floor: default identity call (flat prescription, distance 0) returns
power ratio 0.927 (w=12 um), 0.691 (w=18 um), 0.645 (w=30 um) while
field fidelity stays 0.9997.  Pre-5.24.3 defaults gave ~0.9.  The
module docstring still claims "the t=0 resolution of identity is exact
(energy ratio 1.0)".  The existing identity test hard-codes
`p_max=0.06, n_p=15` and so never exercises the default.  Radiometric
callers of the default path get silently ~30% low |E|^2.
Fix: floor `p_max` at `max(content-cover, C/(k*w0))` with C~2-3, or
default `normalize_output='power'`; fix the docstring.
Confidence: high (mechanism isolated: power vs p_max sweep 0.03 ->
0.579, 0.06 -> 0.991, 0.10 -> 0.9996).

### S2-3 [P1][physics] JAX complex64 ASM/MFT/Fresnel silently lose ~26 dB of phase accuracy vs the documented dtype contract -- `propagators/asm.py:191`, `propagators/mft.py:198`, `propagators/fresnel.py:145`

The numpy path computes the transfer-function phase in float64 with a
mod-2pi fold before casting to complex64 (asm.py:239-244), honoring the
documented "float32 noise floor, does not degrade with phase magnitude"
contract (probe: rel err 1.6e-7).  The JAX branches build
`arange(dtype=float64)` -- which JAX silently truncates to float32 when
x64 is off (the JAX default) -- and apply no mod-2pi fold, so `kz*z`
(~1e4-1e6 rad) is evaluated in float32: rel err 2.1e-3 at z=8 mm,
N=256 -- ~13,000x worse than numpy c64.  Anyone doing differentiable
optics in complex64 (the common JAX default) through
`angular_spectrum_propagate`, `_mft`, `fresnel_propagate`, or
`propagate_through_system_jax` gets distance-growing phase error that
corrupts gradients and contrast metrics.  complex128 defaults are
unaffected.  Fix: compute the JAX H/chirp builders' phase at float64
(enable x64 locally for the build) then cast, mirroring numpy; or warn.
Confidence: high (probe-pinned).

### S2-4 [P2][physics] `propagate_gbd_through_prescription(per_surface=False)` lands at the exit vertex and silently ignores `z_image` -- `propagators/gbd.py:3104`

The default `per_surface=False` path applies only the air-to-air lens
ABCD (front-vertex -> back-vertex; probed `M[0,1]=t/n`) and
reconstructs with NO leg to `z_image`; `per_surface=True` propagates to
`z_image` (default = BFL = focus).  Probe: collimated beam, f=20 mm
biconvex -- rms 102.7 um (False, field at exit vertex) vs 32.7 um
(True, converging); passing `z_image=bfl` with False is byte-identical
to not passing it (accepted-but-dropped kwarg).  Toggling one flag
moves the output plane by ~a BFL with no signal.
Fix: add the free-space leg (or raise if z_image passed on the
unsupported path); document the output plane.

### S2-5 [P2][physics] GBD `apply_aperture_to_beamlets(soft_edge=True)` uses launch-time `waist0`, not the propagated width -- `propagators/gbd.py:872`

`waist0` is never updated by propagation (freespace:756, thin-lens:811,
ABCD:2665 all pass it through); the true 1/e radius comes from Im(Q)
(exactly what `_reconstruct_windowed:1166` derives).  Probe: after
10 z_R the true width is 80.4 um vs waist0 8.0 um; a rim beamlet 30 um
inside the stop gets vignetting fraction 1.0000 (stale) vs correct
0.7716.  Since `propagate_gbd_thin_lens` applies the stop AFTER the
source->lens propagation, the advertised soft-edge accuracy gain
silently collapses to a hard cut whenever `z_to_lens > 0`.
Fix: derive per-beamlet width from Im(Q) inside the aperture routine.

### S2-6 [P2][physics] FGA flat-prescription NA cap truncates wide-angle free-space beams -- `propagators/fga.py:515-521`

For the canonical free-space `flat` prescription, `_default_p_max`
returns its 0.05 floor (flat plate: NA -> 0), so auto `p_max` is capped
at 0.075 regardless of field content.  Probe: content 0.094 (~0.22 rad)
-> fid 0.9957; content 0.152 (~0.35 rad) -> fid 0.9619 (vs 1.0000 with
adequate p_max).  The NA cap is physically meaningless for free space
yet clamps the swarm; the v5.24.3 "diverging beam reconstructs to ~1"
claim holds only to ~0.1 rad.  Fix: skip the NA cap when the
prescription carries no real power.

### S2-7 [P2][seam] FGA separable vs direct analysis kernels differ at integer R/dx -- which is the DEFAULT (nsig*w0_factor = 15) -- `propagators/fga.py:169-182`

The two kernels compute different window-box bounds (direct
`int((cxq+-R-x0)/dx)`; separable `jx +- int(R/dx) +- 1`), so at exactly
integer R/dx the boundary pixel falls inside one box and outside the
other: worst coefficient diff 7.8e-4, end-to-end field diff 1.3e-4 on
the default configuration (w0_factor 4.999 -> 2.5e-15, 5.000 -> 7.8e-4,
5.001 -> 2.0e-15).  The docstrings/CHANGELOG assert "numerically
identical (ULP-level)" -- false exactly on the default.  The historic
gate fix addressed the gate expression but not the box bounds.
Fidelity vs oracle unaffected (0.99999973 both).  Fix: make the
separable box identical to the direct box; or soften the claim.

### S2-8 [P2][physics] Multibranch axial-focus catastrophe unguarded: 1e6x energy blow-up at the BFL plane on the default `caustic_band='ludwig'` -- `elements/_lens_traced_multibranch.py:530,650`

At a rotationally-symmetric focus a RING of branches coalesces
(probe: max_n_branch=208, power/Pin=1.04e6 at d=BFL); the Ludwig swap
regularizes only the closest PAIR, leaving ~206 branches with divergent
ART amplitude `1/sqrt(ratio)`; no warning.  The docstring claims the
Ludwig default "still REGULARIZES (finite where the plain sum
diverges)" -- false for the axial catastrophe.  Fix: detect >=3-branch
coalescence, clamp/skip with a warning, or route those pixels to the
cusp/axial-accurate Maslov-levin evaluator.

### S2-9 [P2][gap] GBD strongly-diverging relays lose most energy with no runtime diagnostic -- `propagators/gbd.py:2745`

The documented low-fidelity regime (roadmap section 4: 0.16/0.71
roundtrip energy) produces silently low-energy fields; no
energy/fidelity metadata or warning exists on the prescription paths
(probe: output/input energy 0.05-0.02, warnings list empty).  Fix:
compare summed beamlet power vs reconstructed grid energy after
reconstruction and warn below a threshold.

### S2-P3 bundle

- S2-10 [P3][organization] ASM transfer-function construction duplicated
  ~6x across `asm.py:116,184,224`, `mft.py:190,220`, `asm.py:842`
  (tilted) -- the mod-2pi c64 mitigation landed in ONE copy (that is
  finding S2-3); the band-limit `<` vs `<=` needed two historical fixes
  for the same reason.  All copies currently agree (H diff 0.0 / 5e-14).
  Consolidate into one parametrized builder.
- S2-11 [P3][convention] `z=0` ASM is not identity for dx < lambda/2
  (evanescent bins zeroed even at z=0; probe rel err ~1.0 vs doc claim
  "returns the input unchanged") -- `fft_infra.py:1887`, `asm.py:122`;
  also `propagate(z=None)` copies input while `z=0` filters: two
  no-propagation spellings that disagree.
- S2-12 [P3][seam] `fft_backend_for` reports 'pyfftw' for real-dtype
  arrays that `_fft2` actually routes to scipy/numpy
  (`backend/fft.py:193` missing the `iscomplexobj` gate of
  `fft_infra.py:1716`).
- S2-13 [P3][seam] JAX system kernel drops `dy` on propagate steps
  (`propagators/system.py:1173,1406`): anamorphic grids propagate y at
  the wrong pitch on the JAX path only (numpy threads `current_dy`).
- S2-14 [P3][organization] GBD tensor free-space Moebius+amplitude block
  triplicated (`gbd.py:741,2915,2994`); FGA scalar/vector HK-prefactor
  and coeff-prune pre-pass duplicated (`fga.py:706/1012, 643/945`);
  Maslov Newton-saddle solver triplicated + Tukey window re-defined
  (`lenses_maslov.py:741,2645,3205`).  All copies currently agree.
- S2-15 [P3][physics] FGA auto `n_p` clamp at 61 silently re-enters the
  under-sampled dp regime for p_max > ~0.24 (probe: fid 0.917 clamped
  vs 1.0000 unclamped) -- `fga.py:531`; warn when post-clamp dp exceeds
  the target.
- S2-16 [P3][physics] FGA Herman-Kluk slope->direction-cosine conversion
  uses a scalar Jacobian where the true d(p)/d(u) is a full 2x2
  (`fga.py:708,1016`): O(u^2) prefactor error growing with NA/skew;
  moderate-NA validations unaffected (0.997-0.999).
- S2-17 [P3][physics] GBD tensor amplitude branch continuity assumes
  Im(lambda_i) < 0 without a guard (`gbd.py:744,2916`): extreme
  aberration could flip a beamlet's sign silently (not observed).
- S2-18 [P3][robustness] Levin integrator returns over-tolerance pixels
  with no warning when the depth-6 per-pixel fallback cannot meet the
  bound the deep pass (depth 12) already failed
  (`lenses_maslov.py:3164`); only a progress string reports it.
- S2-19 [P3][convention] `ray_subsample` docstring says default 1;
  signature default is 8 (`elements/_lens_traced.py:1295,1425`).  GBD
  `decompose_field_adaptive` summary contradicts its own (correct)
  Notes: coarse cells are kept + residual-corrected, not "dropped"
  (`gbd.py:585`).  `axial_opl` comment claims a BFL leg the code does
  not implement (`gbd.py:3137`).
- S2-20 [P3][gap] EME-style silent scope limits, propagator edition:
  `reconstruct_field_from_beamlets` public default `window=None` is the
  dense O(N^2 * beamlets) path while every internal caller passes 5.0
  (perf footgun); `direction_sampling=False` default silently loses
  ~40% energy on an already-tilted input (documented at the decompose
  layer, not surfaced at the propagate_gbd_* entry points).

### Stage 2 addendum -- real-lens family + universal dispatcher (auditor 5 re-run)

- S2-21 [P1][physics] **`_caustic_zone` uses `shape[-1]` (Nx) as a ROW
  index** -- `propagators/fga.py:1138-1142`.  `row = E_in[cx, :]` with
  `cx = Nx//2`: `apply_real_lens_universal`/`_auto` HARD-CRASH
  (IndexError, probe-confirmed) on any wide grid (Ny < Nx) and silently
  read the wrong meridional row on tall grids (wrong near-caustic
  routing).  Reached unconditionally in `apply_real_lens_auto`
  (fga.py:1234) and in universal's single-valued high-NA branch
  (fga.py:1441).  Fix: `cy = shape[0]//2`.
- S2-22 [P2][seam] FGA default `normalize_output='none'` returns
  ~4-8x power (scales WITH `w0_factor`); the dispatchers route to fga
  with no power-matching, so a multi-plane `method='auto'` workflow has
  a silent ~5x intensity discontinuity between fga-routed and
  ps/traced/gbd-routed planes (`fga.py:749,1470,1246`).  Probe: input
  5047 -> ps 5046, gbd 5035, fga-default 26,900 (w0_factor=5).
  Dispatcher should pass `normalize_output='power'`.
- S2-23 [P2][seam] Mirror/fold guard present in
  phase_screen/traced/maslov but ABSENT from gbd and fga
  (`elements/_lens_real.py:770-796` vs lenses_gbd/fga): a
  `glass_after='MIRror'` prescription raises via `apply_real_lens` but
  silently runs via `apply_real_lens_gbd/fga` -- and FGA's image-side
  leg unconditionally adds a FORWARD `z_image` leg (fga.py:700), so the
  reflected beam's output plane is on the wrong side.  Route through
  the shared `_check_no_silent_fold_drop`.
- S2-24 [P2][gap] `_caustic_zone` is a single x-meridional center-row
  model and ignores `dy` (`fga.py:1142`): astigmatic/anamorphic
  caustics (probe: y-only line focus) are never routed to the caustic
  specialist -- universal falls back to phase_screen exactly where FGA
  is the accuracy member.  Trace meridional + sagittal fans.
- S2-25 [P2][convention] `_carrier_residual_rms` divides BOTH gradient
  components by `dx` (`elements/_lens_traced.py:803,806`): the F1
  diverging-vs-collimated split shifts on anamorphic grids (the sibling
  `_tilt_dispersion` IS dy-aware).  Thread `dy`.
- S2-26 [P2][physics] `_tilt_dispersion` conflates Nyquist aliasing
  with multi-valuedness (`fga.py:1263-1321`): an under-sampled SINGLE
  converging beam scores 0.081 > 0.06 and routes to fga (docstring
  claims single beams score <0.006); scores drift 0.034 -> 0.011 across
  N=128 -> 512 (not scale-invariant near the cut), and the multivalued
  gate fires BEFORE the F1 diverging split, so an aliased steep
  diverging beam is grabbed by fga contra the F1 resolution.  Mostly a
  speed/robustness issue (fga is "never wrong"), plus a doc overclaim.
- S2-27 [P3] bundle: full refraction pipeline duplicated between
  row-band and whole-grid `apply_real_lens` paths (~120 lines,
  `_lens_real.py:1043-1169` vs `1326-1463`); NaN input never rejected
  and `_tilt_dispersion` reports a NaN field as maximally single-valued
  (routes to traced, inverting the "uncertain -> fga" design;
  `_validation.py:91`, `fga.py:1301`); paraxial `-k r^2/2f` lens phase
  re-implemented in >=4 places (`_lens_thin.py:164`, `doe.py:245,324,
  399`); `_caustic_zone` uses a global `np.unwrap` where its siblings
  use the wrap-safe conjugate-product (`fga.py:1147`).
- Verified sound (auditor 5): DOE sign conventions correct
  (diffractive lens/kinoform/FZP all focus at +f, consistent with thin
  lens and thick singlet); phase_screen/gbd/fga output-grid conventions
  mutually consistent (fidelity 0.999/0.992/0.994, lossless members
  conserve power); per-surface dispersion threading `n(lambda)` correct;
  multivalued detector separates genuine multi-emitter by >10x when
  Nyquist-sampled; F1 well-sampled diverging -> phase_screen dispatch
  exact; 16x16 and square grids route fine.

### Stage 2 verified sound (oracle-backed, spot list)

- ASM: forward physics vs analytic Gaussian (3e-6-1.6e-5, estimator
  limited); reciprocity 4.9e-16; MFT==ASM same-grid 5e-14; backend
  parity pyfftw/scipy/numpy 4e-16; no input mutation (byte-checked);
  batch==scalar bit-identical; chunked-H==one-shot bit-identical;
  tilted-ASM centroid exact; c64 numpy mod-2pi fold works (1.7e-7);
  all caches keyed completely and LRU/byte-bounded;
  `set_default_complex_dtype` flushes the H cache; Bluestein and RS
  caches copy-on-hit (no double-buffer aliasing); SAS warns past its
  paper z-limit.
- GBD: single-beamlet Q+amplitude vs analytic Gaussian 1e-16 through
  focus (no KMAH flip); dense/windowed/FFT reconstruction kernels agree
  1.1e-15/1.7e-15 with identical sign conventions; free-space GBD vs
  ASM converges as documented; Husimi launch reproduces tilted-Gaussian
  centroid/energy (0.030 rel-L2, 0.96 energy); OPL vertex-plane fix
  present and non-regressed (powered-exit vs planar-exit singlets show
  no curvature doubling); near-normal Fresnel-Jones frame is
  singularity-guarded; thin-lens B=0 amplitude/piston consistent with
  the F-1 pin.
- FGA: momentum + Nq chunking identities hold to <=1.7e-14 on
  non-dividing chunk sizes WITH pruning on, scalar and vector;
  coeff-prune global pre-pass correct for late-chunk peaks (6e-15);
  scatter kernels ULP-identical incl. recurrence drift 1.9e-13 over 160
  steps; pruning conservative on Airy rings/speckle/weak-ring fields
  (fidelity 1.0000000); vector Ez transversality exact vs analytic;
  <=0.1 rad diverging-beam fix confirmed (fid 0.99848); caustic
  peak-error 7.8% (top of documented 0.8-7.9%); mem-budget raise fires
  only when documented.
- Traced/Maslov/Levin: traced exit-vertex OPL/wavefront curvature
  correct (R_fit 19.855 mm vs f=20 mm analytic; no 2x bug); traced vs
  maslov power agreement 0.07% with aligned output conventions; Levin
  engine canonical fold 1.6e-9-3.3e-9 with honored bounds INCLUDING a
  stationary point inside the cell; KMAH `exp(-i pi/2)` per fold
  correct for the library's exp convention; poly_order='auto' rejects
  overfit via held-out split; MSL-1 det_safe guard present at all three
  saddle sites; uniform Airy/Pearcey branch discipline consistent.

### Stage 2 feature gaps (ranked)

1. No `n_medium` on the ASM/Fresnel/MFT/SAS family (only GBD/HFPI take
   it); propagation in glass requires the undocumented
   `wavelength=lambda_vac/n` workaround.  One kwarg or a docstring.
2. No Q>1 / undersampling advisor on the DIRECT
   `angular_spectrum_propagate` entry (auto-routing to SAS exists only
   via `propagate(method='auto')`); direct callers in the
   walked-off-grid regime silently lose energy off-grid.
3. Multibranch axial-caustic routing to Maslov-levin (S2-8's proper
   fix; roadmap section 5 interim).
4. `_select_poly_order_auto` scores OPD only, not s1x/s1y amplitude
   residual (15% peak diff seen order 3 vs 4 on a smooth optic).
5. HFPI quantitative mode (per-path 1/r + output-binning Jacobian);
   currently an honestly-documented phase-structure diagnostic.
6. GBD CSP path not deeply verified this pass (non-default, documented
   scope); worth a dedicated high-NA probe if it becomes load-bearing.
7. 1-D `fft` helper has no pyfftw path (minor).

---

## Stage 3 -- Ray tracing, sources, analysis, IO

Four auditors this stage (raytrace core + Seidel; JAX trace twin;
sources/coherence; analysis metrics).  IO (zemax/storage) runs in the
Stage 4 batch.  Two P1s, both of a systemic character: a Seidel sign
inconsistency masked by a self-referential test, and a second instance
of the silent-JAX-float32 trap found in Stage 2.

### S3-1 [P1][physics] Seidel S4 (Petzval) sign inconsistent with S1-S3, corrupting S5 (distortion) and `seidel_wfe` field curvature -- `raytrace/seidel.py:1036,1080,1039,1081,1123`

An independent Welford (Ch. 8) oracle reproduces the code's S1,S2,S3
EXACTLY as `-S_Welford` (ratio 1.0000 incl. the Cassegrain
mirror-parity case) -- but S4 comes out `+S_IV_Welford` (ratio
-1.0000): the opposite convention, mixed into
`S5 = -(A_c/A_m)(S3 + H^2 S4)`.  Convention-independent anchors:
(a) distortion must vanish for a stop at a thin lens -- independent
oracle |S5/S1| = 3e-6, code 1.9e-4 (~60x); (b) for a positive singlet
S3 and S4*H^2 must have the SAME sign -- code gives +1.46e-7 vs
-1.14e-7, so the `(1/4)(S3+S4 H^2) rho^2` field-curvature term in
`seidel_wfe` nearly cancels (~8x too small); (c) Cassegrain S5 total:
code -8.23e-9 vs independent +8.16e-9.  The pinned "Welford hand calc"
test (`tests/unit/test_audit_misc.py:2760`) admits in its own comment
that the expected values come from the library's formula -- a
non-independent oracle, which is how this survived ~30 audits.
S1/S2/S3 (the most-used sums) are correct.
Fix: negate S4 at seidel.py:1036/:1080 and flip the S5 prefactor sign
at :1039/:1081/:1123 (verified to reproduce the independent oracle for
all five sums across singlet/meniscus/Cassegrain); then add an
INDEPENDENT S5 gate (stop-at-thin-lens -> 0).  Confidence: high.

### S3-2 [P1][physics/seam] JAX through-focus silently computes in float32: Strehl > 1, 14.5% peak error -- `analysis/through_focus.py:1056`

`_build_through_focus_scan_jax_kernel` requests float64 but JAX
truncates to float32 when `jax_enable_x64` is off (JAX's default;
lumenairy never enables it).  Probe: NumPy best-focus Strehl 1.0019 vs
JAX 1.1429 (peak_I rel diff 0.145); with x64 on, parity is 7.9e-16.
Reachable via `through_focus_scan(backend='jax')` and
`monte_carlo_tolerancing(_jax)`: silently mis-ranks designs and
corrupts MC Strehl statistics.  The sibling `asymptotic_jax_twin.py`
RAISES in exactly this situation, so the guarding precedent exists.
This is the same trap class as S2-3 (ASM/MFT/Fresnel JAX c64): a
library-wide `_require_jax_x64()` (or float64-computed kernels) would
close the whole class.  Confidence: high (both directions measured).

### S3-3 [P2][physics] Point-source / tilted-plane-wave chirp aliases silently -- `sources/core.py:789-812,717-722`

The only guard is `|z0| < dx`.  Probe (N=512, dx=1 um, z0=200 um):
edge-NA 0.79 vs Nyquist limit 0.32, phase step 3.13 rad/px, no
warning; the focused peak-power fraction halves (0.123 vs 0.245
well-sampled).  `create_tilted_plane_wave` guards evanescence but not
`sin(angle) > lambda/(2 dx)`.  Fix: edge-NA Nyquist warning mirroring
the existing guard pattern.

### S3-4 [P2][physics] `create_fiber_mode`'s `na` argument is silently inert -- `sources/core.py:1023-1027,1078-1090`

Docstring: "Gaussian with NA-defined divergence"; implementation uses
MFD only -- fields for na=0.05 vs 0.15 are BIT-IDENTICAL (probe).  `na`
only triggers a >0.2 warning.  A user setting a datasheet NA believes
they control divergence and do not.  Fix: doc-correct, and warn when
`na` disagrees with the MFD-implied divergence.

### S3-5 [P2][physics] Second-moment widths have no ISO-11146 background guard -- `analysis/beam_stats.py:83,134,205`

`beam_d4sigma`/`M2` integrate over the whole grid: a 1e-4-of-peak
pedestal inflates D4-sigma 3.1x (1e-3 -> 8.1x); `single_plane_metrics`
and `find_best_focus(metric='spot'|'rms')` inherit it, so noisy fields
get a wrong best-focus plane.  Fix: optional background subtraction +
iterative integration aperture (~3 D4sigma) per ISO 11146; document
the sensitivity.

### S3-6 [P2][seam] `make_lg_aberration_merit_jax` advertises 6 differentiable inputs; 4 crash -- `optimize/jax_merits.py:184`, `raytrace/jax_trace.py:888`

`wavelength` (glass resolved via `float(get_glass_index(...))` --
TracerArrayConversionError), `pupil_box_half`, `object_distance`,
`source_box_half` all raise deep in the stack when the analytic
jacobian is requested via `design_optimize(jac='auto')`; only
`w_s`/`w_p` work (both FD-verified to ~1e-6).  Loud, not silent --
but the documented chromatic-optimization use case is broken.
Fix: JAX-traceable dispersion (or n_pre/n_post leaves); trace-safe
guards in `fit_canonical_polynomials_jax:853,862`; trim the docstring.

### S3-7 [P2][memory] Source factories materialize meshgrid X/Y everywhere broadcasting would do -- `sources/core.py:297,459,612,725,787,939,1001,1434,2090,2328`

Two dense N x N float64 grids + squared temporaries per factory
(~1 GB transient at N=8192) vs exact-identical broadcasting on 1-D
axes.  Also 10 verbatim copies of the same grid block.  Fix: shared
`_centered_axes` helper + broadcasting (bit-identical result).

### S3-8 [P2][seam] JAX through-focus zero-plane divergence -- `analysis/through_focus.py:1255-1262`

All-zero plane: NumPy returns 0.0 cleanly; JAX leaves NaN metrics,
warns "All-NaN slice", and can raise under warnings-as-errors.
Mirror the NumPy guard.

### S3-P3 bundle

- S3-9 [P3][physics] Differential trace `_adrt_step` uses `abs()` for
  the transfer-leg OPL (`raytrace/differential.py:463,381`),
  contradicting the main trace's signed RT-1 convention: base-ray OPL
  feeding GBD piston diverges from `ray_transfer_jacobian` whenever
  tau2 < 0 (overlapping sags / post-mirror folds).  Drop the abs.
- S3-10 [P3][organization] Conic-intersection + Snell implemented
  independently in 4 places (intersection.py, differential.py,
  seidel.py paraxial x2, jax_trace.py); Seidel's removed
  `_paraxial_trace` (RT-3) is prior evidence this duplication ships
  bugs.
- S3-11 [P3][convention] `make_rings` samples equal-radius rings with
  constant rays/ring: areal density ~1/r, so unweighted `spot_rms` is
  center-biased small (best-focus LOCATION robust); sampling
  undocumented (`raytrace/trace.py:934`).
- S3-12 [P3][physics] `_transfer_jax` advances position/OPL of DEAD
  rays (`raytrace/jax_trace.py:574`); numpy freezes them.  All in-tree
  consumers mask by `alive` (verified), but unmasked
  `jax_state_to_raybundle` readers see backend-dependent garbage.
- S3-13 [P3][gap] JAX Newton fixed 8 iters vs numpy 10-with-exit, and
  the float32 residual tolerance is ~5.4e-4 m (`jax_trace.py:70,84`):
  marginal aspheres could alive-mask-diverge in float32 (not
  reproduced; latent).
- S3-14 [P3][convention] Building a JAX merit flips global
  `jax_enable_x64` as a constructor side effect
  (`optimize/jax_merits.py:268`, `:426`); prefer require-and-raise.
- S3-15 [P3][perf] Eager `trace_jax` cache key embeds all numeric
  prescription values -> re-JIT per perturbed prescription in sweep/FD
  loops (32-entry LRU thrash) (`jax_trace.py:1208`); signpost
  `trace_jax_with_params`.
- S3-16 [P3][convention] Sources: `seed` kwarg on the newest
  randomness APIs violates the CONVENTIONS.md `rng` rule
  (`sources/core.py:1966,2128,2238,2956,3060`); normalization contract
  inconsistent across the zoo (peak vs power vs raw, some without a
  kwarg: `:223,384,542,943,1005,1437`); `sigma` vs `w0` naming
  divergence (`:218`); dead `N` parameter in
  `_schell_phase_realizations` (`:1890`).
- S3-17 [P3][gap] `Source` dataclass has no polarization/Jones channel
  (`sources/core.py:2400`): vectorial pipelines must bypass it.
- S3-18 [P3][perf] `make_shack_hartmann_wfs` rebuilds the Zernike
  reconstructor (FD influence matrix + pinv) EVERY call
  (`analysis/ao.py:1223`); cache like `_calib_cache`.
- S3-19 [P3][convention] `plot_psf` docstring says "peak-normalized"
  but `compute_psf` defaults power-normalized (`analysis/plotting.py:574`);
  `strehl_marechal` returns 0-d ndarray for scalar input
  (`analysis/strehl.py:148`); single-plane metrics duplicated between
  numpy `single_plane_metrics` and the inline JAX block
  (`through_focus.py:1219`, already produced the zero-field
  divergence).

### Stage 3 verified sound (oracle-backed, spot list)

- Raytrace core: sphere intersection exact vs analytic (convex,
  concave, reversed); vector Snell exact + TIR kills with flag;
  sag/normal FD-consistency ~1e-12 (conic+asphere, biconic,
  per-axis-asphere; out-of-domain -> NaN); system_abcd EFL vs thick
  lensmaker 8e-19; Petzval stop-invariance; OPL telescoping incl. RT-1
  signed convention; coord-break parity between intersection and
  differential paths; Welford mirror parity consistent; S1/S2/S3 exact
  vs independent Welford.
- JAX twin: forward parity machine-precision (positions 7e-18, OPL
  1.2e-17); RT-6 exact-transfer invariant across 100x gap sweep;
  gradients d(spot^2)/dR, d(OPD)/dt, d/d(conic), d/dA4, d/d(DOE
  period) all FD-verified 1e-7..1e-10; NaN-safe double-where guards
  verified at TIR/vignette/tangency boundaries; DOE grating kick
  bit-identical to numpy; NO un-mirrored numpy fixes (each recent fix
  located in the twin); unsupported surfaces raise cleanly on both
  entry points.
- Sources: waist and divergence conventions exact; HG/LG orthonormal
  <1e-4 with correct OAM handedness; Gaussian-Schell ensemble
  coherence width 0.984 of requested; MCF Hermitian/PSD with SRC-1
  fix confirmed; point-source sign convention matches exp(+ikz);
  source grid convention identical to propagators (no half-pixel
  seam); 3-D Schell ensembles rejected with a named error.
- Analysis: Airy PSF/sampling exact; Rayleigh/Sparrow/EE 83.78% vs
  theory; MTF vs analytic circular-pupil autocorrelation <0.0015;
  STREHL WITH APODIZED PUPILS CORRECT (Gaussian-apodized peak-ratio ==
  phase-integral +-0.00000 -- the classic silent bug is absent);
  Zernike OSA orthonormal (RMS 1.000, Gram off-diag 9e-4);
  slope_to_modal recovers analytic slopes with zero cross-talk;
  ao_closed_loop leak/gain law verified; M2 immune to phase curvature
  (Wigner cross-term); numpy<->jax(x64) parity 8e-16.

### Stage 3 feature gaps (ranked)

1. Library-wide JAX x64 guard (fixes the S2-3/S3-2 class in one move).
2. Traceable-wavelength dispersion for chromatic JAX optimization
   (unlocks the advertised use case, S3-6).
3. Sources: z/ROC/Gouy placement for Gaussian/HG/LG (waist-only
   today); true LP fiber-mode solver (current Gaussian-MFD approx;
   docstring references a `Source.from_array` that does not exist);
   Bessel-Gauss (apodized) variant; polarization channel on `Source`;
   partial-coherence realization iterator (docstring-promised).
4. Raytrace: reflect-on-TIR option (ghost/prism work); Zemax BICONIC
   exactness (documented separable approximation); mirrors/biconic/
   coord-breaks in the JAX twin (clean raises today).
5. Analysis: ISO-11146 background handling (S3-5); on-axis vs peak
   Strehl convention note; Fringe Zernike ordering (documented
   NotImplementedError).

---

## Stage 4 -- Optimization, GUI seam, infrastructure, duplication, test health

Five auditors (IO/zemax/storage; optimize; GUI seam; API/infra/packaging;
whole-library duplication + dead code).  Three more P1s -- two at the
GUI seam, one in IO -- and the structural explanation for why this
audit's JAX findings survived so long: JAX is exercised in no CI leg.

### S4-1 [P1][physics] `.txt` Zemax loader silently drops COORDBRK axial gaps -- `io/prescriptions_zemax.py:1116`

`load_zemax_prescription_data_txt` builds thicknesses with NO
coordinate-break gap folding; the `.zmx` twin got the ZX-1 folding fix
(lines 571-584) and the `.txt` copy never did.  Probe (folded
2-element design, CB `DISZ 5 mm`): `.zmx` -> [0.004, 0.009, 0.004];
`.txt` -> [0.004, 0.006, 0.004] -- the 5 mm gap VANISHED, shifting
every downstream surface and the image plane, silently.  Textbook
fix-one-of-two-copies: the loaders are near-duplicate ~430-line twins
(S4-8).  Fix: port the folding loop or (better) share one post-parse
core.  Confidence: high (probe-pinned).

### S4-2 [P1][seam] GUI wave optimizer: bounds built in mm against a metre-unit x0 -- `ui/optimizer_dock.py:923-949`

`_start_wave_optimize` builds the prescription in METRES
(model.py:2776 converts) but bounds from `get_variable_values()` in
MILLIMETRES (model.py:2428-2430).  Probe: singlet R=+-50 mm, t=5 mm
gives x0=[0.05,-0.05,0.005] against bounds [(25,100),(-100,-25),
(2.5,10)] -- x0 outside every box; scipy clips to a 25-METRE radius
start.  The flagship "Wave Optimize" button is broken for ALL length
variables (only dimensionless conic escapes).  Additionally,
`thickness`/`semi_diameter` free variables KeyError outright
(S4-6).  Fix: build bounds from the metre-unit template values.
Confidence: high (probe-pinned against the real DesignParameterization).

### S4-3 [P1][physics] GUI default `lens_model='asm'` drops aspheric coefficients and biconic curvature -- `ui/waveoptics_dock.py:865`

The default per-surface phase screen calls the rotationally-symmetric
`surface_sag_general(h_sq, radius, conic)`: trace surfaces carry
`radius_y/conic_y` (ignored) and no `aspheric_coeffs` at all
(model.py:2138 omits them from the Surface ctor).  A cylindrical
surface focuses to a POINT instead of a line; aspheres collapse to
their base conic -- silently, on the documented default propagation
path -- while `layout_2d.py:429` DRAWS the same surface with the full
biconic sag (picture right, physics wrong).  Fix: branch to
`surface_sag_biconic` + thread aspheric_coeffs, or warn and steer to
the real_lens model when those fields are non-trivial.

### S4-4 [P2][test-coverage] JAX is exercised in NO CI leg -- `.github/workflows/unit-tests.yml:73`

No workflow installs jax (unit-tests.yml:64-65 deliberately removed
it); 340 test files guard jax paths behind skipif/importorskip and so
skip everywhere, including all numpy<->jax parity gates.  Every
JAX-side finding in this audit (S2-3, S3-2, and the historical
RandomState bugs) is invisible to CI.  Fix: one non-matrix CI job with
`[jax]` running the unit gate.

### S4-5 [P2][physics] Optimizer failure-direction bugs -- `optimize/merit_terms.py:145,203`, `optimize/driver.py:840,1127`

(a) A degenerate wave leg (fully-vignetted exit field) leaves
`ctx.strehl_best = nan` and `StrehlMerit` evaluates
`max(0.0, min_strehl - nan) = 0.0` -- a PERFECT score for a failed
design (wrappers use the correct 0.0-sentinel penalty; the main leg
disagrees in SIGN of the failure handling).  (b) `SpotSizeMerit` reads
the `rms_radius_best` default `np.inf` when the scan is skipped
(|BFL| > 10 m early-return) and injects `inf` into the scipy merit
(L-BFGS-B line-search stall).  (c) `method='lm'` silently clamps
negative merit contributions (`sqrt(max(m,0))`, driver.py:1127) --
CallableMerit reward-style terms are zeroed under 'lm' only.
Fix: coerce non-finite strehl_best to 0.0; cap SpotSizeMerit finite;
warn on negative terms under 'lm'.

### S4-6 [P2][seam] GUI wave optimizer cannot carry thickness/semi_diameter variables -- `ui/optimizer_dock.py:933-937`

Maps `('surfaces', i, 'thickness')` but the legacy prescription
surface dict has no such keys -> KeyError -> "Wave optimizer setup
failed".  Route thickness to the `('thicknesses', i)` slot.

### S4-7 [P2][robustness] GUI geometric optimizer mutates the live model from the worker thread -- `ui/optimizer_dock.py:122`, `ui/model.py:2474,2598`

Every scipy iteration calls `set_variable_values` ->
`recompute_element_frames()` on the shared model while main-thread
views read it, and `system_changed` is emitted from the worker.  The
waveoptics dock deep-snapshots (correct discipline); the geometric
path predates it.  Also: `run_optimization` passes no `bounds=` to
scipy (model.py:2596), so bounded-method choices run unconstrained
(negative thicknesses reachable).

### S4-8 [P2][organization] `.zmx` / `.txt` loaders are diverged near-duplicate twins -- `io/prescriptions_zemax.py:58,747`

~430 lines each duplicating CB filtering, surface-range autodetect
(byte-identical), medium resolution, element build, aperture
resolution; ZX-1 and ZX-3 fixes landed only in `.zmx` (-> S4-1 and the
stop-loss below).  Extract one shared `_finalize_surfaces` core.

### S4-9 [P2][robustness] IO silent-fallback bundle

- `.txt` loader omits `stop_index`/`is_stop`/`coord_breaks`: stop
  relocates to surface 0 on re-export (probe-confirmed); fold geometry
  unavailable downstream (`prescriptions_zemax.py:1135`).
- Unknown `UNIT` token silently treated as mm (`:160`, txt `:869`) --
  potential order-of-magnitude mis-scale with no warning.
- No-STOP + no-DIAM prescription yields `aperture_diameter = 0.0`
  silently (`:592`): downstream fully-clipped field, no diagnostic.
- `append_plane` metadata `None` crashes HDF5 backend but stores fine
  on zarr (`io/storage.py:685,463` missing the None-skip guard that
  `save_field_h5` has) -- backend-divergent behavior of the unified
  API.

### S4-10 [P2][seam] EME scalar vs vector mode-scan density drift -- `elements/eme/eme_2d.py:153` (n_scan=600) vs `eme_2d_vector.py:530` (n_scan=400)

The vector Bloch mode-finder samples the qz^2 axis 33% coarser than
the scalar path; closely-spaced modes the scalar path resolves can be
missed by the vector path pre-refinement.  Same-knob different-default
drift (see also PMM2DStack n_orders=11 vs PMM2DStackPure n_orders=7 --
sibling stacks truncating differently, `pmm/stack2d.py:87` vs
`stack2d_pure.py:121`).

### S4-11 [P2][organization] New duplication clusters (whole-library scan; all currently AGREE)

- Zernike polynomial: full independent copy in
  `elements/elements.py:395` vs `analysis/zernike.py:75` (agree to
  4.4e-16 TODAY, but the Noll-sign audit fix exists only in the
  analysis copy's branch structure) -- delete the elements copy.
- BOR radial coupled-mode operator assembly byte-identical in
  `elements/bor/zcascade.py:146` and
  `elements/bor/coupled_radial_eigensolver.py:258` -- both live via
  bor_solve imports.
- HFPI scalar/vector: six parallel function pairs share byte-identical
  kernels (`propagators/hfpi.py:170-537` vs
  `vectorial_hfpi.py:75-393`), synced by hand-comments.
- DOE grating kick duplicated `raytrace/trace.py:178` vs
  `raytrace/world_trace.py:168`.
- Chebyshev Vandermonde private copy in `elements/_lens_traced.py:658`
  missed by the v5.2 consolidation; Sellmeier evaluator duplicated
  `glass.py:633` vs `elements/coatings.py:611`; Strehl-Marechal
  re-inlined in `analysis/image_plane_wfe.py:154`; D4sigma->waist
  estimate duplicated `propagators/hf.py:476` vs `subaperture.py:429`;
  1-inch default aperture constant hand-copied in >=6 files.

### S4-12 [P2][gap] Import time 3.5-6 s cold; scipy eagerly imported -- `backend/scipy.py:21`, `__init__.py:29-1003`

`-X importtime`: scipy ~3.6 s + numpy ~1.4 s + charset_normalizer
~0.9 s (transitive via scipy).  `backend/scipy.py` already has the
lazy pattern for jax but imports scipy.linalg/special eagerly; the
1894-line `__init__` re-exports everything eagerly (no PEP 562).
Lazy scipy + deferring solver subpackages would roughly halve cold
import for propagate-only workflows.

### S4-P3 bundle

- S4-13 [P3][physics] GUI interferometry "Extract" is silently
  meaningless: phase shifts never applied to the frames AND the
  `(phase, modulation)` tuple is not unpacked (numpy coerces to a
  (2,H,W) array; reported "RMS 0.586 rad" with no error)
  (`ui/interferometry_dock.py:224-240`).
- S4-14 [P3][robustness] codegen writes with locale encoding
  (`io/codegen.py:226`, no `encoding=`): cp1252 UnicodeEncodeError on
  Windows for non-latin glass/system names; platform-dependent output.
- S4-15 [P3][gap] `clear_caches_on_exit` fallback fan-out hand-lists 7
  clearers, omitting berreman/pmm/rcwa/glass/wrapper_merit/eme_jax/
  bluestein (`_context.py:294`) -- the registry exists precisely to
  kill this pattern; call `clear_all_registered_caches()` instead.
  Also `_JAX_IFT_SOLVER_CACHE` never registered
  (`propagators/asymptotic_jax_twin.py:559`) -- pins compiled XLA
  memory across `clear_asm_caches()`.
- S4-16 [P3][organization] Dead code: `_assemble_2d_tensor`,
  `_require_inplane_tile` (pmm/twod_jones.py:123,91), `_axis_pair`,
  `_seg_outer_eps` (twod_staggered.py:329,333), `cascade()`
  (bor/zcascade.py:227), `_reconstruct` (fga.py:445),
  `_gram_cho_factor` (lenses_maslov.py:824), `_surface_sag_scalar`
  (raytrace/surface.py:276, in __all__ with zero consumers).  NOTE:
  `_reject_jax_offplane` (rcwa/_core.py:2115) is never called yet
  exported and cited as an ACTIVE "tracer contract" by three pmm
  modules -- verify the OOP-JAX rejection it documents is actually
  enforced elsewhere (queued for Stage 6).
- S4-17 [P3][convention] Deprecated Zemax aliases shipped 24 releases
  past their announced v5.0 removal with a now-self-contradictory
  warning (`__init__.py:1006-1017`); Migration-Guide never reconciled.
- S4-18 [P3][convention] Optimizer hygiene: analytic jac silently
  dropped for basin-hopping/DE/dual-annealing local searches
  (`driver.py:1208`); x0 not clipped into bounds (TNC raises, others
  clip silently, `driver.py:543`); merit families on wildly different
  scales (normalized vs dioptre^2 vs absolute m^2,
  `merit_terms.py:66,1208`); hard-coded seed=42 on all stochastic
  methods (`driver.py:1202`); Zernike-RMS quadrature formula
  duplicated 4x (`context.py:290`, `merit_terms.py:289,848,899`);
  MaxThicknessMerit penalizes AIR gaps contra docstring
  (`merit_terms.py:1290`, sibling Min- skips air).
- S4-19 [P3][convention] IO hygiene: `.txt` `_parse_float` silently
  substitutes defaults on malformed numerics (`:944`); Quadoa export
  leaves aspheric coeffs unscaled for units != M
  (`prescriptions_quadoa.py:44`); storage metadata not round-trip
  faithful (list->ndarray on h5, un-reversed dict flattening,
  `storage.py:821,1184`); `replay_run` ignores stored per-plane
  wavelength (`:1560`); stale `.lock` docstring (`:63`); user_library
  silent clobber + sanitized-name collisions (`user_library.py:74`)
  and `eval()` on expression masks with full np exposed (`:585`,
  known/tracked).
- S4-20 [P3][convention] `_validation.py` absent from the mypy strict
  whitelist (`pyproject.toml:288`); MANIFEST.in omits
  Migration-Guide/CONVENTIONS/ROADMAP from the sdist; GUI psf_mtf
  ray-traced pupil uses `exp(+ik OPD)` vs wave-optics lens screens'
  conjugate convention (mirror-flipped PSF between the two sources,
  `ui/psf_mtf_dock.py:259`); lensmaker seeds omit the thick-lens term
  (`ui/main_window.py:2288`).

### Stage 4 verified sound (spot list)

- IO: even-asphere PARM unit scaling exact (incl. IN/CM); curvature/
  conic conventions; .zmx STOP + coord-break folding + UTF-16
  detection; make_singlet -> export -> reload -> apply_real_lens
  BIT-IDENTICAL; zarr+HDF5 complex round-trips bit-identical; sane
  chunking (16.7 MB gzip-4); CODE V + Quadoa loaders emit stop_index
  and round-trip internally.
- Optimize: bounds enforcement verified across 6 scipy methods;
  JaxMeritTerm analytic gradient exact vs closed form; jac='auto'
  combined gradient 1.4e-9 vs central FD; merit terms read
  through_focus D4sigma metrics (do NOT inherit the ring-sampling
  bias); ToleranceAwareMerit uses common-random-numbers across FD
  stencils (gradient-stable); full ray+wave design_optimize converges
  (EFL exact to 1e-11); honest converged/iterations reporting.
- GUI: tilt carrier == library formula; per-surface phase-screen sign
  == library; PSF/MTF/Strehl/zernike/AO/ghost/through-focus docks are
  thin library wrappers (~13 docks verified); waveoptics dock
  deep-snapshot discipline sound; glass lookups thread-safe.
- Infra: __all__ integrity (663 exports, 0 missing/dup); cache
  registry complete by construction (register-on-import); sibling
  clearers complete; set_default_complex_dtype flushes exactly the
  dtype-sensitive caches; id()-keyed BOR jax caches safe (objects
  retained); every module-level cache lock-guarded; backend dispatch
  raises on mixed backends (no silent fallback); RandomState jax
  branch fixed + guarded; py.typed shipped; pyproject extras' env
  markers correct.
- Duplication scan: no DRIFTED physics duplicates found (all reported
  clusters currently agree) -- the risk is forward maintenance, not a
  live divergence.

### Stage 4 feature gaps (ranked)

1. JAX CI leg (S4-4) -- highest coverage-per-runner-minute in the
   whole audit.
2. Import-time program (S4-12): lazy `backend/scipy.py` + PEP-562
   lazy solver subpackages.
3. GUI optimizer bounds UI (mm) threaded into both geometric and wave
   paths (fixes S4-2 root cause and the unbounded geometric path).
4. Zemax BICONICX/TOROIDAL/DGRATING parsing (cleanly warned today;
   builders already emit radius_y/conic_y so the model side is ready).
5. Windows unit-test CI shard (unit gate is Linux-only; Windows is
   the primary dev platform).
6. MultiFieldMerit per-field best-focus window (off-axis Strehl
   systematically underestimated beyond bfl/20 field curvature).
7. user_library versioning/collision handling; storage metadata
   normalization.

---

## Stage 5 -- Cross-cutting seams, feature gaps, performance and memory

Three auditors (test-suite quality; cross-engine seam consistency;
performance/memory profiling).

### S5-1 [P1][test-quality] The Seidel subsystem has NO independent magnitude oracle; the flagship test is a range window the documented bug passes -- `tests/unit/test_seidel_ground_truth.py:88`

`test_S1_matches_ray_trace_OPD_fit` promises an OPD rho^4 fit
(`8b ~ S1`) but asserts only `1e-5 < |S1| < 5e-4` -- its own docstring
records correct ~5.8e-5 AND the pre-4.9 buggy ~2.6e-4, BOTH inside the
window.  The tight Cassegrain pins (`test_audit_misc.py:2758`) admit
in-comment that expected values come from the library's own
Welford/Schwarzschild formula, referencing an out-of-repo scratch
file.  Validation-side Seidel checks are sign/scaling-only.  This is
the structural hole S3-1 (the S4/S5 sign bug) lived in for ~30 audits.
Fix: implement the promised traced-OPD polynomial fits as absolute
gates for S1-S5; keep formula pins only as labeled behavior-freeze.

### S5-2 [P1][validation-oracle] "Zemax OPD validation" is a self-referential, non-gating, CI-orphaned check -- `validation/real_lens_opd/run_validation.py:1`

The ".zmx prescriptions" are lumenairy-authored exports; the OPD
oracle is lumenairy's OWN ray tracer; the report has no pass/fail
threshold; and the directory contains no `test_*.py`, so
`run_all.py`'s rglob never discovers it -- CI (validate.yml:53) never
runs it.  A systematic wave-vs-geometric OPD bug on the core real-lens
path would trip nothing automated.  Fix: rename honestly, add bounded
residual assertions, expose as test_*.py; commit real OpticStudio
references if external truth is wanted.

### S5-3 [P2][seam] Patterned-medium argument silently switches between index and permittivity across engine entry points -- `elements/rcwa/oned.py:299`, `elements/pmm/oned.py:313`

`rcwa/pmm_efficiency_1d` take `n_ridge/n_groove` (INDEX -- probed
identical results across engines), but `*_jones_1d` take `eps_*` and
ALL 2-D entries take `eps_cell/eps_pillar` (PERMITTIVITY) -- while
`n_substrate/n_superstrate` are index EVERYWHERE, so one call mixes
conventions.  A wrong-convention value yields a plausible silent wrong
answer (n=2.1 as eps -> n_eff 1.45).  `rcwa_efficiency_1d` carries a
CONVENTION WARNING docstring; `pmm_efficiency_1d` does not.  Fix:
mirror the warning, document 2-D, add `glass_permittivity()` helper.

### S5-4 [P2][seam] Standalone Jones return tuples have divergent positional meaning; no transmission Jones from rcwa/pmm standalone -- `elements/rcwa/oned.py:1047` vs `elements/berreman.py:552`

rcwa/pmm return `(orders, R, T, J_reflection)`; berreman returns
`(R, T, J_r, J_t)`: `result[3]` is REFLECTION for two engines and
TRANSMISSION for the third; `result[2]` is efficiencies vs a Jones
matrix.  The transmitted Jones -- the observable for transmissive
metasurfaces -- is unavailable from standalone rcwa/pmm jones
functions entirely (Stack classes only).  Fix: named-field result
objects (or add J_trans); document the mismatch now.

### S5-5 [P2][gap] Engine->propagator bridge exists only for RCWA -- `elements/rcwa/stack.py:330,417`

`to_jones_field`/`to_multiorder_field` (verified convention-correct
incl. the subtle per-order power normalization) live only on
`RCWAResult`.  PMM (the phase-accurate engine), Berreman, and BOR
yield raw 2x2 Jones -- users must hand-assemble carriers and
normalization to propagate their output.  The cookbook pipeline is
RCWA-only, and NO example exercises the bridge (zero regression
protection at example level).  Fix: shared mixin keyed on
per_order_amplitudes, or a free `jones_field_from_orders()`.

### S5-6 [P2][seam] Solve-result containers: object vs tuple vs dict across engines -- `rcwa/stack.py:172`, `pmm/stack.py:799`, `berreman.py:743`, `bor/bor_stack.py:232`

RCWAStack -> RCWAResult object; PMMStack -> 4-tuple; BerremanStack ->
3-tuple (result[2] = J_refl where PMM's result[2] = T!); BORStack ->
dict.  Engine-agnostic post-processing must special-case all four.

### S5-7 [P2][test-quality] Known EVENASPH loader bug green-skipped; weak-gate and magic-number pins -- `tests/unit/test_audit_misc.py:691`

`pytest.skip("EVENASPH loader off-by-one bug present ... will activate
when prescriptions.py:580-585 is fixed")` masks an acknowledged
shipped defect as a green skip (VERIFY at HEAD whether the underlying
bug is still live -- queued for Stage 6, since the .zmx PARM path was
separately probed correct).  Also: the Welford-mirror dispatcher pin
degrades to no_raise/no_nan for 5 analysis functions
(`test_v4_14_0_dispatcher_pin_welford_mirror.py:253`); image-plane-WFE
pins an out-of-repo rayoptics magic number
(`validation/analysis/test_image_plane_wfe.py:146`);
`through_focus_smoke` asserts nothing and is CI-orphaned;
`elements/materials.py` (public CSV n/k loader) has ZERO test or
validation coverage.

### S5-8 [P2][perf] Measured no-loss wins (all probe-verified at HEAD)

- `threadpool_limits` rebuilt per RCWA solve (8.8 ms Windows DLL
  enumeration each): caching the ThreadpoolController gives a
  RCWA 20-wavelength sweep 283 -> 105 ms (2.69x), BIT-IDENTICAL --
  `elements/rcwa/_core.py:83,93`.  Also lifts the threaded-stack
  small-problem ceiling (enumeration is GIL-serialized).
- ASM H-build `np.exp` -> numexpr: 20-step 2048^2 ASM chain
  6.14 -> 3.79 s (1.62x), BIT-IDENTICAL (max diff 0.0) --
  `propagators/asm.py:124,191,234,856`.
- FFTW_THREADS defaults to ALL cores (24): threads=8 is 11-18% faster
  at 1024^2-2048^2 (oversubscription) -- `propagators/fft_infra.py:112`
  (roundoff-level, not bit-identical).
- FGA ray launch uses 9x finite-difference ray volume where the
  analytic differential Jacobian exists (rot-symmetric prescriptions):
  the 29.6 s FGA workflow is dominated by it -- `raytrace/
  differential.py:117`, `propagators/fga.py:695` (within ~1e-8, gate
  before switching).
- through_focus rebuilds meshgrid 42x for 21 planes and recomputes
  centroid/|E|^2 per metric (~20-30% of scan time, bit-identical fix)
  -- `analysis/beam_stats.py:72`.
- Idle `pyfftw.interfaces.cache` daemon enabled but never used (the
  library uses raw FFTW plans): a polling thread that pollutes every
  profile -- `propagators/fft_infra.py:88`.
- ASM H built shifted then `ifftshift`-rolled (~28 ms/cold build,
  avoidable by building in natural frequency order, bit-identical) --
  `propagators/asm.py:261`.

### S5-P3 bundle

- S5-9 [P3][seam] No shared LayerSpec across stacks (RCWA sampled
  cells vs PMM segments): one geometry cannot be replayed across
  engines for cross-validation without rebuilding.
- S5-10 [P3][convention] 2-D entries reject `angle=` (TypeError; 1-D
  accepts both `angle`/`theta`); `theta` silently overrides `angle`
  when both passed (CONVENTIONS.md section 6 mandates mutual
  exclusion); te/tm vs x/y bases coincide only at phi=0
  (undocumented); 1-D Jones entries lack conical phi (Berreman has
  it); BOR: mode/order terminology, dict return, not top-level
  exported.
- S5-11 [P3][seam] `RCWAStack.add_layer(eps=<3x3>)` accepted then
  crashes opaquely at solve (`stack.py:1420`) -- no uniform-tensor
  layer path (confirms S1-14 from the API side);
  `to_jones_field`/`apply_reflection` default `port='reflection'` --
  wrong port for the majority transmissive-metasurface use.
- S5-12 [P3][test-quality] Exact-equality (rtol=0=atol) on
  sweep-vs-single and backend-parity comparisons
  (`test_v5_14_0_pmm2d_stack.py:182` etc.) -- cross-platform flake
  risk (BLAS reduction order); atol=0 with rtol on can-be-exactly-0
  quantities (`test_v5_4_zernike_normalization_weighting.py:263`);
  dead xfail scaffolding + no `xfail_strict`; no `filterwarnings`
  config (physics warnings can pass silently); 50-wave RMS gate on a
  paraxial singlet (`test_audit_glass.py:147`); release-keyed test
  naming makes per-module coverage unfindable; helper builders
  re-copied across files.
- S5-13 [P3][coverage] No validation-suite leg for the newest
  propagators (fga, multibranch, levin, chebyshev, mft, ensemble);
  ui/ effectively unverified (~38 of 50 modules zero test refs);
  `raytrace/from_field.py`, `world_trace.py` thin.

### Stage 5 verified consistent / sound

- Cross-engine Jones basis and off-diagonal SIGNS agree: uniform
  anisotropic slab, oblique -- rcwa vs berreman 5.9e-16, pmm vs
  berreman 2.8e-13; conical (theta=25 deg, phi=40 deg) rcwa_2d vs
  berreman 3.3e-16, pmm_2d 1.6e-14.  No silent s/p flip anywhere.
- `apply_jones_matrix` composition convention matches every engine's
  Jones output ([out,in]); te<->Ey / tm<->Ex exact at phi=0; per-pol
  energy accounting uniform; radians + meters uniform everywhere;
  n_substrate/n_superstrate index-valued everywhere.
- RCWA -> JonesField bridge convention-correct: carrier
  exp(+i(kx x+ky y)) on centred grid matches exp(+ikz) forward
  convention and the per-order Poynting normalization is right.
- Test suite strengths: optional-dep skip hygiene strong (no platform
  skips, no unconditional skips); audit-P1 test files use the ideal
  physics-gate + labeled-pin pattern; validation suite ~50% genuinely
  independent analytic oracles; `run_all.py` IS run in CI on
  ubuntu+windows.
- Performance: all previously-shipped speedups ACTIVE at HEAD
  (even-parity fold fires at normal incidence only -- correct;
  threaded stack 6.0x at n_orders=41; shared-eig caches consulted;
  numba MultiFieldMerit kernels present; FGA chunking exercised).
  rfft2 inapplicable (fields inherently complex); checkerboard
  shift-fold REFUTED (rel err 1.44); no systemic complex128 upcast.
  Biggest ASM transient (fftshift roll copies, 537 MB at 2048^2) is
  NOT no-loss-avoidable -- documented as an inherent watermark.

---

## Stage 6 -- Adversarial verification of findings

Five verifiers, each instructed to REFUTE their assigned findings via
independent reproduction (own probes written from the claim text, no
access to the original auditors' scripts).  Working-tree note: HEAD
advanced again to `826e606` (CI-only commit above 2628426); code
verified as present.

**Every P1 was independently CONFIRMED.**  Two candidate escalations
were refuted/downgraded, and one root cause was corrected.  Detail:

| Finding | Verdict | Verifier evidence (independent) |
|---------|---------|--------------------------------|
| S1-1 prepare() OOP truncation | CONFIRMED P1 | own probe max dJ 2.5e-2 (TM channel), energy conserved both paths, zero warnings; in-plane control bit-identical |
| S1-3 PMM-2D hybrid boundary energy | CONFIRMED P2 | own cell: E=1.25 (n_orders=15), E=2.36 (boundary), silent; twod.py has NO energy-warning code at all |
| S4-16 `_reject_jax_offplane` dead | CONFIRMED P3, escalation REFUTED | the OOP-JAX path is correctly SUPPORTED: jax vs numpy 2.96e-15, grad==FD; DELETE the dead function + fix 4 stale citations -- do NOT wire it in (would regress a working differentiable path) |
| S5-7 EVENASPH skip | DOWNGRADED to P3 stale-skip | bug FIXED in v5.16.1; round-trip bit-identical; `prescriptions.py:580-585` no longer exists (106-line shim); un-skip the test |
| S2-1 multibranch double-count | CONFIRMED P1, STRENGTHENED | own flat plate: P ratio 4.54 at h/dx=2 (49.7% pixels multi), up to 9.7x at h/dx=1; excess exactly on the n=2 edge / n=6 node lattice (masking -> 0.61); DEFAULT ray_subsample=2 on power-of-2 grids IS the catastrophic commensurate case (h/dx = sub exactly for flat/afocal); incommensurate grids ~1%; single-branch traced immune (0.956) |
| S2-8 axial-focus blowup | CONFIRMED P2 (borders P1) | 1.039e6 at BFL to the digit, max_n_branch=208, ZERO warnings; ludwig ~ plain (pair-swap does nothing vs 208 branches); docstring "not a blow-up" is false |
| multibranch test luck | CONFIRMED P2 | test passes as committed; d=2.8 mm gives 1.098 (9.81%, fails its own 8% gate); the 2.8 mm excess is the ONSET OF THE AXIAL DIVERGENCE (S2-8), so denser plane sampling would catch both bugs |
| S2-2 FGA power deficit | CONFIRMED P1 | own probes 0.90/0.69/0.66 vs claimed 0.927/0.691/0.645; p_max sweep recovers exactly across the frame bound 2/(k w0); docstring exactness claim verified present; 'power' fully recovers |
| S2-6 FGA NA cap | CONFIRMED P2 (understated) | fid 0.96 at content 0.156, 0.80 at 0.262 vs 1.0000/0.9997 manual |
| S2-21 `_caustic_zone` row index | CONFIRMED P1 (one nuance) | wide grid IndexError reproduced through `apply_real_lens_auto` DEFAULT path (unconditional, :1234) and universal's high-NA branch (:1441 -- universal is NOT unconditional: low-NA routes to phase_screen without crash); tall grid silently reads row 32 not 128 -> ~17x-narrower wrong zone |
| S2-7 FGA separable!=direct | ADJUSTED (cause corrected) | phenomenon real: field diff 4.6e-4 exactly at w0_factor=5.000 (2e-15 at 4.999/5.001), "ULP-identical" docstring false on the default.  BUT analysis coefficients are ULP-identical (1.5e-15) -- the claimed analysis-box-bound cause is REFUTED.  True defect: `_scatter_sep`'s recurrence-advanced `dxr` straddles the r^2<=R^2 gate differently from `_scatter`'s fresh computation for ON-GRID beamlets (rel 6e-3 at the tie).  Fix belongs in `_scatter_sep` (fresh gate coordinate), mirroring the historic analysis-gate fix |
| S2-22 dispatcher/FGA power scale | CONFIRMED P2 | fga default 16.8x-39.1x over w0_factor 3-6 in verifier's stronger-focus setup (vs phase_screen 0.998, gbd 0.992); both dispatcher call sites verified passing no normalize_output; note the dispatchers route to FGA exactly near caustics where the scale error is worst |
| S2-3 JAX c64 propagators | CONFIRMED P1 (understated) | ASM/MFT/Fresnel jax-c64 1.78e-3/1.78e-3/1.92e-3 vs numpy-c64 1.5e-7 (~4 decades, ~80 dB); x64 NOT auto-enabled by import or by these entry points (trigger = passing a jnp array); only JAX's own one-shot arange warning; x64-on restores 3.2e-16 parity |
| S3-2 JAX through-focus f32 | CONFIRMED P1 | divergence grows with NA (3.2% at F/1.5 on a smooth analytic pupil; the 14.5% needs the hard-edged doublet field); x64-on = EXACT numpy parity; `_require_jax_x64` verified wired into all four asymptotic-twin entries and ABSENT here |
| S2-13 JAX system drops dy | CONFIRMED, RAISED to P2 | anamorphic probe: jax system output == dy->dx reference to 5e-16 and 50% wrong vs correct; numpy path bit-exact correct.  Silent 50% physics error justifies P2 (was P3) |
| S3-1 Seidel S4/S5 | CONFIRMED P1 (maximal rigor) | verifier wrote an INDEPENDENT Welford oracle AND ray-traced distortion directly (no Seidel formulas): ray-traced distortion vanishes with lens thickness for stop-at-thin-lens while library S5 does not (1.21e-5 vs 1.0e-9 at d=1 um); S3/S4 sign anchor confirmed; the two-part fix (negate S4 :1036/:1080 + flip S5 prefactor :1039/:1081/:1123) NUMERICALLY VALIDATED to restore vanishing distortion and match the oracle at full precision; BOTH parts required; flat-surface branch needs no change; `seidel_wfe`'s `(1/4)(S3+S4 H^2) rho^2` corruption confirmed at seidel_analysis.py:349-357 |
| S4-1 .txt CB gap drop | CONFIRMED P1 | verifier hand-built its own .txt/.zmx pair: [0.004, 0.006, 0.004] vs [0.004, 0.009, 0.004], no warning; root-cause fold loop present only in .zmx (:571-584) |
| S4-2 GUI optimizer units | CONFIRMED P1 | unambiguous from source: to_prescription mm->m (model.py:2776) vs get_variable_values mm (:2428-2430) -> bounds mm, x0 m (parameterizations.py:207, driver.py:543-544) |
| S4-3 GUI asm drops aspherics | CONFIRMED P1 | build_trace_surfaces (model.py:2138) omits aspheric_coeffs ENTIRELY and the asm loop never reads radius_y/conic_y; `surface_sag_general` even accepts aspheric_coeffs -- the loop just does not pass it; layout_2d:429 draws the full biconic (picture-right/physics-wrong confirmed); no warning path exists |

Verification outcome summary: 10/10 code P1s confirmed (one with a
routing-precision nuance), 6 load-bearing P2s confirmed (one raised
from P3, one adjusted to the correct root cause), 2 candidate
escalations correctly refuted/downgraded.  No finding was withdrawn.

---

## E2. Feature-gap synthesis (cross-stage, ranked by user value)

Consolidated from the per-stage gap lists, the module roadmaps
(docs/*roadmap*), and the seam matrix; known-deferred roadmap items
are included only where this audit adds information.

1. **Engine->propagator bridge for PMM/Berreman/BOR** (S5-5): promote
   `to_jones_field`/`to_multiorder_field` off RCWAResult onto a shared
   mixin.  Unlocks the phase-accurate PMM and exact-anisotropic
   Berreman engines for the propagation pipeline; also add the missing
   transmission Jones to standalone rcwa/pmm jones functions (S5-4).
2. **JAX trust program**: library-wide `_require_jax_x64()` (closes
   the S2-3/S3-2/S2-13 class in one move), one jax CI leg (S4-4),
   traceable-wavelength dispersion for chromatic gradients (S3-6),
   then mirrors/biconics in the jax tracer.
3. **2-D anisotropic cells in the no-floor staggered/pure PMM engine**
   (roadmap Phase C; today anisotropic 2-D exists only on the
   FMM-floored hybrid) and the mixed Li-1997 crossed-tensor rule for
   the hybrid.
4. **Import-time program** (S4-12): lazy `backend/scipy.py` + PEP-562
   lazy solver subpackages; ~halves cold import.
5. **Propagation-in-medium**: `n_medium` kwarg (or documented
   wavelength-scaling recipe) on the ASM/Fresnel/MFT/SAS family.
6. **Result-container unification + neutral LayerSpec** (S5-6, S5-9):
   enables engine-agnostic post-processing and one-geometry
   cross-engine validation.
7. **Multibranch axial-caustic routing** to the Maslov-levin evaluator
   (S2-8's proper fix) after the S2-1 rasterizer fix.
8. **Sources**: z/ROC/Gouy placement for Gaussian/HG/LG; true LP
   fiber-mode solve (na is currently inert -- S3-4); Bessel-Gauss;
   polarization channel on Source (S3-17); realization iterator.
9. **Zemax surface coverage**: BICONICX / TOROIDAL / DGRATING parse
   (model side already supports radius_y/conic_y); .txt loader parity
   (S4-1/S4-9).
10. **Analysis hardening**: ISO-11146 background/aperture option on
    second moments (S3-5); on-axis vs peak Strehl convention note;
    per-field best-focus window in MultiFieldMerit.
11. **Raytrace**: reflect-on-TIR option for ghost/prism work; exact
    Zemax-form biconic sag.
12. **Test infrastructure**: independent-oracle gates for Seidel
    (S5-1); wave-vs-geometric OPD residual gate in CI (S5-2);
    validation legs for fga/multibranch/levin; materials.py coverage;
    Windows unit-CI shard; `xfail_strict`; filterwarnings config.

## E2b. Performance / memory opportunity ranking (verified no-loss)

| # | Change | Gain (measured) | Identity |
|---|--------|-----------------|----------|
| 1 | Cache ThreadpoolController in `_blas_limit` (rcwa/_core.py:83) | RCWA sweeps 2.69x | bit-identical (0.0) |
| 2 | numexpr for ASM H-build exp (asm.py:124,191,234,856) | ASM chains 1.62x | bit-identical (0.0) |
| 3 | FGA: analytic differential Jacobian instead of 9x FD rays for rot-symmetric prescriptions (fga.py:695) | up to several x on the 29.6 s FGA workflow | ~1e-8 (gate first) |
| 4 | FFTW_THREADS default min(cores, ~8) (fft_infra.py:112) | 11-18% FFT-bound | roundoff-level |
| 5 | Hoist meshgrid + share centroid/abs^2 in beam_stats (beam_stats.py:72) | 20-30% of through-focus scans | bit-identical |
| 6 | Build ASM H in natural frequency order (skip ifftshift roll) (asm.py:261) | ~28 ms/cold build | bit-identical |
| 7 | Sources: broadcasting instead of meshgrid (10 sites) | ~1 GB transient at N=8192 | bit-identical |
| 8 | Berreman OOP-oblique eig/interface caching (berreman.py:428) | wavelength sweeps of tilted-director stacks | bit-identical |
| 9 | In-place masked updates in _refract/_intersect (intersection.py:304) | 5-15% ray-heavy paths | bit-identical values |
| 10 | Remove idle pyfftw interfaces-cache daemon (fft_infra.py:88) | profile hygiene, one thread | no compute path touched |
| 11 | PMM OOP-tile detection relative threshold (twod_jones.py:120) | avoids accidental 8x generator + lost parity fold | exact for in-plane |
| 12 | SHWFS reconstructor cache (ao.py:1223); trace_jax sweep signposting (jax_trace.py:1208); nodal-BOR single eig (bor_solve.py:87) | niche paths 2x+ | bit-identical |

All previously-shipped speedups verified ACTIVE (even-parity fold,
threaded stack, shared-eig caches, numba merit JIT, FGA chunking).
Refuted: rfft2 (fields inherently complex), checkerboard shift-fold
(invalid, rel 1.44), systemic complex128 upcast (none found).  The
537 MB fftshift transient at 2048^2 is inherent to the current
algorithm (no bit-identical elimination exists).

---

## Rollups

### Executive summary

30 Opus auditors + 5 adversarial verifiers over six stages.  The
library's core physics is in excellent shape: every EM engine, the
scalar propagation core, the ray tracer, the analysis metrics, and the
glass/IO round-trips passed independent-oracle probes (often at
1e-13..1e-16).  The serious defects concentrate at four kinds of
places, consistent with this repo's audit history:

1. **The newest, least-audited code** (FGA auto-sampler regression
   S2-2, multibranch rasterizer S2-1, dispatcher row-index S2-21).
2. **Un-guarded JAX float32 paths** (S2-3, S3-2, S2-13) -- enabled by
   JAX being in no CI leg (S4-4).
3. **Diverged duplicate twins** (.txt/.zmx loaders S4-1; the exact
   pattern that produced the historical factor-i bug).
4. **Self-referential test oracles** (Seidel S3-1 survived ~30 audits
   because its "ground truth" test re-pins the library's own formula,
   S5-1/S5-2).

12 P1s total (10 code + 2 test-oracle), all adversarially verified.
No P0 (nothing corrupts data or breaks the default path of the core
propagate/solve workflows).  REMEDIATION NOT STARTED -- this document
is the audit deliverable.

### Findings index by severity

P1 -- verified (code):
| ID | Title | Where |
|----|-------|-------|
| S1-1 | PMMStack.prepare() silently drops OOP tensor components | elements/pmm/stack.py:2112 |
| S2-1 | Multibranch rasterizer double-counts shared triangle edges (catastrophic at DEFAULT ray_subsample on power-of-2 grids: 2x-10x energy) | elements/_lens_traced_multibranch.py:583 |
| S2-2 | FGA v5.24.3 auto-sampling: 30-35% power deficit on identity (regression) | propagators/fga.py:521 |
| S2-3 | JAX complex64 ASM/MFT/Fresnel: ~4 decades phase-accuracy loss vs contract | propagators/asm.py:191, mft.py:198, fresnel.py:145 |
| S2-21 | `_caustic_zone` row-index bug: auto/universal dispatchers crash on wide grids, wrong row on tall | propagators/fga.py:1138 |
| S3-1 | Seidel S4 sign inconsistent with S1-S3: S5 and seidel_wfe field curvature corrupted (fix numerically validated) | raytrace/seidel.py:1036,1080 + :1039,:1081,:1123 |
| S3-2 | JAX through-focus silently float32: Strehl>1, design mis-ranking | analysis/through_focus.py:1056 |
| S4-1 | .txt Zemax loader silently drops COORDBRK axial gaps | io/prescriptions_zemax.py:1116 |
| S4-2 | GUI wave optimizer: mm bounds vs metre x0 (all length variables broken) | ui/optimizer_dock.py:923 |
| S4-3 | GUI default 'asm' model drops aspheric coeffs + biconic curvature | ui/waveoptics_dock.py:865, ui/model.py:2138 |

P1 -- verified (test/validation oracles):
| S5-1 | Seidel subsystem has no independent magnitude oracle (flagship test is a range window the documented bug passes) | tests/unit/test_seidel_ground_truth.py:88 |
| S5-2 | "Zemax OPD validation" is self-referential, non-gating, CI-orphaned | validation/real_lens_opd/ |

P2 (34; verifier-confirmed where marked *):
S1-2 RCWA lossless tripwire missing on 2 paths; S1-3* PMM-2D hybrid
boundary energy (E up to 2.4 silently); S1-4 EME sigma on band-top;
S2-4 GBD per_surface=False output plane + dropped z_image; S2-5 GBD
soft-edge aperture stale waist0; S2-6* FGA NA cap truncates wide-angle
free space; S2-7* FGA separable!=direct at default (cause: scatter
recurrence gate); S2-8* multibranch axial blowup unguarded (borders
P1); S2-9 GBD diverging-relay no diagnostic; S2-13* JAX system drops
dy (raised from P3); S2-22* dispatcher->FGA power scale (4-39x);
S2-23 gbd/fga missing mirror guard; S2-24 caustic zone ignores
y-caustics; S2-25 carrier-residual dx-for-dy; S2-26 tilt-dispersion
Nyquist conflation; S3-3 point-source/tilt chirp aliasing unguarded;
S3-4 fiber-mode na inert; S3-5 second moments no background guard;
S3-6 jax merit 4/6 inputs crash; S3-7 sources meshgrid memory;
S3-8 jax through-focus zero-plane divergence; S4-4 JAX in no CI leg;
S4-5 optimizer failure-direction bugs (nan->perfect, inf injection,
lm clamp); S4-6 GUI thickness variables KeyError; S4-7 GUI live-model
thread mutation; S4-8 .zmx/.txt duplicate twins; S4-9 IO silent
fallbacks (UNIT, aperture=0, stop loss, h5/zarr None divergence);
S4-10 EME scalar/vector n_scan drift (+ PMM2D stack n_orders 11 vs 7);
S4-11 new duplication clusters (Zernike x2, BOR operator x2, HFPI x6,
DOE kick x2, Chebyshev, Sellmeier, Strehl-Marechal, waist-estimate);
S4-12 import time 3.5-6 s; S5-3 n-vs-eps convention switching; S5-4
Jones return-tuple divergence + missing transmission Jones; S5-5
engine->propagator bridge RCWA-only; S5-6 result containers
object/tuple/dict; S5-7 test-oracle bundle (EVENASPH part DOWNGRADED
P3 -- stale skip, bug fixed v5.16.1); S5-8 measured perf wins.

P3 (~60): hygiene/convention/docs bundles S1-5..S1-20, S2-10..S2-20,
S2-27, S3-9..S3-19, S4-13..S4-20, S5-9..S5-13 -- notable standouts:
CONVENTIONS.md waveplate sign row would reinstruct a fixed bug (S1-5);
`_solve_core` gauge docstring trap (S1-6); codegen cp1252 trap
(S4-14); `_reject_jax_offplane` dead-but-cited (delete, do NOT wire
in); interferometry dock silently meaningless self-check (S4-13).

### Areas verified sound (independent oracles; safe to build on)

- RCWA: OOP dispersion vs Christoffel 1.3e-15; end-to-end OOP Jones vs
  independent 4x4 8.5e-16; Li rules (1-D TM, 2-D sequential, z-rule);
  S-matrix evanescent stability; cache-key completeness; jax selector
  parity.
- PMM 1-D/2-D: factor-i fix everywhere (incl. magneto-optic); slant
  conventions; multi-region slant; A|B pure cascade vs 1-D multilayer
  + RCWA; mirror symmetry 3.5e-15; even-parity fold; hybrid floor
  honestly documented.
- Berreman/EMT/glass: Fresnel/TMM 1e-16; loss-sign consistent across
  engine families; EMT Rytov convergence; catalogs bit-identical to
  refractiveindex.info.
- BOR/EME: grazing fix 1.22e-11; Fresnel+TIR 5.3e-15; m-order axis
  regularity; jax twins to 9e-14/bit-identical.
- Cross-engine: Jones bases and off-diagonal SIGNS agree
  (rcwa/pmm/berreman, planar+conical, 1e-13..1e-16); per-pol energy
  uniform; radians/meters uniform; apply_jones_matrix convention
  matches; RCWA->JonesField bridge convention-correct incl. per-order
  power normalization.
- Scalar propagation: ASM vs analytic Gaussian; reciprocity 4.9e-16;
  backend parity 4e-16; no input mutation; caches complete + bounded;
  tilted-ASM exact centroid; c64 numpy contract honored.
- GBD/FGA/traced/Maslov/Levin: beamlet Q vs analytic 1e-16; OPL
  vertex-plane fix non-regressed; FGA chunking/pruning identities
  (<=1.7e-14 with pruning, adversarial late-chunk peaks OK); Ez
  transversality exact; Levin bounds honored incl. interior
  stationary points; KMAH sign correct; traced exit curvature correct.
- Raytrace/JAX twin: intersection/Snell/TIR exact; sag/normal
  consistency 1e-12; forward parity 7e-18; ALL gradients FD-verified;
  no un-mirrored numpy fixes; NaN-safe where-guards verified at
  boundaries.
- Sources/analysis: HG/LG orthonormal; Schell statistics; MCF
  Hermitian/PSD; no half-pixel source<->propagator seam; Airy/MTF/EE
  vs analytic; STREHL CORRECT FOR APODIZED PUPILS; Zernike OSA
  orthonormal; M2 phase-curvature immune; ao_closed_loop law.
- Optimize: bounds enforcement (6 methods); JaxMeritTerm gradient
  exact; merit terms free of the ring-sampling bias; tolerancing CRN
  gradient-stable; honest convergence reporting.
- IO/infra: asphere unit scaling exact; builder->export->reload->
  apply_real_lens bit-identical; zarr/h5 complex round-trip exact;
  __all__ integrity 663/663; cache registry complete by construction;
  all caches lock-guarded; RandomState jax branch fixed; extras'
  env-markers correct.

### Feature-gap matrix and performance ranking

See sections E2 / E2b above (kept adjacent to the verification stage
that validated them).

### Suggested remediation order

1. One-line/small fixes with verified P1 impact: S2-21 (`shape[0]`),
   S3-1 (negate S4 + flip S5 prefactor -- fix already validated),
   S2-2 (p_max frame-bound floor), S4-1 (port fold loop), S1-1 (guard
   or route), S4-2 (metre bounds), S4-3 (biconic sag + thread coeffs).
2. Multibranch rasterizer half-open coverage (S2-1) + re-gate its
   energy test at dense planes (catches S2-8's ramp too).
3. JAX trust program: `_require_jax_x64` guards (S2-3, S3-2), dy
   threading (S2-13), jax CI leg (S4-4).
4. Silent-fallback guards: S1-2, S1-3, S1-4, S2-23, S4-9, S3-3.
5. Perf quick wins #1/#2 (threadpool cache, numexpr H) -- both
   bit-identical, both measured.
6. Consolidation program (S4-8, S4-11, S1-8, S1-9, S2-10, S2-14) --
   the fix-one-of-N pattern is this library's dominant historical bug
   vector; every cluster is agree-today.
7. Test-oracle program: S5-1/S5-2 independent gates, then the S5-12
   tolerance hygiene sweep.
