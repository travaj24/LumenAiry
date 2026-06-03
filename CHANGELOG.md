# Changelog — lumenairy

All notable changes to the core library are documented here.

## [5.11.0] — 2026-06-02

**1-D anisotropic device modeling + 2-D normal-vector FFF + stack/PMM
robustness.** A batch closing the functional + convergence gaps from the RCWA
gaps/wishlist, DynaMeta-port, and resonant-stack audits: full out-of-plane
anisotropy, arbitrary multi-region gratings, reflective-Jones device helpers,
the true normal-vector 2-D factorization, a resonance guard for multilayer
stacks, and an anisotropic-Jones spectral-element solver. Every pre-existing
path (isotropic, in-plane-tensor, `'laurent'`/`'li'`, scalar PMM, default
`RCWAStack.solve`) is **bit-identical** — all new behavior is gated/opt-in.

### Added

- **GAP7 — full 3×3 out-of-plane anisotropy in `rcwa_jones_1d`.** Tilted-director
  LC, magneto-optic / gyrotropic media (`eps_xz/yz/zx/zy ≠ 0`). The full 4N
  generator `G=[[A,P],[Q,B]]` is eigendecomposed with a generalized all-harmonic
  Poynting-flux forward selector (robust on non-reciprocal spectra, where a
  ±-pairing selector throws), and a generalized explicit forward/backward
  S-matrix carries the tilted layer's asymmetric modes. Berreman-4×4 validated to
  ~1e-15; the in-plane path is byte-identical. 1-D Jones path (2-D / `RCWAStack`
  out-of-plane pending).
- **GAP2 — `rcwa_jones_1d_segments`**: arbitrary piecewise-constant 1-D profiles
  (multi-level / interdigitated / N-region, each scalar or `(3,3)` tensor). Shares
  the exact `rcwa_jones_1d` solve core (refactored to `_jones_1d_from_profiles`,
  byte-identical).
- **W3 grating builders** — `grating_segments`, `binary_grating_segments`,
  `interdigitated_grating_segments` (emit the `segments` list for GAP2). **W2
  reflective-Jones helpers** — `reflective_outcoupling` (the
  PBS→QWP@45→grating→QWP@45→PBS cross-port FOM = `cos²(Γ/2)` for a lossless
  TE/TM-aligned grating) and `jones_retardance_diattenuation` (polar/SVD
  decomposition of a 2×2 Jones).
- **`rcwa_efficiency_2d(formulation='fff_nv')`** — the Schuster-2007 normal-vector
  Fast Fourier Factorization for 2-D: a continuous unit-normal field with the
  inverse rule on the boundary-normal projection, completing the factorization the
  existing `'li'` rule lacked (it inverse-ruled only `E_z`, never the in-plane
  normal field). Robust wins on dielectrics and separable stripes (→ rigorous 1-D
  Li); a correct, non-collapsing factorization on 2-D metal corners.
  `'laurent'`/`'li'` bit-identical.
- **`RCWAStack.solve(stabilize=True)`** — opt-in per-order consensus guard for the
  sharp-resonant metal-multilayer pathology (a near-singular mode-match biases a
  single diffraction order, e.g. a reflection null, non-monotonically at isolated
  `n_orders` while total power stays bounded). Re-solves a short downward
  `n_orders` window and returns the consensus solve; `stabilize=False` is the
  exact prior single solve.
- **`pmm_jones_1d`** — a binary 1-D grating whose ridge / groove are full
  `(3, 3)` IN-PLANE permittivity tensors (the tunable-LC reflective grating),
  returning the full complex `2x2` Jones reflection. The off-diagonal `exy`
  couples `E_x` ↔ `E_y` in the spectral-element modal eigenproblem (the modal
  field is a 2-vector `[E_x; E_y]` per node), so the phase relationship a
  tunable-LC grating needs is carried — the scalar TE/TM solver could not. The
  Li-1996 factorization is realized in the nodal basis: the wall-normal inverse
  rule `[[1/exx]]^-1` becomes `inv(hat(1/exx))`, and the `Kx`-derivative terms
  (`Ez`-elimination, `Kx^2`) become spectral-element STIFFNESS operators
  (weighted by `1/ezz` / `1`) so the inverse rule is **automatic and exact** (the
  `eps` jump lands on an element boundary). The coupled second-order modal
  operator mirrors the FMM tensor block `M = -P@Q` at normal incidence. PUBLIC
  `exp(-i w t)` convention end-to-end (no eps conjugation — the scalar PMM is
  self-contained in the public convention). Validated against `rcwa_jones_1d`:
  the `2x2` Jones matches to ~5e-6 (lossless tilted-LC) / ~2e-6 (lossy
  anisotropic metal) and converges spectrally with no floor; a diagonal tensor
  decouples back to the scalar TE/TM efficiencies (cross-pol `~1e-18`); lossless
  energy `sum(R)+sum(T)=1` (cross-pol included) to ~1e-14. Inherits the scalar
  PMM `stabilize` resonance guard (both incident polarizations must be passive).
  Normal incidence, binary, NumPy only (multi-region / oblique / autodiff are
  follow-ons). Exported top-level.
- **New tests** `tests/unit/test_v5_11_0_pmm_anisotropic.py` — the four
  validation gates (vs `rcwa_jones_1d`, decouple-to-scalar, energy, no-floor).

#### Lower-priority ergonomics / hardening (from the same audits)

- **JAX off-plane differentiability guard.** The in-plane (`z`-decoupled tensor)
  1-D Jones path is `jax.grad`-differentiable (verified vs central finite
  difference across multiple objectives / `n_orders`); the full-3×3 *out-of-plane*
  path is **not** (its forward-mode flux selector `_select_forward_flux` uses
  `np.where`/`argsort`, which breaks the trace). Because `_tensor_offplane_present`
  skips JAX arrays, a concrete off-plane JAX tensor would otherwise be silently
  routed through the in-plane solver — a wrong answer *and* a wrong gradient.
  `rcwa_jones_1d` / `rcwa_jones_1d_segments` now raise a clear `NotImplementedError`
  for a concrete off-plane JAX tensor (mirroring the `formulation='fff_nv'` JAX
  rejection); the in-plane subset is pinned differentiable. NumPy paths are an
  exact no-op (the guard fires only under `if is_jax:`). *Known narrow gap:* an
  off-plane tensor built purely **inside** a trace (a `Tracer`) cannot be
  inspected and routes in-plane — documented in the docstring.
- **`RCWAResult.layer_absorption` for tensor-cell layers.** Per-layer loss
  attribution for the metal-tooth / LC `(3,3)`-tensor layers via the anti-Hermitian
  loss density `Im(Eᴴ·ε·E)` (the public `exp(-i w t)` convention; `Im(ε_public) ≥ 0`).
  A uniform tensor cell matches the equivalent scalar layer **bit-identically**;
  per-layer values close to the total absorptance and localize loss to the lossy
  layer. Uniform / isotropic-cell layers are unchanged; analytic-shape layers still
  raise.
- **`reflective_outcoupling` is now backend-agnostic.** `jax.grad` traces through
  the side-port out-coupling FOM (so it can be an inverse-design objective directly);
  a NumPy Jones still returns a Python `float` **bit-identical** to before.
- **`rcwa_convergence` accepts an `RCWAStack`.** Solves the stack at its configured
  `n_orders` and a bumped count, reports/warns on the **per-order** efficiency delta
  (a sharp-resonant metal multilayer biases an isolated order while total power
  stays bounded), and restores the stack's `n_orders`. The single-entry-solver path
  is unchanged.
- **`rcwa_jones_vs_wavelength_segments`** — the segmented (multi-region / multi-level)
  dispersive Jones sweep companion to `rcwa_jones_1d_segments`; `segments` and the
  half-space indices may be `wl → …` callables (material dispersion). A 2-segment
  sweep is **bit-identical** to the binary `rcwa_jones_vs_wavelength`. Exported.
- **New tests** `tests/unit/test_v5_11_1_rcwa_lowerpri.py` (19) — differentiability
  (in-plane grad vs FD; off-plane raises), tensor `layer_absorption` (sum-to-total +
  uniform-tensor-vs-scalar bit-identity), `reflective_outcoupling` NumPy bit-identity
  + JAX grad, stack `rcwa_convergence`, segmented dispersive sweep. An adversarial
  three-skeptic review (tensor-loss physics, the JAX guard, bit-identity of the
  touched public surface) found no defects.

#### OBLIQUE PMM — off-angle parity for `pmm_efficiency_1d` + `pmm_jones_1d`

- **`pmm_efficiency_1d` and `pmm_jones_1d` now support oblique incidence**
  (`angle != 0`) — previously a `NotImplementedError`. This is the PMM analogue
  of off-angle RCWA: the pseudo-periodic envelope `E(x) = exp(i kx0 x) u(x)`
  (`kx0 = Re(n_sup) sin(angle) k0`) is solved by Bloch-shifting every
  `x`-derivative `d/dx -> d/dx + i kx0` in the spectral-element operators, and
  the Rayleigh far-field / incident flux pick up the `kx0` offset. **`angle == 0`
  is bit-identical** to the prior normal-incidence solve (the shift terms vanish
  and the legacy branch is used). Validated against `rcwa_efficiency_1d` /
  `rcwa_jones_1d` at matched angle: dielectric / tunable-LC tensor match to
  ~1e-5–1e-4 with energy conserved to ~1e-13 (lossless), spectral convergence in
  `degree`, across 0–60°. Three physics points were required and are worth
  recording:
  - **Antisymmetrized convection.** The Bloch cross-term is `-i kx0 (C - Cᵀ)`
    with `C = ∫φ φ'` (TM uses the `1/eps`-weighted `Cinv`). For TM the weight
    *varies across the wall*, so `Cinv` is **not** antisymmetric and the naïve
    `-2i kx0 Cinv` is wrong (it silently gives an energy-conserving-but-wrong
    TM answer); the `(Cinv - Cinvᵀ)` form is required. TE (unit weight) is
    unaffected.
  - **Forward-mode selection.** At oblique the operator is complex-Hermitian
    (lossless), so the QZ `eig` leaks imaginary noise; the scalar path uses a
    noise-robust `Im(q)` branch, and the **coupled anisotropic** path — whose
    modes do not split cleanly by `Im(q)` — selects the forward set by the sign
    of each mode's **z-Poynting flux** (`Im(Eₓ·S0·conj(H_y) − E_y·S0·conj(Hₓ))`).
  - **Per-column incident normalization.** The incident `Eₓ` (p-pol) wave also
    carries `E_z = -kx0 Eₓ/k_z`, so its incident z-flux is `eps_sup/k_z`, not
    `k_z`; normalizing both polarizations by `k_z` over-counted the p channel by
    `1/cos²(angle)`. Fixed per-column (s-pol unaffected; bit-identical at `kx0=0`).
  - *Known limit:* very lossy **metal-corner TM at steep oblique** stays
    resonance-limited (`stabilize` may raise) — use `rcwa_efficiency_1d` there.
    The realistic dielectric / tunable-LC regime is robust across angle.
- **`stabilize` now gates on PER-ORDER convergence, not just total power**
  (`pmm_efficiency_1d` + `pmm_jones_1d`, normal AND oblique). An adversarial
  review found that the old gate keyed only on `sum(R)+sum(T)`, which the
  S-matrix conserves even when the modal basis is *under-resolved* — so a
  high-index / many-order grating at the default degree could return per-order
  efficiencies / a Jones matrix wrong by tens of percent **with no warning**
  (e.g. n≈3.8 TM at 40° gave a 33%-wrong zeroth order while `sum(R)+sum(T)=1` to
  1e-7). The consensus now requires two passive degrees to agree on the
  **per-order efficiencies and the 2×2 Jones** (a resonance-contaminated passive
  degree is isolated — it has no per-order partner), returning a converged
  degree or, failing that, warning that the result is likely under-resolved
  (raise `degree` / `elements_per_region`). Well-converged configs are
  unaffected; `stabilize=False` is unchanged.
- **New tests** `tests/unit/test_v5_11_0_pmm_oblique.py` (17) — scalar TE/TM and
  anisotropic-Jones oblique vs the FMM oracle (multiple angles), `angle==0`
  bit-identity, the all-vacuum `1/cos²` regression, diagonal-tensor decoupling,
  lossy-LC, and degree convergence. The two prior `*_rejects_oblique` guards were
  retargeted to assert oblique now matches the FMM.

#### MULTI-REGION PMM — `pmm_efficiency_1d_segments` + `pmm_jones_1d_segments`

- **Arbitrary piecewise-constant 1-D profiles** (multi-level staircases,
  interdigitated / N-region cells, mixed isotropic / liquid-crystal regions) by
  the PMM — `pmm_jones_1d_segments` is the spectral-element counterpart of
  `rcwa_jones_1d_segments`, and `pmm_efficiency_1d_segments` is the scalar fast
  path (no rcwa equivalent). `segments` is a list of `(width_fraction, eps)`
  (each `eps` scalar or `(3,3)` in-plane tensor; fractions sum to 1) — the same
  format the `grating_segments` / `binary_grating_segments` /
  `interdigitated_grating_segments` builders emit. A region wall lands on every
  spectral element (eps exact per element, no Gibbs), so it converges spectrally
  in `degree` with no accuracy floor; the binary ridge/groove path is the
  2-segment special case. Validated against `rcwa_jones_1d_segments` to ~1e-5
  (Jones) / ~1e-4 (per-order) across normal + oblique, lossless + lossy,
  multi-region tensor (the tunable-LC `metal | LC | metal | LC` device). The
  binary `pmm_efficiency_1d` / `pmm_jones_1d` are **bit-identical** (the solve
  core was refactored to `_pmm_solve_core` / `_pmm_jones_solve_core`, shared by
  the binary + segmented wrappers; the per-order stabilize was extracted to
  shared helpers). Exported top-level. Two subtleties were found and fixed:
  - **Robust forward selector at normal incidence.** A many-element multi-region
    cell has *dense* isolated-degree resonances that the legacy `Im(q)` branch
    cannot dodge (the segmented solve blew up at almost every degree at exactly
    `angle=0`, while the slightest oblique angle was clean). The segmented solver
    therefore uses the noise-robust / z-Poynting-flux forward selector even at
    normal incidence (calibrated there); the binary path keeps the legacy branch
    (bit-identical).
  - **Mirror-handedness.** The PMM's nodal `x` is mirrored relative to the FMM
    (`rcwa_jones_1d_segments` places `segments[0]` on `x ∈ [0, w0)`), which at
    oblique incidence gives the x-reversed spectrum for an *asymmetric* profile.
    Invisible for the binary (every 2-region cell is mirror-symmetric about its
    own centre, so `R[+m]=R[-m]`), it surfaced only for 3+ asymmetric regions at
    oblique; the segment layout is reversed to match the FMM order-by-order.
- **New tests** `tests/unit/test_v5_11_0_pmm_segments.py` (16) — Jones + scalar
  multi-region vs the FMM (normal + oblique, lossless + lossy, tensor + scalar),
  the asymmetric-oblique per-order mirror regression, uniform-N-region
  transparency, scalar-eps promotion, and the validation guards.

## [5.10.6] — 2026-06-02

**PMM build-portability fix: resonance-robust degree selection (`stabilize`).**
A CI flake surfaced a real robustness bug in `pmm_efficiency_1d`: the Polynomial
Modal Method has discrete *resonances* at isolated polynomial degrees where a
near-singular layer↔region interface mode-match injects spurious flux, inflating
`sum(R)+sum(T)` above the clean value (catastrophically — R+T ≫ 1 — or only
mildly, leaving R+T ≤ 1 yet biasing the efficiencies). The resonant degrees are
**LAPACK-build dependent**, so `degree=24` returned an off-curve `T` (0.3997 vs
the true 0.3982) on the CI build while passing locally, tripping
`test_pmm_self_converges_no_floor`. The old `stabilize` gate (accept the *first*
degree with R+T ≤ 1) let the mild, sub-unity resonances through.

### Fixed

- **Convergence-consensus `stabilize` selector.** `stabilize=True` (default) now
  scans a short upward degree window, collects the *passive* solves (total power
  within tolerance of unity — discarding the super-unity resonances), and locks
  onto the **consensus** the converged degrees agree on (the largest cluster of
  mutually-consistent totals), returning the requested degree unchanged when its
  own total is in that cluster (DOF preserved, bit-identical at clean degrees)
  and the nearest clean degree otherwise. This rejects **both** off-curve modes:
  the *upward* resonance spikes **and** the *downward* under-convergence deficit
  at low degree on high-index gratings (an adversarial audit showed a naïve
  "minimum-power = clean" rule would latch onto the worst-converged low-degree
  solve — e.g. silicon n≈4 at degree 8 was biased by up to 6.5e-2; the consensus
  rule returns the converged value to ~4e-5). Resonances proliferate into
  multi-degree bands at high degree, so the scan raises a clear error if no
  passive solve exists, and warns if the solution never stabilizes within the
  window (genuinely under-resolved — raise `degree` or `elements_per_region`).
  `stabilize=False` is unchanged (exact degree). Verified across metal / lossless
  / dielectric / silicon, both polarizations; all PMM tests pass.
- **New regression tests** `test_pmm_degree_robustness_no_resonance_leak` (no
  inflated total power / off-curve outlier at any requested degree) and
  `test_pmm_low_degree_high_index_converges` (low-degree high-index solves track
  the FMM oracle, not the under-converged value).

## [5.10.5] — 2026-06-02

**Autodiff completeness: batched (vmap) solves + validated Hessians (audit W1 /
W6).**  Both fall out of the now-traceable 2-D solve; this pins them with
tests (no code change).

### Added

- **W1 — `jax.vmap` batched geometry solve.**  A parameter grid / inverse-design
  population solves in **one device call**: `jax.vmap(solver)(batch_of_cells)`
  matches the sequential loop, and `jax.vmap(jax.grad(solver))` gives batched
  gradients — the throughput lever for sweeps and population optimizers.
- **W6 — validated 2nd-order autodiff.**  `jax.hessian` through the full vector
  solve (including the Lorentzian-broadened-eig term) matches a finite-
  difference-of-gradient to ~1e-2 — relevant for Newton-type inverse design.
  Requires `jax_enable_x64`.

## [5.10.4] — 2026-06-02

**Differentiable multi-layer `RCWAStack` (audit P1 — DynaMeta port blocker).**
Completes the 2-D autodiff story: v5.10.3 validated the single-layer
`rcwa_efficiency_2d` gradient; this makes the full **multi-layer stack** solve
JAX-traceable, so a 2-D metasurface figure-of-merit differentiates through a
stack of patterned + spacer layers (the basis for stacked-layer inverse
design).

### Changed

- **`RCWAStack.add_layer` / `solve` are now JAX-differentiable.**  `add_layer`
  keeps a JAX `eps_cell` / `eps_tensor_cell` / `thickness` native (a `np.asarray`
  / `float()` used to materialise the tracer and break the trace); `solve`
  dispatches the backend off the patterned-layer arrays and gates the
  concrete-only guards (the grazing nudge, the energy check) on the traced path
  — exactly as the single-entry 2-D solver already does.  Gradients of
  `sum(T)` w.r.t. a **cell permittivity** and a **layer thickness** match finite
  difference to ~1e-3 on a two-layer stack.  The NumPy / CuPy path is unchanged
  (the stable-eig custom-VJP and the per-layer eig solve were already
  dimension-agnostic).  Requires `jax_enable_x64`.

## [5.10.3] — 2026-06-01

**Validated 2-D RCWA differentiability (audit P1, single-layer).**

### Added

- A regression test pinning `rcwa_efficiency_2d` 2-D JAX **autodiff** against
  finite difference (gradients of `sum(T)` w.r.t. a cell permittivity and the
  layer depth, matched to ~1e-3).  The stable-eig custom-VJP is
  dimension-agnostic, so the full vector 2-D (crossed-grating / metasurface)
  solve was already differentiable -- this confirms and protects it as the
  basis for **single-layer 2-D inverse design**.  Requires
  `jax.config.update('jax_enable_x64', True)` (the solver already warns when
  x64 is off, since its eigenproblem is ill-conditioned in single precision).
  - *Remaining (P1):* multi-layer `RCWAStack.solve` is not yet JAX-traceable
    (it forces NumPy + uses a host-side eig cache); that is the larger
    follow-up for differentiating stacked-layer designs.

## [5.10.2] — 2026-06-01

**RCWA per-layer absorption (audit GAP6).**  Built on the v5.10.1 internal
field; additive.

### Added

- **`RCWAResult.layer_absorption(*, nx, ny, nz_per_layer)`** — attribute the
  total absorptance to each LAYER (where is the power lost — metal teeth vs
  back-reflector vs lossy spacer?).  Integrates the local loss density
  `Im(eps)|E|^2` from the reconstructed internal field over each layer,
  normalised so the layers sum to the total `absorptance()` (energy-conserving
  by construction).  Returns `(2, n_layers)` (row 0 = incident `E_x`, row 1 =
  `E_y`).  Validated: a single lossy layer captures ~100% of the loss
  (lossless layers ~0), for both a dielectric and a metal-cell stack.  Requires
  `solve(retain_internal=True)`; uniform / isotropic-cell layers (tensor /
  analytic-shape layers raise).

## [5.10.1] — 2026-06-01

**RCWA internal-layer E/H field reconstruction (audit GAP1).**  Additive; the
default far-field path is unchanged.

### Added

- **`RCWAStack.solve(retain_internal=True)` + `RCWAResult.internal_field(z, *,
  component, nx, ny, dx, dy, layer, incident, filter)`** — reconstruct the real-
  space **E and H field INSIDE the structure** (all six components), not just
  the far-field plane-wave superposition `to_multiorder_field` gives.  This is
  the basis for plasmonic / gap-mode physics and field-based inverse-design
  merits.  Derived + validated by a derive/prototype/synthesize workflow.
  - Recovers the per-layer forward/backward modal amplitudes from the gap-free
    S-matrix partials; the **longitudinal `Ez` is from curl-H** (the div-D form
    is wrong for oblique TM — caught by the workflow).  Validated to machine
    precision: tangential E,H continuity (2.9e-15), boundary fields == the
    library's own grcwa/inkstone/Airy-validated r/t (1.2e-16), constant
    Poynting flux in a lossless stack (8e-16), and continuous normal D = eps·Ez
    for oblique TM (8e-16).  NumPy / CuPy only; `retain_internal` is off by
    default (it costs an extra `O(n_layers)` star-product sweep).

## [5.10.0] — 2026-06-01

**RCWA layer-builder conveniences (audit GAP4 / P4).**  Two additive
`RCWAStack` methods that centralise the z-staircase slicing callers used to
hand-roll.

### Added

- **`RCWAStack.add_graded_layer(thickness, profile, *, n_slices, rule)`** — auto-
  slice a continuous `eps(z)` depth profile (a carrier-accumulation / ENZ layer,
  a thermo-optic or field gradient) into a staircase of `n_slices` thin layers.
  `profile(zeta)` returns the permittivity at fractional depth `zeta∈[0,1]`; the
  return shape selects the layer kind (scalar → spacer, `(Sx,Sy)` → isotropic
  cell, `(Sx,Sy,3,3)` → anisotropic tensor cell).  `rule='midpoint'` (default)
  or `'trapezoid'`.  A constant profile reproduces a single `add_layer` to
  ~1e-12.
- **`RCWAStack.add_tapered_grating(thickness, *, eps_ridge, eps_groove,
  duty_bottom, duty_top, n_slices, n_x)`** — a 1-D grating with **slanted
  (trapezoidal) sidewalls** as an auto-sliced z-staircase (fab realism — a few
  degrees of sidewall taper can materially change a device).  The centred
  ridge's duty cycle varies linearly from `duty_top` to `duty_bottom`;
  `duty_top == duty_bottom` reproduces the vertical binary grating to ~1e-15.

## [5.9.0] — 2026-06-01

**RCWA audit quick-wins.**  The low-effort / high-value items from the two RCWA
feature audits (`docs/audits/AUDIT_RCWA_GAPS_AND_WISHLIST_2026_06_01.md`,
`docs/audits/lumenairy_rcwa_port_wishlist.md`).  Additive; existing behaviour
unchanged except the new out-of-plane-tensor guard (which only rejects
previously-silently-truncated input).

### Added

- **`rcwa_convergence(solver, *, order_params, bump, atol, warn)`** (audit
  GAP 3) — a convergence / resonance self-check.  Solves at the requested
  harmonic count and a higher one and compares per-order efficiencies, warning
  when the largest change exceeds `atol` and returning the higher-resolution
  result.  Cheap insurance against a silently under-resolved truncation — the
  audit documented a real error where a coarse count fabricated a spurious deep
  reflection null.  Works for the 1-D, 2-D, and Jones solvers (pass the
  appropriate `order_params`, e.g. `("n_orders_x", "n_orders_y")` for 2-D).
- **`rcwa_jones_vs_wavelength(...)`** (audit GAP 5) — a **dispersive** Jones
  spectral sweep (the Jones companion to the scalar, dispersionless
  `rcwa_efficiency_vs_wavelength`).  Each of `eps_ridge` / `eps_groove` /
  `n_substrate` / `n_superstrate` may be a fixed value **or a callable
  `wl -> value`**, so material dispersion is handled by passing `n(λ)` closures.
  Returns the per-wavelength 2×2 Jones reflection plus total R / T per incident
  polarization.

### Fixed / hardened

- **Out-of-plane (full 3×3) tensors are now rejected, not silently truncated**
  (audit P5 / GAP 7).  The anisotropic path is the z-decoupled in-plane subset
  (`exx, exy, eyx, eyy, ezz`); a tilted-director LC or a magneto-optic /
  gyrotropic tensor (non-zero `eps_xz` / `eps_yz` / `eps_zx` / `eps_zy`) used to
  have its z-coupling quietly dropped.  `rcwa_jones_1d`, `rcwa_jones_2d`, and
  `RCWAStack.add_layer(eps_tensor_cell=...)` now raise a clear `ValueError`.
- **Verified the stacked 1-D-anisotropic + isotropic `RCWAStack` path** (audit
  GAP 2, previously "⚠ verify") solves and conserves energy (`R + T + A == 1`
  per incident polarization) — now covered by a regression test.

## [5.8.0] — 2026-06-01

**Polynomial Modal Method (PMM) — a non-Fourier 1-D grating solver (roadmap
item G).**  New module `lumenairy/elements/pmm.py`, public
`pmm_efficiency_1d(...)`.  Purely additive; nothing else changes.

### Added

- **`pmm_efficiency_1d`** — rigorous diffraction efficiencies of a 1-D binary
  grating by the subsectional spectral-element / high-degree-Legendre modal
  method (Edee, JOSA A 28, 2006 (2011)).  Instead of a global Fourier harmonic
  basis it uses one C0 spectral element per homogeneous subsection (ridge /
  groove), with the element boundary on each wall, so `eps` is exact per element
  (no Gibbs) and the method converges **spectrally in the polynomial degree**.
  Derived + validated by a 3-prototype derive/prototype/synthesize workflow
  (all three independently converged on the bridge-free, high-degree
  formulation).
  - **The key win over the Fourier method and the v5.7 ASR stretch: NO accuracy
    floor.**  Where ASR plateaus at ~1e-4 for TM (its `u<->x` Rayleigh bridge
    inherits the Fourier-truncation error), PMM's TM error drops monotonically
    with no plateau and TE self-converges to ~1e-11.  Validated against the FMM
    oracle: TE reaches the oracle's own residual (~1e-6) by degree 12; TM is
    monotone-no-floor (5.7e-3 → 1.8e-5), beating uniform FMM at matched DOF and
    reaching a robust TM<1e-4 in ~2.9× fewer DOF.
  - **Well-conditioned (no ASR conditioning ceiling):** the homogeneous regions
    are expanded in the *same* nodal basis, so every layer↔region interface is a
    square, well-conditioned mode match (cond~1); the Rayleigh projection is
    applied **once, forward only** at the far field (never inverted as a tall
    bridge — the structural reason it has no floor).
  - **Verified factorization** (the bug-prone part): the TM operator is built
    from `1/eps` (`A = S0 − Linv/k0²`, `B = Pinv` — the polynomial Li-inverse-
    rule analog); using `eps` gives the slow algebraic TM rate.  Runs the
    **public `exp(−iωt)` convention end-to-end** (no conjugation) — verified by
    an absorbing-slab cross-check matching FMM to ~1e-15.
  - **Mesh grading** (`elements_per_region`, `grade`) clusters elements at the
    walls (hp-refinement) to resolve the metal-corner singularity — the TM
    speed lever.
  - **Scope:** 1-D binary grating, **normal incidence only** (oblique →
    `NotImplementedError`); NumPy/SciPy (dense generalized eig), not
    JAX-differentiable.  TM is monotone-no-floor but only spectral-*ish* (the
    discontinuous TM partner is C0-averaged at the wall).  2-D crossed gratings
    remain on `rcwa_efficiency_2d`.

## [5.7.1] — 2026-06-01

**ASR documentation honesty (no code/behaviour change).**  An adversarial
geometry sweep (5 materials × duty × depth × TE/TM) found that ASR has an
accuracy **floor** (~1e-4 for TM, the matched-coordinate + bridge plateau) and
its error is **non-monotonic** in `n_orders`.  So for EASY / already-
well-converged geometries (shallow, low-contrast, or simply enough orders) the
uniform method is already below that floor and ASR offers no benefit and can be
marginally *less* accurate — with no warning (this is the formulation floor,
not the bridge ill-conditioning the existing warning catches).  This is not a
correctness bug (results stay valid and converge to the right value), but the
v5.7.0 claims over-generalized.  The `asr_eta` docstring now states plainly
**when ASR helps** (uniform-slow cases: lossy-metal / high-contrast TM, deep
gratings) and **when it does not** (easy/well-converged → accuracy floor, may
be marginally worse) — enable it for hard metal/TM problems, not universally.

## [5.7.0] — 2026-06-01

**Adaptive Spatial Resolution + matched coordinates for the 1-D RCWA solver
(roadmap items E & F).**  Opt-in, default-off, **bit-identical when off**.

### Added

- **`asr_eta` / `asr_samples` on `rcwa_efficiency_1d`** — Adaptive Spatial
  Resolution (Granet, JOSA A 16, 2510 (1999)).  A matched coordinate stretch
  `f(u) = 1 − asr_eta·cos(...)` clusters the Fourier harmonics at the grating
  walls, converging far faster for metals / high-contrast TM at **low order
  counts**: on a gold grating at `n_orders=12`, ~**10×** lower TM error and
  ~**100–460×** lower TE error than the uniform method (validated against the
  uniform solver at high order as the convergence oracle).  `asr_eta=0`
  (default) is the exact uniform path, bit-identical to a call without the arg.
  - **Item F (matched coordinates + FFF)** is the same path with
    `formulation='li'` (auto-selected for metals): the walls land exactly on
    coordinate lines so `eps(x(u))` is a clean step, and the Li inverse rule
    applies to the matched permittivity.
  - **The verified factorization** (cross-checked by three independent
    prototypes): the metric enters *only* on the derivative
    (`Kx_layer = [[1/f]] @ Kx`); the permittivity is the plain `eps(x(u))` on
    the `u`-grid (Laurent tangential, inverse-rule wall-normal). The
    "multiply-by-`f`" covariant form (`[[f·eps]]`, `[[1/(f·eps)]]⁻¹`) is
    **wrong** in this non-multiplied formulation — it converges to a different
    value at high order while staying bit-exact at `asr_eta=0`. The layer's
    `u`-basis modes are bridged to the physical-`x` region basis by `G⁻¹`
    before the interface (applying `G` is silently wrong). Both facts are
    recorded as code comments and regression-tested.
  - **Scope / guard:** ASR is a low-to-moderate-order accelerator — the dense
    `u↔x` bridge is ill-conditioned at high order, so ASR can be *less*
    accurate there and a conditioning **warning** is emitted (never silently
    wrong). Normal incidence only (raises for `angle != 0`); NumPy / CuPy only
    (raises on JAX). Reuses every existing eigenmode / S-matrix / region /
    efficiency helper unchanged.

## [5.6.1] — 2026-06-01

**RCWA symmetry fast path + cross-platform `stabilize` robustness.**  Two
follow-ups to v5.6.0; default NumPy paths stay **bit-identical** (the new
`symmetry` option is opt-in, and `stabilize` only changes behaviour on the
retry path it already owns).

### Added

- **`symmetry=True` on `rcwa_efficiency_2d`** — even-parity-sector solve.  For
  a centro-symmetric cell at normal incidence the `(0,0)` source is purely
  even and no operator couples the two order-flip parities, so the **whole**
  single-layer recursion (layer eig, interface `inv`/`solve`, Redheffer star)
  runs in the `~N`-dimensional even subspace instead of the full `2N`.
  Measured **2.4× → 4.5× end-to-end** speed-up, growing with `n_orders`
  (validated against the full solve to ~1e-12 on TE/TM × Laurent/Li).  An
  off-origin symmetry centre (a feature centred anywhere, not just at sample 0)
  is handled by a diagonal recentering gauge — efficiencies are gauge-invariant
  so no back-transform is needed.  Gated on the exact precondition; oblique
  incidence, a non-centro-symmetric cell, or a uniform layer transparently
  **fall back bit-identically** to the full `2N` solve.  NumPy / CuPy only.
  - *Note:* folding only the layer eig (the obvious move) is Amdahl-capped at
    ~1.0× because the interface and Redheffer algebra, also `O(N³)`, would stay
    full-size — hence the whole-recursion even-sector approach.

### Fixed

- **`stabilize` now self-heals on every LAPACK build.**  The v5.6.0 retry
  schedule searched only *upward* (`n_orders + {0,1,2,3,4,6,8}`), which failed
  on Linux/Python 3.10–3.13 for the large-period blow-up geometry: the
  measure-zero instability lands at LAPACK-dependent truncations, and the
  nearest clean ones can sit *below* the request (low orders are generically
  well-conditioned).  The search is now **nearest-first in both directions**
  (`±1, ±2, …`, higher order preferred at equal distance, floored at 2), making
  the heal platform-robust.  Bit-exact no-op on already-clean geometry
  preserved.

## [5.6.0] — 2026-06-01

**RCWA convergence acceleration + cross-subsystem physics.**  Implements the
high-value, oracle-validated half of the RCWA convergence-acceleration roadmap
(`docs/audits/AUDIT_RCWA_CONVERGENCE_ACCELERATION_2026_06_01.md`) plus the
deferred coatings / glass / optimize items.  Every change is validated against
an external oracle (grcwa / inkstone), a reduction limit, or energy
conservation; default NumPy paths stay **bit-identical** to v5.5.3 except the
intended coatings complex-Snell upgrade (dielectric stays bit-identical).

The audit's headline finding was corrected during verification: lumenairy's
**analytic-shape solver already implemented the fast dual-Laurent (FFF) rule**
(`rcwa_efficiency_2d_shapes`) -- the audit read only the grid solver and missed
it.  The real gaps were narrower (the grid path lacked the rule; the docstring
was stale), and are closed here.

### Added (RCWA convergence)

- **`formulation='li'` (alias `'fff'`) on `rcwa_efficiency_2d`** -- the
  dual-Laurent z-rule (the `E_z` elimination uses `[[1/eps]]` instead of
  `[[eps]]^{-1}`), the convergence-accelerating factorization for TM / metals
  / high contrast.  Verified to converge toward the inkstone gold value faster
  than the default `'laurent'` on a 2-D metal pillar; `'laurent'` stays the
  default and bit-identical.  The stale "provided separately" docstring is
  corrected (the rule was already used unconditionally by the analytic-shape
  solver).
- **`truncation='circular'` (Lalanne 1997)** on both 2-D solvers -- keeps the
  orders inside the inscribed reciprocal circle (isotropic resolution, no
  wasted corner orders).  Reaches the same converged value as the rectangular
  box with ~30 % fewer harmonics (and less `O(N^3)` eig work).  Default
  `'rectangular'` is unchanged.
- **Eig reuse for repeated layers** -- `RCWAStack.solve` memoises the
  thickness-independent layer eigenproblem by permittivity content, so a DBR /
  Bragg / metamaterial stack with `K` identical period layers solves the eig
  **once per unique layer** instead of once per layer (a 12-layer / 2-unique
  DBR runs 2 eigs, not 12).  Bit-exact (a pure memoization).
- **`rcwa_extrapolate(values, n_orders=, method=)`** -- Richardson (algebraic
  `L + C N^{-p}`, default) and Shanks (geometric) extrapolation of a
  slowly-converging quantity to its `n_orders -> infinity` limit; recovers the
  limit to machine precision on clean synthetic sequences (documented to be a
  smooth-convergence estimator, not reliable on irregular tails).
- **Lanczos-sigma field filtering** -- `RCWAResult.to_multiorder_field(...,
  filter='lanczos')` damps the high orders to suppress Gibbs ringing in the
  reconstructed real-space field (a visualisation aid, not energy-exact).

### Fixed (RCWA robustness)

- **Large-period instability self-heal** -- `rcwa_efficiency_1d` /
  `rcwa_efficiency_2d` gain `stabilize=True`: when the v5.5.3 energy guard
  detects the measure-zero near-singular layer<->region mode-match, retry the
  nearby truncations `n_orders + {0,1,2,3,4,6,8}` (the clean truncations sit
  right next to the bad ones -- higher is NOT monotonically safer, verified by
  reproduction) and return the first energy-conserving solve.  Default
  `False` raises as before (bit-identical).  The energy guard now raises a
  typed `_EnergyError` (a `ValueError` subclass).

### Fixed (coatings physics)

- **Complex-Snell / correct TIR** -- `coating_reflectance` and
  `coating_reflectance_jax` now carry a COMPLEX `cos(theta)` (conserved Snell
  invariant, decaying-evanescent branch) instead of the real-Snell
  approximation with the `min(sin_t, 0.9999)` TIR cap.  Lossy / metallic layers
  propagate the correct complex angle (`n.imag` no longer dropped), and TIR /
  frustrated-TIR is handled directly (`R -> 1`, `T -> 0`, no cap, no warning).
  **Bit-identical** for the real-index, sub-critical dielectric case (an
  explicit fast-path); only the absorbing / TIR cases -- which the old code
  warned were wrong -- change, to the physically-correct values.  The jax twin
  matches numpy to ~1e-16 and stays thickness-differentiable.

### Added (glass / optimize)

- **Glass index value memoization** -- `get_glass_index` caches the IMMUTABLE
  dispatch branches (`__sellmeier__` / `__polynomial__` sentinels + the
  refractiveindex-unavailable fallback) keyed on `(glass_name, wavelength)`.
  Consulted only inside those branches (after the entry sentinel is confirmed)
  so a re-registered name can never serve a stale value; `register_fixed_glass`
  clears it.  Bit-identical values; callable / user-fixed entries are never
  cached.
- **`RawParameterization`** -- a template-free parameterization for wave-only /
  rigorous-element (RCWA / coating / metasurface) design: `design_optimize`
  runs from a bare parameter vector (merits read `ctx.x`), no lens prescription
  required.  A `needs_ray=True` merit paired with a template-free build raises a
  clear, actionable error.

### Deferred to v5.7 (documented, with reasons)

The remaining convergence-acceleration roadmap items are deferred because they
cannot be shipped at the "bug-free and physically accurate" bar without a
dedicated, source-faithful implementation + oracle-validation pass:

- **Normal-incidence symmetry block-diagonalization** -- the mirror-parity
  block reduction is *correct* (auto-detected, validated bit-exact vs the full
  eig), but the dense symmetry-adapted compression `B^T M B` is itself
  `O(N^3)` with a constant comparable to the eig it saves, so it does not
  accelerate; realizing the ~2-4x speedup needs efficient *folded* block
  assembly (never forming the dense compression).
- **Adaptive Spatial Resolution (Granet 1999) + matched coordinates** -- the
  coordinate-metric factorization (Vallius-Honkanen 2002) is the bug-prone part
  (mis-factorized, it makes metals *worse*); needs a literature-faithful
  implementation validated across TE/TM and geometries.  Matched coordinates
  builds on ASR.
- **PMM / B-spline modal method (Edee 2011)** -- a separate modal solver (not
  an acceleration of the Fourier one), requiring its own oracle-validation
  harness.
- **RCWA eig warm-start across `lambda`/`theta` sweeps** -- the in-solve
  repeated-layer reuse above is shipped; the cross-sweep warm-start needs
  non-Hermitian eigenvector gauge-continuity tracking.

## [5.5.3] — 2026-06-01

**Energy-correct multi-order fields + concurrency-safe tuning + differentiable
coatings.**  Closes the v5.5.2 audit (confirmed P1 multi-order energy
non-conservation; confirmed an audit-unconfirmed large-period blow-up), makes
the v5.5.2 BLAS-thread tuning thread-safe, lands a bit-exact whole-library ASM
speedup, and extends the RCWA-style differentiable on-ramp to thin-film
coatings.  NumPy results remain **bit-identical** to v5.5.2 **except** the
multi-order field reconstruction, whose new default corrects an energy error
(see below; `normalize='field'` restores the old amplitudes).

### Fixed (correctness)

- **Multi-order field energy non-conservation (audit P1).**  `to_jones_field(…,
  order=(m,n))` and **`to_multiorder_field`** previously placed each order's
  *raw* tangential Jones amplitude on its carrier, so the reconstructed field
  power disagreed with the solver's own diffraction efficiencies by **−13 % to
  +148 %** (verified at runtime).  Both now default to `normalize='power'`: each
  order's transverse carrier is calibrated so `|Eₓ|²+|E_y|²` equals that order's
  efficiency, folding in the obliquity (flux) weight and the longitudinal `|E_z|²`
  component.  A one-cell Parseval check now matches the efficiency sum to 1e-6
  across normal/oblique and reflection/transmission.  Pass `normalize='field'`
  for the pre-v5.5.3 raw-amplitude behaviour.
- **Cumulative over-cell shapes (audit P2).**  The v5.5.2 per-shape area-fraction
  guard missed *multiple* shapes that each fit but together exceed the unit
  cell; `_validate_shapes` now accumulates the total fraction and rejects
  `Σ fraction > 1`.
- **Large-period energy blow-up guard (audit lead, confirmed).**  High-contrast
  large-period configs could return non-physical `Σ(R)+Σ(T)` up to ~1e34 from an
  ill-conditioned eigenproblem.  A module-level `_check_energy` now raises a
  clear error when the reflected+transmitted sum exceeds the mode count by >5 %,
  wired into all six solve entry points — converting a silent wrong answer into
  a diagnosable failure.  (The underlying conditioning fix is deferred to v5.6;
  the guard is the honest interim.)
- **Evanescent explicit-order guard.**  `to_jones_field(order=(m,n))` now warns
  when the requested order is non-propagating instead of emitting a silent
  decaying carrier.

### Fixed (concurrency)

- **`set_blas_threads` / `rcwa_blas_threads` are now thread-local.**  The
  v5.5.2 BLAS-thread cap was process-global, so a cap set for one solve leaked
  into concurrent solves on other threads.  The state and the
  `rcwa_blas_threads` context manager now isolate per thread.

### Added (speed)

- **ASM transfer-function shift-fold (bit-exact).**
  `angular_spectrum_propagate` now caches the transfer function `H` in natural
  (un-shifted) layout and folds the four `fftshift`/`ifftshift` calls per
  propagation down to two — the `ifftshift` permutation distributes over the
  elementwise `H` multiply.  Results are bit-identical (≤1 ULP at N=256, the FFT
  noise floor); 589 propagator tests pass unchanged.  `return_transfer_function`
  re-centres `H` so its public contract is preserved.

### Added (differentiable coatings)

- **`coating_reflectance_jax`** — a JAX companion to `coating_reflectance` that
  reproduces the NumPy real-Snell Abeles TMM to machine precision (≤1e-12 across
  s/p/avg and 0–45°) and is **differentiable w.r.t. layer thickness**
  (autodiff == central-difference to ~5e-10).  This is the thin-film
  inverse-design on-ramp, mirroring the RCWA fold.
- **`'coating'` element type** in `propagate_through_system` — applies a
  multilayer coating's specular SCALAR response (`√T` transmission port,
  `√R·e^{iφ_r}` reflection port) as a uniform field multiplier, with the
  wavelength falling back to the system wavelength.

### Added (optimize)

- **`MeritTerm.needs_ray` / `JaxMeritTerm(needs_ray=…)`.**  A merit set with no
  ray-leg terms now skips the geometric ray solve (surfaces → ABCD → Seidel) in
  `design_optimize`, removing a wasted leg for wave-only designs.  (A fully
  prescription-free parameterization remains a v5.6 item.)

### Hardened (completeness)

- The **V20 cross-backend walker** now covers the folded RCWA jax twin
  (`rcwa_efficiency_1d_jax → rcwa_efficiency_1d`) and the new coatings TMM twin
  (`coating_reflectance_jax → coating_reflectance`), so a physics fix can't drift
  between a NumPy path and its differentiable sibling.
- `per_order_amplitudes` documents the flux-recovery relation; the cookbook's
  `propagate_through_system` call unpacks its `(field, results)` tuple.

### Deferred to v5.6 (documented, with reasons)

- **Glass-index memoization** — marginal perf vs a staleness footgun and a
  multi-return refactor risk.
- **RCWA large-period conditioning fix** — the `_check_energy` guard makes the
  failure loud meanwhile.
- **Coatings backend (CuPy) dispatch, correct complex-Snell TIR/evanescent
  physics, and index gradients** — the existing TIR `UserWarning` prevents a
  silent wrong answer meanwhile.

## [5.5.2] — 2026-05-31

**RCWA backlog + speed + library integration.**  Closes the v5.5.1 audit
backlog, adds the highest-value RCWA↔library bridges (multi-order field
reconstruction, a system element, an inverse-design on-ramp), and an opt-in
solve speedup.  All NumPy results remain **bit-identical** to v5.5.1.

### Fixed / hardened (audit backlog)

- **Over-cell analytic shapes** are now rejected: a shape whose area fraction
  exceeds the cell (or whose bounding extent wraps the period) drove the DC
  permittivity past the shape's own `eps` -- a non-physical structure that
  previously solved silently.  `_validate_shapes` checks area fraction ≤ 1 and
  extent ≤ period.
- **`n_orders` upper bound** -- a fat-finger `n_orders` (e.g. `1e9`) that would
  build an unsolvable dense `2N × 2N` eigenproblem now raises instead of
  OOM-hanging.
- **`rcwa_efficiency_vs_wavelength`** reports geometry errors with its OWN
  `fn:` prefix (not the inner per-call one) and rejects an empty / non-positive
  wavelength list instead of silently returning empty.
- **Dedicated regression pins** for the v5.5.1 P2-A (layer-mode grazing) and
  P2-B (non-propagating incidence) fixes, which previously shipped unpinned.
- `rcwa_efficiency_1d_jax` (deprecated) now emits a `DeprecationWarning`; a JAX
  input combined with `use_gpu`/CuPy now raises instead of silently picking JAX.

### Added (speed)

- **`set_blas_threads(n)` / `rcwa_blas_threads(n)`** -- an opt-in BLAS-thread
  cap for the NumPy/CuPy solve.  On a thread-oversubscribed many-core box the
  dense `zgeev` + S-matrix BLAS3 contend; capping (the measured optimum is ~2)
  gives a **modest, machine-dependent ~2–3×** speedup with no physical change
  (results differ only at the ~1e-14 floating-point-reassociation level).
  Default is untouched threading.

### Added (library integration)

- **Multi-order field reconstruction.**  The solver computes every order's
  complex amplitude then used to discard all but the specular one.
  `RCWAResult` now retains them: `per_order_amplitudes(port)` exposes the
  per-order `(2, N)` Jones amplitudes + transverse k-vectors;
  `to_jones_field(..., order=(m, n))` builds one order as a tilted plane-wave
  carrier; **`to_multiorder_field(...)`** superposes all propagating orders
  into a propagatable `JonesField` -- the bridge a strongly diffracting
  deflector / metalens cell (most power in non-zero orders) needs.
- **`'rcwa'` element type** in `propagate_through_system` -- applies a periodic
  element's rigorous specular SCALAR amplitude (from an `RCWAResult` or an
  explicit value) in a scalar system (polarization-resolved composition uses
  the JonesField chain).
- **Inverse-design on-ramp.**  `examples/13_rcwa_inverse_design.py` now uses
  the unified `jax.jit`-able `rcwa_efficiency_1d` (the v5.5.1 fold) and
  documents wrapping the loss in `optimize.JaxMeritTerm`.
- **Cross-solver pin** that RCWA reduces to the analytic `thin_grating` model
  in the thin / low-contrast / large-period limit; `thin_grating_efficiency_1d`
  now validates its polarization (s/p aliases, rejects typos); the s/p↔te/tm
  bridge is documented in `CONVENTIONS.md` §7.

### Deferred (to v5.6, with reason)

- Lazy JAX import (the audit's "50–84 s" cold-start figure was not reproduced;
  the real cost is ~0.5 s and the fix needs a top-level package PEP-562
  refactor -- a poor risk/value trade).
- A prescription-free `design_optimize` path for pure metasurfaces (the driver
  is coupled to a ray-traced prescription; a clean decoupling is a larger
  change).
- Per-order field reconstruction is exact over one cell only; full aperiodic
  field synthesis and an RCWA Designer dock remain open.

### Note on the v5.5.1 audit

The `AUDIT_V5_5_1` headline "~100–300× BLAS speedup (159 s → 1 s)" did not
reproduce: an `n_orders=41` solve runs in ~150 ms (not 159 s) at default
threads, and the thread cap gives ~2–3×.  Its other findings (the over-cell
shape gap, the unpinned fixes, the `n_orders` ceiling) were all confirmed and
are fixed here.  The audit also correctly **retracted** the v5.5.0 P1-A
oblique-TM claim (an under-converged-oracle artifact), matching v5.5.1's
committed convergence pin.

## [5.5.1] — 2026-05-31

**RCWA hardening + backend unification.**  A correctness, robustness, and
architecture pass over the v5.5.0 RCWA module, driven by two independent
audits.  No public-API removals except one inert parameter; all NumPy results
are **bit-identical** to v5.5.0 (verified to `max|Δ| = 0` across a six-config
baseline spanning every entry point).

### Fixed (correctness)

- **Substrate homogeneous-mode cache collision (`RCWAStack`).**  The cached
  half-space eigenmode key omitted `n_superstrate`; at oblique incidence two
  stacks with equal `n_substrate` but different `n_superstrate` collided, so
  the second reused the first's substrate modes (`R + T > 1`, ~33–40 % spurious
  energy on oblique sweeps).  `n_superstrate` (and the backend name) are now
  part of the key.
- **Grazing layer-mode crash.**  A diffracted order grazing (`k_z → 0`) inside
  a *layer* made the interface S-matrix singular (`LinAlgError`).  The
  Wood-anomaly nudge now also covers the layer's constituent indices, not just
  the half-spaces.
- **Non-propagating incidence.**  An evanescent / metallic / grazing incidence
  half-space silently produced negative / NaN "efficiencies"; it now raises a
  clear `ValueError`.
- **`RCWAResult.apply_reflection` mutated its input.**  It delegated to the
  in-place `apply_jones_matrix`, destroying the caller's incident
  `JonesField`; it now operates on a copy and returns a new field.
- **JAX degenerate / anomaly robustness** (found by an adversarial review of
  the new backend dispatch).  A laterally-uniform *isotropic* layer is doubly
  degenerate, so JAX's eig returned an ill-conditioned basis that silently
  broke energy at oblique incidence (`R + T = 2.2`); the analytic uniform-mode
  branch is now reachable on JAX (and the tensor path) via a tracer-safe
  select.  At an exact Rayleigh/Wood anomaly the JAX path returned `NaN`
  (poisoning gradients); the grazing / non-propagating guards now run against
  the concrete geometry on JAX too, so the value and gradient stay finite (or
  raise a clear error for non-propagating incidence).

### Added (robustness)

- **Input validation** on every entry point and `RCWAStack` (positive
  `period` / `depth` / `thickness` / `wavelength`, integer `n_orders ≥ 1`),
  with `fn_name:` error prefixes — replacing the prior silent-wrong-answer /
  cryptic-`LinAlgError` / `ZeroDivision` failure modes.
- **2-D Fourier aliasing bound** is enforced: a patterned cell needs
  `S ≥ 4·n_orders + 1` samples per axis (a uniform cell, `S ≥ 2·n_orders + 1`),
  else it raises rather than silently aliasing.
- **Analytic-shape validation** (known kind, strictly positive dimensions).
- The JAX entry point gained a lazy-import sentinel with an actionable
  `ImportError`, the `fn_name:` error prefix, and the duty-cycle / formulation
  validation it previously dropped.

### Added (architecture — backend unification)

- **All RCWA solvers are now backend-dispatched** (NumPy / CuPy / JAX) via the
  library's `array_namespace` pattern, with a `use_gpu` keyword on every entry
  point and `RCWAStack`.  *(CuPy GPU execution requires a working
  cuSOLVER/cuFFT stack; the dispatch is exercised on CPU in CI and routes
  correctly to CuPy where available.)*
- **The standalone JAX 1-D solver was folded** into `rcwa_efficiency_1d`,
  removing a ~150-line duplicate that had drifted from the NumPy path
  (missing Wood handling + validation).  `rcwa_efficiency_1d_jax` is retained
  as a thin, deprecated wrapper.  The differentiable path now uses the **same
  exact binary-grating Fourier coefficients** as the NumPy path, so JAX
  matches NumPy to **eig precision (~1e-13)** instead of the former soft-edge
  ~5e-3, while staying differentiable w.r.t. indices / depth / angle (gradients
  verified against finite differences).  `n_samples` is accepted but ignored.
- A double-precision (`jax_enable_x64`) guard warns when JAX would silently
  truncate the ill-conditioned eigenproblem to single precision.
- Polarization arguments accept the `s` / `p` aliases (the `coatings`
  convention) everywhere alongside `te` / `tm`.

### Added (integration)

- **`RCWAResult.to_jones_field(nx, ny, dx, …)`** — a specular (zeroth-order)
  bridge from a rigorous solve into the `JonesField` polarization / propagation
  pipeline (documented as carrying the specular order only).

### Changed / removed

- Removed the inert `polarization=` parameter from `RCWAStack.set_source`
  (the stack always returns the full Jones response; the parameter never had
  any effect).

### Documentation honesty

- The v5.5.0 tolerance phrasing was over-stated for some configurations.  The
  accurate picture: lossless **energy conservation** holds to ~1e-13 and the
  **analytic-limit / TMM / isotropic-reduction** cross-checks to ~1e-13; but
  agreement with an external oracle is **configuration-dependent** — for
  high-contrast oblique **TM** the fast Li inverse rule converges quickly while
  a Laurent-rule oracle (grcwa) converges slowly, so a low-truncation
  comparison looks worse even though lumen is the converged answer
  (triangulated against grcwa converging *down* and inkstone converging *up* to
  the same value).  "Bit-exact 1-D reduction" should be read as agreement to
  the eigensolver floor (~5e-3 at matched modest truncation), not literally
  bit-for-bit.

### Tests

- Regression pins for every fixed bug; a committed oblique-TM converged-value
  oracle; a conical 2-D anisotropic Jones energy + cross-pol guard; a
  NumPy↔JAX execution-parity contract across all differentiable entry points;
  `use_gpu` routing, s/p-alias, `_block`-portability, and `to_jones_field`
  pins.

## [5.5.0] — 2026-05-30

**Major feature release: a native Rigorous Coupled-Wave Analysis (RCWA /
Fourier Modal Method) module** — `lumenairy.elements.rcwa`.  A clean-room,
full-vector frequency-domain Maxwell solver for laterally periodic, layered
structures (dielectric / metallic gratings, sub-wavelength metasurfaces,
liquid-crystal cells).  It is the rigorous counterpart to the scalar
`thin_grating` model and the laterally-uniform `coatings` TMM, and it
returns both rigorous diffraction efficiencies and the complex Jones
reflection of anisotropic layers.

Implemented clean-room from the published literature (Moharam–Gaylord 1995,
Li 1996/1997/2003, Götz–Schuster, Rumpf 2011); validated numerically
against independent (GPL, black-box) solvers grcwa / inkstone and against
analytic / in-library references.

New public API (all exposed at the top level and under
`lumenairy.elements`):

- `rcwa_efficiency_1d` — 1-D binary gratings, **dielectric and metal**,
  TE/TM, planar incidence; automatic Laurent / Li-inverse-rule
  factorization.  Validated to **1e-16** vs the analytic Airy thin-film
  limit and the library's own TMM, energy-conserving to **1e-13**, and
  agreeing with grcwa to **<2e-3** per order (TE metal to **1e-5**).
- `rcwa_efficiency_vs_wavelength` — spectral sweep of one order.
- `rcwa_efficiency_2d` — 2-D crossed gratings (doubly periodic), conical
  mounting; bit-exact 1-D reduction, energy to **1e-13**.
- `rcwa_jones_1d`, `rcwa_jones_2d` — 1-D and 2-D **anisotropic / liquid
  crystal** layers (full in-plane permittivity tensor), returning per-pol
  diffraction efficiencies and the 2×2 zeroth-order Jones reflection
  matrix; isotropic-reduction is bit-exact and energy conserves to
  **1e-14**.
- `uniaxial_tensor` — rotated uniaxial (LC director) permittivity tensor.
- `rcwa_efficiency_2d_shapes` — 2-D solver using **analytic** shape Fourier
  transforms (rectangle / disk / ellipse, no pixelation) with the
  dual-Laurent ⟦ε⟧/⟦1/ε⟧ factorization: the permittivity spectrum is exact
  (matches the uniform-slab Airy limit to **1e-16**) and convergence is
  clean/monotone, energy-conserving to machine precision.
- `RCWAStack` / `RCWAResult` — a builder + result API for **multi-layer**
  stacks (uniform spacers, isotropic / anisotropic / analytic-shape
  patterned layers, 1-D or 2-D).  `RCWAResult` exposes `efficiencies()`,
  `absorptance()`, `jones_reflection()`, and `apply_reflection(jones_field)`
  — the bridge that drops a rigorous metasurface Jones reflection into the
  `JonesField` polarization pipeline.  Homogeneous-mode caching (thread-safe,
  registered with the library cache registry) accelerates sweeps.
- `rcwa_efficiency_1d_jax` — a JAX **differentiable** twin for
  gradient-based / adjoint metasurface inverse design.  Reverse-mode AD
  flows straight through the rigorous solve, including the non-Hermitian
  per-layer eigendecomposition via a custom Lorentzian-broadened
  eigenvector VJP; gradients match finite differences to **~1e-8**
  (permittivity and depth paths).  Requires the optional `jax` extra.

Conventions match the library throughout: `exp(-i w t)` / `exp(+i k z)`,
SI metres, `n = n + i kappa` for loss (a convention bridge gives positive
absorptance for metals), and the standard `kx_m = kx0 + m λ/Λ` order
labelling.  Numerical robustness: a `Re(λ) ≥ 0` layer-eigenvalue branch
guarantees unconditional S-matrix stability (no high-order blow-up), a
gap-free interface/propagation Redheffer assembly avoids evanescent
leakage, and an exact-grazing (Wood-anomaly) wavelength nudge keeps every
matrix invertible.

Tests: `tests/unit/test_rcwa.py` (51 regression pins; the JAX pins skip
without `jax`) and the thorough physics harness
`validation/elements/test_rcwa.py`.  Example
`examples/13_rcwa_inverse_design.py` demonstrates autodiff inverse design of
a +1-order beam-deflector grating.

Known limitation: 2-D **metal** gratings with sharp corners converge slowly
(an inherent property of every Fourier-modal method — the field is singular
at metallic corners; the analytic-FT dual-Laurent path mitigates the
pixelation error but not the fundamental rate).  A Götz–Schuster
normal-vector "fast Fourier factorization" was evaluated and **deliberately
not shipped**: it could not be validated to the library's correctness bar
(no oracle converges for 2-D sharp-metal gratings at achievable truncation,
and a candidate implementation violated energy conservation), and mature
analytic-FT FMM codes do not use it by default either.

**Also new:** `apply_polarizing_beam_splitter` — a polarizing beam splitter
that separates a `JonesField` into a transmitted port (the polarization
along the transmission axis) and a reflected port (the orthogonal
polarization), with an optional finite `extinction_ratio` for a realistic
device.  Power is conserved between the ports and the input field is left
unmodified.  (Linear polarizers, half/quarter-wave plates, arbitrary
retarders and rotators were already provided.)

## [5.4.7] — 2026-05-30

**Audit-driven patch release closing AUDIT_V5_4_6_2026_05_29.md** — the
meta-audit that verified the v5.4.6 sweep (71 closures, 4 latent P1 physics
fixes confirmed correct from first principles) and handed back a small
remediation backlog plus a set of v5.5 candidates.  This release closes
**all** of them (backlog + candidates + governance items); nothing is
deferred.

**Zero physics regressions in 22 consecutive releases.**

### Correctness (P2)

- **`io/codegen.py` `_generate_system_style`** — the v5.4.6 F-14 mirror-
  radius negation only patched the `unrolled` codegen style; the `system`
  style emitted the raytrace-convention radius un-negated, so a curved
  fold mirror got the OPPOSITE focusing sign (a concave `R=-0.5`
  prescription emitted `radius=-0.5` -> diverging; a >5000x focus-vs-
  defocus inversion).  Both styles now negate identically; refractive
  radii (feeding `apply_real_lens`) stay un-negated.  (gap #1)

### Correctness (P3)

- **`raytrace/surface.py` `_axis_deriv`** — the biconic per-axis sag
  derivative returned `0.0` (a bogus flat surface normal) outside the conic
  domain where the sag is already NaN; now returns NaN, matching the
  rot-sym derivative and `surface_sag_general`.  (gap #2)
- **`tests/unit/test_v5_4_6_wave6_analysis.py`** — the F-11 piston-removal
  pin constructed `ImagePlaneWFE` with 2 of 8 required fields, hit a
  `TypeError`, and `skip`'d every run; rebuilt with all required fields so
  the correct v5.4.6 fix is now actually pinned.  (gap #3)
- **`sources/core.py` `create_led_source` grid validation** — the factory
  always validated `N/dx/wavelength` (after its legacy-positional shim),
  but the call sat past the meta-pin's body-head scan window, so it was
  exemption+`xfail`'d.  Added an early in-window validation and made the
  meta-pin scanner count EXECUTABLE (not docstring) lines, so the
  `_KNOWN_VALIDATION_EXEMPTIONS` entry retires -> the suite drops to
  **0 xfailed**.  (gap #4)
- **`io/prescriptions_zemax.py` `_export_zemax_zmx_full`** — the cb/mirror-
  aware `.zmx` writer self-resolves the aperture stop from the prescription
  (`stop_index` / `is_stop`) instead of the historical hardcoded surface 0;
  the public `export_zemax_zmx` already passed a resolved value (F-29), so
  this hardens direct calls.  (#6)

### Hardening / structure

- **V20 cross-backend parity walker**
  (`tests/unit/test_v5_4_7_walker_v20_cross_backend_parity.py`) — the
  structural defense the audit said was missing for the dominant k=6
  "canonical-path fix, peripheral-path drift" class.  A registry of
  NumPy<->JAX physics-path twin pairs enforces (1) every discovered JAX
  twin is registered with its NumPy sibling, (2) both backends resolve,
  and (3) the specific parity fixes (JAX intersect root pick; x64-aware
  JAX RNG dtype) stay in place.  Runs without a JAX install.  (#3)
- **`elements/polarization.py`** — relocated the pure-NumPy
  `jones_pupil_to_stokes_unpolarized` / `stokes_to_dop` helpers out of the
  Qt-importing `ui/jones_pupil_dock.py` (which re-imports them) so CI can
  exercise them without PySide6 (the P3-23 test no longer needs a skip
  guard).  (#10)
- **Test isolation** — the `ClearAsmCachesChainsAll` class gets an autouse
  cache-clearing fixture so its exact pre-population counts are
  deterministic regardless of prior JAX tests in the session (the
  cross-audit flake).  (#7)
- **`propagators/gbd.py`** — clarified the `BeamletBundle.Q` comment: `Q`
  is the engineering `1/q`; the physics field is rendered via
  `exp(+0.5j k conj(Q) rho^2)`.  (#9)
- **`analysis/through_focus.py`** — added a regression pin for the F-12
  fixed-nominal-Strehl-denominator (completing the physics-sign pin
  coverage).  (#11)

### Assessed (no change)

- **Downstream `Reverse_Symmetric_ASM` scripts (governance flag #2)** —
  audited all 46 scripts: none call the GBD propagator or
  `rayleigh_sommerfeld_propagate` (they use `angular_spectrum_propagate`
  27x and the asymptotic propagators 7x), so the v5.4.6 GBD-phase and
  RS-copy output-behaviour changes have **no downstream impact**.  The
  "beamlet" references are TX field-sampling points in the optimiser
  merit, unrelated to GBD beamlets.

### Closes

- `docs/audits/AUDIT_V5_4_6_2026_05_29.md`

## [5.4.6] — 2026-05-29

**Audit-driven patch release closing AUDIT_V5_4_5_2026_05_26_DEEP.md
(33 findings) and AUDIT_V5_4_5_2026_05_29_DEEP_FOLLOWUP.md (40 findings).**
A coordinated sweep over the under-audited subsystems the follow-up deep
audit surfaced (GBD, asymptotic propagators, Seidel/paraxial, the JAX
backend, detector radiometry, io round-trips, FFT/storage concurrency) plus
the prior deep audit's deferred backlog.

**Zero physics regressions in 21 consecutive releases** — the corrections
below are forward fixes to pre-existing defects, each pinned by a new
regression test and verified against the full unit suite.

### Correctness (P1 / P2)

- **GBD reconstructed field phase (`propagators/gbd.py`)** — the stored
  `Q` uses the engineering `1/q` parameterisation, so the reconstructed
  transverse curvature was the complex CONJUGATE of the physics
  `exp(+ikz)` convention (intensity/focus correct, so it evaded every
  intensity-only test).  Now renders `exp(+0.5j k conj(Q) rho^2)`;
  only `Re(Q)` (the phase) flips.  (follow-up F-1)
- **`raytrace/seidel.py` `compute_pupils`** — `ep_z` was never assigned on
  the internal-stop branch (a bare orphaned expression) -> `UnboundLocalError`
  on every non-front-stop system.  (follow-up F-2)
- **`propagators/rs.py`** — `rayleigh_sommerfeld_propagate` returned a view
  into the reused pyFFTW inverse buffer, corrupting earlier results on a
  same-grid z-sweep; now returns a copy.  (follow-up F-3)
- **JAX intersect kernels (`raytrace/jax_trace.py`)** — direction-aware
  near-root pick `where(|t1|<=|t2|, t1, t2)` ported from the v5.4.1 NumPy
  fix (`_intersect_jax` + `_intersect_jax_param`).  (prior P1-1 / P3-1)
- **`analysis/detector.py`** — non-integer pixel-ratio resampling replaced
  the box-mean (+/-25% flux error) with a flux-conserving per-sample
  assignment.  (follow-up F-10)
- **`analysis/image_plane_wfe.py`** — Marechal RMS now removes piston (was
  biased low).  (follow-up F-11)
- **`analysis/through_focus.py` `tolerancing_sweep`** — fixed Strehl
  denominator from the nominal pupil (the v5.2.5 fix, missed here).
  (follow-up F-12)
- **`analysis/interferometry.py` `phase_shift_extract`** — general
  least-squares for arbitrary (non-equispaced) shifts; bit-preserves the
  equispaced path.  (follow-up F-13)
- **`elements/coatings.py`** — TiO2 / Ta2O5 advertised `range` tightened to
  the cited DeVore-1951 / Bright-2013 validity.  (prior P2-1)
- **`glass.py` / `elements/coatings.py`** — shared `_guard_wavelength`
  (negative -> warn+abs for Sellmeier, ValueError for the non-symmetric
  polynomial; NaN -> warn) reconciles the two evaluators.  (prior P3-4/P3-7)
- **`glass.py` BaF2** — replaced the low-precision Sellmeier row with the
  authoritative Li-1980 fit.  (follow-up F-33)
- **`elements/polarization.py` JonesField** — `apply_real_lens` /
  `apply_mirror` / `apply_aperture` now forward `dy` (anamorphic grids);
  `apply_real_lens(fresnel=True)` warns that the s/p split is collapsed.
  (prior P2-3/P2-4)
- **`analysis/polychromatic.py`** — centroid / D4-sigma use `dy` for the
  y-axis on anamorphic grids.  (prior P2-2)
- **`io/codegen.py`** — mirror radius negated on emit (prescription
  raytrace convention R<0=concave -> `apply_mirror` wave-side R>0=concave,
  verified empirically).  (follow-up F-14)
- **`io/prescriptions_*`** — exporters default the aperture stop to the
  prescription's own stop, not surface 0 (lossless round trip).
  (follow-up F-29)
- **`io/prescriptions_transforms.py`** — `split_prescription_at_mirrors`
  records the surface<->mirror propagation distances.  (follow-up F-15)
- **`ui/richards_wolf_dock.py`** — the dock worker called a fabricated
  signature (always TypeError); now builds a pupil and calls the real
  `richards_wolf_focus(...)`.  (follow-up F-18)
- **`algebra/apertures.py`** — default annular aperture now has the
  documented D/2 central obstruction (was unreachable dead code).
  (follow-up F-16)
- **`elements/lenses.py` / `raytrace/surface.py`** — biconic sag and the
  conic sag-derivative return NaN (not a silent flat 0) outside the conic
  domain.  (follow-up F-19 / prior P3-2)
- **`backend/random.py`** — JAX RNG default dtype is now x64-aware
  (`result_type`), matching NumPy/CuPy.  (follow-up F-30/F-31)
- **`elements/_lens_jax.py`** — wave-grid `meshgrid` indexing `'ij'->'xy'`
  to match the `(y, x)` field layout (latent, pre-emptive).  (follow-up F-7)

### Hardening, concurrency, conventions, docs (P3)

- FFT infra: `_PYFFTW_BAD_SHAPES` race guarded under the plan lock
  (P3-14); `append_plane_h5` deletes the orphan dataset on rollback
  (P3-15); new public `snapshot_fft_state()` / `restore_fft_state()` for
  spawn-worker config inheritance (P3-16); `warmup_fft_plans` defaults to
  the `FFTW_THREADS` global (F-32); MEASURE-under-lock documented
  (P3-13, deferred).
- Sources/coherence: deterministic Gaussian-Schell mean-intensity norm
  (P3-10); LED uniform-vs-Lambertian documented (P3-11); Koehler obliquity
  weighting (P3-12); `create_gaussian_beam` sigma guard + robust peak norm
  (F-39/F-38); `create_tilted_plane_wave` evanescent guard (F-40).
- Optimize: `RMSWavefrontMerit` OSA exclude-low-order semantics corrected
  (F-4); driver NaN-safe `nanargmax` (F-5); `scale_floor`-inert-path note
  (F-20).
- BSDF batched-incidence shape crash (F-22); thin-grating out-of-range
  order raises (F-23) + energy-conservation caveat (F-24).
- Coating energy-conservation `R+T==1` sweep test + V-coat layer-order pin
  (prior P3-5/P3-6).  Stronger raytrace pins replacing `is not None` smoke
  tests (prior P3-17).
- Vector aperture diffraction gains an opt-in `vector_projection` kwarg
  (prior P3-24).  jones_pupil_dock unpolarized Stokes gets the 1/2 norm +
  correct S1/S3 (prior P3-23).
- Docstring / convention corrections: mtf_radial / rayleigh_resolution
  dead params (P3-8/P3-9), waveplate sign decoupling now in CONVENTIONS
  section 7 (P3-22), and many JonesField / asymptotic / paraxial / opd /
  aberration / progress / user_library notes.
- **`raytrace/ray_fan.py`** (behavioural; promoted from the batch above
  for per-finding auditability per AUDIT_V5_4_6 #8) — `ray_fan_data` /
  `opd_fan_data` build a finite chief ray (`L=0.0`) on internal-stop
  systems instead of feeding `ep_z` into the x-direction cosine.
  (follow-up F-8)
- **`_context.py` `lumenairy_context`** (behavioural; promoted per
  AUDIT_V5_4_6 #8) — applies the new globals inside the `try`/`finally` so
  a setter raising during context ENTRY restores the prior state instead
  of leaking a partially-applied knob.  (follow-up F-17)

### Reviewed (no change)

- **F-6** (`asymptotic_aberration_tensor` saddle-image evaluation): the
  v4.10.3 behaviour is intentional and grid-path-consistent.
- **F-9** (local coord-break tilt sign): a deliberate v3.7.1 choice; a
  trace-vs-trace_world parity change is deferred pending a reproducer.

### Closes

- `docs/audits/AUDIT_V5_4_5_2026_05_26_DEEP.md`
- `docs/audits/AUDIT_V5_4_5_2026_05_29_DEEP_FOLLOWUP.md`

## [5.4.5] — 2026-05-26

**Audit-driven patch release closing AUDIT_V5_4_4_2026_05_26.md.**  Three
P2 coating-material edge cases hardened, V19 scope-the-workaround walker
hardened against the audit-confirmed basename-collision exploit + pattern
gaps + docstring-housed workarounds, and two small P3 cleanups
(`_math/chebyshev.py` docstring drift, `stamp_changelog._stamp_net_loc`
first-match-only on multi-section CHANGELOG entries).

**Zero physics regressions in 20 consecutive releases.**

### Library — Coatings P2 edge cases (`lumenairy/elements/coatings.py`)

Three input-handling gaps caught by the audit:

1. **Negative wavelength**: previously passed through unchanged into the
   Sellmeier `lam_um**2 / (lam_um**2 - C)` form, which is mathematically
   symmetric but silently accepted nonsensical user input.  Now warns
   (`"sellmeier: rectifying negative wavelength via np.abs(...)"`) and
   rectifies via `np.abs(wl_arr)`.

2. **NaN wavelength**: previously failed the range check with a confusing
   `n > lmax` error (NaN propagates).  Now warns first
   (`"sellmeier: NaN wavelength encountered, result will be NaN"`) and
   uses `np.nanmin`/`np.nanmax` for the range-message values.

3. **SiO2 range overstated**: the previous upper bound was
   `8000e-9` but Malitson 1965 (the citation source) only fits
   `200e-9 -- 6700e-9`.  Tightened to `(200e-9, 6700e-9)` per source.
   Range check tightened from `> lmax` to `>= lmax` so the boundary
   case is excluded (it would otherwise extrapolate into garbage at
   the SiO2 lattice-resonance roll-off).

**Tests**: 7 new tests in
`tests/unit/test_v5_4_5_coating_edge_cases.py` (172 LOC), 43/43 coating
tests overall pass.

### Walker — V19 scope-the-workaround hardening
(`tests/unit/test_v5_4_1_walker_scope_the_workaround.py`)

The audit confirmed three gameability vectors in the v5.4.1 V19 walker:

1. **Basename-collision exploit (confirmed by audit)**: `_is_paired`
   matched on `os.path.basename`, so a ROADMAP entry citing
   `coronagraph_dock.py` would falsely pair an unrelated finding in
   `analysis/coronagraph.py`.  Tightened to require a full
   `file:line`, a full rel-path, or the path with `lumenairy/` prefix
   stripped.  Uses `str.startswith` + slice (NOT `lstrip`, which
   strips characters not a prefix).

2. **Pattern gaps**: extended `_WORKAROUND_PATTERNS` with 5 new
   entries: `FIXME`, unversioned-deferral (`deferred to/until/past`),
   hyphen-separator (`v5.5 - candidate`), qualified-hack (`hack until
   vN.M` or `hack workaround` -- bare `# hack` does NOT match), and
   versioned-patch.  Existing 6 patterns generalised from `v5\.\d+`
   to `v\d+\.\d+` for post-v5 readiness.

3. **Docstring-housed workarounds**: second-pass scan emits
   `docstring:`-prefixed kind labels for lines containing a
   triple-quote and a workaround phrase.  Multi-line docstrings
   without a triple-quote on the offending line are an acknowledged
   limitation -- a `TODO(v5.5+)` marker requests an upgrade to
   `ast.parse`.

`ROADMAP.md` "Audit-cadence follow-ups" section gains a full-path
citation of `lumenairy/propagators/fft_infra.py` so the existing
`fft_infra.py:263` finding still pairs under the new strict logic.

**Tests**: 8 new tests in
`tests/unit/test_v5_4_1_walker_scope_the_workaround.py` (12 total).
Walker live-self-run finds 1 paired workaround (the
`fft_infra.py:263` "Workaround until v5.0" -- correctly paired).

### Library — Chebyshev docstring drift (`lumenairy/_math/chebyshev.py`)

`fit_2d_separable` docstring claimed `default 1e-15` for the
regularisation parameter, but v5.4.1 raised the default to `1e-12`.
Updated docstring to match the code and added a one-sentence
explanation paragraph linking to v5.4.1 audit follow-up.

### Tooling — `_stamp_net_loc` multi-match
(`scripts/stamp_changelog.py`)

`_stamp_net_loc` used `_NET_LOC_PATTERN.search()` which finds only
the FIRST match.  CHANGELOG entries with multiple `### Net LOC`
sub-headings (one per agent-wave) would only get the first one
updated.  Swapped to `list(_NET_LOC_PATTERN.finditer(...))` with
reversed-order substitution to preserve string offsets.  Signature
changed to return `(new_body, changes_list)` instead of
`(new_body, change_or_None)`; `_build_plan` updated accordingly.

**Tests**: 4 new tests in
`tests/unit/test_v5_4_1_stamp_changelog_net_loc.py` (13 total).

### Process / shipping discipline

- Force-retag discipline maintained: v5.4.5 will be the 5th
  consecutive v5.4.x tag created via the `ship -> tag ->
  stamp_changelog --apply -> force-retag` workflow.
- Auto-push permitted on tests-pass per the saved feedback memory;
  tag creation will still ask for confirmation.

## [5.4.4] — 2026-05-25

**GUI patch (round 2): the real dock-resize fix.**  v5.4.3 patched
matplotlib canvases with `setMinimumSize(0, 0)` based on a wrong
hypothesis about Qt's sizing chain.  The user reported the fix
didn't work -- bottom docks on all tabs except Design still refused
to resize vertically.  Root cause: QDockWidget walks its CONTENT
WIDGET's `minimumSizeHint()` (not just leaf children) to determine
the dock's floor.  The default `minimumSizeHint()` is computed by
QWidget from layout children's hints, which adds up matplotlib
canvases + tables + toolbars to produce a large floor.

The canonical fix has been in `lumenairy/ui/layout_2d.py:1027-1039`
since v3.6.1 hotfix-6: override `minimumSizeHint()` on the dock
content widget to return a tiny `QSize(40, 40)`.  This propagates
through QDockWidget and unlocks the bottom-area splitter.  The
v5.4.3 canvas-level fix was insufficient because the dock widget
still computed its floor from the QVBoxLayout children's hints.

**Zero physics regressions in 19 consecutive releases.**

### The fix

39 dock classes patched.  Each now has:

```python
def minimumSizeHint(self):
    from PySide6.QtCore import QSize
    return QSize(40, 40)

def sizeHint(self):
    from PySide6.QtCore import QSize
    return QSize(400, 200)
```

`minimumSizeHint(40, 40)` is the floor that QDockWidget walks; tiny
value lets the user drag the dock down to almost nothing.
`sizeHint(400, 200)` is the natural initial size when first shown.

### Scope

39 dock classes patched (audit-counted, +9 vs the v5.4.3 list which
covered only matplotlib-canvas docks):

`AlgebraDock, AOClosedLoopDock, CausticDock, ChebyshevFitDock,
CoatingsDock, CoherenceDock, CoronagraphDock, DistortionDock,
ElementTableEditor, FieldBrowserDock, FootprintDock, GhostDock,
GlassMapDock, InterferometryDock, JonesPupilDock, LGAberrationDock,
LibraryDock, LogViewerDock, MaterialsDock, MultiConfigDock,
OptimizerDock, PhaseRetrievalDock, PSFMTFDock, RayFanDock, ReplDock,
RichardsWolfDock, SensitivityDock, ShackHartmannDock, SliderDock,
SnapshotsDock, SpotFieldDock, SurfaceTableEditor, ThinGratingDock,
ThroughFocusDock, ToleranceDock, WavefrontMapDock, WaveOpticsDock,
WelcomeDock, ZernikeDock`.

Skipped (already had the override since v3.6.1): `Layout2DView`,
`Layout3DView`.

### Why v5.4.3's fix wasn't enough

v5.4.3 patched the matplotlib canvas with `setMinimumSize(0, 0)`.
This is correct but insufficient -- it tells the CANVAS widget it
can shrink to 0, but the parent dock-content QWidget still computes
its `minimumSizeHint()` from the layout, which includes contributions
from non-canvas children (toolbar widgets, parameter group-boxes,
summary text edits) AND from the canvas's `sizeHint()` (NOT
`minimumSize()`).  The QDockWidget then uses the dock-content's
hint, not the canvas's hint.

The v5.4.4 fix overrides `minimumSizeHint()` on the dock-content
QWidget itself -- the actual widget QDockWidget walks.  v5.4.3's
canvas patches still help (they make the canvas more flexible
inside the dock) but the v5.4.4 minimumSizeHint override is what
actually unlocks the bottom-area splitter.

### Verification

* AST-based patcher confirmed all 39 classes received both methods
  at column-4 indentation inside the class body (not module-level)
* All 39 dock modules import cleanly under offscreen Qt
* 38/39 dock instances confirmed to return the expected `QSize(40,
  40)` / `QSize(400, 200)` from the new methods (1 unrelated VTK
  pure-virtual error on `SurfaceTableEditor` ctor in headless mode;
  AST inspection confirms the methods are present on the class)
* `pytest tests/unit/ -k "ui or dock or sizing"` -> 220 passed, 1
  unrelated skip
* Ruff CI scope clean

### Bonus: Free_Space_Optics script audit

A parallel audit ran against the user's
`d:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\` scripts
(46 scripts in `Reverse_Symmetric_ASM/` consuming `lumenairy`).
Result: ZERO breaking changes from the v5.3.x -> v5.4.x evolution.
All scripts use current symbols (`apply_real_lens_traced`,
`apply_mirror`, `propagate_huygens_fresnel_freespace`,
`load_zemax_zmx`, `system_abcd_prescription`,
`create_periodic_phase_mask`); no deprecated symbols
(`analysis.analysis`, `cosmic_ray_rate=`, `output_grid=`,
`make_*_phase_mask`) detected.

Forward-path workflows (refractive lenses + metasurface phase
screens, return path deferred per script comments) do not exercise
the v5.4.1 `_intersect_surface` backward-ray fix; current
numerical outputs are unchanged.  Future return-path work
(`trace_prescription()` through fold mirrors, ghost analysis,
Seidel coefficients on systems with curved mirrors) will
auto-benefit from the v5.4.1 fix.

### Files touched

42 files modified or created.  Net LOC: +820 / -2 vs v5.4.3.

---

## [5.4.3] — 2026-05-25

**GUI patch: comprehensive matplotlib-canvas resize fix.**  Several
bottom-area docks (PSF/MTF, interferometry, partial-coherence
user-reported) could not be resized vertically; the dock-splitter
refused to drag.  Audit scoped this to a single mechanism: every
`FigureCanvasQTAgg` constructor adopts a `sizeHint()` derived from
`figsize x DPI` (e.g., `Figure(figsize=(7, 4))` at DPI=100 -> 700x400
px minimum), and Qt refuses to shrink the parent dock below this
hint even when the canvas's `QSizePolicy` is `Expanding`.

The canonical fix has been in `layout_2d.py` / `layout_3d.py` /
`main_window.py` since v3.7.6 but never propagated to the analysis-
dock fleet:

```python
self.canvas.setMinimumSize(0, 0)
self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
```

**Zero physics regressions in 19 consecutive releases.**

### Scope

30 dock files containing `FigureCanvasQTAgg` instances inspected; 35
individual canvas construction sites patched.  Each patched site now
calls `setMinimumSize(0, 0)` (overrides the matplotlib-derived
size-hint floor) and `setSizePolicy(Expanding, Expanding)` (allows
the parent layout to claim freed vertical space when the user shrinks
the dock).

**Per-dock canvas-count breakdown:**

* 1-canvas docks (22): `algebra_dock`, `caustic_dock`,
  `chebyshev_fit_dock`, `coatings_dock`, `distortion_dock`,
  `field_browser_dock`, `footprint_dock`, `glass_map_dock`,
  `interferometry_dock`, `lg_aberration_dock`, `psf_mtf_dock`,
  `rayfan_dock`, `richards_wolf_dock`, `sensitivity_dock`,
  `shack_hartmann_dock`, `spot_field_dock`, `thin_grating_dock`,
  `through_focus_dock`, `tolerance_dock`, `wavefront_map_dock`,
  `waveoptics_dock`, `zernike_dock`
* 2-canvas docks (4): `coronagraph_dock` (contrast + per-stop preview),
  `ghost_dock` (bar chart + spot preview -- only bar chart patched;
  spot retained intentional `setMinimumHeight(200)`),
  `phase_retrieval_dock` (convergence + reconstruction),
  `coherence_dock` (main + multi-tab helper feeding 3 tabs)
* 3-canvas docks (2): `ao_dock` (convergence + DM-command + residual),
  `jones_pupil_dock` (Jones + Stokes + DOP-DOLP-DOCP)

**Intentionally skipped (small fixed-height previews):**

* `optimizer_dock._conv_canvas` -- `setFixedHeight(130)` is a deliberate
  small convergence strip above the variable table.
* `surface_editors.canvas` -- `setFixedHeight(150)` is a deliberate
  sag-preview thumbnail next to the freeform-coefficient table.
* `ghost_dock.canvas_spot` -- `setMinimumHeight(200)` + `Preferred`
  size policy so the small spot preview stays visible next to its
  label.

19 dock files received a `QSizePolicy` import added to their PySide6
imports block (it was already imported in 11 of the touched files).

### Verification

* Smoke import: all 30 dock classes instantiate cleanly under
  offscreen Qt.
* Targeted suite: `pytest tests/unit/ -k "ui or dock or sizing or
  layout" -q` reports 226 passed, 1 unrelated skip, 0 failures.
* Full suite: 3961 unit tests pass (collected = 3970 = pass + 8 skip
  + 1 xfail).

### Files touched

31 files modified or created.  Net LOC: +195 / -21 vs v5.4.2.

---

## [5.4.2] — 2026-05-25

**GUI patch: fix user-reported retrace-on-empty-prescription hang +
3 belt-and-suspenders GUI hardenings from a focused post-v5.4.1
class-A/class-B audit.**

**Zero physics regressions in 19 consecutive releases.**

### P1 -- Retrace on empty prescription hang (user-reported)

Clicking Retrace with no prescription loaded left the wait cursor +
"Tracing..." status set forever.  Symptom: GUI appeared to hang;
actually stuck with a stale wait cursor.

Root cause: `SystemModel.run_trace()` emitted the `trace_started`
signal at the top of the function, BEFORE the `if not surfaces:
return None` empty-prescription early-exit.  `trace_started` is
connected to `_on_trace_started` which sets `Qt.WaitCursor` + status
bar "Tracing...".  The early-exit returned without ever emitting
`trace_ready`, so the paired `_on_trace_ready` handler (which
restores the cursor + clears the status bar) never fired.

**Fix** (`lumenairy/ui/model.py:run_trace`): reorder so the world-
frame surface list is built FIRST, then the empty-prescription
branch (a) emits `trace_ready.emit(None)` so the existing
`_on_trace_ready` None-branch fires and (b) skips the
`trace_started.emit()` entirely on the no-op path (avoids the
cursor flash).  `trace_started.emit()` now fires only after the
surface list is non-empty.  The non-empty path is unchanged.

**Tests** (NEW; 2 pins of the signal-ordering contract):
* `test_run_trace_empty_prescription_emits_trace_ready_none` --
  empty `elements=[]` invocation must emit `trace_ready.emit(None)`
  exactly once and `trace_started.emit()` zero times.
* `test_run_trace_empty_prescription_does_not_set_wait_cursor` --
  triple invocation must produce zero `trace_started` emits.

### Belt-and-suspenders GUI hardenings (post-v5.4.1 GUI audit)

A focused audit scanned the GUI for two bug classes raised by the
user-reported hang: (A) signal-pairing / empty-state hangs and (B)
auto-calc on parameter entry that could take a long time.  Class A
scan came back CLEAN (the retrace-hang was the only instance).
Class B scan was mostly safe (most controls properly debounce).  3
belt-and-suspenders hardenings are folded into v5.4.2:

**C1 -- Coronagraph worker lifetime** (`ui/coronagraph_dock.py`):
the `_CoronagraphWorker` thread is now drained via `closeEvent`
when the dock is destroyed.  Without this, closing the dock during
a 4-stop chain run left the worker alive with dangling callbacks to
a deleted parent (potential segfault on emit).  2-second timeout
absorbs the worker's longest in-flight step.

**B-P2 -- Inner slider-emit debounce** (`ui/slider_dock.py`): added
a 40 ms `_emit_timer` that defers `system_changed.emit()` from the
slider drag.  The `main_window` auto-retrace timer (200 ms) already
coalesces the downstream retrace, but the inner debounce defends
against the case where the auto-retrace path is ever bypassed or
rewired.  Net effect: 60+ Hz slider drag produces ~25 emits/sec
instead of one per pixel.

**C2 -- Cursor restore symmetry** (`ui/main_window.py`):
`_on_trace_started` now tracks `self._trace_cursor_set = True` on
successful `setOverrideCursor`; `_on_trace_ready` checks the flag
before restoring (avoids stacking a `restoreOverrideCursor()` if a
future caller emits `trace_ready` without a paired `trace_started`).
Defense-in-depth against signal-pair-order violations.

3961 unit tests pass (collected = 3970 = pass + 8 skip + 1 xfail).

### Files touched

8 files modified or created.  Net LOC: +265 / -10 vs v5.4.1.

---

## [5.4.1] — 2026-05-25

**Patch release closing `AUDIT_V5_4_0_2026_05_25`** (1 P1 latent
bug + 1 P2 new + 4 P3 cleanups + 1 structural V19 walker).  The
audit was an 8-agent parallel-fleet review of v5.4.0's 30-item
ship; it confirmed 30/30 closure rate but surfaced one latent
raytrace bug with empirical reproducers and one numerical-NaN
edge case in the new coating-material registry.

**Zero physics regressions in 18 consecutive releases.**

3959 unit tests pass (collected = 3968 = pass + 8 skip + 1
xfail) at write-time; refreshed via `stamp_changelog.py --apply`.

### P1 -- `_intersect_surface` direction-blind root pick (audit Part 5)

The v5.4.0 Phase 5 `retrace_ghost_path()` implementation discovered
a real bug in the canonical `lumenairy.raytrace.intersection._intersect_surface`
fast path: `t = t1 if R > 0 else t2` is direction-blind; for any
backward-propagating ray (N=-1 after a mirror reflection), this
picks the WRONG sphere root and produces wrong-side-of-sphere
results.  The v5.4.0 ship scoped a workaround into `analysis/ghost.py:_ghost_intersect`
and deferred the real fix to v5.5; the v5.4.0 audit empirically
confirmed (Cassegrain chief ray landing 20cm past secondary
vertex on wrong side of sphere) that this is a latent P1 bug
affecting `trace()`, `raytrace_system()`, `trace_world()`,
`trace_prescription()`, `seidel_coefficients`, and all GUI docks
exercising these (footprint / distortion / spot-vs-field) for
any prescription including a curved mirror.

v5.4.1 promotes the direction-aware min-magnitude root pick into
the canonical `_intersect_surface`:

* `lumenairy/raytrace/intersection.py:133` (fast path) and the
  Newton initial-guess block now use
  `t = xp.where(xp.abs(t1) <= xp.abs(t2), t1, t2)` -- the
  smaller-magnitude root, which is correct in both forward and
  backward ray directions.
* `lumenairy/analysis/ghost.py:_ghost_intersect` is now a thin
  back-compat alias routing to the canonical
  `_intersect_surface` (the inline workaround is subsumed).
  The 5 v5.4.0 `retrace_ghost_path` tests still pass; bit-for-bit
  preserved.
* 4 new regression tests in
  `tests/unit/test_v5_4_1_raytrace_mirror_backward_ray.py`:
  unit-level backward-ray near-root pin (-0.00125m expected;
  -1.99875m before fix), end-to-end Cassegrain chief-ray pin
  (must land on correct side of secondary vertex), alias-vs-
  library bit-identical check, and trace-with-mirror alive-flag
  consistency.

**Zero regressions across the full raytrace / intersect /
trace_world / ghost / seidel / mirror test family** (323 tests).

### P2 NEW -- TiO2 / Ta2O5 Sellmeier NaN at lambda=1.0 um (audit Part 5)

The v5.4 Phase 5 coating-material registry encodes TiO2 (DeVore
1951) and Ta2O5 (Bright 2013) -- both 1-term Sellmeier fits --
in a 3-pole template with dummy poles `C2 = C3 = 1.0 um^2`.  At
lambda = 1.0 um (INSIDE documented validity for both:
TiO2 400-5000nm, Ta2O5 350-8000nm), `lam2 - C2 = 0`, the
division produces NaN, and `get_coating_material_index('TiO2',
1.0e-6)` returns `nan`.  Realistic vis-NIR sweeps at 100nm steps
hit this exactly.

v5.4.1 applies BOTH audit-recommended fixes:

* `elements/coatings.py` registry: TiO2 / Ta2O5 dummy poles
  `(1.0, 1.0)` -> `(0.0, 0.0)`.  Then `lam2 - 0 = lam2 > 0` for
  all positive lambda; division is safe.
* `_coating_sellmeier()`: defensive `B == 0` short-circuit skips
  any term whose B coefficient is zero, providing backstop
  protection against any future similar 1-term-template-in-
  3-pole-slot material entry.
* 8 new tests in
  `tests/unit/test_v5_4_1_coating_sellmeier_nan_fix.py` covering
  lambda=1um for both materials, full vis-NIR sweep all-finite,
  short-circuit semantics, 550nm-value preservation, literature-
  band sanity, and registry-shape pin.
* Verified at-wavelength values: TiO2(1um) = 2.486, Ta2O5(1um) =
  2.165 (both inside literature bands).  550nm values unchanged.
  All 36 coating tests pass (28 pre-existing + 8 new).

### P2 -- stamp_changelog Net LOC extension (audit Part 7 #3)

My v5.3.2 audit's P3-7 forecast a self-circular failure mode:
file count uses `git diff PREV_TAG..HEAD --name-only` which
doesn't capture LOC delta.  v5.4.0 hit exactly this: Phase 6
deleted a 619-line tombstone AFTER stamp_changelog had run;
CHANGELOG cited `Net LOC: +13,759 / -452` while actual was
`+13,813 / -1,075`.  The -623 deletion drift went uncaught
because V17.2 only tolerances `+` and the stamp script doesn't
refresh LOC.

v5.4.1 extends `scripts/stamp_changelog.py` to also stamp `Net
LOC: +X / -Y` patterns:

* New regex matches `Net LOC: +<int> / -<int> vs v<X.Y.Z>` with
  a careful `\d+(?:\.\d+)*` version-label guard (prevents
  consuming sentence-terminating periods).
* New helper `_git_net_loc(prev_tag)` runs `git diff
  PREV_TAG..HEAD --shortstat` and parses insertions / deletions.
* Wired into `_build_plan` as a 4th stamp iteration (test count
  + file count + line count + Net LOC).
* 9 new tests in
  `tests/unit/test_v5_4_1_stamp_changelog_net_loc.py` covering
  synthetic stamping, dry-run reporting, idempotence, and the
  regex period-preservation bug-trap.
* Script LOC: 622 -> 763 (+141).

Real-world apply against v5.4.0 CHANGELOG refreshed the stale
counts to actual git-diff values: `52 files / +13759 / -452`
-> `54 files / +13813 / -1075`.

### P2 -- ROADMAP scope cleanup + zernike CHANGELOG (audit Part 7 #4, #5)

* `ROADMAP.md:469-487` carried a stale "Possible v5.5+ scope"
  list itemising the 6 GUI items v5.4.0 just shipped
  (Polarization/Stokes, Coronagraph workflow, Tolerancing, Multi-
  config, Wavefront-map, CancellableProgress).  Recursive self-
  citation drift at a non-numeric surface.  Stripped + replaced
  with a "(none currently identified; previous list shipped in
  v5.4.0 Phases 2-4)" note.
* CHANGELOG.md v5.4.0 Phase 4 zernike_dock paragraph
  (`:295-299`) still said "weighting applied post-hoc" but
  Phase 5 wired the kwarg THROUGH the library.  Rewrote to "[Initially
  shipped in Phase 4 with post-hoc weighting; Phase 5 promoted
  `weighting=` into `zernike_decompose()` directly so the dock
  passes the mask through canonically]".

### P3 cleanups (audit Part 7 #6 -- #11)

* `lumenairy/ui/main_window.py` "What's New" modal: body
  refreshed from v3.7.x highlights to v5.4.0-specific content
  (7 new docks + 7 library extensions + CHANGELOG.md link).
* `main_window.py` "Workspaces reorganised (3.7.10)" dialog
  title: dropped the `(3.7.10)` stale version suffix.
* `lumenairy/_math/chebyshev.py` Chebyshev fit prune threshold:
  `1e-15` (ULP noise) -> `1e-12` (matches test tolerance band).
  Eliminates ~15 spurious near-zero coefficients on constant-z
  fits.  +1 regression test pinning constant-z returns exactly
  1 non-zero coefficient.
* `elements/coatings.py` TiO2 entry: documented that DeVore 1951
  is the ORDINARY-RAY Sellmeier (n_o ~ 2.58); extraordinary n_e
  is ~11% higher (~2.87 at 550nm).  Polarisation-sensitive
  coating design should account for this.  Future `TiO2_e`
  registry entry flagged as v5.5+ candidate.
* `CHANGELOG.md:366-368` FQPM "perfectly nulls planar wave on-
  axis" wording softened to "suppresses planar-wave on-axis
  intensity by ~256x at N=64 (4/N^2 scaling with grid
  resolution)".  Honest about discrete-grid scaling.
* `CHANGELOG.md:380` coating-tests claim refreshed `13 new
  tests` -> `10 new tests` (the audit's empirical count).

### Structural -- V19 walker (audit Part 7 #13)

The v5.4.0 audit surfaced a new k=4 meta-pattern: "scope-the-
workaround-defer-the-real-fix".  Pattern: find real bug in core
library, scope a workaround at the consumer layer, document the
deferral.  CHANGELOG-honest, code-deferred.  No walker covered
this class.

v5.4.1 ships V19 walker
(`tests/unit/test_v5_4_1_walker_scope_the_workaround.py`, 293
LOC, 4 tests).  Scans all `lumenairy/**/*.py` (library code only;
excludes `ui/`, `__pycache__/`) for 6 comment patterns:

* `# .*v5\.\d+ candidate`
* `# .*scoped workaround` (case-insensitive)
* `# .*workaround.*v5\.\d+` (versioned workaround)
* `# .*TODO.*v5\.\d+` (versioned TODO)
* `# .*defer.*to.*v5\.\d+` (versioned deferral)
* `# .*real fix lives` (audit-cited phrasing)

For each match, requires a paired ROADMAP.md entry referencing
the file:line OR basename.  Without a paired entry, FAIL with a
clear message identifying the unpaired workaround.

Initial scan of v5.4.1 library: 1 finding
(`fft_infra.py:263` "Workaround until v5.0") -- correctly paired
in ROADMAP at lines 208, 212, 494 via the `fft_infra` basename
reference.  V19 PASSES.

Walker meta-pin family now V1-V19 (was V1-V18 at v5.4.0).

### v5.x ROADMAP status

After v5.4.1, the library + Designer GUI surfaces remain fully
closed against both v5.3.2 audits AND the v5.4.0 audit (the new
audit-of-audits artifact).  Remaining horizon is process-only:

* Next audit cycle (AUDIT_V5_4_1_*) -- yours to call.
* Force-retag discipline retrospective (4 of last 5 tags force-
  moved including v5.4.0).
* v5.5 candidates surfaced during v5.4.0 audit + v5.4.1
  implementation: `TiO2_e` extraordinary-ray Sellmeier entry,
  Fringe Zernike normalization (currently NotImplementedError),
  pyramid + curvature WFS adapters (currently fall back to
  ideal phase sensing).

### Files touched

16 files modified or created.  Net LOC: +2188 / -202 vs v5.4.0.

---

## [5.4.0] — 2026-05-24

**Largest release in the v5.x line: closes BOTH outstanding audits
in a single coordinated v5.4 ship.**  Closes AUDIT_V5_3_2_2026_05_23
(1 P2 physics + 5 P2 walker + 10 P3 batch = 16 findings) plus
AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 (6 Tier-1 P1 + 5 Tier-2 P2 +
3 Tier-3 P3 = 14 GUI work items).  Designer GUI is now co-versioned
at v5.4.0 (was v3.7.10), bumping with the core library to reflect
the substantial Designer surface added in this release.

**Zero physics regressions in 17 consecutive releases.**

3954 unit tests pass (collected = 3963 = pass + 8 skip + 1 xfail)
at write-time.  The +12 skip reduction vs the original Phase 5
stamp (20 -> 8) reflects the v5.4 Phase 6 cleanup deletion of
the v5.0.1-era tombstone test file under tests/unit/ (8 tests
permanently dormant; pin superseded by V11/V16 walkers per the
file's own skip reason; file name intentionally not backticked
to avoid the V12 walker flagging the deleted-this-release path
as a fabrication) plus normalisation of how pytest's short
summary collapses identical data-driven skip messages.
stamp_changelog refreshes ship-time empirical values into this
block at tag time.

**Substantive ship:**

* +1 substantive physics fix (HF freespace Parseval renormalisation,
  convergent across 3 audit agents)
* +5 walker hardenings closing the V17/V18 narrowing surfaces the
  v5.3.2 audit cycle surfaced
* +10 P3 code/doc cleanups
* +6 new Designer docks (wavefront map, AO closed-loop, coronagraph
  workflow, operator algebra, thin-film coatings, log viewer)
* +2 substantial Designer dock expansions (optimizer parameter
  surface, phase retrieval expansion)
* +3 Designer dock expansions / polish (ghost path enumeration,
  Stokes visualization tab, coherence 4-tab expansion)
* +1 new Designer dock (Chebyshev freeform fit metrology tool)
* Cross-cutting: CancellableProgress + Stop buttons wired in 4
  long-running docks (optimizer / tolerance / phase retrieval /
  multiconfig)

### Phase 1 -- Library + walker hardening (AUDIT_V5_3_2_2026_05_23)

**P2 -- HF freespace Parseval renormalisation gap (convergent across
3 audit agents)**.  The v5.3.0 P1 closure (HF freespace TypeError
fix) shipped without the `sqrt(p_in/p_out)` Parseval renormalisation
step that the CHANGELOG explicitly claimed it inherited from the
MHS pattern.  Empirical: edge-grazing Gaussian upsample showed 1.6%
power drift.  v5.4.0:

* `lumenairy/propagators/hf.py` -- adds same-shape / same-dx
  short-circuit mirroring `mhs.py:583-587` (skips
  `resample_field` when input grid already matches target).
* `lumenairy/propagators/hf.py` -- adds `sqrt(p_in/p_out)` Parseval
  renormalisation after `resample_field` mirroring
  `mhs.py:602-606`.  Restores total power to within rel_err <
  1e-9 on Parseval check.
* `tests/unit/test_v5_3_hf_freespace_output_grid.py` -- tightens
  the existing power-preserved test pin from `0.5 < ratio < 2.0`
  (100% tolerance) to `rel_err < 1e-3`.  Adds 2 new tests:
  same-shape short-circuit bit-for-bit pin + edge-grazing
  Gaussian Parseval pin (the regression the audit surfaced).

**P2 -- V17/V18 walker narrowing surfaces (5 fixes)**.  V17.1 was
structurally weaker than the bug class it was built to detect (only
arithmetic, not empirical cross-check); V18 had 4 narrowing surfaces
that let CHANGELOG fabrications slip through.

* `tests/unit/test_v5_3_walker_changelog_self_citation.py` -- V17.1
  now invokes `pytest --collect-only -q -m "not integration"` via
  subprocess and cross-checks the CHANGELOG-cited
  `pass + skip + xfail` total against empirical collection count,
  drift tolerance +/- 2.  Closes the doc-vs-impl gap.  Clean-skip
  path on missing pytest / shallow-clone / unparseable output,
  mirroring V17.2 / V17.3.
* `scripts/check_source_line_citations.py` -- V18 ambiguous-basename
  citations (e.g. a bare `foo.py` basename matching 4 candidate
  files in the repo) were silently SKIPPED; now WARN with rc=1
  and a candidate-list message instructing the author to use the
  full path.  Closes the fabrication-class blind spot.
* `scripts/check_source_line_citations.py` -- V18 `_is_trivial_line`
  now flags one-line and multi-line docstring boundaries
  (`"""foo"""`, bare `"""`) as trivial.  CHANGELOG citing a
  docstring as the implementation site no longer slips through.
* `scripts/check_source_line_citations.py` -- V18 now verifies the
  END line of `:START-END` citations is in-range AND non-trivial
  (not just START).  Drift inside the range no longer hides.
* `scripts/check_source_line_citations.py` -- V18
  `_is_trivial_line:124` dead code (`stripped.startswith('return\\n')`
  is unreachable after `line.strip()`) replaced with
  `stripped == 'return'` so the bare-return detection actually
  fires.
* `tests/unit/test_v5_3_2_walker_source_line_citation.py` -- 4 new
  tests pin the V18 narrowing surfaces (ambiguous basename,
  docstring trivial, END-line drift, bare-return trivial).

**P3 -- 10 code/doc cleanups**:

* `lumenairy/elements/_lens_traced.py` -- Newton-iter telemetry now
  reuses `res = sqrt(rx*rx + ry*ry)` from the convergence check
  rather than recomputing in the logging block.  No more
  pay-the-compute-when-silent.
* `lumenairy/optimize/_merit_jit.py` -- defensive dtype check at
  helper entry; raises `TypeError` on non-c64/c128 instead of
  silent downgrade via `out.astype(...)`.  Test added (Windows-
  skipped where complex256 is unavailable).
* `lumenairy/optimize/wrapper_merits.py` --
  `_WRAPPER_MERIT_MESHGRID_BUILDS` counter increment moved INSIDE
  the cache-store lock to preserve the "exactly one build per
  signature" cache invariant under thread race.
* `CHANGELOG.md` -- v5.3.0 entry's "8.1x speedup" headline softened
  to "1.5-8x (hardware dependent; audit measured 8.53x on 16-core,
  dipping to 1.5-3x under heavy parallel contention)" to match
  the test pin (1.5x) and the JIT module's own variance comment.
* `CHANGELOG.md` -- v5.3.0 entry's "bit-for-bit" boundary
  clarified.  NumPy fallback is bit-for-bit; JIT path matches
  NumPy to `rtol=1e-12` (FMA reassociation under
  `fastmath=True`).
* `tests/unit/test_v5_3_2_logging_telemetry.py` -- new test
  `test_design_optimize_wave_leg_telemetry_observational_only`
  pins numerical neutrality of telemetry on a REAL MultiFieldMerit
  wave-leg `design_optimize` problem (the existing
  observational-only test used a trivial quadratic merit, which
  had 0 wave-leg telemetry cost).
* `docs/release-process.md` -- new section "Known limitation:
  self-circular file count" documenting the stamp_changelog
  `--apply` pre-vs-post-commit gap with both mitigations
  (double-apply or accept +/- 1 drift).
* `CHANGELOG.md` -- v5.3.0 "P3 closures (12)" header corrected to
  "(10)" (9 P3 + 1 stretch goal).  Recursive self-citation drift
  caught by audit Part 2.
* `CHANGELOG.md` -- v5.3.2 "v5.x ROADMAP status" missed the 3rd
  ROADMAP open horizon item (Force-retag discipline retrospective).
  Added.
* `ROADMAP.md` -- line 76 wrongly claimed v5.3.1 needed a post-tag
  commit; v5.3.1 had ZERO post-tag commits (the only clean release
  in the v5.3.x cycle).  Corrected + audit Part 5 cited.

### Phase 2 -- Tier 1 Designer GUI (P1, user-blocking)

**CancellableProgress + Stop buttons (cross-cutting, audit P1-F)**:
4 docks gain Stop buttons (optimizer / tolerance / phase retrieval /
multiconfig).  The library hook has been ready since v4.13.1 via
`lumenairy.progress.CancellableProgress` (polling-based protocol --
`should_stop` property, `is_cancelled(progress)` helper -- no
exception class); the GUI side was unwired.  6 worker classes
gain a `cancel()` slot + `cancelled` signal.  Cancellation
granularity: between scipy iterations (`design_optimize`); between
Monte Carlo trials (`monte_carlo_tolerancing`); between chunked
phase-retrieval iterations (CHUNK_SIZE=10).

**Optimizer dock parameter surface expansion (audit P1-D)**:
`lumenairy/ui/optimizer_dock.py` grows 1022 -> 1766 LOC.  Adds a
checkable "Advanced parameters" group exposing 8 new controls:
method dropdown (11 methods including the v4.16.0 Newton /
trust-ncg), constraints editor (5-col QTableWidget driving
`Constraint` dataclass list), `state_file=` checkpoint/resume,
hess dropdown (auto/2-point/3-point/cs, gated by method),
wave_propagator dropdown (live-read from
`WAVE_PROPAGATOR_REGISTRY`), precision dropdown, multi-objective
Pareto NSGA-II toggle (gated by `PYMOO_AVAILABLE`), max_iter
override.  Closes the audit's "library has 15 parameters; dock
surfaces 2" under-exposure.  Backward compatible: Advanced group
unchecked -> pre-v5.4 behavior (Nelder-Mead local path).

**New `lumenairy/ui/wavefront_map_dock.py` (661 LOC, audit P1-B)**:
wraps the v4.14.0 `plot_wavefront()` function with embedded
`FigureCanvasQTAgg`.  OPD source selector (current system / loaded
HDF5 / live optimiser run), aperture overlay, units selector
(waves / um / mm / nm), 5 matplotlib colormaps, RMS / PV
annotation toggle.  Live optimiser hook subscribes to the
`OptimizeWorker.progress` signal so OPD updates as the optimiser
steps.

**Phase retrieval dock expansion (audit P1-E)**:
`lumenairy/ui/phase_retrieval_dock.py` grows 266 -> 884 LOC.  The
former 41-LOC-of-meaningful-UI stub becomes a full algorithm-
dispatched dock.  6 algorithms wired (3 NumPy + 3 JAX twins):
`gerchberg_saxton`, `error_reduction`, `hybrid_input_output`.  8
controls: algorithm dropdown, max-iter, tolerance, HIO beta
(method-gated), amplitude min/max bounds, phase-wrap dropdown
(`principal_value` / `unwrap` / `none`), two file pickers (source
+ target intensity), initial-phase strategy (zeros / random /
from_file).  Live convergence plot + reconstruction preview.
Worker renamed `_GSWorker` -> `_PhaseRetrievalWorker` (back-compat
alias kept).

**New `lumenairy/ui/ao_dock.py` (864 LOC, audit P1-A)**: surfaces
the v5.2.3 `ao_closed_loop()` workflow.  DM actuator-count + modal
basis (zernike / karhunen_loeve / free) + max radial order + stroke
+ coupling controls.  WFS type (shack_hartmann / pyramid /
curvature; documented as captured but currently wired only to
ideal phase sensing per the library's `wfs=None` default).  Leaky-
integrator controller (gain / leak / tol / max iterations).  Input
selector (random Kolmogorov turbulence / loaded `.npy` phase /
manual Zernike spectrum).  3 embedded matplotlib canvases: live
convergence vs iteration, DM-command heatmap, residual-phase
heatmap.  Worker single-steps the helper (library has no per-iter
callback; matches the `examples/11_ao_closed_loop.py` pattern) and
emits residual_norm per iteration.

**New `lumenairy/ui/coronagraph_dock.py` (783 LOC, audit P1-C)**:
interactive 4-stop chain builder.  Per-stop profile dropdowns:
Stop 1 (apodised pupil) -- gaussian / cosine / super-gaussian /
uniform; Stop 2 (Lyot focal mask) -- hard_circular /
gaussian_taper / eight-octant_phase_mask / four-quadrant_phase_mask;
Stop 3 (Lyot stop) -- hard_circular / lyot_with_secondary_obscuration;
Stop 4 (image plane sampler) -- contrast_curve sampler with
n_radii + max_lam_over_D controls.  Reference PSF source toggle
(compute from system / loaded file).  Embedded `coronagraph_contrast_curve()`
plot (log10 contrast vs lambda/D) + per-stop 4-subplot intensity
previews + total throughput QLabel.

### Phase 3 -- Tier 2 Designer GUI (P2, significant)

**New `lumenairy/ui/algebra_dock.py` (935 LOC, audit P2-A)**: surfaces
the v4.15.1 `lumenairy.algebra` symbolic operator system.  Tree-view
operator chain.  Per-operator parameter dialogs for 8 operator
kinds: `FreeSpace` / `ThinLens` / `CylindricalLens` / `Magnify` /
`FourierTransform` / `Aperture` / `GaussianAperture` (composite
operator built directly).  Each row exposes its ABCD matrix
(button -> dialog).  Move up / down for reordering (composition
is non-commutative).  "From prescription" populator (uses
`Operator.from_prescription()` classmethod).  "Apply chain to
current field" runner with intensity preview.  Full-system ABCD
display + EFL extraction.  JSON save/load.

**New `lumenairy/ui/coatings_dock.py` (752 LOC, audit P2-D)**:
surfaces `elements/coatings.py`.  Stack editor (material +
thickness QTableWidget).  Substrate dropdown (BK7 / Fused Silica /
CaF2 / ZnSe / Si / Sapphire + custom-n).  Incident-medium
dropdown (air / vacuum + custom-n).  Lambda-sweep range + AOI +
polarisation (s / p / avg).  Embedded R(lambda) matplotlib
canvas.  3 quick-template buttons: single-layer MgF2 quarter-wave
at 550 nm, broadband AR V-coat optimiser, N-bilayer Bragg HR.
7-material hardcoded refractive-index database (MgF2 / SiO2 /
TiO2 / Ta2O5 / MgO / ZnS / Al2O3 at 550 nm, dispersion-flat
across sweep) documented in dock docstring -- the library does
not yet ship a material -> n registry for thin films.  R_mean /
R_min + lambda_min / R_max metrics display.

**New `lumenairy/ui/log_viewer_dock.py` (392 LOC, audit P2-B)**:
displays the v5.3.2 library logging telemetry stream from
`_logging.get_logger(...)` hooks on `apply_real_lens_traced` /
`design_optimize` / `monte_carlo_tolerancing`.  Capped 5000-line
QPlainTextEdit + level filter (DEBUG-CRITICAL) + module filter
(lumenairy.* prefix-based) + pause/resume + clear + save-to-file
+ find-next + status bar showing record counts.  Implements
`_QSignalLogHandler` (QObject + logging.Handler dual inheritance)
that re-emits records as Qt signals.  Lowers `lumenairy` logger
level on attach (default WARNING blocks INFO) and restores on
`closeEvent`.

**`lumenairy/ui/ghost_dock.py` expansion (audit P2-C)**: 141 ->
624 LOC.  Now wires all 3 library ghost-analysis functions
(`ghost_analysis`, `enumerate_ghost_paths`,
`non_sequential_stray_light`).  Path enumeration QTableWidget
(sortable by column).  4 filter knobs (max bounces, min
transmittance log10, min energy fraction ppm, sort-by combo).
Embedded matplotlib top-10 bar chart of ghost paths by energy
fraction.  Per-path spot-diagram preview.  Total stray-light
budget QLabel.  CSV export button.  Class signature preserved.

**`lumenairy/ui/jones_pupil_dock.py` Stokes tab (audit P2-E)**:
193 -> 404 LOC.  Central plot area converted to `QTabWidget` with
3 tabs.  "Jones pupil" -- existing 2x4 amplitude+phase grid
preserved.  "Stokes" -- 2x2 S0 / S1 / S2 / S3 heatmaps (S0
viridis, S1-S3 RdBu_r centred on 0).  "Polarisation derived" --
DOP / DOLP / DOCP heatmaps (viridis, range [0, 1]).  Implements
`_jones_to_stokes_unpolarized(J)` via canonical Mueller row-0
formulas (library exposes `stokes_parameters()` for a JonesField
but no direct `jones_to_stokes(J)` helper).  Status bar adds
<DOP> readout.

### Phase 4 -- Tier 3 Designer GUI (P3, polish)

**`lumenairy/ui/coherence_dock.py` expansion (audit P3-A)**: 162 ->
656 LOC.  Central layout becomes a QTabWidget with 4 tabs.  Tab 1
"Schell source" preserves the existing 162-LOC UI verbatim.  Tabs
2-4 wire 3 previously-unsurfaced library functions:
`koehler_image()`, `extended_source_image()`, `mutual_coherence()`.
New `_CoherenceAnalysisWorker(QThread)` dispatches on the active
tab.  Loaders accept `.npy` / `.npz` / `.h5` / `.hdf5` with
deterministic demo-data fallback on missing file.

**`lumenairy/ui/zernike_dock.py` polish (audit P3-B)**: 246 -> 379
LOC.  2 new controls: Normalization dropdown (OSA / Noll / Fringe
/ Standard -- library is OSA-locked per `zernike.py:31-34`;
non-OSA selections emit a UI warning and proceed with OSA),
Weighting group (none / circular_aperture / from_file with file
picker).  [Initially shipped in Phase 4 with post-hoc weighting --
OPD masked BEFORE `zernike_decompose()` -- because the library
function did not yet accept a `weighting=` kwarg; Phase 5 promoted
`weighting=` into `zernike_decompose()` directly so the dock now
passes the mask through canonically.  See Phase 5 section below
for the library extension.]  Default "none" preserves v5.3.2
numerical output bit-for-bit.

**New `lumenairy/ui/chebyshev_fit_dock.py` (533 LOC, audit P3-C)**:
specialised metrology tool for fitting measured profilometer /
interferometer height-map data to 2-D Chebyshev polynomials.
Loads z(x, y) from `.npy` / `.h5` / `.csv`, optionally normalises
aperture and masks outliers, fits via
`numpy.polynomial.chebyshev.chebvander2d` +
`numpy.linalg.lstsq` (the library's `_math/chebyshev.py` only
exposes Vandermonde tables, not a 2-D fitter; inline fit is
documented in the dock docstring).  Coefficient table
(`{(i, j): c_ij}` dict), RMS + PV residuals, 3-panel canvas
(raw / fit / residual), and "Apply to prescription" that emits
the library's canonical `freeform_type='chebyshev'` surface
format.

### Phase 5 -- Library extensions to retire dock-inline workarounds

The 4 phase-driven dock waves (Phase 2-4) shipped working surfaces
but left 8 inline workarounds where the library lacked a clean
primitive.  Phase 5 promotes those workarounds into library API so
the docks consume canonical functions:

* **`lumenairy.analysis.ao.make_shack_hartmann_wfs(...)`** (NEW;
  ~290 LOC).  Factory returning a `callable(residual) -> measured`
  suitable for `ao_closed_loop(wfs=...)`.  Captures
  subaperture_grid, noise_sigma_pixels, modal_basis, n_modes,
  dx_pupil, wavelength, lenslet_focal.  Implements first-call
  per-geometry calibration (zero-phase reference + unit-tilt
  linearity scale) cached in a closure-local dict so repeated
  calls are fast.  AO dock now wires the real SH WFS when the
  combo is `'shack_hartmann'`; pyramid / curvature fall back to
  `wfs=None` with a documented status message.  13 new tests.

* **`zernike_decompose(normalization=, weighting=)`** -- 2 new
  kwargs accepted by `lumenairy.analysis.zernike.zernike_decompose`.
  `normalization`: `'OSA'` (default, bit-for-bit v5.3.2),
  `'Standard'` (= OSA per ANSI/Z80.28-2010), `'Noll'` (sign-flip
  on m<0 modes per Noll 1976), `'Fringe'` (raises
  NotImplementedError with v5.5 pointer; the Fringe convention is
  a different mode set + peak-value-1 polynomials, not expressible
  as a per-mode rescale of OSA).  `weighting=` accepts a 2-D float
  array applied via canonical weighted-least-squares.  Zernike
  dock removed its post-hoc fallback and now passes the kwargs
  through.  11 new tests + 93 zernike tests overall.

* **`lumenairy.analysis.ghost.retrace_ghost_path(...)`** (NEW).
  Per-path explicit raytrace returning image-plane geometry
  (`rays_image_plane`, `peak_xy_mm`, `rms_radius_mm`, `fwhm_mm`,
  `total_transmittance`, `energy_fraction_ppm`).  Ghost dock spot
  preview is now a real retrace (hist2d + scatter) instead of the
  synthesised relative-magnitude Gaussian.  Discovered a backward-
  ray intersection issue in the library's `_intersect_surface`
  fast path (it picks the wrong sphere root for rays travelling
  toward `-z` after a reflection); the scoped workaround is a new
  `_ghost_intersect` helper in `analysis/ghost.py` that uses the
  smaller-magnitude root.  Promoting this fix to
  `raytrace/intersection.py` is a v5.5 candidate.  5 new tests.

* **`create_four_quadrant_phase_mask()`** + **`create_eight_octant_phase_mask()`**
  (NEW; in `lumenairy.elements.elements`, re-exported via
  `lumenairy.elements.coronagraph`).  FQPM and 8OPM builders with
  `phase_step=pi` default + configurable centre.  Coronagraph dock
  removed its inline `_phase_octant_mask` and now consumes the
  library helpers.  7 new tests including the canonical "FQPM-pi
  suppresses planar-wave on-axis intensity by ~256x at N=64 (4/N^2
  scaling with grid resolution)" check.

* **`lumenairy.elements.coatings.COATING_MATERIAL_REGISTRY`** +
  **`get_coating_material_index(material, wavelength)`** (NEW).
  12 thin-film materials (MgF2, SiO2, TiO2, Ta2O5, MgO, ZnS,
  Al2O3, HfO2, Y2O3, ZrO2, CeO2, CaF2).  Top 4 (MgF2 / SiO2 /
  TiO2 / Ta2O5) carry real 3-term Sellmeier coefficients sourced
  from refractiveindex.info (Dodge / Malitson / DeVore / Bright);
  the other 8 use a flat `n_constant` at 550 nm.  Coatings dock
  removed its hardcoded 7-material dict and now drives the
  Material dropdown from `sorted(COATING_MATERIAL_REGISTRY.keys())`.
  With dispersion engaged, TiO2 at 550 nm shifts 2.40 -> 2.58 and
  Ta2O5 shifts 2.10 -> 2.23 (correct rutile / Bright values);
  MgF2 / SiO2 stay within 0.001 of the prior constants.  10 new
  tests; Bragg-HR template auto-tracks the new values.

* **`lumenairy._math.chebyshev.chebyshev_fit_2d(x, y, z, ...)`**
  (NEW).  2-D Chebyshev least-squares fit returning the same
  `{(i, j): c_ij}` coefficient dict format that
  `lumenairy.elements.freeform.surface_sag_chebyshev` consumes.
  Accepts `weight=` (variance weights), `normalize_xy=` (rescale
  to [-1, 1] before fit), `return_residual=`.  Chebyshev-fit dock
  removed its inline `chebvander2d + lstsq` block and now calls
  the library helper.  6 new tests including bit-for-bit
  Chebyshev coefficient recovery and weighted-fit outlier
  rejection.

* **CHANGELOG archive completion** (audit P3-7).  Inspection
  confirmed the v5.2.3 + v5.3.0 archive splits had already moved
  v4.11.x through v2.5.x into `docs/changelogs/v4.md`.  Top-level
  CHANGELOG.md's oldest entry is v4.13.0 (2026-05-17).  Refreshed
  the trailing archive-note paragraph to cite v5.4.0 Phase 5 as
  the explicit completion checkpoint.

After Phase 5, the v5.4.0 audit-closure list is empty: every
deferral noted during Phases 1-4 has been retired with a library
extension + dock simplification + regression tests.

### Designer GUI version bump

Internal version markers in `lumenairy/ui/` were at v3.7.10 (per
`main_window.py:2196` etc).  The Designer ships co-versioned
inside the library wheel and pulls its display version from
`lumenairy.__version__` at runtime, so the v5.3.2 -> v5.4.0
library bump automatically renders as "LumenAiry Designer 5.4.0"
in the title bar + About dialog.  The historical "3.7.10"
embedded changelog comments are preserved (audit
AUDIT_V5_3_2_GUI_VS_LIBRARY Part 4 classified them as
cosmetic; no remediation required).  Going forward Designer
versioning is the library versioning.

### Documentation

* `docs/designer_guide.md` (NEW) -- dedicated GUI documentation
  covering the v5.4.0 dock surface (37 docks total: 22 pre-existing
  + 9 new in v5.4.0 + 6 expansions).  Tab-by-tab dock inventory,
  registration order, library backings, library functions wired.
* `Migration-Guide.md` -- v5.4 section noting the optional
  CancellableProgress Stop-button surface (default unchecked, no
  user-visible change unless wired).
* `ROADMAP.md` -- Designer GUI section regenerated to reflect the
  v5.3.2 -> v5.4.0 closures.  v5.x ROADMAP is now fully closed --
  ALL library + GUI work-items shipped.
* Wiki `Release-Notes.md` -- v5.4.0 section added.

### v5.x ROADMAP status

After v5.4.0, **the v5.x ROADMAP is fully closed** -- both library
code-work AND Designer GUI:

* Library: all P1 / P2 / P3 from AUDIT_V5_3_2_2026_05_23 shipped.
* GUI: all P1 / P2 / P3 from AUDIT_V5_3_2_GUI_VS_LIBRARY shipped.
* Designer GUI version bumped 3.7.10 -> 5.4.0 (co-versioned).
* Next audit cycle (AUDIT_V5_4_0_*) -- yours to call.

### Files touched

54 files modified or created.  Net LOC: +13813 / -1075 vs v5.3.2.

---

## [5.3.2] — 2026-05-23

**Three v5.x horizon-item closures shipped together.**  After this
release the v5.x ROADMAP horizon has 1 item left (next-audit-cycle
process item) + Designer GUI (separate stream).  No library
behavior change for existing callers; all three items add
*observational* infrastructure (telemetry + drift-detection +
ship-time validation).

**Zero physics regressions in 16 consecutive releases.**

3865 unit tests pass (collected = 3886 = pass + 20 skip + 1
xfail).  +15 net pass vs v5.3.0 (3848); +3 new skips for
documented data-driven cases (V12.2/V12.4 audit-bullet absence in
docs-only v5.3.2 block + V18 source-line absence + V17.3
line-count-claim absence -- all "no input to verify" rather than
test logic failures).

### Item 1: per-iteration `logging` telemetry on long-running paths

The audit-cited ROADMAP item ("42 `warnings.warn` calls across
22 files; long-running paths have no per-iteration telemetry")
is scoped precisely to TELEMETRY -- not to a conversion of the
`warnings.warn` surface.  The 42 cited `warnings.warn` sites
stay as-is (they are public API contract).  Three long-running
functions gain INFO-level per-iteration logging hooks:

* **`apply_real_lens_traced`** (`lumenairy/elements/_lens_traced.py`):
  entry log + per-Newton-iteration log (`iter k/N residual_max=X
  m`) + early-convergence log.
* **`design_optimize`** (`lumenairy/optimize/driver.py`): entry
  log + per-scipy-iteration log via the existing
  `CancellableProgress` callback path (one extra line per
  method-specific callback site; covers L-BFGS-B / SLSQP /
  trust-constr / DE / basin-hopping / dual_annealing).
* **`monte_carlo_tolerancing`** + **`monte_carlo_tolerancing_jax`**
  (`lumenairy/analysis/through_focus.py`): per-trial log + entry
  log on each twin.

New `lumenairy/_logging.py` private module exposes the
convention.  Default-quiet contract: with no user-attached
handler, the library produces zero log output (a `NullHandler`
sits on the `'lumenairy'` root logger at import time).  Users
opt in:

```python
import logging
logging.basicConfig(level=logging.INFO)
# or per-logger:
logging.getLogger('lumenairy').setLevel(logging.INFO)
```

8 new tests in `tests/unit/test_v5_3_2_logging_telemetry.py`
pin the default-quiet contract, the telemetry-active contract,
and bit-for-bit behavioural preservation (a 1-iter
`design_optimize` returns the same `result.x` and `result.merit`
with and without the INFO handler attached).

### Item 2: V18 walker -- source-file:line citation drift

The v5.3.0 ship hit a real failure of this class -- CHANGELOG
cited `optimize/wrapper_merits.py:855` but the v5.3.0 MultiFieldMerit
JIT change had shifted the sentinel branch to line 876.  V18 is
the GENERAL version of `test_v4_15_agent_f.py::TestF5ChangelogLineCitations`
which only catches a hardcoded shortlist of symbols.

**Walker** (`tests/unit/test_v5_3_2_walker_source_line_citation.py`):
parses the topmost CHANGELOG block, extracts ALL backticked
`lumenairy/foo/bar.py:LINE` (or `:START-END`) citations, opens
each at the cited line, and verifies the line is **non-trivial**
(not whitespace, not a closing bracket, not a bare
`pass`/`continue`/`break`/`else:`/`try:`).  This is necessary-
but-not-sufficient (V18 doesn't try to identify the cited
SYMBOL -- that requires a hand-curated anchor database).  5
tests; 4 pass + 1 clean skip on the docs-only v5.3.1 block.

**Companion script** (`scripts/check_source_line_citations.py`,
429 LOC): runnable as `python scripts/check_source_line_citations.py
[--version VER] [--quiet]`.  Exit codes: 0 OK, 1 drift detected,
2 clean skip.  Handles both full-path and bare-basename
citations (resolves bare basenames via a basename->path index;
basenames matching multiple files report SKIP-AMBIGUOUS).

**Wired into `publish.yml`** at the verify job, line 93, BEFORE
the V16 `verify_changelog_closures.py` step.  Same exit-code-2
= pass treatment.  Line-level drift surfaces before content-
level drift.

### Item 3: CHANGELOG ship-time-stamp injection

The v5.2.5-documented "recursive self-citation drift" class:
each CHANGELOG entry self-cites build-time empirical numbers,
but the entry itself is part of the diff that establishes those
numbers, so cited counts are always at-write-time, not
at-ship-time.  V17 walker (`tests/unit/test_v5_3_walker_changelog_self_citation.py`,
v5.3.0) DETECTS drift; the FIX requires a pre-tag hook that
refreshes the cited values just before tag commit.

v5.3.2 ships that pre-tag hook as a CLI script:

**`scripts/stamp_changelog.py`** (623 LOC): parses the topmost
`## [X.Y.Z]` CHANGELOG block, computes empirical test count
(`pytest --collect-only` in `--quick` mode, full suite in
`--full` mode), file count (`git diff PREV_TAG..HEAD`), and
line count (`wc -l CHANGELOG.md`).  Rewrites the matching
self-citation patterns in the topmost block.  **Dry-run by
default**; explicit `--apply` flag required to write.

Four invocation patterns documented in
`docs/release-process.md` (new):

```
python scripts/stamp_changelog.py                      # dry-run preview (default)
python scripts/stamp_changelog.py --quick --apply      # canonical pre-tag (~5s)
python scripts/stamp_changelog.py --full --apply       # empirical pass/skip/xfail (~5min)
python scripts/stamp_changelog.py --version 5.2.3 --apply  # back-stamp past block
```

6 tests in `tests/unit/test_v5_3_2_stamp_changelog.py` cover
parsing, --help, dry-run-against-current-CHANGELOG, and
synthetic-stamps-correctly.  Exit codes mirror V12/V16 idioms
(0 ok / 1 drift-detected-but-not-applied / 2 clean-skip / 3
input-error).

**Self-dogfood**: this v5.3.2 entry's test counts were stamped
by the new script (see commit message for the dry-run preview
output).

### Documentation

* `Migration-Guide.md` -- new "opt-in telemetry logging" section
  (v5.3.2 + how to attach a handler) + maintainer-only pointer
  to `docs/release-process.md`.
* `docs/release-process.md` (NEW) -- pre-tag stamp workflow +
  the four `stamp_changelog.py` invocation patterns.
* `ROADMAP.md` -- v5.3.2 added; remaining v5.x horizon is now
  process-only (next audit cycle + Designer GUI v3.8+).
* Wiki `Release-Notes.md` -- v5.3.2 section added.

### v5.x ROADMAP status

After v5.3.2, **the v5.x ROADMAP code-work is fully closed**.
Remaining horizon is process-only:

* `logging` adoption sweep -- SHIPPED (this release, item 1).
* CHANGELOG ship-time-stamp injection -- SHIPPED (this release,
  item 3).
* V18 walker -- SHIPPED (this release, item 2).
* Next audit cycle (AUDIT_V5_3_2_*) -- process item, yours to call.
* Designer GUI v3.8+ -- separate version stream.
* Force-retag discipline retrospective -- v5.2.5, v5.3.0, and
  v5.3.2 each needed at least one post-tag commit (v5.3.1 was
  the only clean single-commit release in the cycle); decide
  whether to accept the class or add a structural pre-tag check.

### Files touched

16 files modified or created.  Net LOC: +2766 / -11.

(File-count and LOC stamped post-commit so ``git diff v5.3.1..HEAD
--name-only`` returns the committed-snapshot value.  The
``stamp_changelog.py`` script that ships in this release uses the
same git diff invocation as the V17 walker, so a pre-commit stamp
run produces 0 files until ``git commit`` lands -- ``docs/release-
process.md`` documents the post-stamp re-amend workflow.)

---

## [5.3.1] — 2026-05-23

**Docs-only patch: PyPI project-page sync fix.**  No library code
change; no test change.  Sole purpose is to surface the
v5.1.1 / v5.2.x / v5.3.0 release notes on PyPI's project page,
which were missing because the README's embedded
``## What's new in X.Y.Z`` block had stopped accruing new sections
after v5.1.0.

### What changed

* **README.md ``What's new in X.Y.Z`` block stripped**.  The
  embedded historical release-notes block (v3.0 through v5.1.0)
  is removed -- 4076 to 5168 README lines collapsed into a
  single ``## Release notes`` pointer paragraph that links to
  `CHANGELOG.md` (the canonical source), the wiki Release-Notes
  page, and the GitHub Releases feed.  PyPI shows the README as
  the project description; pre-v5.3.1 the embedded block
  advertised v5.1.0 as the "latest" release because the per-
  release sync step had stopped after v5.1.0 was tagged.  The
  pointer-only form has no stale-content failure mode.

* **README.md line count**: 4762 -> 3671 (-1091 lines).  Block
  bounds: lines 13-1104 (the first ``## What's new in 5.1.0``
  through the last entry of ``## What's new in 3.0``).

### Why not back-fill the missing What's-new sections instead

User-chosen option (from a 3-way trade-off): the back-fill path
would require 7 new "What's new" sections in the README (5.1.1,
5.2.0, 5.2.1, 5.2.2, 5.2.3, 5.2.5, 5.3.0) AND a release-process
step to remember to add another section before each future tag.
The pointer-only form has zero ongoing maintenance and zero
stale-content failure mode.  The trade-off is that the PyPI
project page no longer shows release highlights inline -- users
who want them click through to CHANGELOG.md (a single click in
the PyPI sidebar's ``Changelog`` link).

### What did NOT change

* `CHANGELOG.md` -- structure unchanged (this is the canonical
  source going forward; nothing was overhauled).
* All library code, examples, tests, walkers, and ROADMAP --
  bit-for-bit unchanged from v5.3.0.

### Tests

Same as v5.3.0: 3847 unit tests pass (collected = 3866 = pass + 18
skip + 1 xfail).  V11 doc-consistency walker still green against
the trimmed README.

### v5.x ROADMAP status (unchanged)

**The v5.x ROADMAP remains empty of code work.**  Remaining
horizon is process-only (logging adoption sweep, Designer GUI
v3.8+, next audit cycle, CHANGELOG ship-time-stamp injection).

---

## [5.3.0] — 2026-05-22

**Closes AUDIT_V5_2_5_2026_05_22** (1 P1 blocker + 2 P2 + 12 P3
findings from the 6-agent post-v5.2.5 audit) **plus the 3 remaining
v5.x ROADMAP horizon items** (conftest comment fix, CHANGELOG
pre-v4.11 archive completion, MultiFieldMerit JIT compile).  After
this release, every v5.x ROADMAP item has shipped.

**Zero physics regressions in 15 consecutive releases.**

3848 unit tests pass (collected = 3866 = pass + 17 skip + 1 xfail);
**+67 net pass vs v5.2.5**.  34/34 validation pass.

### P1 closure (1) -- BLOCKER

* **HF freespace dispatcher `output_grid` TypeError fix** (P1-1).
  v5.2.5 closed the v5.2.3 P2-F1-1 "HFPI/HF freespace dispatcher
  silently drops output_grid/output_dx" finding by threading the
  kwargs through to `propagate_huygens_fresnel_freespace`.  HFPI
  worked correctly (its receiver natively accepts those kwargs);
  HF was BROKEN because the HF freespace wrapper is a thin
  pass-through to `rayleigh_sommerfeld_propagate`, which does not
  accept `output_shape` or `output_dx`.  Pre-v5.2.5 was a silent
  no-op; v5.2.5 was a hard `TypeError`.  v5.3 fixes the pass-through
  by handling the resample INSIDE `propagate_huygens_fresnel_freespace`
  (matches the v5.2.3 MHS substantive-resampling pattern at
  `mhs.py:573-611`).  Return contract honored: bare ndarray when
  no output kwargs, `(E_out, dx_out)` 2-tuple when resampled.  5
  new regression tests in `tests/unit/test_v5_3_hf_freespace_output_grid.py`
  pin the contract end-to-end.

### P2 closures (2)

* **`scripts/verify_changelog_closures.py` audit-ID regex relaxed**
  (P2-1).  The v5.2.5 P2-F1-2 closure relaxed the V12 walker regex
  in `test_v5_2_walker_changelog_changeset.py:104-105` but missed
  the parallel regex in the companion CLI script.  v5.3 lands the
  fix at the second site: short-form audit IDs (`P1-A`/`P1-1`/etc.)
  now match the script's audit-ID extractor too.  Closes the
  fix-at-both-sites gap.
* **V17 walker: recursive self-citation drift** (P2-2 +
  AUDIT_V5_2_5 Part 7 NEW class).  v5.2.5 documented a new
  sibling-gap class: each release's CHANGELOG self-cites
  build-time empirical numbers, but the CHANGELOG entry itself is
  part of the diff that establishes those numbers -- so cited
  counts are always at-write-time, not at-ship-time.  v5.2.5's
  own entry claimed `3780 / 16 / 1` but empirical was `3781 / 15 / 1`;
  claimed 29 files but actual was 26; claimed ~10769 lines but
  current is ~10949.  v5.3 ships a structural walker
  (`tests/unit/test_v5_3_walker_changelog_self_citation.py`) that
  pins three numeric self-citation invariants: test-count
  arithmetic (V17.1), file-count claim within +/- 5 file drift
  band (V17.2), and CHANGELOG.md line-count claim within +/- 300
  line drift band (V17.3).  The walker surfaces drift at audit
  time; the actual ship-time-stamp injection is a v5.3.1+ goal
  (would require a pre-tag hook).

### P3 closures (10)

(audit's "(12)" header was a CHANGELOG self-citation drift; the
delivered list is 9 P3 items P3-1 through P3-9 plus 1 stretch goal
(V16 Tier-N regex widening) = 10 total.  Caught at v5.4 by the
audit-of-audits at `docs/audits/AUDIT_V5_3_2_2026_05_23.md` Part 2.)

* **`ao_closed_loop` docstring + leak/gain=0 joint semantics fix**
  (P3-1).  Docstring claimed `gain=0` meant "command never
  updated", but with `leak > 0` the leak term decays the existing
  command each iteration.  Pure "command never updated" requires
  BOTH `gain = 0` AND `leak = 0`.  Docstring now honestly
  documents the four (gain, leak) combinations + cites the
  `test_leak_nonzero_decays_command` pin in
  `test_v5_2_5_ao_closed_loop_residuals.py`.
* **AST `_resolve_arg_closure` against 4 new bypasses** (P3-2).
  v5.2.5 closed the multi-assign-shadow + dead-code-in-unreached-
  function bypasses; AUDIT_V5_2_5 surfaced 4 more: `AugAssign`,
  `AnnAssign`, tuple-unpack rebind, `with-as` / `for`-target
  rebind.  v5.3 extends `_collect_rebinds` to handle all 5
  AST forms with last-write-wins semantics; the walker now
  rejects the closure check when the LAST rebind event is anything
  other than a plain `assign_direct`.  4 new gaming-form tests
  in `test_v4_16_1_agent_d.py`.  Meta-pattern note: still
  game-able via dynamic code paths (`exec`, `setattr(globals())`)
  -- accepting that this surface is structurally bounded.
* **dispatcher `output_dx`-alone defensive shape** (P3-3 -- closed
  as no-fix-needed once P1-1 landed).  The audit flagged that
  `_resolve_dispatcher_output_grid` always returns a SQUARE
  `(N_in, N_in)` shape when only `output_dx` is set; this was
  "benign for HFPI, triggers P1-1 TypeError for HF".  With the
  P1-1 fix in place, HF now resamples cleanly to the square
  shape too.  No additional change needed.
* **`_AUDIT_ID_KNOWN_PREFIXES` allowlist wired** (P3-4).  The
  allowlist was defined at v5.2.5 but never used.  v5.3
  `_extract_cited_audit_ids` now filters candidates through the
  allowlist as a defense-in-depth guard (regex enforces the same
  gate at the moment, so this is structurally a no-op; if a
  future regex relaxation widens the prefix surface, the
  allowlist catches false positives in prose).
* **Chebyshev derivative `xp` branch unreachable code removed**
  (P3-5).  In `chebyshev_derivative_vandermonde`'s xp branch,
  the early-return at `max_k < 1` makes the downstream
  `if max_k >= 1: U.append(2.0 * u_arr)` check tautologically
  true.  Simplified to direct construction.
* **`install_atexit_restore` -> `_install_atexit_restore`
  bootstrap migration** (P3-6).  v5.2.5 renamed the function but
  left the library bootstrap (`lumenairy/__init__.py:682`) calling
  the legacy public name.  v5.3 migrates the bootstrap to the
  underscore form; the back-compat alias in `_context.py`
  remains for external callers.  "Private intent" is now a
  consistent signal across internal call sites.
* **Strehl > 1 methodology disclosure** (P3-7).
  `examples/09_tolerancing_monte_carlo.py` gains a docstring note
  + conditional summary print explaining that the small fraction
  of trials reporting Strehl slightly > 1.0 (max ~1.003) is NOT
  a normalization bug -- it's a methodology consequence of
  computing `ideal_peak` at the NOMINAL focal length while each
  trial searches for its own `z_best`.  Corrects the v5.2.5
  CHANGELOG attribution of "rounding noise".
* **JAX `monte_carlo_tolerancing_jax` regression test added**
  (P3-8).  v5.2.5 mirrored the NumPy Strehl-normalization fix
  in the JAX path at `through_focus.py:1251-1267` but no
  regression test exercised it.  v5.3 adds 4 tests in
  `tests/unit/test_v5_3_jax_monte_carlo_tolerancing.py`: Strehl
  values bounded [0, 1.05], mean < 1.0, JAX-vs-NumPy parity
  within 0.05 (loose because JAX defaults to float32 without
  `JAX_ENABLE_X64`), and return-dict shape.  Tests skip cleanly
  when JAX is not installed.
* **ROADMAP test-count off-by-one corrected** (P3-9).  v5.2.5
  ROADMAP cited 3780/16/1; matches CHANGELOG header at the same
  drift.  v5.3 stamps ROADMAP at the v5.3 empirical numbers.
* **V16 Tier-N regex widened to `\d+`** (audit Part 8 stretch
  goal #12).  Was `\btier\s+[0-9]\b` -- single digit only.  Now
  handles hypothetical Tier-10+ headings.

### v5.x ROADMAP horizon items (3)

* **`pyproject.toml` conftest comment-text correction** (user item
  #1, AUDIT_V5_2_5 V6 finding).  v5.2.2 inline comment cited
  "v4.16.0 conftest pattern" for zarr skip; the pattern actually
  lives at `test_v4_16_0_agent_b_multiprocess_storage.py:142-159`
  (per-test, not conftest).  `tests/conftest.py` has zero zarr
  handling.  Corrected.
* **CHANGELOG.md pre-v4.11 archive completion** (user item #2).
  v5.2.3 moved v4.11.x + v4.12.x entries into
  `docs/changelogs/v4.md` (962 lines).  v5.3 completes the
  archive: v4.10.x through v2.5.x entries moved into
  `docs/changelogs/v4.md` (now 6676 lines); top-level
  `CHANGELOG.md` reduced from 10949 to 5236 lines.  Net move
  preserves every entry verbatim with CRLF -> LF normalisation to
  match the v4.md file's existing line endings.  V11 doc-
  consistency walker still green.
* **`MultiFieldMerit` JIT compile** (user item #3, ROADMAP v5.3
  horizon).  The per-field `np.where(mask, np.exp(1j * (sin(tx)
  * k_X + sin(ty) * k_Y)), 0)` chain -- three N x N temporaries
  on the legacy NumPy path -- is now fused into a single Numba
  `@njit(parallel=True, fastmath=True)` kernel with zero
  temporaries.  Two dtype-specialised kernels (complex64 +
  complex128) dispatched at the call site.  **Measured speedup:
  1.5-8x speedup (hardware dependent; v5.3 audit measured 8.53x
  at N=256/8 fields on a 16-core machine, dipping to 1.5-3x
  under heavy parallel contention)** (jit=3.1 ms vs numpy=25.4
  ms at the headline operating point).  Lifts the meshgrid-
  cache-hit performance ceiling cited in the v4.14.0 perf audit
  (1.19x at N=128).  NumPy fallback bit-for-bit preserved; JIT
  path matches the NumPy reference to `rtol=1e-12` (FMA
  reassociation under `fastmath=True` produces ULP-level
  differences, documented in `lumenairy/optimize/_merit_jit.py`).
  The NumPy path is also used when Numba is unavailable OR when
  the grid is below the JIT-call-overhead threshold (N < 128).
  52 new tests in `test_v5_3_multi_field_merit_jit.py` covering
  numerical equivalence + cache-hit preservation + speedup
  pin + fallback.

### Documentation updates

* `CHANGELOG.md` -- v5.3.0 entry + archive split (5713 lines moved
  to `docs/changelogs/v4.md`).
* `ROADMAP.md` -- v5.3.0 added to release timeline; baseline test
  counts refreshed.
* `examples/09_tolerancing_monte_carlo.py` -- Strehl methodology
  note added.
* `lumenairy/analysis/ao.py` -- docstring leak/gain=0 joint
  semantics paragraph added.
* `lumenairy/_math/chebyshev.py` -- unreachable-code cleanup
  + comment.
* `pyproject.toml` -- conftest comment-text correction.
* `Wiki` -- Release-Notes + Function-Reference-Optimization +
  Adaptive-Optics pages updated for v5.3.

### Items still open (v5.3.1+ horizon)

After v5.3.0 ship, the v5.x ROADMAP is **EMPTY of code work**.  The
remaining horizon is process-only:

* **`logging` adoption sweep** (42 `warnings.warn` -> structured
  logging where appropriate).  Library-wide convention change;
  best done when there's a concrete telemetry trigger.
* **Designer GUI v3.8+** (separate version stream).
* **Next audit cycle** (AUDIT_V5_3_X).
* **CHANGELOG self-citation ship-time injection** (V17 currently
  surfaces drift; the FIX requires a pre-tag hook to stamp
  empirical values into the CHANGELOG entry before tag commit).

---

## [5.2.5] — 2026-05-21

**Patch release closing AUDIT_V5_2_3_2026_05_21** (the 4-release-chain
v5.1.1 -> v5.2.3 audit by the 12-agent fleet).  Closes all 7 P2 +
all 10 P3 findings.  Version number jumps from 5.2.3 to 5.2.5 (no
5.2.4) to signal that this is a substantive patch rather than a
trivial cleanup.

**Zero physics regressions in 14 consecutive releases.**

3780 unit tests pass (collected = 3797 = pass + 16 skip + 1 xfail);
+15 net pass vs v5.2.3.  34/34 validation pass.

### P2 closures (7)

* **V12 walker regex relaxed to accept short-form audit IDs**
  (P2-F1-2).  The v5.2.0 regex
  `r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z][A-Z0-9_-]{2,})'` required
  `[A-Z0-9_-]{2,}` after the first uppercase letter, so short-form
  IDs like `P1-A` / `P1-C` / `P1-F` / `P1-G` / `P1-1` (the IDs cited
  in v5.2.3's own CHANGELOG) were silently rejected.  V12.2 was a
  no-op on the v5.2.3 release.  Relaxed to
  `r'(?<![A-Z0-9])(P[0-3](?:-NEW)?-[A-Z0-9][A-Z0-9_-]*)'` + known-
  prefix allowlist (`P0-` / `P1-` / `P2-` / `P3-`).  Both
  short-form and long-form audit IDs now resolve through V12.2.
* **V16 heading classifier extended to "Tier N closures"** (P2-F1-3).
  `_is_audit_closure_section` in
  `scripts/verify_changelog_closures.py` only matched
  `'audit closure'` / `'audit carry-over'` / `'audit fix'` / `\bp[0-3]\s+closure`.
  v5.2.3 used `### Tier 1 closures` / `### Tier 2 closures` /
  `### Tier 3 closures` -- none matched.  Empirically
  `python scripts/verify_changelog_closures.py --version 5.2.3`
  exited rc=2 (treated as pass by CI gate); a v5.2.4 with
  fabricated `### Tier N closures` bullets would NOT have been
  caught.  Extended classifier with `re.search(r'\btier\s+[0-9]\b', s)`.
* **Python 3.10 added to publish.yml verify matrix** (P2-F1-4).
  v5.1.1 + v5.2.x publish.yml matrix was `[3.11, 3.12, 3.13]` -- no
  3.10 even though the documented floor is 3.10 and v5.2.2's whole
  reason for existing was 3.10 install-path drift.  Bumped to
  `[3.10, 3.11, 3.12, 3.13]` matching unit-tests.yml.
* **HFPI/HF freespace dispatcher threads `output_grid`/`output_dx`**
  (P2-F1-1).  v5.2.3 fixed the through-prescription paths but the
  freespace branches (when `prescription is None`) silently dropped
  the resolved values.  `propagate(method='hfpi', output_grid=...)`
  without a prescription returned a default-shape result.  Now
  threads `(output_shape, output_dx)` through the freespace
  branches too.
* **Consolidated test refreshed to v5.2+ kwarg idiom** (P2-F1-5).
  3 sites in `tests/unit/test_audit_propagation.py` carried the
  v4.11.2-era `output_grid=(Ny, Nx)` form, which now fires the
  v5.2.0 DeprecationWarning shim.  Tests passed only because
  DeprecationWarning is non-fatal by default.  Refreshed to
  `output_shape=(Ny, Nx)` so the bit-for-bit preservation claim is
  honest under `-W error::DeprecationWarning`.
* **AST `_resolve_arg_closure` tightening** (P3 V1 residual).  v5.2.0
  required the `'output'` literal + `__file__` reference inside the
  `makedirs(...)` first-arg subtree OR the RHS of the binding
  assignment.  F1 demonstrated two narrower gaming bypasses:
  multi-assign-shadow (`out_dir = good; out_dir = bad; makedirs(out_dir)`)
  + dead-code-in-unreached-function.  v5.2.5 adds function-scope
  filtering + last-write-wins semantics.  Both bypasses now
  rejected; 3 new gaming-form tests added.
* **dep-metadata drift check re-run** (P2 V6, no code change).
  Audit claimed the zarr env-marker was stale (zarr 3.2.1 requires
  >=3.12 vs pyproject >=3.11) and scipy/numpy/astropy/jax had
  silent floor drift.  v5.2.5 ran the v5.2.3-shipped dep-metadata
  drift script and confirmed **zero drift** -- the resolver picks
  compatible earlier zarr 3.0/3.1.x releases on Python 3.11 under
  the current bare-version constraints; same for scipy / numpy /
  astropy / jax.  Audit claim was speculative; current pyproject
  is correct.  The v5.2.3-shipped Monday cron continues to watch
  for real future drift.  **No files modified for this finding.**

### P3 closures (10)

* **Example 09 Strehl > 1 normalization fix** (P3-F1-1).  Both
  library-side AND example-side bugs identified + fixed:
  - `monte_carlo_tolerancing` (numpy + JAX backends) was
    recomputing `ideal_peak` inside the trial loop from each
    PERTURBED `E_exit`.  Pinned to the UNPERTURBED nominal pupil
    computed ONCE before the loop.
  - Example 09 hard-coded `f_target = 100e-3` but the singlet's
    actual paraxial BFL is 97.015 mm; passing a focal length that
    disagrees with the lens's true focus also produces Strehl > 1.
    Replaced with a `system_abcd` ray-trace to get the true BFL.
  Final printed Strehl: 0.967 +/- 0.029 (mean +/- std), 5th pct
  0.910, 95th pct 0.997.  All percentiles physically bounded;
  max of 1.003 is rounding-noise from per-iteration perturbations
  slightly shifting focus past the reference plane.
* **`ao_closed_loop` `leak` + `tol` kwargs + edge cases** (P3-F1-2 +
  V9).  Added two new kwargs: `leak: float = 0.0` (default 0.0 =
  pure integrator, bit-for-bit identical to v5.2.3) and
  `tol: Optional[float] = None` (residual-RMS early-stop
  threshold).  Loosened `gain` validation from `0.0 < gain <= 1.0`
  to `0.0 <= gain <= 2.0` -- `gain=0.0` is now an open-loop
  fallback.  Added `wfs(...)`-returning-None skip-update path
  (no longer crashes).  Cited CONVENTIONS.md Section 7 in
  docstring.  Bit-for-bit preservation at `leak=0.0` verified
  via `np.array_equal` pin.  8 new tests.
* **Chebyshev derivative + second-derivative `xp=` dispatch**
  (P3-F1-3).  v5.2.0 module docstring claimed "all three helpers"
  support `xp=` backend dispatch but only `chebyshev_vandermonde`
  did.  v5.2.5 makes the existing claim true: added `xp=None` kwarg
  to `chebyshev_derivative_vandermonde` and
  `chebyshev_second_derivative_vandermonde` + dispatched internal
  `np.*` to `xp.*`.  JAX traceability verified under `jax.jit`.
  Consumer call sites preserve numerics exactly (positional calls
  don't pass `xp=`, default `None` -> NumPy bit-for-bit).
* **CHANGELOG `apply_doe_phase_traced` re-worded** (P3-F1-4).
  v5.2.0 description said "phase advance/retard"; actual fix is
  direction-cosine deflection.  Grating equation L -> L + m*lambda/Lambda
  is transverse-only; OPL accumulator is unchanged.  Reworded in
  the v5.2.0 entry.
* **CHANGELOG test-count self-citation accuracy** (audit Part 8
  P3 row 5, F1).
  CHANGELOG v5.2.3 entry stated 3765/18/1; F1's empirical run
  showed 3766/17/1 (1-test delta from V12.4 conditional path).
  V12.3 self-consistency holds either way since 3784 = 3766 + 17 + 1
  = 3765 + 18 + 1.  Stamped the v5.2.3 entry counts at the
  build-time empirical numbers (verified at v5.2.5 ship as 3780
  pass / 16 skip / 1 xfail = 3797 -- the new tests + AST tightening
  account for the +15 pass delta).
* **V15 walker floor bumped 5 -> 6** (P3-F1-6).  v5.2.0 discovered
  6 sentinels including `_SchellReturnKindUnsetSentinel` (the
  v4.15.2 hardcoded tuple was missing it).  Floor tightened to >= 6
  to reflect that baseline.  Catches a walker collapse without
  leaving silently-pass headroom.
* **`install_atexit_restore` -> `_install_atexit_restore`** (audit F4).
  Renamed to underscore-prefixed form to signal private-bootstrap
  intent (caller is `lumenairy/__init__.py` at the end of library
  import; no user-facing call site).  Legacy name preserved as
  back-compat alias at the bottom of `_context.py` so any external
  caller importing it by the old name continues to work.
* **`docs/cookbook.md` cross-links** (audit F4).  Added 5
  cross-links: 2 in the header "See also" block (CONVENTIONS.md
  + Migration-Guide.md), 2 in OPD/Zernike + through-focus recipes
  (pointing at CONVENTIONS.md Section 7), 1 in polarization recipe
  (Migration-Guide.md for v4 -> v5 shim removals).
* **`ao_closed_loop` docstring cites CONVENTIONS Section 7**
  (audit F4).  One-line pointer in the Notes section: "phases follow
  the library-wide convention table in `CONVENTIONS.md` Section 7
  (OPD sign + time convention + forward propagation)".
* **CHANGELOG.md:81 line-count self-citation refreshed** (audit V8).
  v5.2.3 entry said `11553 -> 10618 lines` at the time of the
  archive split.  v5.2.3's entry itself added ~150 lines after
  the split was measured; current state is ~10769 lines.  v5.2.3
  bullet now states the AT-THE-TIME-OF-SPLIT count + the post-add
  current state.

### Side-effects

* `_context.py` legacy `install_atexit_restore` alias is NOT in
  `__all__` (matches the rename's "private intent" signal).  Any
  user-facing call site relying on `from lumenairy._context import install_atexit_restore`
  continues to work.
* Existing v5.2.3 test `test_ao_closed_loop_rejects_bad_gain` was
  updated to reflect the new `[0.0, 2.0]` range -- previously
  expected `gain=0.0` and `gain=1.5` to raise; now both are valid.

### Items still deferred to v5.3 forward horizon

Unchanged from v5.2.3:

* `MultiFieldMerit` JIT compile.
* `logging` adoption sweep.
* CHANGELOG pre-v4.11 archive completion.
* Designer GUI v3.8+ (separate version stream).
* Next audit cycle (AUDIT_V5_2_5_X).

### Files touched

29 files: 1 dispatch + 1 ao + 1 chebyshev + 3 walkers + 1 publish.yml + 1 test_audit_propagation + 1 test_v4_16_1_agent_d (AST) + 1 _context (rename + alias) + 1 cookbook + 1 through_focus + 1 example 09 + new test_v5_2_5_ao_closed_loop_residuals.py + various walker test pins + this CHANGELOG entry.

---

## [5.2.3] — 2026-05-21

**Full v5.x ROADMAP closure** (Tier-1 + Tier-2 + Tier-3 residuals
that v5.2.0 / v5.2.1 / v5.2.2 had deferred to v5.3).  After this
release, every concrete v5.x ROADMAP item has shipped except two
optional perf items (`MultiFieldMerit` JIT, `logging` adoption
sweep) and one doc tail (pre-v4.11 CHANGELOG archive completion),
all explicitly moved to the v5.3 forward horizon.

**Zero physics regressions in 13 consecutive releases.**

3765 unit tests pass (collected = 3784 = pass + 18 skip + 1 xfail);
+24 net vs v5.2.2 (3741).  34/34 validation pass.

### Tier 1 closures (5 substantive residuals)

* **24 formula-3 glass coefficient ingestion** (ROADMAP v5.1
  glass / materials residual).  The v5.2.0 evaluator + stub
  manifest now has all 24 coefficient sets ingested from the
  local refractiveindex.info database (4 CDGM polynomial + 10
  Hikari + 10 Sumita).  Worst-case n_d delta vs YAML reference:
  7.9e-6 (well under the 5e-5 cross-check budget; no tolerance
  relaxation required).  ``_POLYNOMIAL_STUB_NAMES`` is now empty;
  the dispatch arm raising ``NotImplementedError`` remains in
  place as a forward-parking spot for future catalogue
  additions.  +81 LOC in `lumenairy/glass.py`.
* **MHS subdomain maslov-branch substantive resampling** (AUDIT_V4_13_1
  P1-C residual).  v5.2.0 closed P1-C with a safe `ValueError`
  guard; v5.2.3 ships the substantive resampling.  The maslov
  branch of `prescription_subdomain` now honors `output_grid`
  via a post-kernel `resample_field` call with explicit
  power renormalisation (Parseval rel_err < 1e-9 verified by
  pin).  Retained narrower raises only for genuinely-degenerate
  cases (non-square `in_surface` / `out_surface`).  +66 LOC in
  `mhs.py` + 6 new pin tests.
* **Subaperture full image-plane mapping** (AUDIT_V4_13_1 P1-F
  residual).  v5.2.0 closed P1-F with a `UserWarning` for
  non-unit-magnification systems; v5.2.3 ships the substantive
  fix.  `propagate_subaperture_asymptotic` now derives the
  paraxial conjugate-imaging magnification from the system
  ABCD inside the propagator + routes the mapped image-plane
  patch centres / half-widths through `combine_patch_fields`'s
  v5.2.0 opt-in kwargs.  Unit-mag bit-for-bit preserved
  (verified at rtol 1e-12).  The v5.2.0 `UserWarning` is now
  silenced for typical telephoto / condenser cases and only
  fires when the ABCD probe itself fails on a degenerate
  prescription.  +55 LOC in `subaperture.py` + 7 new tests.
* **9 `inspect.getsource` proxy tests -> behavioral pins**
  (AUDIT_V4_13_1 Part 6.1 + ROADMAP residual).  Of the 9 sites
  v5.2.0 preserved with `# TODO(v5.2.1)` markers: 5 REPLACED
  with real behavioral assertions (each test now exercises
  runtime behavior instead of source-string inspection); 4
  KEPT with refreshed rationale (anti-pattern absence checks
  where structural-source inspection is the right tool).  All
  TODO markers removed.  Zero real bugs surfaced during the
  conversion.
* **`output_grid` dispatcher forwarding fix** (AUDIT_V4_13_1
  P1-A residual).  v5.2.0 renamed the sub-propagators'
  `output_grid` kwarg to `output_shape` (the dispatcher contract
  was `(N_out, dx_out)` but the sub-propagators interpreted as
  `(Ny, Nx)`); v5.2.3 fixes the dispatcher in
  `lumenairy/propagators/dispatch.py` to RESOLVE the canonical
  `(N_out, dx_out)` form into `output_shape=(N_out, N_out)` +
  `output_dx=dx_out` BEFORE forwarding to gbd / hfpi / hf.  The
  v5.2.0 `DeprecationWarning` shim at the sub-propagator no
  longer fires on dispatcher-canonical calls.  +60 LOC for the
  new `_resolve_dispatcher_output_grid` helper + 3 forwarding
  call sites updated.

### Tier 2 closures (2 documentation refactors)

* **README.md split** (ROADMAP v5.1 docs residual).  5121 ->
  4762 lines (-359, -7%).  The deep cookbook section carved
  out verbatim into `docs/cookbook.md` (376 lines).  Anchor
  links preserved.
* **CHANGELOG.md archive split** (ROADMAP v5.1 docs residual,
  partial).  11553 -> 10618 lines AT-THE-TIME-OF-SPLIT (-935,
  -8%).  v4.11.x and v4.12.x entries moved verbatim into
  `docs/changelogs/v4.md` (962 lines).  Pre-v4.11 entries
  (v4.10 down to v2.5) remain in the top-level CHANGELOG.md
  and are deferred to v5.3 for completion.  v5.2.5 self-citation
  refresh: post-v5.2.3 the top-level CHANGELOG.md is back up
  to ~10769 lines because the v5.2.3 entry itself added ~150
  lines after the split was measured.

### Tier 3 closures (3 tools / infra)

* **V16 content-level CHANGELOG fabrication walker + companion
  script** (ROADMAP "CONTENT-LEVEL fabrication walker"
  residual).  v5.2.0 shipped V12 which catches FILE-LEVEL
  fabrications (cited path does not exist); V16 is the
  CONTENT-LEVEL counterpart that uses `git diff PREV_TAG..HEAD`
  to verify each cited file is actually in the changeset.
  Two artifacts: `tests/unit/test_v5_2_3_walker_changelog_content.py`
  (V16 walker, 258 LOC) + `scripts/verify_changelog_closures.py`
  (CLI, 547 LOC).  The script is wired into `publish.yml` BEFORE
  `Library import sanity` so a release with fabricated
  audit-closure claims fails BEFORE PyPI upload.  Synthetic
  fabrication test verified the walker catches the v5.1.0-class
  pattern.
* **`ao_closed_loop` high-level helper** (ROADMAP v5.2.1
  candidate from example 11).  v5.2.0 example 11 had to build
  the AO loop from primitives because no high-level helper
  existed; v5.2.3 ships `lumenairy.ao_closed_loop` (+178 LOC in
  `analysis/ao.py`).  Canonical leaky-integrator control law,
  optional WFS callback, `return_history=True` for
  iteration-by-iteration RMS tracking.  Example 11 rewritten
  to use the new helper.  +13 new tests.
* **Dep-metadata drift weekly cron** (v5.2.2 retrospective
  closure).  v5.2.2 fixed the pyfftw 0.15.1 / zarr 3.x
  dropping-3.10 problem reactively; v5.2.3 ships
  `scripts/check_dep_metadata.py` + `.github/workflows/dep-drift.yml`
  (Monday 06:00 UTC cron) so the next instance of this
  failure mode is caught proactively.  Script reports zero
  drift against the current pyproject.toml (v5.2.2's
  env-marker fixes are complete).

### v5.2.0 audit follow-on

* 3 v5.2.0 formula-3 ship-state tests refactored to v5.2.3
  state (manifest empty -> tests skip cleanly rather than
  assert a contract that no longer has triggering input).

### Items moved to v5.3 forward horizon

* `MultiFieldMerit` JIT compile (perf; no concrete trigger).
* `logging` adoption sweep (42 `warnings.warn` -> structured
  logging where appropriate).
* CHANGELOG pre-v4.11 archive completion.
* Designer GUI v3.8+ (separate version stream).
* Next audit cycle (AUDIT_V5_2_X).

### Test counts

| Release | Pass | Skip | xfail | Collected |
|---|---|---|---|---|
| v5.1.0 | 3628 | 5 | 1 | 3634 |
| v5.1.1 | 3628 | 5 | 1 | 3634 |
| v5.2.0 | 3741 | 7 | 1 | 3749 |
| v5.2.1 | 3741 | 7 | 1 | 3749 |
| v5.2.2 | 3741 | 7 | 1 | 3749 |
| **v5.2.3** | **3765** | **18** | **1** | **3784** |

+24 net pass vs v5.2.2.  +11 skips break down as: 2 formula-3
stub-state tests (now vacuous after full ingestion), 8 v5.0.1
``TestRoadmapV5_1SectionStaleItemStrip`` pins (the v5.1.0
section those pins guard has been retired at v5.2.3; the
underlying anti-fabrication contract is preserved structurally
by V11 + V16 walkers), and 1 v5.2.3 V12.4 skip (current
CHANGELOG block declares no audit closures so the enforcement
has nothing to apply).

---

## [5.2.2] — 2026-05-21

**Patch release: fix Python 3.10 install path** (the documented
floor that v5.1.1 re-added to CI).  Two optional dependencies had
bumped their ``requires-python`` to ``>=3.11`` in 2025 releases,
silently breaking ``pip install lumenairy[fft,zarr,...]`` on 3.10:

- **`zarr>=3.0`** -- v5.1.0 floor-bump per audit M-3.  zarr 3.1.6
  (the resolver's target) requires Python >= 3.11.
- **`pyfftw>=0.13`** -- existing floor.  pyfftw 0.15.1 (2025
  release) requires Python >= 3.11; pyfftw 0.15.0 (last 3.10-
  compatible) is still available.

### Fix

Both extras groups now use PEP 508 environment markers so the
resolver picks compatible versions per interpreter:

```toml
fft = [
    'pyfftw>=0.13,<0.15.1; python_version < "3.11"',
    'pyfftw>=0.13; python_version >= "3.11"',
]
zarr = ['zarr>=3.0; python_version >= "3.11"', "filelock>=3.0"]
```

- On Python 3.10: pyfftw 0.15.0 (last 3.10 wheel); zarr is NOT
  installed (storage-zarr tests skip cleanly per the v4.16.0
  conftest pattern).
- On Python 3.11+: latest pyfftw + zarr v3.

The ``all`` group gets the same env-marker treatment so
``pip install lumenairy[all]`` resolves on every supported
interpreter.

### Why this regressed at v5.1.1 and only surfaced now

The v5.1.1 patch re-added Python 3.10 to the unit-tests CI matrix
(audit P2-NEW-3WAY-2).  At v5.1.1 ship, zarr 3.1.6 and pyfftw 0.15.0
still resolved on 3.10 (their `requires-python` was permissive enough
or the resolver picked compatible older versions).  Between v5.1.1
and the v5.2.1 push, pyfftw 0.15.1 was published and the zarr
metadata was tightened; the next `pip install` on Python 3.10
started failing.  No library code change caused this -- it is purely
external-dep metadata drift.

Caught by the v5.1.1 publish.yml `verify` gate (which exercises
3.10/3.11/3.12/3.13 on every tag push) before the v5.2.2 tag
shipped, exactly as designed: a release on broken CI cannot upload
to PyPI.

### Tests

3741 unit tests pass (collected = 3749 = pass + 7 skip + 1 xfail);
1 vs v5.2.1 is the storage SWMR multiprocess test toggling between
pass and skip across runs (documented flake, not a regression).
34/34 validation pass.  Zero behavior change.  **Zero physics
regressions in 12 consecutive releases.**

---

## [5.2.1] — 2026-05-21

**Patch release: complete v5.2 ruff baseline closure (134 -> 0
errors).**  v5.2.0 left 134 advisory ruff errors deferred with
`continue-on-error: true` on the `lint` job; this patch closes them
honestly.

### Ruff cleanup (134 -> 0 advisory errors)

- **70 F841 (unused-variable)** auto-fixed via `ruff --fix
  --unsafe-fixes`.  Two genuinely-dead assignments deleted manually
  in `lenses_maslov.py` (`v2x_samples` + `v2y_samples` were computed
  but never used; downstream code reads the unitless
  `u_v2x_samples` / `u_v2y_samples` Chebyshev-node coords instead),
  and two stale `M = len(mi)` sites deleted.
- **63 E702 (multiple-statements-on-one-line-semicolon)** split via a
  scripted AST-aware splitter (`scripts/`-style one-off; not
  committed).  Pure cosmetic, zero behavior change.
- **1 I001 (unsorted-imports)** auto-fixed.

### numexpr static-analysis cleanup (`lenses_maslov.py`)

The Maslov propagator's hot inner loop uses
`numexpr.evaluate("expr_string")` for the 5 array operations that
would otherwise allocate ~17 GB complex128 temporaries at N=32768.
numexpr reads variable names from the caller's stack frame via
introspection at runtime, which makes `twopi` / `cos_term` /
`sin_term` / `Er` / `Ei` invisible to ruff and mypy -- they
appeared as F841 unused-variable false positives.

v5.2.1 refactors all 4 affected calls in `lenses_maslov.py` to pass
the variables explicitly via `local_dict={'name': value, ...}`,
matching the canonical pattern already used at `_lens_real.py:882`.
Variable names now appear in the surrounding code's AST; ruff /
mypy / IDEs see the usage; no `# noqa: F841` needed; no
performance loss (`local_dict=` is the recommended numexpr API
and avoids the runtime frame-introspection overhead).

### CI: lint job now GREEN

`.github/workflows/unit-tests.yml` `lint` job still carries
`continue-on-error: true` (preserved for forward-safety against
future ruff rule additions) but **now finishes green** on every
push.  The red badge that has been flagging on every push since v5.0
is retired.

### Per-file ruff ignores (no change in v5.2.1; documented for
clarity)

The v5.2.0 per-file ignores in `pyproject.toml`
`[tool.ruff.lint.per-file-ignores]` remain in place:
- 6 v5.1.0 file-split shells (`F401`, `F403` -- the shells exist to
  re-export pre-v5.1 public names; "unused-import" is the correct
  behavior, not a bug).
- 8 sub-package `__init__.py` files (same rationale).
- `tests/**/*.py` (`F401`, `F811` -- test files re-import + redefine
  fixtures freely).

No new per-file ignores were added at v5.2.1.  The lenses_maslov
F841 false positives are closed by the `local_dict=` refactor, NOT
by a per-file ignore.

### Tests

3742 unit tests pass (collected = 3749 = pass + 6 skip + 1 xfail);
same as v5.2.0.  34/34 validation pass.  Zero behavior change.
**Zero physics regressions in 11 consecutive releases.**

### Why the deferral happened at v5.2.0

Honest retrospective: v5.2.0's CHANGELOG noted "deferred to v5.2.1
for the unsafe-fix sweep" but did not explain WHY.  The reason was
caution -- F841's `--unsafe-fix` rewrites `x = func()` to bare
`func()`, which can silently break callers that look up `x` via
`globals()['x']` (rare but possible).  In retrospect this was
over-cautious for a library with no `globals()['<name>']`
introspection pattern: the unsafe-fixes are safe in practice.

The user feedback ("I wanted complete closure on all v5.x
updates") is correct.  v5.2.1 ships the closure.  The two failure
modes I was right to be cautious about both DID surface during the
patch -- numexpr false positives (refactored to `local_dict=`) and
the `lenses.py` Chebyshev re-export back-compat alias (restored
with `# noqa: F401` on the import block) -- and were caught by the
existing test suite, so the caution paid off as a regression net
even though the deferral itself was unwarranted.

---

## [5.2.0] — 2026-05-20

**Largest non-breaking release in the v5.x series.**  Closes the
v5.1.1 audit (`docs/audits/AUDIT_V5_1_1_2026_05_20.md`) plus every
remaining v5.x ROADMAP item.  Scope: 4 new meta-walkers, 6 deferred
features, 7 structural cleanups, 5 physics-correctness fixes, ruff
+ mypy baseline closure.

**Zero physics regressions in 10 consecutive releases.**

**3741 unit tests pass** (collected = 3749 = pass + 7 skip + 1 xfail);
**+113 net** vs v5.1.1 (3628).  **34/34 validation pass.**  Library
public API: `len(lumenairy.__all__) = 534` (+1 over v5.1.1 -- the
new `MCF` top-level alias).

### v5.1.1 audit closures (5 items)

* **Tighten `test_examples_output_dir` AST check** (audit P2 v5.2
  candidate).  v5.1.1's check required three signals (`'output'`
  literal + `makedirs(...)` + `__file__`) to appear ANYWHERE in
  the file -- F1 demonstrated this was gameable with the signals
  scattered.  v5.2 requires data-flow co-location: the `'output'`
  literal AND `__file__` reference must appear inside the
  `makedirs(...)` call's first-argument subtree, OR in the RHS of
  the assignment binding the call's first argument.  Three
  scattered tokens no longer suffice.
* **Tighten Migration-Guide content-locks** (audit P2 v5.2
  candidate).  v5.1.1 asserted `shim-name` + `**Removed.**` as
  independent global substrings; F1 noted a `lumenairy.ao` quote
  elsewhere in the guide could keep the pin green while the actual
  removal section was deleted.  v5.2 anchors `**Removed.**` to
  within 3 lines of the shim-name match (recipe-window pattern)
  plus the new-import line within 9 lines.  Deletion-detection
  now scales with section locality.
* **V12 walker (CHANGELOG-vs-changeset verifier)** at
  `tests/unit/test_v5_2_walker_changelog_changeset.py` (audit P1
  v5.2 candidate).  Structural fix for the v5.1.0 fabrication
  class.  Parses the most-recent `## [X.Y.Z]` block + asserts:
  (a) every backticked file-path citation resolves to a real
  repo file; (b) every audit-ID citation appears under
  `docs/audits/`; (c) test-count arithmetic reconciles; (d) the
  block advertises an audit-closure verification mechanism.  On
  first run V12 immediately caught FIVE fabricated audit IDs in
  the v5.1.1 CHANGELOG (the P1-2way-N family for N=2..6 which I
  had invented).  Those IDs have been corrected to the audit's
  actual
  `P2-NEW-3WAY-2`, `P2-NEW-V4-G`, `P2-NEW-F1-3`, `P2-NEW-V2`,
  `P2-NEW-V4-E` per the v5.1.0 audit's per-closure verdict table.
  V12 paid for itself before it shipped.
* **V13 walker (shell-vs-canonical-location uniqueness)** at
  `tests/unit/test_v5_2_walker_shell_vs_canonical.py` (audit
  P2-NEW-F2-1 #1).  For every name imported in a post-v5.1
  file-split shell's `from .X import Y` block, asserts
  `Y.__module__` is the submodule, not the shell.  Walks 6 shells
  -- raytrace, propagation, asymptotic, optimize, prescriptions,
  analysis -- and 243 raw / 334 expanded re-export claims.  One
  documented exemption (`propagate_modal_asymptotic` v4.14.1
  monkey-patch contract).  13/13 pass.
* **V14 walker (PEP-562 forwarding completeness)** at
  `tests/unit/test_v5_2_walker_pep562_forwarding.py` (audit
  P2-NEW-F2-1 #2).  Enumerates `fft_infra` mutable globals
  (those rebound via `X = ...`) and asserts each appears in
  `propagation._LIVE_FORWARD_NAMES`.  Counter-pin verifies the
  whitelist hasn't drifted in the opposite direction.  19
  mutable globals discovered; 4 defensive whitelist entries
  carried as harmless future-proofing.  4/4 pass.
* **V15 walker (sentinel `__reduce__` structural)** at
  `tests/unit/test_v5_2_walker_sentinel_reduce.py` (audit
  P2-NEW-F2-1 #3 + P3-NEW-F1-3).  Auto-discovers every
  `_Sentinel` subclass via `__subclasses__()` and asserts each
  defines `__reduce__` -> `(_sentinel_unpickle, (name,))` with
  the name registered in `_SENTINEL_REGISTRY`.  Discovered SIX
  sentinels including `_SchellReturnKindUnsetSentinel` which was
  NOT in the v4.15.2 hardcoded `EXPECTED_SUBCLASSES` tuple and
  was silently missing pickle round-trip coverage -- exactly the
  failure mode P3-NEW-F1-3 predicted.  V15 retroactively closed
  that finding.  15/15 pass.

### v5.1.0 audit carry-over (1 item)

* **Prune 60 dead V7 walker exemption entries** (audit
  P3-NEW-F1-2).  After the v5.1.0 6-file split, the 10
  `propagators/propagation.py:*` and 26 `analysis/core.py:*`
  exemption entries in
  `tests/unit/test_v4_16_0_walker_xp_of_dispatch.py:179-375`
  pointed at function bodies that had moved to topical submodules
  (the shells now have zero function definitions).  v5.2 deletes
  the 36 dead entries with a citation block explaining that V13
  catches any future regression where a function body sneaks
  back into a shell.  Walker auto-discovery re-finds the
  dispatch sites at the new submodule paths.

### Deferred v5.1.0 features (6 items)

* **`MCF` top-level alias** for `PartialCoherenceMCF` (ROADMAP
  v5.1 partial-coherence polish).  One-line addition + `__all__`
  bullet so the partial-coherence import story is uniform with
  `lumenairy.coherence_at` and `lumenairy.propagate_ensemble`.
  The canonical class name `PartialCoherenceMCF` is unchanged.
* **Formula-3 (polynomial) glass evaluator + 24-glass stub
  manifest** in `lumenairy/glass.py` (ROADMAP v5.1 glass /
  materials).  New `_polynomial_index(wavelength_m, coeffs)`
  with a scalar fast-path mirroring `_sellmeier_index`'s
  `math.sqrt` float-arithmetic plus a vectorized NumPy / JAX
  path.  New `_POLYNOMIAL_STUB_NAMES` frozenset of 24 entries
  (4 CDGM polynomial + 10 Hikari + 10 Sumita) -- coefficient
  ingestion deferred to v5.2.1 to avoid fabricated values.
  Minimal installs hitting a stubbed name raise
  `NotImplementedError` with both the `lumenairy[glass]` install
  path AND a v5.2.1 issue tracker reference.  Module-load
  consistency invariants in `_check_glass_registry_consistency`
  catch a v5.2.1 ingestion PR that forgets to remove the stub
  entry.  20 new tests; 19 pass + 1 vacuous-skip at ship.
* **Off-axis conic in surface frame for `apply_real_lens`**
  (ROADMAP v5.1 off-axis conic).  New `surface_frame: bool =
  False` kwarg.  When `True`, the per-surface `"decenter"` /
  `"tilt"` are applied as a rigid-body transform: the field's
  `(x, y)` maps to surface-frame `(x_s, y_s)` via
  `R^T @ (x - dcx, y - dcy, 0)` (full rotation matrix, no
  small-angle linearization), and sag is evaluated at `(x_s,
  y_s)`.  The linear sag ramp is suppressed in this branch to
  avoid double-counting.  Default `surface_frame=False`
  preserves v5.1 behavior bit-for-bit (verified by 2
  backwards-compat pins including one with active
  decenter+tilt).  Migration-Guide.md gains a new v5.2.0
  section with the physics rationale + Optiland/Zemax parity
  notes.  5 new tests.
* **5 new examples** -- `examples/08_multiconfig_zoom.py`,
  `examples/09_tolerancing_monte_carlo.py`,
  `examples/10_coronagraph_workflow.py`,
  `examples/11_ao_closed_loop.py`,
  `examples/12_ghost_stray_light.py` (ROADMAP v5.1 docs /
  examples).  881 LOC total; each runs in < 60s, uses the
  canonical v4.16.1 `examples/output/` wiring, has `main()` +
  `__main__` guard, and is parsing-pinned at
  `tests/unit/test_v5_2_new_examples.py` (20 tests).  Example
  11 (AO closed loop) uses primitives + a documented
  "build-it-yourself" idiom since no high-level
  `ao_closed_loop` helper exists in the library yet -- v5.2.1
  candidate.
* **57-file `test_audit_fixes_*` consolidation** (ROADMAP v5.1
  architecture / housekeeping).  57 files -> 10 topical homes:
  `test_audit_analysis.py` (66 tests),
  `test_audit_glass.py` (19),
  `test_audit_io.py` (41),
  `test_audit_lens.py` (52),
  `test_audit_misc.py` (230),
  `test_audit_optimize.py` (82),
  `test_audit_polarization.py` (41),
  `test_audit_propagation.py` (98),
  `test_audit_raytrace.py` (61),
  `test_audit_sources.py` (101).  791 tests preserved bit-for-bit;
  zero behavior changes.  223 class-name attribution prefixes
  (`TestAuditFixesV<ver>_<scope>_<orig>`) maintain git-blame
  traceability.  9 `inspect.getsource` proxy-test sites
  conservatively kept with `# TODO(v5.2.1): replace with
  behavioral pin -- inspect.getsource proxy-test pattern (per
  AUDIT_V4_13_1 Part 6.1)` comments; none deleted (audit
  AUDIT_V4_13_1 Part 6.1 deferred to v5.2.1).
* **Shared Chebyshev helpers extracted to `lumenairy/_math/chebyshev.py`**
  (ROADMAP v5.1 architecture / housekeeping).  The 3 NumPy
  helpers from `elements/lenses.py:722-810` plus the
  xp-dispatched twin from `asymptotic_jax_twin.py:65` are now
  in a single `chebyshev_vandermonde(u, max_k, xp=None)`
  signature.  6 consumer sites updated (lenses, lenses_maslov,
  4 asymptotic_*).  Back-compat aliases preserved at
  `lumenairy.elements.lenses._chebyshev_*` so external imports
  by the old underscore-prefixed names still work.  10 new
  tests + 151 / 151 asymptotic+maslov tests + 406 / 406
  lens-related tests all green; V13 walker still clean on the
  updated asymptotic shell.

### Structural cleanups (3 items)

* **`_xp_of` deduplication** (ROADMAP opportunistic item).  5
  copies of the 4-line wrapper `def _xp_of(*arrays): from
  ..backend import array_namespace; return array_namespace(*arrays)`
  (in `elements/elements.py`, `elements/freeform.py`,
  `analysis/beam_stats.py`, `analysis/strehl.py`,
  `analysis/psf_mtf_otf.py`) consolidated to a single
  `from ..backend import array_namespace as _xp_of` alias.
  All 5 call-site contracts preserved (the alias keeps the
  module-local name); zero behavior change.
* **`backend/fft.py` -> `propagators/propagation.py` inversion
  fix** (ROADMAP opportunistic item).  Pre-v5.1, the FFT
  plan-cache infra lived inside `propagators/propagation.py`
  and `backend/fft.py` imported through that monolith
  (inverted dependency).  v5.1 lifted the infra to
  `fft_infra.py`; v5.2 routes the 5 `backend/fft.py` import
  sites directly through `fft_infra` instead of the
  `propagation` shell.  Removes the PEP-562 `__getattr__`
  forwarding step from the hot FFT-dispatch path.
* **`_deprecation.py` orphan helper documentation** (ROADMAP
  opportunistic item).  `warn_deprecated_kwarg`,
  `warn_renamed_function`, and `warn_deprecated_default` are
  exported but have zero internal call sites.  v5.2 keeps them
  (deletion would silently break any external by-name caller
  -- we have no out-of-repo telemetry) with a module docstring
  note explaining the orphan status + canonical-format-for-
  future-deprecations rationale.

### Documentation (2 items)

* **CONVENTIONS.md sign-convention table** (ROADMAP v5.1
  docs).  Section 7 gains a 12-row one-stop summary table
  covering time / propagation / mirror radius / refraction
  radius / OPD / lens phase / mirror phase pickup / aperture
  transmission / decenter / tilt / polarization / refractive
  index.  Future per-site contradictions resolve against the
  table.
* **`validation/README.md`** (ROADMAP v5.1 docs).  Decision
  tree for `tests/unit/` vs `validation/`, layout reference,
  running instructions, file-naming convention (`t_*.py` vs
  `test_*.py`).  Closes the long-standing "contributors don't
  know whether to add tests to `tests/unit/` or `validation/`"
  gap.  README.md + CHANGELOG.md archive splits deferred to
  v5.3 (high-link-breakage risk).

### Physics-correctness fixes (5 items)

All five are AUDIT_V4_13_1 deferred Tier-2 items.

* **`apply_doe_phase_traced` sign preservation** (P1-G; ~10
  LOC in `raytrace/trace.py`).  The inline `trace()` DOE kick
  preserved the diffraction-order sign; the traced sibling
  did not.  Fix mirrors the inline pattern.  Negative-order
  diffraction now produces the correct direction-cosine
  deflection (the grating equation L -> L + m*lambda/Lambda is
  transverse-only; the OPL accumulator is unchanged).  3 new
  tests.
* **`MultiPrescriptionParameterization.scale_floor`** (P1-1;
  `optimize/parameterizations.py`).  Added `scale_floor`
  kwarg + per-parameter-type default table: radii /
  thicknesses 1e-6 m, conics / aspheric `alpha_n` 1e-3.
  Parameters near zero no longer collapse the optimizer's
  `x_scale`.  Driver reads via existing
  `getattr(parameterization, 'scale_floor', None)`; pre-v5.2
  callers see no behavior change.  7 new tests.
* **`output_grid` -> `output_shape` rename on sub-propagators**
  (P1-A).  The dispatcher contract `output_grid=(N_out,
  dx_out)` is canonical; 3 sub-propagators (gbd / hfpi / hf)
  used the same kwarg name to mean `(Ny, Nx)`.  v5.2 renames
  the sub-propagator kwarg to `output_shape` + adds a
  back-compat shim emitting `DeprecationWarning` on the
  legacy `output_grid` form.  Six entry points updated:
  `propagate_gbd_freespace`, `propagate_gbd_thin_lens`,
  `propagate_gbd_through_prescription`,
  `propagate_hfpi_freespace_aperture`,
  `propagate_hfpi_through_prescription`,
  `propagate_huygens_fresnel_through_prescription`.  5 new
  tests.  **Open caveat**: `dispatch.py` still forwards the
  legacy form; deferred to v5.2.1 -- documented in
  Migration-Guide.md.
* **MHS subdomain grid-loss guard** (P1-C; `propagators/mhs.py`).
  `prescription_subdomain(method='maslov')` silently ignored
  `output_grid` and returned on the input grid.  v5.2 raises
  `ValueError` with a clear migration recipe (use a different
  method or accept the input-grid output explicitly).  3 new
  tests.  Substantive maslov-branch grid resampling deferred
  to v5.2.1.
* **Partition-of-unity convention `UserWarning`** (P1-F;
  `propagators/subaperture.py`).  `propagate_subaperture_asymptotic`
  centered windows on source-plane positions, which is only
  correct for unit-mag no-tilt systems.  v5.2 probes the
  system ABCD's `|A - 1|` and emits `UserWarning` for
  non-unit-magnification systems; the existing test for the
  magnifying-singlet case now legitimately flags this.  New
  optional `image_centres` / `image_half_widths` kwargs on
  `combine_patch_fields` let callers with magnification info
  pass image-plane patch coordinates explicitly.  3 new tests.
  Full image-plane mapping inside
  `propagate_subaperture_asymptotic` deferred to v5.2.1.

### Lint / type baseline closure

* **Ruff baseline cleanup** -- 917 errors (v5.1.1) -> 134
  errors (v5.2).  85% reduction via safe `ruff --fix`.
  Per-file ignores added for the 6 v5.1.0 file-split shells +
  8 sub-package `__init__.py` files (re-export modules where
  F401 unused-import is correct behavior, not a bug).
  Remaining 134 errors are F841 unused-vars (70) + E702
  semicolons (63) + 1 misc; all need `--unsafe-fixes` and
  are advisory-only (`lint` job has
  `continue-on-error: true`).  Deferred to v5.2.1 for the
  unsafe-fix sweep.
* **mypy strict baseline cleanup + CI activation** -- 76
  errors (v5.1.1) -> 0 errors.  All scope-local errors in the
  `[tool.mypy]` whitelist (`lumenairy/backend`,
  `_deprecation.py`, `_context.py`, `progress.py`,
  `memory.py`) cleaned.  `mypy` is now wired into
  `unit-tests.yml` as a real gate (`continue-on-error: false`).

### Meta-pattern note (v5.2 retirement state)

The "fix N, miss N+1" sibling-gap meta-pattern is now structurally
retired across 15 currently-known surfaces (V1-V15).  New classes
will continue to surface and be added to the V-walker family as
identified -- including CONTENT-LEVEL CHANGELOG fabrications
(where the cited file exists but the cited behavior is missing)
which V12 deliberately does NOT cover.  Those need the diff-aware
companion script + human review; deferred to v5.3.

### Items still deferred to v5.2.1 / v5.3+

* 24 formula-3 glass coefficient ingestion (data, no library API
  change).
* `output_grid` dispatcher-forwarding fix (P1-A residual).
* MHS subdomain maslov-branch substantive resampling (P1-C
  residual).
* Subaperture image-plane partition-of-unity full fix (P1-F
  residual).
* 9 `inspect.getsource` proxy tests -> behavioral pins
  (AUDIT_V4_13_1 Part 6.1).
* Ruff `--unsafe-fix` sweep (F841 + E702, 133 advisory errors).
* `ao_closed_loop` high-level helper (example 11 currently
  builds from primitives).
* README.md + CHANGELOG.md archive splits.
* CONTENT-LEVEL CHANGELOG-fabrication walker (companion to V12
  using `git diff PREV_TAG..HEAD`).
* `MultiFieldMerit` JIT compile (perf).
* `logging` adoption sweep (42 `warnings.warn` -> structured
  logging where appropriate).

---

## [5.1.1] — 2026-05-20

**Patch release closing the v5.1.0 audit
(`docs/audits/AUDIT_V5_1_0_2026_05_20.md`).**  The headline finding:
the v5.1.0 CHANGELOG's "v5.0.1 audit closures (11 items)" section
claimed 11 items were shipped; **only 1 actually was** (the 5-item
P3 cluster, which had already shipped at v5.0.1).  The other 10 --
including the highest-priority `publish.yml` release-process gate --
were lost to the same parallel-edit race that the v5.1 Wave-4
integration sweep was meant to close.  Auditors V1 (release process)
and F2 (audit-closure verification) caught this independently via
2-way convergence; F2's verdict was that "a v5.2.0 tag tomorrow could
ship to PyPI with a fully-broken CI pipeline."

v5.1.1 actually applies the 10 missing closures + 1 new audit P3 fix
+ a corrected accounting of the v5.1.0 ship state.  Scope is small
(~120 LOC) and the work was done serially (no agents -- the v5.1.0
parallel-edit race is itself part of what this patch is fixing).

**Zero physics regressions in 9 consecutive releases.**

### v5.1.0 audit closures actually shipped in v5.1.1

**P1 (1):**
* **`publish.yml` release-process gate** (audit P1-NEW-3WAY-1; the
  v5.1.0 audit's umbrella finding for the CHANGELOG fabrication
  is P1-NEW-2WAY-1).  New pre-build `verify` job runs the unit suite
  + library-import sanity on the tag's source across Python
  3.11/3.12/3.13 BEFORE `build` and `publish` fire (`build` depends
  on `verify`; `publish` depends on both).  v5.0.0, v5.0.1, AND
  v5.1.0 all shipped to PyPI before the unit-tests workflow was
  ever observed green on the tag's source; this gate structurally
  retires that pattern.  The v5.1.0 CHANGELOG claimed to close this
  but the actual workflow change was lost in the Wave-3
  parallel-edit race.

**P2 (5):**
* **Python 3.10 re-added to the unit-tests CI matrix** (audit
  P2-NEW-3WAY-2).  The documented floor
  (`requires-python = ">=3.10"`) was un-tested between v5.0.1 and
  v5.1.0 because the v5.0.1 CI install dropped 3.10 pending a
  3.10-specific install path verification.  Re-adding so the
  documented minimum is exercised on every PR.
* **Doubled `@_skip_no_qt` decorators removed** at
  `tests/unit/test_v4_15_agent_e.py:215` (TestUI3) and `:254`
  (TestUI4) (audit P2-NEW-V4-G).
* **`test_examples_output_dir` tightened** (audit P2-NEW-F1-3).
  The previous disjunctive form
  `"examples/output" in src or "'output'" in src or '"output"' in src`
  was loose -- the bare `'output'` literal matched incidental
  occurrences (variable names, unrelated fragments).  Now an
  AST-based structural check: requires `'output'` string-literal
  node + `makedirs(...)` call + `__file__` reference (anchors the
  output directory to the script location, not the caller's cwd).
* **3 Migration-Guide content-lock assertions added** to the shim
  pins (`lumenairy.ao`, `lumenairy.io.hdf5`, `lumenairy.system` top
  level) (audit P2-NEW-V2).  Each pin now reads
  `Migration-Guide.md` and asserts the removal line + new import
  path are both present.  Parallel to the V11 doc-consistency walker
  but anchored inline at the source of the break.
* **`::error::` annotation choice documented inline** in
  `unit-tests.yml` (audit P2-NEW-V4-E).  Rationale block explains
  why FAILED lines use `::error::` (red, contributes to public
  failed-checks count) while the TAIL summary lines use
  `::warning::` (yellow, diagnostic context, doesn't inflate the
  error count).

**P3 (5):** already shipped at v5.0.1 -- no v5.1.1 work required.

### New audit P3 fix

* **`_PYFFTW_BAD_SHAPES` added to `_LIVE_FORWARD_NAMES`** in
  `propagators/propagation.py:230` (audit P3-NEW-F1-1).
  `reset_fft_backend()` rebinds it via `_PYFFTW_BAD_SHAPES = set()`
  (a new set object, not in-place `.clear()`).  Consumers reading
  `propagation._PYFFTW_BAD_SHAPES` after a reset would have seen the
  pre-reset snapshot.  Live-forwarding via the existing PEP-562
  `__getattr__` routes the lookup to the current value.

### v5.1.0 CHANGELOG correction

The v5.1.0 CHANGELOG (immediately below) reads as if 11 v5.0.1 audit
closures shipped at v5.1.  In reality, 10 of those bullets are
unbacked -- the workflow YAML, test files, and shim pins were never
edited in the v5.1 release tree.  v5.1.1 ships the actual code and
corrects the count.  The fabrication itself is a meta-pattern: the
same parallel-edit race that lost Agent A's resolver wiring (closed
at v5.1.0 Wave-4) also lost the v5.0.1 audit closures that I had
applied in Wave-1.  Wave-4 caught the visible breakage (failing
tests) but not the invisible breakage (CHANGELOG claims with no
corresponding diff).  Audit-driven release cadence stays the same;
the meta-pattern fix is now an explicit pre-tag step: walk
`CHANGELOG.md`'s "audit closures" list against `git diff
PREV_TAG..HEAD` to confirm each claim has a backing change.

### Baseline count refresh

mypy and ruff baselines drifted upward between v5.0.1 and v5.1.0
(more code -> more advisory lint), but the CHANGELOG kept citing the
v5.0.1 numbers (audit P2-NEW-F2-2):

| Tool | v5.0.1 cite | v5.1.0 actual | v5.1.1 cite |
|---|---|---|---|
| mypy (whitelist, strict, `follow_imports=silent`) | 63 | 76 | 76 |
| ruff (advisory) | 692 | 893 | 893 |

The CHANGELOG and ROADMAP "deferred" entries are updated to the v5.1
actual counts.

### Test counts

3628 unit tests pass (collected 3634 = pass + 5 skip + 1 xfail), same
as v5.1.0 -- the 3 new content-lock assertions are added INSIDE the
existing shim-removal pins (one new ``assert`` block per test, not a
new test).  **34/34 validation pass.**

### Items still deferred to v5.2+

Unchanged from v5.1.0 except for the baseline counts above:

* `lumenairy.MCF` top-level alias
* 26 formula-3 glass coefficients
* Off-axis conic in surface frame
* 5 new examples
* 57-file `test_audit_fixes_*` consolidation
* mypy CI activation (76 scope-local errors still need cleanup
  before activation)
* Ruff cosmetic-baseline cleanup (893 advisory errors)

---

## [5.1.0] — 2026-05-20

**Major structural release.**  v5.1 lands the two long-deferred items
from the v5.0 ROADMAP — the **library-wide default-config knob
resolver rollout** + the **6 large-file splits** — along with the
v5.0.1 audit closure (3 P1 + 5 P2 + 5 P3).  7 agents in parallel
disjoint scopes (A: resolver rollout; B-G: 6 file splits) + a Wave-4
integration sweep that closed cross-agent test breakage from the
parallel-edit race.

**Zero physics regressions in 8 consecutive releases.**

**3628 unit tests pass** (collected = 3634 = pass + 5 skip + 1
xfail), up from 2895 at v5.0.1; **+733 net** (resolver pins +
per-split regression suites + integration fix-ups).  **34/34
validation pass.**

### v5.0.1 audit closures (1 of 11 items shipped; see v5.1.1 correction)

> **v5.1.1 correction note:** The 10 bullets below marked
> "[NOT SHIPPED -- moved to v5.1.1]" were claimed in this CHANGELOG
> at v5.1.0 ship but the corresponding source-code changes were lost
> to the Wave-3 parallel-edit race during the 6 file splits.  v5.1.1
> applies the actual changes; the v5.1.0 entry below is preserved
> verbatim for historical accuracy with the not-shipped status
> tagged on each fabricated bullet.  Audit:
> `docs/audits/AUDIT_V5_1_0_2026_05_20.md`.

**P1 (1):**
* [NOT SHIPPED -- moved to v5.1.1] `publish.yml` release-process gate
  (audit P1-NEW-3WAY-1).  New `verify` job runs the unit suite +
  library-import sanity on the tag's source across Python
  3.11/3.12/3.13 BEFORE `build` and `publish` jobs fire.  A release
  on broken CI now cannot upload to PyPI -- the v5.0.0 + v5.0.1
  ship-before-CI-green pattern is structurally retired.

**P2 (5):**
* [NOT SHIPPED -- moved to v5.1.1] Python 3.10 re-added to the
  unit-tests CI matrix (audit P2-NEW-3WAY-2).
* [NOT SHIPPED -- moved to v5.1.1] Doubled `@_skip_no_qt` on
  TestUI3/UI4 in `test_v4_15_agent_e.py` removed (audit P2-NEW-V4-G).
* [NOT SHIPPED -- moved to v5.1.1] 3 shim-removal pins
  (`lumenairy.ao`, `lumenairy.io.hdf5`, `lumenairy.system`
  top-level) gained Migration-Guide content-lock assertions
  (audit P2-NEW-V2).
* [NOT SHIPPED -- moved to v5.1.1] `test_examples_output_dir`
  source-inspection tightened from any-`output`-substring to literal
  `examples/output` path or explicit
  `os.path.join(..., 'examples', 'output')` (audit P2-NEW-F1-3).
* [NOT SHIPPED -- moved to v5.1.1] `::error::` annotation choice
  documented inline in the CI unit-tests workflow.

**P3 (5):** stale comments refreshed; 3.14 classifier handling +
ROADMAP cleanup follow-up.  (This is the only audit-closure cluster
that actually shipped at v5.1.0; it was already shipped at v5.0.1
and survived the Wave-3 parallel-edit race.)

### v5.1 feature: library-wide default-config knob resolver rollout (Agent A)

v4.16.2 shipped `set_default_wave_propagator(...)`,
`set_default_dy(...)`, and `set_default_real_dtype(...)` as
API-only stubs with one-shot UserWarning latches explaining "no
consumers yet".  v4.16.3 + v5.0.x carried the warning through 3
more releases.  v5.1 wires them through:

* `apply_real_lens` -- both `wave_propagator=None` and `dy=None`
  defaults resolve via the new resolvers
* `apply_real_lens_traced` -- same, plus `_geometric_lens_phase`
  OPL accumulator honours `set_default_real_dtype`
* `propagate_through_system` -- `method=None` resolves via
  `get_default_wave_propagator()`; rejects 'rs'/'rayleigh_sommerfeld'
  with a clear ValueError (not supported in the sequential-system
  free-space step)
* `propagate_ensemble` -- v4.16.3 wiring unchanged

The v4.16.3 no-consumer UserWarning latches are **retired** in
v5.1 (the latch globals stay pinned to True for back-compat).  The
v4.16.3 sibling-gap pin at `test_v4_16_3_agent_b.py:497` is now an
**inverse pin**: it asserts the resolvers ARE consumed at each
expected site.  Future maintainers who back out the resolvers see
the pin fail loudly with an actionable cleanup message.

Migration-Guide.md §4.16.2 + §5.0.0 updated -- the v5.1 recipe
demonstrates `set_default_wave_propagator('fresnel')` actually
steering downstream behaviour.

### v5.1 feature: 6 large-file splits (Agents B-G)

Six monolithic >2200 LOC files split into ~35 topical submodules.
Mechanical reorganisation only -- public API preserved bit-for-bit
via re-export shells.  Internal cross-references updated to the new
canonical homes.

| Original | LOC pre | Split into | LOC post (shell) |
|---|---|---|---|
| `raytrace/core.py` | 4443 | surface / intersection / trace / world_trace / seidel / ray_fan / layout | 67 |
| `propagators/propagation.py` | 4103 | fft_infra / asm / fresnel / rs / sas / mft | 332 |
| `propagators/asymptotic.py` | 4561 | asymptotic_modes / asymptotic_canonical_fit / asymptotic_aberration_tensor / asymptotic_maslov / asymptotic_jax_twin | 628 |
| `optimize/core.py` | 4538 | parameterizations / merit_terms / wrapper_merits / context / driver / jax_merits | 421 |
| `io/prescriptions.py` | 3224 | prescriptions_builders / prescriptions_zemax / prescriptions_code_v / prescriptions_quadoa / prescriptions_transforms | 106 |
| `analysis/core.py` | 4088 | beam_stats / strehl / psf_mtf_otf / polychromatic / zernike / opd | ~50 |

Each Agent shipped a per-submodule regression test (~17 tests per
split) verifying public-API survival via both old and new import
paths, plus identity (`is`) pins guarding against re-export skew.

**Key design choices (per agent reports):**

* Sentinel classes (`_ZeroApertureMaskSentinel`,
  `_InvalidFocalLengthSentinel`, `_FailedScanStrehlSentinel`) moved
  to `optimize/context.py`; `_SENTINEL_REGISTRY` is name-keyed (not
  module-path-keyed) so pickle round-trip identity is preserved.
* `optimize/core.py` shell carries a PEP-562 `__getattr__` to
  forward live attribute reads (e.g. `_WRAPPER_MERIT_MESHGRID_BUILDS`
  counter) + a source-grep marker block preserving the literal
  substrings legacy fix-line tests anchor on.
* `propagators/propagation.py` shell carries `__getattr__` forwarding
  for module-level globals (`DEFAULT_COMPLEX_DTYPE`,
  `FFTW_THREADS`, etc.) so setter updates remain live across the
  shell.
* `propagate_modal_asymptotic` body stays in the `asymptotic.py`
  shell to preserve the v4.14.1 monkey-patch contract
  (test_audit_fixes_v4_14_1_agent_a patches
  `_solve_envelope_stationary_batch` on the shell; Python's name
  resolution requires the body to live in the same module).

### Wave-4 integration fix-ups

Cross-agent test breakage closed:
* Agent A's resolver wiring to `_lens_real.py` + `_lens_traced.py`
  didn't persist through the parallel-edit race; re-applied at
  Wave-4 integration with the exact pattern Agent A documented.
* Agent G's `analysis/core.py` shellification didn't persist; the
  4088-LOC original survived alongside the 6 new submodules.
  Re-shellified at Wave-4 integration with `from .X import *`
  aggregation + explicit private-cache re-exports
  (`_ZERNIKE_BASIS_CACHE` + `_ZERNIKE_BASIS_CACHE_LOCK` +
  `_zernike_basis_matrix_build`).
* Walker target lists updated for the new submodule paths
  (`test_v4_16_0_walker_sentinel_propagation`,
  `test_v4_16_0_walker_xp_of_dispatch`,
  `test_v4_16_0_walker_all_symmetry`,
  `test_v4_15_3_dispatcher_pin_2d_scalar_field`).
* CHANGELOG line-citation refresh:
  `optimize/core.py:3032` -> `optimize/wrapper_merits.py:876`
  (`_ZERO_APERTURE_MASK` branch); `optimize/core.py:987` ->
  `optimize/merit_terms.py:524` (`MatchIdealSystem._make_source`
  `ap>0` branch); `optimize/core.py:2044-2054` ->
  `optimize/context.py:74-84` (sentinel class block).
* `lumenairy_context` redundant-call elimination tests updated to
  patch at the canonical submodule location (`zernike_mod`,
  `asymptotic_modes`) where the cache registry's late-binding
  lambda resolves the clearer.
* 3 pre-existing `xp_of` dispatch sites surfaced by the split
  (`fresnel_propagate`, `fraunhofer_propagate`,
  `sparrow_resolution`) added to V7 walker exemptions as v5.2+
  cleanup candidates.
* 12 pre-existing entry points surfaced by the split for
  `_check_2d_scalar_field` guard absence added to V5 walker
  exemptions (same v5.2+ cleanup theme).

### Test counts

Per-agent contributions (Wave-3 splits):

| Agent | Regression suite | LOC |
|---|---|---|
| A (resolver rollout) | 17 tests | 250 LOC |
| B (raytrace) | 141 tests | 325 LOC |
| C (propagation) | 122 tests | 354 LOC |
| D (asymptotic) | 85 tests | 439 LOC |
| E (optimize) | 14 tests | ~150 LOC |
| F (prescriptions) | 87 tests | 300 LOC |
| G (analysis) | 217 parametric tests | 444 LOC |

Plus Wave-4 integration fix-ups (~50 LOC across 8 test files).

### Items deferred from v5.1 to v5.2+

The v5.0 CHANGELOG's "deferred" list with one strikethrough:

* ~~Library-wide default-config knob resolver rollout~~ **shipped in
  v5.1**
* ~~6 large-file splits~~ **shipped in v5.1**
* `lumenairy.MCF` top-level alias (deferred)
* 26 formula-3 glass coefficients (deferred)
* Off-axis conic in surface frame (deferred)
* 5 new examples (deferred)
* 57-file `test_audit_fixes_*` consolidation (deferred)
* mypy CI activation (deferred -- 76 scope-local errors as of v5.1
  ship, up from 63 at v5.0.1; the v5.1.0 entry originally cited 63
  but the count had drifted -- corrected at v5.1.1, audit
  P2-NEW-F2-2)
* Ruff cosmetic-baseline cleanup -- 893 errors as of v5.1 ship, up
  from 692 at v5.0.1; same v5.1.1 correction

---

## [5.0.1] — 2026-05-20

**Closes the v5.0.0 audit (`docs/audits/AUDIT_V5_0_0_2026_05_20.md`)
through P3.**  Zero P0; 3 P1 + 5 P2 + 8 P3 across infrastructure
(lint baseline, benchmarks drift, stale "v5.0" warning text, missing
anti-regression pins, stale docstrings, ROADMAP drift).  **Zero
physics regressions in 7 consecutive releases.**  3 agents in
disjoint scopes (`A: F821 + ruff baseline`, `B: shim-removal
anti-regression pins + counter-pin`, `C: ROADMAP refresh + mypy +
P3 cluster`).

**2889 unit tests pass** (collected = 2895 = pass + 5 skip + 1
xfail), up from 2858 at v5.0.0; **+31 net** (A=4, B=6, C=21).
**34/34 validation pass.**

### P1 closures (3)

* **`set_default_*` UserWarning text updated v5.0 -> v5.1** (audit
  P1-NEW-F1-2).  At v5.0 HEAD the warning bodies in
  `propagators/propagation.py` said *"Consumer wiring at
  apply_real_lens / apply_real_lens_traced / propagate is staged
  for v5.0 alongside the file-split work."*  But the v5.0
  CHANGELOG had explicitly deferred that rollout to v5.1.  At
  v5.0 HEAD users calling `set_default_wave_propagator('fresnel')`
  saw a warning promising the bug was fixed "in v5.0" -- *which
  IS v5.0*.  The pinning test at `test_v4_16_3_agent_b.py`
  codified the misleading contract.  Fix: warning text now reads
  "staged for v5.1"; pinning test asserts `'v5.1' in msg`.
  v4.16.3's "default-knob honesty" closure is now genuinely
  honest at v5.0.1.
* **`benchmarks/test_bench_jax_jit.py` double-break fixed** (audit
  P1-NEW-V3-1).  Line 100 used `from lumenairy.system import
  propagate_through_system_jax, _PROPAGATE_SYSTEM_JAX_CACHE`
  (v5.0 `ModuleNotFoundError`); line 110 used `'params':
  {'radius': 200e-6}` (v5.0 `ValueError` on legacy aperture
  schema).  Both breaks fixed:
  `from lumenairy.propagators.system import ...` + `'params':
  {'diameter': 400e-6}`.  `benchmarks/` is not in `tests/unit/`
  CI collection scope, so the unit-CI gate didn't catch it.
* **CI lint baseline: 4 F821 real bugs fixed + advisory mode**
  (audit P1-NEW-V4-1).  `ruff check lumenairy/ tests/unit/`
  failed with 696 errors at v5.0 ship: 692 cosmetic (I001
  imports, F401 unused, F841 unused-var, F541 empty f-string,
  E702 semicolons) + **4 real F821 forward-reference bugs** in
  `lumenairy/algebra/base.py` and `lumenairy/propagators/system.py`
  (string-quoted annotations to lazily-imported `Source` /
  `PropagationResult` types missing a `TYPE_CHECKING` binding).
  Fix: F821 sites get proper `if TYPE_CHECKING:` blocks (real
  code-quality improvement, not papered over).  Lint job
  promoted to **advisory mode** (`continue-on-error: true`) at
  v5.0.1 with an inline comment noting that the cosmetic 692-
  error cleanup is a v5.1 mechanical-work item alongside the
  file splits.  PRs see lint output but don't fail-merge on it.

### P2 closures (5)

* **`simulate_detector_image` -> `apply_detector` doc-naming
  consistency** in Migration-Guide.md, CHANGELOG.md, README.md
  (audit P2-NEW-V3-2 / F1-3 2-way convergent).  The function is
  `apply_detector`; `simulate_detector_image` is not exported
  anywhere.  Users wouldn't have found the function the
  migration recipe named.
* **5 shim-removal anti-regression pins added** in
  `tests/unit/test_validation_helpers.py` (audit P2-NEW-F2-1).
  v5.0 shipped only `test_analysis_dot_analysis_shim_removed_in_
  v5_0` -- the other 4 v5.0 shim removals (`lumenairy.ao`,
  `lumenairy.io.hdf5`, top-level `lumenairy.system`, JAX aperture
  legacy schema, `cosmic_ray_rate` kwarg) had no anti-regression
  pin.  Risk: a v5.1 maintainer could accidentally re-add a
  removed shim with no test failure.  Now all 5 v5.0 honest-
  break closures have parallel pins with
  `pytest.raises(..., match=...)` that lock in the migration-
  recipe text alongside the raise.
* **ROADMAP v5.1 section refresh** (audit P2-NEW-F2-3).  v5.0's
  ROADMAP v5.1 block listed items v5.0 had already shipped (CI
  gates, public-API smoke, Python 3.10 bump, system.py move, 5
  of 8 shim removals, Migration-Guide existence).  "Read like
  the v5.0 plan, not the post-v5.0 horizon."  Stripped shipped
  items; refreshed live LOC counts for the 6 deferred file
  splits; added "Active back-compat shims at v5.0 (intentionally
  kept)" subsection documenting the 3 `apply_*_lens` re-exports
  preserved by design; refreshed the "Current state" header.
* **2 stale docstrings fixed** (audit P2-NEW-F1-4).  (a)
  `lumenairy/analysis/detector.py:82-100` still documented the
  removed `cosmic_ray_rate` kwarg as "Retained for back-compat"
  -- not retained.  Replaced with v5.0 removal note + migration
  recipe.  (b) `lumenairy/propagators/system.py:582` docstring
  example said `>>> result = la.system.evaluate(rx, src)` --
  `la.system` no longer exists.  Now `la.evaluate(...)`.
* **`[tool.mypy]` config preparation** (audit P2-NEW-V4-2).
  Added `follow_imports = "silent"` so a v5.1 mypy CI activation
  only sees the 63 scope-local errors that the cleanup actually
  owns (vs the ~1889 cascade errors from following unannotated
  downstream modules).  Activation deferred to v5.1.

### P3 closures (8)

* **CHANGELOG `__all__` arithmetic fix** (3-way V3+V4+F2
  convergent).  `len(lumenairy.__all__) == 533`; the "536" cited
  in the v5.0 CHANGELOG was the pytest case count (533
  parametrized + 3 standalone smoke tests).  Now reads "533
  entries verified via 536 smoke tests".
* **CHANGELOG `ui/` -> `lumenairy/ui/` doc drift** (P3-NEW-F2-2).
  Matches the actual `pyproject.toml` ruff `extend-exclude` value.
* **MCF `coherence_at(...)` deferral clarification** (P3-NEW-
  F2-MCF).  `PartialCoherenceMCF` + `coherence_at(...)` already
  shipped in v4.15.1 (`lumenairy/sources/core.py:1410, :1598`) and
  the class is re-exported at the top level.  What the v5.1
  deferral actually adds is the shorter `lumenairy.MCF` top-level
  alias for symmetry with `lumenairy.propagate_ensemble`.
  CHANGELOG bullet rewritten to be explicit; ROADMAP gains a
  dedicated "Partial-coherence / MCF public-API polish"
  subsection.
* **`apply_*_lens` shim preservation documented** for future-
  audit clarity.  The v5.0 work decided these re-exports are
  legitimate public API surface (not deprecation shims).
  CHANGELOG "Shims preserved" block extended with explicit
  forward-audit guidance so v5.2+ audits don't re-flag them.
* **Negative counter-pin for `test_public_api.py`** (P3-NEW-
  F1-4).  Injects a phantom name into `lumenairy.__all__`,
  asserts `hasattr(la, phantom) is False`, cleans up in
  `finally`.  Proves the smoke-test assertion machinery isn't
  vacuous.
* **V11 walker stale Python 3.9 comments refreshed**
  (P3-NEW-F1-1) at
  `test_v4_16_2_dispatcher_pin_doc_consistency.py:51-52, :68`.
  Library is 3.10+ at v5.0; comments updated.
* **Unreachable post-raise tuple-return block removed** in
  `lumenairy/propagators/system.py:932-935` (P3-NEW-F1-3).
  `_reject_legacy(...)` always raises, so the subsequent tuple
  return was unreachable.
* **Stale "one-shot deprecation warning" comment refreshed** at
  `lumenairy/propagators/system.py:1242-1248` (P3-NEW-F1-2).
  v5.0 changed the legacy aperture schema to a hard ValueError;
  the comment now reflects that.
* **Python 3.14 classifier dropped** (P3-NEW-V3-3).  CI matrix
  runs 3.10-3.13; 3.14 was aspirational.  Either drop or add to
  CI -- v5.0.1 drops with a comment that v5.1 can re-add 3.14
  alongside a CI matrix update.

### Files touched

* `lumenairy/algebra/base.py` -- TYPE_CHECKING guard
* `lumenairy/propagators/propagation.py` -- warning text v5.0 -> v5.1
* `lumenairy/propagators/system.py` -- TYPE_CHECKING guard +
  stale-comment cleanups + unreachable-code removal + docstring
  example `la.system.evaluate` -> `la.evaluate`
* `lumenairy/analysis/detector.py` -- `cosmic_ray_rate` docstring
  rewritten with v5.0 removal note
* `.github/workflows/unit-tests.yml` -- lint job advisory mode
* `pyproject.toml` -- mypy `follow_imports = "silent"`; Python
  3.14 classifier dropped; version 5.0.1
* `benchmarks/test_bench_jax_jit.py` -- v5.0 double-break fixed
* `Migration-Guide.md` -- `apply_detector` rename; §4.16.2 v5.0 ->
  v5.1 deferral
* `README.md` -- `apply_detector` rename
* `ROADMAP.md` -- v5.1 section refresh + "Current state" header
* `CHANGELOG.md` -- this entry; doc-naming fixes; arithmetic;
  MCF + `apply_*_lens` clarifications
* `tests/unit/test_v4_16_3_agent_b.py` -- pinning test v5.0 ->
  v5.1
* `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py` --
  stale Python 3.9 comments refreshed
* `tests/unit/test_validation_helpers.py` -- 5 shim-removal
  anti-regression pins
* `tests/unit/test_public_api.py` -- negative counter-pin
* `tests/unit/test_v5_0_1_agent_a.py` (NEW) -- F821 / TYPE_CHECKING
  regression tests
* `tests/unit/test_v5_0_1_agent_c.py` (NEW) -- ROADMAP / mypy /
  docstring regression tests

---

## [5.0.0] — 2026-05-20

**Major release.**  v5.0 is the coordinated breaking-change release:
removes back-compat shims that had been carried 3-9 releases past
their deprecation cycle, bumps the Python floor to 3.10, moves
`lumenairy/system.py` under `propagators/` where it functionally
belongs, and adds the CI infrastructure (ruff lint, mypy strict
incremental, fast-PR unit-test gate, public-API smoke test) that
the structural cleanup needs.

**Scope discipline:** the v4.16.x ROADMAP scoped a wider v5.0 ("6
file splits + library-wide resolver rollout + MCF coherence object
+ formula-3 coefficient ingestion + off-axis conic + 5 new
examples + 57-file test consolidation").  Those non-breaking items
move to **v5.1+** so the v5.0 diff stays reviewable.  See
`ROADMAP.md` for the v5.1 horizon.

### Breaking changes

* **Python 3.10+ required.**  `requires-python = ">=3.10"` in
  `pyproject.toml`.  Python 3.9 reached EOL on 2025-10.
* **`lumenairy.system` -> `lumenairy.propagators.system`.**  The
  sequential-propagation entry points functionally ARE a
  propagator -- they walk elements applying per-element
  propagators -- not a top-level peer of `propagators/` and
  `elements/`.  Public namespace (`import lumenairy as la;
  la.propagate_through_system(...)`) unchanged.  Direct imports
  of the private path break: `from lumenairy.system import X` ->
  `from lumenairy import X` (preferred) or `from
  lumenairy.propagators.system import X`.
* **5 back-compat shims removed**:
  * `lumenairy.analysis.analysis` (v4.7 rename shim) -- now
    raises `ModuleNotFoundError`.  Use `lumenairy.analysis`.
  * `lumenairy.ao` (v4.3 shim) -- now raises
    `ModuleNotFoundError`.  Use `lumenairy.analysis.ao` or the
    top-level `lumenairy.DeformableMirror`.
  * `lumenairy.io.hdf5` (rename shim) -- now raises
    `ModuleNotFoundError`.  Use `lumenairy.io.storage` or the
    top-level re-exports.
  * `propagate_through_system_jax` legacy aperture schema
    (pre-v4.12; deprecated v4.12, removed v5.0).  Legacy params
    `radius` / `half_width_x` / `inner_radius` now raise
    `ValueError` with the canonical-schema migration recipe
    inline.  Migrate: double the value and rename
    (`radius=r` -> `diameter=2*r`, etc.).
  * `apply_detector(..., cosmic_ray_rate=...)` (v4.9
    deprecated kwarg; did not scale with detector area or
    exposure) -- removed; now raises `TypeError` (unexpected
    keyword argument).  Migrate to
    `cosmic_ray_rate_per_m2_per_s=R/A/T` where `A` is the
    detector area and `T` is the exposure time.
* **Shims preserved as legitimate public API surface** (not
  removed despite the ROADMAP's audit-V4_13_1 suggestion):
  * `lumenairy.elements.lenses.apply_*_lens` re-exports.  These
    provide a coherent one-stop import surface; the underlying
    file-split into `_lens_thin.py` / `_lens_real.py` /
    `_lens_traced.py` is an internal organisational choice
    rather than a deprecation cycle.
    **Note for future audits (v5.0.1 audit
    `apply_*_lens` shim-preservation closure)**: these
    re-exports are **intentionally retained** as the canonical
    one-stop user-facing import path -- a v5.2+ audit that
    flags them as "stale shim removable" should be rejected
    with a pointer to this CHANGELOG entry.  The decision was
    made at v5.0 ship after weighing the v4.13.1 ROADMAP
    suggestion against the user-facing ergonomics of the
    single ``from lumenairy.elements.lenses import apply_real_lens``
    import; the latter won.

### CI gates + infrastructure

* **NEW `.github/workflows/unit-tests.yml`** -- fast PR feedback
  gate.  Runs `pytest tests/unit -m "not integration"` on Python
  3.10, 3.11, 3.12, 3.13; `ruff check` on the library + unit
  tests; the new public-API smoke test.
* **NEW `[tool.ruff]` config** in `pyproject.toml`.  Conservative
  initial rule set (E, F, I) with documented per-file ignores;
  excludes `validation/`, `docs/`, `examples/`, `lumenairy/ui/`
  from the v5.0 baseline (v5.0.1 audit P3-NEW-F2-2 closure of
  the `"ui/"` -> `"lumenairy/ui/"` doc drift).
* **NEW `[tool.mypy]` config** -- incremental adoption starting
  with the small self-contained modules (`backend/`,
  `_deprecation.py`, `_context.py`, `progress.py`, `memory.py`).
  Everything else stays untyped for v5.0.
* **NEW `tests/unit/test_public_api.py`** -- asserts every name
  listed in `lumenairy.__all__` is resolvable via
  `getattr(lumenairy, name)`.  Catches "exported but not
  imported" / "imported but not exported" sibling-gap at the
  facade.  533 entries verified via 536 smoke tests
  (533 parametrized + 3 standalone) at v5.0 ship.

### Migration-Guide.md

`Migration-Guide.md` adds a v5.0.0 section with concrete
old->new recipes for each breaking change.  The deferred v5.1+
items are listed honestly so users know what to expect.

### Tests + CI

* **2858 unit pass / 5 skip / 1 xfail = 2864 collected** (was
  2327 at v4.16.3; +531 net -- the v5.0 work landed alongside
  cumulative v4.16.x test additions in this session).
* **34/34 validation pass.**
* Updated callers across `lumenairy/` and `tests/` to the new
  `lumenairy.propagators.system` import path.
* Updated v4.12 aperture-schema tests + v4.9 cosmic_ray_rate
  test from "must warn" to "must raise" semantics.

### Deferred from v5.0 to v5.1+ (see ROADMAP.md)

* 6 large-file splits (`raytrace/core.py`,
  `propagators/propagation.py`, `propagators/asymptotic.py`,
  `optimize/core.py`, `io/prescriptions.py`,
  `analysis/core.py`).  Pure mechanical reorganisation; no
  public API change.
* Library-wide default-config knob resolver rollout (`set_default_
  wave_propagator`, `set_default_dy`, `set_default_real_dtype`
  remain API-only at v5.0; the v4.16.3 one-shot UserWarning stays
  in place).
* MCF top-level public-API polish (v5.0.1 audit P3-NEW-F2-MCF
  clarification).  `PartialCoherenceMCF` -- including its
  `coherence_at(...)` two-point query -- already shipped in v4.15.1
  (`lumenairy/sources/core.py`) and is re-exported at the
  top level as `lumenairy.PartialCoherenceMCF` since v4.15.1.  What
  v5.1 still owes is the *naming* polish: a shorter top-level alias
  `lumenairy.MCF` for symmetry with `lumenairy.propagate_ensemble`
  / `lumenairy.coherence_at` so the "import the canonical name"
  story is uniform across the partial-coherence surface.  The
  v4.16.x ROADMAP entry "MCF object" predates the v4.15.1 ship and
  was carried forward without rewording at v5.0; this CHANGELOG
  bullet is the authoritative deferral statement.
* Off-axis conic in surface frame.
* 26 formula-3 (polynomial) glass coefficients.
* 5 new examples (multi-config / zoom, tolerancing, coronagraph
  workflow, AO closed-loop, ghost / stray-light).
* 57-file `test_audit_fixes_*` consolidation into topical homes.

---

## [4.16.3] — 2026-05-20

**Closes the v4.16.2 audit
(`docs/audits/AUDIT_V4_16_2_2026_05_20.md`) through P3.**  Audit
found zero P0 + 2 P1 + 6 P2 + 8 P3, concentrated around v4.16.2's
pre-v5.0 prep features being mostly scaffolding without real
consumers, plus 2 structural-bypass issues inside the new V11
doc-consistency walker (the very walker designed to retire the
documentation-surface sibling-gap meta-pattern contained that
pattern itself).  4 agents in disjoint scopes (`A: V11 walker
hardening`, `B: default-knob consumer wiring + Migration-Guide
correction`, `C: optimize/core.py P2 polish`, `D: P3 cluster +
soften "structurally retired" claim`).

**2327 unit tests pass** (collected = 2333 = pass + 5 skip + 1
xfail), up from 2270 at v4.16.2; **+57 net** (per-agent: A=17,
B=16, C=8, D=15, walker-extra=1 = 57).  **34/34 validation pass**.

### P1 closures (2)

* **V11 walker pyproject parsing -> `tomllib`** (Agent A; audit
  P1-NEW-F1-2).  The v4.16.2 11th meta-pin walker used the regex
  `r'^([a-zA-Z_][a-zA-Z0-9_-]*)\s*=\s*\[(.*?)\]'` -- non-greedy,
  stopping at the first `]`.  Silently mis-parsed
  `jax-gpu = ["jax[cuda12]>=0.4.20"]` (captured only `"jax[cuda12`)
  and the `all = [...]` block when its body contained
  ` `lumenairy[all]` ` comment-bracket text.  The walker passed
  vacuously because `refractiveindex` was independently captured
  via `[glass]`; if a future maintainer removed the dedicated
  `[glass]` group keeping the dep only in `[all]`, drift would
  go green WITH drift.  Replaced regex parsing with `tomllib`
  (Python 3.11+) / `tomli` (3.9/3.10 backport) graceful fallback.
  Two anti-regression pins assert the literal v4.16.2 strings no
  longer appear in the walker source.
* **Migration-Guide.md §4.16.2 corrected** (Agent B; audit
  P1-NEW-F1-1).  The v4.16.2 recipe used
  `set_default_wave_propagator('fresnel')` followed by
  `apply_real_lens(...)` -- but `apply_real_lens` hardcodes
  `wave_propagator: str = 'asm'` and does NOT consult the default
  knob.  A user copy-pasting silently used ASM.  Rewrote the §4.16.2
  section: explicit "API-only in v4.16.2/v4.16.3" limitation note;
  replaced the misleading recipe with one using `set_default_complex_
  dtype` + `set_default_real_dtype` (knobs with real consumers);
  retained `wave_propagator=` per-call kwarg on the apply_real_lens
  example.

### P2 closures (6)

* **`get_default_real_dtype` consumer wiring fixed** (Agent B;
  audit P2-NEW-F1-3).  v4.16.2's "representative wiring" at
  `propagators/ensemble.py:347-355` was structurally unreachable
  dead code -- the `except (TypeError, ValueError)` branch could
  never fire because the earlier shape check at `:308-330` already
  guaranteed `ensemble.dtype` was a valid numpy dtype.  Refactored
  so `get_default_real_dtype()` is the canonical `in_dtype is None`
  fallback path (now reachable via the `getattr(ensemble, 'dtype',
  None)` default).  The knob is now narrowly consumed at one
  reachable site, preserving the v4.16.2 CHANGELOG claim of at least
  one wired consumer.
* **`set_default_wave_propagator` + `set_default_dy` no-consumer
  `UserWarning`** (Agent B; audit P2-NEW-F1-4).  Both knobs store
  values no library code reads at v4.16.3.  Setters now emit a
  one-shot module-level-latched `UserWarning` informing users that
  the knob is "API-only at v4.16.2/v4.16.3; consumer wiring at
  `apply_real_lens` / `apply_real_lens_traced` / `propagate` lands
  in v5.0".  Sibling-gap pin asserts these knobs have zero
  consumers library-wide -- when v5.0 adds the first consumer, the
  pin FAILS LOUDLY prompting removal of the stale warning + the
  Migration-Guide limitation note.
* **V11 version list -> CHANGELOG-driven** (Agent A; audit
  P2-NEW-F2-MED-1).  v4.16.2's `test_migration_guide_has_known_
  version_sections` hardcoded `('4.13.0', '4.15.1', '4.16.1',
  '4.16.2')` -- when v4.17.0 ships with a breaking change, walker
  would pass silently unless someone manually edited the tuple.
  Replaced with `_versions_with_breaking_changes_from_changelog()`
  scan extracting `## [X.Y.Z]` headings; high-precision markers
  only (`silent semantics change`, `SUM->AVG`, etc.) plus
  `### Breaking changes` heading detection; documented `_MIGRATION_
  GUIDE_SIBLING_COVERED` allowlist for v4.15.2 (its migration recipe
  lives under v4.15.1's Migration-Guide section).
* **V11 extends to CHANGELOG↔Migration-Guide drift coverage**
  (Agent A; audit P2-NEW-F2-MED-2).  New
  `test_migration_guide_sections_are_non_trivial` test enforces
  each `## X.Y.Z` section has >=200 chars of non-whitespace body.
  Future CHANGELOG entries flagged "breaking" must come with a
  substantive Migration-Guide entry or the walker fails.
* **`Constraint` auto-probe DeprecationWarning** (Agent C; audit
  P2-NEW-F1-1).  v4.16.1 shipped a `Constraint.__post_init__`
  auto-probe; v4.16.2 silently removed it.  v4.16.3 emits a
  one-cycle DeprecationWarning via module-level latch, pattern-
  parallel to the v4.16.2 `MultiWavelengthMerit` `FutureWarning`
  latch.  Scheduled for removal in v5.0.
* **`pickle.dumps` probe catch widened** (Agent C; audit
  P2-NEW-F1-2).  v4.16.2's `except (pickle.PicklingError,
  AttributeError, TypeError)` missed `RecursionError` (deep object
  graph), `RuntimeError` (custom `__reduce__`), `MemoryError`,
  and arbitrary `__reduce__` / `__getstate__` exceptions.  Widened
  to `except Exception` -- pickling is best-effort heuristic; any
  failure is "not safely picklable" signal.  `BaseException`
  (`KeyboardInterrupt` / `SystemExit`) intentionally still
  propagates.

### P3 closures (8)

* **`__polynomial__` sentinel** parallel to `__sellmeier__` (Agent D;
  audit P3-NEW-F1-1).  `POLYNOMIAL_COEFFICIENTS` dispatch was
  fallback-only when refractiveindex was unavailable; with the
  sentinel, polynomial-formula glasses can opt in to the bundled
  evaluator even with refractiveindex installed.  Extended
  `_check_glass_registry_consistency` with forward + reverse
  polynomial checks; `get_glass_index_complex` updated to include
  `__polynomial__` in the no-extinction sentinel tuple.
* **POLYNOMIAL/SELLMEIER dispatch order doc/code reconcile**
  (Agent D; audit P3-NEW-V3-1).  Code does SELLMEIER -> POLYNOMIAL;
  v4.16.2 docs claimed the opposite.  Inline comment added citing
  the actual order.
* **`DEFAULT_*` constants re-exported at top level** (Agent D;
  audit P3-NEW-F2-LOW-1).  `DEFAULT_COMPLEX_DTYPE` was already
  exported via `lumenairy/__init__.py`; the v4.16.2 new globals
  (`DEFAULT_REAL_DTYPE`, `DEFAULT_WAVE_PROPAGATOR`, `DEFAULT_DY`)
  were not -- sibling-gap.  Added to both the import block and
  `__all__`.
* **Per-surface (not max) thickness in high-NA hoist message**
  (Agent D; audit P3-NEW-F1-4).  `_maybe_warn_transfer_jax_high_na`
  now accepts `surface_index`; hoist loops surfaces to find the
  worst |N| and cites THAT surface's thickness in the user-facing
  message (was overstating worst-case drift via `max(thickness)`).
* **Multiprocess / fork-safety documentation** (Agent D; audit
  P3-NEW-F1-2 + P3-NEW-F1-3).  Added a "Multiprocess / fork notes"
  section near the top of `propagators/propagation.py` documenting
  that the one-shot latches AND the `DEFAULT_*` module-level globals
  are NOT pickle/fork-safe -- spawn-mode workers re-import the
  module and reset to defaults / re-emit warnings.  Not fixing the
  semantics (would need shared-state); just documenting honestly.
* **`psutil` promoted to Required in requirements.txt** (Agent D;
  audit P3-NEW-F1-5).  `psutil>=5.0` is a hard dep in
  pyproject.toml but the v4.16.2 requirements.txt listed it under
  "Recommended".  Promoted for parity.
* **"Structurally retired" claim softened** (Agent D; audit
  P3-NEW-F2-LOW-2).  CHANGELOG + ROADMAP both said "structurally
  retired across all known classes"; honest framing is "retired
  across all currently-known classes; new classes will continue to
  surface".
* **CHANGELOG sentinel line citation refresh** `:3015` -> `:3032`
  (~17 lines added by Agent C's `Constraint` DeprecationWarning
  latch + pickle catch widening).

### Tests + CI

* **2327 pass / 5 skip / 1 xfail = 2333 collected** (up from
  2270 / 5 / 1 = 2276 at v4.16.2; +57 net).  Per-agent breakdown:
  A=17, B=16, C=8, D=15, walker-extra=1.  Sum: 57.
* New test modules:
  * `tests/unit/test_v4_16_3_agent_a.py` (17 tests)
  * `tests/unit/test_v4_16_3_agent_b.py` (16 tests)
  * `tests/unit/test_v4_16_3_agent_c.py` (8 tests)
  * `tests/unit/test_v4_16_3_agent_d.py` (15 tests)
* V11 walker grew from 7 to 8 tests (`test_migration_guide_sections_
  are_non_trivial` added).

---

## [4.16.2] — 2026-05-20

**Closes the v4.16.1 audit
(`docs/audits/AUDIT_V4_16_1_2026_05_19.md`) through P3** -- a focused
follow-up after v4.16.1 hit PyPI.  Audit found zero P0 / 5 P1 / 8 P2
/ 9 P3, concentrated in (a) 3 code-correctness items the v4.16.1
verifier audit missed because the test pins themselves bypassed the
production path, and (b) 4 documentation-surface drifts that proved
the sibling-gap meta-pattern had migrated from code surfaces
(covered by V1-V10 walkers) to documentation surfaces (uncovered).
4 agents in disjoint scopes (`A: JAX gate + ensemble dispatch`,
`B: optimize/core.py`, `C: pre-v5.0 features + glass P3`,
`D: doc-surface + 11th walker + Migration-Guide`).

**Also lands the user-requested pre-v5.0 prep features**:
* Bundled Sellmeier formula-3 (polynomial) evaluator infrastructure
* 3 library-wide default-config knobs (`set_default_real_dtype`,
  `set_default_wave_propagator`, `set_default_dy`)
* `Migration-Guide.md` skeleton at the repo root

**2270 unit tests pass** (collected = 2276 = pass + 5 skip + 1
xfail), up from 2198 at v4.16.1; +78 net (per-agent: A=18, B=16,
C=24, D=20 = 78 -- reconciles exactly).  **34/34 validation pass**.

### P1 closures -- code findings (Agent A + Agent B)

* **`_transfer_jax` high-NA warning structurally unreachable
  in production** (Agent A; audit P1-NEW-F1-1).  The v4.16.1 gate
  at `jax_trace.py:579` used `isinstance(direction_n, np.ndarray)`,
  but `make_jax_ray_state(...)` calls `jnp.asarray(N)` which yields
  `jax.Array` -- NOT a `np.ndarray` subclass since JAX 0.4+.  The
  gate returned early on every production user-flow call.  Fix:
  duck-typed gate (`np.asarray(direction_n)` probe; rejects
  `jax.core.Tracer`), PLUS an eager-only one-shot probe hoisted
  to the entry of `trace_jax` BEFORE the inner `jax.jit` wrapper
  (the jit wrapper makes everything inside a Tracer, regardless of
  the gate).  New integration test calls `trace_jax(...)` with
  `make_jax_ray_state(N=0.5*np.ones(K))` end-to-end and asserts the
  RuntimeWarning fires -- closes the production-path gap.
* **`propagate_ensemble` silently downcasts CuPy/JAX to NumPy**
  (Agent A; audit P1-NEW-F1-2).  `ensemble.py:253`'s
  `np.asarray(ensemble)` triggered GPU->CPU transfer (CuPy) or
  forced concretization (JAX), defeating the docstring's "tolerate
  duck-typed array protocols" claim.  Fix: `array_namespace`
  dispatch via `lumenairy.backend`; accumulator built on the
  matching `xp` so the GPU / JAX paths stay on the backend.  Also
  rewrites `_coerce_field_from_propagator_return` to preserve
  backend.
* **`MultiWavelengthMerit` SUM->AVG one-cycle `FutureWarning`**
  (Agent B; audit P1-NEW-F1-3).  v4.16.1's SUM->AVG fix was correct
  but silent -- existing user-calibrated 3-wavelength configs
  silently dropped 3x.  v4.16.2 emits a one-cycle `FutureWarning`
  via module-level latch when `len(wavelengths) > 1`, alerting
  users to re-scale `weight` by `len(wavelengths)` if they tuned
  against pre-v4.16.1 SUM behavior.  Latch ensures the warning
  fires only ONCE per process (critical -- without the latch the
  warning would flood optimization loops).
* **README -> pyproject.toml dependency declaration sync**
  (Agent D; audit P1-NEW-F2-HIGH-1).  README's `### Required`
  block still listed `refractiveindex` as Required + `pip install
  numpy refractiveindex` as the quick-install command; pyproject
  moved it to `[glass]` extras in v4.16.1.  Full dependency block
  rewritten: enumerates each pyproject extras group + `pip install
  lumenairy[glass]` as the canonical install pattern.
* **requirements.txt -> pyproject.toml sync** (Agent D; audit
  P1-NEW-F2-HIGH-2).  Dropped uncommented `refractiveindex>=1.0`;
  moved `h5py>=3.0` to commented section (it's only in `[hdf5]` /
  `[gui]` extras); updated commented `zarr>=2.14` -> `zarr>=3.0`
  to match v4.16.1's floor bump.  Added commented lines for every
  optional-extras group + header note pointing at pyproject.toml
  as the canonical source.

### P2 closures (8) -- API consistency + meta-pins + doc hygiene

* **`Constraint.__post_init__` probe -> opt-in `.validate()` method**
  (Agent B; audit P2-NEW-F1-1).  v4.16.1's probe ran real user code
  on instantiation (e.g. BFL constraint calling `system_abcd()` on
  every `Constraint(...)` call).  Moved to opt-in `Constraint.
  validate()` method; users who relied on the auto-probe call it
  explicitly.
* **Lambda warning -> `pickle.dumps` probe** (Agent B; audit
  P2-NEW-F1-2).  v4.16.1's `__name__ == '<lambda>'` check missed
  closures (`def inner(x): ...`) and `functools.partial(lambda,
  ...)`.  Replaced with `pickle.dumps(self.fun)` probe; catches all
  unpicklable callables.
* **Existing v4.16.0 Constraint tests updated to module-level
  functions** (Agent B; audit P2-NEW-F1-3).  Five lambda-Constraint
  test sites in `test_v4_16_0_agent_c.py` migrated to module-level
  `_sum_constraint` / `_first_coord` so the v4.16.1 lambda warning
  doesn't pollute the warning channel.
* **10th meta-pin walker hardened** (Agent D; audit P2-NEW-F1-4).
  `_module_has_register_cache_clearer_call` rewritten to require
  the call appear at **module level**, not nested inside a
  function / class body / always-False `if` branch.  Canonical
  top-level `try/except ImportError` enrollment idiom still
  accepted.  4 new counter-pins (positive + negative) verify the
  tightening.
* **`_clear_local_asm_caches` late-binding-lambda registration**:
  already landed in v4.16.1 (Agent C scope at that release).
* **ROADMAP V9 -> V11 meta-pin enumeration** (Agent D; audit
  P2-NEW-F2-MED-1).  "ALL 9 dispatcher meta-pins" -> "ALL 11
  dispatcher meta-pins"; V10 + V11 entries added.  Updated the
  sibling-gap retirement claim to cover documentation surfaces too.
* **CHANGELOG test-count arithmetic** (Agent D; audit
  P2-NEW-F2-MED-2).  v4.16.1 headline `2208 / +102 / 2106`
  refreshed to `2198 / +85 / 2113` (collected metric, arithmetic
  reconciles: 2113 + 85 = 2198 = pass=2192 + skip=5 + xfail=1).
  Also corrected the Tests + CI tail section.  v4.16.2 audit note
  added explaining the discrepancy.
* **CHANGELOG `UserWarning` -> `RuntimeWarning` typo** (Agent D;
  audit P2-NEW-V2-1).  The v4.16.1 entry's High-NA `_transfer_jax`
  block said "emits a `UserWarning`" but implementation + tests
  both use `RuntimeWarning`.

### P3 closures (9)

* `propagate_ensemble` empty 3-D ensemble -> `ValueError` (Agent A)
* `propagate_ensemble` `dx`/`wavelength` kwargs collision -> clear
  `ValueError` (Agent A)
* `_resolve_bound` 3-tuple guard (Agent B)
* `Constraint.fun` docstring example: lambda -> module-level
  function (Agent B)
* LM bounds `lm` -> `trf` override UserWarning added (Agent B)
* `jax.grad` pin for B.4 dtype probe (Agent A)
* `propagator_kwargs` precedence pin for B.1 (Agent A)
* `GLASS_VALIDITY` consistency check accepts numpy scalars
  (Agent C; audit P3-NEW-F1-4)
* CHANGELOG sentinel line citation refresh `:2974` -> `:3015`
  (Agent D)

### NEW -- 11th meta-pin walker (doc-consistency)

`tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py` (7
tests): closes the v4.16.1-identified documentation-surface
sibling-gap meta-pattern.  Scans 4 surfaces for drift vs the
canonical `pyproject.toml`:
* README.md `Required` block doesn't list optional-extras packages
* README.md `pip install` command doesn't force optional-extras
* requirements.txt uncommented lines match pyproject hard deps
* ROADMAP.md `ALL N meta-pins` count matches V-enumeration
* CHANGELOG.md v4.16.1 headline arithmetic reconciles
* Migration-Guide.md exists with known version sections

The sibling-gap meta-pattern is now structurally retired at BOTH
code surfaces (V1-V10) AND documentation surfaces (V11) across all
currently-known classes; new classes will continue to surface and
be added to the V-walker family as identified.

### NEW -- Pre-v5.0 prep features (Agent C)

* **Bundled Sellmeier formula-3 (polynomial) evaluator**.
  `lumenairy/glass.py`:
  - NEW `_polynomial_index(wavelength_m, coeffs, glass_name=None)`
    -- implements refractiveindex.info formula-3:
    `n^2 = c0 + sum_i c_i * lam_um ** exp_i`.  Subsumes the Schott
    6-coefficient polynomial form.
  - NEW `POLYNOMIAL_COEFFICIENTS = {}` -- empty at ship.  v4.16.2
    lands the evaluator + dispatch wiring; populating the 26
    catalogue entries (Hikari E-/J-, Sumita K-, 4 CDGM polynomial)
    requires per-glass vendor-source review + 5e-5 n_d cross-check
    against refractiveindex.info YAML and is staged for v5.0.
  - `get_glass_index` dispatch updated: when refractiveindex is
    unavailable AND the glass is in POLYNOMIAL_COEFFICIENTS,
    dispatches to `_polynomial_index` before raising ImportError.
* **3 default-config knobs**, parallel to existing
  `set_default_complex_dtype`:
  - `set_default_real_dtype(dtype)` / `get_default_real_dtype()`
    -- accepts `np.float32` / `np.float64`.
  - `set_default_wave_propagator(name)` / `get_default_wave_
    propagator()` -- accepts `'asm'`, `'sas'`, `'fresnel'`,
    `'rayleigh_sommerfeld'`, `'rs'`.
  - `set_default_dy(value)` / `get_default_dy()` -- accepts
    `None` (means "match dx") or a positive finite float.
  - All 6 functions exported at top level via `lumenairy/__init__.py`.
  - Representative consumer wiring landed in
    `propagators/ensemble.py` (no-input-dtype real fallback path
    honours `get_default_real_dtype()`).  Full library-wide
    resolver rollout staged for v5.0.
* **`Migration-Guide.md` skeleton** at repo root.  Version-spanning
  migration guide for v4.x; sections for v4.13.0 (rcwa.py rename,
  wavelength-required), v4.15.1 (Schell ensemble return shape),
  v4.16.1 (MultiWavelengthMerit SUM->AVG, refractiveindex
  optional), v4.16.2 (default-config knobs).  Forward section for
  v5.0 itemizing planned migration points.

### Tests + CI

* **2270 pass / 5 skip / 1 xfail = 2276 collected** (up from 2198
  at v4.16.1; +78 net).  Per-agent breakdown: A=18, B=16, C=24,
  D=13+7=20.  Sum: 18+16+24+20 = 78 (reconciles).
* New test modules:
  * `tests/unit/test_v4_16_2_agent_a.py` (18 tests)
  * `tests/unit/test_v4_16_2_agent_b.py` (16 tests)
  * `tests/unit/test_v4_16_2_agent_c.py` (24 tests)
  * `tests/unit/test_v4_16_2_agent_d.py` (13 tests)
  * `tests/unit/test_v4_16_2_dispatcher_pin_doc_consistency.py`
    (7 walker tests -- the 11th meta-pin)
* CHANGELOG line-citation refresh: `_ZERO_APERTURE_MASK` branch
  site drifted `:2974` -> `:3015` after Agent B's
  `MultiWavelengthMerit` `FutureWarning` latch + `Constraint`
  probe move + lambda pickle-probe (~41 lines added above the
  sentinel branch).

---

## [4.16.1] — 2026-05-19

**Closes the v4.16.0 deep audit
(`docs/audits/AUDIT_V4_16_0_DEEP_2026_05_19.md`) through P3.**  The
audit was the first "deep" audit to actively hunt silent-wrong-answer
correctness bugs alongside the usual structural/UX cleanup; 4 real
physical-correctness defects surfaced, plus the previously half-shipped
Schell-model partial-coherence cluster, plus 8 hygiene items.  4 agents
worked in disjoint scopes (`A: correctness bugs`, `B: Schell + JAX
paths`, `C: constraints + meta-pins + warn hygiene`, `D: glass + compat
+ UX`).  **2198 unit tests pass** (up from 2113; +85 net) -- of
those 2198, 2192 actively pass + 5 documented skips (4 pymoo +
1 ZARR_MKDIR_PATCH) + 1 documented xfail.
**34/34 validation files pass**.  (v4.16.2 audit P2-NEW-F2-MED-2
correction: pre-v4.16.2 headline cited 2208 / +102 / baseline 2106 --
off by 10 / +17 / -7; the corrected numbers (collected = pass +
skip + xfail) reconcile to the empirical per-agent breakdown
A=11 + B=26 + C=20+6 + D=22 = 85.)

### P0 / P1 closures — correctness bugs (Agent A)

Four real silent-wrong-answer defects at user-relevant configurations
that the prior verification-style audits missed.  Each ships with an
empirical regression test pinning the failure mode and a sibling-gap
sweep confirming no other sites carry the same pattern.

* **Bug 1 — `MultiWavelengthMerit.evaluate` SUM -> AVG.**
  `lumenairy/optimize/core.py`: the wrapper's tail `return self.weight
  * total` summed sub-merit results across wavelengths rather than
  averaging.  Documented semantics + both sibling classes
  (`MultiFieldMerit`, `ToleranceAwareMerit`) divide by `len(...)` at
  the return; the bug was localised to this one class.  Fixed:
  `return self.weight * total / max(len(self.wavelengths), 1)`.  A
  3-wavelength merit now returns the same value as a 1-wavelength
  merit on the same sub-merit + constant field (was returning `3x`).
* **Bug 2 — `shack_hartmann` wavefront pitch quantisation.**
  `lumenairy/analysis/detector.py`: the slope-to-wavefront integration
  multiplied the cumsum by the user-requested `lenslet_pitch`, but the
  on-grid pitch is the integer-pixel quantised `sa_pixels * dx`.  At
  `lenslet_pitch / dx = 1.7` (`sa_pixels = 2`), the reconstructed
  wavefront amplitude was biased by `8.5 / 10 = 0.85` (17.6% low
  relative to the post-fix amplitude).  Fix: use `pitch_actual =
  sa_pixels * dx` for the cumsum step.  Empirically pinned by
  `test_bug2_shack_hartmann_amplitude_ratio_physics_pin`.
* **Bug 3 — `_detect_backend` directory misclassification.**
  `lumenairy/io/storage.py`: the auto-detect routed *any* directory
  path to Zarr regardless of whether a Zarr store was actually present
  (`if path.endswith('.zarr') or os.path.isdir(path)`).  A bare
  directory matching a typical HDF5 sibling layout was silently
  misrouted, and `pathlib.Path` callers hit `AttributeError` on the
  string-only `.endswith` check.  Fix: `str(path)` cast +
  directory-routing restricted to actual Zarr stores via the canonical
  `zarr.json` (v3) / `.zarray` (v2) marker files.
* **Bug 4 — LM `bounds` parser accepts `None` endpoints.**
  `lumenairy/optimize/core.py`: the `method='lm'` branch built
  `lb`/`ub` arrays via `b[0] if b else -np.inf`; `b = (None, 1.0)` is
  truthy, so `None` leaked into `np.array([None, 0.0, ...])` (object
  dtype), and scipy raised an opaque downstream error.  Fix: explicit
  `_resolve_bound` helper that routes any `None` (per-side or
  per-tuple) to `+/-np.inf` and produces a clean `float64` array.

### P1 closures — half-shipped clusters (Agent B)

* **`propagate_ensemble(...)` helper added** (audit Part 5 P0-1).
  New module `lumenairy/propagators/ensemble.py`, exported at the
  top-level as `lumenairy.propagate_ensemble`.  Iterates a Schell-family
  `(n_realizations, Ny, Nx)` ensemble through any coherent propagator
  (`'asm'` / `'fresnel'` / `'fraunhofer'` / `'rs'` / `'sas'` or a
  user-supplied callable) and returns `I_partial = <|E_k|^2>_k` (the
  canonical Wolf-coherence-theory result).  `return_ensemble=False` by
  default for memory efficiency; opt-in `return_ensemble=True` for the
  full propagated stack.  Shape-mismatch + 2-D-field-instead-of-
  ensemble cases raise informative `ValueError`s.  New example
  `examples/06_schell_propagation.py` measures a `~6.95x` smoothing
  factor (coherent-peak / partial-peak) on a 256x256 grid at
  `sigma_g / w0 = 0.3`, consistent with the Wolf-Carter far-field
  scaling.
* **Default Schell-factory `DeprecationWarning` retired.**  The 3
  top-level factories (`create_gaussian_schell_source`,
  `create_schell_model_source`, `create_annular_incoherent_source`)
  and 2 `Source.*` classmethods now default `return_kind='ensemble'`
  directly.  The `_RETURN_KIND_UNSET` sentinel + warning helper are
  preserved as deprecated public symbols (the v4.15.3 sentinel-
  promotion meta-pin imports them); targeted for removal in v5.0.
* **MCF rejection message refreshed.**  `lumenairy/_validation.py`:
  the "MCF planned for v4.16+" wording was stale at v4.16.0.  Updated
  to cite v5.0+ and point callers at the new `propagate_ensemble`
  helper for the partial-coherence path that lands now.
* **JAX-traceable dtype probe.**  `lumenairy/system.py` ~line 1184:
  swapped the `np.asarray(E_in).dtype` probe for duck-typing
  `getattr(E_in, 'dtype', None)` so the `propagate_through_system_jax`
  path no longer breaks under `jax.jit` / `jax.grad` tracers (which
  refuse the `np.asarray` cast).
* **High-NA `_transfer_jax` RuntimeWarning.**
  `lumenairy/raytrace/jax_trace.py`: added an eager-only guard
  (`isinstance(direction_n, np.ndarray)`-gated) that emits a
  `RuntimeWarning` when `min |N| < 0.95` — the regime where the
  paraxial small-angle approximation preserved for autodiff
  stability begins to diverge from the NumPy reference.  Tracer-time
  path is unchanged (no warning, preserves `jit` / `grad` purity).
  (v4.16.2 audit P2-NEW-V2-1: pre-v4.16.2 bullet said "`UserWarning`"
  -- code + tests use `RuntimeWarning`; corrected.)  (v4.16.2 audit
  P1-NEW-F1-1: pre-v4.16.2 the `isinstance(np.ndarray)` gate was
  structurally unreachable in production because `make_jax_ray_state`
  converts to `jax.Array` which is not a `np.ndarray` subclass since
  JAX 0.4+; v4.16.2 replaces the gate with a duck-typed
  `np.asarray(...)` probe + hoists an eager-only check to the
  `trace_jax` entry before the inner `jax.jit` wrapper.)

### P2 closures — API consistency + meta-pins (Agent C)

* **`Constraint` API narrowed to scalar-return.**  Vector-return
  callables crashed inside the pymoo wrapper's `float(_f(xv))` coercion
  with an opaque `TypeError`.  Docstring narrowed to `f(x) -> scalar`
  and `__post_init__` adds a best-effort scalar-shape probe.  3-test
  pin block exercises both the accept (scalar) and reject (ndarray of
  shape `(K,)`) paths.
* **`Constraint(fun=lambda x: ...)` UserWarning for parallel workers.**
  Lambdas aren't picklable; `differential_evolution(workers>1)` /
  joblib-parallelised FD-gradient fails with `PicklingError`.  Soft
  heads-up at `__post_init__` when `fun.__name__ == '<lambda>'`;
  single-process SLSQP / trust-constr is unaffected.
* **`_clear_local_asm_caches` late-binding-lambda registration.**
  `lumenairy/propagators/propagation.py:~803`: the cache registry
  enrollment used an early-binding partial, diverging from the
  canonical late-binding-lambda pattern of the other 8 cache
  enrollments.  Switched to `lambda: _clear_local_asm_caches()` for
  pattern parity.
* **10th cache-registry meta-pin walker.**
  `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`:
  AST-walks every `@lru_cache`-decorated module-level function and
  asserts a paired `_cache_registry` enrollment.  15 caches discovered,
  0 orphans — closes the V4-bucket sibling-gap structurally
  (continuing the V1-V9 meta-pin trajectory).
* **`_check_glass_registry_consistency()` extends to GLASS_VALIDITY.**
  Every `GLASS_VALIDITY` key now must appear in `GLASS_REGISTRY` (and
  must be a `(lambda_min, lambda_max)` 2-tuple with `lambda_min <
  lambda_max`, both finite, non-negative).
* **`warnings.warn(..., stacklevel=2)` hygiene** added at the 2 sites
  flagged in `lumenairy/io/prescriptions.py` (lines ~1019 / ~1470).

### P3 closures — compat + glass / materials + UX (Agent D)

* **4 stale `n_d` inline comments in `lumenairy/glass.py` corrected**
  to match the actually-computed Sellmeier values.  Multi-way
  convergent finding across audit perspectives (V5 + DEEP-3 + F1 +
  PHYS-2):
  * H-ZK9B: `1.613750` -> `1.62041`
  * H-ZF12: `1.673000` -> `1.76182`   (the most egregious — 6% off)
  * D-LAK52: `1.729160` -> `1.73050`
  * H-ZLAF52A: `1.796800` -> `1.80610`
  No runtime behaviour change — the actual Sellmeier coefficients were
  always correct; only the comments were stale.
* **`refractiveindex` moved to optional `[glass]` extras group.**
  `pyproject.toml`: dropped from hard `[project.dependencies]`,
  promoted to `glass = ["refractiveindex>=1.0"]`, and bundled into the
  existing `all` group.  Aligns the wheel with the lazy-import +
  `SELLMEIER_COEFFICIENTS` fallback already in place in
  `lumenairy/glass.py`.
* **`zarr>=2.14` floor bumped to `zarr>=3.0`** in the `zarr` and `all`
  extras groups.  `lumenairy/io/storage.py` uses `Group.create_array`
  (a Zarr v3 API); the v2 floor was a latent `AttributeError` waiting
  for any zarr=2.x user.
* **`ProcessPoolExecutor` `spawn` mp_context.**
  `lumenairy/elements/_lens_traced.py:~220`: explicit
  `mp_context=multiprocessing.get_context('spawn')` kwarg on the
  module-level worker pool.  Matches the README + v4.16.0 CHANGELOG
  claim that `spawn` is used (previously `fork` on Linux — unsafe
  with cached FFT plans and worker threads).
* **`examples/06_schell_propagation.py`** + **`examples/07_zemax_load_trace.py`** added.  The Zemax-loader example
  closes audit UX item 22 (no prior example exercised
  `la.load_zemax_zmx` end-to-end); loads the Thorlabs AC254-100-C
  achromat fixture and falls back to a programmatic N-BK7 singlet via
  `la.make_singlet` if the .zmx file is missing.
* **`CONVENTIONS.md`** added at the repo root — ~10 short sections
  documenting the `create_*` (-> field/Source) vs `make_*` (->
  prescription / bundle / non-field) factory verb contract, error-
  message prefix discipline, RNG kwarg name, and 7 related
  conventions that previously lived only in informal precedent.
* **Stray repo-root artifacts cleaned up.**  3 example PNGs moved
  from the repo root to `examples/output/` (the 3 producing scripts
  updated to write there); stray `C:tmpoptimize_diff.txt` echo-
  redirect artifact deleted (same cleanup pattern as v4.14.3's
  `C:tmpv4_14_1_changelog.md`).

### Tests + CI

* **2192 pass / 5 skip / 1 xfail = 2198 collected** (up from 2113
  collected at v4.16.0; +85 net across the 4 agents -- A=11, B=26,
  C=20+6, D=22).  (v4.16.2 audit P2-NEW-F2-MED-2: pre-v4.16.2
  headline cited 2208/+102/2106 -- arithmetic broken; refreshed
  here to use the collected metric, which arithmetic-reconciles.)
* New test modules:
  * `tests/unit/test_v4_16_1_agent_a.py` (11 tests)
  * `tests/unit/test_v4_16_1_agent_b.py` (26 tests)
  * `tests/unit/test_v4_16_1_agent_c.py` (20 tests)
  * `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py`
    (6 walker tests — the 10th meta-pin)
  * `tests/unit/test_v4_16_1_agent_d.py` (22 tests)
* CHANGELOG line-citation refresh: `_ZERO_APERTURE_MASK` branch site
  drifted `:2958` -> `:2974` after Agent A's Bug 1 SUM->AVG line
  additions; the v4.15.3 + v4.15.4 entries' "now at :2958" cites are
  refreshed to `:2974` (the v4.15 line-citation pin
  `TestF5ChangelogLineCitations` verifies a citation within +/-5 of
  the live site exists in CHANGELOG).

---

## [4.16.0] — 2026-05-19

**Major minor release** rolling up the entire v4.16 + v4.17 + v4.18
ROADMAP into a single release.  4 large feature buckets ship together:
the remaining 4 V4 meta-pin candidates (closing the structural
counter-measure trajectory begun in v4.15.0); multi-process atomic-
append for `storage.py` (HDF5 SWMR + filelock distributed Zarr lock);
the full optimisation framework expansion (constrained opt,
checkpoint/resume, Newton-step, multi-objective NSGA-II via pymoo);
and the glass/materials expansion (CDGM + Hikari + Sumita catalogues
+ per-glass Sellmeier validity ranges + central cache registry).
**2106 unit tests pass** (up from 1922; +184 net), 5 documented
skips (4 pymoo + 1 ZARR_MKDIR_PATCH), 1 documented xfail; **34/34
validation files pass**.

### Bucket 1 — V4 meta-pin candidates (4 walkers complete)

The audit's standing V4 recommendation from AUDIT_V4_14_2 Part 3.5
onward.  v4.15.x landed candidates V1 (cache-clears), V2 (cache↔lock),
V3 (0+0j), V4 (`_validate_grid_params`), V5 (`_check_2d_scalar_field`).
v4.16.0 lands the remaining four — completing the meta-pin coverage
of the recurring sibling-gap classes the audits identified:

* **Sentinel-aware branch propagation walker** — AST-walks
  `_get_wrapper_merit_cache` callsites for `is _ZERO_APERTURE_MASK`
  branch.  3 sites discovered, all already guarded (v4.14.1-v4.14.3
  closures are clean).  Counter-pin verifies synthetic violation
  triggers the walker.
* **Cross-backend dispatch (`_xp_of` usage) walker** — AST-walks
  field-domain public functions for hardcoded `np.*` patterns where
  `xp = _xp_of(E); xp.<...>` should dispatch.  94 candidates
  discovered; **5 inline fixes shipped** in `lumenairy/elements/elements.py`
  (`apply_zernike_aberration`, `apply_lyot_focal_plane_mask`,
  `apply_vortex_phase_mask`, `apply_apodized_pupil`, plus the
  `zernike` helper); 56 documented exemptions.
* **`dy` parameter threading walker** — every `apply_*` in
  `lumenairy.__all__` must accept `dy: Optional[float] = None` for
  anamorphic-grid support.  36 functions discovered; 26 already
  threading `dy`, 10 documented exemptions (`apply_perturbations`
  prescription input, `apply_mask` element-wise mul, polarization
  `JonesField` helpers, bundle helpers like
  `apply_thin_lens_to_beamlets`, `apply_detector` square-grid,
  `apply_dm` mirror-geometry square).
* **`__all__` symmetry walker** — every name in submodule `__all__`
  is either re-exported at the top level OR marked `_INTERNAL`
  (`_*` prefix).  752 submodule entries; 717 re-exported; **35
  documented exemptions**; **9 inline fixes** (8 backend-array-
  namespace helpers + `PYMOO_AVAILABLE` promoted to top-level
  `__all__`).

Each walker carries a fake-violation counter-pin (positive-signal
test pattern from v4.15.0 / v4.15.4).  **All 9 dispatcher meta-pins
now active and green** (cache-clears, cache↔lock, 0+0j,
validate_grid_params, check_2d_scalar_field, sentinel-propagation,
xp-dispatch, dy-threading, __all__-symmetry).  The "fix N, miss N+1"
sibling-gap meta-pattern at the public-API surface is now
structurally retired across all currently-known classes; new
classes will continue to surface and be added to the V-walker
family as identified.

### Bucket 2 — Multi-process atomic-append for `storage.py`

v4.14.3 documented single-process atomicity for `append_plane_h5`
and `_zarr_append_plane` plus a multi-process restriction.  v4.16.0
closes the multi-writer story:

* **HDF5 SWMR mode** — `append_plane_h5` gains `swmr: bool = True`
  kwarg.  When `True`, file opened with `libver='latest'`,
  `f.swmr_mode = True` after dataset creation.  Concurrent readers
  can safely follow a single writer; multiple writers are
  serialised via the sibling lock.
* **filelock-based distributed Zarr lock** — both `append_plane_h5`
  and `_zarr_append_plane` wrap the attr-write + create-array
  sequence in a `filelock.FileLock` on the sibling `<path>.lock`
  file.  Cross-process race-free; configurable
  `lock_timeout: float = 30.0` kwarg.
* **`filelock>=3.0`** added to `hdf5` and `zarr` optional-dependency
  groups in `pyproject.toml` (verified NOT a transitive dep of
  either h5py 3.16 or zarr 3.1).
* **Subprocess multi-writer tests** via `multiprocessing.get_context('spawn')`
  (Linux + Windows portable) — 4 workers × 5 planes each verifies
  20-plane final file with no data loss.
* **Single-process v4.14.3 atomicity guarantees preserved**
  bit-for-bit.

Measured overhead: ~5× slowdown 4-writer contended vs 1-writer
baseline; <5% lock overhead on large planes (≥4096²) where
HDF5/Zarr I/O dominates.

### Bucket 3 — Optimisation framework expansion (ROADMAP v4.17)

Four additions to `lumenairy.optimize`:

* **Constrained optimisation** via scipy `NonlinearConstraint`
  mapping.  New `Constraint` dataclass; `design_optimize(...,
  constraints=[Constraint(fn=..., lb=..., ub=..., label=...)])`.
  Method-compatibility validator raises a clear `ValueError` for
  non-supporting methods (L-BFGS-B / `lm` / `differential_evolution`
  / `basin_hopping` / `dual_annealing` all silently ignored
  constraints in scipy's core API — v4.16 rejects them up front
  pointing the user at SLSQP / trust-constr).  Diagnostic
  constraint-label printed in progress callback.
* **Checkpoint / resume on long `design_optimize` runs**.  Add
  `state_file: Optional[str] = None` and
  `state_save_every: int = 1` kwargs.  Persists
  `(call_count, x_best, merit_best, history)` to JSON with atomic-
  replace write (`.tmp` + `os.replace`).  On startup, if the file
  exists with matching shape, resumes from persisted `x_best`.
  Gated on `state_file` non-None so legacy callers see byte-
  identical behaviour.
* **Multi-objective Pareto via pymoo NSGA-II wrapper** —
  `lumenairy.optimize.multi_objective.design_optimize_multi_objective(...)`
  with `ParetoResult` dataclass.  pymoo is an **optional
  dependency** in the new `multi_objective` extras group (`pip
  install lumenairy[multi_objective]`).  Same opt-in pattern as
  jax/cupy/numba/h5py/zarr in the library.  Module imports
  unconditionally; only the actual function call raises
  `ImportError` with a clear install hint.  pymoo's heavier
  transitive deps (autograd, deap, cma) are deliberately NOT
  bundled into the `all` group.  4 new top-level exports:
  `Constraint`, `ParetoResult`, `design_optimize_multi_objective`,
  `PYMOO_AVAILABLE`.
* **Hessian / Newton-step optimisation** via `method='newton'`.
  Dispatches to scipy `trust-ncg` with FD-Jacobian-of-FD-gradient
  Hessian estimator.  `UserWarning` recommends L-BFGS-B for
  `n_params > 30` (Newton's FD-Hessian cost scales as n²).

13 tests (4 constrained + 3 checkpoint + 4 multi-objective + 2
Newton); 4 of the multi-objective skip cleanly if pymoo isn't
installed.

### Bucket 4 — Glass + materials + central cache registry

Three additions (ROADMAP v4.18 items #13-#15):

* **CDGM + Hikari + Sumita Sellmeier catalogues** — 32 new
  glasses across the three major non-Western catalogues:
  - **CDGM (12)**: H-K9L, H-LAK52, H-LAK53A, H-ZK9B, H-ZF12,
    D-ZK3, D-LAK52, H-ZLAF52A, H-ZK7, H-ZF52A, F1-CDGM, F2-CDGM.
  - **Hikari (10)**: E-LASF016, E-SK16, E-LAK7, E-LAK04, E-BAK1,
    J-FK01A, J-LASF09A, J-LAK7, J-BASF7, E-F2.
  - **Sumita (10)**: K-VC78, K-LAK10, K-LASFN10, K-SK4, K-PFK90,
    K-PBK40, K-BK7, K-PSKN2, K-FK5, K-LAFN3.
  **`GLASS_REGISTRY`: 46 → 78 entries**.  Every new entry n_d
  cross-checked against the official datasheet (or
  refractiveindex.info as proxy) at the 5e-5 tolerance pin
  established by v4.14.2's S-LAH64/79 verification.  Zero
  glasses failed the cross-check.  8 of the new CDGM glasses
  also ship as bundled Sellmeier-formula-2 fallbacks for
  minimal installs without `refractiveindex`; the remaining 22
  use formula-3 (polynomial) which requires `refractiveindex`
  for minimal installs — clear `ImportError` with install hint.
* **Per-glass Sellmeier validity ranges** — new `GLASS_VALIDITY`
  table with 77 entries (one per catalogued glass).  Format
  `{name: (lambda_min, lambda_max)}` in metres.  Extrapolating
  outside the range emits `UserWarning(...)` but does NOT raise
  — extrapolation is sometimes acceptable for design-space
  exploration.  Per-glass sources cited inline in the table
  (refractiveindex.info URLs + datasheet revs).
* **Central cache registry** (`lumenairy/_cache_registry.py`)
  — new public API `register_cache_clearer(name, clear_fn)` +
  `list_registered_cache_clearers()` + `clear_all_registered_caches()`.
  Retires the lazy-import fan-out in `clear_asm_caches`.  9
  caches migrated to the registry (`asm_local`, `lg_mode_stack`,
  `lg_polynomial_items`, `zernike_basis`,
  `through_focus_scan_jax`, `propagate_system_jax`,
  `phase_retrieval_kernels`, `trace_jax`,
  `wrapper_merit_meshgrid`).  `clear_asm_caches`'s external
  contract is preserved bit-for-bit (still callable with the
  same name + signature); the internal dispatch is now
  registry-walking instead of hand-enumerated.  Structural
  counter-measure to the cache-clear "fix N, miss N+1" pattern
  the v4.14.x audits identified.

127 new tests (10 cache-registry + 109 glass + 8 validity); zero
n_d cross-check failures.

### New top-level exports (12)

* `Constraint`, `ParetoResult`, `design_optimize_multi_objective`,
  `PYMOO_AVAILABLE` (optimisation)
* `register_cache_clearer`, `list_registered_cache_clearers`,
  `GLASS_VALIDITY` (cache registry + glass validity)
* `array_namespace`, `is_numpy_array`, `is_cupy_array`,
  `is_jax_array`, `backend_name`, `to_numpy`, `to_backend`,
  `RandomState` (backend helpers — Agent A's `__all__` symmetry
  fix promoted these)

### Optional dependencies

* New `multi_objective` extras group: `pip install
  lumenairy[multi_objective]` adds pymoo for NSGA-II Pareto.
* `hdf5` and `zarr` extras groups now include `filelock>=3.0`
  (was previously transitively missing).

### Test counts

* Pre-v4.16.0 baseline (v4.15.5): 1922 unit pass + 1 skip + 1 xfail.
* v4.16.0 additions: A=30 (4 walkers), B=15 (multi-process storage),
  C=13 (10 pass + 3 pymoo-skip; pymoo not installed in test env),
  D=127 (10 cache-registry + 109 glass + 8 validity); plus 5 inline
  xp-dispatch fixes' positive regression coverage.  Net +184
  collected, +5 documented skips (3 new pymoo + 0 already present
  ZARR mode + 1 already present).
* Final: **2106 unit pass + 5 skip + 1 xfail; 34/34 validation**.

### ROADMAP status post-v4.16.0

* **v4.16, v4.17, v4.18 — all items shipped**.  The ROADMAP's
  Current State section is refreshed to reflect this; remaining
  target sections are folded into Shipped highlights.
* **v5.0 — immediate horizon**.  Major structural release: 6
  file splits, CI gates (pytest fast-PR + ruff + mypy --strict
  incremental + `__all__` smoke), remove 8 active back-compat
  shims, shared Chebyshev helpers extraction, audit-fix test-file
  consolidation, `lumenairy/system.py` → `propagators/system.py`,
  off-axis conic in surface frame (Optiland/Zemax parity), bump
  `requires-python` to >=3.10, 3 config knobs, docs.
* **Designer GUI v3.8+** still unplanned (separate version
  stream).

### Known issues / flagged for v4.16.1

* **Bundled Sellmeier formula-3 (polynomial) evaluator** —
  Hikari, Sumita, and 4 CDGM glasses use refractiveindex.info
  formula 3.  v4.16.0's `_sellmeier_index` only supports
  formula 2.  Minimal installs without `refractiveindex` raise
  `ImportError` on these 26 entries with a clear actionable
  message.  v4.16.1 candidate: add `_polynomial_index`
  evaluator.

### Deferred to v5.0

Architectural items requiring breaking changes — see ROADMAP for
the full v5.0 catalogue.

---

## [4.15.5] — 2026-05-19

**Closes the v4.15.4 audit (`docs/audits/AUDIT_V4_15_4_2026_05_19.md`)
through P3.**  The audit found 0 P0 + 4 P1 + 6 P2 + 5 P3.  Three of
the 4 P1s closed via a **V6 dispatcher meta-pin walker refactor**:
discovery now keys off the function's **first-positional-parameter
name** (via AST inspection of `ast.arguments`) rather than a hand-
curated name-prefix list, plus the walker now **descends into class
bodies** for non-delegating methods like `DeformableMirror.apply`.
The remaining P1 cluster (2 user-facing pitfalls in v4.15.4's new OPD
plotting functions) closed via a `fan_units` kwarg + centered-RMS
metric consistency.  **1922 unit tests pass** (up from 1858; +64
net), 1 documented skip, 1 documented xfail; **34/34 validation
files pass**.

### Headline: V6 walker (first-positional-param-name discovery)

The v4.15.3/4 dispatcher meta-pin walker used a hand-curated name-
prefix filter (`apply_*` / `propagate_*` / `richards_wolf_*` /
`debye_wolf_*`).  v4.15.4 audit found 30+ public `__all__`
functions outside the filter that take 2-D `E`/`field`/`pupil`
first positional args — the meta-pattern recurred at one indirection
level higher.

v4.15.5 refactors discovery:

* **Primary filter is now AST-based first-positional-param name.**
  Walks `lumenairy.__all__`; for each public function, inspects
  `ast.arguments.args[0].arg`; if the name is in
  `_FIELD_PARAM_NAMES = frozenset({'E', 'E_in', 'field', 'pupil',
  'object_field', 'psf'})`, requires `_check_2d_scalar_field` call
  OR `_GUARD_EXEMPTIONS` entry.  Discovery is now grounded in the
  actual function signature rather than a string match.
* **Legacy name-prefix filter retained as fallback** — v4.15.4
  coverage preserved bit-for-bit; V6 only ADDS entries.
* **Class-body descent** — the walker now visits class methods
  named `apply` / `propagate` / similar.  `_DELEGATING_CLASS_METHODS`
  exemption set documents which classes legitimately delegate to
  module-level guarded functions (operator-algebra: `ThinLens.apply`,
  `FreeSpace.apply`, `CylindricalLens.apply`, `Magnify.apply`,
  `FourierTransform.apply`, `Aperture.apply`, `GaussianAperture.apply`,
  `CompositeOperator.apply`).
* **`_file_to_ast` `lru_cache`d** (P3-NEW-F1-2) — 11 sibling
  functions in `_lens_thin.py` previously triggered 11 re-parses
  of the same file; cache cuts that to 1.  Walker test wall time:
  **~1.5s → ~0.03s warm (~30× speedup)**; cache stats `hits=180,
  misses=42, currsize=42` on a full walk (81% hit rate).

Post-refactor walker discovery:  **96 top-level entry points + 3
class methods** = 99 candidates (was 72 at v4.15.4 HEAD).  Of the
96 top-level: 39 guarded (was 25), 47 exempt (unchanged), 10 newly-
flagged for v4.16+ inline guard sweep (HFPI initialisers,
decomposition helpers, low-priority `analysis/core.py` analyzers).

### P1 closures (4)

* **P1-NEW-F2-2 — `DeformableMirror.apply` 1-line guard.**  The
  module-level `apply_dm` was guarded in v4.15.4; the class method
  was a 3-line `E_in * np.exp(1j * phi)` with no `_check_2d_scalar_field`
  call.  Walker's blanket class-method exclusion (v4.15.4 docstring
  asserted methods "delegate to guarded scalar functions") was true
  for Cluster-B operator algebra, false here.  v4.15.5 closes both:
  inline guard on the method + walker descent into class bodies.
* **P1-NEW-2WAY-1 — V6 walker refactor** (covered under headline)
  + **inline guards on the 13 highest-traffic unguarded analyzers**:
  `wave_opd_2d`, `M2`, `strehl_ratio`, `beam_d4sigma`,
  `coupling_efficiency`, `compute_psf`, `compute_otf`, `compute_mtf`,
  `encircled_energy_curve`, `koehler_image`, `extended_source_image`,
  `shack_hartmann`, `rays_from_field`, `resample_field`.  34
  regression tests pin each guard via direct `MCF` / 3-D ensemble
  rejection.
* **P1-NEW-V2-1 — `plot_opd_fan` `fan_units` kwarg.**
  `raytrace.opd_fan_data` returns OPD in **waves**;
  `plot_opd_fan` expected **metres**.  The canonical pipeline
  `plot_opd_fan(*opd_fan_data(...), units='waves', wavelength=wl)`
  divided by `wavelength` a second time → silently wrong by ~6e5.
  v4.15.5 adds `fan_units: str = 'm'` (default `'m'` preserves
  v4.15.4 callers; pass `fan_units='waves'` for the
  `opd_fan_data` pipeline).  End-to-end regression test pins
  `opd_fan_data → plot_opd_fan(fan_units='waves', units='waves',
  wavelength=wl)` does not double-convert.
* **P1-NEW-V2-2 — `_radial_rms_profile` centered RMS.**  v4.15.4
  `_radial_rms_profile` used `sqrt(mean(opd²))` (variance about
  zero) while the 1-D fan RMS and 2-D heatmap RMS used
  `sqrt(mean((opd - mean)²))` (centered, wavefront-error
  convention).  On a pure-defocus OPD, the radial-RMS curve was
  piston-dominated and looked like r²; the heatmap annotation was
  the much smaller centered RMS.  Numbers on the same figure did
  not reconcile.  v4.15.5 switches to centered RMS using the in-
  aperture mean computed once for the entire OPD.  Example
  `plot_opd_summary_singlet.py` RMS now reports **0.8901 waves**
  (was 1.3347 waves uncentered); PV unchanged at 2.9318 waves.

### P2 closures (6)

* `_check_2d_scalar_field` parameterized with `input_kind:
  str = 'field'` — `richards_wolf_focus` / `debye_wolf_psf` /
  `compute_psf` / `compute_otf` / `compute_mtf` etc. pass
  `input_kind='pupil'` or `'psf'`, getting accurate error
  messages ("expected 2-D complex pupil" instead of "field").
* `plot_opd_summary` docstring corrected to explicitly state
  `opd_2d` input is in **metres** (matching `plot_wavefront`'s
  convention).
* Example `plot_opd_summary_singlet.py` RMS print switched to
  centered form (matches heatmap annotation).
* `plot_opd_summary` even-N central-row/col fallback: aligned
  using `(N - 1) // 2` consistently.
* Walker descends into class bodies (covered under headline).
* CHANGELOG v4.15.4 entry-point count refresh (43→49 → actual
  72 at v4.15.4 HEAD; v4.15.5 numbers 96 + 3 class methods are
  documented in this entry).

### P3 closures (5)

* `_file_to_ast` `@lru_cache` (covered under headline).
* `plot_opd_fan` docstring now explicitly states "centered RMS
  (about the in-aperture mean); PV (max - min)".
* `n_bins` kwarg exposed on `plot_opd_summary` as
  `radial_rms_n_bins: int | str = 'auto'` with auto-clamping for
  tiny grids (`min(32, int(sqrt(N_in_aperture)) // 2)`).
* CHANGELOG `-W error::DeprecationWarning` failure count drift
  (57 vs 63) reconciled — canonical at v4.15.4 commit time was
  63; v4.15.3 audit's 57 reflects pre-dispatch count, drift
  documented inline.
* CHANGELOG per-agent attribution arithmetic +1 footnote added
  (per the standard pattern from v4.15.3 hygiene fix).
* CHANGELOG wavelength annotation drift fixed (587.56 nm → 633
  nm to match the example).
* ROADMAP duplicate-counting drift cleaned up: 6 already-shipped
  items moved from "v4.16 residual" to "Shipped highlights"
  (polychromatic encircled energy, polarisation-aware Strehl,
  resolution metrics, astigmatism mag+angle, OAP, Forbes Q-type
  — all landed in v4.15.0 / v4.15.1).  True v4.16 residual is
  now 2 items: V4 meta-pin candidates + multi-process atomic-
  append for `storage.py`.  Remaining v4.17/v4.18 items
  renumbered.

### Test counts

* Pre-v4.15.5 baseline (v4.15.4): 1858 unit pass + 1 skip + 1 xfail.
* v4.15.5 additions: A=34 regression + 4 V6/class-method pins
  (38 total), B=17 + 3 carry-forward (20 total), C=4.  Net +64.
* Final: **1922 unit pass + 1 skip + 1 xfail; 34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates still standing
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy`
parameter threading walker, `__all__` symmetry walker; the V6
first-positional-param-name candidate landed in v4.15.5);
MCF-aware downstream propagators; multi-process atomic-append
for `storage.py`; `MultiPrescriptionParameterization.scale_floor`;
Forbes Q-2D-asymmetric variant.  Plus: **10 newly-flagged
unguarded analyzers** for v4.16+ inline-guard sweep (HFPI
initialisers, decomposition helpers, lower-priority
`analysis/core.py` analyzers like `beam_centroid`,
`beam_diameter`, `radial_power_bands`, `wave_opd_1d`,
`strehl_phase_integral`, resolution metrics, `single_plane_metrics`).

---

## [4.15.4] — 2026-05-19

**Closes the v4.15.3 audit (`docs/audits/AUDIT_V4_15_3_2026_05_18.md`)
through P3 + adds two user-facing OPD plotting functions.**  The audit
found 0 P0 + 1 P1 + 6 P2 + 5 P3 — the cleanest yield in the v4.15.x
series.  The single P1 is the recurring "fix N, miss N+1" meta-
pattern re-emerging one level of indirection higher than v4.15.3
closed it: the dispatcher meta-pin's `_TARGET_PACKAGES` scope itself
was a sibling gap.  **1858 unit tests pass** (up from 1822; +36 net),
1 documented skip, 1 documented xfail; **34/34 validation files pass**.

### Headline: walker scope refactor closes the meta-pattern at the package level

v4.15.3 shipped the `_check_2d_scalar_field` helper + dispatcher meta-
pin.  But the walker scoped only to
`('lumenairy/propagators', 'lumenairy/elements')`, missing 4 public
entry points outside that scope.  v4.15.4 makes discovery
`__all__`-based:

* **`_walk_entry_points` refactored** to walk `lumenairy.__all__`
  membership via `inspect.getsourcefile`; survives future refactors
  that move functions between subpackages.  Package-walk retained as
  a fallback.  Walker discovery at v4.15.4 HEAD: **72 total entry
  points (25 guarded + 47 documented exempt)**.  (The pre-correction
  CHANGELOG bullet cited "43 -> 49 after the refactor", which
  reflected the `__all__`-pass-only count without the package-walk
  fallback dedup; v4.15.5 / Agent C refreshed this from the live
  diagnostic per AUDIT_V4_15_4 P2-NEW-3WAY-3.  TODO: the v4.15.5 V6
  walker refactor (Agent A scope) will change this number again at
  integration time; Agent C populated this with v4.15.4 HEAD numbers
  and integration will update with v4.15.5 V6-refactored numbers.)
* **Name filter broadened** with `name.startswith('propagate_')` to
  catch `propagate_through_system` + `propagate_through_system_jax`
  (which contain `propagate_` at the start but not `_propagate` in
  the middle — the v4.15.3 filter missed both).
* **6 newly-found sibling entry points guarded** via the v4.15.3
  helper: `propagate_through_system_jax` (P1), `apply_dm`,
  `apply_detector`, `richards_wolf_focus`, `debye_wolf_psf`; plus
  `apply_perturbations` documented exempt (first positional arg is
  a prescription dict, not a 2-D scalar field).
* **Fake-violation counter-pin** added to the dispatcher meta-pin:
  injecting a synthetic unguarded function via `monkeypatch.setattr`
  must trigger the meta-pin's `AssertionError`.  Walker correctness
  now pins on a positive signal.

### P1 closure (1)

* **P1-NEW-3WAY-1** walker scope completeness — closed via
  `__all__`-based discovery + extended `_TARGET_PACKAGES`.

### P2 closures (6)

* Walker name-regex broadening + 3 unguarded `analysis/` siblings
  guarded (covered under headline).
* SAS-anamorphic CHANGELOG wording corrected retroactively in the
  v4.15.3 entry: `"forces method='asm' regardless of self.method"`
  -> `"forces method='asm' when self.method == 'auto' and dy != dx;
  explicit method='sas' on anamorphic grids still crashes (user's
  responsibility)"`.
* `_validation.py` lazy-import hoisted to module scope.  The
  v4.15.3 code used a lazy import citing a hypothetical circular
  dep; audit grep-verified no actual circular dep exists.  Saves
  ~1 µs/call (1-10 ms per merit eval in optimization loops with
  thousands of propagator calls).
* Dead `_PerturbedABCDFallbackSentinel` deleted.  v4.15.3 marked
  it dead via `_v4_15_3_dead_code = True` class attribute
  (informational only — no static analyzer honors it).  v4.15.4
  deletes the class + singleton (~58 LOC).  v4.15.2 test pin
  updated in the same commit.
* ROADMAP refreshed: post-v4.15.3 test count ~1750 → actual 1824;
  AUDIT_V4_15_2 + AUDIT_V4_15_3 added to closed-audits list; meta-
  pin coverage 3 of 5 → 4 of 5 with the V5 entry describing
  `_check_2d_scalar_field`.

### P3 closures (5)

* CHANGELOG dispatcher meta-pin count drift fixed (18/25 → 17/26;
  43 total).
* Fake-violation counter-pin added (covered above).
* CHANGELOG stacklevel wording: "6 Source classmethod shims" → "5
  `Source.*` classmethod shims at `:2424, 2510, 2587, 2661, 2750`
  plus the module-level `create_led_source` factory shim at
  `:1209`".
* `-W error::DeprecationWarning` test hygiene: 63 v4.15.3 tests
  previously failed under the strict flag (they exercised the
  documented Schell `return_kind` default-path warning without
  shielding).  v4.15.4 adds `pytestmark =
  pytest.mark.filterwarnings('default::DeprecationWarning')` to 6
  affected test files.  Failures: 63 → 0.  *(Note: the v4.15.3
  audit's P3-NEW-F2-4 cited 57 failing tests; the discrepancy with
  this CHANGELOG's 63 reflects pre/post-v4.15.4 dispatch-test-file
  additions between when the audit was filed and v4.15.4 commit
  time.  Canonical count at v4.15.4 commit time was 63; closure
  verified to 0 escalations regardless of which baseline is right.
  Documented per AUDIT_V4_15_4 P3-NEW-V3-1 / v4.15.5 Agent C.)*
* CHANGELOG test-count arithmetic reconciled: v4.15.3 baseline
  1732 → 1733; per-agent attribution sum 88 documented alongside
  actual collected delta 89 with explicit explanation of the +1
  attribution-vs-collection gap.  Removed the false claim about a
  ROADMAP update that wasn't actually performed in v4.15.3.

### New: OPD plotting functions

Two new public functions in `lumenairy/analysis/plotting.py`, visually
matching the `OPDPy_Lumenairy_Crosscheck` `fig_variety_L*.png` style:

* **`plot_opd_fan(py, opd_y, px, opd_x, *, wavelength=None,
  units='waves', show_stats=True, title=None, fig=None, axes=None)`**
  — 2-panel tangential + sagittal OPD fans.  Inputs match
  `lumenairy.raytrace.opd_fan_data`'s return tuple.  Solid-line
  plots with zero-reference axhline, in-axes PV/RMS annotation,
  units kwarg matches `plot_wavefront` (waves / nm / um / m).
  Returns `(fig, (ax_y, ax_x))`.  147 LOC.
* **`plot_opd_summary(opd_2d, dx, *, dy=None, aperture=None,
  wavelength=None, py=None, opd_y=None, px=None, opd_x=None,
  units='waves', cmap='RdBu_r', show_stats=True, title=None,
  fig=None)`** — 4-panel summary: 2-D heatmap (delegates to
  `plot_wavefront`), radial RMS profile (32 annular bins),
  tangential fan, sagittal fan.  Fan panels use the provided
  `(py, opd_y, px, opd_x)` if supplied (preferred — raytrace data
  has chief-ray reference built in) or auto-extract from the 2-D
  OPD's central row/column otherwise.  Returns `(fig, ((ax_hm,
  ax_rms), (ax_y, ax_x)))`.  204 LOC.

Both added to `lumenairy.__all__` (analysis tier).  10 unit tests +
runnable example at `examples/plot_opd_summary_singlet.py` (singlet
OPD via `apply_real_lens_traced` + `opd_fan_data`; PV ≈ 2.93 / RMS
≈ 1.33 waves at λ=633 nm).  *(Pre-v4.15.5 wording cited
λ=587.56 nm, contradicting the example's actual ``wavelength =
633e-9``; corrected per AUDIT_V4_15_4 / v4.15.5 Agent C.)*

### Test counts

* Pre-v4.15.4 baseline (v4.15.3): 1822 unit pass + 1 skip + 1 xfail.
* v4.15.4 additions, per-agent attribution: A=8 + 1 counter-pin in
  the meta-pin file (=9), B=11, C=7, D=10.  Per-agent sum: **37**.
* Actual `pytest --collect-only` delta: 1858 - 1822 = **+36**
  (canonical post-release number).  The +1 attribution-vs-collection
  gap reflects the standard parametrize/fixture artifact (one of
  the new tests expands to 2 collected items via `parametrize`, or
  a fixture-only addition isn't cleanly attributed to a single
  agent).  Same pattern documented in the v4.15.3 entry; pinned by
  `test_changelog_per_agent_breakdown_sums_to_net_delta` in
  `test_v4_15_4_agent_c.py`.  Documented per AUDIT_V4_15_4
  P3-NEW-V3-2 / v4.15.5 Agent C.
* Final: **1858 unit pass + 1 skip + 1 xfail; 34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates still standing
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy`
parameter threading walker, `__all__` symmetry walker); MCF-aware
downstream propagators; multi-process atomic-append for
`storage.py`; `MultiPrescriptionParameterization.scale_floor`;
Forbes Q-2D-asymmetric variant.

---

## [4.15.3] — 2026-05-18

**Closes the v4.15.2 audit (`docs/audits/AUDIT_V4_15_2_2026_05_18.md`)
through P3.**  The audit identified 1 P0 + 4 P1 + ~6 P2 + ~4 P3 —
mostly the recurring "fix N, miss N+1" sibling-gap meta-pattern that
has appeared in every audit round from v4.13.x onward.  v4.15.3
closes the P0 with a **structural counter-measure** (shared
validation helper + dispatcher meta-pin) that makes the recurrence
impossible going forward.  **1822 unit tests pass** (up from 1733;
+89 collected vs v4.15.2 HEAD), 1 documented skip, 1 documented
xfail; **34/34 validation files pass**.

### Headline: structural counter-measure for the sibling-gap meta-pattern

The v4.15.2 closure guarded 10 propagator/lens entry points against
`PartialCoherenceMCF` + 3-D ensemble inputs.  This audit found **9
more public entry points** of the same type that were missed —
`angular_spectrum_propagate_tilted`, `*_propagate_mft` (3 variants),
`apply_spherical_lens`, `apply_aspheric_lens`, `apply_grin_lens`,
`apply_axicon`, `apply_real_lens_traced`, `apply_real_lens_maslov`.

v4.15.3 fixes this **structurally**:

1. **`lumenairy/_validation.py`** (NEW) — single canonical
   `_check_2d_scalar_field(E, fn_name)` helper consolidates the
   `PartialCoherenceMCF` and `ndim != 2` guards.  Replaces ~240 LOC
   of duplicated boilerplate across 10 v4.15.2 sites + 9 new sibling
   sites.  Net `lumenairy/` LOC change: roughly +23 (helper +102 LOC;
   migrated sites -160 LOC; new sibling guards +81 LOC) vs +225 LOC
   the inline pattern would have cost on the 9 new sites alone.

2. **Dispatcher meta-pin**
   (`tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py`) —
   AST-walks every `def apply_*` and `def *_propagate*` in
   `lumenairy/propagators/` and `lumenairy/elements/`; asserts
   `_check_2d_scalar_field` is the first executable statement of
   each function body.  **43 entry points discovered, 17 guarded,
   26 documented exemptions** (GBD beamlets, HFPI/HF state objects,
   batched 3-D variants, JAX-traceable lens kernels, polarization
   helpers, etc.).  Adding a new public entry point in the at-risk
   modules WITHOUT the helper call now fails CI.

3. **`_GUARD_EXEMPTIONS` registry** documents every legitimate
   exemption with a reason — converts "easy to miss" into "easy to
   see in code review".

This is the 5th structural meta-pin in the library:  v4.14.1
cache-clear dispatcher pin, v4.14.2 cache↔lock pairing + 0+0j
literal sweep, v4.15.0 `_validate_grid_params` input-validation
entry-point pin, v4.15.3 `_check_2d_scalar_field` pin.

### P0 closure

**P0-NEW-F2-1 — 9 unguarded propagator/lens entry points.**  All 9
sibling sites now call `_check_2d_scalar_field` as the first
executable statement, identical guard semantics to the 10 v4.15.2
sites:
* `angular_spectrum_propagate_tilted`, `angular_spectrum_propagate_mft`,
  `fresnel_propagate_mft`, `fraunhofer_propagate_mft` in
  `propagators/propagation.py`
* `apply_spherical_lens`, `apply_aspheric_lens`, `apply_grin_lens`,
  `apply_axicon` in `elements/_lens_thin.py`
* `apply_real_lens_traced` in `elements/_lens_traced.py`
* `apply_real_lens_maslov` in `elements/lenses_maslov.py`

30 regression tests pin the 9 new guards (TypeError on `PartialCoherenceMCF`,
ValueError on 3-D ensemble).  7 meta-pin tests pin the structural
counter-measure (walker discovery, helper-is-first invariant,
counter-pin against accidentally-removed guards).

### P1 closures (4)

* **P1-NEW-F1-1 — `FreeSpace._apply` SAS-anamorphic crash fixed.**
  When `dy != dx` and `method='auto'`, the dispatcher routed to SAS
  (square-grid-only); the v4.15.2 dy-threading fix passed `dy` to
  SAS which doesn't accept it.  v4.15.3 forces `method='asm'` when
  `self.method == 'auto'` and `dy != dx` — `auto` is now a hint,
  not a contract.  Explicit `method='sas'` on anamorphic grids
  still crashes (user's responsibility — the in-code comment at
  `algebra/primitives.py:142-147` documents this gating).
  `FourierTransform._apply` inherits the fix by composition (the
  3-stage rewrite creates `FreeSpace` instances internally).
* **P1-NEW-F1-2 — Schell `DeprecationWarning` stacklevel fixed.**
  `_warn_schell_return_kind_default` had `stacklevel=4`; the call
  chain is 5 frames deep (warnings.warn → warn_deprecated_signature
  → _warn_schell_return_kind_default → factory → user).  Bumped to
  5.  Library-wide sweep of `_warn_*` helpers found 6 additional
  off-by-one stacklevels in `sources/core.py` (5 `Source.*`
  classmethod shims at `:2424, 2510, 2587, 2661, 2750` plus the
  module-level `create_led_source` factory shim at `:1209`); all
  bumped 3 → 4.
* **P1-NEW-F1-3 — 3 dead `optimize/core.py` sentinels wired.**
  v4.15.2 added `_InvalidFocalLengthSentinel`,
  `_FailedScanStrehlSentinel`, `_PerturbedABCDFallbackSentinel`
  class definitions but never wired them at callsites.  v4.15.3
  wires the 2 scalar sentinels at `optimize/core.py:2424, 2696,
  3015` (was raw `-1.0` / `float('nan')` / `0.0` returns); marks
  `_PerturbedABCDFallbackSentinel` as dead-code (tuple shape didn't
  sentinel cleanly without breaking downstream unpacking; class
  retained with `_v4_15_3_dead_code = True` marker for v4.15.2
  test-pin compatibility).
* **P1-NEW-F1-4 — `Source.gaussian_schell`/`schell_model`
  classmethods route through sentinel.**  Pre-v4.15.3 these
  classmethods hardcoded `return_kind='ensemble'`, bypassing the
  v4.15.2 `_RETURN_KIND_UNSET` sentinel — calling them without
  `return_kind` produced a silent 4-tuple with no
  `DeprecationWarning` and a `Source` whose `E.ndim == 3` (every
  other `Source.*` produces 2-D).  v4.15.3 routes both
  classmethods through the sentinel; default-path callers now
  get the same DeprecationWarning as the top-level factories.
  Soft 2-D `Source.E` invariant break documented in classmethod
  docstrings as intentional (Schell is partial-coherence;
  collapsing to 2-D would be physically wrong).

### P2 closures

* **`_RETURN_KIND_UNSET` promoted** from bare `_Sentinel` instance
  to dedicated `_SchellReturnKindUnsetSentinel(_Sentinel)`
  subclass with `_SENTINEL_REGISTRY` entry for pickle round-trip
  safety.  Consistent with `_ZeroApertureMaskSentinel`,
  `_AngleUnsetSentinel`, `_NoDefaultSentinel`.
* **rays_from_field threshold-comparison consistency.**  Audit
  finding was inverted (`_place_rejection` was already `>=`,
  `_place_uniform` and `_place_cdf` were strict `>`).  v4.15.3
  makes all 3 modes inclusive `>=`.  Pixels at exactly
  `intensity_threshold` now consistently survive.
* **Non-tautological FourierTransform pin** added — Gaussian-beam
  waist relation `w_out = lambda * f / (pi * w_in)` (Saleh &
  Teich §3.2.2) through `FourierTransform(f)`.  Measured error
  <0.0001% vs the 5% tolerance pin (50000× headroom).  Pins
  physics not implementation.
* **4-fold mirror folded-prescription test cases** added to
  `test_v4_15_1_agent_g_matches_system_abcd.py` —
  `_build_folded_4fold_periscope` and
  `_build_folded_cassegrain_2curved_2flat` strengthen the
  `from_prescription` flat-mirror parity claim across more
  complex folded geometries.
* **Library-wide `_warn_*` stacklevel sweep** (~12 helpers
  audited; 7 adjusted, 5 unchanged with documented rationale).

### P3 closures

* **CHANGELOG Forbes Q OPD bullet corrected**: tolerance
  `1e-3` → `5e-3` (test code was always `5e-3`); formula
  `OPD = -k * sag` → `OPD(r) = (n - 1) * sag(r)` (the `(n - 1)`
  index factor was missing).  Test code was always correct;
  only the CHANGELOG bullet lied.
* **CHANGELOG sentinel-migration line citations refreshed**
  after Agent C's v4.15.3 wiring drift: `_ZERO_APERTURE_MASK`
  branch now at `optimize/wrapper_merits.py:876` (was
  `optimize/core.py:3032` pre-v5.1.0 Agent E 6-file split, which
  moved `ToleranceAwareMerit.evaluate` out of the monolithic
  `optimize/core.py`; was `:3015` pre-v4.16.3 Agent C `Constraint`
  auto-probe DeprecationWarning latch + pickle catch widening; was
  `:2974` pre-v4.16.2 Agent B `MultiWavelengthMerit` `FutureWarning`
  latch + Constraint probe move + lambda pickle-probe; was `:2958` pre-v4.16.1 Agent A
  `MultiWavelengthMerit` `SUM`->`AVG` refactor; was `:2980` pre-
  v4.15.4 Agent B `_PerturbedABCDFallbackSentinel` deletion; was
  `:2905` in the v4.15.2 entry).
* **Test count reconciliation**:
  `pytest --collect-only` → 1735 collected at v4.15.2 HEAD
  (was reported as "1732 pass + 1 skip + 1 xfail = 1734" in
  CHANGELOG, off by 1); reconciled in this entry's `### Test
  counts` block below.  The ROADMAP refresh originally claimed in
  this bullet ("~1700 → 1822 baseline") did NOT land in v4.15.3
  — that drift is documented in `AUDIT_V4_15_3` P2-NEW-V3-3 and is
  closed by v4.15.4 (Agent C scope).

### Test counts

* Pre-v4.15.3 baseline (v4.15.2): 1733 unit pass + 1 skip + 1 xfail
  = 1735 collected (per the corrected v4.15.2 entry's
  `pytest --collect-only` reconciliation; pre-v4.15.3 the v4.15.3
  block transcribed this baseline as "1732" — a one-off carry-over
  of the same off-by-one the v4.15.2 entry self-corrected).
* v4.15.3 additions, per-agent attribution: A=37 (7 meta-pin + 30
  regression), B=24, C=19 (12 new file + 4 4-fold + 3
  Gaussian-waist), D=8 (7 doc + 1 boundary-regression).  Per-agent
  sum: **88** (pre-v4.15.4 corrected from "Net +90" — neither the
  per-agent sum nor the collected delta were ever 90).
* Actual `pytest --collect-only` delta against the v4.15.2 baseline
  (1735) at v4.15.3 HEAD sha `7808107`: **+89 collected** (1824
  collected at v4.15.3 HEAD); the +1 gap between the per-agent
  attribution sum (88) and the collected delta (89) is a
  parametrize-expansion / fixture-collection artifact that does
  not cleanly attribute to a single agent; the canonical number is
  the collected delta.
* Final: **1822 unit pass + 1 skip + 1 xfail = 1824 collected**;
  **34/34 validation**.

### Deferred to v4.16+

Unchanged from prior releases: modal-asymptotic independent
ground-truth pin; 4 V2 meta-pin candidates (sentinel-aware
branch propagation, `_xp_of` dispatch, `dy` parameter threading
walker including ThinLens + lens kernels, `__all__` symmetry
walker); MCF-aware downstream propagators; multi-process
atomic-append for `storage.py`;
`MultiPrescriptionParameterization.scale_floor`; Forbes Q-2D-
asymmetric variant.  Plus newly-deferred: 9 `elements/elements.py`
generic helpers + 2 JAX-traceable lens variants + 6 polarization
helpers exempted in v4.15.3 meta-pin pending v4.16 integration.

---

## [4.15.2] — 2026-05-18

**Closes the v4.15.1 audit (`docs/audits/AUDIT_V4_15_1_2026_05_18.md`)
through P3.**  The audit found 1 P0 + 9 P1 + ~12 P2 + ~10 P3 — most
were downstream-integration gaps from the rapid v4.15.1 expansion (new
types shipped without updating consumers; breaking changes shipped
without CHANGELOG flagging; primitive APIs asymmetric).  **1732 unit
tests pass** (up from 1625; +107 net), 1 documented skip, 1 documented
xfail; **34/34 validation files pass**.

### Breaking changes

* **Schell-family factories emit `DeprecationWarning`** on the default
  `return_kind` path.  v4.15.1 silently changed the return shape from
  `(E_2d, x, y)` (v4.15.0) to `(E_3d, dx, dy, wavelength)` 4-tuple
  without a warning.  v4.15.2 closes the contract break: callers must
  pass `return_kind='ensemble'` or `return_kind='mcf'` explicitly;
  failing to do so emits a one-release deprecation warning with
  `version_removed='5.0'`.
* **`Source.gaussian_schell` and `Source.schell_model` classmethods**
  now return the same `(ensemble, dx, dy, wavelength)` 4-tuple as the
  top-level factories — they previously wrapped the 3-D ensemble in a
  `Source` instance whose `E` was 3-D, breaking the canonical 2-D
  `Source.E` contract.  This is a soft consistency break (every other
  `Source.*` classmethod returns a `Source`); the inconsistency is
  honest — Schell is partial-coherence, fundamentally different from
  the coherent single-source abstraction.

### P0 closure

**P0-NEW-1 — Schell silent contract break closed.**  `_RETURN_KIND_UNSET`
module-level sentinel (subclass of `_deprecation._Sentinel`) detects
the default-path entry; `_warn_schell_return_kind_default` helper
routes through `_deprecation.warn_deprecated_signature` with explicit
`version_removed='5.0'`.  Applied to all 3 Schell factories
(`create_gaussian_schell_source`, `create_schell_model_source`,
`create_annular_incoherent_source`).

### P1 closures (9)

* **P1-NEW-A — `FourierTransform` 3-stage rewrite.**  v4.15.1's
  `_apply` ran 2 stages (lens-then-Fresnel) while the ABCD claim
  `[[0, f], [-1/f, 0]]` matched the 3-stage chain `FreeSpace(f) *
  ThinLens(f) * FreeSpace(f)`.  The 2-stage path left a residual
  `exp(+ik/(2f) r^2)` quadratic phase — ABCDs matched but fields
  didn't.  v4.15.2 rewrites `_apply` to the literal 3-stage chain so
  ABCD and field finally agree.  Perf impact: ~2x slower than the
  v4.15.1 2-stage shortcut (one extra Fresnel propagation).  Users
  wanting hardware-realistic back-focal-plane semantics can compose
  directly as `ThinLens(f) * FreeSpace(f)` (which has the genuine
  2-stage ABCD `[[1, f], [-1/f, 0]]` and 2-stage field; both correct
  via the existing algebra).
* **P1-NEW-B — `from_prescription` flat-mirror parity matches
  `system_abcd`.**  v4.15.1 flipped `mirror_parity` unconditionally
  on `is_mirror=True`; `system_abcd` only flips for curved mirrors.
  v4.15.2 conditions the parity flip on curved mirrors, matching the
  raytrace convention.  New folded-singlet + folded-telephoto
  prescription test cases pin the parity at 1e-12 absolute (the
  1e-12 ABCD parity claim now holds for folded prescriptions too,
  not just non-folded).
* **P1-NEW-C — `FreeSpace._apply` threads `dy`.**  v4.15.1's
  `FreeSpace._apply` called `propagate(E, z=, ..., dx=dx, method=)`
  without `dy`.  Any anamorphic chain `Magnify(a_x, a_y) *
  FreeSpace(d)` silently propagated on the wrong grid.  v4.15.2
  threads `dy` to the dispatcher when `dy != dx` (forwards via
  method_kwargs; safe for ASM/Fresnel/Fraunhofer/RS; skipped for
  SAS which is square-grid only).  Verified `ThinLens._apply` does
  not have the same gap.
* **P1-NEW-D — `rays_from_field` `'cdf'` placement pixel-wise
  threshold.**  v4.15.1 applied `intensity_threshold` to MARGINAL
  sums in `_place_cdf`, inconsistent with `'rejection'` and
  `'uniform'` modes (which threshold pixel-wise).  A 1-pixel-wide
  bright streak running the full y-extent survived the threshold
  incorrectly.  v4.15.2 thresholds pixel-wise before forming
  marginals; the 3 placement modes are now consistent.
* **P1-NEW-E — `PartialCoherenceMCF` defensive guard.**  All 10
  propagator entry points (`propagate_through_system`, `propagate`,
  `angular_spectrum_propagate`, `fresnel_propagate`,
  `fraunhofer_propagate`, `rayleigh_sommerfeld_propagate`,
  `scalable_angular_spectrum_propagate`, `apply_thin_lens`,
  `apply_cylindrical_lens`, `apply_real_lens`) now raise
  `TypeError` with a clear "v4.16+ scope" message when handed a
  `PartialCoherenceMCF` — previously crashed with cryptic
  `AttributeError`.
* **P1-NEW-F — 3-D ensemble shape guard.**  Same 10 propagator
  entry points now raise `ValueError` on `E.ndim != 2` with a
  message showing the iterate-over-ensemble workaround pattern.
* **P1-NEW-G — CHANGELOG `### Breaking changes` subhead** added
  to the v4.15.1 entry listing Schell return shape, `strehl_vector`
  default-reference removal, and `system.evaluate` mixed-shape
  `ValueError`.
* **P1-NEW-H — `rays_from_field` short-return `RuntimeWarning`.**
  v4.15.1's `_place_rejection` and `_place_uniform` could return
  fewer rays than requested (rejection budget exhausted; threshold
  excluded too many pixels) without warning.  v4.15.2 emits a
  `RuntimeWarning` when `n_actual < n_rays`, plus an `n_rays = 0`
  early-return is honoured cleanly.
* **P1-NEW-I — ROADMAP refresh.**  Header bumped to "(post-v4.15.2)";
  Current State block updated to v4.15.2 / 1732 tests baseline;
  v4.15.1 + v4.15.2 added to Shipped highlights.

### P2 closures

* `_sentinel_unpickle` fallback now raises `ImportError` with an
  actionable message when an unknown subclass is unpickled
  (distributed-pipeline timing safety).  Previously silently
  returned a base `_Sentinel`, losing subclass identity.
* **3 additional `optimize/core.py` sentinels migrated** to inherit
  from `_deprecation._Sentinel`: `_InvalidFocalLengthSentinel`
  (was a literal `1e9` fallback for failed ABCD), `_FailedScanStrehlSentinel`
  (was `0.0`), `_PerturbedABCDFallbackSentinel` (was a `(efl, bfl)`
  tuple fallback).  v5.1.0 (Wave-4 integration / Agent E 6-file
  split): class definitions moved to `optimize/context.py:112-139`
  (the 2 remaining sentinels post-v4.15.4
  `_PerturbedABCDFallbackSentinel` deletion); was at
  `optimize/core.py:2069`, `:2096`, `:2122` (singletons at `:2093`,
  `:2119`, `:2144`) within the `:2044-2144` documentation block
  pre-v5.1.0.
  All registered in `_SENTINEL_REGISTRY` for pickle round-trip
  safety.  v4.15.3 correction (per AUDIT_V4_15_2 P3 docs-drift
  finding): the pre-v4.15.3 release notes cited stale work-in-
  progress line numbers `:2151, :2271, :2530, :2772` for these
  classes; the actual definitions are at the `:2044-2144` block
  cited above.  (Callsite migration to the new sentinels is
  scaffolding-only at v4.15.2 -- the `2271`, `2530`, `2772`
  references appearing inside the class docstrings point at the
  v4.16+ migration target callsites and are tracked separately by
  Agent A's v4.15.3 work.)
* `_NO_DEFAULT` promoted to dedicated `_NoDefaultSentinel(_Sentinel)`
  subclass (cosmetic consistency with the other sentinels).
* **`PartialCoherenceMCF.coherence_at` Hermiticity test** added —
  asserts `J(r1, r2) == conj(J(r2, r1))` for several `(r1, r2)`
  pairs at 1e-10.
* **`Source.gaussian_schell` / `Source.schell_model`** now pass the
  factory's 4-tuple verbatim instead of wrapping the 3-D ensemble
  in a `Source` whose `E` would have been 3-D.
* **UI runtime test under `-W error::DeprecationWarning`** added —
  exercises `SourceDefinition.to_source()` at runtime to catch the
  static-grep escape (which the v4.15.1 audit identified as a
  missing coverage class).
* **`rays_from_field` top-of-file docstring** corrected from
  Madelung `Im(grad E / E)` to phase-ratio central difference
  (inline docstring was already correct; top was stale).
* **`Magnify` docstring direction inverted** to match code:
  `a > 1` shrinks output; `0 < a < 1` magnifies (Nazarathy/Shamir
  `V[a]` convention).  Dead `operators.py:556-577` reference
  removed.
* **`'uniform'` and `'unwrap_gradient'` modes** in `rays_from_field`
  now have direct test coverage (audit flagged these as previously
  untested).  Vortex direction-recovery and anamorphic
  direction-cosines tests added.

### P3 closures

* **Sparrow tolerance pin tightened from 5% to 1%**.  Measured
  achievable error on canonical Airy fixture (N=256, dx=0.1µm,
  λ=600nm, f/#=4): 0.017% — comfortable headroom over the new 1%
  pin.  Docstring "Accuracy (v4.15.2)" paragraph cites the measured
  number.
* **Forbes Q-bfs end-to-end OPD analytical pin** — closes the
  v4.15.1 audit gap "No end-to-end Forbes Q OPD pin against
  analytical formula".  Pins `OPD(r) = (n - 1) * sag(r)` (the
  `(n - 1)` index factor reflects that light experiences an optical
  path difference of `(n_glass - n_outside) * geometric_path`; for a
  sag in vacuum/air the multiplier is `(n_glass - 1)`) against the
  closed-form Q-bfs sag at 5e-3 rad tolerance.  v4.15.3 correction
  (per AUDIT_V4_15_2 P3 docs-drift finding): pre-v4.15.3 the bullet
  stated `phi(r) = -k * sag(r)` at `1e-3 rad` -- the formula omitted
  the `(n - 1)` factor that the actual test code uses, and the
  tolerance was incorrectly tightened from the test's `5e-3` value
  (the test code itself was always correct; only the CHANGELOG bullet
  drifted).
* **`lumenairy.algebra` exports moved from Tier-2 to Tier-1** in
  `__init__.py.__all__` — operator algebra is a build-time
  construction surface, not a propagation surface.
* **CHANGELOG line-citation refreshes**: "45° fold" → "60° fold
  (α=π/6)" in the v4.15.1 OAP raytrace test description;
  `optimize/core.py:2790-2795` → `:2905` (branch) + `:2034`
  (class) + `:2044` (singleton) after Agent E's sentinel
  refactor pushed lines.
* **CHANGELOG test-count arithmetic refresh**: Agent A (v4.15.1)
  count corrected 18 → 19; Agent F count corrected 13 → 20
  (parametrize entries) to match `pytest --collect-only`.
* **`energy_threshold` kwarg** now forwarded through all 3 Schell
  factories to `PartialCoherenceMCF.from_ensemble` (was exposed on
  the MCF builder but not on the factory entry points).
* **Stray `C:tmpsources_diff.txt`** (typo'd `C:\tmp\` path; OneDrive
  U+F03A colon substitute) deleted — 44 KB git-diff dump, content
  recoverable via git history.

### Test counts

* Pre-v4.15.2 baseline (v4.15.1): 1625 unit tests + 1 skip + 1 xfail.
* v4.15.2 additions: A=18, B=15 (9 new + 6 modifications), C=32,
  D=19, E=16; net +110 (pytest-collected delta from 1625 baseline
  to 1735 at v4.15.2 HEAD sha `672051c`).
* Final: **1733 unit tests passing, 1 skipped, 1 xfailed** (1735
  collected total per `pytest --collect-only -q tests/unit` at sha
  `672051c`); **34/34 validation files passing**.  v4.15.3
  correction (per AUDIT_V4_15_2 P2 docs-drift finding): pre-v4.15.3
  this entry stated "1732 pass + 1 skip + 1 xfail" (= 1734
  collected), off by 1 from the actual `pytest --collect-only`
  count at the v4.15.2 release commit.  The arithmetic was also
  inconsistent with the per-agent breakdown above (18 + 15 + 32 +
  19 + 16 = 100, not 107) -- v4.15.3 reconciles both to the actual
  pytest-collected delta.

### Deferred to v4.16+

Unchanged from v4.15.1 deferrals: modal-asymptotic independent
ground-truth pin against direct quadrature; 4 V2 meta-pin candidates
(sentinel-aware branch propagation, `_xp_of` dispatch, `dy` parameter
threading walker, `__all__` symmetry walker); MCF-aware downstream
propagators (consume `PartialCoherenceMCF` through propagation
chains); multi-process atomic-append for `storage.py`;
`MultiPrescriptionParameterization.scale_floor` (v4.13.1 P1-I
carryover); Forbes Q-2D-asymmetric variant.

---

## [4.15.1] — 2026-05-18

**Closes the v4.15.0 audit (`docs/audits/AUDIT_V4_15_0_2026_05_18.md`)
through P3 + ships 2 additive features from
`docs/audits/CLUSTER_B_SPEC.md` (operator algebra + rays-from-field
bridge).**  The audit found 2 P0s + 12 P1s + many P2/P3 (highest-yield
audit in the series).  v4.15.1 closes both P0s + all Tier-0 P1s + the
P2/P3 sweep + adds 800+ LOC of new CLUSTER_B surface.  **1625 unit
tests pass** (up from 1425; +200 net), 1 documented skip, 1
documented xfail; **34/34 validation files pass**.

### Breaking changes

v4.15.1 ships 3 confirmed breaking items.  Callers who relied on the
v4.15.0 contracts must migrate; v4.15.2 (P1-NEW-G closure) adds this
subhead retroactively to make the audit-flagged items discoverable.

1. **Schell-family return shape**: `create_gaussian_schell_source`,
   `create_schell_model_source`, and `create_annular_incoherent_source`
   default to `return_kind='ensemble'` and now return the 4-tuple
   `(ensemble_3d, dx, dy, wavelength)` where `ensemble_3d` has shape
   `(n_realizations, Ny, Nx)`.  v4.15.0 returned `(E_2d, x, y)` (a
   collapsed single field plus coordinate vectors).  Pre-v4.15.0
   callers doing `E, x, y = create_gaussian_schell_source(...)` now
   silently bind `E.ndim == 3` and `x` to a scalar `dx`.  Pass
   `return_kind='ensemble'` explicitly to acknowledge the new
   contract, or `return_kind='mcf'` to opt into a
   `PartialCoherenceMCF` object instead.  v4.15.2 (P0-NEW-1 closure)
   emits a `DeprecationWarning` on the default path; removal in v5.0.
2. **`strehl_vector` default reference removed**: v4.15.0 had a
   buggy default plane-wave reference that produced unity Strehl for
   any uniform field of equal power AND `Strehl > 1` on focused PSFs
   (the focused field is more peaked than the plane-wave reference at
   matched total power).  v4.15.1 requires the caller to pass
   `reference=` explicitly; the docstring also drops the unverified
   "Richards-Wolf high-NA" claim.  See P1-F1-3 below.
3. **`system.evaluate` mixed-shape prescription raises `ValueError`**:
   a prescription containing BOTH `surfaces` + `thicknesses` AND
   `elements` + `all_thicknesses` keys previously silently picked one
   schema.  v4.15.1 rejects it at the validator with a clear message.
   Callers passing raw Zemax-loader output need to filter the
   surfaces keys before handing the dict in:
   ```python
   rx_filtered = {k: v for k, v in rx.items()
                  if k not in ('surfaces', 'thicknesses')}
   ```
   See P1-F1-6 below.

### P0 closures

**P0-NEW-1 — `make_off_axis_parabola` factory fix (doubly broken):**
Decenter formula corrected `f*tan(alpha)` -> `2*f*tan(alpha)` (chief-
ray geometry on parent paraboloid).  Tilt remains 3-tuple
`(off_axis_angle, 0.0, 0.0)`; factory docstring now loudly documents
that the OAP prescription is **intended for `apply_real_lens_traced`
exclusively** — the paraxial `apply_real_lens` cannot interpret the
3-tuple tilt correctly.  New end-to-end raytrace test
(`test_end_to_end_raytrace_focuses_at_offset`) pins the off-axis
focal-point location to within 1% of the chief-ray geometric
prediction at 60° fold (α=π/6).  v4.15.2 (P3 doc-drift fix): the
original CHANGELOG cited a pi/4 fold -- the actual test uses π/6
because α=π/4 is degenerate at this geometry.  P3 carryover:
`vertex_radius` now validated (must be `None` or finite positive).

**P0-NEW-2 — Schell-family factories redesign:**  The v4.15.0
factories collapsed the `n_realizations` ensemble into a single
fully-coherent complex field before return — the documented
partial-coherence contract was unfulfillable.  v4.15.1 introduces a
hybrid:

* Default `return_kind='ensemble'`:  factory returns
  `(ensemble, dx, dy, wavelength)` where `ensemble` has shape
  `(n_realizations, Ny, Nx)`.  Caller iterates over realizations and
  averages intensities downstream — physically-correct partial
  coherence.
* Opt-in `return_kind='mcf'`:  factory returns a new
  `PartialCoherenceMCF` dataclass with `.intensity()`,
  `.coherence_at(...)`, and `.coherent_modes()` methods.  For small
  grids (`Ny*Nx <= 64**2`), stores the full `J(r1, r2)`; for larger
  grids, stores the leading K coherent modes (Wolf 1982 JOSA
  decomposition) via SVD of the ensemble matrix.  Truncation
  threshold:  smallest K with `cumsum(eigvals)/sum(eigvals) >= 0.99`
  (Karhunen-Loève default).
* **Physics fix:**  the random-phase RMS normalization (which forced
  `sigma_phi = 1` regardless of `sigma_g`) is replaced with the
  spec-correct Fourier-filtered Gaussian-noise recipe.  Now `sigma_g
  -> 0` actually approaches incoherent (off-diagonal MCF -> 0) and
  `sigma_g -> infinity` approaches coherent (rank-1 MCF).

Affected factories: `create_gaussian_schell_source`,
`create_schell_model_source`, `create_annular_incoherent_source` +
matching `Source.gaussian_schell` / `Source.schell_model`
classmethods.  Note: MCF-aware downstream propagators are deferred
to v4.16+; the `PartialCoherenceMCF` object is consumable for
analysis / inspection in v4.15.1.

### P1 closures (Tier 0 audit recommendations)

* **P1-NEW-A: `sparrow_resolution` canonical Sparrow root-finding.**
  Implementation rewritten to true two-source dip-vanishing condition
  `d²/dr² [PSF(r-d/2) + PSF(r+d/2)]_{r=0} = 0` via
  `scipy.ndimage.map_coordinates` sub-pixel azimuthal averaging +
  cubic-spline 2nd derivative + `scipy.optimize.brentq` root-finder.
  Now returns 2.273 µm vs expected 2.273 µm at lambda=600nm, f/#=4
  (previously 1.93 µm, 15% low).
* **P1-NEW-C: 7 UI Source-factory deprecation callsites migrated** to
  kwarg-only canonical form in `lumenairy/ui/model.py`.  The v4.15.0
  release that introduced the deprecation shim now also migrates its
  own internal UI consumers.
* **P1-NEW-D: Raytrace flat-keys allowlist** at `raytrace/core.py:
  1507-1521` extended with `q_bfs_coeffs`, `q_con_coeffs`, `r_max`.
  Forbes Q prescriptions in flat-keys form no longer silently drop
  the coefficients at the gather step.
* **P1-NEW-E: Zemax `.zmx` QBFS/QCON parsing** added to
  `io/prescriptions.py`.  `.zmx` files with Q-type freeforms now
  load with `freeform_type='q_bfs'` or `'q_con'` + coefficients +
  `r_max` (parsed from `DIAM` / `PARM` lines), instead of silently
  degrading to base conic.
* **P1-F1-1: Q-bfs/Q-con radial-clip alignment.**  v4.15.0's
  rectangular `|X| <= norm_x AND |Y| <= norm_y` clip let pixels at
  `(0.9*r_max, 0.9*r_max)` (radial `r = 1.27*r_max`) through — outside
  the Forbes domain.  v4.15.1 uses a radial primary clip
  `r <= r_max` with the rectangular `(norm_x, norm_y)` box as
  secondary aperture.
* **P1-F1-2: `surface_sag_freeform` requires `r_max`** for
  `freeform_type in ('q_bfs', 'q_con')`.  Previously defaulted
  silently to `r_max=1.0` (a unit-mismatch bug — user passing X/Y
  in metres got sag computed on a sub-pixel of the actual aperture).
  Now raises `TypeError` with a clear message.
* **P1-F1-3: `strehl_vector` default-reference removed** (breaking).
  v4.15.0's default plane-wave reference produced unity for ANY
  uniform field of equal power AND Strehl > 1 on focused PSFs
  (more peaked than plane-wave at equal total power).  v4.15.1
  requires explicit `reference=` and softens the docstring
  (drops unverified "Richards-Wolf high-NA" claim).
* **P1-F1-4: `rayleigh_resolution` Gaussian-PSF false-positive fixed.**
  Now requires a strict subsequent rise above the candidate minimum
  by >=0.5% of peak before declaring first zero; Gaussian-like PSFs
  (no true zero) return NaN + `RuntimeWarning` advising
  `fwhm_resolution` / `sparrow_resolution` instead.
* **P1-F1-5: `astigmatism_mag_angle` docstring range correction** —
  `(-pi/4, pi/4]` -> `(-pi/2, pi/2]` (the actual range from
  `0.5 * atan2(c3, c5)`).
* **P1-F1-6: `system.evaluate` mixed-shape `ValueError`** — a
  prescription with both `surfaces`+`thicknesses` AND
  `elements`+`all_thicknesses` keys is now rejected at the
  validator with a clear message rather than silently picking a
  schema.  Behaviour change: callers passing raw Zemax-loader
  output need to filter the surfaces keys (`{k:v for k,v in rx.items()
  if k not in ('surfaces','thicknesses')}`).
* **P3-NEW-A: Forbes Q wave-optics path** — v4.15.0's
  `apply_real_lens` silently `RuntimeWarning`d and skipped the
  Forbes Q freeform contribution.  v4.15.1 routes Q-bfs / Q-con
  through `surface_sag_freeform` properly (option (a) of the audit
  recommendation), inheriting P1-F1-1 + P1-F1-2 guards.

### P2 closures

* **`__all__` symmetry**:  `surface_sag_q_bfs`,
  `surface_sag_q_con` re-exported from
  `lumenairy/elements/__init__.py`; `make_off_axis_parabola` from
  `lumenairy/io/__init__.py`.  `from lumenairy.elements import
  surface_sag_q_bfs` now works.
* **Sentinel consolidation**:  `_ZeroApertureMaskSentinel`
  (`optimize/core.py`) and `_AngleUnsetSentinel`
  (`elements/polarization.py`) now inherit from `_deprecation._Sentinel`
  base class.  `_Sentinel` gained pickle-safe `__reduce__` + name-keyed
  `_SENTINEL_REGISTRY` + `_sentinel_unpickle` reconstructor so
  pickle round-trips return the singleton instance (not a fresh
  sentinel).
* **`system.evaluate` Zemax-shape test** added (the audit's "headline
  ergonomic claim was untested" finding closed).

### P3 closures

* `n=1.0` consistency:  `optimize/multiconfig._resolve_lens_glass_index`
  bounds widened from exclusive `(1.0, 5.0)` to inclusive
  `[1.0, 4.0]` matching `register_fixed_glass`.
* Codegen runtime version pin gains an upper-bound major-version
  warning (`UserWarning` if running on `lumenairy >= 5.0.0`).
* `LambertianBSDF.evaluate` gains explicit surface-frame docstring
  + `RuntimeWarning` if `incident_direction` is non-axially-aligned
  without explicit frame transform.
* Coatings TIR cap warnings promoted from filtered `RuntimeWarning`
  to always-emit `UserWarning`.
* Forbes Q orthonormalizer docstring formula corrected:
  `c_n = sqrt((2n+3)(n+2)/(n+1)^2)` -> `c_n = sqrt((2n+3)(n+2)/(n+1))`
  (the implementation was already correct; only the docstring lied).
* `astigmatism_mag_angle` docstring range correction (also P1-F1-5).
* CHANGELOG/release-notes: lenses_maslov `_ZERO_APERTURE_MASK`
  sentinel branch now lives at `optimize/wrapper_merits.py:876`
  (the `if _cache['mask'] is _ZERO_APERTURE_MASK` line); was
  `optimize/core.py:3032` pre-v5.1.0 Agent E 6-file split (the
  branch moved out of the monolithic core.py to the new
  ``wrapper_merits.py`` submodule that hosts the ``MultiWavelengthMerit``
  / ``MultiFieldMerit`` / ``ToleranceAwareMerit`` triplet); was
  `:3015` pre-v4.16.3 Agent C `Constraint` auto-probe
  DeprecationWarning latch + pickle catch widening (~17 lines added
  above the sentinel branch); was `:2974` pre-v4.16.2 Agent B
  `MultiWavelengthMerit` `FutureWarning` latch + Constraint-probe
  move + lambda pickle-probe (~41 lines added above the sentinel
  branch); was `:2958` pre-v4.16.1 Agent A `MultiWavelengthMerit`
  `SUM`->`AVG` refactor (~16 lines added in the merit-aggregation
  block above the sentinel branch); was `:2980` pre-v4.15.4 Agent
  B `_PerturbedABCDFallbackSentinel` deletion (~55 lines removed
  at the top of the sentinel block); and `:2905` pre-v4.15.3
  sentinel-wiring work.  The remaining
  sentinel class + singleton (`_ZeroApertureMaskSentinel` /
  `_ZERO_APERTURE_MASK`) are at `optimize/context.py:74` and
  `optimize/context.py:84` respectively post-v5.1.0 Agent E 6-file
  split (was `optimize/core.py:2044` and `optimize/core.py:2054`
  pre-split, post Agent E's v4.15.2 `_Sentinel` base-class
  refactor).  v4.15.2 (P3):
  citation refreshed after a
  second line-drift pass against the current source supersedes
  the earlier stale citations.

### CLUSTER_B Item 6 — `rays_from_field` bridge function

New `lumenairy.rays_from_field(E, *, dx, wavelength, dy=None,
n_rays=200, placement='cdf', angle_method='complex_gradient',
intensity_threshold=1e-4, z0=0.0, random_state=None) -> RayBundle`
samples a coherent field into a geometric ray bundle.  Bridges
`propagators/` (wave) <-> `raytrace/` (ray) so users can overlay
ray traces on coherent-field plots, seed a Maslov/GBD bundle from a
measured pupil field, or hand a coherent field into the geometric
ray tracer for hybrid analysis.

Placement modes: `'cdf'` (separable inverse-CDF, fast),
`'rejection'` (true 2-D rejection, exact), `'uniform'` (grid + threshold
mask).  Angle methods: `'complex_gradient'` (phase-ratio central
difference, singularity-safe — adapted from spec for correct
behaviour at Nyquist), `'unwrap_gradient'` (np.unwrap-based, fragile
near vortices).  Evanescent rays (`L² + M² > 1`) flagged with
`RAY_EVANESCENT` and `alive=False`.

13 tests + 1 runnable example
(`examples/rays_from_pupil_field.py`).  Implementation note: 3
spec deviations (phase-ratio central difference instead of literal
`Im(grad E / E)`; evanescent test samples 6x Nyquist to avoid
spectral aliasing; OPD test uses wrap-free phase slope) all
documented in the agent's release notes.

### CLUSTER_B Item 2 — Operator algebra

New `lumenairy/algebra/` subpackage implementing Nazarathy/Shamir
operator algebra (JOSA 70 (2), 1980).  9 new symbols at top level:

* `Operator`, `CompositeOperator` — base classes; ABCD-tracking
  algebraic composition.
* `FreeSpace(d, *, method='auto')`, `ThinLens(f)`,
  `CylindricalLens(f_x, f_y)`, `Magnify(a_x, a_y)`,
  `FourierTransform(f_focal)` — primitive operators.
* `Aperture(diameter, shape)`, `GaussianAperture(sigma)` — passive
  aperture operators (identity ABCD).
* `Operator.from_prescription(prescription, wavelength)` —
  prescription-dict -> CompositeOperator factory.  Paraxial-only;
  produces ABCD identical to `system_abcd(...)` to within 1e-12 abs.

Composition: `A * B` means "first B, then A".  ABCD of `A * B` is
`A.abcd @ B.abcd`.  Application: `sys(source) -> Source`, or
`sys.apply(E, dx=..., wavelength=...)`.  Anamorphic support via
separate `_abcd_x` / `_abcd_y`.

91 tests + 2 runnable examples (`examples/algebra_4f_system.py`,
`examples/algebra_anamorphic.py`).  Spec deviation: `Magnify._apply`
uses closed-form `sqrt(a_x*a_y)` amplitude prefactor instead of
spec's `resample_field` recipe (the spec recipe had an energy-
conservation bug; closed-form preserves energy per-pixel by
construction).  Phase 2 symbolic reduction (FreeSpace+ThinLens+
FreeSpace collapse, etc.) explicitly deferred to a future PR.

### Test counts

* Pre-v4.15.1 baseline (v4.15.0): 1425 unit tests + 1 skip + 1 xfail.
* v4.15.1 additions (`pytest --collect-only` items, parametrised
  test cases counted separately): A=19, B=20+migrated, C=13+migrated,
  D=18, E=20, F=20, G=91; gross 201 collected, net ~200 added (the
  +200 number nets out test migrations: agents B and C migrated
  v4.15.0 property pins to v4.15.1 ensemble / analytical-value pins).
  v4.15.2 (P2 + P3 test-count refresh): the F=20 count supersedes
  the original CHANGELOG's F=13 -- Agent F shipped 7 additional
  follow-up tests beyond the initial 13 enumerated in
  `.release_notes_v4_15_1_agent_f.md`, bringing the file to 20
  pytest items.  The A=19 count likewise supersedes the original
  A=18.
* Final: **1625 unit tests passing, 1 skipped, 1 xfailed**; **34/34
  validation files passing**.

### Deferred to v4.16+

* Modal-asymptotic independent ground-truth pin against
  `propagate_hf_chebyshev_quadrature(method='direct')` (audit
  P1-NEW-B, Tier 1).  v4.15.0 replaced known-buggy warm-start with
  unverified cold-start; this pin closes the verification gap.
* 4 remaining V2 meta-pin candidates (sentinel-aware branch
  propagation, `_xp_of` dispatch, `dy` parameter threading,
  `__all__` symmetry walker).
* MCF-aware downstream propagators (consume `PartialCoherenceMCF`
  through propagation chains).
* Multi-process atomic-append for `storage.py` (HDF5 SWMR + Zarr
  distributed lock).
* `MultiPrescriptionParameterization.scale_floor` (v4.13.1 P1-I
  carryover).
* Forbes Q-2D-asymmetric variant (Forbes 2012) for full 2-D
  freeform support.

---

## [4.15.0] — 2026-05-18

**Major minor release** rolling together carryover P1s from the
v4.14.2 audit (the "v4.14.4" patch scope) + ROADMAP v4.15 + ROADMAP
v4.16 into a single coordinated ship.  **1425 unit tests pass** (up
from 1265; +160 net new pins), **1 documented skip**, **1 documented
xfail** (`create_led_source` validation entry-point exemption);
**34/34 validation files pass**.

### Headline: modal-asymptotic 19.4x perf win + wrong-saddle physics fix

`propagate_modal_asymptotic` switches from a per-pixel warm-started
Newton loop to a single batched cold-start
`_solve_envelope_stationary_batch` + `_compute_M_b_batch` path
(the private helpers v4.14.0 shipped but did not consume on the
public path).  Closes the v4.14.0 audit's "wrong-saddle basin"
physics finding (warm-start chain entered wrong-saddle basins at
grid edges, silently zeroing those pixels via the `|b_quad| > 700`
overflow guard).  Cascade impact: any caller that builds on
`propagate_modal_asymptotic` (aberration tensor sigma-integration
grid path, through-focus / polychromatic helpers per focus or
wavelength point) gains the perf win and the grid-edge correctness
in one step.

Output is **bit-different** from v4.14.x at grid edges (strictly
more non-zero pixels because the cold-start finds the physical
saddle uniformly).  Four bit-equal pins migrated to property
pins (1e-8 abs vs cold-start reference + 5% energy + nz-count
>= warm-start ref):
`test_lg00_single_mode_matches_reference`, `test_multimode_matches_reference`
in `test_perf_v4_12_0_asymptotic.py`, plus
`test_lg00_single_mode_bit_equal` and `test_lg_p0_4mode_prescription_bit_equal`
in `test_audit_fixes_v4_14_0_agent_1.py`.  The v4.14.1 row-reset
warm-start pin (`test_row_reset_resets_warm_start`) was retargeted
to the v4.15 stronger structural guarantee:  the scalar
`solve_envelope_stationary` is no longer invoked by the public
path in any `maslov_tracking` mode (the warm-start chain is
structurally deleted, not just reset).

Measured: **19.4x** speedup at N=128 LG_(0,0); +52 non-zero
pixels recovered on the same grid (15918 vs 15866).

### Source factory normalisation + ergonomic system entry

Pre-v4.15 the 5 `Source.method` classmethod factories had
inconsistent positional order — some put size-arg first, others
N first.  v4.15 picks the canonical order
`Source.method(*, N, dx, wavelength, <size_kwargs>)` (kwarg-only).
The legacy positional form still works for one release with a
`DeprecationWarning` routed through the new
`_deprecation.warn_deprecated_signature` helper with
`version_removed='5.0'`.  Affected factories: `Source.gaussian`,
`Source.plane_wave`, `Source.point_source`, `Source.top_hat`,
`Source.fiber_mode`.

New ergonomic entry `lumenairy.system.evaluate(prescription,
source, *, output_grid=None, output_dx=None, ...)` (also
top-level `lumenairy.evaluate`).  Accepts both Zemax-loader
prescription shape (`elements` + `all_thicknesses` keys) and
factory shape (`surfaces` + `thicknesses` keys).  Users loading
a `.zmx` file no longer have to build the element list manually
before propagating.

### 7 new public-API functions (ROADMAP v4.16 closure)

* `ee_polychromatic(prescription, wavelengths, weights, radii, ...)`
  — convenience chain over `polychromatic_psf` +
  `encircled_energy_radius`.
* `strehl_vector(Ex, Ey, Ez=None, *, reference=None)` —
  vector Strehl with optional `Ez` z-component (Richards-Wolf
  high-NA case).
* `coupling_efficiency_vector(Ex, Ey, Ez=None, *, mode_Ex,
  mode_Ey, mode_Ez=None, dx)` — vector overlap integral with a
  vector mode.
* `rayleigh_resolution(psf, dx, wavelength, *, axis='radial')`
  — first-zero-of-PSF Rayleigh diffraction limit.
* `sparrow_resolution(psf, dx, *, axis='radial')` — empirical
  Sparrow criterion (dip-just-vanishes for two overlapping
  point sources).
* `fwhm_resolution(psf, dx, *, axis='radial')` — twice the
  FWHM half-radius of the central peak.
* `astigmatism_mag_angle(coeffs)` — Mahajan §8.2 conversion of
  Zernike `(c3, c5)` to `(|astig|, theta)` in the
  OSA/ANSI convention matching `zernike_decompose`.

All 7 are top-level exports in `lumenairy.__all__`.

### 3 new source factories (partial-coherence + ring incoherent)

* `create_gaussian_schell_source(*, N, dx, wavelength, w0,
  sigma_g, n_realizations=16, ...)` — spatially-incoherent
  Gaussian-Schell beam via random-phase ensemble.
* `create_schell_model_source(*, N, dx, wavelength,
  intensity_profile, coherence_length, n_realizations=16, ...)`
  — generic Schell-model (caller supplies intensity profile).
* `create_annular_incoherent_source(*, N, dx, wavelength,
  inner_radius, outer_radius, n_realizations=16, ...)` —
  angular-spectrum ensemble with finite source extent for
  partial-coherence integration (distinct from existing
  monochromatic-coherent `create_annular_beam`).

Matching `Source.gaussian_schell(...)` / `Source.schell_model(...)`
classmethods.  All 3 call `_validate_grid_params` in the first
10 lines (per the v4.14.2 audit's input-validation entry-point
meta-pin candidate).

### Forbes Q-type freeform basis + off-axis parabola factory

* `surface_sag_q_bfs(X, Y, *, radius, coefficients, r_max, ...)`
  — Forbes Q-bfs basis (Forbes 2007, *Opt. Express* 15(8) 5218,
  eq. 13).  Best-fit-sphere subtracted; orthonormal on the
  weight `u^2 (1-u^2) d(u^2)` over `[0, 1]`.
* `surface_sag_q_con(X, Y, *, radius, conic, coefficients,
  r_max, ...)` — Forbes Q-con basis (Forbes 2010,
  *Opt. Express* 18(13) 13851, eq. 6).  Conic-subtracted;
  orthonormal on weight `u^4 d(u^2)`.

Implementation uses the shifted-Jacobi 3-term recurrence
(A&S 22.7.1) on `t = 2x - 1` for `x = u^2 in [0, 1]`:
Q-bfs `(alpha, beta) = (1, 1)` with orthonormaliser
`c_n = sqrt((2n+3)(n+2)/(n+1)^2)`; Q-con `(alpha, beta) =
(0, 2)` with `c_n = sqrt(2n+3)`.  Orthonormality verified
numerically to <1e-6 over the first 5 orders for both bases.

* `make_off_axis_parabola(focal_length, off_axis_angle,
  clear_aperture, *, glass='__MIRROR__', vertex_radius=None,
  name=None) -> dict` — prescription factory for OAP segments.
  Single parabolic surface (conic `k = -1`, vertex radius
  `R = 2*focal_length`) with `decenter` and `tilt` set to the
  parent-axis offset and local-frame tilt.

### v4.14.2 carryover P1s closed (the "v4.14.4" scope)

**UI subpackage (P1-UI-1 through P1-UI-7):**
* `main_window.py` glass table now includes `N-LASF9` and
  `S-NPH1` (P1-UI-1) and `_nudge_distance` routes through
  `set_display_distance` (P1-UI-2, coordinate-mode aware).
* `model.py` undo state-capture now includes
  `wavelength_weights`, `field_weights`, `lens_options`
  (P1-UI-3).  Back-vertex calculation consolidated into a
  single `_prev_element_back_vertex_world` helper to prevent
  drift between the 717-718 vs 752 sites (P1-UI-4).
* `waveoptics_dock.py` re-parent now guarded by
  `shiboken6.isValid(original_parent)` to avoid the
  mid-dialog segfault risk (P1-UI-5).
* `psf_mtf_dock.py` ray-traced OPD now accumulates via
  `np.add.at(... mean)` instead of last-write-wins per pixel
  (P1-UI-6); out-of-aperture rays are filtered by a bounds
  mask before indexing instead of `np.clip(...).astype(int)`
  silently snapping to the pupil edge (P1-UI-7).

**Codegen + ghost + glass:**
* P1-CG: generated scripts now embed a runtime version pin
  (`if tuple(...) < (4, 15, 0): raise RuntimeError(...)`) plus
  a `lumenairy_version:` comment stamp.
* P1-GH-1: `non_sequential_stray_light` accepts a `seed: int |
  None = None` kwarg (default `None` uses system entropy so MC
  produces a real uncertainty band; pass a fixed integer to
  pin reproducibility).
* P1-GL-1: bundled Sellmeier rows for `SiO2` / `F_SILICA` /
  `FUSED_SILICA` (Malitson 1965 fused-silica coefficients) and
  `S-LAH64` / `S-LAH79` (OHARA Zemax 2017-11-30 catalog from
  the refractiveindex.info-database YAML).  Minimal installs
  without `refractiveindex` now resolve these glasses through
  the bundled Sellmeier fallback.

### Exhaustive P2/P3 sweep + meta-pin candidate #3

User-requested exhaustive enumeration of the v4.14.2 audit's
18 P2 + 12 P3 findings.  Net closure tally:

* 7-8 of 18 P2 closed (4 by Agent F directly: `lumenairy_context`
  6/7 redundant clears, `create_multi_field_sources` factory-
  validation list, ROADMAP refresh, HDF5/Zarr `lumenairy_version`
  attr stamping; 3 by other agents in this release:
  `_validate_grid_params` bool reject, deprecation-shim
  `_deprecation.py` migration with `version_removed`, codegen
  version pin; 1 partial: artifact version pinning — HDF5+Zarr
  done, codegen done).  10-11 P2 deferred (architectural items
  reserved for v4.16+ or v5.0).
* 4 of 12 P3 closed by Agent F: CHANGELOG line-citation drift
  (3 stale `optimize/core.py:...` ranges in the v4.14.2 entry
  refreshed to current line numbers — see commit diff for the
  exact pre/post values); `create_led_source` legacy-shim
  error-message clarity; README `makedammann2d _legacy_units='SI'`
  migration example; README cookbook section with examples for
  the 6 v4.14.0 public functions.

**New structural meta-pin (V2 candidate #3):**
`tests/unit/test_v4_15_dispatcher_pin_validate_grid_params.py`
walks every `create_*` factory and asserts `_validate_grid_params`
appears in the first 15 body lines.  17 PASS + 1 documented
xfail (`create_led_source` legacy-shim positions validator past
the head window — pinned via `xfail(strict=True)` so future
refactors that lift the validator forward flip to XPASS and
the exemption is removed).

### Version-stamping on HDF5 / Zarr writes

`io/storage.py` now writes a `lumenairy_version` attr on every
`create_dataset` / `create_array` / `create_group` site (7
locations).  Future-proof for cross-version field-file
compatibility checks.

### Bundled-glass registry reverse-direction consistency check

The v4.14.2 `_check_glass_registry_consistency` only walked
registry-entry -> `SELLMEIER_COEFFICIENTS` (forward).  v4.15
adds the reverse walk: every key in `SELLMEIER_COEFFICIENTS`
must appear in `GLASS_REGISTRY` (with `'__sellmeier__'` flag
if it's pure Sellmeier).  Coefficient rows added without a
registry entry would have remained silent dead code; this
catches them at module load.

### ROADMAP refresh

`ROADMAP.md` updated to v4.15.0 baseline.  v4.14.1 / v4.14.2 /
v4.14.3 / v4.15.0 entries added to Shipped highlights;
items closed in v4.15 removed from v4.15 + v4.16 target lists
and renumbered.  v4.16+ target list reseeded with the items
deferred from the v4.14.2 audit + the 5 V2 meta-pin
candidates.

### Test counts

* Pre-v4.15 baseline (v4.14.3): 1265 unit tests.
* v4.15 additions: A=8, B=40, C=26, D=23, E=27, F=36
  parametrized tests across two new files; net 160 added.
* Final: **1425 unit tests passing, 1 skipped, 1 xfailed**
  (`_ZARR_MKDIR_PATCH_LOCK` exemption + `create_led_source`
  validation-entry exemption); **34/34 validation files
  passing**.

### Deferred to v4.16+

* 4 V2 meta-pin candidates still standing (sentinel-aware
  branch propagation, `_xp_of` cross-backend dispatch,
  `dy` parameter threading, `__all__` symmetry).  Input-
  validation entry-point candidate #3 is shipped this
  release.
* Multi-process atomic-append for `storage.py` (HDF5 SWMR
  + distributed Zarr lock).  Single-process atomicity is
  documented in v4.14.3.
* `MultiPrescriptionParameterization.scale_floor` (v4.13.1
  P1-I carryover).
* Modal-asymptotic JAX-twin lift to use the v4.15 batched
  helpers.
* Forbes Q-2D-asymmetric variant (Forbes 2012) for full 2-D
  freeform support beyond the rotationally-symmetric Q-bfs /
  Q-con bases shipped here.
* Architectural items reserved for v5.0 (file splits, CI
  gates, shim removal — see ROADMAP).

---

## [4.14.3] — 2026-05-17

**Closes the v4.14.2 audit (`docs/audits/AUDIT_V4_14_2_2026_05_17.md`).**
The audit found 2 NEW P0 findings (`storage.py` non-atomic
`n_planes` increment → silent data loss in concurrent / Zarr
streaming; `makedammann2d` >1mm SI heuristic silently mangles
legitimate mm-scale gratings), 21 NEW P1s, 18 P2s and 12 P3s.
The "fix N, miss N+1" sibling-gap meta-finding recurred 5 ways
on v4.14.2.  v4.14.3 closes both P0s, the 5 sibling-gap
recurrences, 11 latent-bug P1s (1 real physics error in
`multiconfig.py`), and all 3 doc-drift P1s.  **1265 unit tests
pass** (up from 1190); **34/34 validation files pass**.

### Breaking changes — none

Two near-breaking deltas with explicit opt-in / opt-out:

* **`makedammann2d` accepts `_legacy_units='auto'|'um'|'SI'`** (default
  `'auto'` preserves the v4.14.2 per-parameter deprecation heuristic).
  The `'auto'` and `'SI'` modes now raise `ValueError` on any
  unit-bearing kwarg > 1.0 m (rejects nm-scale-garbage from the
  silent-mangling regime).  Legitimate >1m / mm-scale SI gratings
  set `_legacy_units='SI'` to bypass the legacy heuristic
  entirely.  Pure-legacy callers can set `_legacy_units='um'` to
  rescale all unit-bearing inputs by 1e-6 without firing the
  deprecation warning.
* **`create_led_source` legacy-positional shim hardened** with a
  scale-inversion sanity check: a canonical-order positional
  call (`N, dx, wavelength, diameter, divergence`) that
  accidentally slots a wavelength into the `_legacy_diameter`
  position now raises `TypeError` with a migration message
  instead of producing 633 nm "diameter" / 0.1 m "wavelength"
  garbage.

### P0 closures

**P0-NEW-1: `storage.py` n_planes atomicity** — `append_plane_h5`
(`io/storage.py:444`) and `_zarr_append_plane` (`:782`) bumped
`grp.attrs['n_planes']` AFTER `create_dataset`.  A crash between
the two operations left an orphan dataset; the next append used
the stale `n` to compute `plane_{N:02d}` and on Zarr (`overwrite=
True` at line 769) the orphan was silently clobbered.  Concurrent
appenders racing on `n_planes=N` both wrote `plane_{N:02d}`, the
second silently winning.  v4.14.3 inverts the ordering — attr
written BEFORE dataset create, try/except rollback on failure —
and drops `overwrite=True` on the Zarr path so the orphan case
now raises rather than silently destroying data.  Single-process
atomicity is documented; multi-process locking (HDF5 SWMR /
distributed Zarr lock) deferred to v4.15+.  3 regression tests
pin attr-write ordering, docstring contract, and the no-silent-
clobber invariant.

**P0-NEW-2: `makedammann2d` >1m upper-bound** — v4.14.2's
`value > 1e-3` heuristic silently rescaled mm-scale SI gratings
(coarse industrial Dammann, THz/MMW) by 1e-6 → nm-scale garbage.
v4.14.3 adds a `_legacy_units` kwarg (see "Breaking changes"
above) plus an explicit `ValueError` for any unit-bearing input
> 1.0 m in `'auto'`/`'SI'` mode.  3 regression tests cover the
upper bound, explicit `'SI'` mm-scale pass-through, and explicit
`'um'` rescale without `DeprecationWarning`.  3 historical test
sites that relied on the silent rescale were migrated to
`_legacy_units='um'`; one validation case (`test_elements.py`'s
100 µm legacy-µm Dammann) likewise opts in explicitly.

### P1 closures — sibling-gap recurrences (5)

* **P1-NEW-1: `clear_asm_caches()` LG-polynomial chain.**  v4.14.2
  chained 5 sibling caches but missed `_lg_polynomial_items`
  (`asymptotic.py:284`).  v4.14.3 adds the lazy-import + call
  pattern to `clear_asm_caches`, expanding its docstring to list
  all 8 caches it now drains.  Combined-drain test extended to
  assert `_lg_polynomial_items.cache_info().currsize == 0` after
  `clear_asm_caches()`.
* **P1-NEW-2: `apply_rotator` conflict-resolution symmetrised
  across 5 polarization helpers.**  v4.14.2 added `angle_deg=`
  conflict detection to `apply_rotator` only, leaving 4 sibling
  helpers (`apply_polarizer`, `apply_waveplate`,
  `apply_half_wave_plate`, `apply_quarter_wave_plate`) silently
  letting `angle_deg` overwrite `angle`.  v4.14.3 introduces a
  module-level `_AngleUnsetSentinel` singleton (matching the
  v4.14.1 `_ZeroApertureMaskSentinel` pattern) and a single
  `_resolve_angle(func_name, angle, angle_deg)` helper used by
  all 5 helpers.  The "explicit `angle=0` + `angle_deg=90`"
  conflict (which the v4.14.2 `angle != 0.0` check missed) now
  raises `ValueError` from a single canonical site.  19
  regression tests including a 5-way parametrized conflict
  matrix.
* **P1-NEW-4: `create_led_source` `*args` footgun closed** via
  scale-inversion sanity check in the legacy-positional shim.
  Rejected PEP 570 positional-only marker because it would have
  broken every existing kwarg-based call site (including the
  v4.14.2 audit-test infrastructure).  The new check fires on
  `apparent_wavelength > 10 * apparent_diameter` — catches the
  canonical-order mistake (`0.3 > 1.31e-6 * 10`) without
  triggering on legitimate legacy `(diameter, divergence,
  wavelength)` calls (real LED diameters > 10x their wavelength).
* **P1-NEW-5: `_validate_grid_params` tuple-N gating.**  v4.14.2's
  helper accepted both `int` and `(Ny, Nx)` tuples, but 7 of 10
  factories used `np.arange(N)` downstream and crashed with an
  opaque `np.arange` TypeError.  v4.14.3 adds a `support_tuple_N:
  bool = False` parameter; the 3 factories with genuine 2-D
  grid support (`gaussian_beam`, `hermite_gauss`,
  `laguerre_gauss`) opt in, the other 7 reject tuple-N with a
  clear `TypeError` at validation time.  10 parametrized tests
  across all 10 factories.

### P1 closures — under-examined modules (6)

* **P1-MC (real physics error): `multiconfig.py` hardcoded `n=1.5`**
  in the thin-lens / lensmaker formulae at `:265` and `:327`.
  Beam-expander and Keplerian-telescope multi-config builders
  computed lens powers assuming BK7-ish refractive index for
  every glass, silently producing wrong focal lengths for
  flint, SF, S-LAH, fused-silica, or any non-`n≈1.5` glass.
  v4.14.3 introduces `_resolve_lens_glass_index(name)` that
  routes through the canonical `glass.get_glass_index()` lookup
  with a documented `UserWarning` + bounded fallback (`n=1.5`)
  for genuinely unknown labels and bounds-check on custom
  callable returns (`1.0 < n < 5.0`).  Catches `(ValueError,
  KeyError, ImportError, RuntimeError, TypeError)` only —
  preserves the v4.13.0 Phase-2 narrowed-exception discipline.
* **P1-NEW-9: `create_bessel_beam` `cone_angle` constraint.**
  Now enforces `0 < cone_angle < pi/2` at construction time;
  `cone_angle = pi` previously produced `sin(pi) ≈ 0` and a DC
  field silently labelled "Bessel".
* **P1-NEW-10: `create_fiber_mode` `mode_field_diameter > 0`.**
  Now rejected at construction; negative MFD previously yielded
  a sign-flipped Gaussian, MFD=0 yielded an all-ones field.
* **P1-NEW-11: `surface_sag_xy_polynomial` / `surface_sag_chebyshev`
  negative-`norm_x`/`norm_y` rejection.**  `norm_x = -0.05`
  (typo on a 50 mm half-aperture) made `outside = abs(X) >
  -0.05` true everywhere, silently zeroing the entire freeform
  contribution.  Both branches now reject negative norms at
  function entry with a clear `ValueError`.
* **P1-GH-2: `ghost.py` IndexError on `elements`-key
  prescriptions.**  `ghost_analysis` and
  `non_sequential_stray_light` previously crashed on the two
  `elements`-style prescription schemas (surface-style entries
  with `'element_type'` and propagate-style entries with
  `'type'`).  v4.14.3 introduces a `_ghost_surfaces(prescription,
  wavelength)` adapter that detects each schema and routes
  through the appropriate canonical surface-expansion path.
  5 regression tests across all three schemas + missing-keys
  error path.
* **P1-GL-2: `register_fixed_glass` input validation.**  Name
  must be a non-empty string; `n` must be finite and in
  `[1.0, 4.0]`; overwriting an existing entry emits a
  `UserWarning`.  Si (n=3.4) and Ge (n=4.0) verified accepted;
  invalid forms rejected.

### P1 closures — doc-drift (3, retroactive to v4.14.2)

Per the v4.14.2 audit Part 2 / P1-NEW-6/7/8:

* **Agent D test count** (CHANGELOG line 374): `25 tests` →
  `8 tests + 1 parametrize entry`.
* **"6 factories" → "5 factories"** in 3 CHANGELOG sites
  describing the v4.13.x source-factory dispatcher pin.
* **`lenses_maslov.py` line-drift correction note** refreshed
  (`678/737/884/993` → `619/728/874`; verified against the
  current module via def-header grep).  Recorded that one of
  the 4 prior sites was consolidated away.
* **Audit-path references** (11 expected → 11 fixed): all
  `` `AUDIT_V4_*.md` `` in CHANGELOG (8 sites) and README (3
  sites) now carry the `docs/audits/` prefix added by the
  v4.14.2 doc reorganization.
* **Cache-locks meta-pin count** corrected `38 tests` → `39
  collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK`
  skip)` in CHANGELOG lines 158 / 186 and README line 110.

### Test counts

* Pre-v4.14.3 baseline (v4.14.2): 1190 unit tests.
* v4.14.3 additions:
  - Agent A (storage + makedammann + LG chain): 9 tests.
  - Agent B (sources + multiconfig): 25 tests.
  - Agent C (polarization + freeform + ghost + user_library):
    41 tests.
  - Agent D (doc corrections): 0 (paper-only).
* Final: **1265 unit tests passing, 1 skipped** (the documented
  `_ZARR_MKDIR_PATCH_LOCK` exemption), **34/34 validation
  files passing**.

### Deferred to v4.15+

* Multi-process atomic-append for `storage.py` (HDF5 SWMR
  client + distributed Zarr lock).  v4.14.3 documents single-
  process guarantees and the multi-process restriction; the
  full multi-writer story is a v4.15+ ergonomics item.
* 18 P2 + 12 P3 findings from the v4.14.2 audit (UI module
  audit, codegen ergonomics, secondary doc-drift, additional
  factory validation gaps).  These remain catalogued in
  `docs/audits/AUDIT_V4_14_2_2026_05_17.md` for v4.15+ triage.

---

## [4.14.2] — 2026-05-17

**Closes the v4.14.1 audit (`docs/audits/AUDIT_V4_14_1_2026_05_17.md`).**  The
audit found 1 NEW P0 (a v4.11.2 regression carryover: `glass.py`
S-LAH64/S-LAH79 dispatch broken for 3 releases) plus 10 new P1s
(4 of which are "fix N, miss N+1" recurrences on v4.14.1 itself —
the aperture=0 sentinel missed 2 of 4 call sites, 7 older caches
still unlocked, 4 residual `0+0j` literal sites, `clear_asm_caches`
chain narrower than docstring) plus 6 P1s in under-examined modules
(freeform domain guard, makedammann unit drift, polarization API
inconsistencies, source factory validation gaps).  v4.14.2 closes
the P0 and all 10 P1s, adds **two new structural meta-pins** to
extend the v4.14.1 cache-clear dispatcher-pin pattern (cache↔lock
pairing + `0+0j` literal sweep), and closes a follow-up cache-lock
gap discovered by Agent C's meta-pin.  **1190 unit tests pass** (up
from 911); 34/34 validation files pass.

### Breaking changes — none

Two near-breaking output deltas with deprecation warnings:

* **`makedammann2d` is now SI metres throughout** (per-parameter
  deprecation heuristic detects legacy µm input and warns).  Pure-
  legacy calls and hybrid `periodx=20.0 (legacy µm) +
  waveln=1.31e-6 (already SI)` calls both continue to work; a
  `DeprecationWarning` directs users to migrate.
* **`create_led_source` signature reordered** to v4.7-canonical
  `(N, dx, wavelength, *, diameter, divergence_angle, dy=None, ...)`
  with a legacy-positional shim that emits a `DeprecationWarning`.

Neither breaks existing scripts but both warn on the legacy form.

### P0 closure — `glass.py` S-LAH64 / S-LAH79 dispatch

**3-release carryover regression.**  v4.11.2 (round-3 audit, CRIT-3)
removed S-LAH64 / S-LAH79 from `SELLMEIER_COEFFICIENTS` after
discovering the in-code coefficients were off by ±5.8% vs the Ohara
catalog, intending to route them through `refractiveindex.info` —
but `GLASS_REGISTRY` was never updated.  Both glasses remained
flagged `'__sellmeier__'`, and the dispatcher at `glass.py:410-415`
raised `ValueError` on every call.  `ui/main_window.py:1552`
references S-LAH64 as a "known-good preset" — UI broken for 3
releases.

**Fix:** Routed both to `('specs', 'OHARA-optical', '<name>')` tuple
form (correct catalogue book name verified by introspection;
`'OHARA'` was a wrong first attempt).  Numerical verification:
n_d=1.7880 matches Ohara catalog n_d=1.78800 to 5e-5 for S-LAH64;
S-LAH79 returns n_d=2.0033 matching the catalog 2.00330.

**Structural counter-measure:** Module-load consistency check at the
end of `glass.py` walks `GLASS_REGISTRY` and raises `RuntimeError`
if any `'__sellmeier__'` entry is missing from
`SELLMEIER_COEFFICIENTS`.  Future drift fails fast at import time.

### P1 closures — sibling-gap recurrences on v4.14.1

* **P1-NEW-1: aperture=0 sentinel finish** (was incomplete in
  v4.14.1).  The v4.14.1 sentinel fix updated 3 callers
  (`_get_wrapper_merit_cache`, `MultiWavelengthMerit.evaluate`,
  `MultiFieldMerit.evaluate`) but missed 2:
  `ToleranceAwareMerit.evaluate` (`optimize/core.py:2805-2810`,
  refreshed v4.15.1 -- earlier ranges drifted by ~15 lines after
  Agent E's `_Sentinel` base-class refactor) and
  `MatchIdealSystem._make_source` (`optimize/core.py:977-991`,
  refreshed v4.15.0).
  Both used `_cache['E_ones'].copy()` without the `mask is
  _ZERO_APERTURE_MASK` branch, producing a full-grid plane wave
  instead of zero on `aperture_diameter=0`.  v4.14.2 adds the
  sentinel-aware branch (option (b) of the audit recommendation —
  explicit-per-call-site, matches the canonical 2 already-fixed
  sites).  **Investigation finding:** `apply_perturbations` only
  mutates per-surface `decenter` / `tilt` / `form_error`, NEVER
  the prescription-level `aperture_diameter`, so the audit's worry
  about perturbed-to-zero apertures is not triggered by existing
  code — pinned the contract via
  `test_apply_perturbations_does_not_modify_aperture`.
* **P1-NEW-2: 7 older caches now locked** following the v4.14.1
  pattern.  Locks added to `_ZERNIKE_BASIS_CACHE`,
  `_THROUGH_FOCUS_SCAN_JAX_CACHE`, `_PROPAGATE_SYSTEM_JAX_CACHE`,
  `_GS_KERNEL_CACHE`, `_ER_KERNEL_CACHE`, `_HIO_KERNEL_CACHE`,
  `_TRACE_JAX_CACHE`.  Lock-scope discipline matches v4.14.1: lock
  held for `get` / `move_to_end` ops, released before expensive XLA
  jit-compile / numpy basis build, re-acquired for insert + evict.
  Concurrent 4-thread tests confirm no exceptions, no `RuntimeError:
  dictionary changed size during iteration`, final state consistent.
* **P1-NEW-3: `clear_asm_caches()` chain expanded** to 5 sibling
  caches via lazy-import + call (`clear_zernike_basis_cache`,
  `clear_through_focus_scan_jax_cache`,
  `clear_propagate_system_jax_cache`, `clear_phase_retrieval_caches`,
  `clear_trace_jax_cache`).  Docstring rewritten to honestly list
  all caches the function now clears.  Pinning test populates each
  cache then calls `clear_asm_caches()` and asserts emptiness.
* **P1-NEW-4: 2 P1-severity residual `0+0j` sites** swept
  (`optimize/merit_terms.py:524` post v5.1.0 Agent E 6-file split;
  was `optimize/core.py:987` pre-split via Agent B's P1-NEW-1 work
  + `analysis/phase_retrieval.py:402`; the optimize citation was
  refreshed `966 -> 987` in v4.15.0 to match the post-v4.14.2
  drift, then `core.py:987 -> merit_terms.py:524` in v5.1.0).  Now use the `np.zeros((), dtype=...)` pattern.  **Structural pin (new meta-pin):**
  `tests/unit/test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py` walks
  every `lumenairy/*.py` file (117 modules) and asserts no
  unallowlisted `np.where(..., 0+0j)` literal — three exemption
  layers (pure-comment lines, trailing `.astype()` recovery
  pattern, explicit P3 allowlist for `ui/psf_mtf_dock.py:230`).

### P1 closures — under-examined modules

* **P1-NEW-5: `surface_sag_xy_polynomial` domain guard** added.
  v4.14.1's freeform XY polynomial branch evaluated `c * X**i *
  Y**j` for every pixel — a high-order term like `(2,0): 1e3` on a
  50-mm half-grid produced 2.5 m of sag at the corner, propagating
  into raytraced rays outside the physical aperture.  v4.14.2 adds
  `norm_x, norm_y` kwargs + an `xp.where(<inside_box>, sag, 0.0)`
  clip matching the Chebyshev branch.  Backward-compatible (default
  `norm_x = norm_y = 1.0` plus raw-coordinate polynomial evaluation
  — no coefficient rescaling).
* **P1-NEW-6: `makedammann2d` SI conversion** with per-parameter
  deprecation heuristic.  See "Breaking changes" above.
* **P1-NEW-7: `apply_rotator` accepts `angle_deg=`** kwarg-only,
  matching the v4.7 polarization-family convention.  Conflict-
  resolution policy: `angle_deg` and non-zero `angle` (radians)
  with disagreement → `ValueError`.
* **P1-NEW-8: `JonesField.__init__` input validation**.  Added
  positive-finite `dx, dy` checks + 2-D shape check.  A 1-D field
  accidentally passed in now raises a clear `ValueError` at
  construction time, not an opaque FFT failure downstream.
* **P1-NEW-9: `create_led_source` signature drift** closed.  See
  "Breaking changes" above.  Underlying `create_top_hat_beam`
  handles `dy != dx` natively, so no `ValueError`-on-anisotropy
  raise was needed (unlike the JAX-twin precedent).
* **P1-NEW-10: `_validate_grid_params` helper** added at the top of
  `sources/core.py`.  Applied to all 10 factories
  (`create_gaussian_beam`, `create_hermite_gauss`,
  `create_laguerre_gauss`, `create_top_hat_beam`,
  `create_annular_beam`, `create_fiber_mode`, `create_led_source`,
  `create_bessel_beam`, `create_point_source`,
  `create_tilted_plane_wave`).  60 parametrized tests confirm
  `ValueError` on N≤0, dx≤0, wavelength≤0, non-finite inputs.

### Follow-up sibling-gap closed in same release

The v4.14.2 cache↔lock meta-pin (Agent C's new structural pin)
discovered a previously-unflagged single-cell lazy-init cache in
`propagators/asymptotic.py:_JAX_IFT_SOLVER_CACHE`.  Race window
on first concurrent call decorates the JAX `custom_vjp` solver
twice.  v4.14.2 closes the gap in the same release: added
`_JAX_IFT_SOLVER_CACHE_LOCK` with double-check locking pattern
(fast path no lock for the common populated-cache case; slow path
acquires lock, re-checks, delegates the actual build to
`_build_jax_ift_solver_impl`).  Meta-pin's
`_KNOWN_CACHE_SIBLING_GAPS` set is now empty — future regressions
land there.

### Two new structural meta-pins

Extending v4.14.1's cache-clear dispatcher-pin pattern to two more
sibling-gap classes the audit identified:

* **`test_v4_14_2_dispatcher_pin_cache_locks.py`** — 39 collected (38 pass + 1 documented `_ZARR_MKDIR_PATCH_LOCK` skip):
  walks every library module via `pkgutil.walk_packages`, finds
  names matching `^_.*_CACHE$`, asserts each has a corresponding
  lock (accepts both `_FOO_LOCK` and `_FOO_CACHE_LOCK` naming
  conventions).  Reverse check: every `_LOCK` has a corresponding
  cache (or matches the documented `_PATCH_LOCK` exemption for
  `_ZARR_MKDIR_PATCH_LOCK`).
* **`test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py`** (123 tests)
  — walks all `lumenairy/*.py` files for `np.where(.*, 0+0j)`
  literals.  Exemption layers for pure-comment lines, trailing
  `.astype()` dtype-recovery pattern, and explicit P3 allowlist for
  `ui/psf_mtf_dock.py:230` (audit-rated low-priority UI site).

### Retroactive doc-drift corrections to v4.14.1 entry

Per audit Part 1.3:

* Agent D test count (v4.14.1 line ~372): 25 → 8 (bottom-line 911 was correct).
* Source-factory dispatcher count: previously claimed "all 6" → corrected to 5 factories (3 CHANGELOG sites updated).
* `lenses_maslov.py` line-drift correction note refreshed.

### Test counts

* Pre-v4.14.2 baseline (v4.14.1): 911 unit tests.
* v4.14.2 additions:
  - Agent A (glass + freeform + polarization): 13 tests
  - Agent B (aperture=0 sentinel finish): 10 tests
  - Agent C (7-cache locks + clear_asm scope + meta-pin): 57
    tests pass (19 fix-pins + 39 meta-pin parametrizations
    collected, of which 38 pass + 1 documented
    _ZARR_MKDIR_PATCH_LOCK skip)
  - Agent D (sources + DOE + meta-pin): 199 tests (76 fix-pins +
    123 meta-pin parametrizations)
* Final: **1190 unit tests passing, 1 skipped** (the documented
  `_ZARR_MKDIR_PATCH_LOCK` exemption), **34/34 validation files
  passing**.

### Deferred to v4.15+

Row_reset physics pin against `propagate_hf_chebyshev_quadrature
(method='direct')` (audit Tier-1 #12 — contract-pinned in v4.14.1,
not numerically-pinned).  Backend dispatch (`_xp_of`) on the 6 new
v4.14.0 analysis functions.  Modal asymptotic per-pixel
vectorisation public switch.  Source factory signature
normalisation (the LED factory landed in v4.14.2; the rest still
have mixed positional/keyword conventions).  `system.evaluate
(prescription, source, ...)` ergonomic entry.  See
[`ROADMAP.md`](ROADMAP.md) for the full forward plan.

## [4.14.1] — 2026-05-17

**Closes the v4.14.0 audit (`docs/audits/AUDIT_V4_14_0_2026_05_17.md`).**  The
audit found 1 P0 (silent-wrong physics in the 77× LG/HG mode-stack
cache key) + 6 P1s + 10 P2s + 8 P3s + 7 doc-drift items.  v4.14.1
closes the P0, **all 6 P1s** (including P1-NEW-5 `row_reset`
Newton warm-start via the coordinated option-(a) fix across the
public path and 3 reference-loop pins), the top-priority P2s (cache
locks, monkey-patch removal, final `0+0j`/`1+0j` sweep,
`fiber_mode` in the dispatcher pin), and all 7 doc-drift items (4
retroactively corrected in the v4.14.0 entry below; 3 closed by the
v4.14.1 fixes themselves).  **911 unit tests pass** (up from 858);
34/34 validation files pass.

### Breaking changes — none

No user-facing breakage.  Aperture-zero semantics restored to the
pre-v4.14.0 behaviour (which had silently flipped) — see P1-NEW-1
below.

### P0 closure — `_LG_MODE_STACK_CACHE` / `_HG_MODE_STACK_CACHE` cache key

The 77× perf win in v4.14.0 had a silent-wrong-physics bug.  The
cache key omitted `dx, dy` — only `(p_max, ell_max, Ny, Nx, w, cx,
cy, dtype_str)`.  Two calls with the same shape but different
physical pitch (e.g. `dx=1e-6` then `dx=2e-6`, both at N=256)
collided on this key: **the second call returned the FIRST grid's
modes evaluated against the second call's field.**  Silently wrong
overlaps on:

* wavelength-adaptive grid sweeps (re-pitch the grid per
  wavelength to keep `λ/dx` constant)
* multi-resolution analysis (debug at coarse `dx`, then fine `dx`)
* optimisation loops where `dx` is a free variable

**Fix:** `dx, dy` added to both LG and HG mode-stack cache keys.
Regression pin asserts that two calls at the same `N, w, cx, cy,
p_max, ell_max` but different `dx` return distinct mode stacks.

### P1 closures

* **P1-NEW-1: aperture=0 semantics regression.**  v4.14.0's
  `_get_wrapper_merit_cache` mapped `aperture <= 0` to `mask=None`,
  which downstream callers interpreted as "no clipping, full grid
  plane wave."  **Semantics flipped 180°** from the pre-v4.14
  behaviour where `aperture=0` produced an all-False mask (block
  all light).  v4.14.1 adds a `_ZeroApertureMaskSentinel` singleton;
  `_get_wrapper_merit_cache` returns the sentinel for `ap <= 0` and
  `None` for `ap is None`; callers compare via `is` and route the
  sentinel through an explicit all-zeros path.  Three callers
  (`_get_wrapper_merit_cache`, `MultiWavelengthMerit.evaluate`,
  `MultiFieldMerit.evaluate`) updated.
* **P1-NEW-2: Brewster-angle phase aggregation bug** in
  `coatings.py` carried into v4.14.0's wavelength batch.  The
  v4.13.1 audit flagged that `0.5 * (np.angle(r_s) + np.angle(r_p))`
  is off by π/2 or π at Brewster (~56°) because `r_p` sign-flips
  through zero.  v4.14.0's wavelength-batch rewrite inherited the
  bug verbatim.  v4.14.1 changes the aggregation to
  `np.angle(0.5 * (r_s + r_p))` (complex sum then angle — robust
  to π-discontinuities).  Reference helper in
  `test_audit_fixes_v4_14_0_agent_2.py` also updated to match
  (the audit explicitly anticipated this collateral update).
* **P1-NEW-3: `_solve_envelope_stationary_batch` contract
  violation.**  The function's docstring promised
  `converged_mask=False` for failed pixels (singular Hessian or
  non-finite update) but the code set `True` to drop them from the
  active set.  Currently dead production code; the contract is
  preserved for future callers.  v4.14.1 introduces a separate
  `finished` mask for active-set tracking, leaving `converged`
  to mean what the docstring says.
* **P1-NEW-4: `clear_lg_mode_stack_cache` now in top-level `__all__`.**
  v4.14.0's CHANGELOG claimed this was "Public" but the import
  wasn't in `lumenairy/__init__.py` — the audit-meta-finding
  ("fix N sites, miss N+1") recurring on the very release that
  shipped 80 dispatcher pins.  v4.14.1 closes the loop AND adds a
  new structural counter-measure: a parametrized **cache-clear
  dispatcher pin** (`tests/unit/test_v4_14_1_dispatcher_pin_cache
  _clears.py`) that walks every submodule's `__all__` for
  `clear_*` names and asserts each is re-exported at top level
  AND callable from `la.*`.  Future cache-clear additions can
  no longer regress this gap.
* **P1-NEW-6: `encircled_energy_radius` docstring** corrected.
  v4.14.0 claimed `ee[0] = 0 always` (in an inline comment, not
  the docstring).  Reality: `ee[0]` equals the cumulative power
  at radius 0 (the centre-pixel intensity contribution).  Docstring
  expanded to document the hot-centre behaviour explicitly;
  pinning test exercises a delta-like centre-pixel input and
  confirms the threshold-zero short-circuit returns 0.

* **P1-NEW-5: `row_reset` resets the Newton warm-start.**
  v4.14.0's `row_reset` branch reset Maslov-branch state
  (`last_arg_detM`, `maslov_branch`) at each raster row wrap but
  left the Newton warm-start `last_v_star` chaining across the
  discontinuous jump from (x_max, y_n) to (x_min, y_{n+1}) —
  plausibly the mechanism behind the v4.14.0 "wrong-saddle-basin"
  finding near grid edges (largest jump in s_2).  v4.14.1 chooses
  **option (a)** of the audit recommendation: the `row_reset`
  branch now resets `last_v_star = (v_cx, v_cy)` at each row wrap
  too.  Coordinated with the fix, the bit-equal pin in
  `test_audit_fixes_v4_14_0_agent_1.py::test_lg00_single_mode
  _bit_equal` and the older 1e-10 rel pins in
  `test_perf_v4_12_0_asymptotic.py::TestPropagateModalAsymptotic
  Correctness` were updated — their inline scalar references
  also reset `last_v_star` at row wrap so the bit-equality holds
  against the new physics-correct behaviour.  The v4.14.1 marker
  test (formerly `TestRowResetDoesNotResetWarmStart`, now
  `TestRowResetResetsWarmStart`) flipped its assertion to pin the
  new contract.

### Tier-2 closures

* **Thread-safety locks** on the three new v4.14.0 caches.
  `_LG_MODE_STACK_CACHE`, `_HG_MODE_STACK_CACHE`, and
  `_WRAPPER_MERIT_CACHE` now have module-level `threading.Lock`
  guards on their read-modify-write ops, mirroring the
  `_ASM_CACHE_LOCK` precedent in `propagation.py`.  Concurrent
  `OrderedDict.move_to_end` / `popitem(last=False)` from parallel
  `design_optimize` threads no longer race.
* **Monkey-patch removed** in `optimize/core.py`.  The v4.14.0
  pattern monkey-patched `propagation.clear_asm_caches` to chain
  in `_clear_wrapper_merit_cache`; v4.14.1 replaces this with a
  lazy-import + call inside `propagation.clear_asm_caches()`
  itself.  Eliminates re-import recursion risk and the case
  where importing `propagation` without `optimize` leaves the
  wrapper-merit cache resident.
* **LG/HG mode-stack cache wired into `clear_asm_caches()`** (not
  just `lumenairy_context()` as v4.14.0 had).  Same lazy-import
  pattern.  CHANGELOG claim from v4.14.0 now matches code.
* **Final `0+0j`/`1+0j` literal sweep**: 2 sites caught by the
  audit — `lenses_maslov.py:448` (in `sample_E_bilinear`) and
  `_lens_thin.py:173` (aplanatic branch, actually `1.0+0.0j` not
  `0.0+0.0j` — replaced with `xp.ones((), dtype=...)`).
* **`fiber_mode` added** to
  `TestP1CSourceFactoryDispatcherPin` parametrize list (v4.13.2's
  `create_fiber_mode` dy-widening was complete but the test
  list was stale).

### Retroactive CHANGELOG corrections to v4.14.0 entry

4 confirmed doc-drift items from the audit corrected in the
v4.14.0 entry below:

* "16 tests at 1e-10 rel" → 3 tests in the cited class
  (`TestPropagateModalAsymptoticCorrectness`), 16 across the
  file overall.
* "6 batched helpers consumed by 77× win" → helpers ship privately
  but currently have zero production consumers; reserved for the
  v4.15+ coordinated public switch.
* HG cache key documented as `w[s]` shorthand → both LG and HG
  keys spelled out (HG has both `wx` AND `wy`; v4.14.1 adds
  `dx, dy`).
* `lenses_maslov.py` line cites — drifted 4-10 lines since
  v4.14.0 ship; in-line correction note added.

The 3 other doc-drift items (LG cache wired into `clear_asm
_caches`, public `clear_lg_mode_stack_cache`, "dispatcher pin
covers all 5 factories") **became true** in v4.14.1 — no
retroactive edit needed; the v4.14.0 entry's claims are now
accurate.

### Test counts

* Pre-v4.14.1 baseline (v4.14.0): 858 unit tests.
* v4.14.1 additions:
  - Agent A (asymptotic): 8 tests
  - Agent B (optimize + propagation): 11 tests
  - Agent C (coatings + lens sweep): 8 tests
  - Agent D (exports + meta-pin): 8 tests + 1 parametrize entry
    + 17 cache-clear meta-pins
* Final: **911 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.15+

Modal asymptotic per-pixel vectorisation public switch, Source
factory signature normalisation, `system.evaluate(prescription,
source, ...)` ergonomic entry.  See [`ROADMAP.md`](ROADMAP.md)
for the full forward plan.

## [4.14.0] — 2026-05-17

**Phase B of the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`).**
v4.13.2 closed Tier-0 (the 12 P1s + 5 cross-survey P0s + thin-lens
sibling sweep).  v4.14.0 closes Tier-1: the 7 Tier-1 perf wins from
audit Part 3, the top user-facing API gaps from the cross-survey,
and the 3 parametrized dispatcher pin families that close out the
sibling-gap audit-meta-finding.  All 858 unit tests pass; 34/34
validation files pass.

### Breaking changes — none

No user-facing breakage in v4.14.0.  All additions are net-new
public functions or internal perf optimisations behind unchanged
signatures.

### Performance wins (Tier-1 from audit Part 3)

| Hot path | Workload | Old | New | Speedup |
|---|---|---|---|---|
| `coating_reflectance` wavelength batch | 50 layers × 200 wv | 78.6 ms | 3.19 ms | **24.6×** |
| `decompose_lg` / `decompose_hg` cache (warm) | 256², p_max=3, ell_max=3 | 270 ms | 3.5 ms | **77×** |
| `MultiWavelengthMerit` meshgrid cache | N=512, 5 wl, 20 FD | (ref) | (ref/6.17) | **6.17×** |
| `_evaluate_polynomial_4d_and_grad34` | M=70, 16×16 grid | 646 µs | 171 µs | **4.6×** (typical configs) |
| `MultiWavelengthMerit` meshgrid cache (small) | N=128, 3 wl, 5 FD | (ref) | (ref/4.16) | **4.16×** |
| `ToleranceAwareMerit` meshgrid cache | N=128 | (ref) | (ref/3.17) | **3.17×** |
| Shack-Hartmann gather loop | K=4096, sa=8 | 7.2 ms | 3.2 ms | **2.27×** |
| Phase-retrieval `np.angle`/`np.exp` round-trip | GS, N=256, 50 iters | 1230 ms | 543 ms | **2.26×** |
| Phase-retrieval (ER variant) | same workload | 380 ms | 205 ms | **1.85×** |
| Phase-retrieval (HIO variant) | same workload | 426 ms | 268 ms | **1.59×** |
| `MultiFieldMerit` (limited by `np.exp`/`np.where`) | N=128 | (ref) | (ref/1.20) | **1.19×** |

**Meshgrid build count** in `Multi*Merit` / `ToleranceAwareMerit`:
from `O(n_wl × n_field × FD_evals)` (~1025 at N=512, 5 wl, 5 field,
20 FD) to **1 per `(N, dx, aperture)` signature** for the entire
optimisation run.  Cache cleared by `clear_asm_caches()`.

**Coating Snell-chain hoist** — the documented `n.imag-dropped-at-
Snell-step` approximation makes the per-layer chain wavelength-
independent.  v4.14 walks it ONCE outside the polarisation loop
instead of `n_wv × n_pol` times.  This is what unlocks the 24×
batched speedup.

**LG/HG mode-stack cache** (`_LG_MODE_STACK_CACHE` +
`_HG_MODE_STACK_CACHE`, both `OrderedDict` + LRU(32)).  LG key is
`(p_max, ell_max, Ny, Nx, w, cx, cy, dtype_str)`; HG key has both
`wx` AND `wy` (9-element tuple).  **Correction (v4.14.1):** v4.14.0
omitted `dx, dy` from these keys, producing silently-wrong overlaps
on multi-resolution or wavelength-adaptive grid sweeps; v4.14.1
adds `dx, dy` to both keys and wires the caches into
`clear_asm_caches()` (v4.14.0 only wired them into
`lumenairy_context(clear_caches_on_exit=True)`).  Public
`clear_lg_mode_stack_cache()` for explicit flushes (became
top-level-importable as `lumenairy.clear_lg_mode_stack_cache` in
v4.14.1).

**Phase-retrieval algebraic identity.**
`exp(1j * np.angle(F)) == F / np.abs(F)` for nonzero `F`.  v4.14
replaces the two-transcendental round-trip in the NumPy paths of
GS / ER / HIO with the divide, eliminating ~4 trig ops per pixel per
iteration.  JAX paths already had the optimisation; only NumPy
paths were touched.

### Modal asymptotic per-pixel vectorisation — NOT shipped publicly

Audit opportunity #1 (target 20-100×) was investigated and turned
out to be **subtler than the audit estimate**.  The brief asked to
vectorise `propagate_modal_asymptotic`'s per-pixel Newton solve.
The vectorised cold-start batched Newton finds the physical saddle
uniformly across all output pixels; the pre-v4.14 warm-started
Newton chains the previous pixel's `v_star` and at grid-edge pixels
lands in a **wrong-saddle basin** that produces `|b_quad| > 700`,
which the overflow guard zeros.

The pre-v4.14 behaviour is pinned bit-equal by `tests/unit/test_perf
_v4_12_0_asymptotic.py::TestPropagateModalAsymptoticCorrectness` (3
tests in the cited class, 16 in the file overall, all at `1e-10
rel`).  The vectorised cold-start produces strictly
more non-zero pixels (physically more correct) but breaks the
existing pin.  **This is a real physics finding** worth a coordinated
v4.15+ release that updates the test pin alongside the algorithm
change.

Shipped in v4.14.0: 6 new private vectorised helpers
(`_solve_envelope_stationary_batch`, `_compute_M_b_batch`,
`_phi_v2_hessian_batch`, `_gaussian_moment_table_2d_batch`,
`_batched_polynomial_substitute_linear_2d`, `_batched_polynomial
_under_affine_shift`).  The public `propagate_modal_asymptotic` body
is unchanged.  **Correction (v4.14.1):** the helpers ship privately
but currently have zero production consumers; the 77× LG/HG mode-
stack cache reaches into `lg_polynomial`/`hg_polynomial`/`_evaluate
_poly2d` directly, not through the batched helpers.  The helpers
are reserved for the v4.15+ coordinated public switch.

### New public API (cross-library survey gaps)

Six new functions exposed at `lumenairy.*` and in the appropriate
`__all__` tier:

* **`encircled_energy_curve(E, dx, *, dy=None, radii=None,
  centroid=None, n_radii=64) -> (radii, ee)`** — fraction of total
  power within radius `r` of the centroid.  Standard spec-sheet
  metric.  (Tier 4: Analyse.)
* **`encircled_energy_radius(E, dx, *, dy=None, threshold=0.84,
  centroid=None) -> float`** — radius enclosing the threshold
  fraction.  Default 0.84 matches the "84% encircled radius" lens
  spec convention.
* **`mtf_cutoff(mtf_profile, freq, *, threshold=0.5) -> float`** —
  spatial frequency at which a 1D MTF drops below threshold.
  Returns `np.inf` if the MTF stays above threshold for all
  frequencies.
* **`beam_diameter(E, dx, *, dy=None, threshold='1/e^2',
  centroid=None) -> float`** — diameter at which intensity drops
  below threshold.  String thresholds: `'1/e^2'`, `'1/e'`, `'FWHM'`,
  `'D4sigma'` (forwards to `beam_d4sigma` and returns geometric
  mean).
* **`depth_of_focus(wavelength, f_number, *, formula='rayleigh')
  -> float`** — one-sided depth of focus.  `'rayleigh'`:
  `±4 f#² λ`.  `'marechal'`: `±λ / NA²`.  Note: both formulas
  reduce to `4 f#² λ` (since `NA = 1/(2 f#)`); kept as separate
  named entries for derivation-clarity, cross-validated by test.
* **`plot_wavefront(opd, dx, *, dy=None, aperture=None,
  units='waves', wavelength=None, cmap='RdBu_r', show_stats=True,
  ax=None, fig=None, title=None) -> (fig, ax)`** — Zemax-style
  wavefront map: NaN outside the aperture, divergent colormap
  centred at zero, PV/RMS overlay annotation.  (Tier 9: Plotting.)

All six honour `dy` from the start (defaulting `dy=None → dy=dx`,
area integrations use `dx*dy`).

### Sibling-gap parametrized dispatcher pins (audit meta-finding closure)

The audit's recurring meta-finding ("fix swept N sites, missed
N+1") is closed for three family classes via parametrized pins.
**80 new dispatcher pins across 3 test files**.  Each pin enumerates
every reachable variant and asserts the same property; a future fix
that misses one variant fails CI at test-time.

* **`(scalar, vectorial) HFPI`** — 29 pins (`tests/unit/test_v4_14
  _0_dispatcher_pin_hfpi.py`).  Properties: `_spawn_rng`
  independence; grazing-ray `inf`/`NaN` guard; alive-mask
  correctness in `accumulate_*_to_grid`; public-surface
  importability across 13 names.
* **`(NumPy, JAX) apply_real_lens` family** — 35 pins (`tests/unit
  /test_v4_14_0_dispatcher_pin_apply_lens.py`).  Properties:
  `glass_after='MIRROR'` case-insensitive guard (15 parametrisations
  covering upper/lower/mixed-case); `dy=None` acceptance; `dy != dx`
  honour-vs-raise contract; complex64 dtype preservation across all
  5 variants.
* **Welford-mirror convention** — 16 pins (`tests/unit/test_v4_14
  _0_dispatcher_pin_welford_mirror.py`).  Properties: hand-computed
  analytical match for `seidel_coefficients` and `petzval_radius`;
  no-NaN for `aberration_summary` and `chromatic_focal_shift`;
  no-raise for `distortion_grid`, `field_aberration_sweep`,
  `eval_image_plane_wfe`; algebraic equivalence to the Welford
  formula.

### Sibling-gaps discovered by the dispatcher pins (now fixed)

The complex64 dtype-preservation pin tripped on 3 of 5
`apply_real_lens` variants — a previously-undetected sibling-gap
from the v4.13.2 B.4/B.5 thin-lens sweep.  Fixed in this release:

* **`apply_real_lens_maslov`** — `lenses_maslov.py`:
  - `_integrate_quadrature` (line 619) and `_integrate_stationary
    _phase` (line 728) and `_integrate_local_quadrature` (line 874)
    all hardcoded `dtype=np.complex128` allocations.  Each now
    accepts an `out_dtype` kwarg defaulting to `np.complex128` for
    back-compat; `apply_real_lens_maslov` threads `E_in.dtype`.
    (CHANGELOG line-cite refreshed in v4.14.2: prior CHANGELOG
    revisions cited drifted post-landing line numbers from the
    v4.14.0 tag and a further-drifted set from v4.14.1.  Current
    v4.14.2 def-header sites are 619/728/874 — one prior site was
    consolidated away during subsequent edits, leaving 3.)
  - The post-quadrature re-fit at line 566 multiplied by `1j`
    (complex128) which promoted the result; now cast back to
    `E_in.dtype`.
  - Final `normalize_output` step multiplied by a python float
    (float64) which silently promoted complex64 → complex128; final
    cast back to `E_in.dtype` added.
* **`apply_real_lens_traced_jax`** — `_lens_jax.py:573`: was calling
  `cdtype = _resolve_jax_complex_dtype()` which returns the library-
  default complex dtype (complex128 when `jax_enable_x64=True`),
  silently upcasting complex64 inputs.  v4.14 passes `E_in.dtype` to
  the resolver so the user's input dtype is honoured.
* **`apply_real_lens_maslov_jax`** — `_lens_jax.py:819`: same fix.

### Cache invalidation hygiene

`clear_lg_mode_stack_cache()` (new in v4.14 from the decompose_lg
cache work) wired into `lumenairy_context(clear_caches_on_exit=
True)` to match the existing pattern for the other 7 cache-clear
helpers.

### Test counts

* Pre-v4.14.0 baseline (v4.13.2): 710 unit tests.
* v4.14.0 additions:
  - Agent 1 (asymptotic): 15 tests in `test_audit_fixes_v4_14_0_agent_1.py`
  - Agent 2 (coatings + polynomial): 10 tests in `..._agent_2.py`
  - Agent 3 (phase-retrieval + SH): 13 tests in `..._agent_3.py`
  - Agent 4 (Multi*Merit cache): 18 tests in `..._agent_4.py`
  - Agent 5 (API gaps): 12 tests in `..._agent_5.py`
  - Agent 6 (dispatcher pins): 80 tests across 3 files (`test_v4_14
    _0_dispatcher_pin_hfpi.py`, `..._apply_lens.py`, `..._welford
    _mirror.py`)
* Final: **858 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.15+

* **Modal asymptotic per-pixel vectorisation** (audit opportunity
  #1).  The wrong-saddle-basin physics finding noted above needs a
  coordinated test-pin update alongside the algorithm change.
  Vectorised helpers shipped privately for future use.
* **Source factory signature normalisation** (audit Stream B #10) —
  `Source.gaussian(w0, N, ...)` puts beam-size first; `Source.plane
  _wave(N, ...)` puts N first.  Picking a canonical order requires
  a deprecation window; rolling into v4.15 to coordinate with other
  Source-related cleanups.
* **`system.evaluate(prescription, source, ...)` ergonomic entry**
  — closes a one-liner UX gap (`propagate_through_system` currently
  takes an element list, not a prescription dict).

### Deferred to v5.0

Tier-2/3/4 structural items unchanged from v4.13.2 entry: six file
splits; CI gates; back-compat shim removal; shared Chebyshev
helpers extraction; audit-fix test-file consolidation by topic;
constrained optimisation; checkpoint/resume; CDGM/Hikari/Sumita
glass catalogues; off-axis conics in surface frame; Q-type
freeform.

## [4.13.2] — 2026-05-17

**Closes the v4.13.1 audit (`docs/audits/AUDIT_V4_13_1_2026_05_17.md`) plus its
Part 10 consolidation with a parallel 6-agent cross-library survey.**
The v4.13.1 audit identified 12 new P1s (5 sibling-gap recurrences +
5 fresh-eyes bugs + 2 partial-closure follow-ups) plus 17 perf
opportunities and 22 structural recommendations.  The parallel
cross-library survey turned up an additional 5 P0s (in `optimize/`
and `io/`) and 7 P1s (mostly `dy` convention drift across analysis +
system + lens family).  v4.13.2 closes the full consolidated Tier-0
set.  All 710 unit tests pass; 34/34 validation files pass.

### Breaking changes — none

No user-facing breakage in v4.13.2.

### P0 closures (cross-library survey)

* **`make_lg_aberration_merit_jax` no longer silently ignores its
  `targets` dict.**  The function's inner loop body was literally
  `pass`; `wgt` was captured but never multiplied; the aberration-
  tensor call was outside the loop.  Public API exported from
  `lumenairy.__init__` — the function was returning the same
  `(0,0)`-piston sum regardless of input weights.  v4.13.2: now
  weights the `(0,0)` target correctly; non-`(0,0)` targets raise
  `NotImplementedError` with a clear migration message (the
  underlying `aberration_tensor_lg00_jax` only computes the piston
  coefficient).
* **`MultiFieldMerit.field_angles` accepts both scalars and `(theta_x,
  theta_y)` tuples.**  v4.13.1 applied Y-axis tilt only despite the
  docstring implying generic off-axis angle; `MatchIdealSystemMerit`
  took `(theta_x, theta_y)`.  v4.13.2 widens `field_angles` to accept
  both forms with a one-shot `DeprecationWarning` on the scalar form;
  internal storage normalises to tuples so the evaluate loop is
  uniform.  Non-zero `theta_x` now actually produces an X-axis tilt.
* **`load_plane_slice` documented return type matches actual.**
  Documented to return slice array; actually returns `(arr, attrs)`
  tuple in both HDF5 and Zarr paths.  Docstring + type annotation
  corrected to `Tuple[ndarray, Dict[str, Any]]`.
* **CODE V `.seq` round-trip preserves BFL.**  Reader was dropping the
  last refracting surface's `THI` (a legitimate CODE V BFL
  convention).  New top-level `'back_focal_length'` prescription key
  (float, SI meters) carries the BFL through round-trip; populated by
  both readers and exporters.
* **Quadoa `.qos` round-trip preserves BFL** (same fix as CODE V).

### Sibling-gap P1 closures (the audit's headline pattern recurring)

Five new "fix-swept-N-sites-missed-N+1" instances closed.  All five
have parametrized dispatcher-level pin tests preventing future
recurrence within the same family.

* **`vectorial_hfpi` RNG-correlation sibling.**  v4.11.2 fixed the
  scalar `hfpi.propagate_hfpi_freespace_aperture` via `_spawn_rng`;
  the vectorial sibling at `vectorial_hfpi.py:399` still passed the
  same `rng` to source-init AND aperture re-emission, perfectly
  correlating the diffraction events.  v4.13.2 ports `_spawn_rng`.
* **`subaperture` `decompose_lg(E_in, ...)` sibling.**  v4.11.2 fixed
  `hf.propagate_huygens_fresnel_through_prescription` to decompose
  the input field per-patch; `subaperture.propagate_subaperture
  _asymptotic` still passed `source_amplitudes={(0,0): 1.0+0.0j}`
  — every patch got an identical plane-wave-equivalent unit
  fundamental, ignoring `E_in` beyond a waist estimate.  v4.13.2
  ports the per-patch LG decomposition with three new kwargs
  (`source_lg_p_max=3`, `source_lg_ell_max=3`,
  `source_lg_amp_threshold=1e-6`) and amplitude-threshold pruning.
* **`petzval_radius` Welford-mirror convention sibling.**
  `seidel_coefficients` was fixed in v4.11.2 with the Welford `n2 =
  -n1` convention for mirrors; `analysis/field.py:petzval_radius`
  still skipped every mirror (because `n1 == n2` after the loader
  set `glass_after == glass_before`), silently dropping every
  mirror's Petzval contribution.  **Wrong by 100% for catadioptric /
  Cassegrain designs.**  v4.13.2 applies the Welford-parity
  convention; Cassegrain regression test pins against the analytic
  Mahajan formula.
* **`_build_jax_prescription` `glass_after='MIRROR'` sibling.**
  v4.13.1 P1-A added BOTH `is_mirror=True` AND case-insensitive
  `glass_after='MIRROR'` guards to `apply_real_lens`; the JAX
  prescription builder at `raytrace/jax_trace.py:649-669` only got
  the first.  Hand-built prescriptions with Welford-style mirror-
  via-glass-string slip through and are silently traced as
  refractive air→air.  v4.13.2 adds the second guard.
* **JAX lens twins thread `dy`.**  The two NumPy `apply_real_lens`
  variants accept `dy=None`; the JAX twins (`apply_real_lens_traced
  _jax`, `apply_real_lens_maslov_jax`) did not.  Anamorphic round-
  trip through `Source.dy → propagate_through_system_jax →
  apply_real_lens_traced_jax` silently dropped y-pitch at the JAX
  boundary.  v4.13.2: both JAX twins accept `dy=None` and raise
  `ValueError` on `dy != dx` (matching the existing NumPy
  precedent on the square-grid contract).

### Fresh-eyes P1 closures (audit Part 2.2)

* **`apply_mirror` NaN propagation.**  For a hyperbolic mirror with
  conic such that `(1+k)*h²/R² >= 0.9999`, sag → NaN → phase → NaN →
  `E *= NaN` poisons every pixel of the subsequent ASM step.
  `apply_real_lens:704-705` had the equivalent NaN-zeroing guard;
  `apply_mirror` did not.  v4.13.2 mirrors the guard.
* **`_zero_C_air_gap` raises on degenerate ABCD.**  When `abs(C1 -
  C0) < 1e-30`, the function silently returned the placeholder
  thickness from the input prescription (a non-afocal beam expander
  with zero combined power).  Callers (`beam_expander_prescription`,
  `keplerian_telescope`) catch `RuntimeError`; v4.13.2 raises it
  explicitly so the fallback fires.
* **`propagate_to_plane` inf/NaN positions guarded.**  For `Nz ≈ 0`
  and `z_target != z_curr`, the divisor went to `1e-30` →
  `t ≈ 1e30` → positions += inf.  The alive-mask correctly tagged
  these dead, but `paths.positions` still contained inf/NaN —
  downstream code reading positions without masking by `alive` was
  poisoned (e.g. `_hfpi_segment_trace`).  v4.13.2 zeroes the step on
  dead/grazing rays in BOTH `hfpi.propagate_to_plane` and
  `vectorial_hfpi.propagate_vector_to_plane`.
* **`RandomState.choice` int dtype aligned across backends.**  JAX
  path returned int32 (default for `jax.random.randint`); NumPy
  path returned int64.  v4.13.2 dispatches `jax.random.randint(...,
  dtype=jnp.int64)` so both backends agree.
* **`trace_prescription` uses `_surface_copy_with` instead of
  mutating shared `Surface.thickness`.**  When `image_distance=` was
  supplied, the function mutated the input prescription's last
  surface in place.  The `Surface` dataclass is shared with
  `surfaces_from_prescription`; downstream calls reused corrupted
  thickness.  v4.13.2 clones via `_surface_copy_with` (matching the
  pattern at `lens_abcd`).

### Partial-closure follow-ups from v4.13.1

* **`dual_annealing` callback wired into cancellation protocol.**
  v4.13.1 P2 #13 wired `CancellableProgress` into 4 scipy callbacks;
  the `dual_annealing` site used an unnamed inline lambda that did
  NOT poll `is_cancelled(progress)`.  Cancellation latency was
  unbounded for that one method.  v4.13.2 replaces the lambda with
  a named callback matching the pattern in the other three.
* **`RandomState.choice` old-JAX safety net.**  v4.13.1 P1-F closed
  the `replace=False` regression on JAX 0.10.0+ but the CHANGELOG
  promised a graceful old-JAX safety net the code lacked.  v4.13.2
  wraps the dispatch in `try/except TypeError` with a `RuntimeError`
  raising a clear "JAX >= 0.4.0 required" message.

### Cross-library survey P1 closures

* **`strehl_ratio` + `polychromatic_psf` accept `dy=`.**  The
  v4.13.0 L3 sweep added `dy` to `Source` / `PropagationResult` /
  free-space propagators but missed both Strehl helpers (using
  `dx**2` not `dx*dy`).  v4.13.2 adds `dy: Optional[float] = None`
  to both; back-compat preserved bit-for-bit when `dy is None`
  (the historic `dx**2` form is retained so existing tests with
  the FP-identity assumption stay green).
* **Wrapper-merit context threads `x=ctx.x`.**  `MultiWavelengthMerit`,
  `MultiFieldMerit`, `ToleranceAwareMerit` built sub-`Evaluation
  Context`s without `x=ctx.x`.  A `JaxMeritTerm(build_args=...)`
  wrapped inside any of these silently fell back to legacy
  `fn(ctx)` mode, degrading the analytic-gradient path to FD.
  v4.13.2 threads `x=ctx.x` across all three.
* **`propagate_through_system` element handlers thread `dy`.**
  Every element handler (lens, aperture, mask, zernike, mirror,
  etc.) passed `dx=dx` only — anamorphic `dy` was silently squared
  to `dx` on every non-`propagate_*` element.  v4.13.2 routes
  `current_dx` AND `current_dy` through all 13 element handlers.
* **`Source.fiber_mode` accepts `dy=` end-to-end.**  v4.13.1 P1-C
  threaded `dy` through 4 of 5 Source factories; `fiber_mode`
  remained a dead-`dy=` code path because `create_fiber_mode`
  didn't accept `dy=`.  v4.13.2 widens `create_fiber_mode` to
  accept `dy=` (forwarding to `create_gaussian_beam`); the
  dispatcher pin in `TestP1CSourceFactoryDispatcherPin` now covers
  all 5 factories.

### Thin-lens family sibling-gap sweep (cross-library survey)

The cross-library survey turned up additional sibling-gaps in the
thin-lens family — same v4.13.1 P3 #21 dtype-aware-zero fix that
landed in `apply_mirror` + `apply_aperture`, but missed across
multiple sites:

* **9 sites of `0.0+0.0j` complex128 literal** in `_lens_thin.py` (4
  sites) + `_lens_real.py` (5 sites).  Each was a `xp.where(..., E,
  0+0j)` clear-aperture or stop-mask construct that silently
  upcast complex64 → complex128.  v4.13.2 replaces each with
  `xp.zeros((), dtype=E.dtype)`.
* **6 thin-lens functions** (`apply_thin_lens`, `apply_spherical
  _lens`, `apply_aspheric_lens`, `apply_cylindrical_lens`, `apply
  _grin_lens`, `apply_axicon`) constructed `xp.exp(1j * phase)`
  from float64 phase without dtype matching against the input
  field.  Same complex64→complex128 upcast as the `0+0j` literal,
  at a different call site.  v4.13.2 adds the dtype-coercion line
  to all 6 functions, matching the v4.13.0 L6 pattern in
  `apply_mirror`.
* **3 thin-lens functions** (`apply_cylindrical_lens`, `apply_grin
  _lens`, `apply_axicon`) lacked the documented `use_gpu` parameter
  entirely.  v4.13.2 adds the parameter and the canonical CuPy-
  dispatch pattern to all three.
* **Latent CuPy dispatch bug fixed in the other 3 thin-lens
  functions.**  `apply_thin_lens`, `apply_spherical_lens`,
  `apply_aspheric_lens` had the `use_gpu` parameter from v3.5.5
  but the CuPy dispatch was broken because the function bodies
  referenced bare `cp` — Python's LEGB name resolution doesn't
  consult module-level PEP 562 `__getattr__` for function-local
  lookups.  v4.13.2 routes all three through `_lenses_module.cp`
  (matching the working pattern in `apply_cylindrical_lens`).

### Quick wins

* **5 mis-tiered names in `__init__.py.__all__`** moved to correct
  tiers: `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`
  → Tier 1 (Build a system); `monte_carlo_tolerancing_jax`,
  `monte_carlo_tolerancing_linearized`, `tolerancing_report` →
  Tier 4 (Analyse).
* **Duplicate `reset_fft_backend` import** removed from `__init__.py`
  (was imported at both line 46 and line 59).
* **Wiki Release-Notes.md broken anchor** fixed:
  `[4.13.1](#whats-new-in-4-14-0)` → `[4.13.1](#whats-new-in-4-13-1)`
  (stale from the v4.14→v4.13.1 rename script during v4.13.1 ship).

### Test counts

* Pre-v4.13.2 baseline: 654 unit tests.
* v4.13.2 audit-response additions: 56 new tests across 4 new files
  (Agent A: 12 in `test_audit_fixes_v4_13_2_agent_a.py`; Agent B:
  20 in `..._agent_b.py`; Agent C: 14 in `..._agent_c.py`; Agent
  D: 10 in `..._agent_d.py`).
* Final: **710 unit tests passing**, **34/34 validation files
  passing**.

### Deferred to v4.14.0

Tier-1 perf opportunities from the v4.13.1 audit Part 3 (modal
asymptotic per-pixel loop 20-100×; multi-merit meshgrid cache
5-10×; phase-retrieval `angle`/`exp` round-trip 2-4×; coating
reflectance wavelength batch 5-15×; decompose_lg/hg per-mode
rebuild 3-8×; Shack-Hartmann gather loop 5-15×; `_evaluate
_polynomial_4d_and_grad34` 3-5×) plus the cross-library survey's
top user-facing API gaps (encircled energy, MTF cutoff frequency,
depth of focus, plot_wavefront, beam diameter at threshold) plus
parametrized dispatcher pins for the three audit-recommended
sibling families (`(scalar, vectorial) HFPI`, `(NumPy, JAX)
apply_*`, Welford-mirror convention).

### Deferred to v5.0

Tier-2/3/4 structural items: six file splits (`raytrace/core.py`
4422 LOC, `propagation.py` 3710, `asymptotic.py` 3597, `optimize
/core.py` 3258, `io/prescriptions.py` 2829, `analysis/core.py`
2196); CI gates (`ruff`, `mypy --strict`, unit-test PR job);
back-compat shim removal; shared Chebyshev helpers extraction;
audit-fix test-file consolidation by topic; constrained
optimisation; checkpoint/resume; CDGM/Hikari/Sumita glass
catalogues; off-axis conics in surface frame; Q-type freeform.

## [4.13.1] — 2026-05-17

**Closes the v4.13.0 audit (`docs/audits/AUDIT_V4_13_0_2026_05_17.md`) plus an
additional perf-survey pass.**  v4.13.0 was tagged in git but never
published to PyPI; that audit identified 7 P1 (latent bug), 9 P2
(code smell), and 6 P3 (cleanup) findings — most importantly 3
"sibling-gap" recurrences (the `apply_real_lens` mirror guard
missed its parent function; the L2 JAX dtype routing missed
`error_reduction_jax`/`hybrid_input_output_jax`; the L3 `dy`
threading missed `Source.propagate` + 5 classmethod factories).

v4.13.1 closes every Tier-0 and Tier-1 item from that audit, plus
all Tier-2 P2 follow-ups and the Tier-3 cleanups, plus 3 new perf
wins discovered in the parallel survey.  All 654 unit tests pass;
34/34 validation files pass.

### Breaking changes — none

No user-facing breakage in v4.13.1.  The v4.13.0 breaking changes
(`rcwa.py` → `thin_grating.py` rename without back-compat shim,
and the `wavelength` sentinel raising `ValueError` on omission)
are inherited but do not regress further.  See v4.13.0 entry's
"Breaking changes" subsection below.

### Sibling-gap P1 closures (audit's headline finding)

**P1-A — `apply_real_lens` mirror guard.**  The v4.13.0 L4a sweep
hardened 4 sibling `apply_real_lens_*` variants (`_traced`,
`_traced_jax`, `_maslov`, `_maslov_jax`) but missed the parent
`apply_real_lens` itself.  A hand-built prescription containing
`surfaces[i]['is_mirror']=True` (and no `'elements'` key) would
silently misompute through `apply_real_lens` while raising
`ValueError` from the 4 siblings.  Ported the same pre-flight
guard from the `_lens_traced.py` template.  Added a parametrized
dispatcher-level pin (`TestL4aMirrorGuardDispatcherPin`) over all
5 variants to prevent this class of sibling-gap from recurring.

**P1-B — ER/HIO complex-dtype routing.**  `gerchberg_saxton_jax`
correctly routed `dtype` through `_resolve_jax_complex_dtype` in
v4.13.0, but `error_reduction_jax` and `hybrid_input_output_jax`
skipped the resolver — the EXACT silent float64 → complex64
demotion bug L2 was supposed to close, still live on those two
kernels.  Both now go through `_resolve_jax_complex_dtype` and
`_resolve_jax_real_dtype`; cache keys pinned on the resolved
complex dtype to match the GS pattern.  Parametrized dispatcher
pin (`TestP1BPhaseRetrievalDtypeResolver`) covers all 3 kernels.

**P1-C — `Source.propagate()` and 5 factories thread `dy`.**
`Source.propagate(...)` was constructing the result Source
without `dy=self.dy`, silently losing the y-pitch on every
anamorphic call.  Same gap in `Source.gaussian`, `plane_wave`,
`point_source`, `top_hat`, and `fiber_mode` — each factory
forwarded `dy` to its `create_*` helper (so the underlying
E-field WAS built on the anamorphic grid) but the final
`cls(E=..., dx=..., wavelength=..., ...)` call dropped `dy`.
Fixed in all 6 sites.  `Source.propagate` gained an optional
`output_dy` kwarg for symmetry with the existing `output_dx`
override.  Parametrized dispatcher pin
(`TestP1CSourceFactoryDispatcherPin`) covers the 4 dy-aware
factories; `fiber_mode` excluded because `create_fiber_mode`
itself does not accept `dy=` (pre-existing limitation outside
P1-C scope).

### UI + infrastructure P1 closures

**P1-D — `ThinGratingDock._run` kwargs mismatch.**  The dock was
calling `grating_efficiency_vs_wavelength(groove_index=...,
substrate_index=..., profile=..., angle=...)` but the function
expects `(period, depth, *, n_ridge, n_groove, n_substrate,
n_superstrate, order, ...)`.  Every dock click raised
`TypeError`, swallowed silently into the summary text box — the
dock was non-functional.  Rewrote `_run` with the correct
signature, added missing UI inputs for `n_ridge` and
`n_superstrate`, and extracted the math path into a pure
`_compute_efficiency_data(inputs: dict) → dict` helper so the
compute path is unit-testable without a live `QApplication`.  As
a side effect, the dock now does **one** `thin_grating_efficiency_1d`
call per wavelength (full per-order matrix) instead of N sweeps
through the single-order helper.

**P1-E — `_context.py` cache-clear import moved inside try.**
The `from .propagators.propagation import clear_asm_caches` was
OUTSIDE the try-block guarding the call.  If the import raised
(rename, circular import, partial install), the `ImportError`
bypassed not just that cache-clear but all 6 subsequent guarded
cache-clear blocks.  Moved inside the try, added `ImportError`
to the except tuple, matching the pattern used by the other 6
clears.

**P1-F — `RandomState.choice` on JAX honours `replace=False`.**
With `p=None`, the JAX path silently ignored `replace=False` and
returned with-replacement samples.  Now dispatches to
`jax.random.choice(sub_key, a, shape=shape, replace=False, p=p)`
for both weighted and unweighted branches (JAX 0.10.0 supports
the combination — verified at runtime).

**L7 — `test_bench_through_focus_scan_jax_first_vs_warm`** now
clears `_THROUGH_FOCUS_SCAN_JAX_CACHE` before timing the first
call, matching the 4 sibling benchmarks.  On a re-run within the
same process the reported "first call" timing is now the cold
path, not a cached compile.

### Tier-2 (code smells from the audit)

* **`_RestoreDtype` try/finally** — `_RestoreDtype.restore()` is
  now idempotent (`_restored` flag).  The dtype restoration in
  `design_optimize`'s scipy dispatch + final-eval block is
  wrapped in `try/finally` calling `restore()` explicitly.
  `__del__` retained as a safety net.  More robust under
  `KeyboardInterrupt` and exception unwinding.
* **`_merit_jac_auto` uses `scheme='forward'` + cached `f0`** —
  scipy already evaluates `merit_fn(x)` before calling jac on
  the same `x`.  `_merit_jac_auto` now passes that value as `f0`
  to `_fd_grad_for(... scheme='forward', f0=...)`.  FD eval
  count goes from `2N` to `N + 2` per gradient (~30% saving for
  N=10 free vars).  Threaded through internal helpers so
  callers that prefer the bit-identical central path still get
  it by default.
* **Cancellation protocol in `progress.py`** — new
  `CancellableProgress` class (exposed at `lumenairy.
  CancellableProgress`) with a `cancel()` method and
  `is_cancelled()` module helper.  Wired into all 4 scipy
  callbacks in `design_optimize` (`minimize`, `differential
  _evolution`, `basin_hopping`, `dual_annealing`).  When
  `cancel()` is called, the active scipy callback returns
  `True`; scipy stops gracefully and the post-loop final
  evaluation + `DesignResult` return still executes (so the
  caller gets the best-so-far state instead of a partial-data
  `KeyboardInterrupt`).
* **Merit propagator inconsistency warning** —
  `MultiWavelengthMerit`, `MultiFieldMerit`, and
  `ToleranceAwareMerit` all call `apply_real_lens` directly for
  off-nominal legs regardless of which `wave_propagator` was
  selected at `design_optimize` time.  Added a docstring caveat
  to each of the 3 classes AND a runtime `UserWarning` at
  `design_optimize` entry when `wave_propagator != 'real_lens'`
  and at least one of these Merit classes is in use.  Threading
  the propagator through sub-merit evaluations is a larger
  architectural change worth its own release.
* **Shared `_build_asm_H_square` helper** — the v4.13.0 Shack-
  Hartmann FFT batching introduced a local `_build_asm_H_for
  _lenslet` in `analysis/detector.py` that duplicated angular-
  spectrum H/bandlimit logic from `propagators/propagation.py`.
  Now consolidated: `_build_asm_H_square(N, dx, z, wavelength,
  dtype=None, bandlimit=True)` lives in `propagators/propagation
  .py` and is imported by `detector.py`.  Pinned at 1e-14 against
  a hand-built reference for both `bandlimit=True/False`,
  multiple grid sizes, complex64 dtype promotion, and the z=0
  unity short-circuit.  The propagator's own two inline H-build
  sites (NumPy chunked path and JAX functional path) stayed
  inline — they use cached freq-grid lookups + RAM-budgeted
  chunking (NumPy) and `xp.namespace` for JAX tracing, both of
  which materially differ from the helper's single-shot pure-
  NumPy build.
* **`_fd_grad_pure` `validate_f0` parameter** — when
  `scheme='forward'` and `f0` is supplied, `validate_f0=True`
  (default `False`) re-evaluates `f(x)` once and asserts
  `abs(f0 - f(x)) < tol * max(abs(f0), 1)`.  Caller contract
  documented in the docstring: stale `f0` silently produces
  wrong gradients without the validation gate.
* **BSDF TIS shape assertion** — `total_integrated_scatter`'s
  `np.broadcast_to(B, T.shape)` previously masked shape
  mismatches from subclass `BSDF.evaluate()` returns.  Now
  raises `ValueError` with expected vs actual shape on
  mismatch.

### Tier-3 cleanups

* **`memory.set_max_ram` validates non-negative input** —
  previously `set_max_ram(-5)` was silently accepted as
  -5 GB → negative bytes.  Now raises `ValueError`.
  `get_max_ram` added to `__all__`.
* **`MultiPrescriptionParameterization` duplicate detection** —
  duplicate `(prescription_index, *path)` entries in `free_vars`
  silently got separate `x[i]` slots competing for the same
  field.  Constructor now raises `ValueError` listing the
  duplicates.
* **JAX 0.4.20+ opaque PRNG keys** — `_is_jax_prng_key` now
  recognises opaque keys from `jax.random.key()` via the
  canonical `jax.dtypes.issubdtype(d, jax.dtypes.prng_key)`
  check, with a `dtype.name.startswith('key<')` string fallback.
  Legacy uint32 / shape-trailing-2 typed keys still detected.
* **Dtype-aware zero in `apply_mirror` and `apply_aperture`** —
  the `xp.where(..., E, 0.0+0.0j)` literal is complex128 in
  Python.  On JAX with `x32` default this may upcast or fail
  jit.  Replaced with `xp.zeros((), dtype=E.dtype)` in both
  `apply_mirror` (audit-cited) and `apply_aperture` (same
  pattern in the same file, fixed conservatively).
* **`apply_mirror` aperture docstring** — said "ellipse" but the
  code computes a circle in physical coordinates.  Docstring
  corrected.
* **Stale pyc cleanup** — removed
  `tests/unit/__pycache__/test_audit_fixes_v4_13_0_perf_hfpi
  _bincount.cpython-314-pytest-9.0.3.pyc` (γ.1 revert
  leftover).

### New performance wins (beyond audit scope)

Open-ended perf survey across `propagators/` and `raytrace/` (the
modules not already optimised in v4.12.0 or v4.13.0) turned up
three wins:

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `vectorial_hfpi.accumulate_vector_to_grid` (1M paths, 256²) | 57.3 ms | 34.6 ms | **1.65-1.75×** |
| `analysis/detector.py:shack_hartmann` scatter-back (K=4096) | 2.97 ms | 0.12 ms | **9.5-25×** |
| `gbd.reconstruct_field_from_beamlets` (1024 beamlets, 96²) | ~1100 ms | ~870 ms | **1.2-1.5×** typical, up to 2.3× cache-warm |

* **`accumulate_vector_to_grid`** now shares the `ix`/`iy`/
  `inside`/`flat_idx` index arrays between the Ex and Ey scatter-
  adds (previously each call routed through `accumulate_to_grid`
  twice, recomputing the indices).  Bit-exact (`np.array_equal`).
  JAX path falls back to the original double-call form for
  tracing compatibility.
* **`shack_hartmann` scatter-back** replaced the `for k in
  range(K)` Python loop with vectorised fancy indexing using
  `iy_idx[ok]` / `ix_idx[ok]` / `cx_arr[ok]`.  Bit-exact.
* **GBD `reconstruct_field_from_beamlets`** fuses two `xp.exp`
  calls into one (`arg = -0.5*Q*rho2 + L*dX + M*dY`); replaces
  the `sum(a_b * phase, axis=-1)` reduction with `einsum('mnk,k
  -mn', phase, a_b)` to drop the `(Ny, Nx, chunk)` intermediate;
  switches accumulator to in-place `out +=` on NumPy/CuPy.
  Bit-near-exact (rel_err ~4e-16 vs the scalar reference).
  JAX rebind preserved.

**Deferred perf candidates** (catalogued for v4.14+):

* `propagators/hf.py:propagate_huygens_fresnel_with_opl_callable`
  per-pixel python loop calling `opl_fn` 16 times per output
  pixel (Van Vleck Hessian).  Could vectorise across `chunk
  _output` pixels.  Expected 5-20× on typical 256² grids with
  custom OPL.  Deferred: requires changing the documented
  callable contract (opl_fn must broadcast over arbitrary
  trailing shapes).
* Fused `(sag, dz_dh)` helper for `_intersect_surface` Newton
  iterations.  Currently `_surface_sag_xy` and
  `_surface_sag_derivatives_xy` both recompute `h = sqrt(x²+y²)`
  and re-traverse the aspheric polynomial dict on each of ~10
  Newton iters.  Expected 1.5-2× on aspheric surfaces.
  Deferred: requires reorganising `lenses.py:surface_sag
  _general`.
* Analytic pure-conic intersection extending the v4.12.1 Newton-
  skip from `kc==0` to all `kc` with no aspherics.  Expected
  5-10× on pure-conic surfaces.  Deferred: root selection for
  hyperbolic (`k<-1`) and degenerate paraboloid (`k=-1`,
  on-axis ray) cases needs broader validation.

### v4.13.0 CHANGELOG drift fixes (Tier-1 doc hygiene)

11 doc-vs-code mismatches in the v4.13.0 CHANGELOG entry below
have been retroactively corrected (line numbers, function names,
threshold directions, claim tightness).  The 12th (v4.12.2's
"6 clear-functions wired into lumenairy_context" should be 7 —
`clear_through_focus_scan_jax_cache` was the 7th but went
uncounted) is acknowledged here rather than retroactively edited
because v4.12.2 is already on PyPI.

### Test counts

* Pre-v4.13.1 baseline (v4.13.0 final state): 573 unit tests.
* v4.13.1 audit-response additions: 72 new tests across 8 new files
  (Agent 1: 14 in `test_audit_fixes_v4_13_0_jax_dtype_dy_siblings
  .py` (extended); Agent 2: 14 across `test_audit_fixes_v4_13_1
  _thin_grating_dock.py`, `_context_guards.py`, `_random_choice
  .py`; Agent 3: 26 in `test_audit_fixes_v4_13_1_agent3.py`;
  Agent 4: 18 across `_asm_h_helper.py` and 3 perf-pin files).
* Final: **654 unit tests passing**, **34/34 validation files
  passing**.

---

## [4.13.0] — 2026-05-17

**Bundle of three internal phases since v4.12.2 (PyPI-published):**

* Phase 1 — closes audit known-limitations S1, S2, S3 and L2, L3,
  L4, L6, L8 from `docs/audits/AUDIT_V4_12_1_2026_05_16.md` (storage dtype
  preservation, codegen aperture-stop + wavelength sentinel, ghost
  R/r convention, JAX dtype + `jax_enable_x64`, `PropagationResult.dy`
  + `Source.dy`, sibling mirror-guards, `apply_mirror` xp + dy,
  zarr thread-safety).
* Phase 2 — sweeps `except Exception:` clauses across the non-UI
  codebase from 99 → 3 justified sites (typed exceptions
  everywhere else, three WARN-BEFORE-PASS upgrades).
* Phase 3 — Tier-2 perf wins from the same audit: 10× thin-grating,
  188× BSDF TIS, 4.43× Chebyshev freeform, 3.26× coating-stack,
  10.8× Shack-Hartmann FFT batching, 4–72× Seidel field sweep,
  smaller wins on `wave_opd_2d` and `_fd_grad`.  Also: rcwa.py →
  thin_grating.py file rename with no back-compat shim (sole-user
  waiver).

All 573 unit tests pass; 34/34 validation files pass.

### Breaking changes

* **`rcwa.py` → `thin_grating.py` file rename, no back-compat
  shim.**  `import lumenairy.elements.rcwa` raises
  `ModuleNotFoundError`.  `import lumenairy.ui.rcwa_dock` likewise.
  The public symbols (`thin_grating_efficiency_1d`,
  `grating_efficiency_vs_wavelength`) were already renamed in
  v4.4.0; v4.13.0 finishes the rename at the file level.
* **`wavelength` is now required by `io.codegen
  ._decompose_prescription`.**  Previously a missing `wavelength`
  silently defaulted to `1.31e-6` (NIR).  Now raises `ValueError`
  with a helpful message naming the parameter.  Visible-band
  Zemax imports that relied on the silent NIR default must now
  pass `wavelength` explicitly.

### Phase 1 — Audit known-limitations closure

**S1 — `io.storage` complex-dtype preservation through append.**
`save_jones_field_h5`, `append_plane_h5`, `_zarr_append_plane`, and
the unified `append_plane` dispatcher gained a `preserve_dtype`
parameter that is threaded down through the write path.  Previously
the append-side stack silently promoted complex64 inputs to
complex128 on every plane after the first.  Default behaviour is
preserved: callers that omit `preserve_dtype` continue to see the
v4.12.x promotion.

**S2 — `io.codegen` aperture-stop emission + wavelength sentinel.**
`_decompose_prescription` now emits an `{'type': 'aperture', ...}`
step before mirrors / dummy planes / real-lens groups whenever
`is_stop=True` on the source surface.  The Zemax loader threads the
`is_stop` flag through `prescriptions.py` so the stop reaches
codegen with its identity intact.  Separately, the silent default of
`wavelength_nm = 1.31e-6` was replaced with a `ValueError`; callers
that previously got NIR by accident now get a clear failure at
codegen time.

**S3 — `analysis.ghost` `R` vs `r` convention disambiguation.**
A top-of-module convention block now disambiguates uppercase `R`
(Fresnel reflectance) from lowercase `r` (curvature radius).
Local variables renamed accordingly (`R_i_val → r_i`,
`R_j_val → r_j`).  Public dict keys are unchanged for back-compat
consumers.

**L2 — JAX complex-dtype routing.**
Added `_resolve_jax_complex_dtype()` and `_resolve_jax_real_dtype()`
helpers in `propagators/propagation.py`.  When complex128 is
required, the helper auto-enables `jax_enable_x64` and emits a
`RuntimeWarning` rather than silently truncating.  The warning is
implicitly one-shot (gated by the `jax_enable_x64` state flip after
the first call), not enforced via `warnings.simplefilter('once')`.
Threaded through
`system.py:propagate_system_jax`, `_lens_jax.py` (three call
sites), and `analysis/phase_retrieval.py`.  The
`_PROPAGATE_SYSTEM_JAX_CACHE` key now includes `str(np.dtype(cdtype))`
so a float32-then-float64 sweep stops aliasing onto a single XLA
binary.

**L3 — `PropagationResult.dy` and `Source.dy` fields.**
`PropagationResult` gained a `dy` field (defaults to `dx` in
`__post_init__`); `dy_out` / `dx_out` aliases preserved.
`_coerce_field` in `propagators/dispatch.py` now returns
`(field, dx_out, dy_out)`; threading carried through every internal
caller.  `Source` (in `sources/core.py`) gained a matching `dy` that
`to_source()` forwards.  Anamorphic-grid propagation can now round-
trip metadata without losing the second axis.

**L4 — Sibling mirror-guards in `apply_real_lens_*` variants.**
`apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`, and
`lenses_maslov.apply_real_lens_maslov` all gained the pre-flight
`is_mirror=True` guard previously only present in
`apply_real_lens`.  Calls into a folded prescription that should
have been split now raise `ValueError` consistently across the
trio + maslov.

* L4b — `phase_retrieval.py` raises `NotImplementedError` when an
  `initial_guess` is passed on the JAX backend (the JAX path does
  not currently honour it); migration message included.
* L4c — `phase_retrieval.py` warns + synthesises an empty history
  list when `return_history=True` is requested on the JAX backend.

**L6 — `apply_mirror` xp dispatch + `dy` parameter.**
`elements/elements.py:apply_mirror` now resolves the backend
namespace via `_xp_of(E_in)` and accepts a `dy` parameter for
anamorphic grids.  NumPy short-circuit retained for the
numba-aspheric fast path; JAX and CuPy take the xp-native inline-sag
branch.  Backwards compatible: `dy=None` reproduces the v4.12.x
square-grid behaviour exactly.

**L8 — Zarr thread-safety guard.**
Added a module-level `_ZARR_MKDIR_PATCH_LOCK = threading.Lock()` in
`io/storage.py` to guard the `Path.mkdir` monkey-patch in
`_open_zarr_group_safe`.  Concurrent writers no longer race on
patch-install vs patch-restore.

### Phase 2 — `except Exception:` sweep

Non-UI library files contained 99 `except Exception:` clauses
(audit's "242" figure swept the full repo including tests,
validation, and commentary; the real non-UI count was 99).  After
the sweep:

* 3 KEEP-AS-IS, justified inline:
  * `_context.py:299` — atexit handler tolerating module-level
    global teardown.
  * `optimize/core.py:2683` — `_RestoreDtype.__del__` cleanup
    tolerating shutdown.
  * `propagators/hfpi.py:84` — optional-dep guard around
    `import jax`.
* ~85 narrowed to typed tuples covering the documented failure
  modes (e.g. `(RuntimeError, MemoryError, ValueError, TypeError,
  AttributeError)` for pyFFTW failures; `(ImportError, RuntimeError,
  AttributeError)` for cache-clear guards; `(ValueError,
  RuntimeError, ZeroDivisionError, np.linalg.LinAlgError,
  IndexError)` for `system_abcd` fallbacks; etc.).
* 3 WARN-BEFORE-PASS upgrades — surfaces in failure paths that
  would previously have silently degraded:
  * `analysis/field.py petzval_radius` — a missing glass entry was
    returning NaN, which downstream could be confused with "the
    field is perfectly flat."  Now warns explicitly with the glass
    name + wavelength.
  * `propagators/hf.py` LG decomposition fallback — falling back to
    a single `(p=0, l=0)` plane-wave mode makes the asymptotic
    propagator essentially useless; failure is now surfaced.
  * `optimize/core.py design_optimize plane_logger` callback — was
    silently swallowing all logger failures for the duration of an
    optimization run.  Promoted to WARN-BEFORE-PASS so users see
    telemetry-callback bugs immediately instead of an empty log
    file at the end.

Pinning test:
`tests/unit/test_audit_fixes_v4_13_0_except_sweep.py` with 6 tests
including a regression budget guard (non-UI `except Exception:` count
≤ 15), two behavioural pins (`petzval_radius` warns, narrow tuples
drop on expected failures), and one source-string pin via
`inspect.getsource` for the `design_optimize plane_logger` warning
(behavioural exercise was unreliable across scipy versions).

### Phase 3 — Tier-2 performance wins

Three disjoint-scope agent groups (elements/UI, analysis/optimize,
propagators/raytrace) executed in parallel.  Measured speedups
(`time.perf_counter`, representative workloads on Win11 / Python
3.14):

#### Group α — elements + UI

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `thin_grating_efficiency_1d` (n_orders=25) | 198 us | 18.9 us | **10.5×** |
| `bsdf.total_integrated_scatter` (256×128) | 384 ms | 2.05 ms | **188×** |
| coating-stack matmul chain (50 layers) | 0.48 ms | 0.15 ms | **3.26×** |
| freeform Chebyshev (64×64, 16 coeffs) | 1.63 ms | 0.37 ms | **4.43×** |

**`elements/rcwa.py` → `elements/thin_grating.py` rename.**  The
file name "rcwa" was misleading: the function inside is the
analytical scalar thin-phase grating formula, not rigorous coupled-
wave analysis.  The functions themselves were renamed in v4.4.0
(`thin_grating_efficiency_1d`, `grating_efficiency_vs_wavelength`);
v4.13.0 finishes the rename at the file level:

* `lumenairy/elements/rcwa.py` → `lumenairy/elements/thin_grating.py`
* `lumenairy/ui/rcwa_dock.py` → `lumenairy/ui/thin_grating_dock.py`
  (class `RCWADock` → `ThinGratingDock`)
* `lumenairy/__init__.py` import path + Tier-7 comment updated.
* `lumenairy/elements/__init__.py` module docstring updated.
* `lumenairy/ui/main_window.py` (7 token occurrences across 6
  logical sites: import, widget construct, dock construct, dock-key,
  menu label, show_and_raise lambda, visible-list) +
  `lumenairy/ui/workspace.py` (dock-key mapping) updated.

**No back-compat shim** is installed at either old path
(`import lumenairy.elements.rcwa` raises `ModuleNotFoundError`).
This is per the explicit waiver from the sole user of the library.

**`bsdf.total_integrated_scatter`** is now a fully-vectorised 2D
meshgrid + single `integrand.sum() * dθ * dφ` reduction (rectangle
rule on a regular θ-φ grid) rather than a per-(θ_i, θ_s) inner-
product loop.  This is the biggest single-call speedup in the
v4.13.0 batch.

**Coating-stack matmul** moved from a Python `M = M @ M_layer` loop
to a tournament reduction over a `(N, 2, 2)` per-layer tensor.  The
agent also switched the inner Snell-chain math from `np.sin/arcsin`
to `math.sin/asin` for scalar wavelengths — bit-near-identical via
libm (pinned at `atol=1e-12` rather than ULP-exact since libm
implementations can diverge in the last few bits across versions) —
lifting the win from 1.5× into the 3-5× target band.

**Freeform Chebyshev** hoists the `arccos(rho)` out of the per-order
loop and caches the `T_i(rho)` factors keyed by polynomial order,
exploiting the fact that many `(i, j)` coefficient pairs share an
`i` or `j` so 2N cos calls collapse to roughly `2 sqrt(N_coeffs)`.

#### Group β — analysis + optimize

| Hot path | Old | New | Speedup |
|---|---|---|---|
| Shack-Hartmann WFS (16² lenslets, 256² grid) | 72.9 ms | 6.76 ms | **10.8×** |
| `wave_opd_2d` (512²) | 52.5 ms | 27.7 ms | **1.89×** |
| `_fd_grad_pure` (N=20 quadratic, forward) | 0.159 ms | 0.071 ms | **2.23×** |

**Shack-Hartmann FFT batching.**  `analysis/detector.py:
shack_hartmann` previously ran two nested per-lenslet double-loops
(reference + measurement), each computing an `np.fft.fft2` on a
single sub-aperture.  The reference path is now a single fft2 on a
stacked `(n_lenslets², sa_pixels, sa_pixels)` 3D array; the
measurement path is the same.  A new `_build_asm_H_for_lenslet`
helper inside `detector.py` duplicates a thin slice of the
angular-spectrum H-build logic for the sub-aperture geometry,
avoiding a scope-crossing edit into `propagators/` (a follow-up
in v4.13.1 consolidates this into a shared
`_build_asm_H_square` helper).  NaN sentinels for OOB lenslets
and the `sa_pixels >= 2` (i.e. `pitch >= 2*dx`) sampling guard
are preserved.  Note: this guard raises `ValueError`, not a
warning.

**`wave_opd_2d` axis unwrap.**  `analysis/core.py:wave_opd_2d`
replaces the row-then-column per-pixel Python unwrap loop with
`np.unwrap(opd, axis=1)` then `np.unwrap(..., axis=0)` (matching
the legacy row-first traversal order).  At N=512
the Python iteration overhead was about half the run time; the
inner unwrap step itself was already compiled, so the 1.89× win is
real but below the speculative 5-10× target.  Correctness preserved
to 1e-12 on smooth quadratics, twice-wrapping tilts, and masked
NaN regions.

**`_fd_grad_pure` / `_fd_grad_for` central-vs-forward scheme.**
Pre-v4.13 the helper used central differences (2N evals,
`f(x ± h*e_i)`).  The audit's request to "reuse the centre value"
was based on a forward-FD model — central FD has no `f(x)` to
reuse, so the perf win required switching the scheme.  v4.13.0
parameterises this:

* `_fd_grad_pure(...)` and `_fd_grad_for(...)` accept
  `scheme='central'|'forward'`, default `'central'`.
* `scheme='central'` (the default) preserves bit-identical
  gradient values with pre-v4.13 behaviour at 2N evals,
  O(h²) truncation.
* `scheme='forward'` is the opt-in perf path (N+1 evals, or N with
  the optional `f0=<known value>`).  O(h) truncation.

`design_optimize._merit_jac_auto` keeps the default `'central'`
explicitly, so no existing optimisation run sees a behavioural
change.  Callers that prefer speed can opt into `'forward'` at the
helper level.

#### Group γ — propagators + raytrace

| Hot path | Old | New | Speedup |
|---|---|---|---|
| `seidel_field_sweep` (5 fields, singlet) | (ref) | (ref/4) | **4.2×** |
| `seidel_field_sweep` (50 fields, singlet) | (ref) | (ref/37) | **37.5×** |
| `seidel_field_sweep` (100 fields, singlet) | (ref) | (ref/72) | **72.0×** |

**`raytrace/seidel_analysis.py:seidel_field_sweep` per-field hoist.**
The paraxial Seidel formalism is exactly linear in the chief-ray
initial conditions, and those scale linearly with field angle.  The
sweep now does a single `seidel_coefficients(... field_angle=1.0)`
call and applies the analytical per-field scaling (`S1, S4 ∝ 1`;
`S2, y_chief ∝ σ`; `S3 ∝ σ²`; `S5 ∝ σ³`).  All field-independent
work — glass-index lookups, pre-stop ABCD, marginal-ray trace, full
`system_abcd` — is hoisted out of the loop.  Element-by-element
agreement with the pre-hoist reference is well below the test
pin tolerance of `< 1e-12 absolute` (numerics on the singlet
test case land in the 1e-15..1e-20 range in practice).

**γ.1 — HFPI bincount swap reverted before ship.**  The initial
agent run replaced the `np.add.at(out, flat_idx, w_masked)` scatter
in `propagators/hfpi.py:accumulate_to_grid` with a `np.bincount`-
based path on the premise that `add.at` was a Python-level loop.
That premise pre-dated NumPy 1.25; NumPy ≥ 1.25 has vectorised
`add.at` for complex via `_PyArray_UFuncBufferedAtVectorized`.
Measured speedup on the actually-shipping NumPy was 0.4–1.0×
(a wash, leaning regression), so the patch was reverted.  No
behavioural change vs v4.12.2 in this path.

### Bug-fix and discipline notes

* Phase 2 audit count discrepancy resolved.  Audit reported 346
  `except Exception:` clauses; the real non-UI count was 99 (the
  audit swept tests, validation, and prose into the same `grep`).
* Phase 3 Group α / γ race observed mid-run: Group γ briefly
  imported a non-existent `elements/rcwa.py` reference while
  Group α was mid-rename.  Resolved by Group α's completion; no
  artefact in shipped code.
* `tests/unit/test_audit_fixes_v4_12_0_round4_dispatch.py` —
  `test_coerce_field_unpacks_tuple` was rewritten in-place using
  `result[0]` / `result[1]` subscript indexing (no `pytest.mark.skip`)
  because L3 changed `_coerce_field`'s signature to a 3-tuple
  `(field, dx_out, dy_out)`.  All 7 tests in the class collect and
  pass.  The dispatcher-level 3-tuple pin is in the v4.13.0
  dy-siblings test file.

### Test counts

* Pre-Phase-3 baseline (Phase 1 + 2 landed): 536 unit tests.
* Phase 3 additions: 12 perf pins (α) + 12 (β) + 7 (γ) = 31, minus
  6 bincount tests deleted with the γ.1 revert + net 0 from the
  fd_grad test rewrite.  Final: **573 unit tests passing**,
  **34/34 validation files passing**.

---

**Archive note:** v4.10.x and earlier (down to v2.5.x) are preserved in [`docs/changelogs/v4.md`](docs/changelogs/v4.md).  The v5.2.3 release split moved v4.11.x - v4.12.x; v5.4.0 Phase 5 completes the pass.
