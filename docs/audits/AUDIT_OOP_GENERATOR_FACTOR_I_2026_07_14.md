# Out-of-Plane Generator — Missing Factor-i on the Off-Plane Blocks — 2026-07-14

> **STATUS — FIXED (2026-07-14, branch `fix/loose-ends-2026-07`).** All SIX
> generator copies corrected in lockstep
> (`rcwa/_core._layer_eigenmodes_tensor` A/B blocks,
> `_berreman_jax._delta_jax`, `berreman._berreman_delta`,
> `tests/unit/_berreman4x4._berreman_delta`, and — §6 — PMM's
> `_build_generator_metric` + `_cov_generator_4n` cross blocks); independent
> anchor gates in `tests/unit/test_audit_oop_dispersion.py` (11).  One
> documented residual limitation: the covariant layout's discontinuous
> off-plane TM channel (§6; `'auto'` reroutes it to convection).

**Severity: correctness (silent wrong numbers), multi-release.** Since the
full-3x3 generator shipped (v5.11.0 GAP7 for `rcwa_jones_1d`, extended through
v5.14.x to `RCWAStack`, `rcwa_jones_2d` GAP2, the PMM hybrid tensor paths, and
the v5.20.1 Berreman OOP-oblique route), every solve of an **out-of-plane
tensor (`eps_xz/yz/zx/zy != 0`) at OBLIQUE incidence** used a layer generator
whose off-plane cross-blocks were missing relative factors of `-/+i`.

## 1. The defect

The generalized layer ODE `d[E; u]/dz' = G [E; u]` is assembled as
`G = [[A, P], [Q, B]]`. In the modal-`u` state convention that `P`/`Q` are
written in (`H_phys = -i u` — the convention shared by the interface
matching, flux bookkeeping, and internal-field consumers), a row-by-row
derivation of the Maxwell curl equations gives the off-plane cross-blocks as

    A = -i * [[Kx Ez^-1 EZX, Kx Ez^-1 EZY], [Ky Ez^-1 EZX, Ky Ez^-1 EZY]]
    B = -i * [[EYZ Ez^-1 Ky, -EYZ Ez^-1 Kx], [-EXZ Ez^-1 Ky, EXZ Ez^-1 Kx]]

The shipped blocks had **real coefficients** (the same magnitudes, no `i`).
`A = B = 0` for in-plane tensors and at normal incidence, so ONLY
out-of-plane-at-oblique was affected — every other regime stayed exact.

## 2. Physical consequences

Measured on the tilted-35° absorbing uniaxial probe at `theta = 0.45`,
`phi = 0.6`, `wl = 1.55 um`:

- **Wrong extraordinary dispersion**: `eig(G)` gave an artificially
  `+/-`-SYMMETRIC e-pair `kz_e/k0 = +/-1.5646`, vs the exact
  `det(k x (k x .) + eps) = 0` roots `{-1.5214, +1.6090}` — a 3–5%
  propagation-constant error inside the layer (the o-pair was exact, which
  is why ordinary-dominated observables looked fine).
- **Non-Maxwellian mode fields**: the eigenvectors failed the curl rows at
  1e-2..5e-2 under EVERY constant re-scaling (exhaustive convention scan over
  H-scale `{±1, ±i}` × carrier sign × gauge × dz-sign × curl sign).
- **Internal-field pathology**: fields inside OOP layers at oblique violated
  the local Poynting theorem by ~7% (`C/k0 = 1.072 / 0.953` per layer instead
  of 1), making the density-based and flux-difference absorption attributions
  disagree at 3e-3 while every energy BUDGET still closed (the flux sum
  telescopes — the lossless-trap rule in action).
- Far-field `R/T/Jones` for OOP-oblique stacks carried corresponding phase /
  amplitude errors (layer-phase error `~ k0 * dkz * d`, e.g. ~0.1–0.2 rad for
  a 400 nm layer).

## 3. Why five releases of gates never caught it

The `_berreman4x4` **test oracle shared the defect**: it was ported from the
same prototype (`gap7_proto_2.py`) that seeded the solver blocks, so every
"agrees to 1e-10 with the independent 4x4 oracle" validation was **circular**.
Its own anchors (analytic Airy transmission, lossless energy closure) are
blind here: Airy is isotropic (`A = B = 0`) and energy closure is insensitive
(budgets telescope). The v5.20.1 Berreman OOP-oblique route was then validated
against `RCWAStack`/`rcwa_jones_1d` — the same generator again. Rotation
covariance (the audit-F1 conical fix) is also insensitive: the wrong blocks
transform covariantly too.

**Discovery chain (loose-ends round):** porting the C2 retention to
`RCWAStack` → density- vs flux-based absorption splits disagreed at 3e-3 on
machine-identical fields → local Poynting residual isolated OOP layers at
oblique (`C != k0`) → per-mode Maxwell residuals + exhaustive convention scan
ruled out every re-scaling → block-by-block comparison against a fresh
derivation → **exact-dispersion arbitration** (`det`-condition roots vs
`eig(G)`) proved the fix uniquely (`(iA, -iB)` reproduces the exact roots to
1e-14 in both gauges; the alternative `(-iA, +iB)` gives the k-mirrored,
wrong set).

## 4. The fix + new independent anchors

Factors applied in all four copies (solver numpy/jax, native Berreman Delta,
test oracle). New gates (`tests/unit/test_audit_oop_dispersion.py`):

1. `eig(G)` == exact det-condition `kz` roots (1e-14; three geometries), with
   the root-pair-symmetry count pinning the asymmetric e-pair (legacy = 4
   paired roots, exact = 2);
2. per-mode Maxwell residuals < 1e-12 on all six curl rows (public gauge);
3. local Poynting theorem inside OOP layers at oblique (`C/k0 = 1 +- 2e-3`);
4. density- vs flux-based per-layer absorption attribution agreement
   (RCWAStack vs BerremanStack, 1e-4; was 3.4e-3);
5. the fixed oracle itself is pinned to the exact dispersion (no more
   circularity).

Post-fix cross-machinery results: RCWA-vs-Berreman internal fields 1e-15 on
all six components; absorption splits 2.6e-6; C2 budget/lossless/continuity
gates unchanged-green.

## 5. Affected surfaces (all inherit the corrected generator)

`rcwa_jones_1d` full-tensor OOP-oblique; `RCWAStack` OOP layers;
`rcwa_jones_2d` (GAP2) OOP; PMM hybrid tensor paths routing through
`_tensor_layer_modes` -> `_layer_eigenmodes_tensor`; PMM jax twods;
`berreman` OOP-oblique route (incl. the jax twin) and the C2 retained
internals; `pmm_jones_1d_conical_tensor`'s "shared-generator OOP-at-conical
residual" (expected to shrink — the residual was previously attributed to the
PMM side). Existing tests that PINNED pre-fix OOP-oblique values are updated
alongside this fix with the exact-dispersion + Poynting + cross-attribution
anchors as the new ground truth.

## 6. Post-fix sweep + the PMM generators (RESOLVED 2026-07-14)

Post-fix suite sweep: 136/137 across all OOP-affected suites (the fixed
solver and fixed oracle move consistently, so the singular-value / R/T pins
survive).  The one failure exposed the same defect in PMM's own generators:
on the y-uniform patterned OOP grating (`test_v5_14_0_pmm2d_oop.py`), the
fixed `pmm_jones_2d` converges to the dispersion-anchored
`rcwa_jones_1d_segments` (worst per-order dT 9e-4 at n_orders=9), but
`pmm_jones_1d`'s independent metric-generator OOP path sat ~4.5e-2 from both
anchored engines at `m = +/-1` in the OOP-coupled polarization (energy
exactly 1 — energy-blind); its historical "~1e-3 algebraic floor" had been
measured against the pre-fix (wrong) shared-generator reference.

**RESOLUTION — both PMM 1-D generators carried the same missing-factor-i
defect in their off-plane cross blocks; both are now fixed and
dispersion-pinned** (`test_audit_oop_dispersion.py::test_pmm_*`).  The
anchor exploits the fact that for a UNIFORM medium every Galerkin
coefficient mass is exactly a scalar multiple of the unit mass, so each
generator is an exact matrix polynomial in `Dop` and `eig(L)` must land on
the exact det-condition roots at every alpha in the operator's own spectrum
`{kx0 - i*d : d in eig(Dop)}`:

- **metric generator** (`_build_generator_metric`, the `pmm_jones_1d`
  vertical-OOP production path): cross-block signs `(+i, +i, -i, +i)` on the
  legacy terms — the UNIQUE combo of all 256 per-block `{+-1, +-i}` choices
  that closes (1.9e-10 at `kx0 = 0.5 k0`; next-best 1.2e-2; at normal
  incidence it ties only with its exact global mirror, an actual spectral
  degeneracy that oblique breaks).  The y-uniform three-engine cross-check
  drops 4.5e-2 -> 8.7e-4 (test bar re-tightened 6e-2 -> 3e-3).
- **covariant generator** (`_cov_generator_4n`, the spectral slant path):
  cross-block signs `(+i, -i, +i, -i)` — again the UNIQUE combo of 256
  (full-spectrum 4e-12 with the modal Ez closure at slant 0/30/45 on generic
  AND symmetric tensors; 2e-10 on resolved alphas with the production
  div-conforming closure; next-best 1.6e-2).  The oblique-frame eigenvalue
  map is `beta = kz*k0*cos(phi) + alpha*sin(phi)` (calibrated on the
  validated in-plane path).  The cross blocks now also use the POINTWISE
  `[[exz/ezz]]`-style ratio composites (Li Eq.12 discipline) instead of the
  spectral product `[[exz]] @ inv([[ezz]])`.

**NEW DOCUMENTED LIMITATION (research item, not shipped-blocking): the
covariant LAYOUT's discontinuous off-plane TM channel.**  With the corrected
physics, a 3-way referee on the slanted binary OOP grating (slant 30/45 deg)
gives: convection-vs-RCWA-staircase **3.8e-3/3.9e-3** (two independent
corrected engines at the staircase-truncation floor; the staircase moves
1.9e-3 from n_orders 15->25 and 7e-6 from n_slabs 200->800), while the
covariant path is the outlier at **~0.10-0.12 in TM only** (TE is clean to
~1e-6-1e-3; battery: dTM 0.074-0.16 across exz/full/lossy/asym cells).
This matches the 2026-06-08 six-avenue study's unresolved "bare exz/ezx
sub-channel ~5e-2 floor" — now fully expressed because the everyone-wrong
world had all three engines agreeing on the same symmetrized wrong answer at
3e-3 (the old test bars were calibrated there).  The pointwise-composite
refinement did NOT close it (0.1018 vs 0.1019) — the defect is structural to
the covariant factorization at off-plane material discontinuities.
**Response: `'auto'` now routes slanted OOP cells/stacks to `'convection'`**
(`pmm_jones_1d_slanted`, `pmm_jones_1d_slanted_segments`, `PMMStack`);
explicit `'covariant'` still solves OOP with the limitation documented in
its docstrings, and the covariant OOP tests are regression-trackers (TE
clean / TM within the documented floor / energy) rather than convergence
claims.

*Found by the 2026-07-14 loose-ends sweep (bidirectional-adversarial follow-up
to the consumer-API C2/A3 ports). The C2 flux-based `layer_absorption` shipped
in v5.21.5 was itself computed from boundary-exact tangential fields and its
budget closed, but its per-layer split carried the same underlying mode error
at finite theta; corrected values ship with this fix.*
