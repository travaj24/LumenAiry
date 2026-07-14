# Out-of-Plane Generator — Missing Factor-i on the Off-Plane Blocks — 2026-07-14

> **STATUS — FIXED (2026-07-14, branch `fix/loose-ends-2026-07`).** All four
> `Delta`/generator copies corrected in lockstep
> (`rcwa/_core._layer_eigenmodes_tensor` A/B blocks,
> `_berreman_jax._delta_jax`, `berreman._berreman_delta`,
> `tests/unit/_berreman4x4._berreman_delta`); independent anchor gates in
> `tests/unit/test_audit_oop_dispersion.py` (8).

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

## 6. Post-fix sweep + NEW OPEN ITEM

Post-fix suite sweep: 136/137 across all OOP-affected suites (the fixed
solver and fixed oracle move consistently, so the singular-value / R/T pins
survive).  The one failure exposed a **new open finding**: on the y-uniform
patterned OOP grating (`test_v5_14_0_pmm2d_oop.py`), the fixed
`pmm_jones_2d` converges to the dispersion-anchored `rcwa_jones_1d_segments`
(worst per-order dT 9e-4 at n_orders=9), but **`pmm_jones_1d`'s independent
metric-generator OOP path now sits ~4.5e-2 from both anchored engines at
`m = +/-1` in the OOP-coupled polarization** (energy exactly 1 —
energy-blind).  Its historical "~1e-3 algebraic floor" was measured against
the pre-fix (wrong) shared-generator reference, so the metric generator's
OOP channel needs its own audit — OPEN, PMM-side.  The test's oracle is
repointed to the anchored rcwa solver; the pmm-1D leg is kept as a loose
documented cross-check.

*Found by the 2026-07-14 loose-ends sweep (bidirectional-adversarial follow-up
to the consumer-API C2/A3 ports). The C2 flux-based `layer_absorption` shipped
in v5.21.5 was itself computed from boundary-exact tangential fields and its
budget closed, but its per-layer split carried the same underlying mode error
at finite theta; corrected values ship with this fix.*
