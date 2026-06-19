# PMM roadmap — axisymmetric / Body-of-Revolution (BOR) extension (2026-06-19)

Implementation plan for rotationally-symmetric structures in the PMM (and, deferred,
RCWA) solvers: circular/annular gratings, axisymmetric out-couplers and metalenses,
ring resonators, VCSEL-like and fiber-grating stacks — the class of devices where
`eps = eps(r, z)` (no azimuthal dependence). Companion to `pmm_roadmap_v5_14.md`.

## 1. Why this is worth doing, and why PMM

A continuously axisymmetric 3-D scattering problem **separates by azimuthal order**:
expand the fields in `e^{i m phi}` and, because the structure has no `phi`-dependence,
the harmonics decouple into INDEPENDENT 1-D RADIAL problems, one per integer `m`
(Body-of-Revolution / BOR). The payoff:

- A **normal-incidence plane wave on an axisymmetric structure couples only to
  m = +/-1** (a constant transverse field is `cos phi`/`sin phi` in the `(r, phi)`
  frame). So the headline use case is ONE radial solve — orders of magnitude cheaper
  than full 3-D FEM/FDTD for the same device.
- Off-axis / structured inputs (Bessel beams, vortices, focused spots) map to a small,
  known set of `m` — still a handful of cheap radial solves.

PMM is the right vehicle (not RCWA — see Section 8): its in-plane discretization is
already **subsectional GLL spectral elements** (Edee 2011; `oned.py`,
`pmm_jones_1d`/`pmm_efficiency_1d`), so going cylindrical is mostly *swapping the
in-plane operator* and *changing the outer boundary*. Material discontinuities at ring
walls are absorbed by element integration — we avoid the Bessel-convolution
factorization that makes cylindrical RCWA awkward.

## 2. What is reused vs. new (grounded in the current code)

The PMM cleanly separates the per-layer in-plane eigensolve from the z-cascade:

| component | file / symbol | BOR status |
|-----------|---------------|------------|
| layer modal eigenproblem (div-conforming metric generator) | `_core.py::_layer_modes_metric` | **REPLACE** with a radial operator `_layer_modes_radial(m, ...)` |
| uniform half-space modes | `_core.py::_uniform_geo_eig` | **REPLACE** with `_uniform_radial_modes(m, ...)` (Bessel `J_m`; outgoing Hankel `H_m^(1)` for the open superstrate) |
| interface / propagation S-matrices, Redheffer star | `_interface_smatrix`, `_propagation_smatrix`, `_redheffer_star` (shared with `..rcwa`) | **REUSE** — coordinate-agnostic tangential mode-matching; only the mode overlap/normalization is supplied by the radial basis |
| layered stack, source, sweep, prepared eig-cache | `stack.py::PMMStack`, `set_source`, `solve`, `prepare`, `solve_vs_wavelength`, `_PreparedPMMStack` | **EXTEND** with a `coords` / `m` path; cascade, RESUME, dispersion sweeps carry over |
| internal field reconstruction | `stack.py::internal_field` | **EXTEND** (radial basis evaluation; same z-handling) |

So the irreducible new work is the **radial layer operator**, the **open outer
boundary**, the **r=0 axis treatment**, and the **cylindrical far-field** — the
expensive, well-tested cascade/sweep/prepared machinery is inherited.

## 3. Physics formulation (the radial operator)

Per azimuthal order `m`, in `(r, z)` with the `e^{i m phi}` ansatz, Maxwell's curl
operators pick up the **cylindrical metric**: the transverse derivatives carry `1/r`
factors and a **centrifugal `m^2 / r^2` term**. Concretely, the in-plane operator that
`_layer_modes_metric` builds for `d/dx` is replaced by the radial operator built from
`(1/r) d/dr (r .)` and `m/r` couplings, with the div-conforming continuity now imposed
on the **radial-normal field `E_r`** at ring walls (the cylindrical analog of the
Cartesian normal-E inverse rule the metric generator already enforces).

- **Uniform region (a ring of constant eps):** radial eigenfunctions are `J_m(gamma r)`
  and `Y_m(gamma r)` with `gamma^2 = eps k0^2 - q^2` (q the z-propagation constant) —
  the cylindrical counterpart of the plane-wave `_uniform_geo_eig`.
- **Central region (contains r=0):** regularity admits only `J_m` (`Y_m`, `H_m`
  singular at the axis). The element basis on `[0, r1]` must enforce the correct
  `r^|m|` behaviour at the axis.
- **Open superstrate/substrate (radiation):** outgoing cylindrical waves
  `H_m^(1)(gamma r)` — used both for the far-field and (Section 5) the outer boundary.

## 4. Phased implementation plan

Each phase is independently testable; effort/value in the audit's style.

**Phase 0 — standalone prototype + analytic anchor (value: HIGH, effort: S).**
Outside the package (`experiments/` or a scratch module): m=+/-1, single uniform
half-space + one concentric ring, normal incidence. Validate reflection against the
**exact analytic Bessel/Mie-style solution** for a layered radial structure. Goal:
de-risk the radial operator + axis BC before touching the library. GATE: prototype
matches analytic to <1e-6 before Phase 1.

**Phase 1 — radial layer eigensolve (value: HIGH, effort: M).**
`_core.py::_layer_modes_radial(m, regions, k0, ...)`: GLL spectral elements in r over
the concentric homogeneous regions; cylindrical metric (`1/r`, `m^2/r^2`); axis
regularity; `E_r`-normal continuity at ring walls. `_uniform_radial_modes(m, ...)` for
uniform rings (Bessel) and the half-spaces (Hankel outgoing). Unit-test eigenvalues vs
known Bessel roots in a uniform cylinder.

**Phase 2 — open outer boundary (value: HIGH, effort: M-L; THE hard part).**
Cartesian PMM *wraps* (Bloch-periodic); the radial domain is OPEN. Implement a
**radial PML** (complex coordinate stretch `r -> r(1 + i sigma(r))` beyond a cap radius)
as the primary path, with **analytic Hankel matching** at `r = R` as the cross-check
boundary. This is the dominant new-physics risk (spurious reflections / convergence vs
PML strength + cap radius); budget the most validation here.

**Phase 3 — cascade integration (value: HIGH, effort: S-M).**
Feed the radial modes into the existing `_interface_smatrix`/`_propagation_smatrix`/
`_redheffer_star`. Only the mode inner-product/normalization (the `r dr` radial measure
and the `E_phi, E_z, H_phi, H_z` tangential matching) is new; the Redheffer recursion is
unchanged. Per-`m` solve is independent -> trivial loop; sum the m-contributions for the
physical field.

**Phase 4 — public API + geometry + far-field (value: HIGH, effort: M).**
- `PMMStack(coords="cylindrical")` (or a thin `BORStack`), with `set_source` taking the
  azimuthal content (auto m=+/-1 for a normal plane wave; explicit `m=` for beams).
- Ring geometry builder: concentric `(r_inner, r_outer, material)` annuli per layer —
  the cylindrical analog of `SegmentStackGeometry` (reuse the conformal `coat`/`fill`
   idioms; the exp10 Al2O3/Si3N4 conformal coats map to radial shells).
- Cylindrical far-field: asymptotic `H_m^(1)` -> angular `(theta)` distribution +
  total radiated/coupled power. `internal_field` in `(r, z)`.

**Phase 5 — validation suite (value: HIGH, effort: M).** Oracles, in the codebase's
bidirectional-adversarial spirit:
1. **Analytic** layered-cylinder / concentric-ring reflection (exact Bessel).
2. **FDTD-BOR** (DynaMeta's FDTD engine hosts the axisymmetric reduction) on a circular
   grating — independent full-wave cross-check.
3. **Energy conservation** per m (lossless -> sum R+T = 1) and **reciprocity**.
4. **Cartesian limit:** a large-radius annular grating locally approaches a linear
   grating -> compare a far-from-axis patch to the existing 1-D PMM.

**Phase 6 — multi-m, structured sources, docs (value: MED, effort: S-M).**
Bessel/vortex/focused inputs (their `m`-spectra), a cookbook entry, and a
`prepare()`-style eig-cache across wavelengths (the radial layer eig is k0/eps-shareable
for uniform rings, mirroring the v5.14 homogeneous-region eig-share item).

## 5. The three things that are genuinely new (risk register)

1. **Radial PML / open boundary** (Phase 2) — the #1 risk. Cartesian periodicity gave a
   closed basis for free; radiation does not. Mitigation: validate PML against analytic
   Hankel matching on a bare half-space before any structured run.
2. **r = 0 axis singularity** — the `1/r`, `m^2/r^2` terms and `r^|m|` regularity. A
   one-sided element with the axis BC baked in; unit-test against Bessel-root spectra.
3. **Radial-normal inverse rule** — `E_r` continuity at ring walls. The div-conforming
   metric generator already does Cartesian normal-E; this is its cylindrical re-derivation
   (element integration handles the material jump, the PMM advantage over Fourier).

## 6. API sketch (target)

```python
from lumenairy import BORStack          # or PMMStack(coords="cylindrical")
st = BORStack(r_max=..., pml=...)        # open radial domain + PML
st.add_layer(thk, rings=[(0, r1, "Cu"), (r1, r1+al, "Al2O3"), ...])  # concentric
st.set_source(wavelength, m="auto")      # auto -> +/-1 for normal plane wave
res = st.solve()                         # res.R, res.T, res.orders(theta), res.jones
F = st.internal_field(z, component="E")  # (r, z) field
```

## 7. Effort summary

Phases 0-5 are the MVP (axisymmetric PMM, normal-incidence, validated). Realistically a
contained module (~the size of `stack2d.py`), reusing the cascade/sweep/prepared
infrastructure. Phase 2 (open boundary) dominates the risk; Phases 1/3/4 are mostly
re-derivation + plumbing into proven machinery.

## 8. RCWA-cylindrical — DEFERRED (oracle-only, if ever)

Not recommended as a co-equal feature:
- The cylindrical factorization (the Bessel analog of Li's inverse rules) is awkward —
  radial overlaps are not clean convolutions — and converges WORSE for exactly the lossy-
  metal / high-contrast cases that already strain Cartesian RCWA (cf. the v5.14 metal-TM
  study where PMM converges and Fourier struggles).
- Only justified later as a **low-order independent cross-check oracle** for the BOR-PMM,
  never as the production solver.

## 9. Explicitly out of scope

- **Discrete C_N symmetry** (N-fold posts, not continuous): BOR does NOT reduce it to 1-D
  — it only shrinks the domain to a `2*pi/N` wedge with azimuthal Bloch boundaries, still
  a 2-D/3-D solve. That belongs in the (separate) 2-D PMM track, not here.
- **Finite-aperture / edge effects** of an otherwise-uniform device — those are a
  near-to-far-field propagation overlay on the existing solver, not a new modal method.

## 10. References

- Edee, *Modal method based on subsectional Gegenbauer polynomial expansion* (the PMM
  in-plane basis this extends).
- Body-of-Revolution methods (azimuthal `e^{i m phi}` decomposition); cylindrical /
  Fourier-Bessel RCWA and aperiodic FMM in cylindrical coordinates (circular gratings,
  axisymmetric diffractive optics, VCSELs) — survey before Phase 1.
- FDTD-BOR (axisymmetric FDTD) for the Phase 5 full-wave oracle (DynaMeta FDTD engine).
- Hankel-function radiation conditions / radial PML (Phase 2 boundary).
