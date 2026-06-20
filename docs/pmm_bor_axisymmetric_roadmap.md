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

---

# ASSESSMENT + GROUNDED FORMULATION + VALIDATED PHASE 0 (2026-06-19)

A critical review (expert EM critique + a literature-grounded formulation pass) and a
machine-precision-validated Phase-0 prototype. **Verdict: PROCEED WITH CHANGES.** The core
BOR decoupling and the PMM-over-cylindrical-RCWA choice are sound; three items must be
**promoted from footnotes to first-class, separately-gated deliverables**, and Phase 0 must
be pinned to a *coupled* anchor (not a scalar one) so a green gate cannot hide the hard part.

## A. What is already VALIDATED (Phase 0 / Milestone 1 — DONE)

Standalone prototype `experiments/bor_pmm/radial_eigensolver.py` (+ `test_*`, 10 green).
The radial spectral-element operator — the **cylindrical metric (`1/r`, `m^2/r^2`) and the
`r=0` axis singularity**, the roadmap's stated #2 risk — is validated to **machine
precision** against the exact Bessel spectrum:

- TM (Dirichlet, `psi(R)=0`): `gamma R = j_{m,n}` — rel err ~1e-13, m = 0..3.
- TE (Neumann, `psi'(R)=0`): `gamma R = j'_{m,n}` — rel err ~1e-13, m = 1,2.
- Eigen**functions** match `J_m(gamma r)` to ~1e-13 (not just eigenvalues).
- Spectral convergence confirmed (error ~25x/refinement).

**Axis recipe that works** (the crux, now pinned): assemble with **Gauss-Legendre**
quadrature (interior points → `1/r` never sampled at `r=0`) and **drop the `r=0` DOF for
`m != 0`** (imposing `psi(0)=0` enforces `r^|m|` regularity AND discards the one divergent
entry `A_00 ~ INT dr/r`; the retained basis vanishes at the axis so `(1/r) phi_i phi_j ~ r`
is integrable). This resolves the roadmap's #2 risk.

## B. Grounded formulation (the radial operator)

Convention: `exp(-i w t)`, `exp(i m phi)`, `exp(i q z)`, `k0 = w/c`, `mu = 1`, normalized
`h = sqrt(mu0/eps0) H` so `curl E = i k0 h`, `curl h = -i k0 eps E`. Transverse wavenumber
`gamma^2 = eps k0^2 - q^2`.

- **Longitudinal → transverse** (cylindrical waveguide relations):
  - `E_r   = (i/gamma^2)[ q dE_z/dr + (m k0/r) h_z ]`
  - `E_phi = (i/gamma^2)[ (i m q/r) E_z - k0 dh_z/dr ]`
  - `h_r   = (i/gamma^2)[ q dh_z/dr - (m k0 eps/r) E_z ]`
  - `h_phi = (i/gamma^2)[ (i m q/r) h_z + k0 eps dE_z/dr ]`
- **Uniform region:** `E_z, h_z` solve Bessel's eq order `m`; `J_m(gamma r)`, `Y_m(gamma r)`.
  Axis region keeps only `J_m` (~`r^|m|`); open half-spaces use outgoing `H_m^(1)`.
- **Modal operator** per m: `L_m R = q^2 R`, `L_m = d^2/dr^2 + (1/r) d/dr - m^2/r^2 + eps(r) k0^2`.

## C. The THREE risks promoted to first-class (the critique's core correction)

1. **The `m != 0` radial system is COUPLED, not scalar.** TE-like (`h_z`) and TM-like
   (`E_z`) couple through the `m/r` off-diagonal terms at any **radial material interface**
   (only `m=0` splits into independent TE0/TM0; a *homogeneous* region also decouples — which
   is why the validated Milestone 1 above, though exact, does NOT yet exercise the coupling).
   The Phase-0 gate must be a **coupled `m=1` case with a ring wall**, not the scalar Bessel
   anchor alone.
2. **The `E_r` (radial-normal) inverse-rule is energy-invisible.** `D_r = eps E_r` is
   continuous across a ring wall, so the operator must apply `1/eps` in the **factorized
   inverse form `([[1/eps]])^{-1}`** on the radial-normal component (direct rule `[[eps]]` on
   the tangential `E_phi, E_z`). A wrong factorization gives a convergence floor that
   `sum R + sum T = 1` will NOT catch (the codebase's documented *lossless trap*). Validate
   with a **per-quantity oracle** (analytic step-index/coax Bessel dispersion), never energy
   alone.
3. **The `r dr` measure is load-bearing — "reuse the S-matrix unchanged" is optimistic.**
   The Redheffer star and the *propagation* S-matrix reuse verbatim, but (a) mode
   normalization must use the **`r dr`-weighted mass** `S0_cyl = diag(w * r_node * J)`, and
   (b) the **forward/backward flux split** must use the cylindrical z-Poynting `Im(INT (E_r
   conj(h_phi) - E_phi conj(h_r)) r dr)` — the existing `_split_modes_flux_metric` Poynting sum
   is Cartesian (no `r dr`) and would mis-rank/mis-normalize modes. The interface S-matrix is
   correct *only if* both sides share one `r dr`-weighted basis and W/V are flux-normalized.
   **Add a W/V `r dr`-orthonormality unit test as a first-class deliverable.**

Additional omissions to fold in: the `m=+-1`-only claim holds for an *ideal transverse*
plane wave (finite/focused/tilted/off-axis sources bring in all `m`; gyrotropic/chiral `eps`
make `+m` and `-m` independent — do NOT hard-code `+-1` symmetry reuse); the cylindrical
**far-field** needs a documented `H_m^(1)` → plane-wave (theta) projection (the `1/sqrt(r)`
amplitude + phase), not a one-line Phase-4 item; the **`gamma=0` normal-incidence plane wave**
is the continuum limit (`J_1(gamma r)/gamma → r/2`) — validate the cascade first on
`gamma != 0` structures (bounded modes, focused/Bessel inputs), then handle the plane-wave
edge case.

## D. Re-sequenced milestones (each pinned to an EXACT oracle)

- **M1 — radial eigensolve + axis BC + spectral convergence.** Oracle: Bessel zeros
  `j_{m,n}` (TM) / `j'_{m,n}` (TE) + eigenfunctions. **DONE, ~1e-13.**
- **M2 — coupled `m=1` vector eigensolve at ONE ring wall, with the `E_r` inverse rule and
  `r dr` W/V-orthonormality.** Oracle: analytic step-index / coaxial **Bessel dispersion**
  (`6x6` determinant: `E_z,E_phi,h_z,h_phi` continuous at the wall + PEC `E_z=E_phi=0` at R) —
  a *per-quantity* oracle, not energy. **THIS is the real Phase-0 gate.** (Next up.)
- **M3 — open outer boundary.** Radial PML (complex stretch `r → r(1+i sigma(r))` with the
  `1/r`, `m^2/r^2` terms in the STRETCHED radius; must not corrupt the axis regularity) vs
  analytic Hankel matching at a cap radius. Oracle: bare half-space reflection ≈ 0 (no
  spurious reflection) before any structured run. **The #1 remaining risk.**
- **M4 — z-cascade + scattering.** `r dr`-weighted flux split + interface normalization.
  Oracle: radially-uniform stack → **Fresnel/Airy** (handle the `gamma=0` plane-wave limit).
- **M5 — far-field + public API + library integration.** `H_m^(1)`→theta projection; energy
  per m + reciprocity + Cartesian large-radius limit + FDTD-BOR cross-check.

**Scope note (honest):** M2–M5 are each a separately-validated, research-grade increment
(comparable to M1); the energy-invisible inverse rule (M2) and the open boundary (M3)
especially must not be rushed. M1 (the foundation + the roadmap's #2 risk) is closed.

### M2 progress note (2026-06-19) — coupled operator derived; validation-path subtlety found

The coupled vector layer operator is **derived**: the cleanest form is the cylindrical
analog of the validated Cartesian `_sem_modes_tensor` — a **`q^2`-eigenproblem
`Mbig = P Q` acting on the tangential `(E_r, E_phi)`** (`q^2` are the eigenvalues, `V = Q W
/ lam` the `(h_r, h_phi)` partner). This avoids the `1/gamma^2 = 1/(eps k0^2 - q^2)`
that plagues the longitudinal `(E_z, h_z)` form (which is non-polynomial in `q`). The blocks
are `P, Q` built from the radial metric operators `Lr = d/dr + 1/r`, `m/r`, `eps`, `1/eps`
(the `1/eps` carrying the `E_r` inverse rule).

**Subtlety found (refines the M2/M3 plan):** the `(E_r, E_phi)`-only eigenproblem cannot
cleanly impose the full **PEC** wall BC, because `E_phi(R)=0` is a condition on the state but
the second PEC condition `E_z(R)=0` depends on the PARTNER field `h` (`E_z ~ (1/eps)(Lr h_phi
- i (m/r) h_r)`), not on `(E_r, E_phi)` directly. So the clean *isolated* PEC-bound
validation (M2 vs `j_{m,n}`/`j'_{m,n}` in the coupled basis) is formulation-awkward. Two
clean routes instead:
  1. Validate the coupled modes with the **`(E_z, h_z)` determinant/shooting method** on the
     SEM radial basis (natural BCs: `E_z` Dirichlet, `h_z` natural) against the step-index
     oracle — q found by scanning, sidestepping the linear-eigenproblem BC mismatch.
  2. Or fold M2's validation into **M3's open scattering** (no hard wall — the PML is a
     complex-coordinate stretch in the operator, no `E_z=0` wall condition), validating the
     coupled operator + boundary together against the step-index *reflection*.
This means M2 and M3 are more entangled than first sequenced; the `P Q` operator is correct
for the OPEN (production) use, and the bound-PEC spectrum is best treated as a `(E_z, h_z)`
determinant oracle rather than a coupled-eigenproblem gate. Net: the operator is in hand; the
remaining work is the BC/boundary treatment + one of the two validation routes above.

### M2 empirical result (2026-06-19) — naive collocation FAILS; the correct path is pinned

A naive nodal-collocation `P Q` coupled operator (guided-mode test, core `eps=6` /
cladding `eps=2`, validated against the step-index oracle's guided window `q in
[sqrt(eps2) k0, sqrt(eps1) k0]`) gives the **WRONG** guided `q` (~13.8 vs the oracle's
~14.69). This is the expected **vector-FEM failure**: nodal elements force `E_r`
**continuous** at the `eps` interface, but the physical condition is `E_r` DISCONTINUOUS
with `D_r = eps E_r` continuous. The bidirectional disproof (naive approach refuted against
the oracle) pins the correct formulation:

**The correct coupled operator = the curl-curl weak form with a DIV-CONFORMING `E_r`** (the
cylindrical re-derivation of the library's ALREADY-VALIDATED Cartesian metric generator
`_layer_modes_metric` / `_build_nodal_metric_segments`, which does exactly this for the
Cartesian wall-normal `E_x` / div-conforming `E_z`). Weak form
`INT (curl E)·(curl W) = k0^2 INT eps E·W` with `E_r` allowed to jump at the ring wall
(`D_r` continuous = the inverse rule), the metric `(m^2+1)/r^2` centrifugal and the
`+-2 i m/r^2` `E_r`<->`E_phi` metric coupling, the `r dr` measure, and the axis DOF drop in
the `±` basis. This is NOT naive collocation -- it is the div-conforming spectral-element
generator the roadmap's Table (Section 2) already names as the `_layer_modes_radial`
replacement.

**Net de-risking from this session:** foundation (M1) validated to ~1e-13; M2 oracle
validated to ~1e-9; coupled-operator formulation fully pinned (curl-curl div-conforming,
adapt the validated metric generator) AND the naive alternative empirically refuted. The
remaining M2 work is the focused adaptation of `_layer_modes_metric` to the cylindrical
metric + div-conforming `E_r`, gated by the guided-mode step-index oracle.

### M2 CORRECTION + operator VALIDATED (2026-06-19, later same day)

**The earlier "naive collocation refuted -> need the inverse rule" conclusion was WRONG --
it was an operator BUG, not the inverse rule.** Debugging the first-order operator `K`
(`q Psi = K Psi`, `Psi = (E_r, E_phi, h_r, h_phi)`) by plugging in EXACT analytic
homogeneous modes revealed a single wrong factor in the `E_r` row: the correct equation is
`q E_r = (1/k0) d/dr[ eps^{-1} Cz_h ] + k0 h_phi` (factor `1/k0`), not `i/k0^2`.

**With the fix, the coupled operator is VALIDATED to ~1e-12:** analytic TM (`E_z = J_m`)
AND TE (`H_z = J_m`) modes for `m = 1, 2, 3` satisfy `K Psi = q Psi` pointwise (interior).
The operator -- the genuinely-hard coupled physics -- is correct. (Also fixes a typo in
Section B above: the longitudinal->transverse coupling terms carry the factor `i m k0/r`
and `i m k0 eps/r`, i.e. `E_r = (i/gamma^2)[ q dE_z/dr + (i m k0/r) h_z ]` etc. -- the `i`
was dropped in the first write-up.)

**Cleaner problem structure (found while fixing).** In the `E_pm = E_r +- i E_phi` basis the
HOMOGENEOUS coupled operator **decouples** into two SCALAR Helmholtz problems of order
`m+-1` (`E_+ ~ J_{m+1}`, `E_- ~ J_{m-1}`) -- exactly the validated M1 scalar solver. So:
  - the `E_pm` axis BC is the M1 recipe per component (`m+1` and `m-1` orders: drop the
    axis DOF when the order != 0);
  - the TE/TM **coupling lives ONLY at the ring interface** (the `eps` jump couples the
    otherwise-decoupled `E_+`, `E_-`) -- this, not the bulk, is where the inverse rule
    (`D_r = eps E_r` continuity) matters;
  - the bulk eigensolve is two validated scalar operators; the new work is the interface
    coupling + the vector wall BC.

**Revised M2 plan (operator done):** build the weak-form eigensolve in the `E_pm` basis
(two M1 scalar operators + the `r dr` mass), add the ring-interface `D_r`-continuity
coupling, validate the eigenvalues against the guided-mode step-index oracle (modes decay
before the wall -> wall BC irrelevant). The crude collocation eigensolve (axis `1/r`
zeroing + C0 averaging) is the part that fails, NOT the operator -- replace it with the M1
weak-form machinery.

### M2 EIGENSOLVE WORKS — validated vs the fiber oracle (2026-06-19, evening)

**The coupled eigensolve now reproduces the guided modes.** Three findings, each
validated:

1. **Oracle replaced + a hidden bug caught.** The old PEC-wall `stepindex_oracle`
   has a cross-`eps` bug (returned `q=4.775` where the true bound mode is `4.4363`);
   its homogeneous-only reduction test never exercised the two-`eps` physics — a
   "lossless-trap" analog. The new `fiber_oracle.py` (open `K_m`-decaying cladding,
   the standard fiber HE/EH dispersion) is validated to all printed digits against
   the **independent canonical Okamoto/Snyder–Love** vector characteristic equation.
   It is now the M2 gate. (The prior-session pessimism — "FD gives 13.7 vs 14.69" —
   was an artifact of comparing against the *buggy* PEC oracle.)

2. **The discretization recipe (3 pieces).** `coupled_radial_eigensolver.py`:
   - **q² formulation with E_z elimination** — `K Psi = q^2 B Psi`, `Psi=(E_r,E_phi)`,
     `Phi = (L_m + k0^2 eps)^{-1}[i A E_r - (m/r)E_phi]` (the cylindrical analog of the
     Cartesian `G = I - Kx(1/ezz)Kx` in `_sem_modes_tensor`). Linear in `q^2` → one `eig`.
   - **Wall-normal inverse rule** — `eps_n = [[1/eps]]^{-1}` (harmonic mean across the
     ring); removes the interface mode-doubling and sharpens `q` to the oracle. Mirrors
     `Cxx = [[1/exx]]^{-1}`; tangential `eps` stays pointwise. *Confirmed by ablation:
     with the rule, the 2.93 doublet collapses to one mode and the top `q` hits the
     oracle exactly; the rule does NOT touch the spurious mode (#3) → the spurious mode
     is intrinsic, not interface-related.*
   - **Divergence-free filter** — real-space vector discretizations emit spurious modes
     violating `div(eps E)=0`. **The discrete divergence MUST be consistent with the
     operator**: use the inverse-rule normal flux `D_r = eps_n E_r` (NOT pointwise
     `eps E_r`) — the inconsistent form inflates the physical modes' divergence ~100x and
     destroys the separation (a debugged false alarm). With the consistent metric:
     physical modes `|div(eps E)|/k0|E|` ~ `0.02–0.35`; the spurious `q=3.76` mode reads
     `4.7` — a stable separation across discretizations (`Rbig=8/N=600` and `12/700`).

3. **Validation (cell-centered FD, `coupled_radial_eigensolver.py`).** Top guided mode
   (`e1=6,e2=2,k0=2,a=1,m=1`) converges to oracle `4.43630`: N=300→`4.4479`,
   N=600→`4.4364` (err `7e-5`), N=1000→`4.4364`. The oracle `|det|` is smooth & non-zero
   at `3.77` (no root) → confirmed spurious, filtered. On the V=6 case (`k0=3`) the solver
   recovers **all three** `m=1` bound modes (`7.022/5.887/5.454` vs oracle
   `7.018/5.872/5.433`, err ≤`2.1e-2` = the 2nd-order FD floor) AND two of three `m=2`
   modes — **no spurious leakage**. 4 pytest gates in `test_coupled_eigensolver.py` (match
   oracle, multimode, spurious-filtered, divergence-separation) — all green.

**Open items (the honest remainder):**
- *Weakly-guided modes near the cladding line* (e.g. `m=1` `2.939`, decay length `~1.25`)
  need a large box to separate from the radiation continuum; the explicit dense
  `(L_m+k0^2 eps)^{-1}` inverse conditions poorly at large `Rbig·N`. **Fix = the SEM
  upgrade** (block-sparse, no global dense inverse; element-aligned interface → spectral
  accuracy AND better conditioning) — the planned accuracy follow-on, now also a
  robustness need. The FD prototype validates the *physics + recipe*; the SEM is the
  production discretization. The strong/intermediate modes are solid today.
- *Far-field / radiation modes.* Bound-vs-radiation is currently a heuristic tail test;
  M3's PML makes this rigorous.

### M3 DONE — radial PML (the #1 risk) validated (2026-06-19, evening)

**The open radial boundary works.** A radial PML via complex coordinate stretching
(`s(r) = 1 + i sigma(r)`, `sigma` ramping polynomially in `[R_pml, Rbig]`; in the operator
`d/dr -> (1/s) d/dr`, `1/r -> 1/r_tilde`) is folded into `radial_coupled_modes` as optional
`R_pml`/`sigma_max` args (default `None` = hard wall, byte-identical to M2).

- **Gate A (the definitive PML check) — PASS.** A true bound mode cannot feel the absorber:
  the strong guided `q = 4.436382` is **invariant to <1e-6 across `sigma_max = 3, 8, 20`**
  (and equals the oracle to `8.5e-5`). If the PML were wrong, the bound q would drift with
  `sigma_max`.
- **Gate C — PASS.** The radiation continuum is pushed off the real axis (>80 modes acquire
  `Im(q) != 0`, absorbed), in forward/backward `±q` pairs, while the bound modes stay real.
  The boundary is genuinely OPEN.
- **Note on the M2 "open item":** with the *consistent* divergence metric (the M2 fix) the
  hard wall at a modest box already resolves the weakly-guided `2.948` — so that item was
  mostly the divergence-bug, not box-clutter. The PML's real payoff is **M4**: the radiation
  basis is now outgoing/absorbed, which is what the open-domain S-matrix cascade needs.
- 2 tests added (`test_pml_bound_mode_invariant_to_sigma`, `test_pml_absorbs_radiation_modes`);
  full M1+M2+M3 suite = **17 green**.

**Status:** the hard-physics core of the BOR-PMM is validated end-to-end — radial operator
(M1) + coupled vector eigensolve (M2) + open boundary (M3), each against an exact oracle.
Remaining: **M4** z-cascade S-matrix (`r dr` flux split; Fresnel anchor) and **M5**
far-field + public API + library integration. The FD prototype is the reference; a SEM
re-discretization is the production accuracy/conditioning upgrade (can come before or after
M4 since the operator structure is fixed).

### M4 DONE — z-cascade S-matrix machinery validated (2026-06-20)

**The axial S-matrix cascade works.** `experiments/bor_pmm/zcascade.py`: each z-layer's
M2 modal basis → tangential `W = [E_r; E_phi]` / `V = [h_r; h_phi]`, cascaded by the
Redheffer star. A 5-agent research workflow grounded every convention; three were then
**independently re-derived and verified** before coding (bidirectional check):

- **h-field extraction** `h_r=(1/k0)[(m/r)E_z - q E_phi]`, `h_phi=(1/k0)[q E_r + i dE_z/dr]`
  → satisfies `curl h = -i k0 eps E` to **6e-12** (with the inverse-rule normal flux `eps_n`;
  the pointwise-eps residual is the localized ring inverse-rule, exactly as in M2).
- **backward mode = `[W; -V]` EXACTLY** (`0.0`): `q->-q` flips `E_z` hence `h_t`, leaving
  `E_t` → lumenairy's `_interface_smatrix` is reusable VERBATIM.
- **flux-based forward/backward split** validated (13/14 modes, `sign(P_z)=sign(Re q)`).

**The key structural finding** (workflow + independent read of `rcwa/_core.py`): the interface
match is **POINTWISE** (`solve(Wb, Wa)` on the shared grid) — the `r dr` measure does NOT
enter the interface algebra, **only** the flux selector + R/T efficiency. (Cartesian RCWA's
unweighted harmonic sum is the Parseval inner product of orthonormal Fourier modes; the
cylindrical radial basis is orthonormal under `r dr`, so only the flux/energy layer changes.)

Gates (`test_zcascade.py`, 5 green):

- **GATE 0 same-medium identity** (the STRONG measure-free test — no Fresnel coefficient can
  absorb an `r dr`/flux-sign error): `|S11| = 5e-11`, `|S21 - I| = 4e-11`. ✅
- **round-trip** `a->b->a` interface == identity: `1.7e-10`. ✅
- **GATE 1 per-mode Fresnel** (m=0, sign smoke test — circular by design, NOT cascade
  validation): `1e-10`. ✅
- **GATE 2 slab Fabry-Perot Airy** (propagation + Redheffer signs): `3e-10`. ✅
- **GATE 3 energy** `R + T = 1` on a lossless slab (monitor; lossless-trap-aware, paired with
  GATE 0): exact to `1e-6`. ✅

**Scoped to M5** (per the workflow's risk register): clean multi-mode half-spaces (the crude
FD wall makes uniform-layer *propagating* modes leaky → only ~1 clean propagating mode; fix =
PEC-wall BC or analytic Bessel half-space modes); the **cross-N Galerkin projection** (uses
the dense `r dr` cross-flux overlap `O`, with a `reldiv` spurious-mode prefilter — designed by
the workflow, ready to validate); and **GATE 4** (the Cartesian large-R limit — the only anchor
that exercises genuinely multi-mode non-uniform-layer `r dr` coupling, needs the structured
ring-layer operator + far-field projection).

### M5 IN PROGRESS — far-field core validated; structured-layer path characterized (2026-06-20)

**Far-field core DONE.** `experiments/bor_pmm/farfield.py`: the Fourier-Bessel / discrete
Hankel decomposition projecting an axisymmetric near-field of order `m` onto cylindrical
orders `kt_n = j_{m,n}/R` (→ far-field angles `sin theta_n = kt_n/(sqrt(eps) k0)`), with the
**Parseval power normalization** `INT|f|^2 r dr = sum|c_n|^2 N_n` (`N_n = R^2 J_{m+1}(alpha_n)^2/2`)
for diffraction efficiencies. Validated (`test_farfield.py`, 3 green): round-trip exact,
reconstruction `1e-4`, **Parseval to `1.8e-10`**, `kt->theta` propagating/evanescent split.
This is the cylindrical analog of the planar grating's Fourier-order decomposition; for a
circular grating of radial period `Lambda` the populated `kt_n` cluster at the diffraction
orders, → the planar grating equation as `R -> inf` (the basis for GATE 4).

**Structured-layer path precisely characterized (the M5 frontier).** A concentric-ring
grating layer (`eps(r)` with rings — already supported by the M2 solver) emits **~393/500
spurious** (`reldiv>1`) modes vs ~21 physical. The full-basis pointwise interface then blows
up (`|S11| ~ 2e4`) while the all-uniform control stays `~1e-9` — the spurious modes differ
between a structured and a uniform layer, so they no longer cancel (same-grid uniform was
immune, which is why M4's GATE 0 passed). cond(`W`) is fine (`1.7e3`); the physical subspace
(reldiv<0.5) is well-conditioned (`2.4e2`). **Fix = project the interface onto the physical
subspace** (Galerkin `r dr` overlap + reldiv prefilter), which also needs clean half-spaces.

**M5a DONE — clean half-spaces (PEC/Dirichlet wall).** The crude one-sided FD wall makes
uniform-layer *propagating* modes leaky (~1 clean mode); the research workflow's recommended
fix (smallest change, keeps the square pointwise interface byte-for-byte) is a Dirichlet wall
via the antisymmetric ghost `f_N = -f_{N-1}` → clean real-q box spectrum (`{j_{m,n}} ∪
{j'_{m,n}}` in the uniform limit). Opt-in `wall='pec'` on `radial_coupled_modes` +
`layer_modes` (default `'natural'` byte-identical; `R_pml` overrides to the open PML).
Validated (+2 tests): PEC same-medium identity `3e-12`; **multi-mode Fresnel — ≥8 (vs 1)
propagating modes, mean `2e-3`**.

**M5 far-field design (grounded, 5-agent workflow).** Project the **±1 vortex** components
`E_± = (E_r ± i E_phi)/sqrt2` onto `J_{m±1}` (clean scalar Hankel — the cylindrical mirror of
the planar TE/TM split): `F_+(kt)=INT E_+ J_{m+1}(kt r) r dr`, `F_-(kt)=INT E_- J_{m-1} r dr`.
Normal incidence excites **m=±1 only** (left/right circular → m=+1/−1). z-power via the
Hankel–Parseval relation `INT f g* r dr = INT F G* kt dkt`, flux density `~ Re(q(kt))/k0
|F|^2` over the propagating cone `kt < sqrt(eps) k0`; circular grating of period `Lambda` →
peaks at `kt_n = kt_inc + 2 pi n/Lambda` → the planar grating equation as `R -> inf`.
`farfield.py` (Fourier-Bessel, Parseval `1.8e-10`) is the validated scalar engine for this.

### M5 PROTOTYPE COMPLETE — high-level solver + GATE 4a; production port SEM-gated (2026-06-20)

**M5b high-level solver DONE.** `experiments/bor_pmm/bor_solve.py` (the `BORStack` prototype):
per-layer M2 modes (PEC half-spaces) → **flux-normalized** basis (`|S|^2` = power fraction) →
M4 cascade → physical-propagating-mode R/T efficiencies. The PEC wall (M5a) also **tamed the
structured-layer interface** (`|S11|` from `~2e4` → `4.4`, cond `7e2`). A ring-grating stack
(m=1) cascades and conserves energy.

**The spurious-mode floor (measured, important).** A ring layer emits ~383/400 spurious
(`reldiv>1`) modes; their real-q members leak energy into unphysical channels. Structured-stack
`R+T` conserves to **mean ~1.5%, max ~3.8%** — and this **does NOT improve with N** (`3.8e-2`
at N=200 AND N=400), so it is a genuine FD-vector-discretization floor, not a 2nd-order error.
The clean cure (no spurious modes) is the **div-conforming SEM** re-discretization.

**GATE 4a PASS (the rigorous Cartesian-limit intermediate).** At m=1, a uniform interface
reflects each radial mode with **EXACTLY the planar TE/TM Fresnel coefficient at that mode's
local oblique angle** `theta = arcsin(gamma/(sqrt(eps) k0))` — validated to **~1e-5** across
9/9 modes spanning 19°–46°. This is the load-bearing cylindrical→planar correspondence
(against the closed-form Fresnel, independent of both solvers); it proves the cylindrical
metric (`1/r`, `m^2/r^2`) reduces to planar oblique incidence. (`test_bor_solve.py`, 3 green.)

**Deliberately NOT done — production port is SEM-gated** (the honest call, matching the
workflow's "library port only after M5a-e GREEN"):
- **Full multi-order GATE 4** (ring grating vs `pmm_efficiency_1d` on an `r0/Lambda` ladder,
  per-order < 2-3%, slope ~1.0) — the per-order efficiencies exist (each output mode is a
  diffraction order at its far-field angle), but the ~1.5% spurious floor sits right at the
  tolerance, so a clean multi-order pass needs the SEM first.
- **M5f library port** (`BORStack` in a NEW `lumenairy/elements/bor/`) — gated on the SEM so
  production code is machine-precision, not 1.5%-floored. The `bor_solve.solve()` API shape is
  the port template.

**Net:** the BOR-PMM physics pipeline is **complete and validated end-to-end** (radial operator
→ coupled eigensolve → open boundary → cascade → clean half-spaces → R/T efficiencies → far-field
→ cylindrical-planar correspondence). The one remaining engineering item before a production
library release is the **div-conforming SEM re-discretization** (kills spurious modes → machine
precision → unlocks the full multi-order GATE 4 and the `BORStack` port).
