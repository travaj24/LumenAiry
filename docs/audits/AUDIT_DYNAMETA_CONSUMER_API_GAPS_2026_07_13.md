# DynaMeta Consumer — Lumenairy API Gaps & Feature Requests — 2026-07-13

> **STATUS — FULLY SHIPPED (2026-07-14).** A1 / A2 / B(PMMStack) shipped
> 2026-07-13 (v5.21.4); the ENTIRE remainder — B(PMM2D leg), C1, C2, C3, D1 —
> shipped 2026-07-14 on branch `feat/consumer-api-remainder`. Remainder gates
> in `tests/unit/test_audit_dynameta_consumer_api_2.py`; A1/A2/B-PMMStack
> gates in `tests/unit/test_audit_dynameta_consumer_api.py`.
>
> **Remainder ship notes (2026-07-14):**
>
> - **B (PMM2D leg):** `per_order_amplitudes(port)` + `jones_transmission()`
>   on `PMM2DStackHybrid` (and the `PMM2DStack` alias) and `PMM2DStackPure`
>   via a shared `PerOrderAmplitudesMixin` (pmm/_core.py) — the exact
>   `RCWAResult` contract with 2-D `(N, 2)` orders. Hybrid captures the
>   conj-gauge amplitudes in both the full and even-parity-fold branches;
>   Pure is public-gauge end-to-end. Validated vs RCWA complex amplitudes on
>   an identical crossed-pillar cell at normal + conical (Pure ~3e-4, Hybrid
>   ~3e-3 vs a converged reference); flux-recipe closure exact (0.0).
> - **C1:** `BORStack.per_mode_amplitudes(port)` — deterministic PINNED
>   eigenvector gauge (dominant field sample real-positive; the raw `res["S"]`
>   gauge is now documented on `solve`, whose DIAGONAL was always
>   gauge-invariant) — and `BORStack.layer_absorption()` via
>   `solve(retain_internal=True)` partial cascades + the staggered two-grid
>   flux. Oracles: budget `R+T+sum A = 1` at 1e-12 (lossless A ~ 6e-14);
>   split-layer consistency 7e-16; the pinned fundamental-mode COMPLEX
>   reflection matches analytic Fresnel (5e-16) and Fabry–Perot (2e-14) at
>   the mode's own local angle — machine-precision PHASE, directly unblocking
>   `BorResult.fundamental_result`'s `phase_deg = 0`.
> - **C2 (the flagship):** Berreman OOP-tensor-at-OBLIQUE
>   `retain_internal=True` — the generalized (Li 2003) cascade now retains
>   the SAME internals shape as the native core: asymmetric modes sliced from
>   the `M` blocks + generalized-convention partial cascades, mapped to the
>   public gauge by conjugation with a modal-H NEGATION (`-i` H convention is
>   not conj-invariant; both probe-pinned). `internal_field` /
>   `layer_absorption` — already mode-shape-agnostic + full-tensor `E_z`
>   recovery — serve it unchanged. Gates: closed absorption budget on the
>   lossy tilted-director stack at oblique AND conical (9e-16!); lossless
>   zero (1.6e-15); theta->0 continuity vs the native path for absorption
>   (1.8e-7) and all six field components (4.9e-5 at theta=1e-4).
> - **C3:** `PMM2DStackPure.solve(retain_internal=True)` +
>   `layer_absorption()` — Hybrid-pattern partial cascades on the pure
>   staggered cascade; flux via the eps-free block field Gram
>   (`Re(h2^H G1 e1 - h1^H G2 e2)`, the Eq.25 dual pairing, probe-pinned
>   `Re` on the homogeneous-mode oracle). Budget 8e-14 (lossless 2e-13);
>   Pure-vs-Hybrid per-layer cross-gate 6.7e-3.
> - **D1:** RCWAStack traces uniform `eps=` scalars AND `set_source`
>   wavelength/theta/phi (kept raw when traced; backend dispatch includes
>   them; grazing nudge + propagating-incidence guard documented as skipped
>   under trace; homogeneous-mode cache bypassed). Forward parity 4e-15;
>   AD-vs-FD: eps 6e-9, wavelength 1.1e-7, theta 1.7e-8 — the
>   dispersion-engineering `d/d(wavelength)` leg on the stack twin is live,
>   so the consumer's lifted-cell workaround can be dropped behind a version
>   check.
>
> - **A1 shipped:** `BerremanStack.jones_transmission()` — `Jt` retained on all
>   four solve paths (NumPy main + OOP-oblique, JAX plain + retain; the JAX
>   retain twin `_solve_jax_retain` now returns it). Gate: bit-identical to the
>   functional `jones_t` across the five probe case classes; the one-solve
>   consolidation (far field incl. `t` + `layer_absorption`) closes the
>   absorption budget at 1e-10.
> - **A2 shipped:** `RCWAStack.layers` — read-only tuple property, record
>   fields documented as public.
> - **B shipped for `PMMStack` (the acceptance-gate engine):**
>   `PMMStack.per_order_amplitudes(port)` mirroring the
>   `RCWAResult.per_order_amplitudes` contract EXACTLY (same keys, PUBLIC
>   `exp(-iwt)` gauge, k-vectors normalized by `k0`), plus
>   `PMMStack.jones_transmission()` (the minimal-cut `t`). Retained on the
>   classical mount (incl. convection-slant / generalized-OOP close-outs) and
>   BOTH native-conical paths (patterned nodal + uniform Fourier); the
>   covariant uniform-slant cascade and the JAX twin raise (documented).
>   Validated against RCWA per-order COMPLEX amplitudes on identical physics:
>   classical oblique ~7e-5/4e-4, conical `theta=30 phi=25` ~2-4e-4, uniform
>   conical EXACT (0.0); flux-recipe closure `<= 1e-16`; rotated s-hat
>   synthesis matches the RCWA-amplitude oracle at 9e-6 while the naive
>   per-order power sum is off 1.25e-2 (the C4-2 cross-term physics). One
>   incidental fix: the nodal-conical path's exported evanescent `kz` now
>   carries the PUBLIC decaying branch (`Im >= 0`; the old
>   `conj(kz_forward(conj(eps)))` gauge-map flipped it — R/T were unaffected,
>   `Re()`-only).
> - **B open remainder:** the same modal surface for `PMM2DStack` /
>   `PMM2DStackPure` (separate 2-D far-field close-outs; the 1-D acceptance
>   gates named below don't exercise them). `OpticalResult.t` for the 2-D
>   engines stays deferred with them.
> - **C1–C3, D1: OPEN** (roadmap-class as written; C1 was unblocked by the BOR
>   classifier fix and is the natural next pick).
>
> No correctness defect is
> asserted here — that is `AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md`. This
> document is the consumer-driven residue of DynaMeta's bridge-expansion campaign
> (its audit `docs/audit/2026-07-05-deep-audit.md` section 8, items B1-B6, completed
> 2026-07-13 against lumenairy 5.21.3): every item below is something the DynaMeta
> bridge had to double-solve, hard-raise, or work around because the surface it needed
> is computed upstream but not exposed. Items A1/A2 are one-liners; B is the
> substantive gap; C/D are roadmap-class.
>
> **A discovery made while preparing this audit narrows the list:**
> `RCWAResult.per_order_amplitudes` (rcwa/stack.py:239-269) ALREADY provides public
> per-order complex tangential amplitudes with a pinned gauge, both ports, normalized
> k-vectors, and the documented flux recipe. The conical s/p synthesis for patterned
> RCWA cells (DynaMeta audit C4-2 / 8.1-1) is therefore feasible TODAY on the consumer
> side — it is NOT requested here. The request is PMM-family parity with that surface.

All file:line anchors relative to
`d:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy\lumenairy\`.
Consumer anchors relative to DynaMeta
(`d:\Metacept\Neurophos\Python_Test_Scripts\DynaMeta`), branch `fix/deep-audit-2026-07-05`.

---

## Context: what the bridge campaign shipped, and where it hit the API surface

DynaMeta's `optics/lumenairy_bridge/` now spans six backends (RCWA, PMM, PMM2D
Pure+Hybrid, Berreman, BOR, EMT) plus JAX design twins, with a single version floor
(>= 5.21) and cross-engine referee gates. During that campaign the bridge:

- solves Berreman absorption cases TWICE because `BerremanStack.solve` discards the
  transmission Jones it computes (A1);
- reads the private `RCWAStack._layers` slot under a version ceiling because no public
  accessor exists (A2);
- hard-raises conical incidence for PMM/PMM2D patterned cells because their results
  expose only lab-basis row efficiencies + a zeroth-order reflection Jones (B);
- returns `OpticalResult.t = None` for every PMM-family solve (B, minimal cut);
- reports BOR phase as 0 and defers BOR absorption parity (C1);
- degrades gracefully (warn + None) for OOP-tensor absorption at oblique (C2);
- lifts uniform jax eps values into constant `(Smin, Smin)` cells to keep gradients
  alive through `RCWAStack.add_layer` (D1).

Each section states the need, the physics/consumer path it feeds, the current upstream
state, a proposed (non-breaking where possible) API, and the acceptance gate — usually
an existing DynaMeta validation that flips from raise/None to a pinned number.

---

## A. Quick wins (one-liners with consumer-proven value)

### A1. Retain/return the transmission Jones from `BerremanStack.solve`

- **Current state:** the class solve computes `Jt` on BOTH paths and discards it —
  main path `Jr, Jt, R, T = _farfield(...)` returns `R, T, Jr` (berreman.py:700-712);
  OOP-oblique branch unpacks `_Jt` and drops it (berreman.py:695-697). The functional
  `berreman_jones_1d` DOES return it — which is exactly why the DynaMeta bridge runs
  TWO solves per absorption-enabled call: the functional entry for the far field
  (incl. `t`) plus a `retain_internal=True` class solve for `layer_absorption`.
- **Request:** a non-breaking accessor — retain `Jt` in `self._internal` (it is already
  a `_farfield` output; zero extra compute) and add `BerremanStack.jones_transmission()`
  alongside the existing internal observables. A breaking `(R, T, Jr, Jt)` return is
  NOT needed.
- **Value:** ~2x on every Berreman absorption solve in the pipeline seam.
  **The consolidation is pre-proven safe:** a DynaMeta probe (H5d batch, 2026-07-13)
  compared class-vs-functional far fields across isotropic / oblique / in-plane-tensor
  / conical-rotated / OOP-oblique cases — bit-identical (diff 0.0) in all five.
- **Acceptance:** DynaMeta collapses `berreman_backend._solve_at` to one solve;
  `validation/lumenairy_berreman_bridge.py` (gates A-F incl. the conical E2E leg) and
  `tests/test_lumenairy_bridge.py` stay green with `r/t` unchanged to the last bit.

### A2. Public `layers` accessor on `RCWAStack`

- **Current state:** the per-layer records (`thickness/.kind/.data/.dispersive`) live
  only in the private `_layers` list (rcwa/stack.py:921). DynaMeta's reverse translator
  reads it through a version-ceilinged shim (`lumenairy_bridge/_common.py
  stack_layer_records`) that warns on any lumenairy newer than the tested 5.21.x line.
- **Request:** `@property layers` returning a read-only view (tuple) of the records —
  the same shape `solve_vs_wavelength`'s `_materialized_layers` consumes. Document the
  record fields as public.
- **Value:** deletes the last private-surface read in the DynaMeta bridge; the shim
  already prefers a public `layers` attribute if present, so the consumer needs ZERO
  changes the day this ships.
- **Acceptance:** `stack_layer_records` hits the public branch;
  `validation/lumenairy_translate.py` green; the version-ceiling warning becomes dead
  code to delete at the next floor bump.

---

## B. Per-order complex amplitudes for the PMM family (the substantive gap)

- **Current state:** every PMM-family solve returns
  `(orders, R_eff(2,N), T_eff(2,N), jones_reflection(2,2))` — per-order POWERS keyed to
  incident lab `E_x`/`E_y`, plus the zeroth-order reflection Jones only
  (`PMMStack.solve` pmm/stack.py:789; `_solve_conical` :555-633; `PMM2DStack` /
  `PMM2DStackPure` likewise). No per-order complex amplitudes, no transmission Jones
  of any order.
- **Why powers are not enough:** for an incident SUPERPOSITION (the rotated
  `s-hat = (-sin phi, cos phi)` eigen-polarization at conical incidence), the total
  efficiency has cross terms between the two lab columns per order —
  `R(u) != ux^2 R_x + uy^2 R_y`. Without per-order amplitudes the bridge cannot
  synthesize rotated s/p totals, so DynaMeta hard-raises conical for every patterned
  PMM cell (its audit C4-2: the lab rows at `phi != 0` are s/p mixtures; probe showed
  a silent 32% error before the guard). The Berreman covariance shortcut does not
  apply — a patterned lattice is not z-rotation-invariant.
- **Request:** expose per-order complex tangential amplitudes with a pinned gauge,
  MIRRORING the existing `RCWAResult.per_order_amplitudes` contract EXACTLY
  (rcwa/stack.py:239-269: dict of `Ex`/`Ey` `(2, N)` rows keyed to incident lab pol,
  `kx/ky/kz` normalized by `k0`, both ports, public `exp(-iwt)` gauge, flux-recipe
  note). The data exists at solve time: the nodal conical close-out already computes
  the Rayleigh projections (`pmm/conical.py` `_conical_jones_farfield` /
  `_conical_nodal_solve`), and the classical cascade holds the modal amplitudes it
  squares into `R_eff`/`T_eff`.
- **Minimal cut if the full surface is deferred:** the zeroth-order TRANSMISSION Jones
  alone. DynaMeta returns `OpticalResult.t = None` for every PMM-family solve today —
  and `t` is the phase-bearing modulator observable, so PMM (the referee engine for
  exactly the metallic-TM cells modulators use) cannot currently referee the
  transmitted phase.
- **Value chain unlocked:** (1) exact conical s/p for patterned PMM cells — closes
  DynaMeta 8.1-1, the last conical gap after RCWA (consumer-side, see STATUS note) and
  Berreman (shipped via covariance); (2) `t` for PMM/PMM2D; (3) per-ORDER cross-engine
  referee gates (today's gates compare order-summed totals only — per-order agreement
  is the audit-grade oracle, per the standing lossless-trap rule).
- **Acceptance:** a DynaMeta conical-PMM gate pinned against the FEM solver (its
  validated conical incumbent) on a lamellar metal grating at `theta=30, phi=25`;
  `t`-phase gate vs the RCWA bridge on a subwavelength cell; existing
  `validation/lumenairy_pmm_bridge.py` GATES A-E stay green.

---

## C. Roadmap-class physics features

### C1. BOR: pinned-gauge complex amplitudes + per-layer absorption

Was blocked behind `AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md`; that classifier
fix has since landed (branch `fix/bor-grazing-cutoff`), so this item is now actionable.
`BORStack.solve` already returns the raw S-matrix (`res["S"]`, bor_stack.py:265)
but the modal column GAUGE is undocumented, so a consumer cannot take `S11[jp, j]` as a
physically-phased amplitude; and no internal-field/absorption observable exists at all.
DynaMeta's `BorResult.fundamental_result` consequently reports `r = sqrt(R)` with
`phase_deg = 0` and its BOR absorption parity item (B4b) is deferred. Request, in
order: (a) document/pin the flux-normalized column gauge of the returned S-matrix (or
expose `per_order_amplitudes`-style accessors); (b) a `layer_absorption` counterpart
(the z-flux-difference recipe `PMMStack.layer_absorption` uses, pmm/stack.py:1479,
transfers directly). Acceptance: DynaMeta `lumenairy_bor_bridge` grows a phase gate +
an absorption-budget gate; `fundamental_result` stops zeroing the phase.

### C2. Berreman internal fields for OUT-OF-PLANE tensors at oblique

The one regime where the 4x4 tier still refuses internals: berreman.py:683 raises for
OOP-tensor-at-oblique `retain_internal` ("asymmetric-mode recovery in the generalized
convention with a Rayleigh-consistent flux — machinery neither this solver nor
rcwa.RCWAStack currently has"). Far field is exact (the 5.21-line fix); DynaMeta
degrades gracefully (warn + `A_independent=None`, pinned by
`tests/test_lumenairy_bridge.py::test_berreman_oop_oblique_first_class`). But the
blocked case — tilted-LC director stacks at oblique/conical — is DynaMeta's flagship
device class, and per-layer absorption is what feeds its D2 absorption ->
electro-thermal -> reliability chain. This is the highest-physics-value item here and
the hardest; the raise message itself is the design sketch. Acceptance: the DynaMeta
pin test flips from asserting the warn to asserting a closed absorption budget
(`sum A_i == 1 - R - T`) on the tilted-director reproducer in that test.

### C3. `PMM2DStackPure` internal fields / `layer_absorption`

The Pure engine (no Fourier floor — the engine of record for metallic patches on MIM
stacks) has no `retain_internal`; only the Hybrid does. Absorption maps matter most
exactly where Pure is the right engine (lossy metal walls). Same recipe as C1b/PMM.
Acceptance: DynaMeta `pmm2d_backend` absorption=True stops raising on
`engine="pure"`; budget-closure + Hybrid-vs-Pure cross-gate in
`validation/lumenairy_pmm2d_bridge.py`.

---

## D. JAX design-twin ergonomics

### D1. Trace uniform `eps=` scalars and source parameters in the RCWAStack JAX path

`add_layer(eps=...)` complex-casts uniform eps and `set_source` floats
wavelength/theta/phi, severing gradients (only patterned cell VALUES and thicknesses
trace). The DynaMeta twin (`lumenairy_bridge/rcwa_design.py`, campaign item B6) works
around it by LIFTING a traced uniform eps into a constant `(Smin, Smin)` cell — correct
(parity 3.4e-15, AD-vs-FD 2.1e-11) but it pays a patterned-layer eigensolve for what is
physically a uniform slab, and wavelength/angle gradients (dispersion-engineering
objectives) remain impossible on the stack twin — DynaMeta routes those to the PMM
twin, which traces them (tests/unit/test_v5_14_2_jax_stacks.py). Request: accept jax
scalars through the uniform-eps and source slots (trace instead of cast) so the twins
have symmetric capability. Acceptance: DynaMeta drops the lifting workaround behind a
version check; its GATE B (grad vs FD-of-non-JAX-bridge) extended with a
d/d(wavelength) leg on the stack twin.

---

## Explicitly NOT requested (checked during this audit)

- **RCWA per-order amplitudes** — already public (`per_order_amplitudes`,
  rcwa/stack.py:239); the DynaMeta conical-RCWA synthesis is consumer-side follow-on
  work, tracked in the DynaMeta audit, not an upstream gap.
- **`eps_tensor_cell` on PMM2DStackHybrid** — already exists; the tensor raise in
  DynaMeta's pmm2d bridge is consumer-side scope, not an engine gap.
- **Sweep threading for PMM** — the bridge threads its own per-wavelength loop with
  `rcwa_blas_threads` (campaign item B3, byte-identical, x6.1); no upstream change
  wanted.
- **Analytic-`shapes` payload contract** — the bridge rasterizes via its shared
  painter and is converged; pinning the shapes fast-path contract is nice-to-have
  documentation, not blocking anything.

---

*Prepared 2026-07-13 from the DynaMeta deep-audit / bridge-expansion session (same
session as the BOR classifier audit). Companion documents: DynaMeta
`docs/audit/2026-07-05-deep-audit.md` sections 8-9 (the campaign this residue falls
out of); `AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md` (the one correctness
defect, kept separate). No library code modified by this audit.*
