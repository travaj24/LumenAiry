# PMM + RCWA Suite Audit — 2026-07-02 (v5.18.1, post-reorg deep pass)

Scope: every line of `lumenairy/elements/pmm/` (11 files, ~14,600 lines: `_core.py`, `oned.py`,
`stack.py`, `twod.py`, `stack2d.py`, `twod_staggered.py`, `twod_jones.py`, `_jax_twod.py`,
`_jax_stack.py`, `_jax_stack2d.py`, `__init__.py`) and the core of `lumenairy/elements/rcwa/`
(`_core.py`, `oned.py`, `twod.py` in full; `stack.py` solve-path / caching / layer-modes /
stabilize in a targeted pass), checked against `CONVENTIONS.md` and the modal-method literature
(Li 1996/1997/1999/2003, Granet 1999/2017/2023, Edee–Granet 2024, Schuster 2007, Lalanne 1997,
Moharam–Gaylord).  Single-context inline audit (no agent fan-out); every finding below was
verified directly against the current source at the cited lines.  Read-only — no code was
changed by this audit.

Focus questions from the sponsor: (1) speed/accuracy improvements for the PMM and RCWA
functions generally; (2) can the **2-D PMM** be made to run more efficiently?

---

## 1. Executive Summary

The suites are **unusually well-hardened**: the guard rails (energy tripwires, Wood-anomaly
nudges, incidence guards, forward-branch selectors, lossless-closure windows, stabilize
consensus machinery) are best-in-class, and **no confirmed physics errors** were found in the
PMM/RCWA formulas checked (see §6 for the explicit verification list).  What the pass did find:

* **One real defect**: the JAX 2-D *cell* path skips the `max_nodal_dof` cost guard AND
  materializes dense N×N operators the NumPy path specifically factorized away — an unguarded
  OOM/resource blowup (B1).
* **One stale physics docstring**: `rcwa_efficiency_2d`'s `fff_nv` text still describes the
  pre-v5.14.1 dual-Laurent E_z rule that audit F1 replaced (B2).
* **The headline for the 2-D PMM question**: a set of concrete, exactness-preserving speedups
  worth roughly **5–15× combined** for the canonical centered-pillar workload — most of which
  consist of porting optimizations that *already exist and are validated on the RCWA side*
  (structure-aware Redheffer stars, the even-parity symmetry sector, circular truncation,
  layer-eig dedup) but never made it into the PMM copies of the same machinery (§2).
* The single riskiest structural smell is that `pmm/_core.py` deliberately keeps a **local
  copy of the S-matrix algebra** ("kept local so this module stays in ONE convention") that
  has now **performance-diverged** from `rcwa/_core.py`'s — the exact divergence-by-copy
  failure mode the copy was meant to avoid (F1).

**Headline recommendations, in order:** land F1 (propagation-star + zero-block port — free,
bit-exact, benefits every PMM path), then F2 (even-parity fold for the 2-D hybrid — the
biggest single lever for the centered-pillar-at-normal-incidence workload), then F4/F5
(PMM2DStack sweep caching + factorized sandwiches), then the accuracy-per-order items (§4),
which convert into cubic cost savings.  Fix B1 promptly (small, mirrors `_jax_stack2d`).

---

## 2. The 2-D PMM efficiency question

Cost anatomy of one `pmm_efficiency_2d` solve at defaults (degree 11, n_orders 11 →
Nf = 529, modal dim 2Nf = 1058):

| Stage | Cost | Notes |
|---|---|---|
| Axis SEM build + projections | O(Nf²·N), N = nodal DOF | k0-free; already factorized (v5.14) and hoisted by `PreparedPMM2D` |
| **Layer eig** `eig(P@Q)` | O((2Nf)³) zgeev | **dominant**, per wavelength |
| S-matrix chain (2 interfaces + 2 stars) | ~10–15 GEMMs + ~4 inversions at 2Nf | same flop order as the eig; better flop rate, still a solid 20–40 % of wall clock |
| Far-field tail | negligible | |

### F1 (P1-perf) — Port RCWA's structure-aware S-matrix algebra into `pmm/_core.py`

`pmm/_core.py:580-611` keeps a local Redheffer/interface/propagation copy, documented as
"algebra-identical to rcwa's".  It has **diverged**: the RCWA copy gained

* the zero-block shortcut in `_redheffer_star` (`rcwa/_core.py:1298-1304` — skips both dense
  `inv(I - 0)` inversions when either operand is a propagation S-matrix), and
* the diagonal-aware `_propagation_star` / `_propagation_star_general`
  (`rcwa/_core.py:1353-1380`) — audit RCWA-LEV-2 measured **463.5 ms → 7.3 ms at 2N = 722**
  for this one substitution.

PMM's local `_redheffer_star` still computes `D = inv(I − B11@A22)`, `F = inv(I − A22@B11)`
against literal zero blocks, plus the full GEMM chain, at **every propagation star on every
PMM path**: 1-D scalar/Jones/slant, the 2-D hybrid (`pmm/twod.py:656-658`), staggered
(`twod_staggered.py:977-979`), `PMM2DStack`, `PMMStack`, and both JAX twins (`_jpmm_solve`'s
local `_star`).  Expected win: ~1.2–1.5× on a single-layer 2-D solve; substantially more on
multilayer staircases (one propagation star per slice).  Risk: none — algebraic identity,
already proven on the RCWA side.  (Either import rcwa's helpers or mirror them; the
"one-convention" isolation argument does not apply to these two helpers, whose algebra is
convention-free.)

### F2 (P1-perf) — Port the even-parity symmetry solve to the 2-D PMM

RCWA's opt-in `symmetry=True` (`_symmetric_solve_rt` / `_symmetric_cascade_rt`,
`rcwa/_core.py:769-1143`): centro-symmetric cell + normal incidence → the *entire* cascade
(eig, interfaces, Redheffer) runs in the (N+1)-dimensional even sector instead of 2N —
measured **~2–4.5× end-to-end**, growing with n_orders, with off-origin centres handled by a
recentering gauge and a transparent full-solve fallback.  The 2-D PMM hybrid solves in the
*same* Rayleigh order space with the *same* P@Q block structure (`pmm/twod.py:555-579`); the
even-fold machinery is dimension-agnostic and applies directly to `pmm_efficiency_2d`,
`pmm_jones_2d` (in-plane), and `PMM2DStack`.  A centered pillar at normal incidence — the
canonical metasurface unit cell — is exactly the covered case.  Risk: low (same gates, same
fallback; ~1e-12-level result change as documented for RCWA — cf. `docs/TOLERANCE_POLICY.md`).

### F3 (P2-perf) — Both polarizations are nearly free; expose them

In `_pmm2d_solve_core` the eig and the whole S-matrix are polarization-independent;
polarization enters only via `cinc`/`einc_sq` (`pmm/twod.py:661-676`).  `pmm_jones_2d` and
`PMM2DStack` already drive both pols off one S-matrix; the scalar entry points and
`PreparedPMM2D` (which bakes `cinc` at prepare time, `pmm/twod.py:1215-1216`) do not.
Accepting `polarization` at `PreparedPMM2D.solve()` (or returning both rows) makes the second
polarization ~free: **2× for dual-pol workflows.**

### F4 (P1-perf) — `PMM2DStack` has no sweep caching and no repeated-layer dedup

`PMM2DStack.solve_vs_wavelength` (`pmm/stack2d.py:922-957`) loops full `solve()` calls, and
`solve()` rebuilds `_build_axis` + `_scalar_projected_ops` — all wavelength-independent
(nodal assembly, pinv, sandwiches) — **per layer per wavelength** (`pmm/stack2d.py:509-525`).
For a tapered-pillar staircase (8–16 patterned layers) swept over wavelengths this is pure
waste; `PreparedPMM2D` proves the hoist is exact.  Additionally, **repeated identical layers
re-eig every time**: `RCWAStack` dedupes by content key (`_layer_eig_key`,
`rcwa/stack.py:1592-1612` — "a repeated DBR/Bragg period is solved once") and
`_PreparedPMMStack` has the same machinery for 1-D; neither was ported to `PMM2DStack`.

### F5 (P2-perf) — Never materialize `kron(Ty, Tx)`: factorized sandwiches

The doubly-patterned branch of `_scalar_projected_ops` builds `Tp = kron(Ty, Tx)`
((Nf, N), ~2.5 GB at the 150k-DOF cap) and computes `EpsF = (Tp*v) @ Tpinv` in O(Nf²·N)
(`pmm/twod.py:487-491`).  The identical operator is computable by two per-axis tensor
contractions (contract the x-nodal index against `Tx`/`Txp`, then the y-nodal index against
`Ty`/`Typ`): bit-identical math, **no (Nf, N) materialization**, ~10–500× fewer sandwich
flops for large-N staircased cells (a 32-strip disk at N ≈ 83k).  Also raises the practical
`max_nodal_dof` ceiling.  Same restructuring applies to `twod_jones._tensor_layer_modes`'s
dense branch and `_jax_twod` (see B1).

### F6 (P2-perf) — Staggered path: `_fast_geig` fold + sweep-prepared variant

* `_region_modes` / `_homog_geom_cache` use full QZ `sla.eig(L, G)`
  (`pmm/twod_staggered.py:681`, `:737`) although `G` is a Hermitian-PD block-Kronecker Gram.
  The validated `_fast_geig` fold (`pmm/_core.py:452-478`, ~1.5–2× on the dominant eig, with
  the near-singular QZ fallback) applies directly; `inv(G)` at `twod_staggered.py:704` should
  be a Cholesky solve.
* At normal incidence `L = Et + L0/k0²` with `Et`, `L0` k0-free, and the half-space geometric
  spectrum scales as `g2_geo(k0) = g2_geo(1)/k0²` with **k0-independent eigenvectors** — so a
  wavelength sweep needs the half-space eig only **once** (~2× on sweeps; the layer eig
  legitimately re-runs) and can cache the full Galerkin assembly.  No prepared/sweep variant
  exists today.

### F7 (P2-perf) — Parallelize wavelength sweeps

`pmm_efficiency_2d_vs_wavelength` (`pmm/twod.py:1326-1331`) and all stack sweeps are
sequential Python loops around an eig-bound kernel.  zgeev is largely serial and LAPACK
releases the GIL: a thread pool over `prepared.solve(wl)` scales ≈ linearly with cores.
Applies verbatim to `PreparedRCWA2D`.  Composes with the existing `set_blas_threads`
machinery (cap BLAS ~2, run ~cores/2 sweep threads).

### F8 (P2-perf) — Port `truncation='circular'` from RCWA to the 2-D PMM

RCWA supports the Lalanne circular order set (`rcwa/twod.py:90-127`): isotropic resolution,
drops the wasted high-|G| corner orders — Nf → (π/4)·Nf, eig cost ×~0.48 ≈ **2×** at
equal-or-better accuracy per DOF.  The PMM hybrid uses the full square box; its projected
operators only need the order list, so this ports cleanly.

### Combined estimate

For the canonical workload (centered pillar, normal incidence, dual pol, wavelength sweep):
symmetry (~3×) × propagation-star (~1.2×) × dual-pol (~2×) × circular truncation (~2×) ×
sweep threading (~×cores) → **order-of-magnitude wall-clock reduction with zero accuracy
loss**, before the accuracy-per-order items in §4.

**Not recommended:** caching/perturbing the layer eig across nearby wavelengths (not exact —
the projected pencil is a genuine function of k0); complex64 anywhere near the modal eig
(cond ~1e13; cf. `_require_jax_x64`).

---

## 3. Defects found

### B1 (P1 — resource blowup, missing guard) — JAX 2-D cell path: no cost cap + dense N×N assembly

`pmm_efficiency_2d_cell` dispatches to JAX **before** `_validate_cell_cost` runs
(`pmm/twod.py:969-986`), and `_static_prep_cell` (`pmm/_jax_twod.py:179-246`):

* (a) never applies the `max_nodal_dof` cap — the parameter is **silently ignored** on the
  JAX branch; and
* (b) materializes **dense N×N** operators: `Minv = np.diag(1.0 / Mdiag)` then
  `Gx0 = -1j * (Minv @ np.kron(ay["M"], ax["D"]))` — the exact dense-kron assembly the v5.14
  factorization removed from the NumPy path.

A staircased `region_layout` the NumPy path handles factorized (or rejects at the cap) can
attempt ~110 GB at N ≈ 83k and OOM.  The sibling `_jax_stack2d.py:84-101` already uses the
factorized per-axis form *and* is guarded at `add_layer` (via `_append_patterned` →
`_validate_cell_cost`) — fix is to mirror it (and honour `max_nodal_dof`).

### B2 (P2 — stale doc, physics-relevant) — `fff_nv` docstring contradicts the audited F1 fix

`rcwa_efficiency_2d`'s docstring (`rcwa/twod.py:481-485`) still says fff_nv's E_z elimination
"uses the dual-Laurent `[[1/eps]]` rule … the same unbiased E_z rule the analytic-shape
solver uses".  Audit F1 (v5.14.1) changed **both** to the direct rule: `EZZ = E` at
`rcwa/twod.py:332` and the analytic-shape solver at `rcwa/twod.py:1516-1520`.  A reader
implementing from the docstring would reproduce the +0.35-absorptance bug F1 fixed.

### B3 (P3 — API surprise) — `PMM2DStack.solve_vs_wavelength` discards the set_source angle

`pmm/stack2d.py:922-943`: `solve_vs_wavelength(theta=None)` resolves θ to 0.0 rather than
reusing a previously `set_source()`-configured value — silent geometry change for a user who
set the source once and then swept.

### B4 (P3 — dead code) — `pmm/twod_staggered.py:633`: bare no-op statement `xb[seg]`.

### B5 (P3 — fragility) — `_epsF_cache` never invalidated

`pmm/stack2d.py:669-694`: `PMM2DStack._epsF_cache` is keyed by layer index and never cleared
on `add_layer`.  Currently safe (layers append-only, solver knobs init-frozen), but any
future mutation API breaks it silently.  Cheap fix: clear it wherever `self._internal` is
invalidated.

### B6 (P3 — code health) — duplicated l2g construction; dead `robust` parameter

The periodic local→global node map is duplicated inline in five builders (`_build_sem`,
`_build_sem_tensor`, `_build_sem_slant`, `_build_nodal_metric`, `_jpmm_build_topology`)
although `_l2g_periodic` (`pmm/_core.py:1277`) exists and is used by the segment builders —
the same divergence-by-copy risk that materialized in F1.  Also `_sem_modes`' `robust`
parameter has been dead since the v5.14 unconditional robust branch (documented, but
signature debt).

---

## 4. Accuracy-per-order opportunities (cubic cost savings)

These are the *other* axis of "efficiency": every unit of truncation saved is cubed in the eig.

* **A1 — Doubly-patterned Li routing in the 2-D hybrid.**  Acknowledged non-rigorous
  (P3-33 comment, `pmm/twod.py:493-500`): harmonic mean on the Ex slot / Laurent on Ey for a
  doubly-patterned cell, which also breaks 90°-rotation symmetry under `formulation='li'`
  (the separable branches are correctly per-axis).  A per-axis sequential rule in the
  projected-nodal representation (mirroring `rcwa._li_convolutions_2d`) would cut the
  n_orders needed for a given TM accuracy.
* **A2 — `pmm_jones_2d` tensor block is Laurent-only** (Li-1997 mixed rule "NOT implemented",
  `pmm/twod_jones.py:30-33`) — patterned tensor cells converge at the ~1e-3 Laurent floor.
* **A3 — Curved-wall factorization (RCWA; the circular-pillar case).**  The analytic-shape
  solver (`rcwa_efficiency_2d_shapes` — exact Bessel form factors, the natural disk solver)
  is Laurent-only (`rcwa/twod.py:1520`); `fff_nv`, the method designed for oriented walls, is
  correctly **gated off curved geometries** because its cross term mis-splits absorptance
  ~50 %.  For a high-contrast circular pillar today the options are Laurent or
  staircased-'li' — both converge, but slowly in TM.  Repairing the NV cross term (the ~50 %
  absorptance error pattern suggests a fixable defect in the `Δ·[[NxNy]]` assembly or its
  interplay with the direct-rule E_z, rather than a fundamental limit — S4 and RETICOLO field
  working NV implementations), or combining analytic form factors with an NV tensor, is the
  **top accuracy roadmap item for the whole suite** given the circular-pillar application.

Known open accuracy items already acknowledged in-code (listed for completeness, no action
required beyond the roadmap): covariant slant TM floor ~2.5e-3 vs independent oracle
(honestly documented at `pmm/_core.py:3742-3750`); staggered near-cutoff divergence
(loudly warned, `twod_staggered.py:936-944`); ASR high-order bridge conditioning (warned).

---

## 5. Other performance findings (1-D PMM and shared)

* **P1 — 1-D scalar half-spaces don't use the shared geometric eig.**  `_pmm_solve_core`
  runs two full `_sem_modes` eigs for the uniform sup/sub media (`pmm/_core.py:816-822`);
  the Jones path already uses `_uniform_geo_eig` (backlog A2).  Both TE
  (`(Peps − L/k0²)x = q²·S0·x`) and TM (`(S0 − Linv/k0²)x = q²·Pinv·x`) fold, for uniform ε,
  to the *same* geometric pencil `(L/k0²)x = (ε − q²)·S0·x` — one eps-free eig serves both
  half-spaces and both polarizations.  The in-code perf notes say half-spaces are 51–64 % of
  1-D eig time → **~1.5–2× on 1-D scalar solves**, multiplied by every stabilize-scan step.
  The notes defer this as "risky-for-marginal-gain", but A2 has since been implemented and
  validated on the Jones, slant, and stack paths — the risk assessment is stale.  Same
  opportunity in `_pmm_slant_solve`'s half-spaces (`pmm/_core.py:3089-3092, 3117-3118`).
* **P2 — `_sem_fourier_projection` recomputes degree-only quadrature every call**
  (`pmm/_core.py:538-576`): leggauss nodes, barycentric weights, Lagrange values (Python
  loops) depend only on `degree`; the JAX path already factors this
  (`_jpmm_projection_quad`).  An `lru_cache` on the NumPy side is trivial and helps every
  solve and every stabilize-scan step.
* **P3 — 1-D `vs_wavelength` sweeps default `stabilize=True`** — up to a 16-degree consensus
  scan *per wavelength* (`pmm/oned.py:1570-1631` et al.).  A "lock the consensus degree after
  the first wavelength" option would cut long sweeps by roughly the scan length in the
  common case (opt-in, since the resonance set is wavelength-dependent).
* **P4 — `PMMStack`: no content-key dedup of repeated identical patterned layers** within a
  solve (`pmm/stack.py:691-695, 720`); `_PreparedPMMStack` and `RCWAStack._layer_eig_key`
  both demonstrate the pattern.
* **P5 — minor**: `_converged_cluster` recomputes pairwise `_aligned_max_diff` per anchor
  (memoizable; small vs eig); `_assemble_jones_farfield` runs two `lstsq` factorizations
  where one 2-column solve suffices.

---

## 6. Physics and numerics verified sound

Checked in detail against the literature and `CONVENTIONS.md`; **no defects found** in:

* **Conventions end-to-end**: public `exp(-iωt)`, forward `exp(+ikz)`, `n = n + iκ`
  absorbing (CONVENTIONS §7).  The internal `exp(+iωt)` bridges (conjugate-in/out) are
  consistent at every PMM/RCWA entry point read, including the lossy-exit-medium
  un-conjugation in the flux masks (`_forward_flux_kz`; PMM mirrors at
  `pmm/twod.py:687-698`; covariant `kz_ord` at `pmm/_core.py:4017-4024`).
* **Branch cuts**: `_sqrt_forward` (Im ≥ 0, regions) vs `_sqrt_decay` (Re ≥ 0, layer
  propagators — the unconditional S-matrix stability choice with on-cut pinning) are the
  correct pair, for the documented reasons.
* **Flux normalizations**: TM `kz/ε` weighting on H-amplitudes (1-D scalar); the oblique
  p-pol `1 + (kt/kz_inc)²` incident-|E|² factor; `Ez = -(k·E_t)/kz` longitudinal completion —
  correct and mutually consistent across PMM 1-D/2-D, staggered, and RCWA 1-D/2-D.
* **Li factorization**: 1-D inverse-rule placement (Q-block wall-normal); the Li-1996
  anisotropic Schur construction (`_tensor_convolutions`); pointwise-before-convolution
  ezz-Schur for out-of-plane (Li 2003 — the "gen2 trap" correctly avoided in four
  independent implementations); the v5.14.1 direct-rule E_z fix.
* **Bloch shift**: antisymmetrized convection `−i·kx0·(C−Cᵀ) + kx0²·M` weak form (and the
  1/ε-weighted TM variant) — correct pseudo-periodic-envelope discretization.
* **Forward-mode selection**: Poynting-flux selectors with noise tolerances; the
  deep-decay/stability bands in `_select_forward_flux`; unit-consistent rebalance fallbacks.
* **Stabilize machinery**: two-sided passive gate, per-order clique consensus,
  energy-clean pick, `_StabilizeScanExhausted` — sound; the non-monotone-in-degree caveat is
  documented.
* **Redheffer/S-matrix algebra**: both copies algebraically correct (F1's divergence is
  performance-only).

---

## 7. RCWA-side assessment

The RCWA subsystem is in excellent shape: planar TE/TM decoupled fast path (RCWA-LEV-1,
~4–8×), even-parity sector (single-layer + generalized cascade), `PreparedRCWA2D`,
homogeneous-mode LRU cache with the superstrate-in-key fix, layer-eig dedup, BLAS thread
control, ASR with honest scope warnings, Li-1997 sequential rule, circular truncation,
analytic shape factorization, stabilize ladders with closure-warning-aware retries, and
`rcwa_extrapolate` / `rcwa_convergence`.  Remaining items: A3 (curved-wall NV factorization —
the pillar case), F7 (sweep threading), and the parity/naming notes in §8.

---

## 8. Convention / API consistency notes

* **Dimensional vs dimensionless `kx0`** (1-D PMM: rad/m; RCWA/2-D: k0-normalised) is
  documented at both sites (`pmm/_core.py:762-766`, `rcwa/twod.py:631-633`) but remains the
  most likely cross-wiring hazard for future contributors — worth an entry in
  `CONVENTIONS.md` §7's table.
* `degree` = basis-function count M in `pmm_efficiency_2d_staggered` vs GLL polynomial degree
  everywhere else: mitigated by the `n_modes` alias and docstring flag.
* Index-vs-permittivity split (`rcwa_efficiency_1d` takes n; the Jones family takes ε):
  loud CONVENTION WARNINGS exist, but it remains a silent-acceptance trap — a heuristic
  warning when a Jones ε argument looks like an index (all values real and < ~2.5) would
  catch most real mistakes.
* `formulation` defaults differ across the 2-D family (`pmm_efficiency_2d`='li',
  `pmm_jones_2d`='laurent', `rcwa_efficiency_2d`='laurent') — each individually justified in
  its docstring, but the cross-API inconsistency deserves a one-table summary in the docs.
* `rcwa_efficiency_2d` takes `n_orders_x`/`n_orders_y`; PMM-2D takes a single `n_orders` —
  minor parity gap.
* `pmm_efficiency_1d_vs_wavelength` returns total R/T only while the 2-D sweeps return
  per-order — minor asymmetry.

## 9. Feature gaps (PMM/RCWA scope)

* No GPU path for any PMM solver (RCWA has CuPy throughout).  Given the eig-bound profile,
  CuPy support in the 2-D hybrid hinges on `cupy.linalg.eig` availability — feasibility check
  before roadmapping.
* No prepared/sweep variant for `pmm_efficiency_2d_staggered` (F6); no `symmetry` /
  `truncation='circular'` options anywhere in PMM-2D (F2, F8).
* `PMM2DStack`: no prepared caching (F4); no star-power squaring for periodic layer groups
  (DBR-type stacks) — also absent from `RCWAStack`.
* Convergence tooling (`rcwa_extrapolate` / `rcwa_convergence`) has no PMM-facing counterpart
  beyond `stabilize`; a `pmm_convergence` wrapper would be cheap.

## 10. Coverage statement

Fully read: all 11 `pmm/` files; `rcwa/_core.py`, `rcwa/oned.py`, `rcwa/twod.py`;
`CONVENTIONS.md`.  Targeted (solve path, caching, layer modes, stabilize):
`rcwa/stack.py` — its `RCWAResult` field-reconstruction/absorption methods
(lines ~149–773) and builder methods got a structural pass only.  Out of scope for this
audit: the remaining ~120 modules (propagators, raytrace, analysis, optimize, IO, UI) —
recommended as follow-up targeted passes of the same style, one subsystem group per pass.
