# CAMPAIGN PLAN -- PMM per-layer grids, remaining roadmap items

**Date:** 2026-08-04 - **Branch:** `feat/pmm-per-layer-roadmap` (off v5.32.1 main, `013c388`)
**Input roadmap:** `docs/audits/ROADMAP_PMM_PER_LAYER_GRIDS_2026_07_28.md` (updated 2026-08-03)
**Parents:** `AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` (the defect),
`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` (the 1-D implementation),
CHANGELOG v5.32.0 PMM blocks (the memory of record)
**Status:** ASSESSMENT ONLY. No code changed by this document.
**Proposed release target:** **5.33.0** (see S8).

---

## 0. How to read this plan

Evidence classes follow the parent audit's convention, and the distinction is load-bearing:

| tag | meaning |
|---|---|
| **[M]** | **Measured** -- produced by a run |
| **[A]** | **Analysis** -- arithmetic or derivation from the source as it stands today |
| **[H]** | **Hypothesis** -- consistent with the evidence, NOT established. Flagged as such. |

Everything in S1 (the staleness pass) is **[A] read against the tree at `013c388`**, with
`file:line`. Nothing in this plan asserts a numeric behaviour claim -- every such claim is
scheduled as a measurement in S6/S7 with its oracle and its null control named.

**An "agent-cycle" (AC)** = one focused subagent mission: read + implement + measure on both
BLAS builds + write the pins. Roughly one working session. Estimates below are in AC.

---

## 1. Staleness pass -- what the roadmap says vs what the tree says

The instruction to verify every claim was well founded: **eight** statements in the roadmap are
stale, mis-attributed, or wrong, and two of them **invert the shape of the headline item**.

### 1.1 Confirmed correct (do not re-litigate)

| roadmap claim | verdict | evidence |
|---|---|---|
| S0 surface table: classical, conical, `retain_internal`, `solve_vs_wavelength`, slant, OOP, JAX twin all shipped and gated 11/11 | **TRUE** | `tests/unit/test_pmm_per_layer_grids.py` (11 tests, all present); dispatch `stack.py:1286, 1359, 1372, 2747`; `_solve_general_perlayer` `stack.py:1822`; `_solve_vs_wavelength_perlayer` `stack.py:2904`; `_pmm_stack_solve_jax_perlayer` `_jax_stack.py:440` |
| `stabilize='slices'` is N/A by design and raises with an explanation | **TRUE** | `stack.py:1393-1396` |
| `prepare()` per-layer raises, shared named as the alternative | **TRUE** | `stack.py:3111-3115` |
| `_PreparedPMMStack` is coupled to the shared union grid | **TRUE** | `stack.py:3249+` (union-cell-content eig keys) |
| T3-4 (R-2 grid-quality observable) is unshipped and blocked on an unresolved mechanism | **TRUE** | no detector anywhere in `elements/pmm/`; two refuted hypotheses recorded in the parent audit S4.2/S4.5 |
| T3-7 (lattice quantisation) unshipped; the snap is still a cascading pairwise midpoint merge | **TRUE** | `_core.py:3352-3390` |
| S6 shipped: `RCWAStack` OOP -> generalized S-matrix incl. traced OOP gradients | **TRUE** | `rcwa/stack.py:2605, 2621, 2947, 2962-3000`; tracer-as-OOP routing `stack.py:488-503` |
| S6 shipped: GPU-DLL hygiene probes a device op + `cupy.fft` and re-raises a named error | **TRUE** | `rcwa/_core.py:363-372` |
| S6 open: R-1 (mu/bianisotropic), R-2 (hex/oblique lattice), R-4 (`sigma_px` pixel-fixed at 1.5), R-5 (K-matrix, never started) | **ALL TRUE (open)** | R-1: zero hits for `permeability`/`bianisotrop`/`mur` in `elements/rcwa` or `elements/pmm`; R-2: `truncation` accepts only `'rectangular'`/`'circular'` (`rcwa/twod.py:136-162`), `shear=` is a sidewall kwarg (`rcwa/stack.py:1799, 1824-1830`); R-4: `sigma_px=1.5` at `rcwa/twod.py:252`, consumed in pixel units at `twod.py:304-307`, and `_nv_field_2d` takes the periods and never uses them (`twod.py:283-328`); R-5: `Kx`/`Ky` still dense N x N diagonals at every entry point (`twod.py:1000, 1190, 1674, 2120`; `stack.py:2862`; `oned.py:571, 1297, 1499`) |
| S6 positioning: JAX is the gradient path, not the forward-speed path | **TRUE, documented** | `CHANGELOG.md:5957-5959`; `docs/rcwa_roadmap_v5_14.md:86-88`; structurally, the JAX branch skips the NumPy fast paths (`rcwa/stack.py:2902`, `rcwa/_core.py:2008`) |
| S6 positioning: per-layer grids do NOT transfer to RCWA | **TRUE** | zero `layer_grids`/`per_layer` grid concept in `elements/rcwa/` |

### 1.2 STALE / WRONG -- corrections the plan is built on

**S-1. [A] The 2-D pure stack does NOT use a nodal Lagrange/GLL basis, and its segments are
UNIFORM.** Roadmap S1.2 says a 2-D per-layer build takes "each layer's `(Nx_i, Ny_i)`
segmentation from its own cell walls" and that "the 1-D `_lagrange_eval` applies per axis".
Both are false. `PMM2DStackPure` runs on the Granet-2023 staggered **modified-Legendre**
tensor basis (`twod_staggered.py:147` `Basis1D`, doc header lines 10-20), and
`Basis1D.__init__` sets `self.xb = np.linspace(0.0, self.d, self.N + 1)`
(`twod_staggered.py:173`) with a **scalar** jacobian `self.J = 0.5 * self.d / self.N`
(`twod_staggered.py:170-171`) used as one scale factor for the whole period
(`twod_staggered.py:260-266`, `twod_staggered.py:340-347`).

Consequences, all of which reshape S1:
* There are no per-layer *wall positions* in 2-D. There is a per-layer *segment count* on a
  uniform lattice, and `eps_cell` is a constant per lattice cell.
* The 1-D wall-collision pathology **cannot occur** in the 2-D pure stack -- a uniform lattice
  has no near-coincident walls. The roadmap reaches the right conclusion ("a capability gap, not
  the silent-wrong pathology") by the wrong route, and the right route changes what to build.
* The real 2-D blocker for a tapered pillar is **wall representability on a uniform lattice**, not
  interface non-conformity. A 2 deg taper at `n_slice = 6` over 310 nm moves a wall ~1.8 nm per
  slice; representing that exactly on a common uniform lattice of a 700 nm period needs
  `Nx ~ 390`, which is `eigdim = 2*(Nx*(M-1))^2 = 1.5e7` [A] -- not large, *impossible*.
* Therefore **a mortar alone does not unlock 2-D tapers.** The enabling change is non-uniform
  segment boundaries in `Basis1D` (new item **N-1**), and the mortar is what makes the resulting
  per-layer grids affordable. The roadmap has the second half without the first.

**S-2. [A] The pure 2-D stack has NO symmetry fold at all.** Roadmap S1.2 item 5 says "the pure
stack's even-parity/C4 folds assume the shared grid ... or v1 ships without folds, at 2-8x cost".
There is no `symmetry` parameter, no parity fold and no C4 fold anywhere in `stack2d_pure.py` or
`twod_staggered.py` (zero hits for `symmetry`/`parity`/`C4`; `PMM2DStackPure.__init__`
`stack2d_pure.py:120-139` takes only `period_x/period_y/n_*/n_modes/n_orders`). The folds that
exist are the **hybrid**'s (`stack2d.py:89` `symmetry="auto"`) and RCWA's.

This cuts both ways and both matter:
* The claimed *risk* ("per-layer folds must be per-grid") is void -- there is nothing to preserve,
  so S1's scope shrinks.
* The claimed *fallback cost* ("2-8x") is not the right number for the pure stack either: a fold
  that does not exist cannot be dropped. What is real is that **the pure 2-D stack today pays the
  full unfolded eig**, and on the C4 device that is the binding constraint (S-3), so a C2/C4 fold
  becomes a **new prerequisite-class perf item (N-5)**, not a compatibility footnote.

**S-3. [A] The 2-D cost/memory model is not stated, and it is the gate.** `_region_modes` runs a
dense non-symmetric generalized eig `sla.eig(L, G)` on a `2q^2` pencil with `q = N*(M-1)`
(`twod_staggered.py:712-720`; `Basis1D.dim` `twod_staggered.py:248`). Arithmetic [A]
(`n = 2*(N*(M-1))^2`, bytes `= 16 n^2` per matrix, `~30 n^3` flop for `zggev`):

| Nx=Ny | M | eig dim | GB per matrix | flop |
|---|---|---|---|---|
| 3 | 8 | 882 | 0.012 | 2.1e10 |
| 4 | 8 | 1568 | 0.037 | 1.2e11 |
| 6 | 8 | 3528 | 0.185 | 1.3e12 |
| 7 | 8 | 4802 | 0.344 | 3.3e12 |
| 8 | 8 | 6272 | 0.586 | 7.4e12 |
| 13 | 8 | 16562 | 4.09 | 1.4e14 |
| 25 | 8 | 61250 | 55.9 | 6.9e15 |

`sla.eig` holds several such arrays live. A common-grid 2-D taper at `ns = 6` lands at
`Nx ~ 13-25` -- i.e. **4-56 GB per matrix**, which is the same failure class as the 6.6 GB
meshgrid found on a hot path last week, one order worse. Per-layer windows put the same device at
`Nx ~ 7` (0.34 GB/matrix) which is *feasible but heavy*, and a C4 fold would take it to
`~0.02 GB` and `/64` flop. **The 2-D item's real acceptance gate is a memory budget, not an
accuracy number**, and the plan treats it that way.

**S-4. [A] `min_feature` is NOT "inert by construction" on the per-layer path.** The
implementation report S2 and the v5.32.0 CHANGELOG both say "inert by construction (there is no
global union to snap)". The window grids are built by calling the same snapping routine on a
**3-layer union**: `stack.py:1653-1655` passes `self.min_feature / self.period` into
`_pmm_union_grid`, and `_core.py:3353` enters the snap branch whenever
`min_feature > tol = 1e-9`. It merges nothing *at the default* (7 pm) -- which is what was
measured -- but it is live, and it merges exactly the pairs that matter: the tapered staircase's
worst collisions are between **adjacent** slices (per-slice offset 5.41 nm vs a 5.00 nm coat,
parent audit S4.4), and adjacent slices are precisely what a 3-layer window contains. A user who
carries over the shared path's recommended `min_feature = 1.5 nm` gets **window-local snapping
that moves real walls by up to 0.75 nm** with the same warning text, on a path documented as
immune. Same code at `stack.py:1836-1838`, `stack.py:2919-2921`, `conical.py:246-248`,
`_jax_stack.py:483-485`.
Correct statement: *dormant at the library default, and the intended lever for T3-2.*

**S-5. [A] T3-1's "one-line change to the window loop" is a five-site change.** The window
construction `js = [j for j in (i - 1, i, i + 1) if 0 <= j < nlay]` is duplicated verbatim at
`stack.py:1652`, `stack.py:1835`, `stack.py:2920`, `conical.py:245`, `_jax_stack.py:482`.
Widening to +/-2 without first extracting one helper would put five copies of a physics parameter
in five files -- the duplication-kills lesson. **N-4** makes T3-1/T3-2 the one-line change the
roadmap promises.

**S-6. [A] T3-3 is not "latent only" -- it is a live silent rank-deficiency, and it is
inconsistent with all three sibling paths.** `conical.py:202` computes
`cap = (nU * n_el * degree - 1) // 2` from `nU`, the **full-union** cell count, at line 202 --
**before** the per-layer branch at line 235 -- and never re-clamps. On the per-layer path the
half-spaces are built on the *window* grids (`conical.py:266-271`), whose `n_glob` is roughly
`nU/6` on the audit device [A]. The `m_prop > cap` raise at `conical.py:209-213` therefore
**over-states capacity and fails to fire**, and `_sem_fourier_projection(ox, period, mats_sup)`
(`conical.py:344`) then builds a projector with more orders than nodes; `Hsup` goes rank-deficient
and `np.linalg.lstsq(Hsup, rhs, rcond=None)` at `conical.py:373` returns a build-dependent draw.
That is verbatim the C13 mechanism shipped in 5.32.1.
Every sibling gets this right: classical per-layer clamps to `min(n0, nN)` from the *window*
half-spaces and raises (`stack.py:1779-1789`); the per-layer sweep does the same
(`stack.py:2981-2989`); the JAX twin takes "min over the two end grids' capacity"
(`_jax_stack.py:525-529`). Conical is the only outlier. **Promote T3-3 from P3-nicety to P2 with
a fail-before test.**

**S-7. [A] S6's `'li'` even-parity fold is mis-attributed.** The roadmap says the fold reaches
`'li'` "via the generalized `(P, Q)` cascade fold in `rcwa_jones_2d`". In `rcwa_jones_2d` the fold
gate is `twod.py:1702-1704` and requires `formulation == "laurent"`; `'li'`, `'fff_nv'` and any OOP
cell take the full solve there. The `'li'` fold is in **`rcwa_efficiency_2d`** (`twod.py:1086-1093`)
and the prepared 2-D class (`twod.py:1204-1210`), with the per-layer `_tensor_PQ` built in
`rcwa/stack.py:2495-2502`. Also: "ON by default" is not universal --
`rcwa_efficiency_2d_shapes` still defaults `symmetry=False` (`twod.py:1943`), the 1-D core has no
fold at all (no `symmetry` kwarg in `rcwa/oned.py`), and an in-code doc still says "Opt-in
(`symmetry=True`)" five lines after the `'auto'` default is resolved (`rcwa/_core.py:1483-1485`).

**S-8. [A] Two more S6 "shipped" bullets over-read what shipped.**
* **LEV-4** (analytic homogeneous modes from `_cached_homogeneous_eigenmodes`): the cache exists
  (`rcwa/_core.py:3165-3222`) but is consulted **only for the two half-spaces**
  (`rcwa/stack.py:2887-2890`). Uniform *interior* layers call the uncached
  `_homogeneous_eigenmodes` (`rcwa/stack.py:2564`; also `twod.py:1053, 1195, 1696`). They do get
  the analytic non-eig modes, so "instead of running the eig" holds; "draw from the module-level
  cache" does not. Any campaign claim of a cache-hit win on many-uniform-layer stacks is **not yet
  realised**.
* **`fff_nv` rework "incl. crossed cells, ported to the hybrid PMM"**: the Li-2003 L2.L1
  factorization is wired into `rcwa_jones_2d` only (`twod.py:1729-1744, 1773-1784`); the scalar
  pixel engine still builds the Schuster normal-vector field (`twod.py:1042-1051, 1328-1333`) --
  which is the code R-4 is about, so the two S6 bullets are only consistent if "rework" is read as
  *tensor-Jones path only*. The PMM port is **separable-only**: `pmm/twod_jones.py:202-223`
  implements it and `pmm/twod_jones.py:252-260` **raises** on a crossed (both-axes-patterned) cell.
  S5's P-2 states this correctly; S6 does not.

### 1.3 Additional facts the roadmap omits but the plan needs

* **[A] `PMM2DStackPure` also requires `Nx == Ny`** (`stack2d_pure.py:165-170`, and a bare
  `assert` in `Granet2DTransverseE.__init__`, `twod_staggered.py:379`, which `python -O`
  strips). A y-invariant cross-check against the 1-D per-layer result -- S1's own proposed gate --
  therefore requires padding the uniform axis into equal segments, which inflates `q` and the eig.
* **[A] The bare name `PMM2DStack` is a transitional alias to the HYBRID**
  (`stack2d.py:1184-1210`), scheduled to be repointed to the pure stack "once that reaches
  feature + validation parity". S1 is the work that decides that cutover, so S1 carries an API
  consequence the roadmap does not mention. Roadmap S5 P-1's phrase "`PMM2DStack.add_tapered_pillar`"
  is therefore the *hybrid*'s method (`stack2d.py:346`); the pure stack has no taper helper.
* **[A] Cross-masses are hoisted on exactly one of four per-layer paths.** `_sem_cross_mass` is
  computed once per sweep in `_solve_vs_wavelength_perlayer` (`stack.py:2943-2947`) and once per
  trace in the JAX twin (`_jax_stack.py:500-503`), but is **rebuilt on every call** in
  `_solve_vertical_perlayer` (`stack.py:1707`), `_solve_general_perlayer` (`stack.py:1896`) and
  `conical.py:290`. Masses are memoised on all four. The direct per-layer LC sweep -- the working
  substitute for `prepare()` at "2.7 s/point" -- runs through `solve()`, i.e. the *unhoisted*
  path.
* **[A] `_lagrange_eval` (`_core.py:3423-3444`) loops in Python over quadrature points**
  (`for r, x in enumerate(xi)`), and `_sem_cross_mass` calls it twice per union sub-interval
  (`_core.py:3502-3503`). Both are trivially vectorisable.
* **[A] Both mortars materialise `np.kron(np.eye(2), M)`** (`_core.py:3532-3535, 3573-3576`;
  and the JAX twin stores the kron'd forms, `_jax_stack.py:500-503`). That is 4x the memory and
  ~4x the flop of two blockwise half-size operations for a block-diagonal operator.
* **[A] `_redheffer_star_rect` uses `np.linalg.inv` twice** (`_core.py:3603-3604`) where the
  square sibling's own docstring records `solve`-over-`inv` as the deliberate choice
  (`rcwa/_core.py:2033-2035`).
* **[M, subagent] RCWA's conditioning exposure is worse than PMM's and sits on the default path.**
  `rcwa/_core.py:2042` inverts `apb` in `_interface_smatrix` with no `rcond`, no regularisation and
  no fallback -- and `rcwa/_core.py:440` records that this very matrix reaches **cond ~1e13**.
  `rcwa/_core.py:2123-2128` (the 4N generalized interface: `solve(Mb, Ma)` then `inv(T22)`) is
  unguarded and is the newest code. The whole RCWA module contains exactly one `linalg.cond` call
  and it is in an unrelated 1-D path (`oned.py:607`). **RCWA is one of PMM's two in-repo
  independent oracles; an oracle that can draw a build-dependent answer cannot adjudicate.**

---

## 2. Item dossiers

Format per item: *what it is / physics risk / perf+memory risk / oracle / dependencies /
agent-cycles / verdict*.

### N-2. Conditioning hardening of the per-layer mortar, star and Rayleigh solves  **[NEW]**

* **What.** Four unguarded numerical solves on the shipped per-layer path:
  `_interface_smatrix_mortar` -- `solve(kron(I2,Mb) @ Wb, ...)`, `solve(kron(I2,Ma) @ Va, ...)`,
  `inv(I + BA)` (`_core.py:3536-3540`); `_interface_smatrix_general_mortar` -- one dense
  `solve(A, B)` on a `2(n_a+n_b)`-square block system (`_core.py:3581`);
  `_redheffer_star_rect` -- two `inv` (`_core.py:3603-3604`); and
  `lstsq(Hsup, rhs, rcond=None)` at `stack.py:1605, 1814`, `conical.py:373`,
  `stack2d_pure.py:350`, `twod_staggered.py:1052`.
  Apply the 5.32.1 pattern: instrument `cond`, screen, re-solve by QR/SVD where singular, keep the
  re-solve **only if it fits better**, and return historical bits on ties.
* **Physics risk.** *This is the physics risk.* A rank-deficient `Hsup` or a near-singular
  `M_b W_b` returns a null-space draw that differs per BLAS build; the far field then carries
  manufactured or destroyed energy that EE-style and Jones-magnitude metrics do not see. The
  mortar's `V` matrices are the H-partners of modes whose eigenvalues can nearly degenerate at
  Wood anomalies -- the natural place for `M_a V_a` to lose rank.
* **Perf/memory risk.** A `cond` probe is an extra SVD-class cost per interface; must be gated
  (probe cheap, e.g. `1/rcond` from the LU, or probe once per geometry and cache with the mass).
  De-kron'ing (N-3) shrinks these systems 2x in dimension before any screening cost is paid.
* **Oracle.** (a) **Dual-build agreement** (Windows/MKL vs WSL/OpenBLAS) is the primary
  adjudicator -- the C13 precedent moved 6.8e-2 -> 1.2e-10. (b) An **independent re-solve**
  (Householder QR / SVD min-norm) scored on the *residual against the original equations*, never
  on the answer. (c) The **conforming-stack bit-exactness** pin is the null control: on identical
  grids the mortar is bypassed entirely, so every conforming result must stay bit-for-bit.
* **Dependencies.** None. Blocks nothing but should precede any re-pinning of per-layer numbers,
  because a re-pin against a build-dependent value is worthless.
* **Agent-cycles.** 2 AC (1 instrument + census, 1 fix + dual-build pins).
* **Verdict.** **GO -- first.** Highest priority in the campaign: it is a shipped, default-
  reachable path in the exact defect class the library closed one release ago.

### T3-3. Conical per-layer far-field order cap  **[promoted]**

* **What.** Re-clamp `n_orders` (and the `m_prop > cap` raise) to `min(mats_sup["n_glob"],
  mats_sub["n_glob"])` inside the per-layer branch, matching `stack.py:1779-1789`. See S-6.
* **Physics risk.** Today: silent rank-deficient Rayleigh projection at high `ffo` or on a
  many-propagating-order device. After: a loud raise, same text as the classical sibling.
  Residual risk is only that the raise fires on a configuration that used to "work" -- which is
  the point, and is what the fail-before switch documents.
* **Perf/memory risk.** None (the clamp lowers `N`).
* **Oracle.** A **fail-before test that reproduces the rank deficiency**: construct a stack whose
  window capacity is below the union capacity, assert the current code proceeds and that the two
  BLAS builds disagree (or that `cond(Hsup)` exceeds the screen), then assert the fixed code
  raises. Cross-check the *fixed* answer at a legal `ffo` against the shared conical path and
  against `pmm_jones_1d_conical_tensor` on a conforming 2-layer stack (already bit-exact-pinned).
* **Dependencies.** Bundles with N-2 (same file, same failure class, same test harness).
* **Agent-cycles.** 0.5 AC.
* **Verdict.** **GO -- bundle with N-2.**

### N-6. `min_feature` is live inside the per-layer window  **[NEW]**

* **What.** Correct the "inert by construction" claim in the implementation report, the CHANGELOG
  wording and the `layer_grids` docstring; decide the intended contract. Two defensible contracts:
  (i) keep the snap live and document it as the T3-2 lever (recommended), or (ii) pass `None` on
  the per-layer path so the claim becomes true. **Do not choose by preference -- measure.**
  Also: the snap warning now fires up to `nlay` times per solve with `stacklevel=3` pointing into
  the window loop.
* **Physics risk.** A user carrying the shared path's recommended `min_feature = 1.5 nm` onto the
  per-layer path moves real walls by up to 0.75 nm, on a path documented as immune. The parent
  audit measured that a 0.75 nm wall move changes the device ER by ~16% (S6.2).
* **Perf/memory risk.** Negligible either way.
* **Oracle.** The two contracts must be **measured against each other** on the audit device at
  `mf in {default, 0.5, 1.5, 3.0} nm x degree {6,8,10} x ns {2,6,8}`, scored on degree spread AND
  on `|R+T-1|`. Adjudicate the winner against the shared grid at its validated `mf = 1.5 nm` --
  the cross-validation closure that already reads ~1.5% (impl report S5.4).
* **Dependencies.** N-4 (one window helper) makes this a one-place change; feeds T3-2 directly.
* **Agent-cycles.** 1 AC (mostly the sweep).
* **Verdict.** **GO.** A wrong immunity claim in the memory of record is exactly the kind of thing
  that costs months later.

### N-4. Extract one per-layer window-grid helper  **[NEW, prerequisite]**

* **What.** One function `_perlayer_window_grids(layer_segments, min_feature_frac, halfwidth=1)`
  replacing five verbatim copies (S-5). Pure refactor; byte-identical by construction at
  `halfwidth=1`.
* **Physics risk.** None if byte-identity is pinned.
* **Perf/memory risk.** None.
* **Oracle.** Byte-identity across the full per-layer test set and the audit-device gates, on both
  builds. Per the standing rule, compare **by tolerance at 0.0**, not `array_equal`.
* **Dependencies.** Blocks T3-1, T3-2, N-6.
* **Agent-cycles.** 0.5 AC.
* **Verdict.** **GO.**

### T3-1. Mortar residual: +/-2-neighbour window vs third-neighbour enrichment

* **What.** Reduce the non-conforming remainder (`|R+T-1|` ~1e-4 at deg 6 -> ~1e-6 at deg 10)
  toward the shared path's ~1e-14. Two levers: (a) widen the window to +/-2 neighbours,
  (b) enrich each grid with one element at third-neighbour wall positions only.
* **Physics risk.** Low for (a) -- it is more of what already works, and the own-walls-only
  failure (75-83% spread) proved the mechanism is *wall-set coverage at the interface*, so the
  functional form is right. **Real risk for (b)**: adding elements at third-neighbour walls
  reintroduces the near-coincident-wall geometry the whole design removed. Measure (a) first, as
  the roadmap says.
* **Perf/memory risk.** (a) costs ~`(5/3)^3 = 4.6x` per eig, not the roadmap's quoted `(5/3)^3`
  "per eig" figure applied loosely -- and the eig is ~97% of a region solve
  (`twod_staggered.py:~780` records the same ratio in 2-D), so **~4.6x on the whole solve**. That
  puts the 17.8x speed-up at ~3.9x. Cross-masses grow as `n_a x n_b` -> ~2.8x memory per
  interface, which is why N-3's hoist-and-de-kron should land first or alongside.
* **Oracle.** The **shared grid at a validated `min_feature`** is the adjudicating oracle for the
  value; the **energy defect `|R+T-1|` on a lossless synthetic** is the adjudicating oracle for the
  residual (it shares no code with the far-field Jones assembly path being scored). Add a
  **manufactured-solution mortar test**: project a known smooth analytic field from grid A to
  grid B and score the L2 projection error against its closed form -- an oracle for the mortar
  alone, independent of any solve.
* **Dependencies.** N-4. Interacts with T3-2 (both change the window).
* **Agent-cycles.** 1.5 AC.
* **Verdict.** **GO**, as a *measurement with a switch*, not a default change. Ship
  `window_halfwidth` as an opt-in knob with the measured cost/accuracy table in its docstring; flip
  the default only if the measured trade beats 1 at deg 8. The roadmap's own position -- "acceptable
  for ER work at deg >= 8" -- means the default flip must earn its 4.6x.

### T3-2. High-`n_slice` stress band (deg spread up to 5.6% at ns=8)

* **What.** Find the setting where the answer is stationary in `(degree, min_feature, ns)`
  simultaneously, sweeping the two levers: window-local `min_feature` (N-6) and window halfwidth
  (T3-1), at ns = 8-12 on the audit device.
* **Physics risk.** The band is the honest uncertainty on every quoted per-layer number at high
  ns; the winner-suite is already quoted at ns >= 6. If no setting is stationary, the correct
  outcome is a **quoted band**, not a chosen setting -- the plan must not force a false convergence.
* **Perf/memory risk.** This is the compute-heavy item: `ns in {8,10,12} x deg {6,8,10} x mf {4} x
  halfwidth {1,2} x theta {5}` = 1800 solves. At the measured 28-175 s/rung this is a mesh job, not
  a laptop job (see `mesh-run`).
* **Oracle.** The **converged shared reference** at `mf = 1.5 nm` where it converges, plus
  `|R+T-1|` and `sum(A) ~ 1-R-T` scored at every cell. Do **not** score on ER alone: ER is a ratio
  with a deep null in the denominator and is the *sensitive* observable, not an independent one
  (parent audit S4.3).
* **Dependencies.** N-4, N-6, T3-1.
* **Agent-cycles.** 1.5 AC (0.5 to set up, 1 to adjudicate; the compute runs on the mesh).
* **Verdict.** **GO.** This closes the last accuracy caveat on the shipped 1-D surface, which is
  the roadmap's own step 1.

### N-3. Per-layer hot-path efficiency: hoist, vectorise, de-kron  **[NEW]**

* **What.** Four measured-then-fixed items, all on the shipped path:
  1. **Hoist cross-masses** into a geometry cache on `solve()` / general / conical (they are
     already hoisted on the sweep and JAX paths -- S1.3). Geometry is wavelength-, angle- and
     material-independent, so the cache is keyed on `(wkey_a, wkey_b, degree, n_el, grade)` and
     lives for the life of the stack.
  2. **Vectorise `_lagrange_eval`** (Python loop over quadrature points, `_core.py:3436`).
  3. **De-kron the mortars**: apply the block-diagonal `kron(I2, M)` as two half-size operations
     instead of materialising the 4x-larger matrix (`_core.py:3532-3535, 3573-3576`;
     `_jax_stack.py:500-503`).
  4. **`inv` -> `solve` in `_redheffer_star_rect`** (`_core.py:3603-3604`).
* **Physics risk.** (1), (2), (4) must be **bit-identical or better**; (3) changes the arithmetic
  order and will move the last bits -- so (3) needs a tolerance pin and a dual-build check, and
  (4) improves accuracy but also moves bits. Sequence (1)+(2) as byte-identical, then (3)+(4)
  behind one fail-before switch.
* **Perf/memory risk.** This *is* the perf item. Expected [H, to be measured]: (1) removes
  `nlay-1` cross-mass builds per solve -- on a 38-layer stack that is 37 Python-loop assemblies;
  (3) halves the mortar solve dimension (`8x -> 2x n^3`, i.e. ~4x on that step) and cuts the peak
  resident mortar memory ~4x.
* **Oracle.** Byte-identity (tolerance-at-0.0) for (1)+(2). For (3)+(4): the **conforming-stack
  bit-exactness** and the **mortar-reduces-to-plain-interface** identity pins, plus `|R+T-1|` on
  the lossless staircase -- if de-kron'ing changed physics, closure moves. Speed measured against a
  `git archive` baseline on an idle machine with `OMP/OPENBLAS/MKL_NUM_THREADS=1`, both builds.
  Memory measured with `tracemalloc` peak + RSS delta and a largest-array census.
* **Dependencies.** Independent of everything; its cache is ~most of S3's `prepare()` design.
* **Agent-cycles.** 2 AC.
* **Verdict.** **GO.** Directly serves the stated bar ("efficient algorithms in both speed and
  memory") on the path users are told to use, and it de-risks T3-1's 4.6x.

### S3. `prepare()` per-layer

* **What.** A per-layer prepared object: geometry cached forever, a material-key override
  re-eigs only that layer's window and rebuilds its two interfaces.
* **Physics risk.** Low-to-moderate: a second correctness-sensitive cache class, with a
  byte-identical-after-eviction contract to preserve.
* **Perf/memory risk.** The stated payoff is against a 2.7 s/point direct sweep -- and **that
  sweep runs on the unhoisted `solve()` path** (S1.3). N-3 item 1 captures a large share of the
  same win with none of the cache-correctness surface.
* **Oracle.** Byte-identical re-solve after LRU eviction; agreement with the direct per-layer
  sweep to solver round-off; LRU bounds measured in bytes, not entries.
* **Dependencies.** N-3 (its cache is the substrate).
* **Agent-cycles.** 3 AC (mostly cache-correctness tests, per the roadmap's own estimate).
* **Verdict.** **DEFER past 5.33.0.** Re-assess after N-3 measures how much of the win the
  geometry cache already delivers. The roadmap's own trigger -- "when an LC-sweep-heavy campaign
  makes the loop feel slow" -- has not fired.

### N-1. Non-uniform segment boundaries in the 2-D `Basis1D`  **[NEW, prerequisite for S1]**

* **What.** Replace `Basis1D`'s uniform `xb = linspace(...)` and scalar `J` with arbitrary
  boundaries and a per-segment jacobian. The elementary matrices are already computed on the
  reference interval and scaled by one factor per matrix class
  (`twod_staggered.py:260-266`: mass `*J`, stiffness `*1/J`, mixed `*1`), and
  `_global_pair_segmat` already returns per-segment contributions (`twod_staggered.py:340-353`) --
  so the change is a per-segment weight vector in place of a scalar, in two functions.
* **Physics risk.** **Moderate and specific.** The mixed (one-derivative) matrix currently scales
  by exactly 1 because `J` and `1/J` cancel -- that cancellation is *per segment* and survives, but
  it must be re-derived rather than assumed. The Bloch periodic hat (Eq. 33) glues segment `N-1`
  to segment `0` across the seam; with unequal segment lengths the two half-hats have different
  jacobians and the glue coefficient must be checked for the C0 property. The de Rham property
  `d(Btilde) subset span(B)` -- the *entire* justification for the mimetic curl placement
  (`twod_staggered.py:~230-240`) -- is verified at ~1e-15 today and **must be re-verified per
  segment** under non-uniform widths. If it fails, the item fails, and that is the go/no-go.
* **Perf/memory risk.** None intrinsically; it *reduces* the `N` needed to place a wall exactly,
  which is a cubic win on the eig.
* **Oracle.** (a) The de Rham residual itself (a property of the basis, no solver involved).
  (b) A **uniform-boundary regression**: with equal segments the new code must be byte-identical to
  today's. (c) Physical validation against `PMM2DStackHybrid` and `rcwa_jones_2d` on a
  Fourier-friendly geometry where both converge, and against `pmm_efficiency_2d_staggered`'s
  existing position-invariance property (move a pillar within the cell; efficiencies must not
  move) -- which becomes a *stronger* test with non-uniform segments because the pillar can now
  sit anywhere.
* **Dependencies.** Blocks the useful form of S1. Independent of all 1-D work.
* **Agent-cycles.** 1 AC for the de Rham feasibility spike (go/no-go), 2 AC to implement + validate
  if green.
* **Verdict.** **GO for the 1-AC spike in 5.33.0.** Implementation GO/NO-GO on the spike's de Rham
  result. This is the highest-leverage discovery in this assessment: it may deliver most of S1's
  *capability* (exact tapered-pillar walls) without any 2-D mortar at all.

### S1. PMM2DStack per-layer grids + 2-D mortar (the roadmap's headline)

* **What.** Per-layer 2-D segmentations, a 2-D cross-mass, a 2-D mortar, per-grid half-spaces and
  far field. Corrected by S-1/S-2/S-3 above.
* **Physics risk.** **High, and higher than the roadmap states.** Three distinct hazards:
  1. **The two transverse components live on different staggered sets.** `V1 = B(x) x Btilde(y)`
     with Gram `kron(Mtt_y, Mbb_x)`; `V2 = Btilde(x) x B(y)` with Gram `kron(Mbb_y, Mtt_x)`
     (`twod_staggered.py:~437-441`). The roadmap's "blockwise `kron(I2, .)` pattern carries over
     unchanged" is **false**: the two components need *different* cross-mass operators
     (`kron(Ctt_y, Cbb_x)` and `kron(Cbb_y, Ctt_x)`). Getting this wrong is a silent, energy-clean
     error -- the worst class.
  2. **`B` is the discontinuous partner.** An L2 mortar does not require continuity, but the weak
     E/H pairing (E tested below, H tested above) must be posed on spaces that make the system
     square when `q_a != q_b`, and the 1-D argument for squareness does not transfer mechanically
     when the two sides use different set pairs.
  3. **Mortar residual scales with the interface measure** -- in 2-D a curve set rather than a
     point set, so expect a higher residual band at equal degree. Measure before promising.
* **Perf/memory risk.** **This is the gate.** See the S-3 table. Additionally: the roadmap's
  proposed 2-D cross-mass "by tensor-product Gauss over the union rectangle partition" would be a
  **dense `q_a^2 x q_b^2` matrix** -- 144 MB at `q = 42` per interface, GB-scale over a stack.
  **It factors exactly as a Kronecker product of two 1-D cross-masses** (both bases are
  tensor-product; the partitions are rectangular), i.e. two ~42x42 matrices applied by reshape.
  ~1000x memory reduction, and it is the separable/marginal construction the standing rules
  demand. Likewise `M_b^{-1}` is `kron` of per-axis inverses blockwise, so the mortar's
  `solve(M_b W_b, .)` restructures to one dense solve against `W_b` plus separable applications.
  **The plan requires the Kronecker form; a dense 2-D cross-mass is a rejected design.**
* **Oracle.** (a) `PMM2DStackHybrid` on Fourier-friendly geometry (different basis, no shared
  code path with the pure stack's assembly). (b) `rcwa_jones_2d` / `RCWAStack` (a third, wholly
  independent method) -- **but only after N-2/X-1**, since the RCWA interface inverts a matrix
  documented at cond ~1e13. (c) The **1-D per-layer result in the y-invariant limit** (requires
  padding, per S1.3). (d) `|R+T-1|` and `sum(A) ~ 1-R-T` at every gate. (e) Position invariance
  (move the pillar in the cell). (f) External: inkstone / grcwa per the repo's convention, on a
  geometry inside their Fourier floor.
* **Dependencies.** N-1 (else the capability does not arrive), N-5 (else the target device may not
  run), N-2 (else the oracles cannot adjudicate).
* **Agent-cycles.** 8-12 AC (the roadmap's "2-4 weeks", plus N-1 and N-5, minus the void fold-
  compatibility work).
* **Verdict.** **DEFER out of 5.33.0**, per the roadmap's own condition ("gated by the dual-cut C4
  campaign decision -- build it when that campaign is green-lit, not speculatively"). Take the
  N-1 spike now; hold the rest. If the campaign is green-lit, S1 is its own release.

### N-5. C2/C4 symmetry fold for the pure 2-D stack  **[NEW]**

* **What.** A normal-incidence parity/C4 block-diagonalisation of the `2q^2` pencil, the pure
  stack's analogue of RCWA's LEV-3 and the hybrid's `symmetry='auto'`.
* **Physics risk.** Moderate; the precedent is well trodden (RCWA `_symmetric_cascade_rt`,
  `rcwa/_core.py:1759-1849`) and the failure mode is known: the fold's precondition is joint
  centro-symmetry of the whole cascade, and an off-origin symmetry centre gives an inconsistent
  `(P, Q)` -- a defect already caught once in `twod.py:708-710`.
* **Perf/memory risk.** It *is* the perf item: /4 memory and /8 flop for C2, /16 and /64 for C4
  (S-3 arithmetic). Without it the C4 device is a multi-hour, multi-GB solve.
* **Oracle.** The unfolded solve, at equal `M`, bit-comparable within tolerance; plus the
  `symmetry=False` fail-before switch reproducing prior bits exactly.
* **Dependencies.** Independent of S1, but S1's feasibility on the target device depends on it.
* **Agent-cycles.** 3 AC.
* **Verdict.** **DEFER with S1**, but note it is *separable*: it can ship on the shared-grid pure
  stack as a standalone perf win whenever wanted, with no mortar involved.

### T3-5 (R-3). Taper-aware `min_feature` default (shared path)

* **What.** Derive the default from `(thickness/n_slices)*tan(sidewall)` when a taper recipe is
  recorded; warn when geometry-built stacks look tapered.
* **Physics risk.** Changing a default silently alters every existing tapered-stack user's answer
  -- and the parent audit's S9 explicitly declined to change it for that reason. Do it as a
  **warning + a documented recommended value**, not a default flip, or gate the flip behind a
  major-version deprecation.
* **Perf/memory risk.** None.
* **Oracle.** `self._taper_recipes` exists for builder-made stacks (`stack.py:235, 509`), so the
  derived value is exact there. The **geometry-built route records no recipe** -- that was the
  whole of R-1 -- so it needs the wall-offset-statistics heuristic, which must be validated against
  a labelled set of known-tapered and known-vertical stacks (a false "this looks tapered" warning
  is the failure to avoid; R-1b's precedent: a false pathology claim is worse than silence).
* **Dependencies.** None.
* **Agent-cycles.** 1 AC.
* **Verdict.** **GO if the release has room; otherwise DEFER.** Cheap and protects shared-path
  users, which is now the *publication-grade* path -- so it is not as low-urgency as the roadmap
  implies.

### T3-7 (R-7). Lattice wall quantisation

* **What.** Replace the cascading pairwise midpoint snap with deterministic rounding to a lattice
  `Delta`, sized by the separations to REMOVE (~1 nm class), not the features to keep.
* **Physics risk.** Low, and it *improves* the accounting: bounded displacement `<= Delta/2` and
  reproducible, versus a cascade whose displacement had to be instrumented to be reported at all.
  The counter-intuitive sizing rule (finer lattice merges fewer pairs) is already derived and must
  be carried into the docstring verbatim.
* **Perf/memory risk.** None.
* **Oracle.** Per-material total-width preservation across all layers vs the exact reference (the
  parent audit's S6.1 table is the ready-made instrument, and it must be re-run, not cited);
  degree spread and `|R+T-1|` on the audit device at matched `Delta` vs matched `min_feature`.
* **Dependencies.** None. **Note the newly-visible synergy:** N-1's 2-D basis needs walls on a
  representable set, and a lattice is exactly that -- T3-7 and N-1 share a sizing argument.
* **Agent-cycles.** 1 AC.
* **Verdict.** **GO.** Small, self-contained, improves the shared path (now the publication-grade
  reference) and `_pmm_union_grid`'s reproducibility, and it now has a second consumer.

### T3-4 (R-2). Grid-quality observable

* **Verdict.** **DEFER -- correctly blocked.** Two hypotheses measured and refuted; the roadmap's
  instruction ("do not ship a detector premised on an unconfirmed story") stands and is the right
  call. The practical guard exists (R-1 consensus + per-layer as an independent cross-check).
  Revisit only if the mechanism is pinned. **0 AC.**

### T3-6 (R-5). Covariant taper-metric layer (first order in `tan(phi)`)

* **What.** A timeboxed feasibility spike: is the general trapezoid's `u = x/w(z)` z-dependent
  metric treatable at first order in `tan(sidewall)` (2 deg is small) as a single convection-like
  term, or by a Magnus/product-integral treatment?
* **Physics risk.** Research-class; the obstruction is real and named (a shear absorbs a
  translation, not a dilation). Highest science value, least certain -- the roadmap's own words.
* **Perf/memory risk.** If it works it *obsoletes the staircase*, removing the `n_slice` axis
  entirely -- the largest available perf win in the whole plan.
* **Oracle.** The converged **staircase limit** (`ns -> large`) computed per-layer, which is now
  affordable and measured stationary at ns ~ 6-8. That is a genuine independent oracle: the
  covariant layer and the staircase share no assembly code.
* **Dependencies.** None. Papers first (Granet 2017 for shear; Edee & Granet 2024 josaa-41-9-1803
  for the crossed-slant map) before any re-derivation.
* **Agent-cycles.** 1 AC spike (analysis + a scalar prototype, no library change).
* **Verdict.** **GO as an analysis-only spike.** Deliverable is a derivation and a go/no-go, not
  code. Sequenced after T3-1/T3-2 so it does not compete for the same reviewer.

### P-1. Native 2-D slant (Edee & Granet 2024)

* **Verdict.** **DEFER.** The roadmap's sequencing is right: after S1 and T3-6, whose outcomes
  bound how much a native map is still worth. **0 AC now.**

### P-2. Li mixed inverse rules for the hybrid PMM's crossed anisotropic cell

* **What.** The residual gap: the crossed anisotropic cell in the hybrid PMM stays Laurent-floored
  (~1e-3); `pmm/twod_jones.py:252-260` raises on crossed cells today.
* **Verdict.** **DEFER.** Strategically superseded by S1 (the pure stack has no Fourier floor);
  a hybrid-only nicety, correctly classified. **0 AC.**

### R-3 (RCWA). Even-parity fold coverage for OOP and `fff_nv` cells

* **What.** OOP (ezz-Schur) cells and `fff_nv` cells always take the full 2N solve. Exclusion
  sites: `rcwa/stack.py:2515-2517` (an OOP layer kills the fold for the *whole* cascade, because
  `stack.py:2906` requires `all(sp is not None)`), `rcwa/stack.py:2488-2489` (dispersive),
  `rcwa/twod.py:1702` (`rcwa_jones_2d` requires `laurent`), `rcwa/twod.py:1083-1099`
  (`fff_nv` falls through), `rcwa/_core.py:1684-1689`.
* **Physics risk.** Moderate: the fold's precondition under an ezz-Schur term needs deriving, not
  assuming, and the off-origin-symmetry-centre defect precedent (`twod.py:708-710`) shows this
  family bites.
* **Perf/memory risk.** ~3-4x on the affected solves; note `fff_nv` is **unreachable from
  `RCWAStack`** (stack layers accept only `laurent`/`li`, `rcwa/stack.py:397-400, 1562-1565`), so
  half of R-3 is a `twod.py`-entry-point-only win.
* **Oracle.** The unfolded solve at equal orders; `symmetry=False` as the fail-before.
* **Agent-cycles.** 2 AC.
* **Verdict.** **DEFER past 5.33.0.** Real but not on the PMM campaign's critical path, and the
  OOP half touches the same generalized-interface code as X-1 -- sequence it *after* X-1 so it is
  not built on an unguarded solve.

### X-1 (RCWA). Conditioning of the default interface and generalized-interface solves  **[NEW]**

* **What.** `rcwa/_core.py:2042` (`inv(apb)`, documented at cond ~1e13 at `rcwa/_core.py:440`) and
  `rcwa/_core.py:2123-2128` (`solve(Mb, Ma)` + `inv(T22)`, the 4N generalized interface) run with
  no `rcond`, no regularisation, no cond probe and no fallback -- on the **default path of every
  RCWA solve**. The module holds exactly one `linalg.cond` call and it is unrelated
  (`rcwa/oned.py:607`).
* **Physics risk.** The same C13 class. Compounded by role: **RCWA is one of PMM's two in-repo
  independent oracles.** If the oracle can draw a build-dependent answer, every cross-check in
  this campaign is unsound.
* **Perf/memory risk.** As N-2: probe cost must be cheap or cached.
* **Oracle.** Dual-build agreement; an independent QR/SVD re-solve scored on residual; a
  well-conditioned control that must stay bit-identical.
* **Dependencies.** None; blocks the *credibility* of S1's cross-checks and of R-3.
* **Agent-cycles.** 2 AC (scoped to the two named sites plus a census; not a module-wide sweep).
* **Verdict.** **GO -- bundle with N-2** as one "conditioning" mission across both solvers. Same
  technique, same evidence protocol, one reviewer.

### D-1. Roadmap and in-code documentation corrections  **[NEW]**

* **What.** Fix, in `ROADMAP_PMM_PER_LAYER_GRIDS_2026_07_28.md` and the named source files, every
  item in S1.2: S-1 (2-D basis and uniform segments), S-2 (no folds in the pure stack), S-3 (the
  cost table), S-4 (`min_feature` not inert -- also in the impl report and in the `layer_grids`
  docstring), S-5 (five window sites), S-6 (conical cap), S-7 (`'li'` fold location; the
  `symmetry=True` "opt-in" line at `rcwa/_core.py:1483-1485`), S-8 (LEV-4 half-space-only;
  `fff_nv` PMM port separable-only).
* **Verdict.** **GO.** Zero risk, and it is the memory of record. 0.5 AC, folded into whichever
  mission touches each file.

---

## 3. Dependency graph

```
N-4 (window helper) ---> T3-1 (window width) ---> T3-2 (ns band)
                    \--> N-6 (min_feature contract) --^

N-2 + X-1 (conditioning, both solvers) ---> [credibility of every oracle below]
        \--> T3-3 (conical cap)              \--> R-3 (RCWA fold, deferred)

N-3 (hoist / vectorise / de-kron) ---> S3 prepare() (deferred)
                                  \--> de-risks T3-1's 4.6x

T3-7 (lattice quantisation) ---\
N-1 (non-uniform 2-D segments) --+--> S1 (2-D per-layer)  [deferred, campaign-gated]
N-5 (C2/C4 fold, pure 2-D)  ---/

T3-6 (covariant taper spike) --- independent; oracle = the now-affordable ns-converged staircase
T3-5 (taper-aware default)  --- independent
T3-4 --- blocked, no work
P-1, P-2 --- deferred, no work
```

---

## 4. Mission plan for 5.33.0

Five missions. Each is one subagent brief; missions M1-M3 are sequential in *review*, but M1/M4
and M2/M3 can run concurrently (different files) -- respect the 5-concurrent-subagent cap.

### M1 -- "Conditioning: no solver draws an arbitrary answer" (N-2, X-1, T3-3)  -- 4.5 AC

Instrument and harden the five PMM per-layer solves, the two named RCWA sites, and the conical
far-field cap.

**Acceptance gates**
| axis | gate |
|---|---|
| accuracy | Every screened solve's residual against its own original equations is <= the unscreened one, per site, measured. Ties return historical bits (5.32.1 contract). |
| conservation | `\|R+T-1\|` on the lossless staircase and `sum(A)` vs `1-R-T` on the lossy 3-layer: unchanged or better at deg 6, 8, 10. |
| both-builds | Cross-build agreement on the deep-null Jones improves by >= 2 orders on any site the census flags, and does not regress anywhere. Windows/MKL + WSL/OpenBLAS, identical test counts. |
| null control | A well-conditioned control stack (conforming grids, `cond` < 1e6) is **bit-identical** before/after, tolerance-at-0.0, both builds. |
| fail-before | One switch per behaviour change reproduces the prior library bit for bit, verified per configuration. |
| T3-3 | A fail-before test **reproduces the rank deficiency** (cond or cross-build disagreement) before the clamp, and the clamp raises with the classical path's message. |
| speed | No regression > 5% on the audit-device solve pair (idle machine, threads pinned to 1, vs `git archive` baseline). |
| memory | `tracemalloc` peak within 5% of baseline. |

### M2 -- "The window is the knob" (N-4, N-6, T3-1, T3-2)  -- 4.5 AC

One window helper; the `min_feature` contract decided by measurement; `window_halfwidth` shipped
opt-in with its measured cost table; the ns = 8-12 stationarity sweep run on the mesh.

**Acceptance gates**
| axis | gate |
|---|---|
| accuracy | N-4 is byte-identical (tolerance-at-0.0) across the full per-layer suite + the audit-device gates, both builds. `halfwidth=2` reduces `\|R+T-1\|` at deg 8 by a **measured** factor, reported with its cost; no default flips unless that factor beats its measured cost. |
| conservation | Every cell of the ns sweep reports `\|R+T-1\|` **and** `sum(A)` vs `1-R-T` alongside ER. An ER-only table is rejected. |
| both-builds | The chosen stationary setting is stationary on **both** builds; if not, it is quoted as a band. |
| null control | A conforming stack (mortar bypassed) is bit-identical under every window and `min_feature` change. A vertical untapered stack is unaffected. |
| era-pin | Where T3-1/T3-2 legitimately move pinned numbers, the original assertion is era-pinned **verbatim** with a live comparative sibling -- the C9/C10 precedent. |
| oracle | The stationary value is adjudicated against the shared grid at a validated `min_feature`, and against the manufactured-solution mortar test. |
| speed/memory | Cost of `halfwidth=2` measured, not estimated: wall time and `tracemalloc` peak per solve, both builds. |

### M3 -- "Per-layer, efficiently" (N-3)  -- 2 AC

Hoist cross-masses on the three unhoisted paths; vectorise `_lagrange_eval`; de-kron both mortars;
`inv -> solve` in `_redheffer_star_rect`.

**Acceptance gates**
| axis | gate |
|---|---|
| accuracy | Hoist + vectorise: **bit-identical**, tolerance-at-0.0, both builds. De-kron + `inv->solve`: within tolerance, with the mortar-reduction identity and conforming bit-exactness pins both still green. |
| conservation | `\|R+T-1\|` unchanged or better at deg 6/8/10 on the lossless staircase. |
| speed | Measured against a `git archive` baseline, idle machine, threads pinned: report per-solve wall time on (a) a 3-layer synthetic, (b) the 38-layer audit device, (c) a 10-point LC sweep -- the last being the case S3 exists to serve. |
| memory | Largest-array census before/after; `tracemalloc` peak and RSS delta per solve. The de-kron claim is accepted only if the peak drops measurably. |
| both-builds | All of the above on both. |
| fail-before | One switch for the de-kron/`solve` group. |

### M4 -- "Small, self-contained, shared-path" (T3-7, T3-5, D-1)  -- 2.5 AC

Lattice quantisation as an opt-in alternative to the pairwise snap; the taper-aware `min_feature`
recommendation + warning (not a default flip); all documentation corrections from S1.2.

**Acceptance gates**
| axis | gate |
|---|---|
| accuracy | Per-material total-width preservation re-measured (not cited) at matched `Delta` vs matched `min_feature`; degree spread on the audit device at both. |
| conservation | `\|R+T-1\|` at both settings, both builds. |
| null control | T3-5's heuristic must be **silent** on a labelled set of known-vertical and known-sheared stacks (the R-1b precedent: a false pathology claim is worse than silence), and fire on the labelled tapered ones. |
| default | `min_feature`'s default is **not** changed. Quantisation ships opt-in. |
| docs | Every S1.2 correction landed, each with its `file:line`. |

### M5 -- "Two spikes, no code" (N-1 de Rham feasibility, T3-6 covariant taper)  -- 2 AC

Analysis-only. N-1: does the de Rham property `d(Btilde) subset span(B)` survive non-uniform
segment widths, and does the Bloch periodic hat stay C0? Prototype at ~1e-15 on a scalar 1-D
`Basis1D`, no library change. T3-6: derive the first-order-in-`tan(phi)` trapezoid correction from
the literature (papers first), state whether it collapses to a single convection-like term.

**Acceptance gates**
| axis | gate |
|---|---|
| N-1 | de Rham residual measured on >= 3 non-uniform partitions (mild, strong, adversarial ratio) and on the uniform control. Go/no-go stated as a number. Uniform-boundary byte-identity demonstrated in the prototype. |
| T3-6 | A written derivation with the obstruction stated explicitly, plus a named oracle (the ns-converged staircase) and the error order it would be validated to. Go/no-go. |
| deliverable | Two go/no-go memos appended to this plan. No library change, no test change. |

---

## 5. What is explicitly NOT in 5.33.0

| item | reason |
|---|---|
| S1 (2-D per-layer) | Campaign-gated by the roadmap's own condition; and N-1's spike may change its shape. |
| N-5 (pure 2-D C2/C4 fold) | Ships with S1, or standalone later as a pure perf win. |
| S3 (`prepare()` per-layer) | Re-assess after M3 measures how much of the win the geometry cache already delivers. |
| R-3 (RCWA fold coverage) | Sequence after X-1 so it is not built on an unguarded solve. |
| R-1, R-2, R-4, R-5 (RCWA), P-1, P-2 | Research-class or superseded; no consumer is waiting. |
| T3-4 (grid-quality detector) | Correctly blocked on an unresolved mechanism. |
| Any `min_feature` default change | Silent results change for every existing user; the parent audit declined it deliberately. |
| numba, GPU modal eig, staggered near-Wood regularization | On the v5.14 roadmap's explicit REJECTED list. Do not re-litigate. |

---

## 6. Standing rules, applied to this campaign

These are the just-finished campaign's hard-won rules, restated as this campaign's obligations.

1. **Right functional form, not more resolution.** N-1 (non-uniform segments) and T3-3 (the correct
   capacity bound) are form fixes. T3-1's `halfwidth=2` is *more resolution* -- so it ships as a
   measured opt-in knob, not a default, unless it earns its 4.6x.
2. **Every claim measured, with a null-floor control.** Every mission table above names its null
   control: a conforming stack (mortar bypassed), a vertical untapered stack, a well-conditioned
   solve, a uniform-boundary basis.
3. **Both-BLAS-builds for any numeric behaviour claim.** Windows/MKL + WSL/OpenBLAS, identical test
   counts, on every mission. Ruff via WSL before every push.
4. **Check every solve for conditioning.** The census is M1's first deliverable, and it covers the
   nine PMM sites and the two RCWA sites named in S1.2/S1.3 -- not a spot check.
5. **Conservation scored alongside accuracy.** `|R+T-1|` and `sum(A)` vs `1-R-T` appear in every
   results table. Note the standing caution that `R+T+A` can be tautological: where the budget is
   assembled from the same partial cascades as the answer, add the **independent flux oracle** --
   Poynting flux integrated on a z-plane from `internal_field` vs the far-field `R`/`T`.
   ER alone is rejected as a scoring metric (it is the *sensitive* observable, not an independent
   one).
6. **Comparative-envelope assertions only.** The existing suite carries absolute bars
   (`e6 < 5e-4`, `e10 < 2e-5`, `errs[1] < 5e-4`, `abs(...) < 5e-3`). They are physics-set with
   headroom, but M2 must re-examine each against both builds and convert any that is
   BLAS-magnitude-dependent into a comparative envelope (`e10 < e6` is already the right shape).
7. **Era-pin with the original assertion verbatim** wherever T3-1/T3-2 legitimately move behaviour,
   each with a live comparative sibling.
8. **Fail-before switches** on every behaviour change, verified per configuration, not in aggregate.
9. **Adjudicate against an independent oracle before re-pinning.** Named per item in S2. Note the
   ordering constraint: X-1 must precede any adjudication that leans on RCWA.
10. **Memory measured.** `tracemalloc` peak + RSS delta + a largest-array census on every perf
    claim. Prefer separable/marginal constructions -- S1's Kronecker cross-mass is a *requirement*,
    not an optimisation, and a dense 2-D cross-mass is a rejected design.
11. **Independent oracles share no code with the thing under test.** For PMM: the shared-grid path,
    the pure 1-D entry points, `PMM2DStackHybrid`, the RCWA siblings, analytic/homogeneous limits,
    the manufactured-solution mortar test, and inkstone/grcwa externally.
12. **Papers first.** T3-6 and P-1 both have named references (Granet 2017; Edee & Granet 2024,
    josaa-41-9-1803); extract the formulation before re-deriving.
13. **Bidirectional adversarial review.** Refute the claimed successes *and* prove out the claimed
    failures -- including the claims in this document.

**Compute discipline.** T3-2's sweep is ~1800 solves at 28-175 s: run it on the mesh
(`mesh-run`), not locally, and check free RAM and orphan `python` processes before launching --
last month's "silent crashes" were memory pressure, not the solver.

**CI discipline.** Regenerate `.test_durations` before the release tag; the stale file has already
timed out a release-verify shard at the 30-minute cap once.

---

## 7. Measurement protocol (binding on every mission)

* **Speed**: idle machine, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` (the env the
  slow gate uses), median of >= 5 runs, against a `git archive` baseline of the same tree -- never
  against a remembered number.
* **Memory**: `tracemalloc` peak **and** process RSS delta, plus an explicit census of the largest
  live arrays at peak. Report the census, not just the total.
* **Accuracy**: report the metric, its null floor, and the instrument's own band. Where a metric
  can be blind to manufactured energy, report conservation next to it in the same table.
* **Byte-identity**: assert by tolerance at 0.0 on the max absolute difference, not
  `np.array_equal`.
* **Cross-build**: report both numbers, always. A single-build number is not evidence.

---

## 8. Release target

**5.33.0**, containing M1-M4 and M5's two memos. Rationale:

* The version is 5.32.1 and the deprecation horizon was advanced to `'5.36'` in 5.32.0, so nothing
  comes due.
* M1 is a **fix** in a class the library treats as P1 (silent-wrong / build-dependent), M2 closes
  the last accuracy caveat on a **shipped opt-in surface**, M3 is a measured perf/memory change,
  M4 adds an opt-in knob. That is a MINOR under the repo's versioning rule (patch bumps for small
  things, bundle more per release, reserve MINOR for milestones) -- and "the per-layer surface is
  accuracy-closed and conditioning-hardened" is a milestone.
* If M2's `window_halfwidth` default does **not** flip and M3's de-kron turns out bit-identical,
  the release is a legitimate **5.32.2** instead. Decide on the measurements, not in advance.
* S1 (2-D per-layer), if green-lit, is **5.34.0** and its own campaign -- it is a new capability
  with a new API surface and the `PMM2DStack` alias cutover behind it.

Release mechanics reminders: DRAFT releases never trigger `publish.yml`; tag the version's **own**
release commit, not `main` HEAD; push an annotated tag and use `--verify-tag` (a `--target
<short-sha>` will fail); regenerate `.test_durations` first.

---

## 9. Summary table

| id | item | verdict | AC | mission |
|---|---|---|---|---|
| N-2 | per-layer mortar/star/lstsq conditioning | **GO** | 2 | M1 |
| X-1 | RCWA `inv(apb)` + 4N generalized interface conditioning | **GO** | 2 | M1 |
| T3-3 | conical per-layer far-field cap (promoted P3->P2) | **GO** | 0.5 | M1 |
| N-4 | one window-grid helper (5 copies today) | **GO** | 0.5 | M2 |
| N-6 | `min_feature` not inert on the per-layer path | **GO** | 1 | M2 |
| T3-1 | +/-2-neighbour window, measured, opt-in | **GO** | 1.5 | M2 |
| T3-2 | ns 8-12 stationarity band | **GO** | 1.5 | M2 |
| N-3 | hoist cross-mass / vectorise / de-kron / `inv->solve` | **GO** | 2 | M3 |
| T3-7 | lattice wall quantisation | **GO** | 1 | M4 |
| T3-5 | taper-aware `min_feature` (warning, not default) | **GO** | 1 | M4 |
| D-1 | roadmap + in-code corrections (8 items) | **GO** | 0.5 | M4 |
| N-1 | non-uniform 2-D `Basis1D` segments | **GO (spike)** | 1 (+2) | M5 |
| T3-6 | covariant taper metric, first order in `tan(phi)` | **GO (spike)** | 1 | M5 |
| S1 | PMM2D per-layer grids + Kronecker mortar | **DEFER** (campaign-gated) | 8-12 | 5.34 |
| N-5 | C2/C4 fold for the pure 2-D stack | **DEFER** | 3 | 5.34 |
| S3 | `prepare()` per-layer | **DEFER** (re-assess after M3) | 3 | -- |
| R-3 | RCWA fold coverage (OOP, `fff_nv`) | **DEFER** (after X-1) | 2 | -- |
| T3-4 | grid-quality observable | **DEFER** (blocked, correctly) | 0 | -- |
| P-1 | native 2-D slant (Edee & Granet) | **DEFER** | 0 | -- |
| P-2 | Li rules, hybrid crossed anisotropic cell | **DEFER** (superseded) | 0 | -- |
| R-1 | mu / bianisotropic | **DEFER** (research) | 0 | -- |
| R-2 | hex / oblique lattices | **DEFER** (research) | 0 | -- |
| R-4 | `sigma_px` physical scaling | **DEFER** (dormant) | 0 | -- |
| R-5 | K-matrix micro-wins | **DEFER** (P3) | 0 | -- |

**5.33.0 total: ~14.5 AC across 5 missions.**
