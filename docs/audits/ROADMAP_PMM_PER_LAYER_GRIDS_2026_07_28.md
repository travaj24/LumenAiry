# ROADMAP — PMM per-layer grids: 2-D extension, remaining quality items, and `prepare()`

**Date:** 2026-07-28 · **Branch:** `fix/pmm-union-grid-conditioning`
**Parents:** `AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` (the defect),
`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` (the 1-D implementation)
**Updated 2026-08-03:** §5–§6 make this THE consolidated PMM+RCWA solver roadmap.
Every item was re-verified against the code on this date; `docs/pmm_roadmap_v5_14.md`
and `docs/rcwa_roadmap_v5_14.md` are superseded (banners added there).

---

## 0. Where the 1-D surface stands (context for this roadmap)

`PMMStack(layer_grids="per-layer")` now covers, all gated and pinned in
`tests/unit/test_pmm_per_layer_grids.py` (11/11):

| surface | status | pin |
|---|---|---|
| classical solve (φ=0), in-plane vertical | **shipped** | bit-exact on conforming; 91%→5–6% degree spread on the audit device |
| conical solve (φ≠0) | **shipped** | bit-exact on conforming; 1.4–5.8% vs converged shared OOP |
| `retain_internal` → `internal_field` / `layer_absorption` | **shipped** | <1e-8 fields / <1e-10 absorption on conforming; ΣA ≈ 1−R−T to 5e-3 non-conforming |
| `solve_vs_wavelength` (dispersive incl.) | **shipped** | <1e-14 vs per-λ solve and vs the shared sweep on conforming |
| slanted layers (general fwd/back cascade) | **shipped** | <1e-9 vs shared general on conforming; energy decays 5e-3→5e-4 with degree |
| out-of-plane tensors | **shipped** | same gates as slant |
| JAX twin (differentiable) | **shipped** | forward <1e-10 vs NumPy; `jax.grad` vs FD to 5e-5 |
| `prepare()` (material-key sweeps) | **open** — §3 | direct per-layer sweep is the working substitute |
| `stabilize='slices'` | N/A by design | nothing to perturb; raises with explanation |

Not per-layer (and not planned): the **covariant uniform-slant** spectral route
(single-frame construction — per-layer grids have no meaning there).

---

## 1. PMM2DStack — the 2-D extension (the headline item)

### 1.1 Current architecture, assessed

- **`PMM2DStackPure`** (`stack2d_pure.py`) carries the 1-D union-grid design lifted to 2-D — and, unlike
  1-D, it **enforces** it: layers whose `(Nx, Ny)` cell segmentations differ **raise** ("the union-grid
  constraint of the pure stack"). So the 2-D pure stack does not have the 1-D silent-wrong pathology; it
  has a **capability gap**: a 2-D tapered staircase (pillar widths changing per slice) is simply
  *unrepresentable* unless every slice is drawn on one common grid — which recreates exactly the 1-D
  collision geometry, now in two dimensions and at `O((NxNy)³)` eig cost.
- **`PMM2DStackHybrid`** couples layers through a Fourier projection (no union-grid constraint) but pays
  the documented FMM floor — the same trade as RCWA.

**Why this matters for the campaign:** the 2-D / C4-symmetric metasurface is the open frontier for a
true dual-cut (in-plane ≡ OOP by symmetry) out-coupler. Any such device with tapered pillars will need
sliced 2-D geometry — precisely the case the pure stack cannot express and the hybrid blurs.

### 1.2 What a 2-D per-layer implementation needs

The 1-D machinery transfers piece by piece; nothing is conceptually new, but two pieces grow real teeth:

> **CORRECTION S-1 (M4, 2026-08-04) — items 1 and 2 below are WRONG, and they invert the shape of
> this item.** The 2-D pure stack does **not** use a nodal Lagrange/GLL basis and its segments are
> **uniform**. `PMM2DStackPure` runs on the Granet-2023 staggered *modified-Legendre* tensor basis
> (`lumenairy/elements/pmm/twod_staggered.py:148` `Basis1D`, doc header lines 10-21), whose
> `__init__` sets `self.xb = np.linspace(0.0, self.d, self.N + 1)` (`twod_staggered.py:174`, comment
> `# segment boundaries on the eps walls (Eq.31, uniform)`) with a **scalar** jacobian
> `self.J = 0.5 * self.h`, `h = d/N` (`twod_staggered.py:171-172`) applied as ONE scale factor for the
> whole period (`twod_staggered.py:261-266`, `341-346`, `566-568`, `641-643`). `_lagrange_eval` exists
> only on the 1-D shared-grid path (`_core.py:3609`, used at `3642`, `3688-3689`) and has **zero**
> occurrences in `twod_staggered.py` / `stack2d_pure.py`; those files' only "GLL" mentions are explicit
> disclaimers (`twod_staggered.py:911`, `stack2d_pure.py:115`). `eps` is constant per lattice cell
> (`twod_staggered.py:364`, consumed at `409-416`).
>
> **What follows, and it changes what to build:**
> * There are no per-layer *wall positions* in 2-D — only a per-layer *segment count* on a uniform
>   lattice. The 1-D wall-collision pathology **cannot occur** here; a uniform lattice has no
>   near-coincident walls. §1.2 reaches the right conclusion ("a capability gap, not the silent-wrong
>   pathology") by the wrong route.
> * The real 2-D blocker for a tapered pillar is **wall representability on a uniform lattice**. A 2°
>   taper at `n_slice = 6` over 310 nm moves a wall ~1.8 nm per slice; representing that exactly on a
>   common uniform lattice of a 700 nm period needs `Nx ~ 390`, i.e. `eigdim = 2*(Nx*(M-1))^2 = 1.5e7`
>   [A] — not large, *impossible*.
> * Therefore **a mortar alone does not unlock 2-D tapers.** The enabling change is non-uniform
>   segment boundaries in `Basis1D` (campaign item **N-1**); the mortar is what makes the resulting
>   per-layer grids affordable. This roadmap has the second half without the first.

1. **Per-layer 2-D grids.** ~~Each layer's `(Nx_i, Ny_i)` segmentation from its own cell walls~~ — see
   S-1: there are no per-layer cell walls in the pure stack, only a per-layer segment COUNT on a
   uniform lattice. Read as: each layer's own `(Nx_i, Ny_i)` *resolution*, plus the two neighbours'
   (the interface-conforming window — the 1-D lesson that own-walls-only FAILS, measured
   at 75–83% spread, applies verbatim; do not re-learn it).
2. **The 2-D cross-mass.** `C[i,j] = ∬ φ_i^(a) φ_j^(b) dx dy` over the pairwise union of two rectangular
   cell partitions. The union of two axis-aligned rectangle partitions is a rectangle partition
   (`O(Nx_a+Nx_b) × O(Ny_a+Ny_b)` cells); tensor-product Gauss (deg+2 per axis) is exact. ~~and the 1-D
   `_lagrange_eval` applies per axis~~ — see S-1: `_lagrange_eval` is the 1-D nodal-Lagrange evaluator
   and has no counterpart in this basis; the per-axis quantity is the modified-Legendre `Basis1D` set.
   Straightforward but must be written carefully for the wrap. **NB the dense form is a rejected
   design**: a dense `q_a^2 x q_b^2` cross-mass is 144 MB at `q = 42` per interface. It factors
   exactly as a Kronecker product of two 1-D cross-masses (both bases are tensor-product, the
   partitions rectangular) — ~1000x less memory, and the separable construction the standing rules
   require.
3. **The mortar interface.** Identical algebra to 1-D (`_interface_smatrix_mortar` /
   `_interface_smatrix_general_mortar`) — the field stacking is 2-component `(Ex, Ey)` over the 2-D nodal
   set, so the blockwise `kron(I₂, ·)` pattern carries over unchanged. The rectangular Redheffer already
   exists.
4. **Half-spaces + far field per end grid** — `_sem_fourier_projection`'s 2-D counterpart per grid.
5. ~~**The symmetry fold.** The pure stack's even-parity/C4 folds assume the shared grid; per-layer folds
   must be per-grid (or v1 ships without folds, at 2–8× cost — acceptable for a first cut).~~
   **CORRECTION S-2 (M4, 2026-08-04): the pure 2-D stack has NO symmetry fold at all**, so there is
   nothing to preserve and this risk is void. Zero hits for `symmetry` / `parity` / `C4` in
   `lumenairy/elements/pmm/stack2d_pure.py`; the two hits in `twod_staggered.py` (`:45`, `:900`) are
   prose about RCWA parity, not a fold. `PMM2DStackPure.__init__` (`stack2d_pure.py:123-124`) takes
   only `period_x, period_y, n_superstrate, n_substrate, n_modes, degree, n_orders`. The folds that
   exist are the **hybrid**'s (`stack2d.py:89` `symmetry="auto"`, resolved at `:114`, gated at `:629`)
   and RCWA's.
   **This cuts both ways.** The compatibility risk disappears (S1's scope shrinks), but the quoted
   "2–8× cost" is not the pure stack's number either: a fold that does not exist cannot be dropped.
   What is real is that the pure 2-D stack **pays the full unfolded eig today**, and on the C4 device
   that is the binding constraint (S-3) — so a C2/C4 fold is a **new prerequisite-class perf item
   (campaign item N-5)**, not a compatibility footnote.

### 1.3 Effort, risks, gates

> **CORRECTION S-3 (M4, 2026-08-04) — the 2-D cost/memory model was not stated, and it is the gate.**
> `_region_modes` runs a dense non-symmetric generalized eig `sla.eig(L, G)` (`twod_staggered.py:722`)
> on a `2q^2` pencil with `q = N*(M-1)` (`Basis1D.dim` assigned at `twod_staggered.py:249`; chain
> `:379` `q = bx.dim`, `:424` `qq = q*q`, `:442`/`:536` `(2*qq, 2*qq)`, `:552` `dimtot = 2*qq`).
> Arithmetic [A] with `n = 2*(N*(M-1))^2`, bytes `= 16 n^2` per matrix, `~30 n^3` flop for `zggev`:
>
> | Nx=Ny | M | eig dim | GB per matrix | flop |
> |---|---|---|---|---|
> | 3 | 8 | 882 | 0.012 | 2.1e10 |
> | 4 | 8 | 1568 | 0.037 | 1.2e11 |
> | 6 | 8 | 3528 | 0.185 | 1.3e12 |
> | 7 | 8 | 4802 | 0.344 | 3.3e12 |
> | 8 | 8 | 6272 | 0.586 | 7.4e12 |
> | 13 | 8 | 16562 | 4.09 | 1.4e14 |
> | 25 | 8 | 61250 | 55.9 | 6.9e15 |
>
> `sla.eig` holds several such arrays live. A common-grid 2-D taper at `ns = 6` lands at `Nx ~ 13-25`
> — **4–56 GB per matrix**. Per-layer windows put the same device at `Nx ~ 7` (0.34 GB/matrix):
> feasible but heavy. A C4 fold would take it to ~0.02 GB and `/64` flop. **The 2-D item's real
> acceptance gate is a memory budget, not an accuracy number.**
>
> Two further facts this section omits: `PMM2DStackPure` also **requires `Nx == Ny`**
> (`stack2d_pure.py:165-170`, plus a bare `assert` in `Granet2DTransverseE.__init__`,
> `twod_staggered.py:379`, which `python -O` strips) — so the y-invariant cross-check against the 1-D
> per-layer result, proposed as a gate below, requires padding the uniform axis into equal segments,
> which inflates `q` and the eig. And the bare name **`PMM2DStack` is a transitional alias to the
> HYBRID** (`stack2d.py:1184-1210`), scheduled to be repointed at the pure stack once it reaches
> feature+validation parity — so this item carries an API-cutover consequence not mentioned below.

**Effort: ~2–4 weeks** (vs 1–3 for 1-D; the cross-mass bookkeeping and the fold interaction are the
growth). Risks: (a) the mortar residual scales with the non-conforming interface *measure* — in 2-D
that's a 1-D curve set rather than points, so expect the residual band to sit somewhat higher at equal
degree; measure before promising; (b) eig cost per layer is `O((2NxNyp²)³)`-class — per-layer windows are
the difference between feasible and not, which is the point.

**Gates (transplant the 1-D set):** conforming-stack bit-exactness vs `PMM2DStackPure`; a synthetic 2-D
staircase (pillar taper) degree-convergence at default `min_feature`; lossless closure vs degree; the
C4 device cross-checked against the 1-D per-layer result in its y-invariant limit.

**Payoff:** the only laterally-exact route to sliced 2-D tapered metasurfaces, and the enabling
capability for the dual-cut C4 design campaign.

---

## 2. Tier-3 quality items (carried from the audits, with today's status)

| id | item | status / plan |
|---|---|---|
| T3-1 | **Mortar residual** (energy ~1e-4 @ deg6 → ~1e-6 @ deg10, spectral) | Acceptable for ER work at deg ≥ 8. If a use case needs 1e-10: (a) widen the window to ±2 neighbours (removes the leading non-conformity; cost ~(5/3)³ per eig), or (b) enrich each grid with one element at third-neighbour wall positions only. Measure (a) first. **CORRECTION S-5 (M4, 2026-08-04):** "it is a one-line change to the window loop" was FALSE when written — the window comprehension `js = [j for j in (i-1, i, i+1) if 0 <= j < nlay]` was duplicated verbatim at FIVE sites (`stack.py` x3, `conical.py`, `_jax_stack.py`), so widening without first extracting a helper would have put one physics parameter in five files (the duplication-kills lesson). It is TRUE NOW: campaign item N-4 has landed the single helper `_core._perlayer_window_grids(layer_segments, min_feature_frac, halfwidth=1)` (`_core.py:3537`, body `:3592-3606`, `js = [j for j in range(i-hw, i+hw+1) ...]`), called from all five former sites, with the public knob `PMMStack(window_halfwidth=1)` (`stack.py:172`, validated `:185-200`) and `pmm_conical_jones(window_halfwidth=1)` (`conical.py:141`). Also note the **cost is ~4.6x on the WHOLE solve**, not "per eig" loosely: the eig is ~97% of a region solve. |
| T3-2 | **High-`n_slice` stress band** (deg spread up to 5.6% at ns=8) | The window unions acquire thin elements as slices refine. Two levers: window-local `min_feature` (already plumbed — it applies inside `_pmm_union_grid` per window) and the ±2-neighbour window. Sweep both at ns=8–12 on the audit device; pick the setting where the answer is stationary in (degree, min_feature, ns) simultaneously — **and if none is, quote a BAND rather than force a false convergence**. Score `\|R+T-1\|` and `sum(A)` vs `1-R-T` alongside ER; ER alone is rejected (a ratio with a deep null in the denominator — the *sensitive* observable, not an independent one). **CORRECTION (M4, 2026-08-04): "both unexplored" is stale** — the ±2 window is now MEASURED and recorded at `_core.py:3551-3560`. |
| T3-3 | ~~**Per-layer conical far-field order cap** uses the full-union capacity estimate computed before the branch~~ | ~~Latent only (ffo21 sits far below any real capacity). Tighten to `min(end-grid capacities)` when touching the conical path next.~~ **CORRECTION S-6 + SHIPPED (M1, 2026-08-04).** "Latent only" was WRONG: `cap` was computed from `nU`, the full-union cell count, *before* the per-layer branch and never re-clamped, while the half-spaces were built on the WINDOW grids (`n_glob ~ nU/6` on the audit device). The `m_prop > cap` raise therefore over-stated capacity and could not fire; `_sem_fourier_projection` then built a projector with more Rayleigh orders than the grid has nodes, `Hsup` went rank-deficient and `lstsq(..., rcond=None)` returned a build-dependent null-space draw — **silently**, because a null-space component of `cinc` is invisible to `Hsup` and leaves `R + T = 1` intact. Verbatim the C13 mechanism shipped in 5.32.1, and the ONLY sibling that got it wrong (classical per-layer `stack.py:1809-1819`, the sweep `stack.py:3002-3011` and the JAX twin `_jax_stack.py:502-525` all clamp to the window half-spaces and raise). NOW FIXED: `conical.py:249-256` clamps to `(min(n_glob_sup, n_glob_sub) - 1) // 2` on the per-layer branch, the window block is hoisted above the cap (`conical.py:217-247`), fail-before switch `PMM_CONICAL_PERLAYER_ORDER_CAP` (`conical.py:67`). |
| T3-4 | **R-2: a grid-quality observable** | Still **blocked on the unresolved mechanism** — two hypotheses (J→0 slivers; taper/coat resonance) were measured and refuted. Do not ship a detector premised on an unconfirmed story. The practical guard exists: the `min_feature`-perturbation consensus (R-1) plus per-layer as an independent cross-check. Revisit only if the mechanism is ever pinned. |
| T3-5 | **R-3: taper-aware `min_feature` default** (shared path) | Derive from `(thickness/n_slices)·tan(sidewall)` when a taper recipe is recorded; warn when geometry-built stacks look tapered (wall-offset statistics). Low urgency now that per-layer exists, but cheap and protects shared-path users. |
| T3-6 | **R-5: covariant taper-metric layer** (general trapezoid, no staircase) | Research-adjacent: the trapezoid's `u = x/w(z)` metric is z-dependent, so the modal coefficients do not collapse to one eigenproblem (unlike shear). Candidate approaches: a z-expansion of the metric (perturbative in `tan(sidewall)` — 2° is small!), or a Magnus/product-integral treatment of the z-varying generator. The 2° smallness argument deserves a feasibility spike: at first order in `tanφ` the correction may be a single convection-like term. **Highest science value; least certain.** |
| T3-7 | **R-7: lattice wall quantisation** | Deterministic, bounded-displacement replacement for the cascading pairwise snap. Sizing rule (from the audit): Δ set by the separations to REMOVE (~1 nm class), not the features to keep; finer lattices merge fewer pairs. Mostly superseded by per-layer, but still improves the shared path and `_pmm_union_grid`'s reproducibility. Small, self-contained. |

---

## 3. `prepare()` per-layer — design sketch (deferred deliberately)

**Why deferred:** `_PreparedPMMStack` is coupled to the shared union grid throughout (union-cell-keyed
eig caches with audited LRU semantics, P3-32), and the practical need is already met — a direct per-layer
LC sweep costs ~2.7 s/point at deg 8, beating the shared prepared sweep.

**The per-layer design is cleaner than the shared one when it is built:**

- Geometry (window grids, masses, cross-masses, Rayleigh projectors) is **material-independent** — cache
  it once per stack, forever. This is most of what the shared prepared object recomputes or LRU-caches.
- A material-key override re-eigs **only that layer's small grid** (the per-layer eig is ~6× smaller in N
  than the shared one) and rebuilds **its two interfaces** (mortar matrices are geometry-only and cached;
  only the `solve`s against new W/V rerun).
- Cache keying: `(layer_index, eps-bytes-of-that-layer's-window-row, wavelength, angle)` — no union-cell
  content keys, no cross-layer invalidation.
- Contract to preserve: byte-identical re-solve after eviction; LRU bounds sized by per-layer entry size
  (~36× smaller than shared entries at production n_glob).

**Effort: ~1 week**, mostly the cache-correctness tests. Do it when an LC-sweep-heavy campaign makes the
2.7 s/point loop feel slow, not before.

---

## 4. Suggested sequencing

1. **T3-1/T3-2 measurement pass** (days): ±2-neighbour window + window-local `min_feature` at ns=8–12 —
   closes the last accuracy caveats on the 1-D surface.
2. **PMM2D per-layer** (§1, weeks): gated by the dual-cut C4 campaign decision — build it when that
   campaign is green-lit, not speculatively.
3. **T3-6 feasibility spike** (days, timeboxed): the first-order-in-`tanφ` covariant taper. If the 2°
   smallness pans out, it obsoletes the staircase entirely for this fab process.
4. **`prepare()` per-layer** (§3) and the small items (T3-3, T3-5, T3-7) opportunistically.

None of the §5/§6 items below gate this sequence; the research-class ones are unscheduled and the
rest slot in opportunistically.

---

## 5. Carried-over PMM items from the v5.14 roadmap (verified 2026-08-03)

`docs/pmm_roadmap_v5_14.md` is superseded by this section. Of its seven items, five have shipped
(JAX cell twin; 1-D homogeneous-region eig share; `PMM2DStack` out-of-plane layers; graded-profile
helper; the metal-TM "floor" resolved as mutual unconvergence). Its explicitly-REJECTED list (numba,
GPU modal eig, staggered near-Wood regularization) stands — do not re-litigate. Two items remain:

| id | item | status 2026-08-03 |
|---|---|---|
| P-1 | **Native 2-D slant** — Edée & Granet 2024 (josaa-41-9-1803) crossed-slanted coordinate map + Gegenbauer basis | Open, research-grade. Interim: `PMM2DStack.add_tapered_pillar` z-staircase — which the §1 per-layer extension makes laterally exact per slice, and whose small-angle regime the T3-6 covariant-taper spike would treat without slicing. Sequence AFTER §1 and T3-6: their outcomes bound how much a native map is still worth. |
| P-2 | **Li-1997 mixed inverse rules for the 2-D tensor diagonal slots** (Eqs. 8/9 + 31) | Largely superseded: v5.21 shipped the Li-2003 `L2·L1` full-tensor factorization as `formulation='fff_nv'` in `rcwa_jones_2d` (incl. CROSSED cells) and ported the separable version to the hybrid PMM — the first correct in-plane inverse-rule treatment there. Residual gap: the **crossed anisotropic cell in the hybrid PMM** stays Laurent-floored (~1e-3). The pure 2-D stack has no Fourier floor, so §1 is the strategic answer; treat this as a hybrid-only nicety. |

---

## 6. RCWA — the verified-current roadmap (2026-08-03)

`docs/rcwa_roadmap_v5_14.md` is superseded by this section. Verified against the code on this date.
Shipped since that doc — recorded here so they are not re-opened:

- **`RCWAStack` out-of-plane layers**: any OOP layer promotes the cascade to the generalized
  S-matrix, including correct traced OOP stack gradients.
- **Even-parity fold (LEV-3), full scope**: extended to `RCWAStack.solve` (whole-cascade even
  sector for normal-incidence jointly centro-symmetric stacks, ~×3–4 — the "backlog A1" fast path)
  and to `'li'`; ON by default (`symmetry='auto'`) since v5.21 on most entry points.
  > **CORRECTION S-7 (M4, 2026-08-04).** Two errors in the original wording.
  > (a) **The `'li'` fold is NOT in `rcwa_jones_2d`.** That function's fold gate requires
  > `formulation == "laurent"` (`lumenairy/elements/rcwa/twod.py:1701-1704`), so `'li'`, `'fff_nv'`
  > and every OOP cell take the FULL solve there. The `'li'` fold lives in **`rcwa_efficiency_2d`**
  > (`twod.py:1082-1093`) and in the prepared 2-D class (`twod.py:1198-1210`), with the per-layer
  > `_tensor_PQ` built in `rcwa/stack.py:2495-2502`. Note the real gate is `li_ops is not None`, not a
  > formulation string: `li_ops` is set only for a non-uniform, non-JAX cell under
  > `formulation == "li"` (`twod.py:1017-1027`), and a *uniform* `'li'` cell reroutes to `'laurent'`
  > (`twod.py:1018-1019`) and so takes the laurent branch.
  > (b) **"ON by default" is not universal.** `rcwa_efficiency_2d_shapes` still defaults
  > `symmetry=False` (`twod.py:1943`), and the 1-D core has **no fold at all** — a case-sensitive grep
  > for `symmetry` over `rcwa/oned.py` returns exactly one hit and it is a prose comment
  > (`oned.py:1052`), with no `_symmetry_on` import and no fold call. `symmetry="auto"` IS the default
  > at `twod.py:693, 1265, 1379, 1459` and `rcwa/stack.py:2681, 2787`.
  > (c) A stale in-code comment said the fold was **"Opt-in (`symmetry=True`)"** five lines BEFORE
  > `_symmetry_on`, whose own docstring says `"auto"` is the default and requests the fold — the two
  > texts contradicted each other. **FIXED in this mission** at `rcwa/_core.py` (the LEV-3 comment
  > block preceding `_symmetry_on`).
- **Analytic homogeneous-layer modes (LEV-4)**: uniform layers get the analytic non-eig modes
  instead of running the eig.
  > **CORRECTION S-8a (M4, 2026-08-04): "draw from the module-level cache" is over-read.** The
  > `_cached_homogeneous_eigenmodes` LRU (`rcwa/_core.py:3559-3571`) is consulted at exactly TWO call
  > sites, both `RCWAStack`'s half-spaces, and only on a non-traced solve
  > (`rcwa/stack.py:2887-2890`; a traced source bypasses it at `stack.py:2877-2879`). Uniform
  > **interior** layers call the UNCACHED `_homogeneous_eigenmodes` (`rcwa/stack.py:2564`), as do all
  > half-spaces in `twod.py:1053, 1195, 1696, 2131`, all three `oned.py` sites (`:673, 1056, 1071`)
  > and both Berreman modules. They do get the analytic non-eig modes, so "instead of running the eig"
  > holds; the cache-hit half does not. **Any campaign claim of a cache-hit win on
  > many-uniform-layer stacks is NOT YET REALISED.**
- **`fff_nv` rework-or-retire (F2)**: RESOLVED as rework (v5.21.2) — Li-2003 `L2·L1` successive
  full-tensor factorization; uniform cells route to
  `'laurent'`; the closed-form xy-wedge field ships as its cross-check (audit M10, 2026-07-25).
  > **CORRECTION S-8b (M4, 2026-08-04): "incl. crossed cells, ported to the hybrid PMM" reads as one
  > clause covering both, and neither half is general.** The Li-2003 `L2·L1` factorization is wired
  > into **`rcwa_jones_2d` only** (OOP branch `twod.py:1729-1744`; in-plane branch
  > `twod.py:1773-1784`). The scalar pixel engine still builds the **Schuster normal-vector field**
  > (`twod.py:1042-1051`, and the twin in `prepare_rcwa_2d` at `twod.py:1328-1335`) — which is exactly
  > the code open item R-4 is about, so the two bullets are consistent only if "rework" is read as
  > *tensor-Jones path only*. The **PMM port is SEPARABLE-ONLY**: `pmm/twod_jones.py:202-223`
  > implements the separable (single-orientation) rule and `pmm/twod_jones.py:252-260` **raises** on a
  > crossed (both-axes-patterned) cell; `fff_nv` is also excluded from the PMM even-parity fold
  > (`twod_jones.py:615`). §5's P-2 states this correctly; this bullet did not.
- **GPU-DLL hygiene**: the `use_gpu` guard probes a trivial device op + `cupy.fft` import and
  re-raises the friendly RuntimeError naming the missing NVIDIA wheels.

Open items:

| id | item | class |
|---|---|---|
| R-1 | **µ / bianisotropic materials** (GAP4) — no magnetic/bianisotropic support anywhere in the module | research |
| R-2 | **Hex / oblique lattices + parallelogrammic truncation** (GAP6) — `truncation=` today is `'rectangular'`/`'circular'` on a rectangular lattice only; the "parallelogram" in the code is sheared *sidewalls* (GAP1), a different feature | research |
| R-3 | **Even-parity fold coverage for the two documented exclusions**: OOP (ezz-Schur) cells and `'fff_nv'` cells always take the full 2N solve today | perf, medium |
| R-4 | **`sigma_px` physical scaling** — the surviving sliver of F2: the smoothed-gradient NV width is pixel-fixed (1.5 px), so the field depends on sampling resolution S. Low impact post-M10 (wedge cross-check exists; uniform cells rerouted); make the width physical if `fff_nv` accuracy work resumes | accuracy, small |
| R-5 | **K-matrix assembly micro-wins** (LEV-5) — never started; small | perf, P3 |

Two positioning facts that stand: JAX remains the GRADIENT path, not the forward-speed path
(jit-warm ~1–2.4× wall at equal work; NumPy + `prepare()`/sweeps for forward scans), and the
per-layer-grids idea of §0–§1 does NOT transfer to RCWA — a global Fourier basis has no lateral
grid to decouple, so RCWA's accuracy path stays formulation rules (Li). The two solvers serve as
each other's independent cross-checks on tapered stacks.
