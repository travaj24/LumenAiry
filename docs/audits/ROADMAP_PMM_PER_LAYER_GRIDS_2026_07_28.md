# ROADMAP — PMM per-layer grids: 2-D extension, remaining quality items, and `prepare()`

**Date:** 2026-07-28 · **Branch:** `fix/pmm-union-grid-conditioning`
**Parents:** `AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` (the defect),
`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` (the 1-D implementation)

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

1. **Per-layer 2-D grids.** Each layer's `(Nx_i, Ny_i)` segmentation from its own cell walls, plus the
   two neighbours' (the interface-conforming window — the 1-D lesson that own-walls-only FAILS, measured
   at 75–83% spread, applies verbatim; do not re-learn it).
2. **The 2-D cross-mass.** `C[i,j] = ∬ φ_i^(a) φ_j^(b) dx dy` over the pairwise union of two rectangular
   cell partitions. The union of two axis-aligned rectangle partitions is a rectangle partition
   (`O(Nx_a+Nx_b) × O(Ny_a+Ny_b)` cells); tensor-product Gauss (deg+2 per axis) is exact, and the 1-D
   `_lagrange_eval` applies per axis. Straightforward but must be written carefully for the wrap.
3. **The mortar interface.** Identical algebra to 1-D (`_interface_smatrix_mortar` /
   `_interface_smatrix_general_mortar`) — the field stacking is 2-component `(Ex, Ey)` over the 2-D nodal
   set, so the blockwise `kron(I₂, ·)` pattern carries over unchanged. The rectangular Redheffer already
   exists.
4. **Half-spaces + far field per end grid** — `_sem_fourier_projection`'s 2-D counterpart per grid.
5. **The symmetry fold.** The pure stack's even-parity/C4 folds assume the shared grid; per-layer folds
   must be per-grid (or v1 ships without folds, at 2–8× cost — acceptable for a first cut).

### 1.3 Effort, risks, gates

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
| T3-1 | **Mortar residual** (energy ~1e-4 @ deg6 → ~1e-6 @ deg10, spectral) | Acceptable for ER work at deg ≥ 8. If a use case needs 1e-10: (a) widen the window to ±2 neighbours (removes the leading non-conformity; cost ~(5/3)³ per eig), or (b) enrich each grid with one element at third-neighbour wall positions only. Measure (a) first — it is a one-line change to the window loop. |
| T3-2 | **High-`n_slice` stress band** (deg spread up to 5.6% at ns=8) | The window unions acquire thin elements as slices refine. Two levers, both unexplored: window-local `min_feature` (already plumbed — it applies inside `_pmm_union_grid` per window) and the ±2-neighbour window. Sweep both at ns=8–12 on the audit device; pick the setting where the answer is stationary in (degree, min_feature, ns) simultaneously. |
| T3-3 | **Per-layer conical far-field order cap** uses the full-union capacity estimate computed before the branch | Latent only (ffo21 sits far below any real capacity). Tighten to `min(end-grid capacities)` when touching the conical path next. |
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
