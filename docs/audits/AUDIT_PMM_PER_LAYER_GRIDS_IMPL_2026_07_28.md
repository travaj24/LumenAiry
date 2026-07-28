# IMPLEMENTATION — PMMStack per-layer element grids with interface mortar (audit R-6)

**Date:** 2026-07-28 · **Branch:** `fix/pmm-union-grid-conditioning` · **Status:** v1 shipped, opt-in
**Parent audit:** `AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` (R-6: *"give each layer its own
element grid, with a projection / mortar at the interfaces"*)

---

## 1. What shipped

`PMMStack(..., layer_grids="per-layer")` — an opt-in assembly mode for the **all-vertical in-plane**
1-D cascade (the campaign's case). Three design elements:

1. **Per-layer grids with interface-conforming enrichment.** Each layer's SEM operators are built from
   the union of **its own walls and its two neighbours'** (`_pmm_union_grid` over a 3-layer window), not
   the global stack union. The window is what makes the mortar spectrally accurate (§3); it also keeps
   the grid local — wall accumulation across the stack cannot form, and the global `n_slices` coupling
   (`O(n_slices^3.4)` cost, the §5.3 vise of the parent audit) is gone.
2. **An exact L2 mortar at non-conforming interfaces** (`_interface_smatrix_mortar`,
   `elements/pmm/_core.py`). Tangential-E continuity is enforced weakly against the lower grid's basis
   and tangential-H against the upper's (the classic mode-matching pairing, which keeps the system
   square for `n_a ≠ n_b`):

   ```
   A = (M_b W_b)^-1 C_ab^T W_a          E tested on grid b
   B = (M_a V_a)^-1 C_ab   V_b          H tested on grid a
   S11 = (I+BA)^-1 (I-BA)   S12 = 2 (I+BA)^-1 B
   S21 = A (I + S11)        S22 = A S12 - I
   ```

   On identical grids (`M_a = M_b = C_ab`, exact quadrature) this **reduces algebraically to
   `_interface_smatrix`** — verified symbolically in the derivation and pinned numerically
   (`test_mortar_reduces_to_plain_interface_on_identical_grids`). Mass and cross-mass are exact
   (Gauss–Legendre `degree+2`; the cross-mass integrates on the pairwise union of the two element sets,
   where near-coincident walls appear only in the **integration mesh** — harmless — never as spectral
   elements).
3. **Supporting algebra:** `_sem_mass_exact`, `_sem_cross_mass`, `_lagrange_eval`, and
   `_redheffer_star_rect` (the rcwa star sizes its identity from `A11` and requires uniform block
   sizes; per-layer cascades carry rectangular off-diagonal blocks).

Identical-grid interfaces (repeated layers, and the half-spaces, which are built on the grid of the
layer they touch) bypass the mortar and take the plain interface — this is what makes the parity
results below *exact* rather than approximate.

**v1 surface:** `solve()` on all-vertical in-plane NumPy stacks at φ=0. Conical, slant, out-of-plane
tensors, JAX, `stabilize`, `retain_internal`, `prepare()` and `solve_vs_wavelength` **raise** with the
shared-grid alternative named (`test_unsupported_combinations_raise`). The default is
`layer_grids="shared"` — nothing changes for existing users.

---

## 2. Validation gates (all numbers measured; `perlayer_gates.py` in
`Metasurface_QWP/experiments/exp21/`)

| gate | test | result |
|---|---|---|
| 1a | identical-wall 3-layer stack vs shared | **bit-exact** (`max|ΔJones| = 0.0`) |
| 1b | 2-layer different-wall stack vs shared, deg 6–18 | **bit-exact at every degree** (a 2-layer neighbour union *is* the full union — conforming) |
| 2 | the pathological W_hiER2 taper, **default `min_feature`**, deg 6/8/10 × θ 0/8/10 | degree spread **91% → 5.3–6.3%** (deg8→deg10: 0.5–2.8%); values converge onto the shared mf=1.5 nm reference (55–59 vs 49.6/54.9/54.7) |
| 2-vise | `n_slice = 4` (re-breaks the shared grid at 15.7%) | **4.6–4.9%** — the fidelity/conditioning vise is broken |
| 3 | lossless synthetic colliding-wall staircase, `R+T−1` | **1.10e-4 (deg 6) → 1.17e-6 (deg 10)** — spectral decay of the mortar residual (see §3) |
| speed | real stack, deg 8, θ8, peak+null pair | shared **97.3 s** → per-layer **5.5 s**: **17.8×** |

The headline: **the pathological stack is degree-convergent at the library-default `min_feature`** —
`min_feature` is inert by construction on this path (there is no global union to snap) — and the
n_slice ladder is now affordable (seconds per point instead of minutes-to-hours).

---

## 3. The honest residual, and the one failed intermediate

**First attempt (own-walls-only grids) FAILED Gate 2** — spread 75–83%, barely better than shared.
Mechanism, identified from the gate data: the interface field carries structure exactly at the two
adjacent layers' material walls; a grid containing only its *own* walls cannot represent the
neighbour's wall structure, and the L2 mortar then has an O(1) boundary-layer error concentrated in
the **Ex (wall-normal) channel** — measured as an Ex-channel energy defect ~100× the Ey one, fatal for
deep nulls. The **neighbour-window enrichment** is the fix: both wall sets exist on both sides of every
interface, the mortar bridges only smooth-field refinement mismatches, and Gate 2 collapsed from 75–83%
to 5–6%. This failure and its diagnosis are retained here deliberately — it is the design constraint a
future v2 (wider windows, adaptive enrichment) must respect.

**Remaining residual:** the mortar's non-conforming remainder (the third-neighbour walls absent at each
interface) leaves a lossless-energy defect of ~1e-4 at degree 6 falling to ~1e-6 at degree 10 —
spectral decay, but **not** the shared path's ~1e-14 closure. Consequences, stated plainly:

* **deg ≥ 8 is the recommended regime** for per-layer solves of deep-null devices (at deg 6 the
  residual is within reach of a ~1% null).
* The energy tripwire `_warn_stack_energy` still applies unchanged; a per-layer solve that violates
  closure beyond its thresholds warns like any other.
* For publication-grade closure (1e-10+), use the shared grid at a validated `min_feature` — the two
  paths now cross-check each other, which is precisely the independent oracle the parent audit's §7
  found missing.

---

## 4. Files

| file | change |
|---|---|
| `elements/pmm/_core.py` | `_lagrange_eval`, `_sem_mass_exact`, `_sem_cross_mass`, `_interface_smatrix_mortar`, `_redheffer_star_rect` (new section after `_pmm_union_grid`) |
| `elements/pmm/stack.py` | `layer_grids=` constructor param + validation; dispatch + `_solve_vertical_perlayer`; guards on conical/JAX/covariant/prepare/sweep; clones carry the flag |
| `tests/unit/test_pmm_per_layer_grids.py` | 6 pins: bit-exact parity ×2, mortar-reduction identity, spectral energy decay, guard raises, constructor validation |
| `Metasurface_QWP/experiments/exp21/perlayer_gates.py` | the gate script (experiment side, reproduction) |

## 5. Follow-on results (same day)

### 5.1 Conical (φ≠0) per-layer — implemented (v2)

`_conical_nodal_solve(..., layer_grids="per-layer")`: the identical window+mortar construction — the
mortar is geometry-level and gauge-free, so only the ky0-shifted eigensolve differs. Wired through
`PMMStack` (the conical guard removed); a 2-layer patterned stack is pinned **bit-exact** vs shared at
two azimuths (`test_conical_per_layer_matches_shared_bit_exact`). Gate against the **converged** shared
OOP reference on the pathological device:

| deg | θ | shared (oracle) | per-layer | rel | speedup |
|---|---|---|---|---|---|
| 6 | 8° | 14.4177 | 13.5880 | 5.8 % | 13.8× |
| 8 | 8° | 14.3798 | 13.6441 | 5.1 % | 18.5× |
| 6 | 15° | 2.6076 | 2.5724 | 1.4 % | 14.5× |
| 8 | 15° | 2.6054 | 2.5676 | 1.5 % | 17.9× |

Within the mortar band, degree-stable (0.2–0.4 % deg6→8), and both cuts of the campaign device now run
per-layer.

### 5.2 The n_slice ladder — the physical staircase CONVERGES (ns ≈ 6–8)

Previously unaffordable (`O(n_slices^3.4)` + shared-grid conditioning re-breaking at ns=4). Per-layer,
28–175 s per rung; in-plane ER of the pathological device at deg 8, default `min_feature`:

| θ | ns2 | ns3 | ns4 | ns6 | ns8 | ns6→ns8 |
|---|---|---|---|---|---|---|
| 0° | 55.03 | 63.55 | 64.24 | 63.74 | 63.46 | 0.4 % |
| 8° | 57.81 | 56.78 | 54.37 | 52.55 | 52.42 | 0.25 % |
| 10° | 55.41 | 49.62 | 46.42 | 44.45 | 44.39 | 0.1 % |
| 15° | 28.50 | 21.97 | 20.15 | 19.23 | 19.26 | 0.1 % |
| 20° | 10.51 | 8.83 | 8.35 | 8.10 | 8.12 | 0.2 % |

Two conclusions. **(a)** The true-taper limit is reached by ns ≈ 6: the converged in-plane curve is
**63.5 / 52.4 / 44.4 / 19.3 / 8.1** (18.0 / 17.2 / 16.5 / 12.9 / 9.1 dB) — the campaign's ns=2 values
carried **10–30 % staircase-geometry error at oblique angles** (softening the θ15–20 tail by ~1–2 dB,
raising θ0 by ~1 dB; the qualitative in-plane-tolerance claim survives). Winner-level numbers should be
quoted at ns ≥ 6 per-layer from here on. **(b)** The conditioning stress test at ns=8: deg8-vs-deg10
spread **0.15–5.6 %** — the window unions acquire thin elements at high ns and the mortar band widens
mildly (5.6 % worst), but nothing resembling the shared grid's collapse. Usable, with the band quoted.

### 5.3 `prepare()` per-layer — assessed, deferred (design sketch)

`_PreparedPMMStack` is coupled to the shared union grid throughout (caches keyed on union-cell eps
content; audited LRU semantics, P3-32). A per-layer prepared object is *architecturally cleaner* — grids,
masses and cross-masses are geometry-only and cacheable forever; a material-key change re-eigs ONLY that
layer's small grid and rebuilds its two interfaces — but it is a second correctness-sensitive class.
Deferred because the practical need is already served: a direct per-layer LC sweep (~2.7 s/point at
deg 8) beats the shared prepared sweep at deg 8, with no library change.

### 5.4 Cross-validation closure

cBal_db240 — non-convergent on the shared default grid at every degree tested — now agrees between the
two independent treatments (shared mf=1.5 nm vs per-layer default) to **~1.5 %** (61.8/56.2/52.8 vs
62.7/57.1/53.7 at θ0/5/8).

## 6. Follow-ups (open)

* `prepare()` per-layer (§5.3 sketch); `retain_internal` per-layer.
* Winner-suite re-quote at ns ≥ 6 per-layer (minutes of compute).
* Mortar residual reduction if a use case needs closure beyond ~1e-6: deg-scaled cross-quadrature or
  one-element enrichment at the third-neighbour walls.


