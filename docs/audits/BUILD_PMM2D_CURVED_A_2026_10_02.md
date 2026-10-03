# BUILD -- curved cells for the pure staggered 2-D PMM, Phase A (the variable-coefficient assembly, the map protocol, the three traps, the cofactor far field)

Date: 2026-10-02.  Status: BUILT + gated, not pushed.
Mount: worktree `C:/tmp/lum_curved`, branch `feat/pmm2d-curved-cells`, built on
`47c1c3a0` (the plan commit); Windows 11 (tesla-ryzen), CPython 3.14.6,
numpy 2.4.4, scipy 1.17.1, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1` on the command line, `lumenairy.__file__` asserted under
the worktree in every probe.  PRE tree for byte identity: `git archive
47c1c3a0` extracted to `C:/tmp/curved_pre_a/`, run from inside itself with the
same assertion.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.1, approved
as written (map owned by the STACK; the whitened eig NOT in this phase; no map
on `pmm_efficiency_2d_staggered`).
Evidence: every number below is read from a JSON file in
`validation/probe_pmm2d_curved/build_a/` (the probe that wrote it is named in
each table; `_common.py` holds the shared fixture and the two engineered
defects).  Tests: `tests/unit/test_pmm2d_staggered_curved_a.py`.

---

## 0. Words used here

* **Coordinate map** `(x, y) = Phi(u, v)`: a smooth distortion of the plane.
  The solver keeps its straight rectangular wall grid in `(u, v)`; the map
  bends or stretches it in the physical `(x, y)` plane.  One map serves every
  layer and both half-spaces (it is owned by the stack).
* **Jacobian** `J = d(x, y)/d(u, v) = [[x_u, x_v], [y_u, y_v]]`; `det J` is
  the local area magnification; **metric** `g = J^T J`; `sqrt(g)` means
  `det J`.
* **Covariant components** `E' = J^T E`: the field measured along the
  `(u, v)` grid lines.  In `(u, v)` Maxwell's equations keep their Cartesian
  form with the **effective tensors** `eps'_t = eps sqrt(g) g^-1`,
  `eps'_33 = eps sqrt(g)`, `chi_t = [mu'_t]^-1 = g / sqrt(g)`,
  `chi33 = 1 / sqrt(g)` -- so every mapped region, vacuum included, is a
  block-form TENSOR and MAGNETIC region.
* **Separable stretch** `x = f(u)`, `y = h(v)`: the only map this phase
  ships besides the identity.  It curves no wall; with the `(u, v)` walls at
  the PREIMAGES of the physical walls the device is unchanged and only the
  solver's resolution is redistributed, so the mapped solve must converge to
  the unmapped answer.  The gate map is `f(u) = u + a sin(2 pi u / p)`; at
  `a = 0.15 p` the local stretch `f'` runs from 0.058 to 1.94 (33:1).
* **Fail-before**: a deliberately broken arm that must fail the bar, proving
  the test can see the defect it guards.  Here every fail-before is
  ENGINEERED through the real code path (`_common.no_cofactor`,
  `_common.mixed_hgram`, a replaced retained Gram), never a copy of the
  physics.

---

## 1. What was built

| item | file | what it does |
|---|---|---|
| the map protocol | `lumenairy/elements/pmm/_curvemap.py` (new) | `CellMap` base: `period_x/_y`, `u_walls/v_walls` (int or boundary array, resolved to `u_bounds/v_bounds` exactly as `Basis1D` resolves them), `geom(sx, sy, U, V) -> (X, Y, x_u, x_v, y_u, y_v)` on Gauss-node tensors, a content `fingerprint` (SHA-256 of class, periods, walls, curve parameters), and `validate()` (lattice periodicity of `Phi - id` and of `J`, the cell boundary mapped onto itself, `det J > 0` on a probe grid). `IdentityMap`, `SineStretch` (analytic `f`, `f'`, Newton inverse, refusal at `|a| >= p / (2 pi)`), `SeparableStretch(u_walls, v_walls, fx, fy)` with `from_physical_walls(...)`. |
| `_stag_quad_weighted(bx, by, xspec, yspec, W, rule, cache)` | `twod_staggered.py` | ONE function for all 18 weighted blocks: the 2-D Gauss-quadrature generalisation of `_eps_weighted` (mass) and `_eps_dir` (one derivative), flavours `m` / `d` / `dL`, restricted per cell to the supported global functions.  Lifted from the planning scratch `_axis_factor` + `CurvedGranet.blk` (`validation/probe_pmm2d_curved/_curved_scratch.py` lines 364-468). |
| `_stag_map_weights` | same | the effective-tensor node weights (`e11 e12 e21 e22 e33`, `c11 c12 c21 c22 c33`) of a scalar cell under the map; RAISES on `det J <= 0` or a non-finite Jacobian at any node.  Lifted from `CurvedGranet._weights` (scratch lines 413-447). |
| `_stag_map_nodes` | same | NEW (build finding F1, section 4): the ADAPTIVE per-axis node count -- doubles from `2 M + 8` until the Legendre moments (degree `<= 2M - 2`) of the five geometric weight functions agree to `1e-13` between `n` and `2 n` nodes; capped at 256 with a warning -- the EFFECTIVE cap is the largest `2^k (2 M + 8) <= 256` (144 / 160 / 192 at `M = 5 / 6 / 8`), and the search evaluates moments at up to twice that (Phase A verifier D5). |
| `Granet2DTransverseE(..., cmap=None)` | same | `None`: `self.cmap = None` and nothing reads it -- the shipped Kronecker assembly (A1).  A map: validated against the solver's walls and periods; scalar `eps_cell` only (tensor / `mu_cell` / `slant` raise `NotImplementedError` naming Phase D / E); the SAME `_assemble` body runs with `tensor = True` (component weights = `eps'`) and `magnetic = True` (chi weights = the metric's), and `_eps_weighted` / `_eps_dir` dispatch every one of the 18 blocks to `_stag_quad_weighted`.  `Ggram_blocks` (the plain Gram) and `Et_offdiag` are retained, so `_region_modes` takes its plain-Gram branch unchanged. |
| `_homog_geom_cache` (mapped branch) | same | keeps `-R` as the pencil's right-hand matrix (the split survives, A5) and inverts the PLAIN block Gram for the H partner (trap H). |
| `_far_projector_2d(..., cmap=None)` / `_far_projector_mapped` | same | with a map: the pulled-back projector with the COFACTOR `det J J^-T = [[y_v, -y_u], [-x_v, x_u]]`, four blocks (off-diagonal ones `None` when the cofactor entry is identically zero), `max(2M + 16, _stag_quad_order(M, omega))` nodes per cell with `omega` the phase across the cell's PHYSICAL half-extents.  Lifted from `curved_far_projector` (scratch lines 581-622), with the two-stage support-restricted contraction. |
| `_pmm2d_project_orders(..., P12=None, P21=None)` | same | adds the off-diagonal blocks when present; unchanged arithmetic without them. |
| `pmm_jones_2d_staggered(..., cmap=None)` | same | forwards the map to the stack. |
| `pmm_efficiency_2d_staggered(..., cmap=None)` | same | RAISES on a map, pointing at the Jones entry (plan Q5). |
| `PMM2DStackPure(..., cmap=None)` | `stack2d_pure.py` | the stack-owned map: fixes the union grid to the map's `(u, v)` grid; `solve()` builds the homogeneous solver and every layer solver on the map, keys the per-solve eig dedupe by the map fingerprint, retains the PLAIN block Gram as the flux form (`layer_absorption`), and projects with the cofactor projector.  `layer_grids='per-layer'` + map RAISES (Phase E); tensor / `mu` / `slant` layers under a map RAISE (Phase D / E); the two viewers RAISE (they would draw the `(u, v)` cells); the joint segment-redundancy advice is skipped (the map owns the grid). |
| history fingerprints | `docs/history/lumenairy.elements.pmm.stack2d_pure.md` | re-recorded with the reason (`twod_staggered.py` has no history document). |

Not touched: `Basis1D`, the mortar ingredients, the out-of-plane generator,
the parity reduction, the Redheffer cascade, the eigensolver (QZ, as shipped;
the whitened eig is its own later patch).

---

## 2. The formulation as built, and where each piece came from

| piece | plan | as built | measured where |
|---|---|---|---|
| effective tensors | 2.1 | `_stag_map_weights`, per node | A2, A3, A4 |
| 18 weighted blocks by quadrature, metric-free curl / gradient / Gram | 2.2 | the shipped `_assemble` body, block kernel dispatched | A2 (`a2_identity.json`, section `kernel`) |
| staggered C0 basis unchanged | 2.3 | `Basis1D` untouched | A3, A4 |
| geometric split `-R == [eps'_t]/eps` | 2.4 | mapped `_homog_geom_cache` keeps `-R` in the pencil | A5 |
| plain-Gram H partner in EVERY mapped region | 2.4 | `_region_modes` (via `Ggram_blocks`) and the mapped `_homog_geom_cache` | A6 |
| cofactor far field + incident overlap | 2.5 | `_far_projector_mapped`, `Hsup` | A3, A4 |
| plain Gram as the flux form | 1.2 (`solve`, line 1839) | `G_gram` on the mapped path | A7 (NEW) |
| quadrature rule | 3.3 (`2M + 8`) | ADAPTIVE (`_stag_map_nodes`) | F1, `a2q_quadrature_*.json` |
| boundary condition on the map | 4.1 ("identity on the cell boundary") | lattice periodicity + boundary onto itself | F2 |

---

## 3. The gates

All bars are the plan's (re-derived here from this build's measurements, both
gaps stated); every measurement is from this tree, 2026-10-02.

| gate | claim | bar | measured (this tree) | fail-before (measured) | probe JSON |
|---|---|---|---|---|---|
| A1 | no map = today's bytes | SHA equality | **109 / 109** SHA-256 identical against the `47c1c3a0` archive: operator blocks (`Rmat Lmat Stt Schur Agen Bgen Et_blocks Et_offdiag Ggram_blocks`) of scalar integer / non-uniform / oblique, tensor, two magnetic, out-of-plane and slant solvers; region modes (in-plane, out-of-plane with and without parity), the geometric cache and homogeneous modes; three far projectors; `pmm_efficiency_2d_staggered` TE/TM normal + oblique; six `pmm_jones_2d_staggered` cases (conical scalar, tensor, magnetic, out-of-plane auto / no symmetry, slant); a lossy multilayer stack with `layer_absorption`; the per-layer mortar with absorption | identity map through the quadrature path: 28 of 36 operator hashes differ (the 8 equal ones are the `None` attributes), max `|dL|` 9.7e-15 -- the hash sees round-off | `a1_bytes_pre.json`, `a1_bytes_post.json`, `a1_compare.json` |
| A2 | identity map through the quadrature path | rel `<= 1e-11` | 18 block kernels `<= 1.6e-14`; assembled operators `<= 5.5e-14`; matched pencil eigenvalues `<= 4.3e-12`; far projector `<= 5.2e-15` (corrected 2026-10-02 from 4.5e-15, the normal-incidence reading; the oblique case of `a2_identity.json` reads 5.2e-15 -- Phase A verifier D7) with both off-diagonal blocks absent; full solve R / T / Jones `<= 2.6e-14` (3 x 3 pillar, integer and non-uniform walls, `M = 5, 7`) -- 2.3 decades under | a `1e-6 p` stretch moves `L` by 7.8e-6 .. 1.0e-5 (5.9 decades above) | `a2_identity.json` |
| A3 | uniform film under a stretch is exact | `<= 1e-10` at `M = 6` and TM decay `M = 4 -> 6` `>= 4` decades | `a = 0.15 p`: TM 4.0e-07 / 5.3e-09 / 1.29e-12 / 1.2e-15 at `M = 4..7`, TE `<= 3.9e-14`; decay 5.5 decades; `a = 0.05 p`: TM 4.8e-08 / 3.0e-11 / 1.4e-14 / 4.9e-15 | no cofactor: 0.96 (`a = 0.15 p`), 0.052 (`a = 0.05 p`) at `M = 6` | `a3_film.json` |
| A4 | TE stripe under `a = 0.05 p` vs `pmm_efficiency_1d` (degree 40, self-gap 4.1e-07) | `<= 5e-3` at `M = 8` | 4.48e-04 (1.05 decades under); ladder to `M = 11` in section 4.1; **STOP condition met**: 9.67e-07 at `M = 10` (bar 1e-5) | no cofactor: 0.2157 at `M = 8` (1.6 decades above), 0.2175 at `M = 6` | `a4_stripe_a0.05.json`, `a4_nocof_a0.05_M8.json`, `a4_nocof_a0.05_M6.json`, `a4_oracle1d.json` |
| A5 | geometric split `-R == [eps'_t]/eps` | `<= 1e-12` | `<= 7.2e-16` over `a = 0, 0.05, 0.15 p`, `eps = 1, 2.1025, 4 + 0.5i`, `M = 4, 6` (3.1 decades under) | uniform MAGNETIC region, no map: 1.0 (scalar `mu = 2`), 0.5 (`mu = diag(2, 1, 1)`); `mu = 1` control 2.1e-16 | `a5_split.json` |
| A6 | H-partner Gram | correct closure `<= 1e-5` at `M = 8`; MIXED `> 1e-2` | stripe 2.28e-06, pillar 1.03e-06 | MIXED (half-spaces through `-R`): closure 0.157 / 0.133, R/T moved 0.106 / 0.053 (`M = 8`); 0.158 / 0.132 at `M = 6` -- the planning P2b numbers to three digits | `a6_hgram_M8.json`, `a6_hgram_M6.json` |
| A7 | absorption under a map: `sum layer_absorption == 1 - sum R - sum T` | NEW: `<= 1e-3` on the lossy stripe at `M = 7` | stripe (`eps = 4 + 0.5i`, `a = 0.05 p`): 3.1e-03, 1.7e-03, 2.0e-04, **1.99e-05**, 1.9e-06 at `M = 4..8` (1.7 decades under at `M = 7`); unmapped reference 7.2e-08 at `M = 7`; film: `<= 2.6e-14` by `M = 7` | `-R` as the flux Gram: 0.046, 0.070, 0.070, **0.070**, 0.070 (1.8 decades above at `M = 7`) | `a7_absorption.json` |

The A7 bar derivation: the correct arm converges with `M` (one decade per rung
from `M = 6`), the fail-before is flat at 0.070 (a wrong bilinear form, not a
discretisation error); at `M = 7` the bar `1e-3` sits 1.7 decades above the
correct reading and 1.8 below the defect.

---

## 4. Ladders and findings

### 4.1 The stripe under the stretch (A4 and the STOP condition)

`a4_stripe_a{0.0,0.05,0.15}.json`; TE / TM error against the 1-D oracle and the
lossless closure (max of both polarizations).  `a = 0` is the UNMAPPED shipped
solver on the same physical walls (a one-layer per-layer stack).

| M | a = 0 (TE / TM / closure) | a = 0.05 p | a = 0.15 p |
|---|---|---|---|
| 7 | 6.1e-05 / 4.2e-05 / 1.7e-07 | 9.3e-04 / 2.0e-04 / 2.8e-05 | 1.9e-02 / 3.0e-03 / 1.9e-03 |
| 8 | 6.1e-05 / 3.4e-05 / 7.6e-08 | 4.5e-04 / 3.2e-05 / 2.3e-06 | 1.0e-02 / 1.6e-03 / 1.4e-03 |
| 9 | 1.4e-07 / 2.8e-05 / 3.6e-10 | 1.5e-05 / 2.8e-05 / 2.1e-07 | 3.9e-03 / 1.0e-03 / 1.4e-04 |
| 10 | 1.4e-07 / 1.7e-05 / 2.5e-11 | **9.7e-07** / 1.4e-05 / 1.4e-08 | 3.0e-03 / 3.3e-04 / 4.5e-05 |
| 11 | 2.6e-08 / 1.5e-05 / 3.6e-13 | 6.6e-07 / 1.3e-05 / 2.5e-09 | 3.1e-04 / 6.1e-05 / 4.9e-06 |

Every TE entry reproduces the planning table (plan 3.2) to the printed digits;
the mapped solve converges to the unmapped answer, a stretch costs one to two
rungs at `a = 0.05 p` and four or more at `a = 0.15 p`, and TM is capped by the
rim in both solvers.  STOP condition (TE at `a = 0.05 p` within 1e-5 by
`M = 10`): met, 9.7e-07.

### 4.2 F1 -- the fixed quadrature rule is NOT adequate under a strong stretch (formulation finding, fixed)

The plan sized the assembly at `nq = 2 M + 8` nodes per axis per cell and
measured it adequate on the circle map.  Under the stretch the weights carry
`1 / f'(u)`; at `a = 0.15 p`, `f'` falls to 0.058 and `1 / f'` has complex
poles within ~0.07 (in `u`) of the real axis, so a fixed rule converges slowly.
Measured on the stripe, R / T change against an 8x finer rule
(`a2q_quadrature_fixed_base.json`):

| a | M = 5 | M = 7 | M = 9 |
|---|---|---|---|
| 0.05 p | 2.7e-09 | 1.8e-11 | 7.4e-14 |
| 0.15 p | **5.2e-04** | **7.8e-05** | **5.3e-06** |

and the operators themselves move 3.0e-2 (`M = 5`) and 1.1e-2 (`M = 7`)
relative (`a2_identity.json`, section `quad`).  The error is energy-INVISIBLE
in the sense that matters: it is a consistent wrong operator, and at
`a = 0.15 p` it sits below the (large) discretisation error of the ladder in
4.1 -- which is why the planning ladder did not show it (the fixed-rule and
adaptive ladders agree to the printed digits,
`a4_stripe_a0.15_fixedquad.json`) -- but it is a FLOOR that would surface as
soon as the discretisation error falls below it.

Fix (with the measurement as evidence): `_stag_map_nodes` chooses the node
count from the MAP -- doubling from `2 M + 8` until the Legendre moments of the
five geometric weight functions agree to `1e-13` between `n` and `2 n`.
Polynomial weights pass at the first check, so the identity map still runs
`2 M + 8` (A2 unchanged).  Chosen counts (`a8_cost.json`): `a = 0.05 p`: 40,
24, 28 at `M = 6, 8, 10`; `a = 0.15 p`: 160, 192, 112.  The adaptive rule
against the 8x rule (`a2q_quadrature_adaptive.json`): `a = 0.05 p` 1.3e-09 /
3.1e-13 / 7.4e-14, `a = 0.15 p` 0.0 / 0.0 / 3.1e-11 at `M = 5 / 7 / 9`.
For Phase B: the circle map's `1/r` corner weights will not meet `1e-13` and
will run to the cap (256) with a warning; Phase B must decide the circle's
rule (the planning measurement, 2.8e-8 at `2M + 8` vs 4x, suggests a looser
moment tolerance or a corner-graded rule) and re-measure.

### 4.3 F2 -- "identity on the cell boundary" is stronger than the solver needs

The plan's validation list says identity on the cell boundary.  A separable
stretch is NOT pointwise identity there (`Phi(u, 0) = (f(u), 0)` slides points
along the boundary), yet A3 / A4 / A6 / A7 are exact under it.  What the
solver needs is LATTICE PERIODICITY of `Phi - id` and of `J` (so the Bloch glue
`tau` is unchanged and the image of the `(u, v)` cell is one period of the
lattice), plus the boundary mapped onto itself (one shared lattice origin for
the map frame and the lab).  `CellMap.validate` checks exactly that;
pointwise identity is the stronger property the Phase-B transfinite maps will
have anyway.

### 4.4 F3 -- oblique and conical incidence under the stretch (not probed by the plan)

A uniform film under the stretch at `theta = 25 deg` (`phi = 0`) and conical
`phi = 40 deg` against the s / p Airy reflectance (`a3_film.json`, arm
`oblique`):

| arm | M = 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| unmapped, 25 deg | 1.4e-06 | 5.4e-09 | 1.5e-11 | 3.2e-14 |
| a = 0.05 p, 25 deg | 6.1e-05 | 1.8e-06 | 5.1e-08 | 1.6e-09 |
| a = 0.15 p, 25 deg | 9.0e-04 | 6.0e-05 | 1.9e-06 | 9.6e-08 |
| a = 0.15 p, conical 40 deg | 7.0e-04 | 3.9e-05 | 4.9e-07 | 1.5e-08 |

Spectral in every arm: the Bloch glue and the `alpha0` kernel need no change
under a periodic map, and the map costs only resolution (the tilted plane wave
must be resolved on the compressed grid).  This is the stretch only; the
circle at oblique incidence remains gate B10.

### 4.5 F4 -- the film is a weak discriminator for the flux Gram, the stripe a strong one

With `-R` as the flux Gram the FILM's absorption defect falls with `M`
(3.2e-06 .. 1.7e-11 at `a = 0.05 p`, `a7_absorption.json`), because the film's
modes converge to plane waves on which the two forms nearly agree; the STRIPE
holds the defect at 0.070 for every `M`.  A7 is therefore gated on the stripe.

### 4.6 F5 -- where the `(u, v)` walls go matters

A uniform film is exact on a uniform `(u, v)` lattice (A3), but the same film
on the stripe's PREIMAGE walls (`u = 0, 0.134, 1.004, 1.2` at `a = 0.15 p`)
converged three to four decades slower at `M = 4..7` (2.9e-4 .. 4.1e-9, first
run of the probe before its fixture was corrected): one 0.87-period cell holds
the whole compression.  Phase C's shape layer should keep the map's steep
region subdivided.

---

## 5. Cost

`a8_cost.json` (operator assembly only, best of three, 3 x 3 grid):

| M | no map | identity (nq) | stretch 0.05 p (nq) | stretch 0.15 p (nq) |
|---|---|---|---|---|
| 6 | 0.039 s | 0.054 s (20) | 0.059 s (40) | 0.129 s (160) |
| 8 | 0.235 s | 0.275 s (24) | 0.277 s (24) | 0.428 s (192) |
| 10 | 0.926 s | 1.074 s (28) | 1.077 s (28) | 1.146 s (112) |

1.1-1.8x the shipped assembly, against a region QZ of 8-130 s at the same
sizes on this box -- negligible.  The pencil, its size and the eig are
unchanged.

---

## 6. What moved

Nothing shipped: A1, 109 / 109 hashes byte-identical against the parent
commit, over every dispatch branch of the family.  The map is reachable only
through `pmm_jones_2d_staggered(cmap=)` / `PMM2DStackPure(cmap=)` (and the
solver class), and the only maps are the identity and the separable stretch.

---

## 7. Not measured in this phase

* The circle, fillet and sinusoid maps (Phase B) and therefore the singular
  vertices under the library code; the adaptive rule's behaviour on them
  (4.2).
* The probe READINGS on a second build.  The unit gates themselves were run
  on a second build -- WSL Ubuntu, CPython 3.12, numpy 2.4.6, scipy 1.17.1,
  BLAS pinned, `lumenairy` from `/mnt/c/tmp/lum_curved` -- and pass 13 / 13
  (54 s); every bar is two-sided by 0.6-11 decades on deterministic
  discretisation numbers or round-off residuals.
* Multilayer mapped stacks with more than one patterned layer (the cascade is
  the shipped square cascade; not separately gated here).

---

## 8. Reproduction

```
cd /c/tmp/lum_curved
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved
python validation/probe_pmm2d_curved/build_a/a1_bytes.py C:/tmp/lum_curved post idmap   # + the pre tree, see the file
python validation/probe_pmm2d_curved/build_a/a1_compare.py
python validation/probe_pmm2d_curved/build_a/a2_identity.py
python validation/probe_pmm2d_curved/build_a/a2q_quadrature.py fixed_base fixed
python validation/probe_pmm2d_curved/build_a/a2q_quadrature.py adaptive adaptive
python validation/probe_pmm2d_curved/build_a/a3_film.py
python validation/probe_pmm2d_curved/build_a/a4_stripe.py oracle
python validation/probe_pmm2d_curved/build_a/a4_stripe.py ladder 0.05 4 11   # 0.0, 0.15 likewise
python validation/probe_pmm2d_curved/build_a/a4_stripe.py nocof 0.05 8       # and 6
python validation/probe_pmm2d_curved/build_a/a5_split.py
python validation/probe_pmm2d_curved/build_a/a6_hgram.py 8                   # and 6
python validation/probe_pmm2d_curved/build_a/a7_absorption.py
python validation/probe_pmm2d_curved/build_a/a8_cost.py
python -m pytest tests/unit/test_pmm2d_staggered_curved_a.py --capture=sys -p no:randomly
```
