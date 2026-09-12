# PMM-2D audit — the 2-D Polynomial Modal Method (hybrid + staggered + stacks + JAX twins)

Repo `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`, branch `main`,
HEAD `a1ff1e6e` (v5.45.1). Python 3.14 / numpy 2.4.6 / scipy 1.17.1 / jax 0.10.1 (CPU).
All repro scripts and logs:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/PMM-2D/`.

## Scope read (files + line ranges; what I did NOT get to)

Read line by line:

* `lumenairy/elements/pmm/twod.py` (1–1659, all) — the hybrid scalar 2-D core.
* `lumenairy/elements/pmm/twod_jones.py` (1–818, all) — `pmm_jones_2d`, `_tensor_layer_modes`.
* `lumenairy/elements/pmm/stack2d.py` — 1–220 (module + helpers), 264–500 (ctor, `_geom_key`,
  `add_layer`), 777–1015 (`_symmetric_layer_specs`, `_mode_key`, `_build_layer_modes`,
  `_layer_mode_sets`), 1156–1460 and 1520–1640 (`solve`, cascade, frame anchor, far field).
* `lumenairy/elements/pmm/_stack2d_cache.py` (1–252, all).
* `lumenairy/elements/pmm/_jax_twod_jones.py` (1–263, all).
* `lumenairy/elements/pmm/_jax_twod.py` (1–281) incl. `_host_incidence_guard`, `_static_prep`,
  `_static_prep_cell`.
* `lumenairy/elements/pmm/twod_staggered.py` — 1–110 (the method statement), 2372, 2555–2620,
  3115–3170 (the two `sqrt` branch sites, `_region_modes`, `_homog_region_modes`), plus a
  whole-file grep sweep for `sqrt` / bare `except` / silent fallbacks.
* `lumenairy/elements/pmm/stack2d_pure.py` — 1–200 (module statement + scope), 1646–1740
  (`_solve_per_layer`, the mortar guards), plus greps.
* Cross-lane, as needed to close findings: `rcwa/_core.py` `_sqrt_decay` + `_CUT_BAND_REL`
  (1271–1440), `rcwa/twod.py::rcwa_jones_2d` signature, `rcwa/oned.py::rcwa_jones_1d`
  Returns block, `pmm/_core.py::PerOrderAmplitudesMixin` (1985–2040), `CONVENTIONS.md` §7/§7.1/§11,
  `git show 1627f027`, `docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md`.

NOT fully read (declared): the bulk of `twod_staggered.py` (the Granet basis assembly,
`Basis1D`, `Granet2DTransverseE`, `_assemble_oop`, `_stag_*` mortar kernels — roughly lines
110–2370 and 2620–3115) and the bulk of `stack2d_pure.py` (200–1646, 1740–2071). I established
that `twod_staggered.py` is a LIVE independent engine (probe 5 below) rather than dead code, which
is what the partition asked; a line-by-line audit of the Granet assembly is a second pass.
`_jax_stack2d.py` was read only for its `_sqrt_decay` call sites. I also did not run a numeric
ladder across the `PMM2DStackPure` mortar guard threshold (probe 6's second half) — the two probe
runs I launched for it were killed after 1096 s of CPU and 3.9 GB (see the `eps_cell` cost-cliff
finding, which is what those runs turned up instead).

---

## Findings

Thirteen findings: one P1, seven P2, five P3. Ordered P1 → P2 → P3 with one exception — the
`PreparedPMM2D` P2 sits with the two cache/attribute P3s it shares a mechanism with.

### **[P1] `PMM2DStackHybrid` never routes the per-slot Li inverse rule, so the DEFAULT `formulation='li'` breaks 90° rotation invariance** — `lumenairy/elements/pmm/stack2d.py:836` and `:928-931`

**What is wrong.** Audit P3-33 split the hybrid's `formulation='li'` eps operator into a per-slot
pair — `EpnxF` (inverse rule along x) on the `Ex` slot and `EpnyF` on the `Ey` slot — precisely so
that a **y**-patterned cell gets the inverse rule on its wall-normal component. `twod._pmm2d_solve_core`
passes them — `twod.py:888-889` on the even-parity fold and `twod.py:896-898` on the full solve.
`PMM2DStackHybrid` does not:

```python
# stack2d.py:835-839  (_symmetric_layer_specs, the even-parity fold)
if self.formulation == "li":
    EPS_nx, EPS_ny, EPS_inv = EpnF, EpsF, EinvF
# stack2d.py:928-931  (_build_layer_modes, the full solve)
out = _layer_modes_projected(
    GxF, GyF, lops["EpsF"], lops["EinvF"], lops["EpnF"],
    formulation=self.formulation, slant=sl)          # no EpnxF/EpnyF
```

`twod._layer_modes_projected`'s own docstring names the gap: *"Callers not passing the per-slot
operators (the PMM2DStack path) keep the legacy assignment `(Ex <- EpnF, Ey <- EpsF)`"*. For a
y-patterned layer `_scalar_projected_ops` returns `EpnF = kron(Ty @ o1["Epn"] @ Typ, Ix)` — the
**y**-axis inverse-rule operator — and the legacy line puts it on the **Ex** slot while `Ey` keeps
Laurent. That is exactly the case `_scalar_projected_ops`'s own `nsx == 1 and nsy > 1` branch
describes (`twod.py:606-609`): *"Audit P3-33: the legacy single `EpnF` landed on the Ex slot -- BOTH slots anti-Li on
this branch, breaking the 90-degree rotation symmetry that formulation='laurent' has exactly."*
It is still live in the stack.

**Evidence** (`p3b.py`, `q1b_stackli.py`, `q1c.py`; logs `q1b.log`, task `bfy3jltsm`). One 1-D Si
grating, eps 12.25 / 1.0, duty 0.5, Λ = 0.47 µm, λ = 1 µm, d = 0.3 µm, n_sub = 1.5, normal incidence.
`x-pat` is the cell patterned along x driven by `E_x`; `y-pat` is the same cell transposed driven by
`E_y` — a pure 90° rotation of one physical problem, so the two numbers must be equal.

| entry | formulation | degree/n_orders | T00 (x-pat) | T00 (y-pat) | Δ |
|---|---|---|---|---|---|
| `pmm_efficiency_2d_cell` (has the fix) | li | 11 / 5 | 0.02319196 | 0.02319196 | **0.0** |
| `pmm_efficiency_2d_cell` | li | 11 / 9 | 0.03347366 | 0.03347366 | **0.0** |
| `PMM2DStackHybrid` | li | 11 / 5 | 0.02319196 | 0.02295665 | **2.4e-04** |
| `PMM2DStackHybrid` | li | 11 / 9 | 0.03347366 | 0.03224247 | **1.23e-03** (5.3 % of T00) |
| `PMM2DStackHybrid` | laurent | 11 / 9 | — | — | symmetric to 1.1e-14 |

At degree 7 / n_orders 5 the whole per-order vector differs by `max|ΔT| = 4.977e-03` and the
reflection Jones by `|ΔJ| = 2.318e-02` (`p3b.py`). `symmetry='auto'` and `symmetry=False` give the
same break (both code paths carry the legacy line), and `formulation='laurent'` is symmetric to
1.07e-14, which isolates the cause: it is the per-slot routing, not the grid.

**Impact.** `formulation='li'` is `PMM2DStackHybrid`'s constructor DEFAULT. A user's answer depends
on whether they wrote their grating along x or along y; one of the two orientations carries an
anti-Li factorization and converges more slowly. Nothing warns. It also means
`PMM2DStackHybrid(1 layer)` and `pmm_efficiency_2d_cell` — documented as the same physics — disagree
for half of all 1-D-grating orientations.

**Fix.** Two lines, mirroring `twod.py:889-898`:
`_build_layer_modes` → `_layer_modes_projected(..., EpnxF=lops["EpnxF"], EpnyF=lops["EpnyF"], slant=sl)`;
`_symmetric_layer_specs` → `EPS_nx, EPS_ny = lops["EpnxF"], lops["EpnyF"]` under `li`. No cache-key
change is needed (`formulation` is already in `_mode_key`). Gate it with exactly the test above: one
grating, two orientations, assert equality to 1e-13.

---

### **[P2] `pmm_jones_2d(formulation='li')` is measurably WORSE than `'laurent'`, and at the documented defaults the reflection-Jones PHASE is wrong by up to 8.9°** — `lumenairy/elements/pmm/twod_jones.py:305-314`, docstring `:440-459`

**What is wrong.** On the Jones entry `formulation` only selects the `E_z`-elimination rule
(`'li'` = `inv(proj(1/e_zz))`, `'laurent'` = `proj(e_zz)`); the in-plane block is Laurent either way.
The docstring presents `'li'` as "the hybrid's *validated* inverse-rule elimination", which reads as
"at least as good". Measured, it is consistently worse, and the error is in the PHASE — the
observable the user's QWP work depends on.

**Evidence A** (`q3_fffnv.py`, `q3.log`) — separable Si stripe, eps 12.25, duty 0.5, Λ = 0.47 µm,
λ = 1 µm, d = 0.3 µm, n_sub = 1.5, normal incidence, degree 11. Oracle: `rcwa_jones_1d(n_orders=80,
li)` → `Jxx = -0.52123629 - 0.83788469j`.

| n_orders | PMM `fff_nv` | PMM `laurent` | PMM `li` | RCWA-2D `li` |
|---|---|---|---|---|
| 5 | −0.47974−0.86420j | −0.40153−0.90547j | −0.66661−0.72987j | −0.52676−0.83310j |
| 11 | −0.51940−0.83989j | −0.49569−0.85401j | −0.57759−0.79814j | −0.52097−0.83802j |
| **\|err\| at n=11** | **2.7e-03** | 2.90e-02 | **6.85e-02** | **2.7e-04** |

**Evidence B** (`q1_conv.py`, `q1_conv.log`) — C4 Si pillar (eps 12.25, 50 % duty each axis), same
lattice. Converged reference `rcwa_jones_2d(li)` = −0.99688 − 0.0725j (stable from n=9 to n=15).

| n_orders | PMM `laurent` Jxx | PMM `li` Jxx |
|---|---|---|
| 3 | −0.96218 − 0.27239j | −0.58804 **+0.40271j** |
| 7 | −0.98451 − 0.17369j | −0.91779 **+0.22427j** |
| 11 | −0.99550 − 0.09369j | −0.98689 **+0.08226j** |

`'li'` keeps the **wrong sign** on `Im(Jxx)` at every truncation the entry can afford:
`arg(Jxx)` is off by 0.155 rad = **8.9°** at n_orders = 11, against 0.021 rad = 1.2° for `'laurent'`.
Energy closes to 1e-5 on both, so `_check_energy` is silent.

**Impact.** A user reading the docstring will pick `'li'` for "rigour" and get the worse phase.
For a metasurface waveplate, 8.9° of retardance error at the entry's default truncation is a
design-breaking number.

**Fix.** Replace the docstring's neutral framing with the measurement: on a patterned cell the
in-plane block is Laurent-floored for both rules, `'laurent'` is the better-conditioned `E_z` rule,
and `'fff_nv'` (separable cells) is 10–25× better than either. Consider making `'fff_nv'` the
default whenever the cell is separable and in-plane.

---

### **[P2] The 2-D hybrid's lossless energy closure plateaus at 1e-4…1e-2, is NON-MONOTONE in `n_orders`, sits far below the warn threshold — and on one fixture it moves OPPOSITE to the per-order error it stands in for** — `lumenairy/elements/pmm/twod_jones.py:687`, `twod.py:_PASSIVE_TOL_2D:124`

**Evidence** (`q8_energy.py`, `q8.log`). Provably lossless Si pillar (eps 12.11, 50 % duty) in air on
n_sub = 1.45, Px = Py = 0.9 µm, λ = 1 µm, d = 0.35 µm, degree 11. `max|1 − (ΣR+ΣT)|` over both
incident polarizations:

| θ, φ | n_orders 5 | 9 | 11 |
|---|---|---|---|
| 0°, 0° | 1.54e-03 | 1.37e-04 | **5.89e-04** (worse than n=9) |
| 20°, 0° | 7.99e-03 | 2.77e-03 | 4.96e-04 |
| 20°, 30° | 4.67e-03 | 4.51e-04 | 1.48e-04 |

`formulation='li'` is uniformly worse than `'laurent'` here too (e.g. 7.37e-04 vs 5.89e-04 at
n=11, θ=0). On an L-shaped (chiral) cell the excess reaches `ΣR+ΣT = 1.0112` at n=5 and `1.0089`
at n=9 — a **0.9 %** energy excess on a lossless structure.

Every one of these is far inside `_PASSIVE_TOL_2D = 5.0e-2`, so `_check_energy` never fires. The
audit brief's target of `< 1e-8` at converged truncation is **not reachable** with this engine: it
is the documented Fourier floor, and the non-monotonicity means "raise n_orders" is not a reliable
remedy.

**A sharper form of the same problem: the per-order value is NOT tracking the energy closure.**
On one fixed geometry (`q14_stagtime.py`, the 12×12 pixel cell, eps 12.25 pillar at duty 1/2,
Px = Py = 0.9 µm, λ = 1 µm, d = 0.3 µm, n_sub = 1.45, degree 11) the hybrid reads:

| `pmm_efficiency_2d_cell` | ΣR+ΣT | T00 | wall |
|---|---|---|---|
| n_orders = 5 | 1.009688038044 | 0.2715166794 | 0.24 s |
| n_orders = 9 | 1.003634425845 | 0.2373949696 | 2.11 s |
| n_orders = 11 | 0.999564765274 | **0.1542801483** | 7.28 s |

The energy closure improves monotonically (9.7e-03 → 3.6e-03 → 4.4e-04) while `T00` moves by
**−35 %** from n = 9 to n = 11 and shows no sign of settling. So on this fixture the energy
tripwire is *anti-correlated* with the per-order error it is supposed to stand in for — which is
the real cost of the loose `_PASSIVE_TOL_2D`: a user watching energy would conclude n = 11 is the
best of the three.

The no-floor sibling reaches a different regime (`q14_stagtime.py`, 3×3 segment grid, same
periods / thickness / half-spaces). **NB the `T00` columns of the two tables are NOT
cross-comparable**: a 3-segment staggered grid puts the walls at 1/3 and 2/3 (duty 1/3) whereas the
12×12 pixel grid puts them at 1/4 and 3/4 (duty 1/2), so these are two different pillars. Only the
*closure* column carries across:

| `pmm_efficiency_2d_staggered` | ΣR+ΣT | T00 |
|---|---|---|
| M = 5, n_orders = 3 | 0.99999705276564 | 0.3901954721 |
| M = 5, n_orders = **5** | **0.99999705276564** | **0.3901954721** |
| M = 6, n_orders = 3 | 1.00000022847654 | 0.3915210324 |
| M = 6, n_orders = **5** | **1.00000022847654** | **0.3915210324** |
| M = 8, n_orders = 3 | 0.99999999981885 | 0.3939433975 |
| M = 8, n_orders = **5** | **0.99999999981885** | **0.3939433975** |
| M = 10, n_orders = 3 | 1.00000000000109 | 0.3943038006 |

| M = 10, n_orders = **5** | **1.00000000000109** | **0.3943038006** |

The `n_orders`-invariance claim holds **to 14 digits** (identical, not merely close, at every `M`
tested), and the residual tracks only `M`, spectrally — 2.9e-06 at M = 5, 2.3e-07 at M = 6,
1.8e-10 at M = 8, **1.1e-12 at M = 10** — i.e. six to nine decades below the hybrid's closure on a
comparable pillar, and unlike the hybrid the per-order value settles alongside it (T00 moves
3.6e-04 from M = 8 to M = 10). That is the substantiation for pointing energy-critical work at the
staggered engine, and it is why `twod_staggered.py` must stay. The price is wall time: the M = 10
solve is a 1458-dimensional generalized pencil (712–911 s on this contended box, against 7.3 s for
the hybrid at n_orders = 11).

**Fix.** The plateau itself is a scope statement, not a bug — but two things follow that are not.
(i) The tolerance is 2.5 decades looser than the achievable floor, so the tripwire can never catch
a genuine 1e-3-class regression: tighten `_PASSIVE_TOL_2D` toward the measured clean floor (≈3e-3
at these settings), or add a second, tighter *advisory* threshold. (ii) Because energy closure and
per-order error move in OPPOSITE directions on the fixture above, the closure test should not be
the only convergence signal the entry offers — a cheap "T00 between consecutive `n_orders`" drift
check (what `stabilize=True` already computes, at a fraction of the cost) would catch the 35 %
swing the energy test calls an improvement. And point the docstrings at
`pmm_efficiency_2d_staggered` / `PMM2DStackPure` (no floor) for any accuracy-critical use, as
`pmm_efficiency_2d`'s docstring already does but `pmm_jones_2d`'s does not.

---

### **[P2] The JAX twins still build the dense Kronecker projector pair the NumPy path's F5 audit deleted, and compute the per-axis projectors twice** — `lumenairy/elements/pmm/_jax_twod.py:158`, `:216`, `:242-248`

`_static_prep_cell` calls `_projectors(ax, ay, ox, oy)` (line 216), which materializes
`Tp = np.kron(Ty, Tx)` — an `(Nf, N)` dense array — and `Tpinv = np.linalg.pinv(Tp)`. Lines 242–248
*then* build `Tx/Txp/Ty/Typ` again (two more `pinv`s) for the factorized `Gx0F/Gy0F/IprojF`, with a
comment explaining that materializing the dense form is "~110 GB at N ~ 83k". `Tp`/`Tpinv` are
nevertheless kept and used by the jnp `_proj` sandwich
(`_jax_twod_jones.py:137-140`, `_jax_twod` cell path). `_static_prep` (the pillar branch, line 158)
does the same and additionally forms a dense `N×N` `Minv = np.diag(1.0/Mdiag)` and two dense
`Minv @ kron(...)` products (lines 165–167).

At the documented ceiling `_MAX_NODAL_DOF = 150_000` with `n_orders = 11` (Nf = 529) that pair is
≈ 1.4 GB plus an `O(N·Nf²)` pinv — exactly what `twod._sandwich_factorized` (`twod.py:497-509`,
measured "220–1078× faster / 64–138× less memory") exists to avoid on NumPy.

**Fix.** Build `Tx/Txp/Ty/Typ` once; drop the `_projectors` call; express the jnp `_proj` as the two
per-axis `einsum` contractions of `_sandwich_factorized`. The per-axis pieces are already computed
three lines below.

---

### **[P2] Diagonal mass matrices are inverted with `np.linalg.inv` / `np.linalg.solve` on the separable branches** — `twod_jones.py:191`, `twod.py:484`, `twod.py:490`, `twod.py:389`

`_build_axis` returns an exactly diagonal GLL mass `M` (verified: `np.allclose(M, diag(diag(M)))`
is `True`). The crossed-cell branches already exploit that (`mdx = np.diag(ax["M"])`,
`twod.py:546`), but the SEPARABLE branch of `_tensor_layer_modes` still does `Minv = np.linalg.inv(M)`
(`twod_jones.py:191`) and `twod._axis_ops_1d` does `Minv = np.linalg.inv(M)` plus
`np.linalg.solve(P_inv, M)` (`twod.py:484`, `:490`).

**Measured** (`q9_perf.py`, `q9.log`), degree 11 / 3 strips → n = 33: 200 × `np.linalg.inv(M)` =
8.8 ms vs 200 × `1.0/np.diag(M)` = 0.62 ms → **14×**, and the gap grows as O(n³) vs O(n). The
separable branch is the one a 1-D grating layer in a 2-D stack takes, i.e. the user's QWP geometry.

---

### **[P2] At `n_orders ≥ 10` the S-matrix cascade costs ~1.7× the eigensolve for a 2-layer tensor stack** — profile, `stack2d.py::solve`

**Evidence** (`q9_perf.py`, cProfile, `PMM2DStackHybrid` with two distinct in-plane tensor layers,
degree 11, `n_orders = 10` → Nf = 441, 2Nf = 882):

| site | calls | cumtime |
|---|---|---|
| `numpy.linalg.eig` | 2 | 6.598 s |
| `_redheffer_star` (`rcwa/_core.py:2696`) | 4 | 6.188 s (tottime 4.808 s) |
| `_interface_smatrix` (`pmm/_core.py:1846`) | 3 | 5.303 s |
| `numpy.linalg.inv` | 11 | 2.418 s |
| `numpy.linalg.solve` | 6 | 2.271 s |
| `twod_jones._proj` | 10 | 0.467 s |

The cascade + interface algebra (≈11.5 s) exceeds the eigensolve (6.6 s). The existing
`cascade='fused'` / `'tree'` knobs address the star product but are not the default, and the
eleven `inv` + six `solve` calls on 882×882 blocks are the next target
(`_guarded_inverse` accounts for 2.238 s of the 2.418 s of `inv`).

Positive measurement on the same run: the even-parity fold is worth **4.3×** —
`pmm_jones_2d` on a C4 Si pillar at degree 11 / `n_orders=10` takes 2.64 s with `symmetry='auto'`
against 11.24 s with `symmetry=False`.

---

### **[P2] `eps_cell` means two incompatible things across the two 2-D PMM families, and the mistake is an unguarded ~1000× cost cliff** — `twod.py:271-303` / `twod.py:340-356` vs `twod_staggered.py:3176-3185`, `stack2d_pure.py::add_layer`

**What is wrong.** Both families take a parameter called `eps_cell`, described in both docstrings as
the layer permittivity over the unit cell, and they mean different things:

* **Hybrid** (`pmm_efficiency_2d_cell`, `pmm_jones_2d`, `PMM2DStackHybrid.add_layer`): a **PIXEL**
  grid. `_cell_to_walls_tile` (`twod.py:271-303`) makes a wall only where adjacent rows/columns
  actually differ, so redundancy is free — a 12×12 array and the 3×3 that describes the same
  half-fill pillar cost the same. A cost guard (`_validate_cell_cost` / `max_nodal_dof = 150_000`)
  catches over-refinement and names the remedy.
* **Staggered** (`pmm_efficiency_2d_staggered`, `pmm_jones_2d_staggered`,
  `PMM2DStackPure.add_layer`): a **SEGMENT** grid. Every row/column IS an element; the per-component
  DOF is `(Nx(M−1))·(Ny(M−1))` and the generalized pencil is `2(Nx(M−1))²`. There is **no** cost
  guard at all.

Handing the staggered entry the same 12×12 half-fill-pillar array at `degree=6` therefore builds a
**7200×7200** QZ pencil where the geometry needs 450×450 (3 segments/axis).

**Evidence** (measured on this box, CPU time so load is not the explanation; `q12_stag.py`,
`q6_cache_parity.py`, killed after these readings):

| call | segments/axis | pencil | CPU | RSS |
|---|---|---|---|---|
| `pmm_efficiency_2d_staggered(12×12 array, degree=6, n_orders=3)` | 12 | 7200 | **498 s** | **8.4 GB** |
| `PMM2DStackPure(n_modes=5).add_layer(eps_cell=12×12 array)` (1 layer) | 12 | 4608 | **1096 s** | **3.9 GB** |
| `pmm_efficiency_2d_cell(same 12×12 array, degree=11, n_orders=5)` | merged to 3 strips | — | **0.22 s** | < 1 GB |
| `pmm_efficiency_2d_cell(same 12×12 array, degree=11, n_orders=11)` | merged to 3 strips | — | **7.28 s** | < 1 GB |

Neither staggered call raised, warned, or printed anything. The staggered docstring does say
"constant value PER SEGMENT" and enforces `Nx == Ny`, but nothing says a redundant grid is a cubic
cost multiplier, and the parameter name plus the sibling family's behaviour actively invite it.

**Fix.** (a) Add the missing cost guard: refuse (or warn) when `2·(Nx(M−1))²` exceeds a budget, and
put the *merged* segment count in the message ("your grid has 12 segments/axis but only 3 distinct
strips — pass the merged grid or raise the cap"). (b) At minimum, compute the merged strip count
with the `_cell_to_walls_tile` logic that already exists three files away and say so. (c) Cross-
reference PIXEL vs SEGMENT in both families' `eps_cell` docs, the way CONVENTIONS §11 cross-
references `fff_nv`.

---

### **[P3] `_sqrt_decay`'s round-3 docstring guarantee — "a flipped mode is by construction a PROPAGATING one" — is false, and the quoted price understates the bound** — `lumenairy/elements/rcwa/_core.py:1379-1386`, predicate at `:1428`

The predicate is `flip = (|Re(r)| <= band*scale) & (r.imag < 0)` with
`scale = max(max|r|, 1.0)` and `band = 1e-8`. It tests proximity of the *real part* to zero
relative to the *spectrum's top*, not whether the mode is propagating.

**Counterexample, run** (`q4_branch.py`, `q4.log`):

```
_sqrt_decay(np.array([1e-20 - 1e-30j]))  ->  -1.000000e-10 + 5.000000e-21j
```

`lam² = +1e-20` is a positive real — an **evanescent** mode — and it is flipped, giving
`Re(lam) < 0`, i.e. `|exp(-lam k0 L)| > 1`.

**The bound.** A flipped mode can carry `|Re(r)|` all the way up to `band*scale`, so
`|X| <= exp(band · max|lam| · k0 · L)`. Measured with a two-mode spectrum at the docstring's own
worst `k0 L = 1.1424e7`:

| spectrum scale `max|r|` | `|X|` |
|---|---|
| 1 | 1.0588 |
| 1e2 | 3.02e+02 |
| 1e4 | 1.17e+248 |

against the docstring's "worst `|X| - 1 = 1.752949e-09`". In the regimes this partition's entries
actually run (`k0 L ≈ 2`, `max|lam| ≈ 30`) the excess is ≈ 6e-7 and harmless — but the written
guarantee is not what the code implements, and the bound grows with thickness × spectrum scale.

**Confirmed working** on the same probe: the round-3 `-r` (rather than `conj(r)`) choice *does*
deliver continuity. A 21-point θ sweep from 0 to 8e-6 rad across the band on a
near-degenerate-eps hybrid fixture gives `max|ΔJxx| / median|ΔJxx| = 1.95` — no step.
(Round 2's reported failure mode was a 13.8× step.)

**Fix.** Either state the real bound in the docstring, or tighten the predicate to
`& (r.imag**2 > r.real**2)` — "near the imaginary axis" rather than "near the origin" — which
excludes the near-zero evanescent case at no cost to the on-cut case it exists for.

---

### **[P3] `symmetry` is absent from `PMM2DStackHybrid._mode_key`, so mutating it between solves serves a stale modal set** — `lumenairy/elements/pmm/stack2d.py:874-877` vs `:934-936`

`_mode_key`'s `common` tuple is `(wl, k0, kx0, ky0, period_x, period_y, n_orders, formulation)`.
`_build_layer_modes` passes `block_eig=self.symmetry` to `_tensor_layer_modes` for tensor layers,
and `self.symmetry` is a plain public attribute.

**Evidence** (`q6_cache_parity.py` §A, `q6.log`): an out-of-plane tensor layer (
`[[4,0,0.6],[0,3.4,0],[0.6,0,3.9]]` pillar in eps 2.25), solve with `symmetry=True`, set
`symmetry=False`, re-solve → the Jones is **bit-identical to the `symmetry=True` answer**
(`|J_b − J_a| = 0.0`), while a fresh `symmetry=False` object differs by 7.16e-16. The stale hit is
real; its magnitude is machine noise because the parity-sign block reduction is exact. Everything
the W7 A11 hardening covered *does* work — `q6b_cache.py` shows `|reused − fresh| = 0.0` after
mutating `formulation`, `symmetry` (scalar layer), `cascade`, `degree`, `grade` and `n_orders`.

**Fix.** Append `bool(_symmetry_on(self.symmetry))` to `_mode_key`'s `common`.

---

### **[P3] `period_x` / `period_y` are plain public attributes that also feed `add_layer`-time derived state, so the `_geom_key` hardening cannot protect them** — `lumenairy/elements/pmm/stack2d.py:365-393`, `:560-610`

`_geom_key` includes `float(self.period_x), float(self.period_y)`, so the cache correctly misses on
a period change. But `add_layer` has already frozen `L["xw"]`, `L["yw"]` (wall positions in
**metres**) and `L["el_x"] / L["el_y"]` from the OLD period. After the mutation the solve runs
`_build_axis(new_period, old_wall_metres, ...)`, i.e. a silently different duty cycle.

**Evidence** (`q6b_cache.py`, `q10_misc.py` §A): a 50 %-duty pillar at `period_x = 0.9 µm`,
mutated to `0.945 µm`, re-solved → `|J_reused − J_fresh(same cell)| = 4.390e-01` while the
parameter's genuine sensitivity is `|J_fresh − J_base| = 6.492e-01`. Every other public attribute
in the same test reads `0.000e+00`.

**Fix.** Make `period_x` / `period_y` read-only properties after the first `add_layer` (or store the
walls as *fractions* and re-derive metres at solve time), and say so in the class docstring beside
the existing W7 A11 note.

---

### **[P2] `PreparedPMM2D.solve` silently differs from `pmm_efficiency_2d` in three ways — including a MISSING energy tripwire that I reproduced firing on one path and not the other** — `lumenairy/elements/pmm/twod.py` (the `PreparedPMM2D` class docstring, `PreparedPMM2D.solve`, `prepare_pmm_2d` / `prepare_pmm_2d_cell`)

The class docstring claims *"reproduces `pmm_efficiency_2d(...)` to ~1e-13 (the only delta:
`Gx0F/k0` reorders one division vs the single call; the uniform-layer path is byte-identical)"*.
Measured (`q11_prepared.py`, `q11.log`), pillar, degree 9, `n_orders = 4`:

* vs `pmm_efficiency_2d(symmetry='auto')` (the entry's DEFAULT): `max|dR| = 6.33e-14`,
  `max|dT| = 1.08e-13`;
* vs `pmm_efficiency_2d(symmetry=False)`: `max|dR| = max|dT| = **0.000e+00**`.

So the ~1e-13 is not a division reorder — it is the **missing even-parity fold**. `prepare_pmm_2d`
takes no `symmetry` parameter at all and `PreparedPMM2D.solve` never calls `_symmetric_solve_2d`.
Two further silent gaps:

* **`truncation`**: `pmm_efficiency_2d(truncation='circular')` retains 49 orders;
  `pmm_efficiency_2d_vs_wavelength` retains 81 and rejects the keyword with `TypeError` — a user
  sweeping a circular-truncation design silently gets rectangular.
* **`_warn_lossless_energy_2d` is missing — MEASURED.** It is called only from the
  `if not stabilize:` arm of `pmm_efficiency_2d` and of `pmm_efficiency_2d_cell`, never from
  `PreparedPMM2D.solve`, which runs the identical ill-conditioned `_layer_modes_projected`.
  `q13_warn.py` scans for a config where the direct entry warns and finds one immediately —
  eps 12.25 pillar, 50 % duty, Px = Py = 0.9 µm, λ = 1 µm, d = 0.3 µm, **degree 7, n_orders 2**:

  ```
  DIRECT   pmm_efficiency_2d      : sum(R+T) = 1.054125   -> UserWarning (lossless closure violated)
  PREPARED prepare_pmm_2d(...).solve(wl): sum(R+T) = 1.054125   -> 0 warnings
  ```

  Same 5.4 % energy excess on a provably lossless structure, identical to the last digit, and the
  sweep path returns it silently. `pmm_efficiency_2d_vs_wavelength` /
  `pmm_efficiency_2d_cell_vs_wavelength` go through this path, so a whole wavelength sweep can sit
  in this regime with no signal.

**Fix.** Call `_warn_lossless_energy_2d` from `PreparedPMM2D.solve` (it needs only the eps list,
which `_prepare_pmm2d_core` already stores as `eps_reals`), thread `symmetry` and `truncation`
through `prepare_pmm_2d*`, and correct the "only delta" sentence.

---

### **[P3] `pmm_jones_2d` returns no TRANSMISSION Jones — the observable for a transmissive metasurface QWP — and nothing in the module points at the route that does** — `lumenairy/elements/pmm/twod_jones.py:497-510`

`pmm_jones_2d` returns `(orders, R_eff, T_eff, jones_reflection)`. So does `rcwa_jones_2d`. For the
user's LC-grating QWP out-coupler the phase observable is the *transmission* Jones, reachable only
via `PMM2DStackHybrid.jones_transmission()` / `PMM2DStackPure.jones_transmission()` (the
`PerOrderAmplitudesMixin`, `pmm/_core.py:2025`) — neither the module docstring nor the Returns
section mentions this. `rcwa_jones_1d` solved the same problem with
`return_jones_transmission=True` and documents the cross-engine seam in its Notes; the 2-D entries
have no such note.

**Fix.** Add `return_jones_transmission=True` to `pmm_jones_2d` (the amplitudes `tx`/`ty` are
already computed at `twod_jones.py:804`; it is a one-line extraction), or at minimum add the
seam note pointing at `PMM2DStackHybrid.jones_transmission`.

---

### **[P3] CONVENTIONS §7.1's "the 1-D solvers return `te`/`tm` (`s`/`p`)" contradicts `rcwa_jones_1d`'s own docstring and the measured ordering** — `CONVENTIONS.md:198-204`

`rcwa_jones_1d`'s Returns block (`rcwa/oned.py:1198-1206`) says *"Zeroth-order Jones reflection
matrix in the lab (x, y) basis"*. Measurement agrees with the docstring, not with §7.1's phrasing:
for the Λ/λ = 0.2 form-birefringent grating below, `rcwa_jones_1d`'s `J[0,0]` matched
`PMM2DStackHybrid`'s `Jxx` (arg +1.9769 vs +1.9746, |·| 0.9808 vs 0.9804) and `J[1,1]` matched
`Jyy` (arg −2.5598 vs −2.5590, |·| 0.9175 vs 0.9173). Index 0 is **x ≡ tm**, index 1 is **y ≡ te**.
§7.1's "return te/tm" wording invites the reverse (te-first) reading, which silently transposes a
retardance sign. Not my lane to fix, but it is directly upstream of the 2-D Jones contract I was
asked to verify.

---

## The specific probes, answered

**1. `pmm_jones_2d` Jones basis and sign convention — CORRECT on all five sub-tests.**

*(a) Unpatterned limit vs an independent 3-medium TMM* (`p1a.py`, `p1b.py`, `tmm.py`; my own
`Z = kz/ε` Airy oracle, not a library function). Uniform eps-4 film, d = 0.3 µm, λ = 1 µm,
n_sup = 1, n_sub = 1.5, Λ = 0.47 µm (off the Wood anomaly):

| θ | `\|Jxx − r_x(TM)\|` | `\|Jyy − r_s(TE)\|` | `\|Jxy\|`, `\|Jyx\|` |
|---|---|---|---|
| 0° | 6.21e-17 | 6.21e-17 | 0.0 |
| 30° | 1.42e-16 | 1.24e-16 | 0.0 |

So `Jxx ↔ p/tm`, `Jyy ↔ s/te`, with the incident wave normalized to **unit tangential E**
(at θ=30° that is `|E_inc| = sec θ`), and `Jxx(θ→0) = Jyy(θ→0)` — the isotropic-limit sign
convention, not the `r_p = −r_s` one. At Λ = 0.5 µm the residual rises to 9.8e-08 because an order
sits exactly on the layer's Wood anomaly and `_grazing_safe_wavelength` nudges λ — expected and
documented.

*(a′) φ-covariance of the uniform limit.* At φ = 45° the Jones rotated into the incident (p, s)
frame reproduces the same `r_x`, `r_s` to 2e-16 with off-diagonals ≤ 2e-16.

*(b) 90° rotation of a 1-D grating.* `max|J(x-periodic) − swap(J(y-periodic at φ+90°))|`:
1.96e-14 (n=3, θ=0), 7.85e-14 (n=3, θ=20°), 3.30e-14 (n=5, θ=0), 5.44e-14 (n=5, θ=20°),
against `max|J| ≈ 0.99`. **Correct.** (Contrast the P1 finding: the *stack* fails this.)

*(c) Mirror-symmetric cell at normal incidence.* Rectangle in air: `|Jxy| = 2.61e-14`,
`|Jyx| = 1.32e-15` (fold on); `7.93e-15` / `1.14e-13` (`symmetry=False`). Fold-vs-full
`ΔJ = 1.75e-13`. **Correct.**

*(d) C4 square pillar at normal incidence.* `|Jxx − Jyy| = 2.42e-13` (laurent), `8.25e-14` (li);
`|Jxy| ≤ 3.2e-14`; the two `R` rows agree to 1.5e-14. **Correct.**

*(e) Reciprocity / inversion.* On an **L-shaped cell with no mirror line** at normal incidence the
reflection Jones is symmetric — `|Jxy − Jyx| = 2.09e-14` (n=5) and `3.75e-13` (n=9) against
`|Jxy| = 0.32` / `0.21`. That is the Lorentz-reciprocity statement for the zeroth-order reflection
matrix and it **holds**. Separately, for a centro-symmetric pillar,
`J(θ=20°, φ=30°) = J(θ=20°, φ=210°)` to 9.83e-14 — the in-plane inversion symmetry.

**2. Waveplate physics (the user's use case) — sign convention CONFIRMED, accuracy quantified.**

Form-birefringent Si/air grating, Λ/λ = 0.2 (Λ = 200 nm, λ = 1 µm), duty 0.5, n_Si = 3.48.
0th-order EMT: `n_∥ (along the grooves, y) = 2.56031`, `n_⊥ (x) = 1.35921` → slow axis = **y**;
QWP thickness `d = λ/4/(n_∥ − n_⊥) = 208.14 nm`. Retardance defined as
`wrap(arg(J_yy^t) − arg(J_xx^t))` (`q2_qwp.py`, `q2b_qwp.py`; logs `q2.log`, `q2b.log`):

| source | retardance | error vs rigorous |
|---|---|---|
| `rcwa_jones_1d(n_orders=60, li)` transmission Jones | **+100.0660°** | — (reference) |
| 0th-order EMT slab (my own analytic TMM) | +95.8730° | −4.193° |
| `PMM2DStackHybrid.jones_transmission()`, n_orders 5 | +97.6137° | −2.452° |
| … n_orders 9 | +98.6685° | −1.397° |
| … n_orders 15 | **+100.2411°** | **+0.175°** |

The retardance is **POSITIVE** with the **slow** axis carrying it — i.e. the slow axis picks up
`exp(+i·retardance)`, exactly CONVENTIONS §7's *"Waveplate slow-axis phase `exp(+i * retardance)`"*.
No conjugation is needed to drop a PMM-2D transmission Jones into a `JonesField` pipeline.
Cross-pol leakage `|Jxy|` is 4.1e-15 at n=5 but 1.7e-11 at n=15 (the projection conditions
worse with truncation).

Closing the Stokes loop: an ideal QWP built as `diag(1, exp(+iπ/2))` with the **fast** axis rotated
to +45° acting on x-pol gives `S = (1, 0, 0, −1)` under the library's own
`S3 = −2 Im(Ex conj(Ey))` (`CONVENTIONS.md:157`) — i.e. **S3 = −1 ('left')**, matching the
convention table's claim.

Two practical notes for the QWP project: (i) the hybrid needs `n_orders ≈ 15` at Λ/λ = 0.2 to get
inside 0.2° of retardance, and `n_orders = 9` is still 1.4° out; (ii) `pmm_jones_2d` itself cannot
deliver this number — see the missing-transmission-Jones finding above.

**3. `fff_nv` — the raise works, the separable reduction is algebraically correct, and it is the
best PMM formulation measured.**

* *The crossed-cell raise* (`q3_fffnv.py`): `pmm_jones_2d(..., formulation='fff_nv')` on an
  x-AND-y-patterned cell raises `ValueError` with the documented "requires a SEPARABLE
  (single-orientation…) cell" message, while `rcwa_jones_2d(..., formulation='fff_nv')` on the same
  cell succeeds (`Jxx = -0.99960 - 0.02799j`). Exactly CONVENTIONS §11's table.
  `fff_nv` + an out-of-plane tensor also raises (`twod_jones.py:135-139`). A UNIFORM cell silently
  accepts `fff_nv` — harmless, since all three rules coincide there.
* *The separable reduction* (`twod_jones.py:199-213`). I re-derived Li's 1-D anisotropic
  factorization for walls normal to **n** with in-plane tensor `[[a,b],[c,d]]`: continuity of
  `D_n` and `E_t` gives
  `D̂_n = [[1/a]]⁻¹ Ê_n + [[1/a]]⁻¹[[b/a]] Ê_t` and
  `D̂_t = [[c/a]][[1/a]]⁻¹ Ê_n + ([[d − cb/a]] + [[c/a]][[1/a]]⁻¹[[b/a]]) Ê_t`.
  The code's four slots `inn`, `inn @ b_nt`, `b_tn @ inn`, `schur + b_tn @ inn @ b_nt` are those
  four expressions **term for term**. The wall-normal index selection
  (`nn = 0 if nsy == 1 else 1`) is right, and `EZZ` correctly stays DIRECT (`E_z` is tangential to
  a vertical wall — Li 1997 Eq. 27). **Correct.**
* *Convergence vs the Li-factorised RCWA-2D at equal truncation* — the table in the
  `li`-vs-`laurent` P2 finding above.
  `fff_nv` beats `laurent` by 10× and `li` by 25× on the reflection Jones of a high-contrast Si
  stripe; `rcwa_jones_2d(li)` still beats `fff_nv` by 10× at the same `n_orders`, which qualifies
  `twod.py`'s module-docstring claim that the hybrid "tends to be at least as accurate" as RCWA at
  a given `n_orders` (true for the scalar efficiency entry it cites; not for this Jones case).
* *One perf note*: the even-parity fold is disabled for `fff_nv`
  (`twod_jones.py:745`, `formulation != "fff_nv"`), so the best formulation is also the slowest.
* *The "fff_nv degeneracy gate" of commit `1627f027`*: this is a **test** re-derivation, not a
  solver gate. `git show 1627f027 -- lumenairy/elements/pmm/` is EMPTY. The degeneracy is the
  layer-eps-exactly-equals-region-eps mode-match singularity, and the gate is
  `test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`,
  re-derived after the branch-cut fix cured the pathology it was asserting. Continuity is not a
  meaningful question for it; the underlying *cure* (the `_sqrt_decay` pin) I did test for
  continuity — see the `_sqrt_decay` finding above; it is continuous.

**4. Branch cuts — the rule is right, the census claim is not.** See the `_sqrt_decay` finding
above. Additionally I verified
the round-2 "ONE DEFINITION" claim by grep: `_sqrt_decay` is defined exactly once
(`rcwa/_core.py:1292`) and imported at four call sites in `pmm/` (`twod.py:92`, `_jax_twod.py:341`,
`_jax_twod_jones.py:107`, `_jax_stack2d.py:120`); `twod_staggered.py` retains only the comment at
line 2582 recording its deleted copy. The claim **holds**.

**5. `twod_staggered.py` is LIVE, not dead.** The changelog phrase *"the staggered copy confirmed
dead at runtime"* (`FIX_BRANCH_CUT_ROUND2_2026_09_11.md:169`, "The staggered copy is deleted rather
than routed. It was never called.") refers **only** to a private dead copy of `_sqrt_decay` inside
that module, already deleted. The module itself is an independent engine — Granet 2023 staggered
modified-Legendre, the no-floor 2-D PMM — reached by three public surfaces:
`pmm_efficiency_2d_staggered`, `pmm_jones_2d_staggered` (both in `lumenairy.__all__`) and
`PMM2DStackPure`. **Recommendation: keep.** There is no maintenance cost to reclaim here, and the
hybrid's measured 1e-3-class energy floor (the third P2 finding) is the argument *for* it: on the
same grating the staggered engine closes to 1.8e-10 at M = 8 and is `n_orders`-invariant to
14 digits.

**6. `stack2d.py` vs `stack2d_pure.py` — two engines, not duplication.** `PMM2DStackHybrid` is the
Fourier-projected cascade (per-layer walls with no union-grid constraint, tapered z-staircases,
out-of-plane tensors, a JAX twin, an `n_orders` floor); `PMM2DStackPure` is the no-floor staggered
cascade (union grid or L2 mortar, no Fourier floor, `n_modes`-driven). A bitwise parity test is
**not applicable** — they discretize differently and the pure engine's whole point is that its
energy is `n_orders`-invariant. The mortar guards are correctly layered (a geometric
minimum-segment-width contract at grid-build time plus a conditioning backstop on the mortar
solves, `stack2d_pure.py:1659-1678`), and round 4's per-axis narrowing of the sliver-band warning
(`stack2d_pure.py:1711-1715`, `_stag_mortared_axes`) is the right fix for round 3's defect: the
warning must be conditioned on the axis that actually builds a cross-grid interface. I did not get
to a numeric ladder sweep across the guard threshold — declared as not-done.

**Cache key completeness** (`q6b_cache.py`): `formulation`, `symmetry` (scalar), `cascade`,
`degree`, `grade`, `n_orders` all give `|reused − fresh| = 0.0` after mutation — the W7 A11 fix
works. The two gaps are the `symmetry`-not-in-`_mode_key` finding (tensor layers) and the
`period_x/y` finding above.
**Thread-safety**: `LayerCache` holds a `threading.Lock` and every accessor takes it; `copy.copy`
clones share the cache object by design (the sweep's workers), which is now safe because the lock
is in the object rather than the solver. The generation stamp makes "a single solve never evicts
what it is about to reuse" true by construction. I read this design and found no defect.

**7. JAX twins — clean.** (`q7_jax.py`, `q7.log`.)

* x64 enforced: `RuntimeError: pmm_jones_2d: the JAX (differentiable) path requires double
  precision…` without `jax_enable_x64`. ✓
* NumPy/JAX forward parity on a both-axes-patterned Si pillar (degree 5, n_orders 3): RMS relative
  `R = 8.64e-15`, `T = 8.47e-11`, `J = 2.21e-14`. ✓
* `jax.grad` vs central FD, against the 1e-4 gate: `d(ΣT)/d(eps_pillar)` rel **1.60e-08**;
  `d/d(depth)` rel **1.47e-06**; `d/dθ` at θ=0.3 rel **4.09e-09**. All finite. ✓
* At exactly normal incidence the θ-gradient is the clean symmetry zero (−4.7e-17) and stays finite
  at θ = 1e-7 and 1e-5, with AD/FD agreeing to 3 % on the one-sided FD at 1e-5. This is the
  round-3 "finite gradients across the cut" claim, and it **holds**.
* No recompilation pathology: three successive calls take 0.092 / 0.082 / 0.100 s. But note this is
  *eager* jnp dispatch — nothing in `_jax_twod_jones` is `jit`-wrapped, so there is also no
  compilation win to be had; a user must wrap the call themselves.

**8. Energy / reciprocity** — the energy-plateau P2 finding and probe 1(e) above.

**9. Performance** — the four P2 performance findings above, plus: the `LayerCache` over a 6-wavelength
dispersive-style sweep at `n_orders = 6` reads `geom {entries 1, hits 5, misses 1, 174 KB}` and
`eig {entries 6, hits 0, misses 6, 21.97 MB}` — 3.66 MB per modal entry, linear in sweep length as
designed, against a 3.83 GB budget. Extrapolating to `n_orders = 11` (2Nf = 1058) an entry is
≈36 MB, so a 500-point sweep would exceed the budget and exercise the refuse-never-degrade arm.
That arm is correct (the value is still returned), but a user sweeping wide will silently lose all
eig reuse; `cache_stats()['eig']['refused']` is the signal and nothing surfaces it.

**10. Alternatives** — see the dedicated section below.

---

## Performance opportunities

1. **Drop the dense Kronecker projector from the JAX twins** (`_jax_twod.py:158`, `:216`): reuse the
   per-axis `Tx/Txp/Ty/Typ` already built three lines below and express `_proj` as the two einsum
   contractions of `twod._sandwich_factorized` (`twod.py:497`). Saves an `(Nf, N)` + `(N, Nf)` pair (≈1.4 GB at the
   documented ceiling), one `O(N·Nf²)` pinv, and two redundant per-axis pinvs.
2. **Stop inverting diagonal matrices** (`twod_jones.py:191`, `twod.py:484`, `:490`): measured 14×
   at n = 33, growing as O(n³)/O(n). The crossed branch already does this correctly.
3. **The cascade, not the eig, is the bottleneck above `n_orders ≈ 10`** (6.19 s of `_redheffer_star`
   + 5.30 s of `_interface_smatrix` against 6.60 s of `eig` for a 2-layer stack at Nf = 441).
   Making `cascade='fused'` the default (it only changes the association, and `'tree'` measured
   bit-identical to `'fast'` in `q6b_cache.py`) is free. The eleven `np.linalg.inv` calls
   (2.42 s, of which `_guarded_inverse` is 2.24 s) on 882×882 blocks are the next target — several
   are `inv` followed immediately by a matmul and should be `solve`.
4. **Reuse the `E_z`-eliminated in-plane operators across φ.** `_tensor_layer_modes`'s projected
   masses (`c`, `czz`, `cizz`, the `_proj` sandwiches) are functions of geometry + tensor only —
   `k0`, `kx0`, `ky0` enter afterwards through `GxF/GyF`. `PMM2DStackHybrid._geom_cache` exploits
   this for scalar layers (`lops` is cached) but `_build_layer_modes` passes `lops=None` for tensor
   layers (`stack2d.py:922`) and rebuilds the projected tensor operators on **every** wavelength and
   every angle. Caching them the way the scalar branch does would remove the `_proj` sandwiches and
   the `_axis_projection` pinvs from a sweep. (Measured share on my fixture is small — `_proj` was
   0.47 s of ~20 s — but it grows as `O(Nf² N)` with cell complexity, and `_axis_projection` is
   `O(nq · degree²)` per call.)
5. **`fff_nv` cannot fold** (`twod_jones.py:745`). Since `fff_nv` is only reachable on a separable
   cell, and a separable centro-symmetric cell at normal incidence satisfies the fold's
   precondition, the exclusion looks conservative rather than necessary: the `return_ops=True`
   branch would need the `fff_nv` `c` dict wired through, which is mechanical.
6. **Surface `cache_stats()['eig']['refused']`** when it becomes non-zero on a sweep, so a user
   losing all eig reuse learns about it.

## Alternative algorithms / methods

* **(a) Li-2003 successive full-tensor `L2 L1` factorization** (the rule `rcwa_jones_2d` uses for
  `fff_nv`, CONVENTIONS §11) **vs the PMM's separable reduction.** The PMM's reduction is exact for
  a constant wall normal and correctly *refuses* a crossed cell; Li's successive factorization
  handles a varying normal. My measurements (the `li`-vs-`laurent` P2 finding) say the RCWA-Li route is ~10× more accurate than
  the PMM `fff_nv` route at equal `n_orders` on a high-contrast separable Si cell, and ~250× more
  accurate than PMM `'li'`. **Recommendation for the user's QWP:** for a purely 1-D LC grating use
  `rcwa_jones_1d` (or `pmm_jones_1d`, which is no-floor); reach for the 2-D PMM only when the cell
  is genuinely crossed or the stack needs per-layer walls.
* **(b) Matched-coordinate / adaptive-spatial-resolution FMM** (Granet, *JOSA A* 16:2510 (1999);
  Weiss et al., *Opt. Express* 17:8051 (2009)). A coordinate transform that stretches resolution
  toward the walls. This is the regime `_tensor_layer_modes` explicitly names when it refuses a
  crossed `fff_nv` cell ("the research-grade matched-coordinate FFF regime"). It would lift that
  refusal and is the standard cure for the Laurent floor on crossed high-contrast cells — exactly
  the 1e-3-class floor measured in the two convergence P2 findings above. This is the single
  highest-value algorithmic addition for this partition.
* **(c) Aperiodic FMM with PML** (Lalanne & Silberstein, *Opt. Lett.* 25:1092 (2000)). Not relevant
  here — every entry in this partition is genuinely doubly periodic.
* **(d) Chandezon / C-method** (Chandezon et al., *JOSA* 72:839 (1982)). Excellent for smooth
  profiles; the wrong tool for the axis-aligned rectangular walls this module is built around, and
  the staggered engine already resolves those walls exactly.
* **(e) For the QWP specifically: 0th-order EMT + a 2nd-order correction as a fast surrogate.**
  Measured, the 0th-order Rytov EMT at Λ/λ = 0.2 is **4.19°** out on retardance against the
  rigorous answer — a usable first guess for a design sweep but not a final number. Rytov's
  second-order terms (`n²_∥,2 = n²_∥,0 + (π Λ f(1−f)/λ)²(n_r²−n_g²)²/3` and the dual for `n_⊥`;
  Lalanne & Lemercier-Lalanne, *J. Mod. Opt.* 43:2063 (1996)) typically recover most of that at
  Λ/λ = 0.2. Concrete suggestion: a `lumenairy/elements/emt.py` second-order variant would give the
  QWP project a ~µs surrogate that is within a few tenths of a degree, with the rigorous solver
  reserved for the final check.
* **(f) Circular (Lalanne-1997) truncation is already implemented** for
  `pmm_efficiency_2d[_cell]` (`twod.py:833-848`, Nf → ~(π/4)Nf, eig ×0.48) but is **not** exposed on
  `pmm_jones_2d`, on `PMM2DStackHybrid`, or on the `*_vs_wavelength` helpers. Extending it is
  mechanical and roughly halves the eig cost at fixed accuracy.

## Code organization observations

* `twod_jones._tensor_layer_modes` (`twod_jones.py:125-345`, 220 lines) carries four
  discretization branches (uniform / separable / separable-`fff_nv` / crossed) plus the out-of-plane
  Schur fold plus the `return_ops` fold gate plus the `block_eig` gauge. The four branches duplicate
  the `_proj` / `_mass` shape three times with different spellings. Splitting the operator assembly
  (pure, geometry+tensor, `k0`-free) from the eigen-solve dispatch would also make the caching
  opportunity in Performance §4 trivial.
* `_assemble_2d` (`twod.py:385-407`) is dead in production (referenced only by
  `tests/unit/test_v5_14_0_pmm_audit_fixes.py` as the dense-kron reference oracle and by
  `validation/probe_scope_bor_guards/e_sliver.py`). That is a legitimate kept-reference; a one-line
  "reference implementation, not on any live path" note at the top would stop the next auditor
  re-deriving it.
* `PMM2DStackHybrid` is 2028 lines with `solve` alone spanning ~480 (1156–1640). The
  per-layer-modes / cascade / far-field phases are cleanly separable and each has its own cache and
  its own gate; three methods would read better than one.
* The `# noqa: F401` re-export of `is_jax_array` in `_jax_twod.py:50` and `_jax_twod_jones.py:80`
  is dead — the dispatch imports it from `...backend` directly (`twod_jones.py:63`).
* Comment density: `twod_jones.py` and `stack2d.py` are roughly 45 % prose. The measured-number
  comments are genuinely valuable (I used several as hypotheses), but several now describe the
  *history* of a line rather than the line (e.g. `twod.py:86-90`, `twod_jones.py:545-560`). Moving
  the archaeology into `docs/audits/` and leaving the invariant in the code would cut the files
  substantially without losing the evidence.

## Unverified suspicions

* `_pmm_jones_2d_at` computes `rz = -(kxv*rx + kyv*ry)/kz_ref` (`twod_jones.py:790`). For a
  wave travelling in `-z` the divergence condition gives `rz = +(kx rx + ky ry)/kz`, so the sign
  looks inverted — but `rz` is used only inside `|rz|²`, so no output moves. If the longitudinal
  reflected amplitude is ever exposed (it is not today), this would become real. Same expression
  at `twod.py:919` and `stack2d.py:1590`.
* `_jax_twod._host_incidence_guard` uses `Re(n_superstrate)` as the incidence index
  (`_jax_twod.py:112-118`) where the NumPy path uses `Re(sqrt(conj(n_sup²)))`. They agree to ~1e-5
  for a weakly lossy superstrate and exactly for a real one; I did not construct a case where the
  Wood-anomaly nudge decides differently between the two.
* The two bare `except Exception: pass` in `_host_incidence_guard` (`_jax_twod.py:124`, `:130`)
  are documented as "traced entries are dropped from the nudge list", but they would equally
  swallow a genuine malformed-eps error and silently disarm the nudge. I did not construct that case.
* I did not exercise the mortar guard ladder (`_guarded_mortar_solve`,
  `_warn_stag_sliver_band`) numerically — the round-3/4 "exactly-one-promoted-side" and
  "family-scoped bars" decisions are read-only in this report.

## Checked and found correct

* **Jones basis, columns, and sign convention** of `pmm_jones_2d` against an independent TMM at
  θ = 0° and 30°, φ = 0° and 45° — 1e-16 (probe 1a/1a′).
* **90° rotation covariance** of `pmm_jones_2d` itself — 5e-14 (probe 1b).
* **Mirror symmetry → zero cross-pol** — 2.6e-14 (probe 1c).
* **C4 symmetry → Jxx = Jyy** — 2.4e-13 (probe 1d).
* **Lorentz reciprocity of the reflection Jones on a chiral cell** — `|Jxy − Jyx| = 2.1e-14`
  against `|Jxy| = 0.32` (probe 1e).
* **In-plane inversion symmetry** `J(θ,φ) = J(θ,φ+π)` for a centro-symmetric cell — 9.8e-14.
* **Even-parity fold** vs the full solve — 1.75e-13, and worth 4.3× wall time.
* **Waveplate slow-axis sign** `exp(+i·retardance)` and the `S3 = −1` QWP statement (probe 2).
* **`fff_nv`**: the crossed-cell raise, the out-of-plane raise, the algebra of the separable
  reduction against Li's 1-D anisotropic factorization, and that it is the most accurate PMM
  formulation measured (probe 3).
* **`_sqrt_decay` is one definition**, correctly imported at all four `pmm/` call sites; the
  round-3 `-r` flip is continuous across the band (probe 4).
* **`twod_staggered.py` is live** and its "dead copy" was only the deleted `_sqrt_decay` (probe 5).
* **`_geom_key` / `_mode_key`** are complete for `formulation`, `degree`, `grade`, `n_orders`,
  walls, element counts and tile bytes — the W7 A11 hardening works (probe 6). `cascade` is in
  neither key and does not need to be: it only reassociates the Redheffer fold, and
  `cascade='tree'` measured bit-identical to `'fast'` (`|Δ| = 0.0`).
* **`LayerCache`**: locked, generation-stamped, RAM-priced, refuse-never-degrade; shared by
  `copy.copy` clones by design and safe because the lock lives in the cache (probe 6).
* **JAX twins**: x64 enforcement, NumPy parity to 1e-14, AD-vs-FD to 1e-8…1e-6 on eps / depth /
  θ, finite gradients at exactly normal incidence, no per-call recompilation (probe 7).
* **The frame-anchor phase** (`stack2d.py:1570-1573`, `:1607-1609`) is applied to the transmitted
  amplitudes only and only for layers that genuinely enter a sheared frame
  (`_layer_enters_slant_frame`), leaving `R`, `T` and the reflection Jones bit-identical — the
  derivation and the gating both read correctly.
* **The slant refusals** on the JAX dispatch (`twod_jones.py:576-597`, `stack2d.py:1273-1284`)
  close a genuine silent-wrong (the jnp twins take no `slant`), and the `_slanted_cell_is_a_frame_noop`
  escape hatch is consistent with `_layer_enters_slant_frame`'s own uniform-tile test.
