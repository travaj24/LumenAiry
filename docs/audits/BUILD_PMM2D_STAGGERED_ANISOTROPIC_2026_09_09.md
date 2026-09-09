# BUILD -- in-plane (block-form) anisotropy for the PURE staggered 2-D PMM

Date: 2026-09-09.  Status: Stage A BUILT (Sections 1-3 of
`docs/audits/PLAN_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md`).  Branch
`feat/pmm2d-staggered-anisotropic`, worktree `C:/tmp/lum_aniso`, off `main`
`f70628d` (= 5.42.1 + the three BOR SEM commits).  Stage B (out-of-plane) is
NOT in this build: an out-of-plane tensor raises `NotImplementedError`.

Measurement environment for every number below unless stated otherwise:
Windows 11, CPython 3.14, NumPy 2.x, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1`, worktree-pinned `PYTHONPATH`.  All probes assert
`lumenairy.__file__` starts with the worktree path.

---

## 1. What was built

### 1.1 Library

| item | file | what changed |
|---|---|---|
| `Granet2DTransverseE(..., eps_cell)` | `lumenairy/elements/pmm/twod_staggered.py` | accepts `(Nx, Ny)` scalar OR `(Nx, Ny, 3, 3)` block-form tensor.  ONE `_assemble` body: the component maps `e11/e12/e21/e22/e33` are `None` on the scalar path, which makes every eps-weighted call fall back to `self.eps_cell` -- the shipped isotropic arithmetic, unchanged (gate G1). |
| `_eps_weighted(refx, refy, wmap=None)` / `_eps_dir(..., wmap=None)` | same | new optional per-cell WEIGHT MAP.  `None` = `self.eps_cell` (the isotropic default, byte-identical).  The tensor assembly passes ONE permittivity component per block. |
| `Et_offdiag` | same | new retained attribute: `(Et_12, Et_21)` for a tensor, `None` for a scalar (so the isotropic solver retains not one byte more than before -- audit P3-37 holds). |
| `_region_modes` | same | folds `Et_offdiag` into the Eq. 25 `Lhh` when present.  `H1` still lives in `V2` and `H2` in `V1`, so the interface match stays a SQUARE modal match and the cascade is untouched. |
| `_homog_geom_cache` | same | now RAISES on a tensor assembly: the eps-free split needs `Meps33 = eps*G3` and `Kzt = eps*Kzt0` to cancel, which a tensor breaks. |
| `_require_block_form(fn_name, tile33)` | same | new Stage-A gate.  Out-of-plane -> `NotImplementedError` naming `pmm_jones_2d`; `e33 == 0` -> `ValueError`.  REUSES (does not copy) the hybrid's `_tile_is_offplane` relative `1e-12 * scale` floor and `_require_nonzero_ezz`, so the two engines' contract cannot drift apart. |
| `_validate_stag_cell(fn_name, eps_cell)` | same | shared shape / SQUARE-grid / block-form validation, so each entry names ITSELF in its message while enforcing one contract. |
| `pmm_jones_2d_staggered(...)` | same | NEW PUBLIC ENTRY (below). |
| `pmm_efficiency_2d_staggered` | same | a 4-D `eps_cell` now raises a `ValueError` naming `pmm_jones_2d_staggered`.  Otherwise untouched. |
| `PMM2DStackPure.add_layer` | `stack2d_pure.py` | `eps=` accepts a scalar or a `(3, 3)` block-form tensor (uniform anisotropic layer); `eps_cell=` accepts `(Nx, Ny)` or `(Nx, Ny, 3, 3)`.  Union-grid rule unchanged (compared on the `(Nx, Ny)` prefix). |
| `PMM2DStackPure.solve` | same | a uniform TENSOR layer is expanded to a constant cell on the common grid and takes its own region eig, deduped by `(shape, bytes)` like a patterned cell.  Tensor layers contribute their DIAGONALS' real parts to the Wood-anomaly `_grazing_safe_wavelength` list (the `_pmm_jones_2d_at` rule); SCALAR layers are deliberately left out, exactly as before, so no shipped scalar result moves. |
| `_warn_stag_closure` / `_stack_is_lossless` / `_tensor_is_hermitian` / `_STAG_CLOSURE_TOL` | same | NEW lossless-closure tripwire (Section 4). |
| exports | `pmm/__init__.py`, `lumenairy/__init__.py` | `pmm_jones_2d_staggered` added to both `__all__`s and imported at the top level (required by the v4.16.0 `__all__`-symmetry walker). |
| `wmap=` mirror | `validation/m5_derham_nonuniform.py` | the non-uniform-partition subclass overrides `_eps_weighted` / `_eps_dir`; both gained the same optional `wmap` so the shipped validation script keeps working (and, as a free consequence, would take a tensor cell). |

**New public API**

```python
pmm_jones_2d_staggered(period_x, period_y, eps_cell, n_substrate,
                       n_superstrate, depth, wavelength, *, degree=8,
                       n_modes=None, n_orders=7, theta=0.0, phi=0.0)
    -> (orders (Nfo, 2), R (2, Nfo), T (2, Nfo), jones (2, 2))
```

`eps_cell` is `(Nx, Ny, 3, 3)` block-form or `(Nx, Ny)` scalar (promoted to
`e * I`).  Row 0 = incident `E_x`, row 1 = incident `E_y`; `jones` is the
order-0 REFLECTION Jones in the PUBLIC `exp(-i w t)` gauge with NO
conjugation (unlike `pmm_jones_2d`, which solves in an internal conjugated
gauge and conjugates back at extraction).  It is implemented as the
single-layer case of `PMM2DStackPure` -- one implementation of the physics,
so the two entries cannot drift; the cell is validated FIRST, in the entry, so
the message names the entry the user called.

### 1.2 Equations to code

Granet, J. Opt. Soc. Am. A **40**, 652 (2023).  The paper uses
`exp(+i w t)`; this module is PUBLIC `exp(-i w t)` end to end.  The operators
of Eqs. 19-25 carry no explicit `i` (Eq. 18 absorbs it into `Etilde_3`), so
the convention bridge is exactly "use the public `eps`" -- and every tensor
QUOTED from the paper must be conjugated before being passed in.

| paper | code (`twod_staggered.Granet2DTransverseE._assemble`) |
|---|---|
| Eq. 24 `R = C[chi_t]C = -I` | `Rmat = -blockdiag(G1, G2)`, unchanged (nonmagnetic) |
| Eq. 24 `L = k^2[eps_t] + S_tt - K_tz (eps33)^-1 K_zt` | `Lmat`, unchanged in shape (`2q^2 x 2q^2`) |
| Eq. 40 `eps11` (V1 test, V1 trial) | `Et_11 = _eps_weighted((bx, m, B, B), (by, m, Btil, Btil), e11)` |
| Eq. 40 `eps22` (V2, V2) | `Et_22 = _eps_weighted((bx, m, Btil, Btil), (by, m, B, B), e22)` |
| Eq. 40 `eps12` = `<V1| e12 |V2>` **(NEW)** | `Et_12 = _eps_weighted((bx, m, B, Btil), (by, m, Btil, B), e12)` -- a kron of the two UNLIKE-set 1-D masses `<B|Btil>_x` and `<Btil|B>_y` |
| Eq. 40 `eps21` = `<V2| e21 |V1>` **(NEW)** | `Et_21 = _eps_weighted((bx, m, Btil, B), (by, m, B, Btil), e21)` |
| Eq. 41 `eps33` | `Meps33 = _eps_weighted((bx, m, Btil, Btil), (by, m, Btil, Btil), e33)` -- weighted by `e33`, not by the scalar eps |
| Eq. 42 `S_tt` | unchanged (`chi33 = 1`): pure geometry, the `Curl^dag Gw^-1 Curl` mimetic form |
| Eq. 43 `K_tz` | unchanged (`chi_t = I`): the mimetic gradient `Grad1/Grad2` |
| Eq. 44 `K_zt` col 1 = `<V3| d1(e11 .) |V1> + <V3| d2(e21 .) |V1>` | `Kzt_E1 = -_eps_dir(bx,"Btilde","dL","B", by,"Btilde","m","Btilde", e11)/k0` **minus** `_eps_dir(bx,"Btilde","m","B", by,"Btilde","dL","Btilde", e21)/k0` (the NEW second term) |
| Eq. 44 `K_zt` col 2 = `<V3| d2(e22 .) |V2> + <V3| d1(e12 .) |V2>` | `Kzt_E2 = -_eps_dir(bx,"Btilde","m","Btilde", by,"Btilde","dL","B", e22)/k0` **minus** `_eps_dir(bx,"Btilde","dL","Btilde", by,"Btilde","m","B", e12)/k0` (NEW) |
| Eq. 25 `gamma C [H1;H2] = (k^2[eps_t] + S_tt)[E1;E2]` | `_region_modes`: `Lhh` gains `Lhh[:qq, qq:] = Et_12`, `Lhh[qq:, :qq] = Et_21` |

`K_zt` is the divergence of `D_t = eps_t E_t`, which is why column 1 pairs the
`x`-derivative with `e11` AND the `y`-derivative with `e21`, and column 2
pairs `y` with `e22` and `x` with `e12`.  In every term the derivative sits on
the CONTINUOUS `V3` test function (the shipped `"dL"` device); the two new
terms simply put it on the OTHER axis.

Field spaces are unchanged: `V1 = B(x1) x Btil(x2)` (E1), `V2 = Btil(x1) x
B(x2)` (E2), `V3 = Btil(x1) x Btil(x2)` (E3), matching the paper's Eq. 34.

### 1.3 What in-plane anisotropy does NOT change

The eigenproblem stays SECOND ORDER at dimension `2q^2`, so the
`[W; -V] <-> -lam` symmetry, the square Redheffer cascade
(`_interface_smatrix` / `_propagation_smatrix` / `_redheffer_star`), the
once-only forward far-field Rayleigh projection, the `retain_internal`
partial cascades and the eps-free block-Gram flux of `layer_absorption` are
all untouched.  That is the whole reason Stage A is low risk, and G5's
no-floor measurement confirms the architecture survived.

---

## 2. Measurement tables

### T1 -- G1 reduction (scalar dispatch vs the tensor `e * I` arm)

Probe `validation/probe_pmm2d_staggered_aniso/p1_reduction.py`.  Cell
`[[6.25, 1.0], [1.0, 2.25]]`, `(2,2)` grid, `M = 6`, `alpha0x = 0.31`,
`alpha0y = -0.17`, `k0 = 2 pi / 0.62`.

| operator | scalar sha256[:16] | tensor `e*I` sha256[:16] | max abs diff |
|---|---|---|---|
| `Lmat` | `c486dfd28943a2e4` | `c486dfd28943a2e4` | 0.000e+00 |
| `Rmat` | `95330a5328a4f2aa` | `95330a5328a4f2aa` | 0.000e+00 |
| `Stt` | `8c7fafec3a854d69` | `8c7fafec3a854d69` | 0.000e+00 |
| `Schur` | `c6ab2136acbe070f` | `c6ab2136acbe070f` | 0.000e+00 |
| `Et_blocks[0]` | `3ace323a8193d06a` | `3ace323a8193d06a` | -- |
| `Et_blocks[1]` | `9e83bf162dacbe4f` | `9e83bf162dacbe4f` | -- |
| `Et_offdiag` | `None` (nothing retained) | `(0, 0)` exactly | -- |

Public entries on the same cell (period 0.8 um, depth 0.3 um, wl 0.633 um,
n_sub 1.5, degree 6, n_orders 3): `pmm_efficiency_2d_staggered(..., 'tm')`
versus `pmm_jones_2d_staggered(..., e*I)` row 0 -- **max |dR| = max |dT| =
0.000e+00** (bit identical, not merely close).  Closure of the tensor arm:
1.0000235704637488 / 1.0000235704636900 for the two polarizations.

The plan allowed a `~1e-14` relative bar here.  The measured result is
EXACT identity, so the test asserts identity -- the stronger, build-free
statement.  Mechanism: `wmap = e11` IS `eps_cell` elementwise, so
`_eps_weighted` performs the identical einsum; the two new `K_zt` terms
subtract exact `+0.0` matrices; the two `Lmat` off-diagonal blocks are
assigned exact zeros where `np.zeros` already put them.

### T2 -- G2 published oracle (Granet Table 2 / Fig. 4)

> **CORRECTED 2026-09-09 by the independent verification** --
> `docs/audits/VERIFY_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` section 4.
> **This section's VERDICT ("the absolute Table-2 values are NOT reproduced")
> is WRONG, and so is the conclusion drawn from the gyrotropic-sign arm.**
> Everything below is left as written -- the measurements are all
> reproducible -- but read the correction first:
>
> * Granet's second example is Li, *J. Opt. A* **5**, 345 (2003), Example 1
>   (his own text says the FMM values "correspond perfectly with those
>   reported by Li"), and Granet's restatement of it carries two
>   transcription errors this build followed: `1 - i5` is Li's refractive
>   **INDEX** `n^(-1) = 1 + i5` (so `eps_sub = -24 + 10i`), and Li's Table 1
>   lists the **REFLECTED** orders, not the transmitted ones (Li's Fig. 3
>   caption, and the tabulated order set is exactly the vacuum superstrate's
>   propagating set).  Under Li's reading this build reproduces all SIX
>   published efficiencies to **8.74e-05** at M=8 -- inside the plan's 5e-4
>   bar.  The efficiency-definition sweep below is moot: the oracle is a
>   reflected set into vacuum.
> * Li's values are already PUBLIC `exp(-i w t)`, so his tensors are used
>   **unconjugated**.  The "gyrotropic sign discriminator" table below
>   compares OUR `(1,1)`/`(-1,1)` against GRANET's labels, which are mirrored
>   on one axis relative to Li's -- his `(1,1) = 0.0268` is Li's `(1,-1)`.
>   The build's conjugation and that label mirror cancel, so the arm does not
>   establish the bridge it claims to.  The `e12`/`e21` placement IS right
>   (T7's swapped-block control, and now Li's two published rows both ways),
>   but by a different argument.
> * Open item 1 of section 5 ("G2 is not closed") is therefore CLOSED, with
>   nothing tuned: Li's fill fractions are the stated 0.5/0.5 and were never
>   varied.  The G2 tests were rewritten accordingly (commit `d87fee4`).


Geometry read from the paper: `d_x = 2.4 lam`, `d_y = 1.4 lam` (the text's
"d_y = 2.4, d_y = 1.4" is a typo), `w_x = 0.5 d_x`, `w_y = 0.5 d_y`,
`h = lam`, vacuum cover, substrate `eps = 1 - 5i` (paper) -> `1 + 5i` (ours),
pillar `eps_b` of Eq. 37 conjugated, host `eps_a = eps_b^*` conjugated, normal
incidence, `E` along `x`, transmitted orders.  Probes `p2`, `p3`, `p4`, `p5`,
`p6`.

**Paper values.**  Table 2 (SEM) converges at `M >= 4` to
(1,1) 0.0268, (-1,1) **0.0139**, (0,-1) 0.0620, (0,0) 0.2979.
Table 3 (FMM) at `M = 10`: 0.0269, **0.0137**, 0.0619, 0.2980.  (The plan's
quoted set mixes the two tables; both are recorded here.  The two published
methods themselves differ by 2e-4 on (-1,1).)

**This build**, `(2,2)` corner pillar, shipped Poynting-flux definition:

| M | (1,1) | (-1,1) | (0,-1) | (0,0) | sum R + sum T |
|---|---|---|---|---|---|
| 5 | 0.020963 | 0.016580 | 0.058210 | 0.529659 | 0.997550 |
| 7 | 0.021156 | 0.015940 | 0.058300 | 0.527969 | 0.999663 |
| 9 | 0.021137 | -- | -- | 0.527766 | 0.999700 |

Position invariance verified: the `(4,4)` CENTRED pillar at `M = 5` gives
0.52800 for (0,0) against 0.52797 for the `(2,2)` corner pillar at `M = 7`,
max per-order |dT| = **6.72e-05** over the whole propagating set.  (A literal
3x3 decomposition with walls at 0.25 / 0.75 is not expressible in this basis --
`Basis1D` cuts an axis into EQUAL segments -- so the centred arm uses `(4,4)`
uniform quarters with the pillar occupying the middle two, which is the same
geometry.)

**Definitions tried** (order-0 row shown; `kz_i = 1` at normal incidence):

| definition | (1,1) | (-1,1) | (0,-1) | (0,0) | max dev vs SEM |
|---|---|---|---|---|---|
| A `Re(kz_t/kz_i) (|tx|^2+|ty|^2+|tz|^2)` (shipped) | 0.02116 | 0.01594 | 0.05830 | 0.52797 | 2.30e-01 |
| B tangential only | 0.01937 | 0.01465 | 0.05300 | 0.52797 | 2.30e-01 |
| C `|kz_t|/kz_i` full | 0.02902 | 0.02186 | 0.07870 | 0.68271 | 3.85e-01 |
| D `Re(n_sub)` scaled | 0.03694 | 0.02784 | 0.10180 | 0.92198 | 6.24e-01 |
| E bare `|t|^2` | 0.01296 | 0.00977 | 0.03511 | 0.30234 | **2.69e-02** |

Definition A is the exact Poynting-flux ratio at the interface: for a plane
wave in an isotropic medium with complex `eps` and real `kx, ky`,
`S_z ~ Re(kz)(|Ex|^2+|Ey|^2+|Ez|^2)` (using `k.E = 0`), so this is not a
choice but a derivation; the alternatives are recorded only because the paper
does not state its own.

**Other readings swept** (probes p4/p5/p6), each under definitions A/B/E:
axis assignment `(2.4, 1.4)` vs `(1.4, 2.4)`; which tensor is the pillar;
tensor conjugated or not; `eps_sub = 1+5i` vs `1-5i`; host = `eps_a` vs
vacuum; pillar = vacuum; `1 - i5` read as an INDEX (`eps_sub = -24+10i`);
`eps_sub` = vacuum and 2.25.  BEST maximum deviation over the four quoted
orders across ALL of them: **2.65e-02** (definition E, `d = (1.4, 2.4)`).
Runner-up families sit at 7.2e-02 (vacuum host) and 2.09e-01 (definition A).

**VERDICT: the absolute Table-2 values are NOT reproduced, by ~2 to ~230
times the 5e-4 bar the plan set.**  Nothing was tuned.  Per the plan the
ambiguity is recorded and the physics claim rests on G3/G4/G5.  Two facts
bound what this can and cannot mean:

* the discrepancy is NOT in the new assembly.  On the IDENTICAL geometry the
  independent hybrid `pmm_jones_2d` (degree 9, n_orders 13) agrees with the
  staggered tensor solve to **1.82e-04** on the four quoted orders (n_orders 9:
  3.37e-03), and the staggered arm closes energy to 3.4e-04 at M=7 (the layer
  is Hermitian, so `R + T = 1` is exact).  Two independent discretizations
  cannot both be wrong in the same way and still close energy.
* what the paper DOES pin, and this build DOES reproduce, is the SIGN of the
  gyrotropic order asymmetry (below).

**Gyrotropic sign discriminator** (the observable no energy check can see;
`M = 6`, definition A):

| arm | T(1,1) | T(-1,1) | T(1,1) - T(-1,1) |
|---|---|---|---|
| paper tensor CONJUGATED into `exp(-iwt)` | 0.021328 | 0.016157 | **+5.171e-03** |
| paper tensor used RAW (wrong bridge) | 0.016157 | 0.021328 | **-5.171e-03** |
| control: isotropic pillar in isotropic host | 0.022521 | 0.022521 | -1.020e-15 |

The paper's sense is `T(1,1) > T(-1,1)` in BOTH published methods.  This
build reproduces that sense with the conjugation bridge and reverses it
without -- so the `exp(+iwt) -> exp(-iwt)` bridge and the `e12`/`e21` signs
are confirmed against a published result even though the magnitudes are not.

### T3 -- G3 uniform in-plane tensor slab vs Berreman 4x4

Probe `p7`, re-measured through the test helpers in `p14`.  Period 0.40 um,
depth 0.55 um, wl 1.0 um, `n_sup = 1`, `n_sub = 1.5`, `n_orders = 2`.
Residual = max over (R, T summed per polarization, and the complex `2x2`
Jones) against `berreman_jones_1d`.

| tensor | grid | theta / phi (deg) | M = 5 | M = 7 |
|---|---|---|---|---|
| LC `uniaxial_tensor(1.5, 1.8, pi/2, 0.55)` | (2,2) | 0 / 0 | 2.842e-14 | 9.326e-14 |
| LC | (3,3) | 0 / 0 | 5.158e-15 | 1.431e-14 |
| LC | (2,2) | 25 / 0 | 7.223e-11 | 4.077e-14 |
| LC | (3,3) | 25 / 0 | 2.831e-12 | 4.718e-14 |
| LC | (2,2) | 25 / 40 | 5.454e-12 | 5.240e-14 |
| LC | (3,3) | 25 / 40 | 2.255e-13 | 7.105e-14 |
| gyrotropic `e12 = -e21 = 0.5i` | (2,2) | 0 / 0 | 1.910e-14 | 6.539e-14 |
| gyro | (3,3) | 0 / 0 | 6.911e-15 | 3.153e-14 |
| gyro | (2,2) | 25 / 0 | **9.711e-11** | **1.665e-14** |
| gyro | (3,3) | 25 / 0 | 3.796e-12 | 3.307e-14 |
| gyro | (2,2) | 25 / 40 | 7.953e-12 | 5.884e-14 |
| gyro | (3,3) | 25 / 40 | 3.187e-13 | 6.173e-14 |

Worst `M = 7` residual **9.326e-14** -> bar `1e-11` (107x).  The two-sided
ladder claim is made on the (2,2)/gyro/25-deg row, which drops by 5.8e3;
the normal-incidence rows are already at the roundoff plateau at `M = 5`, so
the ladder is not resolvable there and no claim is made about them.

This gate exercises everything the tensor path added: the LC tensor has
`e12 = e21 != 0` REAL (the symmetric mixed masses), the gyrotropic one has
`e12 = -e21` IMAGINARY (the antisymmetric case, which is where a swapped
placement or a conjugation slip shows), both have `e33 != e11`, and conical
incidence exercises the Bloch phases on both axes.  The Jones agrees WITHOUT
any conjugation, confirming the PUBLIC-gauge claim end to end.

### T3b -- G3 companion: each NEW term is LOAD-BEARING (fail-before)

Probe `p18_fail_before.py`.  Same G3 residual (R, T and the complex Jones vs
`berreman_jones_1d`), `(2,2)` grid, `M = 7`, conical `theta = 25 deg`,
`phi = 40 deg`, with ONE new contribution zeroed at a time (class/module-level
monkeypatch, no source edit).

| arm | LC slab | gyrotropic slab |
|---|---|---|
| ALL TERMS PRESENT (reference) | 5.240e-14 | 5.884e-14 |
| Eq. 40 MIXED masses -> 0 | **4.101e-02** | **7.749e-02** |
| Eq. 44 SECOND `K_zt` term -> 0 | **5.697e-03** | **4.058e-02** |
| Eq. 25 `Lhh` mixed blocks -> 0 | **1.049e-01** | **1.613e-01** |

Every new term moves the tightest oracle in this build by 11-12 decades, so
none of them is dead code and G3's 1e-14 is not an accident of a
partially-correct operator.  This is the "right conclusion, wrong mechanism"
guard the testing standards call the most dangerous shape; it is a test
(`test_g3_each_new_tensor_term_is_load_bearing`), bar 1e-4 -- 1.7 decades
under the smallest measured break and 10 decades over the reference.

### T4 -- G4 1-D reduction (y-uniform anisotropic stripe)

Probe `p8` / `p14`.  Period 0.90 um, depth 0.30 um, wl 0.55 um, duty 0.5,
ridge `uniaxial_tensor(1.5, 1.8, pi/2, 0.55)`, groove `2.10 I`, `n_sub = 1.5`,
`theta = 0.22` rad, `phi = 0`, `n_orders = 3`.  Oracle `pmm_jones_1d`
(degree 16, `stabilize=False`); cross-oracle `rcwa_jones_1d` (81 orders).

| M | per-order max\|dR\|,\|dT\| | max\|dJones\| | max efficiency in y-forbidden orders | closure |
|---|---|---|---|---|
| 5 | 2.180e-03 | 1.025e-03 | 6.73e-29 | 9.73e-04 |
| 6 | 2.411e-04 | 1.196e-04 | 4.78e-28 | 8.15e-05 |
| 7 | 2.598e-05 | 2.874e-05 | 1.59e-27 | 6.61e-06 |
| 8 | **4.306e-06** | **9.088e-06** | 4.86e-27 | 6.39e-07 |

At normal incidence the same ladder reads 1.605e-04 / 1.321e-05 / 7.885e-06 /
2.382e-06 on R/T.  The two 1-D oracles agree with EACH OTHER to 5.22e-07
(R/T) and 1.51e-06 (Jones) at `theta = 0.22`, i.e. 25x / 18x under the bars
`3 x` the `M = 8` residual (1.3e-05 and 2.7e-05).

### T5 -- G5 three engines + the no-floor property

Probe `p9`.  Period 0.70 um, depth 0.28 um, wl 0.55 um, `n_sub = 1.5`,
`(2,2)` cell, normal incidence.  Order-0 `R` and `T`, both polarizations.

**(a) isotropic pillar (`eps = 4`) in a rotated-uniaxial LC host** (the cell
the test uses):

| engine | R00 (Ex, Ey) | T00 (Ex, Ey) | max dev vs staggered (R / T / Jones) | closure |
|---|---|---|---|---|
| staggered `M = 7`, `n_orders = 5` | 0.049890, 0.044401 | 0.832945, 0.731816 | -- | **2.499e-08** |
| hybrid `degree 9`, `n_orders 9` | 0.049952, 0.044710 | 0.832071, 0.730020 | 3.10e-04 / 1.80e-03 / 7.98e-04 | 2.95e-04 |
| hybrid `degree 11`, `n_orders 13` | 0.049943, 0.044519 | 0.832915, 0.731724 | 1.18e-04 / 9.26e-05 / 3.07e-04 | 2.42e-04 |
| `rcwa_jones_2d`, `n_orders 9` | 0.049947, 0.044552 | 0.832595, 0.731052 | 1.51e-04 / 7.65e-04 / 3.91e-04 | 1.91e-14 |
| `rcwa_jones_2d`, `n_orders 13` | 0.049935, 0.044514 | 0.832666, 0.731234 | 1.13e-04 / 5.82e-04 / 2.98e-04 | 7.88e-14 |

**(b) LC pillar in an isotropic host** (the reverse):

| engine | R00 | T00 | max dev vs staggered | closure |
|---|---|---|---|---|
| staggered `M = 7` | 0.039140, 0.036233 | 0.849607, 0.755336 | -- | 1.85e-07 |
| hybrid `d 9 / no 9` | 0.038920, 0.035963 | 0.847531, 0.752407 | 2.70e-04 / 2.93e-03 / 2.20e-03 | 2.62e-04 |
| hybrid `d 11 / no 13` | 0.039088, 0.036139 | 0.849747, 0.755887 | 9.40e-05 / 5.51e-04 / 4.34e-04 | 7.01e-05 |
| rcwa `no 9` | 0.039094, 0.036041 | 0.850630, 0.757704 | 1.92e-04 / 2.37e-03 / 5.85e-04 | 6.22e-15 |
| rcwa `no 13` | 0.039117, 0.036101 | 0.850360, 0.757020 | 1.32e-04 / 1.68e-03 / 4.66e-04 | 1.13e-14 |

Largest pairwise spread on the tested cell (a): **5.82e-04** -> bar 5e-3
(8.6x).  The residual is the FOURIER engines' floor, not the staggered arm's:
the staggered closes energy to 2.5e-08 while the hybrid closes to 2.4e-04,
and both Fourier arms still move between `n_orders` 9 and 13.

**No-floor, `n_orders` 4 -> 8 at fixed resolution** (order-0 R and T):

| engine | max change |
|---|---|
| staggered (`M = 7`) | **3.886e-15** |
| hybrid (`degree 9`) | **9.687e-03** |

A ratio of 2.5e12.  Bars: staggered `< 1e-10`, hybrid `> 1e-4` -- both sides
asserted, because "no floor" is a claim only if the floored engine is shown
to be floored.

### T6 -- G6 closure, two-sided

Probe `p10`.  Same geometry as T5, `n_orders = 4`.

| cell | M | sum R + sum T (Ex, Ey) | \|dev\| |
|---|---|---|---|
| LC host + iso pillar | 6 | 1.000000991730 / 1.000002111560 | 2.112e-06 |
| LC host + iso pillar | 8 | 1.000000003185 / 1.000000006267 | **6.267e-09** |
| gyrotropic host + iso pillar | 6 | 1.000011709532 / 1.000011709532 | 1.171e-05 |
| gyrotropic host + iso pillar | 8 | 0.999999960725 / 0.999999960726 | **3.927e-08** |
| LC host + LOSSY pillar (`eps = 4 + 0.8i`) | 6 | 0.772979615198 / 0.755134844513 | deficit 0.2270 / 0.2449 |
| LC host + LOSSY pillar | 8 | 0.773009617034 / 0.755512131348 | deficit 0.2270 / 0.2445 |

The GYROTROPIC cell is Hermitian and therefore lossless -- it closes to
3.9e-08 -- which is exactly the case a naive "lossless == real eps" predicate
would mis-classify.  The `M = 6 -> M = 8` drop (298x) makes it a convergence
statement.  Bar for the Hermitian arm `1e-5` = 1e2 x the worse measurement.

### T7 -- G7 discrete symmetries (the mixed-block placement control)

Probes `p10` / `p14` / `p17`, at the test's `degree = 6`, periods
`(0.70, 0.91) um`, `n_orders = 3`.  Residual = max per-order |dR|,|dT| over
the mapped order set; Jones residual reported separately.

| claim | per-order | Jones |
|---|---|---|
| x<->y transpose (`e11<->e22`, `e12<->e21`, periods swapped), LC cell | **2.065e-14** | 1.072e-14 |
| x<->y transpose, GYROTROPIC cell | **1.300e-14** | 3.047e-14 |
| y-mirror (`phi -> -phi`, `e12 -> -e12`), LC cell | **2.845e-15** | 1.161e-14 |
| CONTROL: `e12`/`e21` interchanged in one arm, LC cell | 2.554e-14 | 1.619e-14 |
| CONTROL: `e12`/`e21` interchanged in one arm, GYROTROPIC cell | **3.635e-03** | **8.137e-02** |

The LC tensor has `e12 = e21`, so the swap is a NO-OP there -- which is
precisely why the gyrotropic tensor (`e12 = -e21`) is the discriminator, and
why a control built only on the LC cell would have been vacuous.  Correct
placement and swapped placement are 11 decades apart.  Bars: correct
`< 1e-11`, swapped `> 1e-7`.

### T8 -- G8 stack

Probes `p10` / `p15` / `p14`.

**(a) split vs single** (`M = 6`, `theta = 0.15`, `phi = 0.4`, LC host + iso
pillar): one layer of thickness `2 t` versus two cascaded layers of thickness
`t`:

| quantity | max abs difference |
|---|---|
| R (all orders, both pols) | 2.359e-16 |
| T | 3.331e-16 |
| Jones | 8.738e-16 |

(at `M = 7` the same numbers are 1.249e-16 / 6.661e-16 / 3.647e-16.)

**(b) all-uniform anisotropic multilayer vs Berreman** (LC 0.21 um /
gyrotropic 0.13 um / mirrored-LC 0.17 um, period 0.4 um, wl 1.0 um,
`theta = 25 deg`, `phi = 40 deg`, `n_sub = 1.5`, `n_orders = 2`):

| M | residual (R / T / Jones combined) |
|---|---|
| 5 | 2.404e-12 |
| 7 | **4.208e-14** |

Spectral, dropping by 57x.  Bar `1e-11` (238x) plus the 10x-drop claim.  This
is the gate that exercises the "a uniform TENSOR region takes its own region
eig" path (three distinct tensor layers -> three region eigs, deduped by
bytes) together with the mixed uniform/patterned cascade.

### T9 -- G9 absorption budget for a lossy TENSOR stack

Probe `p15` / `p16`.  Three layers on the common `(2,2)` grid: LC+iso pillar
0.12 um (lossless, patterned tensor), LC + `eps = 4 + 0.8i` pillar 0.28 um
(LOSSY, patterned tensor), uniform gyrotropic 0.09 um (lossless, uniform
tensor).  `theta = 0.12`, `phi = 0.3`, `n_orders = 3`,
`solve(retain_internal=True)`.

| M | \|sum A - (1 - sum R - sum T)\| (Ex / Ey) | A(lossy layer) | max \|A(lossless layers)\| | wall |
|---|---|---|---|---|
| 6 | 2.155e-05 / 1.290e-06 | 0.254416 / 0.291326 | 5.80e-15 | 1.41 s |
| 7 | 3.982e-07 / 1.067e-06 | 0.254468 / 0.291394 | 6.22e-15 | 1.89 s |
| 8 | **8.205e-08 / 1.670e-08** | 0.254447 / 0.291294 | 5.59e-14 | 4.28 s |

The eps-free block-Gram flux form needed no change for tensor modes, as the
plan hoped -- but it was VERIFIED, not assumed.  Bar `1e-5` at `M = 8`
(1e2 x the measurement) plus the 263x `M = 6 -> M = 8` drop.

### T10 -- G10 guards

| input | result |
|---|---|
| `e_xz = e_zx = 0.4` in one cell | `NotImplementedError`, message contains "OUT-OF-PLANE" and names `pmm_jones_2d` |
| same, through `PMM2DStackPure.add_layer(eps_cell=)` | `NotImplementedError` naming `pmm_jones_2d` |
| `add_layer(eps=uniaxial_tensor(1.5, 1.8, 0.6))` (tilted director, genuinely out of plane) | `NotImplementedError` naming `pmm_jones_2d` |
| `uniaxial_tensor(1.5, 1.8, pi/2, 0.55)` -- max off-plane entry **5.168e-17**, tensor scale 2.970, floor `1e-12 * 2.970 = 2.97e-12` | ACCEPTED and solved (a strict `> 0` test would have refused every rotated LC cell) |
| injected `1e-16` xz stray | ACCEPTED |
| injected `1e-3` xz stray | `NotImplementedError` |
| `e33 = 0` in one cell | `ValueError` "e_zz must be nonzero" |
| `(2, 2, 2, 2)` cell | `ValueError` "must be (Nx, Ny, 3, 3)" |
| non-square `(1, 2, 3, 3)` cell | `ValueError` "eps_cell must be SQUARE" |
| `add_layer(eps=np.ones((2,2)))` | `ValueError` "must be a scalar or a (3, 3) block-form tensor" |
| 4-D cell to `pmm_efficiency_2d_staggered` | `ValueError` naming BOTH `pmm_efficiency_2d_staggered` and `pmm_jones_2d_staggered` |
| `_homog_geom_cache` on a tensor assembly | `ValueError` "uniform SCALAR" |

The out-of-plane floor is REUSED from `twod_jones._tile_is_offplane`, not
copied, so the two engines' definition of "in-plane" cannot drift apart.

### T11 -- the lossless-closure tripwire

Probe `p11` (derivation), `p13`/`p13b`/`p13c` (fail-before search).

**Must-not-fire side.**  40 lossless SCALAR configurations of this engine --
every fixture the shipped staggered suites exercise plus a deliberately
adverse set -- measured `|sum R + sum T - 1|`:

| worst configurations | value |
|---|---|
| pure stack, patterned pillar + uniform layer, `n_sub 1.5`, `theta 0.25`, `M = 6` | **9.009e-03** |
| `pmm_efficiency_2d_staggered` pillar, `degree 4` (any `n_orders`) | 1.627e-03 |
| pure stack, same, `n_sub 1.0`, `theta 0.25`, `M = 6` | 1.294e-03 |
| pure stack, `n_sub 1.5`, `theta 0` | 2.964e-04 |
| pillar `degree 5` | 1.419e-04 |
| everything else | <= 2.6e-06 |

The 9.0e-03 entry is a UNIFORM layer at `theta = 0.25` and `M = 6` -- exactly
the degree-limited oblique-uniform regime `stack2d_pure`'s docstring already
warns about, and the tripwire's advice ("raise `n_modes`") is the right advice
there.  `_STAG_CLOSURE_TOL = 5e-2` therefore sits 5.5x above the worst
adverse configuration and ~31x above every shipped-suite one.  It is the same
value as the hybrid's `twod._PASSIVE_TOL_2D`, arrived at independently.

**Must-fire side.**  The staggered basis degrades GRACEFULLY -- there is no
blow-up to exploit -- so the search for an engineered lossless violation had
to be systematic:

| `eps_pillar` (gyrotropic host) | M = 3 | M = 4 | M = 5 | M = 8 |
|---|---|---|---|---|
| 4.0 | 4.775e-02 | 2.462e-04 | 1.159e-04 | 3.928e-08 |
| 16.0 | 8.261e-02 | 2.934e-04 | 3.540e-04 | 1.074e-07 |
| 36.0 | 7.682e-02 | 4.432e-03 | 3.314e-04 | 2.750e-07 |
| 100.0 | **9.084e-02** | 2.532e-04 | 3.882e-04 | 1.508e-07 |

Also swept: periods 0.7 / 1.4 / 2.1 um and depths 0.28 / 0.90 um (max
4.7e-02), and the near-Rayleigh-cutoff route at `1 - 1e-4 .. 1e-10` of the
substrate cutoff (max 8.06e-03 -- the wavelength nudge in
`_grazing_safe_wavelength` caps it).  The strongest available violation is
**9.08e-02 against the 5e-02 window, a factor 1.82**.  That ratio is modest,
but the quantity is a deterministic discretization error whose cross-build
spread is ~1e-06, so the decision sits ~4 decades clear of build noise; and
the test asserts the SAME cell at `M = 8` is silent (1.5e-07), which is the
two-sided half.

**Non-Hermitian silence.**  A lossy tensor cell with a 0.227 deficit emits
nothing -- the guard keys on provable losslessness, not on a closure
deviation, which is what makes it non-tautological.

### T12 -- cost and memory

Probe `p12`.  Grid `(3,3)`, `M = 8` (`q = 21`, eigenproblem dimension 882),
period 0.7 um, depth 0.28 um, wl 0.55 um, `n_orders = 4`.  `tracemalloc` peak
over the whole call.

| arm | wall | tracemalloc peak |
|---|---|---|
| SCALAR `pmm_efficiency_2d_staggered` | 17.93 s | 401.1 MiB |
| TENSOR `pmm_jones_2d_staggered` (drives BOTH polarizations) | 17.04 s | **403.8 MiB** (+0.7 %) |
| region eig only, SCALAR | 6.97 s | 83.3 MiB |
| region eig only, TENSOR | 7.30 s | 83.3 MiB |

Retained operator inventory per solver instance:

| | scalar | tensor |
|---|---|---|
| `Lmat`, `Rmat`, `Stt`, `Schur` (each `2q^2` square) | 4 x 11.9 MiB | 4 x 11.9 MiB |
| `Et_blocks[0..1]` (each `q^2` square) | 2 x 3.0 MiB | 2 x 3.0 MiB |
| `Et_offdiag[0..1]` | -- | 2 x 3.0 MiB |
| total | 53.4 MiB | **59.4 MiB (x1.111)** |

`Curl`, `Kzt`, `Ktz`, `G3` and `Meps33` are asserted absent on BOTH arms
(audit P3-37 holds).  The two extra retained blocks are consumed by
`_region_modes` for the Eq. 25 `Lhh`; nothing else is added, and the eig
dimension is unchanged, which is why the tensor solve is not slower.

---

## 3. Tests run

New file: `tests/unit/test_pmm2d_staggered_anisotropic.py` -- 37 tests
(G1 x2, G2 x2, G3 x8 [6 parametrized + 2] + 3 fail-before, G4 x3, G5 x2,
G6 x3 [2 parametrized + 1], G7 x3, G8 x2, G9 x1, G10 x5, tripwire x3).

Measured durations of the slowest tests in the new file (`--durations`,
single-threaded, final run): 10.30 s `g5_three_engines`, 6.56 s
`g3_multisegment`, 4.54 s `g9_absorption` (its M ladder), 3.29 s
`g6_closure_improves`, 1.78 s `g5_no_fourier_floor`, 1.63 s `g4_ladder`;
**whole file 53.8 s for 37 tests**, no single test above 11 s -- inside the
plan's `< 3 min` file / `< 40 s` test budget.  Every grid is `(2,2)` or
`(3,3)` and every `M <= 8`.

Regression suites run in the worktree (`-p no:randomly`, OMP capped, this
build):

| set | result |
|---|---|
| `test_pmm2d_staggered_anisotropic.py` (this file) | **37 passed**, 53.8 s |
| `test_v5_12_0_pmm2d_staggered` + `test_v5_21_pmm2d_staggered_oblique` + `test_staggered` + `test_audit_p1_staggered_guard` + `test_p2c_pmm2d_stack_cascade` + `test_p2t_pmm2d_tree_cascade` + `test_pmm2d_lossless_closure_two_sided` + `test_audit_s1_3_pmm2d_lossless_tripwire` + `test_v5_14_0_pmm2d_stack` + `test_v5_12_0_pmm2d_loss` + `test_v5_14_0_pmm2d_cell` + `test_audit_w3_entry_validation` + `test_audit_w6_pmm_rcwa` + `test_v5_14_0_pmm_jones_2d` + `test_public_api` + `test_v4_16_0_walker_all_symmetry` (with this file) | **971 passed**, 575.8 s |
| `test_audit_dynameta_consumer_api_2` + `test_niche_audit_w7_pmm` (the PMM2DStackPure consumers; `~5 min` per pure C3 test) + this file | **183 passed**, 902.9 s |
| every `pmm` / `staggered` / `berreman` / `rcwa_jones` / `public_api` / `walker` test in `tests/unit/` (`-k`, 2597 selected) | **2589 passed, 8 skipped, 1 failed**, 1410.5 s -- the single failure is PRE-EXISTING (below) |

The one failure is
`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_stripe_reduces_to_rigorous_1d`.
It is **PRE-EXISTING and unrelated** -- VERIFIED, not assumed: the library
directory was reverted to the pre-change commit `9f212dd` and the same test
failed identically there (`1 failed, 9 passed` in that file), then restored.
The failure is inside that test's own `_sound_1d_reference` ladder, entirely
within `rcwa_jones_1d_segments`, whose docstring already records the fixture as
"a poisoned truncation for this fixture on at least one shipped platform"; the
PMM staggered path is not on its call graph.

The NEW closure tripwire fired ZERO times across all of them (grep for
`PMM2DStackPure.solve: lossless energy closure violated`), which is the
empirical form of the "must not fire on any existing scalar fixture"
requirement -- T11 is its derivation, this is its confirmation.

---

## 4. The lossless-closure tripwire (design note)

`PMM2DStackPure.solve` now ends with `_warn_stag_closure`.  It fires ONLY
when the stack is PROVABLY lossless:

* both half-space permittivities have exactly zero imaginary part (they come
  from real indices, so the exact test is right), and
* every layer is either an exactly-real scalar / scalar map, or a HERMITIAN
  tensor.

Hermiticity uses a RELATIVE floor (`1e-12 * tensor scale`) for the same
reason `_tile_is_offplane` does: a lossless tensor built as `R @ diag @ R.T`
computes its `(0,1)` and `(1,0)` entries by different dot products, so exact
`t == t^H` is not float-attainable, while a genuine absorbing part is
`O(kappa * n)`.  Gyrotropic media are therefore correctly classed lossless --
the case a naive "real eps" predicate gets wrong.

The gate is TWO-SIDED about 1.0 (a deficit is as much a defect as an excess),
following the 2026-08-17 correction to the hybrid's sibling: losslessness has
already been ESTABLISHED, so `R + T = 1` is exact.  A non-Hermitian tensor
gets no unity claim at all.  It WARNS, never raises.

---

## 5. Open / not done

1. **G2 is not closed.**  *(CORRECTED 2026-09-09: it IS closed -- see
   the correction banner on T2 and
   `docs/audits/VERIFY_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md`
   section 4.  The reading was Li 2003 example 1, whose substrate is a
   refractive INDEX and whose table lists the REFLECTED orders; all six
   published values reproduce to 8.74e-05 at M=8, nothing tuned.  The
   speculation below about fill fractions and normalization is void.)*  The published Table-2 magnitudes are not reproduced
   under any of the readings swept (T2).  The most likely remaining
   explanations, none of which this build can settle from the paper alone:
   the paper's Fig. 4 fill fractions may not be the 0.5/0.5 the text states
   (a continuous parameter, so fitting it would be tuning and was refused);
   or the paper's transmitted-efficiency normalization into a lossy substrate
   is one this build did not guess.  The SEM's `dim = 9 M^2` implies a 3x3
   subdomain decomposition, i.e. a CENTRED pillar; that is NOT the cause --
   the centred arrangement was measured and agrees with the corner one to
   6.7e-05 (position invariance, T2).  Deliberately NOT
   tuned.  What IS established against the paper is the gyrotropic order-
   asymmetry SIGN, and the cross-engine agreement in T5 carries the rest.
2. **Out-of-plane (Stage B) is not built** -- by design.  Entries raise
   `NotImplementedError` naming `pmm_jones_2d`.
3. **Anisotropic HALF-SPACES are out of scope** (the Rayleigh match is
   scalar), as in the hybrid.  Not implemented, documented.
4. **Magnetic anisotropy** (`chi_t != I`) is still unimplemented; the paper's
   `R = C[chi_t]C` makes it a one-line follow-on for `Rmat` plus the `chi`
   weights in `S_tt` / `K_tz`, but nothing here touched it.
5. **No JAX twin** for the staggered path (it is NumPy-only today, tensor or
   not).
6. **The scalar path's Wood-anomaly eps list still omits the layer eps.**
   That is PRE-EXISTING behaviour of `pmm_efficiency_2d_staggered` and of
   `PMM2DStackPure`, deliberately left alone here (changing it would move
   every shipped scalar result); tensor layers DO contribute their diagonals,
   which makes the rule inconsistent between the scalar and tensor paths.
   Worth unifying in a separate change, behind its own measurement.
   *CLOSED 2026-09-10* -- unified on branch `fix/wood-list-fffnv`
   (`docs/audits/FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md`, Task G).  Both paths
   now list every region's real permittivity (half-spaces, every distinct
   scalar cell value, every uniform scalar layer, every tensor diagonal).  The
   fear that it would "move every shipped scalar result" was measured and is
   unfounded: the guard's trigger band is `|eps - kt^2| <= 1e-9` (a relative
   wavelength window of ~1.2e-10), so 11 scalar and tensor fixtures came out
   BIT-IDENTICAL to `fb3fd93`, nudged wavelengths included, and only the exact
   coincidence moved (4.591e-08 / 7.561e-09 / 4.977e-08 -> exact 0).
7. **The tripwire's fail-before margin is 1.82x**, not decades (T11).  The
   staggered basis simply does not blow up when under-resolved.  If a future
   improvement pushes the `M = 3` closure below 5e-2, the fail-before test
   goes red and must be re-derived (the durability rule), not widened.
8. **`_STAG_CLOSURE_TOL` was not applied to `pmm_efficiency_2d_staggered`.**
   The scalar single-layer entry keeps no tripwire, so its behaviour is
   byte-identical to before.  Only the stack (and hence the new Jones entry)
   is guarded.
9. **The non-uniform-partition validation subclass** (`validation/
   m5_derham_nonuniform.py`) now mirrors the `wmap` keyword and would
   therefore accept a tensor cell, but that combination was never exercised
   and is not claimed to work.
10. **Not run in this build**: the full library suite, the JAX legs, and any
    cross-platform (WSL / different LAPACK) confirmation of the tables above.
    Every number here is one machine's measurement; the bars are derived so
    that they do not depend on it, but the TABLES are single-build readings.
11. **One PRE-EXISTING red in the regression slice**,
    `test_v5_20_12_rcwa_jones_2d_fff_nv::test_fff_nv_stripe_reduces_to_rigorous_1d`
    -- reproduced on the pre-change library at `9f212dd` (Section 3), so it is
    inherited, not caused here.  Not investigated further: it is an
    `rcwa_jones_1d_segments` truncation-ladder problem, out of this build's
    scope.
    *CLOSED 2026-09-10* on branch `fix/wood-list-fffnv`
    (`docs/audits/FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md`, Task H).  It was
    indeed the truncation ladder, but the ladder was only the symptom: the
    FIXTURE sits on an exact index coincidence (`no^2 = eps_groove = n_sub^2 =
    2.25`), so the layer carries modes EXACTLY degenerate with the region's and
    the interface inverse amplifies the rounding floor by ~1e14.  The ladder
    then hunts for a truncation where that floor happens to land below 1e-9 --
    0 of 16 on this box at 1 BLAS thread, 1 of 16 at 4 threads (the same code
    on the same box PASSES at 4 and FAILS at 1), 0 of 16 on WSL.  No library
    defect: the solver detects and warns, and detuning any one of the three
    coincident permittivities restores machine-precision closure at every
    truncation.  Fixed in the test (non-degenerate groove 2.10, converged
    reference, and a new two-sided test for the degeneracy itself); green on
    Windows at 1 and 4 threads and on WSL.  The PMM sibling
    `test_pmm_fff_nv_stripe_reduces_to_rigorous_1d` shares the fixture and is
    latent (passing today) -- see that report's open items.

---

## 6. Reproduction

Probes live in `validation/probe_pmm2d_staggered_aniso/` (see its `README.md`).
Run any of them as:

```
cd /c/tmp/lum_aniso && PYTHONPATH=/c/tmp/lum_aniso OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_pmm2d_staggered_aniso/<probe>.py
```

Tests:

```
cd /c/tmp/lum_aniso && PYTHONPATH=/c/tmp/lum_aniso OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/unit/test_pmm2d_staggered_anisotropic.py -q
```
