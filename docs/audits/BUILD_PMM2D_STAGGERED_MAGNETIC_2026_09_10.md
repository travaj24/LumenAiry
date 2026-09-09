# BUILD -- MAGNETIC (permeability-tensor) anisotropy for the PURE staggered 2-D PMM

Date: 2026-09-10.  Status: BUILT + gated in the worktree
`C:/tmp/lum_mag`, branch `feat/pmm2d-staggered-magnetic` (off `main` fb3fd93 =
5.43.0 + one backend commit).  Follows
`PLAN_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` Section 5, which named
magnetic anisotropy out of scope for Stage A and predicted "the paper's
`R = C[chi_t]C` makes it a one-line follow-on".  It is more than one line --
three operators gain weights and one shipped ALIASING of two roles has to be
undone -- but it is the same second-order pencil, the same basis, the same
cascade and the same far field.

Machine for every measurement below: Windows 11, py3.14, numpy 2.x,
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`.  Probes:
`validation/probe_pmm2d_staggered_magnetic/` (see its README for the
probe -> table mapping).  Tests: `tests/unit/test_pmm2d_staggered_magnetic.py`.

---

## 1. What was built

### 1.1 Library

| item | file | what changed |
|---|---|---|
| `Granet2DTransverseE(..., mu_cell=None)` | `lumenairy/elements/pmm/twod_staggered.py` | accepts a scalar `(Nx, Ny)` or BLOCK-FORM `(Nx, Ny, 3, 3)` relative permeability.  `None` (the default) takes NO magnetic branch anywhere -- the shipped arithmetic, unchanged (gate G1). |
| `Granet2DTransverseE._chi_maps()` | same | NEW: the per-cell `(chi11, chi12, chi21, chi22, chi33)` maps, `[chi_t] = [mu_t]^-1` as the POINTWISE 2x2 inverse (exact for the piecewise-constant cells this basis is built on -- inverting per cell and discretizing commute when the walls are element boundaries) and `chi33 = 1/m33`.  `None` when nonmagnetic. |
| `_assemble` -- `Rmat` | same | `R = C[chi_t]C` (Eq. 24 / A39): four blocks, the C-rotation swapping the transverse indices.  Nonmagnetic branch untouched. |
| `_assemble` -- `Stt` | same | the `chi33`-weighted curl-curl (Eq. 20 / A42): the mimetic middle operator `Gw^-1` becomes `Gw^-1 Gw_chi Gw^-1` with `Gw_chi = <Vw| chi33 |Vw>`.  With `chi33 = 1` the expression is the shipped matmul chain object-for-object. |
| `_assemble` -- `Ktz` | same | `K_tz = C[chi_t][d2; -d1]` (Eq. 21 / A43): row 1 `-chi22 d1 + chi21 d2`, row 2 `-chi11 d2 + chi12 d1`.  The eps-free `Grad1/Grad2` krons are not even built on the magnetic branch. |
| `Ggram_blocks` | same | NEW retained attribute: `(G1, G2)` -- the PLAIN block field Gram's two `q^2` diagonal blocks -- for a magnetic solver, `None` otherwise (so the nonmagnetic solver retains not one byte more than before).  See Section 2, THE TRAP. |
| `_region_modes` | same | the Eq.-25 H recovery solves BLOCKWISE against `Ggram_blocks` when present; the nonmagnetic path keeps its single `inv(-Rmat)`, bit for bit. |
| `_homog_geom_cache` | same | now RAISES on a magnetic assembly as well as on a tensor one: `L0_geom = Stt - Schur` is not geometric when `chi_t` and `chi33` weight `R`, `K_tz` and `S_tt`. |
| `_require_inplane_mu(fn, tile33)` | same | NEW guard: `m33 != 0`, `[mu_t]` invertible (relative `1e-14 * scale^2` floor on `det`), and BLOCK-FORM only -- out-of-plane raises at the SAME RELATIVE `1e-12 * scale` floor the permittivity uses (`twod_jones._tile_is_offplane`, REUSED not copied). |
| `_validate_stag_mu(fn, mu_cell)` | same | NEW: shape / SQUARE-grid / block-form validation, the mirror of `_validate_stag_cell`, so each entry names ITSELF in its message. |
| `_require_nonmagnetic_halfspace(fn, mu_sup, mu_sub)` | same | NEW: `mu_superstrate` / `mu_substrate` exist only to RAISE (a magnetic half-space changes the wave impedance, hence both the Rayleigh flux normalisation and the incident overlap). |
| `pmm_jones_2d_staggered(..., mu_cell=None, mu_superstrate=None, mu_substrate=None)` | same | the magnetic entry.  Validates, checks the mu grid against the eps grid, refuses mu with an out-of-plane eps, and hands the pair to the single-layer pure cascade. |
| `PMM2DStackPure(..., mu_superstrate=None, mu_substrate=None)` | `stack2d_pure.py` | the same half-space guard at the builder. |
| `PMM2DStackPure.add_layer(..., mu=None, mu_cell=None)` | same | a MAGNETIC layer: any combination of uniform / patterned on the eps and mu sides (uniform eps + patterned mu included).  Dispatches to the new `_add_magnetic_layer`, so the nonmagnetic branch above it is untouched. |
| `_as_layer_cell` / `_spec_is_lossless` | same | NEW helpers: broadcast a uniform spec onto the union grid (the `uniform` flag disambiguates a `(3,3)` UNIFORM tensor from a `(3,3)` scalar GRID), and the per-spec losslessness predicate. |
| `PMM2DStackPure.solve` | same | a magnetic layer takes its own region eig, deduped by `(eps shape+bytes, mu shape+bytes)`; `mu_cell=None` is passed through for every other kind, so those paths are bit-identical. |
| `_stack_is_lossless` | same | LOSSLESS now means Hermitian eps AND Hermitian mu.  A gyrotropic `m12 = -m21 = i b` is Hermitian, hence lossless, despite being complex. |

Not touched, deliberately: `[eps_t]` (Eq. 40), `Meps33` (Eq. 41) and `K_zt`
(Eq. 44) carry permittivity only; the Redheffer cascade, the far field, the
union grid, `retain_internal` / `layer_absorption`, and the out-of-plane
first-order generator (which has no permeability blocks -- hence the guard).

### 1.2 Equations to code

Granet, J. Opt. Soc. Am. A 40, 652 (2023).  `C = [[0, 1], [-1, 0]]` (Eq. 9),
`[chi_t] = [mu_t]^-1` (Eq. 11), `chi33 = 1/m33` (Eq. 12).  The paper uses
`exp(+i w t)`; this module is PUBLIC `exp(-i w t)` end to end -- the magnetic
operators carry no explicit `i`, so the bridge is exactly "use the public mu".

| paper | code (`Granet2DTransverseE._assemble`) |
|---|---|
| Eq. 24 `R = C[chi_t]C = [[-chi22, chi21], [chi12, -chi11]]` | `Rmat[:qq,:qq] = -ew(V1,V1, chi22)`, `Rmat[:qq,qq:] = +ew(V1,V2, chi21)`, `Rmat[qq:,:qq] = +ew(V2,V1, chi12)`, `Rmat[qq:,qq:] = -ew(V2,V2, chi11)`, with `ew` the eps-weighted kron helper and the mixed blocks kron'd from the UNLIKE-set 1-D masses exactly like the Eq. 40 `eps12`/`eps21` blocks.  Appendix-A Eq. 39's index placement (R11<->chi22, R12<->chi21, R21<->chi12, R22<->chi11) is the C-rotation's doing, and it is what the G6 transpose test pins. |
| Eq. 20 `S_tt = [d2; -d1] chi33 [d2, -d1]` (A42) | `Curl` is the WEAK matrix `<Vw \| curl E>`, so `Gw^-1 Curl` are the EXACT Vw coefficients of `curl E` (the de Rham property puts it there strongly).  The bilinear form `-<curl v, chi33 curl E>` is therefore `-Curl^H (Gw^-1 Gw_chi Gw^-1) Curl` with `Gw_chi = <Vw\| chi33 \|Vw>`; `chi33 = 1` gives `Gw_chi = Gw` and the middle collapses to the shipped `Gw^-1`. |
| Eq. 21 `K_tz = C[chi_t][d2; -d1]` (A43) | row 1 `-chi22 d1 + chi21 d2` tested in V1, row 2 `-chi11 d2 + chi12 d1` tested in V2, each assembled by `_eps_dir` with the derivative on the V3 TRIAL function.  `chi_t = I` reproduces the shipped `Grad1 = -kron(Mtt_y, dbt_x)`, `Grad2 = -kron(dbt_y, Mtt_x)`. |
| Eq. 25 `gamma C [H1;H2] = [k^2 [eps_t] + S_tt] [E1;E2]` | `_region_modes`: `Lhh` is unchanged in FORM (its `S_tt` is now the chi33-weighted one), but the projection back to coefficients uses the PLAIN block Gram -- Section 2. |
| Eq. 23 `-gamma^2 R E = L E` | unchanged: `sla.eig(Lmat, -Rmat)`, dimension `2 q^2`.  For a Hermitian positive-definite `[mu_t]`, `R = C[chi_t]C` is Hermitian NEGATIVE definite (`v^H C chi C v = -(Cv)^H chi (Cv)`), so `G = -R` stays the Hermitian PD right-hand matrix the pencil needs. |

A note on the paper's Appendix A: its Eq. 43 pairs the chi components with the
derivatives differently from its own Eq. 21, and its rendering of the second
term in `K_tz22` repeats `d2` where `d1` belongs.  The code follows Eq. 21
(re-derived here from Eqs. 13-14 and checked against the isotropic reduction
`C[d2; -d1] = [-d1; -d2]`), and the appendix's WHICH-chi-in-which-row is
reproduced exactly.  The G2 analytic oracle and the G3 duality oracle both
discriminate that choice (K3 in table M1b).

---

## 2. THE TRAP -- R is not the Gram any more

With `chi_t = I` the shipped code uses ONE object for two different roles:

```python
G = -solver.Rmat        # (a) the pencil's right-hand matrix
...
Ginv = np.linalg.inv(G) # (b) the block field Gram of the Eq.-25 H recovery
Dual = Ginv @ (Lhh @ W)
```

They coincide only because `C I C = -I`, which makes `-R` equal to
`blockdiag(G1, G2)`.  With `chi_t != I` they are different operators:

* the PENCIL needs `R = C[chi_t]C` (Eq. 23-24), and
* the H RECOVERY (Eq. 25) carries NO `chi_t` -- the equation is
  `gamma C [H1;H2] = [k^2 eps_t + S_tt][E1;E2]`, whose left side, tested in V1
  and V2, is `gamma` times the PLAIN block Gram applied to the C-rotated H
  coefficients.  (`chi33` does appear, but only inside `S_tt`, i.e. on the
  right side, where it already is.)

Using `-R` for (b) applies `[chi_t]^-1` to every H partner.  That is an
INTERFACE error: the eigenvalues are untouched (they come from the pencil), the
mode set is untouched, and any energy check that renormalises by the incident
flux still looks plausible -- the failure only shows against an absolute
oracle.  It is exactly the shape the plan's "right conclusion, wrong numbers"
rule warns about.

Handling: `_assemble` retains the two `q^2` diagonal Gram blocks on
`Ggram_blocks` (`None` when nonmagnetic -- 0 extra bytes on the shipped path),
and `_region_modes` branches:

```python
if solver.Ggram_blocks is None:
    Ginv = np.linalg.inv(G); Dual = Ginv @ (Lhh @ W)      # shipped, bit-identical
else:
    G1g, G2g = solver.Ggram_blocks
    LW = Lhh @ W
    Dual = concat([solve(G1g, LW[:qq]), solve(G2g, LW[qq:])])
```

Measured cost of getting it wrong: table M1b, row K1 -- 2.1e-01 against the
analytic oracle where the correct separation reads 1.9e-14 (13 decades).

---

## 3. Measurement tables

### M1 -- G1 reduction

**`mu_cell=None` is a dispatch.**  Two arms, one build: the shipped call
signature (no `mu_cell`) and an explicit `mu_cell=None`, `(2,2)` M=6 at oblique
Bloch phases.  Every retained operator EXACTLY equal (`np.array_equal` true on
`Lmat`, `Rmat`, `Stt`, `Schur`, `Et_blocks[0..1]`); `magnetic is False`,
`mu_cell is None`, `Ggram_blocks is None` on both; none of
`Curl / Kzt / Ktz / G3 / Meps33` retained (audit P3-37 holds).

**`mu = 1` FORCED through the magnetic path** vs the nonmagnetic arm, same
geometry.  Relative `max|diff|`:

| quantity | scalar `ones((2,2))` | tensor `I` |
|---|---|---|
| `Lmat` | 7.947e-15 | 7.947e-15 |
| `Rmat` | 3.284e-17 | 3.284e-17 |
| `Stt` | 4.619e-14 | 4.619e-14 |
| `Schur` | 3.233e-16 | 3.233e-16 |
| eigenvalue SET (nearest-neighbour, relative) | 6.155e-14 | 6.155e-14 |

Through the public entry (uniform `eps = 4` slab, M=5): `dR` 1.414e-15, `dT`
1.332e-15, `dJones` 8.437e-15.  These are RE-SUMMATION differences (the
weighted assembly accumulates the per-cell krons in a different order), so the
bar is 1e-12: 16x over the worst reading and 3.7 decades under the smallest
genuinely magnetic signal in the suite (the M=5 oblique discretization
residual, 5.2e-09, table M3).

### M1b -- FAIL-BEFORE: every magnetic term is load-bearing

Each knockout applied THROUGH the shipped code (monkeypatched `_chi_maps` /
`_assemble`), never a forked copy.  Two metrics: the G2 analytic residual
(uniform isotropic `eps=4, mu=2` slab, theta = 0.35, M=8, max over
{R, T, Jones} x {TE, TM}) and the G3 duality residual (uniform `(LC, LC2)`
tensor pair, conical, M=7).

| arm | analytic | duality |
|---|---|---|
| INTACT | **1.876e-14** | **4.219e-14** |
| K1 -- the R-vs-Gram separation collapsed (`Ggram_blocks = None`) | 2.149e-01 | 1.228e-01 |
| K2 -- `chi33 -> 1` (S_tt unweighted, Eq. 20/A42) | 7.232e-03 | 2.255e-03 |
| K3 -- `chi_t -> I` (R and K_tz unweighted, Eq. 24/A39 + 21/A43) | 2.747e-02 | 1.477e-01 |
| K4 -- `chi12 = chi21 = 0` (the MIXED C-rotated blocks) | 1.876e-14 (invisible) | 3.402e-02 |

K4 is invisible to the analytic arm BY CONSTRUCTION: an isotropic `mu` has no
off-diagonal to lose.  That is why the mixed blocks are gated by duality on an
anisotropic pair, and why the G6 transpose control uses a GYROTROPIC mu.

### M2 -- G2 uniform magnetic slab vs the ANALYTIC Airy oracle

Oracle (written in the test, not imported):
`kz = sqrt(eps*mu - sin^2 th0)`, `Y_TE = kz/mu`, `Y_TM = eps/kz`,
`Y0 = cos th0` / `1/cos th0`,
`M = [[cos d, -i sin d/Y], [-i Y sin d, cos d]]`, `d = k0 t kz`,
`[B; C] = M [1; Y0]`, `r = (Y0 B - C)/(Y0 B + C)`, `t = 2 Y0/(Y0 B + C)`,
`R = |r|^2`, `T = |t|^2`.

**The convention bridge is a measured fact, not an assumption.**  The textbook
`+i` layer matrix belongs to `exp(+i w t)`, where `Im(eps) > 0` is GAIN: with
it a lossy slab reads `dT = 4.4` (it amplifies).  The two forms are IDENTICAL
for real eps and mu, so only a LOSSY arm can pin the bridge -- hence the lossy
rows below.

**M2A -- pin the oracle on the NONMAGNETIC arm first** (max over
{R, T, Jones} x {TE, TM}, M=8, `eps = 4`, `mu = 1`):

| incidence | `mu_cell` omitted (shipped path) | `mu = 1` forced through the magnetic path |
|---|---|---|
| normal | 1.359e-14 | 2.455e-14 |
| theta 0.35 | 6.661e-15 | 1.266e-14 |

Only after this does any magnetic arm use the formula.  Note the Jones is
included and matches in PHASE (not just magnitude), which is what makes the
oracle usable for the sign of the whole magnetic assembly.

**M2B/M2C -- the magnetic arms** (same metric, M=8):

| eps | mu | theta 0 | theta 0.35 |
|---|---|---|---|
| 4.0 | 2.0 | 4.298e-14 | 3.064e-14 |
| 1.0 | 4.0 | 1.870e-14 | 1.177e-14 |
| 4.0 | 2.0 + 0.3i (LOSSY mu) | 3.689e-14 | 3.014e-14 |
| 4.0 + 0.2i | 2.0 (lossy eps in a magnetic host) | 3.613e-14 | 2.766e-14 |

Bar 1e-12: 23x over the worst and 4.4 decades under the M=5 oblique residual.

**M2D -- spectral convergence** (a uniform region is smooth, so the residual
must FALL steeply), `eps = 4`, `mu = 2`, theta = 0.35:

| M | residual |
|---|---|
| 5 | 2.414e-08 |
| 7 | 3.64e-13 |
| 8 | 3.064e-14 |

Ratio M=5 -> M=8 = 7.9e+05; the bar is a drop of at least 1e3 (790x of
headroom).  At NORMAL incidence both ends already sit at round-off (~1e-15), so
the ladder claim is made at oblique, where the transverse Bloch phase has to be
resolved -- the module's documented degree-limited-uniform regime.

Aside, recorded because it is easy to misread: at theta = 0.35 the TOTAL
`sum R + sum T` of the `eps=4, mu=2` slab reads 1.0146, not 1.0 -- the
propagating `(-1, 0)` order of the (uniform!) layer picks up the degree-limited
residue.  The SPECULAR R and T are nonetheless right to 1e-14, which is exactly
what the table measures.  Closure claims are made on properly resolved
configurations in M5.

### M3 -- G3 electromagnetic duality

`(E, H, eps, mu) -> (Z0 H, -E/Z0, mu, eps)` is an exact symmetry of Maxwell;
with VACUUM (self-dual) half-spaces the grating `(eps, mu)` and the SWAPPED
grating `(mu, eps)` are the same physical problem with the incident and
outgoing polarizations rotated by `D = [[0,-1],[1,0]]` in the (s, p) frame --
a UNIT-magnitude map at ANY incidence, which is why the comparison is made
there and not in the lab (x, y) tangential basis (where the same map carries
`1/cos theta` factors and mixes the two drives at conical incidence).  Rules:
per-order `R_dual(p) = R_orig(s)` and `R_dual(s) = R_orig(p)` (likewise T), and
`J_dual = -D J_orig D^-1` (the sign is the two p-hat conventions in play: the
incident p-hat is the optics one, `-(k^ x s^)`, the reflected one `+(k^_r x s^)`).

Metric below: max over {R, T} x {both drive pairings} x {all orders}, and the
Jones residual; "control" is the UNROTATED comparison.

**M3A -- PATTERNED electric cell vs its PURELY MAGNETIC dual** (arm A has
`mu = 1`, i.e. the SHIPPED and already-gated anisotropic path; arm B is
`eps = vacuum, mu = the whole pattern`).  LC host + isotropic pillar:

| incidence | M=5 | M=6 | M=7 | M=8 | control |
|---|---|---|---|---|---|
| normal | 5.991e-03 | 1.165e-03 | 4.925e-04 | 2.512e-04 | 2.94e-01 |
| theta 0.30 | 4.988e-03 | 1.095e-03 | 1.297e-04 | 4.266e-05 | 4.08e-01 |
| conical 0.30 / 0.70 | 1.542e-03 | 1.182e-04 | 2.003e-04 | 7.038e-05 | 5.47e-01 |

(Jones residuals track: 8.556e-03 -> 2.317e-04 normal, 2.932e-03 -> 7.469e-05
conical.)  The staggered DISCRETIZATION is not self-dual (`E3` is expanded in
V3 while `H3` lands in Vw), so the two arms agree only to discretization
accuracy -- the claim is therefore two-sided: bar = 3x the worst M=8 reading
(7.6e-4) AND a ladder drop of at least 5x (measured 21.9x at worst).

**M3B -- UNIFORM `(eps, mu)` tensor pair -- SPECTRAL:**

| incidence | M=5 | M=6 | M=7 | M=8 |
|---|---|---|---|---|
| normal | 1.266e-14 | 4.230e-14 | 2.298e-14 | 1.620e-13 |
| conical 0.30 / 0.70 | 5.218e-09 | 1.193e-11 | 4.219e-14 | 5.868e-14 |

Jones at M=8: 4.198e-14 / 1.755e-13.  Bar 1e-11 (57x over the worst reading);
the no-rotation control sits at 3.01e-01, 10 decades above the bar, so it is
the ROTATION that is being measured and not a coincidence of magnitudes.

**M3C -- BOTH sides patterned and anisotropic** (eps = LC/gyrotropic cell,
mu = LC2/1.6 cell):

| incidence | M=5 | M=6 | M=7 | M=8 |
|---|---|---|---|---|
| normal | 1.742e-01 | 9.899e-04 | 1.194e-04 | 1.997e-05 |
| conical 0.30 / 0.70 | 3.835e-03 | 6.347e-05 | 1.208e-04 | 6.523e-05 |

### M4 -- G4 the 1-D engines, through duality

**ENGINE CENSUS (2026-09-10).**  `grep --include='*.py' -riE 'permeability|\bmu'`
over `lumenairy/elements/{rcwa,pmm,berreman.py}`: NO 1-D diffraction engine in
the library accepts a permeability.

| engine | permeability? |
|---|---|
| `rcwa_jones_1d`, `rcwa_efficiency_1d` | no -- nonmagnetic (`_core.py:2257` "Non-magnetic (`mu = 1`)") |
| `pmm_jones_1d`, `pmm_efficiency_1d`, `PMMStack` | no -- the `mu` occurrences in `pmm/_core.py` are the eps-free geometric EIGENVALUE of `Kx2` (`q^2 = eps - mu`) and the slant metric's `mu^lm = sqrt(g) g^lm`; both are coordinate artefacts, not a material |
| `berreman_jones_1d`, `BerremanStack` | no -- the layer matrix docstring states `mu = 1` |
| `eme/eme_2d_vector._build_generator(..., mu_xy)` | YES, a scalar `(Nx, Ny)` permeability -- but it is a WAVEGUIDE MODE solver (propagation constants of a cross-section), not a diffraction engine, so it cannot produce per-order R/T for a grating |

So the 1-D check runs through DUALITY, which is strictly stronger than "no
check": the ELECTRIC stripe both 1-D engines CAN solve is the dual of the
MAGNETIC stripe this engine solves.  Chain: magnetic staggered -> (exact
duality) -> 1-D PMM / 1-D RCWA.

Fixture: period 0.90 um, wl 0.55 um, depth 0.30 um, fill 0.5, vacuum
half-spaces, ridge = rotated-LC tensor, groove = 2.10 I.  MAGNETIC arm:
`eps_cell = vacuum`, `mu_cell = the stripe`.  R/T need no rescaling (each
library efficiency ROW is already that drive's power response); the JONES is a
TANGENTIAL-amplitude matrix, and duality maps tangential amplitudes by
`A = [[0, -kz], [1/kz, 0]]` in a classical mount, so the rule is
`J_dual = -A J A^-1` (which collapses to `C J C` at normal incidence).

| theta | M | per-order max\|dR\|,\|dT\| vs `pmm_jones_1d` | max\|dJones\| | y-forbidden leak | unswapped control |
|---|---|---|---|---|---|
| 0 | 5 | 2.660e-04 | 8.387e-05 | 6.15e-29 | 3.44e-02 |
| 0 | 6 | 8.818e-05 | 8.768e-05 | 1.74e-27 | 3.44e-02 |
| 0 | 7 | 8.787e-06 | 2.132e-05 | 1.73e-27 | 3.44e-02 |
| 0 | 8 | 9.154e-06 | 2.218e-05 | 4.37e-26 | 3.44e-02 |
| 0.22 | 5 | 2.769e-03 | 1.442e-03 | 2.87e-28 | 4.11e-02 |
| 0.22 | 6 | 1.607e-04 | 5.240e-05 | 3.09e-27 | 4.07e-02 |
| 0.22 | 7 | 2.149e-05 | 2.316e-05 | 3.60e-27 | 4.07e-02 |
| 0.22 | 8 | **7.630e-06** | **1.718e-05** | 4.68e-26 | 4.07e-02 |

Against the SECOND oracle at M=8 (`rcwa_jones_1d`, 81 orders): 8.292e-06 /
2.008e-05 at normal, 6.757e-06 / 1.519e-05 at theta 0.22.  The two 1-D oracles
agree with EACH OTHER to 8.620e-07 (normal) / 8.736e-07 (0.22) on R/T and
2.111e-06 / 1.983e-06 on the Jones -- 34x / 35x under the bars -- so the
oracle's own floor is not what is being measured.

Bars: 3e-5 on R/T (3.9x over the worst M=8) and 7e-5 on the Jones (4.1x), a
ladder drop of at least 20x (measured 363x at theta 0.22), and 1e-20 on the
y-forbidden orders (6 decades above the measured round-off floor).  Recorded
for completeness: with the normal-incidence Jones rule used at theta = 0.22 the
residual sits at 3.90e-03 and does NOT converge -- the `1/cos theta` factors
are real, and the probe carries them.

### M5 -- G5 closure with a HERMITIAN permeability

A Hermitian mu absorbs nothing: the Poynting dissipation term carries the
anti-Hermitian parts of BOTH eps and mu.  The gyrotropic
`m12 = -m21 = +0.4i` used here is Hermitian, hence lossless despite being
complex -- the case a naive `Im(mu) != 0 -> lossy` predicate gets wrong.

**M5A -- lossless: `|sum R + sum T - 1|`**

| cell | M=6 normal | M=6 conical | M=8 normal | M=8 conical |
|---|---|---|---|---|
| uniform LC eps, gyrotropic mu | 1.998e-14 | 8.857e-12 | 1.066e-13 | 2.909e-14 |
| patterned LC/iso eps, gyro/1.2 mu | 1.138e-06 | 1.732e-03 | 1.934e-09 | 1.480e-06 |
| patterned eps, LC-like mu | 7.966e-07 | 1.256e-04 | 1.453e-09 | 7.774e-08 |
| VACUUM eps, patterned mu | 5.359e-07 | 3.907e-05 | 1.011e-09 | 2.347e-08 |

Bars: 1e-11 for the uniform arm (94x) and 1e-4 for the patterned one (68x,
corner-capped ALGEBRAIC convergence, not round-off), plus the two-sided
companion: the patterned defect must fall at least 10x from M=6 to M=8
(measured 1170x).

**M5B -- lossy mu must close BELOW 1** (`Im(m11) = +0.25`):

| M | sum R+T, incident Ex | incident Ey |
|---|---|---|
| 6 | 0.953739 | 0.552147 |
| 8 | 0.953733 | 0.552172 |

A deficit of 0.046 that is stable to 6e-06 in M.  Bar: below 0.99 (4.6x inside
the measured deficit, decades outside the ~1e-6 closure defect of the
corresponding LOSSLESS cell).

**M5C -- the tripwire's magnetic predicate** (LC/iso eps + gyro/1.2 mu,
theta 0.25 / phi 0.60):

| arm | closure defect | warnings raised |
|---|---|---|
| Hermitian mu, M=8 (resolved) | 1.5e-06 | 0 |
| Hermitian mu, M=3 (under-resolved) | 8.488e-01 | 2 (one per incident pol) |
| LOSSY mu, M=8 | (no unity claim) | 0 |

The M=3 arm is ENGINEERED under-resolution -- M=3 is the minimum the basis
admits -- and its defect is 17x the shipped `_STAG_CLOSURE_TOL` = 5e-2 window,
so the firing decision is not near a knife edge.  The tripwire also stays
silent on every existing fixture: the eleven shipped staggered / pure-2D suites
are green (Section 4).

### M6 -- G6 the x<->y transpose with the mu blocks swapped

Transpose the cell about `x = y`: swap the grid axes, the periods, AND the
tensor components (`e11<->e22`, `e12<->e21`, and the SAME on mu).  Orders map
`(m, n) -> (n, m)` and the Jones rows and columns swap.  LC/iso eps +
gyrotropic/1.2 mu, M=6:

| arm | per-order residual | Jones residual |
|---|---|---|
| correct placement | **2.801e-14** | **3.846e-14** |
| `m12`/`m21` SWAPPED in one arm | 5.218e-03 | 1.332e-01 |
| `e12`/`e21` swapped in one arm (control) | 2.243e-14 | 3.434e-14 |

The third row is why the PERMEABILITY needs its own gyrotropic discriminator:
the rotated-LC permittivity has `e12 = e21`, so swapping THOSE is a no-op.
Bars: correct < 1e-11 (260x over the measurement, 8 decades under the break),
swapped > 1e-7.

### M7 -- G7 guards

All raise with the stated type and a message naming the limitation and the
alternative:

| condition | type |
|---|---|
| OUT-OF-PLANE mu (`m13`/`m23`/`m31`/`m32` above the relative 1e-12 floor) | `NotImplementedError` |
| mu together with an OUT-OF-PLANE eps (entry, `add_layer(eps_cell=)`, `add_layer(eps=)`) | `NotImplementedError` |
| `mu_superstrate` / `mu_substrate` != 1 (magnetic half-space) | `NotImplementedError` |
| singular `[mu_t]` (`det = 0`) | `ValueError` |
| `m33 = 0` | `ValueError` |
| `mu_cell` shape not `(Nx,Ny)` / `(Nx,Ny,3,3)` | `ValueError` |
| `mu_cell` grid != `eps_cell` grid | `ValueError` |
| non-square `mu_cell` | `ValueError` |
| zero scalar `mu_cell` | `ValueError` |
| both `mu` and `mu_cell` | `ValueError` |
| uniform `mu` of a bad shape | `ValueError` |
| `_homog_geom_cache` on a magnetic solver | `ValueError` |

And the negative side of the floor: a `1e-16` stray in `m13` does NOT trip the
block-form guard, and its R is BIT-IDENTICAL (`max|dR| = 0.0`) to the
exact-identity mu -- the 2x2 inverse and `chi33` never read the out-of-plane
slots once the guard has passed.  `mu_superstrate=1.0` / `mu_substrate=1` are
accepted (they are the nonmagnetic default written out).

### M9 -- G9 the `layer_absorption` budget with a lossy MAGNETIC layer

`retain_internal=True` on a two-layer stack whose FIRST layer is magnetic and
lossy (LC/iso eps, `Im(m11) = +0.25` mu) over an isotropic `eps = 2.25` layer,
theta 0.2 / phi 0.4.  The identity `sum_i A_i == 1 - sum R - sum T` is
CROSS-MACHINERY: the left side is the internal block-Gram flux quadrature, the
right side the Rayleigh far field.

| M | sum A (Ex / Ey) | 1 - R - T (Ex / Ey) | closure |
|---|---|---|---|
| 5 | 0.055130 / 0.444410 | 0.055121 / 0.444398 | 1.285e-05 |
| 6 | 0.055474 / 0.444840 | 0.055473 / 0.444839 | 7.042e-07 |
| 7 | 0.055441 / 0.446019 | 0.055441 / 0.446019 | 2.192e-08 |
| 8 | 0.055314 / 0.445808 | 0.055314 / 0.445808 | 9.867e-10 |

The flux bilinear form is the eps-free block Gram of the (nonmagnetic)
half-space assembly, which the magnetic path does not touch -- but that is a
prediction, and this is the measurement.  Bar 1e-8 at M=8 (10x) plus a ladder
drop of at least 100x (measured 13,000x).

### M8 -- cost

`(3,3)` grid, M=8 (`q^2 = 441`, pencil `2 q^2 = 882`), patterned LC/iso eps,
patterned LC2/1.6 mu; three repeats:

| arm | assemble (s) | region eig (s) | peak (MiB) | `Ggram_blocks` |
|---|---|---|---|---|
| eps-only tensor | 0.32 / 0.34 / 0.39 | 15.20 / 18.09 / 20.63 | 142.7 | `None` |
| magnetic (eps + mu) | 0.44 / 0.43 / 0.42 | 16.23 / 19.33 / 17.74 | 148.7 | 2 x `q^2` |

* Assembly: +20-35% (four `R` blocks instead of two unweighted krons, one
  `Gw_chi`, four weighted `K_tz` terms).  It is ~2% of the region solve.
* The EIG does not move: the pencil dimension is unchanged and the machine's
  own scatter over three repeats (15.2-20.6 s on the eps-only arm alone) is
  larger than any difference between the arms.
* Peak: +6.0 MiB = +4.2%, which is EXACTLY the two retained `q^2` Gram blocks
  (`441^2 x 16 bytes x 2 = 6.2 MiB`).  Storing the full `2q^2 x 2q^2` Gram
  instead would have cost 4x that for the same information.
* No new retained dead operators: `Gw_chi`, the weighted `K_tz` terms and the
  `Curl` stay assembly locals, and `Ggram_blocks` is `None` on every
  nonmagnetic solver.

---

## 4. Test suite

`tests/unit/test_pmm2d_staggered_magnetic.py` -- **31 tests, 75-101 s**
single-threaded, slowest test 9.5 s (limits: file < 3 min, test < 40 s, grids
<= (3,3), M <= 8).  Every bar cites the table above and is derived from a
measurement made in this build; no cross-build value is pinned anywhere.

| gate | tests |
|---|---|
| G1 | `test_g1_mu_none_dispatch_is_bit_identical`, `test_g1_unit_mu_forced_through_the_magnetic_path` |
| G2 | `test_g2_the_analytic_oracle_matches_the_nonmagnetic_path`, `test_g2_uniform_magnetic_slab_vs_analytic` (x4 params), `test_g2_uniform_slab_converges_spectrally_in_m` |
| G3 | `test_g3_duality_uniform_tensor_pair_is_spectral`, `test_g3_duality_patterned_cell_ladder` |
| G4 | `test_g4_magnetic_stripe_matches_both_1d_engines`, `test_g4_ladder_and_y_momentum` |
| G5 | `test_g5_hermitian_mu_is_lossless`, `test_g5_patterned_closure_improves_with_m`, `test_g5_lossy_mu_closes_below_one`, `test_g5_tripwire_recognises_a_hermitian_mu_as_lossless` |
| G6 | `test_g6_transpose_symmetry_with_the_mu_blocks_swapped`, `test_g6_mu_mixed_block_placement_control` |
| G9 | `test_g9_absorption_budget_closes_for_a_magnetic_layer` |
| fail-before | `test_failbefore_the_r_vs_gram_separation_is_load_bearing`, `test_failbefore_each_chi_weight_is_load_bearing` (x2), `test_failbefore_the_mixed_chi_blocks_are_load_bearing` |
| G7 | `test_g7_out_of_plane_mu_raises`, `test_g7_mu_with_an_out_of_plane_eps_raises`, `test_g7_magnetic_half_space_raises`, `test_g7_shape_and_singularity_guards`, `test_g7_homog_geom_cache_refuses_a_magnetic_region`, `test_g7_float_noise_in_m13_does_not_trip_the_block_form_guard` |
| API | `test_api_uniform_and_patterned_mu_combinations_agree`, `test_api_two_identical_magnetic_layers_share_one_eig` |

Regression suites re-run green in the worktree (they are what pins "the
nonmagnetic path did not move" and "the tripwire does not fire on any existing
fixture"): see Section 6 of the final report.

`ruff check lumenairy/ tests/` -- clean;
`ruff check validation/probe_pmm2d_staggered_magnetic/` -- clean.

---

## 5. Scope / open items

* **In-plane only.**  Out-of-plane mu (and mu with an out-of-plane eps) raise:
  the first-order `4 q^2` generator (`_assemble_oop`) has no permeability
  blocks -- it eliminates `G3` assuming `mu = 1`.  Adding them is a
  self-contained follow-on (the `G3` elimination gains `chi33`, and the two
  transverse magnetic rows gain `[chi_t]`), gated the same way this build was.
* **Half-spaces stay nonmagnetic.**  A magnetic half-space changes the wave
  impedance, hence the Rayleigh flux normalisation AND the incident-amplitude
  overlap; the keywords exist only so that asking for one is loud.
* **The Wood-anomaly `_grazing_safe_wavelength` list** does not yet include a
  magnetic layer's principal indices (`sqrt(eps mu)`).  It is a warning
  heuristic and the layer kinds it already skips (scalar layers) are skipped
  deliberately, so nothing silently changed; a magnetic layer near a LAYER-mode
  cutoff will simply not be nudged.  Left alone in this build to keep the merge
  with the concurrent Wood-anomaly work conflict-free.
* **JAX twin** -- the staggered path is NumPy-only, magnetic or not.
* **`layer_absorption` with a magnetic layer** IS gated (table M9, closing
  9.9e-10 at M=8 on a lossy-mu layer) -- but only for an in-plane magnetic
  layer in a NUMPY cascade; the tapered / z-staircase helpers remain
  hybrid-only, magnetic or not.
