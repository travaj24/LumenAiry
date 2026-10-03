# BUILD -- curved cells for the pure staggered 2-D PMM, Phase E1 (out-of-plane tensors and slanted walls inside curved cells)

Date: 2026-10-03.  Status: BUILT + gated, not pushed.
Builder: Claude Opus 5.5 (model ID `claude-opus-5-5`).
Mount: worktree `C:/tmp/lum_curved_e1`, branch
`feat/pmm2d-curved-e1-oop-slant`, built on `eae470d9` (Phase D); Windows 11
(tesla-ryzen), CPython 3.14.6, numpy 2.4.4, scipy 1.17.1,
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` on the command
line, `lumenairy.__file__` asserted under the tree being measured in every
probe (`build_e1/_common.py`).  PRE tree for byte identity: `git archive
eae470d9` extracted to `C:/tmp/curved_pre_e1/`.  The box was shared with four
sibling agents for the whole build, so every WALL TIME below is an upper
bound; no accuracy number depends on the load.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.5
(Phase E, approved 2026-10-02; the maintainer split it into E1 = this build,
out-of-plane tensors and slant under a map; E2 = per-layer maps; E3 = the JAX
twin).  Phases A-D: `docs/audits/BUILD_PMM2D_CURVED_{A,B,C,D}_*.md`; the
shipped out-of-plane, slant and magnetic builds:
`BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md`,
`BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md`,
`BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md`.
Evidence: every number below is read from a JSON file in
`validation/probe_pmm2d_curved/build_e1/` (the probe that wrote it is named
in each table; `_e1common.py` holds the fixtures, the Berreman slab oracle and
the engineered mutations).  Tests:
`tests/unit/test_pmm2d_staggered_curved_e1.py`.

---

## 0. Words used here

* **Coordinate map** `(x, y) = Phi(u, v)`, Jacobian `J = [[x_u, x_v], [y_u,
  y_v]]`, `sg = det J`, metric `g = J^T J`, `adj(J) = sg J^-1 = [[y_v, -x_v],
  [-y_u, x_u]]`.  **Covariant fields** `E' = Lambda^T E` for a 3-D map with
  Jacobian `Lambda`: the components along the frame's coordinate lines.
* **Out-of-plane (OOP) tensor**: a permittivity with `e13`, `e23`, `e31` or
  `e32` non-zero (a liquid-crystal director tilted out of the plane).  Such a
  layer runs the **first-order generator**: the staggered pencil
  `A x = q B x` on `x = [E1; E2; G1; G2]` (dimension `4 q^2`, `G = i Z0 H`,
  `D = grad / k0`, `D3 = i q`), as opposed to the in-plane **second-order
  pencil** `L E = gamma^2 (-R) E` (dimension `2 q^2`).
* **Permeability blocks**: the places where the inverse permeability
  `chi = mu^-1` enters a discretisation -- Granet's `R = C[chi_t]C`,
  `K_tz = C[chi_t][d2; -d1]` and the `chi33`-weighted Gram `Gw_chi` of the
  in-plane pencil (Eqs. 20, 21, 24).  The shipped first-order generator has
  none: it assumes `mu = 1`.  A map makes every region magnetic
  (`chi_t = g / sg`, `chi33 = 1 / sg` in vacuum), so it needs them.
* **Slant**: a constant x-z / y-z shear of a layer, PUBLIC tangent pair
  `slant = (t_x, t_y)`; the INTERNAL shear of the frame `x = u + t w` is its
  negative.  **Composite map**: the slant composed with the in-plane map,
  `(x, y, z) = (Phi(u, v) + t w, w)`.
* **The two gauge constants** of the shipped out-of-plane path:
  `_OOP_ROT_SIGN = -1` (the 180-degree rotation about z that the basis glue
  and the far-field kernel carry between them, applied as a sign on the four
  OOP entries and on the slant vector) and `_OOP_H_GAUGE = -1j` (the constant
  relating the generator's `G` state to the Eq.-25 `H` partner of the
  in-plane regions and half-spaces).  Both were MEASURED (Berreman), not
  derived.
* **Fail-before / mutation**: a deliberately broken arm reached through the
  real code path (`_e1common.mutate`), restored afterwards.

---

## 1. The derivation (written before the build)

### 1.1 Covariant Maxwell with a general frame, and the two forms of the constitutive law

Time convention `exp(-i omega t)`; `G = i Z0 H` (`Z0 = sqrt(mu0 / eps0)`),
`D = grad / k0`.  Then `curl E = i omega mu0 mu H` and `curl H = -i omega
eps0 eps E` read `D x E = mu G` and `D x G = eps E` (the shipped generator's
normalisation, with `mu = 1`).  For ANY frame `r = r(u, v, w)` with Jacobian
`Lambda = d r / d(u, v, w)` and covariant fields `E' = Lambda^T E`,
`G' = Lambda^T G`, the curl of a covariant field is a contravariant density
(`(curl' E')^i = eps^{ijk} d_j E'_k = det(Lambda) (Lambda^-1 curl E)^i`; the
Phase A verifier's section 3.1 writes the index computation), so

    (D x E')^i = mu'^{ij} G'_j ,    (D x G')^i = eps'^{ij} E'_j ,
    eps' = det(Lambda) Lambda^-1 eps Lambda^-T ,    mu' likewise
                                                         (Ward & Pendry 1996)

with `D x` the metric-free curl in `(u, v, w)`.  Write `c = D x E'`
(`c^1 = D2 E3 - D3 E2`, `c^2 = D3 E1 - D1 E3`, `c^3 = D1 E2 - D2 E1`) and
`chi' = (mu')^-1`.  The constitutive law has two equivalent block forms, and
the derivation uses ONE of each where the discretisation makes it exact:

    transverse:    G'_t = chi'_tt c_t + chi'_t3 c^3                      (I)
    longitudinal:  G'_3 = (c^3 - mu'^{3t} G'_t) / mu'^{33}                (II)

(II) is the `33` row of `mu' G' = c` solved for `G'_3`; (I) is the `t` rows
of `G' = chi' c`.  Their consistency is the block-inverse identity
`chi'_3t chi'_tt^-1 = -mu'^{3t} / mu'^{33}` and `chi'_33 - chi'_3t
chi'_tt^-1 chi'_t3 = 1 / mu'^{33}` -- so (I) and (II) are the SAME law, and
neither needs the full `3 x 3` inverse.

Why these two forms.  In the staggered basis `c^1 = D2 E3 - D3 E2` lies
EXACTLY in `V2`, `c^2` in `V1` and `c^3 = D1 E2 - D2 E1` in `Vw = B (x) B`
(the de Rham property `d Btilde in span(B)`) -- so `c` is known exactly from
the unknowns and (I), tested with the plain `V2` / `V1` Grams, is an exact
Galerkin statement whose only approximation is the weight `chi'`.  The G rows
need `G'_3` only under a single derivative (`D1 G'_3` tested in `V2`, `D2
G'_3` in `V1`); with the derivative moved onto the CONTINUOUS test function
(`D1 V2`, `D2 V1` both lie in `Vw`), the `L2` projection of `G'_3` onto `Vw`
is all that is needed -- (II) supplies it.

### 1.2 The composite metric (slant x map) and which terms are new

The in-plane map alone: `Lambda_m = blockdiag(J, 1)`.  The shipped slant
alone: `A = [[1, 0, t_x], [0, 1, t_y], [0, 0, 1]]` (internal `t`; `det A =
1`).  The slanted layer under a map: `(x, y, z) = (Phi(u, v) + t w, w)`,

    Lambda = [[J, t], [0, 1]] = A Lambda_m ,     det Lambda = sg .

(The other order, `Phi(u + t_u w, ...)`, would move the circle off the map's
own circle; the physical slanted pillar is the cross-section TRANSLATED by
`t z`, which is this composite, and it keeps `Phi - id` lattice-periodic.)
Hence every effective tensor is the map's congruence of the shear's:

    eps' = M(S(eps)),   S(X) = A^-1 X A^-T,   M(X) = sg Lambda_m^-1 X Lambda_m^-T

-- the shipped `_slant_congruence` followed by the Phase D congruence with
its out-of-plane entries (`e'_a3 = adj(J)_ak e_k3`, `e'_3b = e_3k adj(J)_bk`,
no `1 / sg` on them; `e'_33 = sg e_33`, so the shear still leaves `e33`
alone).  For a BLOCK-FORM material `mu` (and vacuum):

    S(mu) = [[mu_t + mu33 t t^T, -mu33 t], [-mu33 t^T, mu33]]
    S(mu)^-1 = A^T mu^-1 A = [[chi_t, chi_t t], [t^T chi_t, t^T chi_t t + chi33]]

so, after the map,

    chi'_tt = J^T chi_t J / sg                     (= Phase D's chi_t: UNCHANGED by the shear)
    chi'_t3 = J^T chi_t t / sg = chi'_tt tau ,     tau = J^-1 t = adj(J) t / sg
    1 / mu'^{33} = 1 / (sg mu33)                   (= Phase D's chi33: UNCHANGED)
    -mu'^{3t} / mu'^{33} = tau^T

and (I), (II) become

    G'_t = chi_t' (c_t + tau c^3) ,     G'_3 = chi33' c^3 + tau . G'_t          (*)

The composite metric `Lambda^T Lambda = [[g, J^T t], [t^T J, 1 + |t|^2]]`
against its two parents (`blockdiag(g, 1)` for the map, `[[I, t], [t^T, 1 +
|t|^2]]` for the shear) carries exactly ONE new coupling: `J^T t`, the shear
seen through the map, which reaches the generator as `tau = J^-1 t` -- the
shear vector expressed in `(u, v)`, now VARYING across every mapped cell --
and as `kappa = chi_t' tau` in the transverse rows.  Every other term is
inherited unchanged: `chi_t'` and `chi33'` are Phase D's (the shear does not
touch them), and the shear's `|t|^2` (the `33` metric entry) CANCELS in the
Schur complement `1 / mu'^{33}` -- exactly as `mu^{33} = g^{33} = 1` let the
shipped slant keep its strong `G3` elimination.  With `J = I` (no map) (*)
is the shipped slant: `G_t = c_t + t c^3`, `G_3 = c^3 + t . G_t` (the
EXPERIMENT_PMM2D_STAGGERED_SLANT S1.3 rows); with `t = 0` it is the map
alone.  A gyrotropic block-form `mu` changes nothing above: `kappa =
chi_t' tau` and the `tau` of (II) hold for any `chi_t` (the block-inverse
identities do not use symmetry).

What the shear does NOT touch: the TANGENTIAL covariant fields.  `Lambda^T =
[[J^T, 0], [t^T, 1]]`, so `E'_t = J^T E_t` exactly as under the map alone --
the far-field cofactor projector, the exact incident decomposition, the
plain-Gram flux form and the interface square match are the map's, and the
shear's only far-field trace is the shipped frame-anchor phase
`exp(-i alpha_m . t d)` on the transmitted orders (the substrate plane sits at
`Phi(u, v) + t d`, a translation of a homogeneous half-space).

### 1.3 The first-order generator with permeability blocks

Test (I) for `G1` in `V2` and for `G2` in `V1`, collect the `q` terms
(`D3 = i q`), and order the rows as the shipped generator does (row 0 tested
in `V1`, row 1 in `V2`):

    q (-R) [e1; e2] = [ -i Ggram1 g2 + i Ktz1 e3 + i <V1| kappa2 |Vw> c3 ;
                        +i Ggram2 g1 + i Ktz2 e3 - i <V2| kappa1 |Vw> c3 ]

    R   = C[chi_t']C = [[-chi22, chi21], [chi12, -chi11]]   (V1 x V1, V1 x V2, ...)
    Ktz = C[chi_t'][d2; -d1]:  row 1 = -chi22 d1 + chi21 d2 (in V1),
                               row 2 = -chi11 d2 + chi12 d1 (in V2)

-- EXACTLY the in-plane pencil's two permeability operators (Granet Eqs. 21,
24; the magnetic build's index placement), so the build calls the SAME
methods (`_chi_R`, `_chi_Ktz`, factored out of `_assemble`; one
implementation).  The derivative terms `<V1| chi D1 |V3> e3` are exact
because `D1 V3` lies in `V1` strongly.  With `chi_t' = I`: `-R =
blkdiag(Ggram1, Ggram2)` and `Ktz = [-P13; -P23]`, the shipped rows
`q Ggram1 e1 = -i Ggram1 g2 - i P13 e3`, `q Ggram2 e2 = +i Ggram2 g1 - i P23
e3`; with `kappa = t` the shipped slant blocks `+i t_y <V1|Vw> G^3`,
`-i t_x <V2|Vw> G^3`.

The two G rows (`D x G' = eps' E'`, transverse components, tested in `V2`
and `V1`) keep their shipped form,

    q Ggram2 g1 = -i [A21 e1 + A22 e2 + A23 e3] - i <V2| D1 G'_3>
    q Ggram1 g2 = +i [A11 e1 + A12 e2 + A13 e3] - i <V1| D2 G'_3>

with `G'_3` from (II) and the derivative on the continuous test function:

    -i <V2| D1 G'_3> = +i CwE2^H g3 + i <D1 V2| tau1 |V2> g1 + i <D1 V2| tau2 |V1> g2
    -i <V1| D2 G'_3> = +i CwE1^H g3 + i <D2 V1| tau1 |V2> g1 + i <D2 V1| tau2 |V1> g2
    g3 = Gw^-1 Gw_chi c3 ,  c3 = Gw^-1 (CwE2 e2 - CwE1 e1) ,  Gw_chi = <Vw| chi33' |Vw>

`c3` is the shipped strong curl (`G3S`), `g3` its `chi33`-weighted `L2`
projection onto `Vw` -- the in-plane `S_tt` middle operator
`Gw^-1 Gw_chi Gw^-1` (the third shared method, `_chi_Gw`).  `CwE^H g3` is
exact for `g3` in `Vw`; the `tau` terms are written with the derivative on
the test function (`'dL'` flavour) because `tau G'_t` is not in a basis space
and may jump across a cell wall (`J` is only `C0`), while `D1 V2` and `D2 V1`
are smooth inside every cell; for constant `tau = t` they equal the shipped
`-i t <V2| D1 |V2>` blocks by the periodic integration by parts the shipped
code already uses (`dtb = -(dbt)^H`).

The `E3` row (`(D x G')^3 = eps'^{3j} E'_j`, tested in `V3`) carries no
permeability and is unchanged:
`e3 = A33^-1 (P23^H g1 - P13^H g2 - A31 e1 - A32 e2)`, `A33 = <V3| sg e33
|V3>`.  `B = blkdiag(-R, Ggram2, Ggram1)`.

### 1.4 What replaces the e33-Schur when mu33 != 1

Nothing replaces it: the `e33`-Schur IS the `E3` elimination, which lives on
the permittivity side (`D x G = eps E` has no `mu`), and it stays pointwise
(`A33` is ONE weighted mass, `e'_33 = sg e33 != 0` wherever `sg > 0`).  The
permeability's `33` entry enters the OTHER longitudinal elimination: `G3`.
In the shipped `mu = 1` generator `G3 = c3` is eliminated STRONGLY (the curl
lands exactly in `Vw`); with `chi33' != 1` the strong part survives (`c3` is
still exact) and the G rows read its `chi33`-weighted projection `Gw^-1
Gw_chi c3` -- the dual of the `e33` treatment, one weighted mass between exact
`Vw` functions, never a product of separately discretised factors (the trap
the 1-D `gen2` prototype fell into).

### 1.5 Does the Cholesky whitening survive a full chi_t block?

Yes wherever `chi_t'` is Hermitian positive definite pointwise: for any
coefficient vector `v`, `v^H (-R) v = INT (C v)^H chi_t' (C v) du dv > 0`, so
`B = blkdiag(-R, Ggram2, Ggram1)` is HPD and the shipped `Lc = chol(B)`
whitening applies unchanged -- with the full `2 q^2` block `-R` (its mixed
`chi12` / `chi21` blocks included) in place of the two diagonal Grams.  This
covers every map in vacuum (`chi_t' = g / sg`, real symmetric positive
definite; near a singular vertex it grows like `1 / r`, integrable, and stays
positive) and every LOSSLESS material `mu` (Hermitian `mu_t`).  It fails for
a non-Hermitian `chi_t'` (a LOSSY `mu`); the build detects that structurally
(the assembled `-R` is not Hermitian, a gap of `Im mu` against round-off) and
the region eig falls back to the generalized eig (QZ).  The forward/backward
FLUX split is unaffected either way: the z-flux `INT (E_t x H_t*)_z dx dy` is
metric-free under a map (Phase A verifier 3.3), so its form is the PLAIN
Gram, which the generator still carries on its two G rows -- the build reads
the whitening factors `L1`, `L2` from those rows (on the shipped branch they
are the same matrices, byte for byte).

### 1.6 The two gauge constants in the mapped setting

* `_OOP_ROT_SIGN`.  The rotation `rho` (180 degrees about z) maps `u -> -u`,
  `x -> -x` together, so `J` is invariant, and it commutes with
  `Lambda_m = blockdiag(J, 1)`: `M(rho X rho) = rho M(X) rho`.  It flips the
  four OOP entries of `eps'` and the slant vector `t`, hence `tau = J^-1 t`.
  So the shipped constant applies verbatim: to the lab tensor's OOP entries
  before the congruence or, equivalently (the congruence is linear), to the
  four mapped OOP weights after it; and to `t` before `tau` is formed.  A
  map-dependent gauge would show as a sheared-map slab missing Berreman at
  oblique / conical incidence with the shipped sign (the stop condition).
* `_OOP_H_GAUGE`.  It relates `G = i Z0 H` to the Eq.-25 partner; under a map
  both are the COVARIANT tangential components (`G'_t = J^T G_t`, the
  half-spaces' `H'_t = J^T H_t`), so the constant is the same.  A
  map-dependent one would show as a mapped slab missing Berreman at ANY
  incidence (the shipped readings: 3.9e-2 Jones with `+1j`, `R + T = 3.69 /
  11.94` with `+/-1`).

### 1.7 The gate that pins both: the Berreman 4x4 oracle for a uniform OOP slab under a map

A uniform out-of-plane slab passed as a uniform layer of a MAPPED stack is,
physically, the plain slab (the map only moves the solver's resolution), so
`berreman_jones_1d` is its exact R, T, reflection AND transmission Jones at
any incidence.  Inside the solver every region -- the slab and both
half-spaces -- carries the map's `chi_t' = g / sg`, `chi33' = 1 / sg`, the
slab's `eps'` has VARYING out-of-plane entries `adj(J) e_t3`, and so every
block of 1.3 is exercised.  A separable stretch has a diagonal `J`; the
sheared transfinite map (four interior vertices moved) has a non-diagonal `J`
-- the one family that separates `J^-1 X` from `X J^-1` and `J^-1 t` from
`t`.  Read on R, T AND both Jones matrices (the dispersion is
transpose-blind, and R / T are blind to the rotation sign at normal
incidence), with each gauge constant flipped as the fail-before.


---

## 2. As built

| piece | where | what |
|---|---|---|
| the congruence's out-of-plane entries | `twod_staggered._stag_map_eff_tensor(..., oop=True)` | `e'_a3 = adj(J)_ak e_k3`, `e'_3b = e_3k adj(J)_bk` (keys `e13 e23 e31 e32`); additive keyword, the default call is Phase D's function byte for byte |
| the composite slant weights | `_stag_map_eff_tensor(..., tvec=)` -> `_stag_map_slant_weights` | `tau = adj(J) t / sg` (`t1 t2`), `kappa = chi_t tau` (`k1 k2`), `t` the INTERNAL rotated shear |
| the node weights | `_stag_map_weights(..., oop=, tvec=)` -> `_stag_map_weights_tensor(**e1kw)` | both rules (tensor + Duffy corner points) through the ONE congruence |
| the permeability operators | `Granet2DTransverseE._chi_R / _chi_Gw / _chi_Ktz` | FACTORED OUT of `_assemble` (Eq. 24 / 20 / 21), called by both the in-plane pencil and the out-of-plane generator: one implementation |
| the generator with permeability blocks | `Granet2DTransverseE._assemble_oop_general` | section 1.3: `B = blkdiag(-R, Ggram2, Ggram1)`, `Ktz` in the E rows, `G3 = Gw^-1 Gw_chi c3` in the G rows, the slant through `kappa` (E rows, against the strong `c3`) and `tau` (G rows, `'dL'` derivative on the continuous test); sets `_bgen_hermitian` |
| the dispatch | `_assemble_oop` | `general = cmap is not None or magnetic`; otherwise the shipped arithmetic verbatim (unmapped nonmagnetic vertical or slanted cells: gate E1-1); the mapped OOP weights get the rotation-gauge sign through `_stag_scale_weight` |
| the region eig | `_region_modes_oop` | the whitening factors of the FLUX split read from the two G rows of `B` (the plain Grams; the same matrices on the shipped branch); a non-Hermitian `chi_t` (lossy `mu`) takes `sla.eig(A, B)` |
| the parity accelerator | `_stag_parity_gauge` | refuses a mapped or magnetic solver (its sector Cholesky would read a non-Hermitian `chi_t` as Hermitian; the reduction for symmetric maps is a separate plan item) |
| the solver's refusals | `Granet2DTransverseE.__init__ / _init_map` | lifted: OOP `eps` under a map, `slant` under a map, `mu_cell` with an OOP `eps`, `slant` with `mu_cell`; kept: an OOP `mu` (with or without a map) |
| the stack | `stack2d_pure` | `_require_map_scope` refuses only an OOP `mu`; `add_layer(shapes=..., slant=)` records the slant on the shape layer; a slanted magnetic layer (`_add_magnetic_layer(slant=)`); `_recompile_shapes` refuses only an OOP permeability; the solve loop, the eig dedupe key (cell, slant, map fingerprint), the frame-anchor phase and the far field are unchanged |
| the Jones entry | `pmm_jones_2d_staggered` | `mu_cell` with an OOP `eps_cell` accepted; `shapes=` with `slant=` forwarded to the shape layer |
| tests updated | `test_pmm2d_staggered_curved_{a,c,d}.py`, `test_pmm2d_staggered_magnetic.py` (`test_g7_mu_with_an_out_of_plane_eps_is_accepted_since_phase_e1`, renamed), `test_pmm2d_staggered_slant.py` | the refusal pins of the lifted limits now assert the positive behaviour (the out-of-plane build's precedent); `mu = I` through the new OOP x mu path equals the nonmagnetic OOP solve (bar 1e-10) |

Not touched: `Basis1D`, the quadrature kernel, the corner rule, the node
count, the far projector, the incident projection, the cascade (the
generalized cascade takes the mapped OOP region as it takes any OOP region),
the per-layer / mortar code (Phase E2), no JAX file (Phase E3).

---

## 3. The gates

Every number is from this tree, 2026-10-03.  "Unit test" names the decision
in `tests/unit/test_pmm2d_staggered_curved_e1.py`; the unit-size readings
are in `e_unit_readings.json` (measured through the test module's own
helpers).  Mounts: normal; oblique = 25 deg, phi 0; conical = 25 deg,
phi 40 deg.  Slab cells read `R / T` (orders summed) `/ Jr / Jt` (the largest
complex reflection / transmission Jones difference).

### 3.1 E1-1 -- no map = today's bytes

| claim | measured | bar | fail-before | JSON |
|---|---|---|---|---|
| no map = the bytes of `eae470d9`: Phase D's 122-key set (every dispatch branch, the mortar, absorption, 13 mapped scalar keys) EXTENDED by 78 E1 keys -- the OOP generator on integer / non-uniform walls at oblique Bloch phases, non-reciprocal and lossy OOP cells, the parity reduction on AND off, slanted scalar and tensor cells, slanted Jones at conical incidence, a slanted uniform layer, a two-layer slanted stack with the transmitted per-order amplitudes (the frame-anchor phase), a lossy OOP mixed stack with `layer_absorption`, and the Phase D MAPPED tensor and MAPPED magnetic solves (the permeability blocks were refactored) | **200 / 200** SHA-256 identical | equality | the identity map through the mapped generator: 42 of 99 operator hashes differ (the rest are `None` attributes), `Agen` 3.4e-15 .. 1.1e-14 relative | `e1_bytes_{pre,post}.json`, `e1_compare.json` |
| no E1 kernel without a map / mu | `_assemble_oop_general`, `_stag_map_slant_weights`, `_stag_scale_weight` booby-trapped; an unmapped OOP, a slanted, an in-plane magnetic and a MAPPED block-form tensor solve run | no trap fires | -- | `test_e1_1_shipped_dispatch_never_reaches_the_e1_code` |
| the existing suites | 261 / 261 (OOP, slant, magnetic, block eig, curved A-D, verifiers A / B) after the refusal pins were turned positive; full sweep in section 8 | green | -- | `suite1_win.txt`, `suite_win.txt` |

### 3.2 E1-2 -- the identity map through the mapped generator IS the shipped generator

`e2_identity_M{5,7}.json`: the re-entrant L cell (tilted LC), a
non-reciprocal pillar on NON-UNIFORM walls, a slanted scalar pillar (0.25,
-0.1) and the slanted L cell, each under an identity `TransfiniteMap` on the
same walls, at the Bloch phases of the three mounts:

| quantity | M = 5 | M = 7 | bar |
|---|---|---|---|
| `Agen`, `Bgen` relative | <= 1.3e-14, <= 3.1e-15 | <= 2.6e-14, <= 7.3e-15 | 1e-11 |
| the full eigenvalue set / the forward set | <= 3.4e-13 / 1.8e-13 | <= 8.9e-13 | 1e-11 |
| the modal fields `W`, `V` (phase-aligned, 288 non-degenerate modes) | <= 3.9e-10 / 3.0e-9 (eigenvector sensitivity to the 1e-14 operator change across close eigenvalues) | | recorded |
| R / T / Jr / Jt, full solve, normal | <= 1.9e-14 | <= 8.1e-14 | 1e-11 |
| R / T / Jr / Jt, full solve, oblique / conical | 8.6e-9 .. 2.1e-7 -- equal in size to the SCALAR in-plane identity-map difference (2.6e-8 .. 1.5e-7): the incident treatment (Phase C F-C3: the mapped path's L2 projection vs the unmapped least squares), not the generator | | recorded |
| fail-before: one interior vertex moved by 1e-6 p | `Agen` moves 1.9e-6 | > 1e-9 | |

**STOP condition 1 is not met**: the identity map reproduces the shipped
generator to 1.3e-14, under the 1e-11 bar by three decades, vertical AND
slanted (the slanted rectangular pillar under the identity transfinite map
IS the shipped slant solve -- E1-6's first gate).  Unit test
`test_e1_2_identity_map_is_the_shipped_out_of_plane_generator[None / slant]`
(`e_unit_readings.json`: A 1.26e-14, eigenvalues 4.8e-14 / 5.0e-14, full
solve <= 1.9e-14, moved vertex 1.9e-6).

### 3.3 E1-3 -- a uniform OOP slab under a map is the Berreman slab

Slab fixture of the shipped OOP suite (period 0.9, depth 0.35, air / 1.5).
Tensors: `oop` (tilt 35, azimuth 25), `nonrec` (Hermitian, `e13 != e31`),
`lossy`, `lnonrec` (lossy AND non-reciprocal); `lc` the IN-PLANE reference
film under the same map.  `e3_slab_<map>_<tensor>_<angle>.json`, R/T / Jr / Jt:

| map, tensor, mount | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| shear (3 x 3, non-diagonal J), nonrec, normal | 3.3e-14 / 2.3e-15 / 2.9e-14 | 1.0e-13 / 4.5e-15 / 4.5e-14 | 6.5e-14 / 6.0e-15 / 3.3e-14 | 9.1e-14 / 4.9e-14 / 7.5e-14 | |
| shear, nonrec, oblique | 4.2e-7 / 1.5e-7 / 5.4e-7 | 1.6e-9 / 5.7e-10 / 2.1e-9 | 3.8e-12 / 1.4e-12 / 5.2e-12 | **1.2e-13 / 8.6e-15 / 5.8e-14** | |
| shear, nonrec, conical | 7.0e-8 / 3.9e-8 / 1.2e-7 | 1.9e-10 / 9.2e-11 / 3.1e-10 | 3.6e-13 / 1.6e-13 / 5.3e-13 | **8.2e-14 / 3.4e-14 / 7.1e-14** | |
| shear, lnonrec, conical | 6.0e-8 / 3.8e-8 / 1.2e-7 (lossy) | 1.7e-10 / 9.0e-11 / 3.0e-10 | 2.5e-13 / 1.5e-13 / 5.0e-13 | 8.2e-14 / 3.1e-14 / 3.8e-14 | |
| s05g3 (sine stretch 0.05 p, 3 x 3), nonrec, normal | 3.9e-7 / 1.1e-7 / 1.4e-6 | 1.8e-9 / 2.5e-9 / 3.1e-8 | 1.2e-11 / 2.3e-11 / 2.7e-10 | 2.1e-13 / 2.3e-13 / 2.6e-12 | **9.8e-14 / 2.0e-14 / 1.0e-13** |
| s05g3, nonrec, oblique | 1.7e-5 / 6.6e-6 / 2.2e-5 | 7.5e-7 / 2.9e-7 / 9.9e-7 | 1.6e-8 / 6.2e-9 / 2.2e-8 | 4.6e-10 / 1.8e-10 / 6.3e-10 | **1.0e-11 / 4.0e-12 / 1.4e-11** |
| s05g3, nonrec, conical | 6.7e-6 / 3.7e-6 / 9.0e-6 | 2.1e-7 / 1.5e-7 / 3.4e-7 | 3.3e-9 / 2.5e-9 / 5.9e-9 | 1.0e-10 / 6.7e-11 / 1.5e-10 | **1.8e-12 / 1.2e-12 / 2.8e-12** |
| s05g3, **lc (in-plane reference)**, oblique | 3.8e-5 / 1.4e-5 / 4.8e-5 | 1.7e-6 / 6.1e-7 / 2.1e-6 | 3.7e-8 / 1.3e-8 / 4.6e-8 | 1.1e-9 / 3.9e-10 / 1.3e-9 | 2.4e-11 / 8.7e-12 / 3.0e-11 |
| s15g3 (0.15 p: 33:1), nonrec, normal | 2.9e-6 / 9.8e-7 / 1.2e-5 | 1.0e-7 / 5.9e-8 / 7.2e-7 | 1.3e-10 / 1.9e-10 / 2.3e-9 | 3.3e-12 / 5.2e-12 / 6.3e-11 | **1.8e-13 / 1.2e-13 / 2.3e-13** |
| s15g3, nonrec, oblique | 1.6e-4 / 6.2e-5 / 2.1e-4 | 1.4e-5 / 5.0e-6 / 1.7e-5 | 5.7e-7 / 2.2e-7 / 7.6e-7 | 2.5e-8 / 9.7e-9 / 3.4e-8 | 1.1e-9 / 4.2e-10 / 1.5e-9 |
| s15g3, **lc**, oblique | 3.3e-4 / 1.2e-4 / 4.3e-4 | 2.9e-5 / 1.1e-5 / 3.7e-5 | 1.3e-6 / 4.7e-7 / 1.6e-6 | 5.8e-8 / 2.1e-8 / 7.2e-8 | 2.5e-9 / 9.1e-10 / 3.2e-9 |
| s15g3, nonrec, conical | 4.5e-5 / 3.1e-5 / 8.0e-5 | 4.1e-6 / 2.5e-6 / 5.8e-6 | 1.3e-7 / 8.6e-8 / 2.0e-7 | 5.2e-9 / 3.3e-9 / 7.4e-9 | **2.1e-10 / 1.4e-10 / 3.0e-10** |
| none (shipped), nonrec, conical | 2.8e-7 / 1.5e-7 / 4.4e-7 | 7.5e-10 / 5.6e-10 / 1.4e-9 | 1.7e-12 / 1.2e-12 / 2.9e-12 | 2.9e-14 / 2.2e-14 / 4.8e-14 | 1.9e-13 / 8.2e-14 / 1.2e-13 |

(The same ladders for `oop`, `lossy`, `lnonrec` and for the 2 x 2 `s05` /
`s15` maps are in their JSON; the four tensors agree with each other to a
factor 1.3 at every rung.)

**STOP condition 2 is not met.**  With BOTH gauge constants at their shipped
values the out-of-plane slab under a stretch reaches 1e-9 of Berreman by
`M = 8` on R, T and both Jones -- at `a = 0.05 p` at every mount (1.4e-11
oblique, the worst), at `a = 0.15 p` at normal and conical incidence
(2.3e-13, 3.0e-10) -- and under the sheared map at every mount by `M = 6`.
The one configuration that stays above 1e-9 at `M = 8`, the 33:1 stretch at
oblique incidence (1.5e-9), is the map's RESOLUTION, not a gauge: the
in-plane LC film under the same map at the same mount reads 3.2e-9 (the
out-of-plane slab is 2x BETTER), both fall 1.3 decades per rung with no
floor, and a wrong gauge would sit flat at 1e-3 .. 0.4 (below).  A
map-dependent gauge constant is therefore excluded on three counts: the
sheared map (non-diagonal `J`, the only family that could separate a
`J`-dependent sign) is round-off at oblique and conical incidence; the
out-of-plane slab tracks the in-plane film rung for rung under every map;
and the flipped constants miss by the SAME amount mapped and unmapped:

| arm (`e3_arms_*_M6.json`, nonrec) | none, normal | none, conical | shear, normal | shear, oblique | shear, conical | s15, oblique |
|---|---|---|---|---|---|---|
| correct | 6.8e-14 / 2.5e-14 / 3.6e-14 | 1.7e-12 / 1.2e-12 / 2.9e-12 | 6.5e-14 / 6.0e-15 / 3.3e-14 | 3.8e-12 / 1.4e-12 / 5.2e-12 | 3.6e-13 / 1.6e-13 / 5.3e-13 | 3.5e-5 / 1.1e-5 / 4.2e-5 |
| `_OOP_ROT_SIGN = +1` | 6.8e-14 (blind) | 1.7e-3 / 4.5e-3 / **0.12** | 7.4e-14 (blind) | 2.0e-3 / 5.7e-3 / **0.16** | 1.7e-3 / 4.5e-3 / **0.12** | 2.0e-3 / 5.7e-3 / 0.16 |
| `_OOP_H_GAUGE = +1j` | 6.0e-14 / **1.4e-2** / **0.40** | 1.3e-5 / 8.2e-3 / 0.21 | 4.3e-14 / **1.4e-2** / **0.40** | 9.7e-6 / 7.1e-3 / 0.22 | 1.3e-5 / 8.2e-3 / 0.21 | 1.4e-3 / 7.1e-3 / 0.22 |
| `_OOP_H_GAUGE = +1` | 0.15 / 0.23 / 0.20, closure **0.10** | 6.1e-2 / 0.12 / 0.10 | 0.15 / 0.23 / 0.20, closure 0.10 | 6.0e-2 / 0.13 / 0.11 | 6.1e-2 / 0.12 / 0.10 | 6.0e-2 / 0.13 / 0.11 |

Read with the memory's warnings in hand: the rotation sign is invisible at
normal incidence (the rotation maps the normal mount onto itself) and the
`+1j` H gauge is invisible to R / T at normal incidence (4e-14) while its
Jones is wrong by 1.4e-2 / 0.40 -- only the Jones matrices pin it there.
The shipped readings (rotation `+1` ~1e-3 on R, the conjugate H gauge
3.9e-2 on Jones, the lossless trap broken on `+/-1`) are reproduced in kind;
the transmission Jones is the most sensitive reading of all three defects.

Unit tests: `test_e1_3_oop_slab_under_a_sheared_map_is_the_berreman_slab`
(normal `M = 4` <= 1e-11, measured 3.3e-14; conical `M = 5` <= 3e-9,
measured 3.1e-10), `test_e1_3_both_gauge_constants_are_pinned_under_a_map`
(rotation `+1` conical `M = 5` Jt 0.12 >= 1e-3, normal 3.4e-14 <= 1e-11;
`+1j` Jr 1.4e-2 >= 1e-3 with R / T 3.6e-14 <= 1e-11; `+1` closure 0.10 >=
1e-2), `test_e1_3_oop_slab_under_a_stretch_converges_like_the_inplane_film`
(2 x 2 `a = 0.15 p`, normal `M = 5` 8.8e-7 <= 2e-6, the `M = 4 -> 5` drop
4.0 decades >= 2).

### 3.4 E1-3m -- an out-of-plane eps WITH a material mu (the magnetic build's wall)

The shipped `berreman_jones_1d` takes no permeability, so an independent
(eps, mu) 4x4 transfer-matrix oracle was written
(`build_e1/_mu_oracle.py`; the test file carries the same 40 lines).  It is
validated before use (`e3m_oracle.json`): against `berreman_jones_1d` at
`mu = I` on three out-of-plane tensors at three mounts, R / T / Jr / Jt
<= 3.0e-15; on an isotropic (eps, mu) slab, lossless and lossy, against the
analytic Airy formula <= 5.6e-16.  Then the slab of 3.3 with `nonrec` and
three permeabilities -- `aniso` (real symmetric), `gyro` (Hermitian,
non-reciprocal, `m12 = -m21 = 0.3i`), `lossy` (non-Hermitian: the QZ branch
of `_region_modes_oop`) -- R / T / Jr / Jt (`e3m_*.json`):

| map, mu, mount | M = 4 | 6 | top |
|---|---|---|---|
| none, gyro, conical | 6.4e-7 / 9.0e-7 / 1.8e-6 | 6.6e-12 / 6.2e-12 / 1.3e-11 | M = 7: 3.8e-14 / 2.8e-14 / 6.0e-14 |
| none, lossy, conical | 4.5e-7 / 5.0e-7 / 1.0e-6 | 3.2e-12 / 3.3e-12 / 7.4e-12 | M = 7: 1.5e-14 / 2.0e-14 / 2.8e-14 |
| shear, gyro, conical | 4.3e-8 / 1.4e-7 / 2.9e-7 | 1.2e-13 / 4.3e-13 / 8.5e-13 | M = 7: 1.4e-13 / 4.6e-14 / 6.4e-14 |
| shear, lossy, oblique | 5.2e-7 / 3.4e-7 / 5.8e-7 | 4.1e-12 / 3.1e-12 / 5.1e-12 | M = 7: 1.4e-14 / 3.1e-14 / 1.6e-14 |
| s05g3, gyro, oblique | 4.5e-5 / 3.0e-5 / 6.3e-5 | 4.4e-8 / 2.8e-8 / 6.1e-8 | M = 8: 2.9e-11 / 1.9e-11 / 4.0e-11 |
| c3 (circle), gyro, normal | 1.1e-6 / 4.3e-6 / 1.0e-5 | 5.2e-11 / 2.6e-10 / 5.8e-10 | M = 7: 3.7e-13 / 1.9e-12 / 4.4e-12 |
| c3, lossy, conical | 7.0e-5 / 1.0e-4 / 1.9e-4 | 4.8e-7 / 5.4e-7 / 9.3e-7 | M = 7: 1.9e-8 / 2.2e-8 / 3.7e-8 |

All 36 ladders (4 maps x 3 mu x 3 mounts) converge spectrally to round-off
or to the map's resolution; the lossy arm's own absorption (closure 0.20) is
reproduced to the same digits.  The material `chi_t` composes with the map's
(`J^T mu_t^-1 J / sg`) through the same node weights.  Unit tests
`test_e1_3m_the_mu_oracle_is_berreman_at_mu_one` (bar 1e-12) and
`test_e1_3m_oop_eps_with_mu_matches_the_eps_mu_oracle[gyro-None / lossy-None
/ gyro-shear]` (unmapped `M = 6` 1.25e-11 / 7.4e-12 <= 1e-9; shear `M = 5`
6.1e-10 <= 1e-8).

### 3.5 E1-4 -- an OOP film under the CIRCLE map (singular vertices + the 4q^2 generator)

`e3_slab_c{3,5}_*.json` (circle radius 0.3 p), nonrec, R/T / Jr / Jt:

| map, mount | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| c3, normal | 1.6e-6 / 4.3e-7 / 2.3e-6 | 1.4e-8 / 2.5e-9 / 2.2e-8 | 2.3e-11 / 1.3e-11 / 1.3e-10 | 1.6e-13 / 6.4e-14 / 6.6e-13 | 8.9e-14 / 2.3e-14 / 1.9e-13 |
| c3, oblique | 2.4e-4 / 6.3e-5 / 2.6e-4 | 2.5e-5 / 7.7e-6 / 2.9e-5 | 1.2e-6 / 4.5e-7 / 1.6e-6 | 6.3e-8 / 2.4e-8 / 8.5e-8 | 2.6e-9 / 1.0e-9 / 3.6e-9 |
| c3, conical | 8.4e-5 / 3.7e-5 / 1.5e-4 | 1.2e-5 / 4.1e-6 / 1.7e-5 | 5.8e-7 / 2.2e-7 / 8.5e-7 | 2.3e-8 / 9.1e-9 / 3.5e-8 | 1.1e-9 / 4.3e-10 / 1.6e-9 |
| c5, normal | 2.8e-9 / 8.0e-10 / 7.1e-9 | 2.8e-12 / 5.8e-13 / 7.3e-12 | 1.4e-13 / 1.3e-14 / 1.7e-13 | | |
| c5, oblique | 3.4e-7 / 1.2e-7 / 4.0e-7 | 3.6e-9 / 1.3e-9 / 4.2e-9 | 2.8e-11 / 1.0e-11 / 3.4e-11 | | |
| c5, conical | 1.3e-7 / 6.1e-8 / 1.9e-7 | 1.7e-9 / 7.8e-10 / 2.7e-9 | 1.3e-11 / 5.8e-12 / 2.0e-11 | | |

The exactness ladder: spectral to round-off at normal incidence on both
topologies; at oblique / conical incidence on the 3 x 3 map spectral and
four decades behind (Phase B's F-B6: the Bloch phase composed with the map),
on the 5 x 5 map round-off-class by `M = 6`.  The four singular vertices and
the 4 q^2 generator together add nothing beyond Phase D's in-plane tensor
film (D3: 7.7e-11 at c3 `M = 6`, normal).  Unit test
`test_e1_4_oop_film_under_the_circle_map_is_spectral` (`M = 6` 1.3e-10 <=
1e-9; drop `M = 4 -> 6` 4.2 decades >= 3).

### 3.6 E1-5 -- an OOP tensor circular pillar

The OOP30 disk (director tilted 30 deg OUT of the plane, azimuth 30 deg; r
0.36, period 1.2, depth 0.5, air / 1.45), the 36-entry R / T vector of the
nine orders, against the c3 `M = 10` rung (`e5_summary.json`):

| family | knob | distance | rung change | closure |
|---|---|---|---|---|
| c3 | M = 4 / 5 / 6 / 7 / 8 / 9 | 4.2e-2 / 8.8e-3 / 4.4e-3 / 3.2e-4 / 1.4e-4 / 1.0e-5 | 3.3e-2 / 4.4e-3 / 4.1e-3 / 1.8e-4 / 1.3e-4 / 1.0e-5 | 2.7e-4 .. 1.4e-7 (M = 9), 7.8e-8 (M = 10) |
| c5 (independent topology) | M = 4 / 5 / 6 / **7** | 3.2e-4 / 4.5e-5 / 1.6e-5 / **4.0e-6** | 2.7e-4 / 2.9e-5 / 1.2e-5 | 1.8e-5 / 2.9e-7 / 5.5e-9 / 3.9e-11 |
| tensor RCWA, EXACT disk form factor (Laurent, every component) | N = 7 / 9 / 13 / 17 / 21 / 25 / 29 orders per axis | 1.0e-2 / 7.2e-3 / 5.2e-3 / 4.0e-3 / 3.3e-3 / 2.7e-3 / 2.3e-3 | distance x N = 0.065 .. 0.071 (1 / N within +-5 %) | <= 1.3e-13 |
| same, Richardson in 1 / N | pairs (21, 25), (25, 29) | 3.1e-4, **7.0e-5** | | |
| shipped OOP solver, staircase (no map) | 4 steps, M = 4 / 6 / 8 / 10 | 4.5e-2 / 1.8e-2 / 1.9e-2 / 1.9e-2 | (converged: 1.9e-2) | 6.6e-11 (M = 10) |
| same | 8 steps, M = 5 / 6 / 7 | 1.72e-2 / 1.73e-2 / 1.73e-2 | (converged) | 1.5e-10 |
| same | 16 steps, M = 3 / 4 | 4.4e-3 / 3.7e-3 | 7e-4 (pre-asymptotic) | 5.0e-6 |

The two independent topologies agree to 4.0e-6 (c5 `M = 7`, dof 3600, vs c3
`M = 10`, dof 2916), at the level of c3's own last rung; the exact-disk
RCWA's ENVELOPE falls like 1 / N toward that answer and its 1 / N Richardson
pairs reach 7.0e-5 -- the RCWA's 1/N floor (its local rate wanders, as
Phase D's D4 recorded; a 1e-4-class reference); the staircases converge in M
to different devices, strictly decreasing toward the curved answer (1.9e-2,
1.7e-2, 3.7e-3).  `fff_nv` was NOT the exact-disk RCWA used, as the brief
suggested: its validated-scope gate refuses a disk (non-separable geometry)
and it builds its factorization from the pixel cell, which would put the
O(1/S) staircase back in; the exact form factor enters through Phase D's
patched Laurent builder instead (`e5_pillar.exact_disk`).  Unit test
`test_e1_5_oop_pillar_lands_on_its_converged_answer` (c5 `M = 4` 3.2e-4 <=
1e-3; fail-before the 4-step staircase at `M = 4`, 4.5e-2 >= 1e-2).

### 3.7 E1-6 -- slant under a map

**(a) The slanted RECTANGULAR pillar under the identity transfinite map IS
the shipped slant solve**: 3.2, rows "slant" and "slantoop" (operators
1.3e-14, eigenvalues <= 9.7e-14, full solve <= 2.9e-14 at normal incidence).

**(b) The composite frame on a uniform slab (the null test).**  A slanted
uniform slab is physically the slab; under the sheared map the composite
frame has `tau = J^-1 t` varying across every cell
(`e3_slab_shear_*_slant+0.20-0.10.json`, nonrec, R/T / Jr / Jt):

| mount | M = 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| normal | 5.8e-8 / 1.1e-15 / 1.6e-14 | 7.1e-11 / 4.5e-15 / 2.6e-14 | 5.3e-12 / 1.0e-14 / 3.8e-14 | 1.3e-13 / 4.9e-14 / 6.5e-14 |
| oblique | 3.9e-7 / 1.4e-7 / 7.6e-7 | 8.0e-9 / 1.6e-9 / 6.2e-8 | 1.0e-10 / 4.1e-11 / 5.7e-10 | 8.5e-12 / 1.9e-12 / 8.0e-11 |
| conical | 2.7e-7 / 4.0e-8 / 3.5e-7 | 1.1e-9 / 4.3e-10 / 1.8e-8 | 9.8e-11 / 1.5e-11 / 3.0e-10 | 2.5e-12 / 3.3e-13 / 1.5e-11 |

(s05g3 slanted: 1.6e-13 at `M = 8` normal, 1.5e-11 oblique; the unmapped
slanted slab, the shipped reference, 1.1e-13.)  The fail-befores
(`e3_arms_shear_nonrec_*_slant+0.20-0.10_M6.json`): the shear NOT composed
with the map (`tau = t`, the shipped constants on a mapped cell) 7.7e-4
normal / 2.9e-3 oblique; the slant blocks dropped invisible at normal
(2.4e-14 -- there `tau . G'_t = t . G_t` is constant and its derivative
vanishes in the exact solution) and 0.14 (Jt) at oblique; `kappa` without
`chi_t` 4.8e-5 (Jt) normal, 1.8e-3 oblique.  Unit tests
`test_e1_6_slanted_uniform_slab_under_a_map_is_the_slab` (normal `M = 5`
7.1e-11 <= 1e-9; oblique 6.2e-8 <= 1e-6) and
`test_e1_6_the_shear_must_be_composed_with_the_map` (`tau = t` 7.7e-4 >=
1e-4; slant blocks dropped at oblique 0.14 >= 1e-3).

**(c) The slanted circular pillar** (eps 4, r 0.36, public slant (0.2, 0);
`e6_summary.json`), the 36-vector against the c3 `M = 10` rung:

| family | knob | distance | rung change / closure |
|---|---|---|---|
| c3 composite | M = 4 / 5 / 6 / 7 / 8 / 9 | 7.1e-2 / 1.4e-2 / 9.0e-3 / 8.7e-4 / 4.8e-4 / 3.8e-5 | 6.9e-2 .. 3.8e-5; closure 2.2e-4 .. 9.0e-7 |
| c5 composite | M = 4 / 5 / 6 / **7** | 6.3e-4 / 7.0e-5 / 3.9e-5 / **2.3e-5** | 5.7e-4 / 4.5e-5 / 2.3e-5; closure 1.6e-4 .. 2.7e-7 |
| shipped slant solver, in-plane staircase | 4 steps, M = 4 / 6 / 8 / 10 | 6.6e-2 / 7.2e-2 / 7.6e-2 / 7.6e-2 | converged in M |
| same | 8 steps, M = 5 / 6 / 7 | 5.3e-2 / 5.3e-2 / 5.3e-2 | converged |
| same | 16 steps, M = 3 / 4 | 9.4e-3 / 9.1e-3 | |
| exact-disk RCWA z-staircase (`RCWAStack`, `shapes=` disks at c + slant z_mid) | N_z = 4 / 8 / 16 at N = 25 orders | 5.4e-3 / 4.8e-3 / 4.7e-3 | closure <= 1e-14 |
| same, Richardson in 1 / N (19, 25) per N_z | N_z = 4 / 8 / 16 | 1.3e-3 / 4.3e-4 / 2.2e-4 | |
| same, then Richardson in 1 / N_z^2 | (4, 8) / (8, 16) | **2.2e-4 / 2.6e-4** | step 8 -> 16: 2.7e-4 |

The two limits agree: the shipped slant solver's in-plane staircases fall
toward the curved composite answer (7.6e-2, 5.3e-2, 9.1e-3; each converged
in M to its own device), and the z-staircase of exact disks, extrapolated in
both its order count and its slice count, lands within 2.2e-4 of it -- the
RCWA's own 1 / N class (the z-limit's last step is 2.7e-4).  The z-staircase
of the CURVED solve itself was not measurable: each slice's circle sits at a
different centre, and one stack-wide map cannot carry overlapping circles at
different centres (the merge refuses crossing outlines) -- that needs
per-layer maps (Phase E2); the exact-disk RCWA z-staircase stands in for it.
The opposite slant (-0.2, 0) at c3 `M = 8` is 2.5e-2 from the answer and
is EXACTLY the x-mirror of the +0.2 rung: R(m, n; -t) = R(-m, n; +t) to
1.7e-13 -- the slant's sign under the composite frame is a physical identity,
not a fitted convention.  Unit test
`test_e1_6_slanted_circle_lands_on_its_converged_answer` (c5 `M = 4` 6.3e-4
<= 2e-3; fail-befores the 4-step slant staircase 6.6e-2 >= 2e-2 and the
mirrored rung 2.5e-2 >= 1e-2).

### 3.8 E1-7 -- slant x OOP x curved on one cell

The OOP30 disk slanted (0.2, 0) under the c3 circle map, conical (25, 40
deg), channel (incidence -> reflected order (-1, 0)) against its reversal
(35.27 deg, -28.06 deg); the singular values of the power-normalized Jones
block (`e7_summary.json`, `e6_summary.json` reciprocity):

| geometry | quantity | M = 5 | M = 7 |
|---|---|---|---|
| slanted OOP curved | reciprocal OOP30: fwd vs rev | 1.3e-4 | 1.5e-6 |
| | non-reciprocal NONREC30 (`e13 = conj(e31)`, +0.25i): fwd vs rev | **6.2e-3** | **5.5e-3** (flat: physics) |
| | NONREC30 fwd vs NONREC30^T rev (the Onsager-Casimir partner) | 8.6e-5 | **1.4e-6** |
| | wrong pairing (the reversed run's specular channel) | 8.9e-2 | 9.2e-2 |
| curved, vertical | OOP30 / NONREC30 / transposed | | 3.0e-6 / 2.2e-3 / 2.8e-6 |
| shipped OOP solver, 4-step staircase (no map) | OOP30 / NONREC30 / transposed | 3.2e-5 / 2.5e-2 / 4.0e-5 | 8.5e-8 / 2.2e-2 / 6.9e-8 |

Order (0, -1) on the slanted cell: transposed partner 2.5e-4 / 3.7e-5 /
5.2e-6 at `M = 5 / 6 / 7`, the non-reciprocal residual 2.5e-3 .. 7.0e-3.
Closure (lossless) of the slanted OOP cell at conical incidence and its
reversals: 1.8e-3 / 9.1e-4 / 1.4e-4 at `M = 5 / 6 / 7`.  The
non-reciprocity "of the measured sign": the non-reciprocal medium breaks
the identity at the physics level, and transposing the tensor restores it
spectrally -- an identity that can only hold if the out-of-plane entries
enter with the right index order AND the right rotation-gauge sign under the
composite frame (the transpose is the field-level discriminator the memory
asks for; dispersion is transpose-blind).  Unit test
`test_e1_7_slant_oop_curved_is_reciprocal_and_a_nonreciprocal_one_obeys_its_transpose`
(c3 `M = 5`: reciprocal 1.3e-4 and transposed partner 8.6e-5 <= 1e-3;
non-reciprocal 6.2e-3 >= 2e-3; wrong pairing 8.9e-2 >= 1e-2).

### 3.9 E1-8 -- oblique and conical incidence on E1-5 and E1-6

| quantity (`e5_summary.json`, `e6_summary.json`) | OOP30 pillar (E1-5) | slanted eps-4 pillar (E1-6) |
|---|---|---|
| c3 rung change, 25 / 0, M = 5 -> 8 (E1-6: -> 7) | 1.1e-2, 2.6e-3, 6.3e-4 | 3.3e-2, 8.3e-3 |
| c5 rung change, 25 / 0 | 2.0e-4, 2.2e-5 (M = 4 -> 6) | 8.0e-4 (M = 4 -> 5) |
| the two topologies (c3 top vs c5 top), 25 / 0 and 25 / 40 | 6.7e-5, 2.3e-4 | 5.9e-3, 6.9e-3 (c3 at M = 7 is still 8e-3 per rung) |
| lossless closure, c3 at the top rung, the five angles | 1.1e-5 .. 1.1e-4 (M = 8) | 7.6e-4 .. 1.8e-3 (M = 7) |
| lossless closure, c5 at the top rung | 5.4e-8 .. 3.7e-7 (M = 6) | 1.7e-5 .. 2.5e-4 (M = 5) |
| reciprocity (-1, 0) at 25 / 0, c3 M = 5 .. 8 (E1-6: .. 7) | 8.2e-6, 1.7e-6, 1.8e-7, 1.7e-8 | 3.0e-5, 1.0e-5, 2.5e-6 |
| reciprocity (-1, 0) at 25 / 40, c3 | 1.4e-4, 3.7e-6, 3.0e-6, 1.5e-8 | 8.5e-4, 1.3e-4, 2.9e-5 |
| reciprocity (0, -1) at 25 / 40, c3 | 1.9e-4, 1.4e-5, 3.7e-6, 3.3e-7 | 1.0e-3, 5.8e-5, 6.5e-5 |
| reciprocity on c5 (-1, 0) at 25 / 40 | 8.4e-7, 1.7e-8, 2.6e-9 (M = 4 .. 6) | 5.3e-5, 3.5e-6 (M = 4, 5) |
| wrong pairing | 0.098 .. 0.13 | 0.076 .. 0.13 |
| non-reciprocal control NONREC30 (unslanted), 25 / 40, (-1, 0) / (0, -1), c3 M = 5 .. 7 | 1.3e-3 .. 2.2e-3 / 5.8e-3 .. 6.9e-3 | -- |
| exact-disk RCWA at oblique / conical, N = 17 / 25, to the c3 top rung | 3.6e-3 / 2.4e-3; 3.5e-3 / 2.3e-3 (1 / N) | -- |

Reciprocity holds spectrally (an identity the solver does not impose) and is
broken only by the non-reciprocal medium.  The slanted pillar at oblique
incidence converges more slowly on the 3 x 3 map than the vertical one
(F-B6 -- the tilted plane wave composed with the map -- plus the slant's
lateral walk): budget the 5 x 5 layout at oblique incidence (Phase B's
advice, re-measured).

### 3.10 E1-9 -- the mutation matrix

Slab rows (Berreman; `e3_arms_*_M6.json`, R/T / Jr / Jt, the correct arm
first) and pillar rows (`e9_summary.json`, M = 6: the largest change of the
36-vector / Jr / Jt against the correct arm).  **Bold** = caught (>= two
decades above the correct arm's own error and above the gate's bar).

| defect | identity map, slab (normal / oblique) | shear, slab, normal | shear, slab, oblique | s15, slab, oblique | slanted slab, shear, normal / oblique | OOP30 pillar c3 normal | pillar c3 conical | slanted eps-4 pillar c3 normal | slanted OOP pillar c3 conical |
|---|---|---|---|---|---|---|---|---|---|
| correct | 5.2e-14 / 1.1e-12 | 6.5e-14 / 6.0e-15 / 3.3e-14 | 3.8e-12 / 5.2e-12 | 3.5e-5 / 4.2e-5 | 5.3e-12 / 5.7e-10 | 0 (closure 3.5e-4) | 0 | 0 | 0 |
| no_mu_blocks | 6.9e-14 / 1.1e-12 (blind -- chi = I) | **3.3e-3 / 7.8e-3 / 5.4e-2** (closure 6.6e-8!) | **1.5e-2 / 7.5e-2** | **5.0e-2 / 0.85** | **2.9e-3 / 7.1e-2** | **7.1e-2** | **0.12** | **0.17** | **0.12** |
| chi33_no_sg | 5.9e-15 / 1.1e-12 | 7.3e-14 (blind: c3 = 0) | **5.1e-4 / 2.5e-3** | **4.7e-3 / 2.1e-2** | 1.1e-7 / **2.9e-3** | **2.4e-2** | **6.3e-2** | **7.7e-2** | **6.8e-2** |
| g3_strong | (as chi33_no_sg: the same weight in vacuum) | | | | | | | | |
| tau_unmapped | -- / blind (identity: J = I) | -- | -- | -- | **7.7e-4 / 2.9e-3** | -- | -- | **2.1e-2** | **4.4e-2** |
| no_slant_blocks | blind / **0.14** | -- | -- | -- | 2.4e-14 (blind) / **0.14** | -- | -- | **2.5e-2** | **0.14** |
| kappa_no_chi | blind (identity) | -- | -- | -- | **4.8e-5** / **1.8e-3** | -- | -- | **7.5e-3** | **2.2e-2** |
| rot_flip | blind / **0.16** | blind | **0.16** | **0.16** | blind / **0.46** | **4.8e-2** (R / T; Jones blind) | **0.18** | **2.5e-2** | **0.34** |
| hgauge_plus_i | **1.4e-2** Jr, R / T blind | **0.40** Jt | **0.22** | **0.22** | **0.40** / **0.22** | REFUSED loudly (generalized-interface conditioning error, rcond 3.6e-18) | refused | **0.39** | **0.30** |
| hgauge_one | **closure 0.10** | **0.20**, closure 0.10 | **0.11** | **0.11** | **0.20** | **1.9** (closure 3.9) | **1.1** | **0.76** | **1.1** |

Every defect is caught by at least one gate by two decades or more, and the
blindnesses are the physics: the permeability blocks are invisible under the
identity map (chi = I) and NOT under any stretch or shear -- the brief's
required shape (caught by the stretch gate, not by the identity gate) -- and
their loss keeps the lossless closure at 6.6e-8 while the Jones is wrong by
5e-2 (the lossless trap); `chi33` enters only through the longitudinal curl,
which a uniform slab at normal incidence never excites (Phase D's no_sg_e33
finding, out of plane); the slant blocks' `tau` part vanishes on a uniform
slab at normal incidence; the rotation sign is invisible at normal incidence
on a UNIFORM slab but not on a patterned pillar (the shipped G0 finding);
the conjugate H gauge is R / T-blind on the normal slab and caught by the
Jones -- the gates read both Jones matrices everywhere.

### 3.11 E1-10 -- cost

Region assembly + region eig on the 3 x 3 circle map, best of runs, LOADED
box (upper bounds; only same-run ratios mean anything) (`e10_cost_M{6,8}.json`):

| M (dim) | in-plane mapped (Phase D, LC30, QZ) | OOP mapped (E1) | OOP slanted mapped | OOP unmapped, same walls |
|---|---|---|---|---|
| 6 (450 / 900) | 2.96 s (assembly 0.35), 65 MB | 2.93 s (0.46), 133 MB | 2.97 s (0.50), 140 MB | 2.47 s (0.17), 113 MB |
| 8 (882 / 1764) | 22.8 s (0.82), 218 MB | 18.5 s (1.23), 501 MB | 13.5 s (1.43), 503 MB | 12.4 s (0.67), 433 MB |

Bounds: a mapped OOP region costs 1.2-1.5x the unmapped OOP region on the
same walls (the quadrature assembly, 1.8-2.7x the kron one, is 7-16 % of the
total), 0.6-1.0x Phase D's mapped in-plane region (the whitened standard eig
at 4 q^2 beats the in-plane QZ at 2 q^2, as the shipped OOP build measured),
and 2.0-2.3x its peak memory (the shipped OOP path's ~3x of the in-plane
working set, less the quadrature caches both share).  A slant adds nothing
measurable (six extra quadrature blocks).

---

## 4. Findings

### 4.1 F-E1-1 -- the derivation closed the way the plan expected, with one refinement

The plan's note ("the first-order generator needs mu blocks since a map
makes mu' != 1") is right, and the blocks needed are EXACTLY the in-plane
pencil's three permeability operators, in the same index placement, so they
were factored out and shared (one implementation).  The refinement: the
constitutive law is used in its inverse form (`chi`) on the transverse rows
but in its 33 form (`1 / mu'^{33}`, `-mu'^{3t} / mu'^{33}`) on the
longitudinal one, so no 3 x 3 inverse is ever formed and the shear's `|t|^2`
cancels out of the generator entirely (section 1.2).

### 4.2 F-E1-2 -- the composite slant carries ONE new coupling, and it is not a constant

`Lambda^T Lambda` of the slant x map composite differs from its parents only
by `J^T t`: the shear in `(u, v)`, `tau = J^-1 t`, varies across every
curved cell, so its derivative block cannot reuse the shipped constant
`-(dbt)^H` krons; it is written with the derivative on the continuous test
function, which reduces to them for a constant `tau`.  Laying the shipped
constant blocks on a mapped cell (`tau = t`) is wrong by 7.7e-4 .. 4.4e-2
(E1-9).

### 4.3 F-E1-3 -- both gauge constants are map-independent (measured)

Section 3.3.  The two constants were measured, not derived, in the shipped
build; here the flipped constants miss the sheared- and stretched-map slabs
by the same amounts as the unmapped slab (rotation `+1`: 0.12 / 0.16 on Jt;
`+1j`: 1.4e-2 Jr; `+1`: closure 0.10), and the correct ones reach round-off
under the non-diagonal-J map at oblique and conical incidence.

### 4.4 F-E1-4 -- the out-of-plane x material-mu wall falls with the same blocks

The magnetic build's refusal ("an OOP x mu layer refuses") existed only for
want of the permeability blocks.  With them an out-of-plane eps with ANY
block-form mu -- symmetric, gyrotropic, lossy -- is solved with or without a
map, and an independent (eps, mu) 4x4 oracle (validated against
`berreman_jones_1d` to 3e-15 at mu = I) confirms it to round-off (3.4).  A
LOSSY mu makes `-R` non-Hermitian; the whitening is then not available and
the region eig takes QZ (`_bgen_hermitian`, a structural decision: the
assembled `-R` is Hermitian to round-off or off by `Im mu`).  An OUT-OF-PLANE
mu stays refused (it would need the full 3 x 3 constitutive split of 1.1 with
`mu'^{t3} != 0` in the material itself -- the formulas are written, the
gates are not; recorded as a follow-up).

### 4.5 F-E1-5 -- the parity accelerator is refused under a map or a mu

Its structural test reads `R B R = B`, and its sector Cholesky would silently
read a non-Hermitian `chi_t` as Hermitian (numpy reads the lower triangle);
the reduction for symmetric maps is a separate plan item (4.5's last bullet)
with its own gate.  The shipped unmapped path is unchanged (E1-1).

### 4.6 F-E1-6 -- a slanted uniform slab under a map is spectral, not exact

The sheared map is EXACT for a vertical uniform slab at normal incidence
(3.3e-14 from `M = 4`) but a SLANTED one reads 5.8e-8 / 7.1e-11 / 5.3e-12 /
1.3e-13 at `M = 4 .. 7` (Jones exact): the composite frame's `tau = J^-1 t`
is a rational function of `(u, v)` on a bilinear cell, so the discrete
`tau . G'_t` term is not exactly the constant it is in the continuum.
Spectral, recorded, no defect.

### 4.7 F-E1-7 -- mutating the H gauge on a patterned cell can refuse instead of answer

With `_OOP_H_GAUGE = +1j` the OOP pillar's generalized interface becomes
numerically singular (rcond 3.6e-18) and the shipped conditioning guard
RAISES rather than return a number -- the loud failure the guard was built
for (E1-9's two "refused" cells).

---

## 5. What moved

Nothing shipped: E1-1, 200 / 200 SHA-256 identical against `eae470d9`,
including every out-of-plane, slant, parity and magnetic branch and the
Phase D mapped tensor / magnetic solves.  New behaviour, reachable only where
the library raised before: out-of-plane tensors and slant under a map (and
with shapes), an out-of-plane eps or a slant with a material mu.  The refusal
tests of those limits were turned into positive assertions (section 2).

---

## 6. The coordinator's two messages during this build (recorded for the maintainer)

1. A list of Phase C verifier defects (`VERIFY_PMM2D_CURVED_C_2026_10_03.md`
   V-D1 .. V-D7) arrived addressed to the Phase E2 builder ("fold these into
   Phase E2 ... code YOU own -- the merge and the shape layout") but was
   delivered to this E1 session.  It was applied here first (commit
   `4079b09a`), then -- once the coordinator's amendment made clear that only
   the forward-version tokens of `tests/unit/test_v4_16_0_walker_all_symmetry.py`
   (V-D6) are E1's -- reverted (commit `4cda1e1b`) except V-D6 and a one-line
   fix that lets the verifier's `test_vc2` read Phase D's six-output `_merge`
   (`[:5]`).  The verifier branch IS merged here (`708c1a89`; merging the same
   commit on two branches is conflict-free); `test_vc4` / `test_vc5` stay
   strict xfails until E2 lands V-D1 / V-D2.  If E2 also edits `test_vc2`'s
   unpack line, that one line will need a trivial resolution.
2. For the post-E list (NOT implemented here, as instructed): the Phase C
   verifier found that taking the mapped incident wave as the discrete (0, 0)
   eigenmode pair ("mode-pick") is equal to or better than the shipped L2
   projection everywhere it measured (up to 200x on the circle film); the
   unmapped oblique path has the same window dependence (1.5e-5 at `M = 4` on
   `91d00288`), so changing it moves shipped bytes and needs its own fixture
   sweep.

---

## 7. Not measured

* **A z-staircase of the curved solve itself** (E1-6): needs per-layer maps
  (Phase E2); the exact-disk RCWA z-staircase replaced it (3.7 (c)).
* **An independent full-wave oracle for the OOP pillar** (the plan's 3-D FEM
  runner is a scalar quarter-cell formulation that a tilted director
  breaks); the references are the two topologies, the exact-disk tensor RCWA
  (1 / N, Richardson 7.0e-5) and the staircases.
* **`fff_nv` as the exact-disk RCWA** (refused on a disk by its own scope
  gate; section 3.6).
* **An out-of-plane PERMEABILITY** (still refused; F-E1-4).
* **The parity reduction for symmetric maps** (refused; F-E1-5).
* **The probe READINGS on a second build.**  The unit gates and the curved /
  OOP / slant / magnetic suites ran on the WSL build (section 8); the
  ladders here are one build's.
* **Idle wall times** (the box was shared with four sibling agents).

---

## 8. Reproduction and test tails

```
cd /c/tmp/lum_curved_e1/validation/probe_pmm2d_curved/build_e1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e1
# E1-1 (PRE tree = git archive eae470d9 in C:/tmp/curved_pre_e1)
(cd /c/tmp/curved_pre_e1 && PYTHONPATH=C:/tmp/curved_pre_e1 python C:/tmp/lum_curved_e1/validation/probe_pmm2d_curved/build_e1/e1_bytes.py C:/tmp/curved_pre_e1 pre)
python e1_bytes.py C:/tmp/lum_curved_e1 post idmap ; python e1_compare.py
python runjobs.py jobs_e3.txt 5 ; python runjobs.py jobs_e3bc.txt 6 ; python runjobs.py jobs_e3d.txt 6   # E1-2 / E1-3 / E1-4 / E1-6 slab / arms
python e3m_mu.py oracle ; python runjobs.py jobs_e3m.txt 6                                             # E1-3m
python runjobs.py jobs_e5A.txt 4 ; python runjobs.py jobs_e6B.txt 4 ; python runjobs.py jobs_e6x.txt 1   # E1-5 / E1-6
python runjobs.py jobs_e8C.txt 5 ; python runjobs.py jobs_e7.txt 6                                    # E1-7 / E1-8
python e5_pillar.py summary ; python e6_slant.py summary ; python e7_onsager.py summary
python runjobs.py jobs_e9.txt 5 ; python e9_mutations.py summary                                      # E1-9 / E1-10
python e_unit_readings.py
cd /c/tmp/lum_curved_e1 && python -m pytest tests/unit/test_pmm2d_staggered_curved_e1.py --capture=sys -p no:randomly
```

Test tails, both builds:

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), serial:
  `tests/unit/test_pmm2d_staggered_curved_e1.py` -> `18 passed in 41.43s`;
  slowest 7.7 s (E1-7), 4.0 s (E1-6 circle), 3.8 s (E1-4), 3.7 s (E1-5);
  durations spliced into `.test_durations` (18 added)
  (`e1_tests_serial_win.txt`).
* Windows, the existing suites (70 files: every `test_*pmm2d*`,
  `*stack2d*`, `*stagger*`, `*curved*` file incl. Phases A-E1 and the A / B
  / C verifiers' files, census, public API, every walker, doc identifiers,
  dispatcher doc consistency, except budget, history relocation / lint /
  fingerprint tool, kernel consistency, re-exports; `suite_files.txt`),
  `-n 6`: `1665 passed, 8 skipped, 2 xfailed, 96 warnings in 584.17s`
  (`suite_win.txt`).  The 8 skips are the pre-existing premise gates (the
  mortar round-2 LAPACK premise; seven changelog walkers with nothing to
  verify in the `[5.49.0]` block); the 2 xfails are the Phase C verifier's
  strict pins `test_vc4` / `test_vc5` (V-D1 / V-D2, Phase E2's to flip).
  Earlier, right after the library change: the OOP, slant, magnetic, block
  eig, curved A-D and verifier A / B files `261 passed` (`suite1_win.txt`).
  Doc identifiers, history lint / relocation / fingerprint tool and the
  dispatcher doc consistency re-run after this document: `781 passed`.
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS pinned,
  `lumenairy` from `/mnt/c/tmp/lum_curved_e1`): curved E1, D, A, the Phase C
  verifier's file, OOP, slant, magnetic and the `__all__` walker:
  `201 passed, 2 xfailed in 195.79s` (`suite_wsl.txt`).
* `python -m mypy` (the configured strict list): `Success: no issues found
  in 33 source files`.
* WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and `build_e1/`: `All
  checks passed!`
* History fingerprints: `stack2d_pure` re-recorded with the reason;
  `twod_staggered.py` and `shapes2d.py` have no history document.

---

## 9. Addendum -- the Phase D verifier, folded in (2026-10-03)

Merged `verify/pmm2d-curved-d` (`936be59e`; `.test_durations` dict-union);
its four decision tests pass on this tree (section 9.5), including the pin
that `_homog_geom_cache`'s slot 4 is the inverse PLAIN block Gram (4.7e-14)
and 0.40 away from `inv(-R)` under a map.

### 9.1 Wording (VERIFY_D section 10)

D-1 (a transposed gyrotropic film is R / T-blind at NORMAL incidence only;
conical 9.9e-4), D-3 (films under the circle map reach ~1e-13 by `M = 8` at
normal incidence, ~1e-10 conical) and D-4 (the LC30 pillar is fixed to ~4e-6;
the RCWA Richardson value is a 1e-4-class corroboration) applied to
BUILD_D and the CHANGELOG verbatim.  D-2 (the walker's forward version
tokens) was already done on this branch (commit `4079b09a`, kept by
`4cda1e1b`).

### 9.2 `_homog_geom_cache` returns a NamedTuple read by name (F-D3)

`_StagHomogGeom(W0, g2_geo, GW0, SttW0, inv_plain_gram, qq)`: the two
consumers of slot 4 -- the half-spaces' Eq.-25 H partner
(`_homog_region_modes`) and the mapped incident L2 projection
(`_stag_incident_coeffs_mapped`) -- now coerce a plain tuple through
`_stag_homog_geom` (its first six slots) and
read `inv_plain_gram` BY NAME, so the coupling is visible; positional reads
and 6-tuple unpacks (the verifier's test, Phases A / D's patched arms, the
per-layer path) keep working.  Bit identity re-gated on the 200-key E1-1
set: **200 / 200** SHA-256 equal to `eae470d9` (`e1_bytes_postnt.json`).

### 9.3 A non-symmetric mu on a conical film in every magnetic E1 gate

The verifier's mu-transposed-in-chi mutant survived all 26 Phase D ids
because every Phase D magnetic gate used a diagonal mu.  The E1 magnetic
gates are the three `test_e1_3m_oop_eps_with_mu_matches_the_eps_mu_oracle`
ids, ALL at conical incidence (25, 40 deg); two of them --
`[gyro-None]` and `[gyro-shear]` -- carry the NON-SYMMETRIC gyrotropic mu
(`m12 = -m21 = 0.3i`); `[lossy-None]` a symmetric lossy one (the QZ branch).
The mutant, applied to both chi paths (the map's congruence and the
unmapped per-cell inverse), misses the (eps, mu) oracle by 8.4e-3 on R / T
and 0.63 on Jt on BOTH gyro arms, with the lossless closure untouched (1e-11
/ 6e-10: the lossless trap), against 1.25e-11 / 6.1e-10 correct.  Added as
its own decision: `test_e1_3m_a_transposed_mu_is_caught_by_the_non_symmetric_gates`
(bar >= 1e-3 on R / T).  The E1-3m ladders (3.4) carry the gyro mu at all
three mounts on all four maps.

### 9.4 Convergence slows past M = 10 on curved disks (plan 3.3)

The pillar's top and bottom rim carry an edge singularity no in-plane map
removes (plan 3.3 / 3.4; Phase B's scalar disk sits at ~1e-6 per rung at
`M = 11 / 12`; the Phase D verifier measured the LC30 tensor disk's c3
ladder slowing from a decade per rung to ~1.3x per rung past `M = 10`, 3.5e-6
/ 2.8e-6 at `M = 11 / 12`).  The E1 pillars obey the same cap: the OOP30
disk's c3 rung change is 1.0e-5 at 9 -> 10 and the two topologies agree to
4.0e-6, i.e. the E1-5 answer is fixed to the ~1e-5 .. 4e-6 level, not
better; the slanted disk's c3 rung change is 3.8e-5 at 9 -> 10 (c5 vs c3
2.3e-5).  The E1 claims above are stated at that level; the slab, film and
reciprocity gates (no rim inside the measured quantity, or an identity) are
not affected.

### 9.5 Tails after the fold-in

* Windows, `tests/unit/test_pmm2d_staggered_curved_e1.py` serial:
  `19 passed in 70.22s` (`e1_tests_serial_win2.txt`).
* Windows, `-n 6`: the E1 file, Phase D's ids, the merged verifier files
  (A, B, C, D), curved A / B / C, magnetic, anisotropic, OOP, slant, the
  `__all__` walker, doc identifiers, history lint / relocation / fingerprint
  tool, dispatcher doc consistency, except budget and public API:
  `1093 passed, 2 xfailed in 569.22s` (`suite_foldin_win.txt`; the xfails
  are vc4 / vc5, E2's).  The first run of this sweep caught the NamedTuple
  coercion choking on Phase D's patched 7-slot `hgram_R` tuple
  (`test_d6_*`, 2 failed); the consumers now take the first six slots of a
  plain tuple (`_stag_homog_geom`), and the sweep above is the re-run.
* Byte identity after the refactor: 200 / 200 (9.2).  `python -m mypy`:
  no issues in 33 files; WSL ruff: all checks passed; history fingerprints
  `--check`: no drift.
