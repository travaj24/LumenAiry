# PLAN -- anisotropic permittivity for the PURE (staggered, no-floor) 2-D PMM

Date: 2026-09-09.  Status: PLAN (binding brief for the build agents).
Target release: 5.43.0 (MINOR -- a new solver capability).
User order: "add anisotropic functionality to the PMM 2D solver (not the
hybrid version)".  The pure solver is `pmm_efficiency_2d_staggered`
(`lumenairy/elements/pmm/twod_staggered.py`, Granet 2023 staggered
modified-Legendre basis) and its cascade `PMM2DStackPure`
(`stack2d_pure.py`).  Both are ISOTROPIC SCALAR today; both docstrings
already name "full anisotropic tensors and out-of-plane coupling" as the
staged extension, and `docs/PMM_ROADMAP.md` Phase C ("2-D anisotropic --
the foundation") is this work.

## 0. What already exists (do not rebuild)

| piece | where | status |
|---|---|---|
| staggered basis `Basis1D` (B / Btilde sets, m_ref / c_ref / s_ref) | twod_staggered.py:148 | shipped, reuse unchanged |
| isotropic assembly `Granet2DTransverseE._assemble` (R, [eps_t], S_tt, K_tz eps33^-1 K_zt) | twod_staggered.py:355 | shipped; THIS is what generalizes |
| eps-weighted kron helpers `_eps_weighted`, `_eps_dir` | same | reuse; they take a scalar (Nx,Ny) map -- give them a per-component map |
| H-partner recovery Eq.25 `_region_modes` | twod_staggered.py:713 | structure unchanged; `Lhh = k^2[eps_t] + S_tt` gains the off-diagonal blocks |
| eps-free geometric eig for UNIFORM SCALAR regions `_homog_geom_cache` | twod_staggered.py:754 | unchanged; NOT applicable to a uniform TENSOR (Schur is no longer eps-free) |
| square Redheffer cascade `_interface_smatrix` / `_propagation_smatrix` / `_redheffer_star` | pmm/_core.py:1844 | unchanged for in-plane (the `[W;-V] <-> -lam` symmetry holds) |
| far field `_stag_fourier_projection`, `_far_projector_2d`, `_pmm2d_project_orders`, `_pmm2d_order_kz`, `_project_efficiency` | twod_staggered.py:615-711 | unchanged |
| two-polarization Jones driver pattern | stack2d_pure.py `solve()` (rows = incident Ex / Ey, `jones` = order-0 reflection) | copy the pattern for the new single-layer Jones entry |
| hybrid tensor Jones-2D (the API mirror) | twod_jones.py `pmm_jones_2d` (eps_cell (Nx,Ny,3,3), `_tile_is_offplane`, `_require_nonzero_ezz`) | API template + cross-engine oracle (Laurent floor ~1e-3) |
| out-of-plane machinery (Fourier basis) | rcwa/_core.py `_layer_eigenmodes_tensor` (generator G=[[A,P],[Q,B]], factor-i fixed), `_select_forward_flux`, `_interface_smatrix_general`, `_propagation_smatrix_general`, `_generator_block_eig` | reference for Stage B only |
| 1-D SEM first-order OOP generator | pmm/_core.py:5738 `_build_generator_metric` (state [Ex;Ey;iZHx;iZHy], L=A+BC^-1D) | reference for Stage B only |
| tensor helpers | rcwa/_core.py:2790 `uniaxial_tensor`; pmm/_core.py:96 `_promote_eps_tensor`, `_tensor_is_passive` | reuse |
| Berreman oracles | `lumenairy.elements.berreman.berreman_jones_1d` (library), `tests/unit/_berreman4x4.berreman_jones` (test oracle) | exact for uniform layers, any incidence, in-plane AND OOP |
| the paper | `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/PMM_Papers/josaa-40-4-652.pdf` (Granet, JOSA A 40, 652 (2023)) | READ IT (pages 2-3 = Eqs. 5-25; page 9 = Appendix A Eqs. 39-45; page 7 = Table 2 oracle) |

Conventions of the staggered module (PUBLIC end to end, NO conjugation bridge,
unlike twod_jones.py): `exp(-i w t)`, `Im(eps) > 0` = loss, forward
`exp(+i kz z)`, `Im(kz) >= 0`, `gamma^2/k0^2 = n_eff^2`.  The PAPER uses
`exp(+i w t)`: every tensor quoted from it must be CONJUGATED (its Eq. 37
`eps12 = -i 0.5` becomes `+i 0.5`; its substrate `1 - 5i` becomes `1 + 5i`).
A gyrotropic sign error is invisible to every energy check and only shows
in the +/- order asymmetry -- Table 2 (see G2) is the discriminator.

## 1. The formulation -- Granet 2023 is ALREADY the block-form anisotropic case

The shipped code implements the isotropic reduction of the paper's GENERAL
equations.  With `chi_t = I` (nonmagnetic) and the BLOCK-FORM tensor of the
paper's Eq. 7,

    [eps] = [[e11, e12, 0], [e21, e22, 0], [0, 0, e33]]     (per cell, constant)

the paper gives (Eqs. 19-25, Appendix A Eqs. 40-44):

    -gamma^2 R [E1;E2] = L [E1;E2],   R = C[chi_t]C = -I  (unchanged)
    L = k^2 [eps_t] + S_tt - K_tz (eps33)^-1 K_zt

    [eps_t]  : Eq. 40 -- FOUR blocks.  eps11 in V1xV1, eps22 in V2xV2 (as now)
               PLUS the MIXED blocks eps12 = <V1 | e12 | V2>, eps21 = <V2 | e21 | V1>
               (V1 = B(x1)Btil(x2) test / Btil(x1)B(x2) trial: kron of the 1-D
               mixed masses <B|Btil>_x and <Btil|B>_y -- `Basis1D.mass`
               between UNLIKE sets, per-cell weighted by e12 / e21).
    eps33    : Eq. 41 -- `Meps33` weighted by e33 (not the scalar eps).
    S_tt     : Eq. 42 -- unchanged (chi33 = 1).
    K_tz     : Eq. 43 -- unchanged for chi_t = I (the mimetic gradient).
    K_zt     : Eq. 44 -- TWO terms per column:
               Kzt_E1 = <V3 | d1(e11 .)| V1> + <V3 | d2(e21 .) | V1>
               Kzt_E2 = <V3 | d2(e22 .)| V2> + <V3 | d1(e12 .) | V2>
               i.e. the div of D_t = eps_t E_t.  Derivative on the TEST (V3)
               function exactly as the shipped `_eps_dir(..., "dL", ...)`
               realizes it; the two NEW terms pair the y-derivative with the
               E1 trial (V1) and the x-derivative with the E2 trial (V2).
    Eq. 25   : gamma C [H1;H2] = [k^2 [eps_t] + S_tt] [E1;E2] -- the H-partner
               recovery in `_region_modes` is structurally unchanged; `Lhh`
               now carries the eps12/eps21 blocks.  H1 lives in the V2 basis
               and H2 in the V1 basis, as today (the interface match is
               therefore still a SQUARE modal match).

Consequences that matter:
* IN-PLANE (block-form) anisotropy keeps the SECOND-ORDER generalized
  eigenproblem at the SAME dimension 2q^2, the same `[W;-V] <-> -lam`
  symmetry, the same square Redheffer cascade, the same far field.  This is
  a moderate operator-assembly generalization -- LOW risk.  Gyrotropic
  (e12 = -e21 = i b, Hermitian) is included and is lossless.
* OUT-OF-PLANE (e13/e23/e31/e32 != 0) BREAKS the paper's Eq. 16 (div D = 0
  no longer slaves E3 algebraically: E3 appears under d_t through e13/e23
  while gamma multiplies e31/e32).  It is NOT a second-order problem any
  more; it needs a first-order (4q^2) or a linearized quadratic (6q^2)
  formulation whose spurious-mode behaviour in THIS staggered basis is
  unknown.  It is therefore a PROTOTYPE-GATED stage (Section 4), not a
  build.  The user's device need (LC director tilting out of plane under
  field) makes it wanted; the discretization risk makes it prototype-first.
* A UNIFORM in-plane tensor layer/cell is NOT eps-free-separable (Kzt is
  no longer a scalar multiple of Kzt0), so it takes a full region eig like a
  patterned cell.  Half-spaces stay ISOTROPIC (documented; the hybrid has
  the same restriction).

## 2. Stage A -- in-plane block-form anisotropy (BUILD)

### 2.1 Library changes
1. `Granet2DTransverseE(..., eps_cell)` accepts `(Nx, Ny)` scalar OR
   `(Nx, Ny, 3, 3)` block-form tensor.  Scalar input runs the EXISTING code
   path byte-for-byte (a dispatch, not a rewrite -- the reduction gate G1
   is bit identity on `Lmat`, `Rmat`, `Stt`, `Schur`, `Et_blocks`).  Tensor
   input runs the generalized assembly of Section 1.  Keep `Et_blocks` a
   2-tuple for scalar solvers (existing consumers); expose the tensor
   blocks on a new attribute (e.g. `Et_offdiag = (Et_12, Et_21)`, None for
   scalar) and make `_region_modes` fold them into `Lhh` when present.
   Enumerate EVERY consumer of `Granet2DTransverseE`, `_region_modes`,
   `Et_blocks`, `Schur`, `eps_cell` (grep lumenairy/ tests/ validation/
   --include='*.py') before touching signatures.
2. `_require_block_form(fn_name, tile33)`: OOP entries (relative test,
   copy `_tile_is_offplane`'s `1e-12 * scale` floor -- NOT a strict `> 0`)
   raise `NotImplementedError` naming `pmm_jones_2d` (hybrid, OOP-capable)
   until Stage B lands; `e33 == 0` raises (`_require_nonzero_ezz` mirror).
3. NEW public `pmm_jones_2d_staggered(period_x, period_y, eps_cell,
   n_substrate, n_superstrate, depth, wavelength, *, degree=8, n_modes=None,
   n_orders=7, theta=0.0, phi=0.0)` -> `(orders (Nfo,2), R (2,Nfo),
   T (2,Nfo), jones (2,2))` -- the `pmm_jones_2d` / `PMM2DStackPure.solve`
   shape (row 0 = incident Ex, row 1 = incident Ey; `jones` = order-0
   REFLECTION, PUBLIC convention, NO conj).  `eps_cell` scalar `(Nx,Ny)` is
   promoted to `e * I` per cell.  Same guards as `pmm_efficiency_2d_staggered`
   (square grid, M >= 3, propagating incidence, grazing-safe wavelength +
   cutoff warning; include the tensor DIAGONALS' real parts in the
   `_grazing_safe_wavelength` eps list as `_pmm_jones_2d_at` does).
   `pmm_efficiency_2d_staggered` STAYS scalar; a 4-D `eps_cell` there raises a
   ValueError pointing at the Jones entry.  Export: `pmm/__init__.__all__`,
   the package-level re-export wherever `pmm_jones_2d` is exported, and
   `tests/unit/test_public_api.py` if it pins the public list.
4. `PMM2DStackPure.add_layer(thickness, *, eps=None, eps_cell=None)`: `eps`
   accepts scalar or `(3,3)` (uniform anisotropic layer -> its own region eig
   on the union grid, deduped by bytes like patterned cells); `eps_cell`
   accepts `(Nx,Ny)` or `(Nx,Ny,3,3)`.  The union-grid rule is unchanged.
   `retain_internal` / `layer_absorption` / `PerOrderAmplitudesMixin` must
   keep working: the flux form is the eps-free block Gram, so absorption of a
   lossy TENSOR layer should close (G9) -- verify, do not assume.
5. Energy tripwire (5.39.1 lesson: closure guards must match what the path
   establishes): when every tensor in the cell is HERMITIAN (lossless) and
   both half-spaces are lossless, warn if `abs(sum R + sum T - 1) > tol`
   in the Jones entry and the stack -- mirror `twod._warn_lossless_energy_2d`'s
   shape; `tol` DERIVED from the measured closure of the shipped scalar
   fixtures (record the numbers, date them) with decades of gap, and it must
   NOT fire on any existing scalar test fixture (run them).  Non-Hermitian
   tensors get no unity claim (never assert unity -- R+T != 1 is physical).
6. Memory / speed: the tensor path adds two mixed mass blocks (kron of 1-D
   mixed masses, cheap) and two K_zt terms; `Lmat` stays 2q^2 x 2q^2 and the
   eig cost is unchanged.  Do not retain new dead operators (audit P3-37:
   Curl/Kzt/Ktz/Meps33 stay assembly locals).  The scalar path must do NO
   additional work (G1 bit identity is the proof).
7. Docs: module docstrings (scope lines in twod_staggered.py and
   stack2d_pure.py), `docs/PMM_ROADMAP.md` Phase C -> shipped (in-plane),
   OOP status line, CHANGELOG `## [Unreleased]` (create the section -- the
   three BOR SEM commits since 5.42.1 wrote none; add theirs from their
   commit messages while you are there, briefly), and the build doc
   `docs/audits/BUILD_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` with every
   measured number the tests' bars derive from.

### 2.2 Validation gates

All per docs/TESTING_STANDARDS.md: decisions not readings; derived bars with
a gap on both sides and the measurement dated in the comment; two-arm
comparisons only; no cross-build value pins.

| gate | claim | oracle | bar derivation |
|---|---|---|---|
| G1 reduction | scalar cell == tensor `e*I` cell | shipped scalar path | scalar dispatch: BIT-IDENTICAL operators and R/T (hash both arms). Tensor path FORCED on `e*I` (private flag): rel. diff on R/T/jones below a bar derived from a measured spread x 1e2, expected ~1e-14 |
| G2 published oracle | Granet 2023 Table 2 (SEM, M=7): (1,1) 0.0268, (-1,1) 0.0137, (0,-1) 0.0620, (0,0) 0.2979; FMM Table 3 M=10: 0.0269/0.0137/0.0619/0.2980 | the paper | geometry: dx=2.4 lam, dy=1.4 lam (Fig. 4; the text's "d_y = 2.4, d_y = 1.4" is a typo), wx=0.5dx, wy=0.5dy, h=lam, pillar eps_b = conj([[2.25,-0.5i,0],[0.5i,2.25,0],[0,0,2]]), surround eps_a = conj(eps_b^*), substrate eps = 1+5i (lossy!), vacuum cover, normal incidence, E along x, TRANSMITTED orders. Square grid: pillar occupying half of each axis -> (2,2) cell (position invariance makes placement immaterial -- ALSO assert that). Bar: the paper's own M5->M7 movement (<=1e-4) + 4-digit rounding + SEM/FMM disagreement (1e-4 on (0,-1)) -> 5e-4 absolute per order at M>=7. RISKS to resolve by measurement, not assumption: (i) efficiency DEFINITION into a lossy substrate (ours: Re(kz_trn/kz_inc) abs(t)^2; try the alternatives if it disagrees, record which matches); (ii) order-sign convention (m,n) vs the paper's (the gyrotropic +/- asymmetry is the whole point -- the +i/-i conjugation and the order mirror are BOTH sign traps; a wrong one matches (0,0) and mismatches (1,1)/(-1,1)). If NO definition matches within 5e-4, record the ambiguity honestly and rely on G5's three-engine agreement instead -- do not tune. |
| G3 uniform tensor slab | multi-segment grid, uniform in-plane tensor (rotated uniaxial via `uniaxial_tensor(no, ne, pi/2, phi)` and the gyrotropic one) vs Berreman 4x4 | `berreman_jones_1d` + `tests/unit/_berreman4x4` | normal, oblique 25 deg, conical phi 40 deg, both pols, R/T AND Jones. Uniform regions converge SPECTRALLY: measure at M=5 and M=7, bar = 1e2 x the M=7 residual, and assert the M=5->M=7 residual DROPS (two-sided: not just "small"). Note the caveat in stack2d_pure.py: a UNIFORM region at OBLIQUE incidence is degree-limited (Bloch phase must be resolved) -- pick M so the measured residual is decades under the bar |
| G4 1-D reduction | y-uniform anisotropic stripe grating (no corners: 1e-4 by M=8 per the 5.21 record) vs `pmm_jones_1d` (degree 16, `stabilize=False`) and `rcwa_jones_1d` (many orders), PER ORDER, both pols, plus Jones | 1-D engines | convergence ladder M = 5,6,7,8 measured and tabulated in the doc; bar = 3x the M=8 residual (deterministic discretization error, not build noise); also assert y-momentum conservation (y-forbidden orders ~1e-12 -- derive) |
| G5 2-D three-engine | patterned 2-D cell: isotropic pillar in a rotated-uniaxial in-plane LC background (the exp29 comb regime) and the reverse | `pmm_jones_2d` (hybrid, Laurent floor ~1e-3) and `rcwa_jones_2d` | all three agree within a bar derived from the hybrid's floor (measure their mutual spread; bar ~3x); PLUS the no-floor property two-sided: staggered R/T change < derived (~1e-10) when n_orders goes 4->8 while the hybrid's change is visibly larger |
| G6 closure two-sided | Hermitian (lossless, incl. gyrotropic) cell: `abs(R+T-1)` under a derived bar; a lossy cell (anti-Hermitian part) closes BELOW 1 by more than a margin | self | measure closure at two M; bar = 1e2 x measured, must be decades above the 1e-15 build spread |
| G7 symmetry | transpose the cell (x<->y, swap e11<->e22, e12<->e21): orders (m,n)->(n,m), Jones rows/cols swap; mirror y->-y with director phi->-phi: orders (m,n)->(m,-n), off-diagonal Jones flips sign | self (two arms, same build) | these are EXACT symmetries of the discretization on a symmetric grid -> expect ~1e-14; bar derived x 1e2. A swapped e12/e21 placement FAILS these -- that is the point |
| G8 stack | (a) tensor A over isotropic B cascade vs the same as single layers (consistency), (b) an all-UNIFORM anisotropic multilayer vs Berreman multilayer at oblique | Berreman | (b) spectral: bar as G3 |
| G9 absorption budget | lossy tensor layer in `PMM2DStackPure(retain_internal=True)`: `sum layer_absorption == 1 - sum R - sum T` | self | derived from the shipped C3 test's measured closure |
| G10 guards | non-square grid, e33=0, bad tensor shape, OOP tensor -> NotImplementedError (relative floor: a 1e-16 xz stray does NOT trip it), 4-D grid to the scalar entry -> ValueError | -- | messages name the alternative |

Test-cost rule: every new test file < 3 min single-threaded, no single test
> 40 s, grids <= (3,3), M <= 8 (the staggered eig is `2*(Nx*(M-1))^2`-dim;
`test_two_pillar_cell_vs_rcwa_and_staggered` already costs 183 s -- do not
add another).  Set the OMP/OPENBLAS/MKL caps at file top as the neighbours
do.  Record measured durations in the doc.

## 3. Deliverables of Stage A
* code: twod_staggered.py, stack2d_pure.py, pmm/__init__.py (+ package
  re-export), tests/unit/test_pmm2d_staggered_anisotropic.py (+ a stack
  file if cleaner), docs as in 2.1.7, BUILD doc.
* the build doc's measurement tables ARE the evidence: every bar in every
  test cites the table row it derives from.
* `ruff check lumenairy/ tests/` clean; the touched suites + every staggered
  / pure2d test file green in the worktree (list them in the doc).

## 4. Stage B -- out-of-plane coupling (PROTOTYPE, gated)

Goal: a GO / NO-GO verdict with measured tables for an OOP-capable staggered
formulation, in `validation/probe_pmm2d_staggered_oop/` (no library changes),
written up in `docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md`.

Candidates to prototype (both, on tiny cells, self-contained copies of the
staggered assembly helpers are fine):

 (a) FIRST-ORDER staggered generator on [E1;E2;H1;H2] (dim 4q^2): eliminate
     H3 STRONGLY (curl_t E lands exactly in Vw = B(x)B) and E3 WEAKLY through
     the longitudinal H-equation tested in V3 (derivative moved onto the
     continuous V3 test, the `_eps_dir("dL")` device), with the POINTWISE
     e33-Schur effective in-plane tensor (per cell, exact for piecewise
     constant) and the A/B cross-blocks of Li 2003 -- the structure of
     rcwa `_layer_eigenmodes_tensor`'s generator (READ its FACTOR-i comment
     and `docs/audits/AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14.md` -- six
     copies once shipped the wrong relative +/-i) transplanted into the
     staggered spaces (H1 in V2, H2 in V1, E3 in V3, H3 in Vw).  Forward /
     backward split by `_select_forward_flux` (needs Ex,Ey,Hx,Hy blocks in a
     consistent flux inner product -- use the block Gram, cf. PMM_ROADMAP
     C-FLUX), generalized S-matrix (`_interface_smatrix_general`).

 (d) QUADRATIC-in-gamma E-formulation that KEEPS Granet's div(D)=0 slaving:
     state [E1;E2;E3], gamma^2 diag(R,0) + gamma M1 + M0 = 0 with the OOP
     terms in M1 (i gamma e_zt E_t, i gamma e33 E3) and M0 (d_t(e_tz E3)),
     linearized to a 6q^2 pencil.  Slower (~3.4x the eig of (a)) but it
     preserves the constraint that made the isotropic solver spurious-free.

Measurements (tables in the doc, every one on BOTH candidates):
 1. UNIFORM-SLAB DISPERSION (the replicable probe from the factor-i audit):
    uniform OOP tensor `uniaxial_tensor(1.5, 1.7, 35 deg)` on a (2,2)/(3,3)
    grid; eig of the generator vs the EXACT roots of det(k x k x . + k0^2 eps)
    = 0 per Bloch harmonic, normal AND oblique.  Machine precision expected
    for the physical branches; census of everything else (spurious).
 2. Berreman R/T/Jones of the uniform OOP slab vs `berreman_jones_1d`,
    convergence in M (table), normal / oblique / conical.
 3. IN-PLANE LIMIT: cross terms -> 0.  Compare the candidate's eigenvalue
    SET against Stage A's E-form (or an in-place copy of the shipped
    isotropic solver) on the same cell: are the two discretizations
    EQUIVALENT (distance ~1e-12) or merely both convergent?  This decides
    whether an OOP path can reduce to the in-plane path exactly.
 4. y-uniform OOP stripe grating vs `pmm_jones_1d` (OOP metric generator)
    and `rcwa_jones_1d` (GAP7 full tensor), per order, both pols.
 5. A genuinely 2-D OOP pillar vs `pmm_jones_2d` (hybrid OOP, 4Nf generator).
 6. Spurious census + cascade stability: modes with |flux| ~ 0 and small
    |Re gamma| not on a physical branch; the cascade energy of a lossless
    Hermitian OOP cell must close (two-sided as G6) with NO exp(+|Re gam|L)
    blow-up at 2-3 depths.
 7. Cost: dims and eig wall times vs the in-plane 2q^2 path; whether the
    normal-incidence anti-commuting-involution block reduction
    (`_generator_block_eig`, EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md)
    verifies on the staggered generator.

Verdict rule: GO for a candidate only if 1, 2, 4, 6 pass at their derived
bars AND the spurious census is empty or provably benign (bounded-and-
priced); otherwise NO-GO with the failure mechanism pinned by measurement.
Honest negatives are deliverables (cf. EXPERIMENT_PMM2D_EIG_RECYCLE).

Stage B integration (a later wave, only on GO): route an OOP tile through
the winning generator + generalized cascade inside `Granet2DTransverseE` /
`_region_modes` (6-tuple like `_tensor_layer_modes`), `PMM2DStackPure`
mixed cascades (`_modes_to_M` with isotropic half-spaces `[W, W; V, -V]`),
G1-style reduction to Stage A, and the Stage-B gates above as tests.

## 5. Out of scope (say so in the docs)
JAX twin (the staggered path is NumPy-only today); magnetic anisotropy
(`chi_t != I`, though the paper's R = C[chi_t]C makes it a one-line follow-on);
anisotropic HALF-SPACES; slant / curved (Phases D/E, which REUSE this tensor
operator); the GPU eig (user-excluded).

## 6. Process rules (binding)
* opus subagents, <= 5 concurrent; adversarial verification re-MEASURES,
  never reads comments; "right conclusion, wrong numbers" is a defect.
* Bit-identity claims only same-build / two-arm; never cross-build pins.
* No supervisor / retry scripts; PYTHONPATH pinned to the worktree with an
  `assert lumenairy.__file__` guard in every probe; OMP caps exported before
  python starts; `grep --include='*.py'`.
* Release: TAG ONLY AFTER the full un-masked main-CI matrix is green on the
  merge; `NEXT_REMOVAL_VERSION` (lumenairy/_deprecation.py, currently '5.44')
  is slipped proactively in the release commit.
