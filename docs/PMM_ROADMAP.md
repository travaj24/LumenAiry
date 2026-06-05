# LumenAiry PMM (Polynomial Modal Method) — Roadmap

Status as of 2026-06-04. The PMM family is the non-Fourier (spectral-element)
complement to the Fourier modal method (RCWA). This roadmap tracks what ships,
what's in flight, what's planned, and — honestly — what's hard or unresolved.

---

## 1. Capability matrix (PMM solvers)

Legend: ✅ shipped · 🔄 in flight · ⬜ planned · ❌ open/known-hard ·
(via grid) = already covered by an existing input form.

| Regime | Isotropic binary | Isotropic multi-region | Anisotropic (Jones) | Aniso + multi-region |
|---|---|---|---|---|
| **1-D vertical** | ✅ `pmm_efficiency_1d` | ✅ `pmm_efficiency_1d_segments` | ✅ `pmm_jones_1d` | ✅ `pmm_jones_1d_segments` |
| **1-D slanted** (normal inc.) | ✅ `pmm_efficiency_1d_slanted` | 🔄 (= diagonal case of below) | 🔄 `wiiihr0hl` | 🔄 `wiiihr0hl` |
| **1-D oblique + slant** | ❌ cross-term unresolved | ❌ | ❌ | ❌ |
| **2-D vertical** (rect pillars) | ✅ `pmm_efficiency_2d` (FMM-floored) + ✅ `pmm_efficiency_2d_staggered` (no-floor) | ✅ (via `eps_cell` grid; staggered) | ⬜ planned | ⬜ planned |
| **2-D slanted** | ⬜ planned (moderate) | ⬜ | ⬜ | ⬜ |
| **2-D curved** (cylinder/ellipse) | ⬜ planned (hard) | ⬜ | ⬜ | ⬜ |

Plus: `PMMStack` (multilayer 1-D), `grating_convergence_class` /
`classify_from_grating` (Li-Granet edge convergence predictor).

---

## 2. Shipped this cycle (pushed to `main`, unreleased 5.11.0)

| Commit | Capability |
|---|---|
| `fff7301` | PMM element-size conditioning fix (thin-feature / tapered-stack `Singular matrix`) |
| `ac66203` | **Slant 1-D PMM** (`pmm_efficiency_1d_slanted`, ~30–70× vs RCWA-staircase, normal inc.) + **convergence predictor** |
| `652c9d3` | **Canonical no-floor 2-D PMM** (`pmm_efficiency_2d_staggered`, Granet staggered basis) |

---

## 3. In flight / latest

- **1-D slanted anisotropic + multi-region (Jones)** — *the hard kernel is solved.*
  Round 1 (`wuziwokcj`) proved the vertical reduction + all plumbing but the
  finite-slant **tensor convection operator** was wrong. Round 2 (`wiiihr0hl`)
  **fixed and verified the operator** (the 1/εxx-weighted antisymmetrized
  convection + sec²(φ) metric + the `Px=blkdiag([[1/εxx]],I)` similarity bridge +
  re-derived V): the diagonal-tensor Ex/TM eigenvalues now match the scalar TM
  slant solver to **9.2e-10** (was 2e-2…0.23), confirmed a *true operator
  equality*, not the passive-wrong trap. **One localized issue remains:**
  finite-slant **energy** doesn't yet conserve (severe at steep tilt) — *proven
  not to be the operator* (the diagonal tensor, whose blocks are matrix-identical
  to the scalar operators, leaks identically). The leak is isolated to the
  **far-field / half-space projection plumbing** (it matches inclined layer modes
  against *vertical*-tensor half-spaces with a lab-frame projection; the working
  scalar `_pmm_slant_solve` builds half-spaces + projection in the *same inclined
  frame*). **Fix = a localized plumbing change** (mirror `_pmm_slant_solve`'s
  inclined-frame half-spaces/projection), then integrate. *Build paused per
  user request (roadmap mode); queued as Phase A's remaining step.*

---

## 4. Roadmap phases (ordered; difficulty + dependencies)

### Phase A — finish 1-D anisotropic slant *(in flight)*
`wiiihr0hl` → integrate `pmm_jones_1d_slanted` / `…_segments`. **Risk:** the
finite-slant tensor convection is the genuinely hard kernel (same flavor as the
open oblique+slant cross-term); may take another round. **Multi-region falls out
for free** once the operator is right.

### Phase B — speed-at-accuracy benchmark *(queued, quiet CPU)*
Wall-clock-at-matched-accuracy of the no-floor staggered 2-D PMM vs the
FMM-floored hybrid vs RCWA, across contrast regimes. **Purpose:** settle, with
numbers, where PMM beats RCWA *today* (speed and/or the no-floor accuracy
ceiling on hard Gibbs cases). Informs whether to prioritize the 2-D builds.
Cheap; must run on a quiet CPU (no concurrent workflows).

### Phase C — 2-D anisotropic (general tensor operator) — **the foundation**
Generalize the 2-D staggered eigensolver from the isotropic reduction
(`R=−I`, `[ε_t]=ε·I`) to the **full transverse tensor** (Granet 2023 general
Eqs.23-24 / Appendix A). `eps_cell` → `(Nx,Ny,3,3)` tensor per segment.
**Reuses unchanged:** the staggered basis, square S-matrix, far field,
spurious-free structure. **Difficulty:** moderate (operator-assembly
generalization). **Why first:** see §6 — it's the same machinery slant/curved
needs. In-plane (block-form) first; full out-of-plane is a harder follow-on.
*(2-D multi-region is already supported via the `eps_cell` grid — only needs a
multi-region test.)*

### Phase D — 2-D slant — **the speed-win regime**
Affine coordinate shear (Granet Eq.38) → constant metric → constant anisotropic
tensor. Lifts the 1-D slant idea to 2-D; staircase-avoidance speed win
(Edee-Granet 2024 ~3.4× vs FMM). **Reuses Phase C's tensor operator.**
**Difficulty:** moderate. **This is the user's tapered-device regime.**

### Phase E — 2-D curved (cylinder/ellipse) — **hard**
Granet's transfinite curved-quad mapping (Sec.3A): geometric construction of
curved elements on the boundary + **within-element Gauss-Legendre quadrature**
for the position-dependent metric + full position-varying tensor operator.
Unlocks the cylinder anchor (n_eff 1.200502441) and spectral convergence on
smooth boundaries. **Reuses Phase C.** **Difficulty:** hard (~3-5 rounds).

### Phase F — exhaustive codebase audit
Bugs / physics / adversarial sweep (explicitly requested earlier; long-pending).

### Smaller, independent items
- 2-D multi-region **test** (the capability exists; just unvalidated on large grids).
- Port the **sparse shift-invert eigensolver** (built proto, ~20×) for large PMM.
- Fix the 1-D **TM ~1e-4 floor** properly (needs a DG/mortar S-matrix; see §5).

---

## 5. Open items / known-hard / honest negatives

- **1-D oblique + slant** — the inclined-frame Bloch↔slant convection cross-term
  is unresolved (scalar *and* tensor). Currently guarded with
  `NotImplementedError`. Same kernel `wiiihr0hl` is wrestling with.
- **1-D TM ~1e-4 floor** — matched-coordinates is a verified dead-end (it's the
  identity in a wall-fitted PMM). The working lever is the shipped hp-refinement
  (`grade` + `elements_per_region`, ~1e-5). A true sub-1e-5 fix needs a DG/mortar
  S-matrix (a larger, separate effort), not a metric add-on.
- **2-D vertical-pillar corner cap** — *fundamental* (Li-Granet 2011): a
  right-angle dielectric corner caps convergence to algebraic for *every* method.
  So the no-floor 2-D PMM is **RCWA parity per DOF on vertical pillars, not a
  speed win** — its win there is accuracy *quality* (no floor, exact sidewalls,
  position invariance, pinning the value RCWA converges toward). The genuine
  *speed* win lives in slant/curved (Phases D/E).
- **Full out-of-plane anisotropy** (2-D non-block tensor, magneto-optic / tilted
  director) — breaks Granet's block-form assumption; harder than in-plane.
- **No-floor 2-D for arbitrary curved boundaries without the transfinite map** —
  rounding/rasterizing a curve in a Cartesian basis does *not* restore spectral
  convergence (verified); needs Phase E or a 2-D ASR coordinate stretch.

---

## 6. Strategic dependency graph (what reuses what)

```
  [staggered basis + square S-matrix + far field]   ← SHIPPED (2-D foundation)
                        │  (reused by every 2-D extension)
                        ▼
            [general tensor operator]   ← Phase C (2-D anisotropic)
                   │              │
   (metric → effective tensor)   │  (material tensor)
                   ▼              ▼
        [2-D slant, Phase D]   [2-D anisotropic materials]
                   │
                   ▼
        [2-D curved, Phase E]  (+ transfinite map + quadrature)

  [1-D anisotropic-slant convection lessons, wiiihr0hl] ──► transfer to Phase C
```

**Key insight:** the curvilinear metric turns even isotropic ε into an effective
anisotropic tensor (Granet Eq.5), so **building the general tensor operator once
(Phase C) serves both anisotropic *materials* and slant/curved *geometry*.**
That's why Phase C is the linchpin and is sequenced before D/E.

---

## 7. Which solver wins which job (the recalibrated "speed-AND/OR-accuracy" bar)

| Job | Recommended solver | Why |
|---|---|---|
| Vertical / axis-aligned, many-layer stacks, dispersion sweeps, inverse-design (autodiff) | **RCWA** (`rcwa_efficiency_2d` / `RCWAStack`) | Mature, FFT-fast, JAX-differentiable; parity-or-better on vertical |
| 2-D vertical, need exact energy / no-floor / position-invariance / pinned value on hard Gibbs cases | **`pmm_efficiency_2d_staggered`** | No Fourier floor; accuracy ceiling (to be quantified by Phase B) |
| 1-D / 2-D **slanted or tapered** sidewalls | **PMM slant** (1-D shipped; 2-D = Phase D) | Staircase avoidance → genuine speed win; the device regime |
| Anisotropic / tunable-LC / Jones | **`pmm_jones_1d`** (1-D); 2-D = Phase C | Full tensor → Jones |
| Curved (cylinder/ellipse) pillars, spectral accuracy | **PMM curved** (Phase E) | Transfinite map removes the Gibbs floor |

RCWA stays the **workhorse**; PMM is the **specialist** for slant/taper/curved,
no-floor accuracy, and anisotropic Jones.

---

## 8. Closing the coupled-slant-multi-region 1-D PMM — scaffolding + inter-component contracts

*Added 2026-06-04 after rounds 11–17. The binary slanted-tensor solver ships
(`pmm_jones_1d_slanted`, 039890e). This section is the complete papers-based plan
to close the remaining **multi-region + slant + full off-diagonal tensor** case.
The literature audit (2026-06-04) established that **no single paper does this
triple** — it is an assembly of ~5 papers. The scaffolding's payoff: **8 of 9
components are validated; component #4 is the sole open kernel.***

### 8.1 The pipeline (data-flow path)

```
for each region r (ridge / groove / segment), at slant φ:
  [ ε(x) tensor , slant metric g^ij ]
        │   #1 operator assembly · #2 inverse-rule · #3 div-conforming Ez-elim
        ▼
  L_r   = div-conforming coupled-slant operator             ── component #4 (OPEN)
        │   generalized eig:  −γ² R E = L E
        ▼
  { q_m , E_m }   modal eigenpairs (per region)
        │   #5 magnetic partner
        ▼
  H_m = (1/γ) C⁻¹[k²ε_t+S_tt] E_m   →   W_r = [ E_t ; H_t ] columns (fwd+bwd)
        │
  super/substrate (homogeneous, slant=0):
  W_sup , W_sub   = analytic Rayleigh modes                 ── component #6
        │   #7 S-matrix: Li-1996 W-continuity across z=const interfaces
        ▼
  S = ⊛_r [ interface(W_r , W_{r+1}) · propagator(q_r , d_r) ]   (Redheffer)
        │   #8 per-order Poynting flux
        ▼
  R_q , T_q   diffraction efficiencies   ( Σ = 1 for lossless )
```

Each arrow is a **contract**: an object produced in one convention that the next
box consumes. The contracts are where energy silently leaks when a convention
disagrees — that is what cost rounds 12–16.

### 8.2 The unifying object — the normalized tangential state

Every modal column, every Rayleigh column, and every S-matrix block lives in the
**single** normalized tangential 4-vector (Edee-Granet 2024 Eq. 3):

```
ψ = [F1; F2; G1; G2] = [Ex ; Ey ; iZ·Hx ; iZ·Hy] ,   Z = √(μ0/ε0) ,  i² = −1
```

The `+iZ` weighting (not `+Z`, not `−iZ`) is **load-bearing** — it is what lets
the layer modes and the half-space Rayleigh modes share one S-matrix with no
per-side renormalization. Maxwell becomes symmetric in (F,G):
`χˡᵐᵖ ∂_m F_p = −k μˡᵐ G_m`, `χˡᵐᵖ ∂_m G_p = −k εˡᵐ F_m`, `k = k0`.

### 8.3 Component map (paper source + status)

| # | Component | Paper / source | Status |
|---|---|---|---|
| 1 | Vertical coupled-tensor layer operator (div-conforming) | Granet 2023 Eq. 24 | ✅ (Eq. 37 oracle; `_sem_modes_tensor`) |
| 2 | Slant / inclined-frame metric | Granet 2017 Eq. 5 | ✅ (scalar slant shipped) |
| 3 | Div-conforming Ez-elimination (spurious-free) | Granet 2023 Eq. 16–18 · Liu 2015 Eq. 2.3b | ✅ **diagonal** (round 16) |
| **4** | **Coupled (εxy/εyx≠0) slant-TM convection, div-conforming** | Granet 2023 Eq. 24 in slant metric + `slant_tensor_fix` 4-piece | ⬜ **OPEN KERNEL** |
| 5 | Magnetic partner H | Granet 2023 Eq. 25 | ✅ |
| 6 | Analytic homogeneous half-spaces | Li 1997 §3 · Randriamihaja 2016 §2B | ✅ (round 14) |
| 7 | Layer↔half-space S-matrix match | Li 1996 Eq. 15a · covariant continuity (identity) | ✅ (round 15) |
| 8 | Per-order Poynting flux / efficiency | Randriamihaja 2016 Eq. 14 | ✅ |
| 9 | Multi-region segmented assembly | `_build_sem_tensor_segments` | ✅ vertical (round-12 DISC2) |

### 8.4 Inter-component contracts (the math that must match)

The pieces compose **only if** these objects agree in convention at each boundary.

**C-OP — operator → modes (#1 → eig).** `−γ² R [E1;E2] = L [E1;E2]`,
`R = C[χ_t]C`, `C = [[0,1],[−1,0]]`,
`L = k²[ε_t] + S_tt − K_tz (ε33)⁻¹ K_zt` (Granet 2023 Eq. 24).
Convention: `q = γ/k0`, `λ = −iq`; forward modes have `Im(q) ≥ 0` (a tiny lossy
part in the grating ε fixes the branch; Li 1996 / Granet 2017). **The same
q-branch must be used in the layer and in both half-spaces** (else C-SMAT leaks).

**C-IR — inverse-rule placement (inside #1/#2/#3).** Per field component, set by
its continuity across the ridge/groove wall:
- wall-normal `Ex` (`Dx = εxx·Ex` continuous → `Ex` jumps): **inverse rule**
  `[[1/εxx]]⁻¹`;
- tangential `Ey` (continuous): **direct / Laurent** `[[εyy]]`;
- longitudinal `Ez` (eliminated by div D = 0): the `(ε33)⁻¹` must be
  **div-conforming** — `1/εzz` sits *inside* the z-derivative stiffness
  `∫(1/εzz)B′B′`, **NOT** a pointwise multiply on a separately-formed `Ez`. A
  pointwise `(ε33)⁻¹` admits a Gauss-law-violating static null → a spurious
  propagating mode pinned at the **harmonic-mean** index `√(2/(1/εr+1/εg))`
  (Liu 2015; "#spurious ≈ #internal nodes"). This is the round-16 defect, still
  latent in the shipped binary's per-order accuracy (~2e-4, energy unaffected).
- slant folds these (Granet 2017 metric Eq. 5): wall-normal
  `Oε11 = [[1/(εxx + εzz·tan²φ)]]⁻¹`, cross `Oε13 = −[[εzz]]·tanφ`, both kept
  *inside* the K-operators.

**C-CONV — slant convection (inside #4, THE OPEN piece).** Slant adds a
linear-in-q convection `(i·t/k0)·∂_u`, `t = −tanφ`. The TM/wall-normal convection
must be **antisymmetrized in the inverse-rule frame**:
`(i·t/k0)(Cinv − Cinvᵀ)` — **not** `2·Cinv` (the naive form converges cleanly to a
*wrong, energy-violating* answer). A `Px = blkdiag([[1/εxx]], I)` physical-field
similarity recasts the operator so the diagonal blocks are exactly scalar TM/TE
statics + the correct Li coupling; map `φx → Ex_phys = Cinv_xx·φx` **before** the
magnetic partner and the far-field projection (else wall-normal leaks ~6 %).

**C-PARTNER — E → H (#5).** `γ C [H1;H2] = (k²[ε_t]+S_tt)[E1;E2]` (Granet 2023
Eq. 25): H is the operator applied to the **converged** E (never read off a
companion block), scales as `+q` (growing), built in the **same frame** as E.

**C-FRAME — layer ↔ half-space (#4↔#6; the round-15 pin).** Layer modes are in
the inclined-frame **covariant** components; half-spaces are lab-frame
**Cartesian**. Because `det(g) = 1` and the interface `z = const` is a coordinate
surface common to both frames, the **only** Cartesian component altered by the
shear `u = x − tanφ·z` is the **normal** one: `Az_cov = tanφ·Ax + Az` — which is
*not* matched. So the **tangential** state `[Ex;Ey;iZHx;iZHy]` matches with the
**identity** (no tanφ, no √g). (Granet 2017 §5 + App. A; Edee-Granet 2024 p.1804:
covariant `E_p,H_p` continuous across `x^p = const`.)

**C-HALF — half-space Rayleigh modes (#6).** Per Fourier order, exactly one TE +
one TM analytic plane-wave pair, `γ_q = √(ε_a·k0² − α_q²)`, branch
`Re γ + Im γ > 0` (Li 1997 §3 Eq. 16). Built **analytically** — never by
eigendecomposing the homogeneous operator (TE/TM are exactly degenerate there →
`np.linalg.eig` loses the TE channel; round 13). Emitted in the **same ψ
convention** as the layer.

**C-SMAT — S-matrix stack (#7).** `W_r` = square matrix of all fwd+bwd modal ψ
columns of region r at the interface; continuity
`W_{r+1}[u;d]_{r+1} = W_r[u;d]_r`; interface t-matrix `t = W_{r+1}⁻¹ W_r`;
Li-1996 Eq. 15a **normalized** recursion (unconditionally stable). Conserves for
*any* complete fwd/bwd basis on both sides **iff** both sides share the ψ
convention (C-FRAME) and the q-branch (C-OP).

**C-FLUX — per-order efficiency (#8).** `S_z = ½ Re(Ex·Hy* − Ey·Hx*)` per order
(Randriamihaja 2016 Eq. 14); with `det g = 1` this is the plain Cartesian flux.
The **modal** flux used to sort fwd/bwd and to normalize must use the inner
product **matching the basis**: S0-Galerkin-mass-weighted for the weak-form modal
state, plain node-sum for the nodal-collocation state. Mixing them reads a true
propagating mode as null (round-16 diagnostic) or a flux-orthogonal mode as leaky
(round-15 diagnostic).

### 8.5 The one open kernel — component #4

The coupled-anisotropic (εxy/εyx ≠ 0) slant-TM convection in the div-conforming
inverse-rule frame. Every ingredient is validated **separately**:
the vertical coupled operator reduces to `_sem_modes_tensor` exactly; the TE slant
convection attaches cleanly; the div-conforming Ez-elim is proven for the
**diagonal** channel (round 16); the `slant_tensor_fix` 4-piece reached
REDUCTION-2 ≈ 9.2e-10 for the operator. The open problem is their **simultaneous**
assembly — making the *coupled* TM slant convection div-conforming. Rounds 2–8 and
the 6 round-16 attempts produced a garbage high-q spectrum when folding the TM
convection into the inverse-rule frame.

**Round 17 outcome (2026-06-04) — NOT closed; the bind is now precisely isolated
(a frontier wall in the PMM/SEM basis).** The 2nd-order pure-E operator
(`slant_tensor_fix`) passes REDUCTION-2 (9.2e-10) **and** is spurious-free (it was
already div-conforming — `1/εzz` inside the z-stiffness; no harmonic-mean null;
the round-16 garbage-q was a frame-mixing bug, missing the `Px` bridge). **The
*operator* is solved.** But its reconstructed magnetic partner `V = Q W/λ` is
**structurally wrong at finite slant** (partner ratio drifts 1.0 → 0.54 → −0.18
over 0→60°) → energy blows up. This is the rounds-2–8 V-partner wall, re-confirmed.
The 1st-order [E;H] Berreman generator gives V for free (eigenvector lower block)
but carries the round-16 spurious mode (pointwise Ez) and converges pathologically
slowly at high contrast (εr≥6). **The two formulations are mutually exclusive in
this nodal basis: 2nd-order = good operator / bad V; 1st-order = good V / spurious
mode.** Energy ~1e-4 holds only at low contrast (εr ≤ 2.4, slant ≤ 60°).

**Round 18 outcome (2026-06-05) — the round-17 "wall" is NOT a far-field
projection bug; it is the V-partner/layer-basis, exactly as round 17 found.** A
proto-author flag on `slant_tensor_vshear.py` (lines ~116–131) re-attributed the
residual TM/Ex leak (2.3e-2 at slant 0 → 38 % at slant 45°) to the *wall-normal
TM far-field Rayleigh projection* (plain `Tp` on physical `Ex` instead of a
`1/ε`-weighted TM-potential projection mirroring `_pmm_slant_solve`). **This
reframing is REFUTED, three ways, BLAS-pinned:**

- **PROBE B** (`round18_projfix_test.py`): keep the proto's *identical*
  ε-unweighted projection but swap ONLY the layer modes to the in-library
  flux-orthogonal metric oracle `_layer_modes_metric` → energy conserves to
  **1.76e-12** on the exact DIAGONAL+COUPLED gate-3 cells where the proto leaks up
  to 38 %. Same projection + flux-orthogonal modes ⇒ conserves ⇒ the projection
  cannot be the defect.
- **PROBE C** (`round18_probeC.py`): apply the reframing's prescribed
  `1/ε`-weighted TM projection to the proto's *own* convection-pencil modes →
  leak gets **worse** (Ex sumRT → 0.475, 52 % error). The ε-multiplied direction
  (`slant_vshear_proj.py`) is worse still (GATE2 0→1.19, GATE3 0.38→2.02). Both
  weighting directions span the homogeneous-half-space inner product; neither
  helps.
- **Basis check** (`round18_basis_ortho.py`): the proto `[W;V]` layer basis
  carries worst off-diagonal cross-flux **7.77** (8× the diagonal scale) in its
  own flux form, vs **0.999** for the metric oracle — the proto basis is **not
  flux-orthogonal**, the precise property the interface S-matrix needs to
  conserve. Per-mode admittance matching scalar to 1e-9 (GATE0) is necessary but
  NOT sufficient for basis flux-orthogonality.

The shipped `pmm_jones_1d_slanted` (metric generator) was re-verified to conserve
to **≤6.3e-12** across slant 0–85°, coupled εxy=εyx, high-contrast εr≈12, and
gyrotropic ±0.6 i — i.e. the *modes* change, not a projection change, is the
genuine and already-shipped fix. **No shipped-operator change was made; pmm.py
working tree is clean.** The round-17 verdict stands, recast precisely: the open
piece is the proto's V-partner / layer basis (the reshaped 2nd-order convection
pencil is not symplectic), not the operator, not the V per-mode admittance, and
not the far-field projection.

**Status: documented frontier limit.** The exact triple (slant × coupled-tensor ×
div-conforming SEM with a consistent V) is genuinely unpublished (confirmed by
both the in-folder audit and an external web reconnaissance). It is an *assembly*
of separately-solved halves, not an impossibility — but it is out of scope for
further blind PMM rounds. **Path forward when revisited (NOT more PMM grinding):**
(1) the external papers now in `PMM_Papers/` target it directly —
`popov_neviere_josaa18-11-2886_2001` (coupled-tensor local-frame factorization for
the V/normal projector), `jiang_liu_PIER148_151_2014` (Gauss-law-constrained
coupled mixed-FEM — spurious-free *and* coupled), `faghihifar_arxiv_2606.03537`
(oracle-free spurious-mode selector); (2) **build a Li-1999 oblique-coordinate
C-method oracle** — it absorbs slant into the metric (sidestepping the
V-partner/convection fold) and is the closest published formulation to the full
case, giving a much-needed second source of truth. Meanwhile the binary
slanted-tensor (`pmm_jones_1d_slanted`) + vertical multi-region
(`pmm_jones_1d_segments`) cover the common cases; use the RCWA staircase for the
niche coupled-slant-multi-region geometry.

**Banked win (independent of #4):** the diagonal TM spurious-mode cure
(div-conforming Ez-elimination; round 16, proto `slant_tm_opfix.py`) — replace the
shipped binary's pointwise `(ε33)⁻¹` (`pmm.py:_build_metric_generator`) with the
`1/ε`-in-stiffness form to close its latent ~2e-4 per-order accuracy gap
(energy-neutral; behavior change → review before shipping).

### 8.6 Reduction / consistency gates (must hold at every assembly step)

| Gate | Assertion | Tol |
|---|---|---|
| Vacuum | empty cell → T00 = 1, **0 spurious modes** | 1e-12 |
| REDUCTION-1 | slant = 0 → `_sem_modes_tensor` / `pmm_jones_1d` | 1e-12 |
| REDUCTION-2 | diagonal tensor → scalar TE/TM slant spectrum | 9e-10 |
| Spurious-free | no mode pinned at `√(2/(1/εr+1/εg))` | — |
| Energy | lossless Σ(R+T) = 1, all slants/widths/tensors | 1e-4 |
| Shipped-binary | fixed operator + library half-spaces still conserves | 1e-6 |

**Validated protos:** `slant_metric_gen.py` (#1+#5, binary, shipped), `slant_analytic_hs.py`
(#6, round 14), `slant_interface_fix.py` (#7 identity, round 15), `slant_tm_opfix.py`
(#3 diagonal cure, round 16), `slant_tensor_fix.py` (the 4-piece operator, REDUCTION-2 9.2e-10).
