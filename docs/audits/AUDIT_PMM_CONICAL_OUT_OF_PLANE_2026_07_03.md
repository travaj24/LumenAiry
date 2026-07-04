# PMM Conical / Out-of-Plane Incidence — Implementation Audit — 2026-07-03

**Scope.** How to add **conical (out-of-plane) incidence** — a nonzero azimuth `phi`, i.e.
`ky0 = Re(n_sup)*sin(theta)*sin(phi)*k0 != 0` — to the **1-D** PMM solve
(`pmm_efficiency_1d` / `pmm_jones_1d` / `PMMStack`), so that a 1-D grating (period in x,
invariant in y) can be simulated off the classical mount and return a clean 2x2 zeroth-order
reflection Jones matrix. Motivating consumer: exp12 (oblique-angle assessment of the exp10/exp11
LC-QWP out-couplers, in-plane **and** out-of-plane, up to 25 deg).

All file:line anchors are relative to
`d:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy\lumenairy\`.
Evidence gathered by a 3-agent read-only sweep of the current source (no code was modified).

---

## 0. Executive summary

**This is not a solver rewrite.** The initial read ("conical PMM is a multi-week research problem")
was wrong, and this audit corrects it. The coupled-solve machinery that conical incidence needs
already exists, is **basis-agnostic**, and is **already used by the 2-D PMM** — which is itself fully
conical. A native 1-D conical PMM is a **reduction of the existing 2-D conical build**: the y-axis
collapses from a Fourier-order sweep to a single scalar shift `ky0*I`.

Concretely there are two paths, and I recommend doing both in sequence:

- **Path A — PMM2DStack bridge (capability *today*, ~0.5 day).** `PMM2DStack.set_source(theta, phi)`
  is already conical (`stack2d.py:395-407, 471-472`). Feed the 1-D grating as a **y-invariant 2-D
  cell**; the solver hits its exact separable-in-y fast path. Correct now, at an `O(N^2)`-orders
  cost. This unblocks exp12's out-of-plane cut immediately and doubles as the same-family oracle.
- **Path B — native 1-D conical PMM (`O(N)`, ~3-6 days).** A dedicated 1-D conical entry point that
  builds the SEM operators with `GyF = ky0*Ip` and routes through the **already-shared** coupled
  eigenmode + generalized S-matrix + Jones chain. This is the 2-D build minus the wasted `(2n+1)^2`
  order set.

The hard pieces — the coupled 2N eigenproblem, the full-3x3 out-of-plane *tensor* generator, conical
half-space matching, forward-flux mode selection in a non-orthogonal basis, and Jones extraction —
are **done and shared** (`rcwa/_core.py`, consumed unchanged by `pmm/twod_jones.py`). The 1-D port
supplies operators to them; it does not reimplement them.

---

## 1. Terminology (load-bearing — do not conflate)

The codebase uses "out-of-plane" in **two unrelated senses**. Conflating them will wreck the work.

- **OOP *tensor*** = the permittivity tensor has z-coupling (`eps_xz/eps_yz/eps_zx/eps_zy != 0`). This
  is about the **material**, and it is **already shipped** in native 1-D (v5.11.0; slant added
  2026-06-07): `oned.py:113-119, 189-197, 274-294`, generator at `_core.py:3392-3520, 3765-3972`.
- **OOP / conical *incidence*** = the incident wavevector has a y-component (`ky0 != 0`, `phi != 0`).
  This is about the **illumination**, and it is **NOT implemented** in native 1-D (PMM *or* RCWA).
  This audit is exclusively about this second sense.

Almost every "out-of-plane" string in `pmm/` is the *tensor* sense. There is **no prior native
conical-incidence 1-D attempt to revive** — the 1-D angle is hard-wired to a single in-plane Bloch
shift with `ky == 0` (`oned.py:127` "classical mount, `ky=0`"; RCWA twin `rcwa/oned.py:223-225`,
which literally sets `Ky = zeros` at `rcwa/oned.py:519, 1152, 1337`). Conical 1-D is **new work**.

The one roadmap "open gate" that looks adjacent — `docs/PMM_ROADMAP.md:111-113`, "1-D oblique +
slant ... Bloch<->slant convection cross-term ... guarded with NotImplementedError" — is about
in-plane `kx0` vs grating **slant**, not conical `ky0`, and has since partly shipped
(`oned.py:845-863`). It is not a blocker for this work.

---

## 2. Physics: why conical couples TE and TM

At the classical mount (`phi=0`, `d/dy=0`), a 1-D grating's Maxwell system **separates** into two
independent scalar problems:

- **TE**: `(E_y, H_x, H_z)`, governed by a scalar Helmholtz ODE in `E_y`.
- **TM**: `(H_y, E_x, E_z)`, governed by a scalar ODE in `H_y`.

The PMM engine exploits exactly this: `_sem_modes` solves one *or* the other via a `polarization`
switch (`_core.py:482-534`), and the two are never coupled.

Under **conical incidence** the invariant-axis wavenumber `ky0` is nonzero. Every `d/dy` in Maxwell's
equations becomes `i*ky0`, which injects `Kx*Ky` / `Ky*Kx` cross-terms into the modal operator. These
off-diagonal blocks vanish at `Ky=0` (recovering the scalar split) and are nonzero for `phi!=0`. A
single incident polarization then excites **both** tangential field components — TE and TM merge into
one **coupled 2N** (for isotropic layers) or **4-field 4N** (for full-tensor layers) system. The
scalar `_sem_modes` path is structurally unusable at `phi!=0`; the solve must go through a coupled
operator everywhere.

The upshot for the Jones matrix: at `phi=0` the off-diagonal Jones entries are nonzero **only** from
material anisotropy (`eps_xy`); at `phi!=0` they also acquire a **purely geometric** contribution
from `ky0`. That geometric cross-polarization is the physics exp12 is trying to measure.

---

## 3. Current 1-D architecture and every `phi=0` assumption to break

Files: `pmm/oned.py` (entry points), `pmm/stack.py` (`PMMStack`), `pmm/_core.py` (numerics). The
scalar and Jones solves share the `_pmm_solve_core` / `_pmm_jones_solve_core` skeleton. Confirmed:
**no `ky0`/`k_y`/azimuth anywhere in the 1-D path** (every `ky0` hit is in `twod*.py`/`stack2d.py`;
every 1-D `phi` is the Lagrange basis `phi_i`, the *slant*-frame angle, or a doc note that the mount
fixes azimuth 0).

Stage-by-stage, with the assumption that breaks:

| # | Stage | Location | `phi=0` assumption baked in | Conical change |
|---|---|---|---|---|
| 1 | k-vector setup | `_core.py:766, 1133`; `stack.py:638` | only scalar `kx0 = Re(n_sup)*sin(angle)*k0` computed | add `ky0 = Re(n_sup)*sin(theta)*sin(phi)*k0` |
| 1 | order grid | `_core.py:809, 1187`; `stack.py:851` | `kx` per order; **no `ky` array** | add `ky` (for 1-D: the constant `ky0`) |
| 2a | scalar modal eig | `_sem_modes` `_core.py:482-534` | TE/TM solved as two **independent scalar** ODEs via `polarization` | scalar path invalid; must solve the coupled operator |
| 2b | tensor modal eig | `_sem_modes_tensor` `_core.py:1003-1008` | builds a 2n `Mbig`/`Q` with **`Ky` hard-zeroed** (comment "the (Ky=0) Q block", `:1007, 875`) | restore the `Ky` blocks -> full 4-field system |
| 2b | tensor->scalar reduction | `_t3` `_core.py:1123-1125` | reads only `exx,exy,eyx,eyy,ezz`; **drops `exz/eyz/ezx/ezy`** | must retain (route through the 4n generator) |
| 2b | tensor mass build | `_build_sem_tensor` `_core.py:922-950` | no y-derivative masses assembled | add `Ky`-weighted operators |
| 2b | 4n metric generator | `_cov_generator_4n` `_core.py:3873-3905` | cross-blocks ride only `cos*(Dop + i*kx0)` (`:3901`); `cos` is *slant*, not azimuth | add the `i*ky0` terms |
| 3 | uniform half-space eig | `_uniform_geo_eig` `_core.py:1046, 1083` | 2n problem block-diagonalizes into **two identical n-blocks** (true only at `ky0=0`) | the two blocks differ under `ky0` |
| 3 | forward/back selection | flux selector `_core.py:1023, 1097` | `Sz = Im(Ex*Hy - Ey*Hx)` omits the `ky0` flux term | include the `ky0` Poynting contribution |
| 4 | half-space `kz` | `_kz_forward` `_core.py:615-619` | `kz = sqrt(eps - kx^2)` (no `-ky^2`) | `kz = sqrt(eps - kx^2 - ky^2)` |
| 6 | Jones assembly | `_assemble_jones_farfield` `_core.py:670-672` | longitudinal `rz = -(kx*rx)/kz`; incident `E_y` treated as pure-s (`Ez_inc=0`) | `rz = -(kx*rx + ky*ry)/kz`; **both** columns carry `Ez_inc` |

**What does NOT change** (Stage 5): the Redheffer star product `_redheffer_star` (`_core.py:603-611`)
and the interface S-matrix `_interface_smatrix` (`_core.py:585-591`) are **polarization-blind** — they
operate on square blocks of whatever size. The only effect of conical here is that the cheap
per-polarization scalar cascade (block size `n_glob`) can no longer exist; every layer runs at the
coupled block size. The star algebra itself is reused verbatim.

---

## 4. The reference: the conical core is already basis-agnostic

The critical discovery. RCWA's conical solve (`rcwa/stack.py`, `rcwa/_core.py`) is built from
functions that take **generic dense operators** `Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ, EZX, ...` — they do
**not** care whether those operators are Fourier/Toeplitz (RCWA) or SEM mass-matrix projections (PMM).
The 2-D PMM already proves this: `pmm/twod_jones.py` builds SEM-basis operators and hands them to the
**same** `_layer_eigenmodes_tensor` etc. So the reusable core is:

- **Angle -> k** (copy verbatim): `kx0 = nre*sin(theta)*cos(phi)`, `ky0 = nre*sin(theta)*sin(phi)`,
  `kz_inc = Re(sqrt(eps_sup - kx0^2 - ky0^2))` — `rcwa/stack.py:1756-1760, 1892`; PMM mirror
  `pmm/twod_jones.py:430-431, 443`.
- **Coupled isotropic eig** (`Q`, `P`, `Omega^2 = P@Q`, size 2N): `_layer_Q_matrix`
  (`rcwa/_core.py:749-765`), `_layer_eigenmodes` (`:1146-1230`). At `Ky=0` the four blocks are
  block-diagonal (the planar fast path); `Ky!=0` turns on the `Kx@Ky` cross-blocks that couple s/p.
- **Coupled full-tensor eig** (4N generator `G = [[A,P],[Q,B]]`, Li-2003): `_layer_eigenmodes_tensor`
  (`rcwa/_core.py:1887-1980`). `ky0` enters every `Ky` contraction in `P/Q/A/B`; `A` carries
  `EZX/EZY`, `B` carries `EXZ/EYZ`. This is the same 4-field structure as the 1-D `_cov_generator_4n`,
  but with the `ky0` terms present.
- **Forward/backward flux selection** in a possibly non-orthogonal basis: `_select_forward_flux`
  (`rcwa/_core.py:1806-1883`) — net-Poynting-z over all harmonics with deep-decay/deep-noise
  overrides. Its comments (`:1848-1858`) were **written for the PMM-2D generator** — i.e. it is
  already validated on the exact non-orthogonal SEM basis the 1-D port will use.
- **Conical half-spaces**: `_homogeneous_eigenmodes(Kx, Ky, eps)` — `kz = sqrt(eps - kx^2 - ky^2)`,
  `W = I` (a Rayleigh order is its own mode), `V = Q @ diag(1/lam)` (`rcwa/_core.py:1234-1259`).
  Because `W=I`, modal amplitudes are already the tangential-E lab harmonics.
- **S-matrix + Jones** (fully basis-agnostic): `_interface_smatrix(_general)`, `_redheffer_star`,
  `_propagation_star`, and the Jones tail — `rcwa/_core.py:1277, 1313-1337`; `rcwa/stack.py:1890-1932`.
  Zeroth-order 2x2 Jones is literally `[[conj(rx0|Ex), conj(rx0|Ey)], [conj(ry0|Ex), conj(ry0|Ey)]]`
  (the `conj` maps internal `exp(+iwt)` to the public `exp(-iwt)` gauge). No s/p->xy rotation because
  `W=I`.

**Basis-specific piece the PMM supplies (the only real new code):** the momentum and material
operators.
- RCWA momentum: `Kx = diag(kx0 + m*wl/px)`, and for **1-D conical `Ky = ky0*I`** (`rcwa/stack.py:1774`).
- PMM momentum (2-D form, `pmm/twod_jones.py:273-274`):
  `GxF = Gx0F/k0 + kx0*Ip`, `GyF = Gy0F/k0 + ky0*Ip`. For a **1-D conical** grating the y-axis has no
  subsections, so `Gy0F -> 0` and this collapses to `GyF = ky0*Ip` — the exact analog of RCWA's
  `Ky = ky0*I`.
- PMM material operators: the Toeplitz/Li convolution matrices become **SEM mass-matrix projections**
  (`CxxF, CxyF, ..., EZZ, EZX, ...`), e.g. `pmm/twod_jones.py:149-309`. The `ezz`-Schur reduction must
  be done **pointwise per region BEFORE projection** (`pmm/twod_jones.py:181-187`; the same discipline
  as `rcwa/_core.py:1780-1784`) — doing it spectrally is the documented "gen2 trap" (`twod_jones.py:176-180`).

Net porting statement: **build `GxF/GyF` (with `+kx0*Ip`, `+ky0*Ip`) and the projected material
operators in the SEM basis, then hand them to the unmodified `_layer_eigenmodes(_tensor)` +
`_homogeneous_eigenmodes` + `_interface_smatrix(_general)` + `_redheffer_star` + Jones chain.** The 2-D
PMM already does exactly this (`twod_jones.py:308`); the 1-D port is the same pattern with the y-axis
reduced to the single `ky0*Ip` term.

---

## 5. Path A — PMM2DStack bridge (capability available today)

`PMM2DStack` is genuinely conical and can run a y-invariant 1-D grating **now**, with no library
change:

- `set_source(wavelength, *, theta, phi, angle)` — `stack2d.py:395-407`; `kx0/ky0` from `(theta,phi)`
  at `stack2d.py:471-472`; shared Rayleigh grid `kxv/kyv` at `:497-498`.
- A cell with walls **only in x** (uniform in y) hits the **separable exact-diag-in-y** fast path: the
  y-operators are exact `diag(k)` (no projection floor), y-momentum conserved to machine precision.
  Validated at `phi=0` by `tests/unit/test_v5_14_0_pmm2d_conical.py:104-134`
  (`test_y_uniform_cell_reduces_to_1d_oblique`: forbidden `n_y!=0` orders < 1e-12, matches native
  `pmm_efficiency_1d`).

**Usage sketch for exp12** (build the exp10/exp11 geometry into a y-invariant 2-D cell, or kron-tile
the existing 1-D cell along y):

```
st = PMM2DStack(period_x=P, period_y=P_any, ...)   # y-invariant cell
J  = st.set_source(pt.WL, theta=deg2rad(tilt), phi=deg2rad(azimuth)).solve().jones_reflection()
side = pt.outcouple(J)
```

**Costs / gotchas (call these out in exp12):**
1. You pay for a full 2-D order set `(2*n_orders+1)^2` though only the `n_y=0` row carries power ->
   `O(N^2)` Fourier orders and a larger modal eig than a native 1-D conical solver's `O(N)`.
2. `period_y` (defaults to `period_x`, `stack2d.py:87`) is arbitrary for a y-invariant cell — it only
   positions the empty `n_y!=0` orders.
3. The published reduction test is at `phi=0`. **Add a `phi!=0` test** (a y-invariant cell vs a
   Berreman/RCWA conical reference) before trusting it — that missing test is the single gap.

Path A is the right way to unblock exp12's out-of-plane cut immediately; it also becomes the
same-family regression oracle for Path B.

---

## 6. Path B — native 1-D conical PMM (the O(N) reduction)

A dedicated conical 1-D entry point, built as the `n_orders_y = 0`, exact-diag-in-y reduction of the
`PMM2DStack.solve` build (`stack2d.py:412-654`). Proposed phasing:

**Phase 0 — plumb `ky0`.** Add `phi`/azimuth to `_resolve_incidence` and the three `kx0` sites
(`_core.py:766, 1133`; `stack.py:638`); compute the scalar `ky0`; add `ky` (the constant `ky0`) beside
every `kx` order array (`_core.py:809, 1187`; `stack.py:851`). Update `_kz_forward` to accept `ky`
(-> `sqrt(eps - kx^2 - ky^2)`), or route half-spaces through the existing `_kz_forward2`-style path.
*Guard:* when `ky0 == 0`, dispatch to the existing classical path unchanged (zero regression risk).

**Phase 1 — isotropic conical layers.** For `ky0 != 0`, stop using scalar `_sem_modes`. Assemble the
SEM `GxF = Dop/k0 + kx0*Ip` and `GyF = ky0*Ip`, build the isotropic `Cxx=Cyy=eps-projection`, and call
the **shared** coupled path. Two implementation options, in order of preference:
  - (a) **Reuse the 2-D SEM builders directly** with a degenerate y-axis (single node): call
    `pmm/twod_jones.py` `_layer_modes_projected` / `_tensor_layer_modes` with `ky0` and a 1-node y —
    this is the least new code and inherits the 2-D validation.
  - (b) Extend the existing 1-D **4n metric generator** `_cov_generator_4n` (`_core.py:3873-3905`) by
    adding the `i*ky0` cross-terms it currently omits, then feed the generalized cascade
    (`stack.py:790-825`) that already handles 4n blocks. This keeps everything in the 1-D module but
    duplicates logic the 2-D path already has.
  Recommend (a): maximal reuse, and the flux selector, generalized S-matrix, and Jones tail are then
  literally the 2-D code.

**Phase 2 — full-tensor (LC) conical layers.** Route the off-plane tensor through
`_layer_eigenmodes_tensor` (the 4N `G=[[A,P],[Q,B]]` generator) with SEM-projected `EZZ/EZX/EZY/EXZ/EYZ`
and `ky0` present. Preserve the **pointwise-per-region `ezz`-Schur before projection** discipline
(`twod_jones.py:181-187`). This is exactly the exp10/exp11 LC path.

**Phase 3 — half-spaces + Jones under conical.** Use `_homogeneous_eigenmodes(Kx, Ky, eps)` for the
super/substrate (or the existing `_sem_modes_uniform` extended so its two n-blocks differ under
`ky0`). Fix the Jones longitudinal term to `rz = -(kx*rx + ky*ry)/kz` and give **both** incident
columns the `Ez_inc` term (`_core.py:670-672`). Because region `W=I`, the 2x2 assembly is copied
unchanged.

**Phase 4 — `PMMStack` wiring + public API.** Add `phi=` to `PMMStack.set_source` and the 1-D
`pmm_jones_1d`/`pmm_efficiency_1d` signatures; when `phi!=0`, force the generalized (coupled) cascade
(`stack.py:790-825`) exactly as an out-of-plane *tensor* layer already forces it today.

**Non-goals / do-not-touch:** the Redheffer star and interface S-matrix (already polarization-blind);
the slant/covariant machinery (orthogonal concern — but note conical + slant would need the `ky0` and
`tan(phi_slant)` convections to compose, which is out of scope here and should raise until tested).

---

## 7. Validation strategy

Build the gates in this order; each is a superset check of the last.

| Gate | Oracle | Exact callable | Validates | Evidence |
|---|---|---|---|---|
| G0 | **classical regression** | existing `pmm_jones_1d(phi=0)` | `ky0=0` path byte-identical to today | — |
| G1 | **Berreman 4x4 conical** (analytic, tightest) | `berreman_jones_1d(layers, n_sub, n_sup, wl, theta=, phi=)` / `BerremanStack` | conical **half-space + modal wiring** at duty=1 / uniform cell (< 1e-8) | `berreman.py:259-320` (`Kx,Ky` from `phi`, `:286, 315-316`); pattern in `test_v5_14_0_pmm2d_oop.py:56-69` |
| G2 | **PMM2DStack conical** (same-family) | `PMM2DStack.set_source(theta, phi).solve()` on the y-invariant cell | the grating projection + reduction, same numerics family | `stack2d.py:395-407`; reduction test pattern `test_v5_14_0_pmm2d_conical.py:104-134` |
| G3 | **RCWAStack conical** (cross-method) | `SegmentStackGeometry.to_rcwa_stack(...).set_source(theta, phi).solve()` | independent method, patterned grating, in-plane tensors; Cu at adequate `n_orders` | `rcwa/stack.py:1304-1322, 1759-1760` |
| G4 | **in-plane continuity** | existing classical `PMMStack(angle=theta)` vs new conical at `phi=0` sweeping `theta` | the two engines agree along the shared `phi=0` cut (this is also exp12's built-in cross-check) | Sec. 4 |

**Traps to encode as review gates (from prior OOP-tensor work):**
- **`rcwa_jones_1d` / `rcwa_efficiency_1d` are NOT conical oracles** — they hardwire `Ky=0`
  (`rcwa/oned.py:519, 1152, 1337`). Validating conical against them silently compares the wrong
  geometry. Use `RCWAStack` / `rcwa_jones_2d` / Berreman(`phi=`) only.
- **Never gate a non-reciprocal / non-Hermitian OOP tensor on energy = 1.** A lossless cell
  auto-balances power even with a wrong per-order split, and `eps_xz != eps_zx` gives total power
  physically != 1. Gate against the oracle's **per-order** value, not energy closure
  (`test_v5_14_0_pmm2d_oop.py:114-129`; `test_v5_12_0_pmm_covariant_oblique.py:175-200`).
- **Forward-flux selection under conical** in the non-orthogonal SEM basis is the main numerical risk;
  `_select_forward_flux` already has the deep-decay/deep-noise overrides for this
  (`rcwa/_core.py:1859-1868`), but new conical modes should be spot-checked for mis-classified
  near-grazing evanescent pairs.
- **Lossy Cu:** use the **Li inverse rule** for the wall-normal permittivity (`formulation='li'`);
  Laurent + TM on metals may never converge. **ASR (`asr_eta`) is normal-incidence only**
  (`rcwa/oned.py:541-544`) — unavailable under conical; rely on `stabilize` + `li` + adequate orders.

---

## 8. Effort, risk, recommendation

**Effort (revised down from the initial "multi-week research problem"):**
- Path A (PMM2DStack bridge + `phi!=0` reduction test): **~0.5 day.** Immediate exp12 capability.
- Path B (native 1-D conical, Phases 0-4 + G0-G4 gates): **~3-6 days**, dominated by wiring the SEM
  operators into the shared coupled path (Option 6a) and the validation matrix — not by novel solver
  math, which is already done and shared.

**Risk: low-to-moderate.** The physics and the coupled machinery are proven (RCWA + 2-D PMM both ship
it; the flux selector is already validated on the SEM basis). Residual risks are bookkeeping: the
`ezz`-Schur-before-projection ordering, the Jones `Ez_inc`/`rz` conical terms, forward-flux edge
cases near grazing, and Cu convergence at oblique — all covered by the gate ladder above.

**Recommendation.**
1. **Now:** implement Path A and add the `phi!=0` reduction test (G2 vs G1). This unblocks exp12's
   out-of-plane cut immediately and gives the same-family oracle.
2. **Then:** implement Path B (Option 6a — reduce the 2-D SEM builders) if the `O(N^2)` cost of Path A
   proves limiting for production sweeps, or if a first-class conical `pmm_jones_1d` is wanted for the
   library. Path B is a reduction, not research.

For exp12 specifically, **Path A is sufficient** — the y-invariant PMM2DStack at `phi!=0`, validated
against Berreman/RCWA, gives trustworthy out-of-plane numbers without waiting on the native solver.

---

*Prepared 2026-07-03. Source-verified against the current tree by a 3-agent read-only sweep
(current-PMM internals, RCWA conical reference, prior-OOP/oracles). No code modified. Companion
consumer: the exp12 oblique-angle study of the exp10/exp11 LC-QWP out-couplers.*
