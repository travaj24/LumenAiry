# AUDIT — Element-size-aware scaling for PMM (fix the thin-feature `Singular matrix`)

**Date:** 2026-06-03
**Component:** `lumenairy/elements/pmm.py` (spectral-element modal solver — scalar + tensor, segment + stack paths)
**Severity:** High (hard failure) — blocks PMM on an entire device class (conformal thin coatings / barriers, tapered staircases)
**Status:** ✅ **IMPLEMENTED + VALIDATED 2026-06-03** (folded into the untagged 5.11.0). See the implementation note below.
**Found via:** external use (`pbs_qwp_mirror_sim`) — a Ta/Al₂O₃-coated, 2°-tapered Cu/SiCN LC out-coupler. PMM is `Singular`; RCWA/FMM solves it fine at ~40 s/solve. The motivation for the fix is to recover PMM's ~10–100× speed for this resonant device class.

---

> ### ✅ Implementation note (2026-06-03)
> The fix was verified against the source first (root cause confirmed at `pmm.py:623-646` Jacobian scaling, `:671` `iS0=inv(S0)`, `:879-897` segment builder) and is shipped. **Both parts implemented, with three corrections to this audit's scope:**
> - **(A) Equilibration** is GATED on an ill-scaling test (`_ill_scaled`, `cond >~ 1e12`) so every well-scaled geometry takes the **plain, bit-identical** path — applied via `_safe_inv` / `_safe_solve` / `_safe_geig`. *(Unconditional application, as originally proposed, is only `≤1e-10` not bit-identical; the gate preserves the repo's bit-identical policy with zero regression.)*
> - **Scalar-path scope corrected:** this audit's §4 pointed at lines ~405/422/423 as "`_sem_modes` inversions" — those are actually `_interface_smatrix`/`_redheffer_star` (the S-matrix algebra). The *real* scalar SE sites are **`pmm.py:338` `np.linalg.solve(S0, Pinv)`** (the `invop`, a *solve* not an inv → handled by `_safe_solve`) **and the generalized eig `sla.eig(A, B)` with `B ∝ J`** (→ `_safe_geig`, congruence-equilibrated). Both were patched.
> - **(B) Wall-merge** was implemented **only in the `PMMStack` union grid** (`_pmm_union_grid`), not the single-layer `_segment_elem_bnds`: `_segment_walls` already rejects `width ≤ 0`, and a near-coincident wall *within one layer* is the user's intentional geometry (merging it would silently alter their structure) — whereas near-coincident walls *across layers* in the union are the genuine artifact.
>
> **Validation:** all **104 PMM tests pass** (8 new in `test_v5_11_0_pmm_element_size_scaling.py` + 96 existing, well-scaled paths bit-identical). The degenerate-wall reproducer (two layers' shared wall offset by `1e-11·P`) goes `Singular` → finite/physical/energy-conserving, and matches the exactly-aligned result to `0.0` (the spurious sub-pm wall is merged away). The §C eigenproblem balancing was left as the documented optional hardening (not needed once 3A+3B are in).

---

## 0. TL;DR

`pmm_jones_1d*`, `pmm_efficiency_1d*`, and `PMMStack` raise `numpy.linalg.LinAlgError: Singular matrix` (or return non-physical modes) whenever the in-plane geometry contains **elements whose widths span a large ratio** — e.g. a 1 nm Ta liner and a 6 nm Al₂O₃ coating sitting next to ~100–500 nm regions, and/or a **tapered z-staircase** whose per-slice walls land ~1–2 nm apart (occasionally coincident) on the shared nodal grid.

**Root cause:** the spectral-element operators carry the element Jacobian `J = (x_r − x_l)/2`. `Dphys = Dref/J`, `Kloc ∝ 1/J`, and the mass `S0 ∝ J`. With widths from ~1 nm to ~500 nm, `S0` (the matrix that is explicitly inverted) has a condition number `≳ w_max/w_min`, compounded across the union grid to the point of numerical singularity. This is the classic FEM **sliver-element** conditioning pathology — **independent of wavelength** (all features can be deeply subwavelength and it still fails; the failure is in the *spatial-discretization conditioning*, not in modal resolution).

**Fix (two parts):**
- **(A) Symmetric diagonal equilibration** of every SE matrix inversion (`inv(S0)`, `inv(Cinv_xx)`, the scalar S-matrix inversions). For `S0` this is **mathematically exact** (regression-safe) and removes the element-size term from the conditioning.
- **(B) Near-coincident-wall merge / zero-width-element drop** when building `elem_bnds` and the `PMMStack` union grid, so truly-degenerate (`J≈0`) elements — which equilibration alone cannot rescue — never form. This is the dominant failure mode for *tapered* staircases.

**Payoff:** unlocks PMM for conformal-coating / barrier-liner devices and tapered stacks (today RCWA-only). For the motivating device this is ~40 s/solve (FMM) → ~0.1–2 s/solve (PMM).

---

## 1. Symptom (reproducible)

```
File ".../lumenairy/elements/pmm.py", line 671, in _sem_modes_tensor
    iS0 = np.linalg.inv(S0)
numpy.linalg.LinAlgError: Singular matrix
```

Observed for the Ta/Al₂O₃-coated device for **every** config tried — air *and* LC superstrate, vertical *and* 2° tapered walls, `n_slice ∈ {1,2,4}`, Ta ∈ {1,10} nm. When the inverse happens to *not* throw (a borderline-conditioned case), the returned Jones is non-physical (`|out-coupling| ~ 10¹⁰`), i.e. silently wrong — so a guard alone is not enough; the conditioning must be fixed.

Empirically the trigger is narrower than "thin features" alone: a **single thin segment solves fine**, and so does a **vertical (aligned-wall) multilayer stack** — it is the **tapered multi-slice `PMMStack` union grid** that fails. The per-slice wall offsets (`dz·tan θ`) pile up near-coincident walls; `nB ≥ 2` slices *with* a 2° taper → `Singular`/`nan`, while `nB = 1` or `taper = 0` pass (§5.1).

---

## 2. Root cause (with code references)

SE assembly — `_build_sem_tensor` (≈ line 621), identical structure in `_build_sem`, `_build_sem_segments`, `_build_sem_tensor_segments`:

```python
for e in range(n_el):
    xl, xr, t = elem_bnds[e]
    J     = 0.5 * (xr - xl)        # element Jacobian (half-width)
    wel   = ref_w * J              # quadrature weights        ∝ J
    Dphys = Dref / J               # physical derivative        ∝ 1/J
    Mloc  = np.diag(wel)           # local mass                 ∝ J
    Kloc  = (Dphys.T * wel) @ Dphys   # local stiffness         ∝ 1/J
    ...
return dict(..., S0=mass["one"], ...)     # S0 = unit-weight global mass  ∝ J
```

`_sem_modes_tensor` (≈ line 670):

```python
S0  = mats["S0"]
iS0 = np.linalg.inv(S0)            # <-- throws
...
Cxx = np.linalg.inv(Cinv_xx)       # second SE inversion (also size-sensitive)
...
q2, W2 = np.linalg.eig(Mbig)       # Mbig carries 1/J^2 in the stiffness blocks
```

- A node interior to a width-`w` element contributes `diag(S0) ∝ w`. So `cond(S0) ≳ w_max / w_min`. For 1 nm vs 500 nm regions that is ~5×10². The downstream eigen-operator `iS0 @ stiff / k0²` carries **`1/J²`** (physical high-order evanescent content of the tiny element), so the *assembled* operator spreads ~`(w_max/w_min)²` ≈ 2.5×10⁵.
- **The `PMMStack` union grid amplifies this.** All layers are solved on the union of every layer's walls. A 2° taper shifts each tooth wall by `dz·tan(2°)` per z-slice (~1–2 nm for typical slicing); the Ta wall sits 1 nm and the Al₂O₃ wall 6 nm from each Cu edge. The union therefore contains many ~1 nm elements and, where walls from different slices nearly coincide, **sub-nm or zero-width** elements → `J → 0` → a ~zero row in `S0` → exactly singular.
- **Not a wavelength problem.** Every feature here is ≪ λ (λ = 1310 nm). PMM is not limited by modal resolution of these features; it is limited by the **conditioning of the real-space spectral discretization**. (Contrast FMM/RCWA: a global Fourier basis with *no* spatial elements — a thin feature costs more *orders* but never produces a singular inverse. This is precisely why RCWA solves the same device cleanly.)

---

## 3. The fix

> **Priority (from the isolation in §1/§5.1):** the confirmed trigger is **near-coincident walls in the tapered union grid** (degenerate, `J≈0` elements), so **§3B is the operative fix** for it — and equilibration alone cannot rescue a genuinely zero-width element. **§3A (the element-size-aware scaling you asked about) is the deeper conditioning robustness** — it removes the `1/J`/`1/J²` scaling from the inversions, hardening the genuinely-thin-but-finite elements (1 nm Ta, 6 nm Al₂O₃) and the borderline cases, and is what keeps PMM accurate (not merely non-throwing) as `degree`/`n_slice` grow. **Implement both.**

### 3A. Symmetric diagonal equilibration of the SE inversions (primary)

Replace each `np.linalg.inv(A)` in the SE solver with an equilibrated inverse. Add one helper:

```python
def _equilibrated_inv(A):
    """inv(A) computed after symmetric diagonal (Jacobi) equilibration.
    For A with a real-positive diagonal (the SE mass S0) this is EXACT:
        inv(A) = Di @ inv(Di A Di) @ Di,  Di = diag(1/sqrt(diag A)),
    but the matrix actually inverted (Di A Di) has unit diagonal, so its
    condition number is O(degree^2) instead of O(w_max/w_min). Regression-safe:
    identical (to round-off) for the well-conditioned geometries in use today."""
    d = np.sqrt(np.abs(np.diag(A)))
    d = np.maximum(d, d.max() * 1e-13)         # floor near-zero nodes (degenerate guard)
    di = 1.0 / d
    Ah = (di[:, None] * A) * di[None, :]
    return di[:, None] * np.linalg.inv(Ah) * di[None, :]
```

Apply it at:
- `_sem_modes_tensor`: `iS0 = _equilibrated_inv(S0)` (line ~671) and `Cxx = _equilibrated_inv(Cinv_xx)` (line ~676).
- `_sem_modes` (scalar TE/TM): the three inversions at lines ~405 (`iapb`), ~422, ~423.
- Any other `np.linalg.inv` on an SE-assembled (period-grid) operator.

Notes:
- `S0 = mass["one"]` has a **real, positive** diagonal (`ref_w·J`, no ε), so equilibration is exact and `inv(S0)` is reproduced to machine precision — existing results are unchanged.
- For complex operators (`Cinv_xx`, scalar S-blocks) the `sqrt(|diag|)` scaling is a (very effective) heuristic, not exact — still a similarity-type rescale that slashes the condition number; verify against the regression set (§5).
- Prefer this over a blanket `pinv`/Tikhonov: equilibration keeps the answer exact for good cases and well-defined for bad ones, where a regularizer would perturb the physics.

### 3B. Near-coincident-wall merge / zero-width-element drop (companion, geometry side)

In `_segment_elem_bnds` (≈ line 879) and the `PMMStack` union-grid construction, after gathering all wall positions for a layer (and, for the stack, the union over layers):

```python
tol = max(1e-12, 1e-6 * period)            # sub-pm; physically negligible
walls = np.sort(walls)
merged = [walls[0]]
for w in walls[1:]:
    if w - merged[-1] > tol:
        merged.append(w)
    # else: snap to the existing wall (merge)
# build elements from `merged`; assign each element the material of its midpoint;
# never emit an element with (x_r - x_l) <= tol.
```

This eliminates the `J≈0` elements that equilibration cannot save (the exact-singular case from tapered staircases). Merging walls below `tol` perturbs the geometry by < 1 pm → no physical effect; document the tolerance.

### 3C. (follow-on, optional) eigenproblem balancing

`np.linalg.eig(Mbig)` can still lose digits on the small (propagating) eigenvalues because the stiffness blocks legitimately carry `1/J²`. After 3A+3B the singular failure is gone; if residual accuracy loss is seen on extreme grids, balance `Mbig` (e.g. `scipy.linalg.eig` performs LAPACK balancing, or apply a manual two-sided diagonal balance) before the eig. Not required to fix the singularity — list as a hardening step.

---

## 4. Scope — functions to patch

| Path | File location | Change |
|---|---|---|
| tensor / Jones modes | `_sem_modes_tensor` (≈651–719) | `inv(S0)`, `inv(Cinv_xx)` → `_equilibrated_inv` |
| scalar TE/TM modes | `_sem_modes` (≈304–) | 3 inversions (≈405, 422, 423) → `_equilibrated_inv` |
| grid construction | `_segment_elem_bnds` (≈879) + `PMMStack` union builder | wall-merge + zero-width drop (3B) |
| (callers — no change) | `pmm_jones_1d`, `pmm_efficiency_1d`, `*_1d_segments`, `PMMStack` | inherit the fix |

---

## 5. Validation

### 5.1 Minimal reproducer — the taper × multi-slice union (CONFIRMED trigger)

Empirically isolated (this experiment): a single thin segment is fine, and an *aligned* (vertical) multilayer stack is fine — **the failure needs the `PMMStack` union grid of a TAPERED staircase**. Each z-slice's walls are offset by ≈ `dz·tan(θ)`, so the union contains near-coincident walls and, where slices cross, sub-nm/zero-width elements. Confirmed minimal case (degree 10):

```python
import numpy as np
from lumenairy.elements.pmm  import PMMStack
from lumenairy.elements.rcwa import uniaxial_tensor

P, wl = 550e-9, 1310e-9
ecu, eta, eal = -83.13+2.70j, -147.79+24.50j, 2.762            # Cu, Ta(Werner), Al2O3
elc = uniaxial_tensor(1.5, 1.8, np.pi/2, phi=np.radians(90.0)) # LC tensor (n_o,n_e)
wc, wf, g, H, t, ta, al = 130., 220., 100., 350., 70., 1., 6.  # nm; ta=Ta, al=Al2O3
tn = np.tan(np.radians(2.0))                                   # 2 deg taper  (tn=0 -> PASSES today)
hg = lambda z: wc/2 - (H-z)*tn; hf = lambda z: wf/2 - (H-z)*tn; gz = lambda z: g + 2*(H-z)*tn
norm = lambda wm: [(w/sum(x for x, _ in wm), m) for w, m in wm]

st = PMMStack(P, n_substrate=np.sqrt(ecu), n_superstrate=1.5, degree=10)
nB = 4; dz = (H - t) / nB                                       # nB=1 -> PASSES today
for k in range(nB):                                            # 12-seg coated teeth, tapered slices
    z = H - (k + 0.5) * dz; a, b, c = hg(z), hf(z), gz(z); lcw = c - 2*(ta + al)
    st.add_layer(dz*1e-9, segments=norm([(2*a, ecu), (ta, eta), (al, eal), (lcw, elc), (al, eal), (ta, eta),
                                         (2*b, ecu), (ta, eta), (al, eal), (lcw, elc), (al, eal), (ta, eta)]))
st.set_source(wl)
J = st.solve()[3]      # BEFORE fix: LinAlgError(Singular)  or  |J| ~ 1e23 / nan
                       # AFTER  fix: finite, physical (|J| <= ~1), energy-conserving
```

Controls that **pass today** (so they pin the trigger to the taper-union, not the thin segments per se):
`nB = 1` (single slice) → PHYSICAL · `tn = 0` (vertical, identical walls every slice) → PHYSICAL.

Reference: FMM solve of the *identical* stack (`RCWAStack` with the pixelated tapered cells) — `pbs_qwp_mirror_sim/validation/check_coated_rcwa.py`, energy-exact (R+A=1.000).

Acceptance:
1. No `LinAlgError` and `|J|` finite/physical for `nB ≥ 2` with the 2° taper.
2. `cond(S0)` (and the equilibrated `S0_hat`) bounded (< 1e8) across the union grid; **no zero-width elements** emitted by the grid builder.
3. PMM ≈ FMM: out-coupling (or Jones) within a few % at converged `degree`/`n_orders`.
4. Energy conserved: `R + T + A = 1 ± 1e-6`.
5. **Degenerate-wall unit test:** two layers whose shared wall differs by `1e-12·P` → §3B must merge/drop it and return the un-perturbed result.

### 5.2 Full target — the motivating device, RCWA reference

The Ta/Al₂O₃-coated, 2°-tapered Cu/SiCN device via `PMMStack`, checked against the **already-validated, energy-exact RCWAStack** result:

- Geometry: `w_c=130, w_f=220, g=100, H=350, t=70` nm; Ta=1 nm, Al₂O₃=6 nm; superstrate isotropic n=1.50; McPeak Cu, SiCN n=1.781, Ta(Werner), Al₂O₃ n=1.662.
- RCWA reference (energy-exact, R+A=1.000): **peak (φ=0) 71.5 %, null (φ=90) 6.2 %, extinction ~12:1** for Ta=1 nm (and 56.3 / 4.6 / 12:1 for Ta=10 nm). Reference builder: `pbs_qwp_mirror_sim/src/pmm_taper.py::coated_tapered_jones` (RCWAStack path) and `validation/check_coated_rcwa.py`.
- After fix: `PMMStack` reproduces the out-coupling-vs-φ curve to within a few % (cross-method, as for the bare device), **non-singular**, energy-conserving, and **≫ faster** than the RCWA reference.

Acceptance:
1. No `LinAlgError` for any φ, any `n_slice`, vertical or 2° tapered.
2. `cond(S0_hat)` bounded (< 1e8) across the union grid.
3. PMM ≈ FMM within a few % on peak/null (same tolerance band as the existing bare-device PMM-vs-FMM agreement).
4. Energy conserved.
5. **Regression (critical):** all existing PMM results are unchanged to ≤ 1e-10 — in particular the bare tapered device (`pmm_jones_1d` / `pmm_jones_1d_segments` on chunky geometries) and the current PMM test suite. The `S0` equilibration is exact, so this should hold to round-off; CI must assert it.

### 5.3 Repro / reference artifacts already on disk (external)
- `pbs_qwp_mirror_sim/validation/pmm_min_singular_repro.py` — the **minimal** reproducer of §5.1; prints `GARBAGE (|J|=5.0e+23)` for the nB=4/2° trigger and `PHYSICAL` for both controls (nB=1, and vertical). Use it as the before/after gate for the fix.
- `pbs_qwp_mirror_sim/validation/check_pmm_coated_air.py` — the full coated PMM stack; `Singular matrix` for all configs (air and LC superstrate).
- `pbs_qwp_mirror_sim/validation/check_coated_rcwa.py` — the FMM (`RCWAStack`) reference of the same device, energy-exact.

---

## 6. Alternatives considered (and why not)

- **Uniform-size subdivision** (split big regions into ~6 nm elements so all elements match the thinnest): well-conditioned, but the DOF count explodes and PMM loses the very DOF-efficiency that makes it worth using — ad hoc and self-defeating.
- **Route thin-feature geometries to FMM** (today's state): correct but forfeits the ~10–100× PMM speedup, which is the entire reason to want PMM for these sharp-resonant devices.
- **Regularize the inverse** (`pinv` / Tikhonov on `S0`): would change results for good geometries and perturb the physics; equilibration is exact where it matters and strictly better-conditioned where it doesn't.

---

## 7. References

- M.-A. Edee, *Modal method based on subsectional Gegenbauer polynomial expansion* (PMM), JOSA A 28 (2011).
- Spectral/finite-element **sliver-element conditioning**: standard FEM result (Babuška–Aziz; diagonal/Jacobi and Ruiz equilibration for ill-scaled stiffness/mass systems).
- This fix is the modal-solver analogue of the diagonal scaling already used to robustify ill-scaled linear systems elsewhere in the library.
