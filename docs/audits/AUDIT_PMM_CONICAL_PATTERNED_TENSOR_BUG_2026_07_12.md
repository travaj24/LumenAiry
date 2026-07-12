# PMM Conical — Patterned Anisotropic-Tensor Bug — 2026-07-12

> **STATUS — FIXED (2026-07-12), with an expanded root cause.**  Remediation on
> branch `fix/pmm-conical-patterned-tensor`; gates in
> `tests/unit/test_v5_20_0_pmm_conical.py` (7 new tests).
>
> **The audit's localization was verified and found INCOMPLETE in two ways:**
>
> 1. **The scalar path is affected too.**  A patterned ISOTROPIC grating shows
>    the SAME resolution-independent degenerate-limit gap (`||Jco − Jcl||`
>    2–4e-3 for a 50% n=1.7 grating) — the existing scalar reduction test's
>    5e-3 "slow-TM-channel" tolerance masked it.  §4's "NOT affected: scalar"
>    claim is wrong.
> 2. **The root cause is NOT the tensor factorization rule.**  Swapping the
>    in-plane blocks to the correct Li-1996 composite (`fff_nv`), the EZZ rule,
>    or any combination of the two leaves a flat ~2e-3 gap.  The defect is the
>    **Fourier-projected operator build itself** (`_tensor_layer_modes` /
>    `_layer_modes_projected`): compressing the nodal operators through
>    `T · op · pinv(T)` before the eigensolve carries a systematic error that
>    **saturates in `far_field_orders` and GROWS with `degree`** (measured:
>    gap 2.822e-3 identical at ffo=41 and 81; 1.7e-3 → 4.3e-3 over degree
>    6 → 18 at fixed ffo) — a genuine defect, not a convergence floor.
>
> **Fix (matches §5's direction exactly): the pure-NODAL conical cascade.**
> `_sem_modes_tensor` (pmm/_core.py) is generalized with a dimensional `ky0`
> (the full dimension-agnostic P/Q tensor blocks assembled from the SAME
> weak-form mass/stiffness/convection operators; every added term carries a
> `ky0` factor, so at `ky0 = 0` it is bit-identical to the classical solve),
> and a new `_conical_nodal_solve` (pmm/conical.py) runs the classical
> union-grid Redheffer cascade end-to-end in the nodal basis (public gauge),
> closing with the conical vector far field via the nodal→Rayleigh projection.
> `PMMStack._solve_conical`, `pmm_jones_1d_conical`, and
> `pmm_jones_1d_conical_tensor` route every PATTERNED in-plane cell through
> it; a patterned OUT-OF-PLANE cell now raises `NotImplementedError` (the old
> path returned silently-wrong numbers for it); all-UNIFORM cells keep the
> exact Fourier path (Berreman-validated, incl. out-of-plane).
>
> **Validation:** reproducer gap 3.29e-3 → **1.4e-14**; retardance offset
> 3.83° → **0.00°**; scalar degenerate gap → 1.4e-14; lossless energy at
> genuine conical closes to 1e-10; θ→0 continuity is quadratic;
> degree-CONVERGENT at θ=25°, φ=35° (2.3e-5 → 6.2e-6); the independent
> `rcwa_jones_2d` cross-oracle converges TOWARD the nodal answer as its
> orders rise.  All 13 pre-existing conical tests pass (one return-contract
> alignment: the stack keeps its 1-D `(m,)` orders contract).
>
> `PMM2DStack` (hybrid) still uses the projected machinery and remains
> FMM-floored for patterned cells under conical — use `PMM2DStackPure` for a
> no-floor 2-D answer.  The exp12 out-of-plane cut is UNBLOCKED for the
> in-plane LC devices.

**Severity: correctness (silent wrong numbers).** The native conical (out-of-plane, `phi != 0`) 1-D
PMM path — shipped in v5.20.0, `pmm/conical.py` + `PMMStack._solve_conical`, designed per
`AUDIT_PMM_CONICAL_OUT_OF_PLANE_2026_07_03.md` — produces a **systematic error for PATTERNED
anisotropic (tensor) layers**. It is exact for *uniform* anisotropic slabs (the Berreman-validated
v5.20.0 case) but wrong for a *patterned* tensor grating (an anisotropic ridge + groove), even at
**normal incidence** where the conical solve must reduce identically to the classical mount.

Found by the exp12 consumer (oblique-angle study of the exp10/exp11 LC-QWP out-couplers): the
patterned layer is the switchable **LC** slot, so the bug corrupts exactly the device physics.

All file:line anchors relative to
`d:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy\lumenairy\`.

---

## 1. Summary

At `ky0 = 0` (any `theta=0`, or `phi=0`) the conical solve is the **same physics** as the classical
mount, so `PMMStack.set_source(theta=0, phi=90).solve()` MUST equal
`set_source(theta=0, phi=0).solve()` element-for-element.

- **Uniform anisotropic slab:** it does — to machine precision (`||Jco - Jcl|| = 5e-15`).
- **Patterned anisotropic layer (ridge + groove):** it does **not** — `||Jco - Jcl|| ~ 3e-3`, a
  ~3.5 deg reflection-**retardance** offset that is **resolution-independent** (does not shrink as
  `degree`/`far_field_orders` rise), i.e. **systematic, not a convergence gap**.

The classical patterned-tensor path is the validated production path (exp10/exp11) and is converged;
the conical path is the v5.20.0 addition and is the party that fails the required degenerate-limit
reduction. Therefore the **conical patterned-tensor build is wrong**.

**Amplification:** the per-layer retardance error is small, but a multilayer, high-index, deep-null
device compounds it. On the exp11 pillar out-coupler (`355/165/120/H370/t75`, coarse config) the
normal-incidence conical solve returns peak **84.2%** / null **82.2%** (ext ~1:1) versus the correct
classical **75.6%** / **1.47%** (ext 51:1) — the switch is destroyed. The extinction is a ratio at a
deep null (retardance ~180 deg), which is exquisitely sensitive to a few-degree retardance error.

---

## 2. The validated reproducer (library-level, no experiment deps)

One layer, one anisotropic material (`no=1.5, ne=1.7`, director 30 deg), NORMAL incidence, solved two
ways via `PMMStack`. `phi=90` at `theta=0` gives `ky0 = sin(0)*sin(90) = 0`, so it must equal `phi=0`.

```python
import numpy as np
from lumenairy.elements.pmm import PMMStack
WL, P, DEPTH = 1310e-9, 700e-9, 300e-9
def aniso(a, no=1.5, ne=1.7):
    c, s = np.cos(np.radians(a)), np.sin(np.radians(a)); no2, ne2 = no*no, ne*ne
    T = np.diag([no2, no2, no2]).astype(complex)
    T[0,0] = ne2*c*c + no2*s*s; T[1,1] = ne2*s*s + no2*c*c; T[0,1] = T[1,0] = (ne2-no2)*c*s
    return T
lc = aniso(30.0)
st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=8, grade=True, far_field_orders=15)
st.add_layer(DEPTH, segments=[(0.5, lc), (0.5, 1.0+0j)])          # PATTERNED: aniso ridge + air
Jcl = np.asarray(st.set_source(WL, theta=0.0, phi=0.0).solve()[3])
Jco = np.asarray(st.set_source(WL, theta=0.0, phi=np.deg2rad(90)).solve()[3])
# ||Jco - Jcl|| = 3.2e-3   (should be ~1e-15)
```

Swap the layer for `segments=[(1.0, lc)]` (UNIFORM) and `||Jco - Jcl|| = 5e-15` — exact. The ONLY
difference is patterning.

### Convergence (bug vs. convergence gap)

Sweep resolution on the patterned case; watch the classical-vs-conical gap and each retardance
`arg(J00) - arg(J11)`:

| degree/ffo | `||Jco - Jcl||` | retardance classical | retardance conical | `|Δret|` |
|---:|---:|---:|---:|---:|
| 8 / 15 | 3.25e-3 | -48.62 deg | -44.79 deg | 3.83 |
| 10 / 21 | 2.83e-3 | -48.62 deg | -45.22 deg | 3.40 |
| 12 / 25 | 2.74e-3 | -48.62 deg | -45.28 deg | 3.34 |
| 14 / 31 | 2.85e-3 | -48.62 deg | -45.13 deg | 3.49 |
| 16 / 41 | 3.10e-3 | -48.62 deg | -44.82 deg | 3.80 |

The classical retardance is **stable at -48.62 deg** (converged); the conical **plateaus at ~-45 deg**
and does **not** approach it. Flat gap under refinement ⇒ **systematic error**, not discretization.

---

## 3. Localization

- **UNIFORM tensor conical is correct** (machine-precision reduction) ⇒ the conical *half-space
  wiring*, `ky0` setup, S-matrix, forward-flux selection, and Jones extraction are all fine. This
  matches the v5.20.0 Berreman gate (`test_v5_20_0_pmm_conical.py::test_native_conical_matches_berreman_uniform_slab`).
- **PATTERNED tensor conical is wrong** ⇒ the fault is in the **projected patterned-tensor modal
  build**, i.e. the SEM projection of a `(3,3)` tensor cell WITH x-walls under the conical/2-D path.

At `ky0 = 0` the conical tensor build must produce the *same modal operators* as the classical 1-D
tensor branch. It does not. The two builds to reconcile:

- **Conical / 2-D path (suspect):** `PMMStack._solve_conical` routes a tensor layer through
  `_tensor_layer_modes` with `oy=[0]` — `stack.py:700`; def `pmm/twod_jones.py:149-309`
  (projected `Cxx/Cxy/Cyx/Cyy/EZZ/EZX/EZY/EXZ/EYZ`, `ezz`-Schur pointwise-before-projection
  `twod_jones.py:181-187`, then the shared `_layer_eigenmodes_tensor`, `rcwa/_core.py:1887-1980`).
  The single-layer public entry `pmm_jones_1d_conical_tensor` (`pmm/conical.py:179`) is the same path,
  so it should reproduce the bug directly (a good place to pin a regression).
- **Classical 1-D path (reference, converged):** `_sem_modes_tensor` (`pmm/_core.py:957-1032`) +
  `_build_sem_tensor` (`_core.py:886-953`); the in-plane factorization `Cxx = inv([[1/exx]])`,
  `Cxy = Cxx[[exy/exx]]`, etc. (`_core.py:981-988`).

The discrepancy is confined to the **wall-normal factorization of the patterned tensor** — the
uniform cell has no wall so the two builds coincide; adding a wall makes them diverge. Prime
suspects, in order: (a) the `1/exx` / off-diagonal `exy,eyx` Li-inverse-rule projection differs
between `_tensor_layer_modes` and `_build_sem_tensor` for a patterned cell; (b) the
`ezz`-Schur-before-projection ordering (`twod_jones.py:181-187`) is applied inconsistently with the
classical `_t3`/`_build_sem_tensor` reduction; (c) a projector/mass-matrix normalization in the SEM
tensor projection. The tell is that it is a **phase/retardance** offset (magnitudes are close), which
points at the complex off-diagonal factorization rather than the real diagonal.

---

## 4. Impact / blast radius

Any conical (`phi != 0`) solve whose cell contains a **patterned** `(3,3)` tensor is affected:
- `pmm_jones_1d_conical_tensor` (single patterned tensor layer);
- `PMMStack._solve_conical` for stacks with any patterned tensor layer (the exp10/exp11 LC out-couplers, and any liquid-crystal / magneto-optic / stress-birefringent grating);
- by the shared `_tensor_layer_modes`, **`PMM2DStack` is likely affected too** for patterned tensors under conical — needs the same reduction check (unverified here).

NOT affected: scalar (isotropic) gratings under conical (`pmm_jones_1d_conical` scalar path);
uniform/planar anisotropic layers under conical (Berreman-exact); all classical-mount (`phi=0`)
solves. So the exp12 **in-plane** cut (classical) is correct; only the **out-of-plane** cut is blocked.

Note the failure is silent — there is no energy violation to trip on (retardance errors conserve
power), and the v5.20.0 test suite validates only the *uniform* tensor and *scalar* grating conical
cases, so the patterned-tensor gap passed CI.

---

## 5. Fix direction + validation gates

**Fix:** reconcile the patterned-tensor projection in `_tensor_layer_modes` (conical/2-D) with the
converged classical `_sem_modes_tensor`/`_build_sem_tensor` at `ky0 = 0`. Concretely, dump the modal
operators from both builds for the reproducer's single patterned layer at `ky0=0` and diff the
`Cxx/Cxy/Cyx/Cyy` (and `EZZ`) blocks — they must be identical; the first block that differs is the
bug. Focus on the wall-normal `1/exx`-family Li factorization and the `ezz`-Schur ordering.

**Regression gates (add to `test_v5_20_0_pmm_conical.py`):**
1. **Patterned degenerate-limit reduction (the missing test).** The reproducer above:
   `pmm_jones_1d_conical_tensor` (or `PMMStack` conical) on a patterned anisotropic layer at
   `theta=0, phi=90` must equal the classical `phi=0` Jones to ~1e-10. This is the exact analogue of
   the existing `test_native_conical_reduces_to_classical_at_phi0`, but with a **patterned** tensor
   (the current test's cell is not patterned-tensor, which is why the bug shipped).
2. **Retardance tracking.** Sweep the anisotropic director angle; the conical `arg(J00)-arg(J11)` must
   track the classical to ~0.1 deg (the reproducer shows a fixed ~3.5 deg offset today).
3. **Converged cross-oracle.** A patterned tensor grating vs a HIGH-order converged `PMM2DStack`
   (y-invariant) conical solve, `phi != 0` — pins the true off-normal answer, not just the reduction.
   (Do NOT use `rcwa_*_1d` — planar-only; and RCWAStack does not converge for lossy-metal
   metasurfaces at feasible order counts, so it is not a usable oracle here.)

**Do NOT gate on energy** for these anisotropic cells (a lossless cell auto-balances power even with a
wrong per-order/retardance split) — gate on the Jones/retardance value, per the standing OOP-tensor
rule (`test_v5_14_0_pmm2d_oop.py:114-129`).

---

## 6. Status for the consumer (exp12)

- **In-plane angular-tolerance cut:** unblocked (classical PMM, validated).
- **Out-of-plane cut:** blocked until this is fixed. RCWA is not a fallback here (does not converge
  for this metasurface — n_orders 15 vs 25 swing 32.5%->37.4% peak, neither near the PMM 75.6%).
  Options: fix the native conical patterned-tensor build (this audit), or validate a HIGH-order
  `PMM2DStack` y-invariant path if it proves unaffected.

---

*Prepared 2026-07-12. Reproducer + convergence data run on box B (lumenairy 5.21.2, editable install
of the mirror). Companion design doc: `AUDIT_PMM_CONICAL_OUT_OF_PLANE_2026_07_03.md` (the Path-B
implementation this bug is in). No library code modified by this audit.*
