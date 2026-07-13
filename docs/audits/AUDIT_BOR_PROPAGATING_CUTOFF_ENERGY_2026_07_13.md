# BOR-PMM — Propagating-Mode Cutoff Drops Near-Grazing Orders (Energy Leak) — 2026-07-13

> **STATUS — FIXED (2026-07-13).** Remediation on branch `fix/bor-grazing-cutoff`;
> gates in `tests/unit/test_audit_bor_grazing_cutoff.py`. The real-axis cutoff was
> floored at the `q ~ 0` degenerate point only (`q/k0 > 1e-6`) in all THREE classifier
> twins (`bor_stack.solve`'s `prop()`, `bor_solve._physical_propagating`,
> `_jax_bor._mask`); imag and index-ceiling legs unchanged.
>
> - §5 caution 2 (the flux-normalizer seam) was probed and is **unreachable**: the
>   modal flux ratio scales as `P/fnrm = q/k0` exactly for the limiting polarization
>   family (every kept mode on the reproducer basis has `|P| = 1.000000`), so a mode
>   at the 1e-6 floor sits 4 decades above the `1e-10 * fnrm` field-norm fallback —
>   kept implies flux-normalized; the one-predicate redesign was not needed.
> - §5 caution 1 honoured: no bit-identity constant-halving; the k0=2.0 gates pass
>   because that scale has no near-cutoff modes (verified by running the suites),
>   not by construction.
> - Reproducer restored to the pre-`fca4665` values to ALL DIGITS: 319 incident
>   modes, `max|R+T-1| = 1.2216561096067835e-11` (at m, um, AND nm scales);
>   fundamental-mode `R = 0.146135` (gate 3's lossless-trap guard) with per-mode
>   closure 1e-13.  JAX-twin parity + the `_physical_propagating` classifier
>   unit gate + the DynaMeta `lumenairy_bor_bridge` consumer gate (all four
>   legs, incl. the previously-red GATE C) pass; the 21 pre-existing BOR tests
>   pass unchanged (the k0=2.0 scale has no near-cutoff modes — verified by
>   running, per caution 1).
>
> **NEW FOLLOW-UP FINDING (out of this audit's scope, discovered by gate 5):**
> the legacy NODAL `build_layer`/`solve` cascade (`bor_solve.py`) has a separate,
> PRE-EXISTING energy blow-up on large cells — `max|R+T-1| ~ 1e25..1e32` for
> `Rbig >= ~12 lambda` (worst columns are near-AXIS modes with low reldiv), and
> ~1.2 even at `Rbig = 6 lambda`.  A/B monkeypatch shows the result is IDENTICAL
> under the old and new classifier constants, so it is unrelated to this fix and
> predates it.  The production staggered path (`BORStack`) is unaffected (this
> reproducer closes at 1.2e-11 end-to-end).  The cascade-level gate-5 check was
> therefore replaced by a direct `_physical_propagating` classifier unit test;
> the nodal-path blow-up is left as a documented open item for a future audit.

**Severity: correctness (silent wrong numbers).** Commit `fca4665` ("fix(bor):
unit-invariant flux normalization + propagating-mode classifiers — audit P1-01, P2-06",
2026-07-02, first released in v5.18.0, still present at HEAD v5.21.3) replaced the
absolute propagating-mode thresholds with dimensionless ones and chose the real-axis
cutoff `q/k0 > 0.05`. That cutoff **excludes genuinely propagating near-grazing modes**
from the incident/outgoing sets, so `BORStack.solve` returns per-order `R`/`T` sums that
are **missing the power scattered into those orders** — biased low, with `energy = R + T`
reporting the deficit. On the reproducer below the lossless energy closure degrades from
**1.2e-11 (at `fca4665^`) to 2.28e-2**, and the incident-mode count drops 319 → 318.

The irony is exact: P2-06's unit-invariance goal was achieved — the defect is now
**bit-identical at m-, um- and nm-scale unit systems** (all 2.276e-2) — but the
dimensionless constant chosen to preserve bit-identity at the validated `k0 = 2.0` scale
is far more aggressive than the old absolute threshold was at other scales, and CI never
saw the difference (§4).

All file:line anchors relative to
`d:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy\lumenairy\`.

---

## 1. Summary

A propagating mode of a lossless axisymmetric half-space is one with (essentially) real
axial wavenumber `q`, up to `q -> 0+` (grazing). The classifier

```python
# elements/bor/bor_stack.py:249-251 (solve's inner prop()); twins in
# bor_solve.py:67-69 (_physical_propagating) and _jax_bor.py:184-186
qn = L["q"] / k0
(np.abs(qn.imag) < 5e-5) & (qn.real > 0.05) & (np.sqrt(eps).real - qn.real > -5e-10)
```

imposes an ANGULAR cutoff: in a medium of index `n`, `qn.real > 0.05` keeps only modes
with polar angle `theta < arccos(0.05 / n)` — for `n = 1.41` that is `theta < 88.0 deg`.
A near-grazing order carrying real power past that angle is classified out; every
`R[j] = sum_jp |S11[jp, j]|^2` / `T[j] = sum_jp |S21[jp, j]|^2` then silently omits its
row, and `energy` reads `< 1`.

The pre-fix absolute threshold (`q.real > 0.1`, units 1/length) was unit-DEPENDENT — the
P2-06 finding was real — but at the common um-unit scale (`k0 = 2*pi/1.0 = 6.283`) it
amounted to `qn.real > 0.0159` (`theta < 89.35 deg`), which kept the modes this case
needs. The replacement did not preserve that behavior; it preserved the `k0 = 2.0`
behavior (`0.1 / 2.0 = 0.05`), where the old and new classifiers are bit-identical by
construction.

---

## 2. The validated reproducer (library-level, no experiment deps)

A lossless concentric ring grating between index-matched half-spaces, `m = 1`. Runs in
seconds; the unit scale `S` is irrelevant to the numbers (verified m / um / nm).

```python
import numpy as np
from lumenairy.elements.bor import BORStack
S = 1e6                                    # um units (same result at 1.0 and 1e9)
LAM = 1.0e-6
k0 = 2.0 * np.pi / (LAM * S)
s = BORStack(Rbig=48e-6 * S, m=1, N=256, n_superstrate=1.41 + 0j, n_substrate=1.41 + 0j)
s.add_layer(0.5e-6 * S, rings=(3.0e-6 * S, 0.5, 2.45 + 0j, 1.41 + 0j))
s.set_source(k0=k0)
res = s.solve()
e = np.asarray(res["energy"], float)
print(e.size, float(np.max(np.abs(e - 1.0))))
# fca4665 and HEAD v5.21.3:  318  0.022756155665881184
# fca4665^ (pre-fix):        319  1.2216561096067835e-11
```

Bisect: `fca4665^` closes at 1.2e-11 with 319 incident modes; `fca4665` (and every
release since) leaks 2.28e-2 with 318. The DynaMeta bridge only relays `res["energy"]`,
so this is not an adapter artifact.

---

## 3. Localization — the dropped mode, the failing leg, and the exonerated suspect

Forensics on the reproducer's superstrate basis (staggered `layer_modes`, shared by
`inc`/`out` since the half-spaces are index-matched):

- **Exactly ONE mode differs** between the old and new classifiers:
  index 475, `qn = 0.049293 - 6.8e-17j`, polar angle **88.00 deg** — a numerically
  clean, genuinely propagating near-grazing mode.
- It fails **only** the `qn.real > 0.05` leg — by 1.4% (0.0493 vs 0.05). The imag leg
  (`6.8e-17 < 5e-5`) and the index-ceiling leg both pass. No mode is ADDED by the new
  classifier in this case (the looser imag/ceiling legs admit nothing new here).
- **Same-S-matrix discrimination:** recomputing `R`/`T` from the ALREADY-SOLVED
  S-matrix with the pre-fix mode sets restores closure —
  `max|R+T-1|`: 2.276e-2 (new sets, 318 modes) vs **1.222e-11** (old sets, 319 modes).
  The worst-case power scattered into the dropped order from a kept incident mode is
  **0.0228 — exactly the energy defect.**
- **P1-01 (flux normalization) is EXONERATED for this case:** the relative-threshold
  flux split (`zcascade.py:93-94`, `bor_solve.py:42-43`) normalizes every kept mode
  correctly — closure is machine-clean the moment the missing order is re-included, on
  the same solve. The classifier constant is the whole defect here.

Per-order values are wrong too, not just the energy report: the fundamental (near-axis)
mode's reflectance is **0.146135** with the correct sets (`R+T = 1.00000000`) vs
**0.145113** as shipped — R and T are each biased low by whatever fraction of that
incident mode's power exits near grazing.

---

## 4. Impact / blast radius

Affected: **any BOR solve whose scattered field has propagating content past the
angular cutoff** `arccos(0.05/n)` — i.e. diffractive cells (ring gratings, radial DOEs)
with dense mode combs, which is precisely the regime `Rbig >> lambda` targets (the mode
spacing in `qn` scales as ~`1/(k0 * Rbig)`, so large cells ALWAYS populate the
near-grazing band). All three classifier twins carry the same constant:

- `bor_stack.py:249-251` — `BORStack.solve` (this reproducer);
- `bor_solve.py:67-69` — `_physical_propagating` (the `build_layer`/`solve` cascade;
  also gated by `reldiv < 0.5`, untested here);
- `_jax_bor.py:184-186` — the differentiable twin's mask (its :98 flip-mask relative
  imag test serves a different purpose and is not implicated).

NOT affected: cells whose scattered power lives entirely below 88 deg (small `Rbig`,
weak diffraction, the `k0 = 2.0` validated cases); the flux normalization itself
(P1-01 — verified sound above); far-field `R`/`T` of modes that ARE kept, except for
the missing cross-power rows documented in §3.

**Why CI stayed green:** `tests/unit/test_audit_p1_bor_flux.py` (the fca4665 gate) and
the pre-existing BOR gates run at the validated `k0 = 2.0` scale, where the new
constants are bit-identical to the old by construction — the behavior CHANGE at every
other `k0` was never sampled. The downstream DynaMeta gate that would have caught it
(`validation/lumenairy_bor_bridge.py` GATE C, um-scale) sat outside DynaMeta's smoke
tier until 2026-07-12 and had silently rotted; its re-activation is what surfaced this.

**Consumer status:** DynaMeta `fix/deep-audit-2026-07-05` leaves GATE C red pointing at
this audit; the BOR bridge absorption/phase-parity expansion (DynaMeta audit §8, B4b) is
deferred until this is fixed.

---

## 5. Fix direction + validation gates

**Fix:** keep the dimensionless (unit-invariant) form, drop the angular aggression. A
propagating mode is classified by a (relative) real-axis test near ZERO, not at 0.05 —
e.g. `qn.real > 1e-6` (guarding only the `q ~ 0` degenerate point) with the existing
imag and index-ceiling legs unchanged. Apply identically to all three twins
(`bor_stack.py`, `bor_solve.py`, `_jax_bor.py` — the JAX mask must stay in lockstep or
the twin-parity gates fork). Two design cautions:

1. **Do NOT preserve fca4665's `k0 = 2.0` bit-identity this time.** If near-cutoff
   modes exist in a validated-scale case, keeping them CHANGES those results — that is
   the correction working, not a regression. Re-derive the affected pins instead of
   halving constants to force bit-identity (that shortcut is what shipped this bug).
2. **Watch the true-grazing edge.** As `qn.real -> 0` the modal flux `P -> 0` and the
   flux-normalized `|S|^2` for that column grows ill-conditioned; the field-norm
   fallback branch (`abs(P) > 1e-10 * fnrm`) then breaks the power-fraction property
   for a mode the classifier keeps. If a planted probe shows this band is reachable,
   tie the classifier to the SAME flux criterion the normalizer uses (a mode is
   "propagating" iff it was flux-normalized) rather than to an independent `qn.real`
   constant — one predicate, no seam.

**Regression gates:**

1. **The reproducer (§2), at three unit scales** (m / um / nm): `max|R+T-1| <= 1e-9`
   and mode count 319. This fails on HEAD (2.28e-2 / 318) and passed at `fca4665^`.
2. **Near-grazing band probe.** Assert the classifier keeps a planted mode with
   `qn.real` in (1e-3, 0.05) and essentially-zero imag — the band this bug silenced.
3. **Lossless-trap guard (per the standing OOP-tensor rule): do not gate on closure
   alone.** Pin the fundamental-mode `R = 0.146135` (this reproducer, `+-1e-4`) — the
   pre-fix converged value re-derived in §3 from the shipped S-matrix, so a fix that
   "closes" energy by renormalizing rather than by restoring the missing order fails it.
4. **JAX twin parity** on the reproducer (`sum(R)/sum(T)` match the NumPy path) so
   `_jax_bor.py:184` cannot drift from the fixed constant.
5. **`bor_solve` twin**: the same case through `build_layer`/`solve` (the
   `_physical_propagating` + `reldiv` path) closes too — it carries the same constant
   AND an extra `reldiv < 0.5` leg this audit did not exercise.
6. Re-run DynaMeta's `python -m validation.lumenairy_bor_bridge` (all four gates) as
   the consumer acceptance check.

---

## 6. Chronology / provenance

- 2026-07-02 `fca4665` ships P1-01 + P2-06 (v5.18.0 line); gates at `k0 = 2.0` are
  bit-identical by construction; CI green.
- 2026-07-12 DynaMeta's audit-remediation campaign converts its validation runner to
  run-everything discovery; the long-dormant BOR bridge gate runs again and fails at
  2.3e-2.
- 2026-07-12/13 bisect to `fca4665` (worktrees at `fca4665^`/`fca4665`, identical
  inputs: 1.2e-11 vs 2.28e-2); dropped-mode forensics + same-S-matrix set-swap prove
  the classifier constant is the entire defect and exonerate the flux normalization;
  unit-scale sweep confirms the defect is now scale-invariant.

---

*Prepared 2026-07-13 from the DynaMeta deep-audit session (probes in that session's
scratchpad: `bor_probe.py`, `bor_dropped_mode_probe.py`, `bor_units_probe.py`; run on
the editable install of this repo at HEAD `bc0924e` / v5.21.3). Companion documents:
DynaMeta `docs/audit/2026-07-05-deep-audit.md` §9 (upstream-finding entry) and the
fca4665 commit message (the P1-01/P2-06 rationale this audit partially overturns). No
library code modified by this audit.*
