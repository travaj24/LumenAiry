# RCWA + PMM Post-Reorg Audit -- 2026-06-09

*Method: 9-dimension multi-agent adversarial workflow (61 agents, 3.3M tokens) over the
freshly-reorganised `rcwa/` + `pmm/` packages. 51 raw findings → bidirectional adversarial
verification → 33 survived / 18 refuted → 30 distinct after de-dup. One dimension
specifically stress-tested the 1-D/2-D package split.*

## 1. Executive summary

Good release health. **30 distinct findings: 1 P0, 5 P1, ~14 P2, ~10 P3.** No confirmed
correctness defect on the lossless dielectric/TE paths the suite covers; the reorg
preserved behaviour. The single ship-blocker is a **P0 covariant-slant `kz`
sign/conjugation bug** (`pmm/_core.py` `_pmm_jones_oblique_core`): the covariant solver
conjugates `eps` to the internal `exp(+iwt)` convention but reuses the public-convention
`Im(kz)<0 -> -kz` forward/backward split, so the decay constants carry the wrong
imaginary sign — masked today only because tests are lossless. (Note: covariant is an
OPT-IN factorization; the slant default is convection.)

## 2. Reorg integrity verdict

Mostly clean — **no numerical regression from the split**. Two real debts:
- **Public-surface gap (P2, confirmed):** `pmm/__init__.py` re-exports only `_core`/`oned`/
  `stack`, omitting `twod`/`twod_staggered` — so `pmm_efficiency_2d` /
  `pmm_efficiency_2d_staggered` are NOT reachable via `lumenairy.elements.pmm.*` (the
  top-level `lumenairy.pmm_efficiency_2d` still works). Asymmetric with `rcwa/__init__`.
- **Stale monolith references (P2/P3):** several docstrings/comments still point at the
  deleted `pmm.py`/`rcwa.py` monoliths; the facade docstring lists 3 of 5 submodules.

## 3. Findings by severity

### P0 — Ship-blocker
- **Covariant slant `kz` conjugation bug** (`pmm/_core.py` `_pmm_jones_oblique_core` kz_ord
  3569-3571 vs the eps-conjugation 3638-3640). Wrong half-space forward/backward split +
  decay constants on **lossy** materials. Fix: conjugate `kzo_s/kzo_b` back to public sign
  after `kz_ord` (or move the eps-conjugation past `kz_ord`). Add a lossy covariant test.

### P1 — High
- **`fff_nv` curved-wall lossless-trap** (`rcwa/twod.py:507-524`): cross term validated only
  for axis-aligned/separable; on disks/ellipses-on-metal `R+T+A` closes but absorptance is
  mis-split ~50%. Text-only warning doesn't block. Fix: raise on non-separable geometry now;
  validate/correct the cross-term as follow-up; add a metal-disk-vs-oracle test.
- **Unguarded interface inversion** (`rcwa/_core.py:1100,1155` + `pmm/_core.py:519`):
  `inv(a+b)`/`inv(T22)` with no conditioning guard at the near-singular mode-match. Prefer
  `solve()`; estimate `cond` first.
- **PMM 1-D vs 2-D internal-convention divergence** (`pmm/twod.py:394-397`): 1-D conjugates
  `eps` to internal `exp(+iwt)`; 2-D stays public `exp(-iwt)`. Latent loss bug + stack-mixing
  hazard. Fix: align, or prove the public choice sound for the hybrid basis, or warn loudly.
- **ASR + Laurent silent-wrong** (`rcwa/oned.py:392-468`): `formulation='auto'`+TE → use_li
  False; `asr_eta>0` then runs ASR with the wrong (Laurent) rule silently. Fix: raise.
- **`_check_energy` skipped on JAX** (`rcwa/oned.py:554-555`): a solve that would raise
  `_EnergyError` on NumPy returns wrong efficiencies/grads silently on JAX. Fix: concrete-
  magnitude energy check, or require stabilize, or document.

### P2 — Medium (14)
2-D `'li'` only inverse-rules `E_z` not in-plane `E_x` (convergence-rate, not wrong);
kx0 dimensional(PMM-1D) vs dimensionless(RCWA-1D) cross-suite hazard; stabilize fallback can
return a resonance-band degree; `_ill_scaled` gates on diagonal ratio not `cond(A)`; diagonal-
cure preconditions raised deep not at the entry; **PMM `_redheffer_star` has no zero-block gate**
(RCWA does); `rcwa/stack.py:499` `_internal_cpm` unguarded inv; `rcwa/_core.py:503` exact-zero
uniformity compare (ULP-fragile); `pmm/twod.py:63-66` misleading convention docstring; **the
`pmm/__init__` 2-D re-export gap**; `rcwa_jones_1d/_2d` omit the `theta` alias doc; `rcwa/twod.py:456`
"See pmm.py" stale ref; + 2-D RCWA / covariant / isolated-degree per-order test gaps.

### P3 — Low (10)
kx0 naming; ASR conjugation order-dependency; `PMMStack` 1-D-only vs RCWAStack claim;
`n_orders` "2-D PMM spelling" doc (2-D not public); "mirrors rcwa.py" stale ref; facade
docstring lists 3/5 submodules; duplicated `_t3` closures; dead `_seg_outer_eps`; 2-D energy-
bound rationale; 2-D shapes per-order oracle gap.

## 4. Top recommendations (ranked)
1. Fix the **P0 covariant `kz`** + add a lossy covariant-slant test.
2. Block the **`fff_nv` curved-wall** lossless-trap (raise now; validate later).
3. Resolve the **PMM 1-D/2-D internal-convention** split (align or warn loudly).
4. Add the **ASR+TE and JAX-energy guards** (low-effort tripwires).
5. Restore the **PMM 2-D public surface** (`pmm/__init__` re-exports + `__all__`).
6. Harden the **unguarded inversions** (`solve()`/zero-block gate; cond-gate equilibration;
   tolerance-based uniformity compare).
7. Sweep the **reorg doc debt** (stale monolith refs, facade docstring, kx0 cross-suite
   warning, theta-alias docs, dead `_seg_outer_eps`, duplicated `_t3`).
8. Close the **per-order test gaps** so energy isn't the only correctness gate.

## 5. Physics-correctness verdict

Lossless dielectric/TE physics is **clean and trustworthy**. The lossy + curved-wall regimes
carry the P0 (covariant `kz` under loss) + two P1 per-order risks (`fff_nv` curved walls;
PMM 1-D/2-D convention under loss) that must be fixed or fenced off before a release that
advertises those paths. The 2-D `'li'` partial-inverse-rule is a convergence-rate concern,
not a wrong-answer one. The unguarded inversions are stability/reproducibility (BLAS-build)
risks at measure-zero mode-match coincidences, caught post-hoc by RCWA's energy guard on
NumPy but not on JAX.
