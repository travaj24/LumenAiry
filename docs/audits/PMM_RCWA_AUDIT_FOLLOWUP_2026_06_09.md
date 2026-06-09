# PMM + RCWA Exhaustive Audit (follow-up) -- 2026-06-09

*Method: 9-dimension multi-agent adversarial workflow (45 agents, 2.3M tokens). 35
raw findings → bidirectional adversarial verification (refute claimed bugs; stress-test
claimed-safe session changes) → 19 survived / 16 refuted → 8 distinct after
de-duplication. One dimension specifically stress-tested this session's Stage 2-3
commits.*

## 1. Executive summary

The PMM + RCWA suites are in good overall health: **no P0 issues**, and every confirmed
finding is a usability, convention, test-coverage, or documentation concern rather than a
numerical-correctness defect in the shipped physics. After de-duplication across the 9
audit dimensions there are **8 distinct findings: 1 P1, 4 P2, 3 P3**. The single most
important action is to fix `_promote_eps_tensor`, which silently fails the documented
"scalar promoted to isotropic tensor" contract for JAX scalar inputs — the one finding
that breaks a publicly advertised API path. The remaining items are a cross-suite
`set_source` override asymmetry (P2), two test gaps around already-correct optimizations
(P2), and three documentation/DRY cleanups (P3).

## 2. Verdict on this session's Stage 2-3 changes

| Change | Verdict | Note |
|---|---|---|
| `n_orders` / `theta` aliases | **CONFIRMED-SAFE** | Aliases byte-identical; `_resolve_incidence` returns `theta` if given else `angle`. Only docstring clarity (F6) and the cross-suite override direction (F2) flagged — the 1-D alias resolution itself is correct. |
| `pmm_1d` promotion | **FLAGGED (F1, P1)** | NumPy promotion correct, but `_promote_eps_tensor` does not promote JAX scalars — breaks the documented scalar contract on the JAX surface. |
| `EPS_normal` rename | **CONFIRMED-SAFE** | `EPS_normal = EPS_II if use_li else EPS` sound; routing correct. Residual: missing end-to-end `use_li=False` test (F4). |
| P1 Redheffer zero-block opt | **CONFIRMED-SAFE** | The `is_jax_array` gate correctly prevents `bool(.any())` on a tracer; results unchanged. Only the comment wording is misleading (F7). |
| P4 `EPS_II` skip | **CONFIRMED-SAFE (correctness), FLAGGED (test gap F4)** | Mathematically sound — the Laurent path never reads `EPS_II`. EPS byte-identity is tested; the end-to-end result claim is not, and there's no `laurent`+`tm` guard. |

## 3. Findings by severity

### P0 — Critical
**None.**

### P1 — High

- **F1 — `_promote_eps_tensor` does not promote JAX scalars** (`pmm.py:173-184`). The JAX
  early-return passes a 0-d JAX array through unchanged, so `pmm_1d(eps_ridge=jnp.array(2.0))`
  reaches `pmm_jones_1d` which raises `ValueError` requiring `(3,3)`. Violates the
  documented contract on the differentiable surface. **Fix:** promote 0-d JAX arrays
  (`eps * jnp.eye(3)`, stays differentiable) + add a JAX-scalar test.

### P2 — Medium

- **F2 — `set_source` override asymmetry** (`pmm.py` PMMStack vs `rcwa.py` RCWAStack).
  PMMStack resolves `theta`-wins; RCWAStack resolves `angle`-wins; the RCWA docstring
  falsely claims it "matches PMMStack". The same call `set_source(angle=A, theta=T)` gives
  different incidence across suites. **Fix:** make `theta` win uniformly (the cross-suite
  canonical polar angle), correct the docstring.
- **F3 — PMM `set_source` both-supplied override untested** (`test_v5_12_0_naming_aliases.py`).
  RCWA has the both-given test; PMM doesn't. **Fix:** add it (alongside F2).
- **F4 — `use_li=False` end-to-end unproven + no `laurent`+`tm` guard** (`rcwa.py`).
  EPS byte-identity is tested but not the end-to-end TE/Laurent result; `laurent`+`tm`
  would feed Laurent `EPS` into the wall-normal field (lossless-trap class — energy
  conserves, per-order wrong). **Fix:** end-to-end regression test + precondition guard.
- **F5 — PMM slant forward-mode rebalance mixes flux and decay unnormalized**
  (`pmm.py` `_sem_modes_slant`). Argsorts `np.where(prop, Sz, q.imag)` — `Sz` (~length²)
  and dimensionless `q.imag` — without normalization; the RCWA analog was normalized
  (commit 66181fd) but PMM was not. Latent mode-selection risk in the oblique/lossy
  regime when the initial forward count ≠ n. **Fix:** normalize before mixing + an
  oblique+lossy test. *(Pre-existing, not a session change; touches the load-bearing flux
  selector — verify adversarially before editing.)*

### P3 — Low

- **F6 — `_resolve_incidence` docstring** doesn't state explicitly that 1-D `angle` IS
  `theta` (no scaling), measured from the surface normal. Clarity polish.
- **F7 — P1 Redheffer comment** conflates "block is zero" with "backend is JAX"; the gate
  is correct but the wording implies the JAX path handles the zero-test. Reword.
- **F8 — R/T efficiency tail duplicated** across ~4 RCWA solver sites + factorization
  dispatch across PMM entry points. No correctness/perf risk; control-flow context differs
  so consolidation is non-trivial. Pure DRY. **The audit partially REFUTED the Stage 4
  monolith-split motivation:** backends are unified via the `xp` namespace (~7 conditionals
  total), so the split is a P3 DRY cleanup, not an architectural necessity.

## 4. Top recommendations (ranked)

1. **Fix `_promote_eps_tensor` for JAX scalars (F1)** + JAX-scalar test. Only finding that
   breaks a documented public API path; gating for release.
2. **Resolve the `set_source` asymmetry (F2 + F3)** — uniform `theta`-wins, fix the false
   docstring, add the PMM override test. Without this the cross-suite aliasing goal isn't
   actually met.
3. **Close the `use_li=False` gap + add the `laurent`+`tm` guard (F4)** — exactly the
   lossless-trap class.
4. **Normalize the PMM slant rebalance (F5)** to match the shipped RCWA flux-tolerance fix
   + an oblique+lossy test. Verify adversarially first (load-bearing selector).
5. **Doc/DRY polish (F6, F7, F8)** — batchable; none block release.

## 5. Deferred work re-confirmed

- **Stage 4 reorg:** touched only by F8, and the audit **partially refutes** the
  motivation — the "three solvers × three backends" framing is inaccurate (backends unified
  via `xp`; the only real duplication is the ~6-line R/T tail + the factorization dispatch).
  **Keep Stage 4 deferred; it is a P3 DRY cleanup, not an architectural necessity.**
- **PP1 perf lever:** not implicated by any confirmed finding; remains the top *optional*
  perf item (removes 2 of 3 eigs per PMM solve) for a dedicated effort.
