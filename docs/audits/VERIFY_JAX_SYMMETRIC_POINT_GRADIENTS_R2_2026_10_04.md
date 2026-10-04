# Independent verification: JAX symmetric-point gradients, round 2

Date: 2026-10-04.  Verifier: Claude Opus 5.5 (`claude-opus-5-5`).  Subject:
round 2 of the symmetric-point gradient fix, commits `77376229` (fix) and
`7f711e53` (docs), merged at `67d492fe` / `a4a219e7`; claims in
`BUILD_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_03.md` section 10 and in the
CHANGELOG `### Fixed` entry of `[Unreleased]`.  Base for PRE readings: a
worktree of `7c0bc8bd` (removed after use).

Builds: Windows 11 / CPython 3.14.6 / jax 0.11.0 ("win"); WSL / CPython
3.12.3 / jax 0.10.2 ("wsl"), import path asserted to this tree in every run.
The machine was shared with another session's `pytest -n 4` for the whole
verification, so every timing below is from a loaded box.

Probes: `validation/probe_jax_symgrad/verify_r2/` (`p1` .. `p11`, JSON and
logs per build).  Tests: `tests/unit/test_verify_jax_symmetric_point_gradients_r2.py`
(27 passing pins, 6 strict xfails, both builds).

Terms.  "AD" = `jax.jit(jax.jacrev(...))` unless stated.  "FD" = a
Richardson-extrapolated central difference of the NumPy (or concrete) forward
on three rungs whose h^2 premise (ratio of successive rung changes = 11.375
within 12 %) is checked; where the premise could not be shown on the
default rungs the larger rungs that satisfy it are named.  "ON" / "OFF" =
`lumenairy.backend.jax_cluster_rule(True / False)`; OFF is the plain
composition, i.e. the gradient without the rule.  Errors are relative to
the largest FD entry.

## Verdicts

| claim | verdict |
|---|---|
| (1) every routed solver correct at its symmetric point | PARTLY: confirmed on 15 own fixtures; REFUTED at two kinds of symmetric point (P1-1, P1-2) |
| (2) forward bytes PRE = POST | CONFIRMED (108 / 112 per build; the 4 others are the intended 'li' change) |
| (3) traced Li route built | CONFIRMED |
| (4) RCWAStack cache leak fixed | CONFIRMED (reproduced on base, fixed on HEAD) |
| (5) the switch | PARTLY: the four pins hold; the documented remedy after a change does not work (P2-1); interleaved scopes leave it off (P3-2) |
| (6) cost | PARTLY: run-time ratios as claimed; forward compile is NOT unchanged; gradient compile and memory above the CHANGELOG figures (P2-2) |
| (7) controls unchanged | CONFIRMED on two controls |
| B: record / replay mechanism | CONFIRMED sound for deterministic solves; one unhelpful error (P3-1); no guard against a reordered replay (note) |

## Claim (1): own fixtures, every routed family

Fixtures differ from the builder's in cell (24 x 24 cross, +x half-arm
direction), contrast, loss, orders (`n_orders` 3 = 49 orders), layer count,
half-space indices and incidence.  `p2_families.py`.

| family | fixture | ON win / wsl | OFF win / wsl |
|---|---|---|---|
| `berreman_jones_1d` (traced) | two lossy isotropic layers, normal | 2.9e-10 / 1.7e-10 | 0.84 / 0.84 |
| | same, 0.35 rad | 4.4e-10 / 3.9e-10 | 0.45 / 0.45 |
| `pmm_jones_1d` | lossy in-plane-anisotropic ridge, degree 10 | 1.7e-10 / 1.6e-10 | 0.24 / 0.24 |
| `pmm_efficiency_1d` | lossy ridge, TE + TM | 1.3e-10 / 5.8e-11 | 0.85 / 0.85 |
| `rcwa_jones_2d` | lossy cross (6.25+0.8j) | 5.4e-10 / 9.2e-10 | 0.042 / 0.072 |
| | 'li', uniaxial cross | 3.6e-10 / 6.2e-10 | 0.063 / 0.078 |
| `rcwa_efficiency_2d` | lossy cross, TE + TM | 5.0e-10 / 4.1e-10 | 0.10 / 0.15 |
| `RCWAStack` | cross / spacer / lossy cross | 5.2e-10 / 4.2e-10 | 0.21 / 0.071 |
| `PMMStack` shared | 3 layers, lossy, degree 10 | 1.3e-9 / 1.3e-9 | 1.54 / 1.54 |
| `PMMStack` per-layer | same | 1.3e-9 / 1.3e-9 | 1.52 / 1.52 |
| | 2 layers, degree 8 (p10, win) | 5.7e-10 shared, 5.4e-10 per-layer | 1.58 |
| hybrid `PMM2DStack` | two patterned layers, both traced layouts, degree 7 | 1.9e-10 / 3.0e-10 | 0.13 / 0.094 |
| | same, degree 5 (p10) | 9.5e-11 / 1.7e-10 | 0.096 / 0.063 |

On the RCWA rows the h^2 premise of the default rungs (1e-3 .. 1e-4) fails
on a few small components: the NumPy forward's round-off near the split
clusters exceeds the rung changes of a nearly linear response.  On rungs
(1e-2, 3e-3, 1e-3) it holds on every component above 1e-2 of max |FD| and the
tests use those rungs; the AD readings agree with both FDs at the 1e-10
level.

Probes for new breakage (none found):

* A symmetry-KEEPING parameter (all four arms of the cross): ON and OFF
  agree (2.5e-10 vs FD for both; ON vs OFF 2e-14 class).
* An accidental near-degeneracy (the builder's `pmm_jones_1d` grating at
  0.2 rad, an Mbig pair 3.1e-7 of the spectrum apart): ON 2.0e-11 / 2.8e-11,
  OFF the same -- the lifted branch runs and changes nothing.
* Lossy materials: every fixture above except the first Berreman layer pair
  carries loss.
* A cluster with more than two members: a uniform layer at normal incidence
  (8-member clusters) just off the uniform point (t = +-1e-6, inside gap_rel):
  mean of AD(+-1e-6) vs FD 4.1e-9 / 3.7e-9.  A synthetic 5 x 5 non-normal
  matrix with two pairs: 7.0e-10 / 1.2e-9 (OFF 1.31 / 2.48).
* Conical incidence: `RCWAStack` at (0.2, 0.3) with a uniform layer over a
  cross, just off the uniform point: 2.0e-9 / 1.9e-9.
* `jax.vmap(jax.grad)` on a batch [0, 1e-2, 2e-2] (one-layer Berreman,
  oblique): ON every member 1.9e-11 .. 8.8e-11 (win), <= 9.9e-10 (wsl) vs FD;
  vmap vs per-point 5.3e-10 / 2.5e-10 absolute.  `jax.jacrev` is the AD of
  every row.  Eager (non-jit) `jax.jacrev` at a Berreman symmetric point
  (whose first pass takes the analytic isotropic branch and whose traced
  passes take the eig branch) matches the FD (pin).
* `jax.hessian`: as the CHANGELOG states -- finite for a thickness, and
  `NotImplementedError` for a parameter entering an eig; `jax.jvp` /
  `jax.jacfwd` through any routed eig raise `TypeError` (custom VJP), as
  before the fix.

### P1-1 (REFUTES claim 1): near a Rayleigh anomaly, `pmm_jones_1d` and `PMMStack` are wrong at normal incidence

`_jax_cluster_routed(solve, anchor=0.0)` hands the rule the SAME anchor, 0,
for every recorded problem (`lumenairy/elements/rcwa/_core.py:5104-5106`).
The anchor is where the consumer is not smooth in that problem's
eigenvalues; the lift of a cluster close to it is shortened.  For the
layer operators (eigenvalue q^2 or kz^2) 0 is right.  For the half-spaces'
shared geometric eig `Kx2` of `pmm_jones_1d` and `PMMStack`
(`lumenairy/elements/pmm/_core.py:4357`, routed at `pmm/_core.py:4507`,
`pmm/_jax_stack.py:459` and `:746`) the eigenvalue is mu ~ (kx / k0)^2 and
the consumer takes q = sqrt(eps_half - mu): it is non-smooth at
mu = eps_sub and mu = eps_sup, a Rayleigh anomaly.  At normal incidence the
+-m orders are an exact pair of `Kx2`, so the lifted branch always runs,
and close to an anomaly the unshortened lift approaches or crosses the
branch point.

`pmm_jones_1d` (P 1.2, ridge 4 / groove 1, n_sub 1.0, n_sup 1.45, degree 12;
wl = 1.2 (1 + delta), the m = 1 substrate order just evanescent; d / d angle
at 0 of R, T of the +-1 orders, both incident polarizations; `p1`):

| delta | AD vs FD (win = wsl) | mirror-identity defect of AD | of the FD |
|---|---|---|---|
| 3e-2 | 3.3e-11 | 1.3e-11 | 2.8e-11 |
| 1e-2 | 3.2e-9 | 2.3e-9 | 6.1e-10 |
| 3e-3 | 2.7e-7 | 2.7e-7 | 9.2e-10 |
| 1e-3 | 1.8e-5 | 2.2e-5 | 5.7e-9 |
| 3e-4 | 1.7e-3 | 2.1e-3 | 4.7e-8 |
| 1e-4 | 19.3 | 0.50 | 1.1e-7 |
| -3e-4 (propagating side) | 2.4e-3 | 1.7e-3 | 2.4e-11 |
| -1e-4 | 1.28 | 0.41 | 3.4e-10 |

The error grows as delta^-4, the stencil's truncation (d / a)^4 with the
lift d fixed and the distance a to the branch point shrinking.  The mirror
identity d X_{+1} / d angle = - d X_{-1} / d angle (an exact property of the
mirror-symmetric grating) is FD-free: the FD obeys it to 1e-7 and AD breaks
it by 0.50.  Attribution (`p1b`): passing the anchors (0, eps_sub, eps_sup)
to every problem reduces the error from 1.8e-5 to 4.1e-8 (delta 1e-3) and
from 19.3 to 1.2e-6 (delta 1e-4), both builds.  `pmm_efficiency_1d`, whose
half-space eigenvalue is q^2 itself (anchor 0 correct), stays at 2.3e-8 /
3.3e-6 at the same distances.  `PMMStack` (`p10b`, 2 layers, n_sup 1.45,
n_sub 1.0, degree 10): 3.4e-3 at delta 1e-3 and 3.3 at delta 1e-4 (both
builds), mirror defect 6.6e-2 (FD 4.7e-8).

Severity P1 (a wrong number from the fixed path).  It is not a regression:
OFF (and the base tree) is wrong there by 1 - 38.  The CHANGELOG's
"no branch point near the pairs" holds for the builder's fixtures, not for
this regime.  Pins: `test_r2v_pmm_jones_1d_mirror_identity_near_a_rayleigh_anomaly`,
`test_r2v_pmmstack_mirror_identity_near_a_rayleigh_anomaly` (strict xfail).

### P1-2 (REFUTES claim 1, pre-existing): `RCWAStack` at an exactly uniform layer

The most symmetric point of a patterned layer is a uniform one, and it is the
usual starting point of a topology optimisation.  `RCWAStack` (JAX) with a
uniform `eps_cell` layer (2.25) plus t x a zero-mean random pattern, over a
cross layer (`p9`):

| case | AD(0) vs FD | mean of AD(+-1e-6) vs FD | base tree, AD(0) |
|---|---|---|---|
| normal incidence | 8.50 / 8.50 | 4.1e-9 / 3.7e-9 | 8.50 / 8.50 |
| conical (0.2, 0.3) | 5.74 / 5.74 | 2.0e-9 / 1.9e-9 | 5.74 / 5.74 |
| same cell as `eps_tensor_cell` | 1.8e-8 / 8.3e-9 | -- | -- |

Cause: `_layer_eigenmodes` (`lumenairy/elements/rcwa/_core.py:2785-2792`)
returns `xp.where(uniform, analytic modes, eig modes)` on JAX.  At t = 0 the
analytic branch is selected; its modes are W = I and kz from `EPS[0, 0]`
only, so the gradient carries the pattern's mean and nothing of its
first-order coupling to the other layer's diffracted orders.  The eig
branch, which the cluster rule now differentiates exactly, is discarded by
the `where`.  A single layer is not affected (at first order only the mean
reaches the specular orders), nor a tensor cell (a traced tensor goes to the
general generator path, which has no such select).  ON and OFF are
identical: the rule never sees it.  P1 (wrong number at a symmetric point of
a routed solver), pre-existing.  Pin:
`test_r2v_rcwastack_uniform_layer_pattern_gradient_at_the_uniform_point`
(strict xfail).

## Claim B: the record / replay helper

Read: `_jax_cluster_routed` (`rcwa/_core.py:5032-5106`) runs the solve once
with a recording route (a `contextvars.ContextVar`), and when the rule
applies re-records the problems inside a closure-converted trace and replays
the eigenpairs into the solve as the consumer.  Probed on a synthetic routed
solve (matrix exponential through `_jax_twin_eig`; `p3`, both builds):

* Correctness of the helper itself: ON 7.0e-10 / 1.2e-9 (jit and eager),
  OFF 1.31 / 2.48.  A routed solve NESTED in another: 7.3e-10 / 4.8e-10.
* An exception raised in the recording pass, the traced re-recording or
  the replay propagates; the route is reset (`None`) and the next gradient
  is exact.  No half-recorded state survives (the problem lists are local to
  the call; the route is reset in `finally`).
* Desynchronisation by state outside the trace (a call counter changing the
  eig count on the replay): one FEWER eig raises the helper's
  `RuntimeError` ("took 1 eigs on replay but 2 when recorded").  One MORE
  raises a bare `StopIteration` from `next(it)` (`rcwa/_core.py:5090`) --
  P3-1, a misleading error; a solve that iterates a `map` over its eigs could
  even swallow it, though the count check would then still fire.  Strict
  xfail `test_r2v_more_eigs_on_the_replay_raise_the_named_error`.
* Same count and shapes, different ORDER on the replay: the jitted FORWARD
  value is silently wrong (0.177 relative).  Nothing ties a replayed eig to
  the problem it was recorded from (the replay ignores its `L`).  Reachable
  only by a solve whose control flow is nondeterministic between two traced
  passes; no library solve is.  Recorded as a note, not pinned (there is no
  cheap traced check of identity).
* Concrete-versus-traced branches: the Berreman twin's analytic isotropic
  branch is taken in the eager first pass and the eig branch in both traced
  passes, which agree; the eager pin passes.
* Caches between record and replay: the library's routed closures hold no
  module-level cache (the `RCWAStack` per-call `_mode_cache` is created
  inside the closure, fresh per pass; the half-space modes are computed
  outside it).
* An eig inside the solve's own `jax.vmap` / `lax.scan`: a loud leaked-trace
  error, the route reset.  No library solve does this.
* Static-shape retrace (`jit` with a static size 5 -> 2 -> 5): each correct.
* Threads: the route is a `ContextVar` (per thread); four threads tracing
  routed gradients concurrently are each exact.  The SWITCH is a module
  global: two `jax_cluster_rule` scopes left out of order (what two threads
  each using the context manager do) leave the process with the rule OFF --
  P3-2, strict xfail `test_r2v_interleaved_switch_scopes_do_not_leave_the_rule_off`.

## Claims (2) and (3): forward bytes and the traced Li route

`p4_bytes.py` on HEAD and on the base tree, 112 entries per build: NumPy /
eager JAX / `jax.jit` of `rcwa_efficiency_2d` (TE, TM, normal, conical),
`rcwa_jones_2d` (isotropic, in-plane anisotropic with eps_xy = 0.3, tilted
with eps_xz = 0.4; 'laurent' and 'li'; normal and conical), `RCWAStack`
(two patterned layers incl. an anisotropic tensor cell, normal and
conical), `berreman_jones_1d` (off-plane, in-plane, scalar; normal and
oblique), `pmm_efficiency_1d`, `pmm_jones_1d` (0 and 0.2 rad), `PMMStack`
(shared and per-layer, 0 and 0.2 rad), the hybrid stack (concrete and
traced layout, 'laurent' and 'li') and the 1-D RCWA control.

* 108 / 112 byte-identical on each build.  The four that differ are exactly
  the jitted `rcwa_jones_2d(formulation='li')` on in-plane cells (iso and
  aniso, normal and conical): PRE they equalled the Laurent answer
  (5.2e-3 .. 7.2e-3 from NumPy 'li', = the NumPy 'li' vs 'laurent' gap);
  POST they equal NumPy 'li' to 3.5e-15 .. 1.7e-14 (both builds) --
  including the eps_xy cell, so the in-plane off-diagonal blocks agree too.
* Out-of-plane tensor under 'li': NumPy 'li' = NumPy 'laurent' exactly (the
  general path is the direct rule); jitted and eager JAX = NumPy to
  2.4e-14 / 6.1e-15, unchanged PRE vs POST; no scope notice and no refusal
  on any of the three routes (same behaviour).

## Claim (4): the `RCWAStack` cache leak

Own fixture (cross 6.25, t on the +x half-arm, theta 0.1): on the base tree
`jit(f)` then `jit(grad(f))` and an eager `f` both raise
`UnexpectedTracerError` and the module cache holds tracers (both builds); on
HEAD the three succeed, eager and jitted values agree to 5.6e-17 / 1.0e-16
and the cache holds no tracer.  Pin `test_r2v_rcwastack_cache_leak_sequence_is_fixed`.

## Claim (5): the switch

Confirmed (both builds; `p5`, tests):
* OFF at a symmetric point is wrong (0.024 .. 1.65 on every own fixture
  above), ON and OFF agree away from symmetry and at an accidental cluster.
* With the switch off, a vmapped gradient's jaxpr holds one eig per eig
  problem of the plain solve (2 for a two-layer `RCWAStack`; ON: 14).
* A vmapped batch mixing a symmetric and generic points: ON every member
  exact; OFF the symmetric member wrong (1.65) and the others exact.
* The context manager restores on exception and nests correctly.
* `LUMENAIRY_JAX_CLUSTER_RULE` is read once, at import: '0' / 'OFF' before
  import turn the rule off; setting it after import changes nothing (the
  docs call it the process default -- correct); an unrecognised value
  ('disabled') silently leaves the rule on.
* A jitted function keeps its setting for an already-compiled signature;
  the same jitted function RETRACED for a new input dtype under a changed
  setting takes the new setting (pin
  `test_r2v_a_jitted_function_keeps_its_setting_per_compiled_signature`).

P2-1 (refutes the documented remedy).  The setter's docstring
(`lumenairy/backend/array.py:266-267`) says to "re-create the jitted
function after changing it".  `jax.jit` caches compiled programs by the
wrapped callable, so a NEW `jax.jit` wrapper of the SAME callable reuses the
program traced under the old setting (`p11`, both builds): compiled OFF,
switched back ON and re-wrapped, the gradient at a symmetric point stays
wrong (1.60); only a new callable takes the new setting.  This is the
dangerous direction: a user who turns the rule back on and follows the
docstring still gets wrong gradients.  Strict xfail
`test_r2v_rewrapping_a_jitted_callable_takes_the_new_setting`.

## Claim (6): cost

Windows, rule ON vs OFF in one process, jitted, median of 7 after the first
call, two runs (`p6`, the second with a fresh callable per jitted forward --
see P2-1 for why the first run's forward compile figures were cache hits).
The box was loaded throughout; spreads between the two runs are given.

| case | cluster | gradient ON / OFF | gradient compile | forward run | forward compile |
|---|---|---|---|---|---|
| RCWA cross (n_orders 3) at 0 | yes | 4.0x / 7.8x | 3.4x / 4.2x | 0.56x | 1.14x |
| RCWA cross + 1e-2 random, theta 0.2 | no | 0.96x / 1.14x | 2.8x / 2.9x | 1.26x | 1.67x |
| `pmm_jones_1d` at 0 | yes | 4.4x / 2.5x | 4.2x / 2.2x | 1.08x | 1.26x |
| `pmm_jones_1d` at 0.2 (accidental) | yes | 3.8x / 7.2x | 4.2x / 4.1x | 1.67x | 1.51x |
| `PMMStack` 2 layers at 0 | yes | 4.3x / 3.7x | 5.7x / 5.9x | 1.55x | 1.85x |
| Berreman isotropic at 0 | yes | 7.9x / 2.3x | 3.9x / 4.1x | 0.45x | 1.86x |
| vmap of 4, no cluster | no | 1.01x / 0.75x | 3.9x / 3.5x | -- | -- |
| vmap of 4, one cluster in the batch | yes | 4.0x / 3.5x | 3.8x / 3.6x | -- | -- |

Larger order count (`p7`, fresh subprocess each, `rcwa_efficiency_2d`,
n_orders 5 = 121 orders, four-fold cell): `jit(grad)` 0.82 s vs 0.16 s
(5.1x), peak RSS 0.73 vs 0.56 GB, first call 5.3 vs 2.2 s; `jit(jacrev)`
(242 outputs) 34.4 s vs 8.5 s (4.0x), peak RSS 12.8 vs 4.1 GB (3.1x), first
call 42.7 vs 11.9 s.  (A TE + TM `jit(jacrev)` of that cell reached 26 GB.)

* Confirmed: with a cluster ~4x (2.3 - 7.9x on this loaded box); without one
  ~1.0x; vmap without a cluster ~1.0x (the batch-OR works); a batch with one
  cluster pays the lifted cost for all.
* P2-2 (overstated / missing):
  - "The forward pass and its compile are unchanged" (CHANGELOG): the
    jitted FORWARD compile is 1.1 - 1.9x longer with the rule on (the solve
    is traced three times -- record, traced re-record, replay -- and the
    rule's custom VJP is staged); run time is within noise.
  - Gradient compile: CHANGELOG "2 - 4x"; measured up to 5.9x (`PMMStack`),
    and the build record's own section 10.6 says up to 5.5x.
  - Public docstrings (`berreman.py:868`, `pmm/oned.py:214`, `:499`,
    `pmm/stack.py:2877`, `pmm/stack2d.py:1531`, `rcwa/stack.py`) say
    "4 - 13x longer ... compiles 2.5 - 5x" -- a third set of numbers,
    traceable to no table in the record.
  - Memory is not mentioned anywhere: peak RSS 1.3x (`grad`) to 3.1x
    (`jacrev` with 242 outputs) at a cluster.

## Claim (7): controls

Own fixtures (`p8`), HEAD vs base, both builds: `rcwa_efficiency_1d`
(lossy ridge, 13 orders) d / d angle at exactly 0: 7.3e-11 (TE), 3.0e-10
(TM) win, 4.5e-11 / 2.3e-10 wsl; `BORStack` (R 2.5, N 48, ring 0.7 / 0.4,
m = 1 and 2) d / d n_ring at the radially homogeneous point: 2.7e-10 /
2.3e-10 win, 2.2e-10 / 3.0e-10 wsl.  Values and gradients byte-identical
HEAD vs base on each build.  CONFIRMED.

## Claim F: the CHANGELOG, the comments and the docstrings, read as a physicist

* Traceable numbers: the before / after table and the switch pins trace to
  the probe JSON; the cost paragraph and the docstrings do not agree with
  each other (P2-2).
* Overstatement (P2-3): the docstrings say gradients at a symmetric
  configuration -- "a four-fold cell, a mirror-symmetric grating at exactly
  normal incidence, an isotropic layer" -- "are exact"; the `_core.py`
  comment block says every routed twin "is then exact at and near the
  degenerate point"; the CHANGELOG says "no branch point near the pairs".
  P1-1 (a mirror-symmetric grating at normal incidence near a Rayleigh
  anomaly) and P1-2 (an isotropic uniform layer in `RCWAStack`) are
  counter-examples of exactly the named kinds.
* Entries named correctly: the routed / not-routed lists match the code (the
  uniform-layer `where` of `_layer_eigenmodes` is the one RCWA mode path
  that the rule cannot see; it is not mentioned).
* The switch's meaning and default are unmistakable ("OFF ... WRONG",
  default on, env var = process default).  Wording (P3-3): "a jax.jit-
  compiled function keeps the setting it was compiled with" is true per
  compiled signature only; the remedy given is ineffective (P2-1); the
  setting is process-global and not safe to scope from several threads
  (P3-2).

## Defects

| id | rank | where | what |
|---|---|---|---|
| P1-1 | P1 | `rcwa/_core.py:5104-5106`; Kx2 at `pmm/_core.py:4357` | one anchor (0) for every problem; `pmm_jones_1d` / `PMMStack` d/dangle at normal incidence wrong near a Rayleigh anomaly (19x at 1e-4) |
| P1-2 | P1 (pre-existing) | `rcwa/_core.py:2785-2792` | `RCWAStack` gradient at an exactly uniform scalar `eps_cell` layer wrong (8.5x), the eig branch discarded by a `where` |
| P2-1 | P2 | `backend/array.py:266-267`, `:291` | "re-create the jitted function" does not take the new setting (jit caches by callable) |
| P2-2 | P2 | CHANGELOG cost paragraph; public docstrings | forward compile not unchanged (1.1-1.9x); gradient compile up to 5.9x; three inconsistent cost sets; memory 1.3-3.1x unstated |
| P2-3 | P2 | docstrings, `_core.py` comment, CHANGELOG | "exact" at symmetric points stated without the P1-1 / P1-2 exceptions |
| P3-1 | P3 | `rcwa/_core.py:5090` | more eigs on replay -> bare `StopIteration` |
| P3-2 | P3 | `backend/array.py:271-316` | process-global switch; interleaved scopes leave it off |
| P3-3 | P3 | setter docstring, CHANGELOG | "keeps the setting it was compiled with" holds per compiled signature only |
| note | -- | `rcwa/_core.py:5085-5091` | a same-shape reordered replay is not detectable (silent wrong forward); no library solve can do it |

## Tests

`tests/unit/test_verify_jax_symmetric_point_gradients_r2.py`:
27 passing pins (claims 1, 3, 4, 5, 7 and the helper), 6 strict xfails
(P1-1 x 2, P1-2, P2-1, P3-1, P3-2).  Windows (`-n 2`, BLAS threads 2):
27 passed, 6 xfailed in 260 s; slowest single tests 45 - 57 s on the loaded
box (the PMMStack pair and the uniform-layer xfail, which compiles three
gradients).  WSL (serial, import path asserted): 27 passed, 6 xfailed in
583 s; slowest 34 - 58 s.  On this loaded box several tests exceed the
~20 s target by up to 3x; on a quiet box the same tests ran 10 - 25 s in
the probes.  The builder's and round 1's files
(`test_jax_symmetric_point_gradients.py`,
`test_verify_jax_symmetric_point_gradients.py`) on Windows: 53 passed.

## Not done

* GPU; an idle-box timing (the machine carried another session's suite
  throughout, so the cost ratios carry a 2x spread).
* The WSL cost table (only the Windows one was measured).
* A full-suite run on either build (only the files named above).
* The anchor fix and the uniform-select fix were probed (P1-1's anchor
  patch), not built.
* Whether `PMMStack` with a traced uniform `segments` eps or the hybrid
  stack's uniform layers have a select like P1-2's.
