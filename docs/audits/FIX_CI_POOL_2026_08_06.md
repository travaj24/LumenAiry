# FIX -- two CI failures rooted in the Newton pool

**2026-08-06.  Branch `feat/pmm-per-layer-roadmap`.  Closes the two CI
failures the `FIX_POOL_MEMORY_2026_08_06.md` work left behind: its own
sec 8.1 (`serial == pool` was CONDITIONAL) and the memory clamp's notice
breaking a physics-guard suppression test on small runners.  No git command was
run; `CHANGELOG.md` was not touched.**

---

## 0. VERDICT

> **Both failures are one sentence apart: a resolved decision that did not
> travel, and a resource notice with no policy surface.**
>
> 1. `test_niche_newton_pool_both_fits.py::test_pool_result_is_bit_identical_
>    to_serial[polynomial]` -- the pickled Newton payload pinned `newton_fit`,
>    `fit_poly_order`, `fit_weights` and `newton_max_iters` but NOT which of the
>    two implementations of `_Cheb2DEvaluator.ev_value_and_grad` the parent had
>    resolved.  A worker that resolved the other one ran the same polynomial in
>    a different floating-point ORDER.  The payload now pins `cheb_backend` and
>    the worker HONOURS it, or refuses the chunk.  Bit-identity is
>    unconditional again and the contract test stands as written.
> 2. `test_hammer_h3_traced_nyquist_guard.py::test_h3_guard_suppressed_by_
>    silent_policy` -- on CI's ~12 GB runners the new memory clamp correctly
>    ran 2 workers where 4 were asked for and said so, and a test asserting
>    that the H3 *silent* policy leaves NO warnings failed on a resource notice
>    that has nothing to do with H3.  The notice now routes through its own
>    `on_pool_memory` knob, gated at entry like every other string mode knob in
>    that signature.

Neither change can move a number: (1) makes the pool agree with serial where it
previously did not, and (2) changes only what is reported about a clamp whose
own path is bit-identical to serial.

---

## 1. FAILURE 1 -- the evaluator backend is a resolved decision

### 1.1 What CI showed

```
Unit tests (Python 3.10, shard 2/3)
tests/unit/test_niche_newton_pool_both_fits.py:690: in test_pool_result_is_bit_identical_to_serial
    assert np.array_equal(serial, pooled)
E   AssertionError: newton_fit='polynomial': pool result differs from serial,
    max|delta| = 1.358e-11
```

`[spline]` passed in every lane it ran in (3.10 shard 1, 3.11 shard 3, 3.12
shard 3, 3.13 shard 3).  Only the POLYNOMIAL parametrisation failed, and only
where it ran.  That asymmetry is the whole diagnosis: the spline worker rebuilds
a `RectBivariateSpline` and there is exactly one of those, while the polynomial
worker rebuilds a `_Cheb2DEvaluator` -- which has TWO evaluation paths.

### 1.2 The mechanism, in the code

`_Cheb2DEvaluator.ev_value_and_grad` computes one formula two ways:

* an `@njit(cache=True, parallel=True, fastmath=True)` Chebyshev recurrence
  (`_get_cheb2d_val_grad_numba`), taken when `xp is np and _NUMBA_AVAILABLE`
  and the kernel loads; and
* a pure-xp Vandermonde contraction, `xp.sum(c * Tu_K * Tv_K, axis=0)`.

They agree to ~1e-16 RELATIVE -- different summation order over the 28
total-degree basis terms -- not bit for bit.  Both are per-sample, so CHUNKING
cannot change either one; that is why the pool was identical whenever the two
sides agreed on a branch.

`_newton_invert_chunk` rebuilds the evaluator **in a fresh interpreter**, so the
branch it takes is whatever THAT process resolves.  The payload carried no
backend flag, so a parent on the other branch disagreed with its own pool.

The parent's resolution is genuinely process state, not merely environment:

```python
def _get_cheb2d_val_grad_numba():
    if "cheb2d" in _NUMBA_KERNELS:
        return _NUMBA_KERNELS["cheb2d"]     # <- caches None FOREVER
    if not _load_numba():
        _NUMBA_KERNELS["cheb2d"] = None
        return None
```

so any single failed `_load_numba()` in a long pytest session pins the parent to
the pure-xp branch for the rest of it, while every freshly spawned worker still
gets the kernel.  A worker cannot infer that.  The flag has to be pinned.

This is the same class of gap as audit E-H2's `newton_max_iters`, and
FIX_POOL_MEMORY sec 8.1 ledgered it as a known conditional rather than fixing
it, because that change's licence was that it could not move a number.

### 1.3 FAIL-BEFORE, both directions, measured

`scratchpad/repro_backend_split.py` runs this file's own contract shape
(N=1024, `ray_subsample=2` -> 262 144 Newton points, `newton_fit='polynomial'`,
serial vs 4 workers) with the split forced, on Windows 11 / py3.14.6 /
numpy 2.4.4 / numba 0.65.1:

| emulation | how | PRE-FIX | POST-FIX |
|---|---|---|---|
| control, no split | -- | identical, `0.000e+00` | identical, `0.000e+00` |
| parent NumPy, workers numba | `LT._NUMBA_AVAILABLE = False` in the parent | **DIFFER, `5.167e-14`** | identical, `0.000e+00` |
| parent numba, workers NumPy | child-only `sitecustomize.py` on `PYTHONPATH` that patches `importlib.util.find_spec` to return `None` for `numba` | **DIFFER, `5.167e-14`** | identical, `0.000e+00` + one warning |

The first emulation reproduces FIX_POOL_MEMORY sec 8.1's number exactly
(`5.167e-14`).  CI's `1.358e-11` is the same class on different libraries and a
different CPU; no claim is made that the two numbers should match, only the
mechanism.

The SECOND row is why the fix has two halves.  There, the parent's backend is
one the worker cannot provide, and no pin can conjure it -- so the worker must
refuse rather than substitute, and the parent runs the chunk itself where the
pinned backend IS the local one.

### 1.4 What shipped

All in `lumenairy/elements/_lens_traced.py`.

```
_resolved_cheb_backend(newton_fit='polynomial')  'numba' / 'numpy' / None
NewtonWorkerBackendUnavailable                   worker refusal (RuntimeError)
_Cheb2DEvaluator(..., backend=None)              new slot, honoured in
                                                 ev_value_and_grad
_note_pool_backend_refusal()                     warn-once latch + re-arm
```

* **Parent.**  `_invert_newton_parallel` sets
  `_spline_data['cheb_backend'] = _resolved_cheb_backend(newton_fit)` at the
  DISPATCH site, not where `_spline_data` is built.  Two reasons: only a
  dispatch needs the value, so a spline-only or never-pooling process does not
  compile a numba kernel purely to describe a payload (hence the `None` return
  for a non-polynomial fit); and the pinned value must be the branch in force at
  dispatch time.
* **Worker.**  `_newton_invert_chunk` reads the key and passes it to all three
  evaluators.  A payload with NO key -- one written before the pin -- keeps the
  historical "resolve locally" behaviour, mirroring the `get('newton_fit',
  'spline')` and `get('newton_max_iters', ...)` tolerances beside it.  A payload
  pinned to `'numba'` in a worker whose kernel is `None` raises
  `NewtonWorkerBackendUnavailable`.
* **Parent, again.**  A new `except NewtonWorkerBackendUnavailable` clause sits
  BEFORE the pool-infrastructure clause, falls back to the serial closure, and
  warns ONCE per process naming the cause, the measured size of the error
  (5.2e-14 to 1.4e-11 of the field) and two remedies.  The exception derives
  from `RuntimeError` deliberately: a refactor that drops the specific handler
  loses the diagnostic but still lands in the generic clause, i.e. still falls
  back to serial rather than propagating.
* **...and it latches.**  `_POOL_BACKEND_REFUSED` is a fact about this process's
  workers, so subsequent dispatches go straight to serial instead of re-paying a
  round trip to be told the same thing.  `close_worker_pool()` clears it (it
  tears those workers down, so the next dispatch is entitled to ask a fresh
  set), through the same `_reset_newton_pool_resource_state` the
  unguarded-`__main__` ledger uses and the same `_MAIN_GUARD_LOCK` -- the
  `parallel_amp` ThreadPoolExecutor can reach this path from two threads.

### 1.5 Why not "make the two backends agree"

Because they are two implementations of one formula and neither is wrong;
requiring bitwise agreement between a hand-written njit recurrence and a NumPy
reduction would be a permanent constraint on both, for no accuracy gain.  The
pin costs one dict key and one branch, and it is the same remedy the library
already applied to `newton_max_iters`.

`test_the_two_evaluator_backends_are_not_bit_identical` pins the PREMISE, so if
a future change ever does make them identical the pin gets retired
deliberately, on that test's evidence, rather than discovered from a 1e-11 CI
failure.

---

## 2. FAILURE 2 -- a resource notice with no policy surface

### 2.1 What CI showed

```
tests/unit/test_hammer_h3_traced_nyquist_guard.py:90: in test_h3_guard_suppressed_by_silent_policy
    la.apply_real_lens_traced(E0, prescription=_singlet_f5(), ...)
lumenairy/elements/_lens_traced.py:1484: in _newton_resolve_workers
E   RuntimeWarning: apply_real_lens_traced: the Newton process pool asked for 4
    workers, which projects to ~7.5 GB (1.87 GB per worker at 4096 Newton
    points/chunk and a 140625-point ray-fit grid), but only 12.3 GB is
    available ... running 2 worker(s) instead.
```

on all four python lanes of shard 1 (12.3 / 12.4 / 12.5 GB available).  The
clamp is RIGHT: 4 x 1.87 GB = 7.5 GB against a budget of
`0.5 x 12.3 GB - 2 GB = 4.2 GB`.  The test is also right about what it wants.
The defect is that a physics-guard suppression test could only be satisfied on a
big box.

### 2.2 The contract chosen: (a), the knob -- and why

The task offered two contracts.  This fix takes (a) -- give the cap notice the
same `on_*` routing the other guards have -- for three reasons, all from the
code's own conventions:

1. **The signature's house rule is that guards route through `on_*` actions.**
   `on_undersample`, `on_noncollimated`, `on_aperture_beam` and
   `on_fit_domain_basis` all do.  The cap notice was the only warning
   `apply_real_lens_traced` can emit with no way to say "understood" -- which is
   how it broke a test that had correctly silenced everything it knew about.
2. **The D5 ledger says a knob is the gated thing, not the assertion.**
   `tests/unit/test_fix_d5_fit_domain_basis.py::test_no_new_string_mode_knob_
   ships_without_a_gate` sweeps the whole signature and requires every
   string-default parameter to refuse junk with `ValueError` on an all-default
   call; `_KNOWN_UNGATED = {'on_undersample', 'caustic_band'}` is a disclosed,
   **shrink-only** ledger.  Adding a gated knob needs no ledger change and is
   covered by that sweep automatically -- which is exactly what happened
   (38 passed, no edit to `_KNOWN_UNGATED`).  Writing the gate instead of
   adding a name to that set is the behaviour the ledger's own comment asks for.
3. **Option (b) weakens the test in the direction that matters.**  "Assert on
   the specific guard message" stops catching a NEW warning that fires under a
   policy that said not to -- which is precisely the class of defect this very
   failure is an instance of.  The blanket zero-warning form is the strong one
   and is worth keeping; it just needs every warning on the path to have a
   policy surface, which is now true.

`on_pool_memory: str = 'warn'` therefore ships, with `'silent'` and the
carrier-house aliases `'ignore'` / `'off'`, gated at ENTRY:

```python
on_pool_memory = _pool_memory_policy(on_pool_memory)
```

next to the `on_noncollimated` / `on_fit_domain_basis` gates.  Entry, not inside
the warning branch, and the reason is this failure's own shape: the cap binds on
a 12 GB runner and never on a 256 GB workstation, so a gate inside the branch
would validate the knob on CI and not on a dev box -- the `on_undersample`
pathology the D5 ledger records, reproduced.

There is deliberately no `'error'`.  The clamp exists because the box cannot
hold the pool; raising there would turn a run that completes with a
bit-identical answer into one that does not.  `on_aperture_beam` is the sibling
precedent for a two-value warn/silent knob.

### 2.3 What is NOT routed through it

Rule 1 of the resolver -- the unguarded-`__main__` refusal -- stays unmutable.
It reports that every spawn worker would re-run the caller's whole program,
side effects included; that is a correctness hazard, not a resource notice, and
there is no worker count at which it is acceptable (FIX_POOL_MEMORY sec 4.4).
A caller who quietened the memory cap has not asked to be told nothing about
correctness.  `test_the_unguarded_main_refusal_is_not_routed_through_the_knob`
pins that line so it stays a decision.

### 2.4 FAIL-BEFORE / AFTER, on a pinned 12 GB box

`scratchpad/repro_h3_cap.py` freezes `psutil.virtual_memory().available` and
`lumenairy.memory.get_ram_budget` at 12 GB and makes the exact call the h3 test
makes:

```
PRE-FIX  (no on_pool_memory): RuntimeWarning RAISED -> h3 FAILS here
    apply_real_lens_traced: the Newton process pool asked for 4 workers, which
    projects to ~7.5 GB (1.87 GB per worker at 4096 Newton points/chunk and a
    140625-point ray-fit grid), but...
SHIPPED  (on_pool_memory=silent): NO RuntimeWarning -> the h3 contract holds
```

The per-worker figure, the chunk size and the fit-grid size reproduce CI's
message digit for digit, so the emulation is the CI condition and not a
lookalike.

Both halves are now asserted IN THE SUITE by
`test_the_pool_cap_notice_is_silenced_by_its_own_knob`, which pins the same
12 GB snapshot and requires (1) that the default policy still announces the cap
there -- otherwise the emulation has gone vacuous and half 2 proves nothing --
and (2) that `on_pool_memory='silent'` leaves the same call warning-free.  That
is what makes the h3 file pass on a 12 GB runner and a 256 GB box for the SAME
reason rather than by luck.

`test_h3_guard_silent_on_benign_slow_beam` gets the same knob, for the same
reason: it also uses `simplefilter('error', RuntimeWarning)` and it passed on CI
only because it landed in a shard whose runner did not bind.

---

## 3. TESTS

### 3.1 New (`tests/unit/test_fix_newton_pool_memory.py`, 31 -> 40)

| test | what it pins |
|---|---|
| `test_the_two_evaluator_backends_are_not_bit_identical` | the PREMISE: the two branches agree to 1e-12 relative and NOT bitwise (skips where there is only one branch) |
| `test_the_evaluator_honours_a_pinned_backend` | `backend='numpy'` never consults the kernel getter; `'numba'` does; junk raises |
| `test_the_worker_answers_in_the_pinned_order_not_its_own` | FAIL-BEFORE through the real worker entry point: two payloads differing only in `cheb_backend` give different OPL, and a keyless payload still behaves as it did |
| `test_a_worker_that_cannot_honour_the_pin_refuses_the_chunk` | the refusal, and that a `'numpy'` pin is still served by the same worker |
| `test_the_dispatch_path_pins_the_backend_and_handles_the_refusal` | wiring: the pin, its position before chunking, and the refusal clause BEFORE the generic pool clause |
| `test_the_backend_refusal_latch_says_it_once_and_close_rearms_it` | once per process, re-armed by `close_worker_pool` |
| `test_the_cap_notice_has_a_policy_knob_that_only_moves_the_report` | exactly one notice at 12 GB; all three silencing spellings; the clamped COUNT is unchanged by the knob |
| `test_the_cap_knob_is_gated_at_entry_not_inside_the_warning_branch` | signature default, chain-forwarded default, junk raises even at 400 GB free, and the entry gate is in `apply_real_lens_traced` |
| `test_the_unguarded_main_refusal_is_not_routed_through_the_knob` | the line of sec 2.3 |

### 3.2 Changed

* `test_a_capped_pool_is_bit_identical_to_serial` -- arm 2 (`serial == pool`)
  was a `pytest.skip` on mismatch, because sec 8.1 made it conditional.  It is
  now an unconditional ASSERT.  The uncapped pool is still measured first, but
  only to localise a failure (library contract vs clamp), not to excuse one.
* `test_the_dispatch_path_consults_the_resolver` -- its `src[i:i + 6000]`
  window broke on the first change to the dispatcher after it was written (the
  pin's comments pushed `_get_persistent_worker_pool(n_cpu)` past 6000
  characters and failed a pin whose subject had not moved).  Replaced by
  `_dispatch_closure_source()`, bounded by the NEXT closure at the same indent:
  cannot rot that way, and strictly widens what the pin covers.
* `test_worker_payload_carries_what_the_polynomial_fit_needs`
  (`test_niche_newton_pool_both_fits.py`) -- also requires the payload to pin
  `cheb_backend`, the worker to read it, and the worker to know
  `NewtonWorkerBackendUnavailable`.
* `test_hammer_h3_traced_nyquist_guard.py` -- both `simplefilter('error')`
  tests pass `on_pool_memory='silent'`; two tests added (sec 2.4 and an
  entry-gate/alias check); module docstring records why a resource notice is
  not a physics guard.

### 3.3 Suites run (all green, on the final code)

| suite | result |
|---|---|
| `test_niche_newton_pool_both_fits.py` (the contract that failed) | **23 passed** |
| `test_fix_newton_pool_memory.py` | **40 passed** (was 31) |
| `test_hammer_h3_traced_nyquist_guard.py` (the other failure) | **5 passed** (was 3) |
| `test_fix_d5_fit_domain_basis.py` (the gate-witness suite; new knob is covered by its signature sweep) | **38 passed** |
| `test_niche_audit_e_prepared_and_enums.py` (enum gates + prepared-path kwargs) | **35 passed** |
| `test_audit_w4_glass_registry_meshgrid.py` + `test_v4_16_2_agent_d.py` + `test_v5_1_0_agent_d_split.py` + `test_v5_2_walker_shell_vs_canonical.py` (the `_*_CACHE`/lock/registry meta-walkers) | **127 passed** |
| `test_niche_audit_w3_elements.py` + `w9_traced_determinism` + `c6_fit_guard` + `c11_decentred_fit_arbiter` + `c12_physics_fit_selection` + `p1_traced_tiltaware` | **117 passed** |
| `ruff check` on all four changed files | clean |

---

## 4. WHAT THIS DOES NOT CLOSE

1. **Which side of the split CI was on is not identified.**  The mechanism is
   proved in BOTH directions and the pin closes both, so the fix does not
   depend on knowing -- but no test in `tests/` is known to leave
   `_NUMBA_AVAILABLE` flipped, and FIX_POOL_MEMORY sec 8.1 could not name the
   polluter either.  If it is the parent-loses-numba direction, the pin makes
   the pool answer in the parent's order and nothing else changes; if it is the
   worker-loses-numba direction, CI will now show one `RuntimeWarning` and a
   serial Newton leg on that shard, which is a visible, correct degradation.
2. **The lstsq fit is still assumed reproducible across parent and worker.**
   `_solve_lstsq_thread_safe` runs normal equations through BLAS, whose
   reduction order is thread-count dependent; a spawn worker inherits the same
   environment and CPU count as its parent, so the two agree in every
   configuration measured here (`0.000e+00` after the pin, both split
   directions).  A lane that pinned BLAS threads in the parent only -- e.g. via
   `threadpoolctl` at runtime rather than the workflow `env:` block -- would
   re-open the same CLASS of defect through a different door.  The `unit`
   matrix deliberately does not pin (see `tests/conftest.py`), so this is not
   live today.
3. **No pool speed-up is claimed or measured here.**  Both changes are
   report-and-ordering changes; FIX_D1's measurements stand.
4. **`prepare_real_lens_traced` does not expose `on_pool_memory`** -- it does
   not expose `on_aperture_beam` either, and the default is the historical
   behaviour.  Left alone deliberately rather than widening the signature.
