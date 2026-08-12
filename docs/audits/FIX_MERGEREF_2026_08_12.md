# FIX -- merge-ref CI red on `feat/inverse-map`

**2026-08-12.  Branch `feat/inverse-map` merged locally with `origin/main` @
`80e2da10`.  NO file changed in this worktree except this note.
`CHANGELOG.md` was not touched.  No `git commit`, no `git push`, no `gh`.**

---

## 0. VERDICT

> **The MERGE is clean -- and the CI red is REAL, but it is not the
> merge's.**
>
> This branch is the one of the three that is not behind PR 29: it was cut
> from `a185cfc`, a commit INSIDE PR 29, so it already carries both the
> tilt-quadratic OPL piston fix (`63e6905`) and the rationalized sphere
> eikonal (`a185cfc`).  The merge with `main` adds only the p2/e4 test
> commit (`6ba17cd`) and two documents -- **zero library files** -- and
> everything it adds is green.
>
> The full fast gate then found **three failures with ONE root cause, in
> the branch's own new module** `lumenairy/elements/_lens_imap.py`: its
> module-level LRU cache `_IMAP_CACHE` has no companion lock and is not
> enrolled with the central cache registry.  These are library-code
> defects that are red on the branch tip as well as on the merge ref.
>
> **STOPPED AND REPORTED, NOT FIXED.**  The remediation is a concurrency
> contract for a new module -- lock granularity, whether the hit/miss
> counters are exact under contention, whether the clearer is registered or
> exempted -- which is the branch author's design decision, not a merge-ref
> repair.  Section 3 gives the exact defect, the exact remediation the pin
> itself specifies, and the two lines of prior art.

---

## 1. MERGE

`git merge origin/main`: **clean, zero conflicts.**  What it actually adds:

| path | delta |
| --- | --- |
| `docs/audits/FIX_PR29_BLAST_2026_08_11.md` | new |
| `docs/audits/FIX_TILT_QUADRATIC_OPL_2026_08_11.md` | +37 |
| `tests/unit/test_niche_e4_corrected_relay_oracle.py` | +118 |
| `tests/unit/test_niche_p2_design_battery.py` | +127 |

Zero library files, because the merge base is already inside PR 29.

### 1.1 The seam that could have been a silent gap, and was not

PR 29 captures `_opl_piston` at the OPL-referencing site and re-applies it as
a unit phasor at THREE exit-phase assembly sites in
`apply_real_lens_traced`.  This branch adds +381 lines to that same function
-- exactly the shape of change that can acquire a fourth assembly site the
piston never reaches, which would be a silent global-phase defect invisible
to every single-element test.

It does not.  On the merged tree the evaluator replaces the INVERSION step
that produces `opl_map` and then feeds the same three assembly sites, all of
which carry `_opl_piston_phasor` (lines 10758, 10810, 10821).  The branch
even records the constant in its own probe channel
(`_imap_out['probe_opl_piston']`, line 10797) -- it was developed on top of
the piston fix, not merged against it.  `TRACED_INVERSE_MAP` also ships
`False` and `propagate_traced_carrier_chain` scopes it off for intermediate
legs, so the default path is byte-identical to shipped either way.

---

## 2. WHAT WAS RUN

No failure name was available (no `gh`), so the candidate set was built from
what the merge changes and what the branch touches -- and then, because that
came back green, the WHOLE fast gate was run to settle whether the red was
an environment axis or a test outside the sweep set.  It was the latter.

| suite | mount | result |
| --- | --- | --- |
| p2 + e4 (the two files the merge adds) | Windows py3.14 | 36 passed |
| c15 + c14 + p2 + e4 + tqopl + w3 oracles | WSL py3.12 | 276 passed, 3 skipped, 0 failed |
| **full fast gate**, `-m "not integration and not slow"` | Windows py3.14 | **4 failed, 11389 passed, 74 skipped, 235 deselected** (3 h 12 m) -- 3 in section 3, 1 in section 3.4 |

`test_niche_c14_encapsulation.py::test_the_newest_era_reproduces_the_live_shipped_values`
is the one that could have caught an era-registry collision -- the branch
adds a fourth era `v5.34` to `_traced_flags.ERAS` plus two flags from a
third module.  PR 29 does not touch `_traced_flags.py`, so there is no
collision, and it passes.

**No `xfail` and no `skip` was added by this work.**
`ruff check lumenairy/ tests/unit/` -- all checks passed.

---

## 3. THE REAL DEFECT -- `_IMAP_CACHE` has neither a lock nor an enrollment

Three failing outcomes, one cause:

```text
tests/unit/test_v4_14_2_dispatcher_pin_cache_locks.py::
    test_cache_has_companion_lock[lumenairy.elements._lens_imap-_IMAP_CACHE]
tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py::
    test_every_cache_owning_module_enrolls_with_registry
tests/unit/test_audit_w4_glass_registry_meshgrid.py::
    TestP242WalkerCaseInsensitive::test_main_meta_pin_passes_with_broadened_filter
```

The third is a META-pin that simply calls the second, so it is the same
finding reported twice.

### 3.1 Why the pins are right

`lumenairy/elements/_lens_imap.py` declares a hand-rolled LRU at module
level (line 662):

```python
_IMAP_CACHE: Dict[str, InverseCharacteristic] = {}
_IMAP_CACHE_ORDER: list = []
_IMAP_CACHE_STATS = {'hits': 0, 'misses': 0}
```

and both accessors perform NON-ATOMIC read-modify-write sequences across two
containers:

* `_cache_get` -- `get`, then `_IMAP_CACHE_ORDER.remove(key)`, then
  `append(key)`, plus a `+= 1` on the stats dict;
* `_cache_put` -- `__setitem__`, then `append`, then a `while` loop doing
  `pop(0)` on the order list and `pop(old)` on the dict.

Interleave two threads and the two containers can disagree: a key evicted
from `_IMAP_CACHE_ORDER` but still in `_IMAP_CACHE` leaks past the capacity
bound, and the reverse drops a live entry.  `threading` is not imported in
the module at all, and there is no `_IMAP_LOCK` / `_IMAP_CACHE_LOCK`.

This is not hypothetical for this branch specifically: `apply_real_lens_traced`
takes `n_workers`, and the branch's own change to
`lumenairy/propagators/carrier.py` adds `'lumenairy.elements._lens_imap'` to
`_WORKER_STATE_MODULES`, i.e. it deliberately enrolls this module in the
multi-worker state path.

Separately, `inverse_map_cache_clear()` already exists (line 719) and does
the right thing, but is never handed to the central registry, so
`lumenairy`'s global cache-clear does not drain it -- the exact leak the
v4.16.0 enrollment pin was written for.

### 3.2 The remediation, as the pins themselves specify

Either (1) add the lock and the enrollment:

```python
import threading
_IMAP_LOCK = threading.RLock()      # name must match the pin's expected set
# ... wrap the bodies of _cache_get / _cache_put / inverse_map_cache_clear

try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    import sys as _sys
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'inverse_map_cache',
        lambda: getattr(_this_mod, 'inverse_map_cache_clear')(),
    )
except ImportError:
    pass
```

The pin accepts `_IMAP_LOCK` or `_IMAP_CACHE_LOCK`; prior art for the
enrollment block is `lumenairy/elements/_lens_traced.py:1527`
(`_register_cache_clearer('lens_traced_main_guard', ...)`).

Or (2) add `('lumenairy/elements/_lens_imap.py', '_IMAP_CACHE')` to
`_CACHE_REGISTRY_EXEMPTIONS` with a cited rationale, and the pair to
`_SHARED_LOCK_MAPPING` if one lock is meant to guard it together with a
sibling cache.

**Option (2) is NOT recommended on the evidence above** -- the cache is
mutable, capacity-bounded, and explicitly enrolled in the multi-worker state
path, so it is neither a build-once singleton nor transitively drained.

### 3.3 The fourth failure is NOT this branch's, and not the merge's

```text
tests/unit/test_v5_14_1_device_geometry.py::test_pmm2d_stack_dispersive_sweep
AssertionError: assert np.float64(4.562322741819003e-16) == 0.0
```

A strict byte-identity pin (`np.max(np.abs(a - b)) == 0.0`) between two arms
of a PMM2D dispersive sweep, reading one ulp.  Nothing about it touches the
traced path, the inverse map, or anything PR 29 changed.

**Reproduced identically on the merge BASE** -- worktree `C:/tmp/lum_base`
at `755ad99`, i.e. plain `main` before PR 29 -- on **both** mounts (Windows
py3.14 / numpy 2.4.4 and WSL py3.12 / numpy 2.4.6).  It is therefore
pre-existing on `main` in this local environment and attributable to the
numpy/BLAS build, not to any of the three branches or to the merge.  Out of
scope here; recorded so it is not re-diagnosed later.

### 3.4 Why this note stops here

The choice between those options, the lock's granularity, and whether the
hit/miss counters must be exact under contention are design decisions about
a module this work did not author and was not asked to change.  The rule
followed throughout this campaign is that a merge-ref repair may re-pin a
stale TEST, but a real defect in LIBRARY code is reported, not quietly
patched.  Nothing in this worktree was modified.

**These three outcomes are red on the branch tip as well as on the merge
ref** -- they are not caused by, and cannot be fixed by, the merge.

---

## 4. THE W4-T1 ERA PIN

`test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`
passes here, inside the 276.  The full adjudication is in the sibling
`docs/audits/FIX_MERGEREF_2026_08_12.md` on `feat/carrier-field`.  Summary:

* it compares a live `aberration_tensor` value against Windows-frozen
  decimals at rel `1e-2`;
* its own docstring records a GitHub-runner drift of `3.1e-3` on that value,
  i.e. **3.2x headroom**, and that a tighter earlier version already broke
  CI once on that axis (`1664c92`);
* both local mounts, Linux py3.12 included, read the frozen value to
  `2.2e-9`, so the excursion belongs to the runner's numpy/BLAS build, not
  to Linux;
* `asymptotic*` has no dependency on either file PR 29 changes, and the
  values are bit-identical with the pre-merge sources swapped in.

A recurrence should be treated as that known fragility, not a new defect.
Nothing was re-pinned.

---

## 5. GREEN

Merged worktree.  No library change, no test change, no `CHANGELOG` entry.

| suite | Windows py3.14 / numpy 2.4.4 | WSL py3.12 / numpy 2.4.6 |
| --- | --- | --- |
| p2 + e4 (the files the merge adds) | 36 passed | included below |
| c15 + c14 + p2 + e4 + tqopl + w3 oracles | -- | 276 passed, 3 skipped |
| full fast gate | 11389 passed, 74 skipped, 235 deselected, 4 failed | -- |

Of the four: **three are the one real `_IMAP_CACHE` defect** (section 3), and
one (`test_pmm2d_stack_dispersive_sweep`) is pre-existing on the merge base
`755ad99` on both mounts (section 3.3).
