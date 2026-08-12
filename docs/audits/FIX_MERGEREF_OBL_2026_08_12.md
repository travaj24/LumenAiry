# FIX -- merge-ref CI red on `feat/screen-obliquity`

**2026-08-12.  Branch `feat/screen-obliquity` merged locally with
`origin/main` @ `80e2da10` (PR 29: the tilt-quadratic piston fix + the
rationalized sphere eikonal).  NO file changed in this worktree except this
note.  `CHANGELOG.md` was not touched.  No `git commit`, no `git push`, no
`gh`.**

CI builds the MERGE REF (branch + `main`), not the branch tip.  This note
records what was searched for, what was found, and why nothing was re-pinned.

---

## 0. VERDICT

> **NO defect in the merge, and NO test on this branch was changed.**  The
> merge is textually and semantically disjoint from PR 29, and the merged
> tree is green on every test that touches the seam: 2,239 passed across the
> 65 unit files that reference `apply_real_lens` / `_lens_real` /
> `screen_obliquity`, plus the three files PR 29 adds or modifies, plus the
> era-pin file.  Zero failures.
>
> The reported red on
> `test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`
> is the SAME pin reported red on `feat/carrier-field`, a branch that shares
> no file with this one.  Two disjoint branches cannot both break one
> `asymptotic`-module pin.  Section 3 attributes it, with numbers, to the
> pin's own documented GitHub-runner drift -- a **pre-existing** fragility
> with only 3.2x headroom, not a consequence of either merge.

---

## 1. MERGE

`git merge origin/main` on `feat/screen-obliquity`: **clean, zero
conflicts.**

* the branch changes ONE library file, `lumenairy/elements/_lens_real.py`
  (+398), and adds `tests/unit/test_screen_obliquity.py` plus a validation
  tree;
* PR 29 changes `lumenairy/elements/_lens_traced.py` and
  `lumenairy/propagators/carrier.py`.

Disjoint at file granularity.  The one live coupling is that the branch's
`_screen_obliquity_angle_field` imports `TiltedCarrier` and
`_compute_carrier` FROM `_lens_traced`; PR 29 touches neither of those --
its `_lens_traced` diff is confined to `apply_real_lens_traced`'s OPL
referencing and its three exit-phase assembly sites.  So the import surface
the branch depends on is unchanged.

---

## 2. WHAT WAS RUN, AND WHAT IT FOUND

No failure name was available (no `gh`), so the seam was swept by
construction rather than by guess: every unit file matching
`apply_real_lens|_lens_real|screen_obliquity`, plus the three files PR 29
adds or modifies, plus the era-pin file.

| suite | mount | result |
| --- | --- | --- |
| `test_screen_obliquity.py` | Windows py3.14 | 28 passed |
| 65-file seam sweep + PR-29 files + `test_niche_audit_w3_oracles.py` | WSL py3.12 | **2239 passed, 91 skipped, 18 deselected, 0 failed** (48 min) |
| full fast unit gate, `-m "not integration and not slow"` | WSL py3.12 | **2 failed, 10339 passed, 562 skipped, 235 deselected** (2 h 55 m) -- both pre-existing, section 2.1 |

### 2.1 The two full-gate failures are pre-existing on `main`, not this branch

Both were re-run on worktree `C:/tmp/lum_base`, which sits at `755ad99` --
plain `main` BEFORE PR 29 -- and both reproduce there.

```text
tests/unit/test_v5_14_1_device_geometry.py::test_pmm2d_stack_dispersive_sweep
    AssertionError: assert np.float64(4.562322741819003e-16) == 0.0
```

A strict byte-identity pin (`np.max(np.abs(a - b)) == 0.0`) between two arms
of a PMM2D dispersive sweep, reading ONE ULP.  Nothing in it touches
`_lens_real`, the traced path, or anything PR 29 changed.  Reproduces on the
merge base on BOTH mounts (Windows py3.14 / numpy 2.4.4 and WSL py3.12 /
numpy 2.4.6), so it is a property of the local numpy/BLAS build, not of any
branch.

```text
tests/unit/test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught
    ... should have produced exit code 1; got rc=2 instead
```

An artifact of running a **Windows-hosted `git worktree` from WSL**, not a
code defect.  These worktrees' `.git` files carry a Windows absolute path,
so git from Linux reports:

```text
fatal: not a git repository:
  /mnt/c/tmp/lum_base/D:/Metacept/.../Lumenairy/.git/worktrees/lum_base
```

`git diff v5.1.0..v5.1.1` therefore exits 128, the walker exits 2, and the
V16 self-test sees rc=2 where it wants rc=1.  Every git-dependent test fails
this way under WSL in a worktree.  It PASSES on the Windows mount of this
same worktree, and it reproduces on the merge base under WSL.  CI does a
normal `actions/checkout` (with `fetch-depth: 0` / `fetch-tags: true`), not
a worktree, so this cannot occur there.

**Neither failure is attributable to this branch or to the merge.**

The 91 skips are all pre-existing optional-dependency guards (JAX, astropy,
Optiland, PySide6) plus the documented `_..._PATCH_LOCK` exemptions.  **No
`xfail` and no `skip` was added by this work.**

`ruff check lumenairy/ tests/unit/` -- all checks passed.

---

## 3. THE W4-T1 ERA PIN -- ATTRIBUTED, NOT RE-PINNED

`tests/unit/test_niche_audit_w3_oracles.py::test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`
compares a LIVE arm (`|L[1,0]|^2` from `aberration_tensor` at
`sigma_grid_n=64`) against two STORED constants at rel `1e-2`.  It passes
here, inside the 2,239.

`FIX_TILT_QUADRATIC_OPL_2026_08_11` declares PR 29 moves intensity bytes at
float non-commutativity level; the house rule is that a pin at `1e-2` is six
decades above that.  The measurements confirm the merge cannot reach this
one at all:

1. **No dependency.**  The six `lumenairy/propagators/asymptotic*.py`
   modules -- the whole path behind `fit_canonical_polynomials` and
   `aberration_tensor` -- contain no import of and no call into
   `propagators/carrier.py` or `elements/_lens_traced.py`.
2. **Bit identity under source swap** (measured on the `feat/carrier-field`
   worktree, same merged `main` content): with both pre-merge files copied
   back in, the two merit values are `9.096897522969563e-14` /
   `7.197559771149873e-14` -- identical to the last bit to the merged tree's
   own values.
3. **Not platform- or BLAS-fragile on either mount.**  Windows py3.14 /
   numpy 2.4.4 and Linux py3.12 / numpy 2.4.6 agree to `4e-10` relative, and
   the value is byte-stable across `OMP/OPENBLAS/MKL_NUM_THREADS` = 1 / 4 /
   8 / 16 / unpinned.
4. **Branch-independence.**  The same pin is reported red on
   `feat/carrier-field`, whose changed files (`carrier_field.py`,
   `validation/pipeline/`) intersect this branch's (`_lens_real.py`) in
   nothing.  A single `asymptotic`-module constant cannot be broken by both.

### 3.1 The attribution the numbers support

The pin's own docstring records the axis: the frozen decimals are
**Windows-frozen chirp integrals with a measured cross-platform drift**, and
that "CI Linux reads `9.0687538911e-14` for the first".  Against the frozen
`9.0968975e-14` that is a relative deviation of

```text
(9.0968975 - 9.0687538911) / 9.0968975 = 3.1e-03
```

against a tolerance of `1e-2`, i.e. **3.2x headroom** -- and the docstring
also records that an earlier, tighter version of this pin already broke CI
once on exactly this axis (`1664c92`).

Both of my mounts read essentially the frozen value (deviation `2.2e-9`,
including Linux py3.12), so the `3.1e-3` excursion is specific to the
GitHub runner's numpy/BLAS build, **not to Linux**.  CI runs four Python
versions (3.10 / 3.11 / 3.12 / 3.13) against four different numpy wheels;
only 3.12 and 3.14 are reachable here.

**Therefore: a recurrence of this pin going red should be treated as the
known 3.2x-headroom runner-drift fragility, not as a new defect** -- unless
the reported value differs from `9.0968975e-14` by MORE than about 1e-2,
which would be a hundred times the drift ever observed and would mean
something else.

**Nothing was re-pinned.**  The merge cannot move this number, so there is
no adjudication from the merge to re-pin against, and widening a guard to
silence a symptom whose cause is in the runner would have destroyed the
guard for nothing.  The durable fix, if it recurs, is to re-derive the
assertion from the pin's OWN stated load-bearing claim -- that
`sigma_grid_n=64` separates from the adaptive default, which differs by
`2.9e-2` (R1 = 51.5 mm) / `2.8e-1` (R1 = 60 mm) -- instead of chasing the
frozen Windows decimals.  That is a change to the pin's design and is
deliberately out of scope for a merge-ref fix.

---

## 4. WHY NO "WON DEMONSTRATION" TEST EXISTS HERE

`feat/carrier-field` had one real merge-ref failure of a specific class: a
test that DEMONSTRATED a defect which `main` then fixed, leaving the
demonstration vacuous.  This branch was checked for the same class --
`test_screen_obliquity.py` contains no assertion about the sphere eikonal,
catastrophic cancellation, or any quantity PR 29 moves -- and has none.  Its
residual guards are scored against an exact-ray oracle with common-mode
control at the exit plane, which is insensitive to the global phase PR 29
restores.

---

## 5. GREEN

Merged worktree.  No `xfail`, no `skip`, no `CHANGELOG` entry, no library
change.

| suite | Windows py3.14 / numpy 2.4.4 | WSL py3.12 / numpy 2.4.6 |
| --- | --- | --- |
| `test_screen_obliquity.py` | 28 passed | included below |
| 65-file seam sweep + PR-29 files + w3 oracles | -- | 2239 passed, 91 skipped, 18 deselected, 0 failed |
| full fast unit gate | -- | 10339 passed, 562 skipped, 235 deselected, 2 failed (both pre-existing on `755ad99` -- section 2.1) |

**Zero failures attributable to this branch or to the merge.**

`ruff check lumenairy/ tests/unit/` -- all checks passed, both worktrees.
