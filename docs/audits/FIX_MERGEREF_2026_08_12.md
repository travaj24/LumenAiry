# FIX -- merge-ref CI red on `feat/carrier-field`

**2026-08-12.  Branch `feat/carrier-field` merged locally with `origin/main`
@ `80e2da10` (PR 29: the tilt-quadratic piston fix + the rationalized sphere
eikonal).  ONE test file changed
(`tests/unit/test_carrier_field.py`); no library file was touched.
`CHANGELOG.md` was not touched.  No `git commit`, no `git push`, no `gh`.**

CI builds the MERGE REF (branch + `main`), not the branch tip, so a branch
that was green on its own tip can be red on the ref.  This note records what
was red, what the adjudication was, and what changed.

---

## 0. VERDICT

> **ONE real failure, and it was a test that WON.**
> `test_round_trip_floor_is_the_eikonal_cancellation` asserted that the
> `CarrierField` round-trip residual IS the library's
> `sqrt(r^2+R^2) - |R|` catastrophic cancellation -- linear in `|R|` at
> `k0 * eps * |R|`.  `main` then shipped the rationalized form
> (`a185cfc`), the cancellation disappeared, and the test's own scaling
> assertion went vacuous: both radii read the resample floor and the ratio
> that had to exceed 3.0 came back **1.025**.  The demonstration was
> obsolete because it had succeeded.  It is now INVERTED -- the pre-fix
> expression is monkeypatched back in as a live degraded arm and must be at
> least 3x worse than shipped.
>
> **The second reported failure,
> `test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`,
> is NOT reachable from this merge and was NOT re-pinned.**  Swapping the
> pre-merge `carrier.py` and `_lens_traced.py` back into the merged tree
> reproduces both pinned merit values **bit for bit**.  Weakening a pin that
> the change under test cannot move would have destroyed a guard to silence
> a symptom whose cause is elsewhere.

---

## 1. MERGE

`git merge origin/main` on `feat/carrier-field`: **clean, zero conflicts.**
The branch is purely additive (`carrier_field.py`, `validation/pipeline/`,
two test files, +34 lines of re-export in `propagators/__init__.py`); PR 29
touches `propagators/carrier.py` and `elements/_lens_traced.py`.  Disjoint.

---

## 2. FAILURE 1 -- the eikonal-cancellation fingerprint

### 2.1 What broke

`carrier.py::_exact_sphere_eikonal` at `755ad99`:

```python
return sgn * (np.sqrt(r2 + R * R) - abs(R))
```

and on `main` after `a185cfc`:

```python
return sgn * (r2 / (np.sqrt(r2 + R * R) + abs(R)))
```

Algebraically identical; in float64 the first loses `eps * |R|` metres to
catastrophic cancellation wherever `r << |R|`, i.e. `k0 * eps * |R|` radians
of absolute phase.  `CarrierSpec.phasor_on` calls that routine, so the whole
A -> B -> A round trip inherited it.

The test walked a 100x ladder in `|R|` and asserted the residual rose with
it (`assert 3.0 < rel / prev < 30.0`).  With the cancellation gone the
residual is FLAT, so:

```text
assert 3.0 < (3.807555044383519e-13 / 3.7143446087575964e-13)
```

### 2.2 Measured, both mounts

Round-trip relative L2 of the envelope, shipped arm vs the pre-fix
expression monkeypatched over the module binding `re_reference` actually
calls.  Windows py3.14 / numpy 2.4.4 and Linux py3.12 / numpy 2.4.6 agree to
five figures; the Windows column is quoted.

| `\|R\|` (m) | `dx` (m) | shipped | pre-fix | ratio | `k0*eps*\|R\|` |
| --- | --- | --- | --- | --- | --- |
| 5.0e-04 | 3.0e-07 | 3.714e-13 | 4.343e-13 | 1.17 | 5.325e-13 |
| 5.0e-03 | 1.0e-06 | 3.808e-13 | 2.199e-12 | 5.78 | 5.325e-12 |
| 5.0e-02 | 2.0e-06 | 3.730e-13 | 2.050e-11 | 54.95 | 5.325e-11 |

Bare resample, carrier UNCHANGED, same top radius: **3.693e-13**.  That is
the shipped column, to 1 %: the analytic phasor now costs nothing.

### 2.3 The fix

`test_round_trip_floor_is_the_eikonal_cancellation` ->
`test_round_trip_floor_is_the_resample_not_the_eikonal`.  The name was
asserting something that had become false, which is the failure class this
repo classifies as silent-wrong; it is renamed rather than patched in place.

The pre-fix routine is carried in the test file as
`_subtraction_form_sphere_eikonal` -- verbatim except for its last line --
and installed with `monkeypatch.setattr(carrier_field,
'_exact_sphere_eikonal', ...)`.  It is a LIVE reference implementation, not
a stored number, so both arms are the same round trip in the same process
and nothing in the comparison can drift with the platform, the BLAS or the
numpy build.

Four claims, all of which a regression to the subtraction form violates:

1. **Envelope rule.**  The shipped residual is under `_ROUND_TRIP_FLOOR =
   2.0e-12` at every radius -- an envelope over both mounts with 5.2x
   headroom above the worst measured 3.808e-13, not a fit to either.
2. **Fingerprint, moved onto the degraded arm.**  The pre-fix residual is
   under `k0 * eps * |R|` and within a decade of it, so it is that mechanism
   and not a coincidence.  This is the ORIGINAL fingerprint rule, relocated
   to where it is still true.
3. **Separation.**  Where the cancellation is above the resample floor at
   all (`|R| >= 5 mm`), the pre-fix arm is at least 3x worse.
4. **Shape.**  Over the 100x span the shipped residual is flat (`max/min <=
   2.0`; measured 1.03) because it is the resample, and the pre-fix one
   rises (`max/min >= 10.0`; measured 47).

The `|R| = 0.5 mm` row is pinned in the OTHER direction: the predicted
cancellation there (5.3e-13) has fallen to the resample floor (3.7e-13), so
the two arms must NOT be resolvable, and the test asserts `pre-fix < 3x
shipped`.  That locates the crossover -- exactly where the old defect
stopped mattering -- instead of quietly dropping the row.

### 2.4 Fail-before

Restoring the subtraction form to `carrier.py` in the merged worktree fails
the new test on two independent counts:

```text
AssertionError: |R|=5.000e-03: shipped round trip 2.199e-12 > the resample
                floor bar 2.000e-12
```

and, had the envelope bar been looser, claim 3 as well (both arms become the
same expression, so the ratio is 1.0).  It also fails the sibling control's
new cross-check (below).  `carrier.py` was restored with `git checkout --`
immediately; the worktree carries no library modification.

### 2.5 Sibling control, same cause

`test_round_trip_with_no_carrier_change_is_the_bare_resample` still PASSED,
but its docstring claimed the carrier-unchanged residual "drops two decades"
below the carrier-changed one.  Post-rationalization it does not -- they
agree to 1 % (3.693e-13 vs 3.730e-13).  The docstring is corrected to state
the new truth, the assertion against the OLD form's floor is kept (still
true, still biting), and two lines are added pinning the agreement.  A true
assertion under a false explanation is how the next reader gets misled.

---

## 3. FAILURE 2 -- the W4-T1 era pin: ADJUDICATED, NOT RE-PINNED

`tests/unit/test_niche_audit_w3_oracles.py::test_w4_t1_explicit_sigma_grid_n_64_is_the_pre_fix_default_bit_for_bit`
compares a LIVE arm (`|L[1,0]|^2` from `aberration_tensor` at
`sigma_grid_n=64`) against two STORED constants (9.0968975e-14 /
7.1975598e-14) at rel 1e-2.

`FIX_TILT_QUADRATIC_OPL_2026_08_11` declares that PR 29 moves intensity
bytes at float non-commutativity level.  The house rule is that era pins
comparing two live arms survive that, and that a stored-constant pin at 1e-2
is six decades above it either way -- so if this one is red, something
specific broke.  Four measurements say nothing did:

1. **No dependency.**  `lumenairy/propagators/asymptotic*.py` -- the entire
   path behind `fit_canonical_polynomials` and `aberration_tensor` -- makes
   no import of and no call into `propagators/carrier.py` or
   `elements/_lens_traced.py`, the only two library files PR 29 touches.
   (`grep` for `_lens_traced|apply_real_lens_traced|_exact_sphere_eikonal|
   propagators.carrier` across the six `asymptotic*` modules: empty.)
2. **Bit identity under source swap.**  Both pre-merge files copied back
   into the merged tree, both values re-measured:
   `9.096897522969563e-14` / `7.197559771149873e-14` -- **identical to the
   last bit** to the merged tree's own values.  The merge does not move this
   number at all, not even at the non-commutativity level the fix doc
   declares.
3. **Not platform- or BLAS-fragile here.**  Windows py3.14 / numpy 2.4.4 and
   Linux py3.12 / numpy 2.4.6 agree to 4e-10 relative, and the value is
   byte-stable across `OMP/OPENBLAS/MKL_NUM_THREADS` = 1 / 4 / 8 / 16 /
   unpinned.
4. **Green, in every order tried.**  Alone; in its own 179-test file on both
   mounts; and in one process AFTER the three test files PR 29 adds or
   modifies (`test_fix_tilt_quadratic_opl.py`,
   `test_niche_e4_corrected_relay_oracle.py`,
   `test_niche_p2_design_battery.py`) -- 222 passed -- which is the
   cross-file-state leak a pytest-split shard reshuffle could expose.
   `pytest-randomly` is not in the `dev` extra, so CI order is deterministic.

**Adjudication: the merge cannot reach this pin, so there is nothing in the
merge to re-pin against, and the assertion was left exactly as it is.**  The
pin retains only 3.2x headroom over the 3.1e-3 GitHub-runner drift its own
docstring records, which is thin -- but that is a pre-existing property of
the pin, unchanged by and unrelated to this merge, and widening it here
would be tuning a guard to a symptom whose cause was not found.

---

## 4. GREEN

Merged worktree, both mounts, no `xfail`, no `skip` added.

| suite | Windows py3.14 / numpy 2.4.4 | WSL py3.12 / numpy 2.4.6 |
| --- | --- | --- |
| `test_carrier_field.py` (alone) | 34 passed | 34 passed |
| `test_carrier_field.py` + `test_pipeline.py` | 74 passed | 74 passed (*) |
| `test_niche_audit_w3_oracles.py` | 179 passed | 176 passed, 3 skipped (JAX absent) (*) |
| PR-29 files + w3 oracles, one process | 222 passed | -- |

(*) the WSL figures come from a single combined invocation of all three
files: **250 passed, 3 skipped**.

`ruff check lumenairy/ tests/unit/` -- all checks passed.

Documentation of the numbers behind the bars:
`docs/audits/BUILD_CARRIER_FIELD_2026_08_11.md` S5 (the original
attribution) and section 2.2 above (the post-fix two-arm ladder).
