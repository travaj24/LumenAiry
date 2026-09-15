# VERIFY-WP-B14 -- independent adversarial re-verification of the known reds, the CI hardening and the stacklevel sweep

Target: branch `fix/known-reds-and-stacklevels` at **e61c6467** (also `main`), base **96cb2096**, report
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B14_KNOWN_REDS_REPORT.md`.
Verification branch `verify/wp-b14`, worktree `C:/tmp/lum_vreds`; PRE tree `C:/tmp/lum_vreds_pre` at 96cb2096.

I did not write WP-B14.  Every number below was **re-measured**, not read: the halo bisection was re-run in my own
`git archive` trees, the two-caller instrument was rebuilt from scratch, the GBD byte ladder was re-derived from an
independent allocation model, and the `fft_infra` finding was taken apart until its mechanism came out.

## 0. The two builds, and how every run was pinned

| | Windows | WSL (the CI condition) |
|---|---|---|
| interpreter | CPython **3.14.6** (MSVC 1944) | CPython **3.12.3** (GCC 13.3) |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | `libscipy_openblas64_` 0.3.31.188.0, pthreads, **Haswell** | same version, Haswell |
| `refractiveindex` | installed | **absent** -- the real no-glass-extra arm |
| pyFFTW | 0.15.1 | 0.15.1 |

Every invocation carries `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command line,
`PYTHONPATH` pinned to the tree under test, pytest with `--capture=sys`, and every tail grepped for
`passed|failed|error|no tests ran`.  Every probe prints `lumenairy.__file__`.  The kernel ladder
(`OPENBLAS_CORETYPE` in {HASWELL, NEHALEM, KATMAI, SANDYBRIDGE} x `OPENBLAS_NUM_THREADS` {1, 4}) was confirmed arm by
arm with `threadpoolctl` -- all eight arms reported the architecture they were asked for.

Probes and per-arm JSON: `validation/probe_verify_b14/`.  New decision tests:
`tests/unit/test_verify_b14_known_reds.py` (6 ids, 15-22 s, green on both builds and on four ladder arms).

---

## 1. Verdict table

| # | Claim (WP-B14) | Verdict | My numbers |
|---|---|---|---|
| 1a | c7/c8 are a real regression bisecting to `bbb6c02d` (WP-A26) | **CONFIRMED** | halo beyond 3 w, bound OFF/ON: `bbb6c02d^` and `adff3652` **4.594672922387141e-02 / 8.913411901995892e-04** (ratio 51.5478); `bbb6c02d`, `4bf26c5e`, `e61c6467` **1.5217194665917584e-04** both (ratio 1.00).  Bit-for-bit as reported, in my own archives. |
| 1b | "reachable at order 10 and only order 10" | **RESTATED** | The report's ladder skipped the odd orders.  Measured 5..20: **order 9 reads 4.594672836982075e-02 / 8.913411596874076e-04 (ratio 51.5479)** -- the band is **two orders wide, 9 and 10**.  Everything else is the 1.46e-04 / 1.52e-04 baseline. |
| 1c | The C6 guard still reaches (delta 7.01e-02); the G8 refusal is a passenger | **CONFIRMED (not independently re-derived)** | Guard on vs off at order 10 moves the halo from 0.0 to 4.59e-02 on my fixture, which is the same statement; I did not re-run the G8 inverse-map arm. |
| 1d | Fix layer = the fixtures state `decentred_fit_poly_order=10`; WP-A26's default is sound | **CONFIRMED, with a scope finding** | The layer is right and no bar moved.  But see **F1**: the defect class the two files exist for is **still reachable at the shipped default 16** -- 2.3728e-02 / 7.8841e-05, **ratio 301.0x**, on the same `_GHOST` construction at `alpha=3.0, cx=1.5 mm, z=12 mm`, and 3164x at `alpha=3.5, cx=1.35 mm, z=13 mm`.  So WP-A26 left **no** accuracy hole and the C8 bound is not dead code -- but the two fixtures now exercise a retired default, and nothing exercises the shipped one. |
| 2 | T3-1 red only on SANDYBRIDGE/4; restated, no bar loosened | **CONFIRMED** | PRE at SANDYBRIDGE/4: `AssertionError ... assert 0 >= 1` with the identical screened census `[('uncoated ns=3', 10, 0, 2, 0), ('25 nm coat ns=8', 6, 0, 0, 1), ...]`.  POST: `1 skipped` with the reading, message byte-identical to the report's.  Diff read line by line: the decay claim moved from `len(dJs)==3` to `len(dJs)>=2` (**strictly stronger**), the hard "measured nothing" failure is kept, only ladder completeness is gated. |
| 3 | The glass one-shot pin is a partial reset of coupled state | **CONFIRMED (cause) / REFUTED (reproduction)** | Mechanism reproduced exactly without pytest: 5 calls cold -> **1** warning; after clearing `_validity_warned` alone -> **0**; after `clear_asm_caches()` -> **1**; `clear_asm_caches()` takes `(1,1) -> (0,0)`.  But the report's 3.1 ("red only when it ran AFTER the meshgrid file ... green in the reverse order and green alone") is **wrong**: on the PRE tree the validity file is red in **both** orders **and alone** (`1 failed, 7 passed`), because its own first test memoises `('N-BK7', 200e-9)`.  It was intra-file self-poisoning, not an inter-file order dependence.  The fix is correct and is in fact **stronger** than claimed. |
| 4a | Dense GBD: declared 16 B/cell-col, measured 72.0-96.8, up to 6.0x, 3073 MB at a 512 MB budget | **CONFIRMED** | My ladder reproduces all seven reported cells to the digit: 196.5 / 151.1 / 1208.5 / 1360.1 / 384.1 / 387.1 / **3073.5 MB**, B/cell-col 93.7 / 72.1 / 72.0 / 72.1 / 96.5 / 96.8 / 96.1. |
| 4b | `'legacy'` default byte-identical; `'measured'` 387 MB, differ by 2.1e-17 | **CONFIRMED** | legacy 3073.514722 MB / measured 387.063026 MB, `rel = 2.117089433990483e-17`, `array_equal` False (round-off only), unknown value behaves as legacy (`array_equal` True). |
| 4c | "`'measured'` makes the budget a bound" | **BOUNDED** | True only above one beamlet column.  N=256: 512 MB -> 0.76x (bounded), 16 MB -> 0.60x, **4 MB -> 2.39x**, **1 MB -> 9.58x**.  Independent model fitted here and agreeing with every ladder row to <= 0.2 %: `peak = Ny*Nx*(48 + C*chunk)`, `C = 72` single chunk, `C = 96` multi-chunk.  So the legacy constant under-counts the per-column term by exactly 96/16 = 6.0 (that is where the saturation comes from) **and** there is a fixed ~48 B/cell term the chunk arithmetic does not model at all -- no constant can make `mem_budget_mb` a bound below ~`N^2 * 48` B. |
| 5a | Two-caller instrument: 2/4 misattributed before, 0/4 after | **CONFIRMED and widened** | My own instrument (synthetic library frames, 13 cases x 3 depths): base **26 of 36** emissions misattributed, head **6 of 36**; all 6 residual are one site in `propagators/asm.py` reached through `carrier.py` -- an unswept module, exactly the open work section 5.3 records.  Zero from the three swept modules. |
| 5b | The sweep moved no values | **CONFIRMED** | 12 numeric entry points through the three modules, md5 of the raw bytes: **12 of 12 identical** base vs head. |
| 5c | Helpers take `stacklevel=None`; explicit integers keep their meaning | **CONFIRMED** | Read line by line; `_warn_undeduped`'s `caller_stacklevel() - 1` offset is correct (its `sys._getframe(stacklevel)` counts from its own caller, one frame in from `caller_stacklevel`'s counting). |
| 6a | py3.10 tomllib fallback | **CONFIRMED** | Emulated by blocking `tomllib`/`tomli` for that module only after pytest's own config parse.  PRE: `Interrupted: 1 error during collection`.  HEAD, no parser: `7 passed, 4 skipped`.  HEAD, parser present: `11 passed`. |
| 6b | PEP 701 digest scheme is version-independent | **CONFIRMED on the two interpreters I have** | All 123 registered modules, both fingerprints, recomputed outside pytest: **0 AST mismatches, 0 token mismatches** on py3.14.6 AND py3.12.3, and the sha256 **of all 246 fingerprints is identical on the two**: `b9af7d83175a807faf8aeeee5cf1563d7cbc126168d2227a84b80ba52e4a3f83`.  110 of 123 modules carry an f-string, 13 do not -- the report's split, reproduced.  The 49/49 reproduction of CPython 3.11's own digests I **could not** verify (no 3.11 here); see section 5. |
| 6c | a8 glass: `glass.py` untouched, ImportError is the real index, no `pytest.skip` | **CONFIRMED, on a real no-package arm** | `git diff 96cb2096 e61c6467 -- lumenairy/glass.py` is empty.  `a8_glass + a8_verify`: **64 passed, 0 skipped** with the package present (Windows) and **64 passed, 0 skipped** with it genuinely absent (WSL) -- a stronger arm than the report's blocker fixture, same count both sides. |
| 6d | Six platform pins root-caused | **CONFIRMED for the two I re-derived; READ for the rest** | The b5 `sin(pi/4)` claim is exact: MSVC returns `sin` and `cos` both as `0x3fe6a09e667f3bcd`; glibc returns `sin` as `0x3fe6a09e667f3bcc`, one ULP low, `|sin-cos| = 1.1102230246251565e-16`.  The a6 re-association and the a11/b8 `temp_elide` premise are corroborated by my own elision measurement (see F3) and by the two WSL skips, which fire with their measured premise in the message. |
| 6e | `--maxfail` 10 -> 50; slow lane 5 -> 8 splits; durations cover the moved files | **CONFIRMED** | Workflow read: fast lane `--maxfail=50`, `--splits 5` with `shard: [1..5]`; slow lane `--splits 8` with `shard: [1..8]`, step cap 30 -> 45, job cap 35 -> 50.  Slow selection re-collected here: **935 ids, 935 with durations, 0 missing, 9630.0 s total -> 1926.0 s at 5 splits, 1203.8 s at 8** -- every digit of section 6.5.  Fast lane: 15 272 ids, 7 without durations (all pre-existing, three unrelated files); **every id of all 17 touched files has a duration, including the 5 new GBD ones**. |
| 7a | Second BLAS classification: w3 `test_w4_t1_...`, SANDYBRIDGE, rel 1.223e-08 | **CONFIRMED** | PRE: HASWELL t1/t4 pass; **SANDYBRIDGE t1 and t4 fail** with `assert (0.7446794710376422 / 60906429.913441636) < 1e-08` = **1.2227e-08**.  POST: SANDYBRIDGE t1/t4 and KATMAI t1/t4 all pass; the whole file at SANDYBRIDGE t4 reads **181 passed**. |
| 7b | `fft_infra` DETERMINISM DEFECT: `_ifft2(_fft2(E)*H)` is not a function of its inputs | **REFUTED as to mechanism; the user-visible defect REMAINS, restated** | See section 3.  The transforms **are** functions of their inputs (6-8 identical evaluations -> one byte image, every mode, both builds).  The A/B is made by NumPy's temporary elision, which the ping-pong flips by returning a non-owning view instead of a copy; on the Linux NumPy build the right-operand elision of a complex128 multiply moves the last bits. |
| 8 | `test_niche_d8_congruence_workers.py` is green on the committed tree | **CONFIRMED** | **36 passed** on Windows (35.02 s) and **36 passed** on WSL (29.50 s) at e61c6467.  No regression in D's tree.  The warning output names `test_niche_d8_congruence_workers.py:564` -- the caller's frame -- which is the sweep working through the congruence-worker path. |

---

## 2. Defects found

### D1 (P2, introduced by e61c6467, CI-relevant) -- the a11 RETAIN guard asserts a fact about PROCESS HISTORY

`tests/unit/test_audit2609_a11_polar_sources_infra.py:578`, in
`test_z3_estimate_lens_memory_real_bounds_apply_real_lens`, the new guard
`assert 5.5 < held < 7.0` ("the 6.0 of N-sized FFT/ASM cache it builds") holds only while those caches are **cold for
this N**.  Any earlier test in the same process that has touched N = 512 leaves the call nothing to build.

**Reproducer** (both builds, deterministic, kernel-independent):

```
pytest tests/unit/test_audit2609_a6_carrier.py \
       "tests/unit/test_audit2609_a11_polar_sources_infra.py::test_z3_estimate_lens_memory_real_bounds_apply_real_lens"
  -> 1 failed, 80 passed        (Windows py3.14 AND WSL py3.12)
     "the measured call retained 2.00 full complex grids"
the id alone            -> 1 passed
the a11 file alone      -> 104 passed
base commit 96cb2096    -> 81 passed        (the guard did not exist)
```

It also fires in every arm of my kernel ladder over the six pin files (`1 failed, 711 passed` on HASWELL t1/t4 and
NEHALEM t1/t4, same id, same reading) -- i.e. it is not a BLAS fact, it is a selection fact.

**Measured, byte-identically on both builds** (`probe_v8_a11_retain_state.py`):

| state | retained | peak | estimate | est/peak | RETAIN guard | the test's own bar |
|---|---|---|---|---|---|---|
| cold | 6.09 grids | 46.5 MB | 49.5 MB | 1.064 | pass | pass |
| **warm** | **1.00 grids** | 25.2 MB | 49.5 MB | **1.965** | **fail** | **fail** |
| drained | 6.01 grids | 46.2 MB | 49.5 MB | 1.071 | pass | pass |

In the warm state **both** bars fail -- the guard is doing its job, it is the reading that is not engineered.  Note
the CI consequence: the fast lane assigns files to shards by recorded duration, so whether `a6_carrier` and
`a11_polar_sources_infra` land in one process is not the test's to decide.

**Exact fix (requested, outside my ownership):** add one line after the warm-up call in that test --
`la.clear_asm_caches()` -- which empties the ASM caches and the pyFFTW plan cache together and restores the cold
reading on any process history (measured above: 6.01 grids, est/peak 1.071, both bars clear).  This is not a
relaxation; no bar moves.  The property is pinned meanwhile by
`test_verify_b14_known_reds.py::test_the_lens_memory_reading_is_engineered_not_inherited`.

### D2 (P3, documentation) -- the glass fix's own docstring states a reproduction that is false

`tests/unit/test_v4_16_0_agent_d_validity_ranges.py`'s new fixture docstring says the pin was red "only when it ran
AFTER `tests/unit/test_audit_w4_glass_registry_meshgrid.py`", and the WP report and the CHANGELOG repeat it.
Measured on the PRE tree: the file is red **alone** (`1 failed, 7 passed`), red with the meshgrid file first
(`1 failed, 23 passed`) and red with it last (`1 failed, 23 passed`).  The poisoner is the file's own first test,
`test_validity_warning_emitted_outside_range`, which memoises the same `('N-BK7', 200e-9)` pair.  Only the single id
in isolation is green.  Right conclusion, wrong reproduction -- the shape `docs/TESTING_STANDARDS.md` calls the most
dangerous one.  Suggested edit: drop "only when it ran AFTER ..." for "whenever any earlier call in the process has
memoised the pair -- including this file's own first test".

### D3 (P3, scope) -- "`'measured'` makes the budget a bound" is true only above one column

`tests/unit/test_wave5_gbd_dense_mem_budget.py::test_the_measured_accounting_makes_the_budget_a_bound` asserts it on
one cell; the module note in `gbd.py` states it without scope.  Measured (section 1, 4c): at N = 256 a 4 MB budget
reads 2.39x and a 1 MB budget 9.58x in `'measured'` mode, because the chunk floors at 1 and the fixed ~48 B/cell term
is outside the chunk arithmetic entirely.  Pinned two-sided by
`test_verify_b14_known_reds.py::test_the_measured_accounting_bounds_the_budget_only_above_one_column`.

### D4 (P3, contract) -- `set_fft_double_buffer`'s byte-identity claim, correctly scoped

See section 3.  The knob's registered doc says "values are byte-identical either way".  That is true of the
**transform's values** (verified here, unconditionally, both builds) and false of any **downstream NumPy expression**
on the returned array, on the Linux build.  The one-line accurate statement is "the transform's values are
byte-identical either way; the object handed back is a live workspace view in one mode and a private copy in the
other, which NumPy's temporary elision can distinguish".

### F1 (finding, not a defect) -- the halo fixtures now exercise a retired default

Measured at the shipped default order 16 over a 30-cell sweep of the `_GHOST` construction's own free parameters
(`probe_v1b_default_order_reach.py`): the C8 bound still removes a manufactured lobe, **ratio 301.0x** at
`alpha=3.0, cx=1.5 mm, z=12 mm` (2.3728e-02 -> 7.8841e-05) and **3164x** at `alpha=3.5, cx=1.35 mm, z=13 mm`, both
power-subtractive.  So the answer to "did WP-A26 leave an accuracy hole the fixtures were the only witness of" is
**no** -- and the C8 guard is emphatically not dead code at the shipped default.

I did **not** turn that into a test.  A 27-cell neighbourhood sweep around the strongest cell
(`v1c_default_order_neighbourhood_win.json`) reads ratio 1.00 in 23 of 27 cells: the default-order stimulus is a
knife edge in `(alpha, cx, z)`, which is the S1/S5 shape `TESTING_STANDARDS` forbids pinning.  A durable
default-order fail-before needs a ladder over cells AND a cheaper geometry than 768^2 (the sweep costs ~14 min).
**Requested as follow-up work for the owner of niche C7/C8**, with the measurements above as the starting point.

---

## 3. The `fft_infra` characterisation (the brief's item 7b)

**What the report claims** (section 6.8): with the ping-pong on, `_ifft2(_fft2(E) * H)` "is not a function of its
input values alone"; it vanishes with `set_fft_double_buffer(False)` and with `USE_PYFFTW=False`; only at
`n >= FFTW_MIN_SIZE`; only on Linux; mechanism not established.

**What I measured.**  Four probes, in order.

1. **The transforms are deterministic.**  `probe_v7_fft_determinism.py`: eight identical evaluations of the round
   trip, of `_fft2` alone and of `_ifft2` alone on a fixed operand, at n = 128 / 256 / 512, in all three modes ->
   **one distinct byte image every time**, on both builds.  The two ping-pong slots exist and are distinct
   (`n_slots=2`, different plan ids, buffer addresses 0 and 32 mod 64), and using them alternately changes nothing.

2. **The two routes hand `_ifft2` different OPERANDS.**  `probe_v7b_fft_ab.py`, tapping both calls:
   at n >= 256 in shipped mode, `fft_operand_identical = True`, `fft_result_identical = True`,
   **`ifft_operand_identical = False`**, `ifft_result_identical = False`.  In `single_buf` and `no_pyfftw`, all four
   are True.  So the inverse transform is innocent: it is fed different numbers.

3. **The difference is made by the multiply, and it is not clobbering, alignment, or `np.exp`.**
   `probe_v7c` / the `v7d`/`v7e`/`v7f` bench: the forward workspace hashes the same at return and at the `_ifft2`
   call (no clobber); `pyfftw` view vs numpy copy vs numpy view at the same 64-byte offset all multiply identically;
   `np.exp(1j*P)` is bit-stable across repeats, across operand alignments and across an intervening FFT.  The
   surviving asymmetry is which operand is an unreferenced temporary:

   `probe_v7d_elision.py`, four spellings of the SAME product:

   | build | both temporaries | neither (named) | left temporary | **right temporary** |
   |---|---|---|---|---|
   | **Linux**, numpy 2.4.6 | = named | - | = named | **differs**, rel 1.0e-16 .. 1.8e-16, 16-17 % of the doubles, at n >= 128 |
   | **Windows**, numpy 2.4.4 | = named | - | = named | = named, rel 0.0, every n |

4. **Putting it together.**  In `_fft2(E) * H` the left operand is whatever `_fft2` returned.  With the ping-pong
   ON that is a **non-owning aligned view** into the pyFFTW workspace, which NumPy's `temp_elide` cannot claim, so
   the RIGHT temporary (`H`) is elided into -- the row that moves.  With the ping-pong OFF `_fft2` returns
   `buf.copy()`, a numpy-owned temporary, so the LEFT one is elided -- the row that equals the named form.  Below
   `FFTW_MIN_SIZE` pyFFTW is not used at all, so the left operand is always numpy-owned: that is why the effect
   starts at 256 in the FFT setting while the underlying elision asymmetry starts at 128.

**Consequences.**

* The defect is **real and user-visible**, so the report's decision to raise it was right.  Its **location is not
  `fft_infra`'s transforms** and it is not a determinism defect: `_fft2`/`_ifft2` are functions of their inputs.
* What `fft_infra` does wrong is narrower and easier to state: the ping-pong changes the **kind of object** handed
  back (live view vs private copy), and that is observable in the last bits of any downstream elementwise
  expression on the Linux NumPy build.  Either the docstring is scoped (D4) or the dispatchers return a copy at the
  shapes where it matters.
* One level down there is an **upstream question for NumPy**: on the Linux 2.4.6 build the complex128 multiply is
  not elision-invariant.  `probe_v7d_elision.py` is a five-line, lumenairy-free reproducer suitable for filing.
* **Not fixed here**, as instructed.  It becomes its own item, with the mechanism now established.

---

## 4. Runs

All with the pinning of section 0; every tail grepped.

| run | Windows py3.14.6 | WSL py3.12.3 |
|---|---|---|
| the 17 touched + consumer files | **1702 passed, 0 failed, 0 skipped** (556.62 s) | **1699 passed, 3 skipped** (639.96 s) -- the 3 skips are the `refractiveindex` importorskip and the two a11/b8 elision-premise gates firing with their reading |
| `test_niche_d8_congruence_workers.py` | **36 passed** (35.02 s) | **36 passed** (29.50 s) |
| census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget.py` + every CHANGELOG-reading test (38 files) | **968 passed, 11 skipped, 0 failed** (388.93 s) | -- |
| `a8_glass + a8_verify` | 64 passed, 0 skipped | 64 passed, 0 skipped |
| `a15a_packaging`, parser present / absent / PRE-absent | 11 passed / 7 passed 4 skipped / `Interrupted: 1 error` | -- |
| a17 fingerprints, no pytest, 123 modules | 0 + 0 mismatches | 0 + 0 mismatches, same digest-of-digests |
| kernel ladder over the six pin files (a14, a6, a11, b8, b5, w3) | HASWELL t1/t4, NEHALEM t1/t4, KATMAI t1 all `1 failed, 711 passed` -- **the single failure is D1 (`a11:578`) on every arm**, so it is a selection fact and not a BLAS one.  KATMAI t4 and the two SANDYBRIDGE arms of that 712-id selection were still running when this was written; the two ids in it that ARE build-dependent were run at SANDYBRIDGE t1 and t4 separately (rows below) | -- |
| T3-1 id, SANDYBRIDGE t4 | PRE `1 failed`; POST `1 skipped` with the census | -- |
| w3 `test_w4_t1_...`, SANDYBRIDGE t1/t4 | PRE `1 failed` (rel 1.2227e-08); POST `1 passed`; whole file `181 passed` | -- |
| `tests/unit/test_verify_b14_known_reds.py` (new) | **6 passed** (15.22 s); SANDYBRIDGE t1/t4 and KATMAI t1/t4 **6 passed** each | **6 passed** (14.44 s) |
| `ruff check lumenairy/ tests/ validation/probe_verify_b14/` (WSL) | -- | **All checks passed!** |

The report's own section 7.5 records one pre-existing failure,
`test_public_api.py::test_installed_metadata_version_matches_source_version`.  It does **not** reproduce here: the
editable install now reports 5.47.0 and the id passes, so my sweep is 0 failed.

---

## 5. What I could not verify

1. **The 49/49 reproduction of CPython 3.11's own digests.**  No 3.11 interpreter exists on this box (`py -0p`
   lists 3.14.6 and 3.13.13; WSL has 3.12.3).  What I CAN say is that the scheme is stable across the two
   interpreters I have, hash for hash over all 123 modules, and that the design choice is right for the reason the
   report gives: the shipped route collapses each **physical** `FSTRING_START..FSTRING_END` run, which is what a
   pre-3.12 tokenizer emitted (one `STRING` token per physical literal).  An `ast`-based emulation collapses
   implicit concatenation groups instead and reads 212 constructs where the tokenizer reads 971 in `carrier.py` --
   i.e. the obvious alternative would NOT have reproduced 3.11.  Only a green 3.10/3.11 matrix closes this.
2. **The py3.10 lane end to end.**  The collection abort and the three parser arms are reproduced by emulation; the
   ~2 900 ids that lane has never run are still unrun.
3. **The access violation.**  Not reproduced here either.  The bound in section 1 (4a) stands; causation does not.
4. **The C6 guard / G8 passenger arms (1c)** were read, not independently re-derived.
5. **CI itself.**  A matrix run on e61c6467 is the authority on the CI classes; my work is the library-touching
   parts and the claims.  Note that D1 is a CI-relevant failure my ladder found and CI may or may not have hit,
   depending on the shard split.

---

## 6. Ship recommendation

**Do not tag 5.47.0 on e61c6467 as it stands.  Tag a follow-up commit that carries the D1 one-liner.**

Reasoning, in the order it matters:

* **D1 is a real red in a plain selection, on both builds, introduced by this commit** (`1 failed, 80 passed` where
  the base is `81 passed`).  Whether it fires in CI is decided by pytest-split's shard layout, which is exactly the
  "green locally, red on the runner that mattered" shape this package exists to remove.  The fix is one line
  (`la.clear_asm_caches()` after the warm-up), it moves no bar, and it is measured on both builds.
* Everything else in the package is sound.  The four reds are correctly diagnosed and correctly fixed; the CI
  classes reproduce; the sweep is byte-neutral over 12 entry points and takes the misattribution count from 26/36
  to 6/36 with the residue correctly scoped to an unswept module; the slow-lane and durations arithmetic is exact.
* The three P3 items (D2 doc, D3 scope, D4 contract) are corrections to written claims, not to behaviour.  They
  can ride the same follow-up or a later one; none of them blocks a tag.
* `DENSE_MEM_BUDGET_ACCOUNTING = 'legacy'` **as the default is defensible and should stay** for this release: it
  preserves byte-identity on a default path, which is the house rule, and the honest mode is one assignment away.
  But it IS a silent hazard as shipped -- a caller who sets `mem_budget_mb` to fit a machine can be handed 6x what
  they asked for, and nothing at call time says so.  My recommendation to the maintainer: keep `'legacy'` for
  5.47.0, and either (a) flip the default in the next MINOR with the Migration note, or (b) keep `'legacy'` and
  emit a one-shot notice from the dense path when `mem_budget_mb` is set, naming the 6x factor and the two
  mitigations -- a warning is not a byte move.  D3's scope sentence should go into the constant's note either way.
* The `fft_infra` item should be re-filed with the mechanism from section 3 rather than as an FFT determinism
  defect: the fix surface is the dispatcher's return type and one docstring, plus an upstream NumPy report.

## 7. Commits

| | |
|---|---|
| target verified | `e61c6467` (`fix/known-reds-and-stacklevels`, also `main`) |
| base | `96cb2096` |
| bisection witness | `bbb6c02d` (WP-A26) and `bbb6c02d^`; `adff3652` (v5.45.1); `4bf26c5e` (5.47.0) |
| this verification | branch `verify/wp-b14`, commit recorded in the commit message trailer of `verify(tests): WP-B14 ...` |
