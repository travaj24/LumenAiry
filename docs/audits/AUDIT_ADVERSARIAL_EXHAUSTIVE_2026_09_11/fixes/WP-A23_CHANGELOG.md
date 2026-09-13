# WP-A23 — changelog text

Assembled by the orchestrator into `CHANGELOG.md`.  Test-and-instrument only:
**no library file was changed**, so there is no user-visible behaviour change
and no migration note is required.

---

### Fixed -- CI kernel census: the committed table had gone stale under a default change, and the consistency gate said so

`tests/unit/test_ci_kernel_consistency.py::test_this_arm_agrees_with_the_committed_census`
was RED: the census (recorded 2026-09-11, commit `da377e0b`) records
`sliver/pmm1d@1e-05` as `wrong` on all ten measured arms, while every arm on
this tree read the eight 1-D rows `correct`, and two `pmm1d_interface/warned@`
rows took `warn` at answer class `correct` — a decision the census's rule table
forbids.  The gate was working: it is built to fail when the table stops
describing the library.

**The cause was measured, not attributed.**  Audit finding G2 (WP-A12,
`56a76f22`) raised `elements/pmm/stack.py::_MIN_FEATURE_DEFAULT_FRAC` from
`period*1e-5` to `period*1e-3`.  The census's 1-D fixture is a cross-layer wall
pair 1e-04 / 1e-05 of a period apart and did not pin `min_feature`, so the new
default — two decades above both separations — snapped both pairs to
coincidence: the two layers become geometrically identical and the
ill-conditioned interface every row in that section is about ceases to exist.
Flipping only that constant in process, in one interpreter, and back again:

| `min_feature` | `rcond` @1e-04 | `rcond` @1e-05 | `max(R+T)` @1e-05 | class | sliver guard |
|---|---|---|---|---|---|
| `period*1e-5` (the census's) | 9.6940e-11 | 9.7297e-13 | **3.6124215325** | wrong | refuse |
| `period*1e-3` (shipped) | 4.2942e-04 | **5.2104e-04** | **1.0000000000000628** | correct | return |

Nine decades of `rcond`, reversible in process, with the band and branch-cut
sections byte-identical across the flip as the control.

**The repair, and it is a hardening, not a relaxation.**

* `validation/probe_ci_kernel_sweep/probe_decisions.py` now **pins**
  `min_feature = period*1e-5` as a fixture parameter (`_1D_MF_PINNED`) — the
  same remedy WP-A12 applied to its three inherited fixtures and VERIFY-A12
  applied to the mortar round-2 rationale test this section exists to mirror.
  The fixture is ill-conditioned again, which is strictly harder.
* The **shipped default is censused beside it** under a new `@mf-default` tag,
  so the next default change is a row rather than an erasure.
* `pmm1d_interface/warned@` now measures the **energy tripwire** — the voice
  its own comment always named — instead of "did any warning come out".  G2
  also added a deliberate geometry notice (`_pmm_union_grid: snapped N
  pair(s)`) that speaks on a correct answer; counting it as a warning made the
  census report a guard crying wolf when no guard had spoken.  The
  answer-independent voices get their own row, `pmm1d_interface/notices@`.
  **The rule table is byte-identical**: a correct answer still may not be
  warned about.
* Environment advisories (a `RuntimeWarning` saying a package is "not
  installed") are classified separately and kept out of every decision, with
  the count retained as a reading.  Measured: the WSL build here raises
  `memory.py`'s "psutil not installed" three times per solve and Windows zero,
  which would otherwise have made two builds disagree on a row with no answer
  class.

`tests/unit/test_ci_kernel_consistency.py`: **7 ids (one failing) → 8 ids, all
passing**, 2.45 s.

### Changed -- the kernel census is re-recorded, dated, and carries its own provenance

`validation/probe_ci_kernel_sweep/decisions.json`: **12 arms → 24**, now in
three explicit kinds, so an archived reading can never again be mistaken for a
current one:

* **12 LIVE** arms measured 2026-09-12 on `audit-fixes-2026-09` `2622449f` —
  2 builds × 5 OpenBLAS kernels (SkylakeX / Haswell / Sandybridge / Nehalem /
  Katmai) × 2 thread widths, each carrying its date, its tree, the library
  version and the `min_feature` default it measured;
* **11 HISTORICAL** arms — the 2026-09-11 census, **kept and marked, not
  deleted** (`"historical": true`, re-keyed `<arm>@2026-09-11`, with the tree
  and the reason).  Not one measured value in them was touched.  They are the
  only evidence of a host that could not execute AVX-512, and the before-side
  of the change above;
* **1 SYNTHETIC** arm — the transcribed CI runner, unchanged.

`test_ci_kernel_consistency.py` gained the requirement that stops this
recurring: the coverage spans stay asserted of every arm that was RUN, and a
second, **freshness** requirement is asserted of the LIVE arms alone (at least
six, two kernels, two widths, all recorded on the tree the file names).  A
requirement an archived arm can satisfy is satisfied forever, which is how the
old table kept passing while its rows described a library that no longer
existed.  `merge_arms.py` refuses a historical arm with no date, and an arm
marked both historical and synthetic.

### Added -- the census can name its own BLAS kernel without `threadpoolctl`

`probe_decisions._arm_id` falls back to reading `openblas_get_corename()` out
of the loaded library with `ctypes` when `threadpoolctl` is unavailable —
the same runtime-dispatch read-back `threadpoolctl` performs, so the census's
"measured, not requested" invariant is untouched.  This matters on two
machines the census has to speak about: the GitHub runner (which is why the
transcribed arm is called `CI-unknown-t1`) and this workstation, where a
declared core dependency is simply not installed.  Without it four different
kernels all want the same arm key and the census's kernel axis collapses
exactly where it is meant to discriminate.  Wheel symbol spellings measured:
`scipy_openblas_get_corename` (scipy's LP64 build) and
`scipy_openblas_get_corename64_` (numpy's ILP64 build); the unmangled name is
absent from both.

### Fixed -- the 2026-09-11 OPEN item: why the CI runner answers these fixtures correctly

`docs/audits/CI_PREMISE_GATES_2026_09_11.md` §2 recorded as **OPEN** why the CI
runner solves the ill-conditioned 1-D interface correctly where every local arm
got it wrong, having ruled out the thread width, the numpy version and the OS —
and having been unable to rule the BLAS micro-kernel either way, because the
2026-09-11 host could not execute the AVX-512 kernels at all (SIGILL).  On an
AVX-512 host it is reproducible:

| arm | `R+T` @1e-04 | `R+T` @1e-05 | class |
|---|---|---|---|
| CI runner (transcribed, 5.45.0 matrix) | 1.0000003658397656 | 1.0000010471871335 | correct |
| **`WSL-SkylakeX-t4`** (measured 2026-09-12) | **1.0000003658397656** | **1.0000010471871335** | correct |
| `WIN-SkylakeX-t1`, same silicon | 1.0000004784011818 | **3.6124215324602997** | wrong |

**Bit for bit on both readings, over two independent runs.**  So the CI arm is
a Linux + AVX-512 arm; the divergence is a micro-kernel effect after all; and
the campaign now has a local reproduction instead of a transcription.  The
premise gates in six test families that cite this divergence were re-measured
on this tree and **all still hold** (268 passed, 2 skipped, 0 failed over 20
files; neither skip is the census divergence).  Two audit documents now carry
premises that are refuted rather than merely unproven — the runner being AMD
EPYC 7763, and "the local default arm IS the CI kernel" — and the corrections
are recorded in `test_ci_kernel_consistency.py`'s docstring.

### Added -- `tests/unit/test_audit2609_a23_census_mechanism.py`

Five tests pinning the finding so it cannot recur silently, split the way
`CI_PREMISE_GATES` §3 requires: the geometry — the raised default snaps the
fixture, `rcond` moving nine decades and the union grid saying so — is
**unconditional**, because no BLAS kernel takes part in comparing two floats;
only the claim that the pinned fixture reads WRONG *here* is premise-gated,
measured, and skipped with its reading on an arm of the CI class.  Verified to
skip rather than fail.

---

**Files:** `validation/probe_ci_kernel_sweep/{probe_decisions,merge_arms,make_ci_arm}.py`,
`arms/*.json` (11 marked historical, 12 new live, 1 regenerated),
`decisions.json`; `tests/unit/test_ci_kernel_consistency.py`;
**new** `tests/unit/test_audit2609_a23_census_mechanism.py`; a dated docstring
note in the eight premise-gated files that cite the `3.61` / `1.000115` kernel
readings (`test_audit2609_a12_pmm1d.py`, `test_audit2609_a12_verify_pmm1d.py`,
`test_fix_pmmstack_sliver_round3.py`, `test_fix_pmmstack_sliver_round4.py`,
`test_fix_pmmstack_sliver_walls.py`, `test_fix_pmmstack_sliver_walls_round2.py`,
`test_verify_pmmstack_sliver_round2.py`, `test_verify_pmmstack_sliver_walls.py`)
— docstrings only, no assertion, bar or fixture touched.

**Nothing under `lumenairy/` was changed.**

### Changed -- CI kernel census: re-recorded with `threadpoolctl` installed; the probe's threadpoolctl branch now records the per-library table

`threadpoolctl` -- a declared core dependency that was absent from this workstation for the
whole campaign -- was installed at campaign close, as WP-A23 section 7 asked, and the six Windows
live arms were re-recorded through it (`validation/probe_ci_kernel_sweep/arms/win_live_AUTO_t1.json`
and its five siblings; `validation/probe_ci_kernel_sweep/decisions.json` re-merged, exit 0).  Every
one of the 61 decisions, 12 classes and 30 readings per arm is bit-identical to the ctypes-era
recording, the arm names are unchanged, and `kernel_source` now reads `threadpoolctl` instead of
`ctypes(openblas_get_corename)`.  The first re-run exposed a defect in the branch no host had been
able to execute: `_arm_id`'s threadpoolctl path recorded the kernel and thread width but left
`blas_libraries` empty, where the ctypes fallback fills one row per loaded BLAS build.
`validation/probe_ci_kernel_sweep/probe_decisions.py` now records the same table from
`threadpool_info()` (library basename -> corename, thread width; OpenMP runtimes ignored), and
`tests/unit/test_audit2609_a23_census_mechanism.py` pins the branch with a synthetic
`threadpool_info` (fail-before: the pre-fix probe returns `{}`).  The six WSL arms keep their
ctypes-era recording (that environment has no threadpoolctl) and `merge_arms.py` accepts the mixed
provenance.  `set_blas_threads` / `rcwa_blas_threads` are live on this box for the first time; the
13 census tests pass.
