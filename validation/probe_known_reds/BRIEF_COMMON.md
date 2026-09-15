# Common brief -- Wave 5 item D, CI-matrix remediation on `fix/known-reds-and-stacklevels`

You are one of several engineers working in the SAME worktree on disjoint file
sets.  Read this whole file before touching anything.

## Where you are

* Worktree: `C:/tmp/lum_reds`, branch `fix/known-reds-and-stacklevels`, base
  `96cb2096` (the 5.47.0 release commit `4bf26c5e` plus the Wave-5 plan doc).
* **Every shell command MUST start with `cd /c/tmp/lum_reds && `.**  Agent
  threads reset their cwd between bash calls.
* A `pip -e` install points at `D:\...\Lumenairy`, a DIFFERENT branch.  Pin the
  tree on every python invocation with `PYTHONPATH=/c/tmp/lum_reds` and, in any
  probe you write, print/assert `lumenairy.__file__` contains `lum_reds`.
* Windows interpreter: `python` (3.14.6, numpy 2.4.4, OpenBLAS 0.3.31 built
  Haswell).  WSL: `wsl -e bash -lc 'cd /mnt/c/tmp/lum_reds && ~/lumvenv/bin/python ...'`.

## Invocation rules (non-negotiable)

* **Always** put `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
  on the COMMAND LINE of every python/pytest call.
* **Always** run pytest with `--capture=sys` (handoff 4.1b: on this box tests
  that spawn worker processes hang under pytest's default fd capture).
* **Always** grep the pytest tail for `passed|failed|error|no tests ran`.  A run
  that collected nothing is not a pass.
* Do **not** touch `lumenairy/elements/_lens_traced.py`'s pool code -- another
  agent owns it.
* **Never run a git write command** (`add`, `commit`, `stash`, `checkout`,
  `restore`, `reset`, `worktree`).  The orchestrator commits.  `git log`,
  `git diff`, `git show`, `git archive` are fine.
* **Never kill processes.**  `python -u q2b_qwp.py` on this box is the
  maintainer's own multi-day run and must not be disturbed.
* Do not create a second worktree; if you need an isolated tree use
  `git archive <commit> lumenairy | tar -x -C <scratchdir>` and run with
  `PYTHONPATH=<scratchdir>`.

## Kernel ladder -- mandatory for any build-dependence claim

`OPENBLAS_CORETYPE` in {`HASWELL`, `NEHALEM`, `KATMAI`, `SANDYBRIDGE`} x
threads {1, 4}.  (`ZEN` aliases Haswell on this box; `SKYLAKEX` crashes -- do
not use it.)  Confirm the kernel actually changed with `threadpoolctl`:

```
cd /c/tmp/lum_reds && OPENBLAS_CORETYPE=NEHALEM OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/c/tmp/lum_reds \
  python -c "import threadpoolctl,json;print(json.dumps(threadpoolctl.threadpool_info()))"
```

A reusable driver is `validation/probe_known_reds/ladder_t31.sh <pytest-args> <outdir>`.
Report the ladder as a table: arm x outcome x the measured number.

## The house rules you are judged against

`docs/TESTING_STANDARDS.md` and the maintainer's standing rule
**"flakiness is bad math"**:

1. A test that is red on one arm and green on another is a **defect to be
   root-caused by measurement**.  Never mask it, never rerun-to-green, never
   loosen a bar without a **derived two-sided** replacement (state the measured
   ladder, the margin, and why both sides are where they are).
2. **A decision that depends on the kernel, the thread count, the wheel or the
   interpreter version is a library defect**, not a test problem -- fix it at
   the library layer if that is where it lives.
3. Tests assert **invariants unconditionally**.  A claim that a *pathology*
   reproduces is **premise-gated**: measure the premise on the running arm, and
   if it does not hold `pytest.skip` **with the measured reading in the message**
   -- never silently pass, never assert the pathology as if it were universal.
   Keep a hard failure for "this test measured nothing at all".
4. CI runners are a per-job mix of AMD EPYC 9V74 (Zen 4) and EPYC 7763 with
   older wheels, Python 3.10-3.14, Linux.  Anything you pin must hold across
   that set or be premise-gated with the reason.
5. Oracles must be independent of the code under test.  No wall-clock
   assertions (count operations instead).
6. Comments say what the code does NOW and why.  Version narrative goes to
   `docs/history/<dotted.module>.md`; if you change a module's AST/token
   fingerprint you must re-record it:
   `python scripts/record_history_fingerprints.py <module> --reason "..."`.

## The evidence you are working from

The 5.47.0 commit is RED on CI run 34914295323.  Every completed job's log is
at `C:/tmp/ci_5470/<job>/unit_test_output.txt` (17 jobs: py3.10 shards 1-5,
py3.11 shards 1-5, py3.12 shards 1-2, py3.13 shards 1-3, py3.14 shards 4-5).
**Read the logs for your files first** -- they carry the CI-side numbers you
must reproduce or bound.  Note py3.11 aborted at `--maxfail=10`, so the 3.11
failure set is a LOWER bound.

## What you deliver back to the orchestrator

1. The edits, in YOUR FILES ONLY (listed in your own brief).  Do not edit any
   other file; if you believe another file must change, say so in your report.
2. Any probe you wrote, saved under `validation/probe_known_reds/`, plus its
   JSON output per arm (`<probe>_<ARM>.json`).
3. A final report (text, not a file) with, per item:
   * the reproduction (exact command + tail),
   * the root cause **with numbers**,
   * the fix layer (library / test / workflow) and why that layer,
   * evidence per arm (the ladder table where build dependence is claimed),
   * the test tail proving the fix, run BOTH the failing arm and a green arm,
   * anything you could NOT establish.
4. Do **not** write a summary `.md` file; the orchestrator writes the report.
