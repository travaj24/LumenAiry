"""One-shot editor: fill sections 10 (runs) and 11 (not established) of the
round-2 report, and add the WSL conditioning row to 4.7."""
import io

RUNS = r"""## 10. Runs

All with `-p no:randomly`.  "pinned" is `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
on the command line, which is the setting that exposed the round-1 defect.

| battery | Windows py3.14 | WSL py3.12 |
|---|---|---|
| the GATE: `test_fix_rcwa_even_sector_wsl.py` + `test_verify_rcwa_even_sector.py` + `test_m1_conditioning_guard.py` + `test_v5_14_2_backlog_batch.py` + `test_fix_branch_cut_round2.py`, pinned | **86 passed**, 107.67 s | **86 passed**, 114.32 s |
| the same, threads UNPINNED | **86 passed**, 2991.10 s | **86 passed**, 3551.84 s |
| `test_pmm*.py` + `test_fix_pmm*.py` + `test_verify_pmm*.py` + `test_rcwa*.py` + `test_niche*rcwa*.py`, pinned | **728 passed, 1 skipped**, 2464.70 s | not run (see section 11) |
| the JAX-guarded PMM / RCWA / Berreman files, pinned | **66 passed**, 180.63 s | not run (see section 11) |
| census / walker sweep (`walker\|census\|dispatcher_pin\|public_api\|doc_consistency`, `-x`) | **1284 passed, 12 skipped**, 254.22 s | not run (see section 11) |
| `ruff check lumenairy/ tests/ validation/probe_fix_branch_cut_round2/` (WSL) | -- | **All checks passed!** |

The one Windows skip in the PMM/RCWA battery is the `threadpoolctl`-dependent
`test_niche_audit_m4_m5_m6_rcwa.py::test_set_blas_threads_numerically_equivalent`.
The 12 skips in the census sweep are the walker's own
"this CHANGELOG block has no such claim" arms.

**The M1 file is the load-bearing row.**  It was the ONE file the round-1 change
turned red, and the verification's ship condition 5 asked for it to be added to
the battery that gates this work.  It is, and it now reads **28 passed, 0
skipped** on both builds -- against `1 failed, 21 passed, 5 skipped` on the
merged tree before this round.

The `DLASCL` lines appear in the WSL logs, exactly twice, from
`test_m1_conditioning_guard.py::test_guarded_lstsq_stands_aside_on_a_non_finite_system`
-- the deliberate NaN matrix the verification attributed them to (its section
10).  They are Fortran unit-6 output from `zgelsd`, not this solve's, and not a
defect.

New DECISION tests, `tests/unit/test_fix_branch_cut_round2.py`, 16 tests,
104.02 s for the whole file pinned; the slowest single test is 42.29 s
(`test_the_pure_staggered_engine_never_reaches_this_function`, whose cost is
the staggered engine's own first solve), well inside the 60 s budget.  Node ids
spliced into `.test_durations` (5 removed for the two renamed M1 tests, 21
added, sorted, json-valid).

---

## 11. What is NOT established

* **The widest admissible `_CUT_BAND_REL`.**  Section 5 chooses the SHAPE by
  measurement and re-derives the shipped constant's margins, but no ladder was
  run over the constant itself.  It is bounded from below by the cutoff
  population (a mode reaches 6.7172e-09 in the verification's deeper ladder, so
  anything at or under ~1e-9 would start missing modes) and from above by the
  signal side (2.1844e-05 at a cutoff mount, 1.2612e-02 in ordinary ones).
* **A deeper cutoff than `min|lam^2| = 6.5341e-10`.**  My bounded minimisation
  reached that; the verification's trisection reached 4.495e-15.  I therefore
  reproduce the SHAPE of D3/D4 rather than their worst values, and quote theirs
  as the binding ones.
* **Whether `_forward_branch_flip`'s scale should be per-mode.**  Section 5.5
  shows its two-sided gap is not cleanly measurable with either classification I
  tried, because its natural discriminator is `|Im q| / |q|` rather than
  `|Im q| / max|q|`.  What is established is that the worst `|Im q| / |q|` it
  actually acts on over 42,310 modes is 3.5588e-09 -- unambiguously rounding --
  on both builds.  Round 2 does not change that function.
* **The ARMED `T22` refusal population.**  My M1 sweep reached it on 0 of 110
  guarded inverses, so the verification's reading (minimum equilibrated `rcond`
  2.792e-02 on both arms, eight decades above its own `1e-10` bar) stands
  un-re-measured here.
* **The WSL side of the wide batteries.**  The PMM/RCWA battery, the JAX files
  and the census sweep were run pinned on Windows only; WSL carried the gate
  battery (pinned and unpinned) and every probe.  The box is shared and the
  Windows PMM/RCWA battery alone took 41 CPU-minutes.
* **CuPy / GPU.**  Every measurement here is NumPy or JAX on CPU.  The shared
  body is `xp`-generic and `array_namespace` routes CuPy the same way it routes
  NumPy, but no GPU arm was run.
* **`stabilize=True`.**  Not re-characterised; its retry schedule exists for
  this failure class and whether it can now be narrowed is still open, as both
  round 1 and the verification say.
* **Whether the hybrid PMM has OTHER coincidence partners.**  Sections 4.2-4.8
  establish the REGION, the SUPERSTRATE and the UNIFORM LAYER.  A slanted
  patterned layer is exempt (it solves through the 4N generator, and
  `stack_slant_coinc` is bit-identical between the arms), but "no other partner
  exists" is not a claim any of this supports.

---

## 12. Reproduction

```
git worktree add -b fix/branch-cut-round2 C:/tmp/lum_bc2 2898767

# every probe, either build -- PYTHONPATH=. is REQUIRED (each probe calls
# b_fixtures.require_local_tree(), which REFUSES to produce a number if
# lumenairy was imported from anywhere but the working directory)
cd C:/tmp/lum_bc2 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=. python -u \
  validation/probe_fix_branch_cut_round2/b7_spacer.py out.json
```

Every JSON carries a `_stamp` block naming the `lumenairy.__file__` it measured,
the arm it detected from the LIVE source of the PMM copies, the interpreter,
numpy, the platform and the three thread environment variables.

| probe | what it measures |
|---|---|
| `b0_smoke.py` | which surfaces reach which selector at all |
| `b1_pmm_wrong.py` | 30 PMM surfaces, both arms, with the eig census |
| `b2_pmm_interface.py` | `cond(a+b)` at every PMM interface mode-match |
| `b3_x1.py` | the X-1 `THIN` ladder, both arms, census armed |
| `b4_census.py` | the 49-fixture bit-identity census + the refactor's own |
| `b5_jax.py` | the three JAX twins: forward, NumPy parity, gradients |
| `b6_band_scale.py` | ARRAY-MAX vs PER-MODE over the ordinary populations |
| `b6b_cutoff.py` | the same, on 72 mounts driven onto a LAYER CUTOFF |
| `b6c_staggered_flip.py` | `_forward_branch_flip`'s own band, on its own quantity |
| `b7_spacer.py` | the uniform-spacer family: class, detune, modulation, threads |
| `b8_reference.py` | the repaired answer against an independent RCWA solve |
| `b9_m1_instrument.py` | the M1 equilibration instrument's motivating population |
| `b10_manufactured_energy.py` | the mount that returns `sum R + T` up to 110 |
"""

NOT = "## 11. What is NOT established\n\n*(filled)*\n"

p = "docs/audits/FIX_BRANCH_CUT_ROUND2_2026_09_11.md"
s = io.open(p, encoding="utf-8").read()
i = s.index("## 10. Runs")
io.open(p, "w", encoding="utf-8", newline="").write(s[:i] + RUNS)
print("sections 10-12 written")
