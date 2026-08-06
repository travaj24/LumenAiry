# PMM fifth name -- `test_sweep_matches_perwavelength` thread-count dependence

Date: 2026-08-05
Branch: `feat/pmm-per-layer-roadmap`
Referral source: `docs/audits/PMM_FOURNAME_ADJUDICATION_2026_08_05.md` (found
during the four-name adjudication; pre-existing, not introduced by that
campaign).

Subject: `tests/unit/test_v5_13_0_pmm_tapered.py::test_sweep_matches_perwavelength`
FAILED at the default BLAS thread count and PASSED at
`OPENBLAS_NUM_THREADS=1`.

---

## 1. Verdict

**Not a library defect in the sweep, and not a modal-classification flip.**
`PMMStack.solve_vs_wavelength` caps the process-global BLAS pool to
`blas_per_worker` (default **1**) around its whole dispatch, while a bare
`PMMStack.solve()` runs at the environment's pool (24 threads here). The test
compared the **capped** sweep against the **uncapped** per-wavelength solve at
`atol=1e-10`. The two therefore ran different LAPACK reduction orders, and the
S-matrix cascade amplified that last-bit difference to ~2e-9 -- 18x over the
bar.

The comparison was apples-to-oranges. **In either regime, taken consistently,
the two paths agree to the LAST BIT (max |dR| = |dT| = exactly 0.0).**

Decision branch taken (per the referral's decision tree): **(b) -- legitimate
FP non-associativity through the cascade; make the two paths numerically
identical (same threading regime) rather than loosen a tolerance.** No
tolerance is used anywhere in the repaired test.

---

## 2. Root cause

### 2.1 The 2x2 controlled experiment

The failure is governed by whether the library's BLAS cap can actually be
*applied*, which requires the optional `threadpoolctl` package. Replaying the
pre-fix test body verbatim on both mounts at both thread counts:

| mount | `threadpoolctl` | BLAS pool | cap applied? | old test | worst \|diff\| |
|---|---|---|---|---|---|
| Windows | 3.6.0 present | 24 | **YES** | **FAIL** | **1.785e-09** |
| Windows | 3.6.0 present | 1 | yes (no-op) | PASS | 0.000e+00 |
| WSL | **ABSENT** | 24 (16 effective, measured) | no (inert) | PASS | 0.000e+00 |
| WSL | **ABSENT** | 1 | no (inert) | PASS | 0.000e+00 |

The failure appears in exactly the one cell where the cap creates a thread-count
*mismatch* between the sweep and the reference solve. Everywhere the two paths
share a regime -- whether at 1 thread or at 24 -- the disagreement is **exactly
zero**, not merely small. That zero is the proof that the sweep runs the same
arithmetic as `solve()`; the only variable is the BLAS reduction order.

> **Correction to the referral's premise.** The referral stated the failure
> reproduces on both mounts. It does not: this WSL venv
> (`/home/travaj/lumen_venv`) has no `threadpoolctl`, so the cap is inert there
> and both paths run at the ambient pool. WSL's BLAS *is* genuinely
> multithreaded (measured 16.0 effective threads on a 2000x2000 zgemm, 24
> cores), so the mount is not single-threaded -- it is *cap-less*. The
> discriminator is the cap, not the thread count.

### 2.2 Where the difference enters, and the amplification chain

Instrumented on the worst cell (vertical stack, wl = 0.600 um, `n_glob = 36`,
72x72 cascade), comparing the ambient (24-thread) and capped (1-thread) runs
stage by stage:

| stage | abs diff | rel diff |
|---|---|---|
| half-space modes `Wsup, Vsup, Wsub, Vsub` | 0.0 | 0.0 |
| layer eigenVALUES `lam0`, `lam1` | 0.0 | 0.0 |
| layer eigenVECTORS `W0` | 1.13e-16 | **1.17e-16** |
| layer eigenVECTORS `W1` | 3.33e-16 | 3.46e-16 |
| ONE interface S-matrix | 5.8e-15 | ~7e-15 |
| **cascaded** S-matrix (3 interfaces + 2 propagations) | 1.28e-08 | **1.09e-08** |
| resulting efficiency | 1.785e-09 | ~1e-08 |

Reading: LAPACK returns **bit-identical eigenvalues** at both thread counts;
only the **eigenvectors** of the patterned layers move, by ~1 ulp. The
Redheffer star cascade amplifies that by ~**1e8** (each star inverts
`I - S22_a S11_b`, and the chain compounds), landing at ~2e-9 absolute in
efficiencies that are O(0.1). The individual interface solve is *well*
conditioned (`cond([Wsup -W0; Vsup V0]) = 2.56e3`, `cond(Wsup) = 2.8`,
`cond(W0) = 6.3`) -- the amplification is the cascade, not any single solve.

### 2.3 Ruled out: modal classification flip

The four-name adjudication found that thread count can change modal
forward/backward classification on near-cut modes. **It does not here.** The
shipped mode-cut census (`_record_mode_cut` / `_MODE_CUT_CENSUS`) was armed on
this exact geometry at both regimes, all 5 wavelengths, both stacks:

* `classification_identical = True` at every point -- identical
  `(site, n_prop, n_grow, n_risk)` rows in both regimes;
* `n_grow = 0` everywhere (channel A silent: no growing mode in the forward
  set);
* cut margins range 4.7e2 to 7.0e8, i.e. **47x to 7e7x above** the
  `_MODE_CUT_MARGIN_WARN = 10` bar. The tightest, 472.7, is at the vertical
  stack's wl = 0.600 um -- the same point that carries the largest round-off
  amplification, which is consistent with a conditioning story and inconsistent
  with a coin-flip classification.

### 2.4 Ruled out: X-1, and a sweep dispatch bug

* **Not X-1.** X-1 is manufactured energy through a high-condition 2-D
  interface solve. Here the interface conditioning is a modest 2.6e3, energy is
  conserved in both regimes, and the discrepancy is a smooth ~1e8 amplification
  of a 1-ulp eigenvector perturbation.
* **Not the M4 disease.** The M4 per-worker-cap race is already fixed at this
  site (one cap around the dispatch, `stack.py` line ~2996). Measured: sweep at
  default `max_workers` vs `max_workers=1` is **byte-identical (0.0)** for both
  geometries. That contract is independently pinned by
  `tests/unit/test_v5_21_pmm_threaded_sweep.py`, so the repaired test does not
  duplicate it.

---

## 3. The fix

### 3.1 Library -- `lumenairy/elements/pmm/stack.py`

The investigation surfaced a genuine **shipped-contract defect**: the
`solve_vs_wavelength` docstring claimed, unconditionally,

> "Bit-identical to per-wavelength `solve()` on the propagating orders."

That is false at the shipped default on any multi-core box with
`threadpoolctl` installed -- precisely the configuration the test failed in.
The RCWA twin's docstring already gets this right (it claims byte-identity to a
*serial sweep* and states the same-thread-count precondition explicitly); the
PMM twin did not.

Corrected to state the precondition, quantify the measured departure, record
that classification is unaffected, and name the escape hatch. The
`blas_per_worker` parameter doc now documents `None` as the setting under which
the sweep is bit-identical to a bare `solve()`. **Docstring only -- no numeric
behaviour changed anywhere in the library.**

The default was deliberately left at `blas_per_worker=1`: it is a sound
performance default (it prevents `max_workers * pool` oversubscription), and
changing it would alter every existing user's last bits for no accuracy gain.

### 3.2 Test -- `tests/unit/test_v5_13_0_pmm_tapered.py`

`test_sweep_matches_perwavelength` now asserts the equivalence **in both BLAS
regimes**, with **no tolerance at all**:

* **Regime A (shipped default)** -- default sweep vs per-wavelength `solve()`
  run under the *same* cap; `np` equality, gap must be `== 0.0`.
* **Regime B (ambient)** -- `solve_vs_wavelength(max_workers=1,
  blas_per_worker=None)` vs a bare `solve()`; gap must be `== 0.0`. Serial
  because an uncapped threaded sweep would oversubscribe the pool.
* **Comparative leg** -- the default (capped) sweep's residual disagreement
  with an *uncapped* `solve()` must be no larger than the disagreement
  `solve()` has with **itself** across the two regimes. Self-calibrating: it
  attributes the residual entirely to BLAS reduction order and fails the moment
  the sweep contributes error of its own. Zero magic numbers; degenerates to
  `0.0 <= 0.0` on a cap-less mount.

The helper `_reference_solves` deliberately uses the **library's own**
`_blas_threads_quiet` / `_blas_limit` pair rather than `threadpoolctl`
directly, so the reference solves match the sweep's regime **by construction on
every mount**: where the cap applies, both sides get it; where `threadpoolctl`
is absent, `_blas_limit()` degrades to a null context for both sides alike.
This is what makes the repair portable rather than Windows-specific.

The module's `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` preamble was
left in place but re-commented: OpenBLAS reads those at library-load time, so
the setdefault is **inert** whenever anything imported numpy first (the normal
case under pytest). It had been quietly creating the impression that this file
runs single-threaded. Nothing in the file may rely on it.

### 3.3 Envelope-rule compliance

No absolute numeric bar calibrated on one configuration survives in the test.
Every assertion is structural (exact equality) or comparative (measured against
the same run's own cross-regime spread). The old `atol=1e-10` is gone.

### 3.4 Negative control -- the repair is strictly MORE sensitive

The concern with replacing `atol=1e-10` by exact equality is the reverse
failure: a test that can no longer catch anything. Verified by injecting a
genuine algorithmic error into the sweep (layer depths scaled by `1 + 1e-13`,
`solve()` left correct):

| injected depth error | resulting \|dR\| / \|dT\| | old test (atol 1e-10) | new test |
|---|---|---|---|
| 1e-13 relative, tapered | 2.60e-13 / 2.38e-13 | would MISS | **CAUGHT** |
| 1e-13 relative, vertical | 6.67e-14 / 1.43e-13 | would MISS | **CAUGHT** |
| 1e-15 relative, tapered | 3.55e-15 / 2.28e-15 | would MISS | would catch |

The repaired test detects algorithmic divergence ~5 orders of magnitude smaller
than the old bar, while being immune to the 2e-9 threading artifact that made
the old one fail. Sensitivity went up, not down.

---

## 4. Evidence matrix

Command: `python -m pytest <file> -q -p no:randomly`.
WSL leg prefix: `wsl -e bash -c "source /home/travaj/lumen_venv/bin/activate &&
cd /mnt/d/.../Lumenairy && ..."`, run serially.

### 4.1 Target file -- `tests/unit/test_v5_13_0_pmm_tapered.py`

| mount | BLAS threads | before fix | after fix |
|---|---|---|---|
| Windows native (OpenBLAS 0.3.31, Haswell) | default (24) | **1 failed, 5 passed** | **6 passed** (166.1 s) |
| Windows native | `OPENBLAS_NUM_THREADS=1` | 6 passed | **6 passed** (43.6 s) |
| WSL (OpenBLAS 0.3.31, SkylakeX, no threadpoolctl) | default (24) | 6 passed (239 s) | **6 passed** (249.6 s) |
| WSL | `OPENBLAS_NUM_THREADS=1` | 6 passed | **6 passed** (37.3 s) |

(Times are from the final re-run of all four cells against the shipped code;
wall times vary with machine contention -- the Windows default cell measured
51.3 s on an idle box and 166.1 s alongside the WSL leg.)

4/4 cells green. Fail-before-switch is documented in S2.1: the pre-fix body
fails only in the Windows/default cell, at 1.785e-09 against a 1e-10 bar.

### 4.2 Regression -- green before, green after

| file | Windows (default threads) | WSL (default threads) |
|---|---|---|
| `test_pmm_m2_window_contract.py` + `test_pmm_m3_efficiency.py` | **60 passed**, 24 warnings (73.9 s) | **60 passed**, 24 warnings (115.8 s) |

The 24 warnings are the pre-existing expected `_pmm_union_grid` near-coincident
wall-snap notices, unchanged.

### 4.3 Lint

`ruff check lumenairy/elements/pmm/stack.py
tests/unit/test_v5_13_0_pmm_tapered.py` -- **All checks passed.** Both files
verified ASCII-clean in the edited regions (the 12 non-ASCII bytes in
`stack.py` are pre-existing `phi` glyphs at lines 1371-1378, untouched).

---

## 5. Files changed

* `lumenairy/elements/pmm/stack.py` -- `solve_vs_wavelength` docstring: the
  bit-identity claim and its precondition, plus the `blas_per_worker` parameter
  doc. Docstring only.
* `tests/unit/test_v5_13_0_pmm_tapered.py` --
  `test_sweep_matches_perwavelength` rewritten to assert both regimes exactly;
  new module-level helpers `_sweep_wls`, `_sweep_stack`, `_reference_solves`,
  `_max_gap`; corrected comment on the inert thread-env preamble.
* `docs/audits/PMM_FIFTHNAME_TAPERED_SWEEP_2026_08_05.md` -- this document.

## 6. Open / follow-on

* **`threadpoolctl` missing from the WSL venv.** It is declared in the `[dev]`
  extra, but `/home/travaj/lumen_venv` does not have it, so every BLAS cap in
  the library is inert on that mount -- which is why this bug, and the M4 race
  before it, could never reproduce there. WSL is therefore **not** a valid
  mount for validating cap-related behaviour today. Installing it would make
  the WSL leg genuinely independent. Not done here (no environment mutation in
  scope).
* The ~1e8 round-off amplification through the Redheffer cascade is inherent to
  S-matrix cascading at this size and is **not** flagged as a defect: energy is
  conserved, classification is stable, and the absolute error (~2e-9 on
  O(0.1) efficiencies) is far below any physical scale. Recorded here as the
  quantitative context for any future ULP-level pin on cascaded PMM results --
  such pins must fix the BLAS regime.
