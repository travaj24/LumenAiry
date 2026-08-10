# VERIFY -- independent adversarial verification of `perf/traced-hotpath` @ `7890b7d`

**2026-08-10.  REPORT ONLY: no library, test, validation or doc file was
edited; no git command that writes was run.  Every probe lives in the session
scratchpad
(`C:\Users\Tesla\AppData\Local\Temp\claude\C--Users-Tesla\372a2d1f-...\scratchpad`).
The verifier had no part in the work under test.**

Claims list: `FIX_PERF_POLY_LOCALS_2026_08_09.md`,
`FIX_PERF_CACHES_BLUESTEIN_2026_08_09.md`, `FIX_PERF_PARALLEL_2026_08_10.md`,
`ADJUDICATION_NFC_8192_2026_08_10.md`.

Box: Windows 11 Pro 10.0.26200, Ryzen 9 5950X (24 logical), 137.4 GB physical,
python 3.14.6 / numpy 2.4.4 / scipy 1.17.1 / pyfftw 0.15.1 / numexpr 2.14.1.
Linux arm: WSL `~/lumen_venv`, python 3.12.3 / numpy 2.4.6.

---

## 0. HEADLINE

> **The four headline correctness claims survive my own probes.**  `_poly` is
> bit-identical on 53 fixtures the fix's tests do not use; the group-exit and
> readout fields are byte-identical across a pre/post package snapshot on a
> DECENTRED + TILTED configuration the fix never ran; the separable readout is
> banner-identical at `rel L2 = 3.0e-16` on that same configuration with a
> character-identical replica guard; and k=1 vs k=3 reproduce
> BIT-IDENTICALLY in a fresh k-ladder, down to the same tile sha256 prefixes
> the doc printed.  The loud refusal exits 2 with the numbers named, and the
> auto-budget banner is identical to HEAD~1's on all 36 metric lines.
>
> **Four defects, one of them a silently wrong number in a public API.**
> (D1) `memory.py`'s new plan-buffer branch builds the wrong dtype code --
> a no-op on Windows, a 43 % UNDER-estimate of `estimate_asm_memory` on Linux
> at N = 8192 complex128.  (D2) `del E_analytic` does not free: numexpr's
> `_numexpr_last` holds it to the end of the chain, so item #2's `-34.9 GB` is
> really `-30.6 GB`.  (D3) an over-cap `_H_FFT_CACHE` entry is `np.copy`'d
> before it is rejected -- the byte cap converts a 4.86 GB retention into a
> 4.86 GB transient rather than avoiding it.  (D4) the re-derived clamp model
> reads 1.54x my fresh measurement at `n_fine = 4096`, outside the 1.5x bound
> the branch's own new test declares; the test stays green only because it
> pins the doc's higher number.
>
> **`FIX_PERF_PARALLEL` sec 3.2 / 3.3 / 3.4 and the sec 4.2 clamp transcript
> are from a `_FINE_GRID_BASE_BYTES = 2.3 GB` era; the branch ships 4.5e9.**
> Every model figure in those three sections is wrong against the code in the
> same commit, including the approved-worker ladder (shipped: 2/3/3/3/3/3, doc:
> 2/3/4/4/4/4).  The conclusions do not move; the arithmetic does.

---

## 1. VERDICT TABLE

| # | claim | my probe | verdict |
|---|---|---|---|
| 1a | `_poly` rewrite is BIT-identical | 53 fresh cases: duplicated exponents, negative coefficients, degree 7-10, empty term set, all-zero fit, exponents reachable only through a `c == 0` term, on-circle/origin/NaN/inf queries, scalars, 1-D, decentred+frozen -- `np.array_equal` on all 6 `_poly` outputs, `value`, `grad`, and `hess=False` vs `hess=True` | **CONFIRMED** 53/53, 0 failures |
| 1b | `.value()` 3.78x, `_poly` 2.72x, `grad()` 2.47x | my own timing, verbatim `c8bcbcb` reference vs shipped class, 256 x 16384 band, deg 6 / 27 terms, min of 4, threads pinned to 1 | **CONFIRMED** (better): **4.02x / 2.80x / 2.54x** |
| 1c | `hess=False` returns `None`, first three slots bit-equal | asserted in every case above | **CONFIRMED** |
| 2a | group exit + readout byte-identical | two package snapshots differing in exactly ONE file (`_lens_traced.py` from `c8bcbcb`), on a **decentred + tilted** D6-class singlet with `ray_density` + `remap` + `lattice`, `ray_subsample=2`, `n_fine=2048` -- a configuration outside the fix's 11-case matrix | **CONFIRMED**: group exit sha `248ffaf1...` and readout sha `fd1508fd...` identical both arms |
| 2b | 16.25 full-grid equivalents freed from the frame | my own `sys.settrace` census at the element's return | **CONFIRMED**: 44 arrays / 769.59 MB -> 30 / 224.33 MB = -545.26 MB = **exactly -16.25** float64 grid equivalents; what remains is `E_in`, `E_out`, `amp`, `_cW` as claimed |
| 2c | the `del` sites really free the arrays | weakref taken at each of the 12 `del` sites, checked at element return | **DEFECT (D2)**: 11 of 12 freed, **`E_analytic` STILL ALIVE** |
| 2d | honest caveat -- the whole-process peak lives in the readout | RSS sampled at 20 Hz with stage markers | **CONFIRMED**: fine leg max 1.2812 GB, readout max **1.6972 GB**, peak stage = `readout` |
| 3a | clamp model is an upper bound within 1.5x | fresh `kladder_121.py` run at the doc's own sec 3.1 configuration, `NFC=4096`, 2 orders, `RAMB=inf` | **DEFECT (D4)**: measured **6.648 GB**, model 10.238 GB, ratio **1.540** |
| 3b | `_H_FFT_CACHE` byte caps bind under eviction pressure, values unchanged, cache still hits | 20 distinct alphas under 3-entry / 1-entry / half-entry total caps; per-entry cap below the entry | **CONFIRMED**: 3 entries / 15.925 MB vs 84.935 MB under the count cap alone; byte-identical in every cap regime; 0 hits on 3 distinct orders, 1 hit on a repeat |
| 3c | an over-cap entry "is NOT stored at all" | `tracemalloc` around `_h_fft_cache_store` with the per-entry cap at 1 B | **DEFECT (D3)**: retained 0.000 MB but traced peak rose the **full 67.109 MB** -- the copy happens before the cap check |
| 3d | pyFFTW per-entry copy-decision closes the aliasing hazard | cap set below a shape's workspace with the GLOBAL ping-pong still ON; two calls at the same key, all four dispatchers | **CONFIRMED**: `_fft2` / `_ifft2` / `_fft2_nd` / `_ifft2_nd` all survive; results do not share memory; first result still correct to 1e-12; cap round-trips through `snapshot`/`restore_fft_state` |
| 3e | `estimate_asm_memory` now reads the same predicate | direct probe, Windows + WSL | **DEFECT (D1)**: wrong dtype code; no-op on Windows, under-estimate on Linux |
| 4a | #7 banner digits + energy, flag on vs off | my own decentred + tilted readout, 3 geometries (V3 axis readout of a decentred congruence; chief-ray readout; on-axis) | **CONFIRMED**: banner identical to every printed digit; worst `rel L2` **2.96e-16**, worst \|power ratio - 1\| **2.2e-16** vs a 4e-5 bar |
| 4b | V3 replica-guard message is character-identical under both routes | period read from the run (2177.1240 um), windows straddling it, `on_replica` = warn and error | **CONFIRMED**: guard text and `RuntimeError` text character-identical under both routes on all three geometries |
| 5a | k-ladder outputs are bit-identical across k | my own `kladder_121.py` arms, `NFC=8192 RAMB=48`, orders `(0,0)`,`(+1,0)`, k=1 vs k=3 | **CONFIRMED**: field sha256 identical; 2 orders x 14 fields (13 scalars + tile sha) at `rel=0 abs=0`, 0 mismatches; `np.array_equal` True on both 1024^2 tiles -- and the tile shas `d7fc7c2f3b8619ca` / `0b7ef51f19266129` are **the same two the doc printed** |
| 5b | the clamp's chosen k vs its own model arithmetic | `_fine_grid_peak_bytes` + `_multi_resolve_workers` arithmetic at the doc's stated 105.1 GB free | **DOC MISMATCH**: shipped model 26.344 GB -> **2/3/3/3/3/3**; doc sec 3.4 says 24.14 GB -> 2/3/4/4/4/4.  Chosen k = 3 either way |
| 6a | budget-binding run exits nonzero with numbers named | `RAMB=30 focus_scan_121.py` | **CONFIRMED**: **exit 2**, names 8192 -> 4096, "needs 42.9 GB", "the leg will see 30.00 GB", "modelled at 27.5 GB", 4 remedies |
| 6b | auto-budget run is banner-identical to HEAD~1's | `git show 0097e5a:.../focus_scan_121.py` vs the tip, both auto | **CONFIRMED**: all 36 metric lines byte-identical (`AT-PLANE`, 33 `dz=` rows, both `BEST-FOCUS`); the ONLY differing line is `chain done 190s` vs `209s` |
| 6c | the silent degradation reproduces (fail-before) | HEAD~1 script + `set_max_ram(30e9)`, warnings tee'd | **CONFIRMED, stronger than claimed**: identical scored digits (3.350 / 90.3 / 99.7 / 99.8), 96 s vs 190 s, `pk` 5.529e+03 -> 5.505e+03, and **0 warnings reached the tee** |
| 7a | fix docs' suite counts reproduce (spot 3) | see sec 4 | **CONFIRMED** 3/3 |
| 7b | `ruff` | `ruff check` on all 6 library files + 4 test files | **CONFIRMED** All checks passed (`validation/` is excluded by `pyproject.toml`) |
| 7c | cp1252 / ASCII | every changed file decoded as cp1252; added lines scanned for non-ASCII | **CONFIRMED**: 0 non-ASCII bytes in any added line |
| 7d | new absolute bars respect the envelope rule | see D5 | **DEFECT (D5)**: `_FINE_GRID_BASE_BYTES`'s own ENVELOPE note is violated by its universal use in `_multi_resolve_workers` |
| 7e | hindsight sibling sweep | see sec 5 | **TWO CLASSES FOUND, both with more members** |
| ADJ | adjudication's headline numbers | recomputed from `_adj_nfc_8192_rows.csv` independently | **CONFIRMED**: 16/32 breach, worst at `(+2,-1)`, max \|dEE3\| 0.0079 pt at `(+0,-2)`, dFWHM 0.0000 on all 32, pAN_A 1.384e-3..2.039e-3, pAN_B == 0 on all 32, shadow 8192 on all 64 rows, all 64 exits 0, window classes 18/14 |

---

## 2. DEFECTS, RANKED

### D1 -- `memory.py` builds the WRONG dtype code; the public estimator UNDER-predicts on Linux  (P1)

`lumenairy/memory.py`, `estimate_asm_memory`:

```python
cb = _as_complex_itemsize(complex_dtype)          # 16 for complex128
...
n_bufs = int(_plan_entry_n_bufs((N, N), np.dtype(f'c{2 * cb}')))
```

`cb` is ALREADY the complex itemsize (`np.dtype(...).itemsize`; the line above
uses it as `_ASM_COMPLEX_ARRAYS * cb * npix`).  `f'c{2 * cb}'` therefore asks
for a dtype of TWICE the element size.  Two different wrong behaviours:

* **Windows** -- `np.dtype('c32')` raises `TypeError: data type 'c32' not
  understood` (no `complex256` on MSVC).  The `except (ImportError, TypeError,
  ValueError)` swallows it and `n_bufs = 2`.  **The whole branch is a silent
  no-op for `complex128`, the default dtype it was written for.**
* **Linux / WSL (the repo's own CI proxy)** -- `np.dtype('c32')` IS
  `complex256`, so the workspace is priced at 32 B/element and the estimator
  reports ONE buffer where the runtime builds TWO.  MEASURED on
  `~/lumen_venv` (numpy 2.4.6), `plan_cache_keys=8`:

| dtype | N | n_bufs used | n_bufs true | `estimate_asm_memory` | correct | shortfall |
|---|---|---|---|---|---|---|
| complex128 | **8192** | 1 | **2** | **11.172 GB** | **19.762 GB** | **-8.59 GB (-43 %)** |
| complex128 | 11000 | 1 | 2 | 20.096 GB | 35.584 GB | -15.49 GB |
| complex64 | 12288 | 1 | 2 | 12.984 GB | 22.648 GB | -9.66 GB |

`estimate_asm_memory` is public, is the base of `estimate_sim_memory`, and its
own docstring pins it as a conservative BOUND ("est/measured = 1.06-1.09 ...
conservative (a bound) at every one").  This makes it an under-estimate at
exactly the shape the same commit's `fft_infra` comment calls "this library's
most common large shape".  The added comment says "Unchanged at every N the
A-6 pins sample" -- which is why no test sees it: the pins are at N = 512 /
1024, where both codes return 2.

Fix is one character class: `np.dtype(f'c{cb}')`, or just
`np.dtype(complex_dtype)`.

### D2 -- `del E_analytic` does not free; numexpr holds it to the end of the chain  (P2)

`FIX_PERF_POLY_LOCALS` sec 3.2 lists `E_analytic` as a full-grid complex128
freed at "the exit assembly (both branches)", **4.295 GB at
`n_fine = 16384`**, and counts it in the 16.25 equivalents / `-34.9 GB`.

Weakref taken at line 10135 immediately before the `del`, checked after
`gc.collect()`:

```
  line 10135  E_analytic       67.109 MB (2048, 2048) complex128  -> STILL ALIVE
        referrer dict keys=['out'] (keys: out, order, casting, ex_uses_vml)
```

That dict is numexpr's `necompiler._numexpr_last.l['kwargs']`, which retains
the last `out=` array per thread indefinitely.  `apply_real_lens` produces
`E_analytic` through numexpr, so the `del` drops the local and nothing else.
Reproduced across configurations, checked at three moments:

| traced configuration | alive @ element return | @ fine-leg return | @ chain end | held by `_numexpr_last` |
|---|---|---|---|---|
| ray_density + remap + lattice (**the shipped route**) | YES | YES | YES | YES |
| ray_density + remap + full | YES | YES | YES | YES |
| screen + `preserve_input_phase=True` | YES | no | no | no |
| screen + `preserve_input_phase=False` | YES | YES | YES | YES |

The other 11 `del` sites all free correctly (`_mag0`, `_bright0`, `_coords`,
`_a_rd`, `_nan_rd`, `valid`, `_absE`, `ard_map`, `_ard`, `_unit`,
`_rd_resid_map`); site 7485 is the unreached all-dark branch, as documented.

**Consequence for the claim.**  The frame census is structurally blind to this
(it sums `f_locals`, and the name is gone).  Of the measured -16.25 float64
grid equivalents, `E_analytic` is 2 of them, so the memory that actually
returns to the allocator is **-14.25 equivalents = -30.6 GB at
`n_fine = 16384`, not -34.9 GB**.  It is not a regression -- the array was held
before too -- but the doc's headline number is over-stated by one complex128
full grid, on the route design 121 ships.

### D3 -- an over-cap `_H_FFT_CACHE` entry is copied before it is rejected  (P2)

`_bluestein._h_fft_cache_store`:

```python
H = np.copy(H_FFT)                 # <-- full L^2 complex128 allocation
nb = int(H.nbytes)
if nb > int(_H_FFT_CACHE_MAX_BYTES_PER_ENTRY):
    return                         # ...and it is thrown away
```

The sibling this was modelled on, `fft_infra._h_cache_store`, checks
`_entry_bytes(H)` FIRST and never copies.  MEASURED (`tracemalloc`, per-entry
cap 1 B, a pre-allocated 67.109 MB entry): retained `+0.000 MB`, traced peak
`+67.109 MB`.

The cap binds at `L >= 11181`; the comment's own worked example is the
`window_factor = 7` geometry at `L = 17424`, i.e. a **4.858 GB transient
allocation for an entry that is immediately discarded**, at exactly the shapes
the cap exists to protect and on a run whose peak is the thing being defended.
The doc's wording -- "an over-cap entry is not stored at all (the transform
still returns it -- only the retention is skipped)" -- is true of the retention
and false of the allocation.  Currently latent because item #7's separable
route (default ON) keeps the shipped readout off the 2-D path.

Fix: read `H_FFT.nbytes` before the `np.copy`.

### D4 -- the re-derived clamp model is 1.54x my fresh measurement at n_fine = 4096, outside its own test's 1.5x bound  (P2)

`test_niche_d8_congruence_workers.py::test_the_cost_model_is_an_upper_bound_on_the_measured_peak`
pins six absolute byte values and asserts `model <= 1.5 * measured`.  I re-ran
the doc's OWN sec 3.1 configuration (`kladder_121.py`, `CW=1`, orders
`(0,0)`+`(+1,0)`, `RN=1024 RS=4 NW=1 DXO=0.2um NOUT=8192 TILE=1024 WF=4.0
LEG=auto RAMB=inf`, 1 Hz whole-tree sampler) at `NFC=4096`:

| point | measured | model (shipped constants) | ratio |
|---|---|---|---|
| 4096, 2 orders -- **doc's pinned value** | 7.123 GB | 10.238 GB | 1.437 |
| 4096, 2 orders -- **MY fresh run** | **6.648 GB** | 10.238 GB | **1.540** |
| 8192, 2 orders -- doc | 23.968 GB | 26.344 GB | 1.099 |
| 8192, 2 orders -- MY fresh run | 22.760 GB | 26.344 GB | 1.157 |
| 8192, largest CHILD -- MY k=3 arm | 25.952 GB | 26.344 GB | 1.015 |
| 16384, 2 orders -- doc | 84.589 GB | 90.768 GB | 1.073 |

The test is green only because it carries the higher of two measurements of the
same configuration.  The 4096 row has ~4 % headroom against a bar that a 7 %
run-to-run spread already crosses; the test's own message says "re-derive the
constants rather than widening this bound".  The direction is safe (the model
over-prices), and the child rows -- the ones the worker clamp actually needs --
are tight (1.015).  Reported because the bound as written is not reproducible.

### D5 -- `_FINE_GRID_BASE_BYTES` is applied outside its own stated envelope  (P2)

The constant carries: *"ENVELOPE: this is a design-121-CLASS congruence
process, not a universal python floor."*  `_multi_resolve_workers` now prices
EVERY worker with `_fine_grid_peak_bytes(int(n_fine_cap or 0), n_px=n_px)`,
and the comment says the paraxial case "builds no fine grid but still pays the
floor".  Measured effect on worker approval (8 GB reserve):

| leg | per-worker before | after | free 16 GB | free 32 GB | free 128 GB |
|---|---|---|---|---|---|
| paraxial, 128^2 input | 0.0058 GB | **4.5058 GB** (781x) | 1387 -> **1** | 4161 -> 5 | 20807 -> 26 |
| paraxial, 1024^2 input | 0.3691 GB | **4.8691 GB** (13.2x) | 21 -> **1** | 65 -> 4 | 325 -> 24 |

A paraxial multi-congruence run on a 16 GB-free box now gets ONE worker where
it got 21, on the strength of a 4.5 GB floor measured on a six-order EXACT-leg
design-121 congruence.  Nothing in the three docs measures a paraxial worker.
This is a throughput regression, not a correctness one, but it is the exact
shape the envelope note exists to prevent.

### D6 -- minor / documentation-level

* `fft_infra.py`: *"A 2 GB cap binds at 11586^2 and above"*.  MEASURED: the cap
  binds at **N >= 11181** (`11180^2 * 16 = 1.9999e9`, `11181^2 * 16 =
  2.0002e9`).  11586 is the **2 GiB** crossover -- which the same comment says
  it deliberately did NOT use ("Decimal 2e9 rather than 2 GiB deliberately").
* `carrier.py` `ram_budget` docstring still reads "``_FINE_GRID_WORK_ARRAYS`` =
  16 complex128 arrays"; the constant three hundred lines above is **20**.
* `focus_readout={'n_fine_cap': None}` -- the readout's new docstring says
  "pass `n_fine_cap=None` for no count cap", but the chain forwards it through
  `int(fr.get('n_fine_cap', 16384))` and raises a bare `TypeError`.
  **Pre-existing** (`carrier.py:7905`, the leg's own forward, is unchanged by
  this branch), so not a regression -- but the new docstring now advertises a
  value the chain-level knob rejects.

---

## 3. COUNT / ARITHMETIC MISMATCHES

**`FIX_PERF_PARALLEL_2026_08_10.md` sections 3.2, 3.3, 3.4 and the sec 4.2
clamp transcript were written against `_FINE_GRID_BASE_BYTES = 2.3e9`.  The
same commit ships `4.5e9`.**  Sections 4.4 and 5.3 ARE consistent with the
shipped value, so the document mixes two eras.

| where | doc says | shipped code gives |
|---|---|---|
| sec 3.2 table | `_FINE_GRID_BASE_BYTES` now **2.3 GB** | **4.5e9** |
| sec 3.3 model row 4096 | 8.038 GB, ratio 1.128 | **10.238 GB, 1.437** |
| sec 3.3 model row 8192 | 24.144 GB, ratio 1.007 | **26.344 GB, 1.099** |
| sec 3.3 model row 16384 | 88.568 GB, ratio 1.047 | **90.768 GB, 1.073** |
| sec 3.4 D8 clamp, NFC=8192 | MODEL 24.14 GB, `2/3/4/6/8/32 -> 2/3/4/4/4/4` | MODEL **26.34 GB**, `-> 2/3/3/3/3/3` |
| sec 3.4 / 3.2 prose | "half a budget covers a **2.3 GB** floor ... above ~4.6 GB" | 4.5 GB floor -> above **~9 GB** |
| sec 4.2 quoted clamp message | "would need ~96.6 GB (**24.1 GB** per worker)" | 4 x 26.344 = **105.4 GB** |
| `carrier.py` `_FINE_GRID_WORK_ARRAYS` comment | "At (20, **2.3 GB**) ... 1.17x / 1.02x / 1.05x" | (20, **4.5 GB**) -> 1.437 / 1.099 / 1.073 |

Independently confirmed conclusions: the clamp still CHOOSES k = 3 at
`NFC=8192` and k = 1 at `NFC=16384` under the shipped constants, and my k=3 arm
reproduced k=1 bit-identically.  Only the arithmetic in those sections is
stale.

Other numeric mismatches:

| where | doc says | reality |
|---|---|---|
| `FIX_PERF_CACHES` sec 3.2 vs `fft_infra.py` comment | "pays **~1.6 s** of a ~900 s order (**0.2 %**)" | the code comment for the same trade says "**~4.5 s** ... (**0.5 %**)" |
| `FIX_PERF_CACHES` sec 0/2.4 and `ADJUDICATION` sec 0/7.2 | 16384 needs **137.4 GB** of budget | at the branch tip (`n_work=20`) it needs **171.8 GB** (`FIX_PERF_PARALLEL` sec 2.3 has the right number) |
| `ADJUDICATION` sec 5.1 table | `dthru` column labelled RELATIVE, worst **-3.52e-04** | -3.52e-04 is the ABSOLUTE delta; the RELATIVE one is **-3.544e-04**.  Verdict-neutral (throughput ~ 0.993) |
| `FIX_PERF_POLY_LOCALS` sec 4 | the 97-file traced-niche-set row: "**STILL RUNNING** ... **This row must be filled in before the branch is proposed for merge**" | still unfilled at `7890b7d`.  **Open merge blocker by the document's own words.** |

Reproduced exactly (no mismatch): `FIX_PERF_POLY` sec 2.3's 3.78x (I read
4.02x, same direction, better); `FIX_PERF_CACHES` sec 2.2's whole fail-after
block; sec 4.3's replica-guard claim; `FIX_PERF_PARALLEL` sec 4.3's tile
hashes; the entire `ADJUDICATION` headline.

### Suite counts spot-checked (3)

| suite | doc | measured |
|---|---|---|
| `test_niche_perf_poly_locals.py` | 20 passed (POLY sec 4) | **20 passed**, 13.4 s |
| `test_niche_p2_guards.py` + `test_niche_d3_guards.py` | 53 passed (PARALLEL sec 5.3) | **53 collected** |
| `+ test_niche_d8_congruence_workers.py` (all three) | 87 passed (PARALLEL sec 5.3, WSL arm) | **87 passed**, 279.6 s |
| `test_niche_d8_congruence_workers.py` alone | 30 (CACHES sec 6, written at `6464384`) | **34 passed** -- consistent: 30 + the 4 tests PARALLEL adds |

---

## 4. WHAT I MEASURED, IN MY OWN WORDS

### 4.1 Item #1 -- the poly battery

`p1_poly.py` carries a VERBATIM `git show c8bcbcb` copy of `_poly` / `_eval` /
`value` / `grad` as `_Ref` (transcribed from git, not from the fix's test
module, so the reference is independent of the work under test).  53 cases,
`np.array_equal(..., equal_nan=True)` on all six `_poly` outputs plus `value`,
`grad`, and the `hess=False` contract.  Deliberately outside the fix's
fixtures: 12 random term lists with 1-5 DUPLICATED `(i, j)` pairs and 20 %
exactly-zero coefficients; degrees 7, 8, 9, 10; the empty term set; the
all-zero fit; `-0.0` coefficients; a fit whose only high exponent appears on a
`c == 0` term (so the power table must NOT build it); queries exactly on the
freeze circle and at `r = 0`; NaN/+-inf queries; float32 coordinates handed
straight to `_poly`.  **0 failures.**

The one contract subtlety I set out to break -- the new
`_dt = np.result_type(u, coef)` vs the old first-iteration upcast -- does not
bite: `self.coef` is always `np.asarray(..., float64)`, so under NEP 50 both
paths land on float64 even for float32 `ex`/`ey` (probed directly:
`ref=float64 new=float64 bit-identical=True`).

### 4.2 Item #2 -- the A/B and the reference hunt

Snapshot method: full package copy, then `git show c8bcbcb:` over
`elements/_lens_traced.py` only; `filecmp` over the tree reports exactly one
differing file.  Fixture: the niche-D6 stand-in (flat entrance + `K = -n^2`
conic exit, `f = 3 mm`, exit NA 0.20) with the beam DECENTRED by its own
radius and carried by a `TiltedCarrier`, `amplitude_model='ray_density'`,
`preserve_input_phase='remap'`, `remap_sampling='lattice'`,
`ray_subsample=2`, `n_fine_cap=2048`, `ram_budget=inf`.

* group exit `sha256 = 248ffaf1df24d54635795e99c12a20434a071d2d4f9654457c6bddfee96ab3f5` -- identical both arms
* chain readout `sha256 = fd1508fdd6151fe273f97a292888a3edf94b538e26d5b0bedf1dc96d420690e2` -- identical both arms
* warning set identical both arms
* frame census 44 / 769.59 MB -> 30 / 224.33 MB; the freed set is
  `_unit`, `_rd_resid_map`, `_coords`, `E_analytic`, `ard_map`, `_nan_rd`,
  `_mag0`, `_ard`, `_absE`, `_bright0`, `valid`, `_a_rd`, `X`, `Y`

The reference hunt is sec D2.  `gc.get_objects()` is useless here -- plain
numeric ndarrays are not GC-tracked -- which is why the probe uses weakrefs
taken at the `del` line itself.

### 4.3 Items #4 / #6

Cache probe drives `_bluestein_2d` through `fft_infra._fft2` (the cache is
keyed on that exact callable; passing `_scipy_or_numpy_fft2` silently bypasses
it -- worth knowing for the next person who writes this probe).  Results in the
verdict table.  The aliasing case is built exactly as the fix describes it:
`set_fft_plan_max_bytes_per_buffer(workspace // 2)` with
`get_fft_double_buffer() is True`, so the key resolves to `n_bufs = 1` while
the global ping-pong is on -- the configuration that, before the entry-count
branch, would have handed back a live workspace.  All four dispatchers copy.
I also confirmed the uncapped ping-pong is still only 2 calls deep, which is
pre-existing and unchanged.

### 4.4 Item #7

Three geometries, flag on vs off, banner recomputed from the field:

| geometry | rel L2 | max\|d\|/max\|F\| | power ratio | banner digits |
|---|---|---|---|---|
| DECENTRED + TILTED, AXIS readout (V3) | 2.960e-16 | 5.48e-16 | 1.000000000000 | identical |
| DECENTRED + TILTED, chief-ray readout | 2.958e-16 | 4.79e-16 | 1.000000000000 | identical |
| on-axis | 2.856e-16 | 5.29e-16 | 1.000000000000 | identical |

Replica guard: period read from the run (2177.1240 um), `N_out` chosen at
0.94 p (PASS) and 1.02 p (REFUSE).  Guard message text and the
`on_replica='error'` `RuntimeError` text are **character-identical** under both
routes on all three geometries, including the V3 case where the message carries
`centre_out = (-5.780816e-04, 0.000000e+00)` and the `2*|centre_out| + N*dx`
span.  On the two decentred geometries the 0.94-period window also refuses --
correctly, since `2|centre_out|` spends the rest -- and it refuses identically
under both routes.

### 4.5 Items #5 / #6 (parallel)

My k-ladder: `KEEP='0,0;1,0' NFC=8192 RAMB=48 RAMRES=4 NOUT=8192 TILE=1024
DXO=0.2e-6 RN=1024 RS=4 NW=1 WF=4.0 LEG=auto OTEG=error`.

| arm | chainB | s/order | peak tree | largest child | children |
|---|---|---|---|---|---|
| k=1 | 260.3 s | 130.1 | 22.76 GB | -- | 0 |
| k=3 | 212.3 s | 106.2 | 42.30 GB | **24.17 GiB = 25.95 GB** | 2 |

k=3 spawns only 2 children because there are only 2 orders, so the 1.23x here
is not comparable to the doc's 1.905x on six orders and I make no claim about
it.  The acceptance -- bit-identity -- is what I checked, and it holds
completely.  The largest child at 25.95 GB against the shipped model's
26.344 GB (1.015x) independently corroborates `FIX_PERF_PARALLEL` sec 4.4.

`focus_scan_121.py`, four runs:

| run | script | budget | AT-PLANE banner | pk | wall | warnings seen | exit |
|---|---|---|---|---|---|---|---|
| 1 | `0097e5a` | auto | 3.350 / 90.3 / 99.7 / 99.8 | 5.529e+03 | 190 s | 0 | 0 |
| 2 | tip | auto | **identical, all 36 metric lines** | 5.529e+03 | 209 s | 19 RuntimeWarnings + GRID INTENT + grid check | 0 |
| 3 | `0097e5a` | `set_max_ram(30)` | **3.350 / 90.3 / 99.7 / 99.8** | **5.505e+03** | **96 s** | **0** | 0 |
| 4 | tip | `RAMB=30` | -- REFUSED -- | -- | ~1 s | refusal | **2** |

Run 3 is the campaign's named failure, reproduced independently: same four
scored digits on half the grid, twice as fast, and the ONLY quantity that moves
is the unscored peak (0.44 %).  Run 2's transcript is 15698 chars against run
1's 3715 -- the difference is entirely warnings the blanket filter used to
swallow.

---

## 5. HINDSIGHT SIBLING SWEEP

**Class A -- harness scripts without a `__main__` guard and/or with a blanket
`filterwarnings('ignore')`.**  The fix hardened 2.  There are 16 scripts under
`validation/repro_traced_carrier_121/` that drive the traced chain with no
guard; **9** also carry a bare `filterwarnings('ignore')` and **7** pass
`n_workers=8` (which `_script_has_main_guard` silently forces serial).  The
ones carrying BOTH -- i.e. able to reproduce the silent-8192 failure exactly as
`focus_scan_121.py` did -- are:

```
  capstone_stageB.py            n_workers=8 + blanket ignore
  carrier_chain_121.py          n_workers=8 + blanket ignore
  repro_dx_scaling.py           n_workers=8 + blanket ignore
  review_real_chain_convention.py  n_workers=8 + blanket ignore
  traced_group_dx_probe.py      n_workers=8 + blanket ignore
  traced_group_oracle.py        n_workers=8 + blanket ignore
  ablate_exitna_transpose.py    blanket ignore
  review_carrier_convention_2x2.py, stigmatic_control_121.py   blanket ignore
  _c14_pre_baseline_lens_traced.py, _d121_common.py, approx_common.py,
  wfe_probe_common.py, adjudicate_nfc_8192.py, capstone_stageC.py,
  p2diag_capture.py            unguarded (no blanket filter)
```

`capstone_stageB.py` is the capstone's own runner.

**Class B -- caches with a COUNT cap and no BYTE cap, holding grid-sized
arrays.**  I checked the whole package.  Two genuine members remain (the two
`fft_infra` caches that look like members, `_FREQ_GRID_CACHE` and
`_BANDLIMIT_CACHE`, are NOT: they store 1-D `kx_sq`/`ky_sq` and `bl_x`/`bl_y`,
so their exposure is O(N), not O(N^2)):

| cache | count cap | entry | exposure |
|---|---|---|---|
| `analysis/beam_stats._MESHGRID_CACHE` | 8 | a TUPLE OF TWO full `(Ny,Nx)` float64 grids from `np.meshgrid` | 8 x 2 x 2.147 GB = **34.4 GB at N = 16384**, unbounded in bytes |
| `analysis/zernike._ZERNIKE_BASIS_CACHE` | 32 | `(basis_matrix, mask)`; the matrix is `(n_modes, Npix)` float64 | 36 modes on a 4096^2 pupil = **4.8 GB per entry**, x 32 |

`_MESHGRID_CACHE` is the sharpest one: it is exactly the `np.meshgrid`
full-grid materialisation that item #2's headline removed from
`_lens_traced.py`, except here it is deliberately RETAINED in a module global,
bounded only by a count of 8.

---

## 6. NOT PROBED BY ME

Stated so nothing in this report reads as coverage it is not.

1. **The 32-order design-121 fan, at either cap.**  I re-derived every headline
   number of `ADJUDICATION_NFC_8192` from its committed CSV, but I did not
   re-run a shard.  The CSV is the harness's own output, so this checks the
   document against its data, not the data against the physics.
2. **The 11-configuration byte-compare of `FIX_PERF_POLY` sec 3.7.**  I ran ONE
   configuration, chosen to be outside that matrix.
3. **The 97-file traced niche set** -- the row the doc itself leaves unfilled.
4. **The WSL suite arms.**  WSL was used only for the numpy `c32` check.
5. **`k=2` at `NFC=16384`** -- refused by both guards, as the doc says.
6. **The pyFFTW copy-cost table** (184 / 597 / 6328 ms) that set the 2 GB
   threshold, and the `-14.03 GB` / `-35.7 GB` production-order totals: those
   are arithmetic on shapes nobody re-ran here either.
7. **`FIX_PERF_CACHES` sec 5.2's acceptance RSS column** (64.4 -> 62.4 GB).
8. **Concurrency.**  `_h_fft_cache_store` and the plan-cache entry-count branch
   were exercised single-threaded only.
9. **CuPy / JAX backends** of the separable route (documented as ignored there;
   I did not verify the `xp is np` gate on a device array).
10. **The fix's own fail-before substitutions** (the V3 `_poly` variant, the
    1-ulp `broadcast_to` shim).  My identity evidence is the independent
    `c8bcbcb` reference, not those.
11. **`_FINE_GRID_RAM_FRAC = 0.5`** -- untouched by the branch and untested by
    me; the branch says so too.

---

## 7. SUGGESTED ORDER OF WORK

1. **D1** -- one-character fix, but it is a wrong number in a public estimator
   and it is platform-split.  A pin at N = 8192 / complex128 that compares
   `estimate_asm_memory` against `keys * n_bufs * cb * npix` with the TRUE
   `n_bufs` would have caught it, and would run on both CI arms.
2. **Sec 3's stale arithmetic** -- `FIX_PERF_PARALLEL` sec 3.2/3.3/3.4 and the
   sec 4.2 transcript, plus the `_FINE_GRID_WORK_ARRAYS` comment's own
   "(20, 2.3 GB)" line and the two `137.4 GB` references.  These are the
   numbers a future re-tune will read.
3. **D2** -- either drop `E_analytic` from item #2's table and restate the
   total as `-30.6 GB`, or add a `numexpr` cache-drain to the element (the
   `_cache_registry` already exists for exactly this).
4. **D3** -- move the `np.copy` after the size check.
5. **D5 / D4** -- decide whether the paraxial worker should pay a
   design-121-class floor, and re-derive the 4096 row or widen its bar with a
   reason.
6. **The unfilled traced-niche-set row**, which the branch's own document names
   as a merge precondition.
