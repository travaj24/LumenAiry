# VERIFY-A12 — adversarial re-verification of WP-A12 (PMM 1-D / stack / core + CONVENTIONS §7.1)

Independent verifier.  Diff under test: `56a76f22` ("fix(pmm-1d): WP-A12 …"), base `56a76f22^`, files
`lumenairy/elements/pmm/{_core,stack,oned,conical}.py` and `CONVENTIONS.md`.  The working tree also
carries the PMM 2-D engineer's uncommitted edits to `twod*.py` / `stack2d*.py` / `_jax_twod*.py`;
those were not touched and are excluded from every judgement below except where a failure provably
originates in them.

Everything below was re-measured on this machine with `OPENBLAS_NUM_THREADS=1`.  No wall-clock number
is used as evidence.

---

## 1. Verdicts

| ID | Verdict | The one-line reason |
|---|---|---|
| **G1** (P1) | **VERIFIED-WITH-NOTES** | Both guards demonstrably run on the twin; one "after" number in the WP report does not reproduce by re-running the repro it cites (the fixture is now snapped away by G2). |
| **G2** (P2) | **VERIFIED-WITH-NOTES** | The cure reproduces exactly (6 / 1 / 0 of 11 rungs); collision-free stacks are byte-identical; the single-layer ownership rule holds. Two notes: an unreported user-visible collateral (a new snap warning + a changed solved geometry on ordinary fixtures, which broke a PMM-2-D test), and a shipped docstring sentence the WP's own data falsifies. |
| **G3** (P2) | **VERIFIED** | 0 of 400 + 0 of 400 + 0 of 300 kernel-level bit-identity, 0.000e+00 end-to-end A/B over 7 fixtures including lossy/anisotropic/conical, lazy-vs-eager verdict identity on 8 rows, digest key complete. |
| **G4** (P3) | **VERIFIED** | 0 mismatches in 34 322 fuzzed order-budget cells against a transcription of the pre-change block; the prepared-path defect reproduced and closed with numbers; the §7.1 sentence holds against an oracle I wrote, on both 1-D Jones solvers, over 6 angles. |
| Deferred items (Gegenbauer / hp mesh / eigen-recycling / memo) | **reasonable** | The `kx0`-pencil "not reproducible as proposed" conclusion is independently confirmed by cache-entry counts. |

**No REGRESSION found in the WP's own files.**  Five open items for the orchestrator are in §7; two
defects assigned to me mid-task and one documentation defect I found are fixed and re-verified in §8.

---

## 2. G1 — the hoisted guards on the differentiable `PMMStack.solve`

### 2.1 Re-running the audit's own repro (`repro/PMM-1D/p11_jax_guards.py`), unmodified

```
=== 11c: NaN / gain / grazing on the JAX path ===
  gain n_sup     numpy: RAISE ValueError :: PMMStack.solve: gain incidence medium (Im(n_superstrate) < 0; ...
                 jax  : RAISE ValueError :: PMMStack.solve: gain incidence medium (Im(n_superstrate) < 0; ...
```

against the audit's committed `out_jaxg.txt`, where the same row read
`jax : tot=[-0.84773 -0.86262]`.  **The gain half of G1 is closed, byte-identically, on both
branches.**  11a (the over-capacity ladder) still reads NumPy-vs-JAX `max|dR|` 1.249e-15 … 3.386e-15,
i.e. parity is untouched.

**Discrepancy (report accuracy, not code).**  The WP report §4 lists

> `G1 JAX guards (p11)` | audit: gain silent, sliver 8.35 silent | re-measured: **gain refuses, sliver warns twice**

Re-running `p11_jax_guards.py` as committed gives, for 11b, **neither**:

```
  deg= 12..20  numpy: T0(Ey)=0.7658976 tot=1.000000   jax: T0(Ey)=0.7658976 tot=1.000000
```

— no refusal, no warning, on either branch, at every degree.  The reason is G2: at the new default
`min_feature = period·1e-3` the script's `s = 1.5e-5` sliver is snapped away and the fixture carries
no sliver at all.  The claim is only true with `min_feature` pinned at the old default (which is what
the WP's own new test file does, via `_MF_OLD`).  The sentence as written in the report is not
reproducible from the artefact it names.

### 2.2 Independent re-measurement of the sliver half

Fixture the WP did not use: TiO₂-like 2.35 / 1.46, Λ = 0.55 µm, λ = 0.70 µm, 31°, two 0.22 µm layers,
`min_feature` pinned at `period·1e-5`, `s = 1.5e-5`:

| degree | NumPy branch | JAX branch | warnings on the JAX branch |
|---|---|---|---|
| 10 | returns, tot = 1.0000096 | returns, **tot = 8.2957** | SLIVER-SCREEN + ENERGY |
| 14 | **ValueError** (REFUSED) | returns, **tot = 42.666** | SLIVER-SCREEN + ENERGY |
| 18 | **ValueError** | returns, **tot = 8.2886** | SLIVER-SCREEN + ENERGY |
| 22 | **ValueError** | returns, **tot = 8.2832** | SLIVER-SCREEN + ENERGY |
| 26 | returns, tot = 1.0000434 | returns, tot = 1.0000026 | SLIVER-SCREEN only |

i.e. exactly the shipped contract: the geometric screen fires at **every** degree (it is geometry),
and the energy tripwire fires at exactly the corrupt ones.  At the shipped default the same geometry
is snapped and both branches return 1.0000000 under the `_pmm_union_grid` snap warning.

### 2.3 Adversarial probes the WP did not run

* **`jax.jit`, no re-trace pathology.**  Two distinct `eps_ridge` values through one `jax.jit`
  wrapper: **1 trace**, results 0.290340548034 and 0.290340817645.  The hoisted guard and the screen
  are host-side and do not enter the trace cache key.
* **`jax.grad`, clean stack.**  AD −/+: `d T0/d eps_h` = 0.488205328074 vs a central difference
  0.488205333225 at `h = 1e-6·eps` — **1.06e-8 relative**, finite, and **zero** warnings.
* **`jax.grad`, sliver stack.**  Gradient finite (10.4481 − 2.0992j), the geometric screen fires
  under the trace, the energy tripwire correctly stands down (outputs are Tracers).
* **TRACED `n_superstrate` — the documented carve-out.**  Inside `jax.jit` with `n_sup` as the traced
  argument, a gain medium `1 − 1e-3j` returns `ΣR+ΣT = −1.7199` with **no** raise and **no** energy
  warning.  This is the documented scope limit (the docstring now says so), but it is worth the
  orchestrator's attention: the geometric sliver screen *does* still fire in that configuration, so
  the trace path is not wholly unguarded.  With a *concrete* `n_sup` the guard raises at trace time,
  so the ordinary inverse-design shape (differentiate w.r.t. layer `eps`, fixed half-spaces) **is**
  covered under `jit`.
* **TRACED segment widths** — the case the new `allow_traced=True` screen could have broken, since
  `_cross_layer_sliver` calls `float()` on wall coordinates.  Measured with the screen shipped and
  with it bypassed in-process: **both** raise `ConcretizationTypeError`, from the twin itself, not
  from the screen.  **No new failure mode.**
* **No duplicated diagnostics from the hoist.**  `_require_propagating_incidence` is now called twice
  on a NumPy solve.  Warning count on a lossy superstrate (`n_sup` = 1+0.01j and 1+0.1j), hoist
  neutralised vs shipped: **1 → 1** in both cases.
* **Broad-except budget.** `git show`-counted `except Exception` per WP file, base vs HEAD:
  `_core.py` 5→5, `stack.py` 3→3, `oned.py` 2→2, `conical.py` 0→0, `_jax_stack.py` 3→3.  The report's
  "my files add zero" claim is correct.

**Verdict: VERIFIED-WITH-NOTES** — the fix is real and complete on the eager path; the report carries
one non-reproducible "after" line (§2.1) that the orchestrator should correct before it is quoted.

---

## 3. G2 — the raised `min_feature` default

### 3.1 The audit's own oracle, re-run (`repro/PMM-1D/p3e_band2.py`)

Reproduced **row for row**, including the cross-setting agreements the finding rests on:

| `min_feature` | rungs of 11 showing degree-scatter | verified |
|---|---|---|
| `period·1e-5` | 1×, 1.5×, 2×, 3×, 5×, 8× → **6 of 11** | ✔ |
| `period·1e-4` | only 1×, at one degree of five (0.19377343 at d26) → **1 of 11** | ✔ |
| `period·1e-3` | **0 of 11**, every rung degree-independent to 7 digits | ✔ |

`s = 1.5e-4` reads **0.19839028** under `1e-5` and `1e-4`; `s = 3e-4` reads **0.19755266** under both;
`s = 1e-3` reads **0.19366045** under `1e-5` and `1e-3` — to every printed digit.

`PMMStack(period).min_feature == period * 1e-3` exactly, for three periods.

### 3.2 Independent checks

* **Byte-identity on a collision-free stack** (Si₃N₄-like 2.02 / SiO₂ 1.45, Λ = 0.42 µm, λ = 0.633 µm,
  18°, three layers, a fixture the WP did not use): `max|A − B|` over `(orders, R, T, J)` between the
  old and the new default = **0.000e+00** at degree 10, 16 and 22, with **0** warnings on both arms.
* **Single-layer ownership under the raised default.**  A liner of width `t` that *one* layer owns,
  beside another layer's wall: the union grids at `1e-5` and `1e-3` are **byte-identical** for
  `t` = 1e-2, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6 of a period, with no snap warning — the liner survives at
  every width.  The **cross-layer twin** of the same separation is snapped at the new default only
  (narrowest union cell `s` → ≈ 0.5 of a period, i.e. the cell is *removed*, not thinned) for
  `s` = 5e-4, 1e-4, 1e-5, and untouched at `s` = 1e-3 (at the threshold) and above.  Now pinned in
  `tests/unit/test_audit2609_a12_verify_pmm1d.py`.

### 3.3 NOTE 1 — unreported collateral: the snap now fires, and moves geometry, on ordinary fixtures

Raising the default two decades means `_pmm_union_grid` now snaps — and **warns**
(`_pmm_union_grid: snapped N pair(s) of NEAR-COINCIDENT cross-layer walls …`) — on any stack whose
cross-layer walls are 1e-5 … 1e-3 of a period apart.  The WP report's residual-risk paragraph states
the geometry bound (`≤ min_feature/2` wall movement) but does not mention the new warning, and lists
only three fixtures as fallout.  There is a fourth, reported to me by the PMM-2-D engineer and
confirmed here: `tests/unit/test_fix_pmm2d_mortar_round2.py::
test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`.

Measured on that fixture (a plain 1-D `PMMStack` whose two layers' walls differ by `delta`):

| | `delta` = 1e-4 | `delta` = 1e-5 |
|---|---|---|
| min LAPACK rcond of `Wb`/`Vb`, **old** default | 9.6940e-11 | 9.7297e-13 |
| min LAPACK rcond, **new** default | **4.2942e-04** | **5.2104e-04** |
| max `R+T`, old default | 1.0000004784 | **3.6124215325** |
| max `R+T`, new default | 1.0000000000 | 1.0000000000 |
| warnings, old default | 0 | 1 (`energy not conserved`) |
| warnings, new default | **1 (`_pmm_union_grid: snapped 2 pair(s)`)** | **1 (same)** |

So the new default does not merely add a warning there: both wall pairs snap to coincidence, the two
layers become **geometrically identical**, and the near-singular interface the test exists to measure
ceases to exist (rcond moves 7 decades, the measured gap goes from 1.998 decades to −0.084).  Three
of its assertions break, not one (`nwarn_ok == 0`, `rc_ok < 1e-9`, and the premise gate).  **Fixed
here** — see §8.1.

A second trace of the same collateral is inside the WP's own diff: `test_pmm_m3_efficiency.py`'s
T3-4 fixture gained `warnings.filterwarnings("ignore", message=".*_pmm_union_grid.*")` alongside its
`min_feature` pin.  The new warning was therefore known at authoring time; it just did not reach the
report or the changelog, where it belongs.

The general statement for the changelog: *a stack with cross-layer wall collisions between 1e-5 and
1e-3 of a period now solves a different (snapped) geometry and says so.*  On the audit's own second
fixture the answer moves by **3.8e-3 relative** at `s = 3e-4·P` (T₀ = 0.19829790 snapped vs
0.19755266 unsnapped).  That number belongs in the migration note beside the geometry bound.

### 3.4 NOTE 2 — a shipped sentence the WP's own data falsifies

`stack.py::_MIN_FEATURE_DEFAULT_FRAC`, the new `PMMStack.__init__` comment, the new `min_feature`
`Parameters` entry and the changelog all say:

> the corruption band is MEASURED at roughly `[1, 8] * min_feature` in the collision width

Read literally that makes the fix impossible: everything below `min_feature` is snapped, so every
*surviving* collision would sit in `[1×, 8×]` and be corrupt, whatever the threshold.  The WP's own
`p3e` table refutes it — at `min_feature = 1e-3·P` the rungs **1.0× … 8.0×** (`s` = 1e-3 … 8e-3 of a
period) are degree-independent to 7 digits.  The hazard band is **absolute in period fractions**
(≈ `[1e-5, 1e-4]·P` on both audit fixtures over degrees 10–26), which is precisely *why* a `1e-3`
default clears it; the `[1,8]×min_feature` framing was an artefact of both audit sweeps having been
run at `min_feature = 1e-5·P`.

The `1e-4` row is the second, independent refutation: the ratio reading predicts 6 of 11 rungs
corrupt there (1× … 8× of `1e-4` = 1e-4 … 8e-4), and **1** is measured.

The rule of thumb three lines below it ("pass `min_feature` at least ~10× the collision scale") is
correct and is what a caller should act on — but it contradicted the sentence above it.  **Corrected
in the source** at all three sites (§8.3); the same correction is still owed to
`WP-A12_CHANGELOG.md`, which I did not edit (§7, item 2).

**Verdict: VERIFIED-WITH-NOTES.**

---

## 4. G3 — cost: diagonal masses, the lazy arbiter, the cache key

### 4.1 The exactly-real-diagonal fast path (including the lossy `1/eps` case the brief names)

| probe | result |
|---|---|
| `_real_diagonal` on a real `float64` diagonal | fast path (3 of 3 sizes) |
| on a **real-valued `complex128`** diagonal | fast path |
| on a genuinely **complex (lossy `1/eps`)** diagonal | **`None` — stays dense** |
| on a near-diagonal matrix (one 1e-30 off-diagonal entry) | `None` |
| on a diagonal with a zero entry | `None` |
| `_safe_inv` / `_safe_solve` vs LAPACK, well-scaled **real f64** diagonals | **0 of 400 differ** |
| same, well-scaled **real-valued complex128** | **0 of 400 differ** |
| same, genuinely complex diagonals (must stay on the shipped dense arm) | **0 of 400 differ from plain `inv`** |
| `_row_scale_apply(S0)(X)` vs `inv(S0) @ X` | **0 of 300 differ** |
| **ill-scaled** diagonals must stay on the *equilibrated* arm | **0 of the trials left it** |

Physics-side gate (now pinned in my test file): for a three-segment SEM cell,
`mass["inv_xx"]` of a **lossless** middle segment takes the fast path and of a **lossy** one
(`eps = 2.25 + 0.4j`) returns `None`; `S0` takes it in both cases (it is geometric).  This is the
claim the brief asked me to break and it holds.

### 4.2 End-to-end bit-identity A/B (mine, not the WP's captured pair)

`_real_diagonal` forced to `None` in-process (which restores the pre-change arithmetic exactly)
against the shipped path:

| fixture | `max\|ON − OFF\|` |
|---|---|
| single-layer Jones, normal | 0.000e+00 |
| single-layer Jones, oblique 0.31 rad | 0.000e+00 |
| single-layer Jones, **lossy Au** (0.18+3.43j) | 0.000e+00 |
| single-layer Jones, **anisotropic lossy** diag(4+0.2j, 4.4+0.1j, 4.2+0.3j) | 0.000e+00 |
| 3-layer `PMMStack`, lossless | 0.000e+00 |
| 3-layer `PMMStack`, **lossy** | 0.000e+00 |
| 3-layer `PMMStack`, **conical φ = 0.4** | 0.000e+00 |

### 4.3 The lazy arbiter

Deterministic counts of `PMMStack.solve` invocations (the audit's instrument), guard OFF / eager /
lazy, on the O-11 taper:

| n_slices | `s` | degree | OFF | eager | lazy | verdict identical? | returned answer identical? |
|---|---|---|---|---|---|---|---|
| 2 | 3e-4 | 14 | 1 | 4 | **2** | yes | **bit-identical** |
| 2 | 1e-4 | 14 | 1 | 4 | **2** | yes | **bit-identical** |
| 4 | 3e-4 | 14 | 1 | 4 | **2** | yes | **bit-identical** |
| 8 | 3e-4 | 12 | 1 | 4 | 4 | yes (same `ValueError` text) | n/a — both refuse |
| 8 | 1e-4 | 12 | 1 | 4 | 4 | yes | n/a — both refuse |
| 16 | 5e-5 | 10 | 1 | 4 | 4 | yes | n/a — both refuse |
| 4 | 0 | 14 | 1 | 1 | 1 | yes (control) | bit-identical |
| 1 | 0 | 14 | 1 | 1 | 1 | yes (control) | bit-identical |

Evidence dict on a `truncation` fork, eager vs lazy: the **only** difference is that
`{d12, d0_over_d12, closed_super_unity}` become `None`.  Every other field is equal —
`move = 6.67687e-04`, `w_wide = 1e-04`, `mf_fix = 2.4e-10`, `closure = 1e-05`,
`violation = 8.33343e-07`, `drop = 2.15692e+07`, `d0_over_w = 6.67687`,
`snapped_super_unity = 3.86358e-14`.  Now pinned field-by-field in my test file.

**One residual, theoretical, severity LOW.**  Before the change, a failing `_sliver_collapse_solve`
(it returns `None` when the re-solve raises — e.g. the collapsed grid's smaller `n_glob` tripping the
order-budget refusal) turned *any* fork into `("unknown", None)`, whose caller **raises** above the
super-unity bar on a provably passive stack.  Under `PMM_SLIVER_ARBITER_LAZY = True` a `truncation`
row never reaches those solves, so that same stack would now be returned with a warning instead of
refused.  I hunted for the combination (verdict `truncation` **and** `worst > 1 + 1e-2`) over
**120 cells** (n_slices ∈ {2,3,4,6,8,12} × s ∈ {5e-5,1e-4,2e-4,3e-4,5e-4} × degree ∈ {10,12,14,18})
and found **0**.  Forcing `_sliver_collapse_solve → None` on the fixture rows that do exist gives the
same outcome on both arms (both refuse on the `sliver` rows, both return on the `truncation` rows).
Recorded, not blocking.

### 4.4 The geo-eig digest key

| property | result |
|---|---|
| digest length | **32 bytes**; full key repr < 512 chars against a 147 456-byte operator |
| same array twice | same key |
| different tag (`scalar` vs `tensor`) | different key |
| **1-ULP** change in one entry | different key |
| different **dtype** (`complex128` → `complex64`) | different key |
| same bytes, different **shape** | different key |
| F-order copy of the same values | same key (correct: same operator) |
| `kx0`/`k0` handling — normal incidence, 2 wavelengths | **1** cache entry (reuse, as designed) |
| oblique 0.31 rad, 2 wavelengths | **2** entries (no reuse — the documented structural limit) |
| enrolment | `isinstance(_GEO_EIG_CACHE, ByteBudgetedLRU)` ✔, present in `_LIVE_CACHES` ✔, in `cache_report()` ✔ |

Two different operators with equal digest is a 256-bit collision; and the key carries dtype and shape
*outside* the digest, so a length-extension-style aliasing across differently-shaped blocks is also
excluded.  Stale-hit stress from the audit's own `p9_cache_threads.py`: **0 mismatches** over 8
`(wl, angle, n_ridge)` cases and **0** on the shuffled replay; 16 concurrent threaded solves vs serial
`max|d| = 0.000e+00` (so the `OrderedDict`+lock → `ByteBudgetedLRU` swap is thread-safe in practice as
well as by construction — `ByteBudgetedLRU` takes the shared `_BUDGET_MUTEX`).

`_GEO_EIG_CACHE_SIZE` / `_GEO_EIG_CACHE_LOCK` are referenced nowhere else in the repository
(grepped); `test_v5_20_7_pmm_geo_eig_cache.py` uses `len(_GEO_EIG_CACHE)`, which the new class
supports, and passes.

**Verdict: VERIFIED.**

---

## 5. G4 — the order-budget helper, the prepared path, `pol=`, §7.1

### 5.1 `_farfield_order_set` against the pre-change block, fuzzed

I transcribed the pre-change block from `56a76f22^` (both shapes: scalar `n_glob`, and the
`(n0, nN)` per-layer pair) and compared **orders, `half`, whether it refused, and the exact refusal
message**:

* scalar: **0 mismatches in 9 234 cells** — period ∈ {0.4, 1.0, 3.1} µm × wl ∈ {0.4, 0.633, 1.55} µm
  × n_max ∈ {1.0, 1.5, 3.48} × ffo ∈ {1,2,5,6,11,12,21,40,61} (odd **and** even, below and above
  `2m+5`) × n_glob 2…39 (odd **and** even capacity, on both sides of the refusal).
* pair: **0 mismatches in 25 088 cells**.
* `kx`: identical to `(kx0 + m·G)/k0` on 3 `(kx0, k0, period)` triples, including negative `kx0`.

A trimmed version (2 187 + 6 400 cells) is now a regression test in
`tests/unit/test_audit2609_a12_verify_pmm1d.py`, so the *behaviour* of the survivor is gated and not
only the *absence* of copies.

### 5.2 The `_PreparedPMMStack.solve` refusal — reproduced, with numbers

Fixture where it matters: Λ = 3.0 µm, λ = 0.5 µm into n = 1.5, `far_field_orders = 41`, so **19
orders propagate** (m_prop = 9).  Pre-change block restored in-process vs shipped:

| degree | BEFORE (`clamp, no refusal`) | AFTER |
|---|---|---|
| 4 | returns **7 orders**, max `R+T` = **0.893705**, **0 warnings** | `ValueError: PMMStack.prepare().solve: degree=4 too low …` |
| 5 | returns **9 orders**, max `R+T` = **0.993955**, 0 warnings | refuses |
| 6 | returns **11 orders**, max `R+T` = **0.959242**, 0 warnings | refuses |
| 20 (capacity sufficient) | returns 39 orders | returns 39 orders, identical |

This is exactly the described defect — sub-unity power invisible to a one-sided tripwire — and the
WP report gives no numbers for it.  They are now pinned (order count `< 2m+1`, total `< 1`, warning
count `0` on the pre-change arm; a raise on the shipped one).

### 5.3 `internal_field(pol=…)` vs the `R_eff` row order

`pol=0` is **exactly** `incident=(1,0)` and `pol=1` exactly `incident=(0,1)` (array-equal over every
returned field).  `'tm'`, `'TM'`, `' p '`, `'P'`, `'x'` ≡ row 0; `'te'`, `'TE'`, `'s'`, `'S'`, `'y'`
≡ row 1.  `'xy'`, `''`, `'tem'`, `2`, `-1`, `1.5` all raise with the §2 `PMMStack.internal_field: `
prefix; `None` → row 0 (documented).  The row-order claim itself, measured independently:
`|J[:,0]|² = 0.01849492 = R_eff[0, m0]` and `|J[:,1]|² = 0.00809560 = R_eff[1, m0]`.

Minor: `'x'` / `'y'` are accepted but appear neither in the `internal_field` docstring nor in the
error message that "names every accepted value".

### 5.4 CONVENTIONS §7.1 — `J[0,0] = −r_p` at φ = 0, both 1-D solvers

Oracle: a three-medium characteristic-matrix TMM written by me, convention stated explicitly as
`r_p = (n₂cosθ₁ − n₁cosθ₂)/(n₂cosθ₁ + n₁cosθ₂)` (so `r_p(0°) = −r_s(0°)`), no library code in it.
Slab n = 2.1, d = 0.32 µm, λ = 0.55 µm, n_sup = 1, n_sub = 1.5:

| angle | 0° | 20° | 30° | 45° | 60° | 75° |
|---|---|---|---|---|---|---|
| `pmm_jones_1d` `J[0,0]/r_p` | −1.0000000 | −1.0000000 | −1.0000000 | −1.0000000 | −1.0000000 | −1.0000000 |
| `pmm_jones_1d` `J[1,1]/r_s` | +1.0000000 | +1.0000000 | +1.0000000 | +1.0000000 | +1.0000000 | +1.0000000 |
| `PMMStack` `J[0,0]/r_p` | −1.0000000 | −1.0000000 | −1.0000000 | −1.0000000 | −1.0000000 | −1.0000000 |
| `PMMStack` `J[1,1]/r_s` | +1.0000000 | +1.0000000 | +1.0000000 | +1.0000000 | +1.0000000 | +1.0000000 |

The WP's own test covers only `pmm_jones_1d` at 0/30/60°; `PMMStack` (a different cascade and a
different far-field assembly) at 0/20/45/75°, plus `|J[0,1]|, |J[1,0]| < 1e-13` on the unpatterned
cell, is now pinned in my file.  The §7.1 sentence is correct as written for **both** 1-D solvers.

### 5.5 The Wood-nudge `fn_name` wiring (G3(d) in the report)

Solving at `wl = period` on an air-clad cell, the `WoodNudgeWarning` names:
`pmm_efficiency_1d` ✔, `pmm_jones_1d` ✔, `PMMStack.solve (conical)` ✔.  The claim holds.  There is
**no regression test** for this anywhere in the WP's test file — it is a behaviour change (a new
warning category on ~8 entry points) resting on a manual check only.

### 5.6 Residual: `conical.py` is still outside the "one definition"

`conical.py` keeps its own order-budget block (`cap = (min(n_glob_sup, n_glob_sub) − 1)//2` or
`(nU·n_el·degree − 1)//2`, then `if m_prop > cap: raise`) — a genuinely different contract, and the
WP says so.  But the shipped DISCOVERED gate
(`test_g4_there_is_one_definition_of_the_far_field_order_budget`) asserts that tokens
`"n_proj = max("` and `"cap = n_glob if"` are absent from `conical.py`, which is vacuously true
because that file never used them.  So the gate does not actually guard the conical copy; a divergence
there would still ship silently.  Severity LOW (the T3-3 defect this pattern caused was in that very
file), recorded as open item 4.

**Verdict: VERIFIED.**

---

## 6. Collateral damage sweep, and the audit's "keep intact" list

### 6.1 The audit's own repro scripts, re-run on this tree

| property | audit | re-measured here |
|---|---|---|
| energy, lossless Si/SiO₂ (`p5`) | ≤ 3.1e-12 | **3.114e-12** worst |
| reciprocity TM `m = −1` (`p5`) | 1.8e-11 | **1.775e-11** |
| PMM vs the library's RCWA, TE 0/17/45° (`p5b`) | 4.6e-8 / 7.5e-8 / 1.2e-6 | **4.622e-8 / 7.528e-8 / 1.157e-6** |
| unpatterned vs an independent TMM (`p6`) | ≤ 3.8e-14 | **3.841e-14**; `Δ\|r\|` 2.30e-14, Δphase ≤ 1.7e-13; cross-pol ≤ 9.1e-45 |
| conical vs classical (`p7`) | 2.3e-13 | **2.325e-13**; `pmm_jones_1d_conical` vs `PMMStack(φ=π/2)` **max\|ΔJ\| = 0.0** |
| conical energy ladder (`p7`) | ≤ 6.2e-11 | ≤ **2.62e-11** |
| cache content-keying + shuffled replay (`p9`) | 0 mismatches | **0**, twice |
| 16 threads vs serial (`p9`) | exactly 0 | **0.000e+00** |
| cached arrays read-only (`p9`) | "geo mu: WRITEABLE" in the committed `out_cache.txt` | **identical** — pre-existing, not a regression |
| unit invariance over 13 decades (`p9d`) | R₀ = 0.08536814608764 ± 6e-14 | **0.08536814608760 … 0.08536814608770** over km → m → µm → nm → Å, i.e. the same ±6e-14 band; `J₀₀` identical to 12 digits |
| G2 hazard band (`p3e`) | 6 / 1 / 0 of 11 | **6 / 1 / 0**, unsnapped rungs agree exactly |
| G1 JAX guards (`p11`) | gain silent → refuses | **refuses, byte-identical string** (sliver half: see §2.1) |

### 6.2 Test runs

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| what | result | time |
|---|---|---|
| **batch 1** — the WP's new file + the 7-file sliver corpus | **108 passed, 1 failed, 1 skipped** | 178 s |
| **batch 2** — 24 further PMM 1-D / alias / conventions / cache / conical / taper / autodiff files | **601 passed, 1 failed** | 489 s |
| **my new** `tests/unit/test_audit2609_a12_verify_pmm1d.py` (16 tests) | **16 passed** | 1.8 s |
| **fail-before** — both A12 test files with A12's behavioural levers neutralised in-process | **14 failed, 23 passed** | 23 s |
| `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why` after my fix | **1 passed** | 1.1 s |
| `ruff check` on the five WP source files + `pmm/__init__.py` + three test files | **All checks passed** | — |

Totals across the two batches: **709 passed, 2 failed, 1 skipped**, and both failures are
pre-existing and not WP-A12's:

1. `tests/unit/test_verify_pmmstack_sliver_walls.py::test_the_pure_stacks_shared_grid_cannot_express_a_sliver`
   — the traceback terminates in `lumenairy/elements/pmm/twod_staggered.py:554`,
   `PMM2DStackPure.add_layer`'s pencil-DOF refusal (`299538x299538 dense generalized pencil … above
   max_pencil_dof=12000`).  That module is PMM-2-D, owned by WP-A13, and is uncommitted in the
   working tree.  No 1-D code path is reached.
2. `tests/unit/test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band`
   — I re-ran it with **every** WP-A12 behavioural lever neutralised in-process (`_real_diagonal` →
   `None`, `_MIN_FEATURE_DEFAULT_FRAC` → 1e-5, `PMM_SLIVER_ARBITER_LAZY` → `False`, the three new
   guard hooks stubbed out).  It fails with the **byte-identical** screen tuple
   `[('uncoated ns=3', 10, 0, 2, 0), ('25 nm coat ns=8', 6, 0, 1, 0), ('25 nm coat ns=8', 8, 0, 2, 0),
   ('25 nm coat ns=8', 10, 0, 2, 4)]`, i.e. every cell classification-unsound (`n_growing > 0`) on
   the halfwidth-2 runs.  That confirms the WP's attribution: a build property of this box, not a
   library change.  (Its `_mk` now pins `min_feature = PERIOD*1e-5`, which is exactly the pre-fix
   default, so the geometry under test is unchanged from base.)

**Fail-before, independently.**  With the same neutralisation plugin, 8 of the WP's 21 tests fail
(`test_g1_*` ×3, `test_g2_*` ×3, `test_g3_a_truncation_verdict_pays_two_solves_not_four`,
`test_g3_diagonal_mass_shortcuts_are_bit_identical_to_lapack`) and 6 of my 16 fail.  The rest of both
files are fail-before *structurally* — the symbol under test (`_farfield_order_set`, `_geo_eig_key`,
`_resolve_internal_pol`, `_row_scale_apply`, `_warn_stack_energy_concrete`, `_is_traced_output`,
`PMM_SLIVER_ARBITER_LAZY`) does not exist at `56a76f22^`, so they error there rather than fail.

---

## 7. Open items for the orchestrator

1. **(MEDIUM — report/changelog accuracy, no code change)**  WP-A12_REPORT §4 and the changelog's G1
   table claim `p11_jax_guards.py` now reads "sliver warns twice".  Re-running that script as
   committed shows the sliver fixture **snapped away by G2** and silent on both branches (§2.1).
   Correct the line to say the sliver half is measured with `min_feature` pinned at `period·1e-5`
   (which is what the WP's own test does), or the claim will not survive the next re-run.

2. **(MEDIUM — changelog, already fixed in the source)**  The `[1, 8] * min_feature` framing is
   corrected in `stack.py` at all three sites (§8.3), but it also appears in
   `WP-A12_CHANGELOG.md` ("a collision of size `s` corrupts the answer for `s` in roughly
   `[1, 8] * min_feature`"), which I did not edit.  Replace it there with the absolute band before
   the orchestrator assembles the real CHANGELOG entry.  Also in that file: the `### Performance`
   heading reads *"the sliver arbiter's extra solves are lazy **and memoized**"* while the body of
   the same section records that the memo was implemented and then **withdrawn** — drop "and
   memoized" from the heading.

3. **(MEDIUM — migration note)**  Add the **answer-side** number to the G2 migration note beside the
   geometry bound: on the audit's own second fixture, a collision at `s = 3e-4·P` now solves the
   snapped geometry and reads `T₀ = 0.19829790` against 0.19755266 unsnapped — **3.8e-3 relative**.
   And state that the snap **warns** now where it did not before, because that warning is what broke
   a sibling test (§3.3) and will surface in user logs.

4. **(LOW)**  Three small residuals, none blocking: `conical.py`'s order budget is outside the
   consolidation *and* outside the gate that claims to guard it (§5.6); the Wood-nudge `fn_name`
   wiring has no regression test (§5.5); `internal_field` accepts `'x'`/`'y'` without documenting
   them (§5.3).  Also recorded: the lazy arbiter's `truncation` + failing-collapse-solve corner
   (§4.3), unreached in 120 cells.

5. **(LOW — test-suite hygiene, not a defect)**  Three of the WP's new tests hard-fail rather than
   skip when a BLAS-dependent premise is absent
   (`test_g1_the_jax_twin_screens_the_geometry_that_numpy_refuses`,
   `test_g2_the_raised_default_kills_the_degree_scatter`,
   `test_g3_a_truncation_verdict_pays_two_solves_not_four`).  Their *unconditional* halves are
   build-free and correct, and hard-failing is the conservative choice — but the sibling round-4 file
   uses a documented `pytest.skip` for this same premise, and the round-4 docstring records that the
   CI runner's kernel reads `R+T = 1.000115` where this box reads 3.61.  Expect at least the first of
   the three to be red on that runner.  I have closed the most important gap (the energy tripwire's
   contract) build-free in my own file, so the G1 claim is now pinned on every arm regardless.

---

## 8. Defects I fixed, and their verification

The first two were assigned to me mid-task by the orchestrator; the third I found (§3.4).

### 8.1 `tests/unit/test_fix_pmm2d_mortar_round2.py` — the plain-1-D interface fixture

The brief offered two options (scope the snap warning, or narrow the `nwarn_ok == 0` assertion).
**Neither is right**, and the measurement in §3.3 is why: the warning is only the first of three
assertions to break, and the two behind it break because the fixture's *object of measurement* — a
near-singular interface built from a cross-layer wall pair — is destroyed by the snap.  Narrowing the
warning assertion would leave a green test measuring a well-conditioned interface between two
identical layers, i.e. exactly the "test that pins the wrong thing" shape §14 of the audit is about.
Scoping the shipped warning would be worse: the snap genuinely changed the solved geometry there, and
suppressing the notice would hide that from users.

I pinned `min_feature=_P * 1e-5` on the fixture's `PMMStack(...)` — the same remedy WP-A12 applied to
its three other inherited fixtures — with a comment carrying the measured before/after.  This
**restores the committed census numbers exactly**: `rcond` 9.6940e-11 (the value the test's own
comment records as "MEASURED 9.6940e-11 .. 9.6969e-11 over the eight committed arms"), gap **1.998**
decades (recorded range 1.965–1.999), `R+T` = 3.6124 at `delta` = 1e-5 (recorded 3.6116 on
Sandybridge).  It is a fixture pin, not a relaxed bar: every assertion and every bar is unchanged.
Verified: `1 passed in 1.09s`.

### 8.2 `lumenairy/elements/pmm/__init__.py`

Added `"pmm_2d_order_drift"` to `__all__` beside `"pmm_efficiency_2d_cell_vs_wavelength"`, as
requested.  `twod.py` already exports it in its own `__all__` and the facade does `from .twod import
*`, so the name resolves; only the facade's `__all__` was missing it.  The top-level
`lumenairy/__init__.py` re-export was left alone per the instruction.  `ruff check` clean.

### 8.3 `lumenairy/elements/pmm/stack.py` — the `[1, 8] * min_feature` sentence (comments only)

Three sites (`_MIN_FEATURE_DEFAULT_FRAC`'s `#:` block, the `min_feature` entry in `PMMStack`'s
`Parameters`, and the `PMMStack.__init__` comment) told the caller the corruption band scales with
`min_feature`, which makes the fix that same paragraph justifies logically impossible (§3.4).  Each
now states the band as measured — **absolute, ~1e-5 … 1e-4 of a period, both fixtures, degrees
10–26** — and says in one clause why that is what makes raising the threshold a cure.  The
collision-scale rule of thumb, the ladder and every number the WP measured are untouched.

Comment-only: no executable line changed.  `ruff check` clean; both A12 test files re-run after the
edit — **37 passed in 43 s** (21 the WP's + 16 mine).

---

## 9. Tests added

`tests/unit/test_audit2609_a12_verify_pmm1d.py` — 16 tests, **16 passed in 1.8 s**, ruff clean.
Each closes a gap in the WP's own file; all but one fail structurally on `56a76f22^` (the symbol
under test does not exist there), and the exception is noted below.

| test | what it pins that the WP's file does not | fail-before |
|---|---|---|
| `test_g1_the_concrete_energy_tripwire_is_exact_and_build_free` | the three severities of `_warn_stack_energy_concrete` on arrays built in the test, probed one decade either side of `_STACK_SUPERUNITY_BAR` — the G1 energy half with **no** BLAS-dependent premise | symbol absent at base |
| `test_g1_the_tripwire_stands_down_on_a_traced_output` | `_is_traced_output` False on a concrete `jax.Array` (so the eager case the audit measured is still tripped) and True inside `jit`, where the tripwire returns `False` and raises nothing | symbol absent |
| `test_g2_a_single_layer_thin_feature_survives_the_raised_default` (×6 widths) | byte-identical union grids old-vs-new default for a liner one layer owns, 1e-2 … 1e-6 of a period | `_MIN_FEATURE_DEFAULT_FRAC` absent |
| `test_g2_the_cross_layer_twin_of_that_collision_is_snapped` (×3) | the two-sided half: the same separation across two layers *is* removed | same |
| `test_g4_the_order_budget_helper_is_the_pre_change_block_on_every_input` | 2 187 + 6 400 fuzzed cells against a transcription of the pre-change block (orders, `half`, refusal, refusal text), plus the `kx` formula | `_farfield_order_set` absent |
| `test_g4_the_prepared_path_returned_orders_missing_before_the_fix` | the fail-before **with numbers**: 7/9/11 orders where 19 propagate, `R+T` 0.894/0.994/0.959, 0 warnings; then the refusal | no refusal at base |
| `test_g4_the_stack_solver_also_returns_the_lab_cartesian_jones` | §7.1 on `PMMStack` (the other 1-D Jones solver) at 0/20/45/75° against a TMM written in the test, plus vanishing off-diagonals | characterisation only — the library was always internally consistent; only the document was wrong |
| `test_g3_a_lossy_inverse_permittivity_mass_stays_on_the_dense_path` | the real-vs-complex gate from the **physics** side (a lossy segment's `1/eps` mass), two-sided against its lossless twin | `_real_diagonal` absent |
| `test_g3_the_lazy_arbiter_leaves_every_measured_field_untouched` | field-by-field equality of the evidence dict between the two switch arms, with the `None` set bounded to exactly the three collapse-solve products | `PMM_SLIVER_ARBITER_LAZY` absent |

---

## 10. Files I touched

* `tests/unit/test_audit2609_a12_verify_pmm1d.py` (**new**, 16 tests)
* `tests/unit/test_fix_pmm2d_mortar_round2.py` (one fixture pin + comment, §8.1 — authorised by the
  orchestrator)
* `lumenairy/elements/pmm/__init__.py` (`__all__` += `"pmm_2d_order_drift"`, §8.2)
* `lumenairy/elements/pmm/stack.py` (three comment/docstring sentences, §8.3 — no executable line)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-A12.md` (this file)

I found no *behavioural* defect in `_core.py`, `stack.py`, `oned.py`, `conical.py` or
`CONVENTIONS.md`: every G1–G4 change measures as claimed, the bit-identity claims hold at 0.000e+00
on fixtures the WP did not use, and the order-budget consolidation reproduces the pre-change block on
34 322 fuzzed inputs.  The only source edit is the documentation correction in §8.3.  No git write of
any kind was performed, and no file owned by another work package was modified apart from the one
test assertion the orchestrator assigned.
