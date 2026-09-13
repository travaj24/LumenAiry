# VERIFY-WP-B6 — independent adversarial re-verification of WP-B6 (the PMM Gegenbauer negative, and the PMM-2D k0-free tensor operator cache)

Verifier: VERIFY-B6.  Diff under test: commit `3af33dff` (parent `3af33dff^` = `2871e92e`), files
`lumenairy/elements/pmm/{_core,stack2d,twod_jones}.py`, `docs/history/lumenairy.elements.pmm.stack2d.md`
and the new `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py`.  Machine: Windows 11 Pro
for Workstations, CPython 3.14.6, numpy 2.4.6, scipy 1.17.1, OpenBLAS; every invocation with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a time.  Branch HEAD
had moved to `8dab7de5` by the end of this pass; nothing after `3af33dff` touches
`lumenairy/elements/pmm/*`, and those files were clean in the working tree throughout.

**Every byte-identity and cost measurement below is ARCHIVE-TO-ARCHIVE** — `git archive 3af33dff^ lumenairy`
and `git archive 3af33dff lumenairy`, each extracted read-only into its own scratch directory and
imported in a child process whose `cwd` and `PYTHONPATH` are that tree, with `lumenairy.__file__`
asserted before anything is measured, never through pytest.  No sibling's in-flight edit to the
carrier, rcwa/eme/bor, analysis, sources, raytrace, `_lens_traced`, `lenses_maslov` or the
propagator kernels is inside any A/B here.

I did not write WP-B6.  Nothing in its report was taken on trust: item 1 was re-derived from
scratch on my own fixtures with my own ultraspherical family, my own exact-integration control and
my own extrapolated RCWA oracle; item 2 was re-measured on cells the engineer did not use, and the
cache was attacked at its key, its budget, its freezing and its gates.

---

## 1. Verdicts

| Claim (WP-B6 report) | Verdict | My oracle and my numbers |
|---|---|---|
| **Item 1 / Finding 1** — with exact element integration the whole λ family gives ONE answer; a nodal-basis change cannot move a Galerkin answer on the fixed C0 piecewise-`P_N` space | **VERIFIED** (stronger than claimed) | My own family (Golub–Welsch on the monic symmetric-Jacobi recurrence, cross-checked against `scipy.special.roots_jacobi` to ≤ 7.8e-16) + my own exactly-integrated `_build_sem`.  Max spread over λ ∈ {0, ¼, ½, ¾, 1, 1½} of `concat(R, T)`, on **five** fixture/polarisation combinations at degrees 16/24/32: **EXACT 5.2e-14 … 8.4e-13** against **LUMPED 7.5e-06 … 8.0e-03** — ratios 2.5e+07 … 7.2e+10.  The CONSISTENT error ladders are identical **digit for digit** across all six λ on every fixture (§3.2) |
| **Item 1 / Finding 2** — what λ varies is the QUADRATURE, and only GLL satisfies `M D + (M D)^T = diag(−1, 0…0, +1)` exactly (1e-14 vs 0.4–1.5) | **VERIFIED** | My own `D` (product rule, not barycentric; agrees with the shipped `_lagrange_derivative_matrix` to ≤ 4.5e-13) and my own interpolatory weights (Gauss quadrature of the Lagrange basis, cross-checked against a Legendre-Vandermonde moment solve to ≤ 2.0e-15).  Residual at λ = ½: **1.41e-14 (deg 8) … 1.32e-12 (deg 44)**; at every other λ **3.70e-01 … 2.20e+00**; exact integration restores it (≤ 1.3e-12) at every λ.  Table in §3.3 |
| **Item 1 / Finding 2** — the λ ≠ ½ lumping breaks the lossless oblique closure (3e-14 → 1e-08…1e-05) | **VERIFIED** | Three lossless oblique cells at degree 24: λ = ½ closes to **6.4e-14 / 1.1e-13 / 1.5e-13**; λ ∈ {0, ¼, ¾, 1, 1½} to **1.2e-08 … 8.9e-05**; the CONSISTENT control restores ~1e-14 at *every* λ, so the defect is the quadrature and nothing else (§3.4) |
| **Item 1 / Finding 3** — TE regresses 40–110× | **VERIFIED, and understated** | Against a degree-120 PMM self-reference validated against my RCWA TE oracle to **1.9e-11 / 3.9e-12**: at degree 24, λ = 0 is **489×** (Ag/air) and **211×** (dielectric) worse than λ = ½; λ = 1 is 1.3e+04× / 2.6e+04×; λ = 1½ is 2.5e+05× / 6.4e+05× (§3.5) |
| **Item 1 / Finding 3** — λ ≥ 1.0 can flip `_energy_clean_pick`'s "evidently lossless" classification | **VERIFIED** | 4 lossless cells × 2 polarisations, worst closure over degrees 16/18/20 against the 1e-6 classifier: λ = ½ stays *ok* on **8 of 8**; λ = ¾ flips 4 of 8; **14 of 16 λ ≥ 1.0 rows flip** (§3.6) |
| **Item 1 / Finding 3** — "the TM win is robust but modest and fixture-shaped" (λ = 0 beats λ = ½ on all seven cells, 5.3×–86×) | **VERIFIED-WITH-NOTES — far MORE fixture-shaped than reported** | Same protocol (degree 32 against a degree-140 self-reference at the same λ) on six cells of mine: λ = 0 wins **85.6×** (lossless 3.48), **12.5×** (dielectric), **3.9×** (sub-wavelength Ag) — but only **1.1×** and **1.2×** on my deep Ag cells and it **LOSES (0.6×)** on the shallow Ag cell.  On genuinely metallic TM fixtures the "win" is inside the noise or negative (§3.7).  This strengthens the WP's negative |
| **Item 1 / the gate** — no λ delivers the audit's "exponential convergence"; the local rate stays algebraic | **VERIFIED, with a sharper demonstration** | Read out to **degree 100** (the WP stopped at 44).  λ = ½ settles at **2.92 / 3.05**; λ = ¼ at 3.15 / 3.02; λ = 1 at 2.81.  λ = 0's rate **DECAYS to 0.23** on the lossless cell as its error plateaus at 7.4e-08 — the exact signature of a bounded error cancellation, not a better rate.  An exponential rate grows without bound in the degree; none does (§3.2) |
| **Item 1 / the oracle** | **VERIFIED-WITH-NOTES** | I rebuilt the extrapolated RCWA oracle on four cells.  Its floor is fixture-dependent by five decades: **3.4e-09** (lossless `n = 3.48`), 2.9e-07 (dielectric), 1.9e-08 (TE) — but **~1.7e-04** on a deep lossy Ag TM cell, where the RCWA `richardson` limit and the PMM's own limit **bracket** the answer from opposite sides and do not meet.  The WP's 2.62e-06 is right for *its* shallow Au cell; the gate is only readable to 1e-06 on cells where the oracle is sound, and that should be said in the deferred hp design's gate (§3.1, Follow-up 1) |
| **Item 2** — the moved code is BIT-IDENTICAL, cached vs uncached and before vs after | **VERIFIED** | **1 080 comparisons, 0 differ, worst \|A−B\| = 0.000e+00**, archive-to-archive, on cells the engineer did not use: 3-strip separable x and y, a **3 × 2** crossed cell, complex off-diagonal tiles, reciprocal out-of-plane tiles, φ = 41° and φ = 115°, circular truncation, slanted layers at normal *and* oblique, `symmetry='auto'` block-eig on a mirror-symmetric 3 × 3 crossed cell, degree 9 and 13, `elements_per_strip=2` + `grade=True`, `period_x ≠ period_y`, `laurent`/`li`/`fff_nv`/`auto`, and the `return_ops` fold hand-offs — plus the fifth call site the report does not enumerate, `PMMStack._solve_conical`'s tensor-segment path (§3.8) |
| **Item 2** — the `formulation` key change closes a hole it would otherwise have opened | **VERIFIED** | Seven mutable solver attributes mutated after a solve, each reproducing a freshly built object **bit for bit** (`max\|d\| = 0.000e+00`); the `li`/`laurent` non-vacuity spread is **9.6e-03**.  Deleting `self.formulation` from `_geom_key` on an isolated tree turns `::test_b6_the_geom_key_splits_on_the_formulation` red (§3.9, §3.10) |
| **Item 2** — refusal degrades to a rebuild and never to a wrong answer | **VERIFIED** | `cache_max_bytes` swept **across one entry's exact size** (1 / nb/2 / nb−1 / nb / nb+1 / 4nb, nb = 412 864 B): refused, refused, refused, stored, stored, stored — and the answer is `np.array_equal` in all six (§3.9c) |
| **Item 2** — the cached arrays are handed out read-only | **VERIFIED** | 28 / 25 / 32 arrays reachable from the `_geom_cache` and `_eig_cache` entries of a crossed / separable / out-of-plane stack; **0 accepted a write** (§3.9d) |
| **Item 2** — `_symmetric_layer_specs`' `tops=None` path cannot be reached by a tensor layer | **VERIFIED** | Instrumented: the scalar fold fires (non-vacuity), and **no** tensor layer reaches it across five tensor shapes at normal incidence with `symmetry='auto'`.  The gate at `stack2d.py:1602` is load-bearing, not decorative — see §3.9e for what would happen without it |
| **Item 2** — "source-free assemblies per 9-point sweep 9 → 1, 0.474 → 0.009 s, 4.2 % of that solve" | **VERIFIED** | Two independent instruments.  (i) `twod._axis_projection` calls per 5-point sweep, base → after: crossed **10 → 2**, separable **5 → 1** — a function WP-B6 did not touch and no test spies on.  (ii) At the WP's own size (crossed, degree 11 / `n_orders` 7, 9 points, min of 3): base in-call assembly **0.467 s** of a **10.460 s** sweep = **4.5 %**; after, 0.013 s in-call + 0.043 s for the single cached build (§3.11) |
| **Item 2** — retention grows ~48× per tensor entry | **VERIFIED in kind and mechanism, smaller at my truncations** | Largest `_geom_cache` entry, base → after: **0.10 → 0.39 MiB** (deg 9 / `n_orders` 3, ×3.9), **0.15 → 1.94 MiB** (deg 11 / 5, ×12.9), **0.09 → 1.65 MiB** (separable deg 11 / 5, ×18).  The ratio grows with `Nf²`, so the WP's ×48 at `n_orders` 7 is consistent (§3.11) |
| **No PMM default moved** | **VERIFIED** | `PMM2DStackHybrid` still `formulation='li'`, `truncation='rectangular'`, `symmetry='auto'`, `cascade='fast'`, `degree=11`, `n_orders=11`; `pmm_jones_2d` still `formulation='laurent'`; `pmm_efficiency_1d` still `degree=16`, `grade=True`, `stabilize=True`.  The whole 1-D surface is bit-identical base vs after (46 of 46 arrays) |
| **`_core.py`'s edit really was documentation-only** | **VERIFIED** | `scripts/record_history_fingerprints.py --check` reports **OK** on all seven `lumenairy.elements.pmm.*` documents, `_core.md` included, and `_core.md` was NOT re-recorded in `3af33dff` — so the AST-and-token pin is what proves it, exactly as the report says |
| **The 44 pins bite** | **NOT FIXED — two of them did not; both closed here, in the test file** | Eight source mutations applied to an isolated copy of the tree.  Six turn the right test red.  **Two are INVISIBLE to all 44**: dropping `ops=tops` at `stack2d.py:1127` (the exact G10(d) regression — 44 passed) and adding `self.truncation` to `_geom_key` (44 passed).  Two tests added; the same mutants now turn 7 and 1 ids red (§4) |
| **The census** | **VERIFIED** | `tests/unit/test_ci_kernel_consistency.py` — **7 passed**, 2.59 s.  No PMM decision moved |
| **The known-red T3-1** | **VERIFIED** | Run by direct import on both archives: the screened tuple is **identical character for character**, and so is every other character of the message.  Only the wall-clock line differs (12.4 s vs 12.1 s) (§5) |

**Summary.** The physics and the engineering of WP-B6 hold up under an independent attack on a
different fixture family, and item 1's negative is *more* robust than the report claims (the λ = 0
"TM win" evaporates or reverses on my metallic cells, and its rate decays to 0.23 by degree 100).
Item 2's bit-identity survives 1 080 comparisons on cells the engineer never used, plus a fifth
call site the report does not enumerate.  The one thing that did **not** hold is the *test* side:
the headline "9 → 1 assemblies" contract was measured through a counter that cannot see the
rebuild it forbids, and the "one entry serves both truncations" contract never shared an entry
between two truncations.  Both are closed by addition, in my ownership; no library code was
changed and no changelog is therefore due.

---

## 2. What I did NOT take on trust, and where it came out differently

1. **The oracle floor is not a constant of the method.**  The WP reports 2.62e-06 for its Au cell
   and 9.95e-07 for its lossless twin.  On a *deep* lossy Ag cell (0.28 µm, 22°) my extrapolated
   RCWA oracle is only good to **~1.7e-04**: `rcwa_extrapolate(richardson)` on `n_orders`
   101…1201 returns 0.459138757901 while the raw 1201 sample is 0.459089782608 *below* it and the
   PMM's own high-degree limit is 0.4592503 *above* it.  The two methods bracket and do not meet.
   This does not touch any WP-B6 claim — the WP's own gate fixture is shallow — but it does mean
   the deferred hp design's gate ("the same Au/air TM fixture and extrapolated RCWA oracle this
   report builds ... reusable as-is") is only reusable at that geometry, and the reuse instruction
   should say so.
2. **"The TM win ... λ = 0 beats λ = 0.5 on all seven [cells]".**  Not on mine.  On a *shallow*
   Ag/air TM cell λ = 0 is **1.7× worse** than λ = ½ (1.96e-05 against 1.17e-05 at degree 32), and
   on my two deep Ag cells it wins by only 1.1× and 1.2×.  The large wins (12×–86×) are all on
   lossless or dielectric cells.  The WP's word "fixture-shaped" is right; the size of the effect
   is much larger than "modest".
3. **The `_gll_nodes_weights` docstring's "3.7e-01 .. 1.5e+00 at every other `lambda`"** is a range
   over the λ set that was measured ({0, ¼, ¾, 1, 1½}), not over the family.  I measure **1.98 …
   2.20 at λ = 2**, outside the quoted range.  The claim it supports — eleven decades — is
   unaffected, and I have left the docstring alone; noted so a later reader does not treat the
   interval as a bound.
4. **The report's §2.2 "Residual risk" paragraph** says `_symmetric_layer_specs` writes `tops=None`
   and "a tensor layer can never read that entry because `L["kind"]` is in the key".  True, and I
   verified it by instrumentation — but the reason it matters is stronger than the report says: if
   a tensor layer *did* reach `_symmetric_layer_specs`, a tensor tile whose nine slots are all
   equal passes that function's `np.all(|tile − tile.flat[0]| < 1e-12)` uniform test and would be
   cascaded as an **isotropic** film of that value.  The gate at `stack2d.py:1602` is the only
   thing between that cell and a silently wrong answer, and it deserves the report's emphasis
   rather than a "if it somehow did, `ops=None` rebuilds — correct, just slower".

---

## 3. Per item

### 3.1 The instruments I built

**The family.**  Nodes for parameter λ are ±1 plus the roots of `d/dx C_N^(λ) ∝ C_{N−1}^(λ+1) ∝
P_{N−1}^(a,a)` with `a = λ + ½`.  I built them **twice**: Golub–Welsch on the monic symmetric-Jacobi
recurrence, which I re-derived from the monic Gegenbauer recurrence
`β_n = n(n+2L−1)/(4(n+L)(n+L−1))` at `L = a + ½` (giving `β_n = n(n+2a)/((2n+2a+1)(2n+2a−1))`), and
`scipy.special.roots_jacobi`.  Agreement ≤ **7.8e-16** over degrees 8…44.  Weights `w_i = ∫ l_i dx`
also twice: Gauss–Legendre quadrature of the Lagrange basis, and the Legendre-Vandermonde moment
solve — ≤ **2.0e-15**.  At λ = ½ this reproduces the **shipped** `_gll_nodes_weights` to
**3.0e-15 (nodes) / 2.0e-15 (weights)**, `Σw − 2 ≤ 8.9e-16`, and every weight is positive for
λ ∈ {0 … 2} × degree ∈ {8 … 44} (min 5.2e-04).  My differentiation matrix is the product rule, not
the barycentric form the library uses; they agree to ≤ **4.5e-13** at degrees 8/24/44.

**The exact-integration control.**  A drop-in `_core._build_sem` that evaluates every element
integral exactly (Gauss–Legendre, `n + 6` points): `S0 = J·∫l_i l_j`, `L = ∫l_i' l_j'/J`,
`C = ∫l_i l_j'` (J-free), same global numbering, same periodic wrap, same return dict.  Nothing in
the repository was edited to measure any of this; both are installed in process.

**The oracle.**  `rcwa_efficiency_1d` + `rcwa_extrapolate` — **called, not edited** — on four cells.

| fixture | `n_orders` ladder | `R0(∞)` | spread over all 4 estimators | spread over the 3 *algebraic* ones |
|---|---|---|---|---|
| **F1** Ag/air TM (0.130+3.19j, P 0.45 µm, d 0.28 µm, duty 0.4, λ 0.532 µm, 22°, n_sub 1.46) | 101…1201 | 0.459138757901 | 1.92e-04 | 5.84e-05 |
| **F3** lossless `n = 3.48` twin (same geometry) | 101…801 | 0.108292213762 | 2.73e-07 | **3.36e-09** |
| **F2** dielectric `n = 2.4` (P 0.9 µm, d 0.32 µm, duty 0.65, λ 0.85 µm, 7°) TM | 101…601 | 0.153298534758 | 2.89e-07 | 2.89e-07 |
| F1 TE | 101…601 | 0.024210222638 | 1.88e-08 | 1.88e-08 |
| F2 TE | 51…401 | 0.007598858254 | 7.75e-08 | 7.59e-08 |

(Estimators: `rcwa_extrapolate(last 3, richardson)`, `(last 4, richardson)`, `(last 3, shanks)`,
and the audit's own least-squares `R_inf + c/n` on the last four.  `shanks` models a *geometric*
tail and is the outlier on every row; the algebraic-only spread is the honest floor.)

Every fixture differs from the auditor's Au/air cell in metal, period, depth, duty, wavelength,
angle **and** substrate.  The PMM's own degree-160 answer agrees with the oracle to **1.6e-07**
(F3), **7.6e-09** (F2 TM), **1.9e-11** (F1 TE) and **7.3e-13** (F2 TE) — so the oracle is validated
as a limit on four of the five, and on F1 TM it is not (§2.1), which is why the F1 TM ladder in
§3.7 is read against a PMM self-reference instead.

### 3.2 Finding 1 — the basis swap is a no-op, and λ = 0's "rate" decays

Max spread over λ ∈ {0, ¼, ½, ¾, 1, 1½} of `concat(R, T)`:

| fixture | degree | EXACT integration | LUMPED (shipped assembly) | ratio |
|---|---|---|---|---|
| F1 Ag/air TM | 16 / 24 / 32 | 1.10e-13 / 2.14e-13 / 3.35e-13 | 7.35e-04 / 3.40e-04 / 1.97e-04 | 6.7e+09 / 1.6e+09 / 5.9e+08 |
| F3 lossless TM | 16 / 24 / 32 | 5.21e-14 / 1.56e-13 / 3.79e-13 | 9.54e-04 / 2.63e-04 / 1.06e-04 | 1.8e+10 / 1.7e+09 / 2.8e+08 |
| F2 dielectric TM | 16 / 24 / 32 | 5.20e-14 / 1.04e-13 / 2.10e-13 | 2.55e-03 / 6.80e-04 / 2.77e-04 | 4.9e+10 / 6.5e+09 / 1.3e+09 |
| F1 Ag/air TE | 16 / 24 / 32 | 6.57e-14 / 1.25e-13 / 2.97e-13 | 6.55e-05 / 1.83e-05 / 7.47e-06 | 1.0e+09 / 1.5e+08 / 2.5e+07 |
| F2 dielectric TE | 16 / 24 / 32 | 1.11e-13 / 2.87e-13 / 8.36e-13 | 7.98e-03 / 1.99e-03 / 7.92e-04 | 7.2e+10 / 6.9e+09 / 9.5e+08 |

The CONSISTENT error ladders print the **same digits** for all six λ on every fixture — e.g. F3,
degrees 8…44: `5.57e-04 2.11e-04 1.03e-04 3.64e-05 1.71e-05 7.34e-06` six times over.  **Confirmed:
a nodal-basis change on the fixed C0 piecewise-`P_N` space cannot move the answer.**

The high-degree tail, against the oracle (this is further than the WP went, and it is where the
audit's "exponential" claim finally dies):

| F3 lossless TM | 24 | 32 | 44 | 60 | 80 | 100 | local rate |
|---|---|---|---|---|---|---|---|
| λ = 0.00 | 7.60e-07 | 2.86e-07 | 1.33e-07 | 9.01e-08 | 7.75e-08 | **7.37e-08** | 3.39 / 2.41 / 1.25 / 0.52 / **0.23** |
| λ = 0.25 | 2.01e-05 | 9.14e-06 | 3.79e-06 | 1.58e-06 | 6.81e-07 | 3.37e-07 | 2.74 / 2.77 / 2.82 / 2.93 / 3.15 |
| **λ = 0.50 (shipped)** | 4.08e-05 | 1.87e-05 | 7.82e-06 | 3.32e-06 | 1.47e-06 | 7.69e-07 | 2.72 / 2.73 / 2.76 / 2.82 / **2.92** |
| λ = 1.00 | 7.24e-05 | 3.37e-05 | 1.43e-05 | 6.16e-06 | 2.79e-06 | 1.49e-06 | 2.66 / 2.69 / 2.72 / 2.76 / 2.81 |

| F2 dielectric TM | 24 | 32 | 44 | 60 | 80 | 100 | local rate |
|---|---|---|---|---|---|---|---|
| λ = 0.00 | 2.71e-07 | 8.29e-08 | 2.51e-08 | 8.34e-09 | 3.04e-09 | 1.35e-09 | 4.11 / 3.75 / 3.55 / 3.51 / 3.65 |
| λ = 0.25 | 1.05e-06 | 4.43e-07 | 1.69e-07 | 6.55e-08 | 2.73e-08 | 1.39e-08 | 2.98 / 3.03 / 3.05 / 3.04 / 3.02 |
| **λ = 0.50 (shipped)** | 2.55e-06 | 1.04e-06 | 3.89e-07 | 1.49e-07 | 6.19e-08 | 3.13e-08 | 3.11 / 3.10 / 3.08 / 3.07 / 3.05 |
| λ = 1.00 | 6.29e-06 | 1.41e-06 | 1.82e-07 | 2.35e-08 | 3.65e-08 | 2.67e-08 | 5.20 / 6.43 / 6.60 / −1.53 / 1.40 (a sign crossing, not a rate) |

Nothing leaves the algebraic regime.  An exponential rate `e ~ exp(−cN)` has a local rate `∝ N`
that grows without bound; every arm here is flat or decaying.  λ = 0 on F3 is the interesting one:
its early rate (3.4) looks like the audit's promise and then **collapses to 0.23** as its error
plateaus at 7.4e-08 — which is exactly what a fixed error cancellation against the corner looks
like, and exactly what WP-B6's Finding 2 predicts.  Read only to degree 44, as the WP's own table
is, that arm still looks like a rate; read to 100 it is unambiguous.

### 3.3 Finding 2 — summation-by-parts

`max|M D + (M D)^T − B|`, `B = diag(−1, 0, …, 0, +1)`, with `M = diag(w)` the lumped nodal mass,
my weights and my `D`:

| degree | λ = 0.00 | 0.25 | **0.50** | 0.75 | 1.00 | 1.50 | 2.00 | exact, any λ |
|---|---|---|---|---|---|---|---|---|
| 8 | 9.224e-01 | 4.198e-01 | **1.414e-14** | 3.704e-01 | 7.112e-01 | 1.351e+00 | 1.980e+00 | 6.72e-15 |
| 16 | 9.630e-01 | 4.407e-01 | **9.640e-14** | 3.922e-01 | 7.563e-01 | 1.449e+00 | 2.145e+00 | 3.59e-14 |
| 24 | 9.704e-01 | 4.446e-01 | **3.608e-13** | 3.966e-01 | 7.655e-01 | 1.470e+00 | 2.182e+00 | 2.84e-13 |
| 32 | 9.730e-01 | 4.460e-01 | **2.680e-13** | 3.982e-01 | 7.689e-01 | 1.478e+00 | 2.196e+00 | 1.57e-13 |
| 44 | 9.746e-01 | 4.469e-01 | **1.324e-12** | 3.992e-01 | 7.710e-01 | 1.483e+00 | 2.204e+00 | 1.32e-12 |

Eleven to thirteen decades.  Exact integration restores the identity at every λ — so the identity
belongs to the **rule**, not to the nodes, which is the whole of Finding 2.

### 3.4 The consequence on the solver — lossless oblique closure

`|Σ R + Σ T − 1|` at degree 24:

| cell | mass | λ = 0.00 | 0.25 | **0.50** | 0.75 | 1.00 | 1.50 |
|---|---|---|---|---|---|---|---|
| F3 lossless 3.48 TM | LUMPED | 5.442e-08 | 3.886e-08 | **1.059e-13** | 2.335e-08 | 4.100e-08 | 3.430e-07 |
| F3 lossless 3.48 TM | CONSISTENT | 1.621e-13 | 8.749e-14 | 4.663e-14 | 5.285e-14 | 3.109e-14 | 7.105e-15 |
| F2 dielectric TM | LUMPED | 2.120e-08 | 1.241e-08 | **6.373e-14** | 2.031e-07 | 2.156e-06 | 6.065e-05 |
| F2 dielectric TM | CONSISTENT | 2.931e-14 | 4.796e-14 | 4.874e-14 | 2.198e-14 | 1.110e-15 | 1.640e-13 |
| F2 dielectric TE | LUMPED | 6.227e-08 | 3.916e-08 | **1.499e-13** | 3.037e-07 | 3.365e-06 | 8.873e-05 |
| F2 dielectric TE | CONSISTENT | 5.340e-14 | 1.937e-13 | 1.252e-13 | 2.998e-14 | 2.220e-16 | 4.108e-14 |

The CONSISTENT rows are the control and they are flat: the defect is the quadrature.

### 3.5 The TE price

Reference: PMM at degree 120, λ = ½ — validated against my RCWA TE oracle to **1.9e-11** (F1) and
**3.9e-12** (F2), so it is a limit and not a self-flattering choice.  `|R0 − R0(ref)|`:

| F1 Ag/air TE | 8 | 12 | 16 | 20 | 24 | vs λ = ½ at degree 24 |
|---|---|---|---|---|---|---|
| λ = 0.00 | 5.15e-05 | 2.17e-06 | 3.42e-07 | 8.57e-08 | 2.81e-08 | **489×** worse |
| λ = 0.25 | 2.28e-05 | 9.77e-07 | 1.71e-07 | 4.64e-08 | 1.62e-08 | 282× |
| **λ = 0.50** | 2.32e-07 | 1.08e-08 | 1.27e-09 | 2.34e-10 | **5.74e-11** | — |
| λ = 0.75 | 1.52e-06 | 2.19e-06 | 5.53e-07 | 1.91e-07 | 8.12e-08 | 1 415× |
| λ = 1.00 | 4.51e-05 | 1.35e-05 | 3.99e-06 | 1.57e-06 | 7.36e-07 | 12 824× |
| λ = 1.50 | 3.38e-04 | 1.19e-04 | 4.91e-05 | 2.48e-05 | 1.42e-05 | 247 737× |

F2 dielectric TE is the same shape: λ = ½ reaches **2.71e-10** at degree 24 against 5.71e-08
(λ = 0, 211×), 6.96e-06 (λ = 1, 25 657×) and 1.73e-04 (λ = 1½, 636 305×).  TE is the polarisation
the library documents as spectrally convergent with no floor; every member of this family takes
that away.

### 3.6 `stabilize`'s lossless classification

`_energy_clean_pick` calls a structure "evidently lossless" at `|Σ R + Σ T − 1| < 1e-6`.  Worst
closure over degrees 16 / 18 / 20, four lossless cells × two polarisations (`*` = flips to the
lossy branch):

| cell | pol | λ = 0.00 | 0.25 | **0.50** | 0.75 | 1.00 | 1.50 |
|---|---|---|---|---|---|---|---|
| 3.48 twin | te | 2.31e-07 | 1.21e-07 | **1.03e-13** | 5.05e-07 | 4.03e-06 `*` | 5.96e-05 `*` |
| 3.48 twin | tm | 4.59e-07 | 3.14e-07 | **4.06e-14** | 2.79e-07 | 3.18e-07 | 1.29e-06 `*` |
| 3.48 duty 0.7 | te | 1.20e-07 | 4.09e-08 | **6.77e-14** | 4.57e-08 | 8.81e-08 | 1.18e-05 `*` |
| 3.48 duty 0.7 | tm | 3.25e-07 | 2.13e-07 | **2.68e-14** | 6.48e-08 | 1.85e-06 `*` | 3.40e-05 `*` |
| dielectric 2.4 | te | 7.87e-07 | 4.12e-07 | **2.07e-13** | 2.04e-06 `*` | 1.82e-05 `*` | 3.14e-04 `*` |
| dielectric 2.4 | tm | 1.35e-07 | 7.36e-08 | **6.00e-14** | 1.58e-06 `*` | 1.32e-05 `*` | 2.42e-04 `*` |
| dielectric 2.4 normal | te | 9.00e-07 | 5.30e-07 | **1.33e-14** | 2.90e-06 `*` | 2.56e-05 `*` | 4.31e-04 `*` |
| dielectric 2.4 normal | tm | 3.17e-07 | 1.27e-07 | **5.42e-14** | 2.64e-06 `*` | 2.08e-05 `*` | 3.41e-04 `*` |

λ = ½ stays *ok* on 8 of 8; λ = ¾ flips 4 of 8; **14 of 16** λ ≥ 1.0 rows flip.  A knob that moves
a classification is a different class of risk from one that moves digits — the WP's wording, and
my numbers agree with it.

### 3.7 The TM "win" — more fixture-shaped than reported

Degree 32 against a degree-140 self-reference **at the same λ** (the WP's own protocol):

| cell | λ = 0 | λ = 0.5 | λ = 0 is |
|---|---|---|---|
| Ag deep 0.28 µm / duty 0.4 / 22° | 6.516e-05 | 6.914e-05 | 1.1× better |
| **Ag shallow 0.09 µm / duty 0.4 / 22°** | 1.958e-05 | 1.172e-05 | **0.6× — WORSE** |
| Ag 0.28 µm / duty 0.7 / normal | 6.388e-05 | 7.352e-05 | 1.2× better |
| lossless 3.48 twin | 2.151e-07 | 1.842e-05 | 85.6× better |
| dielectric 2.4 | 8.264e-08 | 1.030e-06 | 12.5× better |
| Ag sub-wavelength P 0.3 µm | 1.190e-04 | 4.696e-04 | 3.9× better |

And the metal ladder itself, read against a PMM self-reference (degrees 120/160/200/260 →
0.459266165 / 0.459258640 / 0.459254402 / 0.459250782, extrapolated 0.4592503, residual ~3e-06):

| F1 Ag/air TM | 8 | 12 | 16 | 24 | 32 | 44 | 60 | local rate |
|---|---|---|---|---|---|---|---|---|
| λ = 0.00 | 5.05e-04 | 2.72e-04 | 1.80e-04 | 1.02e-04 | 6.66e-05 | 4.03e-05 | 2.34e-05 | 1.52 / 1.43 / 1.42 / 1.47 / 1.58 / 1.76 |
| **λ = 0.50** | 1.32e-03 | 2.70e-04 | 2.18e-05 | 7.74e-05 | 8.07e-05 | 6.51e-05 | 4.66e-05 | 3.91 / 8.74 / −3.13 / −0.14 / 0.67 / 1.08 |

On the actual metal, λ = 0 buys a factor of 2 at degree 60 and the rate is ~1.5–1.8 for both.  The
5×–25× the WP quotes is a property of its lossless and dielectric cells.  **This makes the WP's
"nothing ships" verdict safer, not shakier**, and it is worth having in the record for whoever
re-opens alternative (c).

### 3.8 Item 2 — byte identity, archive to archive

Two captures, one per archive, each in its own child process with `cwd` + `PYTHONPATH` = that tree
and `lumenairy.__file__` asserted; pytest never involved.

| group | what | arrays | identical |
|---|---|---|---|
| A | the seven projected operators + the four out-of-plane blocks + the block-eig gauge + the returned modal set, captured at the hand-off to `_layer_eigenmodes_tensor`, over **22 cells** and 4 `return_ops` fold hand-offs | 283 | **283** |
| B | end to end through `PMM2DStackHybrid`: 12 stacks × wavelength and angle sweeps, plus `solve_vs_wavelength(jones=True)` | 108 | **108** |
| C | `pmm_jones_2d`, 12 calls (`laurent`/`li`/`fff_nv`/`auto`, rectangular and circular, in-plane and out-of-plane, slanted at normal and oblique, uniform) | 48 | **48** |
| D | the 1-D PMM surface (te/tm × degree 8/17/26 × angle 0 / 0.21 / 0.55 rad, `stabilize=True`, graded `elements_per_region=3`, `pmm_jones_1d`, `pmm_jones_1d_conical_tensor`) | 46 | **46** |
| E | after tree: **cached path vs uncached path**, same 283 captures | 283 | **283** |
| F | **base uncached vs after cached** (the transitive claim) | 283 | **283** |
| G | `PMMStack._solve_conical`'s tensor-segment path (`stack.py:2768`) — 2/3-segment tensor, out-of-plane, uniform, φ = 0 and φ = 0.61 | 29 | **29** (28 bit-identical + 1 identical `NotImplementedError` refusal) |

**1 080 comparisons, 0 differ, worst `|A − B| = 0.000e+00`.**  The 22 cells are deliberately not
the engineer's: three-strip separable-x and separable-y cells, a **3 × 2** crossed cell, tiles with
complex `e_xy`, reciprocal out-of-plane tiles, `period_x ≠ period_y` (0.53 / 0.37 µm), φ = 41° and
φ = 115°, circular truncation at oblique **and** normal incidence, slant `(0.31, −0.17)` at both,
`block_eig` on a mirror-symmetric 3 × 3 crossed cell, degree 13, `elements_per_strip = 2` with
`grade=True`.  The `return_ops` refusals for out-of-plane and slanted cells are preserved on both
trees.

### 3.9 Item 2 — attacks on the cache itself

**(a) mutate a solver attribute after a solve.**  Seven attributes, each mutated on a stack whose
caches are already warm, against a freshly built object at the new value:

| attribute | change | mutated vs fresh | the two settings genuinely differ by |
|---|---|---|---|
| `formulation` | `'li'` → `'laurent'` | **0.000e+00** | 9.598e-03 |
| `degree` | 9 → 11 | **0.000e+00** | 5.961e-05 |
| `n_orders` | 3 → 4 | **0.000e+00** | different order set |
| `grade` | False → True | **0.000e+00** | 6.679e-15 |
| `truncation` | rect → circular | **0.000e+00** | different order set |
| `symmetry` | `'auto'` → False | **0.000e+00** | 0.0 (see below) |
| `cascade` | `'fast'` → `'monolithic'` | **0.000e+00** | 0.0 (see below) |

`period_x` assignment after `add_layer` is refused (`ValueError`), as documented.  Two of the seven
non-vacuity arms are **vacuous on this fixture** and I say so rather than counting them: `cascade`
`'fast'` vs `'monolithic'` is bit-identical when no two adjacent layers share a modal basis (by
construction), and `symmetry` is only load-bearing for an out-of-plane or slanted tensor layer at
normal incidence — I re-ran it there and it is **also** bit-identical (0.000e+00), i.e. the
parity-sign block reduction is a different computation of the same modal set on this cell.  The
mutation contract still holds for both; the *difference* they are supposed to cause is simply not
observable here.

**(b) two layers that differ only in one tile value.**  A stack `[T, T + 1e-9 in one component]`:
the two layers mint **2** distinct `_geom_cache` entries, the cached answer is bit-identical to the
priced-out (always-rebuild) answer, and it differs from `[T, T]` by 2.795e-11 — a colliding key
would have made those equal.

**(c) `cache_max_bytes` across the entry boundary.**  One tensor entry is **412 864 B** (0.39 MiB)
at degree 9 / `n_orders` 3.  Budgets 1 / 206 432 / 412 863 / 412 864 / 412 865 / 1 651 456 B give
`refused=1, refused=1, refused=1, entries=1, entries=1, entries=1` — and the solve's four returns
are `np.array_equal` to the unpriced reference in **all six**.  Refusal costs a rebuild and
nothing else.

**(d) the frozen arrays.**  Every ndarray reachable from every `_geom_cache` *and* `_eig_cache`
value (walking tuples, lists, dicts and object `__dict__`s) was offered a write: **28 of 28**
(crossed), **25 of 25** (separable), **32 of 32** (out-of-plane) raised.  The ops **dict** itself is
rebindable — but so is the scalar `lops` dict it was modelled on, and no in-tree caller mutates
either, so this is the same hardening level, not a new hole.

**(e) `_symmetric_layer_specs`.**  Instrumented across five tensor shapes (crossed 3 × 2, separable
x3, out-of-plane, isotropic-valued tensor, anisotropic uniform tensor) at normal incidence with
`symmetry='auto'`: **zero** tensor layers reach it, while a symmetric *scalar* stack does (so the
probe is not vacuous).  The gate is `all(L["kind"] != "tensor" ...)` at `stack2d.py:1602`.  What it
prevents, stated because the report under-sells it: a tensor tile whose nine slots are all equal
satisfies that function's `np.all(|tile − tile.flat[0]| < 1e-12)` uniform test, so without the gate
it would be cascaded as an **isotropic** film of that value — a silently wrong answer, not a slow
one.

**(f) the deterministic saving on my cells.**  Four branches × two sweep axes × 7 points, warm vs
`cache_max_bytes=1`:

| branch | wavelength | angle |
|---|---|---|
| crossed 3 × 2 | **7 → 1** | **7 → 1** |
| crossed 3 × 2, circular truncation | **7 → 1** | **7 → 1** |
| separable x3 | **7 → 1** | **7 → 1** |
| out-of-plane separable | **7 → 1** | **7 → 1** |

`_tensor_layer_modes` stays at 7 per sweep in every row (the eig does depend on the source), and
every return of every point is `np.array_equal` between the two arms.

### 3.10 Item 2 — the `formulation` key

The report calls this "one key change ... against the design's *no key change*", and it is right to.
`_scalar_projected_ops` returns every rule's operator side by side, so `lops` is
formulation-independent; the tensor `EZZ` is `inv([[1/e_zz]])` under `'li'` and `[[e_zz]]`
otherwise, so `tops` is not.  I confirmed the two rules genuinely disagree on my crossed cell
(9.598e-03) and that mutating `st.formulation` after a warm solve reproduces a fresh object bit for
bit.  Deleting the key element on an isolated tree turns exactly one test red (§4).

Because `_geom_key` carries the **raw** attribute while the cached value depends only on
`ez_rule = 'li' if formulation == 'li' else 'laurent'`, the key can only over-split, never
under-split — the conservative direction, and the report says so.

### 3.11 Item 2 — what it costs and what it buys, measured on both archives

**An instrument the WP did not use and no test spies on.**  `twod._axis_projection` is called once
per patterned axis per source-free assembly and was not touched by this diff.  Calls per 5-point
wavelength sweep, base → after: **crossed 10 → 2**, **separable 5 → 1**.  That is the 9 → 1 claim,
confirmed from outside the changed code.

**The share, at the WP's own size** (crossed, degree 11 / `n_orders` 7, 9-point sweep, min of 3,
same instrumentation on both trees):

| | base `3af33dff^` | after `3af33dff` |
|---|---|---|
| sweep wall | 10.460 s | 10.098 s |
| `_tensor_layer_modes` total | 3.040 s / 9 calls | 2.680 s / 9 calls |
| … of which the eig | 2.573 s | 2.666 s |
| … so the **in-call assembly** | **0.467 s** | **0.013 s** |
| the single cached assembly | — | 0.043 s |
| **assembly share of the solve** | **4.5 %** | 0.6 % |

The WP reports 0.474 → 0.009 s and "4.2 % of the whole solve" on its cell.  Mine is a different
crossed cell at the same size: **0.467 s and 4.5 %**.  The claim is reproduced.

**The retention**, largest `_geom_cache` entry, base → after:

| fixture | base | after | × |
|---|---|---|---|
| crossed 3 × 2, degree 9, `n_orders` 3 | 0.10 MiB | 0.39 MiB | 3.9 |
| crossed 3 × 2, degree 11, `n_orders` 5 | 0.15 MiB | 1.94 MiB | 12.9 |
| separable x3, degree 11, `n_orders` 5 | 0.09 MiB | 1.65 MiB | 18.3 |

The stored operators are ~11 dense `Nf × Nf` complex128 blocks while the axis records are
`O(n_glob²)` per axis, so the ratio grows with `Nf²`; the WP's ×48 at `n_orders` 7 sits on the same
curve.  `LayerCache` prices it and refusal degrades to a rebuild (§3.9c).

---

## 4. The pins — two did not bite, both closed

The 44 tests use **derived** envelopes, not per-build numbers, and there is no wall-clock assertion
anywhere: the SBP bar is `1e3 · n · eps` (the `O(n)` roundoff floor) with the upper bar `1e6 ×`
that; the invariance bar is `1e-4 ×` the LUMPED family's own spread **measured on the running
build**; the closure bar is `1e3 ×` the measured GLL closure; the tensor gates are `np.array_equal`
and integer counts.  The three absolute constants (1e-9, 1e-7, 1e-6) are two-sided premise guards,
not the gate.  The identity tests do re-derive on the running build — swapping
`_gll_nodes_weights` for the Chebyshev–Lobatto rule turns both of them red.

I applied eight single mutations to an isolated copy of the `3af33dff` tree (the repository itself
was never edited) and ran the file against each:

| mutation | before (44 pins) | after my two additions (51 pins) |
|---|---|---|
| `ops=tops` → `ops=None` at `stack2d.py:1127` — **the exact G10(d) regression** | **44 passed — INVISIBLE** | **7 failed** / 2 functions |
| `self.truncation` appended to `_geom_key` | **44 passed — INVISIBLE** | **1 failed** |
| separable-x branch labelled `kind = "uniform"` | 4 failed / `::test_b6_the_k0_free_split_is_the_algebra_it_claims` | 4 failed |
| `self.formulation` deleted from `_geom_key` | 1 failed / `::test_b6_the_geom_key_splits_on_the_formulation` | 1 failed |
| `GxF = (Gx0F + (k0·kx0)·IpxF)/k0` (algebraically identical, not bit identical) | 1 failed / `::test_b6_a_supplied_ops_is_actually_consumed` | 1 failed |
| tensor slot cached WITHOUT `_freeze_cached` | 1 failed / `::test_b6_the_cached_tensor_operators_are_handed_out_read_only` | 1 failed |
| `_gll_nodes_weights` → Chebyshev–Lobatto | 5 failed / `::test_b6_only_the_gll_rule_satisfies_summation_by_parts`, `::test_b6_a_non_gll_nodal_rule_loses_the_exact_energy_identity` | 5 failed |
| (control) no mutation | 44 passed | 51 passed |

**Why the first two were invisible.**

1. `_count_builds` spies on `stack2d._tensor_projected_ops`.  `stack2d` and `twod_jones` hold
   **separate module bindings** for that function, and the rebuild `_tensor_layer_modes` performs
   when it is handed no `ops` resolves `twod_jones`'s own global.  So a stack that stores a cached
   build and then never passes it still shows **one** assembly on that counter — warm 1, cold 5,
   assertion satisfied — while doing all nine rebuilds.  The headline claim of item 2 had no pin
   that could fail.
2. `::test_b6_the_cached_entry_serves_both_truncations` is parametrised over `truncation` but
   constructs a **fresh stack per parameter**, so no cache is ever shared between the two order-set
   shapes.  Putting `truncation` into `_geom_key` — which destroys exactly the "one entry serves
   both" property the test is named for — leaves it green.

**What I added** (`tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py`; additions only,
nothing weakened, both S5-shaped with a stated fail-before and a two-sided arm):

* `_count_every_assembly` — counts assemblies at **both** module bindings.
* `::test_b6_the_stack_hands_its_cached_build_to_the_layer_modes_builder` (6 ids: 3 cells × 2 sweep
  axes) — requires **1** assembly over a 5-point sweep counting both bindings, `npts` in the
  priced-out arm, and bit-identity between the arms.  Fail-before: 6 assemblies (1 + 5) under the
  `ops=None` mutation.
* `::test_b6_one_entry_serves_a_truncation_FLIP_on_the_same_stack` — ONE stack solves rectangular,
  has `truncation` flipped, solves again; requires **1** assembly total, requires the two arms to
  have **different order-set shapes** (non-vacuity), and requires each to be bit-identical to a
  freshly built stack at that truncation.  Fail-before: 2 assemblies under the
  `truncation`-in-key mutation.

Result: **51 passed, 7.94 s** (7.28 s on the final re-run; was 44 passed, 5.53 s).  `ruff check`
clean on the file and on `lumenairy/elements/pmm/`.

**No library code was changed**, so there is no `VERIFY_WP-B6_CHANGELOG.md` and no history document
needed re-recording: `scripts/record_history_fingerprints.py --check` reports **OK** on all seven
`lumenairy.elements.pmm.*` documents (the only drift in the tree is
`lumenairy.elements._lens_traced.md`, another Wave-4 engineer's in-flight edit).

---

## 5. The census, and the known-red T3-1

`tests/unit/test_ci_kernel_consistency.py` — **7 passed**, 2.59 s.  No PMM decision moved.

`test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band` is
red at the audit base on this workstation.  Run by **direct import** (never through pytest) against
`git archive 3af33dff^ lumenairy` and `git archive 3af33dff lumenairy`, each in its own child
process with `lumenairy.__file__` asserted, the message is identical **character for character** —
`diff` of the two captures shows only the wall-clock line (12.4 s vs 12.1 s):

```
every cell of both devices was screened out as classification-unsound, so T3-1 was not measured
at all.  ...  screened: [('uncoated ns=3', 10, 0, 2, 0), ('25 nm coat ns=8', 6, 0, 1, 0),
('25 nm coat ns=8', 8, 0, 2, 0), ('25 nm coat ns=8', 10, 0, 2, 4)]
```

which is the tuple WP-B6's report quotes.  Nothing this diff does, and nothing I did, moves it.

---

## 6. Defects found and fixed

| # | Where | Defect | Fail-before | Fix |
|---|---|---|---|---|
| V1 | `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` | The "9 → 1 source-free assemblies" contract is counted at `stack2d`'s binding of `_tensor_projected_ops` only, which cannot observe the rebuild `_tensor_layer_modes` performs through `twod_jones`'s own global.  The exact G10(d) regression (`ops=tops` → `ops=None`) leaves all 44 tests green | mutation applied to an isolated tree: **44 passed** | `_count_every_assembly` + `::test_b6_the_stack_hands_its_cached_build_to_the_layer_modes_builder` — both bindings counted; the same mutation now turns **7 ids / 2 functions** red |
| V2 | same file | `::test_b6_the_cached_entry_serves_both_truncations` builds a fresh stack per parametrised truncation, so it never shares one cache entry between the two order-set shapes its name claims.  Adding `self.truncation` to `_geom_key` leaves all 44 green | mutation applied to an isolated tree: **44 passed** | `::test_b6_one_entry_serves_a_truncation_FLIP_on_the_same_stack` — one stack, `truncation` flipped between solves, one assembly required, different order-set shapes required; the same mutation now turns **1** red |

Nothing in `lumenairy/elements/pmm/*.py` needed changing: every library claim I attacked held.

---

## 7. Requested changes outside my ownership

**None.**  Nothing outside `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` and this
report was edited, and nothing outside them needs to be.

Two observations for the orchestrator, neither a request:

1. WP-B6's §6 item 1 offers its oracle script and "its 2.62e-06 estimator floor ... reusable as-is"
   as the gate for the deferred hp mesh.  That floor is a property of the auditor's *shallow* Au
   cell.  On a deeper lossy metal cell the same construction is only good to ~1.7e-04 (§2.1), so
   whoever takes the hp item should re-establish the floor on whatever fixture they choose before
   reading a rate from it.
2. `PMM2DStackHybrid.formulation` is validated in `__init__` but is a plain attribute afterwards, so
   `st.formulation = 'fff_nv'` is accepted and then behaves as `'laurent'` (`ez_rule` maps
   everything that is not `'li'`).  Pre-existing, unchanged by WP-B6, and not worth a behaviour
   change in a verification pass — recorded as Follow-up 3.

---

## 8. Follow-up

1. **State the oracle floor per fixture in the hp gate.**  §2.1 / §3.1: the extrapolated-RCWA
   oracle spans five decades of quality across four cells, and on a deep lossy metal TM cell the
   RCWA and PMM limits bracket rather than meet.  The hp item's gate should carry the
   two-estimator spread it actually measures, on the cell it actually uses.
2. **Read the λ ladder past degree 44 if alternative (c) is ever re-opened.**  §3.2: λ = 0's local
   rate looks like 3.4 at degree 44 and is 0.23 by degree 100.  Any future basis claim should be
   required to show a rate that is still rising at degree 100, not one measured on a ladder that
   stops where the cancellation still looks like convergence.
3. **`formulation` / `symmetry` / `cascade` assignment is unvalidated** on `PMM2DStackHybrid`.  The
   class documents them as mutable between solves and the caches key on them correctly, but an
   out-of-vocabulary value is silently coerced (`'fff_nv'` → the `'laurent'` branch).  A property
   guard mirroring `__init__`'s validation is a small, self-contained change; it is a behaviour
   change (it would start raising) and so belongs in a normal WP, not here.
4. **`::test_b6_a_tensor_sweep_assembles_the_projected_operators_once` and
   `::test_b6_the_cached_entry_serves_both_truncations` are now the weaker halves of two pairs.**
   They are kept (they are not wrong, and they pin the cache-fill half) but the load-bearing arm is
   the one added in §4.  If either is ever edited, the pair should move together.
5. **WP-B6 §6 item 3** (`pmm_jones_2d` assembles the tensor operators twice on a normal-incidence
   out-of-plane or slanted cell) is real and still open: my capture confirms the fold hand-off
   returns `None` for exactly those cells and the full solve then rebuilds.  Measured cost agrees
   with the report's ~0.05 s per solve at degree 11 / `n_orders` 7.

---

## 9. Commands, counts and durations

Every command with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a
time.  `<S>` is the scratch root
`…/78f7e8ef-9607-4e41-ba1d-ebd1c2e74c7e/scratchpad/verify_b6`; `<S>/base` and `<S>/head` are
`git archive 3af33dff^ lumenairy` and `git archive 3af33dff lumenairy` extracted read-only.

Durations are given where the run reported its own (pytest's summary line, or a script's own
timing prints); probe scripts that did not self-time are marked `—` rather than estimated.

| # | command | result | duration |
|---|---|---|---|
| 1 | `git archive 3af33dff^ lumenairy \| tar -x -C <S>/base` ; same for `3af33dff` → `<S>/head` | 3 files differ (`_core.py`, `stack2d.py`, `twod_jones.py`) | — |
| 2 | `cd <S>/head && PYTHONPATH=<S>/head python <S>/v1_family.py` — the ultraspherical family, node/weight cross-checks, SBP residuals | §3.1, §3.3 | — |
| 3 | `cd <S>/head && … python <S>/v2_oracle.py` — the extrapolated RCWA oracle, 5 ladders, 30 `rcwa_efficiency_1d` solves to `n_orders` 1201 | §3.1 table | 293 s (sum of its own per-solve prints) |
| 4 | `… python <S>/v3_ladder.py` — rate ladders (6 λ × 2 mass treatments × 3 fixtures × 6 degrees), invariance (5 fixture/pol × 3 degrees), closure, TE price | §3.2, §3.4, §3.5 | 0.3–0.7 s per ladder row (self-timed); total — |
| 5 | `… python <S>/v4_metal.py` — the metal self-reference, the metal ladder to degree 60, the 6-cell TM-win survey, the 8-row classification table | §3.6, §3.7 | — |
| 6 | `… python <S>/v5_tail.py` — the degree-24…100 tail on F3 and F2 | §3.2 | — |
| 7 | `cd <S>/head && … python <S>/cap.py <S>/head <S>/cap_head.npz` | 768 arrays | — |
| 8 | `cd <S>/base && … python <S>/cap.py <S>/base <S>/cap_base.npz` | 485 arrays | — |
| 9 | `python <S>/cmp.py` | **1 051 of 1 051 bit-identical, 0 differ** | — |
| 10 | `… python <S>/cap2.py <S>/{head,base} …` + compare — `PMMStack._solve_conical` tensor segments | **29 of 29** | — |
| 11 | `cd <S>/head && … python <S>/v6_cache.py <S>/head` — cache attacks (a)–(f) | **ALL CACHE ATTACKS PASSED** (31 checks) | — |
| 12 | `… python <S>/v7_cost.py <S>/{base,head}` — `_axis_projection` counts and entry bytes | §3.11; sweep min-of-3 self-timed (0.122–1.176 s) | — |
| 13 | `… python <S>/v8_share.py <S>/{base,head}` — the assembly share at degree 11 / `n_orders` 7 | §3.11; sweep min-of-3 10.460 s (base) / 10.098 s (after) | — |
| 14 | `python <S>/mutate.py <m>` × 8, before my additions | 2 mutants invisible; §4 | 5.1–6.1 s per pytest run |
| 15 | `pytest tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` (pre-existing) | **44 passed** | 5.53 s |
| 16 | `pytest tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` (with my two additions) | **51 passed** | 7.94 s; 7.28 s on the final re-run |
| 17 | `python <S>/mutate.py <m>` × 8, after my additions | every mutant turns the right test red; §4 | 7.2–8.2 s per pytest run |
| 18 | `python -m ruff check tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py lumenairy/elements/pmm/` | **All checks passed** | — |
| 19 | `pytest tests/unit/test_audit2609_a12_{pmm1d,verify_pmm1d}.py tests/unit/test_audit2609_a13_{jax,stack2d,staggered_cost,twod,verify_guards}.py` | **113 passed** | 92.0 s |
| 20 | `pytest tests/unit/test_v5_14_0_pmm_jones_2d.py test_v5_14_0_pmm2d_oop.py test_pmm2d_oop_block_eig.py test_v5_20_13_pmm_jones_2d_fff_nv.py` | **25 passed** | 101.6 s |
| 21 | `pytest tests/unit/test_ci_kernel_consistency.py` (the census) | **7 passed** | 2.6 s |
| 22 | `pytest tests/unit/test_audit2609_a17_history_lint.py` | **5 passed** | 8.6 s |
| 23 | `python scripts/record_history_fingerprints.py --check` | all 7 `lumenairy.elements.pmm.*` **OK**; the only drift is `lumenairy.elements._lens_traced.md` (another engineer's in-flight edit) | — |
| 24 | `cd <S>/base && … python <S>/t31.py <S>/base` ; same on `<S>/head` ; `diff` | message **identical character for character** | 12.4 s / 12.1 s |
| 25 | `pytest tests/unit -k "pmm2d or jones_2d or stack2d or m2_window or pmm_m3"` | **746 passed, 1 failed, 2 skipped** (15 107 deselected).  The one failure is the documented `test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band` with the screened tuple of §5, character for character.  The 2 skips are premise-absent arms (PySide6 absent; numpy and scipy do not resolve to one LAPACK build on this box) | 2 209.3 s (36:49) |

Files I changed: `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` (additions only) and
this report.
