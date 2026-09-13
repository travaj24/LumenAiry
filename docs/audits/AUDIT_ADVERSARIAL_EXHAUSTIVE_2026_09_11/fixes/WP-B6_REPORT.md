# WP-B6 report — PMM: the ultraspherical (Gegenbauer) basis for the TM wall corner, and the PMM-2D k0-free tensor operator cache

Branch `audit-fixes-2026-09`, base `284daccc` (the WP-B3 commit).  The two items WP-A12 §6 item 3 and
WP-A13 §6.1 deferred **with designs**: the audit's alternative (c) for the `O(N^-2.7)` TM wall corner
(`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/PMM-1D.md`, "Alternative algorithms" (c)), and G10(d) for
`PMM2DStackHybrid` (`.../PMM-2D.md`, G10).

Every number below was measured on this branch, on this workstation, with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and one process at a time.  Performance
is reported as **deterministic build counts** first and wall clock second, with the noise stated: the
box was running ~20 sibling agents throughout (WP-A12 records three failed timing probes for the same
reason).

---

## 1. Summary

| ID | Status | Files : lines | Tests | Oracle | Measured before → after |
|---|---|---|---|---|---|
| **Item 1** — ultraspherical / Gegenbauer basis (WP-A12 §6.3, audit alt. (c)) | **measured, NOT shipped** — a deliberate negative; no `basis=` knob, no default moved, the 1-D surface byte-identical | `pmm/_core.py:356` `_gll_nodes_weights` — **docstring only** (the AST/token fingerprints are unchanged, so `record_history_fingerprints.py --check` stays green on `_core`) | `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py::test_b6_only_the_gll_rule_satisfies_summation_by_parts` (4), `::test_b6_an_exactly_integrated_ultraspherical_basis_cannot_move_the_answer`, `::test_b6_a_non_gll_nodal_rule_loses_the_exact_energy_identity` | the auditor's Au/air TM fixture (`repro/PMM-1D/p8b_conv_rate.py`) against an **extrapolated RCWA oracle** built from `rcwa_efficiency_1d` at `n_orders` 101…1201 + `rcwa_extrapolate(method='richardson')` — called, not edited | the basis swap **cannot** move the answer: with exact element integration the whole λ family agrees to **3.1e-13**.  What λ actually varies is the QUADRATURE: λ = 0 buys TM 5–25× at 40–110× worse TE and destroys the exact discrete energy identity (lossless oblique closure **3.0e-14 → 2.8e-08 … 1.2e-05**).  No λ delivers the audit's "exponential convergence"; local rate stays algebraic (2.2 → 4.1 at best).  §2.1 has the full ladder |
| **Item 2** — the k0-free projected TENSOR operator cache (WP-A13 §6.1, G10(d)) | **fixed**, bit-identical | `pmm/twod_jones.py:149` new `_tensor_projected_ops` (the three assembly branches MOVED, `kind` at `:208/:287/:298/:333`, `:368` the returned dict); `pmm/twod_jones.py:372` `_tensor_layer_modes(..., ops=None)`, rebuild at `:404–415`; `pmm/stack2d.py:71` import, `:523` `_geom_key` gains `formulation`, `:965/:977` and `:1087/:1101/:1103/:1127` the 4-slot cache entry | `::test_b6_the_cached_tensor_ops_reproduce_the_uncached_build_bit_for_bit` (18), `::test_b6_a_supplied_ops_is_actually_consumed`, `::test_b6_the_k0_free_split_is_the_algebra_it_claims` (9), `::test_b6_a_tensor_sweep_assembles_the_projected_operators_once` (6), `::test_b6_the_cached_entry_serves_both_truncations` (2), `::test_b6_the_geom_key_splits_on_the_formulation`, `::test_b6_the_cached_tensor_operators_are_handed_out_read_only` | the pre-change tree itself (`git archive 284daccc lumenairy`, extracted read-only, imported in a child process with `lumenairy.__file__` asserted, never through pytest) | source-free assemblies over a 9-point sweep **9 → 1** on every branch and on BOTH the wavelength and the angle axis; assembly seconds per sweep **0.474 → 0.009 s** (crossed, degree 11 / `n_orders` 7) and **0.169 → 0.002 s** (crossed, degree 9 / 5); **210 of 210** operator arrays and **134 of 134** end-to-end arrays `np.array_equal`, worst `|A−B| = 0.000e+00`; `_geom_cache` retention **0.13 → 6.31 MiB** per tensor entry at degree 11 / `n_orders` 7 |
| Byte identity of every path whose default did not move | **proved** | — | the b6 file's cached-vs-uncached arms; `tests/unit -k pmm` | base vs after tree, child processes | 1-D PMM surface **172 of 172** arrays identical (`worst 0.000e+00`); 2-D operators 210/210; end-to-end 134/134; `test_pmm_m2_window_contract` T3-1's screened tuple identical character for character |

**No PMM default moved.**  The PMM-2D formulation default is untouched, `cascade='fused'` stays
rejected (WP-A13 §2.6 — a ruling this WP does not reopen), `PMM2DStackHybrid.formulation` still
defaults to `'li'`, `truncation` to `'rectangular'`, `symmetry` to `'auto'`.  Item 1 ships no knob at
all, so there is nothing for a Migration note to say.

---

## 2. Per item

### 2.1 Item 1 — the ultraspherical / Gegenbauer basis: measured, and not shipped

**What was asked.**  `_gll_nodes_weights(degree)` and `_lagrange_derivative_matrix(nodes)` are the
only two places the basis enters, so an ultraspherical Gauss–Lobatto rule with parameter λ (λ = 1/2
recovering today's Legendre/GLL) gives a one-parameter family.  The audit calls it "the highest-value
algorithmic move available to this partition" and claims it "recovers exponential convergence for the
TM wall-corner singularity where the Legendre/GLL basis is `O(N^-2.7)`".  The gate WP-A12 set: ship
only if the measured local rate leaves that regime on the Au/air TM fixture, against the extrapolated
RCWA oracle.

**The family, as built.**  Nodes = ±1 plus the roots of `d/dx C_N^(λ)`, i.e. the roots of
`C_{N-1}^(λ+1) ∝ P_{N-1}^(a,a)` with `a = λ + 1/2`, by Golub–Welsch on the monic symmetric-Jacobi
recurrence `β_n = n(n+2a)/((2n+2a+1)(2n+2a-1))`.  At λ = 1/2 this reproduces the shipped GLL nodes and
weights to **3.0e-15 / 1.3e-15** (degree 4…32) — close, but *not* bit-identical, which is already the
first thing a `basis=` knob would have to handle by dispatching its default to the existing body
rather than to the general one.  Two mass treatments were measured, because the two readings of
WP-A12's "the element mass is diagonal only for λ = 1/2" lead to different code:

* **LUMPED** — the nodal (interpolatory, unweighted) rule `w_i = ∫ l_i dx` at those nodes, so the
  SEM's quadrature mass `diag(w·J)` keeps its diagonal shape and `_build_sem*` needs no structural
  change.  All weights are positive at every λ and degree measured (λ ∈ {0, 0.25, 0.5, 0.75, 1, 1.5,
  2} × degree ∈ {8, 16, 24, 32}), and `Σw = 2` to 1e-12, so the rule is a legitimate quadrature.
  λ = 0 is Chebyshev–Lobatto, i.e. Clenshaw–Curtis lumping.
* **CONSISTENT** — the same nodes with every element integral evaluated EXACTLY (Gauss–Legendre,
  `N+5` points), so the mass is the true non-diagonal `∫ l_i l_j dx`.  This is the variant WP-A12
  describes; it needs `_build_sem*` to stop assuming a lumped mass, which is why the design flagged
  `_real_diagonal` returning `None` for a non-diagonal `S0`.

Both were applied by replacing `_core._gll_nodes_weights` (and, for CONSISTENT, `_core._build_sem`)
in-process.  Nothing in the repository was edited to measure them.

**The oracle.**  `rcwa_efficiency_1d` on the auditor's cell (period 0.6 µm, depth 0.1 µm, duty 0.5,
`n_Au = 0.18+3.43j`, air groove, λ = 0.633 µm, 10°, TM, order 0 in R) at `n_orders` 101/201/301/401/
601/801/1201, pushed to the limit by `rcwa_extrapolate`:

| n_orders | 101 | 201 | 301 | 401 | 601 | 801 | 1201 |
|---|---|---|---|---|---|---|---|
| `R0` | 0.415457249620 | 0.415453025811 | 0.415467856501 | 0.415478858861 | 0.415490900089 | 0.415496903719 | 0.415502589176 |

`rcwa_extrapolate(last 3, richardson)` = `rcwa_extrapolate(last 4)` = **0.415512036392**; the audit's
own least-squares `R_inf + c/n` on the last four gives 0.415514655975; the raw `n_orders = 1201`
sample is 9.4e-06 short of the limit.  **The oracle's own floor is therefore 2.62e-06**
(estimator-to-estimator spread), which matters: the PMM ladder reaches below it, so the finest rungs
are oracle-limited and are reported but not leaned on.  A second, cleaner fixture — the audit's
LOSSLESS `n = 3.48` twin, where the rate reads a flat 2.7 out to degree 44 — was built the same way
(`n_orders` 301/401/601/801 → **0.522035139074**, floor 9.95e-07).

#### The rate ladder — Au/air TM, `|R0 − R0_inf|` and the local rate, degrees 8 / 12 / 16 / 24 / 32 / 44

| basis | 8 | 12 | 16 | 24 | 32 | 44 | local rate (12→44) |
|---|---|---|---|---|---|---|---|
| LUMPED λ = 0.00 (Chebyshev–Lobatto) | 3.43e-04 | 6.80e-05 | 2.94e-05 | 9.48e-06 | 3.97e-06 | **1.09e-06** | 3.99 / 2.92 / 2.79 / 3.03 / 4.05 |
| LUMPED λ = 0.25 | 4.53e-04 | 1.49e-04 | 7.46e-05 | 2.82e-05 | 1.39e-05 | 5.92e-06 | 2.74 / 2.41 / 2.40 / 2.47 / 2.67 |
| **LUMPED λ = 0.50 (shipped GLL)** | 5.46e-04 | 2.24e-04 | 1.17e-04 | 4.62e-05 | 2.34e-05 | 1.06e-05 | 2.20 / 2.25 / 2.29 / 2.36 / 2.48 |
| LUMPED λ = 0.75 | 6.40e-04 | 2.94e-04 | 1.57e-04 | 6.28e-05 | 3.22e-05 | 1.50e-05 | 1.91 / 2.20 / 2.25 / 2.32 / 2.41 |
| LUMPED λ = 1.00 | 7.33e-04 | 3.64e-04 | 1.94e-04 | 7.77e-05 | 4.00e-05 | 1.88e-05 | 1.73 / 2.19 / 2.25 / 2.31 / 2.38 |
| LUMPED λ = 1.50 | 8.59e-04 | 5.06e-04 | 2.69e-04 | 1.04e-04 | 5.20e-05 | 2.35e-05 | 1.31 / 2.19 / 2.34 / 2.42 / 2.50 |
| CONSISTENT, **every** λ in {0, 0.25, 0.5, 0.75, 1, 1.5} | 5.90e-04 | 2.34e-04 | 1.21e-04 | 4.71e-05 | 2.37e-05 | 1.07e-05 | 2.28 / 2.30 / 2.33 / 2.38 / 2.50 |

(The last row is one row because the six λ produced the same digits.  Errors below 2.6e-06 sit under
the oracle floor.)

#### The same ladder on the LOSSLESS `n = 3.48` TM cell (the audit's flat-2.7 reading)

| basis | 8 | 12 | 16 | 24 | 32 | 44 | local rate |
|---|---|---|---|---|---|---|---|
| LUMPED λ = 0.00 | 7.13e-05 | 3.22e-05 | 8.46e-06 | 1.48e-06 | 4.79e-07 | **1.55e-07** | 1.96 / 4.65 / 4.30 / 3.93 / 3.54 |
| LUMPED λ = 0.25 | 6.02e-04 | 1.22e-04 | 5.59e-05 | 1.81e-05 | 8.01e-06 | 3.23e-06 | 3.94 / 2.71 / 2.78 / 2.83 / 2.85 |
| **LUMPED λ = 0.50 (shipped)** | 8.34e-04 | 2.67e-04 | 1.19e-04 | 3.81e-05 | 1.69e-05 | 6.82e-06 | 2.81 / 2.80 / 2.81 / 2.83 / **2.84** |
| LUMPED λ = 0.75 | 5.36e-04 | 3.65e-04 | 1.69e-04 | 5.55e-05 | 2.49e-05 | 1.02e-05 | 0.95 / 2.67 / 2.75 / 2.79 / 2.82 |
| LUMPED λ = 1.00 | 6.44e-04 | 3.33e-04 | 1.75e-04 | 6.25e-05 | 2.90e-05 | 1.21e-05 | 1.62 / 2.24 / 2.54 / 2.67 / 2.74 |
| LUMPED λ = 1.50 | 7.85e-03 | 9.72e-04 | 3.62e-04 | 1.02e-04 | 4.35e-05 | 1.76e-05 | 5.15 / 3.43 / 3.13 / 2.95 / 2.85 |
| CONSISTENT, every λ | 6.56e-04 | 2.24e-04 | 1.04e-04 | 3.47e-05 | 1.57e-05 | 6.47e-06 | 2.65 / 2.67 / 2.71 / 2.75 / 2.79 |

The audit's `O(N^-2.7)` reading is reproduced exactly (2.84 on the lossless cell, 2.48 on Au/air);
**nothing in the family is exponential**, and the flat algebraic rate is unchanged for every λ ≥ 1/2.

#### Finding 1 — the basis swap is provably a NO-OP

With exact integration the six λ give **one** answer, to 4.4e-14 (Au TE), 1.4e-13 (Au TM), 1.9e-13
(lossless TE), 3.1e-13 (lossless TM) in the efficiencies at degree 24.  That is not a coincidence to
be re-measured on each fixture: Galerkin on a fixed space with exact integration is invariant under a
change of nodal basis — the operators transform by a congruence and the pencil by a similarity — and
the discrete space here is the same C0 piecewise-`P_N` space for every λ.  **The corner rate is a
property of that space, not of the nodes in it**, so the audit's alternative (c), as stated (a basis
swap in `_gll_nodes_weights` + `_lagrange_derivative_matrix`), cannot deliver what it promises.
Gate: `::test_b6_an_exactly_integrated_ultraspherical_basis_cannot_move_the_answer`, which measures
the exact-integration arms against the LUMPED arms' own spread on the running build rather than
against a pinned constant.

#### Finding 2 — what λ actually varies is the quadrature, and only GLL keeps summation-by-parts

The shipped assembly evaluates the element mass with the NODAL rule (`Mloc = diag(ref_w * J)`), which
is exact only to degree `2N` at GLL — the standard SEM lumping.  That crime is benign at λ = 1/2 for
a specific reason: the GLL rule is exact to degree `2N−1`, which makes `(diag(w), D)` satisfy
summation-by-parts EXACTLY, `M D + (M D)^T = diag(−1, 0, …, 0, +1)` — the discrete integration by
parts every energy and reciprocity identity in the module rests on.  Measured residual
`max|M D + (M D)^T − B|`:

| degree | λ = 0.00 | λ = 0.25 | **λ = 0.50 (shipped)** | λ = 0.75 | λ = 1.00 | λ = 1.50 | exact integration, any λ |
|---|---|---|---|---|---|---|---|
| 8 | 9.224e-01 | 4.198e-01 | **1.329e-14** | 3.704e-01 | 7.112e-01 | 1.351e+00 | 8.549e-15 |
| 16 | 9.630e-01 | 4.407e-01 | **9.731e-14** | 3.922e-01 | 7.563e-01 | 1.449e+00 | 2.408e-14 |
| 24 | 9.704e-01 | 4.446e-01 | **1.001e-13** | 3.966e-01 | 7.655e-01 | 1.470e+00 | 1.402e-13 |
| 32 | 9.730e-01 | 4.460e-01 | **1.927e-13** | 3.982e-01 | 7.689e-01 | 1.478e+00 | 2.358e-13 |

Eleven decades, not a tolerance question.  The consequence on the solver, on a LOSSLESS cell at
oblique incidence (degree 24, `|Σ R + Σ T − 1|`):

| mass | pol | λ = 0.00 | λ = 0.25 | **λ = 0.50** | λ = 0.75 | λ = 1.00 | λ = 1.50 |
|---|---|---|---|---|---|---|---|
| LUMPED | te | 2.772e-08 | 1.541e-08 | **3.020e-14** | 7.471e-08 | 6.566e-07 | 1.190e-05 |
| LUMPED | tm | 4.872e-08 | 3.653e-08 | **1.554e-13** | 6.265e-08 | 3.132e-07 | 7.491e-06 |
| CONSISTENT | te | 7.216e-14 | 4.285e-14 | 2.265e-14 | 4.818e-14 | 4.086e-14 | 3.020e-14 |
| CONSISTENT | tm | 3.542e-14 | 2.927e-13 | 8.948e-14 | 1.656e-13 | 3.020e-14 | 9.237e-14 |

The CONSISTENT row is the control: exactness restores closure at every λ, so the defect is the
quadrature and nothing else.  So the LUMPED λ = 0 "win" in the ladder above is an error-cancellation
against the corner, bought by breaking a conservation identity — not a better basis.

#### Finding 3 — the price, measured on both sides

* **TE regresses.**  `|R0 − R0(λ=0.5, degree 120)|` at degrees 8 / 12 / 16 / 20 / 24, Au/air TE:
  λ = 0.50 `9.45e-06 → 1.14e-09`; λ = 0.00 `3.30e-04 → 1.27e-07` (**110× worse at degree 24**);
  λ = 0.25 `→ 7.19e-08` (63×); λ = 1.00 `→ 2.43e-06` (2100×).  Lossless TE at degree 24:
  3.48e-10 (λ = 0.5) against 1.33e-08 (λ = 0), 1.77e-06 (λ = 1).  TE is the polarization for which
  the library documents spectral convergence with no floor; this family takes it away.
* **`stabilize=True` — the DEFAULT on `pmm_efficiency_1d` — can change its verdict.**
  `_core._energy_clean_pick` calls a structure "evidently lossless" at `|Σ R + Σ T − 1| < 1e-6` and
  then picks the cluster member by energy fitness; otherwise it keeps the historical pick.  Worst
  closure over degrees 16…20 on four lossless cells: every cell stays *ok* at λ ≤ 0.75 (except
  duty 0.3, which flips at 0.75), and λ ≥ 1.0 flips **6 of 8** cell/polarization rows from `ok` to
  the lossy branch (e.g. lossless 3.48 TE `7.62e-14 → 4.84e-06`, duty 0.3 TM `6.88e-14 → 1.81e-04`).
  A knob that silently moves a *classification* is a different class of risk from one that moves
  digits.
* **The TM win is robust but modest and fixture-shaped.**  TM order-0 error at degree 32, against a
  degree-140 self-reference, over seven cells (Au gate; lossless 3.48; duty 0.3; 4× deeper; normal
  incidence; sub-wavelength period; the Si/1 µm cell): λ = 0 beats λ = 0.5 on all seven, by 5.3× /
  23× / 6.7× / 86× / 31× / 35× / 31×.  Real — and entirely explained by Finding 2.

#### Verdict

**Nothing ships for item 1.**  The gate's rate criterion is arguably met by LUMPED λ = 0 on the gate
fixture (local rate ≈ 3.2 average against 2.3, error 9.7× lower at degree 44), and it is reported as
such — but the mechanism is not the Gegenbauer basis (Finding 1 shows that mechanism is empty), it is
not exponential, and its price is the exact discrete energy identity plus a 40–110× TE regression
plus a moveable `stabilize` classification.  A `basis=` knob carrying that is not worth exposing, and
the honest recommendation for the wall corner is the construction WP-A12 §6 item 4 already describes:
a genuine hp mesh (geometric grading σ ≈ 0.15 **with a linearly decreasing degree toward the
corner**, Babuška–Guo), which attacks the space rather than the nodes in it and is the only route in
this family to `exp(−c√DOF)`.  What DID ship is the reason, in the one place a future attempt will
look: `_core.py:356`'s docstring now records why the GLL rule specifically, with the measured SBP and
closure numbers and a pointer to this report, and three tests re-derive them on the running build.

**What a second attempt must re-measure** (it is not enough to re-run the ladder): the TE ladder, the
lossless oblique energy closure, and `_energy_clean_pick`'s lossless classification — plus a
diagonal-norm SBP construction, since restoring the identity on non-GLL nodes needs a `D` built to
satisfy it and not the Lagrange differentiation matrix, which is no longer "a basis swap in two
functions".

### 2.2 Item 2 — the k0-free projected TENSOR operator cache (G10(d))

**What was wrong.**  `PMM2DStackHybrid._build_layer_modes` passed `lops=None` for a tensor layer, so
`_geom_cache` stored `(ax, ay, None)`: the nodal axis build was reused across a sweep, but
`_tensor_layer_modes` rebuilt the per-axis projections (`_axis_projection` + `pinv`) and the `_proj`
sandwiches at **every wavelength and every angle**, where the scalar branch cached them.

**What was done — the code was MOVED, not rewritten.**  `twod_jones.py:149`
`_tensor_projected_ops(ax, ay, x_walls, y_walls, tile_i, ox, oy, formulation)` holds the off-plane
Schur fold and the three discretization branches verbatim; the only lines that changed inside them
are the four that used to spell `GxF`/`GyF` and now name the k0-free half and the branch
(`twod_jones.py:208, :287, :298, :333`).  `kind` says which axes carry a k0-free part:

| `kind` | cell | `GxF` | `GyF` |
|---|---|---|---|
| `'uniform'` | no walls | `diag(kxv)` | `diag(kyv)` |
| `'x'` | separable, x-patterned | `Gx0F/k0 + kx0·IpxF` = `kron(Iy, g1)/k0 + kx0·kron(Iy, ip1)` | `diag(kyv)` |
| `'y'` | separable, y-patterned | `diag(kxv)` | `Gy0F/k0 + ky0·IpyF` |
| `'xy'` | crossed | `Gx0F/k0 + kx0·Ip` | `Gy0F/k0 + ky0·Ip` (one `Ip`, both slots) |

`_tensor_layer_modes` keeps its signature and gains `ops=None` (`twod_jones.py:372`); with `ops`
supplied it rebuilds `GxF`/`GyF` from `kind` (`:413–416`) and goes straight to the `keep` restriction,
the `return_ops` fold gate, the `block_eig` gauge and the eig.  `offp` is read back from the ops'
own out-of-plane slots rather than re-derived from the tile, so a cached build and a fresh one cannot
disagree about which cascade a layer takes.  One provably dead statement went with the move
(`Nf = int(np.count_nonzero(keep))` in the `keep` branch: `Nf` had no reader after it).

`stack2d.py` caches it exactly as it caches `lops`: the `_geom_cache` value is now a 4-tuple
`(ax, ay, lops, tops)` (`:965`, `:977`, `:1087`, `:1103`), the tensor build is assembled on the FULL
order box so ONE entry serves both truncations (`:1101`; `keep` is applied at use inside
`_tensor_layer_modes`, the same property `_restrict_lops` gives the scalar branch), and the entry is
handed to `_tensor_layer_modes` as `ops=tops` (`:1127`).

**One key change was necessary, against the design's "no key change".**  The cached *scalar* `lops`
are formulation-independent — `_scalar_projected_ops` returns every rule's operator side by side and
the caller routes — but the cached *tensor* operators are not: `EZZ` is `inv([[1/e_zz]])` under
`'li'` and the direct `[[e_zz]]` otherwise.  `formulation` is a plain public attribute with no
property guard, i.e. exactly the W7 A11 shape, so it joined `_geom_key` (`stack2d.py:523`).  Without
it, mutating `st.formulation` after a solve would have served the stale tensor build with no signal —
a defect this change would have *introduced*.  Gate:
`::test_b6_the_geom_key_splits_on_the_formulation`, which first shows the two rules genuinely
disagree (so the contract cannot pass vacuously) and then requires the mutated object to reproduce a
fresh one bit for bit.

**Bit-identity, the gate the design set.**  Two independent A/B captures, each run once per tree in
its own child process with `cwd` + `PYTHONPATH` = that tree and `lumenairy.__file__` asserted before
anything is measured; the reference tree is `git archive 284daccc lumenairy` extracted read-only into
the scratch directory; pytest is never involved (it would put the repo root ahead of `PYTHONPATH`).

* **The seven operators at the hand-off** (`_layer_eigenmodes_tensor`'s arguments, plus the four
  out-of-plane blocks and the block-eig gauge), over 20 cells + 3 fold hand-offs + a block-eig cell +
  a 3-point wavelength/angle sweep: uniform / separable-x / separable-y / crossed, `'laurent'` /
  `'li'` / `'fff_nv'`, in-plane and out-of-plane, normal and oblique, rectangular and circular
  truncation, vertical and slanted — **210 arrays compared, 0 differ, worst `|A−B| = 0.000e+00`**.
* **End to end**, over 28 fixtures (14 oblique stacks, 3 normal-incidence stacks including the
  even-parity fold and the block-eig gauge, a wavelength sweep, an out-of-plane wavelength sweep, two
  4-point angle sweeps, 6 `pmm_jones_2d` calls including `fff_nv` and circular truncation,
  `pmm_jones_1d_conical_tensor`, and `PMMStack`'s conical tensor-segment path) — **134 arrays
  compared, 0 differ, worst `0.000e+00`**.
* **The 1-D surface**, which item 1 leaves alone and item 2 does not touch: 172 arrays over
  te/tm × degree 8/16/24/33 × 0°/10°/35° × Au/dielectric, plus `stabilize=True`, the graded
  `elements_per_region=3` mesh, `pmm_jones_1d`, `pmm_jones_1d_conical`,
  `pmm_efficiency_1d_segments`, `PMMStack` scalar and tensor-segment — **0 differ, worst
  `0.000e+00`**, base vs after vs the live working tree.
* **In-tree, forever**: `::test_b6_the_cached_tensor_ops_reproduce_the_uncached_build_bit_for_bit`
  runs the same comparison on 9 branches × 2 truncations inside the suite, and
  `::test_b6_a_supplied_ops_is_actually_consumed` is its non-vacuity guard (a
  `_tensor_layer_modes` that ignored `ops` would pass a bit-identity check trivially, so the test
  hands it a deliberately perturbed build and requires the perturbation to come out the other side).

**What it recovers.**  Deterministic first — the number of source-free assemblies over a 9-point
sweep, counted by wrapping `_tensor_projected_ops` (and `_tensor_layer_modes`, which must stay at 9
because the eig genuinely depends on the source):

| fixture | sweep | `_tensor_layer_modes` calls | assemblies before → after | assembly seconds before → after | sweep min-of-3 before → after | `_geom_cache` bytes before → after |
|---|---|---|---|---|---|---|
| crossed, degree 9, `n_orders` 5 | wavelength | 9 | **9 → 1** | 0.169 → 0.002 s | 2.363 → 2.053 s | 0.09 → 1.88 MiB |
| crossed, degree 9, `n_orders` 5 | angle | 9 | **9 → 1** | 0.144 → 0.002 s | 2.276 → 1.976 s | 0.09 → 1.88 MiB |
| crossed, degree 11, `n_orders` 7 | wavelength | 9 | **9 → 1** | **0.474 → 0.009 s** | 11.186 → 10.269 s | 0.13 → 6.31 MiB |
| crossed, degree 11, `n_orders` 7 | angle | 9 | **9 → 1** | **0.510 → 0.009 s** | 10.894 → 9.925 s | 0.13 → 6.31 MiB |
| separable, degree 11, `n_orders` 7 | wavelength | 9 | **9 → 1** | 0.031 → 0.006 s | 10.218 → 9.760 s | 0.07 → 5.48 MiB |
| separable, degree 11, `n_orders` 7 | angle | 9 | **9 → 1** | 0.045 → 0.004 s | 9.986 → 9.971 s | 0.07 → 5.48 MiB |
| out-of-plane separable, degree 9, `n_orders` 5 | wavelength | 9 | **9 → 1** | 0.036 → 0.005 s | 5.248 → 4.874 s | 0.05 → 2.51 MiB |
| out-of-plane separable, degree 9, `n_orders` 5 | angle | 9 | **9 → 1** | 0.030 → 0.006 s | 5.037 → 4.641 s | 0.05 → 2.51 MiB |

**Read honestly**: the CROSSED branch is where this matters — 0.474 s of an 11.19 s sweep, i.e.
**4.2 % of the whole solve recovered**, which is the same order as the audit's own "0.47 s of a ~20 s
solve" and is genuinely outside the noise.  The separable and out-of-plane branches assemble far less
(0.03–0.05 s per sweep, ≤ 0.5 %), so their whole-sweep deltas here — up to 0.46 s — are **contention,
not this change**, and I am claiming nothing from them.  The deterministic claim is the whole claim:
9 → 1 assemblies on every branch and on both sweep axes, with the assembly seconds measured at the
same two points in both trees.

**The cost, stated.**  Retaining the operators grows a tensor `_geom_cache` entry ~48× (0.13 →
6.31 MiB at degree 11 / `n_orders` 7 — eleven dense `Nf × Nf` complex128 blocks at `Nf = 225`).
`LayerCache` prices it (`cache_max_bytes`, `LUMENAIRY_CACHE_BUDGET_MB`, default floor 256 MiB), and
refusal degrades to a rebuild and never to a wrong answer — which is asserted, not assumed:
`::test_b6_a_tensor_sweep_assembles_the_projected_operators_once` runs its fail-before arm with
`cache_max_bytes=1`, requires the count to return to 9 per 9 points, and requires **every return of
every sweep point to be `np.array_equal` between the two arms**.

**Residual risk.**  The one behaviour-visible change is the `_geom_key` extension, which can only
SPLIT keys that were previously shared — never merge — so its worst case is a rebuild.  Cached arrays
are frozen by `_freeze_cached` and the W7 A13 poisoning guard is extended to the new slot
(`::test_b6_the_cached_tensor_operators_are_handed_out_read_only`).  `_symmetric_layer_specs` writes
`tops=None` (only scalar/uniform layers reach it) and a tensor layer can never read that entry
because `L["kind"]` is in the key; if it somehow did, `ops=None` rebuilds — correct, just slower.

---

## 3. Files touched

Source (all within this WP's ownership):

* `lumenairy/elements/pmm/twod_jones.py` — `_tensor_projected_ops` extracted, `_tensor_layer_modes`
  gains `ops=`.
* `lumenairy/elements/pmm/stack2d.py` — the 4-slot `_geom_cache` entry, `_geom_key` + `formulation`.
* `lumenairy/elements/pmm/_core.py` — `_gll_nodes_weights` **docstring only** (why GLL, with the
  measured SBP and closure numbers).

Documents:

* `docs/history/lumenairy.elements.pmm.stack2d.md` — re-recorded in this change with
  `scripts/record_history_fingerprints.py` and a `re_recorded:` reason.  `_core.md` did **not** need
  re-recording and was not touched: the AST-and-token fingerprints ignore docstrings, so
  `--check` proves the `_core.py` edit really was documentation-only, which is the whole point of the
  pin.  `twod_jones.py` has no history document.
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B6_REPORT.md` (this file) and
  `WP-B6_CHANGELOG.md`.

Tests:

* `tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` — new, 44 tests, 6.2 s.

---

## 4. Tests run

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_b6_pmm_basis_and_tensor_cache.py` | **44 passed**, 6.18 s |
| `pytest tests/unit/test_audit2609_a12_pmm1d.py test_audit2609_a12_verify_pmm1d.py test_audit2609_a13_{jax,stack2d,staggered_cost,twod,verify_guards}.py` | **113 passed**, 101.8 s |
| `pytest tests/unit/test_v5_14_0_pmm_jones_2d.py test_v5_14_0_pmm2d_oop.py test_pmm2d_oop_block_eig.py test_v5_20_13_pmm_jones_2d_fff_nv.py` | **25 passed**, 106.3 s |
| `pytest tests/unit -k "pmm"` | **1725 passed, 1 failed, 3 skipped**, 37.5 min — the one failure is the documented `test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band` (see below); the 3 skips are the suite's own premise-absent arms (numpy/scipy LAPACK split, and a dense grid with no WRONG rows) |
| `pytest tests/unit/test_ci_kernel_consistency.py` (the census) | **7 passed** — no PMM decision moved |
| `python validation/run_all.py` | **ALL 37 files passed** |
| `ruff check .` | clean (repo-wide) |
| `python scripts/record_history_fingerprints.py --check` | all 7 `lumenairy.elements.pmm.*` documents **OK** (the drift it reports is in `raytrace/*`, `elements/_lens_real.py`, `lenses_maslov.py` and `propagators/carrier.py` — other Wave-4 engineers' in-flight edits, none of them mine) |
| `pytest tests/unit/test_audit2609_a17_history_lint.py` | 4 passed, **1 failed** — the ratchet grew in `lumenairy/elements/lenses_maslov.py` (1 → 3), WP-B1's file.  None of my three modules appears in the offender list |

### The known-red T3-1, before and after

`test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band` is
red at the audit base on this workstation (BLAS-build classification of its degree ladder).  Its
numbers are unchanged, proved by running the test function itself by DIRECT IMPORT (never through
pytest, which would import the working tree either way) against the archived base and against the
after tree.  All three readings — base `284daccc`, base + this diff, and the live working tree — are
identical character for character:

```
every cell of both devices was screened out as classification-unsound, so T3-1 was not measured at
all.  ...  screened: [('uncoated ns=3', 10, 0, 2, 0), ('25 nm coat ns=8', 6, 0, 1, 0),
('25 nm coat ns=8', 8, 0, 2, 0), ('25 nm coat ns=8', 10, 0, 2, 4)]
assert 0 >= 1
```

---

## 5. Requested changes outside my ownership

**None.**  Nothing outside `lumenairy/elements/pmm/*.py`, their history documents, the new b6 test
file and the two report files was edited, and nothing outside them needs to be.

Two observations for the orchestrator, neither a request against my files:

1. `tests/unit/test_audit2609_a17_history_lint.py::test_no_module_accumulates_more_version_history`
   is red on `lumenairy/elements/lenses_maslov.py` (1 → 3 version-history lines, at `:153`, `:1764`,
   `:2122`, all naming "pre-5.47").  That is WP-B1's file and its call — either the comments move to
   `docs/history/lumenairy.elements.lenses_maslov.md` or the baseline is re-recorded with the reason
   in the commit message.
2. The working tree was transiently unimportable mid-session (`lumenairy/analysis/psf_mtf_otf.py` had
   no `encircled_energy_profile` while `analysis/core.py` re-exported it).  It resolved on its own.
   Every byte-identity and performance measurement in this report was therefore taken against
   isolated trees in the scratch directory (`git archive 284daccc lumenairy`, and that same archive
   plus only this WP's two source files), so no sibling's in-flight edit can be inside any A/B above;
   the suite runs in §4 are against the live tree.

---

## 6. Deferred, with designs

1. **A genuine hp mesh for the TM wall corner** — geometric grading σ ≈ 0.15 **with a linearly
   decreasing degree toward the corner** (Babuška–Guo), i.e. WP-A12 §6 item 4.  After this WP's
   measurements it is not merely the other option, it is the **only** one of the two that can work:
   §2.1 Finding 1 shows the corner rate belongs to the C0 piecewise-`P_N` space, and hp is the
   construction that changes that space.  `_graded_boundaries` already produces the mesh; what is
   missing is a per-element degree, today a single scalar threaded through `_build_sem*`,
   `_l2g_periodic`, `_sem_fourier_projection` and every `mats["degree"]` reader.  Effort: ~2 days.
   Gate: the same Au/air TM fixture and extrapolated RCWA oracle this report builds (the oracle
   script and its 2.62e-06 estimator floor are reusable as-is), plus the TE ladder and the lossless
   oblique closure, which §2.1 Finding 3 shows are the two things a corner cure must not spend.
2. **A diagonal-norm SBP operator on non-GLL nodes.**  If the clustering of §2.1's λ < 1/2 is wanted
   without the conservation loss, `D` must be *built* to satisfy `M D + D^T M = B` rather than being
   the Lagrange differentiation matrix.  That is a real construction and a real literature, but it is
   no longer "a basis swap in two functions", and it should be measured against hp first (hp is
   cheaper to reach from here and attacks the larger term).
3. **`pmm_jones_2d` assembles the tensor operators twice on a normal-incidence out-of-plane or
   slanted cell.**  It calls `_tensor_layer_modes(return_ops=True)` for the even-parity fold; that
   call returns `None` for such a cell, and the full solve below then rebuilds.  Design: build the
   ops once with `_tensor_projected_ops` and pass the same dict to both calls — bit-identical by
   construction (it IS the same assembly), entirely inside `twod_jones.py`.  Not taken here because
   it adds a branch to a surface whose byte-identity is already fully proved, for a measured ~0.05 s
   per solve at degree 11 / `n_orders` 7 (one assembly, from the §2.2 table).  Effort: ~1 h including
   its own count test.
4. **The same cache for `PMM2DStackPure` / the staggered engine**, which has its own per-layer build.
   Out of scope here (WP-B5 owns `eme/*`-adjacent work and `stack2d_pure.py` was left untouched to
   keep this diff's identity proof exact); the shape of the fix is the one this WP just took.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B6_CHANGELOG.md`
