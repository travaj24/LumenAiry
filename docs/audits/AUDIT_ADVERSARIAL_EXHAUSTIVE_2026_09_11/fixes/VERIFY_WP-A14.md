# VERIFY-A14 -- independent adversarial re-verification of WP-A14 (RCWA / EME / BOR)

Verifier: VERIFY-A14 (did not write the fixes).  Subject: commit `e1bf79be`
(`fix(rcwa,eme,bor): WP-A14 ...`), diff base `e1bf79be~1` = `1ada5bc5`.
Findings: H1-H6 (report section 13), G11 (section 12), partition report
`RCWA-EME-BOR.md`.  Branch `audit-fixes-2026-09`.

Everything below was re-MEASURED on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.  Nothing is
quoted from the WP report without an independent re-measurement; where my number
differs from the WP's, both are given.

Two oracles were written for this verification and share no code with the
library or with the WP's gate file:

* an **`expm` transfer-matrix 1-D RCWA** -- the layer is a single
  `scipy.linalg.expm` of the first-order Maxwell system, so it has **no layer
  eigendecomposition** (hence no mode ordering, no `_sqrt_decay` branch choice,
  no `_inv_lam` floor), no S-matrix and no Redheffer star.  Cross-checked against
  the audit's own independent 4N boundary-match oracle over three mounts x two
  polarizations (off-anomaly, M = 5): **max|dR| <= 6.4e-15, max|dT| <= 2.6e-14**,
  and the two agree on the lossy mount's absorptance to 4 digits -- so the two
  independent formulations pin each other at the 1e-14 level, 10 decades below
  every bar used here;
* the **analytic Dirichlet-disk normalisation** `c_n = sqrt(2)/|J_{m+1}(j_{m,n})|`
  for the BOR eigenvectors (pure `scipy.special`).

---

## 1. Summary

| # | Verdict | Independent oracle / method | Measured (verifier's own numbers) |
|---|---|---|---|
| **H1** | **VERIFIED-WITH-NOTES** + **1 defect found and FIXED** | exact hybrid HE11 `fiber_oracle` on 6 V-values and 2 new index systems | band fix real: HE11 returned at V = 2.30/2.40/2.405/2.41/2.60/3.20 (dn = 0.010), n_eff err 4.6e-05..6.1e-05 at N = 200; raise fires exactly at dn <= 2e-6. **BUT** the "silent `[]` cannot return through another door" claim is FALSE: Si/SiO2 (dn = 2.04) V = 2.0 returned `[]` with **0 warnings** while the oracle has HE11 at 1.846898458 -- the `tail_tol` screen, not the band. Fixed (see section 3). |
| **H2** | **VERIFIED-WITH-NOTES** | `expm` transfer matrix + the audit's 4N oracle, on 3 mounts WP-A14 did not use; 2-D crossed grating (continuity/closure); 4000-call bit-identity census for the shared nudge | announcement + `wl_eff` + argument forwarding all exact (0.0). sqrt(delta) law CONFIRMED independently (per-decade ratio 3.15-3.18 vs sqrt(10) = 3.162 on 8 mount/pol pairs). Gain over the one-sided nudge is **1.05x .. 60.5x**, not uniformly 60x; post-fix relative residual reaches **2.9e-04** on a `n_sub = 1.5` mount (the WP's headline 6.7e-06 is its fixture's). Grazing artefact is **3.25 decades** below the specular on a lossy Ag mount, not "4-5". At a MATCHED bracket the averaging helps on 3 of 8 mount/pol pairs and hurts on 5 (up to 10x): the reliable lever is the narrower bracket, the average buys CONTINUITY (V14). |
| **H3** | **VERIFIED** | the cell's own symmetry; a 3-cell non-transpose-symmetric convergence study; 65-fixture Manhattan false-positive census | symmetrisation converges to the same limit as `L2L1` on non-symmetric cells (|mean - L2L1| 8.3e-04 -> 1.2e-04 for M = 4 -> 12, both -> `li@M=16`); exact on BOTH axes (y-uniform stripe 4.1e-14, **x-uniform** stripe 8.7e-14) incl. non-square raster/period and Mx != My; notice false-positive rate 1/65 on Manhattan cells. Gap: the notice does NOT fire on an OUT-OF-PLANE tensor cell. |
| **H4** | **VERIFIED** | full `2N` solve; Bessel zeros; the ANALYTIC disk normalisation; 288-configuration robustness sweep | fold 3.15x/3.27x (`li`), 2.63x/2.96x (`fff_nv`), 3.12x/3.48x (`laurent`) at M = 6/9, peak RSS 46.6 -> 31.5 MB; max\|dJ\| **bit-identical to the WP's table**. Fold does NOT engage at 20 deg for any formulation and falls through bit-identically on a non-centro-symmetric cell. `eigh` errors reproduce digit for digit; 0 failures in 288 configurations; eigenvectors verified M-orthonormal against the ANALYTIC normalisation to 1.0e-13..4.1e-11 (2-norms 14.0-17.7, so definitely not unit 2-norm). |
| **H5** | **VERIFIED** (code half); dependency line still **OUTSTANDING** | direct probe | once-per-process; text carries INERT / 400x / 140x / 2.29 s / 18.2 s / `threadpoolctl` / the three env vars. Re-confirmed `threadpoolctl` is in NEITHER `pyproject.toml` NOR `requirements.txt`. |
| **H6** | **VERIFIED-WITH-NOTES** + **1 defect found and FIXED** | mutation probe on both ports; 80-solve passive census; Bloch-phase Hermiticity; scalar/array spelling sweep | 0 aliased keys on `reflection` AND `transmission`; passive bar 0 false positives in 80 solves (60 absorbing pillars + 20 Ag gratings), `stabilize` ladder unchanged; `ref_2d_modes` max\|Im lam\| **exactly 0.0** at (kx0, ky0) = (0,0), (0.37,0), (0.37,-0.9).  **BUT** the 1-D passive guard packed its two layer permittivities into one `np.array([...])` OUTSIDE `_passive_media`'s own except-clause, so `n_ridge = np.array([2.04+0j])` with a scalar `n_groove` raised `ValueError` from the energy guard AFTER a complete solve.  Fixed (section 3, V11). |
| **G11** | **VERIFIED** | the definition `Re(lam^2) < 0`, on a 5000-draw randomised population spanning 12 decades; NumPy-vs-JAX set identity | flip set is EXACTLY `band & (Im<0) & (Im^2 > Re^2)`; every flipped root has `Re(lam^2) < 0` with **no tolerance**; round-1 predicate is a strict superset and every dropped entry has `Re^2 >= Im^2`. No genuinely on-cut PROPAGATING mode can be wrongly rejected (the conjunct IS the propagating test). JAX flip set identical on 1200 values, values 8.0e-14 apart. |
| extra | **VERIFIED** | -- | the WP's new BOR superposition-closure gate passes on re-run. |

**Collateral damage: none attributable to WP-A14.**  One failing test found in the
batches I ran -- `tests/unit/test_v5_21_delta_audit.py::test_d3_air_focus_multibranch_runs_without_warning`
-- comes from `lumenairy/elements/_lens_traced_multibranch.py`, an UNCOMMITTED
working-tree file belonging to another work package (reproduced standalone; the
warning is `apply_real_lens_traced_multibranch: reconstructed grid power is
2.32x the input aperture power`).

---

## 2. Per finding

### H1 -- `guided_modes`' guard band

**Re-measured, band half.**  With the exact hybrid HE11 determinant
(`fiber_oracle.fiber_modes`) as the oracle, `lambda = 1.55 um`, `Rbig = 6a`,
`N = 200`, dn = 0.010:

| V | 2.30 | 2.40 | 2.405 | 2.41 | 2.60 | 3.20 |
|---|---|---|---|---|---|---|
| exact HE11 n_eff | 1.445036158 | 1.445293173 | 1.445305560 | 1.445317905 | 1.445756491 | 1.446819582 |
| `guided_modes` | 1.444975112 | 1.445233747 | 1.445246218 | 1.445258645 | 1.445700485 | 1.446773607 |
| error | -6.10e-05 | -5.94e-05 | -5.93e-05 | -5.93e-05 | -5.60e-05 | -4.60e-05 |

A mode is returned at every V, on both sides of the 2.405 single-mode cut-off;
pre-fix the list was empty at every one of them (the pre-fix arithmetic
`window <= 2 * 5e-3 * k0` is asserted in the WP's own gate and reproduces).
The FD error is the ~6e-5 floor the WP's residual-risk note describes; it is
NOT monotone in `N`, as the WP says.

The degenerate-window **raise** fires exactly where the rule says:
dn = 5e-6 / 2.1e-6 / 2.0e-6 -> returns `[]`; dn = 1.9e-6 / 1.0e-6 / 0.0 /
-1e-3 -> `ValueError` naming both permittivities.  The high-contrast regime the
old band was harmless in is unmoved (legacy `e1 = 6, e2 = 2` fixtures return 2
modes at N = 150/300/600, unchanged).

**Defect found.**  The report and the changelog both state that an empty result
"cannot come back through a different door".  It can, and does.  The notice was
armed only on the GUARD BAND, while `guided_modes` has three rejecting filters.
Measured on HEAD before my change:

| fixture | exact oracle | raw FD spectrum held | rejected by | warnings |
|---|---|---|---|---|
| Si/SiO2 (dn = 2.04) V = 2.0, N = 200 | **1.846898458** | 1.882075103 (reldiv 6.2e-02) | `tail` = 1.00 | **0** |
| Si/SiO2 V = 2.0, N = 400 | 1.846898458 | 1.931711794 | `tail` = 1.00 | **0** |
| SiN/SiO2 (dn = 0.56) V = 2.0, N = 300 | **1.637263984** | 1.637309955 (**4.6e-05 away**) | `tail` = 0.599 | **0** |
| dn = 0.010, V = 2.6, m = 0 | 1.440496050, 1.440490486 | 1.440712635, 1.440579728 | `tail` = 1.00 | **0** |
| dn = 0.010, V = 2.6, m = 2 | 1.440476919 | 1.440501215 | `tail` = 1.00 | **0** |

The SiN row is the sharpest: the raw eigensolver has the HE11 to 4.6e-05 and
the public function returns `[]` with no signal -- exactly the H1 user-visible
failure, on a different door.  **Fixed** (section 3).

**Residual, NOT fixed and reported as an open item:** a PARTIAL result is still
silent.  Si/SiO2 at V = 4.0 returns 1 mode where the oracle has 3
(3.038391486 / 1.566626407 / 1.440009396); the notice only fires when the list
is EMPTY.

### H2 -- the Rayleigh-anomaly nudge

**(a) The announcement, `wl_eff` and the forwarding: VERIFIED, exactly.**
One `WoodNudgeWarning` per solve on all eight covered entry points; the category
is filterable and promotable (`warnings.filterwarnings('error', ...)` raises
cleanly); `Efficiency2D.wl_eff` / `RCWAResult.wl_eff` carry the `(lo, hi)` pair
on-anomaly and a bare float off it; `_wl_eff` is `KEYWORD_ONLY` on every entry
point, so no positional call can collide with it.

The `_WoodAnomaly` + `inspect.signature().bind` re-entry was the item I most
expected to break.  It does not.  Comparing `f(args)` against the hand-made mean
of `f(args, _wl_eff=lo)` and `f(args, _wl_eff=hi)`:

| entry point / argument shape | max\|f - mean(legs)\| |
|---|---|
| `rcwa_efficiency_1d`, 8 positional + tm/laurent/n_orders = 9 | **0.0** |
| `rcwa_efficiency_1d`, ALL keyword + te/li/n_orders = 13 | **0.0** |
| `rcwa_efficiency_1d`, ASR (`asr_eta = 0.5`, `asr_samples = 4096`) | **0.0** |
| `rcwa_jones_1d`, tensor ridge + `return_jones_transmission=True` | **0.0** |
| `rcwa_jones_1d_segments`, 3 segments, n_orders = 7 | **0.0** |
| `rcwa_efficiency_2d`, li, 3x2, tm, `symmetry=False` | **0.0** |
| `rcwa_efficiency_2d`, laurent, `symmetry=True` (a DIFFERENT solve path) | **0.0** |
| `rcwa_efficiency_2d_shapes`, analytic rectangle | **0.0** |
| `rcwa_jones_2d`, `fff_nv` + `allow_nonseparable_nv=True` | **0.0** |
| `rcwa_jones_2d`, li + `symmetry=True` | **0.0** |
| `PreparedRCWA2D.solve` | **0.0** |
| `RCWAStack._solve_once` (a METHOD -- `self` carried) | **0.0** |

Pinned by `tests/unit/test_audit2609_a14_verify.py::test_wood_symmetric_reentry_forwards_every_argument`
(8 parametrised arms) and `::test_wood_symmetric_reentry_forwards_stack_arguments`.

**(b) The sqrt(delta) law: INDEPENDENTLY CONFIRMED.**  I drove the shipped code
path at brackets 1e-6 .. 1e-11 by overriding `_WOOD_PAIR_STEP_REL` and measured
the symmetric average's error against the 4N oracle at the EXACT wavelength:

| mount / pol | 1e-6 | 1e-7 | 1e-8 | 1e-9 | ratio per decade |
|---|---|---|---|---|---|
| Moharam TM | +3.30e-05 | +1.04e-05 | +3.30e-06 | +1.04e-06 | 3.17 / 3.16 / 3.16 |
| Moharam TE | -1.81e-06 | -5.72e-07 | -1.81e-07 | -5.72e-08 | 3.16 |
| n_sub = 1.5 TE | -1.31e-03 | -4.14e-04 | -1.31e-04 | -4.15e-05 | 3.16 |
| n_sub = 1.5 TM | -1.26e-03 | -4.00e-04 | -1.26e-04 | -4.00e-05 | 3.15 / 3.16 |
| lossy Ag TE | +2.07e-04 | +6.50e-05 | +2.05e-05 | +6.48e-06 | 3.18 / 3.17 |
| lossy Ag TM | -2.42e-03 | -7.66e-04 | -2.42e-04 | -7.66e-05 | 3.16 |
| n_sub = 2.0 TE | -1.28e-05 | -4.06e-06 | -1.28e-06 | -4.06e-07 | 3.16 |
| n_sub = 2.0 TM | +5.76e-06 | +1.82e-06 | +5.76e-07 | +1.82e-07 | 3.16 |

`sqrt(10) = 3.162`.  The WP is right and the audit's "continuous limit to
O(delta^2)" is wrong; the WP's decision to report the measured law rather than
quote the brief is correct.  I also confirm the constant's own claim that
`_WOOD_DETECT` is the floor, by measurement rather than by reading: brackets of
1e-10 and 1e-11 return **exactly** the 1e-9 numbers, because the geometric
search grows `rel` until both sides clear `_WOOD_DETECT = 1e-9`.

**(c) The accuracy claim is fixture-specific.**  At the shipped bracket, against
my independent oracles at the exact wavelength:

| mount | pol | oracle R0 | one-sided (pre-fix rule) | symmetric (shipped) | gain |
|---|---|---|---|---|---|
| Moharam `Lambda = lambda` | tm | 0.155845785 | +6.31e-05 (4.05e-04 rel) | +1.04e-06 (6.7e-06 rel) | **60.5x** |
| Moharam | te | 0.040264629 | +5.46e-07 | -5.72e-08 | 9.5x |
| **substrate anomaly, n_sub = 1.5** | te | 0.240443122 | -2.47e-04 (1.03e-03 rel) | **-4.15e-05 (1.7e-04 rel)** | 5.96x |
| **substrate anomaly, n_sub = 1.5** | tm | 0.137471533 | -3.92e-04 (2.85e-03 rel) | **-4.00e-05 (2.9e-04 rel)** | 9.78x |
| lossy Ag ridge | te | 0.033385951 | +1.56e-04 | +6.48e-06 | 24.0x |
| lossy Ag ridge | tm | 0.929155563 | +8.02e-05 | **-7.66e-05** | **1.05x** |
| n_sub = 2.0, duty 0.35 | te | 0.009738758 | -1.47e-06 | -4.06e-07 | 3.62x |
| n_sub = 2.0, duty 0.35 | tm | 0.026703978 | +3.24e-06 | +1.82e-07 | 17.8x |

The fix is a genuine improvement on every mount I tried, but the gain ranges
1.05x-60.5x and the post-fix relative residual reaches **2.9e-04** -- the same
order as the 4.05e-04 that made H2 a P1 in the first place.  The WP's report and
changelog do scope their headline to "the Moharam mount", and the warning text
does too; the point for the orchestrator is that **the finding is mitigated and
announced, not eliminated**, and the CHANGELOG's "closer to the exact-wavelength
answer" is the honest description while "6.7e-06" is not a general figure.

**(c2) WHICH of the two levers actually pays -- and the WP's mechanism claim is
not general.**  The report and the changelog both say "the average buys a factor
**6.05** on the COEFFICIENT plus continuity; the other 10x comes from narrowing
the bracket".  I isolated the two levers by measuring, at a MATCHED bracket
`delta = 1e-7`, the one-sided error against the symmetric-average error:

| mount | pol | one-sided @1e-7 | AVERAGE @1e-7 | ratio (>1 = the average helps) |
|---|---|---|---|---|
| Moharam | te | +5.458e-07 | -5.723e-07 | 0.95 |
| **Moharam** | **tm** | +6.305e-05 | +1.043e-05 | **6.05** (the WP's number) |
| n_sub = 1.5 | te | -2.472e-04 | -4.145e-04 | 0.60 |
| n_sub = 1.5 | tm | -3.915e-04 | -3.996e-04 | 0.98 |
| lossy Ag | te | +1.556e-04 | +6.499e-05 | 2.39 |
| **lossy Ag** | **tm** | +8.015e-05 | -7.655e-04 | **0.10** |
| n_sub = 2.0 | te | -1.471e-06 | -4.061e-06 | 0.36 |
| n_sub = 2.0 | tm | +3.236e-06 | +1.822e-06 | 1.78 |

At matched bracket the averaging helps on **3 of 8** mount/polarization pairs
and hurts on 5, by up to **10x** (lossy Ag TM).  The 6.05x is its best case, and
it is the WP's own fixture.  What DOES pay reliably is the 100x narrower
bracket, which the measured `sqrt(delta)` law converts into a guaranteed ~10x on
every mount; the averaging then modulates that by 0.10x-6.05x, which is why the
NET gain ranges 1.05x-60.5x.

This does not change the verdict -- the shipped configuration is a net
improvement everywhere I measured, and CONTINUITY (which no one-sided rule can
have, and which I verified in 1-D and 2-D) is a property of the average alone,
not of the bracket width.  But the mechanism sentence should be restated: the
bracket is the lever, the average is what makes the result a limit rather than a
spike.  Recorded as V14.

**(d) The exactly-grazing-order artefact is larger than the WP states.**
Census over the same four mounts at the shipped 1e-9 bracket, on every order the
oracle returns EXACTLY 0 for:

| mount | pol | order | library value | specular | decades below |
|---|---|---|---|---|---|
| Moharam | te | R(+/-1) | 3.963e-07 | 4.027e-02 | 5.01 |
| Moharam | tm | R(+/-1) | 2.344e-06 | 1.558e-01 | 4.82 |
| Moharam | tm | T(+/-1) | 4.376e-06 | 8.441e-01 | 5.29 |
| n_sub = 1.5 | tm | T(+/-1) | 3.816e-05 | 8.625e-01 | 4.35 |
| **lossy Ag** | te | R(+/-1) | 1.205e-05 | 3.339e-02 | **3.44** |
| **lossy Ag** | tm | T(+/-1) | 4.004e-06 | 7.056e-03 | **3.25** |
| n_sub = 2.0 | tm | T(+/-1) | 4.989e-07 | 9.733e-01 | 6.29 |

The WP's regression test is named
`test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular` and asserts
`R[k] < 1e-4 * R[M]`.  That bar is **violated on the lossy Ag mount**
(3.61e-04); the test passes only because it runs on the Moharam fixture.  The
gate is fine as a pin; its NAME and its docstring generalise a fixture property.
Recommend re-scoping both (section 6, item V3).

The **closure bookkeeping is correct**, which is the substantive half: the power
genuinely comes out of the specular order.  Measured closure at every bracket
width 1e-6..1e-11: 1e-14-level on the three lossless mounts (max 5.8e-14) and
exactly the absorptance on the lossy one (-3.2e-02 TE / -6.4e-02 TM, matching
the oracle's -3.17e-02 / -6.38e-02).

**(e) 2-D crossed grating (no oracle exists -- continuity and closure instead).**
Square pillar, eps 4 in 1, period = 1 um, depth 0.3 um, TE, `li`, 3x3 orders,
sweep `delta = -1e-6 .. +1e-6` through `Lambda = lambda`:

* R0 **strictly monotone** across all 11 samples, and the anomaly value lies
  strictly between its neighbours;
* closure `|sum R + sum T - 1| <= 2.2e-15` at every sample INCLUDING the anomaly;
* exactly one `WoodNudgeWarning`, at delta = 0 only;
* `|R0(0) - mean(R0(-d), R0(+d))|` = 1.09e-06 / 2.26e-06 / 4.55e-06 for
  d = 1e-8 / 3e-8 / 1e-7 -- the sqrt(delta) residual, not a jump;
* on `rcwa_jones_2d` the C4 identity survives the average: `|Jxx - Jyy|` =
  1.30e-15 AT the anomaly (2.5e-15 / 3.8e-16 at the neighbours).

**(f) PMM gets bit-identical numbers.**  I transcribed the pre-WP body of
`_grazing_safe_wavelength` from `e1bf79be~1` and compared over **4000
randomised calls** (random truncation, periods over two decades, random
permittivity lists, half of them placed EXACTLY on an anomaly; 342 actually
nudged): `max|new - old| = 0.0`.  End to end, `pmm_efficiency_1d` at an exact
anomaly returns `R`/`T` **bit-identical** to an explicit `wl*(1+1e-7)` solve, so
PMM still receives the one-sided nudge with the old step.  The only change PMM
sees is the warning, and the PMM suites pass.

**(g) Interactions I probed and their results.**

* `stabilize=True` at an exact anomaly: identical numbers to `stabilize=False`
  (R0 0.155846827297 both), warnings NOT swallowed
  (`_stabilize_closure_failure` re-emits every non-`_EnergyWarning`).
* `RCWAStack.solve(retain_internal=True)` keeps the one-sided nudge by design and
  therefore returns **`max|dR| = 5.8e-05` different** from `solve()` on the same
  stack at the same wavelength (1.2e-03 on a 2-D pillar fixture).  Documented,
  but two public calls on one object disagree; the `retain_internal` warning does
  not say so.  Suggest one clause in that warning (section 6, V4).
* A strict caller promoting `WoodNudgeWarning` to an error gets a clean raise.
* `pmm_efficiency_1d` emits TWO `WoodNudgeWarning`s per call (two nudge call
  sites) and `RCWAStack.solve(stabilize=True)` emits five (one per ladder rung
  plus the re-emission).  Cosmetic noise, not wrong.
* **Not a regression, but worth recording:** under `jax.jit` with a TRACED layer
  index, `rcwa_efficiency_1d` at an exact Wood anomaly returns **NaN** -- and it
  does so with the pre-fix one-sided value forced in (`_wl_eff = wl*(1+1e-7)`),
  and only at the anomaly (off-anomaly `jit` is fine and matches eager to 1e-16).
  So WP-A14 did not cause it.  `test_rcwa.py::test_jax_wood_anomaly_no_nan` runs
  EAGERLY and does not cover `jit`.  Open item V5.

**(h) The audit's "keep intact" list, re-run.**

* `repro/RCWA-EME-BOR/p02_oracle.py` (24 configurations): **22 agree to
  <= 1.50e-13** (audit bar 2.0e-13).  The two exceptions are exactly the two
  Wood rows: `case1 te ang=0` max|dR| 3.963e-07 (audit pre-fix 5.46e-07, 1.4x
  better) and `case1 tm ang=0` max|dR| 2.344e-06 (audit pre-fix 6.31e-05, **27x
  better**).  Note the WP quoted only max|dR|: the TM max|dT| at that row is
  **1.448e-05**, which is the largest single post-fix deviation anywhere in the
  sweep.
* `p01_energy.py` (12 configurations): closure `<= 1.03e-13` (audit bar 1.4e-13).
* `p06_jones_tmm.py`: TMM parity in amplitude AND phase, residuals
  5.6e-17 .. 3.3e-16 (the "flipped" column is the documented sign convention).
* `validation/run_all.py test_rcwa`: **PASS** (2.1 s).

### H3 -- the symmetrised `fff_nv` Jones operator

**Reproduced** on the WP's own fixtures.  Three things the WP did not measure:

**(a) A cell that is NOT transpose-symmetric.**  Three axis-aligned fixtures
(off-centre 24x56 rectangle, 12x72 rectangle, two-bar union), reference `li` at
M = 16:

| fixture | M | \|mean - ref\| | \|L2L1 - ref\| | \|li - ref\| | \|mean - L2L1\| |
|---|---|---|---|---|---|
| rect 24x56 | 4 | 4.01e-03 | 3.34e-03 | 5.30e-03 | 8.28e-04 |
| | 8 | 3.03e-04 | 2.55e-04 | 6.81e-04 | 2.36e-04 |
| | 12 | **7.54e-05** | 1.21e-04 | 2.78e-04 | 1.22e-04 |
| rect 12x72 | 4 | 8.27e-03 | 6.79e-03 | 1.00e-02 | 1.49e-03 |
| | 12 | 2.10e-04 | 1.96e-04 | 4.48e-04 | 1.56e-04 |
| two-bar union | 4 | 2.67e-03 | 2.29e-03 | 3.60e-03 | 5.98e-04 |
| | 12 | **9.05e-05** | 1.01e-04 | 2.18e-04 | 6.49e-05 |

Both orders converge to the same limit, the mean is equal or slightly BETTER at
every M, and their difference falls ~1/M.  The transpose argument does not
damage the general case.

**(b) Both axes, non-square raster and Mx != My.**  On a 120x96 cell with
periods 0.6 / 0.4 um and Mx != My the symmetrisation is still exact where
Li-2003 reduces to Li-1996: the **y-uniform** stripe matches `li` to 4.14e-14
and the **x-uniform** stripe -- which only the TRANSPOSED leg can get right --
to 8.68e-14.  Dtype/layout adversarials (real `float64`, Fortran order,
`complex64` input) all return bit-identical Jones matrices.

**(c) The scope notice's false-positive rate on axis-aligned cells.**  65
Manhattan fixtures (60 random rectangles, a cross, an L, two disjoint squares, a
1-pixel stripe, an 8-pixel checkerboard): **1 fires** (1.5%) -- the checkerboard,
`frac = 0.0667` against the 0.06 threshold.  Median `frac` 0.0097, max over the
other 64 is 0.0397.  True positives: disk 0.1779, ellipse 0.1391, 45-degree bar
0.8641.  Well separated; the one false positive is a pathological corner-density
fixture where a staircase caveat is arguably right anyway.

The notice fires for `fff_nv` only (0 for `laurent`/`li`), including at oblique
incidence, and is silenced by `allow_nonseparable_nv=True`.

**Gap:** it does NOT fire on an OUT-OF-PLANE (3x3) tensor cell -- the call sits
inside the `if not offplane:` block -- and that path still runs the
UN-symmetrised `_li_convolutions_2d_tensor_full`.  The WP defers the off-plane
symmetrisation (D3) but does not record that the scope notice is missing there
too.  Open item V6.

**Migration-note gap:** the changelog says the separable stripe is unchanged.  It
does not say that on a GENERAL axis-aligned cell the shipped default now differs
from the pre-fix `fff_nv` Jones by ~1.2e-04 at M = 12 (inside the truncation
error, same limit).  Worth one sentence.

**`RCWAStack._li_blocks` is numerically UNCHANGED**, which is the claim that
matters for the `symmetrize=False` carve-out: I verified TEXTUALLY that the
pre-fix `_li_convolutions_2d_tensor` body and the new `_li_tensor_l2l1` body are
**byte-identical** (docstrings stripped, `str.strip()` comparison of the two
extracted bodies -> `True`), so `symmetrize=False` is the historical operator bit
for bit.  Separately I could not reproduce that method's PRE-EXISTING docstring
claim that the companion-pair path reduces to `_li_convolutions_2d` with `Cyy`
"to 4.2e-16" -- measured `Cxx` 0.0 (bit-identical, as claimed) but `Cyy`
4.5e-02.  Out of scope here (unchanged code); recorded as V13.

### H4 -- performance items

**(a) The even-parity fold.**  Re-measured independently, 5 interleaved runs,
`perf_counter` medians, `tracemalloc` peaks, 96x96 square cell:

| formulation | M | t_sym | t_full | speedup | max\|dJ\| | peak MB sym / full |
|---|---|---|---|---|---|---|
| laurent | 6 | 127.2 ms | 397.0 ms | 3.12x | 2.231e-13 | 31.5 / 46.6 |
| laurent | 9 | 914.3 ms | 3186.4 ms | 3.48x | 3.378e-13 | 137.3 / 206.6 |
| li | 6 | 139.2 ms | 439.0 ms | **3.15x** | 1.079e-13 | 32.0 / 46.6 |
| li | 9 | 897.2 ms | 2937.7 ms | **3.27x** | 7.364e-14 | 139.3 / 206.6 |
| fff_nv | 6 | 169.5 ms | 445.5 ms | **2.63x** | 1.184e-13 | 32.0 / 46.6 |
| fff_nv | 9 | 1201.4 ms | 3559.1 ms | **2.96x** | 2.164e-13 | 139.3 / 206.6 |

The `max|dJ|` column is **bit-identical** to the WP report's table, which is
strong evidence those numbers are real measurements rather than estimates.  The
timings differ by <= 20% from the WP's, consistent with a shared machine.

Engagement conditions, which is what the brief asked me to break:

* at `theta = 20 deg`, `_symmetric_cascade_rt` is **not called at all** for
  `laurent` / `li` / `fff_nv`, with `symmetry='auto'` AND with an explicit
  `symmetry=True` (so an explicit request cannot force the wrong basis);
* on a NON-centro-symmetric cell the fold is attempted once, returns `None`, and
  the fall-through is **bit-identical** to `symmetry=False` for all three
  formulations (max|dJ| = 0.0, `np.array_equal` True).

Gated by `::test_h4_even_fold_does_not_engage_at_oblique_incidence`.

**(b) The isotropic single-build guard.**  1 `_li_convolutions_2d` call for an
isotropic cell, **2** for a cell whose `eyy` differs from `exx` in ONE pixel by
one part in 1e-9 (and the two answers differ, so the distinction is not
cosmetic).  The guard is identity-then-exact-equality, not a tolerance -- which
is the right shape, because the resulting Jones difference is only ~2e-12 and a
tolerance guard would silently build `Cyy` from `exx`.  Gated by
`::test_h4_isotropic_single_build_guard_sees_a_one_pixel_difference`.
Note: the WP's own gate for this
(`test_h4_isotropic_cell_builds_the_li_operators_once`) ends with
`a = orig(...); b = orig(...); assert array_equal(a, b)` -- the SAME call twice,
which is a tautology for a deterministic function and does not check what its
comment claims.  Item V2.

**(c) `eigh(A, M)` for the BOR pencils.**  Relative error against the Bessel
zeros reproduces the WP's table digit for digit: 1.191e-13 (m = 0 D),
9.259e-14 (1 D), 1.377e-14 (3 D), 1.741e-13 (1 N), 3.808e-14 (3 N).
A **288-configuration robustness sweep** (degree 2..14 x n_el 1..24 x
m in {0,1,5} x both bc) produced **0 failures**, so `eigh`'s SPD requirement
never refuses where the old `eig(solve(M, A))` did not.

The eigenvector normalisation change is real and I pinned it against an oracle
the library cannot produce: for the unit disk with Dirichlet bc the exact modes
are `c_n J_m(j_{m,n} r)` with `INT_0^1 r psi^2 dr = c_n^2 J_{m+1}(j)^2/2`, so an
`M`-orthonormal column is `sqrt(2)/|J_{m+1}(j_{m,n})| J_m(j_{m,n} r)` exactly.
Measured relative profile error 1.00e-13 / 2.17e-13 / 3.85e-12 / 4.08e-11 for
modes 0-3, with 2-norms 14.04 / 15.88 / 16.94 / 17.67 (so definitively NOT unit
2-norm).  `vecs[0, :] == 0.0` exactly for m != 0.

**Both** existing consumers are scale-INVARIANT
(`test_radial_eigensolver::test_eigenfunctions_match_bessel_profiles`
least-squares-fits a scale; `test_niche_audit_w6_bor::…_rdr_orthonormal_and_regular`
normalises the Gram), so nothing pinned this public scale change in either
direction.  Gated now by
`::test_h4_radial_spectrum_modes_are_M_orthonormal_against_bessel`.

### H5 -- the inert BLAS cap

Verified by direct probe: one warning on two calls (once per process), text
contains `INERT`, `400x`, `140x`, `2.29 s`, `18.2 s`, `threadpoolctl` and all
three environment variables.  `threadpoolctl` is confirmed absent from BOTH
`pyproject.toml` (`dependencies` = numpy/scipy/matplotlib/psutil) and
`requirements.txt`, so the WP's requested line (report section 5a) is still
outstanding for the tests/CI package.

### H6 -- aliasing, the passive bar, docs

* **`per_order_amplitudes`**: zeroing every array key of the returned dict and
  re-calling returns the original arrays bit for bit, on `reflection` AND
  `transmission` (the WP tested one port); `res.orders` is not poisoned.
  Keys covered: `Ex, Ey, kx, ky, kz, orders`.
* **The passive bar**: `_passive_media`'s three clauses reproduce.  False-positive
  census: 60 randomised absorbing dielectric pillars under a lossless incidence
  medium (random size/offset/`n_orders`/period/depth) -> **0 warnings**; 20 Ag
  grating solves (`n_orders` 5..61 x te/tm x li/laurent) -> **0 warnings**; the
  `stabilize` ladder returns the same truncation and the same R0 as before.  The
  bar cannot be reached from a real under-resolved metallic 2-D pillar because
  the pre-existing 1.05 hard tripwire fires first (60/60 raised `_EnergyError`
  on a strongly metallic fixture) -- which is the correct ordering, and means
  the new clause covers the (1 + 1e-6, 1.05] window and nothing else, as claimed.
* **M8 docstring**: the restated claim carries the audit's own measured
  `O(1/Sx^2)` table; the scoping is correct.
* **EME `ref_2d_modes`**: `max|Im lam| = 0.0` **exactly** at
  (kx0, ky0) = (0, 0), (0.37, 0) and (0.37, -0.9), i.e. the FD oracle is now
  exactly Hermitian at non-zero Bloch phase, which is the stated point.
  Bit-identity at kx0 = ky0 = 0 is algebraic (`p = 1 + 0j`, `1/p == conj(p)`
  bit for bit).
* **EME dead `isrealobj` arm**: `A` is `np.zeros(..., dtype=complex)`
  unconditionally, so the removed arm was unreachable; 110 EME/BOR tests pass.
* **Defect found (V11)**: see section 3.  The 1-D passive guard's argument
  construction raised on an input the solver had already solved correctly.

### G11 -- `_sqrt_decay`'s on-cut predicate

**The conjunct IS the propagating test.**  `Im(r)^2 > Re(r)^2` is algebraically
identical to `Re(r^2) < 0`, i.e. to `Re(lam^2) < 0`.  So the brief's challenge
-- "construct a genuinely on-cut PROPAGATING mode the new conjunct wrongly
rejects" -- has no solution by construction: a propagating mode satisfies the
conjunct.  The only excluded boundary is `Re(lam^2) == 0` EXACTLY (`lam^2` purely
imaginary), where the principal root already has `Re(lam) = sqrt(|eta|/2) > 0`
and is decaying, so keeping it is correct.  The new discontinuity surface the
conjunct introduces (`|Im r| = |Re r|` inside the band) carries a jump of at most
`2 sqrt(2) * band * scale ~ 2.8e-08 * scale`, two decades smaller than the
pre-existing jump at the band edge.

**Measured, not argued.**  5000 randomised draws spanning 12 decades of
`|lam^2|`, with a quarter placed ON the cut (`-s + i eta`, eta at backward-error
level) and a quarter in the near-zero EVANESCENT family the audit's
counterexample comes from:

* the flip set is **exactly** `band & (Im < 0) & (Im^2 > Re^2)` (`array_equal`);
* every flipped root satisfies `Re(r)^2 - Im(r)^2 < 0` with **no tolerance**;
* the round-1 (band-only) predicate is a **strict superset** (it would have
  flipped more), and every entry it drops has `Re^2 >= Im^2` -- so nothing on
  the cut was lost;
* both outcomes occur, so the identity is not vacuous.

**JAX twin**: identical flip sets over 1200 values, max value difference
**8.04e-14** (XLA `sqrt` vs NumPy `sqrt` in the last bits -- the same 8.04e-14
separates the raw square roots, so it is not this predicate's doing).  CuPy is
not installed; the body is `xp.real`/`xp.imag` arithmetic with no host branch,
desk-checked.

Both gated by `::test_g11_flip_set_is_exactly_the_propagating_subset_of_the_band`
and `::test_g11_numpy_and_jax_bodies_select_the_same_flips`.

---

## 3. Defects I found and fixed (in WP-A14's own files)

### V1 -- `guided_modes` still returned a SILENT `[]` through the reldiv / tail screens

**File**: `lumenairy/elements/bor/coupled_radial_eigensolver.py` (the
`guided_modes` filter loop and its docstring).

**What was wrong**: the notice WP-A14 added was armed only when the GUARD BAND
was the rejecting filter (`near` was assigned inside the band branch).
`guided_modes` has three rejecting filters, and on a high-contrast fibre or one a
little above its next mode's cut-off it is the `tail_tol` radiation screen that
empties the list -- with no signal at all.  Measurements in section 2 (H1):
Si/SiO2 dn = 2.04 at V = 2.0 and SiN/SiO2 dn = 0.56 at V = 2.0 both returned
`[]` with **0 warnings** while the exact hybrid oracle has a guided HE11 (the
SiN raw spectrum is 4.6e-05 from it).

**What I changed**: every filter now records `(mode, why)` for the closest
in-window candidate it rejects, and the notice names both the mode and the
REJECTING SCREEN ("the SPURIOUS-mode screen" / "the RADIATION screen ... Rbig is
most likely too small for this mode's cladding tail" / "inside the ... guard
band").  No numeric behaviour changes: the three filter predicates are
byte-identical (the tail comparison is the same expression, only bound to a local
`float` first), and the returned list is unchanged.  The docstring now states
that there are three doors and that only the band is about contrast.

**Verification**: the five fixtures above now emit exactly one warning naming the
screen; the populated cases stay quiet (legacy `e1 = 6, e2 = 2` at N = 150/300/600
and the weakly-guiding V = 2.4 fibre return their modes with 0 notices).  Gated
by `::test_h1_an_empty_guided_mode_list_is_never_silent` (3 parametrised arms,
each with the exact oracle asserting a mode EXISTS so the gate cannot go vacuous)
and `::test_h1_a_populated_guided_mode_list_stays_quiet`.

### V11 -- the 1-D passive energy guard RAISED on an input the solver accepted

**File**: `lumenairy/elements/rcwa/oned.py:795-806` (`rcwa_efficiency_1d`'s
`_check_energy` call).

**What was wrong**: the new `passive=` argument was built as
`_passive_media(complex(eps_sup), complex(eps_sub),
np.array([_C(n_ridge) ** 2, _C(n_groove) ** 2]))`.  The `np.array([...])` pack is
evaluated in the ARGUMENT expression, outside `_passive_media`'s own
`except (TypeError, ValueError)`, so whenever the two layer indices have
different shapes the whole call raises.  Measured on HEAD:
`rcwa_efficiency_1d(0.5e-6, np.array([2.04+0j]), 1.0, 1.5, 1.0, 0.3e-6, 0.5,
0.633e-6, n_orders=5, polarization='tm')` raised
`ValueError: setting an array element with a sequence ... inhomogeneous shape`
from `oned.py:798` -- i.e. AFTER a complete and correct solve, from the guard.
The same call with `n_groove = np.array([1.0+0j])` succeeded, and the scalar
form has always worked, so the failure is purely the pack.  It is new in
`e1bf79be` (the pre-fix line was `_check_energy(..., lossless=lossless)` with no
array construction).

**What I changed**: pass the two permittivities as separate `*eps_arrays`
entries.  `_passive_media` already screens each independently -- the max over the
pack and the max over the parts are the same boolean -- and it carries its own
`TypeError`/`ValueError` fallback.

**Verification**: all four spellings (`scalar/scalar`, `1-elem/scalar`,
`1-elem/1-elem`, `0-d/0-d`) now return and return the SAME number bit for bit
(R0 = 0.046731048); the predicate is identical packed-vs-separate on all four
clause combinations of `_passive_media` (real / lossy layer / gain layer / gain
substrate).  Gated by
`::test_h6_passive_guard_does_not_break_array_valued_indices`.

### V0 -- coordinator request: mirror the PMM `angle`/`theta` mismatch notice

**Files**: `lumenairy/elements/rcwa/oned.py::_resolve_incidence` (which
`rcwa/stack.py:2293` imports, so `RCWAStack.set_source` is covered by the same
change) and the `RCWAStack.set_source` docstring.

WP-A12 made `pmm/_core.py::_resolve_incidence` WARN when `angle=A` and `theta=T`
are both supplied with `A != T` (the alias-wins rule silently solved `T`).
`tests/unit/test_v5_12_0_naming_aliases.py::test_set_source_theta_wins_consistent_across_suites`
exists to pin that the two resolvers agree, so the notice belongs on both.
Implemented as requested: **resolution unchanged** (`theta` still wins,
bit-for-bit), same category (`UserWarning`), same wording with the `rcwa:`
prefix, same `angle != 0` gate plus an `angle is None` arm for
`RCWAStack.set_source`'s "not supplied" default.  Traced JAX angles skip the
comparison (the `float()` raises a `TypeError` subclass, which is caught);
CONCRETE JAX angles are compared and do warn.

Verified: `_resolve_incidence(0.9, 0.25)` -> `0.25` + one warning naming both;
`(0.0, 0.25)`, `(0.25, 0.25)`, `(0.9, None)`, `(None, 0.25)`, `(None, None)` ->
silent; end to end on `rcwa_efficiency_1d` the both-supplied result is
`array_equal` to the `theta=`-only result and NOT to the `angle=`-only one;
`RCWAStack.set_source(0.633e-6, angle=0.7, theta=0.25)` warns and stores 0.25.
`tests/unit/test_v5_12_0_naming_aliases.py` and the RCWA batch pass (149 passed).
Gated by `::test_conflicting_angle_and_theta_is_audible_on_the_rcwa_side_too`.
(A future coordinated change may turn both into a raise; not here.)

---

## 4. Tests I added

`tests/unit/test_audit2609_a14_verify.py` -- **25 tests, all passing** (89 s).

| test | what it pins |
|---|---|
| `test_conflicting_angle_and_theta_is_audible_on_the_rcwa_side_too` | V0: the RCWA mirror of the PMM notice; resolution unchanged; five silent arms |
| `test_wood_symmetric_reentry_forwards_every_argument` (8 arms) | H2: `f(x) == mean(f(x, _wl_eff=lo), f(x, _wl_eff=hi))` EXACTLY, over positional / keyword / ASR / tensor / `symmetry` / `allow_nonseparable_nv` |
| `test_wood_symmetric_reentry_forwards_stack_arguments` | H2: the same for the `RCWAStack._solve_once` METHOD (`self` carried) |
| `test_h2_symmetric_average_beats_the_one_sided_nudge_off_the_wp_fixture` (3 arms) | H2: against my `expm` oracle on a SUBSTRATE anomaly and a LOSSY Ag mount; bars derived from the measured 5.7x-24x family, not from the WP's fixture |
| `test_g11_flip_set_is_exactly_the_propagating_subset_of_the_band` | G11: set identity on 5000 randomised draws; no-tolerance propagating claim; round-1 strict superset |
| `test_g11_numpy_and_jax_bodies_select_the_same_flips` | G11: cross-backend flip-set identity, value bar 1e-12 (measured 8.0e-14) |
| `test_h4_even_fold_does_not_engage_at_oblique_incidence` (3 arms) | H4: the generalised fold did NOT loosen the normal-incidence precondition |
| `test_h4_isotropic_single_build_guard_sees_a_one_pixel_difference` | H4: the guard is exact equality, not a tolerance |
| `test_h1_an_empty_guided_mode_list_is_never_silent` (3 arms) | V1: the empty list is audible through EVERY screen, with the exact oracle asserting a mode exists |
| `test_h1_a_populated_guided_mode_list_stays_quiet` | V1's other side: no noise on ordinary calls |
| `test_h4_radial_spectrum_modes_are_M_orthonormal_against_bessel` | H4: the public eigenvector SCALE, against the analytic Bessel normalisation (nothing pinned it before) |
| `test_h6_passive_guard_does_not_break_array_valued_indices` | V11: the passive guard must not RAISE on an input the solver accepted; four index spellings return bit-identically; packed == separate on all four `_passive_media` clauses |

Every bar carries its oracle, the oracle's floor, the measured value and the
decades of gap, per `docs/TESTING_STANDARDS.md`.  No wall-clock assertion, no
`pytest.skip` on a resource precondition, no per-build census.

---

## 5. Audit of the WP's own tests against `docs/TESTING_STANDARDS.md`

`tests/unit/test_audit2609_a14_rcwa_eme_bor.py` (32 tests, re-run: **32 passed,
29.6 s**) is largely exemplary -- oracles are written out rather than imported,
fail-before behaviour is RE-MEASURED in-test through the private
`symmetrize=False` switch rather than quoted, and the bars carry derivations.
Three weaknesses:

* **V2 (vacuous assertion)** --
  `test_h4_isotropic_cell_builds_the_li_operators_once` ends with
  `a = orig(ex, orders, 4, 4, np); b = orig(ex, orders, 4, 4, np);
  assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])` under the
  comment "one call and two calls return the SAME blocks".  That is the same call
  twice -- a tautology for a deterministic function -- and it does not compare
  the one-call path against the two-call path.  My
  `::test_h4_isotropic_single_build_guard_sees_a_one_pixel_difference`
  covers the real property; the vacuous block should be deleted or replaced.
* **V3 (over-general name/bar)** --
  `test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular` asserts
  `R[k] < 1e-4 * R[M]`.  That holds on the Moharam fixture (4.82-6.21 decades)
  but NOT generally: measured 3.44 decades (lossy Ag TE, R) and 3.25 decades
  (lossy Ag TM, T).  Rename to name its mount, or restate the bar as an absolute
  one (`< 1e-4` absolute, which does hold across my census: worst 3.8e-05).
* **V7 (trivial assertion)** -- `test_h4_bor_pencil_eigh_accuracy`'s
  `assert np.all(np.isreal(ev))` is trivially true because
  `eigh(..., eigvals_only=True)` returns a real dtype; it cannot fail even if the
  code reverted to `eig(...).real`.  Assert the dtype or the M-orthonormality
  instead.

The two restated third-party tests are sound:
`test_fix_branch_cut_round2.py::test_the_shared_body_narrows_the_round_one_SELECTOR_to_the_cut`
genuinely strengthens what it replaced (it adds a subset claim plus a
non-vacuity guard), and the two `_check_energy` stubs now take `**kw` with the
lossless-tripwire one RECORDING what it forwards.

---

## 6. Open items for the orchestrator

**STATUS after the orchestrator's rulings (see section 9): V1-residual, V5,
V14, V3, V2, V7, V6, V13, V10 and V12 are all CLOSED in this branch.  V8 was
reassigned to VERIFY-A13, H5 to WP-A15a, and the multibranch `RuntimeWarning`
to WP-A3.  Nothing in this table is still owed by WP-A14.**

| id | severity | item |
|---|---|---|
| **V1-residual** | **medium** | `guided_modes` still returns a PARTIAL result silently: Si/SiO2 (dn = 2.04) at V = 4.0 returns 1 mode where the exact oracle has 3 (3.038391486 / 1.566626407 / 1.440009396), with no notice.  Closing it needs a mode-count comparison (or a documented `Rbig`/`N` adequacy check), not just an empty-list guard.  H1's headline -- "any fiber study at realistic index contrast gets 'no guided modes'" -- is mitigated, not closed. |
| **V2** | low | delete/replace the vacuous tail of `test_h4_isotropic_cell_builds_the_li_operators_once` (section 5). |
| **V3** | low | re-scope `test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular` (section 5): the "four decades below the specular" property is fixture-specific (measured 3.25 decades on a lossy Ag mount). |
| **V4** | low | `RCWAStack.solve(retain_internal=True)` returns numbers `5.8e-05`..`1.2e-03` different from `solve()` at an exact anomaly.  It is documented in the report; add one clause to the `retain_internal` warning text so the user reading it knows the DEFAULT path would have answered differently. |
| **V5** | **medium**, NOT WP-A14 | under `jax.jit` with a traced layer index, `rcwa_efficiency_1d` at an exact Wood anomaly returns **NaN**; forcing the PRE-FIX one-sided wavelength reproduces it, and off-anomaly `jit` is clean, so this predates WP-A14.  `test_rcwa.py::test_jax_wood_anomaly_no_nan` runs eagerly and does not cover `jit`.  Worth its own finding. |
| **V6** | low | `_li_tensor_scope_notice` is inside `if not offplane:`, so an OUT-OF-PLANE (3x3) tensor cell with a curved pattern gets `fff_nv` with NO validated-scope notice AND the un-symmetrised `_li_convolutions_2d_tensor_full` (the WP's deferred D3).  Record the notice gap alongside D3. |
| **V7** | low | `test_h4_bor_pencil_eigh_accuracy`'s `np.isreal` assertion is trivially true (section 5). |
| **V8** | low, PMM WP's file | the PMM entry points now WARN through the shared nudge, but their `Efficiency2D` constructions (`pmm/twod.py:1167, 1340, 1385, 1515, 1552, 1674`, `pmm/twod_staggered.py:3641`) never pass `wl_eff`, so `pmm_efficiency_2d(...).wl_eff is None` even when the wavelength WAS substituted.  H2's "surface `wl_eff` on the result" is therefore only half-delivered across the suites. |
| **V9** | low | `pmm_efficiency_1d` emits TWO `WoodNudgeWarning`s per call and `RCWAStack.solve(stabilize=True)` emits five.  Cosmetic; a `warn=False` on the inner call site (the kwarg already exists) would fix the PMM one. |
| **V10** | low, docs | the H2 changelog's headline residual (`6.7e-06 relative`) is the Moharam mount's; measured up to `2.9e-04` on a `n_sub = 1.5` mount.  The H3 changelog's migration note should also state the ~1.2e-04 (M = 12) change on a GENERAL axis-aligned `fff_nv` cell, not only that the stripe is unchanged. |
| **H5 (still open)** | low | `threadpoolctl` remains absent from `pyproject.toml` and `requirements.txt`; the WP's exact requested lines (its report section 5a) are still owed by the tests/CI package. |
| **V14** | **medium, docs** | the H2 report/changelog sentence "the average buys a factor 6.05 on the COEFFICIENT plus continuity; the other 10x comes from narrowing the bracket" is a ONE-FIXTURE result.  Measured at a matched `delta = 1e-7` bracket on 8 mount/polarization pairs, the averaging helps on 3 and HURTS on 5, by up to 10x (lossy Ag TM: one-sided +8.0e-05 vs average -7.7e-04).  The reliable lever is the 100x narrower bracket (a guaranteed ~10x through the verified `sqrt(delta)` law); the average's contribution ranges 0.10x-6.05x, and what it buys unconditionally is CONTINUITY, not accuracy.  Restate the mechanism (the numbers themselves and the verdict are unaffected). |
| **V13** | low, PRE-EXISTING (not WP-A14) | `RCWAStack._li_blocks`' docstring claims the companion-pair path "reduces to `_li_convolutions_2d` EXACTLY when both companions are the cell (measured: `Cxx` bit-identical, `Cyy` to 4.2e-16)".  I reproduce `Cxx` bit-identically (0.0) but measure **`max|Cyy - _li_convolutions_2d(eyy)[1]| = 4.5e-02`** on a 64x64 rectangle cell at `n_orders` 4x3.  This is NOT caused by WP-A14: I verified textually that the pre-fix `_li_convolutions_2d_tensor` body and the new `_li_tensor_l2l1` body are **byte-identical**, and `_li_blocks` passes `symmetrize=False`, so its numbers are unchanged.  Either the docstring's measurement used a fixture where the truncated composite does coincide, or the claim needs re-scoping; worth its own look by whoever owns `stack.py` next. |
| **V12** | low, docs | the WP's H2 table quotes `max|dR|` only.  Re-running `p02_oracle.py` shows the largest post-fix deviation anywhere in the 24-configuration sweep is the Moharam TM row's **`max|dT| = 1.448e-05`**, not the 2.34e-06 in `max|dR|`.  Worth adding so the changelog's "closer to the exact-wavelength answer" is not read as "within 2e-06". |

Nothing in this verification asks for a change to another WP's SOURCE; V5 and V8
are observations about files I do not own, recorded for the orchestrator.

---

## 6a. Files I touched

Source (all inside WP-A14's ownership, `rcwa/*` and `bor/*`):

* `lumenairy/elements/rcwa/oned.py` -- `_resolve_incidence` (V0: the PMM mirror,
  resolution unchanged) and the `rcwa_efficiency_1d` passive-guard argument
  (V11: two separate `eps_arrays` instead of one pack).
* `lumenairy/elements/rcwa/stack.py` -- `RCWAStack.set_source` DOCSTRING only
  (records the V0 notice; it imports `_resolve_incidence` from `oned.py`, so the
  behaviour change reaches it through that one edit).
* `lumenairy/elements/bor/coupled_radial_eigensolver.py` -- `guided_modes`'
  empty-result notice now covers every filter, and the docstring says so (V1).
  No numeric behaviour change: the three filter predicates are byte-identical.

Tests:

* NEW `tests/unit/test_audit2609_a14_verify.py` (25 tests).

Docs:

* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-A14.md`
  (this file).

No edits to `CHANGELOG.md`, `README.md`, `Migration-Guide.md`, `CONVENTIONS.md`,
`pyproject.toml`, `lumenairy/__init__.py`, `pmm/`, or to WP-A14's own report,
changelog or gate file.

**Changelog text for the orchestrator** (my three changes, in the repository's
voice; the WP's own `WP-A14_CHANGELOG.md` is untouched):

```
### Fixed -- BOR: an empty `guided_modes` result is no longer silent on ANY filter (H1, VERIFY-A14)

The 2026-09-12 H1 fix made the guard band contrast-invariant and warned when the
BAND emptied the list.  `guided_modes` has three rejecting filters, and on a
high-contrast fibre or one a little above its next mode's cut-off it is the
`tail_tol` radiation screen that empties it -- measured on Si/SiO2 (dn = 2.04)
at V = 2.0, which returned `[]` with NO warning while the exact hybrid oracle has
the HE11 at n_eff = 1.846898458 and the raw spectrum held 1.882075103; and on
SiN/SiO2 at V = 2.0, where the raw mode is 4.6e-05 from the exact HE11.  The
notice now records the closest in-window candidate rejected by ANY filter and
names the screen that rejected it ("the RADIATION screen ... Rbig is most likely
too small for this mode's cladding tail").  No numeric change: the three filter
predicates are byte-identical and the returned list is unchanged.

### Fixed -- RCWA: `angle=A, theta=T` with `A != T` now warns, as it already does in PMM (VERIFY-A14)

`rcwa/oned.py::_resolve_incidence` (shared with `RCWAStack.set_source`) mirrors
the notice WP-A12 added to `pmm/_core.py::_resolve_incidence`: two DIFFERENT
non-zero angles in one call is a caller mistake with no legitimate reading and
used to resolve to `theta` with no signal.  The RESOLUTION is unchanged --
`theta` still wins, bit for bit, as
`test_v5_12_0_naming_aliases::test_set_source_theta_wins_consistent_across_suites`
requires of BOTH suites -- and a bare `theta=` call stays silent (its `angle`
sits at the `0.0` / `None` default).  Traced JAX angles skip the comparison;
concrete ones are compared.

### Fixed -- RCWA: the 1-D passive energy guard no longer raises on array-valued indices (H6, VERIFY-A14)

`rcwa_efficiency_1d` built its new `passive=` argument as
`np.array([eps_ridge, eps_groove])`.  That pack is evaluated outside
`_passive_media`'s own except-clause, so `n_ridge = np.array([2.04+0j])` with a
scalar `n_groove` raised `ValueError: setting an array element with a sequence`
from the ENERGY GUARD -- after a complete and correct solve.  The two
permittivities now go in as separate `eps_arrays`; the predicate is identical
(the max over the pack and over the parts are the same boolean) and all four
scalar/array spellings return bit-identical efficiencies.
```

---

## 7. Tests run (exact commands, all with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`)

| command | result | duration |
|---|---|---|
| `tests/unit/test_audit2609_a14_rcwa_eme_bor.py` (the WP's gate, unmodified) | **32 passed** | 29.6 s |
| `tests/unit/test_audit2609_a14_verify.py` (NEW, mine) | **25 passed** | 89.4 s |
| `test_fix_branch_cut_round2.py test_verify_branch_cut_round2.py test_fix_rcwa_even_sector_wsl.py test_verify_rcwa_even_sector.py test_niche_audit_w7_rcwa.py test_v5_21_delta_audit.py test_v5_14_1_rcwa_audit_fixes.py test_niche_audit_m4_m5_m6_rcwa.py test_audit_s1_2_rcwa_lossless_tripwire.py test_pmm2d_staggered_wood_list.py test_audit_w3_pmm_jax_guards.py` | **218 passed, 1 failed** (the failure is `_lens_traced_multibranch.py`, another WP's uncommitted file) | 187.7 s |
| `test_radial_eigensolver.py test_niche_audit_w6_bor.py test_coupled_eigensolver.py test_eme_2d.py` | **110 passed** | 911 s |
| `test_v5_12_0_naming_aliases.py test_rcwa.py test_v5_20_12_rcwa_jones_2d_fff_nv.py test_v5_20_6_rcwa_jones_2d_li.py test_v5_11_0_rcwa_fff_nv_2d.py test_v5_6_1_rcwa_symmetry.py` (after my `_resolve_incidence` change) | **149 passed** | 633.8 s |
| `python validation/run_all.py test_rcwa --quiet` | **PASS** | 2.1 s |
| `test_coupled_eigensolver.py test_niche_audit_w6_bor.py test_radial_eigensolver.py test_audit2609_a14_rcwa_eme_bor.py test_bor_sem.py` (AFTER my `guided_modes` change) | **159 passed** | 762.6 s |
| `test_v5_10_3_rcwa_2d_autodiff.py test_v5_20_1_rcwa_2d_oop_jax.py test_v5_20_3_rcwa_1d_oop_jax.py test_eme_2d_vector.py test_eme_diffraction.py test_eme_census_determinacy.py test_fix_eme_branch_cut.py test_audit_w6_eme.py test_niche_audit_w6_eme.py test_eme_jax_modes.py` | **152 passed** | 1245.2 s |
| `test_rcwa.py test_niche_audit_m4_m5_m6_rcwa.py test_audit_s1_2_rcwa_lossless_tripwire.py test_v5_12_0_naming_aliases.py test_audit2609_a14_rcwa_eme_bor.py test_audit2609_a14_verify.py test_v5_6_rcwa_convergence.py test_v5_7_0_rcwa_asr.py test_v5_11_0_rcwa_segments.py` (FINAL, after every edit of mine) | **256 passed, 2 skipped** | 862.1 s |
| `repro/RCWA-EME-BOR/p02_oracle.py` (24 configs), `p01_energy.py` (12), `p04_conv.py`, `p06_jones_tmm.py` | agreement <= 1.50e-13 off-anomaly; closure <= 1.03e-13; **Li metallic-TM table bit-for-bit identical to the audit's published values**; TMM parity 5.6e-17..3.3e-16 | ~200 s |

Two further "keep intact" items spot-checked outside the suite:

* **Li's metallic-TM convergence signature** (`p04_conv.py` section C, Ag
  0.135+3.99i, `Lambda` 0.5 um, d 0.2 um, duty 0.5, `lambda` 0.6328 um, normal,
  n_sub 1.5) reproduces the audit's published table **to every digit**:
  R0(li) 0.506619 / 0.520008 / 0.521328 / 0.521841 / 0.522258 / 0.522461 at
  N = 11/43/123/203/303/403, R0(laurent) 0.778438 / 0.543236 / 0.526067 /
  0.517855 / 0.511455 / 0.519943, A(li) 6.88e-02 -> 4.23e-02 (monotone, settled
  by N ~ 83), A(laurent) 1.70e-01 -> 5.56e-02 (oscillating).  The same script's
  section B, which the audit ran to show the nudge was SILENT, now prints the
  `WoodNudgeWarning` -- the H2 fix, visible in the audit's own reproducer.
* **Conical Jones basis-rotation covariance** (CONVENTIONS 7.1), C4 cell at
  `theta = 20 deg`: `max|J(phi=90) - R(90) J(phi=0) R(90)^T|` = 2.99e-13
  (laurent) / 1.82e-13 (li) / **1.23e-13 (fff_nv)** -- the same 1e-13 level the
  audit measured, and `fff_nv` is now on it too.

---

### 7a. Failures seen, and their attribution

| failure | attribution |
|---|---|
| `tests/unit/test_v5_21_delta_audit.py::test_d3_air_focus_multibranch_runs_without_warning` | **NOT WP-A14.**  `RuntimeWarning: apply_real_lens_traced_multibranch: reconstructed grid power is 2.32x the input aperture power`, raised from `lumenairy/elements/_lens_traced_multibranch.py:1006` -- an UNCOMMITTED working-tree file owned by another work package (`git status` shows it modified).  Reproduced standalone in 6.9 s with nothing of WP-A14's in the call path. |

No other failure was observed in any file I ran.  The 2 skips in the final batch
are the documented `threadpoolctl`-conditional branches (`threadpoolctl` is not
installed on this machine, which is also why the H5 warning fires in the suite --
by design).

---

## 8. Verdict

Every finding WP-A14 claims fixed IS fixed, and every number I re-measured on the
WP's own fixtures reproduced (several -- the even-fold `max|dJ|` table, the
`eigh` Bessel errors, the Li metallic-TM convergence table -- bit for bit).  The
two P1s are the ones that needed the hardest look:

* **H2's machinery is sound** where it was most likely to be wrong: the
  `_WoodAnomaly` + `signature().bind` re-entry forwards every argument EXACTLY
  on all eight entry points; the shared nudge PMM depends on is bit-identical
  over 4000 randomised calls; the closure bookkeeping holds at 1e-14; the sweep
  is monotone in 1-D and in 2-D.  What needs restating is the ADVERTISING: the
  6.7e-06 residual, the "4-5 decades" grazing artefact and the "factor 6.05 from
  the average" are all single-fixture results (V10, V3, V14).
* **H1 was half-closed.**  The band fix is real and verified against the exact
  hybrid oracle on six new fixtures, but the contract the WP claimed -- "the
  silent `[]` cannot come back through a different door" -- was measurably false
  on a high-contrast fibre.  Fixed here (V1); the partial-result case remains
  open (V1-residual).

Three defects were found and fixed inside WP-A14's own files (V1, V11, and the
coordinator-requested V0), each with its own regression gate.  25 new tests,
`256 passed, 2 skipped` on the final consolidated batch, and no collateral
damage attributable to this work package.

---

## 9. Follow-up (orchestrator rulings on section 6, implemented 2026-09-12)

All seven assigned items are implemented and measured.  V8 (PMM `wl_eff`), H5
(`threadpoolctl`) and the multibranch `RuntimeWarning` were reassigned and are
not touched here.

### 9.1 V1-residual -- a SHORT `guided_modes` result is now loud too

**What was added.**  `_step_index_root_census(m, a, eps_core, eps_clad, k0)` --
a sign-change scan of `fiber_oracle.fiber_det`, the exact 4x4 hybrid
boundary-match determinant, over `n_eff` in `(n_clad, n_core)` at
`_CENSUS_SCAN = 2001` samples, for the requested azimuthal order.  No bisection:
one determinant per sample.  `guided_modes` now warns, naming BOTH counts and
the order, when it returns fewer modes than the census finds; `census=False`
skips the scan.

**Scan resolution and its failure mode, as required.**  The census resolves two
roots only if they are more than one cell -- `(n_core - n_clad) / 2000` in
`n_eff` -- apart.  A near-degenerate pair (the HE/EH partners of one LP group at
weak contrast) or a TANGENTIAL double root is counted once or not at all.  That
error is **one-sided in the safe direction**: the census can only UNDER-count,
so the notice can miss a genuine shortfall but can never invent one.  That is
what makes a decision assertion legitimate here instead of a two-sided bar.

**Verified.**  Counts identical to the BISECTING `fiber_modes` on **12 of 12**
fixtures spanning `m` = 0..5, `V` = 1.8..9 and three index systems:

| m | n_core / n_clad | V | census | `fiber_modes` |
|---|---|---|---|---|
| 1 | 3.48 / 1.44 | 2.0 | 1 | 1 |
| 1 | 3.48 / 1.44 | 4.0 | 3 | 3 |
| 0 | 3.48 / 1.44 | 4.0 | 2 | 2 |
| 2 | 3.48 / 1.44 | 4.0 | 1 | 1 |
| 3 | 3.48 / 1.44 | 6.0 | 1 | 1 |
| 5 | 3.48 / 1.44 | 9.0 | 2 | 2 |
| 1 | 1.45 / 1.44 | 2.4 | 1 | 1 |
| 0 | 1.45 / 1.44 | 2.6 | 2 | 2 |
| 2 | 1.45 / 1.44 | 2.6 | 1 | 1 |
| 0 | 1.45 / 1.44 | 1.8 | 0 | 0 |
| 1 | 2.00 / 1.44 | 2.0 | 1 | 1 |
| 1 | 1.45 / 1.445 | 2.4 | 1 | 1 |

End to end: **Si/SiO2 V = 4.0 -> solver 1, census 3, warnings 0 -> 1** (pinned);
**dn = 0.010 V = 2.4 -> solver 1, census 1, warnings 0** (pinned, the quiet
side); Si/SiO2 V = 2.0 -> solver 0, census 1, **two** notices (the empty-list
one from section 3 plus the census one).  The census REFUSES rather than
guessing where it cannot be a census: `None` for a lossy core, `None` for an
inverted profile.

**Cost, measured:** 49-72 ms for 2001 samples against **0.72 s / 17.9 s /
67.8 s** for the FD eigensolve at `N` = 150 / 400 / 600 -- 9.9 % of the call at
the smallest grid anyone uses, 0.1-0.4 % at the grids real work runs.

Gate: `test_audit2609_a14_verify.py::test_h1_a_short_guided_mode_list_is_never_silent`
(five arms: the 12-fixture oracle cross-check, the short list, the agreeing
list, the two refusals, and `census=False`).

### 9.2 V5 -- `jax.jit` no longer returns NaN at an exact Wood anomaly

**Root cause, located rather than assumed.**  Under `jax.jit` every constant
built inside the traced function -- including `jnp.asarray(1e-6)` -- is a
`DynamicJaxprTracer`, so `_is_traced(kx0)` and `_is_traced(wavelength)` are both
True, `geom_concrete` is False and the ENTIRE guard block (the propagating-
incidence check AND the Wood nudge) is skipped: instrumented, the nudge was
called **zero** times under `jit`.  The solve then ran AT the anomaly.

**Why a `lam` floor was not enough** (measured, not reasoned): flooring the
modal eigenvalue left the NaN exactly where it was.  For a uniform half-space
`det Q = eps^N * prod(kz^2)` -- the isotropic `Q` of `_layer_Q_matrix` has
diagonal blocks whose determinant is `-kx^2 ky^2 + (eps - kx^2)(eps - ky^2) =
eps * kz^2` per order -- so `V = Q diag(1/lam)` is EXACTLY rank-deficient when
any `kz = 0`, whatever `1/lam` is floored to.  The interface `b = solve(Vb, Va)`
is then singular; NumPy raises `LinAlgError`, JAX silently returns NaN.

**The fix.**  `_traced_grazing_floor(eps)` returns
`sqrt(2 * _WOOD_PAIR_STEP_REL * |eps|)` = 4.47e-05 for `eps = 1` -- exactly
where the `+rel` leg of the shipped bracket puts a grazing order's `|kz|`.
`_homogeneous_eigenmodes(..., grazing_floor=)` pushes any order with
`|kz^2| < floor^2` onto the EVANESCENT side at that `|kz|` and builds `Q` from
the SAME shifted `kz^2`; `_layer_eigenmodes(..., grazing_floor=)` carries the
companion floor for a uniform layer at cut-off.  Both default to `None` =
unchanged, and only `rcwa_efficiency_1d`'s traced branch passes a value.

**Measured.**

| | before | after |
|---|---|---|
| `jax.jit` at `Lambda = lambda`, te | `R = [0,0,0,0,0,nan,0,...]` | 0.040057738059 |
| ... tm | NaN | 0.158125996111 |
| closure abs(sum R + sum T - 1) | NaN | **0.0 (te) / 4.4e-16 (tm)** |
| `jax.grad` through it | NaN | 0.4396660341 (finite) |
| agreement with the EAGER bracket mean | -- | **9.4e-08 (te) / 5.3e-06 (tm)** at `n_ridge` 2.04; 3.2e-07 / 1.2e-04 at 1.5 |
| exactly grazing order `m = +/-1` | NaN | **exactly 0.0** (the eager mean leaves 4.0e-07 .. 2.3e-05 there) |
| OFF-anomaly, floor ON vs floor OFF | -- | **0.000e+00 on all 10 arms** (te/tm x 0.9 / 1.3 / 0.7 / 1.111 / 0.5123 um) |

The agreement with the bracket mean is `sqrt(2 * 1e-9) = 4.5e-05` times an O(1)
coefficient of 0.002 .. 3.8 -- i.e. inside the `sqrt(delta)` law's own accuracy,
which is the strongest statement this formulation allows.  The traced path is in
one respect BETTER than the concrete one: its grazing order stays evanescent and
therefore carries exactly zero power, so it does not have the symmetric
average's `~sqrt(delta)` artefact (section 2, H2(d)).

**Scope.**  The 2-D entry points are unaffected: with a traced wavelength they
refuse LOUDLY (`TracerArrayConversionError`) before reaching the solver, and
with a concrete wavelength and a traced cell they take the host-side nudge as
before (verified finite, closure 2.2e-16, at and off the anomaly).  `berreman.py`
and `_berreman_jax.py` call both helpers positionally and are untouched by the
new keyword-only parameter.

Gate: `tests/unit/test_rcwa.py::test_jax_wood_anomaly_no_nan`, extended with a
`jit` + traced-index arm for BOTH polarizations (value, closure, grazing-order
zero, finite `grad`) and a jit-vs-jit off-anomaly bit-identity arm.

### 9.3 V14 -- the mechanism restated in all three places

`_wood_symmetric`'s docstring, `WP-A14_REPORT.md` section 2 (H2, item 2) and
`WP-A14_CHANGELOG.md` now carry the matched-bracket table and say: the 100x
narrower bracket buys the guaranteed ~10x through the verified `sqrt(delta)`
law; the average buys CONTINUITY of the sweep, not accuracy (it helps on 3 of 8
mount/polarization arms at a matched bracket and hurts on 5, by up to 10x).  The
three single-fixture numbers are replaced by the measured ranges: residual up to
**2.9e-04 relative**, gain **1.05x .. 60.5x**, grazing artefact worst **3.25
decades** below the specular (lossy Ag TM).

### 9.4 V3 -- the grazing-order bar is now derived

`test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular` is renamed
`::test_h2_exactly_grazing_orders_stay_under_the_sqrt_bracket_bound` and its bar
moved from `< 1e-4 x specular` (which the lossy Ag mount violates at 3.61e-04)
to a bound derived from the bracket: the artefact is
`sqrt(2 * _WOOD_PAIR_STEP_REL) = 4.5e-05` times an O(1) coefficient, measured
worst **3.81e-05 absolute / 5.67e-04 relative** over four mounts x two
polarizations, so the bars are **1e-3 absolute / 1e-2 relative** -- 1.4 and 1.2
decades above the measured worst and 2 decades below the O(1e-1) a lost
evanescent mask would put there.  Verified to hold on the lossy Ag mount.

### 9.5 V2 -- the vacuous isotropic-once assertion is replaced

`test_h4_isotropic_cell_builds_the_li_operators_once`'s tail compared
`orig(...)` with `orig(...)` -- the same call twice.  It now rebuilds the OLD
two-call form explicitly, under the same counter, and asserts
`(n_one, n_two) == (1, 2)` with `Cxx` and `Cyy` **bit-identical** between the
one-call and two-call paths (`array_equal` True on both blocks).

### 9.6 V7 -- the trivially-true `np.isreal` is replaced

`test_h4_bor_pencil_eigh_accuracy` now asserts, alongside the 1e-12 accuracy
bar, that the spectrum is a real dtype, strictly POSITIVE (SPD pencil) and
strictly ASCENDING with a forward gap `> 1.0` -- the discriminating claim, since
a non-symmetric `eig` returns an arbitrary order and needed the explicit
`argsort` the old body carried.  MEASURED smallest forward gap over the five
fixtures **24.69** (`gamma^2` 5.7832 -> 30.4713 at m = 0 Dirichlet), largest
54.57 -- ~15 decades above the 1e-13 these eigenvalues carry, so the `> 1.0`
bar has 1.4 decades below it and 13 above.

### 9.7 V6 -- the scope notice reaches the off-plane `fff_nv` path

`_li_tensor_scope_notice` is now called from the OUT-OF-PLANE branch of
`rcwa_jones_2d` with the same wording.  Measured: an out-of-plane disk warns
once and names the diagonal fraction (0 before); an out-of-plane square does
not; `laurent` / `li` never do; `allow_nonseparable_nv=True` silences it.
Gate: `::test_h3_offplane_fff_nv_gets_the_scope_notice_too`.

### 9.8 V13 -- the `_li_blocks` docstring corrected to what I measured

The claim "reduces to `_li_convolutions_2d` EXACTLY when both companions are the
cell (measured: `Cxx` bit-identical, `Cyy` to 4.2e-16)" was read off a
SEPARABLE cell.  Re-measured at `n_orders` 4x3 and 6x6:

| cell | dCxx | dCyy | relative |
|---|---|---|---|
| uniform | 0.0 | **0.0** | 0 |
| y-uniform stripe | 0.0 | 1.78e-15 | **4.2e-16** |
| x-uniform stripe | 0.0 | 1.78e-15 | 4.3e-16 |
| rectangle 32x24 | 0.0 | 4.55e-02 | **1.55e-02** |
| square (C4) | 0.0 | 5.17e-02 | 1.62e-02 |
| disk | 0.0 | 6.27e-02 | **2.10e-02** |

`Cxx` IS bit-identical on every cell (the `L2 L1` order factorizes x first, so
the `xx` block is the pure x-inverse rule); `Cyy` agrees only for a separable
cell, because `L2` applies the y-inverse rule to an operator x has already been
factorized out of.  Both converge to the same limit (Li 2003 Sec. 5.2).  The
docstring now says exactly that, and records that `symmetrize=False` keeps this
call on the historical operator BIT for BIT (verified textually: the pre-fix
`_li_convolutions_2d_tensor` body and the new `_li_tensor_l2l1` body are
byte-identical).

### 9.9 V10 / V12 -- the two changelog scoping gaps

* **V10**: the H3 migration note now states the change on a GENERAL
  (non-transpose-symmetric) axis-aligned cell -- `max|dJ|` between the
  symmetrized and single-order operators **8.3e-04 at `n_orders` 4 falling to
  1.2e-04 at 12**, inside the truncation error and converging to the same limit
  -- not only that a separable stripe is unchanged.
* **V12**: the H2 measured list now carries the TRANSMITTED residual, TM
  `max|dT| = 1.45e-05`, which is the largest single deviation anywhere in the
  24-configuration oracle sweep and was previously unquoted.

### 9.10 Tests re-run after the follow-up (all `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`)

| command | result | duration |
|---|---|---|
| `test_audit2609_a14_rcwa_eme_bor.py test_audit2609_a14_verify.py test_rcwa.py test_v5_12_0_naming_aliases.py test_niche_audit_m4_m5_m6_rcwa.py test_audit_s1_2_rcwa_lossless_tripwire.py test_v5_20_12_rcwa_jones_2d_fff_nv.py test_v5_20_6_rcwa_jones_2d_li.py test_v5_11_0_rcwa_fff_nv_2d.py test_v5_6_1_rcwa_symmetry.py test_v5_11_0_rcwa_internal_field.py test_niche_audit_w7_rcwa.py` (the consolidated batch) | **321 passed** | 1068 s |
| `test_coupled_eigensolver.py test_niche_audit_w6_bor.py test_radial_eigensolver.py test_bor_sem.py test_eme_2d.py` (BOR/EME, exercises the new census) | **133 passed** | 985 s |
| `test_v5_10_3_rcwa_2d_autodiff.py test_v5_20_1_rcwa_2d_oop_jax.py test_v5_20_3_rcwa_1d_oop_jax.py test_v5_14_4_berreman.py test_v5_20_1_berreman_offplane_oblique.py test_v5_14_5_emt_and_berreman_jax.py test_fix_branch_cut_round2.py test_verify_branch_cut_round2.py test_v5_7_0_rcwa_asr.py test_v5_6_rcwa_convergence.py` (JAX + the `berreman` consumers of the two changed helpers + branch cut) | **122 passed, 2 skipped** | 965 s |
| `python validation/run_all.py test_rcwa --quiet` | **PASS** | 3.3 s |
| `test_rcwa.py::test_jax_wood_anomaly_no_nan` + `test_audit2609_a14_verify.py` (final re-run after the last docstring edits) | **28 passed** | 271 s |

**576 passed, 2 skipped, 0 failed** across the four batches; `validation` PASS.
No new failure, and the one failure recorded in section 7a
(`test_v5_21_delta_audit.py::test_d3_air_focus_multibranch_runs_without_warning`,
another WP's uncommitted `_lens_traced_multibranch.py`) is reassigned to WP-A3.

`berreman.py` and `_berreman_jax.py` are the only out-of-package callers of
`_homogeneous_eigenmodes` / `_layer_eigenmodes`; they call them positionally and
their three test files are in the third batch above, green.

Lint: `ruff check` on the eight edited files reports only the `I001`
import-sorting and `F401` notices that HEAD already carries on the same files
(8 at HEAD, 10 now, the two extra being in-function imports in the new test
arms, matching the existing style of the WP's own gate file).  The repo's
`[tool.ruff]` line-length is 100 and `E501` is ignored; my longest added line is
86.

### 9.11 Files touched in the follow-up

Source (all inside WP-A14 ownership):

* `lumenairy/elements/rcwa/_core.py` -- `_traced_grazing_floor`;
  `grazing_floor=` on `_homogeneous_eigenmodes` and `_layer_eigenmodes`;
  `__all__`; the restated `_wood_symmetric` mechanism paragraph.
* `lumenairy/elements/rcwa/oned.py` -- the traced branch computes and forwards
  the floor to the region and layer mode builders.
* `lumenairy/elements/rcwa/twod.py` -- the off-plane `fff_nv` scope notice.
* `lumenairy/elements/rcwa/stack.py` -- the corrected `_li_blocks` docstring.
* `lumenairy/elements/bor/coupled_radial_eigensolver.py` -- `_CENSUS_SCAN`,
  `_step_index_root_census`, `census=` on `guided_modes` and the shortfall
  notice.

Tests:

* `tests/unit/test_rcwa.py` -- `test_jax_wood_anomaly_no_nan` extended.
* `tests/unit/test_audit2609_a14_rcwa_eme_bor.py` (the WP's gate) -- V2, V3, V7.
* `tests/unit/test_audit2609_a14_verify.py` (mine) -- two new gates.

Docs:

* `.../fixes/WP-A14_REPORT.md`, `.../fixes/WP-A14_CHANGELOG.md`,
  `.../fixes/VERIFY_WP-A14.md` (this section).
