# Scoping the three 5.45.0 error-class guards onto the BOR multilayer solver — 2026-09-12

> **STATUS — SCOPING ONLY. No file under `lumenairy/` was modified.** Every number below
> was measured on the worktree `C:\tmp\lum_borscope` (branch `scope/bor-guards`, commit
> `352173e`, lumenairy 5.44.0 dev), on both local builds (Windows / py3.14 and WSL / py3.12), across a
> 1 / 2 / 4 / 8 thread ladder, and under three distinct OpenBLAS kernels (Haswell,
> Katmai, Nehalem -- section 5.1 records why ZEN and SkylakeX were not reachable here).
> Probes and raw JSON: `validation/probe_scope_bor_guards/`.
>
> **Verdicts.**
>
> * **Class A -- NO-GO. A P1 defect is present and reproduces.** The flux-oriented rule is
>   immune on ordinary geometry (0 backward modes in 456 modal bases; `cond(a+b) <= 5.85`
>   at the coincidence that drove the Cartesian engine to 1.97e15), but the CLASSIFIER that
>   decides which rule applies is scaled by the mode's own collapsing magnitude. Near a
>   radial cutoff it hands the orientation to the eigensolver's backward error: the shipped
>   band is CROSSED by its own noise side on **12 of 126** measured populations (worst 8.30
>   decades above its own bar), a lossless stack's closure degrades to **2.16e-04**, a
>   propagating channel is dropped, and the channel count moves on **21 of 24** rungs with
>   the BLAS kernel and on **35 of 39** with the thread count alone. The candidate band
>   (the shape the library's five other sites already use) is measured 170x better across
>   kernels, moves **0 of 24** class verdicts, and moves **0 of 12** ordinary solves.
> * **Class B -- NO-GO for the Cartesian conjunction as written; GO for the census hook and
>   for a passivity refusal on the legacy path.** The production cascades have ONE
>   population (2,031 inverses, rcond 1.99e-07 .. 1.0, residual <= 3.41e-13) and should
>   stay unguarded by measurement. A genuine broken population exists on the legacy nodal
>   basis (`R + T` up to **966.7**, returned unwarned below 4 vacuum wavelengths) but its
>   inverse satisfies `A X = I` to 8.2e-13, so the rcond-AND-residual conjunction refuses
>   **0 of 6 rows on every kernel**. What separates there is passivity, not conditioning.
> * **Class C -- NO-GO. The union / enrichment sliver is present, unguarded and unwarned.**
>   A wall coincidence of 1e-7 of the domain radius moves the per-order answer by **586x to
>   5,428x** the physical wall shift (1-D bar: 100x) and injects spurious axial wavenumbers
>   **5.01e+06** times the physical index ceiling, with **zero** warnings and a closure that
>   stays at its healthy baseline over four decades of the ladder. A control that keeps the
>   same geometry but denies the window its union is clean on every quantity. The bar is
>   derived in section 7.2 and keys on the spurious wavenumber and the geometry -- NOT on
>   the energy violation, which this scoping measures to move **70.90x with the BLAS kernel
>   alone**, reproducing the 5.45.0 CI finding on the cylindrical engine.

---

## 1. Terms, and what is being ported

Three error classes were found and fixed on the Cartesian engines during the 5.45.0 wave.
Each is named here by its mechanism, because the BOR peer of each mechanism is a different
piece of code and only one of the three ports across as written.

**Class A — the branch cut.** A modal solver returns a squared axial wavenumber and takes
its square root. Which of the two roots is the *outgoing* one is decided by a sign test.
When the quantity that test reads is, for the mode in question, the eigensolver's backward
error rather than physics, the root is chosen by rounding — and rounding is a function of
the BLAS kernel and the thread count. On the Cartesian side the failing test was the exact
`Re(r) == 0` pin in `rcwa/_core._sqrt_decay`; the fix replaced it with a band taken
*relative to the spectrum's largest root* (`_CUT_BAND_REL = 1e-8`), in one shared
implementation that five private copies were deleted in favour of
(`FIX_BRANCH_CUT_ROUND2_2026_09_11.md`).

**Class B — interface conditioning.** The S-matrix cascade forms explicit inverses. When
one of them is numerically singular the returned number is build-dependent. The guard
(`rcwa/_core._guarded_inverse`) is a **conjunction**: it refuses only when the
equilibrated reciprocal 1-condition is below a per-site bar **and** the inverse misses its
own defining equation `A X = I` by more than `_INV_RESID_REFUSE = 1e-8` in the same
scaling. The conjunction exists because rank-deficient-but-consistent operands are
ordinary and must not be refused (`FIX_PMM2D_MORTAR_ROUND3_2026_09_11.md`), and the guard
is armed at exactly one site because only there did the population separate two-sided.

**Class C — mortar and sliver.** Two adjacent layers whose material walls differ by a small
`delta` cause the discretisation to manufacture an element of width `delta`. The
spectral-element Jacobian makes the nodal stiffness scale as `1 / w^2`, the layer's modal
spectrum acquires spurious wavenumbers far above any physical index, and the interface
mode match conditions accordingly. Two guards were built:
`pmm/stack.py`'s union-grid refusal (a provably-passive super-unity screen, an own-scale
attribution, and an arbiter re-solve on the prescribed `min_feature` grid) and
`pmm/twod_staggered.py`'s per-layer L2 mortar contract (a minimum segment width of
`1e-3` of the period, refused, with a `1e-3 .. 3e-2` degradation-band warning).

**The BOR solver.** `BORStack(Rbig, m, ..., basis='fd'|'sem')` solves an axisymmetric stack
at one azimuthal order `m`. Fields go as `exp(i m phi + i q z)`; `q` is the axial
wavenumber and `qn = q / k0` the dimensionless axial index. Two radial bases exist:
`'fd'`, a Yee div-conforming staggered finite-difference basis on ONE uniform radial grid
shared by every layer; and `'sem'`, per-layer spectral-element meshes aligned with the ring
walls, with a `+-1` neighbour-enrichment window and cross-tested Galerkin mortar
interfaces. A third, legacy nodal FD basis remains reachable through
`bor_solve.build_layer(basis='nodal')`.

**Kernel-independence.** Following the 5.45.0 CI finding that the 1-D sliver guard's
super-unity trigger and attribution came out differently on AMD EPYC (OpenBLAS ZEN), every
decision quantity below is reported per kernel, and every candidate bar proposed in section 7 is
keyed on geometry or on a spurious-wavenumber ratio, never on the magnitude of an energy
violation. Section 5 gives the per-kernel tables and states which kernels were actually reachable.

---

## 2. Class A on BOR — the forward-orientation rule

### 2.1 Every orientation site, and what rule it carries

The BOR engines never select a root by the sign of `Im(q^2)`. They classify each mode with
a band and then orient it by a physical quantity:

```
prop = |Im q| < 1e-9 * max(|Re q|, 1e-300)
q    = q if (flux >= 0 if prop else Im q > 0) else -q
```

There are **five copies** of that rule and they are numerically identical today:

| # | site | basis | shape |
|---|---|---|---|
| 1 | `lumenairy/elements/bor/zcascade.py:86` | staggered FD (production `basis='fd'`) | per-mode loop |
| 2 | `lumenairy/elements/bor/zcascade.py:227` (`_orient_forward`) | legacy nodal FD | per-mode loop |
| 3 | `lumenairy/elements/bor/sem_radial.py:428` | SEM (`basis='sem'`) | vectorized |
| 4 | `lumenairy/elements/bor/_jax_bor.py:98` | FD JAX twin | vectorized `jnp` |
| 5 | `lumenairy/elements/bor/_jax_sem.py:247` | SEM JAX twin | vectorized `jnp` |

Two companion constants are also five- and six-fold duplicated: the flux-normalizer
fallback `|P| > 1e-10 * fnrm` (`zcascade.py:96`, `sem_radial.py:436`, `bor_solve.py:51`,
`_jax_bor.py:108`, `_jax_sem.py:255`) and the R/T channel gate
`|qn.imag| < 5e-5 and qn.real > 1e-6` (`bor_solve.py:180`, `bor_stack.py:692`,
`bor_stack.py:890`, `_jax_bor.py:195`, `_jax_sem.py:390`).

**The scale is the anomaly.** These five copies scale the band by **the mode's own
`|Re q|`**. Every other band of this shape in the library scales it by **the spectrum's
largest element, floored at 1.0**:

| site | scale |
|---|---|
| `rcwa/_core._sqrt_decay` (`_CUT_BAND_REL = 1e-8`) | `max(max|r|, 1.0)` |
| `elements/berreman.py:188`, `:350` | `max(1.0, max|g|)`, `max(1.0, max|M|)` |
| `elements/_berreman_jax.py:76` | `maximum(max|gam|, 1.0)` |
| `elements/eme/eme_2d_vector.py:255` | `max(1.0, max|ky|)` |
| `elements/pmm/_core.py:6603` | `max(max|flux|, 1.0)` |
| **`elements/bor/*` (5 copies)** | **`max(|Re q| of this mode, 1e-300)`** |

`_CUT_BAND_REL`'s own docstring records that this exact choice was measured against the
spectrum-scaled one and rejected: *"at a cutoff the mode's own magnitude has collapsed and
judging its real part against it is judging noise against noise ... The spectrum's top is
the only stable scale there."* The BOR copies make the rejected choice.

### 2.2 The ordinary population: the rule is immune, with numbers

A census of **106 fixtures / 456 layer modal bases** (lossless n = 1.41 / 1.50 / 2.00 and
lossy n = 1.50+0.05i / 2.00+0.20i; `m` = 0, 1, 2; both bases; uniform-layer-equals-region
and uniform-layer-equals-neighbour coincidences; a 1e-6 detune of both; thin annular rings;
near-cutoff `k0`; and index contrast across the stack) — `a1_orientation_census.py`:

| quantity | result |
|---|---|
| modes shipped BACKWARD (propagating with negative z-flux) | **0** |
| evanescent modes shipped with `Im q < 0` (a growing forward propagator) | **0** |
| propagating modes whose flux fell into the `1e-10 * fnrm` noise fallback | **0** |
| modes where the flux rule GOVERNED and a resolvable `Im q` disagreed | **0** (of 40,128 modes) |
| classifier NOISE side, worst `\|Im q\|/\|Re q\|` among modes called propagating | 5.7395e-13 — **3.24 decades** below the 1e-9 bar |
| classifier SIGNAL side, smallest such ratio among modes called evanescent | 3.3335e-02 — **7.52 decades** above |

**The coincidence that broke the Cartesian engines does not break this one.** Over every
coincident and 1e-6-detuned FD interface in the census, `cond(a + b)` runs
**1.0000 .. 5.845**. The Cartesian failure read `cond(a + b) = 1.97e15`. The reason is
structural: a uniform BOR layer's modes and the half-space region's modes are produced by
the same code path from the same permittivity, so the mode match is the identity, and the
flux rule cannot hand a lossless propagating mode the incoming root the way a sign test on
`Im(lam^2)` could.

**The classifier's crossing with real loss** is exactly linear in the imaginary index:
`|Im q| / |Re q| = 0.709 * Im(n)` on a uniform n = 1.41 cylinder at every `m` measured, so
the bar is crossed at `Im(n) = 1.41e-9`. Below that a genuinely lossy mode is *called*
propagating — but at that loss the flux rule and the decay rule agree, and the measured
prop-governed disagreement count is 0, so the misclassification is harmless on ordinary
geometry. The JAX twins, which carry their own copies of the rule, agree with NumPy to
`dR, dT <= 1.5e-13` on four (basis, m) combinations.

### 2.3 The binding population: a near-cutoff order, and the defect

The Cartesian band's binding population was a **layer cutoff**, for the structural reason
`_CUT_BAND_REL` records: for `lam^2 = -s + i eta` the principal root's real part is
`eta / (2 sqrt(s))`, so at fixed backward error the discriminating ratio grows without
limit as `s -> 0`. The cylindrical peer is a radial order approaching its own cutoff,
`q^2 = k0^2 eps - gamma_j^2 -> 0`, whence

```
rho = |Im q| / |Re q|  ~  |Im q^2| / (2 (Re q)^2)  ~  1 / qn^2 .
```

Because the PEC-walled cylindrical spectrum is discrete, an ordinary `k0` sweep cannot
reach the degenerate point; `a2_deep_cutoff_jax.py` and `a4_cutoff_crossing.py` therefore
solve for the cutoff of one named radial order and approach it geometrically, setting
`k0 = gamma_j / (n sqrt(1 - delta))` so that `qn = n sqrt(delta)` exactly.

**The crossing (`a4`, FD basis, Rbig = 24, N = 120, n = 1.41, 39 rungs over m = 0, 1, 2):**

| quantity | result |
|---|---|
| first crossing of the 1e-9 bar | `qn = 4.4588e-03`, `rho = 1.9753e-09` |
| rungs where a PHYSICALLY propagating order (`\|Re q\| > 10 \|Im q\|`) was called EVANESCENT | **32 / 39** |
| of those, still counted as an R/T channel by `solve()`'s own gate | **22** |
| rungs where the shipped FORWARD mode carries BACKWARD z-flux | **10 / 39** |
| lossless-stack closure at those rungs (baseline elsewhere ~1e-13) | up to **1.2167e-04** (m = 0) |
| flux magnitude at those rungs (`\|P\|/fnrm`; the noise fallback fires at 1e-10) | 1.4e-05 .. 2.5e-03 — the flux is SIGNAL |

The last row is what makes this a defect rather than a degeneracy: the mode's own z-flux is
five to eight decades above the noise floor and says *forward*, while `Im q` — which is
the eigensolver's backward error at these `qn` — says the opposite, and `Im q` is what the
shipped rule consults once the band has been crossed.

**The causal chain, end to end.** (1) `rho` crosses 1e-9 near cutoff. (2) The order is
called evanescent, so it is oriented by `sign(Im q)`. (3) `Im q` is backward error, so the
sign is arbitrary; when it selects the root with `Re q < 0` the order is shipped backward.
(4) `solve()`'s channel gate `qn.real > 1e-6` then **drops that order from the R/T channel
set**. (5) The closure of the surviving channels degrades to 1.2e-04, and the channel
count itself becomes a function of the arithmetic.

### 2.4 The answer moves with the BLAS thread count

`a4` re-run at `OPENBLAS_NUM_THREADS` = 1 / 2 / 4 / 8 on one build:

| quantity | result |
|---|---|
| rungs whose CLASS or ORIENTATION moves across the thread ladder | **35 / 39** |
| spread of `rho` itself across the ladder (worst rung, m = 1, delta = 1e-5) | 5.10e-12 .. 2.01e-09 — **394x**, straddling the 1e-9 bar |
| worst spread of `sum R` on the observable | **1.0000** (0.0050213801 at t1/t4 against 1.0050214667 at t2/t8) |
| channel count at that rung | **2 orders at t1/t4, 3 at t2/t8** |
| worst per-order closure over the ladder | 3.8050e-04 |

A whole diffraction channel appears and disappears with the thread count. Per
`docs/TESTING_STANDARDS.md` this is a defect, not noise.

### 2.5 The candidate bar, and its arbiter

Super-unity is the detector, not the attribution, so `a5_arbiter_band.py` re-solves with
exactly one thing changed. The candidate is the shape every other engine in the library
already uses:

```
prop = |Im q| <= band * max(max|q| over the layer's spectrum, k0)      band = 1e-8
```

The probe re-implements `zcascade._layer_modes_staggered` with the classifier as a
parameter; with `rule='shipped'` it reproduces the library **bit for bit** (`dR = dT =
0.0`), so the comparison is of one line and nothing else.

| measurement | shipped | candidate |
|---|---|---|
| worst lossless closure over the 39-rung cutoff ladder | **1.2167e-04** | **1.9655e-07** (**619x better**) |
| rungs where the two disagree on the CHANNEL COUNT | — | **10 / 39** (shipped drops an order the candidate keeps) |
| worst ratio shipped/candidate closure on a single rung | — | **57,904x** (m = 0, delta = 3.16e-06) |
| ORDINARY battery (m = 0,1,2,5 x k0 = 0.8, 2.0, 3.5), solves moved | — | **0 of 12** |

The candidate is a pure widening: it changes nothing on ordinary geometry and removes the
defect where it exists.

**Two-sided margins of the candidate bar** (`a6_candidate_band_margins.py`).  The candidate
changes the discriminating ratio from `rho = |Im q| / |Re q|` to
`sigma = |Im q| / max(max|q|, k0)`, so BOTH populations are re-measured against `sigma`:

| side | population | n | worst `sigma` | room at `band = 1e-8` |
|---|---|---|---|---|
| NOISE (the band MUST reach it) | lossless propagating modes, ORDINARY geometry (2 bases x n = 1.41/1.50/2.00 x m = 0,1,2 x k0 = 0.8/2.0/3.5) | 54 | 1.3024e-15 (WIN) / 6.4173e-16 (WSL) | **6.89 decades**, worst of both builds |
| NOISE at a DEEP CUTOFF (the binding population) | same, over `delta` from 1e-4 to 1e-26, i.e. `qn` from 1.4e-02 to 1.4e-13 | 72 | 8.7301e-10 (WIN) / 2.3089e-10 (WSL) | **1.06 decades**, worst of both builds |
| SIGNAL (the band must NOT reach it) | genuinely lossy modes at `Im(n) = 1e-3` | 6 | 2.5742e-05 | **3.41 decades** |
| SIGNAL, the thin end | same at `Im(n) = 1e-6` | 6 | 2.5742e-08 | **0.41 decades** |

`sigma` is exactly linear in the imaginary index (`sigma = 2.574e-02 * Im(n)`), so the
candidate crosses at `Im(n) = 3.9e-07` where the shipped band crosses at `Im(n) = 1.4e-09`.
The candidate therefore calls a slightly wider band of weakly lossy media "propagating" and
orients them by flux instead of by decay. **That is harmless where it was measured:** over
645 physically propagating modes at `Im(n)` from 1e-06 down to 1e-10 (m = 0, 1, 2), the flux
verdict and the decay verdict agree on **every one** (0 disagreements, with the flux at
`|P|/fnrm >= 0.12`, eleven decades above the noise fallback).

**The decisive comparison of the two bars, over the same 126 measured populations:**

| bar | rungs whose NOISE side CROSSES its own bar |
|---|---|
| SHIPPED `1e-9 * max(\|Re q\|, 1e-300)` | **12 / 126** on WIN, **11 / 126** on WSL (worst `rho` = 1.9844e-01 / 9.3114e-03, i.e. **8.30 / 6.97 decades ABOVE** its own bar) |
| CANDIDATE `1e-8 * max(max\|q\|, k0)` | **0 / 126 on BOTH builds** |

That the shipped band's worst crossing itself differs by 1.3 decades between the two builds
is the point: the quantity it thresholds is backward error, so how far past the bar it lands
is not reproducible either.

### 2.6 Verdict — Class A

**NO-GO. A P1 defect is present.** The flux-oriented rule is immune on ordinary geometry
(0 backward modes in 456 bases, `cond(a+b) <= 5.85` at the coincidence that broke RCWA),
but the *classifier that decides which rule applies* is scaled by the mode's own collapsing
magnitude, and near a radial cutoff it hands the orientation to the eigensolver's backward
error. Measured consequence: a lossless stack's closure degrades to 1.2e-04, a propagating
channel is dropped, and the channel count and `sum R` move with the BLAS thread count. The
remedy is the library's own existing shape, measured 619x better with zero movement on
ordinary geometry.

---

## 3. Class B on BOR — the explicit inverses

### 3.1 The sites

Every explicit inverse and linear solve in the BOR cascade is unguarded. The census
instrument (`_common.inv_census`, a measurement-only monkeypatch that scores each operand
with the library's own `_rcond_1_equilibrated` and `_equilibrated_inverse_residual`)
identified fourteen distinct call sites:

| site | what it inverts |
|---|---|
| `zcascade.py:237`, `:238` | `solve(Wb, Wa)`, `solve(Vb, Va)` — the mode match |
| `zcascade.py:241` | `inv(a + b)` — the interface transmission block |
| `zcascade.py:260`, `:261` | `inv(I - B11 A22)`, `inv(I - A22 B11)` — the Redheffer denominators |
| `zcascade.py` (nodal path) | `inv(Lm + k0^2 eps)` — the `E_z` elimination |
| `coupled_radial_eigensolver.py:242` | `Lei`, the staggered `E_z` elimination |
| `sem_radial.py:348` | `inv(Mz)` — the SEM `E_z` elimination (already carries an LU-pivot fallback to an unreduced QZ pencil) |
| `sem_radial.py:523`, `:525`, `:527`, `:529` | the four mortar mass-matrix projections |
| `sem_radial.py:532`, `:533` | `alpha = solve(Wb, Wa_b)`, `gamma = solve(Va, Vb_a)` |
| `sem_radial.py:537` | `inv(I + gamma alpha)` — the cross-tested mortar interface |
| `bor_stack.py` `layer_absorption` | `solve(I - S22_above S11_below, ...)` |

### 3.2 The healthy population — one population, not two

`b1_inverse_census.py`, **132 fixtures / 2,031 inverses**, built to contain every candidate
broken family (coincident uniform layers, 1e-6 detune, exactly degenerate twin layers,
thin layers down to 1e-12, near-cutoff `k0`, `m` up to 10, many rings, SEM degrees 6 / 8 /
12 / 16 with `elements_per_segment` 1 and 3, and a lossy metal ring):

| site | n | equilibrated rcond (min .. max) | residual (max) |
|---|---|---|---|
| `zcascade.py:237` `solve(Wb,Wa)` | 150 | 1.986e-07 .. 5.368e-03 | 2.222e-13 |
| `zcascade.py:238` `solve(Vb,Va)` | 150 | 2.261e-07 .. 2.490e-03 | 1.463e-13 |
| `sem_radial.py:532` `alpha` | 100 | 6.677e-07 .. 2.019e-03 | 6.520e-14 |
| `sem_radial.py:533` `gamma` | 100 | 1.368e-06 .. 4.914e-04 | 5.964e-14 |
| `zcascade.py:241` `inv(a+b)` | 150 | 1.957e-06 .. 1.000e+00 | 1.156e-14 |
| `sem_radial.py:537` `inv(I+ga)` | 100 | 1.962e-06 .. 9.906e-02 | 1.464e-13 |
| `sem_radial.py:348` `inv(Mz)` | 99 | 2.600e-06 .. 8.547e-04 | 3.408e-13 |
| `zcascade.py:260`, `:261` Redheffer | 360 each | 5.522e-06 .. 1.000e+00 | 5.531e-15 |
| `coupled_radial_eigensolver.py:242` `Lei` | 62 | 2.616e-05 .. 1.898e-03 | 8.787e-14 |
| `sem_radial.py:523/525/527/529` mortar masses | 100 each | 5.603e-02 .. 1.000e+00 | 2.541e-16 |

**Whole census: rcond 1.986e-07 .. 1.000, residual at most 3.408e-13.** There is no second
population here, and therefore no bar. `_INV_T22_RCOND_REFUSE = 1e-10` sits three decades
below the healthy floor; `_MORTAR_RCOND_REFUSE = 1e-12` five; `_MORTAR_RESID_REFUSE = 1e-6`
seven decades above the worst residual. A conjunction guard armed on this population would
be dormant on every fixture measured.

### 3.3 The broken population that does exist, and why the conjunction cannot see it

`AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md` recorded a genuine broken population on
the **legacy nodal FD basis**: its divergence-violating spurious-mode sea carries zero
z-flux, so those modes' orientation is decided by the sign of noise, adjacent layers orient
near-identical spurious modes oppositely, and the interface transmission block acquires a
null vector. That was remediated by changing `build_layer`'s **default** to the staggered
basis and adding a `UserWarning` past ~4 vacuum wavelengths. `basis='nodal'` remains a
documented escape hatch, **and on it the wrong answer is still returned rather than
refused**.

`b2_nodal_broken.py` (m = 1, N = 140, five-layer ring stack, Rbig from 2 to 16 vacuum
wavelengths):

| Rbig / lambda | staggered `max(R+T)` | nodal `max(R+T)` | nodal `rcond(a+b)` min | nodal residual max | warned? |
|---|---|---|---|---|---|
| 2 | 1.0000000000005 | **74.03** | 2.246e-05 | 6.914e-14 | **no** |
| 4 | 1.0000000000004 | **16.38** | 7.980e-06 | 7.609e-14 | **no** |
| 6 | 1.0000000000004 | 39.80 | 9.475e-06 | 1.490e-13 | yes |
| 8 | 1.0000000000001 | 128.77 | 3.552e-06 | 1.279e-13 | yes |
| 10 | 1.0000000000004 | 65.00 | 1.040e-05 | 1.360e-13 | yes |
| 12 | 1.0000000000001 | 248.85 | 8.854e-06 | 2.028e-13 | yes |
| 14 | 1.0000000000002 | **407.32** | 2.973e-06 | 5.998e-13 | yes |
| 16 | 1.0000000000004 | 177.14 | 3.974e-06 | 3.223e-13 | yes |

A refinement sweep in `N` (140 / 200 / 300 / 450) reaches `max(R+T) = 966.7` at 12
wavelengths and shows the blow-up is present at **1 vacuum wavelength** as well
(`max(R+T) = 5.06` at N = 140), i.e. below the warning's own threshold.

**Two conclusions, and they point in opposite directions.**

1. **The conjunction guard as written would refuse none of these rows.** The refusal
   requires the equilibrated residual to exceed `_INV_RESID_REFUSE = 1e-8`; the worst
   measured here is **5.998e-13** (1.7e-11 over the refinement sweep), four to five decades
   *below* the bar. The nodal `a + b` is not numerically singular in the sense the
   Cartesian guard tests for — the inverse satisfies its own equation to 1e-13; the
   operator merely amplifies. And the rcond populations overlap: the broken population's
   rcond runs 2.973e-06 .. 1.000 against the healthy 4.656e-04 .. 1.000, so the *worst*
   broken row is only **2.19 decades** below the healthy floor and most broken rows are
   inside it.
2. **The detector that does separate is passivity.** `max(R + T)` reads 1.0000000000005
   on every staggered row and 5.06 .. 966.7 on every nodal row — a clean separation with
   nothing in between, on a provably passive lossless stack. That is the 1-D sliver
   guard's screen, not the interface-conditioning guard's.

### 3.4 Verdict — Class B

**NO-GO for porting `_guarded_inverse` as an armed refusal; GO for the census hook and for
a passivity screen on the legacy path.**

* On the **production** bases (staggered FD and SEM) the population is single and healthy
  across 2,031 inverses and 132 fixtures; every site should stay **unguarded by
  measurement**, exactly as the plain 1-D PMM site does. Porting the census hook
  (`_INV_CENSUS`-style, `None` by default and one `is None` test per inverse) is free and
  is what would let a future population be measured rather than assumed.
* On the **legacy nodal** path a wrong answer up to `R + T = 966.7` is returned today, and
  at Rbig <= 4 vacuum wavelengths it is returned with **no warning at all**. This is a
  finding for 5.45.1, but the guard it needs is a provably-passive super-unity refusal on
  the lossless-incidence case, not a conditioning conjunction. Section 7.3 proposes it.
* The one site with a genuine remedy already in place is `sem_radial.py:348` (`inv(Mz)`),
  which detects a near-singular `E_z` elimination by an LU-pivot ratio and falls back to an
  unreduced QZ pencil. That fallback is a *different* shape from the Cartesian guard
  (repair, not refusal) and it is correct for its site; nothing needs porting there.

---

## 4. Class C on BOR SEM — the union / enrichment sliver

### 4.1 The mechanism, as built

`BORStack._solve_sem` gives every layer its own radial element mesh, but layer `i`'s
breakpoint set is the **window union** of the ring walls of layers `i-1`, `i` and `i+1`:

```python
u = set(walls[i])
if i > 0:            u |= set(walls[i - 1])
if i + 1 < len(walls): u |= set(walls[i + 1])
```

Two layers whose walls differ by `delta` therefore manufacture an element of width exactly
`delta` in **both** of their meshes. `build_mesh` merges breakpoints only when they are
closer than `1e-12 * Rbig`; there is no minimum-feature contract, no degradation warning,
and no super-unity screen anywhere on the path. `equalize_meshes` then pads every mesh in
the stack — including the half-spaces — to the largest element count, so the manufactured
cell's element budget propagates through the whole stack.

**A wall-free layer is not safe.** Because the window is `i-1, i, i+1`, a uniform spacer
placed *between* the two ring layers inherits **both** of their walls and carries the
sliver itself, in a layer that has no walls at all. This was found while building the
attribution control for `c4` (the one-spacer control broke harder than the adjacent arm)
and is reported here as a finding in its own right; the valid control needs **two**
spacers.

### 4.2 The delta ladder

`c4_sliver_attrib.py`, Rbig = 24, k0 = 2, `m` = 1, N = 200, walls at r = 6 and r = 6+delta
in two adjacent ring layers, per-order R compared **matched by axial wavenumber** (the
modal ordering permutes when the mesh changes, and an index-wise comparison — which an
earlier pass of `c1` used — compares different physical channels and is not reported here):

| delta / Rbig | delta (abs) | narrowest element | `\|q\|max` / `n_max k0` | closure | `R+T-1` | dR (q-matched) | interface rcond | warnings |
|---|---|---|---|---|---|---|---|---|
| 1e-01 | 2.400e+00 | 6.000e-01 | 2.794e+01 | 8.27e-10 | +3.85e-10 | 1.866e-01 | 1.86e-05 | 0 |
| 3e-02 | 7.200e-01 | 7.200e-01 | 2.722e+01 | 2.13e-10 | +1.83e-10 | 1.018e-01 | 4.21e-05 | 0 |
| 1e-02 | 2.400e-01 | 2.400e-01 | 5.926e+01 | 1.01e-10 | +3.37e-11 | 1.854e-02 | 4.21e-05 | 0 |
| 3e-03 | 7.200e-02 | 7.200e-02 | 1.761e+02 | 9.83e-11 | +4.02e-11 | 9.694e-03 | 1.89e-05 | 0 |
| 1e-03 | 2.400e-02 | 2.400e-02 | 5.103e+02 | 2.04e-10 | +4.29e-11 | 3.800e-03 | 6.50e-06 | 0 |
| 3e-04 | 7.200e-03 | 7.200e-03 | 1.680e+03 | 8.74e-11 | +6.71e-11 | 1.188e-03 | 1.99e-06 | 0 |
| 1e-04 | 2.400e-03 | 2.400e-03 | 5.023e+03 | 1.04e-09 | +1.04e-09 | 4.000e-04 | 6.67e-07 | 0 |
| 1e-05 | 2.400e-04 | 2.400e-04 | 5.039e+04 | 2.69e-07 | +2.69e-07 | 4.019e-05 | 6.69e-08 | 0 |
| 1e-06 | 2.400e-05 | 2.400e-05 | 5.260e+05 | 1.71e-06 | +1.71e-06 | 7.287e-06 | 6.69e-09 | 0 |
| 1e-07 | 2.400e-06 | 2.400e-06 | **5.014e+06** | 2.37e-04 | +2.37e-04 | **1.094e-04** | **6.69e-10** | **0** |

(degree 12; the degree 6 / 8 ladders and the nm-unit fixture — Rbig = 20 um, wl = 1550 nm,
delta down to 2 pm — are in `c1_union_sliver_win_t1.json` and `c4_sliver_attrib_win_t1.json`
and behave the same.)

Reading the table: `dR` tracks the physical wall shift with a continuity slope of
**7.777e-02 per unit shift** all the way down to `delta / Rbig = 1e-06`, then breaks. The
1-D guard's attribution bar is `_SLIVER_MOVE_FACTOR = 100` — "the answer moved more than
100x the geometric perturbation the snap describes". Measured here:

| degree | worst dR / (slope * delta) |
|---|---|
| 6 | 149x |
| 8 | **5,428x** |
| 12 | 586x |

against the 1-D bar of 100x.

**The spurious wavenumber is the clean signal.** `|q|max / (n_max k0)` grows from 27.9 at
`delta/Rbig = 1e-1` to **5.014e+06** at 1e-7 — five million times any physical axial index
the medium can support — and it is a pure ratio of the geometry: measured
`|q|max / (n_max k0) ~ 0.3 .. 0.5 * (Rbig / w)` on both the native and the nm-unit fixture,
whose `Rbig` differ by 833x and whose wavelengths differ by 338x.

### 4.3 The attribution control: it is the window, not the geometry

`delta` is a real geometric change, so the ladder above cannot on its own say whether the
damage is the sliver or the physics.  The control puts the two walls in layers **three
apart**, with TWO wall-free spacers between them, so the `+-1` window never spans both wall
sets while the geometry keeps exactly the same `delta`.  (ONE spacer is not enough and the
first attempt at this control was invalid: because `win[i]` is
`walls[i-1] | walls[i] | walls[i+1]`, a wall-free layer BETWEEN the two ring layers
inherits both of their walls and carries the sliver itself.  The one-spacer control read
`min_element = delta` exactly and broke harder than the adjacent arm.)

| quantity, at `delta/Rbig = 1e-07` | ADJACENT (window unions) | SEPARATED (control) | ratio |
|---|---|---|---|
| narrowest element | 2.400e-06 | **1.500e+00** (an ordinary element) | 625,000x |
| `\|q\|max` / `n_max k0` | **5.014e+06** | **27.22** | 184,200x |
| closure | 2.372e-04 | 3.550e-10 | 668,000x |
| q-matched per-order `dR` | 1.094e-04 | 7.988e-07 | 137x |
| worst move / physical wall shift over the ladder | **586x** (deg 12), **5,428x** (deg 8) | **3x** | ~195x |

The separated arm's `dR` falls **linearly with `delta` on every rung**, from 3.103e-01 at
`delta/Rbig = 1e-01` to 7.988e-07 at 1e-07, its narrowest element never leaves 1.071 (degree
8) / 1.500 (degree 12), and its `|q|max` never leaves 19.60 / 27.22.  The geometry is
harmless; the window's union of it is not.

### 4.4 The within-layer thin annular liner

`c2_within_layer.py` places an annulus of width `w` inside ONE layer's own segment list —
so the narrow element is built by that layer's own walls, not by the window — and sets the
liner's permittivity **equal to the material it replaces**, so the device is independent of
`w` by construction and every deviation is numerical damage with no physics to subtract.
This is the peer of the 2-D staggered PMM's `_STAG_MIN_SEG_FRAC` surface.

Degree ladder of the q-matched per-order error (Rbig = 24, k0 = 2, core wall at r = 6):

| w / Rbig | degree 6 | degree 8 | degree 12 | degree 16 |
|---|---|---|---|---|
| 1e-01 | 2.553e-04 | 1.163e-05 | 7.399e-06 | 6.610e-07 |
| 1e-02 | 1.533e-04 | 1.990e-05 | 9.579e-06 | 2.661e-06 |
| 1e-03 | 1.283e-05 | 2.451e-05 | 9.594e-06 | 2.747e-06 |
| 1e-04 | 1.463e-05 | 2.041e-05 | 8.167e-06 | 1.983e-06 |
| 1e-05 | 2.892e-05 | 1.788e-05 | 7.284e-06 | 1.062e-06 |
| **1e-06** | 5.090e-05 | 4.422e-05 | **1.335e-04** | **7.344e-03** |
| **1e-07** | 2.420e-04 | **7.640e-03** | **2.923e-02** | **2.914e-02** |

Two features matter. Down to `w / Rbig = 1e-05` the degree ladder still *converges* — the
error falls monotonically with degree, so the SEM handles a within-layer liner four decades
narrower than the domain. At `1e-06` and below the ladder **inverts**: degree 16 is 144x
*worse* than degree 6. That inversion is the sliver signature, because the spurious
wavenumber scales as `p(p+1) / w`; measured `|q|max / (n_max k0)` reaches **8.371e+06** at
degree 16. Super-unity reaches **+5.297e-02** (degree 12) and **+4.182e-02** (degree 16),
above the 1-D refusal bar of 1e-2 — but only on the last two rungs, and **0 warnings** are
emitted anywhere on the ladder.

### 4.5 Is the damage energy-invisible?

Partly, and the part that is invisible is the part a ported 1-D guard would miss.

| rung (degree 12, within-layer) | closure | q-matched dR | ratio dR / closure |
|---|---|---|---|
| `w/Rbig = 1e-04` | 8.828e-09 | 8.167e-06 | 925 |
| `w/Rbig = 1e-05` | 2.714e-07 | 7.284e-06 | 27 |
| `w/Rbig = 1e-06` | 1.481e-04 | 1.335e-04 | 0.9 |
| `w/Rbig = 1e-07` | 5.297e-02 | 2.923e-02 | 0.55 |

and on the union ladder (degree 12) `delta/Rbig = 1e-04` reads closure 1.04e-09 against
dR 4.00e-04 — the answer has moved four decades more than the closure has.

So the closure is a *late* detector: it stays at its healthy 1e-10 .. 1e-9 baseline while
the per-order answer has already moved by 1e-4, and it only rises once the element is six
decades below the domain radius. **A ported super-unity screen at the 1-D bars
(`_SLIVER_TRIGGER_BAR = 1e-3`, `_STACK_SUPERUNITY_BAR = 1e-2`) fires on 4 of 60 rungs of
the union ladder and on 2 of 40 rungs of the within-layer ladder** — it catches only the
catastrophic tail, and would leave the 1e-4 .. 1e-3 per-order errors silent. This matches
the 2-D mortar's finding that the damage there is energy-invisible, and it is why section 7.2's
proposed bar is keyed on geometry and on `|q|max`, not on the energy violation. It is also
independently the right choice under the CI's kernel finding.

### 4.6 The taper staircase

`c3_taper_staircase.py`, a cone from r = 8 to r = 2 over height 1.2, sliced into 4 / 8 / 16
/ 32 / 64 layers each carrying its own ring radius, so adjacent slices' walls differ by
`(r_top - r_bot) / n_slices`:

| slices | wall delta | narrowest element | / Rbig | / lambda_core | closure (deg 12) | `\|q\|max`/ceiling | warnings |
|---|---|---|---|---|---|---|---|
| 4 | 1.500e+00 | 1.375e+00 | 5.729e-02 | 1.072e+00 | 1.011e-08 | 29.69 | 0 |
| 8 | 7.500e-01 | 7.500e-01 | 3.125e-02 | 5.849e-01 | 6.557e-08 | 34.38 | 0 |
| 16 | 3.750e-01 | 3.750e-01 | 1.562e-02 | 2.924e-01 | 3.545e-08 | 59.25 | 0 |
| 32 | 1.875e-01 | 1.875e-01 | 7.812e-03 | 1.462e-01 | 2.273e-08 | 118.1 | 0 |
| 64 | 9.375e-02 | 9.375e-02 | 3.906e-03 | 7.311e-02 | 2.163e-08 | 235.8 | 0 |

A 64-slice taper does **not** reach the damaging regime: its narrowest element is 3.9e-03
of `Rbig`, three decades above the onset, the closure is healthy at 2.2e-08, and the answer
converges in slices. But it is walking toward it exactly as the 2-D round-4 census
described: `|q|max / ceiling` **doubles with every doubling of the slice count**, so the
staircase is a linear walk into the family, and a 4,096-slice taper (or a 64-slice taper on
a cone whose radius change is 64x smaller) lands in it. The 2-D peer's warning band exists
for precisely this surface.

### 4.7 Which scale governs the bar: measured, and NOT settled

The 2-D contract is a pure PERIOD fraction.  The cylindrical peer has three candidate
scales -- a fraction of `Rbig`, a fraction of the local wavelength
`lambda_loc = 2 pi / (k0 n)`, and a fraction of the layer's OWN ordinary element width
(which the DPW = 8 cap sets to `degree * lambda_loc / 8`, the cylindrical peer of the 1-D
guard's `_SLIVER_OWN_SCALE_RATIO`).  `c5_which_scale.py` separates them by sweeping `k0` at
fixed `Rbig` (so `lambda` moves 16x and `Rbig` does not) and `Rbig` at fixed `k0`.  The
onset is the widest rung whose q-matched error exceeds 1e-3 on the width-INDEPENDENT liner
device:

| `Rbig` | `k0` | `Rbig/lambda_core` | onset `w` | `w/Rbig` | `w/lambda` | `w/h_ordinary` | `\|q\|max`/ceiling |
|---|---|---|---|---|---|---|---|
| 24 | 0.5 | 4.68 | 7.200e-06 | 3.000e-07 | 1.404e-06 | 9.600e-07 | 6.686e+06 |
| 24 | 2.0 | 18.72 | 7.200e-05 | 3.000e-06 | 5.615e-05 | 2.400e-05 | 2.152e+05 |
| 24 | 8.0 | 74.87 | 2.400e-06 | 1.000e-07 | 7.487e-06 | 2.933e-06 | 1.254e+06 |
| 12 | 2.0 | 9.36 | 1.200e-06 | 1.000e-07 | 9.358e-07 | 5.333e-07 | 1.003e+07 |
| 48 | 2.0 | 37.43 | 4.800e-05 | 1.000e-06 | 3.743e-05 | 1.500e-05 | 2.617e+05 |

**No scale holds still.**  Spread of the onset: `w/Rbig` **30.0x**, `w/lambda` 60.0x,
`w/h_ordinary` 45.0x, `|q|max/ceiling` 46.6x.  `w/Rbig` is the tightest but only by 1.5x to
2x over the others, so this measurement does NOT select a scale, and any width-fraction bar
carries a factor-30 uncertainty in where its onset actually is.  That is the reason the bar
proposed in section 7.2 leads with `|q|max / (n_max k0)` -- whose ORDINARY population
(19.60 .. 235.8) and DAMAGING population (2.152e+05 .. 1.003e+07 at the onsets above) are
**2.96 decades apart** -- and uses the width fraction only as the attributing conjunct.

### 4.8 The FD basis is structurally immune

`basis='fd'` puts every layer on one uniform radial grid `_fd_grid(Rbig, N)` that does not
know where the walls are, so no cell is ever manufactured by a wall coincidence. Measured
control at every rung of the union ladder: the FD per-order R at `delta/Rbig <= 1e-3` is
**bit-identical** to the `delta = 0` answer (difference exactly `0.0e+00`) — the uniform
grid cannot resolve the shift at all. That is immunity to Class C, and it is also why the
FD arm is **not** usable as an accuracy oracle for the SEM answer: at N = 200 / 400 / 800
its own `R[0]` reads 0.03552 / 0.03636 / 0.04494, still moving in the third decimal, while
the SEM `delta = 0` reference is converged across degrees 6 / 8 / 12 to 0.1600232 /
0.1600086 / 0.1600157.

### 4.9 Verdict — Class C

**NO-GO. The union / enrichment sliver is present, unguarded, and produces wrong answers.**
On the shipped SEM path a wall coincidence of `1e-7` of the domain radius moves the
per-order answer by 586x to 5,428x the physical wall shift, injects spurious axial
wavenumbers five million times the physical ceiling, drives the interface rcond to
6.7e-10, and emits **zero** warnings. The within-layer liner arm shows the same family
with an inverted degree ladder (degree 16 worse than degree 6) from `w/Rbig <= 1e-06`. A
wall-free spacer layer inherits its neighbours' walls and is not a refuge. The FD basis is
structurally immune. The taper staircase is safe at 64 slices but scales into the family.

---

## 5. The kernel matrix

### 5.1 Which kernels were reachable, and which was not

The bundled OpenBLAS 0.3.31 is `DYNAMIC_ARCH` and honours `OPENBLAS_CORETYPE`. The host is
an **AMD Ryzen 9 5950X (Zen 3, no AVX-512)**, and what a requested kernel name actually
produces was read back from `threadpoolctl` in every run rather than assumed:

| requested | obtained (Windows, py3.14) | obtained (WSL, py3.12) | usable |
|---|---|---|---|
| `HASWELL` | Haswell | Haswell | yes |
| `PRESCOTT` | **Katmai** | **Katmai** | yes |
| `NEHALEM` | Nehalem | Nehalem | yes |
| `ZEN` | **Haswell** | **Haswell** | **no — silent fallback** |
| `SKYLAKEX` | SkylakeX (reported) | SkylakeX (reported) | **no — dies on the first LAPACK call** |

Two of the five requests do not do what their name says, and both failures are silent:

* **`OPENBLAS_CORETYPE=ZEN` does not give a Zen kernel on this build.** It reports
  `Haswell`, and a 200x200 complex generalized eigendecomposition under `ZEN` is
  **bit-identical** to the same problem under `HASWELL` (SHA-256 of the sorted spectrum
  `1a3791dffe5855fa` in both). The default with `OPENBLAS_CORETYPE` unset is also
  `Haswell`. **The CI's ZEN kernel therefore could NOT be reproduced here**; see section 7.
* **`SKYLAKEX` loads and reports itself** but the process dies with no output on the first
  real LAPACK factorisation, because this CPU has no AVX-512.

So the matrix below is **three distinct kernels — Haswell, Katmai, Nehalem — across two
builds (Windows / py3.14 and WSL / py3.12), eight runs**, with `ZEN` recorded as a
duplicate of Haswell rather than dropped, so the table is honest about what it contains.
Three genuinely different kernels already move the shipped rule's decisions, which is
sufficient to settle every claim below; the CI's ZEN row remains unmeasured.

### 5.2 Class A — orientation decisions per kernel

24 near-cutoff rungs (`m` = 0, 1, 2 against `qn` from 1.41e-02 down to 1.40e-05), each
solved under the SHIPPED rule and under the CANDIDATE spectrum-scaled band:

| decision | SHIPPED rule | CANDIDATE band |
|---|---|---|
| rungs whose CLASS verdict (propagating vs evanescent) moves with the kernel | **7 / 24** | **0 / 24** |
| rungs whose R/T CHANNEL COUNT moves with the kernel | **21 / 24** | 0 / 24 (fixed at 3) |
| rungs whose orientation verdicts (flux sign, `Im q` sign) move | **24 / 24** | — |
| worst closure over all kernels | **2.1579e-04** | **1.2716e-06** (**170x better**) |

Worst single-rung spread of the lossless closure, with the kernel alone:

| rung | shipped closure across kernels | candidate |
|---|---|---|
| m = 0, delta = 3.16e-06 (`qn` = 2.507e-03) | 2.584e-11 .. 1.217e-04 — **4,708,231x** | 2.584e-11 .. 2.495e-09 (96.6x) |
| m = 0, delta = 1.00e-06 | 1.210e-10 .. 6.848e-05 — 566,008x | 1.210e-10 .. 3.139e-09 (25.9x) |
| m = 0, delta = 1.00e-07 | 1.661e-09 .. 2.166e-05 — 13,044x | 1.461e-09 .. 6.300e-09 (4.3x) |

On 21 of 24 rungs the shipped rule ships either two or three propagating channels depending
on which BLAS kernel ran, and on many of them the forward-oriented mode carries backward
z-flux. The candidate band fixes the channel count at 3 on every kernel and every rung.

### 5.3 Class B — the inverse populations per kernel

| population | `max(R+T)` across all 8 runs | `rcond(a+b)` | residual |
|---|---|---|---|
| staggered (production) | 1.0000000000 .. 1.0000000000 | 8.2571e-04 .. 4.8870e-03 | 4.4316e-16 .. 4.2130e-15 |
| legacy nodal | **7.4033e+01 .. 4.0732e+02** | 2.9730e-06 .. 2.2463e-05 | 5.3783e-14 .. 8.1951e-13 |

Every nodal row reads the **same** `max(R+T)` to four significant figures on all four runs
per build and on both builds (74.03 at 2 wavelengths, 128.8 at 8, 407.3 at 14). The nodal
blow-up is therefore a **deterministic discretisation defect, not an arithmetic one** — it
does not need a kernel to appear, and a passivity screen against it is kernel-stable.

**Would the Cartesian conjunction (`rcond < 1e-10` AND `residual > 1e-8`) refuse any row?
0 of 6 rows, on every kernel, on both builds.** The residual is four to five decades below
the bar on every broken row.

### 5.4 Class C — which candidate bar survives the kernel

This is the question the CI raised, and on BOR it has a clean answer. Worst spread of each
candidate decision quantity across the eight runs, over the whole sliver ladder (degrees 8
and 12, `delta/Rbig` from 1e-2 to 1e-7, adjacent and separated arms):

| candidate decision quantity | worst kernel spread | verdict |
|---|---|---|
| narrowest manufactured element / `Rbig` (**GEOMETRY**) | **1.0000x** | kernel-EXACT |
| `|q|max / (n_max k0)` (spurious wavenumber vs the physical ceiling) | **1.1895x** | kernel-stable |
| q-matched per-order `dR` | 34.37x | not usable as a bar |
| closure / super-unity (**ENERGY VIOLATION**) | **70.90x** | **not usable as a bar** |

The 70.9x row is the CI's finding reproduced on the cylindrical engine: at
`delta/Rbig = 1e-7`, degree 8, the same fixture reads a super-unity anywhere between
2.153e-05 and 1.527e-03 depending only on which BLAS kernel multiplied the matrices. A
trigger at `1e-3` (the 1-D `_SLIVER_TRIGGER_BAR`) sits **inside** that spread, so the same
solve would be arbitrated on one kernel and pass silently on another.

The separated (no-window-union) control reads **1.0000x on every quantity including the
energy** — because it never manufactures a narrow element, there is no ill-conditioning for
a kernel difference to amplify. That is the mechanism of the kernel dependence, isolated.

---
## 6. The other multilayer engines — census

Structural reachability, the guard that exists today, and one measured probe per reachable
class. Measured on the same tree and both builds
(`validation/probe_scope_bor_guards/e_*.json`).

| engine | classes reachable | guard today | measured probe | verdict |
|---|---|---|---|---|
| **BOR `basis='fd'`** (`bor/zcascade.py`, `coupled_radial_eigensolver.py`) | A at `zcascade.py:86`; B at 5 cascade sites; **C unreachable** (one uniform radial grid, no manufactured cell) | none | A: near-cutoff channel count 2 vs 3 across threads and kernels, closure to 2.16e-04; B: 2,031 inverses, rcond >= 1.986e-07; C: per-order R bit-identical to the `delta = 0` answer for every `delta/Rbig <= 1e-3` | **A: NO-GO. B: unguarded by measurement. C: NOT-APPLICABLE** |
| **BOR `basis='sem'`** (`bor/sem_radial.py`, `bor_stack._solve_sem`) | A at `sem_radial.py:428`; B at 7 sites; **C at the `+-1` window union AND within one layer's own mesh** | `inv(Mz)` has an LU-pivot / QZ fallback; nothing else | A: same band, same defect; B: rcond >= 6.677e-07, residual <= 3.408e-13; C: `\|q\|max`/ceiling to **5.014e+06**, move/physical to **5,428x**, 0 warnings | **A: NO-GO. B: unguarded by measurement. C: NO-GO** |
| **BOR legacy nodal** (`bor_solve.build_layer(basis='nodal')`) | A at `zcascade.py:227`; B at `inv(a+b)`; C unreachable | a `UserWarning` past `Rbig/lambda > 4`, and a changed default | `max(R+T)` = **5.06 .. 966.7**; unwarned at 1, 2 and 4 wavelengths; rcond(a+b) 2.97e-06, residual 8.20e-13 | **NO-GO — wrong answer returned; needs a PASSIVITY refusal, not the conditioning conjunction** |
| **BOR JAX twins** (`_jax_bor.py`, `_jax_sem.py`) | A at `_jax_bor.py:98`, `_jax_sem.py:247` (own copies) | none | R/T parity with NumPy `dR, dT <= 1.5e-13` on four (basis, m) combinations | **A: NO-GO (inherits); parity GO** |
| **EME** (`elements/eme/eme_2d.py`, `eme_diffraction.py`) | **A at `eme_2d.py:130`** (`np.where(ky.imag < 0.0, -ky, ky)` — an EXACT-ZERO pin) and `eme_diffraction.py:176` (same pin on `qz`); B at `eme_2d.py:142/143/145/159/160`, `eme_diffraction.py:192`; C unreachable | **none** — although the module's own vector sibling `eme_2d_vector.py:255` already carries the correct relative band | A: `layer_modes` at Nx=96, k0=20pi, ky0=0.37 returns **62 modes** with real eps against **69 (WIN) / 71 (WSL)** with eps + i1e-30; gap **2.55% / 3.64%**, and the mode lists differ BETWEEN BUILDS; deciding `\|Im ky\|/\|ky\|` = 1.34e-16 against a physical 1.81e-29. B: `a+b` cond 2.09 .. 19.34 healthy, 1.86e+04 at a band edge, `LinAlgError` at the edge | **A: NO-GO (defect). B: GO, census-only. C: NOT-APPLICABLE** |
| **Berreman** (`elements/berreman.py`, `_berreman_jax.py`) | A at `:594`; B at T22 and the Redheffer star; C unreachable (planar, one order) | **takes the shared `_sqrt_decay`** (identity-checked; no private copy in either file), `_split_fwd_bwd` already a relative band, T22 **armed** at 1e-10 | killer fixture (layer eps = eps_sup = eps_sub = 2.25 exactly, +-1e-6 detune, theta = 0 / 0.35, off-plane exz = 0.6): closure **2.2e-16 .. 2.2e-15**, T22 rcond 0.698 .. 1.0, **0 refusals**; WIN vs WSL **0.00e+00**; JAX twin agrees to 2.6e-15, `jax.grad` to 1.9e-10 | **GO — already guarded and verified** |
| **RCWAStack** (`rcwa/stack.py`, `_core.py`) | A at `_core.py:1284`; B at 4 cascade sites; C unreachable (pure Fourier basis) | `_sqrt_decay` (the definition); `_guarded_inverse` at the star, at `a+b`, and **armed** at T22 (1e-10) | same fixture: closure **4.4e-16 .. 3.9e-15**, `a+b` rcond 1.000, 0 refusals; WIN vs WSL <= 6.7e-16 | **GO — the reference** |
| **PMMStack (1-D)** (`pmm/stack.py`) | A, B, C all reachable (`_pmm_union_grid`) | shared `_sqrt_decay`; guarded `a+b` + armed T22 + `_guarded_lstsq`; `PMM_SLIVER_GUARD = True` with the full bar set | A/B: closure 4.2e-15 .. 3.5e-14, rcond 0.36 .. 1.0, 0 refusals. C: **REFUSED at `delta` = 1e-4 and 1e-5**, snapped + warned at 1e-6; identical refusal sets on both builds | **GO — armed and two-sided** |
| **PMM2DStackHybrid** (`pmm/stack2d.py`) | A (shared root); B at `a+b`/T22 plus unguarded `stack2d.py:839 inv(EpsF)` (Laurent only) and `twod.py:393/408`; **C NOT reachable** | shared `_sqrt_decay`; cascade guarded | A/B: closure 3.257e-10, `a+b` rcond 7.6e-03 .. 1.0 at width 162, 0 refusals, WIN = WSL to 1.3e-15. C: intra-layer sliver to `delta` = **1e-14** of the period converges (closure 7.5e-3 -> 6.67e-11) although `cond(M)` reaches 1.06e+15 — the layer modes live in **Rayleigh order space**, so a narrow element supplies only Galerkin-projected entries and cannot inject a spurious modal wavenumber | **A/B: GO. C: NOT-APPLICABLE (measured)** |
| **PMM2DStackPure** (`pmm/stack2d_pure.py`) | A, B, C all reachable (per-layer L2 mortar) | `_forward_branch_flip` / `_select_forward_flux` relative bands; `_guarded_mortar_solve` (1e-12 rcond in-plane, 1e-6 residual generalized) + `_guarded_lstsq` + armed T22; `_STAG_MIN_SEG_FRAC = 1e-3` refuse + `_STAG_SLIVER_BAND_FRAC = 3e-2` warn | A/B: closure 1.3e-15 .. 1.1e-14, rcond 2.9e-03 .. 1.0 at width 800, 0 refusals. C: silent >= 3e-2, **warned on [1e-3, 1e-2]**, **REFUSED below 1e-3**; identical bars on both builds | **GO — armed and two-sided on all three** |
| **thin_grating** (`elements/thin_grating.py`) | **none** | n/a | zero `np.linalg` call sites, zero sqrt branch, no cascade: order amplitudes are analytic Fourier coefficients of a phase screen and `R_eff = np.zeros(N)` by construction | **NOT-APPLICABLE (all three)** |
| **coatings** (`elements/coatings.py`) | **A only**, at `:177-:181` (exact `(n*ct).imag < 0.0` pin plus an absolute `abs(ct) < 1e-12 -> 1e-12` floor) | that pin and that floor | `coating_reflectance` at and +-1e-3 / +-1e-9 around the exact critical angle (n_amb 1.5, n_lay 1.0, d = 50 nm and 1 um): `R + T = 1.0 +- 7e-16` on every row, bit-identical WIN vs WSL and t1 vs t4 | **A: GO** (the argument is exactly real for real `n`, so the pin is decided in exact arithmetic). **B / C: NOT-APPLICABLE** — a 2x2 Abeles transfer-matrix product with a closed-form scalar r/t and zero `inv`/`solve`/`lstsq` sites |

**Cross-engine determinism.** Worst `|R+T|` difference over every Cartesian cascade row:
WIN t1 vs t4 **9.881e-15**, t1 vs t8 **1.232e-14**, WIN vs WSL **1.377e-14** — all at the
arithmetic floor for a width-800 modal block. **0 refusals** on every engine, every thread
count, both builds.

**The second wrong answer this scoping found, outside BOR.**
`lumenairy/elements/eme/eme_2d.py:130` `_ky_forward` decides the forward lateral root of a
**propagating** strip mode on the exact sign of an imaginary part that, for such a mode, IS
the eigensolver's backward error — the same exact-zero pin round 1 removed from
`_sqrt_decay`. Fixture: `eme_2d.layer_modes`, `Nx=96`, `Lx=Ly=1`, `k0=20pi`, `kx0=0`,
`ky0=0.37`, two strips `[(e1, 0.5), (e2, 0.5)]` with `e1 = 2.25 | 12.0` split at `Lx/2` and
`e2 = 2.25 | 12.0 | 2.25` on `[Lx/4, 3Lx/4)`, window `(26055.8, 35530.6)`, `n_scan=300`.
Real eps (`eigh`, `Im(lam)` exactly zero) gives **62 modes, identical on both builds**; the
same eps plus `i*1e-30` on the high region only (`scipy.linalg.eig`) gives **69 modes on
WIN and 71 on WSL**, a gap of **890.55 absolute / 2.55 % relative (WIN)** and
**1273.91 / 3.64 % (WSL)**, with the mode lists differing between builds (first mode
26137.93 WIN against 26065.96 WSL). Flip counts are build-dependent (29 WIN / 30 WSL at
Nx = 96; 32 WIN / 22 WSL at Nx = 128; 51 / 44 at Nx = 128, k0 = 40pi); the thread count
does not move it, the build does. Onset: 0 flipped propagating modes at `Nx <= 64`, flips
from `Nx = 96` up as the `eig` backward error grows with `||A|| ~ 4 Nx^2 / Lx^2`. The fix
is a one-line port of the module's own vector sibling
(`eme_2d_vector.py:255`, `tol = 1e-9 * max(1, max|ky|)`) or of the shared `_sqrt_decay`
band. `eme_diffraction.py:176` carries the same pin on `qz`.

Secondary, and a degradation rather than a wrong answer: approaching a strip band edge,
EME's unguarded `cond(Vb)` runs 101 -> **9.49e+07** and `cond(a+b)` 1.00 -> **1.86e+04**,
with `sigma_min(M)` flooring at 5e-5 .. 8e-5 over four decades of approach and non-monotone;
at the edge the solve raises `LinAlgError('Singular matrix')`. Identical on WIN, WSL and
t4.

---
## 7. Proposed build plan for 5.45.1

### 7.1 One flux-orientation rule, with `xp=`

Replace the five copies of section 2.1 with a single implementation, in the shape round 2
gave `_sqrt_decay(x, xp=None, band=...)`:

```python
# lumenairy/elements/bor/_orient.py  (new module)
_BOR_CUT_BAND_REL = 1e-8

def forward_orient(q, flux, k0, *, xp=None, band=_BOR_CUT_BAND_REL):
    """Forward-orient a layer's axial wavenumbers: propagating modes by the
    sign of their own r-dr z-flux, evanescent modes by decay in +z."""
    if xp is None:
        xp = array_namespace(q)
    scale = xp.maximum(xp.max(xp.abs(q)), k0) if q.size else k0
    prop = xp.abs(xp.imag(q)) <= band * scale
    flip = xp.where(prop, flux < 0.0, xp.imag(q) < 0.0)
    return xp.where(flip, -q, q)
```

* **Callers.** `zcascade.py:86`, `zcascade.py:227` and `sem_radial.py:428` detect `xp` from
  the array; `_jax_bor.py:98` and `_jax_sem.py:247` pass `xp=jnp` explicitly so the traced
  body is the same object the eager path runs. Only `q.size` is read as a Python value and
  that is static under tracing, so the body is `jit`- and `grad`-safe.
* **The floor is `k0`, not `1.0`.** Every Cartesian peer floors the spectrum scale at 1.0
  because its eigenvalue is dimensionless. `q` here has units of inverse length, so a
  literal 1.0 would make the band unit-system-dependent — exactly the failure audit P2-06
  fixed for the channel gate ("absolute thresholds on `q` silently returned empty R/T for
  small-k0 unit systems"). `k0` is the natural non-zero floor and is what was measured.
* **Consolidate the two companion constants the same way**: the flux-normalizer fallback
  `1e-10 * fnrm` (6 copies) into one `flux_normalize(...)`, and the shared
  `{imag, real-floor}` core of the channel gate (5 copies) into one helper, leaving each
  basis's own leg (`reldiv` for nodal, index ceiling for staggered) at the call site — the
  split audit S1-16 already documents and justifies.

**Measured effect of the band change** (`a5_arbiter_band.py`, `k1`/`k2`):

| measurement | shipped | candidate | margin |
|---|---|---|---|
| worst closure, 39-rung cutoff ladder, one kernel | 1.2167e-04 | 1.9655e-07 | **619x** |
| worst closure over 3 kernels x 2 builds | 2.1579e-04 | 1.2716e-06 | **170x** |
| rungs whose CLASS verdict moves with the kernel | 7 / 24 | **0 / 24** | — |
| rungs whose CHANNEL COUNT moves with the kernel | 21 / 24 | **0 / 24** | — |
| ORDINARY battery (12 solves, m = 0,1,2,5 x k0 = 0.8, 2.0, 3.5) moved | — | **0 of 12** | bit-identical |
| re-implementation vs library with `rule='shipped'` | `dR = dT = 0.0` | — | the arbiter is valid |

### 7.2 A spurious-wavenumber contract for the SEM mesh, attributed by geometry

Two candidate decision quantities survive the kernel matrix: the narrowest manufactured
element as a fraction of `Rbig` (kernel-EXACT, 1.0000x) and `|q|max / (n_max k0)`
(kernel-stable, 1.1895x).  The energy violation does not (70.90x) and must not be used.

Between the two survivors, the one with a two-sided gap is `|q|max / (n_max k0)`:

| population | `\|q\|max` / `n_max k0` | source |
|---|---|---|
| ORDINARY -- the separated control, every rung, degrees 8 and 12 | **19.60 .. 27.22** | `c4` |
| ORDINARY -- a 64-slice taper staircase (walking toward the family) | **235.8** | `c3` |
| ORDINARY -- ring gratings and segment layers, degrees 6-16 | 16.0 .. 36.5 | `c1`, `c2` |
| DAMAGING -- the onset of measurable damage over five (Rbig, k0) cases | **2.152e+05 .. 1.003e+07** | `c5` |
| DAMAGING -- the union ladder at `delta/Rbig <= 1e-5` | 2.521e+04 .. 5.014e+06 | `c1`, `c4` |

**Two-sided gap: 235.8 to 2.152e+04 is 1.96 decades against the union ladder's mildest
damaging rung, and 235.8 to 2.152e+05 is 2.96 decades against the c5 onset population.**
A bar at

```
_BOR_Q_EXCESS = 1.0e+04          # |q|max / (n_max k0), the spurious-wavenumber screen
```

sits **1.63 decades above** the worst ordinary geometry measured and **0.33 decades below**
the mildest damaging rung of the union ladder (1.33 decades below the mildest c5 onset).
This is the cylindrical peer of the 1-D guard's `_SLIVER_Q_EXCESS`, and its idea is the
same: a mode called propagating that no propagating mode of this medium can be.

**The geometry is the ATTRIBUTION, not the detector.**  As with the 1-D union-grid guard,
the screen alone would fire on a stack that is merely under-resolved, so the refusal is a
CONJUNCTION with the geometric cause:

```
_BOR_MIN_ELEM_FRAC    = 1e-6     # a cell MANUFACTURED below this fraction of Rbig
_BOR_SLIVER_BAND_FRAC = 1e-3     # WARN in [_BOR_MIN_ELEM_FRAC, this)
```

  (a) the post-window, post-DPW, post-`equalize_meshes` breakpoint set contains a cell
      narrower than `_BOR_MIN_ELEM_FRAC * Rbig` **whose two walls come from DIFFERENT
      layers** (the own-scale test: the union manufactured it, no single layer asked for
      it); **and**
  (b) that layer's modal spectrum reads `|q|max / (n_max k0) > _BOR_Q_EXCESS`.

Neither conjunct is the energy violation, and both are kernel-stable.

**Where the width fraction stands, honestly.**  Section 4.7 measured the onset at
`w/Rbig` from 1.000e-07 to 3.000e-06 over five `(Rbig, k0)` cases -- a **30x** spread, and
no other scale is tighter.  `1e-6` is chosen as the conservative (widest) end of that
measured band, so the geometric conjunct fires at or before the onset on every case
measured; but it is a factor-30 quantity, which is exactly why it is the conjunct that
ATTRIBUTES rather than the one that DECIDES.

**Derivation of the warn edge, and its binding constraint.**  On the union ladder the
answer tracks the physical wall shift (continuity slope 7.777e-02 per unit shift) with a
move factor of 2.0x to 2.5x down to `delta/Rbig = 1e-05`, 3.9x to 5.5x at 1e-06, and
**586x (degree 12) / 5,428x (degree 8)** at 1e-07 -- against the 1-D guard's
`_SLIVER_MOVE_FACTOR = 100`.  On the within-layer liner ladder the DEGREE ladder inverts
from `w/Rbig = 1e-06` (degree 16 becomes 144x worse than degree 6).  The warn edge is set
by the FALSE-POSITIVE census, not by the accuracy, exactly as the 2-D round-4 note records:

| ordinary geometry | narrowest element / `Rbig` | margin above the 1e-3 warn edge |
|---|---|---|
| uniform / ring layer at degree 8 (DPW-capped) | 4.464e-02 | 44.6x |
| uniform / ring layer at degree 12 | 6.250e-02 | 62.5x |
| 32-slice taper | 7.812e-03 | 7.8x |
| **64-slice taper staircase** | **3.906e-03** | **3.9x** |

The 64-slice taper is the binding ordinary geometry at **3.9x**, and it walks:
`|q|max / ceiling` **doubles with every doubling of the slice count** (29.7 / 34.4 / 59.3 /
118.1 / 235.8 at 4 / 8 / 16 / 32 / 64 slices), so a 256-slice taper enters the warn band.
That is the intent -- the same surface the 2-D contract deliberately warns on from nine
slices -- but the warn edge must be a warning and never a refusal, and the census must be
re-run on the running build by the test suite, as `test_fix_pmm2d_mortar_round3.py` does
for the Cartesian peer.  **The 3.9x margin is sample-scoped** (four geometry families
against the 2-D census's 47) and section 8 lists widening it as a precondition.

**Do NOT key any Class-C decision on super-unity.**  Kernel spread of the closure on this
ladder: **70.90x**, straddling the 1-D `_SLIVER_TRIGGER_BAR` of 1e-3, so the same solve
would be arbitrated on one kernel and pass silently on another.  Independently, on the
union ladder the closure sits at its healthy 1e-10 .. 1e-9 baseline while the per-order
answer has already moved by 4e-04 (`delta/Rbig = 1e-04`, degree 12) -- the damage is
**energy-invisible** over four decades of the ladder, exactly as the 2-D mortar's is, and a
super-unity screen at the 1-D bars fires on only **4 of 60** union rungs and **2 of 40**
within-layer rungs.

**The window itself needs a second look.**  `win[i] = walls[i-1] | walls[i] | walls[i+1]`
means a WALL-FREE layer inherits both of its neighbours' walls, so a uniform spacer placed
between two ring layers carries the sliver in a mesh that has no walls of its own.  Any
contract must be applied to the POST-window, POST-DPW, POST-`equalize_meshes` breakpoint
set, not to the user's wall list.

### 7.3 A passivity refusal on the legacy nodal cascade

`bor_solve.solve` returns `R + T` from 5.06 to 966.7 on the legacy nodal basis, and below
`Rbig = 4` vacuum wavelengths it returns it with **no warning at all** (74.03 at 2
wavelengths, 16.38 at 4, 5.06 at 1). The conditioning conjunction cannot see this
(section 3.4); the detector that separates is passivity, and it is kernel-exact:

| population | `max(R+T)` |
|---|---|
| staggered, all kernels and builds | 1.0000000000005 |
| nodal, all kernels and builds | 5.06 .. 966.7 |

`_STACK_SUPERUNITY_BAR = 1e-2` sits **2.6 decades above** the staggered ceiling and **2.6
decades below** the nodal floor. Proposal: on a provably passive stack with a lossless
propagating incidence medium, `bor_solve.solve` should REFUSE with a named error naming
`basis='nodal'` and the staggered remedy, rather than return the number; and the existing
`Rbig / lambda > 4` warning should be replaced by the measurement rather than kept as a
proxy that misses the smallest two thirds of its own population. This is a behaviour change
on a documented escape hatch and should carry the same fail-before switch the 1-D guard
carries (`PMM_SLIVER_GUARD`).

### 7.4 The EME branch-cut pin (outside BOR, same class, same wave)

`eme_2d.py:130` and `eme_diffraction.py:176` carry the exact-zero pin round 1 removed from
`_sqrt_decay`, and it produces a build-dependent mode set (62 against 69 / 71 modes, 2.55 %
/ 3.64 % gap, mode lists differing between WIN and WSL). The module's own vector sibling at
`eme_2d_vector.py:255` already has the correct relative band. This is a one-line port and
belongs in the same release as section 7.1, because it is the same defect.

### 7.5 Sites that should stay UNGUARDED, by measurement

* Every explicit inverse on the **production** FD and SEM cascades: `inv(a+b)`, both
  Redheffer denominators, `inv(Mz)`, `inv(I + gamma alpha)`, the four mortar mass
  projections, `alpha`, `gamma`, `Lei`, and `layer_absorption`'s solve. One population,
  2,031 inverses over 132 fixtures, rcond 1.986e-07 .. 1.000, residual <= 3.408e-13 — three
  to seven decades clear of every Cartesian bar. Port the **census hook** only (`None` by
  default, one `is None` test per inverse) so the population can be re-measured rather than
  assumed.
* `sem_radial.py:348` (`inv(Mz)`) already carries a correct, different remedy (LU-pivot
  detection with a fallback to the unreduced QZ pencil). Nothing to port.
* `basis='fd'`: structurally immune to Class C. No contract needed, and none should be
  added.

### 7.6 Tests to add

| test | asserts |
|---|---|
| `test_scope_bor_orientation_band.py::test_near_cutoff_closure` | a fixture at `qn ~ 2.5e-03` reads lossless closure < 1e-08 (shipped: 1.2167e-04) |
| `...::test_near_cutoff_channel_count_is_stable` | the channel count is 3 at `OPENBLAS_NUM_THREADS` 1 and 4 (shipped: 2 against 3) |
| `...::test_no_forward_mode_carries_backward_flux` | over the a1 census fixtures, 0 modes with `prop` and `flux < 0`, and 0 crossed orders whose flux disagrees with the shipped orientation |
| `...::test_band_two_sided_population` | the noise side stays at least 2 decades below the band and the signal side at least 2 above, re-measured on the running build |
| `...::test_one_orientation_implementation` | the five sites call the one helper — the peer of the round-2 `_sqrt_decay` single-definition test |
| `test_scope_bor_sem_min_element.py::test_contract_refuses` | `delta/Rbig = 1e-07` raises, naming the manufactured element |
| `...::test_degradation_band_warns` | `delta/Rbig` in [1e-6, 1e-3) warns and still returns |
| `...::test_ordinary_geometry_census` | every geometry in the shipped battery, INCLUDING a 64-slice taper, lands outside the band on the running build |
| `...::test_wall_free_spacer_inherits_walls` | pins the window's reach, so a future change to `win[i]` is caught |
| `...::test_fd_basis_is_immune` | the FD arm is unchanged by any `delta` |
| `test_scope_bor_nodal_passivity.py::test_nodal_superunity_refused` | `Rbig` = 2, 8, 14 lambda nodal stacks raise; the staggered twins do not |
| `test_bor_sem_jax.py` (extend) | JAX twin R/T parity after the consolidation (baseline `dR, dT <= 1.5e-13`) |
| `test_eme_branch_cut.py` | the `eme_2d` mode count is build-independent under an infinitesimal `Im(eps)` |

### 7.7 The bit-identity contract

**148 BOR gates pass on this tree today** (`test_bor_sem.py`, `test_bor_solve.py`,
`test_bor_anisotropic.py`, `test_audit_bor_grazing_cutoff.py`, `test_audit_p1_bor_flux.py`,
`test_audit_w6_bor.py`, `test_niche_audit_w6_bor.py`,
`test_audit_v5_24_2_b2_bor_exports.py`; 340.6 s, 1 warning), plus `test_bor_sem_jax.py` and
`test_v5_20_11_bor_jax.py`.

The band change must leave every one of them **bit-identical**, and that must be
established by RUNNING them, not asserted — audit
`AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md` made exactly this point for the
previous classifier change ("the k0 = 2.0 suites have no near-cutoff modes — verified by
running, not assumed"). The evidence in hand that this will hold: **0 of 12** ordinary
solves moved by the candidate band over `m` = 0, 1, 2, 5 and `k0` = 0.8, 2.0, 3.5, and the
whole 106-fixture orientation census sits 3.24 decades below the band on the noise side.
The JAX twins must be re-checked for parity after the consolidation because they move from
their own copies of the rule to the shared one; the pre-change parity is `dR, dT <= 1.5e-13`
on four (basis, m) combinations.

The Class-C contract is NOT bit-identity-preserving where it fires, by design: it converts a
returned wrong number into a raise. Everywhere it does not fire it must be bit-identical,
and the ordinary-geometry census is the gate that proves it.

### 7.8 Suggested order of work

1. The consolidation (7.1) with the band unchanged at the shipped `1e-9 * |Re q|` — a pure
   refactor, 148 gates bit-identical. This is the step that makes 2 a one-line change.
2. The band change to `1e-8 * max(max|q|, k0)` (7.1) with the near-cutoff and thread gates,
   and the EME pin (7.4) in the same commit — same class, same fix.
3. The nodal passivity refusal (7.3) — self-contained, on a legacy path, behind a switch.
4. The SEM minimum-element contract (7.2), with the ordinary-geometry census FIRST, because
   the census is the binding constraint on the warn edge and must be measured on the shipped
   battery before the edge is fixed.
5. The census hooks (7.5).

---

## 8. What could not be measured

* **The CI's ZEN kernel.** `OPENBLAS_CORETYPE=ZEN` silently returns the Haswell kernel on
  this host's OpenBLAS 0.3.31 (bit-identical 200x200 complex `eig`; the unset default is
  also Haswell), and `SKYLAKEX` loads but dies on the first LAPACK factorisation because
  this CPU (Ryzen 9 5950X, Zen 3) has no AVX-512. The kernel matrix therefore covers
  Haswell, Katmai and Nehalem across two builds. Three kernels were enough to move the
  shipped rule's decisions on 24/24 rungs and to spread the Class-C closure by 70.90x, so
  the conclusions stand; but the specific ZEN row the CI reported is **not reproduced here**
  and the candidate bars' ZEN behaviour is inferred from their kernel-exactness on the three
  kernels measured, not observed. Re-running `k1_kernel_matrix.py` on the EPYC CI runner is
  the outstanding item.
* **Which of `Rbig`, the local wavelength, or the layer's own ordinary element width
  governs the Class-C width bar.**  `c5_which_scale.py` DID complete and the answer is that
  **none of the three holds still**: over five `(Rbig, k0)` cases spanning `Rbig/lambda`
  from 4.7 to 74.9, the onset spreads by **30.0x** in `w/Rbig`, 60.0x in `w/lambda` and
  45.0x in `w/h_ordinary` (section 4.7).  `w/Rbig` is the tightest but only by 1.5x to 2x,
  so a width-fraction bar cannot be pinned better than a factor of 30 by this measurement.
  That is why section 7.2 makes the width fraction the ATTRIBUTING conjunct and puts the
  DECISION on `|q|max / (n_max k0)`, whose two populations are 1.96 to 2.96 decades apart.
  Settling the width scale would need a wider `(Rbig, k0, degree)` design than the five
  cases run here.
* **A false-positive census over the library's whole shipped BOR geometry battery.** Four
  ordinary geometry families were measured (uniform, ring grating, explicit segments, taper
  at 4-64 slices); the 2-D peer's census covers 47. The 3.9x margin the 64-slice taper
  leaves is a sample property, not a library property, and the census must be widened before
  the warn edge is fixed — this is exactly the round-4 correction the 2-D contract needed
  ("the census margin is sample-scoped, and it is 1.67x, not 3.6x").
* **An armed refusal bar for the EME class-B inverses.** The population is measured
  (`a+b` cond 2.09 .. 19.34 healthy, 1.86e+04 approaching a band edge) but there is no
  two-sided gap: the EME scalar mode solver has no energy oracle, so the only separator
  available was distance-to-band-edge, which is a continuum, not two populations.
* **An accuracy oracle for the SEM answer.** `basis='fd'` is immune to Class C and was used
  as the structural control, but it is not converged at these settings: its own `R[0]` reads
  0.03552 / 0.03636 / 0.04494 at N = 200 / 400 / 800, still moving in the third decimal,
  while the SEM `delta = 0` reference is converged across degrees 6 / 8 / 12 to 0.1600232 /
  0.1600086 / 0.1600157. Class C's wrongness is therefore established by CONTINUITY (the
  answer moved 586x to 5,428x the physical wall shift) and by the separated-arm control, not
  against an external oracle. `stepindex_oracle` / `fiber_oracle` cover the bound-mode
  spectrum of a step-index fibre and do not reach the multi-layer cascade fixtures used
  here.
* **The anisotropic and PML paths.** `SemRadialMesh(R_pml=...)` is reachable only by direct
  construction (`BORStack._solve_sem` never sets it), and `equalize_meshes` rebuilds meshes
  as `SemRadialMesh(b, eps, msh.p)` — **dropping `R_pml`, `sigma_max`, `pml_p` and
  `nq_extra`**. That is a latent trap for any caller who builds PML meshes and equalizes
  them by hand; it is not reachable through `BORStack` today and was not measured further.
* **`m` beyond 10, degrees beyond 16, `N` beyond 200 on the Class-C ladders**, and the JAX
  twins under Class-C fixtures (parity was measured on ordinary geometry only).
* **`pmm/stack2d.py:839 inv(EpsF)`** — reached only with `formulation="laurent"`; every
  fixture used the default `"li"`, so that site is unexercised. Likewise the non-Berreman
  JAX twins and `eme_2d_vector.layer_vector_modes` (a different B surface that bypasses the
  `_interface` / `_star` cascade entirely).
