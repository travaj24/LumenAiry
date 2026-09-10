# FIX — the `PMMStack` near-coincident-wall (SLIVER) defect, O-11

**Date** 2026-09-11 · **Branch** `fix/pmmstack-sliver-walls` (from
`wave2/pmm2d` @ `3034bcb`) · **Scope** `lumenairy/elements/pmm/stack.py` and
the 1-D-only helper `_pmm_union_grid` in `lumenairy/elements/pmm/_core.py`

**Reported as** open item **O-11** of
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md` (F5.5), whose
reproducer is `validation/probe_pmm2d_staggered_mortar/f5f_attrib.py`.
**This fix's reproducers** are `validation/probe_pmmstack_sliver/` (ten probes,
a README, and the JSON every table below is read from).

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Summary

| | |
|---|---|
| **The defect** | Two adjacent `PMMStack` layers whose wall sets differ by `delta` of the period put a SLIVER element of exactly that width on the shared union grid. Past a degree-dependent onset the cascade returns a deterministic, energy-violating wrong answer — up to **8.62** in absolute per-order efficiency. |
| **The mechanism** | The element Jacobian `J = w P / 2` scales the GLL mass by `J` and the stiffness by `1/J`, so the nodal `Kx^2` grows as `1/w^2` and the layer's modal spectrum acquires SPURIOUS wavenumbers `\|q\| ≈ 0.65 · N(N+1)/4 / (k0 J)` — 7.7e+04 at `w` = 1e-4 of a 1.2 µm period against a physical index ceiling of 3. The interface mode-match then conditions as `1/w^2` and its S-matrix stops being bounded. |
| **What O-11 got wrong** | It is **not** energy-invisible in the 1-D stack. Its own `R+T` reads **2.17 / 3.61 / 23.4** and `_warn_stack_energy` fires on every wrong row. The 1.6e-08 closure in the O-11 table is the 2-D MORTAR arm's; `f5f_attrib.py` sets `warnings.simplefilter("ignore")` at module scope, which is why nobody saw the warning. |
| **What ships** | A **screen-and-REFUSE** guard: the CONJUNCTION of (a) a union cell the union manufactured, ≥ 100× finer than any wall spacing the input geometry asked for, and (b) super-unity above the library's own 1e-2 bar on a PROVABLY PASSIVE stack with a lossless propagating incidence medium — where `R+T ≤ 1` is a theorem. The refusal names the exact `min_feature` that removes the cell. Fail-before switch `PMM_SLIVER_GUARD`. |
| **Separation** | max `\|R+T-1\|` among CORRECT rows **4.125e-06**; min `R+T-1` among WRONG rows **1.159e+00**. The bar at 1e-2 sits **3.39 decades** above the first and **2.06 decades** below the second — **identical on both builds**. |
| **Bit-identity** | 18 shipped fixtures, hashed against the READ-ONLY main clone at `D:/…/Lumenairy` — **18/18 identical**. |
| **Tests** | `tests/unit/test_fix_pmmstack_sliver_walls.py`, **18 tests**, 6.63 s (Windows) / 7.25 s (WSL). |
| **2-D stacks** | Neither can form this hazard: `PMM2DStackHybrid` is Fourier-projected (no union grid at all) and `PMM2DStackPure`'s union IS the caller-supplied common `(Nx, Ny)` PIXEL lattice — aspect ratio exactly **1.0**. No reproducer to report. See S7 for the caveat that matters to the in-flight mortar work. |

---

## S1. The two builds

Every measured table in this document was taken on both.

| | Windows | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| BLAS/LAPACK | scipy-openblas 0.3.31.188.0, kernel **Haswell**, MAX_THREADS 24 | scipy-openblas 0.3.31.188.0, kernel **SkylakeX**, MAX_THREADS 64 |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` | same |

Different OpenBLAS microarchitecture kernels, hence genuinely different
reduction orders — which is what makes the agreement below evidence rather than
one build's sampling.

---

## S2. The fixture, and reproduction

Two slices of the F5 taper on a `P` = 1.2 µm period, `λ` = 0.85 µm,
`θ` = 0.15 rad, `dz` = 80 nm, `ε` = 2.25 / 9.0. Layer 0's walls are
`(0.27865, 0.62505)` of the period; layer 1's are opened by `delta`:
`(0.27865 - delta, 0.62505 + delta)`. The **`delta → 0` limit — the two layers
with IDENTICAL walls — is an EXACT reference**, because the physical structure
is continuous in `delta`.

`p1_repro.py` reproduces `f5f_attrib.py`'s oracle arm to every digit it
printed, and adds the column that was missing:

| `delta` | sliver (m) | self-gap `deg 12 vs 14`, SHARED | self-gap, PER-LAYER | `orc(d) − orc(0)` | **`\|R+T−1\|`** |
|---|---|---|---|---|---|
| 1.00e-02 | 1.20e-08 | 2.84e-08 | 2.84e-08 | 9.26e-03 | 2.74e-12 |
| 2.60e-03 | 3.12e-09 | 3.53e-08 | 3.53e-08 | 2.83e-03 | 3.04e-10 |
| 1.00e-03 | 1.20e-09 | 4.34e-08 | 4.34e-08 | 1.13e-03 | 1.71e-09 |
| 3.00e-04 | 3.60e-10 | 6.43e-08 | 6.43e-08 | 3.44e-04 | 1.06e-08 |
| **1.00e-04** | 1.20e-10 | **4.79e-01** | **4.79e-01** | **4.79e-01** | **1.17e+00** |
| **3.00e-05** | 3.60e-11 | **8.17e+00** | **8.17e+00** | **8.62e+00** | **2.24e+01** |
| **1.00e-05** | 1.20e-11 | 1.18e-04 | 1.18e-04 | **9.30e-01** | **2.61e+00** |
| 3.00e-06 | 3.60e-12 | 5.47e-08 | 5.47e-08 | 3.46e-06 | 2.00e-15 |
| 1.00e-06 | 1.20e-12 | 5.47e-08 | 5.47e-08 | 1.15e-06 | 7.11e-15 |
| 0 | 0 | 5.47e-08 | 5.47e-08 | 0 | 3.22e-14 |

WSL reads the same five significant figures in every cell (last column
3.00e-15 / 4.00e-15 on the two round-off rows).

**Three readings from this table alone.**

1. `delta = 1e-5` is the row O-11 called a converged-looking wrong answer: the
   self-gap is 1.18e-04 while the answer sits 9.30e-01 from the limit. It is
   real, and it is **not quiet** — `R+T` is 3.61.
2. The two `layer_grids` spellings agree to 16 digits **because a 2-layer
   window IS the whole union** at `window_halfwidth = 1`. The per-layer path is
   not a second opinion here; see S6.
3. `delta ≤ 3e-6` is clean because the library-default `min_feature`
   (`period × 1e-5`) SNAPS it. `delta = 1e-5` sits exactly on that threshold
   and the float comparison merges **one** of the two wall pairs and leaves the
   other — an asymmetric geometry nobody asked for. That is a second, smaller
   defect of the default and is logged as open item **A** in S9.

---

## S3. The mechanism, measured

### S3.1 The union grid and its operators (`p2_mech.py`, degree 14)

| `delta` | union cells | narrowest cell `w` | `k0 J` | `cond(S0)` | `‖Kx²‖` | `\|q\|max` L1 | `cond` interface | max entry of interface S |
|---|---|---|---|---|---|---|---|---|
| 1.00e-02 | 5 | 1.00e-02 | 4.44e-02 | 1.37e+02 | 5.21e+05 | 7.93e+02 | 1.28e+05 | 1.69e+00 |
| 2.60e-03 | 5 | 2.60e-03 | 1.15e-02 | 5.36e+02 | 7.71e+06 | 2.97e+03 | 1.82e+06 | 2.37e+00 |
| 1.00e-03 | 5 | 1.00e-03 | 4.44e-03 | 1.40e+03 | 5.21e+07 | 7.69e+03 | 1.23e+07 | 3.13e+00 |
| 3.00e-04 | 5 | 3.00e-04 | 1.33e-03 | 4.67e+03 | 5.79e+08 | 2.56e+04 | 1.36e+08 | 3.94e+00 |
| **1.00e-04** | 5 | 1.00e-04 | 4.44e-04 | 1.40e+04 | 5.21e+09 | 7.66e+04 | 1.22e+09 | **3.86e+02** |
| **3.00e-05** | 5 | 3.00e-05 | 1.33e-04 | 4.67e+04 | 5.79e+10 | 2.55e+05 | 5.48e+10 | **1.51e+05** |
| **1.00e-05** | **4** | 1.00e-05 | 4.44e-05 | 1.40e+05 | 5.21e+11 | 7.66e+05 | 3.31e+11 | **3.20e+05** |
| ≤ 3.00e-06 | 3 | 2.79e-01 | 1.24e+00 | 1.37e+01 | 1.94e+03 | 5.21e+01 | 1.16e+03 | 1.00e+00 |

Every scaling in the table is exact: `cond(S0) ∝ 1/w`, `‖Kx²‖ ∝ 1/w²`,
`|q|max ∝ 1/w`, `cond(interface) ∝ 1/w²`. The **4** at `delta = 1e-5` is the
asymmetric snap of S2 note 3.

### S3.2 The spurious-wavenumber predictor is FREE

`|q|max · k0 J / (N(N+1)/4)`, measured at `w` = 3e-3 (`k0 J` = 1.331e-02):

| degree | 8 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|
| constant | **0.6786** | **0.6575** | **0.6537** | **0.6513** | **0.6485** |

So `|q|max ≈ 0.65 · N(N+1)/4 / (k0 J)` predicts the sliver's spurious spectrum
from geometry and degree alone, with no eig. It is what the refusal quotes,
and `test_the_spurious_wavenumber_predictor_matches_the_measured_spectrum`
re-derives it on the running build (it requires 0.55–0.80 with a
max/min spread under 1.15 — measured 1.048).

`N(N+1)/4` is the largest entry of the GLL differentiation matrix, so the
constant says the extreme eigenvalue of the sliver element's `1/J`-scaled
stiffness sits at about two thirds of the operator's largest entry — a
discretisation number, not a physical one.

### S3.3 The onset is DEGREE-DEPENDENT and no free per-layer instrument sees it

`p3_onset.py`, snap disabled, degrees 8/12/14/16/20 over `delta` 3e-3…3e-6.
The first `delta` at which the answer leaves the physical shift:

| degree | 8 | 12 | 14 | 16 | 20 |
|---|---|---|---|---|---|
| last CORRECT `w` | 3.0e-05 | 1.2e-04 | 1.2e-04 | 1.0e-04 | 1.2e-04 |
| first WRONG `w` | 1.0e-05 | 1.0e-04 | 1.0e-04 | 7.0e-05 | 1.0e-04 |
| `\|q\|max` at last correct | 9.11e+04 | 4.77e+04 | 6.39e+04 | 9.89e+04 | 1.27e+05 |
| `\|q\|max` at first wrong | 2.73e+05 | 5.73e+04 | 7.66e+04 | 1.41e+05 | 1.52e+05 |

The `|q|max` rows OVERLAP across degrees (degree 8 is still correct at 9.1e+04
where degree 12 is already wrong at 5.7e+04), so a bar on the spurious spectrum
refuses correct solves. The same is true of `cond(interface)` and of the width
itself.

**The shipped T3-4 instruments are blind here** (`p4_census.py`, census armed):

| instrument | on CORRECT rows | on WRONG rows |
|---|---|---|
| `n_grow_post` (channel A residual) | **0** | **0** — the 2026-08-06 `_forward_growth_flip` repair redirects every growing mode |
| `margin` (channel B) | 1.00 – 6.74 | 1.00 – 1.66 — the populations overlap |
| `q_excess` | crosses 1 while the answer is still right (1.95 at `delta` = 3e-4, degree 14, err = the physical 3.44e-04) and then **saturates** across the onset: 3.66 on the last correct degree-14 row and 3.66 on the first wrong one | — |

`q_excess` is the right *idea* — a mode called propagating that no propagating
mode of the layer can be — and it is exactly why it cannot be the bar: it fires
early and stops moving at the transition.

The Rayleigh-projection guard `_guarded_lstsq` (`_LSTSQ_RESID_BAR`) never fires
either: the defect is in the cascade, not in the far-field projection.

### S3.4 What DOES separate: the answer's own passivity

`p5_dense.py`, 46 log-spaced `delta` in 3e-3…3e-6 × degrees 12/14/20 = 138
rows per build, snap disabled. Classification uses the structure's own
continuity and no fitted constant: the measured shift is `err = 1.15 × delta`
over four decades, so `err ≤ 10 delta` is CORRECT, `err > 100 delta` is WRONG,
and the band between is neither.

| | Windows | WSL |
|---|---|---|
| rows | 138 | 138 |
| CORRECT / grey / WRONG | 78 / 1 / 59 | 78 / 1 / 59 |
| max `err/delta` among CORRECT | 1.196 | 1.196 |
| **max `\|R+T−1\|` among CORRECT** | **4.125e-06** | **4.125e-06** |
| **min `R+T−1` among WRONG** | **+1.159e+00** | **+1.159e+00** |
| the single grey row | degree 12, `delta` 5.54e-06, err 1.56e-04 (28× the shift), `R+T−1` = +2.28e-04 | identical |

Every wrong row is super-unity by **more than 1**, on both builds. Three
discrete failure values recur — `R+T−1` = 1.17, 2.61, 22.4 — which is the
signature of a mis-assembled forward set rather than a smooth loss of digits.

---

## S4. The remedy and the guard

### S4.1 The remedy already exists; the default does not reach it

`min_feature` snaps colliding CROSS-LAYER walls to their midpoint and warns.
`p9_remedy.py` scores the prescribed value against the exact `delta → 0`
reference, on both builds, at degrees 12/14/16/20:

| `delta` | prescribed `min_feature` | `err` vs the exact limit | bar `2 delta` | `err/delta` | `R+T` |
|---|---|---|---|---|---|
| 1.0e-04 | 2.4e-10 m | 1.152e-04 | 2.000e-04 | 1.152 | 1.000000 |
| 5.0e-05 | 1.2e-10 m | 5.765e-05 | 1.000e-04 | 1.153 | 1.000000 |
| 3.0e-05 | 7.2e-11 m | 3.460e-05 | 6.000e-05 | 1.153 | 1.000000 |
| 1.0e-05 | 2.4e-11 m | 1.153e-05 | 2.000e-05 | 1.153 | 1.000000 |
| 3.0e-06 | (default already snaps) | 3.461e-06 | 6.000e-06 | 1.154 | 1.000000 |

Identical at all four degrees and on both builds — 20/20 rows pass.

**BAR derivation (`err ≤ 2 delta`).** The structure is continuous in `delta`
and this build measures the slope at **1.152–1.154**; the snap moves each wall
by at most `delta`, so the snapped geometry cannot be further from the limit
than the original was. `2 delta` therefore carries **1.73× headroom** over a
measured constant, and it is re-derived at runtime — nothing is pinned.

The reason the default does not reach it is the one the 2026-07-28 audit
already recorded: `min_feature` defaults to `period × 1e-5`, which scales with
the PERIOD while a wall collision's scale is the per-slice offset. On this
1.2 µm period the default is 0.012 nm and the hazard band runs to ~0.24 nm.

### S4.2 What ships: a CONJUNCTION

```
(a) the union grid MANUFACTURED a cell — one whose two walls share no
    owning layer — at least _SLIVER_OWN_SCALE_RATIO times finer than the
    finest wall spacing any single layer asked for;
AND
(b) the solve reads super-unity above _STACK_SUPERUNITY_BAR on a PROVABLY
    PASSIVE stack with a LOSSLESS propagating incidence medium, where
    R + T <= 1 is a theorem and not a tolerance.
```

This is M1's `_guarded_lstsq` lesson (rank **and** residual) applied again, and
both halves are load-bearing:

* **(b) alone** would promote to a refusal every solve that today only warns —
  including the many-interface quasi-resonance `_warn_stack_energy`'s docstring
  says was deliberately left a warning "so it never breaks an existing working
  solve", and the 2-D stacks' ordinary low-`M` truncation (measured `R+T` =
  1.031 for `PMM2DStackHybrid` and 1.12–1.26 for `PMM2DStackPure` at the
  smallest usable mode counts, S7). Those callers pass no `stack` and keep the
  warning untouched.
* **(a) alone** fires on correct solves: it is true from `delta` = 3e-3 down,
  where the answer still tracks the physical shift to 1e-4.

**BAR (b) = `_STACK_SUPERUNITY_BAR` = 1e-2.** Not a new constant — it is the
one `_warn_stack_energy` has warned at since v5.14, now named once and used by
both the warning and the refusal so they cannot drift apart. Against the S3.4
populations, on BOTH builds:

| | value | distance to 1e-2 |
|---|---|---|
| max `\|R+T−1\|` among CORRECT | 4.125e-06 | **3.39 decades** below the bar |
| min `R+T−1` among WRONG | 1.159e+00 | **2.06 decades** above the bar |

`test_the_bar_has_decades_of_gap_on_both_sides_measured_here` re-measures both
populations on the running build and requires ≥ 2 decades below and ≥ 1.5
decades above, so the claim survives an algorithm change that keeps a real
separation and fails honestly on one that does not.

**BAR (a) = `_SLIVER_OWN_SCALE_RATIO` = 100.** Scale-free (a ratio of two
widths on one grid), so it carries no period, wavelength or degree.

| population | ratio | distance to 100 |
|---|---|---|
| the WIDEST cross-layer cell that ever produced a wrong answer (`w` = 8.786e-05 against own-scale 0.27865, degree 20) | 3.172e+03 | **1.50 decades** above |
| an ordinary NON-CONFORMING stack (two layers whose walls differ by a real 5% feature: walls 0.30/0.50 vs 0.35/0.55) | 2 – 10 | **1.0 – 1.7 decades** below |
| the M2 audit-class 2° coated taper (1.2 nm collisions against ~200 nm features) | ~1.7e+02 | 0.23 decades above |

It is an **ATTRIBUTION filter, not the pathology detector** — its job is to
confine the behaviour change and to let the message name the cause and the
cure. The detector is (b), which is the conjunct with the decades. The M2
taper's 0.23-decade margin is why (a) is set generously and why (b) carries the
decision.

**Ownership, exactly.** A cell is manufactured when *no single layer owns both
its bounding walls* (the period ends are owned by every layer). A thin feature
INSIDE one layer — a 1 nm liner — owns both of its walls and is never flagged,
which is the same rule `_pmm_union_grid`'s snap already uses ("a close pair
owned by a single layer is that layer's own intentional thin feature and is
never thinned"). The screen re-builds the union with the SAME `min_feature` the
solve used, so it reads the grid the cascade actually ran on.

### S4.3 What the code change is

| file | change |
|---|---|
| `_core.py` | `_pmm_union_grid` gains keyword-only `return_owners=False` (appends the per-wall owner frozensets, post-snap) and `warn=True` (so the guard's diagnostic re-build does not double-report the snap). Both default to the pre-fix behaviour; the 2-tuple return is byte-identical. `_pmm_union_grid` has no 2-D caller. |
| `stack.py` | `_cross_layer_sliver`, `_stack_provably_passive`, `_sliver_refusal`, the two bars and `PMM_SLIVER_GUARD`; `_warn_stack_energy` gains `stack=None` and raises on the conjunction before it warns; the nine `PMMStack` call sites pass `stack=self` (the prepared path passes `stack=self._st`). `stack2d.py`'s call is left at `stack=None`. |

Cost on a healthy solve: **nothing**. The geometric screen is only reached
after the super-unity test has already fired.

### S4.4 The guard, two-sided (`p8_guard.py`, 36 rows per build)

`PMM_SLIVER_GUARD = False` restores the pre-fix path bit for bit and is the
pre-fix arm.

| | Windows | WSL |
|---|---|---|
| rows | 36 | 36 |
| rows whose pre-fix `R+T` = 1 (CORRECT) | 22 | 22 |
| … of which **bit-identical** with the guard armed | **22 / 22** | **22 / 22** |
| rows whose pre-fix `R+T` ∈ {2.17, 3.59, 3.61, 23.4, 23.9} (WRONG) | 14 | 14 |
| … of which **REFUSED** | **14 / 14** | **14 / 14** |
| false positives | **0** | **0** |
| false negatives | **0** | **0** |

Every `R+T` agrees to six significant figures between the two builds.

---

## S5. Bit-identity of the shipped fixtures

`bitid_fixtures.py` builds 18 `PMMStack` solves drawn from the shipped test
files, covering the shared union grid, per-layer window grids at halfwidth 1
and 2, the tapered staircase with the snap DORMANT and ACTIVE, a Bragg ABAB
stack (the eig memo), conical, slant, out-of-plane tensors, a lossy layer,
`solve_vs_wavelength`, `prepare()`, `stabilize='slices'`, `retain_internal` +
`internal_field`, and `layer_absorption`. `p10_bitid.py` hashes the raw bytes
of every returned array (sha256 over dtype, shape and buffer) and asserts which
`lumenairy` it imported.

The reference is the **READ-ONLY main clone** at
`D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`
(HEAD `5adf571`, version 5.44.0). Its `lumenairy/elements/pmm/{stack,_core,
conical,oned,_jax_stack,twod}.py` are **byte-identical** to this branch's base
`3034bcb` (verified with `diff --strip-trailing-cr`), so it is a valid
independent pre-fix reference.

| fixture | hash (first 32 hex) | fix tree == main clone |
|---|---|---|
| `f01_one_layer_normal` | `9f717fd86bf47882c80cb0ada1c83b5f` | ✅ |
| `f02_one_layer_oblique` | `91e0bb8f26e4896da3007360dea27df3` | ✅ |
| `f03_two_layer_tensor_oblique` | `f6269d727bd5b99e2a543f603904fbbf` | ✅ |
| `f04_bragg_abab` | `72383692ec5fe4a2d0391eb7e821ed1b` | ✅ |
| `f05_taper_5_slices` | `7f8e20f82a9900e3d02e318b14362229` | ✅ |
| `f06_taper_8_slices_oblique` | `89b722d339aa67573321325e8a3be837` | ✅ |
| `f07_taper_snap_active` | `45b93a1d98ea8d03cd39997d9012c928` | ✅ |
| `f08_per_layer_halfwidth1` | `23c5026fab8039ce8c058262e1ce3377` | ✅ |
| `f09_per_layer_halfwidth2` | `a703a4a2004ec4846820b6fc78ae3abe` | ✅ |
| `f10_conical` | `bc33996940ac4734211b54157262a017` | ✅ |
| `f11_slanted` | `4688112bb782d8423fce47834c6e24a0` | ✅ |
| `f12_out_of_plane_tensor` | `210c46f88aa33e04c5d2afdc20ca46dc` | ✅ |
| `f13_lossy_layer` | `ccccc08e9c3eb2ab009daca2ce7bf45e` | ✅ |
| `f14_solve_vs_wavelength` | `b14f96672464f9c94adac2f1b60bc612` | ✅ |
| `f15_prepare` | `47072c1e71ae2bc74b70000c20af680f` | ✅ |
| `f16_stabilize_slices` | `eaf7cc240e0d1fa6a62182d7b57fc1ac` | ✅ |
| `f17_internal_field` | `620dc9aeecdf1ce58f73ca33599d6c85` | ✅ |
| `f18_layer_absorption` | `c863f00f60896e6a22568131e1c5eca1` | ✅ |

**18 / 18 bit-identical.**

---

## S6. `layer_grids='per-layer'` is not an escape

The refusal names it third and states the caveat, which is measured:
at `window_halfwidth = 1` a stack of 2 layers has every window equal to the
whole stack, so the per-layer path rebuilds the SAME union grid. On the O-11
fixture the two spellings agree to **< 1e-12** at `delta` = 1e-4 and 3e-5
(`test_per_layer_grids_is_not_a_second_opinion_on_a_two_layer_stack`), and the
S2 table shows the two self-gap columns identical to 16 digits at every
`delta`. It only helps above `2 · window_halfwidth + 1` layers, and even then
`min_feature` is live on it (M2 / N-6).

---

## S7. The 2-D stacks — measured, and clean

`p7_2d.py`.

| stack | grid | verdict |
|---|---|---|
| `PMM2DStackHybrid` | **no union grid at all** — layers are coupled through the Fourier projection, so walls may differ per layer | a cross-layer cell cannot exist. Structurally immune. |
| `PMM2DStackPure` | HAS a union-grid constraint, but the union is the caller-supplied **common `(Nx, Ny)` PIXEL lattice**: every cell is `period/Nx` wide | a sliver is an ASPECT-RATIO pathology and this grid's aspect ratio is exactly 1 |

Measured cell geometry of the pure stack's grid at four sizes:

| `N` | cells | every cell | **aspect ratio** | finest wall offset the API can express |
|---|---|---|---|---|
| 12 | 12 | 0.08333 | **1.0** | 0.08333 |
| 24 | 24 | 0.04167 | **1.0** | 0.04167 |
| 64 | 64 | 0.01562 | **1.0** | 0.01562 |
| 256 | 256 | 0.003906 | **1.0** | 0.003906 |

To place two walls `delta` apart the caller must pass `Nx ≥ 1/delta` — at which
point **every** cell is `delta` wide, i.e. uniform refinement, which carries no
aspect ratio and no sliver. There is nothing to report and no reproducer to
file.

Both 2-D stacks do read super-unity from ordinary modal under-convergence at
the smallest usable mode counts (hybrid `R+T` = 1.0313 / 1.0312 at degree 7/9,
self-gap 1.10e-02; pure `R+T` = 1.123 / 1.257 at `M` = 4/5, self-gap 1.56e-01).
That is the concrete reason the guard is a conjunction and the reason
`stack2d.py`'s `_warn_stack_energy` call keeps `stack=None`: on those paths
super-unity is a convergence signal, not a theorem violation.

**Caveat for the in-flight mortar work.** The pure stack's immunity comes
entirely from the uniform-lattice constraint. Roadmap item **N-1 / F5**
(non-uniform per-layer segment walls) removes exactly that constraint and gives
each layer an arbitrary wall list. When those walls are unioned onto a shared
grid, this defect's mechanism applies verbatim — `1/w²` in the nodal operator
and in the interface conditioning. The F5 probe already measured the
non-uniform mortar arm as smooth and monotone in `delta` all the way to zero
(9.32e-03 → 2.05e-04, no discontinuity), because the mortar keeps each layer on
its OWN grid and never forms the union — so the mortar route is the safe one,
and it is the union-forming route (`_pmm_union_grid` extended to wall lists, or
`PMM2DStackPure`'s `(2, 2)` grid fallback O-7 rewritten as a per-layer WALL
list) that must carry a screen of this shape.

---

## S8. Tests

`tests/unit/test_fix_pmmstack_sliver_walls.py` — **18 tests**, OMP/OPENBLAS/MKL
capped at file top before numpy is imported.

| | Windows | WSL |
|---|---|---|
| result | **18 passed** | **18 passed** |
| wall | 6.63 s | 7.25 s |

| test | what it pins |
|---|---|
| `test_fail_before_the_pre_fix_path_returns_a_wrong_energy_violating_answer` | FAIL-BEFORE, executed on the pre-fix code path via `PMM_SLIVER_GUARD = False`; the magnitude bar is derived from the build's own measured continuity slope |
| `test_fail_before_the_pre_fix_path_only_warns_it_does_not_refuse` | the pre-fix library RETURNED the wrong answer with a `UserWarning` — why it propagated into a warning-suppressing probe |
| `test_the_guard_refuses_every_wrong_row_and_no_right_one` | TWO-SIDED over a 13-row ladder: refused ⇔ wrong by continuity, bit-identical ⇔ right, ≥ 4 rows in each population |
| `test_a_delta_far_outside_the_hazard_is_bit_for_bit_untouched` | the explicit outside arm — orders, both efficiencies and Jones |
| `test_the_bar_has_decades_of_gap_on_both_sides_measured_here` | rule 5, re-derived at runtime: ≥ 2 decades below, ≥ 1.5 above |
| `test_the_refusal_names_the_geometry_and_the_two_remedies` | the message carries the sliver, the prescribed `min_feature`, the per-layer caveat, the switch and the doc |
| `test_the_prescribed_min_feature_lands_within_the_derived_bar` | the remedy, scored against the exact `delta → 0` reference at `err ≤ 2 delta` and closure < 1e-6 |
| `test_an_ordinary_non_conforming_stack_is_not_a_sliver` | conjunct (a) does not fire on real non-conforming geometry |
| `test_a_thin_feature_inside_ONE_layer_is_never_flagged` | the ownership rule: a 1e-4 liner owned by one layer is intentional |
| `test_a_cross_layer_sliver_is_flagged_and_reports_its_own_geometry` | the positive arm, including the reported walls and own-scale |
| `test_the_snap_removes_the_cell_and_the_screen_then_reads_clean` | the screen reads the POST-snap grid |
| `test_a_gain_layer_is_not_provably_passive_so_the_guard_stays_silent` | negative control: super-unity is legal with gain |
| `test_an_absorbing_superstrate_is_exempt_from_the_theorem` | the `_lossy_incidence` exemption, two-sided |
| `test_a_lossy_but_passive_stack_still_satisfies_the_theorem` | loss does not disable the theorem |
| `test_return_owners_is_additive_and_warn_false_is_silent` | the helper's contract: the 2-tuple is unchanged, owners are additive, `warn=False` is silent and `warn=True` still reports |
| `test_the_spurious_wavenumber_predictor_matches_the_measured_spectrum` | the MECHANISM claim, re-derived across five degrees |
| `test_the_wavenumber_the_message_quotes_is_the_one_the_solve_actually_has` | the refusal's own `\|q\| ~ ...` against the layer's measured spectrum (7.69e+04 quoted vs 7.66e+04 measured, 0.4%) -- right-conclusion-wrong-numbers is the shape that hides in a message |
| `test_per_layer_grids_is_not_a_second_opinion_on_a_two_layer_stack` | the caveat the refusal states |

`.test_durations` spliced with the measured Windows timings.

**Regression.** Every test file that imports `PMMStack`
(`grep tests/ --include='*.py' -l PMMStack`, 37 files) plus this one: green.
`ruff check lumenairy/ tests/`: clean.

**One shipped test file needed a switch, and it is the right one.**
`tests/unit/test_m1_conditioning_guard.py` drives the audit staircase — six
slices whose walls shift 4 nm on a 1 µm period, so the union grid carries 2 nm
cross-layer cells at ratio **157** — deliberately past its capacity, and
several of its arms disarm `INTERFACE_CONDITIONING_GUARD` or
`PMM_CONICAL_PERLAYER_ORDER_CAP` to HARVEST a pre-fix draw whose `R+T` then
reads **2.13** and **15.7**. Those harvests are exactly what the sliver guard
refuses, so two of its 27 tests raised instead of returning
(`test_rcond_of_hsup_would_have_been_the_wrong_instrument` and
`test_t3_3_fail_before_reproduces_the_over_capacity_draw`). Which arms trip it
is a BLAS fact — that file's own docstring records closure moving from 6.65e-06
to 2.14e+01 between one and two OpenBLAS threads on the SAME cell — so the fix
is a MODULE-scope autouse fixture that throws `PMM_SLIVER_GUARD` off for the
whole file, not a per-arm patch that a second build would defeat. Nothing in
that file asserts the sliver behaviour; this fix's own file owns it. With the
fixture: 27 + 18 = **45 passed**, 10.98 s.

That the guard fires on the M1 staircase unprompted is the strongest available
evidence that its conjunction is real: a colliding-wall staircase driven past
capacity is the same defect family, and the guard found it without being aimed
at it.

---

## S9. Open items

| | |
|---|---|
| **A — the default `min_feature` snaps ASYMMETRICALLY at its own threshold** | At `delta` = 1e-5 of the period the two wall pairs are the same width to the last bit, but the strict `d < mf` comparison merges one and leaves the other, so the solved geometry is a stack nobody asked for. The guard refuses the resulting answer, but the snap itself should be all-or-nothing across a symmetric pair. Small, contained, not fixed here. |
| **B — the quiet band the guard does not reach** | One row in 138 per build (degree 12, `delta` = 5.54e-06) is 28× off the physical shift at `R+T−1` = 2.28e-04, i.e. wrong at the 1.6e-04 level and BELOW the super-unity bar. The guard's floor is the theorem it uses; a detector for that band needs something the campaign has not found (the M2 lead — a two-degree consensus on the propagating-mode count — costs a second eig and is not free). |
| **C — the `stack.py` comment that promised a report that does not exist** | `stack.py`'s `min_feature` block says "A cross-layer sliver left in the grid is now reported by `_pmm_union_grid`". It is not: `_pmm_union_grid` warns only when it SNAPS. The refusal now covers the wrong answers; a plain report on a sliver LEFT in the grid is still unwritten (it would be a warning on correct solves, which is the trade S4.2 declines). |
| **D — the union-forming route of the mortar work** | S7's caveat: N-1 non-uniform walls + a shared union grid reproduce this mechanism verbatim. Owner: the mortar build. |
| **E — `_warn_stack_energy` on the conical / covariant / sweep paths** | The guard is wired into all nine `PMMStack` sites, but the ONSET was mapped only on the classical vertical cascade. The conical and covariant paths are covered by the conjunction's logic, not by a measured onset of their own. |
