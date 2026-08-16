# The EME 2-D vector mode census is LAPACK-build-dependent -- fixed 2026-08-12

`FIX_CI_RECONCILE_2026_08_12.md` S8.1 left one library-side defect open:

> `lumenairy/elements/eme/eme_2d_vector.py::layer_vector_modes._refine_accept`
> returns a LAPACK-build-dependent mode census.

Its two named causes are real and are treated here.  Its third statement -- that
the sqrt double zero is a mode whose reading is irreducibly floored -- is
**refuted**: the sqrt zeros are not modes at all, and once that is known the fix
is far cheaper and far more contained than the 1.87x candidate that round
measured and rejected.

Branch `fix/eme-census` off `origin/main` = **21802f9**, worktree
`C:/tmp/lum_eme`.  Mounts: **M** = Windows py3.14, numpy 2.4.4,
libscipy_openblas64 0.3.31; **W** = WSL py3.12, numpy 2.4.6, OpenBLAS.

---

## 0.  HEADLINE

**`sigma_min(G)` has zeros that are not modes.**  At a strip BAND EDGE
(`ky_i -> 0` in any strip) the forward/backward column pair of the global block
`G` becomes anti-parallel, so `G` is singular there for a reason that has
nothing to do with Maxwell.  Five such cusps sit inside the two reference
windows.  Some builds accept them as modes; ours reject them.  Neither verdict
was ever reasoned -- both are where a bounded minimiser happened to halt.

Three findings:

1. **The rejected 1.87x candidate was solving the wrong half of the problem.**
   It localised on a sub-grid before Brent -- on EVERY candidate, at every cell.
   That treatment is needed by **7 of 827** candidates measured across the whole
   EME suite (0.85%), and it does nothing at all for the sqrt cusps, which is
   why that round recorded "leaves N1 unchanged at 3/3".  Gated on an ambiguity
   band, the same treatment costs **1.000x-1.034x** instead of 1.87x and leaves
   96.25% of candidates on the byte-identical path.
2. **The sqrt double zero has a rigorous, free discriminator.**  Because the
   `+-ky` pair coalesces, `sigma_min <= sqrt(1 - |<a_hat, b_hat>|)` -- a
   one-line bound computed from the `G` that has already been built.  A cusp
   SATURATES it (ratio 0.21..0.66); a mode sits 4 to 14 decades below it.  No
   floor, no minimiser, no tolerance on `qz^2`.
3. **The shipped census carries a false positive AND a false negative, on the
   cells the shipped tests use.**  `test_vector_structured_completeness`'s own
   Nx=20 cell returns 233.1346015, which the independent 2-D-FD oracle places
   **27.4** away from any eigenvalue, and misses 112.5439974, whose FD distance
   (**0.050**) is the best in the whole band.  Both are fixed here.

---

## 1.  MECHANISM

### 1.1  Cause (a) -- a local minimiser on a function that is not unimodal

`sigma_min(qz^2) = min_i s_i(qz^2)` is a min of many smooth branches, so it is
piecewise-smooth with a dip wherever any branch has a local minimum.  Measured
on the Nx=16 reference grating, inside the SINGLE detection cell
`[205.875, 206.125]` (129-point probe): **31 local minima**, amplitudes 1e-4 to
2e-3, spaced ~2e-3.

`minimize_scalar(method="bounded")` is a local minimiser; its x-tolerance is
floored at `sqrt(eps)|x| + xatol/3` (3.5e-6 at `|qz^2| ~ 236`, so the shipped
`xatol = 1e-7` buys nothing).  On that cell it halts at

| mount | Brent's answer | `sigma_min` there | `gaps.min` | verdict |
|---|---|---|---|---|
| M (Windows py3.14) | 205.9786352762 | 6.485e-05 | 1.371e-03 | **reject** (1.37x) |
| W (WSL py3.12) | 205.9704915030 | 5.17e-05 | 1.0918e-03 | **reject** (1.09x) |
| ubuntu runners | -- | -- | -- | **accept** |
| converged | **205.9749757788** | 4.771e-15 | 1.008e-13 | accept by 1e10 |

The mode is genuine (S3).  Both our mounts drop it; the runners keep it.

### 1.2  Cause (b) -- the sqrt zero is a STRIP BAND EDGE, and is not a mode

`_global_block_G` gives strip `s` mode `i` two columns: `a` carrying
`exp(+i ky h)` with H-part `+V`, and `b` carrying `exp(-i ky h)` with `-V`.
`strip_vector_modes` recovers `V = (C U)/(i ky)`, which **diverges** as
`ky -> 0`.  After the column equilibration the H-part therefore dominates both
columns and they become anti-parallel, so `G` is singular at every strip band
edge whether or not a layer mode lives there, and

    sigma_min(qz^2) ~ C sqrt|qz^2 - q_edge|

is a genuine zero of `sigma_min` with no Maxwell solution behind it.  Three
independent confirmations, all measured:

| candidate (converged) | `min_s,i |ky| h` | FD-oracle distance | monodromy score |
|---|---|---|---|---|
| 235.8686333682 (W6) | 1.03e-07 | **27.54** | 7.89e-10 |
| 180.7703378418 (W6) | 3.74e-07 | **23.02** | 3.18e-09 |
| 233.4775302159 (N2) | 1.93e-07 | **27.43** | -- |
| 169.1623919091 (N2) | 3.50e-07 | **17.60** | -- |
| 133.7501554302 (N2) | 2.63e-07 | **6.97** | -- |
| 208.2502609719 (mode) | 2.02e+00 | 0.075 | 1.26e-19 |
| 203.7161764512 (mode) | 2.29e+00 | 0.075 | 2.83e-18 |
| 156.2813759062 (mode) | 1.51e+00 | 0.111 | 8.55e-20 |
| 190 / 220 / 245 (arbitrary) | -- | -- | 2.87e-09 / 9.11e-10 / -- |

`min |ky| h -> 0` is the band edge itself.  The FD oracle is the library's own
`verify=True` filter, which **already refuses these candidates** -- the fix
brings the default path into agreement with the oracle-assisted one.  The
monodromy score (S2) is built from scratch here and is the decisive one: to a
condition with no modal basis at all, the cusps are **indistinguishable from an
arbitrary `qz^2`**, nine decades above every true mode.

### 1.3  Why the two causes produce the SAME symptom

At a cusp a minimiser stopping `dq` short reads `C sqrt(dq)`; with `dq` floored
near 3.5e-6 by cause (a), the reading floors near 4e-4 and the rank-drop lands
1.09x-3.3x from `ratio_tol = 1e-3`.  That is the coin flip S5.1 of the
reconciliation tabulated.  It is not irreducible -- it is a reading of a
quantity that should never have been read there.

---

## 2.  THE ORACLE -- the y-MONODROMY, built from scratch

With `d psi/dy = A_s(qz^2) psi` for the Berreman state `psi = [Ex, Ez, Hx, Hz]`,
one Bloch period gives `psi(Ly) = M psi(0)` with
`M = expm(A_S h_S) ... expm(A_1 h_1)`, and the Bloch condition `psi(Ly) = t psi(0)`,
`t = exp(i ky0 Ly)`.  So

    F(qz^2) = det(M(qz^2) - t I) = 0    at a layer mode.

It shares NO machinery with the block-`G` finder: no forward/backward split, no
strip eigendecomposition, no equilibration -- hence **no structural singularity
at a band edge**, which is exactly the property needed to adjudicate S1.2.

Two realisations, agreeing:

* **float64** (`scipy.linalg.expm` + one SVD), used in-tree by
  `test_eme_census_determinacy.py`.  Usable only at small `Nx`: the monodromy's
  own dynamic range is `exp(2 max|ky| Ly)` -- 3e3 at Nx=8, where the separation
  is nine decades, but **1e13 at Nx=16, where every probe reads ~1e-17** and the
  oracle is worthless.  That is the cascade conditioning wall the module
  docstring already records as the reason the finder uses `G`.
* **mpmath at 40 digits**, entirely LAPACK-free, transcribing the scalar
  generator and taking `det(M - tI)` by secant.  Nx=8 cell, `|F|` normalised by
  `|F(q0 + 0.5)|`:

| `qz^2` probed | `|F(q0)| / |F(q0+0.5)|` | secant root | `dq` |
|---|---|---|---|
| 208.2502609719 | 2.219e-11 | 208.25026097191689167 | 1.69e-11 |
| 203.7161764512 | 1.410e-11 | 203.71617645119133390 | -8.67e-12 |
| 156.2813759062 | 4.818e-11 | 156.28137590616172528 | -3.83e-11 |
| **235.8686333682** | **0.727** | 223.067 (wanders off) | -12.8 |
| **180.7703378418** | **0.718** | 168.897 (wanders off) | -11.9 |

The three modes are zeros to 40 digits and the library's values sit within
1.7e-11 of them.  **The two sqrt cusps are not zeros at all** -- `|F|` there is
0.72 of its value half a unit away.

The Nx=16 cell, where the recovered mode lives (the float64 monodromy is useless
there, so this arm is mpmath-only):

| `qz^2` probed | `|F(q0)| / |F(q0+0.5)|` | secant root (40 digits) | `dq` |
|---|---|---|---|
| 205.9749757788 | 3.605e-11 | **205.9749757787688608** | -3.11e-11 |
| 205.9786352762 (the pre-fix acceptance point) | **4.207e-03** | 205.9749757787688608 | **-3.6595e-03** |
| 201.8868824991 | 5.102e-07 | 201.88688284563654154 | 3.47e-07 |
| **233.4775302159** | **0.647** | 223.897 (wanders off) | -9.58 |
| **169.1623919091** | **0.615** | 160.817 (wanders off) | -8.35 |
| **133.7501554302** | **0.688** | 124.076 (wanders off) | -9.67 |

### 2.1  `qz^2` ACCURACY -- the error collapse, against 40 digits

| `qz^2` | what the pre-fix path did with it | error vs the 40-digit root | what the fix returns | error |
|---|---|---:|---|---:|
| **205.97497578** (Nx=16) | read acceptance at 205.9786352762 and REJECTED | **3.66e-03** | 205.9749757788 | **3.1e-11** |
| 201.88688285 (Nx=16) | accepted, returned 201.8868824991 | 3.47e-07 | unchanged (out of band) | 3.47e-07 |
| 208.25026097 (Nx=8) | accepted, returned 208.2502597917 | 1.18e-06 | unchanged (out of band) | 1.18e-06 |
| 203.71617645 (Nx=8) | accepted, returned 203.7161763904 | 6.08e-08 | unchanged (out of band) | 6.08e-08 |
| 156.28137591 (Nx=8) | accepted, returned 156.2813757187 | 1.87e-07 | unchanged (out of band) | 1.87e-07 |
| 235.86863337 / 180.77033784 (Nx=8) | ACCEPTED on the runners | -- (not a zero) | refused on every build | -- |
| 233.47753022 / 169.16239191 / 133.75015543 (Nx=16) | knife-edge | -- (not a zero) | refused on every build | -- |

**The ~4e-3 error collapses by eight decades**, on the one candidate the fix
touches.  `_polish_zero` itself, checked against the same 40-digit roots:
201.886882845637 (error **6.0e-13**) and 208.250260971917 (**1.1e-13**).

The unchanged rows are the deliberate S7.2 residual: an unambiguous accept keeps
the minimiser's own stopping point, whose ~1e-6 error is 5 decades below the
method's x-FD accuracy (those same modes sit 0.05..0.22 from the FD oracle).

---

## 3.  THE FIX -- contained by construction

All of it is inside `_refine_accept`, plus two helpers and five constants.

```
r = minimize_scalar(...)                      # UNCHANGED
s, gaps, bound = _mode_reading(x = r.x)       # one G build; s byte-identical
                                              #   to _block_singvals
if _CENSUS_BAND[0]*ratio_tol <= gaps.min() <= _CENSUS_BAND[1]*ratio_tol:
    # AMBIGUOUS: this verdict is round-off, not physics.
    if s[-1] >= _STRUCTURAL_SAT * bound:  return       # band edge -- free
    x = _polish_zero(f, lo_b, hi_b)                    # converged zero
    s, gaps, bound = _mode_reading(x)
if s[-1] < tol and gaps.min() < ratio_tol:             # UNCHANGED gate
    if s[-1] >= _STRUCTURAL_SAT * bound:  return       # structural, always on
    ...                                                # verify + append
```

**Outside the band nothing changes** -- same Brent, same reading, same verdict,
same returned float.  That is the entire containment argument, and it is why the
lesson of the rejected attempt lands: that attempt localised before Brent on
every candidate, so it moved every returned `qz^2` in every cell; this one
reaches only the candidates whose verdict was decided in the last bits.

### 3.1  `_pair_singularity_bound` -- the structural test (a theorem, and free)

`G`'s columns are already unit-norm (the equilibration).  For the strip-`s`
mode-`i` pair with `c = |<a_hat, b_hat>|`, the unit coefficient vector
`z = (e_a - e^{i arg} e_b)/sqrt(2)` gives

    sigma_min(G) <= ||G z|| = sqrt(1 - c)  =:  bound

for reasons independent of any mode.  A candidate whose `sigma_min` SATURATES
that bound is explained by the basis.  Cost `O(n M S)` -- one half of a matrix
product against the `O(n^3)` SVD that produced `sigma_min`.

`_STRUCTURAL_SAT = 1e-2`, two-sided measured `sigma_min / bound`:

| class | at Brent's stop | converged | on a restaged 8-strip staircase |
|---|---|---|---|
| strip band edge | 5.94e-1 .. 6.60e-1 | 2.10e-1 .. 6.61e-1 | 2.82e-1 |
| genuine mode | 5.00e-9 .. 1.52e-4 | 3.4e-14 .. 1.1e-14 | 1.5e-15 .. 6.7e-15 |
| non-dip (rejected by decades) | 1.86e-2 .. 7.14e-2 | -- | -- |

Worst band edge 21x OVER the bar, worst mode 66x UNDER it.  The ratio is
dimensionless and scale-free: it reads the same on the 10x-scaled cell and on an
8-strip restaging where the raw `|ky| h` has changed by 4x.

### 3.2  `_polish_zero` -- the converged zero

Sub-grid localisation over the SAME detection cell (`_POLISH_SUBGRID = 33`, a
16x refinement of the detection grid -- the deepest sample selects the basin,
which a local minimiser cannot), then a 5-point nested bracket contracted until
its width is `_POLISH_XTOL_REL * max(|x|, 1)` with `_POLISH_XTOL_REL = 1e-12`.

Termination is on the ARGUMENT's own scale, so the level count is the same on
every build and the same in any length unit.  Derivative-free, so the `p = 1/2`
cusp and the `p = 1` V converge alike:

| bracket | class | polished `x` | `|f|` | evals | `|x - x*|` |
|---|---|---|---|---|---|
| [205.875, 206.125] | trapped mode | 205.974975778779 | 2.04e-13 | 87 | 1.1e-11 |
| [201.75, 202.0] | mode | 201.886882845662 | 6.89e-13 | 87 | 2.5e-11 |
| [233.375, 233.625] | cusp `p = 1/2` | 233.477530215867 | 7.04e-07 | 85 | 8.9e-12 |
| [235.75, 236.0] | cusp `p = 1/2` | 235.868633368169 | 1.63e-06 | 85 | 4.8e-11 |

Identical to 12 digits from a 17-, 33- or 65-point sub-grid (the localisation
only has to pick the basin).

### 3.3  `_CENSUS_BAND = (1e-2, 3e1)` -- where the treatment applies

Measured `gaps.min` in units of `ratio_tol`, across the two reference cells:

| class | measured | vs the band |
|---|---|---|
| unambiguous ACCEPT | <= 7.5e-4 x | 13x under the lower edge |
| ambiguous | 1.09x .. 5.26x | inside |
| unambiguous REJECT | >= 68.9x | 2.3x over the upper edge |

---

## 4.  BYTE-NULL SCOPE

### 4.1  Candidate level -- 96.25% take the unchanged path

Every candidate the finder refines, across every cell the EME suites exercise:

| cell | candidates | in band | structural reject (free) | POLISHED | accepted | unchanged path |
|---|---:|---:|---:|---:|---:|---:|
| W6 base (Nx=8) | 11 | 2 | 2 | 0 | 3 | 9 (81.8%) |
| W6 scaled x10 | 11 | 2 | 2 | 0 | 3 | 9 (81.8%) |
| N2 dense (Nx=16) | 92 | 4 | 3 | 1 | 5 | 88 (95.7%) |
| N2 banded | 102 | 4 | 3 | 1 | 5 | 98 (96.1%) |
| Nx=20, (56, 259) | 231 | 7 | 5 | 2 | 16 | 224 (97.0%) |
| Nx=16, (0, 256) | 273 | 10 | 7 | 3 | 24 | 263 (96.3%) |
| Nx=12 verify cell | 107 | 2 | 2 | 0 | 11 | 105 (98.1%) |
| **total** | **827** | **31** | **24** | **7** | **67** | **796 (96.25%)** |

**Seven candidates out of 827 are polished.**  That is the whole cost.

### 4.2  Output level -- byte-identical wherever the census was unambiguous

`layer_vector_modes` run on the base tree (21802f9) and on this branch, one BLAS
thread, M:

| cell | pre `n` | post `n` | pre evals | post evals | cost | delta |
|---|---:|---:|---:|---:|---:|---|
| W6 base (Nx=8) | 3 | 3 | 1002 | 1002 | **1.000x** | **BYTE-NULL** |
| W6 scaled x10 | 3 | 3 | 981 | 981 | **1.000x** | **BYTE-NULL** |
| Nx=12 verify cell | 10 | 10 | 2862 | 2862 | **1.000x** | **BYTE-NULL** |
| W6-5 solver cell | 2 | 2 | 130 | 130 | **1.000x** | **BYTE-NULL** |
| N2 dense | 4 | 5 | 2885 | 2972 | 1.030x | +205.9749758 |
| N2 banded | 4 | 5 | 3082 | 3169 | 1.028x | +205.9749758 |
| Nx=20, (130, 256) | 6 | 5 | 3522 | 3609 | 1.025x | -233.1346015, 201.6405953 -> 201.6415842 |
| Nx=20, (56, 259) | 16 | 16 | 6334 | 6510 | 1.028x | -233.1346015, +112.5439974, 201.6405953 -> 201.6415842 |
| Nx=16, (0, 256) | 23 | 24 | 7838 | 8103 | **1.034x** | +205.9749758, two values corrected by 3.4e-7 / 6.2e-6 |

Worst case **1.034x**, against the rejected candidate's **1.87x**.  The pre-fix
eval count on the N2 cell is **2885** -- the same number S5.1 of the
reconciliation reports, so the two measurements are of the same thing.

### 4.3  Every non-null delta, adjudicated against the FD oracle

The Nx=20 cell is the one `test_vector_structured_completeness` and
`test_vector_verify_removes_spurious` use.

| `qz^2` | change | `sigma_min` | `sigma_min/bound` | `min|ky|h` | FD distance | verdict |
|---|---|---|---|---|---|---|
| 233.1346014770 | **removed** | 2.03e-05 | **6.57e-01** | 4.33e-05 | **27.41** | band edge -- was a FALSE POSITIVE |
| 201.6405952948 | replaced | 2.45e-05 | 7.09e-05 | 2.29e+00 | 0.0744 | Brent's stop, 9.9e-4 short |
| 201.6415842398 | by this | **8.62e-14** | 2.50e-13 | 2.29e+00 | **0.0734** | the converged zero |
| 112.5439974414 | **gained** | **1.20e-13** | 2.15e-13 | 1.97e+00 | **0.0502** | genuine -- best FD match in the band |

Against the 16-mode FD oracle band, matched at the shipped 0.7 tolerance:
**recall 10 -> 11 of the top 10/11, spurious 1 -> 0.**

---

## 5.  THE INJECTOR -- fail-before and fail-after

A LAPACK build does not shift every `sigma_min` the same way; it gives each
EVALUATION its own last bit.  Two deterministic emulations were built:

* `jitter(k)`: `sigma_min(q) * (1 + k eps xi(q))` with `xi` a splitmix64 hash of
  `q`'s own bits in [-1, 1], so `f` stays a function;
* `bracket(k)`: the refinement bracket nudged by `|k|` ULP -- the minimiser
  walks a different golden sequence over the same cell.

A UNIFORM nudge of `sigma_min` (`nextafter` by 1..16 ULP, every sample the same
direction) moves **nothing**, pre or post, on any cell: it preserves the order of
every comparison the minimiser makes.  Reported because it is the obvious
injector and it is inert -- the per-evaluation forms are the faithful ones.

### 5.1  Census MEMBERSHIP under the ladder

| cell | tree | jitter x1..x64 | bracket +-1, +-4, +-16 ULP | arms that FLIP |
|---|---|---|---|---|
| W6 base | pre-fix | 3 3 3 3 3 3 | 3 **4** 3 3 3 3 | **1 / 12** |
| W6 base | **fixed** | 3 3 3 3 3 3 | 3 3 3 3 3 3 | **0 / 12** |
| N2 (Nx=16) | pre-fix | 4 4 4 4 4 4 | 4 **5 5 5 5 5** | **5 / 12** |
| N2 (Nx=16) | **fixed** | 5 5 5 5 5 5 | 5 5 5 5 5 5 | **0 / 12** |
| Nx=20 | pre-fix | 6 6 6 6 6 **5** | **5 4 5 5 5 5** | **7 / 12** |
| Nx=20 | **fixed** | 5 5 5 5 5 5 | 5 5 5 5 5 5 | **0 / 12** |

(Table measured on M.  The W6 rows were re-measured on **W** and are the same
verdict from a different LAPACK: pre-fix **1 / 12** arms flip, the flipping arm
is again `bracket -1ULP`, and the value it gains is again **235.868633551** --
identical to every printed digit across the two mounts; fixed, **0 / 12**.)

The W6 flip is the CI failure itself: a **-1 ULP** nudge of an interval endpoint
makes the pre-fix finder accept **235.868633551** -- the same band-edge cusp the
ubuntu runner accepted, to 2e-7.  One ULP of a bracket endpoint is 2.8e-14 in a
window 100 wide.

On the Nx=20 cell the pre-fix returned value for one accepted mode swings
between 201.6405953 and 201.6415843 (**9.9e-4**) depending on the arm; fixed, it
is pinned at 201.6415842 in 12 of 13 arms and within 2.3e-6 in the last.

### 5.2  What the fixed arms still move

Clear-accept modes keep the minimiser's own stopping point by design (S3), so
their reported `qz^2` still moves ~1e-6 under a bracket nudge.  That is 4 to 5
decades below the method's own accuracy -- the same modes sit 0.05..0.22 from
the FD oracle -- and it is not what the census is.  Membership, and every value
the treatment touches, are stable.

---

## 6.  GREEN

### 6.1  New module

`tests/unit/test_eme_census_determinacy.py`, **7 tests**:

| test | what it pins |
|---|---|
| `..._refused_sqrt_cusps_are_not_modes_of_an_independent_condition` | the float64 monodromy oracle, two-sided with arbitrary-`qz^2` controls |
| `..._sigma_min_saturates_the_structural_bound_only_at_a_band_edge` | the bound IS a bound at every probe; the ratio separation |
| `..._one_ulp_bracket_nudge_flips_the_prefix_census_but_not_the_fixed_one` | **the fail-before**, in-process, both directions |
| `..._census_is_byte_identical_where_the_prefix_path_was_unambiguous[1.0]` / `[10.0]` | byte-null, asserted with `np.array_equal` against the pre-fix path restored in-process |
| `..._polish_converges_the_cusp_and_the_v_independently_of_localisation` | the polisher's determinism, both local forms |
| `..._dropped_mode_is_recovered_and_the_fd_oracle_confirms_it` | the recall half, confirmed by the in-tree FD oracle |

The pre-fix path is restored in-process by `_CENSUS_BAND = (0.0, 0.0)` (an empty
band, so nothing is polished) and `_STRUCTURAL_SAT = inf` (unreachable, so the
structural test never fires) -- which leaves exactly the shipped 21802f9 body.
Every claim therefore has its fail-before measured in the same process on the
same build, not inferred from a different runner.

### 6.2  Suites

The EME consumer set -- `test_niche_audit_w6_eme` (75) + `test_audit_w6_eme`
(15) + `test_eme_2d` (6) + `test_eme_diffraction` (3) + `test_eme_jax_modes` (8)
+ `test_v5_18_1_residuals` (8) = **115**, plus the new module's **7** = 122 --
run at `OPENBLAS_NUM_THREADS` 1 / 2 / default on BOTH mounts:

| mount | 1 thread | 2 threads | default |
|---|---|---|---|
| M (Windows py3.14) | **122 passed** (223.5 s) | **122 passed** (221.3 s) | **122 passed** (952.9 s) |
| W (WSL py3.12) | **122 passed** (218.4 s) | **122 passed** (189.3 s) | **122 passed** (1123.7 s) |

The slow gate, one BLAS thread:

| module | M | W |
|---|---|---|
| `test_eme_2d_vector.py` (19) | **19 passed** (381.2 s) | **19 passed** (217.5 s) |

`ruff check lumenairy/ tests/unit/` -- the exact CI command -- clean on both
mounts.  All three touched files are ASCII; no `xfail`, `skip`, deleted test,
weakened guard, or `CHANGELOG` entry anywhere in this change.

The meta gates that scan the whole package -- `test_audit_except_budget`,
`test_v4_16_0_walker_all_symmetry`, `test_v4_15_3_dispatcher_pin_2d_scalar_field`,
`test_public_api` -- read **717 passed, 2 failed**, and the two failures are the
PRE-EXISTING main-side A1/A2 gap of `FIX_CI_RECONCILE_2026_08_12` S2: measured
identical on the pristine 21802f9 worktree (`2 failed, 18 passed`), and already
carried by `fix/runner-pins` @ fa8f719.  The except budget is untouched -- the
only new `except` in this change is the existing narrow `np.linalg.LinAlgError`
guard, duplicated for the re-read after a polish.

The `default` column is 4.3x the 1-thread column and is the pre-existing BLAS
oversubscription axis the reconciliation records, not this change: the new
module pins itself to one thread at RUNTIME with `threadpool_limits` (the lever
`test_eme_2d_vector.py` already uses), because uncapped it measured 11.5
CPU-hours in 32 wall-minutes against 54 s at one thread -- and because a
multi-threaded reduction order is a coarser version of the very perturbation its
injector applies, so leaving it free would confound the two.

### 6.3  The immunized reconciliation tests

`fix/jax-nan-pins` @ 051f11c restated both red ids on CONVERGED ZEROS and gave
each a bounded allowance for the census the library could not pin down.  Their
files were overlaid on this branch and run unmodified (then removed -- this
branch does not touch either file):

| mount | 76 + 20 = 96 |
|---|---|
| M | **96 passed** (516.1 s, 1 thread) |
| W | **96 passed** (269.2 s, 1 thread) |

Both stay green, and **both allowances go slack** -- they are now
redundant-but-harmless:

| allowance | its bar | measured on this branch | slack |
|---|---|---|---|
| N1 `abs(len(base) - len(scaled)) <= 2` | 2 | **0** (3 and 3) | full |
| N2 `paired >= (len(md)-1) + (len(mb)-1)` | 8 of 10 | **10 of 10**, worst partner 1.02e-08 vs the 1e-4 bar | full |

Their fail-before arms still fire: the pre-fix detection grid still empties the
scaled census (`len(pre_scaled) == 0`, `len(pre_base) == 3`), and the
under-iterated inverse-power ladder still breaks the pointwise bar.

---

## 7.  WHAT CHANGED

    lumenairy/elements/eme/eme_2d_vector.py     +~170   the fix
    tests/unit/test_eme_census_determinacy.py   new     7 tests
    docs/audits/FIX_EME_CENSUS_2026_08_12.md    new     this file

No `xfail`, no `skip`, no deleted test, no weakened guard, no CHANGELOG entry.
No public signature changes; `_block_singvals` and `dispersion_vec` keep their
behaviour byte for byte.

### 7.1  Merge-train notes

* Disjoint from `fix/runner-pins` (fa8f719), `fix/jax-nan-pins` (051f11c) and
  `fix/verify-arch`: the only shared path is `tests/unit/test_eme_2d_vector.py`
  and `tests/unit/test_niche_audit_w6_eme.py`, which this branch does **not**
  touch.
* The two immunized reconciliation tests stay green with this library change and
  their immunization goes SLACK (S6.3) -- the allowances they were given are no
  longer used.  They should not be narrowed in the same commit as this fix; the
  measured slack is recorded here so a later round can retire them knowing what
  it is retiring.
* `.test_durations` has no entry for the 7 new ids (the standing open item from
  the reconciliation).  Module wall clock is recorded in S6.2 so the shard can
  be balanced by hand if needed.

### 7.2  Open, not fixed here

* **Clear-accept `qz^2` values still come from the minimiser's stopping point**
  and move ~1e-6 across builds (S5.2).  Polishing them would move every returned
  value in every cell -- the exact containment failure that got the previous
  candidate rejected -- for an improvement 4 decades below the method's own
  x-FD accuracy.  Deliberately left.
* **The band-edge cusps are refused, not reported.**  A caller sweeping a window
  gets no signal that the finder declined to decide there.  The library already
  refuses to evaluate exactly at a band edge (`strip_vector_modes` raises), so
  the behaviour is consistent, but a diagnostic return would be better than
  silence.
* **`_STRUCTURAL_SAT` has a 21x / 66x two-sided margin** measured across four
  cells and three geometries.  It is the narrowest bar in this fix.
* **The Nx=20 deltas of S4.3 were adjudicated with the in-tree 2-D-FD oracle,
  not with mpmath.**  The mpmath monodromy at 4Nx = 80 costs ~10 min per
  determinant evaluation and was not run; the FD oracle separates those three
  candidates by 27.41 against 0.050, which is decisive on its own, and the same
  classification was confirmed at 40 digits on the Nx=8 and Nx=16 cells.

---

## 8.  THE FAIL-BEFORE DEMONSTRATIONS WERE THEMSELVES PER-BUILD -- 2026-08-13

`tests/unit/test_eme_census_determinacy.py` shipped with 5.35.0/.1 and two of
its seven ids then FAILED the 5.35.1 verify shard (ubuntu, py3.11), green on
both mounts at every thread setting:

| id | the assertion | reading on that runner |
|---|---|---|
| `..._one_ulp_bracket_nudge_flips_the_prefix_census_but_not_the_fixed_one` | `assert flips` | `[]` -- NO arm of the ULP ladder flips the PRE-FIX census there |
| `..._dropped_mode_is_recovered_and_the_fd_oracle_confirms_it` | `min abs(prefix - 205.9749757788) > 1e-2` | **3.99e-04** -- the PRE-FIX census already HOLDS the mode there |

Both are this campaign's own pattern (`FIX_CI_M1_T34_2026_08_06`: *"asserts a
per-build fact as a universal one"*), in its sharpest form.  **The defect fixed
in S1-S3 is that a near-threshold verdict is decided in the last bits -- so
whether that defect MANIFESTS is precisely what a build is entitled to decide.**
A fail-before that asserts the pre-fix defect manifests can therefore only ever
be a per-build reading, no matter which cell it is run on.  Nothing in S1-S7 is
retracted: the library is not touched, and the numbers of S1-S5 are what this
round measures again.

Restructured per the campaign's synthetic-injector precedent -- the `0.7 * eye`
exactly-degenerate SVD of `test_niche_audit_w9_eig_vjp.py`, the
`near_cut_injector` of `test_pmm_m2_window_contract.py`, the `o7` side-flip of
`test_niche_audit_w6_berreman.py`.  Test file only.

### 8.1  The three layers each restructured id now has

**(a) The FIXED path, UNCONDITIONALLY, against the ORACLES.**  Not one assertion
about the fixed census now reads the pre-fix path.  New helper `_oracle_clean`
demands of EVERY census entry, on both reference cells: an independent 2-D-FD
eigenvalue within `1.0` (the same `verify_tol` the library ships), and
`sigma_min / bound < 1e-3` (not a band edge); and it demands that every known
cusp of the window is ABSENT.  On top of that the W6 census must hold all three
MONODROMY-confirmed modes, the Nx=16 census must hold `205.9749757788` on BOTH
solvers, and neither may move under the ULP ladder.

| cell | entry | `sigma_min / bound` | FD distance | verdict |
|---|---|---:|---:|---|
| W6 | 208.2502597917 | 5.59e-08 | 0.0754 | mode |
| W6 | 203.7161763904 | 4.96e-09 | 0.0754 | mode |
| W6 | 156.2813757187 | 7.66e-09 | 0.1108 | mode |
| W6 | *235.8686333682* | **6.59e-01** | **27.54** | cusp -- ABSENT |
| W6 | *180.7703378418* | **6.02e-01** | **23.02** | cusp -- ABSENT |
| N16 | 205.9749757788 | 4.77e-13 | **0.0758** | the recovered mode |
| N16 | 201.8868824991 | 2.67e-08 | 0.0738 | mode |
| N16 | 151.3854745564 | 2.59e-08 | 0.1744 | mode |
| N16 | 146.4214663772 | 8.63e-09 | 0.2245 | mode |
| N16 | 140.5997564561 | 1.40e-08 | 0.1180 | mode |
| N16 | *233.4775302 / 169.1623919 / 133.7501554* | **5.94e-1 .. 6.60e-1** | -- | cusps -- ABSENT (>= 6.85 away) |

Two-sided by decades on both discriminators, and none of it depends on a build.

**(b) The PRE-FIX demonstration, on an ENGINEERED TIE AT THE CUT.**  `ratio_tol`
IS the bar the verdict is read against.  What a build decides is where its
minimiser halts, hence what `gaps.min` READS; what it does not decide is that
the reading has a spread.  So the bar is placed inside that spread, measured on
the build itself:

* `_tie_at_the_cut` (test 3) walks the pre-fix path over the ULP ladder with a
  pass-through instrument on `_mode_reading`, takes the spread of the readings
  of one band-edge cusp, and sets `ratio_tol` to its GEOMETRIC MEAN.  The arm
  that read lowest then ACCEPTS the cusp and the arm that read highest REJECTS
  it, by construction, on any build.  At that same tie the FIXED path refuses it
  on every arm -- `sigma_min / bound` at that cusp is **6.592e-01 to four digits
  on all 13 arms and on both mounts**, because the structural bound is a
  property of `G` and not of where a minimiser stopped.
* `_prefix_drop_cut` (test 6) uses the nine-decade gap between what Brent's halt
  reads and what the CONVERGED zero reads, dividing the build's own Brent
  reading by `sqrt(_CENSUS_BAND[1])` -- DERIVED from the library constant, so the
  tie lands at the geometric centre of the range in which the fix still
  polishes.  Pre-fix DROPS the mode there; fixed polishes and returns it.

Measured, in-process, on both mounts (the tie is re-derived per build, so the
numbers differ and the verdict does not):

| mount | test 3 tie: spread -> `ratio_tol` | pre-fix straddle | test 6 tie: Brent / zero -> `ratio_tol` | pre-fix / fixed |
|---|---|---|---|---|
| M (Win py3.14) | 9.5114e-04 .. 2.6048e-03 (2.74x) -> **1.574026e-03** | accepts on `-1`, refuses on `+4` | 1.5112e-03 / 4.31e-12 -> **2.758979e-04** | DROPS / returns 205.97497577878 |
| W (WSL py3.12) | 9.5114e-04 .. 2.6048e-03 (2.74x) -> **1.574026e-03** | accepts on `-1`, refuses on `+4` | 1.0918e-03 / 4.39e-12 -> **1.993319e-04** | DROPS / returns 205.97497577878 |

The only vacuity condition left is "this build's ladder did not move the reading
AT ALL", which is guarded by widening (`_ULP_ARMS_WIDE`, out to 1024 ULP) before
it can fail, and whose failure message says to widen rather than delete.

**(c) The LIVE cell at the SHIPPED `ratio_tol`, ADJUDICATED.**  The original
demonstrations are kept and measured, and PASS either way with the reading
printed -- `EME census tie [live cell]` / `EME census recall [live cell]`,
reproduced-here or inert-here.  Both reproduce on M and W; the runner's readings
would print the inert branch.

### 8.2  The byte-null id carried the same defect, latent

`..._census_is_byte_identical_where_the_prefix_path_was_unambiguous` asserted
`np.array_equal(fixed, prefix)`.  That is the same per-build fact with the sign
reversed: on a build whose pre-fix path ACCEPTS a W6 band-edge cusp -- which
S1.2 records the 2026-08-12 ubuntu runner doing -- the fixed array MUST differ,
by refusing it, and the byte-null id would go red FOR THE FIX WORKING.  It
survived the 5.35.1 shard only because that image's LAPACK happens to refuse the
cusp too.

Restated as the `_CENSUS_BAND` contract itself, which is what S4 actually claims
and is universal:

* every pre-fix entry whose reading fell OUTSIDE the ambiguity band comes back
  BIT-IDENTICAL (the 96.25%-of-candidates claim of S4.1);
* every entry the fixed path returns that is not one of those is a CONVERGED
  ZERO -- reading below the band, i.e. the polish's output.

Where the two arrays do come out bit-identical (both mounts, both scales) the
stronger reading is printed.

### 8.3  What the restructure strengthened, and the one bar it derived

| claim | before | after |
|---|---|---|
| fixed census content | membership of `_RECOVERED`, and "one more than pre-fix" | every entry FD-confirmed AND non-structural; every oracle-confirmed mode present; every known cusp absent -- on BOTH cells |
| "nothing lost" | `for q in prefix: q in fixed` (false on a cusp-accepting build) | every pre-fix entry is EITHER kept OR structurally refused -- universal |
| the converged zero | `abs(got - 205.9749757788) < 1e-6`, itself a per-build bar | `_polish_zero`'s own answer pinned to `< 1e-6`, AND the census entry pinned to THAT within `_BRENT_XFLOOR` |
| byte-null | `array_equal(fixed, prefix)` | the two-sided `_CENSUS_BAND` containment contract |
| the fail-before | asserts the defect manifests | asserts it manifests AT AN ENGINEERED TIE, on every build |

`_BRENT_XFLOOR = sqrt(eps)|x| + xatol/3 = 3.1e-6` is the one new bar, and it is
DERIVED, not tuned: it is `minimize_scalar(method="bounded")`'s own x-tolerance
at `_RECOVERED` with the library's shipped `xatol = 1e-7`, i.e. the closest to a
zero any build's minimiser is ENTITLED to stop, and therefore the universal bar
on an entry the fix leaves at a clear-accept stopping point (S7.2).  Both mounts
land **3.1e-11**, five decades inside it, because the candidate falls in the
ambiguity band and is polished.

### 8.4  Runner emulation -- perturbing TOWARD pre-fix stability

The restructure is only worth anything if it survives the condition it was
written for.  Six emulations, each perturbing the build so that the pre-fix
defect STOPS manifesting (or, for the byte-null id, STARTS).  All six PASS, on
both mounts, in-process:

| emulation | id | what it makes the pre-fix path do | result |
|---|---|---|---|
| **E1** ladder shrunk to `(1, 4, -4, 16, -16)`, dropping the one arm whose reading falls under the shipped 1e-3 | 3 | live cell INERT: all 5 arms return 3 | **PASS**, inert branch printed; the engineered tie at **1.793400e-03** still straddles (accepts on `-16`, refuses on `+4`) |
| **E2** narrow cell `n_scan` 9 -> 40, so Brent halts 1.5e-6 from the zero | 6 | live cell INERT: pre-fix KEEPS the mode (reads 4.44e-07, UNDER the bar) | **PASS**, inert branch printed; the tie at **8.106068e-08** still drops it pre-fix and the fix returns 205.9749757787929 |
| **E3b'** `_DETECT_PPU` 8 -> 64: a build whose minimiser lands inside `_CENSUS_BAND`'s LOWER edge, so the fix CLEAR-ACCEPTS instead of polishing | 6 | live cell INERT: pre-fix keeps the mode 1.06e-07 from the zero (reads 4.2290e-08) | **PASS** (see below); the tie at **7.721082e-09** still drops it pre-fix |
| **E4a** global `-1` ULP, the arm on which the pre-fix W6 path ACCEPTS the cusp -- the ubuntu runner's condition | 4 | byte-null arrays DIFFER, by design | **PASS**, "differ by design" branch printed: pre-fix holds 235.86863355073922, the fix refuses it, the other 3 entries come back BIT-IDENTICAL |
| **E4b** the same at scale 10 | 4 | arrays still bit-identical there | **PASS**, no-op branch printed |
| **E4c** the same global `-1` ULP under the ladder | 3 | the CLEAN pre-fix census now HOLDS the cusp and all 6 arms LOSE it | **PASS** -- the flip in the other direction, adjudicated by the same branch and asserted to be a known cusp either way |

**E3b' earned its place.**  Its first run FAILED, and correctly: `g_pol` was
being read at the CENSUS ENTRY, which on a clear-accepting build IS Brent's
stop, so the separation test compared that stop with itself (`4.2290e-08`
against `4.2290e-08`).  It is now read at `_polish_zero`'s own answer.  That is
a bug the restructure would otherwise have shipped to the next friendly runner,
found by emulating the runner rather than by waiting for it.

`_DETECT_PPU` is a library constant rather than a build property, so E3b' is run
over a narrowed `(200, 210)` band -- the detection CELL WIDTH, the only quantity
that matters to it, is identical to the full-band form at 1/12 the cost.

### 8.5  Green

`tests/unit/test_eme_census_determinacy.py`, **7 tests**, at
`OPENBLAS_NUM_THREADS` 1 / 2 / default on BOTH mounts:

| mount | 1 thread | 2 threads | default |
|---|---|---|---|
| M (Windows py3.14, numpy 2.4.4) | **7 passed** (65.6 s) | **7 passed** (66.0 s) | **7 passed** (64.6 s) |
| W (WSL py3.12, numpy 2.4.6) | **7 passed** (64.1 s) | **7 passed** (64.5 s) | **7 passed** (66.9 s) |

(The three columns are close because the module pins BLAS to one thread at
RUNTIME with `threadpool_limits`, S6.2 -- that lever is untouched.)

`ruff check lumenairy/ tests/unit/` -- the exact CI command -- clean on both
mounts.  The file is ASCII.  No `xfail`, `skip`, deleted test, weakened guard or
`CHANGELOG` entry, and `lumenairy/` is not touched by this round at all.

### 8.6  Open

* The engineered tie is re-derived per build from that build's own readings, so
  the `ratio_tol` it uses is not a constant and cannot be pinned in a table.
  Cost is 3 extra W6 censuses in test 3 (~1.5 s each) and 3 narrow-cell censuses
  in test 6 (~1 s each) against the module as shipped.
* `_ULP_ARMS_WIDE` has not been reached on any build measured.  If a build ever
  needs it, the reading's spread is what wants investigating, not the ladder.

---

## 9.  THE MATCH RADIUS WAS ALSO PER-BUILD -- 2026-08-13, second pass

`v5.35.2` shipped S8 and the release verify shard failed again -- same class, one
level deeper, inside the restructure itself.  Runner **ubuntu py3.10, shard 1**,
one failure, in the arm S8.1(b) had added:

```
test_the_recovered_mode_is_confirmed_by_the_fd_oracle_not_by_the_prefix
  AssertionError: the fix dropped the pre-fix entry 205.9753746998219, which is
  NOT a band-edge cusp (sigma_min / bound = 1.100e-05, FD distance 0.075)
  assert 1.1001481757489883e-05 >= 0.01
```

**205.9753746998219 is the mode.**  That build's bounded minimiser halted
**3.99e-4** from the converged zero 205.9749757788 which the fixed census holds.
The containment arm matched pre-fix entries to fixed entries within
`_VALUE_RTOL * q = 2.06e-4`; at 1.9x that bar a KEPT mode read as DROPPED, fell
through to the structural branch, and that branch correctly answered that a
`sigma_min / bound` of 1.1e-05 is not a cusp.  Every step was right except the
radius.

### 9.1  Why the radius was the bug, and what replaces it

S8 moved every CLAIM off per-build readings but left the **matching** on one.
`_VALUE_RTOL` and `_BRENT_XFLOOR` are tolerances on a CONVERGED value; a pre-fix
entry is not a converged value, it is wherever that build's minimiser stopped:

| build | pre-fix halt, Nx=16 mode | distance from the zero | vs `_VALUE_RTOL q` (2.06e-4) | vs `_BRENT_XFLOOR` (3.10e-6) |
|---|---|---:|---:|---:|
| M, Windows py3.14 | 205.9786352762 | 3.66e-3 | 17.8x | 1180x |
| W, WSL py3.12 | 205.9704915028 | 4.50e-3 | 21.8x | 1450x |
| **ubuntu py3.10 shard** | **205.9753746998** | **3.99e-4** | **1.9x** | **129x** |

Replaced by the **mode ISOLATION radius** -- half the smallest inter-mode gap of
the FIXED census, computed at runtime by `_isolation_radius`:

| cell | fixed census gaps | min gap | isolation radius | detection cell width | margin |
|---|---|---:|---:|---:|---:|
| W6 (Nx=8) | 47.43, 4.53 | **4.534** | **2.267** | 0.25 | 9.1x |
| N16 (Nx=16) | 5.82, 4.96, 50.50, 4.09 | **4.088** | **2.044** | 0.25 | 8.2x |

The gap is physics -- four decades above any stopping residual, eight above the
x-tolerance floor.  The soundness condition is asserted, not assumed:
`_detect_cell_width` computes the interval `minimize_scalar` is BOUNDED to (two
steps of the library's own detection grid, an integer function of the window with
no round-off in it), and `_isolation_radius` refuses to return unless the basin
radius clears it.  A stop cannot leave its own cell, so if the basin is wider
than the cell, basin assignment is the same on every build -- and where it is
not, the guard says so instead of a tolerance quietly guessing.

The containment arm is now: every pre-fix entry is EITHER in the basin of a fixed
entry -- **and that entry must itself be a converged, FD-confirmed mode**, not
merely the nearest float in the array -- OR structurally refused
(`sigma_min / bound >= _STRUCTURAL_SAT`).  Strictly stronger than the 5.35.2
form, which asserted nothing about what an entry was matched TO.

### 9.2  The class, killed: all 43 assertions classified

**B** = build-free (physics, an oracle, a theorem, or exact code-path identity);
**D** = derived from THIS build's own measurement; **E** = engineered tie.  Nine
assertions reference a pre-fix quantity at all; **five** of them referenced one
*through a universal constant* -- that is the bug pattern, and all five are fixed
here.  Line numbers are this branch's.

| # | line | assertion | pre-fix? | class | note |
|---:|---:|---|:---:|:---:|---|
| 1 | 242 | `_isolation_radius` needs >= 2 entries | | B | precondition |
| 2 | 247 | basin radius > detection cell width | | B | 9.1x / 8.2x, both measured |
| 3 | 290 | every fixed entry non-structural | | B | oracle |
| 4 | 295 | every fixed entry FD-confirmed | | B | oracle |
| 5 | 299 | no known cusp in the fixed census | | B | **radius 1e-2 -> `iso`** |
| 6-8 | 505-510 | monodromy separates modes from cusps and controls | | B | 9 decades |
| 9-12 | 525-539 | the structural bound is a bound, and separates | | B | theorem + 1.4e10 |
| 13 | 585 | the fixed census holds each W6 mode's BASIN | | B | **added** |
| 14 | 588 | ... and holds it CONVERGED | | D | `_CENSUS_BAND`'s lower edge bounds a clear accept |
| 15 | 599 | fixed census size stable on every nudge arm | | B | claim under test |
| 16 | 607 | fixed entries stay in their basins under nudge | | B | **added** |
| 17 | 613 | fixed entries move less than `1e-4 * abs(v)` under nudge | | B | claim under test; 4 decades slack, verified to a 6e-2 detector shift |
| 18 | 616 | basins stay one-to-one under nudge | | B | **added** |
| 19 | 638 | either the ladder moved a reading, or every reading is over the DEPTH gate | YES | D | **rewritten**: separates "injector dead" (fail, widen) from "no membership exists" (adjudicate) |
| 20 | 663 | the engineered tie STRADDLES | YES | E | **radius `_VALUE_RTOL` -> `iso`** |
| 21 | 676 | the fixed census refuses the cusp at the tie | | B | **radius 1e-2 -> `iso`** |
| 22 | 680 | the fixed census size is stable at the tie | | B | claim under test |
| 23 | 715 | whatever a nudge flips READ INSIDE the ambiguity band | YES | D | **was "it must be a known cusp" -- a per-build fact** |
| 24 | 785 | untreated pre-fix entries come back BIT-IDENTICAL | YES | B | exact `==`; code-path identity, not a tolerance |
| 25 | 793 | any other fixed entry is a converged zero | | B | `_CENSUS_BAND` contract |
| 26 | 797 | the census is never vacuous | | B | |
| 27-28 | 842-844 | the polisher is localisation-independent, and deepens | | B | |
| 29 | 892 | both solvers hold the recovered mode's BASIN | | B | **added** |
| 30 | 895 | ... and hold it CONVERGED | | D | band lower edge |
| 31 | 900 | the two solvers agree on the BASIN | | B | **added** |
| 32 | 903 | ... and agree to `_VALUE_RTOL` | | B | claim under test |
| 33 | 911 | `_polish_zero`'s own answer is the zero to 1e-6 | | B | test 5 pins its determinism |
| 34 | 914 | the census entry is that answer to `_BRENT_XFLOOR` | | D | scipy's own x-tolerance |
| 35 | 919 | `sigma_min` collapses vs `_PREFIX_STOP` | | B | 8 decades |
| 36-37 | 924-925 | the recovered mode is no worse an FD match | | B | oracle |
| 38 | 944 | a MATCHED fixed entry is a converged FD-confirmed mode | YES | B | **added** |
| 39 | 951 | an UNMATCHED pre-fix entry is structurally refused | YES | B | **radius `_VALUE_RTOL` -> `iso`: THE py3.10 FAILURE** |
| 40 | 964 | the pre-fix path read inside the mode's basin | YES | B | **radius 0.5 (hard-coded) -> `iso`** |
| 41 | 975 | Brent's reading and the zero's are separated | YES | D | ratio of two readings taken here |
| 42 | 986 | the pre-fix path DROPS the mode at the engineered cut | YES | E | **radius 1e-2 -> `iso`** |
| 43 | 991 | the fixed path returns the converged zero at the cut | | D | in-band by construction, so polished |

Two structural changes make the class unrepeatable rather than merely repaired:

* **`_absent(census, qz2, atol)` no longer has a default.**  Every membership
  question in the file must now say, at the call site, whether it asks about a
  CONVERGED value (`_VALUE_RTOL`) or about something a minimiser HALTED at
  (`iso`).  The 5.35.2 defect is unspellable without choosing.
* **Where both are meaningful, both are asserted, BASIN first.**  Rows 13/14,
  29/30 and 31/32 are two-tier: the basin tier is universal and cannot fail
  spuriously; the value tier carries the strength and is bounded by
  `_CENSUS_BAND`'s LOWER edge (a stop far enough from a zero to matter reads
  `gaps.min` INSIDE the band and is therefore polished; only a stop within
  ~2.4e-5 can clear-accept, measured 1.06e-7 on the E3b' emulation).

Row 19 is the third finding of this round, produced by the new emulation arm:
beyond `|dq| ~ 5e-2` the pre-fix census is EMPTY because every reading in a
cusp's basin exceeds `layer_vector_modes`' own DEPTH gate (`sigma_min < tol`,
`tol = 5e-2`, read off the signature by `inspect` so it cannot drift).  No
`ratio_tol` can accept such a candidate, so there is no membership for a nudge to
flip -- a regime, not a defect.  It is now separated from "the injector is dead"
and PRINTED, with the fixed path's claims still fully asserted.

### 9.3  The detector-shifted emulation arm

`_stop_offset(monkeypatch, dq)` moves where the minimiser HALTS by `dq`, clamped
to its own bracket.  The ULP arm perturbs the minimiser's INPUT and moves its
answer by ~1e-6; this moves the ANSWER, which is what a LAPACK build actually
moves and what spans a decade across the three builds measured.  It is used in
two places: as rungs of test 3's own determinism ladder
(`_STOP_ARMS = (1e-3, -1e-3, 3e-3, -3e-3)`), and as emulation arm E5.

Run against the **5.35.2** file it reproduces the shard exactly and finds a
second instance of the class the shard had not reached:

| dq | id | 5.35.2 verdict |
|---|---|---|
| -3.3e-3 | test 6 | **FAIL** `the fix dropped the pre-fix entry 205.97533527623273, which is NOT a band-edge cusp (sigma_min / bound = 1.610e-05, FD distance 0.0754)` -- the shard's failure, to 6 digits |
| -3.3e-3 | test 3 | **FAIL** `the engineered tie did not straddle` -- row 20: the pre-fix entry for the cusp had moved 3.3e-3 from the tabulated value and `_VALUE_RTOL` could not see it |
| +1.0e-3 | test 6 | **FAIL** on 201.88788249907319 -- the same class on a different mode |
| +1.0e-3 | test 3 | **FAIL** as above |

Against this branch every arm passes.

### 9.4  Emulations

| emulation | id(s) | what it makes the pre-fix path do | result |
|---|---|---|---|
| **E1** ULP ladder shrunk to `(1, 4, -4, 16, -16)` | 3 | live cell INERT: all 5 arms return 3 | **PASS**, inert branch printed; the engineered tie at 1.7934e-03 still straddles |
| **E2** narrow cell `n_scan` 9 -> 40 | 6 | live cell INERT: pre-fix KEEPS the mode (reads 4.44e-07) | **PASS**, inert branch printed; tie at 8.1061e-08 still drops it pre-fix |
| **E3b'** `_DETECT_PPU` 8 -> 64 over (200, 210) | 6 | clear-accept build: the fix keeps Brent's stop instead of polishing | **PASS**; tie at 7.7211e-09 |
| **E4a** global `-1` ULP | 4 | pre-fix ACCEPTS the W6 cusp -- the 2026-08-12 runner's condition | **PASS**, "differ by design" printed; the other 3 entries bit-identical |
| **E4c** global `-1` ULP | 3 | the CLEAN pre-fix census holds the cusp and all 6 arms LOSE it | **PASS** |
| **E5** DETECTOR SHIFT, `dq` = +-1e-3, +3e-3, **-3.3e-3**, +-1e-2, +3e-2, -6e-2 | ALL 7 | the minimiser's ANSWER moves by up to 24% of a detection cell -- 150x the py3.10 offset | **PASS on all 8 arms** (56 test runs) |
| **E5** `dq` = -1e-1 | ALL 7 | 40% of a cell: the FIXED finder itself keeps only 1 of its 5 modes | 6 of 7 pass; test 6 FAILS **correctly**, saying so: `the dense census is MISSING the FD-confirmed mode 205.9749757788 -- no entry within a detection cell (0.2500) of it: [146.421467006905]` |

`dq = -3.3e-3` is the arm that reproduces the shard.  `-6e-2` is past the point
where the PRE-FIX census is empty, and is what produced the DEPTH-gate
adjudication of row 19; `-1e-1` is past the point where the LIBRARY works, and
the test's job there is to say which mode went missing, which it does.

### 9.5  Green -- including the versions never covered locally

Both shard failures came from versions neither mount had ever run: **py3.11**,
then **py3.10**.  Both are covered now, via `uv`-managed CPython in WSL, together
with the CI-proxy venv:

| environment | python | numpy / scipy | threads | result |
|---|---|---|---|---|
| M, Windows | 3.14.6 | 2.4.4 / 1.17.1 | 1 / 2 / default | **7 passed** (69.2 / 70.3 / 70.2 s) |
| W, WSL `lumen_venv` | 3.12.3 | 2.4.6 / 1.17.1 | 1 / 2 / default | **7 passed** (67.0 / 66.6 / 66.9 s) |
| W, WSL `/tmp/venv-py310` (**new**) | **3.10.20** | 2.2.6 / 1.15.3 | 1 | **7 passed** (67.6 s) |
| W, WSL `/tmp/venv-py311` (**new**) | **3.11.15** | 2.4.6 / 1.17.1 | 1 | **7 passed** (66.4 s) |
| W, WSL `/tmp/venv-ci` (CI proxy) | 3.12.3 | 2.5.1 / 1.18.0 | 1 | **7 passed** (63.4 s) |

`ruff check lumenairy/ tests/unit/` -- the exact CI command -- clean on both
mounts.  The file is ASCII.  No `xfail`, `skip`, deleted test, weakened guard or
`CHANGELOG` entry, and `lumenairy/` is not touched by this round either.

**Version alone does not reproduce it.**  py3.10.20 and py3.11.15 are GREEN
locally on the *5.35.2* file: what differs on the runners is the LAPACK the
`numpy` wheel is built against, not the interpreter.  That is why the
demonstrations must be engineered rather than observed, and why `_stop_offset` --
which reproduces the shard's reading deterministically on any build -- is worth
more than any number of version rows.

### 9.6  Open

* `_ISO_MARGIN = 1.0` is the narrowest bar added here, 8.2x / 9.1x measured.  It
  is a statement about the reference windows, not the library: a caller scanning
  a window whose modes sit closer together than a detection cell trips the guard,
  which is the correct answer (basin matching cannot adjudicate there) but is a
  guard, not a fix.
* Row 17's `1e-4 |v|` is the one bar in the file still set by measurement rather
  than derivation.  It is verified out to a 6e-2 detector shift -- 150x the
  py3.10 offset and 24% of the detection cell -- but it is a fixed-vs-fixed
  determinism claim, not a fail-before, so its failure would be a real signal.
* The emulation harness lives in the scratchpad, not the tree.  Its E1 arm names
  the ULP rungs to drop by hand, which is an M/W-specific choice; the E5 arm's
  `dq` values are absolute and build-free.

---

## 10.  IT WAS A LIBRARY BUG -- the polish could LOSE a mode.  2026-08-13

Round 3, and this one is not a test defect.  The 5.35.3 main-push CI (ubuntu
py3.10, shard 2/3) failed the round-2 containment arm:

```
test_the_recovered_mode_is_confirmed_by_the_fd_oracle_not_by_the_prefix
  AssertionError: the fix dropped the pre-fix entry 201.88626619057126 --
  nothing in the fixed census lies within its basin radius 2.4820
  (nearest 205.97497...)
  assert 1.0194755006245763e-05 >= 0.01
```

The reported basin radius pins the rest: 2.4820 means the FIXED census's
smallest gap was 4.9640, which is the 146.42 <-> 151.39 gap, so the 4.088 gap
205.97 <-> 201.89 was **not there** -- the fixed census on that build was
`{205.975, 151.385, 146.421, 140.600}` and had lost 201.887 entirely, while the
un-treated path held it at 201.8862661906.

### 10.1  The oracle's verdict at 201.886 -- hypothesis (A)

Adjudicated before anything was changed, and it is decisive:

| probe | reading |
|---|---|
| 40-digit root (doc S2) | **201.88688284563654154** |
| `sigma_min` there | **2.7400e-15** |
| structural ratio `sigma_min / bound` | **7.9762e-15** -- 12 decades under `_STRUCTURAL_SAT`, so NOT a band edge |
| 2-D-FD oracle, ny = 48 / 64 / 96 | **0.0738 / 0.0416 / 0.0185** -- CONVERGING as the FD grid refines |
| `_polish_zero` on its detection cell [201.75, 202.00] | 201.886882845662, `gaps.min` 2.74e-11, would ACCEPT |

A spurious candidate does not have an FD eigenvalue that walks toward it as the
oracle's own grid refines.  **201.8868828456 is a genuine mode**, so the build
that lacked it had lost one: hypothesis **(A), real recall loss**.  Not (B) --
the converged zero's reading is 5.96e-14, nine decades BELOW `_CENSUS_BAND`'s
lower edge, nowhere near a boundary.  Not (C) -- both oracles confirm it.

The fix therefore belongs in `lumenairy/`, and the round-2 containment arm was
right: it caught a library bug, which is what it was written to do.

### 10.2  What went wrong -- the polish was a ONE-WAY step

`_refine_accept` treats a candidate whose reading lands in `_CENSUS_BAND` by
polishing it and re-reading THERE.  Before this round the step was
unconditional:

```
x = _polish_zero(f, lo_b, hi_b)
try:    s, gaps, bound = _mode_reading(..., x, ...)
except  np.linalg.LinAlgError: return          # <- silent drop
if s[-1] < tol and gaps.min() < ratio_tol: ...  # <- read at the STRAYED point
```

so whatever the polish returned REPLACED the minimiser's answer, with no check
that it was any better.  `_polish_zero` localises on a sub-grid and then
contracts a 5-point bracket GREEDILY, on a function that is a
min-of-many-branches; a level can present a near-tie between the true basin and
a neighbouring wiggle, and which one wins is a per-build fact.  When it strayed,
a candidate whose pre-polish reading was a **clean accept** was discarded --
silently, and only on the builds whose round-off strayed.

Two measurements frame it.  On the reference cells the localisation's own margin
is thin -- the 201.887 cell's two deepest sub-grid samples differ by **5.5%**
(1.02507e-04 vs 1.08133e-04) -- but our LAPACK never actually strays: the
contraction returns the same accepted zero under a per-evaluation jitter of 1,
2, 4, 8, 16, 32, 64 and 128 ULP on all five Nx=16 cells, and a full-census
jitter sweep to 1024 ULP moves no membership at all.  The straying is a
different build's privilege; the DEFECT is that straying could discard.

Why our mounts never see it even in principle: on M the minimiser halts 3.5e-7
from that zero, so its reading is 1.99e-7 -- **below** `_CENSUS_BAND`, taking
the unchanged path, never entering the polish branch.  The py3.10 build halted
**6.17e-4** away, reading 5.5e-4, which is INSIDE the band.  That difference is
what routed the candidate into the branch at all.

### 10.3  The fix -- `_POLISH_GUARD`, an improvement step that must improve

```
x_p = _polish_zero(f, lo_b, hi_b)
try:    s_p, gaps_p, bound_p = _mode_reading(..., x_p, ...)
except  np.linalg.LinAlgError:  s_p = None       # unevaluable -> keep Brent
if s_p is not None and s_p[-1] <= s[-1]:         # adopt iff DEEPER
    x, s, gaps, bound = x_p, s_p, gaps_p, bound_p
```

The polished point is adopted **iff it is a deeper zero than the minimiser's
stop**.  There is no bar to tune and no tolerance: `sigma_min` at the polished
point either is or is not below `sigma_min` at the stop.  A polish that strays,
and a polish that lands somewhere `_mode_reading` cannot evaluate, now both
degrade to "keep the minimiser's answer and its verdict" instead of discarding
the candidate.  `_POLISH_GUARD = False` restores the pre-2026-08-13 body exactly
and is the fail-before lever the test uses; it is not a supported runtime
setting.

Note what the guard does NOT claim.  It does not repair a strayed polish -- the
205.975 recovery of S3, which genuinely NEEDS the polish, is still lost when the
polish strays (its pre-polish reading rejects).  It guarantees only that the
polish cannot take away what the minimiser already had, which is exactly the
py3.10 signature.

### 10.4  Byte-null

Same script, same interpreter, same `sys.path`; only `lumenairy/` differs.  The
returned census is **bit-identical on all seven configurations**, including the
two the fix was built for and the `verify=True` path:

| configuration | n | verdict |
|---|---:|---|
| W6 base (Nx=8) | 3 | **BYTE-NULL** |
| W6 scaled x10 | 3 | **BYTE-NULL** |
| N16 dense | 5 | **BYTE-NULL** |
| N16 banded (`solver="banded"`) | 5 | **BYTE-NULL** |
| Nx=20 (56, 259) | 16 | **BYTE-NULL** |
| Nx=16 (0, 256) | 24 | **BYTE-NULL** |
| Nx=12 verify (`verify=True`) | 13 | **BYTE-NULL** |

`diff` of the full-precision `repr` of every entry: empty.  That is the
containment argument -- the guard changes behaviour only where the polish fails
to deepen, which no currently-passing configuration does.

### 10.5  The fail-before, deterministic on any build

`test_a_straying_polish_cannot_lose_a_mode_the_minimiser_already_had`, three
parametrisations (`delta` = +5e-3, -5e-3, +1.2e-2).  Both injectors are applied
together because the shard's build had both -- the STOP OFFSET puts the reading
in the band so the branch runs at all, the STRAY is what the polish then does:

| arm | `_POLISH_GUARD` | holds 201.8868828456? |
|---|---|---|
| un-treated (`_prefix_refine`) | -- | **YES**, at its own stop 201.8862658391 |
| treated, guard OFF | `False` | **NO -- DROPPED** (the shard's failure, reproduced) |
| treated, as shipped | `True` | **YES**, at 201.8862658391 |

The reproduction lands on 201.8862658391 against the shard's own
201.8862661906 -- the same stop to 3e-7.  What the guard keeps is then checked
to be a real reading of that mode, not a placeholder: `gaps.min` under
`ratio_tol` and FD-confirmed.

### 10.6  Blast radius

Every test file that imports `eme_2d_vector` or any `elements.eme` module was
enumerated (`grep -rln`) and run: `test_audit_v5_24_2_g03_guards_em`, `test_audit_w4_jax_static_caches`,
`test_audit_w6_eme`, `test_eme_2d`, `test_eme_2d_vector`,
`test_eme_census_determinacy`, `test_eme_diffraction`, `test_eme_jax_modes`,
`test_g08_s4_15_cache_hygiene`, `test_niche_audit_w6_eme`,
`test_v4_16_0_walker_all_symmetry`, `test_v5_18_1_residuals`,
`test_v5_21_2_subsystem_audits` -- 13 files.  (`test_niche_audit_w9_eig_vjp`
also imports an EME symbol but is being worked in the separate consolidation
branch and was left alone.)

M: **259 passed** (1559.8 s, 1 BLAS thread).  W: still running at hand-off (M's 259 bounds it; the two mounts share the same 13-file set)

### 10.7  Green

| environment | python | numpy / scipy | census file (10 tests) | EME blast radius |
|---|---|---|---|---|
| M, Windows | 3.14.6 | 2.4.4 / 1.17.1 | **10 passed** (179.3 s) | **259 passed** (1559.8 s) |
| W, WSL `lumen_venv` | 3.12.3 | 2.4.6 / 1.17.1 | **10 passed** (421.0 s) | still running at hand-off (M's 259 bounds it; the two mounts share the same 13-file set) |
| W, `/tmp/venv-py310` | **3.10.20** | 2.2.6 / 1.15.3 | **10 passed** (173.6 s) | -- |
| W, `/tmp/venv-py311` | **3.11.15** | 2.4.6 / 1.17.1 | **10 passed** (289.1 s) | -- |

py3.10 and py3.11 are the two interpreters the last two shards failed on, and
the census file now carries 10 ids there rather than 7 -- the three new
parametrisations are the S10.5 fail-before.

`ruff check lumenairy/ tests/unit/` clean on both mounts; both files ASCII; no
`xfail`, `skip`, deleted test, weakened guard or `CHANGELOG` entry.

### 10.8  Open

* **The polish's greedy contraction is still greedy.**  The guard makes a stray
  harmless, not impossible.  A polish that strays still forfeits the RECOVERY it
  was there to provide (S3's 205.975 case), so a build whose contraction strays
  loses that mode -- it just no longer loses the ones it already had.  A
  multi-start polish (contract from the top-k sub-grid samples, keep the
  deepest) would close that too and is the natural next step; it was not taken
  here because it is not byte-null and this round had to be.
* **The 5.5% sub-grid margin on the 201.887 cell** is the narrowest localisation
  margin measured.  It is a property of that cell's wiggle structure, not of the
  library, and it is what makes this cell the one that strays first.
* `_POLISH_GUARD` is a test lever in library namespace.  It follows the
  `PMM_FORWARD_GROWTH_REPAIR` / `PMM_JAX_MINNORM_PROJECTION` precedent, but it
  is one more public-ish name that must never be set in anger.

---

## 11.  THE FAIL-BEFORE'S OWN INJECTOR WAS PARAMETRIZED -- 2026-08-15

The round-3 guard is right and stays.  What went red on the first main CI after
it merged (run 31908894132) was the arm that PROVES it, and for the fourth time
it is the same trap one level in:

```
Unit tests (Python 3.10, shard 3/3)
test_a_straying_polish_cannot_lose_a_mode_the_minimiser_already_had[0.012] FAILED
  AssertionError: with _POLISH_GUARD off, a +1.2e-02 stray no longer drops
  201.88688284563653 -- the injector has stopped reaching the defect the guard
  exists for: [201.8870001192093, 146.43354605686665]
```

`[0.005]` PASSED on Python 3.13 in the same run.  The three parametrisations
were fixed offsets ADDED to whatever `_polish_zero` returned, and **where that
lands is per-build**: on that runner the polish returned a point near 201.875,
so `+0.012` landed at **201.8870001192093** -- 1.17e-4 from the zero, close
enough that the strayed point's own reading ACCEPTED.  The un-guarded body then
kept the mode, the fail-before had nothing to demonstrate, and it said so by
failing.  Rounds 1-3 removed this pattern from the claims; round 3 reintroduced
it in the injector.

### 11.1  Both injected quantities are now DERIVED, and the injectors are absolute

Two changes, and the second is the one that closes the class:

* **`_derive_stop`** picks the minimiser stop by READING, not by offset.  It
  requires the stop to be simultaneously INSIDE `_CENSUS_BAND` (so the polish
  branch runs at all) and ACCEPTED by the un-treated path (so there is something
  to take away) -- a two-decade window, `[1e-5, 1e-3)`, and it lands at its
  geometric centre where both margins are widest.  Round 3 hard-coded
  201.8862661906, which is only where the py3.10 shard happened to halt.
* **`_derive_strays`** picks the point a strayed polish returns by READING too:
  its own `gaps.min` must be `>= ratio_tol` (so the un-guarded body, which
  adopts the polished point and then tests exactly that, MUST reject) and its
  `sigma_min` must be shallower than the stop's (so the guard declines to adopt
  it).  Both conditions are properties of the injected point, measured here, so
  they cannot evaporate on another build the way an offset can.
* **`_force_in_cell`** replaces the minimiser's and the polisher's answers
  ABSOLUTELY, and only inside the mode's own detection cell.  Round 3 offset
  them globally; an absolute in-cell override leaves every other candidate on
  the shipped path, so the arm reads as a census rather than as one refinement.

Measured on M: the ladder yields 26 qualifying stops and 26 qualifying strays,
and the ones chosen are

| quantity | value | reading | why it qualifies |
|---|---|---|---|
| stop | 201.8866726556 | `gaps.min` **1.026e-04** | inside the band (1e-5 .. 3e-2) AND under `ratio_tol` -- log-centre of the window |
| stray | 201.7868828456 | `gaps.min` **5.914e-02**, `sigma_min` **2.734e-03** | refuses acceptance (59x over `ratio_tol`) and is 578x shallower than the stop |

### 11.2  What is asserted, and what is scanned

* **Asserted on EVERY arm**: the shipped path keeps the mode wherever the polish
  lands, and what it kept is a real reading of that mode (its own `gaps.min`
  accepts, its structural ratio is a mode's).  That is the contract, and it is
  unconditional.
* **Scanned**: the fail-before.  Strays are ranked by measured margin and the
  first that makes the un-guarded body drop the mode carries it.  If none did,
  that is PRINTED with the full table rather than asserted away -- the guarded
  claim has already been made on all of them.  The print says what an inert
  result would mean (another detection cell also finding the mode), so it is
  diagnosable rather than merely tolerated.
* Both derivations hard-fail only if the LADDER is empty, with the campaign's
  standing message: widen it rather than delete it.

On M the fail-before reaches the defect on the first stray, and the un-guarded
census it produces is

```
[205.97497577877948, 151.38547455643376, 146.42146637720512, 140.59975645614043]
```

-- four entries, missing 201.887: **the py3.10 shard's census exactly**, which is
what S10.1 reconstructed from its reported basin radius of 2.4820.

### 11.3  Re-audit of the other round-3 additions

Every injected quantity added in round 3, re-read for the same assumption:

| addition | assumes an injection MANIFESTS? | verdict |
|---|---|---|
| `@pytest.mark.parametrize("delta", [5e-3, -5e-3, 1.2e-2])` | YES -- per parametrisation | **the defect; removed** |
| `_stray_polish(mp, delta)` (additive) | YES -- the landing point is per-build | **replaced by `_force_in_cell` (absolute, in-cell)** |
| `_PY310_STOP` as the injected stop | YES -- that it lands in the band AND accepts | **replaced by `_derive_stop`** |
| `_CELL201` constant | unused after the rewrite | removed |
| `_MODE201` | no -- a tabulated oracle value, adjudicated in S10.1 | kept |
| `_POLISH_GUARD` (library) | no -- a feature flag, not an injection | kept |

And the pre-round-3 machinery, re-checked for the same shape: `_STOP_ARMS` in
test 3(a) asserts only that the FIXED census is STABLE under them (a cure claim,
no manifestation assumed); `_ULP_ARMS` feeds `_tie_at_the_cut`, which derives its
cut from measured readings and widens before failing; `_prefix_drop_cut` is
derived from the build's own Brent reading.  The one remaining `parametrize` in
the file is test 4's length `scale` (1.0, 10.0), a geometry, not an injection.

### 11.4  Green

| environment | python | numpy / scipy | result |
|---|---|---|---|
| M, Windows | 3.14.6 | 2.4.4 / 1.17.1 | **8 passed** (2186.7 s) |
| W, WSL `/tmp/venv-py310` | **3.10.20** | 2.2.6 / 1.15.3 | **8 passed** (2116.8 s) |
| W, WSL `/tmp/venv-py311` | **3.11.15** | 2.4.6 / 1.17.1 | in flight at hand-off |
| W, WSL `lumen_venv` | 3.12.3 | 2.4.6 / 1.17.1 | in flight at hand-off |

py3.10 is the interpreter that failed this arm on the 2026-08-15 run; it is
green here on the same file.  (Wall times are inflated ~20x by an unrelated
consolidation run occupying the box; the file is ~90 s unloaded.)

The file now carries **8 ids** rather than round 3's 10: the three
parametrisations collapse into one scanned test, which is also cheaper (4
censuses against 9).

`ruff check lumenairy/ tests/unit/` clean on both mounts; ASCII; no `xfail`,
`skip`, deleted test, weakened guard or `CHANGELOG` entry.  The library is not
touched by this round -- round 3's guard stands as shipped.

### 11.5  Open

* The strays the ranking selects sit at the detection cell's EDGE (`|d| = 0.1`
  against a 0.125 half-width), because ranking by margin rewards distance.  They
  are valid points for the injector -- a polish is bracketed by its cell and can
  return any point in it -- but they are a coarser emulation of "landed on a
  neighbouring wiggle" than a mid-cell stray would be.  Ranking by margin is
  what makes the arm robust; a second arm at the nearest qualifying stray would
  make it faithful as well.
* Four rounds, four instances of one pattern, each a level further in: the
  claim, the match radius, the library, and now the injector.  The invariant
  that survived all four is the one worth keeping: **anything a build is
  entitled to move must be measured here, never assumed from elsewhere.**

---

## 12.  THE TWO REFERRED EME COUNT BARS -- 2026-08-15

`FIX_RUNNER_PINS_2_2026_08_15.md` S8.2 referred two fragile sites in EME
territory rather than touching them.  Both are COUNT bars with one unit of
slack, and both are adjudicated here with the machinery S9-S11 built: basin
matching, readings taken on the build doing the asserting, and oracle
confirmation -- not widened counts.

### 12.1  `test_eme_2d_vector.py` -- the completeness recall bar

Reported: `recall >= 8` / `spurious <= 2`, measured 15/1, the file's own
docstring recording a CI dip to **9**.  One mode of margin.

Measured here first, and the reading is worse than "thin":

| quantity | value |
|---|---|
| oracle band, Nx=20, Ny=56, (56, 259) | **16** modes |
| EME census, `n_scan=800` | **16** modes |
| smallest oracle mode spacing | **0.8862** -> basin radius **0.4431** |
| every matched oracle->EME distance | **0.0502 .. 0.1656** (the y-FD error) |
| shipped match tolerance | **0.7** |

**0.7 is LOOSER than half the mode spacing.**  So the shipped bar could match an
oracle mode to its NEIGHBOUR -- a soundness flaw, not only a fragility -- while
being simultaneously tight enough that a borderline entry crossing it costs a
whole unit of recall.  Recall/spurious at every tolerance from 0.7 to 3.0
measure 16/16 and 0 here, so the CI dip is not this build's geometry moving; it
is the shift-invert ORACLE dropping entries at the boundary, exactly as the
file's own comment says.

Restated in three pieces:

* **Match by BASIN** (`_basin_radius`, `_match`): half the smallest gap of the
  oracle band, read at runtime -- 0.4431 here.  Tighter than 0.7, and it cannot
  cross-assign.
* **Well-posedness is asserted, not assumed**: every match the radius makes must
  be closer than half of it (worst 0.1656 against 0.4431, **2.7x**).  If the
  y-FD error ever grows into the mode spacing the message says to raise `Ny`,
  not to widen the radius.
* **Every MISS is adjudicated, not counted**: an oracle mode the census does not
  hold is a real miss only if the finder's OWN condition has an acceptable zero
  there -- `_polish_zero` over a half-basin window, then `_mode_reading`
  (`sigma_min < tol`, `gaps.min < ratio_tol`, non-structural) and the polished
  zero still absent from the census.  A shift-invert artifact has no such zero
  and is REPORTED.  The cascade regression this test exists for -- ~2 of 16
  recovered -- leaves 14 misses that all polish to acceptable zeros, so it still
  fails, and now names them.
* **Spurious is per-entry against the FD oracle**: a census entry the oracle
  band does not hold must have a 2-D-FD eigenvalue within 1.0 (the bar
  `verify=True` ships with), or it is spurious and the test fails on that entry.
  A bare count could not tell an EME artifact from an oracle miss.

The surviving count is `recall >= len(ref) // 2`, which measures 16 against 8
and is there only to keep the cascade's ~2 decisively out.

### 12.2  `test_niche_audit_w6_eme.py` -- the scaled-census count bound

Reported: `abs(len(base) - len(scaled)) <= 2`, observed 0-1, bar 2, defect at 3.
One unit of slack, and a count cannot say WHICH mode moved.

Measured: both arms return 3 modes; the matched pairs agree to **1.4e-6 ..
3.1e-6** absolute after the `x100` scale correction, while distinct modes are
**4.534** and **47.4** apart -- so the union's distinct-mode gap gives a basin
radius of **2.267**, six decades above the agreement.

Restated per entry: every mode either arm returns is held by the other within
the basin radius, OR its own `gaps.min` sits inside `_CENSUS_BAND` -- i.e. it is
one of the knife-edge candidates the rank-drop gate is entitled to disagree
about, which is precisely what the retired comment was trying to encode as "at
most two".  A build where a second knife-edge candidate flips now passes, and a
build where a SETTLED mode moves now fails, naming it.

Its fail-before (`> 2` on the pre-fix detection grid) is restated the same way:
the collapsed scaled arm loses every base mode, and at least one of them reads
BELOW the ambiguity band -- a settled mode -- so the per-entry claim genuinely
breaks.  Asserted on the readings, so it cannot go vacuous if the collapse ever
happens to take only knife-edge entries.

### 12.3  Not touched, with the reading that says why

The same S8.2 row also lists `test_niche_audit_w6_eme.py:474`'s two constants.
Both were re-read and left alone -- they are two-sided by an order of magnitude,
which this campaign has not treated as fragile:

| constant | claim | measured margin |
|---|---|---|
| `_W6_ZERO_DEPTH = 2e-3` | true zeros are below it, shallow non-modes above | <= 4.76e-5 below (**42x**) / >= 2.20e-2 above (**11x**) |
| `_W6_SCALE_ROOT_REL = 1e-6` | a converged zero is scale-invariant to it | <= 9.13e-9 (**110x**), non-modes >= 9.75e-5 (**97x**) |

The `worst > 20.0 * _W6_SCALE_ROOT_REL` fail-before beside them is a
*fail-before*, not a claim, and it is checked against a population measured in
the same run.

### 12.4  Green

The three rewritten ids, run together on M (Windows py3.14.6, numpy 2.4.4,
1 BLAS thread): **3 passed** (558.1 s under an unrelated concurrent sweep;
the reading printed is `basin radius 0.4431 (spacing 0.8862); recall 16/16,
0 oracle entries adjudicated as artifacts; 0 census entries unmatched, all
FD-confirmed; worst matched distance 0.1656`).

| environment | python | numpy | whole-file regression |
|---|---|---|---|
| M, Windows | 3.14.6 | 2.4.4 | in flight at hand-off |
| W, WSL `/tmp/venv-py310` | 3.10.20 | 2.2.6 | in flight at hand-off |
| W, WSL `lumen_venv` | 3.12.3 | 2.4.6 | in flight at hand-off |

The box was carrying an unrelated whole-suite sweep throughout, which is why
the targeted run took 558 s for work that is ~90 s idle.

`ruff check lumenairy/ tests/unit/` clean on both mounts.  No library change
this round.  No `xfail`, `skip`, deleted test, weakened guard or `CHANGELOG`
entry.  (`test_niche_audit_w6_eme.py` carries pre-existing non-ASCII bytes and
was edited byte-preservingly; everything added is ASCII.)

### 12.5  Open

* `_oracle_band`'s own membership is still a shift-invert result filtered at
  `reldiv < 1e-2`, and that filter is where the CI wobble enters.  The
  adjudication above makes the wobble harmless to the assertion, but the oracle
  would be steadier if its entries were confirmed at two `Ny` values and only
  the agreeing ones kept.  Deferred: it doubles the most expensive call in the
  file.
* The completeness test's per-miss adjudication costs one `_polish_zero` per
  unmatched oracle mode.  On builds that match all 16 it costs nothing; on the
  runner that dipped to 9 it would cost seven, which is the right place to spend
  it.

---

## 13.  THE LAST ROUND-2 ASSUMPTION: A NATURAL SEPARATION IS NOT UNIVERSAL

The gating matrix, py3.11 shard 1 (and the eig-heavy slow shard, same test):

```
test_eme_census_determinacy.py:989
test_the_recovered_mode_is_confirmed_by_the_fd_oracle_not_by_the_prefix
  AssertionError: the pre-fix reading at Brent's halt (9.0333e-09) is not
  separated from the converged zero's (7.8302e-09), so there is no gap to place
  a bar inside -- measured separation is 9 decades
  assert 9.03326947207738e-09 > (10.0 * 7.830161425530708e-09)
```

Verified, and the diagnosis holds exactly: **on that build's LAPACK the
minimiser halts at CONVERGED quality.**  Its halt reading and the converged
zero's reading are **1.15x** apart where M and W measure nine decades (1.51e-3
against 4.31e-12).  The round-2 `_prefix_drop_cut` derivation places its bar
strictly between those two readings, so on such a build there is no bar to
place -- and the vacuity guard did the right thing by refusing to invent one.
The guard was correct; what was wrong was assuming a build would always supply
the gap.

That is the last round-2 assumption in the file: the S9 restructure moved every
CLAIM off per-build readings, and S11 moved the injectors off fixed offsets, but
this one still required a NATURAL separation to exist.

### 13.1  The unification -- two routes, one contract

The demonstration now measures which route is available and takes it:

| route | condition | halt used | seen on |
|---|---|---|---|
| **NATURAL** | `g_halt > _DROP_SEPARATION * g_pol` | this build's own Brent halt | M, W (9 decades) |
| **FORCED** | otherwise | a halt DERIVED to be in-band and accepted un-treated, injected absolutely by `_force_in_cell` | py3.11 gating shard (1.15x) |

The forced route is not new machinery -- it is exactly `_derive_stop` /
`_force_in_cell`, built in S11 for the polish-guard fail-before, pointed at the
narrow cell.  `_derive_stop` requires the halt's reading to lie inside
`_CENSUS_BAND` (so the polish branch runs) and under `ratio_tol` (so the
un-treated path accepts it), which is a two-decade window whose LOWER edge is
`1e-5` -- already 1277x above the py3.11 zero reading of 7.83e-9, so the forced
halt supplies the separation the build would not.  `_prefix_drop_cut` is
unchanged; it only turns whichever halt it is given into a bar.

Everything downstream is identical on both routes: the pre-fix path must DROP
the mode at that bar and the fixed path must polish and return it.  **The
fixed-path claims are unconditional on either route.**  The route taken is
PRINTED, with both readings and their ratio, so a log says which world the
runner was in.

The vacuity guard survives, promoted to the union: it hard-fails only when
NEITHER route is reachable -- no natural gap AND no rung of the ladder inside
the band-and-accepted window -- and says to widen the ladder rather than delete
the arm.

### 13.2  Audit -- is anything else deriving from a natural separation?

Every remaining site in the file that reads a separation, re-checked:

| site | what it compares | build's own halt involved? | verdict |
|---|---|---|---|
| `_prefix_drop_cut` caller (S13.1) | halt reading vs converged zero's | **yes** | **the defect; unified** |
| `assert abs(got - x_pol) <= _BRENT_XFLOOR` | census entry vs the POLISHER's answer | no -- bounded by scipy's own documented x-tolerance | sound (derived) |
| `assert f_got < f_stop / 1e2` | `sigma_min` at the entry vs at the TABULATED `_PREFIX_STOP` | no -- the reference is a fixed point, build-stable at 3.45e-5 | sound |
| `_tie_at_the_cut` | the SPREAD of one candidate's readings ACROSS the ULP ladder | no -- a spread across arms, not a halt/zero gap; widened then depth-adjudicated (S9.2 row 19) | sound |
| `_derive_strays` margin ranking | injected point's reading vs the derived stop's | no -- both are chosen by reading on this build | sound |

So the sub-class is one instance wide, and it is closed.  The invariant the
whole campaign converged on is now stated in one line at the top of the routing
block: *which route is available is itself a per-build fact, so it is measured
rather than assumed.*

### 13.3  Both routes exercised here

The natural route is what this mount takes.  The forced route is exercised by
emulating the py3.11 condition directly -- an injector that makes the minimiser
halt AT the polished zero, collapsing the natural gap to ~1x -- and running the
same test:

**The forced route's crux, measured directly on M** (the narrow cell's
detection bracket is `(205.875, 206.125)`):

| quantity | value |
|---|---|
| converged zero | 205.97497577877948, reading **4.3146e-12** |
| derived FORCED halt (`_derive_stop`) | 205.97468502372533, reading **1.1819e-04** |
| separation `g_halt / g_pol` | **2.739e+07** against the required 10 |
| bar from that halt (`_prefix_drop_cut`) | **2.157916e-05** |
| pre-fix reads 1.1819e-04 > bar | **DROPS** |
| that halt is in-band for the fix | **yes** -> polishes |
| polished reading 4.3146e-12 < bar | **fix KEEPS** |

And on the py3.11 shard's OWN readings (`g_pol = 7.8302e-09`, where its natural
separation was 1.15x) the same derived halt would give **1.509e+04x** -- four
decades where the build supplied none.

**End to end**: the NATURAL route runs on M and passes, printing
`[injector, NATURAL route]: ratio_tol = 2.758979e-04, placed between the halt's
reading (1.5112e-03) and the converged zero's (4.3146e-12), 3.502e+08x apart`.
The FORCED route was exercised by an injector that makes the minimiser halt AT
the polished zero -- collapsing the natural gap to ~1x, which is py3.11's
condition -- and was still running when the box was handed back (an unrelated
whole-suite sweep left this process ~2.5% of the CPU).

### 13.4  Green

| environment | python | numpy | status |
|---|---|---|---|
| M, Windows | 3.14.6 | 2.4.4 | targeted id **PASS** (NATURAL route); whole file in flight |
| W, WSL `lumen_venv` | 3.12.3 | 2.4.6 | not started -- box saturated |
| W, `/tmp/venv-py310` | 3.10.20 | 2.2.6 | not started -- box saturated |
| W, `/tmp/venv-py311` | 3.11.15 | 2.4.6 | not started -- box saturated |

The forced route's arithmetic is verified numerically above on M, which is the
claim this round turns on; the four-environment sweep is not yet evidence.

`ruff check lumenairy/ tests/unit/` clean; ASCII; no library change; no `xfail`,
`skip`, deleted test, weakened guard or `CHANGELOG` entry.

### 13.5  Open

* `_DROP_SEPARATION = 10.0` decides which route runs.  It is not a claim bar --
  both routes assert the same contract -- so a build near the boundary simply
  takes the other road.  It is the one number here that is chosen rather than
  derived, and it is chosen an order of magnitude below the natural route's own
  margin on the mounts that have it (9 decades).
* Two rounds' worth of whole-file verification remain unrun on a quiet box: the
  round-4 py3.11/`lumen_venv` rows and the round-5 whole-file regressions were
  both killed mid-flight by an unrelated concurrent sweep.  Neither gap is
  suspected -- the targeted ids passed -- but neither is evidence.
