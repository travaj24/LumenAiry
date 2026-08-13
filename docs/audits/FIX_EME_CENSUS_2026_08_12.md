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
