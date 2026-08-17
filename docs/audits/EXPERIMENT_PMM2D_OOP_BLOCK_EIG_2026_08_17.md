# The out-of-plane 4Nf generator's block structure -- FOUND and SHIPPED, 2026-08-17

Branch `feat/pmm2d-oop-block-eig` off `origin/main` (9e0c8ba, the 5.38.1
release commit).  `git log origin/main..HEAD` was empty before branching.

**The question,** left open as item 3 of `EXPERIMENT_PMM2D_EIG_RECYCLE_2026_08_16.md`
S7: the out-of-plane path is the one genuinely eig-dominated 2-D PMM path
(zgeev 68% of the solve, ceiling 3.12x), and the generator's `[[A, P], [Q, B]]`
block structure is not exploited at all.  Is there a structure-preserving
reduction, and is it worth shipping?

**The verdict: yes.**  The generator is block-ANTI-DIAGONAL under an involution
built from the ORDER FLIP TIMES the E/H SIGN FLIP, so ONE `2Nf` eig delivers
all `4Nf` eigenpairs.  The condition is *normal incidence on a cell whose
assembled operators are flip-symmetric* -- which covers the common metasurface
case (a tilted-uniaxial LC pillar array at normal incidence) and is NOT
restricted to uniaxial: it holds for a fully general biaxial, non-reciprocal,
lossy 3x3.  Measured **1.61x -- 2.35x** whole-solve, interleaved, against a
1.50x ship bar and a 2.6-3.6x ceiling; eigenpair residuals BETTER than the dense
`zgeev` they replace; the flux mode selector classifies identically.  Shipped
behind an exact structural gate with a dense fallback.

**Mount.**  Windows py3.14.6, numpy 2.4.4, scipy 1.17.1, scipy-openblas
0.3.31, 24 cores / 128 GB (tesla-ryzen).  Every timing ran with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` exported **in the
environment before python started**, best-of-3 in-process after a warm-up, on
an otherwise quiet box.  Every probe asserted the import and printed it:
`C:\tmp\lum_oe\lumenairy\__init__.py`, version 5.38.1.

---

## S1.  The ceiling, re-measured on this mount before anything was built

The eig-recycle doc measured "zgeev 67.9%, ceiling 3.12x" on a 4-layer OOP
tensor stack.  Re-measured here per-call on a single-layer `pmm_jones_2d` OOP
solve (patterned tilted-uniaxial pillar, degree 7):

| `n_orders` | `4Nf` | solve | zgeev | share `f` | ceiling `1/(1-f)` |
|---|---|---|---|---|---|
| 3 | 196 | 0.0471 s | 0.0306 s | **64.9%** | 2.85x |
| 4 | 324 | 0.1543 s | 0.1042 s | **67.5%** | 3.08x |
| 5 | 484 | 0.4150 s | 0.2769 s | **66.7%** | 3.00x |

Confirmed.  Repeating the profile across four independent runs (including two
after this branch's code was final) the share moves between **61% and 72%**,
i.e. a ceiling anywhere in **2.6x - 3.6x**: the share is a per-run reading with
real spread, and only the interleaved end-to-end A/B of S5 is quoted as a
result.  What is stable across every run is the ordering -- unlike the in-plane
path, whose zgeev share FALLS with basis size because the Redheffer cascade
outgrows it, the OOP share does not decay with `n`, because the `4Nf` eig grows
at least as fast as the `2Nf` cascade.  The lever does not evaporate with `n`.

## S2.  The mathematics: what structure the `4Nf` pencil actually has

The layer ODE on the state `[E; u]` (`E` the tangential electric harmonics,
`u` the modal-H block; `H_phys = -i u`, the convention every consumer shares)
is `d[E; u]/d(k0 z) = -G [E; u]` with

```
G = [[A, P],
     [Q, B]]
```

and, writing `ez = inv(EZZ)`, `ka = [Kx; Ky]` (2N x N), `eb = [Ky, -Kx]`
(N x 2N), `ea = [EZX, EZY]`, `kb = [EYZ; -EXZ]`:

```
A = -i  ka ez ea                       (from ezx, ezy)
B = -i  kb ez eb                       (from exz, eyz)
P =     ka ez eb  +  [[0, I], [-I, 0]]
Q = [[Cyx + Kx Ky,  Cyy - Kx Kx], [Ky Ky - Cxx,  -(Cxy + Ky Kx)]]
```

(the `A`/`B` factors of `-i`/`+i` are the 2026-07-14 factor-i fix; they were
NOT inherited here -- every claim below was re-derived from the assembled
generator the library actually eigendecomposes, and the reconstruction of `G`
from its captured operators was checked byte-identical to the captured `G`,
`0.00e+00` on every fixture.)

### S2.1  Four candidates, and why three of them die

**(a) Reduction to a `2N` eig by a scalar shift.**  If `A = alpha I` and
`B = beta I` then `(gam - alpha)(gam - beta) E = P Q E` and one `2N` eig gives
everything.  DEAD by rank: `A = ka ez ea` is a `(2N x N)(N x N)(N x 2N)`
product, so `rank(A) <= N < 2N`, and a nonzero multiple of `I_{2N}` has rank
`2N`.  Hence `A ∝ I` forces `A = 0`, i.e. the in-plane case.  Measured on
every out-of-plane fixture at `N = 49`: `rank(A) = rank(B) = 48` (47 for the
off-centre cells) -- the `<= N` bound is tight -- and
`||A - (tr A / 2N) I|| / ||G||` reads 2.7e-02 .. 6.0e-02, never zero.

**(b) The similarity-invariant version of (a).**  Any similarity to
`sigma I + [[0, P'], [Q', 0]]` forces the spectrum to be symmetric about
`sigma`, because a block-anti-diagonal matrix has spectrum
`+/- sqrt(eig(P'Q'))`.  This is a NECESSARY condition testable directly on
`spec(G)`, and it is exactly what the physics decides: the extraordinary-wave
roots of an out-of-plane layer are shifted by `-(exz kx + eyz ky)/ezz` while
the ordinary pair is not, so a UNIFORM tilted uniaxial at oblique incidence
has an asymmetric quartet (the factor-i fix's 35-degree-tilted uniaxial
probe: exact `det(k x k x . + eps) = 0` roots `{-1.5214, +1.6090}`, recorded
in `_layer_eigenmodes_tensor`).  The shift is LINEAR IN THE HARMONIC's transverse
wavevector, so it is not a scalar and cannot be transformed away.  Measured
spectrum-symmetry defect: **3.8e-01 at oblique incidence** -- the reduction is
genuinely impossible there.

**(c) The disguised quadratic eigenproblem.**  Eliminating `u` gives the `2N`
QEP `gam^2 E - (A + P B P^-1) gam E - (P Q - P B P^-1 A) E = 0`.  Its leading
coefficient is `I`, so it has exactly `4N` finite eigenvalues and its standard
linearization is `G` itself: no work is saved, and the rank deficiency of the
damping term buys no deflation (deflation needs a singular LEADING
coefficient).  Eliminating `E` instead gives the mirror QEP with the same
count.  DEAD on arithmetic, not on accuracy.

**(d) Block-triangularization by a computable similarity.**  `T = [[I, 0],
[X, I]]` triangularizes `G` iff `Q + BX - XA - XPX = 0`, a Riccati equation
whose solution is as expensive as the eigendecomposition it would replace.
DEAD a priori.

### S2.2  The structure that survives: parity TIMES sign

`A` and `B` are LINEAR in `Kx, Ky`; `P` and `Q` are quadratic-or-K-free.  (The
2-D slant convection `-i (tx Kx + ty Ky) (x) I4` lands on the same two diagonal
blocks and is linear in `K` too, so it obeys the same rule.)  Let `J` be the
order flip `(m, n) -> (-m, -n)`.  At NORMAL incidence `Kx, Ky` are J-ODD, so

```
F G F = [[-A,  P], [ Q, -B]]      F = I4 (x) J          (parity alone)
S G S = [[ A, -P], [-Q,  B]]      S = diag(I, I, -I, -I)  (sign alone)
R G R = -G                        R = S . F,   R^2 = I
```

Neither factor works alone.  `F` alone is exactly the EVEN-PARITY FOLD the
library already runs for in-plane layers -- and the off-plane cross-blocks,
being J-odd, break it (which is why `twod_jones._tensor_layer_modes` refuses
to fold an out-of-plane cell).  `S` alone is exactly the in-plane
`[W; -V] <-> -lam` symmetry that makes `eig(P Q)` valid -- and the same
cross-blocks break that too.  Their PRODUCT survives both.  Measured, on the
assembled generators:

| cell (all normal incidence unless noted) | `norm(F G F - G)` | `norm(S G S + G)` | `norm(R G R + G)` |
|---|---|---|---|
| uniform tilted uniaxial | 9.6e-02 | 9.6e-02 | **0.0** |
| patterned tilted uniaxial (centred pillar) | 6.4e-02 | 6.4e-02 | **4.4e-15** |
| general biaxial, NON-reciprocal, lossy | 1.2e-01 | 1.2e-01 | **4.3e-15** |
| in-plane SLANTED tensor | 1.2e-01 | 1.2e-01 | **4.5e-15** |
| patterned tilted uniaxial, OBLIQUE th=.25 ph=.4 | 2.3e-01 | 5.8e-02 | 2.3e-01 |
| patterned tilted uniaxial, OFF-CENTRE pillar | 9.2e-02 | 6.4e-02 | 7.8e-02 |

(all three columns relative to `norm(G)`, spectral norm.)

An involution anti-commuting with `G` makes `G` block-ANTI-DIAGONAL in its
eigenbasis.  `R` is a signed permutation, so its `+1` and `-1` eigenspaces are
spanned by `(e_i +/- e_{J(i)})/sqrt(2)` -- each of dimension exactly `2N` (the
`(0,0)` order is the sole fixed point of `J`, contributing two `+1` columns
from the `E` half and two `-1` columns from the `u` half).  With
`U = [U+, U-]`:

```
U^T G U = [[0, X], [Y, 0]]     ->    gam = +/- sqrt(mu),  mu = eig(X Y)
```

with eigenvector `[w; Y w / gam]` in that basis: **ONE `2N` eig for all `4N`
eigenpairs**, the out-of-plane analogue of what `eig(P Q)` does in-plane, with
the `+/-` pair sharing one `w` exactly as the in-plane pair shares one `W`.
Measured anti-diagonality `norm(U+^T G U+)/norm(G)` = 0.0 .. 3.2e-15 across the
fixtures.

### S2.3  The algebraic condition, stated exactly

The reduction holds iff **`R G R = -G`**, i.e. iff every operator entering
`P, Q` and every operator entering `A, B` commutes with the order flip `J` in
the same recentering gauge, and `Kx, Ky` anti-commute with it.  In terms of
the inputs:

1. **normal incidence** (`kx0 = ky0 = 0`) -- otherwise the order set is not
   even closed under `J`, and the `K` operators are not J-odd;
2. **every permittivity component centro-symmetric about ONE centre** --
   note the out-of-plane components must be EVEN, the opposite of what the
   even-parity fold would need of them;
3. **the spectral-element WALL LAYOUT mirror-symmetric about that centre** --
   this is the condition on the DISCRETISATION, and it is not implied by (2).

(3) is not a technicality.  A pillar at pixels `[0:2, 1:3]` of a 6x6 cell has
a perfectly centro-symmetric permittivity about `(P/6, P/3)`, but its wall set
`{0, P/3}` x `{P/6, P/2}` produces strips whose mirror images are not strips,
so the SEM operators break the symmetry at the discretisation floor and the
generator's defect reads **2.5e-02** -- thirteen decades above the satisfied
cases (2.6e-15).  This is why the shipped gate tests the ASSEMBLED generator and not
`eps`: the condition is a property of the operators the eig actually sees.

The condition is INDEPENDENT of the tensor's character.  A fully general
biaxial, non-reciprocal (`e_xz != e_zx`), lossy 3x3 satisfies it just as the
tilted uniaxial does (4.3e-15 vs 4.4e-15) -- the reduction is not a
uniaxial-only trick, and no eps-shape gate is needed or wanted.

## S3.  Prototype: accuracy against the dense `zgeev`

Bars derived per case from the DENSE solve's own normwise backward error on
the very same matrix (nothing pinned from elsewhere), the shape the eig-recycle
experiment used:

| cell | `4N` | dense residual | factored residual | ratio | spectrum match | forward count, dense / factored |
|---|---|---|---|---|---|---|
| uniform tilted uniaxial | 196 | 2.67e-15 | **8.54e-16** | 0.32 | 1.04e-14 | 98 / 98 |
| patterned tilted uniaxial | 196 | 4.45e-15 | **2.40e-15** | 0.54 | 1.88e-14 | 98 / 98 |
| ... at `n_orders`=5 | 484 | 3.49e-15 | **3.00e-15** | 0.86 | 2.26e-14 | 242 / 242 |
| general biaxial non-reciprocal lossy | 196 | 3.70e-15 | **2.19e-15** | 0.59 | 9.74e-15 | 98 / 98 |
| in-plane slanted tensor | 196 | 3.84e-15 | **2.48e-15** | 0.64 | 1.08e-14 | 98 / 98 |

The factored eigenpairs are not merely acceptable, they are **better than the
dense solve's** on every fixture -- expected, since the reduction halves the
dimension the QR iteration works on and the reconstruction `[w; Y w / gam]` is
algebraically exact given `X Y w = mu w`.

**The flux mode selector is preserved.**  `_select_forward_flux` classifies by
`Sz` summed over harmonics with two RELATIVE noise ceilings (`3e-3 * max|Sz|`
and `|Re gam| > 0.5`), so it is sensitive to the eigenvector NORMALISATION,
not just to direction.  The factored vectors are therefore normalised to unit
2-norm, matching `zgeev`'s convention; with that, both paths select exactly
`2N` forward modes and their forward spectra match to 7.6e-15 .. 2.1e-14.
This was a real trap: without the normalisation the deep-decay override sees a
different flux scale and can shunt a growing mode into the forward set, which
blows the cascade up by `exp(+|Re gam| k0 L)`.

**Control: what a WRONG reduction costs.**  With the structural gate disarmed
and the involution mis-built -- the order flip replaced by the identity, i.e.
the E/H sign flip alone, which is the in-plane symmetry misapplied to an
out-of-plane cell -- the reduction is accepted and lands **3.2e-02** off in the
spectrum.  So the failure mode is loud, and the gate (which refuses that exact
gauge, measured) is what stands between it and the answer.  Note that the
recentering gauge on the centred-pillar fixture is `(-1)^m`, whose conjugation
leaves the involution residual algebraically unchanged, so "no gauge" is not a
distinct control there; the gauge matters for cells centred off the
half-period, and the gate catches a wrong one.

## S4.  Cost: the factorisation counted in gemms, and the predicted speedup

The reduction's overhead is: the structural test (`O(n^2)`, row-blocked so no
second `4N x 4N` array is ever allocated), forming `X` and `Y` (`O(n^2)` --
`U` is a real orthogonal signed pairing, and the recentering gauge is folded
into its coefficients so no gauge copy of `G` is made either), ONE `2N x 2N`
gemm, the `2N` eig, and the `O(n^2)` expansion of `4N` eigenvectors.

Measured on the SHIPPED `_generator_block_eig` (so the structural test is
inside `factored total`), two independent clean runs:

| `n_orders` | `4N` | dense zgeev | basis (`X`,`Y`) | `X@Y` gemm | eig(`2N`) | factored total | eig speedup |
|---|---|---|---|---|---|---|---|
| 3 | 196 | 0.0311 / 0.0314 s | 0.00047 s | 0.00012 s | 0.0067 s | 0.0101 / 0.0099 s | **3.1x** |
| 4 | 324 | 0.1097 / 0.1068 s | 0.00106 s | 0.00052 s | 0.0158 s | 0.0243 / 0.0239 s | **4.5x** |
| 5 | 484 | 0.3104 / 0.2975 s | 0.0105 s | 0.0018 s | 0.0415 s | 0.0725 / 0.0662 s | **4.3 - 4.5x** |
| 6 | 676 | 0.7358 / 0.6910 s | 0.0225 s | 0.0065 s | 0.0986 s | 0.1637 / 0.1466 s | **4.5 - 4.7x** |

The flop-count ideal is 8x (`(4N)^3 / (2N)^3`); the realised 4.3-4.7x is the
`O(n^2)` overheads plus the `2N` eig's worse constant.  At `n_orders`=3 the
fixed `O(n^2)` work (structural test + eigenvector expansion) is a fifth of the
factored path, which is why the smallest case only reaches 3.1x -- and it is
exactly the case with the lowest ceiling anyway.  Composed with the measured
share, the prediction is `1/(1 - f + f rho)` with `rho = t_fac/t_dense` ~= 0.31
(`n_orders`=3) and 0.22 (4, 5): **1.7 - 1.8x** at `n_orders`=3 and **1.9 -
2.3x** at 4 and 5, which is what S5 then measures end to end.

**A runtime eigenpair-residual check was considered and rejected.**  It would
cost one `4N` gemm -- 8x the measured `2N` gemm above, so ~0.004 s against a
0.024 s factored path at `n_orders`=4, i.e. +17%, taking `rho` from 0.22 to
0.26 and the whole-solve speedup from ~2.1x to ~1.95x.  The
conditional part of this reduction is the STRUCTURE, and that is verified
exactly and for free; the remaining exposure -- a defective operator making
`eig` return a rank-deficient basis -- is precisely the exposure the shipped
in-plane `eig(P Q)` path has carried since v5.14 and is not made worse here.
The residual claim is asserted in the tests instead, against the dense solve's
own backward error.

## S5.  Interleaved A/B through the public entry points

Per the eig-recycle doc's S5.1 instrument: the two arms are SEPARATE
subprocesses, alternating, round-robin, min over 4 rounds; same worktree, same
process shape, differing only in the `symmetry` flag (`False` = the dense
5.38.1 path, verified byte-identical to the 5.38.1 mount -- see S7).

| case | dense | factored | speedup | per-round dense / factored |
|---|---|---|---|---|
| `pmm_jones_2d`, `n_orders`=3 | 0.0478 s | 0.0297 s | **1.61x** | [.0512 .0484 .0482 .0478] / [.0300 .0297 .0317 .0300] |
| `pmm_jones_2d`, `n_orders`=4 | 0.1585 s | 0.0822 s | **1.93x** | [.1585 .1610 .1594 .1688] / [.0830 .0822 .0831 .0840] |
| `pmm_jones_2d`, `n_orders`=5 | 0.4358 s | 0.2192 s | **1.99x** | [.4453 .4432 .4358 .4430] / [.2200 .2213 .2192 .2204] |
| 6-layer OOP stack, `n_orders`=3 | 0.2608 s | 0.1349 s | **1.93x** | [.2715 .2627 .2608 .2617] / [.1360 .1359 .1378 .1349] |
| 6-layer OOP stack, `n_orders`=4 | 0.8821 s | 0.3756 s | **2.35x** | [.9898 1.0293 .8821 1.0145] / [.4218 .3910 .3756 .3952] |

The whole table is min-over-rounds on both arms, which is the conservative
reading of the ratio wherever the dense arm is the noisier one (the
`stack6 n_orders`=4 dense arm spreads 0.88-1.03 s; taking medians instead would
report 2.6x rather than 2.35x).  An INDEPENDENT earlier run of the identical
driver, before the last two code edits, gave 1.61 / 1.95 / 1.96 / 1.91 / 2.23x
on the same five cases -- so the result reproduces across runs to within
about 5%.  The `n_orders`=3 single-layer case is the weakest (1.61x) because
the fixed non-eig cost is largest there, exactly what S4's `rho` says.

## S6.  Ship criteria, scored

| criterion | required | measured | |
|---|---|---|---|
| whole-OOP-solve speedup, interleaved | >= 1.50x | **1.61x .. 2.35x** across five workloads, reproduced across two independent runs | **PASS** |
| accuracy: exact-or-refuse | yes | structure verified on the assembled generator every call; refusal is the dense path BIT-FOR-BIT | **PASS** |
| eigenpair residual | <= dense's own, x documented decades | 8.5e-16 .. 3.0e-15 vs dense 2.7e-15 .. 4.5e-15 -- BETTER, ratio 0.32 .. 0.86 | **PASS** |
| flux selector classification | unchanged | exactly `2N` forward both paths, forward spectra match 7.6e-15 .. 2.1e-14 | **PASS** |
| end-to-end vs dense | within a derived bar | 8.4e-15 .. 2.8e-14 relative, bar `1e3 * 4Nf * eps` = 4.4e-11 | **PASS** |
| gate refuses where the structure is absent | proven, engineered | oblique + off-centre-pillar arms, both refuse and both return the dense answer bit-for-bit | **PASS** |
| dense path unchanged | byte-identical | SHA256 of (R, T, Jones) over three cases identical to the 5.38.1 mount | **PASS** |

**Shipped.**  What went in, and how it is gated, is S7 below.

## S7.  What shipped

| file | change |
|---|---|
| `lumenairy/elements/rcwa/_core.py` | NEW `_oop_block_gauge`, `_generator_block_eig`, `_OOP_BLOCK_TOL`, `_OOP_GAM_FLOOR`; `_generator_modes` and `_layer_eigenmodes_tensor` take an optional `sym_gauge` |
| `lumenairy/elements/pmm/twod_jones.py` | `_tensor_layer_modes` takes `block_eig=` (default OFF) and offers the gauge for out-of-plane / slanted tensor layers |
| `lumenairy/elements/pmm/stack2d.py` | passes `block_eig=self.symmetry` through to the tensor layer build |
| `tests/unit/test_pmm2d_oop_block_eig.py` | NEW -- 11 tests |
| `tests/unit/test_v5_20_0_pmm_rcwa_upgrades.py` | ONE contract restated -- see S7.5 |
| `docs/audits/EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md` | NEW -- this document |
| `CHANGELOG.md` | `[Unreleased]` entry |

### S7.1  Gating and fallback

The accelerator is offered only when `symmetry` is on (the default `'auto'`;
`symmetry=False` is the documented escape hatch and forces the dense path,
which is what the A/B above measures against) and the layer is out-of-plane or
slanted.  It is also opt-in PER CALL SITE (`_tensor_layer_modes(block_eig=)`,
default off), so only the two entry points validated end to end here --
`pmm_jones_2d` and `PMM2DStackHybrid` -- take it; the native-conical and 1-D
stack callers reach the same function with a degenerate order table and stay
byte-identical (S8).  `_oop_block_gauge` then supplies `(flip, d)`; `flip` is
`None` at oblique incidence, which is a free necessary-condition gate.
Everything else is decided by `_generator_block_eig` on the assembled
generator:

* `max |R Gr R + Gr| <= _OOP_BLOCK_TOL * max|G|` -- **1e-10**, measured
  2026-08-17 in exactly that metric: structure-carrying cells 0.0 .. 3.6e-15,
  structure-violating cells 1.8e-02 .. 3.0e-01.  4.4 decades of gap below,
  7.2 above.  Same bar `_flip_invariant` / `_symmetric_cascade_rt` use for the
  adjacent precondition.
* `min|gam| > _OOP_GAM_FLOOR * max|gam|` -- **1e-13**; the reconstruction
  divides by `gam`.  Measured `min|gam|/max|gam|` over the fixtures:
  6.7e-02 .. 1.2e-01 (the solvers nudge off Wood anomalies, which is what
  would otherwise drive it to zero), so this fires only on a genuinely null
  mode.
* a gauge that is not a usable phase (non-finite or zero), and a non-finite or
  zero-norm reconstruction.

Any of them returns `None` and the dense `zgeev` runs on the untouched `G`.
CuPy and JAX never receive a gauge (`backend_name(xp) == "numpy"` is required),
so the differentiable and GPU paths are byte-for-byte unchanged -- which
matters here: the JAX generator branch is deliberately value-independent so
that forward and gradient walk the same algorithm, and a value-dependent
structural test would break that.

### S7.2  Cache and cascade compatibility

The reduction changes the ALGORITHM, not the inputs: `PMM2DStackHybrid`'s
`_geom_cache` and `_eig_cache` keys are untouched, and a cached re-solve
reproduces its own result exactly (asserted).  Downstream, mode ORDER within
the forward/backward sets differs from `zgeev`'s, which the generalized
S-matrix is invariant to (the modal permutation cancels between `_modes_to_M`
and the propagation diagonal); measured end-to-end agreement on a 4-layer
mixed stack is 1.2e-14.

### S7.3  Memory

Peak transients are comparable to the dense path and the reduction never
allocates a second `4N x 4N`: the structural test is row-blocked at 256 rows,
the gauge is folded into the basis coefficients rather than applied to `G`,
and the largest new arrays are `X`, `Y`, `X@Y` and `w` at `2N x 2N` each
(four quarter-size arrays = one `4N x 4N` equivalent, all freed before the
`4N` eigenvector matrix is built).  Against `zgeev`, which needs its own
internal copy of `G` plus the `4N x 4N` eigenvector output, the factored path
is if anything lighter.

### S7.4  Tests

`tests/unit/test_pmm2d_oop_block_eig.py`, 11 tests, every bar derived at
runtime (`1e3 * 4Nf * eps`, or `1e3 x` the dense solve's own residual on the
same matrix) per `docs/TESTING_STANDARDS.md`:

* engagement is asserted as a DECISION before any agreement is checked, so no
  arm can pass vacuously by refusing (the S3-shape trap);
* the gate is asserted TWO-SIDED: engaged on four structure-carrying cells
  (uniform tilted uniaxial, patterned tilted uniaxial, general biaxial
  non-reciprocal lossy, in-plane slanted), refused on two ENGINEERED
  violations (oblique incidence; a centro-symmetric permittivity on a
  deliberately unmirrored wall layout) -- and a refusal is asserted
  BIT-IDENTICAL to `symmetry=False`, which is exact rather than a tolerance
  because a refusal literally runs the same code;
* the bar's own gap is measured THROUGH the shipped gate by walking a
  tolerance ladder (`_OOP_BLOCK_TOL` is read at call time rather than bound as
  a default argument, precisely so a test can): a carrying cell still engages
  at `default / 1e2`, a violating cell is still refused at `default * 1e3`,
  and that same violating cell DOES engage once the tolerance is opened to
  1.0 -- which proves the refusal is this structural test and not some other
  precondition quietly returning `None`;
* the eigensolve-level residual and the flux-selector classification are
  asserted against the dense solve run on the very same captured generator.

### S7.5  The one existing contract that changed

`test_jones_2d_symmetry_falls_back_offplane` asserted that for an out-of-plane
tensor cell `symmetry=True` is BYTE-IDENTICAL to `symmetry=False`, on the
grounds that such a cell cannot fold.  The premise is still true and is now
asserted directly (the even-sector cascade is never entered -- a decision, not
a byte comparison); the conclusion is not, by design, because `symmetry=True`
now takes the block reduction instead of doing nothing.  Restated as
`test_jones_2d_symmetry_never_folds_offplane`: the fold claim unchanged, plus
agreement within the same derived `1e3 * 4Nf * eps` bar (1.1e-10 at that test's
`n_orders`=5; measured 7.3e-14).  This was the ONLY existing assertion in the
suite that the change invalidated.  Twenty modules -- every one that touches
`_layer_eigenmodes_tensor`, `_generator_modes`, `pmm_jones_2d` or
`PMM2DStackHybrid`, including the unwired conical / 1-D-stack callers -- run
**394 passed, 0 failed** on the final code (12 min, 1 BLAS thread); this test
was the single failure before it was restated.

## S8.  What this leaves open

1. **Three more callers of the same generator, all unwired on purpose.**
   (a) `twod._layer_modes_projected` -- the scalar SLANTED 2-D path, reached
   by slanted scalar layers of `PMM2DStackHybrid` -- builds the same `4N`
   generator, and its convection term is J-odd for the same reason, so it
   carries the same structure; it is one optional keyword and one call site
   away.  (b) `pmm/conical.py` (native conical full-tensor) and (c)
   `pmm/stack.py` (1-D stack with tensor layers) both reach
   `_tensor_layer_modes` with a DEGENERATE order table (`oy = [0]`), for which
   the flip and the recentering read-off are untested.  All three would need
   their own two-sided gate tests, so `block_eig` defaults OFF and they are
   left byte-identical.  The slanted TENSOR path IS covered (it shares the
   wired `_tensor_layer_modes` call sites).
2. **Oblique incidence remains at 1.00x** and provably so (S2.1(b)): the
   extraordinary shift is linear in the harmonic wavevector, and at oblique
   incidence the `+m`/`-m` shifts no longer cancel.  Nothing built on this
   structure can help there; a different lever would be needed.
3. **Off-centre cells** refuse today because the WALL layout, not the
   permittivity, breaks the symmetry.  Rolling the pixel grid to the inferred
   centre before meshing would recover them; unmeasured, and it changes the
   discretisation, so it is a separate experiment rather than a tweak.
4. **The `4Nf` eig is now roughly a third of the OOP solve rather than two
   thirds.**  The next
   marginal gain on this path is the generalized cascade
   (`_interface_smatrix_general` / `_propagation_smatrix_general`), which the
   5.36.0 fast-cascade work has not been applied to.
