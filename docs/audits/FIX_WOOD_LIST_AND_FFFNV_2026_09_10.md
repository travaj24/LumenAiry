# FIX 2026-09-10 -- one Wood-anomaly eps list (Task G) + the fff_nv 1-D ladder (Task H)

Branch `fix/wood-list-fffnv`, worktree `C:/tmp/lum_wood`, off `main`
`fb3fd93` (= 5.43.0 + the `LUMENAIRY_DISABLE_JAX` backend commit).
Binding: `docs/TESTING_STANDARDS.md`.

Two independent, bounded tasks:

* **G** -- unify the Wood-anomaly permittivity list between the SCALAR and
  TENSOR paths of the pure staggered 2-D PMM
  (`docs/audits/VERIFY_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` limitation 2
  = `BUILD_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` open item 6).
* **H** -- harden
  `tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_stripe_reduces_to_rigorous_1d`,
  which fails on this Windows box and was reported to pass on CI Linux (the
  same build doc's open item 11).  The second half of that premise did not
  survive re-measurement -- see H.2.

Every measurement below was taken with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=1` unless a row says otherwise, and every probe asserts
`lumenairy.__file__` for its arm.  The PRE-FIX arm is the read-only main clone
at `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`
(HEAD `fb3fd93`, i.e. the same commit this branch starts from).

Builds used:

| arm | interpreter | numpy / BLAS |
|---|---|---|
| Windows | py3.14.6 (MSC v.1944) | numpy 2.4.4, OpenBLAS |
| WSL | `~/lumvenv/bin/python` py3.12.3 | numpy 2.4.6, scipy-openblas 0.3.31 (the Linux build class) |

---

# TASK G -- one Wood-anomaly permittivity list

## G.1 The defect, measured

`_grazing_safe_wavelength` (`lumenairy/elements/rcwa/_core.py`) nudges the
wavelength by a relative `+1e-7` when a diffraction order sits EXACTLY at
`kz = 0` in any medium whose real permittivity is on the list it is handed.
Its trigger band is `closest(wl) = min_e min_orders |Re(e) - kt^2| <= 1e-9`.

Before this change:

| site | list |
|---|---|
| `pmm_efficiency_2d_staggered` (scalar single layer) | `[eps_sup, eps_sub]` |
| `PMM2DStackPure.solve`, scalar layer | `[eps_sup, eps_sub]` |
| `PMM2DStackPure.solve`, tensor layer | `[eps_sup, eps_sub] + tensor diagonals` |

so a scalar `(Nx, Ny)` cell and its `e * I` promotion -- the same
discretization everywhere else -- took DIFFERENT nudges when an order sat on a
LAYER's own cut-off.  Measured on the pre-fix clone (probes
`g2_oncut.py` / `g2b_stack.py` / `g2d_uniform2.py`, normal incidence,
`n_sup = 1.0`, `n_sub = 1.5`, `M = 5`, `n_orders = 3`):

| fixture | max abs dR (scalar vs `e*I`) | scalar nudge | tensor nudge |
|---|---|---|---|
| uniform `eps = 4` cell, `px = 0.5 um`, `wl = 1.0 um` (the verify report's reproducer) | **4.591e-08** | none | `+1e-7` rel |
| patterned `{4, 1}` cell, same geometry, `PMM2DStackPure` both arms | **7.561e-09** | none | `+1e-7` rel |
| uniform SCALAR layer `eps = 4` (`add_layer(eps=...)`) | **4.977e-08** | none | `+1e-7` rel |
| any of the three, 1e-9 off the cut-off | 0.0 (bit-identical) | none | none |
| far from any cut-off | 0.0 (bit-identical) | none | none |

## G.2 The change

`lumenairy/elements/pmm/twod_staggered.py`

* NEW `_wood_eps_reals(*eps_arrays)` -- the DISTINCT real permittivities of
  whatever it is handed.  It never sniffs a shape (a scalar `(3, 3)` cell is a
  legal 3x3 segmentation grid, so shape-sniffing would be a silent-wrong
  hazard): callers pass already-scalar quantities or a tensor's principal
  diagonal.
* `pmm_efficiency_2d_staggered` now passes
  `_wood_eps_reals(eps_sup, eps_sub, eps_cell)`.

`lumenairy/elements/pmm/stack2d_pure.py`

* `PMM2DStackPure.solve` collects EVERY region: half-spaces, `kind="uniform"`
  scalar layers, `kind="uniform_tensor"` diagonals, patterned scalar cells and
  patterned tensor diagonals, then `_wood_eps_reals(*...)`.

Rationale for unifying UPWARD (adding the layer to the scalar path) rather
than downward: the staggered solver degrades like `~1/sqrt(distance)` near a
cut-off INSIDE a layer exactly as near a half-space one, and an exactly
grazing layer mode is what crashes the interface S-matrix -- the reason the
guard's own docstring gives for listing the layer at all.

The cut-off WARNING was deliberately NOT changed: it keys on the half-spaces,
which are the media whose orders a caller sees in the far field.

## G.3 Gate 1 -- bit-identity off a cut-off, vs the pre-fix clone

`g1_hash.py` hashes (sha256 of the raw float64 bytes) `R` and `T` for 11
fixtures and records every `_grazing_safe_wavelength` call's input and output.
Run on this branch (`PYTHONPATH=/c/tmp/lum_wood`) and on the main clone
(`PYTHONPATH=D:/...` , `lumenairy.__file__` asserted to start with `D:/`),
same interpreter, same thread pins.  `diff` of the two JSON outputs is EMPTY:

| fixture | sha256[:24] R | sha256[:24] T |
|---|---|---|
| `eff_te_normal` (pillar, TE, normal) | `70ca69668996086ddd50dcb0` | `06e24ee4be5eeeda77aad0ab` |
| `eff_tm_oblique` (TM, theta 0.2) | `16ceaf1f70c919ac1527bd8f` | `c7376deb5098cc50009d69bb` |
| `eff_conical` (theta 0.25, phi 0.4) | `b78fe9df4abb28bd09d982bb` | `09452f2112fb00d22502a4a8` |
| `eff_lossy` (Im eps > 0, TM, theta 0.1) | `de8405a97cba87db009f1f4e` | `a1121a3c85445c21bd2018b5` |
| `eff_uniform_cell` | `7ef676f9cdcfda3b7a6d4c30` | `fe5ac886e8613dbcd7937861` |
| `stack_u_p` (uniform + patterned, `jones=False`) | `bffd1145061de54f93a3f48e` | `92539cdd17c557ab90a9a266` |
| `stack_ab_oblique` (A\|B, theta 0.15, phi 0.3) | `ce7ded8a5b9390f60694fbdc` | `dfe7ecf8d054936c5979ca62` |
| `stack_scalar_jones` | `b991bfcb9623ddf9c753a553` | `0dfa7b2617180afb41b69b6e` |
| `stack_lossy_conical` (theta 0.22, phi 0.9) | `9d7bb632cf5e492c2fce5439` | `2f59b75540bf25000a4ca627` |
| `jones_tensor_ctrl` (in-plane tensor -- the TENSOR path control) | `4331f5f503ee7ae3655cfe5f` | `bf7fd92afd49ba15a1df0008` |
| `jones_promoted_pillar` (`e*I` of the scalar pillar) | `b991bfcb9623ddf9c753a553` | `0dfa7b2617180afb41b69b6e` |

All 11 hashes identical pre and post, and all 11 `_grazing_safe_wavelength`
calls returned their input UNCHANGED on both arms
(`6.33000000000000017e-07 -> 6.33000000000000017e-07`), i.e. the fix moved no
nudge off a coincidence.  Note rows 8 and 11 are equal to each other: that IS
the promotion identity, already exact off a cut-off.

## G.4 Gate 2 -- the on-cut-off fixture, fail-before and after

Post-fix, same probes as G.1:

| fixture | pre-fix max abs dR | post-fix |
|---|---|---|
| uniform `eps = 4` cell (verify report's reproducer) | 4.591e-08 | **0.0, bit-identical** |
| patterned `{4, 1}`, `PMM2DStackPure` both arms | 7.561e-09 | **0.0, bit-identical** |
| uniform SCALAR layer `eps = 4` vs its `(3,3)` promotion | 4.977e-08 | 1.381e-15 (see note) |

Note: a uniform SCALAR layer rides the shared eps-free geometric eig while a
uniform TENSOR layer takes its own region eig (by design -- module docstring),
so those two are NOT the same arithmetic and bit-identity is not claimed for
them; the residual 1.381e-15 is that eig-route difference and it is present
pre-fix too.  What the fix removes there is the 4.977e-08 NUDGE difference.

Both arms now take the same nudge, measured directly (`g2c/g2d` spy on
`_grazing_safe_wavelength`):

```
uniform SCALAR layer eps=4 ON its own cutoff:
  PRE   scalar arm: list_len=2 wl 9.99999999999999955e-07 -> 9.99999999999999955e-07  rel +0.000e+00
  PRE   tensor arm: list_len=5 wl 9.99999999999999955e-07 -> 1.00000010000000011e-06  rel +1.000e-07
  POST  scalar arm: list_len=4 wl 9.99999999999999955e-07 -> 1.00000010000000011e-06  rel +1.000e-07
  POST  tensor arm: list_len=4 wl 9.99999999999999955e-07 -> 1.00000010000000011e-06  rel +1.000e-07
```

(`list_len` 5 -> 4 is the deduplication; the wavelength is identical.)

**Nudge magnitude**: relative `+1.000e-07` in wavelength, from the guard's
`wl * (1.0 + 1e-7)` single iteration.  Trigger band `|Re(eps) - kt^2| <= 1e-9`,
which at `kt^2 = 4` is a relative wavelength window of `1e-9 / (2*kt^2)`
= 1.25e-10 -- i.e. only an EXACT coincidence is affected, which is why G.3 came
out bit-identical.

**Consequence of the nudge** (that the rule difference is not noise): at the
patterned `{4,1}` fixture, `solve(nudged)` vs `solve(WL_CUT*(1 - 1e-9))` (a
wavelength the guard leaves alone) differ by max abs dR = 7.637e-09 on a solve
whose max |R| is 0.0115, i.e. ~2.9e3x above a 1000-ULP floor.

## G.5 Gate 3 -- the shipped staggered suites, and the warning

`OMP/OPENBLAS/MKL_NUM_THREADS=1`, `-p no:randomly`:

```
tests/unit/test_v5_12_0_pmm2d_staggered.py  test_v5_21_pmm2d_staggered_oblique.py
tests/unit/test_staggered.py                test_audit_p1_staggered_guard.py
tests/unit/test_p2c_pmm2d_stack_cascade.py  test_p2t_pmm2d_tree_cascade.py
tests/unit/test_pmm2d_lossless_closure_two_sided.py
tests/unit/test_audit_s1_3_pmm2d_lossless_tripwire.py
tests/unit/test_v5_14_0_pmm2d_stack.py      test_pmm2d_staggered_anisotropic.py
tests/unit/test_pmm2d_staggered_oop.py
-> 224 passed, 23 warnings in 462.68 s   (Windows)
```

WARNING behaviour, captured fixture by fixture on both arms (`g3_warn.py`) --
IDENTICAL lists pre and post, message text included:

| fixture | warning |
|---|---|
| ordinary 0.633/0.8 | none |
| `wl = px` (m=1 on the SUPERSTRATE cut-off) | 1x "within 2e-07 (kt^2 units) of a Rayleigh cutoff" |
| `wl = 0.99999 px` (inside the 1e-4 band) | 1x "within 2e-05" |
| LAYER-only cut-off (`eps = 4` cell, `wl/px = 2`) | none |
| oblique conical ordinary / stack ordinary / stack on a layer cut-off | none |

## G.6 Tests

`tests/unit/test_pmm2d_staggered_wood_list.py` -- 18 tests, 2.42 s (Windows).
Shape, against `docs/TESTING_STANDARDS.md`:

* The on-cut-off state is **constructed** from the running build's geometry
  (`wl = px * sqrt(eps_layer)` at normal incidence), and the test asserts the
  construction is exact in float64 (`(wl/px)**2 - eps == 0.0`) rather than
  hoping the build lands there.
* The two RULES are compared by calling the library's own guard with the two
  explicit lists -- no monkeypatching, no reference to pre-fix code, no
  recorded number.
* The nudge TARGET is derived by calling the guard, never by pinning its
  internal `1e-7`, so a future change of the constant tracks automatically.
* The core claims are DECISIONS: nudged / not nudged, and `np.array_equal`
  (bit-identity), not a tolerance.
* The one bar (`test_the_layer_cutoff_nudge_is_consequential`) is derived in
  the run as `1e3 * max|R| * eps_machine` and carries its measured signal
  (7.6e-09, ~2.9e3x above it) and date in the docstring.
* The warning test is two-sided (silent on a layer-only coincidence, exactly
  one warning on a half-space one).

**Fail-before, executed.**  With the two call sites reverted to the pre-fix
list (helper left in place) the file reports `5 failed, 13 passed`: the three
`test_every_scalar_site_takes_the_layer_cutoff_nudge` arms, the
`on the layer cut-off` promotion arm, and
`test_scalar_entry_equals_the_tensor_entry_on_a_layer_cutoff`.  The other 13 --
the off-cut-off invariance battery, the rule comparison, the warning contract,
the helper unit -- pass on BOTH arms, which is what makes them regression
contracts rather than fix-detectors.

## G.7 Side effect: the list is deduplicated

`_grazing_safe_wavelength` evaluates `min(np.min(np.abs(e - kt2)) for e in
eps_reals)` -- a PYTHON-level loop -- per candidate wavelength.  The pre-fix
tensor path pushed `3*Nx*Ny` entries through it.  Measured at a 64x64 cell,
`n_orders = 7`:

| list | time / call | wavelength returned |
|---|---|---|
| raw diagonals (12288 entries) | 46.92 ms | `6.33e-07` |
| deduplicated (2 entries) | 0.02 ms | `6.33e-07` |

Deduplication is numerically inert (the guard takes a MIN), which the test
asserts directly over four wavelengths including the on-cut-off one.

---

# TASK H -- `test_fff_nv_stripe_reduces_to_rigorous_1d`

## H.1 The failure, reproduced and located

```
PYTHONPATH=/c/tmp/lum_wood OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest -q tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py
-> 1 failed, 9 passed, 113.66 s        (Windows py3.14.6 / numpy 2.4.4)
```

The failing assertion is NOT in the test body: it is the `raise
AssertionError` at the end of the helper `_sound_1d_reference`, reached because
its ladder was exhausted --

> no truncation in 11..41 gave the rigorous 1-D solver its own exact lossless
> closure on this build, so there is no sound reference to compare against

The quantity is `|sum R + sum T - 2|` for `rcwa_jones_1d_segments` on
`[(0.5, rot(35 deg, no=1.5, ne=2.3)), (0.5, 2.25*I)]`, `n_sub = 1.5`,
`n_sup = 1.0`, period 0.7 um, wl 1.0 um, depth 0.5 um; the helper needed one
truncation in 11..41 (odd) below `_ONED_SOUND_CLOSURE = 1e-9`.

## H.2 The measurements

Worst / best of that quantity over the helper's own window, and how many of the
16 truncations qualify (probe `h1_ladder.py`, run over the full 5..61 ladder):

| configuration | worst | best | sound (<1e-9) of 16 | test |
|---|---|---|---|---|
| Windows py3.14.6 / numpy 2.4.4, `OPENBLAS_NUM_THREADS=1` | 2.761e-02 | 4.612e-06 | **0** | **FAILS** |
| Windows py3.14.6 / numpy 2.4.4, `OPENBLAS_NUM_THREADS=4` | 2.309e-02 | 2.722e-13 | **1** (n = 35) | **PASSES** |
| WSL py3.12.3 / numpy 2.4.6 (scipy-openblas 0.3.31), 1 thread | 5.001e-02 | 2.061e-06 | **0** | **FAILS** |

**The same box, the same interpreter, the same library, the same fixture:
`OPENBLAS_NUM_THREADS=4` passes and `=1` fails** (verified by running the test
itself under both). That is the definition of a per-build test -- the pass/fail
boundary sits inside the arithmetic spread of the quantity it reads. Also worth
recording: the test does NOT currently pass on WSL either, so the "fails on
Windows, passes on CI Linux" premise is stale -- nothing about the ladder
guarantees a hit on any particular build.

The Jones arm is per-build too, INDEPENDENTLY of the ladder. Against a
converged reference (`n_orders = 81`), on the shipped fixture at `No = 11`:

| configuration | `ef/el` | `jf/jl` | `jf < jl` |
|---|---|---|---|
| Windows, 1 thread | 0.0277 | 0.0200 | yes |
| Windows, 4 threads | 0.0277 | 0.0200 | yes |
| WSL | 0.0999 | **1.1773** | **no** |

## H.3 Verdict: (a) -- the library is sound, the FIXTURE is degenerate

The chain of measurements, each one ruling something out.

1. **It is not the anisotropic solver.** Same call, same geometry, different
   permittivities (`h2_isolate.py`, `h5_where.py`); closure defect at
   `n_orders` 11 / 21 / 31 / 41 / 61:

   | cell | 11 | 21 | 31 | 41 | 61 |
   |---|---|---|---|---|---|
   | rotated 35 deg (the fixture) | 1.05e-02 | -3.07e-03 | 2.15e-03 | -1.00e-04 | 7.74e-07 |
   | same tensor NOT rotated (`exy = 0`) | -1.51e-14 | 1.91e-14 | 6.31e-14 | -2.44e-15 | 1.23e-13 |
   | isotropic | 7.99e-15 | 7.42e-14 | 1.03e-13 | 1.48e-13 | -1.31e-13 |
   | a DIFFERENT symmetric tensor (`exy = 0.5`) | 8.44e-15 | -2.09e-14 | -3.73e-14 | -2.48e-13 | -7.98e-13 |
   | GYROTROPIC Hermitian (`exy = +0.5i`) | 1.07e-14 | -2.84e-14 | -2.95e-14 | 1.49e-13 | 6.93e-14 |
   | the fixture, UNIFORM (no grating) | -4.44e-16 | -4.44e-16 | -4.44e-16 | -4.44e-16 | -4.44e-16 |
   | the fixture, half-spaces matched 1.0/1.0 | -3.35e-14 | -3.38e-13 | 9.19e-14 | 1.15e-13 | 2.84e-12 |

   Off-diagonal anisotropy per se is fine (rows 4 and 5). Only THIS cell, with
   THIS substrate, misbehaves.

2. **It is not a non-Hermitian operator** -- the explanation the file's own
   `_FFF_NV_CLOSURE_ENVELOPE` offered. The Li in-plane factorization
   `[[Cxx, Cxy], [Cyx, Cyy]]` built by `_tensor_convolutions(..., 'li')` is
   Hermitian for a real symmetric tensor: MEASURED `max|C - C^H| / max|C|` =
   6.8e-17..1.7e-16 at `n_orders` 5..61 (`h4_cond.py`). An energy theorem does
   exist there.

3. **It is not eigenproblem conditioning.** Same probe: `cond(W)` = 3.2..7.6 and
   `cond([W; V])` = 18..2521 over `n_orders` 5..61, so `cond * eps_machine` is
   4.1e-15..5.6e-13 -- TEN decades below the 1e-02 defect. Amplification by the
   modal basis cannot produce it.

4. **It IS an exact index coincidence.** The fixture's director has `no = 1.5`,
   the groove is `eps = 2.25`, the substrate is `n = 1.5`:
   `no^2 = eps_groove = n_sub^2 = 2.25`. The ordinary channel of the layer is
   therefore a perfectly uniform medium IDENTICAL to the groove and to the
   substrate, so the layer carries a set of modes EXACTLY degenerate with the
   region's -- `_check_energy`'s own documented "near-degenerate layer<->region
   mode-match at a measure-zero coincidence", except that here it is not near,
   it is exact, and the interface inverse amplifies the rounding floor by ~1e14.

   Detuning ANY ONE of the three by a relative `r` collapses the defect
   (`h6_detune.py`; worst over `n_orders` 7..61 of each row):

   | relative detune `r` | detune the groove | detune `no` | detune `n_sub` |
   |---|---|---|---|
   | 0 | 2.8e-02 | 2.8e-02 | 2.8e-02 |
   | 1e-12 | 2.96e-04 | 8.65e-05 | 7.26e-05 |
   | 1e-09 | 1.30e-07 | 6.45e-08 | 5.48e-08 |
   | 1e-06 | 5.39e-11 | 8.11e-11 | 1.16e-10 |
   | 1e-03 | 3.42e-13 | 6.44e-13 | 2.67e-13 |
   | 1e-02 | 3.95e-13 | 2.35e-14 | 2.44e-13 |

   That is ~`eps_machine / r`, the signature of a division by a vanishing gap,
   with the `r = 0` row's effective gap set by the rounding floor itself --
   which is precisely why its value moves with the LAPACK build and the BLAS
   thread count while the answer's converged part does not.

So the library computes what the formulation prescribes, converges (`sum R` on
the fixture converges to 0.06183 on both builds), DETECTS the degenerate state
(`_EnergyWarning` fires at every poisoned truncation, and the 2-D entry raises
`_EnergyError` outright at `n_orders_x = 9`) and documents the remedy
(`stabilize=True`, or change the geometry). No library defect; nothing in
`rcwa_jones_1d_segments` or the fff_nv 2-D path was changed.

## H.4 What changed in the test

`tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py` only.

1. **The fixture is engineered off the coincidence.** New module constant
   `_STRIPE_EPS_GROOVE = 2.10` -- with the mechanism, the three-configuration
   table and the detuning evidence in its docstring -- replaces the inline
   `2.25` in `test_fff_nv_stripe_reduces_to_rigorous_1d`. On it the rigorous
   1-D closure holds at **16 of 16** truncations in 11..41 on all three
   configurations (worst 2.083e-13 / 1.821e-13 / 1.861e-13).
2. **The build-scanning ladder is gone.** `_sound_1d_reference` (which searched
   for a truncation whose theorem happened to hold) becomes
   `_rigorous_1d_reference`: it solves once at a CONVERGED truncation
   (`_ONED_REF_ORDERS = 81`; derivation in its docstring -- reference residual
   ~5.0e-07 in `sum R`, 61x below the smallest quantity compared against it)
   and ASSERTS the theorem, with a message naming the degeneracy as the thing
   to check. The precondition is forced and gated rather than hoped for
   (TESTING_STANDARDS restatement 4).
3. **The closure bar is derived from what the formulation actually does here.**
   On a y-uniform stripe `fff_nv` reduces to the rigorous (Hermitian) 1-D rule,
   so its closure is machine-exact: measured 1.421e-14 / 2.354e-14 / 5.373e-14
   at `No = 11`, and <= 5.4e-14 over `No` = 9, 11, 13. The test now asserts
   `_ONED_SOUND_CLOSURE` (1e-9) -- 4 decades above the measurement and 4 below
   the 1e-05..1e-04 a degenerate cell gives -- and keeps the library-wide
   `_FFF_NV_CLOSURE_ENVELOPE` assertion behind it. That envelope's comment gets
   an APPENDED dated correction (its three-arm table was a reading of the
   coincidence, not of the formulation), and the out-of-plane test that
   repeated the claim gets a pointer to it. Nothing was rewritten.
4. **The two ratio bars get a gap on both sides.** `ef/el` = 0.0209 and
   `jf/jl` = 0.0182 to FIVE significant figures on all three configurations
   (`ef` = 3.0399e-05, `el` = 1.4527e-03, `jf` = 5.8158e-05, `jl` = 3.2042e-03
   on every one), so the bare `ef < el` was passing on 48x of margin while
   saying nothing about a 10x degradation. Now `< 0.2 *`: 9.6x above the
   measurement, 5x below the 1.0 the claim means.
5. **NEW `test_stripe_fixture_is_free_of_the_mode_match_degeneracy`** -- the
   two-sided replacement for the deleted ladder, and the only test in the file
   that reads the pathology directly. Positive arm: the reference fixture's own
   energy theorem holds at EVERY truncation in 11..41. Negative arm: the
   coincidence is RECONSTRUCTED through the public API (groove set to
   `no^2 = 2.25`) and must still be decades worse -- the claim is the RATIO
   measured in the same run (>= 1e5 asserted, ~1.3e11 measured), floored at
   1e-13 so a lucky clean run cannot inflate the requirement, and the
   `_EnergyWarning` is required to fire on that arm.

## H.5 After

| configuration | result | time |
|---|---|---|
| Windows py3.14.6 / np 2.4.4, 1 BLAS thread | **11 passed** | 190.90 s |
| Windows py3.14.6 / np 2.4.4, 4 BLAS threads | **11 passed** | 94.61 s |
| WSL py3.12.3 / np 2.4.6, 1 BLAS thread | **11 passed** | 181.73 s |

`test_fff_nv_crossed_cell_converges_and_beats_laurent` is 183 / 109 / 175 s of
that and is untouched. `.test_durations` spliced with
`--durations-path .test_durations --store-durations` for the one added test:
12298 -> 12299 entries, 1 added, 0 removed, 0 changed
(`test_stripe_fixture_is_free_of_the_mode_match_degeneracy`, 2.148 s).

## H.6 Open items (NOT done here)

1. **`tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py::test_pmm_fff_nv_stripe_reduces_to_rigorous_1d`
   has the same fixture and the same latent problem.** It uses the identical
   coincident cell (`rot(35 deg, 1.5, 2.3)` against `2.25*I`, `n_sub = 1.5`)
   and carries an even larger apparatus to survive it -- a two-stage
   `_RCWA_REF_STAGES` ladder plus a `_scan` that scores every rung by its own
   closure. It PASSES on this box today (`5 passed, 141.43 s`), so it is
   latent, not failing; the same one-line groove change would delete the
   apparatus and ~46 s of scanning. Out of this task's scope, deliberately not
   touched.
2. **The 1-D anisotropic solver could handle an exact layer<->region mode
   coincidence better.** Today it detects and warns (and the 2-D entry raises),
   which is sound behaviour, but an exactly degenerate mode pair is in
   principle resolvable -- deflate the shared subspace, or match modes by
   invariant subspace rather than through an explicit inverse. Whether that is
   worth doing is a design question, not a defect: reported, not attempted.
3. **`_check_energy`'s message calls this "near-degenerate ... at a measure-zero
   period / n_orders coincidence"**, which invites the reader to change
   `n_orders`. On an exact INDEX coincidence like this fixture's, changing
   `n_orders` does not help (0 of 16 sound truncations at 1 thread); changing
   the geometry does. A sentence naming the index coincidence would have saved
   most of this investigation. Left as a suggestion, no library edit made.
