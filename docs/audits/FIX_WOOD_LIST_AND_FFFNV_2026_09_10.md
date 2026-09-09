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
  which fails on this Windows box and passes on CI Linux (the same build doc's
  open item 11).

Every measurement below was taken with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
MKL_NUM_THREADS=1` unless a row says otherwise, and every probe asserts
`lumenairy.__file__` for its arm.  The PRE-FIX arm is the read-only main clone
at `D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`
(HEAD `fb3fd93`, i.e. the same commit this branch starts from).

Builds used:

| arm | interpreter | numpy / BLAS |
|---|---|---|
| Windows | py3.14.6 (MSC v.1944) | numpy 2.4.4, OpenBLAS |
| WSL | `~/lumvenv/bin/python` py3.12 | numpy 2.5.1 (the Linux build class) |

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
