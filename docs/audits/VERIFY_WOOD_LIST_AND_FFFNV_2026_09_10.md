# VERIFY 2026-09-10 -- adversarial verification of the Wood-anomaly eps list (Task G) and the fff_nv 1-D fixture (Task H)

Independent verification of branch `fix/wood-list-fffnv`
(`f827ee1`, `a811a76`, `7248caa`, `b5b54d7`, off `main` `fb3fd93`), whose
claims are in `docs/audits/FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md`.

Worktree `C:/tmp/lum_vwood`, branch `verify/wood-fffnv`, starting at `b5b54d7`.
Binding: `docs/TESTING_STANDARDS.md`.  Nothing here is read from the builder's
probes: every number below was produced by a probe in
`validation/probe_verify_wood_fffnv/` on fixtures of the verifier's own
construction, except `v3_builder_table.py`, whose entire purpose is to re-run
the builder's OWN geometry so their reported digits can be checked.

Three configurations throughout:

| tag | interpreter | numpy / BLAS | threads |
|---|---|---|---|
| **Win-1** | py3.14.6 (MSC v.1944) | numpy 2.4.4, scipy-openblas | `OMP/OPENBLAS/MKL_NUM_THREADS=1` |
| **Win-4** | the same | the same | `= 4` |
| **WSL** | `~/lumvenv/bin/python` py3.12.3 | numpy 2.4.6, scipy-openblas | `= 1` |

The PRE-FIX arm is the read-only main clone at
`D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`
(HEAD `fb3fd93`).  Every probe takes the lumenairy root its arm must come from
as `argv[1]` and asserts `lumenairy.__file__` against it.

---

## 0. Verdict summary

### Task G -- one Wood-anomaly permittivity list

| # | claim | verdict |
|---|---|---|
| G-a | the pre-fix scalar and tensor paths took DIFFERENT nudges on a LAYER cut-off, with a measurable consequence | **CONFIRMED** |
| G-b | 11 fixtures bit-identical to `fb3fd93` off a cut-off, nudged wavelengths included | **CONFIRMED** and extended -- 14 fixtures of my own, R + T + Jones hashes and every guard call |
| G-c | on a LAYER cut-off the promotion identity is restored (was 4.591e-08 / 7.561e-09, now bit-identical) | **CONFIRMED**, digit for digit on their geometry and on mine |
| G-d | a HALF-SPACE cut-off is unchanged pre/post | **CONFIRMED** |
| G-e | nudge = relative `+1e-7`, trigger band `\|Re eps - kt^2\| <= 1e-9` | **CONFIRMED** |
| G-f | the cut-off WARNING is unchanged | **BOUNDED** -- identical on 12 fixtures both arms, but the nudge moves the 1e-4 boundary by ~8e-7 in `kt^2` units on a geometry that also sits on a layer cut-off; flips measured in BOTH directions |
| G-g | uniform scalar layer vs uniform tensor layer differ by a pre-existing eig-route residual, not by the fix | **CONFIRMED** (my reading of that residual is 4.2188e-15, IDENTICAL on both arms) |
| G-h | G.1's uniform-SCALAR-layer row: 4.977e-08 pre / 1.381e-15 post | **BOUNDED** -- I reconstruct 4.5908e-08 / 1.3045e-15 through the public API; same class, their exact digits not reproduced |
| G-i | deduplication is inert and turns 46.92 ms/call into 0.02 ms | **CONFIRMED** (45.5 -> 0.031 ms; 47.3 -> 0.029 ms on the pre-fix arm) |
| G-j | `test_the_layer_cutoff_nudge_is_consequential`'s stated floor (2.6e-12) and ratio (~2.9e3x) | **REFUTED** -- the code's floor is `1e3 * max\|R\| * eps` = 2.5572e-15 and the measured ratio is 2.99e6.  Docstring arithmetic is wrong by 1e3; the ASSERTION is unaffected.  FIXED in the test docstring, commit `e2ffa97` (see 5.1) |

### Task H -- the fff_nv stripe fixture

| # | claim | verdict |
|---|---|---|
| H-a | the shipped test fails at 1 BLAS thread and PASSES at 4, same box, same build | **CONFIRMED** (executed) |
| H-b | it does not pass on WSL either -- the "passes on CI Linux" premise is stale | **CONFIRMED** (executed) |
| H-c | ladder table: worst / best / sound-of-16 = 2.761e-02 / 4.612e-06 / 0, 2.309e-02 / 2.722e-13 / 1, 5.001e-02 / 2.061e-06 / 0 | **CONFIRMED** to every printed digit |
| H-d | the Jones arm is per-build independently of the ladder: `jf/jl` 0.0200 / 0.0200 / **1.1773** | **CONFIRMED** to every printed digit |
| H-d2 | "it is not the anisotropic solver": the same call with different permittivities is clean at every truncation | **CONFIRMED** -- 7 control cells read 1e-16..1e-12 where the fixture reads 1e-02..1e-07 (section 7.1) |
| H-e | the Li in-plane operator is Hermitian, so an energy theorem exists | **CONFIRMED** (`max\|C - C^H\|/max\|C\|` = 3.5e-17..1.4e-16, M 5..61, both grooves) |
| H-f | eigenproblem conditioning cannot amplify: `cond(W)` 3.2..7.6, `cond([W;V])` 18..2521 | **CONCLUSION CONFIRMED, NUMBERS REFUTED** -- I measure `cond(W)` 3.19..11.47 and `cond([W;V])` 6.8..107.7.  `cond * eps` <= 2.4e-14 either way, twelve decades under the defect |
| H-g | it IS an EXACT layer<->region mode coincidence, amplified by the interface inverse | **CONFIRMED and SHARPENED** -- 10/21/41/61/82/122 layer modes match a region mode to <1e-12 at M = 5/11/21/31/41/61 on the coincident groove and **0** on the detuned one; `cond(a+b)` at the layer->substrate interface 2.7e12..7.4e15 vs 7..165 |
| H-h | detuning any of the three collapses the defect like `eps_machine / r` | **CONFIRMED** (own table, Win-1 and WSL) |
| H-i | the detuned fixture is sound at 16 of 16 truncations on all three configurations (worst 2.083e-13 / 1.821e-13 / 1.861e-13) | **CONFIRMED** to every printed digit |
| H-j | the four compared quantities are build-free to five significant figures | **CONFIRMED and STRENGTHENED** -- `ef`, `el`, `jf`, `jl` and both ratios are identical to all SEVEN printed digits on Win-1, Win-4 and WSL |
| H-k | the new two-sided degeneracy test fails on the coincident fixture and passes on the detuned one | **CONFIRMED** on all three configurations (fail-before executed) |
| H-l | `.test_durations` spliced 12298 -> 12299, 1 added | **CONFIRMED** by inspection |

**No library defect was found in either fix.**  Both changes do what they claim.
Four documentation / reporting defects and two claims needing a qualifier
were found; they are listed in section 11.

---

## 1. Task G -- bit-identity off a cut-off (G.a)

`v1_bitid.py`: FOURTEEN fixtures of my own, none of them the builder's.  For
each it hashes (sha256 of the raw float64 bytes) `R`, `T` and, where produced,
the order-0 Jones, and records EVERY `_grazing_safe_wavelength` call the solve
made -- input wavelength, returned wavelength, and the length of the
permittivity list it was handed.  Run on this branch and on the pre-fix clone,
same interpreter, same thread pins.

| fixture | entry point | sha256[:24] R | sha256[:24] T | sha256[:24] J | `n_eps` pre -> post |
|---|---|---|---|---|---|
| `f01_eff_te_normal` (3x3 L-pillar, TE, normal) | `pmm_efficiency_2d_staggered` | `e4fe0371b726108d0a894cb0` | `60bdd85667b455eeb60ea954` | -- | 2 -> 4 |
| `f02_eff_tm_oblique` (TM, theta 0.27) | same | `494afdabd4efe8c3c2fd927f` | `27176e0daaeba4f490ff5970` | -- | 2 -> 4 |
| `f03_eff_conical` (theta 0.18, phi 0.77) | same | `9b92c7f029cbbcdec1c7f729` | `5d891a8d5c5875e887b98f4c` | -- | 2 -> 4 |
| `f04_eff_lossy_tm` (Im eps > 0) | same | `0a02d2da81ee5c8030731117` | `714c9194b7ecd54219ff11ce` | -- | 2 -> 4 |
| `f05_eff_rect_periods` (px 0.62 / py 0.47) | same | `729117db5cb38e851abcd2ff` | `c0c109e21346484121ba143d` | -- | 2 -> 4 |
| `f06_jones_promoted_cell3` (`e*I` of f01's cell) | `pmm_jones_2d_staggered` | `015e6827ff5e1777d339352e` | `d32cc7115b3b3b32d061bf86` | `6c89236fe8a2c4cdc0ca7918` | 29 -> 4 |
| `f07_jones_tensor_rot40` (rotated director, TENSOR control) | same | `789ce687b1ee4410705ec6a7` | `c7d1a2c56a179e08c60a7fd0` | `fa2407c242580dcf6f9a0722` | 14 -> 6 |
| `f08_jones_promoted_oblique` (theta 0.21, phi 0.44) | same | `d5317cf035add8a4752b4039` | `e7423a42b46351a3f45b191c` | `5ebb6de2146a0076720bff31` | 14 -> 4 |
| `f09_stack_uniform_patterned` (`jones=False`) | `PMM2DStackPure` | `e1a4c38d646337d4eebbc3c4` | `19a4397d3b5f0bc22aaf5255` | -- | 2 -> 4 |
| `f10_stack_ab_oblique` (A\|B, theta 0.20, phi 0.50) | same | `db38d1e1211402bbfd4196d2` | `30ce06a2185b90c4538aee63` | `72c00405a0c6bdf02bd75a74` | 2 -> 5 |
| `f11_stack_uniform_tensor_mix` | same | `f1c34c9b5b543f6543e86d9f` | `c900e2a4c000192d897219c4` | `8cc18e75d4a6f7805df81025` | 5 -> 7 |
| `f12_stack_lossy_conical` (theta 0.23, phi 0.91) | same | `0d33c0cc3d7e848f8a05b872` | `8667136a2e142bdedbaddcc1` | `859c19ee365922dd88f62820` | 2 -> 5 |
| `f13_stack_scalar_jones` | same | `015e6827ff5e1777d339352e` | `d32cc7115b3b3b32d061bf86` | `6c89236fe8a2c4cdc0ca7918` | 2 -> 4 |
| `f14_stack_promoted_jones` | same | `015e6827ff5e1777d339352e` | `d32cc7115b3b3b32d061bf86` | `6c89236fe8a2c4cdc0ca7918` | 29 -> 4 |

* **0 hash mismatches of 36** (14 R + 14 T + 8 Jones) between the two arms.
* **0 wavelength mismatches**: all 14 guard calls returned their input
  unchanged, `8.5e-07 -> 8.5e-07`, on BOTH arms.  Only `n_eps` moves.
* f06 = f13 = f14 on both arms: the promotion identity holds through TWO
  different entry points (`pmm_jones_2d_staggered` delegates to
  `PMM2DStackPure`) and for a scalar cell against its `e*I`.

**G-b CONFIRMED**, on a fixture set the builder did not use.

## 2. Task G -- the on-cut-off state (G.b, G.c, G.d, G.e, G.g)

`v2_oncut.py`.  The cut-off is CONSTRUCTED through the public API and the
construction is asserted exact in float64:

```
px = 5.4e-07   eps_layer = 6.25
wl_layer = px * sqrt(eps_layer) = 1.35e-06     (wl/px)**2 - eps      = 0.0
eps_sub  = 1.45**2 = 2.1025
wl_half  = px * sqrt(eps_sub)  = 7.83e-07      (wl/px)**2 - eps_sub  = 0.0
```

`eps_layer = 6.25` is carried by NEITHER half-space, so only the layer can put
order `m = 1` at `kz = 0`; `eps_sub = 2.1025` is carried by no cell value.  The
fixture therefore isolates the two cases cleanly.  `PMM2DStackPure`, scalar
`(3,3)` cell vs its `e*I` promotion, `M = 5`, `n_orders = 3`, normal incidence:

| wavelength | arm | max abs dR | max abs dT | max abs dJ | bit-identical |
|---|---|---|---|---|---|
| **ON the LAYER cut-off** | PRE | **1.3059e-10** | 1.3060e-10 | **6.2158e-09** | no |
| | POST | **0.0** | 0.0 | 0.0 | **yes** |
| 1e-9 relative below it | PRE | 0.0 | 0.0 | 0.0 | yes |
| | POST | 0.0 | 0.0 | 0.0 | yes |
| far from any cut-off (0.81 um) | PRE | 0.0 | 0.0 | 0.0 | yes |
| | POST | 0.0 | 0.0 | 0.0 | yes |
| **ON the SUBSTRATE cut-off** | PRE | 0.0 | 0.0 | 0.0 | yes |
| | POST | 0.0 | 0.0 | 0.0 | yes |

and the guard calls behind those rows:

```
ON the LAYER cut-off   PRE   scalar  n_eps= 2  1.35e-06 -> 1.35e-06                  rel +0.000e+00
                       PRE   e*I     n_eps=29  1.35e-06 -> 1.3500001350000002e-06    rel +1.000e-07
                       POST  scalar  n_eps= 3  1.35e-06 -> 1.3500001350000002e-06    rel +1.000e-07
                       POST  e*I     n_eps= 3  1.35e-06 -> 1.3500001350000002e-06    rel +1.000e-07
ON the SUBSTRATE cut-off  both arms, both paths: 7.83e-07 -> 7.830000783e-07         rel +1.000e-07
```

The two ENTRY POINTS on the same layer cut-off
(`pmm_efficiency_2d_staggered` TM vs `pmm_jones_2d_staggered` on the promotion,
uniform `eps = 6.25` cell): PRE `max abs dR = 1.3059e-10`, not bit-identical;
POST `0.0`, bit-identical.

**G-c, G-d CONFIRMED.**  Note the magnitude of the pre-fix disagreement is
fixture-dependent -- 1.3e-10 in R and 6.2e-09 in the Jones on mine, 4.6e-08 on
the builder's -- so "the 1e-8 class" is the class of the defect, not a constant.

### The guard itself (G-e)

```
on cut-off:  1.35e-06 -> 1.3500001350000002e-06   rel 1.000000e-07
             out == wl * (1.0 + 1e-7) exactly:  True
```

Trigger window, scanned as `wl = WL_CUT * (1 + r)`; the guard's own
`min |Re eps - kt^2|` is printed beside the decision:

| r | `min\|eps - kt^2\|` | nudged? |
|---|---|---|
| 0 | 0.0 | yes |
| +1e-13 | 1.2479e-12 | yes |
| +1e-11 | 1.2500e-10 | yes |
| +2e-11 | 2.5000e-10 | yes |
| +1e-10 | 1.2500e-09 | **no** |
| +5e-10 | 6.2500e-09 | no |
| -1e-10 | 1.2500e-09 | **no** |
| -5e-10 | 6.2500e-09 | no |

i.e. the boundary is exactly `1e-9` in `|Re eps - kt^2|`, which at `kt^2 = 6.25`
is a relative wavelength window of `1e-9 / (2 kt^2)` = 8.0e-11.  IDENTICAL on
both arms.  **G-e CONFIRMED.**

### Uniform scalar layer vs uniform tensor layer (G-g)

`add_layer(eps=6.25)` against `add_layer(eps=6.25*I)`:

| wavelength | arm | max abs dR | max abs dJ |
|---|---|---|---|
| ON the layer cut-off | PRE | 9.6505e-08 | 1.3245e-07 |
| | POST | 3.1086e-15 | 4.4351e-15 |
| far off | PRE | **4.2188e-15** | 3.9585e-15 |
| | POST | **4.2188e-15** | 3.9585e-15 |

The far-off row is IDENTICAL on the two arms to the last bit, which is the
direct proof that the ~1e-15 residual is the pre-existing eig-route difference
(a `kind="uniform"` scalar layer rides the shared eps-free geometric eig; a
`kind="uniform_tensor"` layer takes its own region eig) and was NOT introduced.
What the fix removes is the 9.65e-08 nudge difference.  **G-g CONFIRMED.**

## 3. Task G -- the builder's own numbers, re-run (G.a, G.h, G.i)

`v3_builder_table.py`, their geometry exactly (`px = 0.5 um`, `wl = 1.0 um`,
`depth = 0.28 um`, `n_sup = 1.0`, `n_sub = 1.5`, `M = 5`, `n_orders = 3`,
`(wl/px)**2 - 4 = 0.0`):

| row | claimed PRE | measured PRE | claimed POST | measured POST |
|---|---|---|---|---|
| uniform `eps = 4` cell, eff vs jones-promotion | 4.591e-08 | **4.5908e-08** | 0.0 | **0.0** |
| patterned `{4,1}` cell, stack scalar vs `e*I` | 7.561e-09 | **7.5613e-09** | 0.0 | **0.0** |
| uniform SCALAR layer `eps = 4` vs `4*I` | 4.977e-08 | **4.5908e-08** | 1.381e-15 | **1.3045e-15** |

and two rows that are POST-fix statements only:

| row | claimed | measured POST | (same probe on the PRE arm) |
|---|---|---|---|
| nudge CONSEQUENCE: `solve(nudged)` vs `solve(WL*(1-1e-9))`, patterned `{4,1}` | 7.637e-09 | **7.6369e-09** | 7.5613e-11 -- pre-fix the scalar arm is not nudged, so both wavelengths are un-nudged and the comparison degenerates, which is the point |
| ... with `max\|R\|` | 0.0115 | **0.0115166** | 0.0115166 |
| guard, raw 12290-entry list | 46.92 ms | 45.491 ms | 47.259 ms |
| guard, deduplicated list | 0.02 ms | **0.031 ms** | 0.029 ms |
| deduplicated wavelength | `6.33e-07` | `6.33e-07` | `6.33e-07` |

Every row reproduces except the uniform-SCALAR-layer one, where I read
4.5908e-08 / 1.3045e-15 against their 4.977e-08 / 1.381e-15 -- the same class
and the same conclusion, but not their digits (their `g2d` probe evidently
differs from my reconstruction in some detail I could not identify from the
claim text).  Recorded as **BOUNDED**, not refuted: the CLAIM (the nudge
difference is removed; the residual is the pre-existing eig route) is
independently confirmed in section 2.

**G-a, G-i CONFIRMED.  G-h BOUNDED.**

## 4. Task G -- the cut-off warning (G.f)

`v4_warn.py`, twelve fixtures, warnings captured message by message on both
arms.  The two arms are IDENTICAL -- same count, same text, same printed gap:

| fixture | warnings, PRE | warnings, POST |
|---|---|---|
| eff ordinary (0.8 / 0.633) | 0 | 0 |
| eff conical ordinary (theta 0.25, phi 0.4) | 0 | 0 |
| eff `wl = px` (SUPERSTRATE cut-off) | 1, "within 2e-07" | 1, "within 2e-07" |
| eff `wl = 0.99999 px` (inside the 1e-4 band) | 1, "within 2e-05" | 1, "within 2e-05" |
| eff LAYER-only cut-off | 0 | 0 |
| eff SUBSTRATE cut-off | 1, "within 4.5e-07" | 1, "within 4.5e-07" |
| eff rect periods ordinary | 0 | 0 |
| stack ordinary | 0 | 0 |
| stack on a LAYER cut-off | 0 | 0 |
| stack on the SUBSTRATE cut-off | 1, "within 4.5e-07" | 1, "within 4.5e-07" |
| stack uniform+patterned, uniform on ITS cut-off | 0 | 0 |
| stack oblique ordinary | 0 | 0 |

### ...and the adversarial construction that BOUNDS the claim

The warning keys on `_gap = min over the two HALF-SPACES of |Re eps - kt^2|`,
computed from the NUDGED wavelength, and fires below `1e-4`.  The fix cannot
change which half-spaces are looked at -- but it CAN change `wl`, hence
`kt^2 = 4 (1 + 2e-7)`, hence `_gap`, by ~8e-7 in `kt^2` units, on a geometry
that also sits exactly on a LAYER cut-off.  Put order (1,0) exactly on
`eps_layer = 4` and place `eps_sub` at the boundary:

| `eps_sub` | PRE warnings | POST warnings |
|---|---|---|
| `4 - 9.990e-05` | **1** ("0.0001") | **0** |
| `4 - 9.996e-05` | **1** ("0.0001") | **0** |
| `4 - 1.0000e-04` | 0 | 0 |
| `4 + 9.990e-05` | 1 ("0.0001") | 1 ("9.9e-05") |
| `4 + 1.0000e-04` | **0** | **1** ("9.9e-05") |
| `4 + 1.0004e-04` | **0** | **1** ("9.9e-05") |
| `4 + 1.0010e-04` | **0** | **1** ("9.9e-05") |

So "the warning behaviour is unchanged" is true off the coincidence and on
every ordinary fixture, and FALSE inside a band of width ~`2 * 1e-7 * kt^2`
(0.8 % of the threshold at `kt^2 = 4`) around the `1e-4` warning boundary, on
geometries that ALSO sit exactly on a layer cut-off.  Both directions were
measured: with `eps_sub` BELOW `kt^2` the nudge widens the gap and the warning
turns OFF; with `eps_sub` ABOVE it the nudge narrows the gap and the warning
turns ON.

This is not a defect -- the post-fix warning is computed at the wavelength
actually solved, which is the more correct of the two -- but the claim needs
the qualifier.  **G-f BOUNDED.**

## 5. Task G -- the shipped test file

`tests/unit/test_pmm2d_staggered_wood_list.py`, 18 tests, 2.84 s (Win-1).
Its shape is sound: the on-cut-off state is constructed and asserted exact, the
two RULES are compared by calling the library's own guard with two explicit
lists (no monkeypatching, no pre-fix reference), the nudge target is DERIVED by
calling the guard, and the core claims are decisions (`np.array_equal`).

### 5.1 One defect in it

`test_the_layer_cutoff_nudge_is_consequential` asserts

```python
signal = float(np.max(np.abs(Rn - Rb)))
floor  = 1e3 * float(np.max(np.abs(Rn))) * float(np.finfo(float).eps)
assert signal > 100.0 * floor
```

and its docstring says the floor is "2.6e-12 here" and the signal "~2.9e3x
above that floor".  MEASURED (Win-1, the test's own geometry):

```
signal            = 7.6369e-09
max|Rn|           = 0.0115166
floor (as coded)  = 1e3 * 0.0115166 * 2.220446e-16 = 2.5572e-15
signal / floor    = 2.986e6           (docstring says ~2.9e3)
signal / (100 x floor, the asserted bar) = 2.986e4
```

The docstring's floor is wrong by a factor of 1e3 and its ratio with it; the
two are internally consistent with each other, so this is a transcription
error, not a mis-sized bar.  The assertion itself is fine and in fact has FOUR
decades of margin over the bar and SIX over the floor, not three.
**G-j REFUTED (documentation).**  RESTATED in the test's docstring (commit
below) with the measured `max|R|`, floor, bar and both ratios, and with a dated
note saying what the two wrong numbers were.  No numeric bar changed.  The same
two figures appear in the builder's own `FIX_...` G.6 bullet and should be
corrected there too.

### 5.2 A behaviour change worth recording, not a defect

The unified list is now `O(distinct cell permittivities)` on the SCALAR path,
where before it was `O(1)` (two half-spaces).  For a two-material cell that is
3 or 4 entries.  For a GRADED cell (a continuously varying `eps` map) it is
`Nx*Ny` entries, and `_grazing_safe_wavelength` evaluates its `min` in a Python
loop per candidate wavelength: measured 15.2 ms/call at 4098 entries (a fully
graded 64x64 scalar cell) and 45.5 ms at 12290 (its tensor promotion), against
0.031 ms for a two-material cell.  That is once per solve and negligible
against a staggered solve of that size, but it is a new cost class on the
scalar path and worth knowing about if the guard is ever called in a loop.

---

## 6. Task H -- the failure, reproduced (H-a, H-b, H-c, H-d)

### 6.1 The pre-fix test, executed on all three configurations

`tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py` at `fb3fd93`, extracted with
`git show` and run unchanged:

| configuration | `test_fff_nv_stripe_reduces_to_rigorous_1d` |
|---|---|
| Win-1 | **FAILED** -- "no truncation in 11..41 gave the rigorous 1-D solver its own exact lossless closure on this build" |
| Win-4 | **passed** (1.13 s) |
| WSL | **FAILED** -- the same message |

Same box, same interpreter, same library, same fixture; only
`OPENBLAS_NUM_THREADS` differs between rows 1 and 2.  **H-a and H-b
CONFIRMED**, executed rather than inferred.

### 6.2 The ladder (`v5_ladder.py`), my own reconstruction of the fixture

`|sum R + sum T - 2|` for `rcwa_jones_1d_segments` on
`[(0.5, rot(35 deg, no=1.5, ne=2.3)), (0.5, eps_g I)]`, `n_sub = 1.5`,
`n_sup = 1.0`, period 0.7 um, wl 1.0 um, depth 0.5 um, `n_orders` 11..41 odd:

| `eps_g` | config | worst | best | sound (<1e-9) of 16 | `sum R` at n = 41 |
|---|---|---|---|---|---|
| **2.25** (coincident) | Win-1 | **2.7607e-02** | **4.6122e-06** | **0** | 0.061838421925 |
| | Win-4 | **2.3094e-02** | **2.7223e-13** | **1** | 0.061828205202 |
| | WSL | **5.0009e-02** | **2.0607e-06** | **0** | 0.061828627079 |
| **2.10** (the shipped fixture) | Win-1 | 2.0828e-13 | 8.8818e-16 | **16** | 0.066449401958 |
| | Win-4 | 1.8208e-13 | 1.3323e-15 | **16** | 0.066449401957 |
| | WSL | 1.8607e-13 | 1.3323e-15 | **16** | 0.066449401958 |
| 2.40 (a second detune) | Win-1 | 1.8474e-13 | 2.2204e-16 | 16 | 0.060957687857 |
| | Win-4 | 3.1397e-13 | 8.8818e-16 | 16 | 0.060957687857 |
| | WSL | 1.8074e-13 | 8.8818e-16 | 16 | 0.060957687857 |
| 2.25 detuned by `r = 1e-10` | Win-1 | 1.0829e-05 | 1.2346e-13 | 2 | 0.061828207673 |
| | Win-4 | 7.4285e-06 | 1.1169e-13 | 1 | 0.061828205204 |
| | WSL | 1.0358e-05 | 1.3345e-13 | 1 | 0.061828207752 |

Every claimed digit reproduces.  Two things the builder's table does not say and
that make the case airtight:

* on the coincidence `sum R` ITSELF differs between builds at the 1e-5 level
  (0.061838 / 0.061828 / 0.061829) -- so it is not only the closure defect that
  is build-dependent there;
* on the detune `sum R` agrees to 1e-12 across all three (0.066449401958 /
  ...957 / ...958), which is what a converged answer looks like.

**H-c CONFIRMED.**

### 6.3 The Jones arm against a converged reference (H-d)

`v7_bars.py`, `No = 11`, reference `n_orders = 81`:

| | Win-1 | Win-4 | WSL |
|---|---|---|---|
| coincident `ef/el` | 2.7678e-02 | 2.7678e-02 | 9.9940e-02 |
| coincident `jf/jl` | 2.0010e-02 | 2.0010e-02 | **1.177317** |
| coincident 1-D reference closure @81 | 5.1132e-05 | 5.4477e-07 | 1.0307e-04 |
| detuned `ef/el` | **2.0926e-02** | **2.0926e-02** | **2.0926e-02** |
| detuned `jf/jl` | **1.8151e-02** | **1.8151e-02** | **1.8151e-02** |

The WSL `jf/jl` crosses the `1.0` the old `jf < jl` assertion needed, exactly as
claimed.  **H-d CONFIRMED.**

## 7. Task H -- the mechanism, re-derived (H-e, H-f, H-g, H-h)

`v6_mech.py` rebuilds what `rcwa_jones_1d_segments` builds internally
(`_tensor_convolutions(..., 'li')`, `_homogeneous_eigenmodes`,
`_layer_eigenmodes_tensor`) and measures four things.  Both grooves, `M` = 5,
11, 21, 31, 41, 61.  Win-1 (WSL agrees; the degeneracy counts are identical and
the conditioning ranges differ only in the last figure):

| `eps_g` | M | `max\|C-C^H\|/max\|C\|` | `cond(W)` | `cond([W;V])` | `cond(a+b)` sup | **`cond(a+b)` sub** | **min \|dlam\|** | **modes < 1e-12** |
|---|---|---|---|---|---|---|---|---|
| **2.25** | 5 | 1.026e-16 | 4.84 | 7.0 | 6.78 | **7.38e15** | **4.45e-16** | **10** |
| | 11 | 1.363e-16 | 4.16 | 19.7 | 16.30 | **2.44e15** | **2.68e-17** | **21** |
| | 21 | 6.803e-17 | 3.46 | 30.2 | 13.32 | **2.18e14** | **2.56e-18** | **41** |
| | 31 | 1.020e-16 | 5.15 | 46.6 | 19.93 | **2.62e14** | **3.43e-16** | **61** |
| | 41 | 1.020e-16 | 7.60 | 59.2 | 29.33 | **3.95e13** | **2.77e-16** | **82** |
| | 61 | 1.359e-16 | 4.47 | 92.5 | 12.73 | **2.70e12** | **1.22e-15** | **122** |
| **2.10** | 5 | 3.505e-17 | 3.19 | 6.8 | 9.21 | 1.42e02 | 5.41e-03 | **0** |
| | 11 | 6.979e-17 | 3.19 | 15.4 | 12.55 | 1.65e02 | 2.42e-03 | **0** |
| | 21 | 1.045e-16 | 11.47 | 30.6 | 35.68 | 3.55e01 | 9.78e-04 | **0** |
| | 31 | 1.044e-16 | 3.19 | 45.2 | 12.56 | 1.61e02 | 8.54e-04 | **0** |
| | 41 | 1.044e-16 | 3.20 | 60.8 | 4.86 | 1.14e01 | 2.98e-04 | **0** |
| | 61 | 1.391e-16 | 3.25 | 89.8 | 4.86 | 4.68e01 | 4.34e-04 | **0** |

Reading, claim by claim:

1. **Not non-Hermiticity.**  `max|C - C^H| / max|C|` is 3.5e-17..1.4e-16 at
   every truncation, for BOTH grooves.  The Li in-plane operator IS Hermitian
   for a real symmetric tensor, so an energy theorem exists and cannot be the
   explanation.  **H-e CONFIRMED.**
2. **Not eigenproblem conditioning.**  `cond(W)` <= 11.47 and `cond([W;V])`
   <= 107.7, so `cond * eps_machine` <= 2.4e-14 -- twelve decades under the
   1e-2 defect.  The CONCLUSION is confirmed; the builder's stated RANGES
   (`cond(W)` 3.2..7.6, `cond([W;V])` 18..2521) are not what I measure.  I could
   not identify what their `[W; V]` stacking or normalisation was.
   **H-f: conclusion CONFIRMED, numbers REFUTED.**
3. **It IS an exact index coincidence, and the interface inverse is the
   amplifier.**  On the coincident groove a whole BLOCK of layer modes matches a
   region mode to machine zero -- 10, 21, 41, 61, 82, 122 of the `4M + 2` layer
   modes at `M` = 5..61, with the smallest pair gap 2.6e-18..2.3e-15 -- and
   ZERO do on the detuned groove, where the smallest gap is 3.0e-04..5.4e-03.
   The matrix `_interface_smatrix` inverts EXPLICITLY at the layer->substrate
   interface (`a + b`, the `S12 = 2(a+b)^-1` the docstring warns about) is
   correspondingly `cond` 2.7e12..7.4e15 against 7..165.  The superstrate
   interface (`eps = 1`, not coincident) is 4.9..35.7 in both.
   **H-g CONFIRMED and sharpened** -- this is a direct measurement of the
   degeneracy and of the amplifier, which the builder inferred.

4. **The detune law.**  Worst `|sum R + sum T - 2|` over the same 11..41 ladder,
   detuning ONE of the three coincident quantities by a relative `r`:

| `r` | groove, Win-1 | `no`, Win-1 | `n_sub`, Win-1 | groove, WSL | `no`, WSL | `n_sub`, WSL | `eps_machine / r` |
|---|---|---|---|---|---|---|---|
| 0 | 2.761e-02 | 2.761e-02 | 2.761e-02 | 5.001e-02 | 5.001e-02 | 5.001e-02 | -- |
| 1e-12 | 4.511e-04 | 1.220e-04 | 7.438e-05 | 5.882e-04 | 1.183e-04 | 1.411e-04 | 2.220e-04 |
| 1e-09 | 4.078e-07 | 2.017e-07 | 8.848e-08 | 2.809e-07 | 2.584e-07 | 1.009e-07 | 2.220e-07 |
| 1e-06 | 5.393e-10 | 4.326e-10 | 1.163e-10 | 6.253e-10 | 3.759e-10 | 1.512e-10 | 2.220e-10 |
| 1e-03 | 4.856e-13 | 3.197e-13 | 2.696e-13 | 4.197e-13 | 4.268e-13 | 1.381e-13 | 2.220e-13 |
| 1e-02 | 2.540e-13 | 1.745e-13 | 1.439e-13 | 2.434e-13 | 1.652e-13 | 1.374e-13 | 2.220e-14 |

Every column tracks `eps_machine / r` within a factor of ~3 down to `r = 1e-3`,
where it floors on the closure floor itself -- the signature of a division by a
vanishing gap.  The `r = 0` row is the only one that differs between builds
(2.761e-02 vs 5.001e-02); every row from 1e-6 down agrees between them to
within a factor of 2.  **H-h CONFIRMED.**

My values differ from the builder's h6 table by up to 3x because their sample
was `n_orders` = 7, 11, 21, 41, 61 and mine is the full 11..41 odd ladder; the
law and the conclusion are the same.

### 7.1 "It is not the anisotropic solver" (`v9_isolate.py`)

The same call, the same geometry, different permittivities.  `sum R + sum T - 2`
(signed), Win-1:

| cell | 11 | 21 | 31 | 41 | 61 |
|---|---|---|---|---|---|
| **the fixture: rot 35 deg vs 2.25** | **1.05e-02** | **-3.07e-03** | **2.15e-03** | **-1.00e-04** | **7.74e-07** |
| the same tensor NOT rotated (`exy = 0`) | -1.51e-14 | 1.91e-14 | 6.31e-14 | -2.44e-15 | 1.23e-13 |
| isotropic 5.29 vs 2.25 | -1.02e-14 | -2.71e-14 | -6.35e-14 | 1.28e-13 | 7.07e-13 |
| a DIFFERENT symmetric tensor (`exy = 0.5`) | 1.73e-14 | 6.79e-14 | 5.82e-14 | 8.62e-14 | 6.33e-13 |
| GYROTROPIC Hermitian (`exy = +0.5i`) | -4.02e-14 | 1.02e-13 | -5.65e-13 | -2.25e-13 | 1.69e-12 |
| the fixture, UNIFORM (no grating) | -4.44e-16 | -4.44e-16 | -4.44e-16 | -4.44e-16 | -4.44e-16 |
| the fixture, half-spaces matched 1.0/1.0 | 9.19e-14 | -3.38e-13 | 1.39e-12 | 1.15e-13 | 2.84e-12 |
| **the fixture, groove detuned to 2.10** | 3.11e-15 | -1.53e-14 | 1.14e-13 | -1.83e-13 | -4.25e-13 |

Row 1 reads 1e-02..1e-07; every other row reads 1e-16..1e-12.  Off-diagonal
anisotropy per se is fine (rows 4 and 5), rotation per se is fine, the tensor is
fine uniform, and the SAME cell with only the groove moved is fine (row 8).
The builder's rows 1, 2, 6 reproduce to every printed digit; rows 3, 4, 5 and 7
are machine-level in both readings but not digit-identical (my control cells are
constructed slightly differently).  **The claim is CONFIRMED.**

## 8. Task H -- the restated test (H-i, H-j, H-k) and its durability

`v7_bars.py` measures every quantity
`tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py` asserts, on all three
configurations.  A cell marked "=" means the three configurations agreed to
every printed digit.

| test | quantity | bar | Win-1 | Win-4 | WSL | gap to bar |
|---|---|---|---|---|---|---|
| operator reduces to Li-1996 | `max\|Cxx - inv_exx\|` | `< 1e-12` | 8.8819e-16 | = | 8.8819e-16 | 1126x |
| stripe reduction | 1-D reference closure @81 | `< 1e-9` | 1.0916e-12 | 8.5931e-13 | 1.0778e-12 | **916x (2.96 dec)** |
| | fff_nv closure @ No = 11 | `< 1e-9` then `< 6e-2` | 1.4211e-14 | 2.3537e-14 | 5.3735e-14 | 1.9e4x / 1.1e12x |
| | fff_nv closure @ No = 9 / 13 | (same) | 5.3735e-14 / 2.4425e-15 | 0.0 / 4.6629e-14 | 7.5495e-15 / 3.1086e-14 | >= 1.9e4x |
| | min per-order `Rf`, `Tf` | `>= 0` | 0.0 / 0.0 | = | = | at the bar by construction |
| | `ef` | -- | 3.039885e-05 | = | = | -- |
| | `el` | -- | 1.452697e-03 | = | = | -- |
| | **`ef/el`** | `< 0.2` | **2.092581e-02** | = | = | **9.56x** (and 47.8x under 1.0) |
| | `jf` | -- | 5.815772e-05 | = | = | -- |
| | `jl` | -- | 3.204165e-03 | = | = | -- |
| | **`jf/jl`** | `< 0.2` | **1.815067e-02** | = | = | **11.0x** (and 55.1x under 1.0) |
| degeneracy test | clean worst closure | `< 1e-9` | 2.0828e-13 | 1.8208e-13 | 1.8607e-13 | 4802x |
| | degenerate worst closure | -- | 2.7607e-02 | 2.3094e-02 | 5.0009e-02 | -- |
| | **ratio** | `> 1e5` | **1.3255e11** | 1.2684e11 | 2.6876e11 | **1.27e6x** |
| beats-laurent | `ef/el` | `< 0.5` | 4.5577e-03 | = | = | 110x |
| lossy absorptance | `A1` | `0.05 < A1 < 0.95` | 4.2956e-01 | = | = | 8.6x / 2.2x |
| | `\|Af-A1\| / \|Al-A1\|` | `< 1` | 5.3682e-02 | = | = | 18.6x |
| crossed cell | `ef[11]/el[11]` | `< 0.3` | 9.5221e-02 | = | = | **3.15x** |
| | `ef[11]/ef[5]` | `< 1` | 3.2675e-01 | = | = | 3.06x |
| | monotone slack | `<= 1e-9` | -3.7548e-03 | = | = | decisive |
| uniform routes | `max\|Rf-Rl\|`, `max\|Jf-Jl\|` | `< 1e-12` | 0.0 / 0.0 | = | = | exact |
| OOP converges fast | `\|e0 - 2\|` | `< 1e-9` | 1.3767e-14 | 4.8184e-14 | 1.3767e-14 | 2.1e4x |
| | **`\|f7 - conv\|`** | `< 2e-5` | **9.664963e-06** | = | = | **2.07x** |
| | `\|f7-conv\| / \|l7-conv\|` | `< 0.5` | **3.194275e-01** | = | = | **1.57x** |
| berreman cross-method | worst `\|sv_r - sv_b\|` | `< 1e-11` | 3.4694e-15 | 3.6082e-15 | 2.4147e-15 | 2771x |
| OOP same limit | `gap25/gap7` | `< 0.5` | **2.815658e-01** | = | = | **1.78x** |

Readings:

* Every quantity the RESTATED test compares (`ef`, `el`, `jf`, `jl` and both
  ratios) is identical to all SEVEN printed digits on the three
  configurations -- stronger than the claimed "five significant figures".
  **H-j CONFIRMED and strengthened.**
* `ef/el` sits 9.56x under its 0.2 bar and 0.2 sits 5x under the 1.0 the claim
  means; `jf/jl` 11.0x and 5x.  Both gaps are as the builder derived them.
* The degeneracy test's ratio bar has six decades of margin and its negative
  arm is five decades above the 1e-13 floor it is taken from.  **H-i, H-k
  CONFIRMED.**
* **Four bars in this file sit within a decade of their measurement**:
  `|f7 - conv|` (2.07x), the OOP convergence ratio (1.57x), the OOP same-limit
  ratio (1.78x) and the crossed-cell ratio (3.15x).  ALL FOUR are pre-existing
  (untouched by this branch), all four are already documented in the file as
  deliberately retained by the 2026-08-15 sibling sweep, and all four measure
  IDENTICAL to seven digits on Win-1, Win-4 and WSL.  The durability rule
  exists to catch a bar whose margin is inside the build spread; here the
  spread is zero, so the reading is "thin but over a deterministic value", the
  same verdict the earlier audit reached with two mounts and which now holds
  with three configurations.  Reported, not restated.
* The one bar whose margin is worth watching is `_ONED_SOUND_CLOSURE = 1e-9`
  against the reference's own closure at `_ONED_REF_ORDERS = 81`: 1.09e-12,
  i.e. 916x.  That is still 3 decades and comfortably outside the rule, but it
  is the SMALLEST margin among the restated bars and it GROWS with the
  reference order (the same quantity is 2.08e-13 at the top of the 11..41
  ladder).  MEASURED (`v9_isolate.py`, Win-1, detuned fixture):

  | `n_ref` | closure | `sum R` | margin to 1e-9 |
  |---|---|---|---|
  | 41 | 1.8319e-13 | 0.066449401958 | 5459x |
  | 61 | 4.2477e-13 | 0.066447791681 | 2354x |
  | **81** (`_ONED_REF_ORDERS`) | **1.0916e-12** | **0.066447219556** | **916x** |
  | 101 | 7.9625e-13 | 0.066446952496 | 1256x |
  | 141 | 2.2196e-12 | 0.066446718238 | 451x |
  | 181 | 7.1898e-13 | 0.066446621134 | 1391x |

  -- noisy but trending down, 451x at the worst sampled order.  The four
  `sum R` values the builder quotes for the `_ONED_REF_ORDERS` derivation
  (0.066447791681 / 0.066447219556 / 0.066446952496 / 0.066446718238 at
  61 / 81 / 101 / 141) reproduce to all twelve digits, so the choice of 81 is
  independently confirmed.  If `_ONED_REF_ORDERS` is ever raised, this margin
  must be re-measured, not assumed.

### 8.1 Fail-before, executed

The groove reverted to 2.25 (one-line `sed` on a copy), everything else
unchanged:

| configuration | `test_fff_nv_stripe_reduces_to_rigorous_1d` | `test_stripe_fixture_is_free_of_the_mode_match_degeneracy` |
|---|---|---|
| Win-1 | FAILS, reference closure 5.113e-05 @81 | FAILS, worst 2.761e-02 in 11..41 |
| Win-4 | FAILS, reference closure 5.448e-07 @81 | FAILS, worst 2.309e-02 |
| WSL | FAILS, reference closure 1.031e-04 @81 | FAILS, worst 5.001e-02 |

and both pass on the detuned fixture on all three (section 10).  **H-k
CONFIRMED**, executed on every configuration.

---

## 9. Follow-ups done

### 9.1 Follow-up A -- `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py`

**The task brief (inherited from the builder's H.6 open item 1) names the wrong
test.**  `test_pmm_fff_nv_stripe_reduces_to_rigorous_1d` does NOT use the
coincident fixture: its cell is `rot(40 deg, no=1.6, ne=3.0)` against air, with
no index coincidence anywhere, and it carries no ladder apparatus at all.  Its
bars are healthy and build-free -- 1-D reference closure 3.135e-13 / 2.873e-13 /
3.817e-13 against `_CLOSE_TOL` = 1e-3, `ef` = 9.097432e-05 against 1e-3 (11x),
`ef/el` = 1.242292e-02 against 0.2 (16.1x), the last two identical to seven
digits on all three configurations -- so nothing was changed there.

The test that DOES use the coincident cell (`rot(35 deg, 1.5, 2.3)` against
`2.25 I`, `n_sub = 1.5`) and that carries `_RCWA_REF_STAGES`, `_scan` and
`_corroborated_reference` is **`test_pmm_fff_nv_matches_rcwa_fff_nv`**.  The
"~46 s of scanning" the builder attributed to the ladder is in fact the
recorded duration of the OTHER test (46.188 s), which does no scanning.

Whether the apparatus exists only to survive the degeneracy was MEASURED
(`v8_pmm1320.py`), RCWA fff_nv closure over the 5-rung `Sx = 64` ladder:

| groove | Win-1 | Win-4 | WSL | rungs under 1e-3 |
|---|---|---|---|---|
| 2.25 | worst **2.642e-02** | worst **1.722e-02** | worst **8.220e-03** | 4 / 4 / **2** of 5 |
| 2.10 | worst **5.373e-14** | worst **4.663e-14** | worst **5.373e-14** | 5 / 5 / 5 of 5 |

and at `Sx = 128`: 8 / 9 / 8 clean of 9 at 2.25, **9 / 9 / 9** at 2.10.  At 2.10
every `sum(R)`, every `sum(T)` and every PMM rung's closure agrees to all nine
printed digits across the three configurations; at 2.25 one and the same rung
reads `sum(R)` = 0.064400 / 0.061786 / 0.062006 on them.  So yes -- the RCWA
side of the apparatus existed entirely to survive the coincidence.  The PMM side
does NOT: its closure is 1.524e-04 / 9.499e-05 / 1.897e-04 at rungs 9 / 11 / 13
at BOTH grooves, which is the hybrid's Fourier floor at degree 11, not an
instability.

Applied (commit `4992fb3`):

* `_STRIPE_EPS_GROOVE = 2.10` (the value the 5_20_12 fix used), with the
  three-configuration table above in its docstring.
* Reference FIXED again at `_RCWA_REF_ORDERS = 15` -- the highest the sampling
  allows -- with the rigorous engine's own Li-1996 theorem ASSERTED at EVERY
  rung (`_RCWA_SOUND_CLOSURE = 1e-9`; worst reading 5.373e-14, 4.3 decades
  under, and ~5 decades under what the coincidence gives).  That assertion is
  the fixture's tripwire, the same shape the 5_20_12 fix used.
* DELETED: `_RCWA_REF_STAGES` (the two-stage sampling refinement),
  `_corroborated_reference` and `_AGREE_TOL`.
* KEPT: `_scan`, `_clean`, `_table`, `_CLOSE_TOL`, `_PMM_LADDER`,
  `_PMM_WANT_CLEAN` -- the PMM-side scan is doing real work.
* `_CLOSE_TOL` (1e-3) and `_CROSS_TOL` (4e-3) keep their VALUES; their sizing is
  now measured and stated: 5.3x over the worst PMM closure (1.897e-04) and 5.9x
  over the worst cross-solver residual (6.727e-04 in `sum T`, against
  `_RCWA_REF_ORDERS`), both identical to nine digits on the three
  configurations.  Both are inside a decade of their measurement and are
  therefore reported here as thin -- but with zero build spread beneath them
  and, for `_CROSS_TOL`, ~15x of room below the 1e-2-and-up a gross
  factorization error gives, there is no wider placement that keeps both gaps.
* The three history entries in the docstring are kept verbatim; only their
  DIAGNOSIS is corrected.

Fail-before executed (groove reverted to 2.25) on all three: the new ladder
assertion fires at 2.642e-02 / 1.722e-02 / 8.220e-03.  Pass-after 5 passed on
all three.

Durations, `--durations-path .test_durations --store-durations` (it MERGES):
**12299 -> 12300 entries, 1 added, 0 removed, 10 re-timed.**

| entry | before | after |
|---|---|---|
| `...::test_closure_warning_names_the_exact_index_coincidence_and_the_detune` | (absent) | **0.010** (added, follow-up B) |
| `...::test_pmm_fff_nv_matches_rcwa_fff_nv` | **22.000** | **12.058** |
| `...::test_pmm_fff_nv_stripe_reduces_to_rigorous_1d` | 46.188 | 46.063 |
| `...::test_pmm_fff_nv_lossy_absorptance_split` | 46.076 | 45.112 |
| `...::test_pmm_fff_nv_crossed_and_offplane_and_jax_raise` | 0.174 | 0.288 |
| `...::test_pmm_fff_nv_uniform_routes` | 0.229 | 0.240 |
| five `test_audit_s1_2_rcwa_lossless_tripwire.py` entries | 0.002..0.042 | 0.002..0.045 |

Whole file, Win-1: 105.72 s before -> 105.05 s after (the ladder was already
cheap on a build where stage 1 worked; the saving is the 9.9 s of the recorded
duration, and the removal of the stage-2 path that a build like WSL-4-threads
used to walk).

### 9.2 Follow-up B -- `rcwa._core._check_energy`'s messages

Commit `8bab706`.  Both messages already named the `(period, n_orders)`
near-degeneracy, whose remedy is to move `n_orders`.  The EXACT index
coincidence is a different state with the opposite remedy, and the message sent
the reader somewhere that cannot help (0 of 16 sound truncations at Win-1).
Both the `_EnergyError` and the `_EnergyWarning` text now name it:

> ...If a LAYER permittivity is EXACTLY EQUAL to a REGION's (a groove, or a
> rotated director's ordinary no^2, equal to n_substrate^2 or n_superstrate^2)
> the layer<->region mode match is exactly -- not nearly -- degenerate at EVERY
> truncation and no n_orders helps: DETUNE one of the coincident permittivities
> by a relative ~1e-6 instead.

and the function docstring gains a paragraph distinguishing the two, carrying
the measured degeneracy counts and `cond(a+b)` readings from section 7.

**No sibling of this shape exists in `pmm/_core.py`.**  The two PMM closure
warnings I found (`pmm/twod.py:183` and `pmm/stack2d_pure.py:199`) diagnose
different mechanisms -- an ill-conditioned Fourier projection at an
`(n_orders, degree)` coincidence, and staggered under-resolution / a Rayleigh
cut-off -- and neither points at `n_orders` for an index coincidence, so
neither was touched.

**The diff is message and docstring text only.**  Verified two ways: the diff
touches only the two f-strings and the docstring (`git diff`), and re-running
`v1_bitid.py` (14 staggered fixtures, 36 hashes, 14 guard calls) and
`v5_ladder.py` (64 closure + `sum R` readings) across the edit gives
BYTE-IDENTICAL JSON.

Test: one assertion, in `tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py`
(the file that owns this guard), that the warning the coincident fixture raises
names `EXACTLY EQUAL` and `DETUNE`.  `n_orders = 19` is used because it is the
WORST truncation on every configuration measured (2.761e-02 / 2.309e-02 /
5.001e-02), so the warning is guaranteed rather than hoped for.  Fail-before
executed against the pre-edit text.

---

## 10. Suites and lint

`OMP/OPENBLAS/MKL_NUM_THREADS=1`, `-p no:randomly`.

| suite | Win-1 | WSL |
|---|---|---|
| `tests/unit/test_pmm2d_staggered_wood_list.py` | **18 passed**, 2.84 s | (in the 12-file set below) |
| `tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py` | **11 passed**, 123.21 s | (in the 3-file set below) |
| `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py` | **5 passed**, 2 warnings, 106.56 s | (in the 3-file set below) |
| the 11-file staggered / pure set | **224 passed**, 23 warnings, 587.38 s | -- |
| `tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py` (follow-up B's home) | **6 passed**, 3 warnings, 0.84 s | (in the 3-file set below) |
| the two fff_nv files + the tripwire file | -- | **22 passed**, 5 warnings, 255.30 s |
| the 11-file staggered set + the wood-list file | -- | **242 passed**, 24 warnings, 600.42 s |

The 11-file staggered set is the one the builder listed: `test_v5_12_0_pmm2d_staggered`,
`test_v5_21_pmm2d_staggered_oblique`, `test_staggered`,
`test_audit_p1_staggered_guard`, `test_p2c_pmm2d_stack_cascade`,
`test_p2t_pmm2d_tree_cascade`, `test_pmm2d_lossless_closure_two_sided`,
`test_audit_s1_3_pmm2d_lossless_tripwire`, `test_v5_14_0_pmm2d_stack`,
`test_pmm2d_staggered_anisotropic`, `test_pmm2d_staggered_oop`.  Its
**224 passed, 23 warnings** matches the builder's count exactly; on WSL it is
run together with the 18 wood-list tests, hence 242.  Zero failures, zero
errors anywhere.  (Windows wall times run ~25 % over the builder's because
several probe runs and the WSL legs shared the box; the counts are what the
gate is.)

Windows at 4 BLAS threads was additionally used for every fail-before /
pass-after pair in sections 6, 8.1 and 9.1 and for
`test_audit_s1_2_rcwa_lossless_tripwire.py` (6 passed), but not for the whole
suite list.

`ruff check lumenairy/ tests/` -- **All checks passed!**

---

## 11. Defects found

None in either library fix.  Four documentation / reporting defects, and two
claims that are true only with a qualifier:

1. **`test_the_layer_cutoff_nudge_is_consequential`'s docstring states the wrong
   floor.**  It says 2.6e-12 and "~2.9e3x above that floor"; the floor the test
   actually computes is `1e3 * max|R| * eps_machine` = 2.5572e-15 and the ratio
   is 2.986e6 (2.986e4 against the asserted `100 * floor`).  Wrong by 1e3, in
   both numbers consistently, so it is a transcription slip.  The bar is
   correctly sized; only the prose is wrong.  FIXED in the test's docstring
   (commit below) with the measured `max|R|`, floor, bar and both ratios plus a
   dated note recording what the wrong numbers were.  The same two figures
   appear in the builder's `FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md` G.6 bullet
   and should be corrected there.
2. **The builder's H.6 open item 1 names the wrong test** (section 9.1):
   `test_pmm_fff_nv_stripe_reduces_to_rigorous_1d` has neither the coincident
   fixture nor any ladder apparatus; both belong to
   `test_pmm_fff_nv_matches_rcwa_fff_nv`.  The "~46 s of scanning" is the
   non-scanning test's duration.  The follow-up was applied to the correct
   test.
3. **`cond([W; V])` is not what the builder reports** (section 7): 6.8..107.7
   measured against a claimed 18..2521 over the same `n_orders` range.  The
   conclusion the number supports is unaffected -- `cond * eps_machine` is
   twelve decades under the defect either way -- but the figure should not be
   quoted.
4. **G.1's uniform-SCALAR-layer row is not reproducible from the claim text**
   (section 3): 4.5908e-08 / 1.3045e-15 measured against a claimed
   4.977e-08 / 1.381e-15.  Same class, same conclusion, different digits.

Qualifiers:

5. **"The warning behaviour is unchanged" holds off the coincidence only.**
   The post-fix nudge moves `kt^2` by `2e-7 * kt^2` and therefore moves the
   `1e-4` warning boundary by ~8e-7 in `kt^2` units on a geometry that also
   sits exactly on a layer cut-off.  Flips measured in both directions
   (section 4).  Not a defect -- the post-fix warning is computed at the
   wavelength actually solved -- but the claim needs the caveat.
6. **The scalar path's nudge list is now `O(distinct cell permittivities)`**
   where it was `O(1)` (section 5.2).  Irrelevant for a two-material cell,
   45 ms/call for a fully graded 64x64 one, once per solve.

## 12. Changes committed on this branch by the verification

| commit | what |
|---|---|
| `d58a97f` | `validation/probe_verify_wood_fffnv/` -- eight probes and their recorded readings for Win-1, Win-4 and WSL |
| `8bab706` | **follow-up B**: `_check_energy`'s two messages + docstring name the exact index coincidence and the detune remedy; one assertion in `test_audit_s1_2_rcwa_lossless_tripwire.py`.  Message text only, bit-identity checked |
| `4992fb3` | **follow-up A**: `test_pmm_fff_nv_matches_rcwa_fff_nv`'s groove detuned to `_STRIPE_EPS_GROOVE` = 2.10, fixed reference + asserted ladder theorem replacing `_RCWA_REF_STAGES` / `_corroborated_reference` / `_AGREE_TOL`, bars re-derived, `.test_durations` spliced |
| `e2ffa97` | defect 1: `test_the_layer_cutoff_nudge_is_consequential`'s docstring floor/ratio corrected (2.6e-12 / ~2.9e3x -> 2.5572e-15 / 2.99e6x).  Docstring only, no bar changed |
| (this file) | the verification report |

No merge, push, tag or version bump.  The only library edit is the
message/docstring text of follow-up B.

## 13. What could not be verified

* **Cross-LAPACK, beyond these three configurations.**  Win-1, Win-4 and WSL
  are one machine and two OpenBLAS builds.  Where a claim is about "any build"
  the evidence here is three readings, not a proof; the thread-count axis is a
  partial substitute for a second machine, not a second machine.
* **The builder's own probe scripts were not executed.**  Everything here is a
  re-measurement, which is why two of their reported figures (defects 3 and 4)
  are recorded as not reproduced rather than as arithmetic errors: I cannot
  tell from the claim text what their probes computed.
* **The JAX legs.**  `test_fff_nv_jax_raises` and the pmm JAX guard run in the
  suites, but no JAX numerical path was exercised; the staggered path is
  NumPy-only.
* **The full library suite.**  Only the files named in the task plan were run.
* **Whether an exactly degenerate layer<->region mode pair could be handled
  rather than avoided** (the builder's H.6 open item 2).  Section 7 measures
  that the interface `a + b` is the amplifier, which is a stronger hint than
  the builder had -- deflating the shared subspace or matching by invariant
  subspace would attack exactly that matrix -- but nothing was attempted and
  no estimate of the work is offered.
