# VERIFY — the `PMMStack` near-coincident-wall (SLIVER) fix, re-measured

**Date** 2026-09-11 · **Branch** `verify/sliver-walls` (from `wave2/pmm2d`
@ `dd06c09`, i.e. the sliver fix `cbeb223..1c9fc3f` merged, plus the per-layer
grid / non-uniform-segment / mortar work) · **Worktree**
`C:/tmp/lum_vsliver` · **Subject**
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md`

**Method** Every number below was RE-MEASURED with scripts written for this
verification (`validation/probe_verify_sliver/`, ten probes + a README), on
fixtures of my own where the claim is general and on the fix's own fixture
where the claim is about a specific number. Nothing was read off the fix's
JSON. Both directions: claimed successes were attacked, and the two open items
the fix left as "small, contained" were re-opened and one of them was fixed.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Verdicts

| # | Claim (fix audit) | Verdict | Key numbers, this verification |
|---|---|---|---|
| 1 | Bit-identity of every shipped `PMMStack` path | **CONFIRMED** | **21 / 21** of MY OWN fixtures identical to the read-only main clone (`a68a0da`), on **both** builds. `_pmm_union_grid`'s 2-tuple byte-identical |
| 2 | Mechanism: nodal `Kx²` ∝ 1/w², `\|q\| ≈ 0.65 N(N+1)/4/(k0 J)` | **CONFIRMED** | fitted exponents 2.000 / 1.000 on my own fixture at 3 degrees × 4 widths; constant 0.6434–0.7076 |
| 2b | "the interface mode-match conditions as 1/w²" | **CONFIRMED, with the origin located** | the BLOCK system `[[Wa,−Wb],[Va,Vb]]` fits p = 2.000 — through **`cond(V)`** (p = 2.000). `cond(W)` and `cond(a+b)` are FLAT (p ≈ 0.01) |
| 3 | O-11 correction 1: not energy-invisible | **CONFIRMED** | `R+T` = 2.17298 / 23.4225 / 3.61285; the warning fires on exactly those 3 rows and no others |
| 4 | O-11 correction 2: per-layer reads identically because a 2-layer window IS the union | **CONFIRMED and STRENGTHENED** | gap = **0.0 exactly** (not "< 1e-12") at every δ; and at GRID level `np.array_equal` for both layers |
| 5 | Separation `4.125e-06` vs `+1.159` on the fix's grid | **CONFIRMED on that grid** (one definitional correction) | I read `1.1588` and `4.593e-06`; the audit's 4.125e-06 scores pol 1 only, the guard tests max over both pols |
| 6 | "the bar sits 3.39 decades above the correct population and 2.06 below the wrong one" | **REFUTED as a property of the family** | on 120 dense δ × 3 degrees: max \|R+T−1\| among CORRECT = **9.867e-05** (2.01 decades), min (R+T−1) among WRONG = **7.142e-03** — **below the bar** |
| 7 | Attribution ratio ≥ 100; widest wrong row 3172; non-conforming 1.3–4.0 | **CONFIRMED** | 3170.5 measured at δ = 8.786e-05, degree 20; control reads exactly 1.33 / 4.0 / 4.0 |
| 8 | "the M2 audit-class 2° coated taper is at ratio ~1.7e+02, 0.23 decades above the bar" | **REFUTED** | `_cross_layer_sliver` reads **12.11** on that device (own-scale is the 5 nm COAT, not the 200 nm ridge) and returns **None** at every `n_slices` |
| 9 | False positives: 0 | **CONFIRMED on the O-11 family, REFUTED off it** | 0/190 dense rows, 0/243 walk rows, 0/64 conical+slant rows — but **110 of 648** realistic staircase configurations refuse a solve within 0.35–8.8× the physical shift |
| 10 | False negatives: 1 row in 138 ("open item B") | **BOUNDED — wider than stated** | **8** returned-but-not-correct rows in 660 samples (5 WRONG, 3 grey) (errors to **2.80e-03**, `R+T−1` to **+7.14e-03**), plus 1 in 40 on the conical path; band = isolated δ, ≤ 0.6% of δ wide |
| 11 | Remedy: `err/δ` = 1.152–1.154 inside a `2 δ` bar | **CONFIRMED on O-11, REFUTED as scale-free** | 1.1524–1.1535 (O-11) ✓; 1.2073–1.2077 (my telecom fixture) ✓; **3.5127–3.5147** (my 0.9 µm fixture) — 9/9 rows outside `2 δ` |
| 12 | Open item A (asymmetric snap) is "small, contained" | **REPRODUCED, and FIXED (follow-up 1)** | when it bites the answer is 0.93 / 2.40 / 16.4 from the exact limit; `<=` does not fix it; a 16-ULP deadband does, 11 of 11,418 cases changed, all on-threshold |
| 13 | `PMM2DStackPure`'s union is the pixel lattice (aspect 1.0) | **CONFIRMED, STRENGTHENED** | aspect exactly 1.0 at N = 8, 12, 24, 64, 129, and `add_layer` REFUSES a second lattice, so no non-trivial union can form |
| 14 | `PMM2DStackHybrid` has no union grid | **CONFIRMED** | two different lattices accepted and solved; super-unity 1.0103 → the plain warning, no SLIVER text |
| 15 | "the mortar route is the safe one" | **CONFIRMED** for this hazard, with one message defect | a within-layer sliver to **1e-6** of a period: `err/d` = 0.34–0.46, `\|R+T−1\|` ≤ 8.0e-07, `n_modes` spread → 7.9e-09. At 1e-7: `LinAlgError: Singular matrix` (loud, unhelpful, not silent) |
| 16 | Open item E: the onset was mapped only on the classical cascade | **CLOSED by measurement** | conical and slant mapped: 0 false positives, 1 miss in 240 rows (conical, degree 10) |
| 17 | Open item C (a comment promising a report that does not exist) | **CONFIRMED, and FIXED (follow-up 2)** | |

Two structural blind spots the fix audit does not name are documented in S4.4.

---

## S1. The two builds, and the reference arm

| | Windows | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | scipy-openblas 0.3.31.188.0 | scipy-openblas 0.3.31.188.0 |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` | same |

The pre-fix ("without") arm is the **READ-ONLY main clone**
`D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`, now at
`a68a0da` (5.44.0 candidate). I verified independently, not by reading the fix
audit, that its `lumenairy/elements/pmm/{stack,_core,conical,oned,_jax_stack,
twod,stack2d}.py` are byte-identical (modulo line endings) to this branch's
pre-fix base `3034bcb`, and that `PMM_SLIVER_GUARD` does not exist there. Every
probe asserts which `lumenairy.__file__` it imported.

The `wave2/pmm2d` work merged after the fix's branch point touches `_core.py`
only ADDITIVELY (four new 2-D mortar functions); no 1-D `PMMStack` path is in
its diff, which is why the verify-tree-vs-main-clone comparison in S2 isolates
the sliver fix.

---

## S2. Bit-identity — `v1_bitid.py`, `v_fixtures.py`

21 fixtures of my own design (different periods, wavelengths, angles, indices,
degrees, segment counts and tensors from the fix's 18), hashed as sha256 over
each returned array's dtype, shape and raw buffer:

single layer normal · single layer 40° oblique · 5-layer shared union ·
the same 5 layers on per-layer windows at halfwidth 1 and 2 · conical
(`phi = 0.62`) · slanted (`slant_angle = 0.17`) · out-of-plane tensor
(`eps_xz` populated) · in-plane tensor with `eps_xy` · lossy layer ·
**absorbing superstrate** · `solve_vs_wavelength` (5 λ, `jones=True`) ·
`prepare()` + a 2-point keyed material sweep · `stabilize='slices'` ·
`retain_internal` + `internal_field` (nodal and `nx=17` resampled) ·
`layer_absorption` (plain and `by_material`) · a 6-slice taper with
`min_feature` SNAPPING ACTIVE · the same taper with the snap DORMANT ·
`per_order_amplitudes` (both ports) · an 8-layer Bragg ABAB at 31 orders ·
`_pmm_union_grid`'s **2-tuple** on six geometries (snap dormant, snap active,
non-conforming, single-layer liner, one-layer stack, 6-slice taper).

| | fix tree vs main clone |
|---|---|
| Windows | **21 / 21 identical**, 0 differing, 0 errors |
| WSL | **21 / 21 identical**, 0 differing, 0 errors |

Cross-build, only 1 of the 21 agrees (the pure-geometry union-grid fixture) —
which is the control that the two builds really do carry different reduction
orders, so the 21/21 within each build is evidence and not one build's luck.

`_pmm_union_grid(..., return_owners=False)` returns a 2-tuple whose two members
hash identically on both trees on all six geometries, including the snapping
one — the additive-keyword contract holds.

---

## S3. The mechanism — `v2_mech.py`

### S3.1 The scalings, FITTED rather than assumed

My own fixture: period 0.9 µm, λ = 0.62 µm, θ = 0.21, ε = 2.0 / 6.5, walls
0.311 / 0.688, snap disabled. Widths `w` = 1e-2, 1e-3, 1e-4, 3e-5 of the
period; degrees 10, 14, 18. The exponent `p` in `X ∝ w^(−p)` is fitted between
consecutive widths, so the claimed powers are read off the data:

| quantity | fitted `p`, degree 10 / 14 / 18 (last, finest pair) |
|---|---|
| `cond(S0)` | 1.000 / 1.000 / 1.000 |
| `‖Kx²‖₂` (`S0⁻¹ K / k0²`) | **2.000 / 2.000 / 2.000** |
| `\|q\|max` | **1.000 / 1.000 / 1.000** |
| `cond([[Wa, −Wb], [Va, Vb]])` (the mode-match block system) | **2.000 / 2.000** / 2.617 (degree 18 is already past its onset at `w` = 3e-5) |
| `cond(V)` | **2.000 / 2.000 / 2.000** |
| `cond(W)` | 0.015 / 0.027 / 0.043 — **flat** |
| `cond(a+b)` (the matrix actually inverted) | 0.001 / 0.006 / 8.767 (the last is the blow-up itself) — **flat until it explodes** |

**The 1/w² in the interface is real and it lives in `V`, not in `W`.** The fix
audit's "cond interface" column is the block system, which its own `p2_mech.py`
builds as `np.block([[Wa, -Wb], [Va, Vb]])`; that is what fits p = 2. The
matrix the solve actually inverts, `a + b`, stays at `cond` ≈ 4.5–6.1 across
three decades of `w` and then jumps to 2.36e+05 at the onset. Worth stating
precisely because it says WHERE the digits are lost: in forming
`b = solve(Vb, Va)` from a `1/w²`-conditioned magnetic partner, not in the
final inverse.

Largest interface S entry on my fixture: 2.41 → 2.74 → 3.10 / 3.29 / 3.38 →
and at degree 18, `w` = 3e-5, **1.5833e+05** — the same shape as the fix's
1.69 → 3.94 → 3.86e+02 → 3.20e+05.

### S3.2 The spurious-wavenumber predictor

`|q|max · k0 J / (N(N+1)/4)`, my own fixture:

| `w` | degree 10 | 14 | 18 | spread |
|---|---|---|---|---|
| 1e-2 | 0.7076 | 0.6992 | 0.6958 | 1.017 |
| 1e-3 | 0.6625 | 0.6522 | 0.6482 | 1.022 |
| **1e-4** | **0.6587** | **0.6476** | **0.6434** | **1.024** |
| 3e-5 | 0.6585 | 0.6473 | 0.6431 | 1.024 |

On the fix's own fixture I reproduce its five values exactly on both builds —
0.6735692208780433 / 0.6517528558849772 / 0.6477621340189265 /
0.6452663038854322 / 0.6424069650561796 at degree 8/12/14/16/20, spread
1.0485085895965318 (WSL agrees to the 15th digit). **CONFIRMED**: the constant
is ≈ 0.65 and its constancy across degree, which is the load-bearing half, holds
on a fixture the fix never saw.

### S3.3 O-11's two corrections

`v2_mech.py` part B rebuilds `f5f_attrib.py`'s oracle arm and adds the two
columns O-11 lacked. Both builds print the same digits:

| δ | self-gap SHARED | self-gap PER-LAYER | per-layer − shared | shift from ref | **R+T** | **warned?** | refused? |
|---|---|---|---|---|---|---|---|
| 1.00e-02 | 2.842e-08 | 2.842e-08 | **0.0** | 9.264e-03 | 1 | no | no |
| 2.60e-03 | 3.531e-08 | 3.531e-08 | **0.0** | 2.829e-03 | 1 | no | no |
| 1.00e-03 | 4.336e-08 | 4.336e-08 | **0.0** | 1.128e-03 | 1 | no | no |
| 3.00e-04 | 6.429e-08 | 6.429e-08 | **0.0** | 3.437e-04 | 1 | no | no |
| **1.00e-04** | **4.789e-01** | **4.789e-01** | **0.0** | 4.789e-01 | **2.17298** | **YES** | YES |
| **3.00e-05** | **8.172e+00** | **8.172e+00** | **0.0** | 8.618e+00 | **23.4225** | **YES** | YES |
| **1.00e-05** | 1.181e-04 | 1.181e-04 | **0.0** | 9.305e-01 | **3.61285** | **YES** | YES |
| 3.00e-06 | 5.468e-08 | 5.468e-08 | **0.0** | 3.461e-06 | 1 | no | no |
| 1.00e-06 | 5.468e-08 | 5.468e-08 | **0.0** | 1.154e-06 | 1 | no | no |
| 0 | 5.468e-08 | 5.468e-08 | **0.0** | 0 | 1 | no | no |

* The two self-gaps the fix quotes — **4.79e-01 @ 1e-4** and **8.17e+00 @ 3e-5**
  — reproduce to four significant figures on both builds. **CONFIRMED.**
* **Correction 1 CONFIRMED.** The 1-D stack is not energy-invisible: `R+T` reads
  2.17 / 23.4 / 3.61 and `_warn_stack_energy` fires on exactly the three wrong
  rows and on no correct one. `f5f_attrib.py` line 24 is
  `warnings.simplefilter("ignore")` at module scope — verified.
* **Correction 2 CONFIRMED and strengthened.** The per-layer column does not
  merely agree to 1e-12; it is **bit-identical** (max difference exactly `0.0`)
  at every δ. At GRID level (`_perlayer_window_grids(..., halfwidth=1)` vs
  `_pmm_union_grid`, `np.array_equal`):

  | layers | windows equal to the union | narrowest cell in every window |
  |---|---|---|
  | 2 | **both** | 1.0e-04 |
  | 3 | only the middle one | 1.0e-04 |
  | 5 | **none** | 1.0e-04 |

  So the caveat is exactly right, and one thing follows that the refusal does
  not say: on a 5-layer stack no window is the union, **and every window still
  carries the sliver**. `layer_grids='per-layer'` escapes cross-STACK wall
  accumulation, never the adjacent-slice collision.

---

## S4. The guard — `v3_guard.py`, `v9_falsepos.py`, `v10_paths.py`

### S4.1 The two populations

**On the fix's own grid** (46 log δ in 3e-3…3e-6 × degrees 12/14/20), re-run
with my script:

| | this verification (Win / WSL) | fix audit |
|---|---|---|
| rows / right / grey / wrong | 138 / 78 / 1 / 59 (both) | 138 / 78 / 1 / 59 ✓ |
| max `err/δ` among CORRECT | 1.1961 / 1.1961 | 1.196 ✓ |
| the grey row | degree 12, δ 5.544e-06, err 1.556e-04 (28.1×), `R+T−1` +2.3e-04 | identical ✓ |
| **min (R+T−1) among WRONG** | **1.1588 / 1.1588** | 1.159 ✓ |
| max \|R+T−1\| among CORRECT | **4.5928e-06 / 4.5916e-06** | 4.125e-06 |

The last row is a **definitional correction, not a disagreement**: the fix's
`p5_dense.py` scores `R[1].sum() + T[1].sum()` — polarization 1 only — while
`_warn_stack_energy` (and therefore the guard) takes `max` over both
polarizations. Scored the way the guard scores it, the correct population
reaches 4.59e-06, so the fix's stated "3.39 decades" is 3.34.

**On MY grid** (120 log δ in 3e-3…**1e-6** × degrees **10/14/20**, snap
disabled, classification by the same continuity rule):

| | Windows | WSL |
|---|---|---|
| rows / right / grey / wrong | 360 / 190 / 1 / 169 | 360 / 190 / 1 / 169 |
| **max \|R+T−1\| among CORRECT** | **9.8667e-05** | **9.8663e-05** |
| **min (R+T−1) among WRONG** | **7.1425e-03** | **7.1424e-03** |
| refused: wrong / right / grey | 168/169 · 0/190 · 0/1 | 168/169 · 0/190 · 0/1 |

**Claim 6 is REFUTED as a property of the family.** The correct population
reaches 9.87e-05 — **2.01 decades** below the 1e-2 bar, not 3.39 — and the wrong
population reaches DOWN to 7.14e-03, i.e. **below the bar**, so "2.06 decades
above" is a property of the 46-point sample, not of the defect. The two builds
agree to five significant figures on both numbers, so this is not sampling
noise on my side either.

Consequence for the shipped test (S7): the same two quantities on the shipped
13-row `_LADDER` read 3.151e-07 and 1.1726, i.e. 317× and 3.91× of headroom
against `bar/100` and `bar*30`. On my 120-row grid the same assertions would
read 9.87e-05 (1.01× — a 1.3% margin) and 7.14e-03 (**fails by 42×**).

### S4.2 The attribution ratio

| population | ratio, measured here | fix audit |
|---|---|---|
| ordinary NON-CONFORMING control (0.30/0.50 vs 0.35/0.55) — its three cross-layer cells | **1.33 / 4.0 / 4.0**, screen returns `None` | 1.33 / 4.0 / 4.0 ✓ |
| the WIDEST cell that produced a wrong answer (δ = 8.786e-05, degree 20) | **3170.5**, err 0.479, `R+T` 2.173 | 3172 ✓ |
| the M1 audit staircase (6 slices, 4 nm wall shift on a 1 µm period) | **157.5**, screen HITS with 10 flagged cells | 157 ✓ |
| the M2 audit-class 2° coated pillar taper | **12.11** (`n_slices` 2, 6, 8) and **3.59** (`n_slices` 3); screen returns **`None`** | "~1.7e+02, 0.23 decades above" ✗ |

**Claim 8 is REFUTED.** `_cross_layer_sliver`'s own-scale is
`min(width)` over EVERY segment of EVERY layer, and on the coated pillar that
is the **5 nm conformal coat** (0.00714 of the 700 nm period), not the ~200 nm
ridge the audit used. The narrowest manufactured cell is 0.4127 nm = 5.896e-04
of a period, so the ratio is 0.00714 / 5.896e-04 = **12.11** — **0.92 decades
BELOW** conjunct (a)'s bar, not 0.23 above it. `_cross_layer_sliver` returns
`None` on that device at every `n_slices` I tried, at the library default and
with the snap off.

This inverts the risk picture the fix states for that device class: the M2
coated taper is not a near-miss false-positive risk, it is **entirely outside
conjunct (a)** — if such a taper ever produces the sliver-wrong answer, the
guard cannot fire on it.

Monotonicity, for the record: at δ = 9.5e-05 (ratio 2932) the row is CORRECT
and at δ = 8.786e-05 (ratio 3170) it is WRONG. The ratio does not order the two
populations, which is the fix's own reason for making (a) an attribution filter
rather than the detector. **CONFIRMED.**

### S4.3 FALSE NEGATIVES — wrong solves the guard returns

`v3_guard.py` sections B, E and F, and `v10_paths.py`.

**Census.** 8 rows in 660 samples (60 log δ in 3e-5…1e-6 × degrees
8/10/12/14/16, plus the 120 × 3 grid of S4.1) that the guard RETURNS and that
are not CORRECT by the fix's own continuity rule — **5 are WRONG** (err > 100 δ)
and 3 sit in its GREY band (13–46 δ). Every one re-solved END TO END with the
guard armed: each RETURNED, and none of them even WARNED (all sit below the
1e-2 bar):

| degree | δ | err vs the exact limit | × the physical shift | `R+T−1` | snapped-remedy err |
|---|---|---|---|---|---|
| 20 | 1.7130e-06 | **2.8003e-03** | 1635× | **+7.1425e-03** | 1.976e-06 (1.15 δ) |
| 14 | 1.5859e-06 | 1.3763e-03 | 868× | +6.8637e-03 | 1.830e-06 (1.15 δ) |
| 12 | 2.2413e-06 | 1.0906e-03 | 487× | +1.7138e-03 | 2.586e-06 (1.15 δ) |
| 8 | 1.1888e-06 | 7.7156e-04 | 649× | +1.9903e-03 | 1.371e-06 (1.15 δ) |
| 14 | 4.7421e-06 | 2.1929e-04 | 46× | +3.3818e-04 | 5.470e-06 (1.15 δ) |
| 10 | 1.8854e-06 | 3.3282e-04 | 177× | +5.1388e-04 | 2.175e-06 (1.15 δ) |
| 10 | 4.6995e-06 | 9.6224e-05 | 20× | +3.7309e-05 | 5.421e-06 (1.15 δ) |
| 8 | 2.3743e-06 | 3.1738e-05 | 13× | +4.1990e-05 | 2.739e-06 (1.15 δ) |

The `delta → 0` reference is exact: its degree ladder reads
`R0 = 0.10569113` at every degree from 9 to 22 with `R+T = 1.000000000000`.
The witness that these rows are wrong is independent of the classification
rule: the `min_feature` the refusal WOULD have prescribed lands at 1.15 δ from
that limit on all 8, while the returned answer is 1–3 decades further. On the
worst row every propagating order is off in the same direction by 1.8e-04 to
1.4e-03 — a coherent systematic error, not a lost digit.

**The band is a set of isolated δ, not an interval.** 81-point LINEAR walks of
±25 % around three of the misses:

| centre | right | refused | quiet (wrong-or-grey, unrefused) | contiguous runs | run widths | max err inside | max `R+T−1` inside |
|---|---|---|---|---|---|---|---|
| deg 20, 1.7130e-06 | **0 / 81** | 75 | 6 | 5 | one 1.07e-08 step, four zero-width | **7.9667e-03** | **+9.6854e-03** |
| deg 14, 1.5859e-06 | 0 / 81 | 78 | 3 | 3 | all zero-width | 2.3631e-03 | +6.8637e-03 |
| deg 12, 2.2413e-06 | 0 / 81 | 78 | 3 | 3 | all zero-width | 1.9737e-03 | +4.0354e-03 |

and around the fix's own grey row (degree 12, δ = 5.544e-06 — the neighbourhood
open item B names), a 121-point walk of 0.7 … 1.3 × that δ:

| right | refused | quiet | contiguous runs | max err inside | max `err/δ` inside | max `R+T−1` inside |
|---|---|---|---|---|---|---|
| **3 / 121** | 108 | 10 | 10, **all zero-width** | 2.9504e-04 | 63.6 | +5.2375e-04 |

Same shape: isolated δ, ~8 % of samples, errors two decades under the refused
population. (Note the trap: at the ROUNDED δ `5.544e-06` the answer reads
err 8.58e+00 at `R+T−1` = +22.3 and IS refused — the audit's grey row is at the
exact `geomspace` float, which my S4.1 re-run of its own grid reproduces
exactly.)

Read this as the guard's FLOOR: inside the hazard band essentially nothing is
correct, the guard refuses ≈ 93 % of it, and the ≈ 7 % it returns are isolated
δ carrying errors up to **8.0e-03** at `R+T−1` up to **+9.69e-03** — 3 % under
its own bar. That is 2.4 decades better than the 0.48–8.6 the refused rows
carry, and 2.4 decades worse than the 4e-06 a correct solve reads.

The fix's open item B calls this "one row in 138 … wrong at the 1.6e-04 level".
**BOUNDED**: it is real, it is roughly ten times deeper than stated, and the
sensitivity to δ is extreme — a five-significant-figure copy of a quiet δ is a
DIFFERENT geometry that reads `R+T−1 = +1.17` where the exact δ reads
`+6.9e-03` (measured; it is why `v3_guard.py` section E reads its candidates
from JSON rather than from the printed log).

**On the conical and slant cascades** (`v10_paths.py`, 40 log δ in 3e-3…1e-6 ×
degrees 10 and 14, each against its own `δ → 0` limit; identical on both
builds) — this closes the fix's **open item E**:

| path | degree | right / wrong | refused wrong | refused right | worst wrong err (`R+T`) | misses |
|---|---|---|---|---|---|---|
| classical | 10 | 22 / 18 | 18/18 | 0/22 | 9.357e-01 (3.633) | 0 |
| classical | 14 | 21 / 19 | 19/19 | 0/21 | 1.011e+01 (28.67) | 0 |
| **conical** | 10 | 22 / 15 | **14/15** | 0/22 | **1.151e+05 (3.003e+05)** | **1** |
| **conical** | 14 | 21 / 19 | 19/19 | 0/21 | 1.244e+05 (4.046e+05) | 0 |
| slant | 10 | 22 / 18 | 18/18 | 0/22 | 9.352e-01 (3.631) | 0 |
| slant | 14 | 21 / 19 | 19/19 | 0/21 | 1.030e+01 (29.28) | 0 |

The conjunction transfers to both paths with the same one-in-forty residual
(the conical miss: δ 1.4422e-05, err 3.62e-03 = 251×, `R+T−1` +5.535e-03), and
the conical path's failures are four decades LARGER in magnitude than the
classical path's. Item E is answered: no new failure mode, same floor.

**The M2 documented energy-blind collapse is CURED and is not a live miss.**
`_perlayer_window_grids`' docstring records the audit-class taper's degree
ladder collapsing to `0.061668 / 0.623403 / 0.623395` at `|R+T−1| ~ 1e-8`. On
this build, both arms and both `min_feature` values:

| `n_slices` | `min_feature` | R(order 0), degree 6…18 | spread | max \|R+T−1\| |
|---|---|---|---|---|
| 2 | default | 0.111000 0.110723 0.110643 0.110613 0.110600 0.110593 0.110589 | 4.108e-04 | 2.39e-08 |
| 2 | 0.5 nm | 0.110880 … 0.110476 | 4.039e-04 | 6.20e-10 |
| 6 | default | 0.111976 … 0.111603 | 3.728e-04 | 8.50e-08 |

Stationary at the library default. The 2026-08-06 `_forward_growth_flip`
repair did what that file's re-pin comment says it did. **Verified cured, both
builds** — so it is not a false negative of this guard.

### S4.4 Two STRUCTURAL blind spots the fix audit does not name

1. **Anisotropy.** `_stack_provably_passive` requires `_tensor_is_passive` on
   every segment, and that function accepts only DIAGONAL tensors. So the guard
   is completely silent on any stack with an off-diagonal or out-of-plane
   permittivity — which is the entire liquid-crystal / birefringent device
   class. Measured on the O-11 fixture with the `ε_xy = 0.2` in-plane tensor
   and the same 1e-4 sliver: `R+T` = **1.437888 / 2.182956**, `provably_passive`
   = `False`, guard silent, plain `energy not conserved` warning, wrong answer
   RETURNED — i.e. exactly the pre-fix behaviour. This is conservative by
   design and the design note says so ("a payload this function does not
   understand can never widen anything"), but the fix audit's coverage
   statement does not carve it out.

2. **The ownership rule.** A thin feature owned by ONE layer is never flagged.
   The mechanism does not care who owns the walls. Measured on a two-layer
   stack whose layers each carry a liner of width `d` (`v6_2d_mortar.py` part C;
   the reference is exact — the liner vanishes as `d → 0`):

   | liner `d` (of a period) | screen fires | err vs the exact limit | err/`d` | max \|R+T−1\| | degree spread | `\|q\|max` | predictor const |
   |---|---|---|---|---|---|---|---|
   | 1e-2 | no | 1.3034e-02 | 1.30 | 2.3e-09 | 4.4e-07 | 7.900e+02 | 0.6674 |
   | 1e-3 | no | 8.7929e-04 | 0.88 | 2.0e-10 | 1.4e-08 | 7.682e+03 | 0.6490 |
   | 1e-4 | no | 8.4706e-05 | 0.85 | 3.9e-08 | 7.7e-08 | 7.663e+04 | 0.6474 |
   | 1e-5 | no | 1.5265e-05 | 1.53 | 7.9e-06 | 1.4e-05 | 7.661e+05 | 0.6472 |
   | **1e-6** | **no** | **1.0578e-03** | **1058** | **7.8e-04** | 7.0e-04 | 7.661e+06 | 0.6472 |
   | **1e-7** | **no** | **1.0536e+00** | 1.05e+07 | **3.19** | 1.1e-01 | 7.661e+07 | 0.6472 |

   At 1e-6 nothing fires at all (the closure is under the bar). At 1e-7 the
   answer is off by 1.05 in absolute per-order efficiency; the degree ladder
   reads `R+T` from **0.571** (degree 14 — SUB-unity, so not even the warning
   fires) to **4.19** (warn-and-return). The predictor constant is 0.647 here
   too, so this is the same mechanism, unscreened by construction.

### S4.5 FALSE POSITIVES — correct solves the guard refuses

**On the O-11 family: none.** 0 of 190 correct rows on the 120 × 3 grid, 0 of
243 rows in the three linear walks, 0 of 43 correct rows on each of the
conical and slant maps, and 0 among the shipped battery
(`v3_guard.py` section C): ordinary non-conforming stack, owned 1e-4 liner,
lossy substrates at Im(n) = 0.05 / 0.5 / 2.0, conical and slant mounts with and
without a sliver, and many-slice tapers at `n_slices` 16–40 and degree 6–8 —
all returned, all closing to ≤ 3e-06.

**The negative controls, independently.** Each built WITH the 1e-4 sliver
present, and `_sliver_refusal` called with a fabricated `worst = 5.0` so the
passivity test is the only thing that can stop it:

| stack | `_stack_provably_passive` | screen (a) hits | refusal at `worst = 5.0` |
|---|---|---|---|
| baseline (passive, lossless) | True | yes | **YES** |
| GAIN layer `ε = (3 − 0.5j)²` | False | yes | no |
| GAIN layer `ε = (3 − 0.001j)²` (marginal) | False | yes | no |
| LOSSY layer `ε = (3 + 0.5j)²` | True | yes | **YES** |
| GAIN substrate `n = 1.5 − 0.01j` | False | yes | no |
| LOSSY substrate `n = 1.5 + 0.5j` | True | yes | **YES** |
| ABSORBING superstrate `n = 1 + 0.01j` | False | yes | no |
| off-diagonal in-plane tensor | False | yes | no |

All correct, and the screen (a) fires on every one of them — so the passivity
conjunct is what is doing the gating, exactly as designed.

**Conjunct (a) is sound by construction.** Because a cell is manufactured only
when no single layer owns both its walls, a flagged cell at ratio > 100 always
means two DIFFERENT layers placed walls within `own/100` of each other — a
genuine sliver. The false-positive exposure is therefore entirely in (b), which
is where the measurement below finds it.

**Off it: 110 in 648.** Conjunct (b) reads super-unity as a theorem violation;
on a provably passive stack it is equally often ordinary under-convergence.
`v9_falsepos.py`, identical on both builds:

* **Part A — the premise, attacked directly.** 960 SLIVER-FREE stacks that
  `_stack_provably_passive` accepts (6 lossy substrates × 4 superstrate indices
  × 5 angles × 4 layer permittivities × degrees 8 and 12). All 960 pass the
  passivity test; **26 of them read above the bar**, worst
  `R+T − 1 = 3.8140e-02` at `n_sub = 1.5+0.05j`, `n_sup = 2.5`, θ = 1.3 rad,
  ε = 12, degree 8. No many-slice quasi-resonance is required: a lossy
  substrate at a large angle and a modest degree is enough.

* **Part B — the census.** Over 648 realistic staircase configurations (3 lossy
  substrates × 2 superstrate indices × 3 angles 1.2–1.45 rad × degrees 6/8/10 ×
  2 permittivities × 2 or 4 slices × δ ∈ {3e-3, 1e-3, 3e-4} — i.e. wall steps
  of 0.36–3.6 nm on a 1.2 µm period), **110 (17.0 %)** are rows whose answer
  tracks the exact `δ → 0` limit to within the fix's own CORRECT rule
  (`err/δ` between **0.35 and 8.77**, bar 10) and which the guard nevertheless
  REFUSES, naming the sliver. `R+T` among them: **1.01008 … 1.12328**.

* **Part C — and the remedy is the wrong one.** On
  `n_sub = 1.5+0.05j, n_sup = 2.5, θ = 1.2, ε = 12, degree 6, δ = 1e-3`:

  | | `R+T` |
  |---|---|
  | unguarded, snap off (the refused solve) | **1.03557** |
  | **remedy (1)** — the prescribed `min_feature = 2.4e-09 m`, named FIRST and with a number | **1.03559** — solves, refusal SILENCED, answer unchanged |
  | remedy (2) — coincident walls (δ = 0) | 1.03593 |
  | **remedy (4)** — raise degree, sliver untouched | 8 → **1.00272**; 10 → 1.00023; 12 → **1.000000**; 14 → 1; 16 → 1 |

  Remedy (1) removes conjunct (a) — the cell is gone, so the guard cannot fire
  — and returns the same unreliable number under the plain warning. A caller
  who follows the message in the order it is written is led away from the
  operative cause.

**What the harm is, stated precisely.** These solves are UNDER-CONVERGED: a
degree-6 answer reading `R+T` = 1.036 is unreliable however the walls are
placed, and the pre-fix response (a warning naming `n_slices` / `degree`) was
the right one. The defect is not that a good answer is thrown away — it is that
the super-unity is ATTRIBUTED to the sliver, which measurably did not cause it
(the answer moves by 0.35–8.8× the physical wall shift, i.e. by the physical
amount), and that the remedy named first removes the ATTRIBUTION rather than
the error. Conjunct (a) is doing exactly what S4.2 of the fix says it does —
confining the behaviour change — but on this family the population it confines
the change to is one where (b) is not reading a theorem violation.

This is the fix's **open item F**, quantified, and materially wider than its
statement of it: F says "a tapered staircase carries BOTH", and mitigates with
remedy (4). The measurement says (i) no taper is needed, (ii) the affected
population is 17 % of an ordinary oblique-incidence-with-lossy-substrate
parameter box, and (iii) the first-named remedy actively hides the problem.

**Suggested change (not made here — it is a behaviour change, not a
follow-up), and it is MEASURED.** One extra solve, at the point where the
library is about to raise anyway, separates the two causes exactly:

> re-solve on the grid the prescribed `min_feature` would produce.
> If the super-unity VANISHES, the sliver was the cause. If it SURVIVES, it
> was not — name `degree` / `n_slices` instead.

Scored two-sided in `v11_discriminator.py`, identical on both builds:

| arm | rows | discriminator correct |
|---|---|---|
| TRUE POSITIVES — the O-11 hazard band, degrees 12/14/20 (`R+T` 2.17 … 23.4) | 8 | **8 / 8** (`R+T` → exactly 1 after the snap) |
| TRUNCATION — the S4.5 family (lossy substrate, large angle, low degree, a harmless sliver) | 63 | **63 / 63** (`R+T` stays above the bar) |

Cost: 205 solves in **19 s** (Windows) / 20 s (WSL), i.e. ~0.1 s for the one
extra solve — paid only on a stack that is already being refused. With it, the
message can name the operative cause and order its remedies by it.

---

## S5. The remedy, and open item A — `v4_remedy.py`, `v5_deadband.py`

### S5.1 The remedy is sound; its BAR is not scale-free

Three fixtures × four degrees (10, 12, 14, 18) × the δ that actually refuse,
each scored against its own exact `δ → 0` reference:

| fixture | period / λ / θ | continuity slope (measured, δ = 3e-3/1e-3/3e-4) | `err/δ` after the prescribed snap | inside `2 δ`? | snapped closure |
|---|---|---|---|---|---|
| O-11 | 1.2 µm / 0.85 µm / 0.15 | 1.078 / 1.128 / **1.146** | **1.1524 – 1.1535** | 10/10 ✓ | ≤ 4.0e-14 |
| mine, telecom | 1.55 µm / 1.31 µm / 0.08 | 0.988 / 1.025 / **1.038** | **1.2073 – 1.2077** | 7/7 ✓ | ≤ 1.5e-06 |
| **mine, visible** | 0.9 µm / 0.62 µm / 0.21 | 4.251 / 4.391 / **4.442** | **3.5127 – 3.5147** | **0/9 ✗** | ≤ 4e-14 |

**The remedy itself is CONFIRMED on all three**: the snapped answer's distance
from the exact limit is BELOW the structure's own measured continuity slope
times δ in every one of the 26 rows. The audit's 1.152–1.154 is reproduced
exactly on the fixture it was measured on.

**The `err ≤ 2 δ` bar is REFUTED as scale-free.** The fix derives it from "the
snap moves each wall by at most δ, so the snapped geometry cannot be further
from the limit than the original was", which compares a GEOMETRIC displacement
with an EFFICIENCY error and silently assumes `dR/dx ≈ 1`. `dR/dx` is a device
property: 1.15, 1.04 and 4.44 on these three. `test_the_prescribed_min_feature_
lands_within_the_derived_bar` passes only because it runs the O-11 fixture. Its
companion bar `closure < 1e-6` is likewise fixture-specific — my telecom
fixture reads 1.537e-06 at degree 10.

The correct statement, and what
`test_the_remedy_lands_on_the_structures_own_continuity_slope` now asserts, is
`err ≤ 2 · slope_measured_here · δ`.

### S5.2 Open item A, reproduced — and FIXED (follow-up 1)

`_pmm_union_grid` merges a pair when `d < min_feature`. `d` is a difference of
cumulative sums of period fractions, so two pairs a caller made the same width
land on either side of a `min_feature` set to that width.

**Reproduced on three fixtures, both builds.** At δ = `min_feature` = 1e-5:

| | value |
|---|---|
| `sep_left = A − (A − δ)` | `1.0000000000010001e-05` — **> mf** |
| `sep_right = (B + δ) − B` | `9.99999999995449e-06` — **< mf** |
| union cells (pre-fix) | **4** — one pair merged, one left |

and the resulting geometry, which nobody asked for, solves:

| fixture | asymmetric grid | both snapped | neither snapped |
|---|---|---|---|
| O-11 | err **9.305e-01** (`R+T` 3.613) | err 1.153e-05 (`R+T` 1) | err 4.790e-01 (`R+T` 2.173) |
| mine, telecom | err **2.397e+00** (`R+T` 10.14) | err 1.208e-05 (`R+T` 1) | err 1.035e+02 (`R+T` 330.9) |
| mine, visible | err **1.640e+01** (`R+T` 47.41) | err 3.515e-05 (`R+T` 1) | err 2.147e-02 (`R+T` 1.097) |

**How often it bites.** Over a 121-point multiplicative walk of δ through
`min_feature` (0.5× … 2.0×), exactly **1** row is asymmetric on each fixture —
the δ = `mf` row itself. It bites only when a wall spacing lands within ~1e-12
relative of `min_feature`; when it does, the answer is catastrophic.

**`<=` is NOT the fix** (measured): the pair STRADDLES `mf`, it does not sit on
it, so `<=` still merges 1 of 2.

**What ships.** An absolute round-off DEADBAND, sized on the measurement:

```
_WALL_SNAP_DEADBAND = 16.0 * np.finfo(float).eps      # 3.5527e-15 of a period
...
mf_thr = mf - _WALL_SNAP_DEADBAND
if d < mf_thr and interior and not (out_o[-1] & ow):
```

* **Sizing.** The walls live in `[0, 1]`, so the deadband is absolute in period
  fractions. Worst `|d_computed − min_feature|` over **120,000** random
  two-layer layouts × six `min_feature` decades: **1.565e-16** = 0.70 ULP of
  1.0. 16 ULP is **23×** that and six decades below the function's own 1e-9
  fractional dedup `tol`.
* **Direction.** It SUBTRACTS, so `min_feature` keeps the "closer than" meaning
  the docstring states, and the change can only ever merge FEWER pairs than
  before — it can never move a wall the pre-fix code left alone. (The opposite
  direction would have changed 4 cases instead of 11 and would hand the caller
  the snapped answer at the threshold rather than a refusal; it violates the
  documented "closer than" and can newly move a wall, so it was not taken. The
  trade is recorded here so the owner can flip it.)
* **Result at the threshold.** Union cells go 4 → 5 on the asymmetric cases and
  3 → 5 on the cases where both had merged; either way the geometry is now the
  one the caller described, and where a sliver remains the refusal catches it
  and prescribes `2 w`.

**Two-sided gate** (`v5_deadband.py`, the same deterministic 11,418-case set run
against this tree and against the read-only main clone):

| | |
|---|---|
| cases | 11,418 (3 wall layouts × 6 `min_feature` × 401 log δ from 0.01× to 100× mf, + 4,000 random two-layer layouts, + 200 random 6-slice tapers) |
| differing | **11** |
| … of which ON the threshold (`\|d − mf\| ≤ 16 ULP`) | **11 / 11** |
| … OFF the threshold | **0** |
| off-threshold cases bit-identical | **11,407 / 11,407** |

---

## S6. The 2-D stacks and the mortar route — `v6_2d_mortar.py`

### S6.1 The two shipped claims

* **`PMM2DStackPure`'s shared grid.** Every cell is `period/N`, aspect ratio
  **exactly 1.0** at N = 8, 12, 24, 64 and 129 (odd sizes included). Stronger than the audit
  states: `add_layer` REFUSES a second patterned layer on a different lattice —
  *"all patterned layers must share ONE common (Nx, Ny) grid (the union-grid
  constraint of the pure staggered cascade)"* — so a non-trivial union is not
  merely aspect-1, it cannot be constructed. **CONFIRMED.**
* **`PMM2DStackHybrid` has no union grid.** Two layers on 6×6 and 8×8 lattices
  are accepted and solved. Its low-order truncation reads `R+T` =
  **1.0103 / 1.0072** at degree 7 / 3 orders and produces the plain
  `energy not conserved` warning with no SLIVER text — which is exactly what
  `stack2d.py` passing `stack=None` is for. **CONFIRMED.**
* **"`_pmm_union_grid` has no 2-D caller."** Enumerated:
  `lumenairy/elements/pmm/{stack,conical,_jax_stack}.py` and `_core.py`'s own
  `_perlayer_window_grids`. No 2-D module calls it. **CONFIRMED** — and note
  that two of the three callers are the CONICAL and JAX paths, which is why
  S4.3's conical map matters and why the JAX path (where
  `_stack_provably_passive` returns `False` on traced payloads) is the one
  place the guard is inert by construction.

### S6.2 A sliver INSIDE one layer's own non-uniform grid — the mortar claim

`_stag_walls_spec` requires only that walls be strictly increasing, so a caller
CAN place two of one layer's walls arbitrarily close through
`add_layer(x_walls=...)` or `add_tapered_pillar`. The reference is exact: as
the two walls approach each other the thin ε_p stripe vanishes and the layer
becomes a homogeneous ε_h slab, so `err` must fall linearly in `d`.

**One layer** (`n_modes` 4…8, `layer_grids='per-layer'`):

| `d` (of a period) | refused | `err` vs the exact limit | `err/d` | max \|R+T−1\| | `n_modes` spread |
|---|---|---|---|---|---|
| 1e-1 | 0/5 | 5.357e-01 | 5.36 | 1.97e-02 | 4.2e-03 |
| 1e-2 | 0/5 | 1.775e-02 | 1.77 | 1.23e-03 | 4.1e-03 |
| 1e-3 | 0/5 | 4.645e-04 | 0.46 | 8.98e-06 | 4.1e-05 |
| 1e-4 | 0/5 | 3.463e-05 | 0.35 | 4.53e-07 | 4.5e-07 |
| 1e-5 | 0/5 | 3.356e-06 | 0.34 | 7.67e-07 | 1.4e-08 |
| 1e-6 | 0/5 | 3.625e-07 | 0.36 | 7.99e-07 | 7.9e-09 |
| **1e-7** | **5/5** | — | — | — | `LinAlgError: Singular matrix` |

**Two mortar-coupled sliver grids** (each layer with its own sliver at a
DIFFERENT position — the configuration the shared union grid turns into a
cross-layer sliver), `n_modes` 4/6/8:

| `d` | refused | `err` | `err/d` | max \|R+T−1\| | `n_modes` spread |
|---|---|---|---|---|---|
| 1e-2 | 0/3 | 1.774e-02 | 1.77 | 1.20e-03 | 2.6e-03 |
| 1e-3 | 0/3 | 4.644e-04 | 0.46 | 8.95e-06 | 3.6e-05 |
| 1e-4 | 0/3 | 3.463e-05 | 0.35 | 4.55e-07 | 4.3e-07 |
| 1e-5 | 0/3 | 3.356e-06 | 0.34 | 7.68e-07 | 1.4e-08 |
| 1e-6 | 0/3 | 3.625e-07 | 0.36 | 7.99e-07 | 7.9e-09 |

**`add_tapered_pillar`** with tip widths 1e-2 / 1e-4 / 1e-6 of the period, 4
slices: all built, all solved, `R+T` = 1.0000015 throughout, no warning.

**Verdict: the mortar route IS the safe one for this hazard, and by four
decades.** Where the shared 1-D union grid is catastrophically wrong at
`w` = 1e-4 (err 0.48, `R+T` 2.17), the per-layer mortar at `d` = 1e-4 reads
`err/d` = 0.35 with `|R+T−1|` = 4.5e-07 and an `n_modes` spread of 4.5e-07 —
converged, continuous and closing — and it stays that way to 1e-6. The claim
the mortar agent was given is **CONFIRMED**, on a reproducer built to break it.

**One defect to report.** At `d` = 1e-7 the per-layer basis raises
`numpy.linalg.LinAlgError: Singular matrix` from inside the solve, with no
message naming the geometry, the offending walls, or a remedy. That is loud
rather than silent — the important property — but it is the one place on this
path where a caller gets no actionable text. Reproducer:
`v6_2d_mortar.py::part_b` at `d = 1e-7`, and
`tests/unit/test_verify_pmmstack_sliver_walls.py::
test_the_mortars_within_layer_sliver_fails_LOUDLY_when_it_fails`.

---

## S7. Durability of `tests/unit/test_fix_pmmstack_sliver_walls.py`

`v7_durability.py` re-measures the quantity behind every bar in that file, on
both builds. All 19 tests pass on both (S8).

| bar | origin | Windows | WSL | headroom |
|---|---|---|---|---|
| `abs(total(ref) − 1) < 1e-6` (premise) | the `δ → 0` reference is exact | 3.220e-14 | 3.220e-14 | 3.1e+07× |
| `0.5 < slope < 2.0` | the O-11 fixture's continuity | 1.078 / 1.128 / **1.146** | 1.078 / 1.128 / 1.146 | 2.16× / **1.75×** — and it is a FIXTURE property (my visible fixture reads 4.44 and would fail) |
| `e > 1000·max(slope)·1e-4` | derived at runtime | 0.4789 vs 0.1146 | same | 4.18× |
| `e > 0.1` | magnitude | 0.4789 | 0.4789 | 4.79× |
| `total(bad) − 1 > 1.0` | magnitude | **1.1730** | **1.1730** | **1.17×** — thinnest bar in the file |
| ladder `right ≥ 4`, `wrong ≥ 4` | composition | 7 right, 6 wrong, 0 grey | same | 1.75× / 1.50× |
| `max(right) ≤ bar/100` | rule 5, re-derived | **3.151e-07** | 3.151e-07 | **317×** on the ladder — **1.01× on a 120-row grid** |
| `min(wrong) ≥ bar·30` | rule 5, re-derived | **1.1726** | 1.1726 | **3.91×** on the ladder — **FAILS by 42× on a 120-row grid** |
| `0.52 < const < 0.78` | measured predictor | 0.6424 … 0.6736 | identical to 15 digits | 1.23× / 1.16× |
| `spread < 1.10` | measured predictor | **1.0485** | 1.0485 | **1.049×** — second-thinnest; cross-build spread ≈ 0, so it is algorithm-fragile, not build-fragile |
| `0.9 < quoted/measured < 1.1` | the message's own \|q\| | 1.00344 (76 900 vs 76 636) | identical | ~29× in log |
| `val > 1e-4·P` | structural (`mf = 2 w P`) | 2.4e-10 vs 1.2e-10 | same | 2.00× |
| `err ≤ 2 δ` (remedy) | mis-derived, see S5.1 | 1.1535 | 1.1535 | 1.73× on O-11, **fails at slope > 2** |
| `closure < 1e-6` (remedy) | fixture | 4.019e-14 | 4.108e-14 | 2.5e+07× on O-11, **1.537e-06 on my telecom fixture** |
| per-layer gap `< 1e-12` | the caveat | **0.0** | **0.0** | exact |

**The file's most important durability defect** is the pair of rule-5 bars:
they are re-derived at runtime, which is right, but on a 13-row `_LADDER` whose
composition is what makes the separation hold. The same two quantities on a
120-row grid of the same family read 9.87e-05 and 7.14e-03, so the test would
pass the first assertion by 1.3 % and fail the second by 42×. Per
TESTING_STANDARDS rule 5, the envelope is the family's, not the sample's.

Two constants are thin but not per-build fragile (`> 1.0` at 1.17×, `spread <
1.10` at 1.049×): both quantities agree between the two builds to 9–15
significant figures, so their pass/fail boundary is far outside the cross-build
spread. They would move under an intentional algorithm change, which is the
gate working.

### S7.1 The module disarm in `test_m1_conditioning_guard.py`

* **What it is.** `@pytest.fixture(autouse=True)` with no `scope=`, declared at
  module level — i.e. a FUNCTION-scoped autouse fixture that applies to every
  test in that module and restores the previous value in a `finally`. The fix
  audit calls it "a MODULE-scope autouse fixture"; that is its reach, not its
  scope. The distinction matters because a function-scoped fixture restores
  after EVERY test, which is what makes it leak-proof.
* **Is it the narrowest correct scope?** I made a copy of the file with the
  disarm inverted (guard ARMED) and ran it in four configurations. **Exactly 2
  of 27 tests trip the guard in every one**:
  `test_rcond_of_hsup_would_have_been_the_wrong_instrument` and
  `test_t3_3_fail_before_reproduces_the_over_capacity_draw`.

  | configuration | failures |
  |---|---|
  | Windows, `OMP/OPENBLAS/MKL = 1` | the same 2 |
  | Windows, 2 threads | the same 2 |
  | Windows, 4 threads | the same 2 |
  | WSL, 1 thread | the same 2 |

  So the shipped justification — "which arms trip it is a BLAS fact, so the fix
  is a MODULE-scope fixture, not a per-arm patch that a second build would
  defeat" — is **not reproduced**: the trip set is stable across two builds and
  three thread counts, and a two-test fixture would have sufficed on all four.
  The module-wide fixture is still the right call for a different reason
  (nothing in that file asserts sliver behaviour, its whole purpose is to drive
  an over-capacity staircase, and a narrower fixture must be re-audited every
  time an arm is added), and it costs nothing. **BOUNDED**: correct decision,
  unsupported stated reason.
* **Can it leak?** No. An autouse fixture declared in a test module applies
  only to that module, and the value is restored in `finally`, so it cannot
  survive a test, let alone a file. Verified operationally:
  `test_the_m1_module_disarm_is_function_scoped_and_restores` in my file asserts
  `PMM_SLIVER_GUARD is True` and that the refusal is live, and the three files
  run green in one process in either order.
* **What DOES leak, and it is not the guard.** Both sliver test files set
  `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` at import via
  `os.environ.setdefault`, and nothing restores them. They persist for the whole
  pytest session. This is a pre-existing repo-wide pattern (the files' own
  comments cite `test_v5_13_0_pmm_tapered`), and `setdefault` means the binding
  only happens if the variable is unset and only bites if the file is imported
  before numpy loads its BLAS — so in a mixed run it is often inert, which is
  itself the hazard: the thread count a file believes it pinned depends on
  collection order. Noted, not changed.

---

## S8. The runs

All on Windows unless stated, one BLAS thread, `PYTHONPATH=/c/tmp/lum_vsliver`.

| run | result |
|---|---|
| `tests/unit/test_verify_pmmstack_sliver_walls.py` (new) | **15 passed**, 19.89 s (at 14 tests: 16.59 / 17.84 / 15.83 s over three runs) |
| `tests/unit/test_fix_pmmstack_sliver_walls.py` | **19 passed**, 6.07 s / 6.76 s |
| `tests/unit/test_m1_conditioning_guard.py` + the fix file | **46 passed**, 9.45 s |
| the three files, **WSL** | **61 passed**, 32.89 s (15 + 19 + 27) |
| the three files, Windows, M1 FIRST (so the leak check follows the disarm) | **61 passed**, 31.10 s |
| the same three files with the M1 disarm INVERTED (verification arm, scratch copy) | **2 failed, 25 passed** — the same 2 tests on Windows at 1 / 2 / 4 BLAS threads and on WSL at 1 |
| every test file importing `PMMStack` (`grep tests/ --include='*.py' -l PMMStack`, **41** files incl. the two new mortar files and mine), Windows, 1 thread, slow markers included | see below |
| `ruff check lumenairy/ tests/` | **All checks passed** (also clean on `validation/probe_verify_sliver/`) |

`.test_durations` spliced with the 15 measured Windows timings.

---

## S9. Defects raised

| # | Defect | Where | Status |
|---|---|---|---|
| **V-1** | The guard REFUSES correct solves whenever super-unity comes from truncation rather than the sliver — **110 of 648** ordinary staircase configurations; and its FIRST-named remedy silences the refusal without changing the number (1.03559 vs 1.03557) while the operative cause is degree. | `v9_falsepos.py`, `test_the_refusal_fires_when_the_super_unity_is_TRUNCATION_not_the_sliver` | OPEN — this is open item F, quantified and wider than stated. A MEASURED discriminator (one extra solve, 8/8 and 63/63 two-sided) is in S4.5 / `v11_discriminator.py`. |
| **V-2** | The fix's separation claim ("3.39 decades below / 2.06 above") is a property of its 46-point sample. A 120-point grid of the same family reads 9.87e-05 / 7.14e-03, the latter BELOW the bar. `test_the_bar_has_decades_of_gap_on_both_sides_measured_here` re-derives its bars on the sample that makes them hold. | S4.1, S7 | OPEN — re-derive the bars on a denser ladder or restate the claim. |
| **V-3** | The M2 audit-class coated taper's attribution ratio is **12.11**, not ~1.7e+02; `_cross_layer_sliver` returns `None` on it. The audit's S4.2 row is wrong and the risk it implies is inverted. | S4.2 | OPEN — documentation correction; the device class is outside conjunct (a). |
| **V-4** | The remedy's `err ≤ 2 δ` bar (and its `closure < 1e-6` companion) are fixture properties. Measured `err/δ` = 3.51 and closure 1.54e-06 on fixtures of mine. The audit's derivation compares a wall displacement with an efficiency error. | S5.1 | OPEN — restate as `err ≤ 2 · slope_measured · δ` (my test does). |
| **V-5** | The guard is structurally silent on any stack with an off-diagonal or out-of-plane permittivity (`_tensor_is_passive` accepts diagonal only) — measured `R+T` = 2.183 with the same sliver, warn-and-return. | S4.4 | OPEN — conservative by design; the coverage statement should carve it out. |
| **V-6** | The ownership exemption is silent at widths where the same mechanism is already catastrophic: a within-layer liner at 1e-6 reads `err` 1.06e-03 with nothing firing, at 1e-7 `err` 1.05 with `R+T` from 0.571 (nothing fires) to 4.19 (warn-and-return). | S4.4 | OPEN — the rule is deliberate; the width at which it becomes dangerous is now measured. |
| **V-7** | The per-layer 2-D basis raises a bare `numpy.linalg.LinAlgError: Singular matrix` on a within-layer sliver at 1e-7 of a period — no geometry, no remedy. | S6.2 | OPEN — message quality only; the failure is loud. |
| **V-8** | The false-negative floor is ~10× deeper than open item B states: 8 misses in 660 samples, errors to 2.80e-03 at `R+T−1` +7.14e-03; the conical path adds 1 in 240. | S4.3 | OPEN — item B, re-measured. |

Follow-ups MADE here (the two the brief authorised):

| | |
|---|---|
| **follow-up 1** | open item **A** — `_WALL_SNAP_DEADBAND` in `_core.py`. Gated two-sided over 11,418 cases against the pre-fix clone: 11 differ, all on-threshold; 11,407 off-threshold identical. S5.2. |
| **follow-up 2** | open item **C** — `stack.py`'s `min_feature` comment no longer promises a report `_pmm_union_grid` does not make; it names the two reports that exist (the snap warning; the conjunction refusal) and the gap between them. |

Open items **D** (the union-forming route of the mortar work) is untouched and
still owned by the mortar build; S6.2 supplies the measurement that the
NON-union mortar route is safe, which is the other half of that item.
Open item **E** is closed by S4.3's conical/slant map. Open item **B** is
re-measured in S4.3 and remains open. Open item **F** is V-1.

---

## S10. What I could not verify

* **JAX / traced paths.** `_stack_provably_passive` returns `False` on traced
  payloads by construction, so the guard is inert there; I did not run a JAX
  arm (no traced fixture in my set), so "inert" is read from the code, not
  measured.
* **The 2-D `stack2d.py` call site keeping `stack=None`.** Verified by reading
  the diff and by the hybrid's behaviour (S6.1), not by an independent
  enumeration of every 2-D call path.
* **`solve_vs_wavelength` under the guard at more than one worker.** My
  bit-identity fixture runs `max_workers=1`; the guard raising from inside a
  thread-pool worker is not exercised anywhere I ran.
* **The fix's claim that the geometric screen costs "nothing" on a healthy
  solve.** True by inspection (it is reached only after the super-unity test),
  not timed.
* **The PREPARED path (`_PreparedPMMStack.solve`) reaching the refusal on a
  stack that actually carries a sliver.** My bit-identity set exercises
  `prepare()` on a healthy stack, and the fix's own file covers the
  no-wavelength message degradation; the prepared path's refusal on a sliver
  stack is covered by reading (`stack=self._st`), not by a run of mine.
* **`stabilize='slices'` interacting with the refusal.** The guard raises
  before `_slices_consensus_check` runs; I did not construct a stack where the
  two would disagree.
* **Whether V-1's 17 % figure generalises beyond the parameter box I scanned.**
  It is a census over one product of realistic values, not a measure over
  device space.

---

## S11. Commits (branch `verify/sliver-walls`)

| | |
|---|---|
| `33b3dc4` | `probe(verify)` — tasks 1–2: 21 bit-identity fixtures of my own + the mechanism re-measurement |
| `88c0763` | `probe(verify)` — tasks 3–5: the guard attacked both ways, the remedy, the 2-D/mortar route |
| `5fb8cfb` | `fix(pmm)` — follow-up 1 (the `min_feature` deadband, open item A) and follow-up 2 (open item C's comment) |
| `bb303ce` | `test(pmm)` — `tests/unit/test_verify_pmmstack_sliver_walls.py` |
| `28be347` | `docs(pmm)` — this report + the probe README |
| `ae82b2d` | `changelog` — the two follow-ups and the verification, in `[Unreleased]` |
| `50e6db2` | `docs(pmm)` — the grey-row band, the negative-control census, what the false positives cost |
| `c996c08` | `test(pmm)` — the floor test never SKIPS (TESTING_STANDARDS rule 4) |
| `288ae12` | `probe(verify)` — the measured discriminator for open item F |

Probes: `validation/probe_verify_sliver/` — `v1_bitid.py` + `v_fixtures.py`,
`v2_mech.py`, `v3_guard.py`, `v4_remedy.py`, `v5_deadband.py`,
`v6_2d_mortar.py`, `v7_durability.py`, `v9_falsepos.py`, `v10_paths.py`,
`v11_discriminator.py`, and their JSON on both builds.
