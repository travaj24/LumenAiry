# VERIFY WP-C1 ROUND 2 -- the re-verification of the round-2 closures

Independent adversarial re-verification of `feat/c1-gray-edge-round2`
(10 commits `7f21eb2c`..`59a284dd` on `7ea01ede`), whose addendum "Round 2
(VERIFY-WP-C1)" in
[`WP-C1_GRAY_EDGE_REPORT.md`](WP-C1_GRAY_EDGE_REPORT.md) claims to close the
four defects D1-D4, the recorded items and the one surviving mutant M3 of
[`VERIFY_WP-C1.md`](VERIFY_WP-C1.md).

Everything below was **re-measured**, never read off the addendum.  Both
builds, every time: Windows py3.14.6 / numpy 2.4.4 / jax 0.11.0 / CuPy 14.0.1
and WSL py3.12.3 / numpy 2.4.6 / jax 0.10.2, both scipy-openblas, both with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line and `lumenairy.__file__` printed by every probe.  The PRE trees are this
re-verification's own `git archive 49ddf4bd` (pre-C1) at `C:/tmp/lum_vc1b_pre49`
and `git archive 7ea01ede` (pre-round-2) at `C:/tmp/lum_vc1b_pre7e`, each run
from inside itself; nothing is compared against a working copy.  Every mutant
is a fresh `git archive 59a284dd` extraction with one edit, run from its own
tree.

Probes and JSON: [`validation/probe_verify_c1_round2/`](../../../../validation/probe_verify_c1_round2/).
Decision tests: `tests/unit/test_verify_c1_round2.py` (24 ids, 16 pass +
8 strict xfail, 16 s, identical on both builds).

**SHIP the C1 chain, with three small defects to close and two documentation
corrections.**  D2, D3, D4, M3 and every recorded item are closed and every
number round 2 prints is reproducible -- several of them to sixteen digits on
both builds.  **D1 is closed for the element shape round 2 measured and only
for that shape**: an `'aperture'` element whose `params` do not resolve still
carries an illegal rim keyword silently past both JAX routes while the NumPy
chain raises (7 of 7 element shapes, both builds).  Two smaller gaps in the
same guard: `edge_samples=True` is silently accepted as the pre-5.49 rim, and
the `int()` refusal family raises a message that names neither the function
nor the keyword.  None of the three is a wrong field; all three are refusal
contracts.

Round 2 also left two things on the "could not measure" list that this round
DID measure, and one of them changes an answer: the downstream boolean-cast
exposure moves a plotted radial-RMS curve by up to 7.3 % and a wrapper merit's
integration by 0.77 %, so it wants a Migration line.

---

## 1. Verdict table

| # | round-2 claim | verdict | this re-verification's numbers |
|---|---|---|---|
| D1a | ONE guard `_validate_edge_kwargs`, THREE call sites, no duplicated guard | **CONFIRMED** | exactly 1 `def`, exactly 3 calls (`elements.py:410`, `system.py:1455`, `system.py:1645`) plus the import at `system.py:41`; both refusal MESSAGES occur in `lumenairy/elements/elements.py` and nowhere else in the package.  Pinned by `test_verify_c1r2_there_is_exactly_one_rim_guard_with_three_call_sites` |
| D1b | same `(raised, message)` on NumPy / eager JAX / jit'd JAX for the 8 illegal dicts | **CONFIRMED, and wider than claimed** | 24-row census (round 2's 8 plus 16 of my own) x 4 entry points (`apply_aperture` as well as the three chain routes): **0 rows split**, and the whole `rows` object is byte-identical WIN vs WSL.  Fail-before on `git archive 7ea01ede`: **5 rows split**, not 2 -- `edge_samples=2.5` and `'4'` on the VERDICT, and `edge=None`, `edge=b'gray'`, `edge_samples=False` on the MESSAGE alone (the jit route's `str()`/`int()` printed `got 'None'.`, `got "b'gray'".`, `got 0.`).  Round 2's table under-claims its own fix |
| D1c | my own illegal spellings | **CONFIRMED except two, filed as R2 / R3** | `edge_samples` `-1` refused; `numpy.int64(4)` **legal on all four routes** (one answer, as it should be); `edge='Gray'`, `b'gray'` refused; `numpy.str_('gray')` legal on all four; `edge_samples` without `edge` legal; **`edge_samples=True` ACCEPTED as 1** (R2) while `False` is refused; `None` / `[4]` / `4+0j` refused with a RAW `TypeError` naming neither `apply_aperture` nor `edge_samples` (R3) |
| D1d | the `_EDGE_UNSET` sentinel separates "unnamed" from `None` | **CONFIRMED** | omitting both returns `None` and raises nothing; `edge=None` raises `ValueError` with `apply_aperture`'s own message on all four entry points; `edge_samples=None` raises (R3's raw `TypeError`) on all four.  Same on both builds |
| D1e | the signature's `str()`/`int()` coercions cannot serve one element for another | **CONFIRMED, two-sided** | in ONE process: `edge_samples` `4` / `4.0` / `numpy.int64(4)` -> one digest `f5b52cf0842fd9bd`; `1` / `True` / `numpy.True_` / `1.0` -> one digest `be29d4edc8ba78fc`; the two groups DIFFER.  `'4'` can no longer collide because it is refused.  Fail-before at `49ddf4bd`: both groups read `be29d4edc8ba78fc`, i.e. the element's value reached nothing |
| D1f | reverting the call reddens 6 ids | **CONFIRMED on both builds** | `if kw: _validate_edge_kwargs(**kw)` -> `return kw` on a fresh archive of `59a284dd`: **6 failed, 53 passed** on Windows AND on WSL, and the six are exactly the ids round 2 names |
| **D1 (open)** | "one element dict is accepted by all three routes or refused by all three" | **REFUTED for unresolvable params -- defect R1** | 7 of 7 element shapes with no usable `params` split on both builds: NumPy chain raises, both JAX routes return a field.  **6 of those 7** read `identical=True` on `git archive 49ddf4bd` (the seventh, `params={'diameter': None}`, already split there for an unrelated reason -- the NumPy arithmetic on a `None` diameter raises), so the rim split is WP-C1's own and round 2 did not close it |
| D2a | the three ratios 5.804555 / 8.815875 / 1.992823, identical to 16 digits on both builds | **CONFIRMED** | my own code reads `5.804555200387276` / `8.815874781721663` / `1.992822748038388`, printed to 17 digits and **character-identical on Windows and WSL** |
| D2b | the bar is DERIVED in the docstring and two-sided | **CONFIRMED as a bar; its stated DERIVATION is refuted -- defect R4** | 1.5 sits 1.3285x below the smallest reading and 1.5x above the degenerate 1.0, both stated in the docstring: a gap on both sides, as `TESTING_STANDARDS` rule 5 asks.  But "a 145-pixel rim is already well sampled, so the staircase it has to beat is the mildest of the three" is false: a 30-point scan of that fixture's neighbourhood finds **4 ratios below 1.0**, the worst **0.003602** at `dy/dx = 1.0, offset 0.23`, where the hard arm's SIGNED area error is `+2.19e-07` (a zero crossing) against the grey arm's `+6.08e-05` |
| D2c | M5 makes all three params red; unmutated 36 passed | **CONFIRMED** | M5 (`edge='hard'` falls through to the grey mask) on a fresh archive: `test_audit2609_a8_verify.py` **4 failed, 32 passed**, and all three `..._beats_hard_at_anamorphic_and_offset_rims` params are among the four.  Unmutated: **36 passed** |
| D3a | the restated doc claim and the four-digit ladder | **CONFIRMED to every printed digit, on both builds** | RS hard `1.680 / 0.203 / 1.359`, gain `9.4555x`, mean **`1.0804`**; RS gray `1.443 / 2.602 / 1.766`, `56.1279x`, **`1.9369`**; HF hard `1.674 / 0.202 / 1.337`, `9.2699x`, **`1.0709`**; HF gray `1.737 / 2.133 / 1.620`, `44.9471x`, **`1.8301`**.  Default bit-identical to `['gray']` on both kernels.  WSL reads every digit the same.  The alias threshold is cleared at every N and recorded |
| D3b | the 1.06 -> 1.0709 correction | **CONFIRMED in the test docstring and the CHANGELOG; NOT in the report of record -- recorded in sec. 4** | `test_verify_c1_gray_edge.py:485-490` quotes 1.0709 and names the slip; the CHANGELOG restatement quotes the corrected value.  `VERIFY_WP-C1.md:73` still prints `**1.06**` with no erratum |
| D3c | "the rise is that optic's" | **CONFIRMED as scope, UNDERSTATED as frequency -- defect R5** | on a THIRD optic (lambda = 532 nm, a = 150 um, window 900 um, z = 30 / 15 mm) the hard arm RISES on the last refinement by a factor **9.3** (RS: `7.6432e-05 -> 7.1042e-04`, step order **-3.216**) and **9.1** (HF, `-3.188`).  Two of three optics rise, and this one far harder than the reference optic's 54 %.  Its other two hard step orders are **2.756 and 3.608**, so "first order at best" is a statement about the LADDER AVERAGE (measured 1.049 / 1.063) and false of the steps -- which the docstring's own table (`1.31 / 3.29 / -0.62`) already shows |
| D4a | archive-to-archive: parent default == `aperture_edge='hard'` == `aperture_edge_samples=1` | **CONFIRMED on my own prescriptions, both builds** | one-stop: `49ddf4bd` default `67fc2c48a6a8bab2` (WIN) / `1772fe33503ff079` (WSL) == round-2 `kw_hard` == `kw_samples_1` == the private-builder way back, bit for bit.  Default `3553955082c3daed` / `ab596c781254cab3` == `kw_gray4` |
| D4b | `None` stamps nothing | **CONFIRMED** | `aperture_edge=None, aperture_edge_samples=None` gives the default digest on every prescription; `_prescription_to_elements` emits elements with no `'edge'` key |
| D4c | validated BEFORE the decomposition | **CONFIRMED, measured not reasoned** | on a prescription whose decomposition emits a `UserWarning` (a DOE placeholder), a bad rim raises with **0 warnings recorded**; the same prescription with a good rim records exactly **1** |
| D4d | only `is_stop` surfaces get the stamp -- what about TWO stops, or a non-stop aperture? | **CONFIRMED and extended** | a prescription with TWO `is_stop=True` surfaces emits **2** aperture elements and **both** carry the stamp; the way back is bit-exact (`9b275f2e12e29e06` WIN / `7926bc1c5e659d15` WSL == the `49ddf4bd` default).  A vignetting `semi_diameter` that is not the stop emits **no aperture element at all**, so there is nothing for the keyword to reach, and `evaluate`'s docstring says exactly that |
| D4e | a prescription with no STOP is byte-identical with and without the keyword ("checked not asserted") | **CONFIRMED and now ASSERTED** | `f93926a229d0da05` (WIN) / `d55f406c7ccb426b` (WSL) on all six keyword spellings and on the `49ddf4bd` archive.  `test_verify_c1r2_a_prescription_with_no_stop_is_byte_identical` asserts it, with the "no aperture element emitted" premise asserted alongside |
| D4f | the three documented moved answers, and is the list complete? | **COMPLETE for executing call sites; one recipe is wrong -- defect R8** | a full sweep of `lumenairy/` finds seven places that render an `apply_aperture` rim without naming `edge=`: `apply_lyot_stop`, `algebra.Aperture._apply`, `JonesField.apply_aperture`, `ao.py:33` (docstring), `coronagraph_dock.py:376`, and codegen's TWO emission sites.  All seven are in the Migration table.  But the codegen row's way back -- "edit the generated `la.apply_aperture(...)` call" -- applies only to `style='unrolled'`; `style='system'` emits `{'type': 'aperture', ...}` with no `'edge'` key and **no `la.apply_aperture(` anywhere in the file**.  The GUI has exactly one site (`coronagraph_dock.py:376`); `ao_dock.py`'s `aperture` is a scalar that never reaches `apply_aperture` |
| M3 | 5 failed / 54 passed under the mutant, `[4]` green | **CONFIRMED on both builds** | `n_sub = None` in `_system_element_signature`: **5 failed, 54 passed** on Windows AND WSL; the red ids are `test_c1_the_jit_kernel_carries_the_elements_edge_samples[1]` plus all four `test_verify_c1_the_jit_kernel_honours_the_elements_edge_samples` params, and `[4]` stays green as claimed |
| M3+ | my own mutant | **CAUGHT** | the jit kernel reading `edge_samples` from the MODULE default (`n_sub = int(apply_aperture.__defaults__[-1])`) instead of the element: **5 failed, 54 passed**, same id set |
| Rec-1 | `.test_durations` 31 ids spliced, JSON valid, gate 4 passed | **CONFIRMED** | 16 655 entries, `json.load` succeeds, no duplicate keys; `test_c1_gray_edge_default.py` **31 collected / 31 present / 0 missing / 0 stale**, 12.0209 s; `test_verify_c1_gray_edge.py` **28 / 28 / 0 / 0**, 28.4076 s.  Gate: **4 passed** (after this round's own 24 ids were spliced, 16 655 -> 16 679) |
| Rec-2 | 1.17 % vs the 5 % bar | **CONFIRMED exactly** | on that test's own fixture (N = 64, dx = 5 um, complex64, D = 97.5 um): **48 / 4096 = 1.1719 %** against a 204-pixel bar, slack **4.2667x** |
| Rec-3 | `algebra/apertures.py` wording | **CONFIRMED at both sites** | the section banner reads "Aperture (sharp-edged amplitude mask)" and the class summary "Sharp-edged (unapodized) amplitude aperture with selectable shape."; the history-fingerprint gate is green without a re-record |
| Sweeps | the four WSL reds are premise-gated / environmental | **CONFIRMED** | on `git archive 7ea01ede` under WSL, `test_public_api.py::test_installed_metadata_version_matches_source_version` and both `test_v18_5_*` citation ids fail **identically** (3 failed, 19 passed, 3 skipped), and the fourth (`test_v16_synthetic_fabrication_is_caught`) **skips** there -- exactly as round 2 describes |
| Unmeasured-1 | the downstream boolean-cast exposure | **MEASURED; it moves -- defect R6** | dilation reproduces exactly (12281 -> 12449, **+168 px, +1.3680 %**, 364 rim pixels, analytic disc 12271.8).  `plotting.py:1491` radial-RMS curve **max 7.2741e-02 relative**; `plotting.py:1757` `_auto_n_bins` **12 -> 13** on a 48-px grid; `plotting.py:1096/1724` in-aperture RMS **5.8832e-03**, PV `2.520837e-07 -> 2.536185e-07`; `wrapper_merits.py:266` integrated power **7.7072e-03**, and the boolean cast OVERSHOOTS the correctly weighted grey mask by **8.9051e-03**.  Identical on both builds |
| Neutrality | (not claimed by round 2) did round 2 move any LEGAL answer? | **NOTHING MOVED** | 23 legal fixtures digested from inside `git archive 7ea01ede` and from inside the round-2 tree: **0 moved** on Windows, **0 moved** on WSL.  `evaluate`'s two new arguments are keyword-only and appended after `*` |
| Unmeasured-2 | the "6.0 float64 grids at N = 2048 vs 5.0" comment | **MEASURED; half right -- defect R7** | steady state, Windows: hard **5.0011**, grey **6.0011** -- the comment's two figures exactly.  WSL: hard **4.0011**, grey **5.1262**.  So the absolute pair is a Windows reading; the build-free claims are the DELTA (1.000 WIN / 1.125-1.134 WSL) and the independence of `n_sub` (2/4/8/16 spread 0.0000 WIN, 0.008 WSL), both confirmed.  A trap the comment does not mention: the FIRST `apply_aperture` call in a process carries ~1.37 (WIN) / ~1.05 (WSL) extra grids, so whichever arm is traced first reads high |

---

## 2. What was measured, and how

### 2.1 The guard census (D1)

`validation/probe_verify_c1_round2/probe_d1_guard_census.py` runs **24** rim
spellings through **four** entry points -- `apply_aperture` itself,
`propagate_through_system`, `propagate_through_system_jax(verbose=True)` and
`propagate_through_system_jax()` -- and records `(raised, "Type: message")` for
each.  Round 2 compared three routes; the fourth is added because the whole
claim is that the chain refuses exactly what the function refuses.

On the round-2 tree: `chain-routes-split = 0`, `chain-vs-apply_aperture-split
= 0`, and the entire `rows` object compares equal between
`d1_guard_R2_WIN.json` and `d1_guard_R2_WSL.json`.

On `git archive 7ea01ede` (`d1_guard_PRE7e_WIN.json`): **5** rows split.
Round 2's addendum table lists two rows as changed (`2.5` and `'4'`).  Three
more were split on the MESSAGE, which round 2's own new test asserts on:

| row | the other three routes | the jit'd route at `7ea01ede` |
|---|---|---|
| `{'edge': None}` | `... got None.` | `... got 'None'.` |
| `{'edge': b'gray'}` | `... got b'gray'.` | `... got "b'gray'".` |
| `{'edge_samples': False}` | `... got False.` | `... got 0.` |

so the fix is worth more than the addendum claims for it.

**The legal spellings, which round 2 did not enumerate.**  `numpy.int64(4)`,
`numpy.float64(4.0)` and `4.0` are accepted; `numpy.str_('gray')` is accepted;
`edge_samples` without `edge` is accepted.  Every one of them is ONE answer on
all four routes, which is the right shape: a NumPy integer is legal, and it is
legal everywhere.

### 2.2 The hole the fixed fixture could not see -- defect R1

Round 2 put the refusal in `_aperture_edge_kwargs` because it is "the ONE place
both backends read the element".  It is the one place both backends read the
element **when the element's params resolve**.  Both JAX routes reach it only
through `_resolve_aperture_params`:

* `_system_element_signature` returns `None` and bypasses the kernel cache when
  `_resolve_aperture_params(elem)` is `None` -- **before** it calls
  `_aperture_edge_kwargs`;
* the eager route's `if resolved is not None:` block **contains** the
  `_aperture_edge_kwargs(elem)` call.

`propagate_through_system` has no such guard: it calls
`_aperture_edge_kwargs(elem)` unconditionally.  `probe_d1_holes.py`, both
builds, identical:

| element | NumPy chain | JAX eager | JAX jit |
|---|---|---|---|
| `{'type':'aperture','shape':'circular','edge':'soft'}` | **raises** | accepts | accepts |
| ... `'edge': None` | **raises** | accepts | accepts |
| ... `'edge_samples': 0` | **raises** | accepts | accepts |
| ... `'edge_samples': 2.5` | **raises** | accepts | accepts |
| ... `'params': {}, 'edge': 'soft'` | **raises** | accepts | accepts |
| `shape='rectangular', 'params': {}, 'edge':'soft'` | **raises** | accepts | accepts |
| ... `'params': {'diameter': None}, 'edge': 'soft'` | **raises** | accepts | accepts |
| control: `'params': {'diameter': 5.3e-5}, 'edge': 'soft'` | raises | raises | raises |

7 of 7, on both builds.  On `git archive 49ddf4bd` six of the seven read
`identical=True` -- all three routes silently ignored the rim key -- so the
split arrived with WP-C1 and round 2 did not close it.  The seventh,
`params={'diameter': None}`, already split at `49ddf4bd`, but for an
unrelated reason (the NumPy arithmetic on a `None` diameter raises), and it
is kept in the census because the rim keyword still fails to reach the two
JAX routes on it.

### 2.3 The ratio bar, and the sentence under it -- defect R4

`probe_d2_ratio_v2.py` reproduces the three shipped readings to every digit on
both builds, then scans 30 fixtures around the binding one (`D/dx = 145`):

| `dy/dx` | offset | `e_hard` (signed) | `e_g4` (signed) | ratio |
|---|---|---|---|---|
| 0.4 | 0.29 | `+6.078e-05` | `+3.050e-05` | **1.9928** (the shipped fixture) |
| 1.0 | 0.29 | `-6.034e-05` | `+7.592e-05` | 0.7948 |
| 0.5 | 0.35 | | | 0.0281 |
| 0.5 | 0.05 | | | 0.0061 |
| **1.0** | **0.23** | **`+2.189e-07`** | `+6.078e-05` | **0.0036** |

Four of thirty are below 1.0, i.e. the HARD arm's transmitted area is closer to
the analytic disc than the grey arm's.  The mechanism is visible in the signed
column: the hard arm's staircase area error oscillates about zero with `D` and
offset, and at `(145, 1.0, 0.23)` it is passing through a zero.  The grey arm's
own sub-sample quantisation residual, about `3e-05` to `8e-05` here, does not
shrink with `D` -- which VERIFY_WP-C1.md sec. 2.5 already recorded for the grey
AREA reading and which the second-order convergence in sec. 2.1 is a FIELD
statement about, not an area one.

The bar is fine.  The sentence explaining it is a rationalisation that sends a
future re-pinner to a red: "pick a bigger D, the staircase is milder" is the
opposite of what the data does.

### 2.4 A third optic -- defect R5

`probe_d3_ladder.py` re-runs VERIFY-C1's optic (reproducing all sixteen
entries) and adds lambda = 532 nm, a = 150 um, window 900 um, z = 30.0 mm (RS
spatial) / 15.0 mm (HF), against the same closed form.  The RS alias threshold
`2 W^2 / (N lambda)` is 23.79 mm at N = 128 and the probe asserts `z` clears it
at every N.

| kernel | arm | errs (N = 128/256/512/1024) | step orders | gain | mean |
|---|---|---|---|---|---|
| RS | hard | `6.2976e-03 9.3199e-04 7.6432e-05 7.1042e-04` | `2.756 / 3.608 / **-3.216**` | 8.8646x | 1.0494 |
| RS | gray | `4.7895e-03 1.1081e-03 2.9414e-04 9.3727e-05` | `2.112 / 1.914 / 1.650` | 51.1000x | 1.8918 |
| HF | hard | `1.0747e-02 1.5966e-03 1.2930e-04 1.1785e-03` | `2.751 / 3.626 / **-3.188**` | 9.1194x | 1.0630 |
| HF | gray | `1.5833e-02 3.3110e-03 9.5393e-04 2.4274e-04` | `2.258 / 1.795 / 1.974` | 65.2234x | 2.0091 |

Windows and WSL print every one of those digits identically.

Two things follow.  The **RATE gap is now confirmed on three optics** spanning
532-1064 nm, a = 62.5-150 um and Fresnel numbers 0.9-1.4: grey mean order
1.83-2.01 against hard 1.05-1.08, ladder gains 45-65x against 8.9-9.5x.  That
is a robust general claim and the right one to ship.  But the hard arm's RISE
is **not rare** -- it happens on two of the three optics, and on this one by a
factor of 9.3 rather than 54 % -- and "first order at best" is false of the
STEPS, two of which here exceed second order.  Both the Migration guide's "that
rise is that optic's" and the docstring's "first order at best" are therefore
true only under a reading the reader has to supply.

### 2.5 `evaluate`'s way back -- D4, on two stops and on none

`probe_d4_evaluate_v2.py` uses three prescriptions of its own (a one-stop
geometry different from the verification's, a TWO-stop one, and one with no
stop and a vignetting semi-diameter), run archive-to-archive.

| arm | one stop | two stops | no stop |
|---|---|---|---|
| `49ddf4bd` default (WIN) | `67fc2c48a6a8bab2` | `9b275f2e12e29e06` | `f93926a229d0da05` |
| round 2, default | `3553955082c3daed` | `8b0da8712e5fcb94` | `f93926a229d0da05` |
| round 2, `aperture_edge='hard'` | **`67fc2c48a6a8bab2`** | **`9b275f2e12e29e06`** | `f93926a229d0da05` |
| round 2, `aperture_edge_samples=1` | `67fc2c48a6a8bab2` | `9b275f2e12e29e06` | `f93926a229d0da05` |
| round 2, `gray` + `samples=4` | `3553955082c3daed` | `8b0da8712e5fcb94` | `f93926a229d0da05` |
| round 2, `None` + `None` | `3553955082c3daed` | `8b0da8712e5fcb94` | `f93926a229d0da05` |

and on WSL, with the same structure: `1772fe33503ff079` / `ab596c781254cab3`,
`7926bc1c5e659d15` / `c24ab1760af43599`, `d55f406c7ccb426b`.

The two-stop row is the one round 2 could not have measured: with one stop,
"every emitted element" and "the first emitted element" are the same claim.
Both elements are stamped (`edges=['hard','hard']`,
`samples=[7,7]`), and the way back is exact.

### 2.6 The boolean-cast exposure, measured -- defect R6

VERIFY-C1 measured the dilation and recorded the exposure as unmeasurable;
round 2 repeated the note.  `probe_boolcast_exposure.py` builds the smallest
fixture that reaches each site with a caller-supplied `apply_aperture` result
and reads what the site returns.  Both builds identical (no BLAS in any of it).

| site | the quantity | hard mask | grey mask | move |
|---|---|---|---|---|
| the cast itself | true pixels on a 0.5 mm disc at dx = 4 um | **12281** | **12449** | **+168 px, +1.3680 %** (analytic disc 12271.8, rim 364 px) |
| `plotting.py:1757` | `_auto_n_bins(n_in_ap)` at N = 48 | 12 bins (657 px) | **13 bins** (697 px) | a whole bin; at N = 256 both saturate at 32 |
| `plotting.py:1491` | `_radial_rms_profile`'s returned curve, 16 bins | -- | -- | **max 7.2741e-02 relative**, last bin 3.1616e-03 |
| `plotting.py:1096`/`:1724` | in-aperture RMS / PV on the NaN-masked map | RMS ref, PV `2.520837e-07` | PV `2.536185e-07` | RMS **5.8832e-03**, PV **6.1e-03** |
| `wrapper_merits.py:266` | `sum(|E|^2)` over the cached mask | ref | -- | **7.7072e-03**, and the cast overshoots the correctly weighted grey mask by **8.9051e-03** |

Two scope notes.  `plot_wavefront(aperture=...)` documents its argument as
`ndarray bool`, so a float grey mask is already outside that contract -- but
before this release the natural way to build such an array,
`apply_aperture(np.ones(...), ...)`, satisfied the cast exactly, and now it
silently dilates by a rim.  And the reachability of `wrapper_merits.py:266`
from in-library code is narrow: two of its three callers `float()` the value
first, and only the call at `wrapper_merits.py:492` forwards
`ctx.prescription['aperture_diameter']` unfloated, so an ndarray there reaches
the array branch.

### 2.7 The peak-memory comment -- defect R7

`probe_peak_memory.py` traces `apply_aperture` at N = 2048 with `tracemalloc`,
having first confirmed that numpy's allocations are traced (a known 1.0-grid
array reads 1.000 grids).  Steady state, with the input field's 2.0 grids
excluded:

| build | hard | grey n=2 | n=4 | n=8 | n=16 | grey - hard |
|---|---|---|---|---|---|---|
| Windows py3.14 / numpy 2.4.4 | **5.0011** | 6.0011 | **6.0011** | 6.0011 | 6.0011 | **1.000** |
| WSL py3.12 / numpy 2.4.6 | **4.0011** | 5.1261 | **5.1262** | 5.1263 | 5.1269 | **1.125** |

The comment's "6.0 ... against 5.0" is exactly the Windows reading and is
therefore not fabricated -- but it is one build's, and the WSL reading is a
grid lower on both arms.  The two claims that ARE build-free, and that the
comment is really making, both hold: the grey branch costs about one extra
float64 grid at peak, and that peak does not grow with `n_sub`.

One trap worth recording because it cost this round a wrong first reading: the
FIRST `apply_aperture` call in a process carries ~1.37 (Windows) / ~1.05 (WSL)
extra grids of one-off allocation, so a trace that measures the hard arm first
reports it at 6.375 grids and concludes -- wrongly -- that the grey branch is
cheaper.

### 2.8 What round 2 broke: nothing

Round 2's library changes are a REFUSAL and two keywords that default to
`None` and stamp nothing, so they should be answer-neutral for every input
that was already legal.  The addendum does not make that comparison, so this
round did.  `probe_round2_moved_nothing.py` digests **23** legal fixtures --
`apply_aperture` on three shapes, an anamorphic grid, an odd grid, complex64,
a field carrying NaN and +-inf, `numpy.int64` `edge_samples`, `apply_lyot_stop`,
the `algebra.Aperture` operator, both `JonesField` components, all three chain
routes on three element spellings (unkeyworded, `edge='hard'`,
`edge_samples=8`), and `lumenairy.evaluate` on a STOP prescription -- run from
inside the `git archive 7ea01ede` extraction and from inside the round-2 tree.

| comparison | Windows | WSL |
|---|---|---|
| `7ea01ede` vs round 2, 23 legal fixtures | **0 moved** | **0 moved** |

The 22 mask-level fixtures are additionally byte-identical BETWEEN the two
builds; only `evaluate_default`, which runs a full propagation, differs across
builds, as it should.

The public surface is clean too: `evaluate`'s two new arguments are
keyword-only and appended after `*`, so no positional call shifts, and
`_prescription_to_elements`'s new arguments are keyword-only on a private
helper.

---

## 3. Defects

### R1 -- D1's refusal does not reach an `'aperture'` element whose params do not resolve (P2)

**Reproducer** (both builds):

```python
import numpy as np, jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from lumenairy.propagators.system import (propagate_through_system,
                                          propagate_through_system_jax)
E = np.ones((64, 64), dtype=complex)
el = [{'type': 'aperture', 'shape': 'circular', 'edge': 'soft'}]   # no params
propagate_through_system(E, el, 1064e-9, dx=1.25e-6)                # ValueError
propagate_through_system_jax(jnp.asarray(E), el, 1064e-9, 1.25e-6,
                             verbose=True)                          # a field
propagate_through_system_jax(jnp.asarray(E), el, 1064e-9, 1.25e-6)  # a field
```

Same with `{'params': {}}`, `{'params': {'diameter': None}}`,
`shape='rectangular'` with no widths, and with `edge=None`, `edge_samples=0`,
`edge_samples=2.5` in place of `edge='soft'`.

**Root.**  `lumenairy/propagators/system.py`, two places, both of which put the
`_resolve_aperture_params` gate ABOVE the element reader:

```python
    if etype == 'aperture':
        ...
        resolved = _resolve_aperture_params(elem)
        if resolved is None:
            return None                     # <-- leaves before reading the rim
        shape, halves = resolved
        edge_kw = _aperture_edge_kwargs(elem)
```

and, in the JAX slow path,

```python
            resolved = _resolve_aperture_params(elem)
            if resolved is not None:
                shape, halves = resolved
                edge_kw = _aperture_edge_kwargs(elem)
```

while `propagate_through_system` calls `_aperture_edge_kwargs(elem)`
unconditionally in its `'aperture'` branch.

**Requested edit** -- hoist the one reading above the resolve gate in both
places, so the element is read once and refused once whatever its params say.
In `_system_element_signature`:

```python
    if etype == 'aperture':
        xc = float(elem.get('xc', 0.0))
        yc = float(elem.get('yc', 0.0))
        # VERIFY-C1-ROUND2 R1: read (and therefore REFUSE) the rim BEFORE the
        # params gate.  ``_resolve_aperture_params`` returns None for an
        # element with no usable params, and returning here first let an
        # illegal ``edge`` / ``edge_samples`` past both JAX routes while
        # ``propagate_through_system`` -- which calls this reader
        # unconditionally -- raised.  Measured 2026-09-20 on both builds,
        # 7 element shapes out of 7.
        edge_kw = _aperture_edge_kwargs(elem)
        resolved = _resolve_aperture_params(elem)
        if resolved is None:
            return None
        shape, halves = resolved
        edge = (str(edge_kw['edge']) if 'edge' in edge_kw else None)
        n_sub = (int(edge_kw['edge_samples'])
                 if 'edge_samples' in edge_kw else None)
```

and in the JAX slow path's `'aperture'` branch:

```python
            xc = elem.get('xc', 0.0)
            yc = elem.get('yc', 0.0)
            edge_kw = _aperture_edge_kwargs(elem)   # VERIFY-C1-ROUND2 R1
            resolved = _resolve_aperture_params(elem)
            if resolved is not None:
                shape, halves = resolved
                ...
```

**Tests that close it.**
`tests/unit/test_verify_c1_round2.py::test_verify_c1r2_every_route_refuses_a_bad_rim_on_an_unresolvable_element`
(5 params, `xfail(strict=True)`; `xfail_strict = true` is on in
`pyproject.toml`, so all five FAIL the moment the hoist lands and the markers
must go with it) and its positive twin
`..._the_unresolvable_split_is_real_and_not_a_collection_quirk`, which goes red
on the fix and says what changed.

### R2 -- `edge_samples=True` is silently the pre-5.49 rim (P3)

`_validate_edge_kwargs` refuses `False` because `int(False) == 0 < 1`, not
because it is a bool, so `True` passes: `int(True) == 1` and `1 != True` is
`False`.  All four entry points agree, so this is not a route split -- it is a
guard-completeness gap, and it is the one spelling where it costs something:
`edge_samples=True` is bit-for-bit `edge='hard'`, i.e. the answer this release
moved away from, selected silently by a caller who was plainly trying to turn
something ON.

The shipped census row is named `edge_samples_bool_false` and reads as "a bool
is refused".  It is green for the wrong reason, and the bool a caller would
actually write is not in the census at all.

**Requested edit** (`lumenairy/elements/elements.py`, in
`_validate_edge_kwargs`, and one line in its `Raises` section):

```python
    if edge_samples is _EDGE_UNSET:
        return None
    # VERIFY-C1-ROUND2 R2: refuse a bool explicitly.  ``int(True) == 1`` and
    # ``1 != True`` is False, so ``True`` slipped through the exact-integer
    # test below and silently selected n_sub = 1 -- which is bit-for-bit
    # ``edge='hard'``, the pre-5.49 rim.  ``False`` was refused only because
    # ``int(False) == 0 < 1``, which is why the census row named for it was
    # green for the wrong reason.  Measured 2026-09-20, both builds.
    if isinstance(edge_samples, (bool, np.bool_)):
        raise ValueError(
            f"apply_aperture: edge_samples must be a positive integer "
            f"(sub-samples per axis), not a bool; got {edge_samples!r}.  "
            f"For the binary pixel-centre rim pass edge='hard'.")
    n_sub = int(edge_samples)
```

and add `('edge_samples_bool_true', {'edge_samples': True})` to
`_BAD_EDGE_ELEMENTS` in `tests/unit/test_c1_gray_edge_default.py`.

**Tests.** `test_verify_c1r2_a_bool_edge_samples_is_answered_the_same_way_everywhere`
pins the four-route agreement (which survives the fix) and
`test_verify_c1r2_edge_samples_true_is_bit_for_bit_the_pre_5_49_rim` pins what
is currently selected; the second goes red on the fix, which is the point.

### R3 -- the `int()` refusal family raises a message that names nothing (P3)

`{'edge_samples': None}`, `[4]` and `4+0j` all raise

    TypeError: int() argument must be a string, a bytes-like object or a
    real number, not 'NoneType'

on all four entry points -- consistent, but naming neither `apply_aperture` nor
`edge_samples`, while `{'edge': None}` on the same guard gets
`apply_aperture`'s own `ValueError`.  `_validate_edge_kwargs`'s docstring
documents the `TypeError`, so this is not a surprise to the library; the
problem is that `evaluate`'s `Raises` section promises

> ValueError ... if `aperture_edge` / `aperture_edge_samples` are not what
> `apply_aperture` accepts (**the same refusal, from the same guard**)

which is wrong for this family, and that the shipped census asserts
`'apply_aperture' in message` for its own eight rows -- a contract this family
does not meet.

**Requested edit** (`lumenairy/elements/elements.py`):

```python
    try:
        n_sub = int(edge_samples)
    except (TypeError, ValueError) as exc:
        # VERIFY-C1-ROUND2 R3: name the function and the keyword.  int()'s own
        # TypeError names neither, which made this the one refusal family the
        # chain census's ``'apply_aperture' in message`` assertion could not
        # cover, and made ``evaluate``'s Raises section ("a ValueError ... the
        # same refusal, from the same guard") wrong for it.
        raise ValueError(
            f"apply_aperture: edge_samples must be a positive integer "
            f"(sub-samples per axis); got {edge_samples!r}.") from exc
```

and drop the `TypeError` entry from the function's `Raises` section.

**Test.** `test_verify_c1r2_the_int_refusal_family_names_the_library`
(3 params, `xfail(strict=True)`).

### R4 -- D2's bar is right; the sentence deriving it is refuted (P3, documentation-in-test)

`tests/unit/test_audit2609_a8_verify.py`'s "Bars" paragraph ends:

> The 145-px fixture is the binding one because a 145-pixel rim is already well
> sampled, so the staircase it beats is the mildest of the three -- which is the
> right fixture for the bar to be derived from.

Thirty fixtures around it say otherwise: four have ratios below 1.0 and the
worst is 0.0036, because at `D/dx = 145` the hard arm's SIGNED area error is
crossing zero.  Size of `D` is not what sets the ratio.

**Requested edit** -- replace that sentence with what the scan shows:

```
    The 145-px fixture is the binding one, but NOT because a bigger rim is a
    milder staircase: the ratio is set by where each arm's SIGNED area error
    falls relative to zero, and at D/dx = 145 the hard arm happens to be near
    a crossing.  A 30-point scan of that fixture's neighbourhood (VERIFY-C1
    ROUND2, validation/probe_verify_c1_round2/d2_ratio_R2_*.json) finds four
    ratios below 1.0, the worst 0.0036 at dy/dx = 1.0, offset 0.23, where
    e_hard = +2.19e-07 against e_g4 = +6.08e-05.  So the bar is a pin on THESE
    THREE fixtures and must be re-measured, not extrapolated, if they change --
    and the AREA ratio is not the convergence claim, which is a field property
    (the grey arm's own sub-sample quantisation residual does not shrink with
    D; see VERIFY_WP-C1.md sec. 2.5).
```

**Test.** `test_verify_c1r2_a_bigger_rim_does_not_mean_a_milder_staircase_to_beat`,
which also asserts that the binding fixture still clears 1.5, so it is a scope
statement and not an attack on the bar.

### R5 -- "the rise is that optic's" and "first order at best" both need one more clause (P3, documentation)

Measured on a third optic: the hard arm rises by a factor 9.3 (RS) and 9.1
(HF), and two of its three step orders are above second order.

**Requested edits** (two places):

* `Migration-Guide.md` section 5.49.0, "### Why", the sentence "That rise is
  that optic's -- whether the staircase error rises at a given refinement
  depends on where the rim falls on the lattice at each N -- so do not read it
  as a library property; on an independent optic ... the hard arm falls at
  every step": append *"-- while on a third (lambda = 532 nm, a = 150 um,
  window 900 um, z = 30 / 15 mm) it rises by a FACTOR of 9.3 on the same last
  refinement.  Rising is common, not exceptional; what is not a library
  property is the direction at any given N."*
* `lumenairy/elements/elements.py`, `apply_aperture` docstring: change *"The
  hard edge is **first order at best and its step orders are erratic**"* to
  *"The hard edge averages **first order at best over a ladder** (1.05-1.32
  measured on three optics) and its individual step orders are **erratic** --
  measured from -3.2 to +3.6, as the table's own `3.29` and `-0.62` show"*.

**Test.** `test_verify_c1r2_the_hard_arms_rise_is_common_and_its_steps_exceed_two`,
whose three decisions (a step above 2.0, a step below 0.0, and the ladder-gain
gap surviving at >= 4x) each carry a measured gap on both sides.

### R6 -- the downstream boolean cast moves answers and has no Migration line (P3)

Measured in sec. 2.6.  Nothing in `lumenairy/` passes an `apply_aperture`
result into these sites, so **no shipped answer is wrong**; what moves is a
caller's, and the pattern that moves it (`apply_aperture(np.ones(...), ...)`
as the way to build an aperture array) is the natural one and was exact before
this release.

**Requested edit** -- `Migration-Guide.md` section 5.49.0, "What moves", add a
row and one sentence after the table:

   | entry point | exposes `edge=`? | the way back |
   |---|---|---|
   | an `apply_aperture` RESULT passed as an array `aperture=` to `plot_wavefront`, `plot_opd_summary` or a wrapper merit | not applicable -- these BOOLEAN-CAST the array | `edge='hard'` when building it, or cast it yourself with the threshold you mean (`mask = arr >= 0.5`) |

   *A grey mask boolean-casts to the union of the open area and the whole rim:
   measured +168 pixels (+1.37 %) on a 12281-pixel disc, which moves
   `plot_opd_summary`'s radial-RMS curve by up to 7.3 %, its automatic bin
   count by a whole bin on small grids, and a wrapper merit's integrated power
   by 0.77 %.  `aperture=` is documented as a boolean mask and still is; what
   changed is that the obvious way to build one no longer produces one.*

and the same row in the CHANGELOG's Migration paragraph.

**Tests.** `test_verify_c1r2_the_boolean_cast_moves_a_plotted_radial_rms_curve`,
`..._moves_a_wrapper_merits_integration`, `..._moves_the_auto_radial_bin_count`.

### R7 -- the peak-memory comment quotes one build's absolute figures (P3, comment)

Confirmed exactly on Windows, one grid lower on both arms under WSL.

**Requested edit** (`lumenairy/elements/elements.py`, the grey branch's
comment):

```python
    # Grey edge: average the binary mask over an n_sub x n_sub lattice of
    # sub-pixel offsets centred on each pixel, giving its open-area
    # fraction.  Accumulated one sub-mask at a time, so the peak stays at
    # the cost of a single sub-mask evaluation instead of growing with
    # n_sub**2.  Allocator trace at N = 2048 (tracemalloc, steady state --
    # the FIRST call in a process carries ~1.4 grids of one-off allocation
    # and reads high), VERIFY-C1-ROUND2 2026-09-20: this branch costs ONE
    # extra float64 grid at peak over the hard branch (Windows py3.14 /
    # numpy 2.4.4: 6.0011 against 5.0011; WSL py3.12 / numpy 2.4.6: 5.1262
    # against 4.0011 -- the absolute figures are the build's, the delta is
    # not), and that peak is independent of n_sub (2 / 4 / 8 / 16 spread
    # 0.0000 grids on Windows, 0.008 on WSL).
```

**Test.** `test_verify_c1r2_the_grey_branch_costs_one_grid_and_is_flat_in_n_sub`,
which asserts only the two build-free claims.

### R8 -- the Migration table's codegen recipe covers one of codegen's two styles (P3)

`generate_simulation_script(..., style='system')` is a documented public
option.  For a STOP surface it emits

```python
    {'type': 'aperture', 'shape': 'circular', 'params': {'diameter': 1.79999999999999995e-03}},
```

with no `'edge'` key and **no `la.apply_aperture(` anywhere in the generated
file**, so the table's way back -- "edit the generated `la.apply_aperture(...)`
call, or re-pin" -- names something the reader cannot find.

**Requested edit** -- `Migration-Guide.md` section 5.49.0, replace the codegen
row with two:

   | entry point | exposes `edge=`? | the way back |
   |---|---|---|
   | a script emitted by `lumenairy.io.codegen` with `style='unrolled'` (the default) for a STOP surface | it emits no keyword | add `edge='hard'` to the generated `la.apply_aperture(...)` call, or re-pin |
   | the same with `style='system'` | it emits no key | add `'edge': 'hard'` to the generated `{'type': 'aperture', ...}` element, or re-pin |

**Test.** `test_verify_c1r2_the_codegen_migration_recipe_misses_the_system_style`,
which asserts both halves so it stays honest if either style changes.

---

## 4. The observations that are not defects

* **The D1 fix is worth more than its own addendum claims.**  Round 2's table
  shows two rows changing verdict; five rows change `(verdict, message)`, and
  the census the shipped test asserts is a message census.
* **A NumPy integer is legal, and it is legal everywhere.**  `numpy.int64(4)`,
  `numpy.float64(4.0)` and `numpy.str_('gray')` are accepted by all four entry
  points with one answer.  That is the right call (they are exact) and it is
  consistent, which was the question.
* **The jit signature cache cannot be tricked.**  Measured in one process:
  `4` / `4.0` / `numpy.int64(4)` give one digest, `1` / `True` / `1.0` another,
  and the two differ.  At `49ddf4bd` both groups gave the SAME digest, which is
  the fail-before for the claim.
* **`VERIFY_WP-C1.md` still prints `1.06`.**  Round 2 corrected the test
  docstring and the CHANGELOG and said so, but the audit report of record keeps
  a figure the follow-up round knows to be 1.0709.  Suggested: add
  *"(erratum, VERIFY-C1-ROUND2: 1.0709; `log2(9.2699)/3`)"* to the `mean order`
  row at line 73.  No assertion reads it.
* **`test_c1_mutation_edge_samples_moved_off_the_knee_is_caught` is still the
  weak row VERIFY-C1 recorded.**  Round 2 did not change it, and did not claim
  to; M8 remains the real evidence and the named test does catch it.
* **The three in-library callers of `_get_wrapper_merit_cache` pass a scalar.**
  Two `float()` it first; only the call at `wrapper_merits.py:492` forwards
  `ctx.prescription['aperture_diameter']` unfloated, so the array branch at
  `:266` is reachable in-library only through a prescription carrying an
  ndarray there.  That narrowness is why R6 is P3 and not P2.
* **The GUI has exactly one site.**  `coronagraph_dock.py:376`; `ao_dock.py`'s
  `aperture` is a scalar that never reaches `apply_aperture`, and
  `chebyshev_fit_dock.py:144` casts a caller-supplied mask (the R6 family, GUI
  side).
* **On a FACTORY-shape prescription the keyword validates and then does
  nothing.**  `la.evaluate(make_singlet(...), src, aperture_edge='hard')`
  returns `10c5e90986bdad5e`, byte-identical to the unkeyworded call and to
  `aperture_edge_samples=16`, while `aperture_edge='soft'` still raises.  That
  is correct -- the factory shape becomes one `real_lens` element whose
  `aperture_diameter` is the lens's own mask, which was never `apply_aperture`
  and which the Migration guide already excludes -- and the docstring's
  "Zemax-shape prescription" plus "prescriptions with no STOP surface emit no
  aperture element" covers it.  But the prescription DOES carry an
  `aperture_diameter` key, so a reader can reasonably expect the rim keyword to
  bite.  Suggested (not filed): append to `evaluate`'s `aperture_edge`
  paragraph *"A factory-shape prescription (`surfaces` + `thicknesses`, e.g.
  from `make_singlet`) becomes a single `'real_lens'` element whose
  `aperture_diameter` is the lens's OWN mask, not an `apply_aperture` rim, so
  this keyword is validated and then has no effect there."*

---

## 5. The runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line, `-p no:randomly --capture=sys -q`, `PYTHONPATH` naming the tree
under test, and `lumenairy.__file__` printed by every probe.

| what | Windows py3.14 | WSL py3.12 |
|---|---|---|
| the 81-file sweep -- every aperture / system / evaluate-touching unit file, plus census, walkers, dispatcher pins, public API, doc consistency, A17, the durations gate, `test_audit_except_budget.py` and the three C1 files | **3856 passed, 14 skipped, 8 xfailed, 0 failed** in 35:20 | **3847 passed, 19 skipped, 8 xfailed, 4 failed** in 45:07, all four premise-gated (sec. 5.1) |
| `tests/unit/test_verify_c1_round2.py` | **16 passed, 8 xfailed** in 16.0 s | **16 passed, 8 xfailed** in 16.3 s |
| `tests/unit/test_c1_gray_edge_default.py` + `tests/unit/test_verify_c1_gray_edge.py` | **59 passed** in 22.4 s | (in sweep) |
| `tests/unit/test_audit2609_a8_verify.py` | **36 passed** in 8.8 s | (in sweep) |
| `tests/unit/test_audit2609_a15a_durations_staleness.py` (after splicing 24 ids) | **4 passed** in 96.0 s | (in sweep) |
| D1-revert mutant, the two C1 files | **6 failed, 53 passed** | **6 failed, 53 passed** |
| M3 mutant, the two C1 files | **5 failed, 54 passed** | **5 failed, 54 passed** |
| M10 mutant (mine: the jit kernel reads `edge_samples` from the module default) | **5 failed, 54 passed** | -- |
| M5 mutant, `test_audit2609_a8_verify.py` | **4 failed, 32 passed** | -- |
| the three premise-gated WSL ids on `git archive 7ea01ede` | -- | **3 failed, 19 passed, 3 skipped** (identical failures; the fourth skips) |
| `ruff check lumenairy/ tests/ scripts/` | -- | **All checks passed!** |
| `ruff check ... validation/probe_verify_c1_round2/` | -- | **All checks passed!** |
| `python -m mypy` (no args) | **Success: no issues found in 33 source files** | -- |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** (rc 0) | -- |

### 5.1 The sweep, and the four WSL reds

The file list is `validation/probe_verify_c1_round2/_sweep_files.txt` (81
files, built by grepping `tests/unit/` for `apply_aperture`,
`apply_lyot_stop`, a `'type': 'aperture'` element, `propagate_through_system`,
`evaluate`, `_prescription_to_elements` and the `algebra.Aperture` operator,
then adding every census / walker / dispatcher-pin / public-API /
doc-consistency / A17 / durations-gate / except-budget file and the three C1
files).  Tails in `_sweep_WIN_tail.txt` and `_sweep_WSL_tail.txt`.

| lane | result |
|---|---|
| Windows py3.14 | **3856 passed, 14 skipped, 8 xfailed, 0 failed** in 2120.38 s |
| WSL py3.12 | **3847 passed, 19 skipped, 8 xfailed, 4 failed** in 2707.24 s |

The 8 xfailed on both lanes are this round's own strict xfails (R1's five
params and R3's three), which is what a strict xfail is for: they fail the
moment either defect is fixed.

**The four WSL reds are the four round 2 names, and none is a library
finding.**  All four are GREEN on the Windows lane on this same commit:

* `test_public_api.py::test_installed_metadata_version_matches_source_version`
  -- the WSL venv's editable install metadata against a 5.48.1 source.
* `test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines`
  and `::test_v18_5_companion_reanchor_tool_exists_and_covers_the_cited_files`
  -- both shell out to `git`, which from WSL cannot resolve this worktree's
  `.git` file (it points at a Windows path).
* `test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught`
  -- same root: the walker returns rc = 2 ("the git plumbing failed") where the
  test expects rc = 1.

Confirmed premise-gated by re-running those three files from inside this
round's own `git archive 7ea01ede` extraction under WSL: **3 failed, 19 passed,
3 skipped** -- the same three failures, with `test_v16_synthetic_fabrication_is_caught`
SKIPPING there on a different premise gate, exactly as round 2 describes.  The
conflated-diagnostic note VERIFY-C1 left on that id (its failure message reads
"the walker is silently passing fabrications", which is wrong for rc = 2)
still stands and is still nobody's to fix in this chain.

---

## 6. What could not be measured

* **CuPy under WSL** -- still absent; the CuPy arm is Windows-only, and this
  round did not add a CuPy probe of its own (the rim keywords are refused in
  pure Python before any array work).
* **A GPU device for the JAX arm.**  Both builds' JAX runs are on the CPU
  backend.  R1's split is a pure-Python control-flow difference and runs before
  any device work, but that is reasoning, not a measurement.
* **The whole 14 666-id unit suite.**  What was run is the 81-file sweep, the
  three C1 files plus `test_audit2609_a8_verify.py` under four mutants, and the
  durations gate.  A release gate needs the full matrix.
* **Whether any EXTERNAL caller passes an `apply_aperture` result into a
  boolean-casting consumer.**  R6 measures what happens when one does -- which
  is what round 2 could not measure -- but not how many do.  Nothing in
  `lumenairy/`, `validation/` or `examples/` does.
* **A generated script executed end to end.**  R8 reads the emitted TEXT of
  both codegen styles on a STOP prescription; neither script was run against
  both trees, so the size of the move in a generated script's own output is
  inferred from `apply_aperture`'s, not measured.
* **`Migration-Guide.md` still has no 5.48.0 section.**  VERIFY_WP-C1.md sec. 5
  says what one would contain; that is still nobody's scope.
* **Whether the 5 %-of-pixels bar in `test_audit_misc.py` should be tightened.**
  Re-measured at 1.1719 % with 4.2667x of slack, unchanged from round 2's
  reading; tightening it remains B1-1's decision.
