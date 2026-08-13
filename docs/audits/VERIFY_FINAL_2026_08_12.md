# VERIFY FINAL -- the release gate for v5.35.0

**2026-08-12/13.  Adversarial re-verification of MERGED MAIN `3154fa7` in a
FRESH detached worktree `C:/tmp/lum_final` (checkout only; `git status`
--porcelain empty at start and at end).  REPORT-ONLY: no file in the
repository was modified by this pass; every probe lives in the session
scratchpad.  The verifier had no part in any of the work being verified.**

---

## 0. HEADLINE

> **ZERO P0.  ONE P1.  THE RELEASE DOES NOT PROCEED.**
>
> Every one of `VERIFY_ARCHITECTURE`'s eight mandated defects (five P0, three
> P1) is CLOSED **on merged main**, and the re-baselined banner reproduces to
> every printed digit on this tree with the import pinned to it.  Each of the
> eight was re-probed with a probe written against the public API rather than
> lifted from the shipped tests, on fixtures deliberately different from the
> ones the fixes were verified with, and -- where the fix is a KEY rather than
> a number -- with the mechanism REVERTED in-process to prove the pins are not
> vacuous.  All eight hold.
>
> **The P1 is not in any of that work.  It is in the seam between two of the
> merged branches, and it is exactly the class this pass was sent to find.**
>
> `TRACED_INVERSE_MAP` was flipped `False -> True` by `fix/fit-domain-symmetry`
> / the G8-probe work.  That branch identified the resulting class -- pins
> whose fail-before arm goes INERT once the map engages -- and re-scoped
> **five** members to `inverse_map=False`.  **Four more members exist, in two
> files that branch never ran**, and the union merge composed the flip with
> them unchanged:
>
> ```
>   FAILED tests/unit/test_niche_d1_tilted_carrier.py::test_tilted_carrier_supplies_the_beam_centre_by_default
>   FAILED tests/unit/test_niche_d7_decentred_fit.py::test_the_decentred_path_really_did_change
>   FAILED tests/unit/test_niche_d7_decentred_fit.py::test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order
>   FAILED tests/unit/test_niche_d7_decentred_fit.py::test_c13_cures_the_hard_mask_fold_at_the_d7_order
>   4 failed, 66 passed
> ```
>
> Deterministic, reproduced three independent ways, in ~5 s to 5 min, with no
> plugin and nothing monkeypatched.  All 70 ids collect under
> `-m "not integration and not slow"`, so **CI's ordinary unit shards will see
> these red**.  Forcing `TRACED_INVERSE_MAP=False` on the same tree gives
> **70 passed, 0 failed** -- the flag flip is the whole cause.
>
> **It is P1 and not P0, and that is a measurement, not a judgement.**  I
> instrumented `build_inverse_map` inside the failing run: the map ENGAGED on
> every failing arm (3/3, 2/2, 2/2, 1 of 2), so the "refuse, never degrade --
> byte-identical" contract is not what broke.  On the d7 fixture the physical
> output is BYTE-STABLE across the flag (off-beam ratio 0.000176 either way,
> folds 0 either way), and in the SHIPPED configuration the map's own
> degree-14 exit fit returns the least-squares answer exactly
> (`ratio = 1.000000`).  No returned bit is wrong.  Four pins are measuring a
> mechanism the accepted map now supersedes, and one of them
> (`test_c13_...`) is maxing a conditioning claim over a census that silently
> grew by the map's own two solves.
>
> Everything else this pass was asked to check is CONFIRMED: banner both arms,
> zero conflict markers, all 241 modules import, c15 = 40, except budget 48,
> EME census 0/28, R1 g5 = 0.012398, plate byte-null 24/24, ruff clean, zero
> non-ASCII added, no `xfail`/`skip` added, every removed test accounted for.
>
> Four P2s and a release-mechanics list follow.  The P2 worth clearing in the
> same wave: `.test_durations` is missing **383** collected ids.

---

## 1. METHOD, AND WHAT "PINNED" MEANS HERE

`validation/repro_traced_carrier_121/focus_scan_121.py` hard-codes
`sys.path.insert(0, r"D:\...\Lumenairy")` at line 70 -- the MAIN checkout, not
whatever tree it is invoked from.  A banner run started from the worktree
therefore measures the WRONG tree unless the package is imported first.  Every
banner run in this report went through a scratchpad wrapper that

* inserts `C:\tmp\lum_final` on `sys.path`,
* imports `lumenairy` and **asserts** `lumenairy.__file__` starts with that
  root (`SystemExit` otherwise),
* prints the resolved path and the live flag values,
* only then `runpy`s `focus_scan_121.py` under `__main__`.

Both runs printed
`PINNED lumenairy: C:\tmp\lum_final\lumenairy\__init__.py  version=5.34.0`.
The same insert-and-assert pattern fronts every other probe.

Environment: Windows 11, py3.14.6, numpy 2.4.4, MKL, zarr 3.x present,
ruff 0.15.13.  ONE mount -- see S8.

---

## 2. VERDICT TABLE

`CONFIRMED` = re-measured here and it holds.  `DEFECT` = re-measured here and
it does not.  `UNVERIFIED-BY-PROBE` = not established by this pass.

### 2.1 The eight mandated P0/P1 of VERIFY_ARCHITECTURE, on merged main

| id | claim | verdict | the number that settles it |
|---|---|---|---|
| P0-1 | the Nyquist guard's DEFAULT path does not accept an aliased answer | **CONFIRMED** | 4 independent fixtures, 28 pitch rows: the guard cuts at margin 1.0, every O(1) cliff sits below it |
| P0-2 | the margin MOVES with the beam width at frozen pitch and ramp | **CONFIRMED** | 0.805 -> 0.232 over a 250 -> 15 um ladder, a **3.46x** span at constant pitch |
| P0-3 | a partial artifact is refused, not resumed | **CONFIRMED** | holed store integrates to 0.0 W against a recorded 1.413659012e-09; `field_exists` still True, `field_is_complete` False |
| P0-4 | the readout terms are in the keys that consume them | **CONFIRMED** | `chains` key moves on `dx_out`/`n_out` **only** when `capture_reference_tile=True` |
| P0-5 | the inverse-map cache key composes BOTH halves of the G8 bar | **CONFIRMED** | all 14 components move the digest, and the pins FAIL when each half is reverted in-process (S5) |
| P1-1 | the screen-obliquity momentum is OPTICAL, `q = n1 (L, M)` | **CONFIRMED** | `qx/L` = 1.0000000 / 1.5035829 / 1.8046713; independent vector-Snell oracle **739.1x** and **607.4x** immersed against **2.1x / 1.7x** pre-fix |
| P1-2 | payload writes are atomic | **CONFIRMED** | interrupted write leaves the previous artifact bit-unchanged, zero temp leftovers |
| P1-3 | a NaN cannot silently disable the guard | **CONFIRMED** | NaN / all-NaN / +inf all REFUSE; a genuinely ZERO-power field still returns 0.0 and is accepted |

### 2.2 The re-baselined banner

| arm | claim | measured on THIS tree | verdict |
|---|---|---|---|
| shipped default (`TRACED_INVERSE_MAP` True) | 3.450 / 90.3 / 99.8 / 99.9, pk 5.486e+03 | **3.450 / 90.3 / 99.8 / 99.9, pk 5.486e+03**, off (+0.00, +0.00) um | **CONFIRMED** |
| `ARM_IMAP=0` | 3.350 / 90.3 / 99.7 / 99.8 | **3.350 / 90.3 / 99.7 / 99.8, pk 5.529e+03**, off (+0.00, +0.00) um | **CONFIRMED** |

Both at `N=2048 / NFC=8192 / WF=4.0`, `CREF/AM/PIP` unset, grid check clean
("the leg ran at the 8192 it was asked for").  `focus_scan_121.py`'s own
printed acceptance line has been re-baselined in the tree to the 3.450 reading
and now matches what the default run produces.

### 2.3 The union-merge scars

| check | verdict | evidence |
|---|---|---|
| conflict markers anywhere in the tree | **CONFIRMED clean** | `^<<<<<<<`, `^>>>>>>>`, `^\|\|\|\|\|\|\|`, `^=======$` -- all four empty |
| every `lumenairy` module imports | **CONFIRMED** | 219 modules walked, **0 failed** |
| every `validation` module imports | **CONFIRMED** | 22 modules, **0 failed** |
| c15 union is 40 tests, green | **CONFIRMED** | `40 passed in 22.41s` |
| except budget exactly 48 | **CONFIRMED** | `_NON_UI_EXCEPT_BUDGET = 48`; live count re-derived independently = **48** (zero headroom) |
| retired holdout knobs gone | **CONFIRMED** | `hasattr` False for both; one comment mentions them, no code does |
| **the flag flip composed with the pins that measure what it supersedes** | **DEFECT -- P1** | **4 failed, 66 passed**; 70 passed with the flag off (S6) |

### 2.4 The four named re-probes

| probe | claim | measured | verdict |
|---|---|---|---|
| EME census ULP-nudge injector | 0/12 post-fix | **0/28** post-fix (14 arms x 2 cells); pre-fix arm **2/14** on W6 and **13/14** on N16 in the same process | **CONFIRMED** |
| R1 g5 residual | 0.012398 w | `group 5 r=3.0 mm 54.87 mrad: 0.25848 -> 0.090692 -> **0.012398**` | **CONFIRMED** |
| screen-obliquity plate byte-null | exact zero at every tilt | **24/24 rows byte-identical, max abs diff 0.000e+00**, with a live powered-element control | **CONFIRMED** |
| suite counts vs docs | see S4.2 | 40+ suites re-run | **CONFIRMED** (2 doc rows stale UPWARD, explained) |

### 2.5 Hygiene

| check | verdict |
|---|---|
| `ruff check lumenairy/ tests/unit/` (the exact CI command) | **CONFIRMED** -- All checks passed |
| ASCII on every line ADDED since 21802f9 | **CONFIRMED** -- 0 non-ASCII added lines over 71 changed files |
| cp1252 decodability | **1 pre-existing failure**, byte-unchanged by the merge (S7.4) |
| no `xfail`/`skip`/`importorskip` added by the merge | **CONFIRMED** -- the only diff hit is the prose "No xfail, no skip." |
| every test removed by the merge has a live successor | **CONFIRMED for 6 of 7**; the 7th was adjudicated away and replaced by two (S6.4) |

---

## 3. THE MANDATED EIGHT, RE-PROBED

### 3.1 P0-1 / P0-2 -- the Nyquist band guard

Probe `p0_nyquist.py`: four fixtures **none of which is the shipped
`_guard_fixture`** -- w = 150 / 80 / 220 / 300 um, ramps 0.040 / 0.070 / 0.030
/ 0.025, `|R|` = 0.75 / 0.30 / inf (collimated) / 1.20 m, at two source
lattices.  The verdict column is the LIBRARY DEFAULT call with
`warnings.simplefilter('error')`, so a warn-disposition cannot pass as an
accept.  The truth column is a guard-silenced round trip measured
independently.

```
  N=1024 dx=2.0um w=150um R=-0.75 ramp=0.04
     dx_c(um)   N_c   margin   DEFAULT     round-trip relL2
       4.0000   512    2.340   accepted     1.7022e-13
       8.0000   256    1.170   accepted     3.6586e-13
      10.0392   204    0.932    REFUSED     3.6906e-11
      12.8000   160    0.731    REFUSED     1.2208e+00   <== the cliff
      16.0000   128    0.585    REFUSED     1.4142e+00

  N=1024 dx=2.0um w=220um R=inf ramp=0.03   (collimated)
      10.0392   204    1.252   accepted     9.4315e-11
      12.8000   160    0.982    REFUSED     1.2400e-10
      16.0000   128    0.786    REFUSED     1.8775e-01   <== the cliff
```

P0-2, pitch frozen at 12.8000 um and ramp frozen at 0.04:

```
      w(um)      band    margin   DEFAULT
      250.0  0.003418     0.805    REFUSED
      150.0  0.005371     0.731    REFUSED
       80.0  0.009766     0.606    REFUSED
       40.0  0.019531     0.440    REFUSED
       15.0  0.051270     0.232    REFUSED
  margin span 3.46x (0.805 -> 0.232)
```

Over-refusal controls, all five accepted: generous target 2.07e-16, 2x finer
3.65e-13, tiny ramp 1.92e-16, no ramp at all 0.0, coarsen 2x 1.64e-13.

**One row of my own matrix needed adjudication, and it is NOT a guard hole.**
On the fourth fixture the default call accepted at margin 1.93 and 1.54 with
round trips of 6.9e-07 / 9.3e-07 -- above the doc's 1e-8 bar.  Drilled down in
`p0_nyquist_f4.py`: with the SAME grids and NO carrier change at all the pure
resample round trip already reads **7.5e-08 / 1.07e-07**, and driving the ramp
to zero walks the error monotonically onto that floor (0.025 -> 6.9e-07,
0.0025 -> 3.8e-07, 2.5e-05 -> 1.8e-07).  That is the MFT resample's own
accuracy on a beam that fills its window, continuous in the ramp -- not the
discontinuous aliasing cliff, which on the same fixture sits between margin
0.967 (2.06e-06) and 0.770 (8.8e-01), again below the cut.  Recorded so the
next reader is not surprised: **the doc's "0 rows worse than 1e-8" is a
property of its fixtures' resample floor, not a universal property of the
guard.**

### 3.2 P1-3 -- the NaN guard

```
  clean              support=  290.0 um            guard -> ACCEPTED
  one NaN sample     support=  REFUSED(ValueError) guard -> REFUSED "is not finite"
  all NaN            support=  REFUSED(ValueError) guard -> REFUSED "is not finite"
  one +inf sample    support=  REFUSED(ValueError) guard -> REFUSED "is not finite"
  ZERO power         support=    0.0 um            guard -> ACCEPTED
```

The empty-field carve-out is intact and distinguished deliberately, which is
the half a naive fix would have broken.

### 3.3 P0-3 / P1-2 / P0-4 -- the pipeline

```
P0-3
  stored power (in-memory): 1.413659012e-09
  power off the store     : 1.413659012e-09      rel 0.000e+00
  intact  -> complete=True  (power verified)
  removed 1 chunk         : field\envelope\c\0\0
  field_exists (PRE-FIX gate) : True             <== the pre-fix gate still says yes
  power off HOLED store   : 0.0
  holed   -> complete=False
  expect_power=None       -> complete=True  (presence only (no recorded power))
  recorded power off by 1e-12 rel -> complete=True
  recorded power off by 1e-10 rel -> complete=True
  recorded power off by 1e-07 rel -> complete=False       (RTOL 1.0e-09)

P1-2
  good artifact power     : 1.413659012e-09
  save interrupted (simulated taskkill)
  artifact power AFTER    : 1.4136590124186937e-09
  previous good artifact SURVIVED: True
  temp leftovers          : []

P0-4   (base spec d121_3order_ab_scalar.json: dx_out=2e-07, n_out=1024)
  capture_reference_tile=False
    decompose   dx_out x4/3: True    n_out halved: False
    chains      dx_out x4/3: False   n_out halved: False
    aggregate   dx_out x4/3: False   n_out halved: False
    leg/readout dx_out x4/3: True    n_out halved: True
  capture_reference_tile=True
    chains      dx_out x4/3: True    n_out halved: True     <== the fix
```

Interruption was injected by making the underlying writer raise AFTER it had
produced bytes under the temp name -- exactly the window the fix exists for.
A narrow residual window remains and is worth stating: on Windows a directory
cannot be `os.replace`d onto an existing directory, so `save_field` must
`rmtree` the destination immediately before the rename.  A kill inside THAT
window loses the previous artifact -- but leaves NO headless store, so the next
run sees no checkpoint and RECOMPUTES.  The failure mode is a recompute, not
the permanent resume deadlock P1-2 was about.  Not ranked.

### 3.4 P1-1 -- the immersed screen-obliquity momentum

Re-derived with an INDEPENDENT vector-Snell oracle (sphere-quadratic
intersection plus the closed refraction vector), cross-checked against
`lumenairy.raytrace.trace` over 1253 rays x 3 media x 3 angles at
**max dx <= 4.3e-19 m, max dW <= 3.5e-18 m**.

Structural, measured from inside `apply_real_lens`:

| first medium | n1 | carrier L | q_x handed in | q_x / L |
|---|---|---|---|---|
| air | 1.0000000000 | 0.054900 | 0.0549000000000000 | 1.0000000 |
| N-BK7 | 1.5035829054 | 0.054900 | **0.0825467015070213** | 1.5035829 |
| N-SF57 | 1.8046713226 | 0.054900 | **0.0990764556131608** | 1.8046713 |

`q_x - n1*L` is exactly 0.0 in all three, on BOTH code paths and all four
carrier vocabularies.

Physics, exit-plane rms waves against exact rays, piston and tilt removed,
R = 19.6 mm N-SSK2, L = 0.0549, 1.2 mm pupil:

| first medium | blind | pre-fix `q=(L,M)` | shipped `q=n1(L,M)` | gain pre-fix | **gain shipped** |
|---|---|---|---|---|---|
| air | 0.006228 | 0.000009 | 0.000009 | 665.0x | 665.0x |
| N-BK7 | 0.001576 | 0.000743 | **0.000002** | 2.1x | **739.1x** |
| N-SF57 | 0.003710 | 0.002210 | **0.000006** | 1.7x | **607.4x** |

Blind and pre-fix columns reproduce the claim doc to every printed digit.
Angle ladder on N-BK7: 0.0100 -> 2566.2x, 0.0549 -> 739.1x, 0.1000 -> 610.5x,
0.1500 -> 571.7x.

Air control, byte level, 131072 uint64 words per field, against the same call
with the momentum monkeypatched back to the bare cosine: three air-first
prescriptions **byte-identical, 0 / 131072 words differing**; the two immersed
ones differ in **131072 / 131072**.

---

## 4. THE UNION MERGE

### 4.1 What it composed

Diff `21802f9..3154fa7`: 71 files, +16496 / -493.  The imap union (`518226d`)
is the only place two branches edited the same function.  Re-probed at the
mechanism in `p0_imapkey.py`: **every** claimed component moves the composed
digest, with a restored-state control.

```
  incumbent CONTENT / incumbent absent / parity_tag (newton_fit, poly order,
  max iters) / census_amp / census_amp->None / exit degree / launch radius /
  wavelength / landing grid / opl / detJ / weights ............ all True
  TRACED_INVERSE_MAP, INVERSE_MAP_GUARD, _IMAP_PARITY_FACTOR,
  _IMAP_PROBE_MAX, _IMAP_PROBE_MIN, _IMAP_PROBE_R2,
  _IMAP_MIN_SAMPLES_PER_TERM, _IMAP_DETJ_MAXMIN, _IMAP_DETJ_SOURCE ... all True
  (sanity) restored key == base : True
```

The fingerprint is genuinely BY EVALUATION: two DISTINCT closures returning the
same answers hash the same; two of identical shape differing by 1e-10 hash
differently; `None` returns the `<no-incumbent>` sentinel.

`_IMAP_CACHE` hygiene -- the item `FIX_MERGEREF_IMAP` left open ("no lock and
no registry enrollment.  STOPPED AND REPORTED, NOT FIXED") -- is now CLOSED:
`_IMAP_LOCK` is a live `threading.Lock`, the central registry lists 23 clearers
including `inverse_map`, and
`test_v4_14_2_dispatcher_pin_cache_locks` (94 passed) /
`test_v4_16_1_dispatcher_pin_cache_registry_enrollment` (6 passed) /
`test_audit_w4_glass_registry_meshgrid` (16 passed) are all green.

### 4.2 Suite counts, re-run on this tree

| suite | doc claim | measured here |
|---|---|---|
| `test_niche_c15_inverse_map.py` | 40 | **40 passed** |
| `test_audit_except_budget.py` | 2 | **2 passed** (budget 48, live count 48) |
| `test_carrier_field.py` | 41 | **41 passed** |
| `test_pipeline.py` | 48 | **48 passed** |
| `test_screen_obliquity.py` | 47 | **47 passed** |
| `test_v4_16_0_agent_d_cache_registry.py` | 12 | **12 passed** |
| `test_eme_census_determinacy.py` | 7 | **7 passed** |
| `test_eme_2d_vector.py` | 20 | **20 passed** |
| `test_doe_rcwa.py` | 41 | **41 passed** |
| `test_fix_d5_fit_domain_basis.py` | 45 | **45 passed** |
| `test_niche_audit_e_prepared_and_enums.py` | 39 | **39 passed** |
| `test_niche_audit_w3_oracles.py` | 181 | **181 passed** |
| `test_niche_audit_w6_berreman.py` | 430 | **430 passed** |
| `test_niche_audit_w6_eme.py` | 76 | **76 passed** |
| `test_niche_audit_w9_eig_vjp.py` | 31 | **31 passed** |
| `test_pmm_m2_window_contract.py` | 20 | **20 passed** |
| `test_v5_14_1_rcwa_deferred.py` | 22 | **22 passed** |
| `test_niche_c6_fit_guard.py` / `c6_stationary_phase_launch.py` | 13 / 21 | **13 / 21 passed** |
| `test_niche_c14_encapsulation.py` | 32 | **32 passed** |
| `test_v4_14_0_dispatcher_pin_apply_lens.py` | 35 | **35 passed** |
| `test_niche_d2_chain_multi.py` | 38 | **38 passed** |
| `test_niche_d1_tilted_carrier.py` | -- | **1 FAILED, 32 passed** |
| `test_niche_d7_decentred_fit.py` | **37/37** (FIX_CI_RECONCILE) | **3 FAILED, 34 passed** |

Whole-tree collection: **11938 collected, 2 deselected**.

Two doc rows are stale UPWARD and both are explained rather than wrong
(`test_screen_obliquity.py` 28 -> 34 -> 47 across two later docs;
`test_niche_audit_w6_eme.py` 75 -> 76 and `test_eme_2d_vector.py` 19 -> 20 from
`fix/jax-nan-pins`).  Both directions are additive; nothing was lost.

`test_niche_d7_decentred_fit.py` is the exception, and it is stale DOWNWARD:
`FIX_CI_RECONCILE` claims 34/37 -> **37/37**; merged main measures **34/37**.

### 4.3 The EME census determinism re-probe

`eme_ulp.py` -- the shipped injector's mechanism widened to **14 arms**
(+/-1, 2, 4, 8, 16, 32, 64) on **both** cells, with the pre-fix arm
(`_CENSUS_BAND=(0,0)`, `_STRUCTURAL_SAT=inf`) run beside it in the same process
so the fail-before cannot go vacuous.  Scored on the shipped criterion (census
SIZE unchanged, every member within 1e-4 relative of a clean member):

| cell | pre-fix flips | **shipped flips** |
|---|---|---|
| W6 (Nx=8, n_scan=3) | 2 / 14 (arms -1 and -8 gain a 4th mode) | **0 / 14** |
| N16 (Nx=16, n_scan=400) | 13 / 14 (clean 4 modes, 13 arms return 5) | **0 / 14** |

Shipped censuses: W6 `[208.2502598, 203.7161764, 156.2813757]` in all 14 arms;
N16 `[205.9749758, 201.8868825, 151.3854746, 146.4214664, 140.5997565]` in all
14 -- including the 205.9749758 the pre-fix path drops.  Residual movement of
accepted values is at most **3.46e-06** in `qz^2` (~2.5e-08 relative), four
decades inside the bar and deliberately left.

`_CENSUS_BAND = (1e-2, 3e1)` and `_STRUCTURAL_SAT = 1e-2` do not exist at
`21802f9`, so the fix is additive at the mechanism.

### 4.4 The R1 g5 residual

`screen_obliquity_derive.py d121`, import pinned, against the local `.zmx`:

```
  group 5 r=1.0 mm 54.87 mrad:  SHIPPED 0.02459 -> snell 0.006971 -> +R1 0.000990
  group 5 r=2.0 mm 54.87 mrad:  SHIPPED 0.10449 -> snell 0.032784 -> +R1 0.004586
  group 5 r=3.0 mm 54.87 mrad:  SHIPPED 0.25848 -> snell 0.090692 -> +R1 0.012398
```

**0.012398 exactly**, and the run regenerated `_screen_obl_d121.json`
byte-identically (`git status` stayed clean, which is itself a determinism
check on the whole d121 derive path).  Groups 0 and 1 (the plates) read exactly
0.000000 on all three arms.

### 4.5 The plate byte-null

`plate_bytenull.py`, deliberately not the shipped fixture: 4 plates (25.4 mm
N-BK7, 3.0 mm N-SF57, 12.0 mm N-LAK22, 1.0 mm N-SSK2) x 6 carriers including a
skew (0.07, -0.05) and 150 mrad, at N = 128 / 192 / 256, dx = 8-25 um, with
`set_fft_auto_promote(False)`:

**24 / 24 rows byte-identical, max abs diff exactly 0.000e+00.**

Non-vacuity, same code path, POWERED elements at the same tilts: R = 19.6 mm
N-SSK2 differs by 2.424e-01 at L = 0.0549 and 6.868e-01 at L = 0.1; R = 50 mm
N-BK7 by 1.024e-01 and 2.427e-01.  All six carrier-free / zero-angle /
`screen_obliquity=False` controls are byte-identical to the plain call.

---

## 5. NON-VACUITY: THE PINS FAIL WHEN THE MECHANISM IS TAKEN AWAY

A key is not a number, so "the test passes" proves nothing about a key unless
the test also fails without it.  A scratchpad pytest plugin (`revert_plugin.py`,
`-p revert_plugin`, nothing written to the repo) reverts one half of the
composed imap key at a time and re-runs the eight cache-key ids:

| revert | result |
|---|---|
| none (control) | **8 passed** |
| drop `incumbent_fp` | **1 failed** -- `test_the_incumbent_fingerprint_is_by_evaluation_not_by_parameter_name`, two distinct incumbents hashing to the same digest `4ff9d399...` |
| drop `parity_tag` | **1 failed** -- `test_the_cache_key_tracks_the_incumbent_by_content[newton_max_iters-40-True]`: "hits 1 -> 2 ... the incumbent is not fully in the key, so G8 can be bypassed" |
| drop `census_amp` | **1 failed** -- `test_the_cache_key_moves_with_the_det_j_source_and_the_census` |

Each half is independently load-bearing on merged main.  H7 and H8 are the pins
the merge commit names, and they are the ones that die.

---

## 6. THE P1 -- THE HALF-COMPOSED DEFAULT FLIP

### 6.1 The finding

`TRACED_INVERSE_MAP` ships `True` on merged main (`_lens_imap.py:196`); it was
`False` at `21802f9`.  `FIX_G8_PROBE` S6.1/S6.2 identified the class the flip
creates -- pins whose fail-before or non-vacuity arm goes INERT once the map
engages -- and re-scoped **five** members to `inverse_map=False`
(`test_the_arbiter_is_announced_when_the_basis_cannot_honour_it`, two
`test_an_inert_fit_domain_knob_is_announced_not_silent` params,
`test_guard_raises_the_fit_order_like_d7`,
`test_the_two_newton_fit_backends_still_describe_the_same_map`).

`FIX_CI_RECONCILE` S6.1 E2 states in as many words: **"Campaign S11 records d7
and c6 were never run at all on that branch."**  c6 was picked up.  d7 was not,
and neither was d1.  The union merge composed the flip with them unchanged.

```
  $ pytest tests/unit/test_niche_d1_tilted_carrier.py \
           tests/unit/test_niche_d7_decentred_fit.py -q -p no:randomly
  FAILED ...test_niche_d1_tilted_carrier.py::test_tilted_carrier_supplies_the_beam_centre_by_default
  FAILED ...test_niche_d7_decentred_fit.py::test_the_decentred_path_really_did_change
  FAILED ...test_niche_d7_decentred_fit.py::test_the_fold_regularisation_is_still_load_bearing_at_the_d7_order
  FAILED ...test_niche_d7_decentred_fit.py::test_c13_cures_the_hard_mask_fold_at_the_d7_order
  4 failed, 66 passed, 7 warnings in 298.35s
```

Reproduced three independent ways (a sequential batch, a plugin run, a bare
run), deterministically, with nothing monkeypatched.  All 70 ids collect under
`-m "not integration and not slow"`, so **CI's ordinary unit shards see this.**

The flag is the whole cause:

```
  IMAP_FLAG=0  (TRACED_INVERSE_MAP forced False)  ->  70 passed, 0 failed
  IMAP_FLAG=1  (the shipped default)              ->  4 failed, 66 passed
```

### 6.2 What each failure actually asserts

| pin | assertion | measured | shape of the failure |
|---|---|---|---|
| d1 `..._supplies_the_beam_centre_by_default` | `max\|derived - origin\| > 1e-6 * scale` | **4.155e-08** vs a 1.02e-06 bar | the origin-vs-derived beam centre stopped being a materially different computation |
| d7 `test_the_decentred_path_really_did_change` | `max\|a - b\| > 1e-8 * scale` | **exactly 0.0** | the decentred path became BYTE-IDENTICAL to its control |
| d7 `..._fold_regularisation_is_still_load_bearing...` | `folds_pre >= 1` | **0** | "the pre-C13 arm stopped folding" |
| d7 `test_c13_cures_the_hard_mask_fold...` | `ratio_good <= 1.001` | **1.029199** (rcond 1.797e-14) | "the weighted restriction stopped conditioning the solve" |

The first three are fail-before / non-vacuity arms going inert -- the same
sentence `FIX_G8_PROBE` S6.2 wrote about the c6 sibling ("the D7 raise became
byte-identical with the map on").

### 6.3 Why this is P1 and not P0 -- measured, not asserted

**(a) The map ENGAGED; the refuse-never-degrade contract is not what broke.**
`build_inverse_map` instrumented inside the failing pytest run:

```
  ===== INVERSE-MAP BUILD CENSUS (flag=True) =====
  test_tilted_carrier_supplies_the_beam_centre_by_default      builds=3 engaged=3 refused={}
  test_c13_cures_the_hard_mask_fold_at_the_d7_order            builds=2 engaged=2 refused={}
  test_the_decentred_path_really_did_change                    builds=2 engaged=2 refused={}
  test_the_fold_regularisation_is_still_load_bearing...        builds=2 engaged=1 refused={'G7': 1}
```

Separately confirmed on a fixture where the map IS refused (G7): the flag-ON
and flag-OFF fields are **byte-identical, max abs diff 0.000e+00** -- the
"refuse, never degrade, byte-identical to `TRACED_INVERSE_MAP=False`" claim in
the refusal message holds.

**(b) The physical output does not move.**  On the d7 fixture the off-beam
ratio -- the quantity the C13 cure exists to protect -- is **0.000176 under
both flag settings**, and the fold-caustic warning count is **0 under both**.

**(c) The one conditioning claim that failed is being maxed over a census that
grew.**  `_solve_census` instruments `_solve_lstsq_thread_safe` and
`_worst_draw` takes the MAX over every solve in the call.  With the map on, the
map's own solves enter that census:

```
  TRACED_INVERSE_MAP=False   rows=6  worst ratio=1.000003
      shapes: (203401,66) x4, (1852,14), (39765,28)
  TRACED_INVERSE_MAP=True    rows=8  worst ratio=1.029199   <== row 7
      shapes: the same 6, PLUS (203401,120) x2      <-- the map's degree-14 exit fit
```

`(203401, 120)` is a total-degree-14 2-D fit -- `(14+1)(14+2)/2 = 120` terms.
The C13 claim about the DECENTRED FORWARD fit is now scored against the map's
fit.  And in the **shipped configuration** (the test forces
`LSTSQ_CONDITIONING_STEPDOWN=False` and degree 4; users get `True` and 6) that
same map fit returns the least-squares answer **exactly**:

```
  SHIPPED CONFIG, TRACED_INVERSE_MAP=True
    [map deg-14 fit] rcond=1.798e-14 ratio=1.000000
    [map deg-14 fit] rcond=1.798e-14 ratio=1.000000
    only row over 1.001: (1852,27) ratio 1.035359 -- present under BOTH flags
```

So the map's exit fit is ill-conditioned by construction and correctly
regularised by C13's step-down in the configuration that ships.  No returned
bit is wrong anywhere in this finding.

### 6.4 The rest of the sibling sweep

Everything else in the traced-carrier family that completed is green at the
shipped default: `c5` 29, `c6_fit_guard` 13, `c6_stationary` 21, `c10` 9,
`c11` 21, `c12` 20, `c13` 20, `c14` 32, `d2` 38, `d3` 41, `d4` 59, plus the
`fix_d5` 45 / `c15` 40 files the flipping branch DID run.  So the class is
NOT diffuse -- it is concentrated in the two files nobody ran.  The remaining
d/s8/hammer/p files were still running when this was written -- see S8.10.

Seven `def test_` lines vanish in the merge diff.  Six have live successors
(`..._on_physical_tensors` and `..._in_the_degenerate_fallback` re-stated in
place; `test_pmm1d_angle_gradient_at_exactly_zero_stays_bounded` ->
`..._is_an_OPEN_defect`; `test_the_map_beats_the_incumbent_on_held_out_ray_samples`
-> `..._at_off_lattice_probe_points`; `test_a_degree_too_low_to_reach_parity_refuses`
gained a `degree` parametrisation; `test_nonparaxial_f_positive_byte_identical_to_historical`
-> `..._matches_the_historical_form_to_its_own_floor`).  The seventh,
`test_guard_fires_on_the_steep_large_angle_case`, was deleted and replaced by
`test_guard_fires_with_the_correction_ON_when_it_cannot_rescue_the_call` and
`test_the_guard_does_not_warn_about_a_call_the_correction_fixed` -- see P2-2.

### 6.5 The suggested fix, for the next wave

Bounded and mechanical, in the shape the campaign already used five times:

1. scope the three d7 arms and the d1 arm to `inverse_map=False` (the knob the
   five siblings use), keeping every assertion word for word; and/or
2. for `test_c13_...`, filter `_solve_census`'s rows to the decentred forward
   fit's own shape before `_worst_draw` maxes over them, so the claim is about
   the solve it names;
3. then re-run d1 + d7 + c6 + fix_d5 + c15 together at the shipped default, and
   add d1/d7 to whatever gate the flipping branch runs, since their absence is
   how this reached main.

---

## 7. DEFECTS

### 7.1 P0 -- none

### 7.2 P1

**P1-1.  Four deterministic red tests on merged main at the shipped default**
(`test_niche_d1_tilted_carrier.py` x1, `test_niche_d7_decentred_fit.py` x3),
caused by the union merge composing the `TRACED_INVERSE_MAP` default flip with
four members of a pin class whose other five members the flipping branch
re-scoped.  Full evidence in S6.  No returned bit is wrong; the release is
blocked because merged main cannot ship a red suite, and because
`FIX_CI_RECONCILE` claims `test_niche_d7_decentred_fit.py` at **37/37** where
main measures **34/37**.

### 7.3 P2

**P2-1.  `.test_durations` is 383 ids short, concentrated in the files this
campaign grew.**

```
  collected 11938   duration entries 11811
  COLLECTED WITHOUT A DURATION ENTRY: 383
      48  tests/unit/test_pipeline.py              (all of it)
      47  tests/unit/test_screen_obliquity.py      (all of it)
      41  tests/unit/test_carrier_field.py         (all of it)
      41  tests/unit/test_doe_rcwa.py              (all of it)
      40  tests/unit/test_niche_c15_inverse_map.py (all of it)
      28  tests/unit/test_verify_perf_fixes_2026_08_10.py
      26  tests/unit/test_niche_perf_round2_2026_08_10.py
      ...
  DURATION ENTRIES NO LONGER COLLECTED: 256
      (214 of them parametrize-id drift in
       test_v4_14_2_dispatcher_pin_zero_plus_zeroj.py, which still collects 226)
```

`pytest-split` charges an unknown id ~0 s, so five whole files land on one
shard.  This repo has already lost a release-tag verify shard to stale
durations.  Clear it in the same wave as P1-1.

**P2-2.  The screen-obliquity guard was re-calibrated in the LOOSENING
direction, with 2x of margin, on rotationally symmetric surfaces only.**
`_SCREEN_OBLIQUITY_RESIDUAL_FRAC` 0.40 -> 0.10 at `_lens_real.py:1991`.  The
guard fires on `estimate * FRAC > TOL`, so a smaller FRAC warns LESS; the
recorded effect is `fires_on: true -> false` on design-121 group 5.  The
justification is measured and sound (R1 took that group's residual fraction
from 0.351 to 0.048, and the deleted pin's truth was 36x INSIDE tolerance, i.e.
a 37x false alarm).  Ranked because the constant's own comment says the
residual it now bounds "has NOT been measured on a decentred / tilted /
freeform element", and 2x is thin.

**P2-3.  Docstring-vs-code contradictions the merge introduced or left.**  None
changes a returned bit; each will mislead the next reader.

| site | doc says | code says |
|---|---|---|
| `validation/pipeline/doe_rcwa.py:256` | `n_orders` "default 12; the convergence ladder and its chosen headroom are in the build note" | `p.get('n_orders', 6)` at `:154`, under a comment reading "DEFAULT 6 IS THE CEILING, NOT A CHOICE ... It is NOT converged" |
| `lumenairy/elements/_lens_real.py:2896` | the guard fires "when **40%** of it still does" | `_SCREEN_OBLIQUITY_RESIDUAL_FRAC = 0.10` (a 4x error in the documented threshold) |
| `lumenairy/elements/_lens_imap.py:867-877` | G8 accepts "on **held-out samples**", and quotes a refusal string `G8: held-out OPL error ...` | the held-out node probe is retired; the code emits `off-lattice OPL error` / `off-lattice entrance-position error` at `:1521, 1531, 1537`.  The prose is the union-merge scar itself -- the fingerprint branch's docstring landed unedited on the G8-probe branch's rework |
| `validation/repro_traced_carrier_121/imap_prod_121.py:11,156` | "held-out ray samples" | same, reporting side |
| `lumenairy/elements/_traced_flags.py:65-75` | "``v5.32.1`` is the current tree"; "``pyproject.toml`` and ``__init__.py`` both still say ``5.32.0``" | newest era is `v5.34`; both say `5.34.0` |

**P2-4.  The era label `v5.34` names two different trees, and contradicts the
layer map.**  `_traced_flags.ERAS[-1] == 'v5.34'`, and its
`TRACED_INVERSE_MAP` value moved `False` (at `21802f9`) -> `True` (on main), so
`traced_era('v5.34')` is not a stable coordinate across commits.  Meanwhile
`TRACED_LAYER_MAP.md:85-86` dates the same two flags to `5.35.0` while `:97`
and `_traced_flags.py:83` call the era `v5.34`.

**I considered and rejected ranking this P1.**  The reasons are checkable:
`lumenairy/elements/_lens_imap.py` **does not exist at tag `v5.34.0`**
(`git cat-file -e v5.34.0:...` fails), and `ERAS` at that tag is
`('v5.31','v5.32','v5.32.1')` -- so the era cannot be, and never was, a
reproduction of released 5.34.0.  The file has explicit precedent for a
source-only era name that matches no release (`v5.32.1`, documented as such).
The invariant that IS pinned -- `ERAS[-1]` reproduces the live shipped defaults
-- is green (`test_niche_c14_encapsulation.py`, 32 passed), all 33 registry
entries match their live module defaults, and all 33 `TRACED_LAYER_MAP` S2 rows
match.  No recorded result anywhere in `docs/` is labelled `traced_era('v5.34')`.
It is a naming defect in a diagnostic preset, and it is resolved by the era
rename that the version bump makes necessary anyway.

### 7.4 P3

* **`lumenairy/__init__.py` does not decode as cp1252** -- 43 lines of
  double-encoded mojibake.  **Pre-existing and unchanged**: the same 43 lines
  at `21802f9`, and the merge's 42 added lines are pure ASCII.  Nine other
  changed files carry non-ASCII lines, all pre-existing.  Zero non-ASCII lines
  were ADDED anywhere in the 71-file diff.
* **`validation/repro_traced_carrier_121/probe_c6_energy.py:33,38,40,50`** read
  and write `_lt._REMAP_RESID_TAPER_IN` / `_OUT`, neither of which exists --
  an unguarded `AttributeError` on first `run()`.  Pre-existing and already
  documented as stale in `APPROXIMATION_AUDIT_POST_C6_2026_07_31.md:765`.
* **Probe scripts committed into `scripts/`** -- `_d5_probe{,2..6}.py`,
  `_d5_byteid.py`, `_d5_spl.py`, `_g8_probe.py`, `_g8_failbefore.py`,
  `_g8_c15lad.py`.  They are the claim docs' named reproducers, so this may be
  deliberate; noted because the campaign's own convention is "probes in the
  scratchpad, never in the repo" and nothing imports or tests them.

### 7.5 Release mechanics -- stated, not ranked

Not defects; the release owner's step, and the tag cannot be pushed without
them.

* `lumenairy/__init__.py:1102` and `pyproject.toml:7` both read **5.34.0**.
* `CHANGELOG.md`'s `[Unreleased]` section is **EMPTY**, immediately followed by
  `## [5.34.0]`.  71 files and ~16.5k lines are undocumented there; every claim
  doc states "no CHANGELOG edit" by design.  **Both changelog walkers are blind
  to this**: `test_v5_2_walker_changelog_changeset.py:245` skips unless the
  CHANGELOG's top version equals `lumenairy.__version__`, and both read
  `5.34.0`, so it audits the already-released 5.34.0 changeset;
  `test_v5_2_3_walker_changelog_content.py` skips for the same reason.  A green
  walker here is not evidence.
* **14 sites already stamp `v5.35.0`** and must not be left contradicting the
  version: `_lens_real.py` :1923, :2480, :2846, :2862, :2984, :2985, :3826,
  :4152; `_lens_imap.py:1034`; `_lens_traced.py:7936`; `polarization.py:422`,
  :429; `propagators/system.py:834`; `tests/unit/test_screen_obliquity.py:1`,
  :585, :987; `docs/audits/TRACED_LAYER_MAP.md:85`, :86.
* `_lens_imap.py:958-959` hashes `__version__` into the inverse-map cache key,
  so the stale stamp is load-bearing, not cosmetic: every cached map built
  before the bump keys under `5.34.0`.

---

## 8. NOT PROBED -- what this verdict does NOT cover

Named so silence is not mistaken for coverage.

1. **ONE MOUNT.**  Windows / MKL / py3.14.6 / numpy 2.4.4 only.  No WSL, no
   OpenBLAS, no second BLAS, none of CI's 3.10-3.13 legs.  Everything
   BLAS-sensitive here -- the EME census above all -- is confirmed on one
   arithmetic.
2. **The full fast gate was not run.**  ~40 suites plus the whole-tree
   collection; `-m "not integration and not slow"` in full is ~3 h.
3. **No GPU / CuPy path** anywhere: not the screen obliquity, not R1, not the
   inverse map (`use_gpu=True` forces `newton_fit='polynomial'` and is excluded
   from the imap gate by design).
4. **No JAX-guarded file was separately verified**; 14 unit files carry a
   module-level `pytest.importorskip('jax')`.
5. **The 32-order design-121 fan was not re-run** -- only the two banner arms.
   The RCWA-vs-scalar A/B (~3.3 h per arm) was not re-run either, so the 134 %
   delta `BUILD_RCWA_DOE_TABLE` flags stands unexamined here.
6. **`prepare_real_lens` / `PreparedAnalyticLens` and
   `JonesField.apply_real_lens`** were not probed for the new keywords.
7. **The EME census's own open item** -- clear-accept `qz^2` still coming from
   the minimiser's stopping point and moving ~1e-6 across builds -- was
   observed (max 3.46e-06) but not chased; it is deliberate per the fix doc.
8. **Decentred / tilted / biconic / freeform surfaces** were not measured for
   either the obliquity correction or R1, which is what makes P2-2 a P2.
9. **The jitter injector, the uniform-`nextafter` control and the Nx=20 census
   ladder are not in the tree** -- the shipped fail-before is the bracket half
   on one cell.  I widened the bracket half to 14 arms on two cells; the jitter
   arms remain doc-side.
10. **The tail of the traced-carrier sweep was still running when this was
    written.**  Completed and green: `c5, c6_fit_guard, c6_stationary, c10,
    c11, c12, c13, c14, d2, d3, d4`.  Still running or not started:
    `d5, d6, d8, d9, s8, tight_focus, fix_tilt_quadratic_opl,
    hammer h1/h2/h3/h6, p1_traced_tiltaware, p2_design_battery,
    p2_displaced_extreme, p2_guards, p10, p11, e4, w9_traced_determinism`, and
    `test_audit_lens_models_2026_07.py` (which `FIX_VERIFY_ARCH` S10 itself
    called "the ONE outstanding suite") was not started.  Given P1-1's
    mechanism, **the fix wave must run that whole family at the shipped
    default** -- more members of the class may be there.  The eleven that did
    complete suggest the class is concentrated rather than diffuse.

---

## 9. PROBE INVENTORY

All under the session scratchpad, never in the repository:

* `pinned_banner.py` -- the import-pinned `focus_scan_121.py` driver with the
  `SystemExit` assert that makes "pinned" checkable; outputs
  `banner_A_default.txt`, `banner_B_arm0.txt`.
* `p0_nyquist.py`, `p0_nyquist_f4.py` -- P0-1 / P0-2 / P1-3, the over-refusal
  controls, and the adjudication of my own flagged rows.
* `p0_pipeline.py` -- P0-3 / P0-4 / P1-2 off the `artifacts` API.
* `p0_imapkey.py` -- P0-5 key composition, the by-evaluation fingerprint, the
  retired-knob check, `_IMAP_CACHE` lock / registry / order hygiene.
* `revert_plugin.py` -- the non-vacuity harness of S5.
* `imapflag_plugin.py`, `imapspy_plugin.py` -- the flag-forcing and
  engage/refuse-census harnesses of S6.
* `d1d7_engage.py`, `d7_census.py`, `d7_census2.py` -- the P1 adjudication.
* `eme_ulp.py` -- the widened 14-arm ULP-nudge census probe with its pre-fix arm.
* `plate_bytenull.py` -- 24 plate rows plus the powered non-vacuity control.
* `cp1252_sweep.py`, `import_sweep.py`, `collected.txt` -- the hygiene sweeps.
* Immersed-momentum probes (structural, oracle cross-check, physics,
  end-to-end + air byte control) under the same directory.

Worktree state at the end of this pass: `git status --porcelain` **empty**.
