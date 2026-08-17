# FIX -- design-121 audit S11 items 1, 2, 3, 5, 6

Implementation wave against
`docs/audits/AUDIT_DESIGN121_MODEL_CONVERGENCE_2026_08_17.md`, which is
committed alongside this file.  Branch `fix/d121-audit-items`, worktree
`C:/tmp/lum_au`, lumenairy 5.38.1, Windows 11 / py 3.14.6 / numpy 2.4.4 on
tesla-ryzen.  Binding law: `docs/TESTING_STANDARDS.md`.

Every number here was RE-MEASURED on this build.  Three of the audit's own
readings did not survive that, and they are called out in S7.

---

## 1. What shipped

| item | where | state |
|---|---|---|
| 1+2 preflight: free-RAM floor, phantom removal, `sag_dtype` term | runner `tx_design_study_sim.py` (outside git) | **shipped**, two-sided test |
| 1 lib: production-grid check path for the float32 validator | `lumenairy/elements/_lens_real.py` | **shipped**, 10 tests |
| 3 `d121_32order.json` `dx_common` + the CLASS fix | `validation/pipeline/specs/`, `tests/unit/` | **shipped**, 11 tests |
| 5 watchdog hardening | `Reverse_Symmetric_ASM/ram_watchdog.py` | **shipped**, test PASSes |
| 6 `sag_chunk_rows` on the analytic path | measurement only | **measured; no change proposed** |

---

## 2. Items 1 + 2 -- the preflight (landed as ONE change)

S9.2 trap 3 is real and was obeyed: the phantom `screen_obliquity` term was
the check's only headroom, so the floor went in with its removal.

### 2.1 The floor derivation

`FREE_RAM_FLOOR_BYTES = 20.0e9`, enforced as `free - need > FLOOR`.  It is
ABSOLUTE because what it covers does not scale with the grid; the pre-existing
`safety_factor = 1.15` MULTIPLIES `need` and covers a different thing.  From
the measured 2026-08-17 failure (audit S7.1, box B, 136.6 GB):

| quantity | value |
|---|---|
| free at launch | 120.5 GB (so 16.1 GB committed then) |
| `need` (peak 100.5 x 1.15) | 115.6 GB |
| `free - need` | **4.9 GB** |
| observed process RSS at peak | ~104 GB |
| system usage when the watchdog fired | ~129 GB (so ~25 GB non-run) |

Two terms, both measured on that run:

* **(a) baseline-commitment drift, +8.9 GB.**  The box committed 16.1 GB at
  launch and 25.0 GB at peak.  `free` is a launch-time snapshot, so the drift
  is exactly what it cannot see.
* **(b) operational reserve, 10.0 GB.**  Below ~10 GB free this box's failure
  mode is silent process death, which is why the run watchdog kills there.  A
  preflight that admits a run whose success requires crossing the watchdog's
  own floor has admitted a run that will be killed.

(a) + (b) = 18.9, rounded up to **20.0 GB**.

Estimate error is deliberately NOT in the floor: the same run measured 100.5
predicted vs ~104 observed (+3.5 %), and the 1.15 factor already reserves
15.1 GB against that -- 4.3x the measured error.  Folding it in twice would
double-count.

**Two-sided, asserted rather than argued** (`preflight_floor_test.py`, run
against the live pricing with the box's free RAM patched):

```
B/D. the 2026-08-17 N=32768 run that died (free 120.5 GB)
  preflight: need ~115.6 GB (peak 100.5 GB x 1.15, analytic lens
             [tangent_facet+66.1 GB]), have 120.5 GB free,
             headroom +4.9 GB against a 20.0 GB floor.        -> REFUSED
C. exp31 (N=16384, same route) must still be ADMITTED
  preflight: need ~28.9 GB ... headroom +91.6 GB              -> ADMITTED
```

The predicted peak and `need` reproduce the audit's 100.5 / 115.6 GB to the
printed digit, which is the check that the pricing change is the one the
audit described.  (The runner's default `FIELD_DTYPE` is `complex128`; every
design-121 run of record used `complex64`, and the test patches it rather
than assuming -- at complex128 the same call prices 155.1 GB.)

### 2.2 The phantom

Removed, version-gated at lumenairy >= 5.37.0, where `_obl_active` excludes
the tangent-facet family at the source (re-verified in the installed 5.38.1
source, not read from the audit).  Below 5.37.0 the block still runs under
the family and the term stays.  The test asserts the discrimination in both
directions: not priced under `tangent_facet`, still priced on the
vertex-plane route.

### 2.3 The `sag_dtype` term -- and why it prices ZERO

Warmed tracemalloc, the shipped anchors' protocol, biconvex N-SSK2 singlet
R = +19.6 / -27.4 mm, 3 mm aperture.  TOTAL peak in float64 grids
(8*N*N bytes); CREDIT = float64 peak - float32 peak.

| route | N=2048 (whole) | credit | N=4096 | credit | N=8192 | credit |
|---|---|---|---|---|---|---|
| no surface model, no carrier | 14.127 / 9.126 | **5.001** | 6.384 / 6.196 | 0.188 | 6.383 / 6.196 | 0.187 |
| vertex screen + carrier | 27.252 / 22.253 | **4.999** | 11.081 / 9.893 | 1.188 | 11.080 / 9.892 | 1.188 |
| `tangent_facet`, no carrier | 24.127 / 20.628 | 3.499 | 10.503 / 10.408 | 0.095 | 10.501 / 10.407 | 0.094 |
| `tangent_facet` + carrier | 24.127 / 20.628 | 3.499 | 14.002 / 14.001 | **0.001** | 14.001 / 14.001 | **0.000** |

The probe reproduces the shipped ANCHOR 2026-08-16c exactly where they
overlap (+10.00 grids whole-grid at N=2048, +7.62 banded finite-radius
carrier at N=4096), so the fixture and protocol are the anchors' own.

Priced as the MINIMUM measured across each regime, rounded DOWN, because a
credit is the one quantity a preflight must UNDER-state.  Banded (N >= 4096):
0.0 for the tangent-facet family, 1.1 vertex+carrier, 0.1 baseline.
Whole-grid (N < 4096): 3.4 / 4.9 / 5.0.

`LENS_SAG_DTYPE` is wired to the analytic call, DEFAULT OFF, and only passed
when set, so the default call stays byte-identical.

---

## 3. Item 1, library side -- the production-grid check

**Adjudication: a true production-grid field A/B is NOT achievable, and the
proxy does NOT bound the production case.**  Both halves measured.

`lens_sag_float32_opd_error` runs two checks.  Check 1, the radial OPD scan,
already IS a production-grid result: it runs axis-to-clear-aperture-edge and
costs nothing.  Check 2, the field A/B, has a WINDOW, and at a production
pitch the window that covers a production aperture IS the production grid --
design 121's groups are 20.4-31.8 mm across at dx = 0.9028 um, i.e.
`field_check_n` of 22592-35200.  Running that A/B needs two such grids and is
precisely the run float32 was supposed to make affordable.

So the achievable fix is to make the proxy declare itself.  Added:
`aperture_cover`, `field_check_window_m`, `field_check_n_for_full_aperture`,
`field_check_covers_aperture`, `field_rel_error_estimate`, and
`on_partial_aperture` (`'warn'` default / `'error'` / `'silent'`) which fires
whenever the window does not cover the clear aperture.  The policy is
validated by VALUE, not identity -- the same defect the audit's S9.1 #1 found
at `_check_screen_obliquity_support`, with a test that passes a runtime-built
`'warn'`.

---

## 4. Item 3 -- the stranded spec, and the class fix

### 4.1 The instance was bigger than the audit said

The audit names `d121_32order.json`.  **All four shipped specs carried
`dx_common = 1.2292e-06`**, and the three that can be certified all bind at
the same pitch because they all include order (-4,-2).

Requirement computed from the guard's own arithmetic, not quoted: the exp29
aggregate ledger records `worst_nyquist_margin = 1.1081836097493127` at
`dx_common = 1.0e-06`, so the binding pitch is **1.108184 um**.  At 1.2292 um
the margin is **0.9015** -- refused under `on_nyquist='error'`.
`git log 0f46efb..HEAD -- lumenairy/propagators/carrier_field.py` is EMPTY,
so that measurement is valid against 5.38.1.

**dx alone cannot fix it.**  Containment needs a 4.5776 mm half-window (worst
|chief ray| 1.9351 mm + 2.6425 mm support); 8192 x 1.108 um gives 4.539 mm
and fails by ~40 um.  Set to **`dx_common = 1.0e-06`, `n_common = 12288`** --
margin 1.1082, containment +1.566 mm, and both are the exp29 run of record's
own values, so the configuration is known to complete.

`d121_3order_probe.json` is left alone and registered UNCERTIFIED: its chains
are `cached_aperture` replays of an archived rs=4 field, so the witness does
not apply, and the archive (~1 GB/order) is not in the tree.  **That spec is
still stranded** -- now visibly rather than silently.

### 4.2 The class fix

`tests/unit/test_pipeline_spec_guard_validity.py` (11 tests) plus
`validation/pipeline/specs/_measured_nyquist.json`.

The witness carries the guard's INPUTS -- both carriers, each beam's support
radius and envelope half-band -- measured on exp29.  Everything but
`env_band` is read straight from that run's artifacts; `env_band` is
recovered by inverting the guard at the pitch that run used, which is exact
because it is the only free quantity there.  **All 32 beams reproduce their
recorded `nyquist_margin` to better than 1e-12 relative on 5.38.1**, and a
test asserts that, so the witness cannot rot.

The load-bearing arm calls `carrier_difference_nyquist` LIVE at each spec's
own `dx_common`.  A future tightening of that arithmetic -- a new additive
term, a different `_BAND_HEADROOM`, a different binding candidate -- fails in
CI in 0.23 s instead of at hour seven of a run.  A second arm reproduces
`re_reference`'s containment arithmetic, which is what decides `n_common`.
A third refuses any spec whose chain signature the witness was not measured
on unless it is named in an `UNCERTIFIED` registry with a written reason.

Fail-before demonstrated: restoring `1.2292e-06 / 8192` fails
`test_the_live_nyquist_guard_admits_every_certified_spec[d121_32order.json]`
and `test_no_shipped_spec_still_carries_the_stranded_pitch`, 2 failed /
9 passed.

**Scope, stated:** the witness does not re-measure `_enclosed_band_radius`
(that needs the 6.7 GB of chain fields), so a change to how the envelope band
itself is measured would pass.  Every other way the binding pitch can move --
including the way it moved in 0f46efb -- is covered.

---

## 5. Item 5 -- the watchdog

Found, not ephemeral: the predecessor was an inline shell loop staged at
`C:\Users\Tesla\AppData\Local\Temp\j7.sh` (2026-08-17 00:58), which selected
victims box-wide by command-line substring:

```python
for p in psutil.process_iter(['name', 'cmdline']):
    if p.info['name'] == 'python.exe' and any('run_poc' in c
                                              for c in p.info['cmdline']):
        p.kill()
break
```

Four defects, all fixed in `Reverse_Symmetric_ASM/ram_watchdog.py`:

1. selection by cmdline substring, box-wide -- would equally have matched an
   unrelated job.  Now the watchdog OWNS the run (it spawns it), so the
   victim set is that PID's own process tree.  `--attach PID` for a run
   already going.
2. no verification of death -- `p.kill()` fire-and-forget inside a bare
   `except Exception: pass`, which is why a 103.8 GB process was reported
   killed and was not.  Now: kill, `wait`, escalate to `taskkill /F /PID`,
   then `/F /T /PID`, verifying between each, with a distinct exit code (126)
   when a process survives everything.
3. `break` after one pass.  Now the reap runs until the tree is empty, and
   re-walks it for late children.
4. a 15 s poll on a process allocating tens of GB per step.  Now 3 s with an
   N-consecutive-sample confirm so one transient dip cannot end a 12 h run.

The allocator is chosen as the **largest-RSS member of the tree** -- the kill
that actually returns the memory -- and reaped first.

**Test evidence** (`ram_watchdog_test.py`, the shape that defeated the
predecessor: a small wrapper and a large allocating child):

```
  tree census (2 process(es), largest RSS first):
    PID 40948    python.exe           RSS     0.73 GB
    PID 38924    python.exe           RSS     0.02 GB
  ALLOCATOR = PID 40948 (python.exe, RSS 0.73 GB) -- killing it FIRST
  allocator PID 40948 VERIFIED dead after kill()
  tree member python.exe PID 38924 VERIFIED dead after kill()
EXIT 125 in 7.5s
  post-mortem ALLOC PID 40948: dead
  post-mortem WRAPPER PID 38924: dead
RESULT: PASS
```

No real memory pressure is created: the floor is set above the box's current
free RAM, and `--grace-s` lets the allocator reach its RSS first so the
allocator-vs-wrapper choice is a genuine discrimination.

---

## 6. Item 6 -- `sag_chunk_rows` on the analytic path

Warmed tracemalloc, N=8192, `tangent_facet` + finite-radius carrier, same
fixture.  AUTO resolves to `max(256, N//16) = 512` rows here (2048 at
N=32768).

| `sag_chunk_rows` | peak (grids) | peak (GB) | wall (s) | byte-identical to AUTO |
|---|---|---|---|---|
| 256 | 14.001 | 7.517 | 46.0 | yes |
| **512 (AUTO)** | **14.001** | **7.517** | 42.6 | (reference) |
| 1024 | 14.001 | 7.517 | 44.2 | yes |
| 2048 | 14.001 | 7.517 | 45.0 | yes |
| 4096 | 16.066 | 8.625 | 45.9 | yes |
| 0 (whole-grid) | 24.125 | 12.952 | 42.4 | yes |

**NO CHANGE PROPOSED, and the audit's framing of this as a "lever" does not
survive measurement.**  The working set is FLAT from 256 to 2048 rows: below
2048 the band is not the peak-setting allocation, so tightening AUTO buys
nothing, and wall clock rises slightly as chunks shrink -- a smaller default
would be strictly worse.  The curve only moves upward: N/2 costs +2.07 grids
and whole-grid +10.12.  AUTO is already at the flat part.

*Byte-identity note.* One run of the sequence reported whole-grid differing
from banded.  It did not reproduce: a dedicated sweep (N = 2048 / 4096 / 8192
x carrier / no carrier, `sag_chunk_rows=0` vs `256`) returned
`np.array_equal` True with `max|d| = 0` on all six arms, and a repeat of the
original sequence returned identity on every arm including whole-grid.  The
outlier occurred while other jobs were on the box, and the library's own
banded-halo test documents FFT plan determinism as a precondition for
byte-equality across calls.  Recorded rather than dismissed; not attributed,
because the mechanism was not proven.

---

## 7. Where the audit was wrong

Re-measurement contradicted three of its readings.  All three point the same
way -- the audit's item 1 is **not** the ready-to-go lever it describes.

### 7.1 `sag_dtype=np.float32` does not halve the geometry stack -- at N >= 4096 it saves nothing

S7.3 / S11 item 1: *"`sag_dtype=np.float32` would halve the geometry stack to
~50 GB, which is comfortably inside the box"*; *"Halving that stack should
put N=32768 near 55-60 GB."*

Measured (S2.3): on `tangent_facet` + carrier the credit is **0.001 grids at
N=4096 and 0.000 at N=8192**.  From v5.37 the analytic path row-bands at
N >= 4096, so the geometry stack is already one band deep and its dtype
barely reaches the peak.  float32 is worth ~3.5 grids only BELOW the
auto-band threshold, and ~1.19 grids on the vertex-plane+carrier route.

**Consequence: item 1 does not unblock N=32768 on the route the run needs.**
The knob is shipped, wired and priced -- and correctly prices zero.

### 7.2 The "747x margin" is a measurement over ~2 % of the pupil

S7.3 quotes worst field relative error 1.338e-06 against a 1e-3 bar at
`field_check_n=512, field_check_dx=0.90 um`, calling it "the production
sampling".  It is the production PITCH; the WINDOW is 512 x 0.9028 um =
0.4622 mm, against clear apertures of 20.4-31.8 mm -- **1.5 to 2.3 % of the
pupil diameter**, and the sag error is largest at the edge that window never
reaches.

The OPD column reproduces exactly (S25-S27 7.7376e-04 waves vs the audit's
7.738e-04).  The field column does not survive the window:

| `field_check_n` | window | cover of the 20.4 mm S25-S27 aperture | field rel error |
|---|---|---|---|
| 512 | 0.462 mm | 2.3 % | 1.1221e-06 |
| 1024 | 0.925 mm | 4.5 % | 5.3847e-06 |
| 2048 | 1.849 mm | 9.1 % | 2.9377e-05 |
| 4096 | 3.698 mm | 18.1 % | 1.2196e-04 |

**109x over three doublings, still climbing at ~4.6x per doubling with 82 %
of the pupil unseen.**  The grid-free full-aperture estimate
`2*pi*7.7376e-04 = 4.862e-03` is ~4.9x ABOVE the 1e-3 gate the audit reports
clearing by 747x, and on a controlled fixture the field reading CONVERGES
once cover reaches 1 (+0.1 % from cover 1.00 to 2.00, against 21x below it)
-- so cover, not pitch, is the sufficiency criterion.

`apply_real_lens` was already warning "prescription aperture(s) exceed the
simulation grid" on every one of those calls.  The warning said the window
was small; nothing said the VERDICT was therefore not a verdict.  It does
now.

Judged on OPD rather than field, the audit's other statement stands: 7.74e-04
waves is well under `tangent_facet`'s own 0.0032-wave residual.  Both are
true; they answer different questions.

### 7.3 `sag_chunk_rows` is not an unused lever

S7.4 calls it "a second, unused lever ... Tightening it cuts the band working
set."  Measured flat from 256 to 2048 rows at N=8192 (S6).  There is nothing
to cut.

### 7.4 A smaller correction

S9.1 #2 names one stranded spec.  All four shipped specs carried the same
pitch (S4.1).

---

## 8. What is still open

* **`d121_3order_probe.json` remains stranded** and uncertified: its archived
  rs=4 arm-A fields are not in the tree, so its guard requirement cannot be
  measured here.  Rebuild the archive (`python sumap_probe_121.py arma`) or
  retire the spec.
* **Audit item 4** (the residual 2.4 % between `tangent_facet` and traced) is
  untouched, and S7.1 above makes it harder, not easier: the grid half still
  needs N=32768 analytic, and float32 does not buy it.  The remaining levers
  are a bigger box or a route that bands the whole peak.
* The one non-reproducing byte-identity reading in S6.
