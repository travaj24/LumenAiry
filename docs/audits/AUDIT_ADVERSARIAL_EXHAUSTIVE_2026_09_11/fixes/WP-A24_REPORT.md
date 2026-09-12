# WP-A24 -- the d6 on-axis EE2 ratio: what moved it, and what the bar should have said

Subject: `tests/unit/test_niche_d6_exact_tilted_leg.py::test_decentred_carrier_decentre_penalty_envelope`
reading `r_on = 0.969787` against `assert r_on > 0.97`; the routing of that
failure to WP-A6's C1 by `WP-A16_REPORT.md` section 8.3; and the stale
calibration in `carrier.py`'s `decentre_fit_frac` warning
(`WP-A16_REPORT.md` section 8.4 / section 5 items 7-8).

Everything below is a MEASUREMENT taken on this machine with
`OPENBLAS_NUM_THREADS=1`, Python 3.14.6 / numpy 2.4.6, against the d6 file's
own **lumenairy-free** oracle (exact conic raytrace to the `K = -n^2` exit
surface + a direct Rayleigh-Sommerfeld/Kirchhoff sum over it, sharing no FFT
grid, no `window_factor`, no carrier convention and no propagator with the code
under test).  Historical revisions were run by extracting them READ-ONLY with
`git archive <rev> lumenairy` into the scratchpad and running the CURRENT test
file against them, so only the library varies.  Nothing was checked out,
stashed, or written to the repository; no git write command was issued.

---

## 1. Summary

| finding | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| **A24-1** C1 is blamed for the d6 crossing (`WP-A16` 8.3 / 5.7) | **NOT REPRODUCIBLE -- the attribution is wrong** | no code change | `test_audit2609_a24_decentre_calibration.py::test_the_exact_final_leg_never_enters_the_paraxial_focus_standoff_resolver` (+ its falsifier) | d6 inline Kirchhoff oracle | `a18ab074^` **0.969787** == `a18ab074` **0.969787** == HEAD **0.969787**; C1's resolver is never called on this path |
| **A24-2** `r_on > 0.97` is a per-build bar | **FIXED** | `tests/unit/test_niche_d6_exact_tilted_leg.py:663-713` (restated arms), `:621` (docstring) | the restated `test_decentred_carrier_decentre_penalty_envelope` | same, plus its measured floor | one-sided `> 0.97` (red at 0.969787, by 2.2e-04) -> two-sided `0.95 < r_on < 1.02`, derived against a 6.4e-04 oracle floor |
| **A24-3** the shipped `decentre_fit_frac` warning quotes a stale, now-inverted calibration (`WP-A16` 8.4 / 5.8) | **FIXED** | `lumenairy/propagators/carrier.py:6707-6740` (docstring), `:6768-6786` (the message) | `test_audit2609_a24_decentre_calibration.py::test_the_shipped_decentre_warning_quotes_the_re_measured_calibration`, `::test_the_warning_names_the_switch_that_sets_the_ordering` | same | `0.997 / 1.002 / 1.005 / 0.977 / 0.983 / 0.923` (2026-07-29) -> `0.970 / 1.010 / 1.008 / 1.002 / 0.986 / 0.903` (2026-09-12) |
| **A24-4** where the 0.997 -> 0.9698 actually went | **CHARACTERISED** (no change: the cause is a documented model choice in a module I do not own) | -- | -- | same | `-0.0251` at `4e8ea247` (the v5.35 inverse-characteristic evaluator; reproduced at HEAD to 6 digits by `inverse_map=False`) + `-0.0017` at `f602b72c` (WP-A1 raytrace) |
| **A24-5** the d6 oracle's own floor was never measured | **ADDED** | `tests/unit/test_niche_d6_exact_tilted_leg.py:139-154` (why-comment), new test file | `::test_the_inline_oracle_s_ee2_floor_is_far_under_the_restated_envelope`, `::test_the_oracle_pupil_patch_keeps_essentially_all_the_launched_power` | the oracle against itself | r_on floor **6.4e-04**; oracle FWHM is NOT invariant (3.15 -> 2.85 um) |

**Headline.**  The deliverable asked which readout -- pre-C1 or post-C1 -- is
closer to the truth.  **Neither: they are the same bits.**  `a18ab074^` and
`a18ab074` both read `r_on = 0.9697869227555169`, `r_off = 0.985518`, field
fidelity 0.998391, on the same fixture and the same oracle.  C1 changed the
PARAXIAL focus readout, and the d6 fixture runs `final_leg='exact'`, which
returns through `carrier_referenced_exact_focus_readout` without ever entering
`carrier_referenced_focus_readout` -- the only caller of
`_default_focus_standoff`.  So deliverable 2's first branch applies (post-C1 is
EQUAL to pre-C1): the bar is restated, the warning is corrected, the
fingerprints are re-recorded, and **no change was made to the C1 resolver or
its guard**.

---

## 2. Per finding

### A24-1 -- WP-A6/C1 did not move this number, and cannot

**What was claimed.**  `WP-A16_REPORT.md` 8.3 bisected the d6 failure to
`a18ab074` (WP-A6): `e3f7185a` "passed > 0.97", `a18ab074` "failed 0.9698",
bit-stable since.  It called `e3f7185a` the commit's "immediate parent".

**What is measured.**  `a18ab074`'s parent is `818251fd`, not `e3f7185a`:
`git log --oneline e3f7185a..a18ab074 -- lumenairy` lists **56 commits**, so
that bisect step sampled across 55 intervening library commits -- WP-A1
(raytrace), WP-A5 (propagators), WP-A7, WP-A8, WP-A10 to WP-A14 and their
VERIFY passes among them.  Extracting both sides of `a18ab074` read-only and
running the current test file against each:

```
rev                              r_on      chain EE2  oracle EE2  fidelity rho  TVD      peak-frac ratio
818251fd  (a18ab074^, PRE-C1)  0.969787   0.759860   0.783533     0.998391    0.02962      0.912764
a18ab074  (WP-A6, POST-C1)     0.969787   0.759860   0.783533     0.998391    0.02962      0.912764
65bea863  (HEAD)               0.969787   0.759860   0.783533     0.998391    0.02962      0.912764
```

Identical to every digit of every statistic, including the decentred arm
(`r_off = 0.985518` at both).  The harness reproduces WP-A16's own pytest
readings exactly (0.969787 / 0.985518), so it is measuring the same thing.
Re-measured again after the tree took `8f0fc547` (VERIFY-A16: `_lens_real.py`,
`lens_config.py`) and `9bc3c912`: **bit-identical on both arms**, so every
number this report and the corrected warning quote is current for the tree
being committed.

**Why it cannot.**  `propagate_traced_carrier_chain` with `final_leg='exact'`
builds the readout at `carrier.py:9075` (`carrier_referenced_exact_focus_readout`)
and returns at `:9093`.  `_default_focus_standoff` -- the function C1 changed,
and the only caller of C1's `_beam_containment_standoff` besides the
containment guard -- is reached only from `carrier_referenced_focus_readout`
(`:3008`), which that branch never calls.  Pinned by measurement rather than by
reading the call graph: poisoning both symbols to raise and running the d6
exact chain completes normally, while the same poison on `final_leg='paraxial'`
fires.  Both arms are in the new test file.

**Residual risk.**  None to the library (nothing changed).  The risk closed is
documentary: the next reader of WP-A16 8.3 would have started at C1.

### A24-2 -- the `r_on` bar was a per-build number, and is now a derived envelope

**What was wrong.**  `assert r_on > 0.97` was a one-sided threshold parked
0.0266 under a single 2026-07-29 reading of 0.9966, with no oracle error floor,
no statement of what size of defect it catches and no upper arm -- exactly the
`docs/TESTING_STANDARDS.md` S5 pattern the audit exists to close.  It went red
on a 2.2e-04 margin (0.02 %).

**The floor it never had.**  The inline oracle has exactly two free parameters.
Sweeping both (the chain field held fixed):

| n_pupil | patch half | pupil pitch | pupil pts | power kept | **r_on** | r_off | oracle EE2 | oracle FWHM |
|---|---|---|---|---|---|---|---|---|
| 141 | 2.2 w *(shipped)* | 0.03143 w | 19509 | 0.999981 | **0.969787** | 0.985518 | 0.783533 | 3.15 um |
| 211 | 2.2 w | 0.02095 w | 43721 | 0.999980 | 0.969810 | 0.985538 | 0.783514 | 3.15 um |
| 281 | 2.2 w | 0.01571 w | 77541 | 0.999980 | 0.969821 | 0.985552 | 0.783505 | 3.15 um |
| 167 | 2.6 w | 0.03133 w | 24409 | 1.000000 | 0.969203 | 0.985176 | 0.784005 | 2.85 um |
| 181 | 2.8333 w | 0.03148 w | 25445 | 1.000000 | 0.969181 | 0.985197 | 0.784023 | 2.85 um |
| 205 | 3.2 w | 0.03137 w | 25629 | 1.000000 | 0.969180 | 0.985207 | 0.784023 | 2.85 um |
| 256 | 4.0 w | 0.03137 w | 25608 | 1.000000 | 0.969181 | 0.985234 | 0.784023 | 2.85 um |
| 281 | 3.2 w | 0.02286 w | 48241 | 1.000000 | 0.969181 | 0.985183 | 0.784023 | 2.85 um |

**r_on spans 6.41e-04 and r_off 3.76e-04 over all of it.**  (`_PUPIL_HALF` is
not merely a sampling extent: the patch is SQUARE and centred on the BEAM, so
it is what apertures the oracle -- the prescription's circular stop, 2.8333 w,
only bites once the patch reaches it, which is why every row at or past 2.8333 w
agrees.  A memory-bounded copy of `_oracle_field` was used for the sweep and
checked BIT-IDENTICAL to the shipped one at the shipped settings.)

**And it is not a metric artefact.**  `_metrics` normalises EE by the total
power in the +/-7.2 um window, so a skirt-level change moves EE2 without the
core moving.  Re-scoring the same fields with a window-normalisation-free
statistic -- EE(2 um)/EE(6 um), core over halo -- tracks it to 1.4e-03:

```
arm                       r(EE2)      r(EE2/EE6)
on axis                   0.969787     0.970581
on axis, inverse_map off  0.996575     0.996889
decentred                 0.985518     0.986413
decentred, inverse_map off 0.971078    0.972486
```

**The restated bar.**  `0.95 < r_on < 1.02`, with the derivation in the test:

* the 0.0268 between 0.9698 and 0.9966 is a documented model choice (A24-4), 42x
  the oracle floor -- so both readings are real and any bar INSIDE that band
  measures the build;
* **below** -- 0.95 is a shortfall of 0.05 from 1: 1.66x the measured shortfall
  (0.0302), 78x the oracle floor, and 9.7x UNDER the mildest on-axis defect this
  fixture has ever shown (the superseded ABCD readout centre, 12.4 um off the
  Fermat focus: EE2 0.363 against the oracle's 0.703, ratio 0.516).  The
  paraxial route reads 0.217 (16x);
* **above** -- 1.02 is 0.010 over the largest value this metric takes anywhere on
  the stand-in's decentre curve (1.010 at 0.25 w).  The chain cannot hold 2 %
  more energy inside 2 um than a stigmatic oracle without the oracle having
  moved or power having appeared.

The docstring's premise ("the chain tracks the oracle on axis and slightly
worse when the same beam is decentred") is corrected to the measured ordering.
`r_off`'s bar is UNCHANGED -- it was already two-sided with a rationale in both
directions -- only its recorded value is updated (0.9828 on 2026-07-29, 0.9855
on 2026-09-12; still inside `0.965 < r_off < 1.005`).

**Residual risk.**  The lower arm is ~1 decade from the mildest measured defect,
not "decades"; the test says so in as many words rather than implying more.

### A24-3 -- the shipped warning quoted a calibration whose ordering had inverted

**What was wrong.**  `_check_decentred_fit` (`carrier.py`) tells USERS
"MEASURED on the K=-n^2 conic stand-in ... 0.00 w -> 0.997; 0.25 w -> 1.002;
0.50 w -> 1.005; 0.75 w -> 0.977; 1.00 w -> 0.983; 1.50 w -> 0.923" and the
same six numbers appear in its docstring.  Re-measured today on the same
stand-in, the same oracle and the same metric (six chain runs + six oracle
evaluations):

| decentre | EE2 ratio 2026-07-29 | **EE2 ratio 2026-09-12** | EE4 ratio | fidelity rho | FWHM ratio |
|---|---|---|---|---|---|
| 0.00 w | 0.997 | **0.969787** | 0.997696 | 0.998391 | 1.0000 |
| 0.25 w | 1.002 | **1.009643** | 1.000044 | 0.999831 | 0.9048 |
| 0.50 w | 1.005 | **1.008053** | 0.999966 | 0.999750 | 1.0000 |
| 0.75 w | 0.977 | **1.002022** | 0.999247 | 0.999604 | 1.0000 |
| 1.00 w | 0.983 | **0.985518** | 0.994936 | 0.999033 | 1.0000 |
| 1.50 w | 0.923 | **0.902811** | 0.953853 | 0.971442 | 1.0000 |

The FALL past one beam radius reproduces (and is worse at 1.5 w), but the
on-axis row is no longer the best of the six: it is now the second-WORST, so
the message asserted the opposite of the measured ordering to every user whose
fan trips the guard.

**What I changed.**  The message and the docstring now carry the 2026-09-12 six
points with their date, the oracle's own floor on the ratio (6.4e-04, so a
reader can see both tables are real measurements), an explicit "READ THE
ORDERING WITH CARE" clause, and the cause with its switch: below ~1 w the six
points differ by less than the readout MODEL CHOICE does, and with
`traced_kwargs={'inverse_map': False}` the same stand-in reads 0.997 at 0 w and
0.971 at 1.0 w -- the 2026-07-29 ordering.  The evaluator improves the WORSE of
the two arms, so the note says trade, not loss.  The guard's own verdict is
unchanged and still stated: per-order image metrics through a decentred
hand-off are a LOWER BOUND.

`docs/history/carrier.md`'s fingerprints were re-recorded in the same change
(`scripts/record_history_fingerprints.py lumenairy/propagators/carrier.py
--reason "..."`), because a warning string is code to both the AST and the token
fingerprint.  `test_audit2609_a17_history_relocation.py -k carrier`: 12 passed.

**The change is string-literals-only, proven rather than asserted.**  Parsing
the HEAD blob and the working tree, blanking every `ast.Constant` whose value is
a `str` (docstrings and message literals alike) and comparing the dumps: **equal**
-- while the raw ASTs differ, as they must.  So nothing the interpreter executes
moved, which is also why every one of the 165 WP-A6 / VERIFY-A6 pins is
untouched.

**Residual risk.**  The message is long (2616 characters at 1.0 w).  It was
already long; this change adds ~600 characters of ordering caveat.  The
alternative WP-A16 offered -- drop the numbers and point at the test -- would
have removed the one thing a user in the field can act on without the repo.

### A24-4 -- where the 0.997 -> 0.9698 actually went

Full read-only bisect (oldest first; `--` means identical to the row above):

| rev | subject | **r_on** | fidelity rho | 1 - rho | TVD | peak-frac ratio |
|---|---|---|---|---|---|---|
| `a4e8e855` | fix(pmm): four cross-build names *(pre-campaign base)* | **0.996575** | 0.999045 | 9.55e-04 | 0.00864 | 0.975800 |
| `6dfc79d6` | feat(traced): exact gap kernel, spline Newton default | *(unscorable -- see note)* | | | | |
| `4e8ea247` | **feat(lens): banded ray-density + inverse-characteristic evaluator** | **0.971526** | 0.998497 | 1.503e-03 | 0.02823 | 0.916438 |
| `e3f7185a` | docs(lens): the 5.44.0 follow-ups | 0.971526 | -- | -- | -- | -- |
| `0067d63b` | fix(elements,materials): WP-A8 | 0.971526 | -- | -- | -- | -- |
| `f602b72c` | **fix(raytrace): WP-A1** (exact conic intersection, entrance eikonal, ...) | **0.969787** | 0.998391 | 1.609e-03 | 0.02962 | 0.912764 |
| `a8e3091f` | verify(raytrace): VERIFY-A1 | 0.969787 | -- | -- | -- | -- |
| `6b801ffa` | fix(propagators): WP-A5 | 0.969787 | -- | -- | -- | -- |
| `818251fd` | verify(rcwa,eme,bor): VERIFY-A14 *(= `a18ab074^`)* | 0.969787 | -- | -- | -- | -- |
| `a18ab074` | **fix(carrier): WP-A6 (C1-C5)** | 0.969787 | -- | -- | -- | -- |
| `65bea863` | HEAD when measured | 0.969787 | -- | -- | -- | -- |
| `9bc3c912` | HEAD after `8f0fc547` (VERIFY-A16) landed -- re-measured | 0.969787 | -- | -- | -- | -- |

The oracle is constant across all of it: `get_glass_index('N-BK7', 1.31 um)`
reads `1.5035829054102239` at every revision measured, so the fixture's conic
(`R = -(n-1)f`, `K = -n^2`) and the oracle built on it are identical throughout.

**The big step is the inverse-characteristic evaluator, and it is provable at
HEAD.**  `apply_real_lens_traced` documents `inverse_map=False` as returning
v5.43.0's bits; the chain leaves the evaluator ON for the terminal fine retrace
only (the niche-C15 scoping at `carrier.py:9147`), which is exactly the leg the
d6 fixture reads.  Passing it through `traced_kwargs` at HEAD:

| arm | `inverse_map` | r(EE2) | r(EE1) | r(EE4) | fidelity rho | 1 - rho | TVD | peak-frac ratio | FWHM ratio |
|---|---|---|---|---|---|---|---|---|---|
| on axis | default (True) | **0.969787** | 0.930224 | 0.997696 | 0.998391 | 1.609e-03 | 0.02962 | 0.912764 | 1.0000 |
| on axis | False | **0.996575** | 0.982893 | 0.998668 | 0.999045 | 9.55e-04 | 0.00864 | 0.975800 | 1.0000 |
| decentred | default (True) | **0.985518** | 0.978521 | 0.994936 | 0.999033 | 9.67e-04 | 0.01872 | 0.975419 | 1.0000 |
| decentred | False | **0.971078** | 0.944947 | 0.993652 | 0.998105 | 1.895e-03 | 0.02685 | 0.933297 | **1.0952** |

`inverse_map=False` at HEAD reproduces `a4e8e855`'s 0.996575 to all seven
digits, so the whole -0.0251 is the evaluator and `6dfc79d6` contributed
nothing to it.

*(Note on `6dfc79d6`: it cannot be scored at all -- the d6 on-axis exact leg
returns an all-zero readout there, so `_metrics` divides by zero.  This is the
"different assertion; `r_on` not reached" row of WP-A16's table, now with its
mechanism.  It is a broken intermediate for this fixture, repaired by
`4e8ea247` two commits later; the `inverse_map` reproduction above is what
closes the gap it leaves in the bisect.)*

**Is it a regression?**  Against this oracle it is a TRADE, and the trade is in
the right direction:

* it moves the ON-AXIS arm away from the oracle (1 - rho 9.55e-04 -> 1.61e-03)
  and the DECENTRED arm towards it (1.895e-03 -> 9.67e-04);
* the WORST arm improves, 1.895e-03 -> 1.609e-03, and the two arms go from a
  2.0x spread to a 1.7x spread in the other direction;
* the decentred spot's WIDTH defect closes outright: FWHM ratio 1.0952 ->
  1.0000.  (The d6 docstring attributes that closure to niche D7; measured
  today, the arm that closes it at HEAD is the evaluator -- with it off, the
  1.0952 is back.)

So the "crossover" the finding describes is real, is the evaluator's, and is
not a defect.  **The `-0.0017` residual is `f602b72c` (WP-A1 raytrace)** --
2.7x the oracle floor, 1/16 of the evaluator's step, on a fixture whose
chain-vs-oracle residual is 1.6e-03 in fidelity; WP-A1's conic-intersection and
entrance-eikonal corrections are independently verified by VERIFY-A1 against
60-digit closed forms, so a small move on a conic singlet is expected and I make
no claim that it is wrong.  It is recorded because it is what took 0.971526
across the 0.97 line.

### A24-5 -- one latent fragility found while measuring the floor

The oracle's ring-binned FWHM is **not** invariant to its pupil patch: 3.15 um
at 2.2 w, 2.85 um at 2.8333 w and beyond (two bins of the 0.15 um readout
pitch, 9.5 %; the wider aperture gives the narrower core).  The d6 file asserts
`abs(m_off['fwhm']/o_off['fwhm'] - 1.0) < 0.05`, so anyone "improving"
`_PUPIL_HALF` to match the prescription's own stop would turn that bar red at a
ratio of 1.105 without any propagator changing.  No bar was moved -- the
constant and the bars live in the same file and are self-consistent today --
but `_PUPIL_HALF` now carries a why-comment saying what it apertures, what the
EE floor is (6.4e-04) and that the FWHM ratios are quoted against THIS value of
it.

---

## 3. Files touched

| file | change |
|---|---|
| `lumenairy/propagators/carrier.py` | `_check_decentred_fit`: docstring calibration block re-measured + an ordering note naming the evaluator and its switch; the shipped `RuntimeWarning` message likewise.  No executable change. |
| `docs/history/carrier.md` | fingerprints re-recorded (`ast` `3829ad9e...` -> `be8794e4...`, `token` `b3e9d7ba...` -> `ab51fed5...`) with the WP-A24 reason line. |
| `tests/unit/test_niche_d6_exact_tilted_leg.py` | `_PUPIL_HALF` why-comment; `test_decentred_carrier_decentre_penalty_envelope` docstring corrected; `r_on` restated as a derived two-sided envelope; `r_off`'s recorded values updated (bar unchanged); FWHM comment extended with the measured `inverse_map` arm. |
| `docs/audits/.../fixes/WP-A24_REPORT.md` | this file (new). |
| `docs/audits/.../fixes/WP-A24_CHANGELOG.md` | changelog text (new). |

**New test file:** `tests/unit/test_audit2609_a24_decentre_calibration.py` (7 tests).

Not touched, deliberately: the C1 resolver `_beam_containment_standoff`, the
guard `_check_focus_containment`, `_achievable_focus_margin`, and every lens
module.

## 4. Tests run

| command | result | duration |
|---|---|---|
| `python -m pytest tests/unit/test_audit2609_a24_decentre_calibration.py -q --no-header -p no:cacheprovider` | **7 passed** | 15.59 s / 15.90 s |
| `python -m pytest tests/unit/test_niche_d6_exact_tilted_leg.py -q --no-header -p no:cacheprovider` | **38 passed** | 153.28 s / 148.83 s |
| `python -m pytest tests/unit/test_audit2609_a17_history_relocation.py -q --no-header -p no:cacheprovider -k carrier` | **12 passed**, 685 deselected | 4.47 s / 3.96 s |
| `python -m pytest tests/unit/test_audit2609_a6_carrier.py tests/unit/test_audit2609_a6_verify_carrier.py -q --no-header -p no:cacheprovider` | **165 passed** | 46.35 s |
| `python -m ruff check` on the three touched python files | **All checks passed** | -- |

Where two durations are given: the first run, and a full re-run after the last
docstring edits.  The re-run is against a tree that had meanwhile taken
`8f0fc547` (VERIFY-A16, which touches `_lens_real.py` and `lens_config.py`)
and `9bc3c912`; the d6 ids are all still green there.

All with `OPENBLAS_NUM_THREADS=1`.  No pre-existing failures were found in any
of these files.  All 38 d6 ids pass; the one that was red before this change is
`test_decentred_carrier_decentre_penalty_envelope`, whose `assert r_on > 0.97`
reads 0.969787 -- measured independently here at HEAD and at four earlier
revisions, and matching WP-A16's pytest reading to six digits.  Its duration is
unchanged (36.76 s, the heaviest id in the file: the restatement adds no chain
runs).  The file is `pytest.mark.slow` (WP-A21 moved it there at 124.7 s on the
committed `.test_durations`); it reads 153 s here with three other agents'
jobs on the box.

**Fail-before, confirmed:**

* `::test_the_shipped_decentre_warning_quotes_the_re_measured_calibration`
  asserts the superseded `0.00 w -> 0.997` / `1.00 w -> 0.983` strings are
  ABSENT, which is precisely what the pre-fix message contains -- it fails on
  the pre-fix module by construction.
* `::test_the_exact_final_leg_never_enters_the_paraxial_focus_standoff_resolver`
  has its own falsifier as the next test: the same poison on
  `final_leg='paraxial'` DOES fire, so the pass is evidence about routing, not
  about an ineffective patch.
* the restated `r_on` envelope is two-sided, so it fails in both directions;
  the pre-restatement bar `> 0.97` fails TODAY at 0.969787, which is the
  failure this WP was opened on.

## 5. Requested changes outside my ownership

1. **`WP-A16_REPORT.md` section 8.3 / section 5 item 7 (WP-A16's owner or the
   orchestrator) -- the bisect's conclusion is wrong and should be corrected in
   place.**  It reads "The crossing is `a18ab074` (WP-A6) ... `e3f7185a` -- its
   immediate parent -- passes".  `a18ab074`'s immediate parent is `818251fd`,
   and `818251fd` reads 0.969787.  Suggested replacement sentence: *"The
   crossing is in two pieces, neither of them WP-A6's: -0.0251 at `4e8ea247`
   (the inverse-characteristic evaluator) and -0.0017 at `f602b72c` (WP-A1
   raytrace).  `a18ab074^` already reads 0.969787."*  The method was sound; the
   sample was 56 commits wide at that step.

2. **The lens work package that owns `_lens_imap.py` / `_lens_traced.py`
   (VERIFY-A16 is in those files now) -- one measurement they may want.**  The
   niche-C15 scoping comment at `carrier.py:9152` says the per-leg decomposition
   "measures exactly that change ... without being able to say which arm is
   right".  On the d6 stand-in there IS an arbiter, and it says: on axis the
   evaluator costs 6.5e-04 of field fidelity (1 - rho 9.55e-04 -> 1.609e-03) and
   0.0268 of EE(2 um) ratio, while decentred it BUYS 9.3e-04 of fidelity and
   closes a 9.5 % FWHM defect.  Worst-arm fidelity improves.  That is one
   fixture at NA 0.20 and is not a reason to change a default; it is the first
   two-sided, oracle-refereed reading of the evaluator's skirt trade that I can
   find, and it belongs with that module's evidence
   (`docs/audits/BUILD_INVERSE_MAP_2026_08_11.md`).  I did not touch either
   file.

3. **Nothing else.**  No change is requested to `pyproject.toml`, the CHANGELOG
   or any lens module.

## 6. Deferred

1. **The `-0.0017` at `f602b72c`** is characterised (2.7x the oracle floor) but
   not decomposed to a single WP-A1 sub-fix.  Doing so means archiving the
   ~7 sub-changes of R1-R7 individually, which is ~1 h of measurement for a move
   that is 1/16 of the evaluator step and 1/6 of the restated envelope's lower
   clearance.  Recorded, not chased.

2. **A second independent oracle for this fixture** (`validation/oracles/debye_oracle_v3.py`,
   the skew-ray + Debye path design 121 uses) would let the evaluator trade be
   scored on a second geometry.  Effort ~half a day: the Debye oracle needs the
   conic's exit-sphere apodisation supplied explicitly.  Not needed for this
   WP's verdict, which only required an oracle stable to 6.4e-04 -- measured,
   this one is.

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A24_CHANGELOG.md`
