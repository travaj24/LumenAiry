# Independent verification of ROUND 2 of the BOR/EME guards — 5.45.1, 2026-09-12

> **STATUS — INDEPENDENT VERIFICATION.** Of
> `docs/audits/FIX_BOR_GUARDS_ROUND2_2026_09_12.md` on branch
> `fix/bor-guards-round2` (`f4206556`, `90a1f607`, `2135ba7e`, `08b93732`,
> `3d76abf4`), which claims to close the four blocking defects D1 / D2 / D13 /
> D9 that `docs/audits/VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md` left open,
> plus D4 and six restatements.
>
> **Every number below was re-measured on fixtures this verification wrote.**
> Nothing is read out of the round-2 report and repeated: where a number of
> theirs appears it is labelled as theirs and is either reproduced from their
> own JSON (§4.5) or contrasted with a number of mine on a different battery.
>
> Tree: `C:\tmp\lum_vbor2`, branch `verify/bor-guards-round2`, forked from
> `fix/bor-guards-round2` (`3d76abf4`).  `lumenairy/` was not edited.
> Pre-round tree: `C:\tmp\lum_vbor2_pre`, detached at `f2d331c5` (the round-1
> verification's tip, which is round 2's own fork point).
> Probes and per-arm JSON: `validation/probe_verify_bor_round2/`.
> New gates: `tests/unit/test_verify_bor_guards_round2.py`.
> Binding: `docs/TESTING_STANDARDS.md`.

---

## 1. Terms

**BOR** — body of revolution: an axisymmetric stack, layers along `z`, each a
radial permittivity profile on `[0, Rbig]`, closed by a PEC wall, solved one
azimuthal order `m` at a time.  `rbl` below is the cell radius in **vacuum
wavelengths** (`Rbig k0 / 2 pi`), the axis along which the legacy nodal basis's
spurious sea grows.

**The two bases compared throughout.** `basis='nodal'` is the historical
non-divergence-conforming FD basis, the only one the passivity screen touches.
`basis='staggered'` is its div-conforming Yee twin on the identical geometry —
used here as the reference for the channel SET and, separately, as a *measured*
budget for how much super-unity flux normalisation can legitimately produce.

**Set-wrong** — a nodal row whose channel COUNT differs from its staggered
twin's on the same geometry.  A definition of damage that reads no energy, so
an energy bar can be scored against it without assuming its conclusion.

**Arm** — a build (Windows py3.14.6 / numpy 2.4.4 / scipy 1.17.1, or WSL
py3.12.3 / numpy 2.4.6 / scipy 1.17.1) × a `OPENBLAS_CORETYPE` × a thread
count.  Every command carries `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` explicitly; the **loaded** kernel is read back from
`threadpoolctl` and recorded in every JSON, never inferred from the request.
`KATMAI` loads Katmai directly on this host (the round-2 report reached it via
`PRESCOTT`; both alias to the same kernel here).

---

## 2. Verdict table

| # | claim | verdict | the number that decides it, RE-MEASURED |
|---|---|---|---|
| D1 | `_stack_is_provably_passive` = `Im eps >= 0` + lossless incidence; loss ladder 4/13 → 13/13; healthy 0/39; gain 1/5 → 0/5 | **CONFIRMED** on the layers it covers, **BOUNDED** on the incidence half-space | my damaging ring ladder: **4/13 → 13/13**, returned violation constant at `max(R+T) = 2.41297` on every rung; healthy uniform **0/39 before and after, 0 warnings**; gain **2/5 → 0/5** (2 not 1 because my ladder has a `-1e-14` rung). **But a loss on `layers[0]` alone still disarms BOTH detectors: 4/13 → 4/13, with the div-conforming twin measuring the legitimate budget at 8.75e-07 against a returned 2.41297** — see GAP 2 |
| D2 (two-sided) | two-sided where lossless, one-sided where absorbing; deficit rows refused | **CONFIRMED** | my lossless battery: **16/18 → 18/18**, the 2 new refusals are pure-deficit rows (`max(R+T) = 0.971423` / `0.971541`, `min` 0.90) whose message names `It is a DEFICIT`; absorbing-exit ladder stays one-sided and **13/13** refused; healthy absorbing stacks reach deficits of 0.131 and are never judged |
| D2 (no scalar bar) | energy populations overlap; the 1e-3 bar justified only on the sub-population it alone detects | **CONFIRMED** | my 216-solve census: set-right reaches **6.599e+00**, set-wrong starts at **2.742e-07** → **7.38 decades of overlap**.  On the sub-population the bar owns: uniform ≤ **3.408e-07**, next row up **9.721e-02** → **5.46 decades**, bar 3.47 above / 1.99 below |
| D2 (index ceiling) | `Re qn > Re sqrt(eps_ceiling)` + 5e-10 slack; 40/44; 0 false positives on 116 | **CONFIRMED**, and stronger on my battery | staggered worst **-1.070e-05**, nodal set-right worst **-1.999e-03**, set-wrong fires on **66 of 66**; **0 false positives on 150 undamaged rows**; mildest caught **+7.1087e-06 = 4.15 decades above the slack** (their number to two digits, on a different battery); the ceiling is the ONLY detector on **6** rows, the mildest of which closes to `6.582e-07` — under the warn edge |
| D2 (`5e-10` provenance) | the slack is the twins' own, not new | **CONFIRMED** | `bor_stack.py:697`, `:910`, `_jax_bor.py:197`, `_jax_sem.py:392` all carry `> -5e-10` |
| D13 | `cut_band` floor `max(max\|z\|, \|k0\|)`, `k0` threaded from every site; unit-invariant | **CONFIRMED** unconditionally | three of my own cells × {um, nm, m} × **10 POST arms**: **0 differing sorted roots, orientation census identical, worst `dT00` 9.33e-15**.  On the pre-round tree, **8 arms**, same cells: `dT00` = **4.188e-08** / **1.674e-08** to four digits on every kernel and both builds, the `mode_match` orientation census differs (6 negated + 1 conj → 4 + 3), and a weak-gain cell moves **80 of 80** strip roots |
| D9 | family envelope 1.2716e-06 over `m` ∈ {0,1,2} and 10 arms; new bar 1e-5 carries 7.86x | **RESTATED — the margin is a sample property of a DIFFERENT, un-swept axis** | on the **shipped** fixture, varying only the radial-cutoff index that `_gamma_of(m, idx=2)` fixes by an undocumented default: idx=2 gives 1.97e-07 / 3.63e-08 / 5.53e-08 / 6.91e-09 for m = 0/1/2/3, while **idx=1 at m=1 gives 1.787437e-04 — 17.9x ABOVE the 1e-5 bar**.  8 of 20 `(m, idx)` combinations exceed 1e-5.  The CHANNEL-COUNT claim (the durable one) holds everywhere I swept — see GAP 5 |
| D4 (attribution) | `warn_own` reads `w_min_own_frac`; the liner's neighbour is no longer blamed | **CONFIRMED** | liner + unstructured neighbour, 11 rungs: PRE `warn_own` on layers **[0, 1]**, two messages; POST **[0]**, one message |
| D4 (domain ends) | the two domain ends count for every layer; the three positions behave identically over a width ladder | **CONFIRMED** for the attribution, **REFUTED as a ladder claim** | `w_min_own_frac` equals the requested width at all three positions and all 11 widths, so the ends do count.  But over the ladder the three positions disagree at **two** rungs, identically on both trees: at `1e-8` of `Rbig` the AXIS liner reads `ok` with `q_excess = inf`, and at `1e-6` it reads `ok` while outer/middle warn — see GAPS 3 and 4.  The report's literal claim (all warn at 1e-7, all ok at 1e-4) is true |
| restated: signal side 0.38 dec | sample-scoped | **CONFIRMED** as a restatement | reproduced from the round-2 probe's own JSON; my own band work did not widen it further |
| restated: `k0` floor binds 0/135 | unit-safety floor, unreachable | **CONFIRMED** | their own JSON reports `binds 0`, `layers 135`, closest ratio **2.0568**; independently re-measured on my own 648-layer sweep (3 unit systems x 3 `k0` x 6 `Rbig/lambda` x 2 `N` x 3 `m` x 2 bases) — see §8.5 |
| restated: SEM warn edge 1.003x at `elements_per_segment=32` | edge documented, not moved | **CONFIRMED**, and the gate is knife-edge | the shipped gate asserts `1.0 <= margin < 1.5`; the measured margin is 1.003x, so its pass boundary is **0.3 % away**.  It held on every arm I ran — see §6 |
| restated: Q bar vacuous on ordinary geometry | one-sided | **CONFIRMED** | `verdict` refuses only on the conjunction; ordinary geometry reads `w_min_union_frac = inf` and can never reach it |
| bit identity | 58/58 BOR + 15/15 EME, nothing moved | **CONFIRMED for their battery, RESTATED for mine** | my 46 BOR + 19 EME fixtures, both builds: **58 of 65 identical**, 6 moved and every one of those is a row the new screen REFUSES, and **1 EME fixture legitimately moved** — `strip_nmunits`, a sub-`k0` spectrum, which is exactly D13's population and which their battery does not contain (their §7 says so) |
| runs | 389/0/0 on 10 arms | **CONFIRMED so far on 6 of 10**, remainder in §7 | 389 passed, 0 failed, 0 errors on every arm that finished |
| `.test_durations` | 12,880 entries, sorted, JSON-valid, 101 keys over the five touched files, nothing over the 60 s cap, the stale renamed key gone | **CONFIRMED, every clause** | 12,880 entries and `list(d) == sorted(list(d))`; **101** keys across the five files (6 / 64 / 11 / 7 / 13); slowest **51.61 s** (86 % of the cap); `test_structured_stack_energy_floor_nodal` absent |
| ruff | clean | **CONFIRMED** | `All checks passed!` over `lumenairy/ tests/ validation/probe_verify_bor_round2/` |

**Five gaps, none of them a regression introduced by round 2, are in §5.**

---

## 3. D1 and D2 — the passivity screen, re-measured

### 3.1 The battery

`validation/probe_verify_bor_round2/_vfix.py` + `v1_census.py`: **216 solves** —
2 bases × 3 profile families (uniform slab, 4-period ring grating, 4-zone
segment stack) × `m` = 0/1/2 × `N` = 120/200 × `Rbig/lambda` = 0.5 / 1 / 2 / 4 /
8 / 16, each geometry solved TWICE (nodal and its staggered twin) with
`BOR_NODAL_PASSIVITY_GUARD = False`, then re-solved armed for the decision.
`v3_ladders.py` adds six loss/gain ladders (A–F, 116 more solves).

None of it is imported from `validation/probe_fix_bor_round2`.

### 3.2 D1 — the loss ladder

`v3_ladders.py`, ladder A: a damaging ring stack (`m = 1`, `rbl = 2`, `N = 200`,
`k0 = 2`, `eps` 2↔6 rings between `eps = 2` half-spaces), relative loss on the
ring's high region.

| `Im/Re` | PRE | POST | nodal `max(R+T)`, guard off | staggered twin `R+T` |
|---|---|---|---|---|
| 0 … 1e-12 | REFUSED | REFUSED | 2.41297 | 1.000000000 |
| **3e-12** | **returned** | **REFUSED** | 2.41297 | 1.000000000 |
| 1e-11 … 1e-6 | returned | REFUSED | 2.41297 | 1.000000000 … 0.99999997 |
| 1e-4 … 1e-1 | returned | REFUSED | 2.41246 … 2.18483 | 0.99998 … 0.966 |

**4 of 13 → 13 of 13**, with the returned violation *unchanged* across the whole
ladder — the round-2 report's central observation reproduced on a different
fixture.  Ladder B (healthy uniform stacks, `m` = 0/1/2 × the same 13 rungs):
**0 of 39 refused and 0 warned, before and after**, with deficits tracking the
loss exactly (`0.999999` at 1e-6 to `0.869` at 1e-1).  Ladder C (gain):
**2 of 5 → 0 of 5**; the moved rungs are `-1e-14` and `-1e-12`, both of which the
old `|Im eps|` test admitted.  At `Im/Re = -1e-3` the nodal cascade returns
`R + T = 8.78e+22` in silence, which is the deliberate consequence of gain
having no theorem in either direction.

### 3.3 D2 — the second side

Ladder F (18 provably lossless segment stacks): **16 of 18 → 18 of 18**.  The two
that moved are **pure-deficit** rows — `max(R + T) = 0.971423` and `0.971541`
with `min(R + T)` at 0.90 — so the pre-round screen's `max(R+T) - 1 = -0.029`
was below its warn edge and it returned them silently.  Their messages read
`... a violation of 0.09721 against the bar 1e-03.  It is a DEFICIT: ...`, and
they say `PROVABLY PASSIVE LOSSLESS`.  Across the 216-solve census, **11**
refusals name a DEFICIT.

Armed on an absorbing stack, the deficit half stays disarmed: ladder B's
legitimate deficits reach 0.131 and are never judged.

### 3.4 The populations

`v5_analyse.py` on the 216-solve census (Windows / Haswell / 1 thread):

| population | n | `\|R+T-1\|` | `Re qn - n_max` |
|---|---|---|---|
| STAGGERED, every row | 108 | 8.882e-16 .. **2.107e-12** | ≤ **-1.0703e-05**, fires on 0 |
| NODAL, channel set RIGHT | 42 | 1.332e-15 .. **6.599e+00** | ≤ **-1.9993e-03**, fires on 0 |
| NODAL, channel set WRONG | 66 | **2.742e-07** .. 2.937e+02 | **fires on 66 of 66** |

* **The energy populations overlap by 7.38 decades** (their battery: 9.97).  The
  round's core honesty claim — no scalar energy bar separates accurate from
  damaged — is confirmed as a population statement on an independent battery.
* **The index ceiling has zero false positives on 150 undamaged rows**
  (108 staggered + 42 nodal-with-the-right-set), against the ≥ 60 this
  verification was asked for.  The closest undamaged row sits **4.33 decades
  below zero** and the mildest caught set-wrong row **4.15 decades above the
  slack** — the two populations are on opposite sides of zero, as claimed.
* **The ceiling is the only detector on 6 rows**, the mildest of which
  (`uniform m=1 N=200 rbl=2`) closes to `|R+T-1| = 6.58e-07` — *below the 1e-6
  warn edge* — while returning 12 channels where the div-conforming twin
  returns 10.  Those 6 rows are the operative value of the conjunct, measured
  rather than argued.
* **The sub-population the energy bar owns** (right set, silent ceiling):
  uniform ≤ **3.4085e-07**, next row up **9.7209e-02**, **5.46 decades with
  nothing in it**; the 1e-3 bar sits 3.47 above and 1.99 below.  The **warn edge
  1e-6 sits 0.47 decades above the accurate ceiling** — round 2's restatement
  from 3.71 to 0.57 decades is confirmed, and my battery makes it thinner still.
* **The union misses 0 of 66 set-wrong rows** on my battery, and the staggered
  basis is never refused (0 of 108).

### 3.5 The 4 set-wrong rows the ceiling still misses — characterised

My battery does not contain that population (I swept `m` ≤ 2; theirs sweeps
`m` ≤ 3), so I re-derived it from their own census JSON
(`validation/probe_fix_bor_round2/r1_passivity_census_win_Haswell_t1.json`).
Every number in their §4 reproduces exactly: 44 set-wrong of 80, ceiling fires
on 40, mildest caught `+5.026594e-06`, staggered worst `-3.027306e-05`,
nodal-right worst `-1.413702e-03`, **0 false positives of 116**, the union
missing 2 of 44, and the bar's sub-population `2.714107e-07` / `2.173049e-02`
= 4.90 decades.

**All four missed rows are the same geometry**: `m = 3`, `rbl = 2`, `N` = 120
and 200, uniform and ring.  The nodal basis returns **9 channels where both the
staggered twin and the closed-form Bessel count return 8**, and the
oracle's **basin is 9.69e-04** — a nearly-degenerate pair of radial roots.  The
mechanism is therefore specific, not mysterious: the non-div-conforming basis
*splits one transverse eigenvalue into two*, and both halves are physically
admissible (`gamma^2 > 0`, `Re qn` sitting **2.13e-02 BELOW** the medium's index
ceiling), so the contradiction the ceiling keys on is simply absent.  Two of the
four also carry no measurable power (`|R+T-1|` = 1.8e-06 and 2.7e-07), which is
why the union misses exactly those two.

**Consequence for scope:** the only population both detectors miss lives at an
azimuthal order the in-test census (`m` ∈ {0,1,2}) does not reach.  That is
correctly recorded in the round-2 report's §9 and is not a defect, but it means
the guard's blind spot is not exercised by any gate in the suite.

### 3.6 The lossy half-space, and the switch

* **Absorbing EXIT half-space** (ladder E): the screen stays armed one-sided —
  **4/13 → 13/13**.  The exit-side ceiling conjunct is correctly gated off by
  `_layer_is_lossless`, and the rows where the energy falls *below* unity
  (`max(R+T) = 0.923` at `Im/Re >= 1e-4`) are caught by the **incidence-side**
  ceiling.  Nothing wrong is returned silently.
* **Absorbing INCIDENCE half-space** (ladder D): both detectors disarm together
  — see GAP 2.
* **The switch.** With `BOR_NODAL_PASSIVITY_GUARD = False`, the SHA-256 of the
  exact IEEE-754 bytes of `R` and `T` is **identical on all 116 ladder rungs
  between the pre-round tree and this one**, and so are the channel counts.  The
  round-2 changes to `build_layer` (`min_rel_im_eps`, and `max_rel_im_eps`'s
  denominator moving to `max|eps|`) are read by the screen and by nothing else.

---

## 4. D13 — the EME branch band

`v2_eme_units.py`.  Three cells of my own — an ordinary 1 um / 1310 nm strip
pair, a long-period low-contrast cell, and a weak-gain cell — each written in
micrometres, nanometres and metres, with lengths × `s` and wavenumbers / `s`.
`ky * s` is the unit-free observable; spectra are sorted before differencing.
Two branch sites are censused separately: `_ky_forward` on the strip spectrum
and `forward_decaying_root` inside `mode_match` on the `qz` spectrum.

All three cells have `max|ky|` **above 1 in micrometres and below 1 in
nanometres** (127.77 → 0.128, 47.86 → 0.0479, 133.25 → 0.133), which is exactly
the population the literal floor decided differently.

| tree | arms | strip roots differing (nm vs um) | `mode_match` `qz` roots differing | orientation census identical | worst `dT00` |
|---|---|---|---|---|---|
| **PRE** (`f2d331c5`) | 8 (4 kernels × 2 builds) | **80 of 80** on the weak-gain cell | **7 of 7** on the other two | **no** (6 negated + 1 conj → 4 + 3) | **4.188e-08** |
| **POST** | 10 (4 kernels × 2 builds, + Haswell t4 on both) | **0** | **0** | **yes, on every arm** | **9.33e-15** |

The PRE `dT00` reads 4.188e-08 (cell A) and 1.674e-08 (cell B) **to four digits
on Haswell, Nehalem, Katmai and Sandybridge and on both builds** — the defect is
a threshold decision, not arithmetic, exactly as claimed.  The metre arm agrees
with the micrometre arm to 1e-15 on both trees, which isolates the cause to the
floor rather than to round-off.

Mode counts and orientation censuses are identical across units, kernels,
threads and builds on the POST tree.  **D13 CONFIRMED.**

---

## 5. Bit identity, on a battery that contains D13's population

`v7_identity.py` + `v8_identity_diff.py`: **46 BOR fixtures and 19 EME
fixtures**, each hashed to the SHA-256 of the exact IEEE-754 bytes of its
answer, a fixture that raises recorded as its exception class.  The BOR half
deliberately includes **22 LEGACY NODAL solves**, which the round-2 report's own
battery does not — its 58 fixtures are `BORStack` (fd / sem) only, where the
passivity screen is structurally unreachable, so its "nothing moved" is a
weaker statement than it looks.

| build / loaded kernel | identical | moved | of which newly REFUSED | other moves |
|---|---|---|---|---|
| Windows py3.14 / Haswell / 1 | **58 of 65** (BOR 40/46, EME 18/19) | 7 | **6** | 1 |
| WSL py3.12 / Haswell / 1 | **58 of 65** (BOR 40/46, EME 18/19) | 7 | **6** | 1 |

The six BOR moves are all `HASH -> BORNodalPassivityError`: three lossless nodal
rows the two-sided screen and the ceiling now refuse, and three lossy ring rows
(`Im/Re` = 3e-12, 1e-6, 1e-2) that D1's predicate now reaches.  **No BOR answer
changed value.**

**The one other move is D13's own population and is the fix working.**
`EME:strip_nmunits` — a 64-point strip written in nanometres — has
`max|ky| = 0.127879` against `k0 = 4.0537e-03`.  The literal 1.0 floor gives a
band of **1.0e-09**; the `k0` floor gives **1.2788e-10**, 7.8x narrower, and
**3 of the 64 modes** fall between the two, so their branch (and therefore the
sign or conjugation of their `ky`) changes.  That is exactly the sub-`k0`
spectrum the round-2 report's §7 says its battery does not contain.  Its
bit-identity claim is honest and its battery is simply not probative for the
EME half of the change; mine is, and the answer moves in the direction D13
requires.

---

## 6. Defects found

### GAP 1 — `test_the_screen_disarms_on_an_absorbing_incidence_medium` is vacuous — P2 (test)

That gate builds its superstrate at `eps = 2 + 2e-3j`, i.e. `Im/Re = 1e-3`.
`_orient.channel_core` drops any mode whose `|Im qn|` exceeds
`_BOR_CHANNEL_IMAG_BAR` (5e-5), so at that loss **the channel set is EMPTY**,
`R + T` is empty, and `_check_nodal_passivity` returns at its `e.size == 0`
line **without ever calling the predicate**.

**Reproducer (mutation test).** Replace `_stack_is_provably_passive` with a copy
whose incidence-lossless conjunct is deleted, then run the shipped gate:

```
RESULT: the shipped gate STILL PASSES with the conjunct deleted -> VACUOUS
```

Measured channel counts on that fixture: `Im(eps_sup)` = 2e-3 → **0 channels**;
2e-5 … 2e-11 → 7–8 channels.  The gate is two decades of loss away from being
live.

**Fixed here** by `test_the_absorbing_incidence_gate_needs_a_non_empty_channel_set`,
which runs the same geometry at `Im/Re = 1e-7`, asserts the premise (the channel
set survives), asserts the predicate answers `False`, asserts the guard is
silent, and asserts (premise-gated) that the lossless twin of the same geometry
IS refused.  Deleting the conjunct flips it.

### GAP 2 — D1's hole is relocated, not removed, on the incidence half-space — P1

`_stack_is_provably_passive` checks `Im eps >= 0` on every layer **and**
`_layer_is_lossless(layers[0])`.  The second conjunct reintroduces, for
`layers[0]` alone, precisely the discontinuity at `Im(eps) = 0+` that D1 was
filed about — and because the index ceiling is evaluated inside
`_check_nodal_passivity` *after* the predicate's early return, a loss there
disarms **both** detectors.

Ladder D, measured:

| `Im/Re` on `layers[0]` | nodal `max(R+T)` | **staggered twin** `R+T` | PRE | POST |
|---|---|---|---|---|
| 0 … 1e-12 | 2.41297 | 1.000000000 | REFUSED | REFUSED |
| **3e-12** | **2.41297** | **1.000000000** | returned | **returned** |
| 1e-9 | 2.41297 | 1.000000001 | returned | returned |
| **1e-6** | **2.41297** | **1.000000875** | returned | **returned** |
| ≥ 1e-4 | — | — | 0 channels | 0 channels |

**4 of 13 → 4 of 13: nothing moved.**  The justification given for the conjunct
— that in an absorbing incidence medium `R + T` is not a power fraction — is
correct in kind but the *size* of that effect is measurable and is measured
here: the div-conforming twin on the identical geometry returns
**8.75e-07** of super-unity at `Im/Re` = 1e-6 and **exactly 1** at 3e-12, while
the nodal cascade returns **2.41297** on every rung.  At `Im/Re` = 1e-6 that is
**6.2 decades** of excess the flux bookkeeping cannot account for, and more at
every rung below it — at 3e-12 the twin closes to 1 exactly, so the whole 1.413
is unexplained.  The caller is told nothing.

**Pinned** by `test_a_negligible_loss_on_the_incidence_medium_must_not_disarm_the_screen`
(`xfail(strict=True)`), which derives the budget from the twin at runtime rather
than pinning a number: a nodal excess three decades above the twin's own must
not be returned in silence.

**Suggested remedy** (not implemented here): keep the screen armed when
`layers[0]` is merely *passive*, widening the super-unity bar by the twin's
measured budget — or, cheaper and fully deterministic, keep the **index
ceiling** armed there, since its Rayleigh argument concerns the half-space's own
`eps` and is untouched by whether the *incidence* medium's flux is conserved.

### GAP 3 — a NON-FINITE `q_excess` is the SEM contract's most benign reading, and the verdict therefore moves with the BLAS kernel — P2 (inherited)

`_sem_contract.verdict` computes `hot = np.isfinite(excess) and excess >
_BOR_Q_EXCESS`.  A spectrum that has actually blown up gives `inf` (or `nan`),
which makes `hot` **False**, so the contract is silent exactly where the damage
is worst.

**Reproducer**: a caller-prescribed liner `1e-8` of `Rbig` wide **at the axis**
(`BORStack(Rbig=24, m=1, N=120, basis='sem', degree=8)`,
`add_layer(0.5, segments=[(2.4e-7, 6.0), (24.0, 2.0)])`):
`w_min_own_frac = 1.0e-08`, `q_excess = inf`, **`verdict = 'ok'`, no message**.
The same liner at `3e-8` — three times wider — warns.  Identical on the
pre-round tree, so it is inherited, not introduced; but it sits in the code path
round 2 rewrote, and the round-2 gate's two rungs (1e-7 and 1e-4) both miss it.

**And because `inf` is a backward-error outcome, the VERDICT moves with the
BLAS kernel.**  The identical geometry, both builds:

| loaded kernel | `q_excess` | verdict |
|---|---|---|
| Haswell | `inf` | **`ok`** |
| Nehalem | `inf` | **`ok`** |
| Katmai | `inf` | **`ok`** |
| **Sandybridge** | **8.89487e+07** | **`warn_own`** |

Windows py3.14 and WSL py3.12 agree row for row.  A mesh-contract decision that
reads `ok` on three kernels and `warn_own` on a fourth, for a geometry the
caller wrote once, is the shape `docs/TESTING_STANDARDS.md` exists to forbid —
and the fix is one word: treat a non-finite excess as hot, not as cold.  The
consequence is bounded (a missing `UserWarning`, never a wrong returned number),
which is why this is P2 and not P1.

Pinned by `test_a_non_finite_q_excess_is_not_a_benign_sem_verdict`
(`xfail(strict=True)`, premise-gated on the arm actually producing `inf`).  On
Sandybridge the premise is genuinely absent and the gate SKIPS with the reading
(`q_excess is finite (8.895e+07)`) rather than reporting a defect that is not
there — which is the premise gate doing exactly its job, measured.

### GAP 4 — the `warn_own` edge is decided by the representation of the width — P3 (inherited)

`verdict`'s own arm is `fa < _BOR_MIN_ELEM_FRAC` — a **strict** comparison
against a quantity the geometry reproduces only to round-off.  At exactly the
edge:

| position | walls | `w_min_own_frac` | verdict |
|---|---|---|---|
| outer | `Rbig - w`, `Rbig` | 9.999999999917482e-07 | **`warn_own`** |
| middle | `6.0`, `6.0 + w` | 9.999999999917482e-07 | **`warn_own`** |
| **axis** | `0`, `w` | **1.0e-06 exactly** | **`ok`** |

The same physical liner warns at one domain end and not at the other.  Pinned by
`test_the_warn_own_edge_is_not_decided_by_the_representation_of_the_width`
(`xfail(strict=True)`), with the passing half of the three-position claim kept
in `test_the_liner_verdict_agrees_at_all_three_positions_away_from_the_edge`
(0.3× and 3× the edge, derived from the constant rather than pinned).

### GAP 5 — the near-cutoff bar's margin is a property of an un-swept axis — P2 (test)

`tests/unit/test_fix_bor_multilayer_guards._gamma_of(m, idx=2)` defaults the
**radial cutoff index** to 2, every caller takes the default, the docstring does
not say which order it picks or why, and the round-2 restatement — which
correctly widened the family over `m` and over ten arms — does not mention it.

Measured on the **shipped** fixture (`_RBIG = 24`, `_NFD = 120`, the same
floor-derived 13-rung ladder), varying only `idx`
(`validation/probe_verify_bor_round2/v11_cutoff_index.py`):

| `m` | idx=0 | **idx=1** | **idx=2 (the gate)** | idx=3 | idx=4 |
|---|---|---|---|---|---|
| 0 | 1.168e-05 | 5.549e-05 | **1.9655e-07** | 7.092e-05 | 3.209e-08 |
| 1 | 1.353e-05 | **1.787437e-04** | **3.6267e-08** | 2.761e-06 | 2.874e-07 |
| 2 | 2.192e-05 | 2.069e-05 | **5.5270e-08** | 1.896e-05 | 1.505e-08 |
| 3 | 2.837e-07 | 8.508e-06 | **6.9146e-09** | 9.330e-06 | 5.610e-09 |

**8 of 20 combinations exceed the new `_CUTOFF_LADDER_BAR = 1e-5`, and the worst
— at `m = 1`, inside the gate's own family — is 17.87x above it and 179x above
the 1e-6 the round replaced.**  The envelope over the gate's own index is
**1.965486e-07**; over the full `(m, idx)` grid it is **1.787437e-04**.
Reproduced on a second, non-coincident geometry and on a finer grid
(`v4_cutoff.py`: `6.691e-05` and `3.743e-05` at the same `(m, idx)` region).
The index the gate happens to use is the one at which the closure is three
decades better than its neighbours.

This is not a library regression — the channel COUNT, which is what the
orientation band exists to protect, is one number on every `(m, idx)` I swept
(19 of 20) except the deepest cutoff `m=0, idx=0`, where the ladder's own order
falls under `_BOR_CHANNEL_REAL_FLOOR` and the count legitimately reads
`[0, 1]` — the channel gate working, which is the same effect
`_CUTOFF_LADDER_FLOOR_MULT` exists to keep the gate away from.  It is the round's own diagnosis (§6.2, §6.4: "a margin
measured on one fixture family is a SAMPLE property") not applied to D9.

Bounded by `test_the_near_cutoff_bar_is_scoped_to_one_radial_cutoff_index`,
which asserts the integer invariant at both indices unconditionally and bounds
the un-swept index by `_CUT_FAMILY_BAR = 2e-3` (11.2x above the measured
envelope).

### GAP 6 — the report attributes a number to a probe that did not produce it — P3 (documentation)

§6.2 of the round-2 report reads *"Re-measured, `r8_band_sides.py`: MIN `sigma`
= 2.3820e-05 at `Im(n)` = 1e-3 and **2.3820e-08** at 1e-6, i.e. **3.38 and 0.38
decades**."*  That probe's own JSON
(`validation/probe_fix_bor_round2/r8_band_sides_win_Haswell_t1.json`,
`summary.signal`) reports minima of **3.7752e-05** and **3.7752e-08**, i.e.
3.577 and 0.577 decades.  2.382e-08 is the ROUND-1 verification's number
(`VERIFY_BOR_MULTILAYER_GUARDS_2026_09_12.md`, its own table).

The conclusion is unaffected — the minimum over a union of populations is the
smaller of the two, so 0.38 decades is the right figure to carry — and the
shipped code comment
(`_orient.orient_band_scale`) attributes both numbers correctly, naming
3.7752e-08 as this probe's and 2.3820e-08 as the verification's.  Only the
report's prose conflates them.  Recorded because it is precisely the
right-conclusion-wrong-numbers shape `docs/TESTING_STANDARDS.md` names as the
most dangerous: it reads as authoritative and the artefact that matters is
right, so nothing catches it except re-reading the JSON.

---

## 7. Durability audit of the round-2 constants and bars

| where | constant / bar | origin stated? | two-sided? | scope | verdict |
|---|---|---|---|---|---|
| `bor_solve._BOR_PASSIVE_DEADBAND` | 16 ULP of `max\|eps\|` | yes — the 1-D peer's `_PASSIVE_ANTIHERM_DEADBAND` | n/a (a deadband) | family | **sound** |
| `bor_solve._BOR_LOSSLESS_REL_IM` | 1e-12 | yes, with the absorbed-power measurement (2e-12 at that ratio) | 9 decades below the deficit bar | family | **sound** |
| `bor_solve._BOR_INDEX_CEILING_SLACK` | 5e-10 | yes — verified present at four twin sites | 4.15 dec above / 4.33 below, re-measured | family | **sound** |
| `bor_solve._BOR_NODAL_SUPERUNITY_BAR` | 1e-3 | yes, on the sub-population it owns | 3.47 / 1.99 decades on my battery | sub-population, stated | **sound** |
| `bor_solve._BOR_NODAL_SUPERUNITY_WARN` | 1e-6 | yes, restated to 0.57 dec | **0.47 dec on my battery** — thin, but it can only warn | sub-population, stated | **sound, at the margin recorded** |
| `test_fix_bor_multilayer._CUTOFF_LADDER_BAR` | 1e-5 | yes, but over `m` only | 7.86x over the swept index; **-17.9x over the un-swept one** | **narrower than claimed** | **GAP 5** |
| `test_fix_bor_multilayer._CUTOFF_LADDER_FLOOR_MULT` | 10.0 | yes — derived from `_BOR_CHANNEL_REAL_FLOOR`, 3.3x inside the measured onset | one-sided by construction | family | **sound** |
| `test_fix_bor_guards_round2` `bar / 30.0`, `30.0 *` | 30 | **no stated origin** (≈1.5 dec against a measured 4.90) | conservative | — | **P3: a numeric constant in a test without a stated origin** |
| `test_fix_bor_guards_round2` `100.0 * slack` | 100 | **no stated origin** (2 dec against a measured 4.15) | conservative | — | **P3, same shape** |
| `test_fix_bor_guards_round2` `1.0 <= margin < 1.5` | the measured 1.003x | yes | **0.3 % from the pass boundary** | one census family | **S4-shaped, but the quantity is mesh arithmetic and it held on every arm run** |
| `test_fix_bor_guards_round2` `12.0 < ratio < 20.0` | the `1/k^2` law (16x) | yes | ±25 % about the law | family | **sound** |
| `test_fix_eme_branch_cut` `abs=1e-12` on `T00` | — | partly | **3 dec above the POST residual, 4.6 below the PRE defect** | family | **sound** |
| `test_fix_eme_branch_cut._WINDOW` | narrowed scan window | yes, with the decisiveness check | n/a | fixture | **sound** |
| `_sem_contract._BOR_Q_EXCESS`, `_BOR_MIN_ELEM_FRAC`, `_BOR_SLIVER_BAND_FRAC` | unchanged; comments restated | yes | one-sided, stated as such | family | **sound** — the restatements are honest |

Premise gating: every pathology-asserting gate in
`test_fix_bor_guards_round2.py` and `test_verify_bor_multilayer_guards.py` that
I read carries a `pytest.skip` on the premise and an unconditional assertion of
the guard, as the round-2 report claims.  One inconsistency, cosmetic: the
in-test census filters the ceiling on `ceiling_excess > 0.0` where production
uses `> _BOR_INDEX_CEILING_SLACK`; the test is the stricter of the two.

---

## 8. Runs

### 8.1 The 20-file BOR/EME set, four kernels × 1 thread + Haswell × 4, both builds

Every arm records `lumenairy.__file__` and the LOADED kernel before pytest runs.
Scripts: `validation/probe_verify_bor_round2/runs/arm_win.sh`, `arm_wsl.sh`.

| # | build | requested | **loaded** | thr | passed | failed | errors | time |
|---|---|---|---|---|---|---|---|---|
| 1 | Windows py3.14 | HASWELL | Haswell | 1 | **389** | 0 | 0 | 1841.95 s |
| 2 | Windows py3.14 | NEHALEM | Nehalem | 1 | **389** | 0 | 0 | 2362.60 s |
| 3 | Windows py3.14 | KATMAI | Katmai | 1 | **389** | 0 | 0 | 2534.09 s |
| 4 | Windows py3.14 | SANDYBRIDGE | Sandybridge | 1 | **389** | 0 | 0 | 2284.20 s |
| 5 | Windows py3.14 | HASWELL | Haswell | 4 | **389** | 0 | 0 | 1762.68 s |
| 6 | WSL py3.12 | HASWELL | Haswell | 1 | **389** | 0 | 0 | 1708.21 s |
| 7 | WSL py3.12 | NEHALEM | Nehalem | 1 | **389** | 0 | 0 | 2246.25 s |
| 8 | WSL py3.12 | KATMAI | Katmai | 1 | **389** | 0 | 0 | 2486.08 s |
| 9 | WSL py3.12 | SANDYBRIDGE | Sandybridge | 1 | **389** | 0 | 0 | 2169.43 s |
| 10 | WSL py3.12 | HASWELL | Haswell | 4 | **389** | 0 | 0 | 1533.19 s |

**Zero failures and zero errors on every one of the ten**, every tail carrying a
real summary line and no arm reporting `no tests ran`.  The count is 389, which
matches the round-2 report's own; this verification's new file is a separate
run (§8.2) so it does not change it.

### 8.2 This verification's own gates

`tests/unit/test_verify_bor_guards_round2.py`, run on the same four kernels x
1 thread plus Haswell x 4 threads on both builds (`runs/newfile_matrix.sh`,
logs in `runs/new_*.log`):

| build | loaded kernel | thr | result | time |
|---|---|---|---|---|
| Windows | Haswell | 1 | 3 passed, 3 xfailed | 34.78 s |
| Windows | Nehalem | 1 | 3 passed, 3 xfailed | 52.56 s |
| Windows | Katmai | 1 | 3 passed, 3 xfailed | 55.14 s |
| Windows | **Sandybridge** | 1 | **3 passed, 1 skipped, 2 xfailed** | 48.69 s |
| Windows | Haswell | 4 | 3 passed, 3 xfailed | 28.85 s |
| WSL | Haswell | 1 | 3 passed, 3 xfailed | 32.08 s |
| WSL | Nehalem | 1 | 3 passed, 3 xfailed | 48.73 s |
| WSL | Katmai | 1 | 3 passed, 3 xfailed | 52.75 s |
| WSL | **Sandybridge** | 1 | **3 passed, 1 skipped, 2 xfailed** | 44.49 s |
| WSL | Haswell | 4 | 3 passed, 3 xfailed | 24.10 s |

Zero failures and zero strict-xfail escapes on any of the ten, and every arm
under the 60 s shard cap (worst 55.14 s, 92 %).  The one skip is GAP 3's premise
gate on Sandybridge, described above.  Its near-cutoff gate runs
two ladders, so its rung list is the library-derived one DECIMATED to every
second rung -- which keeps both ends, reproduces the 1.787437e-04 envelope
exactly, and holds the file at 31.5 s on Haswell against the 60 s shard cap.

### 8.3 ruff

```
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vbor2 && ~/lumvenv/bin/ruff check \
                 lumenairy/ tests/ validation/probe_verify_bor_round2/'
-> All checks passed!
```

### 8.4 The D1/D2 ladders across builds

The decision ladders are a threshold property, not an arithmetic one, and
reproduce identically on both builds:

| tree | build | loaded kernel | thr | A (loss, damaging) | C (gain) | D (absorbing **incidence**) | E (absorbing exit) |
|---|---|---|---|---|---|---|---|
| POST `3d76abf4` | Windows | Haswell | 1 | **13/13** | **0/5** | 4/13 | **13/13** |
| POST `3d76abf4` | Windows | Haswell | 4 | **13/13** | **0/5** | 4/13 | **13/13** |
| POST `3d76abf4` | Windows | Nehalem | 1 | **13/13** | **0/5** | 4/13 | **13/13** |
| POST `3d76abf4` | Windows | Katmai | 1 | **13/13** | **0/5** | 4/13 | **13/13** |
| POST `3d76abf4` | Windows | Sandybridge | 1 | **13/13** | **0/5** | 4/13 | **13/13** |
| POST `3d76abf4` | WSL | Haswell | 1 | **13/13** | **0/5** | 4/13 | **13/13** |
| POST `3d76abf4` | WSL | Katmai | 1 | **13/13** | **0/5** | 4/13 | **13/13** |
| PRE `f2d331c5` | Windows | Haswell | 1 | 4/13 | 2/5 | 4/13 | 4/13 |
| PRE `f2d331c5` | WSL | Haswell | 1 | 4/13 | 2/5 | 4/13 | 4/13 |

Seven POST arms over four loaded kernels, two thread counts and two builds:
**identical to the rung on every one of them**, which is what "the nodal
blow-up is a deterministic discretisation defect, not an arithmetic one" means
measured rather than asserted.  Column D is the one that did not move: GAP 2.

### 8.5 The BOR band's `k0` floor, re-measured on 648 layers

`v9_k0_floor` sweeps `max|q| / |k0|` per layer over 3 unit systems (um / nm / m,
six orders of magnitude apart) x 3 `k0` x 6 `Rbig/lambda` (0.5 .. 16) x 2 `N` x
3 `m` x 2 bases:

* **binds on 0 of 648 layers**;
* closest approach **`max|q| / k0` = 2.4424** (metres, `rbl` = 16, `m` = 1,
  staggered) — a factor of two ABOVE the floor, never below it.

Their own probe's JSON reports `binds 0` over 135 layers with a closest ratio of
2.0568.  The restatement — a unit-safety floor that cannot be exercised through
`BORStack`, kept because the alternative is a dimensioned literal — is confirmed
on five times the population.

### 8.6 `.test_durations`

Spliced by `validation/probe_verify_bor_round2/v10_durations.py`, which measures
with `--durations=0 -vv` (so pytest prints the sub-5 ms entries rather than
hiding them), replaces every key belonging to the new file, re-sorts and
re-validates:

| | |
|---|---|
| entries before / after | 12,880 / **12,886** |
| keys for the new file | **6**, one per collected test, none defaulted |
| sorted, JSON-valid after the splice | **yes** (asserted in the probe) |
| slowest entry | `test_the_near_cutoff_bar_is_scoped_to_one_radial_cutoff_index`, **7.05 s** |
| entries over the 60 s shard cap | **none**; the whole file is 29.69 s |

---

## 9. Ship recommendation for 5.45.1

**SHIP the round-2 library change.**  D1's predicate, D2's second side and the
index-ceiling conjunct, D13's `k0` floor and D4's attribution all do what the
report says they do, re-measured on independent fixtures across four OpenBLAS
kernels, two thread counts and two builds; nothing moved that the new screen
does not refuse (plus the one EME fixture that is D13's whole point); and every
population statement I could re-derive from their own data reproduced exactly.

**Two conditions.**

1. **GAP 2 (P1) should be decided before the tag, not after.**  It is not a
   regression — the pre-round tree behaves identically — but it is the same
   defect D1 was filed for, still live on one layer, and the round's own
   framing ("a 3e-12 loss walks through") applies to it verbatim.  The cheap
   fix is to leave the **index ceiling** armed when only the incidence medium
   absorbs: its Rayleigh bound is about the half-space's own `eps` and does not
   depend on the incidence medium's flux being conserved.  If that is deferred,
   the deferral belongs in the CHANGELOG and in
   `_stack_is_provably_passive`'s docstring, which currently presents the
   incidence conjunct as settled.
2. **GAP 1 (P2) should be fixed with the tag**, because the gate that exists to
   protect the very conjunct GAP 2 is about is currently incapable of failing.
   The replacement in this file is one function.

GAPS 3, 4 and 5 are pre-existing or test-scope and can ride to the next round;
all three carry strict-xfail or bounded gates here, so they cannot be lost.

---

## 10. What I could not verify

* **The CI's actual kernels.**  `ZEN` aliases Haswell and `SKYLAKEX` dies on
  this host (Ryzen 9 5950X, Zen 3); the EPYC 9V74 / 7763 pool is not
  reproducible here.  Every gate I added asserts an invariant unconditionally
  and premise-gates every pathology claim.
* **The `4 of 44` missed population on my own fixtures.**  My census stops at
  `m = 2`; the population exists only at `m = 3`.  I characterised it from the
  round-2 probe's own JSON (§3.5) rather than re-measuring it, which is weaker
  evidence than the rest of this document.
* **Whether the index ceiling would hold on a LOSSY half-space.**  Unchanged
  from the round-2 report: it is gated off there and I did not construct a case
  that would decide it.  GAP 2's suggested remedy assumes it would; that
  assumption is untested.
* **`.test_durations` on a quiet box.**  My timings were taken with up to ten
  pytest arms running, so they are upper bounds — the safe direction for a
  shard balancer, but not a clean measurement.
* **The 720-solve healthy-family sweep** the round-2 report ships but did not
  run (`r5_healthy_ceiling.py`).  I did not run it either; my 216-solve census
  puts the accurate-family ceiling at 3.41e-07 against their 2.71e-07, which
  moves the warn edge's margin the wrong way but does not settle it.
* **PML, the anisotropic Class-C populations, and an external accuracy oracle
  for the SEM answer.**  Unchanged from both prior documents.
