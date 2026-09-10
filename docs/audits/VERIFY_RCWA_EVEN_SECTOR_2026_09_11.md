# Independent verification of the RCWA modal branch-cut fix

Subject: `docs/audits/FIX_RCWA_EVEN_SECTOR_WSL_2026_09_11.md` and the change it
records, `lumenairy/elements/rcwa/_core.py::_sqrt_decay` (commit `6709b7d`,
merged at `75a0c81`), together with its decision file
`tests/unit/test_fix_rcwa_even_sector_wsl.py`.

Every number below was RE-MEASURED on fixtures built for this verification, on
both builds, against both trees.  Nothing was accepted by reading.  The probes
and their JSON are in `validation/probe_verify_rcwa_even_sector/`; the gates
this verification adds are in `tests/unit/test_verify_rcwa_even_sector.py`.

Trees: POST = `75a0c81` (worktree `C:/tmp/lum_vrcwa`, branch
`verify/rcwa-even-sector`); PRE = `48c8747` (detached worktree
`C:/tmp/lum_vrcwa_pre`), the fix's branch point.

Builds: **WIN** = Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1,
scipy-openblas dispatching **Haswell**.  **WSL** = Ubuntu on the same box,
python 3.12.3, numpy 2.4.6, scipy 1.17.1, scipy-openblas dispatching
**SkylakeX**.  Every run pins `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` on the command line except where a thread count is the
variable under study.

---

## 1. Terms

**Modal decay constant.**  RCWA condenses one grating layer's transverse-field
system to `M = P Q` and eigendecomposes it.  Each eigenvalue is `lam^2`; the
modal decay constant is `lam = sqrt(lam^2)`, and the layer's forward amplitudes
propagate through thickness `L` as `X = exp(-lam k0 L)`.  Keeping
`Re(lam) >= 0` is what makes `|X| <= 1`, i.e. what keeps the propagator a
contraction rather than a growing exponential.

**On the branch cut.**  For a PROPAGATING mode of a LOSSLESS layer `lam^2` is
exactly real NEGATIVE, which is exactly the principal square root's branch cut.
The two roots there are `+i|kz|` (OUTGOING -- the branch the homogeneous
half-space modes are built on) and `-i|kz|` (INCOMING).  They are separated
only by the sign of `Im(lam^2)`, and for a value that came out of `eig` that
sign is the eigensolver's backward error, not physics.

**The band.**  The fix replaces an exact `Re(sqrt(lam^2)) == 0` test with a
relative one:

```
scale  = max(max|r|, 1)
on_cut = |Re(r)| <= _CUT_BAND_REL * scale        _CUT_BAND_REL = 1e-8
r      = conj(r)  where  on_cut and Im(r) < 0
```

Throughout this document the **band ratio** of a mode means the quantity that
test thresholds, `|Re(r)| / max(max|r|, 1)`, and the **acted-on population**
means the modes with `Im(r) < 0` whose band ratio is at or below
`_CUT_BAND_REL` -- the only modes the function ever changes.

**Interface mode-match.**  Joining media `a -> b` forms `a = Wb^-1 Wa`,
`b = Vb^-1 Va` and inverts `a + b` EXPLICITLY, because `S12 = 2 (a+b)^-1`.
`a + b` is singular exactly when a FORWARD mode of `a` reproduces a BACKWARD
mode of `b`.

**Coincidence.**  Two media whose permittivities are exactly equal.  The
library's warning text, and the audit built on it, describe the failure class as
"a LAYER permittivity EXACTLY EQUAL to a REGION's".  Section 6 below shows the
partner need not be a region.

**Closure defect.**  `sum R + sum T - 1` (one polarization) or `- 2` (a Jones
return, two polarizations).  A provably lossless cell conserves energy EXACTLY
under the Laurent rule at any truncation, so this is an independent oracle whose
error floor is the arithmetic -- it needs no reference solve and no prior
reading.  Every accuracy statement in this document is made with it.

---

## 2. Task 0 -- the merged-tree gate

`tests/unit/test_fix_rcwa_even_sector_wsl.py`, `test_v5_14_2_backlog_batch.py`,
`test_rcwa.py`, `test_niche_audit_m4_m5_m6_rcwa.py`,
`test_niche_audit_w3_rcwa_pmm.py`, `test_niche_audit_w7_rcwa.py`, all with
`-p no:randomly`:

| run | tail |
|---|---|
| WIN, `OMP/OPENBLAS/MKL_NUM_THREADS=1` | `234 passed, 1 skipped, 5 warnings in 217.75s` |
| WSL, `OMP/OPENBLAS/MKL_NUM_THREADS=1` | `235 passed, 6 warnings in 217.75s` |
| WIN, threads UNPINNED | `234 passed, 1 skipped, 5 warnings in 600.08s` |

The one Windows skip is the `threadpoolctl`-dependent
`test_set_blas_threads_numerically_equivalent`; 234 + 1 matches the WSL 235.
**No failures on either build at either threading.**

The new file alone, Windows: 9 passed at 1 / 4 / 8 / 16 threads and unpinned
(2.65 / 2.27 / 2.40 / 4.01 / 41.78 s).  WSL: included in the 235 above.

---

## 3. Task 1 -- the thread-count dependence, re-measured

`v1_threads.py` lifts the failing comparison (`rcwa_jones_2d` on the
anisotropic block-in-2.25 cell, `n_substrate = 1.5`, `n_orders = 5x5`,
`symmetry=False` vs `True`), runs it THREE times in one process at each
setting, and records `max|R_full - R_even|`, both closure defects, and a
NON-coincident control at `n_substrate = 1.6`.

**Determinism.**  All three repeats returned IDENTICAL DIGITS for every
recorded quantity at every one of the 8 thread settings, on both trees and both
builds -- 32 runs, `all_deterministic = True` in all 32 JSON files.  "Flaky" is
the wrong word: each (build, tree, thread count) computes one number.

**`max|R_full - R_even|`, bar `1e-8`:**

| `OPENBLAS_NUM_THREADS` | WIN PRE | WIN POST | WSL PRE | WSL POST |
|---|---|---|---|---|
| 1 | **2.812111e-10** | 2.776e-16 | **1.625072e-02 (FAIL)** | 4.163e-17 |
| 2 | 3.053856e-12 | 2.776e-17 | 3.652183e-12 | 1.527e-16 |
| 3 | 2.172568e-13 | 1.318e-16 | 3.863174e-12 | 2.151e-16 |
| 4 | 7.727846e-14 | 1.110e-16 | 1.952188e-13 | 9.021e-17 |
| 6 | 3.248096e-14 | 1.457e-16 | 1.590110e-12 | 1.041e-16 |
| 8 | 1.833488e-11 | 1.804e-16 | 8.366551e-12 | 4.163e-17 |
| 16 | 5.512951e-14 | 1.874e-16 | 1.749399e-12 | 1.249e-16 |
| unpinned | 2.006728e-14 | 2.776e-17 | 7.370181e-12 | 1.180e-16 |

Every digit the audit quotes is reproduced exactly, including the two headline
readings (`2.812111e-10` and `1.625072e-02`) and the four-digit entries at 2, 4,
8, 16 and unpinned threads on both builds.  The PRE spread is 1.6e-02 down to
3.2e-14 -- **twelve decades on one quantity, with the `1e-8` bar inside it**.
The POST spread is 2.78e-17 .. 2.78e-16, i.e. **the thread dependence is gone**
(a 4.5-decade band, all of it below the bar by more than four decades).

The closure defect moves with it and CHANGES SIGN pre-fix (WIN full path:
-3.1995e-03, -1.0027e-03, -1.0768e-06, +1.2637e-03, +4.9393e-06 at 1/2/4/8/16
threads), and post-fix reads `|defect| <= 6.7e-15` at every setting on both
builds, with the even path returning EXACTLY 0.0 at several.

**The off-coincidence control** (`n_substrate = 1.6`) reads 5.4e-16 .. 7.1e-15
PRE and 9.7e-17 .. 3.1e-16 POST at every setting -- so the twelve-decade spread
belongs to the coincidence, not to the fixture family.

---

## 4. Task 1 (continued) -- the audit's headline numbers, on its own fixture

`v9_headline.py`, one thread, both arms in one process.  The interface
S-matrix builder is spied at all four module bindings, so both the `2N` and the
even-sector paths are seen.

| quantity | PRE (measured) | audit says | POST (measured) | audit says |
|---|---|---|---|---|
| `cond(a+b)` superstrate -> layer, full | **1.541e+04** | 1.541e+04 | 7.679e+03 | -- |
| ... the same on WSL | **1.049e+04** | -- | 7.196e+03 | -- |
| `cond(a+b)` layer -> substrate, full | **1.973e+15** | 1.973e+15 | **5.633e+03** | 5.633e+03 |
| `cond(a+b)` layer -> substrate, even | **1.074e+15** | 1.074e+15 | 3.928e+03 | 3.928e+03 |
| singular values below `1e-10 s_max`, full | **4** | 4 of 242 | **0** | 0 |
| singular values below `1e-10 s_max`, even | **2** | 2 of 122 | **0** | 0 |
| null-vector inverse participation, full | **2.27** | 2.27 | 3.31 | 3.31 |
| null-vector inverse participation, even | **1.11** | 1.11 | 2.00 | -- |

And on WSL at one thread, against the audit's own WSL table: `cond(a+b)`
layer -> substrate **1.761e+15 -> 5.286e+03** (full, audit 1.761e+15 ->
5.286e+03) and **2.252e+15 -> 4.521e+03** (even, audit 2.25e+15 -> 4.521e+03);
tiny singular values **3 -> 0** (full) and **1 -> 0** (even), audit 3 -> 0 and
1 -> 0; null-vector inverse participation **1.01 -> 3.96** (full), audit 1.01
-> 3.96.  Every WSL digit reproduces too.

The offending eigenvalues, read out of the solve's own `eig`:

```
lam^2 = -2.249999999999987 -2.9111e-15j  ->  lam = +9.7037e-16 -1.499999999999996j
lam^2 = -2.430648695704599 -2.7396e-16j  ->  lam = +8.7861e-17 -1.559053782171930j
lam^2 = -0.829190880912066 -1.6045e-18j  ->  lam = +8.8100e-19 -0.910599187849443j   (even arm)
```

-- identical to the audit's `r8_branch_cut.py` transcript to every printed
digit, including the exponents of the backward error.  The substrate's own
modes are `+1.5000j` and `+0.9000j`, so these three are the substrate's
BACKWARD modes wearing a forward label.  **CONFIRMED.**

At four threads the same probe reads `cond(a+b) = 9.061e+14` (full) /
`5.210e+15` (even) pre-fix and 5.781e+03 / 3.509e+03 post -- the digits above
`1e15` are the SVD's own floor on an exactly singular matrix and carry no
information, exactly as the audit says; the decade drop and the
singular-value COUNT going to zero are what is meaningful.

---

## 5. Task 2 -- the mechanism, on fixtures built here

`v2_mechanism.py` builds twelve fixtures and records, per solve, the condition
number of every interface mode-match, the number of singular values below
`1e-10 s_max`, the on-cut mode census, and the closure defect.  Both arms are
the two TREES (not a monkeypatch), one thread.

### 5.1 Coincident-mode fixtures

| fixture | `cond(a+b)` PRE -> POST, WIN | WSL | closure PRE -> POST, WIN | WSL |
|---|---|---|---|---|
| `c1` aniso block, twist 0.4, bg 2.25, `n_sub` 1.5 | 5.435e+14 -> 2.136e+02 | 7.207e+14 -> 2.739e+02 | -3.119e-05 -> -1.554e-15 | -3.005e-04 -> +4.441e-16 |
| `c2` aniso, `no` 1.7, bg 2.89 = `n_sub^2`, OBLIQUE 0.25 | 4.032e+15 -> 2.740e+03 | 1.830e+15 -> 2.691e+03 | -1.128e-03 -> -1.554e-15 | -1.106e-03 -> +8.882e-16 |
| `c3` aniso, bg 1.96 = **n_SUPERstrate^2** | 2.252e+15 -> 1.947e+02 | 2.252e+15 -> 1.741e+02 | +4.441e-16 -> +8.882e-16 | +5.773e-15 -> 0.0 |
| `c4` aniso, CONICAL theta 0.2 phi 0.7 | 4.718e+15 -> 1.767e+02 | 1.025e+15 -> 1.767e+02 | -1.172e-04 -> +1.332e-15 | -7.696e-04 -> +1.465e-14 |
| `c5` **SCALAR 2-D**, weakly modulated (1e-6) at 2.25 | 2.504e+08 -> 9.008e+01 | 7.260e+07 -> 6.372e+01 | -9.99e-16 -> -1.22e-15 | +1.33e-15 -> 0.0 |
| `c7` 1-D TM, 2 %-duty ridge in an `n_groove = n_sub` host | 2.280e+02 -> 1.437e+00 | 2.280e+02 -> 1.437e+00 | -9.215e-15 -> -9.437e-15 | -9.215e-15 -> -9.437e-15 |

Three things this adds to the audit's own account.

* **The coincidence works on the SUPERSTRATE side too** (`c3`).  The audit's
  probes only ever exercised a substrate coincidence.  `cond(a+b)` there is
  2.252e+15 pre-fix on both builds and 1.7-1.9e+02 post; the closure happens to
  be clean pre-fix at this thread count, which is precisely the "green by
  threading luck" shape the audit itself warns about.
* **The SCALAR 2-D path is not exempt** (`c5`).  The audit concludes that
  "a coincident permittivity alone -- in 1-D, or in the scalar 2-D path -- is
  not automatically one of those places", on the strength of one scalar fixture
  (an `eps = 4` block in an `eps = 2.25` background, which I reproduce at
  `cond = 2.024e+02` pre-fix).  A WEAKLY MODULATED scalar cell at the same
  coincidence -- where the layer really is a perturbation of the substrate --
  takes `cond(a+b)` to 2.5e+08 (WIN) / 7.3e+07 (WSL) pre-fix and back to
  ~1e+02 post.  The conclusion should be about MODE coincidence, which the
  audit states correctly elsewhere; the scalar/1-D exemption as written is too
  strong.
* **1-D is repaired even where its closure never moved.**  The audit's 1-D
  counter-case reads 8.2e-15 before and after, which I reproduce
  (`n1` 1.066e-14 -> 5.329e-15 TE; `n2` 4.663e-15 -> -1.998e-15 TM).  But the
  1-D TM path carries **2 mis-rooted on-cut modes** pre-fix and `cond(a+b)`
  drops 2.280e+02 -> 1.437e+00.  The mis-rooting is real there; only its
  consequence is nil.

### 5.2 Coincident permittivity, no coincident mode

`n1`/`n2` (1-D binary grating, `n_groove = n_sub = 1.5`, ridge 2.1, duty 0.5,
TE and TM), `n3` (scalar 2-D `eps = 4` block in 2.25, `n_sub` 1.5), `n5` (1-D
whose RIDGE index equals the substrate's) all close to 2e-15 .. 1.1e-14 on BOTH
arms and BOTH builds, with `cond(a+b)` between 1.5 and 2.0e+02 pre-fix.
**CONFIRMED**: an equal permittivity is necessary, not sufficient.

One fixture I had classified as non-coincident turned out to be one, and the
way it did is worth recording: `n4` (the anisotropic block filling 90 % of the
cell, so only a sliver of `eps = 2.25` background remains) reads
`cond = 5.301e+03`, closure -2.220e-16 pre-fix on **Windows** and
`cond = 2.252e+15`, closure **-8.652e-03** pre-fix on **WSL**.  Same geometry,
same source, two builds, and only one of them is broken -- which is the whole
defect in one row.  Post-fix both read ~4.5e+03 and ~2e-15.

---

## 6. A coincidence partner the audit does not name: a UNIFORM LAYER

A uniform layer of an `RCWAStack` is built by `_homogeneous_eigenmodes` in
EXACT arithmetic, exactly as a half-space region is.  So a uniform `eps = 2.25`
spacer sitting next to a STRUCTURED layer whose background is 2.25 is the same
degeneracy with no region involved -- and the substrate index is then free.

`v10_stack_spacer.py`: a three-layer stack (uniform 2.25 / anisotropic
block-in-2.25 / uniform 2.25), `n_orders = 4`, both arms in one process.

WINDOWS:

| `n_substrate` | `symmetry` | 1 thread PRE | 4 threads PRE | 8 threads PRE | POST (all) |
|---|---|---|---|---|---|
| 1.5 (region coincides) | auto | REFUSED, `sum R+T = 26.0` | 2.000000000 | REFUSED | 2.000000000000001 |
| 1.5 | True | REFUSED | 2.000000000 | REFUSED | 2.000000000000001 |
| 1.5 | False | WARNED, 2.002831752 | REFUSED | REFUSED | 1.999999999999999 |
| **1.63 (NOTHING coincides with a region)** | auto | **REFUSED** | 2.000000000 | **REFUSED** | 2.000000000000001 |
| **1.63** | True | **REFUSED** | 2.000000000 | **REFUSED** | 2.000000000000001 |
| **1.63** | False | **WARNED, 1.866464833** | **REFUSED** | **REFUSED** | 1.999999999999999 |
| 1.63, spacer moved to 2.56 (control) | auto/True/False | SILENT, 2.000000000 | SILENT | SILENT | 1.999999999999996 |

WSL, one thread, tells the same story with different digits -- which is the
point: `n_sub = 1.5` WARNS at `sum R+T` = 2.008411056 / 2.008411056 /
2.038841058 for auto / True / False, `n_sub = 1.63` is REFUSED for auto and
True and WARNS at **1.591401508** (a 20 % energy error) for False, and the
2.56-spacer control is SILENT at 2.000000000 on every row.  POST reads
1.999999999999999 .. 2.000000000000000 on all nine.

The control row is the discriminator: moving the SPACER off the structured
layer's background -- leaving the substrate exactly where it was -- makes the
pre-fix arm clean at every thread count.  So it is the LAYER-LAYER coincidence
that breaks it, and `n_substrate` is irrelevant.

This is a scope EXTENSION in the fix's favour (it repairs more than claimed)
and a correction to the library's own remedy text, which tells the user to
detune a permittivity against `n_substrate^2` or `n_superstrate^2` and does not
mention that another LAYER can be the partner.

It is also the soundest fail-before I found: at `n_orders = 3` this stack is
REFUSED at `sum R + T = 2.600e+01` on BOTH builds at 1, 4 and 8 threads, for all
three `symmetry` settings -- no thread-count dependence at all.  It is gate 2 of
`tests/unit/test_verify_rcwa_even_sector.py`.

---

## 7. Task 2 (continued) -- the two-sided band

`v3_band.py` censuses 75 fixtures per build: lossless anisotropic 2-D over
twist x truncation x substrate; oblique and conical; scalar 2-D at three
contrasts; 1-D TE and TM at three duties; high-contrast gratings at
`n_orders` 21 and 31; a LOSS LADDER `Im(eps)` 1e-2 .. 1e-14 on both the
anisotropic 2-D and the 1-D paths; four metals down to `eps = -100 + 5j`;
near-Wood mounts on the substrate and superstrate Rayleigh anomalies; four
near-DEGENERATE square cells; and six mounts driven onto a LAYER CUTOFF by
trisection.  The band ratio is computed from the RAW `eig` output, so it is a
property of the eigenproblem and identical on both arms (verified: the PRE and
POST censuses agree row for row).

| | audit (51 fixtures) | this box (75 fixtures) |
|---|---|---|
| `Im(r) < 0` modes | 2150 WIN / 2135 WSL | 2082 WIN / 2054 WSL |
| below the band: count | 117 / 99 | 161 / 153 |
| below the band: **max ratio** | **2.245e-16 / 1.511e-16** | **7.9357e-11 / 7.9357e-11** |
| above the band: **min ratio** | **8.267e-02 / 7.947e-02** | **1.2612e-02 / 1.2612e-02** |
| lossy fixtures contributing below the band | none (ladder to 1e-8) | one, at `Im(eps) = 1e-14`, ratio 1.056e-17 |

So the two populations DO separate, and the bar sits between them on every
fixture measured -- but the margins are not the ones the constant's docstring
states.  Measured over this box the bar is **2.1 decades above the noise side**
(not 7.6) and **6.1 decades below the signal side** (not 6.9).

The noise-side envelope is set entirely by mounts near a LAYER CUTOFF.  For
`lam^2 = -s + i eta` the principal root's real part is `eta / (2 sqrt(s))`, so
the ratio grows without limit as `s -> 0` at fixed backward error.  Measured on
the ladder (`v4_hunt.py`, WIN):

* the largest band ratio of a mode that is BOTH in the acted-on population AND
  cleanly propagating (`Re(lam^2) < 0` with `|Im(lam^2)| <= 1e-6 |Re(lam^2)|`)
  is **7.1008e-13**, at `min|lam^2| = 2.284e-08` -- 4.1 decades below the bar,
  and INSIDE it, so the band never MISSES such a mode on this box;
* the largest band ratio of the mode AT cutoff, whatever its class, reaches
  **6.7172e-09** at `min|lam^2| = 4.495e-15` -- identical on both builds --
  **1.5x below the bar**, i.e.
  0.17 decades.  That mode is no longer cleanly propagating (its `lam^2` sits
  at ~45 degrees), so its imaginary sign is genuinely ambiguous rather than
  wrong; but the "fourteen decades with nothing in between" is a statement
  about the fixtures measured, not about the function.

**Both hunts, stated plainly.**

* *A lossless PROPAGATING mode ABOVE the bar* (the band would miss it and leave
  the build-dependent root): **not found** over 75 fixtures plus 28 engineered
  cutoff mounts on each build.  Closest approach 7.1008e-13, 4.1 decades inside.
* *A genuinely LOSSY or evanescent mode BELOW the bar* (the band would conjugate
  a physical root): **not found at any loss a caller would call loss**.  The
  acted-on population is EMPTY at every rung from `Im(eps) = 1e-2` down to
  `1e-13` on both the 2-D anisotropic and the 1-D paths, on both builds.  The
  first rung that puts a mode inside the band is `Im(eps) = 1e-15` (1-D TM,
  ratio 1.229e-17); the anisotropic path's first is `1e-16`.  Those are BELOW
  the eigensolver's own backward error on the same operator (`|Im(lam^2)|` of
  an on-cut mode measured 5.7e-20 .. 5.9e-15 against `||M|| ~ 72`), so at that
  depth no bar could separate the sign from the noise on any build.  The audit's
  ladder stopped at 1e-8; this one extends the clean result by five decades.
* The reason the ladder is clean is worth stating, because it is stronger than
  "the ratio is large": for a lossy layer the acted-on population is EMPTY --
  every mode's `Im(lam^2)` takes the sign the loss dictates, so there is nothing
  for the flip to act on at all.  Measured: `n_population = 0` at every rung
  from 1e-2 to 1e-12.

### 7.1 The band is relative to the LARGEST mode, not to the mode it judges

`v11_scale_relativity.py`.  Because `scale = max(max|r|, 1)`, whether a given
mode counts as "on the cut" depends on the OTHER modes of the same layer:

```
fires  <=>  (|Re(r)| / |r|) * (|r| / max|r|)  <=  _CUT_BAND_REL
```

so a mode whose real part is a fraction `rho` of ITS OWN magnitude is caught
once its magnitude falls below `_CUT_BAND_REL / rho` of the spectrum's top --
five decades of dynamic range in `|lam|` at `rho = 1e-3`.

Measured over the 75-fixture box, the worst case in an ordinary mount is
`|Re(r)|/|r| = 2.5e-13` (an oblique anisotropic cell) -- entirely safe.  At the
engineered TM layer-cutoff mount it reaches **2.0751e-03** on BOTH builds: a mode whose real
part is 0.2 % of its own magnitude is conjugated, because it is 3.8e-08 of the
spectrum's largest root.  That mode sits at `|lam| = 1.7e-07`, carries no
z-flux, and `_inv_lam` regularises it downstream, so the consequence is nil --
but the scale-relative shape is what makes the cutoff corner possible, and it is
not recorded anywhere in the change.

### 7.2 `lam^2` at and below zero, and the `|X| <= 1` guarantee

`v4_hunt.py` hunt C, and gate 1 of the new test file.

| input | returned `lam` |
|---|---|
| `0 + 0j` | `0 + 0j` |
| `0 - 0j` | `0 - 0j` |
| `+5e-324` (denormal) | `2.2228e-162 + 0j` |
| `-1e-300` | `0 + 1e-150j` |
| `-2.25 + 0j` / `-2.25 - 0j` | `0 + 1.5j` in BOTH cases |
| `-2.25 - 2.911e-15j` | `+9.7033e-16 + 1.5j` (flipped) |
| `+2.25 - 2.911e-15j` | `+1.5 - 9.7033e-16j` (untouched -- evanescent) |
| `nan` | `nan + nanj`, no raise |

Over 4,010 values (4,000 pseudo-random over fifteen decades plus every corner):
`min Re(lam) = 0.0`, `0` roots with `Re(lam) < 0`, max root residual
2.588e-16 relative -- and over all 75 solve fixtures,
`max Re(-lam) = 0.000e+00`.  **The `|X| <= 1` contraction is preserved
everywhere measured.**  `conj` is an isometry that fixes the real part, so this
is structural rather than lucky; gate 1 of the new test file pins both halves.

---

## 8. Task 3 -- blast radius, by class

`v5_scope.py` runs 49 surfaces twice in ONE process: once as shipped, once with
the `48c8747` body reinstated in every module that binds `_sqrt_decay`
(`_core`, `rcwa.oned`, `rcwa.twod`, `rcwa.stack`, `elements.berreman`).  The
within-interpreter A/B is used deliberately: the merge `48c8747 -> 75a0c81` also
carries the unrelated `pmm/stack.py` sliver round-3 work, so a cross-tree diff
of a PMM entry point would mis-attribute that change to this one.  The
transcribed body was validated against the real PRE tree first -- it reproduces
`dR = 2.812111e-10` and closures -3.1995e-03 / -3.3813e-04 on WIN at one
thread, every digit.

| class | n | bit-identical | max motion WIN | max motion WSL |
|---|---|---|---|---|
| (a) off-coincidence lossless | 19 | 4 | **2.0474e-14** (`jones2d_offcoinc_tw1.1`) | **2.3564e-14** (`stack_iso_offcoinc`) |
| (b) ON-coincidence lossless | 13 / 14 | 2 | **3.6376e-03** (`jones2d_fold_False`) | **4.3567e-02** (`jones2d_coinc_tw0.7`) |
| (c) LOSSY | 8 | **8 / 8** | **0.00e+00** | **0.00e+00** |
| (d) does not reach the patched function | 5 | **5 / 5** | 0.00e+00 | 0.00e+00 |
| (e) PMM staggered (own copy) | 2 | **2 / 2** | 0.00e+00 | 0.00e+00 |

Surfaces covered: `rcwa_jones_2d` (anisotropic, on and off the coincidence,
four twists, both fold settings, oblique and conical, an OUT-OF-PLANE-tilted
director, a metal block, three loss rungs); `rcwa_efficiency_2d` (scalar, block
and weakly-modulated, on and off the coincidence, lossy); `rcwa_efficiency_1d`
(TE and TM, two duties, oblique, lossy ridge); `rcwa_jones_1d` (conical,
lossless and lossy); `RCWAStack` (isotropic and tensor, both fold settings, on
and off the coincidence); `berreman_jones_1d` (four); and the PMM entry points
`pmm_efficiency_2d`, `pmm_jones_2d`, `pmm_efficiency_1d`, `pmm_jones_1d`,
`pmm_jones_2d_staggered`, `pmm_efficiency_2d_staggered`, `PMM2DStackPure`.

Two surfaces (WIN) / one (WSL) produced no motion number because the PRE arm
RAISED: `stack_tensor_coinc` and `stack_tensor_offcoinc`, the three-layer stacks
of section 6.  Both are refused by the library's own energy tripwire at
`sum R + T = 2.600e+01` on the pre-fix code and return 2.000000000000001 after.
Note that I had labelled `stack_tensor_offcoinc` class (a) -- "off-coincidence"
-- by looking at the SUBSTRATE, which is exactly the misconception section 6
corrects: its uniform spacer coincides with the structured layer's background,
so it is class (b) after all.

**No surface moved against its class.**  Every lossy surface is bit-identical
on both builds, which is the audit's central "where the sign was physics,
nothing moved" claim -- CONFIRMED on eight independent surfaces instead of two.
Class (a) reaches 2.0e-14 / 2.4e-14 rather than the audit's 5.54e-15 / 4.83e-15
envelope; same order, a wider box, and the audit's own statement that lossless
solves move at rounding level off the coincidence stands.

Three scope statements need correcting or sharpening.

* **`berreman_jones_1d` is a NO-OP surface.**  It does import the patched
  function, but only for the homogeneous REGION modes built in exact arithmetic
  -- where the OLD pin already fired.  All four Berreman surfaces are
  bit-identical between arms on both builds.  The audit lists Berreman among the
  places that "now get the pinned root", which is true and empty.
* **The full-3x3 / OUT-OF-PLANE anisotropic path never calls `_sqrt_decay`.**
  It eigendecomposes the first-order generator `G` and selects the forward set
  by z-FLUX (`_select_forward_flux`), a different mechanism with its own
  relative bars (`1e-9 * max|Sz|`, a `3e-3` projection-noise ceiling and a
  `|Re gam| > 0.5` deep-decay override -- no exact-zero pin anywhere).  Both
  OOP surfaces are bit-identical between arms.  The change does not reach that
  path, in either direction.
* **`pmm_jones_2d` IS in class (b), not (d).**  Its LAYER goes through
  `rcwa._core._layer_eigenmodes_tensor`, i.e. through the PATCHED function; it
  moves 1.0229e-03 between the arms.  Only its homogeneous REGION modes use the
  PMM copy.

---

## 9. Task 4 -- the other copies of the same branch test

The audit's blast-radius paragraph says `_sqrt_decay`

> is shared: `rcwa/oned.py`, `rcwa/twod.py`, `rcwa/stack.py`, `pmm/twod.py`,
> `pmm/_jax_twod.py`, `pmm/_jax_stack2d.py`, `pmm/_jax_twod_jones.py` and
> `elements/berreman.py` all call it, and every one now gets the pinned root.

For the PMM half of that list this is **not true**.  `v7_pmm_copies.py` reads it
off the live objects and the files:

| module | defines its own? | resolves to | exact-zero pin | relative band | flips with |
|---|---|---|---|---|---|
| `rcwa/_core.py` | yes | `rcwa._core` | no | **yes** | `conj` |
| `rcwa/oned.py` | imported | `rcwa._core` | no | **yes** | `conj` |
| `rcwa/stack.py` | imported | `rcwa._core` | no | **yes** | `conj` |
| `elements/berreman.py` | imported | `rcwa._core` | no | **yes** | `conj` |
| `pmm/twod.py:411` | **OWN** | `pmm.twod` | **YES** | no | `-r` |
| `pmm/twod_staggered.py:2084` | **OWN** | `pmm.twod_staggered` | **YES** | no | `-r` |
| `pmm/_jax_twod.py:341` | **OWN** (nested) | -- | **YES** | no | `-r` |
| `pmm/_jax_stack2d.py:175` | **OWN** (nested) | -- | **YES** | no | `-r` |
| `pmm/_jax_twod_jones.py:191` | **OWN** (nested) | -- | **YES** | no | `-r` |

`rcwa/twod.py` does not bind the name at all (it reaches the layer solve through
`_core`), so that entry in the audit's list is also inaccurate, harmlessly.

**And the copies are live.**  `pmm/twod.py`'s own `_sqrt_decay` is handed LAYER
eigenvalues by `_layer_modes_projected` (line 669) and `_symmetric_solve_2d`
(line 756), which is the scalar 2-D PMM path used by `pmm_efficiency_2d_cell`,
`pmm_efficiency_2d`, `PMM2DStackHybrid`, `stack2d` and `conical`.  Measured on
`pmm_efficiency_2d_cell` (WIN, one thread):

| fixture | modes | on the cut | with EXACT zero `Im` | with ROUNDING `Im` | max `\|Im(lam^2)\|` | **INCOMING after the shipped pin** |
|---|---|---|---|---|---|---|
| `pmm_cell_coinc` (bg 2.25, `n_sub` 1.5) | 606 | 18 | 12 | 6 | 2.38e-15 | **5** |
| `pmm_cell_weakmod_coinc` | 606 | 18 | 12 | 6 | 5.14e-16 | **6** |
| `pmm_cell_offcoinc` (`n_sub` 1.63) | 606 | 18 | 12 | 6 | 2.38e-15 | **5** |
| `pmm_cell_full_coinc` (fold off) | 726 | 22 | 12 | 10 | 3.29e-15 | **5** |
| `pmm_cell_oblique_coinc` (theta 0.3) | 726 | 28 | 14 | 14 | 2.68e-15 | **9** |
| `pmm_cell_conical_coinc` (0.2, 0.7) | 726 | 26 | 14 | 12 | 5.94e-15 | **8** |

That is the identical defect, unrepaired: five to nine propagating modes per
solve come back on the INCOMING root, chosen by the last bit of the
eigensolver's backward error.  By contrast `pmm_jones_2d`, whose layer goes
through the FIXED function, shows `noisyIm = 0` and `incomingAfterOldPin = 0` --
its PMM-copy calls are region-only and exact.

**What it costs today, measured.**  Installing the fixed body in the two NumPy
PMM copies moves those six fixtures by 2.429e-15 .. 8.660e-15 in per-order
efficiency -- rounding level.  I could not build a PMM fixture where the
mis-rooting meets a matching region mode and corrupts the answer the way the
RCWA one does; the PMM 2-D solves' own truncation error at degree 5 is ~1e-3,
which would mask a smaller effect anyway.  So this is a LATENT defect of the
same class, not a demonstrated wrong answer -- but it is build-dependent
arithmetic in shipped code, and the audit states it is fixed when it is not.

`pmm/twod_staggered.py:2084` defines `_sqrt_decay` and never calls it (the only
other mention is a comment at line 2302): DEAD CODE carrying the old pattern.

No other exact-zero branch pin exists anywhere in the library: a regex for
`.real == 0` / `.imag == 0` (with or without a sign or a decimal point) over
all of `lumenairy/` returns exactly those five PMM sites and nothing else.  `_sqrt_forward` uses an absolute `1e-300` guard on a
quantity built from exact inputs (and is never handed an eigenvalue);
`_select_forward_flux` and `_select_forward_flux_jax` use relative flux bars.

---

## 10. Task 5 -- the `DLASCL` line

`DLASCL(TYPE, KL, KU, CFROM, CTO, M, N, A, LDA, INFO)` sets `INFO = -4` when
`CFROM` is zero or NaN, and the `D` prefix says the emitting driver is
REAL-valued, not complex.  Fortran unit-6 output is buffered and flushed at
interpreter exit, so the line's position in a log attributes nothing -- the
audit's reasoning on that point is sound and I reproduce its negative result:
the WSL task-0 battery of section 2 (the rcwa files plus
`test_v5_14_2_backlog_batch.py`, 235 tests, one thread) ran with stderr merged
into the captured log, and that log's last line is pytest's own summary -- no
`DLASCL` line anywhere.  Whatever emits it, this battery does not.

To go further I sharded the whole of `tests/unit` (514 files, `-m "not
integration and not slow"`) into six independent WSL pytest processes with
stdout and stderr captured per shard, each carrying a `nanwatch` plugin
(`validation/probe_verify_rcwa_even_sector/nanwatch.py`) that wraps every
`numpy.linalg` and `scipy.linalg` entry point and records, per test nodeid, any
call whose input or output is non-finite.  That sweep was still running when the
coordinator supplied the file-level answer, so it was abandoned in favour of the
much cheaper per-test bisect below; the plugin is kept because it is the general
instrument for this question.

**FOUND.**  On WSL the whole of `tests/unit/test_m1_conditioning_guard.py`
prints exactly two `DLASCL` lines and every other file prints none; running its
27 node ids one at a time narrows those two to a SINGLE test,
`test_guarded_lstsq_stands_aside_on_a_non_finite_system`.  `v13_dlascl.py`
then reproduces that test's two calls in isolation and counts the lines per
CALL:

| call | what it does | WSL lines | WIN lines |
|---|---|---|---|
| clean control | a finite 12 x 4 complex least-squares system | 0 | 0 |
| (a) non-finite **A** | `A[3,1] = nan`, expects `LinAlgError` | **2** | 0 |
| (b) non-finite **b** | `b[5] = nan`, must return NaNs, no raise | 0 | 0 |
| both (the test as written) | | **2** | 0 |

So the line is **BENIGN and DELIBERATE**: it is what LAPACK prints when
`numpy.linalg.lstsq` is handed a matrix containing a NaN, which that test does
ON PURPOSE and asserts raises.  `A` is complex, so the driver is `zgelsd`, and
`zgelsd` scales the REAL singular values with `DLASCL` -- which is why the
complaint names the double-REAL routine on a complex solve.  It has nothing to
do with the RCWA solve, the branch cut or the even-sector fold, and it appears
in a WSL log and not a Windows one because the SkylakeX-dispatched `zgelsd`
reaches `DLASCL` with `CFROM = NaN` while the Haswell-dispatched one screens
the NaN earlier and returns a positive `INFO` instead.

Two details worth recording.  First, the test's own comment already anticipates
the line -- "LAPACK writes a DLASCL complaint to stderr for each non-finite
gelsd, and one line of CI noise is enough to make the point" -- but it is wrong
in two particulars: the ONE call it kept emits **two** lines, not one, and they
go to Fortran unit 6 (**stdout**), not stderr, which is why they survive a
`2>/dev/null` and land in the middle of a captured log.  Second, this is the
audit's section 8 open item, closed: **the `DLASCL` line is not this solve's,
is not a defect, and is fully attributed.**

*Reproducer:*
`wsl ... PYTHONPATH=. python validation/probe_verify_rcwa_even_sector/v13_dlascl.py A`
-> two lines on WSL, none on Windows.

---

## 11. Task 6 -- the fix's own test file, constant by constant

`v8_testbars.py` re-derives every quantity `tests/unit/test_fix_rcwa_even_sector_wsl.py`
asserts on, on both arms, at 1 / 2 / 4 / 6 / 8 / 16 / unpinned threads, on both
builds (14 samples).

| constant | stated origin | re-measured | verdict |
|---|---|---|---|
| `_ETA_LADDER`, `_KZ` (BAR 1) | "walks the measured backward-error range and four decades either side" | engineered inputs; POST returns `Im >= 0` on 60/60 at every setting, `min Re(lam) = 0`; PRE returns the incoming root for 55 of 60 | sound, build-free |
| root residual `1e-12 * max\|lam^2\|` (BAR 1) | not stated | measured **2.64e-14** relative, IDENTICAL on both builds at every thread count -> 1.6 decades of room | thin but build-free (pure `numpy.sqrt` on fixed constants); no build can move it |
| `1e-12` root bar, evanescent guard (BAR 1b) | -- | `min Re(lam) = 0.5` at every setting | sound |
| `_SIGNAL_SIDE_MIN = 7.9e-2` (BAR 2) | "measured; the bar is 6.9 decades below it" | the signal-side minimum over my 75-fixture box is **1.2612e-02** on both builds -- 6.3x below the pinned value | **sample-scoped**; the assertion is unharmed (its second pass at 7.9e-3 already brackets 1.26e-2) but the stated population minimum is one sample's reading |
| `_CLOSURE_BAR = 1e-9` (BAR 3) | "5.2 decades above the post-fix envelope and 3.0 decades below the smallest pre-fix reading (1.084e-06)" | POST side CONFIRMED (`\|defect\| <= 1.5e-14` over 14 samples and 12 extra fixtures). PRE side **REFUTED for the `[True]` parametrization**: the EVEN path reads **-3.109e-15** (WIN, 6 threads), **-3.753e-14** (WSL, 4 threads) and **-4.174e-14** (WSL, unpinned) -- three of fourteen samples where the fail-before does NOT fire | bar sound, **stated margin wrong on one arm** |
| `_CLOSURE_BAR`, oblique parametrizations | pre-fix +7.877e-04 / -1.172e-04 | fires at every one of the 14 samples; smallest pre-fix magnitude 2.175e-06 (WIN unpinned, theta 0.2 phi 0.7) | sound |
| `_FOLD_BAR = 1e-11` (BAR 4) | "4.5 decades above the post-fix envelope ... does NOT have decades below every pre-fix reading" | POST `dR <= 2.78e-16`, `dJ <= 6.95e-16` over 14 samples: CONFIRMED. PRE passes at WIN 4/6/16/unpinned and WSL 4/16 -- 6 of 14 | sound and **honestly disclosed** in the file |
| BAR 5 on-cut census | "10 of 16 on Windows, 8 of 16 on WSL" | POST `0/16` at every one of the 14 samples; PRE 5..13 of 16 -- fires at EVERY setting on both builds | the most robust fail-before in the file |

The file passes at every threading tried: WIN 9 passed at 1 / 4 / 8 / 16 threads
and unpinned; WSL included in the pinned and unpinned batteries of section 2.

**The one durability defect** is the `[True]` arm of BAR 3.  The audit's
pre-fix envelope ("minimum 1.084e-06") is explicitly a FULL-path list, but it is
presented as the margin for a test that is parametrized over both paths, and
the even path's pre-fix reading reaches the arithmetic floor at some thread
counts.  This does not weaken the POST-fix assertion (which is what gates the
release); it means that arm's fail-before demonstration is thread-conditional,
the S2 shape `docs/TESTING_STANDARDS.md` names.  The remedy is one sentence in
the comment, or the uniform-spacer fixture of section 6, whose fail-before fires
identically at every thread count on both builds.

---

## 11b. THE M1 CONDITIONING GUARD, AND X-1

Raised by the coordinator mid-verification and re-measured here from scratch.
Everything in this section is `v12_m1_instrument.py` and `v14_x1.py`, both arms
in one process, at five thread counts on both builds.

### 11b.1 The failing test, and why it fails

`tests/unit/test_m1_conditioning_guard.py::test_anisotropic_cascade_is_not_falsely_refused`
FAILS on the merged tree: Windows one thread reads
`1 failed, 21 passed, 5 skipped`, against `27 passed` on the PRE tree.  It is
not a physics failure -- it is the test's PREMISE that has stopped being true.

That test exists to document the FALSE POSITIVE that made `_guarded_inverse`
score the EQUILIBRATED operator rather than the raw one: a uniaxial
`rcwa_jones_1d` cascade (period 1 um, ridge `uniaxial_tensor(1.5, 1.8, pi/2,
phi = 20 deg)`, groove `1.5^2`, `n_substrate` 1.5, depth 0.4 um, duty 0.5,
angle 10 deg) whose RAW inverse residual runs 1e-02 .. 5e-01 at every
truncation while the equilibrated one reads 1e-15 .. 6e-14 and the answer is
right.  Its premise (b) asserts `raw > 1e3 * _INV_RESID_REFUSE`.

Note the geometry: groove permittivity `1.5^2` and `n_substrate = 1.5`.  It is
a COINCIDENCE cell.  Measured (Windows, one thread; the sites are the Redheffer
star denominators `I - B11 A22` / `I - A22 B11` and the interface mode-match --
NOT the armed `T22` site, which this cell never reaches):

| rung | raw residual PRE / POST | equilibrated PRE / POST | equilibrated `rcond` PRE / POST |
|---|---|---|---|
| M = 5 | **5.155e-01** / 3.660e-16 | 1.591e-15 / 5.093e-16 | 1.683e-03 / 4.260e-02 |
| M = 9 | **2.837e-01** / 2.929e-16 | 3.745e-15 / 3.302e-16 | 3.127e-04 / 4.170e-02 |
| M = 15 | **1.176e-02** / 4.105e-16 | 3.775e-15 / 4.083e-16 | 1.035e-04 / 3.522e-02 |
| M = 19 | **1.712e-02** / 4.564e-16 | 1.269e-14 / 4.656e-16 | 3.227e-05 / 2.592e-02 |
| M = 25 | **1.079e-02** / 4.865e-16 | 5.784e-14 / 5.571e-16 | 1.237e-05 / 2.304e-02 |

and the causal control, the SAME cell with `n_substrate` walked to 1.63 so the
coincidence is gone:

| rung | raw PRE | raw POST |
|---|---|---|
| DETUNED M = 5 | 3.872e-14 | 4.151e-16 |
| DETUNED M = 15 | 2.964e-15 | 4.105e-16 |
| DETUNED M = 25 | 1.368e-14 | 4.865e-16 |

Pre-fix the raw residual is at the floor as soon as the coincidence is removed.
**The "false positive the equilibration exists for" WAS the branch-cut defect.**
It also carries the defect's signature: at fixed geometry the pre-fix raw
residual at M = 25 reads 1.079e-02 / 1.567e-01 / 2.302e-04 / 3.662e-02 /
1.268e-02 at 1 / 2 / 4 / 8 / 16 threads on Windows -- three decades with the
BLAS pool -- while post-fix it reads 4.865e-16 / 4.977e-16 / 4.594e-16 /
4.626e-16 / 4.912e-16, flat.  WSL is the same picture (M = 5 raw 6.991e-01
pre-fix at every count, 3.556e-16 post).

### 11b.2 Does the equilibrated-residual instrument retain a motivating population?

A call is MOTIVATING when a RAW-residual bar would refuse it
(`raw > _INV_RESID_REFUSE = 1e-8`) and the equilibrated one rescues it
(`eq <= 1e-8`) -- that is exactly the population the instrument was chosen for.
Over a 24-fixture sweep (the M1 cell at five truncations and two mounts, its
detuned control, the 2-D anisotropic coincidence, the uniform-spacer stack, the
1-D groove and metal cases, a loss ladder, a conical cell, and four
out-of-plane / Berreman fixtures that DO reach the armed `T22` site):

| | PRE | POST |
|---|---|---|
| motivating calls, WIN | **9 of 106** at 1 / 2 / 4 / 8 / 16 threads | **0 of 110** at every count |
| motivating calls, WSL | **9 of 106** (8 at 4 threads) | **0 of 110** at every count |
| armed `T22` equilibrated `rcond`, minimum | 2.792e-02 | 2.792e-02 (bar `1e-10`) |

Every one of the nine was the branch-cut defect: five M1 rungs, the
normal-incidence M1 variant, the 2-D anisotropic coincidence cell, and two
calls in the uniform-spacer stack.  Post-fix the sweep contains no call at all
where the raw and equilibrated instruments disagree by more than 3x.

The ARMED refusal is a separate population and is untouched: its minimum
equilibrated `rcond` over the sweep is 2.792e-02 on both arms and both builds,
**eight decades above** its own `1e-10` bar, so nothing here comes near it.

### 11b.3 X-1 is CLOSED by this fix

`test_x1_defect_is_reproduced_and_flagged_but_NOT_closed` pins the library's
own documented OPEN instability class on the `THIN` family: period 10 um, ridge
1.55, groove 1.5, **substrate 1.5, superstrate 1.5**, depth 0.5 um, duty 0.5,
`n_orders` 6..30.  The groove index equals BOTH half-spaces': a coincidence on
both sides at once.

`v14_x1.py` runs that whole ladder on both arms with the library's own census
armed:

| ladder | arm | cells that RAISE | cells the census FLAGS | worst closure | worst relative `sum(R)` error | min equilibrated `rcond` |
|---|---|---|---|---|---|---|
| TE, WIN 1 thr | PRE | **7 of 25** | **14 of 25** | 3.196e-02 | **152.6x** (M = 21) | 3.331e-19 |
| TE, WIN 1 thr | POST | **0** | **0** | **1.332e-15** | 5.064e-02 (M = 6, truncation) | 6.250e-02 |
| TM, WIN 1 thr | PRE | 5 | 9 | 2.616e-04 | 1.300x (M = 28) | 3.402e-19 |
| TM, WIN 1 thr | POST | **0** | **0** | **9.992e-16** | 1.887e-02 (M = 6) | 5.603e-02 |
| TE, WSL 1 thr | PRE | 8 | 14 | 3.196e-02 | 152.6x (M = 21) | 3.614e-19 |
| TE, WSL 1 thr | POST | **0** | **0** | 1.443e-15 | 5.064e-02 | 6.250e-02 |
| TE, WSL 4 thr | PRE | 8 | 14 | 5.903e-03 | 28.19x | 5.574e-19 |
| TM, WSL 4 thr | PRE | 5 | 9 | 1.287e-02 | 63.89x (M = 27) | 2.338e-19 |

The three pinned cells, verbatim (Windows, one thread):

| cell | PRE closure / `sum(R)` | POST closure / `sum(R)` |
|---|---|---|
| M = 19 TE (the build-dependent `R+T = 1.018`) | 1.8182e-02 / 1.838764e-02 | **1.1102e-15 / 2.053766e-04** |
| M = 21 TE (the "160x wrong" answer) | 3.1956e-02 / **3.216567e-02** | **5.5511e-16 / 2.095174e-04** |
| M = 20 TE | 3.5080e-06 / 2.088570e-04 | 4.4409e-16 / 2.053491e-04 |
| M = 12 TE | 6.2716e-08 / 2.016454e-04 | 8.8818e-16 / 2.015824e-04 |

`sum(R)` post-fix is 2.05e-04 .. 2.10e-04 at every rung -- the converged value
the M1 test names -- and the POST readings are IDENTICAL on Windows and WSL and
across thread counts.  The library's own X-1 tests confirm it from their side:
they self-widen, find nothing that manifests, verify the census instrument is
still alive on a synthetic `cond = 1e20` operator, and SKIP with

> "X-1 does not manifest anywhere in the scanned ladder on this build ... the
> screen is live and the thin-grating family simply carries no near-cancelling
> star denominator here"

-- five skips post-fix against 27 passed pre-fix.

**So X-1 was the same defect, and this change closes it.**  That is the single
largest consequence of the fix, and neither the audit nor the change's own test
file mentions it.

### 11b.4 Recommendations

1. **The restatement of `test_anisotropic_cascade_is_not_falsely_refused` is
   right in shape.**  Premise (b) must invert -- the raw residual now reads
   3.7e-16 .. 4.9e-16 where it read 5.2e-01 .. 1.1e-02 -- and the closure
   ladder, which the v5.33.1 work correctly refused to bar because it wandered
   twelve decades with the pool, can now carry a bar: the post-fix closure of
   that cell is 1.1e-15 .. 5.6e-16 on both builds at every thread count
   measured, so a `1e-10` bar has five decades below it and, pre-fix, eleven
   decades of signal above it.  The docstring's long "the closure carries no
   truncation information" argument should be kept as HISTORY and marked as
   describing the pre-fix operator: post-fix the same quantity is stationary,
   and the reason it wandered was the branch choice, not the star denominators.
2. **`test_x1_defect_is_reproduced_and_flagged_but_NOT_closed` should be
   re-pinned as CLOSED**, which is what its own docstring asks for ("a future
   fix should make this test fail").  It currently SKIPS instead of failing,
   because its widening path treats "nothing manifests" as probe-backed
   absence; that path was written for a per-build absence, not for a fix.  The
   replacement claim is available and two-sided: on the `THIN` ladder the
   post-fix closure is <= 1.5e-15 at all 25 truncations x 2 polarizations x 2
   builds, the census flags 0 cells, and `sum(R)` agrees with the converged
   2.05e-04 to 5e-02 relative at the COARSEST rung and better everywhere else,
   against 152.6x and 14 flagged cells pre-fix.
3. **The instrument is NOT dead code, but its justification is.**
   `_equilibrated_inverse_residual` and `_rcond_1_equilibrated` still work (the
   suite's own `_census_instrument_is_alive()` still flags a synthetic
   `cond = 1e20` operator, and the armed `T22` refusal is a different
   population that this change does not touch).  What is gone is the MEASURED
   population that chose equilibration over the raw residual: 0 motivating
   calls of 110 across the sweep, 0 flagged cells on the X-1 family, on both
   builds at every thread count.  I would KEEP the guard -- it is dormant, it
   costs nothing, and a user's own coincident geometry can still reach it --
   and re-derive its justification against a state that is ENGINEERED rather
   than found: the pre-fix branch injected deliberately, or a synthetic
   badly-scaled operator.  Deleting it on the strength of an empty population
   would repeat, one level up, the mistake this campaign exists to prevent.
4. Note for the release: the M1 file is the ONLY place the fix turns something
   red, and it turns it red for the right reason.  It was not in this
   verification's task-0 battery (which was scoped to `test_rcwa*` /
   `test_niche*rcwa*`), so `test_m1_conditioning_guard.py` belongs in whatever
   battery gates this change.

---

## 12. Verdict table

| # | Claim | Verdict | Measured, WIN / WSL |
|---|---|---|---|
| 1 | Deterministic per (build, tree, thread count) | **CONFIRMED** | 3/3 identical digits in 32 runs; `all_deterministic = True` in all |
| 2 | Pre-fix `dR` moves twelve decades with the BLAS thread count, `1e-8` bar inside the spread | **CONFIRMED** | 3.2e-14 .. 2.8e-10 / 1.9e-13 .. 1.6e-02, every quoted digit reproduced |
| 3 | Post-fix `dR` is thread-independent | **CONFIRMED** | <= 2.776e-16 / <= 2.151e-16 over 8 settings |
| 4 | `cond(a+b)` 1.97e+15 (layer->sub) vs 1.54e+04 (sup->layer), -> 5.63e+03 | **CONFIRMED** | 1.973e+15 / 1.541e+04 -> 5.633e+03 (WIN, 1 thread), 4 s.f. |
| 5 | Offender `lam^2 = -2.249999999999987 - 2.911e-15j`, null direction one mode at `-1.5j` / `-0.9j` | **CONFIRMED** | identical to every printed digit, both arms, both fold paths |
| 6 | Tiny singular values 4 (full) / 2 (even) -> 0; IPR 2.27 / 1.11 -> 3.31 | **CONFIRMED** | 4 / 2 -> 0 / 0; 2.27 / 1.11 -> 3.31 / 2.00 |
| 7 | Mechanism: a LAYER mode must meet a REGION mode; equal permittivity is necessary, not sufficient | **CONFIRMED, scope EXTENDED** | 4 non-coincident-mode fixtures close at 2e-15..1.1e-14 on both arms; but a UNIFORM LAYER is a partner too (sec. 6), and the SUPERSTRATE side works (sec. 5.1) |
| 8 | The scalar 2-D and 1-D paths are "not automatically one of those places" | **BOUNDED / partly REFUTED** | a weakly-modulated scalar 2-D cell at the coincidence: `cond` 2.504e+08 / 7.260e+07 -> 9.008e+01 / 6.372e+01 |
| 9 | The band separates two populations by fourteen decades; bar 7.6 decades above the noise side, 6.9 below the signal side | **RESTATED (sample-scoped)** | over 75 fixtures: noise side max 7.9357e-11 (both builds), signal side min 1.2612e-02 (both) -> 2.1 and 6.1 decades |
| 10 | A loss ladder to `Im(eps) = 1e-8` contributes nothing to the noise side | **CONFIRMED and EXTENDED** | acted-on population EMPTY from 1e-2 to 1e-13; first in-band mode at `Im(eps) = 1e-15`, below the eigensolver's own backward error |
| 11 | No lossless propagating mode is missed by the band | **CONFIRMED** | closest approach 7.1008e-13 (4.1 decades inside), over 75 fixtures + 28 engineered cutoff mounts |
| 12 | `conj(r)` keeps `Re(lam) >= 0`, so `\|X\| <= 1` | **CONFIRMED** | 0 of 4,010 roots with `Re < 0`; `max Re(-lam) = 0.000e+00` over 75 solve fixtures |
| 13 | `lam^2 = 0` / near-zero handled | **CONFIRMED** | `0 -> 0`, denormal -> 2.2228e-162, `nan` propagates without raising |
| 14 | The two lossy census fixtures are bit-identical between arms | **CONFIRMED, widened** | 8 of 8 lossy surfaces bit-identical on BOTH builds |
| 15 | Off-coincidence lossless motion 1.04e-17 .. 5.54e-15 | **RESTATED** | 2.0474e-14 / 2.3564e-14 over 19 surfaces -- same order, wider box |
| 16 | On-coincidence motion up to 1.81e-02 | **CONFIRMED** | 3.6376e-03 / 4.3567e-02 over 13-14 surfaces |
| 17 | `pmm_efficiency_2d` is bit-identical between the arms | **CONFIRMED, reason REFUTED** | bit-identical because it does not call the patched function at all, not because the coincidence fails to bite |
| 18 | "`pmm/twod.py`, `pmm/_jax_twod.py`, `pmm/_jax_stack2d.py`, `pmm/_jax_twod_jones.py` ... every one now gets the pinned root" | **REFUTED** | five PMM files keep their OWN `_sqrt_decay` with the exact `r.real == 0` pin; measured 5-9 modes per `pmm_efficiency_2d_cell` solve come back INCOMING |
| 19 | Berreman gets the pinned root | **RESTATED** | true but empty: all 4 Berreman surfaces bit-identical (region modes only, exact arithmetic) |
| 20 | `RCWAStack` isotropic unaffected | **CONFIRMED** | 1.010e-14 / 2.356e-14 -- rounding level; the TENSOR stack is REPAIRED from a 26x refusal |
| 21 | The band has the same value and shape as PMM's `_forward_branch_flip` | **CONFIRMED, with a caveat** | both are `1e-8 * max(max\|.\|, 1)`; but they threshold DIFFERENT quantities and flip differently (`-q` vs `conj(r)`), so "one convention for one quantity" is a stretch |
| 22 | The `1e-8` band was not swept; placement inside the gap is immaterial | **BOUNDED** | true away from cutoff; at a layer cutoff the noise side reaches 6.7172e-09, 1.5x below the bar, so the placement is NOT immaterial there |
| 23 | The `DLASCL` line is a bystander; its origin is not established | **CONFIRMED bystander, and now ESTABLISHED** | bisected to `test_m1_conditioning_guard.py::test_guarded_lstsq_stands_aside_on_a_non_finite_system`, whose deliberate NaN matrix makes `zgelsd` call `DLASCL` with `CFROM = NaN`: 2 lines on WSL, 0 on WIN, 0 for every other call |
| 24 | `_CLOSURE_BAR` sits 3.0 decades below the smallest pre-fix reading | **REFUTED for the `[True]` arm** | pre-fix even-path closure reaches -3.109e-15 (WIN, 6 threads) and -4.174e-14 (WSL, unpinned): 3 of 14 samples pass pre-fix |
| 25 | `_SIGNAL_SIDE_MIN = 7.9e-2` is the measured signal-side minimum | **RESTATED (sample-scoped)** | 1.2612e-02 over 75 fixtures, both builds |
| 26 | (not claimed) the fix leaves the M1 conditioning guard's evidence intact | **REFUTED** | the guard's motivating population goes from 9 of 106 calls to **0 of 110**, on both builds at 1/2/4/8/16 threads; `test_anisotropic_cascade_is_not_falsely_refused` FAILS on the merge |
| 27 | (not claimed) X-1, the library's documented OPEN instability class, is unaffected | **REFUTED -- the fix CLOSES it** | the `THIN` ladder goes from 7-8 raising cells, 14 flagged and a 152.6x-wrong `sum(R)` to 0 raising, 0 flagged and <= 1.5e-15 closure, on both builds |

---

## 13. Defects

**D1 -- MEDIUM. Five unfixed copies of `_sqrt_decay`, and a blast-radius claim
that says they are fixed.**
`lumenairy/elements/pmm/twod.py:411`, `twod_staggered.py:2084`,
`_jax_twod.py:341`, `_jax_stack2d.py:175`, `_jax_twod_jones.py:191` each define
their own `_sqrt_decay` with the exact `Re(r) == 0` pin and the `-r` flip.  The
`pmm/twod.py` copy is handed LAYER eigenvalues by `_layer_modes_projected` and
`_symmetric_solve_2d`, and 5-9 propagating modes per `pmm_efficiency_2d_cell`
solve come back on the INCOMING root as a result.  Pre-existing (the PMM copies
were never in scope of this change), so NOT a regression -- but the audit
asserts they are fixed, which is the one statement in it that is wrong in
substance rather than in margin.
*Reproducer:* `python validation/probe_verify_rcwa_even_sector/v7_pmm_copies.py out.json`
(`PYTHONPATH=.`), the `incomingAfterOldPin` column.
*Remedy:* route the five copies through `rcwa._core._sqrt_decay` (the JAX ones
need the traced `xp` form `_forward_branch_flip` already demonstrates), or fix
the audit's paragraph.  Cost measured at 2.4e-15 .. 8.7e-15 on the fixtures
tried, so this is hardening, not a hotfix.

**D2 -- LOW. The `[True]` fail-before of
`test_coincident_layer_and_region_permittivity_closes_energy` is
thread-conditional.**
The audit's quoted pre-fix envelope ("minimum 1.084e-06") is a FULL-path list;
the even path's pre-fix closure reads -3.109e-15 (WIN, 6 threads), -3.753e-14
(WSL, 4 threads) and -4.174e-14 (WSL, unpinned), i.e. the test PASSES on the
pre-fix tree at 3 of 14 (build, thread) samples.  The post-fix assertion is
unaffected.
*Reproducer:* `OPENBLAS_NUM_THREADS=6 python validation/probe_verify_rcwa_even_sector/v8_testbars.py out.json`
on Windows, the `PRE-ARM WOULD PASS` line.
*Remedy:* say so in the comment, or add the uniform-spacer fixture of section 6
(now gate 2 of `tests/unit/test_verify_rcwa_even_sector.py`), whose fail-before
fires at every thread count on both builds.

**D3 -- LOW. `_CUT_BAND_REL`'s stated two-sided margin is sample-scoped.**
The docstring claims 7.6 decades above the noise side and 6.9 below the signal
side.  Over a 75-fixture box the margins are 2.1 and 6.1 decades, and at a mount
driven onto a LAYER CUTOFF the noise side reaches 6.7172e-09 -- 1.5x below the
bar.  The DECISION still holds everywhere measured; the stated envelope does not.
*Reproducer:* `v3_band.py` (`max_below = 7.9357e-11`) and `v4_hunt.py`
(`CLOSEST APPROACH BY THE MODE AT CUTOFF`).
*Remedy:* re-derive the two numbers in the docstring against a box that includes
cutoff mounts, and say that the noise side is set by `sqrt(|lam^2|_min)` rather
than by the eigensolver's backward error alone.

**D4 -- LOW. The band's scale is the largest root of the array, so its verdict
on a mode depends on the other modes.**
Measured worst case in an ordinary mount: `|Re(r)|/|r| = 2.5e-13`.  At the
engineered TM cutoff mount: **2.0751e-03** -- a mode 0.2 % off the cut in its
own terms is conjugated because it is 3.8e-08 of the spectrum's top.  The mode
carries no z-flux and `_inv_lam` regularises it, so no wrong answer follows on
any fixture tried.  Not a regression (the PMM `_forward_branch_flip` has the
same shape), but undocumented.
*Reproducer:* `v11_scale_relativity.py`.

**D5 -- LOW. One mount moves from REFUSED to WARNED.**
At an engineered layer-cutoff mount (`min|lam^2| = 4.495e-15`, TE, `n_orders`
11) the pre-fix arm raises `_EnergyError` and the post-fix arm returns with an
`_EnergyWarning` at a closure defect of **+4.2519e-02**.  Over the whole cutoff
ladder (28 mounts, two polarizations, both arms, both builds -- WSL reads the
same `+4.2519e-02` at the same `|lam^2| = 4.495e-15`) there are **zero**
transitions to SILENT, and the TM ladder is clean at ~1e-15 throughout.  So the library is
never silently wrong there -- but the severity does drop by one step on one
mount.  Gate 4 of the new test file pins the loudness, not the digit.
*Reproducer:* `v6_cutoff.py`, the `RAISED -> WARNED` row.

**D7 -- MEDIUM (test, not library). The M1 guard's evidence base is gone, and
one test fails because of it.**
`test_anisotropic_cascade_is_not_falsely_refused` FAILS on the merged tree
(Windows one thread: `1 failed, 21 passed, 5 skipped`, against `27 passed` on
the PRE tree) because its premise (b) -- "the RAW instruments would refuse
every rung" -- has stopped being true: the raw residual on that cell falls from
5.155e-01 .. 1.079e-02 to 3.66e-16 .. 4.87e-16.  The five skips are the X-1
tests, which no longer manifest.  Section 11b has the full measurement, the
recommended restatement and the dead-code question.  The library is not wrong
here; the tests describe a world the fix has changed.
*Reproducer:* `python -m pytest tests/unit/test_m1_conditioning_guard.py`.

**D6 -- INFO. The remedy text is now doubly stale.**
`_check_energy`'s message tells the user to detune against `n_substrate^2` or
`n_superstrate^2`.  The audit already records that the CAUSE it addresses has
been removed; section 6 adds that the coincidence partner can be another
uniform LAYER, which the text does not mention and against which its advice
gives no guidance.

---

## 14. Ship recommendation for 5.45.0

**SHIP.**  The change is correct, it is a strict improvement everywhere I could
measure, and its central claims survive independent re-measurement:

* it removes a build-dependent branch choice that moved a published quantity
  over twelve decades with the BLAS thread count alone, on code that is
  byte-identical at the published v5.44.0;
* it leaves every lossy solve bit-identical (8 of 8 surfaces, both builds);
* it moves off-coincidence lossless solves only at rounding level
  (<= 2.4e-14 over 19 surfaces, both builds);
* it repairs, on my own fixtures, a three-layer stack that the pre-fix library
  REFUSES at `sum R + T = 26` on both builds and returns at 2.000000000000001
  after -- including at a substrate index where nothing coincides with a region
  at all;
* and the `|X| <= 1` contraction the whole function exists to provide is
  preserved over 4,010 engineered eigenvalues and 75 solve fixtures.

Conditions, none of them blocking:

1. Correct the blast-radius paragraph (D1).  Either route the five PMM copies
   through the fixed function -- which I would do in a separate change, since it
   moves PMM answers at 1e-15 and deserves its own gate -- or say plainly that
   they are unrepaired.  Shipping a fix whose audit claims a wider reach than it
   has is the failure mode `docs/TESTING_STANDARDS.md` calls
   right-conclusion-wrong-numbers.
2. Re-derive the two margin numbers in `_CUT_BAND_REL`'s docstring (D3) and add
   one sentence on the cutoff corner.
3. Qualify the `[True]` fail-before claim (D2), or adopt the uniform-spacer
   fixture as its demonstration.
4. The `_check_energy` remedy text (D6) should name a coincident uniform LAYER
   as well as a region.  The audit deliberately defers this while unreleased;
   at release it is worth one line.
5. Restate the two M1 tests (D7, section 11b), and add
   `tests/unit/test_m1_conditioning_guard.py` to the battery that gates this
   change -- it is the only file the fix turns red, and it turns red for the
   right reason.  Record in the CHANGELOG that this change CLOSES X-1: that is
   a user-visible correctness improvement on the default 1-D path (a 152.6x
   error in `sum(R)` and a build-dependent `R + T = 1.018` both removed) and it
   is currently documented nowhere.

---

## 15. What this verification did NOT establish

* **Whether the PMM copies can produce a WRONG answer, not merely a
  build-dependent one.**  I built nine PMM 2-D fixtures at and off the
  coincidence and none put a PMM layer mode on top of a PMM region mode the way
  the RCWA anisotropic cell does; the largest motion from fixing the copies was
  8.660e-15.  The mechanism is present and measured; the consequence is not.
* **The widest admissible `_CUT_BAND_REL`.**  Not swept.  Section 7 bounds it
  from below on this box (a cutoff mode reaches 6.7e-09, so anything at or below
  ~1e-9 would start missing modes the band currently catches) and from above
  (the signal side begins at 1.26e-02), but no ladder was run.
* **GPU / CuPy and JAX arms.**  Every measurement here is NumPy on CPU.  The
  patched function is `xp`-generic and the audit records that `jax.jit` and
  `jax.grad` pass through it, but the JAX PMM copies (D1) are untouched by both
  of us.
* **`stabilize=True`.**  Not re-characterised; its retry schedule exists for
  this failure class and whether it can now be narrowed is open, as the audit
  says.
* **Whether anything OTHER than the branch cut ever motivated the M1
  equilibration.**  Section 11b shows the population is empty post-fix across
  110 guarded inverses on 24 fixtures and the whole X-1 ladder, but "no fixture
  I built reaches it" is not "nothing can".
* **Anything about `pmm/stack.py`.**  The merge under test also carries the
  unrelated sliver round-3 change; every A/B in section 8 is taken within one
  interpreter precisely so that change cannot leak into these numbers.

---

## 16. Reproduction

```
git worktree add -b verify/rcwa-even-sector C:/tmp/lum_vrcwa 75a0c81
git worktree add --detach C:/tmp/lum_vrcwa_pre 48c8747

# every probe, either tree, either build -- PYTHONPATH=. is REQUIRED
cd C:/tmp/lum_vrcwa && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 PYTHONPATH=. python -u \
  validation/probe_verify_rcwa_even_sector/v3_band.py out.json
```

Each probe calls `v_fixtures.require_local_tree()`, which REFUSES to produce a
number if `lumenairy` was imported from anywhere but the working directory.
That guard is not decorative: `python some/dir/probe.py` puts the SCRIPT's
directory on `sys.path` and never the working directory, and one census in this
verification was taken against an unrelated development checkout on another
drive before the guard was added.  Every JSON carries a `_stamp` block naming
the `lumenairy.__file__` it measured, the arm it detected from the presence of
`_CUT_BAND_REL`, the interpreter, numpy, and the three thread environment
variables.

| probe | what it measures |
|---|---|
| `v1_threads.py` | the thread sweep, 3 repeats per setting |
| `v2_mechanism.py` | 12 fixtures: interface conditioning, on-cut census, closure |
| `v3_band.py` | the 75-fixture band census |
| `v4_hunt.py` | both misclassification hunts + the `lam^2` corners |
| `v5_scope.py` | 48 surfaces, within-interpreter A/B, by class |
| `v6_cutoff.py` | RAISED / WARNED / SILENT over a cutoff ladder |
| `v7_pmm_copies.py` | which modules carry the old pin, and whether it is live |
| `v8_testbars.py` | every constant of the fix's own test file, both arms |
| `v9_headline.py` | the audit's headline numbers on its own fixture |
| `v10_stack_spacer.py` | the uniform-spacer coincidence |
| `v11_scale_relativity.py` | the band's dependence on the rest of the spectrum |
| `nanwatch.py` | pytest plugin: non-finite arrays into LAPACK, per nodeid |
| `v12_m1_instrument.py` | the M1 guard's raw / equilibrated instruments, both arms |
| `v13_dlascl.py` | the `DLASCL` line, attributed per CALL |
| `v14_x1.py` | the X-1 `THIN` ladder, both arms, with the census armed |
