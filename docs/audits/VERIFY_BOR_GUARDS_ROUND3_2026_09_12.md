# INDEPENDENT VERIFICATION of ROUND 3 of the 5.45.1 BOR/EME guards — 2026-09-12

> **STATUS — VERIFICATION.** Re-measures the three behaviour changes and the two
> test restatements of `docs/audits/FIX_BOR_GUARDS_ROUND3_2026_09_12.md`
> (`fix/bor-guards-round3`, `384be78e`, forked from the round-2 verification
> `1ac6de7e`) on fixtures written for this verification, against the same
> baseline.
>
> Tree: `C:\tmp\lum_vbor3`, branch `verify/bor-guards-round3`. Baseline tree:
> `C:\tmp\lum_vbor3_pre`, detached at `1ac6de7e`. Probes and per-arm JSON:
> `validation/probe_verify_bor_round3/`. Binding: `docs/TESTING_STANDARDS.md`.
>
> **Nothing here is read out of the fix round's numbers.** Every geometry is
> this verification's own — `k0` = 3.0 against their 2.0, half-spaces at
> `eps` = 2.25 against their 2.0, four middle-layer families they do not use, a
> near-cutoff fixture at `Rbig` = 19 / `N` = 96 / `n` = 1.63 against their
> 24 / 120 / 1.41, and a SEM liner at `Rbig` = 17 and 12.5 / `m` = 2 / degree 6
> against their 24 / 1 / 8. Where a fix-round number is quoted it is labelled
> as theirs.

---

## 1. Verdict table

| # | claim (round 3) | verdict | what I measured |
|---|---|---|---|
| **2a** | the split refuses ladder D **4/13 → 13/13** ({energy 4, ceiling 9}) | **CONFIRMED in kind, on my own ladder** | my 6-geometry x 13-rung ladder with the loss on `layers[0]` ALONE: **18/78 → 58/78** ({energy 18, ceiling 40}); loss on BOTH half-spaces **18/78 → 42/78** ({energy 18, ceiling 24}); loss on the EXIT half-space unchanged at 70/78. Headline row: `grate`, `m` = 1, `Rbig` = 2 lambda, `Im/Re` = 1e-6 on `layers[0]` only — nodal `max(R+T)` = **8.098737**, div-conforming twin **4.7364e-07**, ceiling excess **+2.1777e-05**; `1ac6de7e` returns it, round 3 refuses it |
| **2b** | **0 refusals and 0 warnings on 52** screened-healthy rows | **CONFIRMED, on 180 — and the population is NARROWER than the claim reads** | 96 candidates screened at zero loss on all three conjuncts (channel set matches the staggered twin, closure inside the 1e-6 WARN edge, ceiling silent) → 10 healthy geometries x 3 loss placements x 6 rungs = **180 rows, 0 refused, 0 warned**. But all ten survivors are UNIFORM middle layers at `Rbig` <= 1 vacuum wavelength: **no structured family and nothing at `Rbig` >= 2 lambda passes the screen at all**, on this grid or the fix round's — see §3.2 and D-V2 |
| **2c** | lossy-half-space census: set-right worst `Re qn - n_max` = -1.999e-03, staggered -1.71e-04, set-wrong mildest +7.11e-06, **0 false positives on 600** | **CONFIRMED to the digit on two independent censuses** | 320 cascade rows / 640 solves: nodal set-right worst **-7.5949e-03**, staggered worst **-1.75045e-04** (theirs -1.712581e-04), set-wrong mildest **+2.1629e-05**, **0 ceiling false positives on the 60 set-right rows**. A separate 864-row half-space census reads mildest positive **+7.5520e-06** (theirs +7.108677e-06) and closest undamaged approach **-2.2346e-05** |
| **2d** | the DERIVED term `Re qn <= n + imag_bar^2/(2n)` = 1.25e-09/n is 5.1 decades from anything measured and **untested by any fixture** | **CONFIRMED, and now settled: it is CORRECT, USED WHERE DERIVED, and VACUOUS ON THIS CODE PATH** | (i) the decision boundary IS the reported slack to ten digits, and the lossy boundary is **2.67x** the lossless one at `n` = 1.5 (1.3333e-09 vs 5.0e-10) — the term is used exactly where the derivation puts it; (ii) over **19,542** exact `(eps, Im/Re, gamma)` combinations the excess of a UNIFORM passive half-space is **at most 0.0, attained at gamma = 0, with zero positive rows at any loss up to `Im/Re` = 1** — the derived widening is never needed there; (iii) **0 of 696** rows of a bisecting physical hunt land inside the window, because the excess crosses zero by a JUMP (the channel SET changes), not continuously. See §4 |
| **2e** | gain disarms everything | **CONFIRMED** | 120 gain rows (5 placements x 6 rungs x 4 geometries, `Im/Re` -1e-14 .. -1e-3): **0 refused, 0 warned on BOTH trees**, while **106 of the 120** carry a quantity a disarmed detector would have fired on |
| **2f** | the switch restores the pre-round behaviour | **CONFIRMED, bit for bit** | `BOR_NODAL_PASSIVITY_GUARD = False` on 72 rows spanning the ladder: 0 raised, 0 warned, and **72 of 72 answer hashes identical to `1ac6de7e`** |
| **3** | a non-finite `q_excess` FORMED from a real spectrum is HOT; a never-formed one stays cold; the axis liner reads `warn_own` on all kernels | **CONFIRMED on 8 arms — but the COST of the old reading is mis-rated, see D-V3** | my liner geometry drives `q_excess` non-finite on 1–2 rungs per arm and **none of them reads `ok`** on any of the 8 arms (2 builds x {Haswell t1, Haswell t4, Sandybridge t1, Katmai t1}). Constructed directly through `measure_layer`: never-formed (no modes / zero ceiling / zero `k0`) → `ok` 3/3; formed `inf` and formed `nan` → `warn_own` 2/2. The change also reaches the `refuse` arm: a MANUFACTURED union sliver on a blown-up spectrum **returned an answer on `1ac6de7e` and raises `BORSemMeshError` on round 3**, so the cost of the old reading was not "a missing `UserWarning`, never a wrong returned number" |
| **4** | `_BOR_FRAC_DEADBAND` is symmetric and the axis / middle / outer liner decide identically — **0 of 9 rungs disagree** | **CONFIRMED on one geometry, REFUTED as a general claim — see D-V1** | geometry A (`Rbig` = 17, interior wall 7.0, reproduced width 80,088 ULP BELOW the bar): **3 of 9 rungs disagree on `1ac6de7e` → 0 of 9 on round 3**, on all 8 arms. Geometry B (`Rbig` = 12.5, wall 5.0, reproduced width **156,767 ULP ABOVE**): 2 of 9 → **1 of 9**, and the rung that survives is **the edge itself**, where `1ac6de7e` reads `ok`/`ok`/`ok` (agreeing) and round 3 reads **`warn_own`/`ok`/`ok`** — so round 3 fixed geometry B's two GAP-3 rungs and INTRODUCED a split at its edge. **63 of 169** `(Rbig, wall)` combinations reproduce the width above the bar |
| **5a** | the near-cutoff channel count is ONE number and is `idx + 1` on 12 of 12 | **CONFIRMED on an independent fixture, on 3 arms** | `Rbig` = 19 / `N` = 96 / `n` = 1.63, 12 ladders x 13 rungs = 156 solves: count is one number and equals `idx + 1` on **12 of 12** on Windows/Haswell, Windows/Sandybridge and WSL/Haswell |
| **5b** | the populations split at **100x** `_BOR_CHANNEL_REAL_FLOOR`: 6.89e-07 (108 rows) vs 1.787e-04 (48 rows), knee 79x | **CONFIRMED in shape; the KNEE is a reading, and its headroom is thinner here** | my fixture: shallow envelope **4.7078e-07** (108 rows) vs deep **2.8880e-05** (48 rows), **1.79 decades** apart (theirs 2.41). The highest rung that exceeds 1e-6 sits at **91.66x** the floor here against their 79x, so `_CUTOFF_ENERGY_FLOOR_MULT = 100` has **1.09x of headroom** on an independently written fixture. Across three arms the shallow envelope moves 6.59e-08 .. 4.71e-07 (7.1x) and the knee 51.55x .. 91.66x — always below 100 |
| **5c** | the residual is on the marginal channel's own `R + T` row, NOT the orientation band (0 in-band backward-flux modes) | **CONFIRMED** | 24 deep rungs: **0 in-band backward-flux modes** in either half-space on both builds, against 2,244 (Windows) / 2,225 (WSL) backward modes overall; the worst `R + T` row IS the marginal channel on **17 of 24** (Windows) and **15 of 24** (WSL) |
| **5d** | `_CUTOFF_LADDER_BAR = 1e-5` carries 14.5x / 12.2x two-sided; `_CUTOFF_DEEP_BAR = 2e-3` carries 11.2x one-sided | **CONFIRMED, with wider margins here** | my fixture: **21.2x** over the shallow envelope (61.5x and 151.7x on the other two arms) and **69.3x** over the deep envelope (256.4x, 45.6x). Both bars clear their measured envelope on BOTH fixtures and all three arms. On whether `_CUTOFF_DEEP_BAR` is a bar or a reading, see §5.3 |
| **1** | the vacuous disarm gate is retired and the live gate is proved by mutation | **CONFIRMED, mutation re-run independently** | `v8_mutation.py`: the replacement gate reads `1 passed` intact and `1 failed` with `_stack_is_provably_passive` patched to `_stack_media_are_passive` at runtime (the conjunct removed), so it is LIVE. And the retired gate's own premise is measured directly: at its `Im/Re` = 1e-3 the incidence half-space returns **0 channels**, so the predicate was never reached — vacuous, as round 2 said; at 1e-5 and below the same geometry returns 12 |
| **id** | bit identity: only newly-refused rows change | **CONFIRMED on a battery built to be the population** | 66 fixtures (54 BOR, 12 EME), **32 of the BOR rows carrying a loss on the incidence half-space or on both**. **Three arms** (Windows / Haswell, Windows / Katmai, WSL py3.12 / Haswell, all
t1): **50 identical, 16 moved, all 16 `HASH -> BORNodalPassivityError`, 0
illegal — the same 16 names on every one**. Every mover is a legacy-nodal row with the loss on a half-space. Cross-build the DECISION is identical on 66 of 66 while the bytes differ on 33 — the expected LAPACK spread, and proof the decision is build-free |

---

## 2. The arm table

Every command carried `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`MKL_NUM_THREADS` on the command line; every probe read the **loaded** kernel
back out of `threadpoolctl` and `lumenairy.__file__` out of the running
interpreter into its JSON, and `PYTHONPATH` pinned the tree.

> **A trap worth recording, because it would have invalidated every probe.**
> This host has an EDITABLE install of `lumenairy` at
> `D:\Metacept\...\Lumenairy` on `sys.path`. Run as `python script.py` the
> script's own directory is `sys.path[0]` and the working directory is NOT on
> the path, so **`import lumenairy` resolves to the D: tree, not the
> worktree** — silently, with a module that has 20 names where the worktree's
> has 37. `python -c` does not show it (the cwd is `sys.path[0]` there). Every
> probe, test run and pytest arm in this document therefore sets `PYTHONPATH`
> explicitly and prints `lumenairy.__file__`, and `_vb3.require_tree()` turns
> the check into a hard `SystemExit` when `LUM_EXPECT_ROOT` is set (the arm
> scripts set it). Verified as a negative control: run without `PYTHONPATH`,
> `v9_tables.py` refuses with *"WRONG TREE: lumenairy resolved to
> `D:\Metacept\...`"* instead of silently measuring it.

| build | python / numpy / scipy | requested | **loaded** | threads | what ran on it |
|---|---|---|---|---|---|
| Windows | 3.14.6 / 2.4.4 / 1.17.1 | HASWELL | Haswell | 1 | everything, on BOTH trees |
| Windows | 3.14.6 / 2.4.4 / 1.17.1 | HASWELL | Haswell | 4 | v3 liner, tests |
| Windows | 3.14.6 / 2.4.4 / 1.17.1 | SANDYBRIDGE | Sandybridge | 1 | v1 ladder, v3 liner (both trees), v4 cutoff, tests, 21-file set |
| Windows | 3.14.6 / 2.4.4 / 1.17.1 | KATMAI | Katmai | 1 | v1 ladder, v3 liner (both trees), v6 identity (both trees), tests, 21-file set |
| WSL | 3.12.3 / 2.4.6 / 1.17.1 | HASWELL | Haswell | 1 | v1 ladder, v3 liner (both trees), v4 cutoff, v6 identity (both trees), tests, 21-file set |
| WSL | 3.12.3 / 2.4.6 / 1.17.1 | HASWELL | Haswell | 4 | v3 liner, tests |
| WSL | 3.12.3 / 2.4.6 / 1.17.1 | SANDYBRIDGE | Sandybridge | 1 | v3 liner (both trees), tests, 21-file set |
| WSL | 3.12.3 / 2.4.6 / 1.17.1 | KATMAI | Katmai | 1 | v3 liner (both trees), tests, 21-file set |

`ZEN` aliases Haswell and `SKYLAKEX` dies on this host (Ryzen 9 5950X, Zen 3),
so the CI pool's EPYC 9V74 / 7763 kernels are not reproducible here — the same
limit the fix round records.

---

## 3. GAP 2 — the split, re-measured

### 3.1 The decision table

Six geometries (four middle-layer families, `m` 0..3, `N` 120 and 200,
`Rbig` from 1 to 16 vacuum wavelengths) x three loss placements x a 13-rung
`Im/Re` ladder from 0 to 1e-1, each solved in BOTH bases with the guard
disarmed and then once armed — **234 nodal rows and 234 staggered twins**
(`v1_gap2.py ladder`, both trees, Windows / Haswell / 1 thread).

| loss on | `1ac6de7e` refused | round 3 refused | round 3 by detector |
|---|---|---|---|
| `layers[0]` alone | 18 / 78 | **58 / 78** | energy 18, **ceiling 40** |
| the EXIT half-space | 70 / 78 | 70 / 78 | energy 62, ceiling 8 |
| BOTH half-spaces | 18 / 78 | **42 / 78** | energy 18, **ceiling 24** |

**64 rows move, every one of them `returned -> REFUSED (ceiling)`**, and the
**guard-disarmed answers are bit-identical on all 234**. The EXIT column does
not move because `layers[0]` is lossless there, so both detectors were already
armed on `1ac6de7e`.

**The whole table is build-free**: Windows / Katmai / 1, Windows /
Sandybridge / 1 and WSL py3.12 / Haswell / 1 all read 58 / 70 / 42 with the
same `{energy 18, ceiling 40}`, `{62, 8}` and `{18, 24}` detector split, digit
for digit — four arms, one decision table
(`v9_tables.py`, `v9_tables_win_Haswell_t1.json`).

### 3.2 The counter-population

"Healthy" is screened on all three conjuncts at zero loss — the channel counts
match the staggered twin, the closure is inside the energy screen's own WARN
edge (1e-6), and the ceiling is silent. Of 96 candidates, **10 qualify**;
running them over 3 loss placements x 6 rungs gives **180 rows, 0 refused and
0 warned** (`v2_ceiling.py fp`). The round asked for 40 and measured 52.

Screening on the channel set ALONE is not enough, and this verification hit the
same wall the fix round records: `grate`, `m` = 1, `Rbig` = 0.5 lambda closes
its channel set exactly (2/2 against the twin's 2/2) and is nonetheless refused
by the ENERGY detector at `Im/Re` = 1e-12 — a genuine deficit on a stack the
predicate calls lossless, which is defect D2's own population, not a false
positive.

**And the counter-population is NARROW, which is worth stating plainly.** All
ten survivors are UNIFORM middle layers at `Rbig` <= 1 vacuum wavelength
(`unif35` and `unif23` at `m` 0/1/3, `Rbig/lambda` 0.5 and 1, closing between
3.8e-15 and 1.2e-07). **No structured family and no geometry at `Rbig` >= 2
lambda survives the screen at zero loss**, on this grid or the fix round's —
because at 2 lambda the nodal half-space already returns a channel above its
own index ceiling (§3.3). So "0 false positives" is a true statement about a
population the screen itself defines, not a statement that the ceiling is
harmless on the geometries a caller is likely to bring. D-V2 is the other half
of that sentence.

### 3.3 What the index ceiling actually reads — a scope statement, measured

`solve` passes the ceiling `(layers[0], layers[0]["q"][inc] / k0)` and
`(layers[-1], layers[-1]["q"][out] / k0)`, and `inc` / `out` come from
`_physical_propagating` applied to **that half-space alone**. Both arguments
are therefore functions of one half-space's `m`, `N`, `Rbig`, `k0` and `eps` —
**and of nothing in the cascade**. Two consequences, both measured:

* an 864-row half-space census (`v2_ceiling.py halfspace`: `m` 0..3 x `N`
  80/120/200 x `Rbig/lambda` 0.5..16 x `eps` 2.25/4.0 x `Im/Re` 0..1e-1)
  decides 260 firings **without a single cascade**, and
  **`0` rows where the loss ITSELF flips the decision** — the lossless twin of
  every firing lossy row fires too;
* a stack whose middle layer is the SAME uniform medium as its half-spaces
  scatters nothing (the exact answer is `R` = 0, `T` = 1 per channel, and the
  nodal cascade returns it to **6.0e-15** at zero loss) and is **refused on 90
  of 180 such rows**, of which **60 are refused only because round 3 armed the
  ceiling on an absorbing half-space**.

That is not a false positive under the round's own definition — the returned
channel LIST does contain a mode above the medium's index, which is what the
detector is for — but it is a SCOPE the round does not state: **the ceiling's
refusal is a statement about the half-space's mode list, not about the answer
the cascade returned**, and round 3 extends it to a population (`Rbig` >~ 2
lambda with any absorbing half-space) that previously received a number. See
D-V2.

---

## 4. GAP 2's derived term — tight, or vacuous?

The fix round lists this under what it could not measure. It is settled here in
three steps (`v5_derived.py`).

**(a) It is used exactly where the derivation puts it.** Feeding
`_check_nodal_passivity` a channel array placed by hand at `n_max + delta` and
bisecting `delta`, the decision boundary IS the reported slack, to ten
significant figures, on all five media tested:

| half-space | `n_max` | lossless? | slack reported | measured boundary | derived term |
|---|---|---|---|---|---|
| `eps` = 2.25 | 1.5 | yes | 5.000000e-10 | 4.999999303e-10 | — |
| `eps` = 2.25, `Im/Re` 3e-12 | 1.5 | no | 1.333333e-09 | 1.333333333e-09 | 8.3333e-10 |
| `eps` = 2.25, `Im/Re` 1e-6 | 1.5 | no | 1.333333e-09 | 1.333333333e-09 | 8.3333e-10 |
| `eps` = 1.0, `Im/Re` 3e-12 | 1.0 | no | 1.750000e-09 | 1.750000034e-09 | 1.2500e-09 |
| `eps` = 36.0, `Im/Re` 3e-12 | 6.0 | no | 7.083334e-10 | 7.083333919e-10 | 2.0833e-10 |

**(b) It is not needed on the population this code path produces.** For a
UNIFORM half-space the transverse eigenvalue is set by the PEC wall and `m`
alone — it is a Bessel zero over `Rbig`, real and independent of `eps` — so the
exact spectrum is `qn^2 = eps - (gamma/k0)^2` and the excess is closed-form.
Evaluated over **19,542** combinations (`eps` in {1, 1.5, 2.25, 4, 6, 12, 30} x
`Im/Re` in {0 .. 1.0} x 401 values of `gamma/k0`, keeping only the modes the
channel gate keeps), the excess `Re qn - Re sqrt(eps)` has maximum **exactly
0.0**, attained at `gamma = 0`, and **zero rows are positive at any loss**. The
`Re(q^2) <= max(Re eps) k0^2` step is tight there, but the step from `Re(q^2)`
to `Re q` — which is where `imag_bar^2 / (2 n)` comes from — throws away the
relation between `Re qn` and `Im qn` that a uniform medium supplies. The
library's own comment records that the homogeneous super/substrate are "the
only layers `_physical_propagating` ever classifies", so the widening can only
bind on a RADIALLY GRADED absorbing half-space — which `BORStack` cannot build
and which `build_layer`'s own documentation says does not arise on this path,
though a caller passing an `eps_profile` straight to `build_layer` could make
one.

**(c) No physical row can reach it, and the reason is structural.** A bisecting
hunt over `m` 0..3 x `N` 80/120/200 x `Im/Re` {3e-12, 1e-9} x 29 values of
`Rbig` in 0.4..6 lambda, bisected wherever the sign changed (**696 rows**),
puts **0 rows inside the window** `(5e-10, 1.3333e-09]`. The excess does not
approach zero continuously: it crosses by a JUMP, because what changes at the
crossing is the channel SET. The closest approach from below over the whole
verification is **-2.2346e-05** (half-space census) and the mildest positive is
**+7.5520e-06** — the two populations sit 4.2 and 3.8 decades from the widened
slack respectively.

**Conclusion.** The derivation is right, the implementation matches it to ten
digits, and the term is **safe but vacuous**: nothing this code path can build
reaches within four decades of it on either side. It is not calibrated to any
measurement, which is the property that matters, and its only measurable cost
is that the ceiling tolerates a **2.67x** larger excess on an absorbing
half-space at `n` = 1.5 — against a mildest violation four decades above.

---

## 5. GAPS 3, 4 and 5

### 5.1 GAP 3 — confirmed, on eight arms

My liner geometry (`Rbig` = 17, `m` = 2, `N` = 96, degree 6, `k0` = 3.5) drives
`q_excess` non-finite on the narrow rungs of the ladder. On all eight arms,
**every non-finite row reads `warn_own`, none reads `ok`**. The READING still
moves with the kernel — on geometry A, Katmai produces ONE non-finite rung
where Haswell and Sandybridge produce two, on both builds — and the DECISION
does not, which is exactly the property `docs/TESTING_STANDARDS.md` asks for.
On `1ac6de7e` the same rows read `ok` (2 of 2 on each geometry, every arm), so
the change is the one the round claims.

**And my fixture holds the premise where the round's does not.** The fix
round's own GAP-3 gate SKIPS on Sandybridge, because at `Rbig` = 24 / `m` = 1 /
degree 8 that kernel returns a FINITE `q_excess` (8.895e+07) — the premise gate
working, and measured again here. This verification's liner
(`Rbig` = 17 / `m` = 2 / `N` = 96 / degree 6 / `k0` = 3.5) drives the ratio
non-finite on **Sandybridge as well**, on both trees and both builds, so the
`ok`-to-`warn_own` flip is observed on all four kernels rather than three.

The `q_measurable` distinction was constructed directly rather than hoped for
(`v3_liner.py formed`): a record with no modes, a zero index ceiling or a zero
`k0` reads `ok` (3 of 3); a record whose ratio was formed and came back `inf`
or `nan` reads `warn_own` (2 of 2); a formed finite ratio below the bar reads
`ok`.

The change is NOT confined to `warn_own`: it reaches the `refuse` arm too, and
there it converts a returned answer into a `BORSemMeshError` — see **D-V3**.

> **One observation, not a defect.** `verdict` reads
> `rec.get("q_measurable", False)`, so a record that does NOT carry the key —
> an older or externally built payload — reverts to the pre-round-3 benign
> reading. Every in-tree caller passes a record straight from `measure_layer`,
> which always sets it, so nothing in the library is affected; it is recorded
> because `verdict`'s docstring advertises it as a pure function of the record.

### 5.2 GAP 4 — confirmed on one geometry, refuted as a general claim

See **D-V1** in §6. The short form: `_BOR_FRAC_DEADBAND` is 16 ULP **of the
bar** (16.78 ULP measured), and the quantity it has to absorb is the mesh's
reproduction of the requested width, whose error is sized by `Rbig / w` and
reaches **156,767 ULP**. Whether it lands above or below the bar is a property
of the wall coordinates: **63 of 169** `(Rbig, wall)` combinations land above,
and 9 of the 13 `Rbig` values tested contain positions that split.

### 5.3 GAP 5 — confirmed, and the knee is a reading

The durable claims — the channel count is ONE number and it is `idx + 1` —
hold on **12 of 12 ladders on all three arms** of a fixture that shares no
numbers with the gate's own. The population split holds too: nothing at or
above 100x the channel floor exceeds 1e-6 on any arm.

| arm | shallow envelope (108 rows) | deep envelope (48 rows) | 1e-5 / shallow | 2e-3 / deep | highest offending rung |
|---|---|---|---|---|---|
| Windows / Haswell / 1 | 4.70775e-07 | 2.88804e-05 | **21.2x** | **69.3x** | 91.66x the floor |
| Windows / Sandybridge / 1 | 1.62475e-07 | 7.79966e-06 | 61.5x | 256.4x | 51.55x |
| WSL / Haswell / 1 | 6.58987e-08 | 4.38251e-05 | 151.7x | 45.6x | 51.55x |

**Is `_CUTOFF_DEEP_BAR` a bar or a reading?** It is a **bound, one-sided, and
honestly declared as such** — and this verification adds the reason it is
sound rather than merely conservative. The fix round argues that the pre-fix
band's failure shows in the COUNT on the deep rungs, which is asserted
unconditionally; I can add that the deep envelope moves by **5.6x across three
arms** (7.80e-06 .. 4.38e-05) and the bound still sits **45.6x — 1.66 decades
— above the worst of them**, so it clears the cross-arm spread of the quantity
it reads by more than a decade. That is a bar by
`docs/TESTING_STANDARDS.md` rule 5 on the noise side;
what it does not have, and cannot have without re-implementing the pre-fix
band, is a measured signal side. **`_CUTOFF_ENERGY_FLOOR_MULT = 100` is the
weaker of the two constants**: it is a reading (79x on their fixture, 91.66x on
mine, 51.55x on two arms), it has 1.09x of headroom at worst, and if a third
fixture put an offending rung above 100x the consequence would be a shallow
envelope around 1.3e-06 — still 7.5x inside the 1e-5 bar, so the GATE would
hold. Recorded as a margin note, not a defect.

**The mechanism** reproduces: over 24 deep rungs there are **0 backward-flux
modes inside the classifier band** in either half-space (against 2,244 backward
modes overall), and the worst `R + T` row IS the marginal channel on 17 of 24
(Windows) and 15 of 24 (WSL) — the same majority-not-invariant shape the fix
round reports (17 of 28). I could not corroborate their in-band `|flux|` spread
(5.5399e-08 against 4.5564e-03): `zcascade.layer_modes(staggered=True)` returns
columns already flux-normalised, so the flux my census reads is 1.0 to 1e-15 by
construction and the raw quantity is not recoverable from the same call.

---

## 6. Defects

### D-V1 (P3, library) — `_BOR_FRAC_DEADBAND` is four decades narrower than the error it has to absorb, and round 3 moved the disagreement rather than removing it

**What.** `verdict`'s three width arms compare `w_min_own_frac` — a difference
of two MESH BREAKPOINTS divided by `Rbig` — against exact decimal literals.
Round 3 routes them through `_below(x, bar)`, which widens the bar by
`16 * eps` RELATIVE, i.e. 16.78 ULP of the 1e-6 bar. The error that comparison
has to absorb is the mesh's reproduction of the requested width: for an
interior liner `fl(r + w) - r`, for an outer one `Rbig - fl(Rbig - w)`, whose
error is sized by `Rbig / w` and reaches **156,767 ULP** — four decades
outside the deadband — and which can land on **either** side of the bar.

**Measured**, pure IEEE-754 arithmetic over 13 `Rbig` values x 11 interior wall
positions plus the outer wall (`v3_liner.py repr`, identical on all 8 arms):
**63 of 169 combinations reproduce the edge width ABOVE the bar**, and `_below`
answers `False` on every one. Nine of the thirteen `Rbig` values contain
positions that split.

**The decision, on 8 arms** (`v3_liner.py ladder`, geometry B: `Rbig` = 12.5,
interior wall 5.0, liner exactly `_BOR_MIN_ELEM_FRAC * Rbig` wide):

| tree | axis | middle | outer | agree? | arms |
|---|---|---|---|---|---|
| `1ac6de7e` | `ok` | `ok` | `ok` | **yes** | 6 (2 builds x Haswell / Sandybridge / Katmai, t1) |
| round 3 | **`warn_own`** | `ok` | `ok` | **no** | 8 (the same six plus Haswell t4 on each build) |

One reading per tree on every arm — the split is arithmetic, not backward
error.

Over the whole 9-rung ladder the two geometries read:

| geometry | reproduced edge width | `1ac6de7e` | round 3 |
|---|---|---|---|
| A, `Rbig` = 17, wall 7.0 | 80,088 ULP **below** the bar | 3 of 9 rungs disagree | **0 of 9** |
| B, `Rbig` = 12.5, wall 5.0 | 156,767 ULP **above** the bar | 2 of 9 rungs disagree | **1 of 9** — the edge |

On both geometries round 3 closes the two GAP-3 rungs (the non-finite
`q_excess` rows, which read `ok` at the axis on `1ac6de7e` and `warn_own` on
round 3 — 2 of 2 on each geometry, on all 8 arms). On geometry A it also closes
the edge; on geometry B it OPENS it. So round 3 fixes the case it measured and
introduces the mirror case it did not. The round's claim — *"the same physical
liner decides identically at the axis, in the interior and at the outer wall"*
— is a property of ONE geometry stated as a family property, which is the same
shape as the D9/GAP 5 finding round 3 was itself fixing.

**Severity P3**: the consequence is a `UserWarning` present or absent, never a
wrong returned number — the same bound GAP 4 carried.

**Reproducer** —
`tests/unit/test_verify_bor_guards_round3.py::test_the_liner_verdict_agrees_at_all_three_positions_at_a_second_rbig`
(`xfail(strict=True)`, premise-gated on all three positions being spectrally
hot), with its arithmetic premise asserted unconditionally by
`test_the_width_deadband_does_not_reach_the_meshs_reproduction_error`.

**Suggested remedy** (not implemented here): size the deadband from the record
rather than from the bar. The reproduction error of a width `w` against a wall
at `r` is about `np.spacing(r) / w` relative, which `measure_layer` already has
in scope; a deadband of a few of those covers both directions and still sits
decades inside the nearest rung any ladder asks for (the neighbouring rungs are
0.3x and 3x the bar). Alternatively carry the CALLER's requested width through
the record instead of re-deriving it from the mesh.

### D-V2 (P3, scope / documentation) — the ceiling's refusal is a statement about the half-space's mode list, and round 3 newly applies it to a large returning population

**What.** Because both of the ceiling's arguments are functions of one
half-space alone (§3.3), a refusal says nothing about whether the cascade's
answer is right. Measured: a stack that scatters nothing at all — middle layer
identical to the half-spaces — is refused on **90 of 180** rows while returning
`R + T` whose worst closure over the refused rows is **4.31e-14** at zero loss
and **4.70e-08** at `Im/Re` = 1e-9, and **60 of those 90 are refused only
because round 3 armed the ceiling on an absorbing half-space**.  The smallest
`Rbig` anywhere in the refused trivial population is **2.0 vacuum
wavelengths**; nothing at 0.5 or 1 is refused. Across the 864-row half-space census, **130 lossy rows that
`1ac6de7e` returned now refuse**.

**Why it is not a false positive.** The returned channel list genuinely
contains a mode above the medium's own index, which is the contradiction the
detector exists to report, and the equivalent refusal at zero loss is
pre-existing round-2 behaviour. The energy is not the thing being screened.

**Why it is still worth filing.** It is a compatibility surface the round's
documentation does not state: on the legacy nodal basis, **any** stack with an
absorbing half-space and `Rbig` above roughly 2 vacuum wavelengths is now
likely to raise where it returned, regardless of how accurate its `R` and `T`
are. The escape hatch is real and was verified bit-for-bit (§1, 2f), and the
nodal basis is the deprecated legacy one, so this is a note rather than a
blocker.

**Reproducer**: `validation/probe_verify_bor_round3/v2_ceiling.py fp`, the
`trivial` population; `v2_fp_win_Haswell_t1.json`, `summary.trivial_refused` =
90 and `summary.trivial_refused_lossy` = 60.

**Suggested remedy**: one sentence in `_check_nodal_passivity`'s docstring and
in the CHANGELOG saying that the ceiling reads the half-space's own mode list
and is therefore independent of the cascade, plus the `Rbig`/loss shape of the
newly refusing population.

### D-V3 (P3, rating / documentation) — GAP 3's change also converts a RETURNED answer into a RAISE, and no identity battery contains that population

**What.** `verdict`'s first arm is
`_below(w_min_union_frac, _BOR_MIN_ELEM_FRAC) and hot`, so GAP 3's change to
`hot` reaches `refuse`, not only `warn_own`. With `BOR_SEM_MESH_GUARD` at its
default `True`, a `refuse` is a `BORSemMeshError`. So on the population where a
MANUFACTURED union cell — one neither layer's own segment list asked for — is
narrower than the bar AND the ratio came back non-finite, a caller who
previously received a number now receives an exception.

**Measured** (`v10_sliver.py`, two layers asking for axis walls a ratio apart
so the union mesh manufactures the cell; Windows / Haswell / 1 thread):

| layer-0 wall | layer-1 wall | `w_min_union_frac` | `q_excess` | `1ac6de7e` | round 3 |
|---|---|---|---|---|---|
| 1e-9 x `Rbig` | 2e-9 x `Rbig` | 1e-09 | **inf** | **returned** (`warn_manufactured`) | **`BORSemMeshError`** |
| 2e-9 x `Rbig` | 3e-9 x `Rbig` | 1e-09 | **inf** | **returned** (`warn_manufactured`) | **`BORSemMeshError`** |
| 1e-8 .. 1e-5 x `Rbig` | +10 % | 1e-09 .. 1e-06 | 1.87e+08 .. 1.87e+05 | `BORSemMeshError` | `BORSemMeshError` |

Exactly the two non-finite rows move, and they move in the direction that is
RIGHT: a manufactured sliver on a spectrum that has blown up is precisely what
`refuse` exists for, and returning a number built on it was the defect.

**Why it is filed.** Two things, both about the record rather than the code:

1. the round (and the round-2 verification before it) rates the cost of the old
   reading as *"bounded — a missing `UserWarning`, never a wrong returned
   number — which is why the verification filed it P2"*. It was not bounded to
   a warning: on this population it was a **returned answer where the contract
   had to refuse**, which is strictly more serious than the rating claims;
2. it is an API behaviour change (a new raise) that **neither round 3's
   65-fixture battery nor this verification's 66-fixture one contains** — the
   same "the battery is not the population" shape round 3 itself called out
   about round 2. It is now pinned by
   `tests/unit/test_verify_bor_guards_round3.py::test_a_manufactured_sliver_on_a_blown_up_spectrum_is_refused_not_returned`,
   which is the only gate anywhere asserting it.

**Severity P3**: the new behaviour is correct and desirable; what needs
changing is one sentence of the rating and one line of the CHANGELOG.

### Not defects, recorded

* **`verdict` on a record without `q_measurable`** reverts to the pre-round-3
  reading (§5.1). No in-tree caller can produce such a record.
* **The energy screen is disarmed by an absorbing INCIDENCE medium but not by
  an absorbing EXIT medium**, although `T` is formed from the exit medium's
  modes and is no more a power fraction there than `R` is on the incidence
  side. Measured: with the loss on the exit half-space alone the energy
  detector stays armed and refuses 62 of 78 ladder rows, identically on both
  trees — so this is inherited, not introduced, and no healthy row in the
  180-row counter-population is affected by it. Worth a look in a later round.

---

## 7. The runs

### 7.1 The 21-file BOR/EME set

`validation/probe_verify_bor_round3/runs/files.txt` (the fix round's own set
definition), `-q -p no:randomly`, every arm printing `lumenairy.__file__`, the
interpreter and the LOADED kernel before pytest starts.

| # | build | requested | **loaded** | thr | passed | skipped | failed | errors | time |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Windows py3.14 | HASWELL | Haswell | 1 | **403** | 0 | 0 | 0 | 1466.08 s |
| 2 | Windows py3.14 | KATMAI | Katmai | 1 | **403** | 0 | 0 | 0 | 2022.84 s |
| 3 | Windows py3.14 | SANDYBRIDGE | Sandybridge | 1 | **402** | 1 | 0 | 0 | 1856.35 s |
| 4 | WSL py3.12 | HASWELL | Haswell | 1 | **403** | 0 | 0 | 0 | 1373.30 s |
| 5 | WSL py3.12 | KATMAI | Katmai | 1 | **403** | 0 | 0 | 0 | 1966.49 s |
| 6 | WSL py3.12 | SANDYBRIDGE | Sandybridge | 1 | **402** | 1 | 0 | 0 | 1754.87 s |

**Zero failures and zero errors on every one of the six.** Every tail carries a
real summary line and no arm reports `no tests ran` — which had to be checked,
because the first launch of this matrix DID report it: `files.txt` was copied
out of the fix round's directory with CRLF line endings and every path reached
pytest with a trailing carriage return, so all six arms exited 4 with
`ERROR: file or directory not found` and `no tests ran in 0.00s`. Grepping the
tail caught it; a green-looking driver log would not have.

**403 on the four kernels that drive the ladder non-finite, 402 + 1 skip on
both Sandybridge arms** — and the skip is GAP 3's own premise gate, carrying
its reading:

    SKIPPED tests/unit/test_verify_bor_guards_round2.py:261: premise absent on
    this arm: the axis liner's q_excess is finite (8.895e+07), so the
    non-finite branch is not exercised

That is the premise gate working, and an independent cross-check of the fix
round at the same time: **8.895e+07 on both builds**, against the round-2
verification's 8.89487e+07 and the fix round's 8.895e+07 — four digits, a
round later, on a separate tree. The counts match the fix round's own
403 / 402 + 1 exactly.

### 7.2 The census / walker / dispatcher-pin sweep

    python -m pytest $(ls tests/unit | grep -iE 'walker|census|dispatcher_pin|
                       public_api|doc_consistency' | sed 's#^#tests/unit/#')
                     -q -p no:randomly -rs

**1288 passed, 12 skipped** (Windows py3.14 / Haswell / 1 thread) — the same
reading the fix round records. Run TWICE: before this verification's files
existed (146.58 s) and again with the new audit document and test file in the
tree (133.78 s), **1288 / 12 both times**, so neither file disturbs a walker.
All 12 skips carry their reason and all 12 are the CHANGELOG walkers declining
to enforce on the `## [5.45.0]` block (no audit closures, no self-citation, no
line-count claim declared there); this verification adds no version header.

### 7.3 ruff

    wsl -e bash -lc 'cd /mnt/c/tmp/lum_vbor3 && ~/lumvenv/bin/ruff check
                     lumenairy/ tests/ validation/probe_verify_bor_round3/'
    -> All checks passed!

### 7.4 `.test_durations`

Spliced by `validation/probe_verify_bor_round3/v7_durations.py`, which runs the
new file with `--durations=0 -vv` (deliberately without `-q`, which drops
sub-5 ms keys), removes any key belonging to it, re-adds the measured set,
re-sorts and re-validates.

| | |
|---|---|
| entries before / after | 12,894 / **12,899** |
| keys dropped / added | 0 / **5** |
| keys per collected test | 5 of 5, none defaulted |
| sorted, JSON-valid after the splice | **yes** (asserted in the probe) |
| slowest key added | `test_the_near_cutoff_populations_separate_on_a_second_fixture[1-1]`, **3.61 s** |
| keys over the 60 s shard cap | **0** |
| working-tree diff | **5 added lines** (one-space indent, no trailing newline, matching the shipped format) |

### 7.5 The new test file, on eight arms

`tests/unit/test_verify_bor_guards_round3.py` — 5 tests, **4 passed and 1
xfailed** on every arm:

| build | requested kernel | threads | result | wall clock |
|---|---|---|---|---|
| Windows py3.14 | HASWELL | 1 | 4 passed, 1 xfailed | 10.78 s |
| Windows py3.14 | HASWELL | 4 | 4 passed, 1 xfailed | 11.36 s |
| Windows py3.14 | SANDYBRIDGE | 1 | 4 passed, 1 xfailed | 18.08 s |
| Windows py3.14 | KATMAI | 1 | 4 passed, 1 xfailed | 20.81 s |
| WSL py3.12 | HASWELL | 1 | 4 passed, 1 xfailed | 9.90 s |
| WSL py3.12 | HASWELL | 4 | 4 passed, 1 xfailed | 8.24 s |
| WSL py3.12 | SANDYBRIDGE | 1 | 4 passed, 1 xfailed | 17.19 s |
| WSL py3.12 | KATMAI | 1 | 4 passed, 1 xfailed | 20.12 s |

The xfail is D-V1 on every arm; no arm XPASSes and no arm skips.

---

## 8. Durability of the round-3 tests

Read against `docs/TESTING_STANDARDS.md` rather than re-run for greenness.

| gate | origin of its numbers | two-sided? | premise-gated? | assessment |
|---|---|---|---|---|
| `test_the_energy_screen_disarms_on_an_absorbing_incidence_medium_but_the_ceiling_does_not` | derived at runtime (the predicate's own answers) plus a premise assertion that the channel set survives | the two predicates are asserted in OPPOSITE senses, which is what the mutation flips | yes, on the arm's cascade actually returning an over-ceiling channel | **sound.** The mutation proof is the part that matters and it reproduces: `v8_mutation.py` reads `1 passed` intact and `1 failed` mutated |
| `test_a_negligible_loss_on_the_incidence_medium_must_not_disarm_the_screen` | the div-conforming TWIN's own excess, measured at runtime; the bar is "three decades above the twin" | yes — the twin supplies the noise side | yes | **sound**, and the strongest gate of the round: nothing is pinned |
| `test_a_non_finite_q_excess_is_not_a_benign_sem_verdict` | the arm's own `q_excess`; the claim is a decision, not a reading | the premise gate IS the other side | yes, and the skip carries the reading | **sound, but partial**: it exercises only the `warn_own` arm. The same change also moves `refuse`, where the consequence is a RAISE rather than a warning, and nothing in the round pins that — see D-V3 |
| `test_the_warn_own_edge_is_not_decided_by_the_representation_of_the_width` | `_BOR_MIN_ELEM_FRAC` itself, derived | — | no | **narrower than it reads.** It exercises ONE `Rbig`, and its property holds only where the mesh reproduces the width below the bar — see D-V1. The assertion is right; the fixture is the whole claim |
| `test_a_caller_prescribed_liner_is_warned_wherever_it_sits_in_r` | six widths DERIVED from `_BOR_MIN_ELEM_FRAC` (1e-3x .. 100x) | yes | no | **sound as a ladder**, same fixture caveat as above |
| `test_near_cutoff_channel_count_is_stable_over_the_ladder[m-idx]` | ladder derived from `_BOR_CHANNEL_REAL_FLOOR`; bars from a measured 156-solve envelope, dated in the constant | the count claims are integers (unconditional); the closure bars carry 14.5x/12.2x and 11.2x | no — and correctly not, because all three claims are invariants | **sound**, and the scope constant is the thin one (§5.3). Corroborated on a second fixture by this verification |
| `_CUTOFF_LADDER_BAR` / `_CUTOFF_ENERGY_FLOOR_MULT` / `_CUTOFF_DEEP_BAR` | measured envelopes, with the measurement and date in the comment | 1e-5 two-sided, 2e-3 one-sided by construction and declared | — | **sound**; see §5.3 for the 1.09x headroom on the scope |

No gate in the changed files reads a pre-fix census, pins a prior version's
number, or skips on a resource check. The three strict xfails round 3 closed
were removed rather than left to XPASS.

---

## 9. Ship recommendation for 5.45.1

**SHIP.** The three behaviour changes do what the round says they do, on
fixtures written independently of theirs, and the one thing that could have
blocked a tag — a changed ANSWER — does not happen: 66 fixtures on two builds,
50 identical, 16 moved, every move a `HASH -> BORNodalPassivityError`, zero
illegal, and the documented escape hatch restores the round-2 bytes exactly on
72 of 72 rows.

Two items, neither a blocker:

1. **D-V1 (P3)** should be fixed, but it does not have to be fixed before the
   tag. It is bounded to a `UserWarning`, it is pinned by a strict xfail that
   will fire the moment it closes, and the round's own gate keeps passing on
   its own geometry. The remedy is small (size the deadband from
   `np.spacing(r_wall) / w`, which `measure_layer` already has in scope) and
   would be better done with a measurement of the other direction than in a
   hurry.
2. **D-V2 and D-V3 (both P3)** are documentation only and should ride with the
   tag, because both are about a caller-visible behaviour change the release
   notes do not mention. The CHANGELOG should say (a) that on the legacy nodal
   basis any stack with an absorbing half-space above roughly 2 vacuum
   wavelengths may now raise where it returned, that the refusal reads the
   half-space's own mode list rather than the cascade's answer, and that
   `BOR_NODAL_PASSIVITY_GUARD = False` restores the previous bytes; and (b)
   that on the SEM basis a manufactured sliver narrower than
   `_BOR_MIN_ELEM_FRAC` whose spectrum came back non-finite now raises
   `BORSemMeshError` instead of returning with a `warn_manufactured`. The
   round's rating of GAP 3's cost ("never a wrong returned number") should be
   corrected in the same edit.

The GAP 5 scope constant `_CUTOFF_ENERGY_FLOOR_MULT = 100` should carry a note
that it is a READING with 1.09x of headroom on an independently written
fixture, and that the bar it scopes keeps 21x there. No change to the number is
warranted.

---

## 10. What I could not verify

* **The CI's actual kernels.** Same limit as the fix round: `ZEN` aliases
  Haswell and `SKYLAKEX` dies on this host, so the EPYC 9V74 / 7763 pool is not
  reproducible. Every gate this verification adds asserts either an arithmetic
  invariant (build-free by construction) or a premise-gated decision.
* **Whether the derived lossy-ceiling term can EVER bind.** §4 settles that it
  cannot on a UNIFORM half-space (the exact excess is at most 0) and that no
  physical row of this code path lands inside its window. It does not rule out
  a radially GRADED absorbing half-space built by a caller who bypasses
  `BORStack` — I did not construct one, and `_physical_propagating`'s own
  comment says that case does not arise in `solve`.
* **The fix round's in-band `|flux|` spread** (5.5399e-08 against 4.5564e-03).
  The call that supplies the spectrum returns flux-normalised columns, so the
  raw quantity is not recoverable from it; the two claims that matter (0
  in-band backward modes, the worst row is the marginal channel) were measured.
* **Whether `_CUTOFF_DEEP_BAR` has a measurable signal side.** Unchanged from
  the fix round: it would need the pre-fix band re-implemented. I bounded its
  NOISE side across three arms instead (§5.3).
* **The `4 of 44` population both detectors miss**, PML, the anisotropic
  Class-C populations, and an external accuracy oracle for the SEM answer.
  Unchanged from all four prior documents.
* **`.test_durations` on a quiet box.** The five timings spliced here were
  taken with the 21-file matrix in flight on both builds, so they are upper
  bounds — the safe direction for a shard balancer.
* **The full un-masked main-CI matrix on the merge.** Out of scope here and the
  release gate by `docs/TESTING_STANDARDS.md`; what this document covers is the
  21-file BOR/EME set on six arms plus the census/walker sweep.
* **Whether D-V1's population is reachable from `BORStack`'s own API in a way a
  caller would hit by accident.** It is reachable deliberately (the fixture
  does it in three lines), and the arithmetic scan says 63 of 169 `(Rbig,
  wall)` combinations sit on the wrong side of the bar; I did not survey which
  `Rbig` values real callers use.

---

## 11. Probe inventory

All in `validation/probe_verify_bor_round3/`, each writing one JSON per arm
named `<probe>_<build>_<loaded kernel>_t<threads>.json`, with `LUM_PROBE_TAG=BASE`
prefixing the name when the same probe is run against `1ac6de7e`.

| probe | parts | what it decides |
|---|---|---|
| `_vb3.py` | — | fixtures, arm bookkeeping (loaded kernel read back from `threadpoolctl`), answer hashing |
| `v1_gap2.py` | `ladder`, `gain`, `switch` | GAP 2's decision table, gain disarm, the escape hatch's bit identity |
| `v2_ceiling.py` | `halfspace`, `census`, `fp` | what the ceiling reads, the set-right/set-wrong populations, the false-positive hunt and the trivial-stack consequence |
| `v3_liner.py` | `ladder`, `repr`, `formed`, `deadband` | GAPS 3 and 4 on two geometries, the arithmetic behind D-V1, the `q_measurable` construction, `_below`'s ULP behaviour |
| `v4_cutoff.py` | `family`, `mechanism` | GAP 5's two populations and the knee on an independent fixture; the backward-flux census |
| `v5_derived.py` | `exact`, `synthetic`, `hunt` | whether the derived lossy-ceiling term is tight or vacuous |
| `v6_identity.py` | — | the 66-fixture bit-identity battery (54 BOR / 12 EME, 32 with loss on a half-space) |
| `v7_durations.py` | — | the `.test_durations` splice, re-validated |
| `v8_mutation.py` | — | GAP 1: the replacement gate is live, the retired one was vacuous |
| `v9_tables.py` | — | aggregates every per-arm JSON into the tables this document quotes |
| `v10_sliver.py` | — | D-V3: GAP 3's change on the `refuse` arm, both trees |

Arm drivers and logs are in `validation/probe_verify_bor_round3/runs/`:
`arms_win.sh` / `arms_wsl.sh` (probe sweeps), `pytest_win.sh` / `pytest_wsl.sh`
(the 21-file set), `files.txt` (the set definition, taken from the fix round),
and one `.log` per arm.
