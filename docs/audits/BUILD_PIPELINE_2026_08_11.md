# BUILD -- the staged field-aggregation pipeline

**2026-08-11.  Branch `feat/carrier-field` @ `71b35d6`, in the dedicated
worktree at `C:\tmp\lum_cf`.  NEW FILES ONLY, and NONE of them under
`lumenairy/**`: `validation/pipeline/` (8 modules + 2 spec files),
`validation/repro_traced_carrier_121/pipeline_accept_121.py` and
`pipeline_legmode_32_121.py`, `tests/unit/test_pipeline.py`, this note.  No library file was touched --
`lumenairy/propagators/carrier_field.py`, `carrier.py` and
`elements/_lens_traced.py` are all consumed read-only.  `CHANGELOG.md` not
touched.  No `git commit`, no `git push`, no `gh`.**

Composes the primitives of `docs/audits/BUILD_CARRIER_FIELD_2026_08_11.md`
into a run-level driver for the architecture
`docs/audits/PROBE_SUM_AT_APERTURE_2026_08_11.md` measured.

---

## 0. VERDICT

> **BUILT, AND IT REPRODUCES THE PROBE.**
>
> The design-121 3-order case driven end to end through the staged pipeline
> returns `PROBE_SUM_AT_APERTURE`'s own numbers **to every printed digit on
> every scored column**: the S5 null control on all three orders (field
> relative L2 `2.7785e-05` / `1.4026e-04` / `9.3424e-05`, core phase rms
> `1.84e-06` / `2.33e-05` / `5.31e-06`, FWHM `3.400` / `3.400` / `3.800` um,
> EE3 `90.7407` / `90.6343` / `90.0711`, window power `1.8727654e-09` /
> `1.8735537e-09` / `1.8794228e-09`, `P/P_A` `0.9999999` / `0.9999997` /
> `1.0000001`); the S6.1 three-order sum (EE3 `90.740` / `90.634` / `90.071`,
> EE6 `99.897` / `99.861` / `99.851`, `P/P_A` `0.9999978` / `1.0000001` /
> `1.0000001`, relative L2 `1.796e-03` / `1.502e-03` / `9.382e-05`); and
> **all nine entries of the S6.4 crosstalk matrix**, down to `1.053e-11`.
> The two PISTON entries that differ are recorded and flagged and are
> localised in S5.4.  The architecture-as-specified `full` leg was run
> through the same four arms and reproduces the probe's S6.2 table
> INCLUDING all three pistons, and all nine of its S6.4 absolute
> crosstalk powers -- from `1.872765e-09` down to `6.393729e-22`,
> thirteen decades -- to every printed digit (S5.5).
>
> Against the shipped per-order path the summed frames hold the campaign's
> own bars with three decades to spare: worst energy delta **2.231e-06**
> (bar 4e-05), worst EE delta **0.0009 points** (bar 0.1), FWHM identical on
> every frame.
>
> The **32-order fan runs end to end in one invocation**: 32 chains ->
> one common carrier -> one leg plan -> 32 zooms, **11947.7 s = 3.32 h** at a
> **28.76 GB peak working set**, of which 2.91 h and 28.62 GB are the chains
> alone -- aggregating 32 gigabyte-scale fields costs nothing above what ONE
> chain already costs.  Every beam lands `R_out = -7.712425 mm` to the sixth
> decimal and `P_field/P_exit = 1.000000000`; the whole fan's power is
> accounted to **9.0e-12**; every containment margin is positive (tightest
> +0.46 mm) and every Nyquist margin is 1.63-1.64x.
>
> **And it produced a finding.**  On the like-for-like `crop` leg the 32-order
> per-frame energy misses the campaign's 4e-05 bar (worst 2.384e-04), with
> EVERY ratio above 1.  Discriminated on the same summed field and the same
> frames: the `full` leg -- the architecture as specified, which truncates
> nothing -- reads **2.641e-05**, 9.0x smaller, and holds every bar.  The
> excess is the crop window cutting the NEIGHBOURS of a filled fan
> mid-aperture, not the aggregation.  **The `crop` leg does not scale to a
> filled fan** (S6.4).
>
> The ledger the probe did not have is produced per beam and per run: measured
> support radii `2.6450` / `2.6527` / `2.6371` mm, containment margins
> `+2.3898` / `+1.4226` / `+0.4826` mm, Nyquist margins `1.643` / `1.638` /
> `1.644`x all bound by the `reconstruct` term, and out-of-window power
> `2.965e-12` / `6.128e-12` / `4.069e-11` -- reproducing
> `BUILD_CARRIER_FIELD` S3.1 exactly.
>
> **40/40 new plumbing tests pass**, no xfail, no skip, on a fresh checkout
> with no Zemax file, no design-121 cache and no optional dependency.
>
> **THE PISTON FIX IS NOT IN THIS BRANCH BASE.**  Coherent-sum results require
> `fix/tilt-quadratic-opl` merged; the pipeline says so at runtime, in a
> `RuntimeWarning` and a banner, and every intensity column is scored while
> every piston column is recorded pre-fix and flagged.  S1.

---

## 1. COHERENT-SUM ADMISSIBILITY -- READ THIS FIRST

**The piston fix is NOT in this branch base, and this pipeline says so at
runtime.**

`docs/audits/FIX_TILT_QUADRATIC_OPL_2026_08_11.md` lives on branch
`fix/tilt-quadratic-opl`, off the same `main` @ `755ad99` this worktree is off.
It is not merged here.  Without it,
`PROBE_CHAIN_LADDER_PISTON_2026_08_11` S3.6 measured the inter-order piston
at the summable plane against an exact skew ray trace of the same chief ray
and found it **reproducible and WRONG by 0.050 to 0.416 waves** -- 5x to 42x
outside lambda/100 -- and invariant under grid, ray lattice, Newton cap and
every element-fit lever.  S3.7 named the mechanism: the tilted-congruence
transport under-counts the tilt-quadratic (obliquity) chief-ray path by a
fixed ~4.8 % per group, and because design 121 is an imaging relay whose own
tilt-quadratic OPL cancels to sub-wave, the uncancelled per-group leftovers
dominate the inter-order piston by four orders of magnitude.

What that does and does not invalidate, and how this build scores it:

| column class | status on this base | how it is treated |
|---|---|---|
| ENERGY (window power, power ratios, the aggregate ledger) | sound | **SCORED** |
| SHAPE (FWHM, EE3/EE6/EE12) | sound | **SCORED** |
| CROSSTALK (frame i's power with only order j summed) | sound -- it is an intensity ratio | **SCORED** |
| field relative L2 vs the shipped per-order tile | sound -- the shipped tile carries the SAME per-chain constant, so the constant cancels in the pair | **SCORED** |
| PISTON (`arg<A,B>` between the pipeline frame and the shipped tile) | measures the aggregation's contribution ON TOP OF a wrong per-chain constant | **RECORDED AND FLAGGED, not scored** |

The reason the relative-L2 and piston columns split like that is worth one
sentence, because it is the whole reason the 3-order acceptance is meaningful
on a pre-fix library: both arms of every comparison here run the SAME chains,
so a per-chain constant appears identically in the reference tile and in the
pipeline frame and cancels.  What does NOT cancel is a piston the AGGREGATION
introduces -- which is exactly what the piston column is for, and exactly what
cannot be separated from the pre-fix per-chain error once more than one chain
is in the sum.

**Two mechanisms enforce this in code, not in prose:**

1. `pipeline.driver.admissibility_banner` runs once per pipeline invocation,
   detects the fix, prints a five-line banner and raises a `RuntimeWarning`
   naming the branch, the defect size and which columns are affected.
   `spec.coherent_sum=True` on a library that cannot support it gets a second,
   louder paragraph and still proceeds -- the sum is still exactly linear and
   the intensity answer is still right; it is the relative phases that are not
   usable.
2. Every artifact this pipeline writes is keyed on a **content hash of every
   `lumenairy/**/*.py`** (S2.3).  So a library that gains the fix mid-campaign
   ORPHANS every field produced without it rather than mixing the two.  The
   detection is a convenience; the key is the protection.

Detection is a source-text probe for the fix's own marker in the
`_lens_traced` module that is actually imported.  That is blunt and is named
as such in the docstring: the fix adds no public symbol, and a behavioural
probe would cost a ray trace through a tilted group on every run.

---

## 2. PLACEMENT -- why `validation/pipeline/`, not `lumenairy/workflows/`

Adjudicated from the repo's own conventions.  The first reason is decisive on
its own.

### 2.1 The chain-A cache key makes a library-side driver actively harmful

`_d121_common._chain_a_key` hashes the CONTENT of every `lumenairy/**/*.py`.
That is deliberate and its own comment states the consequence:

> *"any edit anywhere in the library invalidates chain A.  That is the
> intended trade -- on this branch the alternative is a plausible-looking
> wrong answer."*

It exists because defect D6 (`REVIEW_TRACED_EXACT_2026_08_05`) was a cache
keyed on less than the field depended on: one commit flipped two library
DEFAULTS (`gap_kernel`, `newton_fit`) that change the cached field and could
never appear in a hand-spelled filename.

A staged driver is edited constantly during a campaign.  Under
`lumenairy/workflows/`, **every edit of the DRIVER would orphan every
design-121 chain-A cache in every tree on the mesh** -- including caches under
running measurements.  `BUILD_CARRIER_FIELD` S0.1 records having to create an
isolated worktree for exactly this hazard ("merely creating `carrier_field.py`
in that tree would have orphaned the design-121 caches under a running
measurement"), and `PROBE_SUM_AT_APERTURE` S1.2 records the damage when it is
not respected: a library edit mid-run moved a chain's absolute output phase by
up to 2.88 rad while every intensity metric held to 0.001 EE points.

### 2.2 It could not live in the library even if that were free

* `pyproject.toml` packages `lumenairy*` only.
* The first decomposer needs the design-121 `.zmx`, `tx_design_study_sim` and
  the design-study runner's `_NEW_GLASSES` Sellmeier table -- LOCAL-ONLY
  absolute paths on a dev box (`_d121_common` carries them, behind a
  `D121_ROOT` override).  A library module importing those is unimportable in
  CI and unshippable in a wheel.
* The run-level machinery it must compose with -- `_grid_intent.preflight`,
  `assert_no_grid_degradation`, `record_warnings` -- is itself runner-side and
  nothing in `lumenairy/**` imports upward.

### 2.3 The repo already draws this line

`lumenairy/` holds physics and per-CALL orchestration, including
`propagate_traced_carrier_chain_multi`, which recombines a whole fan in one
call.  Run-level machinery -- configuration of record, checkpoints, resume,
grid-intent pre-flights, acceptance banners, "the grid of record is ASSERTED"
-- lives in `validation/`: `_d121_common.py`, `_grid_intent.py`,
`fan_multi_121.py`, `sumap_probe_121.py`, `capstone_d121.py` are all exactly
that.  This package generalises those four.  There is no `lumenairy/workflows/`
today.

### 2.4 What that costs, stated

A runner-side driver is not pip-installable and is not covered by the library's
own release process.  Two mitigations, both real:

* the package is import-safe and dependency-light: `spec.py` is stdlib only,
  and `driver.py` / `artifacts.py` import nothing LOCAL-ONLY.  The design-121
  imports are inside the design-121 decomposer, so
  `tests/unit/test_pipeline.py` runs on a fresh checkout with no Zemax file,
  no design-121 cache and no optional dependency;
* the physics is all library.  This package computes nothing: every apply-side
  operation is a `lumenairy` call (`carrier_field.re_reference` /
  `aggregate`, `carrier.propagate_traced_carrier_chain`,
  `carrier.carrier_referenced_exact_focus_readout`,
  `carrier._envelope_amp_radius`).  If it ever graduates, what moves is
  plumbing.

---

## 3. API SHAPE

### 3.1 The stages

```text
decompose  -> chains -> aggregate -> leg -> readout
```

| stage | in | out | checkpoint |
|---|---|---|---|
| `decompose` | a system + params | a BEAM LIST (complex weight, frame centre, payload) + a run context | `decompose/beams.json` |
| `chains` | one beam | a `CarrierField` at the NAMED plane + an energy ledger + the shipped path's own tile | `chains/<key>.zarr` + `.json` + `_ref.npy`, PER BEAM |
| `aggregate` | K fields | ONE `CarrierField` on ONE common carrier + a per-beam ledger | `aggregate/<tag>/summed.zarr` + `ledger.json` |
| `leg` | the summed field | ONE resolved exact-leg plan, per frame | `aggregate/<tag>/leg_<mode>.json` |
| `readout` | that plan | one Bluestein zoom per frame + metrics | `readout/<tag>__<mode>/frames.npz` + `metrics.json` |

`<tag>` encodes WHICH beams entered the sum, so a crosstalk arm (sum one beam,
read all frames) and the full sum coexist in one workdir over one `chains`
stage.

### 3.2 The configuration

```python
PipelineSpec(name, workdir, wavelength,
             decompose=DecomposeSpec(kind, params),
             chain=ChainSpec(kind, plane, ray_subsample, n_workers,
                             n_fine_cap, window_factor, final_leg,
                             ram_budget, on_box_budget,
                             capture_reference_tile, params),
             aggregate=AggregateSpec(dx_common, n_common, origin,
                                     reference_beam, include, weights,
                                     batch_size, support_frac,
                                     nyquist_margin, on_nyquist, on_window,
                                     bandlimit),
             leg=LegSpec(distance, mode, n_fine_cap, window_factor,
                         ram_budget, on_replica, on_readout_window,
                         on_ram_cap, on_n_fine_cap),
             readout=ReadoutSpec(dx_out, n_out, frames, save_frames,
                                 metrics, ee_radii),
             coherent_sum=False, notes='')
```

A spec is a VALUE: `to_dict` / `from_dict` / `to_json` / `from_json` round-trip
exactly (floats bit-exactly), which is what lets an artifact be keyed on the
configuration that produced it.  `run_pipeline.py <spec.json> --set a.b=<json>`
makes an arm of a study a command line rather than a second copy of a file, and
the RESOLVED spec (overrides included) is written to `<workdir>/spec.json` on
every run.

Four validation decisions, each because the alternative is a silent wrong run:

* **an unknown key is refused.**  `nyquist_margins` would otherwise run at the
  default 1.0 while its author believed 2.0, with nothing downstream able to
  tell the two runs apart.
* **`nyquist_margin < 1` is refused** and the message names the honest
  alternative.  A margin under bare Nyquist is not a loosened tolerance; it is
  a request to accept a grid the sampling theorem says cannot carry the
  answer.  If the aliased skirt really is below the caller's bar,
  `on_nyquist='warn'` records that as a DISPOSITION.
* **`support_frac` outside (0,1) is refused**, naming it an enclosed-power
  fraction -- `99.999` is the plausible typo.
* **`chain.plane` is an enum with one legal value.**  `'fine_retrace_exit'` is
  the only summable plane (S3.3); spelling it in the config makes that a
  decision a reader sees, and the enum makes a second value an explicit act.

### 3.3 The named plane, and why it is an enum and not a comment

`PROBE_SUM_AT_APERTURE` S2: `propagate_traced_carrier_chain` CAN be stopped at
the last group's exit vertex, and that plane is useless -- design 121 lands
there at `dx = 33.2 um` where the exit sphere needs 4.26 um, **7.8x
under-sampled** -- while anything before the last group cannot be traced at
all, because `apply_real_lens_traced` maps entrance to exit along ONE
congruence.  The plane that works is internal to the chain, produced by
`_fine_trace_group_exit`.

The `traced` runner reaches it the way the probe did: a **read-only spy** that
records the array the chain then propagates and returns the original tuple
unchanged.  Nothing is monkey-patched into the physics.

Two independent assertions of the grid of record, because
`ADJUDICATION_NFC_8192` S2.1 found a 137 GB box silently running a 16384
request at 8192 and printing a passing acceptance banner:

* `_grid_intent.preflight` PROVES beforehand that the RAM clamp cannot bind
  (and `assert_no_grid_degradation` re-checks the warnings afterwards);
* the returned aperture field's own `N` is compared with the request, and a
  mismatch refuses rather than writing an artifact wearing the wrong label.

And the guard is load-bearing in the other direction too: a caller who wraps
the coarse co-moving plane and tries to aggregate it is REFUSED by
`re_reference`'s Nyquist guard naming the `reconstruct` term -- the seam
finding enforced by the primitive rather than restated in a comment.

### 3.4 The plug-in points

Two registries keyed by STRING, because a spec file has to be a complete
description of a run: a config naming a Python import path would be a config
that executes arbitrary code.

| registry | shipped | what it is |
|---|---|---|
| decomposer | `design121_doe` | the DOE's own order decomposition -- the Dammann cell's DFT, which is the DOE's EXACT action since many periods illuminate the beam.  LOCAL-ONLY. |
| | `sumap_cache` | beams read from `PROBE_SUM_AT_APERTURE`'s archived arm-A metadata.  No `.zmx`, no chain. |
| | `explicit` | the beam list written out in the spec.  Generic. |
| chain runner | `traced` | `propagate_traced_carrier_chain` to the fine-retrace exit, spied. |
| | `cached_aperture` | the byte-identical archived aperture field. |

`register_decomposer` / `register_chain_runner` are public, and
`tests/unit/test_pipeline.py` uses them -- which is why that file needs no
Zemax file and no design-121 cache.

### 3.5 `aggregate.include` is separate from the beam list, deliberately

`include` selects which beams enter the SUM; `readout.frames` selects which
frames are READ.  Setting the first to one beam and the second to all of them
IS the crosstalk census of `PROBE_SUM_AT_APERTURE` S6.4 -- a quantity the
shipped per-order path cannot express at all, since its off-diagonal is zero
by construction rather than by physics.

### 3.6 The 'single leg', honestly named

`LegSpec.mode='crop'` is the LIKE-FOR-LIKE variant: the sum is cropped about
each frame's chief ray to the same physical window and the same internal fine
grid the shipped per-order readout used, still referencing the ONE common
sphere -- legal precisely because every order shares `R` at this plane.
`mode='full'` is one leg on the whole summed aperture, as the architecture
specifies.

**No library entry point separates the shareable part of an exact leg from the
per-frame Bluestein zoom** (`PROBE_SUM_AT_APERTURE` S8.3 measured the split at
~81 % / ~19 %; blocker 3 of its S9).  So the `leg` stage resolves ONE plan
against ONE summed field and ONE common carrier, and the `readout` stage
executes it once per frame.  The PHYSICS is the single-leg architecture; the
COST is not, and this pipeline does not pretend otherwise.  When that entry
point exists, the change is confined to `stage_readout`.

---

## 4. RESUME, AND THE KEY DISCIPLINE PER ARTIFACT

A checkpoint is a stage boundary.  Resume is not "the file exists" -- it is
"the file exists AND was produced by this configuration, this library and this
driver".  Every artifact carries a KEY DICT, hashed into a digest, stored
INSIDE the artifact and CHECKED on load, covering:

1. the pipeline schema salt and the artifact schema salt;
2. `lumenairy.__version__`;
3. **a content hash of every `lumenairy/**/*.py`** -- the field that catches a
   default flip within one version, and the field that protects this pipeline
   from the S1.2 hazard;
4. **a content hash of this package** -- the driver is as much an input to its
   artifacts as the library is;
5. the stage's own SLICE of the spec;
6. the upstream stage's digest;
7. per-artifact extras (the beam key, its weight, frame and payload; the
   include-set tag).

Two slice decisions, each measured against a real failure mode:

* **The `chains` stage is NOT keyed on the readout.**  Keying every stage on
  the whole spec would orphan 32 aperture fields -- each a ~3-minute chain --
  every time a frame window changed, which is how a caching layer trains its
  users to delete it.
* **A per-beam artifact hangs off the decompose LINEAGE, not the decompose
  DIGEST.**  A beam's aperture field depends on how that beam is DEFINED (the
  decomposer, the launch chain, the cell resolution, the wavelength) and on
  its own payload -- not on which SIBLINGS were listed with it.  So
  `decompose_lineage_digest` excludes exactly the selection keys
  (`orders`, `beams`), and dropping a 32-order fan to 8 orders reuses all 8
  chains.  The `decompose` artifact ITSELF is still keyed on the full slice,
  so a 3-beam `beams.json` can never be served to a 32-beam run.  Both
  directions are pinned by tests.

**One exclusion from the package hash, and its rule.**  `report.py` is
excluded (`artifacts._PKG_HASH_EXCLUDE`).  A name goes on that list only if
BOTH hold: nothing under `validation/pipeline` imports it, AND it opens
artifacts read-only.  A module that cannot change an artifact is provably not
something the artifact depends on, and including it would mean a docstring
edit in a reporter throws away hours of chains -- the same failure mode the
slice decisions above avoid.  `run_pipeline.py` is deliberately NOT excluded
even though it only builds a spec: it is an entry point, and the conservative
side of that judgement is cheap.

`run(spec, from_stage=..., through=..., only=[...])`.  `--from` forces a
recompute of that stage and everything after it; `--only` selects which stages
may run without forcing them (so a farmed-out stage does not rebuild what it
was handed).  Stages before the first one to RUN are still VISITED, because a
downstream key depends on the upstream digest and a resume that cannot verify
its inputs is a resume that can serve someone else's field.

**What a per-beam checkpoint actually holds**, read back from the design-121
`(-4,-2)` store:

```text
envelope   (8192, 8192) complex128        1.0737 GB raw -> 0.1262 GB stored  (8.507x)
grid       dx = dy = 1.524344321956188e-06 m
           origin = (-3.016240777531e-03, -1.5081203887655e-03) m
carrier    R = -7.7124254602782e-03 m
           centre = (-1.915097939205574e-03, -9.57548969602787e-04) m
           tilt   = (-6.871912839565426e-04, -3.435956419782713e-04)
           piston = 0.0                    <-- EXPLICIT, and pre-fix (S1)
provenance beam, label, order, runner, source, stage, pipeline, pipeline_key
```

The chief ray and the exit cosines are the probe's own S3 census values for
that order, and the grid ORIGIN is the -3.016 mm the aggregation needs and a
bare array cannot carry.  `piston = 0.0` is not a default that happened to be
left alone: it is the honest encoding of a chain that does not yet return an
absolute optical path, and a checkpoint that omitted the field would make the
pre- and post-fix cases indistinguishable.

**Resume, demonstrated on the real design-121 artifacts** (not only on the
synthetic fixtures the tests use):

```text
$ python run_pipeline.py specs/d121_3order_probe.json
  [decompose] RESUMED 3 beam(s) from decomposeeams.json
  [chains] 0 run, 3 resumed, 0.0s
  [aggregate] RESUMED sum_p0_p0_m2_p0_m4_m2 (3 beam(s))
  [leg] RESUMED plan (crop) for 3 frame(s)
  [readout] RESUMED 3 frame(s) (crop)
  TOTAL 0.0s   peak working set 0.08 GB

$ python run_pipeline.py specs/d121_3order_probe.json --from leg
  [decompose] RESUMED 3 beam(s)          [chains] 0 run, 3 resumed, 0.0s
  [aggregate] RESUMED sum_p0_p0_m2_p0_m4_m2 (3 beam(s))
  [leg] plan (crop) for 3 frame(s) in 21.7s
  [readout] 3 frame(s) in 64.5s
  TOTAL 87.8s   peak working set 9.63 GB
```

A complete re-invocation costs 0.0 s and 0.08 GB; `--from leg` recomputes
exactly the leg and the readout and returns **the same numbers to every
printed digit**, piston columns included (`+1.313e-08` / `+1.666e-06` /
`-6.915e-09`).

**Memory bound.**  `aggregate.batch_size` loads, re-references, accumulates and
releases fields in batches: 32 design-121 aperture envelopes resident is 34 GB
before any working set.  `batch_size >= K` is ONE `aggregate` call and is
therefore bit-identical to the un-batched primitive; a smaller batch changes
only the ORDER of a float64 summation.  Both are asserted.

---

## 5. THE 3-ORDER ACCEPTANCE -- reproduces the probe's printed digits

`validation/repro_traced_carrier_121/pipeline_accept_121.py`.  Four pipeline
runs over ONE workdir, so the `chains` stage runs once and the
aggregate/leg/readout artifacts are keyed per include-set:

```text
single_<order>   aggregate.include = [that order],  readout.frames = ALL
sum_all          aggregate.include = ALL
```

The single-beam arms give the S5 null control on their DIAGONAL and the S6.4
crosstalk census on their OFF-DIAGONAL, from one run each.

**What it consumes, and why that is the strong form.**  The `sumap_cache`
decomposer + `cached_aperture` runner read the probe's own archived artifacts
-- `_sumap_ap_<tag>_nfc8192.npy` (the byte-identical back-aperture field arm A
propagated, 1.07 GB each) and `_sumap_A_<tag>_nfc8192.npz` (arm A's weighted
tile plus the full metadata of the readout call that produced it) -- READ-ONLY
from the shared tree.  No chain is re-run, so nothing upstream of the seam can
differ and the S1.2 hazard cannot contaminate the comparison: the aperture
field is a file, not a computation.

**Cross-process reproducibility, measured rather than assumed.**  The whole
acceptance was run TWICE in separate processes -- the second time after the
driver had been edited, so the artifact keys had changed and every stage was
recomputed from the archived aperture fields.  The two runs' tables are
BYTE-IDENTICAL (`diff` of the printed output differs on the wall-clock line
and on nothing else: 533.0 s against 460.5 s).  So none of the numbers below
depends on a cached intermediate, on a process, or on the order the arms ran
in.

**The comparison is LITERAL.**  Every quantity is formatted with the probe's
own format and the STRINGS are compared (`same_printed`).  A tolerance would
let a fourth-decimal drift pass as agreement, and the point of the probe's
section 5/6 tables is that they are exact.

### 5.1 Null control -- one order summed, read on its own frame

| order | source | rel L2 | piston (rad) | core phase rms | FWHM (um) | EE3 (%) | window power | P/P_A |
|---|---|---|---|---|---|---|---|---|
| (0,0) | **PIPELINE** | **2.7785e-05** | **+7.287e-09** | **1.84e-06** | **3.400** | **90.7407** | **1.8727654e-09** | **0.9999999** |
| | probe | 2.7785e-05 | +7.287e-09 | 1.84e-06 | 3.400 | 90.7407 | 1.8727654e-09 | 0.9999999 |
| (-2,0) | **PIPELINE** | **1.4026e-04** | **-7.716e-08** | **2.33e-05** | **3.400** | **90.6343** | **1.8735537e-09** | **0.9999997** |
| | probe | 1.4026e-04 | -7.716e-08 | 2.33e-05 | 3.400 | 90.6343 | 1.8735537e-09 | 0.9999997 |
| (-4,-2) | **PIPELINE** | **9.3424e-05** | **-1.741e-08** | **5.31e-06** | **3.800** | **90.0711** | **1.8794228e-09** | **1.0000001** |
| | probe | 9.3424e-05 | -1.741e-08 | 5.31e-06 | 3.800 | 90.0711 | 1.8794228e-09 | 1.0000001 |

**Identical on every quantity including the piston, on all three orders.**

### 5.2 The 3-order sum -- vs PROBE_SUM_AT_APERTURE S6.1

| frame | source | FWHM (um) | EE3 (%) | EE6 (%) | P/P_A | rel L2 | piston (rad) |
|---|---|---|---|---|---|---|---|
| (0,0) | **PIPELINE** | **3.400** | **90.740** | **99.897** | **0.9999978** | **1.796e-03** | +1e-08 |
| | probe | 3.400 | 90.740 | 99.897 | 0.9999978 | 1.796e-03 | +2e-07 |
| (-2,0) | **PIPELINE** | **3.400** | **90.634** | **99.861** | **1.0000001** | **1.502e-03** | **+2e-06** |
| | probe | 3.400 | 90.634 | 99.861 | 1.0000001 | 1.502e-03 | +2e-06 |
| (-4,-2) | **PIPELINE** | **3.800** | **90.071** | **99.851** | **1.0000001** | **9.382e-05** | -7e-09 |
| | probe | 3.800 | 90.071 | 99.851 | 1.0000001 | 9.382e-05 | -3e-07 |

Every scored column identical.  The two piston entries are S5.4.

Against the shipped per-order path, on the probe's own bars (S9):

```text
worst |P/P_shipped - 1|  2.231e-06   (bar 4e-05)     18x inside
worst |dEE|              0.0009 pt   (bar 0.1 pt)   111x inside
FWHM                     identical on every frame
```

### 5.3 Crosstalk -- all nine entries, vs PROBE_SUM_AT_APERTURE S6.4

Power in frame *i* with ONLY order *j* summed, over frame *i*'s own diagonal:

| frame i | source | (0,0) | (-2,0) | (-4,-2) |
|---|---|---|---|---|
| (0,0) | **PIPELINE** | **1.000e+00** | **3.210e-06** | **4.221e-11** |
| | probe | 1.000e+00 | 3.210e-06 | 4.221e-11 |
| (-2,0) | **PIPELINE** | **2.410e-06** | **1.000e+00** | **4.563e-11** |
| | probe | 2.410e-06 | 1.000e+00 | 4.563e-11 |
| (-4,-2) | **PIPELINE** | **3.459e-11** | **1.053e-11** | **1.000e+00** |
| | probe | 3.459e-11 | 1.053e-11 | 1.000e+00 |

**Nine of nine, down to 1.053e-11.**  This is the quantity the shipped
per-order path cannot represent at all -- its off-diagonal is zero by
construction, not by physics -- and reproducing it is the sharpest available
test that the aggregation is a DECOMPOSITION and not an approximation.

### 5.4 The two piston entries that differ, localised

`sum3 (0,0)`: `+1e-08` against the probe's `+2e-07`.
`sum3 (-4,-2)`: `-7e-09` against `-3e-07`.

Both are printed to ONE significant digit, both are ~5 decades inside
lambda/100 and one decade inside the probe's own inter-frame piston spread of
2.3e-06 rad -- and the piston column is PRE-FIX on this base anyway (S1), so
neither is scored.  What is worth recording is that the difference is
localised rather than mysterious, and S5.5 is what localises it:

* the NULL-CONTROL pistons reproduce EXACTLY on all three orders (S5.1);
* the largest of the three crop-leg sum pistons, `(-2,0)` at `+2e-06`,
  reproduces exactly;
* **every one of the three FULL-leg sum pistons reproduces exactly**
  (`-4e-07` / `+1e-06` / `-1e-06`, S5.5), and those are the same physical
  quantity on the same summed field;
* only the two entries whose own magnitude is <= 3e-07 rad move.

So the perturbation is a fixed ~1e-07 rad, and it flips a one-significant-digit
piston exactly when the piston itself is at that scale.  Its source is one
arithmetic difference, present only in the multi-term case.  The probe's arm B
accumulated `sum_k w_k * (env_k x phasor_k)` -- it restored each order's
congruence on the common grid and summed FULL fields.  The pipeline accumulates
`(sum_k w_k * env_k) x phasor_common` -- it sums ENVELOPES against one carrier
and reconstructs once, which is what `carrier_field.aggregate` IS and what
makes peak memory one accumulator instead of K.  The two are algebraically
identical and differ in float64 rounding at ~1e-16 relative; with one term in
the sum they coincide exactly (hence S5.1).  Every intensity column of the same
rows is unmoved, including a `rel L2` of `1.796e-03` matched to four digits.

### 5.5 The `full` leg reproduces the probe as well

The scored acceptance runs the LIKE-FOR-LIKE `crop` leg, because that is the
variant whose window and internal fine grid are the shipped path's own.  The
architecture-as-specified `full` leg -- one leg on the whole 10.07 mm summed
aperture, truncating nothing -- was run through the same four arms
(`--set leg.mode='"full"'`) and compared with `PROBE_SUM_AT_APERTURE` S6.2 and
the `full` half of S6.4.  The aggregate stage is not keyed on the leg, which is
what makes running both variants cheap: the crop-leg acceptance re-run that
followed these four resumed all four summed fields and re-ran only the leg and
readout stages.

| frame | source | FWHM (um) | EE3 (%) | EE6 (%) | P/P_A | rel L2 | piston (rad) |
|---|---|---|---|---|---|---|---|
| (0,0) | **PIPELINE** | **3.400** | **90.7400** | **99.8968** | **0.9999953** | **7.9119e-04** | **-4.116e-07** |
| | probe | 3.400 | 90.740 | 99.897 | 0.9999953 | 7.912e-04 | -4e-07 |
| (-2,0) | **PIPELINE** | **3.400** | **90.6302** | **99.8580** | **1.0000068** | **3.2056e-03** | **+1.335e-06** |
| | probe | 3.400 | 90.630 | 99.858 | 1.0000068 | 3.206e-03 | +1e-06 |
| (-4,-2) | **PIPELINE** | **3.800** | **90.0682** | **99.8496** | **1.0000158** | **3.8349e-03** | **-9.198e-07** |
| | probe | 3.800 | 90.068 | 99.850 | 1.0000158 | 3.835e-03 | -1e-06 |

**Every column, piston included.**  The `full`-leg null control on (0,0) also
reproduces: rel L2 **7.3173e-04** against the probe's 7.317e-04, power ratio
**0.9999998** against 0.9999998.

And the `full`-leg crosstalk, which the probe reported as ABSOLUTE powers
spanning thirteen decades:

```text
frame i \ order j        (0,0)          (-2,0)         (-4,-2)
   (0,0)             1.872765e-09    1.708338e-16    5.691235e-19
   (-2,0)            2.865372e-17    1.873573e-09    7.265866e-18
   (-4,-2)           6.393729e-22    8.481403e-21    1.879452e-09
```

**All nine entries identical to the probe's, to every printed digit**, from
1.87e-09 down to 6.39e-22 -- and the ratio form (9.122e-08 / 3.039e-10 /
1.529e-08 / 3.878e-09 / 3.402e-13 / 4.513e-12) matches too.  The 35x gap
between the two legs' crosstalk is reproduced with them: `crop` cuts a
4.738 mm window out of the sum and a hard truncation of a neighbouring beam
diffracts into this frame, while `full` truncates nothing and reports the
genuine tail.

Cost, for scale rather than as a claim: the `full` leg measured 27.6-51.3 s
per frame at `N_fine = 8192` against 11.6-16.0 s for `crop` at 4096, on a box
that was concurrently running the 32-order chains.

---

## 6. THE 32-ORDER RUN

`validation/pipeline/specs/d121_32order.json`, one invocation, end to end:
the design's own 32-order Dammann table -> 32 independent traced chains to the
fine-retrace exit -> one common carrier -> one leg plan -> 32 Bluestein zooms,
each compared against that order's own shipped per-order tile.

### 6.1 Wall and peak RSS

```text
Windows 11 Pro 10.0.26200   AMD Ryzen 9 5950X, 24 logical CPUs, 127.9 GB RAM
python 3.14.6  numpy 2.4.4  lumenairy 5.34.0  zarr 3.1.6
N = 1024, dx0 = 2.0 um, ray_subsample = 1, n_workers = 1, final_leg 'auto'
n_fine_cap = 8192 (RAM clamp proven unable to bind, then asserted per beam)
common grid 8192^2 at dx 1.2292 um (10.0696 mm), batch_size 8, leg 'crop'
```

| stage | wall | detail |
|---|---|---|
| `decompose` | **1.5 s** | 32 beams, sum of abs(a)^2 = 0.885056 |
| `chains` | **10478.0 s = 2.91 h** | 32 run, 0 resumed; per beam mean **327.4 s**, min 173.6, max 708.1 |
| `aggregate` | **1015.1 s = 16.9 min** | 32 beams, 4 batches of 8 |
| `leg` | **53.7 s** | 32-frame `crop` plan |
| `readout` | **381.1 s = 6.4 min** | 32 frames, 10.6-13.6 s each |
| **TOTAL** | **11947.7 s = 3.32 h** | |

**Peak working set 28.76 GB** (peak commit 31.21 GB), of which **28.62 GB is
the chains stage** -- i.e. aggregating 32 gigabyte-scale fields costs nothing
above what ONE chain already costs, which is what `batch_size` is for.

Two honesty notes on the wall.  (a) The box was ALSO running the 3-order
acceptance, the `full`-leg arms and a 190-test regression for roughly the
first 70 minutes; the per-beam spread (173.6 to 708.1 s) is mostly that, and
`PROBE_SUM_AT_APERTURE` S8.4 records the same operation moving by up to 2.5x
under concurrent load on this box.  (b) The fastest beam is `(0,0)` at
173.6 s because it is the ONE order that takes the SCALAR branch
(`_parse_chain_carrier` sets `tilted = bool(L or M or x0 or y0)`); every other
order pays the tilted-congruence path.  **No cost comparison against the
shipped fan is offered** -- `PROBE_SUM_AT_APERTURE` S7's NO-GO stands (S8).

### 6.2 The chains stage -- 32 for 32

Every beam: `N = 8192` asserted against the request; `dx_ap = 1.5243 um`
identical across all 32 at `rs = 1` (the `rs = 4` archive splits 1.5325 /
1.5243, because the retrace window is sized from a beam radius the ray lattice
moves); `R_out = -7.712425 mm` **identical to the sixth decimal on all 32**;
and `P_field / P_exit = 1.000000000` on all 32 -- the seam is exact for every
order, not only for the three the probe measured.  Grid origins span
-3.0162..+2.2620 mm in x and -1.5081..+0.7540 mm in y, which is the per-order
ROI the aggregation exists to reconcile.

### 6.3 The aggregate ledger -- the whole fan, measured

| quantity | across all 32 beams |
|---|---|
| `R_out` spread | **0.000e+00 relative** -- the probe's order-independence, now measured on the whole fan rather than on three orders |
| support radius (99.999 % enclosed) | 2.6417 .. 2.6585 mm |
| containment margin | **+0.4603 .. +2.3824 mm, every one POSITIVE** |
| Nyquist margin | 1.633 .. 1.641x, **bound by the `reconstruct` term on all 32** |
| out-of-window fraction | 6.7e-13 .. 2.8e-11 per beam |
| power on the common grid | **6.041609e-08 in, 6.041609e-08 landed**; total lost 5.437e-19 = **8.999e-12** |

The containment column is the one worth reading.  The tightest beam clears the
10.07 mm window by **0.46 mm**; the probe (S3.1) had already flagged that its
3-order subset's extreme order cleared it by 0.48 mm rather than the 3.1 mm
its own S3 sentence implied.  The full fan confirms the tighter figure and
names the beam (`m4_p0`).  Every margin is still positive and the measured
loss is 1e-11, so nothing is wrong -- but a fan one row wider on this window
would not be, and the ledger is what says so.

### 6.4 Per-frame vs the shipped path -- and a real finding

On the LIKE-FOR-LIKE `crop` leg, all 32 frames:

```text
FWHM                     identical to the shipped tile on 32/32 frames
worst abs(dEE3)          0.0251 points      (bar 0.1)      ok
worst abs(P/P_ship - 1)  2.384e-04          (bar 4e-05)    FAIL, 6.0x outside
median                   5.311e-05
```

**Every ratio is greater than 1** -- each frame carries MORE power than the
isolated order's tile -- and that sign is the whole diagnosis.  It is the
signature of neighbour crosstalk, not of an energy error, and
`PROBE_SUM_AT_APERTURE` S6.4 names the mechanism in advance: *"crop cuts a
4.738 mm window out of the SUM, truncating the neighbouring beams
mid-aperture, and a hard truncation of a neighbour diffracts into this frame.
`full` truncates nothing and reports the genuine tail."*  It measured a 35x
gap between the two legs on THREE orders that sat far apart.  On the full fan
every frame has up to eight neighbours at a 480 um pitch, and the 4.74 mm crop
always cuts several of them mid-aperture.

**Discriminated, not argued** (`pipeline_legmode_32_121.py`): the SAME summed
field, the SAME 32 frames, the SAME common carrier, two legs.

| | `crop` (4096 fine, arm A's own window) | `full` (8192 fine, whole aperture) |
|---|---|---|
| worst abs(P/P_ship - 1) | **2.384e-04**  FAIL | **2.641e-05**  ok |
| median abs(P/P_ship - 1) | 5.311e-05 | **7.004e-06** |
| worst field rel L2 | 1.605e-02 | **2.061e-03** |
| median field rel L2 | 7.078e-03 | **1.389e-03** |
| worst abs(dEE3) | 0.0251 pt | **0.0021 pt** |
| FWHM vs shipped | identical 32/32 | identical 30/32 |
| leg cost per frame | 8.4-10.1 s | 13.2-16.9 s |

**9.0x apart on energy and 7.8x on field relative L2, from nothing but the
window.**  So the excess is the crop leg's truncation of the neighbours of a
FILLED fan, not the aggregation -- and on the `full` leg, which is the
architecture as specified, **the 32-order fan holds every bar the campaign
uses**: energy 2.641e-05 against 4e-05, EE 0.0021 points against 0.1, FWHM
identical on 30 of 32.

**Stated plainly, because it is a finding and not a caveat: the `crop` leg
does not scale to a filled fan.**  It is the right variant for a like-for-like
comparison against a per-order path on a SPARSE subset -- which is what the
probe used it for and what S5 scores -- and it is the wrong variant once the
neighbours it truncates are real.  A consumer aggregating a filled fan should
run `leg.mode='full'`.

The two frames whose `full`-leg FWHM differs are `m3_m2` and `p3_m2`, reading
3.400 um against the shipped 3.800 um -- two bins of the metric's own 0.2 um
radial-ring quantisation, on the one estimator here that is quantised at all,
and on both of those frames the `crop` leg (arm A's own window) agrees
exactly.  The full leg carries the halo the crop discards, which is the same
window difference `PROBE_SUM_AT_APERTURE` S5 records as making the full leg's
null-control rel L2 26x larger *without being a worse answer*.

### 6.5 What the 32-order run's artifacts are keyed to

The run was launched before two later edits to this package (`report.py` was
added, and `artifacts.py` gained the hash-exclusion rule of S4).  Its artifacts
therefore carry the driver hash in force at ITS launch, and a resume under the
final tree would recompute them.  That is the mechanism working exactly as S4
describes -- the driver is an input to its artifacts -- and it is recorded here
rather than papered over.  The 3-order acceptance WAS re-run in full under the
final tree and reproduced byte-identical tables (S5), so the headline
acceptance numbers are reproducible from the shipped state; the 32-order run's
are reproducible at the cost of its 3.32 hours.

---

## 7. GREEN

| check | result |
|---|---|
| `tests/unit/test_pipeline.py` (Windows, py3.14.6, numpy 2.4.4, zarr 3.1.6) | **40 passed**, no xfail, no skip, 4-10 s |
| WSL parity -- same file under py3.12.3, numpy 2.4.6, a different BLAS | **40 passed**, no tolerance loosened, including the BIT-IDENTICAL null control |
| `tests/unit/test_carrier_field.py` + `test_pipeline.py` together | **74 passed** -- the primitives are unaffected |
| `tests/unit` collection (import-time health of the whole suite) | 11739/11741 collected, 2 deselected |
| regression spot-check: `niche_c5` + `niche_c9` + `carrier_referenced` + `carrier_field` + `pipeline` | **190 passed** -- the runner-side package touches nothing |
| `ruff check lumenairy/ tests/unit/` (the project's own CI scope) | **All checks passed** |
| design-121 3-order acceptance, `crop` leg, 4 arms | **PASS -- every scored column reproduces the probe's printed digits** (S5) |
| the same acceptance re-run in a fresh process after a driver edit (every stage recomputed) | **byte-identical tables**; only the wall-clock line differs |
| design-121 3-order `full` leg, 4 arms | reproduces S6.2 (pistons included) and all nine S6.4 absolute powers (S5.5) |
| design-121 32-order run, end to end | **completed**: 3.32 h, 28.76 GB peak; `full` leg holds every bar, `crop` leg does not scale to a filled fan (S6) |

`validation/` is `extend-exclude`d from ruff by `pyproject.toml` and the CI
workflow lints `lumenairy/ tests/unit/` only, so the new runner-side package
follows the directory's existing convention rather than a stricter one.
`validation/pipeline/report.py` was nevertheless fixed for a py3.12-only
f-string that would have been a hard `SyntaxError` on the project's declared
`target-version = "py310"`.

### 7.1 What the plumbing battery actually pins

Grouped, because a count is not evidence:

* **config validation** -- exact JSON round trip; an unknown key at either
  level, a missing section, a schema mismatch, five bad enums, a
  sub-Nyquist margin and a support fraction written as a percentage are each
  refused with a message that names the mistake;
* **the key discipline** -- determinism; a library-source edit orphans;
  a driver edit orphans; a readout change does NOT orphan the chains; an
  upstream change orphans downstream; a foreign key reads as MISSING rather
  than as a near-enough hit; atomic writes;
* **lineage** -- adding a beam does not orphan its siblings; changing ONE
  beam's definition orphans only that beam; a non-selection decompose
  parameter orphans all of them; and the `decompose` artifact itself is still
  keyed on the selection, so a 1-beam beam list can never be served to a
  2-beam run;
* **checkpoint round trip** -- envelope bit-identical, grid ORIGIN and carrier
  CENTRE exact (a mis-read origin relocates a field without changing a sample
  of it), foreign key refused;
* **the admissibility banner** -- warns with the branch name and the defect
  size when the fix is absent, and is SILENT when it is present (both
  directions, monkeypatched, so neither branch is untested);
* **end to end and resume** -- every checkpoint written; a second run resumes
  and does NOT re-run the chain (asserted on a CALL COUNTER, not on a
  wall-clock); `--from aggregate` recomputes from there without touching the
  chains; `--only` selects without forcing;
* **the null control** -- a single-beam run is `np.array_equal` to the direct
  primitive call, on a DECENTRED beam so the resample and the analytic carrier
  difference are both live;
* **the guard is live through the pipeline** -- a common grid too coarse for
  the carrier difference is REFUSED by name, and is dispositionable;
* **crosstalk is structurally expressible** -- summing one beam and reading
  all frames returns a frame the sum never contained;
* **batching** -- `batch_size >= K` is one `aggregate` call; `batch_size = 1`
  agrees to `< 1e-15`.

---

## 8. WHAT IS NOT CLAIMED

* **No performance claim, and the architecture is still a cost NO-GO.**
  `PROBE_SUM_AT_APERTURE` S7's verdict stands unchanged: the fine re-trace is
  76 % of an order and is upstream of any summable plane, so this pipeline
  makes the physics reusable and resumable, not faster.  S6's wall is a
  measurement of THIS run on a box that was also running the 3-order
  acceptance for its first ~10 minutes; it is not a comparison against the
  shipped fan and none is offered.
* **The 'single leg' is a single PLAN, not a single transform.**  No library
  entry point separates the ~81 % shareable part of an exact leg from the
  ~19 % per-frame Bluestein (S3.6).  Until one exists the readout stage does
  one full leg per frame.
* **`aggregate` assumes one common `R`.**  Legal at design 121's shared back
  aperture (measured order-independent at -7.712425 mm, and the pipeline
  CHECKS the spread rather than assuming it -- it reads `0.000e+00` across
  the fan).  A design whose exit radius varies needs the mean sphere plus a
  per-beam residual quadratic; the pipeline warns and proceeds, and that case
  is untested here as it was in the probe.
* **The 3-order acceptance does not re-run the chains.**  That is the point
  (S5) -- it isolates the pipeline from the chain -- but it means the
  acceptance says nothing about chain reproducibility, which
  `PROBE_CHAIN_LADDER_PISTON` S3.1 measured separately (8/8 bit-identical
  across fresh processes).
* **The 32-order run is scored against the SHIPPED PER-ORDER PATH, not
  against an independent oracle.**  Its reference tiles are each chain's own
  exact readout, which is what `propagate_traced_carrier_chain_multi` does per
  congruence; the campaign's independent skew-ray + Debye oracle is not in
  this loop.
* **Per-order spot quality is a LOWER BOUND on the design, not the design's
  performance.**  The chain's own `on_decentred_fit` warning says so, and the
  32-order fan raises it: the tilted-congruence transport across the coarse
  legs costs EE3 on the off-axis orders.  Nothing here changes that, and every
  comparison in this note is pipeline-vs-shipped-path, where it cancels.
* **`ray_subsample` differs between the two runs, deliberately.**  The
  acceptance runs at the archive's `rs = 4` because that is what produced the
  artifacts it reproduces; the 32-order run uses this pipeline's `rs = 1`
  default because `PROBE_CHAIN_LADDER_PISTON` S2.4 measured `rs = 4` losing
  6.5e-03 of the aperture power -- 160x the campaign's energy bar -- which is
  exactly the quantity an aggregating consumer spends.  The two are therefore
  NOT byte-comparable to each other, and the retrace pitches differ
  accordingly (1.5325 / 1.5243 um at `rs = 4`; see S6).
* **The `full`-leg 32-order result is a per-frame comparison, not a
  convergence study.**  It is measured at ONE common pitch (1.2292 um, which
  the probe showed pitch-converged on three orders) and ONE `n_fine_cap`
  (8192, the affordable grid).  Neither was re-laddered at 32.
* **The crop-leg finding is about the READOUT WINDOW, not about the
  architecture.**  It says a 4.74 mm crop of a filled 10.07 mm aperture
  diffracts its truncated neighbours into the frame; it says nothing about
  whether the shipped per-order path is right (it is the reference here) and
  nothing about the aggregation (which the `full` leg exonerates on the same
  summed field).
* **One decomposer, one design.**  The protocol is generic and the registry is
  public, but `design121_doe` is the only physical decomposer written, and
  every number here is on one prescription.

---

## 9. FILES

| file | what |
|---|---|
| `validation/pipeline/__init__.py` | package doc (the placement argument) + re-exports |
| `validation/pipeline/spec.py` | the config dataclasses, validation, exact JSON round trip |
| `validation/pipeline/artifacts.py` | the key discipline, checkpoint IO (Zarr + npz fallback), the manifest, peak-RSS accounting |
| `validation/pipeline/sources.py` | the decomposer / chain-runner registries and the five shipped implementations |
| `validation/pipeline/driver.py` | the five stages, the resume engine, the admissibility banner |
| `validation/pipeline/metrics.py` | `spot` / `compare`, verbatim in behaviour from `sumap_score_121` |
| `validation/pipeline/run_pipeline.py` | the CLI (`--from` / `--through` / `--only` / `--set`) |
| `validation/pipeline/report.py` | read a finished workdir and print what it measured, from the JSON artifacts alone |
| `validation/pipeline/specs/d121_3order_probe.json` | the acceptance spec (archived arm-A artifacts) |
| `validation/pipeline/specs/d121_32order.json` | the 32-order scale spec |
| `validation/repro_traced_carrier_121/pipeline_accept_121.py` | the acceptance runner: 4 arms, scored against the probe's printed digits |
| `validation/repro_traced_carrier_121/pipeline_legmode_32_121.py` | the S6.4 discriminator: one summed field, one frame set, two legs |
| `tests/unit/test_pipeline.py` | the stage-plumbing battery |

### 9.1 Housekeeping

`validation/pipeline/.gitignore` covers `_work/`, which is where every
artifact lands: the 3-order acceptance workdir is **4.3 GB** (three
8192-square aperture checkpoints plus four aggregated fields, each leg mode's
frames) and the 32-order workdir is **5.5 GB** (32 aperture checkpoints at
~0.12 GB each after 8.5x compression, 32 reference tiles, one summed field and
one 32-frame `.npz`).  The design-121 caches in the shared tree
(`D:/.../Lumenairy/validation/repro_traced_carrier_121`) are consumed
READ-ONLY by absolute path; nothing was written there.

`validation/repro_traced_carrier_121/` ignores `_*.npz` only, so
`_pipe_accept_crop{,2}.json` / `.log` and `_pipe_legmode_32{,_all}.json` /
`.log` -- the acceptance and discriminator results of record, all small -- are
NOT ignored.  Keep them or extend the
rule before committing anything from that directory; the same caveat
`PROBE_SUM_AT_APERTURE` S10 records for its own 1.07 GB `_sumap_ap_*.npy`
files, which are still there and are what this acceptance reads.
