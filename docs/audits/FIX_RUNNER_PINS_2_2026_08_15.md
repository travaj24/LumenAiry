# Five more runner-axis pins, and the whole-suite sweep for their class -- 2026-08-15

Branch `fix/runner-pins-2` off `origin/main` (1b8600d, the 5.35.4 release
commit).  `git log origin/main..HEAD` was empty before branching.

Five tests across three files went red on the release verify shards and main
CI over the last four tags while staying green on both local mounts.  This is
the same disease `FIX_RUNNER_PINS_2026_08_12` treated, and the one before it,
and the one before that: **a test that asserts a PER-BUILD fact as if it were
universal is the bug, not the runner.**

Two things are different this round.

1. One of the five is a repeat offender in the SAME assertion that
   2026-08-12 already re-stated.  That re-statement replaced an absolute bar
   with two ratios -- but both ratios still had the per-build magnitude in the
   numerator, so the disease survived the treatment.  S2 records what that
   costs and what the actual cure is.
2. Because the observed instances kept arriving one tag at a time, the
   mandate was extended to a **PROACTIVE whole-suite sweep**: enumerate the
   fragile SHAPES mechanically across all 476 files of `tests/unit`, triage,
   and fix the genuinely fragile ones now.  S7 and S8 record that.

Nothing in `lumenairy/` changes.  Every fix is test-side.

**Mounts.**  **W** = Windows py3.14.6, numpy 2.4.4, scipy 1.17.1,
scipy-openblas, 24 cores / 128 GB.  **M** = WSL py3.12.3, numpy 2.5.1,
scipy 1.18.0, OpenBLAS -- this is the numpy 2.5-era wheel the failing ubuntu
job runs, and it reproduces failure 1 exactly.  Also available and used for
the final matrix: `/tmp/venv-py310` (py3.10.20, numpy 2.2.6) and
`/tmp/venv-py311` (py3.11.15, numpy 2.4.6).

**Import provenance.**  Every measurement and every verification in this
document ran against the worktree, not an installed copy.  Probe scripts were
written INSIDE `C:/tmp/lum_cons` (so the script directory that Python puts on
`sys.path` IS the worktree) and every pytest invocation used `python -m pytest`
from inside the worktree.  Confirmed on both mounts:
`lumenairy.__file__` = `C:\tmp\lum_cons\lumenairy\__init__.py` [W] and
`/mnt/c/tmp/lum_cons/lumenairy/__init__.py` [M], version 5.35.4.

---

## S1.  What failed, where

| # | test | reading | bar | shape |
|---|---|---|---|---|
| 1 | `test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_stripe_reduces_to_rigorous_1d` | 0.005325 | < 5e-3 | S4 floor bar |
| 2 | `test_niche_audit_w9_eig_vjp.py::test_pmm1d_angle_gradient_at_exactly_zero_is_an_OPEN_defect[te]` | 16.6x | > 100x | S1 magnitude-ratio defect pin |
| 3 | `test_niche_audit_w9_eig_vjp.py::test_the_theta0_defect_pin_fires_when_the_defect_is_absent_or_unbounded` | same | same | S1 (the control for 2) |
| 4 | `test_niche_audit_e_prepared_and_enums.py::test_h2_pool_path_is_actually_engaged` | `0 >= 1` | -- | S3 env-dependent precondition |
| 5 | `test_niche_audit_e_prepared_and_enums.py::test_h2_resolved_cap_travels_in_the_pickled_payload` | `set() == {12}` | -- | S3 (same fixture as 4) |

Local baseline before any change: 1 REPRODUCES on **M** (0.005101 vs the 5e-3
bar); 2-5 pass on both mounts, because both mounts are big boxes and both
happen to land on the lucky side of the per-build quantity.

---

## S2.  The taxonomy this round hardened

Five shapes, named so the sweep in S7 could be mechanical.

| shape | what it is | the cure |
|---|---|---|
| **S1** MAGNITUDE-RATIO DEFECT PIN | "the defect manifests at >= K times a reference" -- fragile when the manifestation's MAGNITUDE is a build fact | a build-free discriminator: presence of WRONGNESS against a DERIVED error bound, with the magnitude demoted to printed adjudication |
| **S2** PRE-FIX-REFERENCING ARM | a pre-fix / legacy reading frozen as a constant | recompute the pre-fix arm in-process (engineered / injected demonstration) |
| **S3** ENV-DEPENDENT PRECONDITION | "the pool/thread/cache path engaged", sized by box resources; and any skip that hides one | force the precondition deterministically (explicit counts, INJECTED budgets), and assert the resource decision separately in BOTH directions |
| **S4** FLOOR BAR | a numeric bar within ~1 decade of the locally measured value on a BLAS/eig/FFT-derived quantity | a regime-tied measured-envelope bar with a dated derivation, or a reference derived in the same process |
| **S5** EXACT-COUNT / EXACT-SET on nondeterministic machinery | `== {..}`, `len(..) == N` on pool/cache/parallel observations | force determinism, or assert the invariant instead of the count |
| **S6** FIXED-CENSUS READING | an assertion that names a specific mode count / census vector / growing-mode number as if every build produced it | condition on the census THIS build reads and verify the rule for whatever state manifests, plus an engineered arm for the states that did not |
| **S7** PARAMETRIZED-INJECTION assumption | a `parametrize` arm that assumes a particular case will manifest a particular state | force the state, or make the arm self-adjudicating |

S6 and S7 were added mid-round, after main CI produced two more failures of
exactly those shapes (S8.5).  They are the same disease seen through a
different organ: a census reading and a parametrized arm are both per-build
facts when the machinery producing them is conditioning-sensitive.

The permitted cures are exactly four: build-free discriminators,
derived-from-this-build measurements, engineered/injected demonstrations, and
deterministically forced preconditions.  Never xfail, skip, rerun, or blind
widening.  Every changed bar carries a dated derivation comment.

---

## S3.  Failure 1 -- the fff_nv energy bar (S4)

**Reading.** `abs((sum Rf + sum Tf) - 2.0) = 0.005325` against `< 5e-3` on
ubuntu with a numpy 2.5-era wheel.  Reproduced on **M** at 0.005101.

**Adjudication -- and this is the whole finding.**  The bar asserts an energy
theorem that `fff_nv` does not have.  `rcwa_jones_2d`'s own docstring says so,
under INHERENT CLOSURE ERROR (audit M5, 2026-07-25):

> the `fff_nv` in-plane operator is NON-Hermitian, so there is no
> finite-truncation energy theorem behind it -- a LOSSLESS `fff_nv` cell
> violates `R+T = 1` by ~1e-2..6e-2 ... at EVERY truncation, and the
> resulting `_EnergyWarning` is a property of the formulation, not an
> instability signal.

`stabilize=True` does not even count that warning as a failed rung for this
formulation, for exactly that reason.  A 5e-3 bar on that quantity was
calibrated on one build's reading of a number the library documents as
ranging an order of magnitude wider.

**And the fixture is worse than that.**  At the `n_orders=11` this test uses,
the closure residual of ALL THREE arms is a per-build fact.  Measured
2026-08-15, one worktree, one fixture:

| arm | W (py3.14 / np2.4.4) | M (py3.12 / np2.5.1) | spread |
|---|---|---|---|
| `fff_nv` | 2.987e-03 | 5.101e-03 | 1.7x |
| `laurent` | 4.180e-05 | 6.247e-03 | **150x** |
| rigorous 1-D (Li-1996) | 1.742e-02 | 1.764e-05 | **990x** |

plus 5.325e-03 for the `fff_nv` arm on the failing ubuntu wheel.  The
rigorous 1-D arm is the tell: its lossless closure is EXACT and the library
holds it to `<1e-11` on a clean solve, so 1.742e-02 on **W** is not
truncation, it is the measure-zero layer/region mode-match coincidence
`_check_energy` documents -- and at that truncation its per-order answers
are, in the library's own words, "suspect".  **The test was comparing against
a poisoned reference**, and which truncations are poisoned is per-build.

**Fix.**  Three parts.

1. The closure bar becomes REGIME-TIED to the formulation's own documented
   envelope: `_FFF_NV_CLOSURE_ENVELOPE = 6e-2`, the upper end of the
   library's `~1e-2..6e-2`.  That is 11x over the worst reading any build has
   produced (5.325e-03) and still strictly inside the solver's hard energy
   tripwire (which raises at a defect of 1e-1), so the assertion still says
   something the library does not already enforce.  The DOE-closure precedent
   ("two decades over the worst cross-build reading") cannot apply here --
   two decades over 5.3e-3 is 0.53, past the physical maximum -- so the
   regime, not the observation, is what sets the bar.  That is recorded in
   the constant's own comment.
2. Teeth restored where they need no calibration: every per-order efficiency
   must be finite and non-negative.  A solve that has gone non-physical shows
   up there first, and the claim has no bar to move.
3. **The reference is FORCED to be sound** -- `_sound_1d_reference()` searches
   a truncation ladder for the first order at which the rigorous 1-D solver's
   own energy theorem actually holds on the running build (closure < 1e-9,
   which is 100x above the library's documented `<1e-11` clean solve and four
   decades below every poisoned reading), and the reduction claims are stated
   against THAT.  It raises rather than skipping if the window has no sound
   order.

Part 3 also fixed a sibling that was one build away from red, and it made
both surviving claims stronger rather than weaker:

| claim | vs poisoned n=11 ref (W / M) | vs sound ref |
|---|---|---|
| `ef/el` (sum R) | 0.0530 / 0.0484 | **0.0343** (29x) |
| `jf/jl` (Jones) | 0.0410 / **0.4185** | **0.0247** (40x) |

The Jones arm asserts `jf < jl`, i.e. it fails at 1.0.  It read 0.419 on
**M**: a 10x cross-build swing and 2.4x of margin left.  It was next.

---

## S4.  Failures 2 and 3 -- the theta=0 eig-VJP defect pin (S1)

**Reading.** `2.91e-05 > 100 * 1.754e-06` failed; the runner manifested the
defect at only 16.6x.

**Adjudication.**  The defect is real and still OPEN upstream -- 16.6x wrong
is wrong.  What is not universal is HOW wrong.  At an exactly (symmetry-)
degenerate eigenvalue pair the eigenvector basis inside the degenerate
subspace is a LAPACK choice, and `AD(0)` is the size of the resulting jump:

| build | TE `AD(0)` | `AD(0)/clean` |
|---|---|---|
| authoring box (2026-07-27) | -2.221e-03 | 1.3e3 |
| **W** py3.14 / np2.4.4 | +7.793e-03 | 4.4e3 |
| **M** py3.12 / np2.5.1 | +1.506e-02 | 8.6e3 |
| CI ubuntu py3.12 (2026-08-12) | -2.664e-02 | 1.5e4 |
| CI ubuntu, numpy 2.5-era wheel | +2.910e-05 | **1.66e1** <- failed |

Three decades of spread and two sign flips.  **The 2026-08-12 remediation had
already diagnosed this correctly** -- it proved with the floor's own knob that
the value is an eigenvector jump and not a `1/D` reading -- and then re-stated
the claim as two RATIOS.  But both ratios kept `|AD(0)|` in the numerator, so
both still asserted the magnitude.  The envelope arm was in the same state:
`|AD(0)|/sweep` measured 1.333 [W] against a bar of 30 (22x) but 0.0179 [M],
a 75x cross-build spread.  **Restating a magnitude as a ratio does not make it
build-free if the magnitude is still in it.**

**Fix -- presence of wrongness, not how wrong.**  Mirror symmetry forces the
true derivative to be EXACTLY 0 at normal incidence, so the question "is AD
still wrong?" needs no magnitude at all -- only the resolution at which "zero"
can be asserted.  For a function EVEN about 0, the central difference is
analytically zero at every step size (all truncation terms cancel
identically), so its entire reading is float64 cancellation and the resolution
is the derived `eps_mach * |R(0)| / h`.

VERIFIED, not assumed: the FD reading scales as `1/h` over three decades of
`h` on both mounts (TE: 6.9e-13 / 2.1e-11 / 8.3e-10 at h = 1e-5 / 1e-6 / 1e-7)
and lands at 1.38x [W] / 0.46x [M] of the derived floor.

The discriminator is then `|AD(0) - FD(0)| > 1e3 * max(resolution, |FD(0)|)`,
and it separates the two regimes by decades in BOTH directions:

| regime | `|AD(0) - FD(0)| / resolution` |
|---|---|
| still defective | 1.9e6 (the failing runner) .. 2.0e10 (5 builds) |
| defect FIXED (rcwa1d control, analytic half-spaces) | 3.3e-4 .. 7.9e-5 (2 mounts) |

`1e3` sits **3.3 decades below the smallest defective reading ever observed**
and **6.5 decades above the largest clean one**.  It cannot burn a tag, and it
still fires the day the upstream defect is closed -- which is the whole point
of a defect pin.  `clean` and `sweep` are still measured and now appear as
printed adjudication in the failure message.

**The boundedness half moved out** to its own test, stated as the MECHANISM it
actually is, using the injector that already existed:

| arm | splitting x1 | x1e-4 | x1e-8 | law |
|---|---|---|---|---|
| FLOORED (shipped) | +7.792834e-03 | +7.793044e-03 | +7.793047e-03 | insensitive; rel move 2.73e-05 [W] / 2.73e-06 [M] |
| UNFLOORED | -2.030e-03 | -9.822e+01 | -9.823e+05 | `|x1e-8| / |x1e-4|` = **1.0000e+04 on both mounts** |

`F -> 1/conj(D)` makes that 1e4 exact arithmetic, not a calibrated magnitude.
Neither arm contains an absolute bar or a cross-arm magnitude.

**Fail-before.**  The control (`rcwa1d`, no degeneracy) fails the new
discriminator by three decades and still raises with the "re-pin this test"
message, so the pin remains two-sided.  Test count in the file went 31 -> 32
(the old combined fail-before split into a defect-absent arm and a
floor-mechanism test); nothing was removed.

---

## S5.  Failures 4 and 5 -- the Newton pool engagement pins (S3)

**Reading.** `assert 0 >= 1` and `set() == {12}` on 2-core / 7 GB CI runners.

**Adjudication -- the library was RIGHT.**  `_newton_resolve_workers` prices
the pool at `_NEWTON_POOL_RAM_FRAC (0.5) * available - _NEWTON_POOL_MIN_FREE_GB
(2 GB)` against a measured ~1.85 GB per worker.  A 7 GB runner with ~4 GB
available gets `0.5*4 - 2 = 0.0 GB` of budget, correctly refuses every worker,
and falls back to serial.  The observations then come back empty.  Verified
by driving the shipped pricer directly:

| `_free_b` | pool budget | workers resolved |
|---|---|---|
| 64 GB | 30.0 GB | 2 (engaged) |
| 32 GB | 14.0 GB | 2 (engaged) |
| 8 GB | 2.0 GB | 1 (refused) |
| 4 GB | 0.0 GB | 1 (refused) |

The test asserted a resource-dependent precondition unconditionally.  Worse,
the fixture's response to the same situation was a `pytest.skip` -- so on
exactly the boxes that gate the release, **five E-H2 tests silently stopped
running together**.

**Fix -- force the precondition, and assert the resource decision separately.**

* `_POOL_WORKERS = 2` explicitly, so the dispatch does not follow
  `available_cpus()`.
* `_POOL_PRICED_IN_B = 64e9` fed to the shipped pricer through **its own
  documented test hook**, `_newton_resolve_workers(..., _free_b=)`, whose
  docstring reads "overrides the live memory read (tests only)".  Everything
  else in the pricing law still runs as shipped -- rule 1's unguarded-`__main__`
  refusal, the per-worker memory model, the re-price-at-the-count-we-will-run
  loop.  Only the byte count becomes deterministic.
* Both `pytest.skip`s deleted.  The fixture now produces the same
  observations on a 2-core box as on a 24-core one.
* The payload pin keeps its SET form (the invariant: every payload that
  crosses the wire carries the resolved cap) and gains an explicit
  non-emptiness guard, because an empty observation satisfies nothing and
  `set() == {12}` named the wrong defect.
* NEW: `test_h2_pool_engagement_follows_the_ram_pricing_both_ways` asserts the
  resource decision as its own claim in both directions -- priced in, the pool
  is built and the payload crosses; priced out (4 GB, the CI runner's own
  regime), the pool is refused, no worker is requested, and the answer is
  unchanged (the clamp is a pure resource decision).

**Verified under the CI regime, not just described.**  The whole E-H2 section
was re-run with `lumenairy.set_max_ram(4.0e9)` active -- which is what makes
the live pricer refuse -- and all 8 tests pass, because engagement no longer
depends on the live read.

---

## S6.  Sibling sweep of the three failing files

Each file was swept for the same shapes.  Measured on both mounts.

| site | shape | reading (W / M) | verdict |
|---|---|---|---|
| `fff_nv.py` energy bar | S4 | 2.99e-3 / 5.10e-3 vs 5e-3 | **FIXED** (S3 above) |
| `fff_nv.py` Jones arm `jf < jl` | S4 | 0.041 / **0.419** ratio, fails at 1.0 | **FIXED** -- sound reference |
| `fff_nv.py` absorptance split | S4 + **defect** | ratio 0.9972 on both | **FIXED** -- see below |
| `fff_nv.py` OOP closure `< 5e-3` | S4 | 2.04e-14 / 7.99e-15 | **TIGHTENED to 1e-9** |
| `fff_nv.py` `ef[-1] < 0.3*el[-1]` | S1 | 0.0952 / 0.0952 (bit-identical) | sound -- deterministic; derivation recorded |
| `fff_nv.py` `|f7-conv| < 2e-5` | S4 | 9.664963e-06 both (bit-identical) | sound -- deterministic; derivation recorded |
| `fff_nv.py` `< 0.5*|l7-conv|` | S1 | 0.3194 / 0.3194 (bit-identical) | sound -- deterministic; derivation recorded |
| `fff_nv.py` `gaps[2] < 0.5*gaps[0]` | S1 | 0.2816 / 0.2816 (bit-identical) | sound -- deterministic; derivation recorded |
| `fff_nv.py` Berreman SVD `< 1e-11` | S4 | 1.6e-15 / 1.1e-15 | sound (6000x) |
| `w9_eig_vjp.py` `_THETA0_DEFECT_RATIO` | S1 | -- | **FIXED** (S4 above) |
| `w9_eig_vjp.py` `_THETA0_ENVELOPE_RATIO` | S1 | 1.333 / 0.0179 vs 30 | **FIXED** (S4 above) |
| `w9_eig_vjp.py` exact-degeneracy `err < 5.0` | S1 | 0.3131 / 0.4377, both mounts identical | **HARDENED** -- see below |
| `w9_eig_vjp.py` forward nullity `< 1e-13` | S4 | 2.85e-15 / 3.23e-15 (35x / 31x) | **FIXED** -- derived from cond |
| `w9_eig_vjp.py` `|ad-fd| < 1e-8` | S4 | 2.83e-10 / 1.92e-10 (35x / 52x) | sound; 1.5x spread recorded |
| `e_prepared.py` pool skips (x2) | S3 | -- | **FIXED** (S5 above) |
| `e_prepared.py` `set(caps) == {1}/{12}` | S5 | -- | **FIXED** -- invariant + emptiness guard |
| `e_prepared.py` pool-vs-serial `< 1e-11` | S4 | 4.4e-14 documented | sound (250x) |

**The OOP closure tightening, justified.**  The out-of-plane sibling used the
same 5e-3 bar as the failure, but that fixture is a TILTED-uniaxial stripe:
it carries `exz`/`ezx` but `exy = eyx = 0`, so the non-Hermitian in-plane
operator that costs `fff_nv` its energy theorem is never excited at normal
incidence.  A full ladder confirms the regime rather than assuming it --
closure defect by `n_orders`, both formulations, **W**:

| n_orders | 5 | 7 | 9 | 11 | 13 | 15 | 19 | 25 | 31 |
|---|---|---|---|---|---|---|---|---|---|
| `fff_nv` | 4.0e-15 | 7.3e-15 | 1.0e-14 | 4.0e-14 | 2.0e-14 | 1.8e-14 | 4.0e-14 | 5.3e-15 | 2.2e-14 |
| `laurent` | 2.1e-14 | 6.2e-15 | 1.8e-15 | 1.1e-14 | 8.0e-15 | 2.3e-14 | 2.7e-15 | 4.9e-14 | 4.8e-14 |

Worst reading anywhere: 5.3e-14.  The old 5e-3 bar had 2e11 of headroom and
asserted nothing; 1e-9 is 100x above the library's own "<1e-11 on a clean
solve" and 1.9e4 above the worst of the 18 readings above.

Three sites deserve their own note.

**A real test defect, found by the sweep.**
`test_fff_nv_lossy_stripe_absorptance_split` was measuring something other
than what it claimed.  The 2-D arms were reduced PER INCIDENT POLARIZATION
(`np.sum(R, 1)` -> a (2,) vector -> `np.mean`), but the 1-D reference was
built as `1 - sum(R1) - sum(T1)` with the sums over BOTH pols at once -- so
`A1` was `2A - 1`, not `A`:

| | as written | corrected |
|---|---|---|
| `A1` | -0.14089 | 0.42956 |
| `|Af - A1|` | 0.57054 | 9.094e-05 |
| `|Al - A1|` | 0.57214 | 1.694e-03 |
| ratio | **0.99720** | **0.05368** |

Both sides were dominated by the factor-of-two offset, so the assertion
reduced algebraically to `Af > Al` -- nothing to do with tracking the
reference -- and passed on a 0.28% margin.  This is a TEST defect, not a
library one (the library's per-pol convention is consistent), so it was fixed
here.  Correcting the normalisation makes the test assert what its docstring
always claimed AND turns a 1.003x coin flip into an 18.6x margin.

**The exact-degeneracy bound** (`err < 5.0`, worst recorded variant 0.75, so
6.7x) keeps its regime bar -- the eig route is missing ONE of several
comparable terms, so its relative error is O(1) by construction, and the
readings are bit-identical on both mounts -- but the load-bearing claim is now
COMPARATIVE and measured in the same process: the same exactly-degenerate
case with the floor removed and the splitting shrunk by 1e-8 is 1.2e8 / 1.4e8
times worse.

**Forward nullity** was a frozen 1e-13 on a quantity whose scale is set by
conditioning.  It is now derived in-build as `1e2 * eps * cond(A)`;
`cond(A03) = 5.196` and `cond(A_deg) = 1` on both mounts, giving 1.2e-13 and
2.2e-14, with the measured values 42x and 128x under them.

---

## S7.  The proactive whole-suite sweep

`tests/unit` is 476 files / 218,715 lines.  It was sharded five ways and swept
for the five shapes, mechanically enumerated then triaged in context.

| shard | raw hits | distinct sites triaged | fragile |
|---|---|---|---|
| 1 | ~570 | ~500 | 7 |
| 2 | ~1,100 | ~460 | 5 |
| 3 | 751 | ~430 | 4 (+1 watch) |
| 4 | 539 | ~120 read individually | 4 groups |
| 5 | 1,041 | ~150 | 9 |
| **total** | **~4,000** | **~1,660** | **~29 sites** |

The headline is that the suite is in much better shape than the five failures
suggest: the overwhelming majority of hits are sound, and several files are
already textbook implementations of the cures in S2 --
`test_hammer_h3_traced_nyquist_guard.py::_pin_available_ram` (pins BOTH the
psutil read and the RAM budget), `test_fix_newton_pool_memory.py` (injects
`_free_b`, exactly the hook S5 adopts), `test_v5_6_rcwa_convergence.py`
(searches for the blow-up order on THIS LAPACK build before pinning),
`test_niche_c12_physics_fit_selection.py` (module docstring states the
invariant: "every numeric bar is a RATIO between two arms measured in the same
process"), and `test_v5_12_0_pmm_covariant_oblique.py` (whose author
explicitly REMOVED a ratio bar because it "measured 0.2x-41x across
builds/degrees").

**The sweep found two more already-failing tests**, on nobody's list.

* `test_fix_runner_oom_2026_08_13.py:271` fails in a bare process on **W**
  right now (313.726 MB against a 313.520 MB bar, -0.07% margin).  Root cause
  is `tracemalloc` accounting, not memory: the first `_Cheb2DEvaluator` call
  pulls cupy/cupyx/cuda/ml_dtypes in through the array-backend dispatcher,
  ~5.27 MB, which is MORE than the assertion's entire 5.06 MB margin.  It
  passes under pytest today only because an unrelated conftest fixture
  happens to touch cupy first -- i.e. it is decided by test ORDER and by
  whether the runner has cupy installed.
* `test_v4_16_0_agent_b_multiprocess_storage.py` was **already red on W**
  (`1 failed, 52 passed` before any change) -- not at the wall-clock bar the
  sweep flagged, but at the precondition above it: `assert
  os.path.exists(ready_signal)` after a 5.0 s wait.  Measured spawn ->
  lock-held latency on **W**: 4.84 / 8.18 / 9.66 s standalone and 20.98 /
  23.26 / 35.19 s when the child is spawned from inside pytest (the fresh
  interpreter re-imports the module, so it pays pytest + numpy + h5py); on
  **M** the same latency is 5.55 s.  A textbook per-build fact sitting on a
  hard-coded deadline.

---

## S8.  Sweep findings: classification and disposition

Every fragile site found, with its shape and disposition.  Sites marked FIXED
were repaired in this branch under the S2 cures; each carries its own dated
derivation comment at the assertion.

*(Table completed in S8.1 below as the fix wave landed; sites deferred to the
EME owner are listed in S8.2.)*

### S8.1  Fixed in this branch

See the per-file derivation comments; the disposition table is:

| file:line | shape | reading | disposition |
|---|---|---|---|
| `test_fix_runner_oom_2026_08_13.py:271` (**RED on W**) | S4 | cold bare process 315.695 MB [W] vs a 313.520 MB bar; warmed 308.456 MB on BOTH mounts.  The 7.24 MB gap is exactly cupy/cupyx/cupy_backends/cuda/ml_dtypes/_cython first-imported through the array-backend dispatcher -- more than the assertion's whole 5.06 MB margin | FIXED -- `_peak` gained a keyword-only `_warm=` that makes one throwaway call on a tiny input BEFORE opening the tracemalloc window.  **No bar was touched**; only the measurement became build-free.  Fail-before reproduced in the same bare harness at 313.73 MB |
| `test_fix_runner_oom_2026_08_13.py:215` and the `_coexist` GRAM site | S4 | 283.406 vs 324.000 MB; 438.678 vs 544.000 MB | FIXED -- same treatment; both now identical on the two mounts, and each stays 2.1-2.8x above the floor a reintroduced full copy would cross |
| `test_niche_r0_byte_budgeted_cache.py:306,313,260` | S3 | consecutive live psutil reads differ by up to 139,264 B on an IDLE box; the 10%-fraction branch was dead code on every dev box | FIXED -- autouse `_pin_available_ram` at 8 GiB pins both import-bound names; `:313` split into a parametrized fraction-branch (4 GiB) and cap-branch (8 GiB) test, each an exact equality.  Strictly more coverage |
| `test_niche_r0_byte_budgeted_cache.py:426` | S3 | `< 256 MB -> skip` guarding a stress whose measured peak growth is 5.78 MB [W] / 5.89 MB [M] -- 35x oversized | FIXED -- guard removed, the shared-mutex contract now runs unconditionally |
| `test_audit_s2_10_asm_H_consolidation.py:52,139,226` | S4 | 1.34x over an observed CI value, and BELOW the sqrt(2)*ULP worst case | FIXED -- `sqrt(2) * max(half-ULP)` derived from this build's oracle via `np.spacing`; bars 8.429e-8 / 1.686e-7, identical on both mounts (the envelope saturates at \|H\|=1).  Adversarially checked: a 1-step-per-component perturbation passes, 2-step fires on all five sites |
| `test_m1_conditioning_guard.py:1222` | S2/S4 | absolute 1.6e-9 dressed as a ratio; reference 21x stale | FIXED -- reference recomputed in-process |
| `test_niche_audit_w6_asymptotic.py:1838` | S4/S1 | ratio of two cond numbers both past 1/eps | FIXED -- claim moved to the well-conditioned order |
| `test_obl_banded_halo.py:498` | S3 | tracemalloc peak ratio, 1.2x margin | FIXED -- structural band assertion |
| `test_niche_audit_w7_rcwa.py:242` | S4 | 22x with a 10x cross-build spread | FIXED -- scored against the same build's non-uniform closure |
| `test_niche_c13_lstsq_conditioning.py:141,130` | S1/S4 | 2.27x on a LAPACK null-space draw; 10.1x on an rcond | FIXED -- derived optimality bound; `n*eps` |
| `test_v5_11_0_pmm_stack.py:203` | S1 | 1.40x, 14% cross-build move | FIXED -- convergence RATE comparison |
| `test_niche_e4_corrected_relay_oracle.py:396` | S1 | 1.59x on a collapsed-fit denominator | FIXED -- two absolute claims replace one ratio |
| `test_audit_bor_grazing_cutoff.py:183` | S4 | 6.0x on scalar-vs-SIMD summation | FIXED -- same-kernel identity + ULP budget |
| `test_v5_12_0_pmm2d_staggered.py:58,72` | S4 | 15x with a 2.7x spread | FIXED -- scored against this build's modal error |
| `test_v5_14_1_rcwa_deferred.py:381` | S4 | clean closure 3.34e-13/4.68e-13 [W] vs 1.73e-10/1.24e-9 [M] -- the file's own recorded WSL values moved 14.5x and 7400x on a numpy POINT RELEASE; old bar 4.4x over the worst known (2.25e-8, CI ubuntu) | FIXED -- scored against the smallest injected loss, whose ladder reads 1.3401454e-06 identically to all eight figures on BOTH mounts and at 1/4/default BLAS threads.  Honest caveat recorded in the test: the available band is only 59.6x wide, so no placement has large headroom both ways; the bar sits a full decade under the smallest injected defect and 5.96x over the worst clean reading, costing 1.34x detection sensitivity |
| `test_niche_r4_fga_dual_vectorize.py:331` | S3 | wall clock: 13.6x idle / 12.1x under `-n 2` / 11.5x -- 18% spread with nothing changed but the weather | FIXED -- replaced by a DISPATCH COUNT through the public entry (`seen == {'numba': 49, 'numba_none': 0, 'dual': 0}`), plus a guard that the kernel actually built, so the claim cannot pass vacuously on a box where it never compiled.  Timing printed.  Test renamed (a count-claim has no bar to calibrate) |
| `test_audit_propagation.py:2586` | S3 | `t_fused/t_ref` 5 trials/mount: 0.693-0.833 [W] but **1.267** [M] against a 1.30 bar -- 2.6% headroom on the SECOND mount, with 3 of 5 WSL samples showing the "fused" path SLOWER.  `.test_durations` records 0.64 s on CI against ~5.5 s [W] / ~7.5 s [M] -- a 10x machine spread on the asserted quantity | FIXED -- the v4.13.1 win is WORK, counted via a numpy attribute spy: fused `{exp:1, einsum:1, sum:0}`, reference `{exp:2, einsum:0, sum:1}`, and `chunk_beamlets=200` gives `{exp:2, einsum:2}` proving it is per-chunk.  The reference IS the pre-fix two-exp form, so the count is its own fail-before.  Sole timing assertion left is a 60 s blow-up net (14.6x over the worst measured) |
| `test_audit_propagation.py:3482,3546` | S3 | `if warm_peak == 0: pytest.skip(...)` guarding two bit-equality pins -- and pin 3 goes VACUOUSLY TRUE at `warm_nz == 0`, so the guard could only hide the failure beside it.  Measured `warm_peak` 4.994503e-03 / 4.615952e-03, 1024/1024 non-zero, identical both mounts; `-rs` shows it never fired | FIXED -- `assert warm_peak > 0.0` |
| `test_niche_k3_perf.py:204` | S3 | best-of-3 ratio 0.649/0.746 [W], 0.410 [M] idle -- and an in-test single shot of **0.861 under co-tenancy, which the retired 0.85 bar would have FAILED**.  Same code, 0.41-0.86 across two mounts | FIXED -- spies `Delaunay` (verified live on scipy 1.17.1 AND 1.18.0) with `LinearNDInterpolator` as an API-stable second witness: new = 1, old = 2.  RAM gate deleted and the grid cut 768 -> 256 so the claim runs unconditionally |
| `test_niche_d1_tilted_carrier.py:1333` | S1 | 7.3x on an explicitly BLAS-dependent ghost; **already burned once** (widened 1000x -> 50x after a py3.13 CI job read 365x) | FIXED -- two build-free discriminators: the library's OWN fold-detector boolean (0 folds oracle / >=1 broken), and SUPPORT LOCATION -- the true halo's brightest off-disc pixel is by construction the first pixel outside the 3w disc (1.200 mm, both arms, both mounts) while the broken arm's is a DETACHED lobe at 6.737 mm, where the oracle's amplitude is 8e-121 against a 1e-12-of-peak floor.  Ratio printed (8.192e+04), not asserted |
| `test_audit_optimize.py:355,359,481,498` + `test_niche_audit_r_guards_and_merits.py:427,429` | S3 | blanket `except ... -> pytest.skip` wrapping exactly the call under test.  The stated excuse had no basis: at `w_s = 20 um` the fixture reads `v = 0.9937107888169847` to every printed digit on BOTH mounts -- 30 decades above the `>1e-30` floor the skip tested -- and no configuration in a swept envelope (aperture 2-100 mm, source/pupil box 3 decades) was non-finite on either mount | FIXED -- all four blanket excepts and both value-predicate skips DELETED (a raise is the finding); `except NotImplementedError -> pytest.fail` kept; capability gating stays `importorskip`.  NON-VACUITY BANDS added (`1e-4 < |L|^2 < 1`) because without them `v3/v1 == 3` and `nv ~= jv` are satisfied for free by an underflowed tensor |
| `test_v4_16_0_agent_b_multiprocess_storage.py:416` + the 5.0 s boot deadline above it (**RED on W**) | S3 | spawn->lock latency 4.8-35.2 s [W] / 5.6 s [M] against a 5.0 s wait | FIXED -- release-file handshake; the holder provably still owns the lock when the call gives up, and the same call succeeds after release.  Elapsed printed, never asserted |

### S8.2  Referred, not touched (EME territory)

`tests/unit/test_eme_census_determinacy.py` and
`lumenairy/elements/eme/eme_2d_vector.py` are owned by a concurrent agent, and
`tests/unit/test_eme_2d_vector.py` is that library file's direct test.  Two
fragile sites in EME territory are therefore REPORTED, not fixed:

| file:line | shape | reading | note |
|---|---|---|---|
| `test_eme_2d_vector.py:175,176` | S5 | `recall >= 8` / `spurious <= 2`; measured 15/1, own docstring records a CI dip to **9** | one mode of margin on a shift-invert match-boundary count; the file's own comment says the eigenvalues "shift enough between MKL and OpenBLAS to drop a handful of borderline modes" |
| `test_niche_audit_w6_eme.py:546,621,474` | S5/S1/S4 | count bar with one unit of slack (observed 0-1, bar 2, defect at 3); 4.9x on a root-polish artifact; ~9x/11x on `sigma_min` dips | suggested cure: score against the accepted-zero population measured in the same build |

### S8.3  Watch list (sound today, recorded so the next reader can re-derive)

* `test_niche_upsample_lattice_fix.py:225` -- 1.63x, but the reference is an
  exact-ray oracle; would become fragile if C13-class conditioning reaches it.
* `test_niche_audit_w9_raster_harmonic.py:523` -- sound as run (2.368 / 2.199
  bit-identical on both mounts), but the DOCSTRING cites an `n_x = 256` case
  that measures 1.646 against a 1.5 bar (1.10x).  Widening `_MODES`/`n_x` to
  match the docstring would make it fragile immediately.
* `test_niche_audit_w6_berreman.py:771,1113` -- `assert r["corr"] > 0` needs
  NumPy's and JAX's eigensolvers to agree on at least one raw column order.
  If it ever fires, drop the element-wise arm and keep the LAPACK-free RULE
  arm (which measures exactly 0.0), rather than widening.
* `test_audit_s1_3_pmm2d_lossless_tripwire.py:102,117` -- 1.8x / 2.5x on a
  deliberately ill-conditioned pseudo-inverse, but verified digit-for-digit
  across both mounts.
* Live-free-RAM `pytest.skip` gates (`test_niche_c11:65`, `test_niche_c1:79`,
  `test_niche_d3_guards:461`, `test_niche_k1:444,556,605`, `test_niche_p2:304`,
  `test_niche_r9:48`, `test_niche_k2:513`).  These cannot burn a tag, but they
  make coverage a function of what else the box is doing.  The durable cure is
  the `_pin_available_ram` pattern plus an always-running small-N arm;
  deferred as a batch rather than rushed.  (`test_niche_r0:426` was in this
  list and is now CURED -- see S8.1; its guard was 35x oversized.)
* WSL `/tmp/venv-ci` has no `filelock`, so two h5py storage cases in
  `test_v4_16_0_agent_b_multiprocess_storage.py` (gated only on `_HAS_H5PY`)
  raise `ModuleNotFoundError` there.  Pre-existing and unrelated to this
  branch; the durable fixes are adding `filelock` to that venv or gating
  those classes on `_HAS_FILELOCK` too.  Neither was done here -- one mutates
  a shared CI-proxy venv, the other adds a skip.

---

### S8.5  Three more, from main CI mid-round

Main CI (with the EME fix merged) produced three further failures on new
pythons while all five of the originals PASSED on that sampling -- which is
itself the point: they flip per build.

**`test_doe_rcwa.py::test_build_table_round_trips_through_its_cache`**,
closure 2.556e-08 against a 1e-08 bar on the 6-lambda `_tiny_table` fixture.
This one CORRECTS A FIX FROM THIS SAME CAMPAIGN.  Two days earlier (5.35.4)
the closure bar was SPLIT by regime -- 1e-8 kept for the small-period
fixtures, 1e-6 given to the 20-lambda ones -- because a py3.10 verify shard
had read 1.15e-08 at 20 lambda "while passing every small-period site on the
same run".  That last clause was an inference from one build's sampling, and
a 6-lambda fixture has now read 2.556e-08: WORSE than the reading the split
was built around.

| sampling | fixture | reading |
|---|---|---|
| authoring sweep (4 period x 3 index x 5 truncation) | small period | 6e-11 |
| v5.35.3 verify shard, py3.10 wheel LAPACK | 20 lambda | 1.15e-08 |
| main CI, new python | 6 lambda | **2.556e-08** |

The regime axis is real but it was not the binding one; the BUILD axis is,
and it moves every site.  The split is retired: all five closure
preconditions in that file now share one envelope bar of 1e-6 -- 39x over
the worst reading any build has produced, 4 decades below the 1e-2-scale
pathology the module note names, which is what these preconditions exist to
catch.  The `np.array_equal` cache-identity assertions in the same test are
same-process byte claims and are untouched.

The lesson is worth stating plainly: **"the other sites passed on the run I
looked at" is the same mistake as "the bar held on the build I measured".**
A regime split is only sound if the regime, not the sampling, is what
separates the readings.

**`test_pmm_m2_window_contract.py`** (two tests) --
`census reads [0, 0, 0, 0, 2] raw growing mode(s) over degrees (8..16),
spread only 0.004055 > 0.1 failed`.  The raw growing-mode COUNT of the M2
census at high degree is build-dependent (union-grid conditioning, the known
`deg >= 8` pathology), so the "uncured" precondition the threshold rule is
scored against never materialised on that runner and the spread stayed at
its cured value.  This is shape **S6**, and it is being repaired against
measured cross-build census vectors rather than by widening the 0.1 bar.

### S8.4  One observation, reported not fixed

While laddering the OOP fixture for the closure evidence above, the
convergence claim `test_fff_nv_out_of_plane_converges_fast` makes in its
DOCSTRING -- "fff_nv converges FAST (nearly order-independent) while laurent
climbs slowly TOWARD the same value" -- turned out to be weakly supported on
this fixture.  Distance to the `n_orders=31` value:

| n_orders | 5 | 7 | 9 | 11 | 13 | 15 | 19 | 25 |
|---|---|---|---|---|---|---|---|---|
| `|fff_nv - ref|` | 3.26e-5 | 2.07e-5 | 1.55e-5 | 1.27e-5 | 1.10e-5 | 9.82e-6 | 5.03e-6 | 1.64e-6 |
| ratio to laurent | 1.93 | 1.07 | 0.93 | 0.95 | 1.05 | 1.26 | 0.54 | 0.17 |

`fff_nv` only pulls clearly ahead past `n_orders` ~19.  The test's own
`conv` reference is `fff_nv` at `n_orders=13`, which is itself 1.1e-5 from
the n=31 value -- so `|f7 - conv|` compares two unconverged numbers.

This is NOT a runner-fragility finding: every reading above is bit-identical
on both mounts, and the assertions as written pass deterministically.  It is
a claim-quality question about the fixture, it would need a library-side
convergence investigation to settle, and it is therefore REPORTED here rather
than changed under a runner-pin mandate.

---

## S9.  Verification

**Mounts.**  **W** = Windows py3.14.6 / numpy 2.4.4 / scipy 1.17.1 /
scipy-openblas / 24 cores.  **M** = WSL `/tmp/venv-ci`, py3.12.3 /
numpy 2.5.1 / scipy 1.18.0 / OpenBLAS -- the numpy 2.5-era wheel the failing
ubuntu job runs.  Every run from inside the worktree via `python -m pytest`;
`lumenairy.__file__` confirmed to resolve to the worktree on both.

### The three originally-failing files

| file | W | M | notes |
|---|---|---|---|
| `test_niche_audit_w9_eig_vjp.py` | **32 passed** | -- | was 31; the fail-before split into two tests, nothing removed |
| `test_niche_audit_e_prepared_and_enums.py` | **9 passed** (`-k h2`) | -- | was 5 h2 tests + 2 skips-in-waiting; now 9, none skipped |
| `test_niche_audit_e_prepared_and_enums.py` under `set_max_ram(4.0e9)` | **8 passed** (`-k h2`) | -- | the CI regime EMULATED: the live pricer refuses at that budget, and the section still runs because engagement no longer depends on the live read |
| `test_v5_20_12_rcwa_jones_2d_fff_nv.py` | tests 1-4 passed, remainder RUNNING | reproduced the ORIGINAL failure at 0.005101 before the fix | see below |

The `fff_nv` file's own headline test (2) and the corrected absorptance test
(4) both pass on **W**.  The remaining tests in that file were still running
when this document was written: the box was carrying 40+ concurrent
interpreters from the parallel fix wave, and the crossed-cell test alone
builds a 2450x2450 eigenproblem.  Every bar changed in that file is
independently backed by direct measurement recorded in S3 and S6 -- the
sound-reference ladder (both mounts), the 18-point OOP closure ladder (worst
5.3e-14 against the new 1e-9 bar), and the corrected absorptance arithmetic.
**The remaining run must be completed before this branch is merged.**

### Files repaired by the sweep

Each was verified by the agent that repaired it, on BOTH mounts, ruff clean:

| file | W | M |
|---|---|---|
| `test_niche_r0_byte_budgeted_cache.py` | 29 passed | green |
| `test_audit_s2_10_asm_H_consolidation.py` | 10 passed | 9 passed + 1 `importorskip('numexpr')` |
| `test_v4_16_0_agent_b_multiprocess_storage.py` | **15 passed** (was 14 passed / **1 failed**) | 15 passed |
| `test_fix_runner_oom_2026_08_13.py` | green, incl. BARE process | green, incl. bare process |
| `test_niche_r4_fga_dual_vectorize.py` | green | 10 skipped -- no numba in the CI-proxy venv |
| `test_v5_14_1_rcwa_deferred.py` | green | green |
| `test_audit_optimize.py` | 89 passed | 89 passed |
| `test_niche_audit_r_guards_and_merits.py` | 20 passed | 19 passed + 1 numba capability skip |
| `test_audit_propagation.py` | 101 passed | 101 passed |
| `test_niche_k3_perf.py` | 6 passed | 6 passed |
| `test_niche_d1_tilted_carrier.py` | 33 passed | 33 passed |

`python -m ruff check tests/unit/` -- **All checks passed** across the whole
directory.

### Still outstanding when this document was written

* the `fff_nv` file's full run on **W**, and its run on **M**;
* the thread-count axis ({1, default}) and the py3.10 / py3.11 venvs for the
  three originally-failing files;
* `test_doe_rcwa.py` (the S8.5 closure-bar unification) -- collection and
  ruff pass, and the change is a widening from 1e-8 to 1e-6 that every
  currently-passing site clears trivially, but the file has not been run to
  completion here;
* the groups C, D and the M2 window-contract repair were still verifying.

These are listed rather than glossed because a verification table that
claims runs it did not do is the same class of error this whole document is
about.

---

## S9.1  One follow-up left open: `.test_durations`

This branch renames one test and adds several (the split fail-before in
`w9_eig_vjp`, the two pricing tests in `e_prepared`, the parametrized cache
branches in `r0`).  `.test_durations` still carries the OLD name and has no
entry for the new ones, so `pytest-split` will size those shards from the
suite average rather than from a measurement.

That is the known-open durable fix recorded against the release mechanics
(a stale `.test_durations` has timed out a release-tag verify shard before).
It is NOT fixed here because doing it properly needs a full timed suite run,
which is a separate job from a test-hardening branch -- and doing it by
guessing would put fabricated numbers into a file whose whole purpose is to
hold measured ones.  Flagged so the next release run regenerates it.

---

## S10.  Lessons

1. **A ratio is not automatically build-free.**  If the per-build magnitude is
   still in the numerator, restating a bar as a ratio changes nothing.  The
   2026-08-12 fix diagnosed the mechanism correctly and still shipped a pin
   that failed three tags later.  The test is: *could I state this claim
   without ever naming how big the wrong value is?*  For a defect pin the
   answer is almost always yes -- wrongness is a comparison against truth, and
   truth usually has a derivable resolution.
2. **Check whether the theorem you are asserting exists.**  The fff_nv energy
   bar spent four tags policing a conservation law that the library's own
   docstring says the formulation does not obey.
3. **A poisoned reference is invisible until you measure the reference.**  The
   1-D arm of the fff_nv test read 1.7e-02 on one mount and 1.8e-05 on the
   other for a quantity that is exactly zero in theory.  Nothing in the test
   looked at it.  Where an oracle has a checkable property, check it, and
   search for an operating point where it holds.
4. **`pytest.skip` on a resource precondition is how a section leaves the
   gate.**  Five E-H2 tests stopped running together on exactly the boxes the
   release is gated on, and the failure mode was silence.  If a path needs a
   resource, inject the resource; the library's own `_free_b` hook existed for
   precisely this and was already used correctly one file away.
5. **Sweep for the SHAPE, not the instance.**  Five failures arrived one tag
   at a time over four tags.  One mechanical sweep for the same five shapes
   found ~29 more, including one already red.
