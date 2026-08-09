# Adversarial review -- "exact traced chain" campaign (6dfc79d, c02c37a, 9140cab)

**2026-08-05. REPORT-ONLY.** Reviewed against
`docs/audits/HANDOFF_TRACED_EXACT_2026_08_05.md` as the claims list.
Probes were run from a scratchpad against the working tree at
`feat/pmm-per-layer-roadmap`. No library file was modified.

Bidirectional: this review both REFUTES claimed successes (D1, D2, D3) and
PROVES OUT / CORRECTS claimed failures (the `filelock` write-off is refuted;
the profile-staleness self-report is confirmed).

---

## Verdict summary

| # | Review target | Verdict |
|---|---|---|
| 1 | Default flips (newton_fit, gap_kernel, standoff, pool tiers) | **DEFECT** (D1, D2, D3, D4, D5) |
| 2 | Cache-key fix (`final_leg`) | CONFIRMED-GOOD for `final_leg`; **NEEDS-FOLLOW-UP** (D6) |
| 3 | d6 re-baselines (0.60->0.75, 5.0->4.0) | CONFIRMED-GOOD, one erosion caveat |
| 4 | Exact kernel backend genericity | CONFIRMED-GOOD except odd `N` (**D7**), scope caveat D10 |
| 5 | 9140cab drain-by-completion | **CONFIRMED-GOOD** |
| 6 | Known-weak spots (filelock, partial regression, stale profile) | **DEFECT** -- filelock diagnosis refuted |

---

## D1 -- HIGH -- The warm pool tier is unreachable from a cold start. The
## motivating workload is NOT fixed. **DEFECT**

The commit's stated motivation:

> "a multi-group chain calls `apply_real_lens_traced` once per group and only
> the FIRST is cold, so at design-121's rs=4 (65 536 points/group) every group
> ran serial.  Hence `_POOL_MIN_PIXELS` (cold, 200 000) + `_POOL_MIN_PIXELS_WARM`
> (8 000)."

**Root cause.** `_pool_is_warm` is derived from `_PERSISTENT_POOL is not None`
(`lumenairy/elements/_lens_traced.py:7969-7972`), but `_PERSISTENT_POOL` is
created at exactly one site -- `_get_persistent_worker_pool(n_cpu)` at
`_lens_traced.py:7993` -- which is DOWNSTREAM of the very gate that consults it:

```
7969        _pool_is_warm = (_PERSISTENT_POOL is not None
7970                         and _PERSISTENT_POOL_NWORKERS == n_cpu)
7971        _min_px = _POOL_MIN_PIXELS_WARM if _pool_is_warm else _POOL_MIN_PIXELS
7972        if n_cpu <= 1 or n_total < _min_px:
7973            return _invert_newton(Xw, Yw, sub_progress=sub_progress)
...
7993            ex = _get_persistent_worker_pool(n_cpu)
```

`grep -rn "_get_persistent_worker_pool" lumenairy/ --include=*.py` confirms
line 7993 is the only call. So the pool can only ever be created by a call that
has ALREADY cleared the 200 000 COLD bar, and a process whose calls are all
below 200 000 never warms.

**Runtime proof** (fresh process each scenario, `n_workers=4`, spy on the
pool-dispatch site):

```
_POOL_MIN_PIXELS=200000  _POOL_MIN_PIXELS_WARM=8000

A: N=512, rs=2, 3 groups  (65 536 Newton pts/group -- the commit's own case)
   pool created? False   pool-dispatch events: 0
   => ALL 3 GROUPS SERIAL -- warm tier never reached

B: N=1024, rs=2, 2 groups (262 144 pts) -- clears the COLD bar
   pool created? True    pool-dispatch events: 2

C: pool now warm -> repeat A (65 536 pts) in the SAME process
   pool-dispatch events: 2
   => POOLED (the warm tier works, but ONLY after a >=200k call)
```

The mechanism is correct; its trigger is unreachable. At 65 536 points per
group with nothing larger in the process, every group still runs serial --
byte-for-byte the pre-fix behaviour the change was written to remove.

**No test covers the warm tier.** `tests/unit/test_niche_newton_pool_both_fits.py`
states in its module docstring: "These tests deliberately size the grid past
`_POOL_MIN_PIXELS` so the pool actually engages". Its premise guard
`test_the_pool_threshold_is_actually_exceeded` asserts
`_POOL_MIN_PIXELS_WARM <= _POOL_MIN_PIXELS`, which is satisfied by construction
and proves nothing about reachability.

**Fix direction (not applied):** either seed the pool when
`n_total >= _POOL_MIN_PIXELS_WARM` and a pool has been requested at least once
in this process (a separate "has been asked" flag, not "is alive"), or make the
first call above the warm bar pay the spawn deliberately. Add a reachability
test that starts cold at 65 536 points and asserts the SECOND group pools.

---

## D2 -- HIGH -- `_FOCUS_STANDOFF_ZR` 6.0 -> 0.8: the optimum is GRID-EXTENT
## dependent, not NA-independent. The generality claim is refuted. **DEFECT**

The claim (commit message and `carrier.py:185-190`):

> "The optimum is NA-INDEPENDENT... Confirmed by measurement -- the best f is
> 0.75-1.0 at every NA from 0.10 to 0.40, while the PENALTY for sitting at
> f = 6.0 grows steeply with NA (0.26% at NA 0.10, 21.2% at NA 0.40)."

**Their measurement reproduces -- in one regime.** Scanning FWHM error against
an analytic complex-q truth, at grid half-extent 3-4x the beam radius I get
f=0.8 -> 0.61% and f=6.0 -> 11.3%, essentially their 0.60% / 8.84%.

**But the invariant is wrong.** Holding NA fixed at 0.278 and varying only the
grid half-extent in beam radii moves both the optimum and the achievable error
by 20x:

```
FWHM error (%) vs standoff factor f, NA = 0.278, N = 512
  ext    f=0.6   f=0.8   f=1.0   f=1.5   f=2.0   f=3.0   f=4.0   f=6.0   f=8.0   best
  1.5   13.379  13.255  13.255  14.840  14.241  14.332  14.241  13.379  27.170   1.0
  2.0   10.388  10.566   8.258   6.346   5.771   5.838   6.881  10.110  13.982   2.0
  3.0    3.513   0.608   0.633   1.886   2.824   4.885   7.077  11.319  15.188   0.8
  4.0    0.256   0.604   0.939   1.813   2.791   4.899   7.080  11.316  15.192   0.6
  6.0    0.420   0.665   0.955   1.813   2.791   4.899   7.080  11.316  15.192   0.6
 10.0    0.419   0.665   0.955   1.813   2.791   4.899   7.079  11.316  15.192   0.6
```

At half-extent 2.0 (a truncated but entirely ordinary pupil) **the new default
0.8 is WORSE than the old 6.0** (10.566% vs 10.110%), and both are ~2x worse
than the local optimum at f=2.0.

Varying `N` at fixed extent changes nothing (10.394 / 10.388 / 10.387 / 10.386
at N = 256 / 512 / 1024 / 2048), so the driver is aperture truncation / grid
extent -- not sampling, and not NA.

**The NA sweep also fails at NA >= 0.2** once the grid is only 2x the beam:

```
FWHM error (%), grid half-extent 2x beam radius, N = 512
  NA      f=0.6   f=0.8   f=1.0   f=1.5   f=2.0   f=3.0   f=4.0   f=6.0   f=8.0   best
  0.030   2.028   1.720   6.302   5.199   3.724   3.189   2.563   2.278   2.144   0.8
  0.100   2.169   1.870   6.656   5.348   3.955   3.337   2.795   2.508   2.430   0.8
  0.200   4.512   4.600   7.258   5.668   4.499   3.984   3.847   4.599   5.929   4.0
  0.278  10.388  10.566   8.258   6.346   5.771   5.838   6.881  10.110  13.982   2.0
  0.400  23.789  23.681  12.149   9.625  10.831  13.251  16.467  23.784  30.626   1.5
```

"Best f is 0.75-1.0 at every NA from 0.10 to 0.40" holds at NA 0.03-0.10 and
fails at 0.20, 0.278 and 0.40. At NA 0.40 the new and old defaults are a wash
(23.68% vs 23.78%).

**The code already names the right invariant and then does not use it.**
`carrier.py:210-213` and `218-222`: "`standoff_min = 2 * _BRIDGE_FIT_MARGIN *
w0 * |z_focus| / (Nx * dx)` ... Note this floor depends on the INPUT GRID
EXTENT, not on NA." That is the correct scaling; the default resolver ignores it
(`_FOCUS_STANDOFF_BRIDGE_SAFETY` is defined at `carrier.py:223` and **never
read** -- dead code).

**Answer to "is there a test at intermediate NA?": no.** `test_niche_d6` and
`test_niche_d2_chain_multi` each pin one geometry. Nothing sweeps NA or grid
extent against the standoff.

---

## D3 -- HIGH -- The standoff flip shrinks the replica period 7.5x for every
## user of a public readout, with no guard at K=1 and none at all on the direct
## API. **DEFECT**

The Bluestein reconstruction period scales linearly with the standoff, so the
flip divides it by exactly 7.5. Measured on a benign NA 0.03 case:

```
Bluestein period: 0.8 zR -> 57.95 um ;  6.0 zR -> 434.60 um   (ratio 7.50x)
```

Accuracy of `carrier_referenced_focus_readout` vs the analytic complex-q truth,
same beam, widening the requested output window:

```
  win_um  relL2 |F| @0.8zR  relL2 |F| @6.0zR   nwarn@0.8  nwarn@6.0
    16.0        1.9688e-03        1.0649e-04           0          0
    32.0        2.6462e-03        1.4194e-04           0          0
    48.0        3.5938e-03        1.4245e-04           0          0
    64.0        4.8757e-03        1.4250e-04           1          0
    96.0        3.5496e-01        1.4252e-04           1          0
   128.0        2.5535e+00        1.4256e-04           1          0
   256.0        4.8678e+00        1.4261e-04           1          0
```

Two separate regressions in one table:

1. **Small windows: the new default is 14-34x WORSE here** (2.0e-3 vs 1.1e-4 at
   16 um). This is the D2 grid-extent effect again, on a second metric.
2. **Windows past ~1 period: catastrophic.** relL2 goes 0.35 -> 4.87 while the
   old default stays flat at 1.4e-4 out to 256 um.

**The guards do not cover this.**
* `carrier_referenced_focus_readout` (public, in `carrier.py:__all__`) has NO
  replica guard at all -- it only optionally fills the private `_period_out`
  (`carrier.py:1983-1986`). A direct caller gets nothing.
* Through the chain, `carrier.py:7171-7174`: "the guard is a MULTIPLEXING guard:
  at K = 1 there is no neighbouring frame to contaminate ... so it downgrades to
  a warning and 'auto' keeps the requested field of view." So single-congruence
  users are warned, not refused.
* The only thing that fires above is the downstream
  `angular_spectrum_propagate_mft` window-vs-period `UserWarning`, which any
  upstream `filterwarnings('ignore')` silences -- the exact failure mode the
  library's own `on_replica` note (`carrier.py:7161-7164`) says it introduced
  `on_replica='error'` to avoid.

**The campaign's own tests document the regression and work around it.**
`tests/unit/test_niche_d2_chain_multi.py`:
* `_TILE` 256 -> **120**: "at the old 6.0 zR a 256 tile fitted, at the accuracy
  optimum 0.8 zR the period is 0.1739 mm and the guard allows <= 124."
* Three fixtures now pass an explicit `standoff=1.3 * window * fd / (N*dx)`.
* `test_result_grid_convention_is_the_readout_convention` comment: "if the
  256 um readout window exceeds one Bluestein period the outer window fills
  with replicas and the centroid walks (**measured 58.8 um** at the 0.8 zR
  accuracy standoff) ... At K = 1 the replica guard is deliberately permissive
  ... so nothing refuses the run for us here."

Consequence for the handoff's Sec 6 claim: "Green with the new defaults:
`test_niche_d2_chain_multi` (38)" is true only because three of its fixtures
were changed to OPT OUT of the new default and the tile was cut by 2.1x. That
is a coverage reduction, not a clean pass, and it is not disclosed as one.

---

## D4 -- MEDIUM -- `gap_kernel` is unvalidated on both public entry points; a
## typo silently selects the PARAXIAL kernel. **DEFECT**

`propagate_traced_carrier_chain` and `..._multi` validate (`carrier.py:852`,
`carrier.py:948`). `propagate_carrier_referenced` and
`carrier_referenced_focus_readout` -- both public, both given the new
`gap_kernel` argument by this commit -- do not. The resolution chain in
`_carrier_step_fast` is:

```
if gap_kernel == 'auto':   gap_kernel = 'exact'
if gap_kernel == 'exact' and xp is not np:   ...exact xp...
elif gap_kernel == 'exact':                  ...exact numpy...
elif xp is np:                               ...FRESNEL...      <-- catch-all
```

so anything that is not literally `'auto'` or `'exact'` falls through to
Fresnel. Measured:

```
gap_kernel='auto'     -> EXACT
gap_kernel='exact'    -> EXACT
gap_kernel='fresnel'  -> FRESNEL
gap_kernel='exsct'    -> FRESNEL   (dist_to_fresnel = 0.00e+00)
gap_kernel='EXACT'    -> FRESNEL
gap_kernel=None       -> FRESNEL
```

A user who mistypes the knob gets the paraxial transport this entire campaign
exists to remove, silently. This is the same defect class as the
`on_readout_windo` typo already fixed under niche C1 (`carrier.py:7254-7258`).

---

## D5 -- MEDIUM -- The `newton_fit` flip silently disables a guard and two
## shipped features, unremarked. **DEFECT**

The commit justifies `polynomial -> spline` purely on speed: "Accuracy is a tie
against the ray oracle ... so the default is chosen on speed". Three behavioural
consequences are not mentioned anywhere in the commit, the CHANGELOG or the
handoff:

1. **The fit-domain disc restriction does not apply to spline.**
   `_lens_traced.py:7297`: `if _fit_r_max is not None and newton_fit != 'spline':`
   The library's own note at `1850-1852` says spline "skips the polynomial fit
   and its disc restriction entirely (the disc block is gated on
   `newton_fit != 'spline'`)". So `fit_radius_beam_factor` and the D1/D7
   weighting machinery are inert on the new default.

2. **`DECENTRED_FIT_ARBITER` (niche C11, shipped `True` at `_lens_traced.py:1976`)
   becomes dead code for the default CPU configuration.** Its two use sites,
   `6745` and `7334`, are both gated on `newton_fit != 'spline'` (7334 is inside
   the 7297 block). Per project memory this arbiter shipped in v5.32.1; the flip
   removes it from the default path one release later, with no note.
   `DECENTRED_FIT_PREDICTOR` (niche C12) is gated the same way.

3. **"A tie" is true only inside the fit disc.** `_lens_traced.py:3190-3199`
   records the two backends scored pointwise against the exact skew ray trace:

   ```
   zone (about the beam)   polynomial      spline
   inside the 2 w disc      0.000 um       0.007 um
   skirt 2-4 w              5.608 um       0.006 um
   entrance aperture rim   15.079 um       0.002 um
   ```

   The backends differ by up to 15 um outside the disc. Spline is the BETTER
   map there, so this flip is plausibly a net improvement -- but it is an
   unassessed change to which guards run, sold as a pure speed change.

**Not a defect (checked):** `newton_fit` IS validated (`_lens_traced.py:7646`
raises on anything but `'spline'`/`'polynomial'`, with `'auto'` resolved first
at `5567-5568`), `use_gpu=True` correctly refuses spline (`7597-7600`), and
`scipy>=1.7` is a HARD core dependency (`pyproject.toml:51`) -- so the spline
default cannot break a scipy-less or GPU-absent install.

---

## D6 -- MEDIUM -- The cache-key fix is correct; the same disease is present on
## three other axes, two of which this commit changed. **NEEDS-FOLLOW-UP**

**The `final_leg` fix itself is sound.** The filename SHAPE changed:

```
old:  _chainA_{n}_{dx0*1e9:.0f}nm_rs{rs}.npz
new:  _chainA_{n}_{dx0*1e9:.0f}nm_rs{rs}_{final_leg}.npz
```

This is not an appended field that a stale file could still match -- pre-fix
caches cannot satisfy the new pattern. On disk,
`validation/repro_traced_carrier_121/_chainA_1024_2000nm_rs4.npz` (Jul 30 14:53)
and two siblings are now orphaned and will never be read again. Old caches ARE
genuinely invalidated. CONFIRMED-GOOD.

**But the same class of bug remains, and this commit made it live.** `chain_a`
does not take `gap_kernel` or `newton_fit`; it inherits the library defaults --
**both of which this commit flipped**. Neither is in the key, and there is no
version stamp or content hash. So:

* a `_chainA_*_exact.npz` written between the cache-key fix and the default flip
  holds a **Fresnel-kernel, polynomial-fit** field and will be silently reused
  as if it were exact/spline -- the identical failure mode the fix was written
  for, one commit later;
* `nw` (`n_workers`) is a `chain_a` parameter and is not in the key either;
* `dx0 * 1e9:.0f` rounds to integer nm, so two distinct `dx0` differing by
  <0.5 nm collide.

Recommend keying on a short hash of `(final_leg, gap_kernel, newton_fit, nw,
lumenairy.__version__)` rather than enumerating fields.

---

## D7 -- LOW -- `_exact_tf_2d_xp` is order-unity wrong for ODD `N`, creating the
## very NumPy-vs-backend divergence the change was made to prevent. **DEFECT
## (inherited)**

The commit's rationale: "NumPy-only would have meant other backends silently
running the PARAXIAL kernel -- a correctness difference between backends."

```
_exact_envelope_tf_step (NumPy) vs _exact_tf_2d_xp, same inputs, xp=bld=np:
  N=  64 tilt=(0.0,0.0)   relL2 = 3.806e-16   OK
  N=  65 tilt=(0.0,0.0)   relL2 = 1.069e+00   MISMATCH
  N= 128 tilt=(0.05,0.02) relL2 = 4.811e-16   OK
  N= 127 tilt=(0.05,0.02) relL2 = 6.050e-01   MISMATCH
```

Against an independent plain-numpy exact-ASM oracle:

```
  N=  64  numpy-vs-oracle=3.269e-12   xp-vs-oracle=3.269e-12
  N=  65  numpy-vs-oracle=3.340e-12   xp-vs-oracle=1.071e+00
  N= 127  numpy-vs-oracle=2.799e-12   xp-vs-oracle=6.050e-01
  N= 128  numpy-vs-oracle=3.051e-12   xp-vs-oracle=3.051e-12
```

**Root cause.** `_freq_1d_bld` (`carrier.py:296-303`) builds
`(arange(N) - N/2)/(N*d)`, which is HALF-INTEGER for odd `N`; `ifftshift` of
that does not reproduce `fftfreq`:

```
N=5   got: [-0.5  0.5  1.5 -2.5 -1.5]
      ref: [ 0.   1.   2.  -2.  -1. ]
```

**Inherited, not introduced.** `_fresnel_tf_2d_xp` via `_freq_sq_1d_bld`
(`carrier.py:289-293`) has the identical break -- `fresnel_tf_propagate` vs
`_fresnel_tf_2d_xp` gives relL2 1.068e+00 at N=65. So this is a pre-existing
library-wide odd-`N` defect faithfully mirrored into a new function. It is
reported here because (a) the new function's whole justification is backend
parity, and (b) no test in `test_niche_exact_gap_kernel.py` uses an odd `N`.

Low severity only because even grids are the overwhelming norm; nothing
enforces them.

---

## D8 -- LOW -- Committed source contains a stale comment that directly
## contradicts the shipped value. **NEEDS-FOLLOW-UP**

`lumenairy/propagators/carrier.py:191-195`, immediately above the line that
does the opposite:

```
191  # NOT flipped to the measured optimum: setting this to 0.8 broke 5 tests and
192  # errored 8 more in test_niche_d2_chain_multi / test_niche_d6_exact_tilted_leg
193  # (2026-08-05).  The readout default is load-bearing for the multi-congruence
194  # tiling and replica-regime logic, which is calibrated against the longer leg.
195  # Callers who want the accuracy above should pass ``standoff`` explicitly.
...
205  _FOCUS_STANDOFF_ZR = 0.8
```

Two comment blocks describing the same constant were left stacked (the
measurement table appears twice, at 179-190 and 196-204), and the
`_FOCUS_STANDOFF_BRIDGE_SAFETY` block is likewise duplicated (206-213 and
214-222) for a constant that is **never read** anywhere in the repo.

This is a half-merged edit from an abandoned attempt. Its content is also
contemporaneous evidence for D3: the flip did break 13 tests across exactly the
two files that the commit then edited (fixture opt-outs in d2, margin
re-baselines in d6).

---

## D9 -- LOW -- Stale pool documentation left behind

Three sites still describe the pre-commit behaviour:

* `_lens_traced.py:593` -- "pool path (>=200k points with `newton_fit='spline'`)"
* `_lens_traced.py:5016` -- "`>=200k` Newton points with `newton_fit='spline'` on the CPU path"
* `_lens_traced.py:7731` -- "process pool engaged (>=200k points, newton_fit='spline', CPU)"

All three contradict both the two-tier threshold and the both-fits worker this
commit added. The main `n_workers` docstring (4879-4898) WAS updated; these were
missed.

---

## D10 -- LOW -- The exact-kernel oracle validation never exercises a SCALED
## Sziklas-Siegman step. **NEEDS-FOLLOW-UP (scope caveat, honestly disclosed)**

The 1e-12/1e-13-vs-oracle tests in `test_niche_exact_gap_kernel.py` call
`_exact_envelope_tf_step` directly -- an unscaled, `m = 1` transfer step. The
only tests that go through `propagate_carrier_referenced` (lines 239-242) use
`R=0.5, z=1e-3`, i.e. `m = 1.002`, and assert only "the default equals explicit
`'exact'` and differs from `'fresnel'`" -- no oracle.

So the headline accuracy number characterises the KERNEL, not the chain step
that ships it. The spec `SPEC_EXACT_SPHERE_GAP_TRANSPORT_2026_08_05.md` is
explicit and correct about this ("substituting the exact kernel gives an exact
propagator on an approximate frame"), and handoff Sec 2 discloses it -- so this
is a scope gap, not a misrepresentation.

Worth flagging for follow-up: a paraxial kernel is *consistent* with a paraxial
frame, and their errors can partially cancel. Substituting an exact kernel into
a paraxial frame is not guaranteed to be monotonically better, and nothing in
the suite measures the combination against an absolute reference at `m` far
from 1. The spec's own acceptance criteria (>=3 designs, >=2 wavelengths, an
ABSOLUTE reference, a negative control) are the right bar and are not yet met.

---

## Target 5 -- 9140cab drain-by-completion: **CONFIRMED-GOOD**

* **Determinism holds.** Results are stored by the congruence's own index
  (`out[k] = (field, stages, msgs)`) and consumed by `for k in range(K)` after
  the loop, so completion order cannot reach the accumulation order. The only
  order-sensitive thing that changed is the `progress` callback, which the
  commit message correctly identifies and which a shipped test pins
  ("progress fires exactly once per congruence").
* **The 0e3f66e spawn pin SURVIVES.** `_mp.get_context('spawn')` was hoisted
  OUT of the `try` block (necessary: `_pickle` is named in the `except` clause,
  which is evaluated at raise time, so binding it inside the guarded block would
  turn any early failure into a `NameError`), and is still passed as
  `mp_context=_ctx` to `ProcessPoolExecutor`. Verified in the diff.
* Failure handling improved on two axes: the worker's exception is now read as a
  VALUE via `fut.exception()` (so the broad `except` could be narrowed to the
  pool's own typeable modes), and pending futures are cancelled before `break`
  so the `with` exit does not block on stragglers.

No defect found.

---

## Target 3 -- d6 re-baselines: **CONFIRMED-GOOD**, one erosion caveat

**Direction of each bar (both LOOSENED):**

```
- assert m_ex['fwhm'] < 0.60 * m_px['fwhm']        ->  < 0.75 *     (weaker)
- assert m_ex['ee'][2.0] > 5.0 * m_px['ee'][2.0]   ->  > 4.0 *      (weaker)
```

**But the adjudication is correct.** Every ORACLE-anchored assertion in
`test_exact_beats_paraxial_for_a_tilted_congruence_against_the_oracle` was left
untouched:

```
assert abs(m_ex['fwhm'] / orc['fwhm'] - 1.0) < 0.15      # exact vs ray oracle
assert m_ex['ee'][2.0] > 0.90 * orc['ee'][2.0]           # exact vs ray oracle
assert m_ex['ee'][4.0] > 0.97 * orc['ee'][4.0]           # exact vs ray oracle
assert m_px['fwhm'] > 1.25 * orc['fwhm']                 # paraxial vs oracle
assert m_px['ee'][2.0] < 0.25 * orc['ee'][2.0]           # paraxial vs oracle
assert abs(m_px['peak_off'][0]) > 4.0e-6                 # absolute
```

So the exact leg's agreement IS asserted comparatively against the independent
ray oracle (`_oracle_on_grid`), not pinned to 3.1500 um absolutely, and only the
two DERIVED exact-vs-paraxial contrast ratios moved. That is the right procedure
for a re-baseline: the adjudicator was not touched.

**Caveat -- third erosion of the same axis.** `m_px['fwhm'] > 1.25 * orc['fwhm']`
has now seen the measured ratio fall 3.19x -> 1.857x -> **1.476x** against a
1.25 floor: headroom is down from 48% to 18%. The test's own docstring already
calls this "ERODING AXIS, floored not chased". One more improvement to the
paraxial leg's spot placement retires the fail-before half of this test. That
should be converted to a discriminator that does not erode (the EE2 and
peak-offset arms have not) before the next such change.

---

## Target 6 -- known-weak spots

### The `filelock` write-off is REFUTED (in this environment). **DEFECT**

Claim: "the single failure (`test_audit_io ... complex64`) is a **pre-existing
missing `filelock` dependency**, confirmed by forcing polynomial and seeing it
fail identically."

Measured:

```
python -c "import filelock" ->  OK 3.25.2
pyproject.toml:81   hdf5 = ["h5py>=3.0", "filelock>=3.0"]
pyproject.toml:101  zarr = [..., "filelock>=3.0"]
pyproject.toml:153  all  = [..., "filelock>=3.0"]

pytest tests/unit/test_audit_io.py -k complex64   ->  2 passed
pytest tests/unit/test_audit_io.py                ->  42 passed
pytest tests/unit/test_audit_io.py -n 4           ->  42 passed
```

`filelock` is installed AND declared in three extras. The failure does not
reproduce serially or under xdist. Two problems with the diagnosis:

1. It is not a missing dependency here, so whatever failed was environmental,
   transient, or something else entirely -- and remains unexplained.
2. The stated control ("forcing polynomial and seeing it fail identically")
   establishes only that the failure is INDEPENDENT of `newton_fit`. It does not
   establish that it is pre-existing, and it cannot distinguish an
   environmental/flaky failure from a real one. Per the project's own
   "flaky = bad math" rule this should not have been written off.

**Action:** re-run and capture the actual traceback before calling it
pre-existing.

### Partial regression (204 passed, then stopped) -- confirmed genuinely incomplete

Self-report accurate. Not verifiable further here: a full-suite run is out of
scope while another agent is working in the tree.

### Stale profile -- confirmed

`_poly` was 39.6% of the only profile on record and is off the CPU default path
after the flip. Any optimisation derived from that profile must be re-derived,
exactly as the handoff states.

---

## What I could NOT verify, and why

* **The JAX 3-5e-16 backend claim on a real JAX device.** A shipped test does
  exist -- `test_jax_backend_reproduces_the_numpy_exact_kernel`, with
  `pytest.importorskip('jax.numpy')` and `jax_enable_x64` -- so the claim is not
  a one-off probe. I confirmed the NumPy-vs-`_exact_tf_2d_xp` parity at
  3.8e-16 / 4.8e-16 with `xp=bld=np` (even `N`), and all 29 tests in
  `test_niche_exact_gap_kernel.py` + `test_niche_newton_pool_both_fits.py` pass
  here. I did not confirm the number on an actual JAX backend.
* **CuPy path.** No GPU on this box. `_exact_tf_2d_xp` is exercised only with
  `xp=np` in the shipped tests, so the CuPy-specific dtype/device path
  (`_to_dev`, `H.real[...] = ...` in-place assignment on a CuPy array) is
  untested anywhere.
* **complex64 through the exact kernel.** `test_niche_exact_gap_kernel.py`
  mentions `complex64` once. The mod-2pi fold in `_tf_phase_to_H` is shared with
  the Fresnel path, but the exact kernel's larger dynamic range (the docstring
  itself notes "`k z` is large and the exact root carries full precision") is
  not pinned at complex64. Handoff Sec 4 flags complex64 through a 6-group chain
  as unmeasured; that remains true.
* **Full-suite regression.** Deliberately not run (another agent is active in
  `lumenairy/elements/pmm/stack.py` and
  `tests/unit/test_v5_13_0_pmm_tapered.py`).
* **The 121 chain-grid convergence study.** Correctly self-reported as not
  working; not re-attempted.
* **`test_auto_resolves_to_exact_on_every_backend`** guards by searching the
  source text for the literal `"'exact' if xp is np else 'fresnel'"`. That
  catches one specific regression spelling and nothing else; noted, not counted
  as a defect.

---

## Ranked defect list

| # | Sev | One-line root cause |
|---|---|---|
| D1 | HIGH | `_pool_is_warm` reads `_PERSISTENT_POOL`, which is only created downstream of the gate that reads it -- warm tier unreachable cold; the 65k/group case it was written for still runs serial. |
| D2 | HIGH | Standoff optimum was calibrated at one grid extent and generalised on the wrong invariant (NA); it is extent-driven, and at extent 2x beam the new default is worse than the old. |
| D3 | HIGH | 7.5x shorter standoff shrinks the Bluestein period 7.5x; public `carrier_referenced_focus_readout` has no replica guard and the chain's guard only warns at K=1. |
| D4 | MED | `gap_kernel` unvalidated on `propagate_carrier_referenced` / `carrier_referenced_focus_readout`; any typo falls through the `elif` chain to Fresnel. |
| D5 | MED | `newton_fit` default flip crosses three `newton_fit != 'spline'` gates, silently retiring the fit-domain disc restriction and the C11 arbiter / C12 predictor from the default path. |
| D6 | MED | Cache key fixed for `final_leg` only; `gap_kernel`, `newton_fit`, `nw` and library version remain unkeyed -- and two of those were flipped by this same commit. |
| D7 | LOW | `_freq_1d_bld` (mirroring pre-existing `_freq_sq_1d_bld`) is half-integer for odd `N`, so `_exact_tf_2d_xp` diverges from the NumPy kernel by order unity there. |
| D8 | LOW | Half-merged comment: a "NOT flipped to 0.8" note sits directly above `_FOCUS_STANDOFF_ZR = 0.8`; two comment blocks duplicated; `_FOCUS_STANDOFF_BRIDGE_SAFETY` is dead. |
| D9 | LOW | Three pool docstrings still claim spline-only / single-200k threshold. |
| D10 | LOW | Exact-kernel oracle tests are all at `m ~ 1`; the kernel is never adjudicated inside a strongly-scaled Sziklas-Siegman step. |
