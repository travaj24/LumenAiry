# Decisions reserved for the maintainer, with the measurements that inform them

Status: DRAFT 2026-09-19, assembled at the close of Wave 5 from the landed work
packages and their independent verifications.  Five items (marked PENDING) are
still being measured by the Wave 5 agents and will be filled in before the
5.48.0 tag.  Nothing in this document was decided by the agents; every item
here either moves a default, changes what an unmodified call returns, or needs
information only the maintainer has.

## How to read this document

Each item states, in this order: what the setting is in plain language; what
was measured and by whom (the work package `WP-*` that built it and the
independent verification `VERIFY-*` that re-measured it, both under
`fixes/`); what each choice costs; a recommendation with its confidence; and
what saying "yes" would require (usually a re-pinning of the fixtures that
move, which is bounded work with a known size).  A DEFAULT MOVE is a change to
what a call returns when the caller passes nothing; every default move in this
library carries a Migration note.  A FIXTURE is a test's stored optical
configuration; "fixtures move" means their recorded numbers would have to be
re-recorded, which is a chore, not a defect.  BYTE-IDENTICAL means the returned
array is the same to the last bit, proved against a `git archive` of the
parent commit rather than against a working copy.

Items are grouped by the kind of decision: numerical defaults (section 1),
diagnostics ownership (section 2), catalogue and environment data the library
cannot obtain by itself (section 3), and the items still being measured
(section 4).

---

## 1. Numerical defaults that measured better but move fixtures

### 1.1 `apply_aperture(edge='gray')` as the default (WP-B11 item 9)

What it is.  A circular aperture on a square grid is a staircase.  `'hard'`
sets each pixel fully inside or fully outside; `'gray'` gives boundary pixels
their covered fraction (with `edge_samples=4` sub-samples per axis).

Measured (WP-B11 sec. 2.9, both spatial kernels, lambda 633 nm, radius 100 um,
window 512 um, on-axis error against the closed form):

| N | RS kernel, hard | RS kernel, gray | HF quadrature, hard | HF quadrature, gray |
|---|---|---|---|---|
| 128 | 8.30e-03 | 1.48e-03 | 2.77e-02 | 1.44e-02 |
| 256 | 3.35e-03 | 3.61e-04 | 1.11e-02 | 3.49e-03 |
| 512 | 3.42e-04 | 8.10e-05 | 1.15e-03 | 8.57e-04 |
| 1024 | 5.27e-04 | 2.50e-05 | 1.76e-03 | 2.11e-04 |
| convergence order | none reliable (negative last step) | second order | none reliable | second order |

The hard edge has no convergence order at all (the staircase area error does
not shrink monotonically); the grey edge is second order.  The gain at N = 1024
is 21x (RS) and 8x (HF).  Cost: the extra indicator evaluations touch only
boundary pixels, 0.076x of one full-grid pass at N = 256 and 0.018x at 1024;
`edge_samples=4` is already the knee (8 and 16 buy nothing).

What moves: every hard-aperture fixture, relative L2 of 7.7e-03 (N = 256) down
to 1.2e-03 (N = 1024).

Recommendation (high confidence): flip to `'gray'`.  It buys a convergence rate
rather than a constant, for a few per cent of one mask build.  Saying yes
needs one re-pinning pass over the hard-aperture fixtures and a Migration note;
`edge='hard'` stays reachable by keyword.

### 1.2 `transport='collins'` as the carrier chain's default (WP-B4, VERIFY-B4)

What it is.  The traced carrier chain moves a beam envelope from one plane to
the next.  `'sziklas'` (shipped default) is a two-step scaled transform that
must stop short of a focus because its grid collapses there; `'collins'` is the
one-step Collins integral that lands on the target plane in one step, focus
included.

Measured (VERIFY-B4 rows 1, 2, 13): the Collins quadrature is exact to the
transform's rounding on every leg tried, including a focus crossing (relative
L2 1.4e-15 at the focus, 8.9e-15 to 1.7e-14 across a 1 mm sweep through it
under `gap_kernel='fresnel'`); the shipped `'sziklas'` cannot reach the focus
cell at all (it raises on `R_carrier == 0`).  The default is byte-identical to
5.46.0 on a 60-entry fixture set.  What a flip would retire: the whole
focus-standoff machinery (`_default_focus_standoff` and five siblings), which
exists only because the two-step transport cannot land on a focus.  What it
would move: exactly the legs with `N dx^2 <= lambda |z_eff|`, which is
checkable per design without running anything (VERIFY-B4 F2).

Recommendation (medium confidence): flip in a MINOR release, after item 1.5
below is settled, because the near-focus behaviour of `gap_kernel='auto'` is
where `'collins'` is met (it operates at small `A` by design).  Saying yes
needs a CuPy/JAX arm first (WP-B4 sec. 5: the host-only transport is not a
default every backend reaches) and the Migration note.

### 1.3 `sphere_normal='analytic'` as the ray tracer's default, and whether its domain clamp goes (WP-B9, VERIFY-B9)

What it is.  For a pure spherical surface the surface normal has a closed
form; the generic route differentiates the sag numerically.  `'analytic'` is
opt-in today.

Measured (WP-B9 sec. 6, VERIFY-B9 sec. 3): the analytic normal is within 4 ULP
of a 60-digit oracle and never worse than the generic route; the whole trace
speeds up by 1.08x to 1.18x (median 1.13x); the second opt-in,
`renormalize='exit'`, is worth a further 1.03x to 1.10x.  Two pins would have
to be restated first (WP-B9 sec. 5 items 2 and 3: the `ModalAsymptoticStillBitEqual`
arms, whose 4 % margin on a bimodal quantity VERIFY-B9 calls "a coin", and the
`w6_a2` bit pin).

The clamp is a separate question.  Both routes refuse a ray landing between
0.99995 |R| and |R| on a pure sphere (a real hit, killed as `RAY_NAN`).  WP-B9
proposed dropping the clamp on the grounds that the closed form is
well-conditioned there.  VERIFY-B9 sec. 3.2 measured that this is false: the
relative error of `sqrt(1 - u)` is bounded below by `eps/2 * u/(1-u)`, which
is 1.1e-12 (about 5000 ULP of the axial component) at u = 0.9999 for BOTH
routes, because the information is not in the inputs.  Dropping the clamp is
therefore a trade: a grazing normal carrying ~1e-12 relative error instead of a
dead ray.  Either way the generic route's clamp must move with it, or the two
gates diverge by a decade of height instead of 1 ULP (VERIFY-B9 sec. 3.3:
vignetting does move under `'analytic'` today).

Recommendation: (a) flip `sphere_normal` first (bigger win, better accuracy),
after the two pins are restated, then consider `renormalize` (medium
confidence); (b) keep the clamp unless a vignetting sweep shows the false
kills matter for a real prescription, and if it goes, drop it on BOTH routes
with the honest rationale above (low confidence either way; it is a vignetting
change, not an accuracy one).

### 1.4 The caustic route: which member serves a caustic, and the `_ABERRATION_MAX_RAD` budget (WP-B7b, WP-B12, VERIFY-B12)

What it is.  Near a caustic `apply_real_lens_auto` picks one of three members:
the frozen-Gaussian swarm `'fga'`, the `'phase_screen'` model, or the
`'traced'` lens.  Since 5.47.0 the route prefers the phase screen inside a
low-aberration class bounded by `_ABERRATION_MAX_RAD = 2.0` rad, and WP-B12
repaired the FGA member's reference plane, which changed FGA's standing.

Measured after the repair (VERIFY-B12 sec. 10, on the verifier's own optics,
against an exact diffraction oracle):

| regime | screen | FGA | traced | cost ratio FGA / screen |
|---|---|---|---|---|
| estimate below ~0.09 rad | best (0.9980 to 0.9999) | 0.9979 to 0.9997 | 0.9979 to 0.9996 | 100x to 330x |
| 0.09 to 0.40 rad (flat last surface) | 0.9915 to 0.9996 | 0.9946 to 0.9996 | 0.9934 to 0.9993 | 100x to 330x |
| aspheric fixture at its caustic, 0.658 rad | 0.9993 | 0.9998 | 0.9997 | 327x |
| meniscus at its caustic, 0.022 rad | 0.9993 | 0.9991 | 0.9988 | 258x |

The screen's error grows monotonically with the estimate and crosses 1e-3
between 0.086 and 0.156 rad on the verifier's optic, between 0.12 and 0.47 rad
on WP-B12's; it crosses 1e-2 near 0.40 rad and between 0.47 and 0.89 rad
respectively.  So by 2.0 rad the screen is one to two decades past the accuracy
the budget is meant to guarantee, but the boundary is fixture-dependent by a
factor of two to three.  `'traced'` at its shipped `ray_subsample=8` refuses
every fixture in both studies (WP-B12 open item 3); forced to 2 it is never the
best member on the verifier's optics.  WP-B7b's earlier escalation (traced
0.9995 vs screen 0.9991 on one fixture) is therefore a property of that lens.

Recommendation (medium confidence): keep the route where 5.47.0 put it (the
screen, as a cost choice: 100x to 330x cheaper for at most ~5e-4 of fidelity
inside the class) and do NOT retune `_ABERRATION_MAX_RAD` to the "~0.5 rad" one
fixture suggested; a retune needs a fixture-family sweep.  The `aberrated`
condition stays.  Separately, `'traced'`'s sampling guard should be looked at
by its owner, since the shipped `ray_subsample` refuses every caustic fixture
built so far.

### 1.5 `gap_kernel='auto'` near a focus (VERIFY-B4 F3; near-focus table PENDING, Wave 5 hygiene item 20)

What it is.  A carrier leg's exact-kernel refinement is applied over the
reduced frame `z_eff = B/A`, which diverges at a focus.  Its wrap guard K4
bounds the WRAP of the kernel, not its accuracy.

Measured (VERIFY-B4 F3, `w = 0.3 mm`, N = 1024, dx 4 um, lambda 1.064 um,
relative L2 against the analytic Gaussian):

| distance short of focus | K4 | `'fresnel'` | `'auto'` | dropped quartic, rad |
|---|---|---|---|---|
| 1 um | 9.1e-03 | 1.71e-14 | 2.35e-03 | 0.079 |
| 10 um | 9.1e-04 | 1.33e-14 | 2.35e-04 | 0.0079 |
| 100 um | 9.1e-05 | 8.9e-15 | 2.36e-05 | 7.9e-04 |
| 1 mm | 9.3e-06 | 1.15e-14 | 2.41e-06 | 8.1e-05 |
| 5 mm | 2.1e-06 | 1.37e-14 | 5.29e-07 | 1.8e-05 |

The `'auto'` column is linear in |z_eff| and independent of N; K4 stays three
to six decades under its bar the whole way, so nothing warns.  A second
condition of the same shape (`k |z_eff| theta^4 / 8` against
`gap_env_phi_tol`, a tolerance the chain already carries) would make `'auto'`
drop to `'fresnel'` where the refinement stops helping; VERIFY-B4 estimates it
at eight lines and did not write it because it changes what `'auto'` means.

Recommendation: deferred to the PENDING near-focus table (section 4.3), which
sweeps `gap_kernel` x transport x distance-to-focus on a validated fixture and
will state a derived threshold.  The direction is clear (add the condition);
the number is not yet.

### 1.6 The odd-N grid-centring convention (WP-B11 item 13; render alignment PENDING, WP-B7c round 3)

What it is.  The package coordinate array centres a grid at index N/2; the FFT
shift centres it at N // 2.  For even N these coincide; for odd N they differ
by exactly half a pixel and no sample sits at zero.

Measured (WP-B11 sec. 2.13): 149 shift sites in the package, of which 27 are
coordinate-coupled (7 in `propagators/asm.py`, 4 each in
`ui/phase_retrieval_dock.py` and `backend/fft.py`, the rest spread over
analysis, sas, through-focus, doe, carrier and two UI docks).
`compute_psf(method='fft')` on an odd grid returns its peak and centroid
exactly (-0.5, -0.5) px from the coordinate origin, independent of N: a rigid
shift, not a discretisation error, so it never converges away.

Recommendation (high confidence on the mechanism, the convention itself is the
maintainer's): land the proposed `lumenairy/_math/centring.py` helper
(`coordinate_axis`, `dc_index`, `centre_shift`) and the classification without
moving anything, then decide per site between asserting even N and applying
the half-pixel phase correction; the correction moves every odd-grid answer at
that site.  The related question of whether the multibranch renders' pixel
centres span the same window (VERIFY-B7c round 2, E5) is being measured now.

### 1.7 `replica_fill='zero'` as the carrier readouts' default (WP-A25)

What it is.  A readout window wider than one period of the transform returns
periodic replicas of the spot in the outer region.  `replica_fill='zero'`
keeps the window and blanks the replicas; `'repeat'` (shipped) returns them.

Measured (WP-A25 sec. 4): the blanking was built unconditionally and run
against the suite.  Nine tests failed, three of them on DATA rather than on a
message token, and those three are the demonstrations that justify the replica
refusal itself: the walking-chief-ray ghost (a window one period off returns a
peak bit-identical to the real spot's), the "refused window really would have
been corrupt" wing metric, and a K1 field-of-view test whose docstring records
that an earlier silent shrink-and-zero of the requested window was itself a
defect (decided twice, D3 2026-08-06 and V3).  On the battery cell the
per-call `'zero'` takes the EE2w width from 20.50 um / 0.4953 to 18.50 um /
0.9970 and the returned-window power from 5.70x to 0.99873x of the stop
plane's.

Recommendation (medium confidence): keep `'repeat'` as the default.  The
contract "the whole requested window is live" was decided twice and the
per-call keyword gives every caller the honest answer.  If the maintainer
prefers `'zero'`, the three data tests have to be rewritten as demonstrations
of the OPT-IN `'repeat'`, which is a re-statement of what the refusal is for,
not a re-pinning.

### 1.8 `DENSE_MEM_BUDGET_ACCOUNTING` default `'legacy'` (WP-B14, VERIFY-B14)

What it is.  The dense GBD loop sizes its chunks from a memory budget.  The
legacy accounting under-counts by about six-fold, so a caller who sets
`mem_budget_mb` to fit a machine can be handed 6x what they asked for; the
`'measured'` mode counts honestly but changes the chunking and so the returned
bytes.  `'measured'` is a bound only above one column (VERIFY-B14 D3: at
N = 256 a 4 MB budget reads 2.39x and a 1 MB budget 9.58x, because the chunk
floors at one column and a fixed ~48 B/cell term sits outside the chunk
arithmetic).

Recommendation (VERIFY-B14 sec. 6, high confidence): keep `'legacy'` as the
default for byte-identity, and either (a) flip it in a MINOR with a Migration
note, or (b) keep it and emit a one-shot notice from the dense path when
`mem_budget_mb` is set, naming the six-fold factor and the two mitigations.  A
warning is not a byte move, so (b) is the cheaper first step.

### 1.9 `DeformableMirror`'s inclusive `'auto'` cache ceiling (WP-B8, VERIFY-B8)

What it is.  With `cache_basis='auto'` the mirror caches its influence-function
basis when it fits under `_DEFAULT_CACHE_CEILING_BYTES = 536 870 912` (512 MiB),
inclusive.  The audit's 16x16-on-512 case is exactly 16^2 * 512^2 * 8 = 536 870 912
bytes, so it caches half a GiB silently; the warning fires once above half
the ceiling (VERIFY-B8 3c-iii: 12x12 on 512 warns, 11x11 does not).

What a change costs: flipping `<=` to `<` changes the summation order of
`phase()` for that one boundary case, so its answer moves in the last bits.

Recommendation (low stakes, high confidence): leave the inclusive ceiling; the
warning already fires from 256 MiB upward.  If the maintainer wants the
boundary case out of the cache, do it in a MINOR with the boundary case's
re-pinning.

---

## 2. Whose diagnostic it is

### 2.1 The in-glass gap-leg warnings (WP-B11 items 10 and 2b.1, request 4b-1)

What it is.  `apply_real_lens(wave_propagator='sas' | 'fresnel')` runs its
in-glass gaps through a single-FFT Fresnel kernel whose sampling bound is
`z >= N dx^2 / lambda_medium`.  On the covering-array doublet both legs return
1.04e4 times the input power (the chirp aliased 240x); `'fresnel'` warned twice
and `'sas'` was silent until WP-B11b added its near-field gate.  Both warnings
are now emitted, but their `stacklevel` is a literal inside the propagator and
the caller is `_lens_real._propagate_through_glass`, so a user is pointed at
library source rather than at their own call.

Recommendation (medium confidence): the LENS owns the diagnostic.  Making it
name the user needs `_propagate_through_glass` to catch and re-emit at the lens
entry point with `caller_stacklevel()`; that is a design decision about whose
warning it is (propagators/ plus `_lens_real`), not a one-line stacklevel
change, which is why WP-B11 did not make it.  The same applies to
`propagators/carrier.py`'s warning chain, which the Wave 5 sweep left
unswept.

---

## 3. Data and environments the library cannot obtain by itself

### 3.1 Thorlabs AC254-050-C / AC254-100-C / AC254-200-C catalogue rows (WP-A10 I6)

An independent ABCD trace of the bundled prescriptions gives effective focal
lengths 10.9 %, 16.8 % and 31.3 % short of the part numbers' nominal values
(the LA singlets agree to 0.35 %).  The rows now warn on every call; the data
were not corrected because correcting them needs the vendor's surface data.
Decision: obtain the three prescriptions from Thorlabs and re-record, or
retire the rows.

### 3.2 A Zemax `BICONICX` / `TOROIDAL` reference file (WP-A10 I8)

The importer maps these surfaces to the base conic with a loud warning naming
the hand-entry route (`make_biconic`, `radius_y`, `conic_y`).  Whether the
`BICONICX` parameter carries an X radius or an X curvature cannot be settled
offline.  Decision: supply one OpticStudio-written `BICONICX` file (one surface,
X and Y radii known) so the convention can be pinned.

### 3.3 CaF2: Malitson (bundled) versus Daimon-20 (catalogue dispatch) (WP-A8)

With the optional `refractiveindex` catalogue installed, `get_glass_index('CaF2')`
dispatches to Daimon-20; without it, the bundled Malitson row answers.  The two
differ by 2.8e-5 in index.  Decision: pick one as the answer regardless of
install state (the bundled row can be converted to Daimon's coefficients in
about an hour), and say so in the glass documentation.

### 3.4 The HDF5 `lzf` compression hang (WP-A10, observed)

`save_field_h5(..., compression='lzf')` on a 1024^2 complex128 array did not
complete within 400 s on the build box, while `gzip` took 0.43 s and `None`
0.007 s.  `lzf` is not a default anywhere and no test exercises it; the cause
is most likely the h5py filter rather than library code.  Decision: whether to
document `lzf` as unsupported, or to spend an hour reproducing it against a
clean h5py install.

### 3.5 Two UI items that need a real PySide6 (VERIFY-A9 8.2 and the empty-prescription trace)

The UI suite runs against a stub widget layer, so it cannot answer a Qt
selection-signal question: whether a source installed programmatically while
the Source row is already selected refreshes the 13 numeric fields before the
user's next keystroke.  The recipe is in VERIFY-A9 sec. 8.2 (select the Source
row, fire Insert > Source > Fiber mode, edit one visible field, assert the
fibre mode-field diameter is still 10.4).  Decision: run it once on a PySide6
box, or accept it as untested.

---

## 4. Items still being measured (to be filled in before the 5.48.0 tag)

### 4.1 MEASURED, verification in flight: the FFT double buffer's returned object, and the NumPy elision report (Wave 5 item E1)

VERIFY-B14 sec. 3 established the mechanism: with the pyFFTW ping-pong on,
`_fft2` returns a non-owning view of its workspace, so in `_fft2(E) * H` NumPy's
temporary elision claims the RIGHT operand, and on the Linux NumPy build a
right-elided complex128 multiply moves the last bits (16 to 17 % of the
doubles, relative 1e-16 to 1.8e-16).  The transforms themselves are functions
of their inputs.

Measured by item E (WAVE5_E_LEFTOVERS_REPORT.md, both builds): returning a
private copy would cost 18.6 % to 22.1 % on `angular_spectrum_propagate` at
512^2 to 2048^2 (Windows) and 13 % (WSL), one copy being 0.30 to 0.42 of a
forward transform, and would buy NOTHING inside the library: every in-library
product site names its operand, so four entry points x three shapes x two
builds are byte-identical across the switch (12 of 12).  What moves is a
CALLER who spells the elided form, by 3.4e-16 to 4.0e-16 relative on the Linux
wheel and exactly 0 on Windows.  The remedy shipped is therefore the scoped
contract (VERIFY-B14 D4's sentence, verbatim, in the knob doc, the setter
docstring and the module note), not a copy.  Two facts for the upstream
report: the effect is a property of the WHEEL (NumPy 2.4.6 in a clean Windows
venv is unaffected), and the explicit `np.multiply(a, b, out=b)` matches the
named form while only the elided spelling moves, which is what makes it an
elision defect rather than in-place complex rounding.

Decision owed: whether to file the NumPy report.  The lumenairy-free
reproducer and the issue draft are under `validation/probe_fft_elision/`
(`numpy_elision_reproducer.py`, `NUMPY_ISSUE_DRAFT.md`); nothing has been
filed.  Recommendation (medium confidence): file it, with the wheel-not-version
finding, since it affects any NumPy user on that wheel and the draft is ready.

### 4.2 PENDING: the direct-matrix MFT branch, opt-in or threshold-automatic (Wave 5 hygiene item 14)

Being measured: the memory and wall-time crossover between the direct O(N^2 M^2)
matrix transform and the separable reduction over N x M grids on both builds,
and the tolerance between the two.  It ships opt-in; the decision is whether a
measured threshold should select it automatically.

### 4.3 PENDING: the near-focus exact-kernel table (Wave 5 hygiene item 20; informs 1.5)

Being measured: `gap_kernel` x transport x distance-to-focus on the WP-B11
fixture (converging Gaussian, f = 20 mm, w0 = 15.9 um, 1 um to 5 mm short of
focus) against the analytic Gaussian, both builds, once the carrier-envelope
bookkeeping is validated on a case with a known answer.

### 4.4 PENDING: the Newton worker pool's join timeout and the interpreter-exit hang (WP-B13 follow-ups)

VERIFY-B13 confirmed that the pool's broken-pool fallback now reaches the
serial path (byte-identical, 2.5 s where the old tree hung forever) and that
two exposures remain: the healthy path's `as_completed` has no timeout (so a
pool that never answers wedges the caller; `carrier.py::_multi_parallel_results`
has the same shape), and a wedged manager thread still hangs the PROCESS at
interpreter exit even though the computation completes.  Being measured: the
slowest chunk time on a traced-lens ladder (to derive a timeout rule), and an
`atexit` reaper behind a switch defaulting off.  Decision owed: whether a join
timeout becomes a default (it moves the failure mode from "hang" to "raise
after T seconds" on every traced-lens call).

### 4.5 PENDING: the multibranch arbiter's bar and the accept criterion (WP-B7c round 3)

The pixel-halving arbiter (round 2) refuses a multibranch field whose
continuity reading `p_out(dx) / p_out(dx/2)` exceeds `_PIXEL_CONTINUITY_MAX = 1.06`.
VERIFY-B7c round 2 confirmed the mechanism and the bit identity (44 identical,
0 moved, 6 newly refused of 57) but found that the 1.0683x gap the bar was
centred in does not survive a wider population: on 328 planes over 7 optics the
gap is 1.0038x, the refusal-side margin 1.00051x, and two refusals at fidelity
>= 0.95 each carry a 10 % energy error, so whether they are "false" depends on
the accept criterion.  Being measured: a >= 12-optic population with a
full-radius oracle, the cost of the bar at 1.04 / 1.06 / 1.08 and of a bar
derived from the converged reading's own spread (0.9995 to 1.0002, not exactly
1).  Decision owed: the accept criterion (what fidelity or energy error counts
as a wrong field), and with it the bar.

---

## 5. Not decisions, recorded here so the ledger closes

* The second BLAS-classification pin (SANDYBRIDGE, VERIFY-B14 7a) is CONFIRMED
  closed.
* Whether to push and tag the audit branch: done (5.47.0 tagged 2026-09-15,
  5.47.1 in flight with the two publish-verification repairs).
* GBD's inline sag copy (15.5 waves wrong on an asphere, 47.5 at full aperture)
  is a defect, not a decision; it is being replaced by the shared projection
  (Wave 5 item A2).
