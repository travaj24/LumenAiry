# VERIFY-WP-C2 ROUND 2 -- independent re-verification of the twelve closures

Subject: branch `feat/c2-analytic-normal-round2`, sixteen commits
`fe1cafd8`..`8ac607ee` on `verify/c2-analytic-normal` (`61ffe596`), itself on
`feat/c2-analytic-normal-default` (`eadc67ba`) on `49ddf4bd`.  Documents under
test: the "Round 2 (VERIFY-WP-C2)" addendum in
[`WP-C2_ANALYTIC_NORMAL_REPORT.md`](WP-C2_ANALYTIC_NORMAL_REPORT.md), the
twelve defect closures it claims, `tests/unit/test_c2_analytic_normal_default.py`
(41 ids), `tests/unit/test_verify_c2_analytic_normal.py` (17 ids) and
`validation/probe_c2_round2/`.

Verification worktree `C:/tmp/lum_vc2b`, branch
`verify/c2-analytic-normal-round2`.  PRE trees are this verifier's own
`git archive 49ddf4bd` (`C:/tmp/lum_vc2b_pre49`), `git archive 61ffe596`
(`C:/tmp/lum_vc2b_pre61`) and `git archive eadc67ba`
(`C:/tmp/lum_vc2b_preead`), each read in its OWN process with its own
`sys.path` and `lumenairy.__file__` asserted inside it.  Builds: **Windows
py3.14.6 / numpy 2.4.4 / jax 0.11.0** and **WSL py3.12.3 / numpy 2.4.6 / jax
0.10.2**, every probe on both, all with `OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command line.  Nothing below
is read off the round-2 addendum: every number is re-measured on this
verification's own prescriptions, its own oracles, its own Jacobian and its
own mutants.  Evidence: `validation/probe_verify_c2_round2/` (10 probes, JSON
per build); decision tests `tests/unit/test_verify_c2_round2.py` (13 ids).

---

## 0. Verdict table

| # | Round-2 claim under test | Verdict | This verification's numbers |
|---|---|---|---|
| 1 | all SIXTEEN directly-tracing entry points take keyword-only `sphere_normal=` / `renormalize=`, default `None`; 742/742 arrays byte-identical archive to archive; 16/16 move at the default; `None` == omitted | **CONFIRMED, on a larger independent set** | my own census finds the same **20 exported directly-tracing functions, 16 with both keywords (all keyword-only, all defaulting to `None`), 4 lacking = exactly the jax twins**, identical on both builds.  My own fixtures (a spherical doublet, an ASPHERIC singlet, a two-SPHERICAL-MIRROR system; 37 cases, **1506 arrays**): way back **1506/1506 byte-identical**, `None` == omitted **1506/1506**, **16 of 16** entry points move at the default (184 arrays Windows, 185 WSL).  A spy on the tracer sees **43 internal calls** over the 37 cases and EVERY one carries the forwarded pair -- including the three entry points that trace twice |
| 1b | the census is complete | **REFUTED -- there is a SEVENTEENTH** | `lumenairy.apply_real_lens` traces internally (`seidel_correction=True`), carries NEITHER keyword, and MOVES archive to archive at the shipped defaults (max abs delta **2.8389e-13** Windows / **2.8387e-13** WSL on a 256x256 field).  Both censuses miss it because `elements/_lens_real.py` imports the tracer as `trace as _rt_trace`.  **DEFECT VR2-D1** |
| 2 | 35 of 46 transitive callers resolve; 8 of the 11 that do not are PMM grating functions naming no tracer | **CONFIRMED with a correction** | the 11 are **8 `pmm_*` in `elements/pmm/oned.py`** plus **`aberration_tensor`, `design_optimize_multi_objective`, `prescription_subdomain`** -- all eleven verified to name NO tracer in their own bodies.  But **only 31 of the 35 resolve to a parent that carries both keywords**; the other 4 (`monte_carlo_tolerancing`, `monte_carlo_tolerancing_jax`, `optimize_traced_geometry`, `make_lg_aberration_merit_jax`) resolve to `trace_jax`, which has neither by design.  **DEFECT VR2-D6 (P3, wording)** |
| 3 | D1: `Decimal(float(x))` at prec 80; closed form 1.50 vs generic 1.75 to `0.95 R`; whole set 45.86 vs 91.49; 0 of 672 worse by > 1; prec 80 == prec 120 | **CONFIRMED** | `Decimal(float(x))` IS exact for every input the oracle receives -- `x`, `y`, `R` are the same float64s the library is handed and nothing (`h/R`, a normalised direction) is computed before conversion.  On my own 4284-point set with TWO independent oracle formulations agreeing to **3.31e-28** (1.5e-12 ULP): **1.50 vs 1.75** to `0.95 abs(R)`, **0 of 3276** worse by > 1; whole set **49.41 vs 60.95**; prec 100 == prec 160.  Identical to the last digit on both builds.  The retired `repr` conversion costs 1.50 -> 2.50 at 0.95 and 49.41 -> 71.98 overall ON MY SET, so D1 was a real defect |
| 4 | D2: `kappa` = the true induced `inf<-2` norm, 2.9059e+07, 4.65x the seeded draw; bar 10x floor; bracketed | **CONFIRMED, independently recomputed** | my own Jacobian, my own SVD, three FD steps, three reduction methods: **2.90590059e+07** (Windows) / **2.90590059e+07** (WSL), closed form == full SVD to **4e-16** relative and the power iteration ATTAINS it (so the closed form is not an underestimate); FD spread over 1e-12/1e-11/1e-10 **1.00011**.  70 columns is the COMPLETE operator on `coef_phi` (one column per coefficient, not a subsample), and the fit's other float inputs respond at **0 to 530**, five decades below.  Ratio to the shipped seed **4.655x**; floor **6.4524e-09**, bar **6.4524e-08**; bracket **0.9x -> 0.850 of the bar (passes)**, **1.1x -> 1.173 (Win) / 1.196 (WSL) (fails)** -- two-sided |
| 5 | D5: ghost asks `_library_trace_default('sphere_normal')`, `renormalize` stays True | **CONFIRMED** | the ghost leg passes `sphere_normal='analytic'` and OMITS `renormalize` (effective `True`, `_refract`/`_reflect` default); it FOLLOWS the helper (forcing the helper to answer `'generic'` makes the ghost pass `'generic'`), so it asks rather than names.  At the refraction step itself the ghost route's bundle is **byte-identical** to `trace`'s default route (`af2145f1...`) and **differs** from the generic route (`60910cef...`), so the identity is not vacuous.  Digests IDENTICAL on both builds |
| 6 | D7: `EDITED_IN_PLACE` pins a SHA-256 + the release; the three abuses are refused | **CONFIRMED, with three residual holes** | on twelve doctored cases: reverted default, nonsense value, stale OLD-content copy and a package past the recorded release are all **REFUSED** with a record carrying both lines; the shipped content and a package exactly AT the release **FIRE**.  Still fires for: a re-indent (documented -- the digest strips whitespace), **a line with the EXACT expected content under a DIFFERENT enclosing `def`**, and **a digest re-recorded to match a reverted default**.  **DEFECT VR2-D7 (P3)** |
| 7 | D8: four one-ULP floors, max of up/down brackets the max | **CONFIRMED to four figures on both builds** | arm 1 up **2.5864e-03** / down **1.8019e-03** / real **2.5864e-03** / one-element **1.2497e-03**, spread **2.07x** (Windows); up **8.4198e-05** / down **2.2728e-03** / real **8.4198e-05** / one-element **1.8240e-04**, spread **26.99x** (WSL).  arm 2 **0.14036 / 0.17820 / 0.14036 / 0.055353**, spread **3.2194x** (Windows); **0.11079 / 0.11491 / 0.11079 / 0.023985**, spread **4.7909x** (WSL).  `max(up, down)` brackets all four on BOTH builds and BOTH arms.  Margins **65.32x / 74.38x** (arm 1) and **4.691x / 8.587x** (arm 2); the same-degree-twice control reads **exactly 0.0** |
| 8 | D6: the decision test is two-sided | **CONFIRMED by mutation** | a mutant that gives JAX the NumPy clamp turns **2** tests red (`test_c2_the_two_backends_gate_the_sphere_domain_differently`, `test_vc2_the_jax_backend_does_not_apply_the_numpy_domain_clamp`); a mutant that takes the clamp OFF the NumPy side turns **11** red including the same arm.  Both directions caught |
| 9a | D9: the gates bisect to the same float at all 8 radii; the ball lens loses 3024/60000 | **CONFIRMED** | 200-step bisection at eight radii of both signs: both routes give **`0.9999499987499374`** at every one, on both builds -- the band does not exist on the meridian.  A ball lens AND a hemisphere with semi-diameter `abs(R)` lose **3029 of 60 000** rim-packed rays (**5.048 %**) to `RAY_NAN`, identical under all four `(renormalize, sphere_normal)` settings and on both builds |
| 9b | D12: 934/1008 Windows, 935 WSL, derived from the JSON | **CONFIRMED** | the committed JSONs read `n_common = 1008`, `n_moved = 74` (Windows) / `73` (WSL) -> 934 / 935; both the CHANGELOG entry and the Migration-Guide 5.49.0 section carry those four numbers and no longer carry `938 of 1008` or `the 70 that are not` |
| 9c | D3: ratio 1.000 at 3 -> 0.615 at 13, `1e-15` first exceeded at the seventh, `n_surfaces * eps` is THE BOUND | **PARTLY REFUTED -- `n_surfaces * eps` is not a bound** | on an ordinary 3-surface refracting stack with different radii and a different glass the drift is **8.8818e-16 against a 6.6613e-16 envelope, a ratio of 1.3333**, on BOTH builds; over 90 (surface count, glass, radius pair, field angle) combinations **6 exceed** `n * eps`, worst 1.3333 at three surfaces and 1.10 at five, while `2 n eps` holds on all 90.  `1e-15` is first exceeded at the **FIFTH** surface on my ladder (the test bands 5..9, so the test survives; the docstring's "SEVENTH" is a reading).  **DEFECT VR2-D2 (P2)** |
| 10 | the `w6_a2` restatement: a second Newton step < 1e-4 of the first, "a wrong Hessian, a missing prior term, a sign error" give a ratio of order 1 | **PARTLY REFUTED** | the restatement IS a decision that can fail, unlike the retired tautology: no step at all **1.0000**, half the step **0.5000**, a tenth missed **0.9000**, a sign error **2.0000** -- four decades above the bar, both builds.  But the bar is a CHOSEN 1e-4, and the third mode its own message names -- **the prior term dropped from the Hessian** -- reads **2.5329e-05 (Windows) / 2.5698e-05 (WSL)**, i.e. it PASSES.  **DEFECT VR2-D5 (P3)**.  Shipped reading 1.9355e-07 / 8.7659e-08, confirming 1.94e-07 / 8.77e-08 |
| 11 | the census mutant fires "3 failed" on both builds | **CONFIRMED (4 on my run, which includes the 3)** | dropping `sphere_normal` from `ray_fan_data`'s signature turns red: both census arms, VERIFY-WP-C2's own pin, and `none_stamps_nothing[ray_fan_data]`.  My own three mutants: a keyword threaded but DROPPED before the inner trace -- **caught** (1 failed); `_way_back_kwargs` mapping `None` to `'generic'` -- **caught** (9 failed); `_library_trace_default` returning the wrong route for `renormalize` -- **SURVIVES, 58 passed**.  **DEFECT VR2-D3 (P3)**.  A route-less exported caller added to a scratch copy is **named by the census** (2 failed) |
| 11b | the forwarding is pinned | **PARTLY -- 9 of the 16** | `test_c2_none_stamps_nothing_on_the_entry_points` is the only IN-PROCESS pin that the keyword REACHES the trace, and it parametrizes nine.  Dropping the forward in `elements/_lens_traced.py`, `propagators/asymptotic_canonical_fit.py`, `analysis/image_plane_wfe.py` or `analysis/aberration.py` leaves the whole C2 suite green -- **58 passed, 0 failed on all four**.  **DEFECT VR2-D4 (P2)** |
| 12 | round 2 did not move any default | **CONFIRMED** | the round-1 verifier's 594-key set, `61ffe596` archive vs the tip, at the DEFAULT: **594 / 594 byte-identical** on BOTH builds.  The same set still reads **594 / 594** for the way back against `49ddf4bd` and **172 / 594** at the default (422 moved), exactly round 1's numbers |
| 13 | the second direction in the 93.6 s test | **NEEDED, and already minimal** | it costs **15.76 s of 93.6 s** (Windows) and **29.65 s** (WSL).  On Windows `'up'` is the maximum so it changes nothing; on WSL `'down'` is **27x** `'up'` and IS the maximum, and dropping it would read the arm at 2007.7x instead of 74.4x on a floor 27x too small.  The other two directions are provably dominated (`'real'` is bit-identical to `'up'` because the envelope is real; `'one_element'` is the smallest on every build), and they are correctly not evaluated.  No cheaper formulation was found that keeps the floor honest across builds |
| 14 | `.test_durations`, no forward version token, the 3 WSL reds | **CONFIRMED** | `.test_durations` parses as JSON with **16 654** entries, 41 `test_c2_analytic_normal_default.py` ids and 17 `test_verify_c2_analytic_normal.py` ids, the d3 arms at **93.64 / 32.99 s** and the two ModalAsymptotic arms at **6.42 / 8.57 s**; the staleness gate passes.  No forward version token under `lumenairy/` (`__version__ = 5.48.1`; the only `5.49` hit is `5.49e-03` in an RCWA convergence table).  The three WSL reds fail IDENTICALLY on my own `git archive 49ddf4bd` under WSL |

---

## 1. Defects

Requested edits are exact.  Nothing under `lumenairy/` was edited by this
verification.

### VR2-D1 (P2, API) -- a SEVENTEENTH entry point, hidden by an import alias

`lumenairy/elements/_lens_real.py:8300-8303` and `:8324`:

```python
        from ..raytrace import (
            _make_bundle as _rt_make_bundle,
            surfaces_from_prescription as _rt_surfaces_from_prescription,
            trace as _rt_trace,
        )
        ...
        res_fan = _rt_trace(fan, surfs_fan, wavelength)
```

This is inside `_apply_real_lens_impl`, which the exported
`lumenairy.apply_real_lens` calls one hop away whenever
`seidel_correction=True`.  Both entry-point censuses -- the shipped
`_c2_entry_point_census` and VERIFY-WP-C2's own -- decide "this body traces"
by looking for the NAME `trace`; the body names `_rt_trace`, so
`apply_real_lens` appears in neither, and it is absent from the 46-function
transitive population in `r2_transitive_parents_*.json` for the same reason.

Measured (`validation/probe_verify_c2_round2/vr2_alias_{pre,post}_{win,wsl}.json`,
each tree in its own process):

| | Windows | WSL |
|---|---|---|
| `trace` calls from `apply_real_lens(seidel_correction=True)` | 1, both keywords `<omitted>` | 1, both `<omitted>` |
| the same call with `seidel_correction=False` (control) | 0 | 0 |
| digest, `git archive 49ddf4bd` | `ecc24ca9...` | `fb31d3e4...` |
| digest, the tip at its defaults | `6dc7d801...` (**moved**) | `f29c6232...` (**moved**) |
| the tip with the tracer forced back to the old keywords | `ecc24ca9...` (**identical to the archive**) | `fb31d3e4...` (**identical**) |
| max abs field delta, default vs old arithmetic | **2.8389e-13** | **2.8387e-13** |

So the way back exists arithmetically and the only way to reach it is to
monkeypatch `lumenairy.raytrace.trace` -- which is exactly the situation the
campaign rule was written to prevent.

Requested edits.  (a) Give the entry point the pair, threaded through the
private implementation the same way the other sixteen are:

```python
# lumenairy/elements/_lens_real.py -- apply_real_lens
def apply_real_lens(E_in, *, ..., renormalize: Optional[str] = None,
                    sphere_normal: Optional[str] = None) -> np.ndarray:
    ...
        return _apply_real_lens_impl(
            E_in, ..., renormalize=renormalize, sphere_normal=sphere_normal,
            _accum_store=_store)

# and in _apply_real_lens_impl, at the Seidel fan:
        from ..raytrace.trace import _way_back_kwargs
        res_fan = _rt_trace(fan, surfs_fan, wavelength,
                            **_way_back_kwargs(renormalize, sphere_normal))
```

(b) Make BOTH censuses alias-aware, so the next alias cannot hide the next
entry point.  In `_c2_entry_point_census`, collect the module's
`ImportFrom` aliases first and count a bare `Name` whose id is one of them:

```python
        aliases = {a.asname: a.name
                   for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom))
                   for a in node.names
                   if a.asname and a.name in _C2_TRACERS}
        ...
            hit = any(
                (isinstance(sub, ast.Name)
                 and (sub.id in _C2_TRACERS or sub.id in aliases))
                or (isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr in _C2_TRACERS)
                for sub in ast.walk(node))
```

(c) The round-2 addendum's D4 section and the Migration Guide's entry-point
list say "sixteen"; with (a) they become **seventeen**.

Pinned by `test_verify_c2_round2.py::test_vr2_an_aliased_tracer_import_hides_an_exported_entry_point`
and `::test_vr2_the_alias_hidden_entry_point_really_traces_and_really_moves`.

**Scope note, not a defect.**  Three GUI class METHODS name a tracer and
carry no keyword: `ui/model.py::SystemModel.run_trace` (`trace_world`),
`SystemModel.merit_function` (`trace`) and
`ui/tolerance_dock.py::ToleranceWorker.run` (`trace_world`).  Neither census
covers methods.  A GUI user drives these through the dock rather than by
keyword, so this is a documented scope boundary; it is recorded here because
the brief asked about the docks specifically.  (On WSL the 49 `lumenairy.ui.*`
modules fail to import -- PyQt is absent -- and the census reads the same 20
either way, so the GUI contributes no module-level exported tracing function
on either build.)

### VR2-D2 (P2, docs + test) -- `n_surfaces * eps` is a reading, not a bound

`lumenairy/raytrace/trace.py`, the `renormalize` docstring:

> ``n_surfaces * eps`` is the BOUND, and the coefficient in front of it is
> NOT constant. ... So size a tolerance from ``n_surfaces * eps`` itself

and `tests/unit/test_c2_analytic_normal_default.py::test_c2_history_bundles_are_not_unit_under_the_new_default`:

```python
        # the derived envelope, not a reading
        assert worst <= len(S) * eps, (len(S), worst, len(S) * eps)
        ...
    assert all(r <= 1.0 + 4 * eps for _n, r in ratios), (...)
```

It is a reading of that test's own ladder -- the same doublet repeated.  On a
three-surface refracting stack with different radii and a different glass the
drift EXCEEDS it, on both builds:

| stack (3 surfaces) | worst history drift | `3 * eps` | ratio |
|---|---|---|---|
| the shipped ladder (0.0515 / -0.0515 N-BK7 + flat) | 6.6613e-16 | 6.6613e-16 | 1.0000 |
| 0.0623 N-SF5 / -0.0771 / 0.3100, 2 deg field | **8.8818e-16** | 6.6613e-16 | **1.3333** |

Over 90 combinations of surface count (3, 5, 7, 9, 13), glass (N-BK7, N-SF5,
SF11), radius pair and field angle (0, 2, 5 deg), **6 exceed `n * eps`**, worst
**1.3333** at three surfaces and **1.10** at five.  `2 * n_surfaces * eps`
holds on all 90 with 1.5x to spare.  A consumer who sizes a tolerance from
`n_surfaces * eps` on a triplet -- the commonest case, and the one the
docstring explicitly steers them to -- is 33 % under.

Requested edits:

```python
# lumenairy/raytrace/trace.py, the renormalize docstring
        only the FINAL bundle is rescaled, so the intermediate
        ``ray_history`` bundles carry ``| |d| - 1 |`` BOUNDED BY
        ``2 * n_surfaces * eps``.  ``n_surfaces * eps`` is NOT a bound:
        it holds on a repeated-doublet ladder and is exceeded by 1.33x on
        a 3-surface stack of different radii and glasses (measured
        8.8818e-16 against 6.6613e-16, both builds, VERIFY-WP-C2 round 2;
        6 of 90 surface-count / glass / radius / field-angle combinations
        exceed it, worst 1.3333).  The coefficient in front of
        ``n_surfaces * eps`` runs about 1.33 down to 0.54 across 3..13
        surfaces and is not constant; size a tolerance from
        ``2 n_surfaces eps``.

        ``1e-15`` -- the bound this docstring carried before the default
        moved -- is first exceeded between the FIFTH and the SEVENTH
        surface depending on the stack (measured 1.1102e-15 at five on a
        N-SF5 ladder and 1.22e-15 at seven on the N-BK7 one), not the
        eighth.
```

```python
# tests/unit/test_c2_analytic_normal_default.py
        # VERIFY-WP-C2 round 2 (2026-09-20): the envelope is 2 n eps, not
        # n eps.  n eps holds on THIS ladder and is exceeded 1.33x on a
        # 3-surface N-SF5 stack; 2 n eps holds on all 90 stack / glass /
        # radius / field-angle combinations measured.
        assert worst <= 2 * len(S) * eps, (len(S), worst, 2 * len(S) * eps)
        ...
    assert all(r <= 2.0 + 4 * eps for _n, r in ratios), (
        f'the drift left the 2 * n_surfaces * eps envelope: {ratios}')
```

The falling-ratio claim (`first > 1.25 * last`) and the `5 <= exceeding[0] <= 9`
band are both re-measured here and both survive unchanged.

Pinned by `test_verify_c2_round2.py::test_vr2_the_history_drift_envelope_needs_two_n_eps_not_one`,
which compares the two fixtures in one process and therefore carries no
recorded number as a bar.

### VR2-D3 (P3, test) -- `_library_trace_default` is pinned for one keyword only

`lumenairy/raytrace/trace.py::_library_trace_default` exists so a direct
caller of `_refract` / `_reflect` can ASK the library rather than write a
route down, and its docstring offers it for BOTH switches.  Only
`sphere_normal` is pinned -- in
`test_verify_c2_analytic_normal.py:299` and in the D5 arm.

A mutant in which the helper returns `'surface'` for `renormalize` while
`trace` defaults to `'exit'` passes the whole C2 suite: **58 passed, 0 failed**
(`validation/probe_verify_c2_round2/vr2_mutants_win.txt`, mutant M4).  Nothing
consumes that answer today, which is why it is P3; the day a second direct
caller takes the helper up on its offer, a wrong answer is unpinned.

Requested edit -- one loop over the pair the library itself names, in
`test_c2_analytic_normal_default.py`:

```python
    from lumenairy.raytrace.trace import _WAY_BACK_KEYWORDS
    params = inspect.signature(trace).parameters
    for key in _WAY_BACK_KEYWORDS:
        assert _library_trace_default(key) == params[key].default, key
```

Pinned by `test_verify_c2_round2.py::test_vr2_library_trace_default_agrees_with_trace_for_every_keyword`.

### VR2-D4 (P2, test) -- the forward is pinned in process for 9 of the 16

`test_c2_none_stamps_nothing_on_the_entry_points` is the only arm that checks
the forwarded keyword actually REACHES the internal trace (its `a != c` half),
and it parametrizes nine entry points.  The other seven are covered only by
the committed probe JSON, which is a recording rather than a gate.

Measured: dropping the forward (`**_way_back_kwargs()` with no arguments)
leaves the keyword in the signature, raises nothing, and turns nothing red --
**58 passed, 0 failed** on every one of the four tried
(`validation/probe_verify_c2_round2/vr2_mutants2_win.txt`):

| mutant | file | result |
|---|---|---|
| M8 `apply_real_lens_traced` | `elements/_lens_traced.py` | **58 passed** (survives) |
| M9 `fit_canonical_polynomials` | `propagators/asymptotic_canonical_fit.py` | **58 passed** (survives) |
| M10 `eval_image_plane_wfe` | `analysis/image_plane_wfe.py` | **58 passed** (survives) |
| M11 `caustic_diagnostic` | `analysis/aberration.py` | **58 passed** (survives) |

`apply_real_lens_traced` is one of the two headline lens propagators D4 was
raised about, so this is the same hole one level down: the census sees the
signature, and nothing sees the wire.

Requested edit -- extend the existing parametrization rather than add a file:

```python
@pytest.mark.parametrize('name', [
    'trace_prescription', 'raytrace_system', 'ray_fan_data',
    'ray_fan_data_world', 'opd_fan_data', 'opd_fan_data_world',
    'through_focus_rms', 'paraxial_focus_world', 'ray_transfer_jacobian',
    # VERIFY-WP-C2 round 2 (2026-09-20): the other seven.  Dropping the
    # forward in any of them was measured to leave the whole C2 suite
    # green (58 passed) before these ids existed.
    'caustic_diagnostic', 'eval_image_plane_wfe', 'plot_lens_layout',
    'fit_canonical_polynomials', 'fit_hf_polynomials',
    'apply_real_lens_traced', 'apply_real_lens_maslov',
])
```

with the seven fixtures from
`test_verify_c2_round2.py::test_vr2_the_forward_reaches_the_tracer_in_the_seven_unparametrized_entry_points`,
which measures the same property by SPYING on the tracer instead of comparing
answers and costs **4.4 s for all seven** on Windows.

### VR2-D5 (P3, test) -- the `w6_a2` bar misses one of the modes its message names

`tests/unit/test_niche_audit_w6_asymptotic.py`, DECISION 3:

```python
    assert step_ratio < 1e-4, (
        ... f'while a root that is not converged (a wrong Hessian, a '
        f'missing prior term, a sign error) gives a ratio of order 1, '
        f'four decades above it.')
```

Measured (`validation/probe_verify_c2_round2/vr2_w6a2_{win,wsl}.json`), the
same second-step ratio evaluated at five deliberately unconverged expansion
points:

| construction | Windows | WSL |
|---|---|---|
| the shipped `v*` | 1.9355e-07 | 8.7659e-08 |
| no step taken at all | 1.0000 | 1.0000 |
| a sign error in the step | 2.0000 | 2.0000 |
| half of the first step | 0.5000 | 0.5000 |
| a tenth of the first step missed | 0.9000 | 0.9000 |
| **the PRIOR TERM dropped from the Hessian** | **2.5329e-05** | **2.5698e-05** |

Four of the five are four decades above the bar and the restatement is sound:
unlike the retired `norm_offset <= 2 * bound` it CAN fail.  But the prior term
is `I / w_p**2` against `J^T J / w_s**2` with `w_s = 20e-6` and `w_p = 0.02`,
so dropping it moves the step by 2.5e-05 of itself -- four times UNDER the bar.
A solver that forgot the prior term would pass, and the message says it would
not.  The bar is also CHOSEN rather than derived: measured, it detects a
departure of more than 1.0e-04 of the first step and no less.

Requested edit -- tighten to `1e-5` and restate the message with the measured
table:

```python
    assert step_ratio < 1e-5, (
        f'v2* is not a CONVERGED root of the model: a second Newton step '
        f'from it is {step_ratio:.3e} of the first. ...  Measured '
        f'1.9355e-07 (Windows py3.14 / numpy 2.4.4) and 8.7659e-08 (WSL '
        f'py3.12 / numpy 2.4.6) against this 1e-5 bar -- 51x to 114x of '
        f'headroom -- while an unconverged root reads 0.50 (half a step), '
        f'0.90 (a tenth missed), 1.00 (no step) or 2.00 (a sign error), '
        f'and dropping the prior term from the Hessian reads 2.53e-05, '
        f'which a 1e-4 bar would have passed.')
```

Pinned by `test_verify_c2_round2.py::test_vr2_the_w6a2_second_step_bar_misses_the_dropped_prior_term`,
which asserts the gross modes are O(1) and the dropped-prior mode sits between
1e-5 and 1e-4, both of which stay true whatever the bar becomes.

### VR2-D6 (P3, docs) -- "35 of 46 resolve to a parent that carries both"

The round-2 addendum's D4-transitive section states:

> **35 of 46 resolve to a parent that now carries both keywords**

Checked against `inspect.signature` of each claimed parent: **31 do**.  The
other four -- `monte_carlo_tolerancing`, `monte_carlo_tolerancing_jax`,
`optimize_traced_geometry`, `make_lg_aberration_merit_jax` -- resolve to
`trace_jax`, which has neither switch by design.  The section's own sample
table says so ("-> the `*_jax` twins -> `trace_jax` (no switch by design)"),
so this is the headline sentence over-reaching rather than a wrong walk.

Requested edit:

```
**31 of 46 resolve to a parent that now carries both keywords**, in one to
four hops, and four more resolve to ``trace_jax``, which has neither switch
by design -- 35 of 46 reach a tracer through a named parent.
```

(While that line is touched: `monte_carlo_tolerancing`'s CPU path is
`-> apply_real_lens`, not `-> trace_jax`.  It calls `apply_real_lens` with
`seidel_correction` unset, so it does not reach the CPU tracer today; it would
the moment a caller turns the Seidel correction on, which is VR2-D1's edit.)

### VR2-D7 (P3, tooling) -- three residual `EDITED_IN_PLACE` holes

D7's closure is real and this verification reproduced it independently by
doctoring the tool's own line cache (`vr2_reanchor_abuse.py`, twelve cases).
Three cases still FIRE:

| doctored current line | override | comment |
|---|---|---|
| the same content re-indented | **fires** | documented -- `content_digest` strips whitespace |
| the exact expected content under a DIFFERENT enclosing `def` | **fires** | the citation re-anchors to the right TEXT in the wrong function |
| the map's own digest re-recorded to match a REVERTED default | **fires** | an editor with commit access to the map can still bless a false claim |

The second is the only one worth an edit and it is cheap: pin the enclosing
definition alongside the content.

```python
EDITED_IN_PLACE = {
    ('lumenairy/raytrace/trace.py', 61): (
        61, "WP-C2 5.49.0: sphere_normal default 'generic' -> 'analytic'",
        '5d0d6615...', '5.49.0', 'def trace('),
    ...
}
...
    new_num, reason, want_digest, recorded_for, want_owner = entry
    ...
    # VERIFY-WP-C2 round 2 (2026-09-20): the content digest alone accepts
    # the same line under a DIFFERENT definition.  Require the nearest
    # preceding ``def`` / ``class`` to be the one this entry recorded.
    owner = next((ln for ln in reversed(hay[:new_num - 1])
                  if ln.startswith(('def ', 'class '))), '')
    if not owner.startswith(want_owner):
        return _refuse('the line at the mapped coordinate is the expected '
                       'content but belongs to %r, not %r'
                       % (owner.strip()[:40], want_owner))
```

The third is out of scope for a file-content guard (it is a code-review
property, not a tool property) and should be stated in the map's comment
rather than defended against.

---

## 2. Entry-point census -- mine against the sixteen

`validation/probe_verify_c2_round2/vr2_census.py` walks every `.py` under
`lumenairy/`, keys by `module:function` rather than by bare NAME (the shipped
census collapses same-named functions across modules), decides "exported" by
importing EVERY module `pkgutil` finds rather than a hard-coded list of seven,
reports nested and method definitions separately, and reads sources as cp1252
with a utf-8 fallback.  Identical on both builds:

| | Windows | WSL |
|---|---|---|
| module-level definitions naming a tracer | 37 | 37 |
| exported + directly tracing | **20** | **20** |
| carrying BOTH keywords | **16** | **16** |
| lacking one or both | **4** | **4** |
| the four lacking | `apply_real_lens_maslov_jax`, `apply_real_lens_traced_jax`, `fit_canonical_polynomials_jax`, `ray_transfer_jacobian_jax` | same |
| module import failures | 0 | 49 (all `lumenairy.ui.*`; PyQt absent) |

All sixteen carry the pair as **KEYWORD-ONLY** parameters defaulting to
**`None`**, verified from `inspect.signature`:

| entry point | module | way back (identical / total) | arrays moving at the default | `None` == omitted | spy: tracer calls, all keyworded |
|---|---|---|---|---|---|
| `trace_prescription` | `raytrace.trace` | 90 / 90 | 53 (54 WSL) | 90 / 90 | 3 / 3 |
| `raytrace_system` | `raytrace.trace` | 81 / 81 | 51 | 81 / 81 | 2 / 2 |
| `ray_fan_data` | `raytrace.ray_fan` | 12 / 12 | 4 | 12 / 12 | 3 / 3 |
| `ray_fan_data_world` | `raytrace.ray_fan` | 12 / 12 | 4 | 12 / 12 | 3 / 3 |
| `opd_fan_data` | `raytrace.ray_fan` | 12 / 12 | 6 | 12 / 12 | 3 / 3 |
| `opd_fan_data_world` | `raytrace.ray_fan` | 12 / 12 | 6 | 12 / 12 | 3 / 3 |
| `through_focus_rms` | `raytrace.ray_fan` | 6 / 6 | 2 | 6 / 6 | 2 / 2 |
| `paraxial_focus_world` | `raytrace.world` | 4 / 4 | 1 | 4 / 4 | 4 / 4 (2 per call) |
| `ray_transfer_jacobian` | `raytrace.differential` | 14 / 14 | 10 | 14 / 14 | 2 / 2 |
| `caustic_diagnostic` | `analysis.aberration` | 14 / 14 | 1 | 14 / 14 | 2 / 2 |
| `eval_image_plane_wfe` | `analysis.image_plane_wfe` | 24 / 24 | 4 | 24 / 24 | 2 / 2 |
| `plot_lens_layout` | `analysis.plotting` | 35 / 35 | 14 | 35 / 35 | 4 / 4 (2 per call) |
| `fit_canonical_polynomials` | `propagators.asymptotic_canonical_fit` | 596 / 596 | 16 | 596 / 596 | 2 / 2 |
| `fit_hf_polynomials` | `propagators.asymptotic_canonical_fit` | 590 / 590 | 8 | 590 / 590 | 2 / 2 |
| `apply_real_lens_traced` | `elements` | 2 / 2 | 2 | 2 / 2 | 4 / 4 (2 per call) |
| `apply_real_lens_maslov` | `elements` | 2 / 2 | 2 | 2 / 2 | 2 / 2 |
| **total** | | **1506 / 1506** | **16 of 16 move** | **1506 / 1506** | **43 / 43** |
| **(missing)** | `elements._lens_real` | **`apply_real_lens` -- VR2-D1** | moves (2.84e-13) | -- | 1 call, both keywords `<omitted>` |

Fixtures: a spherical doublet (0.0731 / -0.0437 / -0.1013, N-BK7 + N-SF5), an
ASPHERIC singlet (conic -0.62 / -1.7 with 4th and 6th-order terms -- the
control for `sphere_normal`, still moved by `renormalize`), and a two-SPHERICAL
MIRROR system (-0.2500 / -0.0900, `is_mirror=True` -- the analytic route
through `_reflect`).  None of the three is the round-2 probe's.

The three entry points that trace TWICE per call (`paraxial_focus_world`,
`plot_lens_layout`, `apply_real_lens_traced`) thread both calls: the spy's set
of `(sphere_normal, renormalize)` pairs has exactly ONE element per entry
point in every arm.

`jax_trace.py` is byte-identical to `49ddf4bd`
(`f9b49e6f718ff16c3531128990f30f1ba5ad1c18be5f8bd57a0441304e0d8474`); no file
with `jax` in its name under `lumenairy/` changed on the branch, and no diff
line in any changed library file mentions `jax`.

A route-less exported caller added to a scratch copy of the package
(`spot_centroid_quick` in `raytrace/ray_fan.py`, re-exported from
`raytrace/__init__.py`, tracing with no keyword) is **named by the census**:
2 failed, both census arms.

---

## 3. What the mutation runs say

Windows, both C2 test files (58 ids), one fresh `git archive` per mutant:

| mutant | result | verdict |
|---|---|---|
| M0 control | 58 passed | -- |
| M1 `ray_fan_data` loses `sphere_normal` from its SIGNATURE | **4 failed** (both census arms, VERIFY-WP-C2's pin, `none_stamps_nothing[ray_fan_data]`) | the round-2 "3 failed" reproduces and is one arm stronger here |
| M2 the keyword threaded but DROPPED before the inner trace (`ray_fan_data`) | **1 failed** (`none_stamps_nothing[ray_fan_data]`) | caught |
| M3 `_way_back_kwargs` maps `None` to `'generic'` | **9 failed** | caught |
| M4 `_library_trace_default` returns the wrong route for `renormalize` | **58 passed** | **SURVIVES -- VR2-D3** |
| M5 the JAX tracer GAINS the NumPy clamp | **2 failed** | the D6 decision arm is two-sided on the JAX side |
| M6 the NumPy tracer LOSES its clamp | **11 failed** | and on the CPU side |
| M7 a new route-less exported entry point | **2 failed** | the census names it |
| M8-M11 the forward dropped in `_lens_traced` / `asymptotic_canonical_fit` / `image_plane_wfe` / `aberration` | **58 passed** each | **SURVIVE -- VR2-D4** |

---

## 4. Gate runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line, `-q -p no:randomly --capture=sys -rf`.

| run | files | result |
|---|---|---|
| Windows: the 30-file gate the round-2 report names + this verification's own file | 31 | **1629 passed, 1 skipped, 0 failed** in 13:27 |
| WSL: a 17-file subset -- both C2 files, this verification's own, the B9 family, `test_audit_propagation.py`, `test_niche_audit_w6_asymptotic.py`, `test_niche_d3_guards.py`, five raytrace files, `test_public_api.py`, `test_audit_except_budget.py`, the durations and census gates | 17 | **540 passed, 1 skipped, 1 failed** in 15:30 -- the one failure is the pre-existing `test_installed_metadata_version_matches_source_version` |
| WSL: `tests/unit/test_verify_c2_round2.py` alone | 1 | **12 passed, 1 skipped** (the `EDITED_IN_PLACE` arm skips: `git` cannot resolve a Windows worktree from WSL) |
| Windows: `tests/unit/test_verify_c2_round2.py` | 1 | **13 passed** in 13.8 s |
| WSL: `test_public_api.py` + `test_v5_3_2_walker_source_line_citation.py`, on the TIP | 2 | 3 failed, 17 passed, 1 skipped |
| WSL: the same two files on this verifier's own `git archive 49ddf4bd` | 2 | **the same 3 failed**, 17 passed, 1 skipped -- pre-existing |
| mutation matrix, Windows | 12 arms | above |
| mutation matrix, WSL | 5 arms | M1 **4 failed**, M4 **58 passed (survives)**, M5 **2 failed**, M6 **11 failed**, M8 **58 passed (survives)** -- identical to Windows on all five |

WSL `ruff check lumenairy/ tests/ scripts/ validation/probe_verify_c2_round2/`:
**All checks passed!**
`python -m mypy` (no args): **Success: no issues found in 33 source files.**
`python scripts/record_history_fingerprints.py --check`: **OK: every history
document matches its module.**
`python scripts/reanchor_citations.py --check --base f4f18851 --block "[5.47.0]"`:
**0 re-anchored, no `EDITED_IN_PLACE` refusal, exit 0.**  (With the tool's
DEFAULT base `96cb2096` it reports 28 re-anchored and 21 `NEEDS A HUMAN`
notes, all of them in `optimize/core.py` and `elements/lenses.py` and none in
a WP-C2 file; that is the pre-existing state of the whole-CHANGELOG sweep and
is outside this chain's scope.)

`.test_durations`: the 13 ids of `tests/unit/test_verify_c2_round2.py` were
measured serially with the three BLAS variables on the command line through
`pytest-split --store-durations` and spliced; **16 654 -> 16 667** entries, 13
added, 0 updated, re-parsed as JSON, the file's existing key order preserved.
`test_audit2609_a15a_durations_staleness.py`: **4 passed** after the splice.

---

## 5. Ship recommendation

**SHIP the whole C2 chain** -- `eadc67ba` -> `61ffe596` -> `8ac607ee` -- with
**VR2-D1 and VR2-D4 actioned before the release note is written** and VR2-D2,
VR2-D3, VR2-D5, VR2-D6 and VR2-D7 filed.

Round 2 closed twelve defects and this verification reproduced eleven of them
on its own instruments, several with tighter numbers than the round-2 report
claims.  The two default flips remain sound and are now better evidenced than
they were after round 1:

* `sphere_normal='analytic'` is **1.50 ULP against the generic route's 1.75**
  out to `0.95 abs(R)` against an exact-input oracle cross-checked by a second,
  algebraically independent formulation, and is never worse by more than one
  unit at any of 3276 points there;
* the way back is **1506 / 1506 arrays byte-identical archive to archive** over
  sixteen entry points, three prescriptions including a mirror and an asphere,
  in a second process on a read-only tree, on both builds, and the sentinel is
  shown to stamp nothing at EVERY internal trace call by a spy rather than by
  inference;
* round 2 moved no default: the round-1 verifier's 594-key set is **594 / 594
  identical** `61ffe596` against the tip on both builds.

The two that should not go out unqualified are both about REACH rather than
about arithmetic.  **VR2-D1**: the census is one entry point short because an
import alias hides it, and the entry point it hides -- `apply_real_lens` -- is
the plain-lens propagator most users reach for first.  **VR2-D4**: the only
in-process pin that a forwarded keyword reaches the tracer covers nine of the
sixteen, and dropping the wire in any of the other seven is invisible to the
whole suite.  Both have exact edits above and both are cheap.

VR2-D2 is a P2 that costs a docstring sentence and one `2 *` in a test, but it
is the kind of claim a consumer acts on: the release tells them to size a
tolerance from a quantity that a three-surface stack exceeds by a third.

---

## 6. What could not be measured

1. **The 368-file blast radius was not re-run**, unchanged from both previous
   rounds.  What was re-run is section 4.
2. **The 1008-array byte-identity census was not re-run.**  D12's counts are
   verified to be what the committed JSON says and what the two user-facing
   documents now carry; the arrays themselves were not regenerated.  The
   1506-array sixteen-entry-point census here supersedes its conclusion.
3. **The `EDITED_IN_PLACE` abuse matrix is Windows-only.**  Run from WSL
   against a Windows worktree the tool cannot resolve the base commit at all
   (`git rev-parse --git-dir` fails on the `.git` file's Windows path), so
   every case -- including the shipped one -- returns "no opinion" and the
   comparison is vacuous.  The decision test detects that and skips rather
   than passing vacuously.  This is the same environment condition as the two
   pre-existing walker-citation reds.
4. **`mpmath` on WSL**, unchanged.  This verification's oracles use `decimal`
   (stdlib) precisely so both builds could run them, and every oracle number
   above is measured on both.
5. **A clean absolute timing number**, unchanged.  The d3 second-direction
   cost in claim 13 is a wall-clock measurement of one floor evaluation each
   and carries the usual caveat; the DECISION it supports (the second
   direction changes the floor by 27x on WSL) is build-measured and does not
   depend on the timing.
6. **Whether any user's prescription meets the rim band.**  Unchanged: the
   band does not exist on the meridian, sampling cannot reach it, and what a
   real design meets is the pre-existing clamp (5.048 % of a rim-packed bundle
   on a ball lens or hemisphere).
