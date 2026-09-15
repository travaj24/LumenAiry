# WP-B12 -- the FGA reference plane: `ray_transfer_jacobian`'s state is on the last surface, `fga.py` treats it as the exit-vertex plane

Repository: `D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, branch `audit-fixes-2026-09`, HEAD = the 5.47.0 release commit (the top of `CHANGELOG.md`; 5.47.0 closed).
You own `lumenairy/propagators/fga.py`, `lumenairy/raytrace/differential.py` (its docstring only unless you find the primitive itself wrong),
the test files that pin FGA fields and routes (`tests/unit/test_fga.py`, `test_fga_h4_h5.py`, `test_g1_gate_generality.py`,
`test_niche_audit_w9_dispatch2.py`, `test_niche_p8_capstone.py`, `test_niche_p7_seidel_gate.py`, `test_audit2609_a4_fga_s10.py`,
`test_audit2609_b7b_caustic_routing.py`), a new `tests/unit/test_audit2609_b12_fga_reference_plane.py`, `docs/history/` documents you touch, and
your two report files `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B12_REPORT.md` / `WP-B12_CHANGELOG.md`.  Nothing else.
Read first: `fixes/VERIFY_WP-B7b.md` (sections 3.3, 3.5, 8 -- the finding, its localisation and the measured edit), `fixes/WP-B7b_REPORT.md`
with its addendum, `fixes/WP-B7b_CHANGELOG.md`, `docs/TESTING_STANDARDS.md`, `CONVENTIONS.md` sec. 2 and 7.

## The finding (VERIFY-B7b, reproduced by two independent oracles)

`lumenairy.raytrace.differential.ray_transfer_jacobian` and `ray_transfer_jacobian_analytic` return the base ray's state ON the last surface
(`res.image_rays`), not on the exit-vertex plane; `TraceResult.at_exit_vertex()` is the step that projects it.  Four sites in `fga.py` add the
image-side leg from that state as if it were on the vertex plane -- `_fga_core` (`xv = dt.x + z_image*uxo`, `opd_tot = dt.opd + z_image*sqrt(1+u^2)`),
`_fga_coarse`, the coarse trace, and `_caustic_zone` (`z = -x/u`).  On any prescription whose LAST surface is curved every beamlet therefore
carries a spurious phase of `k * sag(r)` (7.58 waves of OPL / 0.64 um of height at the rim of an N-SF11 R = +-1.6 mm biconvex; 15.02 waves on a
bent singlet; exactly zero on a flat last surface).  Measured effect (monkey-patched projection, nothing else changed): FGA fidelity against a
brute-force Rayleigh-Sommerfeld oracle 0.1250 -> 0.9998 at WP-B7b's caustic, 0.0737 -> 0.9998 at the exit vertex, 0.1228 (power 0.352) ->
0.9999 (power 0.980) on a bent singlet; a flat-last-surface control 0.9998 -> 0.9998.  The measured edit, inlined at each site:

```python
    # ray_transfer_jacobian leaves the base ray ON the last surface; this
    # module's image leg starts at the exit-vertex PLANE.
    sag = conic_sag(dt.x, dt.y, surfs[-1].radius, surfs[-1].conic,
                    surfs[-1].aspheric_coeffs, xp=np)
    x_v   = dt.x   - sag * dt.ux
    y_v   = dt.y   - sag * dt.uy
    opd_v = dt.opd - sag * np.sqrt(1.0 + dt.ux ** 2 + dt.uy ** 2)
```

Adding the free-space leg to the Jacobian as well changed nothing beyond the fourth digit -- measure whether the JACOBIAN needs the projection
(it is a 4x4 transfer to the same reference; the sag step is a free-space propagation by `-sag` along the ray) and state the answer.

## What to do, in order

1. **Reproduce the defect on a fixture of your own** (a different glass, wavelength, NA 0.10-0.20, curved last surface) with your own oracle
   (exact conic trace of your own + brute-force Rayleigh-Sommerfeld or angular spectrum, converged, floor stated; check the exit slope vs
   direction-cosine trap), at the exit vertex and at the focus; then a flat-last-surface control where the defect must read zero.
2. **Decide the plane once, at the primitive or at the consumers -- and say why.**  Either `DifferentialTransfer` gains the vertex-plane state
   (and every other consumer is audited: grep the package for `ray_transfer_jacobian`; VERIFY-B7b found only the four `fga.py` sites treat the
   state as a vertex-plane state and `_lens_traced_multibranch.py` calls `at_exit_vertex()` explicitly), or the four sites project.  Prefer the
   change with the smaller blast radius; the primitive's docstring already states the last-surface convention (5.47.0), so keep it truthful.
3. **Ship the repair** and prove: (a) byte-identity on every FLAT-last-surface prescription (archive-to-archive against `git archive <the 5.47.0 release commit>
   lumenairy`, child processes, `lumenairy.__file__` asserted, never through pytest, never the shared tree); (b) the oracle ladder on curved
   last surfaces (yours, WP-B7b's N-SF11 biconvex, VERIFY-B7b's bent singlets), fidelity and power before and after; (c) `apply_real_lens_fga_vector`
   and `method='fga'` through `apply_real_lens_universal` follow.
4. **Re-score the caustic route with the repair in place**: the three members at the caustic on the WP-B7b fixture and on VERIFY-B7b's
   fixture V (`fga` should read ~0.9998, `phase_screen` 0.9991, `traced` 0.9995).  Do NOT move the route yourself; report the numbers and a
   recommendation (the screen is 10x cheaper; the difference is in the fourth digit).  If you can reach the H2 f/5 dual-oracle fixture
   (`test_fga.py`'s H2 rows / M6 of the G1 matrix) at a tractable grid, score the three members there and report -- that is the measurement that
   decides the `aberrated` condition.
5. **Restate the pins that move, each with its own measurement**: the G1 matrix rows, the H2 rows, the w9 / p7 / p8 numbers, the a4 S10 rows, the
   b7b rows.  Every restated number gets the oracle it was checked against in its docstring; no wall-clock asserts; derived two-sided bars.
6. **Also fix `_caustic_zone`** (same state, `z = -x/u`), and state the zone shift it removes (~0.5 % of the focal distance on these fixtures).
7. **Report** `WP-B12_REPORT.md` (fixtures, oracle controls, before/after tables, blast radius, every command + counts + durations, files) and
   `WP-B12_CHANGELOG.md` (release text with a Migration note: every `apply_real_lens_fga` / `_fga_vector` / `universal(method='fga')` field on a
   curved-last-surface prescription changes; flat last surfaces byte-identical; no way back other than the parent commit).  Re-record every
   history document you touch (`python scripts/record_history_fingerprints.py <module> --reason "..."`).

## Rules

* Every python run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`; one process at a time, in the FOREGROUND (never end a turn
  while a child process runs); `-X faulthandler` on long selections.  Scratch files under `...\scratchpad\b12\`.
* NO git write commands of any kind (no add / commit / stash / checkout / restore / reset); do not kill processes.  The orchestrator commits with an
  explicit file list -- end with the exact list of files you changed.
* Comments say what the code does now and why; never a change log (history goes to `docs/history/<module>.md`).  Finish with the report's full
  text as your final message.
