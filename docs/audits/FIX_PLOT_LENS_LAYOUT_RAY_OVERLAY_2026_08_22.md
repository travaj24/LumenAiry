# Remediation -- `plot_lens_layout` ray overlay drew the wrong transverse coordinate -- 2026-08-22

Branch `fix/plot-ray-overlay` off `origin/main` (a2652283, the 5.39.1 release
commit).  Closes
`docs/audits/AUDIT_PLOT_LENS_LAYOUT_RAY_OVERLAY_2026_08_19.md` -- sections 4
(the P1 fix), 5.1 / 5.2 (docstring only), 5.3 (P2 behaviour), 5.4 (the
zero-coverage hole) and 6 (the test plan).

The audit itself is committed here as well; it was untracked on `main`.

**Mount.**  Windows py3.14.6, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8,
scipy-openblas.  Matplotlib forced to `Agg` for every measurement and every
test.

**Import provenance.**  Every number below was measured against the worktree,
not an installed copy: `lumenairy.__file__` = `C:\tmp\lum_pl\lumenairy\__init__.py`,
version 5.39.1, with `PYTHONPATH=C:/tmp/lum_pl` on every invocation.

---

## S1.  The audit's claims, re-measured

Nothing in the audit was taken on trust.  Its section 7 reproduction was
re-run and its section 2 evidence re-derived from scratch.

**S1.1 -- the fan plane (audit 2.1).  CONFIRMED, to the digit.**

```
make_fan(semi_aperture=9.0e-3, n_rays=5, field_angle=radians(2.64),
         wavelength=1.31e-6)

  x +0.000000e+00  y -9.000000e-03  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y -4.500000e-03  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y +0.000000e+00  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y +4.500000e-03  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y +9.000000e-03  L +0.000000e+00  M +4.606039e-02

  ptp(x) = 0.000000e+00   ptp(y) = 1.800000e-02
  max|L| = 0.000000e+00   max|M| = 4.606039e-02
```

Matches the audit's table exactly.  `make_fan`'s default `axis='y'` is what
`plot_lens_layout` gets, since it passes only keyword arguments.

**S1.2 -- the zero survives the trace.  CONFIRMED, and STRONGER than the
audit stated.**  The audit says `rb.x[k]` "is identically zero at every
surface".  Re-measured through a full `trace` of an AC254-100-C at three
field angles:

| field | max\|x\| over `ray_history` | max\|L\| at image | max\|x\| at image |
|---|---|---|---|
| 0.00 deg | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 1.32 deg | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |
| 2.64 deg | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 |

Exactly zero, not merely small -- a meridional ray through rotationally
symmetric surfaces never acquires an x component, so this is algebraic and
build-independent.  That matters for the test below: the fail-before arm rests
on an exact zero, not on a residual that some LAPACK could move.

**S1.3 -- the rendering (audit 2.3).  CONFIRMED on two prescriptions.**
Rendering with `show_rays=True, n_field_angles=3, max_field_deg=2.64,
rays_per_fan=5` and reading the `Line2D` artists back off the axes:

| prescription | ray polylines drawn | max `ptp(ydata)` PRE-FIX | POST-FIX |
|---|---|---|---|
| AC254-100-C achromat (3 surf) | 13 | **0.000000e+00** | 1.668368e-02 m |
| N-BK7 biconvex singlet (2 surf) | 13 | **0.000000e+00** | 1.694031e-02 m |

Every ray flat at h = 0, on both.  The post-fix render is **bit-identical** to
an independent out-of-tree reimplementation of the loop against `rb.y` / `ir.M`
(the same construction the audit used) -- checked as exact float equality over
every `(xdata, ydata)` point of all 13 polylines on both fixtures, not merely
on the max.  The in-tree patch therefore reproduces the audit's out-of-tree
verification exactly.

**S1.4 -- not vignetting (audit 2.2).  CONFIRMED.**  At the function's own
default semi-aperture the rays survive: 5/5, 4/5, 4/5 alive at 0 / 1.32 / 2.64
deg on the achromat, and the same on the singlet.  They were drawn.  They were
drawn at zero.

**S1.5 -- zero coverage (audit 5.4).  CONFIRMED.**
`grep -rn "plot_lens_layout" tests/` returned nothing on `origin/main`.

**S1.6 -- blast radius, checked rather than assumed.**  The audit says the
defect is not design-specific but does not bound it.  Swept: the only other
ray-overlay renderer in the library is the GUI's `Layout2DView`
(`lumenairy/ui/layout_2d.py`), and it reads `history[si].y[r]` and
`input_rays.y[r]` -- **correctly**.  `plot_lens_layout`'s own docstring calls
itself the "standalone script/notebook equivalent" of that widget, so the two
were meant to agree and only one of them was wrong.  That is independent
corroboration that `y` is the intended convention, not a coin flip between two
defensible choices.  No other site in `lumenairy/` reads `rb.x` for a height:
the only remaining `rb.x` in `analysis/` is the `x**2 + y**2` sag argument,
which is correct.

---

## S2.  The fix (audit section 4)

`lumenairy/analysis/plotting.py`, in the ray loop of `plot_lens_layout`:

```
-                    hs.append(float(rb.x[k]))
+                    hs.append(float(rb.y[k]))
...
-                    Lf = float(ir.L[k])
-                    h_image = hs[-1] + (Lf / max(abs(Nf), 1e-9)) * dz
+                    Mf = float(ir.M[k])
+                    h_image = hs[-1] + (Mf / max(abs(Nf), 1e-9)) * dz
```

Exactly the audit's section 4, and nothing more in that block.  The
`surface_sag_general` call above it keeps `x**2 + y**2` -- sag is rotationally
symmetric, and the audit is right that it needs no change.

The audit's "more defensive variant" (pick the axis with the larger `ptp`) was
considered and REJECTED, for the reason the audit itself gives: it hides the
coupling instead of fixing it.  Instead the coupling is now named in three
places -- a block comment at the assignment site, a `Notes` paragraph in the
docstring, and a dedicated test (S4.3) that fails FIRST if `make_fan` is ever
re-planed.

---

## S3.  S5.3 -- the object conjugate and the runaway guard

Two behaviour changes, both P2, both from audit S5.3.

### S3.1  `object_distance` is honoured

When `prescription['object_distance']` is finite and `> 0`:

* the fans are launched **diverging from an object point** at
  `z = -object_distance` (new `_layout_finite_object_fan`), with the field
  angle setting the object HEIGHT `y_obj = -object_distance * tan(theta)` so
  the ray to the first vertex still makes `theta` with the axis.  Built in y-z,
  matching `make_fan`'s plane;
* the image plane is the **finite conjugate** solved at the principal planes
  (new `_layout_finite_conjugate_image_distance`), not the BFL;
* the drawn polyline carries the object-space leg, so the conjugate is visible
  rather than implied.

The conjugate derivation is the index-threaded Gauss form
`n_obj/u_pp + n_img/v_pp = 1/efl` that `analysis/image_plane_wfe.py` already
uses (its W4c note pins it against an exact real-ray oracle); it is reproduced
in `plotting.py` rather than imported so the layout renderer takes on no
dependency on the WFE stack.

Verified here against an **independent real-ray oracle** -- a near-axial ray
launched from the object point, axis crossing read past the last surface,
sharing no algebra with the ABCD derivation.  AC254-100-C at 1.31 um:

| `object_distance` | derived image_z | real-ray oracle | rel | BFL would give | error if ignored |
|---|---|---|---|---|---|
| 0.3 m | 112.679957576 mm | 112.679956725 mm | 7.6e-09 | 80.031754 mm | 32.65 mm |
| 0.5 m | 97.016026026 mm | 97.016025611 mm | 4.3e-09 | 80.031754 mm | 16.98 mm |
| 1.0 m | 87.753814577 mm | 87.753814331 mm | 2.8e-09 | 80.031754 mm | 7.72 mm |
| 2.0 m | 83.725316395 mm | 83.725316201 mm | 2.3e-09 | 80.031754 mm | 3.69 mm |

Against an 8.5 mm total track, the old infinite-conjugate answer was off by up
to four track lengths.

Missing / zero / `None` / non-finite / negative `object_distance` all keep the
historical infinite-conjugate behaviour exactly.  That is the other half of
the claim and it has its own test.

### S3.2  The runaway-focus warning -- and where the audit was WRONG

The audit proposed warning when `find_paraxial_focus` returns a value
"implausible against the system's own track length (e.g. > 10x)".

**That formulation was remeasured and rejected.**  `|image_z| / track` on
ordinary catalog glass, measured on this build at 1.31 um:

| lens | focus | track | `|image_z|/track` |
|---|---|---|---|
| LA1301-C | 254.30 mm | 3.40 mm | **74.79** |
| LA1509-C | 202.72 mm | 3.60 mm | **56.31** |
| LA1050-C | 99.54 mm | 4.10 mm | **24.28** |
| AC254-200-C | 136.50 mm | 6.00 mm | **22.75** |
| AC254-100-C | 80.03 mm | 8.50 mm | 9.42 |

A 10x-of-track bar fires on four of six stock catalog lenses, purely because
a thin lens is thin.  It is not a plausibility test; it is an f-number test.

Adding the aperture to the scale does not save it either: `|image_z| /
max(track, aperture)` is 10.01 for LA1301-C -- straddling the bar -- and 390.8
for a legitimate f/100 plano-convex, while the audit's own pathological case
(design 122, 78 m against a 215 mm track and ~26 mm aperture) reads 363.  On
that metric the pathology and a slow singlet **overlap**, so no bar separates
them.

**What was shipped instead.**  The scale is
`max(track_length, aperture, |efl|)` and the bar stays at the audit's 10x.
Including `|efl|` collapses the ratio to `|A|` -- the ABCD ray-height
magnification -- whenever the focal length dominates, and that is the quantity
that actually distinguishes "long focus" from "no focus".  Measured envelope
over every in-tree prescription:

| lens | ratio |
|---|---|
| LA1050-C | 0.9733 |
| LA1509-C | 0.9883 |
| LA1301-C | 0.9912 |
| AC254-050-C | 0.8677 |
| AC254-100-C | 0.9511 |
| AC254-200-C | 0.9806 |
| f/1 biconvex | 0.8928 |
| f/40 plano-convex | 0.9995 |
| f/100 plano-convex | 0.9998 |

Every focusing system sits at or below 1.0.  The bar at 10 is a full decade
above that envelope.  Above the bar, the S5.3 pathology -- a near-afocal
system fed collimated -- is unbounded: a 20x Galilean expander detuned off its
afocal separation measures

| detune | focus | efl | ratio |
|---|---|---|---|
| 0.03 | 4440.16 mm | 212.06 mm | 20.94 |
| 0.01 | 7351.99 mm | 357.65 mm | 20.56 |
| 3e-3 | 9614.50 mm | 470.78 mm | 20.42 |
| 1e-3 | 10549.94 mm | 517.55 mm | 20.38 |

asymptoting to the expander's own magnification.  That is a design constant,
not a knife edge -- decades of gap on both sides, as
`docs/TESTING_STANDARDS.md` rule 5 requires.

The warning names the system, the ratio and the escape hatch
(`show_image_plane=False, show_rays=False`, or a finite `object_distance`).

### S3.3  Adjudication of the audit's "contributing factor"

Audit section 3 flags the broad
`except (ValueError, RuntimeError, ZeroDivisionError, KeyError, IndexError,
AttributeError, TypeError): continue` around the ray trace.

**It stays.**  A best-effort overlay should not take the whole figure down
because one field angle vignettes out, and the audit itself calls it "a
reasonable guard".  It was never the cause here: the defect produced valid
numbers, so there was nothing to catch.

**It cannot swallow the new warning**, for two independent reasons, and the
second is the durable one:

1. `warnings.warn` does not raise, and even under `-W error` the raised
   `UserWarning` is not a member of that caught tuple.
2. The warning is emitted **structurally outside** the ray loop's `try`, in
   the image-plane derivation block -- so point 1 stays true even if the tuple
   is ever widened.

Both are recorded in a comment at the warning site so a future widening of the
tuple does not quietly re-open the hole.

---

## S4.  Tests (audit section 6)

New file `tests/unit/test_plot_lens_layout_ray_overlay.py`, 11 tests.  All
assertions are on `Line2D` artist data; no image comparison anywhere (fragile
across matplotlib versions, and unable to say WHICH number was wrong).  `Agg`
backend, no per-build facts, every bar carrying its derivation and its
measured values in the docstring.

Two independent fixtures throughout -- a cemented catalog achromat (3
surfaces, 2 glasses) and a bare biconvex singlet (2 surfaces, 1 glass) --
because a fix that happened to work on one shape is not enough.  Designs 121
and 122 are not in-tree, so these stand in for them.

Ray artists are identified by point count, read off the running build:
`plot_lens_layout` draws surface curves at 81 points (hard-coded), the axis
and image lines at 2, and rays at `n_surfaces + 1` (+1 more for the
object-space leg).  The helper asserts that identifying length is unambiguous
rather than assuming it.  Fans are grouped by field angle via their viridis
RGBA -- the colour IS the field-angle label -- and the chief ray of a fan is
the one launched at zero pupil height.

**6.1 `test_layout_ray_overlay_is_not_degenerate`** [2 params].  At least one
ray per field angle with `ptp(ydata) > 0.1 * semi_diameter`.  Carries an
explicit FAIL-BEFORE arm that *demonstrates* the pre-fix premise from the
running build's own trace (max|x| and max|L| exactly 0) rather than asserting
it from memory -- and hard-fails with a "re-derive this test" message if that
premise ever stops holding.

**6.2 `test_layout_chief_ray_heights_follow_f_tan_theta`** [2 params].  Chief
heights strictly increasing in field angle AND matching
`n_obj * efl * tan(theta)`.  Catches what 6.1 lets through: a sign flip, or
one non-degenerate fan drawn three times.

**6.3 `test_make_fan_plane_is_y_z_as_plot_lens_layout_assumes`**.  `ptp(x)==0`,
`ptp(y)>0`, `max|L|==0`, `max|M|>0`, with a docstring that names the renderer
that must be updated alongside any re-planing of `make_fan`.

**S5.3 arms** (5 more): the finite conjugate against the real-ray oracle at
three object distances plus the drawn-artist checks; the infinite-conjugate
control across five spellings of "no object distance"; the runaway warning
firing on an engineered near-afocal system (detune ladder scanned, ratio
measured on the build, hard-fail only when the ladder is exhausted); and the
two-sided silent control over eight ordinary prescriptions.

### Bars, and their gaps

| test | bar | measured below | measured above |
|---|---|---|---|
| 6.1 | `0.1*semi` = 1.270000e-03 m | pre-fix **exactly 0** (unreachable, not merely far) | 1.67e-2 m, 13x the bar |
| 6.2 | rel < 2e-2 | -- | measured 1.9e-5 .. 1.1e-4 (real distortion); a sign flip gives 2.0, a collapse 1.0 |
| S5.3 conjugate | rel < 1e-6 | oracle's own paraxial-limit floor ~1e-8 (`y_pupil`=1e-5 m against ~50 mm radii) | measured 2.3e-9 .. 7.6e-9 |
| S5.3 runaway | 10x | sane envelope <= 1.0 over 9 prescriptions | 20.4x, the expander magnification |

### Fail-before, run

The section-4 patch was reverted in place and the suite re-run:

```
FAILED test_layout_ray_overlay_is_not_degenerate[AC254-100-C achromat]
FAILED test_layout_ray_overlay_is_not_degenerate[N-BK7 biconvex singlet]
FAILED test_layout_chief_ray_heights_follow_f_tan_theta[AC254-100-C achromat]
FAILED test_layout_chief_ray_heights_follow_f_tan_theta[N-BK7 biconvex singlet]
4 failed, 7 passed
```

with

```
AssertionError: AC254-100-C achromat, field-angle group 0: no ray spans more
than 0.1*semi_diameter (1.270000e-03 m); the largest ptp(ydata) over 5 rays is
0.000000e+00 m.
AssertionError: N-BK7 biconvex singlet: chief-ray image heights [0.0, 0.0, 0.0]
are not strictly increasing with field angle.
```

The patch was then restored and all 11 pass.  The seven that pass on both arms
are the fan-plane pin (which is a `make_fan` property, correct before and
after) and the S5.3 arms (which assert on z positions and the image-plane
vline, not on ray height).

---

### Regression evidence

The harness caps any single command at 10 minutes, and `tests/unit` is
~12,400 tests, so the full suite could not be run to completion in one
invocation.  What WAS run, all with `PYTHONPATH` pinned to the worktree and
`MPLBACKEND=Agg`:

| run | result |
|---|---|
| the new file, 11 tests | **11 passed** |
| `-k "plot or layout or seidel or paraxial or raytrace or analysis"` -- every module this change touches or imports | **923 passed, 1 skipped** (4m04s, ran to completion) |
| every test file that imports `lumenairy.analysis.plotting` (6 files, found by grep) | **329 passed** |
| full `tests/unit`, `-p no:randomly` (deterministic order) | reached **54%** (~6,700 tests), **zero failures**, killed by the 10-minute cap |
| `tests/unit` files 243-363, `-n 16` | reached **91%**, zero failures, killed by the cap (the trailing `F` in that log is `[gw0] node down: Not properly terminated` -- the harness killing an xdist worker, not a test) |

Blast radius bounds what that leaves open.  The change touches exactly one
public function plus two module-private helpers in the same file; the only
module-level addition is a float constant, with no import-time side effects.
`grep -rn "plot_lens_layout" tests/ validation/` returns nothing outside the
new file, so **no other test in the tree can exercise the changed code path**,
and every test that so much as imports the module is in the 329 above.

---

## S5.  Ergonomics -- docstring only (audit 5.1, 5.2)

Both are correct-as-designed; the audit says so and says they must not be
"fixed" in a way that changes existing callers' output.  **Zero code change.**
Two `Notes` paragraphs were added to `plot_lens_layout`:

* **Units (5.1).**  Axes are metres, per the library's SI convention.  The
  trap is relabelling to mm without rescaling.  The note gives the
  `FuncFormatter(lambda v, p: '%.4g' % (v * 1e3))` recipe verbatim.
* **Aspect / zoom (5.2).**  `set_aspect('equal', adjustable='datalim')` is the
  right default for a ~215 mm track against ~26 mm of aperture, but it
  silently discards a caller's later `set_xlim` with only the matplotlib
  console message `Ignoring fixed x limits to fulfill fixed data aspect`.  The
  note gives `ax.set_aspect('auto')` as the release.

A `Warns` section documents the new S5.3 warning, and a `Notes` paragraph
documents the object-conjugate behaviour and the y-z overlay plane.

---

## S6.  The workaround site

`validation/repro_traced_carrier_122/make_layout.py` -- the script the audit
came out of, carrying the `LIBRARY DEFECT WORKED AROUND HERE` block and its
local `overlay_fan` -- is **UNTRACKED** in the repository (it sits inside an
untracked `validation/repro_traced_carrier_122/` directory alongside ~75 MB of
`.npz` intermediates).  It was therefore **left alone**, per the branch's
scope.

Whoever owns that script can now delete `overlay_fan` entirely and call
`plot_lens_layout(..., show_rays=True)` directly: the library path is
bit-identical to the workaround's output (S1.3), and design 122's runaway
78 m image plane -- the reason its panels were unreadable -- now raises the
S3.2 warning instead of silently autoscaling.

---

## S7.  What this change does NOT do

* No physics changes.  `plot_lens_layout` is a diagnostic renderer; nothing in
  the propagation, metric, or optimisation paths reads back from it.  The
  design-121 and design-122 campaign results are untouched, exactly as the
  audit says.
* No change for callers without `object_distance`.  The infinite-conjugate
  path is byte-for-byte the old one apart from the coordinate fix, and has its
  own test across five spellings of absent.
* No change to `make_fan`.  Its y-z plane and its per-axis field convention
  are load-bearing for `ray_fan.py`'s RT-5 invariant (see its R-15 note); the
  renderer was the side that was wrong.
* The audit's "select the axis with the larger ptp" variant was not adopted.
