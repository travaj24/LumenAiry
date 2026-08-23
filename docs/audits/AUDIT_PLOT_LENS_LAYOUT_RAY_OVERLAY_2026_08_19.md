# AUDIT -- `plot_lens_layout` ray overlay draws the wrong transverse coordinate

Date: 2026-08-19
Component: `lumenairy/analysis/plotting.py :: plot_lens_layout`
Library version audited: 5.39.1
Found during: design-122 (exp32) prescription layout figures
Status: **REMEDIATED 2026-08-22** on branch `fix/plot-ray-overlay`.  Every
claim below was independently re-measured before the patch was written; the
section-4 fix, the section-5.1/5.2 docstring notes, the section-5.3 behaviour
change and all three section-6 tests are implemented.  See
`FIX_PLOT_LENS_LAYOUT_RAY_OVERLAY_2026_08_22.md`, which also records the one
place this audit was WRONG: section 5.3's suggested "> 10x the track length"
bar fires on ordinary catalog singlets (74.8x on a stock LA1301-C) and was
replaced with a bar that has a measured gap on both sides.

---

## 0. Summary

`plot_lens_layout(..., show_rays=True)` traces its rays correctly and then plots
them on an axis that carries none of their information. Every ray is drawn at
height 0, collapsed onto the optical axis.

The failure mode is **silent-wrong, not silent-absent**, and that is what makes
it worth a P1 rather than a cosmetic ticket. The function does not omit the
rays and it does not raise. It draws a clean horizontal line along the axis for
every ray of every field angle. A reader who does not already know the answer
sees a converging system in which all rays travel parallel to the axis and
arrive on axis -- a coherent, plausible, and entirely false optical picture.

There is **no test anywhere in `tests/` that references `plot_lens_layout`**,
which is the reason a defect this visible survived to 5.39.1.

Severity: **P1** (produces a confidently wrong figure from a public API).
Blast radius: every caller of `plot_lens_layout` with `show_rays=True`, for
every prescription. Not design-specific.
Physics impact: **none.** This is a rendering path only. No propagation,
no metric, and no pipeline result reads from it.

---

## 1. The defect

`make_fan` builds its fan entirely in the **y-z plane**. Every launched ray has
`x = 0`; both the aperture spread and the field angle live in `y` / `M`.

The overlay in `plot_lens_layout` reads `x`:

```
plotting.py:2175        hs.append(float(rb.x[k]))          # per-surface height
plotting.py:2179        Lf = float(ir.L[k])                # image-plane slope
plotting.py:2182        h_image = hs[-1] + (Lf / max(abs(Nf), 1e-9)) * dz
plotting.py:2185        ax.plot(z_world, hs, ...)
```

`rb.x[k]` is identically zero at every surface, and `ir.L[k]` -- the x-direction
cosine -- is identically zero as well, so the image-plane extrapolation adds
zero to zero. The polyline handed to `ax.plot` is flat at h = 0 from the first
surface to the image plane.

Both the per-surface term and the extrapolation term are wrong, and they are
wrong consistently, which is precisely why the output looks tidy instead of
broken.

---

## 2. Evidence

### 2.1 The fan is in y, direct from `make_fan`

```
make_fan(semi_aperture=9.0e-3, n_rays=5, field_angle=radians(2.64), wavelength=1.31e-6)

  x +0.000000e+00  y -9.000000e-03  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y -4.500000e-03  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y +0.000000e+00  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y +4.500000e-03  L +0.000000e+00  M +4.606039e-02
  x +0.000000e+00  y +9.000000e-03  L +0.000000e+00  M +4.606039e-02
```

`ptp(x) = 0`, `ptp(y) = 18 mm`. `max|L| = 0`, `max|M| = 4.6e-02`.

### 2.2 It is not vignetting

The obvious competing explanation -- that the rays die and are skipped by the
`if not r.image_rays.alive[k]: continue` guard -- was tested and refuted.
Rays alive out of 5, traced through design 122's post-DOE relay at three field
angles:

| `semi_aperture` | 0 deg | 1.32 deg | 2.64 deg |
|---|---|---|---|
| 13.30 mm | 3 | 3 | 3 |
| 9.00 mm  | 5 | 5 | 4 |
| 6.00 mm  | 5 | 5 | 5 |
| 4.50 mm  | 5 | 5 | 5 |
| 2.00 mm  | 5 | 5 | 5 |

The function's own default is `aperture_diameter / 2 = 9.7278 mm` on this
prescription, i.e. the 5/5/4 row. The rays survive. They are drawn. They are
drawn at zero.

### 2.3 The figure

Rendering design 121's post-DOE relay with `show_rays=True,
n_field_angles=5, max_field_deg=2.6392` produces a single flat line at h = 0
spanning the panel (all 25 rays superimposed; the visible colour is the
last-drawn viridis-yellow). Replacing `rb.x` with `rb.y` in a local copy of the
loop, with everything else identical -- same surfaces, same trace, same
`find_paraxial_focus` image plane -- produces the correct converging fan, 25
rays resolved, each field angle landing at its own image height.

---

## 3. Root cause

An axis-convention mismatch between two functions in the same package that are
designed to be used together:

* `make_fan` chose the y-z plane as its meridional plane.
* `plot_lens_layout` renders the x-z plane.

Neither choice is wrong on its own. `plot_lens_layout` calls `make_fan`
directly and never reconciles them.

Contributing factor: the ray loop is wrapped in a broad
`except (ValueError, RuntimeError, ZeroDivisionError, KeyError, IndexError,
AttributeError, TypeError): continue`. That is a reasonable guard for a
best-effort overlay, but combined with a defect that produces *valid* numbers
it removes the last chance of noticing -- there is nothing to catch, because
nothing throws.

---

## 4. Proposed fix

Two lines, in the block at `plotting.py:2170-2185`:

```python
-                    hs.append(float(rb.x[k]))
+                    hs.append(float(rb.y[k]))
...
-                    Lf = float(ir.L[k])
+                    Mf = float(ir.M[k])
...
-                    h_image = hs[-1] + (Lf / max(abs(Nf), 1e-9)) * dz
+                    h_image = hs[-1] + (Mf / max(abs(Nf), 1e-9)) * dz
```

`y`/`M` is unambiguously correct here because `plot_lens_layout` **builds its
own fan** via `make_fan` and never accepts caller-supplied rays. There is no
configuration in which the current `x`/`L` reading is right.

The `surface_sag_general` call on line ~2171 already uses `x**2 + y**2` and is
correct as written -- sag is rotationally symmetric, so it needs no change.

A more defensive variant, if the fan plane is ever made configurable, is to
select the coordinate with the larger `ptp` across the launched fan rather than
hard-coding either axis. Not recommended now: it hides the coupling instead of
fixing it.

---

## 5. Secondary findings

These are **not defects** -- each is documented or defensible behaviour -- but
all three cost time on this task and each produced a wrong-looking figure
before being understood. Recorded so the next caller does not re-derive them.

### 5.1 P3 -- axes are in metres with no scaling affordance

`ax.set_xlabel('z [m]')`, `ax.set_ylabel('h [m]')`. Correct and consistent with
the library's SI convention. The trap is for a caller who relabels the axes to
mm for readability without rescaling the data, putting a factor-1000 error on
the figure. A `units='m'|'mm'` keyword, or a documented note pointing at a tick
formatter, would remove a foreseeable error. Workaround: leave data in metres,
attach `FuncFormatter(lambda v, p: '%.4g' % (v * 1e3))`.

### 5.2 P3 -- `aspect='equal', adjustable='datalim'` silently defeats zooming

Set at the end of the function. Any caller `set_xlim` afterwards is discarded
with only a matplotlib console message (`Ignoring fixed x limits to fulfill
fixed data aspect`). For these designs -- ~215 mm long against ~26 mm of
aperture -- equal aspect is the right default, but a tail zoom is exactly what
one wants next. Workaround: `ax.set_aspect('auto')` after the call. Worth a
docstring line.

### 5.3 P2 -- infinite-conjugate assumption is unstated and fails loudly-but-silently

`show_rays` / `show_image_plane` trace from infinity and place the image plane
at `find_paraxial_focus(surfaces, wavelength)`. For a system with a **finite
object** this is simply the wrong model, and when the system is near-afocal fed
collimated the focus runs away. On design 122's full prescription the image
plane landed **78 metres** downstream, autoscaling the panel to ~80 m and
rendering the entire lens stack as a line at the origin.

The prescription dict carries `object_distance`, and the loader populates it
(0.0 here, because the .zmx puts the source surfaces at DISZ 0). The function
ignores it. Suggested: honour `object_distance` when non-zero, and warn when
`find_paraxial_focus` returns a value implausible against the system's own
track length (e.g. > 10x) instead of silently autoscaling to it.

### 5.4 P2 -- zero test coverage

`grep -rn "plot_lens_layout" tests/` returns nothing. No test constructs a
prescription, renders it, and asserts anything about the result. The defect in
section 1 is not subtle once looked at -- it is invisible only because nobody
looks.

---

## 6. Recommended tests

A rendering test does not need image comparison to catch this class of bug.
Assert on the artist data.

1. **Rays are not degenerate.** Render a known converging prescription with
   `show_rays=True, n_field_angles=3`. For the `Line2D` artists added by the
   ray loop, assert `ptp(ydata) > 0.1 * semi_diameter` for at least one ray per
   field angle. This fails on today's code and passes after the section-4 fix.

2. **Field angles separate at the image plane.** Assert the final `ydata` point
   of the chief ray is monotonic in field angle and spans a range consistent
   with `f * tan(theta)`. Catches a sign flip or a collapsed fan that test 1
   would let through if a single fan happened to be non-degenerate.

3. **Fan-plane coupling is pinned.** Assert directly that `make_fan` returns
   `ptp(x) == 0` and `ptp(y) > 0`. If someone later re-planes `make_fan`, this
   test names the coupling that must be updated alongside it, rather than
   letting the layout silently break again in the other direction.

---

## 7. Reproduction

```python
import numpy as np
from lumenairy.raytrace import make_fan

f = make_fan(semi_aperture=9.0e-3, n_rays=5,
             field_angle=np.radians(2.64), wavelength=1.31e-6)
assert np.ptp(f.x) == 0.0        # the fan has no x extent at all
assert np.ptp(f.y) > 0.0         # it is entirely in y
# plot_lens_layout plots f.x -> every ray renders flat on the axis
```

Full-figure reproduction, including the working `rb.y` overlay used to confirm
the fix, is in
`validation/repro_traced_carrier_122/make_layout.py` (see the block headed
`LIBRARY DEFECT WORKED AROUND HERE`). That script renders designs 121 and 122
and is the artifact this audit came out of.

---

## 8. What this audit does NOT claim

* It does not claim any propagation, metric, or pipeline result is affected.
  `plot_lens_layout` is a diagnostic renderer; nothing reads back from it. The
  design-121 and design-122 campaign results stand independent of this finding.
* It does not claim the section-4 fix has been validated in-tree. It was
  verified out-of-tree by reimplementing the loop against `rb.y` and confirming
  the fan renders correctly on two independent prescriptions (121 and 122, 25
  and 24 rays drawn respectively). The patch itself has not been applied or
  run against the test suite.
* Sections 5.1 and 5.2 describe behaviour that is correct-as-designed. They are
  logged as ergonomics, not as bugs, and should not be "fixed" in a way that
  changes existing callers' output.
