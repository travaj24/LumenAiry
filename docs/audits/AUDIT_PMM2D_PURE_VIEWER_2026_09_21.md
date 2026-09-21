# Geometry viewers for the pure staggered 2-D stack

Subject: `PMM2DStackPure.plot_geometry` and `PMM2DStackPure.plot_section`, with the
module-level styling helpers `_stag_eps_parts`, `_stag_eps_key`, `_stag_eps_label`,
`_stag_material_style` and `_stag_style_maps`, in
`lumenairy/elements/pmm/stack2d_pure.py`. Gates:
`tests/unit/test_v5_49_pmm2d_pure_viewer.py` (17). Example:
`examples/16_pmm2d_pure_geometry_view.py`.

Every claim below was measured on this build, and every gate was disarmed and re-run to
confirm it bites. Nothing was accepted by reading.

---

## 1. Why this exists

Four stack families in this library already draw themselves: `PMMStack.plot_geometry`,
`RCWAStack.plot_geometry`, `SegmentStackGeometry.plot` and
`PMM2DStackHybrid.plot_geometry`. The contract they share is stated in the first of them:
"the picture IS the model" -- the renderer reads the same `_layers` the solve consumes, so
figure and physics cannot silently diverge.

`PMM2DStackPure` had no viewer. That gap has a cost that is not cosmetic. It was found in
use, on the exp29 comb campaign: with nothing to draw the cell, the cross-section was drawn
by hand in matplotlib beside the code, and two silent approximations survived into a figure
presented as the model -- two conformal coats merged into one sleeve, and the wrong material
beneath a 1 nm adhesion layer. Both were caught within a minute of rendering the cell from
the stack object instead. A hand drawing agrees with its author's intentions, not with the
solver.

## 2. What the viewers draw

`plot_geometry(axes=None, material_names=None, material_colors=None)` draws one panel per
layer: the layer's exact `(x, y)` segmentation, every rectangle bounded by walls the solver
uses. `plot_section(y=None, ax=None, material_names=None, material_colors=None,
along='x')` draws a z cross-section at a fixed `y` (or `x`), sliced out of the same cell
arrays, with the half-spaces as labelled bands.

Both read through two helpers that are the whole contract:

* `_layer_bounds(L, axis)` returns the layer's stored wall array when it carries one -- the
  per-layer-grid path stores the full boundary array, including the two endpoints -- and
  otherwise the uniform lattice its cell shape implies, which is what the shared-grid path
  means. Nothing is resampled and no pixel grid is introduced.
* `_layer_entry(L, i, j)` returns the permittivity the solver would use in that segment,
  including the uniform, uniform-tensor and magnetic records, where the cell is implicit.

**Slant is drawn, not ignored.** A layer carrying `slant=(t_x, t_y)` is drawn as the
parallelogram that slant implies, centred on the layer, with `+t_x` moving the TOP edge
toward `+x`. This follows the public sign convention `add_layer` documents; the gate pins
the magnitude at `thickness * slant` and pins that reversing the sign reverses the drawn
shear. A viewer that drew a slanted layer as a rectangle would be the exact silent-wrong
this family is most exposed to, because the slant lives only in the record and has no other
visible consequence.

## 3. Material styling, and the defect it replaces

The first draft of this work used a categorical colormap keyed on the order materials happen
to appear in. On a copper-and-liquid-crystal cell that drew **copper green**, and it drew the
same material differently in two figures of the same device. It was rejected in review for
exactly that.

`_stag_material_style(eps)` keys colour on the physics instead:

| entry | family | within the family |
|---|---|---|
| `Re eps < 0` (a metal) | copper to bronze | darkens with `\|Re eps\|`, saturating at 150 |
| `Re eps >= 0` (a dielectric) | bone to slate | darkens with refractive index, saturating at 2.6 |
| anisotropic | its family colour | plus a hatch |

Two consequences are worth stating because they are what the gates check. The same material
is the same colour in every figure this library draws, without the caller supplying a
palette. And an anisotropic cell can never be mistaken on the page for the scalar sharing its
`eps_xx`, because the hatch is driven by `eps_yy - eps_xx` and `eps_xy` directly.

`_stag_eps_key` carries all three of `eps_xx`, `eps_yy` and `eps_xy` to nine digits, so the
same liquid crystal at two director angles occupies two legend entries rather than merging
into one. `material_names` accepts a scalar permittivity, an identity key from `material_key`, or a
sequence of `(eps, name)` pairs -- see section 5 for why the tensor-as-key form does not
exist; `material_colors={name: color}` overrides the default for named materials only.

## 4. Gates, and the fail-before that shows they bite

17 gates in `tests/unit/test_v5_49_pmm2d_pure_viewer.py`, in four groups: the picture is the
record (5), material styling is keyed on the physics (5), slant is drawn with the documented
sign (3), refusals and independence from the source (4). All pass on this build in 3.5 s.

Passing is not evidence on its own, so each claim was disarmed in memory and its gate re-run:

| behaviour disarmed | gate | result |
|---|---|---|
| categorical palette returning green for every entry | metal warm / dielectric cool | caught |
| anisotropy not marked | tensor hatched, scalar not | caught |
| identity collapsed to `eps_xx` | two director angles do not merge | caught |
| walls re-derived on a uniform lattice | rectangles sit on the stored walls | caught |
| slant zeroed before drawing | shear equals thickness times slant | caught |
| cell read from a constant instead of the record | editing the record changes the picture | caught |

**A process note that cost twenty minutes and is worth recording.** The first fail-before run
reported three of six as NOT CAUGHT, which would have meant three blind gates. The gates were
fine: the harness patched `stack2d_pure._stag_material_style` while the test module had
imported that name directly, so the test kept calling the original. Patch the binding the
test actually calls, not the one it came from. A fail-before that reports a blind gate is
itself a claim, and it has to be checked the same way as the thing it audits.

## 5. Two defects the example found, which the tests had not

Writing `examples/16_pmm2d_pure_geometry_view.py` against the finished API turned
up two things seventeen passing gates had not.

**A tensor could not be named.** The docstring offered `material_names={eps: name}`
with "eps may be a scalar or a `(3, 3)` block-form tensor", and a tensor is a numpy
array, which is unhashable and cannot be a dictionary key. Every gate had named
materials by scalar, so none of them met it. Fixed by adding the public
`material_key(eps)` and by accepting any sequence of `(eps, name)` pairs.

**Slant is all-or-nothing across patterned layers.** The example gave the finger
body a slant and left the conformal coat bands vertical, which `solve` refuses by
name: the frame offset between two differently sheared nodal grids is a real
lateral translation, exact only at a whole number of grid cells. The refusal is
correct and pre-existing, and the example now carries one slant on every patterned
layer, which is also what a conformal coat on a slanted wall actually does. Worth
recording because the viewers make the constraint visible for the first time --
before this, a mixed-slant stack looked fine until it was solved.

The general lesson is the one this feature exists to serve: an example exercises an
API the way a stranger will, and a test written by the author exercises it the way
the author already thinks.

## 6. A property that is deliberate, not a defect

Two dielectrics of nearly equal index come out nearly the same colour, because the
ramp encodes the index. Alumina at 1.746 and carbonitride at 1.781 are 2 % apart
and look it. That is the correct behaviour for a palette keyed on physics, and
`material_colors={name: color}` is the remedy when a reader needs to tell two
structurally distinct but optically similar layers apart.

## 7. Scope, and what is deliberately not done

* The viewers are geometry only. They need no source, and drawing does not disturb a solve:
  a gate solves, draws, solves again and requires bit-identical Jones.
* `plot_section` shows one cut. A staircased taper built with `add_tapered_pillar` appears as
  the slices it actually is, which is what the solver sees.
* Tapers, dispersive callables and magnetic layers draw through the same path; a dispersive
  entry is labelled by its permittivity at the stored value, not evaluated at a wavelength,
  because the geometry view is wavelength-free by construction.
* No colour is claimed to be physically calibrated. The ramps encode ordering -- metal versus
  dielectric, stronger versus weaker -- not a measurable quantity.

## 8. Provenance

Build: Windows 11, python 3.14.6, numpy 2.5.2, matplotlib Agg, `LUMENAIRY_DISABLE_JAX=1`,
BLAS pinned to one thread. Prepared 21 September 2026.
