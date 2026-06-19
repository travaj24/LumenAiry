# Viewer roadmap — headless layout-render API (2026-06-19)

Add a public, GUI-free way to render a `SystemModel`'s optical-layout view (2-D primarily,
3-D optionally) to an image file, so figures/reports/docs can be generated programmatically
without launching the Designer. Companion to `designer_guide.md`.

## 1. Motivation — the gap (hit in practice 2026-06-19)

Generating a system-layout PNG headlessly today requires **hand-replicating** the GUI's
render logic. Concretely, rendering the exp10 out-coupler chain
(`source -> Mirror(out-coupler) -> QWP -> PBS -> detector`) needed a ~60-line script that:

1. set `QT_QPA_PLATFORM=offscreen`,
2. built the `SystemModel` (this part is fine — see §6),
3. instantiated `Layout2DView`, called `rebuild()`,
4. then **manually** did `QImage` + `QPainter` + `scene.render(...)` + `img.save(...)` —
   because the only existing export, `Layout2DView._export_scene`, is a **private GUI method
   driven by a `QFileDialog`** (not callable headlessly).

Two things should be library-owned instead of re-derived by every caller:
- The **render-to-file core** (it already exists, tested, inside `_export_scene` — it just
  isn't exposed without a dialog).
- The **offscreen font setup**: under `QT_QPA_PLATFORM=offscreen` the glyphs render fine but
  **text labels come out as `box` (missing-glyph) boxes** because no font is loaded. A
  built-in helper should fix this once.

The 2-D path is `QGraphicsScene` (GL-free) and is already exercised headlessly by
`tests/unit/test_v5_14_5_viewer_polarization.py`, so the rendering core is proven — this is
about exposing it cleanly, not new rendering physics.

## 2. Proposed public API

```python
from lumenairy import render_layout_2d, render_layout_3d   # top-level re-exports

render_layout_2d(system_model, path, *, scale=2, bg=None, size=None, pad=40, labels=True)
# -> writes PNG or SVG (by extension); returns the output path.
# scale: device-pixel multiplier (raster). bg: background color (default from sm.prefs).
# size: optional (w,h) override; otherwise sceneRect + pad. labels: draw element names.

render_layout_3d(system_model, path, *, camera=None, size=(1200, 900), off_screen=True)
# offscreen PyVista screenshot; documents + guards the GL caveat (see §5).
```

Both build the view internally (`Layout2DView` / `Layout3DView`), `rebuild()`, render, save.
A `QApplication.instance() or QApplication([])` guard makes them safe to call from a plain
script (no manual Qt boilerplate).

## 3. Implementation plan (small, contained)

1. **Factor the render core out of the dialog.** Split `Layout2DView._export_scene` into:
   - `_render_scene_to(target, target_rect)` — the existing `QImage`/`QSvgGenerator` +
     `QPainter` + `scene.render` logic (unchanged, already tested), and
   - `_export_scene` — the thin GUI wrapper that only does the `QFileDialog` then calls the core.
2. **Add the module-level helper** `render_layout_2d(sm, path, ...)` in `lumenairy/ui/layout_2d.py`
   that: ensures a `QApplication`, builds `Layout2DView(sm)`, `rebuild()`, calls the core, saves.
   Re-export at the top-level `lumenairy` namespace (matching `spot_diagram`, `abbe_diagram`, ...).
3. **Offscreen font setup** (§4) inside the helper so labels render headlessly.
4. **`render_layout_3d`** mirrors it on `Layout3DView` via PyVista `off_screen=True` +
   `plotter.screenshot(path)`, gated by the GL caveat (§5).
5. Effort: S-M. The rendering, the system model, and the headless 2-D test all already exist;
   this is refactor + expose + font handling + tests.

## 4. The offscreen font problem (the real polish)

Under `QT_QPA_PLATFORM=offscreen`, element labels currently render as `box` missing-glyph
boxes. The helper should select a guaranteed-available font before rendering — e.g. set an
explicit `QFont` family that Qt ships, or register a bundled `.ttf` and assign it to the
scene's text items / the painter. Validate that a rendered PNG contains readable label text
(not just non-empty pixels). This is the single most user-visible fix and belongs in the
library, not in each script.

## 5. 3-D path — offscreen GL caveat (documented, gated)

`Layout3DView` is PyVista/VTK (OpenGL). Headless `off_screen=True` screenshots work on a
machine with a GL context but **VTK hard-crashes (uncatchable) where none exists** — exactly
why `test_v5_14_5_viewer_polarization.py` gates the live 3-D render behind
`LUMENAIRY_TEST_GL=1`. So `render_layout_3d` must: attempt offscreen, document the
requirement, and fail gracefully (clear error, not a segfault) when GL is absent. The 2-D
helper is the safe default for automated pipelines; 3-D is opt-in.

## 6. Explicitly NOT needed

- **System-construction API.** `lumenairy.ui.model.{Element, SurfaceRow, SystemModel}` +
  `insert_element` / `recompute_element_frames` already build chains cleanly (the exp10
  source/Mirror/Waveplate/PBS/Detector chain assembled with no friction). No change.
- **A "polarization chain" convenience builder** (e.g. `optical_chain([...])` sugar over the
  `Element`/`SurfaceRow` boilerplate) is a nice-to-have, not part of this item.

## 7. Validation

- Extend the existing headless `test_v5_14_5_viewer_polarization.py` pattern: call
  `render_layout_2d(sm, tmp_path/'x.png')`, assert the file exists, is non-empty, and has the
  expected dimensions; assert SVG export contains the element-name text strings (label check).
- Round-trip a known system (source -> QWP -> PBS -> detector) and confirm the glyph item
  count matches the live `rebuild()` (the test already counts plate+axis+label, cube+diag+port).
- 3-D test stays GL-gated (`LUMENAIRY_TEST_GL=1`), mesh-geometry checks remain headless.

## 8. References

- `lumenairy/ui/layout_2d.py::Layout2DView`, `_export_scene` (the GUI export to factor).
- `lumenairy/ui/layout_3d.py::Layout3DView` (PyVista 3-D path).
- `lumenairy/ui/model.py::{Element, SurfaceRow, SystemModel}` (construction API, unchanged).
- `tests/unit/test_v5_14_5_viewer_polarization.py` (the headless-2-D render precedent + glyph
  item-count assertions to reuse).
- Prototype: `Metasurface_QWP/experiments/exp10_sin_coat/exp10_optical_setup.py` — the ~60-line
  hand-rolled render this helper replaces (source -> out-coupler -> QWP -> PBS -> detector).
