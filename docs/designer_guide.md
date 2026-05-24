# LumenAiry Designer Guide

**Designer version: 5.4.0** (co-versioned with the core library; the
Designer's title-bar and About dialog read `lumenairy.__version__` at
runtime).

This guide documents the LumenAiry Designer GUI's dock surface and
how each dock maps to the underlying core library API.  Use it as a
reference when wiring a workflow in the GUI, or when extending the
GUI with a new dock.

## Launching

```bash
python -m lumenairy.ui
```

Or, from Python:

```python
from lumenairy.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication
import sys

app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec())
```

## Architecture

The Designer is a PySide6 application with the following structure:

* `lumenairy/ui/main_window.py` -- Application shell, dock registry,
  menu bar, toolbar, status bar, workspace tab system.
* `lumenairy/ui/model.py` -- `SystemModel` -- shared state across
  docks (surfaces, sources, wavelengths, field angles); raytrace
  dispatch.
* `lumenairy/ui/workspace.py` -- Tabbed dock groupings.
* `lumenairy/ui/<topic>_dock.py` -- One dock per analysis or
  workflow topic; each is a `QWidget` subclass that
  `main_window.py` wraps in a `QDockWidget`.

All docks share three patterns:

1. **Constructor signature:** `__init__(self, system_model, parent=None)`.
2. **Worker threads:** long-running library calls run on a
   `QThread` worker that emits Qt signals for progress + result.
3. **Cancellation:** workers that wrap `design_optimize` /
   `monte_carlo_tolerancing` / chunked phase-retrieval honour the
   v4.13.1 `CancellableProgress` polling protocol via the dock's
   Stop button.

## Dock inventory (v5.4.0)

The Designer ships 37 docks organised by function.  Tags after the
dock name show the dock's introduction or last substantive update
release.

### Core system editing

| Dock | File | Library backing |
|------|------|-----------------|
| Surface table | `surface_table.py` | `elements.lenses`, `model.SystemModel.prescription` |
| Element table | `element_table.py` | `elements/elements.py` |
| Surface editors | `surface_editors.py` | `elements.freeform`, `elements.coatings` |
| Layout 2D | `layout_2d.py` | `raytrace.layout` |
| Layout 3D | `layout_3d.py` | `raytrace.layout`, `vtkmodules` |
| Library manager | `library_dock.py` | `user_library.py` |
| Materials / glass map | `materials_dock.py`, `glass_map_dock.py` | `glass.GLASS_REGISTRY`, `glass.GLASS_VALIDITY` |

### Optimisation + design exploration

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Optimizer | `optimizer_dock.py` | `optimize.design_optimize`, `optimize.design_optimize_multi_objective`, `optimize.Constraint`, `WAVE_PROPAGATOR_REGISTRY` | **v5.4 expansion:** Advanced parameters group exposes method (11 options), constraints editor, state_file checkpoint, hess, wave_propagator, precision, multi-objective Pareto (pymoo), max_iter override.  Stop button via `CancellableProgress`. |
| Sensitivity | `sensitivity_dock.py` | `optimize/sensitivity.py` | FD d(merit)/d(var) rankings. |
| Tolerance (Monte Carlo) | `tolerance_dock.py` | `monte_carlo_tolerancing`, `raytrace.world.trace_world` | Stop button + per-trial cancellation. |
| Multi-config / zoom | `multiconfig_dock.py` | `optimize.multiconfig`, `Configuration`, `create_zoom_configs` | Joint-optimisation across configurations.  Stop button. |
| Sliders | `slider_dock.py` | `model.optimizable_variables` | Live-trace as you drag. |
| Snapshots | `snapshots_dock.py` | n/a (internal) | A/B compare named design states. |

### Wave-optics + propagators

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Wave optics | `waveoptics_dock.py` | `propagators.asm`, `propagators.fresnel`, `propagators.hf`, `propagators.mhs`, `apply_real_lens_traced` | Method + backend selector. |
| Through-focus | `through_focus_dock.py` | `analysis.through_focus` | PSF metrics across axial sweep. |
| Wavefront map (v5.4 NEW) | `wavefront_map_dock.py` | `analysis.plot_wavefront`, `analysis.opd`, `io.load_field_h5` | OPD source (current system / loaded HDF5 / live optimiser run), aperture overlay, units (waves / um / mm / nm), 5 colormaps, RMS / PV annotation. |
| Field browser | `field_browser_dock.py` | `io.storage` (HDF5 / Zarr) | Plane browser. |
| Caustic | `caustic_dock.py` | `analysis.caustic_diagnostic` | Chief-ray caustic crossings. |

### Imaging metrics

| Dock | File | Library backing |
|------|------|-----------------|
| PSF / MTF | `psf_mtf_dock.py` | `analysis.psf_mtf_otf`, `polychromatic_psf`, `mtf_cutoff`, `encircled_energy_*` |
| Spot field | `spot_field_dock.py` | `raytrace.ray_fan` |
| Footprint | `footprint_dock.py` | `raytrace.world` |
| Distortion | `distortion_dock.py` | `raytrace.world.trace_world` |
| Zernike | `zernike_dock.py` | `analysis.zernike.zernike_decompose` | **v5.4 polish:** Normalization dropdown + Weighting group (`none` / `circular_aperture` / `from_file`).  Library is OSA-locked; non-OSA selections emit a UI warning. |
| LG aberration tensor | `lg_aberration_dock.py` | `propagators.asymptotic_aberration_tensor`, `decompose_lg` |
| Rayfan | `rayfan_dock.py` | `raytrace.ray_fan` |
| Interferometry | `interferometry_dock.py` | `analysis.interferometry.simulate_fringes` |

### Polarisation + vector diffraction

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Jones pupil / Stokes | `jones_pupil_dock.py` | `elements.polarization.compute_jones_pupil`, `stokes_parameters` | **v5.4 expansion:** QTabWidget with 3 tabs -- Jones pupil (existing 2x4 amplitude+phase), Stokes (S0 / S1 / S2 / S3 heatmaps), Polarisation derived (DOP / DOLP / DOCP). |
| Richards-Wolf focus | `richards_wolf_dock.py` | `propagators.vector_diffraction.debye_wolf_psf` |

### Partial coherence + sources

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Coherence | `coherence_dock.py` | `analysis.coherence.koehler_image`, `extended_source_image`, `mutual_coherence`, `sources.core.create_*_schell_source` | **v5.4 expansion:** 4 tabs (Schell source / Koehler / Extended source / Mutual coherence). |

### Phase retrieval

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Phase retrieval | `phase_retrieval_dock.py` | `analysis.phase_retrieval.gerchberg_saxton`, `error_reduction`, `hybrid_input_output` + JAX twins | **v5.4 expansion:** algorithm dropdown (6 options), max-iter + tolerance + HIO beta + amplitude bounds + phase-wrap dropdown, live convergence plot, reconstruction preview.  Stop button. |

### Adaptive optics (v5.4 NEW)

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| AO closed loop | `ao_dock.py` | `analysis.ao.ao_closed_loop`, `DeformableMirror`, `LeakyIntegrator` | **v5.4 NEW.**  DM config (actuator count, modal basis, stroke, coupling), WFS config (type, subaperture grid, noise sigma), leaky-integrator controller (gain / leak / tol), input source (random Kolmogorov / loaded file / manual Zernike).  Live convergence + DM-command + residual-phase heatmaps.  Worker single-steps the helper. |
| Shack-Hartmann | `shack_hartmann_dock.py` | `analysis.ao.shack_hartmann` |

### Coronagraph (v5.4 NEW)

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Coronagraph workflow | `coronagraph_dock.py` | `analysis.coronagraph.coronagraph_contrast_curve`, `elements.coronagraph.apply_apodized_pupil`, `apply_lyot_focal_plane_mask`, `apply_lyot_stop` | **v5.4 NEW.**  4-stop chain builder (apodised pupil -> Lyot focal mask -> Lyot stop -> image plane).  Per-stop profile dropdowns (gaussian / cosine / super-gaussian / uniform for apodisers; hard / gaussian / 4-quadrant / 8-octant for focal masks).  log10(contrast) vs lambda/D plot + per-stop intensity previews + total throughput. |

### Coatings (v5.4 NEW)

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Thin-film coatings | `coatings_dock.py` | `elements.coatings.coating_reflectance`, `quarter_wave_ar`, `broadband_ar_v_coat` | **v5.4 NEW.**  Stack editor with material + thickness table.  Substrate + ambient + lambda sweep + AOI + polarisation controls.  R(lambda) plot.  3 quick-template buttons (MgF2 QWL AR, broadband V-coat optimiser, Bragg HR).  7-material hardcoded refractive-index database. |

### Operator algebra (v5.4 NEW)

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Operator algebra | `algebra_dock.py` | `algebra.Operator`, `CompositeOperator`, `FreeSpace`, `ThinLens`, `CylindricalLens`, `Magnify`, `FourierTransform`, `Aperture`, `GaussianAperture` | **v5.4 NEW.**  Tree-view chain editor.  Per-operator parameter dialogs.  ABCD matrix display per operator + full-system.  EFL extraction.  From-prescription populator.  Apply chain to current field.  JSON save / load. |

### Ghost + stray light

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Ghost analysis | `ghost_dock.py` | `analysis.ghost.enumerate_ghost_paths`, `ghost_analysis`, `non_sequential_stray_light` | **v5.4 expansion:** path enumeration QTableWidget, 4 filter knobs (max bounces, min transmittance, min energy ppm, sort-by), top-10 bar chart, per-path spot diagram, total stray-light budget, CSV export. |

### Diffractive optical elements + gratings

| Dock | File | Library backing |
|------|------|-----------------|
| Thin grating | `thin_grating_dock.py` | `elements.thin_grating.thin_grating_efficiency_1d`, `grating_efficiency_vs_wavelength` |

### Metrology + freeform fitting (v5.4 NEW)

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Chebyshev fit | `chebyshev_fit_dock.py` | `_math.chebyshev` (for evaluation; fit is inline via `numpy.polynomial.chebyshev.chebvander2d` + `np.linalg.lstsq`) | **v5.4 NEW.**  Specialised metrology dock.  Loads measured profilometer / interferometer height map; fits to 2-D Chebyshev coefficients; emits `freeform_type='chebyshev'` surface format. |

### Infrastructure + workflow

| Dock | File | Library backing | Notes |
|------|------|-----------------|-------|
| Log viewer | `log_viewer_dock.py` | `_logging.get_logger` hooks on `apply_real_lens_traced`, `design_optimize`, `monte_carlo_tolerancing` (v5.3.2) | **v5.4 NEW.**  5000-line capped QPlainTextEdit + level filter (DEBUG-CRITICAL) + module filter + pause/resume + clear + save + find-next.  Implements `_QSignalLogHandler` (QObject + logging.Handler dual inheritance). |
| REPL | `repl_dock.py` | `IPython.embed` or fallback | Embedded Python REPL with model + numpy pre-bound. |
| Command palette | `command_palette.py` | n/a | Global command search. |
| Diagnostics | `diagnostics.py` | `lumenairy.diag` | System-state diagnostic readout. |
| Welcome | `welcome_dock.py` | n/a | Empty-state guidance + recent files. |

## Cancellation protocol (v4.13.1 -> v5.4 GUI)

Long-running library calls support cooperative cancellation via the
`CancellableProgress` protocol.  In the Designer:

* Run a long operation from a supporting dock (Optimizer,
  Tolerance, Phase retrieval, Multiconfig).  A Stop button appears
  next to the progress bar.
* Click Stop.  The worker calls `progress.cancel()`, which sets
  `progress.should_stop = True`.  The library polls this between
  scipy iterations (in `design_optimize`), between Monte Carlo
  trials (in `monte_carlo_tolerancing`), or between chunked
  iterations (in phase retrieval, CHUNK_SIZE=10).
* On the next poll, the library returns cleanly and the worker
  emits a `cancelled` signal.  The dock's status label shows
  "Cancelled by user".

If you're adding a new dock that wraps a long-running library
call, follow the pattern from `optimizer_dock.py` (`OptimizeWorker`)
or `tolerance_dock.py` (`ToleranceWorker`).

## Logging telemetry (v5.3.2 hooks -> v5.4 viewer)

The library emits INFO-level per-iteration telemetry on three
long-running paths:

* `apply_real_lens_traced` -- per-Newton-iteration residual + grid
  shape entry log
* `design_optimize` -- per-scipy-iteration progress + entry log
* `monte_carlo_tolerancing` -- per-trial log

Default-quiet: the library installs a `NullHandler` on the
`'lumenairy'` root logger at import time, so silent unless a
handler attaches.  The v5.4 `LogViewerDock` attaches a Qt-bridged
handler when the dock opens (and detaches when it closes), so
opening the Log viewer dock surfaces all telemetry from then on.

To opt in programmatically without the GUI:

```python
import logging
logging.getLogger('lumenairy').setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
```

## Extending the GUI

To add a new dock:

1. Create `lumenairy/ui/<topic>_dock.py` subclassing `QWidget`.
2. Constructor `__init__(self, system_model, parent=None)`.
3. If the dock wraps a long-running library call, use a `QThread`
   worker and wire `CancellableProgress`.
4. Add a registration block in `lumenairy/ui/main_window.py`
   following the existing pattern (typically near the related
   dock topic).  Use the `dock(title, widget, area, key)` helper.
5. Add an entry to the dock inventory table in this file.

## Audit context

The v5.4.0 GUI surface was driven by the
`AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24` audit which identified
14 P1 / P2 / P3 wiring gaps where library features had no GUI
surface.  All 14 items shipped in v5.4.0; see the v5.4.0 CHANGELOG
entry for the per-item closure detail.

The audit also confirmed **zero broken connections** in the
existing dock-to-library wiring -- the v5.0 / v5.1 / v5.2 / v5.3
library reorganisations preserved the public-shell re-export
pattern across four major releases.

---

For per-release dock changes, see `CHANGELOG.md`.  For library API
reference, see `README.md` + the wiki.
