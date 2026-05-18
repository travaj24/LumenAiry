"""
SystemModel — element-based optical system state.

The central data model uses *elements* (lenses, mirrors, DOEs) rather
than raw surfaces.  Each element owns its internal surfaces and stores
its position as a distance from the previous element.

Author: Andrew Traverso
"""

import copy
import json
import os

from PySide6.QtCore import QObject, Signal
import numpy as np

from ..raytrace import (
    Surface, surfaces_from_prescription, system_abcd,
    trace, make_rings, make_fan, make_ray, spot_rms, spot_geo_radius,
    find_paraxial_focus, TraceResult, seidel_coefficients,
)
from ..raytrace.core import trace_world
from ..glass import get_glass_index, GLASS_REGISTRY


# ════════════════════════════════════════════════════════════════════════
# SurfaceRow — internal surface within an Element
# ════════════════════════════════════════════════════════════════════════

class SurfaceRow:
    """One refracting/reflecting surface inside an Element.

    All spatial values in mm.  ``thickness`` here means the *internal*
    distance to the next surface within the same element (glass path),
    NOT an air gap to another element.
    """

    def __init__(self, radius=np.inf, thickness=0.0, glass='',
                 semi_diameter=np.inf, conic=0.0, surf_type='Standard',
                 radius_y=None, conic_y=None):
        self.radius = radius            # mm (inf = flat)
        self.thickness = thickness      # mm (internal glass thickness to next surface)
        self.glass = glass              # glass name or '' for air
        self.semi_diameter = semi_diameter  # mm
        self.conic = conic
        self.surf_type = surf_type      # 'Standard' or 'Mirror'
        # Biconic / anamorphic: if radius_y is set (not None), the
        # surface has independent x and y curvatures.  When None,
        # the surface is rotationally symmetric (standard).
        self.radius_y = radius_y        # mm or None
        self.conic_y = conic_y          # dimensionless or None


# ════════════════════════════════════════════════════════════════════════
# SourceDefinition
# ════════════════════════════════════════════════════════════════════════

class SourceDefinition:
    """Defines the illumination for the optical system."""

    TYPES = [
        'plane_wave',
        'gaussian',
        'gaussian_aperture',
        'top_hat',
        'fiber_mode',
        'point_source',
        'emitter_array',
    ]

    def __init__(self, source_type='plane_wave', **kwargs):
        self.source_type = source_type
        self.wavelength_nm = kwargs.get('wavelength_nm', 1310.0)
        self.beam_diameter_mm = kwargs.get('beam_diameter_mm', 1.0)
        self.na = kwargs.get('na', 0.1)
        self.sigma_mm = kwargs.get('sigma_mm', 5.0)
        self.object_distance_mm = kwargs.get('object_distance_mm', 1000.0)
        self.emitter_pitch_mm = kwargs.get('emitter_pitch_mm', 0.050)
        self.emitter_nx = kwargs.get('emitter_nx', 12)
        self.emitter_ny = kwargs.get('emitter_ny', 12)
        self.emitter_waist_mm = kwargs.get('emitter_waist_mm', 0.009)
        # 3.6: top-hat + fiber-mode source factories from Source class.
        self.top_hat_diameter_mm = kwargs.get('top_hat_diameter_mm', 2.0)
        self.fiber_mfd_um = kwargs.get('fiber_mfd_um', 10.4)
        self.fiber_NA = kwargs.get('fiber_NA', 0.14)
        # Off-axis field angle for tilted plane-wave / point-source
        self.field_angle_x_deg = kwargs.get('field_angle_x_deg', 0.0)
        self.field_angle_y_deg = kwargs.get('field_angle_y_deg', 0.0)
        # 3.6: source polarization (Jones).  None = unpolarized scalar.
        # Otherwise one of 'linear_x', 'linear_y', 'linear_45',
        # 'rcp', 'lcp'.  Consumed by to_source() when present.
        self.polarization = kwargs.get('polarization', None)

    def describe(self):
        if self.source_type == 'plane_wave':
            return 'Plane wave (fills EPD)'
        elif self.source_type == 'gaussian':
            return f'Gaussian beam d={self.beam_diameter_mm:.3f}mm (1/e2)'
        elif self.source_type == 'gaussian_aperture':
            return f'Gaussian aperture sigma={self.sigma_mm:.2f}mm'
        elif self.source_type == 'point_source':
            return f'Point source at {self.object_distance_mm:.1f}mm'
        elif self.source_type == 'top_hat':
            return f'Top-hat aperture (d={self.top_hat_diameter_mm:.3f}mm)'
        elif self.source_type == 'fiber_mode':
            return (f'Fiber mode (MFD={self.fiber_mfd_um:.1f}µm, '
                    f'NA={self.fiber_NA:.3f})')
        elif self.source_type == 'emitter_array':
            return (f'Emitter array {self.emitter_nx}x{self.emitter_ny}, '
                    f'pitch={self.emitter_pitch_mm:.3f}mm')
        return self.source_type

    def to_source(self, N: int, dx_m: float, *, epd_m: float = 0.0):
        """Build a :class:`lumenairy.Source` instance (3.5.0) on the
        given grid, using the appropriate factory for this source's
        ``source_type``.  Returned ``Source`` carries an ``E_field``
        you can drop into any propagator and a ``.propagate(...)``
        method that chains through the dispatcher.

        Parameters
        ----------
        N : int
            Grid size for the source field.
        dx_m : float
            Grid pitch in metres.
        epd_m : float, optional
            Entrance-pupil diameter in metres.  Used by the
            ``plane_wave`` branch to clip the field to the pupil
            (mirroring the dock's behaviour); other source types
            ignore it.

        Returns
        -------
        :class:`lumenairy.Source`

        Notes
        -----
        Mapping rules (UI source type → core factory):

        * ``plane_wave`` -> :meth:`Source.plane_wave` with the
          field-angle tilts; clipped to the pupil if ``epd_m > 0``.
        * ``gaussian`` -> :meth:`Source.gaussian` with
          ``w0 = beam_diameter_mm * 1e-3 / 2``.
        * ``gaussian_aperture`` -> :meth:`Source.gaussian` with
          ``w0 = sigma_mm * 1e-3`` (sigma is the 1/e amplitude
          radius; Source uses w0 the same way).
        * ``point_source`` -> :meth:`Source.point_source` placed
          at ``z0 = -object_distance_mm * 1e-3``.
        * ``emitter_array`` -> falls back to a tiled superposition
          of ``Source.gaussian`` instances; not a perfect mapping
          but the same physical content as the dock's existing
          plane-wave-clipped path.
        """
        import numpy as _np
        from lumenairy import Source as _Source
        wavelength = self.wavelength_nm * 1e-9

        # v4.15.1 (P1-NEW-C / Agent E): migrated to the canonical
        # kwarg-only ``Source.method(*, N, dx, wavelength, ...)`` form.
        # Pre-v4.15.1 these 7 callsites used the legacy positional form
        # (e.g. ``Source.gaussian(w0, N, dx, wavelength)``) and so
        # emitted ``DeprecationWarning`` at v4.15.0 startup -- the
        # release that introduced the deprecation shim didn't migrate
        # its own internal UI consumers.
        if self.source_type == 'plane_wave':
            ax = _np.deg2rad(self.field_angle_x_deg)
            ay = _np.deg2rad(self.field_angle_y_deg)
            src = _Source.plane_wave(
                N=N, dx=dx_m, wavelength=wavelength,
                angle_x=ax, angle_y=ay,
                name=f'Plane wave ({self.wavelength_nm:.0f} nm)')
            if epd_m and epd_m > 0:
                ax_grid = (_np.arange(N) - N / 2) * dx_m
                X, Y = _np.meshgrid(ax_grid, ax_grid)
                mask = (X * X + Y * Y) <= (epd_m / 2) ** 2
                src.E = src.E * mask
            return src

        if self.source_type == 'gaussian':
            w0 = self.beam_diameter_mm * 1e-3 / 2
            return _Source.gaussian(
                N=N, dx=dx_m, wavelength=wavelength, w0=w0,
                name=f'Gaussian (d={self.beam_diameter_mm:.3f} mm)')

        if self.source_type == 'gaussian_aperture':
            w0 = self.sigma_mm * 1e-3
            return _Source.gaussian(
                N=N, dx=dx_m, wavelength=wavelength, w0=w0,
                name=f'Gaussian-aperture (sigma={self.sigma_mm:.2f} mm)')

        if self.source_type == 'top_hat':
            d = self.top_hat_diameter_mm * 1e-3
            return _Source.top_hat(
                N=N, dx=dx_m, wavelength=wavelength, diameter=d,
                name=f'Top-hat (d={self.top_hat_diameter_mm:.3f} mm)')

        if self.source_type == 'fiber_mode':
            mfd = self.fiber_mfd_um * 1e-6
            return _Source.fiber_mode(
                N=N, dx=dx_m, wavelength=wavelength,
                mode_field_diameter=mfd, na=self.fiber_NA,
                name=f'Fiber mode (MFD={self.fiber_mfd_um:.1f} um, '
                     f'NA={self.fiber_NA:.3f})')

        if self.source_type == 'point_source':
            z0 = -self.object_distance_mm * 1e-3
            return _Source.point_source(
                N=N, dx=dx_m, wavelength=wavelength, z0=z0,
                name=f'Point source ({self.object_distance_mm:.1f} mm)')

        if self.source_type == 'emitter_array':
            # No 1:1 core factory; build by tiled superposition of
            # individual Gaussian emitters at the requested pitch.
            ax_grid = (_np.arange(N) - N / 2) * dx_m
            X, Y = _np.meshgrid(ax_grid, ax_grid)
            E = _np.zeros((N, N), dtype=_np.complex128)
            w0 = self.emitter_waist_mm * 1e-3
            pitch = self.emitter_pitch_mm * 1e-3
            for iy in range(self.emitter_ny):
                for ix in range(self.emitter_nx):
                    cx = (ix - (self.emitter_nx - 1) / 2) * pitch
                    cy = (iy - (self.emitter_ny - 1) / 2) * pitch
                    E += _np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) /
                                  (w0 ** 2))
            src = _Source(
                E=E.astype(_np.complex128),
                wavelength=wavelength, dx=dx_m,
                name=f'Emitter array '
                     f'{self.emitter_nx}×{self.emitter_ny}')
            return src

        # Unknown type — fall through to a unit plane wave so callers
        # always get a valid Source rather than None.
        return _Source.plane_wave(
            N=N, dx=dx_m, wavelength=wavelength,
            name=f'(unknown source_type={self.source_type!r})')


# ════════════════════════════════════════════════════════════════════════
# Element — one optical element (lens, mirror, DOE, source, detector)
# ════════════════════════════════════════════════════════════════════════

class Element:
    """One optical element in the sequential system.

    ``distance_mm`` is the axial distance from the *front vertex* of
    the previous element to the *front vertex* of this element.
    Internal glass thicknesses are stored on the ``SurfaceRow`` objects
    within ``surfaces``.
    """

    TYPES = [
        'Source', 'Singlet', 'Doublet', 'Triplet',
        'Mirror', 'MLA', 'DOE', 'Dammann', 'Detector',
    ]

    # Columns displayed in the element table (relative mode)
    COLUMNS_RELATIVE = [
        'Elem#',       # 0
        'Name',        # 1
        'Type',        # 2
        'Distance',    # 3  (mm from previous element)
        'Tilt X',      # 4  (degrees, relative to beam axis)
        'Tilt Y',      # 5  (degrees)
        'Decenter X',  # 6  (mm, perpendicular to axis)
        'Decenter Y',  # 7  (mm)
    ]
    # Columns in absolute mode
    COLUMNS_ABSOLUTE = [
        'Elem#',       # 0
        'Name',        # 1
        'Type',        # 2
        'Z',           # 3  (mm, absolute position along axis)
        'Rx',          # 4  (degrees, rotation about X)
        'Ry',          # 5  (degrees, rotation about Y)
        'X',           # 6  (mm, absolute lateral position)
        'Y',           # 7  (mm, absolute lateral position)
    ]
    N_COLS = 8

    def __init__(self, elem_num=0, name='', elem_type='Singlet',
                 distance_mm=0.0, tilt_x=0.0, tilt_y=0.0,
                 decenter_x=0.0, decenter_y=0.0,
                 surfaces=None, source=None, aux=None,
                 origin=None, R=None):
        self.elem_num = elem_num
        self.name = name
        self.elem_type = elem_type
        self.distance_mm = distance_mm
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        self.decenter_x = decenter_x
        self.decenter_y = decenter_y
        self.surfaces = surfaces or []
        self.source = source            # SourceDefinition, only for 'Source' type
        self.aux = aux or {}            # MLA/DOE params
        # 3.7.3: per-element world-frame canonical state.  ``origin`` is
        # the front-vertex world position in mm; ``R`` is a 3x3 rotation
        # mapping local axes (x, y, z=optical axis) into world coords.
        # Populated by ``SystemModel.recompute_element_frames()`` after
        # any structural change to the element list, so layout views
        # and ``_build_trace_surfaces_internal`` can read them directly
        # without re-walking distance_mm/tilt_x conventions every call.
        # Constructors may pass (origin, R) explicitly when building a
        # system in absolute coords (e.g., from a Zemax import that
        # has already accumulated cb's into world frames); otherwise
        # they default to identity and recompute_element_frames() fills
        # them in from the relative fields.
        self.origin = (np.zeros(3, dtype=float)
                        if origin is None else np.asarray(origin, dtype=float))
        self.R = (np.eye(3, dtype=float)
                   if R is None else np.asarray(R, dtype=float))

    @property
    def internal_thickness_mm(self):
        """Total glass thickness of this element (sum of internal surface thicknesses)."""
        if not self.surfaces:
            return 0.0
        return sum(s.thickness for s in self.surfaces[:-1])

    def display_value(self, col, display_distance=None):
        """Return string for the element table."""
        if col == 0:
            return str(self.elem_num)
        elif col == 1:
            return self.name
        elif col == 2:
            return self.elem_type
        elif col == 3:
            d = display_distance if display_distance is not None else self.distance_mm
            return f'{d:.4g}'
        elif col == 4:
            return f'{self.tilt_x:.4g}' if self.tilt_x != 0 else '0'
        elif col == 5:
            return f'{self.tilt_y:.4g}' if self.tilt_y != 0 else '0'
        elif col == 6:
            return f'{self.decenter_x:.4g}' if self.decenter_x != 0 else '0'
        elif col == 7:
            return f'{self.decenter_y:.4g}' if self.decenter_y != 0 else '0'
        return ''

    def display_value_absolute(self, col):
        """3.7.3: column display for the prescription editor's
        absolute-coordinates mode.  Reads ``origin`` / ``R`` cached
        on the element by ``SystemModel.recompute_element_frames``.

        Columns: [Elem#, Name, Type, Z, Rx, Ry, X, Y].

        Rx/Ry are derived from R as the rotation angles about the
        world x and y axes that take world (0, 0, 1) to local +z
        (i.e., R[:, 2]).  For a system without tilt_z those are
        the standard Euler "ax-then-ay" decomposition; for systems
        with tilt_z the displayed Rx/Ry are still well-defined
        (they describe the orientation of the local +z axis) but
        don't capture the in-plane spin.
        """
        if col == 0:
            return str(self.elem_num)
        elif col == 1:
            return self.name
        elif col == 2:
            return self.elem_type
        elif col == 3:
            return f'{float(self.origin[2]):.4g}'
        elif col == 4:
            # Rx (deg): rotation about world +x.  Derived from
            # R[:, 2] via arctan2(-R[1,2], R[2,2]) so 0° gives the
            # identity rotation.
            zaxis = self.R[:, 2]
            ax_deg = float(np.degrees(np.arctan2(-zaxis[1], zaxis[2])))
            return f'{ax_deg:.4g}' if abs(ax_deg) > 1e-6 else '0'
        elif col == 5:
            # Ry (deg): rotation about world +y.
            zaxis = self.R[:, 2]
            denom = float(np.sqrt(zaxis[1] ** 2 + zaxis[2] ** 2))
            ay_deg = float(np.degrees(np.arctan2(zaxis[0], denom)))
            return f'{ay_deg:.4g}' if abs(ay_deg) > 1e-6 else '0'
        elif col == 6:
            return f'{float(self.origin[0]):.4g}'
        elif col == 7:
            return f'{float(self.origin[1]):.4g}'
        return ''

    def summary(self):
        """One-line summary for tooltips."""
        parts = [self.elem_type]
        if self.surfaces:
            for s in self.surfaces:
                R_str = f'R={s.radius:.2f}' if np.isfinite(s.radius) else 'flat'
                parts.append(R_str)
                if s.glass:
                    parts.append(s.glass)
        if self.source:
            parts.append(self.source.describe())
        return ', '.join(parts)


# ════════════════════════════════════════════════════════════════════════
# SystemModel
# ════════════════════════════════════════════════════════════════════════

class SystemModel(QObject):
    """Central optical system model using elements.

    Signals
    -------
    system_changed
        Emitted when any element, wavelength, or configuration changes.
    trace_ready
        Emitted with a TraceResult after a ray trace completes.
    element_selected
        Emitted with an element index when clicked in the layout.
    optimization_progress / optimization_finished
        Emitted during optimization.
    """

    system_changed = Signal()       # prescription changed — triggers retrace
    display_changed = Signal()      # display-only change (coord mode) — no retrace
    trace_ready = Signal(object)
    trace_started = Signal()      # emitted at the top of run_trace
    element_selected = Signal(int)
    optimization_progress = Signal(int, float)
    optimization_finished = Signal(bool, str)
    # Undo/redo state: (can_undo, can_redo) — fires on every stack change
    # so toolbar/menu actions can toggle their enabled state.
    history_changed = Signal(bool, bool)
    # Snapshot (design comparison) list changed
    snapshots_changed = Signal()

    # Max depth of the undo stack.  Each snapshot is a deep copy of the
    # element list + source + wavelengths/fields/EPD, so 50 deep is a
    # few hundred KB at worst.
    _UNDO_DEPTH = 80

    def __init__(self, parent=None):
        super().__init__(parent)

        # Per-lens-function kwarg overrides edited from the &Options
        # menu's Lens Options dialog.  Layout: dict keyed by function
        # name ('apply_real_lens', 'apply_real_lens_traced',
        # 'apply_real_lens_maslov'), each value is a dict of kwarg ->
        # user-chosen value.  Only kwargs the user actually changed
        # are stored; library defaults apply for everything else.
        # Read by WaveOpticsDock._run when delegating to the real-lens
        # functions.
        self.lens_options = {
            'apply_real_lens': {},
            'apply_real_lens_traced': {},
            'apply_real_lens_maslov': {},
        }

        # Wavelengths
        self.wavelength_nm = 1310.0
        self.wavelengths_nm = [1310.0]
        # Per-wavelength weights for multi-wavelength merits.  ``None``
        # means "uniform" which the merit terms interpret as 1/N each.
        self.wavelength_weights = None

        # Entrance pupil diameter
        self.epd_mm = 25.4

        # Field angles
        self.field_angles_deg = [0.0]
        self.field_weights = None    # per-field weights, see above

        # Auto-retrace policy: 'on' (default) retraces on every change,
        # 'geometric-only' skips the wave leg, 'manual' only traces when
        # the user hits Ctrl+T.
        self.auto_retrace_mode = 'on'

        # Unit preference: 'engineering' (mm / um / nm) or 'si' (m)
        self.unit_preference = 'engineering'

        # Coordinate display mode
        self._coordinate_mode = 'relative'  # or 'absolute'

        # Element list — always starts with Source and ends with Detector
        self.elements = [
            Element(0, 'Source', 'Source', distance_mm=0.0,
                    source=SourceDefinition()),
            Element(1, 'Detector', 'Detector', distance_mm=100.0),
        ]

        # Optimization variables: list of (elem_idx, surf_idx, field_name)
        self.opt_variables = []

        # Display preferences
        self.prefs = {
            'bg_2d': '#050709',        # 2D layout background
            'bg_3d': '#050709',        # 3D layout background
            'ray_color': '#5cb8ff',    # ray display color (overrides wavelength)
            'ray_use_wavelength': True, # use wavelength-based colors instead
            'accent': '#5cb8ff',       # UI accent color
            'theme': 'dark',           # 'dark' or 'light'
        }

        # Cache
        self._trace_result = None
        self._abcd = None
        self._efl = None
        self._bfl = None
        self._flat_surfaces_cache = None
        self._flat_surfaces_world_cache = None

        # Undo / redo stacks.  Each entry is a deep copy of everything
        # needed to reproduce the system (see _capture_state /
        # _restore_state).  We push *before* every mutation via
        # _checkpoint(), so the current state is what you're about to
        # leave behind.
        self._undo_stack = []
        self._redo_stack = []
        self._in_restore = False  # suppress checkpoints while restoring

        # "Snapshots" are user-named states for A/B comparison -- stored
        # alongside the live model, not on the undo stack.
        self.snapshots = []        # list of {'name': str, 'state': dict, 'efl_mm': float}

        # Suppress-history flag: inside bulk operations (group/ungroup,
        # load_prescription) we still want one checkpoint, not N.
        self._suppress_history = False

    # ── Properties ──────────────────────────────────────────────────

    @property
    def wavelength_m(self):
        return self.wavelength_nm * 1e-9

    @property
    def epd_m(self):
        return self.epd_mm * 1e-3

    @property
    def num_elements(self):
        return len(self.elements)

    @property
    def source(self):
        """The active source definition (from element 0)."""
        return self.elements[0].source if self.elements else None

    @property
    def trace_result(self):
        return self._trace_result

    @property
    def coordinate_mode(self):
        return self._coordinate_mode

    @property
    def efl_mm(self):
        self._ensure_abcd()
        return self._efl * 1e3 if self._efl is not None else np.inf

    @property
    def bfl_mm(self):
        self._ensure_abcd()
        return self._bfl * 1e3 if self._bfl is not None else np.inf

    # ── Coordinate mode ────────────────────────────────────────────

    def set_coordinate_mode(self, mode):
        """Toggle 'relative' / 'absolute'. Display only — no retrace."""
        if mode in ('relative', 'absolute') and mode != self._coordinate_mode:
            self._coordinate_mode = mode
            self.display_changed.emit()

    def element_z_positions_mm(self):
        """Compute absolute z-position of each element's front vertex."""
        z = 0.0
        positions = []
        for elem in self.elements:
            if elem.elem_type == 'Source':
                positions.append(0.0)
            else:
                z += elem.distance_mm
                positions.append(z)
        return positions

    def element_frames_2d_mm(self):
        """3.7.3: per-element 2D world frame ``(z, y, theta_rad)``
        for the 2D side-view layout.  Reads each element's cached
        ``origin`` (3-vector, mm) and ``R`` (3x3 rotation) -- both
        kept in sync by :meth:`recompute_element_frames` after every
        structural change -- and projects them onto the y-z plane.

        ``theta`` is the angle of the local +z axis in the y-z
        plane, measured CCW from world +z.  For unfolded systems
        every element has ``theta = 0``; for folded systems theta
        accumulates the tilts and mirror flips encoded in R.
        """
        frames = []
        for elem in self.elements:
            o = elem.origin
            zaxis = elem.R[:, 2]
            theta = float(np.arctan2(zaxis[1], zaxis[2]))
            frames.append((float(o[2]), float(o[1]), theta))
        return frames

    def element_frames_3d_mm(self):
        """3.7.3: per-element 3D world frame ``(origin_xyz, R)``
        for the 3D layout.  Reads each element's cached
        ``origin`` / ``R`` (kept in sync by
        :meth:`recompute_element_frames`).  ``origin`` is a length-3
        numpy array in scene mm; ``R`` is a 3x3 rotation matrix
        mapping local axes (z = optical axis, y = vertical, x =
        horizontal) into world coordinates.
        """
        return [(elem.origin.copy(), elem.R.copy())
                for elem in self.elements]

    def surface_frames_2d_mm(self):
        """3.7.5: per-traced-surface 2D world frame
        ``(z, y, theta_rad)`` in trace order, plus the image-plane
        frame at the end.  One entry per ACTUAL optical surface --
        no separate coord-break frames -- so it matches the world-
        coord trace surface list emitted by
        :meth:`_build_trace_surfaces_world` one-to-one.

        Walks each element's cached front-vertex frame and steps
        along internal thicknesses to land each internal surface.
        """
        surf_frames = []
        for elem in self.elements:
            if elem.elem_type == 'Source':
                continue
            o = elem.origin
            R = elem.R
            theta = float(np.arctan2(R[1, 2], R[2, 2]))
            if elem.elem_type == 'Detector':
                surf_frames.append((float(o[2]), float(o[1]), theta))
                continue
            cum_t = 0.0
            for srow in elem.surfaces:
                z_s = float(o[2]) + cum_t * np.cos(theta)
                y_s = float(o[1]) + cum_t * np.sin(theta)
                surf_frames.append((z_s, y_s, theta))
                cum_t += float(srow.thickness)
        return surf_frames

    def surface_frames_3d_mm(self):
        """3.7.5: per-traced-surface 3D world frame
        ``(origin_xyz, R_3x3)`` in trace order, plus the image-plane
        frame at the end.  One entry per ACTUAL optical surface --
        matches :meth:`_build_trace_surfaces_world` one-to-one.
        """
        surf_frames = []
        for elem in self.elements:
            if elem.elem_type == 'Source':
                continue
            o = elem.origin
            R = elem.R
            if elem.elem_type == 'Detector':
                surf_frames.append((o.copy(), R.copy()))
                continue
            cum_t = 0.0
            for srow in elem.surfaces:
                surf_frames.append((o + cum_t * R[:, 2], R.copy()))
                cum_t += float(srow.thickness)
        return surf_frames

    def set_element_absolute_field(self, elem_idx, col, value):
        """3.7.3: write an edit from the prescription editor's
        absolute-coordinates mode.  ``col`` is the column index
        (3 = Z, 4 = Rx, 5 = Ry, 6 = X, 7 = Y); ``value`` is the
        new floating-point value the user typed.

        Computes the corresponding relative-field update that
        will make :meth:`recompute_element_frames` reproduce the
        user's intended absolute position / orientation, then
        triggers ``_invalidate`` so the caches and views refresh.

        Position edits (Z / X / Y) update ``distance_mm`` /
        ``decenter_x`` / ``decenter_y`` so the element lands at
        the desired world coordinates relative to the previous
        element's back vertex.

        Orientation edits (Rx / Ry) update ``tilt_x`` /
        ``tilt_y`` so the element's local +z axis points in the
        desired world direction.

        Source rows are read-only in absolute mode (the source
        is anchored at the world origin); detector rows allow
        position edits but not orientation edits.
        """
        if elem_idx <= 0 or elem_idx >= len(self.elements):
            return
        e = self.elements[elem_idx]
        if e.elem_type == 'Source':
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        self._checkpoint()
        # Orientation edits: drive tilt_x / tilt_y directly.
        if col == 4:
            e.tilt_x = v
            self._invalidate()
            self.system_changed.emit()
            return
        if col == 5:
            e.tilt_y = v
            self._invalidate()
            self.system_changed.emit()
            return
        # Position edits (col 3 = Z, 6 = X, 7 = Y).  We start
        # from the element's CURRENT cached origin, replace just
        # the edited component, then back-derive distance_mm /
        # decenter_x / decenter_y from the desired origin minus
        # the previous element's back vertex projected onto the
        # previous element's local axes.
        cur_origin = e.origin.copy()
        if col == 3:
            cur_origin[2] = v
        elif col == 6:
            cur_origin[0] = v
        elif col == 7:
            cur_origin[1] = v
        else:
            return
        prev = self.elements[elem_idx - 1]
        # v4.15 (P1-UI-4): route both call sites through one helper so
        # they can't drift on this calculation again.  Pre-4.15 this
        # used ``sum(all surfaces)`` while :meth:`set_display_distance`
        # used ``Element.internal_thickness_mm`` (``sum(surfaces[:-1])``).
        # Same value for Thorlabs / freshly-inserted singlets (where the
        # back surface has thickness=0), but the two diverge for
        # elements whose back-surface thickness was explicitly written
        # to non-zero (e.g. by ``_build_trace_surfaces_internal``
        # transient state or a ungroup-then-edit workflow), giving the
        # absolute-position editor and the absolute-distance setter
        # different back-vertex Z's for the same element.
        prev_back = self._prev_element_back_vertex_world(prev)
        delta = cur_origin - prev_back
        new_distance = float(np.dot(delta, prev.R[:, 2]))
        # Subtract the axial component to get the perpendicular
        # offset, then project that onto local +x / +y.
        perp = delta - new_distance * prev.R[:, 2]
        new_dx = float(np.dot(perp, prev.R[:, 0]))
        new_dy = float(np.dot(perp, prev.R[:, 1]))
        e.distance_mm = max(0.0, new_distance)
        e.decenter_x = new_dx
        e.decenter_y = new_dy
        self._invalidate()
        self.system_changed.emit()

    def get_display_distance(self, elem_index):
        """Value to show in the Distance column."""
        if self._coordinate_mode == 'relative':
            return self.elements[elem_index].distance_mm
        else:
            return self.element_z_positions_mm()[elem_index]

    @staticmethod
    def _prev_element_back_vertex_world(prev):
        """v4.15 (P1-UI-4): world-coords back vertex of ``prev`` element.

        Single source of truth for "where is element N-1's back-vertex
        in world space?", consumed by:

        * :meth:`_apply_absolute_position_edit` -- to back-derive the
          new ``distance_mm`` from a user-entered absolute Z;
        * :meth:`set_display_distance` -- to convert the absolute-mode
          column value into the element's relative gap.

        Pre-4.15 those two places computed this independently with a
        subtle difference: the first used ``sum(all surface
        thicknesses) * R[:,2]``, the second used the
        :attr:`Element.internal_thickness_mm` property which is
        ``sum(surfaces[:-1])`` (i.e. excludes any non-zero thickness
        on the back surface).  The slicing form is correct for the
        "back vertex of THIS element" question -- the back-surface's
        thickness, when non-zero, encodes the air gap *to the next
        element*, not internal glass path -- so we standardise on it
        everywhere now.
        """
        if not prev.surfaces:
            return prev.origin.copy()
        return prev.origin + prev.internal_thickness_mm * prev.R[:, 2]

    def set_display_distance(self, elem_index, value):
        """Set distance from display value, handling coordinate mode."""
        if elem_index == 0:
            return  # Source is always at z=0
        self._checkpoint()
        if self._coordinate_mode == 'relative':
            self.elements[elem_index].distance_mm = max(0, value)
        else:
            # Absolute mode: convert to relative.  v4.15 (P1-UI-4):
            # the previous-element back vertex is now expressed in the
            # SAME world-frame coords the absolute display column uses
            # (front-vertex Z accumulated via element_z_positions_mm).
            # ``_prev_element_back_vertex_world`` returns a 3-vector;
            # we project its Z component since the column is 1-D.
            positions = self.element_z_positions_mm()
            prev_z = positions[elem_index - 1]
            prev_elem = self.elements[elem_index - 1]
            prev_back = prev_z + prev_elem.internal_thickness_mm
            self.elements[elem_index].distance_mm = max(0, value - prev_back)
        self._invalidate()
        self.system_changed.emit()

    # ── Mutators ───────────────────────────────────────────────────

    def set_wavelength(self, wv_nm):
        if wv_nm != self.wavelength_nm and wv_nm > 0:
            self._checkpoint()
            self.wavelength_nm = wv_nm
            self.wavelengths_nm[0] = wv_nm
            self._invalidate()
            self.system_changed.emit()

    def set_epd(self, epd_mm):
        if epd_mm != self.epd_mm and epd_mm > 0:
            self._checkpoint()
            self.epd_mm = epd_mm
            self._invalidate()
            self.system_changed.emit()

    def set_source(self, source_def):
        """Set the source definition on element 0."""
        if self.elements and self.elements[0].elem_type == 'Source':
            self._checkpoint()
            self.elements[0].source = source_def
            self._invalidate()
            self.system_changed.emit()

    def insert_element(self, index, element):
        """Insert an element before the given index."""
        if index < 1:
            index = 1  # can't insert before Source
        if index >= len(self.elements):
            index = len(self.elements) - 1  # insert before Detector
        self._checkpoint()
        self.elements.insert(index, element)
        self._renumber()
        self._invalidate()
        self.system_changed.emit()

    def delete_element(self, index):
        """Delete an element (cannot delete Source or Detector)."""
        if index <= 0 or index >= len(self.elements) - 1:
            return
        self._checkpoint()
        # Transfer this element's distance to the next element
        removed = self.elements[index]
        if index < len(self.elements) - 1:
            self.elements[index + 1].distance_mm += removed.distance_mm
        self.elements.pop(index)
        self._renumber()
        self._invalidate()
        self.system_changed.emit()

    def move_element(self, index, direction):
        """Move element up (direction=-1) or down (+1). Single atomic operation."""
        if direction == -1 and index > 1:
            self._checkpoint()
            # Swap with previous
            elem = self.elements[index]
            prev = self.elements[index - 1]
            # Swap positions in list
            self.elements[index] = prev
            self.elements[index - 1] = elem
            # Swap distances so spatial positions are preserved
            elem.distance_mm, prev.distance_mm = prev.distance_mm, elem.distance_mm
            self._renumber()
            self._invalidate()
            self.system_changed.emit()
        elif direction == 1 and index > 0 and index < len(self.elements) - 2:
            self._checkpoint()
            elem = self.elements[index]
            nxt = self.elements[index + 1]
            self.elements[index] = nxt
            self.elements[index + 1] = elem
            elem.distance_mm, nxt.distance_mm = nxt.distance_mm, elem.distance_mm
            self._renumber()
            self._invalidate()
            self.system_changed.emit()

    def group_elements(self, indices, group_name):
        """Merge multiple consecutive elements into a single compound element.

        All surfaces from the selected elements are combined into one
        element.  The distances between the merged elements become
        internal air-gap thicknesses on the last surface of each
        sub-element.

        Parameters
        ----------
        indices : list of int
            Element indices to merge (must be consecutive, no Source/Detector).
        group_name : str
            Name for the resulting compound element.
        """
        indices = sorted(indices)
        # Validate
        if len(indices) < 2:
            return
        self._checkpoint()
        for idx in indices:
            if idx <= 0 or idx >= len(self.elements) - 1:
                return  # can't group Source or Detector
        # Check consecutive
        for i in range(len(indices) - 1):
            if indices[i + 1] != indices[i] + 1:
                return

        # Collect all surfaces, merging cemented interfaces.
        # When two adjacent elements have zero distance between them
        # and the back surface of one matches the front surface of the
        # next (same radius), they share a cemented interface — the
        # duplicate surface is removed and the glass is carried through.
        combined_surfaces = []
        first_elem = self.elements[indices[0]]
        distance_mm = first_elem.distance_mm
        tilt_x = first_elem.tilt_x
        tilt_y = first_elem.tilt_y
        decenter_x = first_elem.decenter_x
        decenter_y = first_elem.decenter_y

        for i, idx in enumerate(indices):
            elem = self.elements[idx]

            for si, srow in enumerate(elem.surfaces):
                # Check if this surface is a duplicate of the previous
                # element's last surface (cemented interface)
                if si == 0 and combined_surfaces and i > 0:
                    prev_surf = combined_surfaces[-1]
                    next_elem = elem
                    gap = next_elem.distance_mm

                    # Cemented: same radius, zero gap, previous exits to air
                    same_radius = (abs(prev_surf.radius - srow.radius) < 1e-6
                                   if np.isfinite(prev_surf.radius) and np.isfinite(srow.radius)
                                   else np.isinf(prev_surf.radius) and np.isinf(srow.radius))

                    if same_radius and gap == 0 and not prev_surf.glass:
                        # Merge: the previous surface was the exit of the
                        # prior element (glass=''); replace it with the
                        # cemented version that carries the new glass through.
                        prev_surf.glass = srow.glass
                        prev_surf.thickness = srow.thickness
                        continue  # skip adding the duplicate

                combined_surfaces.append(srow)

            # If this isn't the last element in the group, add the
            # inter-element air gap as thickness on the last surface
            if i < len(indices) - 1:
                next_elem = self.elements[indices[i + 1]]
                if combined_surfaces and next_elem.distance_mm > 0:
                    combined_surfaces[-1].thickness += next_elem.distance_mm

        # Determine type
        n_surf = len(combined_surfaces)
        if n_surf == 2:
            etype = 'Singlet'
        elif n_surf == 3:
            etype = 'Doublet'
        elif n_surf == 4:
            etype = 'Triplet'
        else:
            etype = 'Singlet'  # generic compound

        # Build new element
        grouped = Element(
            0, group_name, etype, distance_mm=distance_mm,
            tilt_x=tilt_x, tilt_y=tilt_y,
            decenter_x=decenter_x, decenter_y=decenter_y,
            surfaces=combined_surfaces,
        )

        # Remove old elements and insert the grouped one
        for idx in reversed(indices):
            self.elements.pop(idx)
        self.elements.insert(indices[0], grouped)

        self._renumber()
        self._invalidate()
        self.system_changed.emit()

    def ungroup_element(self, index):
        """Split a compound element into individual lens elements.

        Each lens element is defined by a contiguous glass region:
        a front surface entering the glass and a back surface exiting it.

        A cemented doublet (3 surfaces: S0/N-BAF10, S1/N-SF6HT, S2/air)
        splits into two singlets:
          - Element A: [S0, S1_copy] with glass N-BAF10
          - Element B: [S1_copy, S2] with glass N-SF6HT

        The shared cemented interface (S1) is duplicated — each element
        gets its own copy.  For non-cemented cases (air gap between
        elements), the air gap thickness becomes the distance between
        the sub-elements.
        """
        if index <= 0 or index >= len(self.elements) - 1:
            return
        elem = self.elements[index]
        if len(elem.surfaces) < 2:
            return
        self._checkpoint()

        # Identify glass regions: each surface with a non-empty glass
        # field starts a glass region that ends at the next surface
        # with a different glass (or no glass).
        #
        # Walk the surfaces and group them into (front, back) pairs
        # per glass region.
        new_elements = []
        si = 0
        surfaces = elem.surfaces

        while si < len(surfaces):
            front = surfaces[si]

            if not front.glass:
                # This surface doesn't enter glass — it's either a
                # standalone surface (mirror) or the exit of a previous
                # region.  If it's a mirror or the last surface, make it
                # its own element.
                if front.surf_type == 'Mirror' or si == len(surfaces) - 1:
                    dist = elem.distance_mm if not new_elements else 0.0
                    new_elements.append(Element(
                        0, f'{elem.name} - S{si+1}',
                        'Mirror' if front.surf_type == 'Mirror' else 'Singlet',
                        distance_mm=dist,
                        tilt_x=elem.tilt_x, tilt_y=elem.tilt_y,
                        decenter_x=elem.decenter_x, decenter_y=elem.decenter_y,
                        surfaces=[front],
                    ))
                si += 1
                continue

            # front.glass is set — this surface enters a glass.
            # Find the back surface: the next surface where the glass
            # changes (to a different glass or to air).
            glass_name = front.glass
            back_si = si + 1

            # The back surface is the next one (it exits this glass)
            if back_si >= len(surfaces):
                # Only one surface left — standalone
                dist = elem.distance_mm if not new_elements else front.thickness
                new_elements.append(Element(
                    0, f'{elem.name} - {glass_name}', 'Singlet',
                    distance_mm=dist,
                    surfaces=[front],
                ))
                si = back_si
                continue

            back = surfaces[back_si]

            # Build a singlet: front surface + back surface
            # For a cemented interface, the back surface is shared with
            # the next element. We need to make a copy for this element.
            # The copy has the same radius/conic but glass='' (exits to air)
            # unless it's cemented (back.glass is set), in which case the
            # original back surface stays as-is in the next iteration.

            front_copy = SurfaceRow(
                front.radius, front.thickness, front.glass,
                front.semi_diameter, front.conic, front.surf_type)

            if back.glass:
                # Cemented: back surface is a glass-to-glass interface.
                # This element's back surface exits the current glass.
                back_copy = SurfaceRow(
                    back.radius, 0.0, '',  # exits to "air" at the cement
                    back.semi_diameter, back.conic, back.surf_type)
            else:
                # Not cemented: back surface exits to air normally.
                back_copy = SurfaceRow(
                    back.radius, back.thickness, back.glass,
                    back.semi_diameter, back.conic, back.surf_type)

            # Distance
            if not new_elements:
                dist = elem.distance_mm
            else:
                # For cemented interfaces, distance = 0 (surfaces touch)
                # For air gaps, distance = previous back surface thickness
                dist = 0.0

            sub_name = f'{elem.name} - {glass_name}'
            new_elements.append(Element(
                0, sub_name, 'Singlet',
                distance_mm=dist,
                tilt_x=elem.tilt_x, tilt_y=elem.tilt_y,
                decenter_x=elem.decenter_x, decenter_y=elem.decenter_y,
                surfaces=[front_copy, back_copy],
            ))

            # Advance: if the back surface also enters a new glass
            # (cemented), the next iteration starts from back_si.
            # If the back surface exits to air, advance past it.
            if back.glass:
                si = back_si  # re-process back surface as front of next
            else:
                si = back_si + 1

        # Handle air gaps between sub-elements from internal thicknesses
        # that were air gaps (back surface with thickness > 0 and no glass)
        for i in range(1, len(new_elements)):
            prev_elem = new_elements[i - 1]
            if prev_elem.surfaces:
                last_s = prev_elem.surfaces[-1]
                if not last_s.glass and last_s.thickness > 0:
                    new_elements[i].distance_mm += last_s.thickness
                    last_s.thickness = 0.0

        # Replace the original element
        self.elements.pop(index)
        for i, sub in enumerate(new_elements):
            self.elements.insert(index + i, sub)

        self._renumber()
        self._invalidate()
        self.system_changed.emit()

    def set_element_field(self, elem_idx, col, value_str):
        """Set an element field from a string. Returns True if changed."""
        if not (0 <= elem_idx < len(self.elements)):
            return False
        elem = self.elements[elem_idx]
        text = value_str.strip()
        try:
            if col == 1:  # Name
                if elem.name == text:
                    return False
                self._checkpoint()
                elem.name = text
            elif col == 2:  # Type
                if text in Element.TYPES and text != elem.elem_type:
                    self._checkpoint()
                    elem.elem_type = text
                else:
                    return False
            elif col == 3:  # Distance
                self.set_display_distance(elem_idx, float(text))
                return True
            elif col == 4:  # Tilt X
                val = float(text) if text else 0.0
                if val == elem.tilt_x:
                    return False
                self._checkpoint()
                elem.tilt_x = val
            elif col == 5:  # Tilt Y
                val = float(text) if text else 0.0
                if val == elem.tilt_y:
                    return False
                self._checkpoint()
                elem.tilt_y = val
            elif col == 6:  # Decenter X
                val = float(text) if text else 0.0
                if val == elem.decenter_x:
                    return False
                self._checkpoint()
                elem.decenter_x = val
            elif col == 7:  # Decenter Y
                val = float(text) if text else 0.0
                if val == elem.decenter_y:
                    return False
                self._checkpoint()
                elem.decenter_y = val
            else:
                return False
            self._invalidate()
            self.system_changed.emit()
            return True
        except ValueError:
            return False

    def set_surface_field(self, elem_idx, surf_idx, field, value):
        """Set an internal surface field. ``field`` is 'radius', 'thickness', etc."""
        elem = self.elements[elem_idx]
        if surf_idx >= len(elem.surfaces):
            return False
        s = elem.surfaces[surf_idx]
        old = getattr(s, field, None)
        # Checkpoint before mutation; if no change happens we'll have an
        # identical snapshot on the stack, which is fine (undo becomes a
        # no-op).  The upside is that checkpoints happen atomically with
        # the attempt -- no risk of half-mutated state being uncapturable.
        self._checkpoint()
        if field == 'radius':
            s.radius = np.inf if str(value).lower() in ('inf', 'infinity', '') else float(value)
        elif field == 'thickness':
            s.thickness = float(value)
        elif field == 'glass':
            s.glass = str(value)
        elif field == 'semi_diameter':
            s.semi_diameter = np.inf if str(value).lower() in ('inf', 'infinity', '') else float(value)
        elif field == 'conic':
            s.conic = float(value)
        elif field == 'radius_y':
            if str(value).strip() == '':
                s.radius_y = None
            elif str(value).lower() in ('inf', 'infinity'):
                s.radius_y = np.inf
            else:
                s.radius_y = float(value)
        elif field == 'conic_y':
            if str(value).strip() == '':
                s.conic_y = None
            else:
                s.conic_y = float(value)
        else:
            return False
        if getattr(s, field) != old:
            self._invalidate()
            self.system_changed.emit()
            return True
        return False

    def _renumber(self):
        for i, e in enumerate(self.elements):
            e.elem_num = i

    def _invalidate(self):
        self._trace_result = None
        self._abcd = None
        self._efl = None
        self._bfl = None
        self._flat_surfaces_cache = None
        self._flat_surfaces_world_cache = None
        # 3.7.3: keep each Element's cached world-frame ``origin`` /
        # ``R`` in sync with the relative fields after any structural
        # change.  Layout views and the trace-surface builder read
        # these directly instead of re-walking distance_mm / tilt_x
        # conventions every call -- that path was the origin of the
        # frame-convention bugs (Rx sign, Rmirror reflection vs
        # rotation, mirror_sign tracking) we burned commits hunting.
        self.recompute_element_frames()

    def recompute_element_frames(self):
        """Walk the element list and assign each element's
        ``origin`` (world-frame front vertex, mm) and ``R`` (3x3
        rotation, local axes in world coords) from the relative
        fields (distance_mm, tilt_x, tilt_y, decenter_x, decenter_y).

        Single place for the relative -> absolute conversion.

        3.7.4 convention:

        * Distances are Zemax-signed (negative post-mirror so the
          unfolded cumulative z keeps moving forward).
        * No ``Rmirror`` is applied at mirror elements -- the
          trace's N-flip handles direction reversal locally, and
          the FRAME stays unchanged at the mirror.  Combined with
          Zemax-signed distances this gives correct world positions
          for normal mirrors AND for fold-mirrors with absorbed
          cb_pre tilts.
        * For elements immediately AFTER a Mirror that carry a
          tilt (the cb_post case absorbed by load_prescription),
          the tilt is applied BEFORE the advance.  This is the
          ASYMMETRY that the cb_pre / cb_post split requires:
          cb_pre sits BEFORE its companion surface so its tilt
          applies AT the destination (advance-then-tilt is right
          for that), but cb_post sits AT the mirror's exit so its
          tilt applies BEFORE the gap to the next surface
          (tilt-then-advance is right for that).
        * Optical-convention ``Rx`` / ``Ry`` (Zemax PARM 3 / 4):
          +tilt_x rotates local +z toward +y.
        """
        origin = np.zeros(3, dtype=float)
        R = np.eye(3, dtype=float)
        prev_was_mirror = False
        for elem in self.elements:
            if elem.elem_type == 'Source':
                elem.origin = origin.copy()
                elem.R = R.copy()
                prev_was_mirror = False
                continue
            d = float(getattr(elem, 'distance_mm', 0.0))
            dx = float(getattr(elem, 'decenter_x', 0.0))
            dy = float(getattr(elem, 'decenter_y', 0.0))
            tx_rad = np.radians(float(getattr(elem, 'tilt_x', 0.0)))
            ty_rad = np.radians(float(getattr(elem, 'tilt_y', 0.0)))
            has_tilt = (tx_rad != 0.0 or ty_rad != 0.0
                        or dx != 0.0 or dy != 0.0)

            cb_post_case = prev_was_mirror and has_tilt

            if cb_post_case:
                # cb_post: apply tilt at the previous element's
                # exit (mirror back vertex), THEN advance.
                if tx_rad:
                    c, s = np.cos(tx_rad), np.sin(tx_rad)
                    R = R @ np.array([[1, 0, 0],
                                      [0, c,  s],
                                      [0, -s, c]])
                if ty_rad:
                    c, s = np.cos(ty_rad), np.sin(ty_rad)
                    R = R @ np.array([[c, 0, -s],
                                      [0, 1,  0],
                                      [s, 0,  c]])
                origin = origin + d * R[:, 2]
                if dx:
                    origin = origin + dx * R[:, 0]
                if dy:
                    origin = origin + dy * R[:, 1]
            else:
                # cb_pre / normal case: advance first, then apply
                # decenter perp to the pre-tilt axis, then tilt.
                origin = origin + d * R[:, 2]
                if dx:
                    origin = origin + dx * R[:, 0]
                if dy:
                    origin = origin + dy * R[:, 1]
                if tx_rad:
                    c, s = np.cos(tx_rad), np.sin(tx_rad)
                    R = R @ np.array([[1, 0, 0],
                                      [0, c,  s],
                                      [0, -s, c]])
                if ty_rad:
                    c, s = np.cos(ty_rad), np.sin(ty_rad)
                    R = R @ np.array([[c, 0, -s],
                                      [0, 1,  0],
                                      [s, 0,  c]])

            # Record the front-vertex frame on the element.
            elem.origin = origin.copy()
            elem.R = R.copy()
            # Advance through internal thickness.  Internal
            # thicknesses are Zemax-signed (negative post-mirror);
            # combined with unchanged R[:,2] this places the back
            # vertex in the correct world direction.
            if elem.surfaces:
                internal = sum(float(s.thickness) for s in elem.surfaces)
                origin = origin + internal * R[:, 2]
            # No Rmirror.
            prev_was_mirror = (elem.elem_type == 'Mirror')

    # ── Undo / redo ────────────────────────────────────────────────

    def _capture_state(self):
        """Return a deep-copied snapshot of everything undo/redo cares about.

        v4.15 (P1-UI-3): added ``wavelength_weights``, ``field_weights``,
        and ``lens_options`` to the snapshot.  Pre-4.15 these three
        fields were excluded from the capture, so editing them and then
        hitting Ctrl+Z left the model in an orphan state: the elements
        list rolled back but the per-wavelength weights / per-field
        weights / per-lens-function kwargs stayed at their post-edit
        values.  All three are now snapshotted via list/dict copies
        (no need for deepcopy because each is a flat container of
        immutables -- floats and strings).
        """
        return {
            'elements': copy.deepcopy(self.elements),
            'wavelength_nm': self.wavelength_nm,
            'wavelengths_nm': list(self.wavelengths_nm),
            'epd_mm': self.epd_mm,
            'field_angles_deg': list(self.field_angles_deg),
            'coordinate_mode': self._coordinate_mode,
            'opt_variables': list(self.opt_variables),
            # v4.15 (P1-UI-3) additions:
            'wavelength_weights': (list(self.wavelength_weights)
                                    if self.wavelength_weights is not None
                                    else None),
            'field_weights': (list(self.field_weights)
                               if self.field_weights is not None
                               else None),
            'lens_options': copy.deepcopy(self.lens_options),
        }

    def _restore_state(self, state):
        """Apply a snapshot captured by :meth:`_capture_state`."""
        self._in_restore = True
        try:
            self.elements = copy.deepcopy(state['elements'])
            self.wavelength_nm = state['wavelength_nm']
            self.wavelengths_nm = list(state['wavelengths_nm'])
            self.epd_mm = state['epd_mm']
            self.field_angles_deg = list(state['field_angles_deg'])
            self._coordinate_mode = state['coordinate_mode']
            self.opt_variables = list(state['opt_variables'])
            # v4.15 (P1-UI-3): restore the three fields that v4.14.x
            # silently dropped on undo/redo.  ``.get(...)`` keeps the
            # restore tolerant of older snapshots that lack the keys
            # (e.g. snapshots captured from a Snapshots-dock save that
            # was serialised before v4.15) -- we leave the live values
            # alone in that legacy case rather than NULL-blanking them.
            if 'wavelength_weights' in state:
                self.wavelength_weights = (
                    list(state['wavelength_weights'])
                    if state['wavelength_weights'] is not None else None)
            if 'field_weights' in state:
                self.field_weights = (
                    list(state['field_weights'])
                    if state['field_weights'] is not None else None)
            if 'lens_options' in state:
                self.lens_options = copy.deepcopy(state['lens_options'])
            self._invalidate()
            self.system_changed.emit()
        finally:
            self._in_restore = False
        self._emit_history_changed()

    def _checkpoint(self):
        """Push the current state onto the undo stack.

        Called by every non-trivial mutator **before** it mutates.
        Redo stack is cleared because doing something new forks history.
        """
        if self._in_restore or self._suppress_history:
            return
        self._undo_stack.append(self._capture_state())
        if len(self._undo_stack) > self._UNDO_DEPTH:
            self._undo_stack = self._undo_stack[-self._UNDO_DEPTH:]
        self._redo_stack.clear()
        self._emit_history_changed()

    def _emit_history_changed(self):
        self.history_changed.emit(bool(self._undo_stack),
                                  bool(self._redo_stack))

    def can_undo(self):
        return bool(self._undo_stack)

    def can_redo(self):
        return bool(self._redo_stack)

    def undo(self):
        """Revert the most recent mutation."""
        if not self._undo_stack:
            return
        # Before restoring, push the CURRENT state to redo so we can
        # come back.
        self._redo_stack.append(self._capture_state())
        state = self._undo_stack.pop()
        self._restore_state(state)

    def redo(self):
        """Re-apply the most recently undone mutation."""
        if not self._redo_stack:
            return
        self._undo_stack.append(self._capture_state())
        state = self._redo_stack.pop()
        self._restore_state(state)

    def clear_history(self):
        """Drop undo/redo history (used after a fresh load)."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_history_changed()

    # ── Snapshots (design comparison) ───────────────────────────────

    def save_snapshot(self, name):
        """Save the current system under a user-visible name.

        Unlike undo history, snapshots persist across mutations and
        are intended for A/B comparison.
        """
        try:
            efl = self.efl_mm
        except Exception:
            efl = float('nan')
        try:
            pres = self.to_prescription()
        except Exception:
            pres = None
        self.snapshots.append({
            'name': name,
            'state': self._capture_state(),
            'efl_mm': efl,
            'prescription': pres,  # for Snapshots-dock Compare
        })
        self.snapshots_changed.emit()

    def load_snapshot(self, index):
        if 0 <= index < len(self.snapshots):
            self._checkpoint()   # loading a snapshot is undoable
            self._restore_state(self.snapshots[index]['state'])

    def delete_snapshot(self, index):
        if 0 <= index < len(self.snapshots):
            del self.snapshots[index]
            self.snapshots_changed.emit()

    # ── Session persistence (autosave) ──────────────────────────────

    @staticmethod
    def _session_path():
        home = os.path.expanduser('~')
        d = os.path.join(home, '.lumenairy')
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            return None
        return os.path.join(d, 'last_session.json')

    def autosave_session(self):
        """Write the current system to ~/.lumenairy/last_session.json.

        Silent no-op on failure (we don't want autosave to spam the
        user with error dialogs).
        """
        path = self._session_path()
        if not path:
            return
        try:
            data = self._state_to_jsonable(self._capture_state())
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.warn('autosave', str(e))
            except Exception:
                pass

    def restore_session(self):
        """Load the autosave file if present.  Returns True on success."""
        path = self._session_path()
        if not path or not os.path.exists(path):
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            state = self._state_from_jsonable(data)
            self._restore_state(state)
            self.clear_history()
            return True
        except Exception as e:
            try:
                from .diagnostics import diag
                diag.warn('session-restore', str(e))
            except Exception:
                pass
            return False

    @staticmethod
    def _state_to_jsonable(state):
        """Convert a _capture_state dict to plain JSON.  Arrays -> lists,
        np.inf -> string sentinel."""
        def enc_val(v):
            if isinstance(v, float):
                if v == float('inf'):
                    return '__inf__'
                if v == float('-inf'):
                    return '__-inf__'
                if v != v:  # NaN
                    return '__nan__'
            return v

        def enc_surface(s):
            return {
                'radius': enc_val(float(s.radius)),
                'thickness': enc_val(float(s.thickness)),
                'glass': s.glass,
                'semi_diameter': enc_val(float(s.semi_diameter)),
                'conic': enc_val(float(s.conic)),
                'surf_type': getattr(s, 'surf_type', 'Standard'),
                'radius_y': enc_val(float(s.radius_y))
                    if s.radius_y is not None else None,
                'conic_y': enc_val(float(s.conic_y))
                    if s.conic_y is not None else None,
            }

        def enc_source(src):
            if src is None:
                return None
            return {
                'source_type': src.source_type,
                'wavelength_nm': src.wavelength_nm,
                'beam_diameter_mm': src.beam_diameter_mm,
                'na': src.na,
                'sigma_mm': src.sigma_mm,
                'object_distance_mm': src.object_distance_mm,
                'emitter_pitch_mm': src.emitter_pitch_mm,
                'emitter_nx': src.emitter_nx,
                'emitter_ny': src.emitter_ny,
                'emitter_waist_mm': src.emitter_waist_mm,
                'field_angle_x_deg': src.field_angle_x_deg,
                'field_angle_y_deg': src.field_angle_y_deg,
            }

        def enc_element(e):
            return {
                'elem_num': e.elem_num,
                'name': e.name,
                'elem_type': e.elem_type,
                'distance_mm': e.distance_mm,
                'tilt_x': e.tilt_x,
                'tilt_y': e.tilt_y,
                'decenter_x': e.decenter_x,
                'decenter_y': e.decenter_y,
                'surfaces': [enc_surface(s) for s in e.surfaces],
                'source': enc_source(e.source),
                'aux': e.aux,
            }

        return {
            'elements': [enc_element(e) for e in state['elements']],
            'wavelength_nm': state['wavelength_nm'],
            'wavelengths_nm': state['wavelengths_nm'],
            'epd_mm': state['epd_mm'],
            'field_angles_deg': state['field_angles_deg'],
            'coordinate_mode': state['coordinate_mode'],
            'opt_variables': list(state['opt_variables']),
            'schema_version': 1,
        }

    @staticmethod
    def _state_from_jsonable(data):
        def dec_val(v):
            if v == '__inf__':
                return float('inf')
            if v == '__-inf__':
                return float('-inf')
            if v == '__nan__':
                return float('nan')
            return v

        def dec_surface(d):
            return SurfaceRow(
                radius=dec_val(d['radius']),
                thickness=dec_val(d['thickness']),
                glass=d.get('glass', ''),
                semi_diameter=dec_val(d['semi_diameter']),
                conic=dec_val(d.get('conic', 0.0)),
                surf_type=d.get('surf_type', 'Standard'),
                radius_y=(dec_val(d['radius_y'])
                          if d.get('radius_y') is not None else None),
                conic_y=(dec_val(d['conic_y'])
                         if d.get('conic_y') is not None else None),
            )

        def dec_source(d):
            if d is None:
                return None
            return SourceDefinition(
                d.get('source_type', 'plane_wave'),
                **{k: v for k, v in d.items() if k != 'source_type'})

        def dec_element(d):
            return Element(
                elem_num=d.get('elem_num', 0),
                name=d.get('name', ''),
                elem_type=d.get('elem_type', 'Singlet'),
                distance_mm=d.get('distance_mm', 0.0),
                tilt_x=d.get('tilt_x', 0.0),
                tilt_y=d.get('tilt_y', 0.0),
                decenter_x=d.get('decenter_x', 0.0),
                decenter_y=d.get('decenter_y', 0.0),
                surfaces=[dec_surface(s) for s in d.get('surfaces', [])],
                source=dec_source(d.get('source')),
                aux=d.get('aux', {}),
            )

        return {
            'elements': [dec_element(e) for e in data['elements']],
            'wavelength_nm': data['wavelength_nm'],
            'wavelengths_nm': data['wavelengths_nm'],
            'epd_mm': data['epd_mm'],
            'field_angles_deg': data['field_angles_deg'],
            'coordinate_mode': data.get('coordinate_mode', 'relative'),
            'opt_variables': [tuple(x) for x in data.get('opt_variables', [])],
        }

    # ── Load from prescription ────────────────────────────────────

    def load_prescription(self, prescription, wavelength_nm=None):
        """Load a prescription dict, grouping surfaces into elements.

        3.6.1 hotfix-5: prefer the importer's full ``elements`` list
        (which includes mirrors with their ``surf_num`` and
        ``comment`` fields) plus ``all_thicknesses`` aligned to it.
        The lens-only ``surfaces`` list strips mirrors and surf_num,
        so consuming it caused fold mirrors to vanish and the
        coord-break absorption pass to find no targets.

        Element labels prefer the per-surface ``comment`` from the
        .zmx COMM field when non-empty; otherwise they get a
        running ``Lens 1`` / ``Mirror 1`` / etc. tag so the
        prescription editor doesn't show every row as the source
        filename.
        """
        self._checkpoint()
        if wavelength_nm is not None:
            self.wavelength_nm = wavelength_nm
            self.wavelengths_nm = [wavelength_nm]

        # Prefer the full elements list (with mirrors + surf_num);
        # fall back to the lens-only surfaces list for legacy
        # callers that don't ship the new key.
        full = prescription.get('elements')
        if full is not None:
            rx_items = full
            rx_thick_full = prescription.get('all_thicknesses', []) or []
        else:
            rx_items = prescription.get('surfaces', []) or []
            rx_thick_full = prescription.get('thicknesses', []) or []

        aperture = prescription.get('aperture_diameter')

        # ------------------------------------------------------------
        # 1. Identify dummy air-air infinite-R surfaces with no
        #    comment.  These are Zemax DISZ spacers / COORDBRK
        #    reference planes and should not appear in the
        #    prescription editor; their thicknesses get absorbed
        #    into the next rendered element's distance.
        # ------------------------------------------------------------
        def _is_dummy(item):
            if item.get('element_type') == 'mirror':
                return False
            gb = item.get('glass_before', 'air')
            ga = item.get('glass_after', 'air')
            R = item.get('radius', np.inf)
            cmt = (item.get('comment') or '').strip()
            return (gb == 'air' and ga == 'air'
                    and not np.isfinite(R) and not cmt)

        # Build the cb map keyed by the next NON-DUMMY surf_num
        # so a coord-break that nominally targets a skipped dummy
        # still lands on the right rendered element.
        cbs = prescription.get('coord_breaks', []) or []
        sorted_cbs = sorted(cbs, key=lambda c: int(c.get('surf_num', 0)))
        rendered_surf_nums = [
            int(it['surf_num']) for it in rx_items
            if not _is_dummy(it) and it.get('surf_num') is not None]
        cb_for_surf = {}
        for cb in sorted_cbs:
            cb_n = int(cb.get('surf_num', 0))
            target = next((sn for sn in rendered_surf_nums
                            if sn > cb_n), None)
            if target is None:
                continue
            slot = cb_for_surf.setdefault(target, {
                'tilt_x_deg': 0.0, 'tilt_y_deg': 0.0,
                'decenter_x_m': 0.0, 'decenter_y_m': 0.0,
            })
            slot['tilt_x_deg'] += float(cb.get('tilt_x_deg', 0.0))
            slot['tilt_y_deg'] += float(cb.get('tilt_y_deg', 0.0))
            slot['decenter_x_m'] += float(cb.get('decenter_x_m', 0.0))
            slot['decenter_y_m'] += float(cb.get('decenter_y_m', 0.0))

        if aperture:
            self.epd_mm = aperture * 1e3
        sd_default = aperture * 1e3 / 2 if aperture else np.inf

        # ------------------------------------------------------------
        # 2. Reset the element list (preserve any existing source).
        # ------------------------------------------------------------
        self.elements = [
            Element(0, 'Source', 'Source', distance_mm=0.0,
                    source=self.elements[0].source if self.elements
                    else SourceDefinition()),
        ]

        def _per_surf_sd(item):
            psd = item.get('semi_diameter')
            if psd is not None and np.isfinite(psd):
                return psd * 1e3
            return sd_default

        def _attach_cb(item):
            cb = cb_for_surf.get(int(item.get('surf_num', -1)))
            if not cb:
                return 0.0, 0.0, 0.0, 0.0
            return (cb['tilt_x_deg'], cb['tilt_y_deg'],
                    cb['decenter_x_m'] * 1e3,
                    cb['decenter_y_m'] * 1e3)

        # ------------------------------------------------------------
        # 3. Walk the items, grouping refractive surfaces into
        #    Singlet / Doublet / Triplet elements and emitting
        #    Mirror elements directly.  Dummies are skipped but
        #    their thickness is rolled into ``pending_dist_mm``.
        #
        # 3.7.4: keep Zemax-signed thicknesses as the canonical
        # storage (negative post-mirror, positive otherwise).  The
        # previous "convert to physical positive + flip R[:,2] at
        # each mirror" approach (3.6.1 hotfix-6) double-counted
        # for tilted mirrors and made the layout's post-fold
        # element positions diverge from the actual ray direction.
        # With Zemax-signed distances and no Rmirror at mirrors
        # in recompute_element_frames, the trace's N-flip handles
        # the direction reversal cleanly and the element world
        # positions match where rays actually go.
        # ------------------------------------------------------------
        pending_dist_mm = 0.0
        n_lens = 0
        n_mirror = 0

        i = 0
        while i < len(rx_items):
            item = rx_items[i]
            item_type = item.get('element_type', 'surface')
            comment = (item.get('comment') or '').strip()
            sd_local = _per_surf_sd(item)
            # 3.7.4: keep the Zemax-signed thickness as-is (was being
            # converted to physical-positive via mirror parity in
            # 3.6.1 hotfix-6; that conversion is reverted).
            t_after_mm = (rx_thick_full[i] * 1e3
                          if i < len(rx_thick_full) else 0.0)

            # Skip dummy spacer surfaces.
            if _is_dummy(item):
                pending_dist_mm += t_after_mm
                i += 1
                continue

            # ── Mirror element ──
            if item_type == 'mirror':
                R_mm = (item['radius'] * 1e3
                        if np.isfinite(item['radius']) else np.inf)
                conic = item.get('conic', 0.0)
                n_mirror += 1
                label = comment or f'Mirror {n_mirror}'
                tx, ty, dx, dy = _attach_cb(item)
                s = SurfaceRow(R_mm, 0.0, '', sd_local, conic,
                                surf_type='Mirror')
                elem = Element(len(self.elements), label, 'Mirror',
                               distance_mm=pending_dist_mm,
                               surfaces=[s],
                               tilt_x=tx, tilt_y=ty,
                               decenter_x=dx, decenter_y=dy)
                self.elements.append(elem)
                pending_dist_mm = t_after_mm
                # 3.7.4: no thickness_sign flip (was reverted to
                # keep Zemax-signed convention as canonical).
                i += 1
                continue

            # ── Refractive surface (or specialty plane) ──
            glass_after = item.get('glass_after', 'air')

            if glass_after != 'air':
                # Start of a lens — find the matching air boundary.
                j = i + 1
                while j < len(rx_items):
                    prev_glass = rx_items[j - 1].get('glass_after', 'air')
                    if prev_glass == 'air':
                        break
                    j += 1
                n_surf = j - i
                if n_surf == 2:
                    etype = 'Singlet'
                elif n_surf == 3:
                    etype = 'Doublet'
                elif n_surf == 4:
                    etype = 'Triplet'
                else:
                    etype = 'Singlet'

                tx, ty, dx, dy = _attach_cb(item)

                surf_rows = []
                for k in range(i, j):
                    rk = rx_items[k]
                    Rk_mm = (rk['radius'] * 1e3
                             if np.isfinite(rk['radius']) else np.inf)
                    if k < j - 1:
                        # 3.7.4: Internal thickness to the next surface
                        # of THIS lens, kept Zemax-signed.
                        tk_mm = (rx_thick_full[k] * 1e3
                                 if k < len(rx_thick_full) else 0.0)
                    else:
                        tk_mm = 0.0
                    glass = rk.get('glass_after', '')
                    if glass == 'air':
                        glass = ''
                    conic = rk.get('conic', 0.0)
                    surf_rows.append(SurfaceRow(
                        Rk_mm, tk_mm, glass,
                        _per_surf_sd(rk), conic))

                n_lens += 1
                label = comment or f'Lens {n_lens}'
                elem = Element(len(self.elements), label, etype,
                               distance_mm=pending_dist_mm,
                               surfaces=surf_rows,
                               tilt_x=tx, tilt_y=ty,
                               decenter_x=dx, decenter_y=dy)
                self.elements.append(elem)
                # 3.7.4: Distance to next element kept Zemax-signed.
                last_k = j - 1
                pending_dist_mm = (
                    rx_thick_full[last_k] * 1e3
                    if last_k < len(rx_thick_full)
                    else 0.0)
                i = j
                continue

            # Specialty air-air plane with a comment (e.g. DGRATING,
            # SpatialFilter): render as a flat zero-thickness
            # element labelled by the comment so the user sees it.
            if comment:
                R_mm = (item['radius'] * 1e3
                        if np.isfinite(item['radius']) else np.inf)
                conic = item.get('conic', 0.0)
                tx, ty, dx, dy = _attach_cb(item)
                s = SurfaceRow(R_mm, 0.0, '', sd_local, conic)
                elem = Element(len(self.elements), comment, 'Singlet',
                               distance_mm=pending_dist_mm,
                               surfaces=[s],
                               tilt_x=tx, tilt_y=ty,
                               decenter_x=dx, decenter_y=dy)
                self.elements.append(elem)
                pending_dist_mm = t_after_mm
                i += 1
                continue

            # Trailing glass→air surface that didn't get grouped —
            # accumulate its thickness and move on.
            pending_dist_mm += t_after_mm
            i += 1

        # Detector at the BFL
        try:
            trace_surfs = self._build_trace_surfaces_internal()
            _, _, bfl, _ = system_abcd(trace_surfs, self.wavelength_m)
            det_dist = bfl * 1e3 if np.isfinite(bfl) and bfl > 0 else 100.0
        except Exception:
            det_dist = 100.0

        self.elements.append(
            Element(len(self.elements), 'Detector', 'Detector',
                    distance_mm=det_dist))

        self._renumber()
        self._invalidate()
        self.system_changed.emit()

    # ── Build flat surface list for trace engine ──────────────────

    def build_trace_surfaces(self):
        """Flatten elements to a list of raytrace.Surface objects."""
        if self._flat_surfaces_cache is not None:
            return self._flat_surfaces_cache
        self._flat_surfaces_cache = self._build_trace_surfaces_internal()
        return self._flat_surfaces_cache

    def build_trace_surfaces_world(self):
        """3.7.7: Public cached accessor for the world-frame
        trace surface list (one :class:`Surface` per actual
        optical surface, each carrying absolute ``world_origin``
        and ``world_R``).  Pair with :func:`trace_world` for
        folded-system-accurate analysis from any dock that
        currently calls ``build_trace_surfaces()`` +
        :func:`trace`.

        The world list has FEWER entries than the legacy local
        list -- no ``is_coordbrk=True`` Surfaces, since tilts
        are absorbed into each surface's ``world_R``.  Any code
        that indexes ray history by hard-coded surface number
        and runs against tilted systems needs to walk
        ``result.surfaces`` rather than the legacy list to stay
        index-aligned.
        """
        if self._flat_surfaces_world_cache is not None:
            return self._flat_surfaces_world_cache
        self._flat_surfaces_world_cache = self._build_trace_surfaces_world()
        return self._flat_surfaces_world_cache

    def build_run_trace_world_surfaces(self, image_distance=None):
        """3.7.7: World-frame surface list with a final
        image-plane Surface appended at the Detector's world
        frame (or, if no Detector / zero distance, at the last
        surface advanced by ``image_distance`` (or the paraxial
        BFL) along its local +z axis).

        This is the surface list that the docks need when they
        want a "full trace with image plane".  Returns a fresh
        copy so the caller can mutate (e.g. override the image
        distance) without affecting the cache.
        """
        world_list = [Surface(
            radius=s.radius, conic=s.conic, semi_diameter=s.semi_diameter,
            glass_before=s.glass_before, glass_after=s.glass_after,
            is_mirror=s.is_mirror, thickness=s.thickness,
            label=s.label, surf_num=s.surf_num,
            radius_y=s.radius_y, conic_y=s.conic_y,
            world_origin=(s.world_origin.copy()
                          if s.world_origin is not None else None),
            world_R=(s.world_R.copy()
                     if s.world_R is not None else None),
        ) for s in self.build_trace_surfaces_world()]
        if not world_list:
            return world_list

        det = self.elements[-1] if self.elements else None
        img_world_origin = None
        img_world_R = None
        if det and det.elem_type == 'Detector' and det.distance_mm > 0:
            img_world_origin = np.asarray(det.origin, dtype=float) * 1e-3
            img_world_R = np.asarray(det.R, dtype=float).copy()
        elif image_distance is None:
            try:
                bfl = find_paraxial_focus(world_list, self.wavelength_m)
                if np.isfinite(bfl) and bfl > 0:
                    image_distance = bfl
            except Exception:
                pass

        if img_world_origin is None and image_distance is not None:
            last = world_list[-1]
            img_world_origin = (last.world_origin
                                + image_distance * last.world_R[:, 2])
            img_world_R = last.world_R.copy()

        if img_world_origin is not None and img_world_R is not None:
            last_glass = world_list[-1].glass_after
            world_list.append(Surface(
                radius=np.inf, semi_diameter=np.inf,
                glass_before=last_glass, glass_after=last_glass,
                label='Image',
                world_origin=img_world_origin,
                world_R=img_world_R,
            ))
        return world_list

    def _build_trace_surfaces_internal(self):
        """Flatten the element list into a sequential Surface list.

        3.7.0: emits a coord-break Surface BEFORE any element that
        has non-zero ``tilt_x`` / ``tilt_y`` / ``decenter_x`` /
        ``decenter_y`` (and a matching un-tilt cb AFTER if the
        element is a Mirror, so the post-mirror axis returns to a
        Zemax-style "always +z" representation for any subsequent
        elements that aren't themselves tilted).  Untilted systems
        emit no coord-breaks → identical behaviour to pre-3.7.

        The trace history will have one entry per surface in this
        list, including coord-breaks; ``surface_frames_2d/3d_mm``
        emit matching world frames in the same order so ray
        rendering stays aligned.
        """
        trace_surfaces = []

        def _has_tilt(elem):
            return (
                float(getattr(elem, 'tilt_x', 0.0)) != 0.0
                or float(getattr(elem, 'tilt_y', 0.0)) != 0.0
                or float(getattr(elem, 'decenter_x', 0.0)) != 0.0
                or float(getattr(elem, 'decenter_y', 0.0)) != 0.0)

        # 3.7.5: cb_post air-gap routing.  When the previous element
        # was a Mirror and the current element carries an absorbed
        # cb (the original .zmx had a cb AFTER the mirror so the
        # post-fold axis re-aligns with the reflected ray), the gap
        # from the mirror to this element MUST be carried on the
        # cb's thickness, NOT on the mirror's surface thickness.
        # Reason: in the mirror's local frame, the reflected ray's
        # N (= cos(45°) ≈ 0.707 for a 45° fold) is not 1, so a
        # transfer of d in that frame moves the ray by d/N ≈ 1.41 d
        # in world along the wrong direction (cb-pre-tilted-z, not
        # the reflected ray direction).  Routing the gap through
        # the cb instead puts the transfer in the POST-cb local
        # frame, where N ≈ 1 and the world displacement matches the
        # GUI's element_frames placement (origin += d * R[:,2] in
        # the post-cb-post rotation).
        prev_elem_was_mirror = False

        for ei, elem in enumerate(self.elements):
            if elem.elem_type in ('Source', 'Detector'):
                continue
            if not elem.surfaces:
                continue

            tilt_pre = _has_tilt(elem)
            cb_post_case = prev_elem_was_mirror and tilt_pre

            # 3.7.0: Insert a coord-break right before this element
            # if it carries any tilt/decenter (typically absorbed
            # from an imported COORDBRK).  The cb sits AT the
            # element's front vertex position.
            #
            # cb_pre case (non-mirror predecessor): the previous
            # surface's air-gap thickness already carried the ray
            # to that position in the OLD (pre-tilt) local frame,
            # the cb transforms the frame in place, and the cb's
            # own thickness is 0 -- the element's first surface is
            # immediately after the cb at the same world position.
            #
            # cb_post case (mirror predecessor): the gap is carried
            # by THIS cb's thickness in the post-cb local frame so
            # the transfer happens with N ≈ 1.  The mirror's
            # surface thickness stays at 0 (no transfer in the
            # cb-pre-tilted frame).
            if tilt_pre:
                cb_thickness = (elem.distance_mm * 1e-3
                                if cb_post_case else 0.0)
                trace_surfaces.append(Surface(
                    is_coordbrk=True,
                    tilt_x_deg=float(getattr(elem, 'tilt_x', 0.0)),
                    tilt_y_deg=float(getattr(elem, 'tilt_y', 0.0)),
                    decenter_x_m=(float(getattr(elem, 'decenter_x', 0.0))
                                   * 1e-3),
                    decenter_y_m=(float(getattr(elem, 'decenter_y', 0.0))
                                   * 1e-3),
                    coordbrk_order=0,
                    thickness=cb_thickness,
                    glass_before='air', glass_after='air',
                    label=f'{elem.name} CB-pre',
                    surf_num=len(trace_surfaces),
                ))

            for si, srow in enumerate(elem.surfaces):
                R_m = srow.radius * 1e-3 if np.isfinite(srow.radius) else np.inf
                sd_m = srow.semi_diameter * 1e-3 if np.isfinite(srow.semi_diameter) else np.inf

                # Glass before/after
                if si == 0:
                    glass_before = 'air'
                else:
                    prev_glass = elem.surfaces[si - 1].glass
                    glass_before = prev_glass if prev_glass else 'air'
                glass_after = srow.glass if srow.glass else 'air'

                # Thickness: internal glass thickness to next surface in element
                if si < len(elem.surfaces) - 1:
                    thick_m = srow.thickness * 1e-3
                else:
                    thick_m = 0.0  # last surface — air gap handled below

                is_mirror = (srow.surf_type == 'Mirror' or elem.elem_type == 'Mirror')

                # Biconic fields: convert mm to m for radius_y
                ry_m = None
                if srow.radius_y is not None:
                    ry_m = srow.radius_y * 1e-3 if np.isfinite(srow.radius_y) else np.inf

                trace_surfaces.append(Surface(
                    radius=R_m, conic=srow.conic, semi_diameter=sd_m,
                    glass_before=glass_before, glass_after=glass_after,
                    is_mirror=is_mirror, thickness=thick_m,
                    label=f'{elem.name} S{si+1}',
                    surf_num=len(trace_surfaces),
                    radius_y=ry_m,
                    conic_y=srow.conic_y,
                ))

            # Set the air gap from this element's last surface to the next
            # element's first surface (but NOT for Detector — that's
            # handled by run_trace).  3.7.5: if the next element will
            # absorb the gap onto its own cb_pre (cb_post case),
            # leave this element's last surface thickness at 0 — the
            # gap is carried in the post-cb frame instead.
            if trace_surfaces and ei + 1 < len(self.elements):
                next_elem = self.elements[ei + 1]
                if next_elem.elem_type != 'Detector' and next_elem.surfaces:
                    next_is_cb_post = (elem.elem_type == 'Mirror'
                                       and _has_tilt(next_elem))
                    if not next_is_cb_post:
                        trace_surfaces[-1].thickness = next_elem.distance_mm * 1e-3

            prev_elem_was_mirror = (elem.elem_type == 'Mirror')

        return trace_surfaces

    def _build_trace_surfaces_world(self):
        """3.7.5: World-frame trace surface list.

        Emits ONE :class:`Surface` per actual optical surface (S1,
        S2, ...).  Each Surface carries:

        * ``world_origin`` (m, shape (3,)): the surface's vertex
          position in world coordinates.
        * ``world_R``      (shape (3, 3)): the local-to-world
          rotation, i.e. the columns are local +x/+y/+z expressed in
          world coords.  Surface curvature / sag is defined in this
          local frame (z = sag(x, y) with vertex at origin).

        Walks each element from front to back and steps along its
        local +z axis (``elem.R[:, 2]``) by the per-surface internal
        thicknesses, so multi-surface elements (Singlets,
        Diffractives, Aspherics) get correctly placed S1, S2, etc.

        No coord-break Surfaces are emitted -- tilts and decenters
        are absorbed into each element's ``R`` and ``origin`` by
        :meth:`recompute_element_frames`.  ``surface.thickness`` is
        carried over for paraxial / ABCD compatibility but is not
        used by :func:`trace_world`.
        """
        world_surfaces = []
        for elem in self.elements:
            if elem.elem_type in ('Source', 'Detector'):
                continue
            if not elem.surfaces:
                continue

            origin = np.asarray(elem.origin, dtype=float)
            R = np.asarray(elem.R, dtype=float)
            z_axis = R[:, 2]

            cum_t_m = 0.0
            for si, srow in enumerate(elem.surfaces):
                R_m = (srow.radius * 1e-3
                       if np.isfinite(srow.radius) else np.inf)
                sd_m = (srow.semi_diameter * 1e-3
                        if np.isfinite(srow.semi_diameter) else np.inf)
                if si == 0:
                    glass_before = 'air'
                else:
                    prev_glass = elem.surfaces[si - 1].glass
                    glass_before = prev_glass if prev_glass else 'air'
                glass_after = srow.glass if srow.glass else 'air'

                # Carry the internal thickness on the surface (in
                # metres) only so that paraxial helpers reading the
                # legacy ``thickness`` field still work; the world
                # trace itself ignores it.
                if si < len(elem.surfaces) - 1:
                    thick_m = srow.thickness * 1e-3
                else:
                    thick_m = 0.0

                is_mirror = (srow.surf_type == 'Mirror'
                             or elem.elem_type == 'Mirror')

                ry_m = None
                if srow.radius_y is not None:
                    ry_m = (srow.radius_y * 1e-3
                            if np.isfinite(srow.radius_y) else np.inf)

                # World vertex position for this surface: element
                # front vertex (in mm per recompute_element_frames)
                # advanced by the cumulative internal thickness (m)
                # along the element's local +z.  Output in metres so
                # the trace's intersection / OPL math stays in SI.
                surf_world_origin = (origin * 1e-3) + cum_t_m * z_axis

                world_surfaces.append(Surface(
                    radius=R_m, conic=srow.conic, semi_diameter=sd_m,
                    glass_before=glass_before, glass_after=glass_after,
                    is_mirror=is_mirror, thickness=thick_m,
                    label=f'{elem.name} S{si+1}',
                    surf_num=len(world_surfaces),
                    radius_y=ry_m,
                    conic_y=srow.conic_y,
                    world_origin=surf_world_origin,
                    world_R=R.copy(),
                ))

                cum_t_m += float(srow.thickness) * 1e-3

        return world_surfaces

    # ── ABCD ───────────────────────────────────────────────────────

    def _ensure_abcd(self):
        if self._abcd is not None:
            return
        surfaces = self.build_trace_surfaces()
        if not surfaces:
            self._abcd = np.eye(2)
            self._efl = np.inf
            self._bfl = np.inf
            return
        try:
            self._abcd, self._efl, self._bfl, _ = system_abcd(
                surfaces, self.wavelength_m)
        except Exception:
            self._abcd = np.eye(2)
            self._efl = np.inf
            self._bfl = np.inf

    def get_abcd(self):
        self._ensure_abcd()
        return self._abcd, self._efl, self._bfl

    # ── Ray trace ──────────────────────────────────────────────────

    def retrace(self):
        """Public retrace helper, honouring auto_retrace_mode.

        * ``'on'``   : runs :meth:`run_trace` unconditionally.
        * ``'geometric-only'`` : same, but skips any wave-level work in
          downstream consumers (this flag is read by them).
        * ``'manual'`` : no-op; the caller is expected to invoke
          :meth:`run_trace` directly (e.g. Ctrl+T).
        """
        if self.auto_retrace_mode == 'manual':
            return None
        return self.run_trace()

    def run_trace(self, num_rings=8, rays_per_ring=36, image_distance=None):
        """Run a geometric ray trace using the active source definition.

        3.7.5: traces in WORLD coordinates via :func:`trace_world`.
        Each Surface carries its own ``world_origin`` and ``world_R``;
        rays propagate sequentially in world coords between surfaces
        and are transformed into each surface's local frame only for
        the intersection / refraction step.  No coord-break surfaces.
        """
        self.trace_started.emit()
        world_list = self._build_trace_surfaces_world()
        surfaces = [Surface(
            radius=s.radius, conic=s.conic, semi_diameter=s.semi_diameter,
            glass_before=s.glass_before, glass_after=s.glass_after,
            is_mirror=s.is_mirror, thickness=s.thickness,
            label=s.label, surf_num=s.surf_num,
            radius_y=s.radius_y, conic_y=s.conic_y,
            world_origin=(s.world_origin.copy()
                          if s.world_origin is not None else None),
            world_R=(s.world_R.copy()
                     if s.world_R is not None else None),
        ) for s in world_list]
        if not surfaces:
            return None

        semi_ap = self.epd_m / 2.0
        wv = self.wavelength_m
        src = self.source

        # Off-axis field angle from the source.  ``make_rings`` accepts
        # a single y-axis tilt; combine X+Y inputs as their RMS so that
        # a (3 deg, 4 deg) source produces a 5 deg meridional fan.  For
        # full XY field grids use the optimizer dock's multi-field path.
        fa_x = float(src.field_angle_x_deg) if src else 0.0
        fa_y = float(src.field_angle_y_deg) if src else 0.0
        fa_rad = np.radians(np.hypot(fa_x, fa_y))

        # Generate rays based on source type
        if src and src.source_type == 'point_source':
            # Point source: rays diverge from a point on axis
            obj_dist = src.object_distance_mm * 1e-3
            from ..raytrace import _make_bundle
            tilt_M = np.tan(np.radians(fa_y)) if fa_y else 0.0
            tilt_L = np.tan(np.radians(fa_x)) if fa_x else 0.0
            all_y = []
            all_M = []
            all_x = []
            all_L = []
            for ring in range(1, num_rings + 1):
                frac = ring / num_rings
                theta = np.linspace(0, 2 * np.pi, rays_per_ring,
                                    endpoint=False)
                for t in theta:
                    all_x.append(0.0)
                    all_y.append(0.0)
                    all_L.append(frac * semi_ap / obj_dist + tilt_L)
                    all_M.append(frac * semi_ap / obj_dist + tilt_M)
            all_x.append(0.0); all_y.append(0.0)
            all_L.append(tilt_L); all_M.append(tilt_M)
            rays = _make_bundle(
                np.array(all_x), np.array(all_y),
                np.array(all_L), np.array(all_M), wv)
        elif src and src.source_type == 'gaussian':
            # Gaussian: rays weighted by beam profile (just use beam radius as aperture)
            beam_rad = src.beam_diameter_mm * 1e-3 / 2
            rays = make_rings(min(beam_rad, semi_ap), num_rings,
                              rays_per_ring, fa_rad, wv)
        else:
            # Default: plane wave filling EPD
            rays = make_rings(semi_ap, num_rings, rays_per_ring,
                              fa_rad, wv)

        # 3.7.5: Image plane in world coords.  Prefer the Detector
        # element's world frame so the image plane is at the
        # detector's true world position even after folds.  Falls
        # back to last-surface + paraxial-bfl along last surface's
        # local +z if no Detector exists or it's at zero distance.
        det = self.elements[-1] if self.elements else None
        img_world_origin = None
        img_world_R = None
        if det and det.elem_type == 'Detector' and det.distance_mm > 0:
            img_world_origin = np.asarray(det.origin, dtype=float) * 1e-3
            img_world_R = np.asarray(det.R, dtype=float).copy()
        elif image_distance is None:
            try:
                bfl = find_paraxial_focus(surfaces, wv)
                if np.isfinite(bfl) and bfl > 0:
                    image_distance = bfl
            except Exception:
                pass

        if img_world_origin is None and image_distance is not None and surfaces:
            last = surfaces[-1]
            img_world_origin = (last.world_origin
                                + image_distance * last.world_R[:, 2])
            img_world_R = last.world_R.copy()

        if img_world_origin is not None and img_world_R is not None:
            last_glass = surfaces[-1].glass_after if surfaces else 'air'
            surfaces.append(Surface(
                radius=np.inf, semi_diameter=np.inf,
                glass_before=last_glass, glass_after=last_glass,
                label='Image',
                world_origin=img_world_origin,
                world_R=img_world_R,
            ))

        result = trace_world(rays, surfaces, wv)
        self._trace_result = result
        self.trace_ready.emit(result)
        return result

    # ── Optimization ───────────────────────────────────────────────

    def get_variable_values(self):
        values = []
        for elem_idx, surf_idx, field in self.opt_variables:
            if elem_idx < len(self.elements):
                elem = self.elements[elem_idx]
                if field == 'distance':
                    values.append(elem.distance_mm)
                elif surf_idx < len(elem.surfaces):
                    values.append(getattr(elem.surfaces[surf_idx], field, 0.0))
        return np.array(values)

    def set_variable_values(self, values):
        for i, (elem_idx, surf_idx, field) in enumerate(self.opt_variables):
            if elem_idx < len(self.elements):
                elem = self.elements[elem_idx]
                if field == 'distance':
                    elem.distance_mm = values[i]
                elif surf_idx < len(elem.surfaces):
                    setattr(elem.surfaces[surf_idx], field, values[i])
        self._invalidate()

    def add_optimization_variable(self, elem_idx, surf_idx, field):
        """3.6.1: append (elem_idx, surf_idx, field) to opt_variables
        if not already present, then emit system_changed so the
        Slider dock and the optimizer's variable-grid dialog notice
        the new variable.

        Used by the OSLO-style "Attach slider to this parameter…"
        right-click action on the prescription editor: a single
        click promotes a cell into an optimization variable without
        the user having to open the optimizer dock first.

        Returns True when a new variable was added; False when the
        triple was already in the list (idempotent).
        """
        triple = (int(elem_idx), int(surf_idx), str(field))
        for existing in self.opt_variables:
            if tuple(existing) == triple:
                return False
        self.opt_variables.append(triple)
        try:
            self.system_changed.emit()
        except Exception:
            pass
        return True

    # Merit type for the geometric optimizer.  Set by the optimizer dock.
    # Valid values: 'rms_spot' (default), 'efl_target', 'bfl_target',
    #               'seidel_spherical', 'min_thickness', 'max_fnumber'
    geo_merit_type = 'rms_spot'
    geo_merit_target = 100.0  # target value in mm (for efl/bfl) or ratio

    def merit_function(self, values):
        self.set_variable_values(values)
        surfaces = self._build_trace_surfaces_internal()
        if not surfaces:
            return 1e10

        wv_m = self.wavelengths_nm[0] * 1e-9
        surfs = [Surface(
            radius=s.radius, conic=s.conic,
            semi_diameter=s.semi_diameter,
            glass_before=s.glass_before, glass_after=s.glass_after,
            is_mirror=s.is_mirror, thickness=s.thickness,
            radius_y=getattr(s, 'radius_y', None),
            conic_y=getattr(s, 'conic_y', None),
        ) for s in surfaces]

        mt = self.geo_merit_type

        # --- EFL / BFL target ---
        if mt in ('efl_target', 'bfl_target'):
            try:
                _, efl, bfl, _ = system_abcd(surfs, wv_m)
            except Exception:
                return 1e10
            target_m = self.geo_merit_target * 1e-3
            if mt == 'efl_target':
                err = (efl - target_m) / max(abs(target_m), 1e-12)
            else:
                err = (bfl - target_m) / max(abs(target_m), 1e-12)
            return err * err

        # --- Seidel spherical ---
        if mt == 'seidel_spherical':
            try:
                seidel_raw = seidel_coefficients(surfs, wv_m)
                if isinstance(seidel_raw, tuple) and isinstance(seidel_raw[0], dict):
                    s1 = float(np.sum(seidel_raw[0].get('S1', np.zeros(1))))
                else:
                    s1 = 0.0
            except Exception:
                s1 = 0.0
            return s1 ** 2

        # --- Min thickness ---
        if mt == 'min_thickness':
            penalty = 0.0
            for elem in self.elements:
                for s in elem.surfaces:
                    if s.thickness < 1.0:  # less than 1 mm
                        penalty += (1.0 - s.thickness) ** 2
            return penalty

        # --- Max f-number ---
        if mt == 'max_fnumber':
            try:
                _, efl, _, _ = system_abcd(surfs, wv_m)
                fnum = abs(efl) / max(self.epd_m, 1e-12)
                target = self.geo_merit_target
                excess = max(0.0, fnum - target)
                return excess * excess
            except Exception:
                return 1e10

        # --- Default: RMS spot (original behavior) ---
        semi_ap = self.epd_m / 2.0
        total = 0.0
        n = 0
        for wv_nm in self.wavelengths_nm:
            wv_m = wv_nm * 1e-9
            for fa_deg in self.field_angles_deg:
                fa_rad = np.radians(fa_deg)
                rays = make_rings(semi_ap, 6, 24, fa_rad, wv_m)
                surfs_copy = [Surface(
                    radius=s.radius, conic=s.conic,
                    semi_diameter=s.semi_diameter,
                    glass_before=s.glass_before, glass_after=s.glass_after,
                    is_mirror=s.is_mirror, thickness=s.thickness,
                    radius_y=getattr(s, 'radius_y', None),
                    conic_y=getattr(s, 'conic_y', None),
                ) for s in surfaces]
                try:
                    bfl = find_paraxial_focus(surfs_copy, wv_m)
                    if np.isfinite(bfl) and bfl > 0:
                        surfs_copy[-1].thickness = bfl
                        surfs_copy.append(Surface(radius=np.inf, semi_diameter=np.inf,
                            glass_before=surfs_copy[-1].glass_after,
                            glass_after=surfs_copy[-1].glass_after))
                    result = trace(rays, surfs_copy, wv_m)
                    rms, _ = spot_rms(result)
                    total += rms ** 2
                    n += 1
                except Exception:
                    return 1e10
        return np.sqrt(total / max(n, 1))

    def run_optimization(self, max_iter=200, callback=None):
        from scipy.optimize import minimize
        if not self.opt_variables:
            return False, 'No variables defined.'
        x0 = self.get_variable_values()
        iteration = [0]
        def _cb(xk):
            iteration[0] += 1
            merit = self.merit_function(xk)
            self.optimization_progress.emit(iteration[0], merit)
            if callback:
                callback(iteration[0], merit)
        try:
            result = minimize(self.merit_function, x0, method='Nelder-Mead',
                              options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-12},
                              callback=_cb)
            self.set_variable_values(result.x)
            self._invalidate()
            self.system_changed.emit()
            msg = f'Merit: {result.fun*1e6:.3f} um after {result.nit} iterations'
            self.optimization_finished.emit(result.success, msg)
            return result.success, msg
        except Exception as e:
            self.optimization_finished.emit(False, str(e))
            return False, str(e)

    # ── Grid recommendation ─────────────────────────────────────────

    def recommend_grid(self):
        """Recommend N and dx based on Nyquist sampling of the system NA.

        Cross-checks the recommendation against the OPD-extraction
        Nyquist rule ``dx <= lambda * f / aperture`` from
        :func:`check_opd_sampling`.  If the NA-based dx fails that
        rule, dx is tightened until the OPD margin is safe (>= 2);
        N is rebuilt to keep the same aperture coverage.

        Returns (N, dx_um) where N is a "nice" FFT size (2^a, 2^a*3, or
        2^a*5) and dx_um is the grid spacing in micrometres.

        Delegates to :func:`lumenairy.recommend_grid_for_prescription`
        when the system can be exported as a prescription dict; that
        path also handles DOE diffraction-order spread correctly.
        Falls back to the local NA-based heuristic when the
        prescription export fails (sourceless system, in-progress
        edits, etc.).
        """
        # Prefer the library implementation when we can build a
        # prescription dict from the current model -- it handles DOE
        # diffraction-order spread and any future heuristic
        # improvements that land in the core.
        try:
            import lumenairy as _la
            pres = self.to_prescription()
            rec = _la.recommend_grid_for_prescription(
                pres, self.wavelength_m)
            return int(rec['N']), float(rec['dx']) * 1e6
        except Exception:
            pass

        surfs = self.build_trace_surfaces()
        wv = self.wavelength_m
        efl = self.efl_mm * 1e-3  # m

        # NA from the entrance pupil and EFL
        if np.isfinite(efl) and abs(efl) > 1e-9:
            na = self.epd_m / (2 * abs(efl))
        else:
            na = 0.1

        # Nyquist: dx <= lambda / (2 * NA)
        dx_nyquist = wv / (2 * max(na, 0.01))
        # Round down to a clean value
        dx = _nice_dx(dx_nyquist)

        # Cross-check against the OPD-unwrap Nyquist rule.  If the
        # converging-wavefront sampling margin is < 2 we tighten dx
        # before locking it in, otherwise the wave-optics dock will
        # produce wrong OPDs even though the NA-based rule looks fine.
        if np.isfinite(efl) and abs(efl) > 1e-9 and self.epd_m > 0:
            try:
                from ..analysis import check_opd_sampling
                samp = check_opd_sampling(
                    dx=dx, wavelength=wv,
                    aperture=self.epd_m, focal_length=abs(efl))
                if not samp.get('ok', True):
                    # check_opd_sampling reports dx_max; halve dx until
                    # we have margin.  Cap at 1 nm to avoid runaway.
                    dx_max = float(samp.get('dx_max', dx))
                    while dx > max(dx_max / 2, 1e-9):
                        dx = dx / 2
                    dx = _nice_dx(dx)
            except Exception:
                pass

        # Grid must span the largest aperture with 1.5x margin
        max_ap = 0.0
        for s in surfs:
            if np.isfinite(s.semi_diameter):
                max_ap = max(max_ap, s.semi_diameter)
        if max_ap == 0:
            max_ap = self.epd_m / 2

        N_min = int(np.ceil(2 * max_ap * 1.5 / dx))
        N = _next_nice_N(max(N_min, 128))

        return N, dx * 1e6  # dx in µm

    # ── Export ─────────────────────────────────────────────────────

    def to_prescription(self):
        """Export the current system as a prescription dict.

        Biconic / anamorphic keys (``radius_y``, ``conic_y``,
        ``aspheric_coeffs_y``) are always included so that an
        export -> import round-trip is information-preserving even when
        the surface is rotationally symmetric (None vs missing-key are
        both interpreted as "symmetric" by the core, but emitting the
        keys makes diff'ing two prescriptions reliable).

        3.7.0: also emits ``elements`` (full chronological list with
        mirrors), ``all_thicknesses`` (aligned to ``elements``), and
        ``coord_breaks`` (one entry per tilted/decentered element)
        so the round-trip through .zmx export preserves fold geometry
        and tilt/decenter information.  The legacy lens-only
        ``surfaces`` / ``thicknesses`` keys remain for callers that
        expect the pre-3.7 format.
        """
        # Legacy lens-only path (refractive surfaces from the trace
        # surface list, with cb's filtered out -- their thicknesses
        # are 0 in the cb emission so this just gives the same
        # refractive sequence as pre-3.7 unfolded systems).
        surfaces = self.build_trace_surfaces()
        rx_surfaces = []
        thicknesses = []
        for i, s in enumerate(surfaces):
            if s.is_coordbrk:
                continue
            surf_dict = {
                'radius': s.radius,
                'conic': s.conic,
                'aspheric_coeffs': getattr(s, 'aspheric_coeffs', None),
                'glass_before': s.glass_before,
                'glass_after': s.glass_after,
                'radius_y': getattr(s, 'radius_y', None),
                'conic_y': getattr(s, 'conic_y', None),
                'aspheric_coeffs_y': getattr(s, 'aspheric_coeffs_y', None),
            }
            rx_surfaces.append(surf_dict)
            if i < len(surfaces) - 1 and not surfaces[i + 1].is_coordbrk:
                thicknesses.append(s.thickness)
            elif i < len(surfaces) - 1:
                # Skip the cb that follows; its 0-thickness is
                # absorbed into the next refractive surface's gap.
                thicknesses.append(s.thickness)

        # 3.7.0: full element list + coord_breaks for round-trip.
        # Walk self.elements directly so we capture the GUI's
        # element-level metadata (mirrors, names, comments, tilts).
        # Emit one synthetic cb per tilted element, with surf_num
        # set so an importer's ``next-following surf > cb_n``
        # heuristic re-attaches it to the same element.
        elements_out = []
        all_thicknesses = []
        coord_breaks_out = []
        running_surf_num = 0   # Zemax-like surface counter

        for ei, elem in enumerate(self.elements):
            if elem.elem_type in ('Source', 'Detector'):
                continue
            # Emit a coord_break BEFORE this element's surfaces if
            # it carries any tilt/decenter.  surf_num = previous
            # rendered surf so the cb sits between elements.
            has_tilt = any(float(getattr(elem, f, 0.0)) != 0.0
                            for f in ('tilt_x', 'tilt_y',
                                       'decenter_x', 'decenter_y'))
            if has_tilt:
                coord_breaks_out.append({
                    'surf_num': running_surf_num,
                    'tilt_x_deg': float(getattr(elem, 'tilt_x', 0.0)),
                    'tilt_y_deg': float(getattr(elem, 'tilt_y', 0.0)),
                    'tilt_z_deg': 0.0,
                    'decenter_x_m': float(getattr(elem, 'decenter_x', 0.0))
                                     * 1e-3,
                    'decenter_y_m': float(getattr(elem, 'decenter_y', 0.0))
                                     * 1e-3,
                    'order': 0,
                    'thickness_m': 0.0,
                })
                running_surf_num += 1
            # Emit each surface within the element.
            for si, srow in enumerate(elem.surfaces):
                running_surf_num += 1
                R_m = (srow.radius * 1e-3
                       if np.isfinite(srow.radius) else float('inf'))
                sd_m = (srow.semi_diameter * 1e-3
                        if np.isfinite(srow.semi_diameter)
                        else float('inf'))
                if srow.surf_type == 'Mirror' \
                        or elem.elem_type == 'Mirror':
                    elements_out.append({
                        'element_type': 'mirror',
                        'radius': R_m,
                        'conic': srow.conic,
                        'aspheric_coeffs':
                            getattr(srow, 'aspheric_coeffs', None),
                        'semi_diameter': sd_m,
                        'surf_num': running_surf_num,
                        'comment': elem.name if elem.name else '',
                    })
                else:
                    if si == 0:
                        glass_before = 'air'
                    else:
                        prev_glass = elem.surfaces[si - 1].glass
                        glass_before = prev_glass if prev_glass else 'air'
                    glass_after = srow.glass if srow.glass else 'air'
                    elements_out.append({
                        'element_type': 'surface',
                        'radius': R_m,
                        'conic': srow.conic,
                        'aspheric_coeffs':
                            getattr(srow, 'aspheric_coeffs', None),
                        'glass_before': glass_before,
                        'glass_after': glass_after,
                        'semi_diameter': sd_m,
                        'surf_num': running_surf_num,
                        'comment': elem.name
                            if (si == 0 and elem.name
                                and elem.elem_type != 'Singlet'
                                and not elem.name.startswith('Lens '))
                            else '',
                    })
                # Thickness from this surface to the next item (the
                # next surface OR the next element's leading cb /
                # first surface).
                if si < len(elem.surfaces) - 1:
                    all_thicknesses.append(srow.thickness * 1e-3)
                else:
                    # Last surface in the element -- thickness goes
                    # to the next element's cb (if tilted) or front
                    # vertex.  Walk forward to the next non-source/
                    # detector element.
                    next_elem = None
                    for ej in range(ei + 1, len(self.elements)):
                        if self.elements[ej].elem_type in (
                                'Source', 'Detector'):
                            if (self.elements[ej].elem_type ==
                                    'Detector'):
                                next_elem = self.elements[ej]
                                break
                            continue
                        next_elem = self.elements[ej]
                        break
                    if next_elem is not None:
                        all_thicknesses.append(
                            next_elem.distance_mm * 1e-3)
                    else:
                        all_thicknesses.append(0.0)

        return {
            'name': 'User design',
            'aperture_diameter': self.epd_m,
            # Legacy lens-only keys (mirrors / cb's filtered out).
            'surfaces': rx_surfaces,
            'thicknesses': thicknesses,
            # 3.7.0: full chronological list + coord_breaks for
            # round-trip preservation of fold geometry.
            'elements': elements_out,
            'all_thicknesses': all_thicknesses,
            'coord_breaks': coord_breaks_out,
        }


def _nice_dx(dx_m):
    """Round dx down to a clean value (1, 2, 5 sequence in each decade)."""
    if dx_m <= 0:
        return 1e-6
    exp = np.floor(np.log10(dx_m))
    mantissa = dx_m / 10**exp
    if mantissa >= 5:
        return 5 * 10**exp
    elif mantissa >= 2:
        return 2 * 10**exp
    else:
        return 1 * 10**exp


def _next_nice_N(n):
    """Round up to next N that is 2^a, 2^a*3, or 2^a*5."""
    candidates = []
    for a in range(7, 16):  # 128 to 32768
        base = 2 ** a
        for mult in (1, 3, 5):
            v = base * mult
            if v >= n:
                candidates.append(v)
    return min(candidates) if candidates else 2 ** int(np.ceil(np.log2(max(n, 128))))
