"""
Vector Optics / Polarization Module
====================================

Jones-vector representation for polarized optical fields. Extends the
scalar field functions to 2-component (Ex, Ey) vector fields.

Main class: :class:`JonesField` — wraps two scalar fields and provides
propagation, optical element application, and polarization analysis.

Polarization elements provided:
    - apply_polarizer
    - apply_waveplate (HWP, QWP, arbitrary retardance)
    - apply_jones_matrix (arbitrary 2×2 transformation)
    - apply_rotator

Sources:
    - create_linear_polarized
    - create_circular_polarized
    - create_elliptical_polarized

Analysis:
    - stokes_parameters
    - degree_of_polarization
    - polarization_ellipse

Convention: Jones vectors written as column vectors [Ex, Ey]^T.
Angles measured from +x axis, counter-clockwise.

All scalar propagators and optical elements can be applied to a
JonesField by calling its methods, which dispatch to the underlying
scalar functions for each component independently (Ex and Ey propagate
or transform identically through non-polarizing elements).

Author: Andrew Traverso
"""

import numpy as np

# Scalar propagation and element functions (dispatched per component)
from .propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
    angular_spectrum_propagate_batch,
    fresnel_propagate,
    fraunhofer_propagate,
    scalable_angular_spectrum_propagate,
)
from .lenses import (
    apply_thin_lens,
    apply_spherical_lens,
    apply_real_lens,
)
from .elements import (
    apply_mirror,
    apply_aperture,
    apply_mask,
)


# =============================================================================
# JONES FIELD CLASS
# =============================================================================

class JonesField:
    """
    A two-component (Ex, Ey) complex field representing a polarized
    optical beam on a discrete 2D grid.

    Attributes
    ----------
    Ex : ndarray (complex, N×N)
        Electric field component along x.
    Ey : ndarray (complex, N×N)
        Electric field component along y.
    dx : float
        Grid spacing in x [m].
    dy : float
        Grid spacing in y [m]. Defaults to dx.

    Notes
    -----
    Non-polarizing optical elements (lenses, apertures, free-space
    propagation) act identically on Ex and Ey. The corresponding methods
    on :class:`JonesField` simply dispatch the scalar functions to each
    component.

    Polarization-dependent elements (polarizers, waveplates, Jones
    matrices) mix Ex and Ey according to their 2×2 Jones matrix.
    """

    def __init__(self, Ex, Ey, dx, dy=None):
        if Ex.shape != Ey.shape:
            raise ValueError(f"Ex and Ey must have the same shape, "
                             f"got {Ex.shape} and {Ey.shape}")
        self.Ex = np.asarray(Ex, dtype=complex)
        self.Ey = np.asarray(Ey, dtype=complex)
        self.dx = dx
        self.dy = dy if dy is not None else dx

    @property
    def shape(self):
        return self.Ex.shape

    def intensity(self):
        """Total intensity |Ex|^2 + |Ey|^2."""
        return np.abs(self.Ex)**2 + np.abs(self.Ey)**2

    def power(self):
        """Total integrated power."""
        return float(np.sum(self.intensity()) * self.dx * self.dy)

    def copy(self):
        """Return a deep copy of the field."""
        return JonesField(self.Ex.copy(), self.Ey.copy(), self.dx, self.dy)

    # ------------------------------------------------------------------
    # Propagation (scalar functions applied to each component)
    # ------------------------------------------------------------------

    # Below this grid size the batched-FFT dispatch overhead exceeds
    # the savings from sharing an H build, so JonesField.propagate
    # falls back to two sequential ASM calls (the second hits the H
    # cache too, so it's essentially free).  Empirically determined
    # on a 6-core CPU; adjust via :func:`set_jones_batch_threshold`.
    _BATCH_PROPAGATE_MIN_N = 512

    def propagate(self, z, wavelength, bandlimit=True):
        """Propagate via the angular spectrum method.

        For grids at or above ``_BATCH_PROPAGATE_MIN_N`` (default 512)
        uses :func:`angular_spectrum_propagate_batch` so Ex/Ey share a
        single H build + one batched FFT pair.  Below that the
        sequential path is faster because FFT-call dispatch dominates
        compute -- both components hit the H cache on the second call
        anyway.
        """
        same_layout = (self.Ex.shape == self.Ey.shape
                       and self.Ex.dtype == self.Ey.dtype)
        N_max = max(self.Ex.shape) if self.Ex.ndim >= 2 else 0
        if same_layout and N_max >= self._BATCH_PROPAGATE_MIN_N:
            stack = np.stack([self.Ex, self.Ey], axis=0)
            out = angular_spectrum_propagate_batch(
                stack, z, wavelength, self.dx, self.dy,
                bandlimit=bandlimit)
            self.Ex, self.Ey = out[0], out[1]
        else:
            self.Ex = angular_spectrum_propagate(
                self.Ex, z, wavelength, self.dx, self.dy,
                bandlimit=bandlimit)
            self.Ey = angular_spectrum_propagate(
                self.Ey, z, wavelength, self.dx, self.dy,
                bandlimit=bandlimit)
        return self

    def propagate_tilted(self, z, wavelength, tilt_x=0, tilt_y=0, bandlimit=True):
        """Propagate via off-axis ASM."""
        self.Ex = angular_spectrum_propagate_tilted(
            self.Ex, z, wavelength, self.dx, self.dy,
            tilt_x=tilt_x, tilt_y=tilt_y, bandlimit=bandlimit)
        self.Ey = angular_spectrum_propagate_tilted(
            self.Ey, z, wavelength, self.dx, self.dy,
            tilt_x=tilt_x, tilt_y=tilt_y, bandlimit=bandlimit)
        return self

    def propagate_fresnel(self, z, wavelength):
        """Propagate via single-FFT Fresnel. Returns new grid spacings."""
        self.Ex, dx_out, dy_out = fresnel_propagate(
            self.Ex, z, wavelength, self.dx, self.dy)
        self.Ey, _, _ = fresnel_propagate(
            self.Ey, z, wavelength, self.dx, self.dy)
        self.dx = dx_out
        self.dy = dy_out
        return self

    def propagate_fraunhofer(self, z, wavelength):
        """Propagate to the far-field via Fraunhofer. Returns new grid spacings."""
        self.Ex, dx_out, dy_out = fraunhofer_propagate(
            self.Ex, z, wavelength, self.dx, self.dy)
        self.Ey, _, _ = fraunhofer_propagate(
            self.Ey, z, wavelength, self.dx, self.dy)
        self.dx = dx_out
        self.dy = dy_out
        return self

    def sas_propagate(self, z, wavelength, pad=2,
                      skip_final_phase=False):
        """Propagate via the Scalable Angular Spectrum Method.

        Applies :func:`scalable_angular_spectrum_propagate` to ``Ex``
        and ``Ey`` independently.  Both components share the same
        output grid (same ``z``, ``wavelength``, ``dx``, ``pad``), so
        this is the polarization-aware analogue of a scalar SAS call.
        The field's ``dx`` / ``dy`` are updated to the new output pitch
        (``lambda * z / (pad * N * dx)``) so downstream element calls
        see the correct coordinates.

        Parameters
        ----------
        z : float
            Propagation distance [m].
        wavelength : float
            Wavelength [m].
        pad : int, default 2
            Zero-padding factor for the SAS kernel.
        skip_final_phase : bool, default False
            Skip the outer quadratic phase; intensity is correct but
            field phase is not.

        Returns
        -------
        self
            Modified in place (same convention as the other propagate
            methods on this class).
        """
        if self.dx != self.dy:
            raise ValueError(
                "JonesField.sas_propagate requires dx == dy "
                f"(got dx={self.dx}, dy={self.dy}).  SAS assumes a "
                f"square grid pitch.")
        self.Ex, dx_out, dy_out = scalable_angular_spectrum_propagate(
            self.Ex, z, wavelength, self.dx,
            pad=pad, skip_final_phase=skip_final_phase)
        self.Ey, _, _ = scalable_angular_spectrum_propagate(
            self.Ey, z, wavelength, self.dx,
            pad=pad, skip_final_phase=skip_final_phase)
        self.dx = dx_out
        self.dy = dy_out
        return self

    # ------------------------------------------------------------------
    # Non-polarizing elements (dispatched per component)
    # ------------------------------------------------------------------

    def apply_thin_lens(self, f, wavelength, **kwargs):
        """Apply a thin lens to both components."""
        self.Ex = apply_thin_lens(self.Ex, f, wavelength, self.dx, self.dy, **kwargs)
        self.Ey = apply_thin_lens(self.Ey, f, wavelength, self.dx, self.dy, **kwargs)
        return self

    def apply_spherical_lens(self, **kwargs):
        self.Ex = apply_spherical_lens(self.Ex, wavelength=kwargs['wavelength'],
                                        dx=self.dx, dy=self.dy, **{k: v for k, v in kwargs.items() if k != 'wavelength'})
        self.Ey = apply_spherical_lens(self.Ey, wavelength=kwargs['wavelength'],
                                        dx=self.dx, dy=self.dy, **{k: v for k, v in kwargs.items() if k != 'wavelength'})
        return self

    def apply_real_lens(self, prescription, wavelength, bandlimit=True,
                        slant_correction=False, fresnel=False, absorption=False):
        """Apply a multi-surface real lens to both components.

        See :func:`lumenairy.lenses.apply_real_lens` for parameter
        documentation.
        """
        self.Ex = apply_real_lens(
            self.Ex, prescription, wavelength, self.dx,
            bandlimit=bandlimit, slant_correction=slant_correction,
            fresnel=fresnel, absorption=absorption)
        self.Ey = apply_real_lens(
            self.Ey, prescription, wavelength, self.dx,
            bandlimit=bandlimit, slant_correction=slant_correction,
            fresnel=fresnel, absorption=absorption)
        return self

    def apply_mirror(self, wavelength, **kwargs):
        self.Ex = apply_mirror(self.Ex, wavelength, self.dx, **kwargs)
        self.Ey = apply_mirror(self.Ey, wavelength, self.dx, **kwargs)
        return self

    def apply_aperture(self, shape='circular', params=None, xc=0, yc=0):
        self.Ex = apply_aperture(self.Ex, self.dx, shape=shape, params=params, xc=xc, yc=yc)
        self.Ey = apply_aperture(self.Ey, self.dx, shape=shape, params=params, xc=xc, yc=yc)
        return self

    def apply_mask(self, mask):
        self.Ex = apply_mask(self.Ex, mask)
        self.Ey = apply_mask(self.Ey, mask)
        return self


# =============================================================================
# POLARIZATION-DEPENDENT ELEMENTS
# =============================================================================

def apply_jones_matrix(field, matrix):
    """
    Apply an arbitrary 2×2 Jones matrix to a JonesField.

    The transformation is:

        [Ex']   [J00 J01] [Ex]
        [Ey'] = [J10 J11] [Ey]

    Parameters
    ----------
    field : JonesField
        Input polarized field.
    matrix : ndarray (complex, 2×2) or callable
        The Jones matrix. If callable, must accept (x, y) meshgrids and
        return a (2, 2, N, N) array for spatially-varying elements.

    Returns
    -------
    JonesField
        Transformed field (modified in place and returned).
    """
    if callable(matrix):
        x = (np.arange(field.shape[1]) - field.shape[1] / 2) * field.dx
        y = (np.arange(field.shape[0]) - field.shape[0] / 2) * field.dy
        X, Y = np.meshgrid(x, y)
        J = matrix(X, Y)  # expect (2, 2, N, N)
        Ex_new = J[0, 0] * field.Ex + J[0, 1] * field.Ey
        Ey_new = J[1, 0] * field.Ex + J[1, 1] * field.Ey
    else:
        J = np.asarray(matrix, dtype=complex)
        Ex_new = J[0, 0] * field.Ex + J[0, 1] * field.Ey
        Ey_new = J[1, 0] * field.Ex + J[1, 1] * field.Ey

    field.Ex = Ex_new
    field.Ey = Ey_new
    return field


def apply_polarizer(field, angle=0.0):
    """
    Apply an ideal linear polarizer at the specified transmission angle.

    Parameters
    ----------
    field : JonesField
        Input field.
    angle : float, default 0
        Transmission axis angle [radians], measured from +x axis.

    Returns
    -------
    JonesField
    """
    c = np.cos(angle)
    s = np.sin(angle)
    # Projection matrix for linear polarizer
    J = np.array([[c*c,  c*s],
                  [c*s,  s*s]], dtype=complex)
    return apply_jones_matrix(field, J)


def apply_waveplate(field, retardance, angle=0.0):
    """
    Apply a waveplate (linear retarder) with arbitrary retardance.

    Parameters
    ----------
    field : JonesField
        Input field.
    retardance : float
        Phase retardance between fast and slow axes [radians].
        - pi/2 = quarter-wave plate
        - pi   = half-wave plate
    angle : float, default 0
        Fast-axis angle [radians], measured from +x axis.

    Returns
    -------
    JonesField

    Notes
    -----
    The Jones matrix for a waveplate with fast axis at angle theta and
    retardance phi is::

        J = R(-theta) * diag(1, exp(i*phi)) * R(theta)

    where R is the 2D rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)
    e = np.exp(1j * retardance)

    # R(-theta) * diag(1, e) * R(theta)
    J = np.array([
        [c*c + e*s*s,     c*s*(1 - e)],
        [c*s*(1 - e),     s*s + e*c*c],
    ], dtype=complex)
    return apply_jones_matrix(field, J)


def apply_half_wave_plate(field, angle=0.0):
    """Convenience wrapper: half-wave plate (retardance = pi)."""
    return apply_waveplate(field, np.pi, angle)


def apply_quarter_wave_plate(field, angle=0.0):
    """Convenience wrapper: quarter-wave plate (retardance = pi/2)."""
    return apply_waveplate(field, np.pi / 2, angle)


def apply_rotator(field, angle):
    """
    Apply a polarization rotator (e.g. Faraday rotator).

    Rotates the polarization state by the given angle without changing
    its ellipticity.

    Parameters
    ----------
    field : JonesField
        Input field.
    angle : float
        Rotation angle [radians].

    Returns
    -------
    JonesField
    """
    c = np.cos(angle)
    s = np.sin(angle)
    J = np.array([[c, -s],
                  [s,  c]], dtype=complex)
    return apply_jones_matrix(field, J)


# =============================================================================
# POLARIZED SOURCES
# =============================================================================

def create_linear_polarized(scalar_field, dx, angle=0.0, dy=None):
    """
    Create a linearly polarized JonesField from a scalar field.

    Parameters
    ----------
    scalar_field : ndarray (complex, N×N)
        Scalar amplitude distribution (e.g. from create_gaussian_beam).
    dx : float
        Grid spacing [m].
    angle : float, default 0
        Polarization angle [radians] from +x axis.
        0 = x-polarized, pi/2 = y-polarized, pi/4 = diagonal.
    dy : float, optional

    Returns
    -------
    JonesField
    """
    Ex = scalar_field * np.cos(angle)
    Ey = scalar_field * np.sin(angle)
    return JonesField(Ex, Ey, dx, dy)


def create_circular_polarized(scalar_field, dx, handedness='right', dy=None):
    """
    Create a circularly polarized JonesField from a scalar field.

    Parameters
    ----------
    scalar_field : ndarray (complex, N×N)
        Scalar amplitude distribution.
    dx : float
        Grid spacing [m].
    handedness : {'right', 'left'}, default 'right'
        Handedness of the circular polarization.
        - 'right': Ey leads Ex by pi/2 (RHC in optics convention)
        - 'left':  Ey lags  Ex by pi/2 (LHC)
    dy : float, optional

    Returns
    -------
    JonesField
    """
    Ex = scalar_field / np.sqrt(2)
    if handedness.lower().startswith('r'):
        Ey = scalar_field * 1j / np.sqrt(2)
    else:
        Ey = scalar_field * (-1j) / np.sqrt(2)
    return JonesField(Ex, Ey, dx, dy)


def create_elliptical_polarized(scalar_field, dx, ellipticity=0.0, orientation=0.0, dy=None):
    """
    Create an elliptically polarized JonesField from a scalar field.

    Parameters
    ----------
    scalar_field : ndarray (complex, N×N)
        Scalar amplitude distribution.
    dx : float
        Grid spacing [m].
    ellipticity : float, default 0
        Ellipticity angle chi [radians]. 0 = linear, ±pi/4 = circular.
    orientation : float, default 0
        Major-axis angle psi [radians] from +x axis.
    dy : float, optional

    Returns
    -------
    JonesField

    Notes
    -----
    The Jones vector for an elliptical polarization is::

        [Ex]   [cos(psi)  -sin(psi)] [cos(chi)]
        [Ey] = [sin(psi)   cos(psi)] [i sin(chi)]
    """
    cp = np.cos(orientation)
    sp = np.sin(orientation)
    cc = np.cos(ellipticity)
    sc = np.sin(ellipticity)
    Ex = scalar_field * (cp * cc - 1j * sp * sc)
    Ey = scalar_field * (sp * cc + 1j * cp * sc)
    return JonesField(Ex, Ey, dx, dy)


# =============================================================================
# POLARIZATION ANALYSIS
# =============================================================================

def stokes_parameters(field):
    """
    Compute the Stokes parameters (S0, S1, S2, S3) of a JonesField.

    Parameters
    ----------
    field : JonesField

    Returns
    -------
    S : dict
        Dictionary with keys 'S0', 'S1', 'S2', 'S3', each a 2D array.
        S0 = |Ex|^2 + |Ey|^2         (total intensity)
        S1 = |Ex|^2 - |Ey|^2         (horizontal vs vertical)
        S2 = 2*Re(Ex * conj(Ey))     (±45 deg linear)
        S3 = -2*Im(Ex * conj(Ey))    (circular, right-hand positive)
    """
    Ex = field.Ex
    Ey = field.Ey
    S0 = np.abs(Ex)**2 + np.abs(Ey)**2
    S1 = np.abs(Ex)**2 - np.abs(Ey)**2
    S2 = 2 * np.real(Ex * np.conj(Ey))
    S3 = -2 * np.imag(Ex * np.conj(Ey))
    return {'S0': S0, 'S1': S1, 'S2': S2, 'S3': S3}


def degree_of_polarization(field):
    """
    Compute the degree of polarization (DOP).

    DOP = sqrt(S1^2 + S2^2 + S3^2) / S0

    For fully coherent fields from a single source, DOP = 1 everywhere
    (where S0 > 0). Values less than 1 indicate depolarization, which
    occurs only for partially coherent / incoherent sources or through
    depolarizing elements.

    Parameters
    ----------
    field : JonesField

    Returns
    -------
    dop : ndarray (real, N×N)
        Local degree of polarization (0 to 1).
    """
    S = stokes_parameters(field)
    total = np.sqrt(S['S1']**2 + S['S2']**2 + S['S3']**2)
    # Avoid division by zero
    dop = np.where(S['S0'] > 1e-30, total / np.maximum(S['S0'], 1e-30), 0.0)
    return dop


def polarization_ellipse(field):
    """
    Compute the polarization ellipse parameters (orientation, ellipticity)
    at each grid point.

    Parameters
    ----------
    field : JonesField

    Returns
    -------
    orientation : ndarray (real, N×N)
        Orientation angle psi of the polarization ellipse major axis
        [radians], from +x axis. Range: [-pi/2, pi/2].
    ellipticity : ndarray (real, N×N)
        Ellipticity angle chi [radians]. Range: [-pi/4, pi/4].
        - chi = 0: linear polarization
        - chi = ±pi/4: circular polarization

    Notes
    -----
    Relationship to Stokes parameters::

        tan(2*psi) = S2 / S1
        sin(2*chi) = S3 / sqrt(S1^2 + S2^2 + S3^2)
    """
    S = stokes_parameters(field)
    orientation = 0.5 * np.arctan2(S['S2'], S['S1'])
    total = np.sqrt(S['S1']**2 + S['S2']**2 + S['S3']**2)
    sin_2chi = np.where(total > 1e-30, S['S3'] / np.maximum(total, 1e-30), 0.0)
    sin_2chi = np.clip(sin_2chi, -1.0, 1.0)
    ellipticity = 0.5 * np.arcsin(sin_2chi)
    return orientation, ellipticity
