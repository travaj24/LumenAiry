"""
lumenairy.propagators.carrier_field -- first-class carrier-referenced fields
and the analytic re-reference / aggregation primitives they compose with.

WHAT THIS IS FOR
----------------
``propagate_traced_carrier_chain`` runs ONE ray congruence at a time: its
entrance->exit map is single-congruence by construction, so a fan of DOE
orders is a fan of chains.  To do anything COHERENT across those chains --
sum them, measure their crosstalk, hand them to a downstream leg once instead
of K times -- the per-order fields have to be brought onto ONE grid against
ONE analytic carrier first.  That operation is what this module packages.

It productizes the arm-B code path of
``docs/audits/PROBE_SUM_AT_APERTURE_2026_08_11.md`` (and its reference
scripts ``validation/repro_traced_carrier_121/sumap_*.py``), which measured
the physics half of the architecture and passed every bar the design-121
campaign uses: null control 2.8e-05 field relative L2 against the shipped
per-order path, 1e-07 on energy, 0.000 EE points, FWHM identical, piston
below 2.3e-06 rad, linearity 2.3e-16.

THREE THINGS THE PROBE PROVED THAT THIS MODULE ENCODES
------------------------------------------------------
1. **The only summable plane is the fine-retrace exit.**  A traced group's
   coarse co-moving exit plane cannot carry its own exit congruence (design
   121: it lands at dx = 33.2 um where the exit sphere needs 4.26 um -- 7.8x
   under-sampled), and a plane BEFORE the last group cannot be traced at all.
   Nothing here can rescue a field sampled at the wrong plane; the caller
   supplies the field, and :func:`re_reference` only refuses when the TARGET
   grid is the thing that cannot hold the answer.

2. **The binding sampling term is the CARRIER-OFFSET RAMP, not the chief-ray
   tilt.**  At a shared back aperture the orders are separated SPATIALLY, not
   angularly: every exit chief ray is parallel to the axis to within 0.7 mrad
   while the same converging sphere sits up to 2.14 mm off centre.  Sizing
   the common grid from the literal tilt spread (7.68e-04 rad -> dx <= 852 um)
   is 400x too coarse.  What a common carrier actually leaves behind is
   ``|c_k - c_0| / |R|`` = 0.2776 rad of residual ramp.  See
   :func:`carrier_difference_nyquist`, which COMPUTES that from the two
   carriers rather than taking it on faith.

3. **The ramp and the beam's own angular band do NOT add.**  Adding them
   costs a factor of 4 in grid and is the natural mistake: the ramp is a
   DIFFERENCE between two spheres, while a beam's absolute angular band is
   set by how far its own sphere is sampled from its OWN centre and does not
   move with the chief ray.  The binding pitch is therefore the ``min`` of
   the two bounds, not ``lambda/(2*(ramp + NA))``.  Measured on design 121:
   the probe ran the whole aggregation at dx = 1.2292 um, which is 1.5x
   INSIDE ``lambda/(2*(ramp + NA))``; if that were the operative bound the
   extreme order would have aliased and moved.  It did not (probe S8.1: two
   pitches 2x apart agree to every printed digit).

PISTON IS EXPLICIT, AND THAT IS THE POINT
-----------------------------------------
:class:`CarrierSpec` carries a ``piston`` term -- a CONSTANT optical path in
METRES -- alongside the sphere and the tilt.  It exists so this module
composes with ``docs/audits/FIX_TILT_QUADRATIC_OPL_2026_08_11.md``, which
makes ``apply_real_lens_traced`` (and hence every chain) return an ABSOLUTE
optical path instead of one referenced to nothing.  Once the element hands
back an absolute path, the inter-order relative phase is a real quantity
(measured there at 0.0042 waves across design 121's whole 51 mrad fan), and a
coherent sum is admissible.  A primitive that quietly dropped the constant
would put the defect straight back.

Two consequences the implementation takes seriously:

* the piston is applied as a UNIT PHASOR and is never added into an eikonal
  array.  ``k0 * piston`` is ~5e+04 rad on a design-121 group while the
  interesting variation of an eikonal map is ~1e-03 rad; adding them would
  spend the mantissa on the constant (this is the same argument
  ``_lens_traced.py`` makes for its own ``_opl_piston_phasor``);
* the piston DIFFERENCE that :func:`re_reference` folds into the envelope is
  reduced to its fractional part IN THE OPL DOMAIN before it is exponentiated
  (``dp - round(dp/lambda)*lambda``, a Sterbenz-exact subtraction), so the
  phase it applies is accurate to ``2*pi*eps*|dp|/lambda`` rather than to
  ``eps*k0*|dp|``.  See :func:`_piston_phase`.

GRID CONVENTION
---------------
Every grid here is the library's own centred lattice --
``x_grid[i] = (i - Nx/2) * dx`` -- PLUS an explicit ``origin``, the ABSOLUTE
transverse position of the grid centre.  So

.. code-block:: text

    x_absolute[i] = (i - Nx/2) * dx + origin[0]

and a carrier's ``centre`` (its chief ray) is likewise ABSOLUTE.  This is the
probe's ROI convention: its per-order retrace grids differ in pitch (1.5324
vs 1.5243 um) AND in origin (0.000 / -1.508 / -3.016 mm), and both have to be
carried for the orders to land on one lattice.  The library's private eikonal
builders take a grid-frame ``centre``, so everything here converts once, at
the call site, via ``centre - origin``.

Author: Andrew Traverso
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .carrier import (
    _check_guard_action,
    _envelope_amp_radius,
    _exact_sphere_eikonal,
    _exact_tilt_reference,
    _guard_dispose,
    _tilt_exactness_phase,
    _tilt_ramp,
)
from .mft import angular_spectrum_propagate_mft

__all__ = [
    'CarrierSpec',
    'FieldGrid',
    'CarrierField',
    'NyquistReport',
    'ReReferenceReport',
    'FieldLedgerRow',
    'AggregateLedger',
    'AggregateResult',
    'carrier_difference_nyquist',
    're_reference',
    'aggregate',
    'save_carrier_field_zarr',
    'load_carrier_field_zarr',
    'CARRIER_FIELD_SCHEMA',
]

#: Storage schema version of the Zarr layout written by
#: :func:`save_carrier_field_zarr`.  Bumped whenever the on-disk group
#: structure or the attribute vocabulary changes in a way a reader written
#: against the previous version could mis-interpret.  The loader REFUSES an
#: unknown version rather than guessing.
CARRIER_FIELD_SCHEMA = 1

#: Default enclosed-power fraction used to size a field's SUPPORT radius --
#: the radius about the chief ray inside which this fraction of the power
#: lies.  Every Nyquist bound and every containment margin in this module is
#: measured over that disc, so the number is a policy and is named.
#:
#: 0.99999 is the probe's own choice (PROBE_SUM_AT_APERTURE S3): on design
#: 121's back aperture it reads 2.64 mm, against a 1/e amplitude radius of
#: 1.185 mm -- i.e. 2.23 w.  The bound it feeds, ``r/sqrt(r^2+R^2)`` = 0.3239
#: -> dx <= 2.0225 um, is the pitch that actually governed that geometry.
SUPPORT_POWER_FRACTION = 0.99999

#: Rows per chunk in the memory-bounded radial reductions.  A design-121
#: aperture grid is 8192^2 complex128 = 1.07 GB; a whole-grid ``hypot`` +
#: ``argsort`` on top of it is not affordable, and the probe's own peak
#: working set (17.5 GB) is already the constraint that decides whether the
#: architecture fits in memory at all.
_RADIAL_CHUNK_ROWS = 256

#: Polar sampling used by :func:`carrier_difference_nyquist` to maximise the
#: carrier-difference slope over a support disc.  The integrand is smooth and
#: the maximum is usually on the boundary, but "usually" is not a proof, so
#: the whole disc is swept.  32 x 256 = 8192 evaluations of a closed form.
_NYQUIST_N_RADII = 32
_NYQUIST_N_ANGLES = 256


# ---------------------------------------------------------------------------
# JSON-safe scalar encoding (Zarr attrs, provenance canonicalisation)
# ---------------------------------------------------------------------------
def _enc_float(v):
    """JSON-safe encoding of a float that may be non-finite.

    A carrier radius of ``+/-inf`` is the LIBRARY'S OWN spelling of a
    collimated congruence (``_exact_sphere_eikonal`` returns the plane-wave
    eikonal there), so it has to survive a round trip through storage
    exactly.  Python's ``json`` writes bare ``Infinity`` / ``NaN``, which are
    not JSON and which a non-Python reader of a Zarr store may reject, so
    non-finite values are written as their ``repr`` STRING and finite ones as
    plain floats (whose ``repr`` round-trips bit-exactly in float64)."""
    v = float(v)
    return v if math.isfinite(v) else repr(v)


def _dec_float(v):
    """Inverse of :func:`_enc_float`."""
    if isinstance(v, str):
        return float(v)
    return float(v)


def _canonical_provenance(prov):
    """Round a provenance mapping through JSON so the in-memory object is a
    FIXED POINT of save/load.

    Without this, ``load(save(f)).provenance == f.provenance`` is false for
    any provenance holding a tuple or a non-string key -- JSON has neither --
    and the round-trip test would be asserting a weaker claim than it looks
    like it is asserting.  Canonicalising at CONSTRUCTION means the object a
    caller holds is already the object storage can return."""
    if prov is None:
        return {}
    try:
        blob = json.dumps(prov, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"CarrierField: provenance must be JSON-serialisable with finite "
            f"numbers only (it is written verbatim into the Zarr attrs and is "
            f"the object's identity for the round-trip contract), but "
            f"json.dumps refused it: {exc}.  Encode non-finite floats as "
            f"strings (see _enc_float) or as a nested dict of primitives."
        ) from exc
    return json.loads(blob)


# ---------------------------------------------------------------------------
# The carrier
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CarrierSpec:
    """One analytic carrier: a sphere, a chief ray, a tilt and a PISTON.

    The eikonal this describes, in ABSOLUTE transverse coordinates and with
    ``u = x - centre[0]``, ``v = y - centre[1]``,
    ``n = sqrt(1 - L^2 - M^2)``:

    .. code-block:: text

        C(x, y) = W(u, v) + piston

        W = sign(R) ( sqrt((u + R L/n)^2 + (v + R M/n)^2 + R^2) - |R|/n )

    -- the exact eikonal of a point source at signed AXIAL distance ``R``
    whose chief ray reaches ``centre`` along ``(L, M, n)``.  That is the
    niche-C5 EXACT tilted-congruence reference, and it is what the library's
    ``_exact_sphere_eikonal`` x ``_tilt_ramp`` x ``_tilt_exactness_phase``
    trio builds: the first two give the sphere-plus-ramp truncation
    ``S + (Lu + Mv)`` and the third supplies exactly the difference
    ``D = W - S - (Lu + Mv)``.  When the C5 switch
    (``_lens_traced.TILTED_CARRIER_EXACT_EIKONAL``) is off, or the tilt is
    identically zero, the carrier IS ``S + (Lu + Mv)`` and this class follows
    it -- see :meth:`gradient`, which reads the same switch, so the Nyquist
    arithmetic can never be computed against a reference the phasor does not
    use.

    Attributes
    ----------
    R : float
        Signed radius of the spherical wavefront (m).  ``R < 0`` converging.
        ``+/-inf`` is COLLIMATED and is a plane wave (``S == 0``), not a
        degenerate sphere.
    centre : (float, float)
        Chief-ray transverse position, ABSOLUTE (m) -- not relative to
        whatever grid the carrier is later evaluated on.
    tilt : (float, float)
        Chief-ray direction cosines ``(L, M)``.  DIRECTION COSINES, not
        slopes: ``L^2 + M^2 < 1`` is required.
    piston : float
        Constant optical path (m), EXPLICIT.  This is the term
        ``FIX_TILT_QUADRATIC_OPL_2026_08_11`` restores to
        ``apply_real_lens_traced``'s exit field, and carrying it here is what
        lets a field re-referenced onto another carrier keep a meaningful
        ABSOLUTE optical path.  Contributes ``exp(i k0 piston)`` -- a global
        unit phasor, intensity-blind by construction.
    """

    R: float
    centre: Tuple[float, float] = (0.0, 0.0)
    tilt: Tuple[float, float] = (0.0, 0.0)
    piston: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, 'R', float(self.R))
        object.__setattr__(self, 'centre', (float(self.centre[0]),
                                            float(self.centre[1])))
        object.__setattr__(self, 'tilt', (float(self.tilt[0]),
                                          float(self.tilt[1])))
        object.__setattr__(self, 'piston', float(self.piston))
        L, M = self.tilt
        if not math.isfinite(L) or not math.isfinite(M):
            raise ValueError(
                f"CarrierSpec: tilt must be finite direction cosines, got "
                f"(L={L!r}, M={M!r}).")
        if L * L + M * M >= 1.0:
            raise ValueError(
                f"CarrierSpec: the tilt (L={L!r}, M={M!r}) has L^2 + M^2 = "
                f"{L * L + M * M:.6g} >= 1, i.e. it is not a propagating "
                f"direction.  L and M are DIRECTION COSINES, not slopes.")
        if not math.isfinite(self.piston):
            raise ValueError(
                f"CarrierSpec: piston must be a finite optical path in "
                f"metres, got {self.piston!r}.")

    # -- geometry ---------------------------------------------------------
    @property
    def is_collimated(self) -> bool:
        return not math.isfinite(self.R)

    def _uses_exact_tilt(self) -> bool:
        """Is the niche-C5 exactness term live for THIS carrier?

        Mirrors ``_tilt_exactness_phase``'s own early returns exactly, so the
        analytic gradient below describes the phasor that is actually built
        and never a different reference."""
        L, M = self.tilt
        if not _exact_tilt_reference():
            return False
        if (L == 0.0 and M == 0.0) or not math.isfinite(self.R) \
                or self.R == 0.0:
            return False
        return True

    def eikonal_at(self, x, y):
        """Optical path ``C(x, y)`` in METRES at ABSOLUTE coordinates,
        INCLUDING the piston.  Accepts scalars or arrays.

        Intended for point diagnostics and for the closed-form checks the
        test battery makes; the whole-grid phasor is built by
        :meth:`phasor_on` through the library's own routines."""
        u = np.asarray(x, dtype=np.float64) - self.centre[0]
        v = np.asarray(y, dtype=np.float64) - self.centre[1]
        L, M = self.tilt
        R = self.R
        if not math.isfinite(R) or R == 0.0:
            # Collimated (or the degenerate R = 0 the library also zeroes):
            # a plane wave, so the sphere contributes nothing and the ramp is
            # the whole of it.
            return L * u + M * v + self.piston
        sgn = 1.0 if R > 0 else -1.0
        if self._uses_exact_tilt():
            n = math.sqrt(1.0 - L * L - M * M)
            uu = u + R * L / n
            vv = v + R * M / n
            return sgn * (np.sqrt(uu * uu + vv * vv + R * R)
                          - abs(R) / n) + self.piston
        return (sgn * (np.sqrt(u * u + v * v + R * R) - abs(R))
                + L * u + M * v + self.piston)

    def gradient_at(self, x, y):
        """``(dC/dx, dC/dy)`` at ABSOLUTE coordinates -- the LOCAL DIRECTION
        COSINES this carrier imposes.

        This is the quantity every Nyquist bound in this module is computed
        from: a phase ``exp(i k C)`` has local spatial frequency
        ``|grad C| / lambda``, so a grid of pitch ``dx`` can represent it
        only while ``|grad C| <= lambda / (2 dx)``."""
        u = np.asarray(x, dtype=np.float64) - self.centre[0]
        v = np.asarray(y, dtype=np.float64) - self.centre[1]
        L, M = self.tilt
        R = self.R
        if not math.isfinite(R) or R == 0.0:
            return (np.full(np.shape(u), L, dtype=np.float64),
                    np.full(np.shape(v), M, dtype=np.float64))
        sgn = 1.0 if R > 0 else -1.0
        if self._uses_exact_tilt():
            n = math.sqrt(1.0 - L * L - M * M)
            uu = u + R * L / n
            vv = v + R * M / n
            den = np.sqrt(uu * uu + vv * vv + R * R)
            return (sgn * uu / den, sgn * vv / den)
        den = np.sqrt(u * u + v * v + R * R)
        return (sgn * u / den + L, sgn * v / den + M)

    # -- the whole-grid phasor -------------------------------------------
    def phasor_on(self, grid, wavelength, sign=1, *, with_piston=True):
        """``exp(sign * i * k0 * C)`` on ``grid`` (a :class:`FieldGrid`).

        Built from the LIBRARY'S OWN three lines --
        ``_exact_sphere_eikonal`` / ``_tilt_ramp`` / ``_tilt_exactness_phase``
        -- in that order, which is what
        ``carrier_referenced_exact_focus_readout`` and the probe's arm B both
        use.  Using the same routines rather than an independent closed form
        is deliberate: it is what makes the probe's null-control numbers
        reproduce through this API instead of merely agreeing with it.

        ``with_piston=False`` returns the SHAPE only, which is what a caller
        who intends to handle the constant separately (or who is comparing
        two carriers that share it) wants."""
        shape = grid.shape
        dx, dy = grid.dx, grid.dy
        cx = self.centre[0] - grid.origin[0]
        cy = self.centre[1] - grid.origin[1]
        k = 2.0 * np.pi / float(wavelength)
        S = _exact_sphere_eikonal(shape, dx, dy, wavelength, self.R,
                                  centre=(cx, cy))
        ph = np.exp((sign * 1j * k) * S)
        del S
        rp = _tilt_ramp(shape, dx, wavelength, self.tilt[0], self.tilt[1],
                        cx, cy, sign)
        if rp is not None:
            ph *= rp
            del rp
        xf = _tilt_exactness_phase(shape, dx, dy, wavelength, self.R,
                                   self.tilt[0], self.tilt[1], sign,
                                   centre=(cx, cy))
        if xf is not None:
            ph *= xf
            del xf
        if with_piston and self.piston != 0.0:
            ph *= np.exp(1j * (sign * _piston_phase(self.piston, wavelength)))
        return ph

    def to_dict(self) -> Dict[str, Any]:
        return {'R': _enc_float(self.R),
                'centre': [_enc_float(self.centre[0]),
                           _enc_float(self.centre[1])],
                'tilt': [_enc_float(self.tilt[0]), _enc_float(self.tilt[1])],
                'piston': _enc_float(self.piston)}

    @classmethod
    def from_dict(cls, d) -> 'CarrierSpec':
        return cls(R=_dec_float(d['R']),
                   centre=(_dec_float(d['centre'][0]),
                           _dec_float(d['centre'][1])),
                   tilt=(_dec_float(d['tilt'][0]), _dec_float(d['tilt'][1])),
                   piston=_dec_float(d.get('piston', 0.0)))


def _piston_phase(piston, wavelength):
    """``k0 * piston`` reduced to ``(-pi, pi]``, ACCURATELY.

    THE ARITHMETIC, because the accuracy of the piston bookkeeping is one of
    this module's acceptance bars and it is bounded by float64, not by the
    code.  A design-121 group's piston is ~3 mm = ~2300 waves = ~1.4e+04 rad.
    Forming ``k0 * piston`` first and exponentiating loses the low bits of
    the ANGLE: the relative error of the product is ``eps``, so its absolute
    error is ``eps * k0 * |piston|`` ~ 3e-12 rad, and it grows without bound
    with the path.

    Reducing in the OPL domain instead:

    .. code-block:: text

        n  = round(piston / wavelength)          # integer waves
        dp = piston - n * wavelength             # Sterbenz-exact subtraction
        phase = 2 * pi * dp / wavelength

    ``n * wavelength`` is within half a wavelength of ``piston``, so the
    subtraction is EXACT in float64 (both operands within a factor of two of
    each other) and ``dp`` inherits the ulp of ``n * wavelength``, i.e.
    ``eps * |piston|`` in METRES.  The resulting phase error is
    ``2 * pi * eps * |piston| / wavelength`` -- the same order as before, and
    that is not an accident: it is the precision with which a float64 can
    REPRESENT a millimetre-scale optical path at 1.31 um at all
    (``eps * 3e-3 m`` = 6.7e-19 m = 3.2e-12 rad).  What the reduction buys is
    that the error is now bounded by the representation of the INPUT and not
    additionally by the magnitude of an intermediate, so a piston small
    enough to matter to a 1e-12 rad bar (|p| <~ 0.9 mm) hits that bar, and a
    larger one degrades linearly and predictably rather than silently.

    ``phase`` is returned rather than the phasor so a caller can difference
    two pistons before exponentiating -- which :func:`re_reference` does,
    because the DIFFERENCE of two mm-scale pistons is usually small even when
    neither is."""
    p = float(piston)
    lam = float(wavelength)
    if p == 0.0:
        return 0.0
    n = math.floor(p / lam + 0.5)
    dp = p - n * lam
    return 2.0 * math.pi * (dp / lam)


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldGrid:
    """A centred lattice plus an ABSOLUTE origin.

    ``x_absolute[i] = (i - Nx/2) * dx + origin[0]`` (and likewise in y with
    ``Ny`` / ``dy``), which is the library's own half-pixel-offset centred
    convention -- see ``_exact_sphere_eikonal``, whose ``x`` is built exactly
    this way.  The ``origin`` is the part the library's private helpers do
    NOT carry, and it is required here because a per-order retrace grid is
    centred on that order's own chief ray (design 121: origins 0.000 /
    -1.508 / -3.016 mm)."""

    shape: Tuple[int, int]
    dx: float
    dy: float = None            # type: ignore[assignment]
    origin: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        ny, nx = int(self.shape[-2]), int(self.shape[-1])
        if ny < 1 or nx < 1:
            raise ValueError(f"FieldGrid: shape must be positive, got "
                             f"{self.shape!r}.")
        object.__setattr__(self, 'shape', (ny, nx))
        object.__setattr__(self, 'dx', float(self.dx))
        object.__setattr__(self, 'dy',
                           float(self.dx if self.dy is None else self.dy))
        object.__setattr__(self, 'origin', (float(self.origin[0]),
                                            float(self.origin[1])))
        if not (self.dx > 0.0 and self.dy > 0.0):
            raise ValueError(f"FieldGrid: dx and dy must be positive, got "
                             f"dx={self.dx!r}, dy={self.dy!r}.")

    @property
    def n(self) -> int:
        """Side length, for the square grids the resample requires."""
        return int(self.shape[-1])

    @property
    def is_square(self) -> bool:
        return self.shape[-1] == self.shape[-2] and self.dx == self.dy

    @property
    def extent(self) -> Tuple[float, float]:
        """Physical span ``(x, y)`` in metres."""
        return (self.shape[-1] * self.dx, self.shape[-2] * self.dy)

    @property
    def half_extent(self) -> Tuple[float, float]:
        return (0.5 * self.shape[-1] * self.dx, 0.5 * self.shape[-2] * self.dy)

    def axes(self):
        """``(x, y)`` 1-D ABSOLUTE coordinate vectors (m)."""
        nx, ny = self.shape[-1], self.shape[-2]
        x = (np.arange(nx, dtype=np.float64) - nx / 2) * self.dx \
            + self.origin[0]
        y = (np.arange(ny, dtype=np.float64) - ny / 2) * self.dy \
            + self.origin[1]
        return x, y

    def same_lattice(self, other) -> bool:
        """Do the two grids sample EXACTLY the same absolute points?

        Exact float equality is the right test: the identity short-circuit in
        :func:`re_reference` is only legal when the resample would be a
        no-op, and 'nearly the same lattice' is precisely the case that must
        still resample."""
        return (self.shape == other.shape and self.dx == other.dx
                and self.dy == other.dy and self.origin == other.origin)

    def to_dict(self) -> Dict[str, Any]:
        return {'shape': [int(self.shape[0]), int(self.shape[1])],
                'dx': _enc_float(self.dx), 'dy': _enc_float(self.dy),
                'origin': [_enc_float(self.origin[0]),
                           _enc_float(self.origin[1])]}

    @classmethod
    def from_dict(cls, d) -> 'FieldGrid':
        return cls(shape=(int(d['shape'][0]), int(d['shape'][1])),
                   dx=_dec_float(d['dx']), dy=_dec_float(d['dy']),
                   origin=(_dec_float(d['origin'][0]),
                           _dec_float(d['origin'][1])))


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------
@dataclass
class CarrierField:
    """An ENVELOPE, the grid it lives on, the CARRIER it is referenced to,
    the wavelength, and provenance.

    ``full_field = envelope * exp(i k0 C)`` with ``C`` the carrier's eikonal
    of :class:`CarrierSpec` (piston included).  The envelope is the SMOOTH
    residual -- which is the whole reason for the split: it is the only thing
    that can be resampled between grids without aliasing, while the carrier
    is analytic and can simply be re-evaluated anywhere.

    Nothing here propagates.  A :class:`CarrierField` is a value; the verbs
    are :func:`re_reference` and :func:`aggregate`."""

    envelope: np.ndarray
    grid: FieldGrid
    carrier: CarrierSpec
    wavelength: float
    provenance: Dict[str, Any] = _dc_field(default_factory=dict)

    def __post_init__(self):
        env = np.asarray(self.envelope)
        if env.ndim != 2:
            raise ValueError(
                f"CarrierField: envelope must be 2-D (Ny, Nx), got shape "
                f"{env.shape!r}.")
        if not np.iscomplexobj(env):
            env = env.astype(np.complex128)
        self.envelope = env
        if not isinstance(self.grid, FieldGrid):
            raise TypeError(
                f"CarrierField: grid must be a FieldGrid, got "
                f"{type(self.grid).__name__}.  Build one explicitly -- the "
                f"ORIGIN is not inferable from an array and getting it wrong "
                f"silently mis-places the field in absolute coordinates.")
        if tuple(env.shape) != tuple(self.grid.shape):
            raise ValueError(
                f"CarrierField: envelope shape {tuple(env.shape)} does not "
                f"match the grid's {tuple(self.grid.shape)}.")
        if not isinstance(self.carrier, CarrierSpec):
            raise TypeError(
                f"CarrierField: carrier must be a CarrierSpec, got "
                f"{type(self.carrier).__name__}.")
        self.wavelength = float(self.wavelength)
        if not (self.wavelength > 0.0 and math.isfinite(self.wavelength)):
            raise ValueError(
                f"CarrierField: wavelength must be finite and positive (m), "
                f"got {self.wavelength!r}.")
        self.provenance = _canonical_provenance(self.provenance)

    # -- basic accessors --------------------------------------------------
    @property
    def shape(self):
        return tuple(self.envelope.shape)

    @property
    def dx(self):
        return self.grid.dx

    @property
    def dy(self):
        return self.grid.dy

    @property
    def origin(self):
        return self.grid.origin

    @property
    def k0(self):
        return 2.0 * np.pi / self.wavelength

    def power(self) -> float:
        """Integrated power ``sum |envelope|^2 dx dy``.

        Identical to the full field's power: the carrier is a unit phasor
        everywhere it is defined."""
        return float((np.abs(self.envelope) ** 2).sum()) * self.dx * self.dy

    def full_field(self) -> np.ndarray:
        """Reconstruct ``envelope * exp(i k0 C)`` on this field's own grid."""
        return self.envelope * self.carrier.phasor_on(
            self.grid, self.wavelength, sign=+1)

    def total_opl_at(self, ix, iy) -> float:
        """TOTAL optical path in RADIANS at grid sample ``(iy, ix)``:
        ``k0 * C(x, y) + arg(envelope)``.

        This is the quantity the piston bookkeeping has to preserve under
        :func:`re_reference`, and it is deliberately reported in radians
        modulo nothing -- the caller compares two of them and takes the
        wrapped difference.  The carrier's piston is folded in through
        :func:`_piston_phase` so the reduction is the same one the phasor
        uses."""
        x, y = self.grid.axes()
        c = self.carrier
        shape_only = CarrierSpec(R=c.R, centre=c.centre, tilt=c.tilt,
                                 piston=0.0)
        eik = float(shape_only.eikonal_at(x[int(ix)], y[int(iy)]))
        return (self.k0 * eik + _piston_phase(c.piston, self.wavelength)
                + float(np.angle(self.envelope[int(iy), int(ix)])))

    def amp_radius(self) -> float:
        """1/e amplitude radius about the carrier's chief ray (m), via the
        library's own ``_envelope_amp_radius`` (niche-D6 decentred form)."""
        cx = self.carrier.centre[0] - self.grid.origin[0]
        cy = self.carrier.centre[1] - self.grid.origin[1]
        return float(_envelope_amp_radius(self.envelope, self.dx, self.dy,
                                          centre=(cx, cy)))

    def support_radius(self, frac=SUPPORT_POWER_FRACTION) -> float:
        """Radius about the chief ray enclosing ``frac`` of the power (m).

        MEASURED, chunked over rows so an 8192-square aperture field does not
        need a whole-grid radius array on top of itself.  Sized this way
        rather than as a multiple of the amplitude radius because the bound
        it feeds -- ``r/sqrt(r^2+R^2)`` -- is a strongly non-linear function
        of it, and design 121's own ratio is 2.23 w (2.64 mm against a 1.185
        mm amplitude radius), not a universal constant."""
        cx = self.carrier.centre[0] - self.grid.origin[0]
        cy = self.carrier.centre[1] - self.grid.origin[1]
        return _enclosed_power_radius(self.envelope, self.dx, self.dy,
                                      (cx, cy), float(frac))

    # -- construction -----------------------------------------------------
    @classmethod
    def from_full_field(cls, E, grid, carrier, wavelength, provenance=None):
        """Divide ``carrier`` out of a FULL field and wrap the envelope.

        This is step (1) of the probe's arm B -- 'divide out the order's own
        congruence' -- packaged, and it is where a traced chain's aperture
        field enters this module."""
        E = np.asarray(E)
        ph = carrier.phasor_on(grid, wavelength, sign=-1)
        env = E * ph
        del ph
        return cls(envelope=env, grid=grid, carrier=carrier,
                   wavelength=float(wavelength),
                   provenance=dict(provenance or {}))

    def with_provenance(self, **extra) -> 'CarrierField':
        """A shallow copy whose provenance carries ``extra`` (same array)."""
        prov = dict(self.provenance)
        prov.update(extra)
        return CarrierField(envelope=self.envelope, grid=self.grid,
                            carrier=self.carrier, wavelength=self.wavelength,
                            provenance=prov)


# ---------------------------------------------------------------------------
# Radial reductions (memory-bounded)
# ---------------------------------------------------------------------------
def _enclosed_power_radius(env, dx, dy, centre, frac):
    """Radius about ``centre`` (GRID frame, m) enclosing ``frac`` of the
    power, from a radial histogram accumulated one row-block at a time.

    Returns the OUTER edge of the first bin at which the cumulative fraction
    reaches ``frac``, so it is an upper bound on the true radius by at most
    one bin (``min(dx, dy)``) -- the conservative direction for a bound that
    is used to SIZE a grid."""
    env = np.asarray(env)
    ny, nx = env.shape[-2], env.shape[-1]
    x = (np.arange(nx, dtype=np.float64) - nx / 2) * dx - float(centre[0])
    y = (np.arange(ny, dtype=np.float64) - ny / 2) * dy - float(centre[1])
    step = min(float(dx), float(dy))
    r_max = float(np.hypot(np.abs(x).max(), np.abs(y).max()))
    nb = int(math.ceil(r_max / step)) + 1
    hist = np.zeros(nb + 1, dtype=np.float64)
    x2 = x * x
    for i0 in range(0, ny, _RADIAL_CHUNK_ROWS):
        i1 = min(i0 + _RADIAL_CHUNK_ROWS, ny)
        blk = np.abs(env[i0:i1]) ** 2
        rr = np.sqrt(x2[None, :] + (y[i0:i1] ** 2)[:, None])
        idx = np.minimum((rr / step).astype(np.int64), nb)
        hist += np.bincount(idx.ravel(), weights=blk.ravel(),
                            minlength=nb + 1)
        del blk, rr, idx
    tot = float(hist.sum())
    if not (tot > 0.0):
        return 0.0
    cum = np.cumsum(hist) / tot
    j = int(np.searchsorted(cum, float(frac), side='left'))
    j = min(j, nb)
    return float((j + 1) * step)


def _power_in_window(env, dx, dy, centre, radius):
    """Fraction of ``|env|^2`` inside a disc of ``radius`` about ``centre``
    (grid frame), chunked.  Used by the aggregate ledger."""
    env = np.asarray(env)
    ny, nx = env.shape[-2], env.shape[-1]
    x = (np.arange(nx, dtype=np.float64) - nx / 2) * dx - float(centre[0])
    y = (np.arange(ny, dtype=np.float64) - ny / 2) * dy - float(centre[1])
    x2 = x * x
    r2 = float(radius) ** 2
    inside = 0.0
    total = 0.0
    for i0 in range(0, ny, _RADIAL_CHUNK_ROWS):
        i1 = min(i0 + _RADIAL_CHUNK_ROWS, ny)
        blk = np.abs(env[i0:i1]) ** 2
        rr2 = x2[None, :] + (y[i0:i1] ** 2)[:, None]
        total += float(blk.sum())
        inside += float(blk[rr2 <= r2].sum())
        del blk, rr2
    return inside, total


# ---------------------------------------------------------------------------
# The Nyquist arithmetic
# ---------------------------------------------------------------------------
class NyquistReport(NamedTuple):
    """Everything :func:`carrier_difference_nyquist` computed, so a refusal
    can be argued with rather than merely obeyed."""

    ramp_max: float             #: max |grad C_src - grad C_dst| over the disc
    dx_ramp: float              #: lambda / (2 * ramp_max) -- the ENVELOPE
    na_src_max: float           #: max |grad C_src| over the disc
    dx_reconstruct: float       #: lambda / (2 * na_src_max) -- the FULL field
    na_dst_max: float           #: max |grad C_dst| over the disc -- DIAGNOSTIC
    dx_dst_ref: float           #: lambda / (2 * na_dst_max) -- NOT a bound
    dx_binding: float           #: min(dx_ramp, dx_reconstruct)
    binding_term: str           #: 'ramp' or 'reconstruct'
    support_radius: float       #: the disc the maxima were taken over
    dx_sum_bound: float         #: lambda/(2*(ramp+NA)) -- NOT operative
    margin: float               #: dx_binding / dx_target


def _max_over_disc(fn, centre, radius):
    """Maximise a smooth non-negative closed form over a disc.

    Swept on a polar lattice rather than evaluated at the boundary only: for
    the geometries this module is aimed at the maximum IS on the boundary,
    but that depends on the sign of the sphere and on where the two chief
    rays sit relative to each other, and a bound that is right 'usually' is
    the wrong kind of bound."""
    rr = np.linspace(0.0, float(radius), _NYQUIST_N_RADII)
    th = np.linspace(0.0, 2.0 * np.pi, _NYQUIST_N_ANGLES, endpoint=False)
    X = centre[0] + rr[:, None] * np.cos(th)[None, :]
    Y = centre[1] + rr[:, None] * np.sin(th)[None, :]
    return float(np.max(fn(X, Y)))


def carrier_difference_nyquist(src_carrier, dst_carrier, wavelength,
                               support_radius, dx_target=None,
                               support_centre=None) -> NyquistReport:
    """The pitch a grid needs to carry the DIFFERENCE between two carriers.

    THE ARITHMETIC (this is the probe's S3 census, computed rather than
    quoted).  TWO quantities bind, and one plausible-looking third does not.

    **(b) The ramp -- what the RE-REFERENCED ENVELOPE carries.**
    Re-referencing multiplies the envelope by ``exp(i k0 (C_src - C_dst))``,
    whose local spatial frequency is ``|grad C_src - grad C_dst| / lambda``.
    Maximised over the disc the field actually occupies:

    .. code-block:: text

        dx_ramp = lambda / (2 * max|grad C_src - grad C_dst|)

    **(c) The source band -- what the RECONSTRUCTED FULL FIELD carries.**
    The full field is ``env * exp(i k0 C_src)`` whatever it is referenced
    to, so its own local frequency is ``|grad C_src| / lambda``.  Any
    consumer that reconstructs (an exact final leg, an FFT, a plot) needs:

    .. code-block:: text

        dx_reconstruct = lambda / (2 * max|grad C_src|)

    **These do NOT add**, and that is the finding, not a simplification.  The
    ramp is a DIFFERENCE of two spheres; a beam's absolute angular band is
    set by how far its own sphere is sampled from its OWN centre and does not
    move with the chief ray.  ``dx_sum_bound`` = ``lambda/(2*(ramp + NA))``
    is reported so the mistake is visible, and it is NOT what binds:
    PROBE_SUM_AT_APERTURE S8.1 ran design 121 at a pitch 1.5x INSIDE that
    bound and the extreme order did not move (two pitches 2x apart agree to
    every printed digit).

    **What is NOT a bound: the destination carrier's own band.**
    ``max|grad C_dst|`` over the SOURCE's support is large whenever the two
    chief rays are far apart -- design 121 reads 0.527, which is the
    union-band red herring wearing a different hat, and taking it seriously
    would demand a 1.24 um pitch where 2.02 um is correct.  It is not a
    bound because the destination phasor is never SAMPLED as a signal: it is
    evaluated pointwise from a closed form and multiplied in, and the
    product de-aliases it (the same argument
    ``carrier_referenced_exact_focus_readout`` makes for accepting a
    co-moving grid that under-samples its own sphere).  It is reported as
    ``dx_dst_ref`` for diagnosis and excluded from ``dx_binding``.

    Design-121 back aperture, order (-4,-2) onto the on-axis common carrier,
    support radius 2.64 mm -- for calibration:

    .. code-block:: text

        ramp           0.2757  ->  dx <= 2.376 um     (probe quoted 0.2776 / 2.359,
                                                       from the paraxial |c|/|R|)
        source band    0.3245  ->  dx <= 2.018 um     (probe quoted 0.3239 / 2.0225)
        BINDING                     dx <= 2.018 um    <== the probe's 2.02 um
        dst band       0.5269  ->  1.243 um           NOT a bound
        sum bound                   0.816 um          NOT a bound (probe: 0.8089)

    Parameters
    ----------
    src_carrier, dst_carrier : CarrierSpec
    wavelength : float
    support_radius : float
        Radius of the disc over which the maxima are taken (m) -- the
        field's own support, MEASURED (see
        :meth:`CarrierField.support_radius`).
    dx_target : float, optional
        If given, ``margin = dx_binding / dx_target`` is filled in; a margin
        below 1 means the grid cannot hold the answer.
    support_centre : (float, float), optional
        Centre of that disc in ABSOLUTE coordinates.  Defaults to the SOURCE
        carrier's chief ray, which is where a source field's support sits.
    """
    lam = float(wavelength)
    c0 = tuple(support_centre if support_centre is not None
               else src_carrier.centre)

    def _ramp(X, Y):
        gsx, gsy = src_carrier.gradient_at(X, Y)
        gdx, gdy = dst_carrier.gradient_at(X, Y)
        return np.hypot(gsx - gdx, gsy - gdy)

    def _src(X, Y):
        gsx, gsy = src_carrier.gradient_at(X, Y)
        return np.hypot(gsx, gsy)

    def _dst(X, Y):
        gdx, gdy = dst_carrier.gradient_at(X, Y)
        return np.hypot(gdx, gdy)

    ramp_max = _max_over_disc(_ramp, c0, support_radius)
    na_src = _max_over_disc(_src, c0, support_radius)
    na_dst = _max_over_disc(_dst, c0, support_radius)

    def _pitch(slope):
        return float('inf') if slope <= 0.0 else lam / (2.0 * slope)

    dx_ramp, dx_src, dx_dst = _pitch(ramp_max), _pitch(na_src), _pitch(na_dst)
    cands = (('ramp', dx_ramp), ('reconstruct', dx_src))
    binding_term, dx_binding = min(cands, key=lambda kv: kv[1])
    return NyquistReport(
        ramp_max=ramp_max, dx_ramp=dx_ramp,
        na_src_max=na_src, dx_reconstruct=dx_src,
        na_dst_max=na_dst, dx_dst_ref=dx_dst,
        dx_binding=dx_binding, binding_term=binding_term,
        support_radius=float(support_radius),
        dx_sum_bound=_pitch(ramp_max + max(na_src, na_dst)),
        margin=(float('inf') if dx_target is None
                else dx_binding / float(dx_target)))


# ---------------------------------------------------------------------------
# PRIMITIVE 2 -- re_reference
# ---------------------------------------------------------------------------
class ReReferenceReport(NamedTuple):
    """What :func:`re_reference` did, recorded on the returned field's
    provenance under ``'re_reference'``."""

    nyquist: Dict[str, Any]
    resampled: bool
    support_radius: float
    containment_margin: float
    power_in: float
    power_out: float
    power_lost: float
    piston_delta: float
    piston_phase: float


def re_reference(field: CarrierField, to_carrier: CarrierSpec,
                 target_grid: FieldGrid, *,
                 support_frac: float = SUPPORT_POWER_FRACTION,
                 support_radius: Optional[float] = None,
                 nyquist_margin: float = 1.0,
                 on_nyquist: str = 'error',
                 on_window: str = 'warn',
                 bandlimit: bool = False,
                 _separable: bool = True) -> CarrierField:
    """Move ``field`` onto ``target_grid`` and onto ``to_carrier``, exactly.

    The full field is INVARIANT under this operation by construction:

    .. code-block:: text

        env_new = resample(env_old) * exp(i k0 (C_old - C_new))

    so ``env_new * exp(i k0 C_new) == env_old * exp(i k0 C_old)`` wherever
    both grids sample.  The carrier difference is ANALYTIC -- derived from
    the two :class:`CarrierSpec` objects and evaluated, never fitted -- which
    is what makes the result exact rather than merely close.

    ORDER MATTERS AND IS NOT ARBITRARY.  The resample comes FIRST, on the
    smooth envelope, and the carrier difference is applied AFTER, pointwise
    on the target lattice.  Doing it the other way round would ask the
    band-limited resample to carry the ramp -- which is exactly the term the
    Nyquist guard below exists to police -- and would alias it into the
    answer instead of refusing.

    THE RESAMPLE is a zero-distance band-limited ASM-MFT
    (:func:`angular_spectrum_propagate_mft` at ``z=0``), which is the
    library's own exact interpolation and the only route that can hit an
    arbitrary PITCH and an arbitrary ORIGIN at once.  Both are needed: a
    traced fan's per-order retrace grids differ in both (design 121: 1.5324
    vs 1.5243 um; origins 0.000 / -1.508 / -3.016 mm).  When the target
    lattice is EXACTLY the field's own, the resample is skipped -- it would
    be an identity, and skipping it makes ``re_reference`` bit-exact for the
    pure re-reference case (see ``resampled`` in the report).

    PISTON.  ``to_carrier.piston`` is honoured explicitly: the returned field
    is referenced to it, and the difference ``field.carrier.piston -
    to_carrier.piston`` is folded into the envelope as a unit phasor through
    :func:`_piston_phase` (which differences the two pistons BEFORE
    exponentiating, so a pair of millimetre-scale absolute paths that differ
    by microns costs the accuracy of the microns, not of the millimetres).
    Total optical path is preserved; ``CarrierField.total_opl_at`` is the
    check.

    Parameters
    ----------
    field : CarrierField
    to_carrier : CarrierSpec
        The carrier to reference the result to.
    target_grid : FieldGrid
        Must be SQUARE when a resample is required
        (:func:`angular_spectrum_propagate_mft` takes one ``N_out``); a
        rectangular target is accepted only when it is the field's own
        lattice.
    support_frac : float
        Enclosed-power fraction used to size the support disc every bound is
        measured over.  Default :data:`SUPPORT_POWER_FRACTION`.
    support_radius : float, optional
        Skip the measurement and use this radius (m).  Provided because the
        measurement is a full pass over the array and a caller
        re-referencing a fan of like fields already knows the answer.
    nyquist_margin : float
        Required ratio ``dx_binding / target_grid.dx``.  1.0 = bare Nyquist.
        The campaign's own habit is 2.0; the default is left at bare Nyquist
        so the guard refuses only what is genuinely unrepresentable, and a
        caller who wants the margin asks for it.
    on_nyquist : {'error', 'warn', 'ignore'}
        Disposition when the target grid cannot hold the carrier difference.
        Default ``'error'``: an aliased ramp is a PLAUSIBLE-LOOKING WRONG
        ANSWER -- it returns a populated, credible field with the wrong
        phase -- which is the failure class this library refuses by default.

        NOTE that the guard's ``reconstruct`` term also refuses a field whose
        OWN grid cannot carry its own carrier -- a chain's coarse co-moving
        exit plane, for instance (design 121: dx 33.2 um where the exit
        sphere needs 4.26 um).  That refusal is correct and is the point:
        PROBE_SUM_AT_APERTURE S2 established that the ONLY plane at which
        traced orders may legitimately be summed is the last group's back
        aperture ON THE EXACT LEG'S FINE RETRACE GRID, and this is that
        finding enforced rather than restated.  A strict no-op (same lattice
        AND same carrier shape) is exempt, since it cannot introduce a
        sampling error.
    on_window : {'error', 'warn', 'ignore'}
        Disposition when the support disc does not fit inside the target
        window.  Default ``'warn'``: losing skirt power is a quantified,
        recorded loss (``power_lost`` in the report), not a wrong answer.
    bandlimit : bool
        Passed to the resample.  Default ``False``, matching the probe: the
        envelope is already inside the band by construction and the
        band-limit filter would clip its skirt.

    Returns
    -------
    CarrierField
        On ``target_grid``, referenced to ``to_carrier``, with a
        :class:`ReReferenceReport` in ``provenance['re_reference']``.
    """
    if not isinstance(field, CarrierField):
        raise TypeError(f"re_reference: field must be a CarrierField, got "
                        f"{type(field).__name__}.")
    if not isinstance(to_carrier, CarrierSpec):
        raise TypeError(f"re_reference: to_carrier must be a CarrierSpec, "
                        f"got {type(to_carrier).__name__}.")
    if not isinstance(target_grid, FieldGrid):
        raise TypeError(f"re_reference: target_grid must be a FieldGrid, got "
                        f"{type(target_grid).__name__}.")
    _check_guard_action('on_nyquist', on_nyquist, 're_reference')
    _check_guard_action('on_window', on_window, 're_reference')

    lam = field.wavelength
    src = field.carrier
    same = field.grid.same_lattice(target_grid)
    same_shape = (to_carrier.R == src.R and to_carrier.centre == src.centre
                  and to_carrier.tilt == src.tilt)
    # A STRICT identity -- same lattice, same carrier shape -- is exempt from
    # the guard.  Not a loophole: the operation then touches nothing but the
    # piston constant, so it cannot introduce a sampling error, and refusing a
    # no-op because its INPUT was already coarse is a false positive.  Every
    # other case is guarded, including a same-lattice re-reference onto a
    # different sphere.
    identity = same and same_shape

    # ---- the support the bounds are measured over -----------------------
    if support_radius is None:
        r_sup = field.support_radius(support_frac)
    else:
        r_sup = float(support_radius)

    # ---- (e) THE NYQUIST GUARD -----------------------------------------
    dx_t = max(target_grid.dx, target_grid.dy)
    rep = carrier_difference_nyquist(src, to_carrier, lam, r_sup,
                                     dx_target=dx_t)
    if not identity and rep.dx_binding < float(nyquist_margin) * dx_t:
        _guard_dispose(
            on_nyquist,
            f"re_reference: the target grid's pitch "
            f"{dx_t * 1e6:.4f} um cannot hold the carrier difference.  The "
            f"binding bound is the {rep.binding_term!r} term at slope "
            f"{rep.ramp_max if rep.binding_term == 'ramp' else rep.na_src_max:.6f}"
            f" rad, i.e. dx <= {rep.dx_binding * 1e6:.4f} um "
            f"(required margin {float(nyquist_margin):.2f}x -> "
            f"{float(nyquist_margin) * dx_t * 1e6:.4f} um requested).  "
            f"Both bounds, over the measured support radius "
            f"{r_sup * 1e3:.4f} mm: the carrier-offset RAMP the re-referenced "
            f"ENVELOPE carries, |grad C_src - grad C_dst| = "
            f"{rep.ramp_max:.6f} rad -> dx <= {rep.dx_ramp * 1e6:.4f} um; and "
            f"the source band the RECONSTRUCTED FULL FIELD carries, "
            f"|grad C_src| = {rep.na_src_max:.6f} -> dx <= "
            f"{rep.dx_reconstruct * 1e6:.4f} um.  These do NOT add -- the ramp "
            f"is a DIFFERENCE of two spheres while a beam's absolute band does "
            f"not move with its chief ray -- so the binding pitch is their "
            f"MINIMUM, not lambda/(2*(ramp+NA)) = "
            f"{rep.dx_sum_bound * 1e6:.4f} um; and the target carrier's own "
            f"band over this support ({rep.na_dst_max:.6f} -> "
            f"{rep.dx_dst_ref * 1e6:.4f} um) is not a bound at all, because "
            f"that phasor is evaluated pointwise and de-aliased by the "
            f"product rather than sampled as a signal.  Refuse rather than "
            f"alias: a target grid this coarse returns a populated, "
            f"credible-looking field carrying the wrong phase.  Raise the "
            f"target N (or lower its dx), move the target carrier closer to "
            f"this field's own, or pass on_nyquist='warn' if the aliased "
            f"skirt is genuinely below your bar.",
            exc=ValueError, stacklevel=3)

    # ---- containment ----------------------------------------------------
    reach_x = abs(src.centre[0] - target_grid.origin[0]) + r_sup
    reach_y = abs(src.centre[1] - target_grid.origin[1]) + r_sup
    hx, hy = target_grid.half_extent
    margin = min(hx - reach_x, hy - reach_y)
    if margin < 0.0:
        _guard_dispose(
            on_window,
            f"re_reference: the field's support does not fit the target "
            f"window.  Its chief ray sits "
            f"({(src.centre[0] - target_grid.origin[0]) * 1e3:+.4f}, "
            f"{(src.centre[1] - target_grid.origin[1]) * 1e3:+.4f}) mm from "
            f"the target grid centre and its measured support radius "
            f"(enclosing {support_frac:.6g} of the power) is "
            f"{r_sup * 1e3:.4f} mm, so it reaches "
            f"{max(reach_x, reach_y) * 1e3:.4f} mm against a window "
            f"half-extent of {min(hx, hy) * 1e3:.4f} mm -- short by "
            f"{-margin * 1e3:.4f} mm.  The band-limited resample is PERIODIC, "
            f"so the overhang does not simply vanish: it wraps to the "
            f"opposite edge.  Grow N_out, or accept the recorded loss "
            f"(power_lost in the returned report).",
            exc=ValueError, stacklevel=3)

    # ---- (2) resample the ENVELOPE --------------------------------------
    p_in = field.power()
    if same:
        env = np.array(field.envelope, copy=True)
        resampled = False
    else:
        if not target_grid.is_square:
            raise ValueError(
                f"re_reference: a resample onto a non-square target grid is "
                f"not available -- angular_spectrum_propagate_mft takes a "
                f"single N_out -- and the target is "
                f"{target_grid.shape} at dx={target_grid.dx!r}, "
                f"dy={target_grid.dy!r}.  Use a square target, or a target "
                f"that is exactly this field's own lattice (then no resample "
                f"is needed).")
        env = angular_spectrum_propagate_mft(
            field.envelope, 0.0, lam, field.dx, target_grid.dx,
            target_grid.n,
            dy_in=field.dy, dy_out=target_grid.dy,
            centre_out=(target_grid.origin[0] - field.grid.origin[0],
                        target_grid.origin[1] - field.grid.origin[1]),
            bandlimit=bool(bandlimit),
            _bluestein_separable=bool(_separable))
        resampled = True

    # ---- (3) the ANALYTIC carrier difference ----------------------------
    # Built from the library's own trio, in the same order and with the same
    # signs the shipped exact readout uses, so this reproduces the probe's
    # arm B rather than merely agreeing with it.  Applied in place: at
    # design-121 scale each of these is a 1.07 GB array.
    # The SHAPE short-circuit is a strict identity, and it has to skip BOTH
    # phasors or neither: exp(+i k C_src) alone is not the carrier
    # DIFFERENCE, it is the carrier, and applying it would silently return a
    # full field wearing an envelope's label.
    if not same_shape:
        shape_src = CarrierSpec(R=src.R, centre=src.centre, tilt=src.tilt,
                                piston=0.0)
        shape_dst = CarrierSpec(R=to_carrier.R, centre=to_carrier.centre,
                                tilt=to_carrier.tilt, piston=0.0)
        ph = shape_src.phasor_on(target_grid, lam, sign=+1, with_piston=False)
        ph *= shape_dst.phasor_on(target_grid, lam, sign=-1,
                                  with_piston=False)
        env *= ph
        del ph

    # ---- the PISTON, differenced before it is exponentiated -------------
    dp = float(src.piston) - float(to_carrier.piston)
    pph = _piston_phase(dp, lam)
    if pph != 0.0:
        env *= np.exp(1j * pph)

    out_grid = target_grid
    p_out = float((np.abs(env) ** 2).sum()) * out_grid.dx * out_grid.dy
    report = ReReferenceReport(
        nyquist=rep._asdict(), resampled=resampled, support_radius=r_sup,
        containment_margin=float(margin), power_in=p_in, power_out=p_out,
        power_lost=p_in - p_out, piston_delta=dp, piston_phase=pph)
    prov = dict(field.provenance)
    prov['re_reference'] = _report_to_json(report)
    return CarrierField(envelope=env, grid=out_grid, carrier=to_carrier,
                        wavelength=lam, provenance=prov)


def _report_to_json(rep):
    d = dict(rep._asdict())
    for k, v in list(d.items()):
        if isinstance(v, float):
            d[k] = _enc_float(v)
        elif isinstance(v, dict):
            d[k] = {kk: (_enc_float(vv) if isinstance(vv, float) else vv)
                    for kk, vv in v.items()}
    return d


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
class FieldLedgerRow(NamedTuple):
    """One field's entry in an :class:`AggregateLedger`."""

    index: int
    weight_real: float
    weight_imag: float
    power_source: float          #: |w|^2 * power on the field's own grid
    power_on_common: float       #: |w|^2 * power landed on the common grid
    power_out_of_window: float   #: the difference -- what the window dropped
    frac_out_of_window: float
    support_radius: float
    containment_margin: float    #: window half-extent - (decentre + support)
    nyquist_margin: float        #: dx_binding / dx_common
    binding_term: str
    resampled: bool


class AggregateLedger(NamedTuple):
    """The energy book for one :func:`aggregate` call.

    An aggregation is only as trustworthy as the power it can account for,
    and the two ways it silently loses power -- a window too small for a
    displaced beam, a pitch too coarse for the residual ramp -- are exactly
    the two the probe had to size by hand.  Both are recorded per field."""

    rows: List[FieldLedgerRow]
    power_source_total: float
    power_on_common_total: float
    power_out_of_window_total: float
    frac_out_of_window_total: float
    worst_containment_margin: float
    worst_nyquist_margin: float
    n_fields: int

    def to_json(self):
        return {'rows': [{k: (_enc_float(v) if isinstance(v, float) else v)
                          for k, v in r._asdict().items()} for r in self.rows],
                'power_source_total': _enc_float(self.power_source_total),
                'power_on_common_total':
                    _enc_float(self.power_on_common_total),
                'power_out_of_window_total':
                    _enc_float(self.power_out_of_window_total),
                'frac_out_of_window_total':
                    _enc_float(self.frac_out_of_window_total),
                'worst_containment_margin':
                    _enc_float(self.worst_containment_margin),
                'worst_nyquist_margin': _enc_float(self.worst_nyquist_margin),
                'n_fields': int(self.n_fields)}


class AggregateResult(NamedTuple):
    field: CarrierField
    ledger: AggregateLedger


def aggregate(fields: Sequence[CarrierField], common_carrier: CarrierSpec,
              grid: FieldGrid, *, weights: Optional[Iterable] = None,
              support_frac: float = SUPPORT_POWER_FRACTION,
              nyquist_margin: float = 1.0,
              on_nyquist: str = 'error',
              on_window: str = 'warn',
              bandlimit: bool = False) -> AggregateResult:
    """Re-reference every field onto ``common_carrier`` / ``grid`` and sum.

    ``aggregate == re_reference + weighted sum``, and the equality is exact:
    with a single field and unit weight the result is bit-identical to the
    :func:`re_reference` call it delegates to (there is nothing to add it
    to).  With K fields it is exactly linear -- the probe measured that at
    2.3e-16, which is what makes a crosstalk decomposition a decomposition
    rather than an approximation.

    Peak memory is one accumulator plus one field's working set, not K of
    them: each field is re-referenced, accumulated and released in turn.
    (The probe's arm B peaked at 17.5 GB doing this, BELOW the shipped
    per-order path's 22.8-26.9 GB, because it never holds a retrace grid and
    a readout grid at once.)

    Parameters
    ----------
    fields : sequence of CarrierField
    common_carrier : CarrierSpec
        The ONE carrier every field is referenced to.  On a traced fan at a
        shared back aperture this is the central frame's own congruence --
        same ``R`` (the paraxial closure is order-independent there:
        design 121 measured -7.712425 mm for every order), chief ray on
        axis, no tilt.  A design whose exit radius VARIES across the fan
        needs the mean sphere plus a residual quadratic per order; that case
        is outside what the probe validated.
    grid : FieldGrid
        The common lattice.
    weights : iterable of complex, optional
        Per-field complex weight (e.g. a Dammann table's own amplitudes).
        Default 1 for every field.

    Returns
    -------
    AggregateResult
        ``(field, ledger)``.  The ledger is also on the field's provenance
        under ``'aggregate_ledger'``.
    """
    fields = list(fields)
    if not fields:
        raise ValueError("aggregate: no fields given -- an empty aggregation "
                         "has no wavelength and no grid to define it.")
    if weights is None:
        ws = [1.0 + 0.0j] * len(fields)
    else:
        ws = [complex(w) for w in weights]
        if len(ws) != len(fields):
            raise ValueError(
                f"aggregate: {len(ws)} weights for {len(fields)} fields.")
    lam = fields[0].wavelength
    for i, f in enumerate(fields):
        if f.wavelength != lam:
            raise ValueError(
                f"aggregate: field {i} is at wavelength {f.wavelength!r} m "
                f"but field 0 is at {lam!r} m.  A coherent sum of fields at "
                f"different wavelengths is not a field; aggregate each "
                f"wavelength separately.")

    acc = np.zeros(grid.shape, dtype=np.complex128)
    rows: List[FieldLedgerRow] = []
    for i, (f, w) in enumerate(zip(fields, ws)):
        rr = re_reference(f, common_carrier, grid,
                          support_frac=support_frac,
                          nyquist_margin=nyquist_margin,
                          on_nyquist=on_nyquist, on_window=on_window,
                          bandlimit=bandlimit)
        rep = rr.provenance['re_reference']
        aw2 = abs(w) ** 2
        p_src = _dec_float(rep['power_in']) * aw2
        p_com = _dec_float(rep['power_out']) * aw2
        if w == 1.0 + 0.0j:
            acc += rr.envelope
        else:
            acc += w * rr.envelope
        rows.append(FieldLedgerRow(
            index=i, weight_real=float(w.real), weight_imag=float(w.imag),
            power_source=p_src, power_on_common=p_com,
            power_out_of_window=p_src - p_com,
            frac_out_of_window=((p_src - p_com) / p_src if p_src > 0.0
                                else 0.0),
            support_radius=_dec_float(rep['support_radius']),
            containment_margin=_dec_float(rep['containment_margin']),
            nyquist_margin=_dec_float(rep['nyquist']['margin']),
            binding_term=str(rep['nyquist']['binding_term']),
            resampled=bool(rep['resampled'])))
        del rr

    p_src_tot = float(sum(r.power_source for r in rows))
    p_com_tot = float(sum(r.power_on_common for r in rows))
    ledger = AggregateLedger(
        rows=rows, power_source_total=p_src_tot,
        power_on_common_total=p_com_tot,
        power_out_of_window_total=p_src_tot - p_com_tot,
        frac_out_of_window_total=((p_src_tot - p_com_tot) / p_src_tot
                                  if p_src_tot > 0.0 else 0.0),
        worst_containment_margin=float(min(r.containment_margin
                                           for r in rows)),
        worst_nyquist_margin=float(min(r.nyquist_margin for r in rows)),
        n_fields=len(rows))
    prov = {'aggregate_ledger': ledger.to_json(),
            'n_fields': len(rows)}
    out = CarrierField(envelope=acc, grid=grid, carrier=common_carrier,
                       wavelength=lam, provenance=prov)
    return AggregateResult(field=out, ledger=ledger)


# ---------------------------------------------------------------------------
# Zarr IO
# ---------------------------------------------------------------------------
def _require_zarr():
    try:
        import zarr
        return zarr
    except ImportError:                                  # pragma: no cover
        raise ImportError(
            "zarr is required for CarrierField storage.  Install with: "
            "pip install 'lumenairy[zarr]'  (or: pip install zarr>=3.0)")


def _default_codecs(zarr, dtype):
    """zstd with a byte SHUFFLE in front, via the spec-registered Blosc codec.

    WHY SHUFFLE AT ALL.  A complex128 sample is 16 bytes, and across a smooth
    envelope the bytes at a given POSITION barely move (exponents and high
    mantissa bytes) while the bytes WITHIN a sample are near-random.
    De-interleaving the byte planes before the entropy coder is what makes
    the compression work.

    MEASURED, design-121 back aperture, 8192^2 complex128 = 1.0737 GB
    (``validation/repro_traced_carrier_121/cf_compression_121.py``; full
    table in ``docs/audits/BUILD_CARRIER_FIELD_2026_08_11.md``):

    .. code-block:: text

        envelope, zstd(5) alone, no shuffle          7.067x
        envelope, blosc zstd(5) + byte shuffle       8.923x   <== the default
        envelope, blosc zstd(5) + bitshuffle         8.816x
        envelope, numcodecs Shuffle(16) + zstd(5)    9.438x
        FULL FIELD, blosc zstd(5) + byte shuffle     7.722x

    WHY NOT THE 9.44x CHAIN, WHICH IS SMALLER.  ``numcodecs.Shuffle`` is not
    in the Zarr v3 specification -- zarr itself warns on every write that
    "other zarr implementations" may be unable to read it -- and a stored
    field is an ARCHIVE.  Blosc's zstd+shuffle is spec-registered, writes 20 %
    faster, and costs 5.8 % in size.  A caller who wants the last 5.8 % passes
    the chain explicitly through ``compressors=``; that is a per-store
    decision, not a library default.

    Returns the ``(serializer, compressors)`` pair Zarr v3 wants: the
    array->bytes step and the bytes->bytes chain are separate slots, and
    handing a serializer to ``compressors=`` is a ``TypeError``."""
    del dtype                       # blosc reads the typesize from the array
    from zarr.codecs import BloscCodec, BloscShuffle, BytesCodec
    return BytesCodec(), [BloscCodec(cname='zstd', clevel=5,
                                     shuffle=BloscShuffle.shuffle)]


def save_carrier_field_zarr(path, field: CarrierField, *, name='field',
                            chunks=None, compressors=None, serializer=None,
                            overwrite=False) -> str:
    """Write ONE :class:`CarrierField` as one Zarr group.

    LAYOUT (schema :data:`CARRIER_FIELD_SCHEMA`)::

        <path>/                        zarr group (the store)
          <name>/                      one group per field
            envelope                   complex128 (Ny, Nx), zstd + shuffle
            .zattrs                    every scalar, plus provenance JSON

    The envelope is the ONLY array; everything else -- grid, carrier,
    wavelength, provenance -- is an attribute, because all of it is scalar
    and because a reader should be able to enumerate what a store holds
    without decompressing a gigabyte.

    Non-finite scalars (a collimated carrier's ``R = +/-inf``) are written as
    their ``repr`` string, so the store stays valid JSON and ``R`` round-trips
    exactly.  Provenance is written as ONE JSON blob rather than as flattened
    attributes: it is the caller's dict, its keys are not under our control,
    and flattening it would make the round trip lossy for nested structure.

    Returns the path written."""
    zarr = _require_zarr()
    if not isinstance(field, CarrierField):
        raise TypeError(f"save_carrier_field_zarr: field must be a "
                        f"CarrierField, got {type(field).__name__}.")
    from ..io.storage import _get_lumenairy_version, _open_zarr_group_safe
    store = _open_zarr_group_safe(zarr, str(path), writable=True)
    if name in store:
        if not overwrite:
            raise FileExistsError(
                f"save_carrier_field_zarr: group {name!r} already exists in "
                f"{path!r}.  Pass overwrite=True to replace it -- silently "
                f"clobbering a stored field is how a run ends up scored "
                f"against someone else's array.")
        del store[name]
    grp = store.create_group(name)
    E = np.ascontiguousarray(field.envelope, dtype=np.complex128)
    ny, nx = E.shape
    if chunks is None:
        chunks = (min(1024, ny), min(1024, nx))
    ser, comp = _default_codecs(zarr, E.dtype)
    if serializer is not None:
        ser = serializer
    if compressors is not None:
        comp = compressors
    grp.create_array(name='envelope', data=E, chunks=tuple(chunks),
                     serializer=ser, compressors=comp)
    a = grp.attrs
    a['schema'] = int(CARRIER_FIELD_SCHEMA)
    a['lumenairy_version'] = _get_lumenairy_version()
    a['wavelength'] = _enc_float(field.wavelength)
    a['grid'] = field.grid.to_dict()
    a['carrier'] = field.carrier.to_dict()
    a['provenance_json'] = json.dumps(field.provenance, sort_keys=True,
                                      allow_nan=False)
    return str(path)


def load_carrier_field_zarr(path, *, name='field') -> CarrierField:
    """Read back what :func:`save_carrier_field_zarr` wrote.

    Refuses an unknown ``schema`` rather than guessing at a layout it does
    not know: a silently mis-read grid ORIGIN moves a field in absolute
    coordinates without changing a single sample of it, which no intensity
    metric can see."""
    zarr = _require_zarr()
    store = zarr.open_group(str(path), mode='r')
    if name not in store:
        raise KeyError(
            f"load_carrier_field_zarr: {path!r} has no group {name!r} "
            f"(it holds {sorted(store.group_keys())}).")
    grp = store[name]
    schema = int(grp.attrs.get('schema', -1))
    if schema != CARRIER_FIELD_SCHEMA:
        raise ValueError(
            f"load_carrier_field_zarr: {path!r}/{name} carries schema "
            f"{schema}, but this build reads schema "
            f"{CARRIER_FIELD_SCHEMA}.  Refusing to guess: the grid ORIGIN "
            f"and the carrier CENTRE are absolute positions, and a layout "
            f"mismatch relocates the field without changing any sample of "
            f"it.")
    env = np.asarray(grp['envelope'][...])
    grid = FieldGrid.from_dict(dict(grp.attrs['grid']))
    car = CarrierSpec.from_dict(dict(grp.attrs['carrier']))
    prov = json.loads(str(grp.attrs.get('provenance_json', '{}')))
    return CarrierField(envelope=env, grid=grid, carrier=car,
                        wavelength=_dec_float(grp.attrs['wavelength']),
                        provenance=prov)
