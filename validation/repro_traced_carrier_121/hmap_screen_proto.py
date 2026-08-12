"""PROTOTYPE -- the angle-aware analytic lens, BUILT AND THEN REFUTED.

**READ THIS PARAGRAPH BEFORE ANY OTHER.  The screen this module computes is
CORRECT as a characterisation of the element's point characteristic (measured
against exact ray traces at 3.0e-14 to 3.7e-09 waves over design 121's six
post-DOE groups, see ``hmap_consumer2_121.py``) and WRONG as a correction to
``apply_real_lens``.  It was wired into that function, measured, and the wiring
was REVERTED.  The library is untouched.  The refutation, in one line: the
quantity ``PROTO_HAMILTON_MAP_2026_08_11`` S7 sizes -- the angular differential
of the ray OPL at fixed ENTRANCE point -- is not the shipped model's error,
because the shipped model's in-glass ANGULAR SPECTRUM step already carries the
whole angular optical path EXACTLY, referenced to the EXIT point.  Measured on
a 25.4 mm N-BK7 plate at 20.7 mrad: the shipped screen model reproduces the
closed form ``k0 t sqrt(n^2 - L^2)`` to +0.000e+00 waves, and adding this
module's correction moves it off by 2.7697 waves.  Full derivation and the
corrected design in ``docs/audits/BUILD_ANGLE_AWARE_LENS_2026_08_11.md``.**

This file is kept because everything BELOW the mis-referencing is measured and
reusable: the reduced-angle parametrisation, the node ladder, the guard set,
the chain-A cache key, and the pupil-degree ladder that the corrected design
will need verbatim.

---

WHAT IT WAS FOR.  ``apply_real_lens`` models an element as a stack of
per-surface sag screens, ``phi += -k0 (n2 - n1) sag(x, y)``, with angular-
spectrum steps between them.  **Every one of those screens is a function of
``(x, y)`` alone: none of them carries an incidence angle.**  The incident
field's own tilt / curvature lives in ``E_in``'s phase, so the wave arrives at
the element with the right direction -- and then the element imprints the map
it would have imprinted at normal incidence.

MEASURED (``docs/audits/PROTO_HAMILTON_MAP_2026_08_11.md`` S7, design 121's six
post-DOE groups x 32 DOE orders, traced at each order's own chief-ray direction
and decomposed by least squares over a disc at fixed ENTRANCE point):

.. code-block:: text

    group  prescription              tilt      piston     tilt      RESIDUAL
      0    plate, N-SF1 25.40 mm    41.5 mrad   9.910 w   0.000 w   0.0000 w
      1    plate, N-BK7  3.20 mm    34.6 mrad   0.970 w   0.000 w   0.0000 w
      2    doublet PK52A/SF57       51.5 mrad   8.651 w   3.484 w   0.0041 w
      3    singlet LAK8             46.7 mrad   4.275 w   5.932 w   0.0070 w
      4    singlet LAK9              7.4 mrad   0.083 w   1.156 w   0.0014 w
      5    doublet SK2/SF57         54.9 mrad  14.713 w  25.682 w   0.2118 w

On the last group at the extreme order and a 3 mm pupil the angle-blind screen
is **0.212 waves rms wrong after piston AND tilt are removed** -- 21x
lambda/100 -- and still 0.122 waves if a defocus is allowed to absorb what it
can.  It grows as the square of the pupil radius (0.020 / 0.085 / 0.212 at
1 / 2 / 3 mm), the field-angle-dependent astigmatism-and-coma signature.  The
piston and tilt columns are ALSO missed, and they are not gauge: a piston is
observable the moment two congruences are summed at an aperture, which is what
a DOE fan does.

THE OBJECT.  For a group whose entrance carries the congruence with local
direction cosines ``(L(x, y), M(x, y))``, the exact quantity the screen is
missing is the ANGULAR DIFFERENTIAL of the group's own point characteristic::

    dOPL(x, y) = OPL(x, y; L, M) - OPL(x, y; 0, 0)

evaluated at fixed ENTRANCE point, with ``OPL`` the traced optical path from
the entrance plane to the exit VERTEX plane (the exit-vertex correction
``apply_real_lens_traced`` applies, verbatim).  Referencing to the normal-
incidence arm is what makes this composable: the shipped model already carries
the ``(0, 0)`` physics to its own accuracy, so ``dOPL`` adds exactly the term
that was absent and **nothing else**.  At normal incidence it is identically
zero, so the existing universe is untouched to the bit.

THE PARAMETRISATION, and it is the whole design.  Indexed by ABSOLUTE angle the
map's domain is the beam's OWN numerical aperture -- +-0.363 / 0.329 rad on
design 121 -- and a tensor node grid on it fires rays at 0.36 rad from pupil
points 9.6 mm off axis, which MISS THE GROUP: 44-50 % of the node rays die at
every node count tried (proto S3.2).  Indexed by the REDUCED angle

.. code-block:: text

    s(x, y) = (L(x, y), M(x, y)) - grad S_R(x, y)

-- the angle measured against ONE axis-centred reference sphere of radius
``R``, so the node lattice in the true angle is SHEARED across the pupil -- the
domain is the congruence's own departure from that sphere, +-0.082 / 0.037 rad
on the same case, 4.4x / 8.9x smaller, and no node ray dies.  **The shear is
not an optimisation; it is the difference between a map that exists and one
that does not.**

The node count of record is the proto's: ``nodes = (6, 5)`` = **30 nodes**, on
anisotropic axes because the two angular directions carry different spreads
(``Hx/Hy = 2.20`` on design 121) and an isotropic grid wastes ~20 % of its
nodes.  The proto measured the reduced-angle arm at 4.03e-04 waves there --
lambda/100 with 25x of margin -- and recorded the structural lesson that the
SHORT axis binds first (``ny = 4`` saturates: 6x4, 7x4 and 8x4 all read
2.87-2.99e-03 waves, so adding x nodes buys nothing once y binds).

Storage is ``(pupil_degree + 1)^2 x nx x ny x n_channels x 8 B``.  The proto
quotes **47.0 kB** for ``7 x 7 x 6 x 5 x 4``, i.e. at pupil degree 6; at the
degree 12 the ladder below forced it is **158.4 kB**, and at the degenerate
one-node angular box a real design 121 group produces it is **5.3 kB**.
Storage is a non-issue at every one of those, which is also the proto's own
conclusion.

WHY THE SOURCE-POSITION LABEL IS NOT USED HERE, although the proto measured it
as the more accurate parametrisation (9.91e-04 vs 2.87e-03 waves at 24 nodes).
The label ``(x_src, y_src) = (x0 - R L/N, y0 - R M/N)`` describes a CONGRUENCE,
so a map indexed by it represents a whole FAN of congruences from one build --
which is what the traced path wanted and what
``docs/audits/BUILD_INVERSE_MAP_2026_08_11.md`` S2 then refuted for that
consumer (the niche-C6 residual eikonal is aberration, not a source
displacement).  ``apply_real_lens`` is handed ONE carrier per call and has no
``a_fit``: there is no fan to share a build across and no augmentation to
represent, so the label buys nothing and costs a dimension.  Consuming the
angle FIELD instead means the map works for every carrier vocabulary the
library has -- ``TiltedCarrier``, a scalar conjugate, ``'auto'``, an explicit
wavefront -- because all of them can produce ``(L(x, y), M(x, y))`` and none of
them need produce a label.

THE FAIL-BEFORE (of the reverted wiring).  ``ANGLE_AWARE_LENS = False``
restored the shipped normal-incidence screen bit for bit -- not "disables the
correction" but "never builds the map", so no ray was traced and no array
touched.  Every guard below REFUSES to that same state rather than degrading
toward it.

AND HERE IS WHERE IT BREAKS, stated once more because it is the reason this
file is not in the library.  The paragraph above claimed ``dOPL`` "does NOT
carry the transverse ray WALK-OFF ... it is a separate and larger term".  **The
walk-off is not a separate term.**  For a plane-parallel plate, substituting
``x0 = x1 - t tan(theta_t)`` into ``k0 L x0 + k0 n t / cos(theta_t)`` gives
``k0 L x1 + k0 n t cos(theta_t)`` exactly -- which is precisely what the
in-glass angular spectrum already produces.  The entrance-referenced path
length and the walk-off are the two halves of ONE identity and they cancel.
So ``dOPL`` at fixed entrance point is the MIRROR IMAGE of the element's true
exit-referenced angular differential, not the missing part of it, and adding it
to the shipped model is wrong by twice the term.  See
``docs/audits/BUILD_ANGLE_AWARE_LENS_2026_08_11.md`` S2.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# the switches
# ---------------------------------------------------------------------------

#: Vestigial: this module is no longer wired into anything (see the header's
#: first paragraph).  Kept so the measurement scripts read the same switch the
#: reverted build used.
ANGLE_AWARE_LENS = True

#: REPORTING-ONLY: what a REFUSED build says.  A refused build keeps the
#: shipped screen whatever this holds, so it cannot change a returned bit.
#: ``'silent'`` suppresses the diagnostic; ``'warn'`` issues a ``UserWarning``
#: naming the guard that refused and the fail-before it fell back to.
ANGLE_AWARE_LENS_GUARD = 'warn'

#: Chebyshev nodes per REDUCED-ANGLE axis, ``(nx, ny)``.  The proto's node
#: count of record (S3.3): 30 nodes hold the interpolation error to 4.03e-04
#: waves on the reduced-angle domain over design 121's whole 32-order fan --
#: lambda/100 with 25x of margin -- and the axes are anisotropic because the
#: two angular directions carry different spreads.  A degenerate axis (an
#: angular half-width of EXACTLY zero, which is what an on-axis or a purely
#: collimated congruence produces) collapses to one node automatically, so the
#: untilted case costs one trace and interpolates nothing.
_HMAP_NODES = (6, 5)

#: Chebyshev degree per PUPIL axis for the least-squares fit of each channel.
#:
#: SET BY MEASUREMENT, and the first value tried was wrong.  6 -- the shipped
#: ``newton_poly_order``, the degree the traced element fits its own
#: entrance-coordinate map at -- was the obvious choice and it FAILS: the
#: quantity fitted here is small and smooth (``dOPL`` is 0.2 waves over 3 mm on
#: the worst measured case, not the ~1e-02 m path itself), but the PUPIL is
#: whatever the caller's grid is, and the correction stiffens fast with radius
#: on a fast element.  Measured on design 121's last post-DOE group (doublet
#: SK2/SF57, R = 19.6 / -27.4 / 12.65 mm) at its extreme DOE order, against
#: direct ray traces (``hmap_consumer2_121.py degree``), max error in waves:
#:
#:     pupil      deg 6     deg 8     deg 10    deg 12    deg 14
#:     3.00 mm    4.13e-04  1.00e-05  2.11e-07  8.18e-09  3.33e-10
#:     4.35 mm    REFUSED   3.89e-04  1.88e-05  1.62e-06  1.40e-07
#:     6.00 mm    REFUSED   REFUSED   1.37e-03  2.48e-04  4.55e-05
#:
#: (``REFUSED`` is G7 declining the build at that degree -- the guard caught
#: every one of these before a wrong screen could be applied, which is what it
#: is for.)  On the singlet LAK8 of group 3 even degree 6 reads 3.5e-06 waves
#: at 6 mm, so the binding case is the fast doublet and nothing else.
#:
#: 12 is the first degree that clears the 3x bar on the HARDEST measured case
#: with margin (2.48e-04 against 3.33e-03, 13x) while being 5 decades inside it
#: on the 3 mm pupil the proto sized.  Beyond it the ladder is still
#: geometric, so a caller who needs a larger pupil on a faster element raises
#: this rather than discovering a quiet error -- G7 refuses first.
_HMAP_PUPIL_DEGREE = 12

#: Samples per axis of the square pupil lattice each node congruence is traced
#: on.  ``49 x 49`` puts 1 885 samples inside the disc against the 169 free
#: coefficients of degree 12 -- 11x, against G6's 4x floor -- and at 30 nodes
#: plus the normal-incidence arm it is ~74 k rays, the same order the traced
#: element already pays per call, measured in tens of milliseconds.
_HMAP_PUPIL_LATTICE = 49

#: Relative pad on the reduced-angle box, so a carrier whose angle field sits
#: at the box edge is interpolated rather than evaluated exactly at an
#: endpoint.  NOT an accuracy knob: G7 measures the built map against direct
#: traces at the caller's OWN angles and refuses on the measurement, so a pad
#: that was too small shows up as a refusal, not as a silent error.
_HMAP_BOX_PAD = 0.05

#: G2: refuse when the congruence Jacobian changes sign over the node lattice
#: (a fold -- the characteristic is not single-valued and no interpolant of it
#: means anything), and when its dynamic range exceeds this (a caustic).  The
#: same 30x the traced ray-density amplitude already uses for its own fold
#: flag; the proto censused design 121's last group at 1.2615 over 102 688
#: samples and 32 congruences, with zero sign flips.
_HMAP_DETJ_MAXMIN = 30.0

#: G7's acceptance bar, in waves, on the map's own measured error against
#: DIRECT ray traces at the caller's own angles.  ``lambda/100`` with the
#: campaign's standard 3x of margin.  This is the guard that carries the
#: accuracy: every other number in this module (the node count, the pupil
#: degree, the box pad, the lattice) is an input to a build whose output is
#: then MEASURED against exact rays and refused if it misses.
_HMAP_ACCEPT_WAVES = 1.0e-2 / 3.0

#: Pupil radii x azimuths of the G7 check set.  Held out from the fit in the
#: only sense that matters -- these are not lattice points and not node angles;
#: they are fresh traces at the caller's own angle field.
_HMAP_CHECK_RINGS = (0.0, 0.35, 0.65, 0.85, 0.98)
_HMAP_CHECK_AZIMUTHS = 8

#: How many built maps to retain.  The key is a SHA-256 over everything the map
#: depends on (see :func:`_hmap_key`) -- the chain-A cache discipline of
#: ``docs/audits/FIX_D4_D6_D7_2026_08_06.md`` D6, applied to an in-process
#: cache: a key that names the CONFIGURATION and not the CONTENT is how a cache
#: silently becomes a cache of something else.
_HMAP_CACHE_SIZE = 8

#: Pixels per row band of the per-pixel evaluation.  Bounds the contraction
#: transient at ``~band x nx x ny x 8`` bytes, so a large wave grid never
#: materialises a second full copy of anything.
_HMAP_EVAL_BAND_BYTES = 64 << 20


class AngleAwareRefusal(Exception):
    """A guard refused to build or to evaluate the map.

    Never propagates to a caller of ``apply_real_lens``: the call site catches
    it, reports it under :data:`ANGLE_AWARE_LENS_GUARD` and keeps the shipped
    normal-incidence screen.  It exists as an exception so that a refusal
    cannot be confused with a zero correction -- one is "the shipped physics",
    the other is "the angle-aware physics, which happens to vanish here"."""

    def __init__(self, guard: str, detail: str):
        self.guard = str(guard)
        self.detail = str(detail)
        super().__init__(f'{self.guard}: {self.detail}')


# ---------------------------------------------------------------------------
# Chebyshev machinery
# ---------------------------------------------------------------------------
def _cheb_nodes(n):
    """Chebyshev-Lobatto nodes on ``[-1, 1]``, ascending; ``[0.0]`` for n == 1.

    ``n == 1`` is the DEGENERATE axis and it is reached by construction, not by
    accident: an untilted, undecentred congruence has a reduced-angle spread of
    exactly zero on both axes, and interpolating a constant over one node is
    the right answer there rather than a special case."""
    if int(n) <= 1:
        return np.array([0.0])
    n = int(n)
    return np.cos(np.pi * np.arange(n - 1, -1, -1) / (n - 1))


def _cheb_node_inverse(n):
    """``inv(V)`` for the Chebyshev-Lobatto Vandermonde of ``n`` nodes."""
    if int(n) <= 1:
        return np.ones((1, 1))
    t = _cheb_nodes(n)
    return np.linalg.inv(np.polynomial.chebyshev.chebvander(t, int(n) - 1))


def _cheb_basis(u, deg):
    """``(deg + 1, u.size)`` Chebyshev values ``T_i(u)`` by recurrence."""
    u = np.asarray(u, dtype=float).ravel()
    out = np.empty((int(deg) + 1, u.size), dtype=float)
    out[0] = 1.0
    if deg >= 1:
        out[1] = u
    for i in range(2, int(deg) + 1):
        out[i] = 2.0 * u * out[i - 1] - out[i - 2]
    return out


def _to_unit(v, centre, half):
    """Map ``v`` onto ``[-1, 1]`` about ``centre`` with half-width ``half``.

    A half-width of EXACTLY zero is the degenerate axis and maps to 0.0 -- the
    single node -- rather than to a division by zero."""
    v = np.asarray(v, dtype=float)
    if half == 0.0:
        return np.zeros_like(v)
    return (v - centre) / half


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
class CongruenceMap:
    """A characterised element: the angular differential of its own point
    characteristic, as a tensor Chebyshev in (pupil x reduced angle).

    Built by :func:`build_congruence_map`.  Immutable in use; the only public
    entry is :meth:`delta_opl`.

    Attributes
    ----------
    coef : ndarray, (n_channels, P+1, P+1, nx, ny)
        Chebyshev coefficients.  Axis 0 indexes :attr:`channels`; axes 1-2 the
        pupil ``(x, y)``; axes 3-4 the reduced angle ``(sx, sy)``.
    channels : tuple of str
        ``('dopl', 'x_out', 'y_out', 'det_j')``.  ``'dopl'`` is the correction;
        the other three are the characteristic's own companions, kept because
        G2 judges the build on them and because a future amplitude consumer
        needs exactly them (the proto measured the ray-tube amplitude
        interpolating 3-4 decades more easily than the OPL, S6.5).
    r_ref : float
        Radius of the axis-centred reference sphere the angle is REDUCED
        against.  ``+/-inf`` means the reference is the plane, i.e. the reduced
        angle is the absolute one.
    pupil_radius : float
        The map's declared pupil domain.  Outside it the correction is zero --
        see :meth:`delta_opl`.
    """

    __slots__ = ('coef', 'channels', 'r_ref', 'pupil_radius', 'angle_centre',
                 'angle_half', 'nodes', 'pupil_degree', 'wavelength',
                 'det_j_range', 'det_j_sign', 'n_rays', 'fit_residual_waves',
                 'check_max_waves', 'key', 'name')

    def __init__(self, coef, channels, r_ref, pupil_radius, angle_centre,
                 angle_half, nodes, pupil_degree, wavelength, det_j_range,
                 det_j_sign, n_rays, fit_residual_waves, check_max_waves,
                 key, name):
        self.coef = coef
        self.channels = tuple(channels)
        self.r_ref = float(r_ref)
        self.pupil_radius = float(pupil_radius)
        self.angle_centre = (float(angle_centre[0]), float(angle_centre[1]))
        self.angle_half = (float(angle_half[0]), float(angle_half[1]))
        self.nodes = (int(nodes[0]), int(nodes[1]))
        self.pupil_degree = int(pupil_degree)
        self.wavelength = float(wavelength)
        self.det_j_range = float(det_j_range)
        self.det_j_sign = int(det_j_sign)
        self.n_rays = int(n_rays)
        self.fit_residual_waves = float(fit_residual_waves)
        self.check_max_waves = float(check_max_waves)
        self.key = key
        self.name = str(name)

    # -- introspection ------------------------------------------------------
    @property
    def nbytes(self) -> int:
        """Coefficient storage in bytes (the proto's 47.0 kB at the defaults)."""
        return int(self.coef.nbytes)

    def __repr__(self):                                   # pragma: no cover
        return (f'<CongruenceMap {self.name!r} nodes={self.nodes} '
                f'deg={self.pupil_degree} r_ref={self.r_ref:.6g} '
                f'r_pupil={self.pupil_radius:.6g} '
                f'{self.nbytes / 1024.0:.1f} kB>')

    # -- the reduced angle --------------------------------------------------
    def reduced_angle(self, X, Y, L, M):
        """``(sx, sy)`` -- the caller's angle field measured against this map's
        own reference sphere.  This is the SHEAR, and it is what keeps the
        node rays inside the element."""
        rx, ry = _reference_sphere_gradient(self.r_ref, X, Y)
        return np.asarray(L, dtype=float) - rx, np.asarray(M, dtype=float) - ry

    def in_box(self, sx, sy):
        """G5 -- is every sample inside the built reduced-angle box?

        Chebyshev extrapolation past the endpoints diverges like the degree,
        and at ``nx = 6`` that is fast, so an out-of-box evaluation is refused
        rather than clipped."""
        cx, cy = self.angle_centre
        hx, hy = self.angle_half
        tol = 1.0 + 1e-9
        ok_x = (np.abs(np.asarray(sx) - cx) <= hx * tol) if hx > 0.0 else \
            np.isclose(np.asarray(sx), cx, rtol=0.0, atol=1e-15)
        ok_y = (np.abs(np.asarray(sy) - cy) <= hy * tol) if hy > 0.0 else \
            np.isclose(np.asarray(sy), cy, rtol=0.0, atol=1e-15)
        return bool(np.all(ok_x) and np.all(ok_y))

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, X, Y, L, M, channel='dopl'):
        """One channel of the map on the query grid.

        ``X, Y`` are ABSOLUTE transverse positions (metres) and ``L, M`` the
        local direction cosines there -- the same absolute frame the carrier
        states itself in.  Outside :attr:`pupil_radius` the result is 0.0: the
        polynomial is not evaluated there at all, so nothing can blow up in a
        region the map was never built on.
        """
        try:
            ch = self.channels.index(str(channel))
        except ValueError:                                # pragma: no cover
            raise KeyError(f'CongruenceMap has no channel {channel!r}; '
                           f'has {self.channels}')
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        sx, sy = self.reduced_angle(X, Y, L, M)
        inside = (X * X + Y * Y) <= self.pupil_radius ** 2
        if not self.in_box(sx[inside] if np.ndim(sx) else sx,
                           sy[inside] if np.ndim(sy) else sy):
            raise AngleAwareRefusal(
                'G5 in-box',
                'the reduced angle of this call leaves the box the map was '
                'built on (centre %r half %r)'
                % (self.angle_centre, self.angle_half))
        out = np.zeros(np.broadcast(X, Y, sx, sy).shape, dtype=float)
        if not np.any(inside):
            return out
        _eval_tensor_into(out, inside, self.coef[ch],
                          _to_unit(X, 0.0, self.pupil_radius),
                          _to_unit(Y, 0.0, self.pupil_radius),
                          _to_unit(sx, self.angle_centre[0],
                                   self.angle_half[0]),
                          _to_unit(sy, self.angle_centre[1],
                                   self.angle_half[1]))
        return out

    def delta_opl(self, X, Y, L, M):
        """THE CORRECTION, in metres of optical path.

        ``dOPL(x, y) = OPL(x, y; L, M) - OPL(x, y; 0, 0)`` at fixed entrance
        point.  Add it to the shipped model's phase as ``exp(+1j k0 dOPL)`` --
        the library's ``phase = exp(+i k0 OPL)`` convention, the same sign the
        Seidel correction and the traced element's own OPL leg use."""
        return self.evaluate(X, Y, L, M, channel='dopl')


def _eval_tensor_into(out, inside, coef, ux, uy, usx, usy):
    """Contract the (P+1, P+1, nx, ny) tensor onto the masked query points.

    Ordering matters and is measured rather than assumed.  Contracting the
    PUPIL axes first costs ``(P+1) x nx x ny`` multiply-adds per point and
    leaves ``nx x ny`` for the angular pair, i.e. ~270 flops/pixel at the
    shipped shape; contracting the ANGULAR axes first costs ``(P+1)^2 x nx x
    ny`` = 1 470.  The pupil axes are also SEPARABLE on a rectangular grid,
    which is what makes the first ordering available at all."""
    P1 = coef.shape[0]
    nx, ny = coef.shape[2], coef.shape[3]
    ux = np.broadcast_to(np.asarray(ux, dtype=float), out.shape)
    uy = np.broadcast_to(np.asarray(uy, dtype=float), out.shape)
    usx = np.broadcast_to(np.asarray(usx, dtype=float), out.shape)
    usy = np.broadcast_to(np.asarray(usy, dtype=float), out.shape)
    idx = np.flatnonzero(np.asarray(inside).ravel())
    if idx.size == 0:                                     # pragma: no cover
        return
    per_pt = max(1, nx * ny * 8)
    band = max(1, int(_HMAP_EVAL_BAND_BYTES // per_pt))
    flat = out.ravel()
    fux, fuy = ux.ravel(), uy.ravel()
    fsx, fsy = usx.ravel(), usy.ravel()
    for b0 in range(0, idx.size, band):
        sel = idx[b0:b0 + band]
        Bx = _cheb_basis(fux[sel], P1 - 1)            # (P1, n)
        By = _cheb_basis(fuy[sel], P1 - 1)            # (P1, n)
        Bs = _cheb_basis(fsx[sel], nx - 1)            # (nx, n)
        Bt = _cheb_basis(fsy[sel], ny - 1)            # (ny, n)
        # (P1, P1, nx, ny) x (P1, n) -> (P1, nx, ny, n)
        U = np.einsum('ijkl,jn->ikln', coef, By, optimize=True)
        # -> (nx, ny, n)
        V = np.einsum('ikln,in->kln', U, Bx, optimize=True)
        flat[sel] = np.einsum('kln,kn,ln->n', V, Bs, Bt, optimize=True)


def _reference_sphere_gradient(r_ref, X, Y):
    """``grad S_R(x, y)`` -- the direction cosines of the AXIS-CENTRED sphere
    of radius ``r_ref``, exactly, from the library's own closed form.

    ``+/-inf`` is the plane, whose gradient is zero, so the reduced angle
    degenerates to the absolute one -- which is the right reduction for a
    collimated carrier and the reason the collimated case needs no special
    branch anywhere else in this module."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if not np.isfinite(r_ref):
        return np.zeros_like(X + Y), np.zeros_like(X + Y)
    from lumenairy.elements._lens_traced import TiltedCarrier, _tilted_carrier_parts
    _W, Lg, Mg = _tilted_carrier_parts(
        TiltedCarrier(float(r_ref), 0.0, 0.0, 0.0, 0.0), X, Y)
    return Lg, Mg


# ---------------------------------------------------------------------------
# the trace -- reproduced from apply_real_lens_traced's own step
# ---------------------------------------------------------------------------
def _trace_characteristic(surfaces, n_exit, px, py, thx, thy, wavelength):
    """``(x_out, y_out, opl, alive)`` for one bundle.

    Reproduces ``apply_real_lens_traced``'s trace step VERBATIM, including the
    EXIT-VERTEX CORRECTION: rays land on the last surface's sag, and the wave
    model's exit plane is the flat vertex plane, so each ray is advanced by
    ``-z/N`` at the exit index.  Omitting it is a per-ray OPL error of the
    last surface's sag depth, which on design 121's last group is tens of
    microns -- tens of waves."""
    from lumenairy.raytrace import _make_bundle, trace
    rays = _make_bundle(x=np.ravel(px), y=np.ravel(py),
                        L=np.ravel(thx), M=np.ravel(thy),
                        wavelength=float(wavelength))
    res = trace(rays, surfaces, float(wavelength), output_filter='last')
    fin = res.image_rays
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(fin.alive & (np.abs(fin.N) > 1e-30), -fin.z / fin.N, 0.0)
    return (fin.x + fin.L * t, fin.y + fin.M * t, fin.opd + n_exit * t,
            np.asarray(fin.alive, dtype=bool))


def _traceable_surfaces(prescription, wavelength):
    """``(surfaces, n_exit)`` for the map's own trace.

    ``aperture_diameter`` is POPPED, exactly as ``apply_real_lens_traced``
    does: it is the element's declared clear aperture and a LAUNCH bound, not a
    per-surface clip, and leaving it in would let it kill rays the wave model
    apertures separately (and identically).  Per-surface ``semi_diameter`` /
    ``clear_aperture`` are KEPT -- those ARE physical clips and a ray that
    misses them has no path to contribute."""
    from lumenairy.glass import get_glass_index
    from lumenairy.raytrace import surfaces_from_prescription
    presc = dict(prescription)
    presc.pop('aperture_diameter', None)
    presc.pop('stop_index', None)
    surfaces = surfaces_from_prescription(presc)
    if not surfaces:
        raise AngleAwareRefusal('G0 traceable',
                                'the prescription yielded no surfaces')
    n_exit = float(get_glass_index(surfaces[-1].glass_after, wavelength))
    return surfaces, n_exit


def _sag_finite_radius(prescription, r_cap):
    """The largest radius at which EVERY surface's sag is finite.

    A sphere of radius ``R`` simply does not exist past ``|R|``, and
    ``apply_real_lens`` handles that by zeroing the NaN OPD there.  The map's
    pupil domain must not reach past it either -- not because the fit would be
    poor, but because there is no surface to trace to.  Measured by evaluating
    the shipped sag helper on a radial scan, so it agrees with the screen the
    correction is added to by construction (a NaN test, not a tolerance)."""
    from lumenairy.elements.lenses import surface_sag_general
    surfaces = prescription.get('surfaces') or []
    rr = np.linspace(0.0, float(r_cap), 513)
    ok = np.ones(rr.shape, dtype=bool)
    for surf in surfaces:
        try:
            sag = surface_sag_general(rr * rr, surf.get('radius'),
                                      surf.get('conic', 0.0),
                                      surf.get('aspheric_coeffs'))
        except (TypeError, ValueError):                   # pragma: no cover
            return float(r_cap)
        ok &= np.isfinite(np.asarray(sag, dtype=float))
    if bool(ok.all()):
        return float(r_cap)
    first_bad = int(np.argmin(ok))
    if first_bad == 0:                                    # pragma: no cover
        raise AngleAwareRefusal('G0 traceable',
                                'no surface has finite sag on the axis')
    return float(rr[first_bad - 1])


def map_pupil_radius(prescription, grid_half_diagonal):
    """The map's declared pupil domain.

    The smallest of three, and the choice of which three is the point:

    * the grid's own half-DIAGONAL, so the disc covers every pixel of a square
      grid rather than only its inscribed circle;
    * the element's declared ``aperture_diameter`` when it is the ENTRANCE stop
      (``stop_index`` unset), which is the one clip ``apply_real_lens`` applies
      before any screen;
    * the radius past which some surface's sag stops existing (a sphere of
      radius ``R`` simply is not there past ``|R|``).

    Outside the result the correction is ZERO, and on the two cases that can
    arise that is not an artefact: with an entrance ``aperture_diameter`` the
    wave field is already zeroed there (any phasor leaves a zero a zero), and
    past the sag-finite radius the shipped screen has already zeroed its own
    NaN OPD, so both models say "nothing" in the same place.

    NOTE WHAT IS DELIBERATELY *NOT* IN THE LIST: per-surface ``semi_diameter``
    and ``clear_aperture``.  ``apply_real_lens`` does not vignette on
    ``semi_diameter`` at all, so shrinking the map's domain to it would put a
    phase STEP in the middle of an illuminated field.  Those clips are instead
    left to kill rays inside the declared domain, where G3 sees them and
    REFUSES the whole map -- the element cannot be characterised over the field
    it is being handed, and the honest answer is the shipped screen, not a
    quietly truncated correction."""
    r = float(grid_half_diagonal)
    ap = prescription.get('aperture_diameter')
    if (ap is not None and prescription.get('stop_index') is None
            and np.isfinite(ap) and ap > 0.0):
        r = min(r, 0.5 * float(ap))
    return _sag_finite_radius(prescription, r)


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------
def build_congruence_map(prescription, wavelength, r_ref, *,
                         pupil_radius, angle_centre, angle_half,
                         nodes=None, pupil_degree=None, pupil_lattice=None,
                         check_angle_fn=None, name='', key=None):
    """Build the angle-aware map for one element.

    Parameters
    ----------
    prescription : dict
        The same dict ``apply_real_lens`` is given.
    wavelength : float
        Free-space wavelength (m).
    r_ref : float
        Reference-sphere radius the angle is reduced against (m); ``+/-inf``
        reduces against the plane.
    pupil_radius : float
        The map's pupil domain -- see :func:`map_pupil_radius`.
    angle_centre, angle_half : (float, float)
        The reduced-angle box.  A half-width of exactly 0.0 collapses that axis
        to a single node.
    check_angle_fn : callable or None
        ``f(x, y) -> (L, M)``, the caller's OWN angle field, used by G7 to
        measure the finished map against direct ray traces at the angles it
        will actually be evaluated at.  Without it G7 cannot run and the build
        is refused: an unmeasured map is not shipped.

    Returns
    -------
    CongruenceMap

    Raises
    ------
    AngleAwareRefusal
        On any guard.  The caller keeps the shipped screen.
    """
    nodes = tuple(_HMAP_NODES if nodes is None else nodes)
    deg = int(_HMAP_PUPIL_DEGREE if pupil_degree is None else pupil_degree)
    nlat = int(_HMAP_PUPIL_LATTICE if pupil_lattice is None else pupil_lattice)
    wavelength = float(wavelength)
    pupil_radius = float(pupil_radius)

    if not np.isfinite(pupil_radius) or pupil_radius <= 0.0:
        raise AngleAwareRefusal('G0 domain',
                                f'pupil radius {pupil_radius!r} is not usable')
    if r_ref == 0.0 or (isinstance(r_ref, float) and np.isnan(r_ref)):
        raise AngleAwareRefusal(
            'G1 reference radius',
            'the congruence states R == 0 (its own focus) or NaN, so no '
            'reference sphere exists to reduce the angle against')
    if check_angle_fn is None:
        raise AngleAwareRefusal(
            'G7 accuracy', 'no angle field was supplied to measure against')

    surfaces, n_exit = _traceable_surfaces(prescription, wavelength)

    hx, hy = float(angle_half[0]), float(angle_half[1])
    cx, cy = float(angle_centre[0]), float(angle_centre[1])
    if not (np.isfinite(hx) and np.isfinite(hy)
            and np.isfinite(cx) and np.isfinite(cy)):
        raise AngleAwareRefusal('G0 domain',
                                'the reduced-angle box is not finite')
    # G-degenerate: an axis with EXACTLY zero spread carries one node.  This is
    # an exact test on a computed half-width, not a tolerance -- an on-axis or
    # a purely collimated congruence produces 0.0 identically.
    nx = int(nodes[0]) if hx > 0.0 else 1
    ny = int(nodes[1]) if hy > 0.0 else 1

    # ---- the pupil lattice ------------------------------------------------
    t = np.linspace(-1.0, 1.0, nlat)
    PX, PY = np.meshgrid(pupil_radius * t, pupil_radius * t, indexing='ij')
    disc = (PX ** 2 + PY ** 2) <= pupil_radius ** 2
    px, py = PX.ravel(), PY.ravel()
    n_rays = 0

    # ---- the NORMAL-INCIDENCE arm (the differential's reference) ----------
    zx0, zy0, opl0, alive0 = _trace_characteristic(
        surfaces, n_exit, px, py, np.zeros_like(px), np.zeros_like(py),
        wavelength)
    n_rays += px.size
    if not bool(alive0.reshape(PX.shape)[disc].all()):
        raise AngleAwareRefusal(
            'G3 alive census',
            'the normal-incidence arm loses %d of %d rays inside the declared '
            'pupil, so the element cannot be characterised on the domain it '
            'declares' % (int((~alive0.reshape(PX.shape)[disc]).sum()),
                          int(disc.sum())))

    # ---- the node congruences --------------------------------------------
    ax = _cheb_nodes(nx)
    ay = _cheb_nodes(ny)
    ref_x, ref_y = _reference_sphere_gradient(r_ref, PX, PY)
    ref_x, ref_y = ref_x.ravel(), ref_y.ravel()

    chans = ('dopl', 'x_out', 'y_out', 'det_j')
    node_vals = np.empty((len(chans), nx, ny, px.size), dtype=float)
    det_min = np.inf
    det_max = -np.inf
    det_sign = 0
    for i, a in enumerate(ax):
        for j, b in enumerate(ay):
            thx = ref_x + cx + hx * a
            thy = ref_y + cy + hy * b
            xo, yo, opl, alive = _trace_characteristic(
                surfaces, n_exit, px, py, thx, thy, wavelength)
            n_rays += px.size
            am = alive.reshape(PX.shape)
            if not bool(am[disc].all()):
                raise AngleAwareRefusal(
                    'G3 alive census',
                    'node congruence (%d, %d) loses %d of %d rays inside the '
                    'declared pupil -- the domain has grown past the element'
                    % (i, j, int((~am[disc]).sum()), int(disc.sum())))
            node_vals[0, i, j] = opl - opl0
            node_vals[1, i, j] = xo
            node_vals[2, i, j] = yo
            # G2's Jacobian, from the TRACED landings on the launch lattice --
            # a reduction of rays already traced, not a new trace, and it is
            # basis-independent by construction (the BUILD_INVERSE_MAP S6.5a
            # lesson: an amplitude that depends on an interpolant choice is a
            # physics claim that is not one).
            gx = xo.reshape(PX.shape)
            gy = yo.reshape(PX.shape)
            d0 = float(PX[1, 0] - PX[0, 0])
            d1 = float(PY[0, 1] - PY[0, 0])
            dXdx, dXdy = np.gradient(gx, d0, d1)
            dYdx, dYdy = np.gradient(gy, d0, d1)
            det = dXdx * dYdy - dXdy * dYdx
            node_vals[3, i, j] = det.ravel()
            dd = det[disc]
            dd = dd[np.isfinite(dd)]
            if dd.size:
                sgn = 1 if float(np.median(dd)) > 0.0 else -1
                if det_sign == 0:
                    det_sign = sgn
                if sgn != det_sign or bool(np.any(dd * det_sign <= 0.0)):
                    raise AngleAwareRefusal(
                        'G2 Jacobian sign',
                        'det J changes sign on node congruence (%d, %d): the '
                        'entrance-to-exit map folds, so the characteristic is '
                        'not single-valued and no interpolant of it is a '
                        'wavefront' % (i, j))
                det_min = min(det_min, float(np.abs(dd).min()))
                det_max = max(det_max, float(np.abs(dd).max()))

    det_range = (det_max / det_min) if det_min > 0.0 else np.inf
    if not np.isfinite(det_range) or det_range > _HMAP_DETJ_MAXMIN:
        raise AngleAwareRefusal(
            'G2 Jacobian caustic',
            'det J dynamic range %.4g exceeds the %.4g cap -- the element is '
            'at or near a caustic on this domain'
            % (det_range, _HMAP_DETJ_MAXMIN))

    # ---- the pupil least-squares fit --------------------------------------
    sel = disc.ravel()
    ux = _to_unit(px[sel], 0.0, pupil_radius)
    uy = _to_unit(py[sel], 0.0, pupil_radius)
    Bx = _cheb_basis(ux, deg)
    By = _cheb_basis(uy, deg)
    A = (Bx[:, None, :] * By[None, :, :]).reshape((deg + 1) ** 2, -1).T
    if A.shape[0] < 4 * A.shape[1]:
        raise AngleAwareRefusal(
            'G6 conditioning',
            'the pupil lattice supplies %d samples for %d free coefficients '
            '(want >= 4x)' % (A.shape[0], A.shape[1]))
    rhs = node_vals[:, :, :, sel].reshape(-1, int(sel.sum())).T
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    resid = A @ sol - rhs
    fit_resid_waves = float(np.abs(resid[:, :nx * ny]).max()) / wavelength
    node_coef = sol.T.reshape(len(chans), nx, ny, deg + 1, deg + 1)
    node_coef = np.moveaxis(node_coef, (1, 2), (3, 4))   # -> (c, P, P, nx, ny)

    # ---- the angular Chebyshev interpolation ------------------------------
    Vx = _cheb_node_inverse(nx)
    Vy = _cheb_node_inverse(ny)
    coef = np.einsum('cijkl,mk,nl->cijmn', node_coef, Vx, Vy, optimize=True)
    coef = np.ascontiguousarray(coef)

    hmap = CongruenceMap(
        coef=coef, channels=chans, r_ref=r_ref, pupil_radius=pupil_radius,
        angle_centre=(cx, cy), angle_half=(hx, hy), nodes=(nx, ny),
        pupil_degree=deg, wavelength=wavelength, det_j_range=det_range,
        det_j_sign=det_sign, n_rays=n_rays,
        fit_residual_waves=fit_resid_waves, check_max_waves=float('nan'),
        key=key, name=name)

    # ---- G7: measure the finished map against DIRECT ray traces ------------
    chk = _measure_against_traces(hmap, surfaces, n_exit, check_angle_fn)
    hmap.check_max_waves = chk
    if not np.isfinite(chk) or chk > _HMAP_ACCEPT_WAVES:
        raise AngleAwareRefusal(
            'G7 accuracy',
            'the built map reads %.4e waves against direct ray traces at this '
            'call\'s own angles, outside the %.4e bar (lambda/100 with 3x)'
            % (chk, _HMAP_ACCEPT_WAVES))
    return hmap


def _box_probe_positions(r_pupil):
    """A rings x azimuths probe of the declared pupil DISC.

    Strictly finer than G7's own check set and containing it, so a box built
    here cannot fail to contain the angles G7 then measures at."""
    rr, aa = np.meshgrid(np.linspace(0.0, 1.0, 9),
                         np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False),
                         indexing='ij')
    return (float(r_pupil) * rr * np.cos(aa)).ravel(), \
        (float(r_pupil) * rr * np.sin(aa)).ravel()


def _measure_against_traces(hmap, surfaces, n_exit, angle_fn):
    """G7 -- the map's own error, in waves, against exact rays.

    Probes on rings x azimuths inside the declared pupil, at the CALLER's own
    angle field, none of which is a lattice point or a node angle.  This is the
    guard that carries the accuracy of the whole module: the node count, the
    pupil degree, the lattice size and the box pad are all inputs to a build
    whose output is then measured here and refused if it misses.  A parameter
    that is wrong shows up as a refusal, never as a quiet error."""
    rr, aa = np.meshgrid(np.asarray(_HMAP_CHECK_RINGS, dtype=float),
                         np.linspace(0.0, 2.0 * np.pi,
                                     _HMAP_CHECK_AZIMUTHS, endpoint=False),
                         indexing='ij')
    qx = (hmap.pupil_radius * rr * np.cos(aa)).ravel()
    qy = (hmap.pupil_radius * rr * np.sin(aa)).ravel()
    Lq, Mq = angle_fn(qx, qy)
    Lq = np.broadcast_to(np.asarray(Lq, dtype=float), qx.shape)
    Mq = np.broadcast_to(np.asarray(Mq, dtype=float), qx.shape)
    x1, y1, opl1, al1 = _trace_characteristic(
        surfaces, n_exit, qx, qy, Lq, Mq, hmap.wavelength)
    x0, y0, opl0, al0 = _trace_characteristic(
        surfaces, n_exit, qx, qy, np.zeros_like(qx), np.zeros_like(qy),
        hmap.wavelength)
    ok = al1 & al0
    if not bool(ok.any()):                                # pragma: no cover
        return float('inf')
    want = (opl1 - opl0)[ok]
    got = hmap.delta_opl(qx[ok], qy[ok], Lq[ok], Mq[ok])
    return float(np.abs(got - want).max()) / hmap.wavelength


# ---------------------------------------------------------------------------
# the cache -- chain-A key discipline
# ---------------------------------------------------------------------------
_HMAP_CACHE: Dict[str, CongruenceMap] = {}
_HMAP_CACHE_ORDER: list = []
_HMAP_CACHE_STATS = {'hits': 0, 'misses': 0}
_HMAP_CACHE_LOCK = threading.RLock()


def _hash_prescription(prescription):
    """A stable digest of everything about the element the trace can see.

    Walks the dict deterministically and hashes ndarray CONTENT, not
    ``repr``: a ``form_error`` map that changed in place must change the key,
    and a dict that merely re-ordered must not."""
    h = hashlib.sha256()

    def walk(obj):
        if isinstance(obj, dict):
            h.update(b'{')
            for k in sorted(obj, key=repr):
                h.update(repr(k).encode('ascii', 'backslashreplace'))
                h.update(b':')
                walk(obj[k])
            h.update(b'}')
        elif isinstance(obj, (list, tuple)):
            h.update(b'[')
            for v in obj:
                walk(v)
            h.update(b']')
        elif isinstance(obj, np.ndarray):
            h.update(b'#')
            h.update(str(obj.dtype).encode('ascii'))
            h.update(str(obj.shape).encode('ascii'))
            h.update(np.ascontiguousarray(obj).tobytes())
        else:
            h.update(repr(obj).encode('ascii', 'backslashreplace'))
    walk(prescription)
    return h.hexdigest()


def _hmap_key(prescription, wavelength, r_ref, pupil_radius, angle_centre,
              angle_half, nodes, pupil_degree, pupil_lattice):
    """SHA-256 over EVERYTHING the map depends on.

    The chain-A cache lesson (``docs/audits/FIX_D4_D6_D7_2026_08_06.md`` D6),
    restated for an in-process cache: **a key that names the CONFIGURATION and
    not the CONTENT is how a cache silently becomes a cache of something
    else.**  So the prescription enters by CONTENT digest (arrays included),
    and the wavelength, the reference radius, the pupil domain, the angular box
    and every build shape enter by exact ``repr`` of their float bits.  A
    prescription edited in place, a wavelength moved by one ulp, or a box that
    grew by a nanoradian all miss."""
    h = hashlib.sha256()
    h.update(b'hmap-v1|')
    h.update(_hash_prescription(prescription).encode('ascii'))
    for v in (wavelength, r_ref, pupil_radius, angle_centre[0],
              angle_centre[1], angle_half[0], angle_half[1]):
        h.update(b'|')
        h.update(float(v).hex().encode('ascii'))
    for v in (nodes[0], nodes[1], pupil_degree, pupil_lattice):
        h.update(b'|')
        h.update(str(int(v)).encode('ascii'))
    return h.hexdigest()


def _cache_get(key):
    with _HMAP_CACHE_LOCK:
        got = _HMAP_CACHE.get(key)
        if got is None:
            _HMAP_CACHE_STATS['misses'] += 1
        else:
            _HMAP_CACHE_STATS['hits'] += 1
            try:
                _HMAP_CACHE_ORDER.remove(key)
            except ValueError:                            # pragma: no cover
                pass
            _HMAP_CACHE_ORDER.append(key)
        return got


def _cache_put(key, hmap):
    with _HMAP_CACHE_LOCK:
        _HMAP_CACHE[key] = hmap
        if key in _HMAP_CACHE_ORDER:                      # pragma: no cover
            _HMAP_CACHE_ORDER.remove(key)
        _HMAP_CACHE_ORDER.append(key)
        while len(_HMAP_CACHE_ORDER) > _HMAP_CACHE_SIZE:
            _HMAP_CACHE.pop(_HMAP_CACHE_ORDER.pop(0), None)


def clear_congruence_map_cache():
    """Drop every cached map and reset the hit/miss counters."""
    with _HMAP_CACHE_LOCK:
        _HMAP_CACHE.clear()
        del _HMAP_CACHE_ORDER[:]
        _HMAP_CACHE_STATS['hits'] = 0
        _HMAP_CACHE_STATS['misses'] = 0


def congruence_map_cache_stats():
    """``{'hits': int, 'misses': int, 'size': int}``."""
    with _HMAP_CACHE_LOCK:
        return {'hits': _HMAP_CACHE_STATS['hits'],
                'misses': _HMAP_CACHE_STATS['misses'],
                'size': len(_HMAP_CACHE)}


# ---------------------------------------------------------------------------
# the call-site entry point
# ---------------------------------------------------------------------------
def _validate_guard_action(where):
    action = ANGLE_AWARE_LENS_GUARD
    if action not in ('silent', 'warn'):
        raise ValueError(
            f'{where}: ANGLE_AWARE_LENS_GUARD must be one of '
            f"('silent', 'warn') (got {action!r}).")
    return action


def _report_refusal(exc, where):
    """Report a refusal under :data:`ANGLE_AWARE_LENS_GUARD` and return.

    A refused build KEEPS THE SHIPPED SCREEN whatever this does, so nothing
    here can change a returned bit."""
    if _validate_guard_action(where) == 'warn':
        import warnings
        warnings.warn(
            f'{where}: the angle-aware element screen was REFUSED by '
            f'{exc.guard} -- {exc.detail}.  The call keeps the shipped '
            f'normal-incidence sag screen (the documented fail-before).  Set '
            f'lumenairy.elements._lens_hmap.ANGLE_AWARE_LENS_GUARD = '
            f"'silent' to stop reporting this.",
            UserWarning, stacklevel=3)


def angle_aware_delta_opl(prescription, wavelength, X, Y, L, M, r_ref,
                          angle_fn, *, nodes=None, pupil_degree=None,
                          pupil_lattice=None, hmap=None,
                          where='apply_real_lens'):
    """``(dOPL, CongruenceMap)`` for one call, or ``(None, None)`` on refusal.

    This is the whole consumer interface.  It resolves the pupil domain and the
    reduced-angle box from the CALL, hits (or fills) the cache, and evaluates.
    Every failure path returns ``(None, None)`` after reporting, so the caller
    has exactly one branch to write: use the correction or keep the shipped
    screen."""
    try:
        _validate_guard_action(where)
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        L = np.broadcast_to(np.asarray(L, dtype=float), X.shape)
        M = np.broadcast_to(np.asarray(M, dtype=float), X.shape)
        if not (np.all(np.isfinite(L)) and np.all(np.isfinite(M))):
            raise AngleAwareRefusal(
                'G4 angle field',
                'the carrier produced a non-finite direction cosine')

        r_half = float(np.max(np.hypot(X, Y))) if X.size else 0.0
        r_pupil = map_pupil_radius(prescription, r_half)
        if not np.isfinite(r_pupil) or r_pupil <= 0.0:
            raise AngleAwareRefusal(
                'G0 domain', 'the element declares no usable pupil radius')

        if hmap is None:
            inside = (X * X + Y * Y) <= r_pupil ** 2
            if not np.any(inside):
                raise AngleAwareRefusal(
                    'G0 domain',
                    'no sample of the grid lies inside the declared pupil')
            # THE BOX IS TAKEN OVER THE DISC, NOT OVER THE GRID.  A square
            # grid samples its inscribed circle densely and its corners once,
            # so a box read off the grid's own pixels does not contain the
            # angles the map is later asked for at intermediate azimuths --
            # including G7's own check rings, which is how this was found.
            # ``angle_fn`` is the carrier's analytic gradient and is defined
            # everywhere, so the disc probe costs no rays and makes the box a
            # property of the ELEMENT and the CARRIER rather than of the
            # caller's sampling (which is also what lets two grids share one
            # cached map).
            bx, by = _box_probe_positions(r_pupil)
            bL, bM = angle_fn(bx, by)
            brx, bry = _reference_sphere_gradient(r_ref, bx, by)
            sx = np.broadcast_to(np.asarray(bL, dtype=float), bx.shape) - brx
            sy = np.broadcast_to(np.asarray(bM, dtype=float), by.shape) - bry
            if not (np.all(np.isfinite(sx)) and np.all(np.isfinite(sy))):
                raise AngleAwareRefusal(
                    'G4 angle field',
                    'the carrier produced a non-finite direction cosine on '
                    'the declared pupil disc')
            lo = np.array([float(sx.min()), float(sy.min())])
            hi = np.array([float(sx.max()), float(sy.max())])
            centre = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo)
            half = half * (1.0 + _HMAP_BOX_PAD)
            # An axis whose spread is EXACTLY zero stays exactly zero: the pad
            # is relative, so a degenerate axis cannot be inflated into a
            # non-degenerate one by it.
            nodes_r = tuple(_HMAP_NODES if nodes is None else nodes)
            key = _hmap_key(prescription, wavelength, r_ref, r_pupil,
                            tuple(centre), tuple(half), nodes_r,
                            int(_HMAP_PUPIL_DEGREE if pupil_degree is None
                                else pupil_degree),
                            int(_HMAP_PUPIL_LATTICE if pupil_lattice is None
                                else pupil_lattice))
            hmap = _cache_get(key)
            if hmap is None:
                hmap = build_congruence_map(
                    prescription, wavelength, r_ref,
                    pupil_radius=r_pupil, angle_centre=tuple(centre),
                    angle_half=tuple(half), nodes=nodes,
                    pupil_degree=pupil_degree, pupil_lattice=pupil_lattice,
                    check_angle_fn=angle_fn,
                    name=str(prescription.get('name') or ''), key=key)
                _cache_put(key, hmap)
        return hmap.delta_opl(X, Y, L, M), hmap
    except AngleAwareRefusal as exc:
        _report_refusal(exc, where)
        return None, None


__all__ = ['ANGLE_AWARE_LENS', 'ANGLE_AWARE_LENS_GUARD', 'AngleAwareRefusal',
           'CongruenceMap', 'build_congruence_map', 'map_pupil_radius',
           'angle_aware_delta_opl', 'clear_congruence_map_cache',
           'congruence_map_cache_stats']
