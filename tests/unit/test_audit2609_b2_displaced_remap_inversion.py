"""WP-B2 (audit 2026-09-11, finding L9): the 2-D displaced remap is inverted on
its own STRUCTURED launch grid, and the launch lattice is a validated per-call
keyword.

WHAT CHANGED.  ``_apply_displaced_remap_2d`` used to rebuild the exit field by
Delaunay-triangulating the scattered exit points and interpolating onto the
field grid.  The launch fan is a REGULAR lattice, so the exit map is a smooth
curvilinear grid: it is now inverted on that grid (Newton on the bilinear
interpolant, seeded from the map's own global affine part) and the transported
amplitude and OPL are read at the launch coordinate that comes back.  The
scattered backend is retained as this one's oracle and reachable through
``_apply_displaced_remap_2d(interp_method='delaunay')``.

WHY THE LATTICE COULD NOT BE RAISED BEFORE, and what it actually was.  WP-A2
measured the image-plane mirror residual of a +d / -d decenter pair -- an EXACT
symmetry of the physics, so any residual is the model's own artefact -- at
7.9e-14 for n_side 181 and 4.1e-14 for 513, but 6.3e-03 at 512 and 7.4e-03 at
1025, and attributed the jump to QHull resolving near-degenerate cells
arbitrarily.  It is not the backend: the remap sampled the input envelope at
launch points with ``mode='constant'``, and the field axis
``(arange(N) - N/2) * dx`` reaches one whole sample further on the -x side than
on +x, so a ray landing in that one-pixel band carried the full envelope on one
side of the axis and nothing on the other.  Whether any ray lands there is a
property of the lattice pitch, which is why the effect looked like a lattice
instability.  ``TestTheInputWindowIsSymmetric`` pins the mechanism and its
fix; ``TestTheMirrorSymmetryDoesNotDependOnTheLattice`` pins the consequence
with BOTH backends, which is what shows the backend was never the cause.

TESTING_STANDARDS.  No wall-clock assertion anywhere in this file (S1): the
performance claims are pinned as OPERATION COUNTS (the default path builds no
triangulation) and as structure (the answer is lattice-independent where the
exit map is affine).  Every bar is either derived at runtime from the running
build's own measurement or carries its derivation and its measured values in
the comment beside it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as LR
from lumenairy.elements._lens_real import (
    _DISP_REMAP_2D_MIN_N_SIDE,
    _DISP_REMAP_2D_N_SIDE,
    _apply_displaced_remap,
    _apply_displaced_remap_2d,
    _build_displaced_ray_map,
    _build_displaced_ray_map_2d,
    _normalise_displaced_n_side,
    _remap2d_interp_structured,
    _residual_input_field,
    _warn_if_remap_lattice_smooths,
    apply_real_lens,
)
from lumenairy.elements.lens_config import LensConfig, LensNumerics

_WL = 1.31e-6
_Z_IMG = 49.162e-3

#: Model glasses for THIS module only; installed and removed by
#: ``tests/conftest.py::_module_glass_registry_guard``.
MODULE_GLASSES = {'_B2A': lambda wl: 1.5168}


# ---------------------------------------------------------------------------
# Fixtures -- the p10 decentered singlet (the acceptance geometry) and a
# tilted plate whose exit map is EXACTLY affine (the closed-form geometry).
# ---------------------------------------------------------------------------

def _singlet(dec=(0.0, 0.0), tilt=(0.0, 0.0)):
    """The f/5 singlet of ``test_niche_p10_transverse_walk_remap.py``."""
    return {'wavelength': _WL, 'aperture_diameter': 10e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': '_B2A', 'semi_diameter': 6e-3,
         'decenter': dec, 'tilt': tilt},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_B2A',
         'glass_after': 'air', 'semi_diameter': 6e-3}],
        'thicknesses': [5e-3], 'stop_index': 0}


def _tilted_plate(tx=4e-3, thickness=5e-3):
    """A plane-parallel plate whose FIRST face is tilted.

    A flat face has ONE normal, so every ray refracts through the same angle:
    the entrance -> exit map is a pure TRANSLATION and the OPL is exactly
    linear in the launch coordinate.  Both are degree-1 polynomials, which a
    bilinear interpolant reproduces exactly -- so the remap's answer on this
    element is exact, independent of the launch lattice, and comparable to a
    closed form.  The tilt also makes the element asymmetric, which is what
    routes the call to the 2-D remap.
    """
    return {'wavelength': _WL, 'aperture_diameter': 4e-3, 'surfaces': [
        {'radius': float('inf'), 'thickness': thickness,
         'glass_before': 'air', 'glass_after': '_B2A',
         'semi_diameter': 3e-3, 'tilt': (tx, 0.0)},
        {'radius': float('inf'), 'thickness': 0.0,
         'glass_before': '_B2A', 'glass_after': 'air',
         'semi_diameter': 3e-3}],
        'thicknesses': [thickness], 'stop_index': 0}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _axes(N, dx):
    x = (np.arange(N) - N / 2) * dx
    return np.meshgrid(x, x)


def _disp(E0, presc, dx, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            surface_model='displaced', **kw))


def _psf_intensity(E_exit, dx, z=_Z_IMG):
    E = la.angular_spectrum_propagate(E_exit.astype(np.complex128), z, _WL, dx)
    return np.abs(E) ** 2


def _mirror_x(I):
    """Mirror an image about the field axis' own centre.

    ``x = (arange(N) - N/2) * dx`` puts index j at -x[N-j], so the mirror is
    the reversed array rolled by one -- the convention
    ``test_niche_p10_transverse_walk_remap.py`` uses.
    """
    return np.roll(I[:, ::-1], 1, axis=1)


def _mirror_residual(N=512, dx=8e-6, d=0.6e-3, **kw):
    """Relative L2 of ``I(+d) - mirror(I(-d))`` on the p10 image plane."""
    E0 = _gauss(N, dx, 3e-3)
    Ip = _psf_intensity(_disp(E0, _singlet(dec=(d, 0.0)), dx, **kw), dx)
    Im = _psf_intensity(_disp(E0, _singlet(dec=(-d, 0.0)), dx, **kw), dx)
    return float(np.linalg.norm(Ip - _mirror_x(Im)) / np.linalg.norm(Ip))


# ---------------------------------------------------------------------------
# A FROZEN copy of the remap's launch-envelope assembly, with the symmetric
# input window switchable.  It exists to reproduce the PRE-FIX path in process
# (the fail-before), so it is deliberately a copy rather than a call; the copy
# is checked against the shipped path on every run by
# ``test_the_windowed_arm_reproduces_the_shipped_call_bit_for_bit``, so it
# cannot silently drift into testing something else.
# ---------------------------------------------------------------------------

def _remap_exit_field(E_in, ray_map_2d, dx, dy, window):
    from scipy.ndimage import distance_transform_edt, map_coordinates
    X0, Y0, XO, YO, OPL, ALIVE, dstep, r_ap = ray_map_2d
    Ny, Nx = E_in.shape
    k0 = 2.0 * np.pi / _WL
    F = _residual_input_field(E_in, None, _WL)
    cx = X0.ravel() / dx + Nx / 2.0
    cy = Y0.ravel() / dy + Ny / 2.0
    amp_in = (map_coordinates(F.real, [cy, cx], order=1, mode='constant',
                              cval=0.0)
              + 1j * map_coordinates(F.imag, [cy, cx], order=1,
                                     mode='constant', cval=0.0)
              ).reshape(X0.shape)
    if window:
        win = ((np.abs(X0) <= (Nx / 2.0 - 1.0) * dx)
               & (np.abs(Y0) <= (Ny / 2.0 - 1.0) * dy))
        amp_in = np.where(win, amp_in, 0.0)
    XOf, YOf = XO.copy(), YO.copy()
    dXO_dy, dXO_dx = np.gradient(XOf, dstep, dstep)
    dYO_dy, dYO_dx = np.gradient(YOf, dstep, dstep)
    det = dXO_dx * dYO_dy - dXO_dy * dYO_dx
    amp_out = amp_in / np.sqrt(np.maximum(np.abs(det), 1e-30))
    in_ap = (X0 * X0 + Y0 * Y0) <= (r_ap * (1.0 + 1e-9)) ** 2
    m = ALIVE & in_ap & np.isfinite(amp_out) & (np.abs(amp_in) > 0.0)
    Xg, Yg = _axes(Nx, dx)
    good = ALIVE & np.isfinite(amp_out) & np.isfinite(OPL)
    amp_src = np.where(good, amp_out, 0.0).astype(np.complex128)
    opl_src = np.asarray(OPL, dtype=np.float64)
    if not bool(good.all()):
        opl_src = opl_src[tuple(distance_transform_edt(
            ~good, return_distances=False, return_indices=True))]
    amp_grid, opl_grid = _remap2d_interp_structured(
        XOf, YOf, amp_src, opl_src, m, float(X0[0, 0]), float(Y0[0, 0]),
        float(dstep), float(r_ap), Xg, Yg)
    opl_ref = float(np.median(OPL[m]))
    return np.asarray(amp_grid * np.exp(1j * k0 * (opl_grid - opl_ref)),
                      dtype=np.complex128)


def _ray_map(presc, n_side):
    return _build_displaced_ray_map_2d(
        presc['surfaces'], presc['thicknesses'], _WL,
        presc['aperture_diameter'] / 2.0, n_side=n_side)


def _depistoned(a, b):
    """``a`` with the global phase that best matches ``b`` removed.

    The remap references its exit phase to ``median(OPL)`` over the RETAINED
    rays, so two calls that retain different rays -- two launch lattices, or
    two backends -- differ by a global piston BY CONSTRUCTION.  A piston is not
    an error of the field, so every cross-call comparison here removes it
    first, and none of them claims anything about it.
    """
    z = np.vdot(b, a)
    if z == 0:
        return a
    return a * np.conj(z / abs(z))


def _affine_fit(U, V, F):
    """Least-squares ``F ~ c0 U + c1 V + c2`` and its residual."""
    A = np.column_stack([U.ravel(), V.ravel(), np.ones(U.size)])
    c, *_ = np.linalg.lstsq(A, np.asarray(F).ravel(), rcond=None)
    res = float(np.max(np.abs(A @ c - np.asarray(F).ravel())))
    return c, res


# ===========================================================================
# 1.  The input window -- the mechanism WP-A2 mis-attributed to QHull
# ===========================================================================

class TestTheInputWindowIsSymmetric:

    @pytest.mark.parametrize('n_side', [181, 257, 512, 513, 1025])
    def test_the_sampled_envelope_is_mirror_symmetric_at_every_lattice(
            self, n_side):
        """The envelope the remap carries must be the same on both sides of the
        axis, whatever the launch pitch.

        Bar: exact mirror equality of the WINDOWED sample to within the
        interpolation's own rounding.  The sampling coordinate is
        ``x/dx + N/2``, and ``fl(q + N/2)`` and ``fl(N/2 - q)`` round
        independently, so the two sides can differ by an ULP of the weights --
        MEASURED 4.4e-16 of the peak across these five lattices (2026-09-13),
        against a 1e-12 bar and an UNWINDOWED difference of 6.3e-01 (below).
        """
        N, dx = 512, 8e-6
        E0 = _gauss(N, dx, 3e-3)
        rm = _ray_map(_singlet(dec=(0.6e-3, 0.0)), n_side)
        a = _sampled_envelope(E0, rm, dx, window=True)
        assert np.max(np.abs(a - a[:, ::-1])) <= 1e-12 * np.max(np.abs(a))

    @pytest.mark.parametrize('n_side,lands_in_band', [
        (181, False), (257, False), (512, True), (513, False), (1025, True)])
    def test_fail_before_the_unwindowed_sample_is_not(self, n_side,
                                                      lands_in_band):
        """FAIL-BEFORE, constructed through the running build's own geometry.

        Without the window the sample is asymmetric by the whole envelope value
        at the grid edge -- but only at the lattices whose rays land in the
        one-pixel band ``(x[-1], x[-1] + dx]`` that the field axis holds on the
        -x side only.  The expectation is DERIVED here from the lattice, not
        quoted: a launch coordinate in that band is what breaks it.
        """
        N, dx = 512, 8e-6
        E0 = _gauss(N, dx, 3e-3)
        rm = _ray_map(_singlet(dec=(0.6e-3, 0.0)), n_side)
        X0 = rm[0]
        hi, lo = (N / 2.0 - 1.0) * dx, (N / 2.0) * dx
        in_band = bool(np.any((np.abs(X0[0]) > hi * (1 + 1e-12))
                              & (np.abs(X0[0]) <= lo)))
        assert in_band == lands_in_band, (
            f'n_side={n_side}: the launch axis reaching the asymmetric band '
            f'reads {in_band}, the parametrisation says {lands_in_band}, so '
            f'the fixture moved.')
        a = _sampled_envelope(E0, rm, dx, window=False)
        rel = float(np.max(np.abs(a - a[:, ::-1])) / np.max(np.abs(a)))
        if lands_in_band:
            # the envelope at the grid edge, exp(-(N dx/2 / w0)^2) ~ 0.63 --
            # DERIVED from the fixture, not a quoted number
            edge = float(np.exp(-((N / 2.0 * dx) / 3e-3) ** 2))
            assert rel > 0.5 * edge, (n_side, rel, edge)
        else:
            assert rel <= 1e-12


def _sampled_envelope(E_in, ray_map_2d, dx, window):
    """The input envelope as the remap samples it at the launch lattice."""
    from scipy.ndimage import map_coordinates
    X0, Y0 = ray_map_2d[0], ray_map_2d[1]
    Ny, Nx = E_in.shape
    F = _residual_input_field(E_in, None, _WL)
    cx = X0.ravel() / dx + Nx / 2.0
    cy = Y0.ravel() / dx + Ny / 2.0
    a = (map_coordinates(F.real, [cy, cx], order=1, mode='constant', cval=0.0)
         + 1j * map_coordinates(F.imag, [cy, cx], order=1, mode='constant',
                                cval=0.0)).reshape(X0.shape)
    if window:
        win = ((np.abs(X0) <= (Nx / 2.0 - 1.0) * dx)
               & (np.abs(Y0) <= (Ny / 2.0 - 1.0) * dx))
        a = np.where(win, a, 0.0)
    return a


# ===========================================================================
# 2.  The acceptance test -- the p10 symmetry trio, as a derived envelope
# ===========================================================================

class TestTheMirrorSymmetryDoesNotDependOnTheLattice:

    def test_the_trio_holds_at_the_lattices_that_used_to_break_it(self):
        """The p10 mirror / centroid / EE80 trio, swept over the launch
        lattice, with the envelope DERIVED from this build's own reading at the
        shipped 181.

        The bar is 1000x the residual this build measures at n_side=181 -- the
        lattice the shipped default sat next to and that was measured as good.
        Gap on both sides (TESTING_STANDARDS 5): measured 2026-09-13 the four
        swept residuals span 5.6e-14..8.5e-14 against a 7.5e-14 reference,
        i.e. within 2x of each other, while the pre-fix readings at 512 and
        1025 were 6.3e-03 and 7.4e-03 -- eleven decades above.  A 1000x bar
        therefore sits three decades above the spread and eight below the
        failure it must catch.
        """
        base = _mirror_residual(displaced_n_side=181)
        assert base < 1e-10, base      # the reference itself must be sane
        for n_side in (257, 512, 513, 1025):
            got = _mirror_residual(displaced_n_side=n_side)
            assert got <= 1000.0 * base, (n_side, got, base)

    def test_the_stability_is_not_the_backend(self, monkeypatch):
        """Both backends are stable at a lattice that used to break, which is
        what shows the triangulation was never the cause.

        Same derived envelope as above, applied to the SCATTERED backend the
        pre-fix path used.  If the instability had been QHull's, this arm would
        fail where the structured one passes.
        """
        import functools
        orig = LR._apply_displaced_remap_2d
        monkeypatch.setattr(
            LR, '_apply_displaced_remap_2d',
            functools.partial(orig, interp_method='delaunay'))
        base = _mirror_residual(displaced_n_side=181)
        got = _mirror_residual(displaced_n_side=512)
        assert got <= 1000.0 * base, (got, base)

    def test_fail_before_the_unwindowed_assembly_breaks_it_at_512(self):
        """FAIL-BEFORE at the field level: the same assembly without the
        symmetric input window loses the mirror symmetry at n_side=512 by
        orders of magnitude, and keeps it at 181.

        Scored on the EXIT field (not the image plane) so the demonstration
        does not depend on the propagator, and the ratio -- not an absolute
        level -- is the claim.
        """
        N, dx, d = 512, 8e-6, 0.6e-3
        E0 = _gauss(N, dx, 3e-3)

        def resid(n_side, window):
            out = []
            for sgn in (+1, -1):
                p = _singlet(dec=(sgn * d, 0.0))
                out.append(_remap_exit_field(E0, _ray_map(p, n_side), dx, dx,
                                             window))
            a, b = np.abs(out[0]), np.abs(out[1])
            return float(np.max(np.abs(a - _mirror_x(b))) / np.max(a))

        assert resid(181, False) < 1e-9                  # 181 never landed
        assert resid(512, True) < 1e-9                   # the fix holds
        assert resid(512, False) > 1e-3 * resid(512, True) * 1e6, (
            resid(512, False), resid(512, True))
        assert resid(512, False) > 1e-2                  # the whole rim

    def test_the_windowed_arm_reproduces_the_shipped_call_bit_for_bit(self):
        """The frozen copy above must BE the shipped assembly, or the
        fail-before is testing something the library does not do."""
        N, dx = 256, 12e-6
        E0 = _gauss(N, dx, 1.2e-3)
        p = _singlet(dec=(0.6e-3, 0.0))
        rm = _ray_map(p, 181)
        mine = _remap_exit_field(E0, rm, dx, dx, window=True)
        theirs = _apply_displaced_remap_2d(E0.copy(), rm, _WL, dx, dx)
        assert np.array_equal(mine, np.asarray(theirs))


# ===========================================================================
# 3.  The inversion is exact where the map is exactly invertible
# ===========================================================================

class TestTheStructuredInversionOnAnAffineMap:
    """A tilted plane-parallel plate maps the pupil AFFINELY -- a flat face has
    one normal, so every ray refracts through the same angle, and the in-glass
    path shortens linearly with the tilted entry height -- with an exactly
    linear OPL.  Degree 1 in the launch coordinate, so the bilinear interpolant
    and its inverse are both exact and the remap has NO discretisation error at
    all.  That makes this the one fixture where an ABSOLUTE correctness bar is
    available, instead of a convergence rate."""

    def test_the_traced_map_is_affine_which_is_what_makes_this_a_closed_form(
            self):
        rm = _ray_map(_tilted_plate(), 181)
        X0, Y0, XO, YO, OPL = rm[0], rm[1], rm[2], rm[3], rm[4]
        span = float(np.ptp(X0))
        for name, F in (('x_out', XO), ('y_out', YO), ('OPL', OPL)):
            _, res = _affine_fit(X0, Y0, F)
            # 1e-9 of the launch span: the trace itself is a Newton
            # intersection, so its own rounding is the floor (measured 2e-18 m
            # on x_out, 4e-19 m on OPL, 2026-09-13)
            assert res <= 1e-9 * span, (name, res, span)

    def test_the_exit_field_matches_the_closed_form(self):
        """``E_out = E_in(M^-1 (r - b)) / sqrt(|det M|) * exp(i k0 OPL)`` with
        ``M``, ``b`` and the OPL ramp read off the traced map itself.

        The input is an AFFINE function of position, which bilinear
        interpolation reproduces exactly, so nothing in the chain has
        discretisation error and the only residue is float rounding.  Bar 1e-11
        of peak in amplitude and 1e-8 rad in phase; measured 2026-09-13 at
        4e-16 and 3e-11.
        """
        N, dx = 256, 16e-6
        p = _tilted_plate()
        X, Y = _axes(N, dx)
        ax, ay = 0.3 / (N * dx), 0.2 / (N * dx)
        E0 = (1.0 + ax * X + ay * Y).astype(np.complex128)
        rm = _ray_map(p, 181)
        X0, Y0, XO, YO, OPL = rm[0], rm[1], rm[2], rm[3], rm[4]
        cx, _ = _affine_fit(X0, Y0, XO)
        cy, _ = _affine_fit(X0, Y0, YO)
        co, _ = _affine_fit(X0, Y0, OPL)
        M = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
        Minv = np.linalg.inv(M)
        px, py = X - cx[2], Y - cy[2]
        U = Minv[0, 0] * px + Minv[0, 1] * py
        V = Minv[1, 0] * px + Minv[1, 1] * py
        want = ((1.0 + ax * U + ay * V)
                / np.sqrt(abs(float(np.linalg.det(M))))
                * np.exp(1j * (2.0 * np.pi / _WL) * (co[0] * U + co[1] * V)))

        E = np.asarray(_apply_displaced_remap_2d(E0.copy(), rm, _WL, dx, dx))
        # interior only: the pupil edge is where the aperture cut lives, and
        # the cut is not part of the affine claim
        core = (np.abs(E) > 0) & (np.hypot(X, Y)
                                  < 0.7 * (p['aperture_diameter'] / 2.0))
        assert int(core.sum()) > 1000, int(core.sum())
        got = _depistoned(E[core], want[core])
        pk = float(np.max(np.abs(want[core])))
        amp_err = float(np.max(np.abs(np.abs(got) - np.abs(want[core]))))
        assert amp_err <= 1e-11 * pk, amp_err
        ph_err = float(np.max(np.abs(np.angle(
            got * np.conj(want[core])))))
        assert ph_err <= 1e-8, ph_err

    def test_the_answer_does_not_move_with_the_launch_lattice(self):
        """On an affine map the lattice carries no information, so raising it
        must change nothing beyond rounding (and the piston the model's own
        ``median(OPL)`` reference moves -- see :func:`_depistoned`).

        This is the structural form of "the lattice is a RESOLUTION knob, not a
        source of arbitrary variation" -- the property the pre-fix path did not
        have, where changing 513 to 512 moved the answer by 0.6 of the peak.

        The bar is DERIVED from the arithmetic, not chosen: the exit phase is
        ``k0 * OPL`` with an OPL of order 7.5 mm, so one float64 ULP of OPL is
        already ``k0 * eps * |OPL|`` = 8e-12 rad of phase, and a handful of
        them is the floor of any two calls that sum the ray path in a different
        order.  Ten ULP is the bar; measured 1.8e-11 over 91 / 512 / 513,
        2026-09-13, against a failure this test exists to catch that is ten
        decades above it.
        """
        N, dx = 256, 16e-6
        p = _tilted_plate()
        X, _ = _axes(N, dx)
        E0 = (1.0 + 0.3 * X / (N * dx)).astype(np.complex128)
        rm181 = _ray_map(p, 181)
        ref = np.asarray(_apply_displaced_remap_2d(
            E0.copy(), rm181, _WL, dx, dx))
        pk = float(np.max(np.abs(ref)))
        floor = (10.0 * (2.0 * np.pi / _WL) * float(np.finfo(float).eps)
                 * float(np.max(np.abs(rm181[4]))))
        assert floor < 1e-9, floor      # still nine decades under the failure
        for n_side in (91, 512, 513):
            got = np.asarray(_apply_displaced_remap_2d(
                E0.copy(), _ray_map(p, n_side), _WL, dx, dx))
            both = (np.abs(ref) > 0) & (np.abs(got) > 0)
            d = np.max(np.abs(_depistoned(got[both], ref[both]) - ref[both]))
            assert d <= floor * pk, (n_side, d, floor * pk)


# ===========================================================================
# 4.  Oracle-refereed: the two backends against the refined-lattice limit
# ===========================================================================

class TestAgainstTheRefinedLatticeLimit:
    """Byte-identity with the shipped path is NOT achievable and must not be
    claimed: barycentric interpolation over the exit triangulation and bilinear
    interpolation in launch space are different O(h^2) approximations of the
    same map.  What IS checkable is that both converge to the same limit, at
    second order, and which of the two is closer on the way."""

    @staticmethod
    def _field(p, E0, dx, n_side, method):
        return np.asarray(_apply_displaced_remap_2d(
            E0.copy(), _ray_map(p, n_side), _WL, dx, dx,
            interp_method=method))

    def test_both_backends_converge_at_second_order_to_one_answer(self):
        """Richardson on the launch pitch: halving it must quarter the
        difference to the refined answer, for BOTH backends, and the two must
        agree in the limit.

        Bar: a convergence ORDER between 1.5 and 2.5 (second order with room
        for the fixture's own curvature), and a backend-to-backend difference
        at the finest lattice below the coarsest lattice's own error.  Derived
        entirely from this run; nothing quoted.
        """
        N, dx = 256, 12e-6
        p = _singlet(dec=(0.6e-3, 0.0))
        E0 = _gauss(N, dx, 0.9e-3)
        fine = self._field(p, E0, dx, 1025, 'structured')
        core = np.abs(fine) > 0.2 * np.max(np.abs(fine))
        assert int(core.sum()) > 500, int(core.sum())

        def err(n_side, method):
            g = self._field(p, E0, dx, n_side, method)[core]
            return float(np.sqrt(np.mean(
                np.abs(_depistoned(g, fine[core]) - fine[core]) ** 2)))

        for method in ('structured', 'delaunay'):
            e1 = err(129, method)
            e2 = err(257, method)
            order = np.log2(e1 / e2)
            assert 1.5 <= order <= 2.5, (method, e1, e2, order)
        s513 = self._field(p, E0, dx, 513, 'structured')[core]
        d513 = self._field(p, E0, dx, 513, 'delaunay')[core]
        d_fine = float(np.sqrt(np.mean(
            np.abs(_depistoned(d513, s513) - s513) ** 2)))
        assert d_fine < err(129, 'structured'), d_fine

    def test_the_structured_backend_leaves_no_holes_in_the_illuminated_pupil(
            self):
        """The triangulation's hull ends at the outermost retained exit point,
        so a grid-truncated input comes back with a ring of exactly-zero pixels
        INSIDE the illuminated pupil; inverting the map instead cuts the
        aperture on the launch coordinate and has none.

        Counted, not timed.  The bar is zero holes for the structured backend
        and more than zero for the scattered one -- a two-sided claim, so a
        fixture that stopped exercising the rim would fail the second arm
        rather than silently pass the first.
        """
        N, dx = 384, 10e-6
        p = _singlet(dec=(0.6e-3, 0.0))
        E0 = _gauss(N, dx, 3e-3)              # truncated by the grid at 0.63
        s = self._field(p, E0, dx, 181, 'structured')
        d = self._field(p, E0, dx, 181, 'delaunay')
        lit = np.abs(s) > 0.2 * np.max(np.abs(s))
        assert int(np.count_nonzero(d[lit] == 0)) > 0
        assert int(np.count_nonzero(s[lit] == 0)) == 0

    def test_the_structured_backend_transmits_at_least_as_much_power(self):
        """Same rim, in energy: the remap is a lossless geometric transfer
        inside the aperture, so power dropped at the hull is power lost."""
        N, dx = 384, 10e-6
        p = _singlet(dec=(0.6e-3, 0.0))
        E0 = _gauss(N, dx, 3e-3)
        ps = float((np.abs(self._field(p, E0, dx, 181, 'structured')) ** 2
                    ).sum())
        pd = float((np.abs(self._field(p, E0, dx, 181, 'delaunay')) ** 2).sum())
        pin = float((np.abs(E0) ** 2).sum())
        assert ps > pd
        assert ps <= pin * (1.0 + 1e-9)


# ===========================================================================
# 5.  Performance, pinned as OPERATION COUNTS (TESTING_STANDARDS S1)
# ===========================================================================

class TestTheDefaultPathBuildsNoTriangulation:

    def test_no_qhull_on_the_default_path_and_one_on_the_legacy_arm(self,
                                                                    monkeypatch):
        """The structural form of the cost claim: the default path constructs
        no ``LinearNDInterpolator`` at all, where the scattered arm constructs
        one per call.  A count, not a clock, so it says the same thing on every
        machine."""
        import scipy.interpolate as si
        built = []
        real = si.LinearNDInterpolator

        class Counting(real):
            def __init__(self, *a, **k):
                built.append(1)
                super().__init__(*a, **k)

        monkeypatch.setattr(si, 'LinearNDInterpolator', Counting)
        N, dx = 192, 16e-6
        p = _singlet(dec=(0.6e-3, 0.0))
        E0 = _gauss(N, dx, 1.2e-3)
        rm = _ray_map(p, 91)
        _apply_displaced_remap_2d(E0.copy(), rm, _WL, dx, dx)
        assert built == []
        _apply_displaced_remap_2d(E0.copy(), rm, _WL, dx, dx,
                                  interp_method='delaunay')
        assert len(built) == 1

    def test_the_inversion_retires_the_grid_in_a_bounded_number_of_sweeps(
            self, monkeypatch):
        """The affine seed plus the residual bar must retire the grid in a
        handful of Newton sweeps, or the loop is running on its cap.

        Counted as ``map_coordinates`` calls, which is what a sweep costs.  Bar
        derived from the algorithm: the seed read, at most 4 sweeps of 6 reads,
        the final residual and cell, and 3 reads of the transported fields --
        MEASURED 2026-09-13: the p10 singlet retires in 2 full sweeps + a 3 %
        tail at every lattice from 181 to 1025, i.e. 16 calls against this
        bar of 30.
        """
        import scipy.ndimage as sn
        calls = []
        real = sn.map_coordinates

        def counting(*a, **k):
            calls.append(1)
            return real(*a, **k)

        monkeypatch.setattr(sn, 'map_coordinates', counting)
        N, dx = 192, 16e-6
        p = _singlet(dec=(0.6e-3, 0.0))
        E0 = _gauss(N, dx, 1.2e-3)
        _apply_displaced_remap_2d(E0.copy(), _ray_map(p, 181), _WL, dx, dx)
        assert len(calls) <= 30, len(calls)


# ===========================================================================
# 6.  The public keyword
# ===========================================================================

class TestDisplacedNSideValidation:

    @pytest.mark.parametrize('bad', [0, 1, 2, -5, 2.5, np.nan, np.inf,
                                     '257', None.__class__, True, (257,)])
    def test_refusals_carry_the_conventions_prefix(self, bad):
        if bad is None:
            pytest.skip('None is the documented default')
        with pytest.raises(ValueError) as exc:
            _normalise_displaced_n_side(bad)
        msg = str(exc.value)
        assert msg.startswith('apply_real_lens:'), msg[:120]
        assert 'displaced_n_side' in msg

    @pytest.mark.parametrize('good,want', [
        (None, None), (3, 3), (257, 257), (np.int64(513), 513), (181.0, 181)])
    def test_accepted_values_normalise_to_an_int(self, good, want):
        assert _normalise_displaced_n_side(good) == want

    def test_the_floor_is_the_module_constant(self):
        assert _normalise_displaced_n_side(_DISP_REMAP_2D_MIN_N_SIDE) == \
            _DISP_REMAP_2D_MIN_N_SIDE
        with pytest.raises(ValueError):
            _normalise_displaced_n_side(_DISP_REMAP_2D_MIN_N_SIDE - 1)

    @pytest.mark.parametrize('kw', [
        dict(surface_model='thin'),
        dict(surface_model='displaced'),                       # symmetric
        dict(surface_model='displaced', displaced_obliquity='pointwise'),
    ])
    def test_a_call_that_does_not_run_the_remap_refuses_it(self, kw):
        """A setting the call would DISCARD is refused, naming why -- the
        class of defect the A16 config work exists to close."""
        N, dx = 64, 40e-6
        E0 = _gauss(N, dx, 0.8e-3)
        p = _singlet() if 'displaced' in kw.get('surface_model', '') \
            else _singlet()
        if kw.get('displaced_obliquity') == 'pointwise':
            p = _singlet(dec=(0.4e-3, 0.0))
        with pytest.raises(ValueError, match='displaced_n_side'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                apply_real_lens(E0, prescription=p, wavelength=_WL, dx=dx,
                                displaced_n_side=257, **kw)

    def test_the_remap_path_accepts_it_and_uses_it(self):
        """Two different lattices must give two different fields; the same
        lattice as the default must give the default's field."""
        N, dx = 128, 24e-6
        E0 = _gauss(N, dx, 1.0e-3)
        p = _singlet(dec=(0.4e-3, 0.0))
        a = _disp(E0, p, dx)
        b = _disp(E0, p, dx, displaced_n_side=_DISP_REMAP_2D_N_SIDE)
        c = _disp(E0, p, dx, displaced_n_side=91)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestLensNumericsCarriesTheField:

    def test_round_trip(self):
        cfg = LensConfig.from_kwargs(displaced_n_side=513)
        assert cfg.numerics.displaced_n_side == 513
        assert cfg.to_kwargs() == {'displaced_n_side': 513}
        assert LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg

    def test_the_default_matches_the_signature(self):
        import inspect
        sig = inspect.signature(apply_real_lens).parameters
        assert sig['displaced_n_side'].default is None
        assert LensNumerics().displaced_n_side is None

    def test_post_init_refuses_a_bad_value_with_the_class_prefix(self):
        with pytest.raises(ValueError) as exc:
            LensNumerics(displaced_n_side=2)
        assert str(exc.value).startswith('LensNumerics:')
        assert 'displaced_n_side' in str(exc.value)

    def test_the_config_object_reaches_the_model(self):
        N, dx = 128, 24e-6
        E0 = _gauss(N, dx, 1.0e-3)
        p = _singlet(dec=(0.4e-3, 0.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            via_cfg = np.asarray(apply_real_lens(
                E0, prescription=p, wavelength=_WL, dx=dx,
                surface_model='displaced',
                numerics=LensNumerics(displaced_n_side=91)))
        assert np.array_equal(via_cfg, _disp(E0, p, dx, displaced_n_side=91))


# ===========================================================================
# 7.  The lattice is a RESOLUTION knob -- and the warning says so
# ===========================================================================

class TestTheLatticeSetsTheTransverseResolution:

    def test_contrast_transfer_collapses_in_launch_samples_per_period(self):
        """The remap smooths input structure finer than the LAUNCH pitch, so a
        ripple at a fixed number of launch samples per period must come back at
        the same contrast whatever the lattice -- and doubling the lattice must
        halve the resolved period.

        Bar: the two readings at 7.2 launch samples/period agree within 10 %
        (measured 0.941 and 0.923 at n_side 181 and 361, 2026-09-13), and the
        transfer is monotone in samples/period.  Derived at runtime; the
        absolute contrasts are not pinned.
        """
        N, dx = 1024, 4e-6
        p = _singlet(dec=(0.5e-3, 0.0))
        X, Y = _axes(N, dx)
        w0, band = 1.2e-3, 0.8e-3
        x = (np.arange(N) - N / 2) * dx

        def transfer(n_side, period):
            E0 = ((1.0 + 0.5 * np.cos(2 * np.pi * X / period))
                  * np.exp(-(X ** 2 + Y ** 2) / w0 ** 2)).astype(np.complex128)
            rm = _ray_map(p, n_side)
            E = _apply_displaced_remap_2d(E0.copy(), rm, _WL, dx, dx)
            return _ripple_contrast(E, x, band) / _ripple_contrast(E0, x, band)

        pitch181 = 2 * (p['aperture_diameter'] / 2.0) / 180
        pitch361 = 2 * (p['aperture_diameter'] / 2.0) / 360
        t_a = transfer(181, 7.2 * pitch181)
        t_b = transfer(361, 7.2 * pitch361)
        assert abs(t_a - t_b) < 0.10 * max(t_a, t_b), (t_a, t_b)
        # monotone: fewer samples per period -> less contrast
        t_c = transfer(181, 3.6 * pitch181)
        assert t_c < t_a, (t_c, t_a)

    def test_the_warning_states_the_lattice_that_would_clear_the_bar(self):
        """The message must be actionable: it names ``displaced_n_side=n`` and
        that ``n`` must actually silence the warning."""
        r_max, dx = 5e-3, 8e-6
        with pytest.warns(RuntimeWarning) as rec:
            _warn_if_remap_lattice_smooths(r_max, dx, dx, 181)
        msg = str(rec[0].message)
        assert 'displaced_n_side=' in msg
        n = int(msg.split('displaced_n_side=')[1].split()[0])
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _warn_if_remap_lattice_smooths(r_max, dx, dx, n)     # silent now

    def test_a_lattice_at_or_below_twice_the_field_pitch_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _warn_if_remap_lattice_smooths(5e-3, 40e-6, 40e-6, 181)


def _ripple_contrast(E, x, band):
    a = np.abs(E[E.shape[0] // 2])
    m = np.abs(x) < band
    a, xx = a[m], x[m]
    res = a - np.polyval(np.polyfit(xx, a, 6), xx)
    return float(np.sqrt(2.0) * np.std(res) / np.mean(a))

# ===========================================================================
# 8.  VERIFY-WP-B2 -- the same input window in the 1-D symmetric remap
#
# WP-B2 recorded this as deferred on the reading that "it is rotationally
# symmetric, so no fixture in the suite exercises a mirror pair through it".
# It does not need a mirror pair: a CENTRED input through a ROTATIONALLY
# SYMMETRIC element is enough, because the remap reads the input at the
# ENTRANCE height ``X * scale`` with ``scale = h_in / r_out > 1`` for a
# converging element -- so the read runs off the +x end of
# ``(arange(N) - N/2) * dx`` while its mirror, one whole sample further out on
# -x, is still on the grid and returns the full envelope.
# ===========================================================================

def _sym_singlet(ap=6e-3):
    """A ROTATIONALLY SYMMETRIC thick singlet: no decenter, no tilt, no
    ``sag_callable``, so ``displaced_mode='remap'`` runs the 1-D remap."""
    return {'wavelength': _WL, 'aperture_diameter': ap, 'surfaces': [
        {'radius': 42.5e-3, 'thickness': 4.2e-3, 'glass_before': 'air',
         'glass_after': '_B2A', 'semi_diameter': ap / 2 * 1.2},
        {'radius': -63.0e-3, 'thickness': 0.0, 'glass_before': '_B2A',
         'glass_after': 'air', 'semi_diameter': ap / 2 * 1.2}],
        'thicknesses': [4.2e-3], 'stop_index': 0}


def _paired(A):
    """Drop row/column 0.

    ``x = (arange(N) - N/2) * dx`` puts index 0 at ``-N/2 dx``, whose mirror
    ``+N/2 dx`` is NOT on the grid -- index 0 is its own partner under
    :func:`_mirror_x` and therefore carries no symmetry constraint at all.
    Every mirror claim that reaches the grid edge is scored on the paired part,
    or it would be scoring the grid convention instead of the model."""
    return A[1:, 1:]


def _remap1d_exit_field(E_in, presc, dx, window):
    """A FROZEN copy of ``_apply_displaced_remap``'s envelope assembly with the
    symmetric input window switchable -- the in-process pre-fix path.

    Checked against the shipped function on every run by
    ``test_the_windowed_1d_arm_reproduces_the_shipped_call_bit_for_bit``, so it
    cannot drift into testing something the library does not do."""
    from scipy.ndimage import map_coordinates
    h_in, h_out, opl = _build_displaced_ray_map(
        presc['surfaces'], presc['thicknesses'], _WL,
        presc['aperture_diameter'] / 2.0,
        carrier_slope=None, eikonal_fn=None)[:3]
    Ny, Nx = E_in.shape
    k0 = 2.0 * np.pi / _WL
    order = np.argsort(h_out)
    ho, hi, op = (np.asarray(h_out)[order], np.asarray(h_in)[order],
                  np.asarray(opl)[order])
    keep = np.concatenate(([True], np.diff(ho) > 0))
    ho, hi, op = ho[keep], hi[keep], op[keep]
    X, Y = _axes(Nx, dx)
    r_out = np.sqrt(X * X + Y * Y)
    rc = np.clip(r_out, ho[0], ho[-1])
    hin_of = np.interp(rc, ho, hi)
    opl_of = np.interp(rc, ho, op)
    mp = np.interp(rc, ho, np.gradient(ho, hi))
    mp = np.where(mp <= 1e-12, 1e-12, mp)
    jac = np.sqrt(np.clip(hin_of, 0.0, None)
                  / (np.clip(rc, 1e-15, None) * mp))
    scale = np.where(r_out > 1e-15, hin_of / np.clip(r_out, 1e-15, None), 1.0)
    cx = (X * scale) / dx + Nx / 2.0
    cy = (Y * scale) / dx + Ny / 2.0
    F = _residual_input_field(E_in, None, _WL)
    amp = (map_coordinates(F.real, [cy, cx], order=1, mode='constant',
                           cval=0.0)
           + 1j * map_coordinates(F.imag, [cy, cx], order=1, mode='constant',
                                  cval=0.0))
    if window:
        win = ((np.abs(X * scale) <= (Nx / 2.0 - 1.0) * dx)
               & (np.abs(Y * scale) <= (Ny / 2.0 - 1.0) * dx))
        amp = np.where(win, amp, 0.0)
    E_out = amp * jac * np.exp(1j * k0 * (opl_of - float(op[0])))
    return np.asarray(np.where(r_out <= ho[-1], E_out, 0.0),
                      dtype=np.complex128)


class TestTheOneDimensionalRemapCarriesTheSameWindow:

    #: N, dx, w0 -- the grid edge must carry real envelope (0.40 and 0.44 of
    #: peak here) or the fixture cannot see the defect, and the traced exit
    #: radius must reach past the grid corner so the ``r_out <= ho[-1]`` cut is
    #: not what zeroes the rim.
    GRIDS = [(256, 12e-6, 1.6e-3), (320, 10e-6, 1.7e-3)]

    @pytest.mark.parametrize('N,dx,w0', GRIDS)
    def test_a_symmetric_element_on_a_centred_input_is_mirror_symmetric(
            self, N, dx, w0):
        """The whole claim in one line: a rotationally symmetric element and a
        centred, rotationally symmetric input must not produce a field that
        differs from its own mirror.

        Bar 1e-12 relative, against a measured 1.0e-16 (2026-09-13) and a
        pre-fix reading of 3.3e-02 -- four decades of gap below and ten above,
        so it is a gap and not a tuned number."""
        p = _sym_singlet()
        E = np.asarray(_disp(_gauss(N, dx, w0), p, dx,
                             displaced_mode='remap'))
        assert float(np.max(np.abs(E))) > 0.1          # the fixture is lit
        got = float(np.linalg.norm(_paired(np.abs(E))
                                   - _paired(np.abs(_mirror_x(E))))
                    / np.linalg.norm(_paired(np.abs(E))))
        assert got <= 1e-12, got

    @pytest.mark.parametrize('N,dx,w0', GRIDS)
    def test_a_decentred_input_field_mirrors(self, N, dx, w0):
        """The mirror-PAIR form the WP-B2 report said no fixture exercises:
        the element is symmetric, the INPUT is decentred by +-x0."""
        p = _sym_singlet()
        X, Y = _axes(N, dx)
        x0 = 0.45e-3

        def run(sgn):
            E0 = np.exp(-((X - sgn * x0) ** 2 + Y ** 2) / w0 ** 2
                        ).astype(np.complex128)
            return np.asarray(_disp(E0, p, dx, displaced_mode='remap'))

        a, b = np.abs(run(+1)), np.abs(_mirror_x(run(-1)))
        got = float(np.linalg.norm(_paired(a) - _paired(b))
                    / np.linalg.norm(_paired(a)))
        assert got <= 1e-12, got

    def test_fail_before_the_unwindowed_1d_assembly_is_not(self):
        """FAIL-BEFORE, in process, on the same fixture: without the window the
        +x rim is dead and its mirror is not.

        Two-sided and RATIO-based -- the windowed arm must be clean on exactly
        the assembly whose unwindowed twin is broken, so a fixture that stopped
        reaching the grid edge would fail the second arm rather than silently
        pass the first."""
        N, dx, w0 = 256, 12e-6, 1.6e-3
        p = _sym_singlet()
        E0 = _gauss(N, dx, w0)

        def resid(window):
            E = _remap1d_exit_field(E0.copy(), p, dx, window)
            a, b = np.abs(E), np.abs(_mirror_x(E))
            return float(np.max(np.abs(_paired(a) - _paired(b)))
                         / np.max(np.abs(a)))

        bad, good = resid(False), resid(True)
        assert good <= 1e-12, good
        assert bad > 1e-2, bad                     # a rim, not a rounding
        assert bad > 1e9 * max(good, 1e-16), (bad, good)

    def test_the_windowed_1d_arm_reproduces_the_shipped_call_bit_for_bit(self):
        """The frozen copy must BE the shipped assembly."""
        N, dx, w0 = 192, 14e-6, 1.3e-3
        p = _sym_singlet()
        E0 = _gauss(N, dx, w0)
        rm = _build_displaced_ray_map(
            p['surfaces'], p['thicknesses'], _WL,
            p['aperture_diameter'] / 2.0, carrier_slope=None, eikonal_fn=None)
        theirs = _apply_displaced_remap(
            E0.astype(np.complex128).copy(), rm[0], rm[1], _WL, dx, dx, rm[2])
        mine = _remap1d_exit_field(E0.copy(), p, dx, window=True)
        assert np.array_equal(mine, np.asarray(theirs))

    def test_the_window_costs_only_the_outermost_ring(self):
        """The price of the fix, pinned so it cannot grow silently: the window
        may only zero samples whose ENTRANCE read is outside the grid's largest
        centred window, i.e. the outer rim, and must leave the illuminated core
        untouched."""
        N, dx, w0 = 256, 12e-6, 1.6e-3
        p = _sym_singlet()
        E0 = _gauss(N, dx, w0)
        off = _remap1d_exit_field(E0.copy(), p, dx, window=False)
        on = _remap1d_exit_field(E0.copy(), p, dx, window=True)
        X, Y = _axes(N, dx)
        core = np.hypot(X, Y) <= 0.90 * (N / 2.0 - 1.0) * dx
        assert np.array_equal(off[core], on[core])
        moved = np.abs(off - on) > 0
        assert int(moved.sum()) > 0
        assert float(np.min(np.hypot(X, Y)[moved])) > 0.90 * (N / 2.0 - 1.0) * dx


# ===========================================================================
# 9.  VERIFY-WP-B2 -- the warning's pitch and the lattice it names are the
#     TRACE's, not the bare 2 r / (n - 1)
# ===========================================================================

class TestTheWarningQuotesThePitchTheTraceUses:

    @staticmethod
    def _dstep(r_max, n_side):
        """The launch pitch the builder actually throws, read off the builder."""
        p = _singlet(dec=(0.6e-3, 0.0))
        return float(_build_displaced_ray_map_2d(
            p['surfaces'], p['thicknesses'], _WL, r_max, n_side=n_side)[6])

    @pytest.mark.parametrize('n_side', [181, 257, 513])
    def test_the_quoted_pitch_is_the_pitch_the_trace_uses(self, n_side):
        """Parse the pitch out of the message and compare it with the ray
        map's own ``dstep``.  Bar: the two agree to the message's own printed
        precision (2 decimals of a micron)."""
        r_max, dx = 5e-3, 4e-6
        with pytest.warns(RuntimeWarning) as rec:
            _warn_if_remap_lattice_smooths(r_max, dx, dx, n_side)
        msg = str(rec[0].message)
        quoted = float(msg.split('launch lattice -- a ')[1].split(' um')[0])
        got = self._dstep(r_max, n_side) * 1e6
        assert abs(quoted - got) <= 0.005, (quoted, got)

    @pytest.mark.parametrize('dx', [8e-6, 4e-6, 2e-6])
    def test_the_named_lattice_really_clears_the_field_pitch(self, dx):
        """The message names ``displaced_n_side=n``; the lattice that ``n``
        actually traces must put the launch pitch at or below twice the field
        pitch.  Measured on the builder, not on the formula -- naming a value
        that clears a MIS-STATED bar is the defect class this campaign exists
        to close."""
        r_max = 5e-3
        with pytest.warns(RuntimeWarning) as rec:
            _warn_if_remap_lattice_smooths(r_max, dx, dx, 181)
        n = int(str(rec[0].message).split('displaced_n_side=')[1].split()[0])
        got = self._dstep(r_max, n)
        assert got <= 2.0 * dx, (n, got, 2.0 * dx)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _warn_if_remap_lattice_smooths(r_max, dx, dx, n)

    def test_the_bare_formula_would_not_have_cleared_it(self):
        """FAIL-BEFORE for the arithmetic: the lattice the bare formula
        ``ceil(r/h) + 1`` names leaves the REAL pitch above the bar, which is
        why one constant and not two formulas decides both."""
        r_max, dx = 5e-3, 8e-6
        n_bare = int(np.ceil(r_max / dx)) + 1
        assert self._dstep(r_max, n_bare) > 2.0 * dx, n_bare

    def test_the_fan_factor_is_one_constant(self):
        """The builder's fan and the warning's pitch must read the SAME
        constant, or they can drift apart again."""
        import inspect
        sig = inspect.signature(_build_displaced_ray_map_2d)
        assert (sig.parameters['r_fan_factor'].default
                == LR._DISP_REMAP_2D_FAN_FACTOR)
        r_max, n = 5e-3, 181
        want = 2 * r_max * LR._DISP_REMAP_2D_FAN_FACTOR / (n - 1)
        assert abs(self._dstep(r_max, n) - want) <= 1e-18, want


# ===========================================================================
# 10.  VERIFY-WP-B2 -- the mirror symmetry BELOW the shipped lattice
#
# ``displaced_n_side`` is public with a floor of 3, so every lattice from 3 up
# is a legal call; the shipped sweep starts at 181.
# ===========================================================================

class TestTheMirrorSymmetryHoldsBelowTheDefaultLattice:

    @staticmethod
    def _pair(n_side, method, N=256, dx=12e-6, w0=1.4e-3, d=0.5e-3):
        out = []
        for sgn in (+1, -1):
            p = _singlet(dec=(sgn * d, 0.0))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out.append(np.asarray(_apply_displaced_remap_2d(
                    _gauss(N, dx, w0), _ray_map(p, n_side), _WL, dx, dx,
                    interp_method=method)))
        a, b = np.abs(out[0]), np.abs(_mirror_x(out[1]))
        return float(np.linalg.norm(_paired(a) - _paired(b))
                     / np.linalg.norm(_paired(a)))

    @pytest.mark.parametrize('method', ['structured', 'delaunay'])
    @pytest.mark.parametrize('n_side', [11, 33, 97])
    def test_a_coarse_lattice_is_as_reflection_stable_as_the_default(
            self, n_side, method):
        """Bar derived from THIS build's reading at the shipped default: a
        coarse lattice may not be more than 1000x less symmetric than the
        default one.  Measured 2026-09-13 the coarse readings span
        3.9e-16..8.0e-14, the same order as the default lattice's own, while
        the failure this catches (the pre-WP-B2 input window) reads 1e-2 --
        twelve decades above."""
        base = max(self._pair(_DISP_REMAP_2D_N_SIDE, method), 1e-16)
        assert base < 1e-10, base
        got = self._pair(n_side, method)
        assert got <= 1000.0 * base, (n_side, method, got, base)


# ===========================================================================
# 11.  VERIFY-WP-B2 -- transmitted power against the geometric oracle, on a
#      fixture whose APERTURE actually binds
# ===========================================================================

class TestTheTransmittedPowerAgainstTheGeometricOracle:
    """The remap is a lossless geometric transfer inside the aperture -- the
    Jacobian factor is exactly the one that makes ``|E|^2 dA`` invariant -- so
    the exit power IS the input power inside the aperture and inside the
    carried window.  That is an absolute oracle rather than a backend
    comparison, and it needs a grid WIDER than the aperture or the aperture
    never binds (on the shipped fixture the 10 mm aperture is 2.6x the grid, so
    nothing is cut and ``P <= P_in`` is nearly free)."""

    #: half-width 6.656 mm against a 4 mm aperture radius, and |E| = 0.37 of
    #: peak at that radius -- the pupil edge is inside the grid AND inside the
    #: illuminated region, which is what makes both claims below measurable.
    N, DX, W0, AP = 512, 26e-6, 4.0e-3, 8e-3

    def _setup(self):
        """``(E_in, prescription, oracle power, the oracle's own floor)``.

        The oracle is a GRID SUM over the pixels inside the aperture and the
        carried window; its own resolution is the power in the ring the grid
        cannot assign to either side of the aperture edge (half a pixel
        diagonal), which is what every bar below is scored against instead of a
        chosen number.  Measured 2026-09-13 on this fixture: floor 5.8e-03 of
        the oracle, structured deviation 1.1e-04 / 7.9e-05 at n_side 181 / 257,
        scattered deviation 1.0e-03 / 9.1e-04 and always NEGATIVE."""
        X, Y = _axes(self.N, self.DX)
        E0 = _gauss(self.N, self.DX, self.W0)
        xw = (self.N / 2.0 - 1.0) * self.DX
        r = np.hypot(X, Y)
        keep = (r <= self.AP / 2.0) & (np.abs(X) <= xw) & (np.abs(Y) <= xw)
        ring = np.abs(r - self.AP / 2.0) <= self.DX * np.sqrt(2.0) / 2.0
        p = dict(_singlet(dec=(0.3e-3, 0.0)))
        p['aperture_diameter'] = self.AP
        return (E0, p, float((np.abs(E0) ** 2)[keep].sum()),
                float((np.abs(E0) ** 2)[ring].sum()))

    def _fields(self, p, E0, n_side):
        rm = _ray_map(p, n_side)
        out = []
        for method in ('structured', 'delaunay'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out.append(np.asarray(_apply_displaced_remap_2d(
                    E0.copy(), rm, _WL, self.DX, self.DX,
                    interp_method=method)))
        return out

    @pytest.mark.parametrize('n_side', [181, 257])
    def test_the_structured_backend_is_lossless_to_the_oracle(self, n_side):
        """Three arms, all scored against the oracle's OWN floor:

        * the structured answer sits inside that floor -- it neither loses the
          rim nor leaks power from beyond the aperture (the cut is on the
          INVERTED launch coordinate, so a leak would show up here as an
          EXCESS, and it does not);
        * the scattered answer is short by more than the structured one's whole
          deviation, one-sided, which is the hull dropping real power;
        * so the structured backend is strictly closer to the lossless answer.
        """
        E0, p, want, floor = self._setup()
        s, d = self._fields(p, E0, n_side)
        ps = float((np.abs(s) ** 2).sum())
        pd = float((np.abs(d) ** 2).sum())
        assert abs(ps - want) <= floor, (ps, want, floor)
        assert pd < want - abs(ps - want), (pd, want, ps)
        assert abs(ps - want) < abs(pd - want), (ps, pd, want)

    def test_the_scattered_backend_leaves_holes_where_the_aperture_binds(self):
        """The two-sided form of the hull claim on this fixture."""
        E0, p, _, _ = self._setup()
        s, d = self._fields(p, E0, 181)
        lit = np.abs(s) > 0.2 * np.max(np.abs(s))
        assert int(np.count_nonzero(d[lit] == 0)) > 0
        assert int(np.count_nonzero(s[lit] == 0)) == 0


def test_interp_method_is_validated():
    """The private backend selector refuses an unknown name rather than
    silently taking the default."""
    p = _singlet(dec=(0.4e-3, 0.0))
    rm = _ray_map(p, 33)
    with pytest.raises(ValueError, match='interp_method'):
        _apply_displaced_remap_2d(_gauss(64, 40e-6, 0.8e-3), rm, _WL,
                                  40e-6, 40e-6, interp_method='qhull')

# ===========================================================================
# 12.  VERIFY-WP-B2 -- two pins for claims nothing was enforcing
#
# Found by MUTATING the library in the source and re-running this file: with
# ``_DISP_REMAP_2D_N_SIDE`` put back to 181 all 75 tests stayed green, and with
# the affine Newton seed replaced by the identity seed all of them did too.  A
# deliverable no test can distinguish from its own pre-state is not pinned.
# ===========================================================================

class TestTheDefaultLatticeIsTheOneTheContractNames:

    def test_the_public_docstring_names_the_shipped_default(self):
        """The keyword's docstring quotes the default as a literal, so the
        constant and the contract can drift silently.  Build-free: it compares
        two things inside this build, no measurement and no tolerance."""
        doc = apply_real_lens.__doc__
        assert doc is not None
        block = doc.split('displaced_n_side : int or None')[1][:900]
        assert f'({_DISP_REMAP_2D_N_SIDE})' in block, (
            f'apply_real_lens\'s displaced_n_side docstring does not name the '
            f'shipped default {_DISP_REMAP_2D_N_SIDE}; one of the two moved '
            f'without the other.')

    def test_the_default_is_above_the_pre_raise_lattice(self):
        """L9 asked for the launch-lattice ceiling to come UP, and the raise is
        the deliverable.  Pinned as the direction plus the accuracy it buys --
        second order in the pitch, so the default must strictly improve on 181
        against a refined-lattice reference -- rather than as the literal 257,
        which would refuse a future re-derivation."""
        assert _DISP_REMAP_2D_N_SIDE > 181
        N, dx = 192, 16e-6
        p = _singlet(dec=(0.6e-3, 0.0))
        E0 = _gauss(N, dx, 1.2e-3)

        def field(n):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return np.asarray(_apply_displaced_remap_2d(
                    E0.copy(), _ray_map(p, n), _WL, dx, dx))

        fine = field(1025)
        core = np.abs(fine) > 0.2 * np.max(np.abs(fine))
        assert int(core.sum()) > 300, int(core.sum())

        def err(n):
            g = field(n)[core]
            return float(np.sqrt(np.mean(
                np.abs(_depistoned(g, fine[core]) - fine[core]) ** 2)))

        assert err(_DISP_REMAP_2D_N_SIDE) < err(181)

    def test_the_floor_and_the_default_are_consistent(self):
        assert _DISP_REMAP_2D_MIN_N_SIDE <= _DISP_REMAP_2D_N_SIDE
        assert _normalise_displaced_n_side(_DISP_REMAP_2D_N_SIDE) == \
            _DISP_REMAP_2D_N_SIDE


class TestTheAffineSeedIsWhyTheLoopIsShort:
    """What is pinned here is the seed's RESIDUAL advantage, not an operation
    count.  Measured 2026-09-13 by swapping the seed for the identity in
    process: the whole ``_apply_displaced_remap_2d`` call costs 20
    ``map_coordinates`` reads EITHER WAY on the p10 singlet at n_side 181 (two
    Newton sweeps plus the final check), so the seed's documented "removes a
    whole sweep" does not reproduce at the shipped residual bar -- Newton
    squares its error, and both seeds are already inside the bar after two
    sweeps.  The advantage that does reproduce is the seed residual itself."""

    def test_the_affine_seed_starts_a_whole_newton_step_closer(self):
        """``_remap2d_affine_seed`` exists because inverting the map's own
        global affine part starts one SQUARING of the Newton error closer than
        seeding at the target.  Pinned as the property -- a ratio measured on
        this build -- not as a sweep count, because the sweep count is bounded
        from above elsewhere and an identity seed slips under that bound.

        Bar: at least 4x.  Measured 2026-09-13 on the p10 singlet at n_side
        181, 23x (2.4e-06 m against 5.5e-05 m); the failure it must catch is
        the seed being removed altogether, which reads 1.0x exactly."""
        from scipy.ndimage import map_coordinates
        p = _singlet(dec=(0.6e-3, 0.0))
        rm = _ray_map(p, 181)
        X0, Y0, XOf, YOf, dstep = rm[0], rm[1], rm[2], rm[3], float(rm[6])
        u0, v0 = float(X0[0, 0]), float(Y0[0, 0])
        n_v, n_u = XOf.shape
        u_hi, v_hi = u0 + (n_u - 1) * dstep, v0 + (n_v - 1) * dstep
        N, dx = 192, 16e-6
        Xg, Yg = _axes(N, dx)
        Xt, Yt = Xg.ravel(), Yg.ravel()
        inside = (np.abs(Xt) < 0.8 * abs(u0)) & (np.abs(Yt) < 0.8 * abs(v0))
        Xt, Yt = Xt[inside], Yt[inside]

        def seed_residual(u, v):
            u = np.clip(u, u0, u_hi)
            v = np.clip(v, v0, v_hi)
            crd = np.stack([(v - v0) / dstep, (u - u0) / dstep])
            return float(np.median(np.hypot(
                Xt - map_coordinates(XOf, crd, order=1, mode='nearest'),
                Yt - map_coordinates(YOf, crd, order=1, mode='nearest'))))

        affine = seed_residual(*LR._remap2d_affine_seed(
            XOf, YOf, dstep, u0, v0, Xt, Yt))
        identity = seed_residual(Xt.copy(), Yt.copy())
        assert affine * 4.0 < identity, (affine, identity)

    def test_a_singular_affine_part_falls_back_to_the_target(self):
        """The documented fallback: a map whose affine part is singular (a fold
        collapsing the exit onto a line) must return the target itself, not a
        non-finite seed."""
        n = 9
        ax = np.linspace(-1e-3, 1e-3, n)
        U, V = np.meshgrid(ax, ax)
        XOf = np.zeros_like(U)                       # rank-0 affine part
        YOf = np.zeros_like(U)
        Xt = np.array([1e-4, -2e-4])
        Yt = np.array([3e-4, 5e-5])
        u, v = LR._remap2d_affine_seed(
            XOf, YOf, float(ax[1] - ax[0]), float(ax[0]), float(ax[0]),
            Xt, Yt)
        assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
        assert np.array_equal(u, Xt) and np.array_equal(v, Yt)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
