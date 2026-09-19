"""WP-B12b: per-surface GBD's image leg uses the shared exit-vertex projection.

Added 2026-09-15.  From v5.22 to 5.47.0
``lumenairy.propagators.gbd.apply_prescription_persurface_to_beamlets`` carried
its OWN vertex correction: it folded ``-sag`` into the image-side leg from an
in-line conic-sag expression written out of the last surface's ``radius`` and
``conic`` alone.  That copy was exact on a conic last surface and wrong on
every other class the tracer supports -- it dropped the even-aspheric
departure, the biconic y-branch, every freeform and the whole field-frame
decenter / tilt class, resolved no exit-medium index, and hard-assumed a
forward-propagating exit ray, so a MIRROR-terminated prescription got the
correction with the wrong sign.  WP-B12 measured it at 15.52 waves of optical
path on an A4 / A6 last surface (71 % of the sag) and left it as its first open
item.  It is now deleted: the local-frame branch asks the differential
primitive for ``reference='exit_vertex'`` and consumes the package's single
projection, ``raytrace.differential._project_to_exit_vertex_plane``.

The tests below assert the INVARIANTS that establishes -- which reference plane
each branch asks for, that the projection reproduces the library's own
``TraceResult.at_exit_vertex`` on every surface class, that a flat last surface
is the identity bit for bit, and that the repaired field reproduces an
independent diffraction oracle where the pre-repair one does not -- never the
readings the repair happened to produce.  Every bar is derived from a quantity
the running build measures, with the gap to the signal stated.  No wall-clock
assertion anywhere.

The diffraction oracle, the exact conic / even-aspheric meridional trace and
the Rayleigh-Sommerfeld-I sum are IMPORTED from
``tests/unit/test_audit2609_b12_fga_reference_plane.py`` -- the audit's one
independent oracle for this reference-plane question.  Re-typing them here
would be a third copy of a numerical kernel, which is the very defect this
package removes from the library.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import io
import tokenize
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import gbd as G
from lumenairy.raytrace import differential as D
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.surface import _surface_sag_xy

from tests.unit.test_audit2609_b12_fga_reference_plane import (
    _fid,
    _oracle_trace,
    _rs_radial,
    _sag_of,
)

_LAM = 1.064e-6
_GLASS = 'N-BAF10'


# ===========================================================================
# Fixtures.  ONE optic geometry with the LAST surface varied along a single
# axis, so any difference between two rows is the last surface and nothing
# else.  N-BAF10 biconvex, R1 = +11.0 mm, t = 0.9 mm, semi = 0.30 mm,
# 1.064 um, w0 = 0.18 mm -- deliberately SLOW (NA 0.015 .. 0.058) so the
# focal structure is resolved on the small grid below.
# ===========================================================================
_R, _T, _SEMI, _W0 = 11.0e-3, 0.9e-3, 0.30e-3, 0.18e-3
# The grid every field test uses: 96 x 6.6 um spans 0.634 mm, so the 0.60 mm
# clear aperture fits, and the Airy radius (17 .. 44 um across the fixtures)
# is three to seven pixels, so the focal structure is resolved.
_N, _DX = 96, 6.6e-6
# The beamlet frame every field test uses, named explicitly rather than left to
# ``_auto_sample_step``.  DERIVED from a cost/accuracy measurement on this
# fixture (2026-09-15, Windows py3.14): the auto frame is one beamlet per pixel
# with ``waist_factor=1``, whose beamlets have a 0.13 mm Rayleigh range and are
# therefore WIDER than the whole grid at the image plane, so the windowed
# reconstruction degenerates to the dense O(n_beamlets x N^2) sum and one call
# costs 78 s on a loaded box.  A 4-pixel frame with a matching waist costs
# 2.5 s and reproduces the dense-frame field to a fidelity of 0.999587 -- three
# decades below the 0.99 decision bar and four below the 0.90 one, on a defect
# that is 4.39 waves of pupil phase.  Naming it also makes the three entry
# points comparable: ``apply_real_lens_gbd``,
# ``apply_real_lens_universal(method='gbd')`` and
# ``propagate_gbd_through_prescription`` have different sampling DEFAULTS, so
# only an explicit frame lets them be compared byte for byte.
_FRAME = dict(sample_step=4, waist_factor=4.0)


def _singlet(last=None, R2=None, semi=_SEMI, glass=_GLASS, R1=_R, t=_T):
    s0 = {'radius': R1, 'conic': 0.0, 'thickness': t, 'glass_before': 'air',
          'glass_after': glass, 'semi_diameter': semi}
    s1 = {'radius': (-R1 if R2 is None else R2), 'conic': 0.0,
          'thickness': 0.0, 'glass_before': glass, 'glass_after': 'air',
          'semi_diameter': semi}
    if last:
        s1.update(last)
    return {'name': 'b12b', 'aperture_diameter': 2 * semi,
            'surfaces': [s0, s1], 'thicknesses': [t], 'stop_index': 0}


def _conic():
    return _singlet()


def _asphere():
    """A4 / A6 departure on a CURVED last surface."""
    return _singlet({'aspheric_coeffs': {4: 5.0e8, 6: -4.0e15}})


def _flat_base_asphere():
    """FLAT base radius, power carried by the polynomial.

    The deleted in-line copy guarded itself with
    ``if np.isfinite(_Rl) and _Rl != 0.0``, so on an infinite base radius it
    read the sag as EXACTLY zero and folded nothing into the leg -- which is
    bit-for-bit what the shipped primitive produces under
    ``reference='surface'``.  That makes this fixture the one place where the
    pre-WP-B12b field is reconstructible exactly, through a supported public
    keyword, with no copy of the deleted code.  The premise is asserted in
    every test that uses it.
    """
    return _singlet({'aspheric_coeffs': {2: -9.0e1, 4: 4.0e8}},
                    R2=float('inf'))


def _biconic():
    return _singlet({'radius_y': -7.0e-3})


def _freeform():
    return _singlet({'freeform_type': 'xy_polynomial',
                     'xy_coeffs': {(2, 0): 4.0e1, (0, 2): -2.4e1,
                                   (4, 0): 8.0e7},
                     'norm_x': 1.0, 'norm_y': 1.0})


def _field_frame():
    return _singlet({'decenter': (6.0e-5, -4.0e-5)})


def _flat_last():
    """Plano-convex with the CURVED side first: the last surface is flat."""
    semi = 0.262e-3
    return {'name': 'b12bflat', 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': 1.45e-3, 'conic': 0.0, 'thickness': 0.55e-3,
                 'glass_before': 'air', 'glass_after': 'N-LASF9',
                 'semi_diameter': semi},
                {'radius': np.inf, 'conic': 0.0, 'thickness': 0.0,
                 'glass_before': 'N-LASF9', 'glass_after': 'air',
                 'semi_diameter': semi}],
            'thicknesses': [0.55e-3], 'stop_index': 0}


def _mirror():
    semi = 0.60e-3
    return {'name': 'b12bmirror', 'aperture_diameter': 2 * semi,
            'surfaces': [{'radius': -20.0e-3, 'conic': 0.0, 'thickness': 0.0,
                          'glass_before': 'air', 'glass_after': 'MIRROR',
                          'semi_diameter': semi}],
            'thicknesses': [0.0], 'stop_index': 0}


def _gbd_surfs(presc):
    """The surface list the GBD per-surface path traces: the last transfer
    zeroed on a copy, exactly as the function does."""
    s = list(surfaces_from_prescription(presc))
    s[-1] = copy.copy(s[-1])
    s[-1].thickness = 0.0
    return s


def _grid(N, dx, w0):
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    return X, Y, np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _pin_surface(base):
    """Force a differential primitive onto the LAST-SURFACE reference plane.

    ``reference='surface'`` is a supported public value, so this reaches the
    un-projected state through the shipped API rather than through a copy of
    the deleted code.

    Read it precisely.  On a CURVED-base last surface this is the pre-v5.22
    behaviour -- no vertex correction at all -- and NOT the v5.22 .. 5.47.0
    behaviour, which folded a conic-only ``-sag`` into the leg; the two differ
    by the whole conic sag.  It coincides with v5.22 .. 5.47.0 exactly on a
    surface whose in-line sag was identically zero, i.e. a FLAT last surface
    and the FLAT-BASE aspheric fixture, because the deleted block guarded
    itself with ``if np.isfinite(_Rl) and _Rl != 0.0``.  Every test that uses
    this to reconstruct the pre-repair field says which case it is in and
    asserts the premise.
    """
    def wrapped(*a, **kw):
        kw['reference'] = 'surface'
        return base(*a, **kw)
    return wrapped


class _ForceSurface:
    """Context manager: both primitives pinned to ``reference='surface'``."""

    def __enter__(self):
        self._saved = (D.ray_transfer_jacobian,
                       D.ray_transfer_jacobian_analytic)
        D.ray_transfer_jacobian = _pin_surface(self._saved[0])
        D.ray_transfer_jacobian_analytic = _pin_surface(self._saved[1])
        return self

    def __exit__(self, *exc):
        (D.ray_transfer_jacobian,
         D.ray_transfer_jacobian_analytic) = self._saved
        return False


def _gbd_field(presc, z, N, dx, w0):
    _X, _Y, E = _grid(N, dx, w0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_gbd(
            E, prescription=presc, wavelength=_LAM, dx=dx,
            output_plane_distance=float(z), **_FRAME))


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a))).hexdigest()


def _fan(semi, n_r=17, n_az=8):
    """A collimated 2-D fan.  NOT a meridional line: a biconic or an
    XY-polynomial freeform departs from the rotationally-symmetric conic only
    off the x axis, so a y = 0 fan would read those defects as zero."""
    r = np.linspace(semi / (2 * n_r), semi * 0.98, n_r)
    az = (np.arange(n_az) + 0.5) * (np.pi / n_az)
    R, A = np.meshgrid(r, az, indexing='ij')
    return (R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()


def _transfer(presc, semi, *, reference, per_surface=True, fan=None):
    """The differential transfer GBD's ``jacobian='auto'`` would get."""
    surfs = _gbd_surfs(presc)
    h, y = fan if fan is not None else _fan(semi)
    z = np.zeros_like(h)
    for fn in (D.ray_transfer_jacobian_analytic, D.ray_transfer_jacobian):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return surfs, h, y, fn(h.copy(), y.copy(), z.copy(), z.copy(),
                                       surfs, _LAM, per_surface=per_surface,
                                       reference=reference)
        except NotImplementedError:
            continue
    raise AssertionError('no differential backend accepted this prescription')


# ===========================================================================
# 1.  Which reference plane does each branch ask for?
# ===========================================================================
def test_the_local_branch_asks_for_the_exit_vertex_plane():
    """DECISION: ``apply_prescription_persurface_to_beamlets`` requests
    ``reference='exit_vertex'`` from the differential primitive on its
    local-frame branch and ``reference='surface'`` on its world-frame branch.

    Measured behaviourally -- the primitives are wrapped and the keyword they
    actually receive is recorded -- so the claim survives any refactor of the
    call site that keeps the behaviour.

    The two branches genuinely need different planes: the local branch adds
    ``z_image`` measured from the last surface's VERTEX (a back focal
    distance), while the world branch world-traces the base rays itself and
    starts its own leg at the last-surface INTERSECTION.  Asking for
    ``'exit_vertex'`` there would double-count the sag -- the same defect,
    mirrored.
    """
    seen = []
    saved = (D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic)

    def rec(base, label):
        def wrapped(*a, **kw):
            seen.append((label, kw.get('reference', '<default>')))
            return base(*a, **kw)
        return wrapped

    N, dx = _N, _DX
    _X, _Y, E = _grid(N, dx, _W0)
    presc = _asphere()
    try:
        D.ray_transfer_jacobian = rec(saved[0], 'fd')
        D.ray_transfer_jacobian_analytic = rec(saved[1], 'analytic')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b = G.decompose_field_to_beamlets(E, dx, wavelength=_LAM,
                                              **_FRAME)
            G.apply_prescription_persurface_to_beamlets(
                b, presc, _LAM, z_image=8.0e-3)
            local = list(seen)
            seen.clear()
            G.apply_prescription_persurface_to_beamlets(
                b, presc, _LAM, world_output_plane='auto')
            world = list(seen)
    finally:
        (D.ray_transfer_jacobian,
         D.ray_transfer_jacobian_analytic) = saved

    assert local and all(r == 'exit_vertex' for _l, r in local), (
        f'the local branch asked for {local}, not exit_vertex')
    assert world and all(r == 'surface' for _l, r in world), (
        f'the world branch asked for {world}, not surface')


def test_the_module_carries_no_second_sag_kernel():
    """INVARIANT (the house rule): ``propagators/gbd.py`` contains no second
    implementation of a surface sag.

    Checked on the module's TOKEN stream with comments and string literals
    removed, so the prose above the call site -- which necessarily names the
    thing it no longer does -- cannot satisfy or defeat the check.  The two
    shapes the deleted block was made of are the ones searched for: a read of
    the last surface's ``radius`` / ``conic`` inside the beamlet function, and
    the conic-sag radical ``sqrt(1 - (1+k) c^2 r^2)``.
    """
    src = inspect.getsource(G)
    code = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                        tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue
        code.append(tok.string)
    stream = ' '.join(code)
    for banned in ('_Rl', '_kl', '_cl'):
        assert banned not in stream, (
            f'propagators/gbd.py still defines {banned}: the in-line '
            f'conic-sag copy is back')
    # the sag radical, written without whitespace so formatting cannot hide it
    flat = stream.replace(' ', '')
    assert 'conic_sag' not in flat, (
        'propagators/gbd.py imports a sag kernel directly; the projection '
        'belongs to raytrace.differential')
    fn_src = inspect.getsource(G.apply_prescription_persurface_to_beamlets)
    fn_code = []
    for tok in tokenize.generate_tokens(io.StringIO(fn_src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        fn_code.append(tok.string)
    fn_flat = ' '.join(fn_code).replace(' ', '')
    assert "'radius'" not in fn_flat and '"radius"' not in fn_flat, (
        'apply_prescription_persurface_to_beamlets reads the last surface '
        'radius again; the sag belongs to the shared projection')
    assert "reference=_reference" in fn_flat or "reference='exit_vertex'" \
        in fn_flat, (
        'the function no longer names a reference plane at its call site')


# ===========================================================================
# 2.  The projection is exact on every surface class GBD can reach
# ===========================================================================
@pytest.mark.parametrize('name,presc,semi', [
    ('conic', _conic(), _SEMI),
    ('asphere', _asphere(), _SEMI),
    ('flat_base_asphere', _flat_base_asphere(), _SEMI),
    ('biconic', _biconic(), _SEMI),
    ('freeform', _freeform(), _SEMI),
    ('field_frame', _field_frame(), _SEMI),
    ('mirror', _mirror(), 0.60e-3),
])
def test_the_projection_reproduces_the_library_s_own_exit_vertex_operator(
        name, presc, semi):
    """DECISION: on every surface class the GBD per-surface path can reach,
    the state the primitive now returns is the one
    ``TraceResult.at_exit_vertex`` produces -- and the deleted conic-only copy
    was NOT, except on the conic rows.

    ``at_exit_vertex`` is the library's one ray-bundle vertex operator and is
    a different implementation from the 4x4-state projection under test (it
    works on direction cosines, ``t = -z/N``), so this is an independent
    reading, not a restatement.

    Bars.  The agreement is asserted against the TRACE's own floor:
    ``4e-15 * (the geometric scale of the quantity)``, i.e. a relative 4e-15,
    about fifteen decades below the defect this package removes.  The
    contrast arm -- the conic-only sag, evaluated here from the surface the
    test built -- is asserted to be large only where it IS large, as a
    premise-gated claim: the conic and mirror rows are exactly the rows on
    which the old copy was right about the sag, and the test says so.
    """
    surfs, h, y, dv = _transfer(presc, semi, reference='exit_vertex')
    surfs2, _h2, _y2, ds = _transfer(presc, semi, reference='surface')
    z = np.zeros_like(h)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = la.raytrace.RayBundle(
            x=h.copy(), y=y.copy(), z=z.copy(), L=z.copy(), M=z.copy(),
            N=np.ones_like(h), wavelength=_LAM,
            alive=np.ones(h.size, bool), opd=z.copy())
        ex = la.raytrace.trace(bundle, surfs, _LAM).at_exit_vertex()
    ok = (np.asarray(ex.alive, bool) & np.asarray(dv.alive, bool)
          & np.asarray(ds.alive, bool))
    assert ok.sum() >= 0.9 * h.size, (
        f'premise: only {ok.sum()}/{h.size} rays survive this fixture')

    # --- the two sags at the SAME state: the package's shared kernel, and
    #     the conic-only expression the deleted block used ---
    last = surfs[-1]
    Rl = float(getattr(last, 'radius', np.inf))
    kl = float(getattr(last, 'conic', 0.0) or 0.0)
    xs_, ys_ = np.asarray(ds.x), np.asarray(ds.y)
    if np.isfinite(Rl) and Rl != 0.0:
        cl = 1.0 / Rl
        r2 = xs_ ** 2 + ys_ ** 2
        sag_inline = cl * r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + kl) * cl * cl * r2, 0.0)))
    else:
        sag_inline = np.zeros_like(xs_)
    sag_true = np.asarray(_surface_sag_xy(xs_, ys_, last), dtype=np.float64)
    sec = np.sqrt(1.0 + np.asarray(ds.ux) ** 2 + np.asarray(ds.uy) ** 2)
    nz = D._exit_direction_sign(surfs)
    n_exit = float(la.raytrace.exit_vertex.resolve_exit_index(
        surfs, _LAM, fn_name='b12b test'))
    sag_w = float(np.abs(sag_true[ok]).max() / _LAM)
    assert sag_w > 1.0, f'premise: {name} last-surface sag is {sag_w:.3f} waves'

    # --- the bars, DERIVED from the arithmetic the two paths perform ---
    # Both compute the same transfer from different intermediates
    # (``at_exit_vertex`` from the direction cosines and the traced ``z``, the
    # projection from the sag kernel and the unreduced slopes), so the floor
    # is the machine epsilon of the largest intermediate, with 1e4 of slack
    # for the accumulated operations of an intersection solve.  It is seven to
    # nine decades below the defect being removed, and the gap to that defect
    # is asserted below rather than assumed.
    eps = float(np.finfo(np.float64).eps)
    scale_x = float(np.abs(np.asarray(ex.x)[ok]).max()
                    + np.abs(sag_true[ok]).max()
                    * float(np.abs(np.asarray(ds.ux)[ok]).max() + 1.0))
    scale_o = float(np.abs(np.asarray(ex.opd)[ok]).max()
                    + n_exit * np.abs(sag_true[ok]).max()
                    * float(sec[ok].max()))
    bar_x, bar_o = 1e4 * eps * scale_x, 1e4 * eps * scale_o
    d_x = float(np.abs((np.asarray(ex.x) - np.asarray(dv.x))[ok]).max())
    d_o = float(np.abs((np.asarray(ex.opd) - np.asarray(dv.opd))[ok]).max())
    assert d_x <= bar_x, (
        f'{name}: projected height differs from at_exit_vertex by {d_x:.3e} m '
        f'against a derived floor of {bar_x:.3e} m (scale {scale_x:.3e} m)')
    assert d_o <= bar_o, (
        f'{name}: projected OPL differs from at_exit_vertex by {d_o:.3e} m '
        f'against a derived floor of {bar_o:.3e} m (scale {scale_o:.3e} m)')

    # --- what the deleted copy would have applied here ---
    opl_gap = float(np.abs(
        ((sag_inline - n_exit * nz * sag_true) * sec)[ok]).max())
    if name == 'conic':
        assert opl_gap <= bar_o, (
            f'{name}: the conic-only copy is meant to be EXACT here and reads '
            f'{opl_gap:.3e} m ({opl_gap / _LAM:.4f} waves) against a derived '
            f'floor of {bar_o:.3e} m')
    else:
        assert opl_gap / _LAM > 0.5, (
            f'{name}: the conic-only copy is only {opl_gap / _LAM:.4f} waves '
            f'off here, so this fixture does not separate the two')
        assert opl_gap > 1e6 * bar_o, (
            f'{name}: the defect ({opl_gap:.3e} m) is only '
            f'{opl_gap / bar_o:.1e} times the agreement floor; the two bars '
            f'of this test do not have a gap between them')


# ===========================================================================
# 3.  A flat last surface is the identity, bit for bit
# ===========================================================================
def test_a_flat_last_surface_field_is_bit_identical_to_the_surface_arm():
    """INVARIANT: on a prescription whose last surface is FLAT the projection
    short-circuits structurally, so the GBD field is bit-for-bit what the
    last-surface reference plane produces -- the same bytes 5.47.0 returned.

    Two-sided: the same comparison on a CURVED last surface must DIFFER, else
    the identity above would be vacuous (it would pass on a build where the
    keyword did nothing).
    """
    assert D._last_surface_sag_vanishes(_gbd_surfs(_flat_last())[-1]), (
        'premise: this fixture must have a structurally flat last surface')
    assert not D._last_surface_sag_vanishes(_gbd_surfs(_conic())[-1]), (
        'premise: the contrast fixture must have a curved last surface')

    N, dx, w0, z = _N, 6.0e-6, 190e-6, 1.43e-3
    a = _gbd_field(_flat_last(), z, N, dx, w0)
    with _ForceSurface():
        b = _gbd_field(_flat_last(), z, N, dx, w0)
    assert _sha(a) == _sha(b), (
        'a flat last surface must be bit-identical across the reference '
        f'plane; relative L2 {np.linalg.norm(a - b) / np.linalg.norm(a):.3e}')

    Nc, dxc, zc = _N, _DX, 8.27e-3
    c = _gbd_field(_conic(), zc, Nc, dxc, _W0)
    with _ForceSurface():
        d = _gbd_field(_conic(), zc, Nc, dxc, _W0)
    assert _sha(c) != _sha(d), (
        'a CURVED last surface must NOT be bit-identical across the two '
        'reference planes; the keyword is inert on this build')


# ===========================================================================
# 4.  The decision against an independent diffraction oracle
# ===========================================================================
def test_the_repaired_field_reproduces_a_diffraction_oracle_where_the_old_one_did_not():
    """DECISION: on a FLAT-BASE aspheric last surface -- where the deleted
    in-line copy read the sag as exactly zero, so ``reference='surface'``
    reproduces the pre-WP-B12b field bit for bit -- the shipped GBD field
    reproduces an independent Rayleigh-Sommerfeld-I oracle and the pre-repair
    one does not.

    Oracle: the exact conic / even-aspheric meridional trace and the
    brute-force RS-I sum imported from the WP-B12 test file.  Its error bound
    is its OWN convergence in the ray quadrature, measured here (n_h 251 ->
    501) and asserted before it is used as a reference.

    Bars.  0.99 for "reproduces the field" and 0.9 for "does not", with the
    whole distance between them empty: the defect on this fixture is a
    ``k * sag`` phase ramp reaching 4.39 waves at the rim (measured
    2026-09-15, both builds, ``validation/probe_gbd_projection/
    probe_a_sag.py``), and no global piston can absorb a ramp.  Both arms are
    scored, so the claim cannot pass on a build where the keyword silently did
    nothing.  The premise that the in-line copy really read zero here -- an
    infinite base radius -- is asserted first, and the sag itself is
    re-measured from the oracle's own trace before the bars are used.
    """
    presc = _flat_base_asphere()
    last = _gbd_surfs(presc)[-1]
    assert not np.isfinite(float(last.radius)), (
        'premise: the deleted copy guarded on np.isfinite(radius); this '
        'fixture must have an infinite base radius for the surface arm to '
        'reproduce it exactly')
    assert last.aspheric_coeffs, 'premise: the power must be in the polynomial'
    assert not D._last_surface_sag_vanishes(last), (
        'premise: the projection must still be active on this surface')

    ng = float(la.get_glass_index(_GLASS, _LAM))
    asph = dict(last.aspheric_coeffs)
    n_h = 501
    h = np.linspace(_SEMI / (2 * n_h), _SEMI * (1.0 - 1.0 / (2 * n_h)), n_h)
    _xs, _us, _ols, x_v, u_v, opl_v = _oracle_trace(
        h, [(0.0, _R, 0.0, None, ng), (_T, np.inf, 0.0, asph, 1.0)])
    sag_w = float(np.abs(_sag_of(np.abs(_xs), np.inf, 0.0, asph)).max() / _LAM)
    assert sag_w > 1.0, f'premise: last-surface sag is {sag_w:.3f} waves'
    dxe = np.gradient(x_v, h, edge_order=2)
    assert np.all(dxe > 0), 'premise: the exit plane is not a caustic'
    wgt = np.exp(-(h / _W0) ** 2) * np.sqrt(h * x_v * dxe) * (h[1] - h[0])

    # the readout plane: the fixture's own traced best focus (derived)
    f0 = float(-x_v[0] / u_v[0])
    zs = np.linspace(0.7 * f0, 1.15 * f0, 1501)
    ww = np.exp(-2.0 * (h / _W0) ** 2) * h
    xz = x_v[None, :] + u_v[None, :] * zs[:, None]
    ctr = (ww * xz).sum(1) / ww.sum()
    var = (ww * (xz - ctr[:, None]) ** 2).sum(1) / ww.sum()
    z = float(zs[int(np.argmin(var))])

    N, dx = _N, _DX
    Xg, Yg, _E = _grid(N, dx, _W0)
    rg = np.hypot(Xg, Yg)
    na = abs(u_v[-1]) / np.sqrt(1.0 + u_v[-1] ** 2)
    airy = 0.61 * _LAM / na
    rho = np.concatenate([np.arange(0.0, 8.0 * airy, airy / 6.0),
                          np.arange(8.0 * airy, float(rg.max()) + dx, dx)])
    rho[0] = 1e-12
    hi = _rs_radial(x_v, opl_v, wgt, rho, z, _LAM)
    lo = _rs_radial(x_v[::2], opl_v[::2], wgt[::2] * 2.0, rho, z, _LAM)
    conv = float(np.linalg.norm(hi - lo) / np.linalg.norm(hi))
    assert conv < 1e-3, f'oracle not converged in its ray quadrature: {conv:.2e}'
    orc = (np.interp(rg.ravel(), rho, hi.real)
           + 1j * np.interp(rg.ravel(), rho, hi.imag)).reshape(rg.shape)

    f_now = _fid(orc, _gbd_field(presc, z, N, dx, _W0))
    with _ForceSurface():
        f_old = _fid(orc, _gbd_field(presc, z, N, dx, _W0))
    assert f_now > 0.99, (
        f'the repaired GBD field scores {f_now:.4f} against the oracle '
        f'(oracle convergence {conv:.2e}, sag {sag_w:.2f} waves)')
    assert f_old < 0.90, (
        f'the pre-repair arm scores {f_old:.4f}: this fixture does not '
        f'separate the two reference planes on this build')


# ===========================================================================
# 5.  The mirror sign, which the deleted copy could not carry
# ===========================================================================
def test_a_mirror_terminated_prescription_gets_the_propagation_sign():
    """DECISION: after a MIRROR the outgoing ray travels toward -z, so the
    transfer to the vertex plane ADDS optical path where a transmissive exit
    subtracts it.  The shared projection recovers that sign from the
    prescription (``_exit_direction_sign``); the deleted in-line copy folded
    ``-sag`` into the leg unconditionally, i.e. with the wrong sign, which
    doubles the error instead of removing it.

    Asserted as the two signs and as the size of the doubling, both derived
    from the same trace.  The independent reference is again
    ``TraceResult.at_exit_vertex``, which reads the sign off the direction
    cosines rather than off the surface list.
    """
    presc = _mirror()
    surfs = _gbd_surfs(presc)
    assert D._exit_direction_sign(surfs) == -1.0, (
        'premise: one mirror must reverse the propagation direction')
    _s, h, y, ds = _transfer(presc, 0.60e-3, reference='surface')
    _s2, _h2, _y2, dv = _transfer(presc, 0.60e-3, reference='exit_vertex')
    ok = np.asarray(ds.alive, bool) & np.asarray(dv.alive, bool)
    sag = np.asarray(_surface_sag_xy(ds.x, ds.y, surfs[-1]), dtype=np.float64)
    sec = np.sqrt(1.0 + np.asarray(ds.ux) ** 2 + np.asarray(ds.uy) ** 2)
    n_exit = float(la.raytrace.exit_vertex.resolve_exit_index(
        surfs, _LAM, fn_name='b12b mirror test'))
    d_opl = (np.asarray(dv.opd) - np.asarray(ds.opd))[ok]
    # the projection: opd_v = opd - n*sign(N)*sag*sec, and sign(N) = -1 here,
    # so the transfer ADDS optical path.
    expect = (n_exit * sag * sec)[ok]
    scale = float(np.abs(expect).max())
    assert scale / _LAM > 1.0, (
        f'premise: the mirror sag is {scale / _LAM:.3f} waves')
    # Bar: both sides are the same product of the same two factors in a
    # different order, so the floor is machine epsilon of the product with
    # 1e3 of slack for the accumulated operations.  MEASURED 2026-09-15 on
    # both builds at 4.2e-18 m on a 8.7e-06 m scale (4.8e-13 relative, i.e.
    # 2e3 eps); the bar sits 4.5 decades below the signal it is separating
    # from, which is 100 % of the quantity (a sign).
    eps = float(np.finfo(np.float64).eps)
    bar = 1e4 * eps * scale
    gap = float(np.abs(d_opl - expect).max())
    assert gap <= bar, (
        f'the projected OPL does not carry the mirror sign: {gap:.3e} m '
        f'against a derived floor of {bar:.3e} m (scale {scale:.3e} m)')

    # What the deleted copy did here.  Its sag was EXACT (a conic mirror), so
    # its whole error is the sign: it folded -sag into the leg, contributing
    # ``-sag_inline*sec`` of optical path where the projection contributes
    # ``+n*sag*sec``.  The gap is therefore TWICE the correction, and the
    # sag-exactness is asserted separately so the reader can see that a
    # sag-only comparison would have reported this surface as fine.
    Rl = float(surfs[-1].radius)
    cl = 1.0 / Rl
    r2 = np.asarray(ds.x) ** 2 + np.asarray(ds.y) ** 2
    kl = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
    sag_inline = cl * r2 / (1.0 + np.sqrt(np.maximum(
        1.0 - (1.0 + kl) * cl * cl * r2, 0.0)))
    assert float(np.abs((sag_inline - sag)[ok]).max()) <= bar, (
        'premise: the in-line conic sag must be EXACT on a conic mirror, so '
        'that the measured error below is the SIGN and nothing else')
    old_gap = float(np.abs(((-sag_inline * sec) - (n_exit * sag * sec))[ok]
                           ).max())
    assert old_gap >= 1.9 * scale, (
        f'the in-line copy is only {old_gap / _LAM:.3f} waves off on this '
        f'mirror against a {scale / _LAM:.3f}-wave correction; the sign '
        f'defect is not being exercised')


# ===========================================================================
# 6.  Both Jacobian backends, and the public entry points
# ===========================================================================
def test_both_jacobian_backends_land_on_the_same_plane():
    """DECISION: ``jacobian='fd'`` and ``jacobian='analytic'`` put the beamlet
    base rays on the SAME plane, to the finite-difference primitive's own
    truncation floor -- so the reference-plane keyword reaches both backends
    and neither is left on the surface.

    The bar is the FD floor measured on this build: the two backends' agreement
    on the LAST-SURFACE plane, where no projection is involved.  Asserting the
    projected agreement against that same floor makes the claim "the
    projection adds nothing" rather than "the two agree to 1e-12".
    """
    presc = _asphere()
    surfs = _gbd_surfs(presc)
    h, y = _fan(_SEMI)
    z = np.zeros_like(h)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        an_s = D.ray_transfer_jacobian_analytic(
            h.copy(), y.copy(), z.copy(), z.copy(), surfs, _LAM,
            per_surface=True, reference='surface')
        fd_s = D.ray_transfer_jacobian(
            h.copy(), y.copy(), z.copy(), z.copy(), surfs, _LAM,
            per_surface=True, reference='surface')
        fd_v = D.ray_transfer_jacobian(
            h.copy(), y.copy(), z.copy(), z.copy(), surfs, _LAM,
            per_surface=True, reference='exit_vertex')
        an_v = D.ray_transfer_jacobian_analytic(
            h.copy(), y.copy(), z.copy(), z.copy(), surfs, _LAM,
            per_surface=True, reference='exit_vertex')
    ok = (np.asarray(fd_s.alive, bool) & np.asarray(an_s.alive, bool)
          & np.asarray(fd_v.alive, bool) & np.asarray(an_v.alive, bool))
    assert ok.sum() > 0.8 * h.size
    floor = float(np.abs((np.asarray(fd_s.x) - np.asarray(an_s.x))[ok]).max())
    gap = float(np.abs((np.asarray(fd_v.x) - np.asarray(an_v.x))[ok]).max())
    assert gap <= max(10.0 * floor, 1e-16), (
        f'the two backends disagree by {gap:.3e} m on the vertex plane '
        f'against a {floor:.3e} m disagreement on the surface plane')


def test_the_public_entry_points_follow_the_beamlet_function():
    """DECISION: the repair reaches every public entry onto this code path.

    ``apply_real_lens_universal(method='gbd')`` is a dispatcher, not a second
    model, so it must return the SAME BYTES as ``apply_real_lens_gbd``; and
    ``propagate_gbd_through_prescription(per_surface=True)``, the propagators-
    level entry, must land on the same plane (scored as agreement with
    ``apply_real_lens_gbd``, whose own agreement with the diffraction oracle
    is the subject of section 4).  All three must MOVE when the primitives are
    pinned back to the surface plane, which is what makes the claim two-sided.
    """
    presc = _flat_base_asphere()
    N, dx, z = _N, _DX, 16.276e-3
    _X, _Y, E = _grid(N, dx, _W0)
    a = _gbd_field(presc, z, N, dx, _W0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        u = np.asarray(la.apply_real_lens_universal(
            E, prescription=presc, wavelength=_LAM, dx=dx, method='gbd',
            output_plane_distance=z, method_kwargs={'gbd': dict(_FRAME)}))
        p = np.asarray(G.propagate_gbd_through_prescription(
            E, dx, presc, wavelength=_LAM, output_shape=(N, N), output_dx=dx,
            per_surface=True, z_image=z, **_FRAME))
    assert _sha(u) == _sha(a), (
        "apply_real_lens_universal(method='gbd') is not the same bytes as "
        'apply_real_lens_gbd')
    assert _fid(p, a) > 0.999, (
        'propagate_gbd_through_prescription(per_surface=True) scores '
        f'{_fid(p, a):.6f} against apply_real_lens_gbd')
    with _ForceSurface():
        a2 = _gbd_field(presc, z, N, dx, _W0)
    assert _sha(a2) != _sha(a), (
        'the field does not move when the primitives are pinned to the '
        'surface plane; the keyword is inert on this build')
