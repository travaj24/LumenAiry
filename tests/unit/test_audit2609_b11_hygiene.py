"""WP-B11 -- the hygiene pass: the consolidations and the small deferred
items, each with the property its refactor was gated on.

Every bar here is DERIVED -- measured on this box, with the falsifying
alternative measured beside it so the number means something -- and nothing in
this file asserts wall-clock time, memory or a per-build reading
(``docs/TESTING_STANDARDS.md`` S5).  The bit-identity proofs themselves are not
here: a refactor's gate is a child-process comparison of the PRE-change library
against this one (``git archive`` into a read-only tree, cwd and PYTHONPATH
both that tree, ``lumenairy.__file__`` asserted), which pytest cannot express
because pytest puts the repo root ahead of PYTHONPATH.  The report carries
those runs.  What IS here is the PROPERTY each consolidation has to keep, so a
later edit that breaks it goes red inside the suite.

Sections, by the work-package item they close:

1.  the shared branch-cut band and the two forward selectors
    (``lumenairy/_branchcut.py``)
3.  the shared row-band schedule of the chunked lens surface paths
    (``_lens_real._row_bands`` / ``_band_in_halo``)
4.  the lens-family leaf (``elements/_lens_kernels.py``) and the import cycles
6.  ``LensConfig.to_kwargs(strict=True)``
7.  the 1-D symmetric remap's centred input window, on a DECENTRED INPUT FIELD
11. ``pmm_jones_2d`` assembles the tensor operators once
12. ``sampling=`` on the free-space HFPI entry points
19. ``PMM2DStackHybrid``'s validated attributes refuse an out-of-vocabulary
    assignment after ``__init__``

and, from part b:

3b. the SAS near-field chirp-sampling gate (``sas._warn_sas_chirp_sampling``)
5b. the ``LensPhysics`` configuration object
8b. ``doe.py``'s named sentinel, and warning ``stacklevel`` attribution across
    the lens family
4b. ``PMM2DStackHybrid.truncation`` joins the guarded attributes
"""
from __future__ import annotations

import ast
import os
import pathlib
import warnings

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]


# ===========================================================================
# Item 1 -- one band mask, two selectors that are NOT interchangeable
# ===========================================================================

class TestTheBranchCutLeaf:
    """``lumenairy/_branchcut.py`` is what the RCWA, EME, PMM and BOR branch
    decisions share, and what they deliberately do not."""

    def test_the_band_mask_is_the_comparison_all_four_engines_make(self):
        """``band_mask(r, scale=s, band=b)`` is exactly ``|r| <= b * s`` --
        including the ``<=``, which is what puts an EXACTLY ZERO component on
        the cut (the lossless real ``lam^2`` case) rather than off it."""
        from lumenairy._branchcut import band_mask
        r = np.array([0.0, -0.0, 1e-30, 1e-9, 1e-8, 1.0000000001e-8, 1.0])
        got = band_mask(r, scale=1.0, band=1e-8)
        assert np.array_equal(got, np.abs(r) <= 1e-8)
        assert bool(got[0]) and bool(got[1]), "zero must classify ON the cut"
        assert bool(got[4]), "the band edge is inclusive"
        assert not bool(got[5])

    def test_a_unit_band_against_the_product_is_the_same_comparison(self):
        """``eme._branch.cut_band`` returns the PRODUCT, so its caller passes
        it as the scale with ``band=1.0``.  ``x * 1.0`` is exact, so that
        spelling must agree with the factored one bit for bit."""
        from lumenairy._branchcut import band_mask
        rng = np.random.default_rng(11)
        r = 10.0 ** rng.uniform(-30.0, 5.0, 4096) * rng.choice([-1.0, 1.0], 4096)
        scale = 3.7e-4
        band = 1e-9
        a = band_mask(r, scale=scale, band=band)
        b = band_mask(r, scale=band * scale, band=1.0)
        assert np.array_equal(a, b)

    def test_the_two_selectors_disagree_on_complex_input(self):
        """The reason there are two.  ``where(flip, -z, z)`` touches only sign
        bits; ``z * where(flip, -1.0, 1.0)`` is a COMPLEX multiply, whose cross
        term rewrites the sign of a zero part and turns an infinite part into a
        NaN.  MEASURED here rather than asserted in prose, because the day the
        two are "obviously the same" is the day somebody merges them and moves
        every branch decision in the library by a sign bit."""
        from lumenairy._branchcut import negate_forward, signed_forward
        z = np.array([1.0 + 0.0j, -0.0j, np.inf + 0.0j, np.nan + 0.0j],
                     dtype=complex)
        flip = np.array([True, True, False, False])
        with np.errstate(invalid='ignore'):
            a = negate_forward(z, flip)
            b = signed_forward(z, flip)
        assert not np.array_equal(a.view(np.float64), b.view(np.float64))
        # and the specific differences, so a change in numpy's complex
        # multiply is a failure here rather than a silent branch move
        assert np.signbit(a[0].imag) and not np.signbit(b[0].imag)
        # the multiply poisons an infinite part even where it does NOT flip:
        # ``1.0`` promotes to ``1 + 0j`` and the cross term is ``inf * 0``.
        assert b[2].real == np.inf and np.isnan(b[2].imag)
        assert a[2] == complex(np.inf, 0.0) and not np.isnan(a[2].imag)
        # on REAL input they agree exactly -- the halves of the contract
        f = np.array([0.0, -0.0, 1.0, -1.0, np.inf])
        m = np.array([True, False, True, False, True])
        assert np.array_equal(negate_forward(f, m).view(np.uint64),
                              signed_forward(f, m).view(np.uint64))

    def test_every_engine_routes_its_band_through_the_leaf(self):
        """A grep with an AST, so a future copy of the comparison is caught the
        way the five-copy defects that produced this leaf were not."""
        sites = {
            'lumenairy/elements/rcwa/_core.py': '_band_mask',
            'lumenairy/elements/eme/_branch.py': '_band_mask',
            'lumenairy/elements/pmm/_core.py': '_band_mask',
            'lumenairy/elements/bor/_orient.py': '_band_mask',
        }
        for rel, name in sites.items():
            src = (REPO / rel).read_text(encoding='utf-8')
            tree = ast.parse(src)
            called = {n.func.id for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            assert name in called, f'{rel} no longer calls the shared band'

    def test_the_selectors_are_the_shared_objects(self):
        from lumenairy import _branchcut
        from lumenairy.elements.bor import _orient
        from lumenairy.elements.pmm import _core as pmm_core
        from lumenairy.elements.rcwa import _core as rcwa_core
        assert rcwa_core._signed_forward is _branchcut.signed_forward
        assert pmm_core._negate_forward is _branchcut.negate_forward
        assert _orient._negate_forward is _branchcut.negate_forward
        assert rcwa_core._band_mask is _branchcut.band_mask


# ===========================================================================
# Item 3 -- one row-band schedule
# ===========================================================================

class TestTheRowBandSchedule:

    @pytest.mark.parametrize('n_rows,chunk,halo', [
        (1, 1, 0), (1, 8, 3), (7, 1, 1), (7, 3, 2), (512, 64, 0),
        (512, 1, 3), (512, 4096, 2), (513, 64, 1), (256, 33, 5),
    ])
    def test_the_bands_tile_the_grid_and_the_halo_is_clipped(
            self, n_rows, chunk, halo):
        """The three properties every banded path's byte-identity rests on:
        the bands TILE the rows exactly once; the halo never leaves the grid
        (which is what keeps the first / last band's one-sided gradient
        stencils the whole grid's); and ``[lo:hi)`` selects the band out of the
        halo."""
        from lumenairy.elements._lens_real import _row_bands
        seen = []
        for r0, r1, h0, h1, lo, hi in _row_bands(n_rows, chunk, halo):
            assert 0 <= r0 < r1 <= n_rows
            assert 0 <= h0 <= r0 and r1 <= h1 <= n_rows
            assert hi - lo == r1 - r0
            probe = np.arange(h0, h1)
            assert np.array_equal(probe[lo:hi], np.arange(r0, r1))
            seen.append((r0, r1))
        assert [a for a, _ in seen] == sorted(a for a, _ in seen)
        assert np.array_equal(
            np.concatenate([np.arange(a, b) for a, b in seen]),
            np.arange(n_rows))

    def test_a_zero_halo_is_the_band_itself(self):
        from lumenairy.elements._lens_real import _row_bands
        for r0, r1, h0, h1, lo, hi in _row_bands(100, 7):
            assert (h0, h1, lo, hi) == (r0, r1, 0, r1 - r0)

    def test_band_in_halo_is_the_offset_the_generator_uses(self):
        from lumenairy.elements._lens_real import _band_in_halo, _row_bands
        for r0, r1, h0, _h1, lo, hi in _row_bands(300, 16, 4):
            assert _band_in_halo(r0, r1, h0) == (lo, hi)

    def test_the_banded_surface_path_is_the_whole_grid_path(self):
        """The property the 44-configuration matrix exists for, on one
        configuration so the suite carries it: the chunked screen reproduces
        the whole-grid screen BYTE for byte, at chunk widths that do and do not
        divide the grid."""
        from lumenairy.elements._lens_real import apply_real_lens
        lam, N, dx = 632.8e-9, 192, 8e-6
        rx = dict(
            surfaces=[dict(radius=50e-3, glass_before='AIR',
                           glass_after='N-BK7'),
                      dict(radius=-40e-3, conic=-0.5,
                           aspheric_coeffs={4: 1e-5},
                           glass_before='N-BK7', glass_after='AIR')],
            thicknesses=[3e-3], aperture_diameter=1.2e-3)
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E = np.exp(-(X ** 2 + Y ** 2) / (0.45e-3) ** 2).astype(np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = apply_real_lens(E.copy(), prescription=rx, wavelength=lam,
                                  dx=dx, sag_chunk_rows=0)
            for cr in (1, 7, 64, 4096):
                got = apply_real_lens(E.copy(), prescription=rx,
                                      wavelength=lam, dx=dx,
                                      sag_chunk_rows=cr)
                assert np.array_equal(ref.view(np.uint8), got.view(np.uint8)), (
                    f'sag_chunk_rows={cr} moved bits against the whole grid')


# ===========================================================================
# Item 4 -- the lens-family leaf and the import cycles
# ===========================================================================

def _module_level_family_edges():
    """``{module: {sibling, ...}}`` over the lens family, MODULE-SCOPE imports
    only -- the ones that execute at import time and can deadlock an import
    order.  In-function imports are deliberately not edges."""
    root = REPO / 'lumenairy' / 'elements'
    fam = sorted(p.stem for p in root.glob('*.py')
                 if p.stem == 'lenses' or p.stem.startswith('lenses_')
                 or p.stem.startswith('_lens'))

    def edges(path):
        out = set()

        class V(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                pass

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_ImportFrom(self, node):
                if node.level == 1 and node.module in fam:
                    out.add(node.module)
                self.generic_visit(node)

        V().visit(ast.parse(path.read_text(encoding='utf-8')))
        return out

    return {m: edges(root / f'{m}.py') for m in fam}


class TestTheLensKernelsLeaf:

    def test_the_leaf_imports_nothing_from_lumenairy(self):
        """What makes it a leaf, and therefore what makes it impossible for it
        to be half of a cycle.  Checked on the SOURCE, at module scope AND in
        function bodies, because a deferred import would still be a dependency
        -- just a later one."""
        src = (REPO / 'lumenairy' / 'elements' / '_lens_kernels.py').read_text(
            encoding='utf-8')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0 and not (
                    node.module or '').startswith('lumenairy'), (
                    f'_lens_kernels imports {node.module!r} (level '
                    f'{node.level}) -- it must stay a leaf')
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert not a.name.startswith('lumenairy'), a.name

    def test_the_facade_re_exports_the_same_objects(self):
        """Not "a function of the same name": the SAME object, so a caller
        that reached the helper through ``lenses`` and one that reached it
        through the leaf cannot diverge."""
        from lumenairy.elements import _lens_kernels as leaf
        from lumenairy.elements import lenses
        for name in ('check_grid_vs_apertures',
                     'recommend_grid_for_prescription',
                     '_collect_semi_diameters',
                     '_warn_if_aperture_exceeds_grid'):
            assert getattr(lenses, name) is getattr(leaf, name), name

    def test_the_traced_entry_point_no_longer_closes_a_cycle(self):
        graph = _module_level_family_edges()
        assert 'lenses' not in graph['_lens_traced'], (
            '_lens_traced imports the lenses facade at module scope again; it '
            'must read elements/_lens_kernels.py, which is a leaf')

    def test_the_family_cycle_count_does_not_grow(self):
        """A RATCHET, not a bar.  The two that remain are enumerated in
        ``docs/lens_configuration.md`` section "Module layout" with the edit
        each needs; this fails if a new one appears."""
        graph = _module_level_family_edges()
        cycles = {tuple(sorted((a, b))) for a, deps in graph.items()
                  for b in deps if a in graph.get(b, ())}
        assert cycles <= {('_lens_real', 'lenses'), ('lenses', 'lenses_maslov')}, (
            f'a new module-level 2-cycle appeared in the lens family: '
            f'{sorted(cycles)}')


# ===========================================================================
# Item 6 -- to_kwargs(strict=True)
# ===========================================================================

class TestToKwargsStrict:

    def test_strict_refuses_a_request_the_entry_point_cannot_carry(self):
        from lumenairy.elements.lens_config import LensConfig
        cfg = LensConfig.from_kwargs(newton_poly_order=8, bandlimit=False)
        assert cfg.to_kwargs(entry_point='apply_real_lens') == {
            'bandlimit': False}, 'the silent default must not have moved'
        with pytest.raises(ValueError, match=r'LensConfig\.to_kwargs:') as e:
            cfg.to_kwargs(entry_point='apply_real_lens', strict=True)
        assert 'newton_poly_order' in str(e.value)
        assert 'narrowed_to' in str(e.value), (
            'the refusal must name the explicit "yes, drop them" route')

    def test_a_field_left_at_its_default_is_not_a_request(self):
        """Dropping a default-valued field changes no argument, so strict is
        quiet about it -- including under ``include_defaults=True``, where the
        field IS emitted for the entry points that accept it."""
        from lumenairy.elements.lens_config import LensConfig
        cfg = LensConfig.from_kwargs(bandlimit=False)
        kw = cfg.to_kwargs(entry_point='apply_real_lens',
                           include_defaults=True, strict=True)
        assert kw['bandlimit'] is False and len(kw) > 1

    def test_narrowed_to_is_the_sanctioned_escape(self):
        from lumenairy.elements.lens_config import LensConfig
        cfg = LensConfig.from_kwargs(newton_poly_order=8, bandlimit=False)
        got = cfg.narrowed_to('apply_real_lens').to_kwargs(
            entry_point='apply_real_lens', strict=True)
        assert got == {'bandlimit': False}

    def test_strict_without_an_entry_point_is_refused(self):
        """It cannot do anything there -- nothing is narrowed, so nothing can
        be dropped -- and accepting it silently is the same class of quiet
        no-op the switch exists to remove."""
        from lumenairy.elements.lens_config import LensConfig
        with pytest.raises(ValueError, match='needs entry_point'):
            LensConfig.from_kwargs(bandlimit=False).to_kwargs(strict=True)

    def test_every_entry_point_accepts_a_strictly_narrowed_config(self):
        """The round trip the switch makes assertable: for each entry point,
        narrowing then flattening under ``strict=True`` never raises."""
        from lumenairy.elements.lens_config import ENTRY_POINTS, LensConfig
        cfg = LensConfig.from_kwargs(newton_poly_order=8, bandlimit=False)
        for ep in ENTRY_POINTS:
            cfg.narrowed_to(ep).to_kwargs(entry_point=ep, strict=True)


# ===========================================================================
# Item 7 -- the 1-D symmetric remap, on a DECENTRED INPUT FIELD
# ===========================================================================

_P10_RX = dict(
    surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
              dict(radius=-50e-3, glass_before='N-BK7', glass_after='AIR')],
    thicknesses=[4e-3], aperture_diameter=6e-3)
_P10_WL = 632.8e-9


def _offset_gauss(N, dx, w, xoff):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-((X - xoff) ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _mirror_x(I):
    """``x = (arange(N) - N/2) * dx`` puts index ``j`` at ``-x[N - j]``, so the
    mirror fixes column 0 and reverses the rest."""
    out = np.empty_like(I)
    out[:, 0] = I[:, 0]
    out[:, 1:] = I[:, :0:-1]
    return out


def _remap_1d(N, dx, xoff):
    from lumenairy.elements._lens_real import (
        _apply_displaced_remap, _build_displaced_ray_map,
    )
    rm = _build_displaced_ray_map(
        _P10_RX['surfaces'], _P10_RX['thicknesses'], _P10_WL,
        _P10_RX['aperture_diameter'] / 2.0,
        carrier_slope=None, eikonal_fn=None)
    E = _offset_gauss(N, dx, 1.2e-3, xoff)
    return _apply_displaced_remap(E, rm[0], rm[1], _P10_WL, dx, dx, rm[2])


def _sampled_envelope(N, dx, xoff, *, window):
    """The remap's input sample, with and without the centred window -- the
    falsifying arm.  Reproduces the sampling step of ``_apply_displaced_remap``
    (the shipped body always windows, so the unwindowed reading cannot be taken
    through it)."""
    from scipy.ndimage import map_coordinates

    from lumenairy.elements._lens_real import (
        _build_displaced_ray_map, _residual_input_field,
    )
    rm = _build_displaced_ray_map(
        _P10_RX['surfaces'], _P10_RX['thicknesses'], _P10_WL,
        _P10_RX['aperture_diameter'] / 2.0,
        carrier_slope=None, eikonal_fn=None)
    order = np.argsort(rm[1])
    ho, hi = np.asarray(rm[1])[order], np.asarray(rm[0])[order]
    keep = np.concatenate(([True], np.diff(ho) > 0))
    ho, hi = ho[keep], hi[keep]
    x = (np.arange(N, dtype=np.float64) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r_out = np.sqrt(X * X + Y * Y)
    rc = np.clip(r_out, ho[0], ho[-1])
    scale = np.where(r_out > 1e-15,
                     np.interp(rc, ho, hi) / np.clip(r_out, 1e-15, None), 1.0)
    cx = (X * scale) / dx + N / 2.0
    cy = (Y * scale) / dx + N / 2.0
    F = _residual_input_field(_offset_gauss(N, dx, 1.2e-3, xoff), None, _P10_WL)
    amp = (map_coordinates(F.real, [cy, cx], order=1, mode='constant', cval=0.0)
           + 1j * map_coordinates(F.imag, [cy, cx], order=1, mode='constant',
                                  cval=0.0))
    if window:
        w = ((np.abs(X * scale) <= (N / 2.0 - 1.0) * dx)
             & (np.abs(Y * scale) <= (N / 2.0 - 1.0) * dx))
        amp = np.where(w, amp, 0.0)
    return amp


class TestTheSymmetricRemapInputWindow:
    """The 1-D remap is ROTATIONALLY SYMMETRIC, so nothing in the suite ever
    put a mirror pair through it -- every p10 mirror fixture decenters the
    ELEMENT, which routes to the 2-D remap instead.  A DECENTRED INPUT FIELD on
    a symmetric element is the fixture that reaches this path, and it is the
    one the window's exposure was invisible to."""

    @pytest.mark.parametrize('N,dx,d', [
        (512, 8e-6, 0.8e-3), (384, 1e-5, 1.0e-3), (640, 6e-6, 0.6e-3)])
    def test_a_decentred_input_pair_comes_out_an_exact_mirror(self, N, dx, d):
        """MEASURED 2.6e-16 / 2.6e-16 / 2.4e-16 relative L2 over these three
        grids (2026-09-13), worst single pixel 1.05e-15 of the peak -- i.e. the
        interpolation's own rounding, since ``fl(q + N/2)`` and ``fl(N/2 - q)``
        round independently.  The bar is 1e-12, four decades above that and ten
        below the unwindowed reading the next test takes."""
        a = np.abs(_remap_1d(N, dx, +d)) ** 2
        b = np.abs(_remap_1d(N, dx, -d)) ** 2
        res = np.linalg.norm(a - _mirror_x(b)) / np.linalg.norm(a)
        assert res <= 1e-12, f'mirror residual {res:.3e}'

    @pytest.mark.parametrize('N,dx,d', [
        (512, 8e-6, 0.8e-3), (384, 1e-5, 1.0e-3), (640, 6e-6, 0.6e-3)])
    def test_without_the_window_the_same_pair_is_not_a_mirror(self, N, dx, d):
        """The falsification arm (S5 V1).  Without the centred window the
        sampled envelope reaches one whole sample further on -x than on +x, so
        the +x side loses a crescent the -x side keeps: MEASURED 1.04e-02 /
        3.12e-02 / 6.81e-03 relative L2 and up to 0.31 of the PEAK on a single
        pixel (2026-09-13).  Without this arm the test above would pass on a
        build that had quietly dropped the window on both sides."""
        a = np.abs(_sampled_envelope(N, dx, +d, window=False)) ** 2
        b = np.abs(_sampled_envelope(N, dx, -d, window=False)) ** 2
        res = np.linalg.norm(a - _mirror_x(b)) / np.linalg.norm(a)
        assert res > 1e-3, f'unwindowed residual {res:.3e} -- too small to be '\
                           f'the asymmetry this arm is measuring'
        w = np.abs(_sampled_envelope(N, dx, +d, window=True)) ** 2
        wb = np.abs(_sampled_envelope(N, dx, -d, window=True)) ** 2
        wres = np.linalg.norm(w - _mirror_x(wb)) / np.linalg.norm(w)
        assert wres < res / 1e9, (
            f'the window must be what closes it: {wres:.3e} vs {res:.3e}')

    def test_the_byte_identity_pin_is_not_moved_by_the_window(self):
        """``test_niche_p10_transverse_walk_remap.py::
        test_symmetric_remap_is_the_p2_1d_remap_byte_identical`` compares two
        calls that BOTH run ``_apply_displaced_remap``, so the window is on
        both sides of it and the pin does not move -- restated here as a
        property rather than left as an assumption."""
        from lumenairy.elements._lens_real import (
            _apply_displaced_remap, _build_displaced_ray_map,
        )
        N, dx = 256, 8e-6
        rm = _build_displaced_ray_map(
            _P10_RX['surfaces'], _P10_RX['thicknesses'], _P10_WL,
            _P10_RX['aperture_diameter'] / 2.0,
            carrier_slope=None, eikonal_fn=None)
        E = _offset_gauss(N, dx, 1.2e-3, 0.0)
        a = _apply_displaced_remap(E.copy(), rm[0], rm[1], _P10_WL, dx, dx,
                                   rm[2])
        b = _apply_displaced_remap(E.copy(), rm[0], rm[1], _P10_WL, dx, dx,
                                   rm[2])
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


# ===========================================================================
# Item 11 -- pmm_jones_2d assembles the tensor operators ONCE
# ===========================================================================

_PMM_PERIOD = 0.7e-6
_PMM_WL = 1.55e-6


def _pmm_tile(offplane=False, nx=6, ny=4):
    t = np.zeros((ny, nx, 3, 3), dtype=complex)
    for j in range(ny):
        for i in range(nx):
            t[j, i] = np.eye(3) * (6.0 if i < nx // 2 else 2.1)
    if offplane:
        t[..., 0, 2] += 0.7
        t[..., 2, 0] += 0.7
    return t


class TestJones2DAssemblesOnce:
    """The layer's source-free projected operators are built once per call.

    The double build was reachable on exactly the cells the even-parity fold
    CANNOT take: at normal incidence with ``symmetry`` on, the fold probe ran
    the whole assembly, answered None because the cell is out-of-plane or
    slanted, and the cascade then assembled it all over again."""

    @pytest.mark.parametrize('label,offplane,slant', [
        ('offplane', True, None),
        ('slanted', False, (0.4, 0.0)),
        ('inplane', False, None),
    ])
    def test_the_projected_operators_are_built_once(self, label, offplane,
                                                    slant, monkeypatch):
        from lumenairy.elements.pmm import twod_jones as tj
        calls = {'n': 0}
        real = tj._tensor_projected_ops

        def counted(*a, **k):
            calls['n'] += 1
            return real(*a, **k)

        monkeypatch.setattr(tj, '_tensor_projected_ops', counted)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tj.pmm_jones_2d(
                _PMM_PERIOD, _PMM_PERIOD, _pmm_tile(offplane), 1.444, 1.0,
                0.25e-6, _PMM_WL, theta=0.0, n_orders=5, degree=7,
                formulation='laurent', symmetry='auto', slant=slant)
        assert calls['n'] == 1, (
            f'{label} cell at normal incidence built the projected operators '
            f'{calls["n"]} times for ONE layer; the hoisted build must serve '
            f'both the fold probe and the cascade.')

    def test_the_counter_is_live_and_counts_per_call(self, monkeypatch):
        """Counter-pin (S5 V1): the wrapper must actually be on the assembly
        the solve reaches.  A patch that never fired -- the name rebound in the
        wrong module, an inlined call -- would read 0 or 1 for any number of
        builds and make the count above meaningless.  TWO solves must read
        exactly two builds."""
        from lumenairy.elements.pmm import twod_jones as tj
        calls = {'n': 0}
        real = tj._tensor_projected_ops

        def counted(*a, **k):
            calls['n'] += 1
            return real(*a, **k)

        monkeypatch.setattr(tj, '_tensor_projected_ops', counted)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(2):
                tj.pmm_jones_2d(
                    _PMM_PERIOD, _PMM_PERIOD, _pmm_tile(True), 1.444, 1.0,
                    0.25e-6, _PMM_WL, theta=0.0, n_orders=5, degree=7,
                    formulation='laurent', symmetry='auto')
        assert calls['n'] == 2


# ===========================================================================
# Item 12 -- sampling= on the free-space HFPI entry points
# ===========================================================================

_HFPI_KW = dict(aperture_radius=200e-6, z_aperture_to_output=1e-3,
                n_paths=16384, rng=3, output_dx=2e-5, cone_half_angle=0.35)


def _hfpi_field():
    return np.ones((32, 32), dtype=complex)


class TestHfpiFreeSpaceSampling:

    def test_the_default_is_the_uniform_draw_byte_for_byte(self):
        """The keyword is an OPT-IN: this is a Monte-Carlo estimator whose
        realisation depends on the placement rule, so the default has to stay
        the sequence every existing caller already gets."""
        from lumenairy.propagators.hfpi import propagate_hfpi
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6, **_HFPI_KW)
            b = propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6,
                               sampling='uniform', **_HFPI_KW)
        assert np.array_equal(a.view(np.uint8), b.view(np.uint8))
        assert np.count_nonzero(a) > 100, (
            'the fixture landed almost nothing, so a byte comparison of two '
            'near-empty grids would pass whatever the sampler did')

    @pytest.mark.parametrize('sampler', ['jittered', 'sobol'])
    def test_stratified_routes_to_the_other_initialiser(self, sampler):
        from lumenairy.propagators.hfpi import propagate_hfpi
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            a = propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6, **_HFPI_KW)
            c = propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6,
                               sampling='stratified', sampler=sampler,
                               **_HFPI_KW)
        assert np.all(np.isfinite(c))
        assert not np.array_equal(a, c), (
            'the stratified request produced the uniform draw -- the selector '
            'is not routing')
        # the two estimators must agree on the integral they estimate, which
        # is what makes the switch a variance choice and not a physics one
        pa = float(np.sum(np.abs(a) ** 2))
        pc = float(np.sum(np.abs(c) ** 2))
        assert abs(pc - pa) / pa < 0.25, (
            f'the two samplers disagree by {abs(pc - pa) / pa:.3f} of the '
            f'total -- far beyond the Monte-Carlo spread at this path count, '
            f'so one of them is estimating a different integral')

    def test_a_sampler_without_stratified_is_refused_not_dropped(self):
        from lumenairy.propagators.hfpi import propagate_hfpi
        with pytest.raises(ValueError, match='silently dropped'):
            propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6,
                           sampler='sobol', **_HFPI_KW)
        with pytest.raises(ValueError, match='silently dropped'):
            propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6,
                           n_strata_xy=(8, 8), **_HFPI_KW)

    def test_an_unknown_sampling_is_refused(self):
        from lumenairy.propagators.hfpi import propagate_hfpi
        with pytest.raises(ValueError,
                           match=r'propagate_hfpi_freespace_aperture:'):
            propagate_hfpi(_hfpi_field(), 1e-3, 633e-9, 8e-6,
                           sampling='quasi', **_HFPI_KW)


# ===========================================================================
# Item 19 -- PMM2DStackHybrid's validated attributes stay validated
# ===========================================================================

class TestStack2DAttributeGuards:

    @pytest.mark.parametrize('attr,bad,good', [
        ('formulation', 'fff_nv', 'laurent'),
        ('cascade', 'turbo', 'monolithic'),
        ('symmetry', 'fff_nv', False),
    ])
    def test_an_out_of_vocabulary_assignment_is_refused(self, attr, bad, good):
        """``st.formulation = 'fff_nv'`` was accepted and then read through an
        ``== 'li'`` test, so the stack quietly behaved as ``'laurent'``.  The
        refusal is the only behaviour change."""
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        st = PMM2DStackHybrid(0.7e-6)
        with pytest.raises(ValueError, match='PMM2DStackHybrid|symmetry'):
            setattr(st, attr, bad)
        setattr(st, attr, good)          # and a legal one still lands
        assert getattr(st, attr) == good

    def test_the_constructor_and_the_setter_share_one_vocabulary(self):
        """One definition, so the two cannot drift: every value ``__init__``
        accepts the setter accepts, and every value it refuses the setter
        refuses."""
        from lumenairy.elements.pmm.stack2d import (
            _CASCADES, _FORMULATIONS, PMM2DStackHybrid,
        )
        probes = list(_FORMULATIONS) + list(_CASCADES) + ['fff_nv', 'turbo']
        for attr, vocab in (('formulation', _FORMULATIONS),
                            ('cascade', _CASCADES)):
            for v in probes:
                by_init = True
                try:
                    PMM2DStackHybrid(0.7e-6, **{attr: v})
                except ValueError:
                    by_init = False
                st = PMM2DStackHybrid(0.7e-6)
                by_set = True
                try:
                    setattr(st, attr, v)
                except ValueError:
                    by_set = False
                assert by_init == by_set == (v in vocab), (
                    f'{attr}={v!r}: __init__ {"took" if by_init else "refused"}'
                    f' it, the setter {"took" if by_set else "refused"} it, '
                    f'vocabulary says {v in vocab}')

    def test_symmetry_resolves_on_assignment_exactly_as_on_construction(self):
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        for value, resolved in (('auto', True), (True, True), (False, False),
                                (1, True), (0, False)):
            st = PMM2DStackHybrid(0.7e-6, symmetry=value)
            assert st.symmetry is resolved
            st2 = PMM2DStackHybrid(0.7e-6)
            st2.symmetry = value
            assert st2.symmetry is resolved

    def test_the_guard_does_not_disturb_a_legal_reassignment(self):
        """The caches key on these attributes, so a LEGAL change must still
        take effect -- the guard is a refusal, not a freeze."""
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        st = PMM2DStackHybrid(0.7e-6, formulation='li', cascade='fast')
        st.formulation = 'laurent'
        st.cascade = 'monolithic'
        assert (st.formulation, st.cascade) == ('laurent', 'monolithic')


# ===========================================================================
# Item 3 (part b) -- the SAS near-field chirp-sampling gate
# ===========================================================================

def _sas_window_filling(N, dx, wfrac=0.35, p=4):
    """A super-Gaussian that FILLS the input window.

    The guard can only know ``(N, dx, lambda)``; the field's own support is
    invisible to it, so its bound is the worst case it can know -- a field out
    to the window edge.  This is that field, and it is also the shape a clipped
    lens aperture presents to the in-glass gap legs.
    """
    n = (np.arange(N) - N / 2) * dx
    r2 = np.add.outer(n ** 2, n ** 2)
    return np.exp(-(r2 / (wfrac * N * dx) ** 2) ** (p / 2)).astype(complex)


def _sas_run(E, z, lam, dx, **kw):
    from lumenairy.propagators.sas import scalable_angular_spectrum_propagate
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out, dxo, _ = scalable_angular_spectrum_propagate(E, z, lam, dx, **kw)
    return out, dxo, [w for w in caught
                      if issubclass(w.category, RuntimeWarning)]


class TestTheSasNearFieldGate:
    """``scalable_angular_spectrum_propagate``'s third step is a single-FFT
    Fresnel sum on the INPUT grid, so its quadratic chirp has to be resolved at
    pitch ``dx`` exactly as ``fresnel_propagate``'s does.  Before WP-B11b the
    only validity test in the module was the paper's FAR bound ``z > z_limit``;
    the near direction returned a four-decade energy gain in silence
    (``test_audit2609_a15a_lens_covering_array`` measures it end to end).
    """

    LAM = 632.8e-9

    def test_the_bound_is_N_dx2_over_lambda_and_the_edge_is_inclusive(self):
        """Below ``z_near = N dx^2 / lambda`` the guard fires; at and above it
        the call is silent.  Both directions, so a guard that fired always --
        or never -- fails."""
        N, dx = 64, 2.0e-6
        z_near = N * dx ** 2 / self.LAM
        E = _sas_window_filling(N, dx)
        for frac, want in ((0.05, True), (0.5, True), (1.0 - 1e-9, True),
                           (1.0, False), (1.0 + 1e-9, False), (3.0, False)):
            _, _, w = _sas_run(E, frac * z_near, self.LAM, dx)
            fired = any('UNDER-SAMPLED' in str(x.message) for x in w)
            assert fired is want, (
                f'z = {frac} x N*dx^2/lambda: guard '
                f'{"fired" if fired else "was silent"}, expected '
                f'{"fired" if want else "silence"} '
                f'({[str(x.message)[:90] for x in w]})')

    def test_the_bound_does_not_move_with_the_padding_factor(self):
        """This is the falsifiable half of the derivation.  ``pad`` enlarges
        the array the chirp is evaluated on, so a bound of ``pad*N*dx^2/lambda``
        is the obvious alternative; it is WRONG, because the precompensation
        ``delta_H`` is a band-limited phase filter whose impulse response stays
        on the input window and the chirp's outer turns multiply zero padding.

        MEASURED (N = 128, dx = 2 um, lambda = 633 nm, window-filling field,
        oracle = the same kernel at 8x finer input pitch): at z = 0.2 x
        ``N dx^2/lambda`` the relative field error is 2.24 at pad 2 and 2.30 at
        pad 4 -- the same ABSOLUTE z breaks both -- and at z = 0.75x it is
        2.0e-3 and 1.8e-3.  A ``pad``-scaled bound would call pad 4 valid at a
        z where it is not, and invalid at three z where it is.
        """
        N, dx = 64, 2.0e-6
        z_near = N * dx ** 2 / self.LAM
        E = _sas_window_filling(N, dx)
        for frac in (0.3, 0.9, 1.2, 4.0):
            verdicts = set()
            for pad in (1, 2, 4):
                _, _, w = _sas_run(E, frac * z_near, self.LAM, dx, pad=pad)
                verdicts.add(any('UNDER-SAMPLED' in str(x.message) for x in w))
            assert len(verdicts) == 1, (
                f'z = {frac} x N*dx^2/lambda: the guard disagreed across '
                f'pad in (1, 2, 4) -- it has picked up a pad dependence the '
                f'measurement says is not there.')
            assert verdicts == {frac < 1.0}

    def test_the_guard_fires_exactly_where_the_answer_is_wrong(self):
        """The gate has to be calibrated against the error, not asserted.

        Oracle: the SAME SAS kernel on an 8x finer input pitch over the SAME
        physical window.  Its own chirp bound is 8x smaller, so it is inside
        its envelope wherever the coarse run is not, and its output pitch
        ``lambda z / (pad N dx)`` is IDENTICAL, so the comparison is sample
        against sample with no interpolation.
        """
        N, dx, M = 32, 4.0e-6, 8
        z_near = N * dx ** 2 / self.LAM
        Ec = _sas_window_filling(N, dx)
        Ef = _sas_window_filling(N * M, dx / M)
        a0 = (N * M - N) // 2
        seen = {}
        for frac in (0.1, 2.0):
            oc, dxo_c, w = _sas_run(Ec, frac * z_near, self.LAM, dx)
            of, dxo_f, _ = _sas_run(Ef, frac * z_near, self.LAM, dx / M)
            assert abs(dxo_c - dxo_f) <= 1e-12 * dxo_c
            ref = of[a0:a0 + N, a0:a0 + N]
            rel = float(np.linalg.norm(oc - ref) / np.linalg.norm(ref))
            gain = float(np.sum(np.abs(oc) ** 2) / np.sum(np.abs(ref) ** 2))
            seen[frac] = (rel, gain,
                          any('UNDER-SAMPLED' in str(x.message) for x in w))
        rel_bad, gain_bad, fired_bad = seen[0.1]
        rel_ok, gain_ok, fired_ok = seen[2.0]
        assert fired_bad and not fired_ok
        assert rel_ok < 1e-2, (
            f'inside the bound the coarse run should track the 8x oracle; '
            f'got relative error {rel_ok:.4g}')
        assert rel_bad > 1.0, (
            f'at 0.1x the bound the aliased quadrature should be wrong by '
            f'order one or more; got {rel_bad:.4g}.  If this has shrunk the '
            f'guard is now warning about an accurate answer.')
        assert gain_bad > 5.0 > 1.05 > gain_ok > 0.95, (
            f'output power against the oracle: {gain_bad:.4g} below the bound '
            f'and {gain_ok:.4g} above it -- the energy gain is the symptom '
            f'the guard exists to name.')

    def test_the_message_names_the_function_the_bound_and_the_way_out(self):
        N, dx = 64, 2.0e-6
        E = _sas_window_filling(N, dx)
        z_near = N * dx ** 2 / self.LAM
        _, _, w = _sas_run(E, 0.1 * z_near, self.LAM, dx)
        assert len(w) == 1
        msg = str(w[0].message)
        # CONVENTIONS sec. 2: the function name is the first token.
        assert msg.startswith('scalable_angular_spectrum_propagate: ')
        assert 'N*dx^2/wavelength' in msg
        assert f'{z_near:.6g}' in msg
        assert 'angular_spectrum_propagate' in msg
        assert 'pad=2' in msg

    def test_the_warning_is_attributed_to_the_caller_not_to_sas_py(self):
        """``stacklevel`` has to reach the caller's frame, or the warning
        points a user at library source they did not write (WP-B11b item 8 is
        the same property swept over the lens family)."""
        N, dx = 64, 2.0e-6
        E = _sas_window_filling(N, dx)
        _, _, w = _sas_run(E, 0.1 * N * dx ** 2 / self.LAM, self.LAM, dx)
        assert len(w) == 1
        # the call is made inside ``_sas_run`` in THIS file
        assert pathlib.Path(w[0].filename).name == pathlib.Path(__file__).name

    def test_the_two_validity_bounds_are_distinguishable_and_bracket_a_window(
            self):
        """``z_limit`` bounds ``z`` from above and the new bound from below;
        the pair is a window, not a contradiction.  MEASURED over eight grids
        the ratio ``z_limit / z_near`` runs 45.9 (N = 1024, dx = 0.5 um) to
        5.6e6 (the covering-array doublet's in-glass gap), so the window is
        never empty on a realistic grid.
        """
        N, dx = 64, 2.0e-6
        E = _sas_window_filling(N, dx)
        z_near = N * dx ** 2 / self.LAM
        _, _, near = _sas_run(E, 0.1 * z_near, self.LAM, dx)
        _, _, far = _sas_run(E, 100.0, self.LAM, dx)
        _, _, mid = _sas_run(E, 4.0 * z_near, self.LAM, dx)
        assert [('UNDER-SAMPLED' in str(x.message)) for x in near] == [True]
        assert [('z_limit' in str(x.message)) for x in far] == [True]
        assert mid == []


# ===========================================================================
# Item 5 (part b) -- the LensPhysics configuration object
# ===========================================================================

_PHYS_RX = dict(
    surfaces=[dict(radius=0.05, glass_before='AIR', glass_after='N-BK7'),
              dict(radius=-0.05, glass_before='N-BK7', glass_after='AIR')],
    thicknesses=[3.0e-3], aperture_diameter=4.0e-4)


def _phys_case():
    return (np.ones((8, 8), dtype=np.complex128),
            dict(prescription=_PHYS_RX, wavelength=633e-9, dx=1e-4))


class TestLensPhysics:
    """The fourth configuration object, and the properties that make it the
    same object as the other three rather than a look-alike.

    The structural census (table-vs-signature agreement, parameter
    classification, the refusal parametrisation) lives in
    ``test_audit2609_a16_lens_config_round_trip.py`` with its three siblings.
    What is here is the part specific to the fourth role: the line it is drawn
    on is measurable, and the cross-object rules it deliberately does not
    restate still fire.
    """

    def test_the_line_against_lensnumerics_is_measurable_not_asserted(self):
        """The documented distinction -- a ``LensNumerics`` field moves the
        answer by its own TRUNCATION error, a ``LensPhysics`` field moves it by
        a TERM -- has a falsifiable form: refine the numerics knob and the
        answer converges; refine anything and the physics term does not appear.

        Witness: ``sag_chunk_rows`` (a pure discretisation of the same screen,
        byte-identical by design) and ``remap_order`` against ``fresnel``.
        """
        from lumenairy import LensNumerics, LensPhysics, LensResources
        from lumenairy.elements._lens_real import apply_real_lens
        e, kw = _phys_case()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plain = apply_real_lens(e, **kw)
            chunked = apply_real_lens(
                e, resources=LensResources(sag_chunk_rows=2), **kw)
            with_term = apply_real_lens(
                e, physics=LensPhysics(fresnel=True), **kw)
            term_and_chunked = apply_real_lens(
                e, physics=LensPhysics(fresnel=True),
                resources=LensResources(sag_chunk_rows=2), **kw)
        assert np.array_equal(plain, chunked), (
            'the discretisation witness moved the answer, so it cannot stand '
            'for "changes nothing but the truncation error" here')
        assert not np.array_equal(plain, with_term)
        # and the term is orthogonal to the discretisation: turning it on
        # moves the answer by the SAME amount at either chunking.
        assert np.array_equal(with_term, term_and_chunked)
        assert float(np.max(np.abs(with_term - plain))) > 1e-3 * float(
            np.max(np.abs(plain))), (
            'the fresnel transmittances move this fixture by less than 0.1 % '
            'of peak, which is too small to distinguish a term from a '
            'truncation error.  Pick a different witness.')

    def test_a_physics_request_is_refused_where_the_term_does_not_exist(self):
        """The empty ``_PHYSICS_FOR`` entries are load-bearing: they are what
        turns a physics request handed to a ray-traced engine into a refusal
        naming the owner, instead of a silently discarded setting."""
        from lumenairy.elements import lens_config as lc
        from lumenairy.elements._lens_traced import apply_real_lens_traced
        from lumenairy import LensConfig, LensPhysics
        assert lc._PHYSICS_FOR['apply_real_lens_traced'] == {}
        e, kw = _phys_case()
        with pytest.raises(ValueError) as exc:
            apply_real_lens_traced(
                e, config=LensConfig(physics=LensPhysics(absorption=True)),
                **kw)
        msg = str(exc.value)
        assert msg.startswith('apply_real_lens_traced:')
        assert 'physics.absorption' in msg and 'apply_real_lens' in msg

    def test_the_cross_object_rules_still_fire_through_the_config(self):
        """``LensPhysics.__post_init__`` deliberately checks only what a field
        can be judged on alone, so the pairs that need a sibling object -- or
        the prescription -- must still be adjudicated by the call.  Three of
        them, each through the CONFIG spelling rather than the keyword one, so
        a config cannot be a way around a guard."""
        from lumenairy import LensConfig, LensGeometry, LensPhysics
        from lumenairy.elements._lens_real import apply_real_lens
        e, kw = _phys_case()
        # 1. the same per-surface coefficient, twice
        with pytest.raises(ValueError) as exc:
            apply_real_lens(e, physics=LensPhysics(slant_correction=True,
                                                   seidel_correction=True),
                            **kw)
        assert 'apply_real_lens:' in str(exc.value)
        # 2. a model term under surface_model='displaced'
        with pytest.raises(ValueError) as exc:
            apply_real_lens(e, config=LensConfig(
                geometry=LensGeometry(surface_model='displaced'),
                physics=LensPhysics(fresnel=True)), **kw)
        assert 'apply_real_lens:' in str(exc.value)
        # 3. screen_obliquity=True with no carrier
        with pytest.raises(ValueError) as exc:
            apply_real_lens(e, physics=LensPhysics(screen_obliquity=True),
                            **kw)
        assert 'carrier' in str(exc.value)
        # ... and the same three built as objects do NOT raise on their own:
        # the refusal belongs to the call, which is the whole point.
        LensPhysics(slant_correction=True, seidel_correction=True)
        LensPhysics(fresnel=True)
        LensPhysics(screen_obliquity=True)

    def test_narrowed_to_reaches_the_new_group(self):
        """Every ``_GROUPS`` walker had to pick the fourth group up for free;
        ``narrowed_to`` is the one whose failure would be silent (it would
        simply not reset the physics fields, and the next call would raise)."""
        from lumenairy import LensConfig
        cfg = LensConfig.from_kwargs(fresnel=True, newton_poly_order=8,
                                     bandlimit=False)
        narrowed = cfg.narrowed_to('apply_real_lens_traced')
        assert narrowed.physics.fresnel is False
        assert narrowed.to_kwargs() == {'newton_poly_order': 8,
                                        'bandlimit': False}
        assert cfg.narrowed_to('apply_real_lens').to_kwargs() == {
            'fresnel': True, 'bandlimit': False}

    def test_to_kwargs_strict_covers_the_new_group_too(self):
        """Item 6's refusal (part a) has to see physics requests, or a caller
        splatting a physics-carrying config into a traced call gets the silent
        drop back."""
        from lumenairy import LensConfig
        cfg = LensConfig.from_kwargs(fresnel=True, bandlimit=False)
        assert cfg.to_kwargs(entry_point='apply_real_lens_traced') == {
            'bandlimit': False}
        with pytest.raises(ValueError) as exc:
            cfg.to_kwargs(entry_point='apply_real_lens_traced', strict=True)
        assert 'fresnel' in str(exc.value) and 'physics' in str(exc.value)

    def test_input_wavevector_saddle_is_still_keyword_only_with_the_reason(
            self):
        """The decision the work package was asked to make, pinned so that
        moving it later is a deliberate act with a visible test change."""
        from lumenairy.elements import lens_config as lc
        assert 'input_wavevector_saddle' not in lc.LensConfig.field_names()
        reason = lc.KWARG_ONLY['apply_real_lens_maslov'][
            'input_wavevector_saddle']
        assert 'INPUT FIELD' in reason
        assert 'LensPhysics' in reason, (
            'the exclusion predates LensPhysics; it must say that it was '
            're-examined when the fourth object landed, or a later reader '
            'cannot tell a decision from an oversight.')

    @pytest.mark.parametrize('name', [
        'surface_model', 'caustic', 'fit_basis'])
    def test_the_three_settings_that_did_not_move_are_where_they_were(self,
                                                                      name):
        """A field that moves between two shipped config objects is a
        migration.  These three fit the physics role by the definition above
        and were deliberately left; this is the pin that makes moving one
        deliberate."""
        from lumenairy import LensGeometry, LensNumerics, LensPhysics
        import dataclasses
        where = {f.name: 'geometry' for f in dataclasses.fields(LensGeometry)}
        where.update({f.name: 'numerics'
                      for f in dataclasses.fields(LensNumerics)})
        where.update({f.name: 'physics'
                      for f in dataclasses.fields(LensPhysics)})
        assert where[name] == ('geometry' if name == 'surface_model'
                               else 'numerics')


# ===========================================================================
# Item 8 (part b) -- the doe.py zero fill, and warning attribution
# ===========================================================================

class TestTheZonePlateZeroFill:
    """``create_fresnel_zone_plate``'s outside-the-aperture fill takes ``T``'s
    own dtype instead of the literal ``0.0 + 0j``.

    WP-A22's structural walk found the site and rated it P3 BY MEASUREMENT --
    the phase is built from Python float literals, so ``T`` is complex128 on
    every reachable call and nothing was promoted -- then allowlisted it with
    the one-line migration written at the entry.  This is that migration, so
    the allowlist entry is gone and the walk now confirms the site.

    WP-A22's forward-looking half of that rating does NOT survive
    re-measurement, and this class records the correction: see
    ``test_what_the_literal_fill_actually_promotes``.
    """

    @pytest.mark.parametrize('binary,n_zones', [
        (True, None), (True, 4), (False, None), (False, 4), (False, 1)])
    def test_the_shipped_answer_is_unchanged(self, binary, n_zones):
        """The returned transmission and its dtype, on both branches and with
        the aperture both active and inactive.  ``f = 0.2 mm`` puts 36 zones
        across this grid, so ``n_zones=4`` really clips."""
        from lumenairy.elements.doe import create_fresnel_zone_plate
        T = create_fresnel_zone_plate(48, 2.0e-6, 0.2e-3, 633e-9,
                                      binary=binary, n_zones=n_zones)
        assert T.dtype == np.complex128
        assert np.all(np.isfinite(T))
        if n_zones is not None:
            # the aperture is real: something outside it is exactly zero
            assert np.any(T == 0)
            assert np.any(T != 0)

    @pytest.mark.parametrize('dt', ['complex64', 'complex128',
                                    'float32', 'float64'])
    def test_the_fill_follows_T_dtype_whatever_T_is(self, dt):
        """The property the migration buys, stated as an invariant rather than
        as a difference: ``np.zeros((), T.dtype)`` is dtype-preserving for
        EVERY ``T``, by construction, and does not depend on how the NumPy in
        use promotes a Python scalar."""
        T = np.ones((4, 4), dtype=dt)
        inside = np.ones((4, 4), dtype=bool)
        inside[0, 0] = False
        assert np.where(inside, T, np.zeros((), T.dtype)).dtype == T.dtype

    def test_what_the_literal_fill_actually_promotes(self):
        """MEASURED 2026-09-14 on NumPy 2.4.6 -- and it CORRECTS WP-A22's
        rationale, which is why it is a test and not a comment.

        That report rated the site P3 today and "P1 the moment the phase is
        built at a narrower dtype".  Under NEP 50 weak promotion a Python
        complex scalar does not widen a complex array at all, so a complex64
        ``T`` would keep complex64 with the literal too; the arm where the
        literal really does change the dtype is a REAL ``T``, which this entry
        point's ``exp(1j * phase)`` can never produce.

        ========== =================== ========================
        T.dtype    literal ``0.0+0j``  ``np.zeros((), T.dtype)``
        ========== =================== ========================
        complex64  complex64           complex64
        complex128 complex128          complex128
        float32    complex64           float32
        float64    complex128          float64
        ========== =================== ========================

        So the migration is worth making because it is explicit and
        version-independent -- NumPy 1.x decided this by value-based casting
        and 2.x by weak promotion -- not because a promotion was waiting to
        happen here.  If this table moves, the reasoning above moves with it.
        """
        inside = np.ones((4, 4), dtype=bool)
        inside[0, 0] = False
        got = {dt: str(np.where(inside, np.ones((4, 4), dtype=dt),
                                0.0 + 0j).dtype)
               for dt in ('complex64', 'complex128', 'float32', 'float64')}
        assert got == {'complex64': 'complex64', 'complex128': 'complex128',
                       'float32': 'complex64', 'float64': 'complex128'}, got

    def test_the_dispatcher_pin_no_longer_exempts_the_site(self):
        import tests.unit.test_v4_14_2_dispatcher_pin_zero_plus_zeroj as pin
        assert not any(p.endswith('doe.py')
                       for p, _ in pin._P3_ALLOWLIST), (
            'the allowlist still exempts doe.py; the migration has landed, so '
            'the walk should confirm the site rather than skip it.')


class TestWarningAttribution:
    """Every warning the lens bodies raise must name the caller's frame.

    ``stacklevel`` counts frames, so a literal is right for exactly one call
    path -- and this family has several to the same source line: the public
    wrapper that owns the accumulator-store context, the configuration
    objects' self-re-entry, ``apply_real_lens_traced`` reaching
    ``apply_real_lens`` internally, and nested closures.  MEASURED before the
    fix: a configured ``apply_real_lens`` call attributed its aperture notice
    to ``_lens_real.py``'s own re-entry line, and ``prepare_real_lens_traced``
    attributed all five of its notices to ``_lens_traced.py``.
    """

    RX = dict(
        surfaces=[dict(radius=0.05, glass_before='AIR', glass_after='N-BK7'),
                  dict(radius=-0.05, glass_before='N-BK7',
                       glass_after='AIR')],
        thicknesses=[3.0e-3], aperture_diameter=4.0e-3)   # aperture > grid

    def _warned(self, fn, *a, **kw):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fn(*a, **kw)
        return caught

    def test_the_computed_level_is_the_first_frame_outside_the_package(self):
        """The helper itself, against a hand-built stack of known depth."""
        from lumenairy.elements._lens_kernels import caller_stacklevel

        def inner():
            return caller_stacklevel()

        def outer():
            return inner()

        # Both frames are in THIS file, which is outside lumenairy, so the
        # first frame outside the package is the immediate one: level 1.
        assert inner() == 1
        assert outer() == 1
        # ... and with the package as the "root", every frame of this file
        # counts as library code, so it walks to the outermost frame instead
        # of returning 1.  That is the "no user frame" arm.
        here = str(pathlib.Path(__file__).parent) + os.sep
        assert caller_stacklevel(_root=here) > 1

    def test_a_configured_call_still_names_the_callers_frame(self):
        """The regression the configuration objects introduced: the re-entry
        ``return apply_real_lens(E_in, **resolve(...))`` adds one frame, which
        every hard-coded level in the body was short by."""
        from lumenairy import LensNumerics, LensPhysics
        from lumenairy.elements._lens_real import apply_real_lens
        e = np.ones((8, 8), dtype=np.complex128)
        kw = dict(prescription=self.RX, wavelength=633e-9, dx=1e-4)
        me = pathlib.Path(__file__).name
        for label, extra in (('plain', {}),
                             ('numerics', {'numerics': LensNumerics(
                                 bandlimit=False)}),
                             ('physics', {'physics': LensPhysics(
                                 absorption=True)})):
            caught = self._warned(apply_real_lens, e, **extra, **kw)
            got = [w for w in caught if 'aperture(s) exceed' in str(w.message)]
            assert got, f'{label}: the aperture notice did not fire'
            for w in got:
                assert pathlib.Path(w.filename).name == me, (
                    f'{label}: the notice names {w.filename}:{w.lineno}, not '
                    f'the caller.  A warning that points at library source '
                    f'tells the reader where the library called itself.')

    def test_the_traced_entry_points_name_the_callers_frame(self):
        """Including the notices ``apply_real_lens_traced`` raises through an
        internal ``apply_real_lens`` call, and ``prepare_real_lens_traced``'s,
        which the literal levels could not reach at all."""
        from lumenairy.elements._lens_traced import prepare_real_lens_traced
        me = pathlib.Path(__file__).name
        caught = self._warned(prepare_real_lens_traced,
                              prescription=self.RX, wavelength=633e-9,
                              dx=1e-5, N=64, ray_subsample=1)
        got = [w for w in caught if issubclass(w.category,
                                               (UserWarning, RuntimeWarning))]
        assert len(got) >= 3, f'expected the pre-flight notices, got {got}'
        bad = [f'{pathlib.Path(w.filename).name}:{w.lineno}' for w in got
               if pathlib.Path(w.filename).name != me]
        assert not bad, (
            f'{len(bad)} of {len(got)} notices name library source: {bad}')

    def test_a_body_without_config_re_entry_names_the_caller_too(self):
        """``_lens_thin.py`` has no configuration re-entry, so its literal
        ``stacklevel=2`` was RIGHT before the sweep; the computed level has to
        give the same answer where the literal was correct, not only where it
        was one frame short.  ``apply_grin_lens`` beyond the quarter pitch is
        the fixture: one ``UserWarning`` from the thin body, and the frame it
        names must be this file's call."""
        from lumenairy.elements._lens_thin import apply_grin_lens
        me = pathlib.Path(__file__).name
        e = np.ones((16, 16), dtype=np.complex128)
        # g*d = 2.0 > pi/2: the single-screen reduction warns
        caught = self._warned(apply_grin_lens, e, n0=1.5, g=1000.0, d=2e-3,
                              wavelength=633e-9, dx=2e-6)
        got = [w for w in caught if 'quarter pitch' in str(w.message)]
        assert len(got) == 1, f'expected one quarter-pitch notice, got {caught}'
        assert issubclass(got[0].category, UserWarning)
        assert pathlib.Path(got[0].filename).name == me, (
            f'the notice names {got[0].filename}:{got[0].lineno}, not the caller')

    def test_no_literal_stacklevel_is_left_in_the_swept_lens_bodies(self):
        """The ratchet.  A literal that creeps back in is right for one call
        path and wrong for the others, and the failure is silent -- the
        warning still fires, it just points at the wrong file.  The tuple
        is every lens body whose warnings were swept onto the helper: the
        analytic and traced bodies, the Maslov (11 sites), GBD (1),
        multibranch (4), thin (2) and inverse-map (1) modules.  Still
        outside it: ``_lens_traced_uniform.py`` (1 site) and
        ``propagators/carrier.py``, recorded in WP-B11 section 4b."""
        for rel in ('lumenairy/elements/_lens_real.py',
                    'lumenairy/elements/_lens_traced.py',
                    'lumenairy/elements/lenses_maslov.py',
                    'lumenairy/elements/lenses_gbd.py',
                    'lumenairy/elements/_lens_traced_multibranch.py',
                    'lumenairy/elements/_lens_thin.py',
                    'lumenairy/elements/_lens_imap.py'):
            src = (REPO / rel).read_text(encoding='utf-8')
            tree = ast.parse(src)
            bad = []
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'warn'):
                    continue
                args = list(node.args) + [k.value for k in node.keywords
                                          if k.arg == 'stacklevel']
                for a in args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, int):
                        bad.append(node.lineno)
            assert not bad, (
                f'{rel}: warnings.warn with a LITERAL stacklevel at lines '
                f'{sorted(set(bad))}.  Use _caller_stacklevel(), which walks '
                f'out to the first frame outside the package.')


# ===========================================================================
# Item 4 (part b) -- PMM2DStackHybrid.truncation joins the guarded attributes
# ===========================================================================

class TestStack2DTruncationGuard:
    """The fourth validated model choice.  Part a guarded ``formulation`` /
    ``cascade`` / ``symmetry`` and recorded that ``truncation`` had the same
    unguarded-after-``__init__`` shape; this closes it with the same pattern
    and the same shared vocabulary."""

    def test_the_constructor_and_the_setter_share_one_vocabulary(self):
        from lumenairy.elements.pmm.stack2d import (
            _TRUNCATIONS, PMM2DStackHybrid,
        )
        for value in list(_TRUNCATIONS) + ['circle', 'rect', '', None, 0]:
            by_init = True
            try:
                PMM2DStackHybrid(0.7e-6, truncation=value)
            except ValueError:
                by_init = False
            st = PMM2DStackHybrid(0.7e-6)
            by_set = True
            try:
                st.truncation = value
            except ValueError:
                by_set = False
            assert by_init == by_set == (value in _TRUNCATIONS), (
                f'truncation={value!r}: __init__ '
                f'{"took" if by_init else "refused"} it, the setter '
                f'{"took" if by_set else "refused"} it, vocabulary says '
                f'{value in _TRUNCATIONS}')

    def test_the_refusal_carries_the_conventions_prefix(self):
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        st = PMM2DStackHybrid(0.7e-6)
        with pytest.raises(ValueError) as exc:
            st.truncation = 'circle'
        msg = str(exc.value)
        assert msg.startswith('PMM2DStackHybrid:')
        assert 'circle' in msg and 'circular' in msg

    def test_a_legal_reassignment_still_takes_effect(self):
        """The guard is a refusal, not a freeze -- and the order set really
        moves, which is what makes a silent typo expensive: it buys back the
        larger, slower, DIFFERENT rectangular answer with nothing said."""
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        st = PMM2DStackHybrid(0.7e-6, n_orders=5)
        n = 2 * 5 + 1
        ox, oy = np.meshgrid(np.arange(-5, 6), np.arange(-5, 6),
                             indexing='ij')
        st.truncation = 'circular'
        assert st.truncation == 'circular'
        circ = st._order_keep_mask(ox, oy)
        assert circ is not None
        kept = int(np.count_nonzero(circ))
        st.truncation = 'rectangular'
        assert st.truncation == 'rectangular'
        assert st._order_keep_mask(ox, oy) is None
        assert 0 < kept < n * n, (
            f'the circular truncation kept {kept} of {n * n} orders; if it '
            f'kept all of them the two settings would be the same answer and '
            f'this guard would be cosmetic.')

    def test_every_validated_model_choice_is_now_a_property(self):
        """The census, so a fifth one added as a plain attribute is caught."""
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        for name in ('formulation', 'cascade', 'symmetry', 'truncation'):
            assert isinstance(getattr(PMM2DStackHybrid, name, None),
                              property), (
                f'{name} is not a property, so an out-of-vocabulary '
                f'assignment after __init__ is accepted silently')
