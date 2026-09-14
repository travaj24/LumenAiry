"""WP-B11a -- the hygiene pass, part a: the consolidations and the small
deferred items, each with the property its refactor was gated on.

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
"""
from __future__ import annotations

import ast
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
