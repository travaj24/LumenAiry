"""v5.1.0 Agent F regression tests -- ``lumenairy/io/prescriptions.py``
6-file split (ROADMAP v5.1.0 structural-reorganisation work item).

Closes the v5.1.0 ROADMAP item that splits the 3224-LOC
``lumenairy/io/prescriptions.py`` monolith into five sibling
submodules:

* ``prescriptions_builders``       (singlet / doublet / Thorlabs)
* ``prescriptions_zemax``          (.zmx / .txt I/O)
* ``prescriptions_code_v``         (.seq I/O)
* ``prescriptions_quadoa``         (.qos I/O)
* ``prescriptions_transforms``     (scale / normalize / fold-split / mirrors)

The split is **mechanical**: no parsing logic, no signature, and no
docstring text changes.  This test module pins the public API
contract that v5.1.0 promises: every name previously importable from
``lumenairy.io.prescriptions`` (and re-exported through
``lumenairy``) is still importable from the same locations and
points at the same callable object.

The tests are deliberately structural / smoke -- the deep parsing
correctness is covered by the existing v4.x prescription-parser
test suite which continues to run against the re-exported names.
"""
from __future__ import annotations

import importlib

import pytest

import lumenairy as la
from lumenairy.io import prescriptions as rx_mod


# ---------------------------------------------------------------------------
# Public-API surface: every previously-exported name must be importable
# from ``lumenairy.io.prescriptions`` AND ``lumenairy`` (top-level
# re-export) with bit-for-bit identical objects.
# ---------------------------------------------------------------------------

# Names that were importable from ``lumenairy.io.prescriptions`` at
# the v5.0.x freeze of the monolith.  Source: the ``__init__.py``
# import list (also mirrored by the top-level ``lumenairy/__init__.py``
# import block).
_PRESCRIPTION_PUBLIC_NAMES = (
    # builders + Thorlabs catalog
    'make_singlet',
    'make_doublet',
    'make_cylindrical',
    'make_biconic',
    'make_off_axis_parabola',
    'thorlabs_lens',
    'THORLABS_CATALOG',
    # Zemax I/O
    'load_zemax_zmx',
    'load_zemax_prescription_data_txt',
    'export_zemax_lens_data',
    'export_zemax_zmx',
    # CODE V I/O
    'load_codev_seq',
    'export_codev_seq',
    # Quadoa I/O
    'export_quadoa_qos',
    'load_quadoa_qos',
    'QUADOA_SCHEMA_VERSION',
    # transforms
    'scale_prescription',
    'normalize_prescription',
    'split_prescription_at_mirrors',
    'has_mirrors',
)


@pytest.mark.parametrize("name", _PRESCRIPTION_PUBLIC_NAMES)
def test_public_name_re_exported_from_prescriptions_module(name):
    """Every documented prescription public name must remain importable
    from ``lumenairy.io.prescriptions``."""
    assert hasattr(rx_mod, name), (
        f"{name!r} is missing from lumenairy.io.prescriptions after "
        f"the v5.1.0 6-file split."
    )


@pytest.mark.parametrize("name", _PRESCRIPTION_PUBLIC_NAMES)
def test_public_name_re_exported_from_top_level(name):
    """Every documented prescription public name must also remain
    importable from the top-level ``lumenairy`` namespace."""
    assert hasattr(la, name), (
        f"{name!r} is missing from the top-level lumenairy namespace "
        f"after the v5.1.0 6-file split."
    )


@pytest.mark.parametrize("name", _PRESCRIPTION_PUBLIC_NAMES)
def test_top_level_and_submodule_re_export_same_object(name):
    """The same callable / constant must be exposed via both import
    paths so callers and ``isinstance`` checks can't accidentally
    diverge."""
    assert getattr(la, name) is getattr(rx_mod, name), (
        f"{name!r} re-export skew: ``lumenairy.{name}`` and "
        f"``lumenairy.io.prescriptions.{name}`` resolve to different "
        f"objects."
    )


# ---------------------------------------------------------------------------
# Per-submodule placement: the function / constant must physically live
# in its v5.1.0 submodule (not just be re-exported).  Pin this so a
# future maintainer cannot quietly fold a function back into
# ``prescriptions.py`` and re-grow the monolith.
# ---------------------------------------------------------------------------

_EXPECTED_SUBMODULE = {
    # builders + Thorlabs catalog
    'make_singlet': 'lumenairy.io.prescriptions_builders',
    'make_doublet': 'lumenairy.io.prescriptions_builders',
    'make_cylindrical': 'lumenairy.io.prescriptions_builders',
    'make_biconic': 'lumenairy.io.prescriptions_builders',
    'make_off_axis_parabola': 'lumenairy.io.prescriptions_builders',
    'thorlabs_lens': 'lumenairy.io.prescriptions_builders',
    # Zemax I/O
    'load_zemax_zmx': 'lumenairy.io.prescriptions_zemax',
    'load_zemax_prescription_data_txt': 'lumenairy.io.prescriptions_zemax',
    'export_zemax_lens_data': 'lumenairy.io.prescriptions_zemax',
    'export_zemax_zmx': 'lumenairy.io.prescriptions_zemax',
    # CODE V I/O
    'load_codev_seq': 'lumenairy.io.prescriptions_code_v',
    'export_codev_seq': 'lumenairy.io.prescriptions_code_v',
    # Quadoa I/O
    'export_quadoa_qos': 'lumenairy.io.prescriptions_quadoa',
    'load_quadoa_qos': 'lumenairy.io.prescriptions_quadoa',
    # transforms
    'scale_prescription': 'lumenairy.io.prescriptions_transforms',
    'normalize_prescription': 'lumenairy.io.prescriptions_transforms',
    'split_prescription_at_mirrors': 'lumenairy.io.prescriptions_transforms',
    'has_mirrors': 'lumenairy.io.prescriptions_transforms',
}


@pytest.mark.parametrize("name,expected_mod",
                         sorted(_EXPECTED_SUBMODULE.items()))
def test_function_lives_in_expected_submodule(name, expected_mod):
    """Each callable's ``__module__`` must match the v5.1.0 submodule
    placement so the split doesn't silently regress."""
    fn = getattr(rx_mod, name)
    assert getattr(fn, '__module__', None) == expected_mod, (
        f"{name!r} should live in {expected_mod!r}, got "
        f"{getattr(fn, '__module__', None)!r}."
    )


def test_all_five_submodules_import_cleanly():
    """The five new submodules must each import without raising."""
    for modname in (
        'lumenairy.io.prescriptions_builders',
        'lumenairy.io.prescriptions_zemax',
        'lumenairy.io.prescriptions_code_v',
        'lumenairy.io.prescriptions_quadoa',
        'lumenairy.io.prescriptions_transforms',
    ):
        importlib.import_module(modname)


def test_prescriptions_shell_is_thin():
    """The post-split ``prescriptions.py`` should be a thin re-export
    shell, not a parser monolith.  The original was 3224 LOC; pin a
    generous upper bound (300 LOC) so a future contributor can't
    quietly fold a parser back in without the test noticing.

    Reads the source from disk directly (not ``inspect.getsource``) so
    the result doesn't depend on ``linecache`` state pre-warmed by
    sibling tests that pulled stack frames out of this module.
    """
    from pathlib import Path
    path = Path(rx_mod.__file__)
    nlines = len(path.read_text(encoding='utf-8').splitlines())
    assert nlines < 300, (
        f"lumenairy/io/prescriptions.py grew back to {nlines} LOC; "
        f"after the v5.1.0 Agent F split it should be a thin re-export "
        f"shell (under ~200 LOC)."
    )


# ---------------------------------------------------------------------------
# Smoke: end-to-end builder / scale / normalize round-trip through the
# top-level public API, exercised after the split.  This catches the
# case where a re-export points at a callable that has lost an attribute
# or default that the v5.0.x monolith carried.
# ---------------------------------------------------------------------------

def test_make_singlet_round_trips_through_transforms():
    """Build a singlet, scale it, normalize it -- exercise all three
    submodule families in one go."""
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    assert rx['surfaces'][0]['radius'] == 50e-3
    assert rx['surfaces'][1]['radius'] == -50e-3
    assert rx['thicknesses'] == [4e-3]
    assert rx['aperture_diameter'] == 10e-3

    # scale (transforms module)
    rx_small = la.scale_prescription(rx, 0.5)
    assert rx_small['surfaces'][0]['radius'] == 25e-3
    assert rx_small['aperture_diameter'] == 5e-3
    assert rx_small['thicknesses'] == [2e-3]

    # normalize (transforms module)
    nrx = la.normalize_prescription(rx)
    assert 'elements' in nrx
    assert 'all_thicknesses' in nrx
    assert nrx['elements'] == nrx['surfaces']

    # has_mirrors (transforms module): pure refractive -> False
    assert la.has_mirrors(rx) is False
    assert la.has_mirrors(nrx) is False


def test_make_doublet_basic_shape():
    """Doublet factory produces three-surface schema with the cemented
    glass pair correctly chained."""
    rx = la.make_doublet(
        R1=33.3e-3, R2=-24.1e-3, R3=-95.3e-3,
        d1=9e-3, d2=3e-3,
        glass1='N-BAF10', glass2='N-SF6HT',
        aperture=25.4e-3)
    assert len(rx['surfaces']) == 3
    assert rx['thicknesses'] == [9e-3, 3e-3]
    s = rx['surfaces']
    assert s[0]['glass_before'] == 'air' and s[0]['glass_after'] == 'N-BAF10'
    assert s[1]['glass_before'] == 'N-BAF10' and s[1]['glass_after'] == 'N-SF6HT'
    assert s[2]['glass_before'] == 'N-SF6HT' and s[2]['glass_after'] == 'air'


def test_thorlabs_catalog_singlet_and_doublet():
    """The Thorlabs catalog re-export must still resolve both singlet
    and doublet entries via :func:`thorlabs_lens`."""
    # Singlet
    rx_sg = la.thorlabs_lens('LA1050-C')
    assert rx_sg['name'] == 'LA1050-C'
    assert len(rx_sg['surfaces']) == 2
    # Doublet
    rx_db = la.thorlabs_lens('AC254-200-C')
    assert rx_db['name'] == 'AC254-200-C'
    assert len(rx_db['surfaces']) == 3
    # Catalog dict re-export
    assert 'AC254-200-C' in la.THORLABS_CATALOG
    assert la.THORLABS_CATALOG['AC254-200-C']['type'] == 'doublet'


def test_off_axis_parabola_geometry():
    """``make_off_axis_parabola`` exercises the geometric input
    validation + decenter relation ``h = 2*f*tan(alpha)``."""
    import math
    f = 152.4e-3
    alpha = math.pi / 4  # 90-deg-fold OAP
    rx = la.make_off_axis_parabola(
        focal_length=f, off_axis_angle=alpha, clear_aperture=50.8e-3)
    assert rx['surfaces'][0]['conic'] == -1.0
    h_expected = 2.0 * f * math.tan(alpha)
    h_got = rx['surfaces'][0]['decenter'][0]
    assert abs(h_got - h_expected) < 1e-12


def test_codev_seq_round_trip(tmp_path):
    """Export a singlet to CODE V .seq, then load it back.  Round-trip
    sanity check across the prescriptions_code_v boundary."""
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    path = tmp_path / 'singlet.seq'
    la.export_codev_seq(rx, str(path), wavelength=1.31e-6,
                        aperture_diameter=10e-3)
    rx2 = la.load_codev_seq(str(path))
    assert abs(rx2['surfaces'][0]['radius'] - 50e-3) < 1e-12
    assert abs(rx2['surfaces'][1]['radius'] - (-50e-3)) < 1e-12
    assert abs(rx2['thicknesses'][0] - 4e-3) < 1e-12


def test_quadoa_qos_round_trip(tmp_path):
    """Export a singlet to Quadoa .qos, then load it back.  Round-trip
    sanity check across the prescriptions_quadoa boundary."""
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    path = tmp_path / 'singlet.qos'
    la.export_quadoa_qos(rx, str(path), wavelength=1.31e-6)
    rx2 = la.load_quadoa_qos(str(path))
    assert abs(rx2['surfaces'][0]['radius'] - 50e-3) < 1e-12
    assert abs(rx2['surfaces'][1]['radius'] - (-50e-3)) < 1e-12
    assert abs(rx2['thicknesses'][0] - 4e-3) < 1e-12
    assert la.QUADOA_SCHEMA_VERSION == '1.0'


def test_zemax_export_zmx_writes_file(tmp_path):
    """Export a singlet to Zemax .zmx and confirm the file exists and
    contains the expected header keywords.  Catches the case where the
    Zemax submodule's writer was accidentally not re-exported."""
    rx = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
    path = tmp_path / 'singlet.zmx'
    la.export_zemax_zmx(rx, str(path), wavelength=1.31e-6)
    assert path.exists()
    text = path.read_text(encoding='utf-8')
    assert 'MODE SEQ' in text
    assert 'SURF 0' in text
    assert 'WAVM 1' in text
