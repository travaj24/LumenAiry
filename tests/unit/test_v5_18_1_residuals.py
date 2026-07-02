"""v5.18.1 post-release residual fixes (deferred items from the AUDIT_V5_17_0
campaign, cleaned up after the v5.18.0 release).

Covered here:
  * ITEM 1  waveoptics_dock load-saved-run used a nonexistent ``self.model``
            (the dock stores the model as ``self.sm``) -> AttributeError
            swallowed as a "Load failed" dialog; the feature never worked.
  * ITEM 2a ``_fd_eig_dist`` (the ``verify=True`` FD oracle) called ARPACK
            ``eigs`` with a random start vector -> non-deterministic
            ``layer_vector_modes(verify=True)`` run-to-run.  Now seeds ``v0``.
  * ITEM 5  the Zemax ``.txt`` paste-table exporter hardcoded ``TYPE=STANDARD``
            for every surface -> aspheric surfaces were mislabelled (the
            export-side sibling of the P3-43 .txt-loader drop).

These are CI-shape safe: no Qt (source-level check) and no jax.
"""
import pathlib

import numpy as np
import pytest

import lumenairy as la


# --------------------------------------------------------------------------- #
# ITEM 1 -- waveoptics_dock references self.sm, never self.model               #
# --------------------------------------------------------------------------- #
def _waveoptics_dock_source() -> str:
    src = pathlib.Path(la.__file__).parent / 'ui' / 'waveoptics_dock.py'
    return src.read_text(encoding='utf-8')


def test_waveoptics_dock_uses_self_sm_not_self_model():
    """The dock stores the model as ``self.sm``; ``self.model`` never existed,
    so any ``self.model.<...>`` reference is a latent AttributeError.  Check
    CODE only (strip inline/full-line comments so this file's own explanatory
    comment mentioning the old name does not trip the assertion)."""
    src = _waveoptics_dock_source()
    code_only = '\n'.join(line.split('#', 1)[0] for line in src.splitlines())
    assert 'self.sm.load_prescription(' in code_only, (
        'load-saved-run must call self.sm.load_prescription')
    assert 'self.model' not in code_only, (
        'WaveOpticsDock has no self.model attribute (it is self.sm); a '
        'self.model reference in code is a latent AttributeError')


def test_waveoptics_dock_stores_model_as_sm():
    """Guard the assumption behind the fix: __init__ binds system_model->sm."""
    src = _waveoptics_dock_source()
    assert 'self.sm = system_model' in src


# --------------------------------------------------------------------------- #
# ITEM 2a -- _fd_eig_dist is deterministic (seeded ARPACK v0)                  #
# --------------------------------------------------------------------------- #
def test_fd_eig_dist_is_deterministic():
    """Two identical calls to the verify FD oracle return byte-identical
    distances (a random ARPACK start vector previously made them wobble)."""
    from lumenairy.elements.eme.eme_2d_vector import _fd_eig_dist
    Nx, ny = 8, 24
    Lx = Ly = 1.0
    k0 = 2.0 * np.pi / 1.0
    ky0 = 0.3
    strips = [(np.full(Nx, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    qz2 = 30.0
    d1 = _fd_eig_dist(strips, Lx, Nx, Ly, k0, 0.0, ky0, qz2, ny)
    d2 = _fd_eig_dist(strips, Lx, Nx, Ly, k0, 0.0, ky0, qz2, ny)
    assert d1 == d2, f'verify FD oracle non-deterministic: {d1!r} != {d2!r}'
    assert np.isfinite(d1)


def test_layer_vector_modes_verify_is_deterministic():
    """End-to-end: verify=True gives an identical mode list across calls."""
    from lumenairy.elements.eme.eme_2d_vector import layer_vector_modes
    Nx = 12
    Lx = Ly = 1.0
    k0 = 2.0 * np.pi / 1.0
    ky0 = 0.3
    strips = [(np.full(Nx, 4.0), 0.5), (np.full(Nx, 2.0), 0.5)]
    a = layer_vector_modes(strips, Lx, Nx, Ly, k0, (60, 150), ky0=ky0,
                           n_scan=120, verify=True)
    b = layer_vector_modes(strips, Lx, Nx, Ly, k0, (60, 150), ky0=ky0,
                           n_scan=120, verify=True)
    assert np.array_equal(np.sort(a), np.sort(b))


# --------------------------------------------------------------------------- #
# ITEM 5 -- Zemax .txt exporter reflects the real surface TYPE                 #
# --------------------------------------------------------------------------- #
def test_txt_surface_type_helper():
    from lumenairy.io.prescriptions_zemax import _txt_surface_type
    assert _txt_surface_type({'radius': 0.05}) == ('STANDARD', False)
    assert _txt_surface_type({'type': 'EVENASPH'}) == ('EVENASPH', True)
    # inferred from non-zero aspheric coefficients, no explicit type
    tp, asph = _txt_surface_type({'aspheric_params': {2: 1e-6}})
    assert tp == 'EVENASPH' and asph is True
    # all-zero aspheric params are NOT aspheric
    assert _txt_surface_type({'aspheric_params': {2: 0.0}}) == ('STANDARD', False)
    # legacy list field also detected
    tp2, asph2 = _txt_surface_type({'aspheric_coeffs': [0.0, 1e-7]})
    assert tp2 == 'EVENASPH' and asph2 is True


def test_txt_exporter_labels_aspheric_surface(tmp_path):
    """A prescription with an aspheric surface exports with TYPE EVENASPH for
    that surface (not STANDARD) and a footnote naming it."""
    from lumenairy.io.prescriptions_zemax import export_zemax_lens_data
    presc = {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'glass_after': 'N-BK7',
             'semi_diameter': 0.01, 'comment': 'front (spherical)'},
            {'radius': -0.05, 'conic': 0.0, 'aspheric_params': {2: 1.0e-6},
             'semi_diameter': 0.01, 'comment': 'back (aspheric)'},
        ],
        'thicknesses': [0.003, 0.0],
        'aperture_diameter': 0.02,
    }
    out = tmp_path / 'lens.txt'
    export_zemax_lens_data(presc, str(out), wavelength=1.31e-6)
    text = out.read_text(encoding='utf-8')
    body = [ln for ln in text.splitlines() if not ln.startswith('#')]
    # exactly one EVENASPH surface row, and it is NOT the front sphere
    evenasph_rows = [ln for ln in body if 'EVENASPH' in ln]
    assert len(evenasph_rows) == 1, text
    assert 'aspheric' in evenasph_rows[0]
    # the spherical surface stays STANDARD
    assert any('STANDARD' in ln and 'spherical' in ln for ln in body)
    # footnote points at the lossless export
    assert 'export_zemax_zmx' in text
    assert 'aspheric coefficients' in text


def test_txt_exporter_all_spherical_has_no_footnote(tmp_path):
    """A purely spherical prescription keeps STANDARD rows and no aspheric
    footnote (byte-clean legacy behaviour for the common case)."""
    from lumenairy.io.prescriptions_zemax import export_zemax_lens_data
    presc = {
        'surfaces': [
            {'radius': 0.05, 'conic': 0.0, 'glass_after': 'N-BK7',
             'semi_diameter': 0.01, 'comment': 'front'},
            {'radius': -0.05, 'conic': 0.0, 'semi_diameter': 0.01,
             'comment': 'back'},
        ],
        'thicknesses': [0.003, 0.0],
        'aperture_diameter': 0.02,
    }
    out = tmp_path / 'lens.txt'
    export_zemax_lens_data(presc, str(out), wavelength=1.31e-6)
    text = out.read_text(encoding='utf-8')
    assert 'EVENASPH' not in text
    assert 'export_zemax_zmx' not in text
