"""G08 / S4-20 (AUDIT_V5_24_2) -- packaging + GUI convention hygiene.

* ``lumenairy/_validation.py`` is in the mypy strict whitelist.
* MANIFEST.in ships the top-level reference docs (Migration-Guide /
  CONVENTIONS / ROADMAP) in the sdist.
* The GUI PSF/MTF ray-traced pupil uses the conjugate phase convention
  ``exp(-i k0 OPD)`` matching the wave-optics lens phase-screens (pre-fix
  it used ``exp(+i k0 OPD)`` -> mirror-flipped PSF between the two pupil
  sources).  The GUI dock needs a Qt runtime to instantiate, so this is
  pinned at the source level against the library convention marker in
  ``elements/lenses.py`` (the repo's established meta-pin idiom).
"""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel):
    return (_REPO_ROOT / rel).read_text(encoding='utf-8', errors='replace')


# =========================================================================
# mypy strict whitelist
# =========================================================================

def test_validation_in_mypy_whitelist():
    try:
        import tomllib
        with open(_REPO_ROOT / 'pyproject.toml', 'rb') as fh:
            data = tomllib.load(fh)
        files = data['tool']['mypy']['files']
    except Exception:
        # Fallback: coarse text scan of the [tool.mypy] files list.
        files = _read('pyproject.toml')
    joined = files if isinstance(files, str) else '\n'.join(files)
    assert 'lumenairy/_validation.py' in joined, (
        'S4-20: _validation.py missing from the mypy strict whitelist.')


# =========================================================================
# MANIFEST.in ships the reference docs
# =========================================================================

def test_manifest_ships_reference_docs():
    manifest = _read('MANIFEST.in')
    for doc in ('CONVENTIONS.md', 'Migration-Guide.md', 'ROADMAP.md'):
        # The doc must be shipped (explicit include) AND exist on disk.
        assert f'include {doc}' in manifest, (
            f'S4-20: MANIFEST.in does not ship {doc} in the sdist.')
        assert (_REPO_ROOT / doc).exists(), (
            f'{doc} referenced in MANIFEST.in but missing from the repo.')


# =========================================================================
# GUI psf_mtf pupil sign convention
# =========================================================================

def test_lens_screen_uses_conjugate_convention():
    """Establish the library convention: wave-optics lens screens apply
    ``exp(-1j*k0*opd)`` (the marker the GUI pupil must match)."""
    src = _read('lumenairy/elements/lenses.py')
    assert 'exp(-1j*k0*opd)' in src, (
        'Library lens-screen phase convention marker not found; the '
        'S4-20 GUI pin below is anchored to it.')


def test_psf_mtf_pupil_matches_conjugate_convention():
    """The ray-traced GUI pupil must use ``exp(-1j * phase)`` (conjugate,
    matching the lens screens) and NOT the pre-fix ``exp(1j * phase)``."""
    src = _read('lumenairy/ui/psf_mtf_dock.py')
    assert 'np.exp(-1j * phase)' in src, (
        'S4-20: GUI psf_mtf pupil does not use the conjugate '
        'exp(-i k0 OPD) convention.')
    assert 'np.exp(1j * phase)' not in src, (
        'S4-20: GUI psf_mtf pupil still carries the pre-fix '
        'exp(+i k0 OPD) convention (mirror-flipped PSF).')


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
