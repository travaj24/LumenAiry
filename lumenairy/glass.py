"""
Glass refractive index lookup via the refractiveindex.info database.

This module provides a simple interface to query refractive indices of common
optical glasses by name and wavelength.  It wraps the ``refractiveindex``
Python package, which itself pulls dispersion data from the open
refractiveindex.info database.

The central data structure is :data:`GLASS_REGISTRY`, a dictionary that maps
human-readable glass names (e.g. ``'N-BK7'``, ``'CaF2'``, ``'F_SILICA'``) to
``(shelf, book, page)`` tuples understood by
:class:`refractiveindex.RefractiveIndexMaterial`.

**Extending the registry** -- users can add new glasses at any time::

    from lumenairy.glass import GLASS_REGISTRY
    GLASS_REGISTRY['MY_GLASS'] = ('specs', 'CATALOG', 'PAGE_NAME')

Browse https://refractiveindex.info to find the correct shelf/book/page path
for the material you need.

Dependencies
------------
* ``refractiveindex`` -- install with ``pip install refractiveindex``.
  The module degrades gracefully: import succeeds even if the package is
  missing, but :func:`get_glass_index` will raise :exc:`ImportError` at
  call time.

Author: Andrew Traverso
"""

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
try:
    from refractiveindex import RefractiveIndexMaterial
    _REFRACTIVEINDEX_AVAILABLE = True
except ImportError:
    _REFRACTIVEINDEX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Glass registry -- (shelf, book, page) tuples for refractiveindex.info
# ---------------------------------------------------------------------------
# Easy to extend: just add entries following the pattern below.

GLASS_REGISTRY = {
    # ----- Schott glasses (specs shelf -- manufacturer data) ---------------
    'N-BK7':        ('specs', 'SCHOTT-optical', 'N-BK7'),
    'N-SF6':        ('specs', 'SCHOTT-optical', 'N-SF6'),
    'N-SF6HT':      ('specs', 'SCHOTT-optical', 'N-SF6HT'),
    'N-BAF10':      ('specs', 'SCHOTT-optical', 'N-BAF10'),
    'N-LAK22':      ('specs', 'SCHOTT-optical', 'N-LAK22'),
    'N-SF2':        ('specs', 'SCHOTT-optical', 'N-SF2'),
    'N-SSK8':       ('specs', 'SCHOTT-optical', 'N-SSK8'),
    'N-LASF9':      ('specs', 'SCHOTT-optical', 'N-LASF9'),

    # ----- Generic materials (main shelf -- literature data) ---------------
    'CaF2':         ('main', 'CaF2', 'Daimon-20'),
    'SiO2':         ('main', 'SiO2', 'Malitson'),
    'MgF2':         ('main', 'MgF2', 'Dodge-o'),

    # ----- Zemax MISC catalog aliases --------------------------------------
    'F_SILICA':     ('main', 'SiO2', 'Malitson'),
    'FUSED_SILICA': ('main', 'SiO2', 'Malitson'),
    'SILICA':       ('main', 'SiO2', 'Malitson'),
    'SILICON':      ('main', 'Si', 'Li-293K'),
}

# ---------------------------------------------------------------------------
# Cache -- avoids re-loading YAML dispersion files on every call
# ---------------------------------------------------------------------------
_glass_cache = {}


def get_glass_index(glass_name, wavelength):
    """
    Look up refractive index by common glass name at a given wavelength.

    Uses the ``refractiveindex`` package (wraps refractiveindex.info).

    Parameters
    ----------
    glass_name : str
        Glass name from GLASS_REGISTRY (e.g. 'N-BK7', 'N-SF6HT', 'CaF2'),
        or 'air' for n=1.0.
    wavelength : float
        Free-space wavelength [m].

    Returns
    -------
    n : float
        Refractive index at the given wavelength.
    """
    if glass_name.lower() == 'air':
        return 1.0

    if not _REFRACTIVEINDEX_AVAILABLE:
        raise ImportError(
            "The 'refractiveindex' package is required for glass index lookup. "
            "Install it with: pip install refractiveindex")

    if glass_name not in GLASS_REGISTRY:
        raise ValueError(
            f"Glass '{glass_name}' not in registry. "
            f"Available: {sorted(GLASS_REGISTRY.keys())}. "
            f"Add it to GLASS_REGISTRY with the correct (shelf, book, page) tuple "
            f"from refractiveindex.info.")

    if glass_name not in _glass_cache:
        shelf, book, page = GLASS_REGISTRY[glass_name]
        _glass_cache[glass_name] = RefractiveIndexMaterial(
            shelf=shelf, book=book, page=page)

    return _glass_cache[glass_name].get_refractive_index(
        wavelength * 1e9, unit='nm')


def get_glass_index_complex(glass_name, wavelength):
    """
    Look up complex refractive index ``n + i*kappa`` by glass name.

    Identical to :func:`get_glass_index` for the real part, but additionally
    queries the extinction coefficient ``kappa`` from the underlying
    refractiveindex.info entry.  Materials that have no tabulated extinction
    silently return ``kappa = 0`` (so the complex value collapses to a real
    one and downstream code is unaffected).

    Parameters
    ----------
    glass_name : str
        Glass name (see :data:`GLASS_REGISTRY`) or ``'air'``.
    wavelength : float
        Free-space wavelength [m].

    Returns
    -------
    n_complex : complex
        ``n + 1j*kappa`` at the given wavelength.  ``kappa > 0`` indicates
        absorption.  Use the imaginary part to compute bulk attenuation as
        ``exp(-2*pi * kappa * thickness / wavelength)``.
    """
    if glass_name.lower() == 'air':
        return 1.0 + 0.0j

    n_real = get_glass_index(glass_name, wavelength)

    # Extinction is optional -- many catalog entries (esp. SCHOTT specs) omit it.
    try:
        kappa = _glass_cache[glass_name].get_extinction_coefficient(
            wavelength * 1e9, unit='nm')
        if kappa is None:
            kappa = 0.0
    except (AttributeError, NotImplementedError, KeyError, ValueError, TypeError):
        kappa = 0.0

    return complex(n_real, float(kappa))
