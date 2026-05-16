"""
Glass refractive index lookup via the refractiveindex.info database.

This module provides a simple interface to query refractive indices of common
optical glasses by name and wavelength.  Three lookup paths are supported:

1. **refractiveindex.info entries** -- pulled live (and cached) via the
   ``refractiveindex`` Python package.  Encoded as a
   ``(shelf, book, page)`` tuple in :data:`GLASS_REGISTRY`.
2. **Bundled Sellmeier coefficients** -- a small library of Schott /
   Ohara dispersion coefficients keyed by name in
   :data:`SELLMEIER_COEFFICIENTS`.  Used as a *fallback* when the
   ``refractiveindex`` package is not installed, or for glasses that
   the database lacks.  No network / file access; pure float math.
3. **User-supplied callables** -- assign ``GLASS_REGISTRY['MY_GLASS']``
   to a function ``f(wavelength_m) -> n_real`` (or returning a complex
   ``n + 1j*kappa``) and it will be used directly.  Useful for custom
   materials, prototype coatings, or temperature-dependent indices.

The central data structure is :data:`GLASS_REGISTRY`.  Each entry can
be one of:

* ``(shelf, book, page)`` tuple of strings -- refractiveindex.info path
* ``callable(wavelength_m) -> float`` -- returns the real index
* ``callable(wavelength_m) -> complex`` -- returns ``n + 1j*kappa``
* the literal string ``'__sellmeier__'`` -- defer to
  :data:`SELLMEIER_COEFFICIENTS`
* the special key ``'__thin_lens__'`` -- sentinel used by the lens
  helpers to flag a thin-lens placeholder; not user-facing

**Adding a new tuple-style glass**::

    from lumenairy.glass import GLASS_REGISTRY
    GLASS_REGISTRY['MY_GLASS'] = ('specs', 'CATALOG', 'PAGE_NAME')

Browse https://refractiveindex.info to find the correct shelf/book/page
path for the material you need.

**Adding a custom dispersion callable**::

    import numpy as np
    from lumenairy.glass import GLASS_REGISTRY

    def my_glass(wavelength_m):
        wl_um = wavelength_m * 1e6
        return 1.5 + 0.01 / wl_um**2  # Cauchy A + B/lam**2

    GLASS_REGISTRY['MY_GLASS'] = my_glass

The callable will be invoked with the wavelength in metres (the
LumenAiry standard).  Return a ``float`` for the real index or a
``complex`` for ``n + 1j*kappa``.

**Querying available glasses**::

    from lumenairy.glass import list_glasses, search_glasses
    list_glasses()              # all names
    search_glasses('SF')        # all SF-type heavy flint glasses

Dependencies
------------
* ``refractiveindex`` (optional) -- ``pip install refractiveindex``.
  When missing, the module falls back to the bundled Sellmeier
  coefficients in :data:`SELLMEIER_COEFFICIENTS`.  Only the tuple-
  style entries that lack a Sellmeier fallback will raise.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util
import math as _math
_REFRACTIVEINDEX_AVAILABLE = (
    _importlib_util.find_spec('refractiveindex') is not None)
RefractiveIndexMaterial = None  # populated lazily on first use


def _ensure_refractiveindex_loaded():
    global RefractiveIndexMaterial
    if RefractiveIndexMaterial is None and _REFRACTIVEINDEX_AVAILABLE:
        from refractiveindex import RefractiveIndexMaterial as _RIM
        globals()['RefractiveIndexMaterial'] = _RIM
    return RefractiveIndexMaterial is not None


# ---------------------------------------------------------------------------
# Bundled Sellmeier coefficients (3-term form, n^2 - 1 = sum B_i*lam^2 /
# (lam^2 - C_i), with lam in MICRONS).  Source: SCHOTT / Ohara catalogues
# at 20 C, 1 atm.
# ---------------------------------------------------------------------------
# Format: SELLMEIER_COEFFICIENTS[name] = ((B1, B2, B3), (C1, C2, C3))
# where C_i is in um^2.

SELLMEIER_COEFFICIENTS = {
    # Schott N-series ---------------------------------------------------
    'N-BK7':      ((1.03961212, 0.231792344, 1.01046945),
                   (6.00069867e-3, 2.00179144e-2, 1.03560653e2)),
    'N-K5':       ((1.08511833, 0.199562005, 0.930511663),
                   (6.61099503e-3, 2.41108660e-2, 1.11982777e2)),
    'N-BAF10':    ((1.5851495, 0.143559385, 1.08521269),
                   (9.26681282e-3, 4.24489805e-2, 1.05613573e2)),
    'N-BAK4':     ((1.28834642, 0.132817724, 0.945395373),
                   (7.79980626e-3, 3.15631177e-2, 1.05965875e2)),
    'N-BAF52':    ((1.43903433, 0.179827671, 1.13174268),
                   (9.07800726e-3, 4.39222348e-2, 1.06317650e2)),
    'N-FK51A':    ((0.971247817, 0.216901417, 0.904651666),
                   (4.72301995e-3, 1.53575612e-2, 1.68681330e2)),
    'N-PSK53A':   ((1.38121836, 0.196745645, 0.886089205),
                   (7.06416337e-3, 2.33251345e-2, 9.74847345e1)),
    'N-LAK22':    ((1.14229781, 0.535138441, 1.04088385),
                   (5.85778594e-3, 1.98546147e-2, 1.00834017e2)),
    'N-LAK33A':   ((1.44116999, 0.571749501, 1.16605226),
                   (6.80933877e-3, 2.22291824e-2, 1.07097324e2)),
    'N-LAK33B':   ((1.42288601, 0.593661336, 1.16135260),
                   (6.70283452e-3, 2.19416210e-2, 1.01736644e2)),
    'N-SK11':     ((1.17963631, 0.229817295, 0.935789652),
                   (6.80282081e-3, 2.19737205e-2, 1.01513232e2)),
    'N-SK16':     ((1.34317774, 0.241144399, 0.994317969),
                   (7.04687339e-3, 2.29005000e-2, 9.27508526e1)),
    'N-SSK2':     ((1.43060270, 0.153150554, 1.01390904),
                   (8.23982975e-3, 3.33736841e-2, 1.06870822e2)),
    'N-SSK8':     ((1.44857867, 0.117965926, 1.06937528),
                   (8.69310149e-3, 4.21566593e-2, 1.11300666e2)),
    'N-F2':       ((1.39757037, 0.159201403, 1.26865430),
                   (9.95906143e-3, 5.46931752e-2, 1.19248346e2)),
    'N-SF2':      ((1.47343127, 0.163681849, 1.36920899),
                   (1.09019098e-2, 5.85683687e-2, 1.27404933e2)),
    'N-SF5':      ((1.52481889, 0.187085527, 1.42729015),
                   (1.12547560e-2, 5.88995392e-2, 1.29141675e2)),
    'N-SF6':      ((1.77931763, 0.338149866, 2.08734474),
                   (1.33714182e-2, 6.17533621e-2, 1.74017590e2)),
    'N-SF6HT':    ((1.77931763, 0.338149866, 2.08734474),
                   (1.33714182e-2, 6.17533621e-2, 1.74017590e2)),
    'N-SF10':     ((1.62153902, 0.256287842, 1.64447552),
                   (1.22241457e-2, 5.95736775e-2, 1.47468793e2)),
    'N-SF11':     ((1.73759695, 0.313747346, 1.89878101),
                   (1.31887070e-2, 6.23068142e-2, 1.55236290e2)),
    'N-SF14':     ((1.69022361, 0.288870052, 1.70451000),
                   (1.30512113e-2, 6.13691880e-2, 1.49517689e2)),
    'N-SF15':     ((1.57055634, 0.218987094, 1.50824017),
                   (1.16507014e-2, 5.97856897e-2, 1.32709339e2)),
    'N-SF57':     ((1.87543831, 0.37375749, 2.30001797),
                   (1.41749518e-2, 6.40509927e-2, 1.77389795e2)),
    'N-LASF31A':  ((1.96485075, 0.475231259, 1.48360109),
                   (9.82060155e-3, 3.44713438e-2, 1.10739863e2)),
    'N-LASF40':   ((1.98550331, 0.274057042, 1.28945661),
                   (1.09583310e-2, 4.74551603e-2, 9.65887504e1)),
    'N-LASF41':   ((1.86348331, 0.413307255, 1.35784815),
                   (9.10368219e-3, 3.39247268e-2, 9.33580610e1)),
    'N-LASF44':   ((1.78897105, 0.386758670, 1.30506243),
                   (8.72506277e-3, 3.08085023e-2, 9.27743824e1)),
    'N-LASF45':   ((1.87140198, 0.267777879, 1.73030008),
                   (1.12189043e-2, 5.05133972e-2, 1.47106505e2)),
    'N-LASF46A':  ((2.16701566, 0.319812761, 1.66004486),
                   (1.23595524e-2, 5.60610282e-2, 1.07047718e2)),
    'N-LASF46B':  ((2.17988922, 0.306495184, 1.56882437),
                   (1.25805384e-2, 5.67191367e-2, 1.05316538e2)),
    'N-LASF9':    ((2.00029547, 0.298926886, 1.80691843),
                   (1.21426017e-2, 5.38736236e-2, 1.56530829e2)),
    'F2':         ((1.34533359, 0.209073176, 0.937357162),
                   (9.97743871e-3, 4.70450767e-2, 1.11886764e2)),
    'F5':         ((1.31044630, 0.196034260, 0.966129770),
                   (9.58633486e-3, 4.57627627e-2, 1.15011883e2)),
    'SF2':        ((1.40301821, 0.231767504, 0.939056586),
                   (1.05795466e-2, 4.93226978e-2, 1.12405955e2)),
    # Ohara S-LAH series ------------------------------------------------
    'S-LAH64':    ((1.97644957, 0.345835202, 1.50865430),
                   (9.06133603e-3, 3.31538835e-2, 9.96362760e1)),
    'S-LAH79':    ((2.13713459, 0.302257453, 1.55230336),
                   (1.07957653e-2, 4.94336323e-2, 1.21532852e2)),
    # Common bulk materials ---------------------------------------------
    'CaF2':       ((0.5675888, 0.4710914, 3.8484723),
                   (2.526430e-3, 1.007833e-2, 1.200556e3)),
    'MgF2':       ((0.48755108, 0.39875031, 2.3120353),
                   (1.882178e-3, 8.951888e-3, 5.661406e2)),
    'BaF2':       ((0.6435, 0.5067, 3.8261),
                   (1.5e-3, 9.5e-3, 2.5e3)),
}


def _sellmeier_index(wavelength_m, coeffs, glass_name=None):
    """Three-term Sellmeier evaluator.

    ``coeffs`` is ``((B1, B2, B3), (C1, C2, C3))`` with ``C_i`` in
    um^2.  Returns the real refractive index at the given vacuum
    wavelength [m].

    4.10: validates that the wavelength does not coincide with a
    Sellmeier resonance (``lam² ≈ C_i``) and that the radicand stays
    positive.  Pre-4.10 a wavelength near a resonance raised an opaque
    ``math domain error``; this version raises ``ValueError`` with the
    glass name and the offending wavelength.
    """
    lam2 = (wavelength_m * 1e6) ** 2  # wavelength^2 in um^2
    (B1, B2, B3), (C1, C2, C3) = coeffs
    label = f" for glass {glass_name!r}" if glass_name else ""
    for ci in (C1, C2, C3):
        if abs(lam2 - ci) < 1e-12:
            raise ValueError(
                f"_sellmeier_index{label}: wavelength {wavelength_m*1e9:.3f} nm "
                f"coincides with a Sellmeier resonance (lam² ≈ C_i = {ci:.6f} "
                f"um²).  Use a wavelength away from the resonance, or "
                f"select a different glass model that covers this range.")
    n_sq_minus_1 = (
        B1 * lam2 / (lam2 - C1)
        + B2 * lam2 / (lam2 - C2)
        + B3 * lam2 / (lam2 - C3)
    )
    if n_sq_minus_1 <= -1.0:
        raise ValueError(
            f"_sellmeier_index{label}: extrapolation produced negative n² "
            f"(n²-1 = {n_sq_minus_1:.6f}) at wavelength {wavelength_m*1e9:.3f} nm. "
            f"This wavelength is likely outside the catalogue's valid "
            f"range; pass a wavelength within the glass's specified band.")
    return _math.sqrt(1.0 + n_sq_minus_1)


# ---------------------------------------------------------------------------
# Glass registry -- entries can be tuples, callables, the
# '__sellmeier__' sentinel, or the '__thin_lens__' marker.
# ---------------------------------------------------------------------------

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

    # ----- 4.7 additions: Schott / Ohara entries served from the bundled
    #       Sellmeier table (no refractiveindex network needed) -----------
    'N-K5':         '__sellmeier__',
    'N-BAK4':       '__sellmeier__',
    'N-BAF52':      '__sellmeier__',
    'N-FK51A':      '__sellmeier__',
    'N-PSK53A':     '__sellmeier__',
    'N-LAK33A':     '__sellmeier__',
    'N-LAK33B':     '__sellmeier__',
    'N-SK11':       '__sellmeier__',
    'N-SK16':       '__sellmeier__',
    'N-SSK2':       '__sellmeier__',
    'N-F2':         '__sellmeier__',
    'N-SF5':        '__sellmeier__',
    'N-SF10':       '__sellmeier__',
    'N-SF11':       '__sellmeier__',
    'N-SF14':       '__sellmeier__',
    'N-SF15':       '__sellmeier__',
    'N-SF57':       '__sellmeier__',
    'N-LASF31A':    '__sellmeier__',
    'N-LASF40':     '__sellmeier__',
    'N-LASF41':     '__sellmeier__',
    'N-LASF44':     '__sellmeier__',
    'N-LASF45':     '__sellmeier__',
    'N-LASF46A':    '__sellmeier__',
    'N-LASF46B':    '__sellmeier__',
    'F2':           '__sellmeier__',
    'F5':           '__sellmeier__',
    'SF2':          '__sellmeier__',
    'S-LAH64':      '__sellmeier__',
    'S-LAH79':      '__sellmeier__',
    'BaF2':         '__sellmeier__',
}


def list_glasses() -> List[str]:
    """Return the sorted list of glass names known to the registry.

    Useful for IDE auto-complete-style discovery and for printing a
    suggestion list on a typo.

    Returns
    -------
    list of str
        Glass names (excludes the internal ``'__thin_lens__'`` marker).
    """
    return sorted(
        name for name in GLASS_REGISTRY
        if not name.startswith('__')
    )


def search_glasses(pattern: str) -> List[str]:
    """Return registry entries whose name contains ``pattern`` (case-
    insensitive).

    Useful for narrowing the catalogue: ``search_glasses('LASF')``
    returns every lanthanum-flint glass in the registry.
    """
    pat = str(pattern).upper()
    return sorted(
        name for name in GLASS_REGISTRY
        if not name.startswith('__') and pat in name.upper()
    )

# ---------------------------------------------------------------------------
# Cache -- avoids re-loading YAML dispersion files on every call
# ---------------------------------------------------------------------------
_glass_cache = {}


def get_glass_index(glass_name: str, wavelength: float) -> float:
    """
    Look up refractive index by common glass name at a given wavelength.

    Resolution order:

    1. ``glass_name`` is ``'air'`` (case-insensitive) -- returns 1.0.
    2. ``GLASS_REGISTRY[glass_name]`` is a callable -- calls it with
       ``wavelength`` (in metres) and returns the real part.
    3. The registry entry is the sentinel ``'__sellmeier__'`` -- looks
       up the coefficients in :data:`SELLMEIER_COEFFICIENTS` and
       evaluates the 3-term Sellmeier formula.
    4. The registry entry is a ``(shelf, book, page)`` tuple --
       dispatches to the ``refractiveindex`` package.  Raises
       :exc:`ImportError` if the package is not installed.

    Parameters
    ----------
    glass_name : str
        Glass name from :data:`GLASS_REGISTRY` (e.g. ``'N-BK7'``,
        ``'N-SF6HT'``, ``'CaF2'``), or ``'air'`` for n=1.0.
    wavelength : float
        Free-space wavelength [m].

    Returns
    -------
    n : float
        Real refractive index at the given wavelength.

    Raises
    ------
    ValueError
        If ``glass_name`` is not in the registry.
    ImportError
        If the entry is a refractiveindex.info tuple but the
        ``refractiveindex`` package is not installed.

    Examples
    --------
    >>> get_glass_index('N-BK7', 587.6e-9)  # d-line
    1.5168...

    >>> from lumenairy.glass import GLASS_REGISTRY
    >>> GLASS_REGISTRY['CONSTANT_1p5'] = lambda wl: 1.5
    >>> get_glass_index('CONSTANT_1p5', 1.31e-6)
    1.5
    """
    if glass_name.lower() == 'air':
        return 1.0

    if glass_name not in GLASS_REGISTRY:
        # Find close matches to suggest in the error.  Try a substring
        # search first (good for 'find me anything with LASF'), then
        # fall back to difflib for closest-spelling matches when the
        # substring lookup returns nothing (good for typos).
        suggestions = search_glasses(glass_name)
        if not suggestions:
            import difflib as _difflib
            suggestions = _difflib.get_close_matches(
                glass_name, list_glasses(), n=5, cutoff=0.4)
        suggest_msg = ''
        if suggestions:
            suggest_msg = f' Did you mean: {suggestions[:5]}?'
        raise ValueError(
            f"Glass {glass_name!r} not in registry.{suggest_msg} "
            f"See lumenairy.glass.list_glasses() for the full list, "
            f"or add the glass by assigning GLASS_REGISTRY[{glass_name!r}] "
            f"to a (shelf, book, page) tuple, a callable f(wavelength_m), "
            f"or the '__sellmeier__' sentinel with coefficients in "
            f"SELLMEIER_COEFFICIENTS.")

    entry = GLASS_REGISTRY[glass_name]

    # User-supplied dispersion callable: f(wavelength_m) -> n_real or
    # n_complex.  We strip any imaginary part for the real-only API.
    if callable(entry):
        n = entry(wavelength)
        if isinstance(n, complex):
            return float(n.real)
        return float(n)

    # Bundled Sellmeier coefficients (no external dependency).
    if entry == '__sellmeier__':
        if glass_name not in SELLMEIER_COEFFICIENTS:
            raise ValueError(
                f"Glass {glass_name!r} is flagged '__sellmeier__' in "
                f"GLASS_REGISTRY but has no entry in "
                f"SELLMEIER_COEFFICIENTS.")
        return _sellmeier_index(wavelength,
                                SELLMEIER_COEFFICIENTS[glass_name])

    # Internal thin-lens placeholder marker (used by the thin-lens
    # helpers); should never reach a propagation path, but if a user
    # somehow calls it, fall back to n=1.5 so the call doesn't crash.
    if entry == '__thin_lens__':
        return 1.5

    # Tuple-style entry: dispatch to refractiveindex.info.
    if not _REFRACTIVEINDEX_AVAILABLE:
        # If we have Sellmeier coefficients bundled for the same name,
        # use them rather than raising.
        if glass_name in SELLMEIER_COEFFICIENTS:
            return _sellmeier_index(wavelength,
                                    SELLMEIER_COEFFICIENTS[glass_name])
        raise ImportError(
            f"Glass {glass_name!r} requires the 'refractiveindex' "
            f"package for live lookup, but it is not installed.  "
            f"Either install it (`pip install refractiveindex`) or "
            f"register a callable / Sellmeier coefficients for this "
            f"glass in GLASS_REGISTRY / SELLMEIER_COEFFICIENTS.")
    _ensure_refractiveindex_loaded()

    if glass_name not in _glass_cache:
        shelf, book, page = entry
        _glass_cache[glass_name] = RefractiveIndexMaterial(
            shelf=shelf, book=book, page=page)

    return _glass_cache[glass_name].get_refractive_index(
        wavelength * 1e9, unit='nm')


def get_glass_index_complex(glass_name: str,
                            wavelength: float) -> complex:
    """
    Look up complex refractive index ``n + i*kappa`` by glass name.

    Same resolution order as :func:`get_glass_index`, but if the
    registry entry is a *callable* and returns a ``complex`` value,
    its imaginary part is preserved as ``kappa``.  For all other
    paths (Sellmeier, refractiveindex.info, ``'air'``) the extinction
    is looked up via the database when available and falls back to
    ``kappa = 0`` otherwise.

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

    if glass_name not in GLASS_REGISTRY:
        # Use the real-side error so the user sees the same suggestion
        # list -- get_glass_index re-raises ValueError.
        get_glass_index(glass_name, wavelength)

    entry = GLASS_REGISTRY[glass_name]

    # Callable: respect a complex return so users can model absorbing
    # / complex-index materials with one function.
    if callable(entry):
        n = entry(wavelength)
        if isinstance(n, complex):
            return n
        return complex(float(n), 0.0)

    # Sellmeier / sentinel paths have no extinction data; return
    # kappa = 0 explicitly.
    if entry in ('__sellmeier__', '__thin_lens__'):
        return complex(get_glass_index(glass_name, wavelength), 0.0)

    # Tuple-style path: try refractiveindex.info for both n and kappa.
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
