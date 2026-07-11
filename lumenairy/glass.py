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

# ---------------------------------------------------------------------------
# Optional dependency
# ---------------------------------------------------------------------------
import importlib.util as _importlib_util
import math as _math
import threading as _threading

# NOTE: imported UN-aliased on purpose -- the v4.16.1 cache-enrollment
# meta-pin's AST walker recognises a cache declaration by the literal
# ``OrderedDict(...)`` / ``dict(...)`` / ``{}`` RHS shape; an aliased
# ``_OrderedDict()`` would hide ``_glass_value_cache`` from discovery
# (the exact blindness audit P2-42 just closed for lower-case names).
from collections import OrderedDict
from typing import List

# v4.16.2 (audit P3-NEW-F1-4): import numpy at module scope to support
# numpy-scalar acceptance in GLASS_VALIDITY well-formedness check.
import numpy as np

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
    # 4.11.2: the previously hard-coded Sellmeier coefficients for these
    # two glasses produced n_d = 1.8458 (S-LAH64) and 1.8853 (S-LAH79)
    # vs Ohara catalog n_d = 1.78800 and 2.00330 respectively -- off by
    # 0.058 and 0.118.  The in-code coefficients appear to be misattri-
    # buted from a different glass.  Removed from the in-code Sellmeier
    # table and routed through the authoritative refractiveindex.info
    # lookup via the '__sellmeier__' sentinel.  Requires ``pip install
    # refractiveindex`` -- without it, a glass lookup for these names
    # will fail with a clear error rather than silently returning a
    # ~3% wrong index.  Caught by AUDIT_ROUND3_2026_05_16.md (CRIT-1).
    #
    # v4.15 (P1-GL-1): re-bundle these as a fallback path.  v4.11.2 left
    # tuple-registered S-LAH64 / S-LAH79 with NO Sellmeier coefficients,
    # so a minimal install (without the ``refractiveindex`` Python pkg)
    # hit the dispatcher's ``ImportError`` branch on every lookup.
    # Bundling the OHARA Sellmeier table (sourced from the
    # refractiveindex.info-database YAMLs, OHARA Zemax 2017-11-30
    # catalog) restores the in-process fallback while still matching
    # ``refractiveindex`` to within 1e-9 (S-LAH64) / 4e-7 (S-LAH79) at
    # n_d.  Verified to ~5e-5 across 488 / 532 / 633 / 1064 / 1310 /
    # 1550 nm; rms residual relative to the refractiveindex.info
    # tabulated values stays below 1e-5 across the catalogued
    # 0.32-2.4 um (LAH64) / 0.37-2.4 um (LAH79) bands.
    'S-LAH64':    ((1.83021453, 0.29156359, 1.28544024),
                   (9.0482329e-3, 3.30756689e-2, 8.93675501e1)),
    'S-LAH79':    ((2.32557148, 0.507967133, 2.43087198),
                   (1.32895208e-2, 5.28335449e-2, 1.61122408e2)),
    # Common bulk materials ---------------------------------------------
    'CaF2':       ((0.5675888, 0.4710914, 3.8484723),
                   (2.526430e-3, 1.007833e-2, 1.200556e3)),
    'MgF2':       ((0.48755108, 0.39875031, 2.3120353),
                   (1.882178e-3, 8.951888e-3, 5.661406e2)),
    # v5.4.6 (audit F-33): replace the low-precision / mis-poled BaF2 row
    # (B=(0.6435,0.5067,3.8261), C=(1.5e-3,9.5e-3,2.5e3)) -- ~0.4-0.5%
    # index error in the visible -- with the authoritative Li 1980 fit
    # (refractiveindex.info main/BaF2/Li).  C_i are the resonance
    # wavelengths squared in um^2.
    'BaF2':       ((0.643356, 0.506762, 3.8261),
                   (0.057789**2, 0.10968**2, 46.3864**2)),
    # v4.15 (P1-GL-1): bundled Sellmeier fallback for tuple-registered
    # fused-silica entries.  Pre-4.15 these were registered as
    # (main, SiO2, Malitson) tuples in GLASS_REGISTRY but absent from
    # SELLMEIER_COEFFICIENTS, so a minimal install raised ImportError
    # on every silica lookup.  Malitson 1965 (J. Opt. Soc. Am.
    # 55, 1205-1209) is the well-known well-cited 3-term Sellmeier for
    # synthetic fused silica, valid 0.21 - 6.7 um.  Coefficients are
    # quoted with C_i in um^2 (i.e. (lambda_i)^2 of the resonance pole
    # locations 0.0684043, 0.1162414, 9.896161 um); the squaring is
    # baked into the table to keep the format consistent with the rest
    # of SELLMEIER_COEFFICIENTS.  Matches refractiveindex.info to
    # ~4e-6 at the d-line; below 5e-5 across the 488-1550 nm band.
    'SiO2':         ((0.6961663, 0.4079426, 0.8974794),
                     (0.0684043**2, 0.1162414**2, 9.896161**2)),
    'F_SILICA':     ((0.6961663, 0.4079426, 0.8974794),
                     (0.0684043**2, 0.1162414**2, 9.896161**2)),
    'FUSED_SILICA': ((0.6961663, 0.4079426, 0.8974794),
                     (0.0684043**2, 0.1162414**2, 9.896161**2)),
    # GL-nit (AUDIT_GLASS_POLARIZATION): 'SILICA' is registry-aliased to
    # Malitson fused silica (like SiO2 / F_SILICA / FUSED_SILICA) but was
    # missing its bundled Sellmeier row, so on a minimal install it raised
    # ImportError while the three siblings resolved.  Copy the row.
    'SILICA':       ((0.6961663, 0.4079426, 0.8974794),
                     (0.0684043**2, 0.1162414**2, 9.896161**2)),

    # ============================================================
    # v4.16.0 (ROADMAP #13) -- CDGM Sellmeier bundled fallback
    # ============================================================
    #
    # CDGM (Tianjin Lingxin Glass) catalogue entries that use the
    # SCHOTT-style Sellmeier-2 dispersion formula (refractiveindex.info
    # "formula 2"):
    #
    #     n² - 1 = sum_i B_i * lam² / (lam² - C_i)
    #
    # where C_i is in um² (i.e. already the squared resonance
    # location, NOT the bare wavelength).  Coefficients sourced from
    # the CDGM Zemax 2022-06 catalogue YAMLs distributed with the
    # refractiveindex.info database (CC0 1.0 public domain).
    # Source: https://refractiveindex.info/?shelf=specs&book=CDGM-optical
    #
    # CDGM glasses using formula 3 (polynomial) are NOT bundled here
    # -- they are registered as tuple-style entries in GLASS_REGISTRY
    # and require the ``refractiveindex`` package for lookup.  Each
    # bundled entry's n_d is cross-checked to 5e-5 in
    # ``tests/unit/test_v4_16_0_agent_d_glass_catalogues.py``.

    # H-K9L -- BK7-equivalent crown (formula 2 in CDGM 2022 catalog)
    # n_d = 1.516800, V_d = 64.20, range 0.302-2.325 um
    'H-K9L':       ((0.614555251, 0.656775017, 1.02699346),
                    (0.0145987884, 0.00287769588, 107.653051)),
    # H-LAK52 -- lanthanum crown (formula 2 in CDGM 2022 catalog)
    # n_d = 1.729160, V_d = 54.68, range 0.302-2.325 um
    'H-LAK52':     ((0.648528792, 1.28450424, 1.23774593),
                    (0.000165816179, 0.0158113883, 85.7406109)),
    # H-LAK53A -- lanthanum crown (formula 2 in CDGM 2022 catalog)
    # n_d = 1.755000, V_d = 52.32, range 0.302-2.325 um
    'H-LAK53A':    ((1.21007394, 1.19495815, 0.806533246),
                    (0.00521679919, 83.5557501, 0.02008035)),
    # H-ZK9B -- zinc crown (formula 2 in CDGM 2022 catalog)
    # n_d = 1.62041 (computed; v4.16.1 audit AUDIT_V4_16_0_DEEP Part 7
    #   4-way convergent: stale inline comment claimed 1.613750.
    #   Coefficients are bit-for-bit correct against refractiveindex.info
    #   CDGM 2022-06 YAML; comment was the lie).
    # V_d = 50.40, range 0.302-2.325 um
    'H-ZK9B':      ((0.874918734, 0.7096273, 1.09453882),
                    (0.00373965378, 0.0163972089, 100.204324)),
    # H-ZF12 -- zinc heavy flint (formula 2 in CDGM 2022 catalog)
    # n_d = 1.76182 (computed; v4.16.1 audit AUDIT_V4_16_0_DEEP Part 7
    #   4-way convergent: stale inline comment claimed 1.673 -- 6% off,
    #   the most egregious drift in the block.  Coefficients bit-for-bit
    #   correct against refractiveindex.info CDGM 2022-06 YAML).
    # V_d = 32.16, range 0.4047-2.325 um
    'H-ZF12':      ((0.323730379, 1.65668629, 2.1214152),
                    (0.0598308169, 0.0120542686, 174.18792)),
    # D-ZK3 -- zinc crown (formula 2 in CDGM 2022 catalog)
    # n_d = 1.589130, V_d = 61.18, range 0.302-2.325 um
    'D-ZK3':       ((1.33936811, 0.148690268, 1.00954033),
                    (0.00760612309, 0.0238444644, 89.0419787)),
    # D-LAK52 -- lanthanum crown (formula 2 in CDGM 2022 catalog)
    # n_d = 1.73050 (computed; v4.16.1 audit AUDIT_V4_16_0_DEEP Part 7
    #   4-way convergent: stale inline comment claimed 1.729160.
    #   Coefficients bit-for-bit correct against refractiveindex.info
    #   CDGM 2022-06 YAML).
    # V_d = 54.84, range 0.302-2.325 um.
    # Note: dispatch sheet requested 'D-ZLAK52' which does not exist
    # in the CDGM catalogue; D-LAK52 is the corresponding family
    # member.
    'D-LAK52':     ((1.04647517, 0.889910862, 1.23173612),
                    (0.0173536556, 0.00298398431, 87.1100488)),
    # H-ZLAF52A -- zinc lanthanum flint (formula 2 in CDGM 2022)
    # n_d = 1.80610 (computed; v4.16.1 audit AUDIT_V4_16_0_DEEP Part 7
    #   4-way convergent: stale inline comment claimed 1.796800.
    #   Coefficients bit-for-bit correct against refractiveindex.info
    #   CDGM 2022-06 YAML).
    # V_d = 45.39, range 0.365-2.325 um
    'H-ZLAF52A':   ((1.85313688, 0.317938591, 1.16396318),
                    (0.00985534639, 0.0392611678, 93.1032763)),
}


# ---------------------------------------------------------------------------
# v4.16.2 (pre-v5.0 prep) -- formula-3 (polynomial) bundled evaluator
# ---------------------------------------------------------------------------
#
# refractiveindex.info formula-3 (the "polynomial" dispersion form):
#
#     n^2 = c[0] + sum_i c[2i+1] * (lam_um) ** c[2i+2]
#
# i.e. a constant plus an even number of (coefficient, exponent) pairs.
# Common exponent sets in the catalogues: {-2, -4, 2, 4, 6}.
#
# v4.16.0 introduced the catalogue entries (Hikari E-/J-, Sumita K-, 4
# CDGM polynomial glasses) but routed them exclusively through the
# optional ``refractiveindex`` package.  v4.16.2 lands the bundled
# evaluator infrastructure so minimal installs no longer fail-fast on
# these 26 entries; per-glass coefficient ingestion is staged for
# v5.0 (one-shot import from the refractiveindex.info YAML dataset
# requires a non-trivial vendor-source review for each catalogue).
#
# POLYNOMIAL_COEFFICIENTS layout: ``{name: (c0, [(coeff, exponent), ...])}``
#
# Adding a new entry::
#
#     POLYNOMIAL_COEFFICIENTS['CDGM-H-ZK7'] = (
#         2.6273,
#         [(-0.011, -2.0),
#          (0.026, 2.0),
#          (-0.00071, 4.0),
#          (0.0000084, 6.0)],
#     )
#
# Cross-check the result against refractiveindex.info's tabulated n_d
# to 5e-5 (matches the v4.14.2 / v4.16.0 cross-check methodology).

POLYNOMIAL_COEFFICIENTS = {
    # v5.2.3 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
    # 24 formula-3 catalogue entries (4 CDGM + 10 Hikari + 10 Sumita)
    # ingested verbatim from the refractiveindex.info YAML dataset.
    # Each tuple is ``(c0, [(c_i, exponent_i), ...])`` matching the
    # ``_polynomial_index`` evaluator contract above.
    #
    # Coefficient values are bit-for-bit copies of the
    # ``coefficients: <c0> <c1> <p1> <c2> <p2> ...`` field of the
    # corresponding YAML's ``type: formula 3`` entry; wavelength
    # validity ranges are tracked separately in GLASS_VALIDITY.
    # Sources:
    #   CDGM    -> CDGM Zemax 2022-06 catalogue YAMLs
    #   Hikari  -> NIKON Zemax 2017-11 catalogue YAMLs (HIKARI shelf)
    #   Sumita  -> SUMITA Zemax 2017-02-02 catalogue YAMLs
    #
    # When ingesting a row, add it here with the documented
    # ``{name: (c0, [(c_i, exp_i), ...])}`` layout AND remove the
    # name from _POLYNOMIAL_STUB_NAMES in the same commit; the v5.2
    # consistency check below enforces the disjoint-state invariant.
    #
    # ----- CDGM (China Daheng Group / Tianjin Lingxin) -----
    # 4 polynomial entries from CDGM 2022-06.  The 8 CDGM Sellmeier-2
    # entries are bundled separately above in SELLMEIER_COEFFICIENTS.
    'F1-CDGM':     (2.5059651,
                    [(-0.0078686119, 2), (0.021574214, -2),
                     (0.00059867966, -4), (-9.5194606e-07, -6),
                     (2.9024858e-06, -8)]),
    'F2-CDGM':     (2.53265982,
                    [(-0.00773825354, 2), (0.0231731432, -2),
                     (0.000409362855, -4), (3.78046829e-05, -6),
                     (1.03464346e-06, -8)]),
    'H-ZF52A':     (3.2506862,
                    [(-0.0133942217, 2), (0.0483625913, -2),
                     (0.00314442751, -4), (-0.000197073674, -6),
                     (3.47645679e-05, -8)]),
    'H-ZK7':       (2.56229558,
                    [(-0.0108844706, 2), (0.0140997791, -2),
                     (0.000379257039, -4), (-2.41612681e-05, -6),
                     (1.41616214e-06, -8)]),
    # ----- Hikari Glass Co. Ltd (NIKON Zemax 2017-11) -----
    # 10 polynomial entries.  E-* glasses use 5 coefficient pairs over
    # 0.4-0.7 um; some J-* entries (J-FK01A, J-LASF09A, J-LAK7,
    # J-BASF7) carry 7 pairs over a wider 0.365-2.058 um span.  The
    # GLASS_VALIDITY entries below pin the documented dispersion fit
    # band; the polynomial evaluator extrapolates outside that band
    # at the caller's own risk.
    'E-BAK1':      (2.44353274,
                    [(-0.0161455184, 2), (0.00881927898, -2),
                     (0.00144687542, -4), (-0.000136331798, -6),
                     (6.03309808e-06, -8)]),
    'E-F2':        (2.55739282,
                    [(-0.0107951229, 2), (0.0216807328, -2),
                     (0.00105165979, -4), (-5.36309816e-05, -6),
                     (7.7466326e-06, -8)]),
    'E-LAK04':     (2.6768086,
                    [(-0.0102173269, 2), (0.0177389501, -2),
                     (2.82247162e-05, -4), (4.32740016e-05, -6),
                     (-2.15847266e-06, -8)]),
    'E-LAK7':      (2.68241693,
                    [(-0.0118445415, 2), (0.016242269, -2),
                     (0.000303298585, -4), (-8.32486246e-06, -6),
                     (7.74390846e-07, -8)]),
    'E-LASF016':   (3.07241378,
                    [(-0.0146849804, 2), (0.0244285857, -2),
                     (0.000398183255, -4), (1.22802438e-05, -6),
                     (1.23709929e-07, -8)]),
    'E-SK16':      (2.58572188,
                    [(-0.0114561541, 2), (0.0140913669, -2),
                     (0.000439966701, -4), (-2.67679001e-05, -6),
                     (1.51331053e-06, -8)]),
    'J-BASF7':     (2.82115391,
                    [(-0.0100514408, 2), (-9.24350756e-05, 4),
                     (0.0249821665, -2), (0.000523468025, -4),
                     (3.70259835e-05, -6), (-2.14786963e-06, -8),
                     (2.97698375e-07, -10)]),
    'J-FK01A':     (2.21785004,
                    [(-0.00552619544, 2), (-4.04219098e-05, 4),
                     (0.00839820345, -2), (8.8019088e-05, -4),
                     (1.15723877e-07, -6), (5.3817861e-08, -8)]),
    'J-LAK7':      (2.6823272,
                    [(-0.0119713031, 2), (-0.00014372436, 4),
                     (0.0164555463, -2), (0.000217295781, -4),
                     (4.69383509e-06, -6), (3.49394854e-08, -8)]),
    'J-LASF09A':   (3.21676146,
                    [(-0.013942445, 2), (-0.000118797124, 4),
                     (0.0279205286, -2), (0.000601395043, -4),
                     (-4.6499054e-06, -6), (1.72221463e-06, -8),
                     (-3.69002554e-08, -10)]),
    # ----- Sumita Optical Glass, Inc. (Zemax 2017-02-02) -----
    # 10 polynomial entries.  All Sumita entries share the catalogue's
    # documented 0.36-1.55 um polynomial fit band and use 5
    # coefficient pairs.  Names are stored uppercase-canonicalised
    # (e.g. K-LaK10 -> K-LAK10) per the GLASS_REGISTRY note above;
    # the catalogue's mixed-case page label is preserved in the
    # GLASS_REGISTRY tuple's third element so the refractiveindex.info
    # live lookup still resolves.
    'K-BK7':       (2.2705778,
                    [(-0.010059376, 2), (0.010414999, -2),
                     (0.00028872517, -4), (-2.2214495e-05, -6),
                     (1.4258559e-06, -8)]),
    'K-FK5':       (2.189197,
                    [(-0.0097642231, 2), (0.0086917382, -2),
                     (0.00023646623, -4), (-1.7699987e-05, -6),
                     (9.6989631e-07, -8)]),
    'K-LAFN3':     (2.827081,
                    [(-0.010220727, 2), (0.021224097, -2),
                     (0.0006653956, -4), (-3.3316539e-05, -6),
                     (3.16429e-06, -8)]),
    'K-LAK10':     (2.8960747,
                    [(-0.011713123, 2), (0.021599254, -2),
                     (0.00043811038, -4), (5.752903e-06, -6),
                     (-1.7364157e-07, -8)]),
    'K-LASFN10':   (3.2128137,
                    [(-0.015587464, 2), (0.028249299, -2),
                     (0.00086439607, -4), (-2.9175496e-05, -6),
                     (2.9912535e-06, -8)]),
    'K-PBK40':     (2.2744024,
                    [(-0.01070289, 2), (0.010577306, -2),
                     (0.00022857431, -4), (-7.2914225e-06, -6),
                     (2.4124054e-07, -8)]),
    'K-PFK90':     (2.1094643,
                    [(-0.0048026812, 2), (0.0067645351, -2),
                     (8.3329962e-05, -4), (3.4196126e-07, -6),
                     (-8.4933854e-08, -8)]),
    'K-PSKN2':     (2.578886,
                    [(-0.010381408, 2), (0.014233563, -2),
                     (0.00012483021, -4), (1.7477672e-05, -6),
                     (-9.6204805e-07, -8)]),
    'K-SK4':       (2.5587726,
                    [(-0.0098095616, 2), (0.014840563, -2),
                     (0.0003193166, -4), (-1.016436e-05, -6),
                     (6.6518794e-07, -8)]),
    'K-VC78':      (2.7376882,
                    [(-0.012269245, 2), (0.017439403, -2),
                     (0.00047088707, -4), (-2.7150134e-05, -6),
                     (1.8172045e-06, -8)]),
}


# v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
# manifest of formula-3 polynomial glass names that are present in
# GLASS_REGISTRY (as ``(shelf, book, page)`` tuples for the optional
# refractiveindex live lookup) but NOT yet ingested into
# POLYNOMIAL_COEFFICIENTS for the bundled-evaluator fallback.
#
# This set lets the dispatcher distinguish "user typo / unknown
# glass" (ValueError with suggestions) from "known formula-3 glass
# without bundled coefficients" (NotImplementedError directing the
# user to install the [glass] extra or open a coefficient-ingestion
# request).  Pre-v5.2 the latter path raised a generic ImportError
# that conflated "package not installed" with "package installed but
# this glass not yet covered", which made the v5.2.1 ingestion gap
# invisible to users.
#
# v5.2.3 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
# all 24 formula-3 catalogue entries (4 CDGM + 10 Hikari + 10 Sumita)
# have now been ingested into POLYNOMIAL_COEFFICIENTS verbatim from
# the refractiveindex.info YAML dataset, so the stub manifest is
# empty.  The frozenset is intentionally retained (rather than
# deleted) so that any future formula-3 catalogue additions land
# here as a stub before their coefficient row, and the migration-
# message dispatch arm in get_glass_index remains reachable.
#
# Catalogue and validity ranges for each entry are recorded in
# GLASS_REGISTRY / GLASS_VALIDITY (search by name).
_POLYNOMIAL_STUB_NAMES = frozenset()


def _guard_wavelength(wavelength_m, fn_label, *, sign_symmetric):
    """v5.4.6 (audit P3-4 / P3-7): shared negative / NaN wavelength guard
    for the dispersion evaluators, so ``glass._sellmeier_index``,
    ``glass._polynomial_index`` and ``coatings._coating_sellmeier`` all
    handle bad wavelengths identically (a fix landing in one no longer
    silently skips the others).

    NaN -> UserWarning (the result will be NaN; usually upstream
    propagation).  Negative wavelength: for a ``sign_symmetric`` form
    (Sellmeier, which depends on lam^2) warn and use ``abs`` -- a negative
    wavelength almost always signals a metres/microns unit error; for a
    NON sign-symmetric form (the polynomial, whose odd/negative exponents
    give ``n(-lam) != n(|lam|)``) raise ``ValueError`` rather than return
    a silently-wrong index.  Returns the (possibly abs'd) wavelength,
    preserving scalar-vs-array shape.
    """
    import warnings
    arr = np.asarray(wavelength_m, dtype=float)
    if np.any(np.isnan(arr)):
        warnings.warn(
            f"{fn_label}: wavelength contains NaN; the returned index will "
            f"be NaN.  Check for upstream NaN propagation.",
            UserWarning, stacklevel=3)
    if np.any(arr < 0.0):
        if sign_symmetric:
            warnings.warn(
                f"{fn_label}: negative wavelength(s) received; using |lambda| "
                f"(the Sellmeier form is sign-symmetric).  A negative "
                f"wavelength usually signals a metres/microns unit error.",
                UserWarning, stacklevel=3)
            if np.ndim(wavelength_m) == 0:
                return abs(float(wavelength_m))
            return np.abs(wavelength_m)
        bad = float(arr.flat[int(np.argmin(arr))])
        raise ValueError(
            f"{fn_label}: negative wavelength {bad*1e9:.3f} nm is not "
            f"permitted -- the polynomial dispersion form is NOT "
            f"sign-symmetric (n(-lambda) != n(|lambda|)).  Pass a positive "
            f"vacuum wavelength in metres.")
    return wavelength_m


def _polynomial_index(wavelength_m, coeffs, glass_name=None):
    """refractiveindex.info formula-3 (polynomial) evaluator.

    ``coeffs`` is ``(c0, [(c_i, exponent_i), ...])`` where ``c0`` is
    the constant term in ``n^2`` and the list pairs each polynomial
    coefficient with its lambda-exponent (in micron-units).  Returns
    the real refractive index at the given vacuum wavelength [m].

    The polynomial form is::

        n^2 = c0 + sum_i c_i * (lam_um) ** exponent_i

    which subsumes the Schott polynomial form
    ``n^2 = a0 + a1*lam^2 + a2*lam^-2 + a3*lam^-4 + a4*lam^-6 +
    a5*lam^-8``.

    v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
    accepts either a Python scalar (returns float, matching the v4.16.2
    contract and the ``_sellmeier_index`` sibling) or an array-like
    ``wavelength_m`` (returns an ndarray with the same shape).  Array
    inputs follow the numpy convention -- JAX / CuPy callers can pass
    their own arrays through and will receive numpy back; downstream
    consumers in ``get_glass_index`` always call this with a scalar
    so the contract stays narrow.
    """
    label = f" for glass {glass_name!r}" if glass_name else ""
    wavelength_m = _guard_wavelength(
        wavelength_m, f"_polynomial_index{label}", sign_symmetric=False)
    c0, pairs = coeffs
    # Scalar fast-path: mirror _sellmeier_index's float math so the
    # common get_glass_index('NAME', 587.6e-9) call stays a pure float
    # round-trip (no numpy allocation).
    if not hasattr(wavelength_m, '__len__') and not hasattr(
            wavelength_m, 'shape'):
        lam_um = float(wavelength_m) * 1e6
        n_sq = float(c0)
        for c_i, exponent_i in pairs:
            n_sq = n_sq + float(c_i) * (lam_um ** float(exponent_i))
        if n_sq <= 0.0:
            raise ValueError(
                f"_polynomial_index{label}: extrapolation produced non-"
                f"positive n^2 = {n_sq:.6f} at wavelength "
                f"{float(wavelength_m)*1e9:.3f} nm.  This wavelength is "
                f"likely outside the catalogue's valid range.")
        return _math.sqrt(n_sq)
    # Vector path: lift via numpy.  Works for lists, tuples, numpy
    # arrays, and (read-only) JAX arrays (which expose ``__array__``).
    lam_um = np.asarray(wavelength_m, dtype=float) * 1e6
    n_sq = np.full_like(lam_um, float(c0), dtype=float)
    for c_i, exponent_i in pairs:
        n_sq = n_sq + float(c_i) * (lam_um ** float(exponent_i))
    if np.any(n_sq <= 0.0):
        bad_idx = int(np.argmin(n_sq))
        bad_lam = float(np.asarray(wavelength_m, dtype=float).flat[bad_idx])
        raise ValueError(
            f"_polynomial_index{label}: extrapolation produced non-"
            f"positive n^2 = {float(n_sq.flat[bad_idx]):.6f} at "
            f"wavelength {bad_lam*1e9:.3f} nm (one of {n_sq.size} "
            f"inputs).  This wavelength is likely outside the "
            f"catalogue's valid range.")
    return np.sqrt(n_sq)


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
    label = f" for glass {glass_name!r}" if glass_name else ""
    wavelength_m = _guard_wavelength(
        wavelength_m, f"_sellmeier_index{label}", sign_symmetric=True)
    lam2 = (wavelength_m * 1e6) ** 2  # wavelength^2 in um^2
    (B1, B2, B3), (C1, C2, C3) = coeffs
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
    # 4.14.2 (P0-NEW-1): S-LAH64 / S-LAH79 were flagged '__sellmeier__'
    # in v4.7 but their Sellmeier rows were REMOVED in v4.11.2 after
    # round-3 audit CRIT-3 found the bundled coefficients were
    # misattributed (off by ~6% and ~12% vs Ohara catalogue n_d).
    # The dispatcher then raised ValueError on every lookup; the
    # v4.11.2 fix forgot to re-route GLASS_REGISTRY at the authorita-
    # tive refractiveindex.info catalogue.  Restored via the tuple
    # form here so prescriptions referencing S-LAH64 / S-LAH79 (e.g.
    # the ui/ "known-good preset" path) work again, falling back to
    # ImportError with a clear message if the optional ``refractive-
    # index`` package is not installed.
    'S-LAH64':      ('specs', 'OHARA-optical', 'S-LAH64'),
    'S-LAH79':      ('specs', 'OHARA-optical', 'S-LAH79'),
    'BaF2':         '__sellmeier__',

    # ============================================================
    # v4.16.0 (ROADMAP #13) -- CDGM / Hikari / Sumita catalogues
    # ============================================================
    # Each entry resolves via refractiveindex.info when the optional
    # ``refractiveindex`` package is installed.  CDGM Sellmeier-2
    # entries also have bundled coefficients in SELLMEIER_COEFFICIENTS
    # as a fallback for minimal installs.  CDGM polynomial entries
    # (formula 3) and all Hikari / Sumita entries require the
    # ``refractiveindex`` package -- formula 3 is not currently in
    # the bundled fallback (v4.16.1 candidate).

    # ----- CDGM (Tianjin Lingxin / China Daheng Group) -----
    # Sellmeier-2 entries (also have bundled Sellmeier coefficients):
    'H-K9L':        ('specs', 'CDGM-optical', 'H-K9L'),
    'H-LAK52':      ('specs', 'CDGM-optical', 'H-LAK52'),
    'H-LAK53A':     ('specs', 'CDGM-optical', 'H-LAK53A'),
    'H-ZK9B':       ('specs', 'CDGM-optical', 'H-ZK9B'),
    'H-ZF12':       ('specs', 'CDGM-optical', 'H-ZF12'),
    'D-ZK3':        ('specs', 'CDGM-optical', 'D-ZK3'),
    'D-LAK52':      ('specs', 'CDGM-optical', 'D-LAK52'),
    'H-ZLAF52A':    ('specs', 'CDGM-optical', 'H-ZLAF52A'),
    # CDGM polynomial-formula entries (refractiveindex required):
    'H-ZK7':        ('specs', 'CDGM-optical', 'H-ZK7'),
    'H-ZF52A':      ('specs', 'CDGM-optical', 'H-ZF52A'),
    'F1-CDGM':      ('specs', 'CDGM-optical', 'F1'),
    'F2-CDGM':      ('specs', 'CDGM-optical', 'F2'),

    # ----- Hikari Glass Co. Ltd -----
    # All Hikari entries below use formula 3 (polynomial) in the
    # refractiveindex.info catalogue and require the optional
    # ``refractiveindex`` package for lookup.
    'E-LASF016':    ('specs', 'HIKARI-optical', 'E-LASF016'),
    'E-SK16':       ('specs', 'HIKARI-optical', 'E-SK16'),
    'E-LAK7':       ('specs', 'HIKARI-optical', 'E-LAK7'),
    'E-LAK04':      ('specs', 'HIKARI-optical', 'E-LAK04'),
    'E-BAK1':       ('specs', 'HIKARI-optical', 'E-BAK1'),
    'J-FK01A':      ('specs', 'HIKARI-optical', 'J-FK01A'),
    'J-LASF09A':    ('specs', 'HIKARI-optical', 'J-LASF09A'),
    'J-LAK7':       ('specs', 'HIKARI-optical', 'J-LAK7'),
    'J-BASF7':      ('specs', 'HIKARI-optical', 'J-BASF7'),
    'E-F2':         ('specs', 'HIKARI-optical', 'E-F2'),

    # ----- Sumita Optical Glass, Inc. -----
    # All Sumita entries below use formula 3 (polynomial) in the
    # refractiveindex.info catalogue and require the optional
    # ``refractiveindex`` package for lookup.
    #
    # NOTE: Sumita uses a mixed-case naming convention in their
    # catalogue (e.g. K-LaK10, K-LaSFn10).  v4.16.0 canonicalises
    # these to all-uppercase in GLASS_REGISTRY (e.g. K-LAK10,
    # K-LASFN10) to keep names consistent with the Schott / Ohara
    # convention and to match user expectations when grepping
    # ``search_glasses('LASF')``.  The third tuple element retains
    # the catalogue's mixed-case page name so the
    # refractiveindex.info lookup still resolves.
    'K-VC78':       ('specs', 'SUMITA-optical', 'K-VC78'),
    'K-LAK10':      ('specs', 'SUMITA-optical', 'K-LaK10'),
    'K-LASFN10':    ('specs', 'SUMITA-optical', 'K-LaSFn10'),
    'K-SK4':        ('specs', 'SUMITA-optical', 'K-SK4'),
    'K-PFK90':      ('specs', 'SUMITA-optical', 'K-PFK90'),
    'K-PBK40':      ('specs', 'SUMITA-optical', 'K-PBK40'),
    'K-BK7':        ('specs', 'SUMITA-optical', 'K-BK7'),
    'K-PSKN2':      ('specs', 'SUMITA-optical', 'K-PSKn2'),
    'K-FK5':        ('specs', 'SUMITA-optical', 'K-FK5'),
    'K-LAFN3':      ('specs', 'SUMITA-optical', 'K-LaFn3'),
}


# ---------------------------------------------------------------------------
# v4.16.0 (ROADMAP #14) -- Sellmeier validity ranges per glass
# ---------------------------------------------------------------------------
#
# Each ``GLASS_VALIDITY[name]`` is a tuple ``(lambda_min, lambda_max)``
# in METRES specifying the wavelength range over which the bundled
# Sellmeier / refractiveindex.info dispersion fit is documented as
# valid.  When ``get_glass_index(name, wavelength)`` is called with a
# wavelength OUTSIDE this range, a ``UserWarning`` is emitted -- but
# the value is still returned (extrapolation is sometimes acceptable
# for design-space exploration).
#
# Sources:
# * Schott / Ohara entries: SCHOTT TIE-29 "Refractive index and
#   dispersion" catalogue ranges (typically 0.30-2.5 um for crown
#   and flint; 0.36-2.4 um for lanthanum-flint due to UV cutoff).
# * Bundled fused-silica: Malitson 1965, J. Opt. Soc. Am. 55, 1205
#   (valid 0.21-6.7 um; we restrict to 0.21-3.7 um where the fit
#   stays within 5e-5 of the tabulated values).
# * CDGM / Hikari / Sumita entries: the ``wavelength_range`` field
#   from the corresponding refractiveindex.info YAML (sourced from
#   the CDGM 2022-06 / HIKARI 2017-11 / SUMITA 2017-02 Zemax
#   catalogue exports).
#
# Glasses without an explicit entry default to ``(0.0, np.inf)`` --
# no warning ever fires.  Documented exemptions: ``'air'``, the
# ``'__thin_lens__'`` sentinel, and user-registered callables.

_DEFAULT_VALIDITY = (0.0, float('inf'))  # no warning if not in dict

GLASS_VALIDITY = {
    # ----- Schott N-series (catalog range 0.30-2.5 um typical) -----
    'N-BK7':      (0.30e-6, 2.5e-6),
    'N-K5':       (0.31e-6, 2.5e-6),
    'N-BAF10':    (0.31e-6, 2.5e-6),
    'N-BAK4':     (0.31e-6, 2.5e-6),
    'N-BAF52':    (0.31e-6, 2.5e-6),
    'N-FK51A':    (0.30e-6, 2.5e-6),
    'N-PSK53A':   (0.31e-6, 2.5e-6),
    # LaK / LaF / LaSF -- higher UV cutoff due to lanthanum content.
    'N-LAK22':    (0.36e-6, 2.4e-6),
    'N-LAK33A':   (0.36e-6, 2.4e-6),
    'N-LAK33B':   (0.36e-6, 2.4e-6),
    'N-SK11':     (0.31e-6, 2.5e-6),
    'N-SK16':     (0.31e-6, 2.5e-6),
    'N-SSK2':     (0.31e-6, 2.5e-6),
    'N-SSK8':     (0.31e-6, 2.5e-6),
    'N-F2':       (0.32e-6, 2.5e-6),
    'N-SF2':      (0.32e-6, 2.5e-6),
    'N-SF5':      (0.36e-6, 2.5e-6),
    'N-SF6':      (0.37e-6, 2.5e-6),
    'N-SF6HT':    (0.37e-6, 2.5e-6),
    'N-SF10':     (0.38e-6, 2.5e-6),
    'N-SF11':     (0.40e-6, 2.5e-6),
    'N-SF14':     (0.40e-6, 2.5e-6),
    'N-SF15':     (0.36e-6, 2.5e-6),
    'N-SF57':     (0.40e-6, 2.5e-6),
    'N-LASF31A':  (0.36e-6, 2.4e-6),
    'N-LASF40':   (0.40e-6, 2.4e-6),
    'N-LASF41':   (0.36e-6, 2.4e-6),
    'N-LASF44':   (0.36e-6, 2.4e-6),
    'N-LASF45':   (0.40e-6, 2.4e-6),
    'N-LASF46A':  (0.40e-6, 2.4e-6),
    'N-LASF46B':  (0.40e-6, 2.4e-6),
    'N-LASF9':    (0.36e-6, 2.4e-6),
    # ----- Legacy Schott flints / SF (pre-N series) -----
    'F2':         (0.36e-6, 2.5e-6),
    'F5':         (0.36e-6, 2.5e-6),
    'SF2':        (0.36e-6, 2.5e-6),
    # ----- Ohara S-LAH (range from refractiveindex.info YAML) -----
    'S-LAH64':    (0.32e-6, 2.4e-6),
    'S-LAH79':    (0.37e-6, 2.4e-6),
    # ----- Generic materials -----
    'CaF2':       (0.13e-6, 9.4e-6),
    'MgF2':       (0.12e-6, 7.0e-6),
    'BaF2':       (0.15e-6, 12.0e-6),
    'SiO2':       (0.21e-6, 3.7e-6),     # Malitson 1965 (valid to 6.7 um)
    'F_SILICA':   (0.21e-6, 3.7e-6),
    'FUSED_SILICA': (0.21e-6, 3.7e-6),
    'SILICA':     (0.21e-6, 3.7e-6),
    'SILICON':    (1.20e-6, 14.0e-6),    # Li-293K range
    # ----- CDGM (v4.16.0, range from refractiveindex.info YAML) -----
    'H-K9L':      (0.302e-6, 2.325e-6),
    'H-LAK52':    (0.302e-6, 2.325e-6),
    'H-LAK53A':   (0.302e-6, 2.325e-6),
    'H-ZK9B':     (0.302e-6, 2.325e-6),
    'H-ZF12':     (0.4047e-6, 2.325e-6),
    'D-ZK3':      (0.302e-6, 2.325e-6),
    'D-LAK52':    (0.302e-6, 2.325e-6),
    'H-ZLAF52A':  (0.365e-6, 2.325e-6),
    'H-ZK7':      (0.365e-6, 1.014e-6),
    'H-ZF52A':    (0.4047e-6, 1.014e-6),
    'F1-CDGM':    (0.365e-6, 0.7065e-6),
    'F2-CDGM':    (0.365e-6, 1.014e-6),
    # ----- Hikari (v4.16.0, range from refractiveindex.info YAML) -----
    # Hikari E-* / J-* glasses use formula 3 in their YAML with a
    # wavelength_range typically 0.4-0.7 um for the polynomial fit;
    # below the lower bound the polynomial extrapolation diverges.
    'E-LASF016':  (0.4e-6, 0.7e-6),
    'E-SK16':     (0.4e-6, 0.7e-6),
    'E-LAK7':     (0.4e-6, 0.7e-6),
    'E-LAK04':    (0.4e-6, 0.7e-6),
    'E-BAK1':     (0.4e-6, 0.7e-6),
    'J-FK01A':    (0.4e-6, 0.7e-6),
    'J-LASF09A':  (0.4e-6, 0.7e-6),
    'J-LAK7':     (0.4e-6, 0.7e-6),
    'J-BASF7':    (0.4e-6, 0.7e-6),
    'E-F2':       (0.4e-6, 0.7e-6),
    # ----- Sumita (v4.16.0, range from refractiveindex.info YAML) -----
    # Sumita polynomial fits documented 0.36-1.55 um.  Names are
    # canonicalised to all-uppercase per the GLASS_REGISTRY note
    # above.
    'K-VC78':     (0.36e-6, 1.55e-6),
    'K-LAK10':    (0.36e-6, 1.55e-6),
    'K-LASFN10':  (0.36e-6, 1.55e-6),
    'K-SK4':      (0.36e-6, 1.55e-6),
    'K-PFK90':    (0.36e-6, 1.55e-6),
    'K-PBK40':    (0.36e-6, 1.55e-6),
    'K-BK7':      (0.36e-6, 1.55e-6),
    'K-PSKN2':    (0.36e-6, 1.55e-6),
    'K-FK5':      (0.36e-6, 1.55e-6),
    'K-LAFN3':    (0.36e-6, 1.55e-6),
}


# v4.16.0: one-shot warn-once memo for out-of-range Sellmeier
# extrapolations.  Pre-v4.16 the dispatcher emitted no signal when
# the user asked for a wavelength outside the documented Sellmeier
# fit band (e.g. ``get_glass_index('N-BK7', 200e-9)``), silently
# returning an extrapolated number that has no physical meaning.
# v4.16 warns the first time per (glass, wavelength_nm) pair.
_validity_warned: set = set()


def _maybe_warn_outside_validity(glass_name, wavelength_m):
    """Emit a one-shot ``UserWarning`` if ``wavelength_m`` falls
    outside the documented Sellmeier validity range for ``glass_name``.

    No-op if the glass has no explicit entry in ``GLASS_VALIDITY``
    (default ``(0.0, np.inf)`` -- no warning ever fires) or if the
    wavelength is inside the documented range.

    The warning is rate-limited to one emission per
    ``(glass_name, wavelength_nm)`` pair to avoid log churn in
    parameter sweeps that scan a few discrete wavelengths.
    """
    lmin, lmax = GLASS_VALIDITY.get(glass_name, _DEFAULT_VALIDITY)
    # GL-nit (AUDIT_GLASS_POLARIZATION): the evaluators and value cache both
    # handle array wavelengths, but this validity check used ``float(...)``
    # which raised on an array.  Be array-safe: warn if ANY element is
    # outside the band, and report the first offending wavelength.
    wl = np.asarray(wavelength_m, dtype=float).reshape(-1)
    inside = (wl >= lmin) & (wl <= lmax)
    if inside.all():
        return
    wl_report = float(wl[~inside][0])
    key = (str(glass_name), round(wl_report * 1e9, 1))
    if key in _validity_warned:
        return
    _validity_warned.add(key)
    import warnings
    warnings.warn(
        f"get_glass_index: {glass_name} Sellmeier validity is "
        f"[{lmin:.3e}, {lmax:.3e}] m; got wavelength "
        f"{wl_report:.3e} m.  Extrapolated value may not be "
        f"physical.",
        UserWarning, stacklevel=3,
    )


# v4.16.1 (audit P1-NEW-F2-1 / C.3): documented exemptions from the
# GLASS_REGISTRY -> GLASS_VALIDITY direction.  These registry entries
# legitimately have no GLASS_VALIDITY entry and the consistency check
# must skip them rather than treating their absence as drift.
#
# * ``'air'`` -- callable; uses Edlen-form ambient model with no
#   single physical Sellmeier validity range to pin.
# * ``'__thin_lens__'`` -- internal marker, not a real glass.
# * ``'__MIRROR__'`` -- internal marker, not a real glass.
# * ``'vacuum'`` -- callable; returns n=1.0 at every wavelength.
_GLASS_VALIDITY_REGISTRY_EXEMPTIONS = frozenset({
    'air',
    '__thin_lens__',
    '__MIRROR__',
    'vacuum',
})


def _check_glass_registry_consistency():
    """4.14.2 (P0-NEW-1 meta-pattern): convert the class-of-bug
    (``GLASS_REGISTRY`` entry flagged ``'__sellmeier__'`` but absent
    from :data:`SELLMEIER_COEFFICIENTS`) into a fail-fast at module
    load, so a future drift can never re-surface as a silent
    ``ValueError`` at first call.

    Six checks (v4.14.2 forward + v4.15 reverse + v4.16.1
    GLASS_VALIDITY -> GLASS_REGISTRY + v4.16.1 tuple well-formedness +
    v4.16.3 polynomial forward/reverse).

    * **Forward** (v4.14.2): every ``'__sellmeier__'``-flagged
      registry entry must have a coefficient row.
    * **Polynomial forward** (v4.16.3, audit P3-NEW-F1-1): every
      ``'__polynomial__'``-flagged registry entry must have a
      :data:`POLYNOMIAL_COEFFICIENTS` row.  Sibling to the Sellmeier
      forward check.
    * **Polynomial reverse** (v4.16.3, audit P3-NEW-F1-1): every row
      in :data:`POLYNOMIAL_COEFFICIENTS` must appear in
      :data:`GLASS_REGISTRY`.  Sibling to the Sellmeier reverse check.
    * **Reverse** (v4.15, P2): every row in
      :data:`SELLMEIER_COEFFICIENTS` must appear in
      :data:`GLASS_REGISTRY`.  Pre-v4.15 a coefficient row added
      without a corresponding registry entry was silent dead code
      (the dispatcher never consulted ``SELLMEIER_COEFFICIENTS``
      unless the registry first routed there) -- the reverse check
      surfaces such an orphan immediately at import time.

      For tuple-style entries (where the registry routes to
      refractiveindex.info first), the coefficient row is a legal
      fallback for minimal installs and not an orphan.  We accept
      the reverse-direction membership regardless of whether the
      registry entry is ``'__sellmeier__'`` or a tuple -- both
      paths consult ``SELLMEIER_COEFFICIENTS`` (the tuple path only
      when ``_REFRACTIVEINDEX_AVAILABLE`` is False).
    * **GLASS_VALIDITY -> GLASS_REGISTRY** (v4.16.1 audit P1-NEW-F2-1
      / C.3): every key in :data:`GLASS_VALIDITY` must appear in
      :data:`GLASS_REGISTRY`.  A validity entry without a registry
      entry is unreachable -- the validity warn helper looks up by
      registry name, so an orphan GLASS_VALIDITY row never fires
      its warning.  (The reverse direction
      GLASS_REGISTRY -> GLASS_VALIDITY is intentionally NOT enforced
      as a hard requirement -- missing validity defaults to the
      no-warning sentinel ``(0.0, inf)``.  Users registering a custom
      callable glass typically do not declare a wavelength range.)
    * **Tuple well-formedness** (v4.16.1 audit P1-NEW-F2-1 / C.3):
      each GLASS_VALIDITY entry must be a 2-tuple
      ``(lambda_min, lambda_max)`` with ``lambda_min < lambda_max``
      and both finite-non-negative.  A malformed tuple would silently
      pass or always-fire on the dispatch path.
    """
    # Forward: __sellmeier__ flag -> row must exist.
    for name, entry in GLASS_REGISTRY.items():
        if entry == '__sellmeier__' and name not in SELLMEIER_COEFFICIENTS:
            raise RuntimeError(
                f"GLASS_REGISTRY drift: {name!r} flagged "
                f"'__sellmeier__' but missing from "
                f"SELLMEIER_COEFFICIENTS.  Either add the Sellmeier "
                f"coefficients or change the registry entry to a "
                f"(shelf, book, page) tuple / callable."
            )
    # v4.16.3 (audit P3-NEW-F1-1): Forward for __polynomial__ flag ->
    # row must exist.  Sibling to the __sellmeier__ forward check above.
    for name, entry in GLASS_REGISTRY.items():
        if entry == '__polynomial__' and name not in POLYNOMIAL_COEFFICIENTS:
            raise RuntimeError(
                f"GLASS_REGISTRY drift: {name!r} flagged "
                f"'__polynomial__' but missing from "
                f"POLYNOMIAL_COEFFICIENTS.  Either add the polynomial "
                f"coefficients or change the registry entry to a "
                f"(shelf, book, page) tuple / callable."
            )
    # Reverse (v4.15, P2): row -> registry entry must exist.  Without
    # a registry entry the row is unreachable: ``get_glass_index``
    # raises ``ValueError`` before it inspects SELLMEIER_COEFFICIENTS,
    # so adding coefficients but forgetting the registry pointer was
    # a silent no-op pre-v4.15.
    for name in SELLMEIER_COEFFICIENTS:
        if name not in GLASS_REGISTRY:
            raise RuntimeError(
                f"GLASS_REGISTRY drift: SELLMEIER_COEFFICIENTS[{name!r}] "
                f"has no corresponding GLASS_REGISTRY entry.  The row "
                f"is unreachable -- get_glass_index({name!r}, ...) will "
                f"raise ValueError before reaching the Sellmeier "
                f"fallback.  Add a registry entry: '__sellmeier__' "
                f"for pure-Sellmeier glasses, or a "
                f"(shelf, book, page) tuple for glasses also covered "
                f"by refractiveindex.info."
            )
    # v4.16.3 (audit P3-NEW-F1-1): Reverse for POLYNOMIAL_COEFFICIENTS.
    # Without a registry entry the row is unreachable -- get_glass_index
    # raises ValueError before consulting POLYNOMIAL_COEFFICIENTS.  A
    # tuple-style entry that also routes here on the refractiveindex-
    # unavailable fallback is accepted (parallel to the Sellmeier
    # reverse check's tuple acceptance above).
    for name in POLYNOMIAL_COEFFICIENTS:
        if name not in GLASS_REGISTRY:
            raise RuntimeError(
                f"GLASS_REGISTRY drift: POLYNOMIAL_COEFFICIENTS[{name!r}] "
                f"has no corresponding GLASS_REGISTRY entry.  The row "
                f"is unreachable -- get_glass_index({name!r}, ...) will "
                f"raise ValueError before reaching the polynomial "
                f"fallback.  Add a registry entry: '__polynomial__' "
                f"for pure-polynomial glasses, or a "
                f"(shelf, book, page) tuple for glasses also covered "
                f"by refractiveindex.info."
            )
    # v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
    # _POLYNOMIAL_STUB_NAMES well-formedness.  Every stub manifest entry
    # must (a) be present in GLASS_REGISTRY (else the
    # NotImplementedError dispatch arm is unreachable) and (b) NOT be
    # present in POLYNOMIAL_COEFFICIENTS (else the dispatcher's
    # POLYNOMIAL_COEFFICIENTS check would short-circuit before the
    # stub check, producing a confusing "evaluator ran with empty
    # coeffs" error instead of the documented migration text).  Both
    # invariants caught at module load so a future ingestion PR that
    # forgets to remove the stub entry surfaces immediately.
    for name in _POLYNOMIAL_STUB_NAMES:
        if name not in GLASS_REGISTRY:
            raise RuntimeError(
                f"GLASS_REGISTRY drift (v5.2): "
                f"_POLYNOMIAL_STUB_NAMES[{name!r}] has no corresponding "
                f"GLASS_REGISTRY entry.  The NotImplementedError "
                f"migration dispatch is unreachable -- get_glass_index"
                f"({name!r}, ...) raises ValueError (unknown-glass) "
                f"before reaching the stub fallback.  Add a registry "
                f"entry or remove {name!r} from _POLYNOMIAL_STUB_NAMES.")
        if name in POLYNOMIAL_COEFFICIENTS:
            raise RuntimeError(
                f"GLASS_REGISTRY drift (v5.2): "
                f"_POLYNOMIAL_STUB_NAMES[{name!r}] also appears in "
                f"POLYNOMIAL_COEFFICIENTS.  These are mutually "
                f"exclusive states (stubbed vs ingested) -- remove "
                f"{name!r} from _POLYNOMIAL_STUB_NAMES at the same "
                f"commit that lands the coefficient ingestion.")
    # v4.16.1 (audit P1-NEW-F2-1 / C.3): GLASS_VALIDITY -> GLASS_REGISTRY.
    # An entry in GLASS_VALIDITY without a GLASS_REGISTRY counterpart is
    # unreachable -- the warn helper looks up by registry name.
    for name in GLASS_VALIDITY:
        if name in _GLASS_VALIDITY_REGISTRY_EXEMPTIONS:
            continue
        if name not in GLASS_REGISTRY:
            raise RuntimeError(
                f"GLASS_REGISTRY drift (v4.16.1): "
                f"GLASS_VALIDITY[{name!r}] has no corresponding "
                f"GLASS_REGISTRY entry.  The validity range is "
                f"unreachable -- the warn helper looks up by "
                f"registry name, so the warning would never fire "
                f"for {name!r}.  Add a registry entry, or remove "
                f"the validity row, or list {name!r} in "
                f"_GLASS_VALIDITY_REGISTRY_EXEMPTIONS with a cited "
                f"rationale."
            )
    # v4.16.1 (audit P1-NEW-F2-1 / C.3): tuple well-formedness for every
    # GLASS_VALIDITY entry.  Each must be a 2-tuple
    # (lambda_min, lambda_max) with lmin < lmax and both finite,
    # non-negative.
    for name, value in GLASS_VALIDITY.items():
        if not isinstance(value, tuple) or len(value) != 2:
            raise RuntimeError(
                f"GLASS_VALIDITY drift (v4.16.1): "
                f"GLASS_VALIDITY[{name!r}] = {value!r} is not a "
                f"2-tuple (lambda_min, lambda_max).  Fix the entry "
                f"format."
            )
        lmin, lmax = value
        # v4.16.2 (audit P3-NEW-F1-4): accept numpy scalars in addition
        # to native Python ``int`` / ``float``.  ``isinstance(np.int32(0),
        # (int, float))`` is False on Python 3.10+; users passing
        # ``GLASS_VALIDITY['X'] = (np.int32(300e-9), np.float32(700e-9))``
        # are tripping the v4.16.1 well-formedness check.
        _NUMERIC_TYPES = (int, float, np.integer, np.floating)
        if not (isinstance(lmin, _NUMERIC_TYPES)
                and isinstance(lmax, _NUMERIC_TYPES)):
            raise RuntimeError(
                f"GLASS_VALIDITY drift (v4.16.1): "
                f"GLASS_VALIDITY[{name!r}] = {value!r} must be a "
                f"2-tuple of numeric (int / float / np.integer / "
                f"np.floating) bounds; got types "
                f"({type(lmin).__name__}, {type(lmax).__name__})."
            )
        if not (lmin < lmax):
            raise RuntimeError(
                f"GLASS_VALIDITY drift (v4.16.1): "
                f"GLASS_VALIDITY[{name!r}] = ({lmin!r}, {lmax!r}) "
                f"requires lambda_min < lambda_max."
            )
        if lmin < 0.0 or lmax <= 0.0:
            raise RuntimeError(
                f"GLASS_VALIDITY drift (v4.16.1): "
                f"GLASS_VALIDITY[{name!r}] = ({lmin!r}, {lmax!r}) "
                f"requires non-negative lambda_min and positive "
                f"lambda_max."
            )


_check_glass_registry_consistency()


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

# v5.17.1 (audit P3-40): lock guarding ``_glass_value_cache`` LRU
# bookkeeping (get / move_to_end / setitem / popitem) and the registry-
# driven ``_clear_glass_caches`` drain.  Follows the ``_ASM_CACHE_LOCK``
# precedent in ``propagators/propagation.py``.  Never held while calling
# out (``compute`` runs outside the lock), so no lock-order coupling
# with any other cache lock (see the P3-55 lock-discipline bug class).
_GLASS_CACHE_LOCK = _threading.Lock()

# Upper-case alias for the v4.14.2 cache<->lock dispatcher pin (its
# runtime walker pairs ``_GLASS_CACHE_LOCK`` with a ``_GLASS_CACHE``
# companion).  SAME dict object as ``_glass_cache`` -- the lower-case
# canonical name predates the ``_FOO_CACHE`` convention and is kept
# because tests / raytrace.trace / user_library mutate it in place by
# that name.  Nothing may ever REBIND either name.
_GLASS_CACHE = _glass_cache

# v5.6: value cache for the IMMUTABLE-catalogue dispatch branches only
# (__sellmeier__, __polynomial__ and the refractiveindex-unavailable
# Sellmeier / polynomial fallback).  Keyed on (glass_name, wavelength in
# picometres) -> float index.  (v5.17.1 audit P3-40: the historical
# "femtometre" wording was wrong -- ``round(wavelength * 1e12)`` is
# picometre resolution, still far below any optical relevance.)  It is
# consulted ONLY inside those branches (after the entry sentinel is
# confirmed), so re-registering a name under a different dispatch (a
# callable, a tuple, register_fixed_glass) can never serve a stale
# value; ``register_fixed_glass`` clears it as well.  Array wavelengths
# bypass it (not hashable / not the hot scalar path).
#
# v5.17.1 (audit P3-40): LRU-bounded OrderedDict (was an unbounded plain
# dict) + enrolled in the central cache registry as ``'glass_caches'``
# (see ``_clear_glass_caches`` at the bottom of this module).  Values
# are immutable floats, so eviction can never mutate a value a caller
# already holds, and the recompute is a pure Sellmeier / polynomial
# evaluation -- byte-identical on hit, miss, and post-eviction.
_glass_value_cache = OrderedDict()

# v5.17.1 (audit P3-40) bound rationale: one entry costs ~100 B (tuple
# key + float).  Only the 28 ``'__sellmeier__'`` / ``'__polynomial__'``
# sentinel glasses (plus the no-refractiveindex fallback) use this
# cache, so a dense dispersive sweep -- e.g. 1000 wavelengths x every
# sentinel glass = 28 000 entries -- fits with >2x headroom and no
# recompute inside a single solve.  65536 entries cap the cache at
# ~6-13 MB while still requiring a >1e6-unique-wavelength Monte-Carlo
# (the audit's leak scenario) to ever cycle.
_GLASS_VALUE_CACHE_SIZE = 65536


def _cached_glass_value(glass_name, wavelength, compute):
    """Memoise an immutable-catalogue index value keyed on
    ``(glass_name, round(wavelength * 1e12))`` (picometre resolution, far
    below any optical relevance).  Scalar wavelengths only; arrays recompute.
    ``compute`` is a zero-arg callable returning the float index.

    v5.17.1 (audit P3-40): LRU-bounded at ``_GLASS_VALUE_CACHE_SIZE``
    entries; mutations serialised by ``_GLASS_CACHE_LOCK``.  ``compute``
    runs OUTSIDE the lock (pure float math; two racing threads may both
    compute, last write wins with an identical value)."""
    if np.ndim(wavelength) != 0:
        return compute()
    key = (glass_name, round(float(wavelength) * 1e12))
    with _GLASS_CACHE_LOCK:
        cached = _glass_value_cache.get(key)
        if cached is not None:
            _glass_value_cache.move_to_end(key)
            return cached
    value = compute()
    with _GLASS_CACHE_LOCK:
        _glass_value_cache[key] = value
        _glass_value_cache.move_to_end(key)
        while len(_glass_value_cache) > _GLASS_VALUE_CACHE_SIZE:
            _glass_value_cache.popitem(last=False)
    return value


def _invalidate_glass_name(glass_name):
    """Drop every cached resolution for ``glass_name`` (v5.17.1, audit
    P2-41): the ``_glass_cache`` object (stale ``_FixedIndex`` or a
    previously-loaded ``RefractiveIndexMaterial``) and any
    ``_glass_value_cache`` entries keyed on the name.

    Called by :func:`lumenairy.user_library.load_material` when its
    catalog branch re-points ``GLASS_REGISTRY[name]``; mirrors the
    surgical invalidation ``raytrace.trace._register_fixed_index``
    performs on overwrite.  The warn-once sets are intentionally left
    alone (a repeated warning is noisier than a stale one is harmful).
    """
    with _GLASS_CACHE_LOCK:
        _glass_cache.pop(glass_name, None)
        for _key in [k for k in _glass_value_cache if k[0] == glass_name]:
            del _glass_value_cache[_key]


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
    # No validity check -- user-supplied callables define their own
    # valid band.
    if callable(entry):
        n = entry(wavelength)
        if isinstance(n, complex):
            return float(n.real)
        return float(n)

    # v4.16.0 (ROADMAP #14): validity-range warning.  Emitted before
    # the actual lookup so the caller sees the warning even if the
    # lookup returns successfully.  Skipped for non-physical entries
    # (callables, '__thin_lens__', user-fixed) which are handled
    # above / below.
    _maybe_warn_outside_validity(glass_name, wavelength)

    # Bundled Sellmeier coefficients (no external dependency).
    # v4.16.3 (audit P3-NEW-V3-1): dispatch order is SELLMEIER -> POLYNOMIAL,
    # NOT the "POLYNOMIAL -> SELLMEIER" wording that drifted into the
    # v4.16.2 CHANGELOG / release notes.  Both sentinels are disjoint at
    # the GLASS_REGISTRY level (a name carries one or the other) and the
    # consistency check enforces the row exists, so the order is
    # cosmetic for correctness; it matters only for documentation
    # truthfulness.
    if entry == '__sellmeier__':
        if glass_name not in SELLMEIER_COEFFICIENTS:
            raise ValueError(
                f"Glass {glass_name!r} is flagged '__sellmeier__' in "
                f"GLASS_REGISTRY but has no entry in "
                f"SELLMEIER_COEFFICIENTS.")
        return _cached_glass_value(
            glass_name, wavelength,
            lambda: _sellmeier_index(wavelength,
                                     SELLMEIER_COEFFICIENTS[glass_name]))

    # v4.16.3 (audit P3-NEW-F1-1): __polynomial__ sentinel parallel to
    # __sellmeier__.  Pre-v4.16.3 the bundled formula-3 polynomial
    # evaluator was reachable ONLY via the refractiveindex-unavailable
    # fallback below -- which means a user who followed the
    # ``pip install lumenairy[glass]`` recommendation in the README
    # could never hit the bundled polynomial path.  The sentinel makes
    # the polynomial dispatch first-class: any glass registered as
    # ``'__polynomial__'`` is resolved via POLYNOMIAL_COEFFICIENTS
    # regardless of whether refractiveindex is installed.  Sibling
    # parity with the __sellmeier__ branch above (error wording,
    # consistency-check coverage).
    if entry == '__polynomial__':
        if glass_name not in POLYNOMIAL_COEFFICIENTS:
            raise ValueError(
                f"Glass {glass_name!r} is flagged '__polynomial__' in "
                f"GLASS_REGISTRY but has no entry in "
                f"POLYNOMIAL_COEFFICIENTS.")
        return _cached_glass_value(
            glass_name, wavelength,
            lambda: _polynomial_index(wavelength,
                                      POLYNOMIAL_COEFFICIENTS[glass_name],
                                      glass_name=glass_name))

    # Internal thin-lens placeholder marker (used by the thin-lens
    # helpers); should never reach a propagation path, but if a user
    # somehow calls it, fall back to n=1.5 so the call doesn't crash.
    if entry == '__thin_lens__':
        return 1.5

    # 4.10.2: user-registered fixed-index glass (via register_fixed_glass).
    # The registry entry is the sentinel tuple
    # ('__user__', '__fixed__', '__fixed__') and the callable lives in
    # _glass_cache.  Pre-4.10.2 this fell through to the tuple branch
    # below, which raised ImportError if refractiveindex wasn't
    # installed -- making register_fixed_glass unusable on minimal
    # installs (the exact case where the feature is most useful).
    if (isinstance(entry, tuple) and len(entry) == 3
            and entry[0] == '__user__' and entry[1] == '__fixed__'
            and entry[2] == '__fixed__'):
        cached = _glass_cache.get(glass_name)
        if cached is not None:
            return float(cached.get_refractive_index(
                wavelength * 1e9, unit='nm'))
        raise ValueError(
            f"Glass {glass_name!r} is flagged as user-fixed but has no "
            f"_glass_cache entry.  Re-call register_fixed_glass().")

    # Tuple-style entry: dispatch to refractiveindex.info.
    if not _REFRACTIVEINDEX_AVAILABLE:
        # If we have Sellmeier coefficients bundled for the same name,
        # use them rather than raising.
        if glass_name in SELLMEIER_COEFFICIENTS:
            return _cached_glass_value(
                glass_name, wavelength,
                lambda: _sellmeier_index(
                    wavelength, SELLMEIER_COEFFICIENTS[glass_name]))
        # v4.16.2 (pre-v5.0 prep): formula-3 polynomial fallback for
        # glasses whose coefficients have been ingested into
        # POLYNOMIAL_COEFFICIENTS.  Empty at v4.16.2 ship; populating
        # the 24 catalogue entries is staged for v5.2.1 (per-glass
        # vendor-source review against refractiveindex.info YAML +
        # 5e-5 n_d cross-check).
        if glass_name in POLYNOMIAL_COEFFICIENTS:
            return _cached_glass_value(
                glass_name, wavelength,
                lambda: _polynomial_index(
                    wavelength, POLYNOMIAL_COEFFICIENTS[glass_name],
                    glass_name=glass_name))
        # v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
        # known-but-stubbed formula-3 glass.  Raise NotImplementedError
        # (not ImportError) so callers can distinguish "package missing"
        # (recoverable: install [glass]) from "glass coefficients not
        # yet ingested in the bundle" (also recoverable but via a
        # different remediation path: open a v5.2.1 issue or contribute
        # the ingestion).
        if glass_name in _POLYNOMIAL_STUB_NAMES:
            raise NotImplementedError(
                f"Glass {glass_name!r} is a known formula-3 (polynomial) "
                f"catalogue entry but its coefficients have not yet been "
                f"ingested into the bundled POLYNOMIAL_COEFFICIENTS "
                f"table (deferred to v5.2.1).  Two remediations: "
                f"(1) install the optional ``refractiveindex`` package "
                f"(``pip install lumenairy[glass]``) for live "
                f"refractiveindex.info lookup; or (2) open a v5.2.1 "
                f"issue requesting coefficient ingestion for "
                f"{glass_name!r} -- include the catalogue source "
                f"(CDGM 2022-06 / HIKARI 2017-11 / SUMITA 2017-02) and "
                f"the wavelength range you need.  See "
                f"lumenairy.glass._POLYNOMIAL_STUB_NAMES for the full "
                f"manifest of glasses awaiting ingestion.")
        raise ImportError(
            f"Glass {glass_name!r} requires the 'refractiveindex' "
            f"package for live lookup, but it is not installed.  "
            f"Either install it (`pip install refractiveindex` or "
            f"`pip install lumenairy[glass]`) or register a callable / "
            f"Sellmeier coefficients for this glass in GLASS_REGISTRY / "
            f"SELLMEIER_COEFFICIENTS.")
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

    # Sellmeier / polynomial / sentinel paths have no extinction data;
    # return kappa = 0 explicitly.  v4.16.3 (audit P3-NEW-F1-1):
    # __polynomial__ added parallel to __sellmeier__.
    if entry in ('__sellmeier__', '__polynomial__', '__thin_lens__'):
        return complex(get_glass_index(glass_name, wavelength), 0.0)

    # Tuple-style path: try refractiveindex.info for both n and kappa.
    n_real = get_glass_index(glass_name, wavelength)

    # Extinction is optional -- many catalog entries (esp. SCHOTT specs) omit it.
    # 4.10: warn once per (glass, wavelength) when kappa is unavailable,
    # so callers that requested complex-n (e.g. `absorption=True`) know
    # they're getting the lossless approximation.
    try:
        kappa = _glass_cache[glass_name].get_extinction_coefficient(
            wavelength * 1e9, unit='nm')
        if kappa is None:
            _warn_missing_kappa_once(glass_name, wavelength)
            kappa = 0.0
    except (AttributeError, NotImplementedError, KeyError, ValueError, TypeError):
        _warn_missing_kappa_once(glass_name, wavelength)
        kappa = 0.0

    return complex(n_real, float(kappa))


_kappa_warned: set = set()


def _warn_missing_kappa_once(glass_name, wavelength):
    """4.10 helper: emit a one-time warning per (glass, wavelength) when
    extinction data is unavailable, so absorbing-glass requests don't
    silently fall back to the lossless approximation."""
    key = (str(glass_name), round(float(wavelength) * 1e9, 1))
    if key in _kappa_warned:
        return
    _kappa_warned.add(key)
    import warnings
    warnings.warn(
        # GL-1 (AUDIT_GLASS_POLARIZATION): the working remediation is a
        # COMPLEX-returning callable (honoured by get_glass_index_complex);
        # register_fixed_glass(name, n) stores a real-index-only _FixedIndex
        # shim with no extinction, so it would land the user right back here.
        f"glass {glass_name!r}: no extinction coefficient available at "
        f"{wavelength * 1e9:.1f} nm; treating as lossless (kappa = 0). "
        f"For absorbing-glass simulations use a catalogue entry with "
        f"complex-n data, or register a complex-returning callable, e.g. "
        f"``GLASS_REGISTRY[{glass_name!r}] = lambda wl_m: n + 1j*kappa`` "
        f"(get_glass_index_complex reads the imaginary part).",
        RuntimeWarning, stacklevel=3,
    )


# ---------------------------------------------------------------------------
# v5.17.1 (audit P3-40): central-registry enrollment for this module's
# four caches.  Pre-v5.17.1 ``clear_asm_caches`` /
# ``lumenairy_context(clear_caches_on_exit=True)`` never drained them
# (the enrollment meta-pin's case-sensitive ``endswith('_CACHE')``
# filter missed the lower-case names -- audit P2-42).
# ---------------------------------------------------------------------------

_USER_FIXED_SENTINEL = ('__user__', '__fixed__', '__fixed__')


def _clear_glass_caches() -> None:
    """Drain the glass module's caches (registered as ``'glass_caches'``).

    Drops:

    * every ``_glass_value_cache`` entry (recomputable pure math),
    * both warn-once sets (``_validity_warned`` / ``_kappa_warned`` --
      cleared warnings simply re-fire once),
    * every RE-LOADABLE ``_glass_cache`` entry, i.e. entries whose
      ``GLASS_REGISTRY`` dispatch is a catalogue ``(shelf, book, page)``
      tuple: the next lookup rebuilds the ``RefractiveIndexMaterial``
      from the registry (this is where the YAML dispersion databases --
      the actual memory -- live).

    PRESERVED: user-fixed entries (registry dispatch is the
    ``('__user__', '__fixed__', '__fixed__')`` sentinel, e.g.
    ``register_fixed_glass`` names, ``'__thin_lens__'`` and the
    ``'__aspheric_*'`` entries added by ``raytrace.trace``).  For those
    names ``_glass_cache`` is the AUTHORITATIVE value store, not a
    cache -- dropping them would make the very next
    ``get_glass_index`` raise ``ValueError`` ("flagged as user-fixed
    but has no _glass_cache entry").  They are a handful of tiny
    ``_FixedIndex`` objects, so retaining them costs nothing.
    """
    with _GLASS_CACHE_LOCK:
        _glass_value_cache.clear()
        _validity_warned.clear()
        _kappa_warned.clear()
        for _name in [n for n in _glass_cache
                      if GLASS_REGISTRY.get(n) != _USER_FIXED_SENTINEL]:
            del _glass_cache[_name]


# Canonical v4.16.0 enrollment idiom (late-binding lambda so
# ``mock.patch.object`` on ``_clear_glass_caches`` is honoured).
try:
    import sys as _sys

    from ._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'glass_caches',
        lambda: getattr(_this_mod, '_clear_glass_caches')(),
    )
except ImportError:
    pass
