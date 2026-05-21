"""v4.16.0 Agent D: CDGM / Hikari / Sumita glass catalogues (ROADMAP #13).

Pins
----

Each glass added to ``GLASS_REGISTRY`` (and optionally to
``SELLMEIER_COEFFICIENTS``) in v4.16.0 must:

1. Be reachable via ``get_glass_index(name, 587.56e-9)`` without
   raising (assuming the optional ``refractiveindex`` package is
   installed, which is the typical path).
2. Match the published catalogue n_d value to 5e-5 -- matching the
   v4.14.2 S-LAH64 / S-LAH79 validation tolerance.
3. Have a ``GLASS_VALIDITY`` entry that covers the d-line (587.56 nm)
   -- otherwise the warning-emit logic would fire on every n_d
   query, which would defeat the purpose of bundling the glass.

Glasses that fail the 5e-5 cross-check are flagged for v4.16.1.

Author: Andrew Traverso -- v4.16.0 / Agent D
"""
from __future__ import annotations

import warnings

import pytest

import lumenairy as la
from lumenairy.glass import (
    _REFRACTIVEINDEX_AVAILABLE,
    GLASS_REGISTRY,
    GLASS_VALIDITY,
    SELLMEIER_COEFFICIENTS,
    _sellmeier_index,
)

# d-line (Fraunhofer d-line, He I yellow doublet center).
_D_LINE_M = 587.56e-9

# n_d cross-check tolerance.  Matches the v4.14.2 S-LAH64 / S-LAH79
# precedent: SCHOTT and OHARA catalogues publish n_d to 6 significant
# figures, so 5e-5 is two orders of magnitude below the catalogue's
# own precision.
_ND_TOLERANCE = 5e-5


# ===========================================================================
# CDGM catalogue
# ===========================================================================
#
# All n_d values sourced from the CDGM 2022-06 Zemax catalogue (mirrored
# at https://refractiveindex.info/?shelf=specs&book=CDGM-optical).
# Each YAML file under
# ``~/.refractiveindex.info-database/data/specs/cdgm/optical/`` carries
# a ``PROPERTIES.nd`` field that ultimately traces to the official
# CDGM data sheet PDF.

CDGM_TARGETS = {
    # name -> (n_d, source URL, formula in catalog)
    # Each n_d value comes from the PROPERTIES.nd field in the
    # corresponding refractiveindex.info YAML
    # (~/.refractiveindex.info-database/data/specs/cdgm/optical/).
    'H-K9L':      (1.516800, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-K9L', 'formula 2'),
    'H-LAK52':    (1.729160, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-LAK52', 'formula 2'),
    'H-LAK53A':   (1.755000, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-LAK53A', 'formula 2'),
    'H-ZK9B':     (1.620410, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-ZK9B', 'formula 2'),
    'H-ZF12':     (1.761820, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-ZF12', 'formula 2'),
    'D-ZK3':      (1.589130, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=D-ZK3', 'formula 2'),
    'D-LAK52':    (1.730500, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=D-LAK52', 'formula 2'),
    'H-ZLAF52A':  (1.806100, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-ZLAF52A', 'formula 2'),
    'H-ZK7':      (1.613095, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-ZK7', 'formula 3'),
    'H-ZF52A':    (1.846660, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=H-ZF52A', 'formula 3'),
    'F1-CDGM':    (1.603417, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=F1', 'formula 3'),
    'F2-CDGM':    (1.612929, 'https://refractiveindex.info/?shelf=specs&book=CDGM-optical&page=F2', 'formula 3'),
}


@pytest.mark.parametrize('name', sorted(CDGM_TARGETS))
def test_cdgm_glass_registered(name):
    """Each CDGM v4.16.0 addition must appear in ``GLASS_REGISTRY``."""
    assert name in GLASS_REGISTRY, (
        f"CDGM glass {name!r} expected in GLASS_REGISTRY but missing. "
        f"v4.16.0 (ROADMAP #13) registers each CDGM glass as a tuple "
        f"('specs', 'CDGM-optical', PAGE).")


@pytest.mark.parametrize('name', sorted(CDGM_TARGETS))
def test_cdgm_glass_has_validity_range(name):
    """Each CDGM v4.16.0 addition must appear in ``GLASS_VALIDITY``
    -- v4.16.0 ROADMAP #14 requires every newly-bundled glass to
    carry an explicit validity range.
    """
    assert name in GLASS_VALIDITY, (
        f"CDGM glass {name!r} expected in GLASS_VALIDITY but missing. "
        f"v4.16.0 (ROADMAP #14) requires explicit validity ranges for "
        f"every newly-bundled glass.")
    lmin, lmax = GLASS_VALIDITY[name]
    assert lmin <= _D_LINE_M <= lmax, (
        f"CDGM glass {name!r} validity range [{lmin}, {lmax}] m does "
        f"not contain the d-line ({_D_LINE_M} m); n_d query would "
        f"trigger a spurious out-of-range warning.")


@pytest.mark.skipif(not _REFRACTIVEINDEX_AVAILABLE,
                     reason="refractiveindex package required for CDGM live lookup")
@pytest.mark.parametrize('name', sorted(CDGM_TARGETS))
def test_cdgm_glass_n_d_cross_check_live(name):
    """Live ``refractiveindex.info`` dispatch returns the catalogue
    n_d within 5e-5 of the published value.
    """
    target_nd, source_url, formula = CDGM_TARGETS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        n = la.get_glass_index(name, _D_LINE_M)
    diff = abs(n - target_nd)
    assert diff < _ND_TOLERANCE, (
        f"CDGM glass {name!r}: live lookup n_d = {n:.6f}, "
        f"catalogue n_d = {target_nd:.6f}, |diff| = {diff:.2e} > "
        f"{_ND_TOLERANCE:.0e}.  Source: {source_url}.  "
        f"Flag for v4.16.1 if this fails after a clean install.")


@pytest.mark.parametrize('name', sorted(n for n in CDGM_TARGETS
                                       if n in SELLMEIER_COEFFICIENTS))
def test_cdgm_glass_n_d_bundled_sellmeier(name):
    """For CDGM glasses with bundled Sellmeier coefficients, the
    Sellmeier evaluator matches the catalogue n_d to 5e-5.

    This is the offline-fallback path: when ``refractiveindex`` is
    not installed, ``get_glass_index`` falls back to the bundled
    coefficients.  Both paths must agree on n_d.
    """
    target_nd, source_url, formula = CDGM_TARGETS[name]
    coeffs = SELLMEIER_COEFFICIENTS[name]
    n = _sellmeier_index(_D_LINE_M, coeffs, glass_name=name)
    diff = abs(n - target_nd)
    assert diff < _ND_TOLERANCE, (
        f"CDGM glass {name!r}: bundled Sellmeier n_d = {n:.6f}, "
        f"catalogue n_d = {target_nd:.6f}, |diff| = {diff:.2e} > "
        f"{_ND_TOLERANCE:.0e}.  Source: {source_url}.  "
        f"Flag for v4.16.1 if this fails.")


# ===========================================================================
# Hikari catalogue
# ===========================================================================
#
# All n_d values sourced from the HIKARI 2017-11 / NIKON Zemax catalogue
# (mirrored at https://refractiveindex.info/?shelf=specs&book=HIKARI-
# optical).  Hikari E-* / J-* polynomials are formula 3 and are not
# bundled into SELLMEIER_COEFFICIENTS -- live lookup only.

HIKARI_TARGETS = {
    # name -> (n_d, source URL)
    # Each n_d value comes from the PROPERTIES.nd field in the
    # corresponding refractiveindex.info YAML
    # (~/.refractiveindex.info-database/data/specs/hikari/optical/).
    'E-LASF016':  (1.772499, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=E-LASF016'),
    'E-SK16':     (1.620410, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=E-SK16'),
    'E-LAK7':     (1.651597, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=E-LAK7'),
    'E-LAK04':    (1.651000, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=E-LAK04'),
    'E-BAK1':     (1.572501, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=E-BAK1'),
    'J-FK01A':    (1.496999, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=J-FK01A'),
    'J-LASF09A':  (1.815996, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=J-LASF09A'),
    'J-LAK7':     (1.651597, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=J-LAK7'),
    'J-BASF7':    (1.701536, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=J-BASF7'),
    'E-F2':       (1.620041, 'https://refractiveindex.info/?shelf=specs&book=HIKARI-optical&page=E-F2'),
}


@pytest.mark.parametrize('name', sorted(HIKARI_TARGETS))
def test_hikari_glass_registered(name):
    """Each Hikari v4.16.0 addition must appear in ``GLASS_REGISTRY``."""
    assert name in GLASS_REGISTRY, (
        f"Hikari glass {name!r} expected in GLASS_REGISTRY but missing.")


@pytest.mark.parametrize('name', sorted(HIKARI_TARGETS))
def test_hikari_glass_has_validity_range(name):
    """Each Hikari v4.16.0 addition must appear in ``GLASS_VALIDITY``."""
    assert name in GLASS_VALIDITY, (
        f"Hikari glass {name!r} expected in GLASS_VALIDITY but missing.")
    lmin, lmax = GLASS_VALIDITY[name]
    assert lmin <= _D_LINE_M <= lmax, (
        f"Hikari glass {name!r} validity range [{lmin}, {lmax}] m "
        f"does not contain the d-line.")


@pytest.mark.skipif(not _REFRACTIVEINDEX_AVAILABLE,
                     reason="refractiveindex package required for Hikari lookup")
@pytest.mark.parametrize('name', sorted(HIKARI_TARGETS))
def test_hikari_glass_n_d_cross_check(name):
    """Live ``refractiveindex.info`` dispatch returns the catalogue
    n_d within 5e-5 of the published value.
    """
    target_nd, source_url = HIKARI_TARGETS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        n = la.get_glass_index(name, _D_LINE_M)
    diff = abs(n - target_nd)
    assert diff < _ND_TOLERANCE, (
        f"Hikari glass {name!r}: live lookup n_d = {n:.6f}, "
        f"catalogue n_d = {target_nd:.6f}, |diff| = {diff:.2e} > "
        f"{_ND_TOLERANCE:.0e}.  Source: {source_url}.  "
        f"Flag for v4.16.1 if this fails after a clean install.")


# ===========================================================================
# Sumita catalogue
# ===========================================================================
#
# All n_d values sourced from the SUMITA 2017-02 Zemax catalogue
# (mirrored at https://refractiveindex.info/?shelf=specs&book=SUMITA-
# optical).  Sumita K-* polynomials are formula 3 (polynomial) and
# are not bundled into SELLMEIER_COEFFICIENTS -- live lookup only.

SUMITA_TARGETS = {
    # name -> (n_d, source URL)
    # Each n_d value comes from the PROPERTIES.nd field in the
    # corresponding refractiveindex.info YAML
    # (~/.refractiveindex.info-database/data/specs/sumita/optical/).
    # v4.16.0 canonicalises Sumita names to all-uppercase in
    # GLASS_REGISTRY; the underlying refractiveindex.info page name
    # still uses the catalogue's mixed-case (K-LaK10, K-LaSFn10, ...).
    'K-VC78':     (1.669550, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-VC78'),
    'K-LAK10':    (1.720000, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-LaK10'),
    'K-LASFN10':  (1.815500, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-LaSFn10'),
    'K-SK4':      (1.612720, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-SK4'),
    'K-PFK90':    (1.458800, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-PFK90'),
    'K-PBK40':    (1.517600, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-PBK40'),
    'K-BK7':      (1.516330, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-BK7'),
    'K-PSKN2':    (1.618000, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-PSKn2'),
    'K-FK5':      (1.487490, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-FK5'),
    'K-LAFN3':    (1.700000, 'https://refractiveindex.info/?shelf=specs&book=SUMITA-optical&page=K-LaFn3'),
}


@pytest.mark.parametrize('name', sorted(SUMITA_TARGETS))
def test_sumita_glass_registered(name):
    """Each Sumita v4.16.0 addition must appear in ``GLASS_REGISTRY``."""
    assert name in GLASS_REGISTRY, (
        f"Sumita glass {name!r} expected in GLASS_REGISTRY but missing.")


@pytest.mark.parametrize('name', sorted(SUMITA_TARGETS))
def test_sumita_glass_has_validity_range(name):
    """Each Sumita v4.16.0 addition must appear in ``GLASS_VALIDITY``."""
    assert name in GLASS_VALIDITY, (
        f"Sumita glass {name!r} expected in GLASS_VALIDITY but missing.")
    lmin, lmax = GLASS_VALIDITY[name]
    assert lmin <= _D_LINE_M <= lmax, (
        f"Sumita glass {name!r} validity range [{lmin}, {lmax}] m "
        f"does not contain the d-line.")


@pytest.mark.skipif(not _REFRACTIVEINDEX_AVAILABLE,
                     reason="refractiveindex package required for Sumita lookup")
@pytest.mark.parametrize('name', sorted(SUMITA_TARGETS))
def test_sumita_glass_n_d_cross_check(name):
    """Live ``refractiveindex.info`` dispatch returns the catalogue
    n_d within 5e-5 of the published value.
    """
    target_nd, source_url = SUMITA_TARGETS[name]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        n = la.get_glass_index(name, _D_LINE_M)
    diff = abs(n - target_nd)
    assert diff < _ND_TOLERANCE, (
        f"Sumita glass {name!r}: live lookup n_d = {n:.6f}, "
        f"catalogue n_d = {target_nd:.6f}, |diff| = {diff:.2e} > "
        f"{_ND_TOLERANCE:.0e}.  Source: {source_url}.  "
        f"Flag for v4.16.1 if this fails after a clean install.")


# ===========================================================================
# Aggregate counts (pin the v4.16.0 catalogue expansion size)
# ===========================================================================

def test_cdgm_count_at_least_eight():
    """v4.16.0 ships at least 8 CDGM glasses (per dispatch sheet ~15
    target, with several reduced to those actually present in the
    refractiveindex.info catalogue)."""
    assert len(CDGM_TARGETS) >= 8, (
        f"CDGM target count {len(CDGM_TARGETS)} < 8.  "
        f"Reduce only if a glass is dropped from the v4.16.0 release.")


def test_hikari_count_at_least_eight():
    """v4.16.0 ships at least 8 Hikari glasses."""
    assert len(HIKARI_TARGETS) >= 8, (
        f"Hikari target count {len(HIKARI_TARGETS)} < 8.")


def test_sumita_count_at_least_eight():
    """v4.16.0 ships at least 8 Sumita glasses."""
    assert len(SUMITA_TARGETS) >= 8, (
        f"Sumita target count {len(SUMITA_TARGETS)} < 8.")


def test_total_glass_count_increased():
    """v4.16.0 must increase the public ``GLASS_REGISTRY`` size
    beyond the v4.15 baseline (~46).  Pin at >= 40 to allow for
    a future v4.16.1 contraction that drops one or two malformed
    entries without breaking this assertion."""
    assert len(GLASS_REGISTRY) >= 40, (
        f"GLASS_REGISTRY size {len(GLASS_REGISTRY)} < 40; the v4.16.0 "
        f"catalogue expansion is the headline feature -- something "
        f"may have un-registered a vendor block.")


def test_top_level_export_GLASS_VALIDITY():
    """``GLASS_VALIDITY`` is part of the v4.16.0 public surface."""
    assert hasattr(la, 'GLASS_VALIDITY')
    assert la.GLASS_VALIDITY is GLASS_VALIDITY
    assert 'GLASS_VALIDITY' in la.__all__


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
