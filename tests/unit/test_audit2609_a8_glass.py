"""WP-A8 regression pins for the glass-catalogue findings of
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`` section 5.

* **E1 (P0)** three bundled Sellmeier rows held a different glass's
  dispersion (N-BAF52, N-LAK33A, N-LAK33B), and the ``'__sellmeier__'``
  dispatch made them the ONLY path for those names.  Pinned here two ways:
  the repaired rows must reproduce the manufacturer data sheet's ``n_d`` /
  ``V_d``, and ``glass._cross_check_bundled_values`` must flag the pre-fix
  coefficients when they are injected back through the public table.
* **E2 (P1)** ``get_glass_index_complex`` raised
  ``refractiveindex...NoExtinctionCoefficient`` instead of the documented
  ``kappa = 0`` fallback for every ``main``-shelf window material.
* **E7 (P3)** the ``'air'`` short-circuit made a registered ambient model
  unreachable; ``'vacuum'`` / ``'__MIRROR__'`` were documented as registry
  entries they never were.

Oracles used
------------
* The SCHOTT data-sheet ``n_d`` / ``V_d`` pairs below are the manufacturer's
  published numbers (SCHOTT optical-glass data sheets / 2017-01-20 catalogue,
  mirrored in the refractiveindex.info YAML ``PROPERTIES`` block and encoded
  a second time in each glass's 6-digit glass code).  They are independent of
  the dispersion coefficients under test.
* ``refractiveindex.info`` (database commit a66ef88) evaluated through the
  ``refractiveindex`` package, which implements the formula-2 / formula-3
  evaluators separately from this library's.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy.glass as G

# d / F / C lines [m].
LD, LF, LC = 587.5618e-9, 486.1327e-9, 656.2725e-9


def _nd_vd(name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nd = float(G.get_glass_index(name, LD))
        nf = float(G.get_glass_index(name, LF))
        nc = float(G.get_glass_index(name, LC))
    return nd, (nd - 1.0) / (nf - nc)


# ---------------------------------------------------------------------------
# E1 -- the three repaired rows, and the gate that keeps them repaired
# ---------------------------------------------------------------------------

# SCHOTT data-sheet values.  The last two columns are the pre-fix numbers the
# bundled rows produced, quoted so the size of the defect is visible next to
# the bar: N-BAF52 was a whole different glass (nearest catalogue neighbour
# N-KZFS11, n_d 1.63775 / V_d 42.41), N-LAK33A/B had the wrong third
# Sellmeier pole (C3 = 107.1 / 101.7 um^2 instead of 80.94 / 80.74).
_DATASHEET = {
    #  name         n_d        V_d      pre-fix n_d   pre-fix V_d
    'N-BAF52':   (1.60863,   46.60,     1.637147,     42.469),
    'N-LAK33A':  (1.75393,   52.27,     1.754279,     53.031),
    'N-LAK33B':  (1.75500,   52.30,     1.755294,     52.940),
    # Audit-verified-correct controls that must not move.
    'N-BK7':     (1.51680,   64.17,     1.516800,     64.167),
    'N-SF11':    (1.78472,   25.68,     1.784720,     25.680),
}

# Bars.  The data sheet quotes n_d to 5 decimals and V_d to 2, so its own
# rounding floor is 5e-6 and 5e-3.  Measured deviation of the repaired rows
# from the data sheet (2026-09-12): n_d +1.01e-6 / +8.7e-8 / +1.65e-7,
# V_d -0.003 / +0.001 / +0.000 -- i.e. at the quote floor.  The bars sit 4x
# and 10x above that floor, and 1400x / 83x BELOW the smallest thing they
# must catch (N-LAK33A's pre-fix n_d error 3.49e-4 is 17x the n_d bar; its
# V_d error 0.76 is 15x the V_d bar; N-BAF52's 2.85e-2 / 4.13 are 1400x /
# 83x).  Decades of gap on both sides.
_ND_TOL = 2e-5
_VD_TOL = 0.05


@pytest.mark.parametrize('name', sorted(_DATASHEET))
def test_e1_bundled_row_reproduces_the_manufacturer_data_sheet(name):
    nd_pub, vd_pub, nd_pre, vd_pre = _DATASHEET[name]
    nd, vd = _nd_vd(name)
    assert abs(nd - nd_pub) <= _ND_TOL, (
        f"{name}: n_d = {nd:.6f}, data sheet {nd_pub} "
        f"(delta {nd - nd_pub:+.3e}, bar {_ND_TOL:.1e}); the pre-fix row "
        f"gave {nd_pre}")
    assert abs(vd - vd_pub) <= _VD_TOL, (
        f"{name}: V_d = {vd:.3f}, data sheet {vd_pub} "
        f"(delta {vd - vd_pub:+.3f}, bar {_VD_TOL}); the pre-fix row gave "
        f"{vd_pre}")


@pytest.mark.parametrize('name', ['N-BAF52', 'N-LAK33A', 'N-LAK33B'])
def test_e1_repaired_rows_are_not_the_pre_fix_rows(name):
    """Fail-before arm: the pre-fix numbers are a hard non-match.

    Each pre-fix n_d is more than 10x the bar away from the data sheet, so
    this assertion is the same statement as the one above read from the
    other side -- it exists so a reviewer can see the defect size without
    re-deriving it.
    """
    nd_pub, _, nd_pre, _ = _DATASHEET[name]
    nd, _ = _nd_vd(name)
    assert abs(nd - nd_pre) > 10 * _ND_TOL, (
        f"{name}: n_d = {nd:.6f} is still the pre-fix value {nd_pre} "
        f"(data sheet {nd_pub})")


def test_e1_whole_bundled_table_agrees_with_refractiveindex_info():
    """Every bundled Sellmeier AND polynomial row vs its catalogue source.

    This is the gate that would have caught E1.  Measured 2026-09-12 over
    76 resolvable rows: worst |dn_d| 4.18e-6 and worst |dV_d|/V_d 1.13e-4
    (both N-LASF40, whose catalogue page carries a marginally different
    fit); 46 of 49 Sellmeier rows and all 24 polynomial rows are exact to
    <= 2.2e-16.  The bars inside ``_cross_check_bundled_values`` (5e-5 /
    1e-3) therefore sit ~1 decade above that floor and ~1 decade below the
    weakest real defect (N-LAK33A, 2.9e-4 / 1.2e-2).
    """
    n_checked, problems = G._cross_check_bundled_values()
    assert problems == [], "\n".join(problems)
    if G._REFRACTIVEINDEX_AVAILABLE:
        # 49 Sellmeier + 24 polynomial rows resolve here; the bar is the
        # count below which the check has quietly stopped covering the
        # table (e.g. a renamed catalogue book resolving nothing).
        assert n_checked >= 70, (
            f"only {n_checked} bundled rows resolved a catalogue page; the "
            f"value cross-check has gone blind")
    else:
        assert n_checked == 0, (
            f"refractiveindex is absent but {n_checked} rows claim to have "
            f"resolved a catalogue page")


def test_e1_value_gate_rejects_the_pre_fix_rows():
    """Fail-before demonstration, engineered through the public table.

    Injects the exact pre-fix coefficients and asserts the value gate
    raises and names all three glasses.  Restores the table afterwards.
    """
    pre_fix = {
        'N-BAF52': ((1.43903433, 0.179827671, 1.13174268),
                    (9.07800726e-3, 4.39222348e-2, 1.06317650e2)),
        'N-LAK33A': ((1.44116999, 0.571749501, 1.16605226),
                     (6.80933877e-3, 2.22291824e-2, 1.07097324e2)),
        'N-LAK33B': ((1.42288601, 0.593661336, 1.16135260),
                     (6.70283452e-3, 2.19416210e-2, 1.01736644e2)),
    }
    if not G._REFRACTIVEINDEX_AVAILABLE:
        n_checked, _ = G._cross_check_bundled_values()
        assert n_checked == 0
        pytest.fail(
            "refractiveindex is not installed, so the value gate cannot "
            "run here; install lumenairy[glass] to exercise this pin")
    saved = {k: G.SELLMEIER_COEFFICIENTS[k] for k in pre_fix}
    try:
        G.SELLMEIER_COEFFICIENTS.update(pre_fix)
        G._clear_glass_caches()
        _, problems = G._cross_check_bundled_values()
        flagged = {p.split(':', 1)[0] for p in problems}
        assert flagged == set(pre_fix), (
            f"value gate flagged {sorted(flagged)}, expected "
            f"{sorted(pre_fix)}")
        with pytest.raises(RuntimeError, match='N-BAF52'):
            G._check_glass_registry_consistency(check_values=True)
    finally:
        G.SELLMEIER_COEFFICIENTS.update(saved)
        G._clear_glass_caches()
    # And the restored table is clean again.
    assert G._cross_check_bundled_values()[1] == []


def test_e1_import_time_check_stays_structural_only():
    """The value check must be opt-in: it parses one catalogue YAML per row,
    which has no place in ``import lumenairy``."""
    import inspect
    sig = inspect.signature(G._check_glass_registry_consistency)
    assert sig.parameters['check_values'].default is False
    src = inspect.getsource(G)
    assert '\n_check_glass_registry_consistency()\n' in src, (
        'the module-level call must remain the no-argument (structural) one')


# ---------------------------------------------------------------------------
# E2 -- the kappa = 0 fallback the docstring promises
# ---------------------------------------------------------------------------

def _tuple_registered_names():
    return sorted(
        name for name, entry in G.GLASS_REGISTRY.items()
        if isinstance(entry, tuple) and entry[0] != '__user__')


@pytest.mark.parametrize('wavelength', [1.31e-6, 1.55e-6])
def test_e2_complex_index_never_raises_for_a_catalogue_glass(wavelength):
    """Pre-fix, 7 of 44 tuple-registered glasses raised
    ``NoExtinctionCoefficient`` at 1.31 um -- CaF2, MgF2, SILICON and the
    four fused-silica aliases, i.e. every ``main``-shelf window material.
    The class subclasses ``Exception`` directly, so it was not in the
    caught tuple.  The documented contract is a warn-once plus kappa = 0.
    """
    raised = []
    for name in _tuple_registered_names():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                G.get_glass_index_complex(name, wavelength)
            except Exception as exc:            # noqa: BLE001 - that is the pin
                raised.append((name, type(exc).__name__))
    assert raised == [], (
        f"get_glass_index_complex raised for {raised} instead of falling "
        f"back to kappa = 0")


@pytest.mark.parametrize('wavelength', [1.31e-6, 1.55e-6])
def test_e2_extinction_is_finite_and_non_negative(wavelength):
    """A page whose tabulated k does not span the wavelength interpolates to
    NaN rather than raising, so ``kappa is None`` was not the only "no data"
    signal: E-BAK1 and E-LAK04 returned ``n + nan*j`` at 1.31 um, which
    poisons every downstream absorption product silently.
    """
    bad = []
    for name in _tuple_registered_names():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nc = G.get_glass_index_complex(name, wavelength)
        if not (np.isfinite(nc.real) and np.isfinite(nc.imag)
                and nc.imag >= 0.0):
            bad.append((name, nc))
    assert bad == [], f"non-finite or negative kappa: {bad}"


@pytest.mark.parametrize('name', ['CaF2', 'FUSED_SILICA', 'MgF2', 'SILICON'])
def test_e2_missing_kappa_warns_once_and_returns_zero(name):
    G._kappa_warned.clear()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        nc = G.get_glass_index_complex(name, 1.31e-6)
        again = G.get_glass_index_complex(name, 1.31e-6)
    assert nc.imag == 0.0 and again == nc
    kinds = [w for w in rec if issubclass(w.category, RuntimeWarning)
             and 'extinction' in str(w.message)]
    assert len(kinds) == 1, (
        f"expected exactly one warn-once for {name}, got {len(kinds)}")


def test_e2_pages_that_do_carry_k_keep_their_value_and_sign():
    """Guard against "fix by swallowing": N-BK7 has real k data and its
    convention (kappa > 0 = absorbing under exp(-i w t)) is audit-verified.
    The reference value 1.4361e-7 at 1.55 um is the catalogue's own
    tabulated-k interpolation, re-measured 2026-09-12; the bar is 1e-4
    relative, a decade above float noise and decades below any sign flip or
    silent zeroing.
    """
    nc = G.get_glass_index_complex('N-BK7', 1.55e-6)
    assert abs(nc.real - 1.5006520) < 1e-6
    assert abs(nc.imag / 1.4361318e-7 - 1.0) < 1e-4, nc


# ---------------------------------------------------------------------------
# Bundled fallback vs catalogue dispatch (surfaced by the E1 cross-check)
# ---------------------------------------------------------------------------

def test_caf2_bundled_fallback_vs_catalogue_dispatch_deviation_is_bounded():
    """``GLASS_REGISTRY['CaF2']`` dispatches to ``main/CaF2/Daimon-20`` when
    ``refractiveindex`` is installed, while the bundled fallback row is
    Malitson's fit -- both legitimate published CaF2 fits, but the returned
    index then depends on which optional packages are present.

    Measured 2026-09-12: n_d 1.4338769 (Daimon-20) vs 1.4338493 (bundled),
    a 2.76e-5 difference, and 3.78e-5 maximum over 0.4-1.6 um.  The bar is
    1e-4, ~4x the measured deviation and ~2 decades below a
    different-material error; it exists so the gap cannot grow unnoticed
    while the two fits stay deliberately different.  Every OTHER bundled row
    with a catalogue tuple agrees with its dispatch to <= 4.2e-6.
    """
    if not G._REFRACTIVEINDEX_AVAILABLE:
        pytest.fail('refractiveindex is required to exercise this pin')
    worst = {}
    wl = np.linspace(0.45e-6, 1.55e-6, 23)
    for name, coeffs in G.SELLMEIER_COEFFICIENTS.items():
        entry = G.GLASS_REGISTRY.get(name)
        if not (isinstance(entry, tuple) and entry[0] != '__user__'):
            continue
        cat = G._catalogue_index_fn_from_entry(entry)
        if cat is None:
            continue
        bundled = np.asarray(G._sellmeier_index(wl, coeffs), dtype=float)
        dispatch = np.array([cat(w) for w in wl])
        worst[name] = float(np.max(np.abs(bundled - dispatch)))
    assert worst, 'no tuple-registered bundled rows resolved'
    assert worst['CaF2'] < 1e-4, worst['CaF2']
    others = {k: v for k, v in worst.items() if k != 'CaF2'}
    assert max(others.values()) < 1e-5, (
        f"a bundled fallback has drifted from the page it falls back for: "
        f"{sorted(others.items(), key=lambda kv: -kv[1])[:3]}")


def test_baf2_row_is_the_malitson_fit_its_comment_now_names():
    """The row's comment used to credit Li 1980; the coefficients are
    Malitson & Dodge 1972, which the two fits disagree about by 8.1e-5 in
    n_d.  Bar 1e-9: the row reproduces main/BaF2/Malitson to 2.2e-16, and
    the wrong attribution is 5 decades above that.
    """
    if not G._REFRACTIVEINDEX_AVAILABLE:
        pytest.fail('refractiveindex is required to exercise this pin')
    cat = G._catalogue_index_fn_from_entry(('main', 'BaF2', 'Malitson'))
    assert cat is not None
    nd_row = float(G._sellmeier_index(LD, G.SELLMEIER_COEFFICIENTS['BaF2']))
    assert abs(nd_row - cat(LD)) < 1e-9, (nd_row, cat(LD))


# ---------------------------------------------------------------------------
# E7 -- 'air' / 'vacuum' / '__MIRROR__'
# ---------------------------------------------------------------------------

def test_e7_air_defaults_to_one_for_every_spelling():
    for name in ('air', 'AIR', 'Air'):
        assert G.get_glass_index(name, 1.064e-6) == 1.0
        assert G.get_glass_index_complex(name, 1.064e-6) == 1.0 + 0.0j


def test_e7_registered_air_callable_is_honoured():
    """Pre-fix the ``'air'`` short-circuit returned 1.0 before consulting the
    registry, so the Edlen ambient model the exemption comment described was
    unimplementable: 273 um of OPD per metre of air path (~256 waves at
    1.064 um) silently absent.  The bar below just checks the registered
    callable's value comes back; the Edlen value itself is the oracle.
    """
    def edlen(wavelength_m):
        s = 1.0 / (wavelength_m * 1e6)
        return 1 + 1e-8 * (8342.54 + 2406147 / (130 - s * s)
                           + 15998 / (38.9 - s * s))

    assert 'air' not in G.GLASS_REGISTRY, 'air must ship unregistered'
    G.GLASS_REGISTRY['air'] = edlen
    try:
        want = edlen(1.064e-6)
        for name in ('air', 'AIR', 'Air'):
            assert G.get_glass_index(name, 1.064e-6) == want
            assert G.get_glass_index_complex(name, 1.064e-6) == complex(want)
        # ~274 um/m of OPD -- the quantity the short-circuit was discarding.
        assert 2.7e-4 < want - 1.0 < 2.8e-4
    finally:
        del G.GLASS_REGISTRY['air']
    assert G.get_glass_index('air', 1.064e-6) == 1.0


@pytest.mark.parametrize('name', ['vacuum', 'MIRROR', '__MIRROR__'])
def test_e7_documented_non_entries_really_are_absent(name):
    """The exemptions comment described ``'vacuum'`` and ``'__MIRROR__'`` as
    registry entries.  They are not, and the comment now says so."""
    assert name not in G.GLASS_REGISTRY
    with pytest.raises(ValueError):
        G.get_glass_index(name, 1.0e-6)


def test_e7_exemption_list_documents_only_legal_names():
    """Every name exempted from the GLASS_VALIDITY -> GLASS_REGISTRY check is
    either a real registry entry or one of the three documented non-entries;
    nothing else may be quietly excused."""
    allowed_non_entries = {'air', 'vacuum', '__MIRROR__'}
    for name in G._GLASS_VALIDITY_REGISTRY_EXEMPTIONS:
        assert name in G.GLASS_REGISTRY or name in allowed_non_entries, name
