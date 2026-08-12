"""Niche C14 (2026-08-03) -- the traced-lens ENCAPSULATION and GUARD-INTEGRITY
layer: the flag registry, the UNIT-C exit-support object, and the four exits of
the chain's dx self-check that used to pass silently.

WHAT THIS FILE IS FOR, in three sentences.  The registry tests make the layer
map CHECKABLE -- a renamed constant, a moved default or a dangling ``:data:``
reference fails here instead of being discovered by an audit six weeks later.
The UNIT-C tests pin that pulling three notions of "where the traced rays
landed" into one object moved no bit, and that the band the two old self-checks
were JOINTLY BLIND to is now watched.  The guard-integrity tests pin that a
self-check which cannot compare says so, and that the convergence flag reaches
EVERY qualifying call rather than the first one per caller line.

Sources: ``docs/audits/ARCH_TRACED_ENCAPSULATION_2026_08_03.md`` (S8 steps 1-3),
``docs/audits/P2_DIDNOTWARN_DIAGNOSIS_2026_08_03.md`` (S4.1, S4.2),
``docs/audits/RECON_PINS_POST_C8_2026_08_01.md`` (S7 item 1),
``docs/audits/C14_ENCAPSULATION_GUARDS_2026_08_03.md`` (this work).
"""
from __future__ import annotations

import importlib
import re
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_imap as IM
from lumenairy.elements import _lens_traced as LT
from lumenairy.elements import _traced_flags as TF
from lumenairy.propagators import carrier as CM

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL


# ===========================================================================
# 1.  The registry IS the layer map's manifest.
# ===========================================================================
def test_every_registered_identifier_exists_on_its_module():
    """THE CI CHECK the layer map asks for (ARCH S8 step 1).

    A registry that names a constant which no longer exists is worse than no
    registry: it reads as coverage.  This is also the check that would have
    caught the dangling ``CHAIN_EXACT_TILTED_REFERENCE`` cross-reference the
    architecture study found by hand."""
    missing = []
    for (mod_name, name) in sorted(TF.FLAGS):
        mod = importlib.import_module(mod_name)
        if not hasattr(mod, name):
            missing.append(f'{mod_name}.{name}')
    assert not missing, f'registered but absent from the module: {missing}'


def test_the_registry_covers_every_documented_era_switch():
    """The thirteen behaviour-changing switches the architecture study
    inventoried (S2.1), plus the C11/C12/C13 selector and solver flags that
    landed after it, must all be in the table.  A new switch that skips the
    registry is a switch with no discoverable fail-before."""
    required = {
        '_F3_GUARD_TILTAWARE_EXPLICIT_CARRIER', '_CARRIER_FIT_RADIUS_FRAC',
        '_FIT_RADIUS_BEAM_FACTOR_DEFAULT', '_FIT_DISC_OUTSIDE_WEIGHT_REL',
        '_DECENTRED_FIT_POLY_ORDER', '_DECENTRE_GATE_PIXELS',
        '_DECENTRE_GATE_W_FRAC', 'TILTED_CARRIER_EXACT_EIKONAL',
        'REMAP_STATIONARY_PHASE_LAUNCH', '_REMAP_RESID_EIKONAL_DEGREE',
        'REMAP_STATIONARY_PHASE_FIT_GUARD', 'REMAP_INVERSE_SUPPORT_BOUND',
        'RAY_DENSITY_HALO_CHECK', 'SPHERE_PARAB_CONVERSION_EXACT',
        'DECENTRED_FIT_ARBITER', 'DECENTRED_FIT_PREDICTOR',
        'LSTSQ_CONDITIONING_STEPDOWN',
    }
    have = {name for (_m, name) in TF.FLAGS}
    assert required <= have, f'not registered: {sorted(required - have)}'


def test_the_newest_era_reproduces_the_live_shipped_values():
    """THE TRAP THIS CLOSES is the campaign's own, measured three times: an
    intervention expressed relative to a default, evaluated after the default
    moved (``fc_production_taper.py``'s nine-minute "baseline" that was in fact
    the exact conversion).  If a shipped default is changed without updating
    the table, every era arm silently starts meaning something else -- so the
    newest era must BE the shipped configuration, by assertion."""
    bad = []
    for (mod_name, name), val in TF.resolve_era(TF.ERAS[-1]).items():
        live = getattr(importlib.import_module(mod_name), name)
        if not (live == val and type(live) is type(val)):
            bad.append(f'{name}: live {live!r} vs era {val!r}')
    assert not bad, (
        f'the {TF.ERAS[-1]} preset no longer matches the shipped defaults: '
        f'{bad}')


def _layer_map_rows():
    """Parse the layer map's switch table (S2) into
    ``{identifier: (module, shipped, fail_before)}``.

    The table is the MANIFEST: a pipe row whose second cell is a layer label
    and whose third is a back-ticked identifier."""
    import pathlib
    doc = (pathlib.Path(__file__).resolve().parents[2]
           / 'docs' / 'audits' / 'TRACED_LAYER_MAP.md')
    rows = {}
    for line in doc.read_text(encoding='utf-8').splitlines():
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 7 or not cells[0].isdigit():
            continue
        ident = cells[2].strip('`')
        if not re.fullmatch(r'_?[A-Z][A-Z0-9_]*', ident):
            continue
        rows[ident] = (cells[3].strip('`'), cells[4].strip('`'),
                       cells[5].strip('`'))
    return rows


def test_the_layer_map_table_names_only_identifiers_that_exist():
    """THE MANIFEST CHECK.  The layer map is the document a future author reads
    before touching any of this; a table in it that names a constant which was
    renamed or removed is an actively misleading artefact.  Bind it to the
    code so it cannot rot silently -- which is exactly the failure mode S8 of
    the map itself records four instances of."""
    rows = _layer_map_rows()
    assert len(rows) >= 29, f'the switch table did not parse: {len(rows)} rows'
    mods = {'_lens_traced': LT, 'carrier': CM, '_lens_imap': IM}
    bad = []
    for ident, (mod_key, _shipped, _fb) in sorted(rows.items()):
        mod = mods.get(mod_key)
        if mod is None:
            bad.append(f'{ident}: unknown module column {mod_key!r}')
        elif not hasattr(mod, ident):
            bad.append(f'{ident}: not in {mod_key}')
    assert not bad, f'TRACED_LAYER_MAP.md S2 names missing identifiers: {bad}'


def test_the_layer_map_shipped_column_matches_the_library():
    """The other half: the table must not merely name real constants, it must
    state their real values.  A shipped default that moves without the table
    moving is the harness trap S7 item 1 in documentation form."""
    rows = _layer_map_rows()
    mods = {'_lens_traced': LT, 'carrier': CM, '_lens_imap': IM}
    bad = []
    for ident, (mod_key, shipped, _fb) in sorted(rows.items()):
        mod = mods.get(mod_key)
        if mod is None or not hasattr(mod, ident):
            continue
        live = getattr(mod, ident)
        try:
            want = eval(shipped, {'__builtins__': {}}, {})   # noqa: S307
        except Exception:
            bad.append(f'{ident}: unparseable shipped cell {shipped!r}')
            continue
        if not (live == want and type(live) is type(want)):
            bad.append(f'{ident}: doc says {want!r}, library has {live!r}')
    assert not bad, f'TRACED_LAYER_MAP.md S2 is stale: {bad}'


def test_the_layer_map_and_the_registry_agree_on_the_switches():
    """Two artefacts, one truth.  Anything the registry calls a switch must be
    in the map, and vice versa -- otherwise a reader of one gets a different
    answer from a reader of the other, which is how the C11/C13 arbiter-default
    contradiction happened in the first place."""
    doc = set(_layer_map_rows())
    reg = {name for (_m, name) in TF.FLAGS}
    assert not (reg - doc), f'registered but not in the layer map: {reg - doc}'
    assert not (doc - reg), f'in the layer map but not registered: {doc - reg}'


def test_traced_flags_restores_on_success_and_on_exception():
    """The discipline all 37 in-tree assignment sites follow by hand.  The
    exception arm is the one worth having: a save/restore that only works when
    nothing raises is the shape of leak ``tests/conftest.py``'s niche-C11 guard
    exists to contain."""
    before = (LT.REMAP_INVERSE_SUPPORT_BOUND, LT._REMAP_RESID_EIKONAL_DEGREE)
    with TF.traced_flags(REMAP_INVERSE_SUPPORT_BOUND=False,
                         _REMAP_RESID_EIKONAL_DEGREE=4):
        assert LT.REMAP_INVERSE_SUPPORT_BOUND is False
        assert LT._REMAP_RESID_EIKONAL_DEGREE == 4
    assert (LT.REMAP_INVERSE_SUPPORT_BOUND,
            LT._REMAP_RESID_EIKONAL_DEGREE) == before

    with pytest.raises(RuntimeError):
        with TF.traced_flags(REMAP_INVERSE_SUPPORT_BOUND=False):
            raise RuntimeError('boom')
    assert (LT.REMAP_INVERSE_SUPPORT_BOUND,
            LT._REMAP_RESID_EIKONAL_DEGREE) == before


def test_an_unknown_override_raises_instead_of_doing_nothing():
    """A typo'd override that silently did nothing would be exactly the class
    of failure this module exists to close."""
    with pytest.raises(KeyError, match='not a registered'):
        with TF.traced_flags(REMAP_INVERSE_SUPPORT_BOUNDS=False):
            pass


def test_the_lattice_corner_the_c8_case_rests_on_is_reachable():
    """The single most-cited comparison in the campaign -- C6 ON with C8 OFF --
    exists at NO point in history, which is the whole argument against an
    ordinal era switch (ARCH S5.2).  An era preset plus per-flag overrides must
    reach it."""
    with TF.traced_era('v5.32.1', REMAP_INVERSE_SUPPORT_BOUND=False):
        assert LT.REMAP_STATIONARY_PHASE_LAUNCH is True
        assert LT.REMAP_INVERSE_SUPPORT_BOUND is False
        assert LT._REMAP_RESID_EIKONAL_DEGREE == 6
    assert LT.REMAP_INVERSE_SUPPORT_BOUND is True


def test_the_oldest_era_is_the_pre_campaign_library():
    with TF.traced_era('v5.31'):
        assert LT.TILTED_CARRIER_EXACT_EIKONAL is False    # C5
        assert LT.REMAP_STATIONARY_PHASE_LAUNCH is False   # C6
        assert LT.REMAP_INVERSE_SUPPORT_BOUND is False     # C8
        assert LT.RAY_DENSITY_HALO_CHECK == 'silent'       # C7
        assert LT._FIT_DISC_OUTSIDE_WEIGHT_REL == 0.0      # D1
        assert LT._DECENTRE_GATE_W_FRAC == 0.0             # C1
    assert LT.TILTED_CARRIER_EXACT_EIKONAL is True


def test_traced_flag_state_names_every_registered_switch():
    """What a runner prints in its provenance banner and what a result cache
    keys on -- the hook for the two harness traps that are not the default
    (``LUMEN_PIN`` selecting a frozen export, ``wfe_probe_orders.py`` caching on
    the configuration rather than the library)."""
    state = TF.traced_flag_state()
    assert len(state) == len(TF.FLAGS)
    assert state['lumenairy.elements._lens_traced.'
                 'REMAP_INVERSE_SUPPORT_BOUND'] is True
    assert state['lumenairy.propagators.carrier.'
                 'SPHERE_PARAB_CONVERSION_EXACT'] is True


def test_no_prose_cross_reference_dangles():
    """THE DEFECT THIS KILLS, found by hand once already (ARCH S2.3a):
    ``TILTED_CARRIER_EXACT_EIKONAL``'s docstring cited
    ``carrier.CHAIN_EXACT_TILTED_REFERENCE``, and no such symbol exists
    anywhere in the repository.  A broken Sphinx reference is cheap; a prose
    graph that carries the load between fifteen correction layers and is
    maintained by hand is not, and it has already frayed in both directions.

    Scope is deliberately narrow so this cannot become a nuisance test: only
    ``:data:`~lumenairy.<module>.<NAME>``` forms, which name a module and a
    module-level constant unambiguously and can therefore be checked exactly.
    """
    pat = re.compile(r':data:`~?(lumenairy\.[A-Za-z_.]+)\.([A-Z_][A-Z0-9_]*)`')
    dangling = []
    for mod in (LT, CM):
        with open(mod.__file__, encoding='utf-8') as fh:
            src = fh.read()
        for mod_name, const in set(pat.findall(src)):
            try:
                target = importlib.import_module(mod_name)
            except ImportError:
                dangling.append(f'{mod.__name__}: no module {mod_name}')
                continue
            if not hasattr(target, const):
                dangling.append(f'{mod.__name__}: {mod_name}.{const}')
    assert not dangling, f'dangling :data: cross-references: {dangling}'


# ===========================================================================
# 2.  UNIT C -- the traced exit support as one object.
# ===========================================================================
def test_the_support_object_carries_all_three_views():
    """The three notions the architecture study found computed separately from
    the same arrays now hang off one object with one set of conventions."""
    S = LT._TracedExitSupport
    assert hasattr(S, 'half_planes')        # the shared hull BUILDER
    assert hasattr(S, 'signed_distance')    # the shared hull RULE
    assert hasattr(S, 'from_landings')
    for view in ('centroid', 'radius',      # C7
                 'hull', 'feather', 'bound', 'taper',   # C8
                 'retained_band_masks'):                # C14
        assert hasattr(S, view), view


def test_the_hull_builder_declines_or_raises_by_contract():
    """The two consumers genuinely differ and sharing the builder must not
    quietly convert one into the other: the C8 bound is optional containment
    (decline), the direct-fit hull IS the output domain (raise)."""
    S = LT._TracedExitSupport
    col = np.linspace(0.0, 1e-3, 8)         # collinear: no 2-D hull
    assert S.half_planes(col, col) is None
    with pytest.raises(Exception):
        S.half_planes(col, col, strict=True)
    # A real hull round-trips.
    t = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    A, b = S.half_planes(np.cos(t) * 1e-3, np.sin(t) * 1e-3)
    assert A.shape[0] == 2 and b.ndim == 1


def test_the_signed_distance_is_the_documented_rule():
    """``s = max_f (n_f . p + d_f)``: <= 0 inside, and the exact distance to
    the boundary outside.  Checked against the closed form on a disc-like
    hull, where the boundary is known."""
    S = LT._TracedExitSupport
    t = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    R = 1e-3
    A, b = S.half_planes(np.cos(t) * R, np.sin(t) * R)
    p = np.array([0.0, 0.5e-3, 1.5e-3, 3.0e-3])
    s = S.signed_distance(A, b, p, np.zeros_like(p))
    assert s[0] < 0.0 and s[1] < 0.0        # inside
    # Outside, s -> |p| - R as the polygon approaches the circle.
    assert np.allclose(s[2:], np.abs(p[2:]) - R, rtol=2e-4, atol=1e-8)


def test_the_band_mask_annulus_shortcut_is_exact():
    """``retained_band_masks`` screens the wave grid radially before the
    O(pixels x facets) reduction, so that a diagnostic does not put a ~30x BLAS
    pass on every ray-density call.  Both screens are STRICT bounds, so the
    verdict must be identical to the brute-force evaluation -- which is what
    this measures rather than assumes."""
    S = LT._TracedExitSupport
    rng = np.random.default_rng(20260803)
    pts = rng.normal(size=(400, 2)) * np.array([1.0e-3, 0.6e-3])
    A, b = S.half_planes(pts[:, 0], pts[:, 1])
    sup = S(hull=(A, b), feather=5e-5,
            hull_c=(float(pts[:, 0].mean()), float(pts[:, 1].mean())))
    sup.hull_rmax = float(np.hypot(pts[:, 0] - sup.hull_c[0],
                                   pts[:, 1] - sup.hull_c[1]).max())
    ax = np.linspace(-4e-3, 4e-3, 121)
    plateau = 8e-5
    inside, band = sup.retained_band_masks(ax, ax, plateau)
    # brute force, same rule, no screening
    X, Y = np.meshgrid(ax, ax)
    s = S.signed_distance(A, b, X, Y)
    w = plateau + sup.feather
    assert np.array_equal(inside, s <= 0.0)
    assert np.array_equal(band, (s > 0.0) & (s <= w))
    assert band.any(), 'the fixture must actually have a retained band'


# ---- the blind spot, and the fail-before switch ---------------------------
def _em6_call(bound=True, band='warn'):
    """``test_niche_audit_w3_elements.py::TestEM6RayDensityEnergySelfCheck``'s
    exact fixture -- the one ``RECON_PINS_POST_C8_2026_08_01`` S7.1 measured
    the joint blindness on (0.19998 of P_ap outside the exact-ray hull, with
    the field's GLOBAL maximum in that band)."""
    N, ap = 256, 3e-3
    dx = 1.01 * ap / N
    axis = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(axis, axis)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (1.4e-3) ** 2).astype(np.complex128)
    presc = {'name': 'biconcave', 'aperture_diameter': ap,
             'thicknesses': [3e-3],
             'surfaces': [
                 {'radius': -3e-3, 'glass_before': 'air',
                  'glass_after': 'N-BK7', 'conic': 0.0,
                  'aspheric_coeffs': None},
                 {'radius': 3e-3, 'glass_before': 'N-BK7',
                  'glass_after': 'air', 'conic': 0.0,
                  'aspheric_coeffs': None}]}
    old = (LT.REMAP_INVERSE_SUPPORT_BOUND, LT.SUPPORT_BAND_CHECK)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    LT.SUPPORT_BAND_CHECK = band
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            E = la.apply_real_lens_traced(
                E0, prescription=presc, wavelength=_WL, dx=dx,
                amplitude_model='ray_density', ray_subsample=8, n_workers=1,
                parallel_amp=False, on_undersample='silent',
                on_aperture_beam='silent')
    finally:
        LT.REMAP_INVERSE_SUPPORT_BOUND, LT.SUPPORT_BAND_CHECK = old
    return np.asarray(E), [str(r.message) for r in rec]


def _band_msgs(texts):
    return [t for t in texts if 'SUPPORT-BAND self-check FAILED' in t]


@pytest.fixture(scope='module')
def _em6():
    """One call, reused: the fixture costs a full traced element call at
    ray_subsample=8 and three tests read the same result."""
    return _em6_call(bound=True, band='warn')


def test_the_band_check_ships_on_with_its_fail_before_documented():
    assert LT.SUPPORT_BAND_CHECK == 'warn'
    assert LT._SUPPORT_BAND_PEAK_RATIO_TOL == 1.0
    doc = LT.__doc__ or ''
    del doc
    # the fail-before value must be the one the note names
    assert 'silent' in _band_doc(), _band_doc()[:200]


def _band_doc():
    with open(LT.__file__, encoding='utf-8') as fh:
        src = fh.read()
    i = src.index('SUPPORT_BAND_CHECK = ')
    return src[max(0, i - 4000):i]


def test_the_band_check_fires_where_both_old_checks_are_blind(_em6):
    """THE POSITIVE CONTROL, and the whole point of UNIT C.

    On this fixture the post-C8 field carries 0.19998 of ``P_ap`` outside the
    exact-ray hull -- in the plateau+feather band C8 retains DELIBERATELY --
    and its global ``|E|`` maximum sits there.  Pre-C14 nothing said so: the
    energy check reads 1.01931, inside its own band, and the halo check reports
    only beyond ``1.25 x r_hull``, which under the bound is territory C8 has
    already zeroed.  Measured here: the band maximum is 2.14x the maximum
    inside the traced support."""
    _E, texts = _em6
    msgs = _band_msgs(texts)
    assert msgs, (
        'the support-band check did not fire on the fixture the joint '
        'blindness was measured on: ' + repr(texts[:4]))
    m = re.search(r'which is ([0-9.]+)x the maximum', msgs[0])
    assert m and float(m.group(1)) > 1.0, msgs[0]
    # and it says the three things a reader needs to act
    assert 'g_band' in msgs[0]
    assert 'NO TRACED RAY OF THIS CALL LANDED THERE' in msgs[0]
    assert 'SUPPORT_BAND_CHECK' in msgs[0]


def test_the_band_message_does_not_collide_with_another_checks_filter(_em6):
    """A NEW WARNING IS A NEW WAY TO BREAK SOMEONE ELSE'S TEST, and this one
    did: the first draft explained itself with the phrase "neither the energy
    self-check nor the HALO self-check can see this band", and
    ``test_niche_audit_w3_elements`` collects warnings with
    ``[t for t in texts if 'energy self-check' in t]`` and then asserts that
    list is EMPTY on the bounded arm.  A purely explanatory clause therefore
    turned a green pin red on a fixture where the new check is *supposed* to
    fire.

    The substrings below are every phrase the suite filters warning text on.
    A message that contains one is claiming to be that other check."""
    _E, texts = _em6
    msgs = _band_msgs(texts)
    assert msgs
    for phrase in ('energy self-check', 'HALO self-check', 'fold caustic'):
        assert phrase not in msgs[0], (
            f'the support-band message contains {phrase!r}, which another '
            f"suite filters warning text on -- it will be misread as that "
            f"check firing")


def test_the_old_checks_really_are_blind_there(_em6):
    """The negative half of the same claim, asserted rather than quoted: on
    this call the halo check and the energy check are BOTH silent about the
    band.  Without this, the positive control above could be re-proving
    something an existing instrument already caught."""
    _E, texts = _em6
    assert not [t for t in texts if 'HALO self-check FAILED' in t]


def test_the_fail_before_switch_restores_pre_c14_reporting(_em6):
    """``SUPPORT_BAND_CHECK = 'silent'`` is the fail-before, and it must be a
    REPORTING switch only -- the field is byte-identical in both states, which
    is what lets the whole of niche C14 claim bit-neutrality."""
    E_on, texts_on = _em6
    E_off, texts_off = _em6_call(bound=True, band='silent')
    assert _band_msgs(texts_on)
    assert not _band_msgs(texts_off)
    assert np.array_equal(E_on, E_off), (
        'the band check moved a returned bit; it is reporting-only')


def test_the_band_check_is_silent_on_a_clean_call():
    """The instrument must separate populations, not fire everywhere.  A
    well-conditioned call whose peak is at its focus -- where every correct
    field's peak is -- must be silent."""
    N, ap, dx = 128, 3e-3, 2.2 * 3e-3 / 128
    axis = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(axis, axis)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    presc = {'name': 'singlet', 'aperture_diameter': ap,
             'thicknesses': [2e-3],
             'surfaces': [
                 {'radius': 20e-3, 'glass_before': 'air',
                  'glass_after': 'N-BK7', 'conic': 0.0,
                  'aspheric_coeffs': None},
                 {'radius': -20e-3, 'glass_before': 'N-BK7',
                  'glass_after': 'air', 'conic': 0.0,
                  'aspheric_coeffs': None}]}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(
            E0, prescription=presc, wavelength=_WL, dx=dx,
            amplitude_model='ray_density', ray_subsample=4, n_workers=1,
            parallel_amp=False, on_undersample='silent',
            on_aperture_beam='silent')
    assert not _band_msgs([str(r.message) for r in rec])


# ===========================================================================
# 3.  Guard integrity -- the dx self-check's four exits.
# ===========================================================================
def test_the_convergence_flag_is_not_deduped_per_caller_line():
    """THE MEASURED LIBRARY DEFECT (P2 diagnosis S4.1).  The guard warned
    through ``warnings.warn(..., stacklevel=3)``, which attributes the warning
    to the CALLER of ``propagate_traced_carrier_chain`` -- correct for blame,
    fatal for delivery, because CPython's ``'default'`` action is once per
    (text, category, module, lineno) and a batch loop calls the chain from ONE
    line.  Measured on the real chain by ``p2diag_prod_dedup.py``: two
    non-converged calls, ONE flag.

    This exercises the delivery primitive directly, under STOCK filters and
    from one source line, because that is the exact condition the defect needs
    -- ``pytest.warns`` enters ``catch_warnings`` and sets ``'always'``, which
    resets the registry and would hide the whole phenomenon."""
    seen = []
    orig = warnings.showwarning

    def _count(message, category, filename, lineno, file=None, line=None):
        seen.append((category.__name__, str(message), lineno))

    with warnings.catch_warnings():
        warnings.resetwarnings()
        warnings.simplefilter('default')     # CPython's stock action
        warnings.showwarning = _count
        try:
            for _ in range(3):               # <-- ONE call site, three times
                CM._warn_undeduped('C14 delivery probe', stacklevel=1)
        finally:
            warnings.showwarning = orig
    assert len(seen) == 3, (
        f'the flag was deduped: {len(seen)} of 3 delivered.  Every later '
        f'non-converged result in a batch loop would return unflagged.')


def test_an_ignore_filter_still_silences_it():
    """The fix must bypass the ``'default'`` action's dedup and NOTHING else:
    a caller who has explicitly silenced this category keeps it silenced.  That
    is why it is not ``catch_warnings() + simplefilter('always')``, which would
    override the caller's own configuration."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.resetwarnings()
        warnings.simplefilter('ignore', RuntimeWarning)
        CM._warn_undeduped('C14 delivery probe', stacklevel=1)
    assert not rec


def test_it_is_attributed_to_the_callers_line_not_the_library():
    """``stacklevel`` semantics must survive the change: the whole reason the
    guard used ``stacklevel=3`` is that the useful blame is the caller's line,
    and the fix reproduces ``warnings.warn``'s own frame resolution."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.resetwarnings()
        warnings.simplefilter('always')
        CM._warn_undeduped('C14 attribution probe', stacklevel=1)
    assert rec and rec[0].filename == __file__, (
        rec[0].filename if rec else 'no warning')


# ---- the three silent-pass holes -----------------------------------------
def _tiny_chain(**over):
    """The smallest chain that reaches the self-check.  Deliberately tiny: the
    holes are control-flow, not physics, so the fixture only has to RUN."""
    N, dx = 96, 8e-6
    axis = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(axis, axis)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (0.15e-3) ** 2).astype(np.complex128)
    groups = [{'prescription': {
        'name': 'g', 'aperture_diameter': 0.9e-3, 'thicknesses': [0.3e-3],
        'surfaces': [
            {'radius': 6e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
             'conic': 0.0, 'aspheric_coeffs': None},
            {'radius': -6e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
             'conic': 0.0, 'aspheric_coeffs': None}]},
        'gap_before': 0.0}]
    kw = dict(r_in=np.inf, ray_subsample=4, n_workers=1,
              traced_kwargs=dict(parallel_amp=False, on_undersample='silent'),
              final_distance=3.0e-3, final_leg='paraxial',
              focus_readout=dict(dx_out=0.3e-6, N_out=64),
              self_check='dx')
    kw.update(over)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.propagate_traced_carrier_chain(E0, groups, _WL, dx, **kw)
    return [str(r.message) for r in rec]


def _declined(texts):
    return [t for t in texts if "self_check='dx': DECLINED" in t]


def test_hole_a_a_degenerate_primary_result_no_longer_reads_as_stable(
        monkeypatch):
    """``_chain_result_metrics`` returns ``{}`` whenever the field's total
    intensity is non-finite or ``<= 0``.  The guard used to ``return``
    silently -- so a primary run that DEGENERATED read as dx-stable and the
    refined chain was never even executed."""
    monkeypatch.setattr(CM, '_chain_result_metrics', lambda res: {})
    texts = _tiny_chain()
    msgs = _declined(texts)
    assert msgs, texts[-3:]
    assert 'PRIMARY result carries no comparable metric' in msgs[0]
    assert 'NO convergence evidence' in msgs[0]


def test_hole_a_declines_before_paying_for_the_second_chain(monkeypatch):
    """It must return BEFORE the refined run: a check that cannot compare
    should not cost 2x to say so.  Poisoning the upsample proves the second
    chain is never started."""
    monkeypatch.setattr(CM, '_chain_result_metrics', lambda res: {})

    def _boom(*a, **k):                       # pragma: no cover - must not run
        raise AssertionError('the refined chain was started anyway')
    monkeypatch.setattr(CM, '_fourier_upsample_crop', _boom)
    assert _declined(_tiny_chain())


def test_hole_b_a_degenerate_refined_run_no_longer_reads_as_stable(
        monkeypatch):
    """If the REFINED run degenerates, ``m2`` is ``{}``, the key intersection
    is empty, ``bad`` stays empty and the guard used to return without warning
    -- after paying for BOTH chains.  A self-check that cannot compare must say
    so."""
    calls = {'n': 0}
    real = CM._chain_result_metrics

    def _second_is_empty(res):
        calls['n'] += 1
        return real(res) if calls['n'] == 1 else {}
    monkeypatch.setattr(CM, '_chain_result_metrics', _second_is_empty)
    msgs = _declined(_tiny_chain())
    assert msgs, 'the refined-run hole is still silent'
    assert 'no metric in common' in msgs[0]
    assert 'DEGENERATED' in msgs[0]


def test_hole_b_a_branch_change_under_refinement_is_reported(monkeypatch):
    """The subtler half of the same hole.  The two metric branches share the
    key ``'power'`` and MEAN DIFFERENT THINGS BY IT (envelope window power vs
    ``sum|E|^2 dx^2`` on a readout grid), so intersecting the keys across a
    branch change compares two unrelated numbers.  A refinement that changes
    the chain's own routing is itself a convergence failure and now says so."""
    calls = {'n': 0}

    def _kind(res):
        calls['n'] += 1
        return 'focal' if calls['n'] == 1 else 'envelope'
    monkeypatch.setattr(CM, '_chain_metric_kind', _kind)
    msgs = _declined(_tiny_chain())
    assert msgs, 'a cross-branch comparison is still made silently'
    assert 'metric branch' in msgs[0] and 'refining' in msgs[0].lower()


def test_hole_c_the_readoutless_mode_is_refused_not_run(monkeypatch):
    """Without a focus readout the compared quantities -- ``w_env``, ``power``,
    ``R`` -- are dx-INVARIANT by construction: measured 0.0867 %, 0.0015 % and
    0 % on the same fixture that moves 52.5 % through a readout.  The mode was
    very nearly a no-op that cost 2x runtime and returned a clean bill of
    health for a chain that is not converged.  It now declines up front, names
    the remedy, and does not start the second chain."""
    def _boom(*a, **k):                       # pragma: no cover - must not run
        raise AssertionError('the refined chain was started anyway')
    monkeypatch.setattr(CM, '_fourier_upsample_crop', _boom)
    msgs = _declined(_tiny_chain(focus_readout=None, final_distance=0.0))
    assert msgs, 'the readout-less mode still runs its dx-invariant comparison'
    assert 'NO focus readout' in msgs[0]
    assert 'dx-INVARIANT' in msgs[0]
    assert 'focus_readout=dict' in msgs[0]      # the remedy is named


def test_a_converged_chain_is_still_silent():
    """The contract that must NOT move: this file adds four ways to speak and
    zero ways to cry wolf.  A chain that compares and agrees says nothing."""
    texts = _tiny_chain(self_check_tol=10.0)
    assert not [t for t in texts if 'self_check' in t], (
        [t[:120] for t in texts if 'self_check' in t])


def test_the_metric_kind_predicate_names_both_branches():
    class _R:
        R = None
    assert CM._chain_metric_kind(_R()) == 'focal'
    _R.R = 0.12
    assert CM._chain_metric_kind(_R()) == 'envelope'
