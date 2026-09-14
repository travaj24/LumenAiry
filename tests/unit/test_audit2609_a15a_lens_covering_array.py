"""Combination coverage for the ``apply_real_lens`` family (audit 2026-09-11, V3).

WHY THIS FILE EXISTS.  The audit measured 673 ``apply_real_lens*`` call sites in
the corpus exercising 177 distinct kwarg combinations, **68 % of them passing
zero or one optional kwarg**.  ``apply_real_lens_traced`` has 48 parameters;
pairwise coverage alone needs of order a thousand combinations.  Four of the
five defects the orchestrator seeded were *interaction* defects -- a flag
combined with a geometry -- and the suite, being organised one file per past
bug and testing one knob at a time against a default background, had by
construction almost no power against them.

WHAT IT ASSERTS.  Deliberately only invariants that hold for EVERY legal
combination, because an oracle that held for every combination would be a
second implementation of the lens:

  1. **finiteness** -- no NaN/Inf anywhere in the returned field;
  2. **shape** -- the output grid matches the input grid;
  3. **no energy gain** -- ``sum |E_out|^2 dx dy <= sum |E_in|^2 dx dy``
     within a measured tolerance.  This is the direction that is physically
     unconditional: a passive lens can only LOSE power (aperture clipping,
     Fresnel reflection, absorption, energy leaving the grid).  A lower bound
     would be combination-dependent -- ``absorption=True`` and a caustic
     evaluated 2 mm past focus legitimately lose most of it -- so the test
     asserts instead that the output is not identically zero, which is the
     part of "did anything happen" that is combination-independent;
  4. **default-passed == omitted, bit for bit** -- the "knob silently
     discarded" detector.  Every physics kwarg, passed explicitly AT ITS
     DOCUMENTED DEFAULT, must reproduce the call that omits it byte-identically.
     A knob whose "unset" sentinel and whose documented default disagree fails
     here and nowhere else in the suite.

These are cheap invariants, and cheap invariants over a covering array is
exactly the shape that catches interaction breakage: a knob pair that routes
into a path with a stale array shape, an unmasked division, or a double-applied
transmittance trips (1)-(3) immediately.

THE ARRAY.  Factors are groups of kwargs, not single kwargs, because several of
these knobs are only legal in the presence of another -- MEASURED 2026-09-12
against the committed signatures (WP-A2 commit b97c0b6e, WP-A3 ``fix(lens-traced):
WP-A3``):

  * ``conjugate`` / ``displaced_mode`` / ``displaced_obliquity`` raise unless
    ``surface_model='displaced'``;
  * ``remap_order`` raises unless ``surface_model='tangent_facet_remap'``;
  * ``caustic`` raises unless ``amplitude_model='ray_density'``;
  * ``output_plane_distance`` raises unless ``caustic`` is one of
    ``'multibranch' / 'uniform' / 'wave'``;
  * ``screen_obliquity=True`` raises without ``carrier=``.

So each factor's levels carry their whole legal group, and the pairwise array
is built over factors.  ``_pairwise_rows`` is a deterministic greedy
(IPOG-style) generator: no randomness, no seed, same rows on every machine.

THE FIXTURE is the geometry the audit says the suite never tests: an AC254-ish
cemented doublet with a **curved rear** (R3 = -291.07 mm -- the exit-vertex
class is invisible on a plano rear) illuminated by a **diverging spherical
wave** from 120 mm in front (non-collimated -- the collimated case is the one
every existing call site uses).  ``lens_covering_array_fixture`` and the factor
tables are module-level and importable on purpose: WP-A16's config-object work
reuses them for its ``LensGeometry``/``LensNumerics``/``LensResources``
bit-identity test, so the two agree on one fixture rather than two.

RUNTIME BUDGET.  N = 64 on a 6 mm aperture, ``ray_subsample=1`` (the traced
undersample guard needs >= 32 coarse samples across the aperture, and 64 px
over 1.2x the aperture gives ~53 -- at the shipped default of 8 the call
refuses, correctly).  MEASURED 2026-09-12: a warm analytic call is 1-7 ms and a
warm traced call 20-45 ms; the slowest single arm is
``displaced_obliquity='pointwise'`` at 0.37 s.  Whole file: see the module test
run in the WP-A15a report -- it is seconds, not minutes, and it is well inside
the fast lane.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements._lens_traced import apply_real_lens_traced

WAVELENGTH = 632.8e-9
APERTURE = 6.0e-3
N_GRID = 64
SOURCE_Z = -120.0e-3          # diverging wave origin, 120 mm before the lens


# ---------------------------------------------------------------------------
# Fixture factory (importable -- WP-A16 reuses it)
# ---------------------------------------------------------------------------

def curved_rear_doublet(aperture: float = APERTURE) -> dict:
    """AC254-ish cemented doublet whose LAST surface is curved.

    R3 = -291.07 mm rather than the plano rear of the existing
    ``seidel_correction`` fixture: on a flat rear the exit vertex coincides
    with the last surface and the whole exit-vertex bug class (audit S15.1) is
    invisible.
    """
    return dict(
        surfaces=[
            dict(radius=33.3e-3, glass_before='AIR', glass_after='N-BAF10'),
            dict(radius=-22.28e-3, glass_before='N-BAF10',
                 glass_after='N-SF6HT'),
            dict(radius=-291.07e-3, glass_before='N-SF6HT',
                 glass_after='AIR'),
        ],
        thicknesses=[9.0e-3, 2.5e-3],
        aperture_diameter=aperture,
    )


def lens_covering_array_fixture(n: int = N_GRID, aperture: float = APERTURE,
                                source_z: float = SOURCE_Z):
    """Return ``(E_in, dx, prescription, wavelength)``.

    The input is a Gaussian-apodised **diverging** spherical wave -- the
    non-collimated case.  ``dx`` puts 1.2 aperture diameters on the grid so
    the aperture edge is resolved and the clip is visible in the energy budget.
    """
    dx = 1.2 * aperture / n
    x = (np.arange(n) - (n - 1) / 2.0) * dx
    xx, yy = np.meshgrid(x, x)
    r2 = xx ** 2 + yy ** 2
    amp = np.exp(-r2 / (0.30 * aperture) ** 2)
    phase = (2.0 * np.pi / WAVELENGTH) * r2 / (2.0 * abs(source_z))
    e_in = (amp * np.exp(1j * phase)).astype(np.complex128)
    return e_in, dx, curved_rear_doublet(aperture), WAVELENGTH


def analytic_base_kwargs(dx: float, prescription: dict) -> dict:
    return dict(prescription=prescription, wavelength=WAVELENGTH, dx=dx)


def traced_base_kwargs(dx: float, prescription: dict) -> dict:
    # ``ray_subsample=1``: at N=64 the shipped default of 8 leaves 5 coarse
    # samples across the aperture and the undersample guard REFUSES the call
    # (correctly -- measured message quoted in the WP-A15a report).  1 gives
    # ~53, comfortably over the guard's threshold of 32.
    return dict(prescription=prescription, wavelength=WAVELENGTH, dx=dx,
                ray_subsample=1)


# ---------------------------------------------------------------------------
# Factor tables.  factor name -> list of levels; level 0 is always "default".
# ---------------------------------------------------------------------------

ANALYTIC_FACTORS: dict[str, list[dict]] = {
    'bandlimit': [{}, {'bandlimit': False}],
    'fresnel': [{}, {'fresnel': True}],
    'absorption': [{}, {'absorption': True}],
    # ``slant_correction`` and ``seidel_correction`` are MUTUALLY EXCLUSIVE --
    # found by this array on its second row, and the library says why:
    # "the two flags replace the SAME per-surface coefficient, so stacking them
    # double-counts the facet obliquity (measured 173.5 -> 1488.6 nm rms exit
    # OPD on an 8 mm cemented doublet with both on)".  Encoding them as one
    # three-level factor is how a covering array expresses an exclusion: the
    # illegal pair is never generated, and the two legal alternatives still get
    # paired against every other factor.
    'opd_correction': [
        {},
        {'slant_correction': True},
        {'seidel_correction': True, 'seidel_poly_order': 4},
    ],
    'surface_frame': [{}, {'surface_frame': True}],
    'surface_model': [
        {},
        {'surface_model': 'displaced', 'displaced_mode': 'remap',
         'displaced_obliquity': 'meridional', 'conjugate': SOURCE_Z},
        {'surface_model': 'tangent_facet_remap', 'remap_order': 1},
    ],
    'carrier': [{}, {'carrier': 'auto', 'screen_obliquity': True}],
    'propagator': [{}, {'wave_propagator': 'rs'}],
    'stream_tf': [{}, {'stream_transfer_function': True}],
}

TRACED_FACTORS: dict[str, list[dict]] = {
    'bandlimit': [{}, {'bandlimit': False}],
    'input_phase': [{}, {'preserve_input_phase': False}],
    'tilt_aware': [{}, {'tilt_aware_rays': True}],
    'fit_radius': [{}, {'fit_radius_beam_factor': 1.5}],
    'fit': [
        {},
        {'newton_fit': 'polynomial', 'newton_poly_order': 4},
        {'newton_fit': 'spline'},
    ],
    'fast_phase': [{}, {'fast_analytic_phase': True}],
    'amplitude': [
        {},
        {'amplitude_model': 'ray_density'},
        {'amplitude_model': 'ray_density', 'caustic': 'multibranch',
         'output_plane_distance': 2.0e-3, 'caustic_band': 'plain',
         'caustic_ray_subsample': 1},
        {'amplitude_model': 'ray_density', 'caustic': 'uniform',
         'output_plane_distance': 2.0e-3},
    ],
    'remap_sampling': [{}, {'remap_sampling': 'full'}],
    'carrier': [{}, {'carrier': 'auto'}],
    'decentred_fit': [{}, {'decentred_fit_poly_order': 4}],
    'newton_tuning': [{}, {'newton_amp_mask_rel': 1.0e-3,
                           'newton_max_iters': 40}],
    'inverse_map': [{}, {'inverse_map': True}],
    'subsample': [{}, {'ray_subsample': 2,
                       'min_coarse_samples_per_aperture': 0}],
}


# ---------------------------------------------------------------------------
# Declared EXCLUSIONS.  ``(factor_a, level_a, factor_b, level_b) -> reason``,
# where the reason quotes the library's own refusal.  Every entry here was
# MEASURED 2026-09-12 by sweeping all 141 analytic and 308 traced factor-level
# pairs against the committed signatures and recording which ones raise: 11 of
# 141 and 3 of 308 respectively.  Nothing in this table is assumed -- and
# ``test_every_declared_exclusion_is_really_refused`` re-measures all 14 on
# every run, so an exclusion that stops being real (or a refusal that silently
# turns into a wrong answer) fails the gate instead of quietly shrinking the
# array's coverage.
ANALYTIC_EXCLUSIONS: dict[tuple, str] = {
    ('fresnel', 1, 'surface_model', 1):
        "surface_model='displaced' is incompatible with ['fresnel']",
    ('absorption', 1, 'surface_model', 1):
        "surface_model='displaced' is incompatible with ['absorption']",
    ('opd_correction', 1, 'surface_model', 1):
        "surface_model='displaced' is incompatible with ['slant_correction']",
    ('opd_correction', 2, 'surface_model', 1):
        "surface_model='displaced' is incompatible with ['seidel_correction']",
    ('opd_correction', 1, 'surface_model', 2):
        "slant_correction=True is not supported with "
        "surface_model='tangent_facet_remap'",
    ('surface_frame', 1, 'surface_model', 1):
        "surface_model='displaced' is incompatible with ['surface_frame']",
    ('surface_frame', 1, 'surface_model', 2):
        "surface_frame=True is not supported with "
        "surface_model='tangent_facet_remap'",
    ('surface_model', 1, 'carrier', 1):
        "carrier= is only supported with the default surface_model='thin'",
    ('surface_model', 2, 'carrier', 1):
        "screen_obliquity=True is not supported with "
        "surface_model='tangent_facet_remap'",
    ('surface_model', 1, 'propagator', 1):
        "surface_model='displaced' requires the ASM in-glass propagator",
    ('surface_model', 2, 'propagator', 1):
        "surface_model='tangent_facet_remap' needs an exact "
        "angular-spectrum gap",
}

TRACED_EXCLUSIONS: dict[tuple, str] = {
    ('amplitude', 1, 'newton_tuning', 1):
        "newton_amp_mask_rel conflicts with amplitude_model='ray_density', "
        "which requires the FULL coarse Newton grid",
    ('amplitude', 2, 'newton_tuning', 1):
        "newton_amp_mask_rel conflicts with amplitude_model='ray_density'",
    ('amplitude', 3, 'newton_tuning', 1):
        "newton_amp_mask_rel conflicts with amplitude_model='ray_density'",
}


def _excluded(factors, exclusions, i_name, a, j_name, b) -> bool:
    return ((i_name, a, j_name, b) in exclusions
            or (j_name, b, i_name, a) in exclusions)


def _pairwise_rows(factors: dict[str, list],
                   exclusions: dict[tuple, str] | None = None) -> list[dict]:
    """Deterministic greedy pairwise covering array over ``factors``.

    Returns a list of ``{factor: level_index}`` rows such that every
    (factor_i = a, factor_j = b) pair appears in at least one row.  Greedy
    IPOG-style: each row is built factor by factor in declaration order,
    choosing at each step the level that covers the most still-uncovered pairs
    (ties broken by the lower level index).  No randomness and no seed, so the
    rows are identical on every machine and in every run -- a covering array
    that changed run to run would make a failure unreproducible, which is the
    opposite of what a gate is for.
    """
    exclusions = exclusions or {}
    names = list(factors)
    want = set()
    for i, j in itertools.combinations(range(len(names)), 2):
        for a in range(len(factors[names[i]])):
            for b in range(len(factors[names[j]])):
                if _excluded(factors, exclusions, names[i], a, names[j], b):
                    continue
                want.add((i, a, j, b))

    rows: list[dict] = []
    # Hard cap: the greedy always terminates (each row covers >= 1 pair while
    # any remain), but a cap turns a logic error into a loud failure instead of
    # a hang.
    for _ in range(len(want) + 1):
        if not want:
            break
        chosen: dict[int, int] = {}
        for i in range(len(names)):
            best_level = None
            best_score = (-1, -1)
            for a in range(len(factors[names[i]])):
                # a level that conflicts with anything already fixed in this
                # row cannot be used: the library refuses the call outright
                if any(_excluded(factors, exclusions, names[i], a, names[j], b)
                       for j, b in chosen.items()):
                    continue
                # pairs this level would close against the factors already
                # fixed in this row -- the real gain
                gain = sum(
                    1 for j, b in chosen.items()
                    if ((j, b, i, a) if j < i else (i, a, j, b)) in want)
                # tie-break: how many factors still unchosen have at least one
                # uncovered pair with (i, a), so the first factor of a row is
                # not picked blind
                reach = sum(
                    1 for j in range(len(names))
                    if j != i and j not in chosen
                    and any(((i, a, j, b) if i < j else (j, b, i, a)) in want
                            for b in range(len(factors[names[j]]))))
                if (gain, reach) > best_score:
                    best_level, best_score = a, (gain, reach)
            assert best_level is not None, (
                f'no legal level for factor {names[i]!r} given the partial row '
                f'{ {names[k]: v for k, v in chosen.items()} } -- the exclusion '
                f'table forbids every level, so the factor tables and the '
                f'exclusions disagree.')
            chosen[i] = best_level
        covered = {(i, chosen[i], j, chosen[j])
                   for i, j in itertools.combinations(range(len(names)), 2)}
        newly = want & covered
        if not newly:
            # greedy stalled: fall back to covering one remaining pair exactly,
            # with every other factor at its default (level 0, always legal).
            i, a, j, b = sorted(want)[0]
            chosen = {k: 0 for k in range(len(names))}
            chosen[i], chosen[j] = a, b
            covered = {(p, chosen[p], q, chosen[q])
                       for p, q in itertools.combinations(range(len(names)), 2)}
            newly = want & covered
        want -= covered
        rows.append({names[i]: chosen[i] for i in range(len(names))})
    assert not want, (
        f"covering-array generator left {len(want)} pairs uncovered -- the "
        f"generator is broken, not the library.")
    return rows


def _kwargs_for(factors: dict[str, list[dict]], row: dict) -> dict:
    kw: dict = {}
    for name, level in row.items():
        kw.update(factors[name][level])
    return kw


def _row_label(factors: dict[str, list[dict]], row: dict) -> str:
    on = [f'{n}={l}' for n, l in row.items() if l]
    return '+'.join(on) if on else 'all-default'


ANALYTIC_ROWS = _pairwise_rows(ANALYTIC_FACTORS, ANALYTIC_EXCLUSIONS)
TRACED_ROWS = _pairwise_rows(TRACED_FACTORS, TRACED_EXCLUSIONS)


def _power(field: np.ndarray, dx: float) -> float:
    return float(np.sum(np.abs(field) ** 2)) * dx * dx


def _as_field(out) -> np.ndarray:
    return np.asarray(out[0] if isinstance(out, tuple) else out)


# Energy bar.  ``sum |E|^2 dx dy`` IS the optical power in this library (the
# propagators are Parseval-unitary and carry no impedance factor -- WP-A2's
# L13 note), so a passive element cannot increase it.  What the bar has to
# absorb is interpolation and re-gridding round-off, not physics.
# MEASURED 2026-09-12 over every arm of both arrays on the fixture below
# (12 analytic rows, 15 traced rows).  The largest P_out/P_in observed is
# **0.996170598** (analytic, all-default: aperture clipping alone) and
# **0.996162541** (traced, all-default); the smallest is 0.354847608 (traced,
# ray_density at a caustic) and 0.438620794 (analytic, displaced model).  So
# the largest EXCESS over 1 across all 27 arms is **-3.83e-03**, i.e. no arm
# gains at all and the nearest arm sits 3.8e-3 BELOW the bar's zero point.
# The bar is 1e-6 relative: seven decades above the float64 accumulation floor
# for a 64x64 reduction (~N^2 * eps ~ 1e-13 relative) and four decades below
# the smallest gain a real defect would produce -- a double-applied ~4 %
# Fresnel transmittance, a dropped obliquity cosine, or an un-normalised
# ray-density Jacobian all land at 1e-2 or larger.  It is a ONE-SIDED bar by
# construction: the loss side is combination-dependent (0.35 to 0.996 here) and
# is deliberately not pinned.
_ENERGY_GAIN_TOL = 1.0e-6


def _assert_sane(field: np.ndarray, e_in: np.ndarray, dx: float, label: str,
                 fn_name: str) -> float:
    assert field.shape == e_in.shape, (
        f'{fn_name}[{label}]: output shape {field.shape} != input shape '
        f'{e_in.shape}.')
    assert np.all(np.isfinite(field)), (
        f'{fn_name}[{label}]: output carries '
        f'{int(np.count_nonzero(~np.isfinite(field)))} non-finite samples out '
        f'of {field.size}.')
    p_in = _power(e_in, dx)
    p_out = _power(field, dx)
    assert p_out <= p_in * (1.0 + _ENERGY_GAIN_TOL), (
        f'{fn_name}[{label}]: output power {p_out:.12e} EXCEEDS input power '
        f'{p_in:.12e} by {p_out / p_in - 1.0:.3e} relative, over the '
        f'{_ENERGY_GAIN_TOL:.0e} tolerance.  A passive lens cannot create '
        f'power; this combination is double-applying a transmittance, '
        f'dropping an obliquity factor, or renormalising a ray-density '
        f'Jacobian it should not.')
    assert p_out > 0.0, (
        f'{fn_name}[{label}]: output is identically zero.  The combination '
        f'is not merely lossy, it produced no field at all.')
    return p_out / p_in


# ---------------------------------------------------------------------------
# 1. The covering arrays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    'row', ANALYTIC_ROWS,
    ids=[_row_label(ANALYTIC_FACTORS, r) for r in ANALYTIC_ROWS])
def test_analytic_covering_array_is_finite_and_loses_no_energy(row):
    """Pairwise over 16 physics kwargs of ``apply_real_lens``."""
    e_in, dx, rx, _ = lens_covering_array_fixture()
    kw = analytic_base_kwargs(dx, rx)
    kw.update(_kwargs_for(ANALYTIC_FACTORS, row))
    label = _row_label(ANALYTIC_FACTORS, row)
    with warnings.catch_warnings():
        # The fixture is deliberately non-collimated and deliberately clipped,
        # so several arms emit the library's (correct, informative) diagnostics.
        # They are not the subject of this test.
        warnings.simplefilter('ignore')
        out = apply_real_lens(e_in, **kw)
    _assert_sane(_as_field(out), e_in, dx, label, 'apply_real_lens')


@pytest.mark.parametrize(
    'row', TRACED_ROWS,
    ids=[_row_label(TRACED_FACTORS, r) for r in TRACED_ROWS])
def test_traced_covering_array_is_finite_and_loses_no_energy(row):
    """Pairwise over 20 physics kwargs of ``apply_real_lens_traced``."""
    e_in, dx, rx, _ = lens_covering_array_fixture()
    kw = traced_base_kwargs(dx, rx)
    kw.update(_kwargs_for(TRACED_FACTORS, row))
    label = _row_label(TRACED_FACTORS, row)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_traced(e_in, **kw)
    _assert_sane(_as_field(out), e_in, dx, label, 'apply_real_lens_traced')


def test_the_covering_arrays_really_cover_every_pair():
    """Counter-pin on the generator itself.

    If ``_pairwise_rows`` silently returned one all-default row, every arm
    above would pass and the file would assert nothing about combinations.
    This re-derives the pair set from the emitted rows and asserts it is
    complete -- and that the arrays are small enough to be worth running
    (a pairwise array over these factors is of order ten rows, not hundreds;
    if it ever explodes, the factor table grew a level nobody intended).
    """
    for name, factors, rows, excl in (
            ('analytic', ANALYTIC_FACTORS, ANALYTIC_ROWS, ANALYTIC_EXCLUSIONS),
            ('traced', TRACED_FACTORS, TRACED_ROWS, TRACED_EXCLUSIONS),
            ('maslov', MASLOV_FACTORS, MASLOV_ROWS, MASLOV_EXCLUSIONS)):
        names = list(factors)
        need = {(i, a, j, b)
                for i, j in itertools.combinations(range(len(names)), 2)
                for a in range(len(factors[names[i]]))
                for b in range(len(factors[names[j]]))
                if not _excluded(factors, excl, names[i], a, names[j], b)}
        got = set()
        for r in rows:
            for i, j in itertools.combinations(range(len(names)), 2):
                got.add((i, r[names[i]], j, r[names[j]]))
        assert need <= got, (
            f'{name} covering array misses {len(need - got)} of {len(need)} '
            f'pairs, e.g. {sorted(need - got)[:3]}.')
        assert len(rows) <= 4 * len(need) ** 0.5, (
            f'{name} covering array is {len(rows)} rows for {len(need)} pairs '
            f'-- the greedy has degenerated toward one row per pair, which '
            f'makes this file a runtime liability rather than a gate.')
        # and every factor really does vary
        for n in names:
            levels = {r[n] for r in rows}
            assert len(levels) == len(factors[n]), (
                f'{name}: factor {n!r} only ever takes levels {sorted(levels)} '
                f'across the array; one of its levels is never exercised.')


@pytest.mark.parametrize(
    'family,key',
    [('analytic', k) for k in sorted(ANALYTIC_EXCLUSIONS)]
    + [('traced', k) for k in sorted(TRACED_EXCLUSIONS)],
    ids=[f'analytic:{k[0]}={k[1]}x{k[2]}={k[3]}'
         for k in sorted(ANALYTIC_EXCLUSIONS)]
    + [f'traced:{k[0]}={k[1]}x{k[2]}={k[3]}'
       for k in sorted(TRACED_EXCLUSIONS)])
def test_every_declared_exclusion_is_really_refused(family, key):
    """Each declared exclusion must RAISE -- the table cannot hide a defect.

    An exclusion table is a hole in the coverage, so it has to earn its place.
    If the library ever stops refusing one of these combinations, the pair
    belongs back in the array; if it starts returning a silently wrong field
    instead of raising, that is a P1 and this is the test that says so.  The
    assertion is on the REFUSAL, not on its message text, except that the
    message must name the offending kwarg -- ``CONVENTIONS.md`` Section 2
    requires the ``f"{fn_name}: ..."`` prefix and a diagnosable message, and a
    bare ``ValueError()`` here would be a contract regression of its own.
    """
    factors, excl, fn, basefn = (
        (ANALYTIC_FACTORS, ANALYTIC_EXCLUSIONS, apply_real_lens,
         analytic_base_kwargs)
        if family == 'analytic' else
        (TRACED_FACTORS, TRACED_EXCLUSIONS, apply_real_lens_traced,
         traced_base_kwargs))
    fa, la, fb, lb = key
    e_in, dx, rx, _ = lens_covering_array_fixture()
    kw = basefn(dx, rx)
    kw.update(factors[fa][la])
    kw.update(factors[fb][lb])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises((ValueError, NotImplementedError)) as exc:
            fn(e_in, **kw)
    msg = str(exc.value)
    assert msg.startswith(fn.__name__ + ':') or fn.__name__ in msg, (
        f'refusal for {fa}={la} x {fb}={lb} does not name the function '
        f'(CONVENTIONS.md Section 2 error prefix): {msg[:200]!r}')


# ---------------------------------------------------------------------------
# 2. "Knob silently discarded" detector
# ---------------------------------------------------------------------------

def _defaults_of(fn) -> dict:
    import inspect
    return {n: p.default for n, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


# The kwargs whose documented default must be a no-op when passed explicitly.
# Excluded on purpose: ``dy`` (its default None MEANS "= dx", so passing None
# is the same call and passing dx is a different spelling, not a default);
# the private ``_*_out`` diagnostics sinks; and the resource knobs
# (``progress``, ``use_gpu``, ``n_workers``, ``sag_*``, ``accumulator_store``,
# ``scratch_dir``, ``parallel_amp*``, ``amp_use_gpu``) which this file does not
# claim to cover.
_ANALYTIC_DEFAULT_KNOBS = (
    'bandlimit', 'fresnel', 'slant_correction', 'absorption',
    'seidel_correction', 'seidel_poly_order', 'surface_frame',
    'surface_model', 'displaced_mode', 'displaced_obliquity', 'remap_order',
    'carrier', 'screen_obliquity', 'on_screen_obliquity', 'conjugate',
    'wave_propagator', 'stream_transfer_function',
)
_TRACED_DEFAULT_KNOBS = (
    'bandlimit', 'ray_subsample', 'min_coarse_samples_per_aperture',
    'on_undersample', 'preserve_input_phase', 'remap_sampling',
    'tilt_aware_rays', 'carrier', 'on_noncollimated', 'fit_radius_beam_factor',
    'on_aperture_beam', 'on_fit_domain_basis', 'beam_centre',
    'decentred_fit_poly_order', 'newton_amp_mask_rel',
    'newton_mask_dilate_coarse_px', 'newton_max_iters', 'inversion_method',
    'fast_analytic_phase', 'newton_fit', 'newton_poly_order',
    'amplitude_model', 'caustic', 'output_plane_distance',
    'caustic_ray_subsample', 'caustic_band', 'caustic_min_area_ratio',
    'origin', 'inverse_map',
)


def _default_identity(fn, base, knobs, e_in, label, min_compared):
    defaults = _defaults_of(fn)
    missing = [k for k in knobs if k not in defaults]
    assert not missing, (
        f'{label}: {missing} are not keyword parameters of {fn.__name__} any '
        f'more.  The signature changed; update this list rather than deleting '
        f'the coverage.')
    compared = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = _as_field(fn(e_in, **base))
        offenders = []
        for k in knobs:
            kw = dict(base)
            # a kwarg the BASE already pins (ray_subsample) is not a default
            if k in base:
                continue
            compared += 1
            kw[k] = defaults[k]
            try:
                got = _as_field(fn(e_in, **kw))
            except Exception as exc:            # noqa: BLE001 - reported below
                offenders.append((k, defaults[k], f'{type(exc).__name__}: {exc}'))
                continue
            if got.shape != ref.shape or not np.array_equal(got, ref):
                if got.shape != ref.shape:
                    why = f'shape {got.shape} != {ref.shape}'
                else:
                    why = (f'max |diff| = '
                           f'{float(np.max(np.abs(got - ref))):.6e}')
                offenders.append((k, defaults[k], why))
    assert compared >= min_compared, (
        f'{label}: only {compared} kwargs were actually compared (expected at '
        f'least {min_compared}).  The loop is skipping knobs, so a pass here '
        f'means nothing.')
    assert not offenders, (
        f'{label}: passing these kwargs AT THEIR OWN DOCUMENTED DEFAULT does '
        f'not reproduce the call that omits them: '
        + '; '.join(f'{k}={d!r} -> {why}' for k, d, why in offenders)
        + '.  Either the "unset" sentinel and the signature default disagree '
          '(so the documented default is not the behaviour a caller gets), or '
          'the kwarg is routed through a branch that the omitted case skips.')


def test_analytic_kwargs_at_their_default_reproduce_the_bare_call():
    """17 kwargs of ``apply_real_lens``, each passed at its signature default,
    must be byte-identical to omitting it."""
    e_in, dx, rx, _ = lens_covering_array_fixture()
    _default_identity(apply_real_lens, analytic_base_kwargs(dx, rx),
                      _ANALYTIC_DEFAULT_KNOBS, e_in, 'apply_real_lens',
                      min_compared=17)


def test_traced_kwargs_at_their_default_reproduce_the_bare_call():
    """29 kwargs of ``apply_real_lens_traced``, each passed at its signature
    default, must be byte-identical to omitting it."""
    e_in, dx, rx, _ = lens_covering_array_fixture()
    _default_identity(apply_real_lens_traced, traced_base_kwargs(dx, rx),
                      _TRACED_DEFAULT_KNOBS, e_in, 'apply_real_lens_traced',
                      min_compared=28)


def test_the_default_identity_detector_is_not_vacuous():
    """Counter-pin: the detector must FAIL when a knob really is honoured.

    Feeds the same machinery a knob list whose "default" has been replaced by
    a value that is known to change the field (``fresnel=True``, measured to
    move the result on this fixture).  If the comparison were vacuous -- a
    stale reference, a shape-only check, an ``allclose`` with a loose tol --
    this would pass and the two tests above would be worthless.
    """
    e_in, dx, rx, _ = lens_covering_array_fixture()
    base = analytic_base_kwargs(dx, rx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = _as_field(apply_real_lens(e_in, **base))
        moved = _as_field(apply_real_lens(e_in, fresnel=True, **base))
    assert not np.array_equal(moved, ref), (
        'fresnel=True did not change the field at all on this fixture, so the '
        'bit-identity comparison in the two tests above cannot distinguish an '
        'honoured knob from a discarded one.  Pick a different witness.')


# ---------------------------------------------------------------------------
# 3. The MASLOV family (WP-B11a item 10, from WP-B1 request 4)
#
# ``apply_real_lens_maslov`` is the third entry point of the family and had no
# cell in this array at all, so none of its ~25 keyword-only knobs was ever
# paired against another.  It runs on the SAME diverging fixture as the two
# arrays above -- the point of a covering array is that the fixture is held
# fixed while the combination varies.
#
# WHAT IS NOT A FACTOR, and why.  ``integration_method`` is the one knob whose
# levels genuinely change the quadrature, and it is left OUT: MEASURED
# 2026-09-13 on this fixture, ``'auto'`` returns in 1.5 s and the explicit
# ``'quadrature'`` in 72.6 s, so a pairwise array over it would put minutes of
# one entry point's quadrature into the fast lane.  The knob keeps its coverage
# in the default-identity test below (passing ``'auto'`` explicitly is
# byte-identical to omitting it), and its levels belong in a slow-lane file.
# ---------------------------------------------------------------------------

MASLOV_FACTORS: dict[str, list[dict]] = {
    'fit': [{}, {'poly_order': 6}],
    'sampling': [{}, {'ray_field_samples': 24, 'ray_pupil_samples': 24}],
    'linear_phase': [{}, {'extract_linear_phase': False}],
    'normalize': [{}, {'normalize_output': 'none'}],
    'output_plane': [{}, {'output_plane_distance': 2.0e-3}],
    'fold': [{}, {'fold_split': True}],
    'saddle': [{}, {'input_wavevector_saddle': True}],
}

# MEASURED 2026-09-13 by running every level of every factor above on this
# fixture: all 14 return a finite, correctly-shaped field and none raises, so
# the exclusion table is EMPTY -- and empty because it was measured, not
# because nobody looked.  ``test_every_declared_exclusion_is_really_refused``
# has nothing to check here; ``test_the_covering_arrays_really_cover_every_pair``
# still asserts every pair is reached.
MASLOV_EXCLUSIONS: dict[tuple, str] = {}

MASLOV_ROWS = _pairwise_rows(MASLOV_FACTORS, MASLOV_EXCLUSIONS)


def maslov_base_kwargs(dx: float, prescription: dict) -> dict:
    return dict(prescription=prescription, wavelength=WAVELENGTH, dx=dx)


@pytest.mark.parametrize(
    'row', MASLOV_ROWS,
    ids=[_row_label(MASLOV_FACTORS, r) for r in MASLOV_ROWS])
def test_maslov_covering_array_is_finite_and_loses_no_energy(row):
    """Pairwise over 9 physics kwargs of ``apply_real_lens_maslov``.

    The energy bar is the array's own (:func:`_assert_sane`), and it means
    something DIFFERENT on this entry point, which is worth saying: the shipped
    ``normalize_output='power'`` rescales the output to carry exactly the input
    power, so on every arm that leaves it alone ``P_out/P_in`` is 1.000000000
    by construction and the bar is a statement about the RESCALING, not about
    the physics.  The ``normalize`` factor's second level turns it off, and
    that is the arm where the bar has teeth -- MEASURED 0.405890451 there
    (2026-09-13), i.e. the raw Maslov integral on this clipped, diverging
    fixture loses 59 % and gains nothing.
    """
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    e_in, dx, rx, _ = lens_covering_array_fixture()
    kw = maslov_base_kwargs(dx, rx)
    kw.update(_kwargs_for(MASLOV_FACTORS, row))
    label = _row_label(MASLOV_FACTORS, row)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = apply_real_lens_maslov(e_in, **kw)
    _assert_sane(_as_field(out), e_in, dx, label, 'apply_real_lens_maslov')


# The maslov kwargs whose documented default must be a no-op when passed
# explicitly.  Excluded for the same reasons as the other two lists: ``dy``
# (whose default None MEANS "= dx"), the resource knobs (``use_gpu``,
# ``progress``, ``verbose``, ``chunk_v2``, ``use_numexpr``) and ``roi`` /
# ``output_subsample`` (output-shaping, not physics).
_MASLOV_DEFAULT_KNOBS = (
    'ray_field_samples', 'ray_pupil_samples', 'poly_order', 'n_v2',
    'output_plane_distance', 'output_plane_n', 'extract_linear_phase',
    'integration_method', 'stationary_newton_iter', 'stationary_newton_tol',
    'local_n_samples', 'local_window_sigma', 'levin_tol', 'collimated_input',
    'input_na', 'input_wavevector_saddle', 'normalize_output', 'fold_split',
)


def test_maslov_kwargs_at_their_default_reproduce_the_bare_call():
    """18 kwargs of ``apply_real_lens_maslov``, each passed at its signature
    default, must be byte-identical to omitting it.

    This is where ``integration_method='auto'`` keeps its coverage: the knob
    whose explicit levels are too slow for this lane is still pinned at its
    default, which is the arm a caller reaching for a config object
    (``LensConfig.to_kwargs`` splats every field it holds) actually hits."""
    e_in, dx, rx, _ = lens_covering_array_fixture()
    from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
    _default_identity(apply_real_lens_maslov, maslov_base_kwargs(dx, rx),
                      _MASLOV_DEFAULT_KNOBS, e_in, 'apply_real_lens_maslov',
                      min_compared=18)


# ---------------------------------------------------------------------------
# 4. The IN-GLASS GAP LEGS (WP-B11a item 10, from WP-B3b D5)
# ---------------------------------------------------------------------------

def test_the_in_glass_gap_legs_are_reached_and_both_of_them_are_gated():
    """``wave_propagator`` is a covering-array factor with levels ``{}`` (the
    ASM default) and ``'rs'``; NEITHER reaches the in-glass ``'sas'`` /
    ``'fresnel'`` gap legs, so the window-against-period gate those legs carry
    was never exercised anywhere in the lens matrix.  This is that coverage --
    and it is a separate test rather than two more factor levels, because
    MEASURED on this fixture both legs are far outside their kernels' validity
    and the array's energy bar would have to be conceded to hold them.

    MEASURED 2026-09-14 on the covering-array doublet (N = 64, dx = 112.5 um,
    lambda = 632.8 nm, gaps 9.0 and 2.5 mm in N-BAF10 / N-SF6HT):

    ======================  ================  ==========================
    ``wave_propagator``     ``P_out/P_in``    diagnostics emitted
    ======================  ================  ==========================
    default (ASM)           0.996170598       none
    ``'rs'``                0.996170187       none
    ``'fresnel'``           10396.714211      2 x RuntimeWarning
    ``'sas'``               10396.710108      2 x RuntimeWarning
    ======================  ================  ==========================

    Both gap legs gain FOUR DECADES of power, and the geometry is why: the
    single-FFT Fresnel kernel's own validity bound is
    ``max(N dx^2) / lambda_medium = 2.13 m`` against a 9 mm gap, so the chirp is
    aliased by a factor of 240.  It is not a fixture artefact that a finer grid
    removes -- the bound FALLS with dx, so at N = 512 the same fixture reads
    0.142 ('fresnel') and 0.0405 ('sas') of the input power instead, and a
    validly-sampled in-glass Fresnel leg on a 7.2 mm window would need
    N ~ 15 000.

    WHAT MOVED, AND WHY THIS TEST'S NAME CHANGED.  WP-B11a measured the ``sas``
    row at ZERO diagnostics and pinned that silence deliberately, because its
    only validity gate was the FAR direction (``z > z_limit``) while this
    failure is the near one.  WP-B11b closed it: ``sas.py`` now carries
    ``_warn_sas_chirp_sampling``, the same ``z >= N dx^2 / lambda`` bound
    ``fresnel_propagate`` already applied, derived for the SAS kernel's own
    third step and measured to be independent of the ``pad`` factor.  The
    values below are unchanged to every digit WP-B11a recorded -- only the
    diagnostic column moved -- so this test now pins the SYMMETRY of the two
    legs instead of the asymmetry, and the count (2, one per gap) is pinned so
    that a guard firing once, or on the wrong leg, still fails.
    """
    e_in, dx, rx, _ = lens_covering_array_fixture()
    base = analytic_base_kwargs(dx, rx)
    p_in = _power(e_in, dx)
    seen = {}
    for leg in ('fresnel', 'sas'):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = _as_field(apply_real_lens(e_in, wave_propagator=leg, **base))
        assert out.shape == e_in.shape
        assert np.all(np.isfinite(out)), (
            f"wave_propagator={leg!r} returned non-finite samples; the aliased "
            f"quadrature is wrong, but it must not be NaN.")
        seen[leg] = (_power(out, dx) / p_in,
                     [w for w in caught if issubclass(w.category,
                                                      RuntimeWarning)])
    # the two legs are the same aliasing, so they must agree closely
    assert abs(seen['fresnel'][0] - seen['sas'][0]) < 1e-4 * seen['sas'][0], (
        f"the two in-glass gap legs disagree by more than the aliasing they "
        f"share: {seen['fresnel'][0]:.6f} against {seen['sas'][0]:.6f}")
    assert seen['fresnel'][0] > 1e3, (
        f"the 'fresnel' gap leg no longer gains four decades on this fixture "
        f"(P_out/P_in = {seen['fresnel'][0]:.6g}).  Either the leg was fixed "
        f"-- in which case DELETE this assertion and put the two legs back in "
        f"ANALYTIC_FACTORS['propagator'] where they belong -- or the fixture "
        f"moved and this test is no longer measuring the gate.")
    for leg in ('fresnel', 'sas'):
        # one per gap: 9.0 mm of N-BAF10 and 2.5 mm of N-SF6HT, both of them
        # three decades inside the bound.
        assert len(seen[leg][1]) == 2, (
            f"the {leg!r} gap leg emitted {len(seen[leg][1])} RuntimeWarnings, "
            f"not one per under-sampled gap.  That warning is the only thing "
            f"standing between a caller and a four-decade energy gain, and "
            f"the doublet has TWO gaps: "
            f"{[str(w.message)[:80] for w in seen[leg][1]]}")
        assert all('UNDER-SAMPLED' in str(w.message) for w in seen[leg][1]), (
            f"the {leg!r} gap leg warned, but not about the under-sampled "
            f"chirp: {[str(w.message)[:120] for w in seen[leg][1]]}")


if __name__ == '__main__':      # pragma: no cover - measurement helper
    _e, _dx, _rx, _ = lens_covering_array_fixture()
    print('analytic rows: %d, traced rows: %d'
          % (len(ANALYTIC_ROWS), len(TRACED_ROWS)))
    for _factors, _rows, _fn, _basefn, _nm in (
            (ANALYTIC_FACTORS, ANALYTIC_ROWS, apply_real_lens,
             analytic_base_kwargs, 'analytic'),
            (TRACED_FACTORS, TRACED_ROWS, apply_real_lens_traced,
             traced_base_kwargs, 'traced')):
        worst = 0.0
        for _r in _rows:
            _kw = _basefn(_dx, _rx)
            _kw.update(_kwargs_for(_factors, _r))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _f = _as_field(_fn(_e, **_kw))
            _ratio = _power(_f, _dx) / _power(_e, _dx)
            worst = max(worst, _ratio)
            print('  %-10s %-70s P_out/P_in=%.9f finite=%s'
                  % (_nm, _row_label(_factors, _r)[:68], _ratio,
                     bool(np.all(np.isfinite(_f)))))
        print('  %s WORST P_out/P_in = %.12f (excess %.3e)'
              % (_nm, worst, worst - 1.0))
