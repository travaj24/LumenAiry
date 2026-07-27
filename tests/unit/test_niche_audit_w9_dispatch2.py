"""W9 dispatcher audit, follow-up wave -- pins for the seven approved items.

Companion to ``test_niche_audit_w9_dispatch.py`` (the first wave, commit
268b019).  Every pin here FAILED at 268b019 in a read-only worktree before the
fix in the same change, except where it is explicitly labelled a regression
fence.  Each names the measurement that produced it.

Item map (coordinator's numbering -> audit tag):
  1 dead DOE branch            -> W9-9
  2 twin far-field threshold   -> W9-7
  3 set_default_wave_propagator-> W9-8
  4 traced knobs via the chain -> W9-11
  5 ray_subsample              -> W9-12
  6 hfpi/asymptotic kwargs     -> W9-10
  7 universal router guard     -> W9-13
"""
import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.dispatch import (
    _auto_select_method,
    _select_asm_variant,
    propagate,
    which_propagator,
)
from lumenairy.propagators.system import propagate_through_system

LAM = 633e-9
WL_IR = 1.31e-6


def _gauss(N, dx, w0):
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _circ(N, dx, D):
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return ((X * X + Y * Y) <= (0.5 * D) ** 2).astype(np.complex128)


def _singlet():
    return la.make_singlet(R1=0.05, R2=-0.05, d=3e-3, glass='N-BK7',
                           aperture=2e-3)


# ===========================================================================
# W9-7 (item 2) -- ONE set of free-space regime thresholds in the module.
# ===========================================================================
#
# Pre-fix ``_select_asm_variant`` carried its own trip points -- ``Q > 20`` ->
# fraunhofer, ``Q > 2`` -> sas, with ``Q = lambda|z|/(N dx^2)`` -- while
# ``_auto_select_method`` used ``N_F < 0.1`` -> fraunhofer, ``Q > 1`` -> sas.
# Since ``N_F * Q = N/4``, the ASM-family fraunhofer trip sat at aperture
# Fresnel number ``N/80``: it fires further INSIDE the near field the bigger the
# grid.  MEASURED just above the old trip (z = 20.05 * L^2/(N*lam), hard
# circular aperture filling half the grid, dx = 2 um, lambda = 633 nm), complex
# overlap fidelity against a pad-converged EXACT angular_spectrum_propagate_mft
# on the central 8x8 patch of each candidate's own output grid:
#
#     N     N_F(ap)   fid(fraunhofer)   fid(sas)
#     128   0.399     0.9516            1.00000
#     256   0.798     0.8185            1.00000
#     512   1.596     0.4111            1.00000
#     1024  3.192     0.4241            1.00000

_FARFIELD_FIDELITY_TABLE = (
    # N, measured fid(fraunhofer) at z = 20.05 * L^2/(N*lam)
    (128, 0.9516),
    (256, 0.8185),
    (512, 0.4111),
)


def _threshold(N, dx, lam=LAM):
    return (N * dx) ** 2 / (N * lam)


@pytest.mark.parametrize('N', [64, 128, 256, 512, 1024])
@pytest.mark.parametrize('Qmult', [0.5, 1.05, 2.0, 5.0, 20.05, 200.0, 3000.0])
def test_the_two_selectors_agree_everywhere(N, Qmult):
    """W9-7: there is exactly ONE regime rule in the module.  Any geometry the
    ASM-family selector can see must get the same free-space answer as the
    canonical selector."""
    dx = 2e-6
    E = np.ones((N, N), dtype=np.complex128)
    z = Qmult * _threshold(N, dx)
    canonical = _auto_select_method(E, z=z, wavelength=LAM, dx=dx,
                                    prescription=None)
    family = _select_asm_variant(E, z, LAM, dx)
    assert family == canonical, (
        f'W9-7: N={N}, Q={Qmult}: canonical says {canonical!r}, ASM-family '
        f'says {family!r} -- two regime rules in one module is the defect.')


@pytest.mark.parametrize('N', [128, 256, 512, 1024])
def test_old_far_field_trip_no_longer_selects_fraunhofer(N):
    """W9-7: at the OLD trip (Q = 20.05) the aperture Fresnel number is N/320
    for a half-grid aperture -- 0.4 at N=128 rising to 3.2 at N=1024 -- i.e.
    near field.  Fraunhofer must not be chosen there any more."""
    dx = 2e-6
    E = _circ(N, dx, 0.5 * N * dx)
    z = 20.05 * _threshold(N, dx)
    assert _select_asm_variant(E, z, LAM, dx) == 'sas'
    assert which_propagator(E, z, LAM, dx)['method'] == 'sas'


@pytest.mark.parametrize('N,fid_fraunhofer', _FARFIELD_FIDELITY_TABLE)
def test_the_routed_member_beats_the_old_one_against_an_exact_oracle(
        N, fid_fraunhofer):
    """W9-7 physics: at the old trip point the NEW routing (sas) must agree
    with an exact oracle far better than the OLD routing (fraunhofer) did.

    Oracle: ``angular_spectrum_propagate_mft`` -- exact band-limited ASM on an
    arbitrary output grid -- evaluated from a ZERO-PADDED input so the probe
    patch sits far inside the Bluestein spatial period (no replica aliasing),
    and checked for pad-convergence in the same call.

    Budget: the routed member must clear 0.99 while the old choice measured
    <= 0.96.  MEASURED sas fidelity is 1.00000 to five places at every N, so
    the 0.99 bar has ~1e4x headroom in deficit terms; the bar is deliberately
    far below the measurement and far above the old member's best (0.9516).
    """
    from lumenairy.propagators.mft import angular_spectrum_propagate_mft
    from lumenairy.propagators.propagation import (
        fraunhofer_propagate,
        scalable_angular_spectrum_propagate,
    )
    dx, npatch = 2e-6, 8
    E = _circ(N, dx, 0.5 * N * dx)
    z = 20.05 * _threshold(N, dx)

    def pad_to(A, Np):
        out = np.zeros((Np, Np), dtype=A.dtype)
        o = Np // 2 - A.shape[-1] // 2
        out[o:o + A.shape[-1], o:o + A.shape[-1]] = A
        return out

    def fid(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        return float(abs(np.vdot(a, b))
                     / (np.linalg.norm(a) * np.linalg.norm(b)))

    def centre(A, n):
        c = A.shape[-1] // 2
        return A[c - n // 2: c + n - n // 2, c - n // 2: c + n - n // 2]

    Np = max(4 * N, 1024)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Ep = pad_to(E, Np)
        Es, dxs, _ = scalable_angular_spectrum_propagate(E, z, LAM, dx)
        Ef, dxf, _ = fraunhofer_propagate(E, z, LAM, dx)
        t_s = angular_spectrum_propagate_mft(Ep, z, LAM, dx, dxs, npatch)
        t_f = angular_spectrum_propagate_mft(Ep, z, LAM, dx, dxf, npatch)
        t_s2 = angular_spectrum_propagate_mft(pad_to(E, 2 * Np), z, LAM, dx,
                                              dxs, npatch)
    assert fid(t_s, t_s2) > 1 - 1e-6, 'oracle is not pad-converged'

    f_new = fid(centre(Es, npatch), t_s)
    f_old = fid(centre(Ef, npatch), t_f)
    assert _select_asm_variant(E, z, LAM, dx) == 'sas'
    assert f_new > 0.99, (
        f'W9-7: the newly-routed member scored {f_new:.5f} at N={N}')
    assert f_old < 0.97, (
        f'W9-7: the OLD choice was expected to be poor here (recorded '
        f'{fid_fraunhofer:.4f}); measured {f_old:.5f} -- if this now passes, '
        f'the oracle or the probe has drifted, not the router.')


def test_far_field_still_reachable_when_it_is_really_far_field():
    """W9-7 fence: the fraunhofer branch must still exist and fire once the
    canonical criterion (grid N_F < 0.1) is genuinely met."""
    N, dx = 64, 2e-6
    E = _circ(N, dx, 0.5 * N * dx)
    z = 3000.0 * _threshold(N, dx)
    assert (0.5 * dx * N) ** 2 / (LAM * z) < 0.1
    assert _select_asm_variant(E, z, LAM, dx) == 'fraunhofer'


def test_asm_family_specific_branches_still_outrank_the_regime():
    """W9-7 fence: delegation must not swallow the three branches that are this
    selector's own (output grid, tilt, back-propagation)."""
    N, dx = 64, 2e-6
    E = _gauss(N, dx, 20e-6)
    z = 3000.0 * _threshold(N, dx)          # deep far field
    assert _select_asm_variant(E, z, LAM, dx, output_dx=3e-6) == 'asm_mft'
    assert _select_asm_variant(E, z, LAM, dx, tilt_x=0.05) == 'asm_tilted'
    assert _select_asm_variant(E, -z, LAM, dx) == 'asm'


def test_canonical_docstring_says_so_and_the_family_one_defers():
    """W9-7: the decision record has to live in the code, not only in a commit
    message -- both docstrings must name which selector owns the regime."""
    from lumenairy.propagators import dispatch
    canon = inspect.getdoc(dispatch._auto_select_method) or ''
    family = inspect.getdoc(dispatch._select_asm_variant) or ''
    assert 'CANONICAL' in canon
    assert '_auto_select_method' in family and 'canonical' in family.lower()


# ===========================================================================
# W9-8 (item 3) -- propagate() honours set_default_wave_propagator().
# ===========================================================================


@pytest.fixture
def _knob():
    """Save/restore the library-wide wave-propagator default."""
    from lumenairy.propagators.propagation import (
        get_default_wave_propagator,
        set_default_wave_propagator,
    )
    saved = get_default_wave_propagator()
    yield set_default_wave_propagator
    set_default_wave_propagator(saved)


def _fs(dx=2e-6, N=64):
    return _gauss(N, dx, 20e-6), dx


def test_unset_method_honours_the_library_default(_knob):
    """W9-8: with the knob moved off its shipped value, a free-space call that
    passes no ``method`` uses it -- mirroring propagate_through_system."""
    E, dx = _fs()
    _knob('fresnel')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        near = propagate(E, z=1e-4, wavelength=LAM, dx=dx)
        far = propagate(E, z=5.0, wavelength=LAM, dx=dx)
    assert near.method == 'fresnel'
    assert far.method == 'fresnel'


def test_explicit_auto_always_auto_selects(_knob):
    """W9-8: ``method='auto'`` is a REQUEST, not silence -- it must out-rank the
    library default."""
    E, dx = _fs()
    _knob('fresnel')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, z=5.0, wavelength=LAM, dx=dx, method='auto')
    assert res.method == 'fraunhofer'


def test_rayleigh_sommerfeld_default_resolves_here(_knob):
    """W9-8: propagate() supports 'rs' natively (the chain rejects it), so both
    spellings of the RS default resolve rather than raise."""
    E, dx = _fs()
    for name in ('rs', 'rayleigh_sommerfeld'):
        _knob(name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = propagate(E, z=1e-4, wavelength=LAM, dx=dx)
        assert res.method == 'rs', f'{name!r} -> {res.method!r}'


def test_prescription_calls_keep_auto_selection(_knob):
    """W9-8: the knob names FREE-SPACE kernels.  MEASURED,
    ``propagate(prescription=rx, method='asm')`` returns the input UNCHANGED
    (z is None, so ASM takes its copy fast path and the prescription is
    ignored), so honouring the knob on a prescription call would make it a
    silent no-op."""
    E = _gauss(64, 40e-6, 5e-4)
    rx = _singlet()
    _knob('fresnel')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, wavelength=WL_IR, dx=40e-6, prescription=rx)
        ignored = propagate(E, wavelength=WL_IR, dx=40e-6, prescription=rx,
                            method='asm')
    assert res.method == 'maslov'
    assert not np.array_equal(np.asarray(res.field), E)
    # the reason the knob is not applied here, pinned:
    assert np.array_equal(np.asarray(ignored.field), E)


def test_shipped_default_means_no_preference_and_is_self_restoring(_knob):
    """W9-8: resolution is STATELESS -- restoring the knob to its shipped value
    restores auto-selection, with no latch left behind."""
    E, dx = _fs()
    _knob('fresnel')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert propagate(E, z=5.0, wavelength=LAM, dx=dx).method == 'fresnel'
    _knob('asm')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert propagate(E, z=5.0, wavelength=LAM,
                         dx=dx).method == 'fraunhofer'


def test_a_knob_resolved_method_is_treated_as_named_not_as_auto(_knob):
    """W9-8 x W9-1 interaction, recorded: the wave-1 re-route of
    ``auto`` + output-grid away from ``sas`` applies to AUTO only.  A caller
    who selected sas through the library knob has named it (process-wide), so
    they get the same ValueError an explicit ``method='sas'`` gets -- which
    points at ``method='asm'``."""
    E, dx = _fs()
    _knob('sas')
    with pytest.raises(ValueError, match='SAS does not support'):
        propagate(E, z=1e-3, wavelength=LAM, dx=dx, output_dx=3e-6)


def test_untouched_knob_leaves_auto_selection_bit_for_bit():
    """W9-8 fence: with nobody touching the knob, the documented routing table
    is exactly what it was."""
    E, dx = _fs()
    table = {1e-5: 'asm', 1e-4: 'asm', 1e-3: 'sas', 1e-2: 'sas',
             0.1: 'fraunhofer', 5.0: 'fraunhofer'}
    for z, want in table.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            assert propagate(E, z=z, wavelength=LAM, dx=dx).method == want


# ===========================================================================
# W9-9 (item 1) -- the dead DOE branch is gone; DOE kwargs are handled.
# ===========================================================================
#
# MEASURED: ``events_json`` occurs exactly ONCE in the repository at 268b019 --
# in dispatch.py itself.  No loader or factory emits it, so the branch could not
# fire; forced, it raised ``TypeError: propagate_hfpi_through_prescription()
# missing 1 required keyword-only argument: 'n_paths'``.  It could not be
# repointed either: this library has NO prescription-embedded DOE
# representation -- diffractive data travels as the ``surface_diffraction`` /
# ``diffracting_surfaces`` KWARGS, accepted only by hfpi and (via fit_kwargs)
# asymptotic.  And there was no measured case for routing to hfpi
# automatically: on a thin air-to-air grating whose analytic exit centroid is
# ``t*tan(asin(m*lam/Lambda))``, hfpi WITH surface_diffraction missed the
# order-1 deflection by 85-97%, no better than maslov's 100%.

_DOE_SD = {0: (1, 0, 20e-6, 20e-6)}


def test_events_json_is_gone_from_the_source():
    """W9-9: the dead trigger must not survive as dead code."""
    from lumenairy.propagators import dispatch
    src = inspect.getsource(dispatch._auto_select_method)
    code = src.split('"""', 2)[-1]      # body only; the note may cite the name
    assert "prescription.get('events_json')" not in code
    assert "== 'doe'" not in code


def test_a_doe_carrying_prescription_no_longer_pretends_to_route():
    """W9-9: a hand-injected ``events_json`` key is now inert -- it must not
    resurrect a route to a call that immediately TypeErrors."""
    E = _gauss(64, 40e-6, 5e-4)
    rx = dict(_singlet())
    rx['events_json'] = [{'type': 'doe', 'period': 10e-6}]
    assert _auto_select_method(E, z=None, wavelength=WL_IR, dx=40e-6,
                               prescription=rx) == 'maslov'


@pytest.mark.parametrize('kw', [{'surface_diffraction': _DOE_SD},
                                {'diffracting_surfaces': [0]}])
def test_doe_kwargs_on_a_non_doe_member_raise_from_the_dispatcher(kw):
    """W9-9: MEASURED pre-fix, ``propagate(prescription=rx,
    surface_diffraction=...)`` auto-selected maslov and died with
    ``TypeError: apply_real_lens_maslov() got an unexpected keyword argument``
    -- a kernel the caller never named."""
    E = _gauss(64, 40e-6, 5e-4)
    with pytest.raises(ValueError) as info:
        propagate(E, wavelength=WL_IR, dx=40e-6, prescription=_singlet(), **kw)
    msg = str(info.value)
    assert 'hfpi' in msg and 'diffractive' in msg
    assert 'prescription' in msg     # explains why 'auto' cannot detect one


def test_doe_kwargs_reach_the_member_that_accepts_them():
    """W9-9: the diagnostic must come with a route that works."""
    E = _gauss(64, 40e-6, 5e-4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, wavelength=WL_IR, dx=40e-6, prescription=_singlet(),
                        method='hfpi', n_paths=300,
                        surface_diffraction=_DOE_SD)
    assert res.method == 'hfpi'
    assert res.field is not None


# ===========================================================================
# W9-10 (item 6) -- required kernel kwargs are named by propagate(), not by a
#                   kernel the caller never called.
# ===========================================================================


@pytest.mark.parametrize('kw,missing', [
    (dict(method='hfpi'), ['n_paths']),
    (dict(method='asymptotic'), ['s2_grid_x', 's2_grid_y']),
])
def test_prescription_members_name_every_missing_required_kwarg(kw, missing):
    E = _gauss(64, 40e-6, 5e-4)
    with pytest.raises(ValueError) as info:
        propagate(E, wavelength=WL_IR, dx=40e-6, prescription=_singlet(), **kw)
    msg = str(info.value)
    assert msg.startswith(f"propagate(method={kw['method']!r})")
    for name in missing:
        assert name in msg, f'{name!r} not named in {msg!r}'


def test_hfpi_freespace_names_all_four_not_just_aperture_radius():
    """W9-10: the old check advertised ``aperture_radius`` alone, so supplying
    exactly that still produced ``TypeError:
    propagate_hfpi_freespace_aperture() missing 3 required keyword-only
    arguments`` (MEASURED)."""
    E = _gauss(64, 40e-6, 5e-4)
    with pytest.raises(ValueError) as info:
        propagate(E, wavelength=WL_IR, dx=40e-6, method='hfpi',
                  aperture_radius=5e-4)
    msg = str(info.value)
    for name in ('z_to_aperture', 'z_aperture_to_output', 'n_paths'):
        assert name in msg, f'{name!r} not named in {msg!r}'


def test_supplying_them_still_runs():
    """W9-10 fence: the check must not become a wall."""
    E = _gauss(64, 40e-6, 5e-4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, wavelength=WL_IR, dx=40e-6, prescription=_singlet(),
                        method='hfpi', n_paths=300)
    assert res.method == 'hfpi' and res.field is not None


def test_no_invented_defaults_for_the_accuracy_knobs():
    """W9-10: ``n_paths`` (a Monte-Carlo budget) and ``s2_grid_*`` (output
    grids) must NOT be silently defaulted -- any value the dispatcher picked
    would be an unannounced accuracy decision."""
    from lumenairy.propagators.dispatch import _REQUIRED_METHOD_KWARGS
    names = {n for spec in _REQUIRED_METHOD_KWARGS.values() for n in spec[0]}
    assert {'n_paths', 's2_grid_x', 's2_grid_y'} <= names


# ===========================================================================
# W9-11 (item 4) -- traced knobs reach the element through the chain, and
#                   unknown keys are loud.
# ===========================================================================
#
# MEASURED pre-fix: output BIT-IDENTICAL to omitting the key for all nine of
# amplitude_model / preserve_input_phase / remap_sampling /
# fit_radius_beam_factor / carrier / on_undersample / n_workers / traced_kwargs
# and an outright typo key.  The v5.29 + S12 validated traced configuration was
# therefore unreachable through this chain API.

_SYS_N, _SYS_DX = 96, 40e-6


def _sys_field():
    return _gauss(_SYS_N, _SYS_DX, 8e-4)


def _traced_elem(**over):
    el = {'type': 'real_lens_traced', 'prescription': _singlet()}
    el.update(over)
    return el


def _run_sys(elem):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_out, _ = propagate_through_system(
            _sys_field(), [elem], WL_IR, _SYS_DX)
    return np.asarray(E_out)


@pytest.mark.parametrize('key,value', [
    ('amplitude_model', 'ray_density'),
    ('fit_radius_beam_factor', 2.0),
])
def test_traced_knobs_change_the_output_through_the_chain(key, value):
    """W9-11: knobs that MUST change the result now do."""
    base = _run_sys(_traced_elem())
    got = _run_sys(_traced_elem(**{key: value}))
    assert not np.array_equal(base, got), (
        f'W9-11: elements[0][{key!r}]={value!r} is still being dropped')


def test_the_validated_configuration_is_reachable_as_a_bag():
    """W9-11: the exact dict ``propagate_traced_carrier_chain`` applies by
    default must be expressible on a chain element."""
    base = _run_sys(_traced_elem())
    got = _run_sys(_traced_elem(traced_kwargs={
        'amplitude_model': 'ray_density',
        'preserve_input_phase': 'remap',
        'remap_sampling': 'full',
        'fit_radius_beam_factor': 2.0,
    }))
    assert not np.array_equal(base, got)
    assert np.all(np.isfinite(got))


@pytest.mark.parametrize('key,value', [
    ('remap_sampling', 'not_a_mode'),
    ('inversion_method', 'not_a_method'),
    ('on_aperture_beam', 'shout'),
])
def test_forwarded_knobs_really_reach_the_element(key, value):
    """W9-11: prove FORWARDING (not just "output differs") for knobs that are
    inert on this probe -- give them values the ELEMENT itself rejects and
    check the element is what complains."""
    with pytest.raises(ValueError, match='apply_real_lens_traced'):
        _run_sys(_traced_elem(**{key: value}))


@pytest.mark.parametrize('key', ['typo_key_that_does_not_exist',
                                 'amplitude_modle', 'dx', 'wavelength'])
def test_unknown_or_managed_keys_raise(key):
    """W9-11: silence on an unrecognised key is the forbidden class -- the same
    one the v5.30 P6 twin fix closed for ``method``."""
    with pytest.raises(ValueError) as info:
        _run_sys(_traced_elem(**{key: 1.0}))
    msg = str(info.value)
    assert key in msg and 'real_lens_traced' in msg
    assert 'traced_kwargs' in msg     # names the accepted spellings


def test_traced_kwargs_must_be_a_dict():
    with pytest.raises(ValueError, match='traced_kwargs'):
        _run_sys(_traced_elem(traced_kwargs=['amplitude_model']))


def test_plain_traced_element_still_works():
    """W9-11 fence: the minimal element is untouched."""
    out = _run_sys(_traced_elem())
    assert out.shape == (_SYS_N, _SYS_N) and np.all(np.isfinite(out))


# ===========================================================================
# W9-12 (item 5) -- the ray_subsample docstring contradiction.
# ===========================================================================


def test_the_docstring_no_longer_recommends_against_the_code():
    """W9-12: it read "default 1; 4 is the recommended production value" while
    the code hard-coded 1 -- a docstring arguing with the line beneath it."""
    doc = inspect.getdoc(propagate_through_system) or ''
    assert '``ray_subsample`` defaults to **1** here' in doc, (
        'W9-12: the element docstring must state the default it actually has')
    # the old wording may only survive as quoted history, never as the spec
    flat = ' '.join(doc.split())
    old = 'default 1; 4 is the recommended production value'
    if old in flat:
        assert 'used to read' in flat, (
            'W9-12: the old contradictory wording is still presented as the '
            'specification rather than as quoted history')
    assert 'min_coarse_samples_per_aperture' in doc, (
        'W9-12: the docstring must say WHY 4 is not the default')


def test_ray_subsample_default_is_documented_and_honoured():
    """W9-12: the default stays 1 (measured: flipping to 4 breaks coarse-grid
    chains via ``min_coarse_samples_per_aperture`` for no fidelity gain), and
    4 is now reachable."""
    from lumenairy.propagators.system import (
        _TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT,
        _resolve_traced_element_kwargs,
    )
    assert _TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT == 1
    assert _resolve_traced_element_kwargs(
        _traced_elem(), 0)['ray_subsample'] == 1
    assert _resolve_traced_element_kwargs(
        _traced_elem(ray_subsample=4), 0)['ray_subsample'] == 4
    assert _resolve_traced_element_kwargs(
        _traced_elem(traced_kwargs={'ray_subsample': 8}),
        0)['ray_subsample'] == 8


def test_the_three_entry_point_defaults_are_recorded():
    """W9-12: element 8 / chain 4 / chain-element 1 is deliberate, so pin the
    three so a silent drift is caught."""
    from lumenairy.elements.lenses import apply_real_lens_traced
    from lumenairy.propagators.carrier import propagate_traced_carrier_chain
    from lumenairy.propagators.system import (
        _TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT,
    )
    el = inspect.signature(apply_real_lens_traced).parameters[
        'ray_subsample'].default
    ch = inspect.signature(propagate_traced_carrier_chain).parameters[
        'ray_subsample'].default
    assert (el, ch, _TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT) == (8, 4, 1)


# ===========================================================================
# W9-13 (item 7) -- the universal router adopts the P2 cliff guard.
# ===========================================================================
#
# MEASURED (E4 corrected relay, N=1536, dx=7 um, beam w=2 mm, every other option
# at the ELEMENT defaults -- i.e. exactly this router's configuration --
# exit-wavefront Strehl in the carrier-referenced envelope):
#
#     aperture     1.50x beam   1.75x beam   2.50x beam
#     frbf=None        0.9701       0.1085       0.0384
#     frbf=2.0         0.9874       0.9820       0.9816
#
# The other three validated options are NOT adopted: they are carrier-regime
# options and this router supplies no carrier.  Measured single-element,
# no-carrier deltas: -0.0025 / -0.0249 / -0.1912 at 1.0x / 1.5x / 1.75x beam.


def _universal_traced_call(**mkw):
    """Capture the kwargs the router hands to apply_real_lens_traced."""
    from lumenairy.propagators import fga
    seen = {}

    def _spy(E_in, **kwargs):
        seen.update(kwargs)
        return np.asarray(E_in)

    E = _gauss(128, 40e-6, 1.2e-3)
    rx = la.make_singlet(R1=0.018, R2=-0.018, d=3e-3, glass='N-BK7',
                         aperture=5e-3)
    real = fga.apply_real_lens_universal
    import lumenairy.elements as _el
    orig = _el.apply_real_lens_traced
    _el.apply_real_lens_traced = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            real(E, prescription=rx, wavelength=WL_IR, dx=40e-6,
                 method='traced', **mkw)
    finally:
        _el.apply_real_lens_traced = orig
    return seen


def test_universal_router_defaults_the_traced_cliff_guard_on():
    """W9-13: the router used to inherit the element's ``None`` and route
    high-NA collimated beams straight into the measured cliff."""
    seen = _universal_traced_call()
    assert seen.get('fit_radius_beam_factor') == 2.0, (
        f'W9-13: router handed fit_radius_beam_factor='
        f'{seen.get("fit_radius_beam_factor")!r}')


def test_the_guard_is_still_opt_out_able():
    seen = _universal_traced_call(
        method_kwargs={'traced': {'fit_radius_beam_factor': None}})
    assert seen.get('fit_radius_beam_factor', 'ABSENT') is None


def test_the_carrier_regime_options_are_not_adopted():
    """W9-13: only the regime-INDEPENDENT guard is adopted.  The other three
    are carrier-regime options and measured equal-or-worse without a carrier."""
    seen = _universal_traced_call()
    for key in ('amplitude_model', 'preserve_input_phase', 'remap_sampling'):
        assert key not in seen, (
            f'W9-13: {key!r} is a carrier-regime option; this router supplies '
            f'no carrier and must leave it at the element default')
