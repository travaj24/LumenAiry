"""W5 shim-removal wave (v5.30) -- removal pins.

Scope
-----
The owner decided to EXECUTE the scheduled deprecation-shim removals at
v5.30 rather than let them slip to v5.32 a third time.  Ten shims came
out; this file is the single place that pins, for each one:

1. **the old calling form raises the documented error** (and does NOT
   merely warn),
2. **the modern form's output is bit-identical to pre-removal** -- against
   a frozen SHA-256 table captured on the pre-removal commit
   (``24c7d30``), so a "removal" that quietly perturbed the surviving path
   cannot pass, and
3. **no ``DeprecationWarning`` fires anywhere on the modern paths.**

Removal inventory (old form -> new form)
----------------------------------------
``sources/core.py``

* ``create_gaussian_beam(sigma=s)`` -> ``w0=s*sqrt(2)``  (v5.25, stated
  horizon v5.27; ``TypeError`` from the signature)
* Schell-family ``seed=<int>`` -> ``rng=<int>``  (v5.25/v5.27; exactly
  equivalent -- ``seed`` was forwarded verbatim into ``rng``)
* ``Source.gaussian`` / ``plane_wave`` / ``point_source`` / ``top_hat`` /
  ``fiber_mode`` legacy positional overloads -> kwarg-only  (v4.15,
  stated horizon v5.0, re-scheduled v5.32 by R-18)
* ``create_led_source(N, dx, diameter, divergence_angle, wavelength, ...)``
  -> ``create_led_source(N, dx, wavelength, *, diameter=...,
  divergence_angle=...)``  (v4.14.2 / v5.0 / v5.32)
* the Schell ``return_kind`` sentinel apparatus --
  ``_RETURN_KIND_UNSET``, ``_SchellReturnKindUnsetSentinel``,
  ``_warn_schell_return_kind_default`` and five no-op branches  (v4.15.1,
  warning already retired in v4.16.1)

``elements/doe.py``

* ``makedammann2d(_legacy_units='auto')`` -> ``'SI'`` (default) or
  ``'um'``  (v4.14.2; ``ValueError`` naming both)

``propagators/``

* ``recommend_gbd_sampling(wavelength=)`` -> drop it  (audit P8)
* ``propagate_huygens_fresnel_with_opl_callable(wavelength=)`` -> drop it
  (audit P7)

``optimize/``

* ``design_optimize(wave_traced=)`` -> ``register_wave_propagator`` +
  ``wave_propagator=<name>``  (R-17)
* ``MatchIdealSystemMerit(use_traced_lens=, ray_subsample=,
  focus_search=, focus_search_range=, focus_search_n=)`` -> an explicit
  ``real_elements`` / ``ideal_elements`` entry  (R-17); the now-dead
  ``_focus_search_penalty`` helper is deleted with them

Error-shape convention (matched to in-repo precedent)
-----------------------------------------------------
* A kwarg RENAME or an inert kwarg is a **plain signature removal** --
  ``TypeError: ... unexpected keyword argument``.  Precedent:
  ``analysis/detector.py``'s v5.0 ``cosmic_ray_rate`` retirement and
  ``optimize/multiconfig.py``'s v5.0 ``wavelength``-default removal, both
  of which removed the parameter and documented the migration in a
  tombstone comment.
* A shim that intercepted **VALUES** (legacy units, positional overloads)
  keeps an always-raising detector so the error can name the modern form.
  Precedent: ``propagators/system.py``'s ``_reject_legacy`` (v5.0
  aperture-schema purge), which likewise rejects permanently and schedules
  nothing.
* The ``return_kind`` sentinel needed neither: an unrecognised value
  already lands on ``_validate_return_kind``'s ``ValueError``, which names
  ``'ensemble'`` / ``'mcf'``.

Explicitly NOT in scope (kept, with reason)
-------------------------------------------
* The **P5 return-contract transition** (``propagators/dispatch.py``
  sentinel + ``_deprecation.API_TRANSITION_VERSION``) -- a scheduled
  default FLIP, not a removal; it stays scheduled.
* ``MatchIdealSystemMerit(match=)`` and its three non-default metric
  kernels -- a live documented feature that R-17 explicitly declined to
  deprecate.  Grep-verified: only ONE of the four "zero-caller" helpers
  (``_focus_search_penalty``) becomes dead when the flags go.
* ``...with_opl_callable(chunk_output=)`` -- deprecated in v5.17 with NO
  stated horizon ("a future release"), so it is not past one.
* ``rcwa_efficiency_1d_jax`` (v6.0.0), the ``load_zmx_prescription`` /
  ``load_zemax_prescription_txt`` aliases (v6.0), the
  ``output_grid`` -> ``output_shape`` sub-propagator renames (no horizon),
  ``MultiFieldMerit`` scalar ``field_angles`` (no horizon),
  ``PMM2DStack`` (no horizon), and the ``Constraint`` auto-probe notice
  (no horizon) -- none are at or past their stated horizon.
"""
from __future__ import annotations

import hashlib
import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements.doe import makedammann2d
from lumenairy.optimize.driver import (
    WAVE_PROPAGATOR_REGISTRY,
    design_optimize,
)
from lumenairy.optimize.merit_terms import MatchIdealSystemMerit
from lumenairy.propagators.gbd import recommend_gbd_sampling
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable as _hf_opl,
)
from lumenairy.sources.core import (
    Source,
    create_annular_incoherent_source,
    create_fiber_mode,
    create_gaussian_beam,
    create_gaussian_schell_source,
    create_led_source,
    create_point_source,
    create_schell_model_source,
    create_tilted_plane_wave,
    create_top_hat_beam,
)

_WL = 633e-9

# ===========================================================================
# Frozen pre-removal baseline
# ===========================================================================
# SHA-256 over ``str(dtype) + str(shape) + arr.tobytes()`` for every array a
# modern-path recipe below produces, captured by running those exact recipes
# on the pre-removal commit 24c7d30 (v5.29.0 working tree, before any W5
# edit).  Any drift in the surviving code path -- a re-ordered arithmetic
# grouping, a changed default, a lost kwarg -- breaks the digest, which is
# the property "the removal touched only the removed thing".
_BASELINE = {
    'dammann_SI_cell': '4b0706887f8813661f4627945d717b3e499e697d91a56db1dbbd0397df40250f',
    'dammann_SI_nf': 'bebe664b09fa7cf8cf907bd551d2c4398697fa776907a4782312b1f365fb6d21',
    'dammann_default_thz_cell': 'f10a64094109b915bf981f23dcf7fa74640de197e07af48983917bad148a7d53',
    'dammann_um_cell': '4b0706887f8813661f4627945d717b3e499e697d91a56db1dbbd0397df40250f',
    'dammann_um_nf': 'bebe664b09fa7cf8cf907bd551d2c4398697fa776907a4782312b1f365fb6d21',
    'fiber_E': 'e206c454082150798a2ef426a17066b9130b251aa6ae9df88da4ae513a0111c5',
    'gbd_rec_1.00e-05': 'b71cd0d765b53f94bb8a06ebd50cbb71157b0552f6260714982fe6097d9c26a5',
    'gbd_rec_1.55e-06': 'b71cd0d765b53f94bb8a06ebd50cbb71157b0552f6260714982fe6097d9c26a5',
    'gbd_rec_4.00e-07': 'b71cd0d765b53f94bb8a06ebd50cbb71157b0552f6260714982fe6097d9c26a5',
    'gbeam_0_none_E': '29794873889277cb184cffdd9c23a5d27eed8c9bc2f0a59aa6f5d124ea379c55',
    'gbeam_0_none_x': '5b5dd4c7317c4a8156e7238567cc1cbe89b901cff230c911a370a8e76fa1179f',
    'gbeam_0_none_y': '5b5dd4c7317c4a8156e7238567cc1cbe89b901cff230c911a370a8e76fa1179f',
    'gbeam_0_peak_E': '29794873889277cb184cffdd9c23a5d27eed8c9bc2f0a59aa6f5d124ea379c55',
    'gbeam_0_peak_x': '5b5dd4c7317c4a8156e7238567cc1cbe89b901cff230c911a370a8e76fa1179f',
    'gbeam_0_peak_y': '5b5dd4c7317c4a8156e7238567cc1cbe89b901cff230c911a370a8e76fa1179f',
    'gbeam_aniso_E': '850bba73f5f48b946b4d107f3d64c44b0ffe6fc538364fcfcc949a515ddb88be',
    'gbeam_aniso_x': '703edfe516f3c5808a7e00e6e57a7e0634c1c943db6e6c874c14a3dcf284365f',
    'gbeam_aniso_y': '254760319ca4dcdcd0c100dc984aab1a1e9f72900f41606ebf4f39b445d565a0',
    'hf_opl_default': 'f5a210ef7db03e91ff8bd927100a0db10ad692ed8985d997c2550a120c6bb533',
    'hf_opl_no_vv': '51c8e498dcf678aeedd0bf3f0840dd16c17197ae0042e568b02b5bd3558fb083',
    'led_E': 'c87ed8175976b3a8d43b2ca2a060a057fc4e49651d953072f02b82d52bf5fd5b',
    'led_angles': '0aa35949fb9d7da2972ce78d79ccb15f64ea942c8be7dffdaf92efa172b472f3',
    'merit_default': 'cb5ae95bd3fb3d2b263a99fee539a2083ca87855a74cd9cd73722d84721a23dd',
    'merit_field_mse': 'be8bb90443e1b852fdf76648f28681f636c5416cff03189ef377399d966a6d66',
    'merit_field_overlap': 'cb5ae95bd3fb3d2b263a99fee539a2083ca87855a74cd9cd73722d84721a23dd',
    'merit_intensity_mse': '7760bb21417d61b46c5c51accf6daf43c2611e781e05b00e0244b0ac7914798f',
    'merit_intensity_overlap': '7fccd106888c84dbd658fe79ab13d929fc5580956173f509adf7690c3f7c51cd',
    'plane_E': 'e3802f785f5f4e97efd41727879287d2d4242bb772114daace17e1b751805731',
    'point_E': '161b65621a3dd121395af1d1a5462d0716ca36728778a3d4d0053be5b3432b94',
    'schell_a_ensemble': '2da671708fade34f002e3088d18f979bfa60ffa359148dd4708f6bd0fc5216ba',
    'schell_g_ensemble': '5f1222fe40002b170c6e65698253128efb2a31bc1aaeb07c0f67852359b989de',
    'schell_g_mcf': '28e401281a45d24181e9882980080b61e2a834c923331f277702ef73166a1ce9',
    'schell_m_ensemble': 'decaefb1122b0aeb5ab4222cfa64990feabbf5e2e6d6f4648718aa25b89fa109',
    'src_fiber_E': 'e206c454082150798a2ef426a17066b9130b251aa6ae9df88da4ae513a0111c5',
    'src_gaussian_E': '29794873889277cb184cffdd9c23a5d27eed8c9bc2f0a59aa6f5d124ea379c55',
    'src_gschell_ens': '174672035ee2cfe81b39cbc1c72fc942ad75e4390addaaada222cd1e8558fcde',
    'src_mschell_ens': '7a410ad5313f2107558dae84f3cfac5e00dfb20f63dc0c771e8eba4f6f46f1d6',
    'src_plane_E': '7887e49fd24381a21ede3f45bfa2ba979f028b54599e5f8f2d7dcc76151cf8f4',
    'src_point_E': '25a7f23360c8f9676568ba425207a0bbfff92fa0d41aac81e2c785f4147c0c89',
    'src_tophat_E': '902b2211c88d016376842cf2fd50caaa64432a21a463859b18435d333263be72',
    'tophat_E': 'f3585416655886defd224e73b15b2413b39ec8f6827b6bdc987373a46b111cc0',
    'wave_real_lens': '2ae93e6ea3d01601d60935f0e6bd220813d4cc230485fc483cca0135d58da6b5',
}


def _digest(arr) -> str:
    a = np.asarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Shared fixtures for the recipes (mirror the capture script exactly)
# ---------------------------------------------------------------------------

_SCHELL_GRID = dict(N=32, dx=2e-6, wavelength=_WL)
_XX = (np.arange(32) - 16) * 2e-6
_IPROF = np.exp(-(_XX[None, :] ** 2 + _XX[:, None] ** 2) / (30e-6) ** 2)

_DAMMANN_KW = dict(diforders=np.ones((4, 4)), itr=5, plot=False, seed=42,
                   phaselevels=4)

_PRES = {
    'surfaces': [
        {'radius': 50e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': -50e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'},
    ],
    'thicknesses': [3e-3],
    'aperture_diameter': 10e-3,
}


class _Ctx:
    """Minimal ``EvaluationContext`` stand-in for the merit recipes."""
    N = 48
    dx = 4e-6
    wavelength = _WL
    prescription = _PRES
    prescriptions = None
    bfl = 0.03
    efl = 0.02


def _hf_grid():
    n = 12
    return (np.ones((n, n), dtype=np.complex128),
            (np.arange(n) - n / 2) * 2e-6)


def _hf_waves_opl(s1x, s1y, s2x, s2y):
    z = 5e-3
    return (z + ((s2x - s1x) ** 2 + (s2y - s1y) ** 2) / (2 * z)) / _WL


def _curved(lam, n=48, dx=1e-6, w=12e-6):
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    return E * np.exp(1j * np.pi * (X ** 2 + Y ** 2) / (lam * 4e-3))


def _modern_arrays() -> dict:
    """Every modern-path recipe whose output is digest-pinned.

    Deliberately built with the SAME literal arguments as the pre-removal
    capture, and only through call forms that survive the removal.
    """
    out = {}

    # -- sources: create_gaussian_beam via the canonical w0 -----------
    for norm in ('peak', 'none'):
        E, x, y = create_gaussian_beam(64, 1e-6, _WL, w0=12e-6,
                                       normalize=norm)
        out[f'gbeam_0_{norm}_E'] = E
        out[f'gbeam_0_{norm}_x'] = x
        out[f'gbeam_0_{norm}_y'] = y
    E, x, y = create_gaussian_beam((32, 48), 1e-6, _WL, w0=9e-6, dy=2e-6,
                                   x0=3e-6, y0=-2e-6)
    out['gbeam_aniso_E'], out['gbeam_aniso_x'], out['gbeam_aniso_y'] = E, x, y

    out['fiber_E'] = create_fiber_mode(64, 1e-6, 1.31e-6,
                                       mode_field_diameter=9e-6)[0]
    out['tophat_E'] = create_top_hat_beam(64, 1e-6, _WL, diameter=30e-6)[0]
    out['point_E'] = create_point_source(64, 1e-6, _WL, z0=-1e-3)[0]
    out['plane_E'] = create_tilted_plane_wave(64, 1e-6, _WL,
                                              angle_x=0.01)[0]

    # -- LED: canonical keyword-only form ----------------------------
    E, angles, _x, _y = create_led_source(64, 16e-6, 1.31e-6,
                                          diameter=100e-6,
                                          divergence_angle=0.3)
    out['led_E'] = E
    out['led_angles'] = np.asarray(angles, dtype=float)

    # -- Source.*: canonical kwarg-only form -------------------------
    out['src_gaussian_E'] = Source.gaussian(
        N=64, dx=1e-6, wavelength=_WL, w0=12e-6).E
    out['src_plane_E'] = Source.plane_wave(
        N=64, dx=1e-6, wavelength=_WL, angle_x=0.02).E
    out['src_point_E'] = Source.point_source(
        N=64, dx=1e-6, wavelength=_WL, z0=-2e-3).E
    out['src_tophat_E'] = Source.top_hat(
        N=64, dx=1e-6, wavelength=_WL, diameter=25e-6).E
    out['src_fiber_E'] = Source.fiber_mode(
        N=64, dx=1e-6, wavelength=1.31e-6, mode_field_diameter=9e-6).E

    # -- Schell family: canonical rng=, explicit return_kind ---------
    out['schell_g_ensemble'] = create_gaussian_schell_source(
        **_SCHELL_GRID, w0=40e-6, sigma_g=20e-6, n_realizations=4, rng=7,
        return_kind='ensemble')[0]
    mcf = create_gaussian_schell_source(
        **_SCHELL_GRID, w0=40e-6, sigma_g=20e-6, n_realizations=4, rng=7,
        return_kind='mcf')
    out['schell_g_mcf'] = (mcf.J_full if mcf.J_full is not None
                           else mcf.modes)
    out['schell_m_ensemble'] = create_schell_model_source(
        **_SCHELL_GRID, intensity_profile=_IPROF, coherence_length=15e-6,
        n_realizations=4, rng=7, return_kind='ensemble')[0]
    out['schell_a_ensemble'] = create_annular_incoherent_source(
        **_SCHELL_GRID, inner_radius=40e-6, outer_radius=80e-6,
        n_realizations=4, rng=7, return_kind='ensemble')[0]
    out['src_gschell_ens'] = Source.gaussian_schell(
        N=32, dx=5e-6, wavelength=_WL, w0=40e-6, sigma_g=20e-6,
        n_realizations=4, rng=2)[0]
    out['src_mschell_ens'] = Source.schell_model(
        N=32, dx=5e-6, wavelength=_WL, intensity_profile=_IPROF,
        coherence_length=15e-6, n_realizations=4, rng=2)[0]

    # -- doe: 'SI' (default) and explicit 'um' -----------------------
    nf, _ff, cell = makedammann2d(periodx=61e-6, periody=61e-6,
                                  waveln=1.31e-6, _legacy_units='SI',
                                  **_DAMMANN_KW)
    out['dammann_SI_nf'] = nf
    out['dammann_SI_cell'] = np.asarray(cell, dtype=float)
    nf, _ff, cell = makedammann2d(periodx=61.0, periody=61.0, waveln=1.31,
                                  _legacy_units='um', **_DAMMANN_KW)
    out['dammann_um_nf'] = nf
    out['dammann_um_cell'] = np.asarray(cell, dtype=float)
    _nf, _ff, cell = makedammann2d(periodx=8e-3, periody=8e-3,
                                   waveln=1.1e-3, **_DAMMANN_KW)
    out['dammann_default_thz_cell'] = np.asarray(cell, dtype=float)

    # -- gbd: no wavelength kwarg ------------------------------------
    for lam in (0.4e-6, 1.55e-6, 10e-6):
        r = recommend_gbd_sampling(_curved(lam), 1e-6)
        out[f'gbd_rec_{lam:.2e}'] = np.array(
            [r['sample_step'], r['waist_factor'], r['n_beamlets']],
            dtype=float)

    # -- hf: no wavelength kwarg ------------------------------------
    E_in, og = _hf_grid()
    out['hf_opl_default'] = _hf_opl(
        E_in, opl_fn=_hf_waves_opl, output_grid_x=og, output_grid_y=og,
        input_grid_dx=2e-6)
    out['hf_opl_no_vv'] = _hf_opl(
        E_in, opl_fn=_hf_waves_opl, output_grid_x=og, output_grid_y=og,
        input_grid_dx=2e-6, apply_van_vleck=False)

    # -- optimize: default merit + all four match kernels ------------
    out['merit_default'] = np.array([MatchIdealSystemMerit(
        ideal_elements=[{'type': 'lens', 'f': 0.02},
                        {'type': 'propagate', 'z': 0.02}],
        weight=1.0).evaluate(_Ctx())], dtype=float)
    for mt in ('field_overlap', 'field_mse', 'intensity_mse',
               'intensity_overlap'):
        out[f'merit_{mt}'] = np.array([MatchIdealSystemMerit(
            ideal_elements=[{'type': 'lens', 'f': 0.02},
                            {'type': 'propagate', 'z': 0.02}],
            match=mt, weight=1.0).evaluate(_Ctx())], dtype=float)

    # -- optimize: the real_lens registry entry, wave_traced gone ----
    E0 = create_gaussian_beam(48, 4e-6, _WL, w0=40e-6)[0]
    out['wave_real_lens'] = WAVE_PROPAGATOR_REGISTRY['real_lens'](
        E0, _PRES, wavelength=_WL, dx=4e-6, N=48, wp_kwargs={}, opts={})

    return out


@pytest.fixture(scope='module')
def modern():
    with warnings.catch_warnings():
        # Physics-plausibility RuntimeWarnings / the P5 transition notice
        # are not this file's business; the deprecation-silence assertions
        # live in ``TestModernPathsAreWarningClean`` with their own filters.
        warnings.simplefilter('ignore')
        return _modern_arrays()


# ===========================================================================
# (2) the modern form's output is bit-identical to pre-removal
# ===========================================================================

class TestModernPathsAreBitIdenticalToPreRemoval:
    """Frozen-digest guard over every module the W5 wave touched.

    Captured on ``24c7d30`` (pre-removal) by running the identical recipes.
    This is the assertion that distinguishes "removed the shim" from
    "removed the shim and perturbed the survivor": the removals rewrote
    ``_coerce_source_rng``, ``_resolve_gaussian_width``,
    ``create_led_source``'s body head, the five ``Source.*`` classmethod
    heads, three Schell factory bodies, ``makedammann2d``'s unit dispatch,
    ``recommend_gbd_sampling``, ``...with_opl_callable``,
    ``_wave_real_lens`` and ``MatchIdealSystemMerit._build_real_elements``
    / ``_evaluate_one``, and none of those may move a bit.
    """

    def test_every_captured_array_is_present(self, modern):
        assert set(modern) == set(_BASELINE), (
            f'recipe/baseline key mismatch: '
            f'only-recipe={sorted(set(modern) - set(_BASELINE))} '
            f'only-baseline={sorted(set(_BASELINE) - set(modern))}')

    @pytest.mark.parametrize('key', sorted(_BASELINE))
    def test_digest_matches_pre_removal_capture(self, modern, key):
        got = _digest(modern[key])
        assert got == _BASELINE[key], (
            f'{key}: modern-path output changed across the v5.30 shim '
            f'removal (sha256 {got} != captured {_BASELINE[key]}).  The '
            f'removal must be behaviour-neutral on the surviving path.')

    def test_the_guard_is_not_vacuous(self, modern):
        """A digest table only guards if a perturbation breaks it."""
        perturbed = np.array(modern['gbeam_0_none_E'], copy=True)
        perturbed.flat[0] = perturbed.flat[0] * (1 + 2 ** -50)
        assert _digest(perturbed) != _BASELINE['gbeam_0_none_E']


# ===========================================================================
# (1) the old calling form raises the documented error
# ===========================================================================

class TestSourcesCoreRemovals:

    def test_gaussian_beam_sigma_kwarg_raises(self):
        with pytest.raises(TypeError, match='sigma'):
            create_gaussian_beam(32, 1e-6, _WL, sigma=5e-6)

    def test_missing_width_error_names_the_sigma_migration(self):
        """Plain signature removal loses the migration recipe, so the
        surviving missing-argument error carries it for one more cycle."""
        with pytest.raises(TypeError) as info:
            create_gaussian_beam(32, 1e-6, _WL)
        msg = str(info.value)
        assert 'w0' in msg and 'sigma' in msg and 'sqrt(2)' in msg, msg

    @pytest.mark.parametrize('factory, kw', [
        (create_gaussian_schell_source,
         dict(w0=20e-6, sigma_g=10e-6)),
        (create_schell_model_source,
         dict(intensity_profile=_IPROF, coherence_length=10e-6)),
        (create_annular_incoherent_source,
         dict(inner_radius=20e-6, outer_radius=40e-6)),
    ], ids=['gaussian_schell', 'schell_model', 'annular_incoherent'])
    def test_schell_seed_kwarg_raises(self, factory, kw):
        with pytest.raises(TypeError, match='seed'):
            factory(**_SCHELL_GRID, n_realizations=2, seed=7, **kw)

    def test_rng_int_reproduces_the_removed_seed_semantics(self):
        """``seed`` was forwarded verbatim into ``rng``, which is why the
        rename is lossless.  Pinned via the frozen digest (``rng=7``
        matches the pre-removal ``rng=7`` capture bit-for-bit) plus
        self-consistency here."""
        a = create_gaussian_schell_source(
            **_SCHELL_GRID, w0=40e-6, sigma_g=20e-6, n_realizations=4,
            rng=7)[0]
        b = create_gaussian_schell_source(
            **_SCHELL_GRID, w0=40e-6, sigma_g=20e-6, n_realizations=4,
            rng=np.random.default_rng(7))[0]
        assert np.array_equal(a, b)

    @pytest.mark.parametrize('factory, args, new_sig', [
        ('gaussian', (20e-6, 32, 5e-6, _WL),
         'Source.gaussian(*, N, dx, wavelength, w0'),
        ('plane_wave', (32, 5e-6, _WL),
         'Source.plane_wave(*, N, dx, wavelength'),
        ('point_source', (32, 5e-6, _WL),
         'Source.point_source(*, N, dx, wavelength'),
        ('top_hat', (50e-6, 32, 5e-6, _WL),
         'Source.top_hat(*, N, dx, wavelength, diameter'),
        ('fiber_mode', (10e-6, 32, 5e-6, _WL),
         'Source.fiber_mode(*, N, dx, wavelength, mode_field_diameter'),
    ])
    def test_source_positional_overload_raises_and_names_canonical(
            self, factory, args, new_sig):
        with pytest.raises(TypeError) as info:
            getattr(Source, factory)(*args)
        msg = str(info.value)
        assert 'REMOVED in v5.30' in msg, msg
        assert new_sig in msg, msg

    def test_source_classmethods_take_no_positionals_at_all(self):
        """The canonical form is kwarg-only, so the rejection message's
        "takes NONE" claim must be literally true."""
        for name in ('gaussian', 'plane_wave', 'point_source', 'top_hat',
                     'fiber_mode'):
            sig = inspect.signature(getattr(Source, name))
            kinds = {p.kind for p in sig.parameters.values()}
            assert inspect.Parameter.POSITIONAL_OR_KEYWORD not in kinds, name

    def test_led_positional_overload_raises_and_names_canonical(self):
        with pytest.raises(TypeError) as info:
            create_led_source(64, 16e-6, 100e-6, 0.3, 1.31e-6)
        msg = str(info.value)
        assert 'REMOVED in v5.30' in msg, msg
        assert 'create_led_source(N, dx, wavelength, *, diameter=' in msg, msg

    def test_led_canonical_order_positionals_also_raise(self):
        """The removed shim's v4.14.3 scale-inversion guard existed only to
        distinguish this mistake from a legitimate legacy call.  With both
        illegal one rejection covers them."""
        with pytest.raises(TypeError, match='REMOVED in v5.30'):
            create_led_source(64, 16e-6, 1.31e-6, 100e-6, 0.3)

    def test_return_kind_sentinel_apparatus_is_gone(self):
        import lumenairy.sources.core as core
        for name in ('_RETURN_KIND_UNSET',
                     '_SchellReturnKindUnsetSentinel',
                     '_warn_schell_return_kind_default',
                     '_OVERDUE_SHIM_VERSION_REMOVED',
                     '_DEPRECATION_VERSION_REMOVED',
                     '_DEPRECATION_VERSION_ADDED'):
            assert not hasattr(core, name), name

    def test_unrecognised_return_kind_names_the_modern_values(self):
        """Why the sentinel removal needed no bespoke rejection: the
        existing validator already names ``'ensemble'`` / ``'mcf'``."""
        with pytest.raises(ValueError, match="'ensemble' or 'mcf'"):
            create_gaussian_schell_source(
                **_SCHELL_GRID, w0=20e-6, sigma_g=10e-6, n_realizations=2,
                return_kind=object())


class TestDoeLegacyUnitsRemoval:

    _KW = dict(diforders=np.ones((3, 3)), itr=3, plot=False, seed=0)
    _THZ = dict(periodx=8.0e-3, periody=8.0e-3, waveln=1.1e-3)

    def test_auto_mode_raises_and_names_both_survivors(self):
        with pytest.raises(ValueError) as info:
            makedammann2d(**self._THZ, _legacy_units='auto', **self._KW)
        msg = str(info.value)
        assert 'REMOVED in v5.30' in msg, msg
        assert "_legacy_units='um'" in msg, msg
        assert 'SI' in msg, msg

    def test_the_miscompilation_auto_produced_is_unreachable(self):
        """``'auto'`` turned an SI 8 mm / 1.1 mm design into 5e-10 m cells.
        No accepted call may now produce that."""
        _nf, _ff, cell = makedammann2d(**self._THZ, **self._KW)
        assert cell[0] > 1e-5, cell
        _nf, _ff, cell_si = makedammann2d(
            **self._THZ, _legacy_units='SI', **self._KW)
        assert cell_si == cell

    def test_um_mode_is_the_documented_migration(self):
        _nf, _ff, cell = makedammann2d(
            periodx=61.0, periody=61.0, waveln=1.31,
            _legacy_units='um', **self._KW)
        n_ord = int(np.ceil(61e-6 / (0.5 * 1.31e-6) * 0.5)) * 2
        assert cell[0] == pytest.approx(61e-6 / n_ord, rel=1e-12)

    def test_accepted_modes_are_exactly_SI_and_um(self):
        with pytest.raises(ValueError, match="'um', 'SI'"):
            makedammann2d(**self._THZ, _legacy_units='metres', **self._KW)


class TestPropagatorInertKwargRemovals:

    def test_recommend_gbd_sampling_wavelength_raises(self):
        assert 'wavelength' not in inspect.signature(
            recommend_gbd_sampling).parameters
        with pytest.raises(TypeError, match='wavelength'):
            recommend_gbd_sampling(_curved(1.55e-6), 1e-6,
                                   wavelength=1.55e-6)

    def test_gbd_lambda_dependence_still_arrives_through_the_field(self):
        """The measurement that licensed the removal: a physical field at a
        shorter wavelength has a steeper phase gradient, and the
        recommendation follows it -- without a wavelength parameter.

        Uses the W3-P8 probe geometry (N=96, w=25 um, R=4 mm), which is the
        configuration the original audit measured
        (``max|grad phi|`` 2.159e5 -> 5.602e4 -> 8.686e3 rad/m for
        lambda = 0.4 / 1.55 / 10 um)."""
        short = recommend_gbd_sampling(_curved(0.4e-6, n=96, w=25e-6), 1e-6)
        long_ = recommend_gbd_sampling(_curved(10e-6, n=96, w=25e-6), 1e-6)
        assert short['sample_step'] < long_['sample_step'], (
            short, long_)

    def test_hf_opl_callable_wavelength_raises(self):
        assert 'wavelength' not in inspect.signature(_hf_opl).parameters
        E_in, og = _hf_grid()
        with pytest.raises(TypeError, match='wavelength'):
            _hf_opl(E_in, opl_fn=_hf_waves_opl, output_grid_x=og,
                    output_grid_y=og, input_grid_dx=2e-6, wavelength=_WL)

    def test_hf_chunk_output_is_KEPT(self):
        """Scope boundary: ``chunk_output`` was deprecated in v5.17 with NO
        stated horizon, so it is not past one and stays warn-only."""
        E_in, og = _hf_grid()
        assert 'chunk_output' in inspect.signature(_hf_opl).parameters
        with pytest.warns(DeprecationWarning, match='chunk_output'):
            _hf_opl(E_in, opl_fn=_hf_waves_opl, output_grid_x=og,
                    output_grid_y=og, input_grid_dx=2e-6, chunk_output=4)


class TestOptimizeDeadFlagRemovals:

    _IDEAL = [{'type': 'lens', 'f': 50e-3}]

    def test_design_optimize_wave_traced_raises(self):
        assert 'wave_traced' not in inspect.signature(
            design_optimize).parameters
        with pytest.raises(TypeError, match='wave_traced'):
            design_optimize(object(), [], _WL, wave_traced=True)

    @pytest.mark.parametrize('kw, val', [
        ('use_traced_lens', True),
        ('ray_subsample', 8),
        ('focus_search', True),
        ('focus_search_range', (-1e-3, 1e-3)),
        ('focus_search_n', 5),
    ])
    def test_merit_flag_raises(self, kw, val):
        with pytest.raises(TypeError, match=kw):
            MatchIdealSystemMerit(self._IDEAL, **{kw: val})

    def test_only_the_fully_dead_helper_was_deleted(self):
        """Grep-verified scope: ``_focus_search_penalty`` was gated ONLY by
        ``focus_search`` so it dies with the flag; the three metric kernels
        are gated by ``match=``, a live feature, and survive."""
        assert not hasattr(MatchIdealSystemMerit, '_focus_search_penalty')
        for helper in ('_field_overlap_penalty', '_field_mse_penalty',
                       '_intensity_mse_penalty',
                       '_intensity_overlap_penalty'):
            assert callable(getattr(MatchIdealSystemMerit, helper)), helper

    def test_match_kwarg_still_selects_all_four_kernels(self):
        for m in ('field_overlap', 'field_mse', 'intensity_mse',
                  'intensity_overlap'):
            assert MatchIdealSystemMerit(self._IDEAL, match=m).match == m
        with pytest.raises(ValueError, match='match must be one of'):
            MatchIdealSystemMerit(self._IDEAL, match='nope')

    def test_prescription_placeholder_expands_to_plain_real_lens(self):
        """``use_traced_lens`` used to switch the placeholder's element
        type; now it is unconditional."""
        m = MatchIdealSystemMerit(self._IDEAL)
        expanded = m._build_real_elements(_Ctx())
        assert [e['type'] for e in expanded] == ['real_lens']
        assert 'ray_subsample' not in expanded[0]

    def test_registered_propagator_is_the_documented_migration(self):
        """``opts`` still carries ``ray_subsample`` so a user-registered
        traced propagator can honour it."""
        from lumenairy.optimize.driver import (
            register_wave_propagator,
            unregister_wave_propagator,
        )
        seen = {}

        def _probe(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
            seen.update(opts)
            return E0

        register_wave_propagator('_w5_removal_probe', _probe)
        try:
            WAVE_PROPAGATOR_REGISTRY['_w5_removal_probe'](
                None, None, wavelength=_WL, dx=1e-6, N=8, wp_kwargs={},
                opts={'ray_subsample': 7})
        finally:
            unregister_wave_propagator('_w5_removal_probe')
        assert seen == {'ray_subsample': 7}, seen
        assert 'wave_traced' not in seen


# ===========================================================================
# (3) no DeprecationWarning fires anywhere on the modern paths
# ===========================================================================

class TestModernPathsAreWarningClean:

    def test_no_deprecation_warning_on_any_modern_recipe(self):
        """Every surviving call form, in one sweep, under
        ``simplefilter('always')``.  Non-deprecation categories (physics
        plausibility RuntimeWarnings) are allowed through."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _modern_arrays()
        dep = [f'{w.category.__name__}: {w.message}' for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert dep == [], dep

    def test_no_removed_shim_leaves_a_live_horizon_behind(self):
        """The registry must not still advertise the abandoned v5.32
        horizon for anything the W5 wave removed."""
        from lumenairy import _deprecation as dep
        assert dep.REMOVAL_SCHEDULE == {}, dep.REMOVAL_SCHEDULE
        assert dep.check_removal_schedule() == []

    def test_the_p5_transition_is_still_SCHEDULED_not_removed(self):
        """Explicitly out of scope: a default FLIP, not a shim removal."""
        from lumenairy import _deprecation as dep
        cur = dep._version_tuple(la.__version__)
        assert dep._version_tuple(dep.API_TRANSITION_VERSION) > cur
        from lumenairy.propagators import dispatch
        assert 'return_result' in inspect.signature(
            dispatch.propagate).parameters
