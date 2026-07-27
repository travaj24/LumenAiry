"""Regression pins for the Territory-P MEDIUM/LOW batch of
``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`` plus the one NEW
finding recorded in that report's status banner.  Every pin below was
verified to FAIL on a pre-fix worktree at ``7ea2eb9``.

* **P5** ``dispatch.py`` -- ``propagate(method='auto')`` returns a bare
  ndarray at the input pitch OR an ``(E, dx_out, dy_out)`` triple at a
  kernel-chosen pitch depending purely on ``z`` (measured: ``z=1e-4`` ->
  ndarray at dx=2.0000e-06; ``z=1e-3`` -> tuple at 2.4727e-06; ``z=5``
  -> tuple at 2.4727e-02), with zero warnings.  Return types are
  deliberately UNCHANGED (deferred API decision); the dispatcher now
  warns and documents the full table.
* **NEW (P6's NumPy twin)** ``system.py`` -- ``propagate_through_system``
  silently fell through to ASM for any unrecognised ``method``
  (measured 0.0 relative difference vs ``method='asm'`` for
  ``'not_a_method'``, ``'ASM'``, ``'gbd'``, ``'fraunhofer'``), and so did
  the per-element ``elem['method']`` override.  Both now raise.
* **P8** ``gbd.py`` (keyword REMOVED in v5.30 / W5; pins superseded
  below) -- ``recommend_gbd_sampling(wavelength=)`` was a
  required keyword the body never read (identical output for lambda =
  0.4 um and 10 um).  Now optional + deprecated (v5.32): the
  wavelength dependence is data-driven, and the sibling
  ``converge_gbd_sampling`` measures the SAME optimal overlap at
  lambda = 0.633 / 1.55 / 3.0 um, so no lambda-dependent divergence
  term was warranted.
* **P9** ``fga.py`` -- ``fga_memory_estimate(nsig=)`` was documented "as
  on apply_real_lens_fga" but never read, while the real path reads
  ``nsig`` 6x: the whole estimate was bit-identical for nsig = 1.0 and
  12.0 at two grid sizes.  ``nsig`` now sizes a new ``gabor_window_mb``
  pool.
* **P10** ``subaperture.py`` -- ``patches_for_box(centred=)`` was
  completely inert (True and False returned bit-identical grids) AND the
  one layout they shared was asymmetric by ``w*overlap/2``: measured
  x-span [-50.000, +75.000] um over a +/-50 um box.  ``centred=True``
  is now symmetric; ``centred=False`` keeps the legacy layout.
* **P11** ``mft.py`` -- the caller-specified output grid was unvalidated
  (``dx_out=-1e-6`` accepted with a finite mirrored result;
  ``dx_out=nan`` accepted with an all-NaN result; ``dx_out=0`` a bare
  ``ZeroDivisionError``) and a window wider than one transform period
  returned silent periodic replicas.
* **P12** ``asm.py`` / ``rs.py`` / ``fft_infra.py`` / ``mft.py`` -- the
  band-limit cutoff ``L/(2*lambda*|z|)`` was cited as Matsushima &
  Shimobaba's criterion; it is the ``z -> infinity`` asymptote of it.
  Docstrings corrected, cutoff deliberately unchanged.
* **P13** dead parameters / guards: ``rs.py``'s ``and z != 0`` conjunct
  under an enforced ``z > 0``; ``_promote_entry_to_measure(entry, ...)``;
  ``_fga_coarse(..., coeff_frac, ...)``.  The ASM / ASM-MFT ``z != 0``
  conjuncts are LIVE (z=0 is the exact identity) and are pinned as such.
* **P14** ``hf.py`` -- the ``beam_d4sigma`` fallback built its x-grid
  from the OUTPUT ``Nx`` and broadcast it against the input-shaped
  field: uncaught ``ValueError: operands could not be broadcast together
  with shapes (32,32) (64,)`` whenever ``output_shape != E_in.shape``.
* **P16** ``result.py`` -- ``PropagationResult.__iter__`` yields 2 items
  where the un-wrapped kernels unpack 3.  Behaviour is pinned; the
  hazard is now documented prominently.
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
import pytest

LAM = 633e-9


def _gauss(N=64, dx=2e-6, w=20e-6):
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


SINGLET = {
    'wavelength': 1.31e-6, 'aperture_diameter': 24e-3,
    'surfaces': [
        {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 12e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 12e-3},
    ],
    'thicknesses': [5e-3], 'stop_index': 0,
}


# ======================================================================
# P5 -- propagate(method='auto') return-shape / pitch instability
# ======================================================================

class TestP5AutoReturnContract:
    """v5.30 (roadmap F1 flip, audit P5): every pin in this class now names
    ``return_result=False`` explicitly.  The shape-instability diagnostic these
    pins are about lives ON that path -- the dispatcher's DEFAULT return became
    a shape-stable ``PropagationResult`` for every method, so there is nothing
    left to warn about there (counter-pins for the new default live in
    ``test_niche_audit_w4_p5_return_contract.py``).  The warning text, its
    gating set and the pitch it quotes are unchanged.
    """

    # z values chosen from the selector's own thresholds at N=64,
    # dx=2 um, lambda=633 nm: Q = lambda*z/(N*dx^2) crosses 1 at
    # z = 4.04e-4 m and N_F = (N*dx/2)^2/(lambda*z) crosses 0.1 at
    # z = 6.47e-2 m.
    @pytest.mark.parametrize('z,expect', [(1e-3, 'sas'), (5.0, 'fraunhofer')])
    def test_auto_grid_changing_selection_warns(self, z, expect):
        """Pre-fix: zero warnings while the return silently became a
        3-tuple at a different pitch."""
        from lumenairy.propagators.dispatch import (
            _auto_select_method,
            propagate,
        )
        E = _gauss()
        assert _auto_select_method(E, z=z, wavelength=LAM, dx=2e-6,
                                   prescription=None) == expect
        with pytest.warns(UserWarning) as rec:
            out = propagate(E, z=z, wavelength=LAM, dx=2e-6, method='auto',
                            return_result=False)
        assert isinstance(out, tuple) and len(out) == 3
        msgs = [str(w.message) for w in rec
                if issubclass(w.category, UserWarning)]
        hit = [m for m in msgs if "method='auto'" in m]
        assert hit, f'no audit-P5 warning among {msgs!r}'
        m = hit[0]
        # names what was selected ...
        assert repr(expect) in m
        # ... quotes the NEW pitch and the input pitch ...
        assert f'{float(out[1]):.6e}' in m
        assert f'{2e-6:.6e}' in m
        # ... and points at the stable contract.
        assert 'return_result=True' in m

    def test_auto_same_grid_selection_is_silent(self):
        """Counter-pin: when ``auto`` picks ASM the return IS a bare
        ndarray at the input pitch, so there is nothing to warn about."""
        from lumenairy.propagators.dispatch import (
            _auto_select_method,
            propagate,
        )
        E = _gauss()
        assert _auto_select_method(E, z=1e-4, wavelength=LAM, dx=2e-6,
                                   prescription=None) == 'asm'
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = propagate(E, z=1e-4, wavelength=LAM, dx=2e-6,
                            method='auto', return_result=False)
        assert isinstance(out, np.ndarray)
        assert not [w for w in rec
                    if issubclass(w.category, UserWarning)
                    and "method='auto'" in str(w.message)]

    def test_explicit_method_is_silent(self):
        """The caller who NAMES sas knows its contract -- no warning."""
        from lumenairy.propagators.dispatch import propagate
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            out = propagate(_gauss(), z=1e-3, wavelength=LAM, dx=2e-6,
                            method='sas', return_result=False)
        assert isinstance(out, tuple)
        assert not [w for w in rec
                    if issubclass(w.category, UserWarning)
                    and "method='auto'" in str(w.message)]

    def test_return_result_is_silent_and_reports_kernel_pitch(self):
        """``return_result=True`` IS the stable contract, so it must not
        warn -- and must carry the kernel's output pitch."""
        from lumenairy.propagators.dispatch import propagate
        E = _gauss()
        bare = propagate(E, z=1e-3, wavelength=LAM, dx=2e-6, method='sas',
                         return_result=False)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            res = propagate(E, z=1e-3, wavelength=LAM, dx=2e-6,
                            method='auto', return_result=True)
        assert not [w for w in rec
                    if issubclass(w.category, UserWarning)
                    and "method='auto'" in str(w.message)]
        assert res.method == 'sas'
        assert res.dx == pytest.approx(float(bare[1]), rel=1e-12)

    def test_docstring_documents_the_full_auto_table(self):
        from lumenairy.propagators.dispatch import propagate
        doc = inspect.getdoc(propagate)
        assert "method='auto'`` return contract" in doc
        for token in ('asm', 'sas', 'fraunhofer', 'maslov', 'gbd', 'hfpi',
                      'hf'):
            assert token in doc
        assert 'N_F' in doc and 'Q = lambda' in doc
        assert 'return_result=True' in doc
        # the deferred-API-decision note must be recorded, not silently
        # dropped
        assert 'deferred' in doc.lower()

    def test_grid_changing_set_matches_the_tuple_returning_kernels(self):
        """Guards the warning's gate: exactly the kernels whose native
        return is a triple are in ``_GRID_CHANGING_METHODS``."""
        from lumenairy.propagators import dispatch
        E = _gauss()
        for m in dispatch.VALID_METHODS:
            if m in ('auto', 'maslov', 'asymptotic', 'gbd', 'hfpi', 'hf',
                     'mhs'):
                continue
            out = dispatch.propagate(E, z=1e-3, wavelength=LAM, dx=2e-6,
                                     method=m, return_result=False)
            is_triple = isinstance(out, (tuple, list)) and len(out) == 3
            assert is_triple == (m in dispatch._GRID_CHANGING_METHODS), m


# ======================================================================
# NEW finding -- propagate_through_system junk-method fall-through
# ======================================================================

class TestNewSystemMethodValidation:

    ELEMS = [{'type': 'propagate', 'z': 1e-3}]

    @pytest.mark.parametrize('bad', ['not_a_method', 'ASM', 'gbd',
                                     'fraunhofer', '', 'Fresnel'])
    def test_junk_entry_method_raises(self, bad):
        """Pre-fix: ACCEPTED, 0.0 rel diff vs method='asm'."""
        from lumenairy.propagators.system import propagate_through_system
        with pytest.raises(ValueError) as ei:
            propagate_through_system(_gauss(), self.ELEMS, LAM, 2e-6,
                                     method=bad)
        m = str(ei.value)
        assert repr(bad) in m                        # names the value
        assert 'asm' in m and 'sas' in m and 'fresnel' in m   # allowed set
        assert 'not a recognised free-space method' in m

    @pytest.mark.parametrize('bad', ['not_a_method', 'gbd', 'fraunhofer'])
    def test_junk_per_element_method_raises(self, bad):
        """Pre-fix: the per-element override took the same silent
        fall-through (measured 0.0 rel diff vs asm)."""
        from lumenairy.propagators.system import propagate_through_system
        elems = [{'type': 'propagate', 'z': 1e-3, 'method': bad}]
        with pytest.raises(ValueError) as ei:
            propagate_through_system(_gauss(), elems, LAM, 2e-6, method='asm')
        m = str(ei.value)
        assert "elements[0]['method']" in m
        assert repr(bad) in m
        assert 'not a recognised free-space method' in m

    def test_per_element_rs_override_rejected(self):
        """'rs' via a per-element override was ALSO a silent ASM
        fall-through pre-fix (the entry-level guard only saw the
        default-resolver path)."""
        from lumenairy.propagators.system import propagate_through_system
        elems = [{'type': 'propagate', 'z': 1e-3, 'method': 'rs'}]
        with pytest.raises(ValueError, match='not supported'):
            propagate_through_system(_gauss(), elems, LAM, 2e-6, method='asm')

    @pytest.mark.parametrize('good', ['asm', 'sas', 'fresnel'])
    def test_honoured_methods_still_run(self, good):
        from lumenairy.propagators.system import propagate_through_system
        E, _inter = propagate_through_system(
            _gauss(), self.ELEMS, LAM, 2e-6, method=good)
        assert E.shape == (64, 64)
        assert np.all(np.isfinite(E))

    def test_per_element_override_still_honoured(self):
        """The override must keep WORKING, not just stop being silent:
        an element-level 'fresnel' must differ from system-level asm."""
        from lumenairy.propagators.system import propagate_through_system
        E = _gauss()
        a, _ = propagate_through_system(E, self.ELEMS, LAM, 2e-6,
                                        method='asm')
        b, _ = propagate_through_system(
            E, [{'type': 'propagate', 'z': 1e-3, 'method': 'fresnel'}],
            LAM, 2e-6, method='asm')
        rel = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
        assert rel > 1e-6, 'per-element fresnel override became inert'

    def test_message_shape_matches_the_jax_twin(self):
        """Audit P6 fixed the JAX twin in 153ca54; the two entry points
        must reject junk with the same wording."""
        from lumenairy.propagators.system import _validate_system_method
        with pytest.raises(ValueError) as np_err:
            _validate_system_method('not_a_method', 'method')
        assert 'is not a recognised free-space method' in str(np_err.value)
        # The same phrase must appear in the JAX twin's message.  Compare
        # on the source (JAX may not be installed) using the fragment that
        # is un-split in both f-strings.
        src = inspect.getsource(
            __import__('lumenairy.propagators.system', fromlist=['x']))
        assert src.count('recognised free-space method') >= 2


# ======================================================================
# P8 -- recommend_gbd_sampling(wavelength=) inert
# ======================================================================

class TestP8RecommendGbdWavelength:

    def _curved(self, lam, N=96, dx=1e-6, R=4e-3, w=25e-6):
        x = (np.arange(N) - N // 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        k = 2 * np.pi / lam
        return (np.exp(-(X ** 2 + Y ** 2) / w ** 2)
                * np.exp(1j * k * (X ** 2 + Y ** 2) / (2 * R))
                ).astype(np.complex128)

    def test_wavelength_is_removed(self):
        """SUPERSEDES ``test_wavelength_is_now_optional``,
        ``test_passing_wavelength_deprecation_warns`` and
        ``test_wavelength_value_is_still_inert_by_construction`` (P8,
        v5.30 warn phase).

        P8 demoted the never-read REQUIRED keyword to optional +
        ``DeprecationWarning``; the W5 wave deletes it in the same release
        rather than shipping an inert keyword to v5.32.  The inertness the
        third pin measured (identical dict for 0.4 um vs 10 um) is exactly
        what makes the deletion output-neutral, so it is now a property of
        the signature rather than of the body.
        """
        from lumenairy.propagators.gbd import recommend_gbd_sampling
        sig = inspect.signature(recommend_gbd_sampling)
        assert 'wavelength' not in sig.parameters, sig
        with warnings.catch_warnings():
            warnings.simplefilter('error')          # must not warn
            out = recommend_gbd_sampling(self._curved(1.55e-6), 1e-6)
        assert set(out) == {'sample_step', 'waist_factor', 'n_beamlets'}
        with pytest.raises(TypeError, match='wavelength'):
            recommend_gbd_sampling(self._curved(1.55e-6), 1e-6,
                                   wavelength=1.55e-6)

    def test_lambda_dependence_arrives_through_the_field(self):
        """The evidence behind declining a lambda-dependent term: a
        PHYSICAL field at a shorter wavelength has a proportionally
        steeper phase gradient, and the recommendation follows it."""
        from lumenairy.propagators.gbd import recommend_gbd_sampling
        short = recommend_gbd_sampling(self._curved(0.4e-6), 1e-6)
        long_ = recommend_gbd_sampling(self._curved(10e-6), 1e-6)
        assert short['sample_step'] < long_['sample_step']

    def test_docstring_records_the_measured_rejection(self):
        """The evidence must survive the keyword: a future reader asking
        "why is there no wavelength here?" needs the measurement, and a
        migrating caller needs to be pointed at
        ``converge_gbd_sampling``.  (Superseded assertion: the docstring
        used to have to carry a ``deprecated:: 5.30`` / v5.32 banner; it
        now carries the ``versionchanged`` removal record instead.)"""
        from lumenairy.propagators.gbd import recommend_gbd_sampling
        doc = inspect.getdoc(recommend_gbd_sampling)
        assert 'Deprecated and unused' in doc
        assert 'converge_gbd_sampling' in doc
        assert 'versionchanged:: 5.30' in doc and 'REMOVED' in doc
        assert 'deprecated:: 5.30' not in doc, (
            'the keyword is gone; the docstring must not still advertise '
            'it as a live deprecation')

    def test_converge_gbd_sampling_does_not_self_warn(self):
        """The internal call must not forward the deprecated kwarg."""
        from lumenairy.propagators.gbd import converge_gbd_sampling
        N, dx = 48, 1e-6
        x = (np.arange(N) - N // 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        E = np.exp(-(X ** 2 + Y ** 2) / (12e-6) ** 2).astype(np.complex128)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            converge_gbd_sampling(E, dx, wavelength=LAM,
                                  test_distance=1e-4, overlaps=(1.5,))
        bad = [str(w.message) for w in rec
               if issubclass(w.category, DeprecationWarning)
               and 'recommend_gbd_sampling' in str(w.message)]
        assert not bad, bad


# ======================================================================
# P9 -- fga_memory_estimate(nsig=) inert
# ======================================================================

class TestP9FgaMemoryEstimateNsig:

    NSIGS = (1.0, 3.0, 6.0, 12.0)

    def _est(self, N, nsig):
        from lumenairy.propagators.fga import fga_memory_estimate
        return fga_memory_estimate(N, 3e-6, SINGLET, 1.31e-6, nsig=nsig,
                                   p_max=0.05, n_p=21)

    @pytest.mark.parametrize('N', [512, 1024])
    def test_nsig_moves_the_estimate(self, N):
        """Pre-fix EVERY key was bit-identical across nsig = 1 -> 12."""
        vals = [self._est(N, ns) for ns in self.NSIGS]
        win = [v['gabor_window_mb'] for v in vals]
        assert all(b > a for a, b in zip(win, win[1:])), win
        # The nsig term must REACH peak_chunked_mb -- but assert it net of
        # chunked_swarm_peak_mb, whose chunk sizing reads the machine's
        # AVAILABLE RAM at call time: on a busy CI runner it jitters
        # ~0.2 MB between the four calls (measured, bc6738b 3.10 shard 2),
        # swamping the sub-MB gabor steps and making raw peak
        # monotonicity a coin flip.  Net of it, the sequence is exactly
        # grid + gabor_window, deterministic on every machine.
        peak_net = [v['peak_chunked_mb'] - v['chunked_swarm_peak_mb']
                    for v in vals]
        assert all(b > a for a, b in zip(peak_net, peak_net[1:])), peak_net
        assert win[-1] > 4.0 * win[0]        # ~linear in the window radius

    @pytest.mark.parametrize('N', [512, 1024])
    def test_nsig_echoed_and_pools_add_up(self, N):
        for ns in self.NSIGS:
            e = self._est(N, ns)
            assert e['nsig'] == pytest.approx(ns)
            assert e['peak_unchunked_mb'] == pytest.approx(
                e['grid_arrays_mb'] + e['gabor_window_mb']
                + e['swarm_peak_mb'], rel=1e-12)
            assert e['peak_chunked_mb'] == pytest.approx(
                e['grid_arrays_mb'] + e['gabor_window_mb']
                + e['chunked_swarm_peak_mb'], rel=1e-12)
            assert e['peak_chunked_mb'] <= e['peak_unchunked_mb'] + 1e-9

    def test_default_nsig_matches_the_real_path(self):
        from lumenairy.propagators.fga import (
            apply_real_lens_fga,
            fga_memory_estimate,
        )
        run_default = inspect.signature(
            apply_real_lens_fga).parameters['nsig'].default
        est_default = inspect.signature(
            fga_memory_estimate).parameters['nsig'].default
        assert est_default == run_default
        assert self._est(512, run_default)['nsig'] == pytest.approx(
            float(run_default))

    def test_window_pool_zero_on_the_direct_kernel(self):
        """The non-separable kernel (n_p < 5) accumulates into scalars and
        allocates nothing window-sized, so the pool must vanish there."""
        from lumenairy.propagators.fga import fga_memory_estimate
        e = fga_memory_estimate(128, 3e-6, SINGLET, 1.31e-6, nsig=12.0,
                                p_max=0.05, n_p=3)
        assert e['use_separable'] is False
        assert e['gabor_window_mb'] == 0.0

    def test_window_model_tracks_the_measured_table_delta(self):
        """est-vs-actual: the Python-visible term of the model reproduces
        the tracemalloc peak DELTA between nsig=3 and nsig=12 (the
        nsig-independent u0 copies cancel in the delta).  Measured
        54.72 kB, model 55.08 kB (ratio 0.993) at N=128 and N=256."""
        import tracemalloc

        import lumenairy.propagators.fga as fga
        N, dx, n_p = 128, 3e-6, 9
        w0 = 5.0 * dx
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        u0 = np.exp(-(X ** 2 + Y ** 2) / (1e-4) ** 2).astype(np.complex128)
        pv = np.linspace(-0.02, 0.02, n_p)
        q = np.array([0.0])

        def peak(nsig):
            tracemalloc.start()
            tracemalloc.reset_peak()
            fga._gabor_coeff_sep(u0, q, q, pv, x[0], x[0], dx, dx, w0,
                                 2 * np.pi / 1.31e-6, 1.0, nsig)
            cur, pk = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            return pk - cur

        for ns in (3.0, 12.0):                       # warm numba + caches
            peak(ns)
        d_meas = float(np.median([peak(12.0) - peak(3.0) for _ in range(7)]))

        def tables(nsig):
            m = int(nsig * w0 / dx) + 1
            return fga._GABOR_TABLES_PER_AXIS * 8.0 * n_p * 2 * (2 * m + 1)

        d_model = tables(12.0) - tables(3.0)
        assert d_meas / d_model == pytest.approx(1.0, rel=0.15)

    def test_docstring_lists_the_third_pool(self):
        from lumenairy.propagators.fga import fga_memory_estimate
        doc = inspect.getdoc(fga_memory_estimate)
        assert 'Three memory pools' in doc
        assert 'gabor_window_mb' in doc
        assert 'versionchanged:: 5.30' in doc


# ======================================================================
# P10 -- patches_for_box(centred=) inert + asymmetric tiling
# ======================================================================

class TestP10PatchesForBoxCentred:

    BOX = (100e-6, 100e-6)
    PATCH = (50e-6, 50e-6)
    OV = 0.25

    def _span(self, g, axis=0):
        c = np.asarray(g.centres)[:, axis]
        h = np.asarray(g.half_widths)[:, axis]
        return float((c - h).min()), float((c + h).max())

    def test_centred_true_is_symmetric(self):
        """Pre-fix: span [-50.000, +75.000] um -- asymmetry +25 um."""
        from lumenairy.propagators.subaperture import patches_for_box
        g = patches_for_box(self.BOX, self.PATCH, overlap=self.OV,
                            centred=True)
        for axis in (0, 1):
            lo, hi = self._span(g, axis)
            assert lo + hi == pytest.approx(0.0, abs=1e-18)
            c = np.unique(np.asarray(g.centres)[:, axis])
            assert float(c.sum()) == pytest.approx(0.0, abs=1e-18)

    def test_centred_false_reproduces_the_legacy_layout_bit_for_bit(self):
        from lumenairy.propagators.subaperture import patches_for_box
        g = patches_for_box(self.BOX, self.PATCH, overlap=self.OV,
                            centred=False)
        W_x, W_y = self.BOX
        w_x, w_y = self.PATCH
        step_x = w_x * (1 - self.OV)
        step_y = w_y * (1 - self.OV)
        n_x = max(1, int(np.ceil(W_x / step_x)))
        n_y = max(1, int(np.ceil(W_y / step_y)))
        cx = np.array([-W_x / 2 + w_x / 2 + i * step_x for i in range(n_x)])
        cy = np.array([-W_y / 2 + w_y / 2 + j * step_y for j in range(n_y)])
        CX, CY = np.meshgrid(cx, cy, indexing='xy')
        legacy = np.stack([CX.reshape(-1), CY.reshape(-1)], axis=-1)
        assert np.array_equal(np.asarray(g.centres), legacy)

    def test_centred_flag_is_no_longer_inert(self):
        """Pre-fix the two were bit-identical.  The offset is HALF THE
        SURPLUS COVERAGE ``((n-1)*step + w - W)/2`` -- the legacy layout
        pushed the whole surplus past the high edge."""
        from lumenairy.propagators.subaperture import patches_for_box
        t = patches_for_box(self.BOX, self.PATCH, overlap=self.OV,
                            centred=True)
        f = patches_for_box(self.BOX, self.PATCH, overlap=self.OV,
                            centred=False)
        d = np.asarray(f.centres) - np.asarray(t.centres)
        assert not np.array_equal(np.asarray(t.centres),
                                  np.asarray(f.centres))
        for axis in (0, 1):
            W, w = self.BOX[axis], self.PATCH[axis]
            step = w * (1 - self.OV)
            n = max(1, int(np.ceil(W / step)))
            assert d[:, axis] == pytest.approx(
                ((n - 1) * step + w - W) / 2)

    def test_offset_reduces_to_w_overlap_over_2_on_an_exact_multiple(self):
        """The ``w*overlap/2`` special case the audit quoted -- and the one
        :func:`propagate_subaperture_asymptotic` hits, since it builds its
        box as ``n * patch * (1 - overlap)``."""
        from lumenairy.propagators.subaperture import patches_for_box
        w, ov, n = 50e-6, 0.25, 3
        step = w * (1 - ov)
        box = (n * step, n * step)
        t = patches_for_box(box, (w, w), overlap=ov, centred=True)
        f = patches_for_box(box, (w, w), overlap=ov, centred=False)
        d = np.asarray(f.centres) - np.asarray(t.centres)
        assert d[:, 0] == pytest.approx(w * ov / 2)
        assert d[:, 1] == pytest.approx(w * ov / 2)

    @pytest.mark.parametrize('overlap', [0.0, 0.1, 0.25, 0.5])
    @pytest.mark.parametrize('box,patch', [((100e-6, 100e-6), (50e-6, 50e-6)),
                                           ((300e-6, 120e-6), (80e-6, 40e-6)),
                                           ((40e-6, 40e-6), (60e-6, 60e-6))])
    def test_centred_true_symmetric_and_covering(self, overlap, box, patch):
        from lumenairy.propagators.subaperture import patches_for_box
        g = patches_for_box(box, patch, overlap=overlap, centred=True)
        for axis in (0, 1):
            lo, hi = self._span(g, axis)
            assert lo + hi == pytest.approx(0.0, abs=1e-18)
            # the symmetric tiling still covers the whole box
            assert lo <= -box[axis] / 2 + 1e-18
            assert hi >= box[axis] / 2 - 1e-18

    def test_single_patch_centres_on_origin(self):
        from lumenairy.propagators.subaperture import patches_for_box
        g = patches_for_box((10e-6, 10e-6), (50e-6, 50e-6), overlap=0.25,
                            centred=True)
        assert len(g) == 1
        assert np.asarray(g.centres)[0] == pytest.approx([0.0, 0.0])

    def test_docstring_documents_both_layouts(self):
        from lumenairy.propagators.subaperture import patches_for_box
        doc = inspect.getdoc(patches_for_box)
        assert 'centred' in doc
        assert 'versionchanged:: 5.30' in doc
        assert 'HALF THE SURPLUS COVERAGE' in doc
        assert 'overlap / 2' in doc


# ======================================================================
# P11 -- MFT output-grid validation + silent periodic replicas
# ======================================================================

class TestP11MftOutputGrid:

    def _kernels(self):
        from lumenairy.propagators.mft import (
            angular_spectrum_propagate_mft,
            fraunhofer_propagate_mft,
            fresnel_propagate_mft,
        )
        return (angular_spectrum_propagate_mft, fresnel_propagate_mft,
                fraunhofer_propagate_mft)

    @pytest.mark.parametrize('dx_out', [-1e-6, 0.0, float('nan'),
                                        float('inf')])
    def test_bad_dx_out_raises_naming_the_propagator(self, dx_out):
        """Pre-fix: -1e-6 returned a finite mirrored field, nan returned
        an all-NaN field, 0.0 gave a bare ZeroDivisionError."""
        E = _gauss()
        for fn in self._kernels():
            with pytest.raises(ValueError) as ei:
                fn(E, 1e-3, LAM, 2e-6, dx_out, 64)
            m = str(ei.value)
            assert fn.__name__ in m
            assert 'dx_out' in m

    @pytest.mark.parametrize('N_out', [0, -4])
    def test_bad_N_out_raises_naming_the_propagator(self, N_out):
        E = _gauss()
        for fn in self._kernels():
            with pytest.raises(ValueError) as ei:
                fn(E, 1e-3, LAM, 2e-6, 2e-6, N_out)
            m = str(ei.value)
            assert fn.__name__ in m and 'N_out' in m

    def test_bad_dy_out_raises(self):
        E = _gauss()
        for fn in self._kernels():
            with pytest.raises(ValueError, match='dy_out'):
                fn(E, 1e-3, LAM, 2e-6, 2e-6, 64, dy_out=-1e-6)

    def test_asm_mft_replica_window_warns_with_numbers(self):
        """Pre-fix: silent.  The ASM-MFT Bluestein inverts the input
        SPECTRUM, so the period is the input cell N_in*dx_in."""
        from lumenairy.propagators.mft import angular_spectrum_propagate_mft
        E = _gauss()
        period = 64 * 2e-6
        with pytest.warns(UserWarning) as rec:
            angular_spectrum_propagate_mft(E, 5e-4, LAM, 2e-6,
                                           4.0 * period / 256, 256)
        m = str(rec[0].message)
        assert 'angular_spectrum_propagate_mft' in m
        assert 'REPLICA' in m
        assert f'{period:.6e}' in m                  # the period, by number
        assert 'N_in*d_in' in m

    def test_asm_mft_natural_and_same_grid_windows_are_silent(self):
        """Counter-pin: the same-grid call sits exactly ON one period."""
        from lumenairy.propagators.mft import angular_spectrum_propagate_mft
        E = _gauss()
        for dx_out, N_out in ((2e-6, 64), (1e-6, 64), (0.5e-6, 128)):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter('always')
                angular_spectrum_propagate_mft(E, 5e-4, LAM, 2e-6,
                                               dx_out, N_out)
            assert not [w for w in rec
                        if issubclass(w.category, UserWarning)
                        and 'REPLICA' in str(w.message)], (dx_out, N_out)

    @pytest.mark.parametrize('name', ['fresnel_propagate_mft',
                                      'fraunhofer_propagate_mft'])
    def test_fresnel_family_replica_period_is_lambda_z_over_dx(self, name):
        """These transform the input FIELD, not its spectrum, so the
        period is lambda*|z|/dx_in -- verified by measurement (three
        equal lobes at exactly that spacing in a 4x window)."""
        import lumenairy.propagators.mft as mft
        fn = getattr(mft, name)
        E, z, dx_in = _gauss(), 5e-4, 2e-6
        period = LAM * z / dx_in
        assert abs(period - 64 * dx_in) > 1e-6 * period   # distinct from ASM
        with pytest.warns(UserWarning) as rec:
            fn(E, z, LAM, dx_in, 4.0 * period / 256, 256)
        m = str(rec[0].message)
        assert f'{period:.6e}' in m
        assert 'lambda*|z|/d_in' in m
        # exactly one period -> silent
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter('always')
            fn(E, z, LAM, dx_in, period / 256, 256)
        assert not [w for w in rec2
                    if issubclass(w.category, UserWarning)
                    and 'REPLICA' in str(w.message)]

    def test_replica_warning_is_a_warning_not_a_change_in_values(self):
        """Guard-only: the field returned for a wide window is unchanged."""
        from lumenairy.propagators.mft import angular_spectrum_propagate_mft
        E = _gauss()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wide = angular_spectrum_propagate_mft(E, 5e-4, LAM, 2e-6,
                                                  2e-6, 256)
        # replica structure IS present -- three lobes one period apart
        row = np.abs(wide[128, :])
        pk = row.max()
        lobes = [i for i in range(1, 255)
                 if row[i] > 0.2 * pk and row[i] >= row[i - 1]
                 and row[i] >= row[i + 1]]
        assert len(lobes) >= 2, lobes


# ======================================================================
# P12 -- band-limit cutoff is the z->infinity asymptote, not the paper's
#        exact criterion
# ======================================================================

class TestP12BandlimitAsymptoteDocumented:

    SITES = [
        ('lumenairy.propagators.fft_infra', '_get_or_make_bandlimit'),
        ('lumenairy.propagators.asm', '_build_asm_H_square'),
        ('lumenairy.propagators.rs', 'rayleigh_sommerfeld_propagate'),
        ('lumenairy.propagators.mft', 'angular_spectrum_propagate_mft'),
    ]

    @pytest.mark.parametrize('mod,fn', SITES)
    def test_docstring_states_the_asymptote(self, mod, fn):
        obj = getattr(__import__(mod, fromlist=['x']), fn)
        doc = inspect.getdoc(obj) or ''
        assert 'asymptote' in doc, f'{mod}.{fn} still mis-attributes it'
        assert 'audit P12' in doc

    def test_canonical_site_derives_and_bounds_it(self):
        from lumenairy.propagators.fft_infra import _get_or_make_bandlimit
        doc = inspect.getdoc(_get_or_make_bandlimit)
        assert 'sqrt((2*z / L)^2 + 1)' in doc
        assert 'never over-filters' in doc
        assert 'Matsushima' in doc

    def test_asymptote_is_an_upper_bound_on_the_exact_limit(self):
        """The physics behind 'one-sided safe': the asymptote is always
        >= the exact local-frequency limit, so the mask keeps every
        frequency the exact criterion would keep."""
        for L in (1.28e-4, 5.12e-4):
            for zr in (0.25, 0.5, 1.0, 2.0, 5.0, 20.0):
                z = zr * L
                exact = 1.0 / (LAM * np.sqrt((2.0 * z / L) ** 2 + 1.0))
                asym = L / (2.0 * LAM * z)
                assert asym >= exact
                assert asym / exact == pytest.approx(
                    np.sqrt(1.0 + (L / (2 * z)) ** 2), rel=1e-12)

    def test_cutoff_itself_is_unchanged(self):
        """The fix is documentation-only: the mask edge must still be
        exactly L/(2*lambda*|z|)."""
        from lumenairy.propagators.fft_infra import _get_or_make_bandlimit
        N, dx, z = 64, 2e-6, 5e-4
        bl_x, bl_y = _get_or_make_bandlimit(N, N, dx, dx, LAM, abs(z), True)
        f = (np.arange(N) - N // 2) / (N * dx)
        f_max = (N * dx) / (2 * LAM * abs(z))
        assert np.array_equal(np.asarray(bl_x), np.abs(f) < f_max)
        assert np.array_equal(np.asarray(bl_y), np.abs(f) < f_max)


# ======================================================================
# P13 -- dead guards / dead parameters
# ======================================================================

class TestP13DeadCode:

    def test_rs_forward_only_so_the_z_guard_was_dead(self):
        from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
        for z in (0.0, -1e-3):
            with pytest.raises(ValueError, match='z must be > 0'):
                rayleigh_sommerfeld_propagate(_gauss(), z, LAM, 2e-6)

    def test_rs_bandlimit_guard_no_longer_tests_z(self):
        import lumenairy.propagators.rs as rs
        src = inspect.getsource(rs.rayleigh_sommerfeld_propagate)
        assert 'if bandlimit and z != 0:' not in src
        assert 'if bandlimit:' in src

    def test_rs_bandlimit_still_functional(self):
        """Removing the dead conjunct must not change any value."""
        from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
        E = _gauss()
        on = rayleigh_sommerfeld_propagate(E, 1e-3, LAM, 2e-6,
                                           bandlimit=True)
        off = rayleigh_sommerfeld_propagate(E, 1e-3, LAM, 2e-6,
                                           bandlimit=False)
        assert np.all(np.isfinite(on)) and np.all(np.isfinite(off))
        # at this geometry the band-limit does bite, so the two differ
        assert float(np.max(np.abs(np.asarray(on) - np.asarray(off)))) > 0.0

    def test_asm_family_z_guards_are_live_and_must_stay(self):
        """Counter-pin: ASM / ASM-MFT / tilted-ASM DO accept z == 0 (the
        exact identity, audit S2-11), so their ``and z != 0`` conjuncts
        are NOT dead and were deliberately left in place."""
        from lumenairy.propagators.asm import (
            angular_spectrum_propagate,
            angular_spectrum_propagate_tilted,
        )
        from lumenairy.propagators.mft import angular_spectrum_propagate_mft
        E = _gauss()
        out = angular_spectrum_propagate(E, 0.0, LAM, 2e-6)
        assert np.allclose(np.asarray(out), E, atol=1e-12)
        angular_spectrum_propagate_tilted(E, 0.0, LAM, 2e-6, tilt_x=1e-3)
        angular_spectrum_propagate_mft(E, 0.0, LAM, 2e-6, 2e-6, 64)
        import lumenairy.propagators.asm as asm
        import lumenairy.propagators.mft as mft
        assert 'if bandlimit and z != 0:' in inspect.getsource(asm)
        assert 'if bandlimit and z != 0:' in inspect.getsource(mft)

    def test_promote_entry_to_measure_lost_its_dead_first_param(self):
        from lumenairy.propagators.fft_infra import _promote_entry_to_measure
        params = list(inspect.signature(_promote_entry_to_measure)
                      .parameters)
        assert params == ['direction', 'shape_t', 'dt', 'threads']

    def test_promote_entry_still_reachable_from_propagation(self):
        """The v5.1.0 split pins this private name as re-exported."""
        from lumenairy.propagators import propagation
        assert hasattr(propagation, '_promote_entry_to_measure')

    def test_fga_coarse_lost_its_dead_coeff_frac_param(self):
        import lumenairy.propagators.fga as fga
        params = list(inspect.signature(fga._fga_coarse).parameters)
        assert 'coeff_frac' not in params
        # the public knob still exists on the entry point
        assert 'coeff_frac' in inspect.signature(
            fga.apply_real_lens_fga).parameters

    def test_mhs_aperture_propagator_keeps_its_protocol_signature(self):
        """NOT deleted (declined): MhsPipeline.run calls every subdomain
        as ``propagator(E, in_surface, out_surface, **kwargs)``, so the
        third positional parameter is protocol, not dead weight --
        removing it would raise TypeError.  Marked with a leading
        underscore instead."""
        from lumenairy.propagators.mhs import HuygensSurface, aperture_subdomain
        s = HuygensSurface(z=0.0, Ny=16, Nx=16, dx=2e-6)
        sub = aperture_subdomain(s, 8e-6)
        params = list(inspect.signature(sub.propagator).parameters)
        assert params[:3] == ['E', 'in_s', '_out_s']
        E = _gauss(16, 2e-6, 8e-6)
        out = sub.propagator(E, s, s)                # 3 positional: works
        assert out.shape == E.shape


# ======================================================================
# P14 -- hf beam_d4sigma fallback used the OUTPUT dims
# ======================================================================

class TestP14HfD4SigmaFallbackDims:

    @pytest.mark.parametrize('out_shape', [(64, 64), (16, 16), (32, 32)])
    def test_fallback_survives_mismatched_output_shape(self, out_shape,
                                                       monkeypatch):
        """Pre-fix: ``ValueError: operands could not be broadcast together
        with shapes (32,32) (64,)`` for any out_shape != (32, 32)."""
        import lumenairy.analysis.core as ac
        from lumenairy.propagators.hf import (
            propagate_huygens_fresnel_through_prescription as f,
        )

        def _boom(*a, **k):
            raise RuntimeError('moments diverged')

        monkeypatch.setattr(ac, 'beam_d4sigma', _boom)
        E = np.ones((32, 32), dtype=np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = f(E, 2e-6, SINGLET, wavelength=1.31e-6,
                    output_shape=out_shape, output_dx=2e-6,
                    n_field=6, n_pupil=6, poly_order=3)
        assert out.shape == out_shape

    def test_fallback_waist_is_output_shape_invariant(self, monkeypatch):
        """The fallback estimates the SOURCE waist, so it cannot depend on
        the requested output grid."""
        import lumenairy.analysis.core as ac
        from lumenairy.propagators.hf import (
            propagate_huygens_fresnel_through_prescription as f,
        )

        def _boom(*a, **k):
            raise RuntimeError('moments diverged')

        monkeypatch.setattr(ac, 'beam_d4sigma', _boom)
        x = (np.arange(32) - 16) * 2e-6
        X, Y = np.meshgrid(x, x, indexing='xy')
        E = np.exp(-(X ** 2 + Y ** 2) / (12e-6) ** 2).astype(np.complex128)
        outs = {}
        for shp in ((32, 32), (64, 64)):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outs[shp] = f(E, 2e-6, SINGLET, wavelength=1.31e-6,
                              output_shape=shp, output_dx=2e-6,
                              n_field=6, n_pupil=6, poly_order=3)
        # the (32,32) window of the 64-grid must equal the 32-grid result:
        # same w_s, same output coordinates for the shared samples
        big = outs[(64, 64)][16:48, 16:48]
        assert np.allclose(big, outs[(32, 32)], rtol=1e-10, atol=1e-30)

    def test_fallback_uses_input_dims_in_source(self):
        import lumenairy.propagators.hf as hf
        src = inspect.getsource(
            hf.propagate_huygens_fresnel_through_prescription)
        assert '_Nx_src' in src and 'audit P14' in src
        assert '(_np.arange(Nx) - Nx / 2) ** 2 * dx ** 2' not in src


# ======================================================================
# P16 -- PropagationResult.__iter__ arity hazard
# ======================================================================

class TestP16ResultIterArity:

    def test_iteration_still_yields_two_items(self):
        """Pinned behaviour -- do NOT change."""
        from lumenairy.propagators.result import PropagationResult
        r = PropagationResult(field=_gauss(), dx=2e-6, wavelength=LAM)
        assert len(list(r)) == 2
        a, b = r
        assert isinstance(a, np.ndarray) and b == []
        with pytest.raises(ValueError):
            _a, _b, _c = r

    def test_unwrapped_kernels_still_yield_three(self):
        from lumenairy.propagators.dispatch import propagate
        out = propagate(_gauss(), z=1e-3, wavelength=LAM, dx=2e-6,
                        method='sas', return_result=False)
        assert len(out) == 3

    def test_class_docstring_documents_the_hazard_prominently(self):
        from lumenairy.propagators.result import PropagationResult
        doc = inspect.getdoc(PropagationResult)
        head = doc[:1600]
        assert 'TWO items' in head
        assert 'audit P16' in head
        assert 'dx_out' in head
        assert 'three' in head.lower()

    def test_iter_method_docstring_names_the_arity(self):
        from lumenairy.propagators.result import PropagationResult
        doc = inspect.getdoc(PropagationResult.__iter__)
        assert '2 items' in doc
        assert 'audit P16' in doc

    def test_dispatch_docstring_documents_the_hazard(self):
        from lumenairy.propagators.dispatch import propagate
        doc = inspect.getdoc(propagate)
        assert 'audit P16' in doc
        assert 'two' in doc.lower() and 'three' in doc.lower()
