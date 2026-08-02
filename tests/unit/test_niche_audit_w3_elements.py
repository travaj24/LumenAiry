"""Pins for the remaining MEDIUM / LOW Territory-E findings of
``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`` (wave 3).

Every test in this file fails on a pre-fix worktree of 7ea2eb9.  One test per
finding wherever the finding has a measurable consequence; the doc-only
findings are pinned as docstring contracts so the text cannot silently rot back
(the audit's cross-cutting pattern 7: "docstrings claiming exactness /
conventions should carry a pin that measures the claim").

Findings covered
----------------
* **E-M6**  ``amplitude_model='ray_density'`` had no energy diagnostic.
* **E-M7**  ``create_periodic_phase_mask`` closed its lattice with ``clip``.
* **E-M8**  ``apply_spherical_lens`` clamped out-of-domain sag to a finite
  value while every sibling returns NaN.
* **E-M1 / E-M9 / E-M11 / E-M12 / E-M15 / E-L19**  documentation contracts.
* **E-L1 / E-L2 / E-L3**  worker-pool atexit leak, lock scope, fallback catch.
* **E-L4 / E-L5**  dead constants / dead numexpr scaffold / dead sag alias.
* **E-L8**  aspheric out-of-domain guard vs the sibling ``norm < 0.9999``.
* **E-L10** ``_radial_amp_sampler`` origin on an odd-N grid.
* **E-L11** turbulence-screen frequency lattice on an odd-N grid.
* **E-L18** ``elements/coronagraph.py`` namespace module had zero coverage.
* **E-L21** ``apply_real_lens_gbd(jacobian=...)`` unvalidated / inert.
* **doe deprecation rot** (shim REMOVED in v5.30 / W5; pins superseded
  below) -- the ``_legacy_units='auto'`` shim was unreachable
  for the very values its comment advertises.
"""
from __future__ import annotations

import atexit
import inspect
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_real as real_mod
from lumenairy.elements import _lens_thin, _lens_traced, _lens_traced_uniform
from lumenairy.elements import coronagraph as coronagraph_ns
from lumenairy.elements import elements as elements_mod
from lumenairy.elements import lenses as lenses_mod
from lumenairy.elements import lenses_gbd as gbd_mod
from lumenairy.elements.doe import create_periodic_phase_mask, makedammann2d
from lumenairy.elements.lenses import surface_sag_general

_WL = 1.31e-6


def _squash(text):
    return ' '.join((text or '').split())


# ===========================================================================
# E-M7 -- create_periodic_phase_mask must close the lattice with ``% cell_N``
# ===========================================================================

class TestEM7PeriodicPhaseMaskLattice:
    """``clip(round(in_cell/cell_px), 0, cell_N-1)`` folds the last HALF pixel
    of every cell onto the last column instead of wrapping it to column 0."""

    _CELL_N = 8
    _CELL_PX = 2.0e-6
    _N = 256

    def _index_map(self):
        """Recover the per-column cell index the function used, by giving each
        cell column a unique phase."""
        phase_1d = 2.0 * np.pi * np.arange(self._CELL_N) / self._CELL_N
        cell = np.tile(phase_1d, (self._CELL_N, 1))
        mask = create_periodic_phase_mask(self._N, self._CELL_PX, cell,
                                          self._CELL_PX)
        ang = np.angle(mask[self._N // 2, :]) % (2 * np.pi)
        idx = np.rint(ang / (2 * np.pi / self._CELL_N)).astype(int) \
            % self._CELL_N
        return idx

    def test_cell_pixel_occupancy_is_uniform(self):
        """On a grid-native design (one cell pixel per grid pixel) every cell
        column must be sampled the SAME number of times.

        Pre-fix measured occupancy: ``[21, 32, 32, 32, 32, 32, 32, 43]`` --
        column 0 starved by 11 and column 7 over-served by 11, because
        ``round`` reaches ``cell_N`` in the last half pixel and ``clip`` sent
        those 11 rows to ``cell_N - 1`` rather than wrapping to 0.
        """
        occ = np.bincount(self._index_map(), minlength=self._CELL_N)
        assert occ.min() == occ.max() == self._N // self._CELL_N, (
            f"non-uniform cell-pixel occupancy {occ.tolist()} (spread "
            f"{int(occ.max() - occ.min())}); the periodic lattice must close "
            f"with ``% cell_N``, not ``clip(..., 0, cell_N - 1)``.")

    def test_no_spurious_even_orders_and_no_off_lattice_leakage(self):
        """A tiled 0/pi 50 %-duty binary grating has power ONLY in the odd
        diffraction orders of its own lattice.

        Pre-fix: 11.28 % of the power landed off the order lattice and 2.95 %
        in the (forbidden) even orders; post-fix both are at the float64 floor.
        """
        cell_n = self._CELL_N
        phase_1d = np.where(np.arange(cell_n) < cell_n // 2, 0.0, np.pi)
        cell = np.tile(phase_1d, (cell_n, 1))
        mask = create_periodic_phase_mask(self._N, self._CELL_PX, cell,
                                          self._CELL_PX)
        per_px = self._N // cell_n
        S = np.abs(np.fft.fftshift(np.fft.fft(mask[self._N // 2, :]))) ** 2
        S = S / S.sum()
        dc = self._N // 2
        on_lattice = np.zeros(self._N, dtype=bool)
        on_lattice[dc % per_px::per_px] = True
        off = float(S[~on_lattice].sum())
        even = sum(float(S[dc + m * per_px])
                   for m in range(-(cell_n // 2), cell_n // 2 + 1)
                   if m % 2 == 0 and 0 <= dc + m * per_px < self._N)
        assert off < 1e-20, (
            f"off-lattice (aperiodic) leakage power {off:.3e}; the tiling is "
            f"not exactly periodic.")
        assert even < 1e-20, (
            f"spurious EVEN-order power {even:.3e} in a 0/pi 50%-duty binary "
            f"grating (analytically zero).")

    def test_mask_equals_the_modulo_reference_pixel_for_pixel(self):
        """Direct equality against a locally-built modulo tiling.

        Pre-fix: 2816 of 65536 pixels differed with ``max|dt| = 2.0`` (a full
        0 <-> pi transmission flip).
        """
        cell_n = self._CELL_N
        phase_1d = np.where(np.arange(cell_n) < cell_n // 2, 0.0, np.pi)
        cell = np.tile(phase_1d, (cell_n, 1))
        mask = create_periodic_phase_mask(self._N, self._CELL_PX, cell,
                                          self._CELL_PX)
        coord = (np.arange(self._N) - self._N / 2) * self._CELL_PX
        in_cell = np.mod(coord, cell_n * self._CELL_PX)
        idx = np.round(in_cell / self._CELL_PX).astype(int) % cell_n
        IX, IY = np.meshgrid(idx, idx)
        ref = np.exp(1j * cell[IY, IX])
        n_diff = int(np.sum(np.abs(mask - ref) > 1e-12))
        assert n_diff == 0, (
            f"{n_diff} of {mask.size} mask pixels differ from the modulo "
            f"tiling (max|dt| = {np.max(np.abs(mask - ref)):.3e}).")


# ===========================================================================
# doe deprecation rot -- the 'auto' shim must be reachable for its own values
# ===========================================================================

class TestDoeLegacyUnitsDeprecationReachability:

    _KW = dict(diforders=np.ones((2, 2)), itr=2, plot=False, seed=1,
               wavsamp=2.0)

    def test_auto_shim_is_removed_and_um_carries_its_documented_values(self):
        """SUPERSEDES ``test_auto_shim_fires_for_the_values_its_comment
        _advertises`` (v5.30 doe deprecation-rot fix).

        The W3 fix moved the >1 m bound after the rescale so the ``'auto'``
        branch became reachable for ``periodx=61.0`` / ``waveln=1.31`` --
        the shim's own documented legacy inputs.  v5.30 (W5) removes the
        branch outright.  The migration target must still accept exactly
        those values, which is what licensed the removal: ``'um'``
        reproduces what the reachable ``'auto'`` branch produced.
        """
        with pytest.raises(ValueError, match='REMOVED in v5.30'):
            makedammann2d(periodx=61.0, periody=61.0, waveln=1.31,
                          _legacy_units='auto', **self._KW)
        _nf, _ff, cell_um = makedammann2d(
            periodx=61.0, periody=61.0, waveln=1.31,
            _legacy_units='um', **self._KW)
        # 61 um / 1.31 um design -> the usual wavsamp sizing.
        n_ord = int(np.ceil(61e-6 / (self._KW['wavsamp'] * 1.31e-6)
                            * 0.5)) * 2
        assert cell_um[0] == pytest.approx(61e-6 / n_ord, rel=1e-12)

    def test_si_mode_still_rejects_metre_scale_input(self):
        """The unambiguous-nonsense bound is NOT weakened for the default
        ``'SI'`` mode -- it fires on the raw input there."""
        for kw, name in (
            (dict(periodx=2.0, periody=61e-6, waveln=1.31e-6), 'periodx'),
            (dict(periodx=61e-6, periody=2.0, waveln=1.31e-6), 'periody'),
            (dict(periodx=61e-6, periody=61e-6, waveln=1.5), 'waveln'),
        ):
            with pytest.raises(ValueError, match=name):
                makedammann2d(itr=1, plot=False, **kw)

    def test_auto_mode_post_rescale_bound_pin_is_SUPERSEDED(self):
        """SUPERSEDES ``test_auto_mode_rejects_post_rescale_nonsense``.

        With ``'auto'`` removed there is no post-rescale bound to test:
        the mode is rejected before any value is inspected.  ``'um'``
        deliberately has NO upper bound (the caller stated the unit), so
        the metre-scale guard that pin exercised now lives only on the
        ``'SI'`` path -- covered by
        :meth:`test_si_mode_still_rejects_metre_scale_input` above."""
        with pytest.raises(ValueError, match='REMOVED in v5.30'):
            makedammann2d(periodx=2.0e7, periody=61.0, waveln=1.31,
                          _legacy_units='auto', **self._KW)


# ===========================================================================
# E-M8 / E-L8 -- out-of-domain sag conventions in _lens_thin
# ===========================================================================

class TestEM8EL8ThinSagDomain:

    _R = 10e-3
    _N = 256
    _DX = 120e-6          # half-width 15.36 mm > R
    _AP = 28e-3           # > 2|R| = 20 mm

    def _grid(self):
        x = (np.arange(self._N) - self._N / 2) * self._DX
        X, Y = np.meshgrid(x, x)
        return X ** 2 + Y ** 2

    def test_em8_spherical_out_of_domain_matches_the_aspheric_sibling(self):
        """With ``aperture_diameter > 2|R|`` the pixels beyond the sphere have
        no surface.  ``apply_spherical_lens`` must NaN them exactly where
        ``apply_aspheric_lens`` (same geometry, k = 0) does.

        Pre-fix: 20916 such pixels left ``apply_spherical_lens`` with
        ``|E| = 1.000000`` and a clamped sag of 0.99 R, while the aspheric
        sibling NaN-ed all 20916.
        """
        h_sq = self._grid()
        E0 = np.ones((self._N, self._N), dtype=np.complex128)
        kw = dict(R1=self._R, R2=-self._R, d=2e-3, n_lens=1.5,
                  wavelength=632.8e-9, dx=self._DX,
                  aperture_diameter=self._AP)
        Es = la.apply_spherical_lens(E0, **kw)
        Ea = la.apply_aspheric_lens(E0, **kw)
        out = h_sq > self._R ** 2
        assert int(out.sum()) > 1000, 'probe grid lost its out-of-domain ring'
        nan_s = ~np.isfinite(Es)
        nan_a = ~np.isfinite(Ea)
        assert int(nan_s.sum()) > 0, (
            "apply_spherical_lens returned no NaN outside the sphere -- it is "
            "still clamping the sag to a finite value and transmitting "
            "|E| = 1 through a surface that does not exist.")
        assert np.array_equal(nan_s, nan_a), (
            f"NaN support differs from the aspheric sibling: "
            f"{int(nan_s.sum())} vs {int(nan_a.sum())} pixels.")
        # Nothing out of domain may survive with unit transmission any more.
        assert not np.any(np.isfinite(Es[out]) & (np.abs(Es[out]) > 0)), (
            "out-of-domain pixels still carry a finite, non-zero field")

    def test_em8_default_aperture_branch_still_zeroes_and_never_nans(self):
        """``aperture_diameter=None`` sets the aperture from the radii, so the
        NaN region is masked to 0 -- the change must be invisible there."""
        E0 = np.ones((self._N, self._N), dtype=np.complex128)
        out = la.apply_spherical_lens(
            E0, R1=self._R, R2=-self._R, d=2e-3, n_lens=1.5,
            wavelength=632.8e-9, dx=self._DX)
        assert np.all(np.isfinite(out)), (
            'the apertureless branch must not leak NaN into the field')

    def test_em8_docstring_states_the_nan_convention(self):
        doc = _squash(la.apply_spherical_lens.__doc__)
        assert 'NaN' in doc and 'aperture_diameter' in doc, (
            "apply_spherical_lens must document that an aperture wider than "
            "the surface domain returns NaN there.")

    def test_el8_aspheric_shell_guard_matches_surface_sag_general(self):
        """The sibling guard is ``norm < 0.9999``, not ``norm < 1.0``.

        On the shell ``0.9999 <= (1+k) h^2/R^2 < 1`` (352 pixels at N = 2048,
        dx = 10 um, R = 10 mm) ``apply_aspheric_lens`` used to return a finite
        ``|E| = 1`` while ``surface_sag_general`` returned NaN for all 352.
        """
        n, dd, k = 2048, 10e-6, 0.0
        xx = (np.arange(n) - n / 2) * dd
        XX, YY = np.meshgrid(xx, xx)
        hh = XX ** 2 + YY ** 2
        shell = ((1 + k) * hh / self._R ** 2 > 0.9999) & \
                ((1 + k) * hh / self._R ** 2 < 1.0)
        assert int(shell.sum()) > 100, 'probe grid does not resolve the shell'
        E = la.apply_aspheric_lens(
            np.ones((n, n), dtype=np.complex128),
            R1=self._R, R2=np.inf, d=2e-3, n_lens=1.5,
            wavelength=632.8e-9, dx=dd, aperture_diameter=self._AP)
        canonical = surface_sag_general(hh[shell], self._R, k, None)
        assert np.all(~np.isfinite(canonical)), 'oracle changed convention'
        assert np.all(~np.isfinite(E[shell])), (
            f"{int(np.sum(np.isfinite(E[shell])))} of {int(shell.sum())} "
            f"shell pixels are still finite; the aspheric guard must use the "
            f"siblings' ``norm < 0.9999`` cut.")


# ===========================================================================
# E-M6 -- ray-density energy self-check
# ===========================================================================

def _fast_singlet(ap=3e-3, r=9e-3):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128), X, Y


class TestEM6RayDensityEnergySelfCheck:

    def test_band_constants_are_subsample_aware(self):
        """The deficit tolerance must GROW with ray_subsample -- the loss is a
        coarse-lattice artefact, so a fixed band cannot be honest at both
        ends."""
        assert _lens_traced._RD_ENERGY_DEFICIT_PER_SUB > 0.0
        lo8 = 1.0 - (_lens_traced._RD_ENERGY_DEFICIT_BASE
                     + _lens_traced._RD_ENERGY_DEFICIT_PER_SUB * 7)
        lo1 = 1.0 - _lens_traced._RD_ENERGY_DEFICIT_BASE
        assert lo8 < lo1
        # Clear of the measured design-battery envelope (worst cell 0.95347 at
        # sub=8, 0.95685 at sub=1) at BOTH ends.
        assert lo8 < 0.9534 and lo1 < 0.9568, (
            f"band [{lo8:.4f} (sub8), {lo1:.4f} (sub1)] would fire on the "
            f"measured battery envelope")
        assert 0.0 < _lens_traced._RD_ENERGY_GAIN_TOL <= 0.10

    def test_docstring_documents_the_finite_subsample_loss(self):
        """E-M6's headline: the docstring said "energy-conserving in the
        geometric limit" and stopped there."""
        doc = _squash(la.apply_real_lens_traced.__doc__)
        assert 'energy self-check' in doc.lower(), (
            'the ray_density docs must describe the energy self-check')
        assert 'ray_subsample = 8' in doc or 'ray_subsample=8' in doc, (
            'the ray_density docs must quote the measured loss at the '
            'SHIPPED ray_subsample')

    def test_silent_on_a_battery_like_default_call(self):
        """A well-conditioned cell at the shipped ray_subsample must NOT warn
        (measured ratio 0.984 at N = 256, band lower bound 0.850)."""
        N, ap = 256, 3e-3
        dx = 2.2 * ap / N
        E0, X, Y = _gauss(N, dx, 1.2e-3)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            la.apply_real_lens_traced(
                E0, prescription=_fast_singlet(ap), wavelength=_WL, dx=dx,
                amplitude_model='ray_density', ray_subsample=8, n_workers=1,
                parallel_amp=False, on_undersample='silent',
                on_aperture_beam='silent')
        bad = [str(r.message) for r in rec
               if 'energy self-check' in str(r.message)]
        assert bad == [], f"spurious energy warning on a good cell: {bad}"

    def test_fires_when_a_fold_caustic_manufactures_energy(self):
        """A strong biconcave on a grid barely covering its aperture drives the
        capped ``1/sqrt(|det J|)`` amplitude past unity: measured ratio 1.100
        at ray_subsample=8 (band upper bound 1.050).

        2026-08-01 -- NICHE C8 REMOVED THIS TEST'S STIMULUS AT SOURCE, AND THE
        MEASUREMENT SAYS THAT IS CORRECT.

        PIN WAS: the check fires at the shipped defaults (ratio 1.100385).
        PIN IS NOW: it fires with ``REMAP_INVERSE_SUPPORT_BOUND = False`` --
        the library state it was calibrated in, original assertions word for
        word -- and with the bound ON it is silent because the field really is
        quieter, which is asserted too.

        THE HYPOTHESIS WAS CHECKED BEFORE IT WAS APPLIED, because this
        fixture's name claims a FOLD CAUSTIC and C8 must NOT be able to
        silence one: the bound only ever zeroes amplitude OUTSIDE the convex
        hull of the alive stop-passing exit landings, whereas a fold lives
        where the rays ARE.  Measured by
        ``validation/repro_traced_carrier_121/recon_em6_stimulus.py``:

        * The fixture DOES fold, and the diagnostic still says so.  Central
          differences on the EXACT traced landing lattice (no fit, no Newton,
          no upsample) give 32 adjacent-cell det J SIGN CHANGES over the 373
          stop-passing samples, det J spanning [-4.126e-01, +1.781e+00] with
          |det J| min/median 0.0489.  The library's fold-caustic warning fires
          with the bound ON and OFF alike -- asserted below.
        * But the ENERGY the check was reading was NOT the fold's.  Split the
          power the bound removes against the call's own exact ray bundle:
          **100.00 % lies outside the convex hull of every alive ray**, 0.00 %
          between the stop-passing and all-rays hulls, and **0.00 % inside the
          support** -- so nothing the rays reach was touched.  The power
          INSIDE the support is 0.81933 of the aperture-transmitted input with
          the bound on and off, to every printed digit.
        * ABSOLUTE ORACLE (raytrace + input field only, no wave model): the
          alive stop-passing rays carry 0.82619 of the power over the test's
          own disc -- rays beyond r = 1.145 mm die on the lens surfaces, well
          inside the 1.500 mm stop.  That is the true ratio ceiling.  The
          pre-C8 library returned 1.10039 (error +0.274) and the shipped one
          returns 1.01931 (error +0.193): **BETTER by 30 % of the error**, and
          the 0.81933 that survives inside the support is 99.17 % of the
          oracle, i.e. a discretisation DEFICIT, not a gain.

        So the stimulus is gone, the subject is not, and no bar moved."""
        N, ap = 256, 3e-3
        dx = 1.01 * ap / N
        E0, X, Y = _gauss(N, dx, 1.4e-3)
        presc = {'name': 'biconcave', 'aperture_diameter': ap,
                 'thicknesses': [3e-3],
                 'surfaces': [
                     {'radius': -3e-3, 'glass_before': 'air',
                      'glass_after': 'N-BK7', 'conic': 0.0,
                      'aspheric_coeffs': None},
                     {'radius': 3e-3, 'glass_before': 'N-BK7',
                      'glass_after': 'air', 'conic': 0.0,
                      'aspheric_coeffs': None}]}

        def _call(bound):
            old = _lens_traced.REMAP_INVERSE_SUPPORT_BOUND
            _lens_traced.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
            try:
                with warnings.catch_warnings(record=True) as rec:
                    warnings.simplefilter('always')
                    E = la.apply_real_lens_traced(
                        E0, prescription=presc, wavelength=_WL, dx=dx,
                        amplitude_model='ray_density', ray_subsample=8,
                        n_workers=1, parallel_amp=False,
                        on_undersample='silent', on_aperture_beam='silent')
            finally:
                _lens_traced.REMAP_INVERSE_SUPPORT_BOUND = old
            texts = [str(r.message) for r in rec]
            return (np.asarray(E),
                    [t for t in texts if 'energy self-check' in t],
                    [t for t in texts if 'fold caustic' in t])

        disc = (X ** 2 + Y ** 2) <= (ap / 2) ** 2
        p_ap = float((np.abs(E0[disc]) ** 2).sum())

        # --- the original pin, in the library state it was calibrated in ---
        E, msgs, folds = _call(False)
        ratio = float((np.abs(E) ** 2).sum()) / p_ap
        assert msgs, (
            f"energy self-check stayed silent at ratio {ratio:.4f} -- it must "
            f"flag a ray-density field that gained energy.")
        assert 'ray_subsample' in msgs[0] and 'band' in msgs[0]

        # --- niche C8: the manufactured light is gone, so the check is ------
        # honestly silent -- and the field really is quieter, not merely
        # unreported.  This is the assertion that stops a true positive being
        # replaced by a green test.
        E8, msgs8, folds8 = _call(True)
        ratio8 = float((np.abs(E8) ** 2).sum()) / p_ap
        assert msgs8 == [], msgs8
        assert ratio8 < ratio, (ratio, ratio8)
        assert ratio8 <= 1.0 + _lens_traced._RD_ENERGY_GAIN_TOL, ratio8
        assert np.all(np.abs(E8) <= np.abs(E) * (1 + 1e-12)), (
            'the support bound may only ever LOWER an amplitude')
        # ... and it is closer to the absolute geometric-transport ceiling,
        # which no wave model enters (see the docstring: 0.82619).
        assert abs(ratio8 - 0.82619) < abs(ratio - 0.82619)
        # --- and the FOLD is still diagnosed, in both states.  C8 must not
        # be able to silence a det J sign change: it only bounds the inverse
        # to the traced samples' support, and a fold lives inside it.
        assert folds, 'the fold-caustic diagnostic stopped firing'
        assert folds8, (
            'niche C8 silenced the FOLD diagnostic -- it must only remove '
            'amplitude outside the traced exit support, where no ray goes')

    def test_screen_mode_never_runs_the_check(self):
        """The default amplitude model is untouched (byte-compat)."""
        N, ap = 128, 3e-3
        dx = 2.2 * ap / N
        E0, _X, _Y = _gauss(N, dx, 1.2e-3)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            la.apply_real_lens_traced(
                E0, prescription=_fast_singlet(ap), wavelength=_WL, dx=dx,
                ray_subsample=8, n_workers=1, parallel_amp=False,
                on_undersample='silent', on_aperture_beam='silent')
        assert not [r for r in rec if 'energy self-check' in str(r.message)]


# ===========================================================================
# E-L1 / E-L2 / E-L3 -- the persistent worker pool
# ===========================================================================

class TestEL1EL2EL3WorkerPool:

    def test_el1_atexit_handler_registered_once_per_process(self):
        """Pre-fix ``atexit._ncallbacks()`` grew 2 -> 8 across five pool
        creations (one leaked ``close_worker_pool`` callback each)."""
        _lens_traced.close_worker_pool()
        base = atexit._ncallbacks()
        try:
            for i in range(5):
                _lens_traced.close_worker_pool()
                _lens_traced._get_persistent_worker_pool(1 + (i % 2))
            grown = atexit._ncallbacks()
        finally:
            _lens_traced.close_worker_pool()
        # The executor itself registers one handler on first construction;
        # ours must be registered at most once no matter how many pools.
        assert grown - base <= 2, (
            f"atexit callbacks grew by {grown - base} across five pool "
            f"creations; the handler must be registered exactly once.")

    def test_el2_lock_exists_at_module_scope_and_teardown_takes_it(self):
        """The lock must not be lazily built inside the getter (broken
        double-checked locking) and ``close_worker_pool`` must hold it while
        mutating the pool globals."""
        assert _lens_traced._PERSISTENT_POOL_LOCK is not None, (
            'the pool lock must be constructed at module scope')
        assert hasattr(_lens_traced._PERSISTENT_POOL_LOCK, 'acquire')
        src = inspect.getsource(_lens_traced.close_worker_pool)
        assert 'with _PERSISTENT_POOL_LOCK' in src, (
            'close_worker_pool mutates the pool globals without the lock')
        getter = inspect.getsource(_lens_traced._get_persistent_worker_pool)
        assert '_PERSISTENT_POOL_LOCK = threading.Lock()' not in getter, (
            'the lock is still built lazily inside the getter')

    def test_el3_a_worker_valueerror_is_not_swallowed(self):
        """A ValueError raised by ``fut.result()`` is a REAL worker fault; the
        old catch swallowed it and silently re-ran the identical work
        serially."""
        from concurrent.futures import Future

        class _Ex:
            def submit(self, fn, a):
                f = Future()
                f.set_exception(ValueError('worker blew up'))
                return f

        N, ap = 512, 3e-3
        dx = 2.2 * ap / N
        E0, _X, _Y = _gauss(N, dx, 1.2e-3)
        orig = _lens_traced._get_persistent_worker_pool
        _lens_traced._get_persistent_worker_pool = lambda nw: _Ex()
        try:
            with pytest.raises(ValueError, match='worker blew up'):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    la.apply_real_lens_traced(
                        E0, prescription=_fast_singlet(ap), wavelength=_WL,
                        dx=dx, ray_subsample=1, newton_fit='spline',
                        n_workers=4, parallel_amp=False,
                        newton_amp_mask_rel=0.0, on_undersample='silent',
                        on_aperture_beam='silent')
        finally:
            _lens_traced._get_persistent_worker_pool = orig

    def test_el3_broken_pool_falls_back_and_resets_the_cached_pool(self):
        """A genuine pool-infrastructure failure still falls back serially --
        and now DROPS the broken executor instead of caching it."""
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool

        class _Ex:
            def submit(self, fn, a):
                f = Future()
                f.set_exception(BrokenProcessPool('spawn blocked'))
                return f

        N, ap = 512, 3e-3
        dx = 2.2 * ap / N
        E0, _X, _Y = _gauss(N, dx, 1.2e-3)
        closed = []
        o1 = _lens_traced._get_persistent_worker_pool
        o2 = _lens_traced.close_worker_pool
        _lens_traced._get_persistent_worker_pool = lambda nw: _Ex()
        _lens_traced.close_worker_pool = lambda: closed.append(1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out = la.apply_real_lens_traced(
                    E0, prescription=_fast_singlet(ap), wavelength=_WL, dx=dx,
                    ray_subsample=1, newton_fit='spline', n_workers=4,
                    parallel_amp=False, newton_amp_mask_rel=0.0,
                    on_undersample='silent', on_aperture_beam='silent')
        finally:
            _lens_traced._get_persistent_worker_pool = o1
            _lens_traced.close_worker_pool = o2
        assert out.shape == (N, N)
        assert closed, (
            'a failed pool must be dropped from the module cache, otherwise '
            'every later call re-uses the broken executor')


# ===========================================================================
# E-L4 / E-L5 -- dead code
# ===========================================================================

class TestEL4EL5DeadCode:

    def test_dead_constants_and_scaffold_are_gone(self):
        assert not hasattr(lenses_mod, '_NEWTON_MAX_ITERS'), (
            'lenses._NEWTON_MAX_ITERS is dead AND its comment contradicted '
            'the shipped Newton cap')
        assert not hasattr(lenses_mod, '_NUMEXPR_MIN_SIZE')
        for name in ('NUMEXPR_AVAILABLE', '_ne', '_ensure_numexpr_loaded',
                     '_NUMEXPR_MIN_SIZE', '_surface_sag_general',
                     'surface_sag_general'):
            assert not hasattr(_lens_traced, name), (
                f'_lens_traced.{name} is dead scaffold and must be removed')

    def test_the_live_twins_survive(self):
        assert lenses_mod.NUMEXPR_AVAILABLE in (True, False)
        assert callable(lenses_mod._ensure_numexpr_loaded)
        assert real_mod._NUMEXPR_MIN_SIZE == 1 << 20
        assert _lens_traced._NEWTON_MAX_ITERS == 12
        assert callable(real_mod._surface_sag_general)

    def test_lens_real_no_longer_points_at_the_deleted_constant(self):
        src = inspect.getsource(real_mod)
        assert 'see lenses.py for rationale' not in src, (
            '_lens_real still cites lenses.py as the canonical home of '
            '_NUMEXPR_MIN_SIZE, which no longer defines it')
        assert 'ONLY live copy of' in src, (
            '_lens_real should now carry the _NUMEXPR_MIN_SIZE rationale '
            'itself')


# ===========================================================================
# E-L10 -- _radial_amp_sampler origin on an odd-N grid
# ===========================================================================

@pytest.mark.parametrize('N', [64, 65, 127, 128])
def test_el10_radial_sampler_is_accurate_on_both_grid_parities(N):
    """The module's grids are ``(arange(N) - N/2) * dx``, so the origin is at
    the FLOAT index N/2.  With the old ``c = N // 2`` anchor an odd-N sampler
    mislabelled each sample's radius by up to ~0.7 px: measured max amplitude
    error 5.106e-2 (best-fit radial shift -0.44 px) at N = 65 and N = 127,
    versus 3.5e-3 at N = 64 and N = 128."""
    dx, w = 1e-5, 8e-5
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    a = np.exp(-((X ** 2 + Y ** 2) / w ** 2)).astype(np.complex128)
    s = _lens_traced_uniform._radial_amp_sampler(a, dx)
    rp = (1.5 + np.arange(0, N // 2 - 3)) * dx
    err = float(np.max(np.abs(np.asarray(s(rp)) - np.exp(-(rp / w) ** 2))))
    assert err < 5e-3, (
        f"N={N}: radial sampler off by {err:.3e} on an exactly "
        f"rotationally-symmetric field; the origin anchor must come from the "
        f"``(arange(N) - N/2) * dx`` grid, not from ``N // 2``.")


# ===========================================================================
# E-L11 -- turbulence-screen frequency lattice on an odd-N grid
# ===========================================================================

def _turbulence_reference(N, dx, r0, seed):
    """Independent integer-DC-anchor implementation (the e29a8db convention)."""
    rng = np.random.default_rng(seed)
    df = 1.0 / (N * dx)
    fx = (np.arange(N) - N // 2) * df
    FX, FY = np.meshgrid(fx, fx)
    f_sq = FX ** 2 + FY ** 2
    psd = 0.023 * r0 ** (-5.0 / 3.0) * np.where(f_sq > 0, f_sq, 1.0) ** (
        -11.0 / 6.0)
    psd[N // 2, N // 2] = 0.0
    noise = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    pf = noise * (np.sqrt(2.0 * psd) * df)
    return np.real(
        np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(pf)))) * N ** 2


@pytest.mark.parametrize('N', [64, 65, 128, 129, 257])
def test_el11_turbulence_frequency_lattice_is_dc_anchored(N):
    """``psd[N//2, N//2] = 0`` and ``ifftshift`` both assume ``fx[N//2] == 0``,
    which the float ``N/2`` anchor only satisfies for even N.

    Pre-fix, odd N had ``fx[N//2] = -df/2``: the DC kill deleted a real bin,
    the lowest surviving |f| carried 3.5636x the Kolmogorov PSD of the correct
    lattice's first non-DC bin, and the screen differed from the integer-anchor
    reference by 75-86 % of its own peak.  Even N was already exact and must
    stay BIT-identical.
    """
    dx, r0, seed = 1e-3, 0.1, 7
    got = elements_mod.generate_turbulence_screen(N, dx, r0, seed=seed)
    ref = _turbulence_reference(N, dx, r0, seed)
    assert np.array_equal(got, ref), (
        f"N={N}: max|screen - integer-anchor reference| = "
        f"{np.max(np.abs(got - ref)):.6e}")


def test_el11_screen_frequency_grid_matches_fftshift_fftfreq():
    """The lattice the docstring now promises."""
    src = inspect.getsource(elements_mod.generate_turbulence_screen)
    assert 'np.arange(N) - N // 2' in src, (
        'generate_turbulence_screen must use the integer DC anchor')


# ===========================================================================
# E-L18 -- the coronagraph namespace module
# ===========================================================================

class TestEL18CoronagraphNamespace:

    def test_every_reexport_is_the_same_object_as_in_elements(self):
        assert coronagraph_ns.__all__, 'namespace module lost its __all__'
        for name in coronagraph_ns.__all__:
            got = getattr(coronagraph_ns, name, None)
            assert got is not None, f'{name} missing from the namespace module'
            assert got is getattr(la.elements, name), (
                f'{name} re-export drifted from lumenairy.elements')

    def test_no_coronagraph_factory_is_left_out(self):
        """Guard against drift: every coronagraph-family factory defined in
        ``elements.py`` must appear in the namespace module."""
        expected = {
            'apply_lyot_focal_plane_mask', 'apply_vortex_phase_mask',
            'apply_lyot_stop', 'apply_apodized_pupil',
            'create_four_quadrant_phase_mask',
            'create_eight_octant_phase_mask',
        }
        assert expected <= set(coronagraph_ns.__all__), (
            f'missing re-exports: {sorted(expected - set(coronagraph_ns.__all__))}')

    def test_docstring_count_matches_the_export_count(self):
        doc = _squash(coronagraph_ns.__doc__)
        assert 'four coronagraph element factories' not in doc, (
            'the module docstring still says "four" while re-exporting six')


# ===========================================================================
# E-L19 -- apply_apodized_pupil gaussian sigma
# ===========================================================================

def test_el19_gaussian_apodization_sigma_is_optional_in_doc_and_code():
    doc = _squash(la.apply_apodized_pupil.__doc__)
    assert "'gaussian'`` : ``T(rho) = exp(-(r/sigma)^2 / 2)``.  Requires" \
        not in la.apply_apodized_pupil.__doc__, 'stale "Requires sigma" text'
    assert 'Requires ``sigma``' not in doc, (
        "the 'gaussian' bullet must not say sigma is required -- the code "
        "defaults it to diameter/6")
    # ... and the code really does default it.
    N, dx, D = 64, 1e-4, 4e-3
    E0 = np.ones((N, N), dtype=np.complex128)
    a = la.apply_apodized_pupil(E0, dx, D, apodization='gaussian')
    b = la.apply_apodized_pupil(E0, dx, D, apodization='gaussian',
                                sigma=D / 6.0)
    assert np.array_equal(a, b)


# ===========================================================================
# E-L21 -- apply_real_lens_gbd(jacobian=...)
# ===========================================================================

class TestEL21GbdJacobian:

    def _presc(self):
        return {'surfaces': [
            {'radius': 50e-3, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': np.inf, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'}],
            'thicknesses': [3e-3], 'aperture_diameter': 4e-3}

    def _field(self):
        N, dx = 64, 100e-6
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        return np.exp(-((X ** 2 + Y ** 2) / (1.2e-3) ** 2)
                      ).astype(np.complex128), dx

    @pytest.mark.parametrize('per_surface', [True, False])
    def test_junk_jacobian_raises_on_both_branches(self, per_surface):
        """Pre-fix the paraxial branch accepted ``jacobian='BOGUS'`` silently
        (bit-identical output, no diagnostic) while the per-surface branch
        raised."""
        E0, dx = self._field()
        with pytest.raises(ValueError, match='jacobian'):
            la.apply_real_lens_gbd(
                E0, prescription=self._presc(), wavelength=632.8e-9, dx=dx,
                per_surface=per_surface, jacobian='BOGUS', sample_step=8,
                beamlets_per_aperture=8)

    def test_inert_jacobian_on_the_paraxial_branch_warns(self):
        E0, dx = self._field()
        with pytest.warns(RuntimeWarning, match='jacobian'):
            la.apply_real_lens_gbd(
                E0, prescription=self._presc(), wavelength=632.8e-9, dx=dx,
                per_surface=False, jacobian='fd', sample_step=8,
                beamlets_per_aperture=8)

    def test_default_jacobian_on_the_paraxial_branch_is_silent(self):
        E0, dx = self._field()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            la.apply_real_lens_gbd(
                E0, prescription=self._presc(), wavelength=632.8e-9, dx=dx,
                per_surface=False, sample_step=8, beamlets_per_aperture=8)
        assert not [r for r in rec if 'jacobian' in str(r.message)]

    def test_jacobian_is_documented(self):
        doc = _squash(la.apply_real_lens_gbd.__doc__)
        assert 'jacobian' in doc and 'per_surface=True' in doc, (
            'apply_real_lens_gbd must document jacobian and its scope')


# ===========================================================================
# Documentation contracts -- E-M1 / E-M9 / E-M11 / E-M12 / E-M15
# ===========================================================================

def test_em1_seidel_poly_order_doc_matches_the_signature():
    default = (inspect.signature(la.apply_real_lens)
               .parameters['seidel_poly_order'].default)
    assert default == 6
    doc = _squash(la.apply_real_lens.__doc__)
    assert f'seidel_poly_order : int, default {default}' in doc, (
        f'the docstring must state the real default ({default})')
    assert 'seidel_poly_order : int, default 8' not in doc


def test_em9_keyword_only_claim_is_scoped_and_names_the_exceptions():
    doc = _squash(la.apply_thin_lens.__doc__)
    assert 'not the whole library' in doc.lower(), (
        'the keyword-only note must state its scope')
    for name in ('apply_axicon', 'apply_mirror', 'apply_aperture',
                 'apply_zernike_aberration', 'thin_grating_efficiency_1d'):
        assert name in doc, f'the scope note must name {name}'


@pytest.mark.parametrize('fn_name', ['apply_axicon', 'apply_mirror',
                                     'apply_aperture',
                                     'apply_zernike_aberration'])
def test_em9_positional_float_entry_points_carry_a_caution(fn_name):
    fn = getattr(la, fn_name)
    params = list(inspect.signature(fn).parameters.values())
    # The API must NOT have been broken into keyword-only.
    assert any(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
               for p in params[1:]), (
        f'{fn_name} was made keyword-only -- that is an API break the audit '
        f'explicitly ruled out')
    doc = _squash(fn.__doc__)
    assert 'positional-or-keyword' in doc.lower(), (
        f'{fn_name} must say its arguments are positional-or-keyword')
    assert 'by keyword' in doc, (
        f'{fn_name} must caution that its float arguments bind positionally')


def test_em11_zernike_example_states_the_right_aperture_size():
    doc = _squash(la.apply_zernike_aberration.__doc__)
    assert '5mm aperture' not in doc, (
        'the example still labels a 5 mm RADIUS as a "5mm aperture"')
    assert 'aperture_radius=5e-3' in doc and '10 mm' in doc


def test_em12_prescription_radius_sign_is_documented_and_true():
    doc = _squash(la.apply_real_lens.__doc__)
    assert 'SIGNED radius of curvature' in doc
    assert 'transmission (downstream) side' in doc, (
        "the prescription ``radius`` key must document its sign convention")
    # ... and the convention the doc states is the one the library implements.
    from lumenairy.raytrace import system_abcd_prescription
    n = la.get_glass_index('N-BK7', 632.8e-9)
    R = 50e-3

    def presc(r0, r1):
        return {'surfaces': [
            {'radius': r0, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'air', 'glass_after': 'N-BK7'},
            {'radius': r1, 'conic': 0.0, 'aspheric_coeffs': None,
             'glass_before': 'N-BK7', 'glass_after': 'air'}],
            'thicknesses': [3e-3], 'aperture_diameter': 4e-3}

    efl_pos = system_abcd_prescription(presc(R, np.inf), 632.8e-9)[1]
    efl_neg = system_abcd_prescription(presc(-R, np.inf), 632.8e-9)[1]
    assert efl_pos == pytest.approx(R / (n - 1), rel=1e-9), (
        'surfaces[0].radius > 0 must be the CONVERGING (lensmaker R>0) sense')
    assert efl_neg == pytest.approx(-R / (n - 1), rel=1e-9)


def test_em15_uniform_fallback_warning_names_its_own_function():
    src = inspect.getsource(
        _lens_traced_uniform.apply_real_lens_traced_uniform)
    assert 'apply_real_lens_traced_uniform (also reached via' in src, (
        "the fallback warning must name apply_real_lens_traced_uniform -- it "
        "fires from a public entry point that has no ``caustic`` kwarg")
