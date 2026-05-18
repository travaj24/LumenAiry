"""Tests for the v4.14.3 audit-fix Agent-A changes.

Scope: ``lumenairy/io/storage.py``, ``lumenairy/elements/doe.py``,
``lumenairy/propagators/propagation.py`` (Agent-A files only).

What this module pins
---------------------

* **A.1 -- P0-NEW-1 ``append_plane_h5`` / ``_zarr_append_plane``
  atomicity.**  The pre-4.14.3 code created the dataset first and
  then bumped ``n_planes`` -- a crash between the two left an orphan
  dataset whose name collided with the next append's computed name.
  The Zarr variant additionally passed ``overwrite=True`` so the
  next append silently clobbered the orphan (data loss).  v4.14.3
  bumps ``n_planes`` BEFORE the create and rolls back on failure,
  so a crash leaves the slot reserved (next append computes ``n + 1``
  and skips it).  Multi-process concurrency is still unsupported
  -- documented in the docstring.

* **A.2 -- P0-NEW-2 ``makedammann2d`` SI heuristic upper bound.**
  The v4.14.2 heuristic treated ``periodx > 1e-3`` as legacy
  micrometres and rescaled by ``1e-6``.  Real-world THz / MMW DOE
  designs use mm-scale SI periods (``periodx=5e-3``) and far-IR
  wavelengths (``waveln=1.1e-3``).  Such a call silently produced
  5 nm garbage with a confusing ``DeprecationWarning`` masking the
  bug.  v4.14.3 adds (1) a hard ``ValueError`` for inputs above 1 m
  (unambiguously wrong in any unit system); and (2) an explicit
  ``_legacy_units`` kwarg with values ``'auto'`` / ``'um'`` / ``'SI'``
  so THz / MMW users can opt out of the heuristic via
  ``_legacy_units='SI'``.

* **A.3 -- P1-NEW-1 ``clear_asm_caches`` chains
  ``clear_lg_polynomial_cache``.**  The v4.14.2 docstring claimed
  ``clear_asm_caches`` left the propagator-adjacent state completely
  pristine -- matching :func:`lumenairy_context` -- but missed
  ``_lg_polynomial_items`` (an ``lru_cache`` in
  ``propagators/asymptotic.py``).  v4.14.3 chains it.

Author: Agent A -- v4.14.3.
"""
from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest


# ============================================================================
# A.1 -- ``append_plane_h5`` / ``_zarr_append_plane`` atomicity
# ============================================================================

class TestA1AppendPlaneAtomicity:
    """Pin the attribute-before-create discipline that keeps the
    append path atomic under a mid-call crash."""

    def test_append_plane_h5_attr_written_before_dataset(self):
        """Pin the rollback contract for ``append_plane_h5``.

        Monkey-patch ``create_dataset`` to raise; assert the
        post-call ``n_planes`` attribute equals the pre-call value
        (rolled back) and the original (pre-crash) plane is still
        intact.  Pre-4.14.3 the create-then-bump order meant the
        attribute was never written on crash; the bug being pinned
        here is the v4.14.3 reverse: bump first AND roll back on
        failure so the slot is neither stranded nor silently
        clobbered.
        """
        import h5py
        from lumenairy.io.storage import append_plane_h5

        E = np.ones((8, 8), dtype=np.complex128)
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, 'crash.h5')
            # First append succeeds, baseline state.
            append_plane_h5(fp, E, dx=1e-6, label='before')
            with h5py.File(fp, 'r') as f:
                n_before = int(f['planes'].attrs['n_planes'])
            assert n_before == 1, (
                f'Expected n_planes=1 after one append, got {n_before}.'
            )

            # Patch h5py.Group.create_dataset to raise on the next call.
            # We patch via the h5py module rather than wrapping the
            # function because ``append_plane_h5`` opens its own file
            # handle inside the call.
            orig_create_dataset = h5py.Group.create_dataset
            def _boom(self, *args, **kwargs):
                raise RuntimeError('simulated mid-append crash')
            h5py.Group.create_dataset = _boom
            try:
                with pytest.raises(RuntimeError, match='simulated'):
                    append_plane_h5(fp, E, dx=1e-6, label='will-crash')
            finally:
                h5py.Group.create_dataset = orig_create_dataset

            # Post-crash state: ``n_planes`` rolled back to the
            # pre-crash value so the next legitimate append goes to
            # the same slot.
            with h5py.File(fp, 'r') as f:
                n_after = int(f['planes'].attrs['n_planes'])
                assert n_after == n_before, (
                    f'n_planes not rolled back: pre={n_before}, '
                    f'post={n_after}.  v4.14.3 contract: on '
                    f'create_dataset failure, n_planes must be '
                    f'restored to its pre-call value.'
                )
                # The pre-crash plane is intact.
                assert 'plane_00' in f['planes']
                assert (f['planes/plane_00'].attrs['label']
                        in ('before', b'before'))

            # A subsequent successful append goes to plane_01 (the
            # slot reserved by the rolled-back attribute).
            append_plane_h5(fp, E, dx=1e-6, label='after')
            with h5py.File(fp, 'r') as f:
                assert int(f['planes'].attrs['n_planes']) == 2
                assert 'plane_01' in f['planes']

    def test_append_plane_h5_docstring_documents_multiprocess_limit(self):
        """The v4.14.3 docstring must call out the multi-process
        restriction so users do not assume HDF5 SWMR is being used
        under the hood.  This is the audit's "documented for sequential
        use" assessment."""
        from lumenairy.io.storage import append_plane_h5
        doc = append_plane_h5.__doc__
        assert doc is not None
        # The atomicity contract is explained.
        assert 'Atomicity' in doc or 'atomicity' in doc, (
            'append_plane_h5 docstring must explain the atomicity '
            'contract introduced in v4.14.3.'
        )
        # The multi-process restriction is called out.
        assert 'multi-process' in doc.lower() or 'multiprocess' in doc.lower(), (
            'append_plane_h5 docstring must document the multi-process '
            'restriction (single-writer only).'
        )

    @pytest.mark.skipif(
        not pytest.importorskip(
            'zarr', reason='zarr not installed; _zarr_append_plane '
            'is optional'),
        reason='zarr not installed'
    )
    def test_zarr_append_plane_no_silent_clobber(self):
        """Pin that a partial-state Zarr store (orphan ``plane_00``
        with ``n_planes=0``) is NOT silently overwritten by the next
        legitimate append.  The pre-4.14.3 ``overwrite=True`` flag
        on ``create_array`` was the silent-clobber vector; v4.14.3
        drops it.

        Two acceptable behaviours: (a) the attribute-before-create
        order means the next append computes ``n + 1`` from the
        already-bumped attribute and skips the orphan (preferred);
        or (b) ``create_array`` raises on the existing slot (the
        ``overwrite=False`` default).  The bug-being-pinned is the
        third behaviour: silently overwriting the orphan."""
        import zarr
        from lumenairy.io.storage import _zarr_append_plane

        E_real = np.ones((4, 4), dtype=np.complex128)
        E_orphan_marker = (np.arange(16, dtype=np.complex128)
                           .reshape(4, 4) + 1000.0)

        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, 'partial.zarr')
            # Synthesize a partial-state store: a ``plane_00`` exists
            # but ``n_planes`` is still 0 -- the exact state left
            # behind by a pre-4.14.3 mid-call crash.
            grp = zarr.open_group(fp, mode='w')
            planes = grp.create_group('planes')
            planes.attrs['n_planes'] = 0
            planes.create_array(
                name='plane_00', data=E_orphan_marker,
                chunks=(4, 4),
            )

            # Read back the orphan to confirm setup.
            grp_r = zarr.open_group(fp, mode='r')
            orphan_before = np.array(grp_r['planes/plane_00'][:])
            assert np.allclose(orphan_before, E_orphan_marker), (
                'Test setup is broken: orphan plane_00 not written.'
            )

            # The legitimate next append:
            try:
                _zarr_append_plane(fp, E_real, dx=1e-6, label='real')
            except Exception:
                # Behaviour (b): create raised on the existing slot.
                # That's an acceptable v4.14.3 outcome -- still NOT
                # the silent-clobber bug we're pinning against.
                grp_r2 = zarr.open_group(fp, mode='r')
                orphan_after = np.array(grp_r2['planes/plane_00'][:])
                assert np.allclose(orphan_after, E_orphan_marker), (
                    'v4.14.3 contract: a raise on the existing slot '
                    'must NOT have clobbered the orphan first.'
                )
                return

            # Behaviour (a): the append succeeded.  The orphan was
            # NOT overwritten -- the real data landed in plane_01.
            grp_r3 = zarr.open_group(fp, mode='r')
            orphan_after = np.array(grp_r3['planes/plane_00'][:])
            assert np.allclose(orphan_after, E_orphan_marker), (
                'SILENT-CLOBBER BUG (v4.14.2): a partial-state Zarr '
                'store had its orphan plane_00 silently overwritten '
                'by the next append.  v4.14.3 contract: either skip '
                'the orphan and use plane_01, OR raise.'
            )
            assert 'plane_01' in grp_r3['planes']


# ============================================================================
# A.2 -- ``makedammann2d`` SI heuristic upper bound + ``_legacy_units``
# ============================================================================

class TestA2MakedammannSIHeuristic:
    """Pin the v4.14.3 upper-bound + ``_legacy_units`` API."""

    def test_makedammann2d_rejects_meter_scale_input(self):
        """Pin the ``> 1 m`` upper-bound guard.  A meter-scale
        ``periodx`` is unambiguously wrong in any unit system and
        must raise rather than being silently rescaled."""
        from lumenairy.elements.doe import makedammann2d

        with pytest.raises(ValueError, match='periodx'):
            makedammann2d(periodx=2.0, periody=61e-6, waveln=1.31e-6,
                          itr=1, plot=False)
        with pytest.raises(ValueError, match='periody'):
            makedammann2d(periodx=61e-6, periody=2.0, waveln=1.31e-6,
                          itr=1, plot=False)
        with pytest.raises(ValueError, match='waveln'):
            makedammann2d(periodx=61e-6, periody=61e-6, waveln=1.5,
                          itr=1, plot=False)

    def test_makedammann2d_explicit_si_units_no_rescale(self):
        """Pin that ``_legacy_units='SI'`` accepts mm-scale SI inputs
        without triggering the heuristic.  The output's
        ``cell_pixel_size`` should match a hand-built SI reference
        rather than being silently rescaled by ``1e-6``."""
        from lumenairy.elements.doe import makedammann2d

        # THz / MMW design: 5 mm grating period at 1.1 mm far-IR.
        periodx = 5e-3
        periody = 5e-3
        waveln = 1.1e-3
        wavsamp = 0.5

        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # _legacy_units='SI' must NOT trigger the heuristic
            # warning, even though every value exceeds 1 mm.
            nf, ff, (dx_out, dy_out) = makedammann2d(
                periodx=periodx, periody=periody, waveln=waveln,
                wavsamp=wavsamp, diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=0,
                _legacy_units='SI',
            )

        # Hand-built SI reference: ndifordersx = ceil(periodx /
        # (wavsamp * waveln) * 0.5) * 2; samplingx = periodx /
        # ndifordersx.  No 1e-6 rescale.
        ndifordersx_ref = int(
            np.ceil(periodx / (wavsamp * waveln) * 0.5)) * 2
        samplingx_ref = periodx / ndifordersx_ref
        # The output ``cell_pixel_size`` is reported in metres.  Pin
        # the order-of-magnitude: dx_out should be mm-scale, not
        # nm-scale.  An accidental rescale by 1e-6 would land it at
        # ~ 1e-9.
        assert dx_out > 1e-5, (
            f'cell_pixel_size={dx_out} m is sub-micron -- the '
            f'_legacy_units="SI" path must NOT have rescaled by 1e-6.  '
            f'Expected mm-scale ({samplingx_ref:.3e} m).'
        )
        # Tighter check: the output matches the SI reference within
        # numerical tolerance.
        assert np.isclose(dx_out, samplingx_ref, rtol=1e-9), (
            f'dx_out={dx_out} != SI reference samplingx={samplingx_ref}.'
        )

    def test_makedammann2d_explicit_um_units_no_warning(self):
        """Pin that ``_legacy_units='um'`` rescales unconditionally
        without emitting a ``DeprecationWarning`` -- the explicit
        opt-in form is a clean migration path."""
        from lumenairy.elements.doe import makedammann2d

        # Legacy micrometre call: periodx=61.0 (61 um).
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            # _legacy_units='um' must NOT emit a deprecation
            # warning -- the caller explicitly acknowledged the unit
            # convention.
            nf, ff, (dx_out, dy_out) = makedammann2d(
                periodx=61.0, periody=61.0, waveln=1.31,
                diforders=np.ones((2, 2)),
                itr=2, plot=False, seed=0,
                _legacy_units='um',
            )
        # The output should be the same as the SI-equivalent call:
        # rescaled to 61e-6 m / 1.31e-6 m internally.
        assert dx_out > 0
        # Quick sanity: the rescale produced micrometre-scale output,
        # not metre-scale.
        assert dx_out < 1e-3, (
            f'Expected ~um-scale cell_pixel_size, got {dx_out} m.'
        )


# ============================================================================
# A.3 -- ``clear_asm_caches`` chains ``clear_lg_polynomial_cache``
# ============================================================================

class TestA3ClearAsmCachesLgPolynomial:
    """Pin that the v4.14.3 ``clear_asm_caches`` chain reaches the
    LG polynomial coefficient cache, closing the v4.14.2 gap."""

    def test_clear_asm_caches_clears_lg_polynomial_cache(self):
        """Populate ``_lg_polynomial_items`` (the lru_cache wrapped by
        :func:`lumenairy.propagators.asymptotic.lg_polynomial`), call
        ``clear_asm_caches``, then assert the cache is empty."""
        from lumenairy.propagators.asymptotic import (
            _lg_polynomial_items,
            lg_polynomial,
        )
        from lumenairy.propagators.propagation import clear_asm_caches

        # Populate the cache by calling the public entry point.  The
        # exact (p, ell, w) values do not matter for cache-clearing
        # purposes; we just need at least one entry.
        lg_polynomial(0, 0, 1e-3)
        lg_polynomial(1, 0, 1e-3)
        assert _lg_polynomial_items.cache_info().currsize > 0, (
            'Test setup failure: lg_polynomial did not populate the '
            'lru_cache.  Cannot pin clear-on-call without a non-empty '
            'starting state.'
        )

        clear_asm_caches()

        assert _lg_polynomial_items.cache_info().currsize == 0, (
            'v4.14.3 contract: clear_asm_caches must chain '
            'clear_lg_polynomial_cache so the LG polynomial '
            'coefficient lru_cache is drained.  This was the 8th '
            'sibling cache missed by the v4.14.2 expansion.'
        )

    def test_clear_asm_caches_docstring_lists_lg_polynomial(self):
        """The v4.14.3 docstring must reference
        ``clear_lg_polynomial_cache`` by name so users searching the
        docs for the entry point can find it.  Mirrors the
        v4.14.2 docstring-pinning test that Agent C added for the
        previous five chained caches."""
        from lumenairy.propagators.propagation import clear_asm_caches
        doc = clear_asm_caches.__doc__
        assert doc is not None
        assert 'clear_lg_polynomial_cache' in doc, (
            'clear_asm_caches docstring missing reference to '
            'clear_lg_polynomial_cache; v4.14.3 must list every '
            'chained cache.'
        )

    def test_clear_asm_caches_drains_all_eight_caches(self):
        """Extend the v4.14.2 "combined drain" promise to include the
        LG polynomial cache.  Agent C's test in
        ``test_audit_fixes_v4_14_2_agent_c.py`` covers the seven
        previously-chained caches; this test adds the 8th
        (``_lg_polynomial_items``).  Together they pin that a single
        ``clear_asm_caches()`` call leaves the propagator-adjacent
        state completely pristine -- matching
        :func:`lumenairy.lumenairy_context`."""
        from lumenairy.propagators.asymptotic import (
            _lg_polynomial_items,
            lg_polynomial,
        )
        from lumenairy.propagators.propagation import (
            _BANDLIMIT_CACHE,
            _FREQ_GRID_CACHE,
            _H_CACHE,
            clear_asm_caches,
        )

        # Populate the LG polynomial cache + the local propagation
        # caches.  We do not need to populate every sibling cache
        # -- Agent C's test covers those; here we just pin that the
        # 8th cache is included in the drain.
        lg_polynomial(0, 0, 1e-3)
        lg_polynomial(2, 1, 2e-3)
        # Populate a freq-grid entry by computing one ASM step.
        from lumenairy.propagators.propagation import (
            angular_spectrum_propagate,
        )
        E = np.ones((16, 16), dtype=np.complex128)
        angular_spectrum_propagate(E, z=1e-3, wavelength=1.31e-6,
                                   dx=1e-6)
        assert _lg_polynomial_items.cache_info().currsize > 0
        assert len(_FREQ_GRID_CACHE) > 0

        clear_asm_caches()

        assert _lg_polynomial_items.cache_info().currsize == 0
        assert len(_FREQ_GRID_CACHE) == 0
        assert len(_BANDLIMIT_CACHE) == 0
        assert len(_H_CACHE) == 0
