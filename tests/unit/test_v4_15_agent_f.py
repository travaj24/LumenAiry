"""v4.15.0 Agent F -- P2/P3 sweep tests.

Covers the P2 and P3 findings from the v4.14.2 audit
(`docs/audits/AUDIT_V4_14_2_2026_05_17.md`) that Agent F closed:

* F.1 P2 -- ``lumenairy_context(clear_caches_on_exit=True)`` no longer
  redundantly fans out to 6 sibling cache clears; a single
  ``clear_asm_caches()`` call drains the whole chain.
* F.1 P2 -- HDF5 ``create_dataset`` and Zarr ``create_array`` sites
  stamp a ``lumenairy_version`` attribute on every write.  This
  pin covers the HDF5 paths; Zarr is gated under the optional
  ``zarr`` import.
* F.1 P2 -- ``create_multi_field_sources`` now appears in the
  ``_validate_grid_params`` factory list (no longer transitively
  validated through the inner ``create_tilted_plane_wave`` call).
* F.1 P3 -- ``create_led_source`` "too many positional arguments"
  message rewritten for clarity.
* F.4 -- README ``## Cookbook`` section exists with at least the 6
  v4.14.0 functions; representative example blocks parse as
  syntactically valid Python.
* F.5 -- CHANGELOG line-citation drift fix: the v4.14.2 entry now
  cites the actual ``optimize/core.py`` line numbers for the
  ``ToleranceAwareMerit._evaluate_perturbed`` and
  ``MatchIdealSystem._make_source`` sentinel-aware branches.

Author: Andrew Traverso -- v4.15.0 / Agent F
"""
from __future__ import annotations

import inspect
import os
import re
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import lumenairy as la

# v4.15.4 (AUDIT_V4_15_3 P3-NEW-F2-4): the
# ``test_too_many_positional_message_spells_both_forms`` test
# legitimately invokes ``create_led_source(...)`` with the legacy
# positional default to verify the v4.14.2 ``DeprecationWarning``
# message body.  Pin DeprecationWarning back to ``default`` for this
# module so the suite stays clean under strict
# ``-W error::DeprecationWarning``.
pytestmark = pytest.mark.filterwarnings(
    'default::DeprecationWarning',
)


# ============================================================================
# Helpers
# ============================================================================

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CHANGELOG_PATH = _REPO_ROOT / 'CHANGELOG.md'
_README_PATH = _REPO_ROOT / 'README.md'


# ============================================================================
# F.1 P2 -- lumenairy_context redundant call elimination
# ============================================================================

class TestF1LumenairyContextRedundantCallEliminated:
    """v4.14.2 audit P2: ``lumenairy_context(clear_caches_on_exit=True)``
    previously called ``clear_asm_caches`` PLUS 7 sibling
    ``clear_*`` helpers explicitly, costing 6 redundant lock
    acquisitions per context-manager exit (v4.14.3 added the 7th
    sibling -- ``clear_lg_polynomial_cache`` -- chained into
    ``clear_asm_caches`` so all sibling fan-outs are now redundant).

    The fix: ``_context.py`` collapses the fan-out to a single
    ``clear_asm_caches()`` call.  This test asserts:

    1. Each sibling clear helper is called EXACTLY ONCE per exit
       (one call from inside ``clear_asm_caches``'s chain).
    2. ``clear_asm_caches`` itself is called exactly once.
    """

    def test_clear_asm_caches_called_once_per_exit(self):
        """Single call (not double) -- the redundancy is gone."""
        import lumenairy._context as ctx_mod
        from lumenairy.propagators import propagation
        counter = {'n': 0}
        real_clear = propagation.clear_asm_caches

        def counting_clear():
            counter['n'] += 1
            real_clear()

        with mock.patch.object(propagation, 'clear_asm_caches',
                               new=counting_clear):
            # The _context module imports lazily INSIDE the
            # context-manager finally block, so each ``with`` call
            # picks up the patched version.
            with la.lumenairy_context(clear_caches_on_exit=True):
                pass
        assert counter['n'] == 1, (
            f'Expected clear_asm_caches to be called exactly once '
            f'per lumenairy_context exit, got {counter["n"]}.')

    def test_sibling_clear_helpers_called_once_via_chain(self):
        """Each sibling clear helper fires exactly once per exit --
        from inside ``clear_asm_caches``'s chain, NOT from
        ``_context.py``'s open-coded fan-out.

        Pre-v4.15 ``_context.py`` called both ``clear_asm_caches``
        AND each of 7 sibling helpers; the sibling helpers were ALSO
        called from inside ``clear_asm_caches``'s chain, so each
        sibling was hit twice -- 6 redundant calls plus the 7th
        depending on which round added it last.  The collapsed
        single call to ``clear_asm_caches`` should fire each
        sibling exactly once via the chain.
        """
        # v5.1.0 (Wave-4 integration / Agent G split): the
        # ``clear_zernike_basis_cache`` function moved from
        # ``analysis/core.py`` to ``analysis/zernike.py``.  The cache
        # registry's late-binding lambda resolves the clearer at the
        # canonical submodule location; mock.patch.object on the old
        # ``analysis.core`` re-export no longer intercepts.  Patch at
        # the canonical submodule location to match registry resolution.
        from lumenairy.analysis import zernike as zernike_mod
        from lumenairy.propagators import asymptotic_modes as asym_mod  # v5.1.0 split: clear_lg_polynomial_cache moved here
        counter_z = {'n': 0}
        counter_lg = {'n': 0}
        real_z = zernike_mod.clear_zernike_basis_cache
        real_lg = asym_mod.clear_lg_polynomial_cache

        def counting_z():
            counter_z['n'] += 1
            real_z()

        def counting_lg():
            counter_lg['n'] += 1
            real_lg()

        with mock.patch.object(zernike_mod, 'clear_zernike_basis_cache',
                               new=counting_z), \
             mock.patch.object(asym_mod, 'clear_lg_polynomial_cache',
                               new=counting_lg):
            with la.lumenairy_context(clear_caches_on_exit=True):
                pass
        assert counter_z['n'] == 1, (
            f'Expected clear_zernike_basis_cache to fire exactly '
            f'once (via the clear_asm_caches chain), got '
            f'{counter_z["n"]} -- the v4.15 redundant-call '
            f'elimination is incomplete.')
        assert counter_lg['n'] == 1, (
            f'Expected clear_lg_polynomial_cache to fire exactly '
            f'once (via the clear_asm_caches chain), got '
            f'{counter_lg["n"]} -- the v4.15 redundant-call '
            f'elimination is incomplete.')

    def test_context_module_chains_via_clear_asm_caches_on_happy_path(self):
        """The body of ``lumenairy_context`` routes the happy path
        through a single ``clear_asm_caches()`` call.

        Pre-v4.15 the block open-coded ``clear_asm_caches`` PLUS 7
        sibling clears, causing 6 redundant lock acquisitions per
        exit.  v4.15 collapses the happy path to a single call;
        the sibling fan-out only fires as a defence-in-depth
        fallback when the chain path itself raises (v4.13.1 P1-E
        contract).
        """
        from lumenairy import _context as ctx_mod
        src = inspect.getsource(ctx_mod.lumenairy_context)
        # Both paths must appear in source.  The happy path = first
        # call to clear_asm_caches; the fallback = the 7 sibling
        # clears guarded by the ``_chain_called`` flag.
        assert 'clear_asm_caches' in src, (
            'lumenairy_context body lost its ``clear_asm_caches`` '
            'happy-path call.')
        assert '_chain_called' in src, (
            'lumenairy_context body lost the ``_chain_called`` flag '
            'that gates the v4.13.1 P1-E defence-in-depth fallback.  '
            'The chain-only happy path must NOT silently strand the '
            'sibling clears when propagation fails to import.')

    def test_happy_path_does_not_call_siblings_directly(self):
        """On a healthy install (``clear_asm_caches`` importable),
        the sibling fan-out path is NOT entered.

        The v4.15 simplification claim is: redundant calls
        eliminated on the happy path.  This test instruments the
        sibling clear helpers and asserts they fire from inside
        ``clear_asm_caches``'s chain (i.e. once each), NOT also
        directly from ``_context.py``'s fan-out block (which
        would double-count to 2 each).
        """
        # v5.1.0 (Wave-4 integration / Agent G split): same as the
        # sibling test above -- patch at the canonical submodule
        # location so the cache registry's late-binding lambda
        # picks up the mock.
        from lumenairy.analysis import zernike as zernike_mod
        from lumenairy.propagators import asymptotic_modes as asym_mod  # v5.1.0 split: clear_lg_polynomial_cache moved here
        from lumenairy.analysis import phase_retrieval as pr_mod
        # Counters for 3 representative sibling clearers.
        counts = {'z': 0, 'lg': 0, 'pr': 0}
        real_z = zernike_mod.clear_zernike_basis_cache
        real_lg = asym_mod.clear_lg_polynomial_cache
        real_pr = pr_mod.clear_phase_retrieval_caches

        def cz():
            counts['z'] += 1
            real_z()

        def clg():
            counts['lg'] += 1
            real_lg()

        def cpr():
            counts['pr'] += 1
            real_pr()

        with mock.patch.object(zernike_mod, 'clear_zernike_basis_cache',
                               new=cz), \
             mock.patch.object(asym_mod, 'clear_lg_polynomial_cache',
                               new=clg), \
             mock.patch.object(pr_mod, 'clear_phase_retrieval_caches',
                               new=cpr):
            with la.lumenairy_context(clear_caches_on_exit=True):
                pass
        # Each sibling must fire EXACTLY ONCE (via the chain) on
        # the happy path; pre-v4.15 they fired twice (once via the
        # chain, once via the open-coded fan-out).
        for name, c in counts.items():
            assert c == 1, (
                f'Sibling clearer {name!r} fired {c} times on the '
                f'happy path; expected exactly 1.  Either the v4.15 '
                f'redundancy elimination is incomplete (count > 1) '
                f'or the canonical chain regressed (count == 0).')


# ============================================================================
# F.1 P2 -- HDF5 + Zarr lumenairy_version stamping
# ============================================================================

class TestF1HDF5VersionStamping:
    """v4.14.2 audit P2: HDF5/Zarr writes don't stamp
    ``lumenairy_version`` attr.  v4.15 adds the stamp at every
    ``create_dataset`` / ``create_array`` site so a downstream
    reader can identify which library version wrote each dataset.

    Each test exercises ONE write entry point and asserts the
    stamp landed.  Optional ``zarr`` is gated.
    """

    def _tmp_h5_path(self):
        f = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
        f.close()
        return f.name

    def test_save_field_h5_stamps_lumenairy_version(self):
        h5py = pytest.importorskip('h5py')
        E = np.ones((16, 16), dtype=np.complex128)
        path = self._tmp_h5_path()
        try:
            la.save_field_h5(path, E, dx=2e-6, wavelength=1.31e-6)
            with h5py.File(path, 'r') as f:
                assert 'lumenairy_version' in f['field'].attrs, (
                    'save_field_h5 did not stamp lumenairy_version')
                got = f['field'].attrs['lumenairy_version']
                if isinstance(got, bytes):
                    got = got.decode()
                assert got == la.__version__, (
                    f'lumenairy_version stamp mismatch: got {got!r}, '
                    f'expected {la.__version__!r}')
        finally:
            os.unlink(path)

    def test_save_planes_h5_stamps_lumenairy_version(self):
        h5py = pytest.importorskip('h5py')
        planes = [
            {'field': np.ones((8, 8), dtype=np.complex128), 'dx': 2e-6},
            {'field': 2 * np.ones((8, 8), dtype=np.complex128), 'dx': 2e-6},
        ]
        path = self._tmp_h5_path()
        try:
            la.save_planes_h5(path, planes, wavelength=1.31e-6)
            with h5py.File(path, 'r') as f:
                # Group-level stamp.
                assert 'lumenairy_version' in f['planes'].attrs
                # Per-plane stamps.
                for i in range(2):
                    name = f'plane_{i:02d}'
                    assert 'lumenairy_version' in f[f'planes/{name}'].attrs, (
                        f'save_planes_h5: per-plane stamp missing on '
                        f'plane {i}')
        finally:
            os.unlink(path)

    def test_append_plane_h5_stamps_lumenairy_version(self):
        h5py = pytest.importorskip('h5py')
        path = self._tmp_h5_path()
        try:
            E = np.ones((8, 8), dtype=np.complex128)
            la.append_plane_h5(path, E, dx=2e-6, label='first')
            la.append_plane_h5(path, 2 * E, dx=2e-6, label='second')
            with h5py.File(path, 'r') as f:
                # Group-level stamp (set on first-touch).
                assert 'lumenairy_version' in f['planes'].attrs
                for i in range(2):
                    name = f'plane_{i:02d}'
                    assert 'lumenairy_version' in f[f'planes/{name}'].attrs, (
                        f'append_plane_h5: per-plane stamp missing on '
                        f'plane {i}')
                    got = f[f'planes/{name}'].attrs['lumenairy_version']
                    if isinstance(got, bytes):
                        got = got.decode()
                    assert got == la.__version__
        finally:
            os.unlink(path)

    def test_save_jones_field_h5_stamps_lumenairy_version(self):
        h5py = pytest.importorskip('h5py')
        from lumenairy.elements.polarization import JonesField
        Ex = np.ones((8, 8), dtype=np.complex128)
        Ey = np.zeros((8, 8), dtype=np.complex128)
        jf = JonesField(Ex, Ey, dx=2e-6, dy=2e-6)
        path = self._tmp_h5_path()
        try:
            la.save_jones_field_h5(path, jf, wavelength=1.31e-6)
            with h5py.File(path, 'r') as f:
                # Group-level stamp.
                assert 'lumenairy_version' in f['jones'].attrs
                # Per-dataset stamps on the Ex / Ey arrays.
                assert 'lumenairy_version' in f['jones/Ex'].attrs
                assert 'lumenairy_version' in f['jones/Ey'].attrs
        finally:
            os.unlink(path)


@pytest.mark.skipif(
    pytest.importorskip('zarr', reason='zarr not installed') is None,
    reason='zarr not installed')
class TestF1ZarrVersionStamping:
    """Zarr counterpart of the HDF5 stamp pin."""

    def test_zarr_append_plane_stamps_lumenairy_version(self, tmp_path):
        zarr = pytest.importorskip('zarr')
        path = str(tmp_path / 'streaming.zarr')
        E = np.ones((8, 8), dtype=np.complex128)
        # Use the dispatch entry point so the zarr branch fires
        # uniformly with the default-backend mechanism.
        la.append_plane(path, E, dx=2e-6, label='first')
        store = zarr.open_group(path, mode='r')
        # Group-level stamp.
        assert 'lumenairy_version' in store['planes'].attrs
        # Per-plane stamp.
        assert 'lumenairy_version' in store['planes/plane_00'].attrs


# ============================================================================
# F.1 P2 -- create_multi_field_sources now in factory-validation list
# ============================================================================

class TestF1MultiFieldSourcesValidates:
    """v4.14.2 audit P2: ``create_multi_field_sources`` was not in
    the factory-validation list.  Pre-v4.15 it transitively
    validated through ``create_tilted_plane_wave``, leaking the
    internal callee name in user-facing error messages.

    v4.15 calls ``_validate_grid_params`` at the entry point.
    Errors now name ``create_multi_field_sources``.
    """

    def test_validates_at_entry_point_with_correct_fn_name(self):
        """``N=-1`` triggers the validator; the error message should
        name ``create_multi_field_sources``, not the inner callee."""
        with pytest.raises(ValueError) as exc_info:
            la.create_multi_field_sources(
                N=-1, dx=2e-6, wavelength=1.31e-6,
                field_angles=[0.0])
        msg = str(exc_info.value)
        assert 'create_multi_field_sources' in msg, (
            f'create_multi_field_sources error did not name itself; '
            f'got: {msg!r}')

    def test_invalid_wavelength_named_at_entry_point(self):
        with pytest.raises(ValueError) as exc_info:
            la.create_multi_field_sources(
                N=64, dx=2e-6, wavelength=0.0,
                field_angles=[0.0])
        assert 'create_multi_field_sources' in str(exc_info.value)


# ============================================================================
# F.1 P3 -- create_led_source error-message clarity
# ============================================================================

class TestF1LedSourceErrorMessage:
    """v4.14.2 audit P3: ``create_led_source`` legacy-shim "max 8
    (legacy) or 3 (canonical)" wording was confusing.  v4.15
    rewrites it to spell out both forms separately.
    """

    def test_too_many_positional_message_spells_both_forms(self):
        """The error message should describe both call forms
        without conflating their positional caps."""
        with pytest.raises(TypeError) as exc_info:
            la.create_led_source(
                64, 16e-6, 100e-6, 0.3, 1.31e-6,
                0.0, 0.0, np.complex128,
                'extra_positional_1', 'extra_positional_2')
        msg = str(exc_info.value)
        # Both forms must be mentioned.
        assert 'legacy' in msg.lower(), (
            f'create_led_source overflow message lost the "legacy" '
            f'reference: {msg!r}')
        assert 'canonical' in msg.lower(), (
            f'create_led_source overflow message lost the "canonical" '
            f'reference: {msg!r}')
        # The confusing "max 8 ... or 3" wording should be gone.
        assert 'max 8 (legacy) or 3' not in msg, (
            'create_led_source still carries the confusing pre-v4.15 '
            '"max 8 (legacy) or 3 (canonical)" wording.')


# ============================================================================
# F.4 -- README cookbook section exists + at least 3 examples parse
# ============================================================================

class TestF4ReadmeCookbook:
    """v4.14.2 audit P3: README cookbook still missing examples for
    the 6 v4.14.0 public functions (``encircled_energy``,
    ``mtf_cutoff``, ``beam_diameter``, ``depth_of_focus``,
    ``plot_wavefront``, and the 6th -- ``encircled_energy_curve``
    / ``encircled_energy_radius``).  v4.15 adds a ``## Cookbook``
    section with one runnable code block per function.
    """

    def test_cookbook_section_exists(self):
        text = _README_PATH.read_text(encoding='utf-8')
        assert '## Cookbook' in text or '## cookbook' in text.lower(), (
            'README is missing the ``## Cookbook`` section that '
            'v4.15.0 adds for the 6 v4.14.0 public functions.')

    def test_cookbook_mentions_all_six_v4_14_0_functions(self):
        text = _README_PATH.read_text(encoding='utf-8')
        # Find the cookbook section; check the names within it.
        m = re.search(
            r'## Cookbook(.*?)(?:\n## |\Z)',
            text, re.DOTALL | re.IGNORECASE)
        assert m, 'Could not locate ``## Cookbook`` section in README'
        section = m.group(1)
        for fn_name in (
                'encircled_energy_curve',
                'encircled_energy_radius',
                'mtf_cutoff',
                'beam_diameter',
                'depth_of_focus',
                'plot_wavefront'):
            assert fn_name in section, (
                f'README cookbook section does not mention '
                f'{fn_name!r}.')

    def test_at_least_three_cookbook_code_blocks_parse(self):
        """Extract Python code blocks from the cookbook section and
        check that at least 3 parse as syntactically valid Python.
        ``compile`` will catch syntax errors without executing the
        code (so we don't pay matplotlib / scipy import costs)."""
        text = _README_PATH.read_text(encoding='utf-8')
        m = re.search(
            r'## Cookbook(.*?)(?:\n## |\Z)',
            text, re.DOTALL | re.IGNORECASE)
        assert m, 'Could not locate ``## Cookbook`` section in README'
        section = m.group(1)
        # Grab every fenced ```python ... ``` block.
        blocks = re.findall(
            r'```python\n(.*?)\n```', section, re.DOTALL)
        assert len(blocks) >= 3, (
            f'Cookbook contains only {len(blocks)} ```python``` '
            f'code blocks; expected at least 3 syntactically '
            f'checked examples.')
        for i, code in enumerate(blocks[:6]):  # spot-check up to 6
            try:
                compile(code, f'<cookbook block {i}>', 'exec')
            except SyntaxError as exc:
                pytest.fail(
                    f'Cookbook code block #{i} does not parse as valid '
                    f'Python: {exc!r}\nBlock:\n{code}')


# ============================================================================
# F.5 -- CHANGELOG line-citation drift fix
# ============================================================================

class TestF5ChangelogLineCitations:
    """v4.14.2 audit P3: CHANGELOG cites stale line numbers for the
    ``ToleranceAwareMerit._evaluate_perturbed`` (claimed
    ``optimize/core.py:2750-2755``; actual ~2792) and the
    ``MatchIdealSystem._make_source`` (claimed ``:958-966``; actual
    ~977-991) sentinel-aware branches.

    v4.15 refreshes both citations.  Pin them so a future
    optimize/core.py edit that drifts the canonical sites again
    forces a CHANGELOG refresh in the same release.
    """

    def test_changelog_no_longer_cites_pre_v4_15_drifted_lines(self):
        text = _CHANGELOG_PATH.read_text(encoding='utf-8')
        # The pre-v4.15 citations the audit flagged.
        assert ':2750-2755' not in text, (
            'CHANGELOG still cites the pre-drift line range '
            '``optimize/core.py:2750-2755``; v4.15 should have '
            'refreshed this to the current line numbers.')
        # Older citations like :958-966 (drifted; actual is 977-991).
        # Allow ``958`` to appear in unrelated contexts so look for
        # the specific range form.
        assert ':958-966' not in text, (
            'CHANGELOG still cites the pre-drift range '
            '``optimize/core.py:958-966``; v4.15 should have '
            'refreshed this to the current line numbers.')

    def test_changelog_cites_actual_current_lines(self):
        """The post-v4.15 citation should land within +/- 5 lines of
        the actual current location of the canonical sites.

        ``ToleranceAwareMerit.evaluate``: target ~2792 (the
        ``if _cache['mask'] is _ZERO_APERTURE_MASK:`` line under
        the ``ToleranceAware`` class body).
        ``MatchIdealSystem._make_source``: target ~977 (the
        ``if ap is not None and np.isfinite(ap) and ap > 0:``
        line).
        """
        text = _CHANGELOG_PATH.read_text(encoding='utf-8')
        # v5.1.0 (Wave-4 integration / Agent E split):
        # ``ToleranceAwareMerit.evaluate`` moved out of the monolithic
        # ``optimize/core.py`` into ``optimize/wrapper_merits.py``.
        # ``MatchIdealSystem._make_source`` stayed in ``merit_terms.py``
        # (the wrapper-merits split's sibling submodule).  Walk both
        # plus the back-compat shell ``core.py`` so the pin keeps
        # finding the anchor strings regardless of which submodule
        # hosts them post-split.
        from lumenairy.optimize import core as opt_core
        from lumenairy.optimize import wrapper_merits as opt_wm
        from lumenairy.optimize import merit_terms as opt_mt
        actual_tolerance_line = None
        actual_match_ideal_line = None
        for _mod in (opt_wm, opt_mt, opt_core):
            opt_src = inspect.getsource(_mod)
            for i, line in enumerate(opt_src.splitlines(), start=1):
                stripped = line.lstrip()
                if (actual_tolerance_line is None
                        and stripped.startswith('if _cache[')
                        and '_ZERO_APERTURE_MASK' in line):
                    actual_tolerance_line = i
                if (actual_match_ideal_line is None
                        and 'np.isfinite(ap) and ap > 0' in line):
                    actual_match_ideal_line = i
            if (actual_tolerance_line is not None
                    and actual_match_ideal_line is not None):
                break
        # We don't pin to an exact line; verify a citation within
        # +/- 5 of either canonical location exists in CHANGELOG.
        for actual_line, fixture_label in (
                (actual_tolerance_line, 'ToleranceAwareMerit sentinel branch'),
                (actual_match_ideal_line, 'MatchIdealSystem._make_source ap>0 branch')):
            if actual_line is None:
                pytest.fail(
                    f'Failed to locate {fixture_label} in '
                    f'optimize/core.py via the test fixture\'s '
                    f'anchor strings.  Either the canonical sites '
                    f'moved or the fixture\'s search is broken.')
            nearby = {str(actual_line + d) for d in range(-5, 6)}
            found = any(
                f':{n}' in text or f':{n}-' in text or f'-{n}' in text
                for n in nearby)
            assert found, (
                f'CHANGELOG does not cite optimize/core.py near '
                f'line {actual_line} (the current location of the '
                f'{fixture_label}).  Refresh the citation in '
                f'CHANGELOG.md.')


# ============================================================================
# F.5 -- alongside the citation update, the canonical site itself
# must still point at the documented sentinel pattern
# ============================================================================

class TestF5OptimizeCoreSentinelSitesIntact:
    """Sanity pin: regardless of CHANGELOG drift, the canonical
    sentinel-aware branches in ``optimize/core.py`` must still
    exist and exercise the documented ``_ZERO_APERTURE_MASK``
    semantics.
    """

    def test_tolerance_aware_evaluate_perturbed_has_sentinel_branch(self):
        # NOTE: the audit refers to a ``_evaluate_perturbed`` helper but
        # the actual class folds the per-trial logic into
        # ``ToleranceAwareMerit.evaluate`` directly.  The sentinel
        # check is at line ~2792 within that body.
        from lumenairy.optimize.core import ToleranceAwareMerit
        src = inspect.getsource(ToleranceAwareMerit.evaluate)
        assert '_ZERO_APERTURE_MASK' in src, (
            'ToleranceAwareMerit.evaluate lost its '
            '``_ZERO_APERTURE_MASK`` sentinel check -- the v4.14.2 '
            'P1-NEW-1 fix has regressed.')

    def test_match_ideal_system_make_source_has_sentinel_logic(self):
        from lumenairy.optimize.core import MatchIdealSystemMerit
        src = inspect.getsource(MatchIdealSystemMerit._make_source)
        # The fix expresses the deliberate-zero branch via ``ap > 0``
        # finite-and-positive checks rather than the sentinel name
        # itself.  Pin the visible behaviour.
        assert 'ap > 0' in src or '_ZERO_APERTURE_MASK' in src, (
            'MatchIdealSystem._make_source lost the canonical '
            'deliberate-zero aperture branch -- the v4.14.2 '
            'P1-NEW-1 fix has regressed.')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
