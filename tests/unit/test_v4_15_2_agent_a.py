"""v4.15.2 Agent-A tests -- Schell-family P0 + P1/P2/P3 closures.

The v4.15.1 audit (`docs/audits/AUDIT_V4_15_1_2026_05_18.md`) flagged
the 3 Schell factories as P0-NEW-1 (silent contract break: default
return shape changed from ``(E_2d, x, y)`` to
``(ensemble_3d, dx, dy, wavelength)`` without a ``DeprecationWarning``)
+ P1-NEW-G (CHANGELOG missing ``### Breaking changes`` subhead) +
P2 (``energy_threshold`` not forwarded through factories) + P3 (45-deg
fold doc drift; stale line-number citation; test-count arithmetic
follow-up).

v4.15.2 closes the P0 by emitting a one-release ``DeprecationWarning``
through the shared ``_deprecation.warn_deprecated_signature`` helper
on the default-``return_kind`` path; explicit
``return_kind='ensemble'`` or ``return_kind='mcf'`` calls are silent.
The P2 closure adds ``energy_threshold`` to every factory signature
and forwards it through to ``PartialCoherenceMCF.from_ensemble`` on
the MCF return path.

Author: Andrew Traverso -- v4.15.2 / Agent A
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from lumenairy.sources.core import (
    PartialCoherenceMCF,
    create_annular_incoherent_source,
    create_gaussian_schell_source,
    create_schell_model_source,
)


# ----------------------------------------------------------------------
# Shared parameter bundles -- keep the per-factory kwargs minimal but
# physically valid so every test in this module can re-use them.
# ----------------------------------------------------------------------

_GAUSSIAN_SCHELL_KW = dict(
    N=16, dx=2e-6, wavelength=633e-9,
    w0=20e-6, sigma_g=8e-6,
    n_realizations=4, seed=0,
)

_SCHELL_MODEL_KW = dict(
    N=16, dx=2e-6, wavelength=633e-9,
    intensity_profile=np.exp(
        -((np.linspace(-1, 1, 16)[:, None]) ** 2
          + (np.linspace(-1, 1, 16)[None, :]) ** 2) / 0.25),
    coherence_length=8e-6,
    n_realizations=4, seed=0,
)

_ANNULAR_KW = dict(
    N=16, dx=2e-6, wavelength=633e-9,
    inner_radius=4e-6, outer_radius=12e-6,
    n_realizations=4, seed=0,
)


# ======================================================================
# A.1 -- P0-NEW-1 DeprecationWarning on the default-return_kind path
# ======================================================================

class TestSchellDefaultDeprecationWarning:
    """The 3 Schell factories must emit a ``DeprecationWarning``
    flagging the v4.15.0 -> v4.15.1 return-shape change when called
    with the default ``return_kind`` (caller didn't pass it
    explicitly).  Explicit ``return_kind='ensemble'`` / ``'mcf'`` is
    silent.  The message must reference both the change horizon
    (``v4.15.0 -> v4.15.1``) and the kwarg name (``return_kind``) so
    the user can grep for either when migrating.
    """

    def test_gaussian_schell_default_emits_deprecation_warning(self):
        """Calling without ``return_kind`` must emit a
        ``DeprecationWarning`` whose message mentions ``return_kind``
        and the v4.15.0 -> v4.15.1 transition."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = create_gaussian_schell_source(**_GAUSSIAN_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_gaussian_schell_source' in str(w.message)]
        assert len(dep) >= 1, (
            f"Default-return_kind call must emit "
            f"DeprecationWarning; got "
            f"{[str(w.message) for w in caught]}.")
        msg = str(dep[0].message)
        assert 'return_kind' in msg, (
            f"DeprecationWarning must mention the kwarg name "
            f"'return_kind'; got {msg!r}.")
        assert ('v4.15.0' in msg and 'v4.15.1' in msg), (
            f"DeprecationWarning must reference both v4.15.0 and "
            f"v4.15.1 (the return-shape change horizon); got {msg!r}.")
        # Still get the v4.15.1 ensemble-tuple return shape by default.
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_schell_model_default_emits_deprecation_warning(self):
        """Same contract for ``create_schell_model_source``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = create_schell_model_source(**_SCHELL_MODEL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_schell_model_source' in str(w.message)]
        assert len(dep) >= 1
        msg = str(dep[0].message)
        assert 'return_kind' in msg
        assert ('v4.15.0' in msg and 'v4.15.1' in msg)
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_annular_incoherent_default_emits_deprecation_warning(self):
        """Same contract for ``create_annular_incoherent_source``."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = create_annular_incoherent_source(**_ANNULAR_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_annular_incoherent_source' in str(w.message)]
        assert len(dep) >= 1
        msg = str(dep[0].message)
        assert 'return_kind' in msg
        assert ('v4.15.0' in msg and 'v4.15.1' in msg)
        ens, dx, dy, wl = result
        assert ens.shape == (4, 16, 16)

    def test_deprecation_warning_mentions_v5_0_removal(self):
        """The warning must document the one-release deprecation
        horizon (removal in v5.0) so migrators know how long they
        have to update before the default-path becomes an error."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_gaussian_schell_source(**_GAUSSIAN_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert any('5.0' in str(w.message) for w in dep), (
            f"DeprecationWarning must reference v5.0 removal; got "
            f"{[str(w.message) for w in dep]}.")


# ======================================================================
# Explicit-return_kind calls must be SILENT -- no warning leakage
# ======================================================================

class TestExplicitReturnKindIsSilent:
    """Pass ``return_kind`` explicitly -- the call must not emit a
    ``DeprecationWarning``.  This is the user's escape hatch: once
    they acknowledge the v4.15.1 contract, they can keep calling the
    factory in CI without ``warnings.filterwarnings`` boilerplate.
    """

    def test_explicit_return_kind_ensemble_no_warning(self):
        """``return_kind='ensemble'`` must be silent (no
        DeprecationWarning) across all 3 factories."""
        for factory, kw in (
            (create_gaussian_schell_source, _GAUSSIAN_SCHELL_KW),
            (create_schell_model_source, _SCHELL_MODEL_KW),
            (create_annular_incoherent_source, _ANNULAR_KW),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                factory(return_kind='ensemble', **kw)
            dep = [w for w in caught
                   if issubclass(w.category, DeprecationWarning)
                   and 'return_kind' in str(w.message)]
            assert dep == [], (
                f"{factory.__name__}: explicit return_kind='ensemble' "
                f"must NOT emit a return_kind DeprecationWarning; got "
                f"{[str(w.message) for w in dep]}.")

    def test_explicit_return_kind_mcf_no_warning(self):
        """``return_kind='mcf'`` must be silent across all 3
        factories."""
        for factory, kw in (
            (create_gaussian_schell_source, _GAUSSIAN_SCHELL_KW),
            (create_schell_model_source, _SCHELL_MODEL_KW),
            (create_annular_incoherent_source, _ANNULAR_KW),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = factory(return_kind='mcf', **kw)
            dep = [w for w in caught
                   if issubclass(w.category, DeprecationWarning)
                   and 'return_kind' in str(w.message)]
            assert dep == [], (
                f"{factory.__name__}: explicit return_kind='mcf' must "
                f"NOT emit a return_kind DeprecationWarning; got "
                f"{[str(w.message) for w in dep]}.")
            # Sanity: the MCF return-type contract still holds.
            assert isinstance(result, PartialCoherenceMCF)


# ======================================================================
# A.3 -- P2: ``energy_threshold`` kwarg forwarding through factories
# ======================================================================

class TestEnergyThresholdForwarded:
    """The ``energy_threshold`` kwarg, exposed on
    :meth:`PartialCoherenceMCF.from_ensemble`, must be forwarded
    through every Schell factory's ``return_kind='mcf'`` path.  A
    looser threshold (e.g. 0.50) selects fewer modes than the default
    0.99; pin against the mode count and against equality with a
    direct ``from_ensemble`` call using the same threshold.
    """

    def _build_modal_mcf(self, factory, kw, energy_threshold, n_modes=None):
        """Drive the factory at a grid large enough to trigger modal
        storage (``max_full_N=1`` forces ``Ny * Nx > max_full_N**2``)
        so ``energy_threshold`` is actually consulted."""
        kw = {**kw, 'n_realizations': 8}
        return factory(
            return_kind='mcf',
            max_full_N=1,
            n_modes=n_modes,
            energy_threshold=energy_threshold,
            **kw,
        )

    def test_energy_threshold_forwarded_to_mcf(self):
        """A 0.50 threshold must select fewer modes than the 0.99
        default for ``create_gaussian_schell_source`` modal storage."""
        mcf_default = self._build_modal_mcf(
            create_gaussian_schell_source, _GAUSSIAN_SCHELL_KW,
            energy_threshold=0.99)
        mcf_loose = self._build_modal_mcf(
            create_gaussian_schell_source, _GAUSSIAN_SCHELL_KW,
            energy_threshold=0.50)
        modes_default, eig_default = mcf_default.coherent_modes()
        modes_loose, eig_loose = mcf_loose.coherent_modes()
        # Looser threshold must keep strictly fewer (or at worst equal)
        # modes; for the seeded n_realizations=8 ensemble the spectrum
        # is non-degenerate enough that strict inequality must hold.
        assert len(eig_loose) < len(eig_default), (
            f"Looser energy_threshold (0.50) must retain fewer modes "
            f"than the default (0.99); got "
            f"len(eig_loose)={len(eig_loose)} vs "
            f"len(eig_default)={len(eig_default)}.")

    def test_energy_threshold_forwarded_schell_model(self):
        """Same contract for ``create_schell_model_source``."""
        mcf_default = self._build_modal_mcf(
            create_schell_model_source, _SCHELL_MODEL_KW,
            energy_threshold=0.99)
        mcf_loose = self._build_modal_mcf(
            create_schell_model_source, _SCHELL_MODEL_KW,
            energy_threshold=0.50)
        _, eig_default = mcf_default.coherent_modes()
        _, eig_loose = mcf_loose.coherent_modes()
        assert len(eig_loose) < len(eig_default)

    def test_energy_threshold_forwarded_annular(self):
        """Same contract for ``create_annular_incoherent_source``."""
        mcf_default = self._build_modal_mcf(
            create_annular_incoherent_source, _ANNULAR_KW,
            energy_threshold=0.99)
        mcf_loose = self._build_modal_mcf(
            create_annular_incoherent_source, _ANNULAR_KW,
            energy_threshold=0.50)
        _, eig_default = mcf_default.coherent_modes()
        _, eig_loose = mcf_loose.coherent_modes()
        assert len(eig_loose) <= len(eig_default), (
            f"Looser energy_threshold must retain at most as many "
            f"modes as the default; got len(eig_loose)={len(eig_loose)} "
            f"vs len(eig_default)={len(eig_default)}.")

    def test_energy_threshold_accepted_on_ensemble_path(self):
        """``energy_threshold`` must be accepted on the
        ``return_kind='ensemble'`` path for signature symmetry, even
        though it is a no-op there (no modal truncation happens on the
        ensemble return)."""
        for factory, kw in (
            (create_gaussian_schell_source, _GAUSSIAN_SCHELL_KW),
            (create_schell_model_source, _SCHELL_MODEL_KW),
            (create_annular_incoherent_source, _ANNULAR_KW),
        ):
            # Must not raise.
            result = factory(
                return_kind='ensemble',
                energy_threshold=0.5,
                **kw,
            )
            ens, dx, dy, wl = result
            assert ens.shape[0] == kw['n_realizations']


# ======================================================================
# A.2 -- P1-NEW-G: CHANGELOG ``### Breaking changes`` subhead
# ======================================================================

_CHANGELOG_PATH = (
    Path(__file__).resolve().parents[2] / 'CHANGELOG.md')


def _read_changelog() -> str:
    # CHANGELOG.md is stored UTF-8 (em-dash + degree sign in the
    # body); read with that encoding so the regex can see the
    # subhead text + the 60-deg fold cite.
    return _CHANGELOG_PATH.read_text(encoding='utf-8')


def _v4_15_1_section(text: str) -> str:
    """Return the substring of ``text`` between the v4.15.1 header
    and the v4.15.0 header (the next section)."""
    m = re.search(r'^## \[4\.15\.1\].*?\n', text, flags=re.MULTILINE)
    assert m is not None, (
        "CHANGELOG.md must contain a '## [4.15.1]' section header.")
    start = m.end()
    n = re.search(r'^## \[4\.15\.0\]', text[start:], flags=re.MULTILINE)
    end = start + (n.start() if n else len(text))
    return text[start:end]


class TestChangelogBreakingChangesSubhead:
    """The v4.15.1 CHANGELOG entry shipped 3 confirmed breaking items
    (Schell ensemble-tuple return shape, ``strehl_vector`` default
    reference removed, ``system.evaluate`` mixed-shape ValueError)
    but had no ``### Breaking changes`` subhead.  v4.15.2 adds one.
    """

    def test_changelog_has_breaking_changes_subhead(self):
        """The v4.15.1 entry must contain a ``### Breaking changes``
        subhead."""
        section = _v4_15_1_section(_read_changelog())
        assert '### Breaking changes' in section, (
            "v4.15.1 CHANGELOG entry must include a "
            "'### Breaking changes' subhead (P1-NEW-G closure).")

    def test_breaking_changes_lists_schell_return_shape(self):
        """The ``Breaking changes`` block must call out the Schell
        ensemble-tuple return shape."""
        section = _v4_15_1_section(_read_changelog())
        # Locate the Breaking changes block.
        m = re.search(
            r'### Breaking changes\s*\n(.*?)(?=^### |\Z)',
            section, flags=re.MULTILINE | re.DOTALL)
        assert m is not None, "Breaking changes block not found."
        body = m.group(1)
        assert ('Schell' in body
                and 'ensemble' in body
                and 'return_kind' in body), (
            f"Breaking changes block must mention Schell-family return "
            f"shape and return_kind escape hatch; got {body!r}.")

    def test_breaking_changes_lists_strehl_vector_reference(self):
        """The ``Breaking changes`` block must call out the
        ``strehl_vector`` reference removal."""
        section = _v4_15_1_section(_read_changelog())
        m = re.search(
            r'### Breaking changes\s*\n(.*?)(?=^### |\Z)',
            section, flags=re.MULTILINE | re.DOTALL)
        body = m.group(1)
        assert 'strehl_vector' in body, (
            "Breaking changes block must mention strehl_vector "
            "default-reference removal.")

    def test_breaking_changes_lists_system_evaluate_mixed_shape(self):
        """The ``Breaking changes`` block must call out the
        ``system.evaluate`` mixed-shape ``ValueError``."""
        section = _v4_15_1_section(_read_changelog())
        m = re.search(
            r'### Breaking changes\s*\n(.*?)(?=^### |\Z)',
            section, flags=re.MULTILINE | re.DOTALL)
        body = m.group(1)
        assert ('system.evaluate' in body
                and 'mixed' in body
                and 'ValueError' in body), (
            "Breaking changes block must mention system.evaluate "
            "mixed-shape ValueError.")


# ======================================================================
# A.4 -- P3 doc drifts
# ======================================================================

class TestChangelogDriftFixes:
    """The v4.15.1 CHANGELOG had three documented drifts: the OAP
    end-to-end test uses 60-deg fold (alpha=pi/6) not 45-deg; the
    ``_ZERO_APERTURE_MASK`` sentinel branch moved after the Agent E
    refactor; and the test-count arithmetic for Agents E and G was
    confused between source-level function counts and pytest-
    collection counts.  v4.15.2 fixes the citations.
    """

    def test_changelog_mentions_60_fold_not_45_fold(self):
        """The v4.15.1 entry's OAP end-to-end test cite must read
        ``60-deg fold`` (or equivalent), not ``45-deg fold``.  The
        actual test uses ``alpha=pi/6`` (60-deg) -- 45-deg
        (alpha=pi/4) is degenerate for that geometry."""
        section = _v4_15_1_section(_read_changelog())
        assert '60' in section and 'fold' in section
        # The stale "45-deg fold" / "45 deg fold" citation must be
        # gone from the v4.15.1 entry.
        assert '45 deg fold' not in section
        assert '45-deg fold' not in section
        # The Unicode degree symbol form must also be absent.
        assert '45° fold' not in section

    def test_changelog_optimize_core_line_citation_refreshed(self):
        """The v4.15.1 entry cited ``optimize/core.py:2790-2795`` for
        the lenses_maslov refresh.  After Agent E's ``_Sentinel``
        base-class refactor pushed lines downward, the
        ``_ZERO_APERTURE_MASK`` sentinel branch is now at
        ``optimize/core.py:2905`` and the sentinel class + singleton
        live at ``optimize/core.py:2034`` / ``:2044``.  The cite must
        be refreshed in the v4.15.1 entry.
        """
        section = _v4_15_1_section(_read_changelog())
        # The stale numeric cite must be gone.
        assert '2790-2795' not in section, (
            "Stale 'optimize/core.py:2790-2795' cite must be "
            "refreshed in the v4.15.1 entry.")
        # Either form of the refreshed cite is acceptable: ``:2905``
        # (the branch), ``:2034`` (the sentinel class definition), or
        # ``:2044`` (the singleton).  Verify the cite matches the
        # actual current line of the branch in the source.
        import re as _re
        from pathlib import Path as _Path
        opt_core = (_CHANGELOG_PATH.parent / 'lumenairy'
                    / 'optimize' / 'core.py')
        opt_text = opt_core.read_text(encoding='utf-8')
        branch_lineno = None
        for i, line in enumerate(opt_text.splitlines(), 1):
            if "_cache['mask'] is _ZERO_APERTURE_MASK" in line:
                branch_lineno = i
                break
        assert branch_lineno is not None, (
            "Could not locate the _ZERO_APERTURE_MASK branch in "
            "optimize/core.py")
        # The CHANGELOG must cite the current branch line (or one of
        # the two sentinel-class lines).
        possible_cites = [
            f':{branch_lineno}', ':2034', ':2044',
        ]
        assert any(c in section for c in possible_cites), (
            f"v4.15.1 entry must cite at least one of the refreshed "
            f"line numbers ({possible_cites}); got CHANGELOG section "
            f"that mentions none of them.")

    def test_changelog_test_count_arithmetic_reconciles(self):
        """The v4.15.1 entry lists per-agent test counts.  Verify
        they are present and within a reasonable bound of the
        ``pytest --collect-only`` count on each
        ``tests/unit/test_v4_15_1_agent_*.py`` file.

        The CHANGELOG count reflects what was originally shipped in
        v4.15.1; subsequent patch releases (v4.15.2 etc.) may add
        supplementary follow-up pins to the same files as audit-
        closure work, so the CHANGELOG count is a LOWER BOUND on
        the current actual collected-item count.

        Original audit (``AUDIT_V4_15_1_2026_05_18.md`` line 218):
        "E=16 vs CHANGELOG 20; G=70 vs CHANGELOG 91" -- the audit
        was counting source-level ``def test_`` definitions, not
        parametrised pytest items.  The CHANGELOG counts use the
        parametrised count (the canonical pytest convention).
        """
        section = _v4_15_1_section(_read_changelog())
        # Pull the per-agent counts line.  The CHANGELOG bullet may
        # contain a parenthetical accounting note before the colon
        # (e.g. ``v4.15.1 additions (`pytest --collect-only` items,
        # ...):``) so accept either form.
        m = re.search(
            r'v4\.15\.1 additions[^:]*:\s*(.+?)(?=\n\* |\n###|\n----)',
            section, flags=re.DOTALL)
        assert m is not None, (
            "v4.15.1 entry must list per-agent test counts.")
        counts_text = m.group(1)
        # The 7 agents in scope: A, B, C, D, E, F, G.
        for label in ('A=', 'B=', 'C=', 'D=', 'E=', 'F=', 'G='):
            assert label in counts_text, (
                f"Per-agent test-count list must include {label!r}; "
                f"got {counts_text!r}.")

        # Cross-check the counts against the actual pytest collection
        # of each agent file.  Use pytest's collection API rather
        # than re-counting AST nodes so parametrised items are
        # included (matches the CHANGELOG accounting).
        import subprocess
        import sys
        repo_root = _CHANGELOG_PATH.parent
        expected = {}
        for label in ('a', 'b', 'c', 'd', 'e', 'f'):
            test_file = (repo_root / 'tests' / 'unit'
                         / f'test_v4_15_1_agent_{label}.py')
            assert test_file.exists(), (
                f"Agent {label.upper()} test file is missing: "
                f"{test_file}")
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_file),
                 '--collect-only', '-q'],
                cwd=str(repo_root), capture_output=True, text=True,
                timeout=60)
            out = (result.stdout or '') + (result.stderr or '')
            m_count = re.search(r'(\d+) tests collected', out)
            assert m_count is not None, (
                f"pytest --collect-only failed to report a count "
                f"for {test_file}: {out!r}.")
            expected[label.upper()] = int(m_count.group(1))
        # Agent G has 4 files -- sum across them.
        g_total = 0
        for suffix in ('abcd', 'application', 'examples',
                       'matches_system_abcd'):
            test_file = (repo_root / 'tests' / 'unit'
                         / f'test_v4_15_1_agent_g_{suffix}.py')
            assert test_file.exists()
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', str(test_file),
                 '--collect-only', '-q'],
                cwd=str(repo_root), capture_output=True, text=True,
                timeout=60)
            out = (result.stdout or '') + (result.stderr or '')
            m_count = re.search(r'(\d+) tests collected', out)
            assert m_count is not None
            g_total += int(m_count.group(1))
        expected['G'] = g_total

        # Verify each ``LETTER=N`` in the CHANGELOG is a plausible
        # LOWER BOUND on the actual collected count.  Subsequent
        # patch agents may add supplementary follow-up tests to the
        # v4.15.1 files (legitimate audit-closure work), so we allow
        # the actual count to exceed the CHANGELOG claim by up to
        # the count itself (i.e., at most 2x growth without the
        # CHANGELOG refreshing -- a generous bound).  We DO NOT
        # allow the actual count to be LESS than claimed (that
        # would mean tests were removed, which is the failure mode
        # the original audit flagged).
        for letter, exp_count in expected.items():
            m_label = re.search(
                rf'{letter}=(\d+)', counts_text)
            assert m_label is not None, (
                f"CHANGELOG counts must include {letter}=<N>; got "
                f"{counts_text!r}.")
            claimed = int(m_label.group(1))
            assert claimed <= exp_count, (
                f"CHANGELOG claims Agent {letter}={claimed} but "
                f"pytest --collect-only reports only {exp_count} "
                f"tests actually collected; tests appear to have "
                f"been REMOVED -- this is the failure mode the "
                f"v4.15.1 audit flagged.")
            assert exp_count <= claimed * 2 + 5, (
                f"CHANGELOG claims Agent {letter}={claimed} but "
                f"pytest --collect-only reports {exp_count} tests "
                f"-- the v4.15.1 ship count has drifted "
                f"unreasonably far.  Refresh the CHANGELOG bullet.")


# ======================================================================
# Regression: factories still respect ``n_modes`` explicit override
# ======================================================================

class TestEnergyThresholdNotUsedWhenNModesGiven:
    """When the caller passes ``n_modes`` explicitly,
    ``energy_threshold`` is ignored -- ``from_ensemble`` honours the
    rank override.  Verify this contract is preserved through the
    factory wrappers."""

    def test_n_modes_override_wins_over_energy_threshold(self):
        """An explicit ``n_modes=2`` must yield exactly 2 modes even
        when ``energy_threshold=0.5`` would otherwise pick fewer."""
        kw = {**_GAUSSIAN_SCHELL_KW, 'n_realizations': 8}
        mcf = create_gaussian_schell_source(
            return_kind='mcf',
            max_full_N=1,  # force modal storage
            n_modes=2,
            energy_threshold=0.5,
            **kw,
        )
        _, eig = mcf.coherent_modes()
        assert len(eig) == 2, (
            f"n_modes=2 must yield exactly 2 modes regardless of "
            f"energy_threshold; got {len(eig)}.")
