"""v4.16.2 Agent D regression tests.

Closes audit P1-NEW-F2-HIGH-1, P1-NEW-F2-HIGH-2, P2-NEW-F2-MED-1,
P2-NEW-F2-MED-2, P2-NEW-V2-1, P2-NEW-F1-4.

Most of the heavy lifting lives in
`test_v4_16_2_dispatcher_pin_doc_consistency.py` (the 11th walker);
this file pins the specific corrected lines + the
Migration-Guide.md skeleton + the hardened 10th walker.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# audit closure: P1-NEW-F2-HIGH-1 -- README refractiveindex moved
# ============================================================================

def test_readme_does_not_cite_refractiveindex_as_required():
    """The README Dependencies block must not list refractiveindex
    in the `Required` section (it moved to [glass] extras in
    v4.16.1)."""
    text = (_REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    match = re.search(
        r'### Required\s*\n(.*?)(?=\n###|\n##\s|\Z)',
        text, re.DOTALL,
    )
    assert match is not None, "README missing `### Required` subsection"
    block = match.group(1)
    assert '`refractiveindex`' not in block, (
        "README.md still cites `refractiveindex` as Required.  "
        "v4.16.1 moved it to [glass] extras.")


def test_readme_pip_install_command_uses_extras_or_omits_refractiveindex():
    """The README quick-install must not force refractiveindex via
    a bare `pip install ... refractiveindex` command."""
    text = (_REPO_ROOT / 'README.md').read_text(encoding='utf-8')
    # Find any pip install command in fenced bash blocks.
    bad = []
    for m in re.finditer(r'pip install ([^\n`]+)', text):
        cmd = m.group(1)
        # If `refractiveindex` appears in the command without `[glass]`
        # or other bracket-extras form, that's a force-install drift.
        if 'refractiveindex' in cmd and 'lumenairy[' not in cmd:
            bad.append(cmd.strip())
    assert not bad, (
        f"README quick-install command(s) force refractiveindex: {bad}.  "
        f"Move to `pip install lumenairy[glass]`.")


# ============================================================================
# audit closure: P1-NEW-F2-HIGH-2 -- requirements.txt sync
# ============================================================================

def test_requirements_txt_drops_refractiveindex():
    text = (_REPO_ROOT / 'requirements.txt').read_text(encoding='utf-8')
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if '#' in stripped:
            stripped = stripped[:stripped.index('#')].strip()
        assert 'refractiveindex' not in stripped.lower(), (
            f"requirements.txt has an uncommented refractiveindex line: "
            f"{line!r}.  v4.16.1 moved it to [glass] extras; the line "
            f"must be commented or removed.")


def test_requirements_txt_zarr_floor_is_3_or_higher():
    text = (_REPO_ROOT / 'requirements.txt').read_text(encoding='utf-8')
    # Find the zarr line (commented or not) and check the floor.
    for m in re.finditer(r'zarr>=([0-9]+)\.([0-9]+)', text):
        major, minor = int(m.group(1)), int(m.group(2))
        assert major >= 3, (
            f"requirements.txt cites zarr>={major}.{minor}; "
            f"v4.16.1 bumped to >=3.0 to match storage.py's "
            f"Group.create_array (Zarr v3) usage.")


# ============================================================================
# audit closure: P2-NEW-F2-MED-1 -- ROADMAP V9 -> V10 (+V11)
# ============================================================================

def test_roadmap_claims_correct_meta_pin_count():
    text = (_REPO_ROOT / 'ROADMAP.md').read_text(encoding='utf-8')
    m = re.search(r'ALL (\d+) dispatcher meta-pins', text)
    assert m is not None, (
        "ROADMAP.md missing `ALL N dispatcher meta-pins` claim.")
    n_claimed = int(m.group(1))
    # v4.16.2 ships V11.  Anything less is stale.
    assert n_claimed >= 11, (
        f"ROADMAP.md claims `ALL {n_claimed} dispatcher meta-pins` "
        f"but v4.16.2 ships V11 (doc-consistency walker).  Update "
        f"the count.")


def test_roadmap_enumerates_v10_v11():
    text = (_REPO_ROOT / 'ROADMAP.md').read_text(encoding='utf-8')
    for v in ('V10', 'V11'):
        assert v in text, (
            f"ROADMAP.md does not enumerate {v}.  v4.16.2 ships "
            f"both (V10 was the v4.16.1 walker that already shipped; "
            f"v4.16.2 adds V11).")


# ============================================================================
# audit closure: P2-NEW-V2-1 -- CHANGELOG UserWarning -> RuntimeWarning
# ============================================================================

def test_changelog_high_na_transfer_jax_uses_runtimewarning():
    text = (_REPO_ROOT / 'CHANGELOG.md').read_text(encoding='utf-8')
    # Find the high-NA _transfer_jax block in the v4.16.1 entry.
    m = re.search(
        r'High-NA `_transfer_jax`.*?(?=\n\* |\n###)',
        text, re.DOTALL,
    )
    assert m is not None, (
        "CHANGELOG missing the High-NA _transfer_jax block in v4.16.1.")
    block = m.group(0)
    assert 'RuntimeWarning' in block, (
        "CHANGELOG High-NA _transfer_jax block does not mention "
        "RuntimeWarning; the implementation uses RuntimeWarning, "
        "not UserWarning.  Fix the typo.")


# ============================================================================
# Migration-Guide.md exists
# ============================================================================

def test_migration_guide_md_exists_at_repo_root():
    guide = _REPO_ROOT / 'Migration-Guide.md'
    assert guide.is_file(), (
        f"{guide} does not exist.  v4.16.2 pre-v5.0 prep should ship "
        f"the Migration-Guide.md skeleton.")


def test_migration_guide_has_v4_16_2_section():
    guide = _REPO_ROOT / 'Migration-Guide.md'
    text = guide.read_text(encoding='utf-8')
    assert '## 4.16.2 -- Default-config knobs' in text, (
        "Migration-Guide.md missing the v4.16.2 section.")


# ============================================================================
# audit closure: P2-NEW-F1-4 -- 10th walker module-level call requirement
# ============================================================================

def test_10th_walker_rejects_function_nested_register_cache_clearer():
    """Synthetic counter-pin: a module containing
    ``register_cache_clearer`` ONLY inside a function definition (not
    at module level) must NOT be accepted as enrolled by the
    hardened walker."""
    from tests.unit.test_v4_16_1_dispatcher_pin_cache_registry_enrollment import (
        _module_has_register_cache_clearer_call,
    )

    src = """
import functools

@functools.lru_cache(maxsize=32)
def _CACHE_FN(x):
    return x

def _setup_late():
    # Pre-v4.16.2 walker would accept this; v4.16.2 walker rejects it.
    register_cache_clearer('cache_fn', _CACHE_FN.cache_clear)
"""
    tree = ast.parse(src)
    assert _module_has_register_cache_clearer_call(tree) is False, (
        "Hardened 10th walker should REJECT a module where the "
        "register_cache_clearer call lives inside a function def.")


def test_10th_walker_rejects_always_false_branch_register_cache_clearer():
    """Synthetic counter-pin: ``if False: register_cache_clearer(...)``
    is unreachable and must not be accepted."""
    from tests.unit.test_v4_16_1_dispatcher_pin_cache_registry_enrollment import (
        _module_has_register_cache_clearer_call,
    )

    src = """
import functools

@functools.lru_cache(maxsize=32)
def _CACHE_FN(x):
    return x

if False:
    register_cache_clearer('cache_fn', _CACHE_FN.cache_clear)
"""
    tree = ast.parse(src)
    assert _module_has_register_cache_clearer_call(tree) is False, (
        "Hardened 10th walker should REJECT an always-False branch.")


def test_10th_walker_accepts_module_level_call():
    """Positive pin: module-level call is accepted."""
    from tests.unit.test_v4_16_1_dispatcher_pin_cache_registry_enrollment import (
        _module_has_register_cache_clearer_call,
    )

    src = """
import functools

@functools.lru_cache(maxsize=32)
def _CACHE_FN(x):
    return x

register_cache_clearer('cache_fn', _CACHE_FN.cache_clear)
"""
    tree = ast.parse(src)
    assert _module_has_register_cache_clearer_call(tree) is True


def test_10th_walker_accepts_top_level_try_except_call():
    """Positive pin: canonical try/except ImportError pattern is
    accepted."""
    from tests.unit.test_v4_16_1_dispatcher_pin_cache_registry_enrollment import (
        _module_has_register_cache_clearer_call,
    )

    src = """
import functools

@functools.lru_cache(maxsize=32)
def _CACHE_FN(x):
    return x

try:
    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _register_cache_clearer('cache_fn', _CACHE_FN.cache_clear)
except ImportError:
    pass
"""
    tree = ast.parse(src)
    assert _module_has_register_cache_clearer_call(tree) is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
