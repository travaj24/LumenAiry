"""Regression tests for the v4.16.1 Agent-D audit-fix scope.

Scope (per `AUDIT_V4_16_0_DEEP_2026_05_19.md` Part 8 items 16-23):

  * D.1 -- 4 stale `n_d` inline comments in `lumenairy/glass.py`
    aligned with the computed Sellmeier values for H-ZK9B, H-ZF12,
    D-LAK52, H-ZLAF52A (CDGM block).  The coefficients themselves
    are bit-for-bit correct against the refractiveindex.info CDGM
    2022-06 YAML; the inline comments lied.  Multi-way convergent
    finding (V5 + DEEP-3 + F1 + PHYS-2).
  * D.2 -- `refractiveindex>=1.0` moved from hard `[dependencies]`
    to `[project.optional-dependencies.glass]` and added to `all`;
    matches the lazy-import + Sellmeier-fallback handling in
    `glass.py`.
  * D.3 -- `zarr>=2.14` floor bumped to `zarr>=3.0` in the `zarr`
    and `all` extras groups; the library uses `Group.create_array`
    (Zarr v3-only API).
  * D.4 -- `_lens_traced.py` worker pool spawns workers via the
    explicit `multiprocessing.get_context('spawn')` to avoid `fork`
    on Linux (unsafe with FFT plan caches / threading).
  * D.5 -- `C:tmpoptimize_diff.txt` echo-redirect artifact at the
    repo root removed.
  * D.6 -- 3 example PNG outputs moved from the repo root to
    `examples/output/`; the 3 producing scripts now write there
    rather than the current working directory.
  * D.7 -- new `examples/07_zemax_load_trace.py` example: loads a
    .zmx file from the validation fixture set and traces a ray fan
    (closes audit P1-3 / UX item 22).
  * D.8 -- new top-level `CONVENTIONS.md` documenting the
    `create_*` (-> field/Source) vs `make_*` (-> prescription /
    bundle / non-field) factory verb contract + neighbouring
    conventions.

Author: Andrew Traverso -- v4.16.1 / Agent D.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_GLASS_PATH = _REPO_ROOT / 'lumenairy' / 'glass.py'
_PYPROJECT = _REPO_ROOT / 'pyproject.toml'
_LENS_TRACED = _REPO_ROOT / 'lumenairy' / 'elements' / '_lens_traced.py'
_EXAMPLES_DIR = _REPO_ROOT / 'examples'
_CONVENTIONS = _REPO_ROOT / 'CONVENTIONS.md'


# ============================================================================
# D.1 -- glass.py inline n_d comments now match computed Sellmeier values
# ============================================================================

# The 4-way convergent finding from the deep audit: pre-v4.16.1 these
# comments lied.  Now they should agree with the computed n_d to 5
# decimal places.  Format: glass_name -> (expected_n_d_in_comment,
# tolerance).
_FIXED_INLINE_ND = {
    'H-ZK9B':    (1.62041, 5e-5),
    'H-ZF12':    (1.76182, 5e-5),
    'D-LAK52':   (1.73050, 5e-5),
    'H-ZLAF52A': (1.80610, 5e-5),
}


def _read_glass_source() -> str:
    return _GLASS_PATH.read_text(encoding='utf-8')


@pytest.mark.parametrize('name', sorted(_FIXED_INLINE_ND))
def test_glass_inline_comment_matches_computed_nd(name):
    """For each of the 4 audit-flagged CDGM glasses, the inline
    ``n_d = ...`` comment in glass.py matches the actually-computed
    value at the d-line (587.56 nm) to 5e-5.
    """
    from lumenairy.glass import get_glass_index

    expected, tol = _FIXED_INLINE_ND[name]
    # Sanity: the actual computation still agrees with the comment.
    actual = get_glass_index(name, 587.56e-9)
    assert abs(actual - expected) < tol, (
        f"glass {name!r}: comment cites n_d={expected:.5f}; "
        f"computed n_d={actual:.6f}; diff={abs(actual-expected):.2e}")

    # Walk the source and verify the inline ``n_d = <value>`` comment
    # in the entry block matches ``expected``.
    src = _read_glass_source()
    # Find the line containing ``'{name}':`` (sellmeier table entry).
    entry_re = re.compile(r"^\s*'" + re.escape(name) + r"':", re.MULTILINE)
    m = entry_re.search(src)
    assert m is not None, f'glass {name!r} not found in SELLMEIER_COEFFICIENTS block'

    # Walk backwards through the preceding comment block (up to 10 lines)
    # to find the most-recent ``# n_d = X.YYYYY`` comment.
    lines = src[:m.start()].splitlines()
    nd_comment_re = re.compile(r'#\s*n_d\s*=\s*([0-9]+\.[0-9]+)')
    cited_val = None
    for line in reversed(lines[-12:]):
        cm = nd_comment_re.search(line)
        if cm:
            cited_val = float(cm.group(1))
            break
    assert cited_val is not None, (
        f"glass {name!r}: could not find a ``# n_d = <value>`` comment "
        f"in the 12 lines preceding the dict entry")
    assert abs(cited_val - expected) < tol, (
        f"glass {name!r}: inline comment cites n_d={cited_val:.6f}; "
        f"computed/expected n_d={expected:.5f}; "
        f"diff={abs(cited_val-expected):.2e} > tol {tol:.0e}.  "
        f"The v4.16.1 fix should have aligned this; comment drift recurred.")


def test_glass_inline_comments_no_stale_pre_v4_16_1_values():
    """Belt-and-suspenders: the four exact pre-v4.16.1 stale strings
    must not appear in glass.py.
    """
    src = _read_glass_source()
    # These four lines were the EXACT pre-fix `n_d = ...` strings;
    # they must NOT appear post-fix.  (The patch may legitimately
    # mention them inside the audit-history commentary, so we look
    # specifically for the bare ``n_d = X.YYYYYY,`` form -- which the
    # pre-fix code used as the leading inline annotation.)
    stale_patterns = [
        r'#\s*n_d\s*=\s*1\.613750\s*,',   # H-ZK9B pre-fix
        r'#\s*n_d\s*=\s*1\.673000\s*,',   # H-ZF12 pre-fix
        r'#\s*n_d\s*=\s*1\.796800\s*,',   # H-ZLAF52A pre-fix
        # D-LAK52 pre-fix cited 1.729160 -- which is *also* H-LAK52's
        # correct value, so we cannot blanket-forbid that string; we
        # rely on the per-glass test above to pin D-LAK52 specifically.
    ]
    for pat in stale_patterns:
        assert re.search(pat, src) is None, (
            f"glass.py still contains stale pre-v4.16.1 n_d comment "
            f"matching {pat!r}.  The v4.16.1 Agent-D patch should "
            f"have rewritten these.")


# ============================================================================
# D.2 -- refractiveindex moved to optional deps
# ============================================================================

def _read_pyproject() -> str:
    return _PYPROJECT.read_text(encoding='utf-8')


def _extract_extras_block(src: str, group_name: str) -> str:
    """Return the body (between ``[`` and the matching ``]``) of a TOML
    array-of-strings assignment of the form ``group_name = [ ... ]``.

    A simple non-greedy regex match cannot be used because the block
    body may contain comments that mention ``]`` (e.g. inside a
    string literal mentioning ``lumenairy[all]``).  This helper
    accepts both single-line (``zarr = ["foo>=3.0", "bar>=1.0"]``)
    and multi-line forms.  For the multi-line form, the body is
    accumulated line-by-line until the first line consisting only of
    optional whitespace + ``]``.
    """
    # Try single-line form first.
    single_line = re.search(
        r'^\s*' + re.escape(group_name) + r'\s*=\s*\[([^\[\]]*)\]\s*$',
        src,
        re.MULTILINE,
    )
    if single_line:
        return single_line.group(1)

    # Multi-line form.
    pattern = re.compile(
        r'^\s*' + re.escape(group_name) + r'\s*=\s*\[\s*$',
        re.MULTILINE,
    )
    m = pattern.search(src)
    if m is None:
        return ''
    start = m.end()
    body_lines = []
    for line in src[start:].splitlines():
        if re.match(r'^\s*\]\s*$', line):
            break
        body_lines.append(line)
    return '\n'.join(body_lines)


def test_pyproject_refractiveindex_not_in_hard_deps():
    """`refractiveindex` must NOT be in `[project.dependencies]`."""
    src = _read_pyproject()
    deps_block = _extract_extras_block(src, 'dependencies')
    assert deps_block, 'pyproject.toml missing dependencies block'
    assert 'refractiveindex' not in deps_block, (
        'refractiveindex must be in [project.optional-dependencies.glass], '
        'not [project.dependencies].  Audit M-1 (v4.16.1) closure.')


def test_pyproject_refractiveindex_in_glass_extras():
    """`refractiveindex>=1.0` must be in the `glass` optional group."""
    src = _read_pyproject()
    glass_block = _extract_extras_block(src, 'glass')
    assert glass_block, (
        'pyproject.toml missing `glass` optional-dependency group')
    assert 'refractiveindex>=1.0' in glass_block, (
        '`glass` extras must declare `refractiveindex>=1.0`')


def test_pyproject_refractiveindex_in_all_extras():
    """`refractiveindex>=1.0` must be in the `all` optional group."""
    src = _read_pyproject()
    all_block = _extract_extras_block(src, 'all')
    assert all_block, 'pyproject.toml missing `all` extras group'
    assert 'refractiveindex>=1.0' in all_block, (
        '`all` extras must include `refractiveindex>=1.0`')


# ============================================================================
# D.3 -- zarr floor bumped to >=3.0
# ============================================================================

def test_pyproject_zarr_floor_v3():
    """`zarr` extras must declare `zarr>=3.0` (Group.create_array is
    Zarr v3-only).  Audit M-3 (v4.16.1) closure.
    """
    src = _read_pyproject()
    # `zarr` group.
    block = _extract_extras_block(src, 'zarr')
    assert block, 'pyproject.toml missing `zarr` extras group'
    assert 'zarr>=3.0' in block, (
        '`zarr` extras must declare zarr>=3.0 (was zarr>=2.14 pre-v4.16.1)')
    assert 'zarr>=2.14' not in block, (
        '`zarr` extras still contains stale `zarr>=2.14` floor')

    # `all` group also.
    all_block = _extract_extras_block(src, 'all')
    assert 'zarr>=3.0' in all_block, (
        '`all` extras must declare zarr>=3.0 (was zarr>=2.14 pre-v4.16.1)')


# ============================================================================
# D.4 -- ProcessPoolExecutor uses mp_context='spawn'
# ============================================================================

def test_lens_traced_pool_uses_spawn_context():
    """`_lens_traced.py` must construct `ProcessPoolExecutor` with an
    explicit `mp_context=` argument bound to a spawn context.  Audit
    M-2 (v4.16.1) closure -- the library worker pool was previously
    using the default fork start method on Linux despite
    CHANGELOG/README claims to the contrary.
    """
    src = _LENS_TRACED.read_text(encoding='utf-8')
    # Both substrings should be present.
    assert "mp_context=" in src, (
        '_lens_traced.py does not pass `mp_context=` to '
        'ProcessPoolExecutor; the v4.16.1 spawn-context fix is missing.')
    assert "get_context('spawn')" in src or 'get_context("spawn")' in src, (
        '_lens_traced.py does not call `multiprocessing.get_context("spawn")`; '
        'the v4.16.1 spawn-context fix is missing.')

    # AST-level: find the ProcessPoolExecutor() call and inspect its
    # kwargs.
    tree = ast.parse(src)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name == 'ProcessPoolExecutor':
                kw_names = [kw.arg for kw in node.keywords]
                assert 'mp_context' in kw_names, (
                    'ProcessPoolExecutor call in _lens_traced.py missing '
                    '`mp_context=` kwarg.')
                found = True
    assert found, (
        '_lens_traced.py: no ProcessPoolExecutor(...) call found at the '
        'AST level (test may need updating if the worker pool moved).')


# ============================================================================
# D.5 -- stray file cleanup verified
# ============================================================================

def test_stray_c_tmp_diff_file_removed():
    """The `C:tmpoptimize_diff.txt` echo-redirect artifact (a git
    diff redirected via a typo'd ``C:\\tmp\\...`` path that Windows
    substituted with the U+F03A private-use code point) must not
    exist at the repo root.  Audit UX item 20 closure.
    """
    # Use os.listdir to surface the Unicode-substituted filename.
    for entry in os.listdir(str(_REPO_ROOT)):
        # The literal filename was 'Ctmpoptimize_diff.txt'.
        if ('tmp' in entry.lower() and 'optimize_diff' in entry.lower()
                and entry.endswith('.txt')):
            pytest.fail(
                f'Stray echo-redirect artifact still present at repo '
                f'root: {entry!r}.  v4.16.1 Agent-D cleanup should '
                f'have removed it.')


# ============================================================================
# D.6 -- stray PNG cleanup verified
# ============================================================================

@pytest.mark.parametrize('png_name', [
    'algebra_4f_system.png',
    'algebra_anamorphic.png',
    'singlet_opd_summary.png',
])
def test_stray_pngs_not_at_repo_root(png_name):
    """The 3 stray example output PNGs must not exist at repo root."""
    assert not (_REPO_ROOT / png_name).exists(), (
        f'{png_name} still exists at repo root.  v4.16.1 Agent-D '
        f'should have moved it to examples/output/.')


def test_examples_output_dir_exists_and_contains_pngs():
    """The 3 example PNGs should be writable to `examples/output/`.

    v5.0.1 (CI fix post-PyPI): the v4.16.1 spec was that the 3 producing
    scripts (``algebra_4f_system.py`` / ``algebra_anamorphic.py`` /
    ``plot_opd_summary_singlet.py``) write their PNG output to
    ``examples/output/``.  The directory + PNGs are both gitignored
    (``output/`` and ``*.png`` in ``.gitignore``), so on a fresh CI
    checkout the directory doesn't exist and the PNGs aren't present
    -- this pin previously required a developer to run the examples
    locally before the test would pass.

    Refresh: assert the 3 producing scripts WOULD write to
    ``examples/output/`` (structural source-inspection check, not a
    filesystem-state check).  If a contributor runs the examples
    locally, the PNGs land in the right place; CI doesn't need them
    pre-built.
    """
    expected_scripts = (
        'algebra_4f_system.py',
        'algebra_anamorphic.py',
        'plot_opd_summary_singlet.py',
    )
    for script_name in expected_scripts:
        script = _EXAMPLES_DIR / script_name
        assert script.is_file(), (
            f'{script_name} is missing from examples/')
        src = script.read_text(encoding='utf-8')
        assert "examples/output" in src or "'output'" in src or '"output"' in src, (
            f'{script_name} does not route its PNG output through '
            f'examples/output/; the v4.16.1 Agent-D move was meant to '
            f'centralise example outputs there.')


# ============================================================================
# D.7 -- new examples/07_zemax_load_trace.py
# ============================================================================

def test_zemax_example_exists():
    """The new `examples/07_zemax_load_trace.py` must exist."""
    target = _EXAMPLES_DIR / '07_zemax_load_trace.py'
    assert target.is_file(), (
        f'{target} does not exist.  v4.16.1 Agent-D should have '
        f'created it (audit UX item 22).')


def test_zemax_example_parses_as_python():
    """The new example must parse cleanly as Python source."""
    target = _EXAMPLES_DIR / '07_zemax_load_trace.py'
    src = target.read_text(encoding='utf-8')
    # parse will raise SyntaxError if the file is malformed.
    tree = ast.parse(src, filename=str(target))
    # Sanity: there should be a ``main`` function defined.
    func_names = {n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)}
    assert 'main' in func_names, (
        '07_zemax_load_trace.py should define a ``main()`` function')


def test_zemax_example_uses_load_zemax_zmx_or_make_singlet_fallback():
    """The example must reference `load_zemax_zmx` (per its purpose)
    and a `make_singlet` fallback so it runs even when the .zmx
    fixture is absent.
    """
    target = _EXAMPLES_DIR / '07_zemax_load_trace.py'
    src = target.read_text(encoding='utf-8')
    assert 'load_zemax_zmx' in src, (
        '07_zemax_load_trace.py must call `la.load_zemax_zmx` to fulfil '
        'its documented purpose')
    assert 'make_singlet' in src, (
        '07_zemax_load_trace.py should provide a `make_singlet` '
        'fallback so the example runs on a fresh checkout without '
        'the validation/ Zemax fixtures')


# ============================================================================
# D.8 -- CONVENTIONS.md exists and documents create_* vs make_*
# ============================================================================

def test_conventions_md_exists():
    """Top-level `CONVENTIONS.md` must exist."""
    assert _CONVENTIONS.is_file(), (
        f'{_CONVENTIONS} does not exist.  v4.16.1 Agent-D should have '
        f'created it (audit item 23).')


def test_conventions_md_documents_create_and_make_factories():
    """CONVENTIONS.md must contain both `create_*` (field/Source)
    and `make_*` (prescription/bundle) sections.
    """
    src = _CONVENTIONS.read_text(encoding='utf-8')
    assert 'create_*' in src, (
        'CONVENTIONS.md must document the `create_*` factory verb')
    assert 'make_*' in src, (
        'CONVENTIONS.md must document the `make_*` factory verb')
    # Each verb should cite at least one example function.
    assert 'create_gaussian_beam' in src, (
        'CONVENTIONS.md should cite `create_gaussian_beam` as a '
        '`create_*` example')
    assert 'make_singlet' in src, (
        'CONVENTIONS.md should cite `make_singlet` as a `make_*` example')


# ============================================================================
# Cross-cutting: factory-verb contract is respected by the live API
# ============================================================================

def test_factory_verb_naming_contract():
    """Walk the public lumenairy namespace and verify the
    `create_*` / `make_*` contract holds.

    Rule: every NEW `create_*` symbol is a callable expected to
    produce a field-like object; every NEW `make_*` symbol is a
    callable expected to produce a non-field aggregate (dict / bundle
    / BSDF / etc).  This test is INTENTIONALLY informational -- it
    only fails if an unexpected `make_*` symbol slips through with a
    field-style return type, where "field-style" is detected by the
    presence of an `N` (grid size) parameter in the function
    signature.
    """
    import inspect
    import lumenairy as la

    # Known-good exceptions:
    #   - `make_lg_aberration_merit_jax` constructs a JAX-merit
    #     closure rather than a field/aggregate (a callable wrapper).
    #   - `make_jax_ray_state` takes ``N`` as the z-direction cosine
    #     of each ray (per the RayBundle convention L/M/N), NOT as a
    #     grid size.  The collision with the heuristic ``N`` token
    #     below is incidental.
    EXEMPT = {'make_lg_aberration_merit_jax', 'make_jax_ray_state'}

    violations = []
    for name in dir(la):
        if not name.startswith('make_'):
            continue
        if name in EXEMPT:
            continue
        obj = getattr(la, name)
        if not callable(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        # If the make_* helper takes a grid-size kwarg `N`, it is
        # likely producing a field -- which should be `create_*`.
        # (Exempt cases where ``N`` is unambiguously a ray-direction
        # cosine -- those are in EXEMPT above.)
        if 'N' in params:
            violations.append(name)

    assert not violations, (
        f'Factory-verb contract violation: these `make_*` symbols take a '
        f'grid-size `N` parameter (suggesting they produce a field, which '
        f'should be `create_*`): {violations}.  Either rename to '
        f'`create_*` or add to the EXEMPT set with a documented reason.')


# ============================================================================
# Bonus -- sanity import check
# ============================================================================

def test_lumenairy_imports_cleanly():
    """Smoke-check: `import lumenairy as la` after Agent-D edits."""
    import importlib
    import lumenairy
    importlib.reload(lumenairy)
    # If we got here, the import succeeded.
    assert hasattr(lumenairy, 'get_glass_index')
