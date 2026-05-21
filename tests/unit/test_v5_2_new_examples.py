"""Parsing-pin tests for the v5.2 ROADMAP "5 new examples" item.

Scope: the 5 new ``examples/08_*.py`` ... ``examples/12_*.py`` scripts
must each (a) exist, (b) parse cleanly as Python, (c) define a
``main()`` function, and (d) carry the canonical
``if __name__ == "__main__": main()`` guard.  Mirrors the v4.16.1
parsing pin ``test_zemax_example_parses_as_python`` for
``examples/07_zemax_load_trace.py`` but parametrised over the 5 new
files.

Author: Andrew Traverso -- v5.2 / examples roadmap.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXAMPLES_DIR = _REPO_ROOT / 'examples'

# The 5 ROADMAP-v5.2 example filenames, parametrised so the per-script
# diagnostic appears under each ID in the pytest output.
_NEW_EXAMPLES = (
    '08_multiconfig_zoom.py',
    '09_tolerancing_monte_carlo.py',
    '10_coronagraph_workflow.py',
    '11_ao_closed_loop.py',
    '12_ghost_stray_light.py',
)


def _has_main_guard(tree: ast.AST) -> bool:
    """Return True iff ``tree`` contains a top-level
    ``if __name__ == '__main__':`` guard with a ``main()`` call inside.

    The match is intentionally permissive: an ``if`` whose test is a
    ``Compare`` of ``__name__`` against the string ``'__main__'`` (in
    either order, with either ``==`` or ``is``), and whose body
    contains at least one ``ast.Call`` to a ``main`` name.
    """
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.If):
            continue
        cmp_node = node.test
        if not isinstance(cmp_node, ast.Compare):
            continue
        # Collect every Name and Constant on either side of the compare.
        operands = [cmp_node.left] + list(cmp_node.comparators)
        names = {n.id for n in operands if isinstance(n, ast.Name)}
        consts = {n.value for n in operands if isinstance(n, ast.Constant)}
        if '__name__' not in names or '__main__' not in consts:
            continue
        # Look for a ``main()`` call in the body.
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == 'main'):
                    return True
    return False


@pytest.mark.parametrize('script_name', _NEW_EXAMPLES)
def test_new_v5_2_example_exists(script_name):
    """Each v5.2 ROADMAP example file must exist under ``examples/``."""
    target = _EXAMPLES_DIR / script_name
    assert target.is_file(), (
        f'{target} does not exist.  v5.2 examples roadmap should '
        f'have created it.')


@pytest.mark.parametrize('script_name', _NEW_EXAMPLES)
def test_new_v5_2_example_parses_as_python(script_name):
    """Each v5.2 ROADMAP example must parse cleanly as Python source."""
    target = _EXAMPLES_DIR / script_name
    src = target.read_text(encoding='utf-8')
    # ``ast.parse`` raises ``SyntaxError`` if the file is malformed --
    # that propagates as a test failure with a useful traceback.
    ast.parse(src, filename=str(target))


@pytest.mark.parametrize('script_name', _NEW_EXAMPLES)
def test_new_v5_2_example_defines_main(script_name):
    """Each v5.2 ROADMAP example must define a ``main()`` function."""
    target = _EXAMPLES_DIR / script_name
    src = target.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(target))
    func_names = {n.name for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)}
    assert 'main' in func_names, (
        f'{script_name} should define a ``main()`` function (mirror '
        f'the idiom in examples/07_zemax_load_trace.py).')


@pytest.mark.parametrize('script_name', _NEW_EXAMPLES)
def test_new_v5_2_example_has_main_guard(script_name):
    """Each v5.2 ROADMAP example must carry the canonical
    ``if __name__ == '__main__': main()`` guard at the bottom of the
    file (so ``python examples/<name>.py`` actually runs ``main()``).
    """
    target = _EXAMPLES_DIR / script_name
    src = target.read_text(encoding='utf-8')
    tree = ast.parse(src, filename=str(target))
    assert _has_main_guard(tree), (
        f'{script_name} is missing the canonical '
        f'``if __name__ == "__main__": main()`` guard.  Mirror the '
        f'idiom in examples/07_zemax_load_trace.py.')
