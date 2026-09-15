"""Probe -- what ELSE in tests/ and scripts/ cannot be imported on Python 3.10?

The 3.10 CI lane on run 34914295323 died at the FIRST collection error
(``tests/unit/test_audit2609_a15a_packaging.py``'s unconditional ``import
tomllib``), and pytest stops collecting at the first one, so the artefacts
cannot say whether a second module would have failed next.  This walks every
``tests/**.py`` and ``scripts/**.py`` statically and reports:

  * module-scope imports of a stdlib module that does not exist on 3.10, with
    whether the import is GUARDED (inside a ``try`` or an ``if``);
  * syntax that 3.10's parser rejects (``ast.parse`` with
    ``feature_version=(3, 10)`` -- catches PEP 695 ``type`` aliases and the
    3.12 generic-parameter syntax);
  * ``except*`` / ``ExceptionGroup`` use, which is 3.11+.

"Module scope" means depth-0 in the module body; an import inside a function
or a class cannot break COLLECTION, only the test that reaches it, so those
are reported separately as a lower-severity class.

Writes ``t3_py310_import_scan_static.json`` beside this file.
"""
import ast
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))

# stdlib modules that do not exist on Python 3.10
NEW_STDLIB = {
    'tomllib': '3.11',
    'asyncio.taskgroups': '3.11',
    'wsgiref.types': '3.11',
    'hashlib.file_digest': '3.11',
    'typing_extensions.override': '-',
    'importlib.resources.abc': '3.11',
    'dbm.sqlite3': '3.13',
    'annotationlib': '3.14',
    'compression': '3.14',
}


def _iter_files():
    for sub in ('tests', 'scripts'):
        for root, dirs, files in os.walk(os.path.join(REPO, sub)):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for fn in sorted(files):
                if fn.endswith('.py'):
                    yield os.path.join(root, fn)


def _guarded(node, guards):
    """Is this node lexically inside a try: or if: at module scope?"""
    return any(node.lineno >= a and node.lineno <= b for a, b in guards)


def main():
    findings = {'module_scope_new_stdlib': [], 'nested_new_stdlib': [],
                'py310_syntax_reject': [], 'exception_group': []}
    n_files = 0
    for path in _iter_files():
        rel = os.path.relpath(path, REPO).replace(os.sep, '/')
        with open(path, encoding='utf-8', errors='replace') as fh:
            src = fh.read()
        n_files += 1
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            findings['py310_syntax_reject'].append(
                {'file': rel, 'line': e.lineno, 'msg': 'unparseable: %s' % e})
            continue
        # 3.10 parser acceptance
        try:
            ast.parse(src, feature_version=(3, 10))
        except SyntaxError as e:
            findings['py310_syntax_reject'].append(
                {'file': rel, 'line': e.lineno, 'msg': str(e)})

        if 'except*' in src or 'ExceptionGroup' in src:
            findings['exception_group'].append(rel)

        # module-scope try:/if: spans -- an import inside one is guarded
        guards = []
        for node in tree.body:
            if isinstance(node, (ast.Try, ast.If)):
                guards.append((node.lineno, max(
                    getattr(n, 'lineno', node.lineno)
                    for n in ast.walk(node))))

        module_scope_lines = set()
        for node in tree.body:
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                    break
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_scope_lines.add(node.lineno)
            elif isinstance(node, (ast.Try, ast.If)):
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        module_scope_lines.add(sub.lineno)

        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for nm in names:
                top = nm.split('.')[0]
                key = nm if nm in NEW_STDLIB else (
                    top if top in NEW_STDLIB else None)
                if key is None:
                    continue
                rec = {'file': rel, 'line': node.lineno, 'module': nm,
                       'since': NEW_STDLIB[key],
                       'guarded': bool(_guarded(node, guards))}
                if node.lineno in module_scope_lines:
                    findings['module_scope_new_stdlib'].append(rec)
                else:
                    findings['nested_new_stdlib'].append(rec)

    findings['files_scanned'] = n_files
    findings['scanner_python'] = sys.version.split()[0]
    path = os.path.join(HERE, 't3_py310_import_scan_static.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(findings, fh, indent=2)

    print('scanned %d .py under tests/ and scripts/' % n_files)
    print('MODULE-SCOPE imports of post-3.10 stdlib (these break COLLECTION '
          'if unguarded):')
    for r in findings['module_scope_new_stdlib']:
        print('   %-6s %s:%d  %s (since %s)'
              % ('GUARD' if r['guarded'] else 'BARE', r['file'], r['line'],
                 r['module'], r['since']))
    print('nested (function/class scope) imports of post-3.10 stdlib:')
    for r in findings['nested_new_stdlib']:
        print('   %-6s %s:%d  %s (since %s)'
              % ('GUARD' if r['guarded'] else 'BARE', r['file'], r['line'],
                 r['module'], r['since']))
    print('files 3.10 parser REJECTS: %d'
          % len(findings['py310_syntax_reject']))
    for r in findings['py310_syntax_reject']:
        print('   %s:%s  %s' % (r['file'], r['line'], r['msg'][:110]))
    print('files mentioning except* / ExceptionGroup (3.11+): %d %s'
          % (len(findings['exception_group']), findings['exception_group']))
    print('json:', path)


if __name__ == '__main__':
    main()
