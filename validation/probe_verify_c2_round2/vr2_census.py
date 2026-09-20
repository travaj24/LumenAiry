"""VERIFY-WP-C2 ROUND 2 -- an INDEPENDENT, MODULE-QUALIFIED census of every
exported function in the package whose own body names a tracer.

Differences from the shipped census in
``tests/unit/test_c2_analytic_normal_default.py::_c2_entry_point_census``:

* keys are ``module:function``, not a bare function NAME.  The shipped
  census collapses same-named functions across modules into one key and
  then reads the signature of whichever object the seven scanned modules
  happen to export under that name, so a tracing private-module function
  can be masked by a same-named public one (and vice versa).
* "exported" is decided by walking EVERY importable ``lumenairy`` module
  and asking whether the object is reachable as an attribute of some
  module, rather than from a hard-coded list of seven modules.
* nested / method definitions are reported separately from module-level
  ones so nothing is silently dropped.
* source is read as cp1252 (this repository's convention) with a utf-8
  fallback, not utf-8-with-replacement.

Usage:  python vr2_census.py <out.json>
"""
import ast
import importlib
import inspect
import json
import pathlib
import pkgutil
import sys

TRACERS = {'trace', 'trace_world', 'trace_prescription',
           'raytrace_system', 'trace_jax', 'trace_jax_world'}
WAY_BACK = ('sphere_normal', 'renormalize')


def read_source(path):
    raw = pathlib.Path(path).read_bytes()
    for enc in ('cp1252', 'utf-8'):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', 'replace'), 'utf-8-replace'


def names_a_tracer(node):
    """Tracer names this function's own body mentions, in ANY form."""
    hits = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in TRACERS:
            hits.add(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in TRACERS:
            hits.add(sub.attr)
    return sorted(hits)


def main(out_path):
    import lumenairy as la
    print('lumenairy.__file__ =', la.__file__)
    root = pathlib.Path(la.__file__).parent

    # ---- 1. AST pass: every function definition in the package ----------
    defs = {}
    for f in sorted(root.rglob('*.py')):
        rel = f.relative_to(root).as_posix()
        modname = 'lumenairy.' + rel[:-3].replace('/', '.')
        if modname.endswith('.__init__'):
            modname = modname[:-len('.__init__')]
        src, enc = read_source(f)
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            defs['%s:<SYNTAXERROR>' % modname] = {'error': str(exc)}
            continue

        def walk(node, prefix, depth, modname=modname, rel=rel, enc=enc):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = (prefix + '.' if prefix else '') + child.name
                    hits = names_a_tracer(child)
                    if hits:
                        a = child.args
                        params = ([x.arg for x in a.posonlyargs]
                                  + [x.arg for x in a.args]
                                  + [x.arg for x in a.kwonlyargs])
                        defs['%s:%s' % (modname, qual)] = {
                            'module': modname, 'qualname': qual,
                            'file': rel, 'encoding': enc,
                            'tracers_named': hits,
                            'top_level': depth == 0,
                            'private_name': child.name.startswith('_'),
                            'ast_params': params,
                            'ast_kwonly': [x.arg for x in a.kwonlyargs],
                            'ast_way_back': [k for k in WAY_BACK
                                             if k in params],
                            'lineno': child.lineno,
                        }
                    walk(child, qual, depth + 1)
                elif isinstance(child, ast.ClassDef):
                    walk(child,
                         (prefix + '.' if prefix else '') + child.name,
                         depth + 1)

        walk(tree, '', 0)

    # ---- 2. import every module, record what each one exports -----------
    exported_at = {}
    import_errors = {}
    modnames = ['lumenairy']
    for m in pkgutil.walk_packages([str(root)], prefix='lumenairy.'):
        modnames.append(m.name)
    for mn in sorted(set(modnames)):
        try:
            mod = importlib.import_module(mn)
        except Exception as exc:          # noqa: BLE001
            import_errors[mn] = '%s: %s' % (type(exc).__name__, exc)
            continue
        for attr in dir(mod):
            if attr.startswith('_'):
                continue
            try:
                obj = getattr(mod, attr)
            except Exception:             # noqa: BLE001
                continue
            if inspect.isfunction(obj):
                exported_at.setdefault(id(obj), []).append(
                    '%s.%s' % (mn, attr))

    # ---- 3. join ---------------------------------------------------------
    rows = {}
    for key, info in sorted(defs.items()):
        if 'error' in info:
            rows[key] = info
            continue
        obj = None
        try:
            obj = importlib.import_module(info['module'])
            for part in info['qualname'].split('.'):
                obj = getattr(obj, part)
        except Exception:                 # noqa: BLE001
            obj = None
        names = exported_at.get(id(obj), []) if obj is not None else []
        info = dict(info)
        info['exported_as'] = sorted(names)
        info['is_exported'] = bool(names)
        info['sig_way_back'] = None
        info['way_back_keyword_only'] = None
        info['defaults_none'] = None
        if obj is not None and inspect.isfunction(obj):
            try:
                params = inspect.signature(obj).parameters
            except (TypeError, ValueError):
                params = None
            if params is not None:
                info['sig_way_back'] = [k for k in WAY_BACK if k in params]
                info['way_back_keyword_only'] = [
                    k for k in info['sig_way_back']
                    if params[k].kind == inspect.Parameter.KEYWORD_ONLY]
                info['defaults_none'] = {
                    k: (params[k].default is None)
                    for k in info['sig_way_back']}
        rows[key] = info

    exported_tracing = {k: v for k, v in rows.items()
                        if v.get('is_exported') and not v.get('private_name')}
    both = {k: v for k, v in exported_tracing.items()
            if v.get('sig_way_back') and len(v['sig_way_back']) == 2}
    lacking = {k: v for k, v in exported_tracing.items()
               if not v.get('sig_way_back') or len(v['sig_way_back']) < 2}

    summary = {
        'python': sys.version.split()[0],
        'lumenairy_file': la.__file__,
        'n_defs_naming_a_tracer': len(defs),
        'n_exported_tracing': len(exported_tracing),
        'n_with_both_keywords': len(both),
        'n_lacking': len(lacking),
        'with_both': sorted(both),
        'lacking': sorted(lacking),
        'lacking_detail': {k: {'exported_as': lacking[k]['exported_as'],
                               'tracers_named': lacking[k]['tracers_named'],
                               'sig_way_back': lacking[k]['sig_way_back'],
                               'file': lacking[k]['file']}
                           for k in sorted(lacking)},
        'import_errors': import_errors,
        'rows': rows,
    }
    pathlib.Path(out_path).write_text(json.dumps(summary, indent=1),
                                      encoding='utf-8')
    print('exported+tracing:', len(exported_tracing),
          ' both keywords:', len(both), ' lacking:', len(lacking))
    for k in sorted(lacking):
        print('   LACKING', k, lacking[k]['sig_way_back'],
              lacking[k]['tracers_named'])
    print('import errors:', len(import_errors))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'vr2_census.json')
