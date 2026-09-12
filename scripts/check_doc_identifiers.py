#!/usr/bin/env python3
# v5.46 (audit 2026-09-11, WP-A18 D1 -> WP-A21): doc-identifier resolver.
"""check_doc_identifiers.py -- does every backticked API name in the docs exist?

The 2026-09-11 adversarial audit sampled 60 backticked tokens from the four
top-level documents with ``random.seed(0)`` and found 13 resolving against the
package and 47 not -- a 78 % miss rate -- while saying of its own number
"*that headline overstates the problem... I did not build a clean
denominator*".  It was right on both counts.  Most of the 47 were never API
claims at all: exception names, keyword-argument names, test ids, other
projects' APIs named as a comparison, and enum-like option STRINGS that the
docs spell bare.  WP-A18 built the clean denominator, drove the genuinely
unresolved count to 0, and asked for the resolver to be committed so the
result is a gate rather than a one-off measurement.  This is that resolver.

WHAT IT CHECKS.  Every ```backticked``` token in ``README.md``,
``ROADMAP.md``, ``Migration-Guide.md`` and ``CONVENTIONS.md`` that LOOKS like a
Python identifier (``IDENT_RE``: a dotted chain of identifier segments, with an
optional trailing ``()``), minus two layers of exclusions, must resolve against
the package.

WHY TWO LAYERS, and why the second one is a list rather than a rule.  The
MECHANICAL layer (``classify``) removes what can be recognised by shape or by
membership in a set the package itself defines: builtin and exception names,
parameter names taken from the live signatures of every callable in the
package, test ids, file names, third-party and stdlib prefixes, enum-like
option values and file-format record tokens (``RDY``, ``THI``, ``DIM``, ...).
The HAND-TRIAGED layer (``CURATED``) is the remainder: 51 tokens that were each
read at their citation site and found not to be a claim about the CURRENT API
-- removed or renamed names that a migration guide has to be able to spell,
local expressions inside an illustrative snippet, names that live in the test
corpus, ROADMAP names for things that deliberately do not exist yet, other
projects' API, placeholders in a naming rule.  Each carries its reason inline.
It is a list and not a heuristic on purpose: a heuristic that swallowed those
would also swallow a real regression, and the denominator has to be auditable.

HOW RESOLUTION WORKS.  Three sources, in order:

* the IMPORTABLE package -- dotted paths rooted at ``lumenairy`` / the ``la``
  alias walk modules then attributes; ``Foo.bar`` chains are tried against
  every module that exposes ``Foo``; a bare name is tried against
  ``lumenairy``, then every submodule's ``dir()``, then module leaf names;
* a STATIC AST index of ``lumenairy/**/*.py`` -- module-level attributes, class
  members (including ``self.x = ...``) and nested ``def`` / ``class``
  definitions.  This is what makes ``lumenairy/ui/`` resolvable at all (it
  needs PySide6, which is not installed on the calibration box or on the CI
  runners), and what makes ``Surface.world_origin``,
  ``MultiFieldMerit.field_angles``, ``_RestoreDtype`` and ``_merit_jac_auto``
  resolve honestly instead of being counted as breakage;
* nothing else.  There is NO network access and no package metadata lookup:
  the script reads four markdown files and the installed source tree.

Runtime is dominated by importing the package and its submodules -- MEASURED
4.8 s end to end including interpreter start, 1 952 backticked occurrences /
1 055 distinct identifier-shaped tokens / a 592-token API-claiming denominator,
on the calibration box.

Exit status: 0 when every API-claiming token resolves, 1 otherwise (the
unresolved tokens are listed with their citation sites).

Usage::

    python scripts/check_doc_identifiers.py
    python scripts/check_doc_identifiers.py --list-unresolved --per-file
    python scripts/check_doc_identifiers.py --no-exclusions   # mechanical only
"""
from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import inspect
import keyword
import os
import pkgutil
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The four documents the audit's D1 deliverable covers.  They are the ones a
#: user reads before touching the API; a stale name in any of them is an
#: ``ImportError`` at the reader's first call.
DOC_FILES = ('README.md', 'ROADMAP.md', 'Migration-Guide.md', 'CONVENTIONS.md')

TOKEN_RE = re.compile(r'`([^`\n]+)`')
IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*'
                      r'(?:\(\))?$')

STDLIB_PREFIXES = (
    'np.', 'numpy.', 'scipy.', 'jax.', 'jnp.', 'cp.', 'cupy.', 'plt.',
    'matplotlib.', 'h5py.', 'zarr.', 'numba.', 'pytest.', 'os.', 'sys.',
    'time.', 'json.', 'math.', 'dataclasses.', 'typing.', 'importlib.',
    'warnings.', 'inspect.', 'pathlib.', 'collections.', 'itertools.',
    'functools.', 'pickle.', 'copy.', 'threading.', 'multiprocessing.',
    'subprocess.', 'logging.', 'unittest.', 'pymoo.', 'pyfftw.', 'numexpr.',
    'psutil.', 'filelock.', 'astropy.', 'refractiveindex.', 'PySide6.',
    'threadpoolctl.',
)
FILE_EXT = ('.py', '.md', '.txt', '.json', '.npz', '.npy', '.csv', '.toml',
            '.yml', '.yaml', '.h5', '.zarr', '.zmx', '.seq', '.cfg', '.in',
            '.ini', '.rst', '.html', '.png', '.dat', '.log', '.lock', '.sh')

THIRD_PARTY = {
    'numpy', 'np', 'scipy', 'jax', 'jnp', 'cupy', 'cp', 'numba', 'numexpr',
    'h5py', 'zarr', 'filelock', 'pyfftw', 'astropy', 'refractiveindex',
    'pymoo', 'matplotlib', 'plt', 'pyvista', 'pyvistaqt', 'PySide6', 'pytest',
    'ruff', 'mypy', 'pip', 'copyreg', 'abc', 'abc.ABC', 'dataclasses',
    'threadpoolctl', 'ProcessPoolExecutor', 'ThreadPoolExecutor',
    'NonlinearConstraint', 'RectBivariateSpline', 'dual_annealing',
    'custom_vjp', 'NullHandler', 'create_dataset', 'create_array', 'QToolBar',
    'fitInView', 'S4', 'Optical_Propagation_Library', '__main__',
    '__subclasses__()', 'py.typed', 'LICENSE', 'Releases', 'Changelog',
    'Cookbook', 'prysm', 'poppy', 'lightpipes', 'diffractio',
    '_PyArray_UFuncBufferedAtVectorized', 'check_source_line_citations',
    'lumenairy_version', 'create_', 'make_', 'exp', 'the', 'vs', 'IS',
}
#: Enum-like OPTION VALUES the docs spell bare.  They are STRINGS in the API
#: (``polarization='te'``), never symbols, so resolving them against the
#: package would be a category error -- plus Qt signal / slot names and a few
#: prescription / result dict keys quoted in prose.
OPTION_VALUES = {
    'te', 'tm', 'fff_nv', 'maslov', 'gaussian', 'top_hat', 'plane_wave',
    'point_source', 'fiber_mode', 'hdf5', 'lm', 'traced', '_traced',
    '_maslov', 'distortion', 'spot', 'rayfan', 'psfmtf', 'waveoptics',
    'jones_pupil', 'bench', 'dev', 'lint', 'r_p', 'r_s', 'E_x', 'E_y',
    'use_traced_lens', 'wave_traced', 'auto_retrace_mode', 'raises',
    'lam0', 'lambda_', 'wvl', 'lam', 'wl', 'n_d', 'z_s', 'delta_H',
    'H1', 'H2', '_m', 'displaced', 'local_quadrature', 'stationary_phase',
    'quadrature', 'single', 'multibranch', 'uniform', 'wave', 'screen',
    'ray_density', 'thin', 'tangent_facet', 'tangent_facet_remap', 'remap',
    'split', 'spline', 'newton', 'fit', 'backward_trace', 'southwell',
    'itoh', 'legacy', 'physical', 'asm', 'auto', 'gzip', 'li', 'laurent',
    # registered process-global knob names (lumenairy._knobs registry)
    'blas_threads', 'cache_budget', 'library_path', 'storage_backend',
    'max_ram', 'fft_threads', 'pyfftw_planner', 'asm_cache_size',
    # prescription / result dict keys and field names quoted in prose
    'sag_callable', 'wavefront', 'E_z', 'Jxx', 'Jyy', 'theta_z', 'N_pix',
    'n_exit', 'rho_fit', 'slopes_x', 'dependencies', 'Returns', 'gained',
}
#: CODE V / Zemax record tokens, log levels and other non-Python data tokens.
DATA_TOKENS = {
    'RDY', 'THI', 'WL', 'GLA', 'STO', 'CON', 'APE', 'DISZ', 'INFO', 'WARNING',
    'DIM', 'MNUM', 'MCON', 'NAME', 'COMM', 'PARM', 'MM', 'CM', 'IN', 'M', 'C',
    'I', 'SI', 'nan', 'inf',
}

#: The hand-triaged remainder.  Every token here was inspected at its citation
#: site and is NOT a claim about the CURRENT lumenairy API.  Kept explicit
#: rather than folded into a heuristic so the denominator stays auditable: a
#: rule general enough to swallow these would also swallow a real regression.
#: ADDING TO THIS LIST IS A DECISION, not a fix -- if the docs cite a name that
#: does not exist and is not one of the categories below, change the docs.
CURATED: Dict[str, str] = {
    # names the docs state were REMOVED or RENAMED -- a migration guide and a
    # release-note history have to be able to name them
    'rcwa_1d': 'removed API, documented as removed',
    'lumenairy.analysis.analysis': 'removed shim, documented as removed',
    'analysis.analysis': 'removed shim, documented as removed',
    'lumenairy.ao': 'removed shim, documented as removed',
    'lumenairy.io.hdf5': 'removed shim, documented as removed',
    'io.hdf5': 'removed shim, documented as removed',
    'lumenairy.asymptotic': 'v3.3 module path, relocation stated inline',
    'make_zemax_singlet': 'removed API, documented as removed',
    'cosmic_ray_rate': 'removed kwarg, documented as removed',
    'half_width_x': 'removed kwarg, documented as removed',
    'width_x': 'removed kwarg, documented as removed',
    'RCWADock': 'renamed class, documented as renamed',
    '_paraxial_trace': 'removed private helper, documented as removed',
    # local variables / expressions inside an illustrative snippet
    'E.dtype': 'local expression in an example',
    'phase_exp.dtype': 'local expression in an example',
    'self.dx': 'local expression in an example',
    'self.dy': 'local expression in an example',
    'rays.alive': 'local expression in an example',
    'state.alive': 'local expression in an example',
    'op.apply_real_lens_maslov': 'local expression in an example',
    'patch_grid.centres': 'local expression in an example',
    'Y.__module__': 'local expression in an example',
    '__init__.py.__all__': 'local expression in an example',
    'bsdf.total_integrated_scatter': 'method named through its module',
    'design_optimize._merit_jac_auto': 'nested def named through its owner',
    # names that live in the TEST corpus, not the package
    'ClearAsmCachesChainsAll': 'test citation',
    'EXPECTED_SUBCLASSES': 'test citation',
    '_DELEGATING_CLASS_METHODS': 'test citation',
    '_INTERNAL': 'test citation',
    '_resolve_arg_closure': 'test citation',
    # ROADMAP names for things that do not exist yet, by design
    'LayerSpec': 'proposed future name (ROADMAP)',
    'Rays': 'proposed future name (ROADMAP)',
    # The three lens config objects were ROADMAP proposals when this list was
    # written and are now real (``elements/lens_config.py``, re-exported at
    # top level).  The entries stay only so that this gate does not depend on
    # that landing; they resolve on their own today and can simply be deleted
    # -- MEASURED 2026-09-12, ``_Index().resolve('LensGeometry')`` ->
    # ``(True, 'top-level')`` for all three.
    'LensGeometry': 'ROADMAP name, since landed -- resolves on its own',
    'LensNumerics': 'ROADMAP name, since landed -- resolves on its own',
    'LensResources': 'ROADMAP name, since landed -- resolves on its own',
    # other projects' API, named as a comparison
    'apply_image_plane_fftmft': "POPPY's API, attributed inline",
    'focus_fixed_sampling': "prysm's API, attributed inline",
    'ndarray': 'numpy type name',
    'parallel_scale': 'pyvista camera attribute',
    'view_angle': 'pyvista camera attribute',
    # placeholders in a naming RULE, not names
    '_FOO_LOCK': 'placeholder in a naming rule',
    '_FOO_CACHE_LOCK': 'placeholder in a naming rule',
    '_PATCH_LOCK': 'placeholder in a naming rule',
    # labels in a designer project, not symbols
    'SpatialFilter': 'designer element label',
    'Detector': 'designer element label',
    # dict / file-attribute keys quoted in prose
    # (``wavelength_nm`` was here as "prescription dict key"; it is really a
    #  field of ``ui.model.SourceDefinition`` and resolves through the static
    #  index, so the entry was removed rather than left with a false reason.)
    'n_planes': 'HDF5 attribute key',
    'z_limit': "a cited paper's symbol",
    'groove_index': 'historical UI-dock kwarg',
    'substrate_index': 'historical UI-dock kwarg',
    # Qt signal
    'finished': 'Qt signal name',
    # Migration-Guide.md documents BOTH spellings of the Wood-anomaly
    # diagnostic's fn_name, because the shipped warning said
    # ``RCWA2DPrepared.solve`` until WP-A21 corrected it to the real class,
    # ``PreparedRCWA2D``.  The old spelling therefore has to stay nameable.
    'RCWA2DPrepared.solve': 'historical warning text; the class is PreparedRCWA2D',
}


class _Index:
    """The resolution sources, built once.

    Kept in a class rather than at module scope so that importing this file is
    cheap (the test harness imports it to call :func:`check` in-process) and
    the ~3 s of package import happens only when a check actually runs.
    """

    def __init__(self) -> None:
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        import lumenairy

        self.root = lumenairy
        self.modules = {'lumenairy': lumenairy}
        for mi in pkgutil.walk_packages(lumenairy.__path__, 'lumenairy.'):
            name = mi.name
            if '.ui' in name or name.endswith('.ui'):
                continue                      # needs PySide6; AST index covers it
            try:
                self.modules[name] = importlib.import_module(name)
            except Exception:                 # noqa: BLE001
                # A submodule can fail to import for any reason an OPTIONAL
                # backend chooses (ImportError, OSError from a missing DLL,
                # RuntimeError from a CUDA probe).  The type set is not ours to
                # name, and a module we cannot import is simply not an import
                # resolution source -- the AST index below still covers it.
                pass

        self.attr_owners: Dict[str, List[str]] = defaultdict(list)
        for mname, mod in self.modules.items():
            for a in dir(mod):
                self.attr_owners[a].append(mname)

        self.module_leaf: Dict[str, List[str]] = defaultdict(list)
        for mname in self.modules:
            self.module_leaf[mname.rsplit('.', 1)[-1]].append(mname)

        self._build_static_index()
        self._build_param_names()

    # -- static AST index -------------------------------------------------
    def _build_static_index(self) -> None:
        self.static_attrs: Dict[str, set] = defaultdict(set)
        self.nested_defs: Dict[str, str] = {}
        self.static_members: Dict[str, set] = defaultdict(set)
        pkg_root = os.path.dirname(self.root.__file__)
        for dirpath, _dirs, files in os.walk(pkg_root):
            for fn in files:
                if not fn.endswith('.py'):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, os.path.dirname(pkg_root))
                dotted = rel[:-3].replace(os.sep, '.')
                if dotted.endswith('.__init__'):
                    dotted = dotted[:-len('.__init__')]
                try:
                    with open(full, encoding='utf-8', errors='replace') as fh:
                        tree = ast.parse(fh.read())
                except SyntaxError:
                    continue
                self.module_leaf[dotted.rsplit('.', 1)[-1]].append(dotted)
                for n in ast.walk(tree):
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                        self.nested_defs.setdefault(n.name, dotted)
                for node in tree.body:
                    self._index_top_level(dotted, node)

        self.static_owners: Dict[str, List[str]] = defaultdict(list)
        for m, names in self.static_attrs.items():
            for n in names:
                self.static_owners[n].append(m)
        self.static_member_names: Dict[str, List[str]] = defaultdict(list)
        for cls, names in self.static_members.items():
            for n in names:
                self.static_member_names[n].append(cls)

    def _index_top_level(self, dotted: str, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            self.static_attrs[dotted].add(node.name)
            if isinstance(node, ast.ClassDef):
                for sub2 in ast.walk(node):
                    if (isinstance(sub2, ast.Attribute)
                            and isinstance(sub2.value, ast.Name)
                            and sub2.value.id == 'self'
                            and isinstance(sub2.ctx, ast.Store)):
                        self.static_members[node.name].add(sub2.attr)
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef,
                                        ast.AsyncFunctionDef)):
                        self.static_members[node.name].add(sub.name)
                    elif isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            if isinstance(t, ast.Name):
                                self.static_members[node.name].add(t.id)
                    elif (isinstance(sub, ast.AnnAssign)
                          and isinstance(sub.target, ast.Name)):
                        self.static_members[node.name].add(sub.target.id)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.static_attrs[dotted].add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            self.static_attrs[dotted].add(node.target.id)

    # -- parameter names --------------------------------------------------
    def _build_param_names(self) -> None:
        """Every parameter name of every callable reachable in the package.

        This is what lets the MECHANICAL layer drop ``return_kind``, ``opl_fn``
        and ``image_centres`` -- three of the twelve tokens the audit named --
        without a hand entry: they are real kwargs, not missing symbols.
        """
        self.param_names: set = set()
        for mod in self.modules.values():
            for a in dir(mod):
                try:
                    obj = getattr(mod, a)
                except Exception:             # noqa: BLE001
                    continue                  # a property/descriptor that raises
                if not callable(obj):
                    continue
                try:
                    self.param_names.update(inspect.signature(obj).parameters)
                except (ValueError, TypeError):
                    continue
                for meth in ('__init__', 'solve', 'prepare'):
                    sub = getattr(obj, meth, None)
                    if callable(sub):
                        try:
                            self.param_names.update(
                                inspect.signature(sub).parameters)
                        except (ValueError, TypeError):
                            pass

    # -- resolution -------------------------------------------------------
    def resolve(self, tok: str) -> Tuple[bool, str]:
        """``(resolved, how)`` for one API-claiming token."""
        name = tok[:-2] if tok.endswith('()') else tok
        parts = name.split('.')
        if parts[0] in ('lumenairy', 'la'):
            parts = ['lumenairy'] + parts[1:]
            path = parts[0]
            cur = self.modules.get(path)
            for p in parts[1:]:
                nxt = f'{path}.{p}'
                if nxt in self.modules:
                    cur, path = self.modules[nxt], nxt
                    continue
                if cur is not None and hasattr(cur, p):
                    cur, path = getattr(cur, p), nxt
                    continue
                return False, f'no attribute {p!r} on {path}'
            return True, 'dotted'
        if len(parts) > 1:
            head, rest = parts[0], parts[1:]
            if len(rest) == 1 and rest[0] in self.static_members.get(head, ()):
                return True, 'static class member'
            if len(rest) == 1 and rest[0] in self.static_attrs.get(
                    self.module_leaf.get(head, [''])[0], ()):
                return True, 'static module attr'
            cands = [getattr(self.modules[owner], head)
                     for owner in self.attr_owners.get(head, [])]
            cands += [self.modules[m] for m in self.module_leaf.get(head, [])
                      if m in self.modules]
            for c in cands:
                ok = True
                cur = c
                for p in rest:
                    if not hasattr(cur, p):
                        ok = False
                        break
                    cur = getattr(cur, p)
                if ok:
                    return True, 'attr-chain'
            return False, 'no owner exposes the chain'
        if hasattr(self.root, name):
            return True, 'top-level'
        if name in self.attr_owners:
            return True, 'submodule ' + self.attr_owners[name][0]
        if name in self.module_leaf:
            return True, 'module ' + self.module_leaf[name][0]
        if name in self.static_owners:
            return True, 'static ' + self.static_owners[name][0]
        if name in self.static_members:
            return True, 'static class'
        if name in self.static_member_names:
            return True, 'member of ' + self.static_member_names[name][0]
        if name in self.nested_defs:
            return True, 'nested def in ' + self.nested_defs[name]
        return False, 'not found'


_BUILTIN_NAMES = set(dir(builtins)) | set(keyword.kwlist)


def classify(index: '_Index', tok: str, ctx: str,
             use_curated: bool = True) -> Optional[str]:
    """An exclusion reason, or ``None`` when the token claims API."""
    name = tok[:-2] if tok.endswith('()') else tok
    if name.startswith(STDLIB_PREFIXES) or name.split('.')[0] in (
            'np', 'numpy', 'scipy', 'jax', 'jnp', 'cp', 'plt', 'os', 'sys'):
        return 'third-party / stdlib'
    if name in THIRD_PARTY or tok in THIRD_PARTY:
        return 'third-party / stdlib'
    if name in DATA_TOKENS:
        return 'file-format record token'
    if name in OPTION_VALUES:
        return 'option string value'
    if use_curated and (tok in CURATED or name in CURATED):
        return 'triaged: ' + CURATED.get(tok, CURATED.get(name, ''))
    if name.endswith(FILE_EXT):
        return 'file name'
    if name in _BUILTIN_NAMES:
        return 'builtin / exception name'
    if name.endswith(('Error', 'Warning', 'Exception')):
        return 'builtin / exception name'
    if name.startswith('test_') or '::' in tok or name.startswith('tests'):
        return 'test citation'
    if f'{name}=' in ctx or f'{name} =' in ctx:
        return 'kwarg / parameter name'
    if name in index.param_names:
        return 'kwarg / parameter name'
    return None


class Result:
    """What one run of the resolver found."""

    def __init__(self) -> None:
        self.occurrences = 0
        self.distinct = 0
        self.excluded: Dict[str, Tuple[str, list]] = {}
        self.resolved: Dict[str, Tuple[str, list]] = {}
        self.unresolved: Dict[str, Tuple[str, list]] = {}

    @property
    def denominator(self) -> int:
        return len(self.resolved) + len(self.unresolved)

    @property
    def unresolved_occurrences(self) -> int:
        return sum(len(sites) for _how, sites in self.unresolved.values())


def scan(docroot: Path, files=DOC_FILES) -> list:
    """``[(file, lineno, token, line), ...]`` for identifier-shaped tokens."""
    rows = []
    for f in files:
        path = Path(docroot) / f
        if not path.exists():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for ln, line in enumerate(text.split('\n'), 1):
            # CONVENTIONS.md uses RST-style ``double backticks``; normalise so
            # the single-backtick scanner cannot match the PROSE between two
            # adjacent double-backtick spans.
            line = re.sub(r'``([^`\n]+)``', r'`\1`', line)
            for tok in TOKEN_RE.findall(line):
                tok = tok.strip()
                if IDENT_RE.match(tok):
                    rows.append((f, ln, tok, line))
    return rows


def check(docroot: Optional[Path] = None, files=DOC_FILES,
          use_curated: bool = True, index: Optional['_Index'] = None) -> Result:
    """Run the resolver.  No network; reads only ``docroot`` and the package."""
    index = index if index is not None else _Index()
    rows = scan(Path(docroot) if docroot is not None else _REPO_ROOT, files)

    seen: Dict[str, list] = {}
    for f, ln, tok, line in rows:
        seen.setdefault(tok, []).append((f, ln, line))

    res = Result()
    res.occurrences = len(rows)
    res.distinct = len(seen)
    for tok, sites in seen.items():
        ctx = ' '.join(s[2] for s in sites)
        why = classify(index, tok, ctx, use_curated=use_curated)
        if why:
            res.excluded[tok] = (why, sites)
            continue
        ok, how = index.resolve(tok)
        (res.resolved if ok else res.unresolved)[tok] = (how, sites)
    return res


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description='Resolve backticked API identifiers in the top-level docs '
                    'against the installed lumenairy package.')
    p.add_argument('--docroot', default=str(_REPO_ROOT),
                   help='directory holding the documents (default: repo root)')
    p.add_argument('--list-unresolved', action='store_true',
                   help='print every unresolved token with its citation sites')
    p.add_argument('--per-file', action='store_true',
                   help='print unresolved OCCURRENCES per document')
    p.add_argument('--no-exclusions', action='store_true',
                   help='skip the hand-triaged list (mechanical exclusions '
                        'only) -- this is the auditable "before" denominator, '
                        'not the gate')
    args = p.parse_args(argv)

    res = check(docroot=Path(args.docroot), use_curated=not args.no_exclusions)

    print(f'backticked tokens scanned (occurrences): {res.occurrences}')
    print(f'distinct identifier-shaped tokens:       {res.distinct}')
    by_reason: Dict[str, int] = defaultdict(int)
    for _tok, (why, _s) in res.excluded.items():
        by_reason[why] += 1
    for why, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f'  excluded: {why:<28} {n}')
    print(f'API-claiming denominator (distinct): {res.denominator}')
    print(f'  resolve:        {len(res.resolved)}')
    print(f'  DO NOT resolve: {len(res.unresolved)}')

    if args.list_unresolved and res.unresolved:
        print('\n--- unresolved ---')
        for tok in sorted(res.unresolved):
            how, sites = res.unresolved[tok]
            where = ', '.join(f'{f}:{ln}' for f, ln, _l in sites[:4])
            print(f'{tok:<46} ({len(sites):3d}x) {where}   [{how}]')
    if args.per_file:
        print('\n--- unresolved occurrences per file ---')
        cnt: Dict[str, int] = defaultdict(int)
        for _tok, (_how, sites) in res.unresolved.items():
            for f, _ln, _l in sites:
                cnt[f] += 1
        for f in DOC_FILES:
            print(f'  {f:<20} {cnt[f]}')

    if res.unresolved:
        print(f'\nFAIL: {len(res.unresolved)} backticked identifier(s) in the '
              f'top-level documents do not resolve against the package '
              f'({res.unresolved_occurrences} occurrence(s)).  Either the doc '
              f'names something that no longer exists -- fix the doc -- or the '
              f'token is not an API claim, in which case add it to CURATED '
              f'with the reason you read at its citation site.')
        return 1
    print('\nOK: every API-claiming backticked identifier resolves.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
