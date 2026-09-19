"""H2-4 (VERIFY-WP-B11c D3) -- the lens facade's write surface, and the
counterfactual that decided which fix to ship.

Run as a CHILD process bound to ONE tree:

    python probe_lens_d3.py <tree> <out.json>

Writes a ``{key: sha256}`` map, so the SAME file is both the behaviour probe
and the archive-to-archive bit-identity probe: keys ``D3-*`` record what the
facade does with a read / write / delete of each of the eight leaf-owned names,
and keys ``L*`` re-run the lens fixtures VERIFY-WP-B11c used, so the numeric
half of the lens family is proved unmoved by the same run that proves the
attribute protocol moved.

WHAT D3 WAS.  Eight names moved to ``_lens_kernels`` are re-exported into
``lenses`` BY VALUE and read at call time out of the leaf's globals, so
``monkeypatch.setattr(lenses, '_is_cupy_array', fake)`` SUCCEEDS, binds a
shadow in ``lenses.__dict__`` and reaches nothing -- the silent no-op the BLAS
half of the same package was arranged to make LOUD.

THE COUNTERFACTUAL, section ``option_A``.  The other fix on the table was to
make all eight LIVE FORWARDS (remove the by-value re-export, serve them from
the module ``__getattr__`` / ``__setattr__`` like ``cp`` and
``_NUMBA_AVAILABLE``).  That fix makes the patch WORK rather than merely fail
loudly, which is better -- if it costs nothing.  It does not cost nothing, and
this section measures the cost on SYNTHETIC modules with no lumenairy import at
all, so the measurement is about Python's attribute and import machinery rather
than about this library:

* ``import *`` on a module WITHOUT ``__all__`` reads ``__dict__`` and consults
  neither ``__getattr__`` nor ``__dir__``.  A name served only by the forward
  therefore VANISHES from ``from ... import *``.  ``lenses`` has no ``__all__``
  (by a documented decision in its own source), and ``CUPY_AVAILABLE`` is the
  one public name among the eight -- so option A is a public-surface change
  inside a durability fix.
* A module's own functions read their globals by ``LOAD_GLOBAL``, which does
  NOT consult ``__getattr__``.  Any future code IN ``lenses.py`` that called
  one of the eight by bare name would raise ``NameError`` under option A.
  (Measured to be zero sites today -- which is why option A is possible at all
  -- but it is a trap the shipped fix does not lay.)

Both are recorded as decisions with their measured readings, so the report's
maintainer paragraph rests on a measurement and not on an argument.
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import hlib  # noqa: E402

import numpy as np  # noqa: E402

hlib.anchor(_TREE)

from lumenairy.elements import _lens_kernels, lenses  # noqa: E402

#: The eight.  Kept in the probe rather than imported from the library so the
#: probe still says what it is measuring when the library's own set moves.
LEAF_OWNED = ("CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
              "_load_numba", "_get_aspheric_sag_accum_numba",
              "_ensure_numexpr_loaded", "_collect_semi_diameters",
              "_warn_if_aperture_exceeds_grid")

LIVE_FORWARD = ("cp", "_ne", "NUMEXPR_AVAILABLE", "_NUMBA_AVAILABLE",
                "_numba", "_njit", "_prange", "_NUMBA_KERNELS")


def _restore(name, had_entry, shell_obj, sentinel):
    """Put ``lenses.__dict__`` back exactly as it was before an attempt."""
    if had_entry:
        if shell_obj is not sentinel:
            vars(lenses)[name] = shell_obj
    else:
        vars(lenses).pop(name, None)


def _attempt(fn, *a):
    """``('ok', value-ish)`` or ``('raised', type, message)``."""
    try:
        return ('ok', repr(fn(*a))[:200])
    except BaseException as exc:                 # noqa: BLE001 -- recorded
        return ('raised', type(exc).__name__, str(exc))


def section_d3(p):
    """Read / write / delete of each of the eight, with the module's state
    RESTORED after every attempt.

    Restored, because on the base tree the attempts SUCCEED: the write leaves a
    shadow in ``lenses.__dict__`` and the delete removes the re-export
    outright, after which ``lenses.CUPY_AVAILABLE`` raises ``AttributeError``
    for the rest of the process.  (That is the defect, stated as a measurement:
    the first run of this probe against the base tree died at exactly that
    point.)  The post-state is digested BEFORE the restore, so what the
    attempt did is still the recorded key.
    """
    sentinel = object()
    for name in LEAF_OWNED:
        leaf_obj = getattr(_lens_kernels, name)
        had_entry = name in vars(lenses)
        shell_obj = vars(lenses).get(name, sentinel)
        # READ: still the leaf's object, still a real dict entry.
        p.add(f"D3-read-{name}", (
            hasattr(lenses, name),
            getattr(lenses, name, sentinel) is leaf_obj,
            had_entry,
            name in dir(lenses),
        ))
        # WRITE: refused on the branch, a silent shadow on the base.
        res = _attempt(setattr, lenses, name, 'SUBSTITUTE')
        p.add(f"D3-write-{name}", res)
        p.add(f"D3-write-had-no-effect-{name}", (
            getattr(lenses, name, sentinel) is leaf_obj,
            name in vars(lenses),
            vars(lenses).get(name, sentinel) is leaf_obj,
            getattr(_lens_kernels, name) is leaf_obj,
        ))
        _restore(name, had_entry, shell_obj, sentinel)
        # DELETE: refused on the branch, removes the re-export on the base.
        res = _attempt(delattr, lenses, name)
        p.add(f"D3-delete-{name}", res)
        p.add(f"D3-delete-had-no-effect-{name}", (
            name in vars(lenses),
            getattr(lenses, name, sentinel) is leaf_obj,
            getattr(_lens_kernels, name) is leaf_obj,
        ))
        _restore(name, had_entry, shell_obj, sentinel)
        # and the restore worked, so the next name starts from the same state
        p.add(f"D3-restored-{name}", (
            name in vars(lenses),
            getattr(lenses, name, sentinel) is leaf_obj,
        ))
    # the LIVE half is untouched by the change
    for name in LIVE_FORWARD:
        saved = getattr(_lens_kernels, name)
        try:
            res = _attempt(setattr, lenses, name, 'LIVE-SENTINEL')
            p.add(f"D3-live-write-{name}", (
                res, getattr(_lens_kernels, name) == 'LIVE-SENTINEL',
                name not in vars(lenses)))
        finally:
            setattr(_lens_kernels, name, saved)
    p.add("D3-dir", tuple(dir(lenses)))
    p.add("D3-facade-type", type(lenses).__name__)
    p.add("D3-mro", tuple(c.__name__ for c in type(lenses).__mro__))


def section_option_a_counterfactual(p):
    """Option A, rebuilt on synthetic modules -- no lumenairy involved."""
    leaf = types.ModuleType('hyg2_leaf')
    leaf.PUBLIC_GATE = True
    leaf._private_fn = lambda: 'leaf'
    sys.modules['hyg2_leaf'] = leaf

    # -- A1: by-value re-export (today's lenses) --------------------------
    shell_v = types.ModuleType('hyg2_shell_byvalue')
    shell_v.PUBLIC_GATE = leaf.PUBLIC_GATE
    shell_v._private_fn = leaf._private_fn
    sys.modules['hyg2_shell_byvalue'] = shell_v
    ns = {}
    exec('from hyg2_shell_byvalue import *', ns)
    p.add("optA-byvalue-import-star-has-public-gate", 'PUBLIC_GATE' in ns)

    # -- A2: served only by __getattr__ (option A) ------------------------
    src = ('_LEAF = __import__("hyg2_leaf")\n'
           '_FWD = {"PUBLIC_GATE", "_private_fn"}\n'
           'import sys as _s, types as _t\n'
           'class _F(_t.ModuleType):\n'
           '    def __getattr__(self, n):\n'
           '        if n in _FWD: return getattr(_LEAF, n)\n'
           '        raise AttributeError(n)\n'
           '    def __setattr__(self, n, v):\n'
           '        if n in _FWD: setattr(_LEAF, n, v); return\n'
           '        super().__setattr__(n, v)\n'
           '    def __dir__(self):\n'
           '        return sorted(set(super().__dir__()) | _FWD)\n'
           '_s.modules[__name__].__class__ = _F\n'
           'def reads_by_bare_name():\n'
           '    return PUBLIC_GATE\n')
    shell_f = types.ModuleType('hyg2_shell_forward')
    sys.modules['hyg2_shell_forward'] = shell_f
    exec(compile(src, 'hyg2_shell_forward', 'exec'), shell_f.__dict__)
    ns = {}
    exec('from hyg2_shell_forward import *', ns)
    p.add("optA-forward-import-star-has-public-gate", 'PUBLIC_GATE' in ns)
    p.add("optA-forward-getattr-works", getattr(shell_f, 'PUBLIC_GATE'))
    p.add("optA-forward-in-dir", 'PUBLIC_GATE' in dir(shell_f))
    p.add("optA-forward-in-vars", 'PUBLIC_GATE' in vars(shell_f))
    # the LOAD_GLOBAL trap: a function IN the shell reading the bare name
    p.add("optA-forward-load-global",
          _attempt(shell_f.reads_by_bare_name))
    # ... and the write really does reach the leaf under option A
    shell_f.PUBLIC_GATE = 'PATCHED'
    p.add("optA-forward-write-reaches-leaf",
          (leaf.PUBLIC_GATE, 'PUBLIC_GATE' not in vars(shell_f)))
    leaf.PUBLIC_GATE = True
    for m in ('hyg2_leaf', 'hyg2_shell_byvalue', 'hyg2_shell_forward'):
        sys.modules.pop(m, None)

    # the same two readings taken on the REAL module, so the counterfactual is
    # anchored to the thing it is about
    ns = {}
    exec('from lumenairy.elements.lenses import *', ns)
    p.add("optA-real-lenses-import-star-public-names",
          tuple(sorted(n for n in ns if n in LEAF_OWNED)))
    src_real = open(os.path.join(_TREE, 'lumenairy', 'elements', 'lenses.py'),
                    encoding='cp1252').read()
    import ast
    tree = ast.parse(src_real)
    bare = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in LEAF_OWNED and \
                isinstance(node.ctx, ast.Load):
            bare.append((node.id, node.lineno))
    p.add("optA-real-lenses-bare-name-load-sites", tuple(sorted(bare)))


# ===========================================================================
# The lens fixtures VERIFY-WP-B11c used -- the numeric half, unmoved
# ===========================================================================

def section_lens_fixtures(p):
    """The same shapes ``validation/probe_verify_b11c/probe_lens.py`` drove:
    the two surface-sag builders over a coefficient grid, the grid-vs-aperture
    census and its warning, the recommendation, and the two fitting helpers.

    Re-driven here (not imported from there) so this campaign's bit-identity
    claim rests on this campaign's own probe.
    """
    h = np.linspace(-9e-3, 9e-3, 64)
    h_sq = h[:, None] ** 2 + h[None, :] ** 2
    coeff_sets = (
        {}, {4: 3.1e2}, {4: 3.1e2, 6: -8.4e6},
        {4: 3.1e2, 6: -8.4e6, 8: 1.7e11},
        {2: -1.0e-1, 4: 0.0, 10: 5.5e14},
    )
    for i, c in enumerate(coeff_sets):
        for j, (R, kk) in enumerate(((3.7e-2, -0.62), (-1.9e-2, 0.0),
                                     (np.inf, 0.0), (5.0e-3, -1.0))):
            p.call(f"L-sag-general-{i}-{j}", lenses.surface_sag_general,
                   h_sq, R, kk, c)
    xg = np.linspace(-9e-3, 9e-3, 48)
    X, Y = np.meshgrid(xg, xg, indexing='xy')
    for j, (Rx, Ry, kx, ky) in enumerate(((3.7e-2, 5.1e-2, -0.62, 0.0),
                                          (np.inf, 5.1e-2, 0.0, -1.0),
                                          (-2.0e-2, -2.0e-2, 0.0, 0.0))):
        p.call(f"L-sag-biconic-{j}", lenses.surface_sag_biconic,
               X, Y, Rx, Ry, kx, ky)
    presc = {
        'aperture_diameter': 1.1e-2,
        'surfaces': [
            {'radius': 3.7e-2, 'thickness': 4e-3, 'semi_diameter': 6e-3,
             'material': 'N-BK7'},
            {'radius': -2.0e-2, 'thickness': 2e-3, 'semi_diameter': 6e-3},
        ],
        'elements': [
            {'surf_num': 1, 'semi_diameter': 6e-3, 'comment': 'front'},
            {'surf_num': 2, 'semi_diameter': 9.5e-3, 'comment': 'rear'},
        ],
    }
    for N, dxp in ((256, 60e-6), (256, 20e-6), (1024, 10e-6)):
        for sf in (1.0, 0.95):
            p.call(f"L-grid-check-{N}-{dxp:.0e}-{sf}",
                   lenses.check_grid_vs_apertures, presc, N, dxp,
                   safety_factor=sf)
        p.call(f"L-grid-recommend-{N}-{dxp:.0e}",
               lenses.recommend_grid_for_prescription, presc, dxp)
        p.call(f"L-warn-aperture-{N}-{dxp:.0e}",
               lenses._warn_if_aperture_exceeds_grid, presc, N, dxp)
    p.call("L-collect-semis", lenses._collect_semi_diameters, presc)
    for n_vars in (1, 2, 3):
        for order in (0, 1, 3, 6):
            p.call(f"L-multi-idx-{n_vars}-{order}",
                   lenses._multi_indices_total_degree, n_vars, order)
    for j, v in enumerate((np.array([0.0, 0.0, 0.0]),
                           np.array([1.0, 1.0, 1.0]),
                           np.array([-1e-9, 1e-9]),
                           np.array([1.0, np.nan, 2.0]),
                           np.linspace(-3e-3, 7e-3, 17))):
        for pad in (0.0, 1e-12, 0.05):
            p.call(f"L-fit-norm-{j}-{pad}", lenses._fit_normaliser, v, pad)


def main():
    out_path = sys.argv[2]
    p = hlib.Probe()
    section_d3(p)
    section_option_a_counterfactual(p)
    section_lens_fixtures(p)
    p.write(out_path)
    print(f"[probe_lens_d3] {len(p.out)} keys, build={hlib.build_tag()}")


if __name__ == '__main__':
    main()
