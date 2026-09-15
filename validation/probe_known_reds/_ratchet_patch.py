"""Extend the b11 warning-attribution ratchet to the swept propagator chain.

Adds, to ``tests/unit/test_audit2609_b11_hygiene.py``:

* a shared literal-stacklevel scanner (one implementation for both ratchets),
* a second ratchet over the three chain modules swept 2026-09-14,
* the two-caller attribution test for ``carrier.py``: the same warn site
  reached at two different LIBRARY depths must name the caller in both arms.
"""
import ast
import io
import sys

P = 'tests/unit/test_audit2609_b11_hygiene.py'

OLD_RATCHET = '''    def test_no_literal_stacklevel_is_left_in_the_swept_lens_bodies(self):
        """The ratchet.  A literal that creeps back in is right for one call
        path and wrong for the others, and the failure is silent -- the
        warning still fires, it just points at the wrong file.  The tuple
        is every lens body whose warnings were swept onto the helper: the
        analytic and traced bodies, the Maslov (11 sites), GBD (1),
        multibranch (4), thin (2), inverse-map (1) and uniform-fold (2)
        modules -- the whole lens family.  ``propagators/carrier.py``'s
        chain is outside it, recorded in WP-B11 section 4b."""
        for rel in ('lumenairy/elements/_lens_real.py',
                    'lumenairy/elements/_lens_traced.py',
                    'lumenairy/elements/lenses_maslov.py',
                    'lumenairy/elements/lenses_gbd.py',
                    'lumenairy/elements/_lens_traced_multibranch.py',
                    'lumenairy/elements/_lens_thin.py',
                    'lumenairy/elements/_lens_imap.py',
                    'lumenairy/elements/_lens_traced_uniform.py'):
            src = (REPO / rel).read_text(encoding='utf-8')
            tree = ast.parse(src)
            bad = []
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'warn'):
                    continue
                args = list(node.args) + [k.value for k in node.keywords
                                          if k.arg == 'stacklevel']
                for a in args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, int):
                        bad.append(node.lineno)
            assert not bad, (
                f'{rel}: warnings.warn with a LITERAL stacklevel at lines '
                f'{sorted(set(bad))}.  Use _caller_stacklevel(), which walks '
                f'out to the first frame outside the package.')
'''

NEW_RATCHET = '''    @staticmethod
    def _literal_stacklevel_lines(rel):
        """Lines in ``rel`` where a ``warnings.warn`` carries a LITERAL
        ``stacklevel``.  One implementation, used by both ratchets below."""
        tree = ast.parse((REPO / rel).read_text(encoding='utf-8'))
        bad = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'warn'):
                continue
            args = list(node.args) + [k.value for k in node.keywords
                                      if k.arg == 'stacklevel']
            for a in args:
                if isinstance(a, ast.Constant) and isinstance(a.value, int):
                    bad.append(node.lineno)
        return sorted(set(bad))

    def test_no_literal_stacklevel_is_left_in_the_swept_lens_bodies(self):
        """The ratchet.  A literal that creeps back in is right for one call
        path and wrong for the others, and the failure is silent -- the
        warning still fires, it just points at the wrong file.  The tuple
        is every lens body whose warnings were swept onto the helper: the
        analytic and traced bodies, the Maslov (11 sites), GBD (1),
        multibranch (4), thin (2), inverse-map (1) and uniform-fold (2)
        modules -- the whole lens family.  The carrier / system chain is
        covered by the sibling ratchet below (swept 2026-09-14)."""
        for rel in ('lumenairy/elements/_lens_real.py',
                    'lumenairy/elements/_lens_traced.py',
                    'lumenairy/elements/lenses_maslov.py',
                    'lumenairy/elements/lenses_gbd.py',
                    'lumenairy/elements/_lens_traced_multibranch.py',
                    'lumenairy/elements/_lens_thin.py',
                    'lumenairy/elements/_lens_imap.py',
                    'lumenairy/elements/_lens_traced_uniform.py'):
            bad = self._literal_stacklevel_lines(rel)
            assert not bad, (
                f'{rel}: warnings.warn with a LITERAL stacklevel at lines '
                f'{bad}.  Use _caller_stacklevel(), which walks '
                f'out to the first frame outside the package.')

    def test_no_literal_stacklevel_is_left_in_the_swept_chain_bodies(self):
        """The same ratchet over the propagator CHAIN modules, swept
        2026-09-14 (handoff 4.4).

        Why these three and not every propagator.  The rationale is that a
        literal encodes ONE call depth, so it is wrong wherever a warn site is
        reachable at more than one.  These three are where that is not a
        possibility but the DOCUMENTED shape: ``carrier.py``'s public entry
        points call each other (``carrier_referenced_focus_readout`` ->
        ``propagate_carrier_referenced``, and the traced chain calls the same
        entry point twice more), ``system.py``'s chain runs its legs through
        the same helpers from ``propagate_through_system`` and from
        ``evaluate``, and ``carrier_field.py`` reports through
        ``carrier._guard_dispose`` from its own entry points.

        MEASURED before the sweep, with the instrument in
        ``validation/probe_known_reds/probe_carrier_attribution.py``: the tilt-
        inert notice (``carrier.py`` line 747 as it then was, ``stacklevel=3``)
        named the CALLER when ``propagate_carrier_referenced`` was called
        directly and named ``carrier.py`` itself -- library source -- when the
        identical warn site was reached through
        ``carrier_referenced_focus_readout``.  2 of 4 emissions misattributed;
        0 of 4 after.  ``test_the_same_warn_site_names_the_caller_at_two_depths``
        below is that measurement as a test.

        The remaining propagator modules carry literals too (measured census:
        ``validation/probe_known_reds/stacklevel_census_base.json``, and the
        reachability screen in ``stacklevel_reach_base.json`` flags 16 more
        files).  They are NOT swept here: each needs its own two-caller
        measurement first, which is not a mechanical change.  That is recorded
        as open work rather than done silently."""
        for rel in ('lumenairy/propagators/carrier.py',
                    'lumenairy/propagators/system.py',
                    'lumenairy/propagators/carrier_field.py'):
            bad = self._literal_stacklevel_lines(rel)
            assert not bad, (
                f'{rel}: warnings.warn with a LITERAL stacklevel at lines '
                f'{bad}.  Use _caller_stacklevel(), which walks '
                f'out to the first frame outside the package.')

    def test_the_same_warn_site_names_the_caller_at_two_depths(self):
        """THE TWO-CALLER FIXTURE, and the reason the chain ratchet exists.

        One warn site -- the ``gap_kernel='fresnel'`` tilt-inert notice inside
        ``propagate_carrier_referenced`` -- reached two ways:

          A. ``propagate_carrier_referenced`` called from this file;
          B. ``carrier_referenced_focus_readout`` called from this file, which
             forwards ``gap_kernel`` and ``tilt`` into the SAME entry point,
             one library frame deeper.

        A correct attribution names THIS FILE in both arms.  A literal tuned
        for A names ``carrier.py`` in B, which is exactly what it did before
        the sweep: the notice pointed at the library's own call line, telling
        the reader where the library called itself."""
        import numpy as _np
        from lumenairy.propagators import carrier as CA
        me = pathlib.Path(__file__).name
        n, dx = 64, 8e-6
        ax = (_np.arange(n) - n // 2) * dx
        X, Y = _np.meshgrid(ax, ax)
        env = _np.exp(-(X ** 2 + Y ** 2) / (60e-6 ** 2)).astype(_np.complex128)
        base = dict(wavelength=633e-9, dx=dx, gap_kernel='fresnel',
                    tilt=(0.12, 0.0))

        arms = {
            'direct': (CA.propagate_carrier_referenced, (env, -0.05, 5e-3),
                       dict(base)),
            'one library frame deeper': (
                CA.carrier_referenced_focus_readout, (env, -0.05, 5e-3),
                dict(base, dx_out=dx, N_out=32)),
        }
        for label, (fn, a, kw) in arms.items():
            caught = self._warned(fn, *a, **kw)
            got = [w for w in caught if 'INERT on this call' in str(w.message)]
            assert got, f'{label}: the tilt-inert notice did not fire'
            for w in got:
                assert pathlib.Path(w.filename).name == me, (
                    f'{label}: the notice names {w.filename}:{w.lineno}, not '
                    f'the caller.  A literal stacklevel is right for one call '
                    f'depth; this site is reached at two.')
'''


def main():
    with io.open(P, encoding='cp1252', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    if src.count(OLD_RATCHET) != 1:
        print('no unique match for the ratchet body:', src.count(OLD_RATCHET))
        return 1
    src = src.replace(OLD_RATCHET, NEW_RATCHET)
    ast.parse(src)
    with io.open(P, 'w', encoding='cp1252', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('patched', P)
    return 0


if __name__ == '__main__':
    sys.exit(main())
