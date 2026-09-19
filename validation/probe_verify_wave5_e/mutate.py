"""VERIFY-WAVE5-E: the mutation harness.

Applies ONE named mutation to the copy of the tree at ``--root`` (never to the
verification worktree), leaving every other file alone, and prints what it
changed.  ``--revert`` restores the file from the pristine tree.

Usage:
    python mutate.py --root C:/tmp/lum_ve_mut --pristine C:/tmp/lum_ve NAME
    python mutate.py --root C:/tmp/lum_ve_mut --pristine C:/tmp/lum_ve --revert
"""
import argparse
import os
import shutil
import sys

#: name -> (relative file, old snippet, new snippet)
MUTATIONS = {
    # ---- E5 / O-1 -----------------------------------------------------
    'e5_unfreeze': (
        'lumenairy/raytrace/differential.py',
        "    jac_v = xp.where(reached[jac_mask_shape], jac_v, jac)",
        "    jac_v = xp.where(reached[jac_mask_shape] | True, jac_v, jac)"),
    'e5_freeze_everything': (
        'lumenairy/raytrace/differential.py',
        "        dtype=bool)\n    jac_v = xp.where(reached[jac_mask_shape], jac_v, jac)",
        "        dtype=bool) & False\n    jac_v = xp.where(reached[jac_mask_shape], jac_v, jac)"),
    'e5_round1_alive': (
        'lumenairy/raytrace/differential.py',
        "reached_surface=np.asarray(balive[:n], bool))",
        "reached_surface=None)", 2),
    # ---- E5 / O-3 -----------------------------------------------------
    'e5_tol_drop_z': (
        'lumenairy/propagators/fga.py',
        "/ max(abs(float(z_image)), lam))",
        "/ lam)"),
    'e5_guard_off_at_coarse': (
        'lumenairy/propagators/fga.py',
        "    _require_non_immersed_exit(surfs, wavelength, z_image, "
        "'_fga_coarse')",
        "    pass  # _require_non_immersed_exit disabled (mutation)"),
    'e5_guard_off_at_through_lens': (
        'lumenairy/propagators/fga.py',
        "    _require_non_immersed_exit(surfs, wavelength, z_image,\n"
        "                               'apply_real_lens_fga')",
        "    pass  # _require_non_immersed_exit disabled (mutation)"),
    # The two arms VERIFY-WAVE5-E D9's fix needs: with a per-CALL-SITE
    # reachability pin in place, each site's deletion must redden its OWN id
    # and no other, so all four deletions are mutable here.
    'e5_guard_off_at_vector': (
        'lumenairy/propagators/fga.py',
        "    _require_non_immersed_exit(surfs, wavelength, z_image,\n"
        "                               'apply_real_lens_fga_vector')",
        "    pass  # _require_non_immersed_exit disabled (mutation)"),
    'e5_guard_off_at_caustic': (
        'lumenairy/propagators/fga.py',
        "    _require_non_immersed_exit(surfs, wavelength, 0.0, "
        "'_caustic_zone')",
        "    pass  # _require_non_immersed_exit disabled (mutation)"),
    # ---- E1 -----------------------------------------------------------
    'e1_unnamed_right_operand': (
        'lumenairy/propagators/asm.py',
        "            E_out = _ifft2(_fft2(E_in) * H).copy()",
        "            E_out = _ifft2(_fft2(E_in) * np.exp(np.log(H))).copy()"),
    'e1_privatise_dispatchers': (
        'lumenairy/propagators/fft_infra.py',
        "return (buf if (_nbufs > 1",
        "return (buf.copy() if (_nbufs > 1", 4),
    'e1_drop_scope_sentence': (
        'lumenairy/propagators/fft_infra.py',
        '    doc="pyFFTW two-buffer ping-pong on/off (True shipped).  Off '
        'halves the "\n        "resident workspace for one array copy per '
        'FFT.  The transform\'s "\n        "values are byte-identical either '
        'way; the object handed back is a "\n        "live workspace view in '
        'one mode and a private copy in the other, "\n        "which NumPy\'s '
        'temporary elision can distinguish.  Changing it "\n        "clears '
        'the plan cache.")',
        '    doc="pyFFTW two-buffer ping-pong on/off (True shipped).  Off '
        'halves the "\n        "resident workspace for one array copy per '
        'FFT; values are "\n        "byte-identical either way.  Changing it '
        'clears the plan cache.")'),
    # ---- E2 -----------------------------------------------------------
    'e2_bound_disabled': (
        'tests/unit/test_wave5_e_c8_default_order.py',
        "    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)",
        "    LT.REMAP_INVERSE_SUPPORT_BOUND = False   # mutation"),
    'e2_annuli_reversed': (
        'tests/unit/test_wave5_e_c8_default_order.py',
        "_FACTORS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)",
        "_FACTORS = (5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0)"),
    'e2_stimulus_dead': (
        'tests/unit/test_wave5_e_c8_default_order.py',
        "        if len(trips) >= 3:",
        "        if len(trips) >= 3 and False:"),
    'e2_control_weak': (
        'tests/unit/test_wave5_e_c8_default_order.py',
        "    Coff = _call(_GHOST, bound=False, cx=0.0, frbf=2.0, order=10)\n"
        "    Con = _call(_GHOST, bound=True, cx=0.0, frbf=2.0, order=10)",
        "    Coff = _call(_GHOST, bound=False, cx=0.0, frbf=2.0, order=10)\n"
        "    Con = Coff   # mutation: the control cannot see the bound"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?')
    ap.add_argument('--root', required=True)
    ap.add_argument('--pristine', required=True)
    ap.add_argument('--revert', action='store_true')
    a = ap.parse_args()
    if a.revert:
        for spec in MUTATIONS.values():
            rel = spec[0]
            src = os.path.join(a.pristine, rel)
            dst = os.path.join(a.root, rel)
            shutil.copyfile(src, dst)
        print('reverted', len({m[0] for m in MUTATIONS.values()}), 'files')
        return 0
    names = a.name.split(',')
    # ALWAYS start from pristine for EVERY file the matrix can touch -- not
    # only the ones this mutation names.  (Restoring just the named file left
    # the previous arm's edit in place and made two arms read the arm before
    # them; caught 2026-09-19 when three unrelated mutations all reported the
    # same 5 failures.)
    for spec in MUTATIONS.values():
        rel = spec[0]
        shutil.copyfile(os.path.join(a.pristine, rel),
                        os.path.join(a.root, rel))
    for nm in names:
        spec = MUTATIONS[nm]
        rel, old, new = spec[0], spec[1], spec[2]
        want = spec[3] if len(spec) > 3 else 1
        p = os.path.join(a.root, rel)
        src = open(p, encoding='utf-8').read()
        if src.count(old) != want:
            print('MUTATION %s: anchor found %d times (want %d) in %s -- '
                  'refusing' % (nm, src.count(old), want, rel))
            return 2
        open(p, 'w', encoding='utf-8').write(src.replace(old, new))
        print('applied %s to %s' % (nm, rel))
    return 0


sys.exit(main())
