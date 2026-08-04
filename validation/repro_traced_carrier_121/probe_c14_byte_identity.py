# niche C14 UNIT-C extraction proof: pulling the three notions of the traced
# exit support into ONE object (``_TracedExitSupport``) changes NO returned bit.
#
# WHY THE REFERENCE IS A FILE AND NOT ``git show HEAD``.
# ``probe_c8_byte_identity.py`` and ``probe_c6_byte_identity.py`` build their
# shadow module from ``git show HEAD:lumenairy/elements/_lens_traced.py``.  That
# device answers "does the working tree still reproduce the last commit", which
# is NOT the question here: this branch carries a large body of uncommitted,
# already-verified C11/C12/C13 work, so HEAD is not the thing the extraction
# must reproduce.  The reference is the working tree as it stood IMMEDIATELY
# BEFORE the C14 edit, captured verbatim as
# ``_c14_pre_baseline_lens_traced.py`` (md5 c8e1a870221565832545144bb1baeb5d,
# 8827 lines) and shipped beside this probe so the comparison is re-runnable
# later without a commit to point at.
#
# THE BOTH-SIDES RULE (C8 audit S11.8).  A byte-identity probe that pins a flag
# on the LIVE side only goes stale the moment the default moves --
# ``probe_c6_byte_identity.py`` now prints ``array_equal=False`` on 17 of its 29
# arms for exactly that reason, while ``probe_c6_tilted_failbefore.py`` survived
# because it set the flag on the SHADOW as well.  Every flag this probe pins is
# therefore written to BOTH modules, from one table.
#
# THE ONE ASYMMETRY, AND WHY IT IS SOUND.  ``SUPPORT_BAND_CHECK`` and
# ``_SUPPORT_BAND_PEAK_RATIO_TOL`` do not exist in the baseline -- they are what
# C14 adds.  Part (c) runs the whole matrix with the band check at its SHIPPED
# 'warn' and again at its fail-before 'silent', and asserts the live field is
# identical in both: that is the direct evidence that the new check is
# reporting-only and cannot move a returned bit, which is the claim the fail-
# before rests on.
#
# COVERAGE NOTE.  Part (b) exists because ``probe_c8_byte_identity.py`` does not
# exercise ``inversion_method='fit'`` at all, and the fit path owns the THIRD
# hull -- the one C14 re-pointed at the shared builder and the shared
# signed-distance rule.  An extraction proof that skipped it would be proving
# the wrong two thirds.
#
# usage:  python probe_c14_byte_identity.py
#         PARTS=ab python probe_c14_byte_identity.py
import importlib.util
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy.elements._lens_traced as LT                   # noqa: E402

LAM = 1.31e-6
K0 = 2 * np.pi / LAM
_HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(_HERE, '_c14_pre_baseline_lens_traced.py')

# Flags pinned identically on BOTH modules for every arm.  Values are the
# shipped ones; the point is not the value but that neither side can drift.
_PIN_BOTH = (
    'REMAP_STATIONARY_PHASE_LAUNCH', 'REMAP_INVERSE_SUPPORT_BOUND',
    'REMAP_STATIONARY_PHASE_FIT_GUARD', 'TILTED_CARRIER_EXACT_EIKONAL',
    '_REMAP_RESID_EIKONAL_DEGREE', '_DECENTRE_GATE_PIXELS',
    '_DECENTRE_GATE_W_FRAC', '_FIT_DISC_OUTSIDE_WEIGHT_REL',
    '_DECENTRED_FIT_POLY_ORDER', 'RAY_DENSITY_HALO_CHECK',
    '_SUPPORT_BOUND_FEATHER_CELLS', 'DECENTRED_FIT_ARBITER',
    'DECENTRED_FIT_PREDICTOR', 'LSTSQ_CONDITIONING_STEPDOWN',
)


def load_baseline():
    """The pre-C14 module, imported INSIDE the live package so that everything
    it resolves by name (backend, prescriptions, raytrace) is the same code the
    live element uses.  Only ``_lens_traced`` differs between the two sides."""
    name = 'lumenairy.elements._lens_traced_c14base'
    spec = importlib.util.spec_from_file_location(name, BASELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    # Same reason probe_c6 rebinds it: the shadow defines its OWN TiltedCarrier
    # NamedTuple, so its isinstance() dispatch would reject the real one.
    mod.TiltedCarrier = LT.TiltedCarrier
    return mod


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _singlet(R1=3.1e-3, R2=-3.1e-3, d=1.0e-3, ap=1.2e-3):
    gb, ga = ['air', 'N-BK7'], ['N-BK7', 'air']
    return {'name': 's', 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': gb[0], 'glass_after': ga[0],
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': gb[1], 'glass_after': ga[1],
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _stimulus(n=256, dx=4.0e-6):
    w, rc, alpha = 200e-6, -0.02, 6.0
    x = (np.arange(n) - n // 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    S = np.sign(rc) * (np.sqrt(r2 + rc ** 2) - abs(rc))
    E = (np.exp(-r2 / w ** 2) * np.exp(1j * K0 * S)
         * np.exp(1j * alpha * (r2 / w ** 2) ** 2)).astype(np.complex128)
    base = dict(prescription=_singlet(), wavelength=LAM, dx=dx, carrier=rc,
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent')
    return E, base


def _pin(head, **over):
    """Write every pinned flag to BOTH modules; return the undo list."""
    undo = []
    for name in _PIN_BOTH:
        val = over.get(name, getattr(LT, name))
        for mod in (LT, head):
            undo.append((mod, name, getattr(mod, name)))
            setattr(mod, name, val)
    return undo


def _unpin(undo):
    for mod, name, val in reversed(undo):
        setattr(mod, name, val)


def _compare(head, cases, base, E, label):
    ok = True
    for lbl, kw in cases:
        k = dict(base)
        k.update(kw)
        a = np.asarray(LT.apply_real_lens_traced(E, **k))
        b = np.asarray(head.apply_real_lens_traced(E, **k))
        eq = bool(np.array_equal(a, b)) and a.dtype == b.dtype
        ok &= eq
        d = np.abs(a - b)
        pk = float(np.abs(b).max())
        print(f"  {lbl:46s} array_equal={eq}  max|dE| {float(d.max()):.3e}"
              f"  ({float(d.max()) / max(pk, 1e-300):.2e} of peak)")
    print(f"  -> {label}: {'ALL IDENTICAL' if ok else '*** DIFFERS ***'}")
    return ok


def part_a(head):
    """The C8 matrix: every preserve_input_phase x amplitude_model, rs 1 and 4,
    with the support bound ON and OFF (the taper path and the no-hull path)."""
    print('\n(a) synthetic matrix -- Newton inverse, bound on and off')
    E, base = _stimulus()
    cases = []
    for pip, amod in (('remap', 'ray_density'), (True, 'ray_density'),
                      (False, 'ray_density'), (True, 'screen'),
                      (False, 'screen')):
        for rs in (1, 4):
            cases.append((f"pip={pip!r} amp={amod} rs={rs}",
                          dict(preserve_input_phase=pip,
                               amplitude_model=amod, ray_subsample=rs)))
    cases.append(("remap lattice rs=4",
                  dict(preserve_input_phase='remap',
                       amplitude_model='ray_density',
                       remap_sampling='lattice', ray_subsample=4)))
    cases.append(("no carrier, remap rs=4",
                  dict(preserve_input_phase='remap', carrier=None,
                       amplitude_model='ray_density', ray_subsample=4)))
    ok = True
    for bound in (True, False):
        undo = _pin(head, REMAP_INVERSE_SUPPORT_BOUND=bound)
        LT.SUPPORT_BAND_CHECK = 'silent'      # fail-before on the live side
        try:
            for _ in range(2):                # W9 warm-up, BOTH sides
                LT.apply_real_lens_traced(E, **dict(base, ray_subsample=4))
                head.apply_real_lens_traced(E, **dict(base, ray_subsample=4))
            ok &= _compare(head, cases, base, E,
                           f'C8 bound={bound}')
        finally:
            _unpin(undo)
            LT.SUPPORT_BAND_CHECK = 'warn'
    return ok


def part_b(head):
    """The THIRD hull: inversion_method='fit'.  Not covered by probe_c8."""
    print("\n(b) direct-fit inverse -- the third hull "
          "(inversion_method='fit')")
    E, base = _stimulus()
    # The reachable population is small and fixed by two exclusions the
    # element enforces: ``inversion_method='fit'`` forbids
    # ``amplitude_model='ray_density'`` (det J comes from the Newton fits),
    # and ``preserve_input_phase='remap'`` REQUIRES it (it reuses the
    # ray-density entrance pullback).  So the third hull is only ever reached
    # at amplitude_model='screen' with preserve_input_phase in (True, False) --
    # that is the whole of it, enumerated here.
    cases = []
    for pip in (True, False):
        for rs in (1, 2, 4):
            cases.append((f"FIT pip={pip!r} screen rs={rs}",
                          dict(preserve_input_phase=pip,
                               inversion_method='fit',
                               amplitude_model='screen',
                               ray_subsample=rs)))
    cases.append(("FIT no carrier rs=4",
                  dict(preserve_input_phase=False, carrier=None,
                       inversion_method='fit', amplitude_model='screen',
                       ray_subsample=4)))
    cases.append(("FIT decentred beam rs=4",
                  dict(preserve_input_phase=True, inversion_method='fit',
                       amplitude_model='screen', ray_subsample=4,
                       beam_centre=(60e-6, -40e-6))))
    undo = _pin(head)
    LT.SUPPORT_BAND_CHECK = 'silent'
    try:
        ok = _compare(head, cases, base, E, 'direct fit')
    finally:
        _unpin(undo)
        LT.SUPPORT_BAND_CHECK = 'warn'
    return ok


def part_c(head):
    """The new check is REPORTING-ONLY: the live field is identical with
    SUPPORT_BAND_CHECK at its shipped 'warn' and at its fail-before 'silent',
    and identical to the baseline in both."""
    print('\n(c) SUPPORT_BAND_CHECK is field-neutral (shipped vs fail-before)')
    E, base = _stimulus()
    cases = [(f"pip={pip!r} rs={rs}",
              dict(preserve_input_phase=pip, amplitude_model='ray_density',
                   ray_subsample=rs))
             for pip in ('remap', True) for rs in (1, 4)]
    undo = _pin(head)
    ok = True
    try:
        for lbl, kw in cases:
            k = dict(base)
            k.update(kw)
            LT.SUPPORT_BAND_CHECK = 'warn'
            a_on = np.asarray(LT.apply_real_lens_traced(E, **k))
            LT.SUPPORT_BAND_CHECK = 'silent'
            a_off = np.asarray(LT.apply_real_lens_traced(E, **k))
            b = np.asarray(head.apply_real_lens_traced(E, **k))
            e1 = bool(np.array_equal(a_on, a_off))
            e2 = bool(np.array_equal(a_on, b))
            ok &= e1 and e2
            print(f"  {lbl:46s} warn==silent {e1}   warn==baseline {e2}")
    finally:
        _unpin(undo)
        LT.SUPPORT_BAND_CHECK = 'warn'
    print(f"  -> band check field-neutral: "
          f"{'YES' if ok else '*** NO ***'}")
    return ok


def main():
    parts = os.environ.get('PARTS', 'abc')
    if not os.path.exists(BASELINE):
        raise SystemExit(f'missing reference: {BASELINE}')
    head = load_baseline()
    print('live     :', LT.__file__)
    print('baseline :', BASELINE)
    print('baseline has SUPPORT_BAND_CHECK:',
          hasattr(head, 'SUPPORT_BAND_CHECK'), '(expected False)')
    print('baseline has _TracedExitSupport:',
          hasattr(head, '_TracedExitSupport'), '(expected False)')
    ok = True
    if 'a' in parts:
        ok &= part_a(head)
    if 'b' in parts:
        ok &= part_b(head)
    if 'c' in parts:
        ok &= part_c(head)
    print('\n' + '=' * 62)
    print('C14 UNIT-C EXTRACTION:',
          'BYTE-IDENTICAL' if ok else '*** NOT IDENTICAL ***')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
