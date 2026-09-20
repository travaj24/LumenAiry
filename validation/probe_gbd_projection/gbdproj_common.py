"""WP-B12b -- shared fixtures for the GBD exit-vertex-projection package.

The diffraction oracle, the Sellmeier table, the exact meridional trace and the
brute-force Rayleigh-Sommerfeld-I sum are NOT re-implemented here: they are
imported from ``validation/probe_wp_b12/b12_common.py``, which is the audit's
one independent oracle for this reference-plane question (the house rule --
one implementation of a numerical kernel -- applies to the probes too).  What
this module adds is the fixture SET this package needs and the WP-B12 oracle
did not cover:

* an EVEN-ASPHERIC (A4 / A6) last surface, rotationally symmetric, so the
  imported oracle can still represent it exactly (``trace_meridional`` already
  takes an ``asph`` list; only ``Fixture.oracle_surfaces`` hard-coded ``None``);
* a FLAT-BASE, purely-aspheric last surface, where the deleted in-line copy's
  own ``np.isfinite(radius)`` guard made its sag EXACTLY zero -- so the
  pre-repair field is reconstructible through a supported public keyword, with
  no copy of the deleted code;
* a BICONIC, a FREEFORM (XY polynomial) and a FIELD-FRAME decentered last
  surface -- none of which is rotationally symmetric, so they carry NO
  diffraction oracle and are measured against the library's own
  ``TraceResult.at_exit_vertex`` instead (model-free, and the decisive reading
  for the sag itself);
* a MIRROR-terminated prescription, for the propagation-direction sign;
* the flat-last-surface CONTROL, where the projection is the identity and
  every field must be byte-identical.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and PYTHONPATH
pinned at the tree under test.

Author: WP-B12b
"""
from __future__ import annotations

import copy
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_B12 = os.path.join(os.path.dirname(_HERE), 'probe_wp_b12')
if _B12 not in sys.path:
    sys.path.insert(0, _B12)

from b12_common import (  # noqa: E402
    Fixture,
    assert_tree,
    build_tag,
    dump,
    fidelity,
    n_sellmeier,
    rel_l2,
    trace_meridional,
)

__all__ = ['Fixture', 'GFixture', 'FIXTURES', 'MECHANISM_ONLY', 'NONSYM',
           'assert_tree', 'build_tag', 'dump', 'fidelity', 'rel_l2',
           'n_sellmeier', 'trace_meridional', 'inline_conic_sag',
           'gbd_surfaces', 'nonsym_grid', 'nonsym_beam']

_LAM = 1.064e-6


# ---------------------------------------------------------------------------
# The in-line conic-sag copy this package DELETES, kept here so the defect can
# be measured after the library stops carrying it.  Transcribed from
# ``lumenairy/propagators/gbd.py`` at commit 1218b24f (v5.22 .. 5.47.0):
#
#     _Rl = float(getattr(surfs[-1], 'radius', np.inf))
#     _kl = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
#     if np.isfinite(_Rl) and _Rl != 0.0:
#         _cl = 1.0 / _Rl
#         _r2 = dt.x ** 2 + dt.y ** 2
#         _sag = _cl * _r2 / (1.0 + np.sqrt(np.maximum(
#             1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
#     else:
#         _sag = np.zeros_like(dt.x)
#     t = (z_image - _sag) / Nz2
#
# This is a PROBE-side reproduction for measurement only; it is not, and must
# not become, a second live implementation.
# ---------------------------------------------------------------------------
def inline_conic_sag(surface, x, y):
    """The deleted in-line sag, evaluated at the last-surface state."""
    _Rl = float(getattr(surface, 'radius', np.inf))
    _kl = float(getattr(surface, 'conic', 0.0) or 0.0)
    if np.isfinite(_Rl) and _Rl != 0.0:
        _cl = 1.0 / _Rl
        _r2 = np.asarray(x) ** 2 + np.asarray(y) ** 2
        return _cl * _r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
    return np.zeros_like(np.asarray(x))


def gbd_surfaces(prescription):
    """The surface list ``apply_prescription_persurface_to_beamlets`` traces:
    ``surfaces_from_prescription`` with the LAST surface's thickness zeroed."""
    from lumenairy.raytrace import surfaces_from_prescription
    surfs = list(surfaces_from_prescription(prescription))
    surfs[-1] = copy.copy(surfs[-1])
    surfs[-1].thickness = 0.0
    return surfs


# ---------------------------------------------------------------------------
# Rotationally-symmetric fixtures -- the imported oracle CAN score these.
# ---------------------------------------------------------------------------
class GFixture(Fixture):
    """``b12_common.Fixture`` plus even-aspheric coefficients per surface.

    ``asph1`` / ``asph2`` are ``{power: coefficient}`` dicts in the library's
    own convention (``sum_p a_p r^p``, even powers).  Both the prescription
    and the oracle read the same dict, so the two still share no code.
    """

    def __init__(self, *a, asph1=None, asph2=None, **kw):
        super().__init__(*a, **kw)
        self.asph1 = dict(asph1 or {})
        self.asph2 = dict(asph2 or {})

    def prescription(self):
        p = super().prescription()
        if not (self.asph1 or self.asph2):
            return p
        s = [dict(p['surfaces'][0]), dict(p['surfaces'][1])]
        if self.asph1:
            s[0]['aspheric_coeffs'] = dict(self.asph1)
        if self.asph2:
            s[1]['aspheric_coeffs'] = dict(self.asph2)
        return {**p, 'surfaces': s}

    def oracle_surfaces(self):
        ng = self.index()
        return [dict(zv=0.0, radius=self.R1, conic=self.conic1,
                     asph=sorted(self.asph1.items()), n_after=ng),
                dict(zv=self.t, radius=self.R2, conic=self.conic2,
                     asph=sorted(self.asph2.items()), n_after=1.0)]


# The ladder fixtures.  ONE optic geometry (N-BAF10 biconvex, R1 = +11.0 mm,
# t = 0.9 mm, semi = 0.30 mm, 1.064 um, w0 = 0.18 mm) with the LAST surface
# varied along a single axis -- conic base, conic constant, a mild even
# asphere, a strong one, a flat-base purely-aspheric one -- so every difference
# between two rows is the last surface and nothing else.  This is the SAME
# optic ``tests/unit/test_audit2609_b12b_gbd_projection.py`` pins, so the
# numbers below and the numbers the test suite asserts are readings of one
# fixture set.
#
# The optic is deliberately SLOW (NA 0.015 .. 0.034): on the 128 x 5.0 um grid
# the aperture fits (0.640 mm span against a 0.600 mm clear aperture) AND the
# Airy radius is 17 .. 44 um, i.e. three to nine pixels, so the focal
# structure is resolved and a fidelity against the diffraction oracle means
# something.  WP-B12 section 6.3 is the counter-example: on a grid that cannot
# resolve the focus every model returns the same smoothed blob and the
# comparison ranks nothing.
_LADDER = dict(glass='N-BAF10', R1=11.0e-3, R2=-11.0e-3, t=0.9e-3,
               semi=0.30e-3, lam=1.064e-6, w0=0.18e-3, N=128, dx=5.0e-6)


def _ladder(key, note, **kw):
    d = dict(_LADDER)
    d.update({k: v for k, v in kw.items() if k in d})
    extra = {k: v for k, v in kw.items() if k not in d}
    return GFixture(key, d['glass'], d['R1'], d['R2'], d['t'], d['semi'],
                    d['lam'], d['w0'], d['N'], d['dx'], note=note, **extra)


FIXTURES = {
    # --- the two CONTROLS the in-line copy got right ---
    'conic_ref': _ladder(
        'conic_ref',
        'the reference optic, CONIC last surface -- the in-line copy was '
        'exact here, so the field may move only by the Jacobian projection '
        'and floating-point reassociation'),
    'conic_k_last': _ladder(
        'conic_k_last',
        'the same optic with conic k = -0.60 on the last surface: the '
        'in-line copy carried k, so this is a second exactness control',
        conic2=-0.60),
    # --- the defect cases ---
    'asphere_mild': _ladder(
        'asphere_mild',
        'the same optic + A4 = 1.5e8 on the LAST surface (a mild departure)',
        asph2={4: 1.5e8}),
    'asphere_field': _ladder(
        'asphere_field',
        'the same optic + A4 = 5.0e8, A6 = -4.0e15 on the LAST surface -- '
        'the headline CURVED-base aspheric fixture',
        asph2={4: 5.0e8, 6: -4.0e15}),
    # The FLAT-BASE, purely-aspheric last surface.  The deleted in-line copy's
    # own guard was ``if np.isfinite(_Rl) and _Rl != 0.0``, so on an infinite
    # base radius it read the sag as EXACTLY zero and folded nothing into the
    # leg -- which is bit-for-bit what the modern primitive produces under
    # ``reference='surface'``.  That makes this fixture the one place where the
    # pre-WP-B12b field can be reconstructed EXACTLY, with no copy of the
    # deleted code, by asking the shipped primitive for the other reference
    # plane.  Both the archive arm and the forced arm are taken on it, and
    # they must agree byte for byte.
    'asphere_flat_base': _ladder(
        'asphere_flat_base',
        'FLAT base radius on the last surface with the power carried by '
        'A2 = -9.0e1 / A4 = 4.0e8 -- the in-line copy read its sag as exactly '
        'zero, so forcing reference=surface reproduces the pre-WP-B12b field '
        'bit for bit',
        R2=float('inf'), asph2={2: -9.0e1, 4: 4.0e8}),
    # --- the flat-last-surface CONTROL: the projection short-circuits ---
    'flat_last_lasf9': GFixture(
        'flat_last_lasf9', 'N-LASF9', 1.45e-3, float('inf'), 0.55e-3,
        0.262e-3, 0.850e-6, 190e-6, 128, 4.3e-6,
        note='N-LASF9 plano-convex, curved side FIRST -- the '
             'flat-last-surface control (every field must be byte-identical)'),
}

# Mechanism-only fixtures: WP-B12 probe D's own optic, so section 5.1's
# "15.52 waves / 71 % of the sag" reading is reproduced like for like.  Its
# aspheric arm bends the marginal ray to sec = 1.5 (NA ~ 0.75), which no
# beamlet sum can represent, so it is measured at the RAY level only.
MECHANISM_ONLY = {
    'b12d_conic': GFixture(
        'b12d_conic', 'N-BAF10', 2.10e-3, -2.10e-3, 0.70e-3, 0.20e-3,
        1.064e-6, 105e-6, 256, 1.9e-6,
        note="WP-B12 probe D's optic, conic last surface"),
    'b12d_asphere': GFixture(
        'b12d_asphere', 'N-BAF10', 2.10e-3, -2.10e-3, 0.70e-3, 0.20e-3,
        1.064e-6, 105e-6, 256, 1.9e-6, asph2={4: 4.0e8, 6: -8.0e17},
        note="WP-B12 probe D's A4 / A6 arm (section 5.1's 15.52 waves)"),
}


# ---------------------------------------------------------------------------
# Non-rotationally-symmetric fixtures.  No diffraction oracle exists for these
# in this package (the imported RS sum assumes rotational symmetry), so they
# are measured against the library's own ``TraceResult.at_exit_vertex`` and
# reported as field MOVEMENT, never as field accuracy.
# ---------------------------------------------------------------------------
def _singlet_last(last_extra, *, R1=11.0e-3, R2=-11.0e-3, t=0.9e-3,
                  semi=0.30e-3, glass='N-BAF10', name='g'):
    s0 = {'radius': R1, 'conic': 0.0, 'thickness': t, 'glass_before': 'air',
          'glass_after': glass, 'semi_diameter': semi}
    s1 = {'radius': R2, 'conic': 0.0, 'thickness': 0.0,
          'glass_before': glass, 'glass_after': 'air', 'semi_diameter': semi}
    s1.update(last_extra or {})
    return {'name': name, 'aperture_diameter': 2.0 * semi,
            'surfaces': [s0, s1], 'thicknesses': [t], 'stop_index': 0}


def _mirror_presc(R=-20.0e-3, semi=0.60e-3):
    return {'name': 'mirror', 'aperture_diameter': 2.0 * semi,
            'surfaces': [{'radius': R, 'conic': 0.0, 'thickness': 0.0,
                          'glass_before': 'air', 'glass_after': 'MIRROR',
                          'semi_diameter': semi}],
            'thicknesses': [0.0], 'stop_index': 0}


NONSYM = {
    'biconic': dict(
        presc=lambda: _singlet_last({'radius_y': -7.0e-3}),
        lam=_LAM, semi=0.30e-3, w0=0.18e-3, N=128, dx=5.0e-6,
        note='biconic last surface (radius_y = -7.0 mm): the in-line copy '
             'evaluated the x-branch radius for both axes'),
    'freeform': dict(
        presc=lambda: _singlet_last({
            'freeform_type': 'xy_polynomial',
            'xy_coeffs': {(2, 0): 4.0e1, (0, 2): -2.4e1, (4, 0): 8.0e7},
            'norm_x': 1.0, 'norm_y': 1.0}),
        lam=_LAM, semi=0.30e-3, w0=0.18e-3, N=128, dx=5.0e-6,
        note='freeform (XY-polynomial) last surface: the in-line copy dropped '
             'the whole departure'),
    'field_frame': dict(
        presc=lambda: _singlet_last({'decenter': (6.0e-5, -4.0e-5)}),
        lam=_LAM, semi=0.30e-3, w0=0.18e-3, N=128, dx=5.0e-6,
        note='field-frame decentered last surface (60 um, -40 um): the '
             'in-line copy evaluated the sag at the undecentered coordinates'),
    'mirror': dict(
        presc=_mirror_presc,
        lam=_LAM, semi=0.60e-3, w0=0.36e-3, N=128, dx=10.0e-6,
        note='concave MIRROR last surface (R = -20.0 mm): the in-line copy '
             'hard-assumed a forward-propagating exit ray, so its sag '
             'correction carried the wrong SIGN and doubled the error'),
}


def nonsym_grid(spec):
    N, dx = spec['N'], spec['dx']
    x1 = (np.arange(N) - N / 2) * dx
    return np.meshgrid(x1, x1)


def nonsym_beam(spec):
    X, Y = nonsym_grid(spec)
    return np.exp(-(X ** 2 + Y ** 2) / spec['w0'] ** 2).astype(np.complex128)
