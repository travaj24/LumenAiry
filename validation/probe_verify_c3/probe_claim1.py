"""VERIFY-WP-C3 CLAIM 1 -- the F1 finding, re-measured, with a SECOND
independent truth for the field 8 mm past the chain exit.

    python probe_claim1.py <tree> <out.json> [--refmax F]

The fixture is WP-B4's own ``_chain_fixture``, rebuilt here from the test
file's numbers (N = 256 at 60 um, two BK7 singlets 60/-60 mm x 3 mm at
14 mm aperture, gaps 20 and 10 mm, r_in = 60 mm, lambda = 1.31 um,
final_distance = 8 mm, readout dx_out = 0.5 um over N_out = 64).
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                     # noqa: E402
import vclib as V                                      # noqa: E402

V.anchor(_TREE)

import lumenairy.propagators.carrier as CA             # noqa: E402

REFMAX = 256
for a in sys.argv[3:]:
    if a.startswith('--refmax'):
        REFMAX = int(a.split('=')[1])

WL = 1.31e-6
FD = 8.0e-3
FR = dict(dx_out=0.5e-6, N_out=64)
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def singlet(R1, R2, d, glass, ap, name):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def fixture(n=256):
    dx = 60e-6 * 256 / n
    w, r_in = 4.5e-3, 60e-3
    g = V.axis(n, dx)
    env = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / w ** 2)
                 ).astype(np.complex128)
    p = singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    return env, dx, r_in, [{'prescription': p, 'gap_before': 20e-3},
                           {'prescription': p, 'gap_before': 10e-3}]


def chain(env, dx, r_in, groups, **kw):
    kw.setdefault('final_leg', 'paraxial')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = CA.propagate_traced_carrier_chain(
            env, groups, WL, dx, r_in=r_in, ray_subsample=16, n_workers=1,
            traced_kwargs=TKW, **kw)
    return r, [str(w.message) for w in rec]


def onaxis(A):
    A = np.asarray(A)
    n = A.shape[-1]
    return float(abs(A[A.shape[-2] // 2, n // 2]) ** 2)


def peak(A):
    """MAX of |A|^2 over the window, and where it sits.  Reported beside the
    centre sample everywhere because the two are NOT the same number on this
    fixture and the report's tables do not say which one they are."""
    A = np.abs(np.asarray(A)) ** 2
    j = int(np.argmax(A))
    iy, ix = divmod(j, A.shape[-1])
    return {'peak': float(A.max()),
            'argmax_iy': iy, 'argmax_ix': ix,
            'centre_iy': A.shape[-2] // 2, 'centre_ix': A.shape[-1] // 2}


def power(A, d):
    return float((np.abs(np.asarray(A)) ** 2).sum() * d * d)


out = {'fixture': {'wl': WL, 'final_distance': FD, 'focus_readout': FR}}

# ===========================================================================
# capture the EXACT exit state the chain's readout runs on
# ===========================================================================
_CAP = {}
_orig_k1 = CA._collins_readout_k1


def _spy(env, R, z, wavelength, dx, dy):
    val = _orig_k1(env, R, z, wavelength, dx, dy)
    _CAP.setdefault('calls', []).append(
        dict(env=np.array(env), R=R, z=z, wl=wavelength, dx=dx, dy=dy,
             k1=val))
    return val


CA._collins_readout_k1 = _spy

# ===========================================================================
# (a) the four readings of the claimed paragraph, N = 256
# ===========================================================================
env0, dx0, r_in, groups = fixture(256)

_CAP.clear()
res_col, w_col = chain(env0, dx0, r_in, groups, final_distance=FD,
                       focus_readout=dict(FR))
res_szi, w_szi = chain(env0, dx0, r_in, groups, final_distance=FD,
                       focus_readout=dict(FR), transport='sziklas')
cap = _CAP['calls'][0]
exit_env, exit_R, exit_dx, exit_dy = cap['env'], cap['R'], cap['dx'], cap['dy']

# the UN-RESOLVED one-step readout, reached directly
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    one = CA._collins_focus_readout(
        exit_env, exit_R, FD, WL, exit_dx, exit_dy,
        dx_out=FR['dx_out'], N_out=FR['N_out'], on_collins_sampling='warn')
w_one = [str(w.message) for w in rec]

out['a_readouts'] = {
    'chain_collins_default': {
        'on_axis': onaxis(res_col.field), 'dx': res_col.dx,
        'peak': peak(res_col.field),
        'power_window': power(res_col.field, FR['dx_out']),
        'route': {k: res_col.stages[-1].get(k) for k in
                  ('readout_route', 'readout_route_k1',
                   'readout_route_reason')},
        'n_warn': len(w_col), 'warnings': w_col},
    'chain_sziklas': {
        'on_axis': onaxis(res_szi.field), 'peak': peak(res_szi.field),
        'power_window': power(res_szi.field, FR['dx_out']),
        'route_keys_present': [k for k in
                               ('readout_route', 'readout_route_k1',
                                'readout_route_reason')
                               if k in res_szi.stages[-1]],
        'n_warn': len(w_szi)},
    'one_step_collins_UNRESOLVED': {
        'on_axis': onaxis(one), 'peak': peak(one),
        'power_window': power(one, FR['dx_out']),
        'n_warn': len(w_one), 'warnings': w_one},
    'bit_identical_collins_vs_sziklas': bool(
        np.array_equal(np.asarray(res_col.field), np.asarray(res_szi.field))),
}

# the FREE leg carrying the same 8 mm, both transports, no readout
free = {}
_ff = {}
for tr in ('collins', 'sziklas'):
    r, wr = chain(env0, dx0, r_in, groups, final_distance=FD, transport=tr)
    dxo = V.pitch2(r.dx)[0]
    _ff[tr] = (np.asarray(r.field), dxo, r)
    free[tr] = {'on_axis': onaxis(r.field), 'dx': dxo,
                'peak': peak(r.field),
                'power_grid': power(r.field, dxo), 'R': r.R,
                'n_warn': len(wr)}
free['bit_identical'] = bool(np.array_equal(_ff['collins'][0],
                                            _ff['sziklas'][0]))
out['a_free_leg'] = free

# and the exit plane itself (final_distance = 0, no readout)
r0, _ = chain(env0, dx0, r_in, groups, final_distance=0.0)
out['a_exit_plane'] = {'dx': V.pitch2(r0.dx)[0], 'R': r0.R,
                       'power_grid': power(r0.field, V.pitch2(r0.dx)[0]),
                       'on_axis': onaxis(r0.field),
                       'matches_spy_dx': bool(
                           abs(V.pitch2(r0.dx)[0] - exit_dx) < 1e-18),
                       'matches_spy_R': bool(abs(r0.R - exit_R) < 1e-18)}

# ===========================================================================
# (b) K1 by hand at N = 256 / 512 / 1024
# ===========================================================================
k1rows = []
exits = {}
for n in (256, 512, 1024):
    e, d, ri, gr = fixture(n)
    _CAP.clear()
    _r, _w = chain(e, d, ri, gr, final_distance=FD, focus_readout=dict(FR))
    c = _CAP['calls'][0]
    ee, RR, dd, dyy = c['env'], c['R'], c['dx'], c['dy']
    r_x, r_y, th_x, th_y = CA._collins_input_box(
        ee, dd, dyy, WL, CA._COLLINS_TAIL_FRAC)
    Ax, B, Cc, D = CA._collins_envelope_abcd(RR, FD, np.inf)
    mine_x = 2.0 * dd * (abs(Ax) * r_x / abs(B) + th_x) / WL
    mine_y = 2.0 * dyy * (abs(Ax) * r_y / abs(B) + th_y) / WL
    mine = max(mine_x, mine_y)
    lib = CA._collins_readout_k1(ee, RR, FD, WL, dd, dyy)
    k1rows.append({
        'N': n, 'exit_pitch_um': dd * 1e6, 'exit_R': RR,
        'exit_support_r_x_mm': r_x * 1e3, 'exit_support_r_y_mm': r_y * 1e3,
        'theta_x_mrad': th_x * 1e3, 'theta_y_mrad': th_y * 1e3,
        'theta_nyquist_mrad': WL / (2 * dd) * 1e3,
        'A': Ax, 'B': B,
        'k1_hand': mine, 'k1_lib': lib,
        'k1_hand_repr': repr(mine), 'k1_lib_repr': repr(lib),
        'identical_bits': bool(mine == lib),
        'stage_k1': _r.stages[-1].get('readout_route_k1'),
        'stage_route': _r.stages[-1].get('readout_route'),
        'edge_amp_rel': float(np.max(np.abs(ee[0, :])) / np.max(np.abs(ee))),
    })
    exits[n] = (ee, RR, dd, dyy)
out['b_k1'] = k1rows
fd0 = []
for n in (256, 512, 1024):
    e, d, ri, gr = fixture(n)
    rr, _ww = chain(e, d, ri, gr, final_distance=0.0)
    dd = V.pitch2(rr.dx)[0]
    ev = CA.carrier_referenced_envelope(np.asarray(rr.field), rr.R, WL, dd)
    fd0.append({'N': n, 'dx': dd, 'R': rr.R,
                'k1_at_z_8mm': _orig_k1(ev, rr.R, FD, WL, dd, dd)})
out['b_k1_from_final_distance_0'] = fd0
out['b_k1_ratios'] = [k1rows[0]['k1_lib'] / k1rows[1]['k1_lib'],
                      k1rows[1]['k1_lib'] / k1rows[2]['k1_lib']]

# ===========================================================================
# (c) MY OWN reference: upsampled separable Fresnel quadrature
# ===========================================================================
ref_rows = []
prev = None
F = 1
while F <= REFMAX:
    E = V.upsampled_fresnel_reference(
        exit_env, exit_R, WL, exit_dx, exit_dy, FD,
        FR['dx_out'], FR['N_out'], F)
    oa = onaxis(E)
    row = {'F': F, 'fine_pitch_um': exit_dx / F * 1e6,
           'readout_k1_at_fine': out['b_k1'][0]['k1_lib'] / F,
           'on_axis': oa, 'peak': peak(E),
           'power_window': power(E, FR['dx_out'])}
    if prev is not None:
        row['ratio_to_prev'] = oa / prev if prev else None
        row['rel_l2_to_prev'] = V.rel_l2(E, prev_E)
    ref_rows.append(row)
    prev, prev_E = oa, E
    F *= 2
out['c_reference_ladder'] = ref_rows
out['c_converged'] = ref_rows[-1]
out['c_vs_claims'] = {
    'claimed_true_1.2359': ref_rows[-1]['on_axis'] / 1.2359,
    'claimed_sziklas_1.0743': ref_rows[-1]['on_axis'] / 1.0743,
    'claimed_collins_9017.1': ref_rows[-1]['on_axis'] / 9017.1,
}

# also: MY reference vs the sziklas readout field, whole window
E_ref = prev_E
def _diag(A, ref):
    A = np.asarray(A)
    c = slice(ref.shape[-1] // 2 - 8, ref.shape[-1] // 2 + 8)
    return {'rel_l2': V.rel_l2(A, ref),
            'rel_l2_abs_only': V.rel_l2(np.abs(A), np.abs(ref)),
            'rel_l2_centre16': V.rel_l2(A[c, c], ref[c, c]),
            'centre_ratio_complex': complex(
                A[A.shape[-2] // 2, A.shape[-1] // 2]
                / ref[ref.shape[-2] // 2, ref.shape[-1] // 2]),
            'peak': peak(A)}


out['c_window_rel_l2'] = {
    'sziklas_readout': _diag(res_szi.field, E_ref),
    'one_step_collins': _diag(one, E_ref),
    'reference_peak': peak(E_ref),
}

# how paraxial is MY reference?  the leading dropped term of the exact
# kernel is exp(-i k rho^4 / 8 z^3); its FIRST-ORDER effect is three separable
# pieces, so it can be evaluated without leaving the two-pass form.
out['c_paraxial_check'] = V.quartic_correction_size(
    exit_env, exit_R, WL, exit_dx, exit_dy, FD,
    FR['dx_out'], FR['N_out'], F=REFMAX)

# the SAME physical reading from a DIFFERENT chain grid -- the reference must
# not depend on which lattice the chain exited on
cross = []
for n in (512, 1024):
    ee, RR, dd, dyy = exits[n]
    Fn = 64 if n == 512 else 32
    E = V.upsampled_fresnel_reference(ee, RR, WL, dd, dyy, FD,
                                      FR['dx_out'], FR['N_out'], Fn)
    E2 = V.upsampled_fresnel_reference(ee, RR, WL, dd, dyy, FD,
                                       FR['dx_out'], FR['N_out'], Fn * 2)
    cross.append({'chain_N': n, 'F': Fn, 'on_axis_F': onaxis(E),
                  'on_axis_2F': onaxis(E2), 'peak_2F': peak(E2),
                  'rel_l2_F_vs_2F': V.rel_l2(E, E2),
                  'rel_l2_vs_N256_reference': V.rel_l2(E2, E_ref),
                  'ratio_to_N256_on_axis': onaxis(E2) / onaxis(E_ref)})
out['c_cross_grid'] = cross

# what the difference between the Sziklas readout and MY reference IS
_d = np.asarray(res_szi.field) / E_ref
_ph = np.angle(_d)
_u = V.axis(FR['N_out'], FR['dx_out'])
_X, _Y = np.meshgrid(_u, _u)
_Ades = np.stack([np.ones(_X.size), _X.ravel(), _Y.ravel(),
                  (_X ** 2 + _Y ** 2).ravel()], axis=1)
_coef, *_ = np.linalg.lstsq(_Ades, _ph.ravel(), rcond=None)
out['c_sziklas_vs_reference'] = {
    'abs_ratio_mean': float(np.mean(np.abs(_d))),
    'abs_ratio_std': float(np.std(np.abs(_d))),
    'phase_rms_rad': float(np.sqrt(np.mean(_ph ** 2))),
    'phase_ptv_rad': float(_ph.max() - _ph.min()),
    'fit_piston_rad': float(_coef[0]),
    'fit_tilt_x_rad_per_m': float(_coef[1]),
    'fit_tilt_y_rad_per_m': float(_coef[2]),
    'fit_defocus_rad_per_m2': float(_coef[3]),
    'phase_rms_after_fit_rad': float(np.sqrt(np.mean(
        (_ph.ravel() - _Ades @ _coef) ** 2))),
    'implied_tilt_rad': float(_coef[1] * WL / (2 * np.pi)),
}

# the free leg's own CENTRE sample against MY reference's, complex -- the one
# point the two grids share (the readout window is smaller than one free-leg
# pixel, so this is all that can be compared directly)
_flf = _ff['sziklas'][0]
_fc = _flf[_flf.shape[-2] // 2, _flf.shape[-1] // 2]
_rc = E_ref[E_ref.shape[-2] // 2, E_ref.shape[-1] // 2]
_sc = np.asarray(res_szi.field)[32, 32]
out['c_centre_sample'] = {
    'free_leg': {'re': float(_fc.real), 'im': float(_fc.imag),
                 'abs': float(abs(_fc)), 'arg': float(np.angle(_fc))},
    'my_reference': {'re': float(_rc.real), 'im': float(_rc.imag),
                     'abs': float(abs(_rc)), 'arg': float(np.angle(_rc))},
    'sziklas_readout': {'re': float(_sc.real), 'im': float(_sc.imag),
                        'abs': float(abs(_sc)), 'arg': float(np.angle(_sc))},
    'free_over_reference': {'abs': float(abs(_fc / _rc)),
                            'arg_rad': float(np.angle(_fc / _rc))},
    'sziklas_over_reference': {'abs': float(abs(_sc / _rc)),
                               'arg_rad': float(np.angle(_sc / _rc))},
}

# ===========================================================================
# (c2) the geometry cross-check
# ===========================================================================
fl = free['sziklas']
out['c2_geometry'] = {
    'free_leg_grid_pitch_um': fl['dx'] * 1e6,
    'free_leg_power': fl['power_grid'],
    'free_leg_on_axis': fl['on_axis'],
    'free_leg_R': fl['R'],
    # a flat-top of the same power over a disc of radius r would read
    # P/(pi r^2); the beam's own on-axis for a Gaussian of 1/e radius w is
    # 2P/(pi w^2).
    'note': 'P/(pi w^2/2) with w the exit beam radius carried to the plane',
}
# measure the beam radius at the readout plane from the free leg itself
r_free = _ff['sziklas'][2]
A = np.abs(np.asarray(r_free.field)) ** 2
dxo = V.pitch2(r_free.dx)[0]
xs = V.axis(A.shape[-1], dxo)
Px = A.sum(axis=0)
tot = Px.sum()
m2 = float((Px * xs ** 2).sum() / tot)            # <x^2>
w_rms = 2.0 * float(np.sqrt(m2))                  # 1/e AMPLITUDE radius:
#   |E|^2 = exp(-2 r^2/w^2)  =>  <x^2> = w^2/4
P = power(r_free.field, dxo)
out['c2_geometry'].update({
    'w_from_second_moment_mm': w_rms * 1e3,
    'P_total': P,
    'gaussian_on_axis_pred': 2.0 * P / (np.pi * w_rms ** 2),
    'flat_top_on_axis_pred': P / (np.pi * w_rms ** 2),
    'measured_on_axis': onaxis(r_free.field),
})

CA._collins_readout_k1 = _orig_k1
V.write_json(sys.argv[2], out)

print('--- a ---')
for k, v in out['a_readouts'].items():
    print(' ', k, v if not isinstance(v, dict) else
          {kk: v[kk] for kk in v if kk != 'warnings'})
print('  free', {k: v for k, v in free.items()})
print('--- b ---')
for r in k1rows:
    print(' ', r['N'], 'pitch', round(r['exit_pitch_um'], 4),
          'r', round(r['exit_support_r_x_mm'], 4),
          'th_mrad', round(r['theta_x_mrad'], 4),
          'K1lib', r['k1_lib'], 'hand', r['k1_hand'], 'same', r['identical_bits'])
print('--- c ---')
for r in ref_rows:
    print(' ', r)
print(' vs claims', out['c_vs_claims'])
print(' window relL2', out['c_window_rel_l2'])
print('--- c2 ---', out['c2_geometry'])
