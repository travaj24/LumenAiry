# Approximation audit -- END-TO-END differential ablation of design 121.
#
# Every row runs the COMPLETE shipped path (chain A cached -> TiltedCarrier
# (order) -> the six post-DOE groups -> the 7.7058 mm trailing leg -> the EXACT
# Bluestein readout) on a FIXED output lattice centred on the order's exact
# chief ray, and differs from the baseline in exactly ONE construction.
#
# Discipline (the artefact lists in SCOPE_TILTED_COARSE_LEG_TRANSPORT S8,
# DIAG_LAST_GROUP_DECENTRE S8 and ABLATE_LAST_GROUP S6 are why):
#   * NO hand-off plane.  ABLATE S6 showed a mid-leg hand-off is not a valid
#     instrument; here nothing is handed off, one construction is swapped
#     inside a complete run.
#   * The differential floor is established by a NULL run (an identity patch)
#     and measures 0.000e+00 relative L2 -- the instrument is bit-exact, so
#     any non-zero relL2 is a real change of the delivered field.
#   * A POSITIVE CONTROL (TILTED_CARRIER_EXACT_EIKONAL=False, the niche-C5
#     defect that was just fixed and is known to cost ~20 EE3 points) is run
#     in the same session, so a table of nulls cannot be a dead instrument.
#   * Sampling adequacy is printed per run from the chain's own
#     na_exit_measured / na_grid_nyquist / exit_power_above_nyquist.
#
# Env knobs: ORD (default -4,-2), SET (which variant batch), RN/RS/NW/NFC/WF/
# NOUT/DXO (see approx_common).
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import approx_common as A                                      # noqa: E402

CM = A.CM
LT = A.LT
LAM = A.LAM


# --------------------------------------------------------------------------
# exact-alternative substitutions, installed at RUNTIME only
# --------------------------------------------------------------------------

def p_rout_scale(f):
    """Perturb the paraxial-Moebius exit carrier radius by a factor.

    A REFERENCE quantity is one the envelope absorbs exactly: perturbing it
    should move the delivered field by ~0.  The sensitivity measured here is
    then multiplied by the ACTUAL paraxial error (approx_rout_exact_121.py) to
    price the approximation."""
    real = CM._paraxial_group_r_out

    def q(presc, R_in, wavelength):
        return real(presc, R_in, wavelength) * f
    return [(CM, '_paraxial_group_r_out', q)]


def p_sphere_taper_off():
    """`_sphere_parab_conversion` with T == 1 (the whole-grid swap)."""
    def q(shape, dx, wavelength, R, sign, w_beam=None, centre=(0.0, 0.0)):
        if not np.isfinite(R) or R == 0.0:
            return None
        n, ny = int(shape[-1]), int(shape[-2])
        x = (np.arange(n, dtype=np.float64) - n / 2) * dx - float(centre[0])
        y = (np.arange(ny, dtype=np.float64) - ny / 2) * dx - float(centre[1])
        r2 = x[None, :] ** 2 + y[:, None] ** 2
        k = 2.0 * np.pi / wavelength
        diff = CM._exact_sphere_eikonal((ny, n), dx, dx, wavelength, R,
                                        centre=centre) - r2 / (2.0 * R)
        return np.exp(sign * 1j * k * diff)
    return [(CM, '_sphere_parab_conversion', q)]


def p_tilt_taper_off():
    """`_tilt_exactness_phase` with T == 1.

    This also removes the coarse-dx / fine-dx r_safe MISMATCH between the two
    calls of the exact readout's +1/-1 pair, so a null here nulls both."""
    def q(shape, dx, dy, wavelength, R, L, M, sign, centre=(0.0, 0.0)):
        if not CM._exact_tilt_reference():
            return None
        L, M = float(L), float(M)
        if (L == 0.0 and M == 0.0) or not np.isfinite(R) or R == 0.0:
            return None
        Ny, Nx = int(shape[-2]), int(shape[-1])
        x = (np.arange(Nx, dtype=np.float64) - Nx / 2) * dx - float(centre[0])
        y = (np.arange(Ny, dtype=np.float64) - Ny / 2) * dy - float(centre[1])
        X, Y = x[None, :], y[:, None]
        sgn = 1.0 if R > 0 else -1.0
        s = L * L + M * M
        n_par = np.sqrt(1.0 - s)
        uu, vv = X + R * L / n_par, Y + R * M / n_par
        r2 = X * X + Y * Y
        D = (sgn * (np.sqrt(uu * uu + vv * vv + R * R) - abs(R) / n_par)
             - sgn * (np.sqrt(r2 + R * R) - abs(R)) - (L * X + M * Y))
        return np.exp(sign * 1j * (2.0 * np.pi / wavelength) * D)
    return [(CM, '_tilt_exactness_phase', q)]


def p_exact_kz():
    """Replace the Fresnel envelope transfer function by the EXACT
    ``exp(i k z sqrt(1 - (lam f)^2))`` kernel on the same reduced leg."""
    def q(E_in, z, wavelength, dx, dy=None):
        if z == 0.0:
            return np.array(E_in)
        dy = dx if dy is None else dy
        E = np.ascontiguousarray(E_in, dtype=np.complex128)
        ny, nx = E.shape[-2], E.shape[-1]
        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dy)
        s = (wavelength * fx)[None, :] ** 2 + (wavelength * fy)[:, None] ** 2
        prop = s < 1.0
        k = 2.0 * np.pi / wavelength
        ph = np.where(prop, k * z * np.sqrt(np.clip(1.0 - s, 0.0, None)), 0.0)
        H = np.where(prop, np.exp(1j * ph), 0.0)
        out = np.fft.ifft2(np.fft.fft2(E) * H)
        return out.astype(E_in.dtype) if np.iscomplexobj(E_in) else out
    return [(CM, 'fresnel_tf_propagate', q)]


def p_tilt_eikonal_off():
    """POSITIVE CONTROL -- revert the niche-C5 exact tilted eikonal."""
    return [(LT, 'TILTED_CARRIER_EXACT_EIKONAL', False)]


# --------------------------------------------------------------------------
# variant registry:  name -> (chain_kw, traced_kw, patches)
# --------------------------------------------------------------------------
SETS = {
    'ctl': [
        ('NULL identity patch', {}, {},
         [(CM, '_sphere_parab_conversion', CM._sphere_parab_conversion)]),
        ('CONTROL tilt eikonal OFF (C5)', {}, {}, p_tilt_eikonal_off()),
    ],
    'a': [
        ('R_out paraxial x(1+1e-3)', {}, {}, p_rout_scale(1.0 + 1e-3)),
        ('R_out paraxial x(1+1e-2)', {}, {}, p_rout_scale(1.0 + 1e-2)),
        ('sphere<->parab taper OFF', {}, {}, p_sphere_taper_off()),
        ('tilt-exactness taper OFF', {}, {}, p_tilt_taper_off()),
    ],
    'b': [
        ('newton_max_iters 12 -> 60', {}, {'newton_max_iters': 60}, []),
        ('newton_poly_order 6 -> 10', {}, {'newton_poly_order': 10}, []),
        ('decentred_fit_poly_order 10 -> 14', {},
         {'decentred_fit_poly_order': 14}, []),
        ('newton_fit polynomial -> spline', {},
         {'newton_fit': 'spline'}, []),
    ],
    'c': [
        ('amplitude_model ray_density -> screen', {},
         {'amplitude_model': 'screen', 'preserve_input_phase': True}, []),
        ('remap_sampling full -> lattice', {},
         {'remap_sampling': 'lattice'}, []),
        ('fit_radius_beam_factor -> 3.0', {},
         {'fit_radius_beam_factor': 3.0}, []),
        ('ray_subsample 4 -> 2', {'ray_subsample': 2}, {}, []),
    ],
    'd': [
        ('n_fine_cap 12288 -> 16384',
         {'focus_readout': {'n_fine_cap': 16384}}, {}, []),
        ('window_factor 4 -> 6', {'focus_readout': {'window_factor': 6.0}},
         {}, []),
        ('carrier_reference sphere -> parabola',
         {'carrier_reference': 'parabola'}, {}, []),
        ('Fresnel kernel -> exact kz', {}, {}, p_exact_kz()),
    ],
}


def main():
    order = tuple(int(v) for v in os.environ.get('ORD', '-4,-2').split(','))
    sets = os.environ.get('SET', 'ctl').split(',')
    post, env, R, dx, P_in, L, M, cen = A.setup(order)
    import hashlib
    print("=== approx ablation, order %s, sets %s ===" % (order, sets))
    for mod in (A.LT, A.CM):
        print("   lib %s  sha256 %s" % (
            mod.__file__,
            hashlib.sha256(open(mod.__file__, 'rb').read()).hexdigest()[:16]))
    print("grid: RN=%d dx=%.4f um RS=%d NFC=%d WF=%.1f NOUT=%d DXO=%.3f um"
          % (A.RN, dx * 1e6, A.RS, A.NFC, A.WF, A.NOUT, A.DXO * 1e6))
    E0, st0, s0 = A.run_chain(post, env, R, dx, L, M, cen)
    m0 = A.metrics(E0, P_in)
    last = [s for s in st0 if not s.get('target')][-1]
    print("SAMPLING (baseline last group): na_par %.4f  na_measured %.4f  "
          "na_grid_nyquist %.4f  exit power above nyquist %.3e"
          % (last['na_exit'], last.get('na_exit_measured', np.nan),
             last.get('na_grid_nyquist', np.nan),
             last.get('exit_power_above_nyquist', np.nan)))
    rows = [dict(name='BASELINE (shipping defaults)', relL2=0.0, dphi=0.0,
                 secs=s0, **m0)]
    for st in sets:
        for name, ck, tk, pt in SETS[st]:
            try:
                E, stg, secs = A.run_chain(post, env, R, dx, L, M, cen,
                                           chain_kw=dict(ck), traced_kw=tk,
                                           patches=pt)
                d, ph = A.field_diff(E, E0)
                mm = A.metrics(E, P_in)
                rows.append(dict(name=name, relL2=d, dphi=ph, secs=secs, **mm))
            except Exception as exc:                            # noqa: BLE001
                rows.append(dict(name=name, err='%s: %s'
                                 % (type(exc).__name__, exc)))
            print("  done: %-40s" % name, flush=True)
    print()
    A.report(rows)
    print()
    print("dEE3 (points) vs baseline:")
    for r in rows[1:]:
        if r.get('err'):
            continue
        print("  %-40s %+8.4f  (dEE6 %+8.4f, dPtile %+9.5f)"
              % (r['name'], (r['EE3'] - m0['EE3']) * 100,
                 (r['EE6'] - m0['EE6']) * 100,
                 (r['P_tile'] - m0['P_tile']) * 100))


if __name__ == '__main__':
    main()
