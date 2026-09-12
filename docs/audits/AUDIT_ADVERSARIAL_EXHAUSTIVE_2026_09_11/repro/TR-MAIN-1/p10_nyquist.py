"""Probe 10: the exit-NA Nyquist guard -- how NA_exit is estimated."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
WL = common.WL; k0 = 2*np.pi/WL
n_of = lambda g: la.get_glass_index(g, WL)

def rx_f5(sd=None):
    s = [{'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
          'glass_after': '_AUD_GLASS'},
         {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_AUD_GLASS',
          'glass_after': 'air'}]
    if sd is not None:
        for d in s: d['semi_diameter'] = sd
    return {'wavelength': WL, 'aperture_diameter': 24e-3, 'surfaces': s,
            'thicknesses': [5e-3], 'stop_index': 0}

AP = 24e-3
for sd, tag in ((12e-3, 'semi_diameter=12mm (= aperture/2)'),
                (None, 'NO per-surface semi_diameter')):
    rx = rx_f5(sd)
    for N, dxf in ((512, 1.3),):
        dx = dxf*AP/N
        E_in = np.ones((N, N), np.complex128)
        d = {}
        with warnings.catch_warnings(record=True) as wl_:
            warnings.simplefilter('always')
            apply_real_lens_traced(E_in, prescription=rx, wavelength=WL, dx=dx,
                                   ray_subsample=8, n_workers=1,
                                   min_coarse_samples_per_aperture=0,
                                   on_pool_memory='silent', _exit_na_out=d)
        # TRUE marginal NA at the physical aperture edge, from the oracle
        h = np.array([AP/2*0.999])
        _,_,_,dirv = common.oracle_trace(rx, h, np.zeros(1), n_of=n_of)
        na_true = float(np.hypot(dirv[0,0], dirv[0,1]))
        # ...and at the launch radius 1.5x the aperture radius
        h2 = np.array([0.75*AP])
        _,_,_,dv2 = common.oracle_trace(rx, h2, np.zeros(1), n_of=n_of)
        na_launch = float(np.hypot(dv2[0,0], dv2[0,1]))
        print(f"{tag}: N={N} dx={dx*1e6:.2f}um")
        print(f"   reported na_exit={d.get('na_exit'):.5f}  "
              f"na_nyquist={d.get('na_nyquist'):.5f}  "
              f"power_frac_above_nyq={d.get('power_frac_above_nyquist'):.3e}  "
              f"n_rays={d.get('n_rays')}")
        print(f"   TRUE marginal NA at aperture edge (h=12 mm) = {na_true:.5f}"
              f" ; NA at launch radius (h=18 mm) = {na_launch:.5f}")
        print(f"   overstatement factor = {d.get('na_exit')/na_true:.4f}")
        for m in wl_:
            if 'NA_exit' in str(m.message):
                print("   WARN:", str(m.message)[:180])
