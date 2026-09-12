"""Probe 8: does a complex64 TILTED chain stay complex64?"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.glass import GLASS_REGISTRY
GLASS_REGISTRY['_P8GLASS'] = (lambda wl: 1.5168)
from lumenairy.propagators.carrier import propagate_traced_carrier_chain
from lumenairy.elements._lens_traced import TiltedCarrier
WL = 1.31e-6
def singlet(sd=4e-3):
    return {'wavelength': WL, 'aperture_diameter': 2*sd,
            'surfaces': [
              {'radius': 51.68e-3, 'thickness': 5e-3, 'glass_before': 'air',
               'glass_after': '_P8GLASS', 'semi_diameter': sd},
              {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': '_P8GLASS',
               'glass_after': 'air', 'semi_diameter': sd}],
            'thicknesses': [5e-3], 'stop_index': 0}
N = 256; dx = 8e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x, indexing='xy')
env = np.exp(-(X**2+Y**2)/(300e-6)**2).astype(np.complex64)
TK = dict(amplitude_model='ray_density', preserve_input_phase='remap',
          remap_sampling='full', fit_radius_beam_factor=2.0)
for carrier in (np.inf, TiltedCarrier(np.inf, 0.02, 0.0, 0.0, 0.0)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate_traced_carrier_chain(
            env, [{'prescription': singlet(), 'gap_before': 2e-3}], WL, dx,
            r_in=carrier, ray_subsample=4, final_distance=1e-3,
            traced_kwargs=TK, final_leg='paraxial',
            on_decentred_fit='ignore', on_gap_paraxial='ignore',
            on_gap_frame='ignore', on_multi_congruence='ignore')
    print(f"carrier={carrier}: input c64 -> output dtype {np.asarray(res.field).dtype}")
