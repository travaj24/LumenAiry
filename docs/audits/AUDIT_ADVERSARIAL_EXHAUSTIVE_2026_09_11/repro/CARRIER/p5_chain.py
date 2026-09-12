"""Probe 5: traced carrier chain vs a brute-force ASM + apply_real_lens_traced chain.

point source -> singlet (collimating) -> singlet (focusing) -> image.
The BRUTE chain uses plain band-limited ASM on a grid fine enough to sample the
full field, and the SAME element call, so only the carrier transport / hand-off
bookkeeping differs.
"""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.carrier import (
    propagate_traced_carrier_chain, carrier_referenced_reconstruct,
    carrier_referenced_envelope, propagate_carrier_referenced)
from lumenairy.propagators.propagation import angular_spectrum_propagate
from lumenairy.elements import apply_real_lens_traced
from lumenairy.raytrace.seidel import system_abcd_prescription

WL = 1.31e-6; k = 2*np.pi/WL
NG = 1.5168; R1s, R2s, TC = 51.68e-3, -51.68e-3, 5e-3
from lumenairy.glass import GLASS_REGISTRY
GLASS_REGISTRY['_P5GLASS'] = (lambda wl: NG)

def singlet(sd=10e-3, glass='_P5GLASS'):
    return {'wavelength': WL, 'aperture_diameter': 2*sd,
            'surfaces': [
                {'radius': R1s, 'thickness': TC, 'glass_before': 'air',
                 'glass_after': glass, 'semi_diameter': sd},
                {'radius': R2s, 'thickness': 0.0, 'glass_before': glass,
                 'glass_after': 'air', 'semi_diameter': sd}],
            'thicknesses': [TC], 'stop_index': 0}

M, efl, bfl, ffl = system_abcd_prescription(singlet(), WL)
print("EFL", efl, "BFL", bfl, "FFL", ffl, "ABCD", M.ravel())

# --- source: Gaussian waist w0, placed so the beam arrives at the lens with R=ffl-ish
w0 = 6.0e-6
zR = np.pi*w0**2/WL
z1 = 30e-3                      # source-to-lens-vertex distance
R_in = z1*(1+(zR/z1)**2)
w_L = w0*np.sqrt(1+(z1/zR)**2)
print(f"w0={w0*1e6:.2f}um zR={zR*1e6:.3f}um  at lens: w={w_L*1e3:.4f}mm R={R_in*1e3:.4f}mm NA={w_L/R_in:.4f}")

# grid: must resolve the carrier fringe at the lens for the brute chain
N = 2048
dx = 2*3.0*w_L/N
print(f"N={N} dx={dx*1e6:.4f}um half={0.5*N*dx*1e3:.3f}mm; carrier step at edge="
      f"{k*dx*w_L/abs(R_in):.4f} rad/px")

x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x, indexing='xy'); r2 = X**2+Y**2
env0 = np.exp(-r2/w_L**2).astype(np.complex128)   # ENVELOPE at the lens front vertex
E0 = carrier_referenced_reconstruct(env0, R_in, WL, dx)

# second lens a distance g after the first; then focus
g = 40e-3
TK = dict(amplitude_model='ray_density', preserve_input_phase='remap',
          remap_sampling='full')

# --- CARRIER CHAIN --------------------------------------------------------
groups = [ {'prescription': singlet(), 'gap_before': 0.0},
           {'prescription': singlet(), 'gap_before': g} ]
# paraxial R_out of the pair, to find the focus
R_a = (M[0,0]*R_in + M[0,1])/(M[1,0]*R_in + M[1,1])
R_b = R_a + g
R_c = (M[0,0]*R_b + M[0,1])/(M[1,0]*R_b + M[1,1])
z_img = -R_c
print(f"R_after_L1={R_a*1e3:.4f}mm  at L2={R_b*1e3:.4f}mm  R_out={R_c*1e3:.4f}mm -> image at {z_img*1e3:.4f}mm")

with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter('always')
    res = propagate_traced_carrier_chain(
        env0, groups, WL, dx, r_in=R_in, ray_subsample=2,
        final_distance=z_img*0.5, final_leg='paraxial',
        traced_kwargs=TK, carrier_reference='sphere')
    wmsg = [str(m.message)[:100] for m in W]
print("chain stages:", [{kk: (round(v,8) if isinstance(v,float) else v) for kk,v in s.items()
                         if kk in ('name','R_in','R_out','dx','w','power')} for s in res.stages])
print("chain warnings:", wmsg)
E_chain = carrier_referenced_reconstruct(res.field if res.R is None else res.field, res.R, WL, res.dx) \
    if False else np.asarray(res.field)
dx_chain = res.dx
print("chain out: R=", res.R, "dx=", dx_chain)

# --- BRUTE CHAIN ----------------------------------------------------------
Eb = E0.copy()
Eb = apply_real_lens_traced(Eb, prescription=singlet(), wavelength=WL, dx=dx,
                            carrier=R_in, ray_subsample=2, **TK)
Eb = angular_spectrum_propagate(np.asarray(Eb), g, WL, dx)
Eb = apply_real_lens_traced(Eb, prescription=singlet(), wavelength=WL, dx=dx,
                            carrier=R_b, ray_subsample=2, **TK)
Eb = angular_spectrum_propagate(np.asarray(Eb), z_img*0.5, WL, dx)

print(f"brute grid dx={dx*1e6:.4f}um  chain grid dx={dx_chain*1e6:.4f}um  ratio={dx_chain/dx:.6f}")
# compare on the coarser of the two by resampling the brute field? they should have
# the SAME pitch only if the carrier magnification is 1. Report both.
I_c = np.abs(E_chain)**2; I_b = np.abs(Eb)**2
print(f"power chain={I_c.sum()*dx_chain**2:.6e}  brute={I_b.sum()*dx**2:.6e}  ratio={(I_c.sum()*dx_chain**2)/(I_b.sum()*dx**2):.6f}")
def cen(I, d):
    n = I.shape[0]; xx = (np.arange(n)-n/2)*d
    t = I.sum()
    return (float((I.sum(0)*xx).sum()/t), float((I.sum(1)*xx).sum()/t),
            float(np.sqrt((I*( (xx[None,:])**2 + (xx[:,None])**2 )).sum()/t)))
print("chain centroid/r2m:", cen(I_c, dx_chain))
print("brute centroid/r2m:", cen(I_b, dx))
print("chain peak", I_c.max(), "brute peak", I_b.max()*(dx/dx_chain)**2, "(brute rescaled to chain pitch)")
