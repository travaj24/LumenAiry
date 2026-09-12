"""Whole-grid vs row-banded assembly: byte identity across the v5.44 matrix."""
import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
p = fixt.small_singlet()
N, dx = 512, 6e-6
E = fixt.gauss(N, dx, 0.4e-3)
base = dict(prescription=p, wavelength=fixt.WL, dx=dx, ray_subsample=8,
            on_undersample='silent', on_pool_memory='silent',
            on_aperture_beam='silent')
cases = [
 ('screen, imap=True ', dict(inverse_map=True)),
 ('screen, imap=False', dict(inverse_map=False)),
 ('rd,     imap=True ', dict(amplitude_model='ray_density', inverse_map=True)),
 ('rd,     imap=False', dict(amplitude_model='ray_density', inverse_map=False)),
 ('rd+remap, imap=T  ', dict(amplitude_model='ray_density', inverse_map=True,
                             preserve_input_phase='remap')),
 ('rd+remap, imap=F  ', dict(amplitude_model='ray_density', inverse_map=False,
                             preserve_input_phase='remap')),
 ('rd+remap+full,imapT', dict(amplitude_model='ray_density', inverse_map=True,
                             preserve_input_phase='remap', remap_sampling='full')),
]
for lbl, extra in cases:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        whole = apply_real_lens_traced(E, sag_chunk_rows=0, **base, **extra)
        band  = apply_real_lens_traced(E, sag_chunk_rows=64, **base, **extra)
    d = np.abs(whole-band)
    n = np.linalg.norm(whole) or 1.0
    print('%-20s max|d|=%.3e  rel=%.3e  bitwise-equal=%s'
          % (lbl, float(d.max()), float(np.linalg.norm(whole-band)/n),
             bool(np.array_equal(whole, band))))
