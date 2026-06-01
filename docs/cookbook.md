# LumenAiry Cookbook

Detailed runnable examples + use-case recipes.  This file was carved
out of README.md in v5.2.3 to keep the README focused on
installation + quick-start + API surface.

See the top-level [README.md](../README.md) for installation and the
library's organising principles.

See also (canonical sibling docs):

- [CONVENTIONS.md](../CONVENTIONS.md) -- sign conventions, units,
  factory-verb naming (``create_*`` vs ``make_*``), kwarg vocabulary,
  and other library-wide invariants every recipe below relies on.
- [Migration-Guide.md](../Migration-Guide.md) -- v4 -> v5 API moves,
  deprecated kwargs, and shim-removal schedule.  Consult before
  porting older code into a snippet here.

---

## Cookbook

Worked recipes for specific use cases.

### Basic propagation

```python
import numpy as np
import lumenairy as la

# Create a Gaussian beam
E, x, y = la.create_gaussian_beam(N=512, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)

# Propagate 10 cm through free space
E_prop = la.angular_spectrum_propagate(E, z=0.1, wavelength=1.3e-6, dx=2e-6)

# Analyze
cx, cy = la.beam_centroid(E_prop, 2e-6)
dx_b, dy_b = la.beam_d4sigma(E_prop, 2e-6)
print(f"Centroid: ({cx*1e6:.1f}, {cy*1e6:.1f}) um")
print(f"D4sigma:  {dx_b*1e6:.0f} x {dy_b*1e6:.0f} um")
```

### Geometric ray tracing

```python
import lumenairy as la

# Load a prescription and ray-trace it
rx = la.thorlabs_lens('AC254-100-C')
surfaces = la.surfaces_from_prescription(rx)

# ABCD matrix and focal lengths
abcd, efl, bfl, ffl = la.system_abcd(surfaces, wavelength=1.31e-6)
print(f"EFL = {efl*1e3:.1f} mm, BFL = {bfl*1e3:.1f} mm")

# Trace rays and generate a spot diagram
result = la.trace_prescription(rx, wavelength=1.31e-6, num_rings=8,
                               image_distance=bfl)
la.spot_diagram(result, units='um')
la.trace_summary(result)

# Same element list for wave-optics AND ray-optics
elements = [
    {'type': 'propagate', 'z': 50e-3},
    {'type': 'lens', 'f': 100e-3},
    {'type': 'propagate', 'z': 100e-3},
]
# Wave-optics
E_out, _ = la.propagate_through_system(E_in, elements, 1.31e-6, dx)
# Geometric ray trace — same element list
result, surfs = la.raytrace_system(elements, 1.31e-6, semi_aperture=5e-3)
```

### Real lens from Zemax file

```python
# Load a lens prescription from a Zemax .zmx file
rx = la.load_zemax_zmx('path/to/lens.zmx')

# Or use a Thorlabs catalog lens
rx = la.thorlabs_lens('AC254-200-C')

# Fast analytic thin-element model (default)
E_out = la.apply_real_lens(E_in, prescription=rx, wavelength=1.3e-6, dx=2e-6)

# Higher-accuracy hybrid wave/ray model -- sub-nm OPD on doublets
E_out = la.apply_real_lens_traced(E_in, prescription=rx,
                                   wavelength=1.3e-6, dx=2e-6,
                                   ray_subsample=4)
```

### Generate a simulation script from a prescription

```python
# Turn a Zemax .zmx into a self-contained Python sim script
import lumenairy as la

rx = la.load_zemax_zmx('AC254-100-C.zmx')
code = la.generate_simulation_script(
    rx,
    wavelength=1.31e-6,
    N=2048,
    style='unrolled',          # or 'system' for a single propagate_through_system call
    include_analysis=True,
    include_plotting=True,
)
with open('sim_AC254_100C.py', 'w') as f:
    f.write(code)

# Or one-shot from a file path
code = la.generate_script_from_zmx('AC254-100-C.zmx', wavelength=1.31e-6)
```

The output is a runnable script with the prescription data inline, ready
to drop into version control alongside the design or hand to a
collaborator.

### Anamorphic / cylindrical / biconic elements

```python
# Cylindrical lens (focuses in x only)
pres = la.make_cylindrical(R_focus=50e-3, d=3e-3, glass='N-BK7', axis='x')
E_line_focus = la.apply_real_lens(E_in, prescription=pres,
                                   wavelength=1.3e-6, dx=2e-6)

# Biconic singlet (independent x and y curvatures)
pres = la.make_biconic(R1_x=50e-3, R1_y=70e-3,
                        R2_x=-30e-3, R2_y=-40e-3,
                        d=4e-3, glass='N-BK7')
E_anam = la.apply_real_lens(E_in, prescription=pres,
                             wavelength=1.3e-6, dx=2e-6)
```

### Zernike decomposition of an OPD map

OPD sign and the converging-phase convention used here are pinned in
[CONVENTIONS.md section 7](../CONVENTIONS.md#7-sign-conventions);
follow that document if you need to compare these numbers against an
external optical design package.

```python
# Extract the OPD map from a wave field
E_exit = la.apply_real_lens(E_in, prescription=prescription,
                             wavelength=wavelength, dx=dx)
X, Y, opd = la.wave_opd_2d(E_exit, dx, wavelength,
                            aperture=10e-3, focal_length=100e-3,
                            f_ref=100e-3)

# Decompose into 21 Zernike modes (covers up through 5th-order spherical)
coeffs, names = la.zernike_decompose(opd, dx, aperture=10e-3, n_modes=21)
for j, (c, n) in enumerate(zip(coeffs, names)):
    print(f'  Z{j:2d} {n:30s}: {c*1e9:+8.2f} nm RMS')

# Reconstruct from a coefficient set
opd_recon = la.zernike_reconstruct(coeffs, dx, opd.shape, aperture=10e-3)
```

### Sampling check for OPD extraction

```python
# Before committing to a long simulation, verify the grid is fine
# enough for clean OPD unwrap at the pupil edge
samp = la.check_opd_sampling(dx=4e-6, wavelength=1.31e-6,
                              aperture=12e-3, focal_length=45e-3)
print(f'  Nyquist margin: {samp["margin"]:.2f}  (>= 2 = safe)')
if not samp['ok']:
    for rec in samp['recommendations']:
        print('  Suggestion:', rec)
```

### Hybrid wave/ray lens-design optimization

```python
# Refine a Thorlabs achromat to hit a custom focal-length target
template = la.thorlabs_lens('AC254-100-C')
template['aperture_diameter'] = 10e-3

param = la.DesignParameterization(
    template=template,
    free_vars=[
        ('surfaces', 0, 'radius'),
        ('surfaces', 1, 'radius'),
        ('surfaces', 2, 'radius'),
        ('thicknesses', 0),
    ],
    bounds=[(50e-3, 80e-3),
            (-60e-3, -30e-3),
            (-250e-3, -150e-3),
            (4e-3, 8e-3)])

merit = [
    la.FocalLengthMerit(target=110e-3, weight=1.0),
    la.SphericalSeidelMerit(weight=1e-10),
    la.StrehlMerit(min_strehl=0.95, weight=10.0),
]

result = la.design_optimize(parameterization=param,
                             merit_terms=merit,
                             wavelength=1.31e-6,
                             N=256, dx=20e-6,
                             method='L-BFGS-B',
                             max_iter=50)
print(f'Final EFL: {result.context_final.efl*1e3:.3f} mm')
print(f'Best Strehl: {result.context_final.strehl_best:.4f}')
print('Optimised prescription:', result.prescription)
```

### Progress reporting from long-running operations

Any of the core library's slow entry points accept an optional
`progress` callback so scripts and GUIs can drive a progress bar
from the same hook:

```python
import lumenairy as la

def cb(stage, fraction, message=''):
    print(f'{stage}: {fraction*100:5.1f}%  {message}')

# Wave-optics pipeline
E_out = la.apply_real_lens_traced(
    E_in, prescription=prescription, wavelength=1.31e-6, dx=2e-6,
    ray_subsample=4, progress=cb)
E_out, _ = la.propagate_through_system(
    E_in, elements, wavelength=1.31e-6, dx=2e-6, progress=cb)

# Through-focus and tolerancing
scan = la.through_focus_scan(E_exit, dx, wavelength, z_values, progress=cb)
results = la.tolerancing_sweep(prescription, wavelength, N, dx, E_source,
                                perturbations, focal_length=bfl,
                                aperture=ap, progress=cb)
stats = la.monte_carlo_tolerancing(prescription, wavelength, N, dx, E_source,
                                    spec, focal_length=bfl, aperture=ap,
                                    n_trials=100, progress=cb)

# Design optimization (progress is per merit-function evaluation)
result = la.design_optimize(parameterization=param, merit_terms=merits,
                             wavelength=1.31e-6, max_iter=200, progress=cb)
```

The callback signature is `(stage: str, fraction: float, message: str)`
where `fraction` is in `[0, 1]`.  Implementations should be cheap and
thread-safe; exceptions raised inside the callback are swallowed so a
broken progress UI cannot crash a simulation.

`ProgressScaler` lets a parent caller nest sub-tasks within a budget
so long pipelines (`apply_real_lens` inside `apply_real_lens_traced`,
which itself is one of many surfaces inside `propagate_through_system`,
which might be inside `tolerancing_sweep`) report a single monotonic
0\u20131 timeline.  See `lumenairy/progress.py` for the full
protocol.

### Through-focus and tolerancing

The Strehl denominator ``ideal_peak`` is pinned to the unperturbed
nominal pupil so tolerancing trials report Strehl bounded to [0, 1];
units and phase-sign conventions used here follow
[CONVENTIONS.md section 7](../CONVENTIONS.md#7-sign-conventions).

```python
# Run a 21-plane through-focus scan
E_exit = la.apply_real_lens(E_in, prescription=prescription,
                             wavelength=wavelength, dx=dx)
ideal_peak = la.diffraction_limited_peak(E_exit, wavelength, bfl, dx)
z_values = bfl + np.linspace(-1e-3, +1e-3, 21)
scan = la.through_focus_scan(E_exit, dx, wavelength, z_values,
                              ideal_peak=ideal_peak,
                              bucket_radius=20e-6)
z_best, strehl_best = la.find_best_focus(scan, 'strehl')
la.plot_through_focus(scan, best_z=z_best, path='through_focus.png')

# Tolerancing: how does Strehl change with surface tilt / decenter?
perts = [
    la.Perturbation(surface_index=0, tilt=(1e-3, 0),       name='S0 tilt 1 mrad'),
    la.Perturbation(surface_index=1, decenter=(50e-6, 0),  name='S1 decenter 50 um'),
    la.Perturbation(surface_index=2, form_error_rms=100e-9,
                    random_seed=42, name='S2 form error 100 nm RMS'),
]
results = la.tolerancing_sweep(prescription, wavelength, N, dx,
                                E_in, perts,
                                focal_length=bfl, aperture=10e-3,
                                bucket_radius=20e-6)
```

### Polarization

The Jones-field method API (``field.apply_thin_lens``,
``field.propagate``, ...) and the ``create_*`` vs ``apply_*`` split
shown below are v5 conventions; pre-v5 free-function shims were
removed in the v5.0 cleanup -- see
[Migration-Guide.md](../Migration-Guide.md) if you are porting v4
polarization code.

```python
# Create a right-hand circularly polarized Gaussian beam
scalar, _, _ = la.create_gaussian_beam(256, 2e-6, 1.3e-6, sigma=30e-6)
field = la.create_circular_polarized(scalar, dx=2e-6, handedness='right')

# Propagate through a half-wave plate at 22.5°
la.apply_half_wave_plate(field, angle=np.pi/8)

# Apply a lens (polarization-preserving)
field.apply_thin_lens(f=100e-3, wavelength=1.3e-6)

# Propagate
field.propagate(z=100e-3, wavelength=1.3e-6)

# Measure Stokes parameters
S = la.stokes_parameters(field)
print(f"S3/S0 = {S['S3'].mean() / S['S0'].mean():+.3f}")
```

### Rigorous gratings / metasurfaces (RCWA)

`rcwa_*` solves the full vector Maxwell equations in a laterally periodic
layer — the rigorous counterpart to the scalar `thin_grating` and the
laterally-uniform `coatings` TMM. Every solver is backend-dispatched
(NumPy / CuPy via `use_gpu` / differentiable JAX), accepts `te`/`tm` (or the
`s`/`p` aliases), and conserves energy for lossless media.

```python
import numpy as np
import lumenairy as la

# 1-D binary grating: rigorous diffraction efficiencies (oblique TM)
orders, R, T = la.rcwa_efficiency_1d(
    period=1.2e-6, n_ridge=2.5, n_groove=1.0, n_substrate=1.5,
    n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.4, wavelength=0.55e-6,
    angle=np.deg2rad(15), polarization='tm', n_orders=40)
print(f"R+T = {R.sum() + T.sum():.6f}  (lossless -> 1)")

# A multilayer stack with a 2-D patterned layer; bridge the specular
# (zeroth-order) Jones reflection into the polarization pipeline.
cell = np.where(((np.add.outer(np.arange(32) - 16, np.zeros(32))) ** 2
                 + (np.add.outer(np.zeros(32), np.arange(32) - 16)) ** 2)
                < 8 ** 2, 6.0, 2.0).astype(complex)
res = (la.RCWAStack(0.8e-6, period_y=0.8e-6, n_substrate=1.5, n_orders=4,
                    n_orders_y=4)
       .add_layer(0.30e-6, eps_cell=cell)
       .set_source(0.633e-6, theta=np.deg2rad(10))
       .solve())
jones = res.to_jones_field(256, 256, dx=0.8e-6, incident=(1.0, 0.0))  # specular
```

**Full pipeline: deflector cell → diffracted field → propagate → focal spot.**
A strongly diffracting cell puts most power in non-zero orders, which
`to_multiorder_field` reconstructs as a propagatable `JonesField` (the v5.5.2
multi-order bridge):

```python
# A 1-D beam-deflector grating (period 2*wl -> propagating +/-1 orders)
S = 64
xprof = (np.arange(S) + 0.5) / S
cell1d = np.where(xprof < 0.5, 2.5 ** 2, 1.0).astype(complex)
res = (la.RCWAStack(2.0e-6, n_superstrate=1.0, n_substrate=1.0, n_orders=12)
       .add_layer(0.4e-6, eps_cell=cell1d)
       .set_source(0.633e-6, theta=np.deg2rad(8)).solve())

# Inspect where the power went, then reconstruct the diffracted field
o, R, T = res.efficiencies()
amps = res.per_order_amplitudes('transmission')          # (2, N) per order

# One order as a tilted carrier, or the full multi-order superposition:
deflected = res.to_jones_field(512, 512, dx=2.0e-6 / 64, order=+1,
                               port='transmission')
field = res.to_multiorder_field(512, 512, dx=2.0e-6 / 64, port='transmission')
field.propagate(z=50e-6, wavelength=0.633e-6)            # into the JonesField pipeline

# A periodic element can also drop into a scalar system as its specular order:
E_out = la.propagate_through_system(
    np.ones((256, 256), complex),
    [{'type': 'rcwa', 'result': res, 'port': 'transmission'},
     {'type': 'propagate', 'z': 10e-3}],
    wavelength=0.633e-6, dx=2e-6)
```

For gradient-based metasurface inverse design, pass `jax.numpy` arrays (and
set `jax.config.update('jax_enable_x64', True)` — RCWA needs double
precision); `rcwa_efficiency_1d` / `_2d` then differentiate w.r.t. the
permittivities, depth, and angle via `jax.grad` (see
`examples/13_rcwa_inverse_design.py`). On thread-oversubscribed machines,
`la.set_blas_threads(2)` gives a modest (~2–3×) solve speedup.

### Phase retrieval (Gerchberg-Saxton CGH design)

```python
# Design a phase-only DOE to turn a Gaussian into a flat-top
x = np.linspace(-1, 1, 256)
X, Y = np.meshgrid(x, x)
source = np.exp(-(X**2 + Y**2) / 0.3**2)
target = (np.sqrt(X**2 + Y**2) < 0.4).astype(float)

phase, err = la.gerchberg_saxton(source, target, n_iter=500)
# 'phase' is the design phase-only DOE
```

### Save and load multi-plane simulations (HDF5)

```python
# Save a propagation simulation with multiple planes
planes = [
    {'field': E0, 'dx': 2e-6, 'z': 0.0,    'label': 'source'},
    {'field': E1, 'dx': 2e-6, 'z': 10e-3,  'label': 'after lens'},
    {'field': E2, 'dx': 2e-6, 'z': 100e-3, 'label': 'focal plane'},
]
la.save_planes_h5('simulation.h5', planes, wavelength=1.3e-6)

# Load back later
planes, meta = la.load_planes_h5('simulation.h5')
print(f"Wavelength: {meta['wavelength']*1e9:.0f} nm")
for p in planes:
    print(f"  {p['label']}: z={p['z']*1e3:.1f} mm, shape={p['field'].shape}")

# Append planes incrementally during a long simulation
la.append_plane_h5('simulation.h5', E_new, dx=2e-6, z=200e-3,
                   label='detector plane')

# Save a polarized Jones field
la.save_jones_field_h5('polarized.h5', jones_field, wavelength=1.3e-6)
```

### Plotting

```python
import matplotlib.pyplot as plt

# Single field intensity and phase
fig, axes = la.plot_field(E, dx=2e-6, title='Focal plane')

# Log-scale intensity
fig, ax = la.plot_intensity(E, dx=2e-6, log=True)

# Cross-section with phase overlay
fig, ax = la.plot_cross_section(E, dx=2e-6, axis='x', show_phase=True)

# Multi-plane grid from a loaded HDF5 file
planes, _ = la.load_planes_h5('simulation.h5')
fig, axes = la.plot_planes_grid(planes, n_cols=3, suptitle='Propagation')

# PSF and MTF
fig1, ax1 = la.plot_psf(psf, dx_psf=dx_psf, log=True)
fig2, ax2 = la.plot_mtf(freq, mtf_profile, diffraction_limit=100)

# Stokes parameters for a polarized field
fig, axes = la.plot_stokes(jones_field)

# Polarization ellipses overlaid on intensity
fig, ax = la.plot_polarization_ellipses(jones_field, n_ellipses=12)

plt.show()
```

### PSF / MTF analysis

```python
# Build a circular pupil with some spherical aberration
pupil = la.apply_aperture(
    np.ones((256, 256), dtype=complex),
    dx=25e-6, shape='circular',
    params={'diameter': 5e-3}
)
pupil = la.apply_zernike_aberration(
    pupil, dx=25e-6,
    coefficients={(4, 0): 0.25},       # 1/4 wave spherical
    aperture_radius=2.5e-3
)

# Compute PSF and MTF
psf, dx_psf = la.compute_psf(pupil, wavelength=1.3e-6, f=50e-3, dx_pupil=25e-6)
mtf = la.compute_mtf(psf)
freq, mtf_profile = la.mtf_radial(mtf, dx_psf, 1.3e-6, 50e-3)
```
