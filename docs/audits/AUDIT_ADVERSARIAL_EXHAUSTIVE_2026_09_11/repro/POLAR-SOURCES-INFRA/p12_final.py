import sys, numpy as np, warnings, inspect
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.elements.coatings import coating_reflectance
import lumenairy.sources.core as SC

wl = 633e-9
print("=== 1. dy threading (CONVENTIONS section 5) across every source factory ===")
N, dx, dy = 64, 2e-6, 5e-6
fns = [
    ('create_gaussian_beam', lambda **k: SC.create_gaussian_beam(N, dx, wl, w0=20e-6, **k)),
    ('create_hermite_gauss', lambda **k: SC.create_hermite_gauss(N, dx, 20e-6, wl, m=1, **k)),
    ('create_laguerre_gauss', lambda **k: SC.create_laguerre_gauss(N, dx, 20e-6, wl, l=1, **k)),
    ('create_tilted_plane_wave', lambda **k: SC.create_tilted_plane_wave(N, dx, wl, angle_y=0.01, **k)),
    ('create_point_source', lambda **k: SC.create_point_source(N, dx, wl, z0=1e-3, **k)),
    ('create_top_hat_beam', lambda **k: SC.create_top_hat_beam(N, dx, wl, diameter=40e-6, **k)),
    ('create_annular_beam', lambda **k: SC.create_annular_beam(N, dx, wl, outer_diameter=60e-6,
                                                               inner_diameter=20e-6, **k)),
    ('create_bessel_beam', lambda **k: SC.create_bessel_beam(N, dx, wl, 0.05, **k)),
    ('create_fiber_mode', lambda **k: SC.create_fiber_mode(N, dx, wl, mode_field_diameter=20e-6, **k)),
]
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for name, f in fns:
        try:
            r0 = f(); r1 = f(dy=dy)
            y0 = np.asarray(r0[2]); y1 = np.asarray(r1[2])
            uses = not np.allclose(y0, y1)
            fieldchg = not np.allclose(np.asarray(r0[0]), np.asarray(r1[0]))
            print("  %-26s y-axis honours dy: %-5s  field changes: %s"
                  % (name, uses, fieldchg))
        except TypeError as ex:
            print("  %-26s NO dy kwarg: %s" % (name, str(ex)[:70]))
    # led
    try:
        E0, a0, x0, y0 = SC.create_led_source(N, dx, wl, diameter=40e-6, divergence_angle=0.2)
        E1, a1, x1, y1 = SC.create_led_source(N, dx, wl, diameter=40e-6, divergence_angle=0.2, dy=dy)
        print("  %-26s y-axis honours dy: %-5s  field changes: %s"
              % ('create_led_source', not np.allclose(y0, y1), not np.allclose(E0, E1)))
    except Exception as ex:
        print("  create_led_source:", type(ex).__name__, str(ex)[:80])
    for nm, f in (('create_gaussian_schell_source',
                   lambda **k: SC.create_gaussian_schell_source(N=N, dx=dx, wavelength=wl,
                                                                w0=20e-6, sigma_g=10e-6,
                                                                n_realizations=2, rng=1, **k)),
                  ('create_annular_incoherent_source',
                   lambda **k: SC.create_annular_incoherent_source(N=N, dx=dx, wavelength=wl,
                                                                   inner_radius=10e-6,
                                                                   outer_radius=30e-6,
                                                                   n_realizations=2, rng=1, **k))):
        a = f(); b = f(dy=dy)
        print("  %-26s dy accepted; ensemble changes: %s"
              % (nm, not np.allclose(a[0], b[0])))

print("")
print("=== 2. coatings p-transmittance with an ABSORBING exit medium ===")
def airy(n1, n2, n3, d, wl, th1, pol):
    s1 = n1 * np.sin(th1); c1 = np.cos(th1)
    def ct(n):
        c = np.sqrt(1 - (s1 / n) ** 2 + 0j)
        return c if (n * c).imag >= 0 else -c
    c2, c3 = ct(n2), ct(n3)
    if pol == 's':
        r12 = (n1 * c1 - n2 * c2) / (n1 * c1 + n2 * c2); r23 = (n2 * c2 - n3 * c3) / (n2 * c2 + n3 * c3)
        t12 = 2 * n1 * c1 / (n1 * c1 + n2 * c2); t23 = 2 * n2 * c2 / (n2 * c2 + n3 * c3)
    else:
        r12 = (n2 * c1 - n1 * c2) / (n2 * c1 + n1 * c2); r23 = (n3 * c2 - n2 * c3) / (n3 * c2 + n2 * c3)
        t12 = 2 * n1 * c1 / (n2 * c1 + n1 * c2); t23 = 2 * n2 * c2 / (n3 * c2 + n2 * c3)
    b = 2 * np.pi / wl * n2 * d * c2
    r = (r12 + r23 * np.exp(2j * b)) / (1 + r12 * r23 * np.exp(2j * b))
    t = (t12 * t23 * np.exp(1j * b)) / (1 + r12 * r23 * np.exp(2j * b))
    if pol == 's':
        T = np.real(n3 * c3) / np.real(n1 * c1) * abs(t) ** 2
    else:
        T = np.real(np.conj(n3) * c3) / np.real(np.conj(n1) * c1) * abs(t) ** 2
    return abs(r) ** 2, T
for nsub in (4.0 + 0.05j, 1.5 + 0.5j, 0.27 + 2.78j):
    for thd in (0.0, 55.0):
        for pol in ('s', 'p'):
            R, T, ph = coating_reflectance([(1.46, 100e-9)], 550e-9, angle=np.radians(thd),
                                           n_substrate=nsub, polarization=pol)
            Ra, Ta = airy(1.0, 1.46, nsub, 100e-9, 550e-9, np.radians(thd), pol)
            print("  n_sub=%-12s %4.0fdeg %s: R %.8f/%.8f (d=%.1e)  T %.8f/%.8f (d=%.2e)"
                  % (nsub, thd, pol, R, Ra, abs(R - Ra), T, Ta, abs(T - Ta)))

print("")
print("=== 3. GSM 'mcf' return: coherent-mode spectrum vs Starikov-Wolf ===")
Ns, dxs, w0s, sg = 20, 2.5e-6, 20e-6, 10e-6
mcf = SC.create_gaussian_schell_source(N=Ns, dx=dxs, wavelength=wl, w0=w0s, sigma_g=sg,
                                       n_realizations=4000, rng=3, return_kind='mcf')
lam, modes = mcf.coherent_modes()
lam = np.asarray(lam, float); lam = lam / lam.sum()
sigma_s = w0s / 2.0
a = 1 / (4 * sigma_s ** 2); b = 1 / (2 * sg ** 2); c = np.sqrt(a * a + 2 * a * b)
l1 = np.array([(np.pi / (a + b + c)) ** 0.5 * (b / (a + b + c)) ** n for n in range(10)])
l2 = np.sort(np.array([p * q for p in l1 for q in l1]))[::-1]
l2 = l2 / l2.sum()
print("  library modes (normalized) [:6]:", np.round(lam[:6], 5))
print("  Starikov-Wolf analytic     [:6]:", np.round(l2[:6], 5))
print("  n_modes retained:", len(lam))

print("")
print("=== 4. create_multi_field_sources ===")
src, x, y = SC.create_multi_field_sources(32, 4e-6, wl, [0.01, (0.02, -0.01)])
print("  n sources:", len(src), " element shapes:", [(s[0].shape, s[1], s[2]) for s in src])
