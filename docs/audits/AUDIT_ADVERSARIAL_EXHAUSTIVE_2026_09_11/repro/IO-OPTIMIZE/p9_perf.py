import sys, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np, lumenairy as la
import lumenairy.optimize.core as _core
tmpl = la.make_singlet(50e-3, -50e-3, 4e-3, 'N-BK7', aperture=10e-3)
wl = 587.6e-9
# warm caches
s = _core.surfaces_from_prescription(tmpl); _core.system_abcd(s, wl); _core.seidel_coefficients(s, wl)
import copy
def t(label, fn, n=20):
    t0=time.perf_counter()
    for _ in range(n): fn()
    dt=(time.perf_counter()-t0)/n
    print(f"  {label:38s} {dt*1e3:8.3f} ms")
    return dt
a=t("deepcopy(template)", lambda: copy.deepcopy(tmpl))
b=t("surfaces_from_prescription", lambda: _core.surfaces_from_prescription(tmpl))
sf = _core.surfaces_from_prescription(tmpl)
c=t("system_abcd", lambda: _core.system_abcd(sf, wl))
d=t("seidel_coefficients", lambda: _core.seidel_coefficients(sf, wl), n=5)
print(f"  TOTAL per merit eval (ray only)     {(a+b+c+d)*1e3:8.3f} ms")
from lumenairy.glass import get_glass_index
e=t("get_glass_index('N-BK7')", lambda: get_glass_index('N-BK7', wl), n=200)
# vary the wavelength slightly to bust the cache
ws = np.linspace(580e-9, 600e-9, 50); i=[0]
def uncached():
    i[0]=(i[0]+1)%50; return get_glass_index('N-BK7', float(ws[i[0]]))
t("get_glass_index (cold wavelengths)", uncached, n=50)
