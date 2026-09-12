import numpy as np, sys, traceback, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.glass as G

print("refractiveindex available:", G._REFRACTIVEINDEX_AVAILABLE)

# ---- P1 candidate: get_glass_index_complex crashes on catalogue glasses w/o kappa
print("\n=== get_glass_index_complex over ALL tuple-registered glasses ===")
bad=[]
for name, entry in sorted(G.GLASS_REGISTRY.items()):
    if not isinstance(entry, tuple): continue
    if entry[0] == '__user__': continue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            nc = G.get_glass_index_complex(name, 1.31e-6)
            ok = f"{nc.real:.5f}+{nc.imag:.3e}j"
        except Exception as e:
            ok = f"RAISED {type(e).__name__} ({type(e).__mro__[1].__name__})"
            bad.append((name, type(e).__name__))
    print(f"  {name:14s} {ok}")
print(f"\n  -> {len(bad)} of the tuple glasses RAISE instead of falling back to kappa=0")
if bad:
    import refractiveindex
    exc = None
    try:
        G.get_glass_index_complex(bad[0][0], 1.31e-6)
    except Exception as e:
        exc = e
    print("  exception class:", type(exc), "MRO:", [c.__name__ for c in type(exc).__mro__])
    print("  caught tuple in glass.py is (AttributeError, NotImplementedError, KeyError, ValueError, TypeError)")

# ---- Abbe number from the bundled Sellmeier vs published catalogue V_d
print("\n=== Abbe number V_d = (n_d-1)/(n_F-n_C) : bundled Sellmeier vs catalogue ===")
LD, LF, LC = 587.5618e-9, 486.1327e-9, 656.2725e-9
PUB_VD = {'N-SF11':25.68,'N-SSK2':53.27,'F2':36.37,'N-SF10':28.53,'N-LAK22':55.89,
          'N-F2':36.43,'N-SK16':60.32,'N-SF57':23.78,'N-FK51A':84.47,'N-LASF31A':40.76,
          'N-K5':59.48,'N-BAK4':43.87,'N-BAF52':46.60,'N-PSK53A':63.48,'N-LAK33B':52.30,
          'N-SK11':60.80,'N-SSK8':49.83,'N-SF2':33.82,'N-SF5':32.21,'N-SF14':27.38,
          'N-SF15':30.20,'N-LASF40':43.20,'N-LASF41':43.13,'N-LASF44':46.50,
          'N-LASF45':31.32,'N-LASF46A':28.53,'N-LASF46B':28.06,'F5':40.73,'SF2':33.85,
          'N-BAF10':47.11,'N-LAK33A':52.30,'BaF2':81.67,'CaF2':95.23,'MgF2':None,
          'N-BK7':64.17,'N-SF6':25.36,'N-LASF9':32.17,'S-LAH64':40.83,'S-LAH79':25.46,
          'H-K9L':64.20,'H-LAK52':54.68,'H-LAK53A':52.32,'H-ZK9B':50.40,'H-ZF12':32.16,
          'D-ZK3':61.18,'D-LAK52':54.84,'H-ZLAF52A':45.39,
          'SiO2':67.82,'F_SILICA':67.82}
rows=[]
for name, vd_pub in PUB_VD.items():
    if vd_pub is None: continue
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            nd = G.get_glass_index(name, LD); nF = G.get_glass_index(name, LF); nC = G.get_glass_index(name, LC)
    except Exception as e:
        print(f"  {name:12s} ERR {type(e).__name__}"); continue
    vd = (nd-1)/(nF-nC)
    rows.append((abs(vd-vd_pub), name, vd, vd_pub, nd))
rows.sort(reverse=True)
print(f"  {'glass':13s} {'V_d code':>9s} {'V_d pub':>8s} {'delta':>8s}   {'n_d':>8s}")
for d,name,vd,vp,nd in rows:
    flag = " <<<<" if d > 0.5 else ""
    print(f"  {name:13s} {vd:9.3f} {vp:8.2f} {vd-vp:+8.3f}   {nd:8.5f}{flag}")
