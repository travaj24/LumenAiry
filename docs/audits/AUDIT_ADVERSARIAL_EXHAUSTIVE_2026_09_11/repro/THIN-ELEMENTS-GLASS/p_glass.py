import numpy as np, sys
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.glass import get_glass_index, get_glass_index_complex, SELLMEIER_COEFFICIENTS, GLASS_REGISTRY

# Published reference values (Schott/Ohara/RefractiveIndex.info datasheets)
REF = {
 # name: {wl_m: n_published}
 'N-BK7': {587.5618e-9:1.51680, 632.8e-9:1.51509, 1064.0e-9:1.50669},
 'N-SF11':{587.5618e-9:1.78472, 632.8e-9:1.77862, 1064.0e-9:1.75450},
 'N-SSK2':{587.5618e-9:1.62229, 632.8e-9:1.61949, 1064.0e-9:1.60700},
 'F2':    {587.5618e-9:1.62004, 632.8e-9:1.61656, 1064.0e-9:1.60097},
 'N-SF10':{587.5618e-9:1.72825, 632.8e-9:1.72309, 1064.0e-9:1.70240},
 'N-LAK22':{587.5618e-9:1.65113, 632.8e-9:1.64807, 1064.0e-9:1.63461},
 'SiO2':  {587.5618e-9:1.45846, 632.8e-9:1.45702, 1064.0e-9:1.44963},
 'CaF2':  {587.5618e-9:1.43384, 632.8e-9:1.43300, 1064.0e-9:1.42846},
 'MgF2':  {587.5618e-9:1.37774, 632.8e-9:1.37693, 1064.0e-9:1.37151},
 'BaF2':  {587.5618e-9:1.47437, 632.8e-9:1.47301, 1064.0e-9:1.46856},
 'N-SF6': {587.5618e-9:1.80518, 632.8e-9:1.79893},
 'N-LASF9':{587.5618e-9:1.85025, 632.8e-9:1.84281},
 'N-F2':  {587.5618e-9:1.62005, 632.8e-9:1.61656},
 'N-SK16':{587.5618e-9:1.62041, 632.8e-9:1.61727},
 'H-K9L': {587.5618e-9:1.51680},
 'H-LAK52':{587.5618e-9:1.72916},
 'H-LAK53A':{587.5618e-9:1.75500},
 'D-ZK3': {587.5618e-9:1.58913},
 'S-LAH64':{587.5618e-9:1.78800},
 'S-LAH79':{587.5618e-9:2.00330},
 'N-LASF31A':{587.5618e-9:1.88300},
 'N-SF57':{587.5618e-9:1.84666},
 'N-FK51A':{587.5618e-9:1.48656},
}
print("glass            wl[nm]   n_code      n_ref     delta")
worst=[]
for g, d in REF.items():
    for wl, nref in d.items():
        try:
            n = get_glass_index(g, wl)
        except Exception as e:
            print(f"{g:12s} {wl*1e9:8.2f}  ERROR {type(e).__name__}: {e}")
            continue
        dn = n-nref
        flag = "  <<<<" if abs(dn)>2e-4 else ""
        print(f"{g:14s} {wl*1e9:8.2f} {n:.6f} {nref:.6f} {dn:+.2e}{flag}")
        worst.append((abs(dn), g, wl, n, nref))
worst.sort(reverse=True)
print("\nTop-5 discrepancies:")
for a,g,wl,n,nr in worst[:5]:
    print(f"  {g} @{wl*1e9:.1f}nm  code={n:.6f} ref={nr:.6f} d={n-nr:+.2e}")

print("\n--- name normalisation ---")
for nm in ['BK7','NBK7','n-bk7','N-BK7 ','FusedSilica','vacuum','MIRROR','AIR','Air']:
    try:
        print(f"  {nm!r:16s} -> {get_glass_index(nm, 1.064e-6)}")
    except Exception as e:
        print(f"  {nm!r:16s} -> {type(e).__name__}: {str(e)[:90]}")

print("\n--- complex index / kappa sign ---")
import warnings
for nm in ['N-BK7','SiO2','SILICON']:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            nc = get_glass_index_complex(nm, 1.55e-6)
            print(f"  {nm:10s} -> {nc}   warns={[str(x.message)[:60] for x in w]}")
        except Exception as e:
            print(f"  {nm:10s} -> {type(e).__name__}: {str(e)[:100]}")

print("\n--- Abbe number helper present? ---")
import lumenairy.glass as G
print("  abbe in glass module:", [n for n in dir(G) if 'abbe' in n.lower()])

print("\n--- registry keys count:", len(GLASS_REGISTRY), " sellmeier rows:", len(SELLMEIER_COEFFICIENTS))
