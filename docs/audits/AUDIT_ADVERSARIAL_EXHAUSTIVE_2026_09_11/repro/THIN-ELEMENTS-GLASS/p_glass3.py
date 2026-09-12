import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.glass as G
from refractiveindex import RefractiveIndexMaterial as RIM

CAT = {}
for shelf,book in [('specs','SCHOTT-optical'),('specs','OHARA-optical'),('specs','CDGM-optical'),
                   ('specs','HIKARI-optical'),('specs','SUMITA-optical')]:
    CAT[book]=(shelf,book)

def db_n(name, wl_m, books=('SCHOTT-optical','OHARA-optical','CDGM-optical')):
    for b in books:
        try:
            m = RIM(shelf='specs', book=b, page=name)
            return float(m.get_refractive_index(wl_m*1e9, unit='nm')), b
        except Exception:
            continue
    return None, None

LD, LF, LC = 587.5618e-9, 486.1327e-9, 656.2725e-9
print("=== bundled SELLMEIER_COEFFICIENTS vs refractiveindex.info catalogue ===")
print(f"{'glass':14s} {'book':16s} {'n_d(code)':>10s} {'n_d(db)':>10s} {'dn_d':>10s} {'Vd(code)':>9s} {'Vd(db)':>9s} {'maxdn 0.4-1.6um':>15s}")
scan = np.linspace(0.40e-6, 1.60e-6, 61)
issues=[]
for name, co in sorted(G.SELLMEIER_COEFFICIENTS.items()):
    ndc = G._sellmeier_index(LD, co); nFc = G._sellmeier_index(LF, co); nCc = G._sellmeier_index(LC, co)
    vdc = (ndc-1)/(nFc-nCc)
    nd_db, book = db_n(name, LD)
    if nd_db is None:
        print(f"{name:14s} {'(not in db)':16s} {ndc:10.5f}")
        continue
    nF_db,_ = db_n(name, LF, (book,)); nC_db,_ = db_n(name, LC, (book,))
    vd_db = (nd_db-1)/(nF_db-nC_db)
    mx = 0.0
    for wl in scan:
        try:
            nd2,_ = db_n(name, wl, (book,))
            if nd2 is None: continue
            mx = max(mx, abs(G._sellmeier_index(wl, co)-nd2))
        except Exception:
            pass
    flag = " <<<<" if (abs(ndc-nd_db)>5e-5 or mx>2e-4) else ""
    print(f"{name:14s} {book:16s} {ndc:10.5f} {nd_db:10.5f} {ndc-nd_db:+10.2e} {vdc:9.3f} {vd_db:9.3f} {mx:15.2e}{flag}")
    if flag: issues.append((name, ndc, nd_db, mx))
print("\nFLAGGED:", issues)
