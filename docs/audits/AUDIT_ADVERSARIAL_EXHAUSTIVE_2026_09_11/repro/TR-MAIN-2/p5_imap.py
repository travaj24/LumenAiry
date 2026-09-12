import fixt, numpy as np, warnings, time
import lumenairy as la
fixt.register_glass()
from lumenairy.elements import _lens_imap as IM
from lumenairy.elements._lens_traced import apply_real_lens_traced
print('TRACED_INVERSE_MAP =', IM.TRACED_INVERSE_MAP)
p = fixt.small_singlet()
N, dx = 512, 6e-6
E = fixt.gauss(N, dx, 0.4e-3)
kw = dict(prescription=p, wavelength=fixt.WL, dx=dx, on_undersample='silent',
          on_pool_memory='silent', on_aperture_beam='silent', ray_subsample=8)
res = {}
for label, im in [('imap=True (shipped default)', True), ('imap=False', False)]:
    rec = {}
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        t = time.perf_counter()
        o = apply_real_lens_traced(E, inverse_map=im, _imap_out=rec, **kw)
        dt = time.perf_counter() - t
    res[label] = o
    print(f'{label:30s} {dt:7.3f}s  rec={ {k:v for k,v in rec.items() if not isinstance(v,np.ndarray)} }')
    for w in W: print('   warn:', w.category.__name__, str(w.message)[:110])
a, b = res['imap=True (shipped default)'], res['imap=False']
pk = np.abs(b).max()
m = np.abs(b) > 1e-3*pk
dph = np.angle(a[m]) - np.angle(b[m]); dph = (dph+np.pi)%(2*np.pi)-np.pi
print('n compared', m.sum())
print('RMS phase diff  = %.4e rad (%.3e waves)' % (np.sqrt((dph**2).mean()), np.sqrt((dph**2).mean())/(2*np.pi)))
print('max phase diff  = %.4e rad' % np.abs(dph).max())
print('rel field RMS   = %.4e' % (np.linalg.norm(a-b)/np.linalg.norm(b)))
print('P ratio         = %.8f' % (float((np.abs(a)**2).sum())/float((np.abs(b)**2).sum())))
