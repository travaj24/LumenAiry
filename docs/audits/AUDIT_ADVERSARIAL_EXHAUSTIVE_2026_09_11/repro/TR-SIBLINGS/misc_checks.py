import re, pathlib, numpy as np
src = pathlib.Path('lumenairy/elements/_lens_traced_uniform.py').read_text(encoding='utf-8')
for nm in ('_count_interior_turning_points','_classify_catastrophe','_CATASTROPHE_NAMES',
           '_radial_amp_sampler','_roots_in_segments','pearcey','_fold_airy_eval'):
    hits=[i+1 for i,l in enumerate(src.splitlines()) if re.search(r'\b%s\b'%nm,l)]
    print('%-34s occurrences at lines %s'%(nm,hits))
print('  -> `wavelength` inside _build_pearcey_cusp_field body?',
      'wavelength' in src.split('def _build_pearcey_cusp_field')[1].split('\ndef ')[0].split('"""')[2])
print()
print('== eval_into with a NON-CONTIGUOUS out array ==')
from lumenairy.elements._lens_imap import InverseCharacteristic, _td_terms
terms=_td_terms(2)
coef=np.zeros((terms.shape[0],1)); coef[0,0]=7.0   # constant model -> every value 7
m=InverseCharacteristic(coef=coef,terms=terms,degree=2,exit_c=(0.,0.),exit_h=(1.,1.),
                        hull=None,hull_c=None,hull_rmax=None,launch_radius=1.0,
                        wavelength=1e-6,n_samples=10,residual=np.zeros(1),det_j_range=1.0,
                        det_j_sign=1.0,guards={},key=None,build_seconds=0.0)
X=np.linspace(-.5,.5,6); Y=np.zeros(6)
buf=np.zeros((6,2)); view=buf[:,0]              # non-contiguous (stride 2)
print('   view contiguous?', view.flags['C_CONTIGUOUS'])
m.eval_into(X,Y,[view],channels=(0,))
print('   after eval_into, view =', view, '   <-- should be all 7')
cont=np.zeros(6); m.eval_into(X,Y,[cont],channels=(0,))
print('   contiguous control   =', cont)
