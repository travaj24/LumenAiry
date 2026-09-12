import pathlib, re
for f, pat in (('tests/unit/test_v5_21_delta_audit.py','multibranch'),
               ('tests/unit/test_niche_audit_w3_elements.py','multibranch'),
               ('tests/unit/test_v5_21_lens_accuracy_extensions.py','multibranch')):
    t = pathlib.Path(f).read_text(encoding='utf-8', errors='replace').splitlines()
    print('='*20, f)
    for i,l in enumerate(t):
        if re.search(pat, l):
            lo=max(0,i-14); hi=min(len(t),i+22)
            print('\n'.join('%5d| %s'%(j+1,t[j][:118]) for j in range(lo,hi)))
            print('   ....')
            break
