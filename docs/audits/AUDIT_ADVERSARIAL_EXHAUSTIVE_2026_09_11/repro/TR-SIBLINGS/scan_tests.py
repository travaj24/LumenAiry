import pathlib, re, os
roots = [pathlib.Path('tests'), pathlib.Path('validation'), pathlib.Path('examples')]
pat = re.compile(r"multibranch|caustic\s*=\s*['\"]uniform|apply_real_lens_traced_uniform")
files=[]
for R in roots:
    if not R.exists(): continue
    for p in R.rglob('*.py'):
        t=p.read_text(encoding='utf-8',errors='replace')
        if pat.search(t): files.append((p,t))
print('files touching multibranch/uniform:', len(files))
for p,t in files:
    # last-surface radius of every prescription literal in the file
    rads = re.findall(r"'radius'\s*:\s*([^,\}\n]+)", t)
    mk   = re.findall(r"make_singlet\([^)]*\)", t)
    print('  %-58s  radius literals: %s' % (str(p), ', '.join(r.strip() for r in rads[:12]) or '-'))
    for m in mk[:4]: print('       %s' % m.replace('\n',' ')[:110])
print()
print('CPU count:', os.cpu_count())
