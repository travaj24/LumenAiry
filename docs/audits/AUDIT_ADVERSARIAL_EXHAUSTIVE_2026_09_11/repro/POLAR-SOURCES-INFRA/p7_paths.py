from pathlib import Path
lib = Path.home() / '.lumenairy' / 'library' / 'materials'
names = ['normal', 'C:pwned', '..', '../../evil', '..\\..\\evil',
         'C:/abs/evil', '\\\\server\\share\\x', 'con', '.hidden', '', 'a.b/../../c']
base = str(lib.resolve()).lower()
for nm in names:
    safe = nm.replace('/', '_').replace('\\', '_').replace(' ', '_')
    p = lib / (safe + '.json')
    try:
        r = str(p.resolve()).lower()
    except Exception as ex:
        r = 'ERR ' + str(ex)
    esc = 'ESCAPES' if not r.startswith(base) else 'ok'
    print("%-22r -> safe=%-22r -> %-70s [%s]" % (nm, safe, p, esc))
print("")
print("library base:", lib)
