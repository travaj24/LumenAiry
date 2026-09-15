"""Run the SHIPPED a17 fingerprint helpers on an interpreter with no pytest.

The checker imports pytest at module scope only for its parametrisation
decorators, so a two-attribute stub is enough to import it and call
``ast_fingerprint`` / ``token_fingerprint`` -- the real functions, not a copy.
"""
import importlib.util
import pathlib
import re
import sys
import types

root = pathlib.Path(sys.argv[1]).resolve()

stub = types.ModuleType("pytest")


class _Mark:
    def parametrize(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    def __getattr__(self, name):
        def deco(*a, **k):
            def inner(fn):
                return fn
            return inner
        return deco


stub.mark = _Mark()
stub.skip = lambda *a, **k: None
stub.fixture = lambda *a, **k: (lambda fn: fn)
sys.modules.setdefault("pytest", stub)

path = root / "tests" / "unit" / "test_audit2609_a17_history_relocation.py"
spec = importlib.util.spec_from_file_location("_a17_checker", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

H = re.compile(r"<!--\s*lumenairy-history-doc\s*(?P<body>.*?)-->", re.S)
ok_ast = ok_tok = n = 0
bad = []
for md in sorted((root / "docs" / "history").glob("*.md")):
    if md.name.upper() == "README.MD":
        continue
    body = H.search(md.read_text(encoding="utf-8")).group("body")
    header = {}
    for line in body.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()
    src = (root / header["module"]).read_text(encoding="utf-8")
    n += 1
    a = mod.ast_fingerprint(src) == header["ast_sha256"]
    t = mod.token_fingerprint(src) == header["token_sha256"]
    ok_ast += a
    ok_tok += t
    if not (a and t):
        bad.append((md.stem, a, t))

print("interpreter       :", sys.version.split()[0])
print("documents         :", n)
print("ast  matches      :", ok_ast, "/", n)
print("token matches     :", ok_tok, "/", n)
print("mismatches        :", bad)
sys.exit(0 if not bad else 1)
