import os, re, sys
ROOT = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"


def show(rel, pat=None, limit=None, head=None):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"### {rel}  -- MISSING")
        return
    txt = open(p, encoding="utf-8", errors="replace").read()
    lines = txt.splitlines()
    print(f"\n########## {rel}  ({len(lines)} lines, {len(txt)} bytes) ##########")
    if head:
        for i, l in enumerate(lines[:head], 1):
            print(f"{i:5d}| {l}")
        return
    rx = re.compile(pat) if pat else None
    n = 0
    for i, l in enumerate(lines, 1):
        if rx is None or rx.search(l):
            print(f"{i:5d}| {l}")
            n += 1
            if limit and n >= limit:
                print("   ... (truncated)")
                break


show(".github/workflows/unit-tests.yml",
     r"^\s*(name:|jobs:|runs-on:|strategy:|matrix:|timeout-minutes:|python-version:|- name:|"
     r"shard|split|continue-on-error|if:|env:|fail-fast)|pytest|maxfail|-n |xdist|durations|"
     r"PYTEST|TEST_", limit=200)
show(".github/workflows/publish.yml", r"^\s*(name:|jobs:|runs-on:|- name:|if:)|pytest|twine|build", limit=60)
show(".github/workflows/validate.yml", head=60)
show(".github/workflows/dep-drift.yml", r"^\s*(name:|jobs:|- name:|run:)|pip|python", limit=40)
show(".gitignore", head=40)
show("MANIFEST.in", head=45)
show("pyproject.toml", r"^\[|^\s*(requires|python|dependencies|=|name|version|numpy|scipy|jax|"
     r"numba|pyfftw|h5py|zarr|Programming Language|addopts|testpaths|markers|line-length|target-version|"
     r"select|ignore|strict|classifiers)", limit=200)
