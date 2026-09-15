"""Tiny CRLF-preserving literal patcher used by the Wave-5 D fixes."""
import io
import sys


def patch(path, old, new, count=1):
    raw = io.open(path, "rb").read()
    crlf = b"\r\n" in raw
    s = raw.decode("cp1252").replace("\r\n", "\n")
    if s.count(old) != count:
        raise SystemExit(f"{path}: expected {count} occurrence(s), "
                         f"found {s.count(old)}")
    s = s.replace(old, new, count)
    out = s.replace("\n", "\r\n") if crlf else s
    io.open(path, "wb").write(out.encode("cp1252"))
    print(f"patched {path}")
