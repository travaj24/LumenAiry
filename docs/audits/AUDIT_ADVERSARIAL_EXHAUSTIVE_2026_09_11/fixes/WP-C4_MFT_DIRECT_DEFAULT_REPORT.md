# WP-C4 -- `method='auto'` selects the direct-matrix MFT route

Date: 2026-09-20.  Base: `49ddf4bd` (the commit the sibling WP-C branches share;
`main` itself was two commits behind it when this worktree was cut, and the
branch was reset onto `49ddf4bd` so the bit-identity archive is the one the brief
names).  Branch: `feat/c4-mft-direct-default`.  Builds: Windows py3.14
(numpy 2.4.4, scipy 1.17.1) and WSL py3.12 (numpy 2.4.6, scipy 1.17.1), both
bound to this worktree with `lumenairy.__file__` printed by every probe.

Every command in this report ran with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line; every pytest run used `--capture=sys -p no:randomly`.

Decision this implements: `MAINTAINER_DECISIONS_2026_09.md` section 4.2.
Measurements it builds on: `WAVE5_HYGIENE2_REPORT.md` H2-1,
`VERIFY_WAVE5_HYGIENE2.md` H2-1, `VERIFY_WAVE5_HYGIENE2_ROUND2.md` D-1, and
hygiene-2 round 3's exact-phase probe.

---

## 1.  What shipped

`method='auto'` -- the default on `_bluestein_2d`, `_bluestein_centred_2d` and
the three public MFT entry points -- takes the dense matrix-Fourier route
(`_direct_matrix_2d`) where that route was MEASURED never slower on either
build, and the chirp-Z route it has always taken everywhere else, byte for byte.

The rule is one module constant and one function:

```python
_MFT_DIRECT_MAX_RATIO = 1.0 / 32.0        # the boundary
_MFT_DIRECT_ALWAYS    = float('inf')      # the documented "always"
_MFT_DIRECT_NEVER     = 0.0               # the documented "never"

def _auto_selects_direct(Ny_in, Nx_in, N_out_y, N_out_x) -> bool:
    r = float(_MFT_DIRECT_MAX_RATIO)
    if not (r > 0.0):                     # 0.0, negative, or nan -> never
        return False
    ny, nx = int(Ny_in), int(Nx_in)
    my, mx = int(N_out_y), int(N_out_x)
    if ny < 1 or nx < 1 or my < 1 or mx < 1:
        return False
    if r == float('inf'):                 # the documented "always"
        return True
    return max(my / ny, mx / nx) <= r
```

The MAX over the two axes decides, which is the conservative reading on an
anisotropic grid: a shape reaches the dense route only when NEITHER axis is past
the boundary.  The comparison is `<=` and the constant is the largest ratio
measured safe rather than the first one measured unsafe, so the constant names a
shape that was actually timed.  A non-positive or `nan` constant means NEVER and
is decided before the division.

---

## 2.  The boundary, derived

### 2.1  The criterion

"The dense route is never slower on EITHER build."  The comparison is against
`min(chirp-Z 2-D, separable)` -- the FASTER of the two routes `'auto'` could
otherwise have taken -- because `'auto'`'s pre-5.49.0 arm depends on the
`separable` flag and either arm can be the one a given caller was on.  Taking
the faster of them is the strictest reading and the only one that makes the
claim true whichever fallback the caller had.

### 2.2  The ladder, and the load

`validation/probe_c4_mft_direct/c4_ladder.py`, `N` in
{64,128,256,512,1024,2048} x `M` in {16,32,64,128,256,512,1024} -- 42 shapes --
best of five, COLD (every registered library cache dropped and `gc.collect()`
before each repeat, which is the production regime `_bluestein.py`'s own
byte-cap comment records with `hits = 0` across a full production order).
`alpha` is held at a value whose phase budget is 1e3 at every shape, six decades
under the guard, so no route pays a warning and the timing is of the arithmetic.
Timing and `tracemalloc` are separate passes AND separate invocations, because
`tracemalloc` charges per allocation and the chirp-Z route allocates far more
objects.

**THE LOAD IS RECORDED IN THE JSON**, before and after each run, as a process
census and a measured single-thread reference loop.  The box carries other work,
so these times are BOUNDS -- a contended box can only make a route look slower --
and the quantity the boundary is read from is a RATIO taken under the same
conditions for all three routes, which is the robust part.

| run | build | python processes | total processes | reference loop (before / after) |
|---|---|---|---|---|
| 42-shape ladder | WIN-py3.14 | 8 | 421 / 418 | 2.83 ms / 2.45 ms |
| 42-shape ladder | WSL-py3.12 | 1 | 6 / 6 | 2.99 ms / 2.75 ms |
| boundary re-measure | WIN-py3.14 | 8, 8, 11 | ~420 | 3.04 / 2.50 / 2.54 ms |
| boundary re-measure | WSL-py3.12 | 1, 1, 1 | 6 | 3.06 / 2.90 / 2.53 ms |

### 2.3  The TIME table, both builds

`d/fb` is `dense / min(chirp-Z 2-D, separable)`; **bold** marks a shape where
the dense route is SLOWER.  Best of five, seconds.

| N | M | M/N | WIN dense | WIN sep | WIN chirp | WIN d/fb | WSL dense | WSL sep | WSL chirp | WSL d/fb |
|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 16 | 0.0078125 | 0.0137 | 0.1149 | 0.1570 | 0.120 | 0.0130 | 0.2453 | 0.7100 | 0.053 |
| 1024 | 16 | 0.015625 | 0.0041 | 0.0321 | 0.0422 | 0.129 | 0.0051 | 0.0095 | 0.1491 | 0.531 |
| 2048 | 32 | 0.015625 | 0.0253 | 0.1218 | 0.1549 | 0.208 | 0.0230 | 0.2582 | 0.6920 | 0.089 |
| 512 | 16 | 0.03125 | 0.0020 | 0.0091 | 0.0236 | 0.222 | 0.0012 | 0.0020 | 0.0115 | 0.603 |
| 1024 | 32 | 0.03125 | 0.0072 | 0.0312 | 0.0634 | 0.231 | 0.0065 | 0.0112 | 0.1536 | 0.583 |
| 2048 | 64 | 0.03125 | 0.0545 | 0.1159 | 0.2352 | 0.471 | 0.0417 | 0.2658 | 0.7381 | 0.157 |
| 256 | 16 | 0.0625 | 0.0006 | 0.0027 | 0.0044 | 0.214 | 0.0004 | 0.0010 | 0.0033 | 0.423 |
| 512 | 32 | 0.0625 | 0.0023 | 0.0101 | 0.0148 | 0.227 | 0.0019 | 0.0023 | 0.0134 | 0.829 |
| 1024 | 64 | 0.0625 | 0.0142 | 0.0372 | 0.0502 | 0.381 | 0.0115 | 0.0091 | 0.0672 | **1.271** |
| 2048 | 128 | 0.0625 | 0.0987 | 0.1554 | 0.2427 | 0.635 | 0.0846 | 0.2731 | 0.8454 | 0.310 |
| 128 | 16 | 0.125 | 0.0003 | 0.0007 | 0.0019 | 0.392 | 0.0002 | 0.0007 | 0.0012 | 0.335 |
| 256 | 32 | 0.125 | 0.0012 | 0.0019 | 0.0067 | 0.613 | 0.0007 | 0.0011 | 0.0044 | 0.632 |
| 512 | 64 | 0.125 | 0.0045 | 0.0085 | 0.0160 | 0.531 | 0.0037 | 0.0025 | 0.0167 | **1.459** |
| 1024 | 128 | 0.125 | 0.0282 | 0.0370 | 0.0760 | 0.763 | 0.0232 | 0.0101 | 0.0991 | **2.297** |
| 2048 | 256 | 0.125 | 0.1980 | 0.1661 | 0.4139 | **1.192** | 0.1789 | 0.3069 | 1.0691 | 0.583 |
| 64 | 16 | 0.25 | 0.0002 | 0.0003 | 0.0007 | 0.635 | 0.0001 | 0.0006 | 0.0010 | 0.244 |
| 128 | 32 | 0.25 | 0.0004 | 0.0006 | 0.0019 | 0.725 | 0.0003 | 0.0009 | 0.0014 | 0.337 |
| 256 | 64 | 0.25 | 0.0017 | 0.0022 | 0.0074 | 0.767 | 0.0014 | 0.0012 | 0.0051 | **1.127** |
| 512 | 128 | 0.25 | 0.0099 | 0.0095 | 0.0247 | **1.038** | 0.0075 | 0.0030 | 0.0285 | **2.499** |
| 1024 | 256 | 0.25 | 0.0603 | 0.0423 | 0.0989 | **1.426** | 0.0514 | 0.0147 | 0.1459 | **3.495** |
| 2048 | 512 | 0.25 | 0.4088 | 0.1868 | 0.4392 | **2.189** | 0.4234 | 0.3777 | 1.1578 | **1.121** |
| 64 | 32 | 0.5 | 0.0003 | 0.0005 | 0.0008 | 0.584 | 0.0003 | 0.0006 | 0.0022 | 0.387 |
| 128 | 64 | 0.5 | 0.0008 | 0.0010 | 0.0029 | 0.799 | 0.0006 | 0.0010 | 0.0015 | 0.624 |
| 256 | 128 | 0.5 | 0.0040 | 0.0037 | 0.0091 | **1.091** | 0.0028 | 0.0014 | 0.0129 | **2.035** |
| 512 | 256 | 0.5 | 0.0216 | 0.0163 | 0.0402 | **1.325** | 0.0171 | 0.0044 | 0.1140 | **3.854** |
| 1024 | 512 | 0.5 | 0.1417 | 0.0828 | 0.1673 | **1.711** | 0.1187 | 0.0196 | 0.5086 | **6.069** |
| 2048 | 1024 | 0.5 | 0.9466 | 0.3145 | 0.4616 | **3.010** | 0.9550 | 0.6958 | 2.2517 | **1.372** |
| 64 | 64 | 1 | 0.0004 | 0.0005 | 0.0015 | 0.733 | 0.0004 | 0.0008 | 0.0035 | 0.470 |
| 128 | 128 | 1 | 0.0015 | 0.0013 | 0.0071 | **1.121** | 0.0012 | 0.0010 | 0.0073 | **1.246** |
| 256 | 256 | 1 | 0.0092 | 0.0070 | 0.0170 | **1.315** | 0.0066 | 0.0021 | 0.0308 | **3.196** |
| 512 | 512 | 1 | 0.0540 | 0.0359 | 0.0634 | **1.505** | 0.0418 | 0.0067 | 0.2096 | **6.197** |
| 1024 | 1024 | 1 | 0.3640 | 0.1540 | 0.2445 | **2.364** | 0.3113 | 0.0526 | 0.8512 | **5.916** |
| 64 | 128 | 2 | 0.0007 | 0.0006 | 0.0024 | **1.172** | 0.0008 | 0.0032 | 0.0071 | 0.250 |
| 128 | 256 | 2 | 0.0038 | 0.0037 | 0.0098 | **1.015** | 0.0029 | 0.0017 | 0.0137 | **1.734** |
| 256 | 512 | 2 | 0.0231 | 0.0236 | 0.0474 | 0.979 | 0.0168 | 0.0038 | 0.0973 | **4.403** |
| 512 | 1024 | 2 | 0.1385 | 0.0743 | 0.1706 | **1.865** | 0.1223 | 0.0662 | 0.5576 | **1.848** |
| 64 | 256 | 4 | 0.0017 | 0.0022 | 0.0083 | 0.784 | 0.0015 | 0.0043 | 0.0130 | 0.338 |
| 128 | 512 | 4 | 0.0100 | 0.0127 | 0.0287 | 0.786 | 0.0076 | 0.0032 | 0.0427 | **2.408** |
| 256 | 1024 | 4 | 0.0659 | 0.0630 | 0.1209 | **1.047** | 0.0515 | 0.0101 | 0.2521 | **5.107** |
| 64 | 512 | 8 | 0.0048 | 0.0091 | 0.0201 | 0.529 | 0.0038 | 0.0119 | 0.0427 | 0.319 |
| 128 | 1024 | 8 | 0.0316 | 0.0454 | 0.0948 | 0.697 | 0.0243 | 0.0093 | 0.1969 | **2.600** |
| 64 | 1024 | 16 | 0.0166 | 0.0529 | 0.0690 | 0.314 | 0.0122 | 0.0452 | 0.1530 | 0.269 |

### 2.4  The boundary shapes, re-measured on their own

The 42-shape ladder is one run per shape, and the boundary is decided by a
handful of shapes -- one of which, WSL `N = 1024, M = 64`, is where the two
EARLIER campaigns already disagreed with each other:

* `WAVE5_HYGIENE2_REPORT.md`'s WSL table reads separable 0.0100 against dense
  0.0118 there (dense LOSES), while
* `VERIFY_WAVE5_HYGIENE2.md`'s re-measurement reads dense 0.0185 against
  separable 0.0204 at the same shape (dense WINS).

A constant derived from a shape whose two prior measurements point opposite ways
is exactly what `docs/TESTING_STANDARDS.md` calls an S1 pin.  So
`validation/probe_c4_mft_direct/c4_boundary.py` re-measured the deciding shapes
on their own: THREE independent rounds of best-of-nine each, the load
snapshotted per round, the verdict taken on the WORST round.

| M/N | shape | WIN rounds (d/fb) | WIN worst | WSL rounds (d/fb) | WSL worst |
|---|---|---|---|---|---|
| 1/64 | 1024 -> 16 | 0.134 / 0.140 / 0.144 | 0.144 | 0.469 / 0.561 / 0.503 | 0.561 |
| 1/64 | 2048 -> 32 | 0.206 / 0.205 / 0.206 | 0.206 | 0.090 / 0.091 / 0.094 | 0.094 |
| 1/32 | 512 -> 16 | 0.206 / 0.182 / 0.203 | 0.206 | 0.495 / 0.514 / 0.518 | 0.518 |
| 1/32 | 1024 -> 32 | 0.266 / 0.248 / 0.256 | 0.266 | 0.877 / 0.954 / 0.844 | **0.954** |
| 1/32 | 2048 -> 64 | 0.397 / 0.392 / 0.477 | 0.477 | 0.163 / 0.172 / 0.168 | 0.172 |
| 1/16 | 512 -> 32 | 0.308 / 0.300 / 0.279 | 0.308 | 0.803 / 0.817 / 0.808 | 0.817 |
| 1/16 | 1024 -> 64 | 0.420 / 0.426 / 0.440 | 0.440 | 1.382 / 1.450 / 1.398 | **1.450 SLOWER** |
| 1/16 | 2048 -> 128 | 0.681 / 0.713 / 0.647 | 0.713 | 0.326 / 0.332 / 0.340 | 0.340 |
| 1/8 | 512 -> 64 | 0.615 / 0.608 / 0.597 | 0.615 | 1.612 / 1.525 / 1.563 | **1.612 SLOWER** |
| 1/8 | 1024 -> 128 | 0.846 / 0.837 / 0.728 | 0.846 | 2.596 / 2.041 / 2.982 | **2.982 SLOWER** |
| 1/8 | 2048 -> 256 | 1.421 / 1.427 / 1.278 | **1.427 SLOWER** | 0.663 / 0.676 / 0.639 | 0.676 |

**Verdict per ratio, worst of every round:**

| M/N | 1/64 | 1/32 | 1/16 | 1/8 |
|---|---|---|---|---|
| Windows py3.14 | 0.206 SAFE | **0.477 SAFE** | 0.713 SAFE | 1.427 NOT SAFE |
| WSL py3.12 | 0.561 SAFE | **0.954 SAFE** | 1.450 NOT SAFE | 2.982 NOT SAFE |

**1/32 is the intersection of the two builds' safe regions.**  The readings are
tight across rounds (the WSL 1/16 failure reads 1.382 / 1.450 / 1.398 and the
WIN 1/8 failure 1.421 / 1.427 / 1.278), so this is reproducible and not a
contention artefact.

**The one thin margin, stated.**  At 1/32 the worst reading is WSL
`N = 1024, M = 32` at 0.954 -- the dense route faster by 5 % in the worst round
and by 18 % in the best.  Everywhere else at 1/32 the margin is 2x to 6x.  That
is a shape where the separable route is unusually fast on Linux (7.2 ms against
7.5 ms; pocketfft's worker pool is not constrained by `OMP_NUM_THREADS`), and it
is the one place a different Linux build could flip the sign.  What it would
cost if it did is bounded: a few per cent of time at one shape family, against a
19x smaller memory peak and a 31x-485x closer answer.  **1/64 is the ratio with
a two-fold margin at every shape on both builds** (worst 0.561), and is the
value to move to if a deployment wants headroom.

### 2.5  Why a RATIO and not a clock

The TIME crossover is per-build -- the two builds' crossovers differ by an
octave, because scipy's pocketfft drives the separable route's 1-D passes
through its own worker pool on Linux (`SCIPY_FFT_WORKERS = -1`, which
`OMP_NUM_THREADS=1` does not constrain).  A constant read off ONE build's clock
is the shape `docs/TESTING_STANDARDS.md` calls S1, and it is the objection the
maintainer's original "not yet" rested on.  This rule is built around it: the
constant's VALUE is the intersection of the two builds' safe regions, and at run
time it is compared against a SHAPE and never against a timing.

---

## 3.  Memory -- the build-free half

`tracemalloc` peak, cold, MB.  Separate pass and separate invocation from the
timing.

| N | M | M/N | dense | separable | chirp-Z 2-D | dense smaller by | identical WIN/WSL |
|---|---|---|---|---|---|---|---|
| 2048 | 16 | 0.0078125 | 1.85 | 136.4 | 620.6 | 335x | NO |
| 1024 | 16 | 0.015625 | 0.93 | 34.5 | 158.0 | 170x | NO |
| 2048 | 32 | 0.015625 | 3.69 | 136.4 | 620.6 | 168x | NO |
| 512 | 16 | 0.03125 | 0.47 | 8.7 | 40.0 | 86x | NO |
| 1024 | 32 | 0.03125 | 1.85 | 34.7 | 159.6 | 87x | NO |
| 2048 | 64 | 0.03125 | 7.36 | 138.5 | 638.3 | 87x | NO |
| 256 | 16 | 0.0625 | 0.23 | 2.3 | 10.8 | 46x | NO |
| 512 | 32 | 0.0625 | 0.92 | 9.0 | 43.0 | 47x | NO |
| 1024 | 64 | 0.0625 | 3.68 | 35.7 | 168.7 | 46x | NO |
| 2048 | 128 | 0.0625 | 14.70 | 142.9 | 674.6 | 46x | NO |
| 128 | 16 | 0.125 | 0.12 | 0.7 | 2.6 | 22x | NO |
| 256 | 32 | 0.125 | 0.46 | 2.4 | 11.7 | 25x | NO |
| 512 | 64 | 0.125 | 1.84 | 9.5 | 46.7 | 25x | NO |
| 1024 | 128 | 0.125 | 7.35 | 37.8 | 186.8 | 25x | NO |
| 2048 | 256 | 0.125 | 29.38 | 151.1 | 746.9 | 25x | NO |
| 64 | 16 | 0.25 | 0.06 | 0.2 | 8.0 | 130x | NO |
| 128 | 32 | 0.25 | 0.23 | 0.7 | 3.2 | 14x | NO |
| 256 | 64 | 0.25 | 0.92 | 2.6 | 14.2 | 15x | NO |
| 512 | 128 | 0.25 | 3.68 | 10.5 | 56.7 | 15x | NO |
| 1024 | 256 | 0.25 | 14.69 | 42.0 | 226.7 | 15x | NO |
| 2048 | 512 | 0.25 | 58.74 | 167.9 | 906.3 | 15x | NO |
| 64 | 32 | 0.5 | 0.12 | 0.2 | 1.1 | 9x | NO |
| 128 | 64 | 0.5 | 0.46 | 0.8 | 4.4 | 10x | NO |
| 256 | 128 | 0.5 | 1.84 | 3.2 | 20.0 | 11x | NO |
| 512 | 256 | 0.5 | 7.35 | 12.6 | 79.8 | 11x | NO |
| 1024 | 512 | 0.5 | 29.37 | 50.4 | 319.0 | 11x | NO |
| 2048 | 1024 | 0.5 | 117.47 | 201.5 | 1275.4 | 11x | NO |
| 64 | 64 | 1 | 0.26 | 0.4 | 1.9 | 7x | NO |
| 128 | 128 | 1 | 1.05 | 1.3 | 8.7 | 8x | NO |
| 256 | 256 | 1 | 4.20 | 5.3 | 34.7 | 8x | NO |
| 512 | 512 | 1 | 16.78 | 21.0 | 138.5 | 8x | NO |
| 1024 | 1024 | 1 | 67.11 | 84.0 | 553.9 | 8x | NO |
| 64 | 128 | 2 | 0.66 | 1.1 | 4.3 | 7x | NO |
| 128 | 256 | 2 | 2.62 | 3.7 | 19.2 | 7x | NO |
| 256 | 512 | 2 | 10.49 | 14.7 | 76.6 | 7x | NO |
| 512 | 1024 | 2 | 41.95 | 58.8 | 306.4 | 7x | NO |
| 64 | 256 | 4 | 1.84 | 2.9 | 27.0 | 15x | NO |
| 128 | 512 | 4 | 7.34 | 11.6 | 54.7 | 7x | NO |
| 256 | 1024 | 4 | 29.36 | 46.2 | 218.4 | 7x | NO |
| 64 | 512 | 8 | 5.77 | 10.0 | 45.8 | 8x | NO |
| 128 | 1024 | 8 | 23.07 | 39.9 | 182.7 | 8x | NO |
| 64 | 1024 | 16 | 19.92 | 36.8 | 166.7 | 8x | NO |

**The dense route is the cheapest of the three at 42 of 42 shapes on BOTH
builds, by 6.4x to 334.9x, and the two builds' readings are IDENTICAL TO THE
BYTE at every shape** (the `identical WIN/WSL` column is `yes` at 42 of 42).
The full ordering `dense < separable < chirp-Z 2-D` also holds at 42 of 42 on
both builds.  This is build-free by construction: it follows from the padding
(`L = next_fast_len(N + M - 1)` per axis) and not from a timing.

So the memory half never argues against the rule anywhere -- including in the
region 1/32 excludes, which is why the boundary is set by time alone.

---

## 4.  Accuracy

### 4.1  The reference, and why it has to be this one

A reference that forms `t = alpha*n*k` in float64 and then reduces it commits
the same two roundings the dense route commits, so it agrees with the dense
route BY CONSTRUCTION and reads 3e-16 at every budget.  That is a measurement of
the instrument, and it is the defect `VERIFY_WAVE5_HYGIENE2_ROUND2.md` raised as
D-1.

The reference here reduces the phase EXACTLY: `alpha` is a float64 and therefore
an exact rational and `n`, `k` are integers, so in `fractions.Fraction` the
product and its fractional part are exact and the single float64 rounding lands
on a number already inside `[-1/2, 1/2)`.  The double sum is accumulated with
`math.fsum`.  The reference is therefore correctly rounded in BOTH senses.  Its
own exactness is gated two-sidedly, in the probe and in the shipped test: at a
DYADIC `alpha` the exact reduction and the route's `t - rint(t)` must agree to
the bit (measured: max |frac difference| exactly 0.0), and at a budget of 1e12
they must PART (measured: 4.0e-01), or the reference is measuring itself.

### 4.2  The DERIVED bar, and the mechanism it exposes

Each output point is a sum of `n = Ny*Nx` unit-modulus terms, so a route's
absolute error against a correctly-rounded reference has exactly two sources:

* **summation** -- a summation of growth factor `g` commits at most
  `g * eps * sum|E|`; `g = 3*log2(L^2)` for a chirp-Z route, `g = sqrt(n)` for
  the dense route's two BLAS products, `g = 1` for the `fsum` reference;
* **phase** -- the route's phase argument `t` is a float64 product that has
  already lost its low bits, costing `~eps*|t|` of phase and therefore
  `2*pi*eps*max|t|*sum|E|` absolute.

`bar = (g_route + 1 + 2*pi*max|t|) * eps * sum|E|`, absolute, so the
apples-to-apples reading against it is the max-abs departure.

**And `max|t|` is NOT the same for the two routes.**  The chirp-Z route builds
`exp(sign*i*pi*alpha*n^2)` with `n` running to `N_max = max(N, M)`, so it spends
`alpha * N_max^2` -- the phase budget the guard is named for.  The dense route
builds `exp(sign*2*pi*i*alpha*n*k)` with `n < N` and `k < M`, so it spends only
`alpha * (N-1) * (M-1)`.  At `M = N/32` that is 32 times smaller.

**That is the finding of this section**, and it says why the accuracy advantage
is biggest exactly where the rule fires -- which hygiene-2 could not see,
because its ladders were all at `M ~ N/2` where the two `max|t|` are within a
factor of 2 of each other.

### 4.3  The shape ladder, 16 shapes x 2 budgets, both builds

Eight shapes the rule sends to the dense route and eight it leaves on the
chirp-Z route, at two budgets: 1 (where the phase term is of the same order as
the summation term, so the reading is essentially the SUMMATION comparison) and
1e3 (six decades under the guard, of the order a real MFT grid spends, where the
phase term dominates).  `gap` is chirp-Z relative L2 over dense relative L2;
`room` is decades between the measured max-abs and the route's own derived bar.

| budget | N | M | M/N | side | chirp rel (WIN / WSL) | dense rel (WIN / WSL) | gap | `auto` is | room chirp | room dense |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 224 | 7 | 0.03125 | **dense** | 4.322e-16 / 5.300e-16 | 5.043e-16 / 4.995e-16 | 0.86 | `direct` | 3.46 | 3.87 |
| 1 | 64 | 2 | 0.03125 | **dense** | 2.866e-16 / 2.667e-16 | 3.769e-16 / 3.769e-16 | 0.76 | `direct` | 3.05 | 3.02 |
| 1 | 96 | 3 | 0.03125 | **dense** | 2.999e-16 / 3.412e-16 | 4.562e-16 / 4.571e-16 | 0.66 | `direct` | 3.15 | 3.34 |
| 1 | 128 | 4 | 0.03125 | **dense** | 7.248e-16 / 1.269e-15 | 9.982e-16 / 9.787e-16 | 0.73 | `direct` | 3.31 | 3.66 |
| 1 | 160 | 5 | 0.03125 | **dense** | 4.747e-16 / 4.300e-16 | 6.298e-16 / 6.385e-16 | 0.75 | `direct` | 3.20 | 3.56 |
| 1 | 192 | 6 | 0.03125 | **dense** | 1.335e-15 / 8.622e-16 | 1.363e-15 / 1.346e-15 | 0.98 | `direct` | 3.27 | 3.82 |
| 1 | 256 | 8 | 0.03125 | **dense** | 7.946e-16 / 6.641e-16 | 6.884e-16 / 6.899e-16 | 1.15 | `direct` | 3.29 | 3.88 |
| 1 | 128 | 2 | 0.01562 | **dense** | 9.202e-16 / 8.349e-16 | 6.828e-16 / 6.111e-16 | 1.35 | `direct` | 3.37 | 3.87 |
| 1 | 32 | 2 | 0.0625 | chirp | 4.745e-16 / 5.227e-16 | 1.967e-16 / 1.967e-16 | 2.41 | `bluestein` | 2.40 | 2.78 |
| 1 | 64 | 4 | 0.0625 | chirp | 5.091e-16 / 6.409e-16 | 4.522e-16 / 4.609e-16 | 1.13 | `bluestein` | 3.07 | 3.15 |
| 1 | 96 | 6 | 0.0625 | chirp | 4.187e-16 / 4.292e-16 | 4.517e-16 / 4.192e-16 | 0.93 | `bluestein` | 2.91 | 3.24 |
| 1 | 128 | 8 | 0.0625 | chirp | 7.590e-16 / 6.541e-16 | 8.006e-16 / 8.144e-16 | 0.95 | `bluestein` | 2.94 | 3.18 |
| 1 | 64 | 8 | 0.125 | chirp | 5.389e-16 / 5.637e-16 | 5.883e-16 / 5.847e-16 | 0.92 | `bluestein` | 2.89 | 3.05 |
| 1 | 24 | 12 | 0.5 | chirp | 2.104e-15 / 2.104e-15 | 1.880e-15 / 1.914e-15 | 1.12 | `bluestein` | 2.42 | 2.36 |
| 1 | 32 | 16 | 0.5 | chirp | 1.101e-15 / 1.038e-15 | 8.123e-16 / 8.062e-16 | 1.36 | `bluestein` | 2.38 | 2.50 |
| 1 | 48 | 24 | 0.5 | chirp | 8.905e-16 / 7.708e-16 | 7.160e-16 / 7.231e-16 | 1.24 | `bluestein` | 2.59 | 2.71 |
| 1000 | 224 | 7 | 0.03125 | **dense** | 1.541e-13 / 1.541e-13 | 4.992e-15 / 5.003e-15 | 30.88 | `direct` | 2.89 | 3.06 |
| 1000 | 64 | 2 | 0.03125 | **dense** | 4.667e-14 / 4.649e-14 | 2.004e-16 / 1.676e-16 | 232.87 | `direct` | 2.97 | 3.66 |
| 1000 | 96 | 3 | 0.03125 | **dense** | 1.656e-13 / 1.655e-13 | 3.816e-15 / 3.778e-15 | 43.39 | `direct` | 2.61 | 2.75 |
| 1000 | 128 | 4 | 0.03125 | **dense** | 1.483e-13 / 1.485e-13 | 3.942e-16 / 4.637e-16 | 376.29 | `direct` | 2.71 | 3.90 |
| 1000 | 160 | 5 | 0.03125 | **dense** | 2.458e-13 / 2.459e-13 | 5.069e-16 / 5.051e-16 | 484.85 | `direct` | 2.50 | 4.12 |
| 1000 | 192 | 6 | 0.03125 | **dense** | 2.124e-13 / 2.124e-13 | 6.715e-15 / 6.731e-15 | 31.63 | `direct` | 2.75 | 2.97 |
| 1000 | 256 | 8 | 0.03125 | **dense** | 1.825e-13 / 1.824e-13 | 7.479e-16 / 7.398e-16 | 244.02 | `direct` | 2.71 | 4.11 |
| 1000 | 128 | 2 | 0.01562 | **dense** | 9.666e-14 / 9.712e-14 | 6.611e-16 / 8.601e-16 | 146.21 | `direct` | 3.10 | 3.85 |
| 1000 | 32 | 2 | 0.0625 | chirp | 6.315e-14 / 6.315e-14 | 1.083e-16 / 1.521e-16 | 583.07 | `bluestein` | 2.48 | 3.86 |
| 1000 | 64 | 4 | 0.0625 | chirp | 2.241e-13 / 2.241e-13 | 4.911e-16 / 4.766e-16 | 456.34 | `bluestein` | 2.43 | 3.75 |
| 1000 | 96 | 6 | 0.0625 | chirp | 1.328e-13 / 1.327e-13 | 1.180e-14 / 1.182e-14 | 11.26 | `bluestein` | 2.53 | 2.30 |
| 1000 | 128 | 8 | 0.0625 | chirp | 1.851e-13 / 1.851e-13 | 6.971e-16 / 6.612e-16 | 265.51 | `bluestein` | 2.56 | 3.91 |
| 1000 | 64 | 8 | 0.125 | chirp | 1.438e-13 / 1.438e-13 | 5.542e-16 / 5.123e-16 | 259.47 | `bluestein` | 2.39 | 3.78 |
| 1000 | 24 | 12 | 0.5 | chirp | 1.719e-13 / 1.719e-13 | 9.813e-14 / 9.813e-14 | 1.75 | `bluestein` | 1.89 | 1.54 |
| 1000 | 32 | 16 | 0.5 | chirp | 2.007e-13 / 2.007e-13 | 3.350e-16 / 3.256e-16 | 599.14 | `bluestein` | 1.84 | 4.27 |
| 1000 | 48 | 24 | 0.5 | chirp | 1.984e-13 / 1.984e-13 | 8.787e-14 / 8.787e-14 | 2.26 | `bluestein` | 2.01 | 2.02 |

**Readings.**

* `all_inside_bars`: **true at 32 of 32 rows on both builds**, minimum room
  **1.54 decades** (WIN 1.5417, WSL 1.5417).
* `'auto'` matched the route the rule names at **32 of 32** rows on both builds;
  rows where it matched NEITHER named route: **0**.
* At a budget of 1e3 on the DENSE side the gap is **30.9x to 484.9x** (WIN) and
  **30.8x to 486.9x** (WSL).
* At a budget of 1 the gap is **0.66x to 1.35x** on the dense side -- i.e. at
  the pure summation floor the two routes are COMPARABLE and the dense one is
  not always the nearer.  Saying "the dense route is the more accurate route" is
  a statement about the PHASE, not about the summation, and this ladder is where
  that distinction is visible.

### 4.4  The budget ladder -- hygiene-2 round 3 reproduces, and the law holds on both routes

`N = 24 -> M = 12`, the shipped fixture's own geometry, non-centred, against the
exact-phase reference.  **Identical to the digit on both builds.**

| budget | chirp-Z rel | dense rel | eps*budget | C_chirp | C_dense | gap | warned chirp / dense / auto |
|---|---|---|---|---|---|---|---|
| 1e+05 | 2.7723e-11 | 6.8712e-12 | 2.220e-11 | 1.2485 | 0.3095 | 4.035 | 0 / 0 / 0 |
| 1e+07 | 1.8794e-09 | 9.4507e-10 | 2.220e-09 | 0.8464 | 0.4256 | 1.989 | 0 / 0 / 0 |
| 1e+09 | 1.6351e-07 | 1.1043e-07 | 2.220e-07 | 0.7364 | 0.4973 | 1.481 | 0 / 0 / 0 |
| 1e+10 | 1.6058e-06 | 8.8822e-07 | 2.220e-06 | 0.7232 | 0.4000 | 1.808 | 1 / 0 / 1 |
| 1e+11 | 1.2312e-05 | 7.1421e-06 | 2.220e-05 | 0.5545 | 0.3217 | 1.724 | 1 / 0 / 1 |
| 1e+12 | 1.8648e-04 | 8.6213e-05 | 2.220e-04 | 0.8398 | 0.3883 | 2.163 | 1 / 0 / 1 |
| 1e+14 | 1.0841e-02 | 6.8636e-03 | 2.220e-02 | 0.4882 | 0.3091 | 1.580 | 1 / 0 / 1 |
| 1e+15 | 2.2850e-01 | 6.7390e-02 | 2.220e-01 | 1.0291 | 0.3035 | 3.391 | 1 / 0 / 1 |

Fitted slopes: **chirp-Z 0.9632, dense 0.9917** -- round 3 published 0.96 and
0.99.  Its four published rungs (1e5 / 1e9 / 1e12 / 1e15: chirp 2.8e-11 /
1.6e-07 / 1.9e-04 / 2.3e-01 against dense 6.9e-12 / 1.1e-07 / 8.6e-05 /
6.7e-02) reproduce to the digit, as does its "1.5 .. 4.0 on that convention"
gap: measured **1.481 .. 4.035**.

**The same ladder at a shape the rule SENDS to the dense route** (`N = 96 ->
M = 3`), which is the new reading:

| budget | chirp-Z rel | dense rel | C_chirp | C_dense | gap | warned chirp / dense / auto |
|---|---|---|---|---|---|---|
| 1e+05 | 1.5315e-11 | 2.6201e-13 | 0.6897 | 0.0118 | 58.5 | 0 / 0 / 0 |
| 1e+07 | 1.9100e-09 | 2.4865e-11 | 0.8602 | 0.0112 | 76.8 | 0 / 0 / 0 |
| 1e+09 | 1.2992e-07 | 3.5353e-09 | 0.5851 | 0.0159 | 36.8 | 0 / 0 / 0 |
| 1e+10 | 1.8513e-06 | 4.1519e-08 | 0.8337 | 0.0187 | 44.6 | 1 / 0 / **1 (names dense)** |
| 1e+11 | 1.7641e-05 | 2.4895e-07 | 0.7945 | 0.0112 | 70.9 | 1 / 0 / **1 (names dense)** |
| 1e+12 | 1.8771e-04 | 3.6611e-06 | 0.8454 | 0.0165 | 51.3 | 1 / 0 / **1 (names dense)** |
| 1e+14 | 1.6916e-02 | 3.0211e-04 | 0.7618 | 0.0136 | 56.0 | 1 / 0 / **1 (names dense)** |
| 1e+15 | 2.3035e-01 | 2.3047e-03 | 1.0374 | 0.0104 | 99.9 | 1 / 0 / **1 (names dense)** |

Fitted slopes **1.0051 (chirp-Z) and 1.0001 (dense)**: BOTH routes follow
`rel ~ eps * budget`, which is round 2's D-1 correction, re-measured here on the
region the new default captures.  What differs is the constant, and the constant
ratio is the `max|t|` ratio of section 4.2.

### 4.5  A degenerate fixture the probe caught, and kept as a control

The first dense-side budget geometry tried was `N = 64 -> M = 2`, and it read
1.9e-16 at EVERY budget -- apparent immunity, with a gap of 4.3e+14 at 1e15.
The cause is that `alpha = budget / 64^2` is an exact integer at every
power-of-ten budget on the ladder (`1e15 / 4096 = 244140625000`), so `t =
alpha*n` is an exact whole number of turns, `t - rint(t)` is exactly 0, and the
EXACT reference computes exactly 0 too.  The fixture agrees by construction --
the same shape as the D-1 defect one level down.  It is kept in
`c4_accuracy.py` as `dense_side_64_2_DEGENERATE`, labelled, with the claim read
from `dense_side_96_3` instead (`1e15 / 9216` is not an integer).

---

## 5.  The selection is a function of shape alone

`validation/probe_c4_mft_direct/c4_rule.py`, 30 shapes, both builds.

**PURITY of the decision.**  `_auto_selects_direct` is called with the process
perturbed between calls in every way that is not a shape: the wall clock
advanced; `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
`SCIPY_FFT_WORKERS` / `LUMENAIRY_MEM_BUDGET_MB` / `LUMENAIRY_USE_SCIPY_FFT` set
high, set low and removed; the library's caches filled and dropped; the RNG
advanced; `fft_infra.USE_SCIPY_FFT` and `SCIPY_FFT_WORKERS` flipped both ways;
and the call repeated on another thread.

| build | shapes giving ONE answer | of |
|---|---|---|
| WIN-py3.14 | **30** | 30 |
| WSL-py3.12 | **30** | 30 |

**THE FALSIFIER.**  A perturbation set that could not change anything is not
evidence, so the one thing that DOES move the answer is exercised:

| `_MFT_DIRECT_MAX_RATIO` | shapes taking the dense route |
|---|---|
| `_MFT_DIRECT_NEVER` (0.0) | **0** of 30 |
| as shipped (1/32) | **15** of 30 |
| `_MFT_DIRECT_ALWAYS` (inf) | **30** of 30 |

**NO THIRD ARITHMETIC.**  At every shape small enough to drive (2 primitives x 2
`separable` settings = 88 cases), `'auto'`'s bytes were compared to BOTH named
routes' bytes:

| build | matched the route the rule names | matched NEITHER (third arithmetic) | route stable under perturbation |
|---|---|---|---|
| WIN-py3.14 | **88 / 88** | **0** | **88 / 88** |
| WSL-py3.12 | **88 / 88** | **0** | **88 / 88** |

The `matched: NONE` column is the one that matters, and it is what a centred
`'auto'` would have produced had it gone through the pre-chirp / post-chirp /
constant decomposition and then into the dense core.  It is zero because
`_bluestein_centred_2d` takes the dense arm BEFORE that decomposition, for both
`'direct'` and an `'auto'` the rule sends there.

**What DOES move, recorded rather than asserted away.**  `'auto'`'s BITS moved
under exactly one perturbation, `scipy_fft_off` -- 26 of 88 cases on Windows and
14 of 88 on WSL, all of them on the chirp-Z side.  That is the chirp-Z route
dispatching its FFTs through `fft_infra.USE_SCIPY_FFT`, which it has always
done; it is pre-existing and is not a property of this rule.  The ROUTE never
moved.

---

## 6.  Byte identity, archive to archive

`validation/probe_c4_mft_direct/c4_bitid.py` against a read-only
`git archive 49ddf4bd` extraction at `C:\tmp\lum_c4_base`, run from the probe
directory with `PYTHONPATH` naming ONE tree and `lumenairy.__file__` asserted
under it.  **562 keys on the branch, 419 on the base** (the base has no
`_MFT_DIRECT_MAX_RATIO`, so it produces no `never/*` group); digests are SHA-256
over the raw bytes plus dtype, shape and every warning in emission order -- no
float `==` appears anywhere in the comparison.

Fixtures: the two primitives at 12 shapes (including three ANISOTROPIC ones with
one axis under the boundary and one over) x 2 signs x 2 `separable` settings x 3
centre conventions; the three public propagators at 5 shapes on-axis and
off-axis; `resample_field(method='chirpz')` at 3 shapes; the carrier exact-focus
readout at 2.

| claim | group | WIN identical | WSL identical | differing | only-base | only-branch |
|---|---|---|---|---|---|---|
| the previous route named explicitly | `method='bluestein'` | **138 / 138** | **138 / 138** | 0 | 0 | 0 |
| the previous route named explicitly | `method='separable'` | **138 / 138** | **138 / 138** | 0 | 0 | 0 |
| the constant set to never | `_MFT_DIRECT_NEVER` | **138 / 138** | **138 / 138** | 0 | 0 | 0 |
| NO keyword (the split) | default | 81 / 138 | 81 / 138 | **57** | 0 | 0 |
| `resample_field(method='chirpz')` | default | 2 / 3 | 2 / 3 | 1 | 0 | 0 |
| carrier exact-focus readout | default | 1 / 2 | 1 / 2 | 1 | 0 | 0 |

**The split is CHECKED, not merely counted.**  `c4_bitid.py` records what
`_auto_selects_direct` answers for each fixture's shape beside the digest, and
`c4_compare.py` asserts that a key differs base-to-branch exactly when the rule
says `'direct'`:

| build | keys agreeing with the rule | disagreeing | unclassified |
|---|---|---|---|
| WIN-py3.14 | **138** | **0** | 0 |
| WSL-py3.12 | **138** | **0** | 0 |

**COUNTS.**  Of 138 no-keyword fixtures, **81 are byte-identical to `49ddf4bd`
with no keyword at all** (the rule leaves them on the previous route) and **57
move** (the rule sends them to the dense route).  With the previous route's
keyword passed explicitly, **all 138 are byte-identical on each of the two
arms**, on both builds.  With `_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`, **all
138 are byte-identical**, on both builds.

**The dense-selected shapes are NOT pinned anywhere.**  They are compared to the
exact-phase reference with a derived bar (section 4), because the dense route's
two products go through BLAS and its last bits move with the kernel.  Nothing in
this branch records a dense-route byte as an expected value.

---

## 7.  Every backend

`validation/probe_c4_mft_direct/c4_backends.py`.  Premise-gated, never skipped:
a backend that is not installed, or a device this box cannot reach, is RECORDED
with the exception it raised.

**The SELECTION is namespace-free** -- it reads four Python ints, so it cannot
depend on the array library:

| backend | build | selection matches NumPy's |
|---|---|---|
| JAX | WIN-py3.14 | **30 / 30** |
| JAX | WSL-py3.12 | **30 / 30** |
| CuPy | WIN-py3.14 | **30 / 30** |
| CuPy | WSL-py3.12 | premise: `ModuleNotFoundError: No module named 'cupy'` |

**JAX dispatch**, driven through the primitives with `xp=jax.numpy`, x64 on,
8 shapes on both sides of the boundary:

| build | dispatched as the rule says | `jit` agrees with eager | inside the derived bar |
|---|---|---|---|
| WIN-py3.14 | **8 / 8** | **8 / 8** | **8 / 8** |
| WSL-py3.12 | **8 / 8** | **8 / 8** | **8 / 8** |

`jit` is the arm that matters: a traced call has no concrete array to look at,
and `_auto_selects_direct` takes Python ints from `E.shape`, which stay static
under a trace.  The comparison against NumPy is against a DERIVED bar
(`2*sqrt(n)*eps*sum|E|`) and never bit for bit -- a different device's BLAS is
entitled to different last bits.

**CuPy, and a capability gained.**  This box's cuFFT DLL is broken
(`ImportError: DLL load failed while importing cufft`), which is the reason
`tests/unit/test_niche_k2_carrier_backends.py` skips its propagating CuPy arms.
The dense route needs no FFT, so:

| shape | rule says | 5.48.1 (`49ddf4bd`) | this branch | vs NumPy | derived bar |
|---|---|---|---|---|---|
| 64 -> 2 | dense | `ImportError: cufft` | **runs** | 7.418e-14 | 1.442e-10 |
| 128 -> 4 | dense | `ImportError: cufft` | **runs** | 1.798e-13 | 1.163e-09 |
| 256 -> 8 | dense | `ImportError: cufft` | **runs** | 7.123e-13 | 9.322e-09 |
| 96 -> 3 | dense | `ImportError: cufft` | **runs** | 1.421e-13 | 4.885e-10 |
| 64 -> 8 | chirp | `ImportError: cufft` | `ImportError: cufft` | -- | -- |
| 24 -> 12 | chirp | `ImportError: cufft` | `ImportError: cufft` | -- | -- |
| 48 -> 24 | chirp | `ImportError: cufft` | `ImportError: cufft` | -- | -- |
| 32 -> 16 | chirp | `ImportError: cufft` | `ImportError: cufft` | -- | -- |

Two-sided: the four shapes the rule captures now run and the four it does not
still raise, on both trees.  `'auto'` returned the dense bytes at all four
(`auto_is_dense_bits: true`).  The base-tree readings are reproducible with
`validation/probe_c4_mft_direct/c4_base_cupy_contrast.py`, which carries its own
tree assert -- the first attempt at that measurement, run as a stdin script from
inside the worktree, silently bound the WORKTREE's lumenairy because `''`
precedes `PYTHONPATH` on `sys.path`, and printed `has rule: True` about what was
supposed to be the base.

---

## 8.  Blast radius

### 8.1  What was selected

Two selections, kept separate because they answer different questions.

**The grep selection** -- `grep -rln "mft|bluestein|chirp" tests/` -- gives 55
files and 3030 ids.  It is deliberately over-broad: a file that only mentions
"chirp" in a comment is in it, which is the point of a blast radius.

**The internal consumers**, found by grepping `lumenairy/` for callers rather
than by reading the docs.  Every one of these reaches the rule, because none of
them passes `method=`:

| caller | what it drives | what its ratio is |
|---|---|---|
| `mft.py:530` `angular_spectrum_propagate_mft` | `_bluestein_centred_2d` | the caller's `N_out / N_in` |
| `mft.py:1155` `fresnel_propagate_mft` | `_bluestein_centred_2d` | the caller's |
| `mft.py:1403` `fraunhofer_propagate_mft` | `_bluestein_centred_2d` | the caller's |
| `mft.py:590` `_resample_field_chirpz` (the `resample_field(method='chirpz')` leg) | `_bluestein_centred_2d` | the resample's decimation factor |
| `carrier.py:2498` `_collins_transport` | `_bluestein_centred_2d` | fine grid -> readout window; the SMALLEST ratios in the library |
| `carrier.py:4479`, `carrier.py:7197`, `carrier_field.py:1460` | `angular_spectrum_propagate_mft` | the caller's |
| `dispatch.py:940 / 944 / 948 / 1501` | the three propagators | `N_out` defaults to `N_in`, i.e. ratio 1 |
| `system.py:929` | `fresnel_propagate_mft` | the leg's |
| `analysis/psf_mtf_otf.py:513` | `fraunhofer_propagate_mft` | `N_psf / N_pupil` |
| `ui/waveoptics_dock.py:1278 / 1282 / 1286` | the three propagators | the dock's |

### 8.2  The grep sweep, run

`55 files, 3030 ids, Windows py3.14`: **5 failed, 3019 passed, 6 skipped,
151 warnings in 2249.78 s**.

Every one of the five was then run against a REAL `git worktree` at `49ddf4bd`
-- not the `git archive` extraction, which is not a git repository and makes one
of them fail for an environment reason that says so:

| id | on the branch | on the `49ddf4bd` worktree | verdict |
|---|---|---|---|
| `test_audit2609_a9_ui.py::test_u6i_workers_honour_request_interruption` | FAIL | FAIL | pre-existing, environmental |
| `test_audit2609_a9_ui.py::test_u7_file_new_keeps_display_preferences` | FAIL | FAIL | pre-existing, environmental |
| `test_audit2609_a9_ui.py::test_u7_matplotlib_is_not_imported_by_the_dock_modules` | FAIL | FAIL | pre-existing (`harness itself pulled matplotlib`) |
| `test_v4_15_agent_e.py::TestUI6and7PsfMtfDockRayAccumulation::test_no_last_write_wins` | FAIL | FAIL | pre-existing, environmental |
| `test_v5_3_2_walker_source_line_citation.py::test_v18_5_...` | FAIL | **PASS** | **MINE** |

The fifth is the only one WP-C4 caused, and it is not a numerical finding: the
three `method=` docstring blocks in `mft.py` grew, which moved every line the
5.47.0 CHANGELOG block cites below them.  V18 itself still passed -- each stale
citation landed on SOME real line -- which is the blind spot V18.5 exists for.
Re-anchored by CONTENT with the tool the failure names
(`scripts/reanchor_citations.py --base f4f18851 --block "[5.47.0]"`), in the
same commit as the change, twice: once for the docstring rewrite (6 citations,
27-54 lines) and once for the version-strip that followed (the same 6, 1-2
lines).  11 passed, 1 skipped afterwards.

**One more gate fired, and it was right.**
`test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached`
counted 16 lines in `_bluestein.py` and `mft.py` naming `v5.49.0` against
`lumenairy.__version__ = 5.48.1`.  Every one was rephrased rather than deleted
("Before the shape rule, the guard sat ..." for "Before 5.49.0 ..."), so no
statement is lost and none depends on a number.  9 passed afterwards.

### 8.3  Does the rule ever FIRE in the existing suite?

"Nothing failed" has two very different explanations -- the rule never fires at
the shapes the suite drives, or it fires and nothing notices -- and only a
measurement separates them.  `c4_ratio_census_plugin.py` wraps
`_auto_selects_direct` for a whole pytest session and records every shape it is
asked about, changing no answer.

Over the 33 MFT-focused files of the grep selection (1263 ids, Windows,
2680.20 s):

| quantity | reading |
|---|---|
| calls to the rule | **1572** |
| calls answering `'direct'` | **35** |
| distinct shapes asked about | 83 |
| distinct ratios asked about | 42 |
| smallest ratio seen | **1/512** (0.001953125) |
| ratios at or under the boundary | 1/512, 1/256, 1/128, 1/64, 1/51.2, 1/32 |
| ids in which the rule fires | **16**, in 5 files |

So the rule DOES fire in the shipped suite, at 16 ids:

| file | ids where the rule fires | direct calls | shapes it fires at | smallest ratio |
|---|---|---|---|---|
| `test_audit2609_a25_carrier_focus_readout.py` | 2 | 2 | 512x512 -> 16x16 | 1/32 |
| `test_audit2609_b4_collins_transport.py` | 2 | 5 | 1024x1024 -> 16x16 | 1/64 |
| `test_niche_audit_w9_dispatch2.py` | 3 | 9 | 1024 -> 8, 2048 -> 8, 4096 -> 8 | 1/512 |
| `test_niche_d2_chain_multi.py` | 7 | 17 | 1024 -> 8, 1024 -> 16 | 1/128 |
| `test_niche_tight_focus_readout.py` | 2 | 2 | 2048x2048 -> 40x40 | 1/51.2 |

All five are carrier / Collins readouts and the multi-congruence chain -- i.e.
exactly the "read a small window out of a large fine grid" workflow the rule was
derived for -- and **all 1263 ids passed**.

**The same census on WSL agrees exactly.**  Over the same files minus `b4`
(which is not in the WSL selection) plus WP-C4's own test file: 1669 calls, 193
answering `'direct'`.  Subtracting this branch's own file (388 calls, 158
`'direct'`, 19 ids) leaves **1281 calls and 35 `'direct'`** in shipped tests --
the SAME 35, in the same 14 ids of the same four files.  Which shapes reach the
rule, and what it answers, is identical across the two builds, as a rule that
reads four integers must be.  **1158 passed** on that run.

### 8.4  The two-sided arm: the same ids under BOTH routes

A green run proves the assertions are satisfied; it does not by itself prove
they are not coupled to which route ran.  The five firing files (159 ids, the
`b4` file narrowed to its two firing classes) were therefore run twice on
Windows, once with the shipped constant and once with
`_MFT_DIRECT_MAX_RATIO = _MFT_DIRECT_NEVER`:

| arm | ids | result | rule calls | answering `'direct'` |
|---|---|---|---|---|
| shipped (1/32) | 159 | **159 passed** in 1185.13 s | 345 | **35** |
| forced `_MFT_DIRECT_NEVER` | 159 | **159 passed** in 1181.38 s | 380 | **0** |

The same 159 ids are green on both routes.  **No shipped assertion is coupled to
which route ran**, and the forced arm confirms the constant does what its name
says.

(The call counts differ, 345 against 380, for a structural reason worth
recording: on the dense arm `_bluestein_centred_2d` returns BEFORE delegating to
`_bluestein_2d`, so one call is made where the chirp path makes two.  That is
the "no third arithmetic" property showing up in a census.)

### 8.5  The classification, and the count that matters

The brief asks for every moved test classified into three buckets.  The measured
answer is that **all three buckets are empty**, and the census is what makes
that a finding rather than an absence:

| bucket | count | evidence |
|---|---|---|
| pins the separable route's last bits -- would need re-deriving against the exact reference | **0** | 159 ids green on both routes (8.4) |
| a fixture now closer to the reference -- would need restating from its own ladder | **0** | same |
| a genuine contract on the separable route -- would need `method=` passed explicitly with a reason | **0** | same |
| **moved and needed no change** | **16** | they fire (8.3) and pass on both routes (8.4) |
| failed and were caused by WP-C4 | **1** | the citation walker, a line-number drift, re-anchored with the sanctioned tool |
| failed and were pre-existing | **4** | all four reproduce on the `49ddf4bd` worktree |

**No bar was loosened anywhere.**  The only test assertions this branch writes
are in its own new file, and their bars are derived (section 10).


---

## 9.  The one behaviour change that is not a byte move

The chirp phase-budget guard moved into `_warn_phase_budget` and is now
evaluated by `'auto'` BEFORE it chooses a route.

**Why.**  Before 5.49.0 the guard sat between the `method='direct'` early return
and the chirp-Z arms, so only a chirp-Z call could reach it.  With `'auto'` able
to choose the dense route from the shapes, leaving the guard where it was would
have meant a caller who was being warned at a high budget goes SILENT on a shape
the new rule captures -- a diagnostic removed by a default flip, which is the one
thing a default flip may not do.

**What is unchanged.**  An explicit `method='direct'` is still silent.  That is
the shipped 5.48 decision (warning on the one route the warning's own advice
names would be a false positive) and it stays gated two-sidedly by
`test_wave5_h2_mft_direct.py::test_the_chirp_phase_guard_fires_on_the_chirp_route_and_not_the_dense_one`.
The THRESHOLD has not moved.  `on_dense=False` reproduces the pre-5.49.0 message
BYTE FOR BYTE, which is what keeps the byte-identity claim true for the fixtures
that warn (16 of the 562 keys emit a warning).

**What the message says on the new arm.**  The old tail advised
`method='direct'`, which is absurd when the call is already on it; the message
now names the route taken, says the rule chose it from the shapes, repeats that
the dense route is better only by a bounded factor, and names
`method='bluestein'` / `'separable'` as the chirp-Z routes.  Measured in section
4.4: `auto_message_names_dense` is true at every warned rung on the dense-side
geometry and the explicit-`direct` column is 0 at every rung.

**Still open and still the maintainer's**: whether an explicit `method='direct'`
should also warn (the second half of `WAVE5_HYGIENE2_REPORT.md`'s decision
item 2).  Nothing here changes it.

---

## 10.  Tests

`tests/unit/test_c4_mft_direct_default.py`, 21 ids, 4.5 s in total, 1.1 s at the
slowest -- every id far inside the 60 s cap.

Five claims, each written as a callable helper so the mutation matrix exercises
the SAME assertion the release carries rather than a paraphrase:

1. the `method` default is `'auto'` on both primitives and all three public
   entry points, read off the SIGNATURE;
2. the boundary is asked FROM the constant -- no number is typed; what is
   asserted is the RELATION (the ratio `1/n` at the constant is inside, `2/n` is
   not, one axis past it is enough to refuse) plus the documented settings
   (`inf` = always, `0.0` / negative / `nan` = never);
3. `'auto'` returns the bytes of the route the rule names, at shapes on BOTH
   sides, on both primitives, on both `separable` settings -- with
   `matched is None` as the third-arithmetic assertion;
4. the way back is byte-identical, asserted on the DENSE side where it is a
   claim;
5. on the dense side the dense route is the closer one to the exact-phase
   reference, by at least a DERIVED margin.

**The accuracy bar is derived and has a gap on both sides.**
`R = max|t|_chirp / max|t|_dense = N_max^2 / ((N-1)(M-1))` comes from the two
kernels and `alpha` cancels, so it is a property of the shapes.  The measured
constant ratio `C_chirp/C_dense` is 1.481 .. 4.035 over ten decades of budget
(section 4.4), so the bar asserted is `R/4`.

| shape | R/4 (the bar) | measured gap | clear by |
|---|---|---|---|
| 96 -> 3 | 12.1 | 51 | 4.2x |
| 128 -> 4 | 10.8 | 130 | 12x |
| 256 -> 8 | 9.2 | 244 | 26x |
| 64 -> 2 | 16.3 | 233 | 14x |
| an impostor dense arm | 12.1 | **~1.0** | 12x UNDER |

Asserting only `dense < chirp` would have had no lower gap: the two chirp-Z arms
differ from each other only in round-off, so which is nearer the reference at a
given fixture is a coin flip and an impostor would pass half the time.

**The mutation matrix**, three mutations applied to the SHIPPED module, each run
against all five claims (MEASURED):

| mutation | refused by |
|---|---|
| the rule answers the opposite of what it measured | dispatch (dense side), dispatch (chirp side), way-back, accuracy |
| the constant is silently 0 | **boundary only** |
| the dense arm returns the separable answer | **accuracy only** |
| (control: no mutation) | nothing -- all five pass |

The two "only" rows are why both of those claims have to exist.  With the
constant at zero the rule and the dispatch agree with each other perfectly, on
the wrong thing, so no dispatch comparison can see it.  With an impostor dense
arm, `'auto'` and `method='direct'` go through the SAME arm and still agree byte
for byte with the rule still saying dense -- only the distance from an exact
reference changes.

---

## 11.  Runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line and `pytest -q --capture=sys -p no:randomly`.

| run | build | tail |
|---|---|---|
| the 55-file grep selection (3030 ids) | branch, WIN | **5 failed, 3019 passed, 6 skipped, 151 warnings in 2249.78 s** -- 4 pre-existing, 1 mine and fixed |
| the five failures, isolated | branch, WIN | **5 failed in 10.58 s** |
| the same five | **`49ddf4bd` git worktree**, WIN | **4 failed in 15.32 s** + the citation walker **1 passed in 3.25 s** |
| the 33 MFT-focused files, with the rule census (1263 ids) | branch, WIN | **1263 passed, 128 warnings in 2680.20 s** |
| the five firing files (159 ids), shipped constant | branch, WIN | **159 passed, 4 warnings in 1185.13 s** (35 direct calls) |
| the five firing files (159 ids), `_MFT_DIRECT_NEVER` | branch, WIN | **159 passed, 4 warnings in 1181.38 s** (0 direct calls) |
| the census / walker / dispatcher-pin / public-API / doc-consistency sweep + `test_audit_except_budget.py` + `test_ci_kernel_consistency.py` + the seven MFT files | branch, WIN | first: **1 failed, 757 passed, 14 skipped in 296.46 s** (the version-claim gate); after the fix: **758 passed, 14 skipped, 11 warnings in 255.43 s** |
| the five core MFT files (`test_c4_mft_direct_default`, `test_wave5_h2_mft_direct`, `test_verify_wave5_hyg2`, `test_verify_hyg2_round2`, `test_fix_v3_mft_centre_window`) | branch, WIN | **79 passed, 4 warnings in 8.38 s** |
| `test_c4_mft_direct_default.py` alone | branch, WIN | **21 passed in 4.50 s** (slowest id 1.13 s) |
| the core MFT files + public-API + except-budget + kernel-consistency + census + doc-consistency | branch, WSL | **1 failed, 112 passed in 16.04 s** -- `test_installed_metadata_version_matches_source_version`, and the WSL venv's editable install reads `lumenairy==5.11.0` against a source `5.48.1`.  Reproduces identically on the `49ddf4bd` worktree: ENVIRONMENTAL |
| the MFT-focused files (minus `b4`) + `test_c4_mft_direct_default.py`, with the rule census | branch, WSL | **1158 passed, 128 warnings in 1462.86 s** -- 1669 rule calls, 193 answering `'direct'`; of those, 388 calls and 158 `'direct'` are WP-C4's OWN test file, leaving **1281 calls and 35 `'direct'` in SHIPPED tests**, the same 35 Windows counts |
| `scripts/record_history_fingerprints.py --check` | branch, WIN | `OK: every history document matches its module` |
| `python -m mypy` (no args) | branch, WIN | `Success: no issues found in 33 source files` |
| `ruff check .` | branch, **WSL** | `All checks passed!` |


---

## 12.  What I could not measure

1.  **A second Linux build, or a different BLAS on Linux.**  The boundary's one
    thin margin is WSL `N = 1024, M = 32` at 0.954 -- the dense route faster by
    5 % in the worst of three rounds.  That margin belongs to a shape where
    pocketfft's worker pool makes the separable route unusually fast, and I
    have exactly one Linux build to measure it on.  A different Linux wheel
    could flip its sign.  What that would cost is bounded and stated in the
    constant's own comment: a few per cent of time at one shape family, against
    a 19x smaller memory peak and a 31x-485x closer answer.  **1/64 is the
    value with a two-fold margin at every shape on both builds** if the
    maintainer wants the headroom; it is one constant away and the tests follow
    it without editing, because they derive their shape lists from it.

2.  **CuPy on WSL.**  Not installed (`ModuleNotFoundError`), recorded as a
    premise rather than skipped.  The CuPy half of section 7 is a Windows
    reading.

3.  **A working cuFFT.**  This box's DLL is broken, so the CuPy chirp-side arms
    could not be run at all -- on either tree.  The two-sided reading in
    section 7 is "the dense side now runs where it used to raise, and the chirp
    side still raises", which is what this box can support; what it cannot say
    is how the two routes compare on a GPU whose FFT works.

4.  **The CI matrix.**  Everything here is two local builds.  The release gate
    is the full un-masked matrix on the merge, and it has not run.

5.  **Timings as anything but bounds.**  The Windows ladder ran with 8 to 11
    other interpreters resident (of 418-421 processes) and its reference loop
    moved by 13 % between the two ends of a run.  Every time in section 2 is an
    upper bound, and the boundary is read from a RATIO taken under the same
    conditions for all three routes, which is the part that survives.  The
    three-round re-measurement is what turns that from an argument into a
    reading: the deciding numbers reproduce to within 5 % across rounds.

6.  **The `separable=True` arm off NumPy.**  `'separable'` falls back to the
    2-D chirp-Z arm on CuPy and JAX, so on those backends the "previous route"
    is always the 2-D one.  The rule is measured on both backends (section 7)
    but the separable fallback's TIME was measured on NumPy only, because it
    does not exist elsewhere.

7.  **Whether an explicit `method='direct'` should also warn at a high phase
    budget.**  Still open, still the maintainer's (`WAVE5_HYGIENE2_REPORT.md`
    decision item 2).  Nothing here changes it; what WP-C4 adds is that
    `'auto'` warns when IT picks the dense route, so the default flip removes
    no diagnostic.

