#!/bin/sh
# SUM-AT-APERTURE probe -- arm-B run set of RECORD, at the coarse common grid
# (sec 8.1 shows it is pitch-converged), rebuilt against the CURRENT arm-A
# realisation after the working tree's lumenairy/** changed mid-probe.
set -x
cd "$(dirname "$0")"
export DXC=1.2292e-6
export NC=8192
P="python -W ignore::UserWarning sumap_probe_121.py armb"
$P --orders "0,0;-2,0;-4,-2" --leg both --suffix _r2
$P --orders "0,0"            --leg both --suffix _r2
$P --orders "-2,0"           --leg both --suffix _r2
$P --orders "-4,-2"          --leg both --suffix _r2
