"""R3-3 A/B -- the SAME planes read under both half-pitch lattices.

The round-3 alignment moves the arbiter's fine lattice by half a fine pixel so
that it subdivides each coarse pixel and the two renders integrate the
identical window (E5).  It does not touch the returned FIELD -- the coarse
render and the completion are untouched -- so the oracle fidelity measured
once on the pre-fix tree is still the fidelity of the post-fix field, and only
the READING has to be re-measured.

This probe re-reads every plane of the population TWICE in the same process,
once with the shipped offset and once with the round-2 convention
(``_HALF_PITCH_CENTRE_OFFSET = 0.0``), so the two readings differ by the
lattice and by nothing else -- same build, same process, same call.  It also
asserts the invariant the join depends on: the returned POWER is bit-equal
under the two lattices.

Usage:
    python r3ab.py <out.json> <plane-source.json> [...]
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import r3fixtures as FX
import r3join
import r3oracle as OR
import r3scan as SC


def read_one(fx, z, offset, ray_subsample=2):
    from lumenairy.elements import _lens_traced_multibranch as MB, _lens_traced_uniform as U
    prev = MB._HALF_PITCH_CENTRE_OFFSET
    MB._HALF_PITCH_CENTRE_OFFSET = offset
    try:
        E_in = FX.input_field(fx)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            E, d = U.apply_real_lens_traced_uniform(
                E_in, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=fx['dx'],
                output_plane_distance=float(z),
                ray_subsample=ray_subsample, n_fan=4000,
                return_diagnostics=True)
        return E, d, [str(x.message)[:60] for x in w]
    finally:
        MB._HALF_PITCH_CENTRE_OFFSET = prev


def main():
    out_path = sys.argv[1]
    planes = r3join.collect(sys.argv[2:])
    import lumenairy
    from lumenairy.elements import _lens_traced_multibranch as MB
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    shipped = float(MB._HALF_PITCH_CENTRE_OFFSET)
    print('shipped offset', shipped, flush=True)
    SC._disable_bars()
    rows = []
    for k, r in sorted(planes.items(), key=lambda kv: (str(kv[0][0]),
                                                       kv[0][1])):
        nm, z_um = r['fixture'], float(r['z_um'])
        fx = FX.FIXTURES[nm]
        row = dict(fixture=nm, z_um=z_um, N=fx['N'], dx_um=fx['dx'] * 1e6,
                   ray_subsample=r.get('ray_subsample', 2))
        try:
            Ea, da, wa = read_one(fx, z_um * 1e-6, shipped)
            Eb, db, wb = read_one(fx, z_um * 1e-6, 0.0)
        except Exception as exc:                            # noqa: BLE001
            row['error'] = f'{type(exc).__name__}: {str(exc)[:160]}'
            rows.append(row)
            print(json.dumps(row), flush=True)
            continue
        row.update(
            pixel_continuity=da.get('pixel_continuity'),
            pixel_continuity_unaligned=db.get('pixel_continuity'),
            multibranch_pixel_continuity=da.get('multibranch_pixel_continuity'),
            multibranch_pixel_continuity_unaligned=db.get(
                'multibranch_pixel_continuity'),
            pixel_continuity_of=da.get('pixel_continuity_of'),
            pixel_continuity_scope=da.get('pixel_continuity_scope'),
            pixel_continuity_decision=da.get('pixel_continuity_decision'),
            reason=da.get('reason'), fell_back=da.get('fell_back'),
            returned_power=OR.power(Ea, fx['dx']),
            returned_power_unaligned=OR.power(Eb, fx['dx']),
            field_bit_identical=bool(np.array_equal(
                np.asarray(Ea).view(np.float64),
                np.asarray(Eb).view(np.float64))),
            warnings=sorted(set(wa)))
        rows.append(row)
        print(json.dumps(row), flush=True)
    ok = [r for r in rows if r.get('pixel_continuity') is not None
          and r.get('pixel_continuity_unaligned') is not None]
    moved = [r for r in rows if r.get('field_bit_identical') is False]
    summary = dict(
        n=len(rows), n_read=len(ok), n_field_moved=len(moved),
        shipped_offset=shipped,
        max_abs_reading_change=max((abs(r['pixel_continuity']
                                        - r['pixel_continuity_unaligned'])
                                    for r in ok), default=None),
        aligned_closer_to_1=sum(1 for r in ok
                                if abs(r['pixel_continuity'] - 1.0)
                                < abs(r['pixel_continuity_unaligned'] - 1.0)),
        unaligned_closer_to_1=sum(1 for r in ok
                                  if abs(r['pixel_continuity'] - 1.0)
                                  > abs(r['pixel_continuity_unaligned'] - 1.0)),
        field_moved_rows=[dict(fixture=r['fixture'], z=r['z_um'])
                          for r in moved[:20]])
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(lumenairy=lumenairy.__file__, summary=summary,
                       rows=rows), f, indent=1)
    print(json.dumps(summary, default=float))
    print('wrote', out_path)


if __name__ == '__main__':
    main()
