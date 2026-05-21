#!/usr/bin/env python3
"""Test timeseries2velocity.py with partial-date observations."""

import os
import tempfile

import h5py
import numpy as np

from mintpy import timeseries2velocity as ts2vel
from mintpy.cli import timeseries2velocity as cli
from mintpy.utils import ptime, readfile


DATES = ['20200101', '20210101', '20220101', '20230101', '20240101', '20250101']
VEL = 0.05


def make_timeseries(fname):
    tbase = np.array(ptime.yyyymmdd2years(DATES), dtype=np.float32)
    tbase -= tbase[0]
    data = np.zeros((len(DATES), 2, 4), dtype=np.float32)
    data[:, :, :] = (VEL * tbase).reshape(-1, 1, 1)

    # Pixel 1 has partial date coverage but enough observations for a linear fit.
    data[2, :, 1] = np.nan

    # Pixel 2 does not have enough observations for residue uncertainty.
    data[:, :, 2] = np.nan
    data[:2, :, 2] = (VEL * tbase[:2]).reshape(-1, 1)

    # Pixel 3 is completely missing.
    data[:, :, 3] = np.nan

    with h5py.File(fname, 'w') as f:
        f.create_dataset('date', data=np.array(DATES, dtype=np.bytes_))
        f.create_dataset('timeseries', data=data)
        f.attrs['FILE_TYPE'] = 'timeseries'
        f.attrs['LENGTH'] = '2'
        f.attrs['WIDTH'] = '4'
        f.attrs['UNIT'] = 'm'
        f.attrs['WAVELENGTH'] = '0.056'
        f.attrs['REF_DATE'] = DATES[0]
        f.attrs['REF_Y'] = '0'
        f.attrs['REF_X'] = '0'


def run_velocity(ts_file, vel_file, allow_partial_date=False, fit_coh_file=None):
    args = [ts_file, '-o', vel_file, '--ram', '0.1']
    if allow_partial_date:
        args.append('--allow-partial-date')
    if fit_coh_file:
        args += ['--fit-coh-file', fit_coh_file]
    inps = cli.cmd_line_parse(args)
    ts2vel.run_timeseries2time_func(inps)
    return readfile.read(vel_file, datasetName='velocity')[0]


def test_allow_partial_date_velocity():
    with tempfile.TemporaryDirectory() as temp_dir:
        ts_file = os.path.join(temp_dir, 'timeseries.h5')
        vel_file = os.path.join(temp_dir, 'velocity.h5')
        fit_coh_file = os.path.join(temp_dir, 'fitCoherence.h5')
        make_timeseries(ts_file)

        velocity = run_velocity(ts_file, vel_file, allow_partial_date=True, fit_coh_file=fit_coh_file)
        fit_coh = readfile.read(fit_coh_file, datasetName='fitCoherence')[0]

    assert np.isclose(velocity[0, 0], VEL, atol=1e-5)
    assert np.isclose(velocity[0, 1], VEL, atol=1e-5)
    assert velocity[0, 2] == 0
    assert velocity[0, 3] == 0
    assert np.isclose(fit_coh[0, 0], 1.0, atol=1e-5)
    assert np.isclose(fit_coh[0, 1], 1.0, atol=1e-5)
    assert fit_coh[0, 2] == 0
    assert fit_coh[0, 3] == 0


def test_default_behavior_raises_on_partial_date_nan():
    with tempfile.TemporaryDirectory() as temp_dir:
        ts_file = os.path.join(temp_dir, 'timeseries.h5')
        vel_file = os.path.join(temp_dir, 'velocity.h5')
        make_timeseries(ts_file)

        try:
            run_velocity(ts_file, vel_file, allow_partial_date=False)
        except ValueError:
            return

    raise AssertionError('default timeseries2velocity unexpectedly accepted partial-date NaN observations')


def main():
    test_allow_partial_date_velocity()
    test_default_behavior_raises_on_partial_date_nan()
    print('Pass.')


if __name__ == '__main__':
    main()
