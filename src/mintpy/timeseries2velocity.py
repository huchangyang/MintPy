############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, Yuan-Kai Liu, 2013 #
############################################################
# Recommend import:
#   from mintpy import timeseries2velocity as ts2vel


import os
import time

import numpy as np
from scipy import linalg

from mintpy.objects import HDFEOS, cluster, giantTimeseries, timeseries
from mintpy.objects.velocity_spatial import (
    accumulate_linear_normal_eq,
    solve_constrained_velocity,
)
from mintpy.utils import ptime, readfile, time_func, writefile

DATA_TYPE = np.float32
# key configuration parameter name
key_prefix = 'mintpy.timeFunc.'
config_keys = [
    # date
    'startDate',
    'endDate',
    'excludeDate',
    'allowPartialDate',
    'fitCoherenceFile',
    'spatialSmooth',
    'spatialSmoothStrength',
    # time functions
    'polynomial',
    'periodic',
    'stepDate',
    'exp',
    'log',
    # uncertainty quantification
    'uncertaintyQuantification',
    'timeSeriesCovFile',
    'bootstrapCount',
]


############################################################################
def run_or_skip(inps):
    print('update mode: ON')
    flag = 'skip'

    # check output file
    if not os.path.isfile(inps.outfile):
        flag = 'run'
        print(f'1) output file {inps.outfile} NOT found.')
    else:
        print(f'1) output file {inps.outfile} already exists.')
        ti = os.path.getmtime(inps.timeseries_file)
        to = os.path.getmtime(inps.outfile)
        if ti > to:
            flag = 'run'
            print(f'2) output file is NOT newer than input file: {inps.timeseries_file}.')
        else:
            print(f'2) output file is newer than input file: {inps.timeseries_file}.')

    # check configuration
    if flag == 'skip':
        atr = readfile.read_attribute(inps.outfile)
        if any(str(vars(inps)[key]) != atr.get(key_prefix+key, 'None') for key in config_keys):
            flag = 'run'
            print(f'3) NOT all key configuration parameters are the same: {config_keys}.')
        else:
            print(f'3) all key configuration parameters are the same: {config_keys}.')

    # result
    print(f'run or skip: {flag}.')
    return flag


############################################################################
def read_date_info(inps):
    """Read dates used in the estimation and its related info.

    Parameters: inps - Namespace
    Returns:    inps - Namespace, adding the following new fields:
                       date_list - list of str, dates used for estimation
                       dropDate  - 1D np.ndarray in bool in size of all available dates
    """
    # initiate and open time-series file object
    ftype = readfile.read_attribute(inps.timeseries_file)['FILE_TYPE']
    if ftype == 'timeseries':
        ts_obj = timeseries(inps.timeseries_file)
    elif ftype == 'giantTimeseries':
        ts_obj = giantTimeseries(inps.timeseries_file)
    elif ftype == 'HDFEOS':
        ts_obj = HDFEOS(inps.timeseries_file)
    else:
        raise ValueError(f'Un-recognized time-series type: {ftype}')
    ts_obj.open()

    # exclude dates - user inputs
    ex_date_list = ptime.get_exclude_date_list(
        date_list=ts_obj.dateList,
        start_date=inps.startDate,
        end_date=inps.endDate,
        exclude_date=inps.excludeDate)

    # exclude dates - no obs data [for offset time-series only for now]
    if os.path.basename(inps.timeseries_file).startswith('timeseriesRg'):
        data, atr = readfile.read(inps.timeseries_file)
        flag = np.nansum(data, axis=(1,2)) == 0
        flag[ts_obj.dateList.index(atr['REF_DATE'])] = 0
        if np.sum(flag) > 0:
            print(f'number of empty dates to exclude: {np.sum(flag)}')
            ex_date_list += np.array(ts_obj.dateList)[flag].tolist()
            ex_date_list = sorted(list(set(ex_date_list)))

    # dates used for estimation - inps.date_list
    inps.date_list = [i for i in ts_obj.dateList if i not in ex_date_list]

    # flag array for ts data reading
    inps.dropDate = np.array([i not in ex_date_list for i in ts_obj.dateList], dtype=np.bool_)

    # print out msg
    print('-'*50)
    print(f'dates from input file: {ts_obj.numDate}\n{ts_obj.dateList}')
    print('-'*50)
    if len(inps.date_list) == len(ts_obj.dateList):
        print('using all dates to calculate the time function')
    else:
        print(f'dates used to estimate the time function: {len(inps.date_list)}\n{inps.date_list}')
    print('-'*50)

    return inps


def _get_part_date_lstsq_inputs(model, date_list, dis_ts, seconds=0):
    """Prepare per-pixel inputs for least squares with partial-date observations."""
    valid_flag = np.isfinite(dis_ts)
    valid_date_list = np.array(date_list)[valid_flag].tolist()
    num_param = time_func.get_num_param(model)

    # Residue-based STD needs at least one degree of freedom.
    if len(valid_date_list) <= num_param:
        return None, None, None

    G = time_func.get_design_matrix4time_func(valid_date_list, model=model, seconds=seconds)
    if np.linalg.matrix_rank(G) < num_param:
        return None, None, None

    return valid_flag, valid_date_list, G


def _estimate_part_date_pixel(model, date_list, dis_ts, seconds=0):
    """Estimate one pixel using only its finite time-series observations."""
    valid_flag, valid_date_list, G = _get_part_date_lstsq_inputs(
        model=model,
        date_list=date_list,
        dis_ts=dis_ts,
        seconds=seconds,
    )
    if valid_flag is None:
        return None, None, None, None

    G, m, e2 = time_func.estimate_time_func(
        model=model,
        date_list=valid_date_list,
        dis_ts=dis_ts[valid_flag].reshape(-1, 1),
        seconds=seconds,
    )
    return valid_flag, G, m, e2


def _calc_fit_coherence(residual, wavelength):
    """Calculate coherence-like quality from time-function residual displacement."""
    if wavelength is None:
        return np.nan

    residual_phase = -4. * np.pi * residual / wavelength
    return np.abs(np.sum(np.exp(1j * residual_phase))) / residual_phase.size


def _is_linear_poly1_only(inps):
    """True if the time function is intercept + velocity only."""
    return (
        int(getattr(inps, 'polynomial', 1)) == 1
        and not getattr(inps, 'periodic', None)
        and not getattr(inps, 'stepDate', None)
        and not getattr(inps, 'polyline', None)
        and not getattr(inps, 'exp', None)
        and not getattr(inps, 'log', None)
    )


def _use_spatial_smooth(inps, print_msg=True):
    """Enable spatial smoothness only for allowPartialDate + linear velocity."""
    if not getattr(inps, 'allowPartialDate', False):
        return False
    if not getattr(inps, 'spatialSmooth', False):
        return False
    if not _is_linear_poly1_only(inps):
        if print_msg:
            print('WARNING: spatialSmooth requires polynomial=1 without '
                  'periodic/step/exp/log/polyline; fall back to per-pixel fitting.')
        return False
    return True


def _read_and_mask_ts_box(inps, atrV, box, num_date):
    """Read one time-series box, apply referencing, and build the valid-pixel mask."""
    box_wid = box[2] - box[0]
    box_len = box[3] - box[1]
    num_pixel = box_len * box_wid
    print(f'reading data from file {inps.timeseries_file} ...')
    ts_data = readfile.read(inps.timeseries_file, box=box)[0]

    if inps.ref_date:
        print(f'referecing to date: {inps.ref_date}')
        ref_ind = inps.date_list.index(inps.ref_date)
        ts_data -= np.tile(ts_data[ref_ind, :, :], (ts_data.shape[0], 1, 1))

    if inps.ref_yx:
        print(f'referencing to point (y, x): ({inps.ref_yx[0]}, {inps.ref_yx[1]})')
        ref_box = (inps.ref_yx[1], inps.ref_yx[0], inps.ref_yx[1]+1, inps.ref_yx[0]+1)
        ref_val = readfile.read(inps.timeseries_file, box=ref_box)[0]
        ts_data -= np.tile(ref_val.reshape(ts_data.shape[0], 1, 1),
                           (1, ts_data.shape[1], ts_data.shape[2]))

    ts_data = ts_data[inps.dropDate, :, :].reshape(num_date, -1)
    if atrV['UNIT'] == 'mm':
        ts_data *= 1./1000.

    print('skip pixels with zero/nan value in all acquisitions')
    ts_stack = np.nanmean(ts_data, axis=0)
    mask = np.multiply(~np.isnan(ts_stack), ts_stack != 0.)
    del ts_stack
    if 'REF_Y' in atrV and 'REF_X' in atrV:
        ry, rx = int(atrV['REF_Y']) - box[1], int(atrV['REF_X']) - box[0]
        if 0 <= rx < box_wid and 0 <= ry < box_len:
            mask[ry * box_wid + rx] = 1
    return ts_data, mask, box_len, box_wid, num_pixel


def _fit_coherence_stack(residual, valid, wavelength):
    """fitCoherence from residuals, ignoring invalid dates (valid is bool mask)."""
    n_pixel = residual.shape[1]
    if wavelength is None:
        return np.full(n_pixel, np.nan, dtype=DATA_TYPE)
    phase = -4. * np.pi * residual / wavelength
    cpx = np.exp(1j * phase)
    cpx[~valid] = 0
    nobs = np.sum(valid, axis=0).clip(min=1)
    return (np.abs(np.sum(cpx, axis=0)) / nobs).astype(DATA_TYPE)


def _estimate_with_spatial_smooth(inps, atrV, length, width, num_date, seconds,
                                  model, num_param, wavelength, box_list, num_box,
                                  start_time):
    """Joint linear velocity with 4-neighbor coverage-weighted spatial smoothness."""
    print('estimating linear velocity with spatial smoothness constraint ...')
    print('using 4-neighbor (azimuth + range) velocity coupling; intercept unconstrained')
    G = time_func.get_design_matrix4time_func(inps.date_list, model=model, seconds=seconds)
    if G.shape[1] != 2:
        raise ValueError('spatialSmooth expects a 2-parameter (intercept, velocity) model')

    ys_all, xs_all, N_all, b_all, valid_all = [], [], [], [], []
    e2_all, std_all = [], []

    for i, box in enumerate(box_list):
        if num_box > 1:
            print(f'\n------- processing patch {i+1} out of {num_box} --------------')
        ts_data, mask, box_len, box_wid, num_pixel = _read_and_mask_ts_box(
            inps, atrV, box, num_date)
        num_pixel2inv = int(np.sum(mask))
        print('number of pixels to invert: {} out of {} ({:.1f}%)'.format(
            num_pixel2inv, num_pixel, num_pixel2inv / max(num_pixel, 1) * 100))
        if num_pixel2inv == 0:
            continue

        idx = np.where(mask)[0]
        pix_y = idx // box_wid + box[1]
        pix_x = idx % box_wid + box[0]
        ys, xs, N, b, valid, e2, m_std = accumulate_linear_normal_eq(
            G, ts_data[:, idx], pix_y, pix_x, min_obs=num_param + 1)
        if ys.size == 0:
            continue
        ys_all.append(ys)
        xs_all.append(xs)
        N_all.append(N)
        b_all.append(b)
        valid_all.append(valid)
        e2_all.append(e2)
        std_all.append(m_std)

    m_full = np.full((num_param, length * width), np.nan, dtype=DATA_TYPE)
    m_std_full = np.full((num_param, length * width), np.nan, dtype=DATA_TYPE)
    e2_full = np.full(length * width, np.nan, dtype=DATA_TYPE)
    mask_all = np.zeros(length * width, dtype=np.bool_)

    if ys_all:
        ys = np.concatenate(ys_all)
        xs = np.concatenate(xs_all)
        N = np.concatenate(N_all, axis=0)
        b = np.concatenate(b_all, axis=0)
        valid = np.concatenate(valid_all, axis=0)
        e2 = np.concatenate(e2_all)
        m_std = np.concatenate(std_all, axis=0)
        gidx = ys * width + xs
        mask_all[gidx] = True
        e2_full[gidx] = e2.astype(DATA_TYPE)
        m_std_full[:, gidx] = m_std.T.astype(DATA_TYPE)

        strength = float(getattr(inps, 'spatialSmoothStrength', 1.0) or 1.0)
        m, info, lam = solve_constrained_velocity(
            N, b, valid, ys, xs, length, width,
            strength=strength, alpha=1.0, print_msg=True)
        m_full[:, gidx] = m.T.astype(DATA_TYPE)
    else:
        print('WARNING: no estimable pixels for spatially constrained velocity')

    block = [0, length, 0, width]
    ds_dict = model2hdf5_dataset(model, m_full, m_std_full, mask=mask_all)[0]
    if inps.uncertaintyQuantification == 'residue':
        residue = np.full(length * width, np.nan, dtype=DATA_TYPE)
        residue[mask_all] = np.sqrt(e2_full[mask_all])
        ds_dict['residue'] = residue
    for ds_name, data in ds_dict.items():
        writefile.write_hdf5_block(inps.outfile,
                                   data=data.reshape(length, width),
                                   datasetName=ds_name,
                                   block=block)

    intercept = m_full[0].reshape(length, width)
    velocity = m_full[1].reshape(length, width)

    if inps.allowPartialDate:
        fit_coh = np.full((length, width), np.nan, dtype=DATA_TYPE)
        for i, box in enumerate(box_list):
            ts_data, mask, box_len, box_wid, num_pixel = _read_and_mask_ts_box(
                inps, atrV, box, num_date)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            pix_y = idx // box_wid + box[1]
            pix_x = idx % box_wid + box[0]
            a = intercept[pix_y, pix_x]
            v = velocity[pix_y, pix_x]
            finite = np.isfinite(a) & np.isfinite(v)
            if not np.any(finite):
                continue
            idx = idx[finite]
            pix_y, pix_x = pix_y[finite], pix_x[finite]
            a, v = a[finite], v[finite]
            d = ts_data[:, idx]
            valid = np.isfinite(d)
            pred = a[None, :] * G[:, 0:1] + v[None, :] * G[:, 1:2]
            residual = d - pred
            fit_coh[pix_y, pix_x] = _fit_coherence_stack(residual, valid, wavelength)
        writefile.write_hdf5_block(inps.fitCoherenceFile,
                                   data=fit_coh,
                                   datasetName='fitCoherence',
                                   block=block)

    if inps.save_res:
        print('calculating the time series residual ...')
        print(f'remove time functions: {inps.rm_timefuns}')
        ts_res = np.full((num_date, length * width), np.nan, dtype=np.float32)
        rm_ds_inds = None if 'all' in inps.rm_timefuns else timefun_names2ds_names(
            inps.rm_timefuns, ds_dict.keys())[1]
        for i, box in enumerate(box_list):
            ts_data, mask, box_len, box_wid, num_pixel = _read_and_mask_ts_box(
                inps, atrV, box, num_date)
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            pix_y = idx // box_wid + box[1]
            pix_x = idx % box_wid + box[0]
            gidx = pix_y * width + pix_x
            m_pix = m_full[:, gidx]
            finite = np.isfinite(m_pix[0]) & np.isfinite(m_pix[1])
            if not np.any(finite):
                continue
            idx = idx[finite]
            gidx = gidx[finite]
            m_pix = m_pix[:, finite]
            d = ts_data[:, idx]
            valid = np.isfinite(d)
            if rm_ds_inds is None:
                pred = np.dot(G, m_pix)
            else:
                pred = np.dot(G[:, rm_ds_inds], m_pix[rm_ds_inds, :])
            res = np.full(d.shape, np.nan, dtype=np.float32)
            res[valid] = (d - pred)[valid]
            ts_res[:, gidx] = res
        writefile.write_hdf5_block(
            inps.res_file,
            data=ts_res.reshape(num_date, length, width),
            datasetName='timeseries',
            block=[0, num_date, 0, length, 0, width],
        )

    m, s = divmod(time.time() - start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.')
    return inps.outfile


def run_timeseries2time_func(inps):
    start_time = time.time()

    # basic file info
    atr = readfile.read_attribute(inps.timeseries_file)
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])

    # read date info
    inps = read_date_info(inps)
    num_date = len(inps.date_list)
    dates = np.array(inps.date_list)
    seconds = atr.get('CENTER_LINE_UTC', 0)

    # use the 1st date as reference if not found, e.g. timeseriesResidual.h5 file
    if "REF_DATE" not in atr.keys() and not inps.ref_date:
        inps.ref_date = inps.date_list[0]
        print('WARNING: No REF_DATE found in time-series file or input in command line.')
        print(f'  Set "--ref-date {inps.date_list[0]}" and continue.')

    # get deformation model from inputs
    model = time_func.inps2model(inps, date_list=inps.date_list)
    num_param = time_func.get_num_param(model)


    ## output preparation

    # time_func_param: attributes
    date0, date1 = inps.date_list[0], inps.date_list[-1]
    atrV = dict(atr)
    atrV['FILE_TYPE'] = 'velocity'
    atrV['UNIT'] = 'm/year'
    atrV['START_DATE'] = date0
    atrV['END_DATE'] = date1
    atrV['DATE12'] = f'{date0}_{date1}'
    if inps.ref_yx:
        atrV['REF_Y'] = inps.ref_yx[0]
        atrV['REF_X'] = inps.ref_yx[1]
    if inps.ref_date:
        atrV['REF_DATE'] = inps.ref_date

    wavelength = float(atr['WAVELENGTH']) if 'WAVELENGTH' in atr.keys() else None
    if inps.allowPartialDate and wavelength is None:
        print('WARNING: WAVELENGTH attribute not found; fitCoherence will be set to NaN.')

    # time_func_param: config parameter
    if not hasattr(inps, 'spatialSmooth') or getattr(inps, 'spatialSmooth', None) is None:
        inps.spatialSmooth = bool(getattr(inps, 'allowPartialDate', False))
    if getattr(inps, 'spatialSmoothStrength', None) is None:
        inps.spatialSmoothStrength = 1.0
    print(f'add/update the following configuration metadata:\n{config_keys}')
    for key in config_keys:
        atrV[key_prefix+key] = str(vars(inps)[key])

    # time_func_param: instantiate output file
    ds_name_dict, ds_unit_dict = model2hdf5_dataset(model, ds_shape=(length, width))[1:]
    # add dataset: residue
    if inps.uncertaintyQuantification == 'residue':
        ds_name_dict['residue'] = [np.float32, (length, width), None]
        ds_unit_dict['residue'] = 'm'

    writefile.layout_hdf5(inps.outfile,
                          metadata=atrV,
                          ds_name_dict=ds_name_dict,
                          ds_unit_dict=ds_unit_dict)

    # fit coherence: quality of time-function fitting in partial-date mode
    if inps.allowPartialDate:
        atrF = dict(atrV)
        atrF['FILE_TYPE'] = 'fitCoherence'
        atrF['UNIT'] = '1'
        ds_name_dict = {'fitCoherence' : [np.float32, (length, width), None]}
        ds_unit_dict = {'fitCoherence' : '1'}
        writefile.layout_hdf5(inps.fitCoherenceFile,
                              metadata=atrF,
                              ds_name_dict=ds_name_dict,
                              ds_unit_dict=ds_unit_dict)

    # timeseries_res: attributes + instantiate output file
    if inps.save_res:
        atrR = dict(atr)
        # remove REF_DATE attribute
        for key in ['REF_DATE']:
            if key in atrR.keys():
                atrR.pop(key)
        # prepare ds_name_dict manually, instead of using ref_file, to support --ex option
        date_digit = len(inps.date_list[0])
        ds_name_dict = {
            "date" : [np.dtype(f'S{date_digit}'), (num_date,), np.array(inps.date_list, np.bytes_)],
            "timeseries" : [np.float32, (num_date, length, width), None]
        }
        writefile.layout_hdf5(inps.res_file, ds_name_dict=ds_name_dict, metadata=atrR)


    ## estimation

    # calc number of box based on memory limit
    memoryAll = (num_date + num_param * 2 + 2) * length * width * 4
    if inps.uncertaintyQuantification == 'bootstrap':
        memoryAll += inps.bootstrapCount * num_param * length * width * 4
    num_box = int(np.ceil(memoryAll * 3 / (inps.maxMemory * 1024**3)))
    box_list, num_box = cluster.split_box2sub_boxes(
        box=(0, 0, width, length),
        num_split=num_box,
        dimension='y',
        print_msg=True,
    )

    if _use_spatial_smooth(inps):
        return _estimate_with_spatial_smooth(
            inps, atrV, length, width, num_date, seconds, model, num_param,
            wavelength, box_list, num_box, start_time)

    # loop for block-by-block IO
    for i, box in enumerate(box_list):
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]
        num_pixel = box_len * box_wid
        if num_box > 1:
            print(f'\n------- processing patch {i+1} out of {num_box} --------------')
            print(f'box width:  {box_wid}')
            print(f'box length: {box_len}')

        # initiate output (NaN = not estimated)
        m = np.full((num_param, num_pixel), np.nan, dtype=DATA_TYPE)
        m_std = np.full((num_param, num_pixel), np.nan, dtype=DATA_TYPE)

        # read input
        print(f'reading data from file {inps.timeseries_file} ...')
        ts_data = readfile.read(inps.timeseries_file, box=box)[0]

        # referencing in time and space
        # for file w/o reference info. e.g. ERA5.h5
        if inps.ref_date:
            print(f'referecing to date: {inps.ref_date}')
            ref_ind = inps.date_list.index(inps.ref_date)
            ts_data -= np.tile(ts_data[ref_ind, :, :], (ts_data.shape[0], 1, 1))

        if inps.ref_yx:
            print(f'referencing to point (y, x): ({inps.ref_yx[0]}, {inps.ref_yx[1]})')
            ref_box = (inps.ref_yx[1], inps.ref_yx[0], inps.ref_yx[1]+1, inps.ref_yx[0]+1)
            ref_val = readfile.read(inps.timeseries_file, box=ref_box)[0]
            ts_data -= np.tile(ref_val.reshape(ts_data.shape[0], 1, 1),
                               (1, ts_data.shape[1], ts_data.shape[2]))

        ts_data = ts_data[inps.dropDate, :, :].reshape(num_date, -1)
        if atrV['UNIT'] == 'mm':
            ts_data *= 1./1000.

        ts_cov = None
        if inps.uncertaintyQuantification == 'covariance':
            print(f'reading time-series covariance matrix from file {inps.timeSeriesCovFile} ...')
            ts_cov = readfile.read(inps.timeSeriesCovFile, box=box)[0]
            if len(ts_cov.shape) == 4:
                # full covariance matrix in 4D --> 3D
                if num_date < ts_cov.shape[0]:
                    ts_cov = ts_cov[inps.dropDate, :, :, :]
                    ts_cov = ts_cov[:, inps.dropDate, :, :]
                ts_cov = ts_cov.reshape(num_date, num_date, -1)

            elif len(ts_cov.shape) == 3:
                # diaginal variance matrix in 3D --> 2D
                if num_date < ts_cov.shape[0]:
                    ts_cov = ts_cov[inps.dropDate, :, :]
                ts_cov = ts_cov.reshape(num_date, -1)

            ## set zero value to a fixed small value to avoid divide by zero
            #epsilon = 1e-5
            #ts_cov[ts_cov<epsilon] = epsilon

        # mask invalid pixels
        print('skip pixels with zero/nan value in all acquisitions')
        ts_stack = np.nanmean(ts_data, axis=0)
        mask = np.multiply(~np.isnan(ts_stack), ts_stack!=0.)
        del ts_stack
        # include the reference point
        ry, rx = int(atrV['REF_Y']) - box[1], int(atrV['REF_X']) - box[0]
        if 0 <= rx < box_wid and 0 <= ry < box_len:
            mask[ry * box_wid + rx] = 1

        #if ts_cov is not None:
        #    print('skip pxiels with nan STD value in any acquisition')
        #    num_std_nan = np.sum(np.isnan(ts_cov), axis=0)
        #    mask *= num_std_nan == 0
        #    del num_std_nan

        num_pixel2inv = int(np.sum(mask))
        idx_pixel2inv = np.where(mask)[0]
        print('number of pixels to invert: {} out of {} ({:.1f}%)'.format(
            num_pixel2inv, num_pixel, num_pixel2inv/num_pixel*100))

        # write NaN for empty boxes and continue
        if num_pixel2inv == 0:
            block = [box[1], box[3], box[0], box[2]]
            ds_dict = model2hdf5_dataset(model, m, m_std, mask=mask)[0]
            if inps.uncertaintyQuantification == 'residue':
                ds_dict['residue'] = np.full(num_pixel, np.nan, dtype=DATA_TYPE)
            for ds_name, data in ds_dict.items():
                writefile.write_hdf5_block(inps.outfile,
                                           data=data.reshape(box_len, box_wid),
                                           datasetName=ds_name,
                                           block=block)
            if inps.allowPartialDate:
                writefile.write_hdf5_block(inps.fitCoherenceFile,
                                           data=np.full((box_len, box_wid), np.nan, dtype=DATA_TYPE),
                                           datasetName='fitCoherence',
                                           block=block)
            continue


        ### estimation / solve Gm = d
        print('estimating time functions via linalg.lstsq ...')
        if inps.allowPartialDate:
            ts_data2inv = ts_data[:, mask]
            obs_flag = np.isfinite(ts_data2inv)
            mask_all_date = np.all(obs_flag, axis=0)
            mask_part_date = np.any(obs_flag, axis=0) & ~mask_all_date
            idx_pixel2inv_all = idx_pixel2inv[mask_all_date]
            idx_pixel2inv_part = idx_pixel2inv[mask_part_date]
            num_pixel2inv_all = int(np.sum(mask_all_date))
            num_pixel2inv_part = int(np.sum(mask_part_date))
            print('pixels with valid observations in all dates: {} ({:.1f}%)'.format(
                num_pixel2inv_all, num_pixel2inv_all/num_pixel2inv*100))
            print('pixels with valid observations in some dates: {} ({:.1f}%)'.format(
                num_pixel2inv_part, num_pixel2inv_part/num_pixel2inv*100))
            e2 = np.full(num_pixel, np.nan, dtype=DATA_TYPE)
            fit_coh = np.full(num_pixel, np.nan, dtype=DATA_TYPE)
        else:
            ts_data = ts_data[:, mask]

        if inps.uncertaintyQuantification == 'bootstrap':
            ## option 1 - least squares with bootstrapping
            # Bootstrapping is a resampling method which can be used to estimate properties
            # of an estimator. The method relies on independently sampling the data set with
            # replacement.
            print('estimating time functions STD with bootstrap resampling ({} times) ...'.format(
                inps.bootstrapCount))

            # calc model of all bootstrap sampling
            rng = np.random.default_rng()
            m_boot = np.zeros((inps.bootstrapCount, num_param, num_pixel2inv), dtype=DATA_TYPE)
            prog_bar = ptime.progressBar(maxValue=inps.bootstrapCount)
            for i in range(inps.bootstrapCount):
                # bootstrap resampling
                boot_ind = rng.choice(num_date, size=num_date, replace=True)
                boot_ind.sort()

                # estimation
                m_boot[i] = time_func.estimate_time_func(
                    model=model,
                    date_list=dates[boot_ind].tolist(),
                    dis_ts=ts_data[boot_ind],
                    seconds=seconds)[1]

                prog_bar.update(i+1, suffix=f'iteration {i+1} / {inps.bootstrapCount}')
            prog_bar.close()
            #del ts_data

            # get mean/std among all bootstrap sampling
            m[:, mask] = m_boot.mean(axis=0).reshape(num_param, -1)
            m_std[:, mask] = m_boot.std(axis=0).reshape(num_param, -1)
            del m_boot

            # get design matrix to calculate the residual time series
            G = time_func.get_design_matrix4time_func(inps.date_list, model=model, ref_date=inps.ref_date, seconds=seconds)


        else:
            ## option 2 - least squares with uncertainty propagation
            G = time_func.get_design_matrix4time_func(inps.date_list, model=model, seconds=seconds)
            if inps.allowPartialDate:
                if num_pixel2inv_all > 0:
                    G, mi, e2i = time_func.estimate_time_func(
                        model=model,
                        date_list=inps.date_list,
                        dis_ts=ts_data2inv[:, mask_all_date],
                        seconds=seconds)
                    m[:, idx_pixel2inv_all] = mi
                    e2[idx_pixel2inv_all] = e2i
                    residual = ts_data[:, idx_pixel2inv_all] - np.dot(G, mi)
                    residual_phase = -4. * np.pi * residual / wavelength if wavelength is not None else None
                    if residual_phase is not None:
                        fit_coh[idx_pixel2inv_all] = np.abs(np.sum(np.exp(1j * residual_phase), axis=0)) / num_date

                if num_pixel2inv_part > 0:
                    num_pixel_skip = 0
                    print('estimating time functions for pixels with partial-date observations pixel-by-pixel ...')
                    prog_bar = ptime.progressBar(maxValue=num_pixel2inv_part)
                    for j, idx in enumerate(idx_pixel2inv_part):
                        valid_flag, Gi, mi, e2i = _estimate_part_date_pixel(
                            model=model,
                            date_list=inps.date_list,
                            dis_ts=ts_data[:, idx],
                            seconds=seconds,
                        )
                        if valid_flag is None:
                            num_pixel_skip += 1
                        else:
                            m[:, idx] = mi.flatten()
                            e2[idx] = np.array(e2i).reshape(-1)[0]
                            residual = ts_data[valid_flag, idx] - np.dot(Gi, mi).flatten()
                            fit_coh[idx] = _calc_fit_coherence(residual, wavelength)

                        prog_bar.update(j+1, every=200, suffix=f'{j+1}/{num_pixel2inv_part} pixels')
                    prog_bar.close()
                    print(f'number of partial-date pixels skipped due to insufficient observations/rank: {num_pixel_skip}')
            else:
                G, m[:, mask], e2 = time_func.estimate_time_func(
                    model=model,
                    date_list=inps.date_list,
                    dis_ts=ts_data,
                    seconds=seconds)
            #del ts_data

            ## Compute the covariance matrix for model parameters:
            #       G * m = d                                       (1)
            #       m_hat = G+ * d                                  (2)
            #     C_m_hat = G+ * C_d * G+.T                         (3)
            #
            # [option 2.1] For weighted least squares estimation:
            #          G+ = (G.T * C_d^-1 * G)^-1 * G.T * C_d^-1    (4)
            # =>  C_m_hat = (G.T * C_d^-1 * G)^-1                   (5)
            #
            # [option 2.2] For ordinary least squares estimation:
            #          G+ = (G.T * G)^-1 * G.T                      (6)
            #     C_m_hat = G+ * C_d * G+.T                         (7)
            #
            # [option 2.3] Assuming normality of the observation errors (in the time domain) with
            # the variance of sigma^2, we have C_d = sigma^2 * I, then eq. (3) is simplfied into:
            #     C_m_hat = sigma^2 * (G.T * G)^-1                  (8)
            #
            # Using the law of integrated expectation, we estimate the obs sigma^2 using
            # the OLS estimation residual as:
            #           e_hat = d - d_hat                           (9)
            # =>  sigma_hat^2 = (e_hat.T * e_hat) / N               (10)
            # =>      sigma^2 = sigma_hat^2 * N / (N - P)           (11)
            #                 = (e_hat.T * e_hat) / (N - P)         (12)
            #
            # Eq. (12) is the generalized form of eq. (10) in Fattahi & Amelung (2015, JGR),
            # which is for linear velocity.

            if inps.uncertaintyQuantification == 'covariance':
                # option 2.2 - linear propagation from time-series (co)variance matrix
                # TO DO: save the full covariance matrix of the time function parameters
                # only the STD is saved right now
                covar_flag = True if len(ts_cov.shape) == 3 else False
                msg = 'estimating time functions STD from time-serries '
                msg += 'covariance pixel-by-pixel ...' if covar_flag else 'variance pixel-by-pixel ...'
                print(msg)

                # calc the common pseudo-inverse matrix
                Gplus = linalg.pinv(G)

                # loop over each pixel
                # or use multidimension matrix multiplication
                # m_cov = Gplus @ ts_cov @ Gplus.T
                prog_bar = ptime.progressBar(maxValue=num_pixel2inv)
                for i in range(num_pixel2inv):
                    idx = idx_pixel2inv[i]

                    # cov: time-series -> time func
                    ts_covi = ts_cov[:, :, idx] if covar_flag else np.diag(ts_cov[:, idx])
                    m_cov = np.linalg.multi_dot([Gplus, ts_covi, Gplus.T])
                    m_std[:, idx] = np.sqrt(np.diag(m_cov))

                    prog_bar.update(i+1, every=200, suffix=f'{i+1}/{num_pixel2inv} pixels')
                prog_bar.close()

            elif inps.uncertaintyQuantification == 'residue':
                # option 2.3 - assume obs errors following normal dist. in time
                print('estimating time functions STD from time-series fitting residual ...')
                if inps.allowPartialDate:
                    if num_pixel2inv_all > 0:
                        G_inv = linalg.inv(np.dot(G.T, G))
                        m_var = e2[idx_pixel2inv_all].reshape(1, -1) / (num_date - num_param)
                        m_std[:, idx_pixel2inv_all] = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), m_var))

                    if num_pixel2inv_part > 0:
                        prog_bar = ptime.progressBar(maxValue=num_pixel2inv_part)
                        for j, idx in enumerate(idx_pixel2inv_part):
                            valid_flag, valid_date_list, Gi = _get_part_date_lstsq_inputs(
                                model=model,
                                date_list=inps.date_list,
                                dis_ts=ts_data[:, idx],
                                seconds=seconds,
                            )
                            if valid_flag is not None:
                                G_inv = linalg.inv(np.dot(Gi.T, Gi))
                                m_var = e2[idx] / (len(valid_date_list) - num_param)
                                m_std[:, idx] = np.sqrt(np.diag(G_inv) * m_var)
                            prog_bar.update(j+1, every=200, suffix=f'{j+1}/{num_pixel2inv_part} pixels')
                        prog_bar.close()
                else:
                    G_inv = linalg.inv(np.dot(G.T, G))
                    m_var = e2.reshape(1, -1) / (num_date - num_param)
                    m_std[:, mask] = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), m_var))

                # simplified form for linear velocity (without matrix linear algebra)
                # equation (10) in Fattahi & Amelung (2015, JGR)
                # ts_diff = ts_data - np.dot(G, m)
                # t_diff = G[:, 1] - np.mean(G[:, 1])
                # vel_std = np.sqrt(np.sum(ts_diff ** 2, axis=0) / np.sum(t_diff ** 2)  / (num_date - 2))

        # write - time func params
        block = [box[1], box[3], box[0], box[2]]
        ds_dict = model2hdf5_dataset(model, m, m_std, mask=mask)[0]
        # save dataset: residue
        if inps.uncertaintyQuantification == 'residue':
            ds_dict['residue'] = np.full(num_pixel, np.nan, dtype=DATA_TYPE)
            if inps.allowPartialDate:
                ds_dict['residue'][mask] = np.sqrt(e2[mask])
            else:
                ds_dict['residue'][mask] = np.sqrt(e2)

        for ds_name, data in ds_dict.items():
            writefile.write_hdf5_block(inps.outfile,
                                       data=data.reshape(box_len, box_wid),
                                       datasetName=ds_name,
                                       block=block)

        if inps.allowPartialDate:
            writefile.write_hdf5_block(inps.fitCoherenceFile,
                                       data=fit_coh.reshape(box_len, box_wid),
                                       datasetName='fitCoherence',
                                       block=block)

        # write - residual file
        if inps.save_res:
            print('calculating the time series residual ...')
            block = [0, num_date, box[1], box[3], box[0], box[2]]
            ts_res = np.full((num_date, box_len*box_wid), np.nan, dtype=np.float32)

            # calculate the time-series residual
            print(f'remove time functions: {inps.rm_timefuns}')
            if inps.allowPartialDate:
                rm_ds_inds = None if 'all' in inps.rm_timefuns else timefun_names2ds_names(inps.rm_timefuns, ds_dict.keys())[1]
                if num_pixel2inv_all > 0:
                    if rm_ds_inds is None:
                        ts_res[:, idx_pixel2inv_all] = ts_data[:, idx_pixel2inv_all] - np.dot(G, m[:, idx_pixel2inv_all])
                    else:
                        ts_res[:, idx_pixel2inv_all] = (
                            ts_data[:, idx_pixel2inv_all]
                            - np.dot(G[:, rm_ds_inds], m[rm_ds_inds, :][:, idx_pixel2inv_all])
                        )

                if num_pixel2inv_part > 0:
                    prog_bar = ptime.progressBar(maxValue=num_pixel2inv_part)
                    for j, idx in enumerate(idx_pixel2inv_part):
                        valid_flag, valid_date_list, Gi = _get_part_date_lstsq_inputs(
                            model=model,
                            date_list=inps.date_list,
                            dis_ts=ts_data[:, idx],
                            seconds=seconds,
                        )
                        if valid_flag is not None:
                            if rm_ds_inds is None:
                                ts_res[valid_flag, idx] = ts_data[valid_flag, idx] - np.dot(Gi, m[:, idx])
                            else:
                                ts_res[valid_flag, idx] = (
                                    ts_data[valid_flag, idx]
                                    - np.dot(Gi[:, rm_ds_inds], m[rm_ds_inds, idx])
                                )
                        prog_bar.update(j+1, every=200, suffix=f'{j+1}/{num_pixel2inv_part} pixels')
                    prog_bar.close()
            else:
                if 'all' in inps.rm_timefuns:
                    ts_res[:, mask] = ts_data - np.dot(G, m)[:, mask]
                else:
                    rm_ds_inds = timefun_names2ds_names(inps.rm_timefuns, ds_dict.keys())[1]
                    ts_res[:, mask] = ts_data - np.dot(G[:, rm_ds_inds], m[rm_ds_inds, :])[:, mask]

            # write to HDF5 file
            writefile.write_hdf5_block(inps.res_file,
                                       data=ts_res.reshape(num_date, box_len, box_wid),
                                       datasetName='timeseries',
                                       block=block)

    # used time
    m, s = divmod(time.time() - start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.')

    return inps.outfile


def timefun_names2ds_names(timefun_names, all_ds_names):
    """Grab all dataset names in the input ds_names_all related to the given time function names.

    Parameters: timefun_names - list(str), list of time function names of interest
                all_ds_names  - list(str), list of available dataset names
    Returns:    out_ds_names  - list(str), list of dataset names matched with the given time function names
                out_ds_inds   - list(int), list of index of the matched datasets in the inverted dataset list
    """
    # remove 1) dataset endswith "Std" and 2) "residue", as they are not directly inverted.
    inv_ds_names = [x for x in all_ds_names if not x.endswith('Std') and x not in ['residue']]

    timefun_name2ds_name_patterns = {
        'polynomial' : ['intercept', 'velocity', 'acceleration', 'poly*'],
        'periodic'   : ['annual*', 'semiAnnual*', 'period*'],
        'step'       : ['step*'],
        'polyline'   : ['velocityPost*'],
        'exp'        : ['exp*'],
        'log'        : ['log*'],
    }

    # grab all matched dataset names
    out_ds_names = []
    for timefun_name in timefun_names:
        ds_name_patterns = timefun_name2ds_name_patterns[timefun_name]
        for ds_name_pattern in ds_name_patterns:
            if ds_name_pattern.endswith('*'):
                # search dataset that matches with the wildcard pattern
                prefix = ds_name_pattern[:-1]
                out_ds_names += [x for x in inv_ds_names if x.startswith(prefix)]
            elif ds_name_pattern in inv_ds_names:
                # grab dataset that matches exactly with the input
                out_ds_names.append(ds_name_pattern)

    # calc the dataset index
    out_ds_inds = [inv_ds_names.index(x) for x in out_ds_names]

    return out_ds_names, out_ds_inds


def model2hdf5_dataset(model, m=None, m_std=None, mask=None, ds_shape=None, residue=None):
    """Prepare the estimated model parameters into a list of dicts for HDF5 dataset writing.
    Parameters: model        - dict,
                m            - 2D np.ndarray in (num_param, num_pixel) where num_pixel = 1 or length * width
                m_std        - 2D np.ndarray in (num_param, num_pixel) where num_pixel = 1 or length * width
                mask         - 1D np.ndarray in (num_pixel), mask of valid pixels
                ds_shape     - tuple of 2 int in (length, width)
    Returns:    ds_dict      - dict, dictionary of dataset values,     input for writefile.write_hdf5_block()
                ds_name_dict - dict, dictionary of dataset initiation, input for writefile.layout_hdf5()
                ds_unit_dict - dict, dictionary of dataset unit,       input for writefile.layout_hdf5()
    Examples:   # read input model parameters into dict
                model = read_inps2model(inps, date_list=inps.date_list)
                # for time series cube
                ds_name_dict, ds_unit_dict = model2hdf5_dataset(model, ds_shape=(200,300))[1:]
                ds_dict = model2hdf5_dataset(model, m, m_std, mask=mask)[0]
                # for time series point
                ds_unit_dict = model2hdf5_dataset(model)[2]
                ds_dict = model2hdf5_dataset(model, m, m_std)[0]
    """
    # deformation model info
    poly_deg   = model['polynomial']
    num_period = len(model['periodic'])
    num_step   = len(model['stepDate'])
    num_pline  = len(model['polyline'])
    num_exp    = sum(len(val) for key, val in model['exp'].items())

    # init output
    ds_dict = {}
    ds_name_dict = {}
    ds_unit_dict = {}

    # assign ds_dict ONLY IF m is not None
    if m is not None:
        num_pixel = m.shape[1] if m.ndim > 1 else 1
        m = m.reshape(-1, num_pixel)
        m_std = m_std.reshape(-1, num_pixel)

        # default mask
        if mask is None:
            mask = np.ones(num_pixel, dtype=np.bool_)
        else:
            mask = mask.flatten()

    # time func 1 - polynomial
    p0 = 0
    for i in range(poly_deg+1):
        # dataset name
        if i == 0:
            dsName = 'intercept'
            unit = 'm'
        elif i == 1:
            dsName = 'velocity'
            unit = 'm/year'
        elif i == 2:
            dsName = 'acceleration'
            unit = 'm/year^2'
        else:
            dsName = f'poly{i}'
            unit = f'm/year^{i}'

        # assign ds_dict
        if m is not None:
            ds_dict[dsName] = m[p0 + i, :]
            ds_dict[dsName+'Std'] = m_std[p0 + i, :]

        # assign ds_name/unit_dict
        ds_name_dict[dsName] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsName] = unit
        ds_name_dict[dsName+'Std'] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsName+'Std'] = unit

    # time func 2 - periodic
    p0 += poly_deg + 1
    for i in range(num_period):
        # dataset name
        period = model['periodic'][i]
        dsNameSuffixes = ['Amplitude', 'Phase']
        if period == 1:
            dsNames = [f'annual{x}' for x in dsNameSuffixes]
        elif period == 0.5:
            dsNames = [f'semiAnnual{x}' for x in dsNameSuffixes]
        else:
            dsNames = [f'period{period}Y{x}' for x in dsNameSuffixes]

        # calculate the amplitude and phase of the periodic signal
        # following equation (9-10) in Minchew et al. (2017, JGR)
        if m is not None:
            coef_cos = m[p0 + 2*i, :]
            coef_sin = m[p0 + 2*i + 1, :]
            period_amp = np.sqrt(coef_cos**2 + coef_sin**2)
            period_pha = np.full(num_pixel, np.nan, dtype=DATA_TYPE)
            # avoid divided by zero warning
            if not np.all(coef_sin[mask] == 0):
                # use atan2, instead of atan, to get phase within [-pi, pi]
                period_pha[mask] = np.arctan2(coef_cos[mask], coef_sin[mask])
            else:
                period_pha[mask] = 0.

            # assign ds_dict
            for dsName, data in zip(dsNames, [period_amp, period_pha]):
                ds_dict[dsName] = data

        # update ds_name/unit_dict
        ds_name_dict[dsNames[0]] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsNames[0]] = 'm'
        ds_name_dict[dsNames[1]] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsNames[1]] = 'radian'

    # time func 3 - step
    p0 += 2 * num_period
    for i in range(num_step):
        # dataset name
        dsName = 'step{}'.format(model['stepDate'][i])

        # assign ds_dict
        if m is not None:
            ds_dict[dsName] = m[p0+i, :]
            ds_dict[dsName+'Std'] = m_std[p0+i, :]

        # assign ds_name/unit_dict
        ds_name_dict[dsName] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsName] = 'm'
        ds_name_dict[dsName+'Std'] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsName+'Std'] = 'm'

    # time func 4 - polyline
    p0 += num_step
    for i in range(num_pline):
        # dataset name
        dsName = 'velocityPost{}'.format(model['polyline'][i])

        # assign ds_dict
        if m is not None:
            # save the cumulative velocity for each segment
            # starting from the velocity of the polynomial function
            vel = np.array(m[1, :], dtype=np.float32)
            vel_var = np.array(m_std[1, :]**2, dtype=np.float32)

            for j in range(i+1):
                vel += m[p0+j, :]
                # assuming the estimation of each polyline segment is
                # independent from each other, maybe unrealistic.
                vel_var += m_std[p0+j, :]**2

            ds_dict[dsName] = vel
            ds_dict[dsName+'Std'] = np.sqrt(vel_var)

        # assign ds_name/unit_dict
        ds_name_dict[dsName] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsName] = 'm/yr'
        ds_name_dict[dsName+'Std'] = [DATA_TYPE, ds_shape, None]
        ds_unit_dict[dsName+'Std'] = 'm/yr'

    # time func 5 - exponential
    p0 += num_pline
    i = 0
    for exp_onset in model['exp'].keys():
        for exp_tau in model['exp'][exp_onset]:
            # dataset name
            dsName = f'exp{exp_onset}Tau{exp_tau}D'

            # assign ds_dict
            if m is not None:
                ds_dict[dsName] = m[p0+i, :]
                ds_dict[dsName+'Std'] = m_std[p0+i, :]

            # assign ds_name/unit_dict
            ds_name_dict[dsName] = [DATA_TYPE, ds_shape, None]
            ds_unit_dict[dsName] = 'm'
            ds_name_dict[dsName+'Std'] = [DATA_TYPE, ds_shape, None]
            ds_unit_dict[dsName+'Std'] = 'm'

            # loop because each onset_time could have multiple char_time
            i += 1

    # time func 6 - logarithmic
    p0 += num_exp
    i = 0
    for log_onset in model['log'].keys():
        for log_tau in model['log'][log_onset]:
            # dataset name
            dsName = f'log{log_onset}Tau{log_tau}D'

            # assign ds_dict
            if m is not None:
                ds_dict[dsName] = m[p0+i, :]
                ds_dict[dsName+'Std'] = m_std[p0+i, :]

            # assign ds_name/unit_dict
            ds_name_dict[dsName] = [DATA_TYPE, ds_shape, None]
            ds_unit_dict[dsName] = 'm'
            ds_name_dict[dsName+'Std'] = [DATA_TYPE, ds_shape, None]
            ds_unit_dict[dsName+'Std'] = 'm'

            # loop because each onset_time could have multiple char_time
            i += 1

    return ds_dict, ds_name_dict, ds_unit_dict


def hdf5_dataset2model(ds_dict, ds_name_dict, ds_unit_dict):
    """ New function, just a place holder now, incomplete.

    Prepare the estimated model parameters into a list of dicts for HDF5 dataset writing.
    Parameters: model        - dict,
                m            - 2D np.ndarray in (num_param, num_pixel) where num_pixel = 1 or length * width
                m_std        - 2D np.ndarray in (num_param, num_pixel) where num_pixel = 1 or length * width
                mask         - 1D np.ndarray in (num_pixel), mask of valid pixels
                ds_shape     - tuple of 2 int in (length, width)
    Returns:    ds_dict      - dict, dictionary of dataset values,     input for writefile.write_hdf5_block()
                ds_name_dict - dict, dictionary of dataset initiation, input for writefile.layout_hdf5()
                ds_unit_dict - dict, dictionary of dataset unit,       input for writefile.layout_hdf5()
    Examples:   # read input model parameters into dict
                model = read_inps2model(inps, date_list=inps.date_list)
                # for time series cube
                ds_name_dict, ds_name_dict = model2hdf5_dataset(model, ds_shape=(200,300))[1:]
                ds_dict = model2hdf5_dataset(model, m, m_std, mask=mask)[0]
                # for time series point
                ds_unit_dict = model2hdf5_dataset(model)[2]
                ds_dict = model2hdf5_dataset(model, m, m_std)[0]

    # deformation model info
    poly_deg   = model['polynomial']
    num_period = len(model['periodic'])
    num_step   = len(model['stepDate'])
    num_exp    = sum(len(val) for key, val in model['exp'].items())

    """
    return
