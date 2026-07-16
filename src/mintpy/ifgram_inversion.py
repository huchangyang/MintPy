############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
# Parallel support added by David Grossman, Joshua Zahner  #
############################################################
# Recommend import:
#     from mintpy import ifgram_inversion as ifginv


import os
import time

import h5py
import numpy as np
from scipy import linalg  # more effieint than numpy.linalg

from mintpy.objects import cluster, ifgramStack
from mintpy.simulation import decorrelation as decor
from mintpy.utils import ptime, readfile, utils as ut, writefile

# key configuration parameter name
key_prefix = 'mintpy.networkInversion.'
config_keys = [
    'obsDatasetName',
    'numIfgram',
    'weightFunc',
    'maskDataset',
    'maskThreshold',
    'minRedundancy',
    'minNormVelocity',
    'allowPartialNetwork',
    'refDate',
]


################################################################################################


def run_or_skip(inps):
    print('-'*50)
    print('update mode: ON')
    flag = 'skip'

    # check output files vs input dataset
    if not all(os.path.isfile(i) for i in inps.outfile):
        flag = 'run'
        print(f'1) NOT ALL output files found: {inps.outfile}.')
    else:
        # check if time-series file is partly written using file size
        # since time-series file is not compressed
        with h5py.File(inps.outfile[0], 'r') as f:
            fsize_ref = f['timeseries'].size * 4
        fsize = os.path.getsize(inps.outfile[0])
        if fsize <= fsize_ref:
            flag = 'run'
            print(f'1) output file {inps.outfile[0]} is NOT fully written.')

        else:
            print(f'1) output files already exist: {inps.outfile}.')
            # check modification time
            with h5py.File(inps.ifgramStackFile, 'r') as f:
                ti = float(f[inps.obsDatasetName].attrs.get('MODIFICATION_TIME', os.path.getmtime(inps.ifgramStackFile)))
            to = min(os.path.getmtime(i) for i in inps.outfile)
            if ti > to:
                flag = 'run'
                print(f'2) output files are NOT newer than input dataset: {inps.obsDatasetName}.')
            else:
                print(f'2) output dataset is newer than input dataset: {inps.obsDatasetName}.')

    # check configuration
    if flag == 'skip':
        atr_ifg = readfile.read_attribute(inps.ifgramStackFile)
        atr_ts = readfile.read_attribute(inps.tsFile)
        inps.numIfgram = len(ifgramStack(inps.ifgramStackFile).get_date12_list(dropIfgram=True))
        meta_keys = [i for i in ['REF_Y', 'REF_X'] if i in atr_ts.keys()]

        if any(str(vars(inps)[key]) != atr_ts.get(key_prefix+key, 'None') for key in config_keys):
            flag = 'run'
            print(f'3) NOT all key configuration parameters are the same: {config_keys}')
        elif getattr(inps, 'refDate', None) and atr_ts.get('REF_DATE') != str(inps.refDate):
            flag = 'run'
            print(f'3) NOT the same REF_DATE: {atr_ts.get("REF_DATE")} vs {inps.refDate}')
        elif meta_keys and any(atr_ts[key] != atr_ifg[key] for key in meta_keys):
            flag = 'run'
            print(f'3) NOT all the metadata are the same: {meta_keys}')
        else:
            print(f'3) all key configuration parameters are the same: {config_keys}.')

    # result
    print(f'run or skip: {flag}.')
    return flag


################################# Time-series Estimator ###################################
def get_date_edges(date_list, date12_list):
    """Return date-index edges for each interferogram.

    Parameters: date_list   - list of str, SAR acquisition dates
                date12_list - list of str, interferogram date12 in YYYYMMDD_YYYYMMDD
    Returns:    edges       - 2D np.ndarray in size of (num_pair, 2), int16
    """
    date2idx = {date: i for i, date in enumerate(date_list)}
    edges = np.empty((len(date12_list), 2), dtype=np.int16)
    for i, date12 in enumerate(date12_list):
        date1, date2 = date12.split('_')
        edges[i, 0] = date2idx[date1]
        edges[i, 1] = date2idx[date2]
    return edges


def get_ref_connected_component(num_date, edges, ref_ind, min_redundancy=1.0):
    """Find dates connected to the global reference after redundancy filtering.

    Parameters: num_date       - int
                edges          - 2D np.ndarray in size of (num_obs, 2)
                ref_ind        - int, index of the global reference date
                min_redundancy - float, min number of ifgrams involving each date
    Returns:    connected      - 1D np.ndarray of bool in size of (num_date,)
                keep_edge      - 1D np.ndarray of bool in size of (num_obs,)
    """
    num_obs = edges.shape[0]
    if num_obs == 0:
        return np.zeros(num_date, dtype=np.bool_), np.zeros(0, dtype=np.bool_)

    keep_edge = np.ones(num_obs, dtype=np.bool_)
    while True:
        degree = np.zeros(num_date, dtype=np.float32)
        used = np.where(keep_edge)[0]
        if used.size:
            np.add.at(degree, edges[used, 0], 1)
            np.add.at(degree, edges[used, 1], 1)
        valid_date = degree >= min_redundancy
        new_keep = keep_edge & valid_date[edges[:, 0]] & valid_date[edges[:, 1]]
        if np.array_equal(new_keep, keep_edge):
            break
        keep_edge = new_keep

    connected = np.zeros(num_date, dtype=np.bool_)
    if not valid_date[ref_ind] or not np.any(keep_edge):
        return connected, keep_edge

    # BFS / DFS from the reference date on the surviving edges
    adj = [[] for _ in range(num_date)]
    for i in np.where(keep_edge)[0]:
        date1, date2 = int(edges[i, 0]), int(edges[i, 1])
        adj[date1].append(date2)
        adj[date2].append(date1)

    stack = [ref_ind]
    connected[ref_ind] = True
    while stack:
        node = stack.pop()
        for neigh in adj[node]:
            if not connected[neigh]:
                connected[neigh] = True
                stack.append(neigh)

    keep_edge = keep_edge & connected[edges[:, 0]] & connected[edges[:, 1]]
    return connected, keep_edge


def build_partial_design_matrix(edges, tbase, connected, keep_edge, ref_ind,
                                min_norm_velocity=True):
    """Build local design matrix on the reference-connected date subset.

    Parameters: edges             - 2D np.ndarray in size of (num_obs_all, 2)
                tbase             - 1D np.ndarray in size of (num_date,), years
                connected         - 1D np.ndarray of bool in size of (num_date,)
                keep_edge         - 1D np.ndarray of bool in size of (num_obs_all,)
                ref_ind           - int
                min_norm_velocity - bool
    Returns:    G                 - 2D np.ndarray local design matrix A or B
                local_date_inds   - 1D np.ndarray of int, global date indices in G order
                                    (includes ref_ind for mapping; G itself excludes ref
                                    for phase, or uses local intervals for velocity)
                local_tbase_diff  - 2D np.ndarray in size of (num_local_date-1, 1) or None
                local_ref_ind     - int, reference index in local_date_inds
                row_inds          - 1D np.ndarray of int, rows in the valid-obs arrays used
    """
    local_date_inds = np.where(connected)[0]
    row_inds = np.where(keep_edge)[0]
    if local_date_inds.size < 2 or row_inds.size < 1:
        return None, local_date_inds, None, 0, row_inds

    local_ref_ind = int(np.where(local_date_inds == ref_ind)[0][0])
    global2local = -np.ones(connected.size, dtype=np.int16)
    global2local[local_date_inds] = np.arange(local_date_inds.size, dtype=np.int16)

    local_edges = global2local[edges[row_inds]]
    local_tbase = tbase[local_date_inds]
    num_obs = row_inds.size
    num_local = local_date_inds.size

    if min_norm_velocity:
        G = np.zeros((num_obs, num_local), dtype=np.float32)
        for i in range(num_obs):
            ind1, ind2 = int(local_edges[i, 0]), int(local_edges[i, 1])
            if ind1 < ind2:
                G[i, ind1:ind2] = local_tbase[ind1 + 1:ind2 + 1] - local_tbase[ind1:ind2]
            else:
                G[i, ind2:ind1] = local_tbase[ind2:ind1] - local_tbase[ind2 + 1:ind1 + 1]
        G = G[:, :-1]
        local_tbase_diff = np.diff(local_tbase).reshape(-1, 1)
    else:
        G = np.zeros((num_obs, num_local), dtype=np.float32)
        for i in range(num_obs):
            ind1, ind2 = int(local_edges[i, 0]), int(local_edges[i, 1])
            G[i, ind1] = -1.
            G[i, ind2] = 1.
        G = np.hstack((G[:, :local_ref_ind], G[:, local_ref_ind + 1:]))
        local_tbase_diff = None

    return G, local_date_inds, local_tbase_diff, local_ref_ind, row_inds


def select_best_ref_date(stack_obj, obs_ds_name='unwrapPhase', mask_ds_name=None,
                         mask_threshold=0.4, water_mask_file=None, max_memory=4.0):
    """Select the SAR date with the largest spatial support for partial-network inversion.

    Support of a date is the number of land pixels where at least one valid observation
    involves that date. Ties keep the earliest date.
    """
    date_list = stack_obj.get_date_list(dropIfgram=True)
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    edges = get_date_edges(date_list, date12_list)
    num_date = len(date_list)
    num_pair = len(date12_list)
    support = np.zeros(num_date, dtype=np.int64)

    box_list = stack_obj.split2boxes(max_memory=max_memory)[0]
    for box in box_list:
        stack_obs = stack_obj.read(datasetName=obs_ds_name,
                                   box=box,
                                   dropIfgram=True,
                                   print_msg=False).reshape(num_pair, -1)
        if 'phase' in obs_ds_name.lower():
            stack_obs = stack_obs.astype(np.float32, copy=False)
            stack_obs[stack_obs == 0.] = np.nan
        stack_obs = mask_stack_obs(stack_obs, stack_obj, box,
                                   mask_ds_name=mask_ds_name,
                                   mask_threshold=mask_threshold,
                                   dropIfgram=True, print_msg=False)[0]

        mask = ~np.all(np.isnan(stack_obs), axis=0)
        if water_mask_file and os.path.isfile(water_mask_file):
            ds_names = readfile.get_dataset_list(water_mask_file)
            ds_name = [i for i in ds_names if i in ['waterMask', 'mask']][0]
            water_mask = readfile.read(water_mask_file, datasetName=ds_name, box=box)[0].flatten()
            mask *= water_mask != 0

        if not np.any(mask):
            continue

        valid = ~np.isnan(stack_obs[:, mask])
        date_hit = np.zeros((num_date, int(np.sum(mask))), dtype=np.bool_)
        for i in range(edges.shape[0]):
            obs_flag = valid[i]
            if not np.any(obs_flag):
                continue
            date_hit[edges[i, 0], obs_flag] = True
            date_hit[edges[i, 1], obs_flag] = True
        support += date_hit.sum(axis=1)

    best_ind = int(np.argmax(support))
    print('select reference date with the largest spatial support for partial network:')
    for i, date in enumerate(date_list):
        flag = ' <--' if i == best_ind else ''
        print(f'  {date}: {support[i]} pixels{flag}')
    return date_list[best_ind]


def estimate_timeseries(A, B, y, tbase_diff, weight_sqrt=None, min_norm_velocity=True,
                        rcond=1e-5, min_redundancy=1., inv_quality_name='temporalCoherence',
                        allow_partial_network=False, date_list=None, date12_list=None,
                        ref_date=None, tbase=None, print_msg=True):
    """Estimate time-series from a stack/network of interferograms with
    Least Square minimization on deformation phase / velocity.

    Problem: A X = y
    opt 1: X = np.dot(np.dot(numpy.linalg.inv(np.dot(A.T, A)), A.T), y)
    opt 2: X = np.dot(numpy.linalg.pinv(A), y)
    opt 3: X = np.dot(scipy.linalg.pinv(A), y)
    opt 4: X = scipy.linalg.lstsq(A, y)[0] [recommend and used]

    opt 4 supports weight.
    scipy.linalg provides more advanced and slighted faster performance than numpy.linalg.
    This function relies on the LAPACK routine gelsd. It computes the minimum-norm
    solution to a linear least squares problem using the singular value decomposition
    of A and a divide and conquer method.

    opt 4 is faster than opt 1/2/3 because it estimates X directly without calculating
    the A_inv matrix.

    opt 2/3 is better than opt 1 because numpy.linalg.inv() can not handle rank defiency of
    design matrix B

    Traditional Small BAseline Subsets (SBAS) algorithm (Berardino et al., 2002, IEEE-TGRS)
    is equivalent to the setting of:
        min_norm_velocity=True
        weight_sqrt=None

    Parameters: A                     - 2D np.ndarray in size of (num_pair, num_date-1)
                B                     - 2D np.ndarray in size of (num_pair, num_date-1),
                                        design matrix B, each row represents differential temporal
                                        baseline history between reference and secondary date of one interferogram
                y                     - 2D np.ndarray in size of (num_pair, num_pixel),
                                        phase/offset of all interferograms with no-data value: NaN.
                tbase_diff            - 2D np.ndarray in size of (num_date-1, 1),
                                        differential temporal baseline history, in the unit of years
                weight_sqrt           - 2D np.ndarray in size of (num_pair, num_pixel),
                                        square root of weight of all interferograms
                min_norm_velocity     - bool, assume minimum-norm deformation velocity, or not
                rcond                 - cut-off ratio of small singular values of A or B, to maintain robustness.
                                        It's recommend to >= 1e-5 by experience, to generate reasonable result.
                min_redundancy        - float, min redundancy defined as min num_pair for every SAR acquisition
                inv_quality_name      - str, inversion quality type/name
                                        temporalCoherence for phase
                                        residual          for offset
                                        no to turn OFF the calculation
                allow_partial_network - bool, if True, invert only the date subset connected to the
                                        global reference date; other dates are filled with NaN.
                date_list             - list of str, required when allow_partial_network=True
                date12_list           - list of str, required when allow_partial_network=True
                ref_date              - str, global reference date for partial-network inversion
                tbase                 - 1D np.ndarray in size of (num_date,), absolute temporal
                                        baseline in years, required when allow_partial_network=True
    Returns:    ts                - 2D np.ndarray in size of (num_date, num_pixel), phase time-series
                inv_quality       - 1D np.ndarray in size of (num_pixel) or float, temporal coherence (for phase) or residual (for offset)
                num_inv_obs       - 1D np.ndarray in size of (num_pixel) or int, number of observations (ifgrams / offsets)
                                    used during the inversion
    """

    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if weight_sqrt is not None:
        weight_sqrt = np.asarray(weight_sqrt, dtype=np.float32)
        if weight_sqrt.ndim == 1:
            weight_sqrt = weight_sqrt.reshape(-1, 1)

    if allow_partial_network:
        if date_list is None or date12_list is None or tbase is None:
            raise ValueError('date_list, date12_list and tbase are required for allow_partial_network!')
        num_date = len(date_list)
        if y.shape[0] != len(date12_list):
            raise ValueError(f'y rows ({y.shape[0]}) != date12_list size ({len(date12_list)})')
    else:
        y = y.reshape(A.shape[0], -1)
        if weight_sqrt is not None:
            weight_sqrt = weight_sqrt.reshape(A.shape[0], -1)
        num_date = A.shape[1] + 1
    num_pixel = y.shape[1]

    # initial output value
    ts = np.zeros((num_date, num_pixel), dtype=np.float32)
    if inv_quality_name == 'residual':
        inv_quality = np.full(num_pixel, np.nan, dtype=np.float32)
    else:
        inv_quality = np.zeros(num_pixel, dtype=np.float32)
    num_inv_obs = np.zeros(num_pixel, dtype=np.int16)

    ##### partial network: invert only the reference-connected date subset
    if allow_partial_network:
        ref_date = ref_date or date_list[0]
        if ref_date not in date_list:
            raise ValueError(f'reference date {ref_date} is not in date_list!')
        ref_ind = date_list.index(ref_date)
        edges_all = get_date_edges(date_list, date12_list)
        tbase = np.asarray(tbase, dtype=np.float32).reshape(-1)
        if tbase.size != num_date:
            raise ValueError(f'tbase size ({tbase.size}) != num_date ({num_date})')

        # pixel-by-pixel because each pixel may keep a different observation subset
        for i in range(num_pixel):
            yi = y[:, i:i+1]
            wi = weight_sqrt[:, i:i+1] if weight_sqrt is not None else None
            valid = (~np.isnan(yi[:, 0])).flatten()
            if not np.any(valid):
                ts[:, i] = np.nan
                continue

            edges = edges_all[valid]
            y_valid = yi[valid, :]
            w_valid = wi[valid, :] if wi is not None else None
            connected, keep_edge = get_ref_connected_component(
                num_date, edges, ref_ind, min_redundancy=min_redundancy)

            if not connected[ref_ind] or not np.any(keep_edge):
                ts[:, i] = np.nan
                continue

            G, local_date_inds, local_tbase_diff, local_ref_ind, row_inds = build_partial_design_matrix(
                edges, tbase, connected, keep_edge, ref_ind,
                min_norm_velocity=min_norm_velocity)
            if G is None or G.shape[0] < 1 or G.shape[1] < 1:
                ts[:, i] = np.nan
                continue

            y_inv = y_valid[row_inds, :]
            w_inv = w_valid[row_inds, :] if w_valid is not None else None
            try:
                if w_inv is not None:
                    X, e2 = linalg.lstsq(np.multiply(G, w_inv),
                                         np.multiply(y_inv, w_inv),
                                         cond=rcond)[:2]
                else:
                    X, e2 = linalg.lstsq(G, y_inv, cond=rcond)[:2]

                if inv_quality_name != 'no':
                    inv_quality[i] = calc_inv_quality(
                        G, X, y_inv, e2,
                        inv_quality_name=inv_quality_name,
                        weight_sqrt=w_inv,
                        print_msg=False)[0]

                ts[:, i] = np.nan
                if min_norm_velocity:
                    ts_local = np.zeros(local_date_inds.size, dtype=np.float32)
                    ts_local[1:] = np.cumsum((X * local_tbase_diff).flatten(), axis=0)
                    ts_local -= ts_local[local_ref_ind]
                    ts[local_date_inds, i] = ts_local
                else:
                    ts[ref_ind, i] = 0.
                    non_ref_local = np.r_[local_date_inds[:local_ref_ind],
                                          local_date_inds[local_ref_ind + 1:]]
                    ts[non_ref_local, i] = X.flatten()

                num_inv_obs[i] = y_inv.shape[0]
            except linalg.LinAlgError:
                ts[:, i] = np.nan

        if num_pixel == 1:
            return ts, float(np.atleast_1d(inv_quality)[0]), int(num_inv_obs[0])
        return ts, inv_quality, num_inv_obs

    ##### skip invalid phase/offset value [NaN]
    y, [A, B, weight_sqrt] = skip_invalid_obs(y, mat_list=[A, B, weight_sqrt])

    # original behavior: skip entire pixel if any date is under-connected
    if np.min(np.sum(A != 0., axis=0)) < min_redundancy:
        if inv_quality_name == 'residual':
            inv_quality = np.nan
        else:
            inv_quality = 0.
        return ts, inv_quality, 0

    ##### invert time-series
    try:
        if min_norm_velocity:
            if weight_sqrt is not None:
                X, e2 = linalg.lstsq(np.multiply(B, weight_sqrt),
                                     np.multiply(y, weight_sqrt),
                                     cond=rcond)[:2]
            else:
                X, e2 = linalg.lstsq(B, y, cond=rcond)[:2]

            if inv_quality_name != 'no':
                inv_quality = calc_inv_quality(B, X, y, e2,
                                               inv_quality_name=inv_quality_name,
                                               weight_sqrt=weight_sqrt,
                                               print_msg=print_msg)

            ts_diff = X * np.tile(tbase_diff, (1, num_pixel))
            ts[1:, :] = np.cumsum(ts_diff, axis=0)

        else:
            if weight_sqrt is not None:
                X, e2 = linalg.lstsq(np.multiply(A, weight_sqrt),
                                     np.multiply(y, weight_sqrt),
                                     cond=rcond)[:2]
            else:
                X, e2 = linalg.lstsq(A, y, cond=rcond)[:2]

            if inv_quality_name != 'no':
                inv_quality = calc_inv_quality(A, X, y, e2,
                                               inv_quality_name=inv_quality_name,
                                               weight_sqrt=weight_sqrt,
                                               print_msg=print_msg)

            ts[1:, :] = X

    except linalg.LinAlgError:
        pass

    num_inv_obs = A.shape[0]

    if num_pixel == 1:
        inv_quality = np.atleast_1d(inv_quality)[0]
        return ts, inv_quality, int(num_inv_obs)

    # broadcast scalar quality / obs count to 1D for multi-pixel standard path
    if np.ndim(inv_quality) == 0:
        inv_quality = np.full(num_pixel, inv_quality, dtype=np.float32)
    if np.ndim(num_inv_obs) == 0:
        num_inv_obs = np.full(num_pixel, num_inv_obs, dtype=np.int16)

    return ts, inv_quality, num_inv_obs


def estimate_timeseries_cov(G, y, y_std, rcond=1e-5, min_redundancy=1.0,
                            allow_partial_network=False, date_list=None,
                            date12_list=None, ref_date=None, tbase=None):
    """Estimate the time-series covariance from network of STD via linear propagation.
    Pixel by pixel only.

    For a system of linear equations: A X = y, propagate the STD from y to X.

    Parameters: G      - 2D np.ndarray in size of (num_pair, num_date-1), design matrix
                y      - 2D np.ndarray in size of (num_pair, 1), stack of obs
                y_std  - 2D np.ndarray in size of (num_pair, 1), stack of obs std. dev.
    Returns:    ts_cov - 2D np.ndarray in size of (num_date-1, num_date-1), time-series obs std. dev.
                         For partial network, disconnected non-reference dates are NaN.
    """
    y = np.asarray(y, dtype=np.float32)
    y_std = np.asarray(y_std, dtype=np.float32)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y_std.ndim == 1:
        y_std = y_std.reshape(-1, 1)

    if allow_partial_network:
        if date_list is None or date12_list is None or tbase is None:
            raise ValueError('date_list, date12_list and tbase are required for allow_partial_network!')
        ref_date = ref_date or date_list[0]
        ref_ind = date_list.index(ref_date)
        edges_all = get_date_edges(date_list, date12_list)
        tbase = np.asarray(tbase, dtype=np.float32).reshape(-1)
        num_date = len(date_list)
        num_param = num_date - 1
        ts_cov = np.full((num_param, num_param), np.nan, dtype=np.float32)

        valid = (~np.isnan(y[:, 0])).flatten()
        if not np.any(valid):
            return ts_cov

        edges = edges_all[valid]
        y_valid = y[valid, :]
        y_std_valid = y_std[valid, :]
        connected, keep_edge = get_ref_connected_component(
            num_date, edges, ref_ind, min_redundancy=min_redundancy)
        if not connected[ref_ind] or not np.any(keep_edge):
            return ts_cov

        # covariance is always propagated in the phase / A parameterization
        G_local, local_date_inds, _, local_ref_ind, row_inds = build_partial_design_matrix(
            edges, tbase, connected, keep_edge, ref_ind, min_norm_velocity=False)
        if G_local is None or G_local.shape[0] < 1 or G_local.shape[1] < 1:
            return ts_cov

        y_inv = y_valid[row_inds, :]
        y_std_inv = y_std_valid[row_inds, :]
        Gplus = linalg.pinv(G_local)
        stack_cov = np.diag(np.square(y_std_inv.flatten()))
        local_cov = np.linalg.multi_dot([Gplus, stack_cov, Gplus.T]).astype(np.float32)

        # map local non-reference dates back into the global (num_date-1) layout
        # whose columns follow date_list with ref_date removed
        global_non_ref = [i for i in range(num_date) if i != ref_ind]
        local_non_ref = np.r_[local_date_inds[:local_ref_ind],
                              local_date_inds[local_ref_ind + 1:]]
        global_inds = [global_non_ref.index(int(i)) for i in local_non_ref]
        for a, ga in enumerate(global_inds):
            for b, gb in enumerate(global_inds):
                ts_cov[ga, gb] = local_cov[a, b]
        return ts_cov

    num_param = G.shape[1]
    ts_cov = np.zeros((num_param, num_param), dtype=np.float32)
    y = y.reshape(G.shape[0], -1)
    y_std = y_std.reshape(G.shape[0], -1)

    # skip invalid phase/offset value [NaN]
    y, [G, y_std] = skip_invalid_obs(y, mat_list=[G, y_std])

    # check network redundancy: skip calculation if < threshold
    if G.shape[0] == 0 or np.min(np.sum(G != 0., axis=0)) < min_redundancy:
        return ts_cov

    Gplus = linalg.pinv(G)
    stack_cov = np.diag(np.square(y_std.flatten()))
    ts_cov = np.linalg.multi_dot([Gplus, stack_cov, Gplus.T])

    return ts_cov


def skip_invalid_obs(obs, mat_list):
    """Skip invalid observations in the stack of phase/offset and update corresponding matrices.
    This applies to the pixel-wised inversion only, because the region-wised inversion has valid obs in all pairs.
    Parameters: obs      - 2D np.ndarray in size of (num_pair, num_pixel),
                           observations (phase / offset) of all interferograms with no-data value: NaN.
                mat_list - list of 2D np.ndarray in size of (num_pair, *) or None
    Returns:    obs / mat_list
    """
    if np.any(np.isnan(obs)):
        # get flag matrix
        flag = (~np.isnan(obs[:, 0])).flatten()

        # update obs
        obs = obs[flag, :]

        # update list of matrice
        for i, mat in enumerate(mat_list):
            if mat is not None:
                mat_list[i] = mat[flag, :]

    return obs, mat_list


def calc_inv_quality(G, X, y, e2, inv_quality_name='temporalCoherence', weight_sqrt=None, print_msg=True):
    """Calculate the inversion quality of the time series estimation.

    Parameters: G                - 2D np.ndarray in size of (num_pair, num_date-1), design matrix A or B
                X                - 2D np.ndarray in size of (num_date-1, num_pixel), solution
                y                - 2D np.ndarray in size of (num_pair, num_pixel), phase or offset
                e2               - 1D np.ndarray in size of (num_pixel,), square of the sum of the 2-norm residual
                inv_quality_name - str, name of the inversion quality parameter
                                   temporalCoherence for phase
                                   residual          for offset
                weight_sqrt      - 2D np.ndarray in size of (num_pair, num_pixel),
                                   weight square root, None for un-weighted estimation.
    Returns:    inv_quality      - 1D np.ndarray in size of (num_pixel), temporalCoherence / residual
    """

    num_pair, num_pixel = y.shape
    inv_quality = np.zeros(num_pixel, dtype=np.float32)

    # chunk_size as the number of pixels
    chunk_size = int(ut.round_to_1(2e5 / num_pair))
    if num_pixel > chunk_size:
        num_chunk = int(np.ceil(num_pixel / chunk_size))
        if print_msg:
            print('calculating {} in chunks of {} pixels: {} chunks in total ...'.format(
                inv_quality_name, chunk_size, num_chunk))

        # loop over each chunk
        for i in range(num_chunk):
            c0 = i * chunk_size
            c1 = min((i + 1) * chunk_size, num_pixel)

            if inv_quality_name == 'temporalCoherence':
                #for phase
                e = y[:, c0:c1] - np.dot(G, X[:, c0:c1])
                inv_quality[c0:c1] = np.abs(np.sum(np.exp(1j*e), axis=0)) / num_pair

            elif inv_quality_name == 'residual':
                #for offset
                if weight_sqrt is not None:
                    # calculate the un-weighted residual for the weighted inversion
                    e = y[:, c0:c1] - np.dot(G, X[:, c0:c1])
                    inv_quality[c0:c1] = np.sqrt(np.sum(np.abs(e) ** 2, axis=0))

                else:
                    # use the un-weighted residual directly
                    inv_quality[c0:c1] = np.sqrt(e2[c0:c1]) if e2[c0:c1].size > 0 else np.nan

            # print out message
            chunk_step = max(1, int(ut.round_to_1(num_chunk / 5)))
            if print_msg and (i+1) % chunk_step == 0:
                print(f'chunk {i+1} / {num_chunk}')

    else:
        if inv_quality_name == 'temporalCoherence':
            #for phase
            e = y - np.dot(G, X)
            inv_quality = np.abs(np.sum(np.exp(1j*e), axis=0)) / num_pair

        elif inv_quality_name == 'residual':
            #for offset
            if weight_sqrt is not None:
                # calculate the un-weighted residual for the weighted inversion
                e = y - G.dot(X)
                inv_quality = np.sqrt(np.sum(np.abs(e) ** 2, axis=0))

            else:
                # use the un-weighted residual directly
                inv_quality = np.sqrt(e2) if e2.size > 0 else np.nan

        else:
            raise ValueError(f'un-recognized inversion quality name: {inv_quality_name}')

    return inv_quality



###################################### File IO ############################################
def check_design_matrix(ifgram_file, weight_func='var'):
    """
    Check Rank of Design matrix for weighted inversion
    """

    date12_list = ifgramStack(ifgram_file).get_date12_list(dropIfgram=True)
    A = ifgramStack.get_design_matrix4timeseries(date12_list)[0]
    if weight_func == 'no':
        if np.linalg.matrix_rank(A) < A.shape[1]:
            print('WARNING: singular design matrix! Inversion result can be biased!')
            print('continue using its SVD solution on all pixels')
    else:
        if np.linalg.matrix_rank(A) < A.shape[1]:
            print('ERROR: singular design matrix!')
            print('    Input network of interferograms is not fully connected!')
            print('    Can not invert the weighted least square solution.')
            print('You could try:')
            print('    1) Add more interferograms to make the network fully connected:')
            print('       a.k.a., no multiple subsets nor network islands')
            print("    2) Use '-w no' option for non-weighted SVD solution.")
            raise Exception()
    return A


def read_stack_obs(stack_obj, box, ref_phase, obs_ds_name='unwrapPhase', dropIfgram=True,
                   print_msg=True):
    """Read unwrapPhase / azimuthOffset / rangeOffset from ifgramStack file

    Parameters: stack_obj - ifgramStack object
                box       - tuple of 4 int
                ref_phase - 1D array or None
    Returns:    stack_obs - 2D array of unwrapPhase in size of (num_pair, num_pixel)
    """
    # Read unwrapPhase
    num_pair = stack_obj.get_size(dropIfgram=dropIfgram)[0]
    if print_msg:
        print(f'reading {obs_ds_name} in {box} * {num_pair} ...')
    stack_obs = stack_obj.read(datasetName=obs_ds_name,
                               box=box,
                               dropIfgram=dropIfgram,
                               print_msg=False).reshape(num_pair, -1)

    # read ref_phase
    if ref_phase is not None:
        # use input ref_phase array
        if print_msg:
            print('use input reference value')

    elif 'refPhase' in stack_obj.datasetNames:
        # read refPhase from file itself
        if print_msg:
            print('read reference phase from file')
        with h5py.File(stack_obj.file, 'r') as f:
            ref_phase = f['refPhase'][:]

    else:
        raise Exception('No reference value input/found on file!'+
                        ' unwrapped phase is not referenced!')

    # reference unwrapPhase
    for i in range(num_pair):
        mask = stack_obs[i, :] != 0.
        stack_obs[i, :][mask] -= ref_phase[i]
    return stack_obs


def mask_stack_obs(stack_obs, stack_obj, box, mask_ds_name=None, mask_threshold=0.4,
                   stack_std=None, dropIfgram=True, print_msg=True):
    """Mask input unwrapped phase by setting them to np.nan."""

    # Read/Generate Mask
    num_pair = stack_obj.get_size(dropIfgram=dropIfgram)[0]
    if mask_ds_name and mask_ds_name in stack_obj.datasetNames:
        if print_msg:
            print(f'reading {mask_ds_name} in {box} * {num_pair} ...')

        msk_data = stack_obj.read(datasetName=mask_ds_name,
                                  box=box,
                                  dropIfgram=dropIfgram,
                                  print_msg=False).reshape(num_pair, -1)
        # set all NaN values in coherence, connectComponent, offsetSNR to zero
        # to avoid RuntimeWarning msg during math operation
        msk_data[np.isnan(msk_data)] = 0
        msk = np.ones(msk_data.shape, dtype=np.bool_)

        if mask_ds_name in ['connectComponent']:
            msk *= msk_data != 0
            if print_msg:
                print(f'mask out pixels with {mask_ds_name} == 0 by setting them to NaN')

        elif mask_ds_name in ['coherence', 'offsetSNR']:
            msk *= msk_data >= mask_threshold
            if print_msg:
                print(f'mask out pixels with {mask_ds_name} < {mask_threshold} by setting them to NaN')

        elif mask_ds_name.endswith('OffsetStd'):
            msk *= msk_data <= mask_threshold
            if print_msg:
                print(f'mask out pixels with {mask_ds_name} > {mask_threshold} by setting them to NaN')

            # keep regions (ignore threshold-based masking) if:
            # 1. high SNR AND
            # 2. relaxed min STD AND
            # despite the above criteria, which is designed for small signals
            min_snr = 10
            obs_med = np.zeros((stack_obs.shape[0],1), dtype=np.float32)
            for i in range(stack_obs.shape[0]):
                obs_med[i] = np.nanmedian(stack_obs[i][msk[i]])
            obs_snr = np.abs(stack_obs - np.tile(obs_med, (1, stack_obs.shape[1]))) / (msk_data + 1e-5)
            msk_snr = np.multiply(msk_data <= mask_threshold * 5, obs_snr >= min_snr)
            msk[msk_snr] = 1
            if print_msg:
                print(f'keep pixels with {mask_ds_name} <= {mask_threshold*5} and SNR >= {min_snr}')

        else:
            raise ValueError(f'Un-recognized mask dataset name: {mask_ds_name}')

        # set values of mask-out pixels to NaN
        stack_obs[msk == 0.] = np.nan
        if stack_std is not None:
            stack_std[msk == 0.] = np.nan
        del msk_data, msk

    return stack_obs, stack_std


def read_coherence(stack_obj, box, dropIfgram=True, print_msg=True):
    """
    Read spatial coherence
    """

    num_pair = stack_obj.get_size(dropIfgram=dropIfgram)[0]
    if print_msg:
        print(f'reading coherence in {box} * {num_pair} ...')
    coh_data = stack_obj.read(datasetName='coherence',
                              box=box,
                              dropIfgram=dropIfgram,
                              print_msg=False).reshape(num_pair, -1)
    coh_data[np.isnan(coh_data)] = 0.
    return coh_data


def calc_weight_sqrt(stack_obj, box, weight_func='var', dropIfgram=True, chunk_size=100000):
    """Read coherence and calculate weight_sqrt from it, chunk by chunk to save memory
    """

    print('calculating weight from spatial coherence ...')

    # read coherence
    weight = read_coherence(stack_obj, box=box, dropIfgram=dropIfgram)
    num_pixel = weight.shape[1]

    if 'NCORRLOOKS' in stack_obj.metadata.keys():
        L = float(stack_obj.metadata['NCORRLOOKS'])
    else:
        # use the typical ratio of resolution vs pixel size of Sentinel-1 IW mode
        L = int(stack_obj.metadata['ALOOKS']) * int(stack_obj.metadata['RLOOKS'])
        L /= 1.94
    # make sure L >= 1
    L = max(np.rint(L).astype(int), 1)

    # convert coherence to weight chunk-by-chunk to save memory
    num_chunk = int(np.ceil(num_pixel / chunk_size))
    print(('convert coherence to weight in chunks of {c} pixels'
           ': {n} chunks in total ...').format(c=chunk_size, n=num_chunk))

    for i in range(num_chunk):
        c0 = i * chunk_size
        c1 = min((i + 1) * chunk_size, num_pixel)
        if i == 0:
            print_msg = True
        else:
            print_msg = False

        # calc weight from coherence
        weight[:, c0:c1] = decor.coherence2weight(weight[:, c0:c1],
                                                  weight_func,
                                                  L=L,
                                                  epsilon=5e-2,
                                                  print_msg=print_msg)
        weight[:, c0:c1] = np.sqrt(weight[:, c0:c1])

        # print out message
        if (i+1) % 1 == 0:
            print(f'chunk {i+1} / {num_chunk}')

    return weight


def get_design_matrix4std(stack_obj):
    """Get the design matrix for time-series STD calculation.
    Parameters: stack_obj - ifgramStack object
    Returns:    A_std     - 2D np.ndarray of float32 in size of (num_ifg, num_date-1)
                ref_ind   - int, index of the reference date in date_list
                ref_date  - str, reference date
                flag_std  - 1D np.ndarray of bool in size of (num_date)
    """

    # get ref_date from template file
    mintpy_dir = os.path.dirname(os.path.dirname(stack_obj.file))
    cfg_file = os.path.join(mintpy_dir, 'smallbaselineApp.cfg')
    ref_date = readfile.read_template(cfg_file)['mintpy.reference.date']

    #for reference_date.txt file
    if ref_date == 'auto':
        ref_date = os.path.join(mintpy_dir, 'reference_date.txt')
        if not os.path.isfile(ref_date):
            ref_date = None

    if ref_date and os.path.isfile(ref_date):
        ref_date = str(np.loadtxt(ref_date, dtype=bytes).astype(str))

    # check
    if not ref_date:
        msg = 'reference date is required for time-series STD calculation,'
        msg += 'but NOT found in mintpy.reference.date!'
        raise ValueError(msg)

    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date_list = stack_obj.get_date_list(dropIfgram=True)
    A_std = stack_obj.get_design_matrix4timeseries(date12_list, refDate=ref_date)[0]
    flag_std = np.array(date_list) != ref_date
    ref_ind = date_list.index(ref_date)

    return A_std, ref_ind, ref_date, flag_std



def run_ifgram_inversion_patch(ifgram_file, box=None, ref_phase=None, obs_ds_name='unwrapPhase',
                               weight_func='var', water_mask_file=None, min_norm_velocity=True,
                               mask_ds_name=None, mask_threshold=0.4, min_redundancy=1.0,
                               allow_partial_network=False, ref_date=None, calc_cov=False):
    """Invert one patch of an ifgram stack into timeseries.

    Parameters: ifgram_file           - str, interferograms stack HDF5 file, e.g. ./inputs/ifgramStack.h5
                box                   - tuple of 4 int, indicating (x0, y0, x1, y1) of the area of interest
                                        Set to None for the whole image
                ref_phase             - 1D array in size of (num_pair), or None
                obs_ds_name           - str, dataset to feed the inversion.
                weight_func           - str, weight function, choose in ['no', 'fim', 'var', 'coh']
                water_mask_file       - str, water mask filename if available, to skip inversion on water
                min_norm_velocity     - bool, minimize the residual phase or phase velocity
                mask_ds_name          - str, dataset name in ifgram_file used to mask unwrapPhase pixelwisely
                mask_threshold        - float, min coherence of pixels if mask_dataset_name='coherence'
                min_redundancy        - float, the min number of ifgrams for every acquisition.
                allow_partial_network - bool, invert only dates connected to ref_date; mark other
                                        dates as NaN instead of skipping the whole pixel.
                ref_date              - str, global reference date used by partial-network inversion
                calc_cov              - bool, calculate the time series covariance matrix.
    Returns:    ts                - 3D array in size of (num_date, num_row, num_col)
                ts_cov            - 4D array in size of (num_date, num_date, num_row, num_col) or None
                inv_quality       - 2D array in size of (num_row, num_col)
                num_inv_obs       - 2D array in size of (num_row, num_col)
                box               - tuple of 4 int
    Example:    run_ifgram_inversion_patch('ifgramStack.h5', box=(0,200,1316,400))
    """

    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    stack_dir, stack_base = os.path.split(ifgram_file)

    ## debug on a specific pixel
    #y, x = 555, 612
    #box = (x, y, x+1, y+1)


    ## 1. input info

    # size
    if box:
        num_row = box[3] - box[1]
        num_col = box[2] - box[0]
    else:
        num_row = stack_obj.length
        num_col = stack_obj.width
    num_pixel = num_row * num_col

    # get tbase_diff in the unit of year
    date_list = stack_obj.get_date_list(dropIfgram=True)
    num_date = len(date_list)
    tbase = np.array(ptime.date_list2tbase(date_list)[0], np.float32) / 365.25
    tbase_diff = np.diff(tbase).reshape(-1, 1)

    # design matrix
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    ref_date = ref_date or date_list[0]
    if allow_partial_network:
        A, B = stack_obj.get_design_matrix4timeseries(date12_list=date12_list, refDate=ref_date)[0:2]
    else:
        A, B = stack_obj.get_design_matrix4timeseries(date12_list=date12_list)[0:2]

    # 1.1 read / calculate weight and stack STD
    weight_sqrt = None
    stack_std = None

    if obs_ds_name.startswith(('unwrapPhase', 'ion')):
        # calculate weight
        if weight_func not in ['no', 'sbas']:
            weight_sqrt = calc_weight_sqrt(stack_obj, box,
                                           weight_func=weight_func,
                                           dropIfgram=True,
                                           chunk_size=100000)

        # calculate stack STD
        if calc_cov:
            if allow_partial_network:
                A_std = stack_obj.get_design_matrix4timeseries(date12_list, refDate=ref_date)[0]
                r0 = date_list.index(ref_date)
            else:
                A_std, r0 = get_design_matrix4std(stack_obj)[:2]
            r1 = r0 + 1
            if weight_func == 'var':
                stack_std = 1. / weight_sqrt
            else:
                stack_std = 1. / calc_weight_sqrt(stack_obj, box,
                                                  weight_func='var',
                                                  dropIfgram=True,
                                                  chunk_size=100000)

    elif 'offset' in obs_ds_name.lower():
        if calc_cov or weight_func == 'var':
            # calculate weight for offset
            print('reading {} in {} * {} ...'.format(obs_ds_name+'Std', box, len(date12_list)))
            weight_sqrt = stack_obj.read(datasetName=obs_ds_name+'Std',
                                         box=box,
                                         dropIfgram=True,
                                         print_msg=False).reshape(len(date12_list), -1)
            # handle anomalies
            weight_sqrt[np.isnan(weight_sqrt)] = 100.
            weight_sqrt[weight_sqrt < 0.005] = 0.005

            print('convert std. dev. to the inverse of variance')
            weight_sqrt = 1. / weight_sqrt  # use squre root of weight, to facilitate WLS, same as for phase.

            # prepare for Std time-series
            if calc_cov:
                if allow_partial_network:
                    A_std = stack_obj.get_design_matrix4timeseries(date12_list, refDate=ref_date)[0]
                    r0 = date_list.index(ref_date)
                else:
                    A_std, r0 = get_design_matrix4std(stack_obj)[:2]
                r1 = r0 + 1
                stack_std = 1. / weight_sqrt

            # reset weight_sqrt to None if no weighting is applied
            if weight_func in ['no', 'sbas']:
                weight_sqrt = None
            elif weight_func == 'var':
                pass
            else:
                raise ValueError(f'un-supported weight_func = {weight_func} for {obs_ds_name}!')
    else:
        raise ValueError(f'un-recognized observation dataset name: {obs_ds_name}')

    # 1.2 read / mask unwrapPhase and offset
    stack_obs = read_stack_obs(stack_obj, box, ref_phase,
                               obs_ds_name=obs_ds_name,
                               dropIfgram=True)

    # translate zero phase value to nan (no-data value)
    # because it's the common filled value used in phase masking
    if 'phase' in obs_ds_name.lower():
        stack_obs[stack_obs == 0.] = np.nan
        print(f'convert zero value in {obs_ds_name} to NaN (no-data value)')

    (stack_obs,
     stack_std) = mask_stack_obs(stack_obs, stack_obj, box,
                                 stack_std=stack_std,
                                 mask_ds_name=mask_ds_name,
                                 mask_threshold=mask_threshold,
                                 dropIfgram=True)

    # 1.3 mask of pixels to invert
    mask = np.ones(num_pixel, np.bool_)

    # 1.3.1 - Water Mask
    if water_mask_file:
        print(f'skip pixels (on the water) with zero value in file: {os.path.basename(water_mask_file)}')
        atr_msk = readfile.read_attribute(water_mask_file)
        len_msk, wid_msk = int(atr_msk['LENGTH']), int(atr_msk['WIDTH'])
        if (len_msk, wid_msk) != (stack_obj.length, stack_obj.width):
            raise ValueError('Input water mask file has different size from ifgramStack file.')

        dsNames = readfile.get_dataset_list(water_mask_file)
        dsName = [i for i in dsNames if i in ['waterMask', 'mask']][0]
        waterMask = readfile.read(water_mask_file, datasetName=dsName, box=box)[0].flatten()
        mask *= np.array(waterMask, dtype=np.bool_)
        del waterMask

    # 1.3.2 - Mask for NaN value in ALL ifgrams
    print(f'skip pixels with {obs_ds_name} = NaN in all interferograms')
    mask *= ~np.all(np.isnan(stack_obs), axis=0)

    # 1.3.3 Mask for zero quality measure (average spatial coherence/SNR)
    # usually due to lack of data in the processing
    if 'offset' in obs_ds_name.lower():
        inv_quality_name = 'residual'
        stack_quality_file = os.path.join(stack_dir, '../avgSpatialSNR.h5')

    elif stack_base.startswith('ion'):
        inv_quality_name = 'temporalCoherence'
        stack_quality_file = os.path.join(stack_dir, '../avgSpatialCohIon.h5')

    else:
        inv_quality_name = 'temporalCoherence'
        stack_quality_file = os.path.join(stack_dir, '../avgSpatialCoh.h5')

    if os.path.isfile(stack_quality_file):
        atr_stack = readfile.read_attribute(stack_quality_file)
        len_stack, wid_stack = int(atr_stack['LENGTH']), int(atr_stack['WIDTH'])
        if (len_stack, wid_stack) == (stack_obj.length, stack_obj.width):
            print(f'skip pixels with zero value in file: {os.path.basename(stack_quality_file)}')
            quality = readfile.read(stack_quality_file, box=box)[0].flatten()
            mask *= quality != 0.
            del quality

    # invert pixels on mask 1+2
    num_pixel2inv = int(np.sum(mask))
    idx_pixel2inv = np.where(mask)[0]
    print('number of pixels to invert: {} out of {} ({:.1f}%)'.format(
        num_pixel2inv, num_pixel, num_pixel2inv/num_pixel*100))


    ## 2. inversion

    # 2.1 initiale the output matrices
    ts = np.zeros((num_date, num_pixel), np.float32)
    ts_cov = np.zeros((num_date, num_date, num_pixel), np.float32) if calc_cov else None
    inv_quality = np.zeros(num_pixel, np.float32)
    if 'offset' in obs_ds_name.lower():
        inv_quality *= np.nan
    num_inv_obs = np.zeros(num_pixel, np.int16)

    # return directly if there is nothing to invert
    if num_pixel2inv < 1:
        ts = ts.reshape(num_date, num_row, num_col)
        ts_cov = ts_cov.reshape(num_date, num_date, num_row, num_col) if calc_cov else ts_cov
        inv_quality = inv_quality.reshape(num_row, num_col)
        num_inv_obs = num_inv_obs.reshape(num_row, num_col)
        return ts, ts_cov, inv_quality, num_inv_obs, box

    # common inversion options
    kwargs = {
        'A'                     : A,
        'B'                     : B,
        'tbase_diff'            : tbase_diff,
        'min_norm_velocity'     : min_norm_velocity,
        'min_redundancy'        : min_redundancy,
        'inv_quality_name'      : inv_quality_name,
        'allow_partial_network' : allow_partial_network,
        'date_list'             : date_list,
        'date12_list'           : date12_list,
        'ref_date'              : ref_date,
        'tbase'                 : tbase,
    }

    # 2.2 un-weighted inversion (classic SBAS)
    if weight_sqrt is None:
        msg = f'estimating time-series for pixels with valid {obs_ds_name} in'

        # a. split mask into mask_all/part_net
        # mask for valid (~NaN) observations in ALL ifgrams (share one B in sbas inversion)
        mask_all_net = np.all(~np.isnan(stack_obs), axis=0)
        mask_all_net *= mask
        mask_part_net = mask ^ mask_all_net
        del mask

        # b. invert once for all pixels with obs in all ifgrams
        if np.sum(mask_all_net) > 0:
            num_pixel2inv_all = int(np.sum(mask_all_net))
            print(f'{msg} all  ifgrams ({num_pixel2inv_all} pixels; {num_pixel2inv_all/num_pixel2inv*100:.1f}%) ...')

            # run
            tsi, inv_quali, num_obsi = estimate_timeseries(
                y=stack_obs[:, mask_all_net],
                weight_sqrt=None,
                **kwargs)

            # save result to output matrices
            ts[:, mask_all_net] = tsi
            inv_quality[mask_all_net] = inv_quali
            num_inv_obs[mask_all_net] = num_obsi

        # c. pixel-by-pixel for pixels with obs not in all ifgrams
        if np.sum(mask_part_net) > 0:
            num_pixel2inv_part = int(np.sum(mask_part_net))
            idx_pixel2inv_part = np.where(mask_part_net)[0]
            print(f'{msg} some ifgrams ({num_pixel2inv_part} pixels; {num_pixel2inv_part/num_pixel2inv*100:.1f}%) ...')

            prog_bar = ptime.progressBar(maxValue=num_pixel2inv_part)
            for i in range(num_pixel2inv_part):
                idx = idx_pixel2inv_part[i]

                # run
                tsi, inv_quali, num_obsi = estimate_timeseries(
                    y=stack_obs[:, idx],
                    weight_sqrt=None,
                    **kwargs)

                # save result to output matrices
                ts[:, idx] = tsi.flatten()
                inv_quality[idx] = inv_quali
                num_inv_obs[idx] = num_obsi

                prog_bar.update(i+1, every=200, suffix=f'{i+1}/{num_pixel2inv_part} pixels')
            prog_bar.close()

    # 2.3 weighted inversion - pixel-by-pixel
    else:
        print('estimating time-series via WLS pixel-by-pixel ...')
        prog_bar = ptime.progressBar(maxValue=num_pixel2inv)
        for i in range(num_pixel2inv):
            idx = idx_pixel2inv[i]

            # run
            tsi, inv_quali, num_obsi = estimate_timeseries(
                y=stack_obs[:, idx],
                weight_sqrt=weight_sqrt[:, idx],
                **kwargs)

            # save result to output matrices
            ts[:, idx] = tsi.flatten()
            inv_quality[idx] = inv_quali
            num_inv_obs[idx] = num_obsi

            prog_bar.update(i+1, every=200, suffix=f'{i+1}/{num_pixel2inv} pixels')
        prog_bar.close()
    del weight_sqrt

    # 2.4 time-series std. dev. - pixel-by-pixel
    if calc_cov:
        print('propagating std. dev. from network of interferograms to time-series (Yunjun et al., 2021, FRINGE) ...')
        prog_bar = ptime.progressBar(maxValue=num_pixel2inv)
        for i in range(num_pixel2inv):
            idx = idx_pixel2inv[i]
            ts_covi = estimate_timeseries_cov(
                A_std,
                y=stack_obs[:, idx],
                y_std=stack_std[:, idx],
                min_redundancy=min_redundancy,
                allow_partial_network=allow_partial_network,
                date_list=date_list,
                date12_list=date12_list,
                ref_date=ref_date,
                tbase=tbase,
            )

            # save result to output matrix
            # fill the (N-1xN-1) matrix into the (NxN) matrix
            ts_cov[:r0, :r0, idx] = ts_covi[:r0, :r0]
            ts_cov[:r0, r1:, idx] = ts_covi[:r0, r0:]
            ts_cov[r1:, :r0, idx] = ts_covi[r0:, :r0]
            ts_cov[r1:, r1:, idx] = ts_covi[r0:, r0:]

            prog_bar.update(i+1, every=200, suffix=f'{i+1}/{num_pixel2inv} pixels')
        prog_bar.close()
    del stack_obs
    del stack_std


    ## 3. prepare output

    # 3.1 reshape
    ts = ts.reshape(num_date, num_row, num_col)
    ts_cov = ts_cov.reshape(num_date, num_date, num_row, num_col) if calc_cov else ts_cov
    inv_quality = inv_quality.reshape(num_row, num_col)
    num_inv_obs = num_inv_obs.reshape(num_row, num_col)

    # 3.2 convert displacement unit to meter
    if obs_ds_name.startswith(('unwrapPhase','ion')):
        phase2range = -1 * float(stack_obj.metadata['WAVELENGTH']) / (4.*np.pi)
        ts *= phase2range
        ts_cov = ts_cov * np.abs(phase2range)**2 if calc_cov else ts_cov
        print('converting LOS phase unit from radian to meter')

    elif (obs_ds_name == 'azimuthOffset') & (stack_obj.metadata['PROCESSOR'] != 'cosicorr'):
        az_pixel_size = ut.azimuth_ground_resolution(stack_obj.metadata)
        az_pixel_size /= float(stack_obj.metadata['ALOOKS'])
        ts *= az_pixel_size
        ts_cov = ts_cov * az_pixel_size if calc_cov else ts_cov
        print(f'converting azimuth offset unit from pixel ({az_pixel_size:.2f} m) to meter')

    elif (obs_ds_name == 'rangeOffset') & (stack_obj.metadata['PROCESSOR'] != 'cosicorr'):
        rg_pixel_size = float(stack_obj.metadata['RANGE_PIXEL_SIZE'])
        rg_pixel_size /= float(stack_obj.metadata['RLOOKS'])
        ts *= -1 * rg_pixel_size
        ts_cov = ts_cov * rg_pixel_size if calc_cov else ts_cov
        print(f'converting range offset unit from pixel ({rg_pixel_size:.2f} m) to meter')

    return ts, ts_cov, inv_quality, num_inv_obs, box


def run_ifgram_inversion(inps):
    """Phase triangulatino of small baseline interferograms

    Parameters: inps - namespace
    Example:    inps = cmd_line_parse()
                run_ifgram_inversion(inps)
    """

    start_time = time.time()

    ## limit the number of threads in numpy/scipy to 1
    #   and save the original value for roll back afterwards
    #   because it does not increase the speed much but does increase the CPU usage significantly
    #   as shown in the test note below.
    # Dataset: SanFranSenDT42 version 1.x, patch 1 (505 x 510 x 1021) only
    # Machine 1: Mac (6 Intel i7 CPUs/cores in 2.6 GHz)
    # | dask (worker) | OMP_NUM_THREADS | Time used (sec) | CPU usage |
    # |   no   (0)    |        4        |      850        | 1 x 300%  |
    # |   no   (0)    |        1        |      930        | 1 x 100%  |
    # | local  (4)    |        4        |      580        | 4 x 250%  |
    # | local  (4)    |        1        |      420        | 4 x 100%  |
    # Machine 2: Linux local cluster (16 Intel E5 CPUs/cores in 2.4 GHz)
    # | dask (worker) | OMP_NUM_THREADS | Time used (sec) | CPU usage |
    # |   no   (0)    |        4        |     1400        | 1 x 400%  |
    # |   no   (0)    |        1        |     1250        | 1 x 100%  |
    # | local  (4)    |        4        |      750        | 4 x 320%  |
    # | local  (4)    |        1        |      500        | 4 x 100%  |
    num_threads_dict = cluster.set_num_threads("1")


    ## 1. input info

    stack_obj = ifgramStack(inps.ifgramStackFile)
    stack_obj.open(print_msg=False)
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date_list = stack_obj.get_date_list(dropIfgram=True)
    length, width = stack_obj.length, stack_obj.width

    # 1.1 read values on the reference pixel
    inps.refPhase = stack_obj.get_reference_phase(unwDatasetName=inps.obsDatasetName,
                                                  skip_reference=inps.skip_ref,
                                                  dropIfgram=True)

    # 1.2 design matrix / reference date
    if not inps.allowPartialNetwork:
        if getattr(inps, 'refDate', None) and inps.refDate != date_list[0]:
            print(f'WARNING: mintpy.networkInversion.refDate={inps.refDate} is used only when '
                  f'allowPartialNetwork=yes; fall back to the first date {date_list[0]}')
        inps.refDate = date_list[0]
    elif not getattr(inps, 'refDate', None):
        inps.refDate = select_best_ref_date(
            stack_obj,
            obs_ds_name=inps.obsDatasetName,
            mask_ds_name=inps.maskDataset,
            mask_threshold=inps.maskThreshold,
            water_mask_file=inps.waterMaskFile,
            max_memory=inps.maxMemory,
        )
    elif inps.refDate not in date_list:
        raise ValueError(f'input reference date {inps.refDate} is not in {date_list}')

    if inps.allowPartialNetwork:
        A = stack_obj.get_design_matrix4timeseries(date12_list, refDate=inps.refDate)[0]
    else:
        A = stack_obj.get_design_matrix4timeseries(date12_list)[0]
    num_pair, num_date = A.shape[0], A.shape[1]+1
    inps.numIfgram = num_pair

    if inps.calcCov:
        if inps.allowPartialNetwork:
            ref_date4std = inps.refDate
        else:
            ref_date4std = get_design_matrix4std(stack_obj)[2]
        ref_msg = f' with REF_DATE = {ref_date4std}'
    else:
        ref_msg = ''

    # 1.3 print key setup info
    msg = '-------------------------------------------------------------------------------\n'
    if inps.minNormVelocity:
        suffix = 'deformation velocity'
    else:
        suffix = 'deformation phase'
    msg += f'least-squares solution with L2 min-norm on: {suffix}\n'
    msg += f'minimum redundancy: {inps.minRedundancy}\n'
    msg += f'weight function: {inps.weightFunc}\n'
    msg += f'calculate covariance: {inps.calcCov} {ref_msg}\n'
    msg += f'allow partial network: {inps.allowPartialNetwork}\n'
    msg += f'reference date: {inps.refDate}\n'

    if inps.maskDataset:
        if inps.maskDataset in ['connectComponent']:
            suffix = f'{inps.maskDataset} == 0'
        elif inps.maskDataset in ['coherence', 'offsetSNR']:
            suffix = f'{inps.maskDataset} < {inps.maskThreshold}'
        elif inps.maskDataset.endswith('OffsetStd'):
            suffix = f'{inps.maskDataset} > {inps.maskThreshold}'
        else:
            raise ValueError(f'Un-recognized mask dataset name: {inps.maskDataset}')
        msg += f'mask out pixels with: {suffix}\n'
    else:
        msg += 'mask: no\n'

    if np.linalg.matrix_rank(A) < A.shape[1]:
        msg += '***WARNING: the network is NOT fully connected.\n'
        msg += '\tInversion result can be biased!\n'
        msg += '\tContinue to use SVD to resolve the offset between different subsets.\n'
    msg += '-------------------------------------------------------------------------------'
    print(msg)

    print(f'number of interferograms: {num_pair}')
    print(f'number of acquisitions  : {num_date}')
    print(f'number of lines   : {length}')
    print(f'number of columns : {width}')

    ## 2. prepare output

    # 2.1 metadata
    meta = dict(stack_obj.metadata)
    for key in config_keys:
        meta[key_prefix+key] = str(vars(inps)[key])

    meta['FILE_TYPE'] = 'timeseries'
    meta['UNIT'] = 'm'
    meta['REF_DATE'] = inps.refDate

    # 2.2 instantiate time-series
    dates = np.array(date_list, dtype=np.bytes_)
    pbase = stack_obj.get_perp_baseline_timeseries(dropIfgram=True)
    ds_name_dict = {
        "date"       : [dates.dtype, (num_date,), dates],
        "bperp"      : [np.float32,  (num_date,), pbase],
        "timeseries" : [np.float32,  (num_date, length, width), None],
    }
    writefile.layout_hdf5(inps.tsFile, ds_name_dict, metadata=meta)

    if inps.calcCov:
        fbase = os.path.splitext(inps.tsFile)[0]
        fbase += 'Decor' if inps.obsDatasetName.startswith('unwrapPhase') else ''
        tsStdFile = f'{fbase}Cov.h5'
        meta['REF_DATE'] = ref_date4std
        ds_name_dict = {"date"       : [dates.dtype, (num_date,), dates],
                        "timeseries" : [np.float32,  (num_date, num_date, length, width), None]}
        writefile.layout_hdf5(tsStdFile, ds_name_dict, meta)

    # 2.3 instantiate invQualifyFile: temporalCoherence / residualInv
    if 'residual' in os.path.basename(inps.invQualityFile).lower():
        inv_quality_name = 'residual'
        meta['UNIT'] = 'pixel'
    else:
        inv_quality_name = 'temporalCoherence'
        meta['UNIT'] = '1'
    meta['FILE_TYPE'] = inv_quality_name
    meta.pop('REF_DATE')
    ds_name_dict = {meta['FILE_TYPE'] : [np.float32, (length, width)]}
    writefile.layout_hdf5(inps.invQualityFile, ds_name_dict, metadata=meta)

    # 2.4 instantiate number of inverted observations
    meta['FILE_TYPE'] = 'mask'
    meta['UNIT'] = '1'
    # ignore NO_DATA_VALUE from ifgram stack file here as 1) it makes sense
    # and 2) to avoid the weird error at https://github.com/insarlab/MintPy/issues/1185
    if 'NO_DATA_VALUE' in meta.keys():
        meta.pop('NO_DATA_VALUE')

    ds_name_dict = {"mask" : [np.float32, (length, width)]}
    writefile.layout_hdf5(inps.numInvFile, ds_name_dict, metadata=meta)

    ## 3. run the inversion / estimation and write to disk

    # 3.1 split ifgram_file into blocks to save memory
    box_list, num_box = stack_obj.split2boxes(max_memory=inps.maxMemory)

    # 3.2 prepare the input arguments for *_patch()
    data_kwargs = {
        "ifgram_file"           : inps.ifgramStackFile,
        "ref_phase"             : inps.refPhase,
        "obs_ds_name"           : inps.obsDatasetName,
        "weight_func"           : inps.weightFunc,
        "min_norm_velocity"     : inps.minNormVelocity,
        "water_mask_file"       : inps.waterMaskFile,
        "mask_ds_name"          : inps.maskDataset,
        "mask_threshold"        : inps.maskThreshold,
        "min_redundancy"        : inps.minRedundancy,
        "allow_partial_network" : inps.allowPartialNetwork,
        "ref_date"              : inps.refDate,
        "calc_cov"              : inps.calcCov,
    }

    # 3.3 invert / write block-by-block
    for i, box in enumerate(box_list):
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]
        if num_box > 1:
            print(f'\n------- processing patch {i+1} out of {num_box} --------------')
            print(f'box width:  {box_wid}')
            print(f'box length: {box_len}')

        # update box argument in the input data
        data_kwargs['box'] = box
        if not inps.cluster:
            # non-parallel
            ts, ts_cov, inv_quality, num_inv_obs = run_ifgram_inversion_patch(**data_kwargs)[:-1]

        else:
            # parallel
            print('\n\n------- start parallel processing using Dask -------')

            # initiate the output data
            ts = np.zeros((num_date, box_len, box_wid), np.float32)
            ts_cov = np.zeros((num_date, num_date, box_len, box_wid), np.float32) if inps.calcCov else None
            inv_quality = np.zeros((box_len, box_wid), np.float32)
            num_inv_obs = np.zeros((box_len, box_wid), np.float32)

            # initiate dask cluster and client
            cluster_obj = cluster.DaskCluster(inps.cluster, inps.numWorker, config_name=inps.config)
            cluster_obj.open()

            # run dask
            ts, ts_cov, inv_quality, num_inv_obs = cluster_obj.run(
                func=run_ifgram_inversion_patch,
                func_data=data_kwargs,
                results=[ts, ts_cov, inv_quality, num_inv_obs])

            # close dask cluster and client
            cluster_obj.close()

            print('------- finished parallel processing -------\n\n')

        # write the block to disk
        # with 3D block in [z0, z1, y0, y1, x0, x1]
        # and  2D block in         [y0, y1, x0, x1]
        # time-series - 3D
        block = [0, num_date, box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(inps.tsFile,
                                   data=ts,
                                   datasetName='timeseries',
                                   block=block)

        if inps.calcCov:
            block = [0, num_date, 0, num_date, box[1], box[3], box[0], box[2]]
            writefile.write_hdf5_block(tsStdFile,
                                       data=ts_cov,
                                       datasetName='timeseries',
                                       block=block)

        # temporal coherence - 2D
        block = [box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(inps.invQualityFile,
                                   data=inv_quality,
                                   datasetName=inv_quality_name,
                                   block=block)

        # number of inverted obs - 2D
        writefile.write_hdf5_block(inps.numInvFile,
                                   data=num_inv_obs,
                                   datasetName='mask',
                                   block=block)

        if num_box > 1:
            m, s = divmod(time.time() - start_time, 60)
            print(f'time used: {m:02.0f} mins {s:02.1f} secs.\n')

    # 3.4 update output data on the reference pixel (for phase)
    if not inps.skip_ref:
        # grab ref_y/x
        ref_y = int(stack_obj.metadata['REF_Y'])
        ref_x = int(stack_obj.metadata['REF_X'])
        print('-'*50)
        print(f'update values on the reference pixel: ({ref_y}, {ref_x})')

        ref_val = 0 if inv_quality_name == 'residual' else 1
        print(f'set {inv_quality_name} on the reference pixel to {ref_val}.')
        with h5py.File(inps.invQualityFile, 'r+') as f:
            f[inv_quality_name][ref_y, ref_x] = ref_val

        print(f'set  # of observations on the reference pixel as {num_pair}')
        with h5py.File(inps.numInvFile, 'r+') as f:
            f['mask'][ref_y, ref_x] = num_pair

    # roll back to the original number of threads
    cluster.roll_back_num_threads(num_threads_dict)

    m, s = divmod(time.time() - start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.\n')
    return
