############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Yang Qingyue, Hu Changyang, 2024                 #
############################################################


import os
import time
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial

import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline, griddata
from scipy.interpolate import RegularGridInterpolator
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Install with 'pip install tqdm' for progress bars.")

from mintpy.objects import timeseries
from mintpy.utils import readfile, writefile

############################################################################

def print_progress(message, start_time=None, current_step=None, total_steps=None):
    """Print progress information"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if start_time and current_step and total_steps:
        elapsed = time.time() - start_time
        if current_step > 0:
            estimated_total = elapsed * total_steps / current_step
            remaining = estimated_total - elapsed
            remaining_str = str(timedelta(seconds=int(remaining)))
            print(f"[{timestamp}] {message} - Step {current_step}/{total_steps} - Estimated remaining time: {remaining_str}")
        else:
            print(f"[{timestamp}] {message} - Step {current_step}/{total_steps}")
    else:
        print(f"[{timestamp}] {message}")

def read_topographic_data(geom_file):
    print_progress('Reading topographic data...')
    dem = readfile.read(geom_file, datasetName='height', print_msg=False)[0]
    Na, Nr = dem.shape
    
    # Check if there are any missing values, if not, return directly
    if not np.any(np.isnan(dem)):
        print_progress(f'Topographic data reading completed - Size: {Na} x {Nr} (no missing data)')
        return dem
    
    dem_filled = scipy.ndimage.interpolation.map_coordinates(
        dem, 
        np.indices(dem.shape), 
        order=0, 
        mode='nearest'
    )
    
    print_progress(f'Topographic data reading completed - Size: {Na} x {Nr}')
    return dem_filled

def process_single_window(args):
    """Process a single window - designed for multiprocessing"""
    (na, nac, nr, nrc, n, dem, ts_data, mask, w, w1, w2, truncate, 
     res_step, Iteration, rg, step, Na, Nr) = args
    
    # Calculate window boundaries
    u = int(max(1, nac - w))
    d = int(min(Na, nac + w))
    l = int(max(1, nrc - w))
    r = int(min(Nr, nrc + w))
    
    # Create window mask
    tmp = np.full((Na, Nr), np.nan)
    tmp[u-1:d, l-1:r] = 1
    mask_process = mask.copy()
    mask_process[np.isnan(tmp)] = np.nan
    
    # Iterative linear fitting
    mask_std = np.ones((Na, Nr)) * mask_process
    coe = [0.0, 0.0]  # Initialize default values
    
    for i in range(Iteration):
        phase_tmp = ts_data[n, :, :] * mask_std
        dem_tmp = dem * mask_std
        valid_idx = ~np.isnan(phase_tmp) & ~np.isnan(dem_tmp)
        
        if np.nansum(valid_idx) == 0:
            break
            
        try:
            coe = np.polyfit(dem[valid_idx], phase_tmp[valid_idx], 1)
            cor_tmp = phase_tmp - coe[0]*dem_tmp - coe[1]
            
            # Update mask
            max_tmp = np.nanmax(np.abs(cor_tmp))
            max_tmp = max_tmp - res_step if max_tmp > res_step else max_tmp
            mask_std[np.abs(cor_tmp) > max_tmp] = np.nan
            
            if np.nansum(~np.isnan(mask_std)) < np.nansum(~np.isnan(mask_process)) / 10:
                break
                
        except (ValueError, np.linalg.LinAlgError):
            break
    
    # Texture correlation
    mask_tmp = mask[u-1:d, l-1:r]
    A = dem[u-1:d, l-1:r]
    A_LP = scipy.ndimage.gaussian_filter(A, sigma=w1, truncate=truncate, mode='nearest')
    A = A - A_LP
    A_line = A[~np.isnan(mask_tmp)]
    
    if len(A_line) == 0 or np.all(A_line == 0):
        return na, nr, n, coe[0], coe[1], coe[0]  # k_LLF, d_LLF, k_htc
        
    A_line = A_line / np.linalg.norm(A_line)
    
    # Search for best slope
    k_left = -rg * step + coe[0]
    k_right = rg * step + coe[0]
    
    # Test left and right boundaries
    phase_left = ts_data[n, u-1:d, l-1:r] - k_left * dem[u-1:d, l-1:r]
    phase_right = ts_data[n, u-1:d, l-1:r] - k_right * dem[u-1:d, l-1:r]
    
    conv_left = calculate_texture_correlation(phase_left, mask_tmp, A_line, w1, truncate)
    conv_right = calculate_texture_correlation(phase_right, mask_tmp, A_line, w1, truncate)
    
    # Determine search range
    if conv_left >= conv_right:
        k_range = np.arange(0, rg + 1) * step + coe[0]
    else:
        k_range = np.arange(-rg, 1) * step + coe[0]
    
    # Search for best slope
    correlations = np.zeros(len(k_range))
    for j, ki in enumerate(k_range):
        phase_test = ts_data[n, u-1:d, l-1:r] - ki * dem[u-1:d, l-1:r]
        correlations[j] = calculate_texture_correlation(phase_test, mask_tmp, A_line, w1, truncate)
    
    best_idx = np.argmin(correlations)
    k_htc = k_range[best_idx]
    
    return na, nr, n, coe[0], coe[1], k_htc  # k_LLF, d_LLF, k_htc

def estimate_local_slope_parallel(dem, ts_data, inps, n_ref, meta):
    """Parallel version of local slope estimation using multiprocessing"""
    
    print_progress('Starting parallel local slope estimation...')
    
    # Parameters
    w1 = 9
    w2 = 13
    res_step = 0.5
    Iteration = 10
    rg = 40
    step = 0.0001
    truncate = ((w2 - 1)/2 - 0.5)/w1

    lamda = float(meta['WAVELENGTH'])
    ref_y = int(meta['REF_Y'])
    ref_x = int(meta['REF_X'])

    N, Na, Nr = ts_data.shape
    W = inps.windowsize
    w = (W-1)/2
    overlap = round(inps.overlapratio*(2*w+1))

    print_progress(f'Reading mask file: {inps.mask_file}')
    mask_coh = readfile.read(inps.mask_file, datasetName='mask')[0]
    mask_coh = np.where(mask_coh < 0.87, np.nan, 1)
    mask = mask_coh

    Na_C = np.arange(w + 1, Na + 1, 2 * w - overlap)
    Nr_C = np.arange(w + 1, Nr + 1, 2 * w - overlap)
    
    total_patches = len(Na_C) * len(Nr_C)
    print_progress(f'Starting to process {total_patches} local windows with parallel processing...')

    # Preprocess time series data
    ts_data = 4 * np.pi / lamda * ts_data[:N, :, :]
    reference_value = ts_data[:, ref_y-1, ref_x-1]
    ts_data = ts_data - reference_value[:, np.newaxis, np.newaxis]
    ts_data[np.isnan(ts_data)] = 0

    # Initialize result arrays
    k_LLF = np.zeros((N, len(Na_C), len(Nr_C)))
    d_LLF = np.zeros((N, len(Na_C), len(Nr_C)))
    k_htc = np.zeros((N, len(Na_C), len(Nr_C)))

    # Prepare arguments for multiprocessing
    args_list = []
    for na, nac in enumerate(Na_C):
        for nr, nrc in enumerate(Nr_C):
            for n in range(N):
                if n != n_ref:  # Skip reference time point
                    args = (na, nac, nr, nrc, n, dem, ts_data, mask, w, w1, w2, 
                           truncate, res_step, Iteration, rg, step, Na, Nr)
                    args_list.append(args)
    
    # Use user-specified number of processes
    num_processes = inps.num_processes
    print_progress(f'Using {num_processes} parallel processes...')
    
    # Process with multiprocessing
    if TQDM_AVAILABLE:
        with tqdm(total=len(args_list), desc="Processing windows", unit="window") as pbar:
            with mp.Pool(processes=num_processes) as pool:
                results = []
                for result in pool.imap_unordered(process_single_window, args_list):
                    results.append(result)
                    pbar.update(1)
    else:
        start_time = time.time()
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_single_window, args_list)
        
        total_time = time.time() - start_time
        print_progress(f'Parallel processing completed in {timedelta(seconds=int(total_time))}')
    
    # Collect results
    for result in results:
        na, nr, n, k_llf, d_llf, k_htc_val = result
        k_LLF[n, na, nr] = k_llf
        d_LLF[n, na, nr] = d_llf
        k_htc[n, na, nr] = k_htc_val
    
    print_progress('Parallel local slope estimation completed')
    return k_htc

def precalculate_windows(Na_C, Nr_C, w, Na, Nr, mask):
    """Pre-calculate all window masks and coordinates for vectorized processing"""
    
    window_masks = []
    window_coords = []
    
    for na, nac in enumerate(Na_C):
        for nr, nrc in enumerate(Nr_C):
            # Calculate window boundaries
            u = max(1, nac - w)
            d = min(Na, nac + w)
            l = max(1, nrc - w)
            r = min(Nr, nrc + w)
            
            # Create window mask
            window_mask = np.full((Na, Nr), np.nan)
            window_mask[u-1:d, l-1:r] = 1
            window_mask = mask * window_mask
            
            # Store window information
            window_masks.append(window_mask)
            window_coords.append((u, d, l, r))
    
    return window_masks, window_coords

def process_time_point_vectorized(n, phase_data, dem, window_masks, window_coords, 
                                w1, w2, truncate, res_step, Iteration, rg, step):
    """Process all windows for a single time point using vectorized operations"""
    
    num_windows = len(window_masks)
    k_LLF_results = np.zeros(num_windows)
    d_LLF_results = np.zeros(num_windows)
    k_htc_results = np.zeros(num_windows)
    
    # Vectorized linear fitting for all windows
    k_LLF_results, d_LLF_results = vectorized_linear_fitting(
        phase_data, dem, window_masks, Iteration, res_step
    )
    
    # Vectorized texture correlation for all windows
    k_htc_results = vectorized_texture_correlation(
        phase_data, dem, window_masks, window_coords, k_LLF_results,
        w1, w2, truncate, rg, step
    )
    
    return k_LLF_results, d_LLF_results, k_htc_results

def vectorized_linear_fitting(phase_data, dem, window_masks, Iteration, res_step):
    """Vectorized linear fitting for all windows"""
    
    num_windows = len(window_masks)
    k_results = np.zeros(num_windows)
    d_results = np.zeros(num_windows)
    
    # Process all windows simultaneously
    for i, mask in enumerate(window_masks):
        # Initialize
        mask_std = mask.copy()
        
        # Iterative linear fitting
        for iter_count in range(Iteration):
            # Apply mask
            phase_masked = phase_data * mask_std
            dem_masked = dem * mask_std
            
            # Find valid indices
            valid_idx = ~np.isnan(phase_masked) & ~np.isnan(dem_masked)
            
            if np.nansum(valid_idx) == 0:
                break
                
            try:
                # Linear fitting
                coe = np.polyfit(dem[valid_idx], phase_masked[valid_idx], 1)
                k_results[i] = coe[0]
                d_results[i] = coe[1]
                
                # Calculate residuals
                cor_tmp = phase_masked - coe[0]*dem_masked - coe[1]
                
                # Update mask
                max_tmp = np.nanmax(np.abs(cor_tmp))
                max_tmp = max_tmp - res_step if max_tmp > res_step else max_tmp
                mask_std[np.abs(cor_tmp) > max_tmp] = np.nan
                
                # Check convergence
                if np.nansum(~np.isnan(mask_std)) < np.nansum(~np.isnan(mask)) / 10:
                    break
                    
            except (ValueError, np.linalg.LinAlgError):
                k_results[i] = 0.0
                d_results[i] = 0.0
                break
    
    return k_results, d_results

def vectorized_texture_correlation(phase_data, dem, window_masks, window_coords, 
                                 k_LLF, w1, w2, truncate, rg, step):
    """Vectorized texture correlation for all windows"""
    
    num_windows = len(window_masks)
    k_htc_results = np.zeros(num_windows)
    
    # Process all windows
    for i, (mask, (u, d, l, r)) in enumerate(zip(window_masks, window_coords)):
        try:
            # 这里使用了错误的索引
            mask_window = mask[u-1:d, l-1:r]  # 错误：应该使用全局坐标
            dem_window = dem[u-1:d, l-1:r]    # 错误：应该使用全局坐标
            phase_window = phase_data[u-1:d, l-1:r]  # 错误：应该使用全局坐标
            
            # Calculate topography texture (A)
            A = dem_window.copy()
            A_LP = scipy.ndimage.gaussian_filter(A, sigma=w1, truncate=truncate, mode='nearest')
            A = A - A_LP
            
            # Normalize A
            A_valid = A[~np.isnan(mask_window)]
            if len(A_valid) == 0 or np.all(A_valid == 0):
                k_htc_results[i] = k_LLF[i]
                continue
                
            A_valid = A_valid / np.linalg.norm(A_valid)
            
            # Search for best slope using vectorized operations
            k_htc_results[i] = find_best_slope_vectorized(
                phase_window, dem_window, mask_window, A_valid, k_LLF[i], 
                w1, truncate, rg, step
            )
            
        except Exception:
            k_htc_results[i] = k_LLF[i]
    
    return k_htc_results

def find_best_slope_vectorized(phase_window, dem_window, mask_window, A_valid, 
                              k_initial, w1, truncate, rg, step):
    """Find best slope using vectorized operations"""
    
    # Determine search direction
    k_left = -rg * step + k_initial
    k_right = rg * step + k_initial
    
    # Test left and right boundaries
    phase_left = phase_window - k_left * dem_window
    phase_right = phase_window - k_right * dem_window
    
    # Calculate texture correlation for boundaries
    conv_left = calculate_texture_correlation(phase_left, mask_window, A_valid, w1, truncate)
    conv_right = calculate_texture_correlation(phase_right, mask_window, A_valid, w1, truncate)
    
    # Determine search range
    if conv_left >= conv_right:
        k_range = np.arange(0, rg + 1) * step + k_initial
    else:
        k_range = np.arange(-rg, 1) * step + k_initial
    
    # Vectorized search for best slope
    correlations = np.zeros(len(k_range))
    for j, ki in enumerate(k_range):
        phase_test = phase_window - ki * dem_window
        correlations[j] = calculate_texture_correlation(phase_test, mask_window, A_valid, w1, truncate)
    
    # Find best slope
    best_idx = np.argmin(correlations)
    return k_range[best_idx]

def calculate_texture_correlation(phase_data, mask, A_valid, w1, truncate):
    """Calculate texture correlation for a given phase data"""
    
    # Apply Gaussian filter
    C = phase_data.copy()
    C[np.isnan(C)] = 0
    C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
    C = C - C_LP
    
    # Extract valid data
    C_valid = C[~np.isnan(mask)]
    if len(C_valid) == 0 or np.all(C_valid == 0):
        return 0
    
    # Normalize and calculate correlation
    C_valid = C_valid / np.linalg.norm(C_valid)
    conv_AC = np.nansum(A_valid * C_valid)
    
    return np.abs(conv_AC)

def slope_interpolation(ts_data, inps, k_htc):
    """Estimated local slope interpolation to obtain full-scale slope
    Parameters: ts_data : 3D array in size of (num_date, length, width)
                inps    : Namespace
                k_htc   : 3D array in size of (num_date, length, width), length and width depend on the window size and overlap ratio
    Returns:    k_htc_interp   : 3D array in size of (num_date, length, width)
    """

    print_progress('Starting slope interpolation...')

    # Filtering parameters for obtaining high-frequency texture correlation
    sigma_slope = 7 # standard deviation for gaussian filter
    w_slope = 7 # window size for gaussian filter
    truncate_slope = ((w_slope - 1)/2 - 0.5)/sigma_slope # truncation factor for gaussian filter
    N, Na, Nr = ts_data.shape
    W = inps.windowsize
    w = (W-1)/2
    overlap = round(inps.overlapratio*(2*w+1))
    Na_C = np.arange(w + 1, Na + 1, 2 * w - overlap)
    Nr_C = np.arange(w + 1, Nr + 1, 2 * w - overlap)
    
    # K_htc
    k_htc_interp = np.zeros((N, Na, Nr))
    
    # Create progress bar
    if TQDM_AVAILABLE:
        interp_iter = tqdm(total=N, desc="Interpolation processing", unit="time point")
    else:
        interp_iter = None
        start_time = time.time()
    
    for n in range(0, N):
        # Update progress
        if interp_iter:
            interp_iter.update(1)
        elif n % max(1, N // 10) == 0:  # Show progress every 10%
            print_progress(f'Slope interpolation progress', start_time, n, N)
        
        # filtering
        k_htc_filt = scipy.ndimage.gaussian_filter(k_htc[n, :, :], sigma=sigma_slope, truncate=truncate_slope, mode='nearest')
        
        y_coords = np.arange(0, Na)
        x_coords = np.arange(0, Nr)
        
        interpolator = RegularGridInterpolator(
            (Na_C, Nr_C),
            k_htc_filt, 
            method='cubic', 
            bounds_error=False, 
            fill_value=None
        )
        
        query_points = np.array(np.meshgrid(y_coords, x_coords, indexing='ij')).reshape(2, -1).T
        
        interpolated_values = interpolator(query_points)
        k_htc_interp[n, :, :] = interpolated_values.reshape(Na, Nr)
        
        # k_htc_interp[n, :, :] = np.where(
        #     np.isnan(k_htc_interp[n, :, :]), 
        #     np.nan, 
        #     k_htc_interp[n, :, :]
        # )
    
    if interp_iter:
        interp_iter.close()
    
    print_progress('Slope interpolation completed')
    return k_htc_interp

def intercept_filtering(dem, ts_data, inps, k_htc_interp, meta):
    """Estimate and correct tropospheric delay using intercept filtering
    Parameters: dem     : 2D array in size of (          length, width)
                ts_data : 3D array in size of (num_date, length, width)
                inps    : Namespace
                k_htc_interp : 3D array in size of (num_date, length, width)
                meta    : Metadata of timeseries
    Returns:    phase_ts_htc_low   : 3D array in size of (num_date, length, width)
    """

    print_progress('Starting intercept filtering...')

    # Filtering parameters for obtaining high-frequency texture correlation
    sigma_intercept = 251 # standard deviation for gaussian filter
    w_intercept = 251 # window size for gaussian filter
    truncate_intercept = ((w_intercept - 1)/2 - 0.5)/sigma_intercept # truncation factor for gaussian filter
    ref_y = int(meta['REF_Y'])
    ref_x = int(meta['REF_X'])
    lamda = float(meta['WAVELENGTH'])
    N = ts_data.shape[0]

    # ts_data = 4 * np.pi / lamda * ts_data[:N, :, :]
    # reference_value = ts_data[:, ref_y-1, ref_x-1]
    # ts_data = ts_data - reference_value[:, np.newaxis, np.newaxis]
    # ts_data[np.isnan(ts_data)] = 0

    phase_ts_htc_low = ts_data.copy()
    intercept = np.zeros(phase_ts_htc_low.shape)
    
    # Create progress bar
    if TQDM_AVAILABLE:
        filt_iter = tqdm(total=N, desc="Intercept filtering", unit="time point")
    else:
        filt_iter = None
        start_time = time.time()
    
    for n in range(0, N):
        # Update progress
        if filt_iter:
            filt_iter.update(1)
        elif n % max(1, N // 10) == 0:  # Show progress every 10%
            print_progress(f'Intercept filtering progress', start_time, n, N)
        
        tmp = ts_data[n, :, :] - k_htc_interp[n, :, :] * dem
        tmp_filt = scipy.ndimage.gaussian_filter(tmp, sigma=w_intercept, truncate=truncate_intercept, mode='nearest')
        tmp1 = tmp - tmp_filt
        phase_ts_htc_low[n, :, :] = tmp1
        intercept[n, :, :] = tmp_filt
    
    if filt_iter:
        filt_iter.close()

    # reference_phase = phase_ts_htc_low[:, ref_y - 1, ref_x - 1]
    # phase_ts_htc_low = phase_ts_htc_low - reference_phase[:, np.newaxis, np.newaxis] #TODO
    
    print_progress('Intercept filtering completed')
    return phase_ts_htc_low


############################################################################

def run_tropo_local_texture(inps):
    
    total_start_time = time.time()
    print_progress('Starting tropospheric local texture correction processing...')
    print_progress(f'Input file: {inps.timeseries_file}')
    print_progress(f'Geometry file: {inps.geom_file}')
    print_progress(f'Mask file: {inps.mask_file}')

    # read time-series data
    print_progress('Reading time series data...')
    ts_obj = timeseries(inps.timeseries_file)
    ts_obj.open()
    ts_data = ts_obj.read()
    inps.date_list = list(ts_obj.dateList)
    n_ref = inps.date_list.index(ts_obj.metadata['REF_DATE'])
    print_progress(f'Time series data reading completed - Number of time points: {len(inps.date_list)}, Reference date: {ts_obj.metadata["REF_DATE"]}')

    # read topographic data (DEM)
    dem = read_topographic_data(inps.geom_file)

    # slope estimation
    k_htc = estimate_local_slope(dem, ts_data, inps, n_ref, ts_obj.metadata)

    # slope interpolation
    k_htc_interp = slope_interpolation(ts_data, inps, k_htc)

    # intercept filtering
    ts_htc_low = intercept_filtering(dem, ts_data, inps, k_htc_interp, ts_obj.metadata)
    lamda = float(ts_obj.metadata['WAVELENGTH'])
    ts_htc_data = lamda / 4 /np.pi * ts_htc_low

    # write corrected time-series file
    print_progress('Writing corrected time series file...')
    meta = dict(ts_obj.metadata)
    if not inps.outfile:
        fbase = os.path.splitext(inps.timeseries_file)[0]
        inps.outfile = f'{fbase}_tropolocaltexture.h5'

    writefile.write(
        ts_htc_data,
        out_file=inps.outfile,
        metadata=meta,
        ref_file=inps.timeseries_file
    )

    total_time = time.time() - total_start_time
    print_progress(f'Tropospheric local texture correction processing completed!')
    print_progress(f'Output file: {inps.outfile}')
    print_progress(f'Total processing time: {timedelta(seconds=int(total_time))}')

    return

def estimate_local_slope(dem, ts_data, inps, n_ref, meta):
    """Wrapper function to choose between original, vectorized, and parallel versions"""
    
    # 检查用户是否指定了并行处理
    if hasattr(inps, 'num_processes') and inps.num_processes > 1:
        print_progress(f'Using parallel processing with {inps.num_processes} processes...')
        return estimate_local_slope_parallel(dem, ts_data, inps, n_ref, meta)
    
    # 如果没有指定并行处理，检查数据大小决定是否使用向量化版本
    N, Na, Nr = ts_data.shape
    W = inps.windowsize
    
    # 如果数据较大，使用向量化版本
    if Na * Nr > 1000000:  # 1M pixels threshold
        print_progress('Using vectorized processing for large dataset...')
        return estimate_local_slope_vectorized(dem, ts_data, inps, n_ref, meta)
    else:
        print_progress('Using original processing for small dataset...')
        return estimate_local_slope_original(dem, ts_data, inps, n_ref, meta)

def estimate_local_slope_original(dem, ts_data, inps, n_ref, meta):
    """Original implementation for backward compatibility"""
    # 这里是原来的代码，保持不变
    print_progress('Starting local slope estimation...')
    
    # Filtering parameters for obtaining high-frequency texture
    w1 = 9  # slope filtering parameters used in gaussian filter
    w2 = 13 # texture correlation filtering parameters used in gaussian filter
    res_step = 0.5 # step size for mask updating
    Iteration = 10 # iteration times for mask updating
    rg = 40 # range for slope searching
    step = 0.0001 # step size for slope searching
    # w2 = 2 * int(truncate * sigma + 0.5) + 1
    truncate = ((w2 - 1)/2 - 0.5)/w1 # truncation factor for gaussian filter

    lamda = float(meta['WAVELENGTH'])
    ref_y = int(meta['REF_Y'])
    ref_x = int(meta['REF_X'])

    N, Na, Nr = ts_data.shape
    W = inps.windowsize
    w = (W-1)/2 # half window size
    overlap = round(inps.overlapratio*(2*w+1))

    print_progress(f'Reading mask file: {inps.mask_file}')
    mask_coh = readfile.read(inps.mask_file, datasetName='mask')[0]
    mask_coh = np.where(mask_coh < 0.87, np.nan, 1)
    mask = mask_coh

    Na_C = np.arange(w + 1, Na + 1, 2 * w - overlap)
    Nr_C = np.arange(w + 1, Nr + 1, 2 * w - overlap)
    
    total_patches = len(Na_C) * len(Nr_C)
    print_progress(f'Starting to process {total_patches} local windows...')

    k_LLF = np.zeros((N, len(Na_C), len(Nr_C)))  # LLF: slope values in spatially discrete distributions
    d_LLF = np.zeros((N, len(Na_C), len(Nr_C)))  # LLF: intercept values in spatially discrete distributions
    k_htc = np.zeros((N, len(Na_C), len(Nr_C)))  # HTC: slope values in spatially discrete distributions

    ts_data = 4 * np.pi / lamda * ts_data[:N, :, :]
    reference_value = ts_data[:, ref_y-1, ref_x-1]
    ts_data = ts_data - reference_value[:, np.newaxis, np.newaxis]
    ts_data[np.isnan(ts_data)] = 0

    # Create progress bar
    if TQDM_AVAILABLE:
        patch_iter = tqdm(total=total_patches, desc="Processing local windows", unit="window")
    else:
        patch_iter = None
        start_time = time.time()

    patch_count = 0
    for na, nac in enumerate(Na_C):
        for nr, nrc in enumerate(Nr_C):
            patch_count += 1
            
            # Update progress
            if patch_iter:
                patch_iter.update(1)
            elif patch_count % max(1, total_patches // 20) == 0:  # Show progress every 5%
                print_progress(f'Local slope estimation progress', start_time, patch_count, total_patches)
            
            # patch
            u = nac - w
            U = np.where(u < 1, 1, u)
            d = nac + w
            D = np.where(d > Na, Na, d)
            l = nrc - w
            L = np.where(l < 1, 1, l)
            r = nrc + w
            R = np.where(r > Nr, Nr, r)

            if na == len(Na_C) - 1:
                D = Na
            if nr == len(Nr_C) - 1:
                R = Nr
            U = int(U)
            D = int(D)
            L = int(L)
            R = int(R)
            tmp = np.full((Na, Nr), np.nan)
            tmp[U-1:D, L-1:R] = 1
            mask_process = mask.copy()
            mask_process[np.isnan(tmp)] = np.nan

            # solve
            result_compare = np.zeros((N, 4))  # line0-3: k_LLF, k_ILLF(iteration), k_htc, d_LLF
            for n in range(N):
                if n == n_ref:
                    continue
                # ----------- initial value ----------- #
                # mask
                mask_std = np.ones((Na, Nr)) * mask_process
                
                # Initialize coe with default values
                coe = [0.0, 0.0]  # [slope, intercept]
                
                # iterative linear fitting
                # res+step = 0.5 # step [rad]
                # Iteration = 10
                for i in range(Iteration):
                    # linear fitting
                    phase_tmp = ts_data[n, :, :] * mask_std
                    dem_tmp = dem * mask_std
                    valid_idx = ~np.isnan(phase_tmp) & ~np.isnan(dem_tmp)
                    if np.nansum(valid_idx) == 0:
                        continue
                    coe = np.polyfit(dem[valid_idx], phase_tmp[valid_idx], 1)
                    cor_tmp = phase_tmp - coe[0]*dem_tmp - coe[1]

                    # result recording
                    if i == 0:
                        result_compare[n, 0] = coe[0]
                        result_compare[n, 3] = coe[1]

                    # mask uploading
                    max_tmp = np.nanmax(np.abs(cor_tmp)) #TODO
                    max_tmp = np.where(max_tmp > res_step, max_tmp - res_step, max_tmp)
                    mask_std[np.abs(cor_tmp) > max_tmp] = np.nan

                    if np.nansum(~np.isnan(mask_std)) < np.nansum(~np.isnan(mask_process)) / 10: #TODO
                        break

                # ----------- texture correlation ----------- #
                mask_tmp = mask[U-1:D, L-1:R]

                A = dem[U-1:D, L-1:R]
                A_LP = scipy.ndimage.gaussian_filter(A, sigma=w1, truncate=truncate, mode='nearest')
                A = A - A_LP
                A_line = A[~np.isnan(mask_tmp)]
                A_line = A_line / np.linalg.norm(A_line)

                # range = 40
                # step = 0.0001
                # left
                k_left = -rg * step + coe[0]
                phase_ts_Scor_tmp = ts_data[n, :, :] - k_left * dem
                C = phase_ts_Scor_tmp[U-1:D, L-1:R]
                C[np.isnan(C)] = 0
                C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
                C = C - C_LP
                C_line = C[~np.isnan(mask_tmp)]
                C_line = C_line / np.linalg.norm(C_line)
                conv_AC = np.nansum(A_line * C_line)
                record_left = np.abs(conv_AC)
                # right
                k_right = rg * step + coe[0]
                phase_ts_Scor_tmp = ts_data[n, :, :] - k_right * dem
                C = phase_ts_Scor_tmp[U-1:D, L-1:R]
                C[np.isnan(C)] = 0
                C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
                C = C - C_LP
                C_line = C[~np.isnan(mask_tmp)]
                C_line = C_line / np.linalg.norm(C_line)
                conv_AC = np.nansum(A_line * C_line)
                record_right = np.abs(conv_AC)

                if record_left >= record_right:
                    k = np.arange(0, rg + 1) * step + coe[0]
                    record = np.zeros(len(k))
                else:
                    k = np.arange(-rg, 1) * step + coe[0]
                    record = np.zeros(len(k))

                for i, ki in enumerate(k):
                    phase_ts_Scor_tmp = ts_data[n, :, :] - ki * dem
                    C = phase_ts_Scor_tmp[U-1:D, L-1:R]
                    C[np.isnan(C)] = 0
                    C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
                    C = C - C_LP
                    C_line = C[~np.isnan(mask_tmp)]
                    C_line = C_line / np.linalg.norm(C_line)

                    conv_AC = np.nansum(A_line * C_line)
                    record[i] = np.abs(conv_AC)

                # slope
                index = np.argmin(record)
                result_compare[n, 1] = coe[0]
                result_compare[n, 2] = k[index]

            # result recording
            k_LLF[:, na, nr] = result_compare[:, 0]
            d_LLF[:, na, nr] = result_compare[:, 3]
            k_htc[:, na, nr] = result_compare[:, 2]
    
    if patch_iter:
        patch_iter.close()
    
    print_progress('Local slope estimation completed')
    return k_htc
