#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################

import os
import time
import argparse

import h5py
import numpy as np
from cvxopt import matrix
from matplotlib import pyplot as plt

from mintpy import ifgram_inversion as ifginv
from mintpy.objects import ifgramStack
from mintpy.utils import ptime, readfile, utils as ut, writefile
from mintpy.utils.solvers import l1regls

key_prefix = 'mintpy.unwrapError.'


##########################################################################################
def solve_integer_ambiguity_robust(C, closure_int):
    """Robust solver for integer ambiguity with multiple fallback methods."""
    
    # 方法1：尝试L1正则化（原版方法）
    try:
        closure_int_matrix = matrix(closure_int.astype(float))
        C_matrix = matrix(C.astype(float))
        
        U_pixel = np.round(l1regls(
            A=-C_matrix,
            y=closure_int_matrix,
            alpha=1e-2,
            show_progress=0,
        )).flatten()
        
        if not np.any(np.isnan(U_pixel)) and not np.any(np.isinf(U_pixel)):
            return U_pixel
    
    except Exception as e:
        pass
    
    # 方法2：尝试更强的正则化
    try:
        U_pixel = np.round(l1regls(
            A=-C_matrix,
            y=closure_int_matrix,
            alpha=1e-1,  # 更强的正则化
            show_progress=0,
        )).flatten()
        
        if not np.any(np.isnan(U_pixel)) and not np.any(np.isinf(U_pixel)):
            return U_pixel
    
    except Exception as e:
        pass
    
    # 方法3：最小二乘法（最稳定）
    try:
        # 添加正则化项
        C_reg = C + 1e-6 * np.eye(C.shape[0])
        U_pixel = np.linalg.lstsq(-C_reg, closure_int, rcond=1e-6)[0]
        U_pixel = np.round(U_pixel)
        
        if not np.any(np.isnan(U_pixel)) and not np.any(np.isinf(U_pixel)):
            return U_pixel
    
    except Exception as e:
        pass
    
    # 如果所有方法都失败，返回0
    raise ValueError("All solving methods failed")


##########################################################################################
def get_pixel_wise_int_ambiguity_aoi(stack_obj, filtered_date12_list, aoi_mask, 
                                    dsNameIn, block_size):
    """Calculate pixel-wise integer ambiguity for AOI region using block processing."""
    print('Processing AOI with block-wise pixel approach...')
    
    length, width = stack_obj.length, stack_obj.width
    num_ifgram = len(filtered_date12_list)
    
    # 构建设计矩阵C
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    
    # 获取参考相位
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,
    ).reshape(num_ifgram, -1)
    
    # 初始化结果数组
    int_ambiguity = np.zeros((num_ifgram, length, width), dtype=np.float32)
    
    # 获取AOI像素坐标
    aoi_coords = np.where(aoi_mask)
    num_aoi_pixels = len(aoi_coords[0])
    
    print(f'Processing {num_aoi_pixels} AOI pixels...')
    
    # 使用分块处理，但只处理AOI内的像素
    block_size_aoi = min(block_size, 100)  # 限制块大小确保稳定性
    num_blocks = int(np.ceil(num_aoi_pixels / block_size_aoi))
    
    # 统计变量
    total_pixels_processed = 0
    total_pixels_failed = 0
    
    prog_bar = ptime.progressBar(maxValue=num_blocks)
    for i in range(num_blocks):
        start_idx = i * block_size_aoi
        end_idx = min(start_idx + block_size_aoi, num_aoi_pixels)
        
        # 处理当前块的像素
        for j in range(start_idx, end_idx):
            r, c = aoi_coords[0][j], aoi_coords[1][j]
            total_pixels_processed += 1
            
            try:
                # 读取像素数据
                unw_pixel = ifginv.read_stack_obs(
                    stack_obj,
                    box=(c, r, c+1, r+1),
                    ref_phase=ref_phase,
                    obs_ds_name=dsNameIn,
                    dropIfgram=True,
                    print_msg=False,
                ).reshape(num_ifgram, -1)
                
                # 计算闭合相位
                closure_pha = np.dot(C, unw_pixel)
                closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
                
                # 检查数值稳定性
                if np.any(np.isnan(closure_int)) or np.any(np.isinf(closure_int)):
                    int_ambiguity[:, r, c] = 0
                    total_pixels_failed += 1
                    continue
                
                # 求解整数模糊度
                try:
                    U_pixel = solve_integer_ambiguity_robust(C, closure_int)
                    
                    if np.any(np.isnan(U_pixel)) or np.any(np.isinf(U_pixel)):
                        int_ambiguity[:, r, c] = 0
                        total_pixels_failed += 1
                    else:
                        int_ambiguity[:, r, c] = U_pixel
                        
                except Exception as e:
                    int_ambiguity[:, r, c] = 0
                    total_pixels_failed += 1
                    
            except Exception as e:
                int_ambiguity[:, r, c] = 0
                total_pixels_failed += 1
        
        prog_bar.update(i+1)
    
    prog_bar.close()
    
    # 打印统计信息
    print('-'*50)
    print('AOI Processing statistics:')
    print(f'Total AOI pixels processed: {total_pixels_processed}')
    print(f'Total AOI pixels failed: {total_pixels_failed}')
    print(f'AOI success rate: {(total_pixels_processed - total_pixels_failed) / total_pixels_processed * 100:.2f}%')
    
    return int_ambiguity


##########################################################################################
def get_region_based_int_ambiguity_other(stack_obj, filtered_date12_list, other_mask, 
                                        dsNameIn, num_sample=100, min_area=2.5e3):
    """Calculate region-based integer ambiguity for non-AOI regions."""
    print('Processing other regions with region-based approach...')
    
    length, width = stack_obj.length, stack_obj.width
    num_ifgram = len(filtered_date12_list)
    
    # 构建设计矩阵C
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    
    # 添加C矩阵诊断信息
    print('-'*50)
    print('Other regions Design matrix C properties:')
    print(f'  Shape: {C.shape}')
    print(f'  Data type: {C.dtype}')
    print(f'  Memory usage: {C.nbytes / 1024**2:.2f} MB')
    
    # 检查C矩阵的稀疏性
    non_zero_elements = np.count_nonzero(C)
    total_elements = C.size
    sparsity = 1 - (non_zero_elements / total_elements)
    print(f'  Sparsity: {sparsity:.2%}')
    
    # 检查C矩阵的条件数
    try:
        cond_num = np.linalg.cond(C)
        print(f'  Condition number: {cond_num:.2e}')
        if cond_num > 1e12:
            print('  WARNING: C matrix is ill-conditioned!')
    except Exception as e:
        print(f'  Could not compute condition number: {e}')
    
    # 检查C矩阵的数值范围
    print(f'  C matrix min/max: {C.min():.6f} / {C.max():.6f}')
    print(f'  C matrix has NaN: {np.any(np.isnan(C))}')
    print(f'  C matrix has Inf: {np.any(np.isinf(C))}')
    
    # 获取参考相位
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,
    ).reshape(num_ifgram, -1)
    
    # 初始化结果数组
    int_ambiguity = np.zeros((num_ifgram, length, width), dtype=np.float32)
    
    # 使用连接组件分析找到区域
    from mintpy.objects import conncomp
    from skimage import measure
    
    # 标记连接组件
    label_img, num_label = conncomp.label_conn_comp(other_mask, min_area=min_area, print_msg=True)
    
    if num_label == 0:
        print('No regions found in other areas, skipping region-based processing')
        return int_ambiguity
    
    print(f'Found {num_label} regions for region-based processing')
    
    # 获取所有区域属性
    regions = measure.regionprops(label_img)
    
    # 统计变量
    total_regions_processed = 0
    total_regions_failed = 0
    failed_regions_by_time = np.zeros(num_ifgram, dtype=int)
    
    # 处理每个区域
    for i, region in enumerate(regions):
        # 确保area是整数
        area = int(region.area)
        
        if area < min_area:
            continue
        
        print(f'Processing region {i+1}/{num_label}: area={area}')
        total_regions_processed += 1
        
        # 随机采样像素
        rng = np.random.default_rng()
        if area > num_sample:
            sample_indices = rng.choice(area, num_sample, replace=False)
        else:
            sample_indices = np.arange(area)
        
        sample_coords = region.coords[sample_indices, :]
        
        # 计算该区域的整数模糊度
        U_region = np.zeros((num_ifgram, len(sample_coords)))
        
        # 统计该区域的求解成功情况
        region_pixels_succeeded = 0
        region_pixels_failed = 0
        
        for j, (r, c) in enumerate(sample_coords):
            try:
                # 读取像素数据
                unw_pixel = ifginv.read_stack_obs(
                    stack_obj,
                    box=(c, r, c+1, r+1),
                    ref_phase=ref_phase,
                    obs_ds_name=dsNameIn,
                    dropIfgram=True,
                    print_msg=False,
                ).reshape(num_ifgram, -1)
                
                # 计算闭合相位
                closure_pha = np.dot(C, unw_pixel)
                closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
                
                # 检查数值稳定性
                if np.any(np.isnan(closure_int)) or np.any(np.isinf(closure_int)):
                    print(f'  Warning: Invalid values in closure_int for region pixel ({r}, {c})')
                    U_region[:, j] = 0
                    region_pixels_failed += 1
                    continue
                
                # 检查C矩阵的条件（只对第一个像素检查，避免重复）
                if j == 0:
                    try:
                        if np.linalg.cond(C) > 1e15:
                            print(f'  Warning: C matrix is ill-conditioned for region {i+1}')
                            # 不立即失败，继续尝试其他像素
                    except:
                        pass
                
                # 求解整数模糊度
                U_pixel = solve_integer_ambiguity_robust(C, closure_int)
                
                # 检查求解结果 - 修复：不仅要检查异常，还要检查结果质量
                if (np.any(np.isnan(U_pixel)) or 
                    np.any(np.isinf(U_pixel)) or 
                    np.all(U_pixel == 0)):  # 新增：检查是否全为零
                    
                    if np.all(U_pixel == 0):
                        print(f'  Warning: All-zero result for region pixel ({r}, {c}) - likely C matrix issue')
                    else:
                        print(f'  Warning: Invalid values in U_pixel for region pixel ({r}, {c})')
                    
                    U_region[:, j] = 0
                    region_pixels_failed += 1
                else:
                    U_region[:, j] = U_pixel
                    region_pixels_succeeded += 1
                    
                    # 只对第一个成功像素打印详细信息
                    if region_pixels_succeeded == 1:
                        print(f'  First successful pixel ({r}, {c}) U_pixel properties:')
                        print(f'    Shape: {U_pixel.shape}')
                        print(f'    Range: {U_pixel.min():.6f} / {U_pixel.max():.6f}')
                        print(f'    Non-zero count: {np.count_nonzero(U_pixel)}')
                        print(f'    Values: {U_pixel.flatten()[:5]}...')  # 显示前5个值
                
            except Exception as e:
                print(f'  Warning: failed to solve for region pixel ({r}, {c}): {e}')
                U_region[:, j] = 0
                region_pixels_failed += 1
        
        # 统计该区域的失败情况 - 修复：考虑结果质量
        if region_pixels_succeeded == 0:
            total_regions_failed += 1
            failed_regions_by_time += 1
            print(f'  Region {i+1} failed: all {len(sample_coords)} pixels failed')
        else:
            print(f'  Region {i+1} succeeded: {region_pixels_succeeded}/{len(sample_coords)} pixels succeeded')
            
            # 检查中位数结果的质量
            U_median = np.median(U_region, axis=1)
            non_zero_median = np.count_nonzero(U_median)
            print(f'  Median result quality: {non_zero_median}/{num_ifgram} time steps have non-zero values')
            
            # 将结果应用到整个区域
            region_mask = label_img == (i+1)
            for t in range(num_ifgram):
                int_ambiguity[t, region_mask] = U_median[t]
    
    # 打印区域处理统计信息
    print('-'*50)
    print('Region-based processing statistics:')
    print(f'Total regions processed: {total_regions_processed}')
    print(f'Total regions failed: {total_regions_failed}')
    print(f'Region success rate: {(total_regions_processed - total_regions_failed) / total_regions_processed * 100:.2f}%')
    
    print('-'*50)
    print('Region-based integer ambiguity statistics by time:')
    for i in range(num_ifgram):
        non_zero_count = np.count_nonzero(int_ambiguity[i, other_mask])
        total_other_pixels = np.sum(other_mask)
        failed_count = failed_regions_by_time[i]
        print(f'Time {i}: {non_zero_count}/{total_other_pixels} other pixels have non-zero values, {failed_count} regions failed')
    
    return int_ambiguity


##########################################################################################
def get_hybrid_int_ambiguity(ifgram_file, mask_file=None, aoi_coords=None, 
                            dsNameIn='unwrapPhase', max_memory=4, block_size=1000,
                            num_sample=100, min_area=2.5e3, aoi_only=False):
    """Calculate integer ambiguity using hybrid approach.
    
    Parameters:
        aoi_coords: str, format 'y0:y1,x0:x1' or tuple (y0, y1, x0, x1)
        aoi_only: bool, if True, only process AOI region and skip other regions
    """
    print('-'*50)
    if aoi_only:
        print('calculating integer ambiguity: pixel-wise in AOI ONLY (skipping other regions)...')
    else:
        print('calculating hybrid integer ambiguity: pixel-wise in AOI, region-based elsewhere...')
    
    # stack info
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    length, width = stack_obj.length, stack_obj.width
    
    # 获取干涉图列表
    filtered_date12_list = stack_obj.get_date12_list(dropIfgram=True)
    all_date12_list = stack_obj.get_date12_list(dropIfgram=False)
    num_ifgram = len(filtered_date12_list)
    
    print(f'Total interferograms: {len(all_date12_list)}')
    print(f'Filtered interferograms for estimation: {num_ifgram}')
    
    # 解析AOI坐标
    aoi_mask = None
    if aoi_coords:
        if isinstance(aoi_coords, str):
            # 解析 'y0:y1,x0:x1' 格式
            try:
                y_part, x_part = aoi_coords.split(',')
                y0, y1 = map(int, y_part.split(':'))
                x0, x1 = map(int, x_part.split(':'))
                
                # 确保坐标在有效范围内
                y0 = max(0, min(y0, length-1))
                y1 = max(0, min(y1, length-1))
                x0 = max(0, min(x0, width-1))
                x1 = max(0, min(x1, width-1))
                
                # 确保 y0 < y1, x0 < x1
                if y0 > y1:
                    y0, y1 = y1, y0
                if x0 > x1:
                    x0, x1 = x1, x0
                
                print(f'AOI coordinates: y0:y1={y0}:{y1}, x0:x1={x0}:{x1}')
                print(f'AOI size: {y1-y0+1} x {x1-x0+1} pixels')
                
                # 创建AOI掩膜
                aoi_mask = np.zeros((length, width), dtype=bool)
                aoi_mask[y0:y1+1, x0:x1+1] = True
                print(f'AOI mask created: {np.sum(aoi_mask)} pixels')
                
            except ValueError as e:
                print(f'Warning: Invalid AOI format "{aoi_coords}". Expected format: "y0:y1,x0:x1"')
                print(f'Error: {e}')
                aoi_coords = None
                aoi_mask = None
        
        elif isinstance(aoi_coords, (tuple, list)) and len(aoi_coords) == 4:
            # 直接使用元组格式 (y0, y1, x0, x1)
            y0, y1, x0, x1 = aoi_coords
            
            # 确保坐标在有效范围内
            y0 = max(0, min(y0, length-1))
            y1 = max(0, min(y1, length-1))
            x0 = max(0, min(x0, width-1))
            x1 = max(0, min(x1, width-1))
            
            # 确保 y0 < y1, x0 < x1
            if y0 > y1:
                y0, y1 = y1, y0
            if x0 > x1:
                x0, x1 = x1, x0
            
            print(f'AOI coordinates: y0:y1={y0}:{y1}, x0:x1={x0}:{x1}')
            print(f'AOI size: {y1-y0+1} x {x1-x0+1} pixels')
            
            # 创建AOI掩膜
            aoi_mask = np.zeros((length, width), dtype=bool)
            aoi_mask[y0:y1+1, x0:x1+1] = True
            print(f'AOI mask created: {np.sum(aoi_mask)} pixels')
        
        else:
            print(f'Warning: Invalid AOI format. Expected string "y0:y1,x0:x1" or tuple (y0,y1,x0,x1)')
            aoi_coords = None
            aoi_mask = None
    
    # 读取主掩膜
    if mask_file and os.path.isfile(mask_file):
        mask = readfile.read(mask_file)[0]
        mask = mask.astype(bool)
        print(f'Main mask loaded: {np.sum(mask)} pixels')
    else:
        mask = np.ones((length, width), dtype=bool)
        print(f'No mask file, using all pixels: {np.sum(mask)} pixels')
    
    # 验证AOI掩膜
    if aoi_mask is not None:
        # 检查AOI掩膜与主掩膜的重叠
        aoi_valid = aoi_mask & mask
        print(f'AOI valid pixels (overlap with main mask): {np.sum(aoi_valid)}')
        
        if np.sum(aoi_valid) == 0:
            print('WARNING: No valid pixels in AOI after applying main mask!')
            print('This might cause AOI processing to fail.')
    
    # 初始化结果数组
    int_ambiguity = np.zeros((num_ifgram, length, width), dtype=np.float32)
    
    # 方法1：AOI内逐像素处理
    if aoi_mask is not None and np.any(aoi_mask & mask):
        print('Processing AOI pixels with pixel-wise approach...')
        print(f'AOI pixels to process: {np.sum(aoi_mask & mask)}')
        
        # 直接调用分块逐像素方法，但只处理AOI区域
        int_ambiguity_aoi = get_pixel_wise_int_ambiguity_aoi(
            stack_obj, filtered_date12_list, aoi_mask & mask, dsNameIn, block_size
        )
        
        # 将AOI结果复制到主数组
        int_ambiguity[:, aoi_mask & mask] = int_ambiguity_aoi[:, aoi_mask & mask]
        
        print(f'AOI processing completed. Non-zero pixels: {np.count_nonzero(int_ambiguity)}')
    
    # 方法2：其他区域区域采样（仅在非aoi_only模式下）
    if not aoi_only:
        other_mask = mask & (aoi_mask == False) if aoi_mask is not None else mask
        if np.any(other_mask):
            print('Processing other regions with region-based approach...')
            print(f'Other pixels to process: {np.sum(other_mask)}')
            
            int_ambiguity_other = get_region_based_int_ambiguity_other(
                stack_obj, filtered_date12_list, other_mask, dsNameIn, num_sample, min_area
            )
            
            # 将其他区域结果复制到主数组
            int_ambiguity[:, other_mask] = int_ambiguity_other[:, other_mask]
            
            print(f'Other regions processing completed. Non-zero pixels: {np.count_nonzero(int_ambiguity)}')
    else:
        print('Skipping other regions processing (aoi_only mode)')
    
    # 最终统计
    total_valid = np.sum(mask)
    total_processed = np.count_nonzero(int_ambiguity)
    print('-'*50)
    print('Processing summary:')
    print(f'Total valid pixels: {total_valid}')
    print(f'Total processed pixels: {total_processed}')
    print(f'Processing coverage: {total_processed/total_valid*100:.2f}%')
    
    if aoi_mask is not None:
        aoi_processed = np.count_nonzero(int_ambiguity[:, aoi_mask & mask])
        aoi_total = np.sum(aoi_mask & mask)
        print(f'AOI processing: {aoi_processed}/{aoi_total} pixels ({aoi_processed/aoi_total*100:.2f}%)')
        
        if not aoi_only:
            other_processed = total_processed - aoi_processed
            other_total = total_valid - aoi_total
            if other_total > 0:
                print(f'Other regions processing: {other_processed}/{other_total} pixels ({other_processed/other_total*100:.2f}%)')
        else:
            print('Other regions: skipped (aoi_only mode)')
    
    return int_ambiguity, filtered_date12_list, all_date12_list


##########################################################################################
def get_pixel_int_ambiguity_robust(ifgram_file, mask_file=None, dsNameIn='unwrapPhase', 
                                   max_memory=4, block_size=1000, median_filter_size=3):
    """Robust pixel-wise integer ambiguity calculation with pixel-by-pixel reading and median filtering.
    
    Parameters:
        ifgram_file: str, path to interferogram stack file
        mask_file: str, path to mask file
        dsNameIn: str, input dataset name
        max_memory: float, maximum memory usage in GB
        block_size: int, block size for processing
        median_filter_size: int, size of median filter kernel (must be odd)
    """
    print('-'*50)
    print('calculating robust pixel-wise integer ambiguity using phase closure...')
    
    # stack info
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    length, width = stack_obj.length, stack_obj.width
    
    # 获取所有干涉图列表（用于最终输出）
    all_date12_list = stack_obj.get_date12_list(dropIfgram=False)
    total_ifgram = len(all_date12_list)
    
    # 获取过滤后的干涉图列表（用于估计）
    filtered_date12_list = stack_obj.get_date12_list(dropIfgram=True)
    num_ifgram = len(filtered_date12_list)
    
    print(f'Total interferograms: {total_ifgram}')
    print(f'Filtered interferograms for estimation: {num_ifgram}')
    print(f'Note: Using filtered subset for model estimation')
    
    # 构建设计矩阵C（基于高质量干涉图）
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,  # 估计阶段用过滤后的
    ).reshape(num_ifgram, -1)
    
    print(f'number of interferograms: {num_ifgram}')
    print(f'number of triplets: {C.shape[0]}')
    print(f'image size: {length} x {width}')
    
    # 添加C矩阵的详细调试信息
    print('-'*50)
    print('Design matrix C properties:')
    print(f'  Shape: {C.shape}')
    print(f'  Data type: {C.dtype}')
    print(f'  Memory usage: {C.nbytes / 1024**2:.2f} MB')
    
    # 检查C矩阵的稀疏性
    non_zero_elements = np.count_nonzero(C)
    total_elements = C.size
    sparsity = 1 - (non_zero_elements / total_elements)
    print(f'  Sparsity: {sparsity:.2%}')
    
    # 检查C矩阵的条件数
    try:
        cond_num = np.linalg.cond(C)
        print(f'  Condition number: {cond_num:.2e}')
        if cond_num > 1e12:
            print('  WARNING: C matrix is ill-conditioned!')
    except Exception as e:
        print(f'  Could not compute condition number: {e}')
    
    # 检查C矩阵的数值范围
    print(f'  C matrix min/max: {C.min():.6f} / {C.max():.6f}')
    print(f'  C matrix has NaN: {np.any(np.isnan(C))}')
    print(f'  C matrix has Inf: {np.any(np.isinf(C))}')
    
    # read mask
    if mask_file and os.path.isfile(mask_file):
        mask = readfile.read(mask_file)[0]
        mask = mask.astype(bool)
    else:
        mask = np.ones((length, width), dtype=bool)
    
    # initialize output array
    int_ambiguity = np.zeros((num_ifgram, length, width), dtype=np.float32)
    
    # 确保中位数滤波核大小为奇数
    if median_filter_size % 2 == 0:
        median_filter_size += 1
        print(f'Adjusted median filter size to odd number: {median_filter_size}')
    
    # 计算处理块大小（基于内存限制）
    # 使用更小的块大小以确保逐像素读取的稳定性
    optimal_block_size = min(block_size, 100)  # 限制最大块大小
    num_loop = int(np.ceil(length / optimal_block_size))
    print(f'processing in {num_loop} blocks with size up to {optimal_block_size} x {width}')
    print(f'using median filter with kernel size {median_filter_size}')
    
    # 添加统计变量
    total_pixels_processed = 0
    total_pixels_failed = 0
    failed_pixels_by_time = np.zeros(num_ifgram, dtype=int)
    
    # 存储所有像素的结果用于后续滤波
    all_pixel_results = np.zeros((num_ifgram, length, width), dtype=np.float32)
    pixel_success_mask = np.zeros((length, width), dtype=bool)
    
    # process block by block
    prog_bar = ptime.progressBar(maxValue=num_loop)
    for i in range(num_loop):
        # box
        r0 = i * optimal_block_size
        r1 = min((r0 + optimal_block_size), length)
        
        print(f'processing block {i+1}/{num_loop}: rows {r0} to {r1-1}')
        
        # process each pixel in the block individually
        for j in range(r1 - r0):
            row = r0 + j
            for k in range(width):
                if mask[row, k]:
                    total_pixels_processed += 1
                    
                    # 逐像素读取数据（更稳定）
                    try:
                        unw_pixel = ifginv.read_stack_obs(
                            stack_obj,
                            box=(k, row, k+1, row+1),
                            ref_phase=ref_phase,
                            obs_ds_name=dsNameIn,
                            dropIfgram=True,
                            print_msg=False,
                        ).reshape(num_ifgram, -1)
                        
                        # 只对第一个像素打印详细信息
                        if total_pixels_processed == 1:
                            print(f'First pixel unw_pixel shape: {unw_pixel.shape}')
                            print(f'First pixel unw_pixel range: {unw_pixel.min():.6f} / {unw_pixel.max():.6f}')
                            print(f'First pixel unw_pixel has NaN: {np.any(np.isnan(unw_pixel))}')
                            print(f'First pixel unw_pixel has Inf: {np.any(np.isinf(unw_pixel))}')
                        
                        # calculate closure phase for this pixel
                        closure_pha = np.dot(C, unw_pixel)
                        
                        # 只对第一个像素打印详细信息
                        if total_pixels_processed == 1:
                            print(f'Closure phase shape: {closure_pha.shape}')
                            print(f'Closure phase range: {closure_pha.min():.6f} / {closure_pha.max():.6f}')
                            print(f'Closure phase has NaN: {np.any(np.isnan(closure_pha))}')
                            print(f'Closure phase has Inf: {np.any(np.isinf(closure_pha))}')
                        
                        # calculate closure integer part
                        closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
                        
                        # 只对第一个像素打印详细信息
                        if total_pixels_processed == 1:
                            print(f'Closure integer shape: {closure_int.shape}')
                            print(f'Closure integer range: {closure_int.min():.6f} / {closure_int.max():.6f}')
                            print(f'Closure integer has NaN: {np.any(np.isnan(closure_int))}')
                            print(f'Closure integer has Inf: {np.any(np.isinf(closure_int))}')
                        
                        # 检查数值稳定性
                        if np.any(np.isnan(closure_int)) or np.any(np.isinf(closure_int)):
                            print(f'Warning: Invalid values in closure_int for pixel ({row}, {k})')
                            all_pixel_results[:, row, k] = 0
                            total_pixels_failed += 1
                            failed_pixels_by_time += 1
                            continue
                        
                        # 检查C矩阵的条件（只检查一次，避免重复计算）
                        if total_pixels_processed == 1:
                            try:
                                if np.linalg.cond(C) > 1e15:
                                    print(f'Warning: C matrix is ill-conditioned')
                            except:
                                pass
                        
                        # solve for integer ambiguity using robust solver
                        try:
                            U_pixel = solve_integer_ambiguity_robust(C, closure_int)
                            
                            # 只对第一个像素打印详细信息
                            if total_pixels_processed == 1:
                                print(f'U_pixel shape: {U_pixel.shape}')
                                print(f'U_pixel range: {U_pixel.min():.6f} / {U_pixel.max():.6f}')
                                print(f'U_pixel non-zero count: {np.count_nonzero(U_pixel)}')
                                print(f'U_pixel has NaN: {np.any(np.isnan(U_pixel))}')
                                print(f'U_pixel has Inf: {np.any(np.isinf(U_pixel))}')
                            
                            # 检查求解结果
                            if np.any(np.isnan(U_pixel)) or np.any(np.isinf(U_pixel)):
                                print(f'Warning: Invalid values in U_pixel for pixel ({row}, {k})')
                                all_pixel_results[:, row, k] = 0
                                total_pixels_failed += 1
                                failed_pixels_by_time += 1
                            else:
                                # store result
                                all_pixel_results[:, row, k] = U_pixel
                                pixel_success_mask[row, k] = True
                            
                        except Exception as e:
                            print(f'Warning: failed to solve for pixel ({row}, {k}): {e}')
                            all_pixel_results[:, row, k] = 0
                            total_pixels_failed += 1
                            failed_pixels_by_time += 1
                            
                            # 添加更详细的错误信息
                            if total_pixels_processed == 1:
                                print(f'Error details for first pixel:')
                                print(f'  C matrix shape: {C.shape}')
                                print(f'  closure_int shape: {closure_int.shape}')
                                print(f'  closure_int range: {closure_int.min():.6f} / {closure_int.max():.6f}')
                                print(f'  closure_int has NaN: {np.any(np.isnan(closure_int))}')
                                print(f'  closure_int has Inf: {np.any(np.isinf(closure_int))}')
                    
                    except Exception as e:
                        print(f'Warning: failed to read data for pixel ({row}, {k}): {e}')
                        all_pixel_results[:, row, k] = 0
                        total_pixels_failed += 1
                        failed_pixels_by_time += 1
        
        prog_bar.update(i+1, every=1, suffix=f'block {i+1}/{num_loop}')
    
    prog_bar.close()
    
    # 应用中位数滤波来改善结果质量
    print('-'*50)
    print('Applying median filtering to improve result quality...')
    
    # 对每个时间步进行中位数滤波
    from scipy import ndimage
    
    for t in range(num_ifgram):
        # 只对成功的像素进行滤波
        valid_data = all_pixel_results[t, :, :].copy()
        valid_mask = pixel_success_mask & (mask)
        
        if np.any(valid_mask):
            # 应用中位数滤波
            filtered_data = ndimage.median_filter(valid_data, size=median_filter_size)
            
            # 只更新有效像素
            int_ambiguity[t, valid_mask] = filtered_data[valid_mask]
            
            # 对失败的像素使用插值填充
            failed_pixels = mask & ~pixel_success_mask
            if np.any(failed_pixels):
                # 使用最近邻插值填充失败的像素
                from scipy.interpolate import griddata
                
                # 获取有效像素的坐标和值
                valid_coords = np.where(valid_mask)
                valid_values = int_ambiguity[t, valid_mask]
                
                if len(valid_coords[0]) > 0:
                    # 获取失败像素的坐标
                    failed_coords = np.where(failed_pixels)
                    
                    if len(failed_coords[0]) > 0:
                        # 准备插值坐标
                        valid_points = np.column_stack(valid_coords)
                        failed_points = np.column_stack(failed_coords)
                        
                        try:
                            # 使用最近邻插值
                            interpolated_values = griddata(
                                valid_points, valid_values, failed_points, 
                                method='nearest', fill_value=0
                            )
                            
                            # 应用插值结果
                            int_ambiguity[t, failed_coords] = interpolated_values
                            
                            print(f'Time {t}: Interpolated {len(failed_coords[0])} failed pixels')
                            
                        except Exception as e:
                            print(f'Warning: Interpolation failed for time {t}: {e}')
                            # 如果插值失败，使用0填充
                            int_ambiguity[t, failed_pixels] = 0
    
    # 打印统计信息
    print('-'*50)
    print('Processing statistics:')
    print(f'Total pixels processed: {total_pixels_processed}')
    print(f'Total pixels failed: {total_pixels_failed}')
    print(f'Success rate: {(total_pixels_processed - total_pixels_failed) / total_pixels_processed * 100:.2f}%')
    
    print('-'*50)
    print('Integer ambiguity statistics by time:')
    for i in range(num_ifgram):
        non_zero_count = np.count_nonzero(int_ambiguity[i, :, :])
        total_pixels = np.sum(mask)
        failed_count = failed_pixels_by_time[i]
        print(f'Time {i}: {non_zero_count}/{total_pixels} pixels have non-zero values, {failed_count} failed')
    
    # mask out invalid pixels
    int_ambiguity[:, mask == 0] = np.nan
    
    print('-'*50)
    print('Robust pixel-wise processing completed with median filtering and interpolation!')
    
    # 返回时包含映射信息
    return int_ambiguity, filtered_date12_list, all_date12_list


##########################################################################################
def get_pixel_int_ambiguity_with_spatial_consistency(ifgram_file, mask_file=None, dsNameIn='unwrapPhase', 
                                                    max_memory=4, block_size=1000, 
                                                    spatial_window=5, consistency_threshold=0.8):
    """Enhanced pixel-wise method with spatial consistency checking and adaptive filtering.
    
    Parameters:
        ifgram_file: str, path to interferogram stack file
        mask_file: str, path to mask file
        dsNameIn: str, input dataset name
        max_memory: float, maximum memory usage in GB
        block_size: int, block size for processing
        spatial_window: int, size of spatial window for consistency checking
        consistency_threshold: float, threshold for spatial consistency (0-1)
    """
    print('-'*50)
    print('calculating pixel-wise integer ambiguity with spatial consistency checking...')
    
    # stack info
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    length, width = stack_obj.length, stack_obj.width
    
    # 获取过滤后的干涉图列表（用于估计）
    filtered_date12_list = stack_obj.get_date12_list(dropIfgram=True)
    num_ifgram = len(filtered_date12_list)
    
    # 获取所有干涉图列表（用于最终输出）
    all_date12_list = stack_obj.get_date12_list(dropIfgram=False)
    
    # 构建设计矩阵C
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    
    print(f'number of interferograms: {num_ifgram}')
    print(f'number of triplets: {C.shape[0]}')
    print(f'image size: {length} x {width}')
    print(f'spatial window size: {spatial_window}')
    print(f'consistency threshold: {consistency_threshold}')
    
    # 构建设计矩阵C
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,
    ).reshape(num_ifgram, -1)
    
    # read mask
    if mask_file and os.path.isfile(mask_file):
        mask = readfile.read(mask_file)[0]
        mask = mask.astype(bool)
    else:
        mask = np.ones((length, width), dtype=bool)
    
    # initialize output array
    int_ambiguity = np.zeros((num_ifgram, length, width), dtype=np.float32)
    
    # 使用更小的块大小确保稳定性
    optimal_block_size = min(block_size, 50)
    num_loop = int(np.ceil(length / optimal_block_size))
    
    # 统计变量
    total_pixels_processed = 0
    total_pixels_failed = 0
    pixel_results = {}  # 存储所有像素结果用于空间一致性检查
    
    # 第一遍：计算所有像素的整数模糊度
    print('First pass: calculating integer ambiguity for all pixels...')
    prog_bar = ptime.progressBar(maxValue=num_loop)
    
    for i in range(num_loop):
        r0 = i * optimal_block_size
        r1 = min((r0 + optimal_block_size), length)
        
        for j in range(r1 - r0):
            row = r0 + j
            for k in range(width):
                if mask[row, k]:
                    total_pixels_processed += 1
                    
                    try:
                        # 逐像素读取
                        unw_pixel = ifginv.read_stack_obs(
                            stack_obj,
                            box=(k, row, k+1, row+1),
                            ref_phase=ref_phase,
                            obs_ds_name=dsNameIn,
                            dropIfgram=True,
                            print_msg=False,
                        ).reshape(num_ifgram, -1)
                        
                        # 计算闭合相位
                        closure_pha = np.dot(C, unw_pixel)
                        closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
                        
                        # 检查数值稳定性
                        if np.any(np.isnan(closure_int)) or np.any(np.isinf(closure_int)):
                            pixel_results[(row, k)] = {'status': 'failed', 'reason': 'invalid_closure'}
                            total_pixels_failed += 1
                            continue
                        
                        # 求解整数模糊度
                        try:
                            U_pixel = solve_integer_ambiguity_robust(C, closure_int)
                            
                            if np.any(np.isnan(U_pixel)) or np.any(np.isinf(U_pixel)):
                                pixel_results[(row, k)] = {'status': 'failed', 'reason': 'invalid_result'}
                                total_pixels_failed += 1
                            else:
                                pixel_results[(row, k)] = {
                                    'status': 'success', 
                                    'result': U_pixel,
                                    'closure_int': closure_int
                                }
                        
                        except Exception as e:
                            pixel_results[(row, k)] = {'status': 'failed', 'reason': str(e)}
                            total_pixels_failed += 1
                    
                    except Exception as e:
                        pixel_results[(row, k)] = {'status': 'failed', 'reason': f'read_error: {e}'}
                        total_pixels_failed += 1
        
        prog_bar.update(i+1)
    
    prog_bar.close()
    
    # 第二遍：空间一致性检查和滤波
    print('Second pass: spatial consistency checking and filtering...')
    
    # 对每个时间步进行空间一致性检查
    for t in range(num_ifgram):
        print(f'Processing time step {t+1}/{num_ifgram}...')
        
        # 创建当前时间步的结果矩阵
        current_results = np.zeros((length, width), dtype=np.float32)
        current_mask = np.zeros((length, width), dtype=bool)
        
        # 填充成功的结果
        for (row, col), info in pixel_results.items():
            if info['status'] == 'success':
                current_results[row, col] = info['result'][t]
                current_mask[row, col] = True
        
        # 空间一致性检查
        half_window = spatial_window // 2
        consistency_scores = np.zeros((length, width))
        
        for row in range(half_window, length - half_window):
            for col in range(half_window, width - half_window):
                if current_mask[row, col]:
                    # 提取邻域窗口
                    window = current_results[row-half_window:row+half_window+1, 
                                          col-half_window:col+half_window+1]
                    window_mask = current_mask[row-half_window:row+half_window+1, 
                                            col-half_window:col+half_window+1]
                    
                    if np.sum(window_mask) > 1:  # 至少需要2个有效像素
                        # 计算与邻域的一致性
                        center_value = current_results[row, col]
                        neighbor_values = window[window_mask]
                        neighbor_values = neighbor_values[neighbor_values != center_value]
                        
                        if len(neighbor_values) > 0:
                            # 计算一致性分数（基于值的相似性）
                            differences = np.abs(neighbor_values - center_value)
                            consistency_score = np.sum(differences < 1.0) / len(neighbor_values)
                            consistency_scores[row, col] = consistency_score
                        else:
                            consistency_scores[row, col] = 1.0  # 没有邻居，认为一致
                    else:
                        consistency_scores[row, col] = 1.0  # 孤立像素，认为一致
        
        # 基于一致性分数进行滤波
        low_consistency_mask = consistency_scores < consistency_threshold
        
        # 对低一致性像素进行重新计算或插值
        for row in range(length):
            for col in range(width):
                if (mask[row, col] and low_consistency_mask[row, col] and 
                    pixel_results.get((row, col), {}).get('status') == 'success'):
                    
                    # 尝试使用邻域中值替换
                    r_start = max(0, row - half_window)
                    r_end = min(length, row + half_window + 1)
                    c_start = max(0, col - half_window)
                    c_end = min(width, col + half_window + 1)
                    
                    neighbor_window = current_results[r_start:r_end, c_start:c_end]
                    neighbor_mask = current_mask[r_start:r_end, c_start:c_end]
                    
                    if np.sum(neighbor_mask) > 1:
                        # 使用邻域中值
                        neighbor_values = neighbor_window[neighbor_mask]
                        median_value = np.median(neighbor_values)
                        current_results[row, col] = median_value
                        
                        # 更新像素结果
                        if (row, col) in pixel_results and pixel_results[(row, col)]['status'] == 'success':
                            pixel_results[(row, col)]['result'][t] = median_value
        
        # 应用最终结果
        int_ambiguity[t, :, :] = current_results
    
    # 打印统计信息
    print('-'*50)
    print('Enhanced processing statistics:')
    print(f'Total pixels processed: {total_pixels_processed}')
    print(f'Total pixels failed: {total_pixels_failed}')
    print(f'Success rate: {(total_pixels_processed - total_pixels_failed) / total_pixels_processed * 100:.2f}%')
    
    # mask out invalid pixels
    int_ambiguity[:, mask == 0] = np.nan
    
    print('-'*50)
    print('Spatial consistency enhanced processing completed!')
    
    return int_ambiguity, filtered_date12_list, all_date12_list


##########################################################################################
def correct_unwrap_error_pixel_wise(ifgram_file, int_ambiguity, filtered_date12_list, 
                                   all_date12_list, dsNameIn='unwrapPhase', 
                                   dsNameOut='unwrapPhase_phaseClosurePixel',
                                   aoi_mask=None):  # 新增AOI掩膜参数
    """Correct unwrapping error using pixel-wise phase closure with AOI support."""
    print('-'*50)
    print('correcting unwrapping error in ifgramStack.h5 using pixel-wise phase closure...')
    
    # 打开文件
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    
    # 使用所有干涉图进行校正
    num_ifgram = len(all_date12_list)
    
    print(f'open ifgramStack file: {ifgram_file}')
    print(f'number of interferograms to correct: {num_ifgram}')
    print(f'Note: Correcting ALL interferograms using model from filtered subset')
    
    # 创建映射：从所有干涉图到过滤后干涉图的索引
    date12_to_filtered_idx = {date12: i for i, date12 in enumerate(filtered_date12_list)}
    
    # 打开文件进行写入
    print(f'open {ifgram_file} with r+ mode')
    with h5py.File(ifgram_file, 'r+') as f:
        print('input  dataset:', dsNameIn)
        print('output dataset:', dsNameOut)
        
        # 创建或访问输出数据集
        if dsNameOut in f.keys():
            ds = f[dsNameOut]
            print(f'access /{dsNameOut} of np.float32 in size of ({num_ifgram}, {stack_obj.length}, {stack_obj.width})')
        else:
            ds = f.create_dataset(
                dsNameOut,
                (num_ifgram, stack_obj.length, stack_obj.width),
                maxshape=(None, None, None),
                chunks=True,
                compression=None)
            print(f'create /{dsNameOut} of np.float32 in size of ({num_ifgram}, {stack_obj.length}, {stack_obj.width})')
        
        # 逐干涉图校正
        prog_bar = ptime.progressBar(maxValue=num_ifgram)
        for i, date12 in enumerate(all_date12_list):
            # 检查这个干涉图是否在过滤后的列表中
            if date12 in date12_to_filtered_idx:
                # 在过滤后的列表中：使用估计的整数模糊度
                filtered_idx = date12_to_filtered_idx[date12]
                U_pixel = int_ambiguity[filtered_idx, :, :]
                
                # 如果有AOI掩膜，打印统计信息
                if aoi_mask is not None:
                    aoi_nonzero = np.count_nonzero(U_pixel[aoi_mask])
                    other_nonzero = np.count_nonzero(U_pixel[~aoi_mask])
                    print(f'Using estimated ambiguity for {date12}: AOI={aoi_nonzero}, Other={other_nonzero}')
                else:
                    print(f'Using estimated ambiguity for {date12}')
            else:
                # 不在过滤后的列表中：设为0（不校正）
                U_pixel = np.zeros((stack_obj.length, stack_obj.width), dtype=np.float32)
                print(f'No correction for {date12} (not in filtered subset)')
            
            # 读取原始数据
            unw = f[dsNameIn][i, :, :].astype(np.float32)
            
            # 应用校正
            unw_cor = unw + 2. * np.pi * U_pixel
            
            # 写入校正后的数据
            ds[i, :, :] = unw_cor
            
            prog_bar.update(i+1, suffix=date12)
        prog_bar.close()
        
        # 设置修改时间
        ds.attrs['MODIFICATION_TIME'] = str(time.time())
    
    print(f'close {ifgram_file} file.')
    print('Pixel-wise phase closure correction completed successfully!')


##########################################################################################
def get_pixel_int_ambiguity_improved(ifgram_file, mask_file=None, dsNameIn='unwrapPhase', 
                                    max_memory=4, block_size=1000):
    """Improved version with better numerical stability."""
    
    # stack info
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    length, width = stack_obj.length, stack_obj.width
    
    # 获取所有干涉图列表（用于最终输出）
    all_date12_list = stack_obj.get_date12_list(dropIfgram=False)
    total_ifgram = len(all_date12_list)
    
    # 获取过滤后的干涉图列表（用于估计）
    filtered_date12_list = stack_obj.get_date12_list(dropIfgram=True)
    num_ifgram = len(filtered_date12_list)
    
    print(f'Total interferograms: {total_ifgram}')
    print(f'Filtered interferograms for estimation: {num_ifgram}')
    print(f'Note: Using filtered subset for model estimation')
    
    # 构建设计矩阵C（基于高质量干涉图）
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,  # 估计阶段用过滤后的
    ).reshape(num_ifgram, -1)
    
    print(f'number of interferograms: {num_ifgram}')
    print(f'number of triplets: {C.shape[0]}')
    print(f'image size: {length} x {width}')
    
    # 添加C矩阵的详细调试信息
    print('-'*50)
    print('Design matrix C properties:')
    print(f'  Shape: {C.shape}')
    print(f'  Data type: {C.dtype}')
    print(f'  Memory usage: {C.nbytes / 1024**2:.2f} MB')
    
    # 检查C矩阵的稀疏性
    non_zero_elements = np.count_nonzero(C)
    total_elements = C.size
    sparsity = 1 - (non_zero_elements / total_elements)
    print(f'  Sparsity: {sparsity:.2%}')
    
    # 检查C矩阵的条件数
    try:
        cond_num = np.linalg.cond(C)
        print(f'  Condition number: {cond_num:.2e}')
        if cond_num > 1e12:
            print('  WARNING: C matrix is ill-conditioned!')
    except Exception as e:
        print(f'  Could not compute condition number: {e}')
    
    # 检查C矩阵的数值范围
    print(f'  C matrix min/max: {C.min():.6f} / {C.max():.6f}')
    print(f'  C matrix has NaN: {np.any(np.isnan(C))}')
    print(f'  C matrix has Inf: {np.any(np.isinf(C))}')
    
    # read mask
    if mask_file and os.path.isfile(mask_file):
        mask = readfile.read(mask_file)[0]
        mask = mask.astype(bool)
    else:
        mask = np.ones((length, width), dtype=bool)
    
    # initialize output array
    int_ambiguity = np.zeros((num_ifgram, length, width), dtype=np.float32)
    
    # 使用更小的块大小确保稳定性
    optimal_block_size = min(block_size, 100)
    num_loop = int(np.ceil(length / optimal_block_size))
    print(f'processing in {num_loop} blocks with size up to {optimal_block_size} x {width}')
    
    # 添加统计变量
    total_pixels_processed = 0
    total_pixels_failed = 0
    failed_pixels_by_time = np.zeros(num_ifgram, dtype=int)
    
    # process block by block
    prog_bar = ptime.progressBar(maxValue=num_loop)
    for i in range(num_loop):
        # box
        r0 = i * optimal_block_size
        r1 = min((r0 + optimal_block_size), length)
        
        print(f'processing block {i+1}/{num_loop}: rows {r0} to {r1-1}')
        
        # process each pixel in the block individually
        for j in range(r1 - r0):
            row = r0 + j
            for k in range(width):
                if mask[row, k]:
                    total_pixels_processed += 1
                    
                    # 逐像素读取数据（更稳定）
                    try:
                        unw_pixel = ifginv.read_stack_obs(
                            stack_obj,
                            box=(k, row, k+1, row+1),
                            ref_phase=ref_phase,
                            obs_ds_name=dsNameIn,
                            dropIfgram=True,
                            print_msg=False,
                        ).reshape(num_ifgram, -1)
                        
                        # calculate closure phase for this pixel
                        closure_pha = np.dot(C, unw_pixel)
                        closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
                        
                        # 检查数值稳定性
                        if np.any(np.isnan(closure_int)) or np.any(np.isinf(closure_int)):
                            print(f'Warning: Invalid values in closure_int for pixel ({row}, {k})')
                            int_ambiguity[:, row, k] = 0
                            total_pixels_failed += 1
                            failed_pixels_by_time += 1
                            continue
                        
                        # solve for integer ambiguity using robust solver
                        try:
                            U_pixel = solve_integer_ambiguity_robust(C, closure_int)
                            
                            # 检查求解结果
                            if np.any(np.isnan(U_pixel)) or np.any(np.isinf(U_pixel)):
                                print(f'Warning: Invalid values in U_pixel for pixel ({row}, {k})')
                                int_ambiguity[:, row, k] = 0
                                total_pixels_failed += 1
                                failed_pixels_by_time += 1
                            else:
                                # store result
                                int_ambiguity[:, row, k] = U_pixel
                            
                        except Exception as e:
                            print(f'Warning: failed to solve for pixel ({row}, {k}): {e}')
                            int_ambiguity[:, row, k] = 0
                            total_pixels_failed += 1
                            failed_pixels_by_time += 1
                    
                    except Exception as e:
                        print(f'Warning: failed to read data for pixel ({row}, {k}): {e}')
                        int_ambiguity[:, row, k] = 0
                        total_pixels_failed += 1
                        failed_pixels_by_time += 1
        
        prog_bar.update(i+1, every=1, suffix=f'block {i+1}/{num_loop}')
    
    prog_bar.close()
    
    # 打印统计信息
    print('-'*50)
    print('Processing statistics:')
    print(f'Total pixels processed: {total_pixels_processed}')
    print(f'Total pixels failed: {total_pixels_failed}')
    print(f'Success rate: {(total_pixels_processed - total_pixels_failed) / total_pixels_processed * 100:.2f}%')
    
    print('-'*50)
    print('Integer ambiguity statistics by time:')
    for i in range(num_ifgram):
        non_zero_count = np.count_nonzero(int_ambiguity[i, :, :])
        total_pixels = np.sum(mask)
        failed_count = failed_pixels_by_time[i]
        print(f'Time {i}: {non_zero_count}/{total_pixels} pixels have non-zero values, {failed_count} failed')
    
    # mask out invalid pixels
    int_ambiguity[:, mask == 0] = np.nan
    
    # 返回时包含映射信息
    return int_ambiguity, filtered_date12_list, all_date12_list

##########################################################################################
def plot_integer_ambiguity_comparison(ifgram_file, int_ambiguity, filtered_date12_list, 
                                     out_file=None, dsNameIn='unwrapPhase'):
    """绘制矫正前后的整数模糊度对比图."""
    print('-'*50)
    print('Plotting integer ambiguity comparison (before vs after correction)...')
    
    if out_file is None:
        out_file = ifgram_file.replace('.h5', '_integer_ambiguity_comparison.png')
    
    # 获取干涉图信息
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    
    # 构建设计矩阵C
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    
    # 获取参考相位
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,
    ).reshape(len(filtered_date12_list), -1)
    
    length, width = stack_obj.length, stack_obj.width
    num_ifgram = len(filtered_date12_list)
    
    # 计算矫正前的整数模糊度（采样一些像素作为示例）
    print('Calculating pre-correction integer ambiguity for sample pixels...')
    
    # 采样像素数量
    sample_size = min(1000, length * width // 100)  # 采样1%的像素或最多1000个
    
    # 随机选择采样像素
    rng = np.random.default_rng()
    sample_rows = rng.integers(0, length, sample_size)
    sample_cols = rng.integers(0, width, sample_size)
    
    pre_correction_ambiguity = np.zeros((num_ifgram, sample_size), dtype=np.float32)
    
    for i in range(sample_size):
        r, c = sample_rows[i], sample_cols[i]
        try:
            # 读取原始像素数据
            unw_pixel = ifginv.read_stack_obs(
                stack_obj,
                box=(c, r, c+1, r+1),
                ref_phase=ref_phase,
                obs_ds_name=dsNameIn,
                dropIfgram=True,
                print_msg=False,
            ).reshape(num_ifgram, -1)
            
            # 计算原始闭合相位
            closure_pha = np.dot(C, unw_pixel)
            closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
            
            pre_correction_ambiguity[:, i] = closure_int.flatten()
            
        except Exception as e:
            pre_correction_ambiguity[:, i] = np.nan
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Integer Ambiguity Comparison: Before vs After Correction', fontsize=16)
    
    # 1. 矫正前的整数模糊度分布（第一个时间步）
    ax1 = axes[0, 0]
    valid_pre = pre_correction_ambiguity[0, :]
    valid_pre = valid_pre[~np.isnan(valid_pre)]
    if len(valid_pre) > 0:
        ax1.hist(valid_pre, bins=30, alpha=0.7, color='red', edgecolor='black', label='Before Correction')
        ax1.set_xlabel('Integer Ambiguity')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Pre-correction Distribution (Time 0)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # 2. 矫正后的整数模糊度分布（第一个时间步）
    ax2 = axes[0, 1]
    valid_post = int_ambiguity[0, :, :]
    valid_post = valid_post[~np.isnan(valid_post)]
    if len(valid_post) > 0:
        ax2.hist(valid_post, bins=30, alpha=0.7, color='blue', edgecolor='black', label='After Correction')
        ax2.set_xlabel('Integer Ambiguity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Post-correction Distribution (Time 0)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # 3. 矫正前后的对比（所有时间步的平均值）
    ax3 = axes[0, 2]
    pre_mean = np.nanmean(pre_correction_ambiguity, axis=1)
    post_mean = np.nanmean(int_ambiguity, axis=(1, 2))
    
    x_range = range(len(pre_mean))
    ax3.plot(x_range, pre_mean, 'r-o', linewidth=2, markersize=6, label='Before Correction')
    ax3.plot(x_range, post_mean, 'b-s', linewidth=2, markersize=6, label='After Correction')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Mean Integer Ambiguity')
    ax3.set_title('Mean Ambiguity vs Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # 4. 矫正前的空间分布（第一个时间步）
    ax4 = axes[1, 0]
    # 创建空间分布图（使用采样像素）
    spatial_pre = np.full((length, width), np.nan)
    for i in range(sample_size):
        r, c = sample_rows[i], sample_cols[i]
        if not np.isnan(pre_correction_ambiguity[0, i]):
            spatial_pre[r, c] = pre_correction_ambiguity[0, i]
    
    im4 = ax4.imshow(spatial_pre, cmap='RdBu_r', aspect='auto')
    ax4.set_title('Pre-correction Spatial Distribution (Time 0)')
    ax4.set_xlabel('Column')
    ax4.set_ylabel('Row')
    plt.colorbar(im4, ax=ax4, label='Integer Ambiguity')
    
    # 5. 矫正后的空间分布（第一个时间步）
    ax5 = axes[1, 1]
    im5 = ax5.imshow(int_ambiguity[0, :, :], cmap='RdBu_r', aspect='auto')
    ax5.set_title('Post-correction Spatial Distribution (Time 0)')
    ax5.set_xlabel('Column')
    ax5.set_ylabel('Row')
    plt.colorbar(im5, ax=ax5, label='Integer Ambiguity')
    
    # 6. 统计信息表格
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # 计算统计信息
    pre_stats = []
    post_stats = []
    
    for t in range(num_ifgram):
        pre_valid = pre_correction_ambiguity[t, :]
        pre_valid = pre_valid[~np.isnan(pre_valid)]
        post_valid = int_ambiguity[t, :, :]
        post_valid = post_valid[~np.isnan(post_valid)]
        
        if len(pre_valid) > 0:
            pre_stats.append([
                f'{np.mean(pre_valid):.3f}',
                f'{np.std(pre_valid):.3f}',
                f'{np.count_nonzero(pre_valid)}'
            ])
        else:
            pre_stats.append(['N/A', 'N/A', 'N/A'])
        
        if len(post_valid) > 0:
            post_stats.append([
                f'{np.mean(post_valid):.3f}',
                f'{np.std(post_valid):.3f}',
                f'{np.count_nonzero(post_valid)}'
            ])
        else:
            post_stats.append(['N/A', 'N/A', 'N/A'])
    
    # 创建统计表格
    table_data = []
    table_data.append(['Time', 'Pre-correction', '', '', 'Post-correction', '', ''])
    table_data.append(['', 'Mean', 'Std', 'Non-zero', 'Mean', 'Std', 'Non-zero'])
    
    for t in range(min(5, num_ifgram)):  # 只显示前5个时间步
        table_data.append([
            f'{t}',
            pre_stats[t][0], pre_stats[t][1], pre_stats[t][2],
            post_stats[t][0], post_stats[t][1], post_stats[t][2]
        ])
    
    if num_ifgram > 5:
        table_data.append(['...', '...', '...', '...', '...', '...', '...'])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # 设置表格样式
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i < 2:  # 表头
                table[(i, j)].set_facecolor('lightgray')
                table[(i, j)].set_text_props(weight='bold')
    
    ax6.set_title('Statistical Comparison', pad=20)
    
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f'Integer ambiguity comparison plot saved to: {out_file}')
    
    # 显示图形
    plt.show()
    
    return out_file


##########################################################################################
def generate_num_triplet_with_nonzero_integer_ambiguity(ifgram_file, int_ambiguity, filtered_date12_list, 
                                                       out_file=None, dsNameIn='unwrapPhase'):
    """生成numTriNonzeroIntAmbiguity.h5文件，用于triplet分析绘图."""
    print('-'*50)
    print(f'Generating numTriNonzeroIntAmbiguity.h5 file for triplet analysis using dataset: {dsNameIn}')
    
    if out_file is None:
        # 根据数据集名称自动生成文件名
        if dsNameIn == 'unwrapPhase_phaseClosurePixel':
            out_file = 'numTriNonzeroIntAmbiguity_corrected.h5'
        else:
            out_file = 'numTriNonzeroIntAmbiguity_original.h5'
    
    # 获取干涉图信息
    stack_obj = ifgramStack(ifgram_file)
    stack_obj.open(print_msg=False)
    
    # 构建设计矩阵C
    C = ifgramStack.get_design_matrix4triplet(filtered_date12_list).astype(float)
    num_triplets = C.shape[0]
    
    length, width = stack_obj.length, stack_obj.width
    num_ifgram = len(filtered_date12_list)
    
    print(f'Number of triplets: {num_triplets}')
    print(f'Image size: {length} x {width}')
    
    # 初始化结果数组
    num_nonzero_triplets = np.zeros((length, width), dtype=np.int32)
    
    # 获取参考相位
    ref_phase = stack_obj.get_reference_phase(
        unwDatasetName=dsNameIn,
        dropIfgram=True,
    ).reshape(num_ifgram, -1)
    
    # 逐像素计算具有非零整数模糊度的triplet数量
    print('Calculating number of triplets with non-zero integer ambiguity for each pixel...')
    
    # 使用较小的块大小
    block_size = 100
    num_loop = int(np.ceil(length / block_size))
    
    prog_bar = ptime.progressBar(maxValue=num_loop)
    for i in range(num_loop):
        r0 = i * block_size
        r1 = min((r0 + block_size), length)
        
        for j in range(r1 - r0):
            row = r0 + j
            for k in range(width):
                try:
                    # 读取像素数据
                    unw_pixel = ifginv.read_stack_obs(
                        stack_obj,
                        box=(k, row, k+1, row+1),
                        ref_phase=ref_phase,
                        obs_ds_name=dsNameIn,
                        dropIfgram=True,
                        print_msg=False,
                    ).reshape(num_ifgram, -1)
                    
                    # 计算闭合相位
                    closure_pha = np.dot(C, unw_pixel)
                    closure_int = np.round((closure_pha - ut.wrap(closure_pha)) / (2.*np.pi))
                    
                    # 计算非零整数模糊度的triplet数量
                    nonzero_count = np.count_nonzero(np.abs(closure_int) > 0.1)  # 允许0.1的误差
                    num_nonzero_triplets[row, k] = nonzero_count
                    
                except Exception as e:
                    # 如果计算失败，设为-1
                    num_nonzero_triplets[row, k] = -1
        
        prog_bar.update(i+1)
    
    prog_bar.close()
    
    # 统计信息
    valid_pixels = num_nonzero_triplets >= 0
    if np.any(valid_pixels):
        valid_data = num_nonzero_triplets[valid_pixels]
        print(f'Statistics of non-zero triplet counts:')
        print(f'  Mean: {np.mean(valid_data):.2f}')
        print(f'  Std: {np.std(valid_data):.2f}')
        print(f'  Min: {np.min(valid_data)}')
        print(f'  Max: {np.max(valid_data)}')
        print(f'  Pixels with 0 non-zero triplets: {np.sum(valid_data == 0)}')
        print(f'  Pixels with >0 non-zero triplets: {np.sum(valid_data > 0)}')
        print(f'  Failed pixels (-1): {np.sum(~valid_pixels)}')
    
    # 写入文件
    print(f'Writing to file: {out_file}')
    meta = dict(stack_obj.metadata)
    meta['FILE_TYPE'] = 'mask'
    meta['UNIT'] = '1'
    
    # 将-1（失败像素）设为NaN，并转换为float32类型
    output_data = num_nonzero_triplets.astype(np.float32)
    output_data[output_data == -1] = np.nan
    
    meta['DATA_TYPE'] = 'float32'  # 最终输出类型为float32
    
    writefile.write(
        output_data,
        out_file=out_file,
        metadata=meta,
    )
    
    print(f'Successfully generated: {out_file}')
    return out_file


##########################################################################################
def generate_triplet_analysis_comparison(ifgram_file, int_ambiguity, filtered_date12_list, 
                                        out_dir=None, dsNameIn='unwrapPhase'):
    """生成矫正前后的triplet分析文件进行对比."""
    print('-'*50)
    print('Generating triplet analysis files for comparison (before vs after correction)...')
    
    if out_dir is None:
        out_dir = '.'
    
    # 生成矫正前的triplet分析文件
    print('Step 1: Generating pre-correction triplet analysis...')
    original_file = generate_num_triplet_with_nonzero_integer_ambiguity(
        ifgram_file=ifgram_file,
        int_ambiguity=int_ambiguity,
        filtered_date12_list=filtered_date12_list,
        out_file=os.path.join(out_dir, 'numTriNonzeroIntAmbiguity_original.h5'),
        dsNameIn='unwrapPhase'  # 使用原始数据
    )
    
    # 生成矫正后的triplet分析文件
    print('Step 2: Generating post-correction triplet analysis...')
    corrected_file = generate_num_triplet_with_nonzero_integer_ambiguity(
        ifgram_file=ifgram_file,
        int_ambiguity=int_ambiguity,
        filtered_date12_list=filtered_date12_list,
        out_file=os.path.join(out_dir, 'numTriNonzeroIntAmbiguity_corrected.h5'),
        dsNameIn='unwrapPhase_phaseClosurePixel'  # 使用矫正后的数据
    )
    
    # 生成对比图
    print('Step 3: Generating comparison plots...')
    try:
        from mintpy.utils import plot as pp
        
        # 生成原始数据的图
        print('  Generating original data plot...')
        pp.plot_num_triplet_with_nonzero_integer_ambiguity(original_file)
        
        # 生成矫正后数据的图
        print('  Generating corrected data plot...')
        pp.plot_num_triplet_with_nonzero_integer_ambiguity(corrected_file)
        
        # 生成对比图
        print('  Generating comparison plot...')
        generate_triplet_comparison_plot(original_file, corrected_file, out_dir)
        
    except ImportError as e:
        print(f'Warning: Could not import plot module: {e}')
        print('Skipping plot generation')
    except Exception as e:
        print(f'Warning: Error generating plots: {e}')
    
    print('-'*50)
    print('Triplet analysis comparison completed!')
    print(f'Original file: {original_file}')
    print(f'Corrected file: {corrected_file}')
    
    return original_file, corrected_file


##########################################################################################
def generate_triplet_comparison_plot(original_file, corrected_file, out_dir):
    """生成triplet分析的对比图."""
    print('Generating triplet comparison plot...')
    
    try:
        import h5py
        import matplotlib.pyplot as plt
        
        # 读取数据
        with h5py.File(original_file, 'r') as f:
            original_data = f['data'][:]
            original_meta = dict(f.attrs)
        
        with h5py.File(corrected_file, 'r') as f:
            corrected_data = f['data'][:]
            corrected_meta = dict(f.attrs)
        
        # 创建对比图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Triplet Analysis Comparison: Before vs After Correction', fontsize=16)
        
        # 1. 原始数据分布
        ax1 = axes[0, 0]
        valid_orig = original_data[~np.isnan(original_data)]
        if len(valid_orig) > 0:
            ax1.hist(valid_orig, bins=30, alpha=0.7, color='red', edgecolor='black', label='Before Correction')
            ax1.set_xlabel('Number of Non-zero Triplets')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Pre-correction Distribution')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 2. 矫正后数据分布
        ax2 = axes[0, 1]
        valid_corr = corrected_data[~np.isnan(corrected_data)]
        if len(valid_corr) > 0:
            ax2.hist(valid_corr, bins=30, alpha=0.7, color='blue', edgecolor='black', label='After Correction')
            ax2.set_xlabel('Number of Non-zero Triplets')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Post-correction Distribution')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # 3. 对比直方图
        ax3 = axes[0, 2]
        if len(valid_orig) > 0 and len(valid_corr) > 0:
            ax3.hist(valid_orig, bins=30, alpha=0.5, color='red', edgecolor='black', label='Before Correction')
            ax3.hist(valid_corr, bins=30, alpha=0.5, color='blue', edgecolor='black', label='After Correction')
            ax3.set_xlabel('Number of Non-zero Triplets')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Overlay Comparison')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # 4. 原始数据空间分布
        ax4 = axes[1, 0]
        im4 = ax4.imshow(original_data, cmap='viridis', aspect='auto')
        ax4.set_title('Pre-correction Spatial Distribution')
        ax4.set_xlabel('Column')
        ax4.set_ylabel('Row')
        plt.colorbar(im4, ax=ax4, label='Non-zero Triplet Count')
        
        # 5. 矫正后数据空间分布
        ax5 = axes[1, 1]
        im5 = ax5.imshow(corrected_data, cmap='viridis', aspect='auto')
        ax5.set_title('Post-correction Spatial Distribution')
        ax5.set_xlabel('Column')
        ax5.set_ylabel('Row')
        plt.colorbar(im5, ax=ax5, label='Non-zero Triplet Count')
        
        # 6. 统计信息表格
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 计算统计信息
        if len(valid_orig) > 0:
            orig_stats = [
                f'{np.mean(valid_orig):.2f}',
                f'{np.std(valid_orig):.2f}',
                f'{np.min(valid_orig)}',
                f'{np.max(valid_orig)}',
                f'{np.sum(valid_orig == 0)}',
                f'{np.sum(valid_orig > 0)}'
            ]
        else:
            orig_stats = ['N/A'] * 6
        
        if len(valid_corr) > 0:
            corr_stats = [
                f'{np.mean(valid_corr):.2f}',
                f'{np.std(valid_corr):.2f}',
                f'{np.min(valid_corr)}',
                f'{np.max(valid_corr)}',
                f'{np.sum(valid_corr == 0)}',
                f'{np.sum(valid_corr > 0)}'
            ]
        else:
            corr_stats = ['N/A'] * 6
        
        # 创建统计表格
        table_data = [
            ['Metric', 'Before Correction', 'After Correction', 'Improvement'],
            ['Mean', orig_stats[0], corr_stats[0], ''],
            ['Std', orig_stats[1], corr_stats[1], ''],
            ['Min', orig_stats[2], corr_stats[2], ''],
            ['Max', orig_stats[3], corr_stats[3], ''],
            ['Zero Count', orig_stats[4], corr_stats[4], ''],
            ['Non-zero Count', orig_stats[5], corr_stats[5], '']
        ]
        
        # 计算改进程度
        if len(valid_orig) > 0 and len(valid_corr) > 0:
            zero_improvement = np.sum(valid_corr == 0) - np.sum(valid_orig == 0)
            if zero_improvement > 0:
                table_data[5][3] = f'+{zero_improvement} (Better)'
            elif zero_improvement < 0:
                table_data[5][3] = f'{zero_improvement} (Worse)'
            else:
                table_data[5][3] = 'No change'
        
        table = ax6.table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                if i < 2:  # 表头
                    table[(i, j)].set_facecolor('lightgray')
                    table[(i, j)].set_text_props(weight='bold')
                elif j == 3 and i > 1:  # 改进列
                    if 'Better' in table_data[i][j]:
                        table[(i, j)].set_facecolor('lightgreen')
                    elif 'Worse' in table_data[i][j]:
                        table[(i, j)].set_facecolor('lightcoral')
        
        ax6.set_title('Statistical Comparison', pad=20)
        
        plt.tight_layout()
        
        # 保存图形
        comparison_plot = os.path.join(out_dir, 'triplet_analysis_comparison.png')
        plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
        print(f'Triplet comparison plot saved to: {comparison_plot}')
        
        # 显示图形
        plt.show()
        
        return comparison_plot
        
    except Exception as e:
        print(f'Error generating triplet comparison plot: {e}')
        return None


##########################################################################################
def run_unwrap_error_phase_closure_pixel(inps):
    """Main function for hybrid phase closure correction."""
    print('Step 1: Calculate pixel-wise integer ambiguity...')
    
    # 添加调试信息
    print(f'Debug: inps.aoi_coords = {getattr(inps, "aoi_coords", None)}')
    print(f'Debug: hasattr(inps, "aoi_coords") = {hasattr(inps, "aoi_coords")}')
    
    # 临时修复：直接检查命令行参数
    import sys
    aoi_coords = None
    aoi_only = False
    if '--aoi_coords' in sys.argv:
        try:
            aoi_index = sys.argv.index('--aoi_coords')
            if aoi_index + 1 < len(sys.argv):
                aoi_coords = sys.argv[aoi_index + 1]
                print(f'Found AOI in command line: {aoi_coords}')
        except (ValueError, IndexError):
            pass
    
    # 检查是否启用aoi_only模式
    if '--aoi_only' in sys.argv:
        aoi_only = True
        print('AOI-only mode enabled: will only process AOI region')
    
    # 检查是否指定了AOI
    if aoi_coords:
        print(f'Using {"AOI-only" if aoi_only else "hybrid"} approach with AOI: {aoi_coords}')
        
        # 混合方法：AOI内逐像素，其他区域区域采样（或跳过）
        int_ambiguity, filtered_date12_list, all_date12_list = get_hybrid_int_ambiguity(
            ifgram_file=inps.ifgram_file,
            mask_file=inps.mask_file,
            aoi_coords=aoi_coords,
            dsNameIn=inps.datasetNameIn,
            max_memory=inps.max_memory,
            block_size=inps.block_size,
            num_sample=getattr(inps, 'region_sample_size', 100),
            min_area=getattr(inps, 'min_region_area', 2.5e3),
            aoi_only=aoi_only,  # 传递aoi_only参数
        )
        
        # 创建AOI掩膜用于矫正阶段
        stack_obj = ifgramStack(inps.ifgram_file)
        stack_obj.open(print_msg=False)
        length, width = stack_obj.length, stack_obj.width
        
        # 解析AOI坐标并创建掩膜
        if isinstance(aoi_coords, str):
            y_part, x_part = aoi_coords.split(',')
            y0, y1 = map(int, y_part.split(':'))
            x0, x1 = map(int, x_part.split(':'))
        else:
            y0, y1, x0, x1 = aoi_coords
        
        aoi_mask = np.zeros((length, width), dtype=bool)
        aoi_mask[y0:y1+1, x0:x1+1] = True
        
    else:
        print('Using improved pixel-wise approach for all pixels')
        aoi_mask = None
        
        # 检查是否启用增强功能
        use_robust_method = getattr(inps, 'use_robust_method', True)
        use_spatial_consistency = getattr(inps, 'use_spatial_consistency', False)
        
        if use_spatial_consistency:
            print('Using spatial consistency enhanced method...')
            int_ambiguity, filtered_date12_list, all_date12_list = get_pixel_int_ambiguity_with_spatial_consistency(
                ifgram_file=inps.ifgram_file,
                mask_file=inps.mask_file,
                dsNameIn=inps.datasetNameIn,
                max_memory=inps.max_memory,
                block_size=inps.block_size,
                spatial_window=getattr(inps, 'spatial_window', 5),
                consistency_threshold=getattr(inps, 'consistency_threshold', 0.8),
            )
        elif use_robust_method:
            print('Using robust pixel-wise method with median filtering...')
            int_ambiguity, filtered_date12_list, all_date12_list = get_pixel_int_ambiguity_robust(
                ifgram_file=inps.ifgram_file,
                mask_file=inps.mask_file,
                dsNameIn=inps.datasetNameIn,
                max_memory=inps.max_memory,
                block_size=inps.block_size,
                median_filter_size=getattr(inps, 'median_filter_size', 3),
            )
        else:
            print('Using original improved pixel-wise method...')
            int_ambiguity, filtered_date12_list, all_date12_list = get_pixel_int_ambiguity_improved(
                ifgram_file=inps.ifgram_file,
                mask_file=inps.mask_file,
                dsNameIn=inps.datasetNameIn,
                max_memory=inps.max_memory,
                block_size=inps.block_size,
            )
    
    print('Step 2: Apply pixel-wise correction...')
    
    # 应用校正（传递AOI掩膜）
    correct_unwrap_error_pixel_wise(
        ifgram_file=inps.ifgram_file,
        int_ambiguity=int_ambiguity,
        filtered_date12_list=filtered_date12_list,
        all_date12_list=all_date12_list,
        dsNameIn=inps.datasetNameIn,
        dsNameOut=inps.datasetNameOut,
        aoi_mask=aoi_mask,  # 传递AOI掩膜
    )
    
    # 修改Step 3: 生成triplet分析文件（如果启用）
    if getattr(inps, 'generate_triplet_analysis', False):
        print('Step 3: Generating triplet analysis files for comparison...')
        
        if getattr(inps, 'generate_comparison', False):
            # 生成对比分析
            original_file, corrected_file = generate_triplet_analysis_comparison(
                ifgram_file=inps.ifgram_file,
                int_ambiguity=int_ambiguity,
                filtered_date12_list=filtered_date12_list,
                out_dir=getattr(inps, 'out_dir', '.'),
                dsNameIn=inps.datasetNameIn,
            )
        else:
            # 只生成矫正前的分析（原有功能）
            triplet_file = generate_num_triplet_with_nonzero_integer_ambiguity(
                ifgram_file=inps.ifgram_file,
                int_ambiguity=int_ambiguity,
                filtered_date12_list=filtered_date12_list,
                out_file=getattr(inps, 'triplet_file', None),
                dsNameIn=inps.datasetNameIn,
            )
            
            # 调用绘图函数
            try:
                from mintpy.utils import plot as pp
                print('Step 4: Generating triplet analysis plot...')
                
                # 创建输出文件名
                if triplet_file.endswith('.h5'):
                    plot_file = triplet_file.replace('.h5', '.png')
                else:
                    plot_file = triplet_file + '.png'
                
                # 调用绘图函数
                pp.plot_num_triplet_with_nonzero_integer_ambiguity(triplet_file)
                print(f'Triplet analysis plot saved to: {plot_file}')
                
            except ImportError as e:
                print(f'Warning: Could not import plot module: {e}')
                print('Skipping triplet analysis plot generation')
            except Exception as e:
                print(f'Warning: Error generating triplet analysis plot: {e}')
    
    # 新增：Step 3/4: 绘制矫正前后对比图（如果启用）
    if getattr(inps, 'plot_comparison', False):
        step_num = "Step 3" if not getattr(inps, 'generate_triplet_analysis', False) else "Step 5"
        print(f'{step_num}: Generating integer ambiguity comparison plot...')
        
        plot_integer_ambiguity_comparison(
            ifgram_file=inps.ifgram_file,
            int_ambiguity=int_ambiguity,
            filtered_date12_list=filtered_date12_list,
            out_file=getattr(inps, 'comparison_plot', None),
            dsNameIn=inps.datasetNameIn,
        )
    
    print('-'*50)
    print('Processing completed successfully!')


##########################################################################################
def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(description='Pixel-wise phase closure correction')
    
    # 现有参数
    parser.add_argument('ifgram_file', help='Input interferogram stack file')
    parser.add_argument('--mask_file', help='Mask file for valid pixels')
    parser.add_argument('--datasetNameIn', default='unwrapPhase', help='Input dataset name')
    parser.add_argument('--datasetNameOut', default='unwrapPhase_phaseClosurePixel', help='Output dataset name')
    parser.add_argument('--max_memory', type=float, default=4, help='Maximum memory usage in GB')
    parser.add_argument('--block_size', type=int, default=1000, help='Block size for processing')
    
    # 修改AOI参数 - 确保参数名完全一致
    parser.add_argument('--aoi_coords', type=str, help='AOI coordinates in format "y0:y1,x0:x1" (e.g., "100:200,50:150")')
    
    # 新增：AOI-only模式参数
    parser.add_argument('--aoi_only', action='store_true', default=False,
                       help='When AOI is specified, only process AOI region and skip other regions')
    
    # 区域采样参数
    parser.add_argument('--region_sample_size', type=int, default=100, help='Sample size for region-based approach')
    parser.add_argument('--min_region_area', type=float, default=2.5e3, help='Minimum region area for sampling')
    
    # 新增：增强功能参数
    parser.add_argument('--use_robust_method', action='store_true', default=True, 
                       help='Use robust pixel-wise method with median filtering')
    parser.add_argument('--use_spatial_consistency', action='store_true', default=False,
                       help='Use spatial consistency enhanced method')
    parser.add_argument('--median_filter_size', type=int, default=3,
                       help='Size of median filter kernel (must be odd)')
    parser.add_argument('--spatial_window', type=int, default=5,
                       help='Size of spatial window for consistency checking')
    parser.add_argument('--consistency_threshold', type=float, default=0.8,
                       help='Threshold for spatial consistency (0-1)')
    
    # 新增：triplet分析选项
    parser.add_argument('--generate_triplet_analysis', action='store_true', default=False,
                       help='Generate numTriNonzeroIntAmbiguity.h5 file for triplet analysis')
    parser.add_argument('--generate_comparison', action='store_true', default=False,
                       help='Generate both pre and post-correction triplet analysis files for comparison')
    parser.add_argument('--triplet_file', type=str, 
                       help='Output file for triplet analysis (default: auto-generated)')
    parser.add_argument('--out_dir', type=str, default='.',
                       help='Output directory for triplet analysis files')
    
    # 新增：绘图选项
    parser.add_argument('--plot_comparison', action='store_true', default=False,
                       help='Generate integer ambiguity comparison plot (before vs after correction)')
    parser.add_argument('--comparison_plot', type=str, 
                       help='Output file for integer ambiguity comparison plot')
    
    # 其他参数
    parser.add_argument('--update_mode', action='store_true', help='Enable update mode')
    
    return parser


##########################################################################################
def main():
    """Main function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.ifgram_file):
        raise FileNotFoundError(f'Input file not found: {args.ifgram_file}')
    
    # 检查AOI文件
    if args.aoi_coords and not args.aoi_coords.replace(':', ',').replace(',', '').isdigit():
        # 如果aoi_coords包含冒号或逗号，则认为它是一个字符串
        pass # 继续，因为已经处理了
    elif args.aoi_coords and os.path.isfile(args.aoi_coords):
        # 如果aoi_coords是一个文件名，则检查其存在性
        pass # 继续，因为已经处理了
    else:
        # 如果aoi_coords是None或无效的文件名，则不进行检查
        args.aoi_coords = None
    
    # 运行主函数
    run_unwrap_error_phase_closure_pixel(args)
    

##########################################################################################
if __name__ == '__main__':
    main()
