############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Heresh Fattahi, Zhang Yunjun, 2013               #
############################################################
# Add --no-offset option by Robert Zinke, Nov 2020


import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale

from mintpy.multilook import multilook_data
from mintpy.utils import plot as pp, readfile, utils as ut, writefile


#############################################################################################
def manual_offset_estimate(mat1, mat2):
    """Manually estimate offset between two data matrix.
    By manually selecting a line from each of them, and estimate the difference.
    It usually used when 2 input data matrix have no area in common.
    """
    def onclick(event):
        if event.button == 1:
            print('click')
            xc.append(int(event.xdata))
            yc.append(int(event.ydata))

    # Select line from data matrix 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(mat1)
    xc = []
    yc = []
    print('please click on start and end point of the desired profile/line')
    print('afterward close the figure to continue the process')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    x0 = xc[0]
    x1 = xc[1]
    y0 = yc[0]
    y1 = yc[1]

    # Select line from data matrix 2
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(mat2)
    xc = []
    yc = []
    print('please click on start and end point of the desired profile')
    print('afterward close the figure to continue the process')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    x00 = xc[0]
    x11 = xc[1]
    y00 = yc[0]
    y11 = yc[1]

    # Calculate the Offset - Difference
    # offset=V2[y00:y11,x00:x11]-V2[y0:y1,x0:x1]
    offset = (  np.nansum(mat2[y00:y11, x00:x11]) / np.sum(np.isfinite(mat2[y00:y11, x00:x11]))
              - np.nansum(mat1[y0:y1, x0:x1]) / np.sum(np.isfinite(mat1[y0:y1, x0:x1])))

    return offset


#############################################################################################
def rescale_data(data, meta, ref_meta):
    """rescale matrix into a different resolution"""

    # calc scaling factor
    scale = (float(meta['Y_STEP']) / float(ref_meta['Y_STEP']),
             float(meta['X_STEP']) / float(ref_meta['X_STEP']))
    # scale
    data_out = rescale(data, scale)
    # update metadata
    meta['Y_STEP'] = ref_meta['Y_STEP']
    meta['X_STEP'] = ref_meta['X_STEP']
    meta['LENGTH'], meta['WIDTH'] = data_out.shape  # 这里更新了属性

    return data_out, meta


def get_corners(atr):
    """Get corners coordinate."""
    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])
    W = float(atr['X_FIRST'])
    N = float(atr['Y_FIRST'])
    lon_step = float(atr['X_STEP'])
    lat_step = float(atr['Y_STEP'])
    S = N + lat_step * length
    E = W + lon_step * width

    return S, N, W, E, width, length


def stitch_two_matrices(mat1, atr1, mat2, atr2, apply_offset=True, 
                       offset_method='mean', print_msg=True):
    """Stitch two geocoded matrices
    and/or shift the 2nd matrix value to match with the 1st one.

    Parameters: mat1/2       - 2D np.ndarray
                atr1/2       - dict, attributes
                apply_offset - bool, estimate offset and adjust 2nd matrix value
                offset_method - str, method to calculate offset ('mean' or 'median')
    Returns:    mat          - 2D np.ndarray, stitched matrix
                atr          - dict, attributes of stitched matrix
    """
    vprint = print if print_msg else lambda *args, **kwargs: None

    # resize the 2nd matrix, if it has different spatial resolution
    ratio_x = abs((float(atr1['X_STEP']) - float(atr2['X_STEP'])) / float(atr1['X_STEP']))
    ratio_y = abs((float(atr1['Y_STEP']) - float(atr2['Y_STEP'])) / float(atr1['Y_STEP']))
    if any(i > 1e-3 for i in [ratio_x, ratio_y]):
        vprint('file 1: X_STEP - {}, Y_STEP - {}'.format(atr1['X_STEP'], atr1['Y_STEP']))
        vprint('file 2: X_STEP - {}, Y_STEP - {}'.format(atr2['X_STEP'], atr2['Y_STEP']))
        vprint('rescale the 2nd matrix into the same spatial resolution as the 1st one ...')
        mat2, atr2 = rescale_data(mat2, meta=atr2, ref_meta=atr1)

    # input spatial extents
    vprint('grab corners of input matrices')
    S1, N1, W1, E1, width1, length1 = get_corners(atr1)
    S2, N2, W2, E2, width2, length2 = get_corners(atr2)

    # output spatial extent
    vprint('calculate corners of output matrix')
    W, E = min(W1, W2), max(E1, E2)
    S, N = min(S1, S2), max(N1, N2)
    lon_step = float(atr1['X_STEP'])
    lat_step = float(atr1['Y_STEP'])
    width  = int(np.ceil((E - W) / lon_step))
    length = int(np.ceil((S - N) / lat_step))

    # index of input matrices in output matrix
    vprint('estimate difference in the overlapping area')
    lon_seq = np.arange(W, W + width  * lon_step, lon_step)
    lat_seq = np.arange(N, N + length * lat_step, lat_step)
    x1, y1 = np.argmin(np.square(lon_seq - W1)), np.argmin(np.square(lat_seq - N1))
    x2, y2 = np.argmin(np.square(lon_seq - W2)), np.argmin(np.square(lat_seq - N2))

    # estimate offset of the overlapping area
    mat11 = np.zeros([length, width]) * np.nan;
    mat22 = np.zeros([length, width]) * np.nan;
    mat11[y1:y1+length1, x1:x1+width1] = mat1
    mat22[y2:y2+length2, x2:x2+width2] = mat2
    mat_diff = mat22 - mat11

    # apply the offset
    if apply_offset:
        if offset_method == 'mean':
            offset = np.nansum(mat_diff) / np.sum(np.isfinite(mat_diff))
        elif offset_method == 'median':
            valid_diff = mat_diff[np.isfinite(mat_diff)]
            offset = np.median(valid_diff) if len(valid_diff) > 0 else 0
        
        if ~np.isnan(offset):
            vprint(f'average offset between two matrices in the common area: {offset} (method: {offset_method})')
            vprint(f'offset all pixel values in the 2nd matrix by {offset} ')
            mat2 -= offset
        else:
            print('*'*50)
            print('WARNING: NO common area found between two matrices!')
            print('    continue the stitching without applying offset')
            print('*'*50)

    # initiate output matrix
    # with the default value of NaN for float type and 0 for the other types
    vprint(f'create output metadata and matrix in shape of {(length, width)}')
    fill_value = np.nan if str(mat1.dtype).startswith('float') else 0
    mat = np.zeros([length, width], dtype=mat1.dtype) * fill_value

    # fill the output matrix
    flag2 = np.isfinite(mat2)
    mat[y1:y1+length1, x1:x1+width1] = mat1
    mat[y2:y2+length2, x2:x2+width2][flag2] = mat2[flag2]

    # output attributes
    atr = dict()
    for key, value in atr1.items():
        atr[key] = value
    atr['WIDTH'] = width
    atr['LENGTH'] = length
    atr['X_FIRST'] = W
    atr['Y_FIRST'] = N
    print(f'update LENGTH/WIDTH: {length}/{width}')
    print(f'update Y/X_FIRST: {N}/{W}')

    # update REF_Y/X
    coord = ut.coordinate(atr)
    ref_y, ref_x = coord.geo2radar(float(atr['REF_LAT']), float(atr['REF_LON']))[:2]
    atr['REF_Y'], atr['REF_X'] = ref_y, ref_x
    print(f'update REF_Y/X: {ref_y}/{ref_x}')

    # delete SUBSET_Y/XMIN/MAX
    for key in ['SUBSET_XMIN', 'SUBSET_XMAX', 'SUBSET_YMIN', 'SUBSET_YMAX']:
        if key in atr.keys():
            atr.pop(key)
            print(f'remove {key}')

    return mat, atr, mat11, mat22, mat_diff


def plot_stitch(mat11, mat22, mat, mat_diff, out_fig=None, disp_scale=1, disp_vlim=None, disp_cmap=None):
    """plot stitching result"""

    # plot settings
    titles = ['file 1', 'file 2', 'stitched', 'difference']
    if disp_scale != 1:
        print(f'scale the data by a factor of {disp_scale} for plotting')
    mat_mli = multilook_data(mat, 20, 20, method='mean')
    vmin = disp_vlim[0] if disp_vlim else np.nanmin(mat_mli) * disp_scale
    vmax = disp_vlim[1] if disp_vlim else np.nanmax(mat_mli) * disp_scale

    fig_size = pp.auto_figure_size(ds_shape=mat.shape, scale=1.4, disp_cbar=True, print_msg=True)

    # plot
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=fig_size, sharex=True, sharey=True)
    for ax, data, title in zip(axs.flatten(), [mat11, mat22, mat, mat_diff], titles):
        im = ax.imshow(data * disp_scale, vmin=vmin, vmax=vmax, cmap=disp_cmap, interpolation='nearest')
        ax.set_title(title, fontsize=12)
        ax.tick_params(which='both', direction='in', labelsize=12, left=True, right=True, top=True, bottom=True)
    fig.tight_layout()

    # colorbar
    fig.subplots_adjust(right=0.9)
    cax = fig.add_axes([0.901, 0.3, 0.01, 0.4])
    plt.colorbar(im, cax=cax)

    # output
    fig.savefig(out_fig, bbox_inches='tight', transparent=True, dpi=150)
    print(f'save figure to file: {out_fig}')

    return


def stitch_files(fnames, out_file, apply_offset=True, offset_method='mean',
                 coherence_file=None, coherence_threshold=0.3, coherence_method='median',
                 use_coherence_fusion=True,  # 添加新参数
                 disp_fig=True, no_data_value=None, disp_scale=1, 
                 disp_vlim=None, disp_cmap=None):
    """Stitch all input files into one
    """
    fext = os.path.splitext(fnames[0])[1]
    atr = readfile.read_attribute(fnames[0])

    # grab ds_names
    ds_names = set(readfile.get_dataset_list(fnames[0]))
    # get the common dataset list among all input files
    for fname in fnames[1:]:
        ds_names.intersection_update(readfile.get_dataset_list(fname))
    ds_names = sorted(list(ds_names))

    # special treatment for velocity/time_function files
    if atr['FILE_TYPE'] == 'velocity' and len(ds_names) > 1:
        ds_names = ['velocity']

    print(f'files to be stitched: {fnames}')
    print(f'datasets to be stitched: {ds_names}')
    print(f'use coherence fusion: {use_coherence_fusion}')

    # stitching
    dsDict = {}
    coherence_stitched = None  # 移到外层，避免重复初始化
    atr_coherence = None  # 添加相干性元数据变量
    
    for ds_name in ds_names:
        # reading
        mat, atr = readfile.read(fnames[0], datasetName=ds_name)
        ds_name_out = ds_name if ds_name else atr['FILE_TYPE']
        print('#'*50)
        print(f'read {ds_name_out} from file: {fnames[0]}')

        # masking
        if no_data_value is not None:
            print(f'convert no_data_value from {no_data_value} to NaN')
            mat[mat==no_data_value] = np.nan

        # skip pixels with zero incidenceAngle for geometry files
        if atr['FILE_TYPE'] in ['geometry', 'los'] and 'incidenceAngle' in ds_names:
            print('ignore pixels with ZERO incidenceAngle')
            inc_angle = readfile.read(fnames[0], datasetName='incidenceAngle')[0]
            mat[inc_angle == 0] = np.nan

        # 初始化相干性数据（只在第一次循环时）
        if coherence_file and coherence_stitched is None:
            coherence1_file = find_coherence_file(fnames[0], coherence_file)
            if coherence1_file:
                coherence_stitched, atr_coherence = readfile.read(coherence1_file)
                print(f"  Using coherence file: {coherence1_file}")
                print(f"  Coherence file type: {atr_coherence.get('FILE_TYPE', 'unknown')}")
                print(f"  Coherence shape: {coherence_stitched.shape}")
            else:
                print(f"  Warning: Could not find coherence file for {fnames[0]}")

        for i, fname in enumerate(fnames[1:]):
            print('-'*30)
            print(f'read data from file: {fname}')
            # reading
            mat2, atr2 = readfile.read(fname, datasetName=ds_name)
            # masking
            if no_data_value is not None:
                mat2[mat2==no_data_value] = np.nan
            # skip pixels with zero incidenceAngle for geometry files
            if atr['FILE_TYPE'] in ['geometry', 'los'] and 'incidenceAngle' in ds_names:
                print('ignore pixels with ZERO incidenceAngle')
                inc_angle2 = readfile.read(fname, datasetName='incidenceAngle')[0]
                mat2[inc_angle2 == 0] = np.nan

            print('stitching ...')
            
            # 读取当前文件的相干性
            coherence2 = None
            if coherence_file:
                coherence2_file = find_coherence_file(fname, coherence_file)
                if coherence2_file:
                    coherence2, _ = readfile.read(coherence2_file)
                    print(f"  Using coherence file: {coherence2_file}")
                else:
                    print(f"  Warning: Could not find coherence file for {fname}")
            
            # 根据use_coherence_fusion参数选择拼接方法
            if coherence_file and coherence_stitched is not None and coherence2 is not None and use_coherence_fusion:
                # 使用相干性引导的拼接
                result = stitch_two_matrices_with_coherence(
                    mat, atr, mat2, atr2,
                    coherence_stitched, coherence2,
                    coherence_threshold, coherence_method,
                    print_msg=True
                )
                
                if len(result) == 3:
                    mat, atr, coherence_stitched = result
                    print(f"Coherence stitched shape: {coherence_stitched.shape}")
                    print(f"Coherence stitched valid pixels: {np.sum(~np.isnan(coherence_stitched))}")
                else:
                    mat, atr = result
                    coherence_stitched = None
                
                # 为了兼容原有的绘图代码，创建虚拟的 mat11, mat22, mat_diff
                mat11 = np.zeros_like(mat) * np.nan
                mat22 = np.zeros_like(mat) * np.nan
                mat_diff = np.zeros_like(mat) * np.nan
            else:
                # 使用标准拼接
                mat, atr, mat11, mat22, mat_diff = stitch_two_matrices(
                    mat, atr,
                    mat2, atr2,
                    apply_offset=apply_offset,
                    offset_method=offset_method)
                
                # 拼接相干性文件（如果提供了相干性文件但不使用融合）
                if coherence_file and coherence_stitched is not None and coherence2 is not None:
                    print('stitching coherence files using standard method...')
                    coherence_stitched = stitch_coherence_proper(
                        coherence_stitched, atr,
                        coherence2, atr2,
                        mat.shape, atr
                    )

            # plot
            if apply_offset:
                print('plot stitching & shifting result ...')
                out_fig = f'{os.path.splitext(out_file)[0]}_{i}{i+1}.png'
                plot_stitch(
                    mat11, mat22,
                    mat, mat_diff,
                    out_fig=out_fig,
                    disp_scale=disp_scale,
                    disp_vlim=disp_vlim,
                    disp_cmap=disp_cmap,
                )

        dsDict[ds_name_out] = mat

    # write output file
    print('#'*50)
    writefile.write(dsDict, out_file=out_file, metadata=atr)
    
    # 保存拼接后的相干性文件
    if coherence_file and coherence_stitched is not None:
        coherence_out_file = out_file.replace('.h5', '_coherence.h5')
        print(f"Saving coherence file with shape: {coherence_stitched.shape}")
        print(f"Valid coherence pixels: {np.sum(~np.isnan(coherence_stitched))}")
        
        # 创建相干性文件的元数据
        # 使用原始相干性文件的元数据作为基础，而不是velocity文件的元数据
        if atr_coherence is not None:
            atr_coherence = atr_coherence.copy()
        else:
            # 如果没有原始相干性元数据，创建一个基本的
            atr_coherence = {}
            atr_coherence['FILE_TYPE'] = 'coherence'
            atr_coherence['UNIT'] = '1'
            atr_coherence['DATA_TYPE'] = 'float32'
        
        # 更新为拼接后的尺寸和位置
        atr_coherence['LENGTH'] = coherence_stitched.shape[0]
        atr_coherence['WIDTH'] = coherence_stitched.shape[1]
        atr_coherence['Y_FIRST'] = atr['Y_FIRST']  # 使用拼接后的位置
        atr_coherence['X_FIRST'] = atr['X_FIRST']
        atr_coherence['Y_STEP'] = atr['Y_STEP']    # 使用拼接后的步长
        atr_coherence['X_STEP'] = atr['X_STEP']
        
        # 确保文件类型正确
        atr_coherence['FILE_TYPE'] = 'coherence'
        atr_coherence['UNIT'] = '1'
        atr_coherence['DATA_TYPE'] = 'float32'
        
        # 创建相干性数据集字典
        coherence_dict = {'coherence': coherence_stitched}

        writefile.write(coherence_dict, out_file=coherence_out_file, metadata=atr_coherence)
        print(f'Saved stitched coherence file: {coherence_out_file}')

    # plot
    if disp_fig:
        print('showing ...')
        plt.show()
    else:
        plt.close()

    return out_file


def find_coherence_file(input_file, coherence_file_pattern):
    """
    在输入文件同目录下查找相干性文件
    
    Parameters:
    -----------
    input_file : str
        输入文件路径
    coherence_file_pattern : str
        相干性文件名模式（如 "temporalCoherence.h5"）
    
    Returns:
    --------
    str or None
        找到的相干性文件路径，如果没找到返回 None
    """
    import os
    
    # 获取输入文件所在目录
    input_dir = os.path.dirname(input_file)
    
    # 构建相干性文件路径
    coherence_file_path = os.path.join(input_dir, coherence_file_pattern)
    
    # 检查文件是否存在
    if os.path.exists(coherence_file_path):
        return coherence_file_path
    else:
        print(f"  Warning: Coherence file not found: {coherence_file_path}")
        return None


def stitch_two_matrices_with_coherence(mat1, atr1, mat2, atr2, 
                                     coherence1, coherence2,
                                     coherence_threshold=0.3, 
                                     coherence_method='median',
                                     print_msg=True):
    """
    使用相干性引导的拼接函数，支持重叠区域的相干性加权融合
    """
    vprint = print if print_msg else lambda *args, **kwargs: None
    
    # 分辨率统一
    ratio_x = abs((float(atr1['X_STEP']) - float(atr2['X_STEP'])) / float(atr1['X_STEP']))
    ratio_y = abs((float(atr1['Y_STEP']) - float(atr2['Y_STEP'])) / float(atr1['Y_STEP']))
    if any(i > 1e-3 for i in [ratio_x, ratio_y]):
        vprint('rescale the 2nd matrix into the same spatial resolution as the 1st one ...')
        vprint(f'Before rescale - mat2 shape: {mat2.shape}')
        vprint(f'Before rescale - coherence2 shape: {coherence2.shape if coherence2 is not None else "None"}')
        
        # 保存原始的 atr2 用于相干性 rescale
        atr2_original = atr2.copy()
        
        mat2, atr2 = rescale_data(mat2, meta=atr2, ref_meta=atr1)
        vprint(f'After rescale - mat2 shape: {mat2.shape}')
        
        if coherence2 is not None:
            # 使用原始的 atr2 来 rescale coherence2
            coherence2, _ = rescale_data(coherence2, meta=atr2_original, ref_meta=atr1)
            vprint(f'After rescale - coherence2 shape: {coherence2.shape}')
    
    # 计算输出范围
    vprint('grab corners of input matrices')
    S1, N1, W1, E1, width1, length1 = get_corners(atr1)
    S2, N2, W2, E2, width2, length2 = get_corners(atr2)
    
    # 修正：使用实际的矩阵形状，而不是属性中的值
    actual_length2, actual_width2 = mat2.shape
    vprint(f'Actual mat2 shape: {actual_length2} x {actual_width2}')
    vprint(f'Attribute mat2 shape: {length2} x {width2}')
    
    # 检查相干性形状
    if coherence2 is not None:
        actual_coh2_length, actual_coh2_width = coherence2.shape
        vprint(f'Actual coherence2 shape: {actual_coh2_length} x {actual_coh2_width}')
        
        # 如果相干性形状和矩阵形状不匹配，需要调整
        if actual_coh2_length != actual_length2 or actual_coh2_width != actual_width2:
            vprint('Warning: Coherence2 shape does not match mat2 shape, adjusting...')
            # 使用插值调整相干性矩阵大小
            from scipy.ndimage import zoom
            scale_y = actual_length2 / actual_coh2_length
            scale_x = actual_width2 / actual_coh2_width
            coherence2 = zoom(coherence2, (scale_y, scale_x), order=1)
            vprint(f'Adjusted coherence2 shape: {coherence2.shape}')
    
    # 输出空间范围
    vprint('calculate corners of output matrix')
    W, E = min(W1, W2), max(E1, E2)
    S, N = min(S1, S2), max(N1, N2)
    lon_step = float(atr1['X_STEP'])
    lat_step = float(atr1['Y_STEP'])
    width  = int(np.ceil((E - W) / lon_step))
    length = int(np.ceil((S - N) / lat_step))
    
    # 创建输出矩阵
    mat = np.zeros([length, width]) * np.nan
    mat1_placed = np.zeros([length, width]) * np.nan
    mat2_placed = np.zeros([length, width]) * np.nan
    
    # 计算索引
    lon_seq = np.arange(W, W + width  * lon_step, lon_step)
    lat_seq = np.arange(N, N + length * lat_step, lat_step)
    x1, y1 = np.argmin(np.square(lon_seq - W1)), np.argmin(np.square(lat_seq - N1))
    x2, y2 = np.argmin(np.square(lon_seq - W2)), np.argmin(np.square(lat_seq - N2))
    
    # 放置矩阵 - 使用实际的矩阵形状
    mat1_placed[y1:y1+length1, x1:x1+width1] = mat1
    mat2_placed[y2:y2+actual_length2, x2:x2+actual_width2] = mat2
    
    # 计算重叠区域
    overlap_mask = ~(np.isnan(mat1_placed) | np.isnan(mat2_placed))
    
    if np.any(overlap_mask):
        vprint(f'Found overlap area with {np.sum(overlap_mask)} pixels')
        
        # 计算重叠区域的相干性
        if coherence1 is not None and coherence2 is not None:
            # 创建相干性矩阵
            coh1_placed = np.zeros([length, width]) * np.nan
            coh2_placed = np.zeros([length, width]) * np.nan
            coh1_placed[y1:y1+length1, x1:x1+width1] = coherence1
            coh2_placed[y2:y2+actual_length2, x2:x2+actual_width2] = coherence2
            
            # 计算重叠区域的最小相干性
            overlap_coherence = np.minimum(coh1_placed, coh2_placed)
            
            # 选择高相干性像素
            high_coh_mask = (overlap_mask & 
                           (overlap_coherence >= coherence_threshold) & 
                           ~np.isnan(overlap_coherence))
            
            vprint(f'High coherence pixels (>{coherence_threshold}): {np.sum(high_coh_mask)}')
            
            if np.any(high_coh_mask):
                # 计算高相干性区域的偏移
                mat1_overlap = mat1_placed[high_coh_mask]
                mat2_overlap = mat2_placed[high_coh_mask]
                mat_diff = mat2_overlap - mat1_overlap
                
                # 使用指定方法计算偏移
                if coherence_method == 'median':
                    offset = np.nanmedian(mat_diff)
                else:  # mean
                    offset = np.nanmean(mat_diff)
                
                vprint(f'High coherence offset ({coherence_method}): {offset:.6f}')
                
                # 应用偏移
                mat2_placed = mat2_placed - offset
            else:
                vprint('Warning: No high coherence pixels found, using all overlap pixels')
                # 回退到使用所有重叠像素
                mat1_overlap = mat1_placed[overlap_mask]
                mat2_overlap = mat2_placed[overlap_mask]
                mat_diff = mat2_overlap - mat1_overlap
                
                if coherence_method == 'median':
                    offset = np.nanmedian(mat_diff)
                else:
                    offset = np.nanmean(mat_diff)
                
                vprint(f'All overlap offset ({coherence_method}): {offset:.6f}')
                mat2_placed = mat2_placed - offset
        else:
            vprint('No coherence files provided, using standard offset calculation')
            # 使用标准方法
            mat1_overlap = mat1_placed[overlap_mask]
            mat2_overlap = mat2_placed[overlap_mask]
            mat_diff = mat2_overlap - mat1_overlap
            
            if coherence_method == 'median':
                offset = np.nanmedian(mat_diff)
            else:
                offset = np.nanmean(mat_diff)
            
            vprint(f'Standard offset ({coherence_method}): {offset:.6f}')
            mat2_placed = mat2_placed - offset
    
    # 使用相干性加权融合进行拼接
    if coherence1 is not None and coherence2 is not None:
        vprint('Using coherence-weighted fusion for stitching')
        
        # 创建完整的相干性矩阵
        coh1_full = np.zeros([length, width]) * np.nan
        coh2_full = np.zeros([length, width]) * np.nan
        
        # 确保相干性矩阵和数据矩阵使用相同的索引
        # 使用实际的矩阵形状，而不是属性中的值
        actual_length1, actual_width1 = mat1.shape
        coh1_full[y1:y1+actual_length1, x1:x1+actual_width1] = coherence1
        coh2_full[y2:y2+actual_length2, x2:x2+actual_width2] = coherence2
        
        vprint(f'Coherence matrix shapes: coh1={coherence1.shape}, coh2={coherence2.shape}')
        vprint(f'Data matrix shapes: mat1={mat1.shape}, mat2={mat2.shape}')
        vprint(f'Placed matrix shapes: mat1_placed={mat1_placed.shape}, mat2_placed={mat2_placed.shape}')
        
        # 计算重叠区域
        overlap_mask = ~(np.isnan(mat1_placed) | np.isnan(mat2_placed))
        
        # 在重叠区域使用相干性加权融合
        if np.any(overlap_mask):
            vprint(f'Applying coherence-weighted fusion in overlap area with {np.sum(overlap_mask)} pixels')
            
            # 获取重叠区域的相干性
            coh1_overlap = coh1_full[overlap_mask]
            coh2_overlap = coh2_full[overlap_mask]
            mat1_overlap = mat1_placed[overlap_mask]
            mat2_overlap = mat2_placed[overlap_mask]
            
            # 检查相干性数据的有效性
            valid_coh_mask = ~(np.isnan(coh1_overlap) | np.isnan(coh2_overlap))
            vprint(f'Valid coherence pixels: {np.sum(valid_coh_mask)} out of {len(coh1_overlap)}')
            
            if np.any(valid_coh_mask):
                # 只对有效相干性像素进行加权融合
                coh1_valid = coh1_overlap[valid_coh_mask]
                coh2_valid = coh2_overlap[valid_coh_mask]
                mat1_valid = mat1_overlap[valid_coh_mask]
                mat2_valid = mat2_overlap[valid_coh_mask]
                
                # 避免除零错误，设置最小相干性阈值
                min_coherence = 1e-6
                coh1_valid = np.maximum(coh1_valid, min_coherence)
                coh2_valid = np.maximum(coh2_valid, min_coherence)
                
                # 计算加权融合
                total_coherence = coh1_valid + coh2_valid
                weight1 = coh1_valid / total_coherence
                weight2 = coh2_valid / total_coherence
                
                # 加权融合
                fused_valid = weight1 * mat1_valid + weight2 * mat2_valid
                
                # 将融合结果放回最终输出矩阵
                overlap_indices = np.where(overlap_mask)
                valid_overlap_indices = (overlap_indices[0][valid_coh_mask], 
                                       overlap_indices[1][valid_coh_mask])
                mat[valid_overlap_indices] = fused_valid
                
                # 对于无效相干性像素，使用简单平均
                invalid_coh_mask = ~valid_coh_mask
                if np.any(invalid_coh_mask):
                    vprint(f'Using simple average for {np.sum(invalid_coh_mask)} pixels with invalid coherence')
                    invalid_overlap_indices = (overlap_indices[0][invalid_coh_mask], 
                                             overlap_indices[1][invalid_coh_mask])
                    mat[invalid_overlap_indices] = (mat1_overlap[invalid_coh_mask] + 
                                                   mat2_overlap[invalid_coh_mask]) / 2
                
                vprint('Coherence-weighted fusion completed')
            else:
                vprint('Warning: No valid coherence data found, using simple average')
                # 回退到简单平均
                mat[overlap_mask] = (mat1_overlap + mat2_overlap) / 2
        else:
            vprint('Warning: No overlap area found')
        
        # 在非重叠区域，使用原始数据
        # 只处理非重叠区域，避免覆盖已经融合的重叠区域
        non_overlap_mask = ~overlap_mask
        mat = np.where(non_overlap_mask & ~np.isnan(mat1_placed), mat1_placed, mat)
        mat = np.where(non_overlap_mask & ~np.isnan(mat2_placed), mat2_placed, mat)
        
    else:
        vprint('No coherence files provided, using simple stitching')
        # 没有相干性文件时，使用简单拼接
        mat = np.where(~np.isnan(mat1_placed), mat1_placed, mat2_placed)
    
    # 更新属性
    atr = atr1.copy()
    atr['LENGTH'] = length
    atr['WIDTH'] = width
    atr['Y_FIRST'] = N
    atr['X_FIRST'] = W
    
    # 更新 REF_Y/X
    coord = ut.coordinate(atr)
    ref_y, ref_x = coord.geo2radar(float(atr['REF_LAT']), float(atr['REF_LON']))[:2]
    atr['REF_Y'], atr['REF_X'] = ref_y, ref_x
    
    # 删除 SUBSET_Y/XMIN/MAX
    for key in ['SUBSET_XMIN', 'SUBSET_XMAX', 'SUBSET_YMIN', 'SUBSET_YMAX']:
        if key in atr.keys():
            atr.pop(key)
    
    # 返回拼接后的相干性矩阵
    if coherence1 is not None and coherence2 is not None:
        # 创建拼接后的相干性矩阵
        coh_out = np.zeros([length, width]) * np.nan
        
        # 在非重叠区域，使用原始相干性数据
        non_overlap_mask = ~overlap_mask
        coh_out = np.where(non_overlap_mask & ~np.isnan(coh1_full), coh1_full, coh_out)
        coh_out = np.where(non_overlap_mask & ~np.isnan(coh2_full), coh2_full, coh_out)
        
        # 在重叠区域，使用最大相干性（或者可以选择其他策略）
        if np.any(overlap_mask):
            overlap_coherence = np.maximum(coh1_full, coh2_full)
            coh_out[overlap_mask] = overlap_coherence[overlap_mask]
        
        return mat, atr, coh_out
    else:
        return mat, atr, None


def stitch_coherence_simple(coh1, shape1, coh2, shape2, atr1, atr2):
    """
    简化的相干性拼接函数，直接基于矩阵形状进行拼接
    """
    # 计算输出形状
    out_length = max(shape1[0], shape2[0])
    out_width = max(shape1[1], shape2[1])
    
    # 创建输出矩阵
    coh_out = np.zeros([out_length, out_width]) * np.nan
    
    # 简单拼接：将两个相干性矩阵放在一起
    coh_out[:shape1[0], :shape1[1]] = coh1
    coh_out[:shape2[0], :shape2[1]] = coh2
    
    return coh_out


def stitch_coherence_proper(coh1, atr1, coh2, atr2, output_shape, output_atr):
    """
    正确的相干性拼接函数，考虑空间位置
    """
    length, width = output_shape
    coh_out = np.zeros([length, width]) * np.nan
    
    # 获取输入矩阵的角点
    S1, N1, W1, E1, width1, length1 = get_corners(atr1)
    S2, N2, W2, E2, width2, length2 = get_corners(atr2)
    
    # 计算索引
    W, E = min(W1, W2), max(E1, E2)
    S, N = min(S1, S2), max(N1, N2)
    lon_step = float(output_atr['X_STEP'])
    lat_step = float(output_atr['Y_STEP'])
    
    lon_seq = np.arange(W, W + width * lon_step, lon_step)
    lat_seq = np.arange(N, N + length * lat_step, lat_step)
    x1, y1 = np.argmin(np.square(lon_seq - W1)), np.argmin(np.square(lat_seq - N1))
    x2, y2 = np.argmin(np.square(lon_seq - W2)), np.argmin(np.square(lat_seq - N2))
    
    # 放置相干性矩阵
    coh_out[y1:y1+length1, x1:x1+width1] = coh1
    coh_out[y2:y2+length2, x2:x2+width2] = coh2
    
    return coh_out
