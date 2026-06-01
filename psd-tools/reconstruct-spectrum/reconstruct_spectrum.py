#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from scipy import signal, ndimage
from scipy.ndimage import label, binary_dilation, gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

try:
    from scipy.signal import cwt, ricker
except ImportError:
    # 针对 SciPy 1.15 + NumPy 2.x 的适配
    from scipy.signal._wavelets import _cwt as cwt
    from scipy.signal._wavelets import _ricker as ricker
    print("Detected SciPy 1.15+ / NumPy 2.x: Using internal wavelet API.")

def reconstruct_ion_spectrum(raw_data, baseline, k_high=4.0, k_low=1.2):
    """
    v2版本：多尺度特征融合与掩模生长重建
    k_high: 判定信号存在的严格阈值（防止噪声误触发）
    k_low:  判定信号边界的宽松阈值（保证峰形完整）
    """
    residual = raw_data - baseline
    
    # --- 优化点 1: 尺度细分 ---
    # 增加细尺度（2-8）捕获窄峰，大尺度（10-100）捕获主峰
    scales_fine = np.linspace(2, 10, 5)
    scales_wide = np.geomspace(10, 100, 15)
    
    cwt_fine = cwt(residual, ricker, scales_fine)
    cwt_wide = cwt(residual, ricker, scales_wide)
    
    # --- 优化点 2: 加权特征融合 ---
    # 窄峰能量强调瞬时突变，宽峰能量强调包络
    # 只保留正向卷积结果（信号向上），负向结果归零
    energy_fine = np.max(np.maximum(cwt_fine, 0), axis=0)
    energy_wide = np.max(np.maximum(cwt_wide, 0), axis=0)
    
    # 混合能量分布：让窄信号也能在 energy_dist 中抬起头
    energy_dist = 0.8 * energy_wide + 0.2 * energy_fine

    # 【关键修改】：使用双重中值滤波消除孤立的单点噪声突起
    # 这种方法在 NumPy 2.x 下比纯高斯平滑更能压制误判的小尖刺
    energy_dist = ndimage.median_filter(energy_dist, size=5) 
    energy_dist = ndimage.gaussian_filter1d(energy_dist, sigma=4) # 减小平滑，保留窄峰

    # --- 优化点 3: 稳健统计 ---
    # 使用百分位数来定义本底，避开主峰对均值/标准差的拉抬
    # 这样可以显著降低 k_high 也不至于引入毛刺
    base_level = np.percentile(energy_dist, 50) # 中位数作为基准
    noise_floor = np.percentile(energy_dist, 75) - base_level
    
    high_mask = energy_dist > (base_level + k_high * noise_floor)
    low_mask = energy_dist > (base_level + k_low * noise_floor)
    
    # 区域生长逻辑保持不变
    labels, n = label(low_mask)
    final_mask = np.zeros_like(low_mask, dtype=float)
    for i in range(1, n + 1):
        if np.any(high_mask[labels == i]):
            final_mask[labels == i] = 1.0
            
    # --- 优化点 4: 渐变带收缩 ---
    # 减小卷积核，防止掩模把附近的噪声也“吸”进来
    smooth_len = 100 # 从200减到100
    kernel = signal.windows.gaussian(smooth_len, std=smooth_len/6)
    final_mask = signal.convolve(final_mask, kernel / kernel.sum(), mode='same')
    
    # 重建逻辑
    denoised_raw = signal.savgol_filter(raw_data, 7, 2)
    reconstructed = (1 - final_mask) * baseline + final_mask * denoised_raw
    
    return reconstructed

def extract_peaks_moments_robust(f_arr, p_log, min_rel_height=0.05, dilation_size=25):
    # 1. 基础权重计算
    weights_raw = np.exp(p_log) - 1
    
    # 【关键改进】二次平滑：防止重构谱过于锐化导致的单点峰问题
    # sigma=1.0 的平滑可以在不改变峰位的前提下，给单点峰分配一点点“邻近权重”
    weights = gaussian_filter1d(weights_raw, sigma=1.0)
    
    max_val = np.max(weights)
    if max_val <= 0: return []

    # 2. 生成掩模 (使用平滑后的权重)
    initial_mask = weights > (max_val * 0.002)
    combined_mask = binary_dilation(initial_mask, structure=np.ones(dilation_size))
    
    labels, n_groups = label(combined_mask)
    
    peak_results = []
    for i in range(1, n_groups + 1):
        idx = (labels == i)
        wi = weights[idx]
        fi = f_arr[idx]
        
        island_max = np.max(wi)
        if island_max < (max_val * min_rel_height):
            continue
            
        sum_w = np.sum(wi)
        if sum_w <= 0: continue
        
        # 3. 矩估计
        mu = np.sum(fi * wi) / sum_w
        
        # 计算方差
        diff_sq = (fi - mu)**2
        var = np.sum(wi * diff_sq) / sum_w
        
        # 如果 var 还是太小，说明平滑力度不够，这里做一个保底
        if var <= 0:
            # 这种情况下，尝试使用原始频率分辨率作为最小宽度参考
            df = np.mean(np.diff(f_arr))
            sigma = df / 2.0  # 给予一个极小的默认物理宽度
        else:
            sigma = np.sqrt(var)

        # 有效样本量计算
        n_eff = sum_w / island_max
        # 限制 n_eff 最小为 1，防止除以 0
        n_eff = max(n_eff, 1.1) 
        
        err_pos = sigma / np.sqrt(n_eff)
        err_sigma = sigma / np.sqrt(2 * n_eff)
        
        peak_results.append({
            'peak_pos': mu,
            'err_pos': err_pos,
            'sigma': sigma,
            'err_sigma': err_sigma,
            'height_ratio': island_max
        })

    return sorted(peak_results, key=lambda x: x['peak_pos'])

def extract_peaks_log_detect(f_arr, p_log, p_arr_raw, p_time_interval, b_log, snr_factor=3.0):
    """
    在对数空间进行统计和种子点寻找，在线性空间进行物理量估计
    """
    # 1. 对数空间的信号 (SNR = psd_log - baseline_log)
    # 理想情况下，无信号区 log_diff 应该围绕 0 波动
    log_diff = p_log
    
    # 对 log_diff 进行轻微平滑，用于稳健地寻找峰尖
    log_diff_smooth = gaussian_filter1d(log_diff, sigma=1.0)
    
    # 2. 在对数空间计算背景统计量 (更稳健)
    # 使用中位数和 MAD 来定义底噪，弱峰在 log 空间会比在线性空间明显得多
    log_diff_notZero = log_diff[log_diff!=0]
    log_median = np.median(log_diff_notZero)
    log_mad = np.median(np.abs(log_diff_notZero - log_median))
    # 转换成类似 std 的尺度
    log_sigma = log_mad / 0.6745
    
    # 种子点阈值：log 空间的背景 + N倍波动
    log_threshold = log_median + snr_factor * log_sigma
    
    # 3. 在 Log 空间寻找种子点
    # distance 可根据你的频率分辨率微调
    seeds, _ = find_peaks(log_diff_smooth, height=log_threshold, distance=15)
    
    # 4. 回到线性空间准备权重
    weights_raw = np.exp(log_diff) - 1
    
    results = []
    used_indices = np.zeros_like(weights_raw, dtype=bool)
    
    for seed in seeds:
        if used_indices[seed]: continue
        
        # 5. 寻找物理边界
        # 策略：从种子点出发，寻找线性权重回落到接近 0 的地方
        # 使用你建议的 5 点滑动平均判断
        n = len(weights_raw)
        l_idx, r_idx = 0, n - 1
        
        # 向左寻边界 (寻找滑动平均跌落回 log 空间的噪声基准)
        # 这里用 exp(log_median)-1 作为线性空间的基准参考
        linear_baseline = np.exp(log_median) - 1
        
        window = 5
        for i in range(seed, window - 1, -1):
            if np.mean(weights_raw[i-window+1 : i+1]) <= linear_baseline:
                l_idx = i
                break
        
        for i in range(seed, n - window + 1):
            if np.mean(weights_raw[i : i+window]) <= linear_baseline:
                r_idx = i
                break
        
        # 标记已处理
        used_indices[l_idx : r_idx+1] = True
        
        # 6. 线性矩估计
        wi = np.maximum(weights_raw[l_idx : r_idx+1], 0)
        fi = f_arr[l_idx : r_idx+1]
        
        sum_w = np.sum(wi)
        if sum_w <= 0 or len(wi) < 2: continue
        
        mu = np.sum(fi * wi) / sum_w
        var = np.sum(wi * (fi - mu)**2) / sum_w
        sigma = np.sqrt(max(var, 0))
        
        if sigma == 0:
            sigma = np.mean(np.diff(f_arr)) / 2.0
            
        n_eff = sum_w / np.max(wi)
        n_eff = max(n_eff, 1.1)

        # 添加 height_valid 信息，在瀑布图中搜寻离子存在的区域平均到时间上，对于中途发现衰变或者中途产生的离子，height_ratio与height_valid不同
        l_jdx, r_jdx = np.searchsorted(f_arr, [mu-6*sigma, mu+6*sigma])
        fj = f_arr[l_jdx : r_jdx+1]
        pj_raw = np.sum(p_arr_raw[:, l_jdx : r_jdx], axis=-1)
        pj_cumsum = np.cumsum(pj_raw - np.mean(pj_raw))
        tdx_change = np.argmax(np.abs(np.diff(np.diff(pj_cumsum)))) + 1
        Zscore_change = np.max(np.abs(np.diff(np.diff(pj_cumsum)))) / np.std(np.diff(np.diff(pj_cumsum)))
        if Zscore_change < 3:
            height_ion = np.max(wi)
            exist_state, exist_timePoints = 0, p_arr_raw.shape[0]
        else:
            _temp_0, _temp_1 = np.exp(np.max(np.log(np.mean(p_arr_raw[:tdx_change, l_jdx : r_jdx], axis=0)) - b_log[l_jdx : r_jdx]))-1, np.exp(np.max(np.log(np.mean(p_arr_raw[tdx_change:, l_jdx : r_jdx], axis=0)) - b_log[l_jdx : r_jdx]))-1
            if _temp_0 > _temp_1 and (np.argmax(pj_cumsum) in [tdx_change-1, tdx_change, tdx_change+1]):
                height_ion = _temp_0
                exist_state, exist_timePoints = 1, tdx_change
            elif _temp_0 < _temp_1 and (np.argmin(pj_cumsum) in [tdx_change-1, tdx_change, tdx_change+1]):
                height_ion = _temp_1
                exist_state, exist_timePoints = 2, p_arr_raw.shape[0] - tdx_change
            else:
                height_ion = np.max(wi)
                exist_state, exist_timePoints = 0, p_arr_raw.shape[0]
        
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].plot(pj_raw)
        ax[0].axvline(x=tdx_change, color='tab:red', ls='dashed')
        if exist_state == 0:
            ax[0].set_title('No Change: {:}\nPSD(total) = {:.2f}, PSD(have ion) = {:.2f}'.format(Zscore_change, np.max(wi), height_ion))
        else:
            ax[0].set_title('Have Change: {:}\nPSD(total) = {:.2f}, PSD(have ion) = {:.2f},\nPSD(before) = {:.2f}, PSD(after) = {:.2f}'.format(Zscore_change, np.max(wi), height_ion, _temp_0, _temp_1))
        ax[1].plot(pj_cumsum)
        ax[1].axvline(x=tdx_change, color='tab:red', ls='dashed')

        results.append({
            'peak_pos': mu,
            'err_pos': sigma / np.sqrt(n_eff),
            'sigma': sigma,
            'err_sigma': sigma / np.sqrt(2 * n_eff),
            'height_ratio': np.max(wi),
            'height_ion': height_ion,
            'exist_state': exist_state, # 0: 一直存在; 1: 中途衰变; 2: 中途产生
            'exist_time': exist_timePoints * p_time_interval # 存在的时间
        })

    # 再检验中途产生情况
        
    return sorted(results, key=lambda x: x['peak_pos'])
