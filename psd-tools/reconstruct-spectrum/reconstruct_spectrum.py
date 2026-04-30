#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from scipy import signal, ndimage
from scipy.ndimage import label, binary_dilation, gaussian_filter1d

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
