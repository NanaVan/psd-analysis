#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
from scipy import signal, ndimage
from scipy.ndimage import label

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
    energy_fine = np.max(np.abs(cwt_fine), axis=0)
    energy_wide = np.max(np.abs(cwt_wide), axis=0)
    
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
